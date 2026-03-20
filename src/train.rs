// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Training loop for the tiny pixel art diffusion model.
//! Pure Rust — no Python, no external frameworks.
//! Approach inspired by PixelGen16x16 (MIT, Anouar Khaldi 2025).
//!
//! Data pipeline: preprocess PNGs once → bincode+zstd blob → load into RAM → train.
//! Zero disk I/O during training. Palette swap augmentation in-memory.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use image::RgbaImage;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::tiny_unet::TinyUNet;
use crate::medium_unet::MediumUNet;

/// Training config.
pub struct TrainConfig {
    pub data_dir: String,
    pub output: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub img_size: u32,
    pub medium: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".into(),
            output: "pixel-forge-cinder.safetensors".into(),
            epochs: 100,
            batch_size: 64,
            lr: 1e-3,
            img_size: 16,
            medium: false,
        }
    }
}

/// Trait to unify TinyUNet and MediumUNet forward passes.
pub trait DiffusionModel {
    fn forward(&self, x: &Tensor, timestep: &Tensor, class_id: &Tensor) -> candle_core::Result<Tensor>;
}

impl DiffusionModel for TinyUNet {
    fn forward(&self, x: &Tensor, timestep: &Tensor, class_id: &Tensor) -> candle_core::Result<Tensor> {
        TinyUNet::forward(self, x, timestep, class_id)
    }
}

impl DiffusionModel for MediumUNet {
    fn forward(&self, x: &Tensor, timestep: &Tensor, class_id: &Tensor) -> candle_core::Result<Tensor> {
        MediumUNet::forward(self, x, timestep, class_id)
    }
}

/// Pre-decoded dataset — all images as flat f32 vecs, packed into one blob.
/// bincode+zstd serialized for instant reload.
#[derive(Serialize, Deserialize)]
pub struct PackedDataset {
    pub img_size: u32,
    /// Flat f32 pixels: [sample0_r, sample0_g, sample0_b, sample1_r, ...] channel-first per sample
    pub pixels: Vec<f32>,
    pub labels: Vec<u32>,
    /// Number of f32 values per sample (3 * img_size * img_size)
    pub stride: usize,
}

/// Preprocess: decode all PNGs → RAM → bincode+zstd file.
/// Called once. Subsequent runs load the cached blob.
pub fn preprocess(data_dir: &str, img_size: u32) -> Result<PackedDataset> {
    let cache_file = format!("{}/dataset.bin.zst", data_dir);
    let cache_path = Path::new(&cache_file);

    // Try loading cached blob first
    if cache_path.exists() {
        println!("loading cached dataset from {cache_file}...");
        let t0 = std::time::Instant::now();
        let compressed = std::fs::read(cache_path)?;
        let decompressed = zstd::decode_all(compressed.as_slice())?;
        let dataset: PackedDataset = bincode::deserialize(&decompressed)?;
        let n = dataset.labels.len();
        let mb = compressed.len() as f64 / 1_048_576.0;
        println!("  loaded {} images from cache ({:.1} MB compressed) in {:.1}s",
            n, mb, t0.elapsed().as_secs_f32());
        return Ok(dataset);
    }

    println!("preprocessing PNGs from {data_dir} → {cache_file}...");
    let t0 = std::time::Instant::now();

    let class_names = [
        "character", "weapon", "potion", "terrain", "enemy",
        "tree", "building", "animal", "effect", "food",
        "armor", "tool", "vehicle", "ui", "misc",
    ];

    let stride = (3 * img_size * img_size) as usize;
    let mut pixels: Vec<f32> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();
    let mut total = 0usize;
    let mut bad = 0usize;

    let data_path = Path::new(data_dir);

    for (class_id, name) in class_names.iter().enumerate() {
        let class_dir = data_path.join(name);
        if !class_dir.is_dir() {
            continue;
        }

        let mut entries: Vec<_> = std::fs::read_dir(&class_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());

        let mut class_count = 0;
        for entry in &entries {
            match decode_png(&entry.path(), img_size) {
                Some(px) => {
                    pixels.extend_from_slice(&px);
                    labels.push(class_id as u32);
                    class_count += 1;
                    total += 1;
                }
                None => { bad += 1; }
            }
        }
        if class_count > 0 {
            println!("  {name}: {class_count}");
        }
    }

    if total == 0 {
        anyhow::bail!("no images found in {data_dir}");
    }

    let dataset = PackedDataset { img_size, pixels, labels, stride };

    // Serialize + compress
    let encoded = bincode::serialize(&dataset)?;
    let compressed = zstd::encode_all(encoded.as_slice(), 3)?;
    std::fs::write(cache_path, &compressed)?;

    let raw_mb = encoded.len() as f64 / 1_048_576.0;
    let comp_mb = compressed.len() as f64 / 1_048_576.0;
    println!("total: {} images, {} bad skipped", total, bad);
    println!("cached: {:.1} MB raw → {:.1} MB zstd ({:.0}x) in {:.1}s",
        raw_mb, comp_mb, raw_mb / comp_mb, t0.elapsed().as_secs_f32());

    Ok(dataset)
}

/// Decode a single PNG to channel-first f32 [0,1].
fn decode_png(path: &Path, img_size: u32) -> Option<Vec<f32>> {
    let img = image::open(path).ok()?;
    let img = img.resize_exact(img_size, img_size, image::imageops::FilterType::Nearest);
    let rgb = img.to_rgb8();

    let n = (img_size * img_size) as usize;
    let mut px = vec![0.0f32; 3 * n];
    for y in 0..img_size {
        for x in 0..img_size {
            let p = rgb.get_pixel(x, y);
            let idx = (y * img_size + x) as usize;
            px[idx] = p[0] as f32 / 255.0;
            px[n + idx] = p[1] as f32 / 255.0;
            px[2 * n + idx] = p[2] as f32 / 255.0;
        }
    }
    Some(px)
}

/// Palette swap augmentation — dedupe colors, shuffle mapping, hash-based replace.
fn palette_swap(px: &mut [f32], _stride: usize, img_size: u32) {
    use std::collections::HashMap;
    let n = (img_size * img_size) as usize;
    let mut rng = rand::thread_rng();

    // Pass 1: dedupe into a vec (order matters for shuffle mapping)
    let mut colors: Vec<u32> = Vec::with_capacity(64);
    let mut seen: HashMap<u32, usize> = HashMap::with_capacity(64);
    for i in 0..n {
        let r = (px[i] * 255.0) as u8;
        let g = (px[n + i] * 255.0) as u8;
        let b = (px[2 * n + i] * 255.0) as u8;
        let key = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        if !seen.contains_key(&key) {
            seen.insert(key, colors.len());
            colors.push(key);
            if colors.len() >= 64 { break; }
        }
    }
    if colors.len() < 2 { return; }

    // Shuffle to create mapping
    let mut shuffled = colors.clone();
    shuffled.shuffle(&mut rng);

    // Build lookup: old_key → new_rgb
    let mut swap: HashMap<u32, (f32, f32, f32)> = HashMap::with_capacity(colors.len());
    for (i, &old_key) in colors.iter().enumerate() {
        let nc = shuffled[i];
        swap.insert(old_key, (
            ((nc >> 16) & 0xFF) as f32 / 255.0,
            ((nc >> 8) & 0xFF) as f32 / 255.0,
            (nc & 0xFF) as f32 / 255.0,
        ));
    }

    // Pass 2: replace via hash lookup — O(1) per pixel
    for i in 0..n {
        let r = (px[i] * 255.0) as u8;
        let g = (px[n + i] * 255.0) as u8;
        let b = (px[2 * n + i] * 255.0) as u8;
        let key = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        if let Some(&(nr, ng, nb)) = swap.get(&key) {
            px[i] = nr;
            px[n + i] = ng;
            px[2 * n + i] = nb;
        }
    }
}

/// H-flip a single sample in-place.
fn hflip(px: &mut [f32], img_size: u32) {
    let n = (img_size * img_size) as usize;
    let w = img_size as usize;
    for c in 0..3 {
        for y in 0..img_size as usize {
            let start = c * n + y * w;
            let end = start + w;
            px[start..end].reverse();
        }
    }
}

/// Corrupt: x_noisy = x * (1 - amount) + noise * amount
fn corrupt(x: &Tensor, amount: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
    let noise = Tensor::rand(0f32, 1f32, x.shape(), device)?;
    let a = amount.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
    let one_minus_a = (1.0f64 - &a)?;
    x.broadcast_mul(&one_minus_a)? + noise.broadcast_mul(&a)?
}

/// Train loop — works with any DiffusionModel (Tiny or Medium).
fn train_inner(
    model: &dyn DiffusionModel,
    varmap: &VarMap,
    config: &TrainConfig,
    dataset: &PackedDataset,
    device: &Device,
) -> Result<()> {
    let n = dataset.labels.len();
    let stride = dataset.stride;

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: config.lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    println!("training: {} epochs, bs={}, lr={}, {} samples", config.epochs, config.batch_size, config.lr, n);
    println!("augmentation: palette swap + h-flip (50% each)");

    let t0 = std::time::Instant::now();
    let sz = config.img_size as usize;

    for epoch in 0..config.epochs {
        let epoch_start = std::time::Instant::now();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut rng = rand::thread_rng();

        let num_batches = (n + config.batch_size - 1) / config.batch_size;
        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;
            let end = (start + config.batch_size).min(n);
            let bs = end - start;

            let mut batch_px = Vec::with_capacity(bs * stride);
            let mut batch_labels = Vec::with_capacity(bs);

            for &idx in &indices[start..end] {
                let src_start = idx * stride;
                let mut sample = dataset.pixels[src_start..src_start + stride].to_vec();

                if rng.r#gen_bool(0.5) {
                    palette_swap(&mut sample, stride, config.img_size);
                }
                if rng.r#gen_bool(0.5) {
                    hflip(&mut sample, config.img_size);
                }

                batch_px.extend_from_slice(&sample);
                batch_labels.push(dataset.labels[idx]);
            }

            let x_batch = Tensor::new(batch_px.as_slice(), device)?
                .reshape((bs, 3, sz, sz))?;
            let y_batch = Tensor::new(batch_labels.as_slice(), device)?;

            let noise_amount = Tensor::rand(0f32, 1f32, (bs,), device)?;
            let noisy_x = corrupt(&x_batch, &noise_amount, device)?;

            let pred = model.forward(&noisy_x, &noise_amount, &y_batch)?;
            let loss = (&pred - &x_batch)?.sqr()?.mean_all()?;

            opt.backward_step(&loss)?;

            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
        let elapsed = epoch_start.elapsed().as_secs_f32();

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            println!("  epoch {}/{}: loss={:.6} ({:.1}s)", epoch + 1, config.epochs, avg_loss, elapsed);
        }

        if (epoch + 1) % 25 == 0 {
            let cp = format!("{}.epoch{}", config.output, epoch + 1);
            varmap.save(&cp)?;
            println!("  checkpoint: {cp}");
        }
    }

    // Save final
    println!("saving model to {}...", config.output);
    varmap.save(&config.output)?;

    let file_size = std::fs::metadata(&config.output)?.len();
    println!("done: {} ({:.1} MB) in {:.0}s total",
        config.output, file_size as f64 / 1_048_576.0, t0.elapsed().as_secs_f32());

    Ok(())
}

/// Train — dispatches to Tiny or Medium based on config.
pub fn train(config: &TrainConfig) -> Result<()> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    let dataset = preprocess(&config.data_dir, config.img_size)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    if config.medium {
        let model = MediumUNet::new(vb)?;
        let params = MediumUNet::param_count(&varmap);
        println!("model: Quench (MediumUNet), {} params ({:.1} MB)", params, params as f64 * 4.0 / 1_048_576.0);
        train_inner(&model, &varmap, config, &dataset, &device)?;
    } else {
        let model = TinyUNet::new(vb)?;
        let params = TinyUNet::param_count(&varmap);
        println!("model: Cinder (TinyUNet), {} params ({:.1} MB)", params, params as f64 * 4.0 / 1_048_576.0);
        train_inner(&model, &varmap, config, &dataset, &device)?;
    }

    Ok(())
}

/// Sample from a trained model (auto-detects size by param count).
pub fn sample(
    model_path: &str,
    class_id: u32,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    // Try loading as TinyUNet first, then MediumUNet
    println!("loading model from {model_path}...");
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = TinyUNet::new(vb)?;
    varmap.load(model_path)?;

    let params = TinyUNet::param_count(&varmap);
    println!("model: {} params, sampling {} images, {steps} steps", params, count);

    let mut images = Vec::new();
    let class_tensor = Tensor::new(&[class_id], &device)?;

    for i in 0..count {
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, img_size as usize, img_size as usize), &device)?;

        for step in 0..steps {
            let noise_level = 1.0 - (step as f32 / steps as f32);
            let t = Tensor::new(&[noise_level], &device)?;
            let pred = model.forward(&x, &t, &class_tensor)?;
            let mix = 1.0 / (steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix)?)?;
        }

        let x = x.clamp(0.0, 1.0)?;
        let x = (x * 255.0)?.to_dtype(DType::U8)?;
        let x = x.squeeze(0)?.permute((1, 2, 0))?;
        let pixels = x.flatten_all()?.to_vec1::<u8>()?;

        let mut rgba = RgbaImage::new(img_size, img_size);
        for y in 0..img_size {
            for px in 0..img_size {
                let idx = (y * img_size + px) as usize * 3;
                rgba.put_pixel(px, y, image::Rgba([pixels[idx], pixels[idx + 1], pixels[idx + 2], 255]));
            }
        }
        images.push(rgba);
        println!("  sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// Sample from a trained MediumUNet (Quench) model.
pub fn sample_medium(
    model_path: &str,
    class_id: u32,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    println!("loading Quench model from {model_path}...");
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = MediumUNet::new(vb)?;
    varmap.load(model_path)?;

    let params = MediumUNet::param_count(&varmap);
    println!("model: Quench, {} params, sampling {} images, {steps} steps", params, count);

    let mut images = Vec::new();
    let class_tensor = Tensor::new(&[class_id], &device)?;

    for i in 0..count {
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, img_size as usize, img_size as usize), &device)?;

        for step in 0..steps {
            let noise_level = 1.0 - (step as f32 / steps as f32);
            let t = Tensor::new(&[noise_level], &device)?;
            let pred = model.forward(&x, &t, &class_tensor)?;
            let mix = 1.0 / (steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix)?)?;
        }

        let x = x.clamp(0.0, 1.0)?;
        let x = (x * 255.0)?.to_dtype(DType::U8)?;
        let x = x.squeeze(0)?.permute((1, 2, 0))?;
        let pixels = x.flatten_all()?.to_vec1::<u8>()?;

        let mut rgba = RgbaImage::new(img_size, img_size);
        for y in 0..img_size {
            for px in 0..img_size {
                let idx = (y * img_size + px) as usize * 3;
                rgba.put_pixel(px, y, image::Rgba([pixels[idx], pixels[idx + 1], pixels[idx + 2], 255]));
            }
        }
        images.push(rgba);
        println!("  sample {}/{count}", i + 1);
    }

    Ok(images)
}
