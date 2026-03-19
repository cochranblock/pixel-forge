// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Training loop for the tiny pixel art diffusion model.
//! Pure Rust — no Python, no external frameworks.
//! Approach inspired by PixelGen16x16 (MIT, Anouar Khaldi 2025).
//! All code written from scratch in Rust/Candle.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use image::RgbaImage;
use rand::seq::SliceRandom;
use std::path::Path;

use crate::tiny_unet::TinyUNet;

/// Training config.
pub struct TrainConfig {
    pub data_dir: String,
    pub output: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub img_size: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".into(),
            output: "pixel-forge-tiny.safetensors".into(),
            epochs: 100,
            batch_size: 64,
            lr: 1e-3,
            img_size: 16,
        }
    }
}

/// Load all PNG images from a directory.
/// Expects subdirectories named by class: data/character/*.png, data/weapon/*.png, etc.
/// Falls back to flat directory with class=14 (misc) if no subdirs.
fn load_dataset(data_dir: &str, img_size: u32, device: &Device) -> Result<(Tensor, Tensor)> {
    let class_names = [
        "character", "weapon", "potion", "terrain", "enemy",
        "tree", "building", "animal", "effect", "food",
        "armor", "tool", "vehicle", "ui", "misc",
    ];

    let mut all_images: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<u32> = Vec::new();
    let data_path = Path::new(data_dir);

    // Try subdirectory-per-class layout first
    let mut found_subdirs = false;
    for (class_id, name) in class_names.iter().enumerate() {
        let class_dir = data_path.join(name);
        if !class_dir.is_dir() {
            continue;
        }
        found_subdirs = true;
        let count = load_pngs_from_dir(&class_dir, img_size, class_id as u32, &mut all_images, &mut all_labels)?;
        if count > 0 {
            println!("  {name}: {count} images");
        }
    }

    // Flat directory fallback — everything is "misc"
    if !found_subdirs {
        let count = load_pngs_from_dir(data_path, img_size, 14, &mut all_images, &mut all_labels)?;
        println!("  flat dir: {count} images (class=misc)");
    }

    if all_images.is_empty() {
        anyhow::bail!("no images found in {data_dir}");
    }

    println!("total: {} images", all_images.len());

    let n = all_images.len();
    let flat: Vec<f32> = all_images.into_iter().flatten().collect();
    let images = Tensor::new(flat.as_slice(), device)?.reshape((n, 3, img_size as usize, img_size as usize))?;
    let labels = Tensor::new(all_labels.as_slice(), device)?;

    Ok((images, labels))
}

/// Load PNGs from a single directory, resize to target, normalize to [0,1].
fn load_pngs_from_dir(
    dir: &Path,
    img_size: u32,
    class_id: u32,
    images: &mut Vec<Vec<f32>>,
    labels: &mut Vec<u32>,
) -> Result<usize> {
    let mut count = 0;
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let img = image::open(entry.path())?;
        let img = img.resize_exact(img_size, img_size, image::imageops::FilterType::Triangle);
        let rgb = img.to_rgb8();

        // Normalize to [0, 1], channel-first layout (C, H, W)
        let mut pixels = vec![0.0f32; (3 * img_size * img_size) as usize];
        for y in 0..img_size {
            for x in 0..img_size {
                let p = rgb.get_pixel(x, y);
                let idx = (y * img_size + x) as usize;
                pixels[idx] = p[0] as f32 / 255.0;                           // R plane
                pixels[(img_size * img_size) as usize + idx] = p[1] as f32 / 255.0; // G plane
                pixels[2 * (img_size * img_size) as usize + idx] = p[2] as f32 / 255.0; // B plane
            }
        }

        images.push(pixels);
        labels.push(class_id);
        count += 1;
    }
    Ok(count)
}

/// Corrupt an image with noise: x_noisy = x * (1 - amount) + noise * amount
fn corrupt(x: &Tensor, amount: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
    let noise = Tensor::rand(0f32, 1f32, x.shape(), device)?;
    // amount shape: (B,) → (B, 1, 1, 1) for broadcasting
    let a = amount.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
    let one_minus_a = (1.0f64 - &a)?;
    let noisy = (x.broadcast_mul(&one_minus_a)? + noise.broadcast_mul(&a)?)?;
    Ok(noisy)
}

/// Train the tiny UNet from scratch on a dataset of PNGs.
pub fn train(config: &TrainConfig) -> Result<()> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32; // F32 for training stability

    println!("loading dataset from {}...", config.data_dir);
    let (images, labels) = load_dataset(&config.data_dir, config.img_size, &device)?;
    let n = images.dim(0)?;

    // Build model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = TinyUNet::new(vb)?;
    let params = TinyUNet::param_count(&varmap);
    println!("model: {} parameters ({:.1} MB)", params, params as f64 * 4.0 / 1_048_576.0);

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: config.lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    println!("training: {} epochs, batch_size={}, lr={}", config.epochs, config.batch_size, config.lr);

    for epoch in 0..config.epochs {
        // Shuffle indices using rand (Candle has no randperm)
        let mut indices: Vec<u32> = (0..n as u32).collect();
        indices.shuffle(&mut rand::thread_rng());
        let perm = Tensor::new(indices.as_slice(), &device)?;
        let images_shuffled = images.index_select(&perm, 0)?;
        let labels_shuffled = labels.index_select(&perm, 0)?;

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        let num_batches = (n + config.batch_size - 1) / config.batch_size;
        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;
            let end = (start + config.batch_size).min(n);
            let bs = end - start;

            let x_batch = images_shuffled.narrow(0, start, bs)?;
            let y_batch = labels_shuffled.narrow(0, start, bs)?;

            // Random noise amounts in [0, 1] per sample
            let noise_amount = Tensor::rand(0f32, 1f32, (bs,), &device)?;
            let noisy_x = corrupt(&x_batch, &noise_amount, &device)?;

            // Forward — model predicts clean image from noisy input
            let pred = model.forward(&noisy_x, &noise_amount, &y_batch)?;

            // MSE loss between prediction and clean image
            let loss = (&pred - &x_batch)?.sqr()?.mean_all()?;

            // Backward + step
            opt.backward_step(&loss)?;

            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count as f64;
        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            println!("  epoch {}/{}: loss={:.6}", epoch + 1, config.epochs, avg_loss);
        }
    }

    // Save weights
    println!("saving model to {}...", config.output);
    varmap.save(&config.output)?;

    let file_size = std::fs::metadata(&config.output)?.len();
    println!("done: {} ({:.1} MB)", config.output, file_size as f64 / 1_048_576.0);

    Ok(())
}

/// Sample from a trained model — iterative denoising.
pub fn sample(
    model_path: &str,
    class_id: u32,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    println!("loading model from {model_path}...");
    let mut varmap = VarMap::new();
    // Build model structure first so varmap has the right keys
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = TinyUNet::new(vb)?;
    // Load saved weights
    varmap.load(model_path)?;

    let params = TinyUNet::param_count(&varmap);
    println!("model: {} params, sampling {} images, {steps} steps", params, count);

    let mut images = Vec::new();
    let class_tensor = Tensor::new(&[class_id], &device)?;

    for i in 0..count {
        // Start from pure noise
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, img_size as usize, img_size as usize), &device)?;

        for step in 0..steps {
            let noise_level = 1.0 - (step as f32 / steps as f32);
            let t = Tensor::new(&[noise_level], &device)?;

            let pred = model.forward(&x, &t, &class_tensor)?;

            // Mix: move from noisy toward predicted clean
            let mix = 1.0 / (steps - step) as f64;
            let one_minus_mix = 1.0 - mix;
            x = ((&x * one_minus_mix)? + (&pred * mix)?)?;
        }

        // Clamp and convert to image
        let x = x.clamp(0.0, 1.0)?;
        let x = (x * 255.0)?.to_dtype(DType::U8)?;
        let x = x.squeeze(0)?; // (3, H, W)
        let x = x.permute((1, 2, 0))?; // (H, W, 3)
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
