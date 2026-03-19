// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Training loop for the tiny pixel art diffusion model.
//! Pure Rust — no Python, no external frameworks.
//! Approach inspired by PixelGen16x16 (MIT, Anouar Khaldi 2025).
//! All code written from scratch in Rust/Candle.
//!
//! Key design: loads batches from disk on-the-fly instead of one giant tensor.
//! Palette swap augmentation — same shape, different colors — teaches structure.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use image::RgbaImage;
use rand::seq::SliceRandom;
use rand::Rng;
use std::path::{Path, PathBuf};

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

/// A single training sample: file path + class label.
/// Images loaded on-the-fly per batch — no giant tensor in memory.
struct Sample {
    path: PathBuf,
    class_id: u32,
}

/// Build the sample index — just paths and labels, no image data yet.
fn build_index(data_dir: &str) -> Result<Vec<Sample>> {
    let class_names = [
        "character", "weapon", "potion", "terrain", "enemy",
        "tree", "building", "animal", "effect", "food",
        "armor", "tool", "vehicle", "ui", "misc",
    ];

    let mut samples = Vec::new();
    let data_path = Path::new(data_dir);
    let mut found_subdirs = false;

    for (class_id, name) in class_names.iter().enumerate() {
        let class_dir = data_path.join(name);
        if !class_dir.is_dir() {
            continue;
        }
        found_subdirs = true;
        let mut count = 0;
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
        for entry in entries {
            samples.push(Sample { path: entry.path(), class_id: class_id as u32 });
            count += 1;
        }
        if count > 0 {
            println!("  {name}: {count}");
        }
    }

    if !found_subdirs {
        let mut count = 0;
        for entry in std::fs::read_dir(data_path)?.filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|e| e.to_str()) == Some("png") {
                samples.push(Sample { path: entry.path(), class_id: 14 });
                count += 1;
            }
        }
        println!("  flat dir: {count} (class=misc)");
    }

    Ok(samples)
}

/// Load a single PNG into a normalized f32 vec (channel-first, [0,1]).
/// Returns None on failure — skip bad files instead of crashing.
fn load_image(path: &Path, img_size: u32) -> Option<Vec<f32>> {
    let img = image::open(path).ok()?;
    let img = img.resize_exact(img_size, img_size, image::imageops::FilterType::Nearest);
    let rgb = img.to_rgb8();

    let n = (img_size * img_size) as usize;
    let mut pixels = vec![0.0f32; 3 * n];
    for y in 0..img_size {
        for x in 0..img_size {
            let p = rgb.get_pixel(x, y);
            let idx = (y * img_size + x) as usize;
            pixels[idx] = p[0] as f32 / 255.0;
            pixels[n + idx] = p[1] as f32 / 255.0;
            pixels[2 * n + idx] = p[2] as f32 / 255.0;
        }
    }
    Some(pixels)
}

/// Palette swap augmentation — find unique colors, shuffle the mapping.
/// Same shape, different colors. Teaches structure over palette.
fn palette_swap(pixels: &mut Vec<f32>, img_size: u32) {
    let n = (img_size * img_size) as usize;
    let mut rng = rand::thread_rng();

    // Collect unique RGB triples
    let mut colors: Vec<[u8; 3]> = Vec::new();
    for i in 0..n {
        let r = (pixels[i] * 255.0) as u8;
        let g = (pixels[n + i] * 255.0) as u8;
        let b = (pixels[2 * n + i] * 255.0) as u8;
        let c = [r, g, b];
        if !colors.contains(&c) {
            colors.push(c);
        }
    }

    if colors.len() < 2 {
        return;
    }

    // Shuffle to create a mapping: old_color[i] → new_color[shuffled[i]]
    let mut shuffled = colors.clone();
    shuffled.shuffle(&mut rng);

    // Apply the swap
    for i in 0..n {
        let r = (pixels[i] * 255.0) as u8;
        let g = (pixels[n + i] * 255.0) as u8;
        let b = (pixels[2 * n + i] * 255.0) as u8;
        let c = [r, g, b];
        if let Some(idx) = colors.iter().position(|x| *x == c) {
            let new_c = shuffled[idx];
            pixels[i] = new_c[0] as f32 / 255.0;
            pixels[n + i] = new_c[1] as f32 / 255.0;
            pixels[2 * n + i] = new_c[2] as f32 / 255.0;
        }
    }
}

/// Load a batch of samples into tensors. Skips bad files (recovery).
/// Optionally applies palette swap augmentation.
fn load_batch(
    samples: &[Sample],
    img_size: u32,
    augment: bool,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let mut batch_pixels: Vec<f32> = Vec::new();
    let mut batch_labels: Vec<u32> = Vec::new();
    let mut rng = rand::thread_rng();

    for sample in samples {
        let mut pixels = match load_image(&sample.path, img_size) {
            Some(p) => p,
            None => continue, // Skip bad file — recovery
        };

        // 50% chance palette swap augmentation
        if augment && rng.gen_bool(0.5) {
            palette_swap(&mut pixels, img_size);
        }

        // 50% chance horizontal flip
        if augment && rng.gen_bool(0.5) {
            let n = (img_size * img_size) as usize;
            for c in 0..3 {
                for y in 0..img_size as usize {
                    let row_start = c * n + y * img_size as usize;
                    let row_end = row_start + img_size as usize;
                    pixels[row_start..row_end].reverse();
                }
            }
        }

        batch_pixels.extend_from_slice(&pixels);
        batch_labels.push(sample.class_id);
    }

    if batch_labels.is_empty() {
        return Ok(None);
    }

    let bs = batch_labels.len();
    let images = Tensor::new(batch_pixels.as_slice(), device)?
        .reshape((bs, 3, img_size as usize, img_size as usize))?;
    let labels = Tensor::new(batch_labels.as_slice(), device)?;

    Ok(Some((images, labels)))
}

/// Corrupt an image with noise: x_noisy = x * (1 - amount) + noise * amount
fn corrupt(x: &Tensor, amount: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
    let noise = Tensor::rand(0f32, 1f32, x.shape(), device)?;
    let a = amount.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
    let one_minus_a = (1.0f64 - &a)?;
    let noisy = (x.broadcast_mul(&one_minus_a)? + noise.broadcast_mul(&a)?)?;
    Ok(noisy)
}

/// Train the tiny UNet from scratch on a dataset of PNGs.
/// Loads batches from disk on-the-fly — constant memory usage regardless of dataset size.
pub fn train(config: &TrainConfig) -> Result<()> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    println!("indexing dataset from {}...", config.data_dir);
    let samples = build_index(&config.data_dir)?;
    let n = samples.len();
    if n == 0 {
        anyhow::bail!("no images found in {}", config.data_dir);
    }
    println!("total: {} images (loaded per-batch, not all at once)", n);

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
    println!("augmentation: palette swap (50%) + h-flip (50%)");

    let t0 = std::time::Instant::now();

    for epoch in 0..config.epochs {
        let epoch_start = std::time::Instant::now();

        // Shuffle sample indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut skipped = 0;

        let num_batches = (n + config.batch_size - 1) / config.batch_size;
        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;
            let end = (start + config.batch_size).min(n);

            // Gather samples for this batch
            let batch_samples: Vec<&Sample> = indices[start..end]
                .iter()
                .map(|&i| &samples[i])
                .collect();

            // Load from disk on-the-fly
            let (images, labels) = match load_batch(
                &batch_samples.iter().map(|s| Sample { path: s.path.clone(), class_id: s.class_id }).collect::<Vec<_>>(),
                config.img_size,
                true, // augment
                &device,
            )? {
                Some(batch) => batch,
                None => { skipped += 1; continue; }
            };

            let bs = images.dim(0)?;

            // Random noise amounts in [0, 1] per sample
            let noise_amount = Tensor::rand(0f32, 1f32, (bs,), &device)?;
            let noisy_x = corrupt(&images, &noise_amount, &device)?;

            // Forward — model predicts clean image from noisy input
            let pred = model.forward(&noisy_x, &noise_amount, &labels)?;

            // MSE loss between prediction and clean image
            let loss = (&pred - &images)?.sqr()?.mean_all()?;

            // Backward + step
            opt.backward_step(&loss)?;

            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
        let elapsed = epoch_start.elapsed().as_secs_f32();

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            println!(
                "  epoch {}/{}: loss={:.6} ({:.1}s, {} batches{})",
                epoch + 1, config.epochs, avg_loss, elapsed, batch_count,
                if skipped > 0 { format!(", {} skipped", skipped) } else { String::new() }
            );
        }

        // Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0 {
            let checkpoint = format!("{}.epoch{}", config.output, epoch + 1);
            varmap.save(&checkpoint)?;
            println!("  checkpoint: {checkpoint}");
        }
    }

    // Save final weights
    println!("saving model to {}...", config.output);
    varmap.save(&config.output)?;

    let file_size = std::fs::metadata(&config.output)?.len();
    let total_time = t0.elapsed().as_secs_f32();
    println!("done: {} ({:.1} MB) in {:.0}s", config.output, file_size as f64 / 1_048_576.0, total_time);

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
            let one_minus_mix = 1.0 - mix;
            x = ((&x * one_minus_mix)? + (&pred * mix)?)?;
        }

        let x = x.clamp(0.0, 1.0)?;
        let x = (x * 255.0)?.to_dtype(DType::U8)?;
        let x = x.squeeze(0)?;
        let x = x.permute((1, 2, 0))?;
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
