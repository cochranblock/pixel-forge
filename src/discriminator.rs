// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel art quality discriminator — learns to distinguish real pixel art
//! from generated output. Tiny CNN (~30K params, <200KB).
//! Trains on the curated 52K dataset (positive) vs generated rejects (negative).
//! Used as a quality gate: reject sprites below threshold, re-roll.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use image::RgbaImage;
use std::path::Path;

/// Discriminator architecture: 32x32x3 → score ∈ [0,1]
/// 1.0 = looks like real pixel art, 0.0 = looks like AI output
pub struct Discriminator {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    conv3: nn::Conv2d,
    conv4: nn::Conv2d,
    gn1: nn::GroupNorm,
    gn2: nn::GroupNorm,
    gn3: nn::GroupNorm,
    gn4: nn::GroupNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Discriminator {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // 32x32x3 → 32x32x32 → 16x16x64 → 8x8x64 → 4x4x64 → FC128 → FC1
        let conv1 = nn::conv2d(3, 32, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c1"))?;
        let conv2 = nn::conv2d(32, 64, 3, nn::Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("c2"))?;
        let conv3 = nn::conv2d(64, 64, 3, nn::Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("c3"))?;
        let conv4 = nn::conv2d(64, 64, 3, nn::Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("c4"))?;
        let gn1 = nn::group_norm(8, 32, 1e-5, vb.pp("gn1"))?;
        let gn2 = nn::group_norm(8, 64, 1e-5, vb.pp("gn2"))?;
        let gn3 = nn::group_norm(8, 64, 1e-5, vb.pp("gn3"))?;
        let gn4 = nn::group_norm(8, 64, 1e-5, vb.pp("gn4"))?;
        // 4x4x64 = 1024
        let fc1 = nn::linear(1024, 128, vb.pp("fc1"))?;
        let fc2 = nn::linear(128, 1, vb.pp("fc2"))?;
        Ok(Self { conv1, conv2, conv3, conv4, gn1, gn2, gn3, gn4, fc1, fc2 })
    }

    /// Score a batch. Returns (B,1) logits — apply sigmoid for probability.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?;
        let h = self.gn1.forward(&h)?;
        let h = nn::ops::silu(&h)?;

        let h = self.conv2.forward(&h)?;
        let h = self.gn2.forward(&h)?;
        let h = nn::ops::silu(&h)?;

        let h = self.conv3.forward(&h)?;
        let h = self.gn3.forward(&h)?;
        let h = nn::ops::silu(&h)?;

        let h = self.conv4.forward(&h)?;
        let h = self.gn4.forward(&h)?;
        let h = nn::ops::silu(&h)?;

        let (b, _, _, _) = h.dims4()?;
        let h = h.reshape((b, 1024))?;
        let h = nn::ops::silu(&self.fc1.forward(&h)?)?;
        self.fc2.forward(&h)
    }

    /// Score a single sprite. Returns 0.0-1.0 quality score.
    pub fn score(&self, x: &Tensor) -> Result<f32> {
        let logit = self.forward(x)?;
        let prob = candle_nn::ops::sigmoid(&logit)?;
        let val = prob.flatten_all()?.to_vec1::<f32>()?;
        Ok(val[0])
    }

    pub fn param_count(varmap: &VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}

/// Training config.
pub struct DiscriminatorTrainConfig {
    pub real_dir: String,       // curated pixel art (positive)
    pub fake_dir: String,       // generated rejects (negative)
    pub output: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
}

/// Load images from a directory into a tensor (N, 3, 32, 32).
fn load_images(dir: &str, device: &Device, max: usize) -> anyhow::Result<Tensor> {
    let mut pixels = Vec::new();
    let mut count = 0;

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if count >= max { break; }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") { continue; }

        let img = image::open(path)?.to_rgb8();
        if img.width() != 32 || img.height() != 32 { continue; }

        // Channel-first, normalized to [0,1]
        for c in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    pixels.push(img.get_pixel(x, y)[c] as f32 / 255.0);
                }
            }
        }
        count += 1;
    }

    if count == 0 {
        anyhow::bail!("no 32x32 PNG images found in {dir}");
    }

    let tensor = Tensor::from_vec(pixels, (count, 3, 32, 32), device)?;
    Ok(tensor)
}

/// Generate negative samples by running the current model and collecting output.
/// If no fake_dir exists, generate them on the fly from a model.
fn generate_negatives(model_path: &str, count: usize, device: &Device) -> anyhow::Result<Tensor> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    // Try loading as TinyUNet (Cinder) — cheapest
    let model = crate::tiny_unet::TinyUNet::new(vb)?;
    varmap.load(model_path)?;

    let mut all_pixels = Vec::new();
    let super_t = Tensor::new(&[0u32], device)?;
    let tags_t = Tensor::zeros((1, crate::class_cond::NUM_TAGS), DType::F32, device)?;

    for _ in 0..count {
        // Generate with few steps — intentionally low quality
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, 16, 16), device)?;
        let steps = 10; // few steps = worse output = good negatives
        for step in 0..steps {
            let noise_level = 1.0 - (step as f32 / steps as f32);
            let t = Tensor::new(&[noise_level], device)?;
            let pred = model.forward(&x, &t, &super_t, &tags_t)?;
            let mix = 1.0 / (steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix)?)?;
        }
        let x = x.clamp(0.0, 1.0)?;
        let flat = x.flatten_all()?.to_vec1::<f32>()?;
        all_pixels.extend_from_slice(&flat);
    }

    let tensor = Tensor::from_vec(all_pixels, (count, 3, 16, 16), device)?;
    Ok(tensor)
}

/// Train the discriminator.
pub fn train(config: &DiscriminatorTrainConfig) -> anyhow::Result<()> {
    let device = crate::pipeline::best_device();

    println!("loading real sprites from {}...", config.real_dir);
    let real = load_images(&config.real_dir, &device, 10_000)?;
    let n_real = real.dim(0)?;
    println!("  loaded {} real sprites", n_real);

    let fake = if Path::new(&config.fake_dir).is_dir() {
        println!("loading fake sprites from {}...", config.fake_dir);
        let f = load_images(&config.fake_dir, &device, n_real)?;
        println!("  loaded {} fake sprites", f.dim(0)?);
        f
    } else {
        // Auto-generate negatives from Cinder model
        println!("generating {} negative samples from Cinder...", n_real);
        let cinder_path = crate::device_cap::Tier::Cinder.model_file();
        if !Path::new(cinder_path).exists() {
            anyhow::bail!("need Cinder model ({cinder_path}) to generate negatives");
        }
        generate_negatives(cinder_path, n_real, &device)?
    };
    let n_fake = fake.dim(0)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Discriminator::new(vb)?;
    let params = Discriminator::param_count(&varmap);
    println!("discriminator: {} params ({:.1} KB)", params, params as f64 * 4.0 / 1024.0);

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: config.lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    // Labels: real=1, fake=0
    let real_labels = Tensor::ones((n_real, 1), DType::F32, &device)?;
    let fake_labels = Tensor::zeros((n_fake, 1), DType::F32, &device)?;

    for epoch in 1..=config.epochs {
        let t0 = std::time::Instant::now();

        // Shuffle by random index
        let perm_r = Tensor::rand(0f32, 1f32, (n_real,), &device)?
            .arg_sort_last_dim(false)?;
        let perm_f = Tensor::rand(0f32, 1f32, (n_fake,), &device)?
            .arg_sort_last_dim(false)?;

        let real_shuffled = real.index_select(&perm_r, 0)?;
        let fake_shuffled = fake.index_select(&perm_f, 0)?;
        let labels_r = real_labels.index_select(&perm_r, 0)?;
        let labels_f = fake_labels.index_select(&perm_f, 0)?;

        let mut epoch_loss = 0.0;
        let mut batches = 0;

        let total = n_real.min(n_fake);
        let bs = config.batch_size;

        for i in (0..total).step_by(bs) {
            let end = (i + bs).min(total);
            let batch_r = real_shuffled.narrow(0, i, end - i)?;
            let batch_f = fake_shuffled.narrow(0, i, end - i)?;
            let lab_r = labels_r.narrow(0, i, end - i)?;
            let lab_f = labels_f.narrow(0, i, end - i)?;

            // Concat real + fake
            let batch = Tensor::cat(&[&batch_r, &batch_f], 0)?;
            let labels = Tensor::cat(&[&lab_r, &lab_f], 0)?;

            let logits = model.forward(&batch)?;
            let loss = candle_nn::loss::binary_cross_entropy_with_logit(&logits, &labels)?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_vec0::<f32>()? as f64;
            batches += 1;
        }

        let avg_loss = epoch_loss / batches as f64;
        let elapsed = t0.elapsed().as_secs_f64();

        if epoch % 5 == 0 || epoch == 1 {
            println!("  epoch {}/{}: loss={:.6} ({:.1}s)", epoch, config.epochs, avg_loss, elapsed);
        }
    }

    // Save
    varmap.save(&config.output)?;
    let file_size = std::fs::metadata(&config.output)?.len();
    println!("saved: {} ({:.1} KB)", config.output, file_size as f64 / 1024.0);

    Ok(())
}

/// Load a trained discriminator.
pub fn load(path: &str, device: &Device) -> anyhow::Result<(VarMap, Discriminator)> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Discriminator::new(vb)?;
    varmap.load(path)?;
    Ok((varmap, model))
}

/// Quality gate — score a sprite, reject if below threshold.
pub fn quality_gate(
    model: &Discriminator,
    sprite: &RgbaImage,
    threshold: f32,
    device: &Device,
) -> anyhow::Result<(bool, f32)> {
    let mut pixels = Vec::with_capacity(3 * 16 * 16);
    for c in 0..3 {
        for y in 0..sprite.height() {
            for x in 0..sprite.width() {
                pixels.push(sprite.get_pixel(x, y)[c] as f32 / 255.0);
            }
        }
    }

    let tensor = Tensor::from_vec(pixels, (1, 3, 16, 16), device)?;
    let score = model.score(&tensor)?;
    Ok((score >= threshold, score))
}

/// Generate with quality gate — keep re-rolling rejects.
/// Returns only sprites that pass the discriminator threshold.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_gate(
    _tier: crate::device_cap::Tier,
    cond: &crate::class_cond::ClassCond,
    count: u32,
    steps: usize,
    threshold: f32,
    max_attempts: u32,
    disc_path: &str,
    device: &Device,
) -> anyhow::Result<Vec<RgbaImage>> {
    let (_varmap, disc) = load(disc_path, device)?;

    let mut accepted = Vec::new();
    let mut attempts = 0;

    while (accepted.len() as u32) < count && attempts < max_attempts {
        let need = count - accepted.len() as u32;
        let batch = crate::device_cap::auto_sample(cond, 16, need, steps)?;

        for sprite in batch {
            let (pass, score) = quality_gate(&disc, &sprite, threshold, device)?;
            if pass {
                accepted.push(sprite);
            } else {
                println!("  rejected sprite (score={:.3}, threshold={:.3})", score, threshold);
            }
            attempts += 1;
            if accepted.len() as u32 >= count { break; }
        }
    }

    if accepted.is_empty() {
        anyhow::bail!("no sprites passed quality gate after {} attempts", attempts);
    }

    Ok(accepted)
}
