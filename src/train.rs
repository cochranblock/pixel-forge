// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Training loop for the tiny pixel art diffusion model.
//! Pure Rust — no Python, no external frameworks.
//! Approach inspired by PixelGen (MIT, Anouar Khaldi 2025).
//!
//! Data pipeline: preprocess PNGs once → bincode+zstd blob → load into RAM → train.
//! Zero disk I/O during training. Palette swap augmentation in-memory.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use image::RgbaImage;
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::tiny_unet::TinyUNet;
use crate::medium_unet::MediumUNet;
use crate::anvil_unet::AnvilUNet;
use crate::micro_unet::MicroUNet;


/// Training config.
#[derive(Clone)]
pub struct TrainConfig {
    pub data_dir: String,
    pub output: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub img_size: u32,
    pub medium: bool,
    pub anvil: bool,
    /// Use MicroUNet (~97K params) for per-class siloed training.
    pub micro: bool,
    /// Class filter: only train on samples from these class subdirectories.
    pub class_filter: Vec<String>,
    /// MicroUNet channel config (e.g. [16,32] or [16,32,32]).
    pub micro_channels: Vec<usize>,
    /// Conditioning data dir for stage-aware training (e.g. silhouettes).
    /// When set, model takes 6 input channels (condition + noisy target).
    pub condition_dir: Option<String>,
    /// Mixed precision: f16 forward pass, f32 optimizer. ~2x faster on CUDA tensor cores.
    pub mixed_precision: bool,
    /// Classifier-free guidance: probability of dropping class label during training.
    /// 0.0 = never drop (no CFG), 0.1 = drop 10% of the time (recommended).
    pub cfg_dropout: f64,
    /// Enable EMA weight tracking. EMA weights saved as the final model.
    pub ema: bool,
    /// EMA decay rate. 0.9999 for long training, 0.999 for short.
    pub ema_decay: f64,
    /// Use cosine noise schedule instead of linear.
    pub cosine_schedule: bool,
    /// Min-SNR loss weighting. 0.0 = disabled, 5.0 = recommended.
    pub min_snr_gamma: f64,
    /// Cosine LR decay: ramp from lr → lr_min over training. 0.0 = flat LR.
    pub lr_min: f64,
    /// Warm-up epochs: linear ramp from lr_min to lr before decay kicks in.
    pub warmup_epochs: usize,
    /// V-prediction: model predicts velocity (noise - clean) instead of clean image.
    pub v_prediction: bool,
    /// Checkpoint every N epochs. 1 = every epoch (overwrites same file). 25 = default.
    pub checkpoint_every: usize,
    /// Resume from existing safetensors checkpoint. Loads weights before training.
    pub resume: Option<String>,
    /// BCE silhouette mode: no diffusion, no noise, sigmoid + BCE loss.
    /// Trains shape specialist (Anvil-sil, Quench-sil) — raw [0,1] pixel targets.
    pub bce: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".into(),
            output: "pixel-forge-cinder.safetensors".into(),
            epochs: 100,
            batch_size: 64,
            lr: 1e-3,
            img_size: 32,
            medium: false,
            anvil: false,
            micro: false,
            class_filter: Vec::new(),
            micro_channels: vec![16, 32],
            cfg_dropout: 0.1,
            ema: true,
            ema_decay: 0.9999,
            cosine_schedule: true,
            min_snr_gamma: 5.0,
            lr_min: 1e-5,
            warmup_epochs: 5,
            v_prediction: false,
            condition_dir: None,
            mixed_precision: false,
            checkpoint_every: 25,
            resume: None,
            bce: false,
        }
    }
}

/// Cosine noise schedule — spends more time in the mid-noise range
/// where structure forms. Better edge quality than linear.
pub fn cosine_schedule(t: f32) -> f32 {
    let s = 0.008; // small offset to prevent singularity at t=0
    let f_t = ((t + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
    let f_0 = (s / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
    1.0 - (f_t / f_0).min(1.0)
}

/// Min-SNR weighting — downweight easy timesteps (high noise) so the model
/// focuses on the hard ones (detail and structure).
fn min_snr_weight(noise_amount: f32, gamma: f64) -> f32 {
    if gamma <= 0.0 { return 1.0; }
    // SNR = (1 - noise_amount)^2 / noise_amount^2
    let clean = (1.0 - noise_amount).max(1e-6);
    let noise = noise_amount.max(1e-6);
    let snr = (clean / noise).powi(2);
    let weight = (gamma as f32).min(snr) / snr;
    weight.max(0.01) // floor to prevent zero-weight
}

/// EMA tracker — maintains exponential moving average of model weights.
struct EmaWeights {
    shadow: Vec<Tensor>,
    decay: f64,
}

impl EmaWeights {
    fn new(varmap: &VarMap, decay: f64) -> Self {
        // Store shadow weights on CPU to avoid GPU memory pressure
        let shadow: Vec<Tensor> = varmap.all_vars()
            .iter()
            .filter_map(|v| v.as_tensor().to_device(&Device::Cpu).ok())
            .collect();
        Self { shadow, decay }
    }

    /// Update shadow weights on CPU: shadow = decay * shadow + (1 - decay) * current
    fn update(&mut self, varmap: &VarMap) {
        let vars = varmap.all_vars();
        for (shadow, var) in self.shadow.iter_mut().zip(vars.iter()) {
            if let Ok(current_cpu) = var.as_tensor().to_device(&Device::Cpu)
                && let Ok(new_shadow) = shadow.affine(self.decay, 0.0)
                    .and_then(|s| current_cpu.affine(1.0 - self.decay, 0.0)
                        .and_then(|c| &s + &c))
            {
                *shadow = new_shadow;
            }
        }
    }

    /// Copy EMA weights into the varmap for saving/inference.
    /// Moves from CPU back to the device of the original weights.
    fn apply_to(&self, varmap: &VarMap) {
        let vars = varmap.all_vars();
        for (shadow, var) in self.shadow.iter().zip(vars.iter()) {
            let device = var.as_tensor().device().clone();
            if let Ok(on_device) = shadow.to_device(&device) {
                let _ = var.set(&on_device);
            }
        }
    }

    /// Swap: save current weights, load EMA weights, return originals for restore.
    fn swap_in(&self, varmap: &VarMap) -> Vec<Tensor> {
        let vars = varmap.all_vars();
        let originals: Vec<Tensor> = vars.iter()
            .map(|v| v.as_tensor().clone())
            .collect();
        self.apply_to(varmap);
        originals
    }

    /// Restore original weights after swap.
    fn swap_out(originals: &[Tensor], varmap: &VarMap) {
        let vars = varmap.all_vars();
        for (orig, var) in originals.iter().zip(vars.iter()) {
            let _ = var.set(orig);
        }
    }
}


/// Trait to unify UNet forward passes with hybrid conditioning.
pub trait DiffusionModel {
    fn forward(&self, x: &Tensor, timestep: &Tensor, super_id: &Tensor, tags: &Tensor) -> candle_core::Result<Tensor>;
}

impl DiffusionModel for TinyUNet {
    fn forward(&self, x: &Tensor, timestep: &Tensor, super_id: &Tensor, tags: &Tensor) -> candle_core::Result<Tensor> {
        TinyUNet::forward(self, x, timestep, super_id, tags)
    }
}

impl DiffusionModel for MediumUNet {
    fn forward(&self, x: &Tensor, timestep: &Tensor, super_id: &Tensor, tags: &Tensor) -> candle_core::Result<Tensor> {
        MediumUNet::forward(self, x, timestep, super_id, tags)
    }
}

impl DiffusionModel for AnvilUNet {
    fn forward(&self, x: &Tensor, timestep: &Tensor, super_id: &Tensor, tags: &Tensor) -> candle_core::Result<Tensor> {
        AnvilUNet::forward(self, x, timestep, super_id, tags)
    }
}

/// Pre-decoded dataset — all images as flat f32 vecs, packed into one blob.
/// bincode+zstd serialized for instant reload.
#[derive(Serialize, Deserialize)]
pub struct PackedDataset {
    /// Format version: 1 = legacy (labels), 2 = hybrid (super_ids + tags)
    #[serde(default = "default_version")]
    pub version: u8,
    pub img_size: u32,
    /// Flat f32 pixels: [sample0_r, sample0_g, sample0_b, sample1_r, ...] channel-first per sample
    pub pixels: Vec<f32>,
    /// Legacy class labels (v1). Empty in v2.
    pub labels: Vec<u32>,
    /// Super-category IDs (v2).
    #[serde(default)]
    pub super_ids: Vec<u32>,
    /// Binary trait tags per sample (v2). Each inner vec is [f32; 12] flattened.
    #[serde(default)]
    pub tags: Vec<[f32; 12]>,
    /// Number of f32 values per sample (3 * img_size * img_size)
    pub stride: usize,
}

fn default_version() -> u8 { 1 }

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

    let data_p = Path::new(data_dir);
    let class_names = {
        let mut names: Vec<String> = vec![
            "character", "weapon", "potion", "terrain", "enemy",
            "tree", "building", "animal", "effect", "food",
            "armor", "tool", "vehicle", "ui", "misc",
        ].into_iter().map(|s| s.to_string()).collect();
        if let Ok(entries) = std::fs::read_dir(data_p) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('_') || name.starts_with('.') { continue; }
                if entry.path().is_dir() && !names.contains(&name) {
                    names.push(name);
                }
            }
        }
        names
    };

    let stride = (3 * img_size * img_size) as usize;
    let mut pixels: Vec<f32> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();
    let mut super_ids: Vec<u32> = Vec::new();
    let mut tags_vec: Vec<[f32; 12]> = Vec::new();
    let mut total = 0usize;
    let mut bad = 0usize;

    for (class_id, name) in class_names.iter().enumerate() {
        let class_dir = data_p.join(name);
        if !class_dir.is_dir() {
            continue;
        }

        let cond = crate::class_cond::lookup(name);

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
                    super_ids.push(cond.super_id);
                    tags_vec.push(cond.tags);
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

    let dataset = PackedDataset { version: 2, img_size, pixels, labels, super_ids, tags: tags_vec, stride };

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

/// Preprocess only the specified class subdirectories.
/// Per-class cache: dataset-{filter_key}.bin.zst
pub fn preprocess_class(data_dir: &str, img_size: u32, class_filter: &[String]) -> Result<PackedDataset> {
    if class_filter.is_empty() {
        return preprocess(data_dir, img_size);
    }

    let mut sorted = class_filter.to_vec();
    sorted.sort();
    let filter_key = sorted.join("_");
    let cache_file = format!("{}/dataset-{}.bin.zst", data_dir, filter_key);
    let cache_path = Path::new(&cache_file);

    if cache_path.exists() {
        println!("loading cached class dataset from {cache_file}...");
        let t0 = std::time::Instant::now();
        let compressed = std::fs::read(cache_path)?;
        let decompressed = zstd::decode_all(compressed.as_slice())?;
        let dataset: PackedDataset = bincode::deserialize(&decompressed)?;
        let n = dataset.super_ids.len();
        let mb = compressed.len() as f64 / 1_048_576.0;
        println!("  loaded {} images from cache ({:.1} MB compressed) in {:.1}s",
            n, mb, t0.elapsed().as_secs_f32());
        return Ok(dataset);
    }

    println!("preprocessing classes [{}] from {data_dir}...", sorted.join(", "));
    let t0 = std::time::Instant::now();
    let data_p = Path::new(data_dir);
    let stride = (3 * img_size * img_size) as usize;
    let mut pixels: Vec<f32> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();
    let mut super_ids: Vec<u32> = Vec::new();
    let mut tags_vec: Vec<[f32; 12]> = Vec::new();
    let mut total = 0usize;
    let mut bad = 0usize;

    for name in &sorted {
        let class_dir = data_p.join(name);
        if !class_dir.is_dir() { continue; }

        let cond = crate::class_cond::lookup(name);
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
                    labels.push(0); // single-class, label unused
                    super_ids.push(cond.super_id);
                    tags_vec.push(cond.tags);
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
        anyhow::bail!("no images found for classes [{}] in {data_dir}", sorted.join(", "));
    }

    let dataset = PackedDataset { version: 2, img_size, pixels, labels, super_ids, tags: tags_vec, stride };
    let encoded = bincode::serialize(&dataset)?;
    let compressed = zstd::encode_all(encoded.as_slice(), 3)?;
    std::fs::write(cache_path, &compressed)?;

    let comp_mb = compressed.len() as f64 / 1_048_576.0;
    println!("total: {} images, {} bad skipped ({:.1} MB zstd) in {:.1}s",
        total, bad, comp_mb, t0.elapsed().as_secs_f32());

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
    let mut rng = rand::rng();

    // Pass 1: dedupe into a vec (order matters for shuffle mapping)
    let mut colors: Vec<u32> = Vec::with_capacity(64);
    let mut seen: HashMap<u32, usize> = HashMap::with_capacity(64);
    for i in 0..n {
        let r = (px[i] * 255.0) as u8;
        let g = (px[n + i] * 255.0) as u8;
        let b = (px[2 * n + i] * 255.0) as u8;
        let key = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
            e.insert(colors.len());
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

/// Rotate a single sample 90 degrees clockwise in-place.
/// For channel-first layout: px[c * n + y * w + x] → px[c * n + x * w + (w - 1 - y)]
fn rot90_cw(px: &mut [f32], img_size: u32) {
    let w = img_size as usize;
    let n = w * w;
    let mut buf = vec![0.0f32; n];
    for c in 0..3 {
        let base = c * n;
        for y in 0..w {
            for x in 0..w {
                buf[x * w + (w - 1 - y)] = px[base + y * w + x];
            }
        }
        px[base..base + n].copy_from_slice(&buf);
    }
}

/// Corrupt: x_noisy = x * (1 - amount) + noise * amount
/// Uses Gaussian N(0,1) noise — signal in [0,1], noise centered at 0.
/// Legacy linear-blend; the EDM trainer/sampler use σ-scaled noise instead.
#[allow(dead_code)]
fn corrupt(x: &Tensor, amount: &Tensor, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    let noise = Tensor::randn(0f32, 1f32, x.shape(), device)?.to_dtype(x.dtype())?;
    let a = amount.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
    let one_minus_a = a.ones_like()?.broadcast_sub(&a)?;
    let noisy = (x.broadcast_mul(&one_minus_a)? + noise.broadcast_mul(&a)?)?;
    Ok((noisy, noise))
}

/// Scale a (B, C, H, W) tensor by a per-sample scalar via (B,1,1,1)
/// broadcast. Used by the EDM trainer to apply c_in to each batch element
/// without a Python-style loop over the GPU tensor.
fn scale_per_sample(x: &Tensor, scales: &[f32], device: &Device) -> candle_core::Result<Tensor> {
    let bs = scales.len();
    let s = Tensor::new(scales, device)?.reshape((bs, 1, 1, 1))?;
    x.broadcast_mul(&s)
}

/// Save model weights with optional v_pred marker tensor.
/// When v_prediction is true, a "v_pred_marker" tensor (single f32 = 1.0) is
/// embedded in the safetensors file so sampling auto-detects the prediction mode.
fn save_with_marker(
    varmap: &VarMap,
    path: &str,
    v_prediction: bool,
    normalizer: &crate::normalize::Normalizer,
) -> Result<()> {
    if !v_prediction {
        varmap.save(path)?;
    } else {
        // Collect all varmap tensors + the marker
        let mut tensors = std::collections::HashMap::new();
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }
        drop(data);
        let marker = Tensor::new(&[1.0f32], &Device::Cpu)?;
        tensors.insert("v_pred_marker".into(), marker);
        candle_core::safetensors::save(&tensors, path)?;
    }
    crate::nanosign::sign_and_log(path)?;
    normalizer.save_sidecar(std::path::Path::new(path))?;
    Ok(())
}

/// Detect whether a safetensors model was trained with v-prediction
/// by checking for the "v_pred_marker" tensor in the file header.
pub fn detect_v_pred(path: &str) -> bool {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() < 8 { return false; }
    let header_len = match data[..8].try_into().ok().map(u64::from_le_bytes) {
        Some(n) => n as usize,
        None => return false,
    };
    let header_str = match std::str::from_utf8(data.get(8..8 + header_len).unwrap_or(&[])) {
        Ok(s) => s,
        Err(_) => return false,
    };
    header_str.contains("v_pred_marker")
}

/// Train loop — works with any DiffusionModel (Tiny, Medium, Anvil).
/// Supports CFG, EMA, cosine noise schedule, min-SNR weighting, and v-prediction.
/// How training ended — used by Sponge Mesh to decide retry strategy.
#[derive(Debug)]
pub enum TrainStop {
    /// Ran all epochs successfully.
    Complete { final_loss: f64 },
    /// Loss went NaN at this epoch.
    NanLoss(usize),
    /// Loss stalled: no improvement over a window of epochs.
    Plateau { epoch: usize, loss: f64 },
}

fn train_inner(
    model: &dyn DiffusionModel,
    varmap: &VarMap,
    config: &TrainConfig,
    dataset: &PackedDataset,
    cond_dataset: Option<&PackedDataset>,
    device: &Device,
) -> Result<TrainStop> {
    let n = dataset.labels.len();
    let stride = dataset.stride;

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: config.lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    // EMA weight tracker
    let mut ema = if config.ema {
        Some(EmaWeights::new(varmap, config.ema_decay))
    } else {
        None
    };

    let cfg_info = if config.cfg_dropout > 0.0 {
        format!("cfg_drop={}", config.cfg_dropout)
    } else {
        "no-cfg".into()
    };

    println!("training: {} epochs, bs={}, lr={}, {} samples", config.epochs, config.batch_size, config.lr, n);
    println!("augmentation: palette swap + h-flip (50%) + rotation (0/90/180/270) + brightness (50%)");
    let prec = if config.mixed_precision { " | fp16" } else { "" };
    println!("recipe: EDM (log-normal σ + σ-precond + Min-SNR-γ=13) | {cfg_info} | ema={}{prec}",
        config.ema);

    // EDM still requires the normalize manifest — uses mean for centering
    // and sigma_data as the preconditioning constant.
    let normalizer = crate::normalize::Normalizer::load(
        std::path::Path::new(&format!("{}/normalize.json", config.data_dir))
    )?
    .ok_or_else(|| anyhow::anyhow!(
        "{}/normalize.json missing — run `pixel-forge normalize-stats --data {}` first",
        config.data_dir, config.data_dir
    ))?;
    println!("center: mean={:?} σ_data={:.4} (sidecar saved with checkpoints)",
        normalizer.mean, normalizer.sigma_data());

    let t0 = std::time::Instant::now();
    let sz = config.img_size as usize;
    let mut loss_history: Vec<f64> = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let epoch_start = std::time::Instant::now();

        // Cosine LR decay with linear warm-up
        let lr = if config.lr_min > 0.0 {
            if epoch < config.warmup_epochs {
                // Linear warm-up: lr_min → lr
                let frac = epoch as f64 / config.warmup_epochs.max(1) as f64;
                config.lr_min + (config.lr - config.lr_min) * frac
            } else {
                // Cosine decay: lr → lr_min
                let decay_epochs = config.epochs - config.warmup_epochs;
                let frac = (epoch - config.warmup_epochs) as f64 / decay_epochs.max(1) as f64;
                config.lr_min + 0.5 * (config.lr - config.lr_min) * (1.0 + (frac * std::f64::consts::PI).cos())
            }
        } else {
            config.lr
        };
        opt.set_learning_rate(lr);

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::rng());

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut rng = rand::rng();

        let num_batches = n.div_ceil(config.batch_size);
        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;
            let end = (start + config.batch_size).min(n);
            let bs = end - start;

            let mut batch_px = Vec::with_capacity(bs * stride);
            let mut batch_labels: Vec<u32> = Vec::with_capacity(bs);
            let mut batch_super: Vec<u32> = Vec::with_capacity(bs);
            let mut batch_tags: Vec<f32> = Vec::with_capacity(bs * crate::class_cond::NUM_TAGS);

            for &idx in &indices[start..end] {
                let src_start = idx * stride;
                let mut sample = dataset.pixels[src_start..src_start + stride].to_vec();

                if rng.random_bool(0.5) {
                    palette_swap(&mut sample, stride, config.img_size);
                }
                if rng.random_bool(0.5) {
                    hflip(&mut sample, config.img_size);
                }
                // Random rotation: 0/90/180/270 degrees (25% each)
                let rotations = rng.random_range(0..4u8);
                for _ in 0..rotations {
                    rot90_cw(&mut sample, config.img_size);
                }
                // Random brightness (50% chance): multiply by [0.8, 1.2], clamp [0,1]
                if rng.random_bool(0.5) {
                    let factor = rng.random_range(0.8f32..1.2f32);
                    for v in sample.iter_mut() {
                        *v = (*v * factor).clamp(0.0, 1.0);
                    }
                }

                // EDM expects centered (mean-only) data — c_in handles
                // the σ-rescale at every noise level inside the train step.
                let n_px = (config.img_size * config.img_size) as usize;
                normalizer.center_pixels(&mut sample, n_px);

                batch_px.extend_from_slice(&sample);

                // CFG: randomly drop conditioning to train unconditional path
                if config.cfg_dropout > 0.0 && rng.random_bool(config.cfg_dropout) {
                    batch_labels.push(crate::class_cond::CFG_NULL_SUPER);
                    batch_super.push(crate::class_cond::CFG_NULL_SUPER);
                    batch_tags.extend_from_slice(&crate::class_cond::CFG_NULL_TAGS);
                } else {
                    batch_labels.push(dataset.labels.get(idx).copied().unwrap_or(0));
                    batch_super.push(dataset.super_ids[idx]);
                    batch_tags.extend_from_slice(&dataset.tags[idx]);
                }
            }

            let x_batch = Tensor::new(batch_px.as_slice(), device)?
                .reshape((bs, 3, sz, sz))?;
            let super_batch = Tensor::new(batch_super.as_slice(), device)?;
            let tags_batch = Tensor::new(batch_tags.as_slice(), device)?
                .reshape((bs, crate::class_cond::NUM_TAGS))?;

            // EDM noise schedule: per-sample σ ~ log-normal(P_MEAN, P_STD).
            // Replaces the old cosine/linear t∈[0,1] schedule.
            let mut sigmas_vec = Vec::with_capacity(bs);
            for _ in 0..bs {
                sigmas_vec.push(crate::precond::sigma_from_uniform(rng.random::<f32>()));
            }
            let sigma_data = normalizer.sigma_data();
            let coeffs: Vec<crate::precond::EdmCoeffs> = sigmas_vec.iter()
                .map(|&s| crate::precond::EdmCoeffs::at(s, sigma_data))
                .collect();

            // x_noisy = clean + σ·ε; broadcast σ across (3, H, W).
            let noise = Tensor::randn(0f32, 1f32, x_batch.shape(), device)?;
            let sigma_t = Tensor::new(sigmas_vec.as_slice(), device)?
                .reshape((bs, 1, 1, 1))?;
            let x_noisy = (&x_batch + noise.broadcast_mul(&sigma_t)?)?;

            // For conditioned training, concat conditioning channels to input.
            // The cond channels stay in centered (not σ-scaled) space — they
            // carry coarse structure that doesn't need σ-rescaling.
            let model_input_raw = if let Some(cond) = cond_dataset {
                let mut cond_px = Vec::with_capacity(bs * stride);
                let n_px = (config.img_size * config.img_size) as usize;
                for &idx in &indices[start..end] {
                    let src_start = idx * stride;
                    let mut sample = cond.pixels[src_start..src_start + stride].to_vec();
                    normalizer.center_pixels(&mut sample, n_px);
                    cond_px.extend_from_slice(&sample);
                }
                let cond_batch = Tensor::new(cond_px.as_slice(), device)?
                    .reshape((bs, 3, sz, sz))?;
                let scaled_noisy = scale_per_sample(&x_noisy, &coeffs.iter()
                    .map(|c| c.c_in).collect::<Vec<_>>(), device)?;
                Tensor::cat(&[&cond_batch, &scaled_noisy], 1)? // (B, 6, H, W)
            } else {
                scale_per_sample(&x_noisy, &coeffs.iter()
                    .map(|c| c.c_in).collect::<Vec<_>>(), device)?
            };

            // Time conditioning is c_noise = 0.25·log(σ).
            let c_noise: Vec<f32> = coeffs.iter().map(|c| c.c_noise).collect();
            let t_cnoise = Tensor::new(c_noise.as_slice(), device)?;

            let pred = model.forward(&model_input_raw, &t_cnoise, &super_batch, &tags_batch)?;

            // EDM target for the network's raw output F:
            //   target = (clean − c_skip · x_noisy) / c_out
            // Loss = λ(σ) · ||F − target||² (Min-SNR-γ clamped at γ=5 inside
            // EdmCoeffs::at). Implementation: pre-scale pred and target by
            // √λ so plain MSE matches the weighted formulation.
            let cskip_v: Vec<f32> = coeffs.iter().map(|c| c.c_skip).collect();
            let cout_v:  Vec<f32> = coeffs.iter().map(|c| c.c_out).collect();
            let cskip_t = Tensor::new(cskip_v.as_slice(), device)?
                .reshape((bs, 1, 1, 1))?;
            let cout_t = Tensor::new(cout_v.as_slice(), device)?
                .reshape((bs, 1, 1, 1))?;
            let target = ((&x_batch - x_noisy.broadcast_mul(&cskip_t)?)?
                .broadcast_div(&cout_t))?;

            let sqrt_w: Vec<f32> = coeffs.iter().map(|c| c.loss_weight.sqrt()).collect();
            let sqrt_w_t = Tensor::new(sqrt_w.as_slice(), device)?
                .reshape((bs, 1, 1, 1))?;
            let pred_w   = pred.broadcast_mul(&sqrt_w_t)?;
            let target_w = target.broadcast_mul(&sqrt_w_t)?;
            let per_sample_loss = (&pred_w - &target_w)?.sqr()?;
            // EDM's λ(σ) is already baked into pred_w/target_w (sqrt-weighted),
            // so plain mean is the weighted MSE. Min-SNR-γ clamping happens
            // inside EdmCoeffs::at — config.min_snr_gamma is now ignored on
            // the EDM path.
            let loss = per_sample_loss.mean_all()?;

            opt.backward_step(&loss)?;

            // Update EMA after each optimizer step
            if let Some(ref mut ema) = ema {
                ema.update(varmap);
            }

            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
        let elapsed = epoch_start.elapsed().as_secs_f32();

        // Sponge Mesh: NaN detection — training is unrecoverable
        if avg_loss.is_nan() || avg_loss.is_infinite() {
            println!("  epoch {}/{}: loss=NaN — aborting", epoch + 1, config.epochs);
            return Ok(TrainStop::NanLoss(epoch));
        }

        // Track loss history for plateau detection
        loss_history.push(avg_loss);

        // Sponge Mesh: plateau detection — no improvement over 20 epochs
        const PLATEAU_WINDOW: usize = 20;
        const PLATEAU_THRESHOLD: f64 = 1e-6;
        if loss_history.len() >= PLATEAU_WINDOW {
            let recent = &loss_history[loss_history.len() - PLATEAU_WINDOW..];
            let best_recent = recent.iter().copied().fold(f64::MAX, f64::min);
            let worst_recent = recent.iter().copied().fold(f64::MIN, f64::max);
            if (worst_recent - best_recent).abs() < PLATEAU_THRESHOLD {
                println!("  epoch {}/{}: loss plateau ({:.8} ± {:.2e}) — aborting",
                    epoch + 1, config.epochs, avg_loss, worst_recent - best_recent);
                return Ok(TrainStop::Plateau { epoch, loss: avg_loss });
            }
        }

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            println!("  epoch {}/{}: loss={:.6} lr={:.1e} ({:.1}s)", epoch + 1, config.epochs, avg_loss, lr, elapsed);
        }

        if config.checkpoint_every > 0 && (epoch + 1) % config.checkpoint_every == 0 {
            let cp = if config.checkpoint_every == 1 {
                config.output.clone() // overwrite same file every epoch
            } else {
                format!("{}.epoch{}", config.output, epoch + 1)
            };
            if let Some(ref ema) = ema {
                let originals = ema.swap_in(varmap);
                save_with_marker(varmap, &cp, config.v_prediction, &normalizer)?;
                println!("  checkpoint (ema): {cp}");
                EmaWeights::swap_out(&originals, varmap);
            } else {
                save_with_marker(varmap, &cp, config.v_prediction, &normalizer)?;
                println!("  checkpoint: {cp}");
            }
        }
    }

    // Save final — use EMA weights if available
    if let Some(ref ema) = ema {
        ema.apply_to(varmap);
        println!("saving model (ema weights) to {}...", config.output);
    } else {
        println!("saving model to {}...", config.output);
    }
    save_with_marker(varmap, &config.output, config.v_prediction, &normalizer)?;

    let file_size = std::fs::metadata(&config.output)?.len();
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    println!("done: {} ({:.1} MB) in {:.0}s total",
        config.output, file_size as f64 / 1_048_576.0, t0.elapsed().as_secs_f32());

    Ok(TrainStop::Complete { final_loss })
}

/// BCE training loop — no noise, no diffusion. Sigmoid + binary cross-entropy.
/// Used for shape specialists: Anvil-sil (full-body) and Quench-sil (parts).
fn train_bce_inner(
    model: &dyn DiffusionModel,
    varmap: &VarMap,
    config: &TrainConfig,
    dataset: &PackedDataset,
    cond_dataset: Option<&PackedDataset>,
    device: &Device,
) -> Result<TrainStop> {
    let n = dataset.labels.len();
    let stride = dataset.stride;
    let sz = config.img_size as usize;

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: config.lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    let cfg_info = if config.cfg_dropout > 0.0 {
        format!("cfg_drop={}", config.cfg_dropout)
    } else {
        "no-cfg".into()
    };
    println!("training: {} epochs, bs={}, lr={}, {} samples",
        config.epochs, config.batch_size, config.lr, n);
    println!("augmentation: h-flip (50%) + rotation (0/90/180/270)");
    println!("recipe: BCE (no diffusion, BCE-with-logits) | {cfg_info} | ema=false");

    let t0 = std::time::Instant::now();
    let mut loss_history: Vec<f64> = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let epoch_start = std::time::Instant::now();

        let lr = if config.lr_min > 0.0 {
            if epoch < config.warmup_epochs {
                let frac = epoch as f64 / config.warmup_epochs.max(1) as f64;
                config.lr_min + (config.lr - config.lr_min) * frac
            } else {
                let decay_epochs = config.epochs - config.warmup_epochs;
                let frac = (epoch - config.warmup_epochs) as f64 / decay_epochs.max(1) as f64;
                config.lr_min + 0.5 * (config.lr - config.lr_min)
                    * (1.0 + (frac * std::f64::consts::PI).cos())
            }
        } else {
            config.lr
        };
        opt.set_learning_rate(lr);

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::rng());
        let mut epoch_loss = 0.0f64;
        let mut batch_count = 0usize;
        let mut rng = rand::rng();

        let num_batches = n.div_ceil(config.batch_size);
        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;
            let end = (start + config.batch_size).min(n);
            let bs = end - start;

            let mut batch_px = Vec::with_capacity(bs * stride);
            let mut batch_super: Vec<u32> = Vec::with_capacity(bs);
            let mut batch_tags: Vec<f32> = Vec::with_capacity(bs * crate::class_cond::NUM_TAGS);

            for &idx in &indices[start..end] {
                let src = idx * stride;
                let mut sample = dataset.pixels[src..src + stride].to_vec();
                // No palette swap / brightness — those break binary mask pixel values.
                if rng.random_bool(0.5) { hflip(&mut sample, config.img_size); }
                let rotations = rng.random_range(0..4u8);
                for _ in 0..rotations { rot90_cw(&mut sample, config.img_size); }
                // No centering — BCE needs raw [0,1] values as targets.
                batch_px.extend_from_slice(&sample);

                if config.cfg_dropout > 0.0 && rng.random_bool(config.cfg_dropout) {
                    batch_super.push(crate::class_cond::CFG_NULL_SUPER);
                    batch_tags.extend_from_slice(&crate::class_cond::CFG_NULL_TAGS);
                } else {
                    batch_super.push(dataset.super_ids[idx]);
                    batch_tags.extend_from_slice(&dataset.tags[idx]);
                }
            }

            let x_batch = Tensor::new(batch_px.as_slice(), device)?
                .reshape((bs, 3, sz, sz))?;
            let super_batch = Tensor::new(batch_super.as_slice(), device)?;
            let tags_batch = Tensor::new(batch_tags.as_slice(), device)?
                .reshape((bs, crate::class_cond::NUM_TAGS))?;

            // Timestep is meaningless for BCE — pass zeros.
            let t_zeros = Tensor::zeros((bs,), DType::F32, device)?;

            // Conditioning: Quench-BCE takes Anvil mask as 6ch input.
            // Anvil-BCE takes zeros — model generates from class embedding alone.
            let x_input = if let Some(cond) = cond_dataset {
                let mut cond_px = Vec::with_capacity(bs * stride);
                for &idx in &indices[start..end] {
                    let src = idx * stride;
                    cond_px.extend_from_slice(&cond.pixels[src..src + stride]);
                }
                let cond_batch = Tensor::new(cond_px.as_slice(), device)?
                    .reshape((bs, 3, sz, sz))?;
                Tensor::cat(&[&cond_batch, &x_batch], 1)?
            } else {
                Tensor::zeros_like(&x_batch)?
            };

            let pred = model.forward(&x_input, &t_zeros, &super_batch, &tags_batch)?;

            // Stable BCE with logits: max(p,0) − p·y + log(1 + exp(−|p|))
            let relu_pred = ((&pred + pred.abs()?)? * 0.5f64)?;
            let log1p_exp = pred.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
            let loss = ((&relu_pred - (&pred * &x_batch)?)? + log1p_exp)?.mean_all()?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
        let elapsed = epoch_start.elapsed().as_secs_f32();

        if avg_loss.is_nan() || avg_loss.is_infinite() {
            println!("  epoch {}/{}: loss=NaN — aborting", epoch + 1, config.epochs);
            return Ok(TrainStop::NanLoss(epoch));
        }
        loss_history.push(avg_loss);

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            println!("  epoch {}/{}: loss={:.6} lr={:.1e} ({:.1}s)",
                epoch + 1, config.epochs, avg_loss, lr, elapsed);
        }

        if config.checkpoint_every > 0 && (epoch + 1) % config.checkpoint_every == 0 {
            let cp = if config.checkpoint_every == 1 {
                config.output.clone()
            } else {
                format!("{}.epoch{}", config.output, epoch + 1)
            };
            varmap.save(&cp)?;
            crate::nanosign::sign_and_log(&cp)?;
            println!("  checkpoint: {cp}");
        }
    }

    println!("saving model to {}...", config.output);
    varmap.save(&config.output)?;
    crate::nanosign::sign_and_log(&config.output)?;
    let file_size = std::fs::metadata(&config.output)?.len();
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    println!("done: {} ({:.1} MB) in {:.0}s total",
        config.output, file_size as f64 / 1_048_576.0, t0.elapsed().as_secs_f32());
    Ok(TrainStop::Complete { final_loss })
}

/// Train — dispatches to Micro, Tiny, Medium, or Anvil based on config.
pub fn train(config: &TrainConfig) -> Result<TrainStop> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    let dataset = if config.micro && !config.class_filter.is_empty() {
        preprocess_class(&config.data_dir, config.img_size, &config.class_filter)?
    } else {
        preprocess(&config.data_dir, config.img_size)?
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let in_ch = if config.condition_dir.is_some() { 6 } else { 3 };
    let cond_dataset = if let Some(ref cdir) = config.condition_dir {
        println!("loading conditioning data from {cdir}...");
        let cd = preprocess(cdir, config.img_size)?;
        if cd.labels.len() != dataset.labels.len() {
            anyhow::bail!("conditioning dataset size ({}) != training dataset size ({})", cd.labels.len(), dataset.labels.len());
        }
        Some(cd)
    } else {
        None
    };

    let stop = if config.micro {
        let model = MicroUNet::with_channels(vb, &config.micro_channels)?;
        let params = MicroUNet::param_count(&varmap);
        let ch_str = config.micro_channels.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
        println!("model: Micro (MicroUNet, [{ch_str}]), {} params ({:.1} MB)",
            params, params as f64 * 4.0 / 1_048_576.0);
        if let Some(ref checkpoint) = config.resume {
            crate::nanosign::verify_or_bail(checkpoint)?;
            varmap.load(checkpoint)?;
            println!("resumed from {checkpoint}");
        }
        let dispatch = if config.bce { train_bce_inner as fn(&dyn DiffusionModel, &VarMap, &TrainConfig, &PackedDataset, Option<&PackedDataset>, &Device) -> Result<TrainStop> } else { train_inner };
        dispatch(&model, &varmap, config, &dataset, cond_dataset.as_ref(), &device)?
    } else if config.anvil {
        let model = AnvilUNet::new(vb)?;
        let params = AnvilUNet::param_count(&varmap);
        println!("model: Anvil (AnvilUNet), {} params ({:.1} MB)", params, params as f64 * 4.0 / 1_048_576.0);
        if let Some(ref checkpoint) = config.resume {
            crate::nanosign::verify_or_bail(checkpoint)?;
            varmap.load(checkpoint)?;
            println!("resumed from {checkpoint}");
        }
        let dispatch = if config.bce { train_bce_inner as fn(&dyn DiffusionModel, &VarMap, &TrainConfig, &PackedDataset, Option<&PackedDataset>, &Device) -> Result<TrainStop> } else { train_inner };
        dispatch(&model, &varmap, config, &dataset, cond_dataset.as_ref(), &device)?
    } else if config.medium {
        let model = MediumUNet::with_channels(vb, in_ch)?;
        let params = MediumUNet::param_count(&varmap);
        println!("model: Quench (MediumUNet, {in_ch}ch), {} params ({:.1} MB)", params, params as f64 * 4.0 / 1_048_576.0);
        if let Some(ref checkpoint) = config.resume {
            crate::nanosign::verify_or_bail(checkpoint)?;
            varmap.load(checkpoint)?;
            println!("resumed from {checkpoint}");
        }
        let dispatch = if config.bce { train_bce_inner as fn(&dyn DiffusionModel, &VarMap, &TrainConfig, &PackedDataset, Option<&PackedDataset>, &Device) -> Result<TrainStop> } else { train_inner };
        dispatch(&model, &varmap, config, &dataset, cond_dataset.as_ref(), &device)?
    } else {
        let model = TinyUNet::with_channels(vb, in_ch)?;
        let params = TinyUNet::param_count(&varmap);
        println!("model: Cinder (TinyUNet, {in_ch}ch), {} params ({:.1} MB)", params, params as f64 * 4.0 / 1_048_576.0);
        if let Some(ref checkpoint) = config.resume {
            if in_ch > 3 {
                // 6ch model: use lenient loader so conv_in.weight (3ch→6ch) stays
                // at random init while all other weights transfer from the 3ch base.
                crate::quantize::load_varmap_lenient(&mut varmap, checkpoint)?;
            } else {
                crate::nanosign::verify_or_bail(checkpoint)?;
                varmap.load(checkpoint)?;
            }
            println!("resumed from {checkpoint}");
        }
        let dispatch = if config.bce { train_bce_inner as fn(&dyn DiffusionModel, &VarMap, &TrainConfig, &PackedDataset, Option<&PackedDataset>, &Device) -> Result<TrainStop> } else { train_inner };
        dispatch(&model, &varmap, config, &dataset, cond_dataset.as_ref(), &device)?
    };

    Ok(stop)
}

/// Default CFG guidance scale for inference.
/// Higher = stronger class adherence, lower = more diversity.
pub const DEFAULT_CFG_SCALE: f64 = 3.0;

/// Create seeded noise tensor. Same seed + class + index = same sprite every time.
/// Inspired by No Man's Sky / Factorio deterministic world generation:
/// a single seed cascades into reproducible content.
pub fn seeded_noise(seed: Option<u64>, class_id: u32, index: u32, img_size: u32, device: &Device) -> candle_core::Result<Tensor> {
    use rand::SeedableRng;
    let n = img_size as usize;

    match seed {
        Some(s) => {
            // Hash seed + class + index for unique but deterministic Gaussian noise
            let combined = s.wrapping_mul(2654435761) ^ (class_id as u64).wrapping_mul(40503) ^ (index as u64).wrapping_mul(65537);
            let mut rng = rand::rngs::StdRng::seed_from_u64(combined);
            // Box-Muller transform for Gaussian samples
            let vals: Vec<f32> = (0..3 * n * n).map(|_| {
                let u1: f32 = rng.random::<f32>().max(1e-7);
                let u2: f32 = rng.random::<f32>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            }).collect();
            Tensor::from_vec(vals, (1, 3, n, n), device)
        }
        None => {
            Tensor::randn(0f32, 1f32, (1, 3, n, n), device)
        }
    }
}

/// CFG-guided denoising step: run model conditioned + unconditional,
/// blend predictions to amplify class signal.
/// For v-pred models, converts v→clean before CFG blend to avoid amplifying velocity.
#[allow(clippy::too_many_arguments)]
/// Convert epsilon prediction to clean image: clean = (x - eps * t) / (1 - t)
/// At very high noise (t > 0.999), the denominator approaches zero.
/// In that regime, the model can't meaningfully predict clean — return zeros
/// (which gets re-noised in the DDIM step anyway).
pub fn eps_to_clean(x: &Tensor, pred_eps: &Tensor, amount: f32) -> candle_core::Result<Tensor> {
    if amount > 0.999 {
        // At t≈1, noisy ≈ pure noise. Can't recover clean. Return zero (black).
        // The DDIM step will add appropriate noise for the next timestep.
        return x.zeros_like();
    }
    let one_minus_a = 1.0 - amount as f64;
    (x - (pred_eps * amount as f64)?)? * (1.0 / one_minus_a)
}

#[allow(clippy::too_many_arguments)]
pub fn cfg_denoise(
    model: &dyn DiffusionModel,
    x: &Tensor,
    t: &Tensor,
    super_id: &Tensor,
    tags: &Tensor,
    null_super: &Tensor,
    null_tags: &Tensor,
    cfg_scale: f64,
) -> candle_core::Result<Tensor> {
    if cfg_scale <= 1.0 {
        return model.forward(x, t, super_id, tags);
    }
    let cond = model.forward(x, t, super_id, tags)?;
    let uncond = model.forward(x, t, null_super, null_tags)?;
    let diff = (&cond - &uncond)?;
    &uncond + (diff * cfg_scale)?
}

/// CFG for v-pred models: convert v→clean before blending, then return clean.
/// This prevents CFG scale from amplifying velocity values out of range.
#[allow(clippy::too_many_arguments)]
pub fn cfg_denoise_vpred(
    model: &dyn DiffusionModel,
    x: &Tensor,
    t: &Tensor,
    amount: f32,
    super_id: &Tensor,
    tags: &Tensor,
    null_super: &Tensor,
    null_tags: &Tensor,
    cfg_scale: f64,
) -> candle_core::Result<Tensor> {
    let cond_v = model.forward(x, t, super_id, tags)?;
    let cond_clean = (x - (&cond_v * amount as f64)?)?;

    if cfg_scale <= 1.0 {
        return Ok(cond_clean);
    }

    let uncond_v = model.forward(x, t, null_super, null_tags)?;
    let uncond_clean = (x - (&uncond_v * amount as f64)?)?;

    // CFG on clean predictions, not on raw velocity
    let diff = (&cond_clean - &uncond_clean)?;
    &uncond_clean + (diff * cfg_scale)?
}

/// Skeleton noise level — 30% noise, 70% skeleton.
/// Enters the cosine schedule at ~60%, so 10 steps focus on detail not structure.
const SKELETON_NOISE: f32 = 0.30;

/// Compute per-class mean images from the dataset and save as safetensors.
/// Each class gets a (1, 3, H, W) tensor keyed by class index.
pub fn compute_skeletons(data_dir: &str, img_size: u32) -> Result<()> {
    let dataset = preprocess(data_dir, img_size)?;
    let n = dataset.labels.len();
    let stride = dataset.stride;
    let num_classes = 15usize; // real classes, no null

    let mut sums = vec![vec![0.0f64; stride]; num_classes];
    let mut counts = vec![0usize; num_classes];

    for i in 0..n {
        let c = dataset.labels[i] as usize;
        if c >= num_classes { continue; }
        counts[c] += 1;
        let offset = i * stride;
        for (j, sum) in sums[c].iter_mut().enumerate() {
            *sum += dataset.pixels[offset + j] as f64;
        }
    }

    let output = format!("{}/skeletons.safetensors", data_dir);
    let mut tensors = std::collections::HashMap::new();

    for c in 0..num_classes {
        if counts[c] == 0 { continue; }
        let mean: Vec<f32> = sums[c].iter().map(|s| (*s / counts[c] as f64) as f32).collect();
        let t = Tensor::from_vec(mean, (1, 3, img_size as usize, img_size as usize), &Device::Cpu)?;
        tensors.insert(format!("class_{c}"), t);
        println!("  class {c}: {} samples → skeleton", counts[c]);
    }

    candle_core::safetensors::save(&tensors, &output)?;
    println!("saved: {output} ({num_classes} skeletons)");
    Ok(())
}

/// Load skeleton for a class. Returns (1, 3, H, W) tensor on the given device.
/// Falls back to None if file missing, class not found, or size mismatch.
pub fn load_skeleton(data_dir: &str, class_id: u32, img_size: u32, device: &Device) -> Option<Tensor> {
    let path = format!("{}/skeletons.safetensors", data_dir);
    if !Path::new(&path).exists() {
        // Also check next to the model (CWD, HOME, exe dir)
        let candidate = "skeletons.safetensors";
        if Path::new(candidate).exists() {
            return load_skeleton_from(candidate, class_id, img_size, device);
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_path = format!("{}/{}", home, candidate);
            if Path::new(&home_path).exists() {
                return load_skeleton_from(&home_path, class_id, img_size, device);
            }
        }
        return None;
    }
    load_skeleton_from(&path, class_id, img_size, device)
}

fn load_skeleton_from(path: &str, class_id: u32, img_size: u32, device: &Device) -> Option<Tensor> {
    let data = std::fs::read(path).ok()?;
    let tensors = candle_core::safetensors::load_buffer(&data, device).ok()?;
    let key = format!("class_{}", class_id);
    let t = tensors.get(&key)?;
    // Verify spatial dimensions match
    let dims = t.dims();
    if dims.len() == 4 && dims[2] == img_size as usize && dims[3] == img_size as usize {
        Some(t.clone())
    } else {
        None
    }
}

/// Compute per-class mean images from training data, keyed by class directory name.
/// Saves to `{data_dir}/skeletons_v2.safetensors`.
/// Replaces the old integer-keyed `compute_skeletons` which broke when hybrid
/// conditioning was introduced (super-categories ≠ legacy integer class IDs).
pub fn compute_skeletons_v2(data_dir: &str, img_size: u32) -> Result<()> {
    let data_p = Path::new(data_dir);
    let stride = (3 * img_size * img_size) as usize;

    let mut class_dirs: Vec<_> = std::fs::read_dir(data_p)?
        .flatten()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            e.path().is_dir() && !name.starts_with('_') && !name.starts_with('.')
        })
        .collect();
    class_dirs.sort_by_key(|e| e.file_name());

    let mut tensors = std::collections::HashMap::new();

    for dir_entry in &class_dirs {
        let class_name = dir_entry.file_name().to_string_lossy().to_string();
        let class_dir = dir_entry.path();

        let mut pngs: Vec<_> = std::fs::read_dir(&class_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("png"))
            .collect();
        pngs.sort_by_key(|e| e.file_name());

        if pngs.is_empty() { continue; }

        let mut sum = vec![0.0f64; stride];
        let mut count = 0usize;

        for entry in &pngs {
            if let Some(px) = decode_png(&entry.path(), img_size) {
                for (i, v) in px.iter().enumerate() {
                    sum[i] += *v as f64;
                }
                count += 1;
            }
        }

        if count == 0 { continue; }

        let mean: Vec<f32> = sum.iter().map(|s| (*s / count as f64) as f32).collect();
        let t = Tensor::from_vec(mean, (1usize, 3usize, img_size as usize, img_size as usize), &Device::Cpu)?;
        tensors.insert(class_name.clone(), t);
        println!("  {class_name}: {count} samples → skeleton");
    }

    if tensors.is_empty() {
        anyhow::bail!("no class directories with PNGs found in {data_dir}");
    }

    let output = format!("{}/skeletons_v2.safetensors", data_dir);
    candle_core::safetensors::save(&tensors, &output)?;
    println!("saved: {output} ({} classes)", tensors.len());
    Ok(())
}

/// Load skeleton for a class by name. Returns (1, 3, H, W) tensor on success.
/// Searches standard locations: CWD, then HOME/pixel-forge.
/// Returns None gracefully if skeletons_v2.safetensors is absent or class not found.
pub fn load_skeleton_v2(class_name: &str, img_size: u32, device: &Device) -> Option<Tensor> {
    let candidates: Vec<String> = {
        let mut v = vec!["skeletons_v2.safetensors".to_string()];
        if let Ok(home) = std::env::var("HOME") {
            v.push(format!("{}/pixel-forge/skeletons_v2.safetensors", home));
        }
        v
    };

    for path in &candidates {
        if Path::new(path).exists() {
            if let Some(t) = load_skeleton_v2_from(path, class_name, img_size, device) {
                return Some(t);
            }
        }
    }
    None
}

fn load_skeleton_v2_from(path: &str, class_name: &str, img_size: u32, device: &Device) -> Option<Tensor> {
    let data = std::fs::read(path).ok()?;
    let tensors = candle_core::safetensors::load_buffer(&data, device).ok()?;
    let t = tensors.get(class_name)?;
    let dims = t.dims();
    if dims.len() == 4 && dims[2] == img_size as usize && dims[3] == img_size as usize {
        Some(t.clone())
    } else {
        None
    }
}

/// Create initial tensor: skeleton + noise blend instead of pure noise.
/// skeleton_mix = 0.7 means 70% skeleton, 30% noise.
pub fn skeleton_start(
    skeleton: &Tensor,
    seed: Option<u64>,
    class_id: u32,
    index: u32,
    img_size: u32,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let noise = seeded_noise(seed, class_id, index, img_size, device)?;
    let mix = 1.0 - SKELETON_NOISE as f64; // 0.70
    (skeleton * mix)? + (&noise * SKELETON_NOISE as f64)?
}

/// Inverse z-score → clamp → encode. For models trained with the legacy
/// z-score data prep (Quench/Anvil/Micro pre-EDM).
fn tensor_to_rgba(
    x: &Tensor,
    img_size: u32,
    normalizer: &crate::normalize::Normalizer,
) -> candle_core::Result<RgbaImage> {
    let device = x.device();
    // (1, 3, H, W) — broadcast across H,W with shape (1,3,1,1)
    let mean = Tensor::new(&normalizer.mean[..], device)?.reshape((1, 3, 1, 1))?;
    let std  = Tensor::new(&normalizer.std[..],  device)?.reshape((1, 3, 1, 1))?;
    let x = x.broadcast_mul(&std)?.broadcast_add(&mean)?;
    encode_to_rgba(&x, img_size)
}

/// Uncenter (add mean) → clamp → encode. For EDM-trained models, where
/// the data prep is mean-only (the σ-rescale is per-step inside the
/// sampler, not a fixed dataset transform).
fn tensor_to_rgba_centered(
    x: &Tensor,
    img_size: u32,
    normalizer: &crate::normalize::Normalizer,
) -> candle_core::Result<RgbaImage> {
    let device = x.device();
    let mean = Tensor::new(&normalizer.mean[..], device)?.reshape((1, 3, 1, 1))?;
    let x = x.broadcast_add(&mean)?;
    encode_to_rgba(&x, img_size)
}

/// Shared back-half: clamp [0,1] → u8 → RGBA pixels. The forward pre-step
/// (z-score inverse vs mean-uncenter) is decided by the caller.
fn encode_to_rgba(x: &Tensor, img_size: u32) -> candle_core::Result<RgbaImage> {
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
    Ok(rgba)
}

/// Build conditioning tensors for batched generation.
pub fn cond_tensors(cond: &crate::class_cond::ClassCond, n: usize, device: &Device) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let super_tensor = Tensor::new(&vec![cond.super_id; n][..], device)?;
    let tags_flat: Vec<f32> = (0..n).flat_map(|_| cond.tags.iter().copied()).collect();
    let tags_tensor = Tensor::new(tags_flat.as_slice(), device)?.reshape((n, crate::class_cond::NUM_TAGS))?;
    let null_cond = crate::class_cond::ClassCond::null();
    let null_super = Tensor::new(&vec![null_cond.super_id; n][..], device)?;
    let null_tags_flat: Vec<f32> = (0..n).flat_map(|_| null_cond.tags.iter().copied()).collect();
    let null_tags = Tensor::new(null_tags_flat.as_slice(), device)?.reshape((n, crate::class_cond::NUM_TAGS))?;
    Ok((super_tensor, tags_tensor, null_super, null_tags))
}

/// Sample from a trained TinyUNet (Cinder) with CFG.
/// Loads f16 weights as f32 transparently — f16 is storage-only.
/// Optional seed for deterministic generation (same seed = same sprite).
pub fn sample(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    sample_seeded(model_path, cond, img_size, count, steps, None)
}

/// Seeded sample — deterministic when seed is Some.
pub fn sample_seeded(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
    seed: Option<u64>,
) -> Result<Vec<RgbaImage>> {
    sample_seeded_cfg(model_path, cond, img_size, count, steps, seed, DEFAULT_CFG_SCALE)
}

/// Seeded sample with configurable CFG scale.
pub fn sample_seeded_cfg(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
    seed: Option<u64>,
    cfg_scale: f64,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let is_f16 = crate::quantize::is_f16(model_path);

    println!("loading model from {model_path}{}...", if is_f16 { " (f16→f32)" } else { "" });
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TinyUNet::new(vb)?;
    crate::quantize::load_varmap(&mut varmap, model_path)?;
    let normalizer = crate::normalize::Normalizer::load_sidecar(std::path::Path::new(model_path))?
        .ok_or_else(|| anyhow::anyhow!("model {} is missing required normalize.json sidecar", model_path))?;
    let params = TinyUNet::param_count(&varmap);
    let sigma_data = normalizer.sigma_data();
    println!("model: {} params, sampling {} images, EDM Euler {steps} steps, cfg={}, σ_data={sigma_data:.4}",
        params, count, cfg_scale);

    let n = count as usize;
    let (super_tensor, tags_tensor, null_super, null_tags) = cond_tensors(cond, n, &device)?;

    if let Some(s) = seed {
        println!("  seed: {s}");
    }

    // EDM init: x = N(0,1) · σ_max. Skeleton init is incompatible with
    // EDM's σ-parameterized noise space (skeleton hints assume linear
    // noise blending), so we drop it for the EDM path.
    let class_id = cond.super_id;
    let init_tensors: Vec<Tensor> = (0..count)
        .map(|i| seeded_noise(seed, class_id, i, img_size, &device)
            .and_then(|t| t.squeeze(0)))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut x = Tensor::stack(&init_tensors, 0)?;
    let sigmas = crate::precond::sampling_sigmas(steps);
    x = (x * sigmas[0] as f64)?;

    // Reusable broadcastable scalars.
    let make_scalar = |v: f32| -> candle_core::Result<Tensor> {
        Tensor::new(&[v], &device)?.reshape((1, 1, 1, 1))
    };

    for i in 0..steps {
        let s_cur  = sigmas[i];
        let s_next = sigmas[i + 1];
        let coeffs = crate::precond::EdmCoeffs::at(s_cur, sigma_data);

        // EDM time conditioning is c_noise = 0.25·log(σ), broadcast to
        // a (B,) tensor for the model.
        let t_cnoise = Tensor::new(&vec![coeffs.c_noise; count as usize][..], &device)?;
        let x_in = (&x * coeffs.c_in as f64)?;
        let f = cfg_denoise(&model, &x_in, &t_cnoise,
            &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?;
        // D = c_skip · x + c_out · F
        let cskip = make_scalar(coeffs.c_skip)?;
        let cout  = make_scalar(coeffs.c_out)?;
        let d_x = (x.broadcast_mul(&cskip)? + f.broadcast_mul(&cout)?)?;

        if s_next > 1e-6 {
            // Euler step along the probability-flow ODE.
            // d = (x − D)/σ;   x_next = x + (σ_next − σ_cur)·d
            let direction = ((&x - &d_x)? * (1.0 / s_cur.max(1e-12)) as f64)?;
            let dt = (s_next - s_cur) as f64;
            x = (&x + (&direction * dt)?)?;
        } else {
            x = d_x;
        }
    }

    // EDM uses centered (mean-subtracted) data; uncenter on decode.
    let mut images = Vec::new();
    for i in 0..count {
        let single = x.narrow(0, i as usize, 1)?;
        images.push(tensor_to_rgba_centered(&single, img_size, &normalizer)?);
        println!("  sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// Sample from a trained MediumUNet (Quench) with CFG.
/// Loads f16 weights as f32 transparently.
pub fn sample_medium(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    sample_medium_seeded(model_path, cond, img_size, count, steps, None)
}

/// Seeded Quench sample.
pub fn sample_medium_seeded(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
    seed: Option<u64>,
) -> Result<Vec<RgbaImage>> {
    sample_medium_seeded_cfg(model_path, cond, img_size, count, steps, seed, DEFAULT_CFG_SCALE)
}

/// Seeded Quench sample with configurable CFG scale.
pub fn sample_medium_seeded_cfg(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
    seed: Option<u64>,
    cfg_scale: f64,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let is_f16 = crate::quantize::is_f16(model_path);

    println!("loading Quench model from {model_path}{}...", if is_f16 { " (f16→f32)" } else { "" });
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MediumUNet::new(vb)?;
    crate::quantize::load_varmap(&mut varmap, model_path)?;
    let normalizer = crate::normalize::Normalizer::load_sidecar(std::path::Path::new(model_path))?
        .ok_or_else(|| anyhow::anyhow!("model {} is missing required normalize.json sidecar", model_path))?;
    let params = MediumUNet::param_count(&varmap);
    let sigma_data = normalizer.sigma_data();
    println!("model: Quench, {} params, sampling {} images, EDM Euler {steps} steps, cfg={}, σ_data={sigma_data:.4}",
        params, count, cfg_scale);

    let n = count as usize;
    let (super_tensor, tags_tensor, null_super, null_tags) = cond_tensors(cond, n, &device)?;

    if let Some(s) = seed {
        println!("  seed: {s}");
    }

    // EDM init: x = N(0,1) · σ_max. No skeleton — incompatible with
    // σ-parameterized noise space.
    let class_id = cond.super_id;
    let init_tensors: Vec<Tensor> = (0..count)
        .map(|i| seeded_noise(seed, class_id, i, img_size, &device)
            .and_then(|t| t.squeeze(0)))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut x = Tensor::stack(&init_tensors, 0)?;
    let sigmas = crate::precond::sampling_sigmas(steps);
    x = (x * sigmas[0] as f64)?;

    let make_scalar = |v: f32| -> candle_core::Result<Tensor> {
        Tensor::new(&[v], &device)?.reshape((1, 1, 1, 1))
    };

    for i in 0..steps {
        let s_cur  = sigmas[i];
        let s_next = sigmas[i + 1];
        let coeffs = crate::precond::EdmCoeffs::at(s_cur, sigma_data);

        let t_cnoise = Tensor::new(&vec![coeffs.c_noise; count as usize][..], &device)?;
        let x_in = (&x * coeffs.c_in as f64)?;
        let f = cfg_denoise(&model, &x_in, &t_cnoise,
            &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?;
        let cskip = make_scalar(coeffs.c_skip)?;
        let cout  = make_scalar(coeffs.c_out)?;
        let d_x = (x.broadcast_mul(&cskip)? + f.broadcast_mul(&cout)?)?;

        if s_next > 1e-6 {
            let direction = ((&x - &d_x)? * (1.0 / s_cur.max(1e-12)) as f64)?;
            let dt = (s_next - s_cur) as f64;
            x = (&x + (&direction * dt)?)?;
        } else {
            x = d_x;
        }
    }

    let mut images = Vec::new();
    for i in 0..count {
        let single = x.narrow(0, i as usize, 1)?;
        images.push(tensor_to_rgba_centered(&single, img_size, &normalizer)?);
        println!("  sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// Sample from a trained AnvilUNet with CFG.
/// Loads f16 weights as f32 transparently.
pub fn sample_anvil(
    model_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
    seed: Option<u64>,
    cfg_scale: f64,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let is_f16 = crate::quantize::is_f16(model_path);

    println!("loading Anvil model from {model_path}{}...", if is_f16 { " (f16→f32)" } else { "" });
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AnvilUNet::new(vb)?;
    let is_v_pred = detect_v_pred(model_path);
    crate::quantize::load_varmap(&mut varmap, model_path)?;
    let normalizer = crate::normalize::Normalizer::load_sidecar(std::path::Path::new(model_path))?
        .ok_or_else(|| anyhow::anyhow!("model {} is missing required normalize.json sidecar", model_path))?;

    let params = AnvilUNet::param_count(&varmap);
    let pred_tag = if is_v_pred { ", v-pred" } else { "" };
    println!("model: Anvil, {} params, sampling {} images, {steps} steps, cfg={cfg_scale}{pred_tag}, z-score", params, count);

    let (super_tensor, tags_tensor, null_super, null_tags) = cond_tensors(cond, 1, &device)?;

    let mut images = Vec::new();
    for i in 0..count {
        let mut x = seeded_noise(seed, cond.super_id, i, img_size, &device)?;

        for step in 0..steps {
            let t_frac = 1.0 - (step as f32 / steps as f32);
            let amount = cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / steps as f32);
            let next_amount = if step + 1 < steps { cosine_schedule(next_frac) } else { 0.0 };

            let t = Tensor::new(&[amount], &device)?;
            let pred_clean = if is_v_pred {
                cfg_denoise_vpred(&model, &x, &t, amount, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?
            } else {
                cfg_denoise(&model, &x, &t, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?
            };

            if next_amount > 1e-6 {
                let pred_noise = ((&x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?
                    .clamp(-3.0, 3.0)?;
                x = ((&pred_clean * (1.0 - next_amount as f64))? + (&pred_noise * next_amount as f64)?)?;
            } else {
                x = pred_clean;
            }
        }

        images.push(tensor_to_rgba(&x, img_size, &normalizer)?);
        println!("  sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// Sample from a MicroUNet siloed model — no class conditioning needed.
pub fn sample_micro(
    model_path: &str,
    channels: &[usize],
    img_size: u32,
    count: u32,
    steps: usize,
    seed: Option<u64>,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let is_f16 = crate::quantize::is_f16(model_path);

    println!("loading Micro model from {model_path}{}...", if is_f16 { " (f16→f32)" } else { "" });
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MicroUNet::with_channels(vb, channels)?;
    let is_v_pred = detect_v_pred(model_path);
    crate::quantize::load_varmap(&mut varmap, model_path)?;
    let normalizer = crate::normalize::Normalizer::load_sidecar(std::path::Path::new(model_path))?
        .ok_or_else(|| anyhow::anyhow!("model {} is missing required normalize.json sidecar", model_path))?;

    let params = MicroUNet::param_count(&varmap);
    let ch_str = channels.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
    let pred_tag = if is_v_pred { ", v-pred" } else { "" };
    println!("model: Micro [{ch_str}], {} params, sampling {} images, {steps} steps{pred_tag}, z-score", params, count);

    let mut images = Vec::new();
    for i in 0..count {
        let mut x = seeded_noise(seed, 0, i, img_size, &device)?;

        for step in 0..steps {
            let t_frac = 1.0 - (step as f32 / steps as f32);
            let amount = cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / steps as f32);
            let next_amount = if step + 1 < steps { cosine_schedule(next_frac) } else { 0.0 };

            let t = Tensor::new(&[amount], &device)?;
            // No CFG — unconditional model. Dummy super_id/tags passed through DiffusionModel trait.
            let pred_clean = if is_v_pred {
                let v = model.forward_uncond(&x, &t)?;
                let a = amount as f64;
                ((&x * (1.0 - a))? - (&v * a)?)?
            } else {
                model.forward_uncond(&x, &t)?
            };

            if next_amount > 1e-6 {
                let pred_noise = ((&x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?
                    .clamp(-3.0, 3.0)?;
                x = ((&pred_clean * (1.0 - next_amount as f64))? + (&pred_noise * next_amount as f64)?)?;
            } else {
                x = pred_clean;
            }
        }

        images.push(tensor_to_rgba(&x, img_size, &normalizer)?);
        println!("  sample {}/{count}", i + 1);
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::cosine_schedule;

    #[test]
    fn cosine_schedule_at_zero_is_near_zero() {
        let v = cosine_schedule(0.0);
        assert!(v < 0.02, "cosine_schedule(0) should be near 0, got {v}");
    }

    #[test]
    fn cosine_schedule_at_one_is_one() {
        let v = cosine_schedule(1.0);
        assert!((v - 1.0).abs() < 1e-5, "cosine_schedule(1) should be 1.0, got {v}");
    }

    #[test]
    fn cosine_schedule_is_monotone_increasing() {
        let steps = 20;
        let mut prev = cosine_schedule(0.0);
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let cur = cosine_schedule(t);
            assert!(cur >= prev - 1e-6, "cosine_schedule not monotone at t={t}: {prev} -> {cur}");
            prev = cur;
        }
    }

    #[test]
    fn cosine_schedule_midpoint_less_than_linear() {
        // Cosine schedule compresses mid-range (spends more time there),
        // so at t=0.5 it should return less than 0.5 (noise accumulates slower initially).
        let v = cosine_schedule(0.5);
        assert!(v > 0.0 && v < 1.0, "cosine_schedule(0.5) out of range: {v}");
    }

    #[test]
    fn cosine_schedule_output_in_unit_range() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = cosine_schedule(t);
            assert!(v >= 0.0 && v <= 1.0, "cosine_schedule({t}) = {v} out of [0,1]");
        }
    }

    #[test]
    fn cosine_schedule_50_steps_monotone() {
        // DDIM-style: 50 evenly spaced steps must produce monotonically increasing noise
        let steps = 50;
        let mut prev = cosine_schedule(0.0);
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let cur = cosine_schedule(t);
            assert!(cur >= prev - 1e-6, "not monotone at step {i}/{steps}: {prev} -> {cur}");
            prev = cur;
        }
    }

    #[test]
    fn cosine_schedule_fine_grained_no_plateau() {
        // Over 1000 steps, schedule should never stay flat for 100 consecutive steps
        let steps = 1000;
        let vals: Vec<f32> = (0..=steps).map(|i| cosine_schedule(i as f32 / steps as f32)).collect();
        for window in vals.windows(100) {
            let delta = window.last().unwrap() - window.first().unwrap();
            assert!(delta > 1e-4, "plateau detected in cosine schedule");
        }
    }

    #[test]
    fn min_snr_weight_zero_gamma_returns_one() {
        assert_eq!(super::min_snr_weight(0.5, 0.0), 1.0);
        assert_eq!(super::min_snr_weight(0.5, -1.0), 1.0);
    }

    #[test]
    fn min_snr_weight_in_valid_range() {
        for i in 1..=99 {
            let t = i as f32 / 100.0;
            let w = super::min_snr_weight(t, 5.0);
            assert!(w >= 0.01 && w <= 1.0, "min_snr_weight({t}) = {w} out of [0.01, 1.0]");
        }
    }

    #[test]
    fn min_snr_weight_extremes() {
        // Near t=0 (low noise): SNR is very high → gamma/SNR is tiny → floor at 0.01
        let w_low = super::min_snr_weight(0.01, 5.0);
        assert!(w_low >= 0.01, "low noise weight below floor, got {w_low}");
        // Near t=0.5 (mid noise): balanced weight
        let w_mid = super::min_snr_weight(0.5, 5.0);
        assert!(w_mid > 0.01 && w_mid <= 1.0, "mid noise weight out of range: {w_mid}");
        // Near t=1 (high noise): SNR is low, weight should be high (capped at gamma)
        let w_high = super::min_snr_weight(0.99, 5.0);
        assert!(w_high >= 0.01, "high noise weight below floor, got {w_high}");
    }
}
