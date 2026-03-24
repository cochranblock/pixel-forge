// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! MicroClassifier — 4-layer CNN binary classifier for sprite quality.
//! ~16,849 params (~66 KB). Learns user taste from swipe data.
//!
//! Architecture: Conv→SiLU→Conv→GN→SiLU→Conv→GN→SiLU→Conv→GN→SiLU→Pool→Linear→Sigmoid
//! Full fine-tune every 5 swipes (LoRA is overkill at 16K params).
//! Retrains from scratch on the full swipe buffer — sub-second on Metal.
//!
//! Confidence calibration: raw sigmoid output is calibrated via temperature
//! scaling so that "score > 0.5" means the Judge actually prefers it.

use anyhow::Result;
use candle_core::{DType, Device, Module, Result as CResult, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use rand::seq::SliceRandom;

use crate::swipe_store::SwipeStore;

/// Number of training epochs when retraining from scratch.
const RETRAIN_EPOCHS: usize = 30;
/// Batch size for Judge training.
const BATCH_SIZE: usize = 16;
/// Learning rate — aggressive since we retrain from scratch each time.
const LR: f64 = 0.01;
/// Minimum swipes before the Judge activates.
const MIN_SWIPES: usize = 10;
/// Score threshold for accepting a sprite.
const ACCEPT_THRESHOLD: f32 = 0.5;

/// 4-layer CNN binary classifier.
///
/// ```text
/// Conv2d(3→16, k3, pad1)    32×32 → 32×32    448 params
/// SiLU
/// Conv2d(16→16, k3, s2, p1) 32×32 → 16×16  2,320 params
/// GroupNorm(4, 16) + SiLU                       32 params
/// Conv2d(16→32, k3, s2, p1) 16×16 → 8×8    4,640 params
/// GroupNorm(8, 32) + SiLU                       64 params
/// Conv2d(32→32, k3, s2, p1) 8×8 → 4×4      9,248 params
/// GroupNorm(8, 32) + SiLU                       64 params
/// AdaptiveAvgPool → (B, 32)
/// Linear(32→1) + sigmoid                        33 params
///                                     Total: ~16,849 params
/// ```
pub struct MicroClassifier {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    conv3: nn::Conv2d,
    conv4: nn::Conv2d,
    gn2: nn::GroupNorm,
    gn3: nn::GroupNorm,
    gn4: nn::GroupNorm,
    head: nn::Linear,
    /// Temperature for calibrated confidence (learned post-hoc)
    temperature: f32,
}

impl MicroClassifier {
    pub fn new(vb: VarBuilder) -> CResult<Self> {
        let conv1 = nn::conv2d(3, 16, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c1"))?;
        let conv2 = nn::conv2d(16, 16, 3, nn::Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("c2"))?;
        let conv3 = nn::conv2d(16, 32, 3, nn::Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("c3"))?;
        let conv4 = nn::conv2d(32, 32, 3, nn::Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("c4"))?;
        let gn2 = nn::group_norm(4, 16, 1e-5, vb.pp("gn2"))?;
        let gn3 = nn::group_norm(8, 32, 1e-5, vb.pp("gn3"))?;
        let gn4 = nn::group_norm(8, 32, 1e-5, vb.pp("gn4"))?;
        let head = nn::linear(32, 1, vb.pp("head"))?;

        Ok(Self { conv1, conv2, conv3, conv4, gn2, gn3, gn4, head, temperature: 1.0 })
    }

    /// Forward pass → raw logit (pre-sigmoid). Use `score()` for calibrated output.
    pub fn logit(&self, x: &Tensor) -> CResult<Tensor> {
        // Layer 1: Conv(3→16) + SiLU
        let h = nn::ops::silu(&self.conv1.forward(x)?)?;
        // Layer 2: Conv(16→16, stride 2) + GN + SiLU
        let h = nn::ops::silu(&self.gn2.forward(&self.conv2.forward(&h)?)?)?;
        // Layer 3: Conv(16→32, stride 2) + GN + SiLU
        let h = nn::ops::silu(&self.gn3.forward(&self.conv3.forward(&h)?)?)?;
        // Layer 4: Conv(32→32, stride 2) + GN + SiLU
        let h = nn::ops::silu(&self.gn4.forward(&self.conv4.forward(&h)?)?)?;

        // Global average pool: (B, 32, 4, 4) → (B, 32)
        let (_, _, _fh, _fw) = h.dims4()?;
        let h = h.mean_keepdim(3)?.mean_keepdim(2)?.squeeze(3)?.squeeze(2)?;

        // Linear → scalar logit
        self.head.forward(&h)
    }

    /// Calibrated probability score in [0, 1].
    pub fn score(&self, x: &Tensor) -> CResult<Tensor> {
        let logit = self.logit(x)?;
        let scaled = (&logit / self.temperature as f64)?;
        nn::ops::sigmoid(&scaled)
    }

    /// Score a single sprite. Returns calibrated probability.
    pub fn score_one(&self, pixels: &[f32], device: &Device) -> CResult<f32> {
        let x = Tensor::new(pixels, device)?.reshape((1, 3, 16, 16))?;
        let s = self.score(&x)?;
        s.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()
    }

    /// Batch score multiple sprites. Returns Vec of calibrated probabilities.
    pub fn score_batch(&self, pixels: &[f32], count: usize, device: &Device) -> CResult<Vec<f32>> {
        let x = Tensor::new(pixels, device)?.reshape((count, 3, 16, 16))?;
        let scores = self.score(&x)?.squeeze(1)?;
        scores.to_vec1::<f32>()
    }

    /// Filter sprites: returns indices of accepted sprites (score > threshold).
    pub fn filter(&self, pixels: &[f32], count: usize, device: &Device) -> CResult<Vec<usize>> {
        let scores = self.score_batch(pixels, count, device)?;
        Ok(scores.iter().enumerate()
            .filter(|&(_, s)| *s > ACCEPT_THRESHOLD)
            .map(|(i, _)| i)
            .collect())
    }

    /// Set calibration temperature (from post-hoc calibration).
    pub fn set_temperature(&mut self, t: f32) {
        self.temperature = t.max(0.1); // prevent division by near-zero
    }

    /// Count parameters.
    pub fn param_count(varmap: &VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}

/// Training result with diagnostics.
pub struct TrainResult {
    pub epochs: usize,
    pub final_loss: f32,
    pub final_accuracy: f32,
    pub calibration_temp: f32,
    pub duration_ms: u128,
}

impl std::fmt::Display for TrainResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "judge trained: {} epochs, loss={:.4}, acc={:.1}%, temp={:.2}, {}ms",
            self.epochs, self.final_loss, self.final_accuracy * 100.0,
            self.calibration_temp, self.duration_ms)
    }
}

/// Train the Judge from scratch on the full swipe buffer.
///
/// Returns a fresh (VarMap, MicroClassifier) pair + training diagnostics.
/// Caller is responsible for saving the varmap.
pub fn train_judge(store: &SwipeStore, device: &Device) -> Result<(VarMap, MicroClassifier, TrainResult)> {
    let n = store.len();
    if n < MIN_SWIPES {
        anyhow::bail!("need at least {} swipes to train Judge, have {}", MIN_SWIPES, n);
    }

    let t0 = std::time::Instant::now();
    let (all_pixels, all_labels) = store.training_data();

    // Split 80/20 for train/val (shuffle first)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rand::thread_rng());
    let val_count = (n / 5).max(2); // at least 2 validation samples
    let train_count = n - val_count;
    let train_idx = &indices[..train_count];
    let val_idx = &indices[train_count..];

    // Fresh model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mut model = MicroClassifier::new(vb)?;
    let params = MicroClassifier::param_count(&varmap);
    println!("judge: {} params, training on {} samples ({} train, {} val)",
        params, n, train_count, val_count);

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: LR,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    let stride = crate::swipe_store::PIXEL_STRIDE;
    let mut best_val_loss = f32::MAX;
    let mut patience = 0;
    let mut final_loss = 0.0f32;
    let mut final_acc = 0.0f32;
    let mut actual_epochs = 0;

    for epoch in 0..RETRAIN_EPOCHS {
        // --- Train ---
        let mut shuffled_train: Vec<usize> = train_idx.to_vec();
        shuffled_train.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0f64;
        let mut batches = 0;

        for chunk in shuffled_train.chunks(BATCH_SIZE) {
            let bs = chunk.len();
            let mut batch_px = Vec::with_capacity(bs * stride);
            let mut batch_lbl = Vec::with_capacity(bs);

            for &idx in chunk {
                let start = idx * stride;
                batch_px.extend_from_slice(&all_pixels[start..start + stride]);
                batch_lbl.push(all_labels[idx]);
            }

            let x = Tensor::new(batch_px.as_slice(), device)?.reshape((bs, 3, 16, 16))?;
            let y = Tensor::new(batch_lbl.as_slice(), device)?.reshape((bs, 1))?;

            let logits = model.logit(&x)?;
            let loss = binary_cross_entropy_with_logits(&logits, &y)?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batches += 1;
        }

        let avg_train_loss = if batches > 0 { epoch_loss / batches as f64 } else { 0.0 };

        // --- Validation ---
        let (val_loss, val_acc) = evaluate(&model, &all_pixels, &all_labels, val_idx, device)?;
        actual_epochs = epoch + 1;
        final_loss = avg_train_loss as f32;
        final_acc = val_acc;

        // Early stopping with patience 5
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience = 0;
        } else {
            patience += 1;
            if patience >= 5 {
                println!("  early stop at epoch {} (val_loss={:.4})", epoch + 1, val_loss);
                break;
            }
        }
    }

    // Post-hoc temperature calibration on validation set
    let temp = calibrate_temperature(&model, &all_pixels, &all_labels, val_idx, device)?;
    model.set_temperature(temp);

    let result = TrainResult {
        epochs: actual_epochs,
        final_loss,
        final_accuracy: final_acc,
        calibration_temp: temp,
        duration_ms: t0.elapsed().as_millis(),
    };

    Ok((varmap, model, result))
}

/// Evaluate on a subset: returns (avg_loss, accuracy).
fn evaluate(
    model: &MicroClassifier,
    all_pixels: &[f32],
    all_labels: &[f32],
    indices: &[usize],
    device: &Device,
) -> Result<(f32, f32)> {
    let stride = crate::swipe_store::PIXEL_STRIDE;
    let n = indices.len();
    if n == 0 { return Ok((0.0, 0.0)); }

    let mut px = Vec::with_capacity(n * stride);
    let mut lbl = Vec::with_capacity(n);
    for &idx in indices {
        let start = idx * stride;
        px.extend_from_slice(&all_pixels[start..start + stride]);
        lbl.push(all_labels[idx]);
    }

    let x = Tensor::new(px.as_slice(), device)?.reshape((n, 3, 16, 16))?;
    let y = Tensor::new(lbl.as_slice(), device)?.reshape((n, 1))?;

    let logits = model.logit(&x)?;
    let loss = binary_cross_entropy_with_logits(&logits, &y)?;
    let loss_val = loss.to_scalar::<f32>()?;

    // Accuracy: threshold at 0.0 (logit space)
    let preds = logits.to_vec2::<f32>()?;
    let correct = preds.iter().zip(lbl.iter())
        .filter(|(p, l)| {
            let pred_pos = p[0] > 0.0;
            let true_pos = **l > 0.5;
            pred_pos == true_pos
        })
        .count();

    Ok((loss_val, correct as f32 / n as f32))
}

/// BCE with logits — numerically stable.
/// loss = max(logit, 0) - logit * target + log(1 + exp(-|logit|))
fn binary_cross_entropy_with_logits(logits: &Tensor, targets: &Tensor) -> CResult<Tensor> {
    let zeros = logits.zeros_like()?;
    let pos_part = logits.maximum(&zeros)?;
    let neg_abs = logits.abs()?.neg()?;
    let log_term = neg_abs.exp()?.broadcast_add(&Tensor::new(1.0f32, logits.device())?)?.log()?;
    let loss = ((&pos_part - logits.broadcast_mul(targets)?)? + log_term)?;
    loss.mean_all()
}

/// Post-hoc temperature calibration using grid search on validation NLL.
/// Finds T that makes sigmoid(logit/T) best match actual label distribution.
fn calibrate_temperature(
    model: &MicroClassifier,
    all_pixels: &[f32],
    all_labels: &[f32],
    val_idx: &[usize],
    device: &Device,
) -> Result<f32> {
    let stride = crate::swipe_store::PIXEL_STRIDE;
    let n = val_idx.len();
    if n < 4 { return Ok(1.0); }

    // Collect validation logits
    let mut px = Vec::with_capacity(n * stride);
    for &idx in val_idx {
        let start = idx * stride;
        px.extend_from_slice(&all_pixels[start..start + stride]);
    }
    let x = Tensor::new(px.as_slice(), device)?.reshape((n, 3, 16, 16))?;
    let logits = model.logit(&x)?.squeeze(1)?.to_vec1::<f32>()?;
    let labels: Vec<f32> = val_idx.iter().map(|&i| all_labels[i]).collect();

    // Grid search T in [0.1, 5.0] — find T that minimizes NLL
    let mut best_t = 1.0f32;
    let mut best_nll = f32::MAX;
    for step in 1..=50 {
        let t = step as f32 * 0.1;
        let nll: f32 = logits.iter().zip(labels.iter()).map(|(&logit, &label)| {
            let p = sigmoid(logit / t);
            let p = p.clamp(1e-7, 1.0 - 1e-7);
            -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
        }).sum::<f32>() / n as f32;

        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
    }

    Ok(best_t)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Load a trained Judge from disk.
pub fn load_judge(path: &str, device: &Device) -> Result<(VarMap, MicroClassifier)> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MicroClassifier::new(vb)?;
    varmap.load(path)?;
    Ok((varmap, model))
}

/// Save Judge weights.
pub fn save_judge(varmap: &VarMap, path: &str) -> Result<()> {
    let target = std::path::PathBuf::from(path);
    let tmp = target.with_extension("tmp");
    varmap.save(&tmp)?;
    std::fs::rename(&tmp, &target)?;
    Ok(())
}
