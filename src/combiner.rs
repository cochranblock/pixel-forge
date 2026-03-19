// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! SlotGridTransformer — arranges sprites into 8×8 scene grids.
//! ~61,792 params (~242 KB). 3-layer transformer with multi-head self-attention.
//!
//! Why transformer: the task is about relationships between cells, not local texture.
//! Self-attention over 64 tokens (trivial compute) captures "if terrain is at (7,3)
//! then a character at (6,3) is valid." A CNN would need many layers for that
//! receptive field.
//!
//! Training: masked prediction — randomly mask 20-50% of cells, predict masked class_ids.
//! Loss: cross-entropy over 16 classes per cell.
//!
//! Multi-head attention implemented from scratch (~60 lines).
//! Reference: Candle BERT/GPT-2 attention patterns.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use rand::Rng;
use rand::seq::SliceRandom;

use crate::scene::{self, SceneGrid, CELL_DIM, GRID_CELLS, TOTAL_CLASSES};

/// Transformer hidden dimension.
const D_MODEL: usize = 48;
/// Number of attention heads.
const N_HEADS: usize = 4;
/// FFN intermediate dimension.
const D_FFN: usize = 96;
/// Number of transformer blocks.
const N_LAYERS: usize = 3;
/// Head dimension.
const D_HEAD: usize = D_MODEL / N_HEADS; // 12

// ─── Multi-Head Self-Attention ───

struct MultiHeadAttention {
    wq: nn::Linear,
    wk: nn::Linear,
    wv: nn::Linear,
    wo: nn::Linear,
}

impl MultiHeadAttention {
    fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            wq: nn::linear(D_MODEL, D_MODEL, vb.pp("wq"))?,
            wk: nn::linear(D_MODEL, D_MODEL, vb.pp("wk"))?,
            wv: nn::linear(D_MODEL, D_MODEL, vb.pp("wv"))?,
            wo: nn::linear(D_MODEL, D_MODEL, vb.pp("wo"))?,
        })
    }

    /// x: (B, T, D_MODEL) → (B, T, D_MODEL)
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, t, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.wq.forward(x)?; // (B, T, D_MODEL)
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape to (B, N_HEADS, T, D_HEAD)
        let q = q.reshape((b, t, N_HEADS, D_HEAD))?.permute((0, 2, 1, 3))?;
        let k = k.reshape((b, t, N_HEADS, D_HEAD))?.permute((0, 2, 1, 3))?;
        let v = v.reshape((b, t, N_HEADS, D_HEAD))?.permute((0, 2, 1, 3))?;

        // Scaled dot-product attention
        let scale = (D_HEAD as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?.broadcast_div(
            &Tensor::new(scale, x.device())?.to_dtype(x.dtype())?
        )?; // (B, N_HEADS, T, T)
        let attn = nn::ops::softmax(&scores, 3)?; // (B, N_HEADS, T, T)
        let out = attn.matmul(&v)?; // (B, N_HEADS, T, D_HEAD)

        // Reshape back to (B, T, D_MODEL)
        let out = out.permute((0, 2, 1, 3))?.reshape((b, t, D_MODEL))?;
        self.wo.forward(&out)
    }
}

// ─── FFN ───

struct FeedForward {
    up: nn::Linear,
    down: nn::Linear,
}

impl FeedForward {
    fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            up: nn::linear(D_MODEL, D_FFN, vb.pp("up"))?,
            down: nn::linear(D_FFN, D_MODEL, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h = nn::ops::silu(&self.up.forward(x)?)?;
        self.down.forward(&h)
    }
}

// ─── Transformer Block ───

struct TransformerBlock {
    ln1: nn::LayerNorm,
    attn: MultiHeadAttention,
    ln2: nn::LayerNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    fn new(vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            ln1: nn::layer_norm(D_MODEL, nn::LayerNormConfig::default(), vb.pp("ln1"))?,
            attn: MultiHeadAttention::new(vb.pp("attn"))?,
            ln2: nn::layer_norm(D_MODEL, nn::LayerNormConfig::default(), vb.pp("ln2"))?,
            ffn: FeedForward::new(vb.pp("ffn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        // Pre-norm attention + residual
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = (x + &h)?;
        // Pre-norm FFN + residual
        let h = self.ln2.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        &x + &h
    }
}

// ─── SlotGridTransformer ───

/// 3-layer transformer that predicts cell class IDs from context.
///
/// Input: (B, 64, 19) — encoded scene grid cells
/// Output: (B, 64, 16) — logits per cell over 16 classes (15 + empty)
pub struct SlotGridTransformer {
    input_proj: nn::Linear,
    pos_emb: Tensor, // learned (64, D_MODEL) position embeddings
    blocks: Vec<TransformerBlock>,
    ln_out: nn::LayerNorm,
    head: nn::Linear,
}

impl SlotGridTransformer {
    pub fn new(vb: VarBuilder) -> CResult<Self> {
        let input_proj = nn::linear(CELL_DIM, D_MODEL, vb.pp("in_proj"))?;

        // Learned position embeddings
        let pos_emb = vb.get((GRID_CELLS, D_MODEL), "pos_emb")?;

        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            blocks.push(TransformerBlock::new(vb.pp(format!("block{i}")))?);
        }

        let ln_out = nn::layer_norm(D_MODEL, nn::LayerNormConfig::default(), vb.pp("ln_out"))?;
        let head = nn::linear(D_MODEL, TOTAL_CLASSES, vb.pp("head"))?;

        Ok(Self { input_proj, pos_emb, blocks, ln_out, head })
    }

    /// Forward pass.
    /// x: (B, 64, CELL_DIM) → (B, 64, TOTAL_CLASSES) logits
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, t, _) = x.dims3()?;

        // Project input + add positional embedding
        let mut h = self.input_proj.forward(x)?;
        let pos = self.pos_emb.unsqueeze(0)?.expand((b, t, D_MODEL))?;
        h = (&h + &pos)?;

        // Transformer blocks
        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        // Output projection
        let h = self.ln_out.forward(&h)?;
        self.head.forward(&h)
    }

    /// Predict class distribution for all cells. Returns (B, 64, 16) probabilities.
    pub fn predict_probs(&self, x: &Tensor, temperature: f32) -> CResult<Tensor> {
        let logits = self.forward(x)?;
        let scaled = (&logits / temperature as f64)?;
        nn::ops::softmax(&scaled, 2)
    }

    pub fn param_count(varmap: &VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}

// ─── Training ───

/// Training config for the Combiner.
pub struct CombinerTrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Fraction of cells to mask (0.2 to 0.5).
    pub mask_frac_min: f32,
    pub mask_frac_max: f32,
}

impl Default for CombinerTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 16,
            lr: 1e-3,
            mask_frac_min: 0.2,
            mask_frac_max: 0.5,
        }
    }
}

/// Train result with diagnostics.
pub struct CombinerTrainResult {
    pub epochs: usize,
    pub final_loss: f32,
    pub final_accuracy: f32,
    pub duration_ms: u128,
}

impl std::fmt::Display for CombinerTrainResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "combiner trained: {} epochs, loss={:.4}, masked_acc={:.1}%, {}ms",
            self.epochs, self.final_loss, self.final_accuracy * 100.0, self.duration_ms)
    }
}

/// Train Combiner on scene data using masked prediction.
///
/// For each scene:
/// 1. Randomly mask 20-50% of cells (set to empty)
/// 2. Feed masked grid through transformer
/// 3. Cross-entropy loss on masked cell predictions vs original class IDs
pub fn train_combiner(
    scenes: &[SceneGrid],
    config: &CombinerTrainConfig,
    device: &Device,
) -> Result<(VarMap, SlotGridTransformer, CombinerTrainResult)> {
    let t0 = std::time::Instant::now();
    let n = scenes.len();
    if n < 10 {
        anyhow::bail!("need at least 10 scenes for Combiner training, have {}", n);
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = SlotGridTransformer::new(vb)?;
    let params = SlotGridTransformer::param_count(&varmap);
    println!("combiner: {} params, training on {} scenes", params, n);

    let mut opt = nn::AdamW::new(varmap.all_vars(), nn::ParamsAdamW {
        lr: config.lr,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    let mut rng = rand::thread_rng();
    let mut final_loss = 0.0f32;
    let mut final_acc = 0.0f32;

    for epoch in 0..config.epochs {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0f64;
        let mut epoch_correct = 0usize;
        let mut epoch_masked_total = 0usize;
        let mut batches = 0;

        for chunk in indices.chunks(config.batch_size) {
            let bs = chunk.len();

            // Build masked batch
            let mask_frac = rng.r#gen_range(config.mask_frac_min..=config.mask_frac_max);
            let mut input_enc = Vec::with_capacity(bs * GRID_CELLS * CELL_DIM);
            let mut target_ids = Vec::with_capacity(bs * GRID_CELLS);
            let mut mask_flags = Vec::with_capacity(bs * GRID_CELLS);

            for &idx in chunk {
                let scene = &scenes[idx];
                let (masked, masked_indices) = scene.masked(mask_frac);

                // Input: masked grid encoding
                input_enc.extend_from_slice(&masked.encode());

                // Target: original class IDs for ALL cells
                for cell in &scene.cells {
                    target_ids.push(cell.class_id as i64);
                }

                // Mask bitmap: 1.0 for masked cells, 0.0 for visible
                let mask_set: std::collections::HashSet<usize> = masked_indices.into_iter().collect();
                for i in 0..GRID_CELLS {
                    mask_flags.push(if mask_set.contains(&i) { 1.0f32 } else { 0.0f32 });
                }
            }

            let x = Tensor::new(input_enc.as_slice(), device)?
                .reshape((bs, GRID_CELLS, CELL_DIM))?;
            let targets = Tensor::new(target_ids.as_slice(), device)?
                .reshape((bs * GRID_CELLS,))?;
            let mask = Tensor::new(mask_flags.as_slice(), device)?
                .reshape((bs * GRID_CELLS,))?;

            // Forward: (B, 64, 16) logits
            let logits = model.forward(&x)?;
            let logits_flat = logits.reshape((bs * GRID_CELLS, TOTAL_CLASSES))?;

            // Cross-entropy only on masked cells (weighted by mask)
            let loss = masked_cross_entropy(&logits_flat, &targets, &mask)?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()? as f64;

            // Accuracy on masked cells
            let preds = logits_flat.argmax(1)?.to_vec1::<u32>()?;
            let targets_vec = targets.to_vec1::<i64>()?;
            let mask_vec = mask.to_vec1::<f32>()?;
            for i in 0..preds.len() {
                if mask_vec[i] > 0.5 {
                    epoch_masked_total += 1;
                    if preds[i] as i64 == targets_vec[i] {
                        epoch_correct += 1;
                    }
                }
            }
            batches += 1;
        }

        final_loss = if batches > 0 { (epoch_loss / batches as f64) as f32 } else { 0.0 };
        final_acc = if epoch_masked_total > 0 {
            epoch_correct as f32 / epoch_masked_total as f32
        } else { 0.0 };

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            println!("  epoch {}/{}: loss={:.4} masked_acc={:.1}%",
                epoch + 1, config.epochs, final_loss, final_acc * 100.0);
        }
    }

    let result = CombinerTrainResult {
        epochs: config.epochs,
        final_loss,
        final_accuracy: final_acc,
        duration_ms: t0.elapsed().as_millis(),
    };

    Ok((varmap, model, result))
}

/// Cross-entropy loss masked to only count specific cells.
/// logits: (N, C), targets: (N,) as i64, mask: (N,) as f32 {0.0, 1.0}
fn masked_cross_entropy(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> CResult<Tensor> {

    // Log-softmax
    let log_probs = nn::ops::log_softmax(logits, 1)?; // (N, C)

    // Gather target log-probs
    let targets_expanded = targets.to_dtype(DType::U32)?.unsqueeze(1)?; // (N, 1)
    let target_log_probs = log_probs.gather(&targets_expanded, 1)?.squeeze(1)?; // (N,)

    // Negate and mask
    let per_sample_loss = target_log_probs.neg()?; // (N,)
    let masked_loss = per_sample_loss.broadcast_mul(mask)?; // (N,)

    // Average over masked cells only
    let mask_sum = mask.sum_all()?;
    let total_loss = masked_loss.sum_all()?;
    total_loss.broadcast_div(&mask_sum.maximum(&Tensor::new(1.0f32, mask.device())?)?)
}

// ─── Inference: autoregressive scene generation ───

/// Generate a complete scene using the Combiner.
///
/// Strategy: bottom-to-top, left-to-right (terrain first, sky last).
/// For each empty cell, predict class distribution and sample.
pub fn generate_scene(
    model: &SlotGridTransformer,
    seed: Option<&SceneGrid>,
    temperature: f32,
    device: &Device,
) -> Result<SceneGrid> {
    let mut grid = seed.cloned().unwrap_or_else(SceneGrid::empty);
    let mut rng = rand::thread_rng();

    // Fill order: bottom-to-top, left-to-right
    let mut fill_order: Vec<(usize, usize)> = Vec::with_capacity(GRID_CELLS);
    for row in (0..scene::GRID_H).rev() {
        for col in 0..scene::GRID_W {
            fill_order.push((col, row));
        }
    }

    for &(col, row) in &fill_order {
        // Skip pre-seeded cells
        if !grid.get(col, row).is_empty() { continue; }

        // Encode current grid state
        let enc = grid.encode();
        let x = Tensor::new(enc.as_slice(), device)?
            .reshape((1, GRID_CELLS, CELL_DIM))?;

        // Get logits for this cell
        let logits = model.forward(&x)?; // (1, 64, 16)
        let cell_idx = row * scene::GRID_W + col;
        let cell_logits = logits.i((0, cell_idx))?.to_vec1::<f32>()?; // (16,)

        // Sample from distribution
        let class_id = sample_with_temperature(&cell_logits, temperature, &mut rng);
        grid.set(col, row, class_id);
    }

    Ok(grid)
}

/// Temperature-scaled softmax sampling.
fn sample_with_temperature(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    let t = temperature.max(0.01);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / t).exp()).collect();
    let sum: f32 = exp.iter().sum();

    let mut r: f32 = rng.r#gen();
    for (i, &e) in exp.iter().enumerate() {
        r -= e / sum;
        if r <= 0.0 { return i as u32; }
    }
    (logits.len() - 1) as u32
}

/// Load a trained Combiner from disk.
pub fn load_combiner(path: &str, device: &Device) -> Result<(VarMap, SlotGridTransformer)> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = SlotGridTransformer::new(vb)?;
    varmap.load(path)?;
    Ok((varmap, model))
}

/// Save Combiner weights (atomic).
pub fn save_combiner(varmap: &VarMap, path: &str) -> Result<()> {
    let target = std::path::PathBuf::from(path);
    let tmp = target.with_extension("tmp");
    varmap.save(&tmp)?;
    std::fs::rename(&tmp, &target)?;
    Ok(())
}
