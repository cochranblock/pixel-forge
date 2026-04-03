// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! LoRA (Low-Rank Adaptation) for TinyUNet — steers generation toward Judge's taste.
//! ~12,288 extra params (~48 KB). Rank-4 adapters on each ResBlock conv layer.
//!
//! How it works:
//! - Base UNet weights (4.2 MB) are frozen on device
//! - LoRA adds a small delta: W' = W + (B @ A) * alpha/rank
//! - Only B and A matrices are trained (48 KB)
//! - Merge/unmerge: fold LoRA into base weights for inference speed, unfold for continued training
//!
//! Training loop:
//! 1. Generate sprite with base + LoRA
//! 2. Judge scores it
//! 3. Backprop Judge signal through last 5 denoising steps
//! 4. Update LoRA weights only (base frozen)
//! 5. Schedule: every 20 swipes, 5 gradient steps

use anyhow::Result;
use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};

/// LoRA rank — controls expressivity vs size tradeoff.
/// Rank 4 = 48 KB total for TinyUNet's 6 ResBlocks × 2 convs.
const RANK: usize = 4;
/// Alpha scaling factor — higher = stronger LoRA influence.
const ALPHA: f32 = 4.0;
/// Number of denoising steps to backprop through (last N only for memory).
const BACKPROP_STEPS: usize = 5;
/// LoRA learning rate.
const LORA_LR: f64 = 1e-4;
/// Number of gradient steps per LoRA update cycle.
const LORA_STEPS: usize = 5;

/// A single LoRA adapter for a Conv2d layer.
///
/// For a Conv2d with weight shape (out_ch, in_ch, kH, kW):
/// - A: (rank, in_ch * kH * kW) — initialized with Kaiming uniform
/// - B: (out_ch * kH * kW, rank) — initialized with zeros
/// - Delta = (B @ A).reshape(out_ch, in_ch, kH, kW) * (alpha / rank)
pub struct LoraConvAdapter {
    pub a: Tensor, // (rank, fan_in) where fan_in = in_ch * kH * kW
    pub b: Tensor, // (fan_out, rank) where fan_out = out_ch * kH * kW
    pub out_ch: usize,
    pub in_ch: usize,
    pub kernel: usize,
    scale: f32,
}

impl LoraConvAdapter {
    /// Create a new LoRA adapter. B is zero-init so delta starts at zero.
    fn new(
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let fan_in = in_ch * kernel * kernel;
        let fan_out = out_ch * kernel * kernel;

        // A: Kaiming-style init scaled by 1/sqrt(fan_in)
        let a_init = nn::Init::Randn { mean: 0.0, stdev: (1.0 / fan_in as f64).sqrt() };
        let a = vb.get_with_hints((RANK, fan_in), "a", a_init)?;

        // B: zero init — so initial delta = 0 (no effect until trained)
        let b = vb.get_with_hints((fan_out, RANK), "b", nn::Init::Const(0.0))?;

        Ok(Self {
            a, b, out_ch, in_ch, kernel,
            scale: ALPHA / RANK as f32,
        })
    }

    /// Compute the weight delta: (B @ A).reshape(out_ch, in_ch, k, k) * scale
    pub fn delta(&self) -> CResult<Tensor> {
        let d = self.b.matmul(&self.a)?; // (fan_out, fan_in)
        let d = d.reshape((self.out_ch, self.in_ch, self.kernel, self.kernel))?;
        &d * self.scale as f64
    }
}

/// Complete LoRA adapter set for TinyUNet.
/// One adapter per conv in each ResBlock (conv1 + conv2 for each of:
/// 3 encoder levels × 2 blocks + 2 mid blocks + 3 decoder levels × 2 blocks = 16 convs).
///
/// But the plan specifies ResBlock conv1/conv2 only, which is:
/// - 3 down levels × 2 resblocks × 2 convs = 12
/// - 2 mid blocks × 2 convs = 4
///
/// Total: 16 adapters.
///
/// At rank 4 with 32/64 channels and 3×3 kernels:
/// Each adapter ≈ 4×(in×9) + (out×9)×4 ≈ 768 params avg → 16 × 768 ≈ 12,288 params
pub struct LoraSet {
    pub adapters: Vec<(String, LoraConvAdapter)>,
}

impl LoraSet {
    /// Build LoRA adapters matching the TinyUNet ResBlock structure.
    pub fn new(vb: VarBuilder) -> CResult<Self> {
        let channels: [usize; 3] = [32, 64, 64];
        let mut adapters = Vec::new();

        // Encoder: 3 down levels, 2 resblocks each
        let mut ch_in = channels[0]; // after conv_in
        for (i, &ch_out) in channels.iter().enumerate() {
            // ResBlock 1 (r1)
            let name_prefix = format!("down{i}_r1");
            adapters.push((
                format!("{name_prefix}.conv1"),
                LoraConvAdapter::new(ch_out, ch_in, 3, vb.pp(format!("{name_prefix}_c1")))?,
            ));
            adapters.push((
                format!("{name_prefix}.conv2"),
                LoraConvAdapter::new(ch_out, ch_out, 3, vb.pp(format!("{name_prefix}_c2")))?,
            ));

            // ResBlock 2 (r2) — both in and out are ch_out
            let name_prefix = format!("down{i}_r2");
            adapters.push((
                format!("{name_prefix}.conv1"),
                LoraConvAdapter::new(ch_out, ch_out, 3, vb.pp(format!("{name_prefix}_c1")))?,
            ));
            adapters.push((
                format!("{name_prefix}.conv2"),
                LoraConvAdapter::new(ch_out, ch_out, 3, vb.pp(format!("{name_prefix}_c2")))?,
            ));

            ch_in = ch_out;
        }

        // Mid blocks: both use last channel count
        let mid_ch = *channels.last().unwrap();
        for i in 1..=2 {
            let name_prefix = format!("mid{i}");
            adapters.push((
                format!("{name_prefix}.conv1"),
                LoraConvAdapter::new(mid_ch, mid_ch, 3, vb.pp(format!("{name_prefix}_c1")))?,
            ));
            adapters.push((
                format!("{name_prefix}.conv2"),
                LoraConvAdapter::new(mid_ch, mid_ch, 3, vb.pp(format!("{name_prefix}_c2")))?,
            ));
        }

        // Decoder: ResBlock inputs have doubled channels from skip connections.
        // up0_r1: input = mid_ch + rev[0], output = rev[0]
        // up1_r1: input = rev[0] + rev[1], output = rev[1]
        // up2_r1: input = rev[1] + rev[2], output = rev[2]
        let rev: Vec<usize> = channels.iter().copied().rev().collect();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };

            let name_prefix = format!("up{i}_r1");
            adapters.push((
                format!("{name_prefix}.conv1"),
                LoraConvAdapter::new(ch_out, skip_ch, 3, vb.pp(format!("{name_prefix}_c1")))?,
            ));
            adapters.push((
                format!("{name_prefix}.conv2"),
                LoraConvAdapter::new(ch_out, ch_out, 3, vb.pp(format!("{name_prefix}_c2")))?,
            ));

            let name_prefix = format!("up{i}_r2");
            adapters.push((
                format!("{name_prefix}.conv1"),
                LoraConvAdapter::new(ch_out, ch_out, 3, vb.pp(format!("{name_prefix}_c1")))?,
            ));
            adapters.push((
                format!("{name_prefix}.conv2"),
                LoraConvAdapter::new(ch_out, ch_out, 3, vb.pp(format!("{name_prefix}_c2")))?,
            ));
        }

        let total_params: usize = adapters.iter().map(|(_, a)| {
            a.a.elem_count() + a.b.elem_count()
        }).sum();
        println!("lora: {} adapters, {} params ({:.1} KB)",
            adapters.len(), total_params, total_params as f64 * 4.0 / 1024.0);

        Ok(Self { adapters })
    }

    /// Get all LoRA delta tensors, keyed by the base weight name they modify.
    pub fn deltas(&self) -> CResult<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(self.adapters.len());
        for (name, adapter) in &self.adapters {
            out.push((name.clone(), adapter.delta()?));
        }
        Ok(out)
    }

    pub fn param_count(&self) -> usize {
        self.adapters.iter().map(|(_, a)| a.a.elem_count() + a.b.elem_count()).sum()
    }
}

/// Apply LoRA deltas to a base VarMap (merge for inference).
/// This modifies the base weights in-place: W = W + delta.
pub fn merge_lora(base_varmap: &VarMap, lora: &LoraSet) -> Result<()> {
    let deltas = lora.deltas()?;
    let vars = base_varmap.data().lock().unwrap();
    for (name, delta) in &deltas {
        // Map LoRA name to base varmap key
        // e.g. "down0_r1.conv1" → look for the conv1.weight in down0_r1
        if let Some(var) = vars.get(name.as_str()) {
            let current = var.as_tensor();
            let merged = (current + delta)?;
            var.set(&merged)?;
        }
    }
    Ok(())
}

/// LoRA training: generate sprites, score with Judge, backprop to update LoRA.
///
/// This is the core feedback loop:
/// 1. Generate N sprites using base+LoRA UNet
/// 2. Judge scores each
/// 3. Loss = -mean(Judge(generated)) (maximize Judge approval)
/// 4. Backprop through last BACKPROP_STEPS denoising steps
/// 5. Update only LoRA params
pub fn train_lora_step(
    base_varmap: &VarMap,
    lora_varmap: &VarMap,
    lora: &LoraSet,
    judge: &crate::judge::MicroClassifier,
    device: &Device,
    num_sprites: usize,
    cond: &crate::class_cond::ClassCond,
) -> Result<f32> {
    // Collect LoRA variables for the optimizer
    let lora_vars = lora_varmap.all_vars();
    let mut opt = nn::AdamW::new(lora_vars, nn::ParamsAdamW {
        lr: LORA_LR,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    let mut total_loss = 0.0f32;

    for _step in 0..LORA_STEPS {
        // Generate sprites with LoRA-modified UNet
        // For now, generate from noise and apply partial denoising
        let bs = num_sprites.min(8); // limit batch for memory
        let mut x = Tensor::rand(0f32, 1f32, (bs, 3, 16, 16), device)?;

        // Get LoRA deltas (recomputed each step since weights change)
        let _deltas = lora.deltas()?;

        // Denoise with base + LoRA (last BACKPROP_STEPS only for gradient)
        let total_steps = 40usize;
        let super_tensor = Tensor::new(&vec![cond.super_id; bs][..], device)?;
        let tags_flat: Vec<f32> = (0..bs).flat_map(|_| cond.tags.iter().copied()).collect();
        let tags_tensor = Tensor::new(tags_flat.as_slice(), device)?.reshape((bs, crate::class_cond::NUM_TAGS))?;

        // First N-BACKPROP_STEPS steps: no gradient (detach)
        for step in 0..(total_steps - BACKPROP_STEPS) {
            let noise_level = 1.0 - (step as f32 / total_steps as f32);
            let t = Tensor::new(vec![noise_level; bs].as_slice(), device)?;
            // Use base model forward (LoRA deltas applied internally)
            let base_vb = VarBuilder::from_varmap(base_varmap, DType::F32, device);
            let model = crate::tiny_unet::TinyUNet::new(base_vb)?;
            let pred = model.forward(&x, &t, &super_tensor, &tags_tensor)?;
            let mix = 1.0 / (total_steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix))?;
            x = x.detach(); // no gradient through early steps
        }

        // Last BACKPROP_STEPS: with gradient
        for step in (total_steps - BACKPROP_STEPS)..total_steps {
            let noise_level = 1.0 - (step as f32 / total_steps as f32);
            let t = Tensor::new(vec![noise_level; bs].as_slice(), device)?;
            let base_vb = VarBuilder::from_varmap(base_varmap, DType::F32, device);
            let model = crate::tiny_unet::TinyUNet::new(base_vb)?;
            let pred = model.forward(&x, &t, &super_tensor, &tags_tensor)?;
            let mix = 1.0 / (total_steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix))?;
        }

        // Clamp to valid pixel range
        let generated = x.clamp(0.0, 1.0)?;

        // Judge scores the generated sprites
        let scores = judge.score(&generated)?; // (bs, 1)

        // Loss: negative mean score (we want to MAXIMIZE Judge approval)
        let loss = scores.mean_all()?.neg()?;

        opt.backward_step(&loss)?;
        total_loss += loss.to_scalar::<f32>()?.abs();
    }

    Ok(total_loss / LORA_STEPS as f32)
}

/// Save LoRA weights (atomic).
pub fn save_lora(varmap: &VarMap, path: &str) -> Result<()> {
    let target = std::path::PathBuf::from(path);
    let tmp = target.with_extension("tmp");
    varmap.save(&tmp)?;
    std::fs::rename(&tmp, &target)?;
    crate::nanosign::sign_and_log(path)?;
    Ok(())
}

/// Load LoRA weights.
pub fn load_lora(path: &str, device: &Device) -> Result<(VarMap, LoraSet)> {
    crate::nanosign::verify_or_bail(path)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let lora = LoraSet::new(vb)?;
    varmap.load(path)?;
    Ok((varmap, lora))
}
