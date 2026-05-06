// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.7
//! Vulkan training backend for TinyUNet (Cinder) via any-gpu.
//!
//! Mirrors `vulkan_backend.rs` but for the larger conditioned TinyUNet:
//! - 2 ResBlocks per encoder/decoder level (vs MicroUNet's 1)
//! - 2 mid blocks
//! - Class conditioning: super-id embedding + 12-tag projection added to time MLP
//! - Configurable input channels (3 or 6, for Cinder-detail conditioning hint)
//! - TIME_DIM = 64, channels [32, 64, 64]
//!
//! Embedding lookup is implemented as one-hot @ matrix matmul since any-gpu
//! does not have a native gather. CPU side builds the (B, 11) one-hot for
//! super_id; tape multiplies by `super_emb` weight (11 × 64).
//!
//! Output safetensors layout matches candle's `TinyUNet::with_channels()` so
//! the trained model loads via `quantize::load_varmap()` for inference.

use anyhow::Result;
use any_gpu::GpuDevice;
use any_gpu::autograd::{Tape, TensorId};
use any_gpu::optim::AdamW;
use std::collections::HashMap;

pub const TIME_DIM: u32 = 64;
pub const CHANNELS: [u32; 3] = [32, 64, 64];
pub const NUM_SUPER: u32 = 11;
pub const NUM_TAGS: u32 = 12;

fn group_count(channels: u32) -> u32 {
    if channels % 8 == 0 { 8 }
    else if channels % 4 == 0 { 4 }
    else { 1 }
}

fn init_vec(n: usize, fan_in: usize, seed: &mut u64) -> Vec<f32> {
    let bound = (6.0 / fan_in.max(1) as f32).sqrt();
    (0..n).map(|_| {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (*seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
        u * bound
    }).collect()
}

fn init_normal(n: usize, std: f32, seed: &mut u64) -> Vec<f32> {
    // Box-Muller
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*seed >> 33) as f32 / (1u64 << 31) as f32).max(1e-7);
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*seed >> 33) as f32 / (1u64 << 31) as f32;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push(r * theta.cos() * std);
        if out.len() < n {
            out.push(r * theta.sin() * std);
        }
    }
    out
}

#[derive(Clone)]
pub struct ConvParams {
    pub weight: Vec<f32>,    // [out, in, k, k]
    pub bias: Vec<f32>,      // [out]
    pub in_ch: u32,
    pub out_ch: u32,
    pub kernel: u32,
}

impl ConvParams {
    fn new(in_ch: u32, out_ch: u32, kernel: u32, seed: &mut u64) -> Self {
        let n = (out_ch * in_ch * kernel * kernel) as usize;
        let fan_in = (in_ch * kernel * kernel) as usize;
        Self {
            weight: init_vec(n, fan_in, seed),
            bias: vec![0.0f32; out_ch as usize],
            in_ch, out_ch, kernel,
        }
    }
}

#[derive(Clone)]
pub struct LinearParams {
    pub weight: Vec<f32>,    // any-gpu storage: [in, out] flat (transposed at save)
    pub bias: Vec<f32>,      // [out]
    pub in_features: u32,
    pub out_features: u32,
}

impl LinearParams {
    fn new(in_features: u32, out_features: u32, seed: &mut u64) -> Self {
        let n = (in_features * out_features) as usize;
        Self {
            weight: init_vec(n, in_features as usize, seed),
            bias: vec![0.0f32; out_features as usize],
            in_features, out_features,
        }
    }
}

/// Embedding params — stored as [vocab, dim] (matches candle's nn::Embedding).
/// At forward time, multiplied via one-hot matmul: (B, vocab) @ (vocab, dim).
#[derive(Clone)]
pub struct EmbeddingParams {
    pub weight: Vec<f32>,    // [vocab, dim]
    pub vocab: u32,
    pub dim: u32,
}

impl EmbeddingParams {
    fn new(vocab: u32, dim: u32, seed: &mut u64) -> Self {
        // candle uses normal(0, 1) for embeddings
        Self {
            weight: init_normal((vocab * dim) as usize, 1.0, seed),
            vocab, dim,
        }
    }
}

#[derive(Clone)]
pub struct GroupNormParams {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub channels: u32,
    pub groups: u32,
}

impl GroupNormParams {
    fn new(channels: u32) -> Self {
        Self {
            gamma: vec![1.0f32; channels as usize],
            beta: vec![0.0f32; channels as usize],
            channels,
            groups: group_count(channels),
        }
    }
}

#[derive(Clone)]
pub struct ResBlockParams {
    pub gn1: GroupNormParams,
    pub conv1: ConvParams,
    pub time_proj: LinearParams,
    pub gn2: GroupNormParams,
    pub conv2: ConvParams,
    pub skip_proj: Option<ConvParams>,
    pub in_ch: u32,
    pub out_ch: u32,
}

impl ResBlockParams {
    fn new(in_ch: u32, out_ch: u32, seed: &mut u64) -> Self {
        Self {
            gn1: GroupNormParams::new(in_ch),
            conv1: ConvParams::new(in_ch, out_ch, 3, seed),
            time_proj: LinearParams::new(TIME_DIM, out_ch, seed),
            gn2: GroupNormParams::new(out_ch),
            conv2: ConvParams::new(out_ch, out_ch, 3, seed),
            skip_proj: if in_ch != out_ch {
                Some(ConvParams::new(in_ch, out_ch, 1, seed))
            } else { None },
            in_ch, out_ch,
        }
    }
}

/// Full TinyUNet (Cinder) param bundle.
pub struct VulkanTinyParams {
    pub in_channels: u32,                              // 3 or 6
    pub conv_in: ConvParams,
    pub time_mlp1: LinearParams,
    pub time_mlp2: LinearParams,
    pub super_emb: EmbeddingParams,
    pub tag_proj: LinearParams,
    pub down_blocks: Vec<(ResBlockParams, ResBlockParams)>,   // 3 levels × 2 res
    pub downsamples: Vec<ConvParams>,                          // 2 (last level skips)
    pub mid1: ResBlockParams,
    pub mid2: ResBlockParams,
    pub up_blocks: Vec<(ResBlockParams, ResBlockParams)>,      // 3 levels × 2 res
    pub upsamples: Vec<ConvParams>,                            // 2 (first level skips)
    pub gn_out: GroupNormParams,
    pub conv_out: ConvParams,
}

impl VulkanTinyParams {
    pub fn new(in_channels: u32) -> Self {
        let mut seed: u64 = 0xCAFE_BABE_F00D_BEEF;

        let conv_in = ConvParams::new(in_channels, CHANNELS[0], 3, &mut seed);
        let time_mlp1 = LinearParams::new(TIME_DIM, TIME_DIM, &mut seed);
        let time_mlp2 = LinearParams::new(TIME_DIM, TIME_DIM, &mut seed);
        let super_emb = EmbeddingParams::new(NUM_SUPER, TIME_DIM, &mut seed);
        let tag_proj = LinearParams::new(NUM_TAGS, TIME_DIM, &mut seed);

        // Encoder
        let mut down_blocks = Vec::new();
        let mut downsamples = Vec::new();
        let mut ch_in = CHANNELS[0];
        for (i, &ch_out) in CHANNELS.iter().enumerate() {
            let r1 = ResBlockParams::new(ch_in, ch_out, &mut seed);
            let r2 = ResBlockParams::new(ch_out, ch_out, &mut seed);
            down_blocks.push((r1, r2));
            if i < CHANNELS.len() - 1 {
                downsamples.push(ConvParams::new(ch_out, ch_out, 3, &mut seed));
            }
            ch_in = ch_out;
        }

        let mid_ch = *CHANNELS.last().unwrap();
        let mid1 = ResBlockParams::new(mid_ch, mid_ch, &mut seed);
        let mid2 = ResBlockParams::new(mid_ch, mid_ch, &mut seed);

        // Decoder (reversed channels, skip-concat doubles input)
        let mut up_blocks = Vec::new();
        let mut upsamples = Vec::new();
        let rev: Vec<u32> = CHANNELS.iter().copied().rev().collect();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
            let r1 = ResBlockParams::new(skip_ch, ch_out, &mut seed);
            let r2 = ResBlockParams::new(ch_out, ch_out, &mut seed);
            up_blocks.push((r1, r2));
            if i < rev.len() - 1 {
                upsamples.push(ConvParams::new(ch_out, ch_out, 3, &mut seed));
            }
        }

        let gn_out = GroupNormParams::new(CHANNELS[0]);
        let conv_out = ConvParams::new(CHANNELS[0], 3, 3, &mut seed);

        Self {
            in_channels,
            conv_in, time_mlp1, time_mlp2, super_emb, tag_proj,
            down_blocks, downsamples, mid1, mid2, up_blocks, upsamples,
            gn_out, conv_out,
        }
    }

    pub fn param_count(&self) -> usize {
        let mut n = self.conv_in.weight.len() + self.conv_in.bias.len();
        n += self.time_mlp1.weight.len() + self.time_mlp1.bias.len();
        n += self.time_mlp2.weight.len() + self.time_mlp2.bias.len();
        n += self.super_emb.weight.len();
        n += self.tag_proj.weight.len() + self.tag_proj.bias.len();
        for (r1, r2) in &self.down_blocks { n += rb_count(r1) + rb_count(r2); }
        for d in &self.downsamples { n += d.weight.len() + d.bias.len(); }
        n += rb_count(&self.mid1) + rb_count(&self.mid2);
        for (r1, r2) in &self.up_blocks { n += rb_count(r1) + rb_count(r2); }
        for u in &self.upsamples { n += u.weight.len() + u.bias.len(); }
        n += self.gn_out.gamma.len() + self.gn_out.beta.len();
        n += self.conv_out.weight.len() + self.conv_out.bias.len();
        n
    }
}

fn rb_count(b: &ResBlockParams) -> usize {
    let mut n = b.gn1.gamma.len() + b.gn1.beta.len();
    n += b.conv1.weight.len() + b.conv1.bias.len();
    n += b.time_proj.weight.len() + b.time_proj.bias.len();
    n += b.gn2.gamma.len() + b.gn2.beta.len();
    n += b.conv2.weight.len() + b.conv2.bias.len();
    if let Some(s) = &b.skip_proj { n += s.weight.len() + s.bias.len(); }
    n
}

// ─── Forward pass machinery ──────────────────────────────────────────────────

pub struct ParamLeaves {
    pub ids: Vec<TensorId>,
    pub sizes: Vec<usize>,
}

fn upload_conv(tape: &mut Tape, p: &ConvParams, leaves: &mut ParamLeaves) -> (TensorId, TensorId) {
    let w = tape.leaf(&p.weight); leaves.ids.push(w); leaves.sizes.push(p.weight.len());
    let b = tape.leaf(&p.bias); leaves.ids.push(b); leaves.sizes.push(p.bias.len());
    (w, b)
}

fn upload_linear(tape: &mut Tape, p: &LinearParams, leaves: &mut ParamLeaves) -> (TensorId, TensorId) {
    let w = tape.leaf(&p.weight); leaves.ids.push(w); leaves.sizes.push(p.weight.len());
    let b = tape.leaf(&p.bias); leaves.ids.push(b); leaves.sizes.push(p.bias.len());
    (w, b)
}

fn upload_emb(tape: &mut Tape, p: &EmbeddingParams, leaves: &mut ParamLeaves) -> TensorId {
    let w = tape.leaf(&p.weight); leaves.ids.push(w); leaves.sizes.push(p.weight.len());
    w
}

fn upload_gn(tape: &mut Tape, p: &GroupNormParams, leaves: &mut ParamLeaves) -> (TensorId, TensorId) {
    let g = tape.leaf(&p.gamma); leaves.ids.push(g); leaves.sizes.push(p.gamma.len());
    let b = tape.leaf(&p.beta); leaves.ids.push(b); leaves.sizes.push(p.beta.len());
    (g, b)
}

fn linear(
    tape: &mut Tape, x: TensorId, w: TensorId, b: TensorId,
    batch: u32, in_features: u32, out_features: u32,
) -> Result<TensorId> {
    let y = tape.matmul(x, w, batch, out_features, in_features)?;
    tape.add_per_col(y, b, batch, out_features)
}

#[allow(clippy::too_many_arguments)]
fn resblock(
    tape: &mut Tape, params: &ResBlockParams, leaves: &mut ParamLeaves,
    x: TensorId, t_emb: TensorId, batch: u32, h: u32, w: u32,
) -> Result<TensorId> {
    let in_ch = params.in_ch;
    let out_ch = params.out_ch;

    let residual = match &params.skip_proj {
        Some(sp) => {
            let (sw, sb) = upload_conv(tape, sp, leaves);
            tape.conv2d(x, sw, Some(sb), batch, in_ch, h, w, out_ch, 1, 1, (1,1), (0,0), (1,1), 1)?
        }
        None => x,
    };

    // Note: candle's ResBlock applies GN→SiLU→Conv1, NOT Conv1→GN→SiLU.
    let (g1, b1) = upload_gn(tape, &params.gn1, leaves);
    let h1 = tape.group_norm(x, g1, b1, batch, in_ch, h * w, params.gn1.groups, 1e-5)?;
    let h1 = tape.swish(h1)?;
    let (cw1, cb1) = upload_conv(tape, &params.conv1, leaves);
    let h1 = tape.conv2d(h1, cw1, Some(cb1), batch, in_ch, h, w, out_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;

    // Time conditioning
    let (tpw, tpb) = upload_linear(tape, &params.time_proj, leaves);
    let t = linear(tape, t_emb, tpw, tpb, batch, TIME_DIM, out_ch)?;
    let t = tape.swish(t)?;
    let h1 = tape.add_broadcast(h1, t, batch * out_ch, h * w)?;

    let (g2, b2) = upload_gn(tape, &params.gn2, leaves);
    let h1 = tape.group_norm(h1, g2, b2, batch, out_ch, h * w, params.gn2.groups, 1e-5)?;
    let h1 = tape.swish(h1)?;
    let (cw2, cb2) = upload_conv(tape, &params.conv2, leaves);
    let h1 = tape.conv2d(h1, cw2, Some(cb2), batch, out_ch, h, w, out_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;

    tape.add(h1, residual)
}

fn downsample(
    tape: &mut Tape, params: &ConvParams, leaves: &mut ParamLeaves,
    x: TensorId, batch: u32, ch: u32, h: u32, w: u32,
) -> Result<TensorId> {
    let (cw, cb) = upload_conv(tape, params, leaves);
    tape.conv2d(x, cw, Some(cb), batch, ch, h, w, ch, 3, 3, (2, 2), (1, 1), (1, 1), 1)
}

fn upsample(
    tape: &mut Tape, params: &ConvParams, leaves: &mut ParamLeaves,
    x: TensorId, batch: u32, ch: u32, h: u32, w: u32,
) -> Result<TensorId> {
    let up = tape.upsample_nearest2d(x, batch, ch, h, w, 2, 2)?;
    let (cw, cb) = upload_conv(tape, params, leaves);
    tape.conv2d(up, cw, Some(cb), batch, ch, h * 2, w * 2, ch, 3, 3, (1, 1), (1, 1), (1, 1), 1)
}

/// Sinusoidal timestep embedding.
pub fn timestep_embedding(noise_amounts: &[f32], dim: u32) -> Vec<f32> {
    let half = (dim / 2) as usize;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(i as f64 * std::f64::consts::LN_2 * 2.0 / half as f64).exp()) as f32)
        .collect();
    let batch = noise_amounts.len();
    let mut out = vec![0.0f32; batch * dim as usize];
    for (b, &t) in noise_amounts.iter().enumerate() {
        for i in 0..half {
            let arg = t * freqs[i];
            out[b * dim as usize + i] = arg.cos();
            out[b * dim as usize + half + i] = arg.sin();
        }
    }
    out
}

/// Build one-hot encoding (B, NUM_SUPER) from super_id list.
fn build_one_hot(super_ids: &[u32], batch: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (batch * NUM_SUPER) as usize];
    for (b, &sid) in super_ids.iter().enumerate() {
        let sid = sid.min(NUM_SUPER - 1);
        out[b * NUM_SUPER as usize + sid as usize] = 1.0;
    }
    out
}

/// Full forward pass: noise prediction.
/// Inputs:
///  - x: (B, in_ch, S, S) noisy image (or [cond, noisy] concat)
///  - t_sin: (B, TIME_DIM) sinusoidal timestep embedding
///  - super_one_hot: (B, NUM_SUPER) one-hot super-id
///  - tags: (B, NUM_TAGS) binary tag floats
#[allow(clippy::too_many_arguments)]
pub fn tiny_forward(
    tape: &mut Tape,
    params: &VulkanTinyParams,
    leaves: &mut ParamLeaves,
    x: TensorId,
    t_sin: TensorId,
    super_one_hot: TensorId,
    tags: TensorId,
    batch: u32,
    img_size: u32,
) -> Result<TensorId> {
    // Time MLP: SiLU(Linear(sin)) → Linear
    let (w1, b1) = upload_linear(tape, &params.time_mlp1, leaves);
    let t = linear(tape, t_sin, w1, b1, batch, TIME_DIM, TIME_DIM)?;
    let t = tape.swish(t)?;
    let (w2, b2) = upload_linear(tape, &params.time_mlp2, leaves);
    let t_emb = linear(tape, t, w2, b2, batch, TIME_DIM, TIME_DIM)?;

    // Class conditioning: super_emb(super_id) + tag_proj(tags)
    // Embedding via one-hot @ matrix: (B, NUM_SUPER) @ (NUM_SUPER, TIME_DIM) → (B, TIME_DIM)
    let emb_w = upload_emb(tape, &params.super_emb, leaves);
    let super_e = tape.matmul(super_one_hot, emb_w, batch, TIME_DIM, NUM_SUPER)?;
    let (tpw, tpb) = upload_linear(tape, &params.tag_proj, leaves);
    let tag_e = linear(tape, tags, tpw, tpb, batch, NUM_TAGS, TIME_DIM)?;
    let c_emb = tape.add(super_e, tag_e)?;
    let t_emb = tape.add(t_emb, c_emb)?;

    // Encoder: conv_in → 3 levels of (r1, r2, optional ds)
    let (cw_in, cb_in) = upload_conv(tape, &params.conv_in, leaves);
    let c0 = CHANNELS[0];
    let in_ch = params.in_channels;
    let mut h = tape.conv2d(
        x, cw_in, Some(cb_in),
        batch, in_ch, img_size, img_size,
        c0, 3, 3, (1,1), (1,1), (1,1), 1,
    )?;

    let mut skips: Vec<(TensorId, u32, u32)> = Vec::new();
    let mut spatial = img_size;
    let mut cur_ch = c0;
    for (i, &ch_out) in CHANNELS.iter().enumerate() {
        let (r1, r2) = &params.down_blocks[i];
        h = resblock(tape, r1, leaves, h, t_emb, batch, spatial, spatial)?;
        h = resblock(tape, r2, leaves, h, t_emb, batch, spatial, spatial)?;
        cur_ch = ch_out;
        skips.push((h, cur_ch, spatial));
        if i < params.downsamples.len() {
            h = downsample(tape, &params.downsamples[i], leaves, h, batch, cur_ch, spatial, spatial)?;
            spatial /= 2;
        }
    }

    // Mid: 2 ResBlocks
    h = resblock(tape, &params.mid1, leaves, h, t_emb, batch, spatial, spatial)?;
    h = resblock(tape, &params.mid2, leaves, h, t_emb, batch, spatial, spatial)?;

    // Decoder: reverse, upsample-then-concat-then-(r1, r2)
    let rev: Vec<u32> = CHANNELS.iter().copied().rev().collect();
    for (i, &ch_out) in rev.iter().enumerate() {
        if i > 0 {
            h = upsample(tape, &params.upsamples[i - 1], leaves, h, batch, cur_ch, spatial, spatial)?;
            spatial *= 2;
        }
        let (skip_id, skip_ch, _) = skips[skips.len() - 1 - i];
        let hw = spatial * spatial;
        h = tape.concat(h, skip_id, batch, cur_ch * hw, skip_ch * hw)?;
        let (r1, r2) = &params.up_blocks[i];
        h = resblock(tape, r1, leaves, h, t_emb, batch, spatial, spatial)?;
        h = resblock(tape, r2, leaves, h, t_emb, batch, spatial, spatial)?;
        cur_ch = ch_out;
    }

    // Output: GN → SiLU → conv_out (3 channels)
    let (gw, gb) = upload_gn(tape, &params.gn_out, leaves);
    let h = tape.group_norm(h, gw, gb, batch, cur_ch, spatial * spatial, params.gn_out.groups, 1e-5)?;
    let h = tape.swish(h)?;
    let (cw_out, cb_out) = upload_conv(tape, &params.conv_out, leaves);
    tape.conv2d(
        h, cw_out, Some(cb_out),
        batch, cur_ch, spatial, spatial,
        3, 3, 3, (1,1), (1,1), (1,1), 1,
    )
}

// ─── Param splat (after AdamW step) ──────────────────────────────────────────

fn splat_resblock(b: &mut ResBlockParams, updated: &[Vec<f32>], idx: &mut usize) {
    if let Some(sp) = &mut b.skip_proj {
        sp.weight.clone_from(&updated[*idx]); *idx += 1;
        sp.bias.clone_from(&updated[*idx]); *idx += 1;
    }
    b.gn1.gamma.clone_from(&updated[*idx]); *idx += 1;
    b.gn1.beta.clone_from(&updated[*idx]); *idx += 1;
    b.conv1.weight.clone_from(&updated[*idx]); *idx += 1;
    b.conv1.bias.clone_from(&updated[*idx]); *idx += 1;
    b.time_proj.weight.clone_from(&updated[*idx]); *idx += 1;
    b.time_proj.bias.clone_from(&updated[*idx]); *idx += 1;
    b.gn2.gamma.clone_from(&updated[*idx]); *idx += 1;
    b.gn2.beta.clone_from(&updated[*idx]); *idx += 1;
    b.conv2.weight.clone_from(&updated[*idx]); *idx += 1;
    b.conv2.bias.clone_from(&updated[*idx]); *idx += 1;
}

fn splat_updated(params: &mut VulkanTinyParams, updated: &[Vec<f32>], leaves: &ParamLeaves) {
    // Order matches upload calls in tiny_forward:
    // time_mlp1, time_mlp2, super_emb, tag_proj, conv_in,
    // for each down level: (r1: skip?, gn1, conv1, time_proj, gn2, conv2),
    //                       (r2: same), then ds if present
    // mid1, mid2 (same resblock pattern)
    // for each up level: (us if i>0), (r1, r2)
    // gn_out, conv_out
    let mut idx = 0usize;
    let mut wb = |w: &mut Vec<f32>, b: &mut Vec<f32>, idx: &mut usize| {
        w.clone_from(&updated[*idx]); *idx += 1;
        b.clone_from(&updated[*idx]); *idx += 1;
    };
    wb(&mut params.time_mlp1.weight, &mut params.time_mlp1.bias, &mut idx);
    wb(&mut params.time_mlp2.weight, &mut params.time_mlp2.bias, &mut idx);
    // super_emb is single-tensor (no bias)
    params.super_emb.weight.clone_from(&updated[idx]); idx += 1;
    wb(&mut params.tag_proj.weight, &mut params.tag_proj.bias, &mut idx);
    wb(&mut params.conv_in.weight, &mut params.conv_in.bias, &mut idx);

    for level in 0..params.down_blocks.len() {
        let (r1, r2) = &mut params.down_blocks[level];
        splat_resblock(r1, updated, &mut idx);
        splat_resblock(r2, updated, &mut idx);
        if level < params.downsamples.len() {
            let ds = &mut params.downsamples[level];
            wb(&mut ds.weight, &mut ds.bias, &mut idx);
        }
    }
    splat_resblock(&mut params.mid1, updated, &mut idx);
    splat_resblock(&mut params.mid2, updated, &mut idx);
    for level in 0..params.up_blocks.len() {
        if level > 0 {
            let us = &mut params.upsamples[level - 1];
            wb(&mut us.weight, &mut us.bias, &mut idx);
        }
        let (r1, r2) = &mut params.up_blocks[level];
        splat_resblock(r1, updated, &mut idx);
        splat_resblock(r2, updated, &mut idx);
    }
    wb(&mut params.gn_out.gamma, &mut params.gn_out.beta, &mut idx);
    wb(&mut params.conv_out.weight, &mut params.conv_out.bias, &mut idx);
    debug_assert_eq!(idx, leaves.ids.len(), "vulkan_tiny: param splat count mismatch");
}

// ─── Train step ──────────────────────────────────────────────────────────────

/// Outcome of a single training step. Surfaced so the trainer can track
/// how often clipping fires and how often a step had to be skipped — both
/// signals that LR is too aggressive for the data distribution.
#[derive(Debug, Clone, Copy)]
pub enum StepOutcome {
    /// Optimizer step applied. `grad_norm` is the pre-clip global L2.
    /// `clipped` is true when `grad_norm > max_grad_norm`.
    Updated { loss: f32, grad_norm: f32, clipped: bool },
    /// Step was skipped because gradients contained NaN or Inf. Optimizer
    /// state is preserved — letting Adam absorb non-finite values
    /// permanently corrupts the moment estimates.
    Skipped { loss: f32, reason: &'static str },
}

impl StepOutcome {
    /// The loss value for the batch — used for epoch averaging regardless
    /// of whether the step actually applied.
    pub fn loss(&self) -> f32 {
        match self {
            StepOutcome::Updated { loss, .. } | StepOutcome::Skipped { loss, .. } => *loss,
        }
    }
}

/// Global L2 norm across every parameter gradient. Returns None when any
/// element is non-finite — the caller should skip the step in that case
/// rather than letting NaN/Inf propagate into the optimizer's state.
fn global_grad_norm(grads: &[Vec<f32>]) -> Option<f32> {
    let mut sumsq = 0f64;
    for g in grads {
        for &v in g {
            if !v.is_finite() {
                return None;
            }
            sumsq += (v as f64) * (v as f64);
        }
    }
    Some(sumsq.sqrt() as f32)
}

/// Scale every gradient element by `scale` in-place. Used to enforce the
/// global-norm clip; cheaper than re-uploading + scaling on the GPU since
/// grads are already on CPU here.
fn scale_grads_in_place(grads: &mut [Vec<f32>], scale: f32) {
    for g in grads {
        for v in g.iter_mut() {
            *v *= scale;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vulkan_tiny_train_step(
    dev: &GpuDevice,
    opt: &mut AdamW,
    params: &mut VulkanTinyParams,
    batch_clean: &[f32],     // (B, 3, S, S)
    batch_cond: Option<&[f32]>, // (B, 3, S, S) when in_channels==6
    batch_noise: &[f32],     // (B, 3, S, S)
    noise_amounts: &[f32],
    super_ids: &[u32],
    tags: &[f32],            // (B, 12)
    batch: u32,
    img_size: u32,
    max_grad_norm: f32,      // 0.0 disables clipping; recommend 1.0 for z-space
) -> Result<StepOutcome> {
    let mut tape = Tape::new(dev);
    let mut leaves = ParamLeaves { ids: Vec::new(), sizes: Vec::new() };

    let stride = (3 * img_size * img_size) as usize;
    // Build noisy_x = clean*(1-t) + noise*t
    let mut noisy = vec![0.0f32; batch as usize * stride];
    for b in 0..batch as usize {
        let t = noise_amounts[b];
        for i in 0..stride {
            noisy[b * stride + i] = batch_clean[b * stride + i] * (1.0 - t)
                + batch_noise[b * stride + i] * t;
        }
    }

    // For 6ch: input = [cond, noisy] concatenated along channel axis.
    // PackedDataset is channel-first per sample, so concat = cond then noisy in memory.
    let input = if params.in_channels == 6 {
        let cond = batch_cond.expect("vulkan_tiny: 6ch model requires batch_cond");
        let mut full = Vec::with_capacity(batch as usize * 2 * stride);
        for b in 0..batch as usize {
            full.extend_from_slice(&cond[b * stride..(b + 1) * stride]);
            full.extend_from_slice(&noisy[b * stride..(b + 1) * stride]);
        }
        full
    } else {
        noisy
    };

    let x_id = tape.leaf(&input);
    let t_sin = timestep_embedding(noise_amounts, TIME_DIM);
    let t_id = tape.leaf(&t_sin);
    let one_hot = build_one_hot(super_ids, batch);
    let oh_id = tape.leaf(&one_hot);
    let tags_id = tape.leaf(tags);
    // Target: clean image (clean-pred mode, matches default Cinder training)
    let target_id = tape.leaf(batch_clean);

    let pred = tiny_forward(&mut tape, params, &mut leaves, x_id, t_id, oh_id, tags_id, batch, img_size)?;

    let loss = tape.mse_loss(pred, target_id)?;
    let loss_val = tape.read(loss)?[0];
    tape.backward(loss)?;

    let param_vecs: Vec<Vec<f32>> = leaves.ids.iter()
        .map(|id| tape.read(*id))
        .collect::<Result<Vec<_>>>()?;
    let mut grad_vecs: Vec<Vec<f32>> = leaves.ids.iter().enumerate()
        .map(|(i, id)| {
            tape.read_grad(*id).map(|opt| opt.unwrap_or_else(|| vec![0.0f32; leaves.sizes[i]]))
        })
        .collect::<Result<Vec<_>>>()?;

    // Gradient hygiene before the optimizer step:
    //   - NaN/Inf → skip the step (Adam's m/v state is permanent;
    //     absorbing a non-finite value corrupts every future update).
    //   - global L2 > max_grad_norm → rescale all grads in place. Disabled
    //     when max_grad_norm == 0.0.
    let raw_norm = match global_grad_norm(&grad_vecs) {
        Some(n) => n,
        None => return Ok(StepOutcome::Skipped {
            loss: loss_val,
            reason: "non-finite gradients",
        }),
    };
    let clipped = max_grad_norm > 0.0 && raw_norm > max_grad_norm;
    if clipped {
        scale_grads_in_place(&mut grad_vecs, max_grad_norm / raw_norm.max(1e-12));
    }

    let mut param_bufs: Vec<_> = param_vecs.iter().map(|v| dev.upload(v)).collect();
    let grad_bufs: Vec<_> = grad_vecs.iter().map(|v| dev.upload(v)).collect();
    opt.step(dev, &mut param_bufs, &grad_bufs)?;

    let updated: Vec<Vec<f32>> = param_bufs.iter()
        .map(|b| dev.read(b))
        .collect::<Result<Vec<_>>>()?;
    splat_updated(params, &updated, &leaves);

    Ok(StepOutcome::Updated { loss: loss_val, grad_norm: raw_norm, clipped })
}

// ─── Save in candle-compatible safetensors layout ────────────────────────────

/// Transpose [in, out] (any-gpu Linear storage) → [out, in] (candle nn::Linear).
fn transpose_linear(weight: &[f32], in_features: u32, out_features: u32) -> Vec<f32> {
    let in_f = in_features as usize;
    let out_f = out_features as usize;
    let mut out = vec![0.0f32; weight.len()];
    for i in 0..in_f {
        for o in 0..out_f {
            out[o * in_f + i] = weight[i * out_f + o];
        }
    }
    out
}

fn t_lin(name: &str, p: &LinearParams, dev: &candle_core::Device, dst: &mut HashMap<String, candle_core::Tensor>) -> Result<()> {
    let w = transpose_linear(&p.weight, p.in_features, p.out_features);
    let wt = candle_core::Tensor::from_vec(w, (p.out_features as usize, p.in_features as usize), dev)?;
    let bt = candle_core::Tensor::from_vec(p.bias.clone(), (p.out_features as usize,), dev)?;
    dst.insert(format!("{name}.weight"), wt);
    dst.insert(format!("{name}.bias"), bt);
    Ok(())
}

fn t_conv(name: &str, p: &ConvParams, dev: &candle_core::Device, dst: &mut HashMap<String, candle_core::Tensor>) -> Result<()> {
    let wt = candle_core::Tensor::from_vec(
        p.weight.clone(),
        (p.out_ch as usize, p.in_ch as usize, p.kernel as usize, p.kernel as usize),
        dev,
    )?;
    let bt = candle_core::Tensor::from_vec(p.bias.clone(), (p.out_ch as usize,), dev)?;
    dst.insert(format!("{name}.weight"), wt);
    dst.insert(format!("{name}.bias"), bt);
    Ok(())
}

fn t_gn(name: &str, p: &GroupNormParams, dev: &candle_core::Device, dst: &mut HashMap<String, candle_core::Tensor>) -> Result<()> {
    let g = candle_core::Tensor::from_vec(p.gamma.clone(), (p.channels as usize,), dev)?;
    let b = candle_core::Tensor::from_vec(p.beta.clone(), (p.channels as usize,), dev)?;
    dst.insert(format!("{name}.weight"), g);
    dst.insert(format!("{name}.bias"), b);
    Ok(())
}

fn t_emb(name: &str, p: &EmbeddingParams, dev: &candle_core::Device, dst: &mut HashMap<String, candle_core::Tensor>) -> Result<()> {
    let w = candle_core::Tensor::from_vec(p.weight.clone(), (p.vocab as usize, p.dim as usize), dev)?;
    dst.insert(format!("{name}.weight"), w);
    Ok(())
}

fn t_resblock(prefix: &str, b: &ResBlockParams, dev: &candle_core::Device, dst: &mut HashMap<String, candle_core::Tensor>) -> Result<()> {
    t_gn(&format!("{prefix}.gn1"), &b.gn1, dev, dst)?;
    t_conv(&format!("{prefix}.conv1"), &b.conv1, dev, dst)?;
    t_lin(&format!("{prefix}.time_proj"), &b.time_proj, dev, dst)?;
    t_gn(&format!("{prefix}.gn2"), &b.gn2, dev, dst)?;
    t_conv(&format!("{prefix}.conv2"), &b.conv2, dev, dst)?;
    if let Some(s) = &b.skip_proj {
        t_conv(&format!("{prefix}.skip"), s, dev, dst)?;
    }
    Ok(())
}

pub fn save_candle_safetensors(params: &VulkanTinyParams, output: &str) -> Result<()> {
    let dev = candle_core::Device::Cpu;
    let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();

    t_conv("conv_in", &params.conv_in, &dev, &mut tensors)?;
    t_lin("time_mlp1", &params.time_mlp1, &dev, &mut tensors)?;
    t_lin("time_mlp2", &params.time_mlp2, &dev, &mut tensors)?;
    t_emb("super_emb", &params.super_emb, &dev, &mut tensors)?;
    t_lin("tag_proj", &params.tag_proj, &dev, &mut tensors)?;

    for (i, (r1, r2)) in params.down_blocks.iter().enumerate() {
        t_resblock(&format!("down{i}_r1"), r1, &dev, &mut tensors)?;
        t_resblock(&format!("down{i}_r2"), r2, &dev, &mut tensors)?;
        if i < params.downsamples.len() {
            t_conv(&format!("down{i}_ds.conv"), &params.downsamples[i], &dev, &mut tensors)?;
        }
    }
    t_resblock("mid1", &params.mid1, &dev, &mut tensors)?;
    t_resblock("mid2", &params.mid2, &dev, &mut tensors)?;
    for (i, (r1, r2)) in params.up_blocks.iter().enumerate() {
        if i > 0 {
            t_conv(&format!("up{}_us.conv", i - 1), &params.upsamples[i - 1], &dev, &mut tensors)?;
        }
        t_resblock(&format!("up{i}_r1"), r1, &dev, &mut tensors)?;
        t_resblock(&format!("up{i}_r2"), r2, &dev, &mut tensors)?;
    }
    t_gn("gn_out", &params.gn_out, &dev, &mut tensors)?;
    t_conv("conv_out", &params.conv_out, &dev, &mut tensors)?;

    candle_core::safetensors::save(&tensors, output)?;
    let _ = crate::nanosign::sign_and_log(output);
    Ok(())
}

/// Load a candle-format safetensors checkpoint into VulkanTinyParams.
/// Used to resume from a 3ch Cinder for the 6ch fine-tune — conv_in.weight
/// is expanded by tiling first 3 input-channels twice (or zero-init).
pub fn load_candle_safetensors(params: &mut VulkanTinyParams, path: &str) -> Result<()> {
    crate::nanosign::verify_or_bail(path)?;
    let raw = std::fs::read(path)?;
    // Strip NanoSign tail
    const MAGIC: &[u8; 4] = b"NSIG";
    const LEN: usize = 36;
    let buf: &[u8] = if raw.len() >= LEN && &raw[raw.len() - LEN..raw.len() - 32] == MAGIC {
        &raw[..raw.len() - LEN]
    } else {
        &raw
    };
    let tensors = candle_core::safetensors::load_buffer(buf, &candle_core::Device::Cpu)?;

    let load_vec = |name: &str, tensors: &HashMap<String, candle_core::Tensor>| -> Result<Option<Vec<f32>>> {
        match tensors.get(name) {
            Some(t) => {
                let f = if t.dtype() == candle_core::DType::F16 {
                    t.to_dtype(candle_core::DType::F32)?
                } else { t.clone() };
                Ok(Some(f.flatten_all()?.to_vec1::<f32>()?))
            }
            None => Ok(None),
        }
    };

    let load_lin = |name: &str, p: &mut LinearParams, tensors: &HashMap<String, candle_core::Tensor>| -> Result<()> {
        // candle stores weight as [out, in]; transpose back to any-gpu's [in, out]
        if let Some(w) = load_vec(&format!("{name}.weight"), tensors)? {
            if w.len() == (p.in_features * p.out_features) as usize {
                let t = transpose_linear(&w, p.out_features, p.in_features); // reverse direction
                p.weight = t;
            }
        }
        if let Some(b) = load_vec(&format!("{name}.bias"), tensors)? {
            if b.len() == p.bias.len() { p.bias = b; }
        }
        Ok(())
    };
    let load_conv = |name: &str, p: &mut ConvParams, tensors: &HashMap<String, candle_core::Tensor>, allow_expand: bool| -> Result<()> {
        if let Some(w) = load_vec(&format!("{name}.weight"), tensors)? {
            let expected = (p.out_ch * p.in_ch * p.kernel * p.kernel) as usize;
            if w.len() == expected {
                p.weight = w;
            } else if allow_expand && w.len() < expected {
                // 3ch → 6ch conv_in: tile first 3 input channels into the 6ch slots.
                // weight layout: [out_ch, in_ch, kH, kW]. Source has in_ch_src.
                let kk = (p.kernel * p.kernel) as usize;
                let in_dst = p.in_ch as usize;
                let in_src = w.len() / (p.out_ch as usize * kk);
                if in_src > 0 && in_src < in_dst && p.out_ch as usize * in_src * kk == w.len() {
                    let mut expanded = vec![0.0f32; expected];
                    for o in 0..p.out_ch as usize {
                        for i in 0..in_dst {
                            let src_i = i % in_src; // tile/repeat
                            for k in 0..kk {
                                let src_idx = (o * in_src + src_i) * kk + k;
                                let dst_idx = (o * in_dst + i) * kk + k;
                                expanded[dst_idx] = w[src_idx] * 0.5; // halve so total magnitude stays similar
                            }
                        }
                    }
                    p.weight = expanded;
                }
            }
        }
        if let Some(b) = load_vec(&format!("{name}.bias"), tensors)? {
            if b.len() == p.bias.len() { p.bias = b; }
        }
        Ok(())
    };
    let load_gn = |name: &str, p: &mut GroupNormParams, tensors: &HashMap<String, candle_core::Tensor>| -> Result<()> {
        if let Some(g) = load_vec(&format!("{name}.weight"), tensors)? {
            if g.len() == p.gamma.len() { p.gamma = g; }
        }
        if let Some(b) = load_vec(&format!("{name}.bias"), tensors)? {
            if b.len() == p.beta.len() { p.beta = b; }
        }
        Ok(())
    };
    let load_emb = |name: &str, p: &mut EmbeddingParams, tensors: &HashMap<String, candle_core::Tensor>| -> Result<()> {
        if let Some(w) = load_vec(&format!("{name}.weight"), tensors)? {
            if w.len() == p.weight.len() { p.weight = w; }
        }
        Ok(())
    };
    let load_rb = |prefix: &str, b: &mut ResBlockParams, tensors: &HashMap<String, candle_core::Tensor>| -> Result<()> {
        load_gn(&format!("{prefix}.gn1"), &mut b.gn1, tensors)?;
        load_conv(&format!("{prefix}.conv1"), &mut b.conv1, tensors, false)?;
        load_lin(&format!("{prefix}.time_proj"), &mut b.time_proj, tensors)?;
        load_gn(&format!("{prefix}.gn2"), &mut b.gn2, tensors)?;
        load_conv(&format!("{prefix}.conv2"), &mut b.conv2, tensors, false)?;
        if let Some(s) = &mut b.skip_proj {
            load_conv(&format!("{prefix}.skip"), s, tensors, false)?;
        }
        Ok(())
    };

    load_conv("conv_in", &mut params.conv_in, &tensors, true)?;
    load_lin("time_mlp1", &mut params.time_mlp1, &tensors)?;
    load_lin("time_mlp2", &mut params.time_mlp2, &tensors)?;
    load_emb("super_emb", &mut params.super_emb, &tensors)?;
    load_lin("tag_proj", &mut params.tag_proj, &tensors)?;

    for i in 0..params.down_blocks.len() {
        let name1 = format!("down{i}_r1");
        let name2 = format!("down{i}_r2");
        let (r1, r2) = &mut params.down_blocks[i];
        load_rb(&name1, r1, &tensors)?;
        load_rb(&name2, r2, &tensors)?;
        if i < params.downsamples.len() {
            load_conv(&format!("down{i}_ds.conv"), &mut params.downsamples[i], &tensors, false)?;
        }
    }
    load_rb("mid1", &mut params.mid1, &tensors)?;
    load_rb("mid2", &mut params.mid2, &tensors)?;
    for i in 0..params.up_blocks.len() {
        if i > 0 {
            load_conv(&format!("up{}_us.conv", i - 1), &mut params.upsamples[i - 1], &tensors, false)?;
        }
        let name1 = format!("up{i}_r1");
        let name2 = format!("up{i}_r2");
        let (r1, r2) = &mut params.up_blocks[i];
        load_rb(&name1, r1, &tensors)?;
        load_rb(&name2, r2, &tensors)?;
    }
    load_gn("gn_out", &mut params.gn_out, &tensors)?;
    load_conv("conv_out", &mut params.conv_out, &tensors, false)?;

    Ok(())
}

// ─── Top-level training loop ─────────────────────────────────────────────────

pub struct VulkanTinyTrainConfig {
    pub data_dir: String,
    pub cond_dir: Option<String>,
    pub output: String,
    pub resume: Option<String>,
    pub epochs: usize,
    pub batch_size: u32,
    pub lr: f64,
    pub img_size: u32,
    /// Global L2-norm clip for parameter gradients each step. 0.0 disables
    /// clipping. ~1.0 is robust for z-scored input; higher values let the
    /// optimizer take larger swings early in training.
    pub max_grad_norm: f32,
}

pub fn vulkan_tiny_train(cfg: &VulkanTinyTrainConfig) -> Result<()> {
    use rand::seq::SliceRandom;
    use rand::Rng;

    let dev = GpuDevice::gpu()?;
    println!("vulkan: {} ({})", dev.adapter_name, dev.backend);

    let in_ch = if cfg.cond_dir.is_some() { 6 } else { 3 };
    let mut params = VulkanTinyParams::new(in_ch);
    println!("vulkan TinyUNet (Cinder, {in_ch}ch): {} params", params.param_count());

    if let Some(ref path) = cfg.resume {
        load_candle_safetensors(&mut params, path)?;
        println!("resumed from {path} (conv_in {}ch→{}ch tile-expand if needed)", 3, in_ch);
        // Save the post-resume / pre-training state as a baseline checkpoint
        // alongside the live output. Lets us A/B "as-resumed" vs "trained".
        let baseline = format!("{}.epoch0", cfg.output);
        save_candle_safetensors(&params, &baseline)?;
        println!("baseline saved → {baseline}");
    }

    let mut opt = AdamW::new(cfg.lr as f32);
    opt.weight_decay = 0.01;

    let dataset = crate::train::preprocess(&cfg.data_dir, cfg.img_size)?;
    let cond_dataset = if let Some(ref cdir) = cfg.cond_dir {
        Some(crate::train::preprocess(cdir, cfg.img_size)?)
    } else { None };

    let normalizer = crate::normalize::Normalizer::load(
        std::path::Path::new(&format!("{}/normalize.json", cfg.data_dir))
    )?;
    if let Some(ref n) = normalizer {
        println!("vulkan z-score: mean={:?} std={:?} (sidecar saved with each checkpoint)", n.mean, n.std);
    }

    let n = dataset.super_ids.len();
    let stride = dataset.stride;
    println!("vulkan: {n} samples, {} epochs, bs={}, lr={:.1e}", cfg.epochs, cfg.batch_size, cfg.lr);
    if let Some(ref cd) = cond_dataset {
        if cd.super_ids.len() != n {
            anyhow::bail!("conditioning size {} != dataset {}", cd.super_ids.len(), n);
        }
    }

    println!("vulkan: max_grad_norm={} (0=off)", cfg.max_grad_norm);

    let t0 = std::time::Instant::now();
    for epoch in 0..cfg.epochs {
        let epoch_start = std::time::Instant::now();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0f64;
        let mut batches = 0u32;
        let mut clipped_count = 0u32;
        let mut skipped_count = 0u32;
        let num_batches = n.div_ceil(cfg.batch_size as usize);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..num_batches {
            let start = batch_idx * cfg.batch_size as usize;
            let end = (start + cfg.batch_size as usize).min(n);
            let bs = (end - start) as u32;
            if bs < cfg.batch_size { break; }

            let mut batch_clean = Vec::with_capacity(bs as usize * stride);
            let mut batch_cond = if cond_dataset.is_some() {
                Vec::with_capacity(bs as usize * stride)
            } else { Vec::new() };
            let mut batch_super: Vec<u32> = Vec::with_capacity(bs as usize);
            let mut batch_tags: Vec<f32> = Vec::with_capacity(bs as usize * NUM_TAGS as usize);

            for &idx in &indices[start..end] {
                let src = idx * stride;
                let mut clean = dataset.pixels[src..src + stride].to_vec();
                if let Some(ref nrm) = normalizer {
                    let n_px = (cfg.img_size * cfg.img_size) as usize;
                    nrm.to_z_pixels(&mut clean, n_px);
                }
                batch_clean.extend_from_slice(&clean);
                if let Some(ref cd) = cond_dataset {
                    let mut cnd = cd.pixels[src..src + stride].to_vec();
                    if let Some(ref nrm) = normalizer {
                        let n_px = (cfg.img_size * cfg.img_size) as usize;
                        nrm.to_z_pixels(&mut cnd, n_px);
                    }
                    batch_cond.extend_from_slice(&cnd);
                }
                batch_super.push(dataset.super_ids[idx]);
                batch_tags.extend_from_slice(&dataset.tags[idx]);
            }

            // Box-Muller noise
            let batch_noise: Vec<f32> = (0..bs as usize * stride)
                .map(|_| {
                    let u1: f32 = rng.r#gen::<f32>().max(1e-7);
                    let u2: f32 = rng.r#gen::<f32>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect();
            // Cosine-scheduled noise levels
            let noise_amounts: Vec<f32> = (0..bs)
                .map(|_| {
                    let t: f32 = rng.gen_range(0.0f32..1.0);
                    crate::train::cosine_schedule(t)
                })
                .collect();

            let cond_slice = if cond_dataset.is_some() { Some(batch_cond.as_slice()) } else { None };
            let step_t0 = std::time::Instant::now();
            let outcome = vulkan_tiny_train_step(
                &dev, &mut opt, &mut params,
                &batch_clean, cond_slice, &batch_noise,
                &noise_amounts, &batch_super, &batch_tags,
                bs, cfg.img_size, cfg.max_grad_norm,
            )?;
            let step_dt = step_t0.elapsed().as_secs_f32();
            let loss = outcome.loss();
            if loss.is_nan() {
                anyhow::bail!("vulkan: NaN loss at epoch {} batch {}", epoch + 1, batch_idx);
            }
            match outcome {
                StepOutcome::Updated { grad_norm, clipped, .. } => {
                    if clipped { clipped_count += 1; }
                    if batch_idx < 3 || batch_idx % 100 == 0 {
                        let clip_tag = if clipped { " *CLIP*" } else { "" };
                        println!("    batch {}/{} loss={:.5} |g|={:.3}{} ({:.2}s/step)",
                            batch_idx + 1, num_batches, loss, grad_norm, clip_tag, step_dt);
                    }
                }
                StepOutcome::Skipped { reason, .. } => {
                    skipped_count += 1;
                    println!("    batch {}/{} loss={:.5} SKIPPED ({reason}) ({:.2}s/step)",
                        batch_idx + 1, num_batches, loss, step_dt);
                }
            }
            epoch_loss += loss as f64;
            batches += 1;
        }

        let avg = if batches > 0 { epoch_loss / batches as f64 } else { 0.0 };
        let dt = epoch_start.elapsed().as_secs_f32();
        println!("  vulkan epoch {}/{}: loss={:.5} clipped={}/{} skipped={} ({:.1}s)",
            epoch + 1, cfg.epochs, avg, clipped_count, batches, skipped_count, dt);

        // Save every epoch — small file, lets us inspect conditioning behavior.
        save_candle_safetensors(&params, &cfg.output)?;
        if let Some(ref nrm) = normalizer {
            nrm.save_sidecar(std::path::Path::new(&cfg.output))?;
        }
    }

    println!("vulkan: done in {:.1}s, saved → {}", t0.elapsed().as_secs_f32(), cfg.output);
    Ok(())
}

/// Probe the Vulkan device. Returns the device info string.
pub fn probe_vulkan() -> Result<String> {
    let dev = GpuDevice::gpu()?;
    Ok(format!("{} ({})", dev.adapter_name, dev.backend))
}
