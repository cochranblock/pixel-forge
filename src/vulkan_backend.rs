// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Vulkan training backend via any-gpu — MicroUNet path.
//!
//! Enables training MicroUNet (~97K params) on GPUs candle can't reach — notably
//! AMD RDNA1/2 (RX 5700 XT on bt) where ROCm is unavailable. Uses any-gpu's
//! tape-based autograd with Vulkan/wgpu compute.
//!
//! Architecture matches candle's MicroUNet (src/micro_unet.rs):
//! - conv_in (3→c0) → time MLP (sinusoidal → 2×Linear)
//! - Down path: ResBlock + Downsample per level (except last)
//! - Mid block
//! - Up path: Upsample + Concat + ResBlock per level (reversed)
//! - gn_out → SiLU → conv_out (c0→3)
//!
//! Parameters are stored as flat Vec<f32> on CPU; each training step uploads
//! them as tape leaves, runs forward+backward, extracts gradients, and updates
//! via AdamW. CPU↔GPU round-trip per step is non-trivial — same trade-off as
//! any-gpu's train_step. Pipeline caching keeps shader compilation cost constant.

use anyhow::Result;
use any_gpu::GpuDevice;
use any_gpu::autograd::{Tape, TensorId};
use any_gpu::optim::AdamW;

/// Timestep embedding dimension — matches MicroUNet's TIME_DIM.
pub const TIME_DIM: u32 = 32;

/// Pick GroupNorm group count — must divide channel count evenly.
/// Mirrors MicroUNet's group_count().
fn group_count(channels: u32) -> u32 {
    if channels % 8 == 0 { 8 }
    else if channels % 4 == 0 { 4 }
    else { 1 }
}

/// Xavier/Kaiming-ish init via LCG — deterministic per-call for repro.
fn init_vec(n: usize, fan_in: usize, seed: &mut u64) -> Vec<f32> {
    // Kaiming uniform: bound = sqrt(6 / fan_in) for ReLU/SiLU
    let bound = (6.0 / fan_in.max(1) as f32).sqrt();
    (0..n).map(|_| {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (*seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0; // [-1, 1)
        u * bound
    }).collect()
}

/// Conv2d parameter block: weight [out, in, k, k] + bias [out].
#[derive(Clone)]
pub struct ConvParams {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_ch: u32,
    pub out_ch: u32,
    pub kernel: u32,
}

impl ConvParams {
    fn new(in_ch: u32, out_ch: u32, kernel: u32, seed: &mut u64) -> Self {
        let n = (out_ch * in_ch * kernel * kernel) as usize;
        let fan_in = (in_ch * kernel * kernel) as usize;
        let weight = init_vec(n, fan_in, seed);
        let bias = vec![0.0f32; out_ch as usize];
        Self { weight, bias, in_ch, out_ch, kernel }
    }
}

/// Linear layer params: weight [in, out] stored row-major as matmul(A, B, m=batch, n=out, k=in)
/// expects B to be [k, n] = [in, out].
#[derive(Clone)]
pub struct LinearParams {
    pub weight: Vec<f32>, // [in, out] flattened
    pub bias: Vec<f32>,   // [out]
    pub in_features: u32,
    pub out_features: u32,
}

impl LinearParams {
    fn new(in_features: u32, out_features: u32, seed: &mut u64) -> Self {
        let n = (in_features * out_features) as usize;
        let weight = init_vec(n, in_features as usize, seed);
        let bias = vec![0.0f32; out_features as usize];
        Self { weight, bias, in_features, out_features }
    }
}

/// GroupNorm affine params: gamma and beta [channels].
#[derive(Clone)]
pub struct GroupNormParams {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub channels: u32,
    pub groups: u32,
}

impl GroupNormParams {
    fn new(channels: u32) -> Self {
        let groups = group_count(channels);
        Self {
            gamma: vec![1.0f32; channels as usize],
            beta: vec![0.0f32; channels as usize],
            channels,
            groups,
        }
    }
}

/// ResBlock parameter bundle. Matches candle's ResBlock layout.
#[derive(Clone)]
pub struct ResBlockParams {
    pub gn1: GroupNormParams,
    pub conv1: ConvParams,
    pub time_proj: LinearParams,
    pub gn2: GroupNormParams,
    pub conv2: ConvParams,
    pub skip_proj: Option<ConvParams>, // 1x1 conv, only if in != out
    pub in_ch: u32,
    pub out_ch: u32,
}

impl ResBlockParams {
    fn new(in_ch: u32, out_ch: u32, seed: &mut u64) -> Self {
        let gn1 = GroupNormParams::new(in_ch);
        let conv1 = ConvParams::new(in_ch, out_ch, 3, seed);
        let time_proj = LinearParams::new(TIME_DIM, out_ch, seed);
        let gn2 = GroupNormParams::new(out_ch);
        let conv2 = ConvParams::new(out_ch, out_ch, 3, seed);
        let skip_proj = if in_ch != out_ch {
            Some(ConvParams::new(in_ch, out_ch, 1, seed))
        } else {
            None
        };
        Self { gn1, conv1, time_proj, gn2, conv2, skip_proj, in_ch, out_ch }
    }
}

/// MicroUNet parameter bundle. Channels [c0, c1, ..., cN].
/// Architecture: 1 ResBlock per level, N-1 downsamples, 1 mid, N ResBlocks up with
/// concat skip connections, N-1 upsamples, gn_out + conv_out.
pub struct VulkanMicroParams {
    pub channels: Vec<u32>,
    pub conv_in: ConvParams,
    pub time_mlp1: LinearParams,
    pub time_mlp2: LinearParams,
    pub down_blocks: Vec<ResBlockParams>,
    pub downsamples: Vec<ConvParams>,
    pub mid_block: ResBlockParams,
    pub up_blocks: Vec<ResBlockParams>,
    pub upsamples: Vec<ConvParams>,
    pub gn_out: GroupNormParams,
    pub conv_out: ConvParams,
}

impl VulkanMicroParams {
    pub fn new(channels: &[u32]) -> Self {
        assert!(!channels.is_empty(), "channels must not be empty");
        let mut seed: u64 = 0xC0FFEE_1BADF00D;

        let conv_in = ConvParams::new(3, channels[0], 3, &mut seed);
        let time_mlp1 = LinearParams::new(TIME_DIM, TIME_DIM, &mut seed);
        let time_mlp2 = LinearParams::new(TIME_DIM, TIME_DIM, &mut seed);

        // Down path
        let mut down_blocks = Vec::new();
        let mut downsamples = Vec::new();
        let mut ch_in = channels[0];
        for (i, &ch_out) in channels.iter().enumerate() {
            down_blocks.push(ResBlockParams::new(ch_in, ch_out, &mut seed));
            if i < channels.len() - 1 {
                downsamples.push(ConvParams::new(ch_out, ch_out, 3, &mut seed));
            }
            ch_in = ch_out;
        }

        // Mid
        let mid_ch = *channels.last().unwrap();
        let mid_block = ResBlockParams::new(mid_ch, mid_ch, &mut seed);

        // Up path (reversed channels, skip concat doubles channels)
        let rev: Vec<u32> = channels.iter().copied().rev().collect();
        let mut up_blocks = Vec::new();
        let mut upsamples = Vec::new();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
            up_blocks.push(ResBlockParams::new(skip_ch, ch_out, &mut seed));
            if i < rev.len() - 1 {
                upsamples.push(ConvParams::new(ch_out, ch_out, 3, &mut seed));
            }
        }

        let gn_out = GroupNormParams::new(channels[0]);
        let conv_out = ConvParams::new(channels[0], 3, 3, &mut seed);

        Self {
            channels: channels.to_vec(),
            conv_in, time_mlp1, time_mlp2,
            down_blocks, downsamples, mid_block,
            up_blocks, upsamples, gn_out, conv_out,
        }
    }

    /// Total trainable parameter count.
    pub fn param_count(&self) -> usize {
        let mut n = self.conv_in.weight.len() + self.conv_in.bias.len();
        n += self.time_mlp1.weight.len() + self.time_mlp1.bias.len();
        n += self.time_mlp2.weight.len() + self.time_mlp2.bias.len();
        for b in &self.down_blocks { n += resblock_count(b); }
        for d in &self.downsamples { n += d.weight.len() + d.bias.len(); }
        n += resblock_count(&self.mid_block);
        for b in &self.up_blocks { n += resblock_count(b); }
        for u in &self.upsamples { n += u.weight.len() + u.bias.len(); }
        n += self.gn_out.gamma.len() + self.gn_out.beta.len();
        n += self.conv_out.weight.len() + self.conv_out.bias.len();
        n
    }

    /// Flatten all params into a single Vec<f32> in stable order.
    pub fn flatten(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_count());
        out.extend(&self.conv_in.weight); out.extend(&self.conv_in.bias);
        out.extend(&self.time_mlp1.weight); out.extend(&self.time_mlp1.bias);
        out.extend(&self.time_mlp2.weight); out.extend(&self.time_mlp2.bias);
        for b in &self.down_blocks { flatten_resblock(b, &mut out); }
        for d in &self.downsamples { out.extend(&d.weight); out.extend(&d.bias); }
        flatten_resblock(&self.mid_block, &mut out);
        for b in &self.up_blocks { flatten_resblock(b, &mut out); }
        for u in &self.upsamples { out.extend(&u.weight); out.extend(&u.bias); }
        out.extend(&self.gn_out.gamma); out.extend(&self.gn_out.beta);
        out.extend(&self.conv_out.weight); out.extend(&self.conv_out.bias);
        out
    }
}

fn resblock_count(b: &ResBlockParams) -> usize {
    let mut n = b.gn1.gamma.len() + b.gn1.beta.len();
    n += b.conv1.weight.len() + b.conv1.bias.len();
    n += b.time_proj.weight.len() + b.time_proj.bias.len();
    n += b.gn2.gamma.len() + b.gn2.beta.len();
    n += b.conv2.weight.len() + b.conv2.bias.len();
    if let Some(s) = &b.skip_proj { n += s.weight.len() + s.bias.len(); }
    n
}

fn flatten_resblock(b: &ResBlockParams, out: &mut Vec<f32>) {
    out.extend(&b.gn1.gamma); out.extend(&b.gn1.beta);
    out.extend(&b.conv1.weight); out.extend(&b.conv1.bias);
    out.extend(&b.time_proj.weight); out.extend(&b.time_proj.bias);
    out.extend(&b.gn2.gamma); out.extend(&b.gn2.beta);
    out.extend(&b.conv2.weight); out.extend(&b.conv2.bias);
    if let Some(s) = &b.skip_proj { out.extend(&s.weight); out.extend(&s.bias); }
}

/// Sinusoidal timestep embedding — matches MicroUNet's timestep_embedding().
/// Input: noise_amounts [B], output: [B, TIME_DIM] flat.
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

/// Upload all params as tape leaves. Returns a flat Vec<TensorId> matching the
/// flatten() order, plus parameter shape metadata for optimizer step.
pub struct ParamLeaves {
    pub ids: Vec<TensorId>,
    pub sizes: Vec<usize>, // elements per leaf
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

fn upload_gn(tape: &mut Tape, p: &GroupNormParams, leaves: &mut ParamLeaves) -> (TensorId, TensorId) {
    let g = tape.leaf(&p.gamma); leaves.ids.push(g); leaves.sizes.push(p.gamma.len());
    let b = tape.leaf(&p.beta); leaves.ids.push(b); leaves.sizes.push(p.beta.len());
    (g, b)
}

/// Apply a Linear layer: y = x @ W + b. x shape (B, in), W (in, out), b (out).
/// matmul in any-gpu: (m, k) @ (k, n) -> (m, n). So y = matmul(x, W, B, out, in).
fn linear(
    tape: &mut Tape,
    x: TensorId, w: TensorId, b: TensorId,
    batch: u32, in_features: u32, out_features: u32,
) -> Result<TensorId> {
    let y = tape.matmul(x, w, batch, out_features, in_features)?;
    tape.add_per_col(y, b, batch, out_features)
}

/// Apply a ResBlock. x: (B, in_ch, H, W), t_emb: (B, TIME_DIM).
/// Returns output shape (B, out_ch, H, W).
#[allow(clippy::too_many_arguments)]
fn resblock(
    tape: &mut Tape,
    params: &ResBlockParams,
    leaves: &mut ParamLeaves,
    x: TensorId, t_emb: TensorId,
    batch: u32, h: u32, w: u32,
) -> Result<TensorId> {
    let in_ch = params.in_ch;
    let out_ch = params.out_ch;

    // Skip projection (conv1x1 if in != out, else identity)
    let residual = match &params.skip_proj {
        Some(sp) => {
            let (sw, sb) = upload_conv(tape, sp, leaves);
            tape.conv2d(x, sw, Some(sb), batch, in_ch, h, w, out_ch, 1, 1, (1,1), (0,0), (1,1), 1)?
        }
        None => x,
    };

    // Main path: GN → SiLU → Conv1 → TimeProj broadcast add → GN → SiLU → Conv2
    let (g1, b1) = upload_gn(tape, &params.gn1, leaves);
    let h1 = tape.group_norm(x, g1, b1, batch, in_ch, h * w, params.gn1.groups, 1e-5)?;
    let h1 = tape.swish(h1)?;

    let (cw1, cb1) = upload_conv(tape, &params.conv1, leaves);
    let h1 = tape.conv2d(h1, cw1, Some(cb1), batch, in_ch, h, w, out_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;

    // Time conditioning: Linear(t_emb) -> SiLU -> broadcast add across spatial
    let (tpw, tpb) = upload_linear(tape, &params.time_proj, leaves);
    let t = linear(tape, t_emb, tpw, tpb, batch, TIME_DIM, out_ch)?;
    let t = tape.swish(t)?;
    // h1: (B, out_ch, H*W), t: (B, out_ch). Broadcast add across spatial.
    // outer = B * out_ch, inner = H * W.
    let h1 = tape.add_broadcast(h1, t, batch * out_ch, h * w)?;

    let (g2, b2) = upload_gn(tape, &params.gn2, leaves);
    let h1 = tape.group_norm(h1, g2, b2, batch, out_ch, h * w, params.gn2.groups, 1e-5)?;
    let h1 = tape.swish(h1)?;

    let (cw2, cb2) = upload_conv(tape, &params.conv2, leaves);
    let h1 = tape.conv2d(h1, cw2, Some(cb2), batch, out_ch, h, w, out_ch, 3, 3, (1,1), (1,1), (1,1), 1)?;

    tape.add(h1, residual)
}

/// Apply a Downsample: stride-2 conv3x3 with padding=1. Spatial halves.
fn downsample(
    tape: &mut Tape, params: &ConvParams, leaves: &mut ParamLeaves,
    x: TensorId, batch: u32, ch: u32, h: u32, w: u32,
) -> Result<TensorId> {
    let (cw, cb) = upload_conv(tape, params, leaves);
    tape.conv2d(x, cw, Some(cb), batch, ch, h, w, ch, 3, 3, (2, 2), (1, 1), (1, 1), 1)
}

/// Apply an Upsample: nearest2x then conv3x3 pad=1. Spatial doubles.
fn upsample(
    tape: &mut Tape, params: &ConvParams, leaves: &mut ParamLeaves,
    x: TensorId, batch: u32, ch: u32, h: u32, w: u32,
) -> Result<TensorId> {
    let up = tape.upsample_nearest2d(x, batch, ch, h, w, 2, 2)?;
    let (cw, cb) = upload_conv(tape, params, leaves);
    tape.conv2d(up, cw, Some(cb), batch, ch, h * 2, w * 2, ch, 3, 3, (1, 1), (1, 1), (1, 1), 1)
}

/// Full MicroUNet forward pass. Returns predicted noise (B, 3, img, img).
/// `x` is noisy input (B, 3, img, img), `t_emb_sinusoidal` is sinusoidal timestep
/// embedding (B, TIME_DIM) — uploaded as a leaf to enable backprop through time MLP.
#[allow(clippy::too_many_arguments)]
pub fn micro_forward(
    tape: &mut Tape,
    params: &VulkanMicroParams,
    leaves: &mut ParamLeaves,
    x: TensorId,
    t_sinusoidal: TensorId,
    batch: u32,
    img_size: u32,
) -> Result<TensorId> {
    // Time MLP: SiLU(Linear(sinusoidal)) -> Linear
    let (w1, b1) = upload_linear(tape, &params.time_mlp1, leaves);
    let t = linear(tape, t_sinusoidal, w1, b1, batch, TIME_DIM, TIME_DIM)?;
    let t = tape.swish(t)?;
    let (w2, b2) = upload_linear(tape, &params.time_mlp2, leaves);
    let t_emb = linear(tape, t, w2, b2, batch, TIME_DIM, TIME_DIM)?;

    // Encoder: conv_in + (ResBlock + Downsample) per level
    let (cw_in, cb_in) = upload_conv(tape, &params.conv_in, leaves);
    let c0 = params.channels[0];
    let mut h = tape.conv2d(
        x, cw_in, Some(cb_in),
        batch, 3, img_size, img_size,
        c0, 3, 3, (1,1), (1,1), (1,1), 1,
    )?;

    let mut skips: Vec<(TensorId, u32, u32)> = Vec::new(); // (id, ch, spatial)
    let mut spatial = img_size;
    let mut cur_ch = c0;
    for (i, &ch_out) in params.channels.iter().enumerate() {
        h = resblock(tape, &params.down_blocks[i], leaves, h, t_emb, batch, spatial, spatial)?;
        cur_ch = ch_out;
        skips.push((h, cur_ch, spatial));
        if i < params.downsamples.len() {
            h = downsample(tape, &params.downsamples[i], leaves, h, batch, cur_ch, spatial, spatial)?;
            spatial /= 2;
        }
    }

    // Mid block
    h = resblock(tape, &params.mid_block, leaves, h, t_emb, batch, spatial, spatial)?;

    // Decoder: Upsample + Concat + ResBlock per level (reversed)
    let rev: Vec<u32> = params.channels.iter().copied().rev().collect();
    for (i, &ch_out) in rev.iter().enumerate() {
        if i > 0 {
            h = upsample(tape, &params.upsamples[i - 1], leaves, h, batch, cur_ch, spatial, spatial)?;
            spatial *= 2;
        }
        // Concat h with skip[N-1-i] along channel axis.
        let (skip_id, skip_ch, skip_sp) = skips[skips.len() - 1 - i];
        debug_assert_eq!(skip_sp, spatial, "skip spatial must match");
        // concat inputs: h is (B, cur_ch, S*S), skip is (B, skip_ch, S*S).
        // Concat along channel axis (axis 1): outer = B, a_inner = cur_ch * S*S, b_inner = skip_ch * S*S.
        let hw = spatial * spatial;
        let concat_ch = cur_ch + skip_ch;
        h = tape.concat(h, skip_id, batch, cur_ch * hw, skip_ch * hw)?;
        // Now h represents (B, concat_ch, S, S) flattened.
        let _ = concat_ch; // keep for clarity; resblock uses params.up_blocks[i].in_ch
        debug_assert_eq!(params.up_blocks[i].in_ch, concat_ch, "ResBlock in_ch must match concat output");
        h = resblock(tape, &params.up_blocks[i], leaves, h, t_emb, batch, spatial, spatial)?;
        cur_ch = ch_out;
    }

    // Output: GN → SiLU → conv_out
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

/// Probe the Vulkan device. Returns the device info string.
pub fn probe_vulkan() -> Result<String> {
    let dev = GpuDevice::gpu()?;
    Ok(format!("{} ({})", dev.adapter_name, dev.backend))
}

/// Assign updated flat params back to the VulkanMicroParams struct.
fn splat_updated(params: &mut VulkanMicroParams, updated: &[Vec<f32>], leaves: &ParamLeaves) {
    // Order of leaves matches upload_* calls in micro_forward. Reconstruct using
    // a running index into the updated slice. This mirrors how the forward pass
    // uploads params.
    let mut idx = 0;
    let mut take_w_b = |w: &mut Vec<f32>, b: &mut Vec<f32>| {
        w.clone_from(&updated[idx]); idx += 1;
        b.clone_from(&updated[idx]); idx += 1;
    };
    // time_mlp1, time_mlp2 (uploaded first in micro_forward)
    take_w_b(&mut params.time_mlp1.weight, &mut params.time_mlp1.bias);
    take_w_b(&mut params.time_mlp2.weight, &mut params.time_mlp2.bias);
    // conv_in
    take_w_b(&mut params.conv_in.weight, &mut params.conv_in.bias);

    let splat_resblock = |b: &mut ResBlockParams, idx: &mut usize| {
        // Order matches resblock(): skip_proj (if any), gn1, conv1, time_proj, gn2, conv2
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
    };

    // Down path: for each level: down_block, then downsample (if not last)
    for i in 0..params.down_blocks.len() {
        splat_resblock(&mut params.down_blocks[i], &mut idx);
        if i < params.downsamples.len() {
            params.downsamples[i].weight.clone_from(&updated[idx]); idx += 1;
            params.downsamples[i].bias.clone_from(&updated[idx]); idx += 1;
        }
    }
    // Mid
    splat_resblock(&mut params.mid_block, &mut idx);
    // Up path: for each level: upsample (if i > 0), then up_block
    for i in 0..params.up_blocks.len() {
        if i > 0 {
            params.upsamples[i - 1].weight.clone_from(&updated[idx]); idx += 1;
            params.upsamples[i - 1].bias.clone_from(&updated[idx]); idx += 1;
        }
        splat_resblock(&mut params.up_blocks[i], &mut idx);
    }
    // Output: gn_out, conv_out
    params.gn_out.gamma.clone_from(&updated[idx]); idx += 1;
    params.gn_out.beta.clone_from(&updated[idx]); idx += 1;
    params.conv_out.weight.clone_from(&updated[idx]); idx += 1;
    params.conv_out.bias.clone_from(&updated[idx]); idx += 1;

    debug_assert_eq!(idx, leaves.ids.len(), "param splat count mismatch");
}

/// Run one training step on the Vulkan MicroUNet backend.
/// Target is the ORIGINAL noise (v-prediction not supported in this first cut).
/// Returns the MSE loss value.
#[allow(clippy::too_many_arguments)]
pub fn vulkan_micro_train_step(
    dev: &GpuDevice,
    opt: &mut AdamW,
    params: &mut VulkanMicroParams,
    batch_clean: &[f32],     // (B, 3, S, S) flattened — clean target images
    batch_noise: &[f32],     // (B, 3, S, S) flattened — noise target
    noise_amounts: &[f32],   // (B,) noise levels in [0, 1]
    batch: u32,
    img_size: u32,
) -> Result<f32> {
    let mut tape = Tape::new(dev);
    let mut leaves = ParamLeaves { ids: Vec::new(), sizes: Vec::new() };

    // Build noisy input on CPU: noisy[b] = clean[b] * (1 - t[b]) + noise[b] * t[b].
    // Doing this CPU-side avoids needing per-sample scale in the tape.
    let stride = (3 * img_size * img_size) as usize;
    let mut noisy = vec![0.0f32; batch as usize * stride];
    for b in 0..batch as usize {
        let t = noise_amounts[b];
        let one_minus_t = 1.0 - t;
        for i in 0..stride {
            noisy[b * stride + i] = batch_clean[b * stride + i] * one_minus_t
                + batch_noise[b * stride + i] * t;
        }
    }

    // Upload inputs (not tracked for grad — the data is fixed).
    let x_id = tape.leaf(&noisy);
    let t_sin = timestep_embedding(noise_amounts, TIME_DIM);
    let t_id = tape.leaf(&t_sin);
    // Target: the noise (standard denoising objective).
    let target_id = tape.leaf(batch_noise);

    // Forward through MicroUNet.
    let pred = micro_forward(&mut tape, params, &mut leaves, x_id, t_id, batch, img_size)?;

    // MSE loss.
    let loss = tape.mse_loss(pred, target_id)?;
    let loss_val = tape.read(loss)?[0];

    // Backward.
    tape.backward(loss)?;

    // Extract params and grads via CPU roundtrip (matches any-gpu::train::train_step).
    let param_vecs: Vec<Vec<f32>> = leaves.ids.iter()
        .map(|id| tape.read(*id))
        .collect::<Result<Vec<_>>>()?;
    let grad_vecs: Vec<Vec<f32>> = leaves.ids.iter().enumerate()
        .map(|(i, id)| {
            tape.read_grad(*id).map(|opt| opt.unwrap_or_else(|| vec![0.0f32; leaves.sizes[i]]))
        })
        .collect::<Result<Vec<_>>>()?;

    // Upload as fresh buffers for AdamW in-place update.
    let mut param_bufs: Vec<_> = param_vecs.iter().map(|v| dev.upload(v)).collect();
    let grad_bufs: Vec<_> = grad_vecs.iter().map(|v| dev.upload(v)).collect();
    opt.step(dev, &mut param_bufs, &grad_bufs)?;

    // Read updated params back to CPU.
    let updated: Vec<Vec<f32>> = param_bufs.iter()
        .map(|b| dev.read(b))
        .collect::<Result<Vec<_>>>()?;

    // Splat back into the params struct.
    splat_updated(params, &updated, &leaves);

    Ok(loss_val)
}

/// Full Vulkan training loop for MicroUNet — trains one silo on bt's AMD GPU.
pub fn vulkan_micro_train(
    data_dir: &str,
    output: &str,
    channels: &[usize],
    class_filter: &[String],
    epochs: usize,
    batch_size: u32,
    lr: f64,
    img_size: u32,
) -> Result<()> {
    let dev = GpuDevice::gpu()?;
    println!("vulkan: {} ({})", dev.adapter_name, dev.backend);

    let ch32: Vec<u32> = channels.iter().map(|&c| c as u32).collect();
    let mut params = VulkanMicroParams::new(&ch32);
    println!("vulkan MicroUNet: {} params, channels={:?}", params.param_count(), channels);

    let mut opt = AdamW::new(lr as f32);
    opt.weight_decay = 0.01;

    // Load dataset (candle-based preprocessing; this is CPU-only so it works on bt).
    let dataset = if class_filter.is_empty() {
        crate::train::preprocess(data_dir, img_size)?
    } else {
        crate::train::preprocess_class(data_dir, img_size, class_filter)?
    };
    let n = dataset.pixels.len() / dataset.stride;
    let stride = dataset.stride;
    println!("vulkan: {} samples, {} epochs, bs={}", n, epochs, batch_size);

    if n == 0 {
        anyhow::bail!("no samples for class filter {:?}", class_filter);
    }

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f64;
        let mut batches = 0u32;

        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());

        let num_batches = n.div_ceil(batch_size as usize);
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size as usize;
            let end = (start + batch_size as usize).min(n);
            let bs = (end - start) as u32;
            // Skip runt batches so tensor shapes stay consistent.
            if bs < batch_size { break; }

            let mut batch_clean = Vec::with_capacity(bs as usize * stride);
            for &idx in &indices[start..end] {
                let src = idx * stride;
                batch_clean.extend_from_slice(&dataset.pixels[src..src + stride]);
            }

            use rand::Rng;
            let mut rng = rand::thread_rng();
            let batch_noise: Vec<f32> = (0..bs as usize * stride)
                .map(|_| {
                    let u1: f32 = rng.r#gen::<f32>().max(1e-7);
                    let u2: f32 = rng.r#gen::<f32>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect();

            // Per-sample noise amounts — sampling from a reasonable range.
            let noise_amounts: Vec<f32> = (0..bs)
                .map(|_| rng.gen_range(0.02f32..0.98))
                .collect();

            let loss = vulkan_micro_train_step(
                &dev, &mut opt, &mut params,
                &batch_clean, &batch_noise, &noise_amounts,
                bs, img_size,
            )?;
            if loss.is_nan() {
                anyhow::bail!("vulkan: NaN loss at epoch {}, batch {}", epoch + 1, batch_idx);
            }
            epoch_loss += loss as f64;
            batches += 1;
        }

        let avg = if batches > 0 { epoch_loss / batches as f64 } else { 0.0 };
        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("  vulkan epoch {}/{}: loss={:.6}", epoch + 1, epochs, avg);
        }
    }

    println!("vulkan: training complete, saving to {output}");
    // Save params as a flat safetensors tensor for now (v2: map to MicroUNet's
    // VarMap layout for candle-compatible inference).
    let flat = params.flatten();
    let t = candle_core::Tensor::from_vec(flat.clone(), (flat.len(),), &candle_core::Device::Cpu)?;
    let tensors = std::collections::HashMap::from([("vulkan_micro_flat".to_string(), t)]);
    candle_core::safetensors::save(&tensors, output)?;
    crate::nanosign::sign_and_log(output)?;

    Ok(())
}
