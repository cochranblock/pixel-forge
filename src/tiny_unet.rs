// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Tiny UNet for 32x32 pixel art diffusion.
//! Architecture inspired by PixelGen (MIT, Anouar Khaldi 2025)
//! and pixartdiffusion (Zak Buzzard 2023).
//! Rewritten from scratch in Rust/Candle — no Python code was copied.
//!
//! Model size: ~1-5MB (vs 4GB for SD v1.4).
//! Operates in direct pixel space — no VAE, no latent encoding.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder, VarMap};

/// Channel config per resolution level.
/// 32x32 input → 16x16 → 8x8
const CHANNELS: [usize; 3] = [32, 64, 64];

/// Legacy class count — kept for backwards compat with old models.
pub const NUM_CLASSES: usize = 16; // 15 classes + 1 null (CFG unconditional)

use crate::class_cond::{NUM_SUPER_WITH_NULL, NUM_TAGS};

/// Class conditioner: legacy (single embedding) or hybrid (super-cat + tags).
enum ClassConditioner {
    Legacy(nn::Embedding),
    Hybrid { super_emb: nn::Embedding, tag_proj: nn::Linear },
}

impl ClassConditioner {
    fn forward_hybrid(&self, super_id: &Tensor, tags: &Tensor) -> Result<Tensor> {
        match self {
            Self::Hybrid { super_emb, tag_proj } => {
                let s = super_emb.forward(super_id)?;
                let t = tag_proj.forward(tags)?;
                s + t
            }
            Self::Legacy(emb) => {
                // Legacy path: super_id is treated as class_id, tags ignored
                emb.forward(super_id)
            }
        }
    }
}

/// Timestep embedding dimension.
const TIME_DIM: usize = 64;

/// Sinusoidal timestep embedding — maps scalar timestep to vector.
fn timestep_embedding(timestep: &Tensor, dim: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(i as f64 * std::f64::consts::LN_2 * 2.0 / half as f64).exp()) as f32)
        .collect();
    let freqs = Tensor::new(freqs.as_slice(), device)?.to_dtype(dtype)?;
    let args = timestep.to_dtype(dtype)?.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
    Ok(emb)
}

/// ResBlock — conv → groupnorm → silu → conv + skip connection + time/class conditioning.
pub(crate) struct ResBlock {
    pub(crate) conv1: nn::Conv2d,
    pub(crate) conv2: nn::Conv2d,
    gn1: nn::GroupNorm,
    gn2: nn::GroupNorm,
    time_proj: nn::Linear,
    skip_proj: Option<nn::Conv2d>,
}

impl ResBlock {
    fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = nn::conv2d(in_ch, out_ch, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv1"))?;
        let conv2 = nn::conv2d(out_ch, out_ch, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv2"))?;
        let groups1 = group_count(in_ch);
        let groups2 = group_count(out_ch);
        let gn1 = nn::group_norm(groups1, in_ch, 1e-5, vb.pp("gn1"))?;
        let gn2 = nn::group_norm(groups2, out_ch, 1e-5, vb.pp("gn2"))?;
        let time_proj = nn::linear(TIME_DIM, out_ch, vb.pp("time_proj"))?;
        let skip_proj = if in_ch != out_ch {
            Some(nn::conv2d(in_ch, out_ch, 1, Default::default(), vb.pp("skip"))?)
        } else {
            None
        };
        Ok(Self { conv1, conv2, gn1, gn2, time_proj, skip_proj })
    }

    fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        let residual = match &self.skip_proj {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };

        let h = self.gn1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        // Add time embedding (broadcast across spatial dims)
        let t = candle_nn::ops::silu(&self.time_proj.forward(t_emb)?)?;
        let t = t.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.broadcast_add(&t)?;

        let h = self.gn2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        h + residual
    }
}

/// Pick group count for GroupNorm — must divide channel count evenly.
fn group_count(channels: usize) -> usize {
    if channels % 8 == 0 { 8 }
    else if channels % 4 == 0 { 4 }
    else { 1 }
}

/// Downsample 2x via strided convolution.
struct Downsample {
    conv: nn::Conv2d,
}

impl Downsample {
    fn new(ch: usize, vb: VarBuilder) -> Result<Self> {
        let conv = nn::conv2d(ch, ch, 3, nn::Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Upsample 2x via nearest interpolation + conv.
struct Upsample {
    conv: nn::Conv2d,
}

impl Upsample {
    fn new(ch: usize, vb: VarBuilder) -> Result<Self> {
        let conv = nn::conv2d(ch, ch, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&x)
    }
}

/// TinyUNet — the full denoising network.
/// Input: (B, 3, H, W) noisy image + timestep scalar + class label
/// Output: (B, 3, H, W) predicted noise (or clean image)
pub struct TinyUNet {
    // Input projection: 3 RGB channels → CHANNELS[0]
    conv_in: nn::Conv2d,
    // Time embedding MLP
    time_mlp1: nn::Linear,
    time_mlp2: nn::Linear,
    // Class conditioning (legacy or hybrid)
    class_cond: ClassConditioner,
    // Encoder — pub(crate) for LoRA adapter access
    pub(crate) down_blocks: Vec<(ResBlock, ResBlock)>,
    downsamples: Vec<Downsample>,
    // Bottleneck
    pub(crate) mid_block1: ResBlock,
    pub(crate) mid_block2: ResBlock,
    // Decoder (channels doubled from skip connections)
    pub(crate) up_blocks: Vec<(ResBlock, ResBlock)>,
    upsamples: Vec<Upsample>,
    // Output projection: CHANNELS[0] → 3 RGB
    conv_out: nn::Conv2d,
    gn_out: nn::GroupNorm,
    // Model dtype for f16-aware tensor creation
    dtype: DType,
}

impl TinyUNet {
    /// Build with default hybrid conditioning.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_hybrid(vb, 3)
    }

    /// Build with legacy class embedding (for loading old models).
    pub fn with_classes(vb: VarBuilder, n_classes: usize) -> Result<Self> {
        Self::with_config(vb, n_classes, 3)
    }

    /// Hybrid conditioning: super-category embedding + tag projection.
    pub fn with_hybrid(vb: VarBuilder, in_channels: usize) -> Result<Self> {
        let dtype = vb.dtype();
        let conv_in = nn::conv2d(in_channels, CHANNELS[0], 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;
        let time_mlp1 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp1"))?;
        let time_mlp2 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp2"))?;
        let class_cond = ClassConditioner::Hybrid {
            super_emb: nn::embedding(NUM_SUPER_WITH_NULL, TIME_DIM, vb.pp("super_emb"))?,
            tag_proj: nn::linear(NUM_TAGS, TIME_DIM, vb.pp("tag_proj"))?,
        };
        Self::build(conv_in, time_mlp1, time_mlp2, class_cond, vb, dtype)
    }

    /// Legacy: configurable input channels + class count.
    pub fn with_config(vb: VarBuilder, n_classes: usize, in_channels: usize) -> Result<Self> {
        let dtype = vb.dtype();
        let conv_in = nn::conv2d(in_channels, CHANNELS[0], 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;
        let time_mlp1 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp1"))?;
        let time_mlp2 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp2"))?;
        let class_cond = ClassConditioner::Legacy(nn::embedding(n_classes, TIME_DIM, vb.pp("class_emb"))?);
        Self::build(conv_in, time_mlp1, time_mlp2, class_cond, vb, dtype)
    }

    fn build(conv_in: nn::Conv2d, time_mlp1: nn::Linear, time_mlp2: nn::Linear, class_cond: ClassConditioner, vb: VarBuilder, dtype: DType) -> Result<Self> {

        // Encoder: 3 levels
        let mut down_blocks = Vec::new();
        let mut downsamples = Vec::new();
        let mut ch_in = CHANNELS[0];
        for (i, &ch_out) in CHANNELS.iter().enumerate() {
            let r1 = ResBlock::new(ch_in, ch_out, vb.pp(format!("down{i}_r1")))?;
            let r2 = ResBlock::new(ch_out, ch_out, vb.pp(format!("down{i}_r2")))?;
            down_blocks.push((r1, r2));
            if i < CHANNELS.len() - 1 {
                downsamples.push(Downsample::new(ch_out, vb.pp(format!("down{i}_ds")))?);
            }
            ch_in = ch_out;
        }

        // Bottleneck
        let mid_ch = *CHANNELS.last().unwrap();
        let mid_block1 = ResBlock::new(mid_ch, mid_ch, vb.pp("mid1"))?;
        let mid_block2 = ResBlock::new(mid_ch, mid_ch, vb.pp("mid2"))?;

        // Decoder: 3 levels (reversed), skip connections double input channels
        let mut up_blocks = Vec::new();
        let mut upsamples = Vec::new();
        let rev: Vec<usize> = CHANNELS.iter().copied().rev().collect();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
            let r1 = ResBlock::new(skip_ch, ch_out, vb.pp(format!("up{i}_r1")))?;
            let r2 = ResBlock::new(ch_out, ch_out, vb.pp(format!("up{i}_r2")))?;
            up_blocks.push((r1, r2));
            if i < rev.len() - 1 {
                upsamples.push(Upsample::new(ch_out, vb.pp(format!("up{i}_us")))?);
            }
        }

        let out_groups = group_count(CHANNELS[0]);
        let gn_out = nn::group_norm(out_groups, CHANNELS[0], 1e-5, vb.pp("gn_out"))?;
        let conv_out = nn::conv2d(CHANNELS[0], 3, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            time_mlp1, time_mlp2,
            class_cond,
            down_blocks, downsamples,
            mid_block1, mid_block2,
            up_blocks, upsamples,
            conv_out, gn_out,
            dtype,
        })
    }

    /// Forward pass.
    /// - x: (B, 3, H, W) noisy image, values in [0, 1]
    /// - timestep: (B,) noise level in [0, 1]
    /// - super_id: (B,) super-category ID (or legacy class_id)
    /// - tags: (B, 12) binary trait tags
    pub fn forward(&self, x: &Tensor, timestep: &Tensor, super_id: &Tensor, tags: &Tensor) -> Result<Tensor> {
        let device = x.device();

        // Time + class conditioning
        let t_emb = timestep_embedding(timestep, TIME_DIM, self.dtype, device)?;
        let t_emb = candle_nn::ops::silu(&self.time_mlp1.forward(&t_emb)?)?;
        let t_emb = self.time_mlp2.forward(&t_emb)?;
        let c_emb = self.class_cond.forward_hybrid(super_id, tags)?;
        let t_emb = (t_emb + c_emb)?;

        // Encoder
        let mut h = self.conv_in.forward(x)?;
        let mut skips = Vec::new();
        for (i, (r1, r2)) in self.down_blocks.iter().enumerate() {
            h = r1.forward(&h, &t_emb)?;
            h = r2.forward(&h, &t_emb)?;
            skips.push(h.clone());
            if i < self.downsamples.len() {
                h = self.downsamples[i].forward(&h)?;
            }
        }

        // Bottleneck
        h = self.mid_block1.forward(&h, &t_emb)?;
        h = self.mid_block2.forward(&h, &t_emb)?;

        // Decoder
        for (i, (r1, r2)) in self.up_blocks.iter().enumerate() {
            let skip = &skips[skips.len() - 1 - i];
            // Upsample first if needed (before concat, to match spatial dims)
            if i > 0 {
                h = self.upsamples[i - 1].forward(&h)?;
            }
            h = Tensor::cat(&[&h, skip], 1)?;
            h = r1.forward(&h, &t_emb)?;
            h = r2.forward(&h, &t_emb)?;
        }

        // Output
        let h = self.gn_out.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.conv_out.forward(&h)
    }

    /// Count total parameters.
    pub fn param_count(varmap: &VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}
