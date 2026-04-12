// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! MicroUNet — per-class siloed model for 32x32 pixel art diffusion.
//! ~97K params with [16,32] channels, ~140K with [16,32,32].
//! No class conditioning — the model IS the class.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder, VarMap};

/// Timestep embedding dimension — half of TinyUNet for minimal overhead.
const TIME_DIM: usize = 32;

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

/// Pick group count for GroupNorm — must divide channel count evenly.
fn group_count(channels: usize) -> usize {
    if channels % 8 == 0 { 8 }
    else if channels % 4 == 0 { 4 }
    else { 1 }
}

/// ResBlock — conv → groupnorm → silu → conv + skip + time conditioning.
/// Single block per level (vs TinyUNet's 2).
struct ResBlock {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
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

        let t = candle_nn::ops::silu(&self.time_proj.forward(t_emb)?)?;
        let t = t.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.broadcast_add(&t)?;

        let h = self.gn2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        h + residual
    }
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

/// MicroUNet — per-class denoising network.
/// Input: (B, 3, 32, 32) noisy image + timestep
/// Output: (B, 3, 32, 32) predicted clean image
/// No class conditioning — each silo trains on one class group.
pub struct MicroUNet {
    conv_in: nn::Conv2d,
    time_mlp1: nn::Linear,
    time_mlp2: nn::Linear,
    down_blocks: Vec<ResBlock>,
    downsamples: Vec<Downsample>,
    mid_block: ResBlock,
    up_blocks: Vec<ResBlock>,
    upsamples: Vec<Upsample>,
    conv_out: nn::Conv2d,
    gn_out: nn::GroupNorm,
    dtype: DType,
}

impl MicroUNet {
    /// Build with default [16, 32] channels.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_channels(vb, &[16, 32])
    }

    /// Build with custom channel config (e.g. [16, 32, 32] for complex classes).
    pub fn with_channels(vb: VarBuilder, channels: &[usize]) -> Result<Self> {
        let dtype = vb.dtype();
        let conv_in = nn::conv2d(3, channels[0], 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;
        let time_mlp1 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp1"))?;
        let time_mlp2 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp2"))?;

        // Encoder: 1 ResBlock per level
        let mut down_blocks = Vec::new();
        let mut downsamples = Vec::new();
        let mut ch_in = channels[0];
        for (i, &ch_out) in channels.iter().enumerate() {
            down_blocks.push(ResBlock::new(ch_in, ch_out, vb.pp(format!("down{i}")))?);
            if i < channels.len() - 1 {
                downsamples.push(Downsample::new(ch_out, vb.pp(format!("down{i}_ds")))?);
            }
            ch_in = ch_out;
        }

        // Single bottleneck block
        let mid_ch = *channels.last().unwrap();
        let mid_block = ResBlock::new(mid_ch, mid_ch, vb.pp("mid"))?;

        // Decoder: 1 ResBlock per level (reversed), skip connections double channels
        let mut up_blocks = Vec::new();
        let mut upsamples = Vec::new();
        let rev: Vec<usize> = channels.iter().copied().rev().collect();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
            up_blocks.push(ResBlock::new(skip_ch, ch_out, vb.pp(format!("up{i}")))?);
            if i < rev.len() - 1 {
                upsamples.push(Upsample::new(ch_out, vb.pp(format!("up{i}_us")))?);
            }
        }

        let out_groups = group_count(channels[0]);
        let gn_out = nn::group_norm(out_groups, channels[0], 1e-5, vb.pp("gn_out"))?;
        let conv_out = nn::conv2d(channels[0], 3, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in, time_mlp1, time_mlp2,
            down_blocks, downsamples, mid_block,
            up_blocks, upsamples, conv_out, gn_out, dtype,
        })
    }

    /// Forward pass — timestep only, no class conditioning.
    pub fn forward_uncond(&self, x: &Tensor, timestep: &Tensor) -> Result<Tensor> {
        let device = x.device();

        let t_emb = timestep_embedding(timestep, TIME_DIM, self.dtype, device)?;
        let t_emb = candle_nn::ops::silu(&self.time_mlp1.forward(&t_emb)?)?;
        let t_emb = self.time_mlp2.forward(&t_emb)?;

        // Encoder
        let mut h = self.conv_in.forward(x)?;
        let mut skips = Vec::new();
        for (i, block) in self.down_blocks.iter().enumerate() {
            h = block.forward(&h, &t_emb)?;
            skips.push(h.clone());
            if i < self.downsamples.len() {
                h = self.downsamples[i].forward(&h)?;
            }
        }

        // Bottleneck
        h = self.mid_block.forward(&h, &t_emb)?;

        // Decoder
        for (i, block) in self.up_blocks.iter().enumerate() {
            let skip = &skips[skips.len() - 1 - i];
            if i > 0 {
                h = self.upsamples[i - 1].forward(&h)?;
            }
            h = Tensor::cat(&[&h, skip], 1)?;
            h = block.forward(&h, &t_emb)?;
        }

        // Output
        let h = self.gn_out.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.conv_out.forward(&h)
    }

    pub fn param_count(varmap: &VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}

/// DiffusionModel impl — ignores super_id/tags (the model IS the class).
impl crate::train::DiffusionModel for MicroUNet {
    fn forward(&self, x: &Tensor, timestep: &Tensor, _super_id: &Tensor, _tags: &Tensor) -> Result<Tensor> {
        self.forward_uncond(x, timestep)
    }
}
