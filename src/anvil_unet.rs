// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! AnvilUNet — XL diffusion model for highest-quality 32×32 sprites.
//! ~16M params (~64 MB). The flagship model.
//!
//! Channel config: [128, 256, 256] (vs Quench's [64, 128, 128])
//! ResBlocks: 4 per level (vs Quench's 3)
//! Self-attention: bottleneck (4×4) AND mid-level (8×8)
//! Time dim: 256 (vs Quench's 128)
//!
//! Targets devices with 3+ GB RAM. Sub-second on modern GPUs.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder, VarMap};

/// Channel config per resolution level — 1.5x wider than Quench.
const CHANNELS: [usize; 3] = [96, 192, 192];

/// Number of class labels for conditioning.
pub const NUM_CLASSES: usize = 16; // 15 classes + 1 null (CFG unconditional)

/// Timestep embedding dimension — 2x wider than Quench.
const TIME_DIM: usize = 256;

/// Sinusoidal timestep embedding.
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

/// Pick group count for GroupNorm.
fn group_count(channels: usize) -> usize {
    if channels % 32 == 0 { 32 }
    else if channels % 16 == 0 { 16 }
    else if channels % 8 == 0 { 8 }
    else { 1 }
}

/// ResBlock with time conditioning.
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

/// Self-attention block — adds global context.
struct SelfAttention {
    qkv: nn::Linear,
    out_proj: nn::Linear,
    gn: nn::GroupNorm,
    channels: usize,
}

impl SelfAttention {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gn = nn::group_norm(group_count(channels), channels, 1e-5, vb.pp("gn"))?;
        let qkv = nn::linear(channels, channels * 3, vb.pp("qkv"))?;
        let out_proj = nn::linear(channels, channels, vb.pp("out"))?;
        Ok(Self { qkv, out_proj, gn, channels })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let hw = h * w;

        let normed = self.gn.forward(x)?;
        let flat = normed.reshape((b, c, hw))?.permute((0, 2, 1))?;

        let qkv = self.qkv.forward(&flat)?;
        let q = qkv.narrow(2, 0, self.channels)?.contiguous()?;
        let k = qkv.narrow(2, self.channels, self.channels)?.contiguous()?;
        let v = qkv.narrow(2, self.channels * 2, self.channels)?.contiguous()?;

        let scale = (self.channels as f32).sqrt();
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let attn = q.matmul(&k_t)?.broadcast_div(
            &Tensor::new(scale, x.device())?.to_dtype(x.dtype())?
        )?;
        let attn = nn::ops::softmax(&attn, 2)?;
        let out = attn.matmul(&v)?;

        let out = self.out_proj.forward(&out)?;
        let out = out.permute((0, 2, 1))?.contiguous()?.reshape((b, c, h, w))?;

        x + &out
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

/// AnvilUNet — XL denoising network.
///
/// Key differences from Quench:
/// - Channels: [128, 256, 256] vs [64, 128, 128]
/// - 4 ResBlocks per level vs 3
/// - Self-attention at 8×8 (level 1) AND 4×4 (bottleneck)
/// - Time dim: 256 vs 128
/// - ~16M params vs ~5.8M params
pub struct AnvilUNet {
    conv_in: nn::Conv2d,
    time_mlp1: nn::Linear,
    time_mlp2: nn::Linear,
    class_emb: nn::Embedding,
    // Encoder: 4 ResBlocks per level
    down_blocks: Vec<Vec<ResBlock>>,
    down_attns: Vec<Option<SelfAttention>>,
    downsamples: Vec<Downsample>,
    // Bottleneck with self-attention
    mid_block1: ResBlock,
    mid_attn: SelfAttention,
    mid_block2: ResBlock,
    // Decoder
    up_blocks: Vec<Vec<ResBlock>>,
    up_attns: Vec<Option<SelfAttention>>,
    upsamples: Vec<Upsample>,
    conv_out: nn::Conv2d,
    gn_out: nn::GroupNorm,
    // Model dtype for f16-aware tensor creation
    dtype: DType,
}

impl AnvilUNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let conv_in = nn::conv2d(3, CHANNELS[0], 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("conv_in"))?;

        let time_mlp1 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp1"))?;
        let time_mlp2 = nn::linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp2"))?;
        let class_emb = nn::embedding(NUM_CLASSES, TIME_DIM, vb.pp("class_emb"))?;

        let blocks_per_level = 4;

        // Encoder: 3 levels, 4 ResBlocks each
        // Self-attention at level 1 (8×8 resolution) — global context at mid-res
        let mut down_blocks = Vec::new();
        let mut down_attns: Vec<Option<SelfAttention>> = Vec::new();
        let mut downsamples = Vec::new();
        let mut ch_in = CHANNELS[0];
        for (i, &ch_out) in CHANNELS.iter().enumerate() {
            let mut blocks = Vec::new();
            for j in 0..blocks_per_level {
                let in_ch = if j == 0 { ch_in } else { ch_out };
                blocks.push(ResBlock::new(in_ch, ch_out, vb.pp(format!("down{i}_r{j}")))?);
            }
            down_blocks.push(blocks);

            // Self-attention at level 1 (8×8 after first downsample)
            if i == 1 {
                down_attns.push(Some(SelfAttention::new(ch_out, vb.pp(format!("down{i}_attn")))?));
            } else {
                down_attns.push(None);
            }

            if i < CHANNELS.len() - 1 {
                downsamples.push(Downsample::new(ch_out, vb.pp(format!("down{i}_ds")))?);
            }
            ch_in = ch_out;
        }

        // Bottleneck with self-attention
        let mid_ch = *CHANNELS.last().unwrap();
        let mid_block1 = ResBlock::new(mid_ch, mid_ch, vb.pp("mid1"))?;
        let mid_attn = SelfAttention::new(mid_ch, vb.pp("mid_attn"))?;
        let mid_block2 = ResBlock::new(mid_ch, mid_ch, vb.pp("mid2"))?;

        // Decoder: 3 levels, 4 ResBlocks each
        let mut up_blocks = Vec::new();
        let mut up_attns: Vec<Option<SelfAttention>> = Vec::new();
        let mut upsamples = Vec::new();
        let rev: Vec<usize> = CHANNELS.iter().copied().rev().collect();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
            let mut blocks = Vec::new();
            for j in 0..blocks_per_level {
                let in_ch = if j == 0 { skip_ch } else { ch_out };
                blocks.push(ResBlock::new(in_ch, ch_out, vb.pp(format!("up{i}_r{j}")))?);
            }
            up_blocks.push(blocks);

            // Self-attention at matching level (mirror of encoder level 1)
            if i == rev.len() - 2 {
                up_attns.push(Some(SelfAttention::new(ch_out, vb.pp(format!("up{i}_attn")))?));
            } else {
                up_attns.push(None);
            }

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
            class_emb,
            down_blocks, down_attns, downsamples,
            mid_block1, mid_attn, mid_block2,
            up_blocks, up_attns, upsamples,
            conv_out, gn_out,
            dtype,
        })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor, timestep: &Tensor, class_id: &Tensor) -> Result<Tensor> {
        let device = x.device();

        // Time + class embedding
        let t_emb = timestep_embedding(timestep, TIME_DIM, self.dtype, device)?;
        let t_emb = candle_nn::ops::silu(&self.time_mlp1.forward(&t_emb)?)?;
        let t_emb = self.time_mlp2.forward(&t_emb)?;
        let c_emb = self.class_emb.forward(class_id)?;
        let t_emb = (t_emb + c_emb)?;

        // Encoder
        let mut h = self.conv_in.forward(x)?;
        let mut skips = Vec::new();
        for (i, blocks) in self.down_blocks.iter().enumerate() {
            for block in blocks {
                h = block.forward(&h, &t_emb)?;
            }
            // Self-attention if present at this level
            if let Some(attn) = &self.down_attns[i] {
                h = attn.forward(&h)?;
            }
            skips.push(h.clone());
            if i < self.downsamples.len() {
                h = self.downsamples[i].forward(&h)?;
            }
        }

        // Bottleneck
        h = self.mid_block1.forward(&h, &t_emb)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block2.forward(&h, &t_emb)?;

        // Decoder
        for (i, blocks) in self.up_blocks.iter().enumerate() {
            let skip = &skips[skips.len() - 1 - i];
            if i > 0 {
                h = self.upsamples[i - 1].forward(&h)?;
            }
            h = Tensor::cat(&[&h, skip], 1)?;
            for block in blocks {
                h = block.forward(&h, &t_emb)?;
            }
            if let Some(attn) = &self.up_attns[i] {
                h = attn.forward(&h)?;
            }
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
