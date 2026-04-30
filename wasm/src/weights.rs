// Unlicense — cochranblock.org
//! Cinder TinyUNet weights, GPU-resident.
//!
//! Layout mirrors `pixel-forge/src/tiny_unet.rs`. Each layer's safetensors
//! entries are uploaded once at load and held as `GpuBuffer`s; forward()
//! consumes them as plain references.
//!
//! Linear weights are TRANSPOSED at load. Candle stores `[out, in]` and
//! computes `x · W^T`; any-gpu's matmul takes the natural `A[m,k] · B[k,n]`,
//! so we materialize the transpose once and use it as `B[in, out]`.

use any_gpu::{GpuBuffer, GpuDevice};
use anyhow::{anyhow, Context, Result};

use crate::class_cond::{NUM_SUPER_WITH_NULL, NUM_TAGS};
use crate::loader::{find, LoadedTensor};

/// 32x32 → 16x16 → 8x8 channel ladder. Must match the trained model.
pub const CHANNELS: [u32; 3] = [32, 64, 64];
/// Time / class embedding width.
pub const TIME_DIM: u32 = 64;

pub struct ConvW {
    pub weight: GpuBuffer,
    pub bias: GpuBuffer,
    pub in_c: u32,
    pub out_c: u32,
    pub kh: u32,
    pub kw: u32,
    pub stride: (u32, u32),
    pub padding: (u32, u32),
}

pub struct LinearW {
    /// Transposed: stored as `[in_features, out_features]` (B in matmul).
    pub weight: GpuBuffer,
    pub bias: GpuBuffer,
    pub in_features: u32,
    pub out_features: u32,
}

pub struct GroupNormW {
    pub gamma: GpuBuffer,
    pub beta: GpuBuffer,
    pub channels: u32,
    pub groups: u32,
}

/// Embedding stays on CPU: vocab is tiny (11) and we look up exactly one
/// row per generate(). Shipping it to the GPU as a one-hot matmul would
/// burn 11×64 ≈ 700 floats and a kernel dispatch for nothing.
pub struct EmbeddingW {
    pub data: Vec<f32>,
    pub vocab: u32,
    pub dim: u32,
}

impl EmbeddingW {
    /// Slice the row for a single token. Result has length `dim`.
    pub fn row(&self, id: u32) -> &[f32] {
        let start = (id * self.dim) as usize;
        let end = ((id + 1) * self.dim) as usize;
        &self.data[start..end]
    }
}

pub struct ResBlockW {
    pub conv1: ConvW,
    pub conv2: ConvW,
    pub gn1: GroupNormW,
    pub gn2: GroupNormW,
    pub time_proj: LinearW,
    pub skip_proj: Option<ConvW>,
    pub in_c: u32,
    pub out_c: u32,
}

pub struct CinderW {
    pub conv_in: ConvW,
    pub time_mlp1: LinearW,
    pub time_mlp2: LinearW,
    pub super_emb: EmbeddingW,
    pub tag_proj: LinearW,
    pub down_blocks: Vec<(ResBlockW, ResBlockW)>,
    pub downsamples: Vec<ConvW>,
    pub mid1: ResBlockW,
    pub mid2: ResBlockW,
    pub up_blocks: Vec<(ResBlockW, ResBlockW)>,
    pub upsamples: Vec<ConvW>,
    pub gn_out: GroupNormW,
    pub conv_out: ConvW,
    /// 3 for standalone Cinder; 6 for cinder-detail (silo conditioning).
    pub in_channels: u32,
}

impl CinderW {
    pub fn load(dev: &GpuDevice, tensors: &[LoadedTensor]) -> Result<Self> {
        // Auto-detect 6ch cinder-detail variant from conv_in.weight shape.
        let conv_in_w = find(tensors, "conv_in.weight")?;
        let in_channels = conv_in_w.shape.get(1).copied().context("conv_in.weight rank")?;
        let conv_in = load_conv(dev, tensors, "conv_in", in_channels, CHANNELS[0], 3, 3, (1, 1), (1, 1))?;

        let time_mlp1 = load_linear(dev, tensors, "time_mlp1", TIME_DIM, TIME_DIM)?;
        let time_mlp2 = load_linear(dev, tensors, "time_mlp2", TIME_DIM, TIME_DIM)?;

        let super_emb = load_embedding(tensors, "super_emb", NUM_SUPER_WITH_NULL as u32, TIME_DIM)?;
        let tag_proj = load_linear(dev, tensors, "tag_proj", NUM_TAGS as u32, TIME_DIM)?;

        // Encoder
        let mut down_blocks = Vec::new();
        let mut downsamples = Vec::new();
        let mut ch_in = CHANNELS[0];
        for (i, &ch_out) in CHANNELS.iter().enumerate() {
            let r1 = load_resblock(dev, tensors, &format!("down{i}_r1"), ch_in, ch_out)?;
            let r2 = load_resblock(dev, tensors, &format!("down{i}_r2"), ch_out, ch_out)?;
            down_blocks.push((r1, r2));
            if i < CHANNELS.len() - 1 {
                downsamples.push(load_conv(
                    dev, tensors, &format!("down{i}_ds.conv"),
                    ch_out, ch_out, 3, 3, (2, 2), (1, 1),
                )?);
            }
            ch_in = ch_out;
        }

        // Bottleneck
        let mid_ch = *CHANNELS.last().unwrap();
        let mid1 = load_resblock(dev, tensors, "mid1", mid_ch, mid_ch)?;
        let mid2 = load_resblock(dev, tensors, "mid2", mid_ch, mid_ch)?;

        // Decoder — channels reversed; first decoder block sees mid_ch from
        // the bottleneck stacked with the matching skip.
        let mut up_blocks = Vec::new();
        let mut upsamples = Vec::new();
        let rev: Vec<u32> = CHANNELS.iter().rev().copied().collect();
        for (i, &ch_out) in rev.iter().enumerate() {
            let skip_ch = if i == 0 { mid_ch + ch_out } else { rev[i - 1] + ch_out };
            let r1 = load_resblock(dev, tensors, &format!("up{i}_r1"), skip_ch, ch_out)?;
            let r2 = load_resblock(dev, tensors, &format!("up{i}_r2"), ch_out, ch_out)?;
            up_blocks.push((r1, r2));
            if i < rev.len() - 1 {
                upsamples.push(load_conv(
                    dev, tensors, &format!("up{i}_us.conv"),
                    ch_out, ch_out, 3, 3, (1, 1), (1, 1),
                )?);
            }
        }

        let gn_out = load_group_norm(dev, tensors, "gn_out", CHANNELS[0])?;
        let conv_out = load_conv(dev, tensors, "conv_out", CHANNELS[0], 3, 3, 3, (1, 1), (1, 1))?;

        Ok(Self {
            conv_in, time_mlp1, time_mlp2, super_emb, tag_proj,
            down_blocks, downsamples,
            mid1, mid2,
            up_blocks, upsamples,
            gn_out, conv_out,
            in_channels,
        })
    }
}

fn load_conv(
    dev: &GpuDevice,
    tensors: &[LoadedTensor],
    prefix: &str,
    in_c: u32, out_c: u32,
    kh: u32, kw: u32,
    stride: (u32, u32), padding: (u32, u32),
) -> Result<ConvW> {
    let w = find(tensors, &format!("{prefix}.weight"))?;
    let b = find(tensors, &format!("{prefix}.bias"))?;
    let expected = (out_c * in_c * kh * kw) as usize;
    if w.data.len() != expected {
        return Err(anyhow!(
            "{prefix}.weight has {} floats, expected {} ({}x{}x{}x{})",
            w.data.len(), expected, out_c, in_c, kh, kw
        ));
    }
    if b.data.len() != out_c as usize {
        return Err(anyhow!("{prefix}.bias has {} floats, expected {}", b.data.len(), out_c));
    }
    Ok(ConvW {
        weight: dev.upload(&w.data),
        bias: dev.upload(&b.data),
        in_c, out_c, kh, kw, stride, padding,
    })
}

fn load_linear(
    dev: &GpuDevice,
    tensors: &[LoadedTensor],
    prefix: &str,
    in_features: u32, out_features: u32,
) -> Result<LinearW> {
    let w = find(tensors, &format!("{prefix}.weight"))?;
    let b = find(tensors, &format!("{prefix}.bias"))?;
    let in_f = in_features as usize;
    let out_f = out_features as usize;
    if w.data.len() != in_f * out_f {
        return Err(anyhow!(
            "{prefix}.weight has {} floats, expected {} ({}x{})",
            w.data.len(), in_f * out_f, out_f, in_f
        ));
    }
    if b.data.len() != out_f {
        return Err(anyhow!("{prefix}.bias has {} floats, expected {}", b.data.len(), out_f));
    }

    // Transpose [out, in] → [in, out] so matmul(x[B,in], W_T[in,out]) works
    // directly. Done once at load.
    let mut w_t = vec![0.0f32; in_f * out_f];
    for r in 0..out_f {
        for c in 0..in_f {
            w_t[c * out_f + r] = w.data[r * in_f + c];
        }
    }

    Ok(LinearW {
        weight: dev.upload(&w_t),
        bias: dev.upload(&b.data),
        in_features, out_features,
    })
}

fn load_group_norm(
    dev: &GpuDevice,
    tensors: &[LoadedTensor],
    prefix: &str,
    channels: u32,
) -> Result<GroupNormW> {
    let gamma = find(tensors, &format!("{prefix}.weight"))?;
    let beta = find(tensors, &format!("{prefix}.bias"))?;
    if gamma.data.len() != channels as usize {
        return Err(anyhow!("{prefix}.weight has {} floats, expected {}", gamma.data.len(), channels));
    }
    Ok(GroupNormW {
        gamma: dev.upload(&gamma.data),
        beta: dev.upload(&beta.data),
        channels,
        groups: group_count(channels),
    })
}

fn load_embedding(
    tensors: &[LoadedTensor],
    prefix: &str,
    vocab: u32, dim: u32,
) -> Result<EmbeddingW> {
    let t = find(tensors, &format!("{prefix}.weight"))?;
    let expected = (vocab * dim) as usize;
    if t.data.len() != expected {
        return Err(anyhow!(
            "{prefix}.weight has {} floats, expected {} ({}x{})",
            t.data.len(), expected, vocab, dim
        ));
    }
    Ok(EmbeddingW { data: t.data.clone(), vocab, dim })
}

fn load_resblock(
    dev: &GpuDevice,
    tensors: &[LoadedTensor],
    prefix: &str,
    in_c: u32, out_c: u32,
) -> Result<ResBlockW> {
    let conv1 = load_conv(dev, tensors, &format!("{prefix}.conv1"), in_c, out_c, 3, 3, (1, 1), (1, 1))?;
    let conv2 = load_conv(dev, tensors, &format!("{prefix}.conv2"), out_c, out_c, 3, 3, (1, 1), (1, 1))?;
    let gn1 = load_group_norm(dev, tensors, &format!("{prefix}.gn1"), in_c)?;
    let gn2 = load_group_norm(dev, tensors, &format!("{prefix}.gn2"), out_c)?;
    let time_proj = load_linear(dev, tensors, &format!("{prefix}.time_proj"), TIME_DIM, out_c)?;
    let skip_proj = if in_c != out_c {
        Some(load_conv(dev, tensors, &format!("{prefix}.skip"), in_c, out_c, 1, 1, (1, 1), (0, 0))?)
    } else {
        None
    };
    Ok(ResBlockW { conv1, conv2, gn1, gn2, time_proj, skip_proj, in_c, out_c })
}

/// Group count for GroupNorm — must divide channel count evenly. Mirrors the
/// helper in `pixel-forge/src/tiny_unet.rs`.
fn group_count(channels: u32) -> u32 {
    if channels % 8 == 0 { 8 }
    else if channels % 4 == 0 { 4 }
    else { 1 }
}
