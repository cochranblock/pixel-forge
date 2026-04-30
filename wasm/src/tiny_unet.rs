// Unlicense — cochranblock.org
//! TinyUNet forward pass on any-gpu.
//!
//! Mirrors `pixel-forge/src/tiny_unet.rs`. Same architecture, same operator
//! sequence, same shapes. Differences:
//!   - Operates on raw `GpuBuffer` instead of Candle `Tensor`.
//!   - Linear weights are pre-transposed at load (see weights.rs).
//!   - Embedding lookup is CPU-side (one row per generate).
//!   - Sinusoidal timestep embedding is precomputed CPU-side (timestep.rs).
//!
//! Batch is fixed at 1 for now. CFG (B=2 cond/uncond) is a later optimization.

use any_gpu::{GpuBuffer, GpuDevice};
use anyhow::Result;

use crate::class_cond::NUM_TAGS;
use crate::timestep;
use crate::weights::{CinderW, ConvW, GroupNormW, LinearW, ResBlockW, TIME_DIM};

/// Run one denoising step.
///
/// - `x`: noisy image, shape `[1, in_channels, h, w]` (in_channels matches
///   the loaded model — 3 for Cinder, 6 for cinder-detail).
/// - `timestep`: scalar in [0, 1].
/// - `super_id`: super-category index (10 = CFG null).
/// - `tags`: 12-element binary trait vector.
/// - `h`, `w`: spatial dimensions of x.
///
/// Returns predicted noise (or clean image, depending on training objective)
/// in the same shape as `x`.
pub fn forward(
    dev: &GpuDevice,
    w: &CinderW,
    x: &GpuBuffer,
    timestep: f32,
    super_id: u32,
    tags: &[f32; NUM_TAGS],
    h: u32,
    w_dim: u32,
) -> Result<GpuBuffer> {
    let batch = 1u32;

    // === Time + class conditioning ===
    // [1, TIME_DIM] sinusoid
    let t_emb_cpu = timestep::embed(&[timestep], TIME_DIM as usize);
    let t_emb_buf = dev.upload(&t_emb_cpu);
    // time_mlp1 → silu → time_mlp2
    let t_emb_buf = linear(dev, &w.time_mlp1, &t_emb_buf, batch)?;
    let t_emb_buf = dev.swish(&t_emb_buf)?;
    let t_emb_buf = linear(dev, &w.time_mlp2, &t_emb_buf, batch)?;

    // class embedding: super_emb[id] + tag_proj(tags)
    let super_row = w.super_emb.row(super_id);
    let super_buf = dev.upload(super_row);
    let tag_buf = dev.upload(tags);
    let tag_buf = linear(dev, &w.tag_proj, &tag_buf, batch)?;
    let c_emb = dev.add(&super_buf, &tag_buf)?;
    let t_emb = dev.add(&t_emb_buf, &c_emb)?;

    // === Encoder ===
    let mut h_buf = conv2d(dev, &w.conv_in, x, batch, h, w_dim)?;
    let mut cur_h = h;
    let mut cur_w = w_dim;

    let mut skips: Vec<(GpuBuffer, u32, u32, u32)> = Vec::new(); // (buf, ch, h, w)
    for (i, (r1, r2)) in w.down_blocks.iter().enumerate() {
        h_buf = res_forward(dev, r1, &h_buf, &t_emb, batch, cur_h, cur_w)?;
        h_buf = res_forward(dev, r2, &h_buf, &t_emb, batch, cur_h, cur_w)?;
        skips.push((dup(dev, &h_buf, batch * r2.out_c * cur_h * cur_w)?, r2.out_c, cur_h, cur_w));
        if let Some(ds) = w.downsamples.get(i) {
            h_buf = conv2d(dev, ds, &h_buf, batch, cur_h, cur_w)?;
            cur_h /= 2;
            cur_w /= 2;
        }
    }

    // === Bottleneck ===
    h_buf = res_forward(dev, &w.mid1, &h_buf, &t_emb, batch, cur_h, cur_w)?;
    h_buf = res_forward(dev, &w.mid2, &h_buf, &t_emb, batch, cur_h, cur_w)?;

    // === Decoder ===
    let total_levels = w.up_blocks.len();
    for (i, (r1, r2)) in w.up_blocks.iter().enumerate() {
        // Upsample first (after the first decoder level), then concat skip,
        // then run the two ResBlocks. Mirrors the original ordering.
        if i > 0 {
            let up_idx = i - 1;
            let up_conv = &w.upsamples[up_idx];
            let upsampled = dev.upsample_nearest2d(&h_buf, batch, up_conv.in_c, cur_h, cur_w, 2, 2)?;
            cur_h *= 2;
            cur_w *= 2;
            h_buf = conv2d(dev, up_conv, &upsampled, batch, cur_h, cur_w)?;
        }
        // Pop matching skip (last-in first-out)
        let (skip_buf, skip_ch, skip_h, skip_w) = skips.remove(skips.len() - 1);
        debug_assert_eq!(skip_h, cur_h);
        debug_assert_eq!(skip_w, cur_w);

        // Concat along channel dim. NCHW with B=1 → outer=1, inner = ch * h * w each.
        let in_ch_pre_concat = if i == 0 {
            // After mid blocks, channels = mid_ch (last in CHANNELS).
            *crate::weights::CHANNELS.last().unwrap()
        } else {
            // After previous decoder level's r2 (out_c).
            w.up_blocks[i - 1].1.out_c
        };
        let outer = batch;
        let a_inner = in_ch_pre_concat * cur_h * cur_w;
        let b_inner = skip_ch * cur_h * cur_w;
        h_buf = dev.concat(&h_buf, &skip_buf, outer, a_inner, b_inner)?;

        h_buf = res_forward(dev, r1, &h_buf, &t_emb, batch, cur_h, cur_w)?;
        h_buf = res_forward(dev, r2, &h_buf, &t_emb, batch, cur_h, cur_w)?;
        let _ = total_levels;
    }

    // === Output ===
    h_buf = group_norm(dev, &w.gn_out, &h_buf, batch, cur_h, cur_w)?;
    h_buf = dev.swish(&h_buf)?;
    h_buf = conv2d(dev, &w.conv_out, &h_buf, batch, cur_h, cur_w)?;

    Ok(h_buf)
}

/// Single ResBlock: GN → SiLU → Conv → +time → GN → SiLU → Conv → +residual.
fn res_forward(
    dev: &GpuDevice,
    r: &ResBlockW,
    x: &GpuBuffer,
    t_emb: &GpuBuffer,
    batch: u32,
    h: u32,
    w_dim: u32,
) -> Result<GpuBuffer> {
    // Residual path. When `skip_proj` is None we just reuse `x` directly —
    // no clone needed since the rest of the function only reads it.
    let residual_owned;
    let residual: &GpuBuffer = match &r.skip_proj {
        Some(skip) => {
            residual_owned = conv2d(dev, skip, x, batch, h, w_dim)?;
            &residual_owned
        }
        None => x,
    };

    // Main path
    let h_buf = group_norm(dev, &r.gn1, x, batch, h, w_dim)?;
    let h_buf = dev.swish(&h_buf)?;
    let h_buf = conv2d(dev, &r.conv1, &h_buf, batch, h, w_dim)?;

    // Time projection [B, out_c] → broadcast over (H, W)
    let t = linear(dev, &r.time_proj, t_emb, batch)?;
    let t = dev.swish(&t)?;
    let h_buf = dev.add_broadcast(&h_buf, &t, batch * r.out_c, h * w_dim)?;

    let h_buf = group_norm(dev, &r.gn2, &h_buf, batch, h, w_dim)?;
    let h_buf = dev.swish(&h_buf)?;
    let h_buf = conv2d(dev, &r.conv2, &h_buf, batch, h, w_dim)?;

    dev.add(&h_buf, residual)
}

/// Linear: y = x · W_T + b. W is pre-transposed at load.
fn linear(dev: &GpuDevice, l: &LinearW, x: &GpuBuffer, batch: u32) -> Result<GpuBuffer> {
    let y = dev.matmul(x, &l.weight, batch, l.out_features, l.in_features)?;
    dev.add_per_col(&y, &l.bias, batch, l.out_features)
}

fn conv2d(dev: &GpuDevice, c: &ConvW, x: &GpuBuffer, batch: u32, h: u32, w: u32) -> Result<GpuBuffer> {
    dev.conv2d(
        x, &c.weight, Some(&c.bias),
        batch, c.in_c, h, w,
        c.out_c, c.kh, c.kw,
        c.stride, c.padding, (1, 1), 1,
    )
}

fn group_norm(dev: &GpuDevice, g: &GroupNormW, x: &GpuBuffer, batch: u32, h: u32, w: u32) -> Result<GpuBuffer> {
    dev.group_norm(x, &g.gamma, &g.beta, batch, g.channels, h * w, g.groups, 1e-5)
}

/// any-gpu has no clone() on GpuBuffer — duplicate by adding to a zero buffer.
/// Used for residual connections and for capturing skip activations before
/// a downsample mutates the active feature map.
fn dup(dev: &GpuDevice, x: &GpuBuffer, len: u32) -> Result<GpuBuffer> {
    let zero = dev.alloc(len as usize);
    dev.add(x, &zero)
}
