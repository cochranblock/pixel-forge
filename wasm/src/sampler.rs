// Unlicense — cochranblock.org
//! DDIM-style sampling loop ported from `pixel-forge/src/train.rs`.
//!
//! v0 simplifications (relative to the desktop sampler):
//!   - No CFG (cfg_scale = 1.0). Model is called once per step.
//!   - No skeleton init. Pure Gaussian noise via Box-Muller.
//!   - No clamp(-3, 3) on the noise estimate. any-gpu has no min/max kernel
//!     yet; the original clamp is a guard against runaway values at low
//!     timesteps. Late-step quality may differ vs. desktop until we add
//!     clamp; visually fine in practice for 32×32.
//!   - No v-pred branch. Cinder is non-v-pred.
//!
//! These are all small wins to add later; they are not blockers for v0.

use any_gpu::{GpuBuffer, GpuDevice};
use anyhow::Result;
use rand::Rng;
use rand::SeedableRng;

use crate::class_cond::NUM_TAGS;
use crate::tiny_unet;
use crate::weights::CinderW;

/// Cosine schedule used during training. `t` is in [0, 1].
/// Mirrors `train.rs::cosine_schedule` — keep in sync if the desktop side
/// changes its `s` offset.
fn cosine_schedule(t: f32) -> f32 {
    let s = 0.008f32;
    let half_pi = std::f32::consts::FRAC_PI_2;
    let f_t = ((t + s) / (1.0 + s) * half_pi).cos().powi(2);
    let f_0 = (s / (1.0 + s) * half_pi).cos().powi(2);
    1.0 - (f_t / f_0).min(1.0)
}

/// Seeded Gaussian noise via Box-Muller. Produces `n` samples with mean 0,
/// stddev 1. Stable across browsers and matches the desktop seed→content
/// contract: same seed + class → same starting noise.
pub fn gaussian_noise(seed: u64, super_id: u32, index: u32, n: usize) -> Vec<f32> {
    let combined = seed
        .wrapping_mul(2654435761)
        ^ (super_id as u64).wrapping_mul(40503)
        ^ (index as u64).wrapping_mul(65537);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(combined);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

/// Run the full DDIM loop and return the final clean prediction (still on
/// the GPU). Caller is responsible for readback + post-processing.
///
/// - `img_size`: 32 for Cinder.
/// - `steps`: ~40 is the desktop default.
pub async fn sample(
    dev: &GpuDevice,
    w: &CinderW,
    super_id: u32,
    tags: &[f32; NUM_TAGS],
    img_size: u32,
    steps: usize,
    seed: u64,
) -> Result<GpuBuffer> {
    let n = (3 * img_size * img_size) as usize;
    let noise = gaussian_noise(seed, super_id, 0, n);
    let mut x = dev.upload(&noise);

    let max_noise = 1.0f32;

    for step in 0..steps {
        let t_frac = 1.0 - (step as f32 / steps as f32);
        let amount = cosine_schedule(t_frac) * max_noise;
        let next_amount = if step + 1 < steps {
            let next_frac = 1.0 - ((step + 1) as f32 / steps as f32);
            cosine_schedule(next_frac) * max_noise
        } else {
            0.0
        };

        let pred_clean = tiny_unet::forward(
            dev, w, &x, amount, super_id, tags, img_size, img_size,
        )?;

        if next_amount > 1e-6 {
            // noise = (x - pred_clean * (1 - amount)) / max(amount, 1e-6)
            let scaled_clean = dev.scale(&pred_clean, 1.0 - amount)?;
            let noise = dev.sub(&x, &scaled_clean)?;
            let noise = dev.scale(&noise, 1.0 / amount.max(1e-6))?;

            // x = pred_clean * (1 - next_amount) + noise * next_amount
            let next_clean = dev.scale(&pred_clean, 1.0 - next_amount)?;
            let next_noise = dev.scale(&noise, next_amount)?;
            x = dev.add(&next_clean, &next_noise)?;
        } else {
            x = pred_clean;
        }
    }

    Ok(x)
}

/// Convert raw NCHW=[1,3,H,W] f32 GPU buffer to a PNG byte vec. Reads back
/// asynchronously from the GPU. Values are clamped to [0, 1] and scaled to
/// 0–255.
pub async fn finalize_png(
    dev: &GpuDevice,
    x: &GpuBuffer,
    img_size: u32,
) -> Result<Vec<u8>> {
    let raw = dev.read_async(x).await?;
    let n = (img_size * img_size) as usize;
    if raw.len() != 3 * n {
        return Err(anyhow::anyhow!(
            "expected {} floats, got {} from GPU",
            3 * n, raw.len()
        ));
    }

    // NCHW=[1,3,H,W] → interleaved RGB bytes.
    let mut rgba = Vec::with_capacity(n * 4);
    for y in 0..img_size {
        for px in 0..img_size {
            let i = (y * img_size + px) as usize;
            let r = (raw[i].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (raw[n + i].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (raw[2 * n + i].clamp(0.0, 1.0) * 255.0) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }

    let mut out = Vec::with_capacity(rgba.len() + 256);
    {
        let mut encoder = png::Encoder::new(&mut out, img_size, img_size);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| anyhow::anyhow!("png header: {e}"))?;
        writer
            .write_image_data(&rgba)
            .map_err(|e| anyhow::anyhow!("png write: {e}"))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_schedule_endpoints() {
        assert!(cosine_schedule(0.0) < 0.02);
        assert!((cosine_schedule(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gaussian_noise_is_deterministic_for_same_seed() {
        let a = gaussian_noise(42, 0, 0, 100);
        let b = gaussian_noise(42, 0, 0, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn gaussian_noise_diverges_for_different_seeds() {
        let a = gaussian_noise(42, 0, 0, 100);
        let b = gaussian_noise(43, 0, 0, 100);
        assert_ne!(a, b);
    }
}
