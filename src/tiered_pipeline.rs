// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//! Tiered Pipeline — pyramid of specialist models.
//!
//! Stage 1 (SHAPE):   Per-class MicroUNet silo → coarse sprite → noise blend seed
//! Stage 2 (PALETTE): PaletteNet → class-appropriate palette (Phase 2, not yet wired)
//! Stage 3 (DETAIL):  Cinder (TinyUNet) denoises from silo-seeded start
//! Gate:              Palette quantization post-generation
//!
//! When no silo exists for a class, falls back to pure Cinder generation.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::RgbaImage;
use std::path::{Path, PathBuf};

use crate::class_cond::ClassCond;
use crate::class_router::ClassRouter;
use crate::palette;
use crate::quantize;
use crate::train::{self, cosine_schedule};
use crate::tiny_unet::TinyUNet;

/// Configuration for tiered generation.
pub struct TieredConfig {
    /// Directory containing per-class MicroUNet models. Also checked for class_config.toml.
    pub silo_dir: PathBuf,
    /// Cinder model path (Stage 3). Use pixel-forge-cinder-detail.safetensors when available.
    pub detail_model: PathBuf,
    /// Palette name for output quantization (e.g. "endesga32").
    pub palette_name: String,
    /// Silo DDIM steps (Stage 1). Fewer steps = faster, noisier coarse sprite.
    pub silo_steps: usize,
    /// Cinder DDIM steps (Stage 3). Applied over the full [0,1] noise range when no silo.
    pub detail_steps: usize,
    /// Noise level to blend silo output into before Cinder refinement.
    /// 0.45 preserves ~55% silo structure; lower = more silo influence.
    pub refine_from_t: f32,
    /// CFG guidance scale for Cinder.
    pub cfg_scale: f64,
    /// Seed for deterministic generation.
    pub seed: Option<u64>,
    /// Optional PaletteNet model path (Phase 2).
    /// When set and the file exists, PaletteNet predicts class-specific colors
    /// that replace the full palette for output quantization.
    pub palette_model: Option<PathBuf>,
    /// Number of colors PaletteNet was trained to predict (must match model).
    pub palette_colors: usize,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            silo_dir: PathBuf::from("models"),
            detail_model: PathBuf::from("pixel-forge-cinder.safetensors"),
            palette_name: "endesga32".into(),
            silo_steps: 20,
            detail_steps: 40,
            refine_from_t: 0.45,
            cfg_scale: 3.0,
            seed: None,
            palette_model: None,
            palette_colors: 8,
        }
    }
}

/// Resolve the best available silo for a class name.
///
/// Search order:
/// 1. `class_config.toml` in `silo_dir` (uses ClassRouter for full routing)
/// 2. `{silo_dir}/{class}.safetensors` direct file
///
/// Returns `(model_path, channel_config)` or `None` if no silo exists.
pub fn resolve_silo(class: &str, silo_dir: &Path) -> Option<(PathBuf, Vec<usize>)> {
    let config_path = silo_dir.join("class_config.toml");
    if config_path.exists() {
        if let Ok(router) = ClassRouter::load(config_path.to_str().unwrap_or("")) {
            if let Some(dc) = router.route(class) {
                // Paths in TOML are relative to CWD (same convention as `silo` command)
                let model_path = PathBuf::from(&dc.model_path);
                if model_path.exists() {
                    return Some((model_path, dc.channels.clone()));
                }
                // Also try relative to silo_dir
                let alt = silo_dir.join(&dc.model_path);
                if alt.exists() {
                    return Some((alt, dc.channels.clone()));
                }
            }
        }
    }

    // Direct fallback: {silo_dir}/{class}.safetensors
    let direct = silo_dir.join(format!("{class}.safetensors"));
    if direct.exists() {
        return Some((direct, vec![16, 32]));
    }

    None
}

/// Convert an RgbaImage to a (1, 3, H, W) f32 Tensor in [0.0, 1.0].
fn rgba_to_tensor(img: &RgbaImage, device: &candle_core::Device) -> candle_core::Result<Tensor> {
    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    let mut pixels = vec![0.0f32; 3 * n];
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
            let base = (y * w + x) as usize;
            pixels[base]       = px[0] as f32 / 255.0; // R
            pixels[n + base]   = px[1] as f32 / 255.0; // G
            pixels[2 * n + base] = px[2] as f32 / 255.0; // B
        }
    }
    Tensor::from_vec(pixels, (1_usize, 3_usize, h as usize, w as usize), device)
}

/// Convert a (1, 3, H, W) f32 Tensor in [0.0, 1.0] to RgbaImage.
fn tensor_to_rgba(x: &Tensor) -> candle_core::Result<RgbaImage> {
    let x = x.clamp(0.0, 1.0)?;
    let (_, _, h, w) = x.dims4()?;
    let px8 = (x.squeeze(0)? * 255.0)?.to_dtype(DType::U8)?
        .permute((1, 2, 0))?
        .flatten_all()?.to_vec1::<u8>()?;
    let mut img = RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for xi in 0..w {
            let i = (y * w + xi) * 3;
            img.put_pixel(xi as u32, y as u32, image::Rgba([px8[i], px8[i+1], px8[i+2], 255]));
        }
    }
    Ok(img)
}

/// 3-stage tiered generation pipeline.
///
/// Each sprite goes through:
/// 1. Silo MicroUNet (Stage 1) — if available for the class
/// 2. Noise blend at `refine_from_t` — turns silo output into a partially-noised seed
/// 3. Cinder DDIM from the noise-blended start (Stage 3)
/// 4. Palette quantization
pub fn tiered_generate(cond: &ClassCond, config: &TieredConfig, count: u32) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let img_size: u32 = 32;

    // Stage 2: PaletteNet — load if available, predict class-specific palette
    let base_pal = palette::load_palette(&config.palette_name)?;
    let pal = if let Some(ref pm_path) = config.palette_model {
        if pm_path.exists() {
            match crate::palette_net::load(pm_path.to_str().unwrap_or(""), config.palette_colors, &device) {
                Ok(pnet) => {
                    match pnet.predict_palette(cond.super_id, &cond.tags, &base_pal, &device) {
                        Ok(class_pal) if !class_pal.is_empty() => {
                            let hex: Vec<String> = class_pal.iter()
                                .map(|c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]))
                                .collect();
                            println!("tiered: Stage 2 PaletteNet → {}", hex.join(" "));
                            class_pal
                        }
                        _ => {
                            println!("tiered: Stage 2 PaletteNet predict failed, using full palette");
                            base_pal
                        }
                    }
                }
                Err(e) => {
                    println!("tiered: Stage 2 PaletteNet load failed ({e}), using full palette");
                    base_pal
                }
            }
        } else {
            base_pal
        }
    } else {
        base_pal
    };

    // Stage 1: resolve silo
    let silo = resolve_silo(&cond.name, &config.silo_dir);
    match &silo {
        Some((path, channels)) => {
            let ch_str: Vec<String> = channels.iter().map(|c| c.to_string()).collect();
            println!("tiered: Stage 1 silo → {} [{}]", path.display(), ch_str.join(","));
        }
        None => println!("tiered: no silo for '{}' — pure Cinder fallback", cond.name),
    }

    // Stage 3: load Cinder (detail model)
    let detail_path = config.detail_model.to_str().unwrap_or("pixel-forge-cinder.safetensors");
    let is_f16 = quantize::is_f16(detail_path);
    let in_ch = quantize::detect_in_channels(detail_path);
    let is_conditioned = in_ch == 6;
    println!("tiered: Stage 3 Cinder from {detail_path} ({}ch{})",
        in_ch, if is_f16 { ", f16→f32" } else { "" });

    let mut cinder_vm = VarMap::new();
    let cinder_vb = VarBuilder::from_varmap(&cinder_vm, DType::F32, &device);
    let cinder = TinyUNet::with_channels(cinder_vb, in_ch)?;
    let cinder_v_pred = train::detect_v_pred(detail_path);
    quantize::load_varmap(&mut cinder_vm, detail_path)?;

    let (super_t, tags_t, null_super, null_tags) = train::cond_tensors(cond, 1, &device)?;

    let detail_steps = config.detail_steps;

    // Precompute the step to begin Cinder from when silo is available.
    // We want the step where cosine_schedule(t_frac) first drops to or below refine_from_t.
    let start_step_with_silo = if silo.is_some() {
        (0..detail_steps).find(|&s| {
            let t_frac = 1.0 - (s as f32 / detail_steps as f32);
            cosine_schedule(t_frac) <= config.refine_from_t
        }).unwrap_or(detail_steps / 2)
    } else {
        0
    };

    let mut images = Vec::new();

    for i in 0..count {
        let per_seed = config.seed.map(|s| s + i as u64);

        // Stage 1: if silo, generate coarse sprite and noise-blend it
        // Also capture clean silo tensor for 6ch Cinder conditioning.
        let (start_x, cond_hint) = if let Some((ref silo_path, ref channels)) = silo {
            let coarse = train::sample_micro(
                silo_path.to_str().unwrap_or(""),
                channels,
                img_size,
                1,
                config.silo_steps,
                per_seed,
            )?;
            let silo_t = rgba_to_tensor(&coarse[0], &device)?;

            // Noise blend: x = silo * (1-t) + noise * t
            let noise = train::seeded_noise(per_seed, cond.super_id, i, img_size, &device)?;
            let t = config.refine_from_t as f64;
            println!(
                "  blend: {:.0}% silo + {:.0}% noise (refine_from_t={:.2})",
                (1.0 - t) * 100.0, t * 100.0, config.refine_from_t
            );
            let blended = ((&silo_t * (1.0 - t))? + (&noise * t)?)?;
            (blended, Some(silo_t))
        } else {
            let noise = train::seeded_noise(per_seed, cond.super_id, i, img_size, &device)?;
            (noise, None)
        };

        // Stage 3: Cinder DDIM starting from start_step_with_silo
        let start_step = if silo.is_some() { start_step_with_silo } else { 0 };
        let mut x = start_x;

        for step in start_step..detail_steps {
            let t_frac = 1.0 - (step as f32 / detail_steps as f32);
            let amount = cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / detail_steps as f32);
            let next_amount = if step + 1 < detail_steps { cosine_schedule(next_frac) } else { 0.0 };

            // 6ch conditioned: prepend clean silo hint; fall back to zeros if no silo.
            let model_in = if is_conditioned {
                let hint = match &cond_hint {
                    Some(h) => h.clone(),
                    None => Tensor::zeros_like(&x)?,
                };
                Tensor::cat(&[&hint, &x], 1)? // (1, 6, H, W)
            } else {
                x.clone()
            };

            let t_tensor = Tensor::new(&[amount], &device)?;
            let pred_clean = if cinder_v_pred {
                train::cfg_denoise_vpred(
                    &cinder, &model_in, &t_tensor, amount,
                    &super_t, &tags_t, &null_super, &null_tags, config.cfg_scale,
                )?
            } else {
                train::cfg_denoise(
                    &cinder, &model_in, &t_tensor,
                    &super_t, &tags_t, &null_super, &null_tags, config.cfg_scale,
                )?
            };

            if next_amount > 1e-6 {
                let recovered_noise = ((&x - (&pred_clean * (1.0 - amount as f64))?)? *
                    (1.0 / amount.max(1e-6) as f64))?.clamp(-3.0, 3.0)?;
                x = ((&pred_clean * (1.0 - next_amount as f64))? +
                    (&recovered_noise * next_amount as f64)?)?;
            } else {
                x = pred_clean;
            }
        }

        let raw_img = tensor_to_rgba(&x)?;
        let quantized = palette::quantize(&raw_img, &pal);
        images.push(quantized);
        println!("  tiered {}/{count}", i + 1);
    }

    Ok(images)
}
