// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! MoE cascade pipelines.
//!
//! Mobile/embedded (2-stage): Quench lays foundation → Cinder adds detail.
//! Desktop (single-stage): Anvil does everything — 16M params, needs 3+ GB VRAM.
//!
//! Both pipelines use the same cosine-schedule DDIM sampler + CFG as training.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::RgbaImage;

use crate::anvil_unet::AnvilUNet;
use crate::expert;
use crate::medium_unet::MediumUNet;
use crate::tiny_unet::TinyUNet;
use crate::train::{self, DiffusionModel};

/// Cascade config for 2-stage Quench → Cinder pipeline.
pub struct CascadeConfig {
    /// Steps for Quench (foundation phase).
    pub quench_steps: usize,
    /// Steps for Cinder + experts (detail phase).
    pub cinder_steps: usize,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            quench_steps: 25,
            cinder_steps: 15,
        }
    }
}

/// DDIM step: predict clean → extract noise → recompose at next noise level.
/// When v_pred is true, the model output is velocity (noise - clean) and we
/// recover clean via: clean = noisy - amount * v_pred.
#[allow(clippy::too_many_arguments)]
fn ddim_step(
    model: &dyn DiffusionModel,
    x: &Tensor,
    amount: f32,
    next_amount: f32,
    super_id: &Tensor,
    tags: &Tensor,
    null_super: &Tensor,
    null_tags: &Tensor,
    cfg_scale: f64,
    device: &candle_core::Device,
    v_pred: bool,
) -> candle_core::Result<Tensor> {
    let t = Tensor::new(&[amount], device)?;
    let pred_clean = if v_pred {
        train::cfg_denoise_vpred(model, x, &t, amount, super_id, tags, null_super, null_tags, cfg_scale)?
    } else {
        train::cfg_denoise(model, x, &t, super_id, tags, null_super, null_tags, cfg_scale)?
    };

    if next_amount > 1e-6 {
        let noise = ((x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?
            .clamp(-3.0, 3.0)?;
        (&pred_clean * (1.0 - next_amount as f64))? + (&noise * next_amount as f64)?
    } else {
        Ok(pred_clean)
    }
}

/// 2-stage cascade: Quench foundation → Cinder detail + Experts.
pub fn cascade_sample(
    quench_path: &str,
    cinder_path: &str,
    experts_path: Option<&str>,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    config: &CascadeConfig,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let total_steps = config.quench_steps + config.cinder_steps;

    // Load Quench (foundation — runs first)
    println!("loading Quench from {quench_path}{}...",
        if crate::quantize::is_f16(quench_path) { " (f16→f32)" } else { "" });
    let mut quench_vm = VarMap::new();
    let quench_vb = VarBuilder::from_varmap(&quench_vm, DType::F32, &device);
    let quench = MediumUNet::new(quench_vb)?;
    let quench_v_pred = train::detect_v_pred(quench_path);
    crate::quantize::load_varmap(&mut quench_vm, quench_path)?;

    // Load Cinder (detail — runs second)
    println!("loading Cinder from {cinder_path}{}...",
        if crate::quantize::is_f16(cinder_path) { " (f16→f32)" } else { "" });
    let mut cinder_vm = VarMap::new();
    let cinder_vb = VarBuilder::from_varmap(&cinder_vm, DType::F32, &device);
    let cinder = TinyUNet::new(cinder_vb)?;
    let cinder_v_pred = train::detect_v_pred(cinder_path);
    crate::quantize::load_varmap(&mut cinder_vm, cinder_path)?;

    // Load experts (optional — applied during Cinder detail phase)
    let experts = if let Some(path) = experts_path {
        if std::path::Path::new(path).exists() {
            println!("loading experts from {path}...");
            let (_, exp) = expert::load_experts(path, &device)?;
            Some(exp)
        } else {
            println!("no experts file, running without expert correction");
            None
        }
    } else {
        None
    };

    // CFG setup — hybrid conditioning
    let cfg_scale = train::DEFAULT_CFG_SCALE;
    let (super_tensor, tags_tensor, null_super, null_tags) = train::cond_tensors(cond, 1, &device)?;

    println!("cascade: {} Quench steps → {} Cinder+Expert steps = {} total, cfg={}",
        config.quench_steps, config.cinder_steps, total_steps, cfg_scale);

    let mut images = Vec::new();

    for i in 0..count {
        let mut x = train::seeded_noise(None, cond.super_id, i, img_size, &device)?;

        // Phase 1: Quench foundation (structure, shapes, composition)
        for step in 0..config.quench_steps {
            let t_frac = 1.0 - (step as f32 / total_steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / total_steps as f32);
            let next_amount = train::cosine_schedule(next_frac);

            x = ddim_step(&quench, &x, amount, next_amount, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale, &device, quench_v_pred)?;
        }

        // Phase 2: Cinder detail (edges, highlights, fine pixel work)
        for step in config.quench_steps..total_steps {
            let t_frac = 1.0 - (step as f32 / total_steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_step = step + 1;
            let next_amount = if next_step < total_steps {
                let next_frac = 1.0 - (next_step as f32 / total_steps as f32);
                train::cosine_schedule(next_frac)
            } else {
                0.0
            };

            let t = Tensor::new(&[amount], &device)?;
            let pred_clean = if cinder_v_pred {
                train::cfg_denoise_vpred(&cinder, &x, &t, amount, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?
            } else {
                train::cfg_denoise(&cinder, &x, &t, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?
            };

            // Expert correction during detail phase
            let pred_clean = if let Some(ref exp) = experts {
                let stage = expert::route(step - config.quench_steps, config.cinder_steps);
                exp.correct(&pred_clean, stage)?
            } else {
                pred_clean
            };

            if next_amount > 1e-6 {
                let noise = ((&x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?;
                x = ((&pred_clean * (1.0 - next_amount as f64))? + (&noise * next_amount as f64)?)?;
            } else {
                x = pred_clean;
            }
        }

        let img = tensor_to_rgba(&x.clamp(0.0, 1.0)?, img_size)?;
        images.push(img);
        println!("  cascade sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// Desktop pipeline: Anvil handles everything in one stage.
pub fn anvil_sample(
    anvil_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();

    println!("loading Anvil from {anvil_path}{}...",
        if crate::quantize::is_f16(anvil_path) { " (f16→f32)" } else { "" });
    let mut anvil_vm = VarMap::new();
    let anvil_vb = VarBuilder::from_varmap(&anvil_vm, DType::F32, &device);
    let anvil = AnvilUNet::new(anvil_vb)?;
    let anvil_v_pred = train::detect_v_pred(anvil_path);
    crate::quantize::load_varmap(&mut anvil_vm, anvil_path)?;

    let cfg_scale = train::DEFAULT_CFG_SCALE;
    let (super_tensor, tags_tensor, null_super, null_tags) = train::cond_tensors(cond, 1, &device)?;
    let mut images = Vec::new();

    println!("anvil: {} steps, {} samples, cfg={}", steps, count, cfg_scale);

    for i in 0..count {
        let mut x = train::seeded_noise(None, cond.super_id, i, img_size, &device)?;

        for step in 0..steps {
            let t_frac = 1.0 - (step as f32 / steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / steps as f32);
            let next_amount = if step + 1 < steps { train::cosine_schedule(next_frac) } else { 0.0 };

            x = ddim_step(&anvil, &x, amount, next_amount, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale, &device, anvil_v_pred)?;
        }

        let img = tensor_to_rgba(&x.clamp(0.0, 1.0)?, img_size)?;
        images.push(img);
        println!("  anvil sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// MoE cascade with discriminator quality gate.
#[allow(clippy::too_many_arguments)]
pub fn cascade_with_gate(
    quench_path: &str,
    cinder_path: &str,
    experts_path: Option<&str>,
    disc_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    config: &CascadeConfig,
    threshold: f32,
    max_attempts: u32,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();

    let (_dvm, disc) = crate::discriminator::load(disc_path, &device)?;
    println!("discriminator loaded, threshold={:.3}", threshold);

    let mut accepted = Vec::new();
    let mut attempts = 0u32;
    let batch_size = count.max(4);

    while (accepted.len() as u32) < count && attempts < max_attempts {
        let need = count - accepted.len() as u32;
        let batch = need.min(batch_size);

        let sprites = cascade_sample(
            quench_path, cinder_path, experts_path,
            cond, img_size, batch, config,
        )?;

        for sprite in sprites {
            let (pass, score) = crate::discriminator::quality_gate(
                &disc, &sprite, threshold, &device,
            )?;
            attempts += 1;

            if pass {
                println!("  accepted (score={:.3}, attempt={})", score, attempts);
                accepted.push(sprite);
                if accepted.len() as u32 >= count { break; }
            } else {
                println!("  rejected (score={:.3} < {:.3}, attempt={})", score, threshold, attempts);
            }

            if attempts >= max_attempts { break; }
        }
    }

    if accepted.is_empty() {
        anyhow::bail!("no sprites passed quality gate after {} attempts (threshold={:.3})", attempts, threshold);
    }

    println!("cascade gate: {}/{} accepted ({} attempts)", accepted.len(), count, attempts);
    Ok(accepted)
}

/// Anvil with discriminator quality gate.
#[allow(clippy::too_many_arguments)]
pub fn anvil_with_gate(
    anvil_path: &str,
    disc_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
    threshold: f32,
    max_attempts: u32,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();

    let (_dvm, disc) = crate::discriminator::load(disc_path, &device)?;
    println!("discriminator loaded, threshold={:.3}", threshold);

    let mut accepted = Vec::new();
    let mut attempts = 0u32;
    let batch_size = count.max(4);

    while (accepted.len() as u32) < count && attempts < max_attempts {
        let need = count - accepted.len() as u32;
        let batch = need.min(batch_size);

        let sprites = anvil_sample(anvil_path, cond, img_size, batch, steps)?;

        for sprite in sprites {
            let (pass, score) = crate::discriminator::quality_gate(
                &disc, &sprite, threshold, &device,
            )?;
            attempts += 1;

            if pass {
                println!("  accepted (score={:.3}, attempt={})", score, attempts);
                accepted.push(sprite);
                if accepted.len() as u32 >= count { break; }
            } else {
                println!("  rejected (score={:.3} < {:.3}, attempt={})", score, threshold, attempts);
            }

            if attempts >= max_attempts { break; }
        }
    }

    if accepted.is_empty() {
        anyhow::bail!("no sprites passed quality gate after {} attempts (threshold={:.3})", attempts, threshold);
    }

    println!("anvil gate: {}/{} accepted ({} attempts)", accepted.len(), count, attempts);
    Ok(accepted)
}

/// Stage-aware cascade: Cinder-sil → structure map → Quench-detail (6ch).
/// The full pipeline: generate silhouette, compute SDF+normals, paint sprite.
pub fn stage_cascade_sample(
    sil_path: &str,
    detail_path: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    sil_steps: usize,
    detail_steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();

    // Load Cinder-sil (3ch → silhouette)
    println!("loading Cinder-sil from {sil_path}...");
    let mut sil_vm = VarMap::new();
    let sil_vb = VarBuilder::from_varmap(&sil_vm, DType::F32, &device);
    let sil_model = TinyUNet::new(sil_vb)?;
    crate::quantize::load_varmap(&mut sil_vm, sil_path)?;

    // Load Quench-detail (6ch → sprite)
    println!("loading Quench-detail from {detail_path}...");
    let mut det_vm = VarMap::new();
    let det_vb = VarBuilder::from_varmap(&det_vm, DType::F32, &device);
    let det_model = MediumUNet::with_channels(det_vb, 6)?;
    crate::quantize::load_varmap(&mut det_vm, detail_path)?;

    let cfg_scale = train::DEFAULT_CFG_SCALE;
    let (super_tensor, tags_tensor, null_super, null_tags) = train::cond_tensors(cond, 1, &device)?;
    let sz = img_size as usize;

    println!("stage cascade: {} sil steps → structure → {} detail steps", sil_steps, detail_steps);

    let mut images = Vec::new();

    for i in 0..count {
        // Phase 1: Generate silhouette from noise
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, sz, sz), &device)?;
        for step in 0..sil_steps {
            let t_frac = 1.0 - (step as f32 / sil_steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_amount = if step + 1 < sil_steps {
                let nf = 1.0 - ((step + 1) as f32 / sil_steps as f32);
                train::cosine_schedule(nf)
            } else { 0.0 };
            x = ddim_step(&sil_model, &x, amount, next_amount, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale, &device, false)?;
        }
        let sil = x.clamp(0.0, 1.0)?;

        // Phase 2: Threshold silhouette to binary, then compute structure
        // Average RGB channels, threshold at 0.3 to get clean binary mask
        let sil_mean = sil.mean_keepdim(1)?; // (1, 1, H, W)
        let threshold = Tensor::new(&[0.3f32], &device)?.broadcast_as(sil_mean.shape())?;
        let mask = sil_mean.ge(&threshold)?;  // boolean mask
        let ones = mask.ones_like()?.to_dtype(DType::F32)?;
        let zeros = mask.zeros_like()?.to_dtype(DType::F32)?;
        let binary = mask.where_cond(&ones, &zeros)?;
        let binary_3ch = Tensor::cat(&[&binary, &binary, &binary], 1)?; // (1, 3, H, W)
        let sil_img = tensor_to_rgba(&binary_3ch, img_size)?;
        let (sdf, nx, ny, _outline) = crate::relight::compute_structure(&sil_img);

        // Pack as 3ch tensor: R=SDF, G=0.5+0.5*nx, B=0.5+0.5*ny
        let mut struct_px = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for px in 0..sz {
                let idx = y * sz + px;
                struct_px[idx] = sdf[idx];                           // R = SDF
                struct_px[sz * sz + idx] = 0.5 + 0.5 * nx[idx];     // G = normal X
                struct_px[2 * sz * sz + idx] = 0.5 + 0.5 * ny[idx]; // B = normal Y
            }
        }
        let struct_tensor = Tensor::from_vec(struct_px, (1, 3, sz, sz), &device)?;

        // Phase 3: Quench-detail — 6ch input (structure + noisy sprite)
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, sz, sz), &device)?;
        for step in 0..detail_steps {
            let t_frac = 1.0 - (step as f32 / detail_steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_amount = if step + 1 < detail_steps {
                let nf = 1.0 - ((step + 1) as f32 / detail_steps as f32);
                train::cosine_schedule(nf)
            } else { 0.0 };

            // Concat structure + noisy: (1, 6, H, W)
            let input_6ch = Tensor::cat(&[&struct_tensor, &x], 1)?;
            let t = Tensor::new(&[amount], &device)?;
            let pred_clean = train::cfg_denoise(&det_model, &input_6ch, &t, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?;

            if next_amount > 1e-6 {
                let noise = ((&x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?;
                x = ((&pred_clean * (1.0 - next_amount as f64))? + (&noise * next_amount as f64)?)?;
            } else {
                x = pred_clean;
            }
        }

        let img = tensor_to_rgba(&x.clamp(0.0, 1.0)?, img_size)?;
        images.push(img);
        println!("  stage cascade sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// Detail-only: load real structure maps from disk, run Quench-detail only.
/// For testing — bypasses silhouette generation.
pub fn detail_only_sample(
    detail_path: &str,
    struct_dir: &str,
    class_name: &str,
    cond: &crate::class_cond::ClassCond,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let sz = img_size as usize;

    // Load Quench-detail (6ch)
    println!("loading Quench-detail from {detail_path}...");
    let mut det_vm = VarMap::new();
    let det_vb = VarBuilder::from_varmap(&det_vm, DType::F32, &device);
    let det_model = MediumUNet::with_channels(det_vb, 6)?;
    crate::quantize::load_varmap(&mut det_vm, detail_path)?;

    let cfg_scale = train::DEFAULT_CFG_SCALE;
    let (super_tensor, tags_tensor, null_super, null_tags) = train::cond_tensors(cond, 1, &device)?;

    // Load real structure PNGs
    let struct_class_dir = std::path::Path::new(struct_dir).join(class_name);
    let mut pngs: Vec<_> = std::fs::read_dir(&struct_class_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("png"))
        .collect();
    pngs.sort_by_key(|e| e.file_name());

    println!("detail-only: {} real structures from {}, {} steps", pngs.len().min(count as usize), struct_class_dir.display(), steps);

    let mut images = Vec::new();
    for i in 0..count.min(pngs.len() as u32) {
        let img = image::open(pngs[i as usize].path())?.to_rgb8();
        // Convert to tensor (channel-first, normalized)
        let mut px = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let p = img.get_pixel(x as u32, y as u32);
                let idx = y * sz + x;
                px[idx] = p[0] as f32 / 255.0;
                px[sz * sz + idx] = p[1] as f32 / 255.0;
                px[2 * sz * sz + idx] = p[2] as f32 / 255.0;
            }
        }
        let struct_tensor = Tensor::from_vec(px, (1, 3, sz, sz), &device)?;

        // Run Quench-detail: structure + noise → sprite
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, sz, sz), &device)?;
        for step in 0..steps {
            let t_frac = 1.0 - (step as f32 / steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_amount = if step + 1 < steps {
                let nf = 1.0 - ((step + 1) as f32 / steps as f32);
                train::cosine_schedule(nf)
            } else { 0.0 };

            let input_6ch = Tensor::cat(&[&struct_tensor, &x], 1)?;
            let t = Tensor::new(&[amount], &device)?;
            let pred_clean = train::cfg_denoise(&det_model, &input_6ch, &t, &super_tensor, &tags_tensor, &null_super, &null_tags, cfg_scale)?;

            if next_amount > 1e-6 {
                let noise = ((&x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?;
                x = ((&pred_clean * (1.0 - next_amount as f64))? + (&noise * next_amount as f64)?)?;
            } else {
                x = pred_clean;
            }
        }

        let img = tensor_to_rgba(&x.clamp(0.0, 1.0)?, img_size)?;
        images.push(img);
        println!("  detail sample {}/{}", i + 1, count);
    }

    Ok(images)
}

/// Convert a (1, 3, H, W) f32 tensor to an RGBA image.
fn tensor_to_rgba(x: &Tensor, img_size: u32) -> Result<RgbaImage> {
    let x = (x * 255.0)?.to_dtype(DType::U8)?;
    let x = x.squeeze(0)?.permute((1, 2, 0))?;
    let pixels = x.flatten_all()?.to_vec1::<u8>()?;

    let mut rgba = RgbaImage::new(img_size, img_size);
    for y in 0..img_size {
        for px in 0..img_size {
            let idx = (y * img_size + px) as usize * 3;
            rgba.put_pixel(px, y, image::Rgba([pixels[idx], pixels[idx + 1], pixels[idx + 2], 255]));
        }
    }
    Ok(rgba)
}
