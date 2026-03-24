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
fn ddim_step(
    model: &dyn DiffusionModel,
    x: &Tensor,
    amount: f32,
    next_amount: f32,
    class_tensor: &Tensor,
    null_class: &Tensor,
    cfg_scale: f64,
    device: &candle_core::Device,
    v_pred: bool,
) -> candle_core::Result<Tensor> {
    let t = Tensor::new(&[amount], device)?;
    let raw_out = train::cfg_denoise(model, x, &t, class_tensor, null_class, cfg_scale)?;
    let pred_clean = if v_pred {
        (x - (raw_out * amount as f64)?)?
    } else {
        raw_out
    };

    if next_amount > 1e-6 {
        let noise = ((x - (&pred_clean * (1.0 - amount as f64))?)? * (1.0 / amount.max(1e-6) as f64))?;
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
    class_id: u32,
    img_size: u32,
    count: u32,
    config: &CascadeConfig,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let total_steps = config.quench_steps + config.cinder_steps;

    // Load Quench (foundation — runs first)
    println!("loading Quench from {quench_path}{}...",
        if crate::quantize::is_f16(quench_path) { " (f16→f32)" } else { "" });
    let quench_classes = train::detect_class_count_pub(quench_path).unwrap_or(crate::medium_unet::NUM_CLASSES);
    let mut quench_vm = VarMap::new();
    let quench_vb = VarBuilder::from_varmap(&quench_vm, DType::F32, &device);
    let quench = MediumUNet::with_classes(quench_vb, quench_classes)?;
    let quench_v_pred = train::detect_v_pred(quench_path);
    crate::quantize::load_varmap(&mut quench_vm, quench_path)?;

    // Load Cinder (detail — runs second)
    println!("loading Cinder from {cinder_path}{}...",
        if crate::quantize::is_f16(cinder_path) { " (f16→f32)" } else { "" });
    let cinder_classes = train::detect_class_count_pub(cinder_path).unwrap_or(crate::tiny_unet::NUM_CLASSES);
    let mut cinder_vm = VarMap::new();
    let cinder_vb = VarBuilder::from_varmap(&cinder_vm, DType::F32, &device);
    let cinder = TinyUNet::with_classes(cinder_vb, cinder_classes)?;
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

    // CFG setup
    let has_null_q = quench_classes > 15;
    let has_null_c = cinder_classes > 15;
    let cfg_scale = train::DEFAULT_CFG_SCALE;
    let max_class = (quench_classes.min(cinder_classes) - 1) as u32;
    let class_tensor = Tensor::new(&[class_id.min(max_class)], &device)?;
    let null_class = Tensor::new(&[train::CFG_NULL_CLASS], &device)?;

    println!("cascade: {} Quench steps → {} Cinder+Expert steps = {} total, cfg={}",
        config.quench_steps, config.cinder_steps, total_steps, cfg_scale);

    let mut images = Vec::new();

    for i in 0..count {
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, img_size as usize, img_size as usize), &device)?;

        // Phase 1: Quench foundation (structure, shapes, composition)
        let q_cfg = if has_null_q { cfg_scale } else { 1.0 };
        for step in 0..config.quench_steps {
            let t_frac = 1.0 - (step as f32 / total_steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / total_steps as f32);
            let next_amount = train::cosine_schedule(next_frac);

            x = ddim_step(&quench, &x, amount, next_amount, &class_tensor, &null_class, q_cfg, &device, quench_v_pred)?;
        }

        // Phase 2: Cinder detail (edges, highlights, fine pixel work)
        let c_cfg = if has_null_c { cfg_scale } else { 1.0 };
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
            let raw_out = train::cfg_denoise(&cinder, &x, &t, &class_tensor, &null_class, c_cfg)?;
            let pred_clean = if cinder_v_pred {
                (&x - (raw_out * amount as f64)?)?
            } else {
                raw_out
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
    class_id: u32,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();

    println!("loading Anvil from {anvil_path}{}...",
        if crate::quantize::is_f16(anvil_path) { " (f16→f32)" } else { "" });
    let anvil_classes = train::detect_class_count_pub(anvil_path).unwrap_or(crate::anvil_unet::NUM_CLASSES);
    let has_null = anvil_classes > 15;
    let mut anvil_vm = VarMap::new();
    let anvil_vb = VarBuilder::from_varmap(&anvil_vm, DType::F32, &device);
    let anvil = AnvilUNet::new(anvil_vb)?;
    let anvil_v_pred = train::detect_v_pred(anvil_path);
    crate::quantize::load_varmap(&mut anvil_vm, anvil_path)?;

    let cfg_scale = if has_null { train::DEFAULT_CFG_SCALE } else { 1.0 };
    let max_class = (anvil_classes - 1) as u32;
    let class_tensor = Tensor::new(&[class_id.min(max_class)], &device)?;
    let null_class = Tensor::new(&[train::CFG_NULL_CLASS], &device)?;
    let mut images = Vec::new();

    println!("anvil: {} steps, {} samples, cfg={}", steps, count, cfg_scale);

    for i in 0..count {
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, img_size as usize, img_size as usize), &device)?;

        for step in 0..steps {
            let t_frac = 1.0 - (step as f32 / steps as f32);
            let amount = train::cosine_schedule(t_frac);
            let next_frac = 1.0 - ((step + 1) as f32 / steps as f32);
            let next_amount = if step + 1 < steps { train::cosine_schedule(next_frac) } else { 0.0 };

            x = ddim_step(&anvil, &x, amount, next_amount, &class_tensor, &null_class, cfg_scale, &device, anvil_v_pred)?;
        }

        let img = tensor_to_rgba(&x.clamp(0.0, 1.0)?, img_size)?;
        images.push(img);
        println!("  anvil sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// MoE cascade with discriminator quality gate.
pub fn cascade_with_gate(
    quench_path: &str,
    cinder_path: &str,
    experts_path: Option<&str>,
    disc_path: &str,
    class_id: u32,
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
            class_id, img_size, batch, config,
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
pub fn anvil_with_gate(
    anvil_path: &str,
    disc_path: &str,
    class_id: u32,
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

        let sprites = anvil_sample(anvil_path, class_id, img_size, batch, steps)?;

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
