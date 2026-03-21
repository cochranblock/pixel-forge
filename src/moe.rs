// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! MoE cascade pipeline — Cinder drafts, Quench + Experts refines.
//!
//! Full pipeline:
//!   1. Cinder runs steps 1-N (fast draft from noise)
//!   2. Hand tensor to Quench
//!   3. Quench encodes → expert corrects bottleneck → Quench decodes
//!   4. Repeat for remaining steps with stage-routed experts
//!
//! Two models + four experts = 27 MB total. Sub-second on Metal.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::RgbaImage;

use crate::expert;
use crate::medium_unet::MediumUNet;
use crate::tiny_unet::TinyUNet;

/// Cascade config.
pub struct CascadeConfig {
    /// Steps for Cinder (draft phase).
    pub cinder_steps: usize,
    /// Steps for Quench + experts (refine phase).
    pub quench_steps: usize,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            cinder_steps: 10,
            quench_steps: 30,
        }
    }
}

/// Full MoE cascade: Cinder → Quench + Experts.
pub fn cascade_sample(
    cinder_path: &str,
    quench_path: &str,
    experts_path: Option<&str>,
    class_id: u32,
    img_size: u32,
    count: u32,
    config: &CascadeConfig,
) -> Result<Vec<RgbaImage>> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;
    let total_steps = config.cinder_steps + config.quench_steps;

    // Load Cinder
    println!("loading Cinder from {cinder_path}...");
    let mut cinder_vm = VarMap::new();
    let cinder_vb = VarBuilder::from_varmap(&cinder_vm, dtype, &device);
    let cinder = TinyUNet::new(cinder_vb)?;
    cinder_vm.load(cinder_path)?;

    // Load Quench
    println!("loading Quench from {quench_path}...");
    let mut quench_vm = VarMap::new();
    let quench_vb = VarBuilder::from_varmap(&quench_vm, dtype, &device);
    let quench = MediumUNet::new(quench_vb)?;
    quench_vm.load(quench_path)?;

    // Load experts (optional — works without them, just no correction)
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

    println!("cascade: {} Cinder steps → {} Quench+Expert steps = {} total",
        config.cinder_steps, config.quench_steps, total_steps);

    let class_tensor = Tensor::new(&[class_id], &device)?;
    let mut images = Vec::new();

    for i in 0..count {
        // Start from noise
        let mut x = Tensor::rand(0f32, 1f32, (1, 3, img_size as usize, img_size as usize), &device)?;

        // Phase 1: Cinder draft (fast, rough shape)
        for step in 0..config.cinder_steps {
            let noise_level = 1.0 - (step as f32 / total_steps as f32);
            let t = Tensor::new(&[noise_level], &device)?;
            let pred = cinder.forward(&x, &t, &class_tensor)?;
            let mix = 1.0 / (total_steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix)?)?;
        }

        // Phase 2: Quench + Experts (encode → correct → decode)
        for step in config.cinder_steps..total_steps {
            let noise_level = 1.0 - (step as f32 / total_steps as f32);
            let t = Tensor::new(&[noise_level], &device)?;

            // Encode
            let (mut features, skips, t_emb) = quench.encode(&x, &t, &class_tensor)?;

            // Expert correction (if experts loaded)
            if let Some(ref exp) = experts {
                let stage = expert::route(step - config.cinder_steps, config.quench_steps);
                features = exp.correct(&features, stage)?;
            }

            // Decode
            let pred = quench.decode(&features, &skips, &t_emb)?;
            let mix = 1.0 / (total_steps - step) as f64;
            x = ((&x * (1.0 - mix))? + (&pred * mix)?)?;
        }

        // To image
        let x = x.clamp(0.0, 1.0)?;
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
        images.push(rgba);
        println!("  cascade sample {}/{count}", i + 1);
    }

    Ok(images)
}

/// MoE cascade with discriminator quality gate.
/// Rejects sprites where discriminator score < threshold (0.94 ≈ loss < 0.06).
/// Re-rolls until enough pass or max_attempts exhausted.
pub fn cascade_with_gate(
    cinder_path: &str,
    quench_path: &str,
    experts_path: Option<&str>,
    disc_path: &str,
    class_id: u32,
    img_size: u32,
    count: u32,
    config: &CascadeConfig,
    threshold: f32,
    max_attempts: u32,
) -> Result<Vec<image::RgbaImage>> {
    let device = crate::pipeline::best_device();

    // Load discriminator
    let (_dvm, disc) = crate::discriminator::load(disc_path, &device)?;
    println!("discriminator loaded, threshold={:.3}", threshold);

    let mut accepted = Vec::new();
    let mut attempts = 0u32;
    let batch_size = count.max(4); // generate in batches

    while (accepted.len() as u32) < count && attempts < max_attempts {
        let need = count - accepted.len() as u32;
        let batch = need.min(batch_size);

        let sprites = cascade_sample(
            cinder_path, quench_path, experts_path,
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
