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
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::RgbaImage;

use crate::expert::{self, ExpertSet};
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
