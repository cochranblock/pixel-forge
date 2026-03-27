// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Expert head training — freeze Quench base, train tiny expert corrections.
//! Each expert trains in minutes on 52K tiles.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{self as nn, Optimizer, VarBuilder, VarMap};
use rand::seq::SliceRandom;

use crate::expert::{self, ExpertSet};
use crate::medium_unet::MediumUNet;
use crate::train::preprocess;

/// Train all four expert heads on frozen Quench base.
pub fn train_experts(
    quench_path: &str,
    data_dir: &str,
    output: &str,
    epochs: usize,
    batch_size: usize,
) -> Result<()> {
    let device = crate::pipeline::best_device();
    let dtype = DType::F32;

    // Load frozen Quench
    println!("loading frozen Quench from {quench_path}...");
    let mut base_vm = VarMap::new();
    let base_vb = VarBuilder::from_varmap(&base_vm, dtype, &device);
    let base = MediumUNet::with_classes(base_vb, 15)?;
    base_vm.load(quench_path)?;
    println!("  Quench loaded (frozen)");

    // Load dataset
    let dataset = preprocess(data_dir, 16)?;
    let n = dataset.labels.len();
    let stride = dataset.stride;
    println!("  dataset: {} images", n);

    // Create expert weights
    let expert_vm = VarMap::new();
    let expert_vb = VarBuilder::from_varmap(&expert_vm, dtype, &device);
    let experts = ExpertSet::new(expert_vb)?;
    let expert_params = expert_vm.all_vars().len();
    println!("  experts: {} var groups", expert_params);

    let mut opt = nn::AdamW::new(expert_vm.all_vars(), nn::ParamsAdamW {
        lr: 1e-3,
        weight_decay: 0.01,
        ..Default::default()
    })?;

    let total_steps = 40usize;

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::thread_rng());

        let mut epoch_loss = 0.0f64;
        let mut batches = 0;

        let num_batches = (n + batch_size - 1) / batch_size;
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n);
            let bs = end - start;

            let mut batch_px = Vec::with_capacity(bs * stride);
            let mut batch_labels = Vec::with_capacity(bs);
            for &idx in &indices[start..end] {
                let src = idx * stride;
                batch_px.extend_from_slice(&dataset.pixels[src..src + stride]);
                batch_labels.push(dataset.labels[idx]);
            }

            let x_clean = Tensor::new(batch_px.as_slice(), &device)?
                .reshape((bs, 3, 16, 16))?;
            let y_batch = Tensor::new(batch_labels.as_slice(), &device)?;

            // Pick a random denoising step to simulate
            let step = rand::random::<usize>() % total_steps;
            let noise_level = 1.0 - (step as f32 / total_steps as f32);
            let noise = Tensor::rand(0f32, 1f32, x_clean.shape(), &device)?;
            let x_noisy = ((&x_clean * (1.0 - noise_level as f64))? + (&noise * noise_level as f64)?)?;

            let t = Tensor::new(vec![noise_level; bs].as_slice(), &device)?;

            // Build conditioning tensors from labels (use super_id=label, null tags for legacy compat)
            let tags_batch = Tensor::zeros((bs, crate::class_cond::NUM_TAGS), DType::F32, &device)?;

            // Encode with frozen base (no grad through base)
            let (features, skips, t_emb) = base.encode(&x_noisy, &t, &y_batch, &tags_batch)?;
            let features = features.detach();
            let skips: Vec<Tensor> = skips.iter().map(|s| s.detach()).collect();
            let t_emb = t_emb.detach();

            // Apply expert correction based on step
            let stage = expert::route(step, total_steps);
            let corrected = experts.correct(&features, stage)?;

            // Decode
            let pred = base.decode(&corrected, &skips, &t_emb)?;

            // Loss: MSE against clean image
            let loss = (&pred - &x_clean)?.sqr()?.mean_all()?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()? as f64;
            batches += 1;
        }

        let avg = if batches > 0 { epoch_loss / batches as f64 } else { 0.0 };
        if epoch % 2 == 0 || epoch == epochs - 1 {
            println!("  epoch {}/{}: loss={:.6} ({:.1}s)",
                epoch + 1, epochs, avg, t0.elapsed().as_secs_f32());
        }
    }

    // Save
    expert::save_experts(&expert_vm, output)?;
    let size = std::fs::metadata(output)?.len();
    println!("saved experts: {} ({:.1} KB)", output, size as f64 / 1024.0);
    Ok(())
}
