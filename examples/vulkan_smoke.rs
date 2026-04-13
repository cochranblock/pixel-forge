// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Smoke test: initialize the Vulkan backend and run one training step on
//! synthetic data. Run with:
//!   cargo run --features vulkan --example vulkan_smoke
//! On Mac this uses Metal via wgpu; on bt it uses Vulkan against the AMD 5700 XT.

use pixel_forge::vulkan_backend::{
    VulkanMicroParams, ParamLeaves, probe_vulkan, micro_forward, timestep_embedding,
    vulkan_micro_train_step,
};
use any_gpu::{GpuDevice, optim::AdamW, autograd::Tape};

fn main() -> anyhow::Result<()> {
    println!("probe: {}", probe_vulkan()?);

    let dev = GpuDevice::gpu()?;
    let channels = vec![16u32, 32u32];
    let mut params = VulkanMicroParams::new(&channels);
    println!("MicroUNet: {} params, channels={:?}", params.param_count(), channels);

    let mut opt = AdamW::new(2e-4);
    opt.weight_decay = 0.01;

    let batch: u32 = 2;
    let img: u32 = 32;
    let stride = (3 * img * img) as usize;

    // Synthetic clean data — a spatial gradient so it's non-trivial.
    let clean: Vec<f32> = (0..batch as usize * stride)
        .map(|i| (i % 256) as f32 / 255.0)
        .collect();
    // Random-ish noise.
    let noise: Vec<f32> = (0..batch as usize * stride)
        .map(|i| {
            let x = (i as f32 * 0.37).sin();
            x * 0.5
        })
        .collect();
    let noise_amounts = vec![0.3f32, 0.7f32];

    // First, try a bare forward pass to localize any shape bug.
    println!("forward-only pass…");
    let mut tape = Tape::new(&dev);
    let mut leaves = ParamLeaves { ids: Vec::new(), sizes: Vec::new() };
    let stride_px = (3 * img * img) as usize;
    let mut noisy = vec![0.0f32; batch as usize * stride_px];
    for b in 0..batch as usize {
        let t = noise_amounts[b];
        for i in 0..stride_px {
            noisy[b * stride_px + i] = clean[b * stride_px + i] * (1.0 - t) + noise[b * stride_px + i] * t;
        }
    }
    let x_id = tape.leaf(&noisy);
    let t_sin = timestep_embedding(&noise_amounts, pixel_forge::vulkan_backend::TIME_DIM);
    let t_id = tape.leaf(&t_sin);
    match micro_forward(&mut tape, &params, &mut leaves, x_id, t_id, batch, img) {
        Ok(pred) => {
            let v = tape.read(pred)?;
            println!("  forward OK, pred len = {}", v.len());
        }
        Err(e) => {
            eprintln!("  forward FAILED: {}", e);
            return Err(e);
        }
    }

    println!("running 3 training steps…");
    for step in 0..3 {
        let loss = vulkan_micro_train_step(
            &dev, &mut opt, &mut params,
            &clean, &noise, &noise_amounts,
            batch, img,
        )?;
        println!("  step {}: loss={:.6}", step + 1, loss);
    }
    println!("smoke test passed");
    Ok(())
}
