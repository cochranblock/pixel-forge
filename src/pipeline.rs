// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Diffusion pipeline — SDXL Turbo via candle for pixel art generation.
//! Metal (Apple Silicon GPU) auto-detected. 1-step inference.
//! Downloads model on first run, caches via hf-hub.

use anyhow::Result;
use hf_hub::api::sync::Api;
use image::RgbaImage;

/// SD v1.4 — open access, no auth required. Works with standard CLIP.
const MODEL_REPO: &str = "CompVis/stable-diffusion-v1-4";

pub const PIXEL_ART_SUFFIX: &str = ", pixel art, 16-bit style, game sprite, clean pixels, \
    no anti-aliasing, transparent background, centered, stardew valley style, \
    starbound style, detailed shading, warm palette";
pub const NEGATIVE_PROMPT: &str = "blurry, 3d, realistic, photo, smooth, gradient, \
    anti-aliased, text, watermark, signature, low quality, deformed";

/// Fewer steps for speed — 4 steps with higher guidance for pixel art.
const NUM_STEPS: usize = 4;
const GUIDANCE_SCALE: f64 = 9.0;

/// Auto-detect best device: Metal GPU on Apple Silicon, CPU fallback.
pub fn best_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            match candle_core::Device::new_cuda(0) {
                Ok(d) => {
                    println!("  device: CUDA (NVIDIA GPU)");
                    return d;
                }
                Err(e) => eprintln!("  cuda init failed: {e}, falling back"),
            }
        }
    }
    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            match candle_core::Device::new_metal(0) {
                Ok(d) => {
                    println!("  device: Metal (Apple Silicon GPU)");
                    return d;
                }
                Err(e) => eprintln!("  metal init failed: {e}, falling back to CPU"),
            }
        }
    }
    println!("  device: CPU (slow — build with --features cuda or --features metal)");
    candle_core::Device::Cpu
}

/// Check if model is cached.
pub fn model_ready() -> bool {
    let api = match Api::new() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let repo = api.model(MODEL_REPO.into());
    repo.get("unet/diffusion_pytorch_model.safetensors")
        .is_ok()
}

/// Download and cache SD model. Called by `pixel-forge setup`.
pub fn download_model() -> Result<()> {
    println!("downloading from {MODEL_REPO}...");
    println!("(~4GB total, first time only)");

    let api = Api::new()?;
    let repo = api.model(MODEL_REPO.into());

    let files = [
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "text_encoder/model.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ];

    for file in &files {
        print!("  {file}... ");
        match repo.get(file) {
            Ok(_) => println!("ok"),
            Err(e) => println!("skip ({e})"),
        }
    }

    println!("model cached");
    Ok(())
}

/// Generate images — test pattern if no model, SDXL Turbo if cached.
pub fn generate(prompt: &str, size: u32, count: u32) -> Result<Vec<RgbaImage>> {
    println!("prompt: \"{prompt}\"");

    if model_ready() {
        println!("SD ready — {NUM_STEPS}-step Metal inference");
        generate_turbo(prompt, size, count)
    } else {
        println!("no model cached — run `pixel-forge setup` first");
        println!("using test pattern to verify pipeline...");
        let mut images = Vec::new();
        for frame in 0..count {
            images.push(generate_test_pattern(size, frame, count));
        }
        Ok(images)
    }
}

/// SDXL Turbo inference — 1 step, no guidance, Metal GPU.
fn generate_turbo(prompt: &str, size: u32, count: u32) -> Result<Vec<RgbaImage>> {
    use candle_core::{DType, Module, Tensor};
    use candle_transformers::models::stable_diffusion::{self, schedulers::Scheduler};

    let device = best_device();
    let dtype = DType::F16; // FP16 for speed + lower memory

    // Generate at 256x256 — we're downscaling to pixel art anyway
    let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, Some(256), Some(256));

    let api = Api::new()?;
    let repo = api.model(MODEL_REPO.into());

    let full_prompt = format!("{prompt}{PIXEL_ART_SUFFIX}");

    // 1. Tokenize — SDXL uses two CLIP encoders, we use the primary one
    let clip_repo = api.model("openai/clip-vit-large-patch14".into());
    let tokenizer_path = clip_repo.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let encode = |text: &str| -> Result<Vec<i64>> {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        ids.truncate(77);
        while ids.len() < 77 {
            ids.push(49407);
        }
        Ok(ids)
    };

    let prompt_tokens = encode(&full_prompt)?;
    let uncond_tokens = encode("")?;

    // 2. Text encoder
    let clip_weights = repo.get("text_encoder/model.safetensors")?;
    let text_model = stable_diffusion::build_clip_transformer(
        &sd_config.clip,
        clip_weights,
        &device,
        dtype,
    )?;
    println!("  text encoder loaded");

    let text_embeddings = text_model.forward(
        &Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?,
    )?;
    let uncond_embeddings = text_model.forward(
        &Tensor::new(uncond_tokens.as_slice(), &device)?.unsqueeze(0)?,
    )?;
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?;
    drop(text_model);

    // 3. UNet
    let unet_weights = repo.get("unet/diffusion_pytorch_model.safetensors")?;
    let unet = sd_config.build_unet(unet_weights, &device, 4, false, dtype)?;
    println!("  unet loaded");

    // 4. VAE
    let vae_weights = repo.get("vae/diffusion_pytorch_model.safetensors")?;
    let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
    println!("  vae loaded");

    // 5. Scheduler — 1 step for Turbo
    let mut scheduler = sd_config.build_scheduler(NUM_STEPS)?;
    let timesteps = scheduler.timesteps().to_vec();

    let latent_h = (sd_config.height / 8) as usize;
    let latent_w = (sd_config.width / 8) as usize;

    let mut images = Vec::new();
    let t0 = std::time::Instant::now();

    for i in 0..count {
        let frame_start = std::time::Instant::now();

        // Random latent noise
        let latents = Tensor::randn(0f32, 1f32, (1, 4, latent_h, latent_w), &device)?
            .to_dtype(dtype)?;
        let mut latents = (latents * scheduler.init_noise_sigma())?;

        // Denoise — just 1 step for Turbo
        for &timestep in &timesteps {
            let latent_input = if GUIDANCE_SCALE > 0.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let noise_pred = unet.forward(&latent_input, timestep as f64, &text_embeddings)?;

            if GUIDANCE_SCALE > 0.0 {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let guided = (&noise_pred[0]
                    + ((&noise_pred[1] - &noise_pred[0])? * GUIDANCE_SCALE)?)?;
                latents = scheduler.step(&guided, timestep, &latents)?;
            } else {
                latents = scheduler.step(&noise_pred, timestep, &latents)?;
            }
        }

        // VAE decode
        let decoded = vae.decode(&(&latents / 0.18215)?)?;
        let decoded = ((decoded / 2.)? + 0.5)?
            .clamp(0., 1.)?
            .to_dtype(DType::F32)?;
        let decoded = (decoded * 255.)?.to_dtype(DType::U8)?;
        let decoded = decoded.squeeze(0)?;

        let (channels, h, w) = decoded.dims3()?;
        let pixels = decoded
            .permute((1, 2, 0))?
            .flatten_all()?
            .to_vec1::<u8>()?;

        let img = if channels == 3 {
            let mut rgba = RgbaImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    rgba.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgba([pixels[idx], pixels[idx + 1], pixels[idx + 2], 255]),
                    );
                }
            }
            rgba
        } else {
            RgbaImage::from_raw(w as u32, h as u32, pixels)
                .ok_or_else(|| anyhow::anyhow!("image from tensor failed"))?
        };

        let resized =
            image::imageops::resize(&img, size, size, image::imageops::FilterType::Triangle);
        images.push(resized);

        println!(
            "  frame {}/{count} done ({:.1}s)",
            i + 1,
            frame_start.elapsed().as_secs_f32()
        );
    }

    println!(
        "total: {count} frames in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    Ok(images)
}

/// Test pattern — verifies palette/grid/sheet pipeline without a model.
fn generate_test_pattern(size: u32, frame: u32, total_frames: u32) -> RgbaImage {
    use image::Rgba;

    let mut img = RgbaImage::new(size, size);
    let offset = if total_frames > 1 {
        (frame * 2) % size
    } else {
        0
    };

    for y in 0..size {
        for x in 0..size {
            let ax = (x + offset) % size;
            let cx = size / 2;
            let cy = size / 2;
            let dx = (ax as i32 - cx as i32).unsigned_abs();
            let dy = (y as i32 - cy as i32).unsigned_abs();

            let pixel = if dx < size / 6 && dy < size / 3 {
                Rgba([180, 100, 60, 255])
            } else if dx < size / 4 && dy < size / 6 {
                Rgba([220, 180, 140, 255])
            } else if dy > size / 3 && dx < size / 5 {
                Rgba([80, 80, 120, 255])
            } else {
                Rgba([0, 0, 0, 0])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img
}
