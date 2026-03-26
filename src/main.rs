// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! pixel-forge — Pixel art game asset generator.
//! Stardew Valley / Starbound quality. Local-first, Rust-native.

use pixel_forge::*;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pixel-forge", about = "Pixel art game asset generator")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a sprite from a text description.
    Sprite {
        /// What to generate: "knight idle", "potion red", "tree oak"
        prompt: String,
        /// Output size in pixels (square). Default 32.
        #[arg(short, long, default_value_t = 32)]
        size: u32,
        /// Number of animation frames. 1 = static sprite.
        #[arg(short, long, default_value_t = 1)]
        frames: u32,
        /// Palette: stardew, starbound, snes, nes, gameboy, or custom .pal path.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file path.
        #[arg(short, long, default_value = "sprite.png")]
        output: String,
    },
    /// Generate a seamless tileset.
    Tileset {
        /// Theme: "dungeon", "forest", "cave", "village", "space"
        theme: String,
        /// Tile size in pixels (square). Default 16.
        #[arg(short, long, default_value_t = 32)]
        size: u32,
        /// Number of tile variations. Default 16.
        #[arg(short, long, default_value_t = 16)]
        count: u32,
        /// Palette name or path.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file path.
        #[arg(short, long, default_value = "tileset.png")]
        output: String,
    },
    /// List available palettes.
    Palettes,
    /// Download/cache the diffusion model.
    Setup,
    /// Train a tiny pixel art diffusion model from a dataset of PNGs.
    /// Dataset: directory of PNGs, optionally organized by class subdirectory.
    Train {
        /// Path to training data directory (PNGs, optionally in class subdirs).
        #[arg(short, long, default_value = "data")]
        data: String,
        /// Output model file path (.safetensors).
        #[arg(short, long, default_value = "pixel-forge-cinder.safetensors")]
        output: String,
        /// Number of training epochs.
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,
        /// Batch size.
        #[arg(short, long, default_value_t = 64)]
        batch_size: usize,
        /// Learning rate.
        #[arg(short, long, default_value_t = 1e-3)]
        lr: f64,
        /// Image size (square). Must match training data.
        #[arg(long, default_value_t = 32)]
        img_size: u32,
        /// Use MediumUNet (~5.8M params) instead of TinyUNet (~1.1M params).
        #[arg(long)]
        medium: bool,
        /// Use AnvilUNet (~16M params) — the XL model.
        #[arg(long)]
        anvil: bool,
        /// Disable EMA (saves ~50% RAM during training).
        #[arg(long)]
        no_ema: bool,
        /// Train with v-prediction (model predicts velocity instead of clean image).
        #[arg(long)]
        v_pred: bool,
        /// Minimum learning rate for cosine LR decay. 0 = flat LR.
        #[arg(long, default_value_t = 1e-5)]
        lr_min: f64,
        /// Warm-up epochs: linear ramp from lr_min to lr.
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        /// Conditioning data dir for stage-aware training (silhouettes, colorblocks).
        #[arg(long)]
        condition: Option<String>,
        /// Mixed precision: f16 forward pass, f32 optimizer. ~2x faster on CUDA.
        #[arg(long)]
        fp16: bool,
    },
    /// Curate raw downloaded datasets into class-sorted training directories.
    Curate {
        /// Path to raw downloads directory.
        #[arg(short, long, default_value = "data/raw")]
        raw: String,
        /// Path to output class directories.
        #[arg(short, long, default_value = "data")]
        output: String,
        /// Target tile size (square).
        #[arg(long, default_value_t = 32)]
        size: u32,
    },
    /// Precompute per-class skeleton images from training data.
    /// Skeletons give the sampler a head start — 70% structure, 30% noise.
    Skeletons {
        /// Path to training data directory.
        #[arg(short, long, default_value = "data")]
        data: String,
        /// Image size (must match training data).
        #[arg(long, default_value_t = 32)]
        img_size: u32,
    },
    /// Extract structural layers for stage-aware cascade.
    /// Generates 5-channel conditioning data: SDF, normal X/Y, outline, luminance.
    PrepStages {
        /// Path to training data directory with class subdirs.
        #[arg(short, long, default_value = "data_v2_32")]
        data: String,
        /// Number of palette colors for color block-in stage (unused, kept for compat).
        #[arg(long, default_value_t = 8)]
        palette_colors: usize,
    },
    /// Stage cascade: Cinder-sil → structure map → Quench-detail → sprite.
    StageCascade {
        /// Class to generate.
        class: String,
        /// Cinder-sil model (silhouette generator).
        #[arg(long, default_value = "pixel-forge-cinder-sil.safetensors")]
        sil_model: String,
        /// Quench-detail model (6ch conditioned detail filler).
        #[arg(long, default_value = "pixel-forge-quench-detail.safetensors")]
        detail_model: String,
        /// Number of sprites.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Silhouette generation steps.
        #[arg(long, default_value_t = 20)]
        sil_steps: usize,
        /// Detail generation steps.
        #[arg(long, default_value_t = 30)]
        detail_steps: usize,
        /// Palette.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file.
        #[arg(short, long, default_value = "stage-cascade.png")]
        output: String,
        /// Generate 4-directional sprite sheet via relighting.
        #[arg(long)]
        four_dir: bool,
    },
    /// Relight a sprite into 4-directional views (front/left/right/back).
    /// Uses SDF + normal maps — no model needed, pure math.
    Relight {
        /// Input sprite PNG.
        image: String,
        /// Output sprite sheet (4 views side by side).
        #[arg(short, long, default_value = "relit.png")]
        output: String,
        /// Also save individual direction PNGs (front.png, left.png, etc).
        #[arg(long)]
        split: bool,
    },
    /// Generate pixel art using a trained model (no SD required).
    Generate {
        /// Class to generate: character, weapon, potion, terrain, enemy, etc.
        class: String,
        /// Path to trained model (.safetensors).
        #[arg(short, long, default_value = "pixel-forge-cinder.safetensors")]
        model: String,
        /// Output size in pixels (must match training size).
        #[arg(short, long, default_value_t = 32)]
        size: u32,
        /// Number of images to generate.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Denoising steps (more = higher quality, slower).
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Palette for color quantization.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file path.
        #[arg(short, long, default_value = "generated.png")]
        output: String,
        /// Use MediumUNet (Quench) model architecture.
        #[arg(long)]
        medium: bool,
        /// Seed for deterministic generation. Same seed = same sprite.
        #[arg(long)]
        seed: Option<u64>,
        /// Generate 4-directional sprite sheet (front/left/right/back) via relighting.
        #[arg(long)]
        four_dir: bool,
    },
    /// Record a swipe (good/bad) on a generated sprite for Judge training.
    Swipe {
        /// Path to the sprite image (32×32 PNG).
        image: String,
        /// "good" or "bad" (swipe right or left).
        verdict: String,
        /// Class of the sprite.
        #[arg(short, long, default_value = "misc")]
        class: String,
        /// Path to swipe store file.
        #[arg(long, default_value = "swipes.bin")]
        store: String,
    },
    /// Train the Judge model from recorded swipes.
    TrainJudge {
        /// Path to swipe store file.
        #[arg(short, long, default_value = "swipes.bin")]
        store: String,
        /// Output Judge model path.
        #[arg(short, long, default_value = "judge.safetensors")]
        output: String,
    },
    /// Score sprites through a trained Judge — filter good from bad.
    Judge {
        /// Path to sprite images (or directory of PNGs).
        input: String,
        /// Path to trained Judge model.
        #[arg(short, long, default_value = "judge.safetensors")]
        model: String,
    },
    /// Generate a scene: 8×8 grid of sprites (128×128 composite).
    Scene {
        /// Generation mode: "bootstrap" (rule-seeded), "model" (trained Combiner), "seed" (user-seeded).
        #[arg(short, long, default_value = "bootstrap")]
        mode: String,
        /// Path to Combiner model (for mode=model).
        #[arg(short = 'M', long, default_value = "combiner.safetensors")]
        combiner_model: String,
        /// Biome for bootstrap: forest, dungeon, village, cave, plains.
        #[arg(short, long, default_value = "forest")]
        biome: String,
        /// Sampling temperature (lower = more deterministic).
        #[arg(short, long, default_value_t = 0.8)]
        temperature: f32,
        /// Number of scenes to generate.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Output file path.
        #[arg(short, long, default_value = "scene.png")]
        output: String,
    },
    /// Train the Combiner model on rule-seeded or user-accepted scenes.
    TrainCombiner {
        /// Number of bootstrap scenes to generate for training.
        #[arg(short, long, default_value_t = 500)]
        num_scenes: usize,
        /// Training epochs.
        #[arg(short, long, default_value_t = 20)]
        epochs: usize,
        /// Output model path.
        #[arg(short, long, default_value = "combiner.safetensors")]
        output: String,
    },
    /// Update Generator LoRA from Judge feedback.
    TrainLora {
        /// Path to base Generator model.
        #[arg(short, long, default_value = "pixel-forge-cinder.safetensors")]
        model: String,
        /// Path to trained Judge model.
        #[arg(short, long, default_value = "judge.safetensors")]
        judge: String,
        /// Output LoRA weights path.
        #[arg(short, long, default_value = "generator-lora.safetensors")]
        output: String,
        /// Class to generate during LoRA training.
        #[arg(short, long, default_value = "character")]
        class: String,
    },
    /// Show swipe store stats.
    SwipeStats {
        /// Path to swipe store file.
        #[arg(short, long, default_value = "swipes.bin")]
        store: String,
    },
    /// Full pipeline: generate → judge → combine → render scene.
    Pipeline {
        /// Path to Generator model.
        #[arg(short, long, default_value = "pixel-forge-cinder.safetensors")]
        model: String,
        /// Path to Judge model (skip judging if missing).
        #[arg(short, long, default_value = "judge.safetensors")]
        judge: String,
        /// Path to Combiner model (use bootstrap if missing).
        #[arg(short = 'M', long, default_value = "combiner.safetensors")]
        combiner: String,
        /// Path to LoRA weights (skip if missing).
        #[arg(short, long, default_value = "generator-lora.safetensors")]
        lora: String,
        /// Palette for color quantization.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output scene image path.
        #[arg(short, long, default_value = "scene.png")]
        output: String,
    },
    /// Train expert heads on frozen Quench base.
    TrainExperts {
        /// Path to frozen Quench model.
        #[arg(long, default_value = "pixel-forge-quench.safetensors")]
        quench: String,
        /// Training data directory.
        #[arg(short, long, default_value = "data")]
        data: String,
        /// Output experts file.
        #[arg(short, long, default_value = "experts.safetensors")]
        output: String,
        /// Epochs.
        #[arg(short, long, default_value_t = 20)]
        epochs: usize,
        /// Batch size.
        #[arg(short, long, default_value_t = 32)]
        batch_size: usize,
    },
    /// Auto-generate: detect device, pick best model, generate.
    Auto {
        /// Class to generate: character, weapon, potion, terrain, enemy, etc.
        class: String,
        /// Output size in pixels (must match training size).
        #[arg(short, long, default_value_t = 32)]
        size: u32,
        /// Number of images to generate.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Denoising steps.
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Palette for color quantization.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file path.
        #[arg(short, long, default_value = "generated.png")]
        output: String,
    },
    /// Re-probe device capabilities (clears cached profile).
    Probe {
        /// Output as JSON (for cluster probing).
        #[arg(long)]
        json: bool,
    },
    /// Probe all nodes in the forge cluster.
    ClusterProbe,
    /// Deploy pixel-forge binary to all nodes (sync + build).
    ClusterDeploy,
    /// Sync trained models to all nodes.
    ClusterSync,
    /// Generate across the full cluster.
    ClusterGenerate {
        /// Class to generate.
        class: String,
        /// Total sprites to generate (distributed across nodes).
        #[arg(short, long, default_value_t = 16)]
        count: u32,
        /// Denoising steps.
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Palette.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file path.
        #[arg(short, long, default_value = "cluster-output.png")]
        output: String,
    },
    /// GPU scheduling — delegates to kova c2 gpu. Pass args after --.
    Gpu {
        /// Arguments forwarded to `kova c2 gpu`.
        args: Vec<String>,
    },
    /// Forge: generate → discriminator gate → PoA sign → Ghost Fabric packet.
    Forge {
        /// Class to generate.
        class: String,
        /// Number of signed sprites to produce.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Denoising steps.
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// Discriminator quality threshold (0.94 ≈ loss < 0.06).
        #[arg(long, default_value_t = 0.94)]
        threshold: f32,
        /// Max generation attempts before giving up.
        #[arg(long, default_value_t = 50)]
        max_attempts: u32,
        /// Path to discriminator model.
        #[arg(long, default_value = "discriminator.safetensors")]
        disc: String,
        /// Palette for color quantization.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file path.
        #[arg(short, long, default_value = "forged.png")]
        output: String,
        /// GPS latitude (default: Dundalk, MD).
        #[arg(long, default_value_t = 39.2504)]
        lat: f32,
        /// GPS longitude (default: Dundalk, MD).
        #[arg(long, default_value_t = -76.5205)]
        lon: f32,
        /// Use MoE cascade (Quench → Cinder + Experts) instead of auto-detect.
        #[arg(long)]
        cascade: bool,
    },
    /// Sign an existing sprite with PoA — produce a Ghost Fabric packet.
    Sign {
        /// Path to 32x32 PNG sprite.
        image: String,
        /// Class of the sprite.
        #[arg(short, long, default_value = "misc")]
        class: String,
        /// Quality score override (0.0-1.0). If omitted, runs discriminator.
        #[arg(long)]
        score: Option<f32>,
        /// Path to discriminator model (used if --score not set).
        #[arg(long, default_value = "discriminator.safetensors")]
        disc: String,
        /// GPS latitude.
        #[arg(long, default_value_t = 39.2504)]
        lat: f32,
        /// GPS longitude.
        #[arg(long, default_value_t = -76.5205)]
        lon: f32,
    },
    /// Show node public key (for PoA verification).
    NodeKey,
    /// Plugin mode — JSON request/response for kova integration.
    Plugin {
        /// Keep process alive, read JSON lines continuously.
        #[arg(long)]
        r#loop: bool,
    },
    /// Quantize a model from f32 to f16 (halves file size + faster on ARM/Metal).
    Quantize {
        /// Input model file (f32 safetensors).
        input: String,
        /// Output file. Defaults to input-f16.safetensors.
        #[arg(short, long)]
        output: Option<String>,
    },
    /// MoE cascade: Quench foundation → Cinder detail + Experts.
    Cascade {
        /// Class to generate.
        class: String,
        /// Quench model path.
        #[arg(long, default_value = "pixel-forge-quench.safetensors")]
        quench: String,
        /// Cinder model path.
        #[arg(long, default_value = "pixel-forge-cinder.safetensors")]
        cinder: String,
        /// Expert weights path (optional).
        #[arg(long, default_value = "experts.safetensors")]
        experts: String,
        /// Number of images.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Quench foundation steps.
        #[arg(long, default_value_t = 25)]
        quench_steps: usize,
        /// Cinder detail steps.
        #[arg(long, default_value_t = 15)]
        cinder_steps: usize,
        /// Palette.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file.
        #[arg(short, long, default_value = "cascade.png")]
        output: String,
    },
    /// Upscale 16x16 PNGs from data/ to 32x32 in data_v2_32/ (nearest-neighbor).
    Upscale {
        /// Source directory with 16x16 class subdirs.
        #[arg(short, long, default_value = "data")]
        input: String,
        /// Destination directory for 32x32 output.
        #[arg(short, long, default_value = "data_v2_32")]
        output: String,
    },
    /// Desktop generation: Anvil single-stage (needs 3+ GB VRAM).
    Anvil {
        /// Class to generate.
        class: String,
        /// Anvil model path.
        #[arg(long, default_value = "pixel-forge-anvil.safetensors")]
        anvil: String,
        /// Number of images.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Diffusion steps.
        #[arg(short, long, default_value_t = 40)]
        steps: usize,
        /// Palette.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file.
        #[arg(short, long, default_value = "anvil.png")]
        output: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let cmd = match cli.cmd {
        Some(cmd) => cmd,
        None => return app::run(),
    };

    match cmd {
        Cmd::Sprite {
            prompt,
            size,
            frames,
            palette,
            output,
        } => {
            let pal = palette::load_palette(&palette)?;
            println!("generating {frames}x {size}x{size} sprite: \"{prompt}\"");
            println!("palette: {} ({} colors)", palette, pal.len());

            let raw_images = pipeline::generate(&prompt, size, frames)?;

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let resized = grid::snap_to_grid(&img, size);
                    palette::quantize(&resized, &pal)
                })
                .collect();

            if frames == 1 {
                processed[0].save(&output)?;
            } else {
                let sheet = sheet::pack_horizontal(&processed);
                sheet.save(&output)?;
            }
            println!("saved: {output}");
        }
        Cmd::Tileset {
            theme,
            size,
            count,
            palette,
            output,
        } => {
            let pal = palette::load_palette(&palette)?;
            println!("generating {count}x {size}x{size} tiles: \"{theme}\"");

            let mut tiles = Vec::new();
            for i in 0..count {
                let tile_prompt = format!("{theme} tile variation {}", i + 1);
                let raw = pipeline::generate(&tile_prompt, size, 1)?;
                let snapped = grid::snap_to_grid(&raw[0], size);
                let quantized = palette::quantize(&snapped, &pal);
                tiles.push(quantized);
            }

            let sheet = sheet::pack_grid(&tiles, 4);
            sheet.save(&output)?;
            println!("saved: {output}");
        }
        Cmd::Palettes => {
            palette::list_palettes();
        }
        Cmd::Setup => {
            pipeline::download_model()?;
            println!("model cached and ready");
        }
        Cmd::Curate { raw, output, size } => {
            curate::curate(&raw, &output, size)?;
        }
        Cmd::StageCascade { class, sil_model, detail_model, count, sil_steps, detail_steps, palette: palette_name, output, four_dir } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            let pal = palette::load_palette(&palette_name)?;
            let raw_images = moe::stage_cascade_sample(
                &sil_model, &detail_model, class_id, 32, count, sil_steps, detail_steps,
            )?;

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let snapped = grid::snap_to_grid(&img, 32);
                    palette::quantize(&snapped, &pal)
                })
                .collect();

            if four_dir {
                let mut all_sheets = Vec::new();
                for sprite in &processed {
                    let (sheet, _) = relight::four_dir_sheet(sprite);
                    all_sheets.push(sheet);
                }
                if all_sheets.len() == 1 {
                    all_sheets[0].save(&output)?;
                } else {
                    let sw = all_sheets[0].width();
                    let sh = all_sheets[0].height();
                    let mut combined = image::RgbaImage::new(sw, sh * all_sheets.len() as u32);
                    for (i, s) in all_sheets.iter().enumerate() {
                        image::imageops::overlay(&mut combined, s, 0, (i as u32 * sh) as i64);
                    }
                    combined.save(&output)?;
                }
                println!("saved: {output} ({count} × 4 directions)");
            } else {
                if count == 1 {
                    processed[0].save(&output)?;
                } else {
                    let sheet_img = sheet::pack_grid(&processed, 8);
                    sheet_img.save(&output)?;
                }
                println!("saved: {output}");
            }
        }
        Cmd::Relight { image, output, split } => {
            let img = image::open(&image)?.to_rgba8();
            let (sheet, sprites) = relight::four_dir_sheet(&img);
            sheet.save(&output)?;
            println!("saved: {output} ({}x{}, 4 views)", sheet.width(), sheet.height());

            if split {
                let stem = std::path::Path::new(&output).file_stem()
                    .and_then(|s| s.to_str()).unwrap_or("relit");
                let dir = std::path::Path::new(&output).parent().unwrap_or(std::path::Path::new("."));
                for (name, sprite) in &sprites {
                    let path = dir.join(format!("{stem}_{name}.png"));
                    sprite.save(&path)?;
                    println!("  {}", path.display());
                }
            }
        }
        Cmd::Skeletons { data, img_size } => {
            train::compute_skeletons(&data, img_size)?;
        }
        Cmd::PrepStages { data, palette_colors } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];

            let data_path = std::path::Path::new(&data);
            let struct_base = data_path.join("_structure");  // RGBA: SDF, nx, ny, outline
            let lum_base = data_path.join("_luminance");     // Grayscale: light/shadow
            let sil_base = data_path.join("_silhouettes");   // Binary mask
            std::fs::create_dir_all(&struct_base)?;
            std::fs::create_dir_all(&lum_base)?;
            std::fs::create_dir_all(&sil_base)?;

            let mut total = 0usize;
            for class_name in &class_names {
                let class_dir = data_path.join(class_name);
                if !class_dir.is_dir() { continue; }

                let struct_dir = struct_base.join(class_name);
                let lum_dir = lum_base.join(class_name);
                let sil_dir = sil_base.join(class_name);
                std::fs::create_dir_all(&struct_dir)?;
                std::fs::create_dir_all(&lum_dir)?;
                std::fs::create_dir_all(&sil_dir)?;

                let mut count = 0usize;
                for entry in std::fs::read_dir(&class_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("png") { continue; }

                    let img = image::open(&path)?.to_rgba8();
                    let w = img.width() as usize;
                    let h = img.height() as usize;
                    let fname = entry.file_name();

                    // Binary mask: opaque pixels
                    let mut mask = vec![false; w * h];
                    for (x, y, px) in img.enumerate_pixels() {
                        if px[3] > 10 {
                            mask[y as usize * w + x as usize] = true;
                        }
                    }

                    // SDF via distance transform (Chamfer approximation)
                    // Forward + backward pass with 3-4-3 weights
                    let mut dist = vec![f32::MAX; w * h];
                    // Set border pixels to 0, interior to MAX
                    for y in 0..h {
                        for x in 0..w {
                            let i = y * w + x;
                            if !mask[i] {
                                dist[i] = 0.0;
                            } else {
                                // Check if any neighbor is outside — edge pixel
                                let is_edge = (x == 0 || !mask[i - 1])
                                    || (x == w - 1 || !mask[i + 1])
                                    || (y == 0 || !mask[i - w])
                                    || (y == h - 1 || !mask[i + w]);
                                dist[i] = if is_edge { 0.5 } else { f32::MAX };
                            }
                        }
                    }
                    // Forward pass (top-left to bottom-right)
                    for y in 1..h {
                        for x in 1..w {
                            let i = y * w + x;
                            if !mask[i] { continue; }
                            let up = dist[(y - 1) * w + x] + 1.0;
                            let left = dist[y * w + (x - 1)] + 1.0;
                            let diag = dist[(y - 1) * w + (x - 1)] + 1.414;
                            dist[i] = dist[i].min(up).min(left).min(diag);
                        }
                    }
                    // Backward pass (bottom-right to top-left)
                    for y in (0..h - 1).rev() {
                        for x in (0..w - 1).rev() {
                            let i = y * w + x;
                            if !mask[i] { continue; }
                            let down = dist[(y + 1) * w + x] + 1.0;
                            let right = dist[y * w + (x + 1)] + 1.0;
                            let diag = dist[(y + 1) * w + (x + 1)] + 1.414;
                            dist[i] = dist[i].min(down).min(right).min(diag);
                        }
                    }

                    // Normalize SDF to [0, 1]
                    let max_dist = dist.iter().cloned().fold(0.0f32, f32::max).max(1.0);
                    for d in &mut dist {
                        *d /= max_dist;
                    }

                    // Normals from SDF gradient (Sobel-like)
                    let mut nx = vec![0.0f32; w * h];
                    let mut ny = vec![0.0f32; w * h];
                    for y in 1..h - 1 {
                        for x in 1..w - 1 {
                            let i = y * w + x;
                            if !mask[i] { continue; }
                            // Horizontal gradient
                            let gx = dist[i + 1] - dist[i - 1];
                            // Vertical gradient
                            let gy = dist[i + w] - dist[i - w];
                            let len = (gx * gx + gy * gy).sqrt().max(1e-6);
                            nx[i] = gx / len;
                            ny[i] = gy / len;
                        }
                    }

                    // Outline: pixels where SDF ≈ 0.5 (on the edge)
                    // More precisely: mask=true but at least one neighbor is mask=false
                    let mut outline = vec![0u8; w * h];
                    for y in 0..h {
                        for x in 0..w {
                            let i = y * w + x;
                            if !mask[i] { continue; }
                            let has_bg = (x > 0 && !mask[i - 1])
                                || (x < w - 1 && !mask[i + 1])
                                || (y > 0 && !mask[i - w])
                                || (y < h - 1 && !mask[i + w]);
                            if has_bg { outline[i] = 255; }
                        }
                    }

                    // Luminance from original sprite (0.299R + 0.587G + 0.114B)
                    let mut lum = vec![0u8; w * h];
                    for (x, y, px) in img.enumerate_pixels() {
                        let i = y as usize * w + x as usize;
                        if mask[i] {
                            let l = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
                            lum[i] = l.clamp(0.0, 255.0) as u8;
                        }
                    }

                    // Structure: RGBA = SDF, normal_x, normal_y, outline
                    let mut struct_img = image::RgbaImage::new(w as u32, h as u32);
                    // Luminance: grayscale RGB
                    let mut lum_img = image::RgbImage::new(w as u32, h as u32);
                    // Silhouette: binary
                    let mut sil_img = image::RgbImage::new(w as u32, h as u32);

                    for y in 0..h {
                        for x in 0..w {
                            let i = y * w + x;
                            if mask[i] {
                                let r = (dist[i] * 255.0) as u8;
                                let g = ((0.5 + 0.5 * nx[i]) * 255.0) as u8;
                                let b = ((0.5 + 0.5 * ny[i]) * 255.0) as u8;
                                struct_img.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, outline[i]]));
                                lum_img.put_pixel(x as u32, y as u32, image::Rgb([lum[i], lum[i], lum[i]]));
                                sil_img.put_pixel(x as u32, y as u32, image::Rgb([255, 255, 255]));
                            } else {
                                struct_img.put_pixel(x as u32, y as u32, image::Rgba([0, 128, 128, 0]));
                                lum_img.put_pixel(x as u32, y as u32, image::Rgb([0, 0, 0]));
                                sil_img.put_pixel(x as u32, y as u32, image::Rgb([0, 0, 0]));
                            }
                        }
                    }
                    struct_img.save(struct_dir.join(&fname))?;
                    lum_img.save(lum_dir.join(&fname))?;
                    sil_img.save(sil_dir.join(&fname))?;

                    count += 1;
                }
                if count > 0 {
                    println!("{class_name}: {count} sprites → 5-layer structure");
                }
                total += count;
            }
            println!("total: {total} structure maps");
            println!("  _structure/ RGBA: SDF(R), normal_x(G), normal_y(B), outline(A)");
            println!("  _luminance/ grayscale: light/shadow from original");
            println!("  _silhouettes/ binary: shape mask");
            println!("  → 5 conditioning channels for detail model (8ch input: 5 cond + 3 noisy)");
        }
        Cmd::Train {
            data,
            output,
            epochs,
            batch_size,
            lr,
            img_size,
            medium,
            anvil,
            no_ema,
            v_pred,
            lr_min,
            warmup,
            condition,
            fp16,
        } => {
            let config = train::TrainConfig {
                data_dir: data,
                output,
                epochs,
                batch_size,
                lr,
                img_size,
                medium,
                anvil,
                ema: !no_ema,
                v_prediction: v_pred,
                lr_min,
                warmup_epochs: warmup,
                condition_dir: condition,
                mixed_precision: fp16,
                ..Default::default()
            };
            train::train(&config)?;
        }
        Cmd::Generate {
            class,
            model,
            size,
            count,
            steps,
            palette: palette_name,
            output,
            medium,
            seed,
            four_dir,
        } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32; // default to misc

            println!("generating {count}x {size}x{size} class={class} (id={class_id})");

            let pal = palette::load_palette(&palette_name)?;
            let raw_images = if medium {
                train::sample_medium_seeded(&model, class_id, size, count, steps, seed)?
            } else {
                train::sample_seeded(&model, class_id, size, count, steps, seed)?
            };

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let snapped = grid::snap_to_grid(&img, size);
                    palette::quantize(&snapped, &pal)
                })
                .collect();

            if four_dir {
                // Relight each sprite into 4-direction sheets
                let mut all_sheets = Vec::new();
                for sprite in &processed {
                    let (sheet, _) = relight::four_dir_sheet(sprite);
                    all_sheets.push(sheet);
                }
                if all_sheets.len() == 1 {
                    all_sheets[0].save(&output)?;
                } else {
                    // Stack vertically: each row is one character's 4 views
                    let sw = all_sheets[0].width();
                    let sh = all_sheets[0].height();
                    let mut combined = image::RgbaImage::new(sw, sh * all_sheets.len() as u32);
                    for (i, s) in all_sheets.iter().enumerate() {
                        image::imageops::overlay(&mut combined, s, 0, (i as u32 * sh) as i64);
                    }
                    combined.save(&output)?;
                }
                println!("saved: {output} ({count} sprites × 4 directions)");
            } else {
                if count == 1 {
                    processed[0].save(&output)?;
                } else {
                    let sheet_img = sheet::pack_grid(&processed, 8);
                    sheet_img.save(&output)?;
                }
                println!("saved: {output}");
            }
        }
        Cmd::Swipe { image, verdict, class, store } => {
            let good = match verdict.to_lowercase().as_str() {
                "good" | "right" | "yes" | "y" | "1" => true,
                "bad" | "left" | "no" | "n" | "0" => false,
                _ => anyhow::bail!("verdict must be 'good' or 'bad', got '{verdict}'"),
            };

            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            // Load sprite pixels (channel-first f32)
            let img = image::open(&image)?.to_rgb8();
            let img = image::imageops::resize(&img, 16, 16, image::imageops::FilterType::Nearest);
            let n = 256usize; // 16*16
            let mut pixels = vec![0.0f32; 768];
            for y in 0..16u32 {
                for x in 0..16u32 {
                    let p = img.get_pixel(x, y);
                    let idx = (y * 16 + x) as usize;
                    pixels[idx] = p[0] as f32 / 255.0;
                    pixels[n + idx] = p[1] as f32 / 255.0;
                    pixels[2 * n + idx] = p[2] as f32 / 255.0;
                }
            }

            let mut swipe_store = swipe_store::SwipeStore::load_or_default(&store, 200);
            let len = swipe_store.record(pixels, good, class_id);
            swipe_store.save(&store)?;

            let verdict_str = if good { "good" } else { "bad" };
            println!("recorded swipe: {} ({}) → {} swipes in buffer", verdict_str, class, len);
            if swipe_store.judge_ready() {
                println!("judge ready — run `pixel-forge train-judge` to train");
            }
            if swipe_store.should_retrain() {
                println!("retrain triggered (every 5 swipes)");
            }
        }
        Cmd::TrainJudge { store, output } => {
            let device = pipeline::best_device();
            let swipe_store = swipe_store::SwipeStore::load_or_default(&store, 200);

            let (varmap, _model, result) = judge::train_judge(&swipe_store, &device)?;
            judge::save_judge(&varmap, &output)?;

            println!("{result}");
            let file_size = std::fs::metadata(&output)?.len();
            println!("saved: {} ({:.1} KB)", output, file_size as f64 / 1024.0);
        }
        Cmd::Judge { input, model } => {
            let device = pipeline::best_device();
            let (_varmap, judge_model) = judge::load_judge(&model, &device)?;

            let path = std::path::Path::new(&input);
            let files: Vec<std::path::PathBuf> = if path.is_dir() {
                walkdir::WalkDir::new(path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("png"))
                    .map(|e| e.into_path())
                    .collect()
            } else {
                vec![path.to_path_buf()]
            };

            println!("judging {} sprites...", files.len());
            for file in &files {
                let img = image::open(file)?.to_rgb8();
                let img = image::imageops::resize(&img, 16, 16, image::imageops::FilterType::Nearest);
                let n = 256usize;
                let mut pixels = vec![0.0f32; 768];
                for y in 0..16u32 {
                    for x in 0..16u32 {
                        let p = img.get_pixel(x, y);
                        let idx = (y * 16 + x) as usize;
                        pixels[idx] = p[0] as f32 / 255.0;
                        pixels[n + idx] = p[1] as f32 / 255.0;
                        pixels[2 * n + idx] = p[2] as f32 / 255.0;
                    }
                }
                let score = judge_model.score_one(&pixels, &device)?;
                let verdict = if score > 0.5 { "GOOD" } else { "BAD " };
                println!("  {} {:.3} {}", verdict, score, file.display());
            }
        }
        Cmd::Scene { mode, combiner_model, biome, temperature, count, output } => {
            match mode.as_str() {
                "bootstrap" => {
                    let biome_enum = match biome.to_lowercase().as_str() {
                        "forest" => scene::Biome::Forest,
                        "dungeon" => scene::Biome::Dungeon,
                        "village" => scene::Biome::Village,
                        "cave" => scene::Biome::Cave,
                        "plains" => scene::Biome::Plains,
                        _ => {
                            println!("unknown biome '{biome}', using forest");
                            scene::Biome::Forest
                        }
                    };

                    for i in 0..count {
                        let grid = scene::generate_seeded(biome_enum);
                        let img = grid.render_placeholder();
                        let out = if count == 1 {
                            output.clone()
                        } else {
                            format!("{}-{}.png", output.trim_end_matches(".png"), i + 1)
                        };
                        img.save(&out)?;
                        println!("scene {}/{}: {} sprites, saved {}", i + 1, count, grid.sprite_count(), out);
                    }
                }
                "model" => {
                    let device = pipeline::best_device();
                    let (_varmap, model) = combiner::load_combiner(&combiner_model, &device)?;

                    for i in 0..count {
                        let grid = combiner::generate_scene(&model, None, temperature, &device)?;
                        let img = grid.render_placeholder();
                        let out = if count == 1 {
                            output.clone()
                        } else {
                            format!("{}-{}.png", output.trim_end_matches(".png"), i + 1)
                        };
                        img.save(&out)?;
                        println!("scene {}/{}: {} sprites, saved {}", i + 1, count, grid.sprite_count(), out);
                    }
                }
                _ => anyhow::bail!("unknown scene mode '{mode}' — use 'bootstrap' or 'model'"),
            }
        }
        Cmd::TrainCombiner { num_scenes, epochs, output } => {
            let device = pipeline::best_device();

            println!("generating {} bootstrap scenes...", num_scenes);
            let scenes = scene::generate_bootstrap(num_scenes);

            let config = combiner::CombinerTrainConfig {
                epochs,
                batch_size: 16,
                lr: 1e-3,
                mask_frac_min: 0.2,
                mask_frac_max: 0.5,
            };

            let (varmap, _model, result) = combiner::train_combiner(&scenes, &config, &device)?;
            combiner::save_combiner(&varmap, &output)?;

            println!("{result}");
            let file_size = std::fs::metadata(&output)?.len();
            println!("saved: {} ({:.1} KB)", output, file_size as f64 / 1024.0);
        }
        Cmd::TrainLora { model, judge: judge_path, output, class } => {
            let device = pipeline::best_device();

            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            // Load base Generator
            let mut base_varmap = candle_nn::VarMap::new();
            let base_vb = candle_nn::VarBuilder::from_varmap(&base_varmap, candle_core::DType::F32, &device);
            let _base_model = tiny_unet::TinyUNet::new(base_vb)?;
            base_varmap.load(&model)?;
            println!("loaded base model: {}", model);

            // Load Judge
            let (_jvm, judge_model) = judge::load_judge(&judge_path, &device)?;
            println!("loaded judge: {}", judge_path);

            // Create or load LoRA
            let mut lora_varmap = candle_nn::VarMap::new();
            let lora_vb = candle_nn::VarBuilder::from_varmap(&lora_varmap, candle_core::DType::F32, &device);
            let lora_set = lora::LoraSet::new(lora_vb)?;

            if std::path::Path::new(&output).exists() {
                lora_varmap.load(&output)?;
                println!("loaded existing lora: {}", output);
            }

            let class_ids = vec![class_id; 8];
            let loss = lora::train_lora_step(
                &base_varmap, &lora_varmap, &lora_set, &judge_model, &device, 8, &class_ids,
            )?;
            println!("lora training: avg_loss={:.4}", loss);

            lora::save_lora(&lora_varmap, &output)?;
            let file_size = std::fs::metadata(&output)?.len();
            println!("saved: {} ({:.1} KB)", output, file_size as f64 / 1024.0);
        }
        Cmd::SwipeStats { store } => {
            let swipe_store = swipe_store::SwipeStore::load_or_default(&store, 200);
            println!("{}", swipe_store.stats());
            if swipe_store.judge_ready() {
                println!("judge: ready to train");
            } else {
                let need = 10 - swipe_store.len().min(10);
                println!("judge: need {} more swipes", need);
            }
        }
        Cmd::Pipeline { model, judge: judge_path, combiner: combiner_path, lora: _lora_path, palette: palette_name, output } => {
            let device = pipeline::best_device();
            let pal = palette::load_palette(&palette_name)?;

            // 1. GENERATE — produce candidates per class
            println!("step 1: generating sprite candidates...");
            let mut all_sprites: Vec<(u32, Vec<f32>, image::RgbaImage)> = Vec::new();
            let classes_to_gen: Vec<u32> = vec![0, 3, 4, 5]; // character, terrain, enemy, tree
            let count_per_class = 5u32;

            for &cid in &classes_to_gen {
                let raw_images = train::sample(&model, cid, 16, count_per_class, 40)?;
                for img in raw_images {
                    let snapped = grid::snap_to_grid(&img, 16);
                    let quantized = palette::quantize(&snapped, &pal);

                    // Convert to channel-first f32 for Judge
                    let n = 256usize;
                    let mut pixels = vec![0.0f32; 768];
                    for y in 0..16u32 {
                        for x in 0..16u32 {
                            let p = quantized.get_pixel(x, y);
                            let idx = (y * 16 + x) as usize;
                            pixels[idx] = p[0] as f32 / 255.0;
                            pixels[n + idx] = p[1] as f32 / 255.0;
                            pixels[2 * n + idx] = p[2] as f32 / 255.0;
                        }
                    }
                    all_sprites.push((cid, pixels, quantized));
                }
            }
            println!("  generated {} sprites across {} classes",
                all_sprites.len(), classes_to_gen.len());

            // 2. JUDGE — filter if Judge model exists
            let accepted: Vec<(u32, Vec<f32>, image::RgbaImage)>;
            if std::path::Path::new(&judge_path).exists() {
                println!("step 2: judging sprites...");
                let (_jvm, judge_model) = judge::load_judge(&judge_path, &device)?;
                accepted = all_sprites.into_iter().filter(|(_, pixels, _)| {
                    judge_model.score_one(pixels, &device).unwrap_or(0.0) > 0.5
                }).collect();
                println!("  accepted {}/{} sprites", accepted.len(),
                    classes_to_gen.len() as u32 * count_per_class);
            } else {
                println!("step 2: no judge model, accepting all sprites");
                accepted = all_sprites;
            }

            if accepted.is_empty() {
                anyhow::bail!("no sprites passed the Judge — need more swipe data or lower threshold");
            }

            // 3. COMBINE — arrange into scene
            println!("step 3: arranging scene...");
            let grid = if std::path::Path::new(&combiner_path).exists() {
                let (_cvm, combiner_model) = combiner::load_combiner(&combiner_path, &device)?;
                combiner::generate_scene(&combiner_model, None, 0.8, &device)?
            } else {
                println!("  no combiner model, using bootstrap layout");
                scene::generate_seeded(scene::Biome::Forest)
            };

            // 4. RENDER — composite sprites into grid
            println!("step 4: rendering scene...");
            let mut rng = rand::thread_rng();
            use rand::seq::SliceRandom;
            let mut sprites_by_class: std::collections::HashMap<u32, Vec<&Vec<f32>>> =
                std::collections::HashMap::new();
            for (cid, pixels, _) in &accepted {
                sprites_by_class.entry(*cid).or_default().push(pixels);
            }

            let scene_img = grid.render(|class_id| {
                sprites_by_class.get(&class_id)
                    .and_then(|sprites| sprites.choose(&mut rng))
                    .map(|px| px.to_vec())
            });

            scene_img.save(&output)?;
            println!("saved scene: {} ({}×{} px, {} sprites placed)",
                output, scene::SCENE_PX, scene::SCENE_PX, grid.sprite_count());
        }
        Cmd::Auto { class, size, count, steps, palette: palette_name, output } => {
            let pal = palette::load_palette(&palette_name)?;
            println!("auto-detect: picking best model for this device...");

            let raw_images = device_cap::auto_sample(
                {
                    let class_names = [
                        "character", "weapon", "potion", "terrain", "enemy",
                        "tree", "building", "animal", "effect", "food",
                        "armor", "tool", "vehicle", "ui", "misc",
                    ];
                    class_names.iter()
                        .position(|&n| n == class.to_lowercase())
                        .unwrap_or(14) as u32
                },
                size,
                count,
                steps,
            )?;

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let snapped = grid::snap_to_grid(&img, size);
                    palette::quantize(&snapped, &pal)
                })
                .collect();

            if count == 1 {
                processed[0].save(&output)?;
            } else {
                let sheet_img = sheet::pack_grid(&processed, 8);
                sheet_img.save(&output)?;
            }
            println!("saved: {output}");
        }
        Cmd::Probe { json } => {
            let profile = device_cap::reprobe()?;
            if json {
                // Machine-readable output for cluster probing
                println!("{}", serde_json::to_string(&profile)?);
            } else {
                println!("backend: {}", profile.backend);
                println!("ram: {} MB", profile.ram_mb);
                println!("cinder: {:.1} ms/step", profile.cinder_ms);
                if let Some(q) = profile.quench_ms {
                    println!("quench: {:.1} ms/step", q);
                }
                println!("selected tier: {}", profile.tier);
                println!("model file: {}", profile.tier.model_file());
            }
        }
        Cmd::ClusterProbe => {
            let state = cluster::probe_cluster()?;
            println!("\ncluster summary:");
            let nodes = state.active_nodes();
            for n in &nodes {
                let tier = n.profile.as_ref().map(|p| p.tier.to_string()).unwrap_or("?".into());
                let backend = n.profile.as_ref().map(|p| p.backend.to_string()).unwrap_or("?".into());
                println!("  {:6} | {:5} | {:6} | {:.1} spr/s",
                    n.name, backend, tier, n.throughput);
            }
            println!("  total: {:.1} sprites/sec", state.total_throughput);
        }
        Cmd::ClusterDeploy => {
            cluster::deploy_all()?;
        }
        Cmd::ClusterSync => {
            cluster::sync_models_all()?;
        }
        Cmd::ClusterGenerate { class, count, steps, palette: palette_name, output } => {
            let state = cluster::probe_cluster()?;
            let pngs = cluster::cluster_generate(&class, count, steps, &palette_name, &state)?;

            if pngs.is_empty() {
                anyhow::bail!("no sprites generated");
            }

            // Decode PNGs into images for grid packing
            let images: Vec<image::RgbaImage> = pngs.iter()
                .filter_map(|bytes| {
                    image::load_from_memory(bytes).ok().map(|img| img.to_rgba8())
                })
                .collect();

            if images.len() == 1 {
                images[0].save(&output)?;
            } else {
                let sheet_img = sheet::pack_grid(&images, 8);
                sheet_img.save(&output)?;
            }
            println!("saved: {} ({} sprites)", output, images.len());
        }
        Cmd::Gpu { args } => {
            // Delegate to kova c2 gpu
            let mut cmd = std::process::Command::new("kova");
            cmd.args(["c2", "gpu"]);
            cmd.args(&args);
            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("kova c2 gpu failed");
            }
        }
        Cmd::Forge { class, count, steps, threshold, max_attempts, disc, palette: palette_name, output, lat, lon, cascade: use_cascade } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            let pal = palette::load_palette(&palette_name)?;
            let signing_key = poa::load_or_create_keypair()?;

            println!("forge: class={class} threshold={threshold} cascade={use_cascade}");

            // Generate with quality gate
            let sprites = if use_cascade {
                let config = moe::CascadeConfig::default();
                moe::cascade_with_gate(
                    "pixel-forge-quench.safetensors",
                    "pixel-forge-cinder.safetensors",
                    Some("experts.safetensors"),
                    &disc, class_id, 32, count, &config, threshold, max_attempts,
                )?
            } else {
                let device = pipeline::best_device();
                let tier = device_cap::best_available();
                discriminator::generate_with_gate(
                    tier, class_id, count, steps, threshold, max_attempts, &disc, &device,
                )?
            };

            // Sign each sprite, save images, produce Ghost Fabric packets
            let device = pipeline::best_device();
            let mut packet_count = 0usize;
            let sprite_count = sprites.len();
            let mut processed = Vec::new();

            for sprite in sprites {
                let (_dvm, disc_model) = discriminator::load(&disc, &device)?;
                let (_, score) = discriminator::quality_gate(&disc_model, &sprite, 0.0, &device)?;
                let packet = poa::sign_artifact(&sprite, class_id, score, lat, lon, &signing_key);
                assert!(packet.verify(), "PoA self-verification failed");
                println!("  {}", packet.summary());
                let wire = packet.to_bytes();
                println!("  wire: {} bytes (LoRa ready: {})", wire.len(), wire.len() <= 255);
                packet_count += 1;

                let snapped = grid::snap_to_grid(&sprite, 16);
                processed.push(palette::quantize(&snapped, &pal));
            }

            if count == 1 {
                processed[0].save(&output)?;
            } else {
                let sheet_img = sheet::pack_grid(&processed, 8);
                sheet_img.save(&output)?;
            }
            println!("saved: {output} ({} sprites, {} PoA packets)", sprite_count, packet_count);
        }
        Cmd::Sign { image, class, score, disc, lat, lon } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            let img = image::open(&image)?.to_rgba8();
            let signing_key = poa::load_or_create_keypair()?;

            let quality = if let Some(s) = score {
                s
            } else {
                let device = pipeline::best_device();
                let (_dvm, disc_model) = discriminator::load(&disc, &device)?;
                let (_, s) = discriminator::quality_gate(&disc_model, &img, 0.0, &device)?;
                println!("discriminator score: {:.3}", s);
                s
            };

            let packet = poa::sign_artifact(&img, class_id, quality, lat, lon, &signing_key);
            let wire = packet.to_bytes();
            println!("{}", packet.summary());
            println!("verified: {}", packet.verify());
            println!("wire: {} bytes (LoRa ready: {})", wire.len(), wire.len() <= 255);

            // Write packet to .poa file alongside the image
            let poa_path = format!("{}.poa", image.trim_end_matches(".png"));
            std::fs::write(&poa_path, &wire)?;
            println!("saved: {poa_path}");
        }
        Cmd::NodeKey => {
            let sk = poa::load_or_create_keypair()?;
            let vk = sk.verifying_key();
            let pk_hex: String = vk.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
            println!("node public key: {pk_hex}");
            println!("key file: ~/.pixel-forge/node.key");
        }
        Cmd::Plugin { r#loop } => {
            if r#loop {
                plugin::run_loop()?;
            } else {
                plugin::run()?;
            }
        }
        Cmd::TrainExperts { quench, data, output, epochs, batch_size } => {
            expert_train::train_experts(&quench, &data, &output, epochs, batch_size)?;
        }
        Cmd::Quantize { input, output } => {
            let out = if let Some(o) = output {
                let size = quantize::quantize_f32_to_f16(&input, &o)?;
                println!("wrote {} ({:.1} MB)", o, size as f64 / 1_048_576.0);
                o
            } else {
                quantize::quantize_in_place(&input)?
            };
            println!("done: {out}");
        }
        Cmd::Cascade { class, quench, cinder, experts, count, quench_steps, cinder_steps, palette: palette_name, output } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            let pal = palette::load_palette(&palette_name)?;
            let experts_path = if std::path::Path::new(&experts).exists() {
                Some(experts.as_str())
            } else {
                None
            };

            let config = moe::CascadeConfig {
                quench_steps,
                cinder_steps,
            };

            let raw_images = moe::cascade_sample(&quench, &cinder, experts_path, class_id, 32, count, &config)?;

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let snapped = grid::snap_to_grid(&img, 32);
                    palette::quantize(&snapped, &pal)
                })
                .collect();

            if count == 1 {
                processed[0].save(&output)?;
            } else {
                let sheet_img = sheet::pack_grid(&processed, 8);
                sheet_img.save(&output)?;
            }
            println!("saved: {output}");
        }
        Cmd::Upscale { input, output } => {
            let classes = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let mut total = 0u32;
            for class in &classes {
                let src_dir = std::path::Path::new(&input).join(class);
                let dst_dir = std::path::Path::new(&output).join(class);
                if !src_dir.is_dir() {
                    continue;
                }
                std::fs::create_dir_all(&dst_dir)?;
                let mut count = 0u32;
                for entry in std::fs::read_dir(&src_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("png") {
                        continue;
                    }
                    let fname = path.file_name().unwrap();
                    let dst_path = dst_dir.join(fname);
                    if dst_path.exists() {
                        continue;
                    }
                    let img = image::open(&path)?.to_rgba8();
                    let resized = image::imageops::resize(
                        &img, 32, 32,
                        image::imageops::FilterType::Nearest,
                    );
                    resized.save(&dst_path)?;
                    count += 1;
                }
                if count > 0 {
                    println!("{class}: {count} upscaled");
                }
                total += count;
            }
            println!("total: {total} tiles upscaled to 32x32");
        }
        Cmd::Anvil { class, anvil, count, steps, palette: palette_name, output } => {
            let class_names = [
                "character", "weapon", "potion", "terrain", "enemy",
                "tree", "building", "animal", "effect", "food",
                "armor", "tool", "vehicle", "ui", "misc",
            ];
            let class_id = class_names.iter()
                .position(|&n| n == class.to_lowercase())
                .unwrap_or(14) as u32;

            let pal = palette::load_palette(&palette_name)?;
            let raw_images = moe::anvil_sample(&anvil, class_id, 32, count, steps)?;

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let snapped = grid::snap_to_grid(&img, 32);
                    palette::quantize(&snapped, &pal)
                })
                .collect();

            if count == 1 {
                processed[0].save(&output)?;
            } else {
                let sheet_img = sheet::pack_grid(&processed, 8);
                sheet_img.save(&output)?;
            }
            println!("saved: {output}");
        }
    }

    Ok(())
}
