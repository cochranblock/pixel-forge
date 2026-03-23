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
    },
    /// Record a swipe (good/bad) on a generated sprite for Judge training.
    Swipe {
        /// Path to the sprite image (16×16 PNG).
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
        /// Use MoE cascade (Cinder → Quench + Experts) instead of auto-detect.
        #[arg(long)]
        cascade: bool,
    },
    /// Sign an existing sprite with PoA — produce a Ghost Fabric packet.
    Sign {
        /// Path to 16x16 PNG sprite.
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
    /// MoE cascade: Cinder drafts → Quench + Experts refines.
    Cascade {
        /// Class to generate.
        class: String,
        /// Cinder model path.
        #[arg(long, default_value = "pixel-forge-cinder.safetensors")]
        cinder: String,
        /// Quench model path.
        #[arg(long, default_value = "pixel-forge-quench.safetensors")]
        quench: String,
        /// Expert weights path (optional).
        #[arg(long, default_value = "experts.safetensors")]
        experts: String,
        /// Number of images.
        #[arg(short, long, default_value_t = 1)]
        count: u32,
        /// Cinder draft steps.
        #[arg(long, default_value_t = 10)]
        cinder_steps: usize,
        /// Quench refine steps.
        #[arg(long, default_value_t = 30)]
        quench_steps: usize,
        /// Palette.
        #[arg(short, long, default_value = "stardew")]
        palette: String,
        /// Output file.
        #[arg(short, long, default_value = "cascade.png")]
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

            if count == 1 {
                processed[0].save(&output)?;
            } else {
                let sheet_img = sheet::pack_grid(&processed, 8);
                sheet_img.save(&output)?;
            }
            println!("saved: {output}");
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
                    "pixel-forge-cinder.safetensors",
                    "pixel-forge-quench.safetensors",
                    Some("experts.safetensors"),
                    &disc, class_id, 16, count, &config, threshold, max_attempts,
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
        Cmd::Cascade { class, cinder, quench, experts, count, cinder_steps, quench_steps, palette: palette_name, output } => {
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
                cinder_steps,
                quench_steps,
            };

            let raw_images = moe::cascade_sample(&cinder, &quench, experts_path, class_id, 16, count, &config)?;

            let processed: Vec<image::RgbaImage> = raw_images
                .into_iter()
                .map(|img| {
                    let snapped = grid::snap_to_grid(&img, 16);
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
