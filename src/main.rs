// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! pixel-forge — Pixel art game asset generator.
//! Stardew Valley / Starbound quality. Local-first, Rust-native.

mod palette;
mod grid;
mod sheet;
mod pipeline;
mod tiny_unet;
mod train;
mod curate;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pixel-forge", about = "Pixel art game asset generator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
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
        #[arg(short, long, default_value_t = 16)]
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
        #[arg(short, long, default_value = "pixel-forge-tiny.safetensors")]
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
        #[arg(long, default_value_t = 16)]
        img_size: u32,
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
        #[arg(long, default_value_t = 16)]
        size: u32,
    },
    /// Generate pixel art using a trained tiny model (no SD required).
    Generate {
        /// Class to generate: character, weapon, potion, terrain, enemy, etc.
        class: String,
        /// Path to trained model (.safetensors).
        #[arg(short, long, default_value = "pixel-forge-tiny.safetensors")]
        model: String,
        /// Output size in pixels (must match training size).
        #[arg(short, long, default_value_t = 16)]
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
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
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
        } => {
            let config = train::TrainConfig {
                data_dir: data,
                output,
                epochs,
                batch_size,
                lr,
                img_size,
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
            let raw_images = train::sample(&model, class_id, size, count, steps)?;

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
    }

    Ok(())
}
