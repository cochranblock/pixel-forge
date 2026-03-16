// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! pixel-forge — Pixel art game asset generator.
//! Stardew Valley / Starbound quality. Local-first, Rust-native.

mod palette;
mod grid;
mod sheet;
mod pipeline;

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
    }

    Ok(())
}
