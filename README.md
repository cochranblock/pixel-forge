# Pixel Forge

**Free pixel art generator. Train your own model. Embed it in your app. Pure Rust.**

Generate game-ready pixel art sprites from a tiny diffusion model that fits in a mobile binary. No cloud. No Python. No Solidity. One language.

## What It Does

- **Generate** pixel art sprites from text classes (character, weapon, terrain, enemy, etc.)
- **Train** your own model on curated artist-made pixel art datasets
- **Embed** the trained model (4.2 MB) directly in your Rust binary via `include_bytes!`
- **Palette quantize** output to authentic retro palettes (Stardew, NES, Game Boy, etc.)

## Architecture

```
PNG Dataset → bincode+zstd blob (10 MB) → RAM → TinyUNet training → safetensors (4.2 MB)
                                                                          ↓
                                                              pixel-forge generate
                                                                          ↓
                                                              16×16 pixel art sprites
                                                                          ↓
                                                              palette quantize + grid snap
                                                                          ↓
                                                              game-ready PNG
```

**TinyUNet** — 1.09M parameters. 3-level encoder/decoder (32→64→64 channels). Sinusoidal timestep embedding. 15 class labels. Skip connections. Operates in direct pixel space — no VAE, no latent encoding.

**Data pipeline** — decode 52k PNGs once, pack to bincode+zstd (15× compression), load into RAM in one read. Zero disk I/O during training. Palette swap + horizontal flip augmentation.

## Quick Start

```bash
# 1. Pull training data (7 artist-made CC0/CC-BY sources, ~17k quality sprites)
./scripts/pull-datasets.sh

# 2. Curate: slice sprite sheets → 16×16 tiles → class directories
cargo run --release -- curate --raw data/raw --output data --size 16

# 3. Train (Metal GPU on Mac, CPU fallback)
cargo run --release -- train --data data --epochs 100 --img-size 16

# 4. Generate
cargo run --release -- generate character --palette stardew -o hero.png
cargo run --release -- generate enemy --palette nes --count 16 -o enemies.png
cargo run --release -- generate weapon --palette endesga32 -o swords.png
```

## Commands

| Command | What It Does |
|---------|-------------|
| `pixel-forge sprite "prompt"` | Generate via Stable Diffusion (4GB model, high quality) |
| `pixel-forge train` | Train tiny model from PNG dataset |
| `pixel-forge generate <class>` | Generate via trained tiny model (4.2 MB, fast) |
| `pixel-forge curate` | Slice sprite sheets into training tiles |
| `pixel-forge setup` | Download SD model for `sprite` command |
| `pixel-forge palettes` | List built-in palettes |

## Training Data

52,139 curated tiles from 7 quality-gated sources. All hand-pixeled by known artists.
See [data/SOURCES.md](data/SOURCES.md) for full attribution.

| Source | Sprites | License | Artist |
|--------|---------|---------|--------|
| Dungeon Crawl Stone Soup | 6,000+ | CC0 | DCSS art team |
| DawnLike v1.81 | 5,000+ | CC-BY 4.0 | DragonDePlatino + DawnBringer |
| Kenney Roguelike/RPG | 1,700 | CC0 | Kenney |
| Kenney Pixel Platformer | 1,100 | CC0 | Kenney |
| Kenney 1-Bit Pack | 1,078 | CC0 | Kenney |
| Hyptosis Tiles | 1,000+ | CC-BY 3.0 | Hyptosis |
| David E. Gervais Tiles | 1,280 | CC-BY 3.0 | David E. Gervais |

No AI-generated images. No copyrighted game rips. No scraped content.

## Built-In Palettes

| Palette | Colors | Vibe |
|---------|--------|------|
| stardew | 48 | Warm earth tones (inspired by ConcernedApe's Stardew Valley) |
| starbound | 64 | Vibrant sci-fi |
| endesga32 | 32 | Popular indie pixel art |
| pico8 | 16 | PICO-8 fantasy console |
| snes | 256 | Super Nintendo |
| nes | 54 | Nintendo Entertainment System |
| gameboy | 4 | Original Game Boy |

## Model Specs

| Spec | Value |
|------|-------|
| Architecture | TinyUNet (3-level, 32/64/64 channels) |
| Parameters | 1,092,451 |
| File size | ~4.2 MB (safetensors) |
| Input | 16×16 RGB + timestep + class label |
| Output | 16×16 RGB (predicted clean image) |
| Classes | 15 (character, weapon, potion, terrain, enemy, tree, building, animal, effect, food, armor, tool, vehicle, ui, misc) |
| Training | DDPM-style noise corruption, MSE loss, AdamW |
| Augmentation | Palette swap (50%) + horizontal flip (50%) |
| Inference | 40-step iterative denoising |

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Framework | Candle (Hugging Face) |
| GPU | Metal (Apple Silicon), CUDA (NVIDIA), CPU fallback |
| Serialization | bincode + zstd (dataset), safetensors (model) |
| Image Processing | image crate |
| CLI | clap |

Zero Python. Zero JavaScript. One language.

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits including:
- PixelGen16x16 by Anouar Khaldi (architecture inspiration)
- pixartdiffusion by Zak Buzzard (training patterns)
- ConcernedApe / Eric Barone (Stardew Valley palette inspiration)
- Candle by Hugging Face (ML framework)

## License

Unlicense (public domain). See [LICENSE](LICENSE).

---

Built by [The Cochran Block](https://cochranblock.org).
