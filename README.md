<p align="center">
  <img src="assets/pixel-forge-logo.png" alt="Pixel Forge" width="256">
</p>

> This repo is part of [CochranBlock](https://cochranblock.org) — Unlicense Rust repositories. All source code is public domain.
>
> Ships with **[Proof of Artifacts](PROOF_OF_ARTIFACTS.md)** and a **[Timeline of Invention](TIMELINE_OF_INVENTION.md)** documenting what was built and when.

---

# Pixel Forge v0.6.0

**Pixel art sprite generator. Three diffusion models. Pure Rust. No cloud.**

Train and run Gaussian diffusion models that generate 32x32 pixel art sprites. Runs locally on CPU, Metal (Apple Silicon), or CUDA (NVIDIA). No Python. No cloud APIs.

**Status: In active development.** Output quality is improving but not yet game-ready. Training and inference work; sprite fidelity is the current focus.

## What Exists Today

- **Three model architectures** — Cinder (1.09M params), Quench (5.83M params), Anvil (16.9M params)
- **Training loop** — Gaussian noise, clean prediction, cosine schedule, min-SNR weighting, CFG dropout, per-epoch checkpoints, `--resume` for fine-tuning
- **Augmentation** — palette swap, horizontal flip, 90/180/270 rotation
- **108 class directories** mapped via hybrid conditioning (10 super-categories + 12 binary tags)
- **7 color palettes** — stardew, starbound, snes, nes, gameboy, pico8, endesga32
- **f16 quantization** — `quantize` command halves model file size
- **GUI** — egui desktop app with class picker, palette selector, generation
- **CLI** — full command set for training, generation, and data pipeline
- **Distributed generation** — cluster module distributes work across SSH nodes
- **11 governance documents** baked into binary (`pixel-forge govdocs`)
- **Android build scaffold** — AAB builds, not yet published to Play Store

### What's In Progress

- **Output quality** — models generate recognizable but rough sprites; not yet usable as game art
- **Anvil v7 training** — fine-tuning from v6 with rotation augmentation on worker node
- **MoE cascade** — Cinder→Quench pipeline code exists, expert heads designed but not validated at quality
- **Community sprite sharing** — planned, not implemented

### What's Not Done

- **iOS app** — scaffold only, not buildable
- **Web/PWA** — HTML scaffold only, WASM target not functional
- **GitHub Releases** — binaries built locally but not published as GitHub releases yet

## Quick Start

```bash
# Build (Metal GPU on macOS, CPU fallback)
cargo build --release

# Build with CUDA (NVIDIA GPU on Linux)
cargo build --release --features cuda --no-default-features

# Launch GUI
cargo run --release

# Generate (requires a trained model)
cargo run --release -- anvil character --count 4 --steps 40 --palette stardew

# List palettes
cargo run --release -- palettes
```

### Training

Dataset: ~20K balanced tiles in `data_v3_32/`. See `data/SOURCES.md` for sources and licensing.

```bash
# Train Anvil (~12 min/epoch on CPU, ~42 hrs for 200 epochs)
cargo run --release -- train --data data_v3_32 --anvil --epochs 200 \
  --lr 2e-4 --warmup 10 --batch-size 16 --no-ema --checkpoint-every 1

# Fine-tune from existing checkpoint
cargo run --release -- train --data data_v3_32 --anvil --epochs 100 \
  --lr 5e-5 --batch-size 8 --resume pixel-forge-anvil-v6.safetensors \
  -o pixel-forge-anvil-v7.safetensors

# Train Cinder (fast, ~1.5 min/epoch)
cargo run --release -- train --data data_v3_32 --epochs 500 \
  --lr 2e-4 --batch-size 128 --no-ema
```

### Known Issues (Bug History)

The diffusion models produced blobs for weeks. Root causes found and fixed:
1. **Uniform [0,1] noise** — signal and noise occupied the same range. Fixed: Gaussian N(0,1) noise.
2. **CFG scale too high** — CFG 3.0 inverted outputs. Dialed back, then re-enabled at 3.0 after fixing the unconditional path.
3. **Noise distribution mismatch** — sampling started from uniform noise while training used Gaussian. Fixed in v0.6.0.

## Commands

| Command | Status | What It Does |
|---------|--------|-------------|
| `train` | working | Train Cinder/Quench/Anvil from dataset |
| `generate <class>` | working | Generate via Cinder with palette quantization |
| `anvil <class>` | working | Generate via Anvil |
| `cascade <class>` | working | MoE cascade: Cinder → Quench |
| `auto <class>` | working | Auto-detect GPU, pick best model |
| `quantize <model>` | working | Convert f32 → f16 |
| `curate` | working | Slice sprite sheets into training tiles |
| `ingest-gemini` | working | Slice Gemini sprite sheets into tiles |
| `relight <image>` | working | 4-directional sprite sheet via SDF + normals |
| `probe` | working | Device capability detection |
| `palettes` | working | List built-in palettes |
| `govdocs` | working | Show embedded compliance documents |
| `plugin` | working | JSON protocol for kova integration |
| `cluster-probe` | working | Probe forge cluster nodes |
| `cluster-generate` | working | Distribute generation across cluster |
| `forge <class>` | needs trained discriminator | Generate → quality gate → PoA sign |
| `train-experts` | needs validation | Train expert heads on frozen Quench |
| `train-judge` | needs swipe data | Train quality classifier |
| `train-lora` | needs judge model | Fine-tune from Judge feedback |
| `scene <mode>` | needs trained combiner | Generate 8x8 biome grids |
| `pipeline` | needs all sub-models | Full pipeline end-to-end |
| `stage-cascade` | experimental | Structure-aware cascade |

## Model Tiers

| Tier | Name | Params | Size (f32/f16) | Channels | Notes |
|------|------|--------|----------------|----------|-------|
| Tiny | **Cinder** | 1.09M | 4.2 / 2.1 MB | [32, 64, 64] | Fast, mobile-suitable |
| Medium | **Quench** | 5.83M | 22 / 11 MB | [64, 128, 128] + self-attention | Balanced |
| XL | **Anvil** | 16.9M | 64 / 32 MB | [96, 192, 192], 4 ResBlocks/level | Highest capacity |

## Training Data

**19,876 balanced tiles** (capped 2K/class, 68 active class directories):

| Source | Sprites | License | Notes |
|--------|---------|---------|-------|
| Dungeon Crawl Stone Soup | ~6,000 | CC0 | DCSS art team |
| DawnLike v1.81 | ~5,000 | CC-BY 4.0 | DragonDePlatino + DawnBringer |
| Kenney (3 packs) | ~3,878 | CC0 | Roguelike, Platformer, 1-Bit |
| Hyptosis Tiles | ~1,000 | CC-BY 3.0 | Hyptosis |
| David E. Gervais Tiles | ~1,280 | CC-BY 3.0 | David E. Gervais |
| **Gemini-generated** | **~14,000** | **AI-generated** | Fills class gaps via text prompts |

~70% of the balanced training set is AI-generated (Gemini). These sprites were generated via text prompts, sliced from grids, background-removed, and quality-checked. They fill classes that had <50 artist-made samples. See [data/SOURCES.md](data/SOURCES.md).

## Built-In Palettes

| Palette | Colors | Style |
|---------|--------|-------|
| stardew | 48 | Warm earth tones |
| starbound | 64 | Vibrant sci-fi |
| endesga32 | 32 | Popular indie pixel art |
| pico8 | 16 | PICO-8 fantasy console |
| snes | 256 | Super Nintendo |
| nes | 54 | NES |
| gameboy | 4 | Original Game Boy |

## Platforms

| Platform | GPU | Status |
|----------|-----|--------|
| macOS ARM (M1/M2/M3) | Metal | working |
| macOS Intel | CPU | working |
| Linux x86_64 | CUDA / CPU | working |
| Android ARM64 | CPU | builds, not published |
| iOS ARM64 | Metal | scaffold only |
| Web (PWA) | CPU | scaffold only |

## Tech Stack

| Layer | Tool |
|-------|------|
| ML | Candle 0.8 (Hugging Face) — pure Rust |
| GPU | Metal, CUDA, CPU fallback |
| Data | bincode + zstd (dataset), safetensors (models) |
| GUI | egui / eframe |
| CLI | clap |

~11,370 lines of Rust. Zero Python. Zero JavaScript.

## Governance Documents

11 compliance-oriented documents embedded in the binary (self-assessed, not independently audited):
```bash
pixel-forge govdocs              # list all docs
pixel-forge govdocs sbom         # Software Bill of Materials
pixel-forge govdocs security     # security posture
```

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits.

## License

Unlicense (public domain). See [LICENSE](LICENSE).

---

Built by [The Cochran Block](https://cochranblock.org). Powered by [KOVA](https://github.com/cochranblock/kova).
