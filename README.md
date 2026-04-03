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

- **Three model architectures** — [Cinder](src/tiny_unet.rs#L16) (1.09M params, `[32,64,64]`), [Quench](src/medium_unet.rs#L17) (5.83M params, `[64,128,128]`), [Anvil](src/anvil_unet.rs#L17) (16.9M params, `[96,192,192]`). Param counts verified at [device_cap.rs:492](src/device_cap.rs#L492).
- **Training loop** ([src/train.rs](src/train.rs#L501)) — [Gaussian noise](src/train.rs#L432), clean prediction, [cosine schedule](src/train.rs#L91), [min-SNR weighting](src/train.rs#L100) (gamma=5), [CFG dropout](src/train.rs#L578) (10%), per-epoch checkpoints, [`--resume`](src/train.rs#L59) for fine-tuning
- **Augmentation** — [palette swap](src/train.rs#L348), [horizontal flip](src/train.rs#L399), [90/180/270 rotation](src/train.rs#L413)
- **108 class directories** mapped via [hybrid conditioning](src/class_cond.rs#L12) ([10 super-categories](src/class_cond.rs#L12) + [12 binary tags](src/class_cond.rs#L16), [lookup table](src/class_cond.rs#L84))
- **7 color palettes** — [load_palette](src/palette.rs#L17), [quantize](src/palette.rs#L55)
- **f16 quantization** — [quantize_f32_to_f16](src/quantize.rs#L40)
- **GUI** — [egui app](src/app.rs) with class picker, palette selector, generation
- **CLI** — [clap command definitions](src/main.rs#L94)
- **Distributed generation** — [cluster module](src/cluster.rs#L20) distributes work across [4 SSH nodes](src/cluster.rs#L20)
- **NanoSign model integrity** — [BLAKE3 signing](src/nanosign.rs) on all model saves, verification on all loads. Tampered files rejected. [Spec](https://github.com/cochranblock/kova/blob/main/docs/NANOSIGN.md).
- **11 governance documents** [baked into binary](src/main.rs#L10) via `include_str!` ([govdocs/](govdocs/))
- **Android build scaffold** — [android/](android/) builds AAB, not yet published to Play Store

### What's In Progress

- **Output quality** — models generate recognizable but rough sprites; not yet usable as game art
- **Anvil v7 training** — fine-tuning from v6 with rotation augmentation on worker node
- **MoE cascade** — Cinder→Quench pipeline code exists, expert heads designed but not validated at quality
- **[any-gpu](https://github.com/cochranblock/any-gpu) backend** — wgpu/Vulkan tensor engine with 31 ops (conv2d, group_norm, attention, etc.). Planned as future training backend for AMD GPUs (bt's 5700 XT). Needs autograd + optimizer.
- **Tiered pipeline** — planned: palette specialist (50K) → silhouette generator (200K) → detail painter (Anvil-class), each specialist trained separately

### What's Not Done

- **iOS app** — scaffold only, not buildable
- **Web/PWA** — HTML scaffold only, WASM target not functional
- **Community sprite sharing** — planned, not implemented
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
1. **Uniform [0,1] noise** — signal and noise occupied the same range. Fixed: [Gaussian N(0,1) noise](src/train.rs#L432) in corruption. Commit `541720cd`.
2. **CFG scale** — inverted outputs at high scale. Now set to [3.0](src/train.rs#L759) with proper [unconditional path](src/train.rs#L814). Commit `68f2183a`.
3. **Noise distribution mismatch** — sampling started from uniform noise while training used Gaussian. Fixed: all sampling paths now use [seeded_noise](src/train.rs#L764) (Gaussian). Commit `68f2183a`.

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

| Tier | Name | Params | Size (f32/f16) | Channels | Source |
|------|------|--------|----------------|----------|--------|
| Tiny | **Cinder** | 1.09M | 4.2 / 2.1 MB | [32, 64, 64] | [src/tiny_unet.rs:16](src/tiny_unet.rs#L16) |
| Medium | **Quench** | 5.83M | 22 / 11 MB | [64, 128, 128] + self-attention | [src/medium_unet.rs:17](src/medium_unet.rs#L17) |
| XL | **Anvil** | 16.9M | 64 / 32 MB | [96, 192, 192], 4 ResBlocks/level | [src/anvil_unet.rs:17](src/anvil_unet.rs#L17) |

## Training Data

**19,876 balanced tiles** (capped 2K/class, 68 active class directories). Dataset loader: [train::preprocess](src/train.rs#L221). Tile count printed at [train.rs:320](src/train.rs#L320).

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

Device auto-detection: [pipeline::best_device](src/pipeline.rs#L25). Feature flags: [Cargo.toml:19](Cargo.toml#L19).

| Platform | GPU | Status |
|----------|-----|--------|
| macOS ARM (M1/M2/M3) | Metal | working |
| macOS Intel | CPU | working |
| Linux x86_64 | CUDA / CPU | working |
| Android ARM64 | CPU | builds ([android/](android/)), not published |
| iOS ARM64 | Metal | scaffold only ([ios/](ios/)) |
| Web (PWA) | CPU | scaffold only ([web/](web/)) |

## Tech Stack

| Layer | Tool | Source |
|-------|------|--------|
| ML | Candle 0.8 (Hugging Face) — pure Rust | [Cargo.toml:32](Cargo.toml#L32) |
| GPU | Metal, CUDA, CPU fallback | [pipeline.rs:25](src/pipeline.rs#L25) |
| Data | bincode + zstd (dataset), safetensors (models) | [train.rs:221](src/train.rs#L221) |
| GUI | egui / eframe | [app.rs](src/app.rs) |
| CLI | clap | [main.rs:94](src/main.rs#L94) |

~11,479 lines of Rust across 31 `.rs` files. Zero Python. Zero JavaScript.

## Governance Documents

11 compliance-oriented documents [embedded via include_str!](src/main.rs#L10) (self-assessed, not independently audited). Source files in [govdocs/](govdocs/).
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
