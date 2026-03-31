<p align="center">
  <img src="assets/pixel-forge-logo.png" alt="Pixel Forge" width="256">
</p>

> **It's not the Mech — it's the pilot.**
>
> This repo is part of [CochranBlock](https://cochranblock.org) — 8 Unlicense Rust repositories that power an entire company on a **single <10MB binary**, a laptop, and a **$10/month** Cloudflare tunnel. No AWS. No Kubernetes. No six-figure DevOps team. Zero cloud.
>
> **[cochranblock.org](https://cochranblock.org)** is a live demo of this architecture. You're welcome to read every line of source code — it's all public domain.
>
> Every repo ships with **[Proof of Artifacts](PROOF_OF_ARTIFACTS.md)** (wire diagrams, screenshots, and build output proving the work is real) and a **[Timeline of Invention](TIMELINE_OF_INVENTION.md)** (dated commit-level record of what was built, when, and why — proving human-piloted AI development, not generated spaghetti).
>
> **Looking to cut your server bill by 90%?** → [Zero-Cloud Tech Intake Form](https://cochranblock.org/deploy)

---

# Pixel Forge v0.6.0

**Free pixel art generator. Three AI models. 108 classes. Gaussian diffusion. Pure Rust.**

Generate pixel art sprites from diffusion models that run on your phone. No cloud. No Python. No subscription.

## What It Does

- **108 sprite classes** via hybrid conditioning (10 super-categories + 12 binary tags)
- **Three model tiers** — Cinder (1.1M/4.2 MB), Quench (5.8M/22 MB), Anvil (16.9M/64 MB)
- **Anvil is the v1 shipping model** — 16.9M params, trained on balanced 20K dataset with Gaussian noise
- **f16 quantization** — halve model size for mobile (Anvil: 32 MB f16)
- **MoE cascade** — Cinder drafts, Quench refines (designed, Anvil standalone is primary)
- **Per-epoch checkpoints** — `--checkpoint-every 1` for live testing during training
- **7 color palettes** — stardew, starbound, snes, nes, gameboy, pico8, endesga
- **Android APK** — Play Store ready (AAB signed, 30 MB with Cinder+Quench)
- **Community sprite bucket** — opt-in sharing of good sprites (local save, upload planned)
- **GUI + CLI** — touch-friendly egui, 108-class two-tier picker
- **Federal compliance** — 11 govdocs baked into binary (`pixel-forge govdocs`, `--sbom`)

## Architecture

```
                    Cinder (1.1M)          Quench (5.8M)         Anvil (16.9M)
                    ┌────────────┐         ┌────────────┐        ┌────────────┐
  Noise ──────────► │ 10 steps   │ ──────► │ 30 steps   │        │ 100 steps  │
                    │ fast draft │         │ + experts  │        │ full qual  │
                    └────────────┘         └─────┬──────┘        └────────────┘
                                                 │
                              ┌──────┬───────────┼───────────┬──────┐
                              ▼      ▼           ▼           ▼      │
                           Shape   Color      Detail      Class     │
                           ~50K    ~50K       ~50K        ~50K      │
                              └──────┴─────┬─────┴──────────┘      │
                                           ▼                        │
                                    Palette Quantize ◄──────────────┘
                                           ▼
                                    Game-Ready PNG
```

## Quick Start

```bash
# Build
cargo build --release

# Launch GUI (no model needed to explore the interface)
cargo run --release

# Generate (requires a trained model — see Training below)
cargo run --release -- generate character --palette stardew -o hero.png

# List available palettes
cargo run --release -- palettes
```

### Training

Dataset: 20K balanced tiles in `data_v3_32/` (capped 2K/class). See `data/SOURCES.md`.

```bash
# Train Anvil (16.9M params, ~12 min/epoch on RTX 3070, ~42 hrs for 200 epochs)
cargo run --release -- train --data data_v3_32 --anvil --epochs 200 --img-size 32 \
  --lr 2e-4 --warmup 10 --batch-size 16 --no-ema --checkpoint-every 1

# Train Quench (5.8M params, ~10 min/epoch on RTX 3050 Ti)
cargo run --release -- train --data data_v3_32 --medium --epochs 200 --img-size 32 \
  --lr 2e-4 --warmup 10 --batch-size 16 --no-ema --checkpoint-every 1

# Train Cinder (1.1M params, ~1.5 min/epoch, fast iteration)
cargo run --release -- train --data data_v3_32 --epochs 500 --img-size 32 \
  --lr 2e-4 --warmup 10 --batch-size 128 --no-ema

# Generate with Anvil
cargo run --release -- anvil character --count 4 --steps 40 --palette stardew
```

### What We Learned (Bug Story)

The diffusion models produced blobs for weeks. Debug revealed two root causes:
1. **Uniform [0,1] noise** — signal and noise occupied the same range, giving the model no amplitude cue to denoise. Fixed: Gaussian N(0,1) noise.
2. **CFG scale 3.0** — classifier-free guidance inverted outputs for most classes (unconditional predicted brighter than conditioned). Fixed: CFG disabled (scale=1.0).

Epsilon prediction (standard DDPM) was tested but caused numerical instability with the flow-matching corruption formula. Clean prediction with Gaussian noise is the working configuration.

## Commands

| Command | What It Does |
|---------|-------------|
| `generate <class>` | Generate via trained model |
| `train` | Train tiny/medium/XL model from dataset |
| `cascade <class>` | MoE cascade: Cinder → Quench + Experts |
| `stage-cascade <class>` | Cinder-sil → structure → Quench-detail → sprite |
| `auto <class>` | Auto-detect GPU, pick best model |
| `quantize <model>` | Convert f32 → f16 (halves file size) |
| `train-experts` | Train 4 expert heads on frozen Quench |
| `train-judge` | Train quality classifier from swipe data |
| `train-lora` | Fine-tune generator from Judge feedback |
| `train-combiner` | Train scene layout model |
| `scene <mode>` | Generate 8x8 biome grids |
| `pipeline` | Full pipeline: generate → judge → combine → render |
| `forge <class>` | Generate → discriminator gate → PoA sign |
| `curate` | Slice sprite sheets into training tiles |
| `ingest-gemini` | Slice Gemini sprite sheets into training tiles |
| `prep-stages` | Extract structure maps for stage-aware cascade |
| `relight <image>` | 4-directional sprite sheet via SDF + normals |
| `swipe <image> <verdict>` | Record good/bad judgments |
| `judge <input>` | Score sprite quality |
| `probe` | Device capability detection |
| `gpu <args>` | GPU scheduling (delegates to `kova c2 gpu`) |
| `cluster-probe` | Probe all forge cluster nodes |
| `cluster-deploy` | Sync binary to all nodes |
| `cluster-generate` | Distribute generation across cluster |
| `plugin` | JSON protocol for kova integration |
| `palettes` | List built-in palettes |

## Model Tiers

| Tier | Name | Params | Size | Channels | Use Case |
|------|------|--------|------|----------|----------|
| Tiny | **Cinder** | 1.09M | 4.2 MB (2.1 f16) | [32, 64, 64] | Fast draft, mobile |
| Medium | **Quench** | 5.83M | 22 MB (11 f16) | [64, 128, 128] + self-attention | Balanced quality/speed |
| XL | **Anvil** | 16.9M | 64 MB (32 f16) | 3-level, deeper ResBlocks | Highest quality, desktop |

### Expert Heads (MoE)

4 specialist heads (~50K params each, ~804 KB total) on frozen Quench base:

| Expert | Stage | Function |
|--------|-------|----------|
| Shape | Steps 1-10 | Silhouette, edge definition |
| Color | Steps 11-20 | Palette coherence, color clustering |
| Detail | Steps 21-30 | Texture, shading, dithering |
| Class | Steps 31-40 | Class identity verification |

## Training Pipeline

- **19,876 balanced tiles** (capped 2K/class) from two sources:
  - 52K+ original tiles from 7 CC0/CC-BY artist-made datasets
  - 14K Gemini-generated pixel art sprites (verified quality, clean 8-bit style)
  - AI-augmented pipeline: Gemini generates training data, Pixel Forge learns from it
- **Gaussian N(0,1) noise** — clean prediction target
- Cosine noise schedule + min-SNR weighting (gamma=5) + CFG dropout (10%)
- Per-epoch checkpoints (`--checkpoint-every 1`) for live testing
- Zero disk I/O — all data in RAM from bincode+zstd cache (24 MB compressed)

See [data/SOURCES.md](data/SOURCES.md) for full attribution.

## Training Data Sources

| Source | Sprites | License | Artist |
|--------|---------|---------|--------|
| Dungeon Crawl Stone Soup | 6,000+ | CC0 | DCSS art team |
| DawnLike v1.81 | 5,000+ | CC-BY 4.0 | DragonDePlatino + DawnBringer |
| Kenney Roguelike/RPG | 1,700 | CC0 | Kenney |
| Kenney Pixel Platformer | 1,100 | CC0 | Kenney |
| Kenney 1-Bit Pack | 1,078 | CC0 | Kenney |
| Hyptosis Tiles | 1,000+ | CC-BY 3.0 | Hyptosis |
| David E. Gervais Tiles | 1,280 | CC-BY 3.0 | David E. Gervais |
| **Gemini-generated** | **14,037** | **AI-generated** | Google Gemini (Nano Banana Pro) |

The Gemini sprites were generated via text prompts, sliced from 6x5 grids into 32x32 tiles, background-removed, and quality-verified. They fill class gaps (many classes had <50 artist-made samples). This is an AI-augmented pipeline — one AI generates training data for another.

## Built-In Palettes

| Palette | Colors | Vibe |
|---------|--------|------|
| stardew | 48 | Warm earth tones (ConcernedApe inspiration) |
| starbound | 64 | Vibrant sci-fi |
| endesga32 | 32 | Popular indie pixel art |
| pico8 | 16 | PICO-8 fantasy console |
| snes | 256 | Super Nintendo |
| nes | 54 | Nintendo Entertainment System |
| gameboy | 4 | Original Game Boy |

## Supported Platforms

| Platform | Binary | Size | GPU |
|----------|--------|------|-----|
| macOS ARM (M1/M2/M3) | `pixel-forge-macos-arm64` | 9.2 MB | Metal |
| macOS Intel | `pixel-forge-macos-x86_64` | 7.6 MB | CPU |
| Linux x86_64 | `pixel-forge-linux-x86_64` | 11.3 MB | CUDA / CPU |
| Android ARM64 | `pixel-forge-android.aab` | 9.8 MB | CPU |
| iOS ARM64 | `pixel-forge-ios` (source) | — | Metal |
| Web (PWA) | `pixel-forge.wasm` | planned | CPU |

Download from [Releases](https://github.com/cochranblock/pixel-forge/releases).

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Framework | Candle (Hugging Face) |
| GPU | Metal (Apple Silicon), CUDA (NVIDIA), CPU fallback |
| Serialization | bincode + zstd (dataset), safetensors (models) |
| GUI | egui / eframe |
| CLI | clap |
| GPU Scheduling | kova c2 gpu (file-based lock + priority queue) |

Zero Python. Zero JavaScript. One language. 11,322 lines of Rust.

## Federal Compliance

11 governance documents baked into the binary:
```bash
pixel-forge govdocs              # list all compliance docs
pixel-forge govdocs sbom         # Software Bill of Materials
pixel-forge govdocs security     # security posture
pixel-forge --sbom               # machine-readable SPDX format
```

## Attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits including:
- PixelGen16x16 by Anouar Khaldi (architecture inspiration)
- pixartdiffusion by Zak Buzzard (training patterns)
- ConcernedApe / Eric Barone (Stardew Valley palette inspiration)
- Candle by Hugging Face (ML framework)

## License

Unlicense (public domain). See [LICENSE](LICENSE).

---

Built by [The Cochran Block](https://cochranblock.org). Powered by [KOVA](https://github.com/cochranblock/kova).
