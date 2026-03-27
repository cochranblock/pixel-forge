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

# Pixel Forge v0.5.0

**Free pixel art generator. Three AI models. First MoE diffusion under 30MB. Pure Rust.**

Generate game-ready pixel art sprites from tiny diffusion models that fit in a mobile binary. No cloud. No Python. No subscription. One language.

## What It Does

- **Generate** pixel art sprites from 108 class dirs via hybrid conditioning (10 super-categories + 12 binary tags)
- **Train** your own models on curated artist-made pixel art datasets
- **Three tiers** — Cinder (4.2 MB, fast), Quench (22 MB, balanced), Anvil (64 MB, highest quality)
- **f16 quantization** — halve model size for mobile (Cinder: 2.1 MB, Quench: 11 MB)
- **MoE cascade** — Cinder drafts, Quench + 4 expert heads refine. Better than either alone.
- **Judge model** — binary quality classifier filters bad sprites automatically
- **LoRA adapters** — fine-tune from user feedback without retraining
- **Scene generation** — 8x8 biome grids with rule-based or model-based placement
- **Cluster distribution** — fan out generation across GPU nodes
- **GUI + Android app** — touch-friendly, device auto-detection, Cinder-only APK under 10 MB

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
# Train Cinder (tiny, ~10 hrs on RTX 3070 with 75K tiles)
cargo run --release -- train --data data_v2_32 --epochs 100 --img-size 32

# Train Quench (medium)
cargo run --release -- train --data data_v2_32 --epochs 100 --img-size 32 --medium

# Generate
cargo run --release -- generate character --palette stardew -o hero.png

# MoE cascade (Cinder drafts → Quench + Experts refines)
cargo run --release -- cascade character --count 16 -o characters.png

# Auto-detect device, pick best model
cargo run --release -- auto character

# Launch GUI
cargo run --release
```

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

- **75,182 curated tiles** from 7 CC0/CC-BY sources + Gemini-generated sprites across 108 class dirs
- No AI-generated images. No copyrighted game rips.
- Cosine noise schedule + min-SNR weighting + CFG dropout (10%) + EMA tracking
- Zero disk I/O during training — all data in RAM from bincode+zstd cache

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

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Framework | Candle (Hugging Face) |
| GPU | Metal (Apple Silicon), CUDA (NVIDIA), CPU fallback |
| Serialization | bincode + zstd (dataset), safetensors (models) |
| GUI | egui / eframe |
| CLI | clap |
| GPU Scheduling | kova c2 gpu (file-based lock + priority queue) |

Zero Python. Zero JavaScript. One language. 11,137 lines of Rust.

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
