# pixel-forge (pixel art generator)

## Identity

Part of the Cochran Block ecosystem. Powered by KOVA. Human direction, AI execution.

## Build Commands

| Action | Command |
|--------|---------|
| Build (Metal) | `cargo build --release -p pixel-forge` |
| Build (CUDA) | `cargo build --release -p pixel-forge --features cuda --no-default-features` |
| Build (CPU) | `cargo build --release -p pixel-forge --no-default-features` |
| Train Anvil | `cargo run --release -- train --data data_v3_32 --anvil --epochs 200 --lr 2e-4 --batch-size 16 --no-ema --checkpoint-every 1` |
| Train Quench | `cargo run --release -- train --data data_v3_32 --medium --epochs 200 --lr 2e-4 --batch-size 16 --no-ema --checkpoint-every 1` |
| Train Cinder | `cargo run --release -- train --data data_v3_32 --epochs 500 --lr 2e-4 --batch-size 128 --no-ema` |
| Train Experts | `cargo run --release -- train-experts --data data --epochs 50` |
| Train Judge | `cargo run --release -- train-judge` |
| Train PaletteNet | `cargo run --release -- train-palette-net --data data_v3_32 --output models/palette.safetensors` |
| Prep Silo Cond | `cargo run --release -- prep-silo-cond --data data_v3_32 --output data_silo_cond_32` |
| Train Cinder-detail | `cargo run --release -- train --data data_v3_32 --condition data_silo_cond_32 --epochs 100 --lr 1e-4 --resume pixel-forge-cinder.safetensors --output models/cinder-detail.safetensors` |
| Generate (Anvil) | `cargo run --release -- anvil character --count 4 --steps 40 --palette stardew` |
| Generate (Cinder) | `cargo run --release -- generate character --palette stardew` |
| Tiered Pipeline | `cargo run --release -- tiered warrior --count 4 --palette endesga32` |
| Cascade (MoE) | `cargo run --release -- cascade character --count 16` |
| Auto-detect | `cargo run --release -- auto character` |
| Scene | `cargo run --release -- scene biome` |
| GUI | `cargo run --release` |
| Plugin (kova) | `cargo run --release -- plugin` |

## Architecture

| Module | File | Purpose |
|--------|------|---------|
| tiny_unet | src/tiny_unet.rs | Cinder model — 1.09M params, [32,64,64] channels |
| medium_unet | src/medium_unet.rs | Quench model — 5.83M params, [64,128,128] + self-attention |
| anvil_unet | src/anvil_unet.rs | Anvil model — 16.9M params, XL |
| train | src/train.rs | Training loop, sampling, data pipeline |
| app | src/app.rs | egui GUI — device auto-detect, generation, gallery |
| device_cap | src/device_cap.rs | Device detection, tier selection, benchmarks |
| cluster | src/cluster.rs | Distributed generation across SSH nodes |
| combiner | src/combiner.rs | SlotGridTransformer — scene composition |
| judge | src/judge.rs | MicroClassifier — quality scoring |
| expert | src/expert.rs | MoE expert heads — shape/color/detail/class |
| expert_train | src/expert_train.rs | Expert training on frozen Quench |
| moe | src/moe.rs | Cascade pipeline — Cinder → Quench + Experts |
| scene | src/scene.rs | 8x8 SceneGrid, biome generation |
| swipe_store | src/swipe_store.rs | Tinder-style swipe data for judge training |
| lora | src/lora.rs | Rank-4 LoRA adapters for TinyUNet |
| discriminator | src/discriminator.rs | Quality gate — binary classifier |
| palette | src/palette.rs | Color palettes + quantization |
| class_cond | src/class_cond.rs | Hybrid conditioning: 10 super-categories + 12 binary tags |
| plugin | src/plugin.rs | JSON protocol for kova integration |
| poa | src/poa.rs | Proof of Authorship signing |
| pipeline | src/pipeline.rs | SD pipeline (optional, desktop) |
| gpu_lock | src/gpu_lock.rs | File-based GPU lock for training |
| relight | src/relight.rs | 4-directional sprites from SDF + normals |
| quantize | src/quantize.rs | f32 → f16 model quantization |
| nanosign | src/nanosign.rs | NanoSign BLAKE3 model integrity — sign on save, verify on load |
| tiered_pipeline | src/tiered_pipeline.rs | Tiered pipeline — silo router → noise blend → Cinder detail |
| palette_net | src/palette_net.rs | PaletteNet — ~100K MLP predicting class palette from conditioning |

## Model Tiers

| Tier | Name | Token | Params | Size | File |
|------|------|-------|--------|------|------|
| Tiny | Cinder | m0 | 1.09M | 4.2MB | pixel-forge-cinder.safetensors |
| Medium | Quench | m1 | 5.83M | 22MB | pixel-forge-quench.safetensors |
| XL | Anvil | m2 | 16.9M | 64MB | pixel-forge-anvil.safetensors |

## Tiered Pipeline

Three specialist models chained for higher quality output:

```
class_cond::lookup(class)
    ↓
[Stage 1: Shape]  Per-class MicroUNet silo (~97K params)
    → coarse RGB sprite → blend with noise at refine_from_t=0.45
    → falls back to pure noise if no silo available
    ↓
[Stage 2: Palette]  PaletteNet (~100K params)   [Phase 2 — train first]
    → predict 8 class-appropriate palette colors
    → quantize to nearest endesga32 entries
    ↓
[Stage 3: Detail]  Cinder (1.09M params, or Cinder-detail 6ch)
    → DDIM from silo-seeded start → full-resolution sprite
    ↓
palette::quantize() → final output
```

**Training sequence for best results:**
1. `train-palette-net` — 10 min CPU (extracts K-means colors per class)
2. `prep-silo-cond --data data_v3_32 --output data_silo_cond_32` — coarsen training images (32→16→32) as conditioning hints
3. `train --data data_v3_32 --condition data_silo_cond_32 --epochs 100 --lr 1e-4 --resume pixel-forge-cinder.safetensors --output models/cinder-detail.safetensors` — fine-tune 6ch Cinder
4. `tiered warrior --count 8 --detail-model models/cinder-detail.safetensors` — compare vs `generate`

**Cinder-detail (6ch):** auto-detected at inference via `conv_in.weight` shape in safetensors header. When 6ch, the DDIM loop prepends the silo output as a 3ch structural hint before each denoising step.

**Silo routing:** reads `class_config.toml` in `--silo-dir`. Falls back to `{silo_dir}/{class}.safetensors`. Falls back to pure Cinder if nothing found.

## Compression Map

Kova P13 compliant. Full map in `docs/compression_map.md`.

Key ranges: f0–f79 (functions), t0–t31 (types), m0–m2 (models), c0–c23 (CLI commands), M0–M28 (modules).

## Tech Stack

- ML: Candle (Hugging Face) — pure Rust
- GPU: Metal (Apple Silicon), CUDA (NVIDIA), CPU fallback
- Serialization: bincode + zstd (dataset), safetensors (models)
- GUI: egui / eframe
- CLI: clap
- GPU Scheduling: kova c2 gpu (file-based lock + priority queue)

## Kova Integration

- Plugin protocol: JSON RPC over stdin/stdout (`pixel-forge plugin`)
- GUI panel: kova T220 discovers binary, drives via plugin
- GPU scheduling: shared with kova c2 gpu lock
- Cluster: uses same IRONHIVE nodes (n0–n3) as kova

## Standards (inherited from kova)

- Use "augment" not "intent" in user-facing text
- P12 AI Slop Eradication — banned words apply here too
- P13 Compression — respect tokenization map
- P15 No circular dependencies
- Single binary model — one crate, lib + bin
- Error handling: anyhow (not thiserror — standalone project)
- No Python. No JavaScript. Pure Rust.

## Class Conditioning (v2)

Hybrid system replacing the old 16-class integer embeddings:
- **10 super-categories** — small embedding table (11 entries incl. CFG null)
- **12 binary tags** — `[alive, humanoid, held_item, worn, nature, built, magical, tech, small, hostile, edible, ui]`
- 108 class dirs mapped in `src/class_cond.rs`
- `class_cond::lookup("dragon")` → super=monster, tags=[alive, hostile, magical]
- New classes = new tag combos, zero retraining

## Training Data

- 52K+ artist-made tiles from 7 CC0/CC-BY sources
- 14K Gemini-generated pixel art sprites (AI-augmented, verified quality)
- Balanced to 19,876 tiles in `data_v3_32/` (capped 2K/class, 68 active classes)
- Full unbalanced set: 75K+ tiles in `data_v2_32/` (108 class dirs)
- No copyrighted game rips. Gemini sprites are AI-generated, not scraped.
- Sources documented in `data/SOURCES.md`

## NanoSign Model Integrity

All `.safetensors` model files are signed with NanoSign (BLAKE3, 36 bytes appended).
- **Save:** `nanosign::sign_and_log()` after every `varmap.save()` or `save_with_marker()`
- **Load:** `nanosign::verify_or_bail()` before every `varmap.load()` or `load_varmap()`
- **Unsigned files:** pass (backward compat). **Tampered files:** rejected.
- Spec: `~/kova/docs/NANOSIGN.md`

## P23 Triple Lens

Architecture decisions use three opposing perspectives (per kova P23):
- **Optimist:** best case, what works, competitive advantages
- **Pessimist:** what fails, gaps, hardest unsolved problems
- **Paranoia:** security risks, attack vectors, failure modes
- **Synthesis:** combines into one honest assessment with priority-ordered action items

Applied to: documentation audits, feature planning, architecture reviews.

## Future: any-gpu Backend

[any-gpu](https://github.com/cochranblock/any-gpu) (wgpu/Vulkan tensor engine) is planned as an alternative training backend. Would enable training on AMD GPUs (bt's 5700 XT) alongside NVIDIA (lf's 3070). Currently has 31 forward ops; needs autograd + optimizer before pixel-forge can use it for training.

## Future: Tiered Pipeline

Planned decomposition of monolithic diffusion into specialists:
1. Palette specialist (~50K params) — picks 8-16 colors for a given class
2. Silhouette generator (~200K params) — binary mask shape only
3. Detail painter (Anvil-class) — fills in details on correct silhouette + palette

Each specialist trains faster and does its job better than one model doing everything.

## Anti-Patterns

- No mocks — use real tensors, temp dirs
- No Python dependencies
- No cloud APIs for generation
- No copyrighted training data
- Warning suppression needs justification
