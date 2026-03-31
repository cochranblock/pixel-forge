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
| Generate (Anvil) | `cargo run --release -- anvil character --count 4 --steps 40 --palette stardew` |
| Generate (Cinder) | `cargo run --release -- generate character --palette stardew` |
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

## Model Tiers

| Tier | Name | Token | Params | Size | File |
|------|------|-------|--------|------|------|
| Tiny | Cinder | m0 | 1.09M | 4.2MB | pixel-forge-cinder.safetensors |
| Medium | Quench | m1 | 5.83M | 22MB | pixel-forge-quench.safetensors |
| XL | Anvil | m2 | 16.9M | 64MB | pixel-forge-anvil.safetensors |

## Compression Map

Kova P13 compliant. Full map in `docs/compression_map.md`.

Key ranges: f0–f40 (functions), t0–t24 (types), m0–m2 (models), c0–c18 (CLI commands).

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

- 75,182 curated tiles from 7 CC0/CC-BY sources + Gemini-generated sprites
- 108 class directories in `data_v2_32/`
- No copyrighted game rips.
- Data dir: `data_v2_32/` with bincode+zstd cache (v2 format with super_ids + tags)
- Sources documented in `data/SOURCES.md`

## Anti-Patterns

- No mocks — use real tensors, temp dirs
- No Python dependencies
- No cloud APIs for generation
- No copyrighted training data
- Warning suppression needs justification
