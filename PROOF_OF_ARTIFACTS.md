<!-- Unlicense — cochranblock.org -->

# Proof of Artifacts

*Concrete evidence that this project works, ships, and is real.*

> Three AI models. First MoE diffusion under 30MB. Pure Rust. No cloud.

## Architecture

```mermaid
flowchart TD
    User[User] --> GUI[egui GUI / CLI]
    GUI --> Auto[Device Auto-Detect]
    Auto --> Cinder[Cinder: 1.09M params, 4.2 MB]
    Auto --> Quench[Quench: 5.83M params, 22 MB]
    Auto --> Anvil[Anvil: 16.9M params, 64 MB]
    Cinder --> Cascade[MoE Cascade]
    Quench --> Cascade
    Cascade --> Experts[4 Expert Heads]
    Experts --> Shape[Shape Expert]
    Experts --> Color[Color Expert]
    Experts --> Detail[Detail Expert]
    Experts --> Class[Class Expert]
    Cascade --> Judge[Judge Model: 16K params]
    Judge --> Output[Game-Ready Sprite]
    GUI --> Scene[Scene Gen: 8x8 Biome Grids]
    GUI --> Cluster[Cluster Distribution: 4 GPU Nodes]
```

## Build Output

| Metric | Value |
|--------|-------|
| Lines of Rust | 11,322 across 30 modules |
| Public functions | 170+ across all modules |
| Direct dependencies | 16 required + 3 optional |
| Binary size (desktop ARM) | 9.2 MB (opt-level=z, LTO, strip) |
| Binary size (desktop x86) | 7.6 MB |
| Binary size (Linux) | 11.3 MB |
| Model: Cinder v6 | 1.09M params, 4.2 MB, Gaussian noise, clean prediction |
| Model: Quench v6 | 5.83M params, 22.2 MB, training on gd (RTX 3050 Ti) |
| Model: Anvil v6 | 16.9M params, 64.5 MB, training on lf (RTX 3070) |
| Training data | 19,876 balanced tiles (capped 2K/class), 68 active classes |
| Class directories | 108 across 10 super-categories |
| Dataset size | 24 MB zstd-compressed bincode (RAM-loaded, zero disk I/O) |
| Android AAB | 30.5 MB (Cinder + Quench for cascade) |
| Sprite classes | 108 via hybrid conditioning (10 supers + 12 tags) |
| ML framework | Candle (pure Rust — Metal, CUDA, CPU) |
| Federal govdocs | 11 documents baked into binary |
| Noise type | Gaussian N(0,1) — fixed from uniform [0,1] |
| Prediction target | Clean image (epsilon prediction tested, reverted) |

## Training Loss Progress (v6, Gaussian noise)

| Model | Epoch 1 | Epoch 6 | Epoch 25 | Epoch 50 | Epoch 100 |
|-------|---------|---------|----------|----------|-----------|
| Cinder (batch 128, lr 2e-4) | 0.64 | 0.13 | 0.10 | 0.09 | 0.08 |
| Anvil (batch 16, lr 2e-4) | 0.19 | 0.08 | training | — | — |
| Quench (batch 16, lr 2e-4) | 0.22 | training | — | — | — |

## QA Results (2026-03-27)

| Round | Test | Result |
|-------|------|--------|
| QA Round 1 | `cargo build --release` | PASS — 0 errors, 0 warnings |
| QA Round 1 | `git status` | PASS — all committed |
| QA Round 2 | `cargo clean && cargo build --release` | PASS — 5m28s from scratch |
| QA Round 2 | `cargo clippy --release -- -D warnings` | PASS — 46 lints fixed, 0 remaining |
| QA Round 2 | test binary | N/A — no test feature |
| User Story | Empty class validation | PASS — clear error |
| User Story | Count=0 validation | PASS — clear error |
| User Story | Missing model validation | PASS — helpful error with training instructions |
| User Story | Invalid palette | PASS — lists available palettes |

## Key Artifacts

| Artifact | Description |
|----------|-------------|
| MoE Cascade | First under 30MB — Cinder drafts (10 steps), Quench + 4 expert heads refine (30 steps) |
| Expert Routing | Shape (steps 1-10), Color (11-20), Detail (21-30), Class (31-40) |
| Judge Model | Binary classifier trained from user swipes — filters bad sprites in microseconds |
| LoRA Adapters | Rank-4 on all conv layers — fine-tune from 200 swipes without retraining |
| Scene Generation | 8x8 biome grids (dungeon, forest, cave, village, space) with constraint satisfaction |
| Device Auto-Detect | Probes GPU/RAM, selects optimal tier, benchmarks, degrades gracefully |
| f16 Quantization | Halves model sizes for mobile without quality loss |
| Proof of Authorship | Ed25519 signed Ghost Fabric packets |
| Hybrid Conditioning | 10 super-categories + 12 binary tags — new classes without retraining |
| Play Store Pipeline | deploy-play.sh: cargo ndk → bundleRelease → fastlane supply |
| Federal Governance | 11 docs: SBOM, SSDF, FIPS, CMMC, ITAR/EAR, FedRAMP, etc. |

## Training Data Sources

| Source | Count | License |
|--------|-------|---------|
| Dungeon Crawl Stone Soup | 6,000+ | CC0 |
| DawnLike v1.81 | 5,000+ | CC-BY 4.0 |
| Kenney Roguelike/RPG | 1,700 | CC0 |
| Kenney Pixel Platformer | 1,100 | CC0 |
| Kenney 1-Bit Pack | 1,078 | CC0 |
| Hyptosis Tiles | 1,000+ | CC-BY 3.0 |
| David E. Gervais Tiles | 1,280 | CC-BY 3.0 |

## How to Verify

```bash
cargo build --release -p pixel-forge
cargo run --release -- auto character    # Auto-detect hardware, generate sprite
cargo run --release -- cascade character --count 16   # MoE cascade
cargo run --release                      # Launch GUI
```

---

*Part of the [CochranBlock](https://cochranblock.org) zero-cloud architecture. All source under the Unlicense.*
