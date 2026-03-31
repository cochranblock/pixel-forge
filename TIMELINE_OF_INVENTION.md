<!-- Unlicense — cochranblock.org -->

# Timeline of Invention

*Dated, commit-level record of what was built, when, and why. Proves human-piloted AI development — not generated spaghetti.*

> Every entry below maps to real commits. Run `git log --oneline` to verify.

---

## Entries

### 2026-03-28/29/30 — Diffusion Debug + Gaussian Noise Fix + Anvil Training + Multi-Arch

**What:** Debugged fundamental output quality issue. Root cause: uniform [0,1] noise made signal/noise indistinguishable. Fixed: Gaussian N(0,1). Tested epsilon prediction — numerical instability with flow-matching, reverted to clean prediction. Dataset rebalanced 75K→20K (capped 2K/class). Added `--checkpoint-every` flag for per-epoch saves. Anvil v6 (16.9M params) training on RTX 3070 — loss 0.08 at epoch 6, already showing structured output. Quench v6 on RTX 3050 Ti in parallel. Multi-arch release: macOS ARM/Intel, Linux x86, Android AAB. iOS scaffold + PWA scaffold. Mobile GUI: removed Anvil button, cascade default, hybrid model validation, clickable footer. Community sprite upload. Feature graphic. Store screenshots on Pixel 9 Pro XL.

**Commits (14):** `b84b5b8b`→`541720cd`

**AI Role:** AI performed systematic architecture debug (overfit test, noise analysis, tensor stats), identified uniform noise as root cause. Human directed the investigation priorities and training configuration.

### 2026-03-27 — Hybrid Class Conditioning + Play Store Pipeline + Federal Compliance

**What:** Replaced the hardcoded 16-class integer embedding system with hybrid conditioning: 10 super-categories + 12 binary tags. 108 class directories mapped via `class_cond.rs`. Stripped all legacy code paths. Trained Cinder v2 on lf (RTX 3070) and Quench v2 on gd (RTX 3050 Ti) in parallel with 75,182 tiles. Ingested 14,037 Gemini-generated sprites. Expanded GUI class picker to two-tier super-category → class browser. Set up Google Play deployment pipeline (AAB build, release signing, fastlane upload script). Shrank binary 25.8 MB → 9.2 MB via opt-level=z, LTO, strip, removing tokio/walkdir/dirs. Passed QA Round 1 + Round 2 (clean build + clippy -D warnings). Full user story analysis with scores. Added input validation (empty class, count=0, missing model). Created 11 federal governance documents (SBOM, SSDF, FIPS, CMMC, etc.). Deployed IRONHIVE C2 tmux dispatcher.

**Why:** The old 16-class system couldn't scale to 108 dirs. Hybrid conditioning makes the model composable — new classes via tag combos, zero retraining. Binary shrink targets mobile APK under 10 MB. Federal govdocs prepare for government procurement.

**Commits (11):**
- `5c45b2a5` — Gemini sprite ingest pipeline + 92 generation prompts
- `8e72544c` — hybrid class conditioning: 10 super-categories + 12 binary tags
- `750d3d66` — update docs for hybrid class system
- `80d2123c` — strip legacy from MediumUNet + AnvilUNet
- `7b96936d` — strip legacy ClassConditioner from TinyUNet
- `2115b82a` — v0.6.0: Cinder-only APK, strip Quench from mobile bundle
- `32313817` — Play Store pipeline + 108-class GUI picker + QA cleanup
- `a83c081e` — fix all clippy warnings (zero errors with -D warnings)
- `a64a990b` — shrink binary 25.8→9.2 MB: strip deps + size-optimize profile
- `c04ffe9d` — user story analysis + top 3 fixes from walkthrough
- `4c8a473c` — federal governance docs (SBOM, SSDF, FIPS, CMMC, etc.)

**AI Role:** AI implemented all code changes, training pipeline, GUI expansion, and document generation. Human directed architecture decisions (hybrid conditioning design, super-category taxonomy, binary size targets), validated training strategy, and managed IRONHIVE node deployment.

### 2026-03 — Structural Conditioning + Relight

**What:** Added 5-layer structural conditioning pipeline (SDF, normals, outline, luminance, skeleton). Stage-aware cascade: silhouette to color to detail. Relight module generates 4-directional sprites from SDF + normal maps. dtype-aware UNets for f16 inference.
**Why:** Moving from noise-to-image generation toward controllable, game-ready sprite output. Structural conditioning gives artists control over pose and shape.
**Commit:** `b77a3e2e`, `9bf4f349`, `83d7a43d`, `d92b4be1`
**AI Role:** AI implemented UNet modifications and conditioning layers. Human directed the pipeline architecture and validated output quality against game art standards.

### 2026-03 — v-Prediction + Cosine Sampling + Cascade Fix

**What:** Added v-prediction as alternative to clean-image prediction. Cosine LR decay with warm-up for training. Fixed cascade sampler to use cosine schedule + DDIM + CFG matching training. Skeleton-seeded generation (70/30 class mean / noise blend).
**Why:** v-prediction improves sample quality at low step counts. Cosine schedule fixes the mismatch between training and inference that was causing blurry outputs.
**Commit:** `e1141647`, `1b9825ab`, `7c501dbf`, `74da857f`
**AI Role:** AI implemented sampler math. Human identified the train/inference mismatch and directed the fix.

### 2026-03 — Mobile UX + Cascade Pipeline

**What:** Flipped cascade to Quench then Cinder, added Anvil desktop pipeline. Mobile default 10 steps (4x faster at 32x32). Bundled Quench + Cinder in APK for on-device cascade. Raised resolution floor from 16x16 to 32x32. Hidden model/settings behind advanced toggle.
**Why:** Making the app usable on phones — faster generation, cleaner UI, models bundled so no download needed.
**Commit:** `2fdb16b2`, `bedc1b48`, `2905f642`, `528fe452`, `c7047d93`
**AI Role:** AI built UI and bundling logic. Human directed UX decisions and performance targets.

### 2026-03 — f16 Quantize + Icon + Seed-Based Generation

**What:** f16 quantized model loading (storage savings without compute pain). Seed-based deterministic generation with procgen attribution. New nano banana pixel art anvil icon. Dual-color UI (cyan sparks + orange forge button). APK split by architecture.
**Why:** Shrink model files for mobile distribution while maintaining inference quality. Deterministic seeds enable reproducible art.
**Commit:** See `git log --oneline` March 22 entries
**AI Role:** AI implemented f16 casting and seed pipeline. Human directed quantization strategy and verified output quality.

### 2026-03 — From-Scratch Diffusion Models

**What:** Built Spark/Flame/Blaze tier transformer models from scratch using Candle. BPE tokenizer. On-device training with cosine LR, DPO data support, cross-entropy fix. Three bundled models: Cinder (4.2MB), Quench (22MB), Anvil (64MB).
**Why:** No dependency on external model providers. Models trained on curated data, embedded in the binary, run on-device with zero cloud calls.
**Commit:** See kova repo `kova_model` entries
**AI Role:** AI implemented transformer architecture and training loop. Human directed model tiers, data curation, and quality gates.

---

*Part of the [CochranBlock](https://cochranblock.org) zero-cloud architecture. All source under the Unlicense.*
