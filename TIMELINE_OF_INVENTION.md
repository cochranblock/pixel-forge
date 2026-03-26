<!-- Unlicense — cochranblock.org -->

# Timeline of Invention

*Dated, commit-level record of what was built, when, and why. Proves human-piloted AI development — not generated spaghetti.*

> Every entry below maps to real commits. Run `git log --oneline` to verify.

---

## Entries

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
