<!-- Unlicense — cochranblock.org -->

# Timeline of Invention

*Dated, commit-level record of what was built, when, and why. Proves human-piloted AI development — not generated spaghetti.*

> Every entry below maps to real commits. Run `git log --oneline` to verify.

---

## Human Revelations — Invented Techniques

*Novel ideas that came from human insight, not AI suggestion. These are original contributions to the field.*

### NanoSign Model Integrity (April 2026)

**Invention:** A 36-byte model signing standard (4-byte magic `NSIG` + 32-byte BLAKE3 hash) appended to any model file format, verified on every load — tampered files rejected, unsigned files pass for backward compatibility.

**The Problem:** AI model files (`.safetensors`, `.onnx`, `.gguf`) are treated as trusted inputs. No framework verifies whether a model file has been modified after training. Supply chain attacks on model weights are invisible — a poisoned model produces subtly wrong output with no error.

**The Insight:** Every executable on modern operating systems is code-signed. Model files are executable logic (they determine program behavior at runtime) but have zero integrity verification. The fix doesn't need PKI or certificates — just append a hash of the file's own contents and check it on load. 36 bytes. Works with any format because it's appended, not embedded.

**The Technique:**
1. On save: compute BLAKE3 hash of file contents, append `NSIG` + 32 bytes
2. On load: read last 36 bytes, check for `NSIG` magic, verify hash matches file contents minus trailer
3. Unsigned files (no `NSIG` magic): pass silently for backward compatibility
4. Tampered files (hash mismatch): hard reject with error

**Result:** Every model file in the pixel-forge pipeline is integrity-verified on load. Zero performance cost (BLAKE3 is fast). Zero format dependency. Works with safetensors, ONNX, GGUF, or any binary format.

**Named:** NanoSign
**Commit:** `92748094`
**Origin:** Michael Cochran's defense background — weapons systems verify firmware integrity before execution. AI models should have the same guarantee. Designed as an open standard, published at `kova/docs/NANOSIGN.md`.

### MoE Cascade Pipeline — Cinder to Quench to Anvil (March 2026)

**Invention:** A tiered diffusion pipeline where a tiny model (Cinder, 1.09M params) generates rough output, a medium model (Quench, 5.83M params) refines it, and an XL model (Anvil, 16.9M params) adds detail — each tier uses the previous tier's output as its starting noise rather than pure random noise.

**The Problem:** Diffusion models are all-or-nothing. A 1M param model produces blurry output. A 17M param model produces good output but takes 10x longer and can't run on phones. There's no way to get "good enough for mobile, great on desktop" from one pipeline.

**The Insight:** Image generation is like painting — you don't start with details. Sketch the shape, block in color, then refine. Each stage needs less intelligence than the previous one because it has more structure to work with. A tiny model can sketch. A medium model can color. An XL model can detail. Chain them.

**The Technique:**
1. Cinder (1.09M, 4.2MB): generates 32x32 rough sprite from pure noise
2. Quench (5.83M, 22MB): takes Cinder output as starting point, refines with fewer diffusion steps
3. Anvil (16.9M, 64MB): takes Quench output as starting point, adds fine detail
4. On mobile: Cinder only (fast, small). On desktop: full cascade. User never chooses — device detection picks the tier.
5. Expert heads (shape/color/detail/class) provide MoE routing within tiers

**Result:** Mobile gets sprites in <1 second from Cinder alone. Desktop gets full cascade quality. Same training data, same pipeline, three quality tiers. No cloud API needed.

**Named:** Cinder-Quench-Anvil Cascade
**Commit:** See March 2026 cascade entries
**Origin:** Art school fundamentals — every painting teacher says "block in big shapes first, details last." Applied to diffusion model architecture as a tiered pipeline.

### Hybrid Class Conditioning (March 2026)

**Invention:** Replace integer class labels with a two-part conditioning system: 10 super-categories (small embedding table) + 12 binary tags (alive, humanoid, held_item, etc.) — new classes are just new tag combinations, zero retraining needed.

**The Problem:** Standard class-conditioned diffusion uses integer labels (class 0 = sword, class 1 = dragon, ...). Adding a new class requires retraining the entire model. Scaling from 16 to 108 classes means the embedding table grows linearly and every new class needs training examples.

**The Insight:** Game items aren't unique categories — they're combinations of properties. A dragon is [alive, hostile, magical]. A magic sword is [held_item, magical]. A robot is [alive, tech, hostile]. If the model learns properties instead of categories, new classes are just new property combinations — no retraining.

**The Technique:**
1. 10 super-categories: character, monster, weapon, armor, nature, building, food, vehicle, furniture, ui
2. 12 binary tags: alive, humanoid, held_item, worn, nature, built, magical, tech, small, hostile, edible, ui
3. `class_cond.rs`: lookup table mapping 108 class names to super-category + tag vector
4. Model trains on (super_category_embedding + tag_vector), not integer labels
5. New class = new lookup entry, zero weight changes

**Result:** Scaled from 16 to 108 classes without retraining. New classes (e.g., "cybernetic_owl") just need a tag combination [alive, tech, small] and super-category [monster].

**Named:** Hybrid Class Conditioning
**Commit:** `8e72544c`
**Origin:** Game design taxonomy — every RPG item system uses categories + properties, not flat lists. Applied to diffusion model conditioning.

### 2026-04-08 — Human Revelations Documentation Pass

**What:** Documented novel human-invented techniques across the full CochranBlock portfolio. Added Human Revelations section with NanoSign, MoE Cascade Pipeline, and Hybrid Class Conditioning.
**Commit:** See git log
**AI Role:** AI formatted and wrote the sections. Human identified which techniques were genuinely novel, provided the origin stories, and directed the documentation pass.

---

## Entries

### 2026-04-03 — NanoSign Model Integrity + Documentation P23

**What:** Integrated [NanoSign](src/nanosign.rs) — BLAKE3 integrity signing for all model files. Spec: [kova/docs/NANOSIGN.md](https://github.com/cochranblock/kova/blob/main/docs/NANOSIGN.md). Every `.safetensors` file is signed on save (36 bytes: `NSIG` + BLAKE3 hash) and verified on load. Tampered files are rejected; unsigned files pass for backward compat. Covers all 6 save paths ([train](src/train.rs#L442), [discriminator](src/discriminator.rs), [judge](src/judge.rs), [expert](src/expert.rs), [combiner](src/combiner.rs), [lora](src/lora.rs)) and all 10 load paths ([quantize::load_varmap](src/quantize.rs#L161), --resume, and module-specific loaders). Added `blake3` dependency.

Full doc audit per P23 Triple Lens: guest analysis (outsider perspective), code verification (every claim source-linked), truthfulness pass (false claims corrected). Updated all docs: README, PROOF_OF_ARTIFACTS, TIMELINE_OF_INVENTION, SOURCES.md, MOE_PLAN, ATTRIBUTION.

Planned tiered micro-model architecture: palette specialist (50K) → silhouette generator (200K) → detail painter (Anvil-class). Decomposes the generation problem so each specialist does less work. Infrastructure partially exists in [moe.rs](src/moe.rs) stage cascade.

Investigated [any-gpu](https://github.com/cochranblock/any-gpu) for cross-vendor GPU training (AMD 5700 XT on bt). any-gpu has 31 forward ops (conv2d, group_norm, attention, etc., 54/54 tests passing) but needs autograd + optimizer for training. Path forward: add backward ops to any-gpu, then `--features any-gpu` in pixel-forge.

**Commits:** `92748094` (NanoSign), `d1e60bf6` (doc audit), `2cfc5c0a` (source-linked README), `801bc218` (truthful README)

**AI Role:** AI implemented NanoSign module, performed P23-style guest analysis (optimist: solid engineering; pessimist: output quality not game-ready; paranoia: unsigned models, false data claims). Human directed NanoSign integration and approved documentation rewrites.

### 2026-04-01/02 — Inference Bug Fixes + Training Improvements + Anvil v7

**What:** Found and fixed three inference bugs that caused blurry output:
1. [CFG scale was 1.0](src/train.rs#L759) (disabled) — raised to 3.0. Model trains with CFG dropout but sampling never used the unconditional path.
2. [Anvil sampling started from uniform noise](src/train.rs#L1183) while training used Gaussian — distribution mismatch wasted early denoising steps. Fixed all 4 sampling paths to use [seeded_noise](src/train.rs#L764).
3. [DDIM noise extraction](src/moe.rs#L63) had no clamping — division by small values caused numerical blow-up. Added [-3,3] clamp.

Added [`--resume` flag](src/train.rs#L59) for fine-tuning from existing checkpoints. Added [rotation augmentation](src/train.rs#L413) (0/90/180/270). Kicked off Anvil v7 training on lf (fine-tuning from v6, rotation, lr=5e-5, bs=8).

Tested CUDA on lf's RTX 3070 — same epoch speed as CPU (~780s), EMA caused OOM (15GB RSS). CPU training is the stable path on 16GB nodes.

**Commits:** `d4b28270` (resume + rotation), `68f2183a` (inference fixes)

**AI Role:** AI identified inference bugs via code analysis, implemented fixes. Human directed investigation priorities and managed worker node deployment.

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

**What:** Built diffusion models from scratch using Candle. Early codenames were Spark/Flame/Blaze (later renamed to Cinder/Quench/Anvil). Three UNet tiers: [Cinder](src/tiny_unet.rs#L16) (1.09M/4.2MB), [Quench](src/medium_unet.rs#L17) (5.83M/22MB), [Anvil](src/anvil_unet.rs#L17) (16.9M/64MB). On-device training with cosine LR, CFG dropout, min-SNR weighting.
**Why:** No dependency on external model providers. Models trained on curated data, run on-device with zero cloud calls.
**Commit:** See kova repo `kova_model` entries and pixel-forge early commits
**AI Role:** AI implemented UNet architecture and training loop. Human directed model tiers, data curation, and quality gates.

---

*Part of the [CochranBlock](https://cochranblock.org) zero-cloud architecture. All source under the Unlicense.*
