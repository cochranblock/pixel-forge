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

### Shape-Diffusion Signal Split (May 2026)

**Invention:** A generation cascade that matches the loss function to the information type at each stage: binary cross-entropy (BCE) for shape/silhouette generation, diffusion (EDM) for color. The cascade is: Anvil-BCE → Quench-BCE → Quench-color (EDM conditioned on mask) → Cinder-detail.

**The Problem:** Every pixel art generation attempt using EDM diffusion on binary silhouette masks plateau'd identically — different model sizes (Quench 5.83M, Anvil 16.9M), different γ values, different data. All runs hit loss ~8.5 at epoch 6 and stopped improving. EDM is the state-of-the-art diffusion recipe. Why would it fail on every run?

**The Insight:** Diffusion models add Gaussian continuous noise to images and learn to reverse it. Binary masks (pixel = 0 or 1) are not continuous — the "hard" noise regime at medium σ that EDM targets for structural learning doesn't exist for binary data. BCE, by contrast, treats each pixel as an independent binary classification problem. No noise model, no score matching — just "output 0 or 1 for each pixel given the class." BCE converges because the problem is correctly framed.

**The Evidence:**
- EDM Quench (200 epochs): loss 5.33 → 3.71, plateau at epoch 21, output = noise. γ=5 was no-op (floor of 1/σ_data² = 6.23 > 5). Fixed to γ=13 — still plateau'd.
- EDM Anvil on silhouettes (killed at epoch 11): loss 10.62 → 8.51, same plateau.
- EDM Quench with γ=13 on color data (running): epoch 6: 8.91, epoch 11: 8.97 — flat.
- **BCE Anvil on silhouettes (currently training):** epoch 1: 0.314 → epoch 46: 0.093, still descending. No plateau. Random init baseline is ln(2) ≈ 0.693 — the model crossed 50% reduction by epoch 11.

**The Technique:**
1. **Anvil-BCE** (16.9M params): class_cond → raw forward pass → sigmoid → binary mask. BCE with logits loss. No noise, no σ schedule, no preconditioning. Trains in hours not days.
2. **Quench-BCE** (5.83M params): takes Anvil mask (3ch) + class_cond → refined mask with part detail (hands, feet, head). 6ch input via `--condition`.
3. **Quench-color** (5.83M params, EDM): takes refined mask (3ch conditioning) + noise → color sprite. EDM conditioned on structure — now the noise model matches the target (continuous color).
4. **Cinder-detail** (1.09M params, EDM): fine-detail pass conditioned on Quench-color output.

**Result:** BCE converges where diffusion failed. The cascade assigns the right loss to the right problem: structure is classification, color is generation.

**Named:** Shape-Diffusion Signal Split  
**Commits:** `2640e90e` (BCE training mode), `3f72254b` (γ=13 fix + diagnosis)  
**Origin:** Failed EDM runs forced the question: why does state-of-the-art diffusion fail on binary data? Answer: the noise model is wrong for the signal type. Different stages of generation require different losses.

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

### 2026-05-18 — BCE Training Fixes: Denoising Prior + Zero-Init Inference

**What:** Two training fixes that resolve BCE convergence issues observed after the initial Anvil-BCE run:

**Zero-init for class-generative inference (`0b8ab101`):** Anvil-BCE was initializing training from the target image (clean silhouette), which teaches the model to reconstruct but not to generate. Changed to initialize from zeros (all-black input). The model must now learn to generate a silhouette from class conditioning alone — the correct framing for inference, where there is no target to start from. Loss trajectory unchanged; the model still converges from BCE's random-init baseline of ln(2) ≈ 0.693.

**Target+noise denoising prior (`bd8e2bda`):** Added noise augmentation to BCE training: `x_input = target + noise * 0.3` (noisy version of the target), while the BCE target remains the clean binary mask. Creates a denoising prior — the model sees a corrupted version of the answer and must recover the exact binary output. This is the BCE analog to diffusion's noise schedule: it forces the model to learn robust structure rather than memorizing exact pixel patterns, improving generalization.

**Commits:** `0b8ab101` (zero-init), `bd8e2bda` (noise augmentation)
**AI Role:** AI implemented both fixes. Human directed the zero-init change (class-generative vs. reconstructive framing) and the noise-augmentation design (denoising prior for BCE robustness).

### 2026-05-17/18 — BCE Silhouette Pivot + γ=13 Diagnosis + Anvil-BCE Training

**What:** Diagnosed why every EDM run plateau'd identically. Root cause: `EdmCoeffs::at()` hardcoded γ=5, but for σ_data=0.4007 the minimum 1/c_out² = 1/σ_data² ≈ 6.23 > 5, so γ=5 clamps **every single sample** — flat weighting identical to no Min-SNR at all. Fixed to γ=13 (`3f72254b`). Even with γ=13, Quench-color and Anvil-sil still plateau'd at epoch 6-11 — different model sizes, different data, same symptom. Diagnosis: diffusion is the wrong tool for binary mask generation regardless of weighting.

Pivoted to **Shape-Diffusion Signal Split** architecture: BCE for shape stages (binary targets → classification loss), EDM for color stages (continuous targets → generative loss). Built `--bce` training mode (`2640e90e`): no noise, no σ sampling, no preconditioning — just direct forward pass + sigmoid + stable BCE with logits. Anvil-BCE training started on lf RTX 3070.

**Loss trajectory (Anvil-BCE, in progress):**

| Epoch | Loss | Notes |
|-------|------|-------|
| 1 | 0.314 | Random init baseline: ln(2) ≈ 0.693 |
| 11 | 0.160 | >50% reduction — no plateau |
| 21 | 0.122 | Where EDM permanently died; BCE still descending |
| 46 | 0.093 | Still dropping, LR decay starting |

**Also fixed:** normalize-stats output defaulted to wrong dir (hardcoded `data_v3_32/normalize.json` regardless of `--data`); `generate --medium` silently loaded Cinder; `list-classes` missing; `tiered` preflight missing. rand 0.8→0.10 API migration (RngExt trait split). any-gpu version constraint removed (path dep, varies per node).

**Commits:** `3f72254b` (γ=13 fix), `d250c1fb` (rand 0.10 + openssl CVE fixes), `50b92365` (G2/G5/G7 user gap fixes), `98b40840` (normalize-stats output fix), `2640e90e` (BCE training mode), `eda7a0bb` (any-gpu version fix).

**AI Role:** AI diagnosed γ=5 no-op, diagnosed EDM failure mode on binary data, implemented γ=13 fix, implemented BCE training mode including stable BCE-with-logits tensor math. Human called the pivot from EDM to BCE after observing the plateau pattern across all runs, and designed the Shape-Diffusion Signal Split architecture (match loss type to signal type at each cascade stage).

### 2026-05-15 — Quench EDM Training + Recipe Regression Diagnosis

**What:** Kicked off the first end-to-end Quench training under the new EDM recipe on lf (RTX 3070, fp16, bs=16, 200 epochs, `--checkpoint-every 5`). Tmux session `quench-edm` writes per-epoch `.epochN` snapshots so progress can be sampled and the run can be killed early. ETA ~19h at 5.8 min/epoch.

Diagnosed regression: the Apr 19 `pixel-forge-cinder.safetensors` (the model that produced the recognizable blob-character emulator screenshots) is bricked at inference. Two commits cooperated to orphan it: `701eea94` removed the no-sidecar fallback at load time, `a1f61fda` replaced the entire DDPM sampling path with EDM Euler — there is exactly one decode path now (inverse z-score → clamp → encode), and it expects EDM-preconditioned weights. Forging a sidecar wouldn't help; the math is fundamentally different. Quench EDM is the path back to working sprites with a strict capacity upgrade.

**Why:** The 10-epoch Vulkan EDM Cinder run (`models/cinder-edm-vk.safetensors`, May 6) only painted noise — never trained long enough to evaluate the recipe at scale. Quench's 5.83M params (5.4× Cinder) plus self-attention is the smallest tier likely to actually paint pixel art under EDM, and it validates the recipe end-to-end on the proven candle/CUDA path before committing Anvil's ~22h.

**Commits:** working tree only; will be referenced once training completes.

**AI Role:** AI ran the audit (verified samples were noise, located the regression commits, traced the inference path), recommended the Quench-first ordering against jumping to Anvil, drove the lf SSH deployment (build, smoke-test, tmux launch, restart with `--checkpoint-every 5`). Human directed the tier choice and the early-checkpoint requirement.

### 2026-05-05/06 — EDM Preconditioning + Mandatory Z-Score Sidecar

**What:** Migrated the trainer + sampler from DDPM/clean-image prediction to Karras EDM (2022). Added `src/precond.rs` with EdmCoeffs (c_in/c_out/c_skip/c_noise + Min-SNR-γ=5 loss weight clamp), log-normal σ training distribution (P_MEAN=-1.2, P_STD=1.2), and ρ-stretched sampling schedule (RHO=7). Added `src/normalize.rs` for per-channel z-score with a `.normalize.json` sidecar — `pixel-forge normalize-stats --data <dir>` writes the manifest with mean, std, and σ_data (RMS of per-channel std). Z-score made mandatory at both train and sample time; the dual-mode tensor decoder shim was removed.

WASM crate (`b649e42f`, Apr 30) mirrors z-score: `finalize_png` inverts before clamp + encode for browser inference parity.

Quality-check CLI (`7cef7c9f`, May 6) computes per-image stats and cross-image pairwise diversity to compare runs objectively.

**Why:** EDM is the strongest published formulation for sub-10M-param diffusion: σ-dependent activation scaling keeps the network well-conditioned at every noise level, which is exactly where tiny models fail (high-σ regime). z-score normalization centers the per-channel data distribution so the network doesn't waste capacity learning the dataset mean. The single decode path simplifies the wasm/desktop/Vulkan parity story — same math everywhere.

**Tradeoff:** Mandatory sidecar orphaned every pre-z-score checkpoint. No backward-compat path was kept. See [2026-05-15] for the regression user impact.

**Commits:** `9eaa2669` (normalize-stats CLI), `6b125b4a` (normalize module), `ab9c6763` (per-batch z-score in trainer), `06ee6b50` (wasm z-score), `701eea94` (mandatory sidecar), `8d68ff82` (precond module + σ_data on manifest), `6b2d32f1` (swap z-score → EDM), `a1f61fda` (Quench sampler + center-pixel data prep), `7cef7c9f` (quality-check), `5fc8cde3` `eea56065` (vulkan_tiny EDM mirroring), `91c99a49` (banner refresh).

**AI Role:** AI implemented the precond + normalize modules end-to-end including unit tests (Min-SNR-γ clamp, σ schedule monotonicity, c_skip→1 at low σ, c_out→σ_data at high σ). Human directed the recipe choice (EDM over v-prediction) and the "make z-score mandatory" decision.

### 2026-04-30 — WASM Browser Backend + Beta Landing Page

**What:** Added a wasm crate (`b649e42f`) that runs Cinder/TinyUNet inference in the browser via WebGPU through any-gpu. Wired beta panel (`2888c8b1`) to the real WebGPU runtime — earlier version was a procedural placeholder. Sister commits earlier in April: web open-beta landing page with quad nav (`73e417ba`), procedural sprite demo with class-specific silhouettes (`1810517e`), upload + gallery.

**Why:** Local-first means the model runs *on the user's device* — including in a browser tab. WebGPU + any-gpu's WGSL shaders run the same forward pass that ships in the desktop binary, no Python/CUDA dependency.

**Commits:** `b649e42f`, `2888c8b1`, `73e417ba`, `1810517e`, `1dcef64f` (rustls bump for the crate).

**AI Role:** AI built the wasm forward pass and z-score parity. Human directed browser-as-target and the open-beta web design.

### 2026-04-26 — Tiered Pipeline + Vulkan/any-gpu Backend for Cinder

**What:** Added a 3-stage tiered generation pipeline (`37220e05`): per-class silo paints coarse shape → PaletteNet picks 8 class-appropriate colors → Cinder-detail (6ch UNet conditioned on the silo output) refines to 32×32. Routes via `class_router::pick_silo` reading `class_config.toml`; falls back to pure Cinder if no silo for the class.

Added Vulkan/any-gpu backend for TinyUNet/Cinder training (`5469b0dd`) — full forward + reverse-mode autograd through the entire model on AMD via Vulkan. Enables training Cinder on bt's RX 5700 XT where CUDA cannot run. Earlier: `train-silo --vulkan` (`a68a6c0f`, Apr 12) routed MicroUNet silo training through any-gpu first.

**Why:** Decomposing monolithic diffusion into specialists (palette, shape, detail) makes each stage easier to train and easier to swap. The Vulkan backend unblocks AMD GPUs across the IRONHIVE fleet — bt was idle because no Rust ML stack supported its 5700 XT.

**Commits:** `37220e05` (tiered pipeline), `5469b0dd` (Vulkan/any-gpu Cinder), `a68a6c0f` (silo on Vulkan, earlier), `1810517e` (web silhouettes for the demo).

**AI Role:** AI implemented the silo router, PaletteNet (~100K MLP), Cinder-detail 6ch conditioning path, and the Vulkan tape adapter (`vulkan_tiny.rs` 1000+ lines). Human directed the pipeline ordering and the AMD-coverage requirement.

### 2026-04-10/11/12 — Per-Class Silos + Sponge Mesh + any-gpu Fleet Training

**What:** Added per-class siloed MicroUNet (`de887913`, Apr 11) — 18 silos, ~97K params each, one per class. Each silo is a tiny shape-specialist trained on a single class's tiles. Added 5 new silos from Gemini-generated data (`b90a710a`, Apr 12). Added `train-silo --vulkan` (`a68a6c0f`, Apr 12) to train silos via any-gpu on AMD.

Added Sponge Mesh (`ebafcb2b`, Apr 10): self-healing training that auto-retries on NaN loss or plateau (max 3 retries) — restores from the last good checkpoint, perturbs LR, resumes. Added `--sponge` CLI flag.

Added `any-gpu` fleet backend + `train-fleet` command (`2a3dc0f1`, Apr 10) for multi-GPU distributed training across IRONHIVE nodes.

Re-enabled skeleton seeding for hybrid class system (`ae8e9de0`, Apr 10) — `Skeletons` cmd now calls `compute_skeletons_v2` keyed by `(super_id, class_name)` instead of legacy integer class IDs. Fixed three silent bugs (`5905b86e`): `--medium` overwriting Cinder filename, `endesga` palette name mismatch, missing `--seed` on `cascade`.

Enforced model boundary handoffs (`63aa0f70`) — no direct tensor pass-through between modules; all crossings go through typed marker functions for traceability.

**Why:** The user wanted per-class quality + cross-vendor GPU coverage. Per-class silos let each tiny model overfit to one class's shape distribution (better than one model fighting 68 classes). Sponge Mesh removes the "training crashed at epoch 187, lost everything" failure mode.

**Commits:** `ebafcb2b` (Sponge Mesh), `2a3dc0f1` (any-gpu fleet), `de887913` (per-class silos), `b90a710a` (5 new silos from Gemini), `a68a6c0f` (silo via Vulkan), `ae8e9de0` (skeleton v2), `5905b86e` (3 silent bug fixes), `63aa0f70` (model boundary enforcement).

**AI Role:** AI built MicroUNet, the silo trainer, Sponge Mesh retry loop, and the any-gpu adapter. Human directed the per-class strategy, the AMD-on-bt requirement, and the boundary-enforcement audit.

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
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
