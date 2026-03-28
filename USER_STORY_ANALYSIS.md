# User Story Analysis — Pixel Forge v0.6.0

Date: 2026-03-27
Analyst: Claude Opus 4.6 (simulating first-time user)

---

## 1. DISCOVERY

**First impression of README:** Clear in 5 seconds. "Free pixel art generator. Three AI models. Pure Rust." The one-liner is strong. The CochranBlock banner at the top is a bit much for someone who doesn't know what CochranBlock is — the product pitch starts at line 17, below the fold.

**What's clear:** It generates pixel art. It's local. No cloud. Multiple models.
**What's unclear:** Do I need to train a model first? Or does it ship with one? The Quick Start section starts with "Train Cinder" — that's the wrong first step for a user. A user wants to GENERATE, not train.

**Score: 7/10** — Good pitch, wrong ordering. Generate should be step 1.

---

## 2. INSTALLATION

```
cargo build --release    # 5m28s clean build
cargo run --release -- --help    # Works, lists 30+ commands
```

**Works.** Help text is overwhelming — 30 commands for a new user. The primary commands (generate, train, cascade) are buried in an alphabetical list next to cluster-deploy and gpu-lock and ghost-fabric signing.

**Score: 6/10** — Builds fine. Help text needs grouping (Generation / Training / Advanced).

---

## 3. FIRST USE (Happy Path)

User tries the obvious:
```
cargo run --release -- generate character --palette stardew
```

**RESULT: CRASH.**
```
Error: TensorNotFound("tag_proj.bias")
```

The default model file (`pixel-forge-cinder.safetensors`) is a v1 model with old `class_emb` weights. The code now only supports hybrid conditioning (`super_emb` + `tag_proj`). No v2 model is shipped or available for download.

**A new user who clones this repo cannot generate a single sprite.**

This is a SHIP-BLOCKER. The product literally does not work out of the box.

**Score: 0/10** — Product crashes on the primary use case.

---

## 4. SECOND USE CASE

Tried training:
```
cargo run --release -- train --data data_v2_32 --epochs 5 --img-size 32
```

This works IF you have training data. A new user cloning the repo does NOT have `data_v2_32/` — it's gitignored (83 MB compressed, 885 MB raw). The README doesn't explain how to get training data.

**Score: 3/10** — Training works but data acquisition is undocumented.

---

## 5. EDGE CASES

| Input | Result | Verdict |
|-------|--------|---------|
| `generate character` (no palette) | Uses default stardew. Crashes on model load. | Bad (crash) |
| `generate ""` | Accepts empty string, maps to ui_fx super. Crashes on model. | Bad (should reject) |
| `generate character --palette fake` | Clear error: "unknown palette: fake. Run palettes to list." | Good |
| `generate character --count 0` | Proceeds with "generating 0x" then crashes on model. | Bad (should reject count=0) |
| `generate character --model missing.st` | "No such file or directory" — clear. | Good |
| `palettes` | Lists all 7 palettes with descriptions. | Good |
| `--help` | Shows all commands. | Good |
| `generate character --count -1` | Clap rejects: "invalid value" | Good (clap handles) |
| No subcommand | Launches GUI | Good |

**Score: 5/10** — Error handling is mixed. Good for invalid palettes and clap validation. Bad for missing model (crashes) and empty/zero inputs.

---

## 6. FEATURE GAP ANALYSIS

What a user would EXPECT but can't do:

1. **Text-to-sprite** — The `sprite` command exists and accepts text descriptions, but the model doesn't actually use text conditioning. It's class-based only. The prompt input in the GUI is cosmetic.
2. **Download pre-trained models** — No `setup` command that actually downloads a working model. The `setup` command downloads SD models (4GB), not the tiny pixel-forge models.
3. **Batch export** — Can generate a grid, but no way to export individual sprites from a grid to separate files.
4. **Animation** — Pixel art users want walk cycles, attack animations. No frame sequencing.
5. **Custom palettes** — Only 7 built-in palettes. No way to import a custom palette (e.g., Lospec .hex format).
6. **Undo/history in GUI** — Generate, don't like it, no way to go back to the previous result.
7. **Resolution options** — Locked to 32x32. Users may want 16x16 (retro) or 64x64 (modern).
8. **Tileset mode** — `tileset` command exists but generates edge-connected tiles. Users would expect seamless tiling preview.

---

## 7. DOCUMENTATION GAPS

Questions a user would have that docs don't answer:

1. **"Where do I get the model?"** — No download link, no setup guide. The README says "Train Cinder" as step 1 but doesn't explain you need 75K images first.
2. **"What's MoE cascade?"** — Term used everywhere but never explained in user terms.
3. **"What's the difference between generate, auto, cascade, stage-cascade, forge, anvil?"** — 6 different generation commands with no guidance on which to use.
4. **"How do I use this on my phone?"** — No mention of APK download or mobile build in README.
5. **"What are super-categories and tags?"** — The hybrid conditioning system is explained in CLAUDE.md but not in user-facing docs.
6. **"Can I train on my own art?"** — Not explicitly answered. The curate command exists but the workflow isn't documented.

---

## 8. COMPETITOR CHECK

| Tool | Price | Quality | Ease of Use | Offline |
|------|-------|---------|-------------|---------|
| **Pixel Forge** | Free | Low-Medium (32x32, early models) | Hard (must train first) | Yes |
| **PixelLab** | $12/mo | High (128x128, animation) | Easy (web app) | No |
| **Aseprite + AI plugins** | $20 one-time | Medium | Medium | Partially |
| **DALL-E / Midjourney** | $20/mo | High (but not pixel-native) | Easy | No |
| **PixelVibe** | Free tier | Medium | Easy (web) | No |

**Honest assessment:** Pixel Forge's value is OFFLINE + FREE + TINY. No competitor runs a 4MB model on a phone. But the quality gap is significant — PixelLab produces game-ready assets, Pixel Forge produces 32x32 blobs that need human cleanup. The MoE cascade narrows this gap but isn't proven with the hybrid models yet.

**Differentiation:** Only local-first, pure-Rust, mobile-deployable pixel art AI. That's a real niche. The question is whether the output quality justifies the product.

---

## 9. SCORES

| Category | Score | Notes |
|----------|-------|-------|
| Usability | 3/10 | Crashes on first use. No pre-trained model shipped. |
| Completeness | 6/10 | Training, generation, cascade, GUI, Android — features exist but rough. |
| Error Handling | 5/10 | Good for some inputs (invalid palette), bad for others (empty class, missing model). |
| Documentation | 4/10 | README is good marketing but bad onboarding. No quickstart that works. |
| Would You Pay? | 2/10 | Not today. After models are trained and quality improves, maybe $5 one-time. |

---

## 10. TOP 3 FIXES

### Fix 1: Ship a working model OR graceful error
The #1 problem: product crashes on first use because no v2 model exists locally. Until training finishes, at minimum show a helpful error instead of a cryptic tensor panic.

### Fix 2: README Quick Start should work out of the box
The Quick Start section tells users to train first. For users who just want to generate, add a "Download pre-trained model" step or bundle one.

### Fix 3: Validate inputs before loading model
Empty class names, count=0, and missing model files should produce clear errors BEFORE attempting to load a 4MB model into GPU memory.

---

*Analysis complete. Implementing top 3 fixes below.*
