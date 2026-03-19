# Pixel Forge MoE — Mixture of Experts for Micro Diffusion

## The Idea

One base model (Quench, 5.8M params) loaded once in RAM. Multiple tiny expert heads (~50K params each) that specialize in different aspects of sprite generation. A router (the Judge model we already built) decides which expert fires at each denoising stage.

Nobody has done MoE at this scale for diffusion. The research papers are all 1B+ param models. We're proving it works at 6M params on a phone.

## Architecture

```
                    ┌─────────────────────┐
                    │   Noise (16×16×3)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │     Quench Base (5.8M params)    │
              │     Encoder → Bottleneck         │
              │     (frozen after training)       │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Router (Judge)     │
                    │   Scores features    │
                    │   Picks expert       │
                    └──────────┬──────────┘
                               │
           ┌───────────┬───────┴───────┬───────────┐
           ▼           ▼               ▼           ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
    │  Shape     │ │  Color     │ │  Detail    │ │  Class     │
    │  Expert    │ │  Expert    │ │  Expert    │ │  Expert    │
    │  ~50K      │ │  ~50K      │ │  ~50K      │ │  ~50K      │
    └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
          │              │              │              │
          └──────────────┴──────┬───────┴──────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │     Quench Base (decoder)          │
              │     (frozen, shared weights)        │
              └─────────────────┬─────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Output (16×16×3)   │
                     └─────────────────────┘
```

## The Four Experts

### Expert 1: Shape (Silhouette)
**When:** Steps 1-10 (high noise, establishing structure)
**What it learns:** Binary silhouette — is this pixel part of the sprite or background? Edge definition, mass distribution, symmetry.
**Architecture:** 2 Conv layers on bottleneck features → sigmoid mask
**Training signal:** Edge-detected versions of training sprites. Binary cross-entropy on silhouette prediction.
**Params:** ~51K (Conv 128→64→1, k=3)

### Expert 2: Color (Palette Coherence)
**When:** Steps 11-20 (shape established, filling in color)
**What it learns:** Which colors belong together. Skin tones cluster. Foliage clusters. Metal clusters. Prevents muddy color mixing.
**Architecture:** 2 Conv layers → per-pixel palette index prediction
**Training signal:** Palette-quantized training sprites. Cross-entropy on palette bin assignment.
**Params:** ~52K (Conv 128→64→48, k=3, 48 = max Stardew palette size)

### Expert 3: Detail (Texture)
**When:** Steps 21-30 (colors locked, adding internal structure)
**What it learns:** Shading gradients, highlights, dithering patterns, internal features (eyes, buttons, leaf veins).
**Architecture:** 2 Conv layers → residual detail map added to current output
**Training signal:** Difference between palette-quantized sprite and the raw sprite. MSE on the detail residual.
**Params:** ~49K (Conv 128→64→3, k=3)

### Expert 4: Class (Identity Verification)
**When:** Steps 31-40 (final pass — is this actually what we asked for?)
**What it learns:** Class-specific features. Characters have heads. Trees have trunks. Weapons are elongated. Corrects class drift.
**Architecture:** 2 Conv layers → class logit + correction vector
**Training signal:** Class classification loss + directional correction toward class centroid in feature space.
**Params:** ~53K (Conv 128→64→15+3, k=3, 15 classes + 3 correction channels)

### Total Expert Overhead
```
Shape:   ~51K params  (~200 KB)
Color:   ~52K params  (~204 KB)
Detail:  ~49K params  (~192 KB)
Class:   ~53K params  (~208 KB)
─────────────────────────────────
Total:  ~205K params  (~804 KB)
```

Combined with Quench: 5.8M + 205K = ~6M params, ~23 MB total.

## Router Logic

The router doesn't need to be smart. It's deterministic by denoising stage:

```rust
fn route(step: usize, total_steps: usize) -> Expert {
    let progress = step as f32 / total_steps as f32;
    match progress {
        p if p < 0.25 => Expert::Shape,
        p if p < 0.50 => Expert::Color,
        p if p < 0.75 => Expert::Detail,
        _              => Expert::Class,
    }
}
```

Phase 2 (after v1 ships): Replace deterministic routing with the Judge model. Judge looks at the intermediate output and scores which expert would help most. This lets the model spend more steps on shape if shape is bad, or skip straight to detail if shape is already clean. Adaptive compute per sprite.

## How Experts Integrate with the Base Model

Each expert operates on the bottleneck features (128-dim after Quench's encoder). The expert produces a small correction tensor that gets ADDED to the decoder input before the base model's decoder runs.

```rust
fn forward_with_expert(
    base: &MediumUNet,
    expert: &ExpertHead,
    x: &Tensor,
    t: &Tensor,
    class: &Tensor,
) -> Tensor {
    // Run encoder (shared)
    let features = base.encode(x, t, class);

    // Expert produces correction
    let correction = expert.forward(&features);

    // Add correction to features
    let augmented = features + correction;

    // Run decoder (shared)
    base.decode(augmented)
}
```

This means we need to split MediumUNet's forward() into encode() and decode(). Minor refactor — the bottleneck is already a clean split point.

## Training Strategy

1. **Freeze Quench.** Base model weights never change.
2. **Train each expert independently.** One at a time. Each trains on the full 52K dataset but with its specialized loss function.
3. **Training time per expert:** ~50K params on 52K images = minutes on the RTX 3070. All four experts train in under an hour.
4. **Train on lf (RTX 3070).** The GPUs are free right now.

### Training Order
1. Shape expert first (most impactful — bad silhouettes ruin everything)
2. Color expert second (palette coherence is the next biggest visual win)
3. Detail expert third (shading and texture refinement)
4. Class expert last (fine-tuning class identity)

### Expert Training Loop
```
for each expert:
    freeze base model
    for epoch in 0..20:
        for batch in dataset:
            x = add_noise(batch, noise_level_for_this_expert_range)
            features = base.encode(x)
            correction = expert(features)
            augmented = features + correction
            output = base.decode(augmented)
            loss = expert_specific_loss(output, target)
            backward(loss)  // only expert weights update
```

## Cascade Integration (Cinder → Quench + Experts)

Once experts are trained, the full pipeline:

```
1. Cinder runs steps 1-10    (fast draft — shape from noise)
2. Hand tensor to Quench
3. Quench + Shape expert      steps 11-15  (refine silhouette)
4. Quench + Color expert      steps 16-25  (lock palette)
5. Quench + Detail expert     steps 26-35  (add texture)
6. Quench + Class expert      steps 36-40  (verify identity)
```

Total: 40 steps, two models, four experts. Faster than 100-step Quench alone, better output than either model solo.

## File Plan

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `src/expert.rs` | ~250 | ExpertHead struct, 4 expert variants, forward, training |
| `src/moe.rs` | ~150 | Router, cascade pipeline, MoE forward pass |
| Modify `src/medium_unet.rs` | +30 | Split forward into encode() + decode() |
| Modify `src/main.rs` | +20 | Add `generate-moe` command |

## Model Files on Device

```
pixel-forge-quench.safetensors      22.3 MB  (base)
pixel-forge-cinder.safetensors       4.2 MB  (draft)
expert-shape.safetensors            ~200 KB
expert-color.safetensors            ~204 KB
expert-detail.safetensors           ~192 KB
expert-class.safetensors            ~208 KB
────────────────────────────────────────────
Total on device:                    ~27.1 MB
```

## Why This Wins

1. **First MoE diffusion model under 30 MB.** Nobody's done this.
2. **Each expert trains in minutes.** Users can train their own experts from swipe data (Judge feedback loop already built).
3. **The experts are swappable.** Different color expert for NES palette vs Stardew palette. Different class expert for fantasy vs sci-fi. Modular creativity.
4. **The cascade makes it fast.** Cinder handles the boring noise-to-blob phase. Quench + experts handle the interesting part. Sub-second generation on Metal.
5. **It ships inside a free app that installs Rogue Repo.**
