# Attribution

## Pixel Forge Tiny UNet

The tiny diffusion model architecture and training approach were inspired by:

### PixelGen16x16
- **Author:** Anouar Khaldi (kld-anouar)
- **License:** MIT + CC BY 4.0
- **Repository:** https://github.com/kld-anouar/PixelGen16x16
- **What we used:** The concept of applying a small UNet2D diffusion model to 16x16 pixel art generation. All code was rewritten from scratch in Rust/Candle — no Python code was copied.
- **Acknowledgment:** PixelGen16x16 credits the "Diffusion Models Class by Hugging Face" for resources and inspiration.

### pixartdiffusion
- **Author:** Zak Buzzard (zzbuzzard)
- **License:** No explicit license in repository
- **Repository:** https://github.com/zzbuzzard/pixartdiffusion
- **What we used:** Reference for 32x32 pixel art diffusion model training patterns and iterative denoising sampling approach. No code was copied.

### PixDiff-PIG (Research Reference)
- **Authors:** (see paper)
- **Paper:** "PixDiff-PIG: Palette-Informed Diffusion for Pixel Art Generation" (December 2025)
- **What we used:** The concept of palette-aware diffusion for pixel art — validating that small UNet architectures produce high-quality pixel art when combined with palette constraints.

### Stardew Valley (Palette Inspiration)
- **Creator:** Eric "ConcernedApe" Barone
- **What we used:** The "stardew" built-in palette is inspired by the warm earth-tone color palette of Stardew Valley. No assets from the game were used — the palette colors were hand-selected to evoke a similar visual feel.
- **Acknowledgment:** Stardew Valley set the standard for modern pixel art in indie games. ConcernedApe built the entire game solo — art, music, code, writing — and proved one person can ship something extraordinary.

### Starbound (Palette Inspiration)
- **Developer:** Chucklefish
- **What we used:** The "starbound" built-in palette is inspired by Starbound's vibrant sci-fi color palette. No game assets were used.

### Anvil Sprite (App Icon)
- **Creator:** Stendhal art team
- **License:** CC0 (Creative Commons Zero — public domain)
- **Source:** https://opengameart.org/content/anvil-2
- **What we used:** The 64x64 black anvil sprite as the centerpiece of the app icon. Scaled and composited onto a dark background with cyan spark effects and "PF" monogram. No modifications to the original sprite pixels.

### Candle ML Framework
- **Author:** Hugging Face
- **License:** MIT / Apache 2.0
- **Repository:** https://github.com/huggingface/candle
- **What we used:** ML inference and training framework for Rust. The backbone of all model operations.

### Procedural Generation Techniques (Concept Attribution)
- **No Man's Sky** (Hello Games) — Seed-based deterministic generation. A single seed cascades through algorithms to produce reproducible content. Applied to pixel-forge's `--seed` flag: same seed + class = same sprite every time. Reference: https://nomanssky.fandom.com/wiki/Procedural_generation
- **Factorio** (Wube Software) — Deterministic chunk-based world generation from a seed. The map is endless but only generated when needed. Applied to on-demand sprite generation per game chunk. Reference: https://wiki.factorio.com/Map_generator
- **Wave Function Collapse** (Maxim Gumin) — Constraint-based tile placement from adjacency rules. Applied to scene generation (8x8 biome grids). MIT license. Reference: https://github.com/mxgmn/WaveFunctionCollapse

### tokio-prompt-orchestrator (Architecture Reference)
- **Author:** Matt Busel (mattbusel)
- **Repository:** https://github.com/mattbusel/tokio-prompt-orchestrator
- **What we used:** DAG pipeline pattern with bounded channels, circuit breaker, request deduplication, and backpressure handling. Adapted from LLM inference orchestration to kova C2 distributed job queue for training/generation across GPU nodes.
