# Backlog

Prioritized. Most important at top. Max 20 items. Self-reorganizes based on recency and relevance.

Last updated: 2026-04-03. Session context: NanoSign shipped, inference bugs fixed, Anvil v7 training on lf.

---

1. `[fix]` Test output quality after CFG 3.0 + Gaussian noise fix — generate sample grids, compare against pre-fix blobs. Blocked on: Anvil v7 completing on lf.
2. `[feature]` Add `--cfg <f64>` CLI flag to generate/anvil/cascade commands — currently hardcoded at 3.0, needs runtime tuning. [src/train.rs:759](src/train.rs#L759), [src/main.rs](src/main.rs).
3. `[feature]` Auto-apply palette quantization after generation — snap f32 output to discrete colors. Add `--no-quantize` flag. [src/palette.rs:55](src/palette.rs#L55), [src/main.rs](src/main.rs).
4. `[build]` Rebuild on lf with NanoSign — `git pull && cargo build --release --no-default-features` on lf so next training run produces signed checkpoints.
5. `[fix]` EMA causes OOM on 16GB nodes (lf) with Anvil — investigate: can EMA shadows stay on disk instead of CPU RAM? Or reduce batch size further. [src/train.rs:110](src/train.rs#L110).
6. `[feature]` Add brightness/contrast augmentation — random ×[0.8,1.2] per sample to reduce dark-blob tendency. [src/train.rs](src/train.rs) batch loop.
7. `[feature]` Tiered pipeline: silhouette generator — train Cinder on binary masks extracted from training sprites. New preprocessing step in [src/train.rs](src/train.rs). Feeds into stage cascade in [src/moe.rs](src/moe.rs).
8. `[feature]` Tiered pipeline: palette specialist — 50K param model picks 8-16 colors per class. New module. Trains in seconds.
9. `[research]` Add autograd to [any-gpu](https://github.com/cochranblock/any-gpu) — backward ops for conv2d_weight, group_norm, swish, attention, upsample. Plus AdamW optimizer. ~1500 lines WGSL. Unblocks GPU training on bt's 5700 XT.
10. `[feature]` Add `--device vulkan` flag — wire any-gpu's `GpuDevice` into `pipeline::best_device()` as fourth backend option. Depends on: any-gpu autograd (#9). [src/pipeline.rs:25](src/pipeline.rs#L25).
11. `[feature]` Add self-attention at 16×16 resolution in Anvil — currently only at 8×8 and 4×4. Adds ~0.4M params, requires retrain. [src/anvil_unet.rs:17](src/anvil_unet.rs#L17).
12. `[feature]` DDPM stochastic sampler — add `--sampler ddpm|ddim` flag. DDPM injects noise each step, often produces crisper edges at cost of more steps. [src/train.rs](src/train.rs), [src/moe.rs](src/moe.rs).
13. `[docs]` Generate hero image for README — after quality fixes land, cherry-pick 16 best outputs across classes, create 4×4 grid at 4× upscale. Save as `assets/hero-grid.png`.
14. `[build]` Publish GitHub Releases — build platform binaries (macOS ARM/Intel, Linux x86) and upload as release artifacts. README already references the Releases page.
15. `[fix]` `.git/` is 1GB — large blobs leaked into history. Run `git filter-repo` to clean safetensors from history. Coordinate with any clones.
16. `[feature]` Sign existing model files — run NanoSign on all `.safetensors` in project root and on worker nodes (lf, gd). Currently only new saves are signed.
17. `[test]` Verify NanoSign tamper detection — corrupt a signed model file, confirm load is rejected. Corrupt an unsigned file, confirm it loads (backward compat).
18. `[research]` Noise schedule bias toward mid-range — oversample t∈[0.3,0.7] where detail forms. Beta distribution or stratified sampling. [src/train.rs:596](src/train.rs#L596).
19. `[docs]` Update `docs/compression_map.md` — add nanosign functions (f48+), NanoSign types. Verify existing mappings still match code.
20. `[research]` Distributed training across lf (CUDA/CPU) + bt (Vulkan via any-gpu) — different models or pipeline tiers per node. Depends on: any-gpu autograd (#9), `--device vulkan` (#10). Uses [src/cluster.rs](src/cluster.rs).

---

Dependencies on other cochranblock projects:
- **[any-gpu](https://github.com/cochranblock/any-gpu):** #9, #10, #20 — autograd needed before pixel-forge can train on AMD GPUs
- **[kova](https://github.com/cochranblock/kova):** NanoSign spec lives in kova docs. Plugin protocol ([src/plugin.rs](src/plugin.rs)) for GUI integration. C2 cluster orchestration for distributed training.
