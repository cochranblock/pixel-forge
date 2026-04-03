# Backlog

Prioritized. Most important at top. Max 20 items. Self-reorganizes based on recency and relevance.

Last updated: 2026-04-03. Anvil v7 training restarted on lf (bs=4, no EMA, lr=5e-5). Tests added: 32 passing (nanosign, palette, class_cond, train::cosine_schedule).

---

1. `[fix]` Test output quality after CFG 3.0 + Gaussian noise fix — generate sample grids, compare against pre-fix blobs. Anvil v7 training on lf (bs=4, checkpoint every 5 epochs). Check `ssh lf 'tail -20 ~/pixel-forge/train-anvil-v7.log'`. **Blocked:** requires trained model files on lf. Run after v7 checkpoint lands.
2. `[feature]` Add brightness/contrast augmentation — random ×[0.8,1.2] per sample to reduce dark-blob tendency. [src/train.rs](src/train.rs) batch loop ~line 570.
3. `[feature]` Tiered pipeline: silhouette generator — train Cinder on binary masks extracted from training sprites. New preprocessing step in [src/train.rs](src/train.rs). Feeds into stage cascade in [src/moe.rs](src/moe.rs).
4. `[feature]` Tiered pipeline: palette specialist — 50K param model picks 8-16 colors per class. New module. Trains in seconds.
5. `[research]` Add autograd to [any-gpu](https://github.com/cochranblock/any-gpu) — backward ops for conv2d_weight, group_norm, swish, attention, upsample. Plus AdamW optimizer. ~1500 lines WGSL. Unblocks GPU training on bt's 5700 XT.
6. `[feature]` Add `--device vulkan` flag — wire any-gpu's `GpuDevice` into `pipeline::best_device()` as fourth backend option. Depends on: any-gpu autograd (#5). [src/pipeline.rs:25](src/pipeline.rs#L25).
7. `[feature]` Add self-attention at 16×16 resolution in Anvil — currently only at 8×8 and 4×4. Adds ~0.4M params, requires retrain. [src/anvil_unet.rs:17](src/anvil_unet.rs#L17).
8. `[feature]` DDPM stochastic sampler — add `--sampler ddpm|ddim` flag. DDPM injects noise each step, often produces crisper edges at cost of more steps. [src/train.rs](src/train.rs), [src/moe.rs](src/moe.rs).
9. `[docs]` Generate hero image for README — after quality fixes land, cherry-pick 16 best outputs across classes, create 4×4 grid at 4× upscale. Save as `assets/hero-grid.png`.
10. `[build]` Publish GitHub Releases — build platform binaries (macOS ARM/Intel, Linux x86) and upload as release artifacts. README already references the Releases page.
11. `[fix]` `.git/` is 1GB — large blobs leaked into history. Run `git filter-repo` to clean safetensors from history. Coordinate with any clones.
12. `[feature]` Sign existing model files — run NanoSign on all `.safetensors` in project root and on worker nodes (lf, gd). Currently only new saves are signed.
13. `[test]` Verify NanoSign tamper detection — corrupt a signed model file, confirm load is rejected. Corrupt an unsigned file, confirm it loads (backward compat).
14. `[research]` Noise schedule bias toward mid-range — oversample t∈[0.3,0.7] where detail forms. Beta distribution or stratified sampling. [src/train.rs:596](src/train.rs#L596).
15. `[docs]` Update `docs/compression_map.md` — add nanosign functions (f48+), NanoSign types, sample_seeded_cfg/sample_medium_seeded_cfg. Verify existing mappings still match code.
16. `[research]` Distributed training across lf (CPU) + bt (Vulkan via any-gpu) — different models or pipeline tiers per node. Depends on: any-gpu autograd (#5), `--device vulkan` (#6). Uses [src/cluster.rs](src/cluster.rs).

---

Done this session:
- ~~`[test]` Unit + integration tests~~ — 32 tests passing. nanosign (sign/verify/tamper/strip/backward-compat), palette (quantize, transparency, nearest-color, all named palettes), class_cond (super-cat routing, tag correctness, unknown fallback), train::cosine_schedule (bounds, monotonicity, unit range).
- ~~`[feature]` Add `--cfg <f64>` CLI flag~~ — `a6fa027cb`. Generate/anvil/cascade all accept `--cfg`.
- ~~`[feature]` Auto-apply palette quantization~~ — already done in all generation commands.
- ~~`[build]` Rebuild on lf with NanoSign~~ — built, v7 restarted with bs=4.
- ~~`[fix]` EMA OOM on 16GB nodes~~ — workaround: bs=4 no-ema uses 1.1GB RAM. EMA needs >16GB with Anvil.

Dependencies on other cochranblock projects:
- **[any-gpu](https://github.com/cochranblock/any-gpu):** #5, #6, #16 — autograd needed before pixel-forge can train on AMD GPUs
- **[kova](https://github.com/cochranblock/kova):** NanoSign spec lives in kova docs. Plugin protocol ([src/plugin.rs](src/plugin.rs)) for GUI integration. C2 cluster orchestration for distributed training.
