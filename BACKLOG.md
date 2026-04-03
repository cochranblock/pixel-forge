# Backlog

Prioritized. Most important at top. Max 20 items. Self-reorganizes based on recency and relevance.

Last updated: 2026-04-03. Anvil v7 training restarted on lf (bs=4, no EMA, lr=5e-5). Tests added: 32 passing. P23 triple lens run — top 3 replaced with synthesis recommendations.

---

1. `[fix]` **Re-enable skeleton seeding for hybrid class system** — Skeleton warm-start (70% structure + 30% noise) was disabled when hybrid conditioning shipped (`train.rs:1036-1037`). Every sample now starts from pure noise — this is a quality regression for all structured classes. Without it, the CFG fix can't be fairly evaluated: remaining blob artifacts may be skeleton-absence artifacts, not CFG artifacts. Fix: recompute per-class average skeletons keyed by `(super_id, class_name)` instead of legacy integer class IDs. New skeleton dir: `skeletons_v2/`. Touches: `train.rs:sample_seeded_cfg`, `moe.rs:cascade_sample`.
2. `[fix]` **Three silent bugs: medium filename (data loss), plugin endesga, cascade seed** — (a) `train --medium` defaults output to `pixel-forge-cinder.safetensors` — silently overwrites Cinder on a 200-epoch Quench train run. Fix: detect `medium`/`anvil` flags, default to `pixel-forge-quench.safetensors`/`pixel-forge-anvil.safetensors`. (b) `plugin.rs:50` lists `"endesga"` but `palette.rs` requires `"endesga32"` — kova GUI palette broken. Fix: `"endesga"` → `"endesga32"`. (c) `cascade` subcommand has no `--seed` flag — non-reproducible. Add `seed: Option<u64>` to `Cmd::Cascade`, thread to `cascade_sample`. All three: `main.rs`, `plugin.rs`, `moe.rs`.
3. `[fix]` **Clean git history + harden cluster.rs** — `.git/` is 1GB (safetensors blobs in history). Every clone on every worker node pays this; every cluster-deploy is slow. Run: `git filter-repo --strip-blobs-bigger-than 1M`. Coordinate with lf/gd clones. Simultaneously: make `REMOTE_BIN`/`REMOTE_MODELS_DIR` in `cluster.rs:31-34` configurable via env (`PIXEL_FORGE_REMOTE_BIN`) or `~/.pixel-forge/cluster.toml` — currently hardcoded to `/home/mcochran`, breaking anyone else's deployment and creating a writable-binary RCE vector if a node is compromised.
4. `[feature]` Tiered pipeline: palette specialist — 50K param model picks 8-16 colors per class. New module. Trains in seconds.
5. `[research]` Add autograd to [any-gpu](https://github.com/cochranblock/any-gpu) — backward ops for conv2d_weight, group_norm, swish, attention, upsample. Plus AdamW optimizer. ~1500 lines WGSL. Unblocks GPU training on bt's 5700 XT.
6. `[feature]` Add `--device vulkan` flag — wire any-gpu's `GpuDevice` into `pipeline::best_device()` as fourth backend option. Depends on: any-gpu autograd (#5). [src/pipeline.rs:25](src/pipeline.rs#L25).
7. `[feature]` Add self-attention at 16×16 resolution in Anvil — currently only at 8×8 and 4×4. Adds ~0.4M params, requires retrain. [src/anvil_unet.rs:17](src/anvil_unet.rs#L17).
8. `[feature]` DDPM stochastic sampler — add `--sampler ddpm|ddim` flag. DDPM injects noise each step, often produces crisper edges at cost of more steps. [src/train.rs](src/train.rs), [src/moe.rs](src/moe.rs).
9. `[docs]` Generate hero image for README — after quality fixes land, cherry-pick 16 best outputs across classes, create 4×4 grid at 4× upscale. Save as `assets/hero-grid.png`.
10. `[build]` Publish GitHub Releases — build platform binaries (macOS ARM/Intel, Linux x86) and upload as release artifacts. README already references the Releases page.
11. `[fix]` `.git/` is 1GB — large blobs leaked into history. Run `git filter-repo` to clean safetensors from history. Coordinate with any clones.
12. `[feature]` Sign existing model files — run NanoSign on all `.safetensors` in project root and on worker nodes (lf, gd). Currently only new saves are signed.
13. ~~`[test]` Verify NanoSign tamper detection~~ — done `ee7550e82`. 8 unit tests cover sign/verify/tamper/strip/unsigned-compat/verify_or_bail.
14. `[research]` Noise schedule bias toward mid-range — oversample t∈[0.3,0.7] where detail forms. Beta distribution or stratified sampling. [src/train.rs:596](src/train.rs#L596).
15. `[docs]` Update `docs/compression_map.md` — add nanosign functions (f48+), NanoSign types, sample_seeded_cfg/sample_medium_seeded_cfg. Verify existing mappings still match code.
16. `[research]` Distributed training across lf (CPU) + bt (Vulkan via any-gpu) — different models or pipeline tiers per node. Depends on: any-gpu autograd (#5), `--device vulkan` (#6). Uses [src/cluster.rs](src/cluster.rs).

---

Done this session:
- ~~`[test]` Unit + integration tests~~ — 32 tests passing. nanosign (sign/verify/tamper/strip/backward-compat), palette (quantize, transparency, nearest-color, all named palettes), class_cond (super-cat routing, tag correctness, unknown fallback), train::cosine_schedule (bounds, monotonicity, unit range).
- ~~`[feature]` Add `--cfg <f64>` CLI flag~~ — `a6fa027cb`. Generate/anvil/cascade all accept `--cfg`.
- ~~`[feature]` Add brightness augmentation~~ — `de106aaf8`. 50% chance ×[0.8,1.2], clamp [0,1]. Targets dark-blob tendency.
- ~~`[feature]` Auto-apply palette quantization~~ — already done in all generation commands.
- ~~`[build]` Rebuild on lf with NanoSign~~ — built, v7 restarted with bs=4.
- ~~`[fix]` EMA OOM on 16GB nodes~~ — workaround: bs=4 no-ema uses 1.1GB RAM. EMA needs >16GB with Anvil.

Dependencies on other cochranblock projects:
- **[any-gpu](https://github.com/cochranblock/any-gpu):** #5, #6, #16 — autograd needed before pixel-forge can train on AMD GPUs
- **[kova](https://github.com/cochranblock/kova):** NanoSign spec lives in kova docs. Plugin protocol ([src/plugin.rs](src/plugin.rs)) for GUI integration. C2 cluster orchestration for distributed training.
