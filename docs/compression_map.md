<!-- Unlicense — cochranblock.org -->
<!-- Contributors: GotEmCoach, KOVA, Claude Opus 4.6 -->

# Pixel Forge Compression Map

Tokenization for traceability. Kova P13 compliant.

## Functions (fN)

| Token | Human name | Module |
|-------|------------|--------|
| f0 | preprocess | train |
| f1 | train | train |
| f2 | sample | train |
| f3 | sample_medium | train |
| f4 | sample_anvil | train |
| f5 | curate | curate |
| f6 | snap_to_grid | grid |
| f7 | pack_horizontal | sheet |
| f8 | pack_grid | sheet |
| f9 | auto_select | device_cap |
| f10 | reprobe | device_cap |
| f11 | auto_sample | device_cap |
| f12 | best_available | device_cap |
| f13 | probe_cluster | cluster |
| f14 | cluster_generate | cluster |
| f15 | deploy_to_node | cluster |
| f16 | deploy_all | cluster |
| f17 | sync_models | cluster |
| f18 | sync_models_all | cluster |
| f19 | train_combiner | combiner |
| f20 | generate_scene | combiner |
| f21 | load_combiner | combiner |
| f22 | save_combiner | combiner |
| f23 | train_experts | expert_train |
| f24 | route | expert |
| f25 | save_experts | expert |
| f26 | load_experts | expert |
| f27 | cascade_sample | moe |
| f28 | train_judge | judge |
| f29 | load_judge | judge |
| f30 | generate_seeded | scene |
| f31 | generate_bootstrap | scene |
| f32 | encode_batch | scene |
| f33 | run (app) | app |
| f34 | quality_gate | discriminator |
| f35 | generate_with_gate | discriminator |
| f36 | acquire (gpu) | gpu_lock |
| f37 | release (gpu) | gpu_lock |
| f38 | detect_class_count | train |
| f39 | load_palette | palette |
| f40 | quantize | palette |

## Types (tN)

| Token | Human name | Module |
|-------|------------|--------|
| t0 | TinyUNet | tiny_unet |
| t1 | MediumUNet | medium_unet |
| t2 | AnvilUNet | anvil_unet |
| t3 | PixelForgeApp | app |
| t4 | TrainConfig | train |
| t5 | PackedDataset | train |
| t6 | Tier | device_cap |
| t7 | DeviceProfile | device_cap |
| t8 | Backend | device_cap |
| t9 | ClusterState | cluster |
| t10 | NodeProfile | cluster |
| t11 | WorkUnit | cluster |
| t12 | SlotGridTransformer | combiner |
| t13 | CombinerTrainConfig | combiner |
| t14 | MicroClassifier | judge |
| t15 | ExpertHead | expert |
| t16 | ExpertSet | expert |
| t17 | ExpertStage | expert |
| t18 | SceneGrid | scene |
| t19 | Biome | scene |
| t20 | SwipeStore | swipe_store |
| t21 | Swipe | swipe_store |
| t22 | SwipeStats | swipe_store |
| t23 | Discriminator | discriminator |
| t24 | DiscriminatorTrainConfig | discriminator |

## Models (mN)

| Token | Human name | File | Params | Size |
|-------|------------|------|--------|------|
| m0 | Cinder | pixel-forge-cinder.safetensors | 1.09M | 4.2MB |
| m1 | Quench | pixel-forge-quench.safetensors | 5.83M | 22MB |
| m2 | Anvil | pixel-forge-anvil.safetensors | 16M | ~64MB |

## Modules (src/)

| Token | Module | Purpose |
|-------|--------|---------|
| M0 | tiny_unet | Cinder model — channels [32,64,64] |
| M1 | medium_unet | Quench model — channels [64,128,128], self-attention |
| M2 | anvil_unet | Anvil model — channels [96,192,192], XL |
| M3 | train | Training loop, sampling, data pipeline |
| M4 | app | egui GUI — mobile + desktop |
| M5 | device_cap | Device detection, tier selection, benchmarks |
| M6 | cluster | Distributed generation across SSH nodes |
| M7 | combiner | SlotGridTransformer — scene composition |
| M8 | judge | MicroClassifier — quality scoring |
| M9 | expert | MoE expert heads — shape/color/detail/class |
| M10 | expert_train | Expert training on frozen Quench |
| M11 | moe | Cascade pipeline — Cinder → Quench + Experts |
| M12 | scene | 8x8 SceneGrid, biome generation |
| M13 | swipe_store | Tinder-style swipe data, ring buffer |
| M14 | lora | Rank-4 LoRA adapters for TinyUNet |
| M15 | discriminator | Quality gate — binary classifier |
| M16 | palette | Color palettes + quantization |
| M17 | grid | Snap to grid, tile alignment |
| M18 | sheet | Sprite sheet packing |
| M19 | curate | Data curation pipeline |
| M20 | gpu_lock | File-based GPU lock for training |
| M21 | plugin | Kova plugin interface |
| M22 | poa | Proof of Authorship signing |
| M23 | pipeline | SD pipeline (optional, desktop) |

## CLI Commands (cN)

| Token | Command | Description |
|-------|---------|-------------|
| c0 | (none) | Launch GUI app |
| c1 | sprite | Generate sprite from text |
| c2 | tileset | Generate seamless tileset |
| c3 | train | Train a model |
| c4 | generate | Generate from trained model |
| c5 | curate | Curate training data |
| c6 | swipe | Record swipe feedback |
| c7 | train-judge | Train quality judge |
| c8 | judge | Score a sprite |
| c9 | scene | Generate a scene |
| c10 | train-combiner | Train scene combiner |
| c11 | train-lora | Train LoRA adapters |
| c12 | pipeline | Run SD pipeline |
| c13 | cascade | MoE cascade generation |
| c14 | train-experts | Train expert heads |
| c15 | auto | Auto-detect and generate |
| c16 | probe | Probe device capabilities |
| c17 | cluster-deploy | Deploy to cluster nodes |
| c18 | cluster-sync | Sync models to cluster |

## Class IDs

| ID | Name |
|----|------|
| 0 | character |
| 1 | weapon |
| 2 | potion |
| 3 | terrain |
| 4 | enemy |
| 5 | tree |
| 6 | building |
| 7 | animal |
| 8 | effect |
| 9 | food |
| 10 | armor |
| 11 | tool |
| 12 | vehicle |
| 13 | ui |
| 14 | misc |
| 15 | (null/CFG) |

## Palette IDs

| ID | Name |
|----|------|
| 0 | stardew |
| 1 | starbound |
| 2 | snes |
| 3 | nes |
| 4 | gameboy |
| 5 | pico8 |
| 6 | endesga |
