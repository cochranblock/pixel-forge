# Software Bill of Materials (SBOM)

**Project:** pixel-forge v0.6.0
**Format:** Markdown (EO 14028 compliant)
**Generated:** 2026-03-27
**License:** Unlicense (public domain)
**Build system:** Cargo (Rust)
**Lock file:** Cargo.lock (pinned, checked in)

## Direct Dependencies

| Dependency | Version | License | Purpose |
|---|---|---|---|
| candle-core | 0.8.4 | Apache-2.0 | Tensor operations, GPU dispatch (Metal/CUDA) |
| candle-nn | 0.8.4 | Apache-2.0 | Neural network layers (Conv2d, GroupNorm, Linear) |
| candle-transformers | 0.8.4 | Apache-2.0 | Stable Diffusion pipeline components (optional: `sd-pipeline`) |
| clap | 4.6.0 | MIT/Apache-2.0 | CLI argument parsing with derive macros |
| serde | 1.0.228 | MIT/Apache-2.0 | Serialization framework for config and data structs |
| serde_json | 1.0.149 | MIT/Apache-2.0 | JSON parsing for plugin protocol and class definitions |
| anyhow | 3.0.11 | MIT/Apache-2.0 | Error handling with context propagation |
| image | 0.25.10 | MIT/Apache-2.0 | PNG encode/decode, RGBA pixel buffer operations |
| rand | 0.8.5 | MIT/Apache-2.0 | Random number generation for diffusion noise sampling |
| bincode | 1.8.3 | MIT | Binary serialization for training data cache |
| zstd | 0.13.3 | MIT | Zstandard compression for cached training datasets |
| eframe | 0.33.3 | MIT/Apache-2.0 | Desktop/mobile GUI application framework |
| egui | 0.33.3 | MIT/Apache-2.0 | Immediate-mode GUI widgets and layout |
| base64 | 0.22.1 | MIT/Apache-2.0 | Base64 encoding for plugin protocol sprite data |
| ed25519-dalek | 2.2.3 | BSD-3-Clause/Apache-2.0 | Ed25519 signing for Proof-of-Artifacts packets |
| sha2 | 0.10.9 | MIT/Apache-2.0 | SHA-256 hashing for sprite content fingerprinting |
| half | 0.4.13 | MIT/Apache-2.0 | f16 type for dtype-aware model inference |
| hf-hub | 0.2.1 | Apache-2.0 | Hugging Face model downloads (optional: `sd-pipeline`) |
| tokenizers | 0.21.4 | Apache-2.0 | Text tokenization for SD prompts (optional: `sd-pipeline`) |

## Optional Feature Gates

| Feature | Dependencies Activated | Default | Notes |
|---|---|---|---|
| `metal` | candle-core/metal, candle-nn/metal | Yes | Apple Silicon GPU acceleration |
| `cuda` | candle-core/cuda, candle-nn/cuda | No | NVIDIA GPU acceleration |
| `sd-pipeline` | hf-hub, tokenizers, candle-transformers | Yes (desktop) | Disabled on mobile/Android builds |

## Supply Chain Notes

- All dependencies sourced from crates.io
- No vendored binaries
- No pre-built shared libraries (except Android JNI output from cargo-ndk)
- Cargo.lock pins all transitive dependency versions
- Source available on GitHub under Unlicense
