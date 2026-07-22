# Supply Chain Integrity

**Project:** pixel-forge v0.6.0
**Date:** 2026-03-27

## Dependency Sources

All dependencies come from [crates.io](https://crates.io), the official Rust package registry. No dependencies are sourced from:

- Git repositories
- Private registries
- Local path references
- Vendored source archives

## Version Pinning

`Cargo.lock` is checked into version control and pins every direct and transitive dependency to an exact version. The lock file is the single source of truth for reproducible builds.

To verify: `cargo build --locked` will fail if `Cargo.lock` is out of sync with `Cargo.toml`.

## No Vendored Binaries

The repository contains zero pre-built binaries, shared libraries, or object files. Every native artifact is compiled from Rust source during `cargo build`.

**Exception:** Android builds produce a JNI shared library (`libpixel_forge.so`) via `cargo-ndk`. This is a build output, not a vendored input.

## No Pre-built Model Weights in Source

Model weights (`.safetensors` files) are training outputs stored outside the repository. The training pipeline (`cargo run --release -- train`) produces weights from the curated dataset in `data/`.

## Build from Source

```bash
# Desktop (macOS with Metal)
cargo build --release -p pixel-forge

# Desktop (NVIDIA GPU)
cargo build --release -p pixel-forge --features cuda --no-default-features

# Desktop (CPU only)
cargo build --release -p pixel-forge --no-default-features

# Android (ARM64)
cargo ndk -t arm64-v8a build --release --no-default-features
```

## Audit

```bash
# Check for known CVEs in dependency tree
cargo audit

# Lint with all warnings as errors
cargo clippy -p pixel-forge -- -D warnings
```

## Source Availability

Source code is available on GitHub under the Unlicense (public domain). Any party can:

1. Inspect every line of code
2. Audit the dependency tree via `Cargo.lock`
3. Build from source and compare output
4. Fork and modify without restriction

## Reproducible Builds

The release profile enforces deterministic compilation:

- `lto = true` — full link-time inlining
- `codegen-units = 1` — single codegen unit eliminates nondeterministic parallelism
- `strip = true` — removes debug symbols that vary by build environment
- `panic = 'abort'` — no unwinding tables

Same source + same Rust toolchain + same target = same binary.
