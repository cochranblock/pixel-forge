# NIST SP 800-218 (SSDF) Mapping

**Project:** pixel-forge v0.6.0
**Framework:** Secure Software Development Framework (SSDF) v1.1
**Date:** 2026-03-27

## PS: Prepare the Organization

### PS.1 — Define Security Requirements

- **License:** Unlicense (public domain). No proprietary restrictions on audit or inspection.
- **Secrets policy:** No secrets in source code. Ed25519 node keys generated at runtime and stored at `~/.pixel-forge/node.key` with `0o600` permissions (see `src/poa.rs:159`).
- **Gitignore:** `.gitignore` excludes keystores (`*.keystore`, `*.jks`), build artifacts, and model weights.
- **Dependency policy:** All deps from crates.io. Pinned via `Cargo.lock`. No vendored binaries.

### PS.2 — Roles and Responsibilities

- Single maintainer (GotEmCoach) with full ownership of source, build, and release.
- AI contributors (KOVA, Claude Opus 4.6) documented in file headers per project convention.

## PW: Produce Well-Secured Software

### PW.1 — Design Software to Meet Security Requirements

- **No network calls:** Generation runs entirely on-device. No HTTP, no DNS, no sockets during inference.
- **No user accounts:** No authentication, no passwords, no session tokens.
- **No server component:** Desktop/mobile application only. No listening ports.
- **Local-only data:** Sprites, swipe data, and model weights stored on local filesystem only.

### PW.2 — Review the Software Design

- Architecture documented in `CLAUDE.md` with module table and file mapping.
- Model tiers (Cinder/Quench/Anvil) documented with parameter counts and file sizes.

### PW.4 — Reuse Existing Well-Secured Software

- ML framework: candle (Hugging Face, Apache-2.0) — pure Rust, memory-safe tensor ops.
- Crypto: ed25519-dalek (BSD-3-Clause/Apache-2.0) — widely audited Ed25519 implementation.
- Hashing: sha2 (RustCrypto project) — standard SHA-256 implementation.
- GUI: egui/eframe — no JavaScript, no WebView, no embedded browser.

### PW.5 — Reuse of Components

- All dependencies from crates.io (Rust package registry).
- `Cargo.lock` checked into version control, pinning every transitive dependency.
- No git dependencies. No path dependencies. No private registries.

### PW.6 — Build Environment

Release profile in `Cargo.toml`:
```toml
[profile.release]
opt-level = 'z'       # Size-focused
lto = true            # Link-time inlining across all crates
codegen-units = 1     # Deterministic codegen
panic = 'abort'       # No unwinding — smaller binary
strip = true          # Strip debug symbols from release
```

- Deterministic builds: `codegen-units = 1` + `lto = true` produce consistent output.
- Android: built with `cargo-ndk` targeting `aarch64-linux-android`, signed with upload keystore.

### PW.7 — Code Review

- All source in a single crate (`src/*.rs`). ~11,000 LOC.
- Git history tracks every change with descriptive commit messages.

## RV: Respond to Vulnerabilities

### RV.1 — Vulnerability Identification

- `cargo audit` checks for known CVEs in dependency tree.
- `cargo clippy -D warnings` runs as a quality gate — treats all warnings as errors.
- No unsafe code blocks in project source (candle handles GPU dispatch internally).

### RV.2 — Vulnerability Remediation

- Dependency updates via `cargo update` with lock file regeneration.
- Critical CVEs: patch or pin to fixed version, rebuild, and redeploy.

## PO: Protect the Organization

### PO.1 — Protect Release Integrity

- Android APK signed with upload keystore (not checked into source).
- Google Play Store delivery with Play Signing (Google holds release key).
- Desktop releases built from tagged commits. Binary stripped of debug info.
- Source available on GitHub for independent verification.
