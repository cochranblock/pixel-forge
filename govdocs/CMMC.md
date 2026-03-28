# CMMC Level 1-2 Mapping

**Project:** pixel-forge v0.6.0
**Framework:** Cybersecurity Maturity Model Certification (CMMC 2.0)
**Date:** 2026-03-27

## Overview

pixel-forge is a standalone, offline application with no network access, no user accounts, and no data storage beyond local files. Most CMMC domains have limited applicability. This mapping documents what applies and what does not.

## AC — Access Control

**Applicability:** Minimal

- **AC.L1-3.1.1 (Authorized access):** N/A. No user accounts. No authentication. The application runs with the invoking user's OS-level permissions.
- **AC.L1-3.1.2 (Transaction control):** N/A. No transactions. No network access.
- **AC.L1-3.1.20 (External connections):** N/A. Zero external connections during operation.
- **File permissions:** Ed25519 node key at `~/.pixel-forge/node.key` is created with `0o600` (owner read/write only). See `src/poa.rs:159`.

## AU — Audit and Accountability

**Applicability:** Build and development process

| Control | Implementation |
|---|---|
| AU.L2-3.3.1 (System audit) | Git history records every source change with author and timestamp |
| AU.L2-3.3.2 (Audit actions) | `cargo clippy -D warnings` as quality gate — all warnings treated as build failures |
| Build provenance | Release profile is deterministic: `lto = true`, `codegen-units = 1`, `strip = true` |
| Dependency audit | `cargo audit` checks dependency tree against RustSec advisory database |

## AT — Awareness and Training

**Applicability:** N/A — standalone tool, no organizational training requirements.

## CM — Configuration Management

**Applicability:** Build configuration

| Control | Implementation |
|---|---|
| CM.L2-3.4.1 (Baseline configuration) | `Cargo.lock` pins all dependency versions. `Cargo.toml` defines build profiles. |
| CM.L2-3.4.2 (Security settings) | Release build: LTO, strip symbols, abort on panic, size-tuned optimization |
| CM.L2-3.4.5 (Access restrictions) | Source control via git. Signed commits. Branch protection on GitHub. |

## IA — Identification and Authentication

**Applicability:** N/A — no user accounts, no authentication mechanisms.

## IR — Incident Response

**Applicability:** Limited

- Vulnerability reports accepted via GitHub issues
- `cargo audit` for automated CVE detection in dependencies
- No running service to produce security incidents

## MA — Maintenance

**Applicability:** Software updates

- Dependency updates via `cargo update` with lock file regeneration
- Rust toolchain updates via `rustup update`
- No runtime maintenance — stateless application

## MP — Media Protection

**Applicability:** N/A — no removable media handling, no data classification requirements.

## PE — Physical and Environmental Protection

**Applicability:** N/A — software application, no physical infrastructure.

## PS — Personnel Security

**Applicability:** N/A — open source project, no personnel clearance requirements.

## SA — Security Assessment

**Applicability:** Development process

| Control | Implementation |
|---|---|
| SA.L2-3.12.1 (Assess controls) | `cargo clippy -D warnings` — static analysis with zero tolerance |
| SA.L2-3.12.3 (Monitor controls) | `cargo audit` — dependency vulnerability scanning |
| Code review | All changes tracked in git with descriptive commit messages |
| Testing | Quality gate: compilation, unit tests, integration tests, exit code verification |

## SC — System and Communications Protection

**Applicability:** Directly relevant

| Control | Implementation |
|---|---|
| SC.L1-3.13.1 (Boundary protection) | Zero network communications. No ports open. No listeners. |
| SC.L1-3.13.5 (Public access) | No public-facing component. Desktop/mobile app only. |
| SC.L2-3.13.6 (Network deny-by-default) | The application has no network stack in use during operation. |

## SI — System and Information Integrity

**Applicability:** Directly relevant

| Control | Implementation |
|---|---|
| SI.L1-3.14.1 (Flaw remediation) | `cargo audit` for CVE detection. `cargo update` for patching. |
| SI.L1-3.14.2 (Malicious code protection) | No dynamic code loading. No eval. No plugin execution. Static binary. |
| SI.L2-3.14.3 (Security alerts) | GitHub Dependabot + `cargo audit` for dependency advisories |
| Build integrity | `Cargo.lock` pinning. Reproducible builds. Strip + LTO. |
