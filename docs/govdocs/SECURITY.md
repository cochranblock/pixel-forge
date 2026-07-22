# Security Posture

**Project:** pixel-forge v0.6.0
**Date:** 2026-03-27

## Cryptographic Primitives

All crypto lives in `src/poa.rs` (Proof of Artifacts module). No other module performs cryptographic operations.

| Primitive | Library | Version | Use |
|---|---|---|---|
| Ed25519 | ed25519-dalek | 2.2.3 | Signing GhostPacket payloads (89-byte message) |
| SHA-256 | sha2 | 0.10.9 | Hashing sprite pixel data (32x32x3 = 3072 bytes RGB) |

- **No encryption.** Ed25519 is a signing scheme, not an encryption algorithm. Data is never encrypted at rest or in transit.
- **No AES, no Argon2, no HKDF.** These do not exist in this project.
- **Key storage:** Ed25519 signing key stored at `~/.pixel-forge/node.key` (32 bytes, `0o600` permissions on Unix). Generated on first run via `rand::thread_rng()`.

## No Network Access

pixel-forge makes zero network calls during generation. There are no:

- HTTP clients
- DNS lookups
- Socket connections
- WebSocket listeners
- Telemetry endpoints

The optional `sd-pipeline` feature uses `hf-hub` for one-time model downloads from Hugging Face. This feature is disabled on mobile builds and is not active during generation.

## No Secrets in Source

- No API keys, tokens, or passwords in the repository
- `.gitignore` excludes `*.keystore`, `*.jks`, and Android signing files
- Ed25519 keys generated at runtime, stored in user home directory

## Input Validation

| Input | Validation | Location |
|---|---|---|
| Class names | Checked against known class list before generation | `src/class_cond.rs` |
| Palette names | Matched against built-in palette enum | `src/palette.rs` |
| Model files | safetensors header validation on load | candle-core internals |
| Image data | PNG header + CRC validation | image crate internals |
| Plugin JSON | serde_json deserialization with typed structs | `src/plugin.rs` |
| CLI arguments | clap derive validation with type constraints | `src/main.rs` |

## Attack Surface

| Surface | Risk | Mitigation |
|---|---|---|
| Model loading (.safetensors) | Malformed header could cause panic | safetensors format includes header length validation; candle validates tensor shapes |
| PNG parsing | Malformed PNG could cause panic/OOM | image crate validates chunk types and CRC checksums |
| GUI input (egui) | Text input overflow | egui handles text input with bounded buffers |
| Plugin protocol (stdin JSON) | Malformed JSON | serde_json rejects invalid input; typed deserialization prevents unexpected fields |
| Training data (bincode+zstd) | Corrupted cache | zstd frame validation; bincode size-bounded deserialization |

## No Authentication

- No user accounts
- No passwords
- No session tokens
- No OAuth flows
- No database

## No Server Component

pixel-forge is a standalone desktop/mobile application. It does not:

- Listen on any port
- Serve HTTP/HTTPS
- Accept remote connections
- Expose any API (except local stdin/stdout plugin protocol)
