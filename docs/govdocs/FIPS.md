# FIPS 140-2/3 Status

**Project:** pixel-forge v0.6.0
**Date:** 2026-03-27

## Current Status: NOT FIPS Validated

pixel-forge is **not** FIPS 140-2 or FIPS 140-3 validated. None of its cryptographic modules have undergone CMVP (Cryptographic Module Validation Program) testing.

## Cryptographic Inventory

| Primitive | Library | Implementation | FIPS Status |
|---|---|---|---|
| Ed25519 | ed25519-dalek 2.2.3 | Pure Rust, curve25519-dalek backend | Not validated |
| SHA-256 | sha2 0.10.9 | Pure Rust, RustCrypto project | Not validated |

Both libraries are pure Rust implementations with no FFI calls to OpenSSL or any validated module.

## Crypto Usage Context

Crypto in pixel-forge is used **only** for Proof-of-Artifacts (PoA) packet signing in `src/poa.rs`. This is a provenance mechanism, not a security boundary.

- **Ed25519:** Signs a 89-byte payload containing sprite hash, class ID, quality score, GPS coordinates, timestamp, and public key. Produces a 153-byte GhostPacket for LoRa transmission.
- **SHA-256:** Hashes the RGB pixel data (32x32x3 = 3072 bytes) of a generated sprite to produce a content fingerprint.

### What crypto is NOT used for

- No encryption of data at rest
- No encryption of data in transit
- No key exchange
- No TLS/SSL
- No password hashing
- No session tokens

## Risk Assessment

The absence of FIPS validation is low-risk because:

1. **No security boundary depends on the crypto.** PoA signing is a provenance feature (proof that a specific node generated a specific sprite), not a confidentiality or access control mechanism.
2. **No sensitive data is encrypted.** Generated pixel art is not classified or controlled information.
3. **No network transmission of secrets.** The application makes zero network calls during operation.

## Path to FIPS Compliance

If FIPS validation is required for federal deployment:

1. **Replace ed25519-dalek** with `aws-lc-rs` (AWS LibCrypto, FIPS 140-3 validated, certificate #4631).
2. **Replace sha2** with the SHA-256 implementation in `aws-lc-rs`.
3. **Verify:** `aws-lc-rs` provides Ed25519 and SHA-256 behind a FIPS-validated boundary.
4. **API impact:** Minimal. `src/poa.rs` is the only file that imports crypto. The signing and hashing interfaces are similar.
5. **Build impact:** `aws-lc-rs` requires CMake and a C compiler. Increases build time. Not available on all targets (may affect Android ARM64 builds).
