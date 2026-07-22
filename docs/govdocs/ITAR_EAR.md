# Export Control Assessment (ITAR / EAR)

**Project:** pixel-forge v0.6.0
**Date:** 2026-03-27

## Summary

pixel-forge is an open-source pixel art generator. It contains no defense-specific technology and its cryptographic functions are ancillary (signing only). It is not ITAR-controlled and qualifies for EAR mass market or publicly available exceptions.

## ITAR (International Traffic in Arms Regulations)

**Status: Not controlled**

- pixel-forge is not on the United States Munitions List (USML)
- Contains no defense articles or defense services
- Generates pixel art game assets — not military or intelligence technology
- No dual-use military application beyond the general-purpose use cases described in `FEDERAL_USE_CASES.md`

## EAR (Export Administration Regulations)

### ECCN Classification

**EAR Category 5, Part 2 (Information Security)** is the relevant category for software containing cryptography.

### Cryptographic Functions

| Algorithm | Type | Purpose | Key Size |
|---|---|---|---|
| Ed25519 | Digital signature | Sign Proof-of-Artifacts packets | 256-bit |
| SHA-256 | Hash | Fingerprint sprite pixel data | N/A |

### Ancillary Crypto Determination

The crypto in pixel-forge qualifies as **ancillary** (not the primary function) under EAR 740.17(b)(3):

1. **Primary function:** Generating pixel art sprites using neural network inference
2. **Crypto function:** Signing generated sprites for provenance tracking (PoA)
3. **No encryption:** Ed25519 is a signing scheme. No data confidentiality protection exists.
4. **No key exchange:** No Diffie-Hellman, no RSA, no TLS.
5. **Crypto cannot be easily repurposed:** The signing function is tightly bound to the 89-byte GhostPacket format in `src/poa.rs`.

### Applicable Exceptions

**EAR 740.17 (Encryption Commodities, Software, and Technology — Mass Market)**

- The software is publicly available under the Unlicense
- Source code is published on GitHub
- No access restrictions on the source code
- Crypto functionality is limited to signing (not encryption)

**EAR 742.15(b) — Publicly Available Source Code**

- Source code for encryption items that is publicly available is not subject to EAR when:
  - The source code is publicly accessible (GitHub, Unlicense)
  - No further obligations on distribution
  - Notification to BIS required (TSU exception, EAR 740.13(e))

### BIS Notification

For EAR compliance, a TSU (Technology and Software Unrestricted) notification should be filed with:

- Bureau of Industry and Security (BIS): `crypt@bis.doc.gov`
- ENC Encryption Request Coordinator: `enc@nsa.gov`

Notification should include: project name, URL, description of crypto functionality (Ed25519 signing, SHA-256 hashing), and confirmation that source code is publicly available.

## No Controlled Technical Data

- No classified information
- No controlled unclassified information (CUI)
- Training data is CC0/CC-BY licensed pixel art (see `data/SOURCES.md`)
- Model architectures are original work, not derived from controlled research
