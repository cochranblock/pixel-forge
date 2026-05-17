# Assumed Breach Threat Model

> **Operating assumption: every component below is already compromised. Design for damage containment and loud detection, not for prevention.**

This document is the canonical threat model for every project in the `cochranblock/*` portfolio. Each project adapts the Threat Surface section for its own context but shares the same first principles, mitigations, and verification protocol.

---

## First Principles

1. **Every record that matters has an external witness.** Hashes published to public git (or equivalent neutral timestamp authority) so tampering requires simultaneously corrupting your system AND the public chain.
2. **No single point of compromise.** Signing keys in hardware (YubiKey / TPM / Secure Enclave). Never in software. Never in env vars. Never in config files.
3. **Default air-gap.** No network dependency for correctness. Network is for backup + publishing hashes, both signed, both verifiable post-hoc.
4. **Append-only everything.** No delete path in any storage layer. Corrections are reversing entries referencing the original. Standard accounting discipline, enforced in code.
5. **Cryptographic audit chain.** Every day's state derives from the previous day's hash. Tampering with any day invalidates every subsequent day.
6. **Disclosure of methodology is a security feature.** If an auditor can independently verify the algorithm, they can independently verify the outputs. No "trust us" layers.
7. **Separation of duties enforced in software.** Entry, approval, and audit live in different trust zones. Compromise of one does not compromise the others.
8. **Redundancy across trust zones.** Local + different-cloud + different-format + offline. Attacker must compromise all to hide damage.
9. **Test breach scenarios regularly.** Triple Sims applied to tamper detection. If the chain does not detect a simulated tamper, the chain is broken.

---

## Threat Surface (pixel-forge)

**Records of consequence this project emits:**

- **Ghost Fabric packets** — signed sprites with ed25519 Proof-of-Authorship + GPS claims. These assert "I generated this sprite at this location at this time." Downstream use: copyright, first-possession claims, training-data provenance.
- **Trained model files (`.safetensors`)** — NanoSigned with BLAKE3. Distributed across the fleet (lf/gd/bt/st) via rsync. Silo architecture means each model file is a standalone authority for its class. Downstream use: generation reproducibility, model-integrity audits.
- **PoA signatures** — ed25519 signatures binding sprite-hash + class + GPS + timestamp + node identity. Verification is public via the node's public key (`pixel-forge node-key`).

**Threats specific to pixel-forge:**

| Assume | Attack | Impact |
|--------|--------|--------|
| Model weights tampered | Attacker edits `.safetensors` to inject steganographic watermarks, biased outputs, or backdoored patterns | Silent: every sprite generated after tampering carries hidden content. NanoSign BLAKE3 check on load catches this. |
| PoA signing key stolen | Attacker produces Ghost Fabric packets claiming authorship of sprites they didn't generate | Catastrophic for provenance claims. Hardware-key backing required. |
| Training data poisoned | Adversary inserts targeted sprites into `data_v3_32/<class>/` before training to bias the class toward specific outputs or inject copyright traps | Model learns the poison. Detected only by comparing generated distribution against expected class behavior. |
| Discriminator adversarial | Crafted inputs score high on discriminator but are garbage, corrupting the Forge quality gate | Forge emits low-quality sprites under legitimate signatures. |
| GPS spoofing in Ghost Fabric | Attacker claims sprite was generated at false location | Provenance claim is fraudulent but cryptographically valid. Current impl uses user-supplied lat/lon — trust boundary is weak. |
| Cluster node compromise | One of lf/gd/bt/st is backdoored, emits poisoned model weights that rsync propagates back | Master node receives and re-signs a poisoned model. NanoSign check at load helps, but if the compromised node also forged the signature, it slips through. Cross-node hash comparison required. |
| Silent model swap at inference | Attacker swaps `models/kart.safetensors` between generation runs | Next sprite is from a different model, but signed as legitimate. Load-time NanoSign + per-run model-hash embedded in Ghost Fabric packet closes this. |
| Supply chain (candle, any-gpu, image crates) | Upstream ML dep ships a version that subtly alters gradient computation or sampling | Generated distribution drifts undetectably across training runs. `cargo audit` + reproducible builds partially mitigate. |
| Physical seizure of training node | Attacker has full disk access to lf/gd/bt/st — reads training data, weights, unsigned checkpoints | Training data is CC0/CC-BY (public); model weights are NanoSigned but not encrypted at rest. Signing key is the sensitive asset. |

**N/A for pixel-forge:**

- **Daily audit hash chain** — pixel-forge does not emit a daily state record. Per-artifact Ghost Fabric packets are the audit unit; each one is its own external-witnessable event. The Public-Chain Deployment pattern below applies per-artifact, not per-day.
- **Separation of duties (entry/approval/audit)** — single-user creative tool. No approval workflow. The author is also the operator. The threat model here is "author-acts-against-themselves-later," addressed by immutable PoA timestamps.
- **Clock manipulation for financial timestamps** — sprite generation has creative/provenance timestamp consequence, not financial/legal audit consequence. Timestamp is informative, not load-bearing. NTP drift is acceptable.
- **DCAA compliance** — `govdocs/` contains federal-use-case references but pixel-forge is not currently on a federal contract deliverable path. If it moves there, re-audit this row.
- **Insider self-tampering on creative output** — creative tool by design allows regeneration, remixing, reinterpretation. PoA packets document specific invocations; prior outputs are not invalidated by later generations.

---

## Mitigations

| Assume | Mitigation | Verification |
|--------|-----------|--------------|
| Binary compromised | Hardware-key signatures for every output of consequence | Anyone can verify the public key matches expected fingerprint |
| Storage compromised | Append-only sled trees. Delete is not a function, not a policy. | Hash chain breaks on any rewrite. External witness detects. |
| Network MITM | Air-gap capable. Network used only for signed backups + hash publishing. | NTP + GitHub timestamp + hardware counter cross-checked. |
| Signing key stolen | Daily hash committed to public git. Stolen key cannot retroactively change committed days. | Any day older than the public commit is immutable in evidence. |
| Audit log tampered | Separate sled tree, write-only from main app. Auditor tool reads both + cross-checks. | Compromise of main app leaves audit log intact. |
| Backup tampered | 3 different targets with 3 different credentials (local USB + off-site cloud + paper). | Attacker needs all three to hide damage. |
| Insider / self-tampering | No admin role. No delete. Reversing entries only. | Legal record immune to author second-thoughts. |
| Clock manipulation | Multiple time sources: local clock, NTP, git commit timestamp, hardware-key counter. | Divergence flags exception requiring supervisor approval. |
| Supply chain (deps) | `cargo audit` in CI. Pinned SBOM. Reproducible builds where possible. | Anyone can reproduce the binary from source + lockfile. |
| Physical device seizure | Full-disk encryption. Hardware key physically separate from device. | Stolen laptop without key is useless for forgery. |

---

## Public-Chain Deployment

This project publishes tamper-evident hashes to a public companion repo: `cochranblock/<project>-chain` (where `<project>` is the project name).

- **Daily cycle:** at 23:59 local, compute BLAKE3 of all records-of-consequence from the day. Sign with hardware key. Commit to chain repo. Push.
- **GitHub timestamp** on the commit = neutral third-party witness. Anyone can cold-verify records were not rewritten after commit time.
- **Verification:** `<project> verify` reads the chain and re-derives hashes. Any divergence = tampering detected.

This pattern is a private Certificate Transparency log for project state. Same primitive Google uses for TLS certs, applied to whatever the project tracks.

---

## Triple Sims for Tamper Detection

Standard Triple Sims gate (run 3x identically) extended with a tamper-scenario sim:

1. Normal run → produce canonical output
2. Simulated tampering (flip one bit in storage) → `verify` must flag it
3. Simulated clock rewind → `verify` must flag it

If any sim fails to detect, the chain is broken. Fix before merge.

---

## Scope of this Document

- Covers: any artifact this project emits that has legal, financial, or audit consequence.
- Does NOT cover: source code itself (public under Unlicense, not sensitive), build outputs (reproducible), marketing content (public by design).
- If your project emits no records of consequence, the relevant sections are zero-length and the public-chain deployment is skipped. Document that explicitly.

---

## Relation to Other Docs

- **TIMELINE_OF_INVENTION.md** — establishes priority dates for contributions. Feeds into the chain's initial state.
- **PROOF_OF_ARTIFACTS.md** — cryptographic signatures on release artifacts. Adjacent pattern, same first principles.
- **DCAA_COMPLIANCE.md** (where applicable) — how this threat model satisfies FAR/DFARS audit requirements.

---

## Status

- [ ] Threat Surface section adapted for this project
- [ ] Hardware-key signing integrated or N/A documented
- [ ] Public-chain repo created and connected or N/A documented
- [ ] Triple Sims tamper-detection test present or N/A documented
- [ ] External verification procedure documented

---

*Unlicensed. Public domain. Fork, strip attribution, adapt, ship.*

*Canonical source: cochranblock.org/threat-model — last revision 2026-04-14*
<!-- COCHRANBLOCK-BRAND-FOOTER:START - generated by cochranblock/scripts/brand-stamp.sh -->

---

<sub>&#9656; **THE COCHRAN BLOCK, LLC** &#183; CAGE `1CQ66` &#183; UEI `W7X3HAQL9CF9` &#183; UNLICENSE &#183; [cochranblock.org](https://cochranblock.org)</sub>
<!-- COCHRANBLOCK-BRAND-FOOTER:END -->
