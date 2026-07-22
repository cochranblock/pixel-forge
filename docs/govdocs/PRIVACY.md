# Privacy Impact Assessment

**Project:** pixel-forge v0.6.0
**Date:** 2026-03-27

## Data Collection Summary

**pixel-forge collects zero data.** It makes zero network calls. It has zero tracking.

## Detailed Assessment

### Personally Identifiable Information (PII)

- No PII is collected, processed, or stored
- No user accounts, no registration, no email addresses
- No names, no device IDs, no IP addresses

### Network Activity

- Zero outbound network connections during normal operation
- Zero inbound connections (no server component, no listening ports)
- Zero DNS queries
- Zero telemetry, zero analytics, zero crash reporting
- The optional `sd-pipeline` feature can download models from Hugging Face on first use. This is a one-time download, disabled on mobile builds, and transmits no user data.

### Local Data Storage

| Data | Format | Location | Transmitted? |
|---|---|---|---|
| Generated sprites | PNG | User-chosen directory | Never |
| Swipe ratings (good/bad) | bincode (binary) | `swipe_data/` | Never |
| Model weights | safetensors | Project directory | Never |
| Training data cache | bincode + zstd | `data/` | Never |
| Ed25519 node key | Raw 32 bytes | `~/.pixel-forge/node.key` | Never |

### Third-Party Services

None. pixel-forge does not contact any third-party service during operation.

## Regulatory Compliance

### GDPR (EU General Data Protection Regulation)

- **Status:** Not applicable. No personal data is processed.
- No data controller role exists — the application processes no personal data.
- No data processor role exists — no data is shared with any party.
- Right to erasure: users delete their own local files. No remote data to erase.

### CCPA (California Consumer Privacy Act)

- **Status:** Not applicable. No consumer data is collected or sold.
- No data sale. No data sharing. No data brokering.

### COPPA (Children's Online Privacy Protection Act)

- **Status:** Not applicable. No data collection from any user, including children.

### HIPAA

- **Status:** Not applicable. No health information is collected or stored.

## Published Privacy Policy

The Android Play Store listing references a privacy policy at `assets/store/privacy-policy.html`. Its contents match this document: zero data collection, zero network calls, zero tracking.

## Contact

Privacy inquiries: directed to project maintainer via GitHub repository.
