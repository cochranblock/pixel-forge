# FedRAMP Applicability Notes

**Project:** pixel-forge v0.6.0
**Date:** 2026-03-27

## FedRAMP Does Not Apply

pixel-forge is a standalone desktop and mobile application. It is **not** a cloud service, SaaS product, or hosted platform.

## Why FedRAMP Is Inapplicable

| FedRAMP Requirement | pixel-forge Reality |
|---|---|
| Cloud service provider | Standalone binary on user's device |
| Authorization boundary | No boundary — runs entirely on host device |
| Multi-tenant infrastructure | Single-user application |
| Data residency | All data on local filesystem |
| Network-accessible service | Zero network listeners, zero ports open |
| Continuous monitoring | No running service to monitor |

## Deployment in Federal Environments

When pixel-forge is installed on a federal workstation or GFE (Government Furnished Equipment):

- The application falls under the **host system's ATO** (Authority to Operate)
- The host system's ISSO/ISSM is responsible for approving software installation
- pixel-forge's security posture (see `SECURITY.md`) supports ATO documentation:
  - Zero network access
  - No data collection
  - No listening ports
  - No privileged operations required
  - No external dependencies at runtime

## For Procurement Officers

- **License:** Unlicense (public domain). No procurement restrictions. No license fees.
- **Source code:** Publicly available for security review.
- **Air-gap safe:** Fully operational with zero network connectivity.
- **Build from source:** Federal agencies can build from audited source rather than trusting pre-built binaries.
- **No vendor lock-in:** Public domain code can be forked, modified, and maintained by any party.
