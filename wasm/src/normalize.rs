// Unlicense — cochranblock.org
//! Z-score normalization for the wasm runtime.
//!
//! Reuses the desktop module verbatim via `#[path]` — single source of
//! truth for the on-disk JSON format, the field names, and the pixel-
//! level forward/inverse math. The desktop crate keeps its candle-flavored
//! tensor adapters in `train.rs`; the wasm crate stays on CPU pixel ops
//! since readback already lands the buffer in CPU memory.

// Filesystem helpers (load/save_sidecar) and the unused forward path
// (to_z_pixels) compile but are dead in this crate; suppress the noise.
#[allow(dead_code)]
#[path = "../../src/normalize.rs"]
pub mod inner;

pub use inner::Normalizer;
