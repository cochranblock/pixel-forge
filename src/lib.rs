// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! pixel-forge library — exposes the app and core modules for mobile builds.
#![allow(dead_code)]
#![allow(clippy::manual_is_multiple_of)]

pub mod class_cond;
pub mod palette;
pub mod grid;
pub mod sheet;
#[cfg(feature = "sd-pipeline")]
pub mod pipeline;
#[cfg(not(feature = "sd-pipeline"))]
pub mod pipeline_stub;
#[cfg(not(feature = "sd-pipeline"))]
pub use pipeline_stub as pipeline;
pub mod tiny_unet;
pub mod train;
pub mod curate;
pub mod judge;
pub mod swipe_store;
pub mod scene;
pub mod combiner;
pub mod lora;
pub mod medium_unet;
pub mod anvil_unet;
pub mod expert;
pub mod expert_train;
pub mod moe;
pub mod device_cap;
pub mod cluster;
pub mod discriminator;
pub mod plugin;
pub mod app;
pub mod gpu_lock;
pub mod poa;
pub mod quantize;
pub mod relight;
pub mod nanosign;

/// Recursively collect all .png files under a directory.
pub fn walk_pngs(dir: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    fn recurse(path: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(path) else { return };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                recurse(&p, out);
            } else if p.extension().and_then(|e| e.to_str()) == Some("png") {
                out.push(p);
            }
        }
    }
    recurse(std::path::Path::new(dir), &mut out);
    out.sort();
    out
}
