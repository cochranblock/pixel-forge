// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! pixel-forge library — exposes the app and core modules for mobile builds.
#![allow(dead_code)]

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
