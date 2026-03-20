// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pipeline stub for mobile builds (no SD, no hf-hub, no tokenizers).
//! Only provides best_device() and model_ready() — the trained models
//! (Cinder/Quench/Anvil) run directly via train::sample_*.

use anyhow::Result;
use image::RgbaImage;

/// Auto-detect best device: Metal GPU on Apple Silicon, CPU fallback.
pub fn best_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            if let Ok(d) = candle_core::Device::new_cuda(0) {
                return d;
            }
        }
    }
    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            if let Ok(d) = candle_core::Device::new_metal(0) {
                return d;
            }
        }
    }
    candle_core::Device::Cpu
}

/// SD model is never available on mobile.
pub fn model_ready() -> bool {
    false
}

/// Generate — not available without sd-pipeline feature.
pub fn generate(_prompt: &str, _size: u32, _count: u32) -> Result<Vec<RgbaImage>> {
    anyhow::bail!("SD pipeline not available — use trained models (generate/auto command)")
}

/// Download — not available on mobile.
pub fn download_model() -> Result<()> {
    anyhow::bail!("SD pipeline not available on this build")
}
