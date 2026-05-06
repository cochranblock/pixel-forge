// Unlicense — cochranblock.org
//! Per-channel z-score normalization for diffusion training.
//!
//! Maps training pixels [0,1] → ~N(0,1) before the noise schedule, and
//! inverts at sampling output. The trainer drops a sidecar JSON next to
//! each saved checkpoint; the sampler/wasm runtime reads it back so the
//! pixel space the model was trained in is recovered exactly.
//!
//! Backward compat: absence of a sidecar means raw [0,1] training — both
//! load and the inverse are no-ops, so old checkpoints keep working.
//!
//! This module is shared with the wasm crate via `#[path]`, so it must
//! stay free of candle/any-gpu dependencies. Tensor adapters live at the
//! call sites.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalizer {
    pub version: u32,
    pub channels: u32,
    pub img_size: u32,
    pub sample_count: u64,
    pub pixel_count: u64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Scalar standard deviation of the *centered* dataset across all
    /// channels and pixels — what EDM calls σ_data. Defaults to RMS of
    /// per-channel std when absent (same value old sidecars would imply).
    /// Used to derive the EDM preconditioning coefficients c_in/c_out/
    /// c_skip — the σ-dependent generalization of plain z-scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma_data: Option<f32>,
}

impl Normalizer {
    /// Load a manifest by direct path (e.g. `data_v3_32/normalize.json`).
    /// Returns Ok(None) if the file is missing — caller decides whether
    /// that is fatal.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("read {}", path.display()))?;
        Self::from_json_bytes(&bytes).map(Some)
    }

    /// Parse a manifest from raw JSON bytes — used by the wasm runtime,
    /// which fetches the sidecar over HTTP rather than from the filesystem.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(bytes)
            .context("parse normalize manifest")?;
        Self::from_json_value(&raw)
    }

    /// Sidecar path: `{model_path}.normalize.json`. Stays adjacent to the
    /// safetensors so the two travel together.
    pub fn sidecar_path(model_path: &Path) -> PathBuf {
        let mut p = model_path.as_os_str().to_owned();
        p.push(".normalize.json");
        PathBuf::from(p)
    }

    /// Load the sidecar that travels with a checkpoint.
    pub fn load_sidecar(model_path: &Path) -> Result<Option<Self>> {
        Self::load(&Self::sidecar_path(model_path))
    }

    /// Write the sidecar next to a checkpoint.
    pub fn save_sidecar(&self, model_path: &Path) -> Result<()> {
        let path = Self::sidecar_path(model_path);
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// EDM σ_data: the scalar std the preconditioning is calibrated for.
    /// When absent from the sidecar (older manifests) we synthesize it
    /// from the per-channel stds so EDM math still works on legacy files.
    /// Identity is RMS — sqrt(mean(σ_c²)).
    pub fn sigma_data(&self) -> f32 {
        self.sigma_data.unwrap_or_else(|| {
            let s2 = (self.std[0].powi(2) + self.std[1].powi(2) + self.std[2].powi(2)) / 3.0;
            s2.sqrt()
        })
    }

    /// In-place [0,1] → z-space on channel-first f32 pixels, layout
    /// `[r0..rN, g0..gN, b0..bN]`. `n_px` = H*W; channels assumed 3.
    pub fn to_z_pixels(&self, px: &mut [f32], n_px: usize) {
        for c in 0..3 {
            let off = c * n_px;
            let m = self.mean[c];
            let s = self.std[c].max(1e-6);
            for k in 0..n_px {
                px[off + k] = (px[off + k] - m) / s;
            }
        }
    }

    /// Inverse of `to_z_pixels` — z-space → [0,1] (no clamp; caller clamps).
    pub fn from_z_pixels(&self, px: &mut [f32], n_px: usize) {
        for c in 0..3 {
            let off = c * n_px;
            let m = self.mean[c];
            let s = self.std[c];
            for k in 0..n_px {
                px[off + k] = px[off + k] * s + m;
            }
        }
    }

    /// Mean-only centering: subtract per-channel mean, leave std intact.
    /// EDM preconditioning expects centered input — c_in handles the
    /// σ-dependent rescale at every noise level, so dividing by std here
    /// would just duplicate work and miscalibrate σ_data.
    pub fn center_pixels(&self, px: &mut [f32], n_px: usize) {
        for c in 0..3 {
            let off = c * n_px;
            let m = self.mean[c];
            for k in 0..n_px {
                px[off + k] -= m;
            }
        }
    }

    /// Inverse of `center_pixels` — add per-channel mean back, no clamp.
    pub fn uncenter_pixels(&self, px: &mut [f32], n_px: usize) {
        for c in 0..3 {
            let off = c * n_px;
            let m = self.mean[c];
            for k in 0..n_px {
                px[off + k] += m;
            }
        }
    }

    fn from_json_value(v: &serde_json::Value) -> Result<Self> {
        // normalize-stats writes f64; coerce to f32 here so the in-memory
        // form matches what we apply on the GPU.
        let mean_arr = v.get("mean").and_then(|m| m.as_array())
            .context("manifest missing 'mean' array")?;
        let std_arr = v.get("std").and_then(|s| s.as_array())
            .context("manifest missing 'std' array")?;
        anyhow::ensure!(mean_arr.len() == 3 && std_arr.len() == 3,
            "expected 3-channel mean/std, got mean={} std={}",
            mean_arr.len(), std_arr.len());
        let f = |a: &serde_json::Value| -> Result<f32> {
            a.as_f64().context("non-numeric stat").map(|x| x as f32)
        };
        Ok(Self {
            version:      v["version"].as_u64().unwrap_or(1) as u32,
            channels:     v["channels"].as_u64().unwrap_or(3) as u32,
            img_size:     v["img_size"].as_u64().unwrap_or(0) as u32,
            sample_count: v["sample_count"].as_u64().unwrap_or(0),
            pixel_count:  v["pixel_count"].as_u64().unwrap_or(0),
            mean: [f(&mean_arr[0])?, f(&mean_arr[1])?, f(&mean_arr[2])?],
            std:  [f(&std_arr[0])?,  f(&std_arr[1])?,  f(&std_arr[2])?],
            sigma_data: v.get("sigma_data").and_then(|s| s.as_f64()).map(|x| x as f32),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Normalizer {
        Normalizer {
            version: 1, channels: 3, img_size: 32,
            sample_count: 100, pixel_count: 100 * 1024,
            mean: [0.4, 0.3, 0.35],
            std:  [0.4, 0.4, 0.4],
            sigma_data: Some(0.4),
        }
    }

    #[test]
    fn round_trip_identity() {
        let n = fixture();
        let mut px = vec![0.0f32, 0.5, 1.0, 0.2, 0.8, 0.3];
        let original = px.clone();
        let n_px = 2; // 3 channels × 2 pixels
        n.to_z_pixels(&mut px, n_px);
        n.from_z_pixels(&mut px, n_px);
        for (a, b) in px.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "round-trip drift: {a} vs {b}");
        }
    }

    #[test]
    fn manifest_round_trip_via_disk() {
        let pid = std::process::id();
        let manifest = std::env::temp_dir()
            .join(format!("pixel-forge-normalize-test-{pid}.json"));
        let n = fixture();
        std::fs::write(&manifest, serde_json::to_vec_pretty(&n).unwrap()).unwrap();
        let loaded = Normalizer::load(&manifest).unwrap().unwrap();
        let _ = std::fs::remove_file(&manifest);
        assert_eq!(loaded.mean, n.mean);
        assert_eq!(loaded.std,  n.std);
    }

    #[test]
    fn sidecar_naming() {
        let p = Path::new("/tmp/cinder.safetensors");
        assert_eq!(
            Normalizer::sidecar_path(p),
            PathBuf::from("/tmp/cinder.safetensors.normalize.json"),
        );
    }

    #[test]
    fn sigma_data_explicit_takes_precedence() {
        let mut n = fixture();
        n.sigma_data = Some(0.123);
        assert!((n.sigma_data() - 0.123).abs() < 1e-6);
    }

    #[test]
    fn sigma_data_falls_back_to_rms() {
        let mut n = fixture();
        n.sigma_data = None;
        // RMS of [0.4, 0.4, 0.4] is 0.4.
        assert!((n.sigma_data() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn legacy_manifest_without_sigma_data_loads() {
        let json = br#"{"version":1,"channels":3,"img_size":32,"sample_count":100,"pixel_count":102400,"mean":[0.4,0.3,0.35],"std":[0.4,0.4,0.4]}"#;
        let n = Normalizer::from_json_bytes(json).unwrap();
        assert!(n.sigma_data.is_none());
        assert!((n.sigma_data() - 0.4).abs() < 1e-6); // RMS fallback works
    }
}
