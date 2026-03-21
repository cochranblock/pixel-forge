// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Device capability detection and automatic model tier selection.
//! Probes GPU, RAM, and inference speed to pick the best model
//! (Cinder / Quench / Anvil) for the current hardware.
//! Caches the result so subsequent launches skip the benchmark.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

use crate::tiny_unet::TinyUNet;
use crate::medium_unet::MediumUNet;


/// Model tier — ordered by resource needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    Cinder, // ~1.09M params, 4.2 MB
    Quench, // ~5.83M params, 22 MB
    Anvil,  // ~16M params, ~64 MB (planned)
}

impl Tier {
    pub fn name(&self) -> &'static str {
        match self {
            Tier::Cinder => "Cinder",
            Tier::Quench => "Quench",
            Tier::Anvil => "Anvil",
        }
    }

    pub fn model_file(&self) -> &'static str {
        match self {
            Tier::Cinder => "pixel-forge-cinder.safetensors",
            Tier::Quench => "pixel-forge-quench.safetensors",
            Tier::Anvil => "pixel-forge-anvil.safetensors",
        }
    }

    /// Full path to the model file, respecting platform storage.
    /// On Android, HOME is set to internal_data_path by the entry point.
    /// On desktop, checks CWD first, then HOME.
    pub fn model_path(&self) -> PathBuf {
        let filename = self.model_file();
        // Check CWD first (desktop dev workflow)
        let cwd = PathBuf::from(filename);
        if cwd.exists() {
            return cwd;
        }
        // Check HOME (Android internal storage, or desktop home)
        if let Ok(home) = std::env::var("HOME") {
            let home_path = PathBuf::from(home).join(filename);
            if home_path.exists() {
                return home_path;
            }
        }
        // Check next to the binary
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let exe_path = dir.join(filename);
                if exe_path.exists() {
                    return exe_path;
                }
            }
        }
        // Fall back to bare filename (will error at load time)
        PathBuf::from(filename)
    }

    pub fn param_count(&self) -> u64 {
        match self {
            Tier::Cinder => 1_090_000,
            Tier::Quench => 5_830_000,
            Tier::Anvil => 16_000_000,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Cached device profile — written to disk after first probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Which GPU backend is active.
    pub backend: Backend,
    /// System RAM in MB.
    pub ram_mb: u64,
    /// Time in ms to run one Cinder inference pass (16x16, 1 step).
    pub cinder_ms: f64,
    /// Time in ms to run one Quench inference pass (16x16, 1 step), if tested.
    pub quench_ms: Option<f64>,
    /// Selected tier based on benchmarks.
    pub tier: Tier,
    /// OS version string for cache invalidation.
    pub os_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    Metal,
    Cuda,
    Cpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Metal => write!(f, "Metal"),
            Backend::Cuda => write!(f, "CUDA"),
            Backend::Cpu => write!(f, "CPU"),
        }
    }
}

/// Where the cached profile lives.
fn profile_path() -> PathBuf {
    let mut p = dirs_or_fallback();
    p.push("device_profile.json");
    p
}

fn dirs_or_fallback() -> PathBuf {
    // Use project-local cache dir
    let mut p = std::env::current_exe()
        .unwrap_or_else(|_| PathBuf::from("."))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    // On mobile, current_exe parent is the app bundle — fine.
    // On desktop, put it next to the binary.
    p.push(".pixel-forge");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Load cached profile, or return None if stale/missing.
fn load_cached() -> Option<DeviceProfile> {
    let path = profile_path();
    let data = std::fs::read_to_string(&path).ok()?;
    let profile: DeviceProfile = serde_json::from_str(&data).ok()?;

    // Invalidate if OS changed (driver updates, new GPU)
    let current_os = os_version();
    if profile.os_version != current_os {
        return None;
    }

    Some(profile)
}

/// Save profile to disk.
fn save_cached(profile: &DeviceProfile) -> Result<()> {
    let path = profile_path();
    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Get current OS version string.
fn os_version() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("/usr/bin/sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_string()
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/version")
            .unwrap_or_default()
            .trim()
            .to_string()
    }
    #[cfg(target_os = "windows")]
    {
        "windows".to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unknown".to_string()
    }
}

/// Detect backend without printing.
fn detect_backend() -> (Backend, Device) {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            if let Ok(d) = candle_core::Device::new_cuda(0) {
                return (Backend::Cuda, d);
            }
        }
    }
    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            if let Ok(d) = candle_core::Device::new_metal(0) {
                return (Backend::Metal, d);
            }
        }
    }
    (Backend::Cpu, Device::Cpu)
}

/// Get system RAM in MB.
fn system_ram_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("/usr/sbin/sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|bytes| bytes / (1024 * 1024))
            .unwrap_or(0)
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                    })
            })
            .map(|kb| kb / 1024)
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

/// Benchmark a single forward pass of TinyUNet (Cinder).
/// Returns time in ms.
fn bench_cinder(device: &Device) -> Result<f64> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = TinyUNet::new(vb)?;

    let x = Tensor::rand(0f32, 1f32, (1, 3, 16, 16), device)?;
    let t = Tensor::new(&[0.5f32], device)?;
    let c = Tensor::new(&[0u32], device)?;

    // Warm up
    let _ = model.forward(&x, &t, &c)?;

    // Time 3 passes, take median
    let mut times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = model.forward(&x, &t, &c)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[1]) // median
}

/// Benchmark a single forward pass of MediumUNet (Quench).
/// Returns time in ms.
fn bench_quench(device: &Device) -> Result<f64> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MediumUNet::new(vb)?;

    let x = Tensor::rand(0f32, 1f32, (1, 3, 16, 16), device)?;
    let t = Tensor::new(&[0.5f32], device)?;
    let c = Tensor::new(&[0u32], device)?;

    // Warm up
    let _ = model.forward(&x, &t, &c)?;

    let mut times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = model.forward(&x, &t, &c)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[1])
}

/// Max acceptable inference time per step in ms.
/// 40 steps at this limit = total generation time.
/// Target: < 2 seconds total → 50ms/step max.
const MAX_STEP_MS: f64 = 50.0;

/// Pick tier based on benchmarks and available models.
fn select_tier(cinder_ms: f64, quench_ms: Option<f64>, ram_mb: u64) -> Tier {
    // Check what model files exist
    let anvil_exists = Tier::Anvil.model_path().exists();
    let quench_exists = Tier::Quench.model_path().exists();

    // RAM gate: need at least 2GB for Quench, 3GB for Anvil
    let ram_ok_quench = ram_mb >= 2048;
    let ram_ok_anvil = ram_mb >= 3072;

    // If Quench runs fast enough and Anvil exists, try Anvil
    // Anvil is ~2.7x Quench params → estimate ~2.7x inference time
    if let Some(q_ms) = quench_ms {
        let anvil_est = q_ms * 2.7;
        if anvil_exists && ram_ok_anvil && anvil_est < MAX_STEP_MS {
            return Tier::Anvil;
        }
        if quench_exists && ram_ok_quench && q_ms < MAX_STEP_MS {
            return Tier::Quench;
        }
    }

    // Quench benchmark not available — estimate from Cinder
    // Quench is ~5.3x Cinder params → estimate ~5.3x inference time
    if quench_ms.is_none() && quench_exists && ram_ok_quench {
        let quench_est = cinder_ms * 5.3;
        if quench_est < MAX_STEP_MS {
            // Anvil estimate
            let anvil_est = quench_est * 2.7;
            if anvil_exists && ram_ok_anvil && anvil_est < MAX_STEP_MS {
                return Tier::Anvil;
            }
            return Tier::Quench;
        }
    }

    Tier::Cinder
}

/// Main entry point: detect device, benchmark, select tier.
/// Returns cached result on subsequent calls.
/// Call this once at app startup — everything else flows from the tier.
pub fn auto_select() -> Result<DeviceProfile> {
    // Check cache first
    if let Some(profile) = load_cached() {
        println!("device: {} | tier: {} (cached)", profile.backend, profile.tier);
        return Ok(profile);
    }

    println!("probing device capabilities...");
    let (backend, device) = detect_backend();
    let ram_mb = system_ram_mb();
    println!("  backend: {backend}");
    println!("  ram: {} MB", ram_mb);

    // Benchmark Cinder (always)
    let cinder_ms = bench_cinder(&device)?;
    println!("  cinder: {:.1} ms/step", cinder_ms);

    // Benchmark Quench if Cinder is fast enough to suggest it's worth trying
    let quench_ms = if cinder_ms < MAX_STEP_MS / 5.3 {
        // Cinder is fast → Quench might fit in budget
        match bench_quench(&device) {
            Ok(ms) => {
                println!("  quench: {:.1} ms/step", ms);
                Some(ms)
            }
            Err(e) => {
                println!("  quench bench failed: {e}");
                None
            }
        }
    } else {
        println!("  quench: skipped (cinder too slow)");
        None
    };

    let tier = select_tier(cinder_ms, quench_ms, ram_mb);
    println!("  selected: {tier}");

    let profile = DeviceProfile {
        backend,
        ram_mb,
        cinder_ms,
        quench_ms,
        tier,
        os_version: os_version(),
    };

    if let Err(e) = save_cached(&profile) {
        eprintln!("  warning: failed to cache profile: {e}");
    }

    Ok(profile)
}

/// Force re-probe (user wants to override or hardware changed).
pub fn reprobe() -> Result<DeviceProfile> {
    let path = profile_path();
    let _ = std::fs::remove_file(&path);
    auto_select()
}

/// Sample using the auto-selected tier.
/// Single entry point — caller doesn't need to know which model.
pub fn auto_sample(
    class_id: u32,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<image::RgbaImage>> {
    let profile = auto_select()?;
    if !profile.tier.model_path().exists() {
        let fallback = best_available_tier();
        println!("  {} not found, falling back to {}", profile.tier, fallback);
        return sample_tier(fallback, class_id, img_size, count, steps);
    }

    sample_tier(profile.tier, class_id, img_size, count, steps)
}

/// Find the best model tier that actually has a file on disk.
pub fn best_available() -> Tier {
    best_available_tier()
}

fn best_available_tier() -> Tier {
    for tier in [Tier::Anvil, Tier::Quench, Tier::Cinder] {
        if tier.model_path().exists() {
            return tier;
        }
    }
    Tier::Cinder
}

/// Sample with a specific tier.
fn sample_tier(
    tier: Tier,
    class_id: u32,
    img_size: u32,
    count: u32,
    steps: usize,
) -> Result<Vec<image::RgbaImage>> {
    match tier {
        Tier::Cinder => crate::train::sample(&tier.model_path().to_string_lossy(), class_id, img_size, count, steps),
        Tier::Quench => crate::train::sample_medium(&tier.model_path().to_string_lossy(), class_id, img_size, count, steps),
        Tier::Anvil => {
            crate::train::sample_anvil(&tier.model_path().to_string_lossy(), class_id, img_size, count, steps)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_selection_low_end() {
        // Slow device, low RAM → Cinder
        let tier = select_tier(100.0, None, 1024);
        assert_eq!(tier, Tier::Cinder);
    }

    #[test]
    fn tier_ordering() {
        assert_eq!(Tier::Cinder.param_count(), 1_090_000);
        assert_eq!(Tier::Quench.param_count(), 5_830_000);
        assert_eq!(Tier::Anvil.param_count(), 16_000_000);
    }

    #[test]
    fn model_filenames() {
        assert!(Tier::Cinder.model_file().contains("cinder"));
        assert!(Tier::Quench.model_file().contains("quench"));
        assert!(Tier::Anvil.model_file().contains("anvil"));
    }
}
