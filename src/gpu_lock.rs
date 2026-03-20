// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! GPU scheduling — thin shim that delegates to `kova c2 gpu`.
//! The real implementation lives in kova's gpu_sched module.

use anyhow::Result;
use std::process::Command;

fn kova_gpu(args: &[&str]) -> Result<()> {
    let mut cmd = Command::new("kova");
    cmd.args(["c2", "gpu"]);
    cmd.args(args);
    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.is_empty() { print!("{stdout}"); }
    if !stderr.is_empty() { eprint!("{stderr}"); }
    if !output.status.success() {
        anyhow::bail!("kova c2 gpu failed");
    }
    Ok(())
}

pub fn acquire(node: &str, job: &str) -> Result<()> {
    kova_gpu(&["lock", node, job])
}

pub fn release(node: &str) -> Result<()> {
    kova_gpu(&["release", node])
}

pub fn status() -> Result<()> {
    kova_gpu(&["status"])
}

pub fn is_available(node: &str) -> bool {
    std::path::Path::new(&format!(
        "{}/.kova/gpu/{node}.lock",
        dirs::home_dir().unwrap_or_default().display()
    ))
    .exists()
    .then(|| false)
    .unwrap_or(true)
}
