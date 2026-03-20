// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! GPU scheduling lock — file-based mutex for GPU training jobs.
//! Any process can acquire/check/release. Prevents mid-inference swaps.
//!
//! Lock file: ~/.forge-gpu/<node>.lock
//! Queue file: ~/.forge-gpu/<node>.queue (one job per line)
//!
//! Usage:
//!   pixel-forge gpu lock lf "expert training 20 epochs"
//!   pixel-forge gpu status
//!   pixel-forge gpu queue lf "quench retrain 100 epochs"
//!   pixel-forge gpu release lf
//!   pixel-forge gpu drain lf   # run next queued job

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

const LOCK_DIR: &str = ".forge-gpu";

#[derive(Serialize, Deserialize, Debug)]
pub struct GpuLock {
    pub node: String,
    pub job: String,
    pub pid: u32,
    pub started: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueueEntry {
    pub job: String,
    pub command: String,
    pub priority: u8, // 0 = highest
    pub added: u64,
}

fn lock_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(LOCK_DIR)
}

fn lock_path(node: &str) -> PathBuf {
    lock_dir().join(format!("{node}.lock"))
}

fn queue_path(node: &str) -> PathBuf {
    lock_dir().join(format!("{node}.queue"))
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Acquire GPU lock for a node. Fails if already locked.
pub fn acquire(node: &str, job: &str) -> Result<()> {
    let dir = lock_dir();
    fs::create_dir_all(&dir)?;

    let path = lock_path(node);
    if path.exists() {
        let existing: GpuLock = serde_json::from_str(&fs::read_to_string(&path)?)?;
        bail!(
            "GPU on {} already locked by: {} (pid {}, {}s ago)",
            node,
            existing.job,
            existing.pid,
            now_epoch().saturating_sub(existing.started)
        );
    }

    let lock = GpuLock {
        node: node.to_string(),
        job: job.to_string(),
        pid: std::process::id(),
        started: now_epoch(),
    };

    fs::write(&path, serde_json::to_string_pretty(&lock)?)?;
    println!("GPU lock acquired on {node}: {job}");
    Ok(())
}

/// Release GPU lock.
pub fn release(node: &str) -> Result<()> {
    let path = lock_path(node);
    if path.exists() {
        let lock: GpuLock = serde_json::from_str(&fs::read_to_string(&path)?)?;
        fs::remove_file(&path)?;
        println!("GPU lock released on {node} (was: {})", lock.job);
    } else {
        println!("no lock held on {node}");
    }
    Ok(())
}

/// Check lock status across all nodes.
pub fn status() -> Result<()> {
    let dir = lock_dir();
    if !dir.exists() {
        println!("no GPU locks (dir doesn't exist)");
        return Ok(());
    }

    let mut found = false;
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".lock") {
            let lock: GpuLock = serde_json::from_str(&fs::read_to_string(entry.path())?)?;
            let elapsed = now_epoch().saturating_sub(lock.started);
            let mins = elapsed / 60;
            println!("{}: LOCKED — {} (pid {}, {}m{}s)", lock.node, lock.job, lock.pid, mins, elapsed % 60);
            found = true;
        } else if name.ends_with(".queue") {
            let queue = load_queue(&name.trim_end_matches(".queue"))?;
            if !queue.is_empty() {
                let node = name.trim_end_matches(".queue");
                println!("{node}: {} queued jobs", queue.len());
                for (i, entry) in queue.iter().enumerate() {
                    println!("  [{i}] p{}: {}", entry.priority, entry.job);
                }
                found = true;
            }
        }
    }

    if !found {
        println!("all GPUs idle, no queue");
    }
    Ok(())
}

/// Add job to queue.
pub fn enqueue(node: &str, job: &str, command: &str, priority: u8) -> Result<()> {
    let dir = lock_dir();
    fs::create_dir_all(&dir)?;

    let mut queue = load_queue(node)?;
    queue.push(QueueEntry {
        job: job.to_string(),
        command: command.to_string(),
        priority,
        added: now_epoch(),
    });
    // Sort by priority (lower = higher priority)
    queue.sort_by_key(|e| e.priority);
    save_queue(node, &queue)?;
    println!("queued on {node} (p{priority}): {job} [{} in queue]", queue.len());
    Ok(())
}

/// Pop next job from queue and print the command.
pub fn drain(node: &str) -> Result<Option<QueueEntry>> {
    let mut queue = load_queue(node)?;
    if queue.is_empty() {
        println!("{node}: queue empty");
        return Ok(None);
    }

    let next = queue.remove(0);
    save_queue(node, &queue)?;
    println!("{node}: next job: {} — {}", next.job, next.command);
    Ok(Some(next))
}

fn load_queue(node: &str) -> Result<Vec<QueueEntry>> {
    let path = queue_path(node);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&data).unwrap_or_default())
}

fn save_queue(node: &str, queue: &[QueueEntry]) -> Result<()> {
    let path = queue_path(node);
    fs::write(&path, serde_json::to_string_pretty(queue)?)?;
    Ok(())
}

/// Check if GPU is available on a node.
pub fn is_available(node: &str) -> bool {
    !lock_path(node).exists()
}
