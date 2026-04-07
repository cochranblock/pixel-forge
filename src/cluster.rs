// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Forge cluster — distribute sprite generation across all available nodes.
//! Probes each node via SSH, benchmarks remotely, distributes work
//! proportionally to speed, streams results back over 10Gbps.
//!
//! Nodes run `pixel-forge` locally with their own GPU (Metal/CUDA/CPU).
//! Results are base64-encoded PNGs piped back over SSH stdout.
//! 32x32 sprites = ~3072 bytes each. Network is never the bottleneck.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::device_cap::{DeviceProfile, Tier};

/// Known nodes in the forge cluster.
const NODES: &[NodeDef] = &[
    NodeDef { name: "lf", host: "lf", token: "n0" },
    NodeDef { name: "gd", host: "gd", token: "n1" },
    NodeDef { name: "bt", host: "bt", token: "n2" },
    NodeDef { name: "st", host: "st", token: "n3" },
];

/// SSH connect timeout in seconds.
const SSH_TIMEOUT: u32 = 5;

/// Where pixel-forge binary lives on remote nodes.
const REMOTE_BIN: &str = "/home/mcochran/bin/pixel-forge";

/// Where model files live on remote nodes.
const REMOTE_MODELS_DIR: &str = "/home/mcochran/pixel-forge";

struct NodeDef {
    name: &'static str,
    host: &'static str,
    token: &'static str,
}

/// A node's profile after probing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProfile {
    pub name: String,
    pub host: String,
    pub reachable: bool,
    pub profile: Option<DeviceProfile>,
    /// Sprites per second this node can produce (estimated from benchmark).
    pub throughput: f64,
}

/// The full cluster state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    /// Local machine profile.
    pub local: NodeProfile,
    /// Remote node profiles.
    pub remotes: Vec<NodeProfile>,
    /// Total cluster throughput (sprites/sec).
    pub total_throughput: f64,
}

impl ClusterState {
    /// All active nodes (local + reachable remotes), sorted fastest first.
    pub fn active_nodes(&self) -> Vec<&NodeProfile> {
        let mut nodes: Vec<&NodeProfile> = std::iter::once(&self.local)
            .chain(self.remotes.iter().filter(|n| n.reachable && n.profile.is_some()))
            .collect();
        nodes.sort_by(|a, b| b.throughput.partial_cmp(&a.throughput).unwrap());
        nodes
    }

    /// Distribute `count` sprites across nodes proportionally to speed.
    /// Returns vec of (node_name, node_host, count, tier, is_local).
    pub fn distribute(&self, total: u32) -> Vec<WorkUnit> {
        let nodes = self.active_nodes();
        if nodes.is_empty() {
            return vec![];
        }

        let total_tp: f64 = nodes.iter().map(|n| n.throughput).sum();
        if total_tp <= 0.0 {
            // Equal split fallback
            let per = (total as usize / nodes.len()).max(1) as u32;
            return nodes.iter().enumerate().map(|(i, n)| {
                let count = if i == 0 { total - per * (nodes.len() as u32 - 1) } else { per };
                WorkUnit {
                    name: n.name.clone(),
                    host: n.host.clone(),
                    count,
                    tier: n.profile.as_ref().map(|p| p.tier).unwrap_or(Tier::Cinder),
                    is_local: n.host == "local",
                }
            }).collect();
        }

        let mut units = Vec::new();
        let mut assigned = 0u32;

        for (i, node) in nodes.iter().enumerate() {
            let share = if i == nodes.len() - 1 {
                // Last node gets remainder to avoid rounding loss
                total - assigned
            } else {
                let frac = node.throughput / total_tp;
                (frac * total as f64).round() as u32
            };

            if share > 0 {
                units.push(WorkUnit {
                    name: node.name.clone(),
                    host: node.host.clone(),
                    count: share,
                    tier: node.profile.as_ref().map(|p| p.tier).unwrap_or(Tier::Cinder),
                    is_local: node.host == "local",
                });
                assigned += share;
            }
        }

        units
    }
}

#[derive(Debug, Clone)]
pub struct WorkUnit {
    pub name: String,
    pub host: String,
    pub count: u32,
    pub tier: Tier,
    pub is_local: bool,
}

/// Estimate throughput from a device profile.
/// Sprites/sec based on the tier's benchmark speed and default 40 steps.
fn estimate_throughput(profile: &DeviceProfile) -> f64 {
    let ms_per_step = match profile.tier {
        Tier::Cinder => profile.cinder_ms,
        Tier::Quench => profile.quench_ms.unwrap_or(profile.cinder_ms * 5.3),
        Tier::Anvil => profile.quench_ms.map(|q| q * 2.7).unwrap_or(profile.cinder_ms * 14.0),
    };
    let ms_per_sprite = ms_per_step * 40.0; // 40 denoising steps
    if ms_per_sprite > 0.0 { 1000.0 / ms_per_sprite } else { 0.0 }
}

/// Probe a single remote node via SSH.
fn probe_remote(node: &NodeDef) -> NodeProfile {
    let ssh_result = Command::new("ssh")
        .args([
            "-o", &format!("ConnectTimeout={SSH_TIMEOUT}"),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            node.host,
            &format!("{REMOTE_BIN} probe --json 2>/dev/null || echo '{{\"error\":\"no binary\"}}'"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match ssh_result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            match serde_json::from_str::<DeviceProfile>(stdout.trim()) {
                Ok(profile) => {
                    let throughput = estimate_throughput(&profile);
                    NodeProfile {
                        name: node.name.to_string(),
                        host: node.host.to_string(),
                        reachable: true,
                        profile: Some(profile),
                        throughput,
                    }
                }
                Err(_) => {
                    // Node reachable but pixel-forge not installed or no JSON
                    NodeProfile {
                        name: node.name.to_string(),
                        host: node.host.to_string(),
                        reachable: true,
                        profile: None,
                        throughput: 0.0,
                    }
                }
            }
        }
        _ => NodeProfile {
            name: node.name.to_string(),
            host: node.host.to_string(),
            reachable: false,
            profile: None,
            throughput: 0.0,
        },
    }
}

/// Probe all nodes in parallel. Returns full cluster state.
pub fn probe_cluster() -> Result<ClusterState> {
    println!("probing forge cluster...");

    // Local probe
    let local_profile = crate::device_cap::auto_select()?;
    let local_throughput = estimate_throughput(&local_profile);
    let local = NodeProfile {
        name: "local".to_string(),
        host: "local".to_string(),
        reachable: true,
        profile: Some(local_profile),
        throughput: local_throughput,
    };
    println!("  local: {:.1} sprites/sec", local_throughput);

    // Remote probes — all in parallel
    let results: Arc<Mutex<Vec<NodeProfile>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for node in NODES {
        let results = Arc::clone(&results);
        let name = node.name;
        let host = node.host;
        let token = node.token;

        handles.push(thread::spawn(move || {
            let node_def = NodeDef { name, host, token };
            let profile = probe_remote(&node_def);
            let status = if profile.reachable {
                if profile.profile.is_some() {
                    format!("{:.1} sprites/sec", profile.throughput)
                } else {
                    "reachable, no pixel-forge".to_string()
                }
            } else {
                "offline".to_string()
            };
            println!("  {}: {}", name, status);
            results.lock().unwrap().push(profile);
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    let remotes = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    let total_throughput = local.throughput
        + remotes.iter()
            .filter(|n| n.reachable && n.profile.is_some())
            .map(|n| n.throughput)
            .sum::<f64>();

    let active_count = 1 + remotes.iter().filter(|n| n.reachable && n.profile.is_some()).count();
    println!("cluster: {} nodes, {:.1} sprites/sec total", active_count, total_throughput);

    Ok(ClusterState {
        local,
        remotes,
        total_throughput,
    })
}

/// Generate sprites across the cluster.
/// Distributes work, runs in parallel, collects PNG bytes back.
pub fn cluster_generate(
    class: &str,
    count: u32,
    steps: usize,
    palette: &str,
    cluster: &ClusterState,
) -> Result<Vec<Vec<u8>>> {
    let work = cluster.distribute(count);
    if work.is_empty() {
        anyhow::bail!("no active nodes in cluster");
    }

    println!("distributing {} sprites across {} nodes:", count, work.len());
    for w in &work {
        println!("  {} ({}): {} sprites via {}", w.name, w.host, w.count, w.tier);
    }

    #[allow(clippy::type_complexity)]
    let results: Arc<Mutex<Vec<(usize, Vec<Vec<u8>>)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for (idx, unit) in work.iter().enumerate() {
        let results = Arc::clone(&results);
        let class = class.to_string();
        let palette = palette.to_string();
        let unit = unit.clone();

        handles.push(thread::spawn(move || {
            let pngs = if unit.is_local {
                generate_local(&class, unit.count, steps, &palette)
            } else {
                generate_remote(&unit.host, &class, unit.count, steps, &palette)
            };

            match pngs {
                Ok(data) => {
                    println!("  {} done: {} sprites", unit.name, data.len());
                    results.lock().unwrap().push((idx, data));
                }
                Err(e) => {
                    eprintln!("  {} failed: {e}", unit.name);
                }
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    // Collect and flatten in order
    let mut collected = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    collected.sort_by_key(|(idx, _)| *idx);
    let all_pngs: Vec<Vec<u8>> = collected.into_iter().flat_map(|(_, pngs)| pngs).collect();

    println!("cluster: {} sprites collected", all_pngs.len());
    Ok(all_pngs)
}

/// Generate locally using auto_sample.
fn generate_local(class: &str, count: u32, steps: usize, palette: &str) -> Result<Vec<Vec<u8>>> {
    let cond = crate::class_cond::lookup(class);
    let pal = crate::palette::load_palette(palette)?;

    let raw_images = crate::device_cap::auto_sample(&cond, 32, count, steps)?;

    let mut pngs = Vec::new();
    for img in raw_images {
        let snapped = crate::grid::snap_to_grid(&img, 32);
        let quantized = crate::palette::quantize(&snapped, &pal);
        let mut buf = Vec::new();
        quantized.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        )?;
        pngs.push(buf);
    }
    Ok(pngs)
}

/// Generate on a remote node via SSH.
/// Runs pixel-forge generate, outputs PNGs as base64 lines to stdout.
fn generate_remote(host: &str, class: &str, count: u32, steps: usize, palette: &str) -> Result<Vec<Vec<u8>>> {
    // Remote command: generate sprites, write each as base64 PNG to stdout
    // We use a temp dir on the remote, generate there, base64 each file, clean up
    let remote_cmd = format!(
        r#"cd {REMOTE_MODELS_DIR} && \
        TMPD=$(mktemp -d) && \
        {REMOTE_BIN} auto {class} \
            --count {count} \
            --steps {steps} \
            --palette {palette} \
            --output "$TMPD/out.png" 2>/dev/null && \
        if [ {count} -eq 1 ]; then \
            base64 "$TMPD/out.png"; \
        else \
            for f in "$TMPD"/out-*.png "$TMPD"/out.png; do \
                [ -f "$f" ] && base64 "$f" && echo "---SPLIT---"; \
            done; \
        fi && \
        rm -rf "$TMPD""#
    );

    let output = Command::new("ssh")
        .args([
            "-o", &format!("ConnectTimeout={SSH_TIMEOUT}"),
            "-o", "BatchMode=yes",
            host,
            &remote_cmd,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("{host}: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_base64_pngs(&stdout)
}

/// Parse base64-encoded PNGs from remote output.
fn parse_base64_pngs(output: &str) -> Result<Vec<Vec<u8>>> {
    use base64::Engine as _;
    let engine = base64::engine::general_purpose::STANDARD;

    let mut pngs = Vec::new();

    if output.contains("---SPLIT---") {
        // Multiple images separated by marker
        for chunk in output.split("---SPLIT---") {
            let b64: String = chunk.chars().filter(|c| !c.is_whitespace()).collect();
            if b64.is_empty() { continue; }
            match engine.decode(&b64) {
                Ok(bytes) => pngs.push(bytes),
                Err(e) => eprintln!("  base64 decode error: {e}"),
            }
        }
    } else {
        // Single image
        let b64: String = output.chars().filter(|c| !c.is_whitespace()).collect();
        if !b64.is_empty() {
            pngs.push(engine.decode(&b64)?);
        }
    }

    Ok(pngs)
}

/// Deploy pixel-forge binary and models to a remote node.
pub fn deploy_to_node(host: &str) -> Result<()> {
    println!("deploying pixel-forge to {host}...");

    // 1. Ensure remote dirs exist
    let mkdir_cmd = format!("mkdir -p /home/mcochran/bin {REMOTE_MODELS_DIR}");
    let status = Command::new("ssh")
        .args(["-o", &format!("ConnectTimeout={SSH_TIMEOUT}"), host, &mkdir_cmd])
        .status()?;
    if !status.success() {
        anyhow::bail!("failed to create dirs on {host}");
    }

    // 2. Rsync the project source for remote build
    println!("  syncing source...");
    let status = Command::new("rsync")
        .args([
            "-az", "--delete",
            "--exclude", "target/",
            "--exclude", ".git/",
            "--exclude", "output/",
            "--exclude", "data/raw/",
            "/Users/mcochran/pixel-forge/",
            &format!("{host}:{REMOTE_MODELS_DIR}/"),
        ])
        .status()?;
    if !status.success() {
        anyhow::bail!("rsync to {host} failed");
    }

    // 3. Build on the remote node with appropriate features
    // Linux nodes: try CUDA first, fall back to CPU-only
    println!("  building on {host}...");
    let build_cmd = format!(
        "cd {REMOTE_MODELS_DIR} && \
        if command -v nvidia-smi >/dev/null 2>&1; then \
            cargo build --release --no-default-features --features cuda 2>&1; \
        else \
            cargo build --release --no-default-features 2>&1; \
        fi && \
        cp target/release/pixel-forge /home/mcochran/bin/pixel-forge"
    );

    let output = Command::new("ssh")
        .args([
            "-o", &format!("ConnectTimeout={SSH_TIMEOUT}"),
            host,
            &build_cmd,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!("build failed on {host}:\n{stdout}\n{stderr}");
    }

    println!("  {host}: deployed");
    Ok(())
}

/// Deploy to all nodes in parallel.
pub fn deploy_all() -> Result<()> {
    println!("deploying pixel-forge to all nodes...");
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for node in NODES {
        let errors = Arc::clone(&errors);
        let host = node.host.to_string();
        let name = node.name.to_string();

        handles.push(thread::spawn(move || {
            if let Err(e) = deploy_to_node(&host) {
                eprintln!("  {name}: deploy failed: {e}");
                errors.lock().unwrap().push(format!("{name}: {e}"));
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    let errs = Arc::try_unwrap(errors).unwrap().into_inner().unwrap();
    if errs.is_empty() {
        println!("all nodes deployed");
    } else {
        println!("{} nodes failed:", errs.len());
        for e in &errs {
            println!("  {e}");
        }
    }

    Ok(())
}

/// Sync model files to a remote node.
pub fn sync_models(host: &str) -> Result<()> {
    println!("  syncing models to {host}...");
    let model_files: Vec<&str> = [Tier::Cinder, Tier::Quench, Tier::Anvil]
        .iter()
        .map(|t| t.model_file())
        .filter(|f| std::path::Path::new(f).exists())
        .collect();

    for model_file in &model_files {
        let status = Command::new("scp")
            .args([
                "-o", &format!("ConnectTimeout={SSH_TIMEOUT}"),
                model_file,
                &format!("{host}:{REMOTE_MODELS_DIR}/{model_file}"),
            ])
            .status()?;
        if !status.success() {
            eprintln!("  warning: failed to sync {model_file} to {host}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_node(name: &str, host: &str, throughput: f64) -> NodeProfile {
        NodeProfile {
            name: name.to_string(),
            host: host.to_string(),
            reachable: true,
            profile: Some(DeviceProfile {
                tier: Tier::Cinder,
                cinder_ms: 10.0,
                quench_ms: None,
                ram_mb: 4096,
                backend: crate::device_cap::Backend::Cpu,
                os_version: "test".to_string(),
            }),
            throughput,
        }
    }

    fn mock_cluster(remotes: Vec<NodeProfile>) -> ClusterState {
        let total_throughput = 1.0 + remotes.iter().map(|n| n.throughput).sum::<f64>();
        ClusterState {
            local: mock_node("local", "local", 1.0),
            remotes,
            total_throughput,
        }
    }

    #[test]
    fn distribute_single_node() {
        let cluster = mock_cluster(vec![]);
        let work = cluster.distribute(10);
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].count, 10);
    }

    #[test]
    fn distribute_preserves_total() {
        let cluster = mock_cluster(vec![
            mock_node("gd", "gd", 3.0),
            mock_node("bt", "bt", 1.0),
        ]);
        let work = cluster.distribute(100);
        let total: u32 = work.iter().map(|w| w.count).sum();
        assert_eq!(total, 100, "distribution must sum to requested count");
    }

    #[test]
    fn distribute_proportional_to_throughput() {
        let cluster = mock_cluster(vec![
            mock_node("fast", "fast", 9.0), // 9x local
        ]);
        let work = cluster.distribute(100);
        let fast = work.iter().find(|w| w.name == "fast").unwrap();
        // fast has 9.0 / 10.0 total → should get ~90
        assert!(fast.count >= 85 && fast.count <= 95, "fast got {}", fast.count);
    }

    #[test]
    fn distribute_zero_count() {
        let cluster = mock_cluster(vec![]);
        let work = cluster.distribute(0);
        // Should not panic, may be empty
        let total: u32 = work.iter().map(|w| w.count).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn active_nodes_excludes_unreachable() {
        let mut cluster = mock_cluster(vec![
            mock_node("gd", "gd", 5.0),
        ]);
        cluster.remotes[0].reachable = false;
        let active = cluster.active_nodes();
        assert_eq!(active.len(), 1); // only local
    }

    #[test]
    fn active_nodes_sorted_by_throughput() {
        let cluster = mock_cluster(vec![
            mock_node("slow", "slow", 0.5),
            mock_node("fast", "fast", 10.0),
        ]);
        let active = cluster.active_nodes();
        assert_eq!(active[0].name, "fast");
    }

    #[test]
    fn estimate_throughput_positive() {
        let profile = DeviceProfile {
            tier: Tier::Cinder,
            cinder_ms: 10.0,
            quench_ms: None,
            ram_mb: 4096,
            backend: crate::device_cap::Backend::Cpu,
            os_version: "test".to_string(),
        };
        let tp = estimate_throughput(&profile);
        assert!(tp > 0.0, "throughput should be positive, got {tp}");
    }

    #[test]
    fn nodes_constant_has_four_entries() {
        assert_eq!(NODES.len(), 4);
    }
}

/// Sync models to all nodes in parallel.
pub fn sync_models_all() -> Result<()> {
    println!("syncing models to all nodes...");
    let mut handles = Vec::new();

    for node in NODES {
        let host = node.host.to_string();
        handles.push(thread::spawn(move || {
            if let Err(e) = sync_models(&host) {
                eprintln!("  {}: model sync failed: {e}", host);
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }
    println!("model sync complete");
    Ok(())
}

