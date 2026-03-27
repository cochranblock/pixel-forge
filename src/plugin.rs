// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Plugin protocol — kova talks to pixel-forge through this interface.
//! JSON request/response over stdin/stdout. One request per invocation.
//!
//! Commands:
//!   probe       → DeviceProfile JSON
//!   generate    → base64 PNG(s)
//!   models      → list available model files + tiers
//!   palettes    → list available palettes
//!   classes     → list sprite classes

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::device_cap::{self, Tier};

#[derive(Debug, Serialize, Deserialize)]
pub struct PluginRequest {
    pub cmd: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct PluginResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl PluginResponse {
    fn success(data: serde_json::Value) -> Self {
        Self { ok: true, error: None, data: Some(data) }
    }
    fn fail(msg: impl Into<String>) -> Self {
        Self { ok: false, error: Some(msg.into()), data: None }
    }
}

const CLASS_NAMES: &[&str] = &[
    "character", "weapon", "potion", "terrain", "enemy",
    "tree", "building", "animal", "effect", "food",
    "armor", "tool", "vehicle", "ui", "misc",
];

const PALETTE_NAMES: &[&str] = &[
    "stardew", "starbound", "snes", "nes", "gameboy", "pico8", "endesga",
];

/// Handle a single plugin request. Called by `pixel-forge plugin`.
pub fn handle(input: &str) -> PluginResponse {
    let req: PluginRequest = match serde_json::from_str(input) {
        Ok(r) => r,
        Err(e) => return PluginResponse::fail(format!("bad request: {e}")),
    };

    match req.cmd.as_str() {
        "probe" => handle_probe(),
        "generate" => handle_generate(&req.args),
        "models" => handle_models(),
        "palettes" => handle_palettes(),
        "classes" => handle_classes(),
        "version" => PluginResponse::success(serde_json::json!({
            "name": "pixel-forge",
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
        })),
        "capabilities" => PluginResponse::success(serde_json::json!({
            "name": "pixel-forge",
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
            "commands": ["probe", "generate", "models", "palettes", "classes"],
            "ui": {
                "selectors": [
                    {
                        "id": "class",
                        "label": "class",
                        "values": CLASS_NAMES,
                        "default": "character",
                    },
                    {
                        "id": "palette",
                        "label": "palette",
                        "values": PALETTE_NAMES,
                        "default": "stardew",
                    },
                ],
                "sliders": [
                    { "id": "count", "label": "n", "min": 1, "max": 32, "default": 4 },
                    { "id": "steps", "label": "steps", "min": 5, "max": 100, "default": 10 },
                ],
                "action": "generate",
                "output": "sprites",
            },
        })),
        other => PluginResponse::fail(format!("unknown command: {other}")),
    }
}

fn handle_probe() -> PluginResponse {
    match device_cap::auto_select() {
        Ok(profile) => PluginResponse::success(serde_json::to_value(&profile).unwrap()),
        Err(e) => PluginResponse::fail(format!("{e}")),
    }
}

fn handle_models() -> PluginResponse {
    let models: Vec<serde_json::Value> = [Tier::Cinder, Tier::Quench, Tier::Anvil]
        .iter()
        .map(|t| {
            let file = t.model_file();
            let exists = std::path::Path::new(file).exists();
            let size = if exists {
                std::fs::metadata(file).ok().map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };
            serde_json::json!({
                "tier": t.name(),
                "file": file,
                "exists": exists,
                "size_bytes": size,
                "params": t.param_count(),
            })
        })
        .collect();
    PluginResponse::success(serde_json::json!({ "models": models }))
}

fn handle_palettes() -> PluginResponse {
    PluginResponse::success(serde_json::json!({ "palettes": PALETTE_NAMES }))
}

fn handle_classes() -> PluginResponse {
    PluginResponse::success(serde_json::json!({ "classes": CLASS_NAMES }))
}

fn handle_generate(args: &serde_json::Value) -> PluginResponse {
    let class = args.get("class").and_then(|v| v.as_str()).unwrap_or("character");
    let count = args.get("count").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let steps = args.get("steps").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let palette = args.get("palette").and_then(|v| v.as_str()).unwrap_or("stardew");

    let cond = crate::class_cond::lookup(&class.to_lowercase());

    let pal = match crate::palette::load_palette(palette) {
        Ok(p) => p,
        Err(e) => return PluginResponse::fail(format!("palette: {e}")),
    };

    let raw_images = match device_cap::auto_sample(&cond, 16, count, steps) {
        Ok(imgs) => imgs,
        Err(e) => return PluginResponse::fail(format!("generate: {e}")),
    };

    use base64::Engine as _;
    let engine = base64::engine::general_purpose::STANDARD;

    let sprites: Vec<serde_json::Value> = raw_images.into_iter().map(|img| {
        let snapped = crate::grid::snap_to_grid(&img, 16);
        let quantized = crate::palette::quantize(&snapped, &pal);
        let mut buf = Vec::new();
        let _ = quantized.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        );
        serde_json::json!({
            "png_b64": engine.encode(&buf),
            "width": quantized.width(),
            "height": quantized.height(),
        })
    }).collect();

    PluginResponse::success(serde_json::json!({
        "class": class,
        "count": sprites.len(),
        "sprites": sprites,
    }))
}

/// Run in plugin mode: read one JSON line from stdin, write response to stdout.
pub fn run() -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let response = handle(input.trim());
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// Run in plugin-loop mode: read JSON lines continuously.
/// Kova can keep the process alive and send multiple requests.
pub fn run_loop() -> Result<()> {
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 { break; } // EOF
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let response = handle(trimmed);
        println!("{}", serde_json::to_string(&response)?);
    }
    Ok(())
}
