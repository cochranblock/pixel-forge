// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! f16 quantization for model weights.
//! Halves file size and memory bandwidth. ARM NEON and Apple Metal
//! handle f16 natively — free speedup on mobile.
//!
//! Usage: `pixel-forge quantize model.safetensors -o model-f16.safetensors`

use anyhow::Result;
use std::collections::HashMap;


/// Read a safetensors file header and return (header_json, raw_data).
fn read_safetensors(path: &str) -> Result<(serde_json::Value, Vec<u8>)> {
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        anyhow::bail!("file too small: {}", path);
    }
    let header_len = u64::from_le_bytes(data[..8].try_into()?) as usize;
    let header_str = std::str::from_utf8(&data[8..8 + header_len])?;
    let header: serde_json::Value = serde_json::from_str(header_str)?;
    Ok((header, data))
}

/// Detect the primary dtype used in a safetensors file.
pub fn detect_dtype(path: &str) -> Result<String> {
    let (header, _) = read_safetensors(path)?;
    let obj = header.as_object().ok_or_else(|| anyhow::anyhow!("bad header"))?;
    for (key, val) in obj {
        if key == "__metadata__" { continue; }
        if let Some(dtype) = val.get("dtype").and_then(|d| d.as_str()) {
            return Ok(dtype.to_string());
        }
    }
    Ok("F32".to_string())
}

/// Quantize a safetensors file from f32 to f16.
/// Returns the output file size in bytes.
pub fn quantize_f32_to_f16(input: &str, output: &str) -> Result<u64> {
    let (header, data) = read_safetensors(input)?;
    let header_len = u64::from_le_bytes(data[..8].try_into()?) as usize;
    let tensor_data = &data[8 + header_len..];

    let obj = header.as_object().ok_or_else(|| anyhow::anyhow!("bad header"))?;

    // First pass: convert all f32 tensors to f16 and build new offsets
    struct TensorInfo {
        name: String,
        shape: Vec<u64>,
        f16_bytes: Vec<u8>,
    }

    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut names: Vec<String> = obj.keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect();
    names.sort(); // deterministic order

    for name in &names {
        let info = &obj[name];
        let dtype = info.get("dtype").and_then(|d| d.as_str()).unwrap_or("F32");
        let shape: Vec<u64> = info.get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        let offsets = info.get("data_offsets")
            .and_then(|o| o.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>())
            .unwrap_or_default();

        if offsets.len() != 2 {
            anyhow::bail!("tensor {} missing data_offsets", name);
        }

        let start = offsets[0] as usize;
        let end = offsets[1] as usize;
        let raw = &tensor_data[start..end];

        let f16_bytes = match dtype {
            "F32" => {
                // Convert f32 → f16
                let f32_count = raw.len() / 4;
                let mut out = Vec::with_capacity(f32_count * 2);
                for i in 0..f32_count {
                    let val = f32::from_le_bytes([raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]]);
                    let f16_val = half::f16::from_f32(val);
                    out.extend_from_slice(&f16_val.to_le_bytes());
                }
                out
            }
            "F16" => {
                // Already f16, pass through
                raw.to_vec()
            }
            _ => {
                // Non-float tensors (unlikely but keep them as-is)
                raw.to_vec()
            }
        };

        tensors.push(TensorInfo {
            name: name.clone(),
            shape,
            f16_bytes,
        });
    }

    // Build new header with updated offsets and dtypes
    let mut new_header: HashMap<String, serde_json::Value> = HashMap::new();
    let mut offset = 0u64;

    for t in &tensors {
        let end = offset + t.f16_bytes.len() as u64;
        let tensor_meta = serde_json::json!({
            "dtype": "F16",
            "shape": t.shape,
            "data_offsets": [offset, end],
        });
        new_header.insert(t.name.clone(), tensor_meta);
        offset = end;
    }

    // Preserve metadata
    if let Some(meta) = obj.get("__metadata__") {
        new_header.insert("__metadata__".to_string(), meta.clone());
    }

    // Serialize
    let header_json = serde_json::to_string(&new_header)?;
    let header_bytes = header_json.as_bytes();
    let header_len_bytes = (header_bytes.len() as u64).to_le_bytes();

    // Write output file
    let mut out = Vec::new();
    out.extend_from_slice(&header_len_bytes);
    out.extend_from_slice(header_bytes);
    for t in &tensors {
        out.extend_from_slice(&t.f16_bytes);
    }

    std::fs::write(output, &out)?;
    Ok(out.len() as u64)
}

/// Check if a safetensors file is already f16.
pub fn is_f16(path: &str) -> bool {
    detect_dtype(path).map(|d| d == "F16").unwrap_or(false)
}

/// Return the candle DType matching a safetensors file.
pub fn candle_dtype_for(_path: &str) -> candle_core::DType {
    // Always compute in f32 — f16 is for storage only.
    // On CPU, f16 compute is slow (no NEON in candle).
    // On Metal, the cast overhead is negligible vs download savings.
    candle_core::DType::F32
}

/// Load a safetensors file into a VarMap, casting f16→f32 if needed.
pub fn load_varmap(varmap: &mut candle_nn::VarMap, path: &str) -> Result<()> {
    use candle_core::DType;

    if is_f16(path) {
        // Load raw tensors, cast f16→f32, then set in varmap
        let tensors = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if let Some(src) = tensors.get(name) {
                let src_f32 = if src.dtype() == DType::F16 {
                    src.to_dtype(DType::F32)?
                } else {
                    src.clone()
                };
                var.set(&src_f32.to_device(var.device())?)?;
            }
        }
        Ok(())
    } else {
        varmap.load(path)?;
        Ok(())
    }
}

/// Quantize in place — writes f16 version next to the original.
/// Returns the path to the f16 file.
pub fn quantize_in_place(path: &str) -> Result<String> {
    if is_f16(path) {
        println!("already f16: {path}");
        return Ok(path.to_string());
    }

    let f16_path = if path.ends_with(".safetensors") {
        format!("{}-f16.safetensors", path.trim_end_matches(".safetensors"))
    } else {
        format!("{}-f16", path)
    };

    let input_size = std::fs::metadata(path)?.len();
    let output_size = quantize_f32_to_f16(path, &f16_path)?;

    let ratio = output_size as f64 / input_size as f64;
    println!("{} ({:.1} MB) → {} ({:.1} MB) [{:.0}% of original]",
        path, input_size as f64 / 1_048_576.0,
        f16_path, output_size as f64 / 1_048_576.0,
        ratio * 100.0,
    );

    Ok(f16_path)
}
