// Unlicense — cochranblock.org
//! Safetensors → flat (name, Vec<f32>, shape) records.
//!
//! Cinder ships f16 (4.2 MB on disk). The browser only deals in f32 today,
//! so we expand to f32 here at load (~8 MB resident). f16 dtype on any-gpu
//! is a future optimization, not a blocker.

use anyhow::{anyhow, Result};
use safetensors::SafeTensors;

/// One unpacked tensor: name, f32 weights, NCHW-style shape (or whatever
/// the safetensors header says — we don't reorder).
pub struct LoadedTensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<u32>,
}

/// Parse a safetensors byte blob into f32 tensors. Accepts F16, BF16, and
/// F32 dtypes; everything else errors.
///
/// NanoSign trailers (36-byte BLAKE3 sigs appended after EOF) are tolerated
/// — `SafeTensors::deserialize` reads only as much as the header declares,
/// so trailing bytes are ignored. We do NOT verify the signature here; the
/// browser is an untrusted environment for a private key. Verification is
/// the desktop binary's job.
pub fn load(bytes: &[u8]) -> Result<Vec<LoadedTensor>> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| anyhow!("safetensors parse: {e}"))?;

    let mut out = Vec::with_capacity(st.tensors().len());
    for (name, view) in st.tensors() {
        let shape: Vec<u32> = view.shape().iter().map(|&d| d as u32).collect();
        let data = match view.dtype() {
            safetensors::Dtype::F32 => bytes_to_f32(view.data()),
            safetensors::Dtype::F16 => f16_bytes_to_f32(view.data()),
            safetensors::Dtype::BF16 => bf16_bytes_to_f32(view.data()),
            other => return Err(anyhow!("tensor {name} has unsupported dtype {:?}", other)),
        };
        out.push(LoadedTensor { name, data, shape });
    }
    Ok(out)
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Look up a tensor by exact name. Used when wiring weights into a model;
/// cheaper than building a HashMap when we visit each tensor once.
pub fn find<'a>(tensors: &'a [LoadedTensor], name: &str) -> Result<&'a LoadedTensor> {
    tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor {name} not in safetensors blob"))
}
