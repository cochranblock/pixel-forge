// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Sonnet 4.6
//! PaletteNet — tiny MLP predicting class-appropriate palette colors.
//!
//! ~100K params. Supervised training from K-means(8) dominant colors
//! extracted per class from the training dataset.
//!
//! Input:  super_id (u32) + tags ([f32; 12])
//! Output: palette_colors × (R, G, B) in [0.0, 1.0], sorted by brightness.
//!
//! Usage in tiered pipeline: predict 8 colors for a class, quantize them
//! to nearest endesga32 entries, and use that reduced palette for output
//! quantization — enforcing class-faithful color selection.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{self as nn, optim::Optimizer, VarBuilder, VarMap};

use crate::class_cond::{NUM_SUPER_WITH_NULL, NUM_TAGS};
use crate::palette::{self, Color};

const DIM: usize = 64;
const MID: usize = 256;

/// PaletteNet: (super_id, tags) → N palette colors.
pub struct PaletteNet {
    embed: nn::Embedding,
    tag_proj: nn::Linear,
    mlp1: nn::Linear,
    mlp2: nn::Linear,
    pub palette_colors: usize,
}

impl PaletteNet {
    pub fn new(vb: VarBuilder, palette_colors: usize) -> candle_core::Result<Self> {
        Ok(Self {
            embed:    nn::embedding(NUM_SUPER_WITH_NULL, DIM, vb.pp("embed"))?,
            tag_proj: nn::linear(NUM_TAGS, DIM, vb.pp("tag_proj"))?,
            mlp1:     nn::linear(DIM * 2, MID, vb.pp("mlp1"))?,
            mlp2:     nn::linear(MID, palette_colors * 3, vb.pp("mlp2"))?,
            palette_colors,
        })
    }

    /// Forward pass: predict palette colors for a batch.
    ///
    /// super_id: (B,) u32 tensor
    /// tags: (B, NUM_TAGS) f32 tensor
    /// Returns: (B, palette_colors, 3) f32 tensor in [0.0, 1.0]
    pub fn forward(&self, super_id: &Tensor, tags: &Tensor) -> candle_core::Result<Tensor> {
        let emb = self.embed.forward(super_id)?;                 // (B, DIM)
        let tag_emb = self.tag_proj.forward(tags)?;              // (B, DIM)
        let combined = Tensor::cat(&[&emb, &tag_emb], 1)?;       // (B, DIM*2)
        let h = nn::ops::silu(&self.mlp1.forward(&combined)?)?;  // (B, MID)
        let out = nn::ops::sigmoid(&self.mlp2.forward(&h)?)?;      // (B, palette_colors*3)
        let b = out.dim(0)?;
        out.reshape((b, self.palette_colors, 3))
    }

    /// Predict palette colors for a single class, quantized to the given palette.
    ///
    /// Returns `palette_colors` entries from `base_palette`, closest to predictions.
    /// Result is sorted by perceived brightness (dark → light).
    pub fn predict_palette(
        &self,
        super_id: u32,
        tags: &[f32; NUM_TAGS],
        base_palette: &[Color],
        device: &Device,
    ) -> candle_core::Result<Vec<Color>> {
        let super_t = Tensor::new(&[super_id], device)?;
        let tags_t = Tensor::new(tags.as_slice(), device)?.reshape((1, NUM_TAGS))?;
        let pred = self.forward(&super_t, &tags_t)?; // (1, palette_colors, 3)
        let pred = pred.squeeze(0)?;                  // (palette_colors, 3)
        let pred_vec = pred.flatten_all()?.to_vec1::<f32>()?;

        let n = self.palette_colors;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let r = (pred_vec[i * 3]     * 255.0) as u8;
            let g = (pred_vec[i * 3 + 1] * 255.0) as u8;
            let b = (pred_vec[i * 3 + 2] * 255.0) as u8;
            // Find nearest base_palette entry
            let nearest = find_nearest_color(r, g, b, base_palette);
            result.push(nearest);
        }

        // Deduplicate and sort by brightness
        result.sort_by(|a, b| {
            let ba = perceived_brightness(a);
            let bb = perceived_brightness(b);
            ba.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        });
        result.dedup_by(|a, b| a == b);

        Ok(result)
    }
}

fn perceived_brightness(c: &Color) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

fn find_nearest_color(r: u8, g: u8, b: u8, palette: &[Color]) -> Color {
    palette.iter().copied()
        .min_by(|&ca, &cb| {
            let da = (ca[0] as i32 - r as i32).pow(2) * 9
                   + (ca[1] as i32 - g as i32).pow(2) * 16
                   + (ca[2] as i32 - b as i32).pow(2) * 4;
            let db = (cb[0] as i32 - r as i32).pow(2) * 9
                   + (cb[1] as i32 - g as i32).pow(2) * 16
                   + (cb[2] as i32 - b as i32).pow(2) * 4;
            da.cmp(&db)
        })
        .unwrap_or([0, 0, 0, 255])
}

/// Load a trained PaletteNet from a safetensors file.
pub fn load(path: &str, palette_colors: usize, device: &Device) -> Result<PaletteNet> {
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, device);
    let net = PaletteNet::new(vb, palette_colors)?;
    crate::quantize::load_varmap(&mut vm, path)?;
    Ok(net)
}

// ─── K-means ─────────────────────────────────────────────────────────────────

/// Run K-means clustering on RGB pixels. Returns k centroids.
/// pixels: interleaved [R, G, B, R, G, B, ...] in [0.0, 1.0]
fn kmeans(pixels: &[[f32; 3]], k: usize, n_iter: usize) -> Vec<[f32; 3]> {
    if pixels.is_empty() || k == 0 {
        return vec![[0.5; 3]; k];
    }
    let n = pixels.len();

    // Evenly-spaced initialization (more stable than pure random for small k)
    let mut centers: Vec<[f32; 3]> = (0..k)
        .map(|i| pixels[(i * n / k).min(n - 1)])
        .collect();

    for _ in 0..n_iter {
        let mut sums = vec![[0.0f32; 3]; k];
        let mut counts = vec![0usize; k];

        for &px in pixels {
            let nearest = centers.iter().enumerate()
                .min_by(|(_, ca), (_, cb)| {
                    let da = color_dist_sq(px, **ca);
                    let db = color_dist_sq(px, **cb);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            sums[nearest][0] += px[0];
            sums[nearest][1] += px[1];
            sums[nearest][2] += px[2];
            counts[nearest] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let c = counts[i] as f32;
                centers[i] = [sums[i][0] / c, sums[i][1] / c, sums[i][2] / c];
            }
        }
    }

    centers
}

#[inline]
fn color_dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)
}

/// Convert (super_id, tags) to a compact group key.
fn group_key(super_id: u32, tags: &[f32; NUM_TAGS]) -> u32 {
    let tag_bits: u32 = tags.iter().enumerate()
        .fold(0u32, |acc, (i, &t)| if t > 0.5 { acc | (1 << i) } else { acc });
    (super_id << 16) | (tag_bits & 0xFFFF)
}

// ─── Training ────────────────────────────────────────────────────────────────

/// Train PaletteNet from the pixel-forge training dataset.
///
/// For each unique (super_id, tags) combination found in the dataset,
/// computes K-means(palette_colors) on all pixel values from that group,
/// then trains the MLP to predict those dominant colors.
pub fn train_palette_net(
    data_dir: &str,
    output_path: &str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    palette_colors: usize,
) -> Result<()> {
    use std::collections::HashMap;
    use candle_core::Device;

    let device = Device::Cpu; // PaletteNet is tiny; CPU is fine

    println!("palette-net: loading dataset from {data_dir}...");
    let dataset = crate::train::preprocess(data_dir, 32)?;
    let n_samples = dataset.super_ids.len();
    let stride = dataset.stride; // 3 * 32 * 32 = 3072
    let n_px = (stride / 3) as usize; // 1024 pixels per sample
    println!("palette-net: {n_samples} samples, stride={stride}");

    // Group pixel triplets by (super_id, tags)
    let mut groups: HashMap<u32, (u32, [f32; NUM_TAGS], Vec<[f32; 3]>)> = HashMap::new();

    for s in 0..n_samples {
        let super_id = dataset.super_ids[s];
        let tags = dataset.tags[s];
        let key = group_key(super_id, &tags);
        let entry = groups.entry(key).or_insert_with(|| (super_id, tags, Vec::new()));
        let base = s * stride;
        // PackedDataset: channel-first within each sample → convert to pixel triples
        for p in 0..n_px {
            entry.2.push([
                dataset.pixels[base + p],
                dataset.pixels[base + n_px + p],
                dataset.pixels[base + 2 * n_px + p],
            ]);
        }
    }

    println!("palette-net: {} unique class groups", groups.len());

    // Compute K-means targets per group
    type TrainPair = (u32, [f32; NUM_TAGS], Vec<[f32; 3]>); // (super_id, tags, sorted_palette)
    let mut train_pairs: Vec<TrainPair> = Vec::new();

    for (_, (super_id, tags, pixels)) in &groups {
        let mut centers = kmeans(pixels, palette_colors, 20);
        // Sort by perceived brightness (dark → light)
        centers.sort_by(|a, b| {
            let ba = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2];
            let bb = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2];
            ba.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        });
        train_pairs.push((*super_id, *tags, centers));
    }

    let n_train = train_pairs.len();
    println!("palette-net: {n_train} training pairs  ({palette_colors} colors each)");
    println!("palette-net: training {epochs} epochs, batch={batch_size}, lr={lr:.1e}");

    // Build model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PaletteNet::new(vb, palette_colors)?;
    let mut opt = candle_nn::optim::AdamW::new_lr(varmap.all_vars(), lr)?;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0usize;

        // Mini-batch loop
        let mut start = 0;
        while start < n_train {
            let end = (start + batch_size).min(n_train);
            let batch = &train_pairs[start..end];
            let b = batch.len();

            let super_ids_vec: Vec<u32> = batch.iter().map(|(sid, _, _)| *sid).collect();
            let tags_flat: Vec<f32> = batch.iter()
                .flat_map(|(_, t, _)| t.iter().copied())
                .collect();
            // Target: (B, palette_colors, 3)
            let target_flat: Vec<f32> = batch.iter()
                .flat_map(|(_, _, colors)| colors.iter().flat_map(|c| c.iter().copied()))
                .collect();

            let super_t = Tensor::new(super_ids_vec.as_slice(), &device)?;
            let tags_t = Tensor::new(tags_flat.as_slice(), &device)?
                .reshape((b, NUM_TAGS))?;
            let target_t = Tensor::new(target_flat.as_slice(), &device)?
                .reshape((b, palette_colors, 3))?;

            let pred = model.forward(&super_t, &tags_t)?;
            let loss = (&pred - &target_t)?.sqr()?.mean_all()?;

            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
            n_batches += 1;
            start = end;
        }

        if epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == epochs {
            println!("  epoch {}/{epochs}  loss={:.5}", epoch + 1, epoch_loss / n_batches as f32);
        }
    }

    // Save
    std::fs::create_dir_all(
        std::path::Path::new(output_path).parent().unwrap_or(std::path::Path::new("."))
    )?;
    varmap.save(output_path)?;
    let _ = crate::nanosign::sign_and_log(output_path);
    println!("palette-net: saved → {output_path}");

    // Smoke-test: predict palette for first group
    if let Some((super_id, tags, _)) = train_pairs.first() {
        let base_pal = palette::load_palette("endesga32").unwrap_or_default();
        if let Ok(colors) = model.predict_palette(*super_id, tags, &base_pal, &device) {
            let hex: Vec<String> = colors.iter()
                .map(|c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]))
                .collect();
            println!("palette-net: sample prediction (super_id={super_id}) → {}", hex.join(" "));
        }
    }

    Ok(())
}
