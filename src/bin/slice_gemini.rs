// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Slice Gemini-generated sprite sheets into 32x32 tiles.
//!
//! Gemini outputs large sheets (e.g. 2816×1536 or 2048×2048) with a grid of sprites
//! separated by white borders. This tool:
//!  1. Detects white separator rows/columns (>95% near-white pixels)
//!  2. Finds content bands between separators
//!  3. Assumes 5 cols × 6 rows layout (as prompted)
//!  4. For each cell, crops to non-white bounding box
//!  5. Pads to square, resizes to 32×32 nearest-neighbor
//!  6. Makes white background transparent
//!  7. Saves as RGBA PNG
//!
//! Usage: slice-gemini <input-sheet.png> <output-dir> <class-name>

use image::{GenericImageView, Rgba, RgbaImage};
use std::path::Path;

const WHITE_THRESHOLD: u8 = 240;
const GAP_RATIO: f32 = 0.95;
const TARGET_SIZE: u32 = 32;
const EXPECTED_COLS: usize = 5;
const EXPECTED_ROWS: usize = 6;

fn is_whiteish(px: &Rgba<u8>) -> bool {
    px[3] > 0 && px[0] > WHITE_THRESHOLD && px[1] > WHITE_THRESHOLD && px[2] > WHITE_THRESHOLD
}

fn is_transparent(px: &Rgba<u8>) -> bool {
    px[3] == 0
}

fn is_background(px: &Rgba<u8>) -> bool {
    is_whiteish(px) || is_transparent(px)
}

/// Find the content bands along an axis.
/// Returns (start, end) pairs in pixel coords, where each band is mostly non-white.
fn find_bands(profile: &[f32], expected: usize) -> Vec<(u32, u32)> {
    let n = profile.len();
    let mut bands: Vec<(u32, u32)> = Vec::new();
    let mut in_band = false;
    let mut start = 0u32;

    for (i, &r) in profile.iter().enumerate() {
        let content = r < GAP_RATIO;
        if content && !in_band {
            start = i as u32;
            in_band = true;
        } else if !content && in_band {
            bands.push((start, i as u32));
            in_band = false;
        }
    }
    if in_band {
        bands.push((start, n as u32));
    }

    // Filter out very thin bands (noise from jpg-like artifacts)
    let min_width = (n / (expected * 4)).max(4) as u32;
    bands.retain(|(s, e)| e - s > min_width);

    // If we got more bands than expected, merge adjacent ones
    while bands.len() > expected {
        // Find the smallest gap between consecutive bands and merge
        let mut min_gap_idx = 0;
        let mut min_gap = u32::MAX;
        for i in 0..bands.len() - 1 {
            let gap = bands[i + 1].0.saturating_sub(bands[i].1);
            if gap < min_gap {
                min_gap = gap;
                min_gap_idx = i;
            }
        }
        let (s1, _) = bands[min_gap_idx];
        let (_, e2) = bands[min_gap_idx + 1];
        bands[min_gap_idx] = (s1, e2);
        bands.remove(min_gap_idx + 1);
    }

    bands
}

fn build_profiles(img: &RgbaImage) -> (Vec<f32>, Vec<f32>) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let mut row_white = vec![0u32; h];
    let mut col_white = vec![0u32; w];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            if is_background(px) {
                row_white[y] += 1;
                col_white[x] += 1;
            }
        }
    }

    let row_profile: Vec<f32> = row_white.iter().map(|&c| c as f32 / w as f32).collect();
    let col_profile: Vec<f32> = col_white.iter().map(|&c| c as f32 / h as f32).collect();
    (row_profile, col_profile)
}

/// Crop to the bounding box of non-background pixels.
fn crop_to_content(cell: &RgbaImage) -> Option<RgbaImage> {
    let w = cell.width();
    let h = cell.height();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut any = false;

    for y in 0..h {
        for x in 0..w {
            let px = cell.get_pixel(x, y);
            if !is_background(px) {
                any = true;
                if x < min_x { min_x = x; }
                if y < min_y { min_y = y; }
                if x > max_x { max_x = x; }
                if y > max_y { max_y = y; }
            }
        }
    }
    if !any { return None; }
    let bw = max_x - min_x + 1;
    let bh = max_y - min_y + 1;
    Some(cell.view(min_x, min_y, bw, bh).to_image())
}

/// Pad image to square (centered), fills extra with transparent.
fn pad_to_square(img: &RgbaImage) -> RgbaImage {
    let w = img.width();
    let h = img.height();
    let side = w.max(h);
    let mut out = RgbaImage::from_pixel(side, side, Rgba([0, 0, 0, 0]));
    let ox = (side - w) / 2;
    let oy = (side - h) / 2;
    for y in 0..h {
        for x in 0..w {
            out.put_pixel(ox + x, oy + y, *img.get_pixel(x, y));
        }
    }
    out
}

/// Replace whiteish pixels with transparent.
fn white_to_alpha(img: &mut RgbaImage) {
    for px in img.pixels_mut() {
        if is_whiteish(px) {
            *px = Rgba([0, 0, 0, 0]);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: slice-gemini <input.png> <output-dir> <class-name> [start-index]");
        std::process::exit(1);
    }
    let input = &args[1];
    let out_dir = &args[2];
    let class_name = &args[3];
    let start_idx: usize = if args.len() > 4 { args[4].parse()? } else { 0 };

    let out_path = Path::new(out_dir);
    std::fs::create_dir_all(out_path)?;

    let mut img = image::open(input)?.to_rgba8();

    // Pre-clean: convert any alpha=0 pixel and whiteish pixel to transparent
    white_to_alpha(&mut img);

    let (row_p, col_p) = build_profiles(&img);
    let row_bands = find_bands(&row_p, EXPECTED_ROWS);
    let col_bands = find_bands(&col_p, EXPECTED_COLS);

    println!("{}: {}×{} → {} rows × {} cols detected",
        Path::new(input).file_name().unwrap().to_string_lossy(),
        img.width(), img.height(), row_bands.len(), col_bands.len());

    if row_bands.len() < 2 || col_bands.len() < 2 {
        eprintln!("  FAIL: not enough bands detected — skipping");
        return Ok(());
    }

    let mut saved = 0u32;
    let mut skipped = 0u32;

    for (r, (ys, ye)) in row_bands.iter().enumerate() {
        for (c, (xs, xe)) in col_bands.iter().enumerate() {
            let cw = xe - xs;
            let ch = ye - ys;
            if cw < 8 || ch < 8 { continue; }
            let cell = img.view(*xs, *ys, cw, ch).to_image();

            let cropped = match crop_to_content(&cell) {
                Some(c) => c,
                None => { skipped += 1; continue; }
            };

            let squared = pad_to_square(&cropped);
            let resized = image::imageops::resize(
                &squared, TARGET_SIZE, TARGET_SIZE,
                image::imageops::FilterType::Nearest,
            );

            let idx = start_idx + r * col_bands.len() + c;
            let fname = format!("{}/gemini_{}_{:04}.png", out_dir, class_name, idx);
            resized.save(&fname)?;
            saved += 1;
        }
    }

    println!("  saved {} tiles, skipped {} (empty)", saved, skipped);
    Ok(())
}
