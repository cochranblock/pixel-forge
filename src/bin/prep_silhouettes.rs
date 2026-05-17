// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//! Extract silhouette (alpha-channel binary mask) from RGBA pixel art tiles.
//!
//! Walks <input-dir> recursively, preserving class subdir structure into
//! <output-dir>. For each RGBA PNG, every pixel becomes:
//!   alpha > threshold → (255, 255, 255, 255)  // foreground
//!   alpha ≤ threshold → (  0,   0,   0, 255)  // background
//!
//! Output is RGB-encoded grayscale (R=G=B=mask) so the existing pixel-forge
//! trainer's `to_rgb8()` load path picks it up unchanged. Used as the training
//! target for Anvil-outline (silhouette specialist) in the tier-roles cascade.
//!
//! Usage: prep-silhouettes <input-dir> <output-dir>

use std::fs;
use std::path::{Path, PathBuf};

const ALPHA_THRESHOLD: u8 = 16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: prep-silhouettes <input-dir> <output-dir>");
        std::process::exit(1);
    }
    let input_dir = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);

    if !input_dir.is_dir() {
        eprintln!("error: input dir does not exist: {}", input_dir.display());
        std::process::exit(1);
    }
    fs::create_dir_all(&output_dir)?;

    let pngs = walk_pngs(&input_dir);
    println!("found {} PNGs under {}", pngs.len(), input_dir.display());

    let t0 = std::time::Instant::now();
    let mut converted = 0u32;
    let mut skipped = 0u32;
    let mut classes_seen = std::collections::HashSet::new();

    for src in &pngs {
        let rel = src.strip_prefix(&input_dir)?;
        let dst = output_dir.join(rel);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
            if let Some(class) = parent.file_name() {
                classes_seen.insert(class.to_owned());
            }
        }

        match extract_silhouette(src, &dst) {
            Ok(()) => converted += 1,
            Err(e) => {
                eprintln!("  skip {}: {}", src.display(), e);
                skipped += 1;
            }
        }

        if converted % 1000 == 0 && converted > 0 {
            println!("  {converted}/{} ({:.1}s)", pngs.len(), t0.elapsed().as_secs_f32());
        }
    }

    println!(
        "done: {converted} converted, {skipped} skipped, {} classes, {:.1}s total",
        classes_seen.len(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

fn walk_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_pngs_into(dir, &mut out);
    out.sort();
    out
}

fn walk_pngs_into(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_pngs_into(&path, out);
        } else if path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("png")).unwrap_or(false) {
            out.push(path);
        }
    }
}

fn extract_silhouette(src: &Path, dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open(src)?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let px = rgba.get_pixel(x, y);
            let foreground = px[3] > ALPHA_THRESHOLD;
            let v: u8 = if foreground { 255 } else { 0 };
            out.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    out.save(dst)?;
    Ok(())
}
