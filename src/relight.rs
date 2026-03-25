// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Normal-map relighting — generate multi-directional sprite views from SDF + normals.
//! Voxel-engine trick (Enshrouded-style): one sprite + structure map → 4 directional views.
//!
//! Input: original RGBA sprite + structure map (SDF/nx/ny/outline from prep-stages)
//! Output: front, left, right, back relit sprites
//!
//! No model needed — pure math on the normal map.

use image::{RgbaImage, Rgba};

/// Light direction presets for 4-directional sprite sheets.
pub struct LightDir {
    pub name: &'static str,
    /// Normalized (x, y, z) light direction.
    pub dir: [f32; 3],
}

/// Standard 4-direction light set for RPG sprites.
pub fn four_directions() -> Vec<LightDir> {
    vec![
        LightDir { name: "front", dir: normalize([0.0, -0.3, 0.9]) },
        LightDir { name: "left",  dir: normalize([-0.8, -0.2, 0.5]) },
        LightDir { name: "right", dir: normalize([0.8, -0.2, 0.5]) },
        LightDir { name: "back",  dir: normalize([0.0, 0.5, -0.8]) },
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-6 { return [0.0, 0.0, 1.0]; }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Compute SDF, normals, and outline from an RGBA sprite.
/// Returns (sdf, nx, ny, outline) as flat f32 arrays, all in [0,1] or [-1,1].
pub fn compute_structure(img: &RgbaImage) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
    let w = img.width() as usize;
    let h = img.height() as usize;

    // Binary mask
    let mut mask = vec![false; w * h];
    for (x, y, px) in img.enumerate_pixels() {
        if px[3] > 10 {
            mask[y as usize * w + x as usize] = true;
        }
    }

    // SDF via Chamfer distance transform
    let mut dist = vec![f32::MAX; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] {
                dist[i] = 0.0;
            } else {
                let is_edge = (x == 0 || !mask[i - 1])
                    || (x >= w - 1 || !mask[i + 1])
                    || (y == 0 || !mask[i - w])
                    || (y >= h - 1 || !mask[i + w]);
                dist[i] = if is_edge { 0.5 } else { f32::MAX };
            }
        }
    }
    // Forward pass
    for y in 1..h {
        for x in 1..w {
            let i = y * w + x;
            if !mask[i] { continue; }
            let up = dist[(y - 1) * w + x] + 1.0;
            let left = dist[y * w + (x - 1)] + 1.0;
            let diag = dist[(y - 1) * w + (x - 1)] + 1.414;
            dist[i] = dist[i].min(up).min(left).min(diag);
        }
    }
    // Backward pass
    for y in (0..h.saturating_sub(1)).rev() {
        for x in (0..w.saturating_sub(1)).rev() {
            let i = y * w + x;
            if !mask[i] { continue; }
            let down = dist[(y + 1) * w + x] + 1.0;
            let right = dist[y * w + (x + 1)] + 1.0;
            let diag = dist[(y + 1) * w + (x + 1)] + 1.414;
            dist[i] = dist[i].min(down).min(right).min(diag);
        }
    }

    // Normalize SDF
    let max_dist = dist.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    for d in &mut dist {
        *d /= max_dist;
    }

    // Normals from SDF gradient
    let mut nx = vec![0.0f32; w * h];
    let mut ny = vec![0.0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            if !mask[i] { continue; }
            let gx = dist[i + 1] - dist[i - 1];
            let gy = dist[i + w] - dist[i - w];
            let len = (gx * gx + gy * gy).sqrt().max(1e-6);
            nx[i] = gx / len;
            ny[i] = gy / len;
        }
    }

    // Outline
    let mut outline = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] { continue; }
            let has_bg = (x > 0 && !mask[i - 1])
                || (x < w - 1 && !mask[i + 1])
                || (y > 0 && !mask[i - w])
                || (y < h - 1 && !mask[i + w]);
            outline[i] = has_bg;
        }
    }

    (dist, nx, ny, outline)
}

/// Relight a sprite using its normal map and a new light direction.
/// Outline pixels stay dark (they're ink lines, not surfaces).
/// Ambient = 0.3, diffuse = 0.7.
pub fn relight(
    sprite: &RgbaImage,
    sdf: &[f32],
    nx: &[f32],
    ny: &[f32],
    outline: &[bool],
    light: &[f32; 3],
) -> RgbaImage {
    let w = sprite.width();
    let h = sprite.height();
    let mut out = RgbaImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let px = sprite.get_pixel(x, y);
            if px[3] <= 10 {
                out.put_pixel(x, y, Rgba([0, 0, 0, 0]));
                continue;
            }

            let i = y as usize * w as usize + x as usize;

            // Outline pixels stay as-is (dark ink lines)
            if outline[i] {
                out.put_pixel(x, y, *px);
                continue;
            }

            // Normal: (nx, ny, sdf as nz proxy)
            let nz = sdf[i];
            let n_len = (nx[i] * nx[i] + ny[i] * ny[i] + nz * nz).sqrt().max(1e-6);
            let n = [nx[i] / n_len, ny[i] / n_len, nz / n_len];

            // Lambertian diffuse
            let dot = n[0] * light[0] + n[1] * light[1] + n[2] * light[2];
            let shade = 0.3 + 0.7 * dot.max(0.0);

            let r = (px[0] as f32 * shade).clamp(0.0, 255.0) as u8;
            let g = (px[1] as f32 * shade).clamp(0.0, 255.0) as u8;
            let b = (px[2] as f32 * shade).clamp(0.0, 255.0) as u8;
            out.put_pixel(x, y, Rgba([r, g, b, px[3]]));
        }
    }
    out
}

/// Generate a 4-direction sprite sheet from a single sprite.
/// Returns (sheet_image, individual_sprites).
pub fn four_dir_sheet(sprite: &RgbaImage) -> (RgbaImage, Vec<(String, RgbaImage)>) {
    let (sdf, nx, ny, outline) = compute_structure(sprite);
    let dirs = four_directions();

    let w = sprite.width();
    let h = sprite.height();
    let mut sheet = RgbaImage::new(w * 4, h);
    let mut sprites = Vec::new();

    for (i, ld) in dirs.iter().enumerate() {
        let relit = relight(sprite, &sdf, &nx, &ny, &outline, &ld.dir);

        // Mirror left/right views for proper facing
        let final_sprite = if ld.name == "left" {
            image::imageops::flip_horizontal(&relit)
        } else {
            relit
        };

        image::imageops::overlay(&mut sheet, &final_sprite, (i as u32 * w) as i64, 0);
        sprites.push((ld.name.to_string(), final_sprite));
    }

    (sheet, sprites)
}
