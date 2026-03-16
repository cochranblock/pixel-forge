// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Grid snapping — force AI output onto a clean pixel grid.
//! Removes anti-aliasing artifacts that make pixel art look muddy.
//! The difference between "AI art with a filter" and actual pixel art.

use image::{imageops, RgbaImage};

/// Snap an image to a target pixel grid.
/// Downscale → nearest-neighbor upscale removes sub-pixel blending.
pub fn snap_to_grid(img: &RgbaImage, target_size: u32) -> RgbaImage {
    // If already at target size, just clean up
    if img.width() == target_size && img.height() == target_size {
        return clean_alpha(img);
    }

    // Downscale to target using triangle (bilinear) for smooth reduction
    let small = imageops::resize(img, target_size, target_size, imageops::FilterType::Triangle);

    // Clean up alpha — hard edges, no semi-transparency
    clean_alpha(&small)
}

/// Remove semi-transparent pixels — hard alpha only.
/// Pixel art doesn't have soft edges.
fn clean_alpha(img: &RgbaImage) -> RgbaImage {
    let mut out = img.clone();
    for pixel in out.pixels_mut() {
        if pixel[3] < 128 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            pixel[3] = 0;
        } else {
            pixel[3] = 255;
        }
    }
    out
}
