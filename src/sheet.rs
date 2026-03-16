// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Sprite sheet packing — arrange frames into atlas images.

use image::RgbaImage;

/// Pack frames into a horizontal strip (standard animation sheet).
pub fn pack_horizontal(frames: &[RgbaImage]) -> RgbaImage {
    if frames.is_empty() {
        return RgbaImage::new(1, 1);
    }

    let w = frames[0].width();
    let h = frames[0].height();
    let total_width = w * frames.len() as u32;

    let mut sheet = RgbaImage::new(total_width, h);
    for (i, frame) in frames.iter().enumerate() {
        image::imageops::overlay(&mut sheet, frame, (i as u32 * w) as i64, 0);
    }
    sheet
}

/// Pack tiles into a grid (e.g., 4 columns for a 16-tile tileset).
pub fn pack_grid(tiles: &[RgbaImage], columns: u32) -> RgbaImage {
    if tiles.is_empty() {
        return RgbaImage::new(1, 1);
    }

    let w = tiles[0].width();
    let h = tiles[0].height();
    let rows = (tiles.len() as u32 + columns - 1) / columns;
    let total_width = w * columns;
    let total_height = h * rows;

    let mut sheet = RgbaImage::new(total_width, total_height);
    for (i, tile) in tiles.iter().enumerate() {
        let col = i as u32 % columns;
        let row = i as u32 / columns;
        image::imageops::overlay(&mut sheet, tile, (col * w) as i64, (row * h) as i64);
    }
    sheet
}
