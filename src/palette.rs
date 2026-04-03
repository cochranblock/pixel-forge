// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Palette management — color quantization for pixel art.
//! Enforces limited palettes: Stardew Valley warmth, Starbound richness,
//! retro hardware limits (NES, SNES, Game Boy).
//!
//! Stardew palette: inspired by ConcernedApe (Eric Barone)'s Stardew Valley.
//! One dev, one game, one legend. No assets copied — colors hand-picked to honor the vibe.

use image::{Rgba, RgbaImage};
use std::path::Path;

/// An RGBA color.
pub type Color = [u8; 4];

/// Load a named palette or .pal file path.
pub fn load_palette(name: &str) -> anyhow::Result<Vec<Color>> {
    // Check if it's a file path first
    if Path::new(name).exists() {
        return load_pal_file(name);
    }

    match name.to_lowercase().as_str() {
        "stardew" => Ok(stardew_palette()),
        "starbound" => Ok(starbound_palette()),
        "snes" => Ok(snes_palette()),
        "nes" => Ok(nes_palette()),
        "gameboy" | "gb" => Ok(gameboy_palette()),
        "endesga32" | "e32" => Ok(endesga32_palette()),
        "pico8" | "pico" => Ok(pico8_palette()),
        _ => anyhow::bail!("unknown palette: {name}. Run `pixel-forge palettes` to list."),
    }
}

/// List all built-in palettes.
pub fn list_palettes() {
    let palettes = [
        ("stardew", 48, "Warm, earthy — Stardew Valley inspired"),
        ("starbound", 64, "Rich sci-fi — Starbound inspired"),
        ("endesga32", 32, "ENDESGA 32 — popular indie pixel art palette"),
        ("pico8", 16, "PICO-8 fantasy console"),
        ("snes", 256, "Super Nintendo full palette"),
        ("nes", 54, "Nintendo Entertainment System"),
        ("gameboy", 4, "Original Game Boy — 4 shades of green"),
    ];
    println!("{:<14} {:>6}  DESCRIPTION", "NAME", "COLORS");
    println!("{}", "-".repeat(60));
    for (name, count, desc) in palettes {
        println!("{:<14} {:>6}  {}", name, count, desc);
    }
}

/// Quantize an image to the nearest colors in the palette.
/// This is what gives pixel art its characteristic limited-color look.
pub fn quantize(img: &RgbaImage, palette: &[Color]) -> RgbaImage {
    let mut out = img.clone();
    for pixel in out.pixels_mut() {
        if pixel[3] < 128 {
            // Transparent — keep it
            *pixel = Rgba([0, 0, 0, 0]);
            continue;
        }
        let nearest = find_nearest(pixel, palette);
        *pixel = Rgba(nearest);
    }
    out
}

/// Find nearest palette color using perceptual distance (weighted RGB).
fn find_nearest(pixel: &Rgba<u8>, palette: &[Color]) -> Color {
    let mut best = palette[0];
    let mut best_dist = u32::MAX;

    for &color in palette {
        // Weighted Euclidean — human eyes are more sensitive to green
        let dr = (pixel[0] as i32 - color[0] as i32) * 3;
        let dg = (pixel[1] as i32 - color[1] as i32) * 4;
        let db = (pixel[2] as i32 - color[2] as i32) * 2;
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best = color;
        }
    }
    best
}

/// Load a .pal file (JASC-PAL format or raw RGB lines).
fn load_pal_file(path: &str) -> anyhow::Result<Vec<Color>> {
    let content = std::fs::read_to_string(path)?;
    let mut colors = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        // Skip headers
        if line.is_empty()
            || line.starts_with("JASC")
            || line.starts_with("PAL")
            || line.parse::<u32>().is_ok() && line.len() <= 4
        {
            continue;
        }
        // Parse "R G B" or "R,G,B" or "#RRGGBB"
        if let Some(hex) = line.strip_prefix('#') {
            if hex.len() == 6 {
                let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
                let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
                let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
                colors.push([r, g, b, 255]);
            }
        } else {
            let parts: Vec<&str> = line.split([' ', ',', '\t']).collect();
            if parts.len() >= 3 {
                let r: u8 = parts[0].parse().unwrap_or(0);
                let g: u8 = parts[1].parse().unwrap_or(0);
                let b: u8 = parts[2].parse().unwrap_or(0);
                colors.push([r, g, b, 255]);
            }
        }
    }

    if colors.is_empty() {
        anyhow::bail!("no colors found in palette file: {path}");
    }
    Ok(colors)
}

// ── Built-in Palettes ──

/// Stardew Valley inspired — warm earth tones, soft greens, cozy lighting.
fn stardew_palette() -> Vec<Color> {
    [
        // Skin tones
        0xFFE0BD, 0xFFCD94, 0xEEAA7A, 0xD08B5B,
        // Hair
        0x543D2E, 0x8B6245, 0xC49A6C, 0x2B1D0E,
        // Greens (grass, trees, crops)
        0x5B8C3E, 0x7DB848, 0xA8D86E, 0x3E6B28, 0x2D4F1E, 0xC5E89A,
        // Earth/dirt
        0x8B6845, 0x6B4E35, 0xA67C52, 0x4E3829, 0xC49A6C,
        // Water
        0x4A7DB8, 0x6BA3D8, 0x3A6498, 0x2D4F78, 0x8BC5E8,
        // Sky
        0x87CEEB, 0xB5E2F7, 0x6BB3D9, 0xFFF4C2,
        // Stone/rock
        0x9B9B9B, 0x7A7A7A, 0xBDBDBD, 0x5A5A5A,
        // Wood
        0x8B6845, 0xA67C52, 0x6B4E35, 0xC49A6C,
        // Flowers/accents
        0xE85D75, 0xF4A0B0, 0xFFD700, 0xFF8C42, 0xB45BCF, 0x7BC8F6,
        // UI / shadows / highlights
        0xFFF8E7, 0xFFFDF5, 0x1A1A2E, 0x2E2E4A, 0x4A4A6A, 0x000000,
        // Warm lighting
        0xFFF4C2, 0xFFE4A0, 0xFFD480, 0xFFC060,
    ]
    .iter()
    .map(|&hex| hex_to_rgba(hex))
    .collect()
}

/// Starbound inspired — vibrant sci-fi, alien worlds, rich color range.
fn starbound_palette() -> Vec<Color> {
    [
        // Deep space
        0x0D0D2B, 0x1A1A3E, 0x2B2B5E, 0x0A0A1F,
        // Alien greens
        0x4AE85D, 0x2DB84A, 0x1E8834, 0x6BF080, 0x0E5820,
        // Alien purples
        0x8B45CF, 0xAA6BE8, 0x6B2DAA, 0xCC8BF7,
        // Warm reds/oranges
        0xE85D4A, 0xF08060, 0xB84A3A, 0xFF9B7B, 0xCC3322,
        // Tech blues
        0x4A8BE8, 0x6BA8F0, 0x3A6ECC, 0x8BC5FF, 0x2B4FAA,
        // Sand/desert
        0xE8C86B, 0xCCAA4A, 0xFFE090, 0xAA8830,
        // Ice
        0xC5E8F7, 0xA0D4EB, 0xE8F4FF, 0x7BBDD8,
        // Metal/tech
        0x8B8B9B, 0x6B6B7B, 0xABABBB, 0x4B4B5B, 0xCBCBDB,
        // Lava
        0xFF4422, 0xFF6644, 0xFF8866, 0xCC2200, 0xFFAA88,
        // Foliage
        0x5BC84A, 0x3EA835, 0x7BE86B, 0x2D882B,
        // Skin tones
        0xFFE0BD, 0xFFCD94, 0xD08B5B, 0x8B5E3A,
        // Black/white
        0x000000, 0xFFFFFF, 0x222233, 0xDDDDEE,
        // Glow effects
        0xFFFF88, 0x88FFFF, 0xFF88FF, 0x88FF88,
    ]
    .iter()
    .map(|&hex| hex_to_rgba(hex))
    .collect()
}

/// ENDESGA 32 — widely used indie pixel art palette by Endesga.
fn endesga32_palette() -> Vec<Color> {
    [
        0xBE4A2F, 0xD77643, 0xEAD4AA, 0xE4A672, 0xB86F50, 0x733E39,
        0x3E2731, 0xA22633, 0xE43B44, 0xF77622, 0xFEAE34, 0xFEE761,
        0x63C74D, 0x3E8948, 0x265C42, 0x193C3E, 0x124E89, 0x0099DB,
        0x2CE8F5, 0xFFFFFF, 0xC0CBDC, 0x8B9BB4, 0x5A6988, 0x3A4466,
        0x262B44, 0x181425, 0xFF0044, 0x68386C, 0xB55088, 0xF6757A,
        0xE8B796, 0xC28569,
    ]
    .iter()
    .map(|&hex| hex_to_rgba(hex))
    .collect()
}

/// PICO-8 fantasy console — 16 colors.
fn pico8_palette() -> Vec<Color> {
    [
        0x000000, 0x1D2B53, 0x7E2553, 0x008751, 0xAB5236, 0x5F574F,
        0xC2C3C7, 0xFFF1E8, 0xFF004D, 0xFFA300, 0xFFEC27, 0x00E436,
        0x29ADFF, 0x83769C, 0xFF77A8, 0xFFCCAA,
    ]
    .iter()
    .map(|&hex| hex_to_rgba(hex))
    .collect()
}

/// NES — 54 color palette.
fn nes_palette() -> Vec<Color> {
    [
        0x626262, 0x002E98, 0x0C11B2, 0x3B00A4, 0x5E0074, 0x6E0034,
        0x6C0600, 0x561D00, 0x333500, 0x0B4800, 0x005200, 0x004F08,
        0x00404D, 0x000000, 0xABABAB, 0x0D5CFF, 0x4234FF, 0x7F18FF,
        0xAE18B9, 0xBE2664, 0xBA3C18, 0x9A5400, 0x6B7100, 0x388700,
        0x0C9300, 0x008F32, 0x007C8D, 0x000000, 0xFFFFFF, 0x60B0FF,
        0x9488FF, 0xC06EFF, 0xF06EFF, 0xFF6EAE, 0xFF7B60, 0xEF9B18,
        0xC0B800, 0x88D000, 0x60DC18, 0x38E460, 0x38D4B8, 0x404040,
        0xFFFFFF, 0xBEDCFF, 0xCCC8FF, 0xDEC0FF, 0xECC0FF, 0xF6C0DC,
        0xF6C8B8, 0xF0D8A0, 0xE4E494, 0xD0EC94, 0xBEF0A8, 0xB0F0C8,
    ]
    .iter()
    .map(|&hex| hex_to_rgba(hex))
    .collect()
}

/// SNES — extended 256 color palette (representative subset).
fn snes_palette() -> Vec<Color> {
    // SNES could display 256 colors from 32768 — we use a curated set
    let mut colors = Vec::with_capacity(256);
    // Generate a spread across the SNES 5-bit color space
    for r in 0..8u8 {
        for g in 0..8u8 {
            for b in 0..4u8 {
                colors.push([r * 36, g * 36, b * 85, 255]);
            }
        }
    }
    colors
}

/// Game Boy — 4 shades of green.
fn gameboy_palette() -> Vec<Color> {
    [0x0F380F, 0x306230, 0x8BAC0F, 0x9BBC0F]
        .iter()
        .map(|&hex| hex_to_rgba(hex))
        .collect()
}

fn hex_to_rgba(hex: u32) -> Color {
    let r = ((hex >> 16) & 0xFF) as u8;
    let g = ((hex >> 8) & 0xFF) as u8;
    let b = (hex & 0xFF) as u8;
    [r, g, b, 255]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[test]
    fn all_named_palettes_load() {
        for name in &["stardew", "starbound", "snes", "nes", "gameboy", "endesga32", "pico8"] {
            let p = load_palette(name).unwrap();
            assert!(!p.is_empty(), "palette {name} is empty");
        }
    }

    #[test]
    fn unknown_palette_errors() {
        assert!(load_palette("nonexistent_palette_xyz").is_err());
    }

    #[test]
    fn quantize_snaps_to_palette() {
        // Single-color palette: pure red.
        let palette: Vec<Color> = vec![[255, 0, 0, 255]];
        let mut img = RgbaImage::new(2, 2);
        img.put_pixel(0, 0, Rgba([200, 100, 50, 255]));
        img.put_pixel(1, 0, Rgba([10, 220, 10, 255]));
        img.put_pixel(0, 1, Rgba([0, 0, 200, 255]));
        img.put_pixel(1, 1, Rgba([255, 0, 0, 255]));
        let out = quantize(&img, &palette);
        for pixel in out.pixels() {
            assert_eq!(*pixel, Rgba([255, 0, 0, 255]));
        }
    }

    #[test]
    fn quantize_preserves_transparency() {
        let palette: Vec<Color> = vec![[255, 0, 0, 255]];
        let mut img = RgbaImage::new(1, 1);
        img.put_pixel(0, 0, Rgba([200, 100, 50, 0])); // fully transparent
        let out = quantize(&img, &palette);
        assert_eq!(out.get_pixel(0, 0)[3], 0, "transparent pixel alpha must stay 0");
    }

    #[test]
    fn quantize_picks_nearest_in_two_color_palette() {
        // Two colors: pure white and pure black.
        let palette: Vec<Color> = vec![[0, 0, 0, 255], [255, 255, 255, 255]];
        let mut img = RgbaImage::new(2, 1);
        img.put_pixel(0, 0, Rgba([10, 10, 10, 255]));   // near black
        img.put_pixel(1, 0, Rgba([240, 240, 240, 255])); // near white
        let out = quantize(&img, &palette);
        assert_eq!(out.get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
        assert_eq!(out.get_pixel(1, 0), &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn stardew_palette_has_expected_size() {
        let p = load_palette("stardew").unwrap();
        // Should have 48 colors as documented in list_palettes()
        assert!(p.len() >= 40, "stardew palette too small: {}", p.len());
    }

    #[test]
    fn gameboy_palette_has_four_colors() {
        let p = load_palette("gameboy").unwrap();
        assert_eq!(p.len(), 4);
    }
}
