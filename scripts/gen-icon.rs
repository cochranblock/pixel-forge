#!/usr/bin/env -S cargo +nightly -Zscript
//! Generate dark-themed Pixel Forge app icon.
//! Anvil + sparks on near-black background with forge orange glow.
//! Outputs all Android mipmap sizes.

use std::path::Path;

fn main() {
    let sizes = [
        ("mdpi", 48),
        ("hdpi", 72),
        ("xhdpi", 96),
        ("xxhdpi", 144),
        ("xxxhdpi", 192),
    ];

    // 48x48 base icon, scaled up for each density
    for (density, size) in &sizes {
        let img = render_icon(*size);
        let dir = format!("android/app/src/main/res/mipmap-{density}");
        std::fs::create_dir_all(&dir).unwrap();
        img.save(format!("{dir}/ic_launcher.png")).unwrap();
        img.save(format!("{dir}/ic_launcher_round.png")).unwrap();
        println!("wrote {dir}/ic_launcher.png ({size}x{size})");
    }

    // Store icon (512x512)
    let store = render_icon(512);
    store.save("assets/store/play-store-icon.png").unwrap();
    println!("wrote assets/store/play-store-icon.png (512x512)");
}

fn render_icon(size: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(size, size);
    let s = size as f32;

    // Fill background — near black with subtle blue tint
    let bg = [12, 12, 18, 255u8];
    for pixel in img.pixels_mut() {
        *pixel = image::Rgba(bg);
    }

    // Draw radial glow from center-bottom (forge fire)
    for y in 0..size {
        for x in 0..size {
            let cx = x as f32 / s - 0.5;
            let cy = y as f32 / s - 0.7; // glow source below center
            let dist = (cx * cx + cy * cy).sqrt();
            let glow = (1.0 - dist * 2.5).max(0.0).powi(2);

            let px = img.get_pixel(x, y);
            let r = (px[0] as f32 + glow * 180.0).min(255.0) as u8;
            let g = (px[1] as f32 + glow * 50.0).min(255.0) as u8;
            let b = (px[2] as f32 + glow * 10.0).min(255.0) as u8;
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }

    // Anvil body — dark steel with highlight
    let anvil_color = [45, 48, 55, 255u8];
    let anvil_highlight = [70, 75, 85, 255u8];
    let anvil_shadow = [25, 27, 32, 255u8];

    // Anvil proportions (normalized 0-1)
    // Horn (left triangle)
    fill_rect(&mut img, size, 0.15, 0.42, 0.35, 0.12, anvil_color);
    // Face (top flat)
    fill_rect(&mut img, size, 0.25, 0.38, 0.50, 0.06, anvil_highlight);
    // Body
    fill_rect(&mut img, size, 0.30, 0.44, 0.40, 0.16, anvil_color);
    // Waist (narrower)
    fill_rect(&mut img, size, 0.35, 0.54, 0.30, 0.06, anvil_shadow);
    // Base (wider)
    fill_rect(&mut img, size, 0.25, 0.60, 0.50, 0.08, anvil_color);
    // Base bottom
    fill_rect(&mut img, size, 0.22, 0.66, 0.56, 0.04, anvil_shadow);

    // Hammer — angled above anvil
    let hammer_handle = [90, 65, 40, 255u8]; // wood brown
    let hammer_head = [100, 105, 115, 255u8]; // steel
    // Handle (diagonal suggested by offset rect)
    fill_rect(&mut img, size, 0.52, 0.18, 0.04, 0.22, hammer_handle);
    // Head
    fill_rect(&mut img, size, 0.46, 0.14, 0.16, 0.08, hammer_head);

    // Sparks — forge orange dots scattered above anvil
    let spark_color = [255, 102, 0, 255u8]; // ACCENT orange
    let spark_bright = [255, 180, 50, 255u8]; // hot yellow-orange
    let spark_white = [255, 240, 200, 255u8]; // white-hot

    let spark_positions: &[(f32, f32, u8)] = &[
        // (x, y, brightness: 0=orange, 1=yellow, 2=white)
        (0.38, 0.30, 0), (0.42, 0.25, 1), (0.50, 0.22, 2),
        (0.55, 0.28, 0), (0.60, 0.20, 1), (0.35, 0.18, 0),
        (0.48, 0.15, 2), (0.58, 0.32, 0), (0.32, 0.24, 1),
        (0.45, 0.34, 0), (0.62, 0.25, 0), (0.40, 0.12, 1),
        (0.52, 0.10, 0), (0.36, 0.35, 2), (0.56, 0.16, 0),
    ];

    let spark_size = (s * 0.025).max(1.0) as u32;
    for &(sx, sy, brightness) in spark_positions {
        let color = match brightness {
            2 => spark_white,
            1 => spark_bright,
            _ => spark_color,
        };
        let px = (sx * s) as u32;
        let py = (sy * s) as u32;
        for dy in 0..spark_size {
            for dx in 0..spark_size {
                let x = px + dx;
                let y = py + dy;
                if x < size && y < size {
                    img.put_pixel(x, y, image::Rgba(color));
                }
            }
        }
    }

    // "PF" monogram in forge orange — bottom center
    // Using pixel-art style block letters
    let letter_y_start = 0.76;
    let letter_scale = 0.04;
    let orange = spark_color;

    // P: vertical bar + top bump
    draw_block_letter_p(&mut img, size, 0.30, letter_y_start, letter_scale, orange);
    // F: vertical bar + two horizontals
    draw_block_letter_f(&mut img, size, 0.52, letter_y_start, letter_scale, orange);

    // Outer glow ring (subtle)
    for y in 0..size {
        for x in 0..size {
            let cx = x as f32 / s - 0.5;
            let cy = y as f32 / s - 0.5;
            let dist = (cx * cx + cy * cy).sqrt();
            // Subtle orange rim at edges
            if dist > 0.42 && dist < 0.48 {
                let rim = ((0.48 - dist) / 0.06).min(1.0) * 0.3;
                let px = img.get_pixel(x, y);
                let r = (px[0] as f32 + rim * 255.0).min(255.0) as u8;
                let g = (px[1] as f32 + rim * 60.0).min(255.0) as u8;
                img.put_pixel(x, y, image::Rgba([r, g, px[2], 255]));
            }
            // Darken outside circle for round icon
            if dist > 0.48 {
                let fade = ((dist - 0.48) * 10.0).min(1.0);
                let px = img.get_pixel(x, y);
                let r = (px[0] as f32 * (1.0 - fade * 0.5)) as u8;
                let g = (px[1] as f32 * (1.0 - fade * 0.5)) as u8;
                let b = (px[2] as f32 * (1.0 - fade * 0.5)) as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
    }

    img
}

fn fill_rect(img: &mut image::RgbaImage, size: u32, x: f32, y: f32, w: f32, h: f32, color: [u8; 4]) {
    let s = size as f32;
    let x0 = (x * s) as u32;
    let y0 = (y * s) as u32;
    let x1 = ((x + w) * s).min(s) as u32;
    let y1 = ((y + h) * s).min(s) as u32;
    for py in y0..y1 {
        for px in x0..x1 {
            if px < size && py < size {
                img.put_pixel(px, py, image::Rgba(color));
            }
        }
    }
}

// Block letter P (5x7 grid)
fn draw_block_letter_p(img: &mut image::RgbaImage, size: u32, x: f32, y: f32, scale: f32, color: [u8; 4]) {
    let pattern = [
        [1,1,1,1,0],
        [1,0,0,1,1],
        [1,0,0,0,1],
        [1,0,0,1,1],
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
    ];
    draw_pattern(img, size, x, y, scale, &pattern, color);
}

// Block letter F (5x7 grid)
fn draw_block_letter_f(img: &mut image::RgbaImage, size: u32, x: f32, y: f32, scale: f32, color: [u8; 4]) {
    let pattern = [
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
    ];
    draw_pattern(img, size, x, y, scale, &pattern, color);
}

fn draw_pattern(img: &mut image::RgbaImage, size: u32, x: f32, y: f32, scale: f32, pattern: &[[u8; 5]; 7], color: [u8; 4]) {
    let s = size as f32;
    let block = (scale * s).max(1.0) as u32;
    for (row, line) in pattern.iter().enumerate() {
        for (col, &val) in line.iter().enumerate() {
            if val == 1 {
                let bx = (x * s) as u32 + col as u32 * block;
                let by = (y * s) as u32 + row as u32 * block;
                for dy in 0..block {
                    for dx in 0..block {
                        let px = bx + dx;
                        let py = by + dy;
                        if px < size && py < size {
                            img.put_pixel(px, py, image::Rgba(color));
                        }
                    }
                }
            }
        }
    }
}
