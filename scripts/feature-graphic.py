#!/usr/bin/env python3
"""Generate Play Store feature graphic (1024x500)."""

import os
import glob
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS = os.path.join(ROOT, "assets", "store", "screenshots")
OUT = os.path.join(ROOT, "assets", "store", "feature-graphic.png")

WIDTH, HEIGHT = 1024, 500
BG = "#0c0c12"
CYAN = "#00d9ff"
WHITE = "#dcdceb"
ORANGE = "#ff7814"
SPRITE_SZ = 32
SCALE = 4
SCALED = SPRITE_SZ * SCALE  # 128


def sprite_colorfulness(sprite):
    """Score a sprite by color variance (more colorful = higher).
    Returns 0 for near-black or mostly transparent sprites."""
    pixels = list(sprite.getdata())
    if not pixels:
        return 0
    unique = set()
    total_brightness = 0
    visible = 0
    for p in pixels:
        r, g, b = p[0], p[1], p[2]
        a = p[3] if len(p) > 3 else 255
        if a < 128:
            continue
        visible += 1
        unique.add((r, g, b))
        total_brightness += r + g + b
    # Skip mostly transparent or very dark sprites
    if visible < len(pixels) * 0.3:
        return 0
    avg_brightness = total_brightness / max(visible, 1)
    if avg_brightness < 60:
        return 0
    return len(unique) * 10 + total_brightness


def extract_sprites(grid_path):
    """Extract 32x32 sprites from an 8-wide grid."""
    img = Image.open(grid_path).convert("RGBA")
    w, h = img.size
    cols = w // SPRITE_SZ
    rows = h // SPRITE_SZ
    sprites = []
    for row in range(rows):
        for col in range(cols):
            x0 = col * SPRITE_SZ
            y0 = row * SPRITE_SZ
            sprite = img.crop((x0, y0, x0 + SPRITE_SZ, y0 + SPRITE_SZ))
            sprites.append(sprite)
    return sprites


def pick_best_per_grid(grids, n):
    """Pick top sprites across all grids, filtering dark/empty ones."""
    all_scored = []
    for g in grids:
        sprites = extract_sprites(g)
        for s in sprites:
            score = sprite_colorfulness(s)
            if score > 0:
                all_scored.append((score, s))

    all_scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in all_scored[:n]]


def load_font(size):
    """Try to load a truetype font, fall back to default."""
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def main():
    # Collect grids
    grids = sorted(glob.glob(os.path.join(SCREENSHOTS, "*_grid.png")))
    total = sum(len(extract_sprites(g)) for g in grids)
    print(f"Found {total} sprites from {len(grids)} grids")

    # Determine sprite count: 8 if they fit, else 6
    sprite_count = 8
    row_width = sprite_count * SCALED + (sprite_count - 1) * 16
    if row_width > WIDTH:
        sprite_count = 6
        row_width = sprite_count * SCALED + (sprite_count - 1) * 16

    best = pick_best_per_grid(grids, sprite_count)
    print(f"Selected {len(best)} sprites, row width = {row_width}px")

    # Create canvas
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_title = load_font(64)
    font_sub = load_font(32)
    font_bottom = load_font(28)

    # Title
    bbox = draw.textbbox((0, 0), "PIXEL FORGE", font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 60), "PIXEL FORGE", fill=CYAN, font=font_title)

    # Subtitle
    bbox = draw.textbbox((0, 0), "AI Sprite Generator", font=font_sub)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 130), "AI Sprite Generator", fill=WHITE, font=font_sub)

    # Sprite row
    x_start = (WIDTH - row_width) // 2
    y_row = 200
    for i, sprite in enumerate(best):
        scaled = sprite.resize((SCALED, SCALED), Image.NEAREST)
        x = x_start + i * (SCALED + 16)
        canvas.paste(scaled, (x, y_row), scaled)

    # Bottom text
    bottom_text = "Free \u00b7 Offline \u00b7 108 Classes"
    bbox = draw.textbbox((0, 0), bottom_text, font=font_bottom)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, 420), bottom_text, fill=ORANGE, font=font_bottom)

    # Save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT, "PNG")
    print(f"Saved: {OUT}")
    print(f"Size: {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
