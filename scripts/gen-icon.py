#!/usr/bin/env python3
"""Generate dark-themed Pixel Forge app icon.
Uses CC0 anvil sprite from OpenGameArt (Stendhal project).
Blue sparks (#00d9ff) match cochranblock brand.

Anvil sprite: CC0 by the Stendhal team — https://opengameart.org/content/anvil-2
"""

from PIL import Image, ImageDraw
import math, os, random

def render_icon(size, anvil_img):
    img = Image.new('RGBA', (size, size), (12, 12, 18, 255))
    draw = ImageDraw.Draw(img)
    s = float(size)
    px = img.load()

    # Radial glow from center-bottom (blue forge fire)
    for y in range(size):
        for x in range(size):
            cx = x / s - 0.5
            cy = y / s - 0.62
            dist = math.sqrt(cx*cx + cy*cy)
            glow = max(0, 1.0 - dist * 2.8) ** 2
            r, g, b, a = px[x, y]
            r = min(255, int(r + glow * 5))
            g = min(255, int(g + glow * 140))
            b = min(255, int(b + glow * 200))
            px[x, y] = (r, g, b, 255)

    # Blit the CC0 anvil sprite, centered, nearest-neighbor scaled
    anvil_scale = max(1, round(size / anvil_img.width * 0.55))
    scaled_w = anvil_img.width * anvil_scale
    scaled_h = anvil_img.height * anvil_scale
    anvil_scaled = anvil_img.resize((scaled_w, scaled_h), Image.NEAREST)
    ax = (size - scaled_w) // 2
    ay = int(size * 0.28)
    img.paste(anvil_scaled, (ax, ay), anvil_scaled)  # use alpha mask
    px = img.load()  # refresh pixel access after paste

    # Sparks — cochranblock cyber blue
    cyan = (0, 217, 255)
    bright = (100, 230, 255)
    white_hot = (200, 245, 255)

    sparks = [
        (0.38, 0.22, 0), (0.42, 0.18, 1), (0.50, 0.14, 2),
        (0.56, 0.20, 0), (0.61, 0.13, 1), (0.34, 0.11, 0),
        (0.48, 0.09, 2), (0.58, 0.24, 0), (0.30, 0.17, 1),
        (0.44, 0.25, 0), (0.63, 0.18, 0), (0.39, 0.07, 1),
        (0.53, 0.05, 0), (0.36, 0.26, 2), (0.57, 0.10, 0),
        (0.46, 0.04, 1), (0.33, 0.09, 0), (0.60, 0.07, 2),
        (0.40, 0.03, 0), (0.52, 0.02, 1),
    ]

    spark_sz = max(1, int(s * 0.02))
    for sx, sy, brightness in sparks:
        color = [cyan, bright, white_hot][brightness]
        px0 = int(sx * s)
        py0 = int(sy * s)
        for dy in range(spark_sz):
            for dx in range(spark_sz):
                x, y = px0 + dx, py0 + dy
                if 0 <= x < size and 0 <= y < size:
                    px[x, y] = (*color, 255)
        # Glow around bright sparks
        if brightness >= 1:
            glow_r = spark_sz + max(2, int(s * 0.01))
            for dy in range(-glow_r, glow_r + 1):
                for dx in range(-glow_r, glow_r + 1):
                    d = math.sqrt(dx*dx + dy*dy)
                    if spark_sz < d < glow_r + 1:
                        intensity = max(0, 1.0 - d / (glow_r + 1)) * 0.25
                        gx, gy = px0 + dx, py0 + dy
                        if 0 <= gx < size and 0 <= gy < size:
                            r, g, b, a = px[gx, gy]
                            g2 = min(255, int(g + intensity * color[1] * 0.5))
                            b2 = min(255, int(b + intensity * color[2] * 0.5))
                            px[gx, gy] = (r, g2, b2, 255)

    # "PF" monogram — cyber blue pixel art
    def draw_pattern(ox, oy, scale, pattern, color):
        block = max(1, int(scale * s))
        for row, line in enumerate(pattern):
            for col, val in enumerate(line):
                if val:
                    bx = int(ox * s) + col * block
                    by = int(oy * s) + row * block
                    for dy in range(block):
                        for dx in range(block):
                            x, y = bx + dx, by + dy
                            if 0 <= x < size and 0 <= y < size:
                                px[x, y] = (*color, 255)

    P = [
        [1,1,1,1,0],
        [1,0,0,1,1],
        [1,0,0,0,1],
        [1,0,0,1,1],
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
    ]
    F = [
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
    ]

    draw_pattern(0.30, 0.78, 0.032, P, cyan)
    draw_pattern(0.54, 0.78, 0.032, F, cyan)

    # Subtle cyan rim
    for y in range(size):
        for x in range(size):
            cx = x / s - 0.5
            cy = y / s - 0.5
            dist = math.sqrt(cx*cx + cy*cy)
            if 0.43 < dist < 0.49:
                rim = min(1.0, (0.49 - dist) / 0.06) * 0.2
                r, g, b, a = px[x, y]
                g2 = min(255, int(g + rim * 180))
                b2 = min(255, int(b + rim * 255))
                px[x, y] = (r, g2, b2, 255)

    return img

# Load CC0 anvil sprite
anvil_path = '/tmp/anvil-64-black.png'
anvil = Image.open(anvil_path).convert('RGBA')
print(f'loaded anvil sprite: {anvil.width}x{anvil.height}')

# Generate all Android mipmap sizes
densities = [
    ('mdpi', 48),
    ('hdpi', 72),
    ('xhdpi', 96),
    ('xxhdpi', 144),
    ('xxxhdpi', 192),
]

for density, sz in densities:
    icon = render_icon(sz, anvil)
    d = f'android/app/src/main/res/mipmap-{density}'
    os.makedirs(d, exist_ok=True)
    icon.save(f'{d}/ic_launcher.png')
    icon.save(f'{d}/ic_launcher_round.png')
    print(f'wrote {d}/ic_launcher.png ({sz}x{sz})')

# Store icon 512x512
icon = render_icon(512, anvil)
icon.save('assets/store/play-store-icon.png')
print('wrote assets/store/play-store-icon.png (512x512)')
