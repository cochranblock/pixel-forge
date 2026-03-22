#!/usr/bin/env python3
"""Generate dark-themed Pixel Forge app icon.
Anvil + sparks on near-black background with forge orange glow.
Outputs all Android mipmap sizes + 512x512 store icon.
"""

import struct, zlib, os, math

def write_png(path, pixels, width, height):
    """Write RGBA pixels as PNG. No PIL needed."""
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter: none
        for x in range(width):
            i = (y * width + x) * 4
            raw += bytes(pixels[i:i+4])

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    idat = zlib.compress(raw, 9)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(sig)
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', idat))
        f.write(chunk(b'IEND', b''))

def render_icon(size):
    """Render the icon at given size. Returns RGBA pixel list."""
    pixels = [0] * (size * size * 4)
    s = float(size)

    def put(x, y, r, g, b, a=255):
        if 0 <= x < size and 0 <= y < size:
            i = (y * size + x) * 4
            pixels[i] = r
            pixels[i+1] = g
            pixels[i+2] = b
            pixels[i+3] = a

    def get(x, y):
        if 0 <= x < size and 0 <= y < size:
            i = (y * size + x) * 4
            return pixels[i], pixels[i+1], pixels[i+2], pixels[i+3]
        return 0, 0, 0, 255

    # Background — near black with subtle blue
    for y in range(size):
        for x in range(size):
            put(x, y, 12, 12, 18)

    # Radial glow from center-bottom (forge fire)
    for y in range(size):
        for x in range(size):
            cx = x / s - 0.5
            cy = y / s - 0.68
            dist = math.sqrt(cx*cx + cy*cy)
            glow = max(0, 1.0 - dist * 2.8) ** 2
            r, g, b, a = get(x, y)
            r = min(255, int(r + glow * 200))
            g = min(255, int(g + glow * 55))
            b = min(255, int(b + glow * 8))
            put(x, y, r, g, b)

    # Helper: filled rect
    def fill_rect(rx, ry, rw, rh, color):
        x0 = int(rx * s)
        y0 = int(ry * s)
        x1 = min(size, int((rx + rw) * s))
        y1 = min(size, int((ry + rh) * s))
        for py in range(y0, y1):
            for px in range(x0, x1):
                put(px, py, *color)

    # Anvil colors
    steel = (45, 48, 55)
    steel_hi = (75, 80, 92)
    steel_sh = (25, 27, 32)

    # Anvil — horn (left taper)
    fill_rect(0.15, 0.43, 0.22, 0.04, steel_hi)
    fill_rect(0.18, 0.47, 0.19, 0.04, steel)
    # Face (top flat surface)
    fill_rect(0.25, 0.38, 0.50, 0.03, steel_hi)
    fill_rect(0.25, 0.41, 0.50, 0.05, steel)
    # Body
    fill_rect(0.30, 0.46, 0.40, 0.12, steel)
    # Waist
    fill_rect(0.35, 0.55, 0.30, 0.05, steel_sh)
    # Base
    fill_rect(0.25, 0.60, 0.50, 0.06, steel)
    fill_rect(0.22, 0.66, 0.56, 0.04, steel_sh)
    # Heel (right side bump)
    fill_rect(0.65, 0.43, 0.10, 0.06, steel_hi)

    # Hammer
    wood = (100, 72, 45)
    hammer_steel = (110, 115, 125)
    # Handle
    bw = max(2, int(0.03 * s))
    for i in range(int(0.22 * s)):
        hx = int(0.54 * s) - int(i * 0.3)
        hy = int(0.18 * s) + i
        for dx in range(bw):
            put(hx + dx, hy, *wood)
    # Head
    fill_rect(0.42, 0.13, 0.18, 0.07, hammer_steel)
    fill_rect(0.43, 0.12, 0.16, 0.02, (130, 135, 145))  # highlight top

    # Sparks
    orange = (255, 102, 0)
    yellow = (255, 180, 50)
    white_hot = (255, 240, 200)

    sparks = [
        (0.38, 0.32, 0), (0.42, 0.26, 1), (0.50, 0.24, 2),
        (0.56, 0.30, 0), (0.61, 0.22, 1), (0.34, 0.20, 0),
        (0.48, 0.16, 2), (0.58, 0.34, 0), (0.30, 0.26, 1),
        (0.44, 0.35, 0), (0.63, 0.27, 0), (0.39, 0.13, 1),
        (0.53, 0.11, 0), (0.36, 0.36, 2), (0.57, 0.18, 0),
        (0.46, 0.08, 1), (0.33, 0.15, 0), (0.60, 0.14, 2),
        (0.40, 0.08, 0), (0.52, 0.06, 1),
    ]

    spark_sz = max(1, int(s * 0.022))
    for sx, sy, brightness in sparks:
        color = [orange, yellow, white_hot][brightness]
        px0 = int(sx * s)
        py0 = int(sy * s)
        for dy in range(spark_sz):
            for dx in range(spark_sz):
                put(px0 + dx, py0 + dy, *color)
        # Glow around bright sparks
        if brightness >= 1:
            glow_r = spark_sz + 1
            for dy in range(-glow_r, glow_r + 1):
                for dx in range(-glow_r, glow_r + 1):
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > spark_sz and d < glow_r + 1:
                        intensity = max(0, 1.0 - d / (glow_r + 1)) * 0.3
                        gx, gy = px0 + dx, py0 + dy
                        r, g, b, a = get(gx, gy)
                        r = min(255, int(r + intensity * color[0]))
                        g = min(255, int(g + intensity * color[1] * 0.3))
                        put(gx, gy, r, g, b)

    # "PF" monogram — pixel art block letters
    def draw_pattern(ox, oy, scale, pattern, color):
        block = max(1, int(scale * s))
        for row, line in enumerate(pattern):
            for col, val in enumerate(line):
                if val:
                    bx = int(ox * s) + col * block
                    by = int(oy * s) + row * block
                    for dy in range(block):
                        for dx in range(block):
                            put(bx + dx, by + dy, *color)

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

    letter_scale = 0.035
    draw_pattern(0.30, 0.76, letter_scale, P, orange)
    draw_pattern(0.54, 0.76, letter_scale, F, orange)

    # Subtle orange rim
    for y in range(size):
        for x in range(size):
            cx = x / s - 0.5
            cy = y / s - 0.5
            dist = math.sqrt(cx*cx + cy*cy)
            if 0.43 < dist < 0.49:
                rim = min(1.0, (0.49 - dist) / 0.06) * 0.25
                r, g, b, a = get(x, y)
                r = min(255, int(r + rim * 255))
                g = min(255, int(g + rim * 50))
                put(x, y, r, g, b)

    return pixels

# Generate all sizes
densities = [
    ('mdpi', 48),
    ('hdpi', 72),
    ('xhdpi', 96),
    ('xxhdpi', 144),
    ('xxxhdpi', 192),
]

for density, sz in densities:
    px = render_icon(sz)
    d = f'android/app/src/main/res/mipmap-{density}'
    write_png(f'{d}/ic_launcher.png', px, sz, sz)
    write_png(f'{d}/ic_launcher_round.png', px, sz, sz)
    print(f'wrote {d}/ic_launcher.png ({sz}x{sz})')

# Store icon
px = render_icon(512)
write_png('assets/store/play-store-icon.png', px, 512, 512)
print('wrote assets/store/play-store-icon.png (512x512)')
