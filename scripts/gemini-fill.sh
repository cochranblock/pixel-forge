#!/usr/bin/env bash
# Fill weak classes with Gemini-generated 32x32 pixel art sprites.
# Reads GEMINI_API_KEY from env or ~/.env.local. Never hardcodes keys.
set -euo pipefail

# Load API key
if [ -z "${GEMINI_API_KEY:-}" ]; then
    if [ -f ~/.env.local ]; then
        GEMINI_API_KEY=$(grep GEMINI_API_KEY ~/.env.local | cut -d= -f2)
    fi
fi
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "error: GEMINI_API_KEY not set. Export it or add to ~/.env.local"
    exit 1
fi

DST="${1:-data_v3_32}"
TARGET=200
DELAY=2  # seconds between requests (free tier safe)
MODEL="${GEMINI_MODEL:-gemini-2.5-flash-image}"
API="https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${GEMINI_API_KEY}"
echo "model: $MODEL"
echo "NOTE: Image generation requires a paid Gemini API plan."
echo "      Free tier has 0 quota for image models."
echo "      Alternative: paste prompts from scripts/gemini-fill-prompts.md into Gemini web UI."
echo ""

# Map class names to descriptive prompts
describe() {
    case "$1" in
        bush_flower) echo "bush with flowers, green leaves, colorful petals" ;;
        cat_space) echo "space cat in a helmet, sci-fi, floating" ;;
        cat_warrior) echo "cat warrior with sword and armor, medieval" ;;
        cat_fat) echo "fat round chubby cat, sitting, cute" ;;
        cat_wizard) echo "cat wearing wizard hat, holding magic staff" ;;
        crop) echo "farm crop plant, wheat corn tomato carrot" ;;
        cyborg) echo "cyborg, half human half machine, glowing eye" ;;
        door_stair) echo "dungeon door or stone staircase, RPG tileset" ;;
        dwarf) echo "dwarf warrior, beard, axe, mining helmet" ;;
        elf) echo "elf archer, pointed ears, bow, forest theme" ;;
        farm_animal) echo "farm animal, cow pig chicken sheep" ;;
        fish) echo "tropical fish, colorful, aquatic" ;;
        fx_ambient) echo "ambient particle effect, sparkle, dust, firefly" ;;
        fx_combat) echo "combat slash effect, impact, explosion" ;;
        goblin) echo "goblin, green skin, dagger, sneaky, small" ;;
        ground_natural) echo "natural ground tile, grass dirt sand gravel" ;;
        hero_cleric) echo "cleric healer, robes, holy staff, RPG character" ;;
        horse) echo "horse, war horse or farm horse, side view" ;;
        mech) echo "bipedal mech robot, sci-fi walker, armed" ;;
        mount) echo "rideable mount, fantasy creature, saddle" ;;
        mushroom) echo "mushroom, red cap, glowing, fantasy style" ;;
        reptile) echo "reptile, lizard snake turtle gecko" ;;
        slime) echo "slime blob, bouncy, RPG enemy, translucent" ;;
        spear_polearm) echo "spear or polearm weapon, long shaft, blade tip" ;;
        terrain_scifi) echo "sci-fi floor tile, metal panel, neon grid" ;;
        tree_conifer) echo "conifer pine tree, evergreen, pointed" ;;
        tree_exotic) echo "exotic tropical tree, palm, baobab" ;;
        wall) echo "stone brick wall segment, dungeon, castle" ;;
        wild_animal) echo "wild forest animal, deer fox rabbit boar" ;;
        wolf) echo "wolf, gray dire ice wolf, howling or walking" ;;
        zombie) echo "zombie, undead, rotting, green skin, RPG enemy" ;;
        dog) echo "dog, varied breed, sitting or standing, pet" ;;
        *) echo "$1, fantasy RPG game asset" ;;
    esac
}

total_generated=0

for dir in "$DST"/*/; do
    class=$(basename "$dir")
    current=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

    [ "$current" -ge "$TARGET" ] && continue

    needed=$((TARGET - current))
    desc=$(describe "$class")

    echo "=== $class: $current → $TARGET (need $needed) ==="

    idx=$current
    generated=0

    while [ "$generated" -lt "$needed" ]; do
        idx=$((idx + 1))
        outfile="$dir/gemini_fill_$(printf '%04d' $idx).png"

        prompt="Generate a single 32x32 pixel art sprite of a ${desc}. 8-bit retro game style, clean single-pixel edges, transparent background, centered in frame, no text, no border. Output only the image."

        # Build JSON payload requesting image generation
        payload=$(cat <<ENDJSON
{
  "contents": [{"parts": [{"text": "$prompt"}]}],
  "generationConfig": {
    "responseModalities": ["TEXT", "IMAGE"]
  }
}
ENDJSON
)

        response=$(curl -s -X POST "$API" \
            -H "Content-Type: application/json" \
            -d "$payload" 2>&1)

        # Extract base64 image from response
        img_data=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    for part in r.get('candidates', [{}])[0].get('content', {}).get('parts', []):
        if 'inlineData' in part:
            print(part['inlineData']['data'])
            sys.exit(0)
    sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null) || true

        if [ -n "$img_data" ]; then
            echo "$img_data" | base64 -d > "$outfile"
            # Resize to 32x32 if needed (Gemini may output larger)
            python3 -c "
from PIL import Image
import sys
try:
    img = Image.open('$outfile')
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.NEAREST)
        img.save('$outfile')
except:
    pass
"
            generated=$((generated + 1))
            total_generated=$((total_generated + 1))
            echo "  [$generated/$needed] $outfile"
        else
            # Check for error
            err=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    if 'error' in r:
        print(r['error'].get('message', 'unknown error'))
    else:
        print('no image in response')
except:
    print('parse error')
" 2>/dev/null)
            echo "  SKIP: $err"
            # If rate limited, wait longer
            if echo "$err" | grep -qi "rate\|quota\|429"; then
                echo "  rate limited — waiting 30s"
                sleep 30
                continue
            fi
        fi

        sleep "$DELAY"
    done

    new_count=$(find "$dir" -maxdepth 1 -name "*.png" | wc -l | tr -d ' ')
    echo "  done: $class now has $new_count samples"
done

echo ""
echo "total generated: $total_generated sprites"
echo "updating manifest..."

# Update manifest
echo "# data_v3_32 Manifest (post-fill)" > "$DST/MANIFEST.md"
echo "" >> "$DST/MANIFEST.md"
echo "Rebalanced + Gemini-filled dataset." >> "$DST/MANIFEST.md"
echo "" >> "$DST/MANIFEST.md"
echo "| Class | Count |" >> "$DST/MANIFEST.md"
echo "|-------|-------|" >> "$DST/MANIFEST.md"
grand=0
for dir in "$DST"/*/; do
    name=$(basename "$dir")
    count=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    [ "$count" -gt 0 ] && echo "| $name | $count |" >> "$DST/MANIFEST.md"
    grand=$((grand + count))
done
echo "" >> "$DST/MANIFEST.md"
echo "Total: $grand" >> "$DST/MANIFEST.md"
echo "done. manifest updated."
