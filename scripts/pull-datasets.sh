#!/bin/bash
# Unlicense — cochranblock.org
# Contributors: GotEmCoach, KOVA, Claude Opus 4.6
#
# Pull training datasets for Pixel Forge tiny diffusion model.
# All sources are hand-pixeled by known artists. No AI slop. No copyrighted rips.
# See data/SOURCES.md for full attribution.

set -e

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
RAW_DIR="$DATA_DIR/raw"

echo "=== Pixel Forge Dataset Pull ==="
echo "Target: $DATA_DIR"
echo ""

# --- Source 1: Dungeon Crawl Stone Soup Tiles (CC0) ---
echo "[1/7] Dungeon Crawl Stone Soup tiles (CC0, ~6000 sprites, 32x32)"
DCSS_DIR="$RAW_DIR/dungeon-crawl"
if [ ! -f "$DCSS_DIR/.done" ]; then
    rm -rf "$DCSS_DIR"
    mkdir -p "$DCSS_DIR"
    cd "$DCSS_DIR"
    curl -L -o crawl_tiles.zip \
        "https://opengameart.org/sites/default/files/crawl-tiles%20Oct-5-2010.zip"
    curl -L -o crawl_full.zip \
        "https://opengameart.org/sites/default/files/Dungeon%20Crawl%20Stone%20Soup%20Full_0.zip"
    unzip -qo crawl_tiles.zip -d tiles/ 2>/dev/null || true
    unzip -qo crawl_full.zip -d full/ 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 2: DawnLike v1.81 (CC-BY 4.0, DragonDePlatino + DawnBringer) ---
echo "[2/7] DawnLike v1.81 (CC-BY 4.0, ~5000 sprites, 16x16)"
DAWN_DIR="$RAW_DIR/dawnlike"
if [ ! -f "$DAWN_DIR/.done" ]; then
    rm -rf "$DAWN_DIR"
    mkdir -p "$DAWN_DIR"
    cd "$DAWN_DIR"
    curl -L -o DawnLike.zip \
        "https://opengameart.org/sites/default/files/DawnLike_3.zip"
    unzip -qo DawnLike.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 3: Kenney Roguelike/RPG Pack (CC0) ---
echo "[3/7] Kenney Roguelike/RPG Pack (CC0, ~1700 sprites, 16x16)"
KENNEY_RL_DIR="$RAW_DIR/kenney-roguelike"
if [ ! -f "$KENNEY_RL_DIR/.done" ]; then
    rm -rf "$KENNEY_RL_DIR"
    mkdir -p "$KENNEY_RL_DIR"
    cd "$KENNEY_RL_DIR"
    curl -L -o pack.zip \
        "https://kenney.nl/media/pages/assets/roguelike-rpg-pack/1cb71b28fb-1677697420/kenney_roguelike-rpg-pack.zip"
    unzip -qo pack.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 4: Kenney Pixel Platformer (CC0) ---
echo "[4/7] Kenney Pixel Platformer (CC0, ~200 sprites)"
KENNEY_PP_DIR="$RAW_DIR/kenney-platformer"
if [ ! -f "$KENNEY_PP_DIR/.done" ]; then
    rm -rf "$KENNEY_PP_DIR"
    mkdir -p "$KENNEY_PP_DIR"
    cd "$KENNEY_PP_DIR"
    curl -L -o pack.zip \
        "https://kenney.nl/media/pages/assets/pixel-platformer/bef991136c-1696667883/kenney_pixel-platformer.zip"
    unzip -qo pack.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 5: Kenney 1-Bit Pack (CC0) ---
echo "[5/7] Kenney 1-Bit Pack (CC0, ~1078 sprites, 16x16)"
KENNEY_1B_DIR="$RAW_DIR/kenney-1bit"
if [ ! -f "$KENNEY_1B_DIR/.done" ]; then
    rm -rf "$KENNEY_1B_DIR"
    mkdir -p "$KENNEY_1B_DIR"
    cd "$KENNEY_1B_DIR"
    curl -L -o pack.zip \
        "https://kenney.nl/media/pages/assets/1-bit-pack/f41b6925f0-1677578516/kenney_1-bit-pack.zip"
    unzip -qo pack.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 6: Hyptosis Tiles (CC-BY 3.0) — these are single PNGs, not zips ---
echo "[6/7] Hyptosis tiles (CC-BY 3.0, ~1000 sprites, 32x32)"
HYPTOSIS_DIR="$RAW_DIR/hyptosis"
if [ ! -f "$HYPTOSIS_DIR/.done" ]; then
    rm -rf "$HYPTOSIS_DIR"
    mkdir -p "$HYPTOSIS_DIR"
    cd "$HYPTOSIS_DIR"
    curl -L -o batch1.png "https://opengameart.org/sites/default/files/hyptosis_tile-art-batch-1.png"
    curl -L -o batch2.png "https://opengameart.org/sites/default/files/hyptosis_til-art-batch-2.png"
    curl -L -o batch3.png "https://opengameart.org/sites/default/files/hyptosis_tile-art-batch-3_0.png"
    curl -L -o batch4.png "https://opengameart.org/sites/default/files/hyptosis_tile-art-batch-4.png"
    curl -L -o batch5_16x16.png "https://opengameart.org/sites/default/files/hyptosis_tile-art-batch-5.png"
    curl -L -o sprites_and_tiles.png "https://opengameart.org/sites/default/files/hyptosis_sprites-and-tiles-for-you.png"
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 7: David E. Gervais Roguelike Tiles (CC-BY 3.0) ---
echo "[7/7] David E. Gervais tiles (CC-BY 3.0, ~1280 sprites, 32x32)"
GERVAIS_DIR="$RAW_DIR/gervais"
if [ ! -f "$GERVAIS_DIR/.done" ]; then
    rm -rf "$GERVAIS_DIR"
    mkdir -p "$GERVAIS_DIR"
    cd "$GERVAIS_DIR"
    curl -L -o AngbandTk.zip "https://opengameart.org/sites/default/files/AngbandTk.zip"
    curl -L -o DO_Items.zip "https://opengameart.org/sites/default/files/DO%20Items.zip"
    curl -L -o DO_SysPics.zip "https://opengameart.org/sites/default/files/DO%20SysPics.zip"
    curl -L -o DO_Terrain.zip "https://opengameart.org/sites/default/files/DO%20Terrain.zip"
    curl -L -o DO_Monsters.zip "https://opengameart.org/sites/default/files/DO%20Monsters.zip"
    curl -L -o DO_More.zip "https://opengameart.org/sites/default/files/DO%20More.zip"
    curl -L -o Silmar.zip "https://opengameart.org/sites/default/files/Silmar.zip"
    curl -L -o Work.zip "https://opengameart.org/sites/default/files/Work.zip"
    for z in *.zip; do
        unzip -qo "$z" -d "$(basename "$z" .zip)/" 2>/dev/null || true
    done
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

echo ""
echo "=== Raw downloads complete ==="
echo ""

# Count what we got
total=0
for d in "$RAW_DIR"/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name "*.png" -o -name "*.PNG" -o -name "*.bmp" -o -name "*.BMP" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $name: $count files"
    total=$((total + count))
done
echo "  TOTAL: $total files"
echo ""
echo "Next: pixel-forge curate --raw data/raw --output data --size 16"
