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
    mkdir -p "$DCSS_DIR"
    cd "$DCSS_DIR"
    # Main set
    if [ ! -f "crawl_tiles.zip" ]; then
        curl -L -o crawl_tiles.zip \
            "https://opengameart.org/sites/default/files/crawl_tiles.zip"
    fi
    unzip -qo crawl_tiles.zip -d main/ 2>/dev/null || true
    # Supplemental set
    if [ ! -f "crawl_tiles_supplemental.zip" ]; then
        curl -L -o crawl_tiles_supplemental.zip \
            "https://opengameart.org/sites/default/files/Crawl_tiles_supplemental.zip"
    fi
    unzip -qo crawl_tiles_supplemental.zip -d supplemental/ 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 2: DawnLike v1.81 (CC-BY 4.0, DragonDePlatino + DawnBringer) ---
echo "[2/7] DawnLike v1.81 (CC-BY 4.0, ~5000 sprites, 16x16)"
DAWN_DIR="$RAW_DIR/dawnlike"
if [ ! -f "$DAWN_DIR/.done" ]; then
    mkdir -p "$DAWN_DIR"
    cd "$DAWN_DIR"
    if [ ! -f "DawnLike.zip" ]; then
        curl -L -o DawnLike.zip \
            "https://opengameart.org/sites/default/files/DawnLike_3.zip"
    fi
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
    mkdir -p "$KENNEY_RL_DIR"
    cd "$KENNEY_RL_DIR"
    if [ ! -f "kenney_roguelike-rpg-pack.zip" ]; then
        curl -L -o kenney_roguelike-rpg-pack.zip \
            "https://kenney.nl/media/13762/kenney_roguelike-rpg-pack.zip"
    fi
    unzip -qo kenney_roguelike-rpg-pack.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 4: Kenney Pixel Platformer (CC0) ---
echo "[4/7] Kenney Pixel Platformer (CC0, ~200 sprites)"
KENNEY_PP_DIR="$RAW_DIR/kenney-platformer"
if [ ! -f "$KENNEY_PP_DIR/.done" ]; then
    mkdir -p "$KENNEY_PP_DIR"
    cd "$KENNEY_PP_DIR"
    if [ ! -f "kenney_pixel-platformer.zip" ]; then
        curl -L -o kenney_pixel-platformer.zip \
            "https://kenney.nl/media/13760/kenney_pixel-platformer.zip"
    fi
    unzip -qo kenney_pixel-platformer.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 5: Kenney 1-Bit Pack (CC0) ---
echo "[5/7] Kenney 1-Bit Pack (CC0, ~1000 sprites, 16x16)"
KENNEY_1B_DIR="$RAW_DIR/kenney-1bit"
if [ ! -f "$KENNEY_1B_DIR/.done" ]; then
    mkdir -p "$KENNEY_1B_DIR"
    cd "$KENNEY_1B_DIR"
    if [ ! -f "kenney_1-bit-pack.zip" ]; then
        curl -L -o kenney_1-bit-pack.zip \
            "https://kenney.nl/media/13759/kenney_1-bit-pack.zip"
    fi
    unzip -qo kenney_1-bit-pack.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 6: Hyptosis Tiles (CC-BY 3.0) ---
echo "[6/7] Hyptosis tiles (CC-BY 3.0, ~1000 sprites, 32x32)"
HYPTOSIS_DIR="$RAW_DIR/hyptosis"
if [ ! -f "$HYPTOSIS_DIR/.done" ]; then
    mkdir -p "$HYPTOSIS_DIR"
    cd "$HYPTOSIS_DIR"
    for batch in 1 2 4 5; do
        if [ ! -f "hyptosis_batch${batch}.zip" ]; then
            echo "  batch $batch..."
            curl -L -o "hyptosis_batch${batch}.zip" \
                "https://opengameart.org/sites/default/files/hyptosis_til-art-batch-${batch}.zip" 2>/dev/null || true
        fi
        unzip -qo "hyptosis_batch${batch}.zip" -d "batch${batch}/" 2>/dev/null || true
    done
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

# --- Source 7: David E. Gervais Roguelike Tiles (CC-BY 3.0) ---
echo "[7/7] David E. Gervais tiles (CC-BY 3.0, ~1280 sprites, 32x32)"
GERVAIS_DIR="$RAW_DIR/gervais"
if [ ! -f "$GERVAIS_DIR/.done" ]; then
    mkdir -p "$GERVAIS_DIR"
    cd "$GERVAIS_DIR"
    if [ ! -f "gervais_roguelike.zip" ]; then
        curl -L -o gervais_roguelike.zip \
            "https://opengameart.org/sites/default/files/Gervais_Roguelike_all.zip" 2>/dev/null || true
    fi
    unzip -qo gervais_roguelike.zip -d . 2>/dev/null || true
    touch .done
    echo "  done"
else
    echo "  already downloaded"
fi

echo ""
echo "=== Raw downloads complete ==="
echo "Run 'pixel-forge curate' or scripts/curate.py to split into class directories."
echo "See data/SOURCES.md for full attribution."
