#!/bin/bash
# Expand pixel-forge training dataset from CC0/CC-BY sprite sources
set -euo pipefail
cd ~/pixel-forge

DATA_RAW="data/raw_expansion"
mkdir -p "$DATA_RAW"

echo "[1/4] Downloading CC0 sprite sheets..."

# OpenGameArt top pixel art tilesets (CC0/CC-BY)
SOURCES=(
  "https://opengameart.org/sites/default/files/16x16_Dungeon_Tileset_Remix.png"
  "https://opengameart.org/sites/default/files/roguelike_transparent.png"
  "https://opengameart.org/sites/default/files/Dungeon_Crawl_Stone_Soup_Full_0.png"
  "https://opengameart.org/sites/default/files/0x72_16x16RobotTileset.v1.png"
  "https://opengameart.org/sites/default/files/colored_tilemap_packed.png"
  "https://opengameart.org/sites/default/files/monochrome-transparent_packed.png"
  "https://opengameart.org/sites/default/files/1-bit-pack.zip"
  "https://opengameart.org/sites/default/files/top_down_tiles.png"
  "https://opengameart.org/sites/default/files/urizen-onebit.zip"
)

for url in "${SOURCES[@]}"; do
  fname=$(basename "$url")
  if [ ! -f "$DATA_RAW/$fname" ]; then
    echo "  downloading $fname..."
    curl -sL "$url" -o "$DATA_RAW/$fname" 2>/dev/null || echo "  failed: $fname"
  fi
done

echo "[2/4] Extracting zips..."
for z in "$DATA_RAW"/*.zip; do
  [ -f "$z" ] && unzip -qo "$z" -d "$DATA_RAW/" 2>/dev/null || true
done

echo "[3/4] Slicing sprite sheets into tiles..."
# Use the existing curate command for 16x16
./target/release/pixel-forge curate --raw "$DATA_RAW" --output data_expansion_16 --size 16
./target/release/pixel-forge curate --raw "$DATA_RAW" --output data_expansion_32 --size 32

echo "[4/4] Running quality filter..."
# Merge with existing curated data
for cls in character weapon potion terrain enemy tree building animal effect food armor tool vehicle ui misc; do
  mkdir -p data_v2_16/$cls data_v2_32/$cls
  cp data_expansion_16/$cls/*.png data_v2_16/$cls/ 2>/dev/null || true
  cp data_expansion_32/$cls/*.png data_v2_32/$cls/ 2>/dev/null || true
done

echo "Done. New totals:"
for d in data_v2_16/*/; do
  cls=$(basename "$d")
  count=$(ls "$d"*.png 2>/dev/null | wc -l)
  echo "  $cls: $count"
done
