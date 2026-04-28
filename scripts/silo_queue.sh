#!/usr/bin/env bash
# Drain the Tier-1 D&D silo queue on bt's 5700 XT via Vulkan/any-gpu.
# Runs sequentially, retries once on NaN, logs each model.
set -u

cd "$(dirname "$0")/.."
mkdir -p /tmp/silo_logs

# Tier-1 classes: flagship D&D entities for the dndaimodel.org launch.
# Each ~30-45 min via train-silo --vulkan.
CLASSES=(
  character     # knight, heroes, villager — PCs + named NPCs
  monster       # dragon, demon, slime, bear, wolf — flagship encounters
  creature      # animals, beasts (random encounters)
  weapon_melee  # sword/axe/spear/lightsaber — every fight UI
  weapon_ranged # bow/gun/staff
  armor         # body/helmet/shield/boots/cloak — loot icons
  consumable    # potion/food/gem — loot icons
  terrain       # ground/floor/wall/door — dungeon tiles
  equipment     # key/scroll — quest items
  ui            # UI elements
)

train_one() {
  local class="$1"
  local out="models/${class}.safetensors"
  local log="/tmp/silo_logs/${class}.log"

  if [ -f "$out" ]; then
    echo "[skip] $class already trained at $out"
    return 0
  fi

  echo "[start] $class → $out"
  local t0=$(date +%s)
  ./target/release/pixel-forge train-silo \
    --vulkan \
    --class "$class" \
    --epochs 200 \
    --batch-size 32 \
    --lr 2e-4 \
    --no-ema \
    --output "$out" \
    > "$log" 2>&1

  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    echo "[ok]    $class in ${dt}s → $out"
  else
    echo "[fail]  $class rc=$rc after ${dt}s — retry once"
    ./target/release/pixel-forge train-silo \
      --vulkan --class "$class" \
      --epochs 200 --batch-size 32 --lr 2e-4 --no-ema \
      --output "$out" \
      >> "$log" 2>&1
    if [ -f "$out" ]; then
      echo "[ok]    $class on retry → $out"
    else
      echo "[skip]  $class failed twice — moving on"
    fi
  fi
}

echo "=== silo queue: ${#CLASSES[@]} classes, est. $(( ${#CLASSES[@]} * 35 )) min ==="
for c in "${CLASSES[@]}"; do
  train_one "$c"
done
echo "=== queue drained at $(date) ==="
