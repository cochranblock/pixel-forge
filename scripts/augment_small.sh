#!/bin/bash
# Augment small classes by adding h-flip variants via pixel-forge curate --augment-below.
set -euo pipefail
cd ~/pixel-forge

THRESHOLD=${1:-200}

echo "Augmenting classes with fewer than $THRESHOLD tiles..."
cargo run --release -- curate \
  --raw data/raw \
  --output data_v2_32 \
  --size 32 \
  --augment-below "$THRESHOLD"

# Clear stale dataset cache so next training reprocesses
rm -f data_v2_32/dataset.bin.zst data_v3_32/dataset.bin.zst
echo "Cache cleared. Ready for retraining."
