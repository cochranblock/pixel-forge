#!/bin/bash
cd ~/pixel-forge
source ~/.cargo/env
classes="character weapon potion terrain enemy tree building"
for class in $classes; do
  mkdir -p data_v2_32/_4dir/$class
  count=0
  for f in data_v2_32/$class/*.png; do
    base=$(basename "$f")
    out="data_v2_32/_4dir/$class/$base"
    [ -f "$out" ] && continue
    ./target/release/pixel-forge relight "$f" --output "$out" 2>/dev/null
    count=$((count+1))
  done
  echo "$class: $count relit"
done
echo "DONE"
