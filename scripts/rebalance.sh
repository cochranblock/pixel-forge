#!/usr/bin/env bash
# Rebalance dataset: cap overrepresented, keep underrepresented as-is.
# Creates data_v3_32/ from data_v2_32/.
# Does NOT generate new data — just subsamples and copies.

set -euo pipefail

SRC="data_v2_32"
DST="data_v3_32"
CAP=2000
SKIP_DIRS="_structure _silhouettes _luminance _colorblocks"

rm -rf "$DST"
mkdir -p "$DST"

total=0
for dir in "$SRC"/*/; do
    name=$(basename "$dir")

    # Skip conditioning dirs
    skip=false
    for s in $SKIP_DIRS; do
        [ "$name" = "$s" ] && skip=true
    done
    $skip && continue

    count=$(find "$dir" -maxdepth 1 -name "*.png" | wc -l | tr -d ' ')
    [ "$count" -eq 0 ] && continue

    mkdir -p "$DST/$name"

    if [ "$count" -le "$CAP" ]; then
        # Under cap — copy all
        cp "$dir"*.png "$DST/$name/"
        echo "$name: $count (all)"
    else
        # Over cap — random subsample
        files=$(find "$dir" -maxdepth 1 -name "*.png" | shuf -n "$CAP")
        echo "$files" | while read f; do
            [ -n "$f" ] && cp "$f" "$DST/$name/"
        done
        echo "$name: $CAP (capped from $count)"
    fi

    actual=$(find "$DST/$name" -maxdepth 1 -name "*.png" | wc -l | tr -d ' ')
    total=$((total + actual))
done

echo ""
echo "total: $total tiles in $DST/"
echo ""

# Write manifest
echo "# data_v3_32 Manifest" > "$DST/MANIFEST.md"
echo "" >> "$DST/MANIFEST.md"
echo "Rebalanced dataset. Cap: $CAP per class." >> "$DST/MANIFEST.md"
echo "" >> "$DST/MANIFEST.md"
echo "| Class | Count |" >> "$DST/MANIFEST.md"
echo "|-------|-------|" >> "$DST/MANIFEST.md"
for dir in "$DST"/*/; do
    name=$(basename "$dir")
    count=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    [ "$count" -gt 0 ] && echo "| $name | $count |" >> "$DST/MANIFEST.md"
done
echo "" >> "$DST/MANIFEST.md"
echo "Total: $total" >> "$DST/MANIFEST.md"
