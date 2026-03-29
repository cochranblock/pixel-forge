#!/usr/bin/env bash
# Build pixel-forge for every supported target.
# Outputs to release/ directory with target-specific filenames.
set -euo pipefail

VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
OUT="release"
mkdir -p "$OUT"

echo "pixel-forge v${VERSION} — multi-architecture build"
echo "================================================"
echo ""

build_local() {
    local target="$1"
    local features="${2:-}"
    local name="pixel-forge-${target}"

    echo "[$target] building..."
    if [ -n "$features" ]; then
        cargo build --release --target "$target" -p pixel-forge $features 2>&1 | tail -1
    else
        cargo build --release --target "$target" -p pixel-forge --no-default-features 2>&1 | tail -1
    fi

    local src="target/${target}/release/pixel-forge"
    if [ -f "$src" ]; then
        cp "$src" "$OUT/$name"
        local size=$(ls -la "$OUT/$name" | awk '{print $5}')
        echo "[$target] done: $name ($((size / 1048576)) MB)"
    else
        echo "[$target] FAILED — no binary produced"
    fi
}

build_remote() {
    local host="$1"
    local target="$2"
    local name="pixel-forge-${target}"

    echo "[$target] building on $host..."
    ssh "$host" "cd ~/pixel-forge && source ~/.cargo/env && cargo build --release -p pixel-forge --no-default-features 2>&1 | tail -1"
    scp "$host":~/pixel-forge/target/release/pixel-forge "$OUT/$name" 2>/dev/null || true

    if [ -f "$OUT/$name" ]; then
        local size=$(ls -la "$OUT/$name" | awk '{print $5}')
        echo "[$target] done: $name ($((size / 1048576)) MB)"
    else
        echo "[$target] FAILED"
    fi
}

build_wasm() {
    echo "[wasm32] building..."
    cargo build --release --target wasm32-unknown-unknown -p pixel-forge --no-default-features 2>&1 | tail -1
    local src="target/wasm32-unknown-unknown/release/pixel_forge.wasm"
    if [ -f "$src" ]; then
        cp "$src" "$OUT/pixel-forge.wasm"
        local size=$(ls -la "$OUT/pixel-forge.wasm" | awk '{print $5}')
        echo "[wasm32] done: pixel-forge.wasm ($((size / 1048576)) MB)"
    else
        echo "[wasm32] FAILED — no .wasm produced"
    fi
}

build_android() {
    echo "[android] building AAB..."
    cd android
    PATH="$HOME/.cargo/bin:$PATH" \
    ANDROID_HOME="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}" \
    KEYSTORE_PASSWORD="${KEYSTORE_PASSWORD:-pixelforge2026}" \
    KEY_PASSWORD="${KEY_PASSWORD:-pixelforge2026}" \
    ./gradlew bundleRelease 2>&1 | tail -1
    cd ..

    local aab="android/app/build/outputs/bundle/release/app-release.aab"
    if [ -f "$aab" ]; then
        cp "$aab" "$OUT/pixel-forge-android.aab"
        local size=$(ls -la "$OUT/pixel-forge-android.aab" | awk '{print $5}')
        echo "[android] done: pixel-forge-android.aab ($((size / 1048576)) MB)"
    else
        echo "[android] FAILED"
    fi
}

# === Local builds (Mac Mini) ===
build_local "aarch64-apple-darwin" "--features metal"  # macOS ARM + Metal
build_local "x86_64-apple-darwin"                       # macOS Intel (CPU)

# === WASM ===
build_wasm

# === Remote builds ===
build_remote "lf" "x86_64-unknown-linux-gnu"  # Linux x86 (RTX 3070 node)

# === Android ===
build_android

echo ""
echo "=== Release artifacts ==="
ls -lah "$OUT"/pixel-forge-* "$OUT"/pixel-forge.wasm 2>/dev/null
echo ""
echo "Upload: gh release upload v${VERSION} $OUT/* --clobber"
