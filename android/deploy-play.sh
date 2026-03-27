#!/usr/bin/env bash
set -euo pipefail

TRACK="${1:-internal}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Stage 1: Build native library (arm64-v8a) ==="
cargo ndk -t arm64-v8a build --release
if [ $? -ne 0 ]; then
    echo "FATAL: cargo ndk build failed"
    exit 1
fi
echo "Stage 1 complete."

echo ""
echo "=== Stage 2: Build AAB ==="
./gradlew bundleRelease
if [ $? -ne 0 ]; then
    echo "FATAL: gradlew bundleRelease failed"
    exit 1
fi
AAB_PATH="app/build/outputs/bundle/release/app-release.aab"
if [ ! -f "$AAB_PATH" ]; then
    echo "FATAL: AAB not found at $AAB_PATH"
    exit 1
fi
echo "Stage 2 complete. AAB: $AAB_PATH ($(du -h "$AAB_PATH" | cut -f1))"

echo ""
echo "=== Stage 3: Upload to Google Play (track: $TRACK) ==="
if ! command -v fastlane &> /dev/null; then
    echo "FATAL: fastlane not installed. Run: brew install fastlane"
    exit 1
fi
if [ ! -f "play-service-account.json" ]; then
    echo "FATAL: play-service-account.json not found in android/"
    exit 1
fi
fastlane supply \
    --aab "$AAB_PATH" \
    --package_name org.cochranblock.pixelforge \
    --json_key play-service-account.json \
    --track "$TRACK"
if [ $? -ne 0 ]; then
    echo "FATAL: fastlane supply failed"
    exit 1
fi
echo "Stage 3 complete. Uploaded to track: $TRACK"

echo ""
echo "=== Deploy complete ==="
