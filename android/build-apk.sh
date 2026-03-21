#!/usr/bin/env bash
# Unlicense — cochranblock.org
# Contributors: GotEmCoach, KOVA, Claude Opus 4.6

# Build Pixel Forge APK end-to-end.
# Prereqs:
#   rustup target add aarch64-linux-android
#   cargo install cargo-ndk
#   Android NDK + SDK (brew install --cask android-commandlinetools)

set -euo pipefail
cd "$(dirname "$0")"

# Auto-detect SDK/NDK paths
export ANDROID_HOME="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
NDK_DIR=$(ls -d "$ANDROID_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$NDK_DIR}"

echo "[1/3] cargo ndk — building .so for arm64-v8a..."
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release 2>&1

SO="app/src/main/jniLibs/arm64-v8a/libpixel_forge_android.so"
if [ ! -f "$SO" ]; then
  echo "FAIL: .so not found at $SO"
  exit 1
fi
echo "  .so: $(du -h "$SO" | cut -f1)"

echo "[2/3] gradle — assembling APK..."
chmod +x gradlew
./gradlew assembleRelease 2>&1

APK="app/build/outputs/apk/release/app-release-unsigned.apk"
if [ ! -f "$APK" ]; then
  # Try debug if release signing isn't set up
  ./gradlew assembleDebug 2>&1
  APK="app/build/outputs/apk/debug/app-debug.apk"
fi

if [ ! -f "$APK" ]; then
  echo "FAIL: APK not found"
  exit 1
fi

echo "[3/3] done"
echo "  APK: $APK ($(du -h "$APK" | cut -f1))"
echo ""
echo "install: adb install -r $APK"
echo "launch:  adb shell am start -n org.cochranblock.pixelforge/android.app.NativeActivity"
