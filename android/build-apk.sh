#!/usr/bin/env bash
# Unlicense — cochranblock.org
# Contributors: GotEmCoach, KOVA, Claude Opus 4.6

# Build Pixel Forge for Android (aarch64-linux-android, API 34)
# Prereqs:
#   rustup target add aarch64-linux-android
#   cargo install cargo-ndk
#   Android NDK r26+ (brew install --cask android-commandlinetools && sdkmanager "ndk;26.1.10909125")
#   export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/26.1.10909125

set -euo pipefail
cd "$(dirname "$0")"

echo "[android] Building pixel-forge for aarch64-linux-android..."
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release

echo "[android] Shared library built:"
ls -lh app/src/main/jniLibs/arm64-v8a/libpixel_forge_android.so

echo ""
echo "[android] To package APK:"
echo "  cd android && ./gradlew assembleDebug"
echo ""
echo "[android] To install:"
echo "  adb install -r app/build/outputs/apk/debug/app-debug.apk"
echo "  adb shell am start -n org.cochranblock.pixelforge/android.app.NativeActivity"
