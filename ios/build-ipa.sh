#!/usr/bin/env bash
# Build Pixel Forge iOS — Rust staticlib + Xcode project.
# Prerequisites: Xcode, rustup target add aarch64-apple-ios
set -euo pipefail

echo "[1/3] cargo build — aarch64-apple-ios"
cargo build --release --target aarch64-apple-ios

echo "[2/3] xcodebuild — archive"
xcodebuild archive \
    -project PixelForge.xcodeproj \
    -scheme PixelForge \
    -archivePath build/PixelForge.xcarchive \
    -sdk iphoneos \
    LIBRARY_SEARCH_PATHS="../target/aarch64-apple-ios/release"

echo "[3/3] xcodebuild — export IPA"
xcodebuild -exportArchive \
    -archivePath build/PixelForge.xcarchive \
    -exportOptionsPlist ExportOptions.plist \
    -exportPath build/

echo "done: build/PixelForge.ipa"
ls -la build/*.ipa
