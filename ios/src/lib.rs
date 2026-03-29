// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel Forge iOS entry point. Provides C-callable functions for Swift bridge.

/// Bundled Cinder model (TinyUNet, ~4.2 MB).
const CINDER_MODEL: &[u8] = include_bytes!("../../pixel-forge-cinder.safetensors");

/// Entry point called from Swift via @_silgen_name.
/// Extracts bundled model to Documents, launches the egui app.
#[unsafe(no_mangle)]
pub extern "C" fn pixel_forge_main() {
    // Set HOME to iOS Documents directory
    if let Some(home) = std::env::var("HOME").ok() {
        let docs = format!("{}/Documents", home);
        std::fs::create_dir_all(&docs).ok();

        // Extract bundled model
        let dest = format!("{}/pixel-forge-cinder.safetensors", docs);
        let needs_write = match std::fs::metadata(&dest) {
            Ok(meta) => meta.len() != CINDER_MODEL.len() as u64,
            Err(_) => true,
        };
        if needs_write {
            std::fs::write(&dest, CINDER_MODEL).ok();
        }
    }

    // egui app launch happens from Swift side via eframe
    // This function just does the Rust-side initialization
}

/// Returns the library version as a C string.
#[unsafe(no_mangle)]
pub extern "C" fn pixel_forge_version() -> *const std::os::raw::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const _
}
