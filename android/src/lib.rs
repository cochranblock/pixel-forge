// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel Forge Android entry point. egui app via NativeActivity.

use android_activity::AndroidApp;

/// Bundled Cinder model (TinyUNet, 4.2MB).
/// Embedded at compile time so it ships inside the .so — no asset extraction needed.
const CINDER_MODEL: &[u8] = include_bytes!("../../pixel-forge-tiny.safetensors");

#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    log::info!("pixel-forge android starting");

    // Set HOME to Android internal storage so model paths resolve
    let data_dir = app.internal_data_path().expect("no internal data path");
    unsafe { std::env::set_var("HOME", &data_dir) };
    log::info!("HOME={}", data_dir.display());

    // Extract bundled model on first launch
    let model_dest = data_dir.join("pixel-forge-cinder.safetensors");
    if !model_dest.exists() {
        log::info!("extracting bundled Cinder model ({} bytes)...", CINDER_MODEL.len());
        if let Err(e) = std::fs::write(&model_dest, CINDER_MODEL) {
            log::error!("failed to extract model: {e}");
        } else {
            log::info!("model extracted to {}", model_dest.display());
        }
    }

    let kb_app = app.clone();

    let options = eframe::NativeOptions {
        android_app: Some(app),
        ..Default::default()
    };

    eframe::run_native(
        "Pixel Forge",
        options,
        Box::new(move |cc| {
            // Scale for high-DPI mobile screens
            cc.egui_ctx.set_pixels_per_point(2.5);

            let app = pixel_forge::app::PixelForgeApp::new(cc);
            Ok(Box::new(MobileApp {
                inner: app,
                android_app: kb_app,
                keyboard_visible: false,
            }))
        }),
    )
    .expect("eframe failed");
}

/// Wrapper for Android-specific concerns (soft keyboard).
struct MobileApp {
    inner: pixel_forge::app::PixelForgeApp,
    android_app: AndroidApp,
    keyboard_visible: bool,
}

impl eframe::App for MobileApp {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        let wants_kb = ctx.wants_keyboard_input();
        if wants_kb && !self.keyboard_visible {
            self.android_app.show_soft_input(true);
            self.keyboard_visible = true;
        } else if !wants_kb && self.keyboard_visible {
            self.android_app.hide_soft_input(false);
            self.keyboard_visible = false;
        }

        self.inner.update(ctx, frame);
    }
}
