// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel Forge Android entry point. egui app via NativeActivity.

use android_activity::AndroidApp;

#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    log::info!("pixel-forge android starting");

    // Set HOME to Android internal storage so model paths resolve
    if let Some(path) = app.internal_data_path() {
        unsafe { std::env::set_var("HOME", &path) };
        log::info!("HOME={}", path.display());
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
