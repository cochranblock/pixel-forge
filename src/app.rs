// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel Forge app — the default experience.
//! Launches on startup (no CLI args). Touch-friendly for mobile.
//! Device detection runs automatically. User taps Generate.

use eframe::egui;
use std::sync::{Arc, Mutex};

use crate::device_cap::{self, DeviceProfile, Tier};
use crate::palette;
use crate::grid;

const CLASS_NAMES: &[&str] = &[
    "character", "weapon", "potion", "terrain", "enemy",
    "tree", "building", "animal", "effect", "food",
    "armor", "tool", "vehicle", "ui", "misc",
];

const PALETTE_NAMES: &[&str] = &[
    "stardew", "starbound", "snes", "nes", "gameboy", "pico8", "endesga",
];

/// App state shared between UI and generation thread.
struct GenerationState {
    status: String,
    result_pixels: Option<Vec<u8>>, // RGBA pixels for display
    result_width: u32,
    result_height: u32,
    generating: bool,
}

pub struct PixelForgeApp {
    profile: Option<DeviceProfile>,
    profile_error: Option<String>,
    selected_class: usize,
    selected_palette: usize,
    gen_count: u32,
    gen_steps: usize,
    gen_state: Arc<Mutex<GenerationState>>,
    texture: Option<egui::TextureHandle>,
    texture_version: u64,
    last_rendered_version: u64,
}

impl Default for PixelForgeApp {
    fn default() -> Self {
        Self {
            profile: None,
            profile_error: None,
            selected_class: 0, // character
            selected_palette: 0, // stardew
            gen_count: 4,
            gen_steps: 40,
            gen_state: Arc::new(Mutex::new(GenerationState {
                status: String::new(),
                result_pixels: None,
                result_width: 0,
                result_height: 0,
                generating: false,
            })),
            texture: None,
            texture_version: 0,
            last_rendered_version: 0,
        }
    }
}

impl PixelForgeApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();

        // Auto-detect device on startup
        match device_cap::auto_select() {
            Ok(p) => app.profile = Some(p),
            Err(e) => app.profile_error = Some(format!("{e}")),
        }

        app
    }

    fn start_generation(&mut self, ctx: &egui::Context) {
        let class_id = self.selected_class as u32;
        let palette_name = PALETTE_NAMES[self.selected_palette].to_string();
        let count = self.gen_count;
        let steps = self.gen_steps;
        let profile = self.profile.clone().unwrap();
        let state = Arc::clone(&self.gen_state);
        let ctx = ctx.clone();

        {
            let mut s = state.lock().unwrap();
            s.generating = true;
            s.status = format!("generating {} {}s with {}...", count, CLASS_NAMES[class_id as usize], profile.tier);
            s.result_pixels = None;
        }

        std::thread::spawn(move || {
            let result = generate_for_display(profile.tier, class_id, count, steps, &palette_name);
            let mut s = state.lock().unwrap();
            s.generating = false;
            match result {
                Ok((pixels, w, h, version_bump)) => {
                    s.status = format!("done — {} sprites", count);
                    s.result_pixels = Some(pixels);
                    s.result_width = w;
                    s.result_height = h;
                }
                Err(e) => {
                    s.status = format!("error: {e}");
                }
            }
            ctx.request_repaint();
        });
    }
}

impl eframe::App for PixelForgeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Dark theme
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(12.0);
                ui.heading("Pixel Forge");
                ui.label("local AI sprite generator");
                ui.add_space(8.0);
            });

            // Device info bar
            if let Some(ref profile) = self.profile {
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "{} | {} MB RAM | {} tier",
                        profile.backend, profile.ram_mb, profile.tier
                    ));
                });
                ui.separator();
            } else if let Some(ref err) = self.profile_error {
                ui.colored_label(egui::Color32::RED, format!("device error: {err}"));
                ui.separator();
            }

            ui.add_space(8.0);

            // Class selector — big buttons for mobile
            ui.label("what to generate:");
            ui.add_space(4.0);
            egui::Grid::new("class_grid")
                .num_columns(5)
                .spacing([6.0, 6.0])
                .show(ui, |ui| {
                    for (i, name) in CLASS_NAMES.iter().enumerate() {
                        let selected = self.selected_class == i;
                        let btn = egui::Button::new(*name)
                            .min_size(egui::vec2(70.0, 36.0))
                            .selected(selected);
                        if ui.add(btn).clicked() {
                            self.selected_class = i;
                        }
                        if (i + 1) % 5 == 0 {
                            ui.end_row();
                        }
                    }
                });

            ui.add_space(12.0);

            // Palette selector
            ui.label("palette:");
            ui.horizontal_wrapped(|ui| {
                for (i, name) in PALETTE_NAMES.iter().enumerate() {
                    let selected = self.selected_palette == i;
                    let btn = egui::Button::new(*name)
                        .min_size(egui::vec2(60.0, 32.0))
                        .selected(selected);
                    if ui.add(btn).clicked() {
                        self.selected_palette = i;
                    }
                }
            });

            ui.add_space(12.0);

            // Count + steps
            ui.horizontal(|ui| {
                ui.label("count:");
                ui.add(egui::Slider::new(&mut self.gen_count, 1..=16));
                ui.add_space(16.0);
                ui.label("steps:");
                ui.add(egui::Slider::new(&mut self.gen_steps, 10..=100));
            });

            ui.add_space(16.0);

            // Generate button
            let generating = {
                let s = self.gen_state.lock().unwrap();
                s.generating
            };

            let can_generate = self.profile.is_some() && !generating;
            ui.vertical_centered(|ui| {
                let btn = egui::Button::new(if generating { "generating..." } else { "Generate" })
                    .min_size(egui::vec2(200.0, 48.0));
                let btn = ui.add_enabled(can_generate, btn);
                if btn.clicked() {
                    self.start_generation(ctx);
                }
            });

            ui.add_space(8.0);

            // Status
            {
                let s = self.gen_state.lock().unwrap();
                if !s.status.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.label(&s.status);
                    });
                }
            }

            ui.add_space(8.0);

            // Result display
            let new_image = {
                let s = self.gen_state.lock().unwrap();
                if s.result_pixels.is_some() {
                    Some((
                        s.result_pixels.clone().unwrap(),
                        s.result_width as usize,
                        s.result_height as usize,
                    ))
                } else {
                    None
                }
            };

            if let Some((pixels, w, h)) = new_image {
                // Update texture if new
                self.texture_version += 1;
                if self.texture_version != self.last_rendered_version {
                    let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &pixels);
                    let opts = egui::TextureOptions {
                        magnification: egui::TextureFilter::Nearest, // pixel art — no smoothing
                        minification: egui::TextureFilter::Nearest,
                        ..Default::default()
                    };
                    self.texture = Some(ctx.load_texture("result", color_image, opts));
                    self.last_rendered_version = self.texture_version;
                }
            }

            if let Some(ref tex) = self.texture {
                ui.vertical_centered(|ui| {
                    // Scale up for visibility — pixel art is tiny
                    let scale = 8.0;
                    let size = tex.size_vec2() * scale;
                    ui.image(egui::load::SizedTexture::new(tex.id(), size));
                });
            }
        });
    }
}

/// Generate sprites and return raw RGBA pixels for display.
fn generate_for_display(
    tier: Tier,
    class_id: u32,
    count: u32,
    steps: usize,
    palette_name: &str,
) -> anyhow::Result<(Vec<u8>, u32, u32, u64)> {
    let pal = palette::load_palette(palette_name)?;
    let img_size = 16u32;

    let model_file = tier.model_file();
    // Fall back if selected model doesn't exist
    let (actual_tier, actual_file) = if std::path::Path::new(model_file).exists() {
        (tier, model_file.to_string())
    } else {
        let fb = device_cap::best_available();
        (fb, fb.model_file().to_string())
    };

    let raw_images = match actual_tier {
        Tier::Cinder => crate::train::sample(&actual_file, class_id, img_size, count, steps)?,
        Tier::Quench | Tier::Anvil => crate::train::sample_medium(&actual_file, class_id, img_size, count, steps)?,
    };

    let processed: Vec<image::RgbaImage> = raw_images
        .into_iter()
        .map(|img| {
            let snapped = grid::snap_to_grid(&img, img_size);
            palette::quantize(&snapped, &pal)
        })
        .collect();

    // Pack into a grid sheet
    let sheet = crate::sheet::pack_grid(&processed, count.min(8));
    let w = sheet.width();
    let h = sheet.height();
    let pixels = sheet.into_raw();

    Ok((pixels, w, h, 1))
}

/// Launch the app. Called when no CLI args are given.
pub fn run() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Pixel Forge")
            .with_inner_size([420.0, 720.0]) // phone-ish aspect ratio
            .with_min_inner_size([320.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Pixel Forge",
        options,
        Box::new(|cc| Ok(Box::new(PixelForgeApp::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
