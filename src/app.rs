// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel Forge app — the default experience.
//! Launches on startup (no CLI args). Touch-friendly for mobile.
//! Device detection runs automatically. User taps Generate.

use eframe::egui;
use std::sync::{Arc, Mutex};

use crate::cluster::{self, ClusterState};
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
    cluster: Arc<Mutex<Option<ClusterState>>>,
    cluster_probing: Arc<Mutex<bool>>,
    use_cluster: bool,
    selected_class: usize,
    selected_palette: usize,
    prompt_text: String,
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
            cluster: Arc::new(Mutex::new(None)),
            cluster_probing: Arc::new(Mutex::new(false)),
            use_cluster: true,
            selected_class: 0, // character
            selected_palette: 0, // stardew
            prompt_text: String::new(),
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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();

        // Auto-detect local device on startup
        match device_cap::auto_select() {
            Ok(p) => app.profile = Some(p),
            Err(e) => app.profile_error = Some(format!("{e}")),
        }

        // Probe cluster in background — desktop only (SSH probes fail on mobile)
        #[cfg(not(target_os = "android"))]
        {
            let cluster = Arc::clone(&app.cluster);
            let probing = Arc::clone(&app.cluster_probing);
            let ctx = cc.egui_ctx.clone();
            *probing.lock().unwrap() = true;
            std::thread::spawn(move || {
                if let Ok(state) = cluster::probe_cluster() {
                    *cluster.lock().unwrap() = Some(state);
                }
                *probing.lock().unwrap() = false;
                ctx.request_repaint();
            });
        }

        app
    }

    fn start_generation(&mut self, ctx: &egui::Context) {
        let class_name = CLASS_NAMES[self.selected_class].to_string();
        let palette_name = PALETTE_NAMES[self.selected_palette].to_string();
        let count = self.gen_count;
        let steps = self.gen_steps;
        let state = Arc::clone(&self.gen_state);
        let ctx = ctx.clone();

        let cluster_state = if self.use_cluster {
            self.cluster.lock().unwrap().clone()
        } else {
            None
        };
        let local_profile = self.profile.clone();

        {
            let mut s = state.lock().unwrap();
            s.generating = true;
            s.result_pixels = None;
            if cluster_state.is_some() {
                let active = cluster_state.as_ref().unwrap().active_nodes().len();
                s.status = format!("forging {} {}s across {} nodes...", count, class_name, active);
            } else {
                let tier = local_profile.as_ref().map(|p| p.tier.to_string()).unwrap_or("?".into());
                s.status = format!("generating {} {}s with {}...", count, class_name, tier);
            }
        }

        std::thread::spawn(move || {
            let result = if let Some(ref cluster) = cluster_state {
                // Cluster mode: distribute across all nodes
                cluster_generate_for_display(&class_name, count, steps, &palette_name, cluster)
            } else if let Some(ref profile) = local_profile {
                // Local only
                generate_for_display(profile.tier, class_name_to_id(&class_name), count, steps, &palette_name)
            } else {
                Err(anyhow::anyhow!("no device profile"))
            };

            let mut s = state.lock().unwrap();
            s.generating = false;
            match result {
                Ok((pixels, w, h, _)) => {
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

/// Brand colors.
const ACCENT: egui::Color32 = egui::Color32::from_rgb(255, 102, 0); // forge orange
const BG_DARK: egui::Color32 = egui::Color32::from_rgb(18, 18, 24);
const BG_CARD: egui::Color32 = egui::Color32::from_rgb(28, 28, 38);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(140, 140, 160);
const TEXT_BRIGHT: egui::Color32 = egui::Color32::from_rgb(230, 230, 240);
const BTN_BG: egui::Color32 = egui::Color32::from_rgb(40, 40, 55);
const BTN_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 102, 0);

fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG_DARK;
    visuals.window_fill = BG_DARK;
    visuals.override_text_color = Some(TEXT_BRIGHT);
    visuals.widgets.inactive.bg_fill = BTN_BG;
    visuals.widgets.inactive.weak_bg_fill = BTN_BG;
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(55, 55, 75);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_BRIGHT);
    visuals.widgets.active.bg_fill = ACCENT;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    visuals.selection.bg_fill = ACCENT;
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    ctx.set_visuals(visuals);
}

fn styled_button(ui: &mut egui::Ui, label: &str, selected: bool, min_size: egui::Vec2) -> bool {
    let (bg, fg) = if selected {
        (BTN_SELECTED, egui::Color32::WHITE)
    } else {
        (BTN_BG, TEXT_BRIGHT)
    };
    let btn = egui::Button::new(
        egui::RichText::new(label).color(fg).size(14.0)
    )
        .fill(bg)
        .rounding(egui::Rounding::same(6))
        .min_size(min_size);
    ui.add(btn).clicked()
}

fn section_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(TEXT_DIM).size(12.0).strong());
    ui.add_space(4.0);
}

impl eframe::App for PixelForgeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_theme(ctx);

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ctx, |ui| {
            // Scrollable for small screens
            egui::ScrollArea::vertical().show(ui, |ui| {

            // Header
            ui.vertical_centered(|ui| {
                ui.add_space(16.0);
                ui.label(egui::RichText::new("PIXEL FORGE").size(28.0).color(ACCENT).strong());
                ui.label(egui::RichText::new("local AI sprite generator").size(13.0).color(TEXT_DIM));
                ui.add_space(8.0);
            });

            // Device info chip
            if let Some(ref profile) = self.profile {
                ui.vertical_centered(|ui| {
                    let chip = format!("{} | {} MB | {}", profile.backend, profile.ram_mb, profile.tier);
                    let label = egui::Label::new(
                        egui::RichText::new(chip).size(11.0).color(TEXT_DIM)
                    );
                    ui.add(label);
                });
            }

            // Cluster info (desktop only)
            #[cfg(not(target_os = "android"))]
            {
                let probing = *self.cluster_probing.lock().unwrap();
                let cluster = self.cluster.lock().unwrap();
                if probing {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("probing cluster...").size(11.0).color(TEXT_DIM));
                    });
                } else if let Some(ref cs) = *cluster {
                    let active = cs.active_nodes().len();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.use_cluster, "");
                        if self.use_cluster {
                            ui.label(egui::RichText::new(
                                format!("cluster: {} nodes | {:.0} spr/s", active, cs.total_throughput)
                            ).size(11.0).color(ACCENT));
                        } else {
                            ui.label(egui::RichText::new("local only").size(11.0).color(TEXT_DIM));
                        }
                    });
                }
            }

            if let Some(ref err) = self.profile_error {
                ui.colored_label(egui::Color32::from_rgb(255, 80, 80), format!("error: {err}"));
            }

            ui.add_space(16.0);

            // Prompt input
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                section_label(ui, "DESCRIBE YOUR SPRITE");
                let response = ui.add(
                    egui::TextEdit::multiline(&mut self.prompt_text)
                        .hint_text("knight with red cape, idle pose...")
                        .desired_rows(2)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Body)
                        .text_color(TEXT_BRIGHT)
                );
                // Character limit
                if self.prompt_text.len() > 500 {
                    self.prompt_text.truncate(500);
                }
                let char_count = self.prompt_text.len();
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{}/500", char_count)).size(10.0).color(
                        if char_count > 400 { ACCENT } else { egui::Color32::from_rgb(80, 80, 100) }
                    ));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new("optional — class selection used if empty").size(10.0).color(
                            egui::Color32::from_rgb(80, 80, 100)
                        ));
                    });
                });
            });

            ui.add_space(8.0);

            // Class selector — card style
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                section_label(ui, "SPRITE CLASS");
                egui::Grid::new("class_grid")
                    .num_columns(5)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        for (i, name) in CLASS_NAMES.iter().enumerate() {
                            let selected = self.selected_class == i;
                            if styled_button(ui, name, selected, egui::vec2(62.0, 36.0)) {
                                self.selected_class = i;
                            }
                            if (i + 1) % 5 == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });

            ui.add_space(8.0);

            // Palette selector
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                section_label(ui, "PALETTE");
                ui.horizontal_wrapped(|ui| {
                    for (i, name) in PALETTE_NAMES.iter().enumerate() {
                        let selected = self.selected_palette == i;
                        if styled_button(ui, name, selected, egui::vec2(56.0, 32.0)) {
                            self.selected_palette = i;
                        }
                    }
                });
            });

            ui.add_space(8.0);

            // Controls
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                section_label(ui, "SETTINGS");
                let max_count = if self.use_cluster { 64 } else { 16 };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("count").size(12.0).color(TEXT_DIM));
                    ui.add(egui::Slider::new(&mut self.gen_count, 1..=max_count).text(""));
                });
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("steps").size(12.0).color(TEXT_DIM));
                    ui.add(egui::Slider::new(&mut self.gen_steps, 10..=100).text(""));
                });
            });

            ui.add_space(16.0);

            // Generate button — big, prominent
            let generating = {
                let s = self.gen_state.lock().unwrap();
                s.generating
            };

            let can_generate = self.profile.is_some() && !generating;
            ui.vertical_centered(|ui| {
                let label = if generating {
                    "FORGING..."
                } else {
                    "FORGE"
                };

                let btn_color = if can_generate { ACCENT } else { BTN_BG };
                let text_color = if can_generate { egui::Color32::WHITE } else { TEXT_DIM };

                let btn = egui::Button::new(
                    egui::RichText::new(label).size(20.0).color(text_color).strong()
                )
                    .fill(btn_color)
                    .rounding(egui::Rounding::same(12))
                    .min_size(egui::vec2(280.0, 56.0));

                let response = ui.add_enabled(can_generate, btn);
                if response.clicked() {
                    self.start_generation(ctx);
                }
            });

            ui.add_space(8.0);

            // Status
            {
                let s = self.gen_state.lock().unwrap();
                if !s.status.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new(&s.status).size(12.0).color(TEXT_DIM));
                    });
                }
            }

            ui.add_space(12.0);

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
                self.texture_version += 1;
                if self.texture_version != self.last_rendered_version {
                    let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &pixels);
                    let opts = egui::TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        minification: egui::TextureFilter::Nearest,
                        ..Default::default()
                    };
                    self.texture = Some(ctx.load_texture("result", color_image, opts));
                    self.last_rendered_version = self.texture_version;
                }
            }

            if let Some(ref tex) = self.texture {
                ui.vertical_centered(|ui| {
                    let available = ui.available_width().min(360.0);
                    let aspect = tex.size_vec2().y / tex.size_vec2().x;
                    let size = egui::vec2(available, available * aspect);
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(tex.id(), size))
                            .rounding(egui::Rounding::same(8))
                    );
                });
            }

            ui.add_space(24.0);
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new("cochranblock.org").size(10.0).color(
                    egui::Color32::from_rgb(80, 80, 100)
                ));
            });

            }); // ScrollArea
        }); // CentralPanel
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

    let model_path = tier.model_path();
    let (actual_tier, actual_file) = if model_path.exists() {
        (tier, model_path.to_string_lossy().to_string())
    } else {
        let fb = device_cap::best_available();
        (fb, fb.model_path().to_string_lossy().to_string())
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

/// Generate via cluster and return RGBA pixels for display.
fn cluster_generate_for_display(
    class: &str,
    count: u32,
    steps: usize,
    palette_name: &str,
    cluster_state: &ClusterState,
) -> anyhow::Result<(Vec<u8>, u32, u32, u64)> {
    let pngs = cluster::cluster_generate(class, count, steps, palette_name, cluster_state)?;

    if pngs.is_empty() {
        anyhow::bail!("no sprites generated from cluster");
    }

    // Decode PNGs into images
    let images: Vec<image::RgbaImage> = pngs.iter()
        .filter_map(|bytes| {
            image::load_from_memory(bytes).ok().map(|img| img.to_rgba8())
        })
        .collect();

    if images.is_empty() {
        anyhow::bail!("failed to decode cluster output");
    }

    let sheet = crate::sheet::pack_grid(&images, (images.len() as u32).min(8));
    let w = sheet.width();
    let h = sheet.height();
    let pixels = sheet.into_raw();

    Ok((pixels, w, h, 1))
}

fn class_name_to_id(class: &str) -> u32 {
    CLASS_NAMES.iter()
        .position(|&n| n == class.to_lowercase())
        .unwrap_or(14) as u32
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
