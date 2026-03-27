// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Pixel Forge app — the default experience.
//! Launches on startup (no CLI args). Touch-friendly for mobile.
//! Device detection runs automatically. User taps Generate.

use eframe::egui;
use std::sync::{Arc, Mutex};

use crate::class_cond;
use crate::cluster::{self, ClusterState};
use crate::device_cap::{self, DeviceProfile, Tier};
use crate::palette;
use crate::grid;

const PALETTE_NAMES: &[&str] = &[
    "stardew", "starbound", "snes", "nes", "gameboy", "pico8", "endesga",
];

/// App state shared between UI and generation thread.
struct GenerationState {
    status: String,
    result_pixels: Option<Vec<u8>>, // RGBA pixels for display (sheet)
    result_width: u32,
    result_height: u32,
    generating: bool,
    /// Individual sprites for swipe review (raw f32 pixels, 32x32x3).
    individual_sprites: Vec<Vec<f32>>,
    /// Class IDs for each individual sprite.
    sprite_class_ids: Vec<u32>,
}

/// Generation mode — which model pipeline to use.
#[derive(Clone, Copy, PartialEq)]
pub enum GenMode {
    Cinder,   // m0 — fast, detail
    Quench,   // m1 — foundation
    Cascade,  // m1→m0 MoE — Quench foundation, Cinder detail
    Anvil,    // m2 — desktop, single-stage
}

impl GenMode {
    fn label(&self) -> &'static str {
        match self {
            GenMode::Cinder => "Cinder",
            GenMode::Quench => "Quench",
            GenMode::Cascade => "Cascade",
            GenMode::Anvil => "Anvil",
        }
    }
    fn desc(&self) -> &'static str {
        match self {
            GenMode::Cinder => "fast, 1M params",
            GenMode::Quench => "foundation, 6M params",
            GenMode::Cascade => "MoE: Quench foundation, Cinder detail",
            GenMode::Anvil => "desktop, 16M params",
        }
    }
}

pub struct PixelForgeApp {
    profile: Option<DeviceProfile>,
    profile_error: Option<String>,
    cluster: Arc<Mutex<Option<ClusterState>>>,
    cluster_probing: Arc<Mutex<bool>>,
    use_cluster: bool,
    selected_super: u32,
    selected_class: String,
    selected_palette: usize,
    prompt_text: String,
    gen_mode: GenMode,
    gen_count: u32,
    gen_steps: usize,
    gen_state: Arc<Mutex<GenerationState>>,
    texture: Option<egui::TextureHandle>,
    texture_version: u64,
    last_rendered_version: u64,
    // Keyboard toggle for Android
    pub keyboard_requested: bool,
    // Advanced settings toggle
    show_advanced: bool,
    // Swipe review state
    swipe_store: crate::swipe_store::SwipeStore,
    swipe_index: usize,         // which sprite we're reviewing
    swipe_texture: Option<egui::TextureHandle>,
    reviewing: bool,            // true = in swipe mode
    lora_training: Arc<Mutex<bool>>,
}

impl Default for PixelForgeApp {
    fn default() -> Self {
        Self {
            profile: None,
            profile_error: None,
            cluster: Arc::new(Mutex::new(None)),
            cluster_probing: Arc::new(Mutex::new(false)),
            use_cluster: true,
            selected_super: 0, // humanoid
            selected_class: "character".to_string(),
            selected_palette: 0, // stardew
            prompt_text: String::new(),
            keyboard_requested: false,
            show_advanced: false,
            gen_mode: GenMode::Cascade,
            gen_count: 4,
            gen_steps: 10,
            gen_state: Arc::new(Mutex::new(GenerationState {
                status: String::new(),
                result_pixels: None,
                result_width: 0,
                result_height: 0,
                generating: false,
                individual_sprites: Vec::new(),
                sprite_class_ids: Vec::new(),
            })),
            texture: None,
            texture_version: 0,
            last_rendered_version: 0,
            swipe_store: crate::swipe_store::SwipeStore::load_or_default("swipes.bin", 200),
            swipe_index: 0,
            swipe_texture: None,
            reviewing: false,
            lora_training: Arc::new(Mutex::new(false)),
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
        let class_name = self.selected_class.clone();
        let palette_name = PALETTE_NAMES[self.selected_palette].to_string();
        let count = self.gen_count;
        let steps = self.gen_steps;
        let mode = self.gen_mode;
        let state = Arc::clone(&self.gen_state);
        let ctx = ctx.clone();

        let cluster_state = if self.use_cluster {
            self.cluster.lock().unwrap().clone()
        } else {
            None
        };
        let _local_profile = self.profile.clone();

        {
            let mut s = state.lock().unwrap();
            s.generating = true;
            s.result_pixels = None;
            s.status = format!("generating {} {}s with {}...", count, class_name, mode.label());
        }

        std::thread::spawn(move || {
            let cond = crate::class_cond::lookup(&class_name);
            let result: anyhow::Result<GenResult> = if let Some(ref cluster) = cluster_state {
                cluster_generate_for_display(&class_name, count, steps, &palette_name, cluster)
                    .map(|(pixels, w, h, _)| GenResult {
                        sheet_pixels: pixels, width: w, height: h, sprites_f32: Vec::new()
                    })
            } else {
                // Auto-detect best pipeline, or use manual mode if advanced
                match mode {
                    GenMode::Cascade | GenMode::Quench => {
                        let quench_path = device_cap::Tier::Quench.model_path();
                        let cinder_path = device_cap::Tier::Cinder.model_path();
                        if mode == GenMode::Cascade && quench_path.exists() && cinder_path.exists() {
                            cascade_for_display(&cond, count, steps, &palette_name)
                        } else if quench_path.exists() {
                            generate_for_display(device_cap::Tier::Quench, &cond, count, steps, &palette_name)
                        } else {
                            generate_for_display(device_cap::Tier::Cinder, &cond, count, steps, &palette_name)
                        }
                    }
                    GenMode::Cinder => {
                        generate_for_display(device_cap::Tier::Cinder, &cond, count, steps, &palette_name)
                    }
                    GenMode::Anvil => {
                        anvil_for_display(&cond, count, steps, &palette_name)
                    }
                }
            };

            let mut s = state.lock().unwrap();
            s.generating = false;
            match result {
                Ok(result) => {
                    s.status = format!("done — {} sprites", count);
                    s.result_pixels = Some(result.sheet_pixels);
                    s.result_width = result.width;
                    s.result_height = result.height;
                    s.individual_sprites = result.sprites_f32;
                    s.sprite_class_ids = vec![cond.super_id; count as usize];
                }
                Err(e) => {
                    s.status = format!("error: {e}");
                }
            }
            ctx.request_repaint();
        });
    }
}

/// Brand colors — dark forge, cyan sparks + orange heat (matches logo).
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 217, 255); // #00d9ff cyan sparks
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(0, 150, 180); // muted cyan
const FORGE_HOT: egui::Color32 = egui::Color32::from_rgb(255, 120, 20); // anvil ember orange
const FORGE_DIM: egui::Color32 = egui::Color32::from_rgb(180, 80, 10); // muted ember
const BG_DARK: egui::Color32 = egui::Color32::from_rgb(12, 12, 18); // matches icon bg
const BG_CARD: egui::Color32 = egui::Color32::from_rgb(20, 20, 28);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(120, 120, 140);
const TEXT_BRIGHT: egui::Color32 = egui::Color32::from_rgb(220, 220, 235);
const BTN_BG: egui::Color32 = egui::Color32::from_rgb(30, 30, 42);
const BTN_HOVER: egui::Color32 = egui::Color32::from_rgb(35, 42, 52);
const BTN_SELECTED: egui::Color32 = egui::Color32::from_rgb(0, 217, 255); // #00d9ff
const BORDER_DIM: egui::Color32 = egui::Color32::from_rgb(30, 40, 50);
const EMBER: egui::Color32 = egui::Color32::from_rgb(40, 18, 5); // warm dark tint

fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG_DARK;
    visuals.window_fill = BG_DARK;
    visuals.override_text_color = Some(TEXT_BRIGHT);
    visuals.widgets.inactive.bg_fill = BTN_BG;
    visuals.widgets.inactive.weak_bg_fill = BTN_BG;
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    visuals.widgets.hovered.bg_fill = BTN_HOVER;
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_BRIGHT);
    visuals.widgets.active.bg_fill = ACCENT;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    visuals.selection.bg_fill = ACCENT;
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    // Group/card frames — dark with subtle orange-tinted border
    visuals.widgets.noninteractive.bg_fill = BG_CARD;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER_DIM);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    // Slider track
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(4);
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
        .corner_radius(egui::CornerRadius::same(6))
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

            // Header — clean, no technical info
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("PIXEL FORGE").size(28.0).color(ACCENT).strong());
                ui.add_space(4.0);
            });

            if let Some(ref err) = self.profile_error {
                ui.colored_label(egui::Color32::from_rgb(255, 80, 80), format!("error: {err}"));
            }

            ui.add_space(12.0);

            // Prompt input
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    section_label(ui, "DESCRIBE YOUR SPRITE");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let kb_btn = egui::Button::new(
                            egui::RichText::new("KB").size(12.0).color(TEXT_BRIGHT)
                        ).fill(BTN_BG).corner_radius(egui::CornerRadius::same(4)).min_size(egui::vec2(36.0, 24.0));
                        if ui.add(kb_btn).clicked() {
                            self.keyboard_requested = !self.keyboard_requested;
                        }
                    });
                });
                let _response = ui.add(
                    egui::TextEdit::multiline(&mut self.prompt_text)
                        .hint_text("knight with red cape, idle pose...")
                        .desired_rows(2)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Body)
                        .text_color(TEXT_BRIGHT)
                );
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

            // Class selector — two-tier: super-categories then class dirs
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                section_label(ui, "SPRITE CLASS");

                // Super-category row
                egui::Grid::new("super_grid")
                    .num_columns(5)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        for id in 0..class_cond::NUM_SUPER as u32 {
                            let selected = self.selected_super == id;
                            if styled_button(ui, class_cond::super_name(id), selected, egui::vec2(62.0, 32.0)) {
                                self.selected_super = id;
                                // Auto-select first class in this super
                                let classes = class_cond::classes_for_super(id);
                                if !classes.is_empty() {
                                    self.selected_class = classes[0].to_string();
                                }
                            }
                            if (id + 1) % 5 == 0 {
                                ui.end_row();
                            }
                        }
                    });

                ui.add_space(4.0);

                // Class dirs for selected super-category (scrollable)
                let classes = class_cond::classes_for_super(self.selected_super);
                egui::ScrollArea::horizontal().id_salt("class_scroll").show(ui, |ui| {
                    egui::Grid::new("class_grid")
                        .num_columns(5)
                        .spacing([4.0, 4.0])
                        .show(ui, |ui| {
                            for (i, name) in classes.iter().enumerate() {
                                let selected = self.selected_class == *name;
                                if styled_button(ui, name, selected, egui::vec2(72.0, 30.0)) {
                                    self.selected_class = name.to_string();
                                }
                                if (i + 1) % 5 == 0 {
                                    ui.end_row();
                                }
                            }
                        });
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

            // Advanced settings toggle
            ui.horizontal(|ui| {
                let toggle_text = if self.show_advanced { "hide advanced" } else { "advanced" };
                if ui.add(egui::Button::new(
                    egui::RichText::new(toggle_text).size(11.0).color(TEXT_DIM)
                ).fill(egui::Color32::TRANSPARENT)).clicked() {
                    self.show_advanced = !self.show_advanced;
                }
            });

            if self.show_advanced {
                ui.add_space(4.0);

                // Model mode selector
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    section_label(ui, "MODEL");
                    ui.horizontal(|ui| {
                        for mode in [GenMode::Cinder, GenMode::Quench, GenMode::Cascade, GenMode::Anvil] {
                            let selected = self.gen_mode == mode;
                            if styled_button(ui, mode.label(), selected, egui::vec2(80.0, 32.0)) {
                                self.gen_mode = mode;
                            }
                        }
                    });
                    ui.label(egui::RichText::new(self.gen_mode.desc()).size(10.0).color(
                        egui::Color32::from_rgb(80, 80, 100)
                    ));
                });

                ui.add_space(4.0);

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

                ui.add_space(4.0);

                // Device info (was at top, now in advanced)
                if let Some(ref profile) = self.profile {
                    ui.vertical_centered(|ui| {
                        let chip = format!("{} | {} MB | {}", profile.backend, profile.ram_mb, profile.tier);
                        ui.label(egui::RichText::new(chip).size(10.0).color(TEXT_DIM));
                    });
                }

                // Cluster info (desktop only)
                #[cfg(not(target_os = "android"))]
                {
                    let probing = *self.cluster_probing.lock().unwrap();
                    let cluster = self.cluster.lock().unwrap();
                    if probing {
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new("probing cluster...").size(10.0).color(TEXT_DIM));
                        });
                    } else if let Some(ref cs) = *cluster {
                        let active = cs.active_nodes().len();
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.use_cluster, "");
                            if self.use_cluster {
                                ui.label(egui::RichText::new(
                                    format!("cluster: {} nodes | {:.0} spr/s", active, cs.total_throughput)
                                ).size(10.0).color(ACCENT));
                            } else {
                                ui.label(egui::RichText::new("local only").size(10.0).color(TEXT_DIM));
                            }
                        });
                    }
                }
            }

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

                let (btn_color, text_color, stroke_color) = if generating {
                    (EMBER, FORGE_HOT, FORGE_DIM)
                } else if can_generate {
                    (FORGE_HOT, egui::Color32::WHITE, FORGE_DIM)
                } else {
                    (BTN_BG, TEXT_DIM, BTN_BG)
                };

                let btn = egui::Button::new(
                    egui::RichText::new(label).size(22.0).color(text_color).strong()
                )
                    .fill(btn_color)
                    .stroke(egui::Stroke::new(if can_generate { 2.0 } else { 0.0 }, stroke_color))
                    .corner_radius(egui::CornerRadius::same(14))
                    .min_size(egui::vec2(300.0, 60.0));

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

            if !self.reviewing {
                // Show sheet result
                if let Some(ref tex) = self.texture {
                    ui.vertical_centered(|ui| {
                        let available = ui.available_width().min(360.0);
                        let aspect = tex.size_vec2().y / tex.size_vec2().x;
                        let size = egui::vec2(available, available * aspect);
                        ui.add(
                            egui::Image::new(egui::load::SizedTexture::new(tex.id(), size))
                                .corner_radius(egui::CornerRadius::same(8))
                        );
                    });

                    // Review button — starts swipe mode
                    let has_sprites = {
                        let s = self.gen_state.lock().unwrap();
                        !s.individual_sprites.is_empty()
                    };
                    if has_sprites {
                        ui.add_space(8.0);
                        ui.vertical_centered(|ui| {
                            if styled_button(ui, "REVIEW SPRITES", false, egui::vec2(200.0, 40.0)) {
                                self.reviewing = true;
                                self.swipe_index = 0;
                                self.swipe_texture = None;
                            }
                        });
                    }
                }
            } else {
                // Swipe review mode
                let sprite_count = {
                    let s = self.gen_state.lock().unwrap();
                    s.individual_sprites.len()
                };

                if self.swipe_index < sprite_count {
                    // Show current sprite
                    let (pixels_f32, class_id) = {
                        let s = self.gen_state.lock().unwrap();
                        (s.individual_sprites[self.swipe_index].clone(),
                         s.sprite_class_ids[self.swipe_index])
                    };

                    // Convert f32 RGB to u8 RGBA for display
                    let size = 16usize;
                    let mut rgba = vec![0u8; size * size * 4];
                    for i in 0..size*size {
                        let r = (pixels_f32[i].clamp(0.0, 1.0) * 255.0) as u8;
                        let g = (pixels_f32[size*size + i].clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (pixels_f32[2*size*size + i].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i*4] = r;
                        rgba[i*4+1] = g;
                        rgba[i*4+2] = b;
                        rgba[i*4+3] = 255;
                    }

                    let color_image = egui::ColorImage::from_rgba_unmultiplied([size, size], &rgba);
                    let opts = egui::TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        minification: egui::TextureFilter::Nearest,
                        ..Default::default()
                    };
                    self.swipe_texture = Some(ctx.load_texture("swipe_sprite", color_image, opts));

                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new(
                            format!("REVIEW {}/{}", self.swipe_index + 1, sprite_count)
                        ).size(14.0).color(ACCENT).strong());
                        ui.add_space(8.0);

                        if let Some(ref tex) = self.swipe_texture {
                            let display_size = egui::vec2(256.0, 256.0);
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::new(tex.id(), display_size))
                                    .corner_radius(egui::CornerRadius::same(8))
                            );
                        }

                        ui.add_space(16.0);

                        // Swipe buttons
                        ui.horizontal(|ui| {
                            ui.add_space(40.0);
                            // Bad (left swipe)
                            let bad_btn = egui::Button::new(
                                egui::RichText::new("NOPE").size(18.0).color(egui::Color32::WHITE).strong()
                            )
                                .fill(egui::Color32::from_rgb(200, 50, 50))
                                .corner_radius(egui::CornerRadius::same(12))
                                .min_size(egui::vec2(120.0, 50.0));
                            if ui.add(bad_btn).clicked() {
                                self.swipe_store.record(pixels_f32.clone(), false, class_id);
                                self.swipe_index += 1;
                                self.swipe_texture = None;
                            }

                            ui.add_space(16.0);

                            // Good (right swipe)
                            let good_btn = egui::Button::new(
                                egui::RichText::new("KEEP").size(18.0).color(egui::Color32::WHITE).strong()
                            )
                                .fill(egui::Color32::from_rgb(50, 180, 50))
                                .corner_radius(egui::CornerRadius::same(12))
                                .min_size(egui::vec2(120.0, 50.0));
                            if ui.add(good_btn).clicked() {
                                self.swipe_store.record(pixels_f32.clone(), true, class_id);
                                self.swipe_index += 1;
                                self.swipe_texture = None;
                            }
                        });

                        ui.add_space(8.0);

                        // Stats
                        let stats = self.swipe_store.stats();
                        ui.label(egui::RichText::new(
                            format!("{} good / {} bad / {} total",
                                stats.good, stats.bad, stats.total_ever)
                        ).size(11.0).color(TEXT_DIM));

                        // LoRA training trigger
                        if self.swipe_store.should_retrain() {
                            let training = *self.lora_training.lock().unwrap();
                            if !training {
                                ui.add_space(8.0);
                                ui.label(egui::RichText::new(
                                    "enough feedback — LoRA training available"
                                ).size(11.0).color(ACCENT));
                            } else {
                                ui.label(egui::RichText::new(
                                    "LoRA training in progress..."
                                ).size(11.0).color(ACCENT));
                            }
                        }
                    });
                } else {
                    // Done reviewing
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("REVIEW COMPLETE").size(16.0).color(ACCENT).strong());
                        ui.add_space(8.0);
                        let stats = self.swipe_store.stats();
                        ui.label(egui::RichText::new(
                            format!("{} good / {} bad", stats.good, stats.bad)
                        ).size(13.0).color(TEXT_BRIGHT));
                        ui.add_space(16.0);

                        if styled_button(ui, "BACK TO FORGE", false, egui::vec2(200.0, 44.0)) {
                            self.reviewing = false;
                            let _ = self.swipe_store.save("swipes.bin");
                        }
                    });
                }
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

/// Result from generation — sheet pixels + individual sprites for swipe review.
struct GenResult {
    sheet_pixels: Vec<u8>,
    width: u32,
    height: u32,
    /// Individual sprites as channel-first f32 (3*32*32 each).
    sprites_f32: Vec<Vec<f32>>,
}

/// Extract channel-first f32 pixels from an RgbaImage for swipe storage.
fn rgba_to_f32(img: &image::RgbaImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            out[idx] = px[0] as f32 / 255.0;
            out[w * h + idx] = px[1] as f32 / 255.0;
            out[2 * w * h + idx] = px[2] as f32 / 255.0;
        }
    }
    out
}

/// Generate sprites and return raw RGBA pixels for display.
fn generate_for_display(
    tier: Tier,
    cond: &crate::class_cond::ClassCond,
    count: u32,
    steps: usize,
    palette_name: &str,
) -> anyhow::Result<GenResult> {
    let pal = palette::load_palette(palette_name)?;
    let img_size = 32u32;

    let model_path = tier.model_path();
    let (actual_tier, actual_file) = if model_path.exists() {
        (tier, model_path.to_string_lossy().to_string())
    } else {
        let fb = device_cap::best_available();
        (fb, fb.model_path().to_string_lossy().to_string())
    };

    let raw_images = match actual_tier {
        Tier::Cinder => crate::train::sample(&actual_file, cond, img_size, count, steps)?,
        Tier::Quench | Tier::Anvil => crate::train::sample_medium(&actual_file, cond, img_size, count, steps)?,
    };

    let processed: Vec<image::RgbaImage> = raw_images
        .into_iter()
        .map(|img| {
            let snapped = grid::snap_to_grid(&img, img_size);
            palette::quantize(&snapped, &pal)
        })
        .collect();

    let sprites_f32: Vec<Vec<f32>> = processed.iter().map(|img| rgba_to_f32(img)).collect();

    let sheet = crate::sheet::pack_grid(&processed, count.min(8));
    let w = sheet.width();
    let h = sheet.height();
    let pixels = sheet.into_raw();

    Ok(GenResult { sheet_pixels: pixels, width: w, height: h, sprites_f32 })
}

/// MoE cascade: Quench foundation → Cinder detail.
fn cascade_for_display(
    cond: &crate::class_cond::ClassCond,
    count: u32,
    steps: usize,
    palette_name: &str,
) -> anyhow::Result<GenResult> {
    let pal = palette::load_palette(palette_name)?;
    let img_size = 32u32;

    let quench_path = device_cap::Tier::Quench.model_path();
    let cinder_path = device_cap::Tier::Cinder.model_path();

    if !quench_path.exists() || !cinder_path.exists() {
        anyhow::bail!("cascade needs both Quench and Cinder models");
    }

    // Split steps: 60% Quench foundation, 40% Cinder detail
    let quench_steps = (steps as f32 * 0.6).max(1.0) as usize;
    let cinder_steps = steps - quench_steps;

    let config = crate::moe::CascadeConfig {
        quench_steps,
        cinder_steps,
    };

    let experts_path_buf = cinder_path
        .parent().unwrap_or(std::path::Path::new("."))
        .join("experts.safetensors");
    let experts_path = if experts_path_buf.exists() {
        Some(experts_path_buf.to_string_lossy().to_string())
    } else {
        None
    };

    let raw_images = crate::moe::cascade_sample(
        &quench_path.to_string_lossy(),
        &cinder_path.to_string_lossy(),
        experts_path.as_deref(),
        cond, img_size, count, &config,
    )?;

    let processed: Vec<image::RgbaImage> = raw_images
        .into_iter()
        .map(|img| {
            let snapped = grid::snap_to_grid(&img, img_size);
            palette::quantize(&snapped, &pal)
        })
        .collect();

    let sprites_f32: Vec<Vec<f32>> = processed.iter().map(|img| rgba_to_f32(img)).collect();

    let sheet = crate::sheet::pack_grid(&processed, count.min(8));
    let w = sheet.width();
    let h = sheet.height();
    let pixels = sheet.into_raw();

    Ok(GenResult { sheet_pixels: pixels, width: w, height: h, sprites_f32 })
}

/// Desktop pipeline: Anvil single-stage.
fn anvil_for_display(
    cond: &crate::class_cond::ClassCond,
    count: u32,
    steps: usize,
    palette_name: &str,
) -> anyhow::Result<GenResult> {
    let pal = palette::load_palette(palette_name)?;
    let img_size = 32u32;

    let anvil_path = device_cap::Tier::Anvil.model_path();
    if !anvil_path.exists() {
        anyhow::bail!("Anvil model not found: {}", anvil_path.display());
    }

    let raw_images = crate::moe::anvil_sample(
        &anvil_path.to_string_lossy(),
        cond, img_size, count, steps,
    )?;

    let processed: Vec<image::RgbaImage> = raw_images
        .into_iter()
        .map(|img| {
            let snapped = grid::snap_to_grid(&img, img_size);
            palette::quantize(&snapped, &pal)
        })
        .collect();

    let sprites_f32: Vec<Vec<f32>> = processed.iter().map(|img| rgba_to_f32(img)).collect();

    let sheet = crate::sheet::pack_grid(&processed, count.min(8));
    let w = sheet.width();
    let h = sheet.height();
    let pixels = sheet.into_raw();

    Ok(GenResult { sheet_pixels: pixels, width: w, height: h, sprites_f32 })
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
