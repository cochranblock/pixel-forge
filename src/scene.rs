// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Scene representation and rendering for 8×8 sprite grids.
//! 256×256 pixels (8×8 cells × 32×32 sprites).
//!
//! SceneGrid is the core data structure shared between the Combiner model
//! and the rendering/UI layers. Rule-seeded generation creates bootstrap
//! training data for the Combiner before any user data exists.

use image::RgbaImage;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::tiny_unet::NUM_CLASSES;

/// Grid dimensions.
pub const GRID_W: usize = 8;
pub const GRID_H: usize = 8;
pub const GRID_CELLS: usize = GRID_W * GRID_H; // 64
pub const CELL_PX: u32 = 32;
pub const SCENE_PX: u32 = GRID_W as u32 * CELL_PX; // 256

/// Empty cell class ID — one past the real classes.
pub const EMPTY_CLASS: u32 = NUM_CLASSES as u32; // 15
/// Total class IDs including empty.
pub const TOTAL_CLASSES: usize = NUM_CLASSES + 1; // 16

/// Cell encoding dimension: 16 one-hot class + 2 position + 1 occupied = 19.
pub const CELL_DIM: usize = TOTAL_CLASSES + 2 + 1; // 19

/// A single cell in the scene grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    /// Class ID (0-14 = sprite class, 15 = empty).
    pub class_id: u32,
    /// Grid position (col, row), normalized to [0, 1].
    pub col: usize,
    pub row: usize,
}

impl Cell {
    pub fn empty(col: usize, row: usize) -> Self {
        Self { class_id: EMPTY_CLASS, col, row }
    }

    pub fn is_empty(&self) -> bool {
        self.class_id == EMPTY_CLASS
    }

    /// Encode to CELL_DIM floats: one-hot class + normalized (col, row) + occupied flag.
    pub fn encode(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; CELL_DIM];
        // One-hot class
        v[self.class_id as usize] = 1.0;
        // Normalized position
        v[TOTAL_CLASSES] = self.col as f32 / (GRID_W - 1) as f32;
        v[TOTAL_CLASSES + 1] = self.row as f32 / (GRID_H - 1) as f32;
        // Occupied flag
        v[TOTAL_CLASSES + 2] = if self.is_empty() { 0.0 } else { 1.0 };
        v
    }
}

/// 8×8 scene grid — 64 cells, each holding a sprite class or empty.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneGrid {
    /// Row-major: cells[row * GRID_W + col]
    pub cells: Vec<Cell>,
}

impl SceneGrid {
    /// Create an empty grid.
    pub fn empty() -> Self {
        let mut cells = Vec::with_capacity(GRID_CELLS);
        for row in 0..GRID_H {
            for col in 0..GRID_W {
                cells.push(Cell::empty(col, row));
            }
        }
        Self { cells }
    }

    /// Get cell at (col, row).
    pub fn get(&self, col: usize, row: usize) -> &Cell {
        &self.cells[row * GRID_W + col]
    }

    /// Set cell class at (col, row).
    pub fn set(&mut self, col: usize, row: usize, class_id: u32) {
        self.cells[row * GRID_W + col].class_id = class_id;
    }

    /// Encode full grid to flat f32 for model input: [64 × CELL_DIM].
    pub fn encode(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(GRID_CELLS * CELL_DIM);
        for cell in &self.cells {
            v.extend_from_slice(&cell.encode());
        }
        v
    }

    /// Decode model output (class logits per cell) back into a grid.
    /// `logits`: [64, TOTAL_CLASSES] — takes argmax per cell.
    pub fn from_logits(logits: &[f32]) -> Self {
        let mut cells = Vec::with_capacity(GRID_CELLS);
        for i in 0..GRID_CELLS {
            let row = i / GRID_W;
            let col = i % GRID_W;
            let start = i * TOTAL_CLASSES;
            let slice = &logits[start..start + TOTAL_CLASSES];
            let class_id = slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(EMPTY_CLASS);
            cells.push(Cell { class_id, col, row });
        }
        Self { cells }
    }

    /// Sample from logits with temperature. Each cell independently sampled.
    pub fn from_logits_sampled(logits: &[f32], temperature: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut cells = Vec::with_capacity(GRID_CELLS);
        for i in 0..GRID_CELLS {
            let row = i / GRID_W;
            let col = i % GRID_W;
            let start = i * TOTAL_CLASSES;
            let slice = &logits[start..start + TOTAL_CLASSES];
            let class_id = sample_categorical(slice, temperature, &mut rng);
            cells.push(Cell { class_id, col, row });
        }
        Self { cells }
    }

    /// Count non-empty cells.
    pub fn sprite_count(&self) -> usize {
        self.cells.iter().filter(|c| !c.is_empty()).count()
    }

    /// Get all (col, row, class_id) triples for non-empty cells.
    pub fn placed_sprites(&self) -> Vec<(usize, usize, u32)> {
        self.cells.iter()
            .filter(|c| !c.is_empty())
            .map(|c| (c.col, c.row, c.class_id))
            .collect()
    }

    /// Create a masked copy — randomly set `frac` of cells to empty.
    /// Returns (masked_grid, mask_indices) where mask_indices are the cells that were masked.
    pub fn masked(&self, frac: f32) -> (Self, Vec<usize>) {
        let mut rng = rand::thread_rng();
        let mut grid = self.clone();
        let mut masked = Vec::new();
        for i in 0..GRID_CELLS {
            if rng.r#gen::<f32>() < frac {
                grid.cells[i].class_id = EMPTY_CLASS;
                masked.push(i);
            }
        }
        (grid, masked)
    }

    /// Render the scene to a 256×256 RGBA image.
    /// `sprite_lookup`: given (class_id), returns 32×32 RGB channel-first pixels,
    /// or None to leave the cell transparent.
    pub fn render<F>(&self, mut sprite_lookup: F) -> RgbaImage
    where
        F: FnMut(u32) -> Option<Vec<f32>>,
    {
        let mut img = RgbaImage::new(SCENE_PX, SCENE_PX);

        for cell in &self.cells {
            if cell.is_empty() { continue; }
            if let Some(pixels) = sprite_lookup(cell.class_id) {
                let ox = cell.col as u32 * CELL_PX;
                let oy = cell.row as u32 * CELL_PX;
                let n = (CELL_PX * CELL_PX) as usize;
                for y in 0..CELL_PX {
                    for x in 0..CELL_PX {
                        let idx = (y * CELL_PX + x) as usize;
                        let r = (pixels[idx].clamp(0.0, 1.0) * 255.0) as u8;
                        let g = (pixels[n + idx].clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (pixels[2 * n + idx].clamp(0.0, 1.0) * 255.0) as u8;
                        img.put_pixel(ox + x, oy + y, image::Rgba([r, g, b, 255]));
                    }
                }
            }
        }

        img
    }

    /// Render with a simple colored placeholder per class (no sprite data needed).
    /// Good for debugging and bootstrap visualization.
    pub fn render_placeholder(&self) -> RgbaImage {
        let colors: [(u8, u8, u8); 16] = [
            (65, 105, 225),   // 0 character — royal blue
            (192, 192, 192),  // 1 weapon — silver
            (148, 0, 211),    // 2 potion — purple
            (139, 119, 101),  // 3 terrain — tan
            (220, 20, 60),    // 4 enemy — crimson
            (34, 139, 34),    // 5 tree — forest green
            (160, 82, 45),    // 6 building — sienna
            (255, 165, 0),    // 7 animal — orange
            (255, 255, 0),    // 8 effect — yellow
            (255, 127, 80),   // 9 food — coral
            (105, 105, 105),  // 10 armor — dim gray
            (210, 180, 140),  // 11 tool — tan
            (70, 130, 180),   // 12 vehicle — steel blue
            (169, 169, 169),  // 13 ui — dark gray
            (128, 128, 128),  // 14 misc — gray
            (0, 0, 0),        // 15 empty — black (shouldn't render)
        ];

        let mut img = RgbaImage::new(SCENE_PX, SCENE_PX);
        // Fill with dark background
        for p in img.pixels_mut() {
            *p = image::Rgba([20, 20, 30, 255]);
        }

        for cell in &self.cells {
            if cell.is_empty() { continue; }
            let (r, g, b) = colors[cell.class_id as usize % 16];
            let ox = cell.col as u32 * CELL_PX;
            let oy = cell.row as u32 * CELL_PX;
            // Draw filled square with 1px border
            for y in 1..CELL_PX - 1 {
                for x in 1..CELL_PX - 1 {
                    img.put_pixel(ox + x, oy + y, image::Rgba([r, g, b, 255]));
                }
            }
        }

        img
    }
}

/// Softmax + weighted random sampling from categorical logits.
fn sample_categorical(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> u32 {
    let t = temperature.max(0.01);
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| ((l - max_logit) / t).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

    let mut r: f32 = rng.r#gen();
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ─── Rule-seeded scene generation (bootstrap training data) ───

/// Biome type determines placement rules.
#[derive(Clone, Copy, Debug)]
pub enum Biome {
    Forest,
    Dungeon,
    Village,
    Cave,
    Plains,
}

impl Biome {
    pub fn all() -> &'static [Biome] {
        &[Biome::Forest, Biome::Dungeon, Biome::Village, Biome::Cave, Biome::Plains]
    }
}

/// Class IDs (matching tiny_unet.rs).
mod class {
    pub const CHARACTER: u32 = 0;
    pub const WEAPON: u32 = 1;
    pub const POTION: u32 = 2;
    pub const TERRAIN: u32 = 3;
    pub const ENEMY: u32 = 4;
    pub const TREE: u32 = 5;
    pub const BUILDING: u32 = 6;
    pub const ANIMAL: u32 = 7;
    pub const EFFECT: u32 = 8;
    pub const FOOD: u32 = 9;
    pub const ARMOR: u32 = 10;
    pub const TOOL: u32 = 11;
    pub const VEHICLE: u32 = 12;
    pub const UI: u32 = 13;
    pub const MISC: u32 = 14;
}

/// Row bands — determines which rows a class can appear in.
/// Row 0 = top (sky), Row 7 = bottom (ground).
struct PlacementRule {
    class_id: u32,
    min_row: usize,
    max_row: usize,
    density: f32, // probability of filling an eligible cell
}

fn rules_for_biome(biome: Biome) -> Vec<PlacementRule> {
    match biome {
        Biome::Forest => vec![
            PlacementRule { class_id: class::TERRAIN,   min_row: 6, max_row: 7, density: 0.95 },
            PlacementRule { class_id: class::TREE,      min_row: 3, max_row: 5, density: 0.40 },
            PlacementRule { class_id: class::CHARACTER,  min_row: 4, max_row: 5, density: 0.15 },
            PlacementRule { class_id: class::ANIMAL,    min_row: 5, max_row: 6, density: 0.10 },
            PlacementRule { class_id: class::ENEMY,     min_row: 4, max_row: 6, density: 0.08 },
            PlacementRule { class_id: class::FOOD,      min_row: 5, max_row: 6, density: 0.05 },
            PlacementRule { class_id: class::POTION,    min_row: 5, max_row: 6, density: 0.03 },
            PlacementRule { class_id: class::EFFECT,    min_row: 1, max_row: 3, density: 0.05 },
        ],
        Biome::Dungeon => vec![
            PlacementRule { class_id: class::TERRAIN,   min_row: 6, max_row: 7, density: 0.90 },
            PlacementRule { class_id: class::BUILDING,  min_row: 2, max_row: 5, density: 0.20 },
            PlacementRule { class_id: class::ENEMY,     min_row: 3, max_row: 6, density: 0.20 },
            PlacementRule { class_id: class::CHARACTER,  min_row: 4, max_row: 5, density: 0.10 },
            PlacementRule { class_id: class::WEAPON,    min_row: 4, max_row: 6, density: 0.12 },
            PlacementRule { class_id: class::ARMOR,     min_row: 4, max_row: 6, density: 0.08 },
            PlacementRule { class_id: class::POTION,    min_row: 5, max_row: 6, density: 0.10 },
            PlacementRule { class_id: class::EFFECT,    min_row: 2, max_row: 5, density: 0.08 },
            PlacementRule { class_id: class::TOOL,      min_row: 5, max_row: 6, density: 0.05 },
        ],
        Biome::Village => vec![
            PlacementRule { class_id: class::TERRAIN,   min_row: 6, max_row: 7, density: 0.95 },
            PlacementRule { class_id: class::BUILDING,  min_row: 2, max_row: 5, density: 0.30 },
            PlacementRule { class_id: class::CHARACTER,  min_row: 4, max_row: 6, density: 0.20 },
            PlacementRule { class_id: class::ANIMAL,    min_row: 5, max_row: 6, density: 0.10 },
            PlacementRule { class_id: class::TREE,      min_row: 3, max_row: 5, density: 0.15 },
            PlacementRule { class_id: class::FOOD,      min_row: 5, max_row: 6, density: 0.08 },
            PlacementRule { class_id: class::TOOL,      min_row: 5, max_row: 6, density: 0.06 },
            PlacementRule { class_id: class::VEHICLE,   min_row: 5, max_row: 6, density: 0.05 },
        ],
        Biome::Cave => vec![
            PlacementRule { class_id: class::TERRAIN,   min_row: 0, max_row: 1, density: 0.80 }, // ceiling
            PlacementRule { class_id: class::TERRAIN,   min_row: 6, max_row: 7, density: 0.90 }, // floor
            PlacementRule { class_id: class::ENEMY,     min_row: 3, max_row: 6, density: 0.15 },
            PlacementRule { class_id: class::CHARACTER,  min_row: 4, max_row: 5, density: 0.10 },
            PlacementRule { class_id: class::POTION,    min_row: 5, max_row: 6, density: 0.08 },
            PlacementRule { class_id: class::EFFECT,    min_row: 2, max_row: 5, density: 0.12 },
            PlacementRule { class_id: class::MISC,      min_row: 5, max_row: 6, density: 0.06 },
        ],
        Biome::Plains => vec![
            PlacementRule { class_id: class::TERRAIN,   min_row: 6, max_row: 7, density: 0.95 },
            PlacementRule { class_id: class::CHARACTER,  min_row: 4, max_row: 5, density: 0.12 },
            PlacementRule { class_id: class::ANIMAL,    min_row: 4, max_row: 6, density: 0.15 },
            PlacementRule { class_id: class::TREE,      min_row: 4, max_row: 5, density: 0.10 },
            PlacementRule { class_id: class::ENEMY,     min_row: 3, max_row: 6, density: 0.08 },
            PlacementRule { class_id: class::FOOD,      min_row: 5, max_row: 6, density: 0.05 },
            PlacementRule { class_id: class::EFFECT,    min_row: 1, max_row: 3, density: 0.04 },
        ],
    }
}

/// Generate one rule-seeded scene for a given biome.
pub fn generate_seeded(biome: Biome) -> SceneGrid {
    let mut rng = rand::thread_rng();
    let mut grid = SceneGrid::empty();
    let rules = rules_for_biome(biome);

    // Apply rules in order (earlier rules = higher priority for cell ownership)
    for rule in &rules {
        for row in rule.min_row..=rule.max_row {
            for col in 0..GRID_W {
                let cell = grid.get(col, row);
                // Only fill empty cells (earlier rules take priority)
                if !cell.is_empty() { continue; }
                if rng.r#gen::<f32>() < rule.density {
                    grid.set(col, row, rule.class_id);
                }
            }
        }
    }

    // Post-pass: enforce adjacency constraints
    enforce_constraints(&mut grid);

    grid
}

/// Enforce spatial relationships:
/// - Weapons/armor must be adjacent to a character or enemy
/// - Characters must be on or above terrain
/// - No floating buildings (must touch terrain or another building below)
fn enforce_constraints(grid: &mut SceneGrid) {
    // Remove floating weapons/armor
    for row in 0..GRID_H {
        for col in 0..GRID_W {
            let cid = grid.get(col, row).class_id;
            if cid == class::WEAPON || cid == class::ARMOR {
                if !has_adjacent(grid, col, row, &[class::CHARACTER, class::ENEMY]) {
                    grid.set(col, row, EMPTY_CLASS);
                }
            }
        }
    }

    // Remove floating buildings (must have terrain or building below)
    for row in (0..GRID_H).rev() {
        for col in 0..GRID_W {
            if grid.get(col, row).class_id == class::BUILDING {
                if row < GRID_H - 1 {
                    let below = grid.get(col, row + 1).class_id;
                    if below != class::TERRAIN && below != class::BUILDING {
                        grid.set(col, row, EMPTY_CLASS);
                    }
                }
            }
        }
    }
}

/// Check if any of the 4-connected neighbors has one of the target classes.
fn has_adjacent(grid: &SceneGrid, col: usize, row: usize, targets: &[u32]) -> bool {
    let deltas: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (dc, dr) in deltas {
        let nc = col as i32 + dc;
        let nr = row as i32 + dr;
        if nc >= 0 && nc < GRID_W as i32 && nr >= 0 && nr < GRID_H as i32 {
            let neighbor = grid.get(nc as usize, nr as usize);
            if targets.contains(&neighbor.class_id) {
                return true;
            }
        }
    }
    false
}

/// Generate `count` bootstrap training scenes across all biomes.
/// Returns scenes ready for Combiner training (masked prediction).
pub fn generate_bootstrap(count: usize) -> Vec<SceneGrid> {
    let biomes = Biome::all();
    let mut scenes = Vec::with_capacity(count);
    for i in 0..count {
        let biome = biomes[i % biomes.len()];
        scenes.push(generate_seeded(biome));
    }
    scenes
}

/// Batch encode scenes for Combiner training.
/// Returns flat f32: [count × GRID_CELLS × CELL_DIM].
pub fn encode_batch(scenes: &[SceneGrid]) -> Vec<f32> {
    let mut v = Vec::with_capacity(scenes.len() * GRID_CELLS * CELL_DIM);
    for scene in scenes {
        v.extend_from_slice(&scene.encode());
    }
    v
}

/// Extract class-ID targets for masked prediction loss.
/// Returns [count × GRID_CELLS] as u32 class IDs.
pub fn target_class_ids(scenes: &[SceneGrid]) -> Vec<u32> {
    let mut v = Vec::with_capacity(scenes.len() * GRID_CELLS);
    for scene in scenes {
        for cell in &scene.cells {
            v.push(cell.class_id);
        }
    }
    v
}
