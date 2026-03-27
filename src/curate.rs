// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Dataset curation — slice sprite sheets into individual tiles,
//! resize to target, sort into class directories for training.
//!
//! Training data attribution (see data/SOURCES.md):
//! - Dungeon Crawl Stone Soup tiles (CC0)
//! - DawnLike v1.81 by DragonDePlatino, palette by DawnBringer (CC-BY 4.0)
//! - Kenney Roguelike/RPG, Platformer, 1-Bit packs (CC0)
//! - Hyptosis tiles (CC-BY 3.0)
//! - David E. Gervais roguelike tiles (CC-BY 3.0)

use anyhow::Result;
use image::{GenericImageView, RgbaImage};
use std::path::{Path, PathBuf};

/// Slice a sprite sheet into individual tiles of tile_size × tile_size.
/// Skips fully transparent tiles.
fn slice_sheet(sheet: &RgbaImage, tile_size: u32) -> Vec<RgbaImage> {
    let mut tiles = Vec::new();
    let cols = sheet.width() / tile_size;
    let rows = sheet.height() / tile_size;

    for row in 0..rows {
        for col in 0..cols {
            let x = col * tile_size;
            let y = row * tile_size;
            let tile = sheet.view(x, y, tile_size, tile_size).to_image();

            // Skip fully transparent tiles
            let has_content = tile.pixels().any(|p| p[3] > 0);
            if has_content {
                tiles.push(tile);
            }
        }
    }
    tiles
}

/// Resize a tile to target_size × target_size using nearest-neighbor (preserves pixel art).
fn resize_tile(tile: &RgbaImage, target_size: u32) -> RgbaImage {
    if tile.width() == target_size && tile.height() == target_size {
        return tile.clone();
    }
    image::imageops::resize(tile, target_size, target_size, image::imageops::FilterType::Nearest)
}

/// Guess class from filename/path.
fn guess_class(path: &Path) -> &'static str {
    let s = path.to_string_lossy().to_lowercase();

    // Characters / players
    if s.contains("player") || s.contains("hero") || s.contains("human")
        || s.contains("knight") || s.contains("mage") || s.contains("rogue")
        || s.contains("warrior") || s.contains("character") || s.contains("npc")
    {
        return "character";
    }
    // Enemies / monsters
    if s.contains("monster") || s.contains("enemy") || s.contains("demon")
        || s.contains("dragon") || s.contains("undead") || s.contains("skeleton")
        || s.contains("zombie") || s.contains("goblin") || s.contains("orc")
        || s.contains("beast") || s.contains("creature") || s.contains("boss")
        || s.contains("spider") || s.contains("snake") || s.contains("bat")
        || s.contains("slime") || s.contains("ghost") || s.contains("rat")
    {
        return "enemy";
    }
    // Weapons
    if s.contains("weapon") || s.contains("sword") || s.contains("axe")
        || s.contains("bow") || s.contains("arrow") || s.contains("dagger")
        || s.contains("spear") || s.contains("staff") || s.contains("mace")
        || s.contains("hammer") || s.contains("blade")
    {
        return "weapon";
    }
    // Armor
    if s.contains("armor") || s.contains("armour") || s.contains("shield")
        || s.contains("helm") || s.contains("boot") || s.contains("glove")
        || s.contains("chest") || s.contains("cloak") || s.contains("ring")
    {
        return "armor";
    }
    // Potions
    if s.contains("potion") || s.contains("flask") || s.contains("vial")
        || s.contains("bottle") || s.contains("elixir") || s.contains("brew")
    {
        return "potion";
    }
    // Terrain
    if s.contains("floor") || s.contains("wall") || s.contains("ground")
        || s.contains("tile") || s.contains("terrain") || s.contains("grass")
        || s.contains("dirt") || s.contains("stone") || s.contains("water")
        || s.contains("sand") || s.contains("snow") || s.contains("road")
        || s.contains("path") || s.contains("lava") || s.contains("cave")
        || s.contains("dungeon") || s.contains("dngn")
    {
        return "terrain";
    }
    // Buildings
    if s.contains("house") || s.contains("building") || s.contains("castle")
        || s.contains("door") || s.contains("window") || s.contains("bridge")
        || s.contains("gate") || s.contains("tower") || s.contains("stair")
        || s.contains("shop") || s.contains("inn") || s.contains("church")
    {
        return "building";
    }
    // Trees / flora
    if s.contains("tree") || s.contains("bush") || s.contains("plant")
        || s.contains("flower") || s.contains("mushroom") || s.contains("vine")
        || s.contains("hedge") || s.contains("forest") || s.contains("leaf")
    {
        return "tree";
    }
    // Animals
    if s.contains("animal") || s.contains("horse") || s.contains("dog")
        || s.contains("cat") || s.contains("bird") || s.contains("fish")
        || s.contains("wolf") || s.contains("bear") || s.contains("deer")
    {
        return "animal";
    }
    // Effects
    if s.contains("effect") || s.contains("spell") || s.contains("magic")
        || s.contains("fire") || s.contains("ice") || s.contains("lightning")
        || s.contains("explosion") || s.contains("particle") || s.contains("aura")
        || s.contains("cloud") || s.contains("smoke") || s.contains("beam")
    {
        return "effect";
    }
    // Food
    if s.contains("food") || s.contains("meat") || s.contains("bread")
        || s.contains("fruit") || s.contains("cheese") || s.contains("pie")
    {
        return "food";
    }
    // Tools
    if s.contains("tool") || s.contains("pick") || s.contains("shovel")
        || s.contains("torch") || s.contains("lantern") || s.contains("rope")
        || s.contains("key") || s.contains("lock") || s.contains("book")
        || s.contains("scroll") || s.contains("map") || s.contains("gem")
        || s.contains("coin") || s.contains("gold") || s.contains("treasure")
    {
        return "tool";
    }
    // Vehicles
    if s.contains("vehicle") || s.contains("car") || s.contains("cart")
        || s.contains("boat") || s.contains("ship") || s.contains("wagon")
    {
        return "vehicle";
    }
    // UI
    if s.contains("gui") || s.contains("icon") || s.contains("button")
        || s.contains("cursor") || s.contains("hud") || s.contains("menu")
        || s.contains("ui") || s.contains("frame") || s.contains("bar")
    {
        return "ui";
    }

    "misc"
}

/// Walk a directory tree, find all PNGs, try to slice sprite sheets or use as-is.
fn collect_pngs(dir: &Path) -> Result<Vec<(PathBuf, RgbaImage)>> {
    let mut results = Vec::new();

    if !dir.exists() {
        return Ok(results);
    }

    for path in crate::walk_pngs(dir.to_str().unwrap_or(""))
    {
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                results.push((path, rgba));
            }
            Err(e) => {
                eprintln!("  skip {}: {}", path.display(), e);
            }
        }
    }

    Ok(results)
}

/// Curate raw downloads into class-sorted training directories.
pub fn curate(raw_dir: &str, output_dir: &str, target_size: u32) -> Result<()> {
    let raw = Path::new(raw_dir);
    let out = Path::new(output_dir);

    println!("curating {} → {}", raw_dir, output_dir);
    println!("target size: {}x{}", target_size, target_size);

    let mut total = 0usize;
    let mut class_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    // Process each raw source directory
    let sources = std::fs::read_dir(raw)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    for source_entry in &sources {
        let source_name = source_entry.file_name().to_string_lossy().to_string();
        println!("\n  processing {}...", source_name);

        let pngs = collect_pngs(&source_entry.path())?;
        let mut source_count = 0;

        for (path, img) in &pngs {
            let (w, h) = (img.width(), img.height());

            // Decide: is this a sprite sheet or a single sprite?
            let tiles = if w > target_size * 2 || h > target_size * 2 {
                // Likely a sprite sheet — try to detect tile size
                let tile_size = detect_tile_size(w, h, target_size);
                if tile_size > 0 {
                    slice_sheet(img, tile_size)
                } else {
                    // Can't detect tile grid — skip or use as single
                    vec![img.clone()]
                }
            } else if w == target_size && h == target_size {
                vec![img.clone()]
            } else if w <= target_size * 4 && h <= target_size * 4 {
                // Small enough to be a single sprite — resize
                vec![img.clone()]
            } else {
                // Too big, unknown structure — skip
                continue;
            };

            let class = guess_class(path);
            let class_dir = out.join(class);
            std::fs::create_dir_all(&class_dir)?;

            for tile in &tiles {
                let resized = resize_tile(tile, target_size);

                // Skip mostly-empty tiles (less than 5% non-transparent)
                let total_pixels = (target_size * target_size) as usize;
                let opaque_pixels = resized.pixels().filter(|p| p[3] > 128).count();
                if opaque_pixels < total_pixels / 20 {
                    continue;
                }

                let count = class_counts.entry(class.to_string()).or_insert(0);
                let filename = format!("{}_{:05}.png", source_name, *count);
                resized.save(class_dir.join(&filename))?;
                *count += 1;
                source_count += 1;
                total += 1;
            }
        }

        println!("    extracted {} tiles", source_count);
    }

    println!("\n=== Curation complete ===");
    println!("Total: {} tiles across {} classes", total, class_counts.len());
    for (class, count) in class_counts.iter() {
        println!("  {}: {}", class, count);
    }

    Ok(())
}

/// Try to detect tile size from sheet dimensions.
fn detect_tile_size(w: u32, h: u32, _target: u32) -> u32 {
    // Common tile sizes, prefer ones matching or close to target
    let candidates = [32, 16, 24, 48, 64, 8];
    for &size in &candidates {
        if w % size == 0 && h % size == 0 && w / size >= 2 {
            return size;
        }
    }
    0
}
