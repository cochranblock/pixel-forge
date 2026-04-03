// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//! Hybrid class conditioning: super-categories + binary tags.
//!
//! Replaces the old fixed NUM_CLASSES=16 integer embedding with:
//! - 10 super-categories (small embedding table, MoE routing)
//! - 12 binary tags (linear projection, composable traits)
//!
//! New classes = new tag combos. Zero retraining needed.

/// Number of super-categories (real).
pub const NUM_SUPER: usize = 10;
/// Including CFG null token.
pub const NUM_SUPER_WITH_NULL: usize = 11;
/// Number of binary trait tags.
pub const NUM_TAGS: usize = 12;
/// CFG null super-category (unconditional).
pub const CFG_NULL_SUPER: u32 = NUM_SUPER as u32; // 10
/// CFG null tags (all zeros).
pub const CFG_NULL_TAGS: [f32; NUM_TAGS] = [0.0; NUM_TAGS];

/// Tag indices for readability.
pub mod tag {
    pub const ALIVE: usize = 0;
    pub const HUMANOID: usize = 1;
    pub const HELD_ITEM: usize = 2;
    pub const WORN: usize = 3;
    pub const NATURE: usize = 4;
    pub const BUILT: usize = 5;
    pub const MAGICAL: usize = 6;
    pub const TECH: usize = 7;
    pub const SMALL: usize = 8;
    pub const HOSTILE: usize = 9;
    pub const EDIBLE: usize = 10;
    pub const UI: usize = 11;
}

/// Super-category IDs.
pub mod super_cat {
    pub const HUMANOID: u32 = 0;
    pub const CREATURE: u32 = 1;
    pub const MONSTER: u32 = 2;
    pub const WEAPON: u32 = 3;
    pub const EQUIPMENT: u32 = 4;
    pub const CONSUMABLE: u32 = 5;
    pub const TERRAIN: u32 = 6;
    pub const NATURE: u32 = 7;
    pub const STRUCTURE: u32 = 8;
    pub const UI_FX: u32 = 9;
}

/// Class conditioning: super-category + binary tags.
#[derive(Clone, Debug)]
pub struct ClassCond {
    pub super_id: u32,
    pub tags: [f32; NUM_TAGS],
}

impl ClassCond {
    pub fn new(super_id: u32, tags: [f32; NUM_TAGS]) -> Self {
        Self { super_id, tags }
    }

    /// CFG null conditioning (unconditional).
    pub fn null() -> Self {
        Self { super_id: CFG_NULL_SUPER, tags: CFG_NULL_TAGS }
    }
}

/// Build tags from a slice of tag indices.
fn tags(indices: &[usize]) -> [f32; NUM_TAGS] {
    let mut t = [0.0f32; NUM_TAGS];
    for &i in indices {
        t[i] = 1.0;
    }
    t
}

use tag::*;
use super_cat as sc;

/// Look up class conditioning for a directory name.
/// Unknown dirs default to ui_fx super-category with no tags.
pub fn lookup(name: &str) -> ClassCond {
    let (s, t) = match name {
        // === Humanoids (super 0) ===
        "character"    => (sc::HUMANOID, tags(&[ALIVE, HUMANOID])),
        "elf"          => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, MAGICAL])),
        "dwarf"        => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, SMALL])),
        "orc"          => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, HOSTILE])),
        "knight"       => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, WORN])),
        "villager"     => (sc::HUMANOID, tags(&[ALIVE, HUMANOID])),
        "goblin"       => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, HOSTILE, SMALL])),
        "skeleton"     => (sc::HUMANOID, tags(&[HUMANOID, HOSTILE])),
        "zombie"       => (sc::HUMANOID, tags(&[HUMANOID, HOSTILE])),
        "fairy"        => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, MAGICAL, SMALL])),
        "cyborg"       => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, TECH])),
        "spacemarine"  => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, TECH, WORN])),
        "hero_warrior" => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, WORN])),
        "hero_mage"    => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, MAGICAL])),
        "hero_cleric"  => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, MAGICAL])),
        "hero_rogue"   => (sc::HUMANOID, tags(&[ALIVE, HUMANOID, SMALL])),

        // === Creatures (super 1) ===
        "animal"       => (sc::CREATURE, tags(&[ALIVE])),
        "farm_animal"  => (sc::CREATURE, tags(&[ALIVE])),
        "wild_animal"  => (sc::CREATURE, tags(&[ALIVE])),
        "bird"         => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "fish"         => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "insect"       => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "spider"       => (sc::CREATURE, tags(&[ALIVE, SMALL, HOSTILE])),
        "crab"         => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "jellyfish"    => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "bat"          => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "bear"         => (sc::CREATURE, tags(&[ALIVE, HOSTILE])),
        "cat_domestic" => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "cat_fat"      => (sc::CREATURE, tags(&[ALIVE, SMALL])),
        "cat_mech"     => (sc::CREATURE, tags(&[ALIVE, TECH, SMALL])),
        "cat_space"    => (sc::CREATURE, tags(&[ALIVE, TECH, SMALL])),
        "cat_warrior"  => (sc::CREATURE, tags(&[ALIVE, SMALL, WORN])),
        "cat_wizard"   => (sc::CREATURE, tags(&[ALIVE, SMALL, MAGICAL])),
        "dog"          => (sc::CREATURE, tags(&[ALIVE])),
        "horse"        => (sc::CREATURE, tags(&[ALIVE])),
        "wolf"         => (sc::CREATURE, tags(&[ALIVE, HOSTILE])),
        "reptile"      => (sc::CREATURE, tags(&[ALIVE])),
        "slime"        => (sc::CREATURE, tags(&[ALIVE, SMALL, HOSTILE])),

        // === Monsters (super 2) ===
        "enemy"         => (sc::MONSTER, tags(&[ALIVE, HOSTILE])),
        "demon"         => (sc::MONSTER, tags(&[ALIVE, HOSTILE, MAGICAL])),
        "dragon"        => (sc::MONSTER, tags(&[ALIVE, HOSTILE, MAGICAL])),
        "ghost"         => (sc::MONSTER, tags(&[HOSTILE, MAGICAL])),
        "golem"         => (sc::MONSTER, tags(&[HOSTILE, BUILT])),
        "mythical_beast" => (sc::MONSTER, tags(&[ALIVE, HOSTILE, MAGICAL])),
        "dinosaur"      => (sc::MONSTER, tags(&[ALIVE, HOSTILE])),
        "alien"         => (sc::MONSTER, tags(&[ALIVE, HOSTILE, TECH])),

        // === Weapons (super 3) ===
        "weapon"        => (sc::WEAPON, tags(&[HELD_ITEM])),
        "sword"         => (sc::WEAPON, tags(&[HELD_ITEM])),
        "axe_hammer"    => (sc::WEAPON, tags(&[HELD_ITEM])),
        "bow_ranged"    => (sc::WEAPON, tags(&[HELD_ITEM])),
        "gun"           => (sc::WEAPON, tags(&[HELD_ITEM, TECH])),
        "spear_polearm" => (sc::WEAPON, tags(&[HELD_ITEM])),
        "staff_wand"    => (sc::WEAPON, tags(&[HELD_ITEM, MAGICAL])),
        "lightsaber"    => (sc::WEAPON, tags(&[HELD_ITEM, TECH, MAGICAL])),

        // === Equipment (super 4) ===
        "armor"         => (sc::EQUIPMENT, tags(&[WORN])),
        "body_armor"    => (sc::EQUIPMENT, tags(&[WORN])),
        "helmet"        => (sc::EQUIPMENT, tags(&[WORN])),
        "shield"        => (sc::EQUIPMENT, tags(&[HELD_ITEM, WORN])),
        "boots_gloves"  => (sc::EQUIPMENT, tags(&[WORN, SMALL])),
        "cloak"         => (sc::EQUIPMENT, tags(&[WORN])),
        "key"           => (sc::EQUIPMENT, tags(&[HELD_ITEM, SMALL])),
        "scroll_book"   => (sc::EQUIPMENT, tags(&[HELD_ITEM, MAGICAL])),

        // === Consumables (super 5) ===
        "potion"        => (sc::CONSUMABLE, tags(&[HELD_ITEM, MAGICAL, SMALL, EDIBLE])),
        "food"          => (sc::CONSUMABLE, tags(&[SMALL, EDIBLE])),
        "crop"          => (sc::CONSUMABLE, tags(&[NATURE, SMALL, EDIBLE])),
        "mushroom"      => (sc::CONSUMABLE, tags(&[NATURE, SMALL, EDIBLE])),
        "gem_treasure"  => (sc::CONSUMABLE, tags(&[SMALL])),

        // === Terrain (super 6) ===
        "terrain"       => (sc::TERRAIN, tags(&[NATURE])),
        "terrain_scifi" => (sc::TERRAIN, tags(&[TECH])),
        "ground_natural" => (sc::TERRAIN, tags(&[NATURE])),
        "floor"         => (sc::TERRAIN, tags(&[BUILT])),
        "wall"          => (sc::TERRAIN, tags(&[BUILT])),
        "water_lava"    => (sc::TERRAIN, tags(&[NATURE])),
        "door_stair"    => (sc::TERRAIN, tags(&[BUILT])),

        // === Nature (super 7) ===
        "tree"           => (sc::NATURE, tags(&[NATURE])),
        "tree_conifer"   => (sc::NATURE, tags(&[NATURE])),
        "tree_deciduous" => (sc::NATURE, tags(&[NATURE])),
        "tree_exotic"    => (sc::NATURE, tags(&[NATURE, MAGICAL])),
        "bush_flower"    => (sc::NATURE, tags(&[NATURE, SMALL])),

        // === Structures (super 8) ===
        "building"          => (sc::STRUCTURE, tags(&[BUILT])),
        "building_scifi"    => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "furniture_decor"   => (sc::STRUCTURE, tags(&[BUILT, SMALL])),
        "furniture_light"   => (sc::STRUCTURE, tags(&[BUILT, SMALL])),
        "furniture_room"    => (sc::STRUCTURE, tags(&[BUILT])),
        "furniture_scifi"   => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "furniture_storage" => (sc::STRUCTURE, tags(&[BUILT])),
        "vehicle"           => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "vehicle_air"       => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "vehicle_land"      => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "vehicle_space"     => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "vehicle_water"     => (sc::STRUCTURE, tags(&[BUILT, TECH])),
        "mech"              => (sc::STRUCTURE, tags(&[BUILT, TECH, HOSTILE])),
        "tool"              => (sc::STRUCTURE, tags(&[HELD_ITEM, BUILT])),
        "mount"             => (sc::STRUCTURE, tags(&[ALIVE])),

        // === UI / FX (super 9) ===
        "ui"            => (sc::UI_FX, tags(&[UI])),
        "ui_frame"      => (sc::UI_FX, tags(&[UI])),
        "ui_icon"       => (sc::UI_FX, tags(&[UI, SMALL])),
        "effect"        => (sc::UI_FX, tags(&[MAGICAL])),
        "fx_ambient"    => (sc::UI_FX, tags(&[NATURE])),
        "fx_combat"     => (sc::UI_FX, tags(&[HOSTILE])),
        "fx_elemental"  => (sc::UI_FX, tags(&[MAGICAL, NATURE])),
        "fx_scifi"      => (sc::UI_FX, tags(&[TECH])),
        "misc"          => (sc::UI_FX, tags(&[])),
        "gemini"        => (sc::UI_FX, tags(&[])),

        // Unknown — default to ui_fx, no tags
        _ => (sc::UI_FX, tags(&[])),
    };
    ClassCond::new(s, t)
}

/// Super-category display name.
pub fn super_name(id: u32) -> &'static str {
    match id {
        0 => "humanoid",
        1 => "creature",
        2 => "monster",
        3 => "weapon",
        4 => "equipment",
        5 => "consumable",
        6 => "terrain",
        7 => "nature",
        8 => "structure",
        9 => "ui_fx",
        10 => "(null)",
        _ => "unknown",
    }
}

/// All known class directory names, grouped by super-category.
/// Returns (super_id, &[class_name]) for each super.
pub fn classes_for_super(super_id: u32) -> &'static [&'static str] {
    match super_id {
        0 => &[
            "character", "elf", "dwarf", "orc", "knight", "villager",
            "goblin", "skeleton", "zombie", "fairy", "cyborg", "spacemarine",
            "hero_warrior", "hero_mage", "hero_cleric", "hero_rogue",
        ],
        1 => &[
            "animal", "farm_animal", "wild_animal", "bird", "fish", "insect",
            "spider", "crab", "jellyfish", "bat", "bear", "cat_domestic",
            "cat_fat", "cat_mech", "cat_space", "cat_warrior", "cat_wizard",
            "dog", "horse", "wolf", "reptile", "slime",
        ],
        2 => &[
            "enemy", "demon", "dragon", "ghost", "golem",
            "mythical_beast", "dinosaur", "alien",
        ],
        3 => &[
            "weapon", "sword", "axe_hammer", "bow_ranged", "gun",
            "spear_polearm", "staff_wand", "lightsaber",
        ],
        4 => &[
            "armor", "body_armor", "helmet", "shield",
            "boots_gloves", "cloak", "key", "scroll_book",
        ],
        5 => &[
            "potion", "food", "crop", "mushroom", "gem_treasure",
        ],
        6 => &[
            "terrain", "terrain_scifi", "ground_natural", "floor",
            "wall", "water_lava", "door_stair",
        ],
        7 => &[
            "tree", "tree_conifer", "tree_deciduous", "tree_exotic", "bush_flower",
        ],
        8 => &[
            "building", "building_scifi", "furniture_decor", "furniture_light",
            "furniture_room", "furniture_scifi", "furniture_storage",
            "vehicle", "vehicle_air", "vehicle_land", "vehicle_space",
            "vehicle_water", "mech", "tool", "mount",
        ],
        9 => &[
            "ui", "ui_frame", "ui_icon", "effect", "fx_ambient",
            "fx_combat", "fx_elemental", "fx_scifi", "misc", "gemini",
        ],
        _ => &[],
    }
}

/// Super-category display color (RGB) for scene visualization.
pub fn super_color(id: u32) -> (u8, u8, u8) {
    match id {
        0 => (65, 105, 225),   // humanoid — royal blue
        1 => (34, 139, 34),    // creature — forest green
        2 => (220, 20, 60),    // monster — crimson
        3 => (192, 192, 192),  // weapon — silver
        4 => (184, 134, 11),   // equipment — dark gold
        5 => (255, 105, 180),  // consumable — hot pink
        6 => (139, 119, 101),  // terrain — tan
        7 => (0, 128, 0),      // nature — green
        8 => (160, 82, 45),    // structure — sienna
        9 => (128, 128, 128),  // ui_fx — gray
        _ => (0, 0, 0),        // null/unknown — black
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_cond_has_null_super() {
        let c = ClassCond::null();
        assert_eq!(c.super_id, CFG_NULL_SUPER);
        assert_eq!(c.tags, CFG_NULL_TAGS);
    }

    #[test]
    fn character_is_humanoid_super() {
        let c = lookup("character");
        assert_eq!(c.super_id, super_cat::HUMANOID);
        assert_eq!(c.tags[tag::ALIVE], 1.0);
        assert_eq!(c.tags[tag::HUMANOID], 1.0);
    }

    #[test]
    fn dragon_is_monster_with_hostile_and_magical() {
        let c = lookup("dragon");
        assert_eq!(c.super_id, super_cat::MONSTER);
        assert_eq!(c.tags[tag::ALIVE], 1.0);
        assert_eq!(c.tags[tag::HOSTILE], 1.0);
        assert_eq!(c.tags[tag::MAGICAL], 1.0);
    }

    #[test]
    fn potion_is_consumable_with_edible_and_magical() {
        let c = lookup("potion");
        assert_eq!(c.super_id, super_cat::CONSUMABLE);
        assert_eq!(c.tags[tag::EDIBLE], 1.0);
        assert_eq!(c.tags[tag::MAGICAL], 1.0);
    }

    #[test]
    fn weapon_has_held_item_tag() {
        let c = lookup("sword");
        assert_eq!(c.super_id, super_cat::WEAPON);
        assert_eq!(c.tags[tag::HELD_ITEM], 1.0);
        assert_eq!(c.tags[tag::ALIVE], 0.0);
    }

    #[test]
    fn unknown_class_returns_valid_cond() {
        let c = lookup("some_unknown_class_xyz");
        // Must not panic, super_id must be in range
        assert!(c.super_id <= CFG_NULL_SUPER);
    }

    #[test]
    fn tags_are_binary() {
        for class in &["character", "dragon", "potion", "weapon", "tree", "building"] {
            let c = lookup(class);
            for &v in &c.tags {
                assert!(v == 0.0 || v == 1.0, "tag value {v} for class {class} is not binary");
            }
        }
    }
}
