# Pixel Forge Training Data — Sources & Attribution

Training data comes from two categories: artist-made CC0/CC-BY sprites and AI-generated sprites.
No copyrighted game rips. All artist-made sources are free for commercial use.

## Source 1: Dungeon Crawl Stone Soup Tiles (~6,000 sprites, 32×32)

- **License:** CC0 (Creative Commons Zero — public domain)
- **Artists:** Dungeon Crawl Stone Soup art team
- **URL:** https://opengameart.org/content/dungeon-crawl-32x32-tiles
- **Supplemental:** https://opengameart.org/content/dungeon-crawl-32x32-tiles-supplemental
- **Categories:** Monsters, terrain, walls, items, spells, GUI, player avatars
- **Attribution:** Not required (CC0), but credited here out of respect.
  "Every tile can be used freely, even in a commercial project, even WITHOUT proper attribution."

## Source 2: DawnLike v1.81 (~5,000 sprites, 16×16)

- **License:** CC-BY 4.0 (Creative Commons Attribution 4.0)
- **Artist:** DragonDePlatino
- **Palette:** DawnBringer
- **URL:** https://opengameart.org/content/dawnlike-16x16-universal-rogue-like-tileset-v181
- **GitHub:** https://github.com/hadean-mirrors/dawnlike
- **Categories:** Characters, monsters, weapons, items, dungeon/outdoor terrain, 2-frame animations
- **Attribution Required:** Credit DragonDePlatino (art) and DawnBringer (palette).

## Source 3: Kenney Roguelike/RPG Pack (~1,700 sprites, 16×16)

- **License:** CC0 (public domain)
- **Artist:** Kenney (kenney.nl)
- **URL:** https://kenney.nl/assets/roguelike-rpg-pack
- **Categories:** Floors, walls, doors, furniture, flora, characters
- **Attribution:** Not required (CC0). Kenney is the gold standard for free game assets.

## Source 4: Kenney Pixel Platformer + Redux (~1,100 sprites)

- **License:** CC0
- **Artist:** Kenney
- **URLs:**
  - https://kenney.nl/assets/pixel-platformer
  - https://kenney.nl/assets/platformer-art-pixel-redux
- **Categories:** Characters, terrain, items, enemies

## Source 5: Kenney 1-Bit Pack (~1,000 sprites, 16×16)

- **License:** CC0
- **Artist:** Kenney
- **URL:** https://kenney-assets.itch.io/1-bit-pack
- **Categories:** 1-bit sprites — characters, items, tools, buildings, UI

## Source 6: Hyptosis Tiles & Sprites (~1,000 sprites, 32×32)

- **License:** CC-BY 3.0
- **Artist:** Hyptosis
- **URL:** https://opengameart.org/content/lots-of-free-2d-tiles-and-sprites-by-hyptosis
- **Organized:** https://opengameart.org/content/lots-of-hyptosis-tiles-organized
- **Categories:** Terrain, monsters, houses, castles, caves, plants
- **Attribution Required:** Credit Hyptosis. ("All I want is credit. Anyone can use these for anything.")
- **Downloads:** 293,000+ total across batches.

## Source 7: David E. Gervais Roguelike Tiles (~1,280 sprites, 32×32)

- **License:** CC-BY 3.0
- **Artist:** David E. Gervais
- **URL:** https://opengameart.org/content/roguelike-tiles-large-collection
- **Categories:** 930+ creatures, 350+ items, weapons, armor, town/dungeon tiles
- **Attribution Required:** Credit David E. Gervais.

## Source 8: Gemini-Generated Pixel Art (~14,000 sprites, 32×32)

- **License:** AI-generated (not copyrighted in most jurisdictions)
- **Generator:** Google Gemini (Nano Banana Pro)
- **Method:** Text prompts → 6x5 grid sheets → sliced to 32×32 → background removed → quality verified
- **Ingestion:** `pixel-forge ingest-gemini` command ([src/main.rs](../src/main.rs))
- **Categories:** Fills class gaps where artist-made sprites had <50 samples (e.g., lightsaber, cyborg, mech, fairy)
- **Disclosure:** These are AI-generated sprites used as training data for another AI model. ~70% of the balanced `data_v3_32/` dataset is Gemini-generated. The remaining ~30% is artist-made from Sources 1-7.

---

## License Summary

| Source | License | Attribution Required |
|--------|---------|---------------------|
| Dungeon Crawl | CC0 | No |
| DawnLike | CC-BY 4.0 | Yes — DragonDePlatino + DawnBringer |
| Kenney (all) | CC0 | No |
| Hyptosis | CC-BY 3.0 | Yes — Hyptosis |
| Gervais | CC-BY 3.0 | Yes — David E. Gervais |
| **Gemini** | **AI-generated** | **N/A — disclose as AI-generated** |

## Training Use

These sprites train three diffusion models ([Cinder](../src/tiny_unet.rs), [Quench](../src/medium_unet.rs), [Anvil](../src/anvil_unet.rs)) for pixel art generation. The models learn pixel art patterns, not specific sprites — they generate new, original pixel art.

Artist-made sources (1-7) explicitly permit commercial use and derivative works.
AI-generated sprites (Source 8) fill class gaps and are disclosed as such.

Dataset preprocessing: [train::preprocess](../src/train.rs#L221). Balanced set: 19,876 tiles in `data_v3_32/` (capped 2K/class, 68 active class dirs).
