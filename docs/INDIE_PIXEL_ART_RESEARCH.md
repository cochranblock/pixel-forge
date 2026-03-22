# Indie Pixel Art Research — Processes, Palettes, Canvas Sizes

## Canvas Size Reference

| Game | Tile Size | Character Size | Palette |
|------|-----------|----------------|---------|
| Stardew Valley | 16x16 | ~16x32 | 4+ shades, evolved organically |
| Undertale | varies | 30-49px tall | Monochrome + selective color |
| Celeste | 8x8 grid | ~16x16 | Mood-driven per chapter |
| Hyper Light Drifter | varies | 32x32 | Bold flat + gradient overlays |
| Shovel Knight | 8x8/16x16 | 4-5 colors max | NES 54-color + 2 custom browns |
| Owlboy | varies | 39px tall | Modern color depth, "Hi-Bit" |
| CrossCode | RPGMaker-size | 16-bit scale | SNES-era depth with outlines |
| Terraria | 16x16 | varies | Evolved over 10+ years |
| Dead Cells | varies | 50px tall | 3D-rendered then pixelated |
| Eastward | varies | varies | Retro palette + 3D deferred lighting |
| Sea of Stars | varies | varies | High-res pixel + dynamic lighting |
| Blasphemous | varies | large | Dark golds, reds, purples (Spanish religious art) |

## Key Takeaways for Training

1. **16x16 is just tiles** — characters are usually taller (16x32, 32x32, up to 50px)
2. **4-shade palettes** are the sweet spot (Starbound, Shovel Knight)
3. **Consistent light direction** — Starbound mandates top-right sun
4. **Outlines vs no outlines** — style choice, both valid
5. **Hand-drawn quality = iteration** — ConcernedApe spent 4.5 years, Owlboy took 10 years
6. **Dead Cells cheats** — 3D models rendered to 2D pixel art. We could do similar with AI.

## Tools They Use

| Tool | Used By |
|------|---------|
| Paint.NET | ConcernedApe |
| MS Paint | Toby Fox |
| Aseprite | Chucklefish, Eastward |
| Photoshop CS3 | Owlboy |
| 3DS Max → pixel render | Dead Cells |
| GameMaker | Undertale, Katana ZERO |

## Best Public Resources

- [Saint11 (Pedro Medeiros) — 70+ pixel art tutorials](https://saint11.art/blog/pixel-art-tutorials/)
- [Starbound Art Guide — palette + lighting rules](https://starbounder.org/Guide:Art)
- [Shovel Knight Character Creation — concept to sprite](https://old.yachtclubgames.com/2020/03/creating-a-shovel-knight-character-sprite/)
- [Breaking the NES for Shovel Knight — palette decisions](https://www.yachtclubgames.com/blog/breaking-the-nes/)
- [Celeste Pixel Art — EXOK official blog](https://exok.com/posts/2019-12-10-celeste-pixel-art/)
- [Celeste Tilesets Step-by-Step](https://aran.ink/posts/celeste-tilesets)
- [Dead Cells 3D-to-2D Pipeline](https://www.gamedeveloper.com/production/art-design-deep-dive-using-a-3d-pipeline-for-2d-animation-in-i-dead-cells-i-)
- [Derek Yu (Spelunky) Pixel Art Tutorial](https://www.derekyu.com/makegames/pixelart.html)
- [NES Style Guide](https://eirifu.wordpress.com/2025/02/21/a-style-guide-for-nes-inspired-pixel-art-in-your-retro-game/)

## Contact Info (non-Twitter)

| Studio | Contact |
|--------|---------|
| ConcernedApe | contact@stardewvalley.net, Discord |
| Chucklefish | press@chucklefish.org |
| Heart Machine | heartmachine.com, Discord |
| D-Pad Studio (Owlboy) | contact@dpadstudio.com |
| Yacht Club (Shovel Knight) | yachtclubgames.com/support |
| The Game Kitchen (Blasphemous) | info@thegamekitchen.com |
| Sabotage (Sea of Stars) | info@sabotagestudio.com |
| Radical Fish (CrossCode) | contest@radicalfishgames.com |
| Askiisoft (Katana ZERO) | contact@askiisoft.com |
| Re-Logic (Terraria) | media@re-logic.com |
| Pedro Medeiros (saint11) | saint11.art, patreon.com/saint11 |
| Simon Andersen (Owlboy) | linktr.ee/snakepixel |

## What This Means for Pixel Forge

Our current training at 16x16 matches tile assets (terrain, items, UI).
For character quality, we need 32x32 minimum, 48x48 ideal.
The Starbound 4-shade palette constraint could be a training filter.
Saint11's 70 tutorials could inform our data curation criteria.
