# Gemini Fill Prompts — Weak Classes

Target: bring every class to ~100+ samples via Gemini Nano Banana Pro.
Each prompt generates a 6x5 grid of 30 sprites. Use transparent or solid black background.

## Format
Paste each prompt into Gemini. Save the output image to `data/raw/gemini/` then run:
```
cargo run --release -- ingest-gemini --input data/raw/gemini/ --output data_v3_32/
```

## Prompts (sorted by urgency — lowest sample count first)

### 30 samples (need 3 batches each = 90 new)

```
Generate a 6x5 grid of 32x32 pixel art sprites of bush flowers on black background. 8-bit style, clean single-pixel edges, varied colors, game asset quality. Each sprite unique.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of space cats on black background. 8-bit style, sci-fi themed, armored cats, laser eyes, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of cat warriors on black background. 8-bit style, sword-wielding cats, armored, medieval theme, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of cyborgs on black background. 8-bit style, half-human half-machine, glowing eyes, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of door and stair tiles on black background. 8-bit style, dungeon doors, stone stairs, wooden doors, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of dwarves on black background. 8-bit style, bearded, axes, mining helmets, RPG characters, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of fish on black background. 8-bit style, tropical fish, swordfish, pufferfish, aquatic creatures, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of ambient effects on black background. 8-bit style, sparkles, dust motes, fireflies, fog wisps, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of clerics and healers on black background. 8-bit style, robes, staffs, holy symbols, RPG characters, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of mechs and robots on black background. 8-bit style, bipedal robots, giant robots, sci-fi walkers, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of mounts and rideable animals on black background. 8-bit style, horses, wolves, dragons, fantasy mounts, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of mushrooms on black background. 8-bit style, red cap, blue glow, poison, healing, varied mushroom types, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of sci-fi terrain tiles on black background. 8-bit style, metal floors, neon panels, space station tiles, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of exotic tropical trees on black background. 8-bit style, palm trees, baobab, bonsai, alien trees, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of wild animals on black background. 8-bit style, deer, foxes, rabbits, boars, forest creatures, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of zombies on black background. 8-bit style, undead, rotting, green skin, RPG enemies, game asset quality.
```

### 40 samples (need 2 batches each = 60 new)

```
Generate a 6x5 grid of 32x32 pixel art sprites of fat cats on black background. 8-bit style, round chubby cats, sitting, sleeping, cute, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of wizard cats on black background. 8-bit style, cats with wizard hats, magic staffs, spell effects, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of crops and farm plants on black background. 8-bit style, wheat, corn, tomato, carrot, pumpkin, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of dogs on black background. 8-bit style, varied breeds, sitting, running, barking, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of elves on black background. 8-bit style, pointed ears, bows, magic, forest theme, RPG characters, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of farm animals on black background. 8-bit style, cows, pigs, chickens, sheep, goats, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of combat effects on black background. 8-bit style, slashes, impacts, explosions, blood splatter, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of goblins on black background. 8-bit style, green skin, daggers, sneaky, small, RPG enemies, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of natural ground tiles on black background. 8-bit style, grass, dirt, sand, gravel, mud, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of horses on black background. 8-bit style, war horses, farm horses, unicorns, pegasus, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of reptiles on black background. 8-bit style, lizards, snakes, turtles, geckos, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of slimes on black background. 8-bit style, blue green red purple slimes, bouncy, RPG enemies, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of polearm weapons on black background. 8-bit style, spears, halberds, tridents, lances, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of conifer trees on black background. 8-bit style, pine trees, spruce, fir, snow-covered, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of stone walls on black background. 8-bit style, brick walls, castle walls, dungeon walls, cracked, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of wolves on black background. 8-bit style, gray wolves, dire wolves, ice wolves, howling, game asset quality.
```

### Empty classes (0 samples — need 5+ batches each)

```
Generate a 6x5 grid of 32x32 pixel art sprites of aliens on black background. 8-bit style, grey aliens, tentacle aliens, insectoid aliens, sci-fi, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of bats on black background. 8-bit style, cave bats, vampire bats, fire bats, flying, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of mechanical cats on black background. 8-bit style, robot cats, steampunk cats, cyborg cats, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of dinosaurs on black background. 8-bit style, t-rex, triceratops, raptor, stegosaurus, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of sci-fi furniture on black background. 8-bit style, control panels, hologram tables, stasis pods, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of golems on black background. 8-bit style, stone golems, ice golems, fire golems, crystal golems, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of insects on black background. 8-bit style, beetles, ants, butterflies, moths, ladybugs, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of mythical beasts on black background. 8-bit style, griffins, chimera, manticore, hydra, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of spiders on black background. 8-bit style, tarantulas, black widows, cave spiders, giant spiders, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of generic animals on black background. 8-bit style, raccoons, hedgehogs, squirrels, badgers, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of generic vehicles on black background. 8-bit style, carts, wagons, boats, mine carts, game asset quality.
```

```
Generate a 6x5 grid of 32x32 pixel art sprites of water and lava tiles on black background. 8-bit style, blue water, flowing lava, acid pools, swamp, game asset quality.
```
