# Training Data Sources — Pixel Forge v3 (32×32)

## Primary Sources

| Source | Count (approx) | License | Notes |
|--------|---------------|---------|-------|
| Dungeon Crawl Stone Soup | ~6,000 | CC0 | DCSS art team |
| DawnLike v1.81 | ~5,000 | CC-BY 4.0 | DragonDePlatino + DawnBringer |
| Kenney (3 packs) | ~3,878 | CC0 | Roguelike, Platformer, 1-Bit |
| Hyptosis Tiles | ~1,000 | CC-BY 3.0 | Hyptosis |
| David E. Gervais Tiles | ~1,280 | CC-BY 3.0 | David E. Gervais |
| Gemini-generated | ~14,000 | AI-generated | Text prompts, sliced via slice_gemini |

## Gemini Generation Sessions

### Session: 2026-04-19 — Gavin Van der Merwe reference
- **Context input:** AI/ML math operations cheat sheet by Gavin Van der Merwe (LinkedIn)
- **Method:** Screenshots of Gavin's post used as visual context for Gemini Pro 5×6 sprite sheet generation prompts
- **Attribution:** Gavin Van der Merwe credited in CONTRIBUTORS.md
- **Classes generated:** TBD (pending generation)
- **Slice tool:** `slice_gemini`

## Notes

- 70% of balanced training set is AI-generated via Gemini (fills class gaps <50 samples)
- All AI-generated sprites are tagged with `gemini_` prefix in filename
- Human-created sprites retain original filenames from source packs
