# Accessibility Statement

**Project:** pixel-forge v0.6.0
**Standards:** Section 508 / WCAG 2.1
**Date:** 2026-03-27

## CLI Accessibility

- `--help` available on every command and subcommand (clap derive)
- All errors print to stderr with descriptive messages
- Exit code 0 on success, 1 on failure (standard Unix convention)
- Plain text output — compatible with screen readers and terminal accessibility tools
- No color-dependent information in CLI output (status conveyed through text)

## GUI Accessibility (egui)

### Color Contrast

| Element | Foreground | Background | Contrast Ratio | WCAG Level |
|---|---|---|---|---|
| Body text | #dcdceb | #0c0c12 | ~14:1 | AAA |
| Accent/links | #00d9ff (cyan) | #0c0c12 | ~12:1 | AAA |
| Button text | #dcdceb | Dark panel | ~12:1 | AAA |

High-contrast dark theme is the only theme. No light mode. Contrast ratios exceed WCAG AAA requirements (7:1) for normal text.

### Touch Targets

- Minimum touch target: 30x60 pixels (exceeds WCAG 2.5.5 minimum of 44x44 CSS pixels at standard DPI)
- DPI scaling: 2.5x on mobile (Android), system DPI on desktop
- Gallery tiles sized for finger navigation on touchscreens

### Known Limitations

| Limitation | Cause | Impact |
|---|---|---|
| No screen reader support | egui does not expose an accessibility tree | Blind users cannot navigate the GUI |
| No keyboard navigation | egui immediate-mode rendering lacks focus management | Keyboard-only users cannot navigate the GUI |
| No text scaling preference | egui does not read OS text size settings | Users with low vision must rely on OS-level zoom |
| No reduced motion | Animations are minimal but not toggleable | Motion-sensitive users may be affected |

### Mitigations

- **CLI as full alternative:** Every GUI operation has a CLI equivalent. Users who cannot access the GUI can perform all generation, training, and export tasks through the command line.
- **Plugin protocol:** The JSON stdin/stdout plugin protocol (`pixel-forge plugin`) enables integration with accessible front-ends.

## Output Accessibility

- Generated sprites are standard PNG files — compatible with any image viewer or assistive technology
- File names include class label (e.g., `character_001.png`) for identification without visual inspection
