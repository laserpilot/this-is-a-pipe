# This Is a Pipe

An SVG pipe network generator designed for pen plotting. Uses Wave Function Collapse (WFC) to generate connected pipe layouts with multiple pipe sizes, shading styles, and full occlusion handling for clean plotter output.

Forked from and inspired by the original pipe drawing code by [scumola](https://github.com/scumola).

![Example pipe network](pics/pipes-v1.svg)

## Features

- **Wave Function Collapse** layout generation with configurable tile weights
- **Three pipe sizes**: medium, narrow, and tiny — each with straights, corners, and crossovers
- **Mixed-size crossovers**: pipes of different widths crossing over/under each other (e.g. a narrow pipe passing under a medium pipe)
- **Size reducers**: transition fittings between medium/narrow and narrow/tiny pipes with flanged connections
- **Junction tiles**: crosses, tees, and 4-way crossovers
- **Directional hatching** with configurable light angle, hatch density, jitter, crosshatching, and wiggle — all scaled proportionally to pipe size
- **Multiple shading styles**: none, accent, hatch, double-wall, directional-hatch
- **Pen-plotter-ready SVG**: all strokes are clipped against neighboring pipes using Shapely polygon occlusion — no white fills, just clean strokes
- **Interactive Streamlit app** with live preview, weight sliders, and SVG download

## Running

### Interactive preview (Streamlit)

```
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser UI with sidebar controls for:
- Shading style and stroke widths
- Light direction and hatch parameters (spacing, angle, jitter, crosshatch, wiggle)
- Tile weights by pipe size (medium/narrow/tiny) and shape (straights/corners/junctions/reducers)
- Grid dimensions and zoom level

### Command-line generation

```
pip install -r requirements.txt
python pipes-v1.py
```

This generates an SVG file directly. Edit the script to change grid size and parameters.

## How it works

1. **Wave Function Collapse** fills a grid by choosing tiles that satisfy port-compatibility constraints. Each tile has directional ports with a size tag (`m`, `n`, or `t`), and neighbors must match port sizes.

2. **Tile weights** control the probability of each tile being chosen during collapse. Weight = `size_multiplier x shape_multiplier`, so you can independently dial up narrow pipes or corners.

3. **Rendering** iterates over the grid and draws each tile's outline and shading. For each tile, a Shapely polygon union of all *other* pipes is built as an occlusion mask, and every stroke is clipped against it. Crossover tiles draw the horizontal pipe first (clipped against the vertical pipe's polygon), then the vertical pipe on top.

4. **Directional hatching** places hatch marks on the shadow side of each pipe based on a global light angle. Hatch parameters (band width, spacing, jitter) scale proportionally to pipe size (`half_width / 30`), so narrow and tiny pipes get appropriately sized shading.

## Tile reference

| Category | Tiles |
|----------|-------|
| Medium straights | `\|` `-` |
| Narrow straights | `i` `=` |
| Tiny straights | `!` `.` |
| Medium corners | `r` `7` `j` `L` |
| Narrow corners | `nr` `n7` `nj` `nL` |
| Tiny corners | `tr` `t7` `tj` `tL` |
| Junctions | `X` (cross) `T` `B` `E` `W` (tees) |
| Crossovers | `+` `+n` `+t` `+mn` `+nm` `+mt` `+tm` `+nt` `+tn` |
| Reducers (M-N) | `Rv` `RV` `Rh` `RH` |
| Reducers (N-T) | `Tv` `TV` `Th` `TH` |

## Dependencies

- [drawsvg](https://github.com/cduck/drawsvg) — SVG generation
- [Shapely](https://shapely.readthedocs.io/) — polygon clipping for occlusion
- [Streamlit](https://streamlit.io/) — interactive preview app
