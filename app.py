import streamlit as st
import streamlit.components.v1 as components
import pipe_core
import re
from datetime import datetime

st.set_page_config(page_title="Pipe Shading Preview", layout="wide")
st.title("Pipe Shading Preview")

view_mode = st.sidebar.radio("View Mode", ["Pipe Network", "Tile Catalog"])

with st.sidebar:
    st.header("Shading Settings")

    shading_style = st.selectbox(
        "Shading Style",
        ['none', 'accent', 'hatch', 'double-wall', 'directional-hatch'],
        index=4  # default to directional-hatch
    )

    stroke_width = st.slider("Stroke Width", 0.2, 2.0, 0.5, 0.1)

    auto_shading_width = st.checkbox("Auto shading width (60% of stroke)", value=True)
    if auto_shading_width:
        shading_stroke_width = stroke_width * 0.6
        st.caption("Shading stroke: {:.2f}".format(shading_stroke_width))
    else:
        shading_stroke_width = st.slider("Shading Stroke Width", 0.1, 2.0, stroke_width * 0.6, 0.05)

    # Directional shading controls
    if shading_style == 'directional-hatch':
        st.header("Light & Hatch Settings")
        light_angle = st.slider(
            "Light Angle",
            0, 360, 225, 5,
            help="Direction light comes FROM. 0=right, 90=down, 180=left, 270=up. Default 225=top-left."
        )

        crosshatch = st.checkbox("Enable Crosshatching", value=False,
                                  help="Add a second set of hatch lines perpendicular to the first")

        with st.expander("Advanced Hatch Parameters"):
            band_width = st.slider("Band Width", 1.0, 20.0, 12.0, 0.5,
                                   help="Width of the shaded band (out of 30 half-pipe-width)")
            band_offset = st.slider("Band Offset", 0.0, 10.0, 3.0, 0.5,
                                    help="Gap from pipe wall to shading band")
            hatch_spacing = st.slider("Hatch Spacing", 0.1, 20.0, 10.0, 0.1,
                                      help="Distance between hatch lines (lower = denser)")
            hatch_angle = st.slider("Hatch Angle", 0, 90, 30, 1,
                                    help="Angle of hatch lines from pipe direction")
            jitter_pos = st.slider("Position Jitter", 0.0, 5.0, 1.5, 0.1,
                                   help="Random offset in hatch position")
            jitter_angle = st.slider("Angle Jitter", 0.0, 10.0, 3.0, 0.1,
                                     help="Random rotation of hatch lines")
            band_width_jitter = st.slider("Length Jitter", 0.0, 0.5, 0.0, 0.01,
                                          help="Random variation in hatch length (0-0.5)")
            wiggle = st.slider("Wiggle", 0.0, 5.0, 0.0, 0.1,
                               help="Curved/wavy hatches (0 = straight lines)")
            if crosshatch:
                crosshatch_angle = st.slider("Crosshatch Angle Offset", 45, 135, 90, 5,
                                             help="Angle offset for second set of hatches")
            else:
                crosshatch_angle = 90

        shading_params = {
            'band_width': band_width,
            'band_offset': band_offset,
            'spacing': hatch_spacing,
            'angle': hatch_angle,
            'jitter_pos': jitter_pos,
            'jitter_angle': jitter_angle,
            'crosshatch': crosshatch,
            'crosshatch_angle': crosshatch_angle,
            'band_width_jitter': band_width_jitter,
            'wiggle': wiggle,
        }
    else:
        light_angle = 225
        shading_params = None

    st.header("Tile Weights")
    with st.expander("Pipe Size Weights"):
        w_medium = st.slider("Medium", 0.1, 5.0, 1.0, 0.1)
        w_narrow = st.slider("Narrow", 0.1, 5.0, 1.0, 0.1)
        w_tiny = st.slider("Tiny", 0.1, 5.0, 1.0, 0.1)
    with st.expander("Shape Weights"):
        w_straight = st.slider("Straights", 0.1, 5.0, 1.0, 0.1)
        w_corner = st.slider("Corners", 0.1, 5.0, 3.0, 0.1)
        w_chamfer = st.slider("Chamfered Corners", 0.0, 5.0, 0.0, 0.1)
        w_teardrop = st.slider("Teardrop Corners", 0.0, 5.0, 0.0, 0.1)
        w_dodge = st.slider("Dodge/Zigzag", 0.0, 5.0, 0.0, 0.1)
        w_diagonal = st.slider("Diagonal Pipes", 0.0, 5.0, 0.0, 0.1,
                               help="Diagonal pipes (narrow/tiny only). Set > 0 to enable.")
        w_diag_endcap = st.slider("Diagonal Endcaps", 0.0, 5.0, 0.0, 0.1,
                                   help="Dead-end caps for diagonal pipes. 0 = diagonals must connect.")
        w_junction = st.slider("Junctions", 0.1, 5.0, 2.0, 0.1)
        w_reducer = st.slider("Reducers", 0.1, 5.0, 1.0, 0.1)
        w_endcap = st.slider("Cardinal Endcaps", 0.0, 5.0, 0.0, 0.1,
                              help="Dead-end caps for cardinal pipes. Set > 0 to enable.")
        w_mixed_corner = st.slider("Mixed-Size Corners", 0.0, 5.0, 0.0, 0.1,
                                    help="Corners connecting medium and narrow pipes. Set > 0 to enable.")
        w_sbend = st.slider("S-Bends", 0.0, 5.0, 0.0, 0.1,
                             help="Smooth S-curves with lateral shift. Same ports as straights.")
        w_segmented = st.slider("Segmented Elbows", 0.0, 5.0, 0.0, 0.1,
                                 help="Faceted 90° corners (3 straight segments). Industrial look.")
        w_long_radius = st.slider("Long-Radius Elbows", 0.0, 5.0, 0.0, 0.1,
                                   help="Wide sweeping 90° corners with gentle arcs.")
        w_vanishing = st.slider("Vanishing Pipes", 0.0, 5.0, 0.0, 0.1,
                                 help="Pipes that taper to a point. Surreal dead-ends.")

    tile_weights = {
        'size': {'medium': w_medium, 'narrow': w_narrow, 'tiny': w_tiny},
        'shape': {'straight': w_straight, 'corner': w_corner,
                  'chamfer': w_chamfer, 'teardrop': w_teardrop,
                  'dodge': w_dodge, 'diagonal': w_diagonal,
                  'diagonal_endcap': w_diag_endcap,
                  'junction': w_junction, 'reducer': w_reducer,
                  'endcap': w_endcap, 'mixed_corner': w_mixed_corner,
                  'sbend': w_sbend, 'segmented': w_segmented,
                  'long_radius': w_long_radius, 'vanishing': w_vanishing},
    }

    st.header("Spatial Size Gradient")
    SPATIAL_PRESETS = {
        'None (disabled)': None,
        'Thick \u2192 Thin (L\u2192R)': {
            'type': 'horizontal',
            'medium': {'start': 2.0, 'end': 0.2},
            'narrow': {'start': 1.0, 'end': 1.0},
            'tiny':   {'start': 0.2, 'end': 2.0},
        },
        'Thin \u2192 Thick (L\u2192R)': {
            'type': 'horizontal',
            'medium': {'start': 0.2, 'end': 2.0},
            'narrow': {'start': 1.0, 'end': 1.0},
            'tiny':   {'start': 2.0, 'end': 0.2},
        },
        'Thick \u2192 Thin (T\u2192B)': {
            'type': 'vertical',
            'medium': {'start': 2.0, 'end': 0.2},
            'narrow': {'start': 1.0, 'end': 1.0},
            'tiny':   {'start': 0.2, 'end': 2.0},
        },
        'Radial (thick center)': {
            'type': 'radial',
            'medium': {'start': 2.0, 'end': 0.2},
            'narrow': {'start': 1.0, 'end': 1.0},
            'tiny':   {'start': 0.2, 'end': 2.0},
        },
        'Radial (thin center)': {
            'type': 'radial',
            'medium': {'start': 0.2, 'end': 2.0},
            'narrow': {'start': 1.0, 'end': 1.0},
            'tiny':   {'start': 2.0, 'end': 0.2},
        },
        'Custom': 'custom',
    }

    spatial_preset = st.selectbox(
        "Gradient Preset",
        list(SPATIAL_PRESETS.keys()),
        index=0,
        help="Bias pipe size selection by position across the grid"
    )

    spatial_map = None
    if spatial_preset != 'None (disabled)':
        preset_val = SPATIAL_PRESETS[spatial_preset]

        if preset_val == 'custom':
            gradient_type = st.selectbox("Gradient Type",
                                          ['horizontal', 'vertical', 'radial'])
        else:
            gradient_type = preset_val['type']

        if gradient_type == 'radial':
            start_label, end_label = "Center", "Edge"
        elif gradient_type == 'vertical':
            start_label, end_label = "Top", "Bottom"
        else:
            start_label, end_label = "Left", "Right"

        if preset_val not in (None, 'custom'):
            defaults = preset_val
        else:
            defaults = {
                'medium': {'start': 1.0, 'end': 1.0},
                'narrow': {'start': 1.0, 'end': 1.0},
                'tiny':   {'start': 1.0, 'end': 1.0},
            }

        with st.expander("Per-Size Endpoint Weights",
                          expanded=(preset_val == 'custom')):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(start_label)
                sm_start = st.slider("Medium ({})".format(start_label),
                                      0.1, 3.0, defaults['medium']['start'], 0.1,
                                      key='sp_m_start')
                sn_start = st.slider("Narrow ({})".format(start_label),
                                      0.1, 3.0, defaults['narrow']['start'], 0.1,
                                      key='sp_n_start')
                st_start = st.slider("Tiny ({})".format(start_label),
                                      0.1, 3.0, defaults['tiny']['start'], 0.1,
                                      key='sp_t_start')
            with col2:
                st.caption(end_label)
                sm_end = st.slider("Medium ({})".format(end_label),
                                    0.1, 3.0, defaults['medium']['end'], 0.1,
                                    key='sp_m_end')
                sn_end = st.slider("Narrow ({})".format(end_label),
                                    0.1, 3.0, defaults['narrow']['end'], 0.1,
                                    key='sp_n_end')
                st_end = st.slider("Tiny ({})".format(end_label),
                                    0.1, 3.0, defaults['tiny']['end'], 0.1,
                                    key='sp_t_end')

        spatial_map = {
            'enabled': True,
            'type': gradient_type,
            'medium': {'start': sm_start, 'end': sm_end},
            'narrow': {'start': sn_start, 'end': sn_end},
            'tiny':   {'start': st_start, 'end': st_end},
            'min_weight': 0.05,
        }

    st.header("Grid Settings")
    grid_width = st.slider("Grid Width", 4, 80, 8)
    grid_height = st.slider("Grid Height", 4, 80, 8)

    if grid_width > 16:
        with st.expander("Advanced Generation"):
            st.caption("Large grids use striped generation (width > 16)")
            stripe_width = st.slider("Stripe Width", 4, 16, 8,
                                     help="Base width of each stripe (randomized ±3)")
    else:
        stripe_width = 8

    st.header("Masking")
    mask_enabled = st.checkbox("Enable Masking", value=False,
                                help="Block regions of the grid before generation")

    mask_shapes = []
    mask_invert = False

    if mask_enabled:
        mask_invert = st.checkbox(
            "Invert Mask", value=False,
            help="OFF: shapes define blocked (void) regions. "
                 "ON: shapes define allowed regions — everything else is void.")

        if 'mask_shapes' not in st.session_state:
            st.session_state.mask_shapes = []

        st.caption("Coordinates are in grid cells (0,0 = top-left). "
                   "A {}x{} grid has cells 0\u2013{} across and 0\u2013{} down.".format(
                       grid_width, grid_height, grid_width - 1, grid_height - 1))

        with st.expander("Add Mask Shape", expanded=True):
            shape_type = st.selectbox("Shape Type",
                                       ["rectangle", "circle", "ring"])

            if shape_type == "rectangle":
                col1, col2 = st.columns(2)
                with col1:
                    rect_x0 = st.number_input("X Start", 0, grid_width - 1, 1)
                    rect_y0 = st.number_input("Y Start", 0, grid_height - 1, 1)
                with col2:
                    rect_x1 = st.number_input("X End", 1, grid_width,
                                               min(grid_width, rect_x0 + 3))
                    rect_y1 = st.number_input("Y End", 1, grid_height,
                                               min(grid_height, rect_y0 + 3))
                new_shape = {'type': 'rectangle',
                             'x0': rect_x0, 'y0': rect_y0,
                             'x1': rect_x1, 'y1': rect_y1}

            elif shape_type == "circle":
                cx = st.slider("Center X", 0.0, float(grid_width - 1),
                                float(grid_width) / 2, 0.5)
                cy = st.slider("Center Y", 0.0, float(grid_height - 1),
                                float(grid_height) / 2, 0.5)
                max_r = max(grid_width, grid_height) / 2.0
                r = st.slider("Radius", 0.5, max_r, min(3.0, max_r), 0.5)
                new_shape = {'type': 'circle', 'cx': cx, 'cy': cy, 'r': r}

            elif shape_type == "ring":
                cx = st.slider("Ring Center X", 0.0, float(grid_width - 1),
                                float(grid_width) / 2, 0.5)
                cy = st.slider("Ring Center Y", 0.0, float(grid_height - 1),
                                float(grid_height) / 2, 0.5)
                max_r = max(grid_width, grid_height) / 2.0
                r_inner = st.slider("Inner Radius", 0.5, max_r,
                                     min(1.5, max_r), 0.5)
                r_outer = st.slider("Outer Radius", r_inner + 0.5, max_r + 2,
                                     min(r_inner + 2.0, max_r + 2), 0.5)
                new_shape = {'type': 'ring', 'cx': cx, 'cy': cy,
                             'r_inner': r_inner, 'r_outer': r_outer}

            if st.button("Add Shape"):
                st.session_state.mask_shapes.append(new_shape)
                st.rerun()

        if st.session_state.mask_shapes:
            st.caption("Active Shapes:")
            for idx, shape in enumerate(st.session_state.mask_shapes):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if shape['type'] == 'rectangle':
                        st.text("Rect ({},{}) to ({},{})".format(
                            shape['x0'], shape['y0'],
                            shape['x1'], shape['y1']))
                    elif shape['type'] == 'circle':
                        st.text("Circle c=({},{}) r={}".format(
                            shape['cx'], shape['cy'], shape['r']))
                    elif shape['type'] == 'ring':
                        st.text("Ring c=({},{}) r={}-{}".format(
                            shape['cx'], shape['cy'],
                            shape['r_inner'], shape['r_outer']))
                with col2:
                    if st.button("X", key="rm_shape_{}".format(idx)):
                        st.session_state.mask_shapes.pop(idx)
                        st.rerun()

            if st.button("Clear All Shapes"):
                st.session_state.mask_shapes = []
                st.rerun()

        if st.session_state.mask_shapes:
            mask_preview = pipe_core.build_mask(
                grid_width, grid_height,
                st.session_state.mask_shapes,
                invert=mask_invert)
            allowed, blocked = pipe_core.count_masked_cells(mask_preview)
            total = allowed + blocked
            st.caption("{} allowed, {} blocked ({:.0f}% void)".format(
                allowed, blocked,
                100 * blocked / total if total > 0 else 0))

        mask_shapes = st.session_state.mask_shapes

    st.header("Multi-Layer")
    multilayer_enabled = st.checkbox("Enable Multi-Layer", value=False,
                                      help="Overlay two independent pipe networks")
    if multilayer_enabled:
        layer_0_color = st.text_input("Bottom Layer Color", value="red",
                                       help="data-color attribute for pen plotter")
        layer_1_color = st.text_input("Top Layer Color", value="black",
                                       help="data-color attribute for pen plotter")
        scaled_layers_enabled = st.checkbox("Enable Per-Layer Scaling", value=False,
                                             help="Scale layers independently for depth effect")
        if scaled_layers_enabled:
            layer_0_scale = st.slider("Bottom Layer Scale", 0.5, 2.0, 1.0, 0.05)
            layer_1_scale = st.slider("Top Layer Scale", 0.5, 2.0, 1.2, 0.05)
            with st.expander("Layer Offsets"):
                layer_0_offset_x = st.slider("Bottom X Offset", -200, 200, 0, 10)
                layer_0_offset_y = st.slider("Bottom Y Offset", -200, 200, 0, 10)
                layer_1_offset_x = st.slider("Top X Offset", -200, 200, 0, 10)
                layer_1_offset_y = st.slider("Top Y Offset", -200, 200, 0, 10)
        else:
            layer_0_scale = 1.0
            layer_1_scale = 1.0
            layer_0_offset_x = layer_0_offset_y = 0
            layer_1_offset_x = layer_1_offset_y = 0
    else:
        layer_0_color = "red"
        layer_1_color = "black"
        scaled_layers_enabled = False
        layer_0_scale = 1.0
        layer_1_scale = 1.0
        layer_0_offset_x = layer_0_offset_y = 0
        layer_1_offset_x = layer_1_offset_y = 0

    st.header("Decorations")
    decorations_enabled = st.checkbox("Enable Pipe Decorations", value=False,
                                       help="Add plotter-friendly decorations (couplings, dials, valves, bolts)")
    if decorations_enabled:
        decoration_density = st.slider("Decoration Density", 0.0, 0.50, 0.15, 0.01,
                                        help="Fraction of eligible tiles decorated")
        decoration_scale = st.slider("Decoration Scale", 0.3, 3.0, 1.0, 0.1,
                                      help="Size multiplier for decorations")
        auto_deco_stroke = st.checkbox("Match outline stroke", value=True)
        if not auto_deco_stroke:
            decoration_stroke_width = st.slider("Decoration Stroke Width",
                                                 0.1, 2.0, stroke_width, 0.05)
        else:
            decoration_stroke_width = None
    else:
        decoration_density = 0.15
        decoration_scale = 1.0
        decoration_stroke_width = None

    st.header("View Settings")
    zoom_level = st.slider("Zoom", 25, 200, 100, 5, help="Zoom level (100% = fit to window)")

    if st.button("Regenerate Layout", type="primary"):
        st.session_state.pop('grid', None)
        st.session_state.pop('grids', None)
        st.session_state.pop('grid_dims', None)
        st.session_state.pop('svg_string', None)
        st.session_state.pop('render_key', None)

if view_mode == "Pipe Network":
    # Build mask if enabled
    wfc_mask = None
    if mask_enabled and mask_shapes:
        wfc_mask = pipe_core.build_mask(grid_width, grid_height,
                                         mask_shapes, invert=mask_invert)

    # Generate grid(s) if needed — include mask state in cache key
    mask_key = None
    if mask_enabled and mask_shapes:
        mask_key = (tuple(tuple(sorted(s.items())) for s in mask_shapes),
                    mask_invert)
    spatial_key = str(spatial_map) if spatial_map else None
    current_dims = (grid_width, grid_height, multilayer_enabled,
                    mask_enabled, mask_key, spatial_key)

    def _run_wfc(progress_label="", progress_scale=1.0, progress_offset=0.0, stripe_offset=0):
        """Run WFC with progress callback."""
        def wfc_update(attempt, max_attempts, cells, total, backtracks=0):
            pct = progress_offset + min(cells / total, 1.0) * progress_scale
            bt_text = " ({} backtracks)".format(backtracks) if backtracks else ""
            wfc_progress.progress(min(pct, 1.0),
                                  text="{} Attempt {} — {}/{} cells{}".format(
                                      progress_label, attempt, cells, total, bt_text))

        if grid_width > 16:
            return pipe_core.wave_function_collapse_striped(
                grid_width, grid_height, stripe_width=stripe_width,
                tile_weights=tile_weights, progress_callback=wfc_update,
                stripe_offset=stripe_offset, mask=wfc_mask,
                spatial_map=spatial_map)
        else:
            return pipe_core.wave_function_collapse(
                grid_width, grid_height, tile_weights=tile_weights,
                progress_callback=wfc_update, mask=wfc_mask,
                spatial_map=spatial_map)

    if 'grid' not in st.session_state or st.session_state.get('grid_dims') != current_dims:
        # New grid means cached SVG is stale
        st.session_state.pop('svg_string', None)
        st.session_state.pop('render_key', None)
        st.session_state.pop('grids', None)
        wfc_progress = st.progress(0, text="Generating pipe layout...")

        if multilayer_enabled:
            grid_0 = _run_wfc("Layer 1/2 —", progress_scale=0.5, progress_offset=0.0)
            grid_1 = _run_wfc("Layer 2/2 —", progress_scale=0.5, progress_offset=0.5,
                               stripe_offset=stripe_width // 2)
            wfc_progress.empty()
            if grid_0 and grid_1:
                st.session_state.grids = [grid_0, grid_1]
                st.session_state.grid = grid_0
            else:
                st.session_state.grids = None
                st.session_state.grid = None
        else:
            grid = _run_wfc(progress_scale=1.0)
            wfc_progress.empty()
            st.session_state.grid = grid
            st.session_state.grids = None
        st.session_state.grid_dims = current_dims

    # Determine if we have valid data to render
    has_grid = False
    if multilayer_enabled:
        grids = st.session_state.get('grids')
        has_grid = grids is not None and len(grids) == 2
    else:
        grid = st.session_state.get('grid')
        has_grid = grid is not None

    if has_grid:
        # Cache rendered SVG to avoid re-rendering when only view settings change
        render_key = (stroke_width, shading_style, shading_stroke_width,
                      light_angle, str(shading_params),
                      decorations_enabled, decoration_density, decoration_stroke_width,
                      decoration_scale,
                      multilayer_enabled, layer_0_color, layer_1_color,
                      scaled_layers_enabled, layer_0_scale, layer_1_scale,
                      layer_0_offset_x, layer_0_offset_y,
                      layer_1_offset_x, layer_1_offset_y)

        if st.session_state.get('render_key') != render_key or 'svg_string' not in st.session_state:
            progress_bar = st.progress(0, text="Rendering tiles...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text="Rendering tile {} / {}".format(current, total))

            if multilayer_enabled and scaled_layers_enabled:
                layer_specs = []
                for idx, (g, color, scale, offset) in enumerate([
                    (st.session_state.grids[0], layer_0_color, layer_0_scale,
                     (layer_0_offset_x, layer_0_offset_y)),
                    (st.session_state.grids[1], layer_1_color, layer_1_scale,
                     (layer_1_offset_x, layer_1_offset_y)),
                ]):
                    layer_specs.append(pipe_core.make_layer_spec(
                        grid=g, scale=scale, offset=offset,
                        stroke_width=stroke_width,
                        shading_style=shading_style,
                        shading_stroke_width=shading_stroke_width,
                        shading_params=shading_params,
                        decorations_enabled=decorations_enabled,
                        decoration_density=decoration_density,
                        decoration_stroke_width=decoration_stroke_width,
                        decoration_scale=decoration_scale,
                        name='layer-{}'.format(idx),
                        color=color,
                    ))
                svg_string = pipe_core.render_scaled_multilayer_svg(
                    layer_specs,
                    light_angle_deg=light_angle,
                    progress_callback=update_progress,
                )
            elif multilayer_enabled:
                svg_string = pipe_core.render_multilayer_svg(
                    st.session_state.grids,
                    stroke_width, shading_style, shading_stroke_width,
                    light_angle_deg=light_angle, shading_params=shading_params,
                    progress_callback=update_progress,
                    decorations_enabled=decorations_enabled,
                    decoration_density=decoration_density,
                    decoration_stroke_width=decoration_stroke_width,
                    decoration_scale=decoration_scale,
                    layer_colors=[layer_0_color, layer_1_color],
                )
            else:
                svg_string = pipe_core.render_svg(
                    st.session_state.grid,
                    stroke_width, shading_style, shading_stroke_width,
                    light_angle_deg=light_angle, shading_params=shading_params,
                    progress_callback=update_progress,
                    decorations_enabled=decorations_enabled,
                    decoration_density=decoration_density,
                    decoration_stroke_width=decoration_stroke_width,
                    decoration_scale=decoration_scale,
                )
            progress_bar.empty()
            st.session_state.svg_string = svg_string
            st.session_state.render_key = render_key
        else:
            svg_string = st.session_state.svg_string

        # Make SVG responsive for display
        display_svg = re.sub(r'width="\d+"', 'width="100%"', svg_string, count=1)
        display_svg = re.sub(r'height="\d+"', 'height="100%"', display_svg, count=1)

        # Zoom via CSS transform (no re-render needed)
        scale = zoom_level / 100.0

        html_content = f'''
        <div style="background:#f0f0f0; height:100%; overflow:auto; padding:20px; box-sizing:border-box;">
            <div style="transform-origin:top center; transform:scale({scale});
                        display:inline-block; background:white; padding:10px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        margin: 0 auto; display:block; width:fit-content;">
                <div style="width:80vmin; height:80vmin;">
                    {display_svg}
                </div>
            </div>
        </div>
        '''
        components.html(html_content, height=900, scrolling=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.download_button(
            "Download SVG",
            svg_string,
            file_name="pipes-{}.svg".format(timestamp),
            mime="image/svg+xml"
        )
    else:
        st.error("Failed to generate grid. Click 'Regenerate Layout' to try again.")

elif view_mode == "Tile Catalog":
    st.subheader("Tile Catalog")

    sizes = ['medium', 'narrow', 'tiny']

    # Build HTML table of all representative tiles
    html_parts = [
        '<table style="border-collapse:collapse; background:#f8f8f8; font-family:sans-serif;">',
        '<tr style="border-bottom:2px solid #ccc;">',
        '<th style="padding:8px 12px; text-align:left;">Shape</th>',
    ]
    for size in sizes:
        html_parts.append('<th style="padding:8px 12px; text-align:center;">{}</th>'.format(size))
    html_parts.append('</tr>')

    for shape_name, size_chars in pipe_core.CATALOG_TILES:
        html_parts.append('<tr style="border-bottom:1px solid #e0e0e0;">')
        html_parts.append(
            '<td style="padding:6px 12px; font-weight:bold; vertical-align:middle;'
            ' white-space:nowrap;">{}</td>'.format(shape_name))
        for size in sizes:
            ch = size_chars.get(size)
            if ch:
                svg_str = pipe_core.render_single_tile(
                    ch, stroke_width, shading_style, shading_stroke_width,
                    light_angle_deg=light_angle, shading_params=shading_params)
                # Strip XML declaration for inline embedding
                svg_str = re.sub(r'<\?xml[^?]*\?>\s*', '', svg_str)
                # Set display size
                svg_str = re.sub(r'width="\d+"', 'width="120"', svg_str, count=1)
                svg_str = re.sub(r'height="\d+"', 'height="120"', svg_str, count=1)
                label = '<div style="text-align:center; font-size:11px; color:#666; font-family:monospace;">{}</div>'.format(ch)
                html_parts.append(
                    '<td style="padding:4px 8px; text-align:center; vertical-align:middle;">'
                    '{}{}</td>'.format(svg_str, label))
            else:
                html_parts.append(
                    '<td style="padding:4px 8px; text-align:center; vertical-align:middle;'
                    ' color:#ccc;">&mdash;</td>')
        html_parts.append('</tr>')
    html_parts.append('</table>')

    catalog_html = '\n'.join(html_parts)
    components.html(catalog_html, height=2400, scrolling=True)
