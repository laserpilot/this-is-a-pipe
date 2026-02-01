import streamlit as st
import streamlit.components.v1 as components
import pipe_core
import re
from datetime import datetime

st.set_page_config(page_title="Pipe Shading Preview", layout="wide")
st.title("Pipe Shading Preview")

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

    tile_weights = {
        'size': {'medium': w_medium, 'narrow': w_narrow, 'tiny': w_tiny},
        'shape': {'straight': w_straight, 'corner': w_corner,
                  'chamfer': w_chamfer, 'teardrop': w_teardrop,
                  'dodge': w_dodge, 'diagonal': w_diagonal,
                  'diagonal_endcap': w_diag_endcap,
                  'junction': w_junction, 'reducer': w_reducer,
                  'endcap': w_endcap, 'mixed_corner': w_mixed_corner},
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

    st.header("Multi-Layer")
    multilayer_enabled = st.checkbox("Enable Multi-Layer", value=False,
                                      help="Overlay two independent pipe networks")
    if multilayer_enabled:
        layer_0_color = st.text_input("Bottom Layer Color", value="red",
                                       help="data-color attribute for pen plotter")
        layer_1_color = st.text_input("Top Layer Color", value="black",
                                       help="data-color attribute for pen plotter")
    else:
        layer_0_color = "red"
        layer_1_color = "black"

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

# Generate grid(s) if needed
current_dims = (grid_width, grid_height, multilayer_enabled)

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
            stripe_offset=stripe_offset)
    else:
        return pipe_core.wave_function_collapse(
            grid_width, grid_height, tile_weights=tile_weights,
            progress_callback=wfc_update)

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
                  multilayer_enabled, layer_0_color, layer_1_color)

    if st.session_state.get('render_key') != render_key or 'svg_string' not in st.session_state:
        progress_bar = st.progress(0, text="Rendering tiles...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text="Rendering tile {} / {}".format(current, total))

        if multilayer_enabled:
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
