import streamlit as st
import streamlit.components.v1 as components
import pipe_core
import re

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
        w_junction = st.slider("Junctions", 0.1, 5.0, 2.0, 0.1)
        w_reducer = st.slider("Reducers", 0.1, 5.0, 1.0, 0.1)

    tile_weights = {
        'size': {'medium': w_medium, 'narrow': w_narrow, 'tiny': w_tiny},
        'shape': {'straight': w_straight, 'corner': w_corner,
                  'chamfer': w_chamfer, 'teardrop': w_teardrop,
                  'dodge': w_dodge, 'diagonal': w_diagonal,
                  'junction': w_junction, 'reducer': w_reducer},
    }

    st.header("Grid Settings")
    grid_width = st.slider("Grid Width", 4, 40, 8)
    grid_height = st.slider("Grid Height", 4, 40, 8)

    st.header("View Settings")
    zoom_level = st.slider("Zoom", 25, 200, 100, 5, help="Zoom level (100% = fit to window)")

    if st.button("Regenerate Layout", type="primary"):
        st.session_state.pop('grid', None)
        st.session_state.pop('grid_dims', None)

# Generate grid if needed
current_dims = (grid_width, grid_height)

if 'grid' not in st.session_state or st.session_state.get('grid_dims') != current_dims:
    wfc_progress = st.progress(0, text="Generating pipe layout...")

    def wfc_update(attempt, max_attempts, cells, total):
        pct = cells / total
        wfc_progress.progress(pct, text="Attempt {} — {}/{} cells".format(attempt, cells, total))

    grid = pipe_core.wave_function_collapse(grid_width, grid_height,
                                             tile_weights=tile_weights,
                                             progress_callback=wfc_update)
    wfc_progress.empty()
    st.session_state.grid = grid
    st.session_state.grid_dims = current_dims

grid = st.session_state.grid

if grid:
    progress_bar = st.progress(0, text="Rendering tiles...")

    def update_progress(current, total):
        progress_bar.progress(current / total, text="Rendering tile {} / {}".format(current, total))

    svg_string = pipe_core.render_svg(
        grid, stroke_width, shading_style, shading_stroke_width,
        light_angle_deg=light_angle, shading_params=shading_params,
        progress_callback=update_progress
    )
    progress_bar.empty()
    # Make SVG responsive for display
    display_svg = re.sub(r'width="\d+"', 'width="100%"', svg_string, count=1)
    display_svg = re.sub(r'height="\d+"', 'height="100%"', display_svg, count=1)

    # Use viewport-relative sizing for zoom
    # At 100%, SVG fills nicely; <100% shrinks to see composition; >100% enlarges for detail
    svg_size = zoom_level

    html_content = f'''
    <div style="background:#f0f0f0; height:100%; display:flex; align-items:center;
                justify-content:center; overflow:auto; padding:20px; box-sizing:border-box;">
        <div style="background:white; padding:10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="width:{svg_size}vmin; height:{svg_size}vmin;">
                {display_svg}
            </div>
        </div>
    </div>
    '''
    components.html(html_content, height=700, scrolling=True)

    st.download_button(
        "Download SVG",
        svg_string,
        file_name="pipes-preview.svg",
        mime="image/svg+xml"
    )
else:
    st.error("Failed to generate grid. Click 'Regenerate Layout' to try again.")
