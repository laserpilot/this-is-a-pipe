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
        ['none', 'accent', 'hatch', 'double-wall'],
        index=1
    )

    stroke_width = st.slider("Stroke Width", 0.2, 2.0, 0.5, 0.1)

    auto_shading_width = st.checkbox("Auto shading width (60% of stroke)", value=True)
    if auto_shading_width:
        shading_stroke_width = stroke_width * 0.6
        st.caption("Shading stroke: {:.2f}".format(shading_stroke_width))
    else:
        shading_stroke_width = st.slider("Shading Stroke Width", 0.1, 2.0, stroke_width * 0.6, 0.05)

    st.header("Grid Settings")
    grid_width = st.slider("Grid Width", 4, 16, 8)
    grid_height = st.slider("Grid Height", 4, 16, 8)

    if st.button("Regenerate Layout", type="primary"):
        st.session_state.pop('grid', None)
        st.session_state.pop('grid_dims', None)

# Generate grid if needed
current_dims = (grid_width, grid_height)

if 'grid' not in st.session_state or st.session_state.get('grid_dims') != current_dims:
    with st.spinner("Generating pipe layout..."):
        grid = pipe_core.wave_function_collapse(grid_width, grid_height)
        st.session_state.grid = grid
        st.session_state.grid_dims = current_dims

grid = st.session_state.grid

if grid:
    svg_string = pipe_core.render_svg(grid, stroke_width, shading_style, shading_stroke_width)
    # Make SVG responsive for preview (replace fixed width/height with 100%)
    display_svg = re.sub(r'width="\d+"', 'width="100%"', svg_string, count=1)
    display_svg = re.sub(r'height="\d+"', 'height="100%"', display_svg, count=1)
    html_content = '<div style="background:white; padding:10px;">' + display_svg + '</div>'
    components.html(html_content, height=700, scrolling=True)

    st.download_button(
        "Download SVG",
        svg_string,
        file_name="pipes-preview.svg",
        mime="image/svg+xml"
    )
else:
    st.error("Failed to generate grid. Click 'Regenerate Layout' to try again.")
