import drawsvg as draw
import random
import math
from collections import deque
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely import affinity


# ============================================================================
# SCALED MULTI-LAYER HELPERS
# ============================================================================

def make_layer_spec(grid, scale=1.0, offset=(0, 0),
                    stroke_width=0.5, shading_style='directional-hatch',
                    shading_stroke_width=0.3, shading_params=None,
                    decorations_enabled=False, decoration_density=0.10,
                    decoration_stroke_width=None, decoration_scale=1.0,
                    name=None, color='black',
                    grid_width=None, grid_height=None):
    """Create a LayerSpec dict with all per-layer rendering parameters.

    grid_width/grid_height: optional, for density-preserving mode reference.
    """
    return {
        'grid': grid,
        'scale': scale,
        'offset': offset,
        'stroke_width': stroke_width,
        'shading_style': shading_style,
        'shading_stroke_width': shading_stroke_width,
        'shading_params': shading_params,
        'decorations_enabled': decorations_enabled,
        'decoration_density': decoration_density,
        'decoration_stroke_width': decoration_stroke_width,
        'decoration_scale': decoration_scale,
        'name': name,
        'color': color,
        'grid_width': grid_width,
        'grid_height': grid_height,
    }


def compute_density_preserving_scale(base_width, base_height,
                                      layer_width, layer_height):
    """Compute scale to fit layer_dims into base_dims footprint.

    Returns scale such that layer at layer_dims fits the same world
    footprint as base_dims. Uses minimum of x/y ratios to ensure
    the layer fits entirely.
    """
    if layer_width <= 0 or layer_height <= 0:
        return 1.0
    scale_x = base_width / layer_width
    scale_y = base_height / layer_height
    return min(scale_x, scale_y)


def _poly_local_to_world(poly, scale, offset):
    """Transform a polygon from layer-local coords to world coords.

    World = (local * scale) + offset.
    """
    if poly is None:
        return None
    result = affinity.scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
    if offset != (0, 0):
        result = affinity.translate(result, xoff=offset[0], yoff=offset[1])
    return result


def _poly_world_to_local(poly, scale, offset):
    """Transform a polygon from world coords to layer-local coords.

    Local = (world - offset) / scale.
    """
    if poly is None:
        return None
    result = poly
    if offset != (0, 0):
        result = affinity.translate(result, xoff=-offset[0], yoff=-offset[1])
    result = affinity.scale(result, xfact=1.0 / scale, yfact=1.0 / scale, origin=(0, 0))
    return result


# ============================================================================
# DIRECTIONAL SHADING HELPERS
# ============================================================================

# Default shading parameters (can be overridden via render_svg)
DEFAULT_SHADING_PARAMS = {
    'band_width': 12,       # width of hatch band (out of 30 half-width)
    'band_offset': 3,       # gap from pipe wall to band edge
    'spacing': 10,          # base spacing between hatch lines
    'angle': 30,            # base angle of hatches relative to pipe tangent
    'jitter_pos': 1.5,      # max position jitter in units
    'jitter_angle': 3.0,    # max angle jitter in degrees
    'crosshatch': False,    # enable second hatch direction
    'crosshatch_angle': 90, # offset from primary hatch angle
    'band_width_jitter': 0.0,  # max fraction to vary hatch length (0-1)
    'wiggle': 0.0,          # max perpendicular displacement for curved hatches
}


def _normalize(v):
    mag = math.sqrt(v[0]**2 + v[1]**2)
    if mag < 1e-10:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _rotate_vec(v, angle_deg):
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return (v[0] * cos_a - v[1] * sin_a, v[0] * sin_a + v[1] * cos_a)


def _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                     pipe_polygon=None, occlusion_polygon=None):
    """Draw a single hatch line, optionally with wiggle (curved) effect."""
    if wiggle > 0:
        # Calculate perpendicular direction for wiggle
        perp = (-hatch_dir[1], hatch_dir[0])
        wiggle_offset = random.uniform(-wiggle, wiggle)
        mid_x = (x1 + x2) / 2 + perp[0] * wiggle_offset
        mid_y = (y1 + y2) / 2 + perp[1] * wiggle_offset

        # Draw as quadratic bezier curve
        if pipe_polygon is not None:
            # For clipped wiggly lines, we approximate with segments
            # Clip each segment separately
            segments = [(x1, y1, mid_x, mid_y), (mid_x, mid_y, x2, y2)]
            for sx1, sy1, sx2, sy2 in segments:
                clipped = clip_line_to_polygon(sx1, sy1, sx2, sy2, pipe_polygon, occlusion_polygon)
                for cx1, cy1, cx2, cy2 in clipped:
                    drawing.append(draw.Line(cx1, cy1, cx2, cy2,
                                             stroke='black', stroke_width=sw, fill='none'))
        else:
            # Draw as path with quadratic curve
            path = draw.Path(stroke='black', stroke_width=sw, fill='none')
            path.M(x1, y1).Q(mid_x, mid_y, x2, y2)
            drawing.append(path)
    else:
        # Straight line
        if pipe_polygon is not None:
            clipped_lines = clip_line_to_polygon(x1, y1, x2, y2, pipe_polygon, occlusion_polygon)
            for cx1, cy1, cx2, cy2 in clipped_lines:
                drawing.append(draw.Line(cx1, cy1, cx2, cy2,
                                         stroke='black', stroke_width=sw, fill='none'))
        else:
            drawing.append(draw.Line(x1, y1, x2, y2,
                                     stroke='black', stroke_width=sw, fill='none'))


# ============================================================================
# POLYGON CLIPPING HELPERS
# ============================================================================

# Pipe width parameters: half-width from center to wall
PIPE_HALF_WIDTHS = {
    't': 5,   # Tiny (very narrow) - 10 units total width
    'n': 12,  # Narrow - 24 units total width
    'm': 30,  # Medium (default) - 60 units total width
}


def get_tube_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a straight tube segment."""
    # Determine orientation and width
    if ch == '|':
        hw = 30  # medium half-width
        return Polygon([
            (xloc - hw, yloc - 50), (xloc + hw, yloc - 50),
            (xloc + hw, yloc + 50), (xloc - hw, yloc + 50)
        ])
    elif ch == '-':
        hw = 30
        return Polygon([
            (xloc - 50, yloc - hw), (xloc + 50, yloc - hw),
            (xloc + 50, yloc + hw), (xloc - 50, yloc + hw)
        ])
    elif ch == 'i':  # Narrow vertical
        hw = 12
        return Polygon([
            (xloc - hw, yloc - 50), (xloc + hw, yloc - 50),
            (xloc + hw, yloc + 50), (xloc - hw, yloc + 50)
        ])
    elif ch == '=':  # Narrow horizontal
        hw = 12
        return Polygon([
            (xloc - 50, yloc - hw), (xloc + 50, yloc - hw),
            (xloc + 50, yloc + hw), (xloc - 50, yloc + hw)
        ])
    elif ch == '!':  # Tiny vertical
        hw = 5
        return Polygon([
            (xloc - hw, yloc - 50), (xloc + hw, yloc - 50),
            (xloc + hw, yloc + 50), (xloc - hw, yloc + 50)
        ])
    elif ch == '.':  # Tiny horizontal
        hw = 5
        return Polygon([
            (xloc - 50, yloc - hw), (xloc + 50, yloc - hw),
            (xloc + 50, yloc + hw), (xloc - 50, yloc + hw)
        ])
    return None


def get_sized_corner_polygon(ch, xloc, yloc, half_width, num_arc_points=16):
    """Return Shapely Polygon for a corner arc with specified half-width."""
    # Map corner chars to rotations
    base_chars = {'r': 0, '7': 90, 'j': 180, 'L': 270,
                  'nr': 0, 'n7': 90, 'nj': 180, 'nL': 270,
                  'tr': 0, 't7': 90, 'tj': 180, 'tL': 270}
    rot_deg = base_chars.get(ch)
    if rot_deg is None:
        return None

    # Arc radii based on half-width - scale proportionally for all pipe sizes
    inner_radius = half_width / 3
    center_offset = half_width + inner_radius
    outer_radius = 2 * half_width + inner_radius

    local_center = (center_offset, center_offset)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Build arc polygon
    arc_points = []
    for i in range(num_arc_points + 1):
        frac = i / num_arc_points
        angle_deg = arc_start + frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)
        x = arc_cx + outer_radius * math.cos(angle_rad)
        y = arc_cy + outer_radius * math.sin(angle_rad)
        arc_points.append((x, y))

    for i in range(num_arc_points, -1, -1):
        frac = i / num_arc_points
        angle_deg = arc_start + frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)
        x = arc_cx + inner_radius * math.cos(angle_rad)
        y = arc_cy + inner_radius * math.sin(angle_rad)
        arc_points.append((x, y))

    arc_polygon = Polygon(arc_points)

    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Stub rectangles: connect arc to cell edge
    # Match medium corner proportions: stub starts at arc center (center_offset)
    stub_start = center_offset
    stub_end = 50
    stub_extent = outer_radius / 2

    v_stub_points = [
        transform_point(stub_start, -stub_extent),
        transform_point(stub_end, -stub_extent),
        transform_point(stub_end, stub_extent),
        transform_point(stub_start, stub_extent),
    ]
    v_stub = Polygon(v_stub_points)

    h_stub_points = [
        transform_point(-stub_extent, stub_start),
        transform_point(stub_extent, stub_start),
        transform_point(stub_extent, stub_end),
        transform_point(-stub_extent, stub_end),
    ]
    h_stub = Polygon(h_stub_points)

    return unary_union([arc_polygon, v_stub, h_stub])


def get_reducer_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a reducer tile (tapers between two sizes).

    Includes flange geometry so shading is properly clipped around flanges.
    """
    # Reducer mapping: start_size, end_size, orientation
    reducers = {
        'Rv': ('m', 'n', 'v'),  # Medium top, narrow bottom
        'RV': ('n', 'm', 'v'),  # Narrow top, medium bottom
        'Tv': ('n', 't', 'v'),  # Narrow top, tiny bottom
        'TV': ('t', 'n', 'v'),  # Tiny top, narrow bottom
        'Rh': ('m', 'n', 'h'),  # Medium right, narrow left
        'RH': ('n', 'm', 'h'),  # Narrow right, medium left
        'Th': ('n', 't', 'h'),  # Narrow right, tiny left
        'TH': ('t', 'n', 'h'),  # Tiny right, narrow left
    }

    if ch not in reducers:
        return None

    start_size, end_size, orient = reducers[ch]
    hw_start = PIPE_HALF_WIDTHS[start_size]
    hw_end = PIPE_HALF_WIDTHS[end_size]

    # Flange dimensions (must match draw_reducer_flanges)
    flange_start = hw_start * 0.15
    flange_end = hw_end * 0.15
    flange_width = 2

    if orient == 'v':  # Vertical: tapers along Y axis
        # Main pipe body
        pipe_poly = Polygon([
            (xloc - hw_start, yloc - 50),
            (xloc + hw_start, yloc - 50),
            (xloc + hw_start, yloc - 20),  # Start taper
            (xloc + hw_end, yloc + 20),    # End taper
            (xloc + hw_end, yloc + 50),
            (xloc - hw_end, yloc + 50),
            (xloc - hw_end, yloc + 20),
            (xloc - hw_start, yloc - 20),
        ])
        # Flange at start (y=-20)
        flange1 = Polygon([
            (xloc - hw_start - flange_start, yloc - 20 - flange_width),
            (xloc + hw_start + flange_start, yloc - 20 - flange_width),
            (xloc + hw_start + flange_start, yloc - 20 + flange_width),
            (xloc - hw_start - flange_start, yloc - 20 + flange_width),
        ])
        # Flange at end (y=+20)
        flange2 = Polygon([
            (xloc - hw_end - flange_end, yloc + 20 - flange_width),
            (xloc + hw_end + flange_end, yloc + 20 - flange_width),
            (xloc + hw_end + flange_end, yloc + 20 + flange_width),
            (xloc - hw_end - flange_end, yloc + 20 + flange_width),
        ])
        return unary_union([pipe_poly, flange1, flange2])
    else:  # Horizontal: tapers along X axis
        # Main pipe body
        pipe_poly = Polygon([
            (xloc + 50, yloc - hw_start),
            (xloc + 50, yloc + hw_start),
            (xloc + 20, yloc + hw_start),  # Start taper
            (xloc - 20, yloc + hw_end),    # End taper
            (xloc - 50, yloc + hw_end),
            (xloc - 50, yloc - hw_end),
            (xloc - 20, yloc - hw_end),
            (xloc + 20, yloc - hw_start),
        ])
        # Flange at start (x=+20)
        flange1 = Polygon([
            (xloc + 20 - flange_width, yloc - hw_start - flange_start),
            (xloc + 20 + flange_width, yloc - hw_start - flange_start),
            (xloc + 20 + flange_width, yloc + hw_start + flange_start),
            (xloc + 20 - flange_width, yloc + hw_start + flange_start),
        ])
        # Flange at end (x=-20)
        flange2 = Polygon([
            (xloc - 20 - flange_width, yloc - hw_end - flange_end),
            (xloc - 20 + flange_width, yloc - hw_end - flange_end),
            (xloc - 20 + flange_width, yloc + hw_end + flange_end),
            (xloc - 20 - flange_width, yloc + hw_end + flange_end),
        ])
        return unary_union([pipe_poly, flange1, flange2])


# ============================================================================
# DIAGONAL PIPES
# ============================================================================

_SQRT2 = math.sqrt(2)
_INV_SQRT2 = 1.0 / _SQRT2

# Diagonal straight params: char -> (direction, half_width)
# 'a' = ascending (SW->NE, like /), 'd' = descending (NW->SE, like \)
_DIAG_PARAMS = {
    'nDa': ('a', 12), 'nDd': ('d', 12),
    'tDa': ('a', 5),  'tDd': ('d', 5),
}

# Adapter params: char -> (shape, rotation_deg, half_width)
# Shape A: vertical cardinal -> diagonal (base: S->NE)
# Shape B: horizontal cardinal -> diagonal (base: W->NE)
_ADAPTER_PARAMS = {
    # Shape A (narrow)
    'naSne': ('A', 0, 12), 'naWse': ('A', 90, 12),
    'naNsw': ('A', 180, 12), 'naEnw': ('A', 270, 12),
    # Shape A (tiny)
    'taSne': ('A', 0, 5), 'taWse': ('A', 90, 5),
    'taNsw': ('A', 180, 5), 'taEnw': ('A', 270, 5),
    # Shape B (narrow)
    'naWne': ('B', 0, 12), 'naNse': ('B', 90, 12),
    'naEsw': ('B', 180, 12), 'naSnw': ('B', 270, 12),
    # Shape B (tiny)
    'taWne': ('B', 0, 5), 'taNse': ('B', 90, 5),
    'taEsw': ('B', 180, 5), 'taSnw': ('B', 270, 5),
}

# Diagonal corner params: char -> (rotation_deg, half_width)
# Base shape (0°) is east-pointing V connecting NE+SE ports.
# 90° CCW rotation maps NE→SE, SE→SW (south-pointing).
_DIAG_CORNER_PARAMS = {
    'ndce': (0, 12),   'ndcs': (90, 12),
    'ndcw': (180, 12), 'ndcn': (270, 12),
    'tdce': (0, 5),    'tdcs': (90, 5),
    'tdcw': (180, 5),  'tdcn': (270, 5),
}

# Diagonal endcap params: char -> (direction, half_width)
# Half-length diagonal from center to one corner, capped at center end.
_DIAG_ENDCAP_PARAMS = {
    'ndne': ('NE', 12), 'ndse': ('SE', 12),
    'ndsw': ('SW', 12), 'ndnw': ('NW', 12),
    'tdne': ('NE', 5),  'tdse': ('SE', 5),
    'tdsw': ('SW', 5),  'tdnw': ('NW', 5),
}

# Direction vectors for diagonal endcaps: corner offsets and tangent
_DIAG_DIR_VECS = {
    'NE': (50, -50), 'SE': (50, 50),
    'SW': (-50, 50), 'NW': (-50, -50),
}


def get_diagonal_polygon(ch, xloc, yloc):
    """Return parallelogram polygon for a diagonal straight pipe."""
    direction, hw = _DIAG_PARAMS[ch]
    offset = hw * _INV_SQRT2

    if direction == 'a':  # ascending SW->NE
        pts = [
            (xloc - 50 - offset, yloc + 50 - offset),  # SW inner (NW side)
            (xloc - 50 + offset, yloc + 50 + offset),  # SW outer (SE side)
            (xloc + 50 + offset, yloc - 50 + offset),  # NE outer (SE side)
            (xloc + 50 - offset, yloc - 50 - offset),  # NE inner (NW side)
        ]
    else:  # descending NW->SE
        pts = [
            (xloc - 50 - offset, yloc - 50 + offset),  # NW outer (SW side)
            (xloc + 50 - offset, yloc + 50 + offset),  # SE outer (SW side)
            (xloc + 50 + offset, yloc + 50 - offset),  # SE inner (NE side)
            (xloc - 50 + offset, yloc - 50 - offset),  # NW inner (NE side)
        ]

    poly = Polygon(pts)
    poly = poly.buffer(0)
    if not poly.is_valid:
        return None
    return poly


def draw_diagonal_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a diagonal straight pipe."""
    direction, hw = _DIAG_PARAMS[ch]
    offset = hw * _INV_SQRT2

    if direction == 'a':
        # NW wall (inner)
        clip_and_draw_line(drawing,
                           xloc - 50 - offset, yloc + 50 - offset,
                           xloc + 50 - offset, yloc - 50 - offset,
                           occlusion_poly, sw)
        # SE wall (outer)
        clip_and_draw_line(drawing,
                           xloc - 50 + offset, yloc + 50 + offset,
                           xloc + 50 + offset, yloc - 50 + offset,
                           occlusion_poly, sw)
    else:
        # SW wall (outer)
        clip_and_draw_line(drawing,
                           xloc - 50 - offset, yloc - 50 + offset,
                           xloc + 50 - offset, yloc + 50 + offset,
                           occlusion_poly, sw)
        # NE wall (inner)
        clip_and_draw_line(drawing,
                           xloc - 50 + offset, yloc - 50 - offset,
                           xloc + 50 + offset, yloc + 50 - offset,
                           occlusion_poly, sw)


def draw_diagonal_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                       pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a diagonal straight pipe."""
    direction, hw = _DIAG_PARAMS[ch]

    if direction == 'a':
        tangent = (_INV_SQRT2, -_INV_SQRT2)
        normal_left = (-_INV_SQRT2, -_INV_SQRT2)   # NW side
        normal_right = (_INV_SQRT2, _INV_SQRT2)     # SE side
    else:
        tangent = (_INV_SQRT2, _INV_SQRT2)
        normal_left = (_INV_SQRT2, -_INV_SQRT2)     # NE side
        normal_right = (-_INV_SQRT2, _INV_SQRT2)    # SW side

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right
    band_center_dist = hw - band_offset - band_width / 2

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Pipe diagonal length is 100*sqrt(2), sample along that
    diag_length = 100 * _SQRT2
    num_hatches = max(1, int(diag_length / spacing))
    for base_angle in angles_to_draw:
        for i in range(num_hatches + 1):
            t = -diag_length / 2 + i * (diag_length / num_hatches)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
            base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


def _adapter_base_polygon_A(hw):
    """Compute base polygon vertices for shape A adapter (S->NE) in local coords."""
    offset = hw * _INV_SQRT2
    sqrt2_minus_1 = _SQRT2 - 1

    return [
        (-hw, 50),                              # S edge, west wall
        (hw, 50),                               # S edge, east wall
        (hw, hw * sqrt2_minus_1),               # outer miter
        (50 + offset, -50 + offset),            # NE corner, SE wall
        (50 - offset, -50 - offset),            # NE corner, NW wall
        (-hw, hw * (1 - _SQRT2)),               # inner miter
    ]


def _adapter_base_polygon_B(hw):
    """Compute base polygon vertices for shape B adapter (W->NE) in local coords."""
    offset = hw * _INV_SQRT2

    return [
        (-50, -hw),                             # W edge, north wall
        (-50, hw),                              # W edge, south wall
        (0, hw),                                # south wall endpoint at center
        (offset, offset),                       # SE diagonal wall start (chamfer)
        (50 + offset, -50 + offset),            # NE corner, SE wall
        (50 - offset, -50 - offset),            # NE corner, NW wall
        (hw * (1 - _SQRT2), -hw),               # inner miter
    ]


def get_adapter_polygon(ch, xloc, yloc):
    """Return polygon for an adapter tile (cardinal-to-diagonal bend)."""
    shape, rot_deg, hw = _ADAPTER_PARAMS[ch]

    if shape == 'A':
        local_pts = _adapter_base_polygon_A(hw)
    else:
        local_pts = _adapter_base_polygon_B(hw)

    cos_r = math.cos(math.radians(rot_deg))
    sin_r = math.sin(math.radians(rot_deg))

    world_pts = []
    for px, py in local_pts:
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        world_pts.append((xloc + rx, yloc + ry))

    poly = Polygon(world_pts)
    poly = poly.buffer(0)
    if not poly.is_valid:
        return None
    return poly


def draw_adapter_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for an adapter tile."""
    shape, rot_deg, hw = _ADAPTER_PARAMS[ch]

    if shape == 'A':
        local_pts = _adapter_base_polygon_A(hw)
    else:
        local_pts = _adapter_base_polygon_B(hw)

    cos_r = math.cos(math.radians(rot_deg))
    sin_r = math.sin(math.radians(rot_deg))

    def to_world(px, py):
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        return (xloc + rx, yloc + ry)

    wp = [to_world(px, py) for px, py in local_pts]
    n = len(wp)

    # Draw each edge of the polygon as a clipped line segment
    for i in range(n):
        j = (i + 1) % n
        clip_and_draw_line(drawing, wp[i][0], wp[i][1], wp[j][0], wp[j][1],
                           occlusion_poly, sw)


def draw_adapter_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                      pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on an adapter tile.

    Two sections: cardinal half (straight) and diagonal half, each with its own tangent.
    """
    shape, rot_deg, hw = _ADAPTER_PARAMS[ch]

    cos_r = math.cos(math.radians(rot_deg))
    sin_r = math.sin(math.radians(rot_deg))

    def to_world(px, py):
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        return (xloc + rx, yloc + ry)

    def rot_dir(dx, dy):
        return (dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r)

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    band_center_dist = hw - band_offset - band_width / 2

    # Section 1: cardinal half (base orientation for A: vertical from y=50 to y=0)
    if shape == 'A':
        local_tangent1 = (0, -1)  # pointing north (toward center)
        local_n1_left = (-1, 0)
        local_n1_right = (1, 0)
        section_length1 = 50
    else:
        local_tangent1 = (1, 0)   # pointing east (toward center)
        local_n1_left = (0, -1)
        local_n1_right = (0, 1)
        section_length1 = 50

    tangent1 = rot_dir(*local_tangent1)
    n1_left = rot_dir(*local_n1_left)
    n1_right = rot_dir(*local_n1_right)

    dot1_l = _dot(n1_left, light_dir)
    dot1_r = _dot(n1_right, light_dir)
    shadow1 = n1_left if dot1_l < dot1_r else n1_right

    # Section 2: diagonal half (base: toward NE corner)
    local_tangent2 = (_INV_SQRT2, -_INV_SQRT2)
    local_n2_left = (-_INV_SQRT2, -_INV_SQRT2)
    local_n2_right = (_INV_SQRT2, _INV_SQRT2)

    tangent2 = rot_dir(*local_tangent2)
    n2_left = rot_dir(*local_n2_left)
    n2_right = rot_dir(*local_n2_right)

    dot2_l = _dot(n2_left, light_dir)
    dot2_r = _dot(n2_right, light_dir)
    shadow2 = n2_left if dot2_l < dot2_r else n2_right

    diag_length = 50 * _SQRT2  # half the cell diagonal

    # Draw section 1 (cardinal half)
    num_hatches1 = max(1, int(section_length1 / spacing))
    if shape == 'A':
        section1_center = to_world(0, 25)  # midpoint of vertical stub
    else:
        section1_center = to_world(-25, 0)  # midpoint of horizontal stub

    for base_angle in angles_to_draw:
        for i in range(num_hatches1 + 1):
            t = -section_length1 / 2 + i * (section_length1 / num_hatches1)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = section1_center[0] + tangent1[0] * t + shadow1[0] * band_center_dist
            base_y = section1_center[1] + tangent1[1] * t + shadow1[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent1, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)

    # Draw section 2 (diagonal half)
    num_hatches2 = max(1, int(diag_length / spacing))
    diag_center = to_world(25 * _INV_SQRT2 * _SQRT2, -25 * _INV_SQRT2 * _SQRT2)
    # Simpler: center of the diagonal half is at (25, -25) in local coords
    diag_center = to_world(25, -25)

    for base_angle in angles_to_draw:
        for i in range(num_hatches2 + 1):
            t = -diag_length / 2 + i * (diag_length / num_hatches2)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = diag_center[0] + tangent2[0] * t + shadow2[0] * band_center_dist
            base_y = diag_center[1] + tangent2[1] * t + shadow2[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent2, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


# ---------------------------------------------------------------------------
# Diagonal corner tiles (V-shaped turns between two diagonal directions)
# ---------------------------------------------------------------------------

def _diag_corner_base_polygon(hw):
    """Base polygon for east-pointing V corner (NE+SE) in local coords.

    Returns 6 vertices: NE-inner, NE-outer, outer-miter, SE-outer, SE-inner, inner-miter.
    """
    off = hw * _INV_SQRT2
    hw_sqrt2 = hw * _SQRT2
    return [
        (50 - off, -50 - off),     # NE inner (NW wall at NE corner)
        (50 + off, -50 + off),     # NE outer (SE wall at NE corner)
        (hw_sqrt2, 0),             # Outer miter (east)
        (50 + off,  50 - off),     # SE outer (NE wall at SE corner)
        (50 - off,  50 + off),     # SE inner (SW wall at SE corner)
        (-hw_sqrt2, 0),            # Inner miter (west)
    ]


def get_diagonal_corner_polygon(ch, xloc, yloc):
    """Return polygon for a diagonal corner (V-shaped turn)."""
    rot_deg, hw = _DIAG_CORNER_PARAMS[ch]
    local_pts = _diag_corner_base_polygon(hw)

    cos_r = math.cos(math.radians(rot_deg))
    sin_r = math.sin(math.radians(rot_deg))

    world_pts = []
    for px, py in local_pts:
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        world_pts.append((xloc + rx, yloc + ry))

    poly = Polygon(world_pts)
    poly = poly.buffer(0)
    if not poly.is_valid:
        return None
    return poly


def draw_diagonal_corner_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a diagonal corner tile (all polygon edges)."""
    rot_deg, hw = _DIAG_CORNER_PARAMS[ch]
    local_pts = _diag_corner_base_polygon(hw)

    cos_r = math.cos(math.radians(rot_deg))
    sin_r = math.sin(math.radians(rot_deg))

    def to_world(px, py):
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        return (xloc + rx, yloc + ry)

    wp = [to_world(px, py) for px, py in local_pts]
    n = len(wp)
    for i in range(n):
        j = (i + 1) % n
        clip_and_draw_line(drawing, wp[i][0], wp[i][1], wp[j][0], wp[j][1],
                           occlusion_poly, sw)


def draw_diagonal_corner_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                              pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a diagonal corner tile.

    Two sections (NE half and SE half in base orientation), each with its own
    tangent/normal for shading, rotated by the tile's rotation angle.
    """
    rot_deg, hw = _DIAG_CORNER_PARAMS[ch]

    cos_r = math.cos(math.radians(rot_deg))
    sin_r = math.sin(math.radians(rot_deg))

    def rot_dir(dx, dy):
        return (dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r)

    # Base tangent/normal for each segment (before rotation)
    # NE segment: tangent (1,-1)/√2, normals (1,1)/√2 and (-1,-1)/√2
    # SE segment: tangent (1,1)/√2, normals (-1,1)/√2 and (1,-1)/√2
    segments = [
        ((_INV_SQRT2, -_INV_SQRT2), (-_INV_SQRT2, -_INV_SQRT2), (_INV_SQRT2, _INV_SQRT2)),
        ((_INV_SQRT2, _INV_SQRT2), (_INV_SQRT2, -_INV_SQRT2), (-_INV_SQRT2, _INV_SQRT2)),
    ]

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Each segment is half the diagonal length (center to corner)
    seg_length = 50 * _SQRT2
    band_center_dist = hw - band_offset - band_width / 2

    for base_tangent, base_nl, base_nr in segments:
        tangent = rot_dir(*base_tangent)
        normal_left = rot_dir(*base_nl)
        normal_right = rot_dir(*base_nr)

        dot_left = _dot(normal_left, light_dir)
        dot_right = _dot(normal_right, light_dir)
        shadow_normal = normal_left if dot_left < dot_right else normal_right

        num_hatches = max(1, int(seg_length / spacing))
        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                t = i * (seg_length / num_hatches)
                t += random.uniform(-jitter_pos, jitter_pos)

                base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
                base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


# ---------------------------------------------------------------------------
# Diagonal endcap tiles (half-length diagonal with cap at center end)
# ---------------------------------------------------------------------------

def get_diagonal_endcap_polygon(ch, xloc, yloc):
    """Return polygon for a diagonal endcap (half-diagonal with cap)."""
    direction, hw = _DIAG_ENDCAP_PARAMS[ch]
    cx, cy = _DIAG_DIR_VECS[direction]
    # Tangent from center to corner
    length = math.hypot(cx, cy)
    tx, ty = cx / length, cy / length
    # Normal (perpendicular): rotate tangent 90° CW → (ty, -tx)
    # and CCW → (-ty, tx). We call them "right" and "left".
    # For direction (tx, ty), perpendicular is (-ty, tx) [left] and (ty, -tx) [right]
    nlx, nly = -ty, tx    # left normal
    nrx, nry = ty, -tx    # right normal

    # Wall offsets at ±hw from centerline
    # At center (0,0):
    cap_left = (nlx * hw, nly * hw)
    cap_right = (nrx * hw, nry * hw)
    # At corner (cx, cy):
    corner_left = (cx + nlx * hw, cy + nly * hw)
    corner_right = (cx + nrx * hw, cy + nry * hw)

    pts = [
        (xloc + cap_left[0], yloc + cap_left[1]),       # Cap left
        (xloc + cap_right[0], yloc + cap_right[1]),      # Cap right
        (xloc + corner_right[0], yloc + corner_right[1]),  # Corner right
        (xloc + corner_left[0], yloc + corner_left[1]),    # Corner left
    ]

    poly = Polygon(pts)
    poly = poly.buffer(0)
    if not poly.is_valid:
        return None
    return poly


def draw_diagonal_endcap_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a diagonal endcap (2 wall lines + 1 cap line)."""
    direction, hw = _DIAG_ENDCAP_PARAMS[ch]
    cx, cy = _DIAG_DIR_VECS[direction]
    length = math.hypot(cx, cy)
    tx, ty = cx / length, cy / length
    nlx, nly = -ty, tx
    nrx, nry = ty, -tx

    cap_l = (xloc + nlx * hw, yloc + nly * hw)
    cap_r = (xloc + nrx * hw, yloc + nry * hw)
    corner_l = (xloc + cx + nlx * hw, yloc + cy + nly * hw)
    corner_r = (xloc + cx + nrx * hw, yloc + cy + nry * hw)

    # Cap line (across center end)
    clip_and_draw_line(drawing, cap_l[0], cap_l[1], cap_r[0], cap_r[1],
                       occlusion_poly, sw)
    # Left wall
    clip_and_draw_line(drawing, cap_l[0], cap_l[1], corner_l[0], corner_l[1],
                       occlusion_poly, sw)
    # Right wall
    clip_and_draw_line(drawing, cap_r[0], cap_r[1], corner_r[0], corner_r[1],
                       occlusion_poly, sw)


def draw_diagonal_endcap_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                              pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a diagonal endcap tile."""
    direction, hw = _DIAG_ENDCAP_PARAMS[ch]
    cx, cy = _DIAG_DIR_VECS[direction]
    length = math.hypot(cx, cy)
    tx, ty = cx / length, cy / length
    tangent = (tx, ty)

    normal_left = (-ty, tx)
    normal_right = (ty, -tx)

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right
    band_center_dist = hw - band_offset - band_width / 2

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Half diagonal length (center to corner)
    seg_length = 50 * _SQRT2
    num_hatches = max(1, int(seg_length / spacing))

    for base_angle in angles_to_draw:
        for i in range(num_hatches + 1):
            t = i * (seg_length / num_hatches)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
            base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


# Cardinal endcap params: char -> (direction, half_width)
# Half-cell dead-end tile, open on one cardinal side, capped at center.
_ENDCAP_PARAMS = {
    'eN': ('N', 30),  'eS': ('S', 30),  'eE': ('E', 30),  'eW': ('W', 30),
    'neN': ('N', 12), 'neS': ('S', 12), 'neE': ('E', 12), 'neW': ('W', 12),
    'teN': ('N', 5),  'teS': ('S', 5),  'teE': ('E', 5),  'teW': ('W', 5),
}


def get_endcap_polygon(ch, xloc, yloc, num_arc_points=8):
    """Return Shapely Polygon for a cardinal endcap (half-cell with rounded cap)."""
    direction, hw = _ENDCAP_PARAMS[ch]

    # Build arc points for the rounded cap end (semicircle curving inward)
    def _arc(cx, cy, r, start_deg, end_deg):
        pts = []
        for i in range(num_arc_points + 1):
            a = math.radians(start_deg + (end_deg - start_deg) * i / num_arc_points)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    if direction == 'N':
        # Port at top, cap curves south at y=yloc
        arc = _arc(xloc, yloc, hw, 0, 180)
        pts = [(xloc - hw, yloc - 50), (xloc + hw, yloc - 50)] + arc
    elif direction == 'S':
        # Port at bottom, cap curves north at y=yloc
        arc = _arc(xloc, yloc, hw, 180, 360)
        pts = arc + [(xloc + hw, yloc + 50), (xloc - hw, yloc + 50)]
    elif direction == 'E':
        # Port at right, cap curves west at x=xloc
        # Arc from 270° (top) to 90° (bottom), curving west
        arc = _arc(xloc, yloc, hw, 270, 90)
        pts = [(xloc + 50, yloc - hw)] + arc + [(xloc + 50, yloc + hw)]
    elif direction == 'W':
        # Port at left, cap curves east at x=xloc
        arc = _arc(xloc, yloc, hw, -90, 90)
        pts = [(xloc - 50, yloc - hw)] + arc + [(xloc - 50, yloc + hw)]
    else:
        return None
    return Polygon(pts)


def draw_endcap_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a cardinal endcap: rounded cap arc + 2 wall lines."""
    direction, hw = _ENDCAP_PARAMS[ch]
    if direction == 'N':
        clip_and_draw_arc(drawing, xloc, yloc, hw, 0, 180, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw, yloc, xloc - hw, yloc - 50, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw, yloc, xloc + hw, yloc - 50, occlusion_poly, sw)
    elif direction == 'S':
        clip_and_draw_arc(drawing, xloc, yloc, hw, 180, 360, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw, yloc, xloc - hw, yloc + 50, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw, yloc, xloc + hw, yloc + 50, occlusion_poly, sw)
    elif direction == 'E':
        clip_and_draw_arc(drawing, xloc, yloc, hw, 270, 90, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc, yloc - hw, xloc + 50, yloc - hw, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc, yloc + hw, xloc + 50, yloc + hw, occlusion_poly, sw)
    elif direction == 'W':
        clip_and_draw_arc(drawing, xloc, yloc, hw, -90, 90, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc, yloc - hw, xloc - 50, yloc - hw, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc, yloc + hw, xloc - 50, yloc + hw, occlusion_poly, sw)


def draw_endcap_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                     pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a cardinal endcap (half-tube shading)."""
    direction, hw = _ENDCAP_PARAMS[ch]

    # Tangent points from center toward the open edge
    if direction == 'N':
        tangent = (0, -1)
        normal_left, normal_right = (-1, 0), (1, 0)
    elif direction == 'S':
        tangent = (0, 1)
        normal_left, normal_right = (-1, 0), (1, 0)
    elif direction == 'E':
        tangent = (1, 0)
        normal_left, normal_right = (0, -1), (0, 1)
    else:  # W
        tangent = (-1, 0)
        normal_left, normal_right = (0, -1), (0, 1)

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right
    band_center_dist = hw - band_offset - band_width / 2

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Half-cell: 50 units from center to edge, hatches along 0..50 range
    num_hatches = max(1, int(50 / spacing))
    for base_angle in angles_to_draw:
        for i in range(num_hatches + 1):
            t = i * (50.0 / num_hatches)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
            base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


# Vanishing pipe params: char -> (direction, half_width)
# Tapered dead-end: full width at port edge, tapering to a point at cell center.
_VANISHING_PARAMS = {
    'vN': ('N', 30),  'vS': ('S', 30),  'vE': ('E', 30),  'vW': ('W', 30),
    'nvN': ('N', 12), 'nvS': ('S', 12), 'nvE': ('E', 12), 'nvW': ('W', 12),
    'tvN': ('N', 5),  'tvS': ('S', 5),  'tvE': ('E', 5),  'tvW': ('W', 5),
}


def get_vanishing_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a vanishing pipe (tapered dead-end triangle)."""
    direction, hw = _VANISHING_PARAMS[ch]
    if direction == 'N':
        pts = [(xloc - hw, yloc - 50), (xloc + hw, yloc - 50), (xloc, yloc)]
    elif direction == 'S':
        pts = [(xloc - hw, yloc + 50), (xloc + hw, yloc + 50), (xloc, yloc)]
    elif direction == 'E':
        pts = [(xloc + 50, yloc - hw), (xloc + 50, yloc + hw), (xloc, yloc)]
    elif direction == 'W':
        pts = [(xloc - 50, yloc - hw), (xloc - 50, yloc + hw), (xloc, yloc)]
    else:
        return None
    return Polygon(pts)


def draw_vanishing_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a vanishing pipe: 2 taper lines + port opening line."""
    direction, hw = _VANISHING_PARAMS[ch]
    if direction == 'N':
        clip_and_draw_line(drawing, xloc - hw, yloc - 50, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw, yloc - 50, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw, yloc - 50, xloc + hw, yloc - 50, occlusion_poly, sw)
    elif direction == 'S':
        clip_and_draw_line(drawing, xloc - hw, yloc + 50, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw, yloc + 50, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw, yloc + 50, xloc + hw, yloc + 50, occlusion_poly, sw)
    elif direction == 'E':
        clip_and_draw_line(drawing, xloc + 50, yloc - hw, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + 50, yloc + hw, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + 50, yloc - hw, xloc + 50, yloc + hw, occlusion_poly, sw)
    elif direction == 'W':
        clip_and_draw_line(drawing, xloc - 50, yloc - hw, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - 50, yloc + hw, xloc, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - 50, yloc - hw, xloc - 50, yloc + hw, occlusion_poly, sw)


def draw_vanishing_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                        pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a vanishing pipe (tapered dead-end).

    Uses the same approach as endcap shading but polygon-clipped to the
    triangular taper shape, so hatches naturally shorten toward the tip.
    """
    direction, hw = _VANISHING_PARAMS[ch]

    if direction == 'N':
        tangent = (0, -1)
        normal_left, normal_right = (-1, 0), (1, 0)
    elif direction == 'S':
        tangent = (0, 1)
        normal_left, normal_right = (-1, 0), (1, 0)
    elif direction == 'E':
        tangent = (1, 0)
        normal_left, normal_right = (0, -1), (0, 1)
    else:  # W
        tangent = (-1, 0)
        normal_left, normal_right = (0, -1), (0, 1)

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right
    band_center_dist = hw - band_offset - band_width / 2

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Full cell length: 50 units from center to edge
    num_hatches = max(1, int(50 / spacing))
    for base_angle in angles_to_draw:
        for i in range(num_hatches + 1):
            t = i * (50.0 / num_hatches)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
            base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


# Mixed-size corner params: char -> (rotation_deg, hw_arm1, hw_arm2)
# Base orientation (rot=0): arm1 goes South, arm2 goes East (like 'r' corner)
# rot=90: arm1=S,arm2=W (like '7'), rot=180: arm1=N,arm2=W (like 'j'), rot=270: arm1=N,arm2=E (like 'L')
_MIXED_CORNER_PARAMS = {
    'mnr': (0, 30, 12),   'mn7': (90, 30, 12),
    'mnj': (180, 30, 12), 'mnL': (270, 30, 12),
    'nmr': (0, 12, 30),   'nm7': (90, 12, 30),
    'nmj': (180, 12, 30), 'nmL': (270, 12, 30),
}


def get_mixed_corner_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a mixed-size corner (union of center box + 2 arms)."""
    rot_deg, hw1, hw2 = _MIXED_CORNER_PARAMS[ch]
    hw_max = max(hw1, hw2)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Center box sized to the larger pipe
    center = Polygon([tp(-hw_max, -hw_max), tp(hw_max, -hw_max),
                      tp(hw_max, hw_max), tp(-hw_max, hw_max)])
    # Arm 1 (south in base orientation): extends from center to cell edge
    arm1 = Polygon([tp(-hw1, hw_max), tp(hw1, hw_max),
                    tp(hw1, 50), tp(-hw1, 50)])
    # Arm 2 (east in base orientation): extends from center to cell edge
    arm2 = Polygon([tp(hw_max, -hw2), tp(50, -hw2),
                    tp(50, hw2), tp(hw_max, hw2)])

    poly = unary_union([center, arm1, arm2])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def draw_mixed_corner_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a mixed-size corner with explicit walls and cap lines.

    Base r orientation: arm1=South (hw1), arm2=East (hw2).
    Traces the L-shaped perimeter with clean cap lines at size transitions.
    """
    rot_deg, hw1, hw2 = _MIXED_CORNER_PARAMS[ch]
    hw_max = max(hw1, hw2)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    def line(ax, ay, bx, by):
        p1 = tp(ax, ay)
        p2 = tp(bx, by)
        clip_and_draw_line(drawing, p1[0], p1[1], p2[0], p2[1], occlusion_poly, sw)

    # Build perimeter vertices clockwise (base r orientation: S port + E port).
    # The vertex path differs depending on which arm is wider.
    if hw1 >= hw2:
        # Arm1 (south) is wider — step/cap on arm2 (east) side
        verts = [
            (hw1, 50),          # 0: south port, right
            (hw1, hw2),         # 1: right wall → arm2 bottom cap
            (50, hw2),          # 2: arm2 bottom → cell edge
            (50, -hw2),         # 3: east port top [SKIP 2→3]
            (hw1, -hw2),        # 4: arm2 top cap → right wall
            (hw1, -hw1),        # 5: right wall → top-right corner
            (-hw1, -hw1),       # 6: top edge → top-left corner
            (-hw1, 50),         # 7: left wall → south port [SKIP 7→0]
        ]
    else:
        # Arm2 (east) is wider — step/cap on arm1 (south) side
        verts = [
            (hw1, 50),          # 0: south port, right
            (hw1, hw2),         # 1: arm1 right cap → center bottom-right
            (50, hw2),          # 2: center bottom → cell edge
            (50, -hw2),         # 3: east port top [SKIP 2→3]
            (-hw2, -hw2),       # 4: top-left corner
            (-hw2, hw2),        # 5: left wall → center bottom-left
            (-hw1, hw2),        # 6: arm1 left cap
            (-hw1, 50),         # 7: arm1 left → south port [SKIP 7→0]
        ]

    # Draw all segments except the two open ports (south: 7→0, east: 2→3)
    for i in range(len(verts)):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % len(verts)]
        if i == len(verts) - 1:  # south port
            continue
        if i == 2:  # east port
            continue
        line(ax, ay, bx, by)


def draw_mixed_corner_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                           pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a mixed-size corner (shade each arm).

    Uses per-arm clip polygons so hatches don't bleed across the size step.
    The wider arm shades through the center box; the narrower arm is clipped
    to just its arm region beyond the center box boundary.
    """
    rot_deg, hw1, hw2 = _MIXED_CORNER_PARAMS[ch]

    def rot_dir(dx, dy):
        return _rotate_vec((dx, dy), rot_deg)

    def to_world(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    def make_arm_clip(local_pts):
        """Create a clip polygon from local-coord rectangle, intersected with pipe."""
        world_pts = [to_world(px, py) for px, py in local_pts]
        clip = Polygon(world_pts)
        if pipe_polygon is not None:
            clip = pipe_polygon.intersection(clip)
        return clip if (clip is not None and not clip.is_empty) else pipe_polygon

    # Per-arm clip regions: wider arm extends through center box,
    # narrower arm is limited to beyond the center box boundary.
    if hw1 >= hw2:
        # Arm1 (south) is wider — shades center box + south arm
        arm1_clip = make_arm_clip([(-hw1, -hw1), (hw1, -hw1), (hw1, 50), (-hw1, 50)])
        # Arm2 (east) is narrower — only beyond center box right edge
        arm2_clip = make_arm_clip([(hw1, -hw2), (50, -hw2), (50, hw2), (hw1, hw2)])
    else:
        # Arm2 (east) is wider — shades center box + east arm
        arm2_clip = make_arm_clip([(-hw2, -hw2), (50, -hw2), (50, hw2), (-hw2, hw2)])
        # Arm1 (south) is narrower — only beyond center box bottom edge
        arm1_clip = make_arm_clip([(-hw1, hw2), (hw1, hw2), (hw1, 50), (-hw1, 50)])

    # Shade arm 1 (south in base orientation, hw=hw1)
    _shade_corner_arm(drawing, to_world(0, 25), rot_dir(0, 1),
                      rot_dir(-1, 0), rot_dir(1, 0), hw1,
                      light_dir, sw, params, arm1_clip, occlusion_polygon)
    # Shade arm 2 (east in base orientation, hw=hw2)
    _shade_corner_arm(drawing, to_world(25, 0), rot_dir(1, 0),
                      rot_dir(0, -1), rot_dir(0, 1), hw2,
                      light_dir, sw, params, arm2_clip, occlusion_polygon)


def _shade_corner_arm(drawing, center, tangent, normal_left, normal_right, hw,
                      light_dir, sw, params, pipe_polygon, occlusion_polygon):
    """Shade one arm of a mixed-size corner (half-tube along arm)."""
    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right
    band_center_dist = hw - band_offset - band_width / 2

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    num_hatches = max(1, int(50 / spacing))
    cx, cy = center
    for base_angle in angles_to_draw:
        for i in range(num_hatches + 1):
            t = -25 + i * (50.0 / num_hatches)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = cx + tangent[0] * t + shadow_normal[0] * band_center_dist
            base_y = cy + tangent[1] * t + shadow_normal[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


# Dodge tile params: (orientation, sign, half_width)
_DODGE_PARAMS = {
    'z>':  ('v', +1, 30), 'z<':  ('v', -1, 30),
    'z^':  ('h', -1, 30), 'zv':  ('h', +1, 30),
    'nz>': ('v', +1, 12), 'nz<': ('v', -1, 12),
    'nz^': ('h', -1, 12), 'nzv': ('h', +1, 12),
    'tz>': ('v', +1, 5),  'tz<': ('v', -1, 5),
    'tz^': ('h', -1, 5),  'tzv': ('h', +1, 5),
}


def get_dodge_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a dodge/zigzag tile.

    The pipe follows a Z-shaped path through the cell, shifting laterally
    by dodge_offset at the midpoint, then returning to center at the exit.
    """
    if ch not in _DODGE_PARAMS:
        return None

    orient, sign, hw = _DODGE_PARAMS[ch]
    offset = sign * hw / 2

    if orient == 'v':
        return Polygon([
            (xloc + hw, yloc - 50),
            (xloc + hw + offset, yloc),
            (xloc + hw, yloc + 50),
            (xloc - hw, yloc + 50),
            (xloc - hw + offset, yloc),
            (xloc - hw, yloc - 50),
        ])
    else:
        return Polygon([
            (xloc - 50, yloc - hw),
            (xloc, yloc - hw + offset),
            (xloc + 50, yloc - hw),
            (xloc + 50, yloc + hw),
            (xloc, yloc + hw + offset),
            (xloc - 50, yloc + hw),
        ])


# S-Bend params: (orientation, sign, half_width) — same structure as dodge
_SBEND_PARAMS = {
    's>':  ('v', +1, 30), 's<':  ('v', -1, 30),
    's^':  ('h', -1, 30), 'sv':  ('h', +1, 30),
    'ns>': ('v', +1, 12), 'ns<': ('v', -1, 12),
    'ns^': ('h', -1, 12), 'nsv': ('h', +1, 12),
    'ts>': ('v', +1, 5),  'ts<': ('v', -1, 5),
    'ts^': ('h', -1, 5),  'tsv': ('h', +1, 5),
}


def get_sbend_polygon(ch, xloc, yloc, num_points=16):
    """Return Shapely Polygon for an S-bend tile.

    Smooth S-curve connecting opposite ports, with lateral shift at midpoint.
    Same ports as a straight pipe but with a sinusoidal wall path.
    """
    if ch not in _SBEND_PARAMS:
        return None

    orient, sign, hw = _SBEND_PARAMS[ch]
    offset = sign * hw / 2

    points = []
    if orient == 'v':
        # Right wall: top to bottom
        for i in range(num_points + 1):
            t = i / num_points
            y = yloc - 50 + 100 * t
            x = xloc + hw + offset * math.sin(math.pi * t)
            points.append((x, y))
        # Left wall: bottom to top
        for i in range(num_points + 1):
            t = 1 - i / num_points
            y = yloc - 50 + 100 * t
            x = xloc - hw + offset * math.sin(math.pi * t)
            points.append((x, y))
    else:
        # Top wall: left to right
        for i in range(num_points + 1):
            t = i / num_points
            x = xloc - 50 + 100 * t
            y = yloc - hw + offset * math.sin(math.pi * t)
            points.append((x, y))
        # Bottom wall: right to left
        for i in range(num_points + 1):
            t = 1 - i / num_points
            x = xloc - 50 + 100 * t
            y = yloc + hw + offset * math.sin(math.pi * t)
            points.append((x, y))

    return Polygon(points)


# Segmented elbow params: (base_corner, half_width) — faceted 3-segment corners
_SEGMENTED_PARAMS = {
    'gr': ('r', 30), 'g7': ('7', 30), 'gj': ('j', 30), 'gL': ('L', 30),
    'ngr': ('r', 12), 'ng7': ('7', 12), 'ngj': ('j', 12), 'ngL': ('L', 12),
    'tgr': ('r', 5), 'tg7': ('7', 5), 'tgj': ('j', 5), 'tgL': ('L', 5),
}


# Chamfered corner params: (base_corner, half_width)
_CHAMFER_PARAMS = {
    'cr': ('r', 30), 'c7': ('7', 30), 'cj': ('j', 30), 'cL': ('L', 30),
    'ncr': ('r', 12), 'nc7': ('7', 12), 'ncj': ('j', 12), 'ncL': ('L', 12),
    'tcr': ('r', 5), 'tc7': ('7', 5), 'tcj': ('j', 5), 'tcL': ('L', 5),
}

_CORNER_ROTATIONS = {'r': 0, '7': 90, 'j': 180, 'L': 270}


def get_chamfer_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a chamfered (45-degree) corner tile.

    Three straight sections (vertical, 45-degree diagonal, horizontal) instead
    of a smooth arc. The pipe maintains constant width throughout.
    """
    if ch not in _CHAMFER_PARAMS:
        return None

    base, hw = _CHAMFER_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]
    s2 = hw * math.sqrt(2)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Base r orientation (S+E ports), 7 vertices
    # Outer wall has 3-segment chamfer; inner wall is a simple 90° corner
    points = [
        tp(-hw, 50),          # south edge, inner wall
        tp(hw, 50),           # south edge, outer wall
        tp(hw, s2 - hw),      # outer: vert->diag transition
        tp(s2 - hw, hw),      # outer: diag->horiz transition
        tp(50, hw),           # east edge, outer wall
        tp(50, -hw),          # east edge, inner wall
        tp(-hw, -hw),         # inner corner (single point, 90° turn)
    ]

    poly = Polygon(points)
    # The outer chamfer's vertical and horizontal wall segments inevitably
    # touch at (hw, hw) for non-adjacent edges.  buffer(0) resolves this.
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def get_segmented_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for a segmented (faceted) elbow tile.

    Same arc center/radii as the standard corner but with only 3 straight
    segments per wall instead of a smooth arc. Creates a faceted look.
    """
    if ch not in _SEGMENTED_PARAMS:
        return None

    base, hw = _SEGMENTED_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]

    # Same radii as standard sized corner
    inner_r = hw / 3
    center_off = hw + inner_r
    outer_r = 2 * hw + inner_r

    local_center = (center_off, center_off)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg
    num_segments = 3

    # Build faceted arc polygon
    points = []
    for i in range(num_segments + 1):
        frac = i / num_segments
        angle_rad = math.radians(arc_start + frac * (arc_end - arc_start))
        points.append((arc_cx + outer_r * math.cos(angle_rad),
                       arc_cy + outer_r * math.sin(angle_rad)))

    for i in range(num_segments, -1, -1):
        frac = i / num_segments
        angle_rad = math.radians(arc_start + frac * (arc_end - arc_start))
        points.append((arc_cx + inner_r * math.cos(angle_rad),
                       arc_cy + inner_r * math.sin(angle_rad)))

    arc_polygon = Polygon(points)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    stub_extent = hw
    v_stub = Polygon([tp(center_off, -stub_extent), tp(50, -stub_extent),
                      tp(50, stub_extent), tp(center_off, stub_extent)])
    h_stub = Polygon([tp(-stub_extent, center_off), tp(stub_extent, center_off),
                      tp(stub_extent, 50), tp(-stub_extent, 50)])

    return unary_union([arc_polygon, v_stub, h_stub])


# Long-radius elbow params: (base_corner, half_width)
_LONG_RADIUS_PARAMS = {
    'lr': ('r', 30), 'l7': ('7', 30), 'lj': ('j', 30), 'lL': ('L', 30),
    'nlr': ('r', 12), 'nl7': ('7', 12), 'nlj': ('j', 12), 'nlL': ('L', 12),
    'tlr': ('r', 5), 'tl7': ('7', 5), 'tlj': ('j', 5), 'tlL': ('L', 5),
}


def _long_radius_arc_params(hw):
    """Return (inner_r, center_off, outer_r) for a long-radius elbow."""
    center_off = 50              # Arc center at cell corner for maximum sweep
    inner_r = center_off - hw   # medium: 20, narrow: 38, tiny: 45
    outer_r = center_off + hw   # medium: 80, narrow: 62, tiny: 55
    return inner_r, center_off, outer_r


def get_long_radius_polygon(ch, xloc, yloc, num_arc_points=16):
    """Return Shapely Polygon for a long-radius (wide arc) elbow tile.

    Same structure as standard corner but with maximized arc radius,
    giving a sweeping curve that fills more of the cell.
    """
    if ch not in _LONG_RADIUS_PARAMS:
        return None

    base, hw = _LONG_RADIUS_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]
    inner_r, center_off, outer_r = _long_radius_arc_params(hw)

    local_center = (center_off, center_off)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Build arc polygon
    points = []
    for i in range(num_arc_points + 1):
        frac = i / num_arc_points
        angle_rad = math.radians(arc_start + frac * (arc_end - arc_start))
        points.append((arc_cx + outer_r * math.cos(angle_rad),
                       arc_cy + outer_r * math.sin(angle_rad)))

    for i in range(num_arc_points, -1, -1):
        frac = i / num_arc_points
        angle_rad = math.radians(arc_start + frac * (arc_end - arc_start))
        points.append((arc_cx + inner_r * math.cos(angle_rad),
                       arc_cy + inner_r * math.sin(angle_rad)))

    arc_polygon = Polygon(points)

    # With center_off=50, arc endpoints land exactly at cell edges — no stubs needed
    if center_off >= 50:
        return arc_polygon

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Stubs connecting arc to cell edges (needed when center_off < 50)
    stub_extent = hw
    v_stub = Polygon([tp(center_off, -stub_extent), tp(50, -stub_extent),
                      tp(50, stub_extent), tp(center_off, stub_extent)])
    h_stub = Polygon([tp(-stub_extent, center_off), tp(stub_extent, center_off),
                      tp(stub_extent, 50), tp(-stub_extent, 50)])

    return unary_union([arc_polygon, v_stub, h_stub])


# Teardrop elbow params: (base_corner, half_width)
_TEARDROP_PARAMS = {
    'dr': ('r', 30), 'd7': ('7', 30), 'dj': ('j', 30), 'dL': ('L', 30),
    'ndr': ('r', 12), 'nd7': ('7', 12), 'ndj': ('j', 12), 'ndL': ('L', 12),
    'tdr': ('r', 5), 'td7': ('7', 5), 'tdj': ('j', 5), 'tdL': ('L', 5),
}


def get_teardrop_polygon(ch, xloc, yloc, num_arc_points=16):
    """Return Shapely Polygon for a teardrop elbow tile.

    Inner and outer arcs have different centers, creating an asymmetric profile:
    tight inner curve and bulbous outer curve.
    """
    if ch not in _TEARDROP_PARAMS:
        return None

    base, hw = _TEARDROP_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]

    # Dual-center arc parameters
    c_i = hw * 1.5    # Inner arc center offset
    r_i = c_i - hw    # Inner arc radius (tight: 0.5 * hw)
    c_o = hw * 1.0    # Outer arc center offset
    r_o = c_o + hw    # Outer arc radius (bulbous: 2.0 * hw)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Arc angles in world coords
    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Centers in world coords
    ic = tp(c_i, c_i)
    oc = tp(c_o, c_o)

    # Build polygon by tracing perimeter
    points = []

    # Inner wall: south stub -> inner arc -> east stub
    points.append(tp(hw, 50))       # South edge, inner wall
    points.append(tp(hw, c_i))      # Inner stub to arc start

    # Inner arc: 180° -> 270° (clockwise)
    for i in range(num_arc_points + 1):
        frac = i / num_arc_points
        angle_deg = arc_start + frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)
        x = ic[0] + r_i * math.cos(angle_rad)
        y = ic[1] + r_i * math.sin(angle_rad)
        points.append((x, y))

    points.append(tp(50, hw))       # Inner stub to east edge

    # East cell edge
    points.append(tp(50, -hw))      # East edge, outer wall

    # Outer wall: east stub -> outer arc -> south stub
    points.append(tp(c_o, -hw))     # Outer stub to arc start

    # Outer arc: 270° -> 180° (counterclockwise)
    for i in range(num_arc_points + 1):
        frac = i / num_arc_points
        angle_deg = arc_end - frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)
        x = oc[0] + r_o * math.cos(angle_rad)
        y = oc[1] + r_o * math.sin(angle_rad)
        points.append((x, y))

    points.append(tp(-hw, 50))      # Outer stub to south edge

    return Polygon(points)


def get_corner_polygon(ch, xloc, yloc, num_arc_points=16):
    """Return Shapely Polygon for a corner arc segment including connector stubs."""
    rotations = {'r': 0, '7': 90, 'j': 180, 'L': 270}
    rot_deg = rotations.get(ch)
    if rot_deg is None:
        return None

    # Arc center in world coords
    local_center = (40, 40)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    r_inner = 10
    r_outer = 70
    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Build arc polygon: trace outer arc, then inner arc backwards
    arc_points = []

    # Outer arc (forward)
    for i in range(num_arc_points + 1):
        frac = i / num_arc_points
        angle_deg = arc_start + frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)
        x = arc_cx + r_outer * math.cos(angle_rad)
        y = arc_cy + r_outer * math.sin(angle_rad)
        arc_points.append((x, y))

    # Inner arc (backwards)
    for i in range(num_arc_points, -1, -1):
        frac = i / num_arc_points
        angle_deg = arc_start + frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)
        x = arc_cx + r_inner * math.cos(angle_rad)
        y = arc_cy + r_inner * math.sin(angle_rad)
        arc_points.append((x, y))

    arc_polygon = Polygon(arc_points)

    # Helper to transform local coords to world coords
    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Vertical stub rectangle (local coords: x=40 to 50, y=-35 to 35)
    # Matches original Rectangle(40,-35,5,70)
    v_stub_points = [
        transform_point(40, -35),
        transform_point(50, -35),
        transform_point(50, 35),
        transform_point(40, 35),
    ]
    v_stub = Polygon(v_stub_points)

    # Horizontal stub rectangle (local coords: x=-35 to 35, y=40 to 50)
    # Matches original Rectangle(-35,40,70,5)
    h_stub_points = [
        transform_point(-35, 40),
        transform_point(35, 40),
        transform_point(35, 50),
        transform_point(-35, 50),
    ]
    h_stub = Polygon(h_stub_points)

    # Union all three polygons
    return unary_union([arc_polygon, v_stub, h_stub])


def get_cross_polygon(xloc, yloc, half_width=30):
    """Return Shapely Polygon for a 4-way cross junction (X).

    Shape: Central square with 4 rectangular arms extending to cell edges.
    """
    hw = half_width
    edge = 50
    # Central square (-30, -30) to (30, 30)
    center = Polygon([
        (xloc - hw, yloc - hw),
        (xloc + hw, yloc - hw),
        (xloc + hw, yloc + hw),
        (xloc - hw, yloc + hw),
    ])

    # North arm
    n_arm = Polygon([
        (xloc - hw, yloc - edge),
        (xloc + hw, yloc - edge),
        (xloc + hw, yloc - hw),
        (xloc - hw, yloc - hw),
    ])

    # South arm
    s_arm = Polygon([
        (xloc - hw, yloc + hw),
        (xloc + hw, yloc + hw),
        (xloc + hw, yloc + edge),
        (xloc - hw, yloc + edge),
    ])

    # East arm
    e_arm = Polygon([
        (xloc + hw, yloc - hw),
        (xloc + edge, yloc - hw),
        (xloc + edge, yloc + hw),
        (xloc + hw, yloc + hw),
    ])

    # West arm
    w_arm = Polygon([
        (xloc - edge, yloc - hw),
        (xloc - hw, yloc - hw),
        (xloc - hw, yloc + hw),
        (xloc - edge, yloc + hw),
    ])

    return unary_union([center, n_arm, s_arm, e_arm, w_arm])


def get_tee_polygon(ch, xloc, yloc, half_width=30):
    """Return Shapely Polygon for a 3-way tee junction.

    T = closed south, B = closed north, E = closed west, W = closed east.
    """
    rotations = {'T': 0, 'B': 180, 'E': 90, 'W': 270}
    rot_deg = rotations.get(ch)
    if rot_deg is None:
        return None

    hw = half_width
    edge = 50

    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Base shape (T orientation - closed south):
    # Central square + 3 arms (N, E, W) + flat bottom at y=+30

    # Central square
    center_pts = [
        transform_point(-hw, -hw),
        transform_point(hw, -hw),
        transform_point(hw, hw),
        transform_point(-hw, hw),
    ]
    center = Polygon(center_pts)

    # North arm (y = -30 to -50)
    n_arm_pts = [
        transform_point(-hw, -edge),
        transform_point(hw, -edge),
        transform_point(hw, -hw),
        transform_point(-hw, -hw),
    ]
    n_arm = Polygon(n_arm_pts)

    # East arm (x = 30 to 50)
    e_arm_pts = [
        transform_point(hw, -hw),
        transform_point(edge, -hw),
        transform_point(edge, hw),
        transform_point(hw, hw),
    ]
    e_arm = Polygon(e_arm_pts)

    # West arm (x = -50 to -30)
    w_arm_pts = [
        transform_point(-edge, -hw),
        transform_point(-hw, -hw),
        transform_point(-hw, hw),
        transform_point(-edge, hw),
    ]
    w_arm = Polygon(w_arm_pts)

    return unary_union([center, n_arm, e_arm, w_arm])


def get_pipe_polygon(ch, xloc, yloc):
    """Return Shapely Polygon for any pipe segment."""
    if ch == VOID_CHAR:
        return None

    # Standard tubes (medium, narrow, tiny)
    if ch in ('|', '-', 'i', '=', '!', '.'):
        return get_tube_polygon(ch, xloc, yloc)

    # Medium corners
    elif ch in ('r', '7', 'j', 'L'):
        return get_corner_polygon(ch, xloc, yloc)

    # Narrow corners
    elif ch in ('nr', 'n7', 'nj', 'nL'):
        return get_sized_corner_polygon(ch, xloc, yloc, 12)

    # Tiny corners
    elif ch in ('tr', 't7', 'tj', 'tL'):
        return get_sized_corner_polygon(ch, xloc, yloc, 5)

    # Reducers (medium↔narrow and narrow↔tiny)
    elif ch in ('Rv', 'RV', 'Tv', 'TV', 'Rh', 'RH', 'Th', 'TH'):
        return get_reducer_polygon(ch, xloc, yloc)

    # Dodge/zigzag tiles
    elif ch in _DODGE_PARAMS:
        return get_dodge_polygon(ch, xloc, yloc)

    # S-Bends
    elif ch in _SBEND_PARAMS:
        return get_sbend_polygon(ch, xloc, yloc)

    # Segmented elbows
    elif ch in _SEGMENTED_PARAMS:
        return get_segmented_polygon(ch, xloc, yloc)

    # Long-radius elbows
    elif ch in _LONG_RADIUS_PARAMS:
        return get_long_radius_polygon(ch, xloc, yloc)

    # Chamfered corners
    elif ch in _CHAMFER_PARAMS:
        return get_chamfer_polygon(ch, xloc, yloc)

    # Teardrop elbows
    elif ch in _TEARDROP_PARAMS:
        return get_teardrop_polygon(ch, xloc, yloc)

    # Diagonal straights
    elif ch in _DIAG_PARAMS:
        return get_diagonal_polygon(ch, xloc, yloc)

    # Diagonal adapters
    elif ch in _ADAPTER_PARAMS:
        return get_adapter_polygon(ch, xloc, yloc)

    # Diagonal corners and endcaps
    elif ch in _DIAG_CORNER_PARAMS:
        return get_diagonal_corner_polygon(ch, xloc, yloc)
    elif ch in _DIAG_ENDCAP_PARAMS:
        return get_diagonal_endcap_polygon(ch, xloc, yloc)

    # Cardinal endcaps
    elif ch in _ENDCAP_PARAMS:
        return get_endcap_polygon(ch, xloc, yloc)

    # Vanishing pipes
    elif ch in _VANISHING_PARAMS:
        return get_vanishing_polygon(ch, xloc, yloc)

    # Mixed-size corners
    elif ch in _MIXED_CORNER_PARAMS:
        return get_mixed_corner_polygon(ch, xloc, yloc)

    # Crossovers (all sizes)
    elif ch in CROSSOVER_TUBES:
        v_ch, h_ch = CROSSOVER_TUBES[ch]
        v = get_tube_polygon(v_ch, xloc, yloc)
        h = get_tube_polygon(h_ch, xloc, yloc)
        if v and h:
            return unary_union([v, h])
        return v or h

    # Diagonal crossovers (medium cardinal x narrow diagonal)
    elif ch in DIAG_CROSSOVER_TUBES:
        tube_ch, diag_ch = DIAG_CROSSOVER_TUBES[ch]
        tube_poly = get_tube_polygon(tube_ch, xloc, yloc)
        diag_poly = get_diagonal_polygon(diag_ch, xloc, yloc)
        parts = [p for p in [tube_poly, diag_poly] if p is not None]
        return unary_union(parts) if parts else None

    # Cross junctions (medium, narrow, tiny)
    elif ch in ('X', 'nX', 'tX'):
        hw = 30 if ch == 'X' else (12 if ch == 'nX' else 5)
        return get_cross_polygon(xloc, yloc, half_width=hw)

    # Tee junctions (medium, narrow, tiny)
    elif ch in ('T', 'B', 'E', 'W', 'nT', 'nB', 'nE', 'nW', 'tT', 'tB', 'tE', 'tW'):
        if ch in ('T', 'B', 'E', 'W'):
            hw = 30
            base_ch = ch
        elif ch.startswith('n'):
            hw = 12
            base_ch = ch[1:]
        else:
            hw = 5
            base_ch = ch[1:]
        return get_tee_polygon(base_ch, xloc, yloc, half_width=hw)

    return None


def build_occlusion_polygon(grid, exclude_x, exclude_y, pad=0):
    """Build a union polygon of all pipe segments EXCEPT the one at (exclude_x, exclude_y).

    This is used to clip strokes so they don't extend into other pipes.

    Args:
        grid: 2D list of pipe characters
        exclude_x, exclude_y: Position of pipe to exclude from occlusion
        pad: Buffer amount to account for stroke width (use max(stroke_width, shading_stroke_width) * 0.5 + 0.1)
    """
    width = len(grid)
    height = len(grid[0]) if width > 0 else 0

    polygons = []
    for x in range(width):
        for y in range(height):
            if x == exclude_x and y == exclude_y:
                continue
            ch = grid[x][y]
            # Center the grid properly (works for both odd and even dimensions)
            xloc = (x - (width - 1) / 2.0) * 100
            yloc = (y - (height - 1) / 2.0) * 100
            poly = get_pipe_polygon(ch, xloc, yloc)
            if poly is not None:
                poly = poly.buffer(0)  # Clean any invalid geometry
                if poly.is_valid and not poly.is_empty:
                    if pad > 0:
                        poly = poly.buffer(pad, join_style=2)
                    polygons.append(poly)

    if not polygons:
        return None
    return unary_union(polygons).buffer(0)  # Clean final geometry


def build_occlusion_polygon_cached(poly_cache, exclude_x, exclude_y, pad=0):
    """Build occlusion polygon using precomputed pipe polygons.

    Same as build_occlusion_polygon but uses a cache of already-constructed
    and cleaned polygons, avoiding redundant get_pipe_polygon + buffer(0) calls.

    Args:
        poly_cache: dict of (x, y) -> Shapely Polygon (precomputed, already buffered(0))
        exclude_x, exclude_y: Position to exclude
        pad: Stroke width padding
    """
    polygons = []
    for (x, y), poly in poly_cache.items():
        if x == exclude_x and y == exclude_y:
            continue
        if pad > 0:
            poly = poly.buffer(pad, join_style=2)
        polygons.append(poly)
    if not polygons:
        return None
    return unary_union(polygons).buffer(0)


def _build_full_layer_union(poly_cache, pad=0):
    """Union of ALL polygons in a layer's cache (for cross-layer occlusion)."""
    if not poly_cache:
        return None
    polygons = []
    for poly in poly_cache.values():
        if pad > 0:
            poly = poly.buffer(pad, join_style=2)
        polygons.append(poly)
    return unary_union(polygons).buffer(0)


def clip_line_to_polygon(x1, y1, x2, y2, polygon, exclusion=None):
    """Clip a line segment to stay within polygon and outside exclusion.

    Returns list of (x1, y1, x2, y2) tuples for visible line segments.
    """
    line = LineString([(x1, y1), (x2, y2)])

    # First intersect with the pipe polygon (stay inside)
    clipped = line.intersection(polygon)

    if clipped.is_empty:
        return []

    # If there's an exclusion polygon, subtract it
    if exclusion is not None:
        clipped = clipped.difference(exclusion)

    if clipped.is_empty:
        return []

    # Handle result (could be LineString, MultiLineString, or GeometryCollection)
    result = []
    if clipped.geom_type == 'LineString':
        coords = list(clipped.coords)
        if len(coords) >= 2:
            result.append((coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]))
    elif clipped.geom_type == 'MultiLineString':
        for geom in clipped.geoms:
            coords = list(geom.coords)
            if len(coords) >= 2:
                result.append((coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]))

    return result


def clip_line_outside_polygon(x1, y1, x2, y2, occlusion_poly):
    """Clip a line segment to stay OUTSIDE the occlusion polygon.

    Returns list of (x1, y1, x2, y2) tuples for visible line segments.
    """
    if occlusion_poly is None:
        return [(x1, y1, x2, y2)]

    line = LineString([(x1, y1), (x2, y2)])
    clipped = line.difference(occlusion_poly)

    if clipped.is_empty:
        return []

    result = []
    if clipped.geom_type == 'LineString':
        coords = list(clipped.coords)
        if len(coords) >= 2:
            # For multi-point lines, draw segments between consecutive points
            for i in range(len(coords) - 1):
                result.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
    elif clipped.geom_type == 'MultiLineString':
        for geom in clipped.geoms:
            coords = list(geom.coords)
            if len(coords) >= 2:
                for i in range(len(coords) - 1):
                    result.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))
    elif clipped.geom_type == 'GeometryCollection':
        for geom in clipped.geoms:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) >= 2:
                    for i in range(len(coords) - 1):
                        result.append((coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]))

    return result


def clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw):
    """Clip a line against occlusion polygon and draw visible parts."""
    segments = clip_line_outside_polygon(x1, y1, x2, y2, occlusion_poly)
    for sx1, sy1, sx2, sy2 in segments:
        drawing.append(draw.Line(sx1, sy1, sx2, sy2,
                                 stroke='black', stroke_width=sw, fill='none'))


def clip_and_draw_arc(drawing, cx, cy, r, start_deg, end_deg, occlusion_poly, sw, num_segments=32):
    """Approximate arc as polyline segments and clip each against occlusion polygon."""
    # Generate arc points
    points = []
    for i in range(num_segments + 1):
        frac = i / num_segments
        angle_deg = start_deg + frac * (end_deg - start_deg)
        angle_rad = math.radians(angle_deg)
        x = cx + r * math.cos(angle_rad)
        y = cy + r * math.sin(angle_rad)
        points.append((x, y))

    # Draw each segment with clipping
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def clip_and_draw_line_inside(drawing, x1, y1, x2, y2, pipe_poly, occlusion_poly, sw):
    """Clip a line to stay inside pipe_poly and outside occlusion_poly."""
    if pipe_poly is None:
        return
    segments = clip_line_to_polygon(x1, y1, x2, y2, pipe_poly, occlusion_poly)
    for cx1, cy1, cx2, cy2 in segments:
        drawing.append(draw.Line(cx1, cy1, cx2, cy2,
                                 stroke='black', stroke_width=sw, fill='none'))


def clip_and_draw_arc_inside(drawing, cx, cy, r, start_deg, end_deg,
                             pipe_poly, occlusion_poly, sw, num_segments=32):
    """Approximate arc and clip each segment inside pipe_poly, outside occlusion_poly."""
    points = []
    for i in range(num_segments + 1):
        frac = i / num_segments
        angle_deg = start_deg + frac * (end_deg - start_deg)
        angle_rad = math.radians(angle_deg)
        x = cx + r * math.cos(angle_rad)
        y = cy + r * math.sin(angle_rad)
        points.append((x, y))

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        clip_and_draw_line_inside(drawing, x1, y1, x2, y2,
                                  pipe_poly, occlusion_poly, sw)


# ============================================================================
# CLIPPED PIPE OUTLINE DRAWING
# ============================================================================

def draw_tube_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw a straight tube outline with clipping against other pipes."""
    # Determine half-width based on tube type
    tube_params = {
        '|': (30, 'v'),   # Medium vertical
        '-': (30, 'h'),   # Medium horizontal
        'i': (12, 'v'),   # Narrow vertical
        '=': (12, 'h'),   # Narrow horizontal
        '!': (5, 'v'),    # Tiny vertical
        '.': (5, 'h'),    # Tiny horizontal
    }

    if ch not in tube_params:
        return

    hw, orient = tube_params[ch]

    if orient == 'v':
        # Vertical tube: two vertical lines
        clip_and_draw_line(drawing, xloc - hw, yloc - 50, xloc - hw, yloc + 50, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw, yloc - 50, xloc + hw, yloc + 50, occlusion_poly, sw)
    else:
        # Horizontal tube: two horizontal lines
        clip_and_draw_line(drawing, xloc - 50, yloc - hw, xloc + 50, yloc - hw, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - 50, yloc + hw, xloc + 50, yloc + hw, occlusion_poly, sw)


def draw_corner_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw a corner pipe outline with clipping against other pipes."""
    rotations = {'r': 0, '7': 90, 'j': 180, 'L': 270}
    rot_deg = rotations.get(ch)
    if rot_deg is None:
        return

    # Arc center in world coords
    local_center = (40, 40)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    # Arc angles
    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Draw outer arc (r=70)
    clip_and_draw_arc(drawing, arc_cx, arc_cy, 70, arc_start, arc_end, occlusion_poly, sw)

    # Draw inner arc (r=10)
    clip_and_draw_arc(drawing, arc_cx, arc_cy, 10, arc_start, arc_end, occlusion_poly, sw)

    # Draw the connector stubs - simple rectangles
    # Vertical stub: Rectangle(40,-35,5,70) = x:40-45, y:-35 to 35
    # Horizontal stub: Rectangle(-35,40,70,5) = x:-35 to 35, y:40-45

    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Vertical stub rectangle (4 sides)
    # Left side
    x1, y1 = transform_point(40, -35)
    x2, y2 = transform_point(40, 35)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    # Right side
    x1, y1 = transform_point(45, -35)
    x2, y2 = transform_point(45, 35)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    # Top side
    x1, y1 = transform_point(40, -35)
    x2, y2 = transform_point(45, -35)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    # Bottom side
    x1, y1 = transform_point(40, 35)
    x2, y2 = transform_point(45, 35)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Connecting lines from vertical stub to tile edge (at y=±30)
    x1, y1 = transform_point(45, 30)
    x2, y2 = transform_point(50, 30)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    x1, y1 = transform_point(45, -30)
    x2, y2 = transform_point(50, -30)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Horizontal stub rectangle (4 sides)
    # Top side
    x1, y1 = transform_point(-35, 40)
    x2, y2 = transform_point(35, 40)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    # Bottom side
    x1, y1 = transform_point(-35, 45)
    x2, y2 = transform_point(35, 45)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    # Left side
    x1, y1 = transform_point(-35, 40)
    x2, y2 = transform_point(-35, 45)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    # Right side
    x1, y1 = transform_point(35, 40)
    x2, y2 = transform_point(35, 45)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Connecting lines from horizontal stub to tile edge (at x=±30)
    x1, y1 = transform_point(30, 45)
    x2, y2 = transform_point(30, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
    x1, y1 = transform_point(-30, 45)
    x2, y2 = transform_point(-30, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_cross_outline(drawing, xloc, yloc, occlusion_poly, sw, half_width=30):
    """Draw outline for a 4-way cross junction (X).

    Shape: Central square with 4 arms. Draw continuous perimeter.
    """
    hw = half_width
    edge = 50
    # Draw the cross perimeter as a series of connected line segments
    # Starting from top-left of north arm, going clockwise

    # North arm - left side (top to inner corner)
    clip_and_draw_line(drawing, xloc - hw, yloc - edge, xloc - hw, yloc - hw, occlusion_poly, sw)
    # West arm - top side (inner corner to outer)
    clip_and_draw_line(drawing, xloc - hw, yloc - hw, xloc - edge, yloc - hw, occlusion_poly, sw)
    # West arm - left side
    clip_and_draw_line(drawing, xloc - edge, yloc - hw, xloc - edge, yloc + hw, occlusion_poly, sw)
    # West arm - bottom side (outer to inner corner)
    clip_and_draw_line(drawing, xloc - edge, yloc + hw, xloc - hw, yloc + hw, occlusion_poly, sw)
    # South arm - left side (inner corner to bottom)
    clip_and_draw_line(drawing, xloc - hw, yloc + hw, xloc - hw, yloc + edge, occlusion_poly, sw)
    # South arm - bottom side
    clip_and_draw_line(drawing, xloc - hw, yloc + edge, xloc + hw, yloc + edge, occlusion_poly, sw)
    # South arm - right side (bottom to inner corner)
    clip_and_draw_line(drawing, xloc + hw, yloc + edge, xloc + hw, yloc + hw, occlusion_poly, sw)
    # East arm - bottom side (inner corner to outer)
    clip_and_draw_line(drawing, xloc + hw, yloc + hw, xloc + edge, yloc + hw, occlusion_poly, sw)
    # East arm - right side
    clip_and_draw_line(drawing, xloc + edge, yloc + hw, xloc + edge, yloc - hw, occlusion_poly, sw)
    # East arm - top side (outer to inner corner)
    clip_and_draw_line(drawing, xloc + edge, yloc - hw, xloc + hw, yloc - hw, occlusion_poly, sw)
    # North arm - right side (inner corner to top)
    clip_and_draw_line(drawing, xloc + hw, yloc - hw, xloc + hw, yloc - edge, occlusion_poly, sw)
    # North arm - top side
    clip_and_draw_line(drawing, xloc + hw, yloc - edge, xloc - hw, yloc - edge, occlusion_poly, sw)


def draw_tee_outline(drawing, ch, xloc, yloc, occlusion_poly, sw, half_width=30):
    """Draw outline for a 3-way tee junction.

    T = closed south, B = closed north, E = closed west, W = closed east.
    """
    rotations = {'T': 0, 'B': 180, 'E': 90, 'W': 270}
    rot_deg = rotations.get(ch)
    if rot_deg is None:
        return

    hw = half_width
    edge = 50

    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Base shape (T orientation - closed south):
    # Draw perimeter clockwise starting from top-left of north arm

    # North arm - left side (top to inner corner)
    x1, y1 = transform_point(-hw, -edge)
    x2, y2 = transform_point(-hw, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # West arm - top side (inner corner to outer)
    x1, y1 = transform_point(-hw, -hw)
    x2, y2 = transform_point(-edge, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # West arm - left side
    x1, y1 = transform_point(-edge, -hw)
    x2, y2 = transform_point(-edge, hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Bottom (closed side) - flat line from west to east
    x1, y1 = transform_point(-edge, hw)
    x2, y2 = transform_point(edge, hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # East arm - right side
    x1, y1 = transform_point(edge, hw)
    x2, y2 = transform_point(edge, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # East arm - top side (outer to inner corner)
    x1, y1 = transform_point(edge, -hw)
    x2, y2 = transform_point(hw, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # North arm - right side (inner corner to top)
    x1, y1 = transform_point(hw, -hw)
    x2, y2 = transform_point(hw, -edge)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # North arm - top side
    x1, y1 = transform_point(hw, -edge)
    x2, y2 = transform_point(-hw, -edge)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_sized_corner_outline(drawing, ch, xloc, yloc, half_width, occlusion_poly, sw):
    """Draw a corner outline with specified half-width."""
    base_chars = {'r': 0, '7': 90, 'j': 180, 'L': 270,
                  'nr': 0, 'n7': 90, 'nj': 180, 'nL': 270,
                  'tr': 0, 't7': 90, 'tj': 180, 'tL': 270}
    rot_deg = base_chars.get(ch)
    if rot_deg is None:
        return

    # Arc radii based on half-width - scale proportionally for all pipe sizes
    inner_radius = half_width / 3
    center_offset = half_width + inner_radius
    outer_radius = 2 * half_width + inner_radius

    local_center = (center_offset, center_offset)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Draw arcs
    clip_and_draw_arc(drawing, arc_cx, arc_cy, outer_radius, arc_start, arc_end, occlusion_poly, sw)
    clip_and_draw_arc(drawing, arc_cx, arc_cy, inner_radius, arc_start, arc_end, occlusion_poly, sw)

    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Stub: rectangular fitting that connects arc to cell edge
    # Match medium corner proportions: stub starts at arc center (center_offset),
    # giving inner_radius of gap between inner arc and stub
    stub_start = center_offset
    stub_end = min(center_offset + inner_radius * 0.5, 50)
    stub_extent = outer_radius / 2

    # Only draw stub rectangles if there's room (stub_start < stub_end)
    if stub_start < stub_end:
        # Vertical stub rectangle (4 sides)
        x1, y1 = transform_point(stub_start, -stub_extent)
        x2, y2 = transform_point(stub_start, stub_extent)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(stub_end, -stub_extent)
        x2, y2 = transform_point(stub_end, stub_extent)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(stub_start, -stub_extent)
        x2, y2 = transform_point(stub_end, -stub_extent)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(stub_start, stub_extent)
        x2, y2 = transform_point(stub_end, stub_extent)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        # Horizontal stub rectangle (4 sides)
        x1, y1 = transform_point(-stub_extent, stub_start)
        x2, y2 = transform_point(stub_extent, stub_start)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(-stub_extent, stub_end)
        x2, y2 = transform_point(stub_extent, stub_end)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(-stub_extent, stub_start)
        x2, y2 = transform_point(-stub_extent, stub_end)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(stub_extent, stub_start)
        x2, y2 = transform_point(stub_extent, stub_end)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Connecting lines from stub (or arc) to tile edge (at pipe wall position)
    connect_start = stub_end if stub_start < stub_end else half_width
    if connect_start < 50:
        x1, y1 = transform_point(connect_start, half_width)
        x2, y2 = transform_point(50, half_width)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(connect_start, -half_width)
        x2, y2 = transform_point(50, -half_width)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(half_width, connect_start)
        x2, y2 = transform_point(half_width, 50)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        x1, y1 = transform_point(-half_width, connect_start)
        x2, y2 = transform_point(-half_width, 50)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_dodge_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a dodge/zigzag tile (Z-shaped pipe path)."""
    if ch not in _DODGE_PARAMS:
        return

    orient, sign, hw = _DODGE_PARAMS[ch]
    offset = sign * hw / 2

    if orient == 'v':
        # Right wall: top -> mid -> bottom
        clip_and_draw_line(drawing, xloc + hw, yloc - 50,
                          xloc + hw + offset, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw + offset, yloc,
                          xloc + hw, yloc + 50, occlusion_poly, sw)
        # Left wall: top -> mid -> bottom
        clip_and_draw_line(drawing, xloc - hw, yloc - 50,
                          xloc - hw + offset, yloc, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw + offset, yloc,
                          xloc - hw, yloc + 50, occlusion_poly, sw)
    else:
        # Top wall: left -> mid -> right
        clip_and_draw_line(drawing, xloc - 50, yloc - hw,
                          xloc, yloc - hw + offset, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc, yloc - hw + offset,
                          xloc + 50, yloc - hw, occlusion_poly, sw)
        # Bottom wall: left -> mid -> right
        clip_and_draw_line(drawing, xloc - 50, yloc + hw,
                          xloc, yloc + hw + offset, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc, yloc + hw + offset,
                          xloc + 50, yloc + hw, occlusion_poly, sw)


def draw_sbend_outline(drawing, ch, xloc, yloc, occlusion_poly, sw, num_segments=16):
    """Draw outline for an S-bend tile (sinusoidal curved pipe)."""
    if ch not in _SBEND_PARAMS:
        return

    orient, sign, hw = _SBEND_PARAMS[ch]
    offset = sign * hw / 2

    if orient == 'v':
        for wall_x in [hw, -hw]:
            for i in range(num_segments):
                t0 = i / num_segments
                t1 = (i + 1) / num_segments
                x0 = xloc + wall_x + offset * math.sin(math.pi * t0)
                y0 = yloc - 50 + 100 * t0
                x1 = xloc + wall_x + offset * math.sin(math.pi * t1)
                y1 = yloc - 50 + 100 * t1
                clip_and_draw_line(drawing, x0, y0, x1, y1, occlusion_poly, sw)
    else:
        for wall_y in [-hw, hw]:
            for i in range(num_segments):
                t0 = i / num_segments
                t1 = (i + 1) / num_segments
                x0 = xloc - 50 + 100 * t0
                y0 = yloc + wall_y + offset * math.sin(math.pi * t0)
                x1 = xloc - 50 + 100 * t1
                y1 = yloc + wall_y + offset * math.sin(math.pi * t1)
                clip_and_draw_line(drawing, x0, y0, x1, y1, occlusion_poly, sw)


def draw_long_radius_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a long-radius (wide arc) elbow tile."""
    if ch not in _LONG_RADIUS_PARAMS:
        return

    base, hw = _LONG_RADIUS_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]
    inner_r, center_off, outer_r = _long_radius_arc_params(hw)

    local_center = (center_off, center_off)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Draw arcs
    clip_and_draw_arc(drawing, arc_cx, arc_cy, outer_r, arc_start, arc_end, occlusion_poly, sw)
    clip_and_draw_arc(drawing, arc_cx, arc_cy, inner_r, arc_start, arc_end, occlusion_poly, sw)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Stub walls connecting arc to cell edges
    stub_extent = hw
    if center_off < 50:
        # Vertical arm walls
        x1, y1 = tp(center_off, -stub_extent)
        x2, y2 = tp(50, -stub_extent)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
        x1, y1 = tp(center_off, stub_extent)
        x2, y2 = tp(50, stub_extent)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        # Horizontal arm walls
        x1, y1 = tp(-stub_extent, center_off)
        x2, y2 = tp(-stub_extent, 50)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
        x1, y1 = tp(stub_extent, center_off)
        x2, y2 = tp(stub_extent, 50)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_segmented_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a segmented (faceted) elbow — 3 straight segments per wall."""
    if ch not in _SEGMENTED_PARAMS:
        return

    base, hw = _SEGMENTED_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]

    inner_r = hw / 3
    center_off = hw + inner_r
    outer_r = 2 * hw + inner_r

    local_center = (center_off, center_off)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg
    num_segments = 3

    # Draw faceted arcs (straight line segments between sample points)
    for radius in [outer_r, inner_r]:
        pts = []
        for i in range(num_segments + 1):
            frac = i / num_segments
            angle_rad = math.radians(arc_start + frac * (arc_end - arc_start))
            pts.append((arc_cx + radius * math.cos(angle_rad),
                        arc_cy + radius * math.sin(angle_rad)))
        for i in range(len(pts) - 1):
            clip_and_draw_line(drawing, pts[i][0], pts[i][1],
                              pts[i+1][0], pts[i+1][1], occlusion_poly, sw)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Stub walls connecting arc endpoints to cell edges
    stub_extent = hw

    # Vertical arm walls (from arc endpoint to cell edge)
    x1, y1 = tp(center_off, -stub_extent)
    x2, y2 = tp(50, -stub_extent)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(center_off, stub_extent)
    x2, y2 = tp(50, stub_extent)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Horizontal arm walls
    x1, y1 = tp(-stub_extent, center_off)
    x2, y2 = tp(-stub_extent, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(stub_extent, center_off)
    x2, y2 = tp(stub_extent, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_chamfer_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a chamfered (45-degree) corner tile."""
    if ch not in _CHAMFER_PARAMS:
        return

    base, hw = _CHAMFER_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]
    s2 = hw * math.sqrt(2)

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Outer wall: vertical section -> diagonal -> horizontal section
    x1, y1 = tp(hw, 50)
    x2, y2 = tp(hw, s2 - hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(hw, s2 - hw)
    x2, y2 = tp(s2 - hw, hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(s2 - hw, hw)
    x2, y2 = tp(50, hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Inner wall: 90° corner (horizontal section -> vertical section)
    x1, y1 = tp(50, -hw)
    x2, y2 = tp(-hw, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(-hw, -hw)
    x2, y2 = tp(-hw, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_teardrop_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw outline for a teardrop elbow tile (dual-center arcs)."""
    if ch not in _TEARDROP_PARAMS:
        return

    base, hw = _TEARDROP_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]

    c_i = hw * 1.5
    r_i = c_i - hw    # 0.5 * hw
    c_o = hw * 1.0
    r_o = c_o + hw    # 2.0 * hw

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Arc centers in world coords
    ic = tp(c_i, c_i)
    oc = tp(c_o, c_o)

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Draw inner arc (tight)
    clip_and_draw_arc(drawing, ic[0], ic[1], r_i, arc_start, arc_end, occlusion_poly, sw)

    # Draw outer arc (bulbous)
    clip_and_draw_arc(drawing, oc[0], oc[1], r_o, arc_start, arc_end, occlusion_poly, sw)

    # Draw stubs connecting arcs to cell edges
    # Inner vertical stub: (hw, c_i) -> (hw, 50)
    x1, y1 = tp(hw, c_i)
    x2, y2 = tp(hw, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Outer vertical stub: (-hw, c_o) -> (-hw, 50)
    x1, y1 = tp(-hw, c_o)
    x2, y2 = tp(-hw, 50)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Inner horizontal stub: (c_i, hw) -> (50, hw)
    x1, y1 = tp(c_i, hw)
    x2, y2 = tp(50, hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    # Outer horizontal stub: (c_o, -hw) -> (50, -hw)
    x1, y1 = tp(c_o, -hw)
    x2, y2 = tp(50, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_reducer_outline(drawing, ch, xloc, yloc, occlusion_poly, sw):
    """Draw a reducer pipe outline (tapers between two sizes).

    Walls are self-clipped against own flange polygons so flanges visually
    cross the pipe walls. Flanges are drawn as full rectangles (rings).
    """
    reducers = {
        'Rv': ('m', 'n', 'v'),  # Medium top, narrow bottom
        'RV': ('n', 'm', 'v'),  # Narrow top, medium bottom
        'Tv': ('n', 't', 'v'),  # Narrow top, tiny bottom
        'TV': ('t', 'n', 'v'),  # Tiny top, narrow bottom
        'Rh': ('m', 'n', 'h'),  # Medium right, narrow left
        'RH': ('n', 'm', 'h'),  # Narrow right, medium left
        'Th': ('n', 't', 'h'),  # Narrow right, tiny left
        'TH': ('t', 'n', 'h'),  # Tiny right, narrow left
    }

    if ch not in reducers:
        return

    start_size, end_size, orient = reducers[ch]
    hw_start = PIPE_HALF_WIDTHS[start_size]
    hw_end = PIPE_HALF_WIDTHS[end_size]

    # Flange dimensions (must match get_reducer_polygon)
    flange_start = hw_start * 0.15
    flange_end = hw_end * 0.15
    flange_width = 2

    # Build flange polygons for self-clipping
    if orient == 'v':
        flange1 = Polygon([
            (xloc - hw_start - flange_start, yloc - 20 - flange_width),
            (xloc + hw_start + flange_start, yloc - 20 - flange_width),
            (xloc + hw_start + flange_start, yloc - 20 + flange_width),
            (xloc - hw_start - flange_start, yloc - 20 + flange_width),
        ])
        flange2 = Polygon([
            (xloc - hw_end - flange_end, yloc + 20 - flange_width),
            (xloc + hw_end + flange_end, yloc + 20 - flange_width),
            (xloc + hw_end + flange_end, yloc + 20 + flange_width),
            (xloc - hw_end - flange_end, yloc + 20 + flange_width),
        ])
    else:
        flange1 = Polygon([
            (xloc + 20 - flange_width, yloc - hw_start - flange_start),
            (xloc + 20 + flange_width, yloc - hw_start - flange_start),
            (xloc + 20 + flange_width, yloc + hw_start + flange_start),
            (xloc + 20 - flange_width, yloc + hw_start + flange_start),
        ])
        flange2 = Polygon([
            (xloc - 20 - flange_width, yloc - hw_end - flange_end),
            (xloc - 20 + flange_width, yloc - hw_end - flange_end),
            (xloc - 20 + flange_width, yloc + hw_end + flange_end),
            (xloc - 20 - flange_width, yloc + hw_end + flange_end),
        ])

    flange_union = unary_union([flange1, flange2]).buffer(0)

    # Wall occlusion = other tiles + own flanges (so walls break at flanges)
    if occlusion_poly is not None:
        wall_occlusion = occlusion_poly.union(flange_union)
    else:
        wall_occlusion = flange_union

    if orient == 'v':  # Vertical reducer
        # Pipe walls self-clipped against own flanges
        clip_and_draw_line(drawing, xloc - hw_start, yloc - 50, xloc - hw_start, yloc - 20, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc - hw_start, yloc - 20, xloc - hw_end, yloc + 20, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc - hw_end, yloc + 20, xloc - hw_end, yloc + 50, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc + hw_start, yloc - 50, xloc + hw_start, yloc - 20, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc + hw_start, yloc - 20, xloc + hw_end, yloc + 20, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc + hw_end, yloc + 20, xloc + hw_end, yloc + 50, wall_occlusion, sw)

        # Top flange at y=-20: full rectangle (ring across pipe)
        # Top and bottom horizontals span full width
        clip_and_draw_line(drawing, xloc - hw_start - flange_start, yloc - 20 - flange_width,
                          xloc + hw_start + flange_start, yloc - 20 - flange_width, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw_start - flange_start, yloc - 20 + flange_width,
                          xloc + hw_start + flange_start, yloc - 20 + flange_width, occlusion_poly, sw)
        # Outer vertical edges
        clip_and_draw_line(drawing, xloc - hw_start - flange_start, yloc - 20 - flange_width,
                          xloc - hw_start - flange_start, yloc - 20 + flange_width, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw_start + flange_start, yloc - 20 - flange_width,
                          xloc + hw_start + flange_start, yloc - 20 + flange_width, occlusion_poly, sw)

        # Bottom flange at y=+20: full rectangle
        clip_and_draw_line(drawing, xloc - hw_end - flange_end, yloc + 20 - flange_width,
                          xloc + hw_end + flange_end, yloc + 20 - flange_width, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw_end - flange_end, yloc + 20 + flange_width,
                          xloc + hw_end + flange_end, yloc + 20 + flange_width, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - hw_end - flange_end, yloc + 20 - flange_width,
                          xloc - hw_end - flange_end, yloc + 20 + flange_width, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + hw_end + flange_end, yloc + 20 - flange_width,
                          xloc + hw_end + flange_end, yloc + 20 + flange_width, occlusion_poly, sw)

    else:  # Horizontal reducer
        # Pipe walls self-clipped against own flanges
        clip_and_draw_line(drawing, xloc + 50, yloc - hw_start, xloc + 20, yloc - hw_start, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc + 20, yloc - hw_start, xloc - 20, yloc - hw_end, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc - 20, yloc - hw_end, xloc - 50, yloc - hw_end, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc + 50, yloc + hw_start, xloc + 20, yloc + hw_start, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc + 20, yloc + hw_start, xloc - 20, yloc + hw_end, wall_occlusion, sw)
        clip_and_draw_line(drawing, xloc - 20, yloc + hw_end, xloc - 50, yloc + hw_end, wall_occlusion, sw)

        # Start flange at x=+20: full rectangle
        clip_and_draw_line(drawing, xloc + 20 - flange_width, yloc - hw_start - flange_start,
                          xloc + 20 - flange_width, yloc + hw_start + flange_start, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + 20 + flange_width, yloc - hw_start - flange_start,
                          xloc + 20 + flange_width, yloc + hw_start + flange_start, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + 20 - flange_width, yloc - hw_start - flange_start,
                          xloc + 20 + flange_width, yloc - hw_start - flange_start, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc + 20 - flange_width, yloc + hw_start + flange_start,
                          xloc + 20 + flange_width, yloc + hw_start + flange_start, occlusion_poly, sw)

        # End flange at x=-20: full rectangle
        clip_and_draw_line(drawing, xloc - 20 - flange_width, yloc - hw_end - flange_end,
                          xloc - 20 - flange_width, yloc + hw_end + flange_end, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - 20 + flange_width, yloc - hw_end - flange_end,
                          xloc - 20 + flange_width, yloc + hw_end + flange_end, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - 20 - flange_width, yloc - hw_end - flange_end,
                          xloc - 20 + flange_width, yloc - hw_end - flange_end, occlusion_poly, sw)
        clip_and_draw_line(drawing, xloc - 20 - flange_width, yloc + hw_end + flange_end,
                          xloc - 20 + flange_width, yloc + hw_end + flange_end, occlusion_poly, sw)


# ============================================================================
# PIPE CONNECTION LOGIC
# ============================================================================

# Port sizes: 't' = tiny (±8), 'n' = narrow (±15), 'm' = medium (±30)
# None = no opening (closed)

# Direction system for 8-neighbor WFC
DIRECTION_OFFSETS = {
    'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0),
    'NE': (1, -1), 'NW': (-1, -1), 'SE': (1, 1), 'SW': (-1, 1),
}
OPPOSITE = {
    'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E',
    'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW',
}

VOID_CHAR = '\x00'  # Void tile — no ports, no geometry, not rendered

PORTS = {
    VOID_CHAR: {},  # Void — masked cell, no ports in any direction
    # Medium-width pipes (current default, ±30 walls)
    'r': {'S': 'm', 'E': 'm'},
    '7': {'S': 'm', 'W': 'm'},
    'j': {'N': 'm', 'W': 'm'},
    'L': {'N': 'm', 'E': 'm'},
    '|': {'N': 'm', 'S': 'm'},
    '-': {'E': 'm', 'W': 'm'},
    '+': {'N': 'm', 'S': 'm', 'E': 'm', 'W': 'm'},
    # Crossovers: same-size
    '+n': {'N': 'n', 'S': 'n', 'E': 'n', 'W': 'n'},
    '+t': {'N': 't', 'S': 't', 'E': 't', 'W': 't'},
    # Crossovers: mixed-size (first=vertical/top, second=horizontal/bottom)
    '+mn': {'N': 'm', 'S': 'm', 'E': 'n', 'W': 'n'},
    '+nm': {'N': 'n', 'S': 'n', 'E': 'm', 'W': 'm'},
    '+mt': {'N': 'm', 'S': 'm', 'E': 't', 'W': 't'},
    '+tm': {'N': 't', 'S': 't', 'E': 'm', 'W': 'm'},
    '+nt': {'N': 'n', 'S': 'n', 'E': 't', 'W': 't'},
    '+tn': {'N': 't', 'S': 't', 'E': 'n', 'W': 'n'},
    'X': {'N': 'm', 'S': 'm', 'E': 'm', 'W': 'm'},
    'T': {'N': 'm', 'E': 'm', 'W': 'm'},
    'B': {'S': 'm', 'E': 'm', 'W': 'm'},
    'E': {'N': 'm', 'S': 'm', 'E': 'm'},
    'W': {'N': 'm', 'S': 'm', 'W': 'm'},

    # Narrow pipes (±15 walls)
    'i': {'N': 'n', 'S': 'n'},           # Narrow vertical
    '=': {'E': 'n', 'W': 'n'},           # Narrow horizontal
    'nr': {'S': 'n', 'E': 'n'},          # Narrow corner
    'n7': {'S': 'n', 'W': 'n'},
    'nj': {'N': 'n', 'W': 'n'},
    'nL': {'N': 'n', 'E': 'n'},
    'nX': {'N': 'n', 'S': 'n', 'E': 'n', 'W': 'n'},
    'nT': {'N': 'n', 'E': 'n', 'W': 'n'},
    'nB': {'S': 'n', 'E': 'n', 'W': 'n'},
    'nE': {'N': 'n', 'S': 'n', 'E': 'n'},
    'nW': {'N': 'n', 'S': 'n', 'W': 'n'},

    # Tiny pipes (±8 walls)
    '!': {'N': 't', 'S': 't'},           # Tiny vertical
    '.': {'E': 't', 'W': 't'},           # Tiny horizontal
    'tr': {'S': 't', 'E': 't'},          # Tiny corner
    't7': {'S': 't', 'W': 't'},
    'tj': {'N': 't', 'W': 't'},
    'tL': {'N': 't', 'E': 't'},
    'tX': {'N': 't', 'S': 't', 'E': 't', 'W': 't'},
    'tT': {'N': 't', 'E': 't', 'W': 't'},
    'tB': {'S': 't', 'E': 't', 'W': 't'},
    'tE': {'N': 't', 'S': 't', 'E': 't'},
    'tW': {'N': 't', 'S': 't', 'W': 't'},

    # Dodge/zigzag tiles (same ports as straights, Z-shaped path)
    'z>': {'N': 'm', 'S': 'm'},  'z<': {'N': 'm', 'S': 'm'},
    'z^': {'E': 'm', 'W': 'm'},  'zv': {'E': 'm', 'W': 'm'},
    'nz>': {'N': 'n', 'S': 'n'}, 'nz<': {'N': 'n', 'S': 'n'},
    'nz^': {'E': 'n', 'W': 'n'}, 'nzv': {'E': 'n', 'W': 'n'},
    'tz>': {'N': 't', 'S': 't'}, 'tz<': {'N': 't', 'S': 't'},
    'tz^': {'E': 't', 'W': 't'}, 'tzv': {'E': 't', 'W': 't'},

    # S-Bends (same ports as straights, smooth S-curve path)
    's>': {'N': 'm', 'S': 'm'},  's<': {'N': 'm', 'S': 'm'},
    's^': {'E': 'm', 'W': 'm'},  'sv': {'E': 'm', 'W': 'm'},
    'ns>': {'N': 'n', 'S': 'n'}, 'ns<': {'N': 'n', 'S': 'n'},
    'ns^': {'E': 'n', 'W': 'n'}, 'nsv': {'E': 'n', 'W': 'n'},
    'ts>': {'N': 't', 'S': 't'}, 'ts<': {'N': 't', 'S': 't'},
    'ts^': {'E': 't', 'W': 't'}, 'tsv': {'E': 't', 'W': 't'},

    # Long-radius elbows (wide arc corners, same ports as standard corners)
    'lr': {'S': 'm', 'E': 'm'},  'l7': {'S': 'm', 'W': 'm'},
    'lj': {'N': 'm', 'W': 'm'},  'lL': {'N': 'm', 'E': 'm'},
    'nlr': {'S': 'n', 'E': 'n'}, 'nl7': {'S': 'n', 'W': 'n'},
    'nlj': {'N': 'n', 'W': 'n'}, 'nlL': {'N': 'n', 'E': 'n'},
    'tlr': {'S': 't', 'E': 't'}, 'tl7': {'S': 't', 'W': 't'},
    'tlj': {'N': 't', 'W': 't'}, 'tlL': {'N': 't', 'E': 't'},

    # Segmented elbows (faceted corners, same ports as standard corners)
    'gr': {'S': 'm', 'E': 'm'},  'g7': {'S': 'm', 'W': 'm'},
    'gj': {'N': 'm', 'W': 'm'},  'gL': {'N': 'm', 'E': 'm'},
    'ngr': {'S': 'n', 'E': 'n'}, 'ng7': {'S': 'n', 'W': 'n'},
    'ngj': {'N': 'n', 'W': 'n'}, 'ngL': {'N': 'n', 'E': 'n'},
    'tgr': {'S': 't', 'E': 't'}, 'tg7': {'S': 't', 'W': 't'},
    'tgj': {'N': 't', 'W': 't'}, 'tgL': {'N': 't', 'E': 't'},

    # Chamfered corners (45-degree elbows, same ports as standard corners)
    'cr': {'S': 'm', 'E': 'm'},  'c7': {'S': 'm', 'W': 'm'},
    'cj': {'N': 'm', 'W': 'm'},  'cL': {'N': 'm', 'E': 'm'},
    'ncr': {'S': 'n', 'E': 'n'}, 'nc7': {'S': 'n', 'W': 'n'},
    'ncj': {'N': 'n', 'W': 'n'}, 'ncL': {'N': 'n', 'E': 'n'},
    'tcr': {'S': 't', 'E': 't'}, 'tc7': {'S': 't', 'W': 't'},
    'tcj': {'N': 't', 'W': 't'}, 'tcL': {'N': 't', 'E': 't'},

    # Teardrop elbows (asymmetric arcs, same ports as standard corners)
    'dr': {'S': 'm', 'E': 'm'},  'd7': {'S': 'm', 'W': 'm'},
    'dj': {'N': 'm', 'W': 'm'},  'dL': {'N': 'm', 'E': 'm'},
    'ndr': {'S': 'n', 'E': 'n'}, 'nd7': {'S': 'n', 'W': 'n'},
    'ndj': {'N': 'n', 'W': 'n'}, 'ndL': {'N': 'n', 'E': 'n'},
    'tdr': {'S': 't', 'E': 't'}, 'td7': {'S': 't', 'W': 't'},
    'tdj': {'N': 't', 'W': 't'}, 'tdL': {'N': 't', 'E': 't'},

    # Reducers (connect different sizes)
    # Vertical reducers (medium↔narrow)
    'Rv': {'N': 'm', 'S': 'n'},          # Medium-to-narrow vertical (medium on top)
    'RV': {'N': 'n', 'S': 'm'},          # Narrow-to-medium vertical (narrow on top)
    # Vertical reducers (narrow↔tiny)
    'Tv': {'N': 'n', 'S': 't'},          # Narrow-to-tiny vertical (narrow on top)
    'TV': {'N': 't', 'S': 'n'},          # Tiny-to-narrow vertical (tiny on top)
    # Horizontal reducers (medium↔narrow)
    'Rh': {'E': 'm', 'W': 'n'},          # Medium-to-narrow horizontal (medium on right)
    'RH': {'E': 'n', 'W': 'm'},          # Narrow-to-medium horizontal
    # Horizontal reducers (narrow↔tiny)
    'Th': {'E': 'n', 'W': 't'},          # Narrow-to-tiny horizontal (narrow on right)
    'TH': {'E': 't', 'W': 'n'},          # Tiny-to-narrow horizontal

    # Diagonal straights (narrow and tiny only — medium too wide for 45°)
    'nDa': {'SW': 'n', 'NE': 'n'},        # Narrow ascending (SW→NE)
    'nDd': {'NW': 'n', 'SE': 'n'},        # Narrow descending (NW→SE)
    'tDa': {'SW': 't', 'NE': 't'},        # Tiny ascending
    'tDd': {'NW': 't', 'SE': 't'},        # Tiny descending

    # Adapters: shape A (vertical cardinal → diagonal)
    'naSne': {'S': 'n', 'NE': 'n'},       # S→NE narrow
    'naWse': {'W': 'n', 'SE': 'n'},       # W→SE narrow
    'naNsw': {'N': 'n', 'SW': 'n'},       # N→SW narrow
    'naEnw': {'E': 'n', 'NW': 'n'},       # E→NW narrow
    'taSne': {'S': 't', 'NE': 't'},       # S→NE tiny
    'taWse': {'W': 't', 'SE': 't'},       # W→SE tiny
    'taNsw': {'N': 't', 'SW': 't'},       # N→SW tiny
    'taEnw': {'E': 't', 'NW': 't'},       # E→NW tiny

    # Adapters: shape B (horizontal cardinal → diagonal)
    'naWne': {'W': 'n', 'NE': 'n'},       # W→NE narrow
    'naNse': {'N': 'n', 'SE': 'n'},       # N→SE narrow
    'naEsw': {'E': 'n', 'SW': 'n'},       # E→SW narrow
    'naSnw': {'S': 'n', 'NW': 'n'},       # S→NW narrow
    'taWne': {'W': 't', 'NE': 't'},       # W→NE tiny
    'taNse': {'N': 't', 'SE': 't'},       # N→SE tiny
    'taEsw': {'E': 't', 'SW': 't'},       # E→SW tiny
    'taSnw': {'S': 't', 'NW': 't'},       # S→NW tiny

    # Diagonal corners (V-shaped turns between two diagonal directions)
    'ndce': {'NE': 'n', 'SE': 'n'},       # East-pointing V (narrow)
    'ndcs': {'SE': 'n', 'SW': 'n'},       # South-pointing V
    'ndcw': {'NW': 'n', 'SW': 'n'},       # West-pointing V
    'ndcn': {'NW': 'n', 'NE': 'n'},       # North-pointing V
    'tdce': {'NE': 't', 'SE': 't'},       # East-pointing V (tiny)
    'tdcs': {'SE': 't', 'SW': 't'},       # South-pointing V
    'tdcw': {'NW': 't', 'SW': 't'},       # West-pointing V
    'tdcn': {'NW': 't', 'NE': 't'},       # North-pointing V

    # Diagonal endcaps (half-diagonal with cap at center)
    'ndne': {'NE': 'n'},                   # Narrow endcap NE
    'ndse': {'SE': 'n'},                   # Narrow endcap SE
    'ndsw': {'SW': 'n'},                   # Narrow endcap SW
    'ndnw': {'NW': 'n'},                   # Narrow endcap NW
    'tdne': {'NE': 't'},                   # Tiny endcap NE
    'tdse': {'SE': 't'},                   # Tiny endcap SE
    'tdsw': {'SW': 't'},                   # Tiny endcap SW
    'tdnw': {'NW': 't'},                   # Tiny endcap NW

    # Cardinal endcaps (dead-end tiles, single port)
    'eN': {'N': 'm'},  'eS': {'S': 'm'},  'eE': {'E': 'm'},  'eW': {'W': 'm'},
    'neN': {'N': 'n'}, 'neS': {'S': 'n'}, 'neE': {'E': 'n'}, 'neW': {'W': 'n'},
    'teN': {'N': 't'}, 'teS': {'S': 't'}, 'teE': {'E': 't'}, 'teW': {'W': 't'},

    # Vanishing pipes (tapered dead-ends, single port)
    'vN': {'N': 'm'},  'vS': {'S': 'm'},  'vE': {'E': 'm'},  'vW': {'W': 'm'},
    'nvN': {'N': 'n'}, 'nvS': {'S': 'n'}, 'nvE': {'E': 'n'}, 'nvW': {'W': 'n'},
    'tvN': {'N': 't'}, 'tvS': {'S': 't'}, 'tvE': {'E': 't'}, 'tvW': {'W': 't'},

    # Mixed-size corners (medium↔narrow, 90° turns connecting different sizes)
    'mnr': {'S': 'm', 'E': 'n'}, 'mn7': {'S': 'm', 'W': 'n'},
    'mnj': {'N': 'm', 'W': 'n'}, 'mnL': {'N': 'm', 'E': 'n'},
    'nmr': {'S': 'n', 'E': 'm'}, 'nm7': {'S': 'n', 'W': 'm'},
    'nmj': {'N': 'n', 'W': 'm'}, 'nmL': {'N': 'n', 'E': 'm'},

    # Diagonal crossovers (medium cardinal x narrow diagonal)
    'xHa': {'W': 'm', 'E': 'm', 'SW': 'n', 'NE': 'n'},
    'xHd': {'W': 'm', 'E': 'm', 'NW': 'n', 'SE': 'n'},
    'xVa': {'N': 'm', 'S': 'm', 'SW': 'n', 'NE': 'n'},
    'xVd': {'N': 'm', 'S': 'm', 'NW': 'n', 'SE': 'n'},
}

_ALL_WITH_VOID = set(PORTS.keys())

# Backward compatibility: OPENINGS as set of directions (for code that only needs presence)
OPENINGS = {ch: set(ports.keys()) for ch, ports in PORTS.items()}

# Precomputed compatibility table: COMPAT_TABLE[ch][direction] = frozenset of compatible tiles
# Built once at module load; replaces O(106)-per-call iteration in get_compatible_neighbors
# Includes VOID_CHAR so propagation works when masked cells are present.
COMPAT_TABLE = {}
for _ch in _ALL_WITH_VOID:
    COMPAT_TABLE[_ch] = {}
    for _dir in DIRECTION_OFFSETS:
        _opp = OPPOSITE[_dir]
        _port = PORTS.get(_ch, {}).get(_dir)
        COMPAT_TABLE[_ch][_dir] = frozenset(
            c for c in _ALL_WITH_VOID if PORTS.get(c, {}).get(_opp) == _port
        )

# Precomputed edge-constraint sets: tiles with NO opening in each direction
# Includes VOID_CHAR so boundary constraints work for masked cells.
_NO_OPENING = {}
for _dir in DIRECTION_OFFSETS:
    _NO_OPENING[_dir] = frozenset(
        ch for ch in _ALL_WITH_VOID if _dir not in PORTS.get(ch, {})
    )

# Precomputed sets: tiles that HAVE an opening in each direction
_HAS_OPENING = {}
for _dir in DIRECTION_OFFSETS:
    _HAS_OPENING[_dir] = frozenset(
        ch for ch in _ALL_WITH_VOID if _dir in PORTS.get(ch, {})
    )

# ALL_CHARS excludes VOID_CHAR — void tiles only appear via explicit mask assignment
ALL_CHARS = _ALL_WITH_VOID - {VOID_CHAR}

# Tile category mappings for weight system
TILE_SIZE = {}
for _ch in ['|', '-', 'r', '7', 'j', 'L', '+', 'X', 'T', 'B', 'E', 'W',
            'z>', 'z<', 'z^', 'zv', 's>', 's<', 's^', 'sv',
            'gr', 'g7', 'gj', 'gL',
            'lr', 'l7', 'lj', 'lL',
            'cr', 'c7', 'cj', 'cL', 'dr', 'd7', 'dj', 'dL',
            'eN', 'eS', 'eE', 'eW', 'vN', 'vS', 'vE', 'vW']:
    TILE_SIZE[_ch] = 'medium'
for _ch in ['i', '=', 'nr', 'n7', 'nj', 'nL', 'nX', 'nT', 'nB', 'nE', 'nW',
            'nz>', 'nz<', 'nz^', 'nzv', 'ns>', 'ns<', 'ns^', 'nsv',
            'ngr', 'ng7', 'ngj', 'ngL',
            'nlr', 'nl7', 'nlj', 'nlL',
            'ncr', 'nc7', 'ncj', 'ncL', 'ndr', 'nd7', 'ndj', 'ndL',
            'nDa', 'nDd', 'naSne', 'naWse', 'naNsw', 'naEnw', 'naWne', 'naNse', 'naEsw', 'naSnw',
            'ndce', 'ndcs', 'ndcw', 'ndcn', 'ndne', 'ndse', 'ndsw', 'ndnw',
            'neN', 'neS', 'neE', 'neW', 'nvN', 'nvS', 'nvE', 'nvW']:
    TILE_SIZE[_ch] = 'narrow'
for _ch in ['!', '.', 'tr', 't7', 'tj', 'tL', 'tX', 'tT', 'tB', 'tE', 'tW',
            'tz>', 'tz<', 'tz^', 'tzv', 'ts>', 'ts<', 'ts^', 'tsv',
            'tgr', 'tg7', 'tgj', 'tgL',
            'tlr', 'tl7', 'tlj', 'tlL',
            'tcr', 'tc7', 'tcj', 'tcL', 'tdr', 'td7', 'tdj', 'tdL',
            'tDa', 'tDd', 'taSne', 'taWse', 'taNsw', 'taEnw', 'taWne', 'taNse', 'taEsw', 'taSnw',
            'tdce', 'tdcs', 'tdcw', 'tdcn', 'tdne', 'tdse', 'tdsw', 'tdnw',
            'teN', 'teS', 'teE', 'teW', 'tvN', 'tvS', 'tvE', 'tvW']:
    TILE_SIZE[_ch] = 'tiny'
for _ch in ['Rv', 'RV', 'Rh', 'RH']:
    TILE_SIZE[_ch] = 'reducer_mn'
for _ch in ['Tv', 'TV', 'Th', 'TH']:
    TILE_SIZE[_ch] = 'reducer_nt'
# Crossover size categories
TILE_SIZE['+n'] = 'narrow'
TILE_SIZE['+t'] = 'tiny'
for _ch in ['+mn', '+nm']:
    TILE_SIZE[_ch] = 'reducer_mn'
for _ch in ['+mt', '+tm']:
    TILE_SIZE[_ch] = 'crossover_mt'
for _ch in ['+nt', '+tn']:
    TILE_SIZE[_ch] = 'reducer_nt'
for _ch in ['xHa', 'xHd', 'xVa', 'xVd']:
    TILE_SIZE[_ch] = 'reducer_mn'
for _ch in _MIXED_CORNER_PARAMS:
    TILE_SIZE[_ch] = 'reducer_mn'

TILE_SHAPE = {}
for _ch in ['|', '-', 'i', '=', '!', '.']:
    TILE_SHAPE[_ch] = 'straight'
for _ch in ['r', '7', 'j', 'L', 'nr', 'n7', 'nj', 'nL', 'tr', 't7', 'tj', 'tL']:
    TILE_SHAPE[_ch] = 'corner'
for _ch in ['+', '+n', '+t', '+mn', '+nm', '+mt', '+tm', '+nt', '+tn',
            'X', 'T', 'B', 'E', 'W', 'nX', 'nT', 'nB', 'nE', 'nW', 'tX', 'tT', 'tB', 'tE', 'tW']:
    TILE_SHAPE[_ch] = 'junction'
for _ch in ['Rv', 'RV', 'Rh', 'RH', 'Tv', 'TV', 'Th', 'TH']:
    TILE_SHAPE[_ch] = 'reducer'
for _ch in ['z>', 'z<', 'z^', 'zv', 'nz>', 'nz<', 'nz^', 'nzv', 'tz>', 'tz<', 'tz^', 'tzv']:
    TILE_SHAPE[_ch] = 'dodge'
for _ch in ['s>', 's<', 's^', 'sv', 'ns>', 'ns<', 'ns^', 'nsv', 'ts>', 'ts<', 'ts^', 'tsv']:
    TILE_SHAPE[_ch] = 'sbend'
for _ch in ['gr', 'g7', 'gj', 'gL', 'ngr', 'ng7', 'ngj', 'ngL', 'tgr', 'tg7', 'tgj', 'tgL']:
    TILE_SHAPE[_ch] = 'segmented'
for _ch in ['lr', 'l7', 'lj', 'lL', 'nlr', 'nl7', 'nlj', 'nlL', 'tlr', 'tl7', 'tlj', 'tlL']:
    TILE_SHAPE[_ch] = 'long_radius'
for _ch in ['cr', 'c7', 'cj', 'cL', 'ncr', 'nc7', 'ncj', 'ncL', 'tcr', 'tc7', 'tcj', 'tcL']:
    TILE_SHAPE[_ch] = 'chamfer'
for _ch in ['dr', 'd7', 'dj', 'dL', 'ndr', 'nd7', 'ndj', 'ndL', 'tdr', 'td7', 'tdj', 'tdL']:
    TILE_SHAPE[_ch] = 'teardrop'
for _ch in (list(_DIAG_PARAMS.keys()) + list(_ADAPTER_PARAMS.keys()) +
           list(_DIAG_CORNER_PARAMS.keys()) +
           ['xHa', 'xHd', 'xVa', 'xVd']):
    TILE_SHAPE[_ch] = 'diagonal'
for _ch in _DIAG_ENDCAP_PARAMS:
    TILE_SHAPE[_ch] = 'diagonal_endcap'
for _ch in _ENDCAP_PARAMS:
    TILE_SHAPE[_ch] = 'endcap'
for _ch in _VANISHING_PARAMS:
    TILE_SHAPE[_ch] = 'vanishing'
for _ch in _MIXED_CORNER_PARAMS:
    TILE_SHAPE[_ch] = 'mixed_corner'
TILE_SIZE[VOID_CHAR] = 'void'
TILE_SHAPE[VOID_CHAR] = 'void'

# Representative tiles for catalog/debug view: (display_name, {size: tile_char})
CATALOG_TILES = [
    ('straight',         {'medium': '|',     'narrow': 'i',     'tiny': '!'}),
    ('corner',           {'medium': 'r',     'narrow': 'nr',    'tiny': 'tr'}),
    ('junction (cross)', {'medium': '+',     'narrow': '+n',    'tiny': '+t'}),
    ('junction (tee)',   {'medium': 'T',     'narrow': 'nT',    'tiny': 'tT'}),
    ('reducer (M-N)',    {'medium': 'Rv'}),
    ('reducer (N-T)',    {'medium': 'Tv'}),
    ('dodge',            {'medium': 'z>',    'narrow': 'nz>',   'tiny': 'tz>'}),
    ('sbend',            {'medium': 's>',    'narrow': 'ns>',   'tiny': 'ts>'}),
    ('segmented',        {'medium': 'gr',    'narrow': 'ngr',   'tiny': 'tgr'}),
    ('long_radius',      {'medium': 'lr',    'narrow': 'nlr',   'tiny': 'tlr'}),
    ('chamfer',          {'medium': 'cr',    'narrow': 'ncr',   'tiny': 'tcr'}),
    ('teardrop',         {'medium': 'dr',    'narrow': 'ndr',   'tiny': 'tdr'}),
    ('diagonal',         {'narrow': 'nDa',   'tiny': 'tDa'}),
    ('diag adapter',     {'narrow': 'naSne', 'tiny': 'taSne'}),
    ('diag corner',      {'narrow': 'ndce',  'tiny': 'tdce'}),
    ('diagonal endcap',  {'narrow': 'ndne',  'tiny': 'tdne'}),
    ('endcap',           {'medium': 'eN',    'narrow': 'neN',   'tiny': 'teN'}),
    ('vanishing',        {'medium': 'vN',    'narrow': 'nvN',   'tiny': 'tvN'}),
    ('mixed corner',     {'medium': 'mnr'}),
    ('diag crossover',   {'narrow': 'xHa'}),
]

# Crossover tile → (vertical_tube_char, horizontal_tube_char)
# Vertical tube is drawn on top, horizontal underneath
CROSSOVER_TUBES = {
    '+':   ('|', '-'),
    '+n':  ('i', '='),
    '+t':  ('!', '.'),
    '+mn': ('|', '='),
    '+nm': ('i', '-'),
    '+mt': ('|', '.'),
    '+tm': ('!', '-'),
    '+nt': ('i', '.'),
    '+tn': ('!', '='),
}

# Diagonal crossover tiles: medium cardinal x narrow diagonal
# Maps tile char -> (cardinal_tube_char, diagonal_char). Diagonal is on top.
DIAG_CROSSOVER_TUBES = {
    'xHa': ('-', 'nDa'),   # horizontal medium + ascending narrow diagonal
    'xHd': ('-', 'nDd'),   # horizontal medium + descending narrow diagonal
    'xVa': ('|', 'nDa'),   # vertical medium + ascending narrow diagonal
    'xVd': ('|', 'nDd'),   # vertical medium + descending narrow diagonal
}


def _ensure_endcaps_for_mask(tile_weights):
    """Return tile_weights with endcaps/vanishing enabled (needed for mask boundaries)."""
    if tile_weights is None:
        tile_weights = {
            'size': {'medium': 1.0, 'narrow': 1.0, 'tiny': 1.0},
            'shape': {
                'straight': 1.0, 'corner': 3.0, 'junction': 2.0,
                'reducer': 1.0, 'dodge': 0.0, 'diagonal': 0.0,
                'diagonal_endcap': 0.0, 'mixed_corner': 0.0,
                'sbend': 0.0, 'segmented': 0.0, 'long_radius': 0.0,
                'chamfer': 0.0, 'teardrop': 0.0,
                'endcap': 0.5, 'vanishing': 0.5,
            },
        }
    else:
        tw = {
            'size': dict(tile_weights.get('size', {})),
            'shape': dict(tile_weights.get('shape', {})),
        }
        if tw['shape'].get('endcap', 0) <= 0:
            tw['shape']['endcap'] = 0.5
        if tw['shape'].get('vanishing', 0) <= 0:
            tw['shape']['vanishing'] = 0.5
        tile_weights = tw
    return tile_weights


def get_tile_weight(ch, tile_weights):
    """Calculate tile weight from size and shape multipliers."""
    if ch == VOID_CHAR:
        return 0  # Void tiles are never chosen by WFC
    if tile_weights is None:
        shape = TILE_SHAPE.get(ch)
        if shape in ('diagonal', 'diagonal_endcap', 'endcap', 'mixed_corner',
                     'vanishing', 'segmented', 'long_radius', 'sbend'):
            return 0  # Off by default — user opts in
        if ch in CROSSOVER_TUBES:
            return 2
        if shape in ('corner', 'chamfer', 'teardrop'):
            return 3
        return 1

    size_weights = tile_weights.get('size', {})
    shape_weights = tile_weights.get('shape', {})

    size_cat = TILE_SIZE.get(ch, 'medium')
    if size_cat == 'reducer_mn':
        s = (size_weights.get('medium', 1.0) + size_weights.get('narrow', 1.0)) / 2
    elif size_cat == 'reducer_nt':
        s = (size_weights.get('narrow', 1.0) + size_weights.get('tiny', 1.0)) / 2
    elif size_cat == 'crossover_mt':
        s = (size_weights.get('medium', 1.0) + size_weights.get('tiny', 1.0)) / 2
    else:
        s = size_weights.get(size_cat, 1.0)

    shape_cat = TILE_SHAPE.get(ch, 'straight')
    h = shape_weights.get(shape_cat, 1.0)

    return s * h


def _lerp_size_weight(base_size, t, spatial_map):
    """Linearly interpolate between start and end weight for a base size."""
    cfg = spatial_map.get(base_size, {'start': 1.0, 'end': 1.0})
    start = cfg.get('start', 1.0)
    end = cfg.get('end', 1.0)
    return start + (end - start) * t


def compute_spatial_size_weight(size_cat, x, y, global_width, global_height,
                                spatial_map):
    """Compute spatial weight multiplier for a size category at grid position.

    Returns a float >= spatial_map['min_weight']. Returns 1.0 if spatial_map is
    None or disabled.
    """
    if spatial_map is None or not spatial_map.get('enabled', False):
        return 1.0

    map_type = spatial_map.get('type', 'horizontal')
    min_w = spatial_map.get('min_weight', 0.05)

    # Compute normalized parameter t in [0, 1]
    if map_type == 'horizontal':
        t = x / max(global_width - 1, 1)
    elif map_type == 'vertical':
        t = y / max(global_height - 1, 1)
    elif map_type == 'radial':
        cx = (global_width - 1) / 2.0
        cy = (global_height - 1) / 2.0
        max_r = math.sqrt(cx * cx + cy * cy)
        if max_r < 1e-6:
            t = 0.0
        else:
            t = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r
        t = min(t, 1.0)
    else:
        return 1.0

    # For composite size categories, average the constituent sizes
    if size_cat == 'reducer_mn':
        w = (_lerp_size_weight('medium', t, spatial_map)
             + _lerp_size_weight('narrow', t, spatial_map)) / 2.0
    elif size_cat == 'reducer_nt':
        w = (_lerp_size_weight('narrow', t, spatial_map)
             + _lerp_size_weight('tiny', t, spatial_map)) / 2.0
    elif size_cat == 'crossover_mt':
        w = (_lerp_size_weight('medium', t, spatial_map)
             + _lerp_size_weight('tiny', t, spatial_map)) / 2.0
    else:
        w = _lerp_size_weight(size_cat, t, spatial_map)

    return max(w, min_w)


def get_port_size(ch, direction):
    """Get the port size for a tile in a given direction. Returns None if no port."""
    return PORTS.get(ch, {}).get(direction)


def has_opening(ch, direction):
    """Check if tile has an opening in given direction (any size)."""
    return get_port_size(ch, direction) is not None


def get_compatible_neighbors(ch, direction):
    """Get all tiles that can connect to ch in the given direction.
    Uses precomputed COMPAT_TABLE for O(1) lookup."""
    return COMPAT_TABLE[ch][direction]


def find_n(ch):
    return get_compatible_neighbors(ch, 'N')

def find_s(ch):
    return get_compatible_neighbors(ch, 'S')

def find_e(ch):
    return get_compatible_neighbors(ch, 'E')

def find_w(ch):
    return get_compatible_neighbors(ch, 'W')


# ============================================================================
# WAVE FUNCTION COLLAPSE
# ============================================================================

def would_complete_tight_circle(poss_grid, x, y, ch, width, height):
    """Check if placing ch at (x,y) would complete a tight 2x2 circle.

    Uses port-based detection: a tight circle forms when 4 tiles in a 2x2 block
    each have exactly 2 ports forming an L-shape, and port sizes match at all
    shared internal edges. Works for all corner-type tiles (standard, chamfered,
    teardrop, etc.).
    """
    # Required port directions for each role in the 2x2 block:
    #   TL(x,y)   TR(x+1,y)
    #   BL(x,y+1) BR(x+1,y+1)
    _ROLE_DIRS = {
        'TL': frozenset(['S', 'E']),  # like 'r'
        'TR': frozenset(['S', 'W']),  # like '7'
        'BL': frozenset(['N', 'E']),  # like 'L'
        'BR': frozenset(['N', 'W']),  # like 'j'
    }

    # Four possible 2x2 blocks that include (x,y), with (x,y)'s role
    _BLOCKS = [
        ('TL', [(1, 0, 'TR'), (0, 1, 'BL'), (1, 1, 'BR')]),
        ('TR', [(-1, 0, 'TL'), (-1, 1, 'BL'), (0, 1, 'BR')]),
        ('BL', [(0, -1, 'TL'), (1, -1, 'TR'), (1, 0, 'BR')]),
        ('BR', [(-1, -1, 'TL'), (0, -1, 'TR'), (-1, 0, 'BL')]),
    ]

    ch_ports = PORTS.get(ch, {})

    for my_role, neighbors in _BLOCKS:
        # Check if ch has the right port directions for this role
        if frozenset(ch_ports.keys()) != _ROLE_DIRS[my_role]:
            continue

        # Check all neighbors are collapsed and match their roles
        tile_ports = {my_role: ch_ports}
        all_match = True
        for dx, dy, role in neighbors:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                all_match = False
                break
            cell = poss_grid[nx][ny]
            if len(cell) != 1:
                all_match = False
                break
            n_ch = next(iter(cell))
            n_ports = PORTS.get(n_ch, {})
            if frozenset(n_ports.keys()) != _ROLE_DIRS[role]:
                all_match = False
                break
            tile_ports[role] = n_ports

        if not all_match:
            continue

        # Check port size compatibility at all internal edges
        if (tile_ports['TL']['E'] == tile_ports['TR']['W'] and
            tile_ports['TL']['S'] == tile_ports['BL']['N'] and
            tile_ports['TR']['S'] == tile_ports['BR']['N'] and
            tile_ports['BL']['E'] == tile_ports['BR']['W']):
            return True

    return False


def remove_tight_circle_possibilities(poss_grid, width, height):
    any_changes = False
    changed = True
    while changed:
        changed = False
        for x in range(width):
            for y in range(height):
                if len(poss_grid[x][y]) <= 1:
                    continue
                to_remove = set()
                for ch in poss_grid[x][y]:
                    if would_complete_tight_circle(poss_grid, x, y, ch, width, height):
                        to_remove.add(ch)
                if to_remove and len(poss_grid[x][y] - to_remove) > 0:
                    poss_grid[x][y] -= to_remove
                    changed = True
                    any_changes = True
    return any_changes


# ============================================================================
# MASKING SYSTEM
# ============================================================================

def build_mask(width, height, shapes, invert=False):
    """Build a boolean mask grid from shape primitives.

    Each shape 'selects' cells. Multiple shapes are unioned.
    - invert=False: selected cells are BLOCKED (voids). Pipes grow elsewhere.
    - invert=True:  selected cells are ALLOWED. Everything else is void.

    Args:
        width, height: Grid dimensions.
        shapes: list of dicts, each with 'type' and shape-specific keys:
            rectangle: x0, y0, x1, y1  (start inclusive, end exclusive)
            circle:    cx, cy, r
            ring:      cx, cy, r_inner, r_outer
        invert: flip the mask after applying shapes.

    Returns:
        mask[x][y]: True = allowed, False = blocked.
    """
    # Start: all cells unselected
    selected = [[False for _ in range(height)] for _ in range(width)]

    for shape in shapes:
        stype = shape['type']
        if stype == 'rectangle':
            x0 = max(0, int(shape['x0']))
            y0 = max(0, int(shape['y0']))
            x1 = min(width, int(shape['x1']))
            y1 = min(height, int(shape['y1']))
            for x in range(x0, x1):
                for y in range(y0, y1):
                    selected[x][y] = True
        elif stype == 'circle':
            cx, cy, r = shape['cx'], shape['cy'], shape['r']
            r_sq = r * r
            for x in range(width):
                for y in range(height):
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy <= r_sq:
                        selected[x][y] = True
        elif stype == 'ring':
            cx, cy = shape['cx'], shape['cy']
            r_inner = shape['r_inner']
            r_outer = shape['r_outer']
            ri_sq = r_inner * r_inner
            ro_sq = r_outer * r_outer
            for x in range(width):
                for y in range(height):
                    dx = x - cx
                    dy = y - cy
                    dist_sq = dx * dx + dy * dy
                    if ri_sq <= dist_sq <= ro_sq:
                        selected[x][y] = True

    if invert:
        # Inverted: selected = allowed, unselected = blocked
        return [[selected[x][y] for y in range(height)] for x in range(width)]
    else:
        # Normal: selected = blocked, unselected = allowed
        return [[not selected[x][y] for y in range(height)] for x in range(width)]


def count_masked_cells(mask):
    """Return (allowed_count, blocked_count) for a mask grid."""
    blocked = sum(1 for col in mask for cell in col if not cell)
    total = len(mask) * len(mask[0]) if mask else 0
    return (total - blocked, blocked)


def scale_mask(mask, src_width, src_height, dst_width, dst_height):
    """Scale a mask from source dimensions to destination dimensions.

    Uses nearest-neighbor sampling to preserve mask shape at different
    resolutions. Used for density-preserving layers where each layer
    may have different grid dimensions.

    Args:
        mask: 2D boolean grid [x][y] where True=allowed, False=blocked
        src_width, src_height: original mask dimensions
        dst_width, dst_height: target dimensions

    Returns:
        New mask with target dimensions, or None if input is None.
    """
    if mask is None:
        return None

    if src_width == dst_width and src_height == dst_height:
        return mask

    new_mask = [[True for _ in range(dst_height)] for _ in range(dst_width)]

    for x in range(dst_width):
        for y in range(dst_height):
            # Map to source coordinates (normalized)
            src_x = int((x / dst_width) * src_width)
            src_y = int((y / dst_height) * src_height)
            # Clamp to valid range
            src_x = min(src_x, src_width - 1)
            src_y = min(src_y, src_height - 1)
            new_mask[x][y] = mask[src_x][src_y]

    return new_mask


# ============================================================================
# WFC — POSSIBILITY GRID
# ============================================================================

def create_possibility_grid(width, height):
    return [[ALL_CHARS.copy() for _ in range(height)] for _ in range(width)]


def get_constrained_possibilities(possibilities, x, y, width, height,
                                   open_edges=None):
    if open_edges is None:
        open_edges = frozenset()
    valid = possibilities
    # Cardinal edges
    if x == 0 and 'W' not in open_edges:
        valid = valid & _NO_OPENING['W']
    if x == width - 1 and 'E' not in open_edges:
        valid = valid & _NO_OPENING['E']
    if y == 0 and 'N' not in open_edges:
        valid = valid & _NO_OPENING['N']
    if y == height - 1 and 'S' not in open_edges:
        valid = valid & _NO_OPENING['S']
    # Diagonal edges — block port if neighbor cell would be out of bounds
    # and that boundary is not open
    if (x == width - 1 and 'E' not in open_edges) or (y == 0 and 'N' not in open_edges):
        valid = valid & _NO_OPENING['NE']
    if (x == 0 and 'W' not in open_edges) or (y == 0 and 'N' not in open_edges):
        valid = valid & _NO_OPENING['NW']
    if (x == width - 1 and 'E' not in open_edges) or (y == height - 1 and 'S' not in open_edges):
        valid = valid & _NO_OPENING['SE']
    if (x == 0 and 'W' not in open_edges) or (y == height - 1 and 'S' not in open_edges):
        valid = valid & _NO_OPENING['SW']
    return valid


def propagate_constraints(poss_grid, width, height, dirty_cells=None):
    """Propagate constraints using AC-3-style dirty-cell queue.

    Returns False if a contradiction is found (any cell has 0 possibilities).
    If dirty_cells is provided, only those cells are initially queued.
    Otherwise all cells are queued (used for initial propagation).
    """
    queue = deque()
    in_queue = set()

    if dirty_cells is None:
        for x in range(width):
            for y in range(height):
                queue.append((x, y))
                in_queue.add((x, y))
    else:
        # Seed both the changed cells AND their neighbors, since a change to
        # cell (x,y) means neighbors may need their possibilities reduced.
        # Without this, a collapsed cell re-checks itself (no change) and
        # neighbors never get queued.
        seed = set()
        for (cx, cy) in dirty_cells:
            seed.add((cx, cy))
            for _, (dx, dy) in DIRECTION_OFFSETS.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    seed.add((nx, ny))
        for cell in seed:
            queue.append(cell)
            in_queue.add(cell)

    while True:
        # Phase A: AC-3 constraint propagation
        phase_a_changed = set()
        while queue:
            x, y = queue.popleft()
            in_queue.discard((x, y))
            old = poss_grid[x][y]
            current = get_constrained_possibilities(old, x, y, width, height)
            for direction, (dx, dy) in DIRECTION_OFFSETS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    opp = OPPOSITE[direction]
                    nposs = poss_grid[nx][ny]
                    if len(nposs) == 1:
                        valid = COMPAT_TABLE[next(iter(nposs))][opp]
                    else:
                        valid = frozenset().union(
                            *(COMPAT_TABLE[n][opp] for n in nposs)
                        )
                    current &= valid
            if len(current) == 0:
                poss_grid[x][y] = current
                return False  # contradiction
            if current != old:
                poss_grid[x][y] = current
                phase_a_changed.add((x, y))
                for direction, (dx, dy) in DIRECTION_OFFSETS.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if (nx, ny) not in in_queue:
                            queue.append((nx, ny))
                            in_queue.add((nx, ny))

        # Phase B: tight-circle removal (scoped to neighborhood of changes)
        # Skip for large grids (>200 cells) where it over-constrains the search
        # and makes solutions impossible. Tight circles are purely aesthetic.
        if width * height <= 200:
            check_cells = set()
            for (cx, cy) in phase_a_changed:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            check_cells.add((nx, ny))

            tight_changed = set()
            tc_any = True
            while tc_any:
                tc_any = False
                for (x, y) in list(check_cells):
                    if len(poss_grid[x][y]) <= 1:
                        continue
                    to_remove = set()
                    for ch in poss_grid[x][y]:
                        if would_complete_tight_circle(poss_grid, x, y, ch, width, height):
                            to_remove.add(ch)
                    if to_remove and len(poss_grid[x][y] - to_remove) > 0:
                        poss_grid[x][y] -= to_remove
                        tight_changed.add((x, y))
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    check_cells.add((nx, ny))
                        tc_any = True

            # If tight-circle pass changed cells, re-queue and loop back
            if tight_changed:
                for cell in tight_changed:
                    if cell not in in_queue:
                        queue.append(cell)
                        in_queue.add(cell)
                continue

        break  # converged

    return True  # no contradiction


def find_min_entropy_cell(poss_grid, width, height):
    min_entropy = float('inf')
    min_cells = []
    for x in range(width):
        for y in range(height):
            entropy = len(poss_grid[x][y])
            if 1 < entropy < min_entropy:
                min_entropy = entropy
                min_cells = [(x, y)]
            elif entropy == min_entropy:
                min_cells.append((x, y))
    if min_cells:
        return random.choice(min_cells)
    return None


def collapse_cell(poss_grid, x, y, tile_weights=None, width=None, height=None,
                  boost_port_dirs=None, spatial_map=None,
                  global_x_offset=0, global_width=None, global_height=None):
    possibilities = list(poss_grid[x][y])
    if not possibilities:
        return False
    weights = [get_tile_weight(ch, tile_weights) for ch in possibilities]
    filtered = [(ch, w) for ch, w in zip(possibilities, weights) if w > 0]
    if not filtered:
        return False

    # Apply spatial size weight multiplier
    if spatial_map is not None and spatial_map.get('enabled', False):
        gw = global_width if global_width is not None else width
        gh = global_height if global_height is not None else height
        gx = x + global_x_offset
        spatially_weighted = []
        for ch, w in filtered:
            size_cat = TILE_SIZE.get(ch, 'medium')
            sw = compute_spatial_size_weight(size_cat, gx, y, gw, gh,
                                            spatial_map)
            spatially_weighted.append((ch, w * sw))
        filtered = spatially_weighted

    # Least-constraining value: bias toward tiles that leave more options
    # for the most-constrained uncollapsed neighbor. Cheaper than checking
    # all 8 neighbors while still avoiding contradiction-prone choices.
    if width is not None and height is not None:
        # Find the most constrained uncollapsed neighbor
        mc_dir = None
        mc_count = float('inf')
        mc_poss = None
        for direction, (dx, dy) in DIRECTION_OFFSETS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                nposs = poss_grid[nx][ny]
                nc = len(nposs)
                if 1 < nc < mc_count:
                    mc_count = nc
                    mc_dir = direction
                    mc_poss = nposs

        if mc_dir is not None:
            scored = []
            for ch, w in filtered:
                compat = len(COMPAT_TABLE[ch][mc_dir] & mc_poss)
                scored.append((ch, w * (1 + compat)))
            possibilities, weights = zip(*scored)
        else:
            possibilities, weights = zip(*filtered)
    else:
        possibilities, weights = zip(*filtered)

    # Boundary boost: increase weight of tiles with ports in specified directions
    if boost_port_dirs:
        boosted = []
        for ch, w in zip(possibilities, weights):
            tile_ports = PORTS.get(ch, {})
            if any(d in tile_ports for d in boost_port_dirs):
                boosted.append((ch, w * 3.0))
            else:
                boosted.append((ch, w))
        possibilities, weights = zip(*boosted)

    chosen = random.choices(list(possibilities), weights=list(weights), k=1)[0]
    poss_grid[x][y] = {chosen}
    return True


def _wfc_backtrack(stack, poss_grid, width, height, tile_weights):
    """Pop stack frames until we find a cell with viable alternatives."""
    while stack:
        snapshot, bx, by, excluded = stack.pop()
        # Restore full grid state (undoes AC-3 + tight-circle changes)
        for i in range(len(snapshot)):
            for j in range(len(snapshot[i])):
                poss_grid[i][j] = snapshot[i][j]

        # Remove excluded tiles
        poss_grid[bx][by] -= excluded

        if len(poss_grid[bx][by]) == 0:
            continue  # empty cell — backtrack further

        # Check for viable options (non-zero weight)
        if not any(get_tile_weight(ch, tile_weights) > 0
                   for ch in poss_grid[bx][by]):
            continue  # all remaining options zero-weight — backtrack further

        # Re-propagate with reduced options
        if propagate_constraints(poss_grid, width, height,
                                  dirty_cells={(bx, by)}):
            return True  # recovered

        # Propagation found contradiction — backtrack further

    return False


def wave_function_collapse(width, height, tile_weights=None,
                           max_attempts=200, progress_callback=None,
                           mask=None, spatial_map=None):
    """Run WFC with backtracking. Returns a 2D grid of pipe chars, or None."""
    if mask is not None:
        tile_weights = _ensure_endcaps_for_mask(tile_weights)
    max_backtracks = width * height  # fail fast on bad seeds
    total_cells = width * height
    for attempt in range(1, max_attempts + 1):
        poss_grid = create_possibility_grid(width, height)

        # Apply mask: fix blocked cells to VOID_CHAR before constraints
        if mask is not None:
            for x in range(width):
                for y in range(height):
                    if not mask[x][y]:
                        poss_grid[x][y] = {VOID_CHAR}

        for x in range(width):
            for y in range(height):
                poss_grid[x][y] = get_constrained_possibilities(
                    poss_grid[x][y], x, y, width, height)

        if not propagate_constraints(poss_grid, width, height):
            continue

        stack = []  # [(snapshot, x, y, excluded_set), ...]
        backtracks = 0
        cells_collapsed = 0
        success = True

        while True:
            cell = find_min_entropy_cell(poss_grid, width, height)
            if cell is None:
                break  # all collapsed — done

            x, y = cell

            # Save state BEFORE collapse
            snapshot = [[c.copy() for c in row] for row in poss_grid]

            if not collapse_cell(poss_grid, x, y, tile_weights,
                                 width=width, height=height,
                                 spatial_map=spatial_map,
                                 global_width=width, global_height=height):
                # No viable options in this cell — backtrack
                recovered = _wfc_backtrack(
                    stack, poss_grid, width, height, tile_weights)
                if not recovered:
                    success = False
                    break
                backtracks += 1
                cells_collapsed = len(stack)
                if backtracks > max_backtracks:
                    success = False
                    break
                continue

            chosen = next(iter(poss_grid[x][y]))
            stack.append((snapshot, x, y, {chosen}))

            if not propagate_constraints(poss_grid, width, height,
                                          dirty_cells={(x, y)}):
                recovered = _wfc_backtrack(
                    stack, poss_grid, width, height, tile_weights)
                if not recovered:
                    success = False
                    break
                backtracks += 1
                cells_collapsed = len(stack)
                if backtracks > max_backtracks:
                    success = False
                    break
                continue

            cells_collapsed += 1
            if progress_callback:
                progress_callback(attempt, max_attempts,
                                  cells_collapsed, total_cells, backtracks)

        if not success:
            continue

        # Convert to final grid
        final_grid = [['' for _ in range(height)] for _ in range(width)]
        for x in range(width):
            for y in range(height):
                if len(poss_grid[x][y]) == 1:
                    final_grid[x][y] = next(iter(poss_grid[x][y]))
                else:
                    final_grid[x][y] = (random.choice(list(poss_grid[x][y]))
                                        if poss_grid[x][y] else '+')
        return final_grid

    return None


def _solve_stripe(width, height, tile_weights=None, max_attempts=200,
                  progress_callback=None, open_edges=None,
                  left_constraints=None, mask=None, spatial_map=None,
                  global_x_offset=0, global_width=None, global_height=None):
    """WFC solver for a single stripe with optional open edges and left-neighbor constraints.

    Args:
        open_edges: frozenset of directions ('W', 'E') that are open (connect to adjacent stripes)
        left_constraints: list of tile chars for the column to the left (len=height), or None
        mask: mask[x][y] where True=allowed, False=blocked (local stripe coords), or None
    """
    if mask is not None:
        tile_weights = _ensure_endcaps_for_mask(tile_weights)
    max_backtracks = width * height * 3  # more generous for stripes
    total_cells = width * height
    for attempt in range(1, max_attempts + 1):
        poss_grid = create_possibility_grid(width, height)

        # Apply mask: fix blocked cells to VOID_CHAR before constraints
        if mask is not None:
            for x in range(width):
                for y in range(height):
                    if not mask[x][y]:
                        poss_grid[x][y] = {VOID_CHAR}

        for x in range(width):
            for y in range(height):
                poss_grid[x][y] = get_constrained_possibilities(
                    poss_grid[x][y], x, y, width, height,
                    open_edges=open_edges)

        # Apply left-neighbor constraints: column 0 must be compatible
        # with the previous stripe's right column
        if left_constraints:
            for y in range(height):
                # Skip void cells — already fixed by mask, no neighbor compat needed
                if mask is not None and not mask[0][y]:
                    continue
                left_tile = left_constraints[y]
                # W neighbor compatibility
                poss_grid[0][y] = poss_grid[0][y] & COMPAT_TABLE[left_tile]['E']
                # NW diagonal neighbor (left_constraints[y-1] is NW of (0,y))
                if y > 0:
                    nw_tile = left_constraints[y - 1]
                    poss_grid[0][y] = poss_grid[0][y] & COMPAT_TABLE[nw_tile]['SE']
                # SW diagonal neighbor (left_constraints[y+1] is SW of (0,y))
                if y < height - 1:
                    sw_tile = left_constraints[y + 1]
                    poss_grid[0][y] = poss_grid[0][y] & COMPAT_TABLE[sw_tile]['NE']

                if len(poss_grid[0][y]) == 0:
                    break  # contradiction from constraints
            else:
                # No break — check passed
                if not propagate_constraints(poss_grid, width, height):
                    continue
                # Fall through to main loop
                pass
            if any(len(poss_grid[0][y]) == 0 for y in range(height)):
                continue  # retry attempt

        else:
            if not propagate_constraints(poss_grid, width, height):
                continue

        stack = []
        backtracks = 0
        cells_collapsed = 0
        success = True

        while True:
            cell = find_min_entropy_cell(poss_grid, width, height)
            if cell is None:
                break

            x, y = cell
            snapshot = [[c.copy() for c in row] for row in poss_grid]

            # Determine boundary boost for this cell
            boost_dirs = None
            if open_edges:
                bd = set()
                if x == 0 and 'W' in open_edges:
                    bd.add('W')
                if x == width - 1 and 'E' in open_edges:
                    bd.add('E')
                if bd:
                    boost_dirs = bd

            if not collapse_cell(poss_grid, x, y, tile_weights,
                                 width=width, height=height,
                                 boost_port_dirs=boost_dirs,
                                 spatial_map=spatial_map,
                                 global_x_offset=global_x_offset,
                                 global_width=global_width,
                                 global_height=global_height):
                recovered = _wfc_backtrack(
                    stack, poss_grid, width, height, tile_weights)
                if not recovered:
                    success = False
                    break
                backtracks += 1
                cells_collapsed = len(stack)
                if backtracks > max_backtracks:
                    success = False
                    break
                continue

            chosen = next(iter(poss_grid[x][y]))
            stack.append((snapshot, x, y, {chosen}))

            if not propagate_constraints(poss_grid, width, height,
                                          dirty_cells={(x, y)}):
                recovered = _wfc_backtrack(
                    stack, poss_grid, width, height, tile_weights)
                if not recovered:
                    success = False
                    break
                backtracks += 1
                cells_collapsed = len(stack)
                if backtracks > max_backtracks:
                    success = False
                    break
                continue

            cells_collapsed += 1
            if progress_callback:
                progress_callback(attempt, max_attempts,
                                  cells_collapsed, total_cells, backtracks)

        if not success:
            continue

        final_grid = [['' for _ in range(height)] for _ in range(width)]
        for x in range(width):
            for y in range(height):
                if len(poss_grid[x][y]) == 1:
                    final_grid[x][y] = next(iter(poss_grid[x][y]))
                else:
                    final_grid[x][y] = (random.choice(list(poss_grid[x][y]))
                                        if poss_grid[x][y] else '+')

        return final_grid

    return None


def wave_function_collapse_striped(width, height, stripe_width=8,
                                    tile_weights=None, max_attempts=200,
                                    progress_callback=None, stripe_offset=0,
                                    mask=None, spatial_map=None):
    """Generate large grids by solving vertical stripes left-to-right.

    Each stripe is an independent WFC problem with constrained left edge
    matching the previous stripe. Much more scalable than full-grid WFC.

    Stripe widths are randomized around the base stripe_width to avoid
    visible regular seams. stripe_offset shifts the first boundary, useful
    for multi-layer staggering so seams from different layers don't align.
    """
    final_grid = [['' for _ in range(height)] for _ in range(width)]

    # Build randomized stripe boundaries
    min_sw = max(4, stripe_width - 3)
    max_sw = stripe_width + 3
    stripe_starts = [0]

    # First stripe may be shorter if offset is specified (for multi-layer staggering)
    if stripe_offset > 0:
        first_width = max(3, min(stripe_offset, width))
        stripe_starts.append(min(first_width, width))

    while stripe_starts[-1] < width:
        sw = random.randint(min_sw, max_sw)
        next_start = min(stripe_starts[-1] + sw, width)
        # Don't leave a tiny sliver at the end
        remaining = width - next_start
        if 0 < remaining < 3:
            next_start = width
        stripe_starts.append(next_start)

    for i in range(len(stripe_starts) - 1):
        stripe_start = stripe_starts[i]
        stripe_end = stripe_starts[i + 1]
        sw = stripe_end - stripe_start

        # Determine which edges are open (connect to adjacent stripes)
        open_edges = set()
        if stripe_start > 0:
            open_edges.add('W')
        if stripe_end < width:
            open_edges.add('E')
        open_edges = frozenset(open_edges)

        # Left constraints from previous stripe's right column
        left_constraints = None
        if stripe_start > 0:
            left_constraints = [final_grid[stripe_start - 1][y]
                                for y in range(height)]

        # Wrap progress callback to report overall grid progress
        cells_offset = stripe_start * height
        total_cells = width * height

        def stripe_progress(attempt, max_attempts, cells, total, backtracks=0,
                            _offset=cells_offset):
            if progress_callback:
                progress_callback(attempt, max_attempts,
                                  _offset + cells, total_cells, backtracks)

        # Slice mask for this stripe (local coords)
        stripe_mask = None
        if mask is not None:
            stripe_mask = [mask[stripe_start + x] for x in range(sw)]

        stripe_grid = _solve_stripe(
            sw, height, tile_weights=tile_weights,
            max_attempts=max_attempts,
            progress_callback=stripe_progress,
            open_edges=open_edges,
            left_constraints=left_constraints,
            mask=stripe_mask,
            spatial_map=spatial_map,
            global_x_offset=stripe_start,
            global_width=width,
            global_height=height)

        if stripe_grid is None:
            return None  # stripe failed

        # Copy into final grid
        for x in range(sw):
            for y in range(height):
                final_grid[stripe_start + x][y] = stripe_grid[x][y]

    return final_grid


# ============================================================================
# TILE BUILDING & RENDERING
# ============================================================================

def add_tube_shading(group, style, sw):
    if style == 'none':
        return
    elif style == 'accent':
        group.append(draw.Line(22, -50, 22, 50, stroke='black', stroke_width=sw, fill='none'))
    elif style == 'double-wall':
        group.append(draw.Line(27, -50, 27, 50, stroke='black', stroke_width=sw, fill='none'))
    elif style == 'hatch':
        dy_per_dx = math.tan(math.radians(30))
        for y0 in range(-42, 43, 12):
            x1, y1 = -28, y0 - 28 * dy_per_dx
            x2, y2 = 28, y0 + 28 * dy_per_dx
            if y1 < -48:
                x1 = x1 + (-48 - y1) / dy_per_dx
                y1 = -48
            if y2 > 48:
                x2 = x2 - (y2 - 48) / dy_per_dx
                y2 = 48
            if y1 > 48 or y2 < -48:
                continue
            group.append(draw.Line(x1, y1, x2, y2, stroke='black', stroke_width=sw, fill='none'))


def add_corner_shading(group, style, sw):
    if style == 'none':
        return
    cx, cy = 40, 40
    if style == 'accent':
        group.append(draw.Arc(cx, cy, 62, 180, 270, cw=True, stroke='black', stroke_width=sw, fill='none'))
        group.append(draw.Line(43, -30, 43, 30, stroke='black', stroke_width=sw, fill='none'))
        group.append(draw.Line(-30, 43, 30, 43, stroke='black', stroke_width=sw, fill='none'))
    elif style == 'double-wall':
        group.append(draw.Arc(cx, cy, 67, 180, 270, cw=True, stroke='black', stroke_width=sw, fill='none'))
        group.append(draw.Line(42, -30, 42, 30, stroke='black', stroke_width=sw, fill='none'))
        group.append(draw.Line(-30, 42, 30, 42, stroke='black', stroke_width=sw, fill='none'))
    elif style == 'hatch':
        r_inner, r_outer = 12, 68
        num_hatches = 6
        for i in range(num_hatches):
            angle_deg = 185 + i * (80 / (num_hatches - 1))
            angle_rad = math.radians(angle_deg)
            x1 = cx + r_inner * math.cos(angle_rad)
            y1 = cy + r_inner * math.sin(angle_rad)
            x2 = cx + r_outer * math.cos(angle_rad)
            y2 = cy + r_outer * math.sin(angle_rad)
            group.append(draw.Line(x1, y1, x2, y2, stroke='black', stroke_width=sw, fill='none'))
        for y0 in [-15, 0, 15]:
            group.append(draw.Line(41, y0 - 3.5, 48, y0 + 3.5, stroke='black', stroke_width=sw, fill='none'))
        for x0 in [-15, 0, 15]:
            group.append(draw.Line(x0 - 3.5, 41, x0 + 3.5, 48, stroke='black', stroke_width=sw, fill='none'))


def build_corner_tile(stroke_width, shading_style, shading_stroke_width, use_fills=True):
    fill_color = 'white' if use_fills else 'none'
    group = draw.Group(id='l_corner', fill=fill_color)
    mask = draw.Mask()
    box = draw.Circle(40, 40, 10, fill='green')
    mask.append(box)
    group.append(draw.Arc(40, 40, 70, 180, 270, cw=True, stroke='black', stroke_width=stroke_width, fill=fill_color))
    group.append(draw.Lines(40, -30, -30, 40, 30, 40, 40, 30, fill=fill_color, stroke='none', mask=not mask))
    group.append(draw.Circle(40, 40, 10, fill=fill_color))
    group.append(draw.Arc(40, 40, 10, 180, 270, cw=True, stroke='black', stroke_width=stroke_width, fill='none'))
    group.append(draw.Rectangle(40, -35, 5, 70, fill=fill_color, stroke='black', stroke_width=stroke_width))
    group.append(draw.Rectangle(-35, 40, 70, 5, fill=fill_color, stroke='black', stroke_width=stroke_width))
    group.append(draw.Rectangle(46, -29, 5, 58, fill=fill_color, stroke='none'))
    group.append(draw.Line(45, 30, 50, 30, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Line(45, -30, 50, -30, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Rectangle(-29, 46, 58, 5, fill=fill_color, stroke='none'))
    group.append(draw.Line(30, 45, 30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Line(-30, 45, -30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    add_corner_shading(group, shading_style, shading_stroke_width)
    return group


def build_tube_tile(stroke_width, shading_style, shading_stroke_width, use_fills=True):
    fill_color = 'white' if use_fills else 'none'
    group = draw.Group(id='l_tube', fill='none')
    group.append(draw.Rectangle(-30, -50, 60, 100, fill=fill_color, stroke='none'))
    group.append(draw.Line(-30, -50, -30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Line(30, -50, 30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    add_tube_shading(group, shading_style, shading_stroke_width)
    return group


def draw_tube_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                   pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on the shadow side of a straight tube."""
    # Determine tube orientation and half-width
    tube_params = {
        '|': ('v', 30), '-': ('h', 30),   # Medium
        'i': ('v', 12), '=': ('h', 12),   # Narrow
        '!': ('v', 5), '.': ('h', 5),     # Tiny
    }
    if ch not in tube_params:
        return

    orient, half_width = tube_params[ch]

    if orient == 'v':
        tangent = (0, 1)
        normal_left = (-1, 0)
        normal_right = (1, 0)
    else:
        tangent = (1, 0)
        normal_left = (0, -1)
        normal_right = (0, 1)

    # Scale hatch params proportionally to pipe size (tuned for medium hw=30)
    scale = half_width / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right
    band_center_dist = half_width - band_offset - band_width / 2

    # Collect all hatch angles to draw (for crosshatching)
    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    num_hatches = max(1, int(90 / spacing))
    for base_angle in angles_to_draw:
        for i in range(num_hatches + 1):
            t = -45 + i * (90.0 / num_hatches)
            t += random.uniform(-jitter_pos, jitter_pos)

            base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
            base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            # Apply band width jitter
            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            # Draw the hatch (with optional wiggle for curved line)
            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


def draw_corner_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                     pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on the shadow portion of a corner piece."""
    rotations = {'r': 0, '7': 90, 'j': 180, 'L': 270}
    rot_deg = rotations[ch]

    band_width = params['band_width']
    band_offset = params['band_offset']
    spacing = params['spacing']
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos']
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0)

    local_center = (40, 40)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    r_outer = 70
    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Collect all hatch angles to draw (for crosshatching)
    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Calculate num_samples based on spacing (arc length at band_mid radius)
    band_outer = r_outer - band_offset
    band_inner = band_outer - band_width
    band_mid = (band_inner + band_outer) / 2
    arc_length = (math.pi / 2) * band_mid  # 90 degrees of arc
    num_samples = max(4, int(arc_length / spacing))

    for base_angle in angles_to_draw:
        for i in range(num_samples):
            frac = (i + 0.5) / num_samples
            angle_deg = arc_start + frac * (arc_end - arc_start)
            angle_rad = math.radians(angle_deg)

            outward_normal = (math.cos(angle_rad), math.sin(angle_rad))
            if _dot(outward_normal, light_dir) >= 0:
                continue

            base_x = arc_cx + math.cos(angle_rad) * band_mid
            base_y = arc_cy + math.sin(angle_rad) * band_mid

            # Hatch direction: radial (perpendicular to arc), rotated by hatch_angle
            angle_jitter = random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(outward_normal, base_angle + angle_jitter)

            # Apply band width jitter
            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            # Position jitter uses tangent direction (along the arc)
            tangent = (-math.sin(angle_rad), math.cos(angle_rad))

            pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
            base_x += tangent[0] * pos_jitter_val
            base_y += tangent[1] * pos_jitter_val

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            # Draw the hatch (with optional wiggle for curved line)
            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


def draw_sized_corner_directional_shading(drawing, ch, xloc, yloc, half_width, light_dir, sw, params,
                                          pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on the shadow portion of a sized corner piece."""
    # Get rotation based on corner type (strip size prefix)
    base_ch = ch[-1] if ch[0] in ('n', 't') else ch  # 'nr' -> 'r', 'tL' -> 'L'
    rotations = {'r': 0, '7': 90, 'j': 180, 'L': 270}
    rot_deg = rotations[base_ch]

    # Scale hatch params proportionally to pipe size (tuned for medium hw=30)
    scale = half_width / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    # Calculate arc geometry based on half_width (same as get_sized_corner_polygon)
    inner_radius = half_width / 3
    center_offset = half_width + inner_radius
    outer_radius = 2 * half_width + inner_radius

    local_center = (center_offset, center_offset)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Collect all hatch angles to draw (for crosshatching)
    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Calculate num_samples based on spacing (arc length at band_mid radius)
    band_outer = outer_radius - band_offset
    band_inner = band_outer - band_width
    band_mid = (band_inner + band_outer) / 2

    # Clamp band_mid to valid range
    band_mid = max(inner_radius + 1, min(band_mid, outer_radius - 1))

    arc_length = (math.pi / 2) * band_mid  # 90 degrees of arc
    num_samples = max(4, int(arc_length / spacing))

    for base_angle in angles_to_draw:
        for i in range(num_samples):
            frac = (i + 0.5) / num_samples
            angle_deg = arc_start + frac * (arc_end - arc_start)
            angle_rad = math.radians(angle_deg)

            outward_normal = (math.cos(angle_rad), math.sin(angle_rad))
            if _dot(outward_normal, light_dir) >= 0:
                continue

            base_x = arc_cx + math.cos(angle_rad) * band_mid
            base_y = arc_cy + math.sin(angle_rad) * band_mid

            # Hatch direction: radial (perpendicular to arc), rotated by hatch_angle
            angle_jitter = random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(outward_normal, base_angle + angle_jitter)

            # Apply band width jitter
            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            # Position jitter uses tangent direction (along the arc)
            tangent = (-math.sin(angle_rad), math.cos(angle_rad))

            pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
            base_x += tangent[0] * pos_jitter_val
            base_y += tangent[1] * pos_jitter_val

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


def draw_dodge_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                    pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on the shadow side of a dodge tile.

    Two sections (top half and bottom half) with different tangent vectors
    following the wall angle of each Z-segment.
    """
    if ch not in _DODGE_PARAMS:
        return

    orient, sign, hw = _DODGE_PARAMS[ch]
    offset = sign * hw / 2
    scale = hw / 30

    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Two sections with different wall angles
    if orient == 'v':
        sections = [
            ((0, -50), (offset, 0)),     # top half
            ((offset, 0), (0, 50)),      # bottom half
        ]
    else:
        sections = [
            ((-50, 0), (0, offset)),     # left half
            ((0, offset), (50, 0)),      # right half
        ]

    band_center_dist = hw - band_offset - band_width / 2

    for sec_start, sec_end in sections:
        dx = sec_end[0] - sec_start[0]
        dy = sec_end[1] - sec_start[1]
        tangent = _normalize((dx, dy))

        # Normals perpendicular to tangent
        normal_left = (-tangent[1], tangent[0])
        normal_right = (tangent[1], -tangent[0])

        dot_left = _dot(normal_left, light_dir)
        dot_right = _dot(normal_right, light_dir)
        shadow_normal = normal_left if dot_left < dot_right else normal_right

        section_length = math.sqrt(dx**2 + dy**2)
        num_hatches = max(1, int(section_length / spacing))

        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                frac = (i + 0.5) / (num_hatches + 1)

                cx = xloc + sec_start[0] + frac * dx
                cy = yloc + sec_start[1] + frac * dy

                base_x = cx + shadow_normal[0] * band_center_dist
                base_y = cy + shadow_normal[1] * band_center_dist

                pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
                base_x += tangent[0] * pos_jitter_val
                base_y += tangent[1] * pos_jitter_val

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


def draw_sbend_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                    pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on the shadow side of an S-bend tile.

    Two sections (top/bottom or left/right halves) with tangent vectors
    following the S-curve slope at each section's midpoint.
    """
    if ch not in _SBEND_PARAMS:
        return

    orient, sign, hw = _SBEND_PARAMS[ch]
    offset = sign * hw / 2
    scale = hw / 30

    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Two sections — same as dodge but with S-curve tangent at midpoint
    if orient == 'v':
        sections = [
            ((0, -50), (offset, 0)),     # top half
            ((offset, 0), (0, 50)),      # bottom half
        ]
    else:
        sections = [
            ((-50, 0), (0, offset)),     # left half
            ((0, offset), (50, 0)),      # right half
        ]

    band_center_dist = hw - band_offset - band_width / 2

    for sec_start, sec_end in sections:
        dx = sec_end[0] - sec_start[0]
        dy = sec_end[1] - sec_start[1]
        tangent = _normalize((dx, dy))

        normal_left = (-tangent[1], tangent[0])
        normal_right = (tangent[1], -tangent[0])

        dot_left = _dot(normal_left, light_dir)
        dot_right = _dot(normal_right, light_dir)
        shadow_normal = normal_left if dot_left < dot_right else normal_right

        section_length = math.sqrt(dx**2 + dy**2)
        num_hatches = max(1, int(section_length / spacing))

        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                frac = (i + 0.5) / (num_hatches + 1)

                cx = xloc + sec_start[0] + frac * dx
                cy = yloc + sec_start[1] + frac * dy

                base_x = cx + shadow_normal[0] * band_center_dist
                base_y = cy + shadow_normal[1] * band_center_dist

                pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
                base_x += tangent[0] * pos_jitter_val
                base_y += tangent[1] * pos_jitter_val

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


def draw_long_radius_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                          pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a long-radius elbow.

    Samples along the wide arc to place hatches on the shadow side,
    using the long-radius arc parameters.
    """
    if ch not in _LONG_RADIUS_PARAMS:
        return

    base, hw = _LONG_RADIUS_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]
    scale = hw / 30

    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    inner_r, center_off, outer_r = _long_radius_arc_params(hw)

    local_center = (center_off, center_off)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    band_outer = outer_r - band_offset
    band_inner = band_outer - band_width
    band_mid = max(inner_r + 1, min((band_inner + band_outer) / 2, outer_r - 1))

    arc_length = (math.pi / 2) * band_mid
    num_samples = max(4, int(arc_length / spacing))

    for base_angle in angles_to_draw:
        for i in range(num_samples):
            frac = (i + 0.5) / num_samples
            angle_deg = arc_start + frac * (arc_end - arc_start)
            angle_rad = math.radians(angle_deg)

            outward_normal = (math.cos(angle_rad), math.sin(angle_rad))
            if _dot(outward_normal, light_dir) >= 0:
                continue

            base_x = arc_cx + math.cos(angle_rad) * band_mid
            base_y = arc_cy + math.sin(angle_rad) * band_mid

            tangent = (-math.sin(angle_rad), math.cos(angle_rad))

            pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
            base_x += tangent[0] * pos_jitter_val
            base_y += tangent[1] * pos_jitter_val

            angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(tangent, angle)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


def draw_segmented_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                        pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a segmented elbow.

    Two sections (vertical half and horizontal half) with polygon clipping
    to handle the faceted transition region.
    """
    if ch not in _SEGMENTED_PARAMS:
        return

    base, hw = _SEGMENTED_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]
    scale = hw / 30

    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    # Two sections: vertical half and horizontal half (same as chamfer)
    vert_start = tp(0, 50)
    vert_end = tp(0, 0)
    horiz_start = tp(0, 0)
    horiz_end = tp(50, 0)

    sections = [(vert_start, vert_end), (horiz_start, horiz_end)]
    band_center_dist = hw - band_offset - band_width / 2

    for sec_start, sec_end in sections:
        dx = sec_end[0] - sec_start[0]
        dy = sec_end[1] - sec_start[1]
        tangent = _normalize((dx, dy))

        normal_left = (-tangent[1], tangent[0])
        normal_right = (tangent[1], -tangent[0])

        dot_left = _dot(normal_left, light_dir)
        dot_right = _dot(normal_right, light_dir)
        shadow_normal = normal_left if dot_left < dot_right else normal_right

        section_length = math.sqrt(dx**2 + dy**2)
        num_hatches = max(1, int(section_length / spacing))

        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                frac = (i + 0.5) / (num_hatches + 1)

                cx = sec_start[0] + frac * dx
                cy = sec_start[1] + frac * dy

                base_x = cx + shadow_normal[0] * band_center_dist
                base_y = cy + shadow_normal[1] * band_center_dist

                pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
                base_x += tangent[0] * pos_jitter_val
                base_y += tangent[1] * pos_jitter_val

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


def draw_chamfer_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                      pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a chamfered corner.

    Two straight sections (vertical half and horizontal half) with the diagonal
    transition handled by polygon clipping.
    """
    if ch not in _CHAMFER_PARAMS:
        return

    base, hw = _CHAMFER_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Two sections in local coords (base r orientation):
    # Vertical half: (0, 50) -> (0, 0), tangent (0, -1)
    # Horizontal half: (0, 0) -> (50, 0), tangent (1, 0)
    local_sections = [
        ((0, 50), (0, 0), (0, -1), (-1, 0), (1, 0)),
        ((0, 0), (50, 0), (1, 0), (0, -1), (0, 1)),
    ]

    band_center_dist = hw - band_offset - band_width / 2

    for local_start, local_end, local_tangent, local_nl, local_nr in local_sections:
        tangent = _rotate_vec(local_tangent, rot_deg)
        normal_left = _rotate_vec(local_nl, rot_deg)
        normal_right = _rotate_vec(local_nr, rot_deg)

        start_r = _rotate_vec(local_start, rot_deg)
        end_r = _rotate_vec(local_end, rot_deg)
        start_w = (xloc + start_r[0], yloc + start_r[1])
        end_w = (xloc + end_r[0], yloc + end_r[1])

        dot_left = _dot(normal_left, light_dir)
        dot_right = _dot(normal_right, light_dir)
        shadow_normal = normal_left if dot_left < dot_right else normal_right

        num_hatches = max(1, int(50 / spacing))

        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                frac = (i + 0.5) / (num_hatches + 1)

                cx = start_w[0] + frac * (end_w[0] - start_w[0])
                cy = start_w[1] + frac * (end_w[1] - start_w[1])

                base_x = cx + shadow_normal[0] * band_center_dist
                base_y = cy + shadow_normal[1] * band_center_dist

                pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
                base_x += tangent[0] * pos_jitter_val
                base_y += tangent[1] * pos_jitter_val

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


def draw_teardrop_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                       pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on a teardrop elbow.

    Samples along the arc sweep using the outer arc's radial direction for
    surface normal, placing hatches on the shadow side.
    """
    if ch not in _TEARDROP_PARAMS:
        return

    base, hw = _TEARDROP_PARAMS[ch]
    rot_deg = _CORNER_ROTATIONS[base]

    scale = hw / 30
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing'] * scale
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    c_o = hw * 1.0
    r_o = c_o + hw

    def tp(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    oc = tp(c_o, c_o)

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Hatch band positioned relative to outer arc
    band_outer = r_o - band_offset
    band_inner = band_outer - band_width
    band_mid = (band_inner + band_outer) / 2
    band_mid = max(1, min(band_mid, r_o - 1))

    arc_length = (math.pi / 2) * band_mid
    num_samples = max(4, int(arc_length / spacing))

    for base_angle in angles_to_draw:
        for i in range(num_samples):
            frac = (i + 0.5) / num_samples
            angle_deg = arc_start + frac * (arc_end - arc_start)
            angle_rad = math.radians(angle_deg)

            outward_normal = (math.cos(angle_rad), math.sin(angle_rad))
            if _dot(outward_normal, light_dir) >= 0:
                continue

            base_x = oc[0] + math.cos(angle_rad) * band_mid
            base_y = oc[1] + math.sin(angle_rad) * band_mid

            angle_jitter = random.uniform(-jitter_angle, jitter_angle)
            hatch_dir = _rotate_vec(outward_normal, base_angle + angle_jitter)

            width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
            half_len = (band_width * 0.6) * width_mult

            tangent = (-math.sin(angle_rad), math.cos(angle_rad))
            pos_jitter_val = random.uniform(-jitter_pos, jitter_pos)
            base_x += tangent[0] * pos_jitter_val
            base_y += tangent[1] * pos_jitter_val

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                            pipe_polygon, occlusion_polygon)


def draw_reducer_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                     pipe_polygon=None, occlusion_polygon=None):
    """Draw directional hatch marks on the shadow side of a reducer tile.

    The reducer has three sections: straight start, taper, straight end.
    Hatches follow the wall angle in the taper section and scale proportionally
    to the local pipe size.
    """
    reducer_info = {
        'Rv': ('v', 30, 12), 'RV': ('v', 12, 30),
        'Tv': ('v', 12, 5),  'TV': ('v', 5, 12),
        'Rh': ('h', 30, 12), 'RH': ('h', 12, 30),
        'Th': ('h', 12, 5),  'TH': ('h', 5, 12),
    }
    if ch not in reducer_info:
        return

    orient, hw_start, hw_end = reducer_info[ch]

    raw_band_width = params['band_width']
    raw_band_offset = params['band_offset']
    raw_spacing = params['spacing']
    hatch_angle = params['angle']
    raw_jitter_pos = params['jitter_pos']
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    raw_wiggle = params.get('wiggle', 0.0)

    if orient == 'v':
        tangent = (0, 1)
        normal_left = (-1, 0)
        normal_right = (1, 0)
    else:
        tangent = (1, 0)
        normal_left = (0, -1)
        normal_right = (0, 1)

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right

    angles_to_draw = [hatch_angle]
    if params.get('crosshatch'):
        angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

    # Map hw to t-axis direction:
    #   Vertical: N(-t)=start, S(+t)=end
    #   Horizontal: W(-t)=end, E(+t)=start
    if orient == 'v':
        hw_neg = hw_start   # at negative-t end (North)
        hw_pos = hw_end     # at positive-t end (South)
    else:
        hw_neg = hw_end     # at negative-t end (West)
        hw_pos = hw_start   # at positive-t end (East)

    # Three sections: straight, taper, straight
    # Inset from flange zones (flanges at t=±20, band ±2) so angled hatches
    # don't spill across the flange boundaries
    flange_margin = 4  # flange_width(2) + buffer for angled hatches
    # Each: (t_start, t_end, hw_at_t_start, hw_at_t_end)
    sections = [
        (-45, -20 - flange_margin, hw_neg, hw_neg),   # straight (negative-t end)
        (-20 + flange_margin,  20 - flange_margin, hw_neg, hw_pos),   # taper
        ( 20 + flange_margin,  45, hw_pos, hw_pos),   # straight (positive-t end)
    ]

    # Compute taper wall tangent for hatch direction in the taper section
    dhw = hw_pos - hw_neg  # change in half_width across the taper
    if orient == 'v':
        # Shadow on right: wall (dhw, 40); shadow on left: wall (-dhw, 40)
        wall_dx = dhw if shadow_normal[0] > 0 else -dhw
        taper_tangent = _normalize((wall_dx, 40))
    else:
        # Shadow on bottom: wall (40, dhw); shadow on top: wall (40, -dhw)
        wall_dy = dhw if shadow_normal[1] > 0 else -dhw
        taper_tangent = _normalize((40, wall_dy))

    for base_angle in angles_to_draw:
        for (t_start, t_end, sec_hw0, sec_hw1) in sections:
            sec_len = t_end - t_start
            is_taper = (sec_hw0 != sec_hw1)

            # Spacing scaled to average hw in this section
            sec_avg_hw = (sec_hw0 + sec_hw1) / 2
            sec_spacing = raw_spacing * (sec_avg_hw / 30)
            num_sec = max(1, int(sec_len / sec_spacing))

            for i in range(num_sec + 1):
                t = t_start + i * (sec_len / num_sec)

                # Half-width at this position
                if is_taper:
                    frac = (t - t_start) / sec_len
                    half_width = sec_hw0 + frac * (sec_hw1 - sec_hw0)
                else:
                    half_width = sec_hw0

                scale = half_width / 30
                band_width = raw_band_width * scale
                band_offset = raw_band_offset * scale
                jitter_pos = raw_jitter_pos * scale
                wiggle = raw_wiggle * scale

                band_center_dist = half_width - band_offset - band_width / 2
                band_center_dist = max(1, band_center_dist)

                t_jittered = t + random.uniform(-jitter_pos, jitter_pos)

                base_x = xloc + tangent[0] * t_jittered + shadow_normal[0] * band_center_dist
                base_y = yloc + tangent[1] * t_jittered + shadow_normal[1] * band_center_dist

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                local_tangent = taper_tangent if is_taper else tangent
                hatch_dir = _rotate_vec(local_tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


def draw_cross_directional_shading(drawing, xloc, yloc, light_dir, sw, params,
                                    pipe_polygon=None, occlusion_polygon=None, half_width=30):
    """Draw directional hatch marks on the shadow sides of a cross junction."""
    scale = half_width / 30.0
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing']
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    # Draw shading on each arm based on light direction
    # Each arm is treated like a tube segment

    arms = [
        # (arm_direction, tangent, extent_start, extent_end)
        ('N', (0, -1), -half_width, -50),  # North arm: y from -hw to -50
        ('S', (0, 1), half_width, 50),     # South arm: y from hw to 50
        ('E', (1, 0), half_width, 50),     # East arm: x from hw to 50
        ('W', (-1, 0), -half_width, -50),  # West arm: x from -hw to -50
    ]

    for arm_dir, tangent, start, end in arms:
        # Determine normals based on arm direction
        if arm_dir in ('N', 'S'):
            normal_left = (-1, 0)
            normal_right = (1, 0)
        else:  # E, W
            normal_left = (0, -1)
            normal_right = (0, 1)

        # Determine shadow side
        dot_left = _dot(normal_left, light_dir)
        dot_right = _dot(normal_right, light_dir)
        shadow_normal = normal_left if dot_left < dot_right else normal_right

        band_center_dist = half_width - band_offset - band_width / 2
        band_center_dist = max(1, band_center_dist)

        # Calculate arm center and draw hatches
        arm_length = abs(end - start)
        num_hatches = max(1, int(arm_length / spacing))

        angles_to_draw = [hatch_angle]
        if params.get('crosshatch'):
            angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                t = i / num_hatches
                pos = start + t * (end - start)

                if arm_dir in ('N', 'S'):
                    base_x = xloc + shadow_normal[0] * band_center_dist
                    base_y = yloc + pos
                else:
                    base_x = xloc + pos
                    base_y = yloc + shadow_normal[1] * band_center_dist

                base_x += random.uniform(-jitter_pos, jitter_pos)
                base_y += random.uniform(-jitter_pos, jitter_pos)

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


def draw_tee_directional_shading(drawing, ch, xloc, yloc, light_dir, sw, params,
                                  pipe_polygon=None, occlusion_polygon=None, half_width=30):
    """Draw directional hatch marks on the shadow sides of a tee junction."""
    rotations = {'T': 0, 'B': 180, 'E': 90, 'W': 270}
    rot_deg = rotations.get(ch)
    if rot_deg is None:
        return

    scale = half_width / 30.0
    band_width = params['band_width'] * scale
    band_offset = params['band_offset'] * scale
    spacing = params['spacing']
    hatch_angle = params['angle']
    jitter_pos = params['jitter_pos'] * scale
    jitter_angle = params['jitter_angle']
    band_width_jitter = params.get('band_width_jitter', 0.0)
    wiggle = params.get('wiggle', 0.0) * scale

    def transform_vec(vx, vy):
        return _rotate_vec((vx, vy), rot_deg)

    # Base T shape has 3 arms: N, E, W (closed S)
    # We rotate based on ch
    base_arms = [
        ((0, -1), (-half_width, -50)),   # North arm: y from -hw to -50
        ((1, 0), (half_width, 50)),      # East arm: x from hw to 50
        ((-1, 0), (-50, -half_width)),   # West arm: x from -50 to -hw
    ]

    for tangent, (start, end) in base_arms:
        # Rotate tangent
        rot_tangent = transform_vec(*tangent)

        # Determine normals (perpendicular to tangent in 2D)
        if abs(tangent[0]) > 0.5:  # Horizontal arm
            normal_left = (0, -1)
            normal_right = (0, 1)
        else:  # Vertical arm
            normal_left = (-1, 0)
            normal_right = (1, 0)

        rot_normal_left = transform_vec(*normal_left)
        rot_normal_right = transform_vec(*normal_right)

        # Determine shadow side
        dot_left = _dot(rot_normal_left, light_dir)
        dot_right = _dot(rot_normal_right, light_dir)
        shadow_normal = rot_normal_left if dot_left < dot_right else rot_normal_right

        band_center_dist = half_width - band_offset - band_width / 2
        band_center_dist = max(1, band_center_dist)

        arm_length = abs(end - start)
        num_hatches = max(1, int(arm_length / spacing))

        angles_to_draw = [hatch_angle]
        if params.get('crosshatch'):
            angles_to_draw.append(hatch_angle + params.get('crosshatch_angle', 90))

        for base_angle in angles_to_draw:
            for i in range(num_hatches + 1):
                t = i / num_hatches
                pos = start + t * (end - start)

                # Calculate base position in local coords
                if abs(tangent[0]) > 0.5:  # Horizontal arm
                    local_x = pos
                    local_y = 0
                else:  # Vertical arm
                    local_x = 0
                    local_y = pos

                # Transform to world coords
                rot_pos = transform_vec(local_x, local_y)
                base_x = xloc + rot_pos[0] + shadow_normal[0] * band_center_dist
                base_y = yloc + rot_pos[1] + shadow_normal[1] * band_center_dist

                base_x += random.uniform(-jitter_pos, jitter_pos)
                base_y += random.uniform(-jitter_pos, jitter_pos)

                angle = base_angle + random.uniform(-jitter_angle, jitter_angle)
                hatch_dir = _rotate_vec(rot_tangent, angle)

                width_mult = 1.0 + random.uniform(-band_width_jitter, band_width_jitter)
                half_len = (band_width * 0.6) * width_mult

                x1 = base_x - hatch_dir[0] * half_len
                y1 = base_y - hatch_dir[1] * half_len
                x2 = base_x + hatch_dir[0] * half_len
                y2 = base_y + hatch_dir[1] * half_len

                _draw_hatch_line(drawing, x1, y1, x2, y2, hatch_dir, wiggle, sw,
                                pipe_polygon, occlusion_polygon)


# ============================================================================
# CLIPPED SHADING FOR OTHER STYLES (accent, hatch, double-wall)
# ============================================================================

def draw_tube_shading_clipped(drawing, ch, xloc, yloc, style, sw, occlusion_poly):
    """Draw shading on a straight tube with clipping."""
    if style == 'none':
        return

    if ch == '|':
        # Vertical tube
        if style == 'accent':
            clip_and_draw_line(drawing, xloc + 22, yloc - 50, xloc + 22, yloc + 50, occlusion_poly, sw)
        elif style == 'double-wall':
            clip_and_draw_line(drawing, xloc + 27, yloc - 50, xloc + 27, yloc + 50, occlusion_poly, sw)
        elif style == 'hatch':
            dy_per_dx = math.tan(math.radians(30))
            for y0 in range(-42, 43, 12):
                x1, y1 = xloc - 28, yloc + y0 - 28 * dy_per_dx
                x2, y2 = xloc + 28, yloc + y0 + 28 * dy_per_dx
                if y1 < yloc - 48:
                    x1 = x1 + (yloc - 48 - y1) / dy_per_dx
                    y1 = yloc - 48
                if y2 > yloc + 48:
                    x2 = x2 - (y2 - (yloc + 48)) / dy_per_dx
                    y2 = yloc + 48
                if y1 > yloc + 48 or y2 < yloc - 48:
                    continue
                clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    elif ch == '-':
        # Horizontal tube - rotate 90 degrees
        if style == 'accent':
            clip_and_draw_line(drawing, xloc - 50, yloc + 22, xloc + 50, yloc + 22, occlusion_poly, sw)
        elif style == 'double-wall':
            clip_and_draw_line(drawing, xloc - 50, yloc + 27, xloc + 50, yloc + 27, occlusion_poly, sw)
        elif style == 'hatch':
            dx_per_dy = math.tan(math.radians(30))
            for x0 in range(-42, 43, 12):
                y1, x1 = yloc - 28, xloc + x0 - 28 * dx_per_dy
                y2, x2 = yloc + 28, xloc + x0 + 28 * dx_per_dy
                if x1 < xloc - 48:
                    y1 = y1 + (xloc - 48 - x1) / dx_per_dy
                    x1 = xloc - 48
                if x2 > xloc + 48:
                    y2 = y2 - (x2 - (xloc + 48)) / dx_per_dy
                    x2 = xloc + 48
                if x1 > xloc + 48 or x2 < xloc - 48:
                    continue
                clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


def draw_corner_shading_clipped(drawing, ch, xloc, yloc, style, sw, occlusion_poly):
    """Draw shading on a corner pipe with clipping."""
    if style == 'none':
        return

    rotations = {'r': 0, '7': 90, 'j': 180, 'L': 270}
    rot_deg = rotations.get(ch)
    if rot_deg is None:
        return

    # Arc center in world coords
    local_center = (40, 40)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    def transform_point(px, py):
        rx, ry = _rotate_vec((px, py), rot_deg)
        return (xloc + rx, yloc + ry)

    if style == 'accent':
        # Draw accent arc at r=62
        clip_and_draw_arc(drawing, arc_cx, arc_cy, 62, arc_start, arc_end, occlusion_poly, sw)
        # Draw accent lines on stubs
        x1, y1 = transform_point(43, -30)
        x2, y2 = transform_point(43, 30)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
        x1, y1 = transform_point(-30, 43)
        x2, y2 = transform_point(30, 43)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    elif style == 'double-wall':
        # Draw double-wall arc at r=67
        clip_and_draw_arc(drawing, arc_cx, arc_cy, 67, arc_start, arc_end, occlusion_poly, sw)
        # Draw double-wall lines on stubs
        x1, y1 = transform_point(42, -30)
        x2, y2 = transform_point(42, 30)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)
        x1, y1 = transform_point(-30, 42)
        x2, y2 = transform_point(30, 42)
        clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    elif style == 'hatch':
        # Radial hatches on the arc
        r_inner, r_outer = 12, 68
        num_hatches = 6
        for i in range(num_hatches):
            angle_deg = arc_start + 5 + i * (80 / (num_hatches - 1))
            angle_rad = math.radians(angle_deg)
            x1 = arc_cx + r_inner * math.cos(angle_rad)
            y1 = arc_cy + r_inner * math.sin(angle_rad)
            x2 = arc_cx + r_outer * math.cos(angle_rad)
            y2 = arc_cy + r_outer * math.sin(angle_rad)
            clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        # Hatches on vertical stub
        for y0 in [-15, 0, 15]:
            x1, y1 = transform_point(41, y0 - 3.5)
            x2, y2 = transform_point(48, y0 + 3.5)
            clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

        # Hatches on horizontal stub
        for x0 in [-15, 0, 15]:
            x1, y1 = transform_point(x0 - 3.5, 41)
            x2, y2 = transform_point(x0 + 3.5, 48)
            clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)


# ============================================================================
# PLOTTER-FRIENDLY PIPE DECORATIONS
# ============================================================================

DECORATION_ELIGIBLE_SHAPES = {'straight', 'reducer'}

DECORATION_MIN_HW = {
    'coupling_box': 5,
    'round_dial': 12,
    'valve_handle': 12,
    'bolt_heads': 5,
    'inspection_window': 12,
    'flow_arrow': 12,
}

DECORATION_TYPES = list(DECORATION_MIN_HW.keys())


def _deco_scale(hw, extra=1.0):
    return (hw / 30.0) * extra


def is_decoration_eligible(ch):
    return TILE_SHAPE.get(ch) in DECORATION_ELIGIBLE_SHAPES


def select_decorated_tiles(grid, density=0.10, seed=None):
    """Select which tiles receive decorations and what type.

    Returns dict {(x, y): decoration_type_string}.
    """
    width = len(grid)
    height = len(grid[0]) if width > 0 else 0

    eligible = []
    for x in range(width):
        for y in range(height):
            if is_decoration_eligible(grid[x][y]):
                eligible.append((x, y))

    if not eligible:
        return {}

    rng = random.Random(seed if seed is not None else hash(str(grid)))
    count = max(1, int(len(eligible) * density))
    chosen = rng.sample(eligible, min(count, len(eligible)))

    result = {}
    for pos in chosen:
        result[pos] = rng.choice(DECORATION_TYPES)
    return result


_TUBE_PARAMS = {
    '|': ('v', 30), '-': ('h', 30),
    'i': ('v', 12), '=': ('h', 12),
    '!': ('v', 5),  '.': ('h', 5),
}

# (orient, wide_size, wide_end_sign)
# wide_end_sign: direction along axis where the wide section is
# Vertical: start is top (y<0), so Rv (m->n) has wide at top = -1
# Horizontal: start is right (x>0), so Rh (m->n) has wide at right = +1
_REDUCER_ORIENT = {
    'Rv': ('v', 'm', -1), 'RV': ('v', 'm', +1),
    'Tv': ('v', 'n', -1), 'TV': ('v', 'n', +1),
    'Rh': ('h', 'm', +1), 'RH': ('h', 'm', -1),
    'Th': ('h', 'n', +1), 'TH': ('h', 'n', -1),
}


def get_straight_placement(ch, xloc, yloc, light_dir, rng):
    """Return (cx, cy, tangent, normal, hw) for decoration on a straight tile."""
    orient, hw = _TUBE_PARAMS[ch]

    if orient == 'v':
        tangent = (0, 1)
        normal_left, normal_right = (-1, 0), (1, 0)
    else:
        tangent = (1, 0)
        normal_left, normal_right = (0, -1), (0, 1)

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right

    t_offset = rng.uniform(-35, 35)
    wall_inset = 2
    cx = xloc + tangent[0] * t_offset + shadow_normal[0] * (hw - wall_inset)
    cy = yloc + tangent[1] * t_offset + shadow_normal[1] * (hw - wall_inset)

    return cx, cy, tangent, shadow_normal, hw


def get_reducer_placement(ch, xloc, yloc, light_dir, rng):
    """Return (cx, cy, tangent, normal, hw) for decoration on a reducer tile."""
    orient, wide_size, wide_sign = _REDUCER_ORIENT[ch]
    hw = PIPE_HALF_WIDTHS[wide_size]

    if orient == 'v':
        tangent = (0, 1)
        normal_left, normal_right = (-1, 0), (1, 0)
    else:
        tangent = (1, 0)
        normal_left, normal_right = (0, -1), (0, 1)

    # Place on the wide (non-tapered) section
    t_offset = wide_sign * rng.uniform(25, 45)

    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)
    shadow_normal = normal_left if dot_left < dot_right else normal_right

    wall_inset = 2
    cx = xloc + tangent[0] * t_offset + shadow_normal[0] * (hw - wall_inset)
    cy = yloc + tangent[1] * t_offset + shadow_normal[1] * (hw - wall_inset)

    return cx, cy, tangent, shadow_normal, hw


def draw_decoration_coupling_box(drawing, cx, cy, tangent, normal, hw,
                                 pipe_poly, occlusion_poly, sw, rng,
                                 extra_scale=1.0):
    """Small rectangular collar straddling the pipe wall."""
    scale = _deco_scale(hw, extra_scale)
    w = 8 * scale
    h = 6 * scale

    corners = []
    for st, sn in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        px = cx + tangent[0] * (st * w / 2) + normal[0] * (sn * h / 2)
        py = cy + tangent[1] * (st * w / 2) + normal[1] * (sn * h / 2)
        corners.append((px, py))

    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        clip_and_draw_line_inside(drawing, x1, y1, x2, y2,
                                  pipe_poly, occlusion_poly, sw)


def draw_decoration_round_dial(drawing, cx, cy, tangent, normal, hw,
                               pipe_poly, occlusion_poly, sw, rng,
                               extra_scale=1.0):
    """Small circle with a tick mark on the pipe wall."""
    scale = _deco_scale(hw, extra_scale)
    r = 4 * scale

    clip_and_draw_arc_inside(drawing, cx, cy, r, 0, 360,
                             pipe_poly, occlusion_poly, sw, num_segments=16)

    tick_angle = rng.uniform(0, 360)
    tick_rad = math.radians(tick_angle)
    tx = cx + r * 0.8 * math.cos(tick_rad)
    ty = cy + r * 0.8 * math.sin(tick_rad)
    clip_and_draw_line_inside(drawing, cx, cy, tx, ty,
                              pipe_poly, occlusion_poly, sw)


def draw_decoration_valve_handle(drawing, cx, cy, tangent, normal, hw,
                                 pipe_poly, occlusion_poly, sw, rng,
                                 extra_scale=1.0):
    """Small circle with a perpendicular crossbar."""
    scale = _deco_scale(hw, extra_scale)
    r = 3.5 * scale
    bar_len = 5 * scale

    clip_and_draw_arc_inside(drawing, cx, cy, r, 0, 360,
                             pipe_poly, occlusion_poly, sw, num_segments=16)

    # Crossbar along tangent
    x1 = cx - tangent[0] * bar_len
    y1 = cy - tangent[1] * bar_len
    x2 = cx + tangent[0] * bar_len
    y2 = cy + tangent[1] * bar_len
    clip_and_draw_line_inside(drawing, x1, y1, x2, y2,
                              pipe_poly, occlusion_poly, sw)


def draw_decoration_bolt_heads(drawing, cx, cy, tangent, normal, hw,
                               pipe_poly, occlusion_poly, sw, rng,
                               extra_scale=1.0):
    """2-4 small diamond shapes spaced along the tangent direction."""
    scale = _deco_scale(hw, extra_scale)
    bolt_size = 2 * scale
    spacing = 6 * scale

    if hw >= 30:
        count = rng.choice([3, 4])
    elif hw >= 12:
        count = rng.choice([2, 3])
    else:
        count = 2

    total_span = (count - 1) * spacing
    start_t = -total_span / 2

    for i in range(count):
        t = start_t + i * spacing
        bx = cx + tangent[0] * t
        by = cy + tangent[1] * t

        s = bolt_size / 2
        bolt_corners = []
        for dt, dn in [(-s, 0), (0, -s), (s, 0), (0, s)]:
            px = bx + tangent[0] * dt + normal[0] * dn
            py = by + tangent[1] * dt + normal[1] * dn
            bolt_corners.append((px, py))

        for j in range(4):
            x1, y1 = bolt_corners[j]
            x2, y2 = bolt_corners[(j + 1) % 4]
            clip_and_draw_line_inside(drawing, x1, y1, x2, y2,
                                      pipe_poly, occlusion_poly, sw)


def draw_decoration_inspection_window(drawing, cx, cy, tangent, normal, hw,
                                      pipe_poly, occlusion_poly, sw, rng,
                                      extra_scale=1.0):
    """Small rounded rectangle inset from the pipe wall."""
    scale = _deco_scale(hw, extra_scale)
    w = 10 * scale
    h = 5 * scale
    cr = 1.5 * scale

    # Shift inward from wall so window is inside the pipe
    inset = h / 2 + 1 * scale
    wcx = cx - normal[0] * inset
    wcy = cy - normal[1] * inset

    hw2 = w / 2 - cr
    hh2 = h / 2 - cr

    # 4 straight edges (shortened by corner radius)
    edges = [
        (-hw2, h / 2, hw2, h / 2),
        (hw2, -h / 2, -hw2, -h / 2),
        (-w / 2, hh2, -w / 2, -hh2),
        (w / 2, -hh2, w / 2, hh2),
    ]

    for lt1, ln1, lt2, ln2 in edges:
        x1 = wcx + tangent[0] * lt1 + normal[0] * ln1
        y1 = wcy + tangent[1] * lt1 + normal[1] * ln1
        x2 = wcx + tangent[0] * lt2 + normal[0] * ln2
        y2 = wcy + tangent[1] * lt2 + normal[1] * ln2
        clip_and_draw_line_inside(drawing, x1, y1, x2, y2,
                                  pipe_poly, occlusion_poly, sw)

    # 4 quarter-circle corners
    tangent_angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    corner_centers = [
        (hw2, hh2), (-hw2, hh2), (-hw2, -hh2), (hw2, -hh2),
    ]

    for i, (lt, ln) in enumerate(corner_centers):
        acx = wcx + tangent[0] * lt + normal[0] * ln
        acy = wcy + tangent[1] * lt + normal[1] * ln
        start = tangent_angle + i * 90
        end = start + 90
        clip_and_draw_arc_inside(drawing, acx, acy, cr, start, end,
                                 pipe_poly, occlusion_poly, sw, num_segments=4)


def draw_decoration_flow_arrow(drawing, cx, cy, tangent, normal, hw,
                               pipe_poly, occlusion_poly, sw, rng,
                               extra_scale=1.0):
    """Chevron arrow pointing along pipe flow direction."""
    scale = _deco_scale(hw, extra_scale)
    arrow_len = 6 * scale
    arrow_width = 4 * scale

    # Randomly pick forward or backward along tangent
    direction = rng.choice([-1, 1])
    tip_x = cx + tangent[0] * arrow_len * direction
    tip_y = cy + tangent[1] * arrow_len * direction
    # Two base points spread perpendicular to tangent
    base_x = cx - tangent[0] * arrow_len * direction
    base_y = cy - tangent[1] * arrow_len * direction
    left_x = base_x + normal[0] * arrow_width
    left_y = base_y + normal[1] * arrow_width
    right_x = base_x - normal[0] * arrow_width
    right_y = base_y - normal[1] * arrow_width

    # Draw as two lines from base corners to tip (open chevron)
    clip_and_draw_line_inside(drawing, left_x, left_y, tip_x, tip_y,
                              pipe_poly, occlusion_poly, sw)
    clip_and_draw_line_inside(drawing, right_x, right_y, tip_x, tip_y,
                              pipe_poly, occlusion_poly, sw)


DECORATION_DRAW_FUNCTIONS = {
    'coupling_box': draw_decoration_coupling_box,
    'round_dial': draw_decoration_round_dial,
    'valve_handle': draw_decoration_valve_handle,
    'bolt_heads': draw_decoration_bolt_heads,
    'inspection_window': draw_decoration_inspection_window,
    'flow_arrow': draw_decoration_flow_arrow,
}


def draw_decoration(drawing, ch, xloc, yloc, deco_type, light_dir,
                    pipe_poly, occlusion_poly, sw, rng, decoration_scale=1.0):
    """Dispatch decoration drawing for a single tile."""
    shape = TILE_SHAPE.get(ch)

    if shape == 'straight':
        cx, cy, tangent, normal, hw = get_straight_placement(
            ch, xloc, yloc, light_dir, rng)
    elif shape == 'reducer':
        cx, cy, tangent, normal, hw = get_reducer_placement(
            ch, xloc, yloc, light_dir, rng)
    else:
        return

    # Check minimum size for this decoration type
    if hw < DECORATION_MIN_HW.get(deco_type, 5):
        deco_type = 'coupling_box' if hw >= 5 else None
        if deco_type is None:
            return

    draw_fn = DECORATION_DRAW_FUNCTIONS[deco_type]
    draw_fn(drawing, cx, cy, tangent, normal, hw,
            pipe_poly, occlusion_poly, sw, rng,
            extra_scale=decoration_scale)


def _render_layer_to_group(
    grid, outlines_target, shading_target, decorations_target,
    poly_cache, extra_occlusion_poly,
    stroke_width, shading_style, shading_stroke_width,
    light_angle_deg=225, shading_params=None,
    decorations_enabled=False, decoration_density=0.10,
    decoration_stroke_width=None, decoration_scale=1.0,
    progress_callback=None, progress_offset=0, progress_total=None,
):
    """Render a single grid layer's outlines, shading, and decorations.

    All draw functions append to the provided target objects (Drawing or Group).
    extra_occlusion_poly is merged into each tile's occlusion for cross-layer clipping.
    """
    width = len(grid)
    height = len(grid[0]) if width > 0 else 0
    pad = max(stroke_width, shading_stroke_width) * 0.5 + 0.1

    total_tiles = progress_total if progress_total is not None else width * height
    tile_count = 0
    for x in range(width):
        for y in range(height):
            ch = grid[x][y]
            if ch == VOID_CHAR:
                tile_count += 1
                if progress_callback:
                    progress_callback(progress_offset + tile_count, total_tiles)
                continue
            xloc = (x - (width - 1) / 2.0) * 100
            yloc = (y - (height - 1) / 2.0) * 100

            # Build occlusion polygon using cached polygons
            occlusion_poly = build_occlusion_polygon_cached(
                poly_cache, x, y, pad=pad
            )

            # Merge cross-layer occlusion
            if extra_occlusion_poly is not None:
                if occlusion_poly is not None:
                    occlusion_poly = occlusion_poly.union(extra_occlusion_poly)
                else:
                    occlusion_poly = extra_occlusion_poly

            # Draw pipe outlines with clipping
            if ch in ('|', '-', 'i', '=', '!', '.'):
                draw_tube_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('r', '7', 'j', 'L'):
                draw_corner_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('nr', 'n7', 'nj', 'nL'):
                draw_sized_corner_outline(outlines_target, ch, xloc, yloc, 12, occlusion_poly, stroke_width)
            elif ch in ('tr', 't7', 'tj', 'tL'):
                draw_sized_corner_outline(outlines_target, ch, xloc, yloc, 5, occlusion_poly, stroke_width)
            elif ch in _DODGE_PARAMS:
                draw_dodge_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _SBEND_PARAMS:
                draw_sbend_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _SEGMENTED_PARAMS:
                draw_segmented_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _LONG_RADIUS_PARAMS:
                draw_long_radius_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _CHAMFER_PARAMS:
                draw_chamfer_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _TEARDROP_PARAMS:
                draw_teardrop_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _DIAG_PARAMS:
                draw_diagonal_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _ADAPTER_PARAMS:
                draw_adapter_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _DIAG_CORNER_PARAMS:
                draw_diagonal_corner_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _DIAG_ENDCAP_PARAMS:
                draw_diagonal_endcap_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _ENDCAP_PARAMS:
                draw_endcap_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _VANISHING_PARAMS:
                draw_vanishing_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _MIXED_CORNER_PARAMS:
                draw_mixed_corner_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('Rv', 'RV', 'Tv', 'TV', 'Rh', 'RH', 'Th', 'TH'):
                draw_reducer_outline(outlines_target, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('X', 'nX', 'tX'):
                hw = 30 if ch == 'X' else (12 if ch == 'nX' else 5)
                draw_cross_outline(outlines_target, xloc, yloc, occlusion_poly, stroke_width, half_width=hw)
            elif ch in ('T', 'B', 'E', 'W', 'nT', 'nB', 'nE', 'nW', 'tT', 'tB', 'tE', 'tW'):
                if ch in ('T', 'B', 'E', 'W'):
                    hw = 30
                    base_ch = ch
                elif ch.startswith('n'):
                    hw = 12
                    base_ch = ch[1:]
                else:
                    hw = 5
                    base_ch = ch[1:]
                draw_tee_outline(outlines_target, base_ch, xloc, yloc, occlusion_poly, stroke_width, half_width=hw)
            elif ch in CROSSOVER_TUBES:
                # Crossover: draw with depth ordering (vertical on top of horizontal)
                v_ch, h_ch = CROSSOVER_TUBES[ch]
                v_poly = get_tube_polygon(v_ch, xloc, yloc)
                if v_poly:
                    v_poly = v_poly.buffer(pad, join_style=2)

                # Draw horizontal first (underneath)
                h_occlusion = occlusion_poly
                if h_occlusion is not None and v_poly is not None:
                    h_occlusion = h_occlusion.union(v_poly)
                elif v_poly is not None:
                    h_occlusion = v_poly
                draw_tube_outline(outlines_target, h_ch, xloc, yloc, h_occlusion, stroke_width)

                # Draw vertical on top (only clipped against other pipes, not horizontal)
                draw_tube_outline(outlines_target, v_ch, xloc, yloc, occlusion_poly, stroke_width)

            elif ch in DIAG_CROSSOVER_TUBES:
                # Diagonal crossover: diagonal on top, cardinal tube underneath
                tube_ch, diag_ch = DIAG_CROSSOVER_TUBES[ch]
                diag_poly = get_diagonal_polygon(diag_ch, xloc, yloc)
                if diag_poly:
                    diag_poly = diag_poly.buffer(pad, join_style=2)

                # Draw tube underneath — clipped by diagonal + neighbors
                tube_occlusion = occlusion_poly
                if tube_occlusion is not None and diag_poly is not None:
                    tube_occlusion = tube_occlusion.union(diag_poly)
                elif diag_poly is not None:
                    tube_occlusion = diag_poly
                draw_tube_outline(outlines_target, tube_ch, xloc, yloc, tube_occlusion, stroke_width)

                # Draw diagonal on top — only clipped by neighbors
                draw_diagonal_outline(outlines_target, diag_ch, xloc, yloc, occlusion_poly, stroke_width)

            # Draw shading with clipping
            if shading_style == 'directional-hatch':
                params = shading_params if shading_params else DEFAULT_SHADING_PARAMS
                light_dir = _normalize((math.cos(math.radians(light_angle_deg)),
                                        math.sin(math.radians(light_angle_deg))))
                pipe_poly = get_pipe_polygon(ch, xloc, yloc)
                if pipe_poly:
                    pipe_poly = pipe_poly.buffer(0)  # Clean geometry

                if ch in ('|', '-', 'i', '=', '!', '.'):
                    draw_tube_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                  pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('r', '7', 'j', 'L'):
                    draw_corner_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                    pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('nr', 'n7', 'nj', 'nL'):
                    draw_sized_corner_directional_shading(shading_target, ch, xloc, yloc, 12, light_dir, shading_stroke_width, params,
                                                          pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('tr', 't7', 'tj', 'tL'):
                    draw_sized_corner_directional_shading(shading_target, ch, xloc, yloc, 5, light_dir, shading_stroke_width, params,
                                                          pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _DODGE_PARAMS:
                    draw_dodge_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                    pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _SBEND_PARAMS:
                    draw_sbend_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                    pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _SEGMENTED_PARAMS:
                    draw_segmented_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                        pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _LONG_RADIUS_PARAMS:
                    draw_long_radius_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                          pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _CHAMFER_PARAMS:
                    draw_chamfer_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                      pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _TEARDROP_PARAMS:
                    draw_teardrop_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                       pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _DIAG_PARAMS:
                    draw_diagonal_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                      pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _ADAPTER_PARAMS:
                    draw_adapter_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                     pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _DIAG_CORNER_PARAMS:
                    draw_diagonal_corner_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                              pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _DIAG_ENDCAP_PARAMS:
                    draw_diagonal_endcap_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                              pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _ENDCAP_PARAMS:
                    draw_endcap_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                    pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _VANISHING_PARAMS:
                    draw_vanishing_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                        pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _MIXED_CORNER_PARAMS:
                    draw_mixed_corner_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                          pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('Rv', 'RV', 'Tv', 'TV', 'Rh', 'RH', 'Th', 'TH'):
                    draw_reducer_directional_shading(shading_target, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                     pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('X', 'nX', 'tX'):
                    hw = 30 if ch == 'X' else (12 if ch == 'nX' else 5)
                    draw_cross_directional_shading(shading_target, xloc, yloc, light_dir, shading_stroke_width, params,
                                                   pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly,
                                                   half_width=hw)
                elif ch in ('T', 'B', 'E', 'W', 'nT', 'nB', 'nE', 'nW', 'tT', 'tB', 'tE', 'tW'):
                    if ch in ('T', 'B', 'E', 'W'):
                        hw = 30
                        base_ch = ch
                    elif ch.startswith('n'):
                        hw = 12
                        base_ch = ch[1:]
                    else:
                        hw = 5
                        base_ch = ch[1:]
                    draw_tee_directional_shading(shading_target, base_ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                 pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly,
                                                 half_width=hw)
                elif ch in CROSSOVER_TUBES:
                    v_ch, h_ch = CROSSOVER_TUBES[ch]
                    v_poly = get_tube_polygon(v_ch, xloc, yloc)
                    h_poly = get_tube_polygon(h_ch, xloc, yloc)
                    if v_poly:
                        v_poly = v_poly.buffer(0)
                    if h_poly:
                        h_poly = h_poly.buffer(0)

                    h_occlusion = occlusion_poly
                    if h_occlusion is not None and v_poly is not None:
                        v_buffered = v_poly.buffer(pad, join_style=2)
                        h_occlusion = h_occlusion.union(v_buffered)
                    elif v_poly is not None:
                        h_occlusion = v_poly.buffer(pad, join_style=2)

                    draw_tube_directional_shading(shading_target, h_ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                  pipe_polygon=h_poly, occlusion_polygon=h_occlusion)
                    draw_tube_directional_shading(shading_target, v_ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                  pipe_polygon=v_poly, occlusion_polygon=occlusion_poly)

                elif ch in DIAG_CROSSOVER_TUBES:
                    tube_ch, diag_ch = DIAG_CROSSOVER_TUBES[ch]
                    tube_poly = get_tube_polygon(tube_ch, xloc, yloc)
                    diag_poly = get_diagonal_polygon(diag_ch, xloc, yloc)
                    if tube_poly:
                        tube_poly = tube_poly.buffer(0)
                    if diag_poly:
                        diag_poly = diag_poly.buffer(0)

                    # Tube shading: clipped by diagonal (on top) + neighbors
                    tube_occlusion = occlusion_poly
                    if tube_occlusion is not None and diag_poly is not None:
                        diag_buffered = diag_poly.buffer(pad, join_style=2)
                        tube_occlusion = tube_occlusion.union(diag_buffered)
                    elif diag_poly is not None:
                        tube_occlusion = diag_poly.buffer(pad, join_style=2)

                    draw_tube_directional_shading(shading_target, tube_ch, xloc, yloc, light_dir,
                                                   shading_stroke_width, params,
                                                   pipe_polygon=tube_poly,
                                                   occlusion_polygon=tube_occlusion)
                    draw_diagonal_directional_shading(shading_target, diag_ch, xloc, yloc, light_dir,
                                                       shading_stroke_width, params,
                                                       pipe_polygon=diag_poly,
                                                       occlusion_polygon=occlusion_poly)

            elif shading_style in ('accent', 'hatch', 'double-wall'):
                # Draw other shading styles with clipping
                if ch in ('|', '-'):
                    draw_tube_shading_clipped(shading_target, ch, xloc, yloc, shading_style, shading_stroke_width, occlusion_poly)
                elif ch in ('r', '7', 'j', 'L'):
                    draw_corner_shading_clipped(shading_target, ch, xloc, yloc, shading_style, shading_stroke_width, occlusion_poly)
                elif ch in CROSSOVER_TUBES:
                    v_ch, h_ch = CROSSOVER_TUBES[ch]
                    v_poly = get_tube_polygon(v_ch, xloc, yloc)
                    if v_poly:
                        v_poly = v_poly.buffer(pad, join_style=2)

                    h_occlusion = occlusion_poly
                    if h_occlusion is not None and v_poly is not None:
                        h_occlusion = h_occlusion.union(v_poly)
                    elif v_poly is not None:
                        h_occlusion = v_poly

                    draw_tube_shading_clipped(shading_target, h_ch, xloc, yloc, shading_style, shading_stroke_width, h_occlusion)
                    draw_tube_shading_clipped(shading_target, v_ch, xloc, yloc, shading_style, shading_stroke_width, occlusion_poly)

                elif ch in DIAG_CROSSOVER_TUBES:
                    tube_ch, diag_ch = DIAG_CROSSOVER_TUBES[ch]
                    diag_poly = get_diagonal_polygon(diag_ch, xloc, yloc)
                    if diag_poly:
                        diag_poly = diag_poly.buffer(pad, join_style=2)

                    tube_occlusion = occlusion_poly
                    if tube_occlusion is not None and diag_poly is not None:
                        tube_occlusion = tube_occlusion.union(diag_poly)
                    elif diag_poly is not None:
                        tube_occlusion = diag_poly

                    draw_tube_shading_clipped(shading_target, tube_ch, xloc, yloc, shading_style,
                                               shading_stroke_width, tube_occlusion)

            tile_count += 1
            if progress_callback:
                progress_callback(progress_offset + tile_count, total_tiles)

    # === DECORATIONS ===
    if decorations_enabled and decorations_target is not None:
        deco_sw = decoration_stroke_width if decoration_stroke_width is not None else stroke_width
        light_dir = _normalize((math.cos(math.radians(light_angle_deg)),
                                math.sin(math.radians(light_angle_deg))))
        decorated_tiles = select_decorated_tiles(grid, density=decoration_density)
        deco_rng = random.Random(hash(str(grid)))

        for (x, y), deco_type in decorated_tiles.items():
            ch = grid[x][y]
            xloc = (x - (width - 1) / 2.0) * 100
            yloc = (y - (height - 1) / 2.0) * 100
            pipe_poly = poly_cache.get((x, y))
            occlusion_poly = build_occlusion_polygon_cached(
                poly_cache, x, y, pad=pad)
            if extra_occlusion_poly is not None:
                if occlusion_poly is not None:
                    occlusion_poly = occlusion_poly.union(extra_occlusion_poly)
                else:
                    occlusion_poly = extra_occlusion_poly
            draw_decoration(decorations_target, ch, xloc, yloc, deco_type, light_dir,
                            pipe_poly, occlusion_poly, deco_sw, deco_rng,
                            decoration_scale=decoration_scale)


def render_svg(grid, stroke_width, shading_style, shading_stroke_width,
               light_angle_deg=225, shading_params=None, progress_callback=None,
               decorations_enabled=False, decoration_density=0.10,
               decoration_stroke_width=None, decoration_scale=1.0):
    """Render a grid of pipe characters to an SVG string.

    All strokes are properly clipped against other pipes for pen plotting.
    No white fills are used - only strokes.

    Args:
        grid: 2D list of pipe characters
        stroke_width: width of pipe outlines
        shading_style: 'none', 'accent', 'hatch', 'double-wall', or 'directional-hatch'
        shading_stroke_width: width of shading strokes
        light_angle_deg: light direction in degrees (0=right, 90=down, 180=left, 270=up)
        shading_params: dict with keys 'band_width', 'band_offset', 'spacing', 'angle',
                        'jitter_pos', 'jitter_angle'. Uses defaults if None.
        decorations_enabled: if True, add plotter-friendly decorations on eligible tiles
        decoration_density: fraction of eligible tiles to decorate (0.0 to 1.0)
        decoration_stroke_width: stroke width for decorations (None = use stroke_width)
    """
    width = len(grid)
    height = len(grid[0]) if width > 0 else 0

    canvas_w = width * 100
    canvas_h = height * 100
    d = draw.Drawing(canvas_w, canvas_h, origin='center', displayInline=False)

    # Calculate padding for stroke width (prevents visual overlap at boundaries)
    pad = max(stroke_width, shading_stroke_width) * 0.5 + 0.1

    # Precompute all pipe polygons once (avoids redundant construction per tile)
    _poly_cache = {}
    for x in range(width):
        for y in range(height):
            ch = grid[x][y]
            xloc = (x - (width - 1) / 2.0) * 100
            yloc = (y - (height - 1) / 2.0) * 100
            poly = get_pipe_polygon(ch, xloc, yloc)
            if poly is not None:
                poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    _poly_cache[(x, y)] = poly

    # Delegate to shared helper (flat output — Drawing as all targets, no extra occlusion)
    _render_layer_to_group(
        grid, d, d, d, _poly_cache, None,
        stroke_width, shading_style, shading_stroke_width,
        light_angle_deg=light_angle_deg, shading_params=shading_params,
        decorations_enabled=decorations_enabled,
        decoration_density=decoration_density,
        decoration_stroke_width=decoration_stroke_width,
        decoration_scale=decoration_scale,
        progress_callback=progress_callback,
        progress_offset=0,
        progress_total=width * height,
    )

    return d.as_svg()


def render_single_tile(ch, stroke_width, shading_style, shading_stroke_width,
                       light_angle_deg=225, shading_params=None):
    """Render a single tile character as an isolated 1x1 SVG for catalog/debug display."""
    grid = [[ch]]
    return render_svg(grid, stroke_width, shading_style, shading_stroke_width,
                      light_angle_deg=light_angle_deg, shading_params=shading_params,
                      decorations_enabled=False)


def render_multilayer_svg(
    grids, stroke_width, shading_style, shading_stroke_width,
    light_angle_deg=225, shading_params=None, progress_callback=None,
    decorations_enabled=False, decoration_density=0.10,
    decoration_stroke_width=None, decoration_scale=1.0,
    layer_colors=None,
):
    """Render multiple overlapping pipe layers to a single SVG string.

    Layers are rendered bottom-to-top. Each layer gets its own SVG group
    hierarchy. Bottom layers are clipped against all layers above them.

    Args:
        grids: list of 2D pipe character grids. grids[0] = bottom layer,
               grids[-1] = top layer. All grids must have same dimensions.
        layer_colors: list of color names for data-color attribute on layer groups.
                      Defaults to ['red', 'black'] for 2 layers.
        (other args same as render_svg)

    Returns:
        SVG string with grouped layer structure:
        <g id="layer-0" data-color="...">
            <g id="layer-0-outlines">...</g>
            <g id="layer-0-shading">...</g>
            <g id="layer-0-decorations">...</g>
        </g>
        <g id="layer-1" data-color="...">...</g>
    """
    num_layers = len(grids)
    if layer_colors is None:
        layer_colors = ['red', 'black'][:num_layers]

    # All grids must be same dimensions
    width = len(grids[0])
    height = len(grids[0][0]) if width > 0 else 0

    canvas_w = width * 100
    canvas_h = height * 100
    d = draw.Drawing(canvas_w, canvas_h, origin='center', displayInline=False)
    pad = max(stroke_width, shading_stroke_width) * 0.5 + 0.1

    # Phase 1: Precompute polygon caches for all layers
    poly_caches = []
    for layer_idx in range(num_layers):
        grid = grids[layer_idx]
        cache = {}
        for x in range(width):
            for y in range(height):
                ch = grid[x][y]
                xloc = (x - (width - 1) / 2.0) * 100
                yloc = (y - (height - 1) / 2.0) * 100
                poly = get_pipe_polygon(ch, xloc, yloc)
                if poly is not None:
                    poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        cache[(x, y)] = poly
        poly_caches.append(cache)

    # Phase 2: Build full-layer union polygons for cross-layer occlusion
    layer_unions = []
    for layer_idx in range(num_layers):
        layer_unions.append(_build_full_layer_union(poly_caches[layer_idx], pad=pad))

    # For layer i, extra occlusion = union of all layers ABOVE it (j > i)
    extra_occlusions = [None] * num_layers
    for layer_idx in range(num_layers):
        above_polys = []
        for j in range(layer_idx + 1, num_layers):
            if layer_unions[j] is not None:
                above_polys.append(layer_unions[j])
        if len(above_polys) == 1:
            extra_occlusions[layer_idx] = above_polys[0]
        elif len(above_polys) > 1:
            extra_occlusions[layer_idx] = unary_union(above_polys).buffer(0)

    # Phase 3: Render each layer into groups
    total_tiles = width * height * num_layers
    tiles_done = 0

    for layer_idx in range(num_layers):
        grid = grids[layer_idx]
        color = layer_colors[layer_idx] if layer_idx < len(layer_colors) else 'black'

        # Create group hierarchy
        layer_group = draw.Group(id='layer-{}'.format(layer_idx),
                                 **{'data-color': color})
        outlines_group = draw.Group(id='layer-{}-outlines'.format(layer_idx))
        shading_group = draw.Group(id='layer-{}-shading'.format(layer_idx))
        decorations_group = draw.Group(id='layer-{}-decorations'.format(layer_idx))

        _render_layer_to_group(
            grid,
            outlines_group, shading_group,
            decorations_group if decorations_enabled else None,
            poly_caches[layer_idx],
            extra_occlusions[layer_idx],
            stroke_width, shading_style, shading_stroke_width,
            light_angle_deg=light_angle_deg, shading_params=shading_params,
            decorations_enabled=decorations_enabled,
            decoration_density=decoration_density,
            decoration_stroke_width=decoration_stroke_width,
            decoration_scale=decoration_scale,
            progress_callback=progress_callback,
            progress_offset=tiles_done,
            progress_total=total_tiles,
        )
        tiles_done += width * height

        # Assemble group hierarchy
        layer_group.append(outlines_group)
        layer_group.append(shading_group)
        if decorations_enabled:
            layer_group.append(decorations_group)
        d.append(layer_group)

    return d.as_svg()


def render_scaled_multilayer_svg(layer_specs, light_angle_deg=225,
                                  progress_callback=None):
    """Render multiple layers with independent scaling to a single SVG.

    Each layer_spec is a dict from make_layer_spec() with grid, scale,
    offset, and per-layer rendering params.

    Layers are ordered bottom (index 0) to top (index -1).
    Upper layers occlude lower layers. Scaling is applied via SVG group
    transforms; stroke widths are counter-scaled to remain constant.

    Args:
        layer_specs: list of LayerSpec dicts
        light_angle_deg: global light direction
        progress_callback: fn(current, total) for progress

    Returns:
        SVG string
    """
    num_layers = len(layer_specs)

    # --- Phase 1: Build local polygon caches ---
    poly_caches_local = []
    for spec in layer_specs:
        grid = spec['grid']
        width = len(grid)
        height = len(grid[0]) if width > 0 else 0
        cache = {}
        for x in range(width):
            for y in range(height):
                ch = grid[x][y]
                xloc = (x - (width - 1) / 2.0) * 100
                yloc = (y - (height - 1) / 2.0) * 100
                poly = get_pipe_polygon(ch, xloc, yloc)
                if poly is not None:
                    poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        cache[(x, y)] = poly
        poly_caches_local.append(cache)

    # --- Phase 2: Build world-space layer unions for cross-layer occlusion ---
    layer_unions_world = []
    for i, spec in enumerate(layer_specs):
        pad = max(spec['stroke_width'], spec['shading_stroke_width']) * 0.5 + 0.1
        local_union = _build_full_layer_union(poly_caches_local[i], pad=pad)
        world_union = _poly_local_to_world(local_union, spec['scale'], spec['offset'])
        layer_unions_world.append(world_union)

    # --- Phase 3: Build cross-layer occlusion in each layer's local space ---
    extra_occlusions_local = [None] * num_layers
    for i in range(num_layers):
        above_polys = []
        for j in range(i + 1, num_layers):
            if layer_unions_world[j] is not None:
                above_polys.append(layer_unions_world[j])

        if not above_polys:
            continue

        if len(above_polys) == 1:
            world_occlusion = above_polys[0]
        else:
            world_occlusion = unary_union(above_polys).buffer(0)

        # Transform world occlusion into layer i's local coordinate space
        extra_occlusions_local[i] = _poly_world_to_local(
            world_occlusion, layer_specs[i]['scale'], layer_specs[i]['offset']
        )

    # --- Phase 4: Compute canvas bounds (symmetric about origin) ---
    x_extents = []
    y_extents = []
    for spec in layer_specs:
        grid = spec['grid']
        width = len(grid)
        height = len(grid[0]) if width > 0 else 0
        s = spec['scale']
        ox, oy = spec['offset']
        half_w = width * 50 * s
        half_h = height * 50 * s
        x_extents.extend([abs(-half_w + ox), abs(half_w + ox)])
        y_extents.extend([abs(-half_h + oy), abs(half_h + oy)])

    canvas_w = 2 * max(x_extents) if x_extents else 800
    canvas_h = 2 * max(y_extents) if y_extents else 800

    d = draw.Drawing(canvas_w, canvas_h, origin='center', displayInline=False)

    # --- Phase 5: Render each layer (bottom to top) ---
    total_tiles = sum(
        len(spec['grid']) * (len(spec['grid'][0]) if spec['grid'] else 0)
        for spec in layer_specs
    )
    tiles_done = 0

    for i, spec in enumerate(layer_specs):
        grid = spec['grid']
        width = len(grid)
        height = len(grid[0]) if width > 0 else 0
        s = spec['scale']
        ox, oy = spec['offset']
        color = spec.get('color', 'black')
        name = spec.get('name') or 'layer-{}'.format(i)

        # SVG transform: translate then scale (rightmost applied first)
        transform_str = 'translate({},{}) scale({})'.format(ox, oy, s)

        layer_group = draw.Group(
            id=name, transform=transform_str,
            **{'data-color': color}
        )
        outlines_group = draw.Group(id='{}-outlines'.format(name))
        shading_group = draw.Group(id='{}-shading'.format(name))
        decorations_group = draw.Group(id='{}-decorations'.format(name))

        # Counter-scale stroke widths so they appear constant after SVG scaling
        effective_stroke = spec['stroke_width'] / s
        effective_shading_stroke = spec['shading_stroke_width'] / s
        effective_deco_stroke = (
            spec['decoration_stroke_width'] / s
            if spec['decoration_stroke_width'] is not None
            else None
        )

        _render_layer_to_group(
            grid,
            outlines_group, shading_group,
            decorations_group if spec['decorations_enabled'] else None,
            poly_caches_local[i],
            extra_occlusions_local[i],
            effective_stroke, spec['shading_style'], effective_shading_stroke,
            light_angle_deg=light_angle_deg,
            shading_params=spec['shading_params'],
            decorations_enabled=spec['decorations_enabled'],
            decoration_density=spec['decoration_density'],
            decoration_stroke_width=effective_deco_stroke,
            decoration_scale=spec['decoration_scale'],
            progress_callback=progress_callback,
            progress_offset=tiles_done,
            progress_total=total_tiles,
        )
        tiles_done += width * height

        # Assemble group hierarchy
        layer_group.append(outlines_group)
        layer_group.append(shading_group)
        if spec['decorations_enabled']:
            layer_group.append(decorations_group)
        d.append(layer_group)

    return d.as_svg()
