import drawsvg as draw
import random
import math
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

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

    # Base r orientation (S+E ports), 8 vertices
    points = [
        tp(-hw, 50),          # south edge, inner wall
        tp(hw, 50),           # south edge, outer wall
        tp(hw, s2 - hw),      # outer: vert->diag transition
        tp(s2 - hw, hw),      # outer: diag->horiz transition
        tp(50, hw),           # east edge, outer wall
        tp(50, -hw),          # east edge, inner wall
        tp(-s2 + hw, -hw),    # inner: horiz->diag transition
        tp(-hw, -s2 + hw),    # inner: diag->vert transition
    ]

    return Polygon(points)


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

    # Chamfered corners
    elif ch in _CHAMFER_PARAMS:
        return get_chamfer_polygon(ch, xloc, yloc)

    # Teardrop elbows
    elif ch in _TEARDROP_PARAMS:
        return get_teardrop_polygon(ch, xloc, yloc)

    # Crossovers (all sizes)
    elif ch in CROSSOVER_TUBES:
        v_ch, h_ch = CROSSOVER_TUBES[ch]
        v = get_tube_polygon(v_ch, xloc, yloc)
        h = get_tube_polygon(h_ch, xloc, yloc)
        if v and h:
            return unary_union([v, h])
        return v or h

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

    # Inner wall: horizontal section -> diagonal -> vertical section
    x1, y1 = tp(50, -hw)
    x2, y2 = tp(-s2 + hw, -hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(-s2 + hw, -hw)
    x2, y2 = tp(-hw, -s2 + hw)
    clip_and_draw_line(drawing, x1, y1, x2, y2, occlusion_poly, sw)

    x1, y1 = tp(-hw, -s2 + hw)
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

PORTS = {
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
}

ALL_CHARS = set(PORTS.keys())

# Backward compatibility: OPENINGS as set of directions (for code that only needs presence)
OPENINGS = {ch: set(ports.keys()) for ch, ports in PORTS.items()}

# Tile category mappings for weight system
TILE_SIZE = {}
for _ch in ['|', '-', 'r', '7', 'j', 'L', '+', 'X', 'T', 'B', 'E', 'W',
            'z>', 'z<', 'z^', 'zv', 'cr', 'c7', 'cj', 'cL', 'dr', 'd7', 'dj', 'dL']:
    TILE_SIZE[_ch] = 'medium'
for _ch in ['i', '=', 'nr', 'n7', 'nj', 'nL', 'nX', 'nT', 'nB', 'nE', 'nW',
            'nz>', 'nz<', 'nz^', 'nzv', 'ncr', 'nc7', 'ncj', 'ncL', 'ndr', 'nd7', 'ndj', 'ndL']:
    TILE_SIZE[_ch] = 'narrow'
for _ch in ['!', '.', 'tr', 't7', 'tj', 'tL', 'tX', 'tT', 'tB', 'tE', 'tW',
            'tz>', 'tz<', 'tz^', 'tzv', 'tcr', 'tc7', 'tcj', 'tcL', 'tdr', 'td7', 'tdj', 'tdL']:
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
for _ch in ['cr', 'c7', 'cj', 'cL', 'ncr', 'nc7', 'ncj', 'ncL', 'tcr', 'tc7', 'tcj', 'tcL']:
    TILE_SHAPE[_ch] = 'chamfer'
for _ch in ['dr', 'd7', 'dj', 'dL', 'ndr', 'nd7', 'ndj', 'ndL', 'tdr', 'td7', 'tdj', 'tdL']:
    TILE_SHAPE[_ch] = 'teardrop'

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


def get_tile_weight(ch, tile_weights):
    """Calculate tile weight from size and shape multipliers."""
    if tile_weights is None:
        if ch in CROSSOVER_TUBES:
            return 2
        shape = TILE_SHAPE.get(ch)
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


def get_port_size(ch, direction):
    """Get the port size for a tile in a given direction. Returns None if no port."""
    return PORTS.get(ch, {}).get(direction)


def has_opening(ch, direction):
    """Check if tile has an opening in given direction (any size)."""
    return get_port_size(ch, direction) is not None


def get_compatible_neighbors(ch, direction):
    """Get all tiles that can connect to ch in the given direction.

    Tiles connect if:
    - Both have openings in their respective directions (ch.direction, neighbor.opposite)
    - The port sizes match
    OR
    - Neither has an opening (both closed)
    """
    opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    opp_dir = opposite[direction]
    ch_port_size = get_port_size(ch, direction)
    compatible = set()
    for candidate in ALL_CHARS:
        cand_port_size = get_port_size(candidate, opp_dir)
        # Both closed (None == None) or both open with same size
        if ch_port_size == cand_port_size:
            compatible.add(candidate)
    return compatible


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


def create_possibility_grid(width, height):
    return [[ALL_CHARS.copy() for _ in range(height)] for _ in range(width)]


def get_constrained_possibilities(possibilities, x, y, width, height):
    valid = possibilities.copy()
    if x == 0:
        valid = {ch for ch in valid if not has_opening(ch, 'W')}
    if x == width - 1:
        valid = {ch for ch in valid if not has_opening(ch, 'E')}
    if y == 0:
        valid = {ch for ch in valid if not has_opening(ch, 'N')}
    if y == height - 1:
        valid = {ch for ch in valid if not has_opening(ch, 'S')}
    return valid


def propagate_constraints(poss_grid, width, height):
    changed = True
    while changed:
        changed = False
        for x in range(width):
            for y in range(height):
                if len(poss_grid[x][y]) <= 1:
                    continue
                current = poss_grid[x][y].copy()
                current = get_constrained_possibilities(current, x, y, width, height)
                if y > 0:
                    valid_from_north = set()
                    for n_ch in poss_grid[x][y-1]:
                        valid_from_north |= find_s(n_ch)
                    current &= valid_from_north
                if y < height - 1:
                    valid_from_south = set()
                    for s_ch in poss_grid[x][y+1]:
                        valid_from_south |= find_n(s_ch)
                    current &= valid_from_south
                if x > 0:
                    valid_from_west = set()
                    for w_ch in poss_grid[x-1][y]:
                        valid_from_west |= find_e(w_ch)
                    current &= valid_from_west
                if x < width - 1:
                    valid_from_east = set()
                    for e_ch in poss_grid[x+1][y]:
                        valid_from_east |= find_w(e_ch)
                    current &= valid_from_east
                if current != poss_grid[x][y]:
                    poss_grid[x][y] = current
                    changed = True
        if remove_tight_circle_possibilities(poss_grid, width, height):
            changed = True


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


def collapse_cell(poss_grid, x, y, tile_weights=None):
    possibilities = list(poss_grid[x][y])
    if possibilities:
        weights = [get_tile_weight(ch, tile_weights) for ch in possibilities]
        chosen = random.choices(possibilities, weights=weights, k=1)[0]
        poss_grid[x][y] = {chosen}
        return True
    return False


def wave_function_collapse(width, height, tile_weights=None,
                           max_iterations=10000, max_attempts=20):
    """Run WFC and return a 2D grid of pipe characters, or None on failure."""
    for attempt in range(1, max_attempts + 1):
        poss_grid = create_possibility_grid(width, height)

        for x in range(width):
            for y in range(height):
                poss_grid[x][y] = get_constrained_possibilities(
                    poss_grid[x][y], x, y, width, height
                )

        propagate_constraints(poss_grid, width, height)

        contradiction = False
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            cell = find_min_entropy_cell(poss_grid, width, height)
            if cell is None:
                break

            x, y = cell
            if not collapse_cell(poss_grid, x, y, tile_weights):
                contradiction = True
                break

            propagate_constraints(poss_grid, width, height)

            # Check for contradictions
            for cx in range(width):
                for cy in range(height):
                    if len(poss_grid[cx][cy]) == 0:
                        contradiction = True
                        break
                if contradiction:
                    break

            if contradiction:
                break

        if contradiction:
            continue

        # Convert to final grid
        final_grid = [['' for _ in range(height)] for _ in range(width)]
        for x in range(width):
            for y in range(height):
                if len(poss_grid[x][y]) == 1:
                    final_grid[x][y] = list(poss_grid[x][y])[0]
                else:
                    final_grid[x][y] = random.choice(list(poss_grid[x][y])) if poss_grid[x][y] else '+'
        return final_grid

    return None


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


def render_svg(grid, stroke_width, shading_style, shading_stroke_width,
               light_angle_deg=225, shading_params=None, progress_callback=None):
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
    """
    width = len(grid)
    height = len(grid[0]) if width > 0 else 0

    canvas_w = width * 100
    canvas_h = height * 100
    d = draw.Drawing(canvas_w, canvas_h, origin='center', displayInline=False)

    # Calculate padding for stroke width (prevents visual overlap at boundaries)
    pad = max(stroke_width, shading_stroke_width) * 0.5 + 0.1

    # Draw each pipe with proper clipping against all other pipes
    total_tiles = width * height
    tile_count = 0
    for x in range(width):
        for y in range(height):
            ch = grid[x][y]
            # Center the grid properly (works for both odd and even dimensions)
            xloc = (x - (width - 1) / 2.0) * 100
            yloc = (y - (height - 1) / 2.0) * 100

            # Build occlusion polygon (all other pipes, buffered for stroke width)
            occlusion_poly = build_occlusion_polygon(grid, x, y, pad=pad)

            # Draw pipe outlines with clipping
            if ch in ('|', '-', 'i', '=', '!', '.'):
                draw_tube_outline(d, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('r', '7', 'j', 'L'):
                draw_corner_outline(d, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('nr', 'n7', 'nj', 'nL'):
                draw_sized_corner_outline(d, ch, xloc, yloc, 12, occlusion_poly, stroke_width)
            elif ch in ('tr', 't7', 'tj', 'tL'):
                draw_sized_corner_outline(d, ch, xloc, yloc, 5, occlusion_poly, stroke_width)
            elif ch in _DODGE_PARAMS:
                draw_dodge_outline(d, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _CHAMFER_PARAMS:
                draw_chamfer_outline(d, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in _TEARDROP_PARAMS:
                draw_teardrop_outline(d, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('Rv', 'RV', 'Tv', 'TV', 'Rh', 'RH', 'Th', 'TH'):
                draw_reducer_outline(d, ch, xloc, yloc, occlusion_poly, stroke_width)
            elif ch in ('X', 'nX', 'tX'):
                hw = 30 if ch == 'X' else (12 if ch == 'nX' else 5)
                draw_cross_outline(d, xloc, yloc, occlusion_poly, stroke_width, half_width=hw)
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
                draw_tee_outline(d, base_ch, xloc, yloc, occlusion_poly, stroke_width, half_width=hw)
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
                draw_tube_outline(d, h_ch, xloc, yloc, h_occlusion, stroke_width)

                # Draw vertical on top (only clipped against other pipes, not horizontal)
                draw_tube_outline(d, v_ch, xloc, yloc, occlusion_poly, stroke_width)

            # Draw shading with clipping
            if shading_style == 'directional-hatch':
                params = shading_params if shading_params else DEFAULT_SHADING_PARAMS
                light_dir = _normalize((math.cos(math.radians(light_angle_deg)),
                                        math.sin(math.radians(light_angle_deg))))
                pipe_poly = get_pipe_polygon(ch, xloc, yloc)
                if pipe_poly:
                    pipe_poly = pipe_poly.buffer(0)  # Clean geometry

                if ch in ('|', '-', 'i', '=', '!', '.'):
                    # All tube types use same shading logic (with different polygons)
                    draw_tube_directional_shading(d, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                  pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('r', '7', 'j', 'L'):
                    draw_corner_directional_shading(d, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                    pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('nr', 'n7', 'nj', 'nL'):
                    draw_sized_corner_directional_shading(d, ch, xloc, yloc, 12, light_dir, shading_stroke_width, params,
                                                          pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('tr', 't7', 'tj', 'tL'):
                    draw_sized_corner_directional_shading(d, ch, xloc, yloc, 5, light_dir, shading_stroke_width, params,
                                                          pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _DODGE_PARAMS:
                    draw_dodge_directional_shading(d, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                    pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _CHAMFER_PARAMS:
                    draw_chamfer_directional_shading(d, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                      pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in _TEARDROP_PARAMS:
                    draw_teardrop_directional_shading(d, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                       pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('Rv', 'RV', 'Tv', 'TV', 'Rh', 'RH', 'Th', 'TH'):
                    draw_reducer_directional_shading(d, ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                     pipe_polygon=pipe_poly, occlusion_polygon=occlusion_poly)
                elif ch in ('X', 'nX', 'tX'):
                    hw = 30 if ch == 'X' else (12 if ch == 'nX' else 5)
                    draw_cross_directional_shading(d, xloc, yloc, light_dir, shading_stroke_width, params,
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
                    draw_tee_directional_shading(d, base_ch, xloc, yloc, light_dir, shading_stroke_width, params,
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

                    draw_tube_directional_shading(d, h_ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                  pipe_polygon=h_poly, occlusion_polygon=h_occlusion)
                    draw_tube_directional_shading(d, v_ch, xloc, yloc, light_dir, shading_stroke_width, params,
                                                  pipe_polygon=v_poly, occlusion_polygon=occlusion_poly)

            elif shading_style in ('accent', 'hatch', 'double-wall'):
                # Draw other shading styles with clipping
                if ch in ('|', '-'):
                    draw_tube_shading_clipped(d, ch, xloc, yloc, shading_style, shading_stroke_width, occlusion_poly)
                elif ch in ('r', '7', 'j', 'L'):
                    draw_corner_shading_clipped(d, ch, xloc, yloc, shading_style, shading_stroke_width, occlusion_poly)
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

                    draw_tube_shading_clipped(d, h_ch, xloc, yloc, shading_style, shading_stroke_width, h_occlusion)
                    draw_tube_shading_clipped(d, v_ch, xloc, yloc, shading_style, shading_stroke_width, occlusion_poly)

            tile_count += 1
            if progress_callback:
                progress_callback(tile_count, total_tiles)

    return d.as_svg()
