#!/usr/bin/env python
# coding: utf-8

# In[1]:


import drawsvg as draw
import random
import math
import sys
from datetime import datetime
stroke_width = 0.5
max_x_boxes=20
max_y_boxes=20
shading_style = 'directional-hatch'  # 'none', 'accent', 'hatch', 'double-wall', 'directional-hatch'
shading_stroke_width = stroke_width * 0.6

# --- Directional shading parameters ---
SHADOW_BAND_WIDTH = 12     # width of hatch band (out of 30 half-width)
SHADOW_BAND_OFFSET = 3     # gap from pipe wall to band edge
HATCH_SPACING = 10         # base spacing between hatch lines
HATCH_ANGLE_DEG = 30       # base angle of hatches relative to pipe tangent
HATCH_JITTER_POS = 1.5     # max position jitter in units
HATCH_JITTER_ANGLE = 3.0   # max angle jitter in degrees

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

# Light direction: points FROM scene TOWARD light source (top-left)
LIGHT_DIR = _normalize((-1, -1))

d = draw.Drawing(1500, 1500, origin='center', displayInline=False)

l_corner = draw.Group(id='l_corner', fill='white')
#l_corner.append(draw.Rectangle(-50,-50,100,100, fill='#eeeeee', stroke='red', stroke_width=stroke_width))
# pipe corner
mask = draw.Mask()
box = draw.Circle(40, 40, 10, fill='green')
mask.append(box)
# large arc
l_corner.append(draw.Arc(40,40,70,180,270,cw=True, stroke='black', stroke_width=stroke_width, fill='white'))
l_corner.append(draw.Lines(40, -30, -30, 40, 30, 40, 40, 30, fill='white', stroke='none', mask=not mask))
# small arc
l_corner.append(draw.Circle(40, 40, 10, fill='white'))  # fill in the space inside the small curve
l_corner.append(draw.Arc(40,40,10,180,270,cw=True, stroke='black', stroke_width=stroke_width, fill='none'))
#l_corner.append(draw.Arc(40,40,15,200,250, cw=True, fill='none', stroke='black', stroke_width=stroke_width))  # accent
# right rectangle
l_corner.append(draw.Rectangle(40,-35,5,70, fill='white', stroke='black', stroke_width=stroke_width))
# bottom rectangle
l_corner.append(draw.Rectangle(-35,40,70,5, fill='white', stroke='black', stroke_width=stroke_width))
# right lines
l_corner.append(draw.Rectangle(46, -29,5,58, fill='white', stroke='none'))
l_corner.append(draw.Line(45,30,50,30, fill='none', stroke='black', stroke_width=stroke_width))
l_corner.append(draw.Line(45,-30,50,-30, fill='none', stroke='black', stroke_width=stroke_width))
# bottom lines
l_corner.append(draw.Rectangle(-29, 46 ,58, 5, fill='white', stroke='none'))
l_corner.append(draw.Line(30,45,30,50, fill='none', stroke='black', stroke_width=stroke_width))
l_corner.append(draw.Line(-30,45,-30,50, fill='none', stroke='black', stroke_width=stroke_width))

# center dot
#l_corner.append(draw.Circle(0,0,3, fill='none', stroke='black', stroke_width=stroke_width)) #temp center spot

l_tube = draw.Group(id='l_tube', fill='none')
l_tube.append(draw.Rectangle(-30,-50,60,100, fill='white', stroke='none'))
l_tube.append(draw.Line(-30,-50,-30,50, fill='none', stroke='black', stroke_width=stroke_width))
l_tube.append(draw.Line(30,-50,30,50, fill='none', stroke='black', stroke_width=stroke_width))

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
        # Right connector hatches
        for y0 in [-15, 0, 15]:
            group.append(draw.Line(41, y0 - 3.5, 48, y0 + 3.5, stroke='black', stroke_width=sw, fill='none'))
        # Bottom connector hatches
        for x0 in [-15, 0, 15]:
            group.append(draw.Line(x0 - 3.5, 41, x0 + 3.5, 48, stroke='black', stroke_width=sw, fill='none'))

effective_group_style = 'none' if shading_style == 'directional-hatch' else shading_style
add_corner_shading(l_corner, effective_group_style, shading_stroke_width)
add_tube_shading(l_tube, effective_group_style, shading_stroke_width)

def draw_tube_directional_shading(drawing, ch, xloc, yloc, light_dir, sw):
    """Draw directional hatch marks on the shadow side of a straight tube."""
    # Determine tangent and wall normals based on tile character
    if ch == '|':
        tangent = (0, 1)
        normal_left = (-1, 0)
        normal_right = (1, 0)
    elif ch == '-':
        tangent = (1, 0)
        normal_left = (0, -1)
        normal_right = (0, 1)
    else:
        return

    # Shadow side: wall whose outward normal has negative dot with light dir
    dot_left = _dot(normal_left, light_dir)
    dot_right = _dot(normal_right, light_dir)

    if dot_left < dot_right:
        shadow_normal = normal_left
    else:
        shadow_normal = normal_right

    # Band center: offset from pipe center toward shadow wall
    band_center_dist = 30 - SHADOW_BAND_OFFSET - SHADOW_BAND_WIDTH / 2

    # Generate hatch lines along tube length
    num_hatches = int(90 / HATCH_SPACING)
    for i in range(num_hatches + 1):
        t = -45 + i * (90.0 / num_hatches)
        t += random.uniform(-HATCH_JITTER_POS, HATCH_JITTER_POS)

        # Base point at band center, position t along tube
        base_x = xloc + tangent[0] * t + shadow_normal[0] * band_center_dist
        base_y = yloc + tangent[1] * t + shadow_normal[1] * band_center_dist

        # Hatch direction: tangent rotated by hatch angle
        angle = HATCH_ANGLE_DEG + random.uniform(-HATCH_JITTER_ANGLE, HATCH_JITTER_ANGLE)
        hatch_dir = _rotate_vec(tangent, angle)
        half_len = SHADOW_BAND_WIDTH * 0.6

        x1 = base_x - hatch_dir[0] * half_len
        y1 = base_y - hatch_dir[1] * half_len
        x2 = base_x + hatch_dir[0] * half_len
        y2 = base_y + hatch_dir[1] * half_len

        drawing.append(draw.Line(x1, y1, x2, y2,
                                 stroke='black', stroke_width=sw, fill='none'))


def draw_corner_directional_shading(drawing, ch, xloc, yloc, light_dir, sw):
    """Draw directional hatch marks on the shadow portion of a corner piece."""
    rotations = {'r': 0, '7': 90, 'j': 180, 'L': 270}
    rot_deg = rotations[ch]

    # Corner arc center is at local (40, 40), rotated into world coords
    local_center = (40, 40)
    world_offset = _rotate_vec(local_center, rot_deg)
    arc_cx = xloc + world_offset[0]
    arc_cy = yloc + world_offset[1]

    r_outer = 70

    # Arc spans 90 degrees: base 180-270 in local, shifted by rotation
    arc_start = 180 + rot_deg
    arc_end = 270 + rot_deg

    # Sample points along the arc for shadow detection
    num_samples = 8
    for i in range(num_samples):
        frac = (i + 0.5) / num_samples
        angle_deg = arc_start + frac * (arc_end - arc_start)
        angle_rad = math.radians(angle_deg)

        # Outward normal at this arc point (points away from center)
        outward_normal = (math.cos(angle_rad), math.sin(angle_rad))

        # Only draw where surface faces away from light (shadow)
        if _dot(outward_normal, light_dir) >= 0:
            continue

        # Band near the outer edge of the arc
        band_outer = r_outer - SHADOW_BAND_OFFSET
        band_inner = band_outer - SHADOW_BAND_WIDTH
        band_mid = (band_inner + band_outer) / 2

        base_x = arc_cx + math.cos(angle_rad) * band_mid
        base_y = arc_cy + math.sin(angle_rad) * band_mid

        # Hatch direction: tangent to the arc with jitter
        tangent = (-math.sin(angle_rad), math.cos(angle_rad))
        jitter_angle = random.uniform(-HATCH_JITTER_ANGLE, HATCH_JITTER_ANGLE)
        hatch_dir = _rotate_vec(tangent, jitter_angle)
        half_len = SHADOW_BAND_WIDTH * 0.6

        # Position jitter along the arc
        pos_jitter = random.uniform(-HATCH_JITTER_POS, HATCH_JITTER_POS)
        base_x += tangent[0] * pos_jitter
        base_y += tangent[1] * pos_jitter

        x1 = base_x - hatch_dir[0] * half_len
        y1 = base_y - hatch_dir[1] * half_len
        x2 = base_x + hatch_dir[0] * half_len
        y2 = base_y + hatch_dir[1] * half_len

        drawing.append(draw.Line(x1, y1, x2, y2,
                                 stroke='black', stroke_width=sw, fill='none'))

    # Shade the straight connector stubs
    # In local coords, corner has two stubs:
    #   Right stub: tangent (0,1), centered around x=47.5, extends y=-30 to y=30
    #   Bottom stub: tangent (1,0), centered around y=47.5, extends x=-30 to x=30
    stubs = [
        ((0, 1), (47.5, 0), 58),   # right stub: tangent, center offset, length
        ((1, 0), (0, 47.5), 58),   # bottom stub
    ]
    for (tang_local, center_local, stub_len) in stubs:
        tangent = _rotate_vec(tang_local, rot_deg)
        center_offset = _rotate_vec(center_local, rot_deg)
        stub_cx = xloc + center_offset[0]
        stub_cy = yloc + center_offset[1]

        # Determine stub normals (perpendicular to tangent)
        normal_a = _rotate_vec(tangent, 90)
        normal_b = _rotate_vec(tangent, -90)

        dot_a = _dot(normal_a, light_dir)
        dot_b = _dot(normal_b, light_dir)

        if dot_a >= 0 and dot_b >= 0:
            continue  # both sides face light

        shadow_normal = normal_a if dot_a < dot_b else normal_b

        # Draw 3 hatch marks along this stub
        for j in range(3):
            t = -stub_len / 4 + j * (stub_len / 4)
            t += random.uniform(-HATCH_JITTER_POS, HATCH_JITTER_POS)

            base_x = stub_cx + tangent[0] * t + shadow_normal[0] * 1.5
            base_y = stub_cy + tangent[1] * t + shadow_normal[1] * 1.5

            angle = HATCH_ANGLE_DEG + random.uniform(-HATCH_JITTER_ANGLE, HATCH_JITTER_ANGLE)
            hatch_dir = _rotate_vec(tangent, angle)
            half_len = 3.0

            x1 = base_x - hatch_dir[0] * half_len
            y1 = base_y - hatch_dir[1] * half_len
            x2 = base_x + hatch_dir[0] * half_len
            y2 = base_y + hatch_dir[1] * half_len

            drawing.append(draw.Line(x1, y1, x2, y2,
                                     stroke='black', stroke_width=sw, fill='none'))


def sdraw_shading(ch, x, y):
    """Draw directional shading for a placed tile (second pass)."""
    global d
    if shading_style != 'directional-hatch':
        return
    xloc = x * 100
    yloc = y * 100
    if ch in ('|', '-'):
        draw_tube_directional_shading(d, ch, xloc, yloc, LIGHT_DIR, shading_stroke_width)
    elif ch in ('r', '7', 'j', 'L'):
        draw_corner_directional_shading(d, ch, xloc, yloc, LIGHT_DIR, shading_stroke_width)
    elif ch == '+':
        draw_tube_directional_shading(d, '|', xloc, yloc, LIGHT_DIR, shading_stroke_width)
        draw_tube_directional_shading(d, '-', xloc, yloc, LIGHT_DIR, shading_stroke_width)


def sdraw(ch,x,y):
    global d
    xloc = x * 100
    yloc = y * 100
    trans_string = "translate("+str(xloc)+","+str(yloc)+")"
    if ch == "r":
        d.append(draw.Use(l_corner, 0,0, transform=trans_string))
    elif ch == "7":
        d.append(draw.Use(l_corner, 0,0, transform=trans_string+" rotate(90)"))
    elif ch == "j":
        d.append(draw.Use(l_corner, 0,0, transform=trans_string+" rotate(180)"))
    elif ch == "L":
        d.append(draw.Use(l_corner, 0,0, transform=trans_string+" rotate(270)"))
    elif ch == "|":
        d.append(draw.Use(l_tube, 0,0, transform=trans_string+" rotate(0)"))
    elif ch == "-":
        d.append(draw.Use(l_tube, 0,0, transform=trans_string+" rotate(90)"))
    elif ch == "+":
        k = random.randint(0, 1)
        if k == 0:
            d.append(draw.Use(l_tube, 0,0, transform=trans_string+" rotate(0)"))
            d.append(draw.Use(l_tube, 0,0, transform=trans_string+" rotate(90)"))
        else:
            d.append(draw.Use(l_tube, 0,0, transform=trans_string+" rotate(90)"))
            d.append(draw.Use(l_tube, 0,0, transform=trans_string+" rotate(0)"))

# ============================================================================
# PIPE CONNECTION LOGIC
# ============================================================================

# Define which directions each pipe piece has openings
# r = down-right corner, 7 = down-left, j = up-left, L = up-right
# | = vertical, - = horizontal, + = crossover (all directions)
OPENINGS = {
    'r': {'S', 'E'},      # down and right
    '7': {'S', 'W'},      # down and left  
    'j': {'N', 'W'},      # up and left
    'L': {'N', 'E'},      # up and right
    '|': {'N', 'S'},      # vertical
    '-': {'E', 'W'},      # horizontal
    '+': {'N', 'S', 'E', 'W'},  # all directions (crossover)
}

ALL_CHARS = set(OPENINGS.keys())

def has_opening(ch, direction):
    """Check if a pipe piece has an opening in the given direction (N/S/E/W)"""
    return direction in OPENINGS.get(ch, set())

def get_compatible_neighbors(ch, direction):
    """Get all pieces that can be placed in the given direction from ch.

    For pipes to connect: if ch has an opening toward neighbor, 
    neighbor must have an opening back toward ch.
    If ch has NO opening toward neighbor, neighbor must have NO opening toward ch.
    """
    opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    opp_dir = opposite[direction]

    ch_has_opening = has_opening(ch, direction)

    compatible = set()
    for candidate in ALL_CHARS:
        candidate_has_opening = has_opening(candidate, opp_dir)
        # Both must have openings facing each other, or neither
        if ch_has_opening == candidate_has_opening:
            compatible.add(candidate)
    return compatible

def find_n(ch):
    """Characters that can be north (above) of ch"""
    return get_compatible_neighbors(ch, 'N')

def find_s(ch):
    """Characters that can be south (below) of ch"""
    return get_compatible_neighbors(ch, 'S')

def find_e(ch):
    """Characters that can be east (right) of ch"""
    return get_compatible_neighbors(ch, 'E')

def find_w(ch):
    """Characters that can be west (left) of ch"""
    return get_compatible_neighbors(ch, 'W')

# ============================================================================
# WAVE FUNCTION COLLAPSE ALGORITHM
# ============================================================================

# Define the tight 2x2 circle pattern to avoid
# Pattern: r 7
#          L j
TIGHT_CIRCLE = {
    (0, 0): 'r',  # top-left
    (1, 0): '7',  # top-right
    (0, 1): 'L',  # bottom-left
    (1, 1): 'j',  # bottom-right
}

def would_complete_tight_circle(poss_grid, x, y, ch, width, height):
    """Check if placing ch at (x,y) would complete a tight 2x2 circle.

    We check all four 2x2 squares that include position (x,y):
    - (x,y) as top-left
    - (x-1,y) as top-right
    - (x,y-1) as bottom-left
    - (x-1,y-1) as bottom-right
    """
    # Check each 2x2 square that (x,y) could be part of
    checks = [
        # (x,y) is top-left of the 2x2
        ((x, y, 'r'), (x+1, y, '7'), (x, y+1, 'L'), (x+1, y+1, 'j')),
        # (x,y) is top-right of the 2x2
        ((x-1, y, 'r'), (x, y, '7'), (x-1, y+1, 'L'), (x, y+1, 'j')),
        # (x,y) is bottom-left of the 2x2
        ((x, y-1, 'r'), (x+1, y-1, '7'), (x, y, 'L'), (x+1, y, 'j')),
        # (x,y) is bottom-right of the 2x2
        ((x-1, y-1, 'r'), (x, y-1, '7'), (x-1, y, 'L'), (x, y, 'j')),
    ]

    for square in checks:
        # Find which position in this square is (x,y)
        my_pos = None
        for (px, py, expected) in square:
            if px == x and py == y:
                my_pos = (px, py, expected)
                break

        if my_pos is None:
            continue

        # Check if ch matches what would be needed for a tight circle
        _, _, expected_ch = my_pos
        if ch != expected_ch:
            continue

        # ch matches - now check if the other 3 positions are already fixed to circle pattern
        all_fixed = True
        for (px, py, expected) in square:
            if px == x and py == y:
                continue  # skip our position
            if px < 0 or px >= width or py < 0 or py >= height:
                all_fixed = False
                break
            cell_poss = poss_grid[px][py]
            # If this cell is collapsed to the expected circle piece, it's a problem
            if len(cell_poss) == 1 and expected in cell_poss:
                continue
            # If this cell could still be something else, no tight circle yet
            all_fixed = False
            break

        if all_fixed:
            return True

    return False

def remove_tight_circle_possibilities(poss_grid, width, height):
    """Remove possibilities that would complete a tight 2x2 circle.
    Returns True if any changes were made."""
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
    """Create a grid where each cell contains all possible pipe pieces"""
    return [[ALL_CHARS.copy() for _ in range(height)] for _ in range(width)]

def get_constrained_possibilities(possibilities, x, y, width, height):
    """Given border constraints, return valid possibilities for a cell"""
    valid = possibilities.copy()

    # Border constraints: edges can't have openings pointing outward
    # (unless you want pipes going off-screen - set allow_edge_openings=True)
    allow_edge_openings = False

    if not allow_edge_openings:
        if x == 0:  # Left edge - no west openings
            valid = {ch for ch in valid if not has_opening(ch, 'W')}
        if x == width - 1:  # Right edge - no east openings
            valid = {ch for ch in valid if not has_opening(ch, 'E')}
        if y == 0:  # Top edge - no north openings
            valid = {ch for ch in valid if not has_opening(ch, 'N')}
        if y == height - 1:  # Bottom edge - no south openings
            valid = {ch for ch in valid if not has_opening(ch, 'S')}

    return valid

def propagate_constraints(poss_grid, width, height):
    """Propagate constraints through the grid until stable"""
    changed = True
    while changed:
        changed = False
        for x in range(width):
            for y in range(height):
                if len(poss_grid[x][y]) <= 1:
                    continue

                current = poss_grid[x][y].copy()

                # Apply border constraints
                current = get_constrained_possibilities(current, x, y, width, height)

                # Constrain based on neighbors
                # North neighbor (y-1)
                if y > 0:
                    north_possibilities = poss_grid[x][y-1]
                    valid_from_north = set()
                    for n_ch in north_possibilities:
                        valid_from_north |= find_s(n_ch)
                    current &= valid_from_north

                # South neighbor (y+1)
                if y < height - 1:
                    south_possibilities = poss_grid[x][y+1]
                    valid_from_south = set()
                    for s_ch in south_possibilities:
                        valid_from_south |= find_n(s_ch)
                    current &= valid_from_south

                # West neighbor (x-1)
                if x > 0:
                    west_possibilities = poss_grid[x-1][y]
                    valid_from_west = set()
                    for w_ch in west_possibilities:
                        valid_from_west |= find_e(w_ch)
                    current &= valid_from_west

                # East neighbor (x+1)
                if x < width - 1:
                    east_possibilities = poss_grid[x+1][y]
                    valid_from_east = set()
                    for e_ch in east_possibilities:
                        valid_from_east |= find_w(e_ch)
                    current &= valid_from_east

                if current != poss_grid[x][y]:
                    poss_grid[x][y] = current
                    changed = True

        # Also check for tight circle patterns
        if remove_tight_circle_possibilities(poss_grid, width, height):
            changed = True

def find_min_entropy_cell(poss_grid, width, height):
    """Find the cell with fewest possibilities > 1 (lowest entropy)"""
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

def collapse_cell(poss_grid, x, y):
    """Collapse a cell to a single random possibility"""
    possibilities = list(poss_grid[x][y])
    if possibilities:
        # Weight towards more interesting pieces (corners and crosses)
        weights = []
        for ch in possibilities:
            if ch == '+':
                weights.append(2)  # Slightly favor crossovers for complexity
            elif ch in 'r7jL':
                weights.append(3)  # Favor corners for interesting paths
            else:
                weights.append(1)

        chosen = random.choices(possibilities, weights=weights, k=1)[0]
        poss_grid[x][y] = {chosen}
        return True
    return False

def wave_function_collapse(width, height, max_iterations=10000, attempt=1):
    """Main WFC algorithm to generate a valid pipe grid"""
    total_cells = width * height
    print(f"Attempt {attempt}: generating {width}x{height} grid ({total_cells} cells)")
    poss_grid = create_possibility_grid(width, height)

    # Apply initial border constraints
    for x in range(width):
        for y in range(height):
            poss_grid[x][y] = get_constrained_possibilities(
                poss_grid[x][y], x, y, width, height
            )

    propagate_constraints(poss_grid, width, height)

    iterations = 0
    while iterations < max_iterations:
        iterations += 1

        # Find cell with minimum entropy
        cell = find_min_entropy_cell(poss_grid, width, height)

        if cell is None:
            # All cells are collapsed or have 0 possibilities
            break

        x, y = cell

        # Collapse this cell
        if not collapse_cell(poss_grid, x, y):
            print(f"\nFailed to collapse cell ({x}, {y}) - contradiction!")
            return None

        # Propagate constraints
        propagate_constraints(poss_grid, width, height)

        # Report progress
        collapsed = sum(1 for cx in range(width) for cy in range(height) if len(poss_grid[cx][cy]) == 1)
        sys.stdout.write(f"\r  Collapsed {collapsed}/{total_cells} cells ({100*collapsed//total_cells}%)")
        sys.stdout.flush()

        # Check for contradictions (cells with 0 possibilities)
        contradiction = False
        for cx in range(width):
            for cy in range(height):
                if len(poss_grid[cx][cy]) == 0:
                    contradiction = True
                    break
            if contradiction:
                break

        if contradiction:
            # Restart with new random seed
            print(f"\n  Contradiction found, restarting...")
            return wave_function_collapse(width, height, max_iterations, attempt + 1)

    print(f"\n  Done! ({iterations} iterations)")

    # Convert possibility grid to final grid
    final_grid = [['' for _ in range(height)] for _ in range(width)]
    for x in range(width):
        for y in range(height):
            if len(poss_grid[x][y]) == 1:
                final_grid[x][y] = list(poss_grid[x][y])[0]
            else:
                print(f"Warning: Cell ({x}, {y}) not fully collapsed: {poss_grid[x][y]}")
                final_grid[x][y] = random.choice(list(poss_grid[x][y])) if poss_grid[x][y] else '+'

    return final_grid

# ============================================================================
# GENERATE AND DRAW
# ============================================================================

# Generate connected pipe network using Wave Function Collapse
grid = wave_function_collapse(max_x_boxes, max_y_boxes)

if grid:
    # First pass: draw tile outlines
    for x in range(max_x_boxes):
        for y in range(max_y_boxes):
            sdraw(grid[x][y], x - (math.floor(max_x_boxes / 2)), y - (math.floor(max_y_boxes / 2)))

    # Second pass: directional shading (drawn on top of outlines)
    for x in range(max_x_boxes):
        for y in range(max_y_boxes):
            sdraw_shading(grid[x][y], x - (math.floor(max_x_boxes / 2)), y - (math.floor(max_y_boxes / 2)))

filename = f"pipes-{datetime.now().strftime('%Y%m%d-%H%M%S')}.svg"
d.save_svg(filename)
print(f"Saved: {filename}")
d  # Display as SVG


# In[ ]:




