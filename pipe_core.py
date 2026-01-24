import drawsvg as draw
import random
import math

# ============================================================================
# PIPE CONNECTION LOGIC
# ============================================================================

OPENINGS = {
    'r': {'S', 'E'},
    '7': {'S', 'W'},
    'j': {'N', 'W'},
    'L': {'N', 'E'},
    '|': {'N', 'S'},
    '-': {'E', 'W'},
    '+': {'N', 'S', 'E', 'W'},
}

ALL_CHARS = set(OPENINGS.keys())


def has_opening(ch, direction):
    return direction in OPENINGS.get(ch, set())


def get_compatible_neighbors(ch, direction):
    opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    opp_dir = opposite[direction]
    ch_has_opening = has_opening(ch, direction)
    compatible = set()
    for candidate in ALL_CHARS:
        if ch_has_opening == has_opening(candidate, opp_dir):
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
    checks = [
        ((x, y, 'r'), (x+1, y, '7'), (x, y+1, 'L'), (x+1, y+1, 'j')),
        ((x-1, y, 'r'), (x, y, '7'), (x-1, y+1, 'L'), (x, y+1, 'j')),
        ((x, y-1, 'r'), (x+1, y-1, '7'), (x, y, 'L'), (x+1, y, 'j')),
        ((x-1, y-1, 'r'), (x, y-1, '7'), (x-1, y, 'L'), (x, y, 'j')),
    ]
    for square in checks:
        my_pos = None
        for (px, py, expected) in square:
            if px == x and py == y:
                my_pos = (px, py, expected)
                break
        if my_pos is None:
            continue
        _, _, expected_ch = my_pos
        if ch != expected_ch:
            continue
        all_fixed = True
        for (px, py, expected) in square:
            if px == x and py == y:
                continue
            if px < 0 or px >= width or py < 0 or py >= height:
                all_fixed = False
                break
            cell_poss = poss_grid[px][py]
            if len(cell_poss) == 1 and expected in cell_poss:
                continue
            all_fixed = False
            break
        if all_fixed:
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


def collapse_cell(poss_grid, x, y):
    possibilities = list(poss_grid[x][y])
    if possibilities:
        weights = []
        for ch in possibilities:
            if ch == '+':
                weights.append(2)
            elif ch in 'r7jL':
                weights.append(3)
            else:
                weights.append(1)
        chosen = random.choices(possibilities, weights=weights, k=1)[0]
        poss_grid[x][y] = {chosen}
        return True
    return False


def wave_function_collapse(width, height, max_iterations=10000, max_attempts=20):
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
            if not collapse_cell(poss_grid, x, y):
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


def build_corner_tile(stroke_width, shading_style, shading_stroke_width):
    group = draw.Group(id='l_corner', fill='white')
    mask = draw.Mask()
    box = draw.Circle(40, 40, 10, fill='green')
    mask.append(box)
    group.append(draw.Arc(40, 40, 70, 180, 270, cw=True, stroke='black', stroke_width=stroke_width, fill='white'))
    group.append(draw.Lines(40, -30, -30, 40, 30, 40, 40, 30, fill='white', stroke='none', mask=not mask))
    group.append(draw.Circle(40, 40, 10, fill='white'))
    group.append(draw.Arc(40, 40, 10, 180, 270, cw=True, stroke='black', stroke_width=stroke_width, fill='none'))
    group.append(draw.Rectangle(40, -35, 5, 70, fill='white', stroke='black', stroke_width=stroke_width))
    group.append(draw.Rectangle(-35, 40, 70, 5, fill='white', stroke='black', stroke_width=stroke_width))
    group.append(draw.Rectangle(46, -29, 5, 58, fill='white', stroke='none'))
    group.append(draw.Line(45, 30, 50, 30, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Line(45, -30, 50, -30, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Rectangle(-29, 46, 58, 5, fill='white', stroke='none'))
    group.append(draw.Line(30, 45, 30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Line(-30, 45, -30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    add_corner_shading(group, shading_style, shading_stroke_width)
    return group


def build_tube_tile(stroke_width, shading_style, shading_stroke_width):
    group = draw.Group(id='l_tube', fill='none')
    group.append(draw.Rectangle(-30, -50, 60, 100, fill='white', stroke='none'))
    group.append(draw.Line(-30, -50, -30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    group.append(draw.Line(30, -50, 30, 50, fill='none', stroke='black', stroke_width=stroke_width))
    add_tube_shading(group, shading_style, shading_stroke_width)
    return group


def render_svg(grid, stroke_width, shading_style, shading_stroke_width):
    """Render a grid of pipe characters to an SVG string."""
    width = len(grid)
    height = len(grid[0]) if width > 0 else 0

    canvas_w = width * 100
    canvas_h = height * 100
    d = draw.Drawing(canvas_w, canvas_h, origin='center', displayInline=False)

    l_corner = build_corner_tile(stroke_width, shading_style, shading_stroke_width)
    l_tube = build_tube_tile(stroke_width, shading_style, shading_stroke_width)

    half_w = width // 2
    half_h = height // 2

    for x in range(width):
        for y in range(height):
            ch = grid[x][y]
            xloc = (x - half_w) * 100
            yloc = (y - half_h) * 100
            trans = "translate({},{})".format(xloc, yloc)

            if ch == "r":
                d.append(draw.Use(l_corner, 0, 0, transform=trans))
            elif ch == "7":
                d.append(draw.Use(l_corner, 0, 0, transform=trans + " rotate(90)"))
            elif ch == "j":
                d.append(draw.Use(l_corner, 0, 0, transform=trans + " rotate(180)"))
            elif ch == "L":
                d.append(draw.Use(l_corner, 0, 0, transform=trans + " rotate(270)"))
            elif ch == "|":
                d.append(draw.Use(l_tube, 0, 0, transform=trans))
            elif ch == "-":
                d.append(draw.Use(l_tube, 0, 0, transform=trans + " rotate(90)"))
            elif ch == "+":
                k = random.randint(0, 1)
                if k == 0:
                    d.append(draw.Use(l_tube, 0, 0, transform=trans))
                    d.append(draw.Use(l_tube, 0, 0, transform=trans + " rotate(90)"))
                else:
                    d.append(draw.Use(l_tube, 0, 0, transform=trans + " rotate(90)"))
                    d.append(draw.Use(l_tube, 0, 0, transform=trans))

    return d.as_svg()
