"""
ARC SCENE registry -- Gf Phase-2 (batch 1): the extensible perceptual VOCABULARY.

The genuine remaining fluid-intelligence move (docs/GF_PERCEPTION_INDUCTION_RESEARCH.md) is inducing
primitives OUTSIDE the DSL closure FROM PERCEPTION. Phase 1 proved (95% oracle ceiling) the missing
concepts are expressible from generic MEASUREMENTS; Phase 2 replaces arc_solver's CLOSED 5-key object
schema {cells,color,size,bbox,shape} with an extensible attribute/relation SCENE built from a FROZEN,
pre-registered set of measurement GENERATORS (the 13-generator manifest in the roadmap).

This module is ADDITIVE + backward-compatible: it REUSES arc_solver.components() as the seed (G1 CONN)
and NEVER mutates it or _BASE (arc_diagnostic depends on _BASE stability). It only MEASURES -- there is
no synthesis and no ARC-eval change here; the Phase-3 mint will CONSUME these measurements to hypothesize
new typed primitives. Batch 1: G1 CONN (abstraction lattices), G4 PALETTE, G5 MORPH, G6 TOPO, G7 SHAPE-EQ,
G8 ALIGN/ADJ. Batch 2: G2 LATTICE, G3 PERIOD, G9 DIST (Voronoi), G10 PROG, G11 SYM (score+witness),
G12 LOCI (cellsets), G13 COLOR-PAIRING, plus the G7 shape-correspondence + G8 cellset-locus WITNESSES.
Per the frozen manifest, symmetry/shape-equivalence return SCORE+WITNESS (never a magic-cutoff boolean)
and loci return CELLSETS; the only constants are domain-definitional (colours 0-9, connectivity {4,8},
the D8 group, grid bounds).

HONEST (from the Phase-2 de-risk): measurement generators cover ~70% of the Phase-1 concept gap; the other
~30% are OPERATION-not-measurement (constructive geometry -- stamp constructors, sequential path, stateful
transport) which NO measurement supplies -> Phase-3 must grow an operation-synthesis arm or accept the cap.

Determinism (CLAUDE.md): pure functions of the grid; zlib.crc32 for the shape-class id (never builtin
hash); sorted outputs; ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.arc_scene
"""

import zlib
from collections import Counter, deque

from snn_data_lake.reasoning.arc_solver import background, components

# abstraction lattices for G1 CONN (ARGA-style competing object abstractions)
CONN_ABSTRACTIONS = ((4, True), (8, True), (4, False), (8, False))


# ---- G4 PALETTE ---------------------------------------------------------------------------------------
def palette(g):
    """Grid colour statistics: background, per-colour counts, modal, rarest, ABSENT (free) colours."""
    cnt = Counter(v for row in g for v in row)
    present = set(cnt)
    return {
        "background": background(g),
        "counts": dict(cnt),
        "modal": max(cnt, key=lambda c: (cnt[c], -c)),
        "rarest": min(cnt, key=lambda c: (cnt[c], c)),
        "absent": tuple(c for c in range(10) if c not in present),  # free colours (markers/fills)
        "n_colors": len(present),
    }


# ---- G5 MORPH (per object) ----------------------------------------------------------------------------
def morph(obj):
    """Morphology of one object: bbox h/w/side/area, solid-rectangle test, interior(8-erosion)/boundary,
    line test. interior = cells all of whose 8 neighbours are also object cells."""
    r0, c0, r1, c1 = obj["bbox"]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    cells = obj["cells"]
    interior = frozenset(
        (y, x)
        for (y, x) in cells
        if all((y + dy, x + dx) in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    )
    return {
        "height": h,
        "width": w,
        "side": min(h, w),
        "area_bbox": h * w,
        "is_solid_rect": obj["size"] == h * w,
        "is_line": h == 1 or w == 1,
        "interior": interior,
        "boundary": cells - interior,
    }


# ---- G6 TOPO (per object: enclosed holes) -------------------------------------------------------------
def object_holes(obj):
    """Enclosed holes of one object: cells inside the (1-padded) bbox that are NOT the object and are NOT
    reachable from the padded border by 4-connected non-object moves. Returns (list of hole cell-sets in
    original grid coords, n_holes). A donut has 1 hole; a solid blob has 0."""
    r0, c0, r1, c1 = obj["bbox"]
    cells = obj["cells"]
    H, W = (r1 - r0 + 3), (c1 - c0 + 3)  # padded local frame

    def local(y, x):
        return (y - r0 + 1, x - c0 + 1)

    wall = {local(y, x) for (y, x) in cells}
    outside = set()
    q = deque()
    for y in range(H):
        for x in (0, W - 1):
            if (y, x) not in wall:
                q.append((y, x))
                outside.add((y, x))
    for x in range(W):
        for y in (0, H - 1):
            if (y, x) not in wall and (y, x) not in outside:
                q.append((y, x))
                outside.add((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in wall and (ny, nx) not in outside:
                outside.add((ny, nx))
                q.append((ny, nx))
    # enclosed = local non-wall cells not in `outside`; group into 4-connected hole regions
    enclosed = {
        (y, x) for y in range(H) for x in range(W) if (y, x) not in wall and (y, x) not in outside
    }
    holes, seen = [], set()
    for cell in sorted(enclosed):
        if cell in seen:
            continue
        comp, stack = [], [cell]
        seen.add(cell)
        while stack:
            y, x = stack.pop()
            comp.append((y + r0 - 1, x + c0 - 1))  # back to grid coords
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc = (y + dy, x + dx)
                if nc in enclosed and nc not in seen:
                    seen.add(nc)
                    stack.append(nc)
        holes.append(frozenset(comp))
    return holes, len(holes)


# ---- G7 SHAPE-EQ (canonical shape class under the 8 dihedral symmetries) ------------------------------
def _normalize(pts):
    ys = [y for y, _ in pts]
    xs = [x for _, x in pts]
    my, mx = min(ys), min(xs)
    return tuple(sorted((y - my, x - mx) for y, x in pts))


def _d8(pts):
    """The 8 dihedral images of a point set (rot 0/90/180/270 x optional flip)."""
    out = []
    cur = list(pts)
    for _ in range(4):
        cur = [(x, -y) for y, x in cur]  # rotate 90
        out.append(_normalize(cur))
        out.append(_normalize([(y, -x) for y, x in cur]))  # + horizontal flip
    return out


def shape_class(shape):
    """A rotation/reflection-INVARIANT class id for a shape (frozenset of (dy,dx)): crc32 of the
    lexicographically-smallest of its 8 dihedral normalizations. Two objects with the same shape up to
    D8 symmetry share this id -- the basis for non-contiguous same-shape grouping."""
    canon = min(_d8(shape))
    return zlib.crc32(repr(canon).encode("ascii"))


# ---- G8 ALIGN/ADJ (pairwise object relations, bbox-level) --------------------------------------------
def relations(objs):
    """Pairwise relations (i<j) over the object list: same_row / same_col (bbox overlap on that axis),
    bbox_adjacent (bboxes touch or are 1 apart), contains (one bbox strictly inside the other)."""
    rels = []
    for i in range(len(objs)):
        ar0, ac0, ar1, ac1 = objs[i]["bbox"]
        for j in range(i + 1, len(objs)):
            br0, bc0, br1, bc1 = objs[j]["bbox"]
            row_overlap = not (ar1 < br0 or br1 < ar0)
            col_overlap = not (ac1 < bc0 or bc1 < ac0)
            gap_r = max(0, max(ar0, br0) - min(ar1, br1))
            gap_c = max(0, max(ac0, bc0) - min(ac1, bc1))
            a_in_b = br0 <= ar0 and ac0 >= bc0 and ar1 <= br1 and ac1 <= bc1
            b_in_a = ar0 <= br0 and ac0 <= bc0 and br1 <= ar1 and bc1 <= ac1
            r = {"i": i, "j": j}
            if row_overlap:
                r["same_row"] = True
            if col_overlap:
                r["same_col"] = True
            if gap_r <= 1 and gap_c <= 1:
                r["bbox_adjacent"] = True
            if a_in_b or b_in_a:
                r["contains"] = (j, i) if a_in_b else (i, j)  # (outer, inner)
            if len(r) > 2:
                rels.append(r)
    return rels


# ---- the SCENE (registry: base object attrs + generator measurements + relations) --------------------
def scene(g, conn=4, same_color=True):
    """Rich perceptual scene of a grid under one CONN abstraction: grid palette + objects (base schema
    EXTENDED with morph/topo/shape_id) + pairwise relations. Additive: base attrs are unchanged, new keys
    are appended. Use CONN_ABSTRACTIONS to compute competing abstractions."""
    bg = background(g)
    objs = []
    for o in components(g, conn, same_color, bg):
        o = dict(o)  # copy: NEVER mutate the cached components() object
        o.update(morph(o))
        o["holes"], o["n_holes"] = object_holes(o)
        o["shape_id"] = shape_class(o["shape"])
        objs.append(o)
    # G7 grouping: object indices sharing a shape_id (non-contiguous same-shape groups)
    groups = {}
    for idx, o in enumerate(objs):
        groups.setdefault(o["shape_id"], []).append(idx)
    return {
        "grid": palette(g),
        "objects": objs,
        "relations": relations(objs),
        "shape_groups": {k: v for k, v in groups.items() if len(v) > 1},
        "n_objects": len(objs),
        "conn": (conn, same_color),
    }


# ================================================================================================
# Gf Phase-2 BATCH 2 -- the remaining frozen generators + G7/G8 witnesses. Same additive,
# measurement-only, deterministic contract as batch 1; reuses components()/background/_d8/shape_class,
# reimplements no segmentation. Return-type policy (the brain-native rule applied to measurement):
# symmetry/shape-equivalence return a SCORE in [0,1] + a WITNESS correspondence (never a magic-cutoff
# boolean); loci/distance return CELLSETS; the only constants are domain-definitional (colours 0-9,
# connectivity {4,8}, the D8 group, grid bounds). No synthesis, no ARC-eval change.
# ================================================================================================

# ---- G11 SYM + G7 shape-correspondence (scored + witnessed) -----------------------------------
# The 8 dihedral group elements as LABELLED point transforms, same rot/flip convention as _d8.
_D8_LABELED = (
    ("identity", lambda p: (p[0], p[1])),
    ("rot90", lambda p: (p[1], -p[0])),
    ("rot180", lambda p: (-p[0], -p[1])),
    ("rot270", lambda p: (-p[1], p[0])),
    ("flip_h", lambda p: (p[0], -p[1])),
    ("transpose_main", lambda p: (p[1], p[0])),
    ("flip_v", lambda p: (-p[0], p[1])),
    ("transpose_anti", lambda p: (-p[1], -p[0])),
)


def _apply_norm(shape, t):
    """Apply point-transform t to a set of (dy,dx) coords, then min-corner normalise. Returns
    (normalised frozenset, dict {original_coord: normalised_coord})."""
    img = [(p, t(p)) for p in shape]
    my = min(b[0] for _, b in img)
    mx = min(b[1] for _, b in img)
    mapping = {p: (b[0] - my, b[1] - mx) for p, b in img}
    return frozenset(mapping.values()), mapping


def symmetry_axes(g):
    """G11 SYM. Whole-grid symmetry evidence under each candidate D8 axis: vertical mirror
    (r,c)->(r,W-1-c), horizontal mirror (r,c)->(H-1-r,c), the main + anti-diagonal transposes (ONLY
    when H==W), and 180-degree rotation. Each distinct-cell pair x<->T(x) is scored once; `score` in
    [0,1] = matching / comparable, `bg_wildcard_score` also counts a pair as matching when either cell
    holds the background colour (occlusion-aware). Returns a list of dicts, one per transform:
    {transform, score, bg_wildcard_score, correspondence: frozenset of ((rA,cA),(rB,cB)) pairs}. Never
    a bare boolean -- the consumer decides significance. Empty grid -> []; 1x1 -> every score 0.0."""
    h = len(g)
    w = len(g[0]) if h else 0
    if h == 0 or w == 0:
        return []
    bg = background(g)
    transforms = [
        ("vertical_mirror", lambda r, c: (r, w - 1 - c)),
        ("horizontal_mirror", lambda r, c: (h - 1 - r, c)),
    ]
    if h == w:  # diagonal transposes are only in-bounds on a square grid
        transforms.append(("transpose_main", lambda r, c: (c, r)))
        transforms.append(("transpose_anti", lambda r, c: (w - 1 - c, h - 1 - r)))
    transforms.append(("rotate_180", lambda r, c: (h - 1 - r, w - 1 - c)))
    out = []
    for label, tf in transforms:
        seen = set()
        pairs = []
        for r in range(h):
            for c in range(w):
                a = (r, c)
                b = tf(r, c)
                if a == b:  # fixed point: not a distinct-cell correspondence
                    continue
                key = (a, b) if a <= b else (b, a)  # canonical: each unordered pair once
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        comparable = len(pairs)
        if comparable == 0:
            out.append(
                {
                    "transform": label,
                    "score": 0.0,
                    "bg_wildcard_score": 0.0,
                    "correspondence": frozenset(),
                }
            )
            continue
        match = 0
        wild = 0
        for a, b in pairs:
            va = g[a[0]][a[1]]
            vb = g[b[0]][b[1]]
            if va == vb:
                match += 1
                wild += 1
            elif va == bg or vb == bg:  # occlusion-aware: background can hide the mirror partner
                wild += 1
        out.append(
            {
                "transform": label,
                "score": match / comparable,
                "bg_wildcard_score": wild / comparable,
                "correspondence": frozenset(pairs),
            }
        )
    return out


def shape_match(obj_a, obj_b):
    """G7 witness. Shape-correspondence between two components() objects. If they do NOT share
    shape_class() (the D8-invariant id) return None. Otherwise find the first D8 transform that maps
    A's shape EXACTLY onto B's and return {transform, correspondence: dict {(rA,cA):(rB,cB)}} -- a full
    bijection of A's grid-cells onto B's. Returns None if (despite equal class ids) no transform aligns
    (crc32-collision guard), so a non-None result is always a verified alignment."""
    if shape_class(obj_a["shape"]) != shape_class(obj_b["shape"]):
        return None
    ar0, ac0 = obj_a["bbox"][0], obj_a["bbox"][1]
    br0, bc0 = obj_b["bbox"][0], obj_b["bbox"][1]
    a_coord_cell = {(y - ar0, x - ac0): (y, x) for (y, x) in obj_a["cells"]}
    b_coord_cell = {(y - br0, x - bc0): (y, x) for (y, x) in obj_b["cells"]}
    bshape = frozenset(b_coord_cell)
    ashape = frozenset(a_coord_cell)
    for label, t in _D8_LABELED:
        normset, mapping = _apply_norm(ashape, t)
        if normset == bshape:
            corr = {}
            for coord in sorted(mapping):  # deterministic insertion order
                corr[a_coord_cell[coord]] = b_coord_cell[mapping[coord]]
            return {"transform": label, "correspondence": corr}
    return None  # class ids collided but no transform aligns -> not actually the same shape


# ---- G12 LOCI (cellsets) + G8 witness relations -----------------------------------------------
def _bresenham_line(a, b):
    """Integer Bresenham lattice line from cell a to cell b, inclusive of both ends, ordered."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    out = []
    while True:
        out.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return out


def segment(a, b):
    """G12 LOCI. The straight lattice line between cells a and b (integer Bresenham), inclusive of
    both ends, as a cellset (frozenset of (r,c)). UNDIRECTED: the endpoints are canonically ordered
    before rasterising, so segment(a,b) == segment(b,a) (required by object_locus/crossings, whose
    relation is symmetric). a == b -> {a}."""
    lo, hi = (a, b) if a <= b else (b, a)
    return frozenset(_bresenham_line(lo, hi))


def ray(cell, drow, dcol, height, width):
    """G12 LOCI. Cells visited stepping (drow,dcol) from `cell` (EXCLUSIVE of it) until the step
    leaves the height x width grid, as a cellset. Zero step or empty grid -> empty (no infinite loop)."""
    if height <= 0 or width <= 0 or (drow == 0 and dcol == 0):
        return frozenset()
    r, c = cell
    out = []
    r += drow
    c += dcol
    while 0 <= r < height and 0 <= c < width:
        out.append((r, c))
        r += drow
        c += dcol
    return frozenset(out)


def row_line(r, width):
    """G12 LOCI. The full horizontal line at row r across `width` columns, as a cellset."""
    return frozenset((r, c) for c in range(width)) if width > 0 else frozenset()


def col_line(c, height):
    """G12 LOCI. The full vertical line at column c across `height` rows, as a cellset."""
    return frozenset((r, c) for r in range(height)) if height > 0 else frozenset()


def perimeter(height, width):
    """G12 LOCI. The border ring of a height x width grid (row 0, row H-1, col 0, col W-1), a cellset.
    3x3 -> 8 cells; 1x1 -> 1; empty -> empty."""
    if height <= 0 or width <= 0:
        return frozenset()
    ring = set()
    for c in range(width):
        ring.add((0, c))
        ring.add((height - 1, c))
    for r in range(height):
        ring.add((r, 0))
        ring.add((r, width - 1))
    return frozenset(ring)


def locus_meet(s1, s2):
    """G12 LOCI. Set intersection of two cellsets -- where two loci meet. frozenset of (r,c)."""
    return frozenset(s1) & frozenset(s2)


def _obj_centroid(obj):
    """Centroid of an object as an integer cell (round(mean row), round(mean col)); empty -> None."""
    cells = obj["cells"]
    n = len(cells)
    if n == 0:
        return None
    sr = sum(r for r, _ in cells)
    sc = sum(c for _, c in cells)
    return (round(sr / n), round(sc / n))


def object_locus(obj_a, obj_b):
    """G8 witness. The segment cellset joining the two objects' centroids (each rounded to an integer
    cell) -- the geometric line-of-sight witness. frozenset of (r,c); empty if either object has no
    cells. Symmetric (segment canonicalises endpoint order)."""
    ca = _obj_centroid(obj_a)
    cb = _obj_centroid(obj_b)
    if ca is None or cb is None:
        return frozenset()
    return segment(ca, cb)


def crossings(obj_a, obj_b, others):
    """G8 witness. Which objects lie ON the centroid line between obj_a and obj_b: a frozenset of
    INDICES into `others` whose cells intersect object_locus(obj_a, obj_b) (object dicts are
    unhashable, so indices are the deterministic witness). Empty if the locus is empty or nothing hits."""
    locus = object_locus(obj_a, obj_b)
    if not locus:
        return frozenset()
    return frozenset(idx for idx, o in enumerate(others) if o["cells"] & locus)


def border_pairs(objs, height, width):
    """G8 witness. Pairs (i,j), i<j, of object indices on OPPOSITE grid borders: one on row 0 and the
    other on row H-1, OR one on col 0 and the other on col W-1. frozenset of (i,j). Top/bottom counts
    only when H>=2, left/right only when W>=2; empty grid / <2 objects -> empty."""
    if height <= 0 or width <= 0:
        return frozenset()
    top, bottom, left, right = set(), set(), set(), set()
    for i, o in enumerate(objs):
        rows = {r for r, _ in o["cells"]}
        cols = {c for _, c in o["cells"]}
        if 0 in rows:
            top.add(i)
        if (height - 1) in rows:
            bottom.add(i)
        if 0 in cols:
            left.add(i)
        if (width - 1) in cols:
            right.add(i)
    pairs = set()
    n = len(objs)
    for i in range(n):
        for j in range(i + 1, n):
            vert = height >= 2 and ((i in top and j in bottom) or (i in bottom and j in top))
            horiz = width >= 2 and ((i in left and j in right) or (i in right and j in left))
            if vert or horiz:
                pairs.add((i, j))
    return frozenset(pairs)


# ---- G2 LATTICE + G3 PERIOD -------------------------------------------------------------------
def _merge_lines(line_colors):
    """Interval-merge a per-line colour list (colour, or None for a non-monochrome line) into sorted
    (start, end, colour) bands; consecutive lines of the SAME colour coalesce."""
    bands = []
    n = len(line_colors)
    i = 0
    while i < n:
        col = line_colors[i]
        if col is None:
            i += 1
            continue
        j = i
        while j + 1 < n and line_colors[j + 1] == col:
            j += 1
        bands.append((i, j, col))
        i = j + 1
    return bands


def _sep_segments(n, sep_idx):
    """Contiguous runs of range(n) that are NOT separators, as (start, end) pairs."""
    segs = []
    start = None
    for i in range(n):
        if i in sep_idx:
            if start is not None:
                segs.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        segs.append((start, n - 1))
    return segs


def separators(g):
    """G2 LATTICE. Full monochrome ROW and COLUMN lines that carve the grid into rectangular PANELS;
    consecutive same-colour lines interval-merge into bands. Returns {row_separators: sorted
    [(start,end,color)], col_separators: same, panels: sorted [(r0,c0,r1,c1)] of the maximal
    non-separator rectangles, separator_colors: sorted tuple}. A separator's colour is the line's own
    colour, never hardcoded. Empty grid -> all-empty."""
    h = len(g)
    w = len(g[0]) if h else 0
    if h == 0 or w == 0:
        return {"row_separators": [], "col_separators": [], "panels": [], "separator_colors": ()}
    row_color = []
    for r in range(h):
        first = g[r][0]
        row_color.append(first if all(g[r][c] == first for c in range(w)) else None)
    col_color = []
    for c in range(w):
        first = g[0][c]
        col_color.append(first if all(g[r][c] == first for r in range(h)) else None)
    row_bands = _merge_lines(row_color)
    col_bands = _merge_lines(col_color)
    row_sep_idx = set()
    for s, e, _ in row_bands:
        row_sep_idx.update(range(s, e + 1))
    col_sep_idx = set()
    for s, e, _ in col_bands:
        col_sep_idx.update(range(s, e + 1))
    row_segs = _sep_segments(h, row_sep_idx)
    col_segs = _sep_segments(w, col_sep_idx)
    panels = sorted((r0, c0, r1, c1) for (r0, r1) in row_segs for (c0, c1) in col_segs)
    colors = tuple(sorted({col for _, _, col in row_bands} | {col for _, _, col in col_bands}))
    return {
        "row_separators": row_bands,
        "col_separators": col_bands,
        "panels": panels,
        "separator_colors": colors,
    }


def _modal(cnt):
    """Modal colour of a Counter; ties resolve to the smallest colour (deterministic)."""
    return min(cnt, key=lambda col: (-cnt[col], col))


def _consensus_template(g, ph, pw, h, w):
    """The ph x pw consensus template: template[i][j] = the MODAL colour over every tile copy (all
    (r,c) with r%ph==i and c%pw==j)."""
    rows = []
    for i in range(ph):
        row = []
        for j in range(pw):
            cnt = Counter()
            r = i
            while r < h:
                grow = g[r]
                c = j
                while c < w:
                    cnt[grow[c]] += 1
                    c += pw
                r += ph
            row.append(_modal(cnt))
        rows.append(tuple(row))
    return tuple(rows)


def period(g):
    """G3 PERIOD. The smallest STRICTLY-SUB-FULL tiling template that best reproduces g. For every
    (ph,pw) with ph*pw < H*W the per-cell MODAL colour over all tile copies forms a consensus template;
    exactness = fraction of grid cells equal to it. Selection maximises exactness, then minimises area,
    then ph. EXCLUDING the full grid (which tiles itself trivially at exactness 1.0) is what makes
    exactness INFORMATIVE: an exactly periodic grid returns its fundamental tile at exactness 1.0, an
    aperiodic grid returns its best small approximation at exactness < 1.0. `periodic` = exactness==1.0.
    A 1x1 grid (no sub-full period) returns itself, periodic False. Returns
    {ph, pw, template, exactness, periodic}."""
    h = len(g)
    w = len(g[0]) if h else 0
    if h == 0 or w == 0:
        return {"ph": 0, "pw": 0, "template": (), "exactness": 0.0, "periodic": False}
    total = h * w
    if total == 1:
        return {"ph": 1, "pw": 1, "template": ((g[0][0],),), "exactness": 1.0, "periodic": False}
    best_key = None
    best = None
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if (
                ph * pw >= total
            ):  # exclude the trivial full-grid tiling (only sub-full periods count)
                continue
            template = _consensus_template(g, ph, pw, h, w)
            match = 0
            for r in range(h):
                trow = template[r % ph]
                grow = g[r]
                for c in range(w):
                    if grow[c] == trow[c % pw]:
                        match += 1
            key = (match, -(ph * pw), -ph)  # max match, then smallest area, then smallest ph
            if best_key is None or key > best_key:
                best_key = key
                best = (ph, pw, template, match)
    ph, pw, template, match = best
    ex = match / total
    return {"ph": ph, "pw": pw, "template": template, "exactness": ex, "periodic": ex == 1.0}


# ---- G9 DIST + G10 PROG + G13 COLOR-PAIRING ---------------------------------------------------
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def voronoi(g):
    """G9 DIST. Multi-source 4-connected BFS seeded SIMULTANEOUSLY from every non-background object
    cell (objects from components(g), tagged by list index). Returns {'owner': {(r,c): object_index},
    'distance': {(r,c): hops}} for BACKGROUND cells only; ties break toward the smaller object index.
    Empty grid / no objects / unreachable cells -> those cells simply absent (never a crash)."""
    h = len(g)
    w = len(g[0]) if h else 0
    if h == 0 or w == 0:
        return {"owner": {}, "distance": {}}
    bg = background(g)
    objs = components(g, 4, True, bg)
    owner = {}
    dist = {}
    frontier = []
    for idx, o in enumerate(objs):
        for cell in o["cells"]:
            owner[cell] = idx
            dist[cell] = 0
            frontier.append(cell)
    d = 0
    while frontier:
        proposals = {}  # (r,c) -> smallest owning index reaching it at the next level
        for r, c in frontier:
            own = owner[(r, c)]
            for dy, dx in _N4:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and g[nr][nc] == bg:
                    prev = proposals.get((nr, nc))
                    if prev is None or own < prev:
                        proposals[(nr, nc)] = own
        d += 1
        frontier = []
        for cell in sorted(proposals):
            owner[cell] = proposals[cell]
            dist[cell] = d
            frontier.append(cell)
    bg_owner = {}
    bg_dist = {}
    for cell in sorted(owner):
        if g[cell[0]][cell[1]] == bg:
            bg_owner[cell] = owner[cell]
            bg_dist[cell] = dist[cell]
    return {"owner": bg_owner, "distance": bg_dist}


def _collinear(distinct):
    """True iff sorted DISTINCT points all lie on one straight line (zero integer cross product)."""
    if len(distinct) <= 2:
        return True
    (ay, ax), (by, bx) = distinct[0], distinct[1]
    dr, dc = by - ay, bx - ax
    for py, px in distinct[2:]:
        if dr * (px - ax) - dc * (py - ay) != 0:
            return False
    return True


def progression(points):
    """G10 PROG. Geometry of a point list under lexicographic sort: {'is_progression': all consecutive
    step vectors exactly equal (True only for >=2 points), 'step': that common (dr,dc) or None,
    'collinear': all points on one line, 'regularity': fraction in [0,1] of consecutive steps equal to
    the MODAL step (0.0 for <2 points)}. Pure integer geometry, no magic constants."""
    pts = sorted(points)
    n = len(pts)
    if n < 2:
        return {"is_progression": False, "step": None, "collinear": n == 1, "regularity": 0.0}
    steps = [(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(n - 1)]
    cnt = Counter(steps)
    top = max(cnt.values())
    modal = min(s for s, k in cnt.items() if k == top)  # tie -> smallest step tuple
    regularity = cnt[modal] / len(steps)
    all_equal = len(cnt) == 1
    return {
        "is_progression": all_equal,
        "step": steps[0] if all_equal else None,
        "collinear": _collinear(sorted(set(pts))),
        "regularity": regularity,
    }


def color_pairing(obj, g):
    """G13 COLOR-PAIRING. Read a legend/key object into a colour correspondence. The object's SHORT
    bbox axis must be exactly 2 cells thick (else {}); for each line along the LONG axis, pair the
    colour at one short-axis end with the colour at the other -> {colorA: colorB} (first mapping per
    colorA wins). Empty/malformed object (no cells or no bbox) -> {}."""
    bbox = obj.get("bbox") if obj else None
    if not obj or not obj.get("cells") or bbox is None:
        return {}
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) != 2:  # not 2-thick along its short axis
        return {}
    cells = obj["cells"]
    pairing = {}
    if h <= w:  # short axis = rows; iterate columns along the long axis
        for c in range(c0, c1 + 1):
            if (r0, c) in cells and (r1, c) in cells:
                pairing.setdefault(g[r0][c], g[r1][c])
    else:  # short axis = columns; iterate rows along the long axis
        for r in range(r0, r1 + 1):
            if (r, c0) in cells and (r, c1) in cells:
                pairing.setdefault(g[r][c0], g[r][c1])
    return pairing


# ================================================================================================
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def _selfcheck_batch2():
    """Batch-2 generator asserts (ported from the four authored+adversarially-verified clusters, with
    the three merge fixes baked in: symmetric segment, informative sub-full period, bbox-guarded
    color_pairing)."""
    # G11 SYM: perfect vertical symmetry scores 1.0 with a witness; a horizontal axis on it scores <1.
    sym_g = ((1, 2, 1), (3, 0, 3), (4, 5, 4))
    axes = {d["transform"]: d for d in symmetry_axes(sym_g)}
    assert (
        axes["vertical_mirror"]["score"] == 1.0
        and len(axes["vertical_mirror"]["correspondence"]) == 3
    )
    assert axes["horizontal_mirror"]["score"] < 1.0
    bar = ((1, 0, 0), (1, 0, 0), (1, 0, 0))  # occlusion-aware: bg-wildcard rescues the mirror
    vb = {d["transform"]: d for d in symmetry_axes(bar)}["vertical_mirror"]
    assert vb["score"] == 0.0 and vb["bg_wildcard_score"] == 1.0
    assert symmetry_axes(()) == [] and symmetry_axes(((),)) == []
    # G7 shape_match: D8-equivalent Ls align with a non-identity bijection; a domino does not match.
    la = ((5, 0, 0), (5, 5, 0))
    lb = ((6, 6, 0), (0, 6, 0))
    oa = components(la, 4, True, background(la))[0]
    ob = components(lb, 4, True, background(lb))[0]
    mm = shape_match(oa, ob)
    assert mm is not None and set(mm["correspondence"]) == set(oa["cells"])
    assert set(mm["correspondence"].values()) == set(ob["cells"])
    dom = components(((7, 7, 0),), 4, True, background(((7, 7, 0),)))[0]
    assert shape_match(oa, dom) is None
    # G12 LOCI: exact Bresenham + SYMMETRY (the merge fix) + ray/perimeter/meet.
    assert segment((0, 0), (3, 3)) == frozenset({(0, 0), (1, 1), (2, 2), (3, 3)})
    assert segment((0, 0), (2, 4)) == segment((2, 4), (0, 0)), (
        "segment must be undirected (merge fix)"
    )
    assert ray((1, 1), 1, 0, 4, 4) == frozenset({(2, 1), (3, 1)})
    assert ray((1, 1), 0, 0, 4, 4) == frozenset() and len(perimeter(3, 3)) == 8
    assert locus_meet(row_line(2, 5), col_line(2, 5)) == frozenset({(2, 2)})
    # G8 witnesses: a middle object crosses the centroid line; opposite-border pair; order-invariant.
    diagg = ((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 2, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 3))
    d3 = components(diagg, 8, True, background(diagg))
    a3, c3, b3 = d3[0], d3[1], d3[2]
    assert crossings(a3, b3, [c3]) == frozenset({0})
    assert crossings(a3, b3, [c3]) == crossings(b3, a3, [c3]), (
        "crossings must be order-invariant (fix)"
    )
    assert border_pairs(d3, 5, 5) == frozenset({(0, 2)})
    # G2 LATTICE: a separator row carves two panels.
    sepg = ((1, 2, 3), (4, 5, 6), (5, 5, 5), (7, 8, 9), (1, 0, 2))
    s = separators(sepg)
    assert s["row_separators"] == [(2, 2, 5)] and s["panels"] == [(0, 0, 1, 2), (3, 0, 4, 2)]
    # G3 PERIOD: periodic grid -> small period exactness 1.0; aperiodic -> exactness < 1.0 (the fix
    # makes exactness informative instead of a constant 1.0).
    per = period(((1, 2, 1, 2), (3, 4, 3, 4), (1, 2, 1, 2), (3, 4, 3, 4)))
    assert (per["ph"], per["pw"]) == (2, 2) and per["exactness"] == 1.0 and per["periodic"]
    ape = period(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    assert ape["exactness"] < 1.0 and not ape["periodic"] and (ape["ph"], ape["pw"]) != (3, 3)
    # G9 DIST: a Voronoi split with a deterministic tie toward the smaller index.
    vor = voronoi(((3, 0, 0, 0, 5, 0, 0, 0, 7),))
    assert vor["owner"][(0, 2)] == 0 and vor["owner"][(0, 6)] == 1 and vor["owner"][(0, 7)] == 2
    # G10 PROG: a perfect AP; a non-progression reports a fractional regularity.
    ap = progression([(3, 6), (0, 0), (2, 4), (1, 2)])
    assert ap["is_progression"] and ap["step"] == (1, 2) and ap["regularity"] == 1.0
    npr = progression([(0, 0), (1, 1), (2, 5)])
    assert not npr["is_progression"] and npr["regularity"] == 0.5
    # G13 COLOR-PAIRING: a 2xN legend; and the bbox-guard merge fix (malformed object -> {}).
    legend = ((2, 3, 4, 0), (5, 6, 7, 0))
    lo = components(legend, 8, False, background(legend))[0]
    assert color_pairing(lo, legend) == {2: 5, 3: 6, 4: 7}
    assert color_pairing({"cells": frozenset({(0, 0)})}, legend) == {}, (
        "missing bbox -> {} (merge fix)"
    )
    _emit(
        "  [OK] BATCH 2: G11 SYM(score+witness) G7(shape-corr) G12 LOCI(cellsets) G8(locus witness)"
    )
    _emit(
        "       G2 LATTICE G3 PERIOD(sub-full, informative) G9 DIST(Voronoi) G10 PROG G13 COLOR-PAIR"
    )


def main():
    width = 96
    _emit("=" * width)
    _emit("ARC SCENE registry -- Gf Phase-2 batch-1 measurement generators (perceptual vocabulary)")
    _emit("=" * width)

    # a toy grid: bg 0; a 3x3 hollow square (donut) of colour 3 top-left; two 1x1 dots of colour 5.
    g = (
        (3, 3, 3, 0, 5),
        (3, 0, 3, 0, 0),
        (3, 3, 3, 0, 5),
        (0, 0, 0, 0, 0),
    )
    sc = scene(g)

    # G4 PALETTE: bg 0, colours present {0,3,5}, absent includes 1,2,4,6..9; rarest colour count check
    pal = sc["grid"]
    assert pal["background"] == 0 and pal["n_colors"] == 3, pal
    assert 1 in pal["absent"] and 3 not in pal["absent"] and 5 not in pal["absent"], pal["absent"]
    _emit(
        f"  [OK] G4 PALETTE: bg={pal['background']} present={pal['n_colors']} absent[:4]={pal['absent'][:4]}"
    )

    # G6 TOPO: the donut has exactly ONE hole (its centre); the dots have none
    donut = max(sc["objects"], key=lambda o: o["size"])
    assert donut["n_holes"] == 1, donut["n_holes"]
    assert list(donut["holes"][0]) == [(1, 1)], donut["holes"]
    dots = [o for o in sc["objects"] if o["size"] == 1]
    assert all(o["n_holes"] == 0 for o in dots) and len(dots) == 2, "two solid dots, no holes"
    _emit(
        f"  [OK] G6 TOPO: donut has {donut['n_holes']} hole at {sorted(donut['holes'][0])}; dots have 0"
    )

    # G5 MORPH: the donut is NOT a solid rect (size 8 < bbox area 9); a dot IS a solid rect + a line
    assert not donut["is_solid_rect"] and donut["side"] == 3, donut
    assert dots[0]["is_solid_rect"] and dots[0]["is_line"], dots[0]
    _emit(
        f"  [OK] G5 MORPH: donut solid_rect={donut['is_solid_rect']} side={donut['side']}; dot is a line"
    )

    # G7 SHAPE-EQ: the two 1x1 dots share a shape class; a rotated L equals its mirror under D8
    assert dots[0]["shape_id"] == dots[1]["shape_id"], "identical shapes must share a class id"
    l1 = shape_class(frozenset({(0, 0), (1, 0), (1, 1)}))  # an L
    l2 = shape_class(frozenset({(0, 0), (0, 1), (1, 1)}))  # a 90-rotated/mirrored L
    assert l1 == l2, "shape class must be invariant under the 8 dihedral symmetries"
    diff = shape_class(frozenset({(0, 0), (0, 1)}))  # a domino -> a DIFFERENT class
    assert diff != l1, "a genuinely different shape must get a different class id"
    assert len(sc["shape_groups"]) == 1, "the two dots form one non-contiguous same-shape group"
    _emit(
        "  [OK] G7 SHAPE-EQ: D8-invariant class ids; rotated/mirrored L collapse; domino distinct; dots grouped"
    )

    # G8 ALIGN/ADJ: the two dots are in the SAME COLUMN (col 4) and share rows with the donut
    dot_idx = [i for i, o in enumerate(sc["objects"]) if o["size"] == 1]
    same_col = any(r.get("same_col") and {r["i"], r["j"]} == set(dot_idx) for r in sc["relations"])
    assert same_col, "the two vertically-aligned dots must be detected as same_col"
    _emit(
        f"  [OK] G8 ALIGN/ADJ: {len(sc['relations'])} relations; the two dots detected as same_col"
    )

    # G1 CONN: the 4 competing abstractions are available + deterministic
    counts = {ab: len(scene(g, *ab)["objects"]) for ab in CONN_ABSTRACTIONS}
    assert scene(g) == scene(g), "scene must be deterministic"
    _emit(f"  [OK] G1 CONN: 4 abstraction lattices available; object counts {counts}")

    _selfcheck_batch2()  # G2/G3/G9/G10/G11/G12/G13 + G7/G8 witnesses

    _emit("-" * width)
    _emit(
        "self-check OK: additive perceptual registry -- batch 1 (palette/morph/topo/shape-class/relations/"
    )
    _emit(
        "CONN) + batch 2 (SYM+witness, shape-correspondence, LOCI cellsets, locus witnesses, LATTICE,"
    )
    _emit(
        "PERIOD, Voronoi, PROG, COLOR-PAIRING), reusing components() as the seed, mutating nothing."
    )
    _emit(
        "HONEST: measurement ONLY (no synthesis, no ARC-eval change); the Phase-3 mint consumes these."
    )
    _emit(
        "The Phase-3 mint (MDL hypothesis + typed promotion) + its operation-synthesis arm remain."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
