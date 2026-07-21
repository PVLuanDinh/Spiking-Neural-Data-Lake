r"""ARC-domain TEST-TIME PROGRAM SYNTHESIS -- object perception + a grid-transformation DSL +
iterative-deepening search + library learning, scored by the honest arc_bench harness.

WHY THIS EXISTS
---------------
gf.py already does genuine test-time program synthesis (induce_program / OpenGf: iterative deepening
over a grammar, smallest consistent program, growing library) -- but over 1-D numeric/letter
SEQUENCES. ARC-AGI (Chollet 2019) asks for programs over 2-D GRIDS and the OBJECTS in them, and the
brain had no perception of that domain at all: the measured baseline of every live path was 0/100.
This module is the same induction idiom transplanted to the grid domain:

  1. PERCEPTION  : a grid is parsed into OBJECTS (connected components with colour/bbox/size/shape).
                   The component parser EXTENDS arc_agi/rule_inducer.py's connected_components /
                   parse_objects / cells_by_color (same BFS, same attribute style; re-implemented here
                   because that module's top-level `import env` makes it un-importable as a package
                   module -- adds 8-connectivity and multi-colour variants they lack).
  2. DSL         : ~70 small, TOTAL grid->grid primitives (rotate/flip/crop/scale/tile/mirror/object
                   select+delete/gravity/fill/frame/split-logic, plus OBJECT-CENTRIC ops over the
                   extracted components: per-object reflection, modal/odd shape+colour filters,
                   in-place isolation, component counting, bbox fill, reading-order selection,
                   colour-role cells, centering). Inapplicable -> None, never raises.
  3. SEARCH      : iterative deepening over COMPOSITIONS by program size, exactly the
                   gf.induce_program idiom: the SMALLEST program consistent with EVERY training pair
                   wins (MDL / Occam bias). Equivalence pruning (two prefixes producing identical
                   grids on all pairs collapse), a hard node budget and an optional wall-time budget
                   per task (the clock is a RESOURCE GUARD only, never in the logic). Two CLOSERS
                   finish a program: exact match, and a colour-map fitted from the training pairs.
  4. LIBRARY     : a solved task's winning program (>=2 steps) becomes ONE named composite primitive
                   -- OpenGf._learn's pattern in the grid domain -- so later tasks reach deeper
                   within the same size budget. Measured ON vs OFF; a null result is reported as-is.

(A learned per-primitive ordering, ArcWeights, was built and measured here: its bench effect was
byte-identical to the fixed prior -- arc_diagnostic proved the failures are DSL-COVERAGE limited,
so no search order can add solves. Per the adopt-better/retire-inferior policy it was REMOVED;
the hand-authored _BASE order is the only prior.)

HONEST FRAMING (read before quoting any number)
-----------------------------------------------
This solves the small class of ARC tasks whose rule IS a short composition of these primitives
(global geometry, colour permutation, crop-to-object, mirror/tile symmetry, half-grid logic, simple
object selection). It does NOT perceive relations BETWEEN objects, counting, drawing, or novel
concepts outside the DSL -- most ARC tasks -- and every such task returns an honest empty candidate
list. Published pure-DSL-search baselines land in the low single digits to low tens of percent;
anything above the 0.5% trivial floor (arc_bench's constant control) is a real move off 0.0%, and
nothing more is claimed. In particular this is NOT general fluid intelligence: the primitives are
author-chosen, so the search can only ever recombine THIS vocabulary -- the wall of gf.py, unmoved.

Methodology: developed on the TRAINING split only; the evaluation split is never inspected or tuned
on. The self-check scores training tasks and runs a genuinely-failing control (the same search with
an identity-only DSL scores ~0).

Deterministic (fixed iteration order; no RNG needed; wall-clock only as a budget).
ASCII-only output. Representation-only: imports no brainstem, routes no actuator.

  PYTHONPATH=src python -m snn_data_lake.reasoning.arc_solver          # self-check
"""

import sys
import time
from collections import Counter

_MAXS = 30  # ARC grids are at most 30x30; larger outputs are invalid
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


# ====================================================================================================
# grids
# ====================================================================================================


def _tup(g):
    """Grid -> immutable tuple-of-tuples of ints (hashable state for the search)."""
    return tuple(tuple(int(v) for v in row) for row in g)


def _ok(g):
    """Well-formed ARC grid: rectangular, 1..30 per side, colours 0..9."""
    if not isinstance(g, tuple) or not (1 <= len(g) <= _MAXS):
        return False
    w = len(g[0])
    if not (1 <= w <= _MAXS):
        return False
    for row in g:
        if len(row) != w or any(not (0 <= v <= 9) for v in row):
            return False
    return True


def background(g):
    """Most common colour (ties prefer 0, then the smallest colour) -- the ARC background heuristic."""
    cnt = Counter(v for row in g for v in row)
    top = max(cnt.values())
    best = sorted(c for c, n in cnt.items() if n == top)
    return 0 if 0 in best else best[0]


# ====================================================================================================
# 1. PERCEPTION -- objects = connected components (EXTENDS arc_agi/rule_inducer's parser: same BFS +
#    attribute style; adds 8-connectivity and multi-colour variants; background skipped, not EMPTY).
# ====================================================================================================

_SEG_CACHE = {}  # (grid, conn, same_color) -> [obj], cleared per synthesize


def components(g, conn=4, same_color=True, bg=None):
    """Objects of a grid: {'cells','color','size','bbox','shape'}. Deterministic order (by bbox)."""
    if bg is None:
        bg = background(g)
    key = (g, conn, same_color, bg)
    if key in _SEG_CACHE:
        return _SEG_CACHE[key]
    R, C = len(g), len(g[0])
    nei = _N4 if conn == 4 else _N8
    seen = [[False] * C for _ in range(R)]
    objs = []
    for r in range(R):
        for c in range(C):
            if seen[r][c] or g[r][c] == bg:
                continue
            colr = g[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nei:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < R
                        and 0 <= nx < C
                        and not seen[ny][nx]
                        and g[ny][nx] != bg
                        and (not same_color or g[ny][nx] == colr)
                    ):
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            rs = [y for y, _ in cells]
            cs = [x for _, x in cells]
            cnt = Counter(g[y][x] for y, x in cells)
            topn = max(cnt.values())
            dom = min(c2 for c2, n in cnt.items() if n == topn)
            r0, c0 = min(rs), min(cs)
            objs.append(
                {
                    "cells": frozenset(cells),
                    "color": dom,
                    "size": len(cells),
                    "bbox": (r0, c0, max(rs), max(cs)),
                    "shape": frozenset((y - r0, x - c0) for y, x in cells),
                }
            )
    objs.sort(key=lambda o: o["bbox"])
    _SEG_CACHE[key] = objs
    return objs


def _obj_crop(g, obj, bg):
    r0, c0, r1, c1 = obj["bbox"]
    out = [[bg] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for y, x in obj["cells"]:
        out[y - r0][x - c0] = g[y][x]
    return _tup(out)


def _pick(objs, mode):
    """Select ONE object by a general predicate; ambiguity -> None (strict, keeps precision)."""
    if not objs:
        return None
    if mode in ("largest", "smallest"):
        sizes = [o["size"] for o in objs]
        m = max(sizes) if mode == "largest" else min(sizes)
        hits = [o for o in objs if o["size"] == m]
        return hits[0] if len(hits) == 1 else None
    if mode == "unique_color":
        cnt = Counter(o["color"] for o in objs)
        uc = [c for c, n in cnt.items() if n == 1]
        if len(uc) != 1:
            return None
        return next(o for o in objs if o["color"] == uc[0])
    if mode == "odd_shape":
        cnt = Counter(o["shape"] for o in objs)
        rare = [s for s, n in cnt.items() if n == 1]
        if len(rare) != 1 or len(cnt) < 2:
            return None
        return next(o for o in objs if o["shape"] == rare[0])
    return None


# ====================================================================================================
# 2. THE DSL -- small, composable, TOTAL primitives: grid -> grid, or None when inapplicable.
# ====================================================================================================


def _rot90(g):
    return tuple(zip(*g[::-1]))


def _rot180(g):
    return tuple(row[::-1] for row in g[::-1])


def _rot270(g):
    return tuple(zip(*g))[::-1]


def _flip_h(g):
    return tuple(row[::-1] for row in g)


def _flip_v(g):
    return g[::-1]


def _transpose(g):
    return tuple(zip(*g))


def _crop_content(g):
    bg = background(g)
    pts = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    if not pts:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return tuple(row[min(cs) : max(cs) + 1] for row in g[min(rs) : max(rs) + 1])


def _scale(g, k):
    R, C = len(g), len(g[0])
    if R * k > _MAXS or C * k > _MAXS:
        return None
    out = []
    for row in g:
        wide = tuple(v for v in row for _ in range(k))
        out.extend([wide] * k)
    return tuple(out)


def _unscale(g):
    """Inverse of _scale for the largest k>1 with uniform kxk blocks; None if the grid is not blocked."""
    R, C = len(g), len(g[0])
    for k in range(min(R, C), 1, -1):
        if R % k or C % k:
            continue
        ok = True
        for br in range(R // k):
            for bc in range(C // k):
                v = g[br * k][bc * k]
                if any(g[br * k + i][bc * k + j] != v for i in range(k) for j in range(k)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return tuple(tuple(g[br * k][bc * k] for bc in range(C // k)) for br in range(R // k))
    return None


def _tile(g, kr, kc):
    R, C = len(g), len(g[0])
    if R * kr > _MAXS or C * kc > _MAXS:
        return None
    return tuple(tuple(row * kc) for row in g) * kr


def _mirror_h(g):
    return tuple(row + row[::-1] for row in g) if 2 * len(g[0]) <= _MAXS else None


def _mirror_h_l(g):
    return tuple(row[::-1] + row for row in g) if 2 * len(g[0]) <= _MAXS else None


def _mirror_v(g):
    return g + g[::-1] if 2 * len(g) <= _MAXS else None


def _mirror_v_t(g):
    return g[::-1] + g if 2 * len(g) <= _MAXS else None


def _mirror_quad(g):
    if 2 * len(g) > _MAXS or 2 * len(g[0]) > _MAXS:
        return None
    top = tuple(row + row[::-1] for row in g)
    return top + top[::-1]


def _half(g, which):
    R, C = len(g), len(g[0])
    if which in ("top", "bottom"):
        if R < 2:
            return None
        return g[: R // 2] if which == "top" else g[R - R // 2 :]
    if C < 2:
        return None
    return (
        tuple(row[: C // 2] for row in g)
        if which == "left"
        else tuple(row[C - C // 2 :] for row in g)
    )


def _sel(g, conn, same, mode):
    bg = background(g)
    obj = _pick(components(g, conn, same, bg), mode)
    return _obj_crop(g, obj, bg) if obj is not None else None


def _del(g, conn, same, mode):
    bg = background(g)
    obj = _pick(components(g, conn, same, bg), mode)
    if obj is None:
        return None
    return tuple(
        tuple(bg if (r, c) in obj["cells"] else v for c, v in enumerate(row))
        for r, row in enumerate(g)
    )


_GRAV = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}


def _gravity(g, d):
    """Move every (8-connected, multi-colour) object along d until blocked by a wall or settled object."""
    dr, dc = _GRAV[d]
    bg = background(g)
    objs = components(g, 8, False, bg)
    if not objs:
        return g
    R, C = len(g), len(g[0])

    def far(o):  # settle the objects nearest the wall first
        ys = [r * dr + c * dc for r, c in o["cells"]]
        return -max(ys)

    out = [[bg] * C for _ in range(R)]
    occupied = set()
    for o in sorted(objs, key=lambda o: (far(o), o["bbox"])):
        s = 0
        while True:
            moved = [(r + dr * (s + 1), c + dc * (s + 1)) for r, c in o["cells"]]
            if all(0 <= y < R and 0 <= x < C and (y, x) not in occupied for y, x in moved):
                s += 1
            else:
                break
        for r, c in o["cells"]:
            y, x = r + dr * s, c + dc * s
            out[y][x] = g[r][c]
            occupied.add((y, x))
    return _tup(out)


def _fill_enclosed(g, colr):
    """Fill background regions NOT reachable from the border (holes) with `colr`; no hole -> None."""
    bg = background(g)
    R, C = len(g), len(g[0])
    seen = set()
    stack = [(r, c) for r in range(R) for c in (0, C - 1) if g[r][c] == bg]
    stack += [(r, c) for c in range(C) for r in (0, R - 1) if g[r][c] == bg]
    seen.update(stack)
    while stack:
        y, x = stack.pop()
        for dy, dx in _N4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < R and 0 <= nx < C and (ny, nx) not in seen and g[ny][nx] == bg:
                seen.add((ny, nx))
                stack.append((ny, nx))
    holes = [(r, c) for r in range(R) for c in range(C) if g[r][c] == bg and (r, c) not in seen]
    if not holes:
        return None
    out = [list(row) for row in g]
    for r, c in holes:
        out[r][c] = colr
    return _tup(out)


def _frame(g, colr):
    R, C = len(g), len(g[0])
    if R + 2 > _MAXS or C + 2 > _MAXS:
        return None
    top = tuple([colr] * (C + 2))
    return (top,) + tuple((colr,) + row + (colr,) for row in g) + (top,)


# ---- OBJECT-CENTRIC primitives: transforms/filters/counts over the extracted components. --------
#      Added after arc_diagnostic proved (11/11 sampled failures) the ceiling is DSL COVERAGE,
#      not search: the only lever that can add solves is a richer vocabulary. All reuse
#      components(); all are parameter-free, total, deterministic.


def _obj_reflect_each(g, mode):
    """Reflect every (8-connected, multi-colour) object in place within its own bbox.
    A reflected cell landing on another object's cell -> None (strict, keeps precision)."""
    bg = background(g)
    objs = components(g, 8, False, bg)
    if not objs:
        return None
    R, C = len(g), len(g[0])
    out = [[bg] * C for _ in range(R)]
    occ = set()
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        for y, x in sorted(o["cells"]):
            ny = (r0 + r1) - y if mode in ("v", "r") else y
            nx = (c0 + c1) - x if mode in ("h", "r") else x
            if (ny, nx) in occ:
                return None
            occ.add((ny, nx))
            out[ny][nx] = g[y][x]
    return _tup(out)


def _recolor_modal(g):
    """Recolor every non-background cell to the modal non-background colour (ties -> smallest)."""
    bg = background(g)
    cnt = Counter(v for row in g for v in row if v != bg)
    if not cnt:
        return None
    top = max(cnt.values())
    col = min(c for c, n in cnt.items() if n == top)
    return tuple(tuple(col if v != bg else v for v in row) for row in g)


def _filter_objs(g, attr, keep_modal):
    """Keep only the objects whose `attr` ('shape' or 'color') equals (or differs from) THE
    unique modal value over all objects; ambiguity or nothing-to-filter -> None."""
    bg = background(g)
    objs = components(g, 4, True, bg)
    if len(objs) < 2:
        return None
    cnt = Counter(o[attr] for o in objs)
    top = max(cnt.values())
    modal = [k for k, n in cnt.items() if n == top]
    if len(modal) != 1 or len(cnt) < 2:
        return None
    keep = [o for o in objs if (o[attr] == modal[0]) == keep_modal]
    out = [[bg] * len(g[0]) for _ in range(len(g))]
    for o in keep:
        for y, x in o["cells"]:
            out[y][x] = g[y][x]
    return _tup(out)


def _isolate(g, conn, same, mode):
    """Keep ONE object (chosen by _pick) in place; every other cell becomes background --
    the in-place complement of _sel (which crops to the object's bbox)."""
    bg = background(g)
    obj = _pick(components(g, conn, same, bg), mode)
    if obj is None:
        return None
    out = [[bg] * len(g[0]) for _ in range(len(g))]
    for y, x in obj["cells"]:
        out[y][x] = g[y][x]
    return _tup(out)


def _count_grid(g, conn, same, square):
    """COUNT the objects: k of them -> a k x k (or 1 x k) grid of the modal object colour."""
    bg = background(g)
    objs = components(g, conn, same, bg)
    k = len(objs)
    if k == 0 or k > _MAXS:
        return None
    cnt = Counter(o["color"] for o in objs)
    top = max(cnt.values())
    col = min(c for c, n in cnt.items() if n == top)
    return tuple((col,) * k for _ in range(k if square else 1))


def _objs_bbox_fill(g):
    """Fill every object's bounding box with the object's colour (deterministic overwrite order)."""
    bg = background(g)
    objs = components(g, 4, True, bg)
    if not objs:
        return None
    out = [list(row) for row in g]
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        for y in range(r0, r1 + 1):
            for x in range(c0, c1 + 1):
                out[y][x] = o["color"]
    return _tup(out)


def _sel_pos(g, conn, same, idx):
    """Crop of the FIRST or LAST object in reading order (components are bbox-sorted)."""
    bg = background(g)
    objs = components(g, conn, same, bg)
    if len(objs) < 2:
        return None
    return _obj_crop(g, objs[idx], bg)


def _color_cell(g, modal):
    """1x1 grid of the modal (or the rarest) non-background colour; ties -> smallest."""
    bg = background(g)
    cnt = Counter(v for row in g for v in row if v != bg)
    if not cnt:
        return None
    pickn = max(cnt.values()) if modal else min(cnt.values())
    return ((min(c for c, n in cnt.items() if n == pickn),),)


def _obj_center(g):
    """Move the single (8-connected, multi-colour) object so its bbox is centred in the grid."""
    bg = background(g)
    objs = components(g, 8, False, bg)
    if len(objs) != 1:
        return None
    o = objs[0]
    r0, c0, r1, c1 = o["bbox"]
    R, C = len(g), len(g[0])
    dr = (R - (r1 - r0 + 1)) // 2 - r0
    dc = (C - (c1 - c0 + 1)) // 2 - c0
    out = [[bg] * C for _ in range(R)]
    for y, x in o["cells"]:
        out[y + dr][x + dc] = g[y][x]
    return _tup(out)


# ---- the base (parameter-free) primitive registry: (name, fn). The hand order is the search ----
#      prior ('cheap/likely first') and the tie-break; this list itself is NEVER mutated
#      (arc_diagnostic reflects into it, determinism depends on it).

_BASE = [
    ("rot90", _rot90),
    ("rot180", _rot180),
    ("rot270", _rot270),
    ("flip_h", _flip_h),
    ("flip_v", _flip_v),
    ("transpose", _transpose),
    ("crop_content", _crop_content),
    ("mirror_h", _mirror_h),
    ("mirror_h_l", _mirror_h_l),
    ("mirror_v", _mirror_v),
    ("mirror_v_t", _mirror_v_t),
    ("mirror_quad", _mirror_quad),
    ("scale2", lambda g: _scale(g, 2)),
    ("scale3", lambda g: _scale(g, 3)),
    ("unscale", _unscale),
    ("tile_h2", lambda g: _tile(g, 1, 2)),
    ("tile_v2", lambda g: _tile(g, 2, 1)),
    ("tile_22", lambda g: _tile(g, 2, 2)),
    ("tile_h3", lambda g: _tile(g, 1, 3)),
    ("tile_v3", lambda g: _tile(g, 3, 1)),
    ("tile_33", lambda g: _tile(g, 3, 3)),
    ("half_top", lambda g: _half(g, "top")),
    ("half_bottom", lambda g: _half(g, "bottom")),
    ("half_left", lambda g: _half(g, "left")),
    ("half_right", lambda g: _half(g, "right")),
    ("sel_largest", lambda g: _sel(g, 4, True, "largest")),
    ("sel_smallest", lambda g: _sel(g, 4, True, "smallest")),
    ("sel_unique_color", lambda g: _sel(g, 4, True, "unique_color")),
    ("sel_odd_shape", lambda g: _sel(g, 4, True, "odd_shape")),
    ("sel_largest_m8", lambda g: _sel(g, 8, False, "largest")),
    ("sel_smallest_m8", lambda g: _sel(g, 8, False, "smallest")),
    ("sel_odd_shape_m8", lambda g: _sel(g, 8, False, "odd_shape")),
    ("del_largest", lambda g: _del(g, 4, True, "largest")),
    ("del_smallest", lambda g: _del(g, 4, True, "smallest")),
    ("del_unique_color", lambda g: _del(g, 4, True, "unique_color")),
    ("gravity_down", lambda g: _gravity(g, "down")),
    ("gravity_up", lambda g: _gravity(g, "up")),
    ("gravity_left", lambda g: _gravity(g, "left")),
    ("gravity_right", lambda g: _gravity(g, "right")),
    # object-centric additions (see the OBJECT-CENTRIC section above; appended, never reordered)
    ("obj_flip_h_each", lambda g: _obj_reflect_each(g, "h")),
    ("obj_flip_v_each", lambda g: _obj_reflect_each(g, "v")),
    ("obj_rot180_each", lambda g: _obj_reflect_each(g, "r")),
    ("recolor_modal", _recolor_modal),
    ("keep_modal_shape", lambda g: _filter_objs(g, "shape", True)),
    ("keep_odd_shape", lambda g: _filter_objs(g, "shape", False)),
    ("keep_modal_color", lambda g: _filter_objs(g, "color", True)),
    ("keep_odd_color", lambda g: _filter_objs(g, "color", False)),
    ("isolate_largest", lambda g: _isolate(g, 4, True, "largest")),
    ("isolate_largest_m8", lambda g: _isolate(g, 8, False, "largest")),
    ("count_sq", lambda g: _count_grid(g, 4, True, True)),
    ("count_row", lambda g: _count_grid(g, 4, True, False)),
    ("count_sq_m8", lambda g: _count_grid(g, 8, False, True)),
    ("objs_bbox_fill", _objs_bbox_fill),
    ("sel_first", lambda g: _sel_pos(g, 4, True, 0)),
    ("sel_last", lambda g: _sel_pos(g, 4, True, -1)),
    ("sel_first_m8", lambda g: _sel_pos(g, 8, False, 0)),
    ("sel_last_m8", lambda g: _sel_pos(g, 8, False, -1)),
    ("modal_cell", lambda g: _color_cell(g, True)),
    ("rare_cell", lambda g: _color_cell(g, False)),
    ("obj_center", _obj_center),
]


# ---- fitted (per-task) primitives: parameters inferred from the TRAINING pairs, never per-task ids --


def _out_colors(train_outs):
    """Non-background colours across the training OUTPUTS, most frequent first (deterministic)."""
    cnt = Counter()
    for y in train_outs:
        bg = background(y)
        for row in y:
            for v in row:
                if v != bg:
                    cnt[v] += 1
    return [c for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]


def _split_layouts():
    def vsep(g):
        C = len(g[0])
        if C < 3 or C % 2 == 0:
            return None
        m = C // 2
        if len({row[m] for row in g}) != 1:
            return None
        return tuple(row[:m] for row in g), tuple(row[m + 1 :] for row in g)

    def hsep(g):
        R = len(g)
        if R < 3 or R % 2 == 0:
            return None
        m = R // 2
        if len(set(g[m])) != 1:
            return None
        return g[:m], g[m + 1 :]

    def veven(g):
        C = len(g[0])
        return (
            (tuple(row[: C // 2] for row in g), tuple(row[C // 2 :] for row in g))
            if (C >= 2 and C % 2 == 0)
            else None
        )

    def heven(g):
        R = len(g)
        return (g[: R // 2], g[R // 2 :]) if (R >= 2 and R % 2 == 0) else None

    return [("vsep", vsep), ("hsep", hsep), ("veven", veven), ("heven", heven)]


def _mk_split_op(splitf, op, out_color):
    def fn(g):
        parts = splitf(g)
        if parts is None:
            return None
        A, B = parts
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            return None
        cnt = Counter(v for half in (A, B) for row in half for v in row)
        top = max(cnt.values())
        best = sorted(c for c, n in cnt.items() if n == top)
        bg = 0 if 0 in best else best[0]
        rows = []
        for ra, rb in zip(A, B):
            row = []
            for va, vb in zip(ra, rb):
                a, b = va != bg, vb != bg
                on = (
                    (a and b)
                    if op == "and"
                    else (a or b)
                    if op == "or"
                    else (a != b)
                    if op == "xor"
                    else (a and not b)
                )
                row.append(out_color if on else bg)
            rows.append(tuple(row))
        return tuple(rows)

    return fn


def _fitted_prims(train_in, train_out):
    """Primitives whose PARAMETERS (colours, split layout) are fitted from the training pairs.
    The fitting procedures are fully general -- no task ids, no per-task branches."""
    prims = []
    colors = _out_colors(train_out)
    for c in colors[:4]:
        prims.append(("fill_enc_%d" % c, lambda g, c=c: _fill_enclosed(g, c)))
        prims.append(("frame_%d" % c, lambda g, c=c: _frame(g, c)))
    for lname, lf in _split_layouts():
        if all(lf(x) is not None for x in train_in):
            if colors:
                for op in ("and", "or", "xor", "diff"):
                    prims.append(
                        ("split_%s_%s_%d" % (lname, op, colors[0]), _mk_split_op(lf, op, colors[0]))
                    )
            break  # first layout that fits every input
    return prims


# ====================================================================================================
# closers -- finish a program at any search state (exact match, or a colour-map fitted from train)
# ====================================================================================================


def _fit_colormap(pred, want):
    """One colour->colour mapping consistent CELLWISE across every (predicted, wanted) pair, or None.
    Identity mappings are None (the exact closer already covers them)."""
    m = {}
    for p, y in zip(pred, want):
        if len(p) != len(y) or len(p[0]) != len(y[0]):
            return None
        for rp, ry in zip(p, y):
            for a, b in zip(rp, ry):
                if m.get(a, b) != b:
                    return None
                m[a] = b
    if all(k == v for k, v in m.items()):
        return None
    return m


def _apply_map(g, m):
    return tuple(tuple(m.get(v, v) for v in row) for row in g)


# ====================================================================================================
# 3. SEARCH -- iterative deepening by program SIZE (the gf.induce_program idiom on grids), with
#    equivalence pruning, a node budget, and an optional wall-time budget (resource guards only).
# ====================================================================================================


class ArcLibrary:
    """Learned composite primitives -- OpenGf._learn's pattern in the grid domain: the winning
    program of a solved task (>=2 steps, flattened, deduped) becomes ONE named step for later tasks."""

    def __init__(self, cap=64):
        self.cap = cap
        self.progs = []  # [tuple(primitive names)]
        self._defs = {}  # composite name -> flat name tuple
        self._seen = set()

    def learn(self, names):
        flat = []
        for n in names:
            flat.extend(self._defs.get(n, (n,)))
        flat = tuple(flat)
        if len(flat) < 2 or flat in self._seen or len(self.progs) >= self.cap:
            return False
        self._seen.add(flat)
        self.progs.append(flat)
        self._defs["lib:" + ">".join(flat)] = flat
        return True


def _pool(train_in, train_out, library):
    """The per-task primitive pool: base + fitted + library composites (refit in THIS task's context)."""
    prims = list(_BASE) + _fitted_prims(train_in, train_out)
    pmap = dict(prims)
    if library is not None:
        for names in library.progs:
            if not all(n in pmap for n in names):
                continue  # a fitted step this task cannot refit
            fns = [pmap[n] for n in names]

            def comp(g, fns=fns):
                for f in fns:
                    g = f(g)
                    if g is None:
                        return None
                return g

            prims.append(("lib:" + ">".join(names), comp))
    return prims


def synthesize(
    train_pairs,
    test_input,
    attempts=2,
    max_depth=3,
    node_budget=4000,
    time_budget=None,
    library=None,
    dsl="full",
):
    """Iterative-deepening program synthesis for ONE ARC task.

    Returns (candidates, programs): up to `attempts` DISTINCT predicted test-output grids (lists of
    lists), smallest-program-first, and the matching (names, closer) descriptions. dsl="identity" is
    the CONTROL: no primitives, exact closer only -- it must score ~0 on real tasks."""
    _SEG_CACHE.clear()
    try:
        train_in = tuple(_tup(i) for i, _o in train_pairs)
        train_out = tuple(_tup(o) for _i, o in train_pairs)
        test = _tup(test_input)
    except (TypeError, ValueError, IndexError):
        return [], []
    if not train_in or not all(_ok(g) for g in train_in + train_out + (test,)):
        return [], []

    full = dsl == "full"
    prims = _pool(train_in, train_out, library) if full else []
    cands, progs, seen_c = [], [], set()

    def close(state, names):
        pred, t = state[:-1], state[-1]
        if pred == train_out:
            _add(t, (names, "exact"))
        if full and len(cands) < attempts:
            m = _fit_colormap(pred, train_out)
            if m is not None:
                _add(_apply_map(t, m), (names, "map"))

    def _add(grid, prog):
        if _ok(grid) and grid not in seen_c and len(cands) < attempts:
            seen_c.add(grid)
            cands.append([list(row) for row in grid])
            progs.append(prog)

    state0 = train_in + (test,)
    close(state0, ())
    seen = {state0}
    frontier = [(state0, ())]
    nodes = 0
    t0 = time.monotonic() if time_budget is not None else None
    for _depth in range(1, max_depth + 1):
        if len(cands) >= attempts or not frontier:
            break
        nxt = []
        for state, names in frontier:
            for pname, fn in prims:
                nodes += 1
                if nodes > node_budget or (
                    t0 is not None and nodes % 64 == 0 and time.monotonic() - t0 > time_budget
                ):
                    frontier = []  # budget exhausted -> truncate honestly
                    nxt = []
                    break
                gs = []
                for g in state:
                    try:
                        g = fn(g)
                    except Exception:
                        g = None  # a primitive must be total; treat as None
                    if g is None or not _ok(g):
                        gs = None
                        break
                    gs.append(g)
                if gs is None:
                    continue
                ns = tuple(gs)
                if ns in seen:
                    continue  # equivalence pruning (same effect, longer)
                seen.add(ns)
                close(ns, names + (pname,))
                if len(cands) >= attempts:
                    break
                nxt.append((ns, names + (pname,)))
            if len(cands) >= attempts or not frontier:
                break
        frontier = nxt
    # weakest fallback: the constant program (all training outputs identical) -- arc_bench's 0.5% floor
    if full and len(cands) < attempts and len(set(train_out)) == 1:
        _add(train_out[0], ((), "const"))
    return cands, progs


def make_solver(library=None, max_depth=3, node_budget=4000, time_budget=5.0, dsl="full"):
    """An arc_bench-shaped solver(train_pairs, test_input, attempts) -> [candidate grids]. With a
    library, each solved task's winning program is learned as a composite for the NEXT tasks."""

    def solver(train, test_input, attempts=2):
        cands, progs = synthesize(
            train,
            test_input,
            attempts=attempts,
            max_depth=max_depth,
            node_budget=node_budget,
            time_budget=time_budget,
            library=library,
            dsl=dsl,
        )
        if library is not None and progs:
            names, _closer = progs[0]
            if len(names) >= 2:
                library.learn(names)
        return cands

    return solver


# ====================================================================================================
# self-check -- primitives, synthetic tasks, a genuinely-failing control, library effect, determinism
# ====================================================================================================


def _solve1(pairs, test, **kw):
    cands, progs = synthesize(pairs, test, **kw)
    return (cands[0] if cands else None), (progs[0] if progs else None)


def main():
    print("=" * 96)
    print("ARC SOLVER -- object perception + grid DSL + iterative-deepening program synthesis")
    print("=" * 96)
    ok = True

    # ---- (a) primitive unit checks -----------------------------------------------------------------
    g = _tup([[1, 2], [3, 4]])
    checks = [
        ("rot90", _rot90(g) == _tup([[3, 1], [4, 2]])),
        ("rot180", _rot180(g) == _tup([[4, 3], [2, 1]])),
        ("flip_h", _flip_h(g) == _tup([[2, 1], [4, 3]])),
        ("transpose", _transpose(g) == _tup([[1, 3], [2, 4]])),
        ("crop_content", _crop_content(_tup([[0, 0, 0], [0, 5, 0], [0, 0, 0]])) == _tup([[5]])),
        ("scale2", _scale(_tup([[1, 2]]), 2) == _tup([[1, 1, 2, 2], [1, 1, 2, 2]])),
        ("unscale", _unscale(_scale(g, 3)) == g),
        ("mirror_h", _mirror_h(_tup([[1, 2]])) == _tup([[1, 2, 2, 1]])),
        ("tile_22", _tile(_tup([[7]]), 2, 2) == _tup([[7, 7], [7, 7]])),
        ("gravity_down", _gravity(_tup([[3, 0], [0, 0]]), "down") == _tup([[0, 0], [3, 0]])),
        (
            "fill_enclosed",
            _fill_enclosed(_tup([[0, 0, 0, 0], [0, 3, 3, 3], [0, 3, 0, 3], [0, 3, 3, 3]]), 6)
            == _tup([[0, 0, 0, 0], [0, 3, 3, 3], [0, 3, 6, 3], [0, 3, 3, 3]]),
        ),
        ("frame", _frame(_tup([[5]]), 2) == _tup([[2, 2, 2], [2, 5, 2], [2, 2, 2]])),
        (
            "sel_largest",
            _sel(_tup([[1, 0, 0], [1, 0, 2], [1, 0, 0]]), 4, True, "largest")
            == _tup([[1], [1], [1]]),
        ),
        (
            "obj_flip_h_each",
            _obj_reflect_each(_tup([[3, 0, 0, 5], [3, 3, 0, 0]]), "h")
            == _tup([[0, 3, 0, 5], [3, 3, 0, 0]]),
        ),
        (
            "obj_flip_h_collision",
            _obj_reflect_each(_tup([[1, 0, 5, 0], [1, 0, 0, 0], [1, 1, 1, 0]]), "h") is None,
        ),
        (
            "keep_odd_color",
            _filter_objs(_tup([[1, 0, 2], [0, 0, 0], [1, 0, 0]]), "color", False)
            == _tup([[0, 0, 2], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            "isolate_largest",
            _isolate(_tup([[1, 1, 0], [0, 0, 0], [0, 0, 5]]), 4, True, "largest")
            == _tup([[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            "count_sq",
            _count_grid(_tup([[1, 0, 1], [0, 0, 0], [2, 0, 0]]), 4, True, True)
            == _tup([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ),
        (
            "sel_first",
            _sel_pos(_tup([[7, 0, 0], [0, 0, 0], [0, 0, 4]]), 4, True, 0) == _tup([[7]]),
        ),
        ("modal_cell", _color_cell(_tup([[0, 1, 2], [0, 2, 0]]), True) == _tup([[2]])),
        (
            "obj_center",
            _obj_center(_tup([[6, 0, 0], [0, 0, 0], [0, 0, 0]]))
            == _tup([[0, 0, 0], [0, 6, 0], [0, 0, 0]]),
        ),
    ]
    for name, passed in checks:
        if not passed:
            print("  FAIL primitive: %s" % name)
            ok = False
    print("primitives: %d/%d unit checks pass" % (sum(p for _n, p in checks), len(checks)))

    # ---- (b) synthetic tasks (hand-built here; NO benchmark data touched) --------------------------
    t_rot = (
        [([[1, 2], [3, 0]], [[0, 3], [2, 1]]), ([[5, 0], [0, 6]], [[6, 0], [0, 5]])],
        [[7, 0], [0, 8]],
        [[8, 0], [0, 7]],
    )  # rule: rot180
    t_map = (
        [([[1, 1], [2, 2]], [[2, 2], [1, 1]]), ([[1, 2], [1, 2]], [[2, 1], [2, 1]])],
        [[2, 2], [2, 1]],
        [[1, 1], [1, 2]],
    )  # rule: swap colours 1<->2
    t_two = (
        [([[0, 0], [0, 3]], [[3, 3], [3, 3]]), ([[5, 0], [0, 0]], [[5, 5], [5, 5]])],
        [[0, 0], [9, 0]],
        [[9, 9], [9, 9]],
    )  # rule: crop_content then scale2
    for tname, (pairs, ti, want) in (("rot180", t_rot), ("colormap", t_map), ("crop+scale", t_two)):
        got, prog = _solve1(pairs, ti)
        if got != want:
            print("  FAIL synthetic '%s': got %r (program %r)" % (tname, got, prog))
            ok = False
        else:
            print("synthetic '%s': solved by %r" % (tname, prog))

    # ---- (c) CONTROL that must genuinely fail: identity-only DSL on the same synthetic tasks -------
    ctrl_hits = sum(
        1
        for pairs, ti, want in (t_rot, t_map, t_two)
        if _solve1(pairs, ti, dsl="identity")[0] == want
    )
    print(
        "control (identity-only DSL) on the 3 synthetic tasks: %d/3 solved (must be 0)" % ctrl_hits
    )
    if ctrl_hits:
        ok = False

    # ---- (d) library learning: a 2-step solve becomes ONE step for the next task -------------------
    lib = ArcLibrary()
    solver = make_solver(library=lib, time_budget=None)
    solver(t_two[0], t_two[1])  # learns ('crop_content','scale2')
    got_nolib, _p = _solve1(t_two[0], t_two[1], max_depth=1)
    got_lib, prog_lib = _solve1(t_two[0], t_two[1], max_depth=1, library=lib)
    print(
        "library: progs=%r; depth-1 without lib -> %s, with lib -> %s"
        % (
            lib.progs,
            "solved" if got_nolib == t_two[2] else "unsolved",
            "solved" if got_lib == t_two[2] else "unsolved",
        )
    )
    lib_name = "lib:" + ">".join(lib.progs[0]) if lib.progs else None
    if not (
        len(lib.progs) == 1
        and len(lib.progs[0]) == 2
        and got_nolib != t_two[2]
        and got_lib == t_two[2]
        and prog_lib[0] == (lib_name,)
    ):
        print("  FAIL library: the learned composite must make the depth-1 solve possible")
        ok = False

    # ---- (d2) pool order: the hand-authored _BASE prior must survive pooling exactly ---------------
    tin = tuple(_tup(i) for i, _o in t_two[0])
    tout = tuple(_tup(o) for _i, o in t_two[0])
    base_names = [n for n, _f in _pool(tin, tout, None)]
    if base_names[: len(_BASE)] != [n for n, _f in _BASE]:
        print("  FAIL pool: _pool must keep the hand-authored _BASE order exactly")
        ok = False
    print("pool order: _BASE prior preserved (%d base primitives)" % len(_BASE))

    # ---- (e) TRAINING-split spot score (develop on training; evaluation is NEVER read here) --------
    from snn_data_lake.reasoning import arc_bench

    tasks = arc_bench.load_tasks("training")
    if not tasks:
        print("SKIP data section: no ARC data at %s" % arc_bench.DATA)
    else:
        n = 40
        r_full = arc_bench.score(
            make_solver(node_budget=1500, time_budget=3.0), "training", limit=n
        )
        r_ctrl = arc_bench.score(
            make_solver(node_budget=1500, time_budget=3.0, dsl="identity"), "training", limit=n
        )
        print(
            "training[:%d]  full DSL   : %d/%d = %.1f%%   solved=%s"
            % (n, r_full["solved"], r_full["total"], 100 * r_full["rate"], r_full["solved_ids"])
        )
        print(
            "training[:%d]  identity   : %d/%d = %.1f%%   (the failing control)"
            % (n, r_ctrl["solved"], r_ctrl["total"], 100 * r_ctrl["rate"])
        )
        if not (
            r_full["solved"] > r_ctrl["solved"] and r_full["solved"] >= 2 and r_ctrl["solved"] <= 1
        ):
            print("  FAIL: full DSL must beat the identity control and clear the trivial floor")
            ok = False
        # determinism: an identical re-run (node budget binding, no wall-clock in logic)
        r_again = arc_bench.score(
            make_solver(node_budget=1500, time_budget=None), "training", limit=n
        )
        r_b = arc_bench.score(make_solver(node_budget=1500, time_budget=None), "training", limit=n)
        if r_again["solved_ids"] != r_b["solved_ids"]:
            print("  FAIL: two budget-bound runs must solve the identical task set")
            ok = False
        print(
            "determinism: two node-budget-bound runs solve the identical set (%d tasks)"
            % r_again["solved"]
        )

    print("-" * 96)
    print(
        "HONEST: this synthesizes short programs over an author-chosen grid vocabulary. It solves"
    )
    print("the geometry/colour/object-selection slice of ARC and returns an honest empty candidate")
    print(
        "list everywhere else; it is NOT general fluid intelligence -- the wall of gf.py, unmoved."
    )
    print("arc_solver self-check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
