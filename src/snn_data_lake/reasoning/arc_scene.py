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
new typed primitives. Batch 1 (this file): G1 CONN (abstraction lattices), G4 PALETTE, G5 MORPH, G6 TOPO,
G7 SHAPE-EQ, G8 ALIGN/ADJ. Batch 2 (next session): G2 LATTICE, G3 PERIOD, G9 DIST, G10 PROG, G11 SYM,
G12 LOCI, G13 COLOR-PAIRING.

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
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


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

    _emit("-" * width)
    _emit(
        "self-check OK: additive perceptual registry (palette + morph + topo/holes + D8 shape-class +"
    )
    _emit(
        "relations + competing CONN abstractions), reusing components() as the seed, mutating nothing."
    )
    _emit(
        "HONEST: measurement ONLY (no synthesis, no ARC-eval change); the Phase-3 mint consumes these."
    )
    _emit(
        "Batch 2 (LATTICE/PERIOD/DIST/PROG/SYM/LOCI/COLOR-PAIRING) + the operation-synthesis arm remain."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
