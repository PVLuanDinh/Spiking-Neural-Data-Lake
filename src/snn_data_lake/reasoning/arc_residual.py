r"""ARC RESIDUAL -- the versioned DESCRIPTOR ALPHABET and the induced-conjunction SELECTOR.

specs/006-fixed-point-gf, DESIGN B (docs/INVENTION.md section 3). The measured Gf bottleneck is
VOCABULARY, not depth: specs/006's census classified 500918 trajectories and 5699 converged ones at
depths 3-19 won 0 of 12 evaluation solves, so iteration reaches new grids and not new SOLVED grids.
Every gain in that feature came from minting new primitives through the typed bottleneck.

WHAT THIS WIDENS. arc_mint could already invent PARAMETERS (the _RULE_FORMS ladder) and it could
already invent WRITE SETS -- but only over an AUTHORED enumeration: 8 authored (unit, attribute)
tuples in _RECOLOR_UNITS and 2 authored entries in _PAINT_CELLSETS, built on 3 of arc_scene's 35
top-level functions (object_holes, period, shape_class -- arc_mint.py:46). The perception layer
computed scene structure the minting layer never asked for. This module turns that authored
enumeration into an INDUCED CONJUNCTION over measured descriptors:

    apply(g):  U = units(g)
               T = { v in U : selector(v) }        # induced conjunction over measured descriptors
               for v in T: write rule(key(v)) into cells(v)

so the mint no longer has to explain EVERY unit to emit a record -- it can induce WHICH units get
written. That is the whole mechanism; nothing else in the design changed.

THE TYPED BOTTLENECK IS PRESERVED BY A VERSIONED ENUMERATION, NOT BY HOPE (contract C10/C11). A
record carries the literal tuple ((predicate-id, value), ...) drawn from DESCRIPTORS below and NEVER
a pointer into mint-internal state, so it stays reconstructible from its declarative fields alone.
DESCRIPTOR_VERSION pins the enumeration: bumping it invalidates every stored record, because a
record referencing predicate #17 is only reconstructible against the alphabet that defines #17.
Reconstruction stays the audit (tests/test_arc_residual.py, tests/test_arc_fix.py).

P1 AUDIT (docs/INVENTION.md 3.5):
  * selector stopping rule -- STRICT improvement in misclassified units. A count, no threshold.
  * literal choice -- the literal that removes the most negatives, ties broken by predicate id.
    Counts and an enumeration order; nothing numeric arbitrates.
  * leave-one-out -- inherited from arc_mint.induce_rule and applied to the selector too.
  * SELECTOR_LITERAL_CAP -- DECLARED RESOURCE GUARD in the same class as node_budget, not an
    arbiter: it bounds how long a conjunction may get, never which conjunction wins. Reported as a
    curve (specs/006-fixed-point-gf/designb/), never as a single value that decides a verdict.
  * DECLARED OFFENDER, named so nobody thinks it was cleaned: the descriptor alphabet itself is
    ASSEMBLED BY A DEVELOPER from arc_scene's public measurements. This design moves the authored
    boundary FROM primitives TO measurements and grows compositionally above it. It does NOT remove
    the boundary, and that prior stays charged against any score, exactly as docs/DEPTH_CEILINGS.md
    charges the current DSL.

ZERO LIVE-TICK COST: this is symbolic Python over grids reached only from arc_mint/arc_solver. It
touches no tick, has no per-tick cost at any population width, and no relationship to the 24-240
live band or the CUDA crossover.

Determinism (CLAUDE.md): pure functions of the grid; no random; no builtin hash on persisted values
(arc_scene.shape_class uses zlib.crc32); sorted iteration; ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.arc_residual
"""

from snn_data_lake.reasoning import arc_scene
from snn_data_lake.reasoning.arc_solver import background, components

# ====================================================================================================
# the VERSIONED DESCRIPTOR ALPHABET -- bump the version and every stored record is invalidated
# ====================================================================================================

DESCRIPTOR_VERSION = 1

# (predicate id, name, family). `family` names the arc_scene measurement that produces it, so the
# cost of a record is auditable: compile-time only computes the families its literals reference.
# Every descriptor is INTEGER-VALUED (or None = undefined on this unit) so a literal is a pair of
# ints and the record carries no object, grid or mint-internal state.
DESCRIPTORS = (
    (0, "color", "base"),
    (1, "size", "base"),
    (2, "height", "morph"),
    (3, "width", "morph"),
    (4, "side", "morph"),
    (5, "area_bbox", "morph"),
    (6, "is_solid_rect", "morph"),
    (7, "is_line", "morph"),
    (8, "n_interior", "morph"),
    (9, "nholes", "topo"),
    (10, "shapecells", "base"),
    (11, "shapeclass", "shape"),
    (12, "shape_rank", "shape"),
    (13, "n_shape_twins", "shape"),
    (14, "is_modal_color", "palette"),
    (15, "is_rarest_color", "palette"),
    (16, "n_color_twins", "palette"),
    (17, "size_rank", "order"),
    (18, "size_rank_desc", "order"),
    (19, "is_largest", "order"),
    (20, "is_smallest", "order"),
    (21, "touches_border", "border"),
    (22, "n_border_sides", "border"),
    (23, "in_border_pair", "bpairs"),
    (24, "panel_index", "sep"),
    (25, "on_separator", "sep"),
    (26, "n_loci_crossed", "locus"),
    (27, "n_sym_self", "sym"),
    (28, "n_aligned", "rel"),
    (29, "n_adjacent", "rel"),
    (30, "is_inner", "rel"),
    (31, "is_outer", "rel"),
)

DESCRIPTOR_NAMES = tuple(name for _pid, name, _fam in DESCRIPTORS)
DESCRIPTOR_FAMILY = tuple(fam for _pid, _name, fam in DESCRIPTORS)
N_DESCRIPTORS = len(DESCRIPTORS)

# DECLARED RESOURCE GUARD (not an arbiter): the longest conjunction the greedy covering may build.
# It bounds work, never choice -- the strict-improvement rule below is what stops the search in
# practice. Reported as a curve alongside every result.
SELECTOR_LITERAL_CAP = 4

# grid -> per-object descriptor rows. The searcher re-reaches the same grids constantly (equivalence
# pruning keeps states, not work), so without this the descriptor families would be recomputed on
# every node. arc_solver.synthesize() clears it per task exactly as it clears _SEG_CACHE.
_DESC_CACHE = {}


def clear_cache():
    """Drop the per-grid descriptor cache. Called by arc_solver.synthesize() per task."""
    _DESC_CACHE.clear()


# ---- the measurements -------------------------------------------------------------------------------
def _objs(g):
    return components(g, 4, True, background(g))


def _rank_map(values):
    """value -> 0-based rank among the sorted DISTINCT values. Grid-relative, so no constant."""
    return {v: i for i, v in enumerate(sorted(set(values)))}


def _family_rows(fam, g, objs):
    """Compute one descriptor family for every object. Returns {pid: [value per object]}.

    Each branch consumes an arc_scene measurement directly -- this is the module's whole point, so
    the mapping from family name to arc_scene function is kept one-to-one and obvious."""
    h, w = len(g), len(g[0])
    n = len(objs)
    if fam == "base":
        return {
            0: [o["color"] for o in objs],
            1: [o["size"] for o in objs],
            10: [len(o["shape"]) for o in objs],
        }
    if fam == "morph":
        ms = [arc_scene.morph(o) for o in objs]
        return {
            2: [m["height"] for m in ms],
            3: [m["width"] for m in ms],
            4: [m["side"] for m in ms],
            5: [m["area_bbox"] for m in ms],
            6: [1 if m["is_solid_rect"] else 0 for m in ms],
            7: [1 if m["is_line"] else 0 for m in ms],
            8: [len(m["interior"]) for m in ms],
        }
    if fam == "topo":
        return {9: [arc_scene.object_holes(o)[1] for o in objs]}
    if fam == "shape":
        cls = [arc_scene.shape_class(o["shape"]) for o in objs]
        rank = _rank_map(cls)
        twins = []
        for i, o in enumerate(objs):
            # shape_class is a crc32 id; shape_match VERIFIES the alignment, so a crc32 collision
            # cannot inflate the twin count (arc_scene.shape_match returns None when nothing aligns).
            twins.append(
                sum(
                    1
                    for j, o2 in enumerate(objs)
                    if j != i and cls[j] == cls[i] and arc_scene.shape_match(o, o2) is not None
                )
            )
        return {11: cls, 12: [rank[c] for c in cls], 13: twins}
    if fam == "palette":
        pal = arc_scene.palette(g)
        cols = [o["color"] for o in objs]
        return {
            14: [1 if c == pal["modal"] else 0 for c in cols],
            15: [1 if c == pal["rarest"] else 0 for c in cols],
            16: [sum(1 for c2 in cols if c2 == c) - 1 for c in cols],
        }
    if fam == "order":
        sizes = [o["size"] for o in objs]
        rank = _rank_map(sizes)
        top = len(rank) - 1
        return {
            17: [rank[s] for s in sizes],
            18: [top - rank[s] for s in sizes],
            19: [1 if rank[s] == top else 0 for s in sizes],
            20: [1 if rank[s] == 0 else 0 for s in sizes],
        }
    if fam == "border":
        ring = arc_scene.perimeter(h, w)
        rows0, rowsl = set(arc_scene.row_line(0, w)), set(arc_scene.row_line(h - 1, w))
        cols0, colsl = set(arc_scene.col_line(0, h)), set(arc_scene.col_line(w - 1, h))
        touch, nsides = [], []
        for o in objs:
            cells = o["cells"]
            touch.append(1 if cells & ring else 0)
            nsides.append(sum(1 for s in (rows0, rowsl, cols0, colsl) if cells & s))
        return {21: touch, 22: nsides}
    if fam == "bpairs":
        pairs = arc_scene.border_pairs(objs, h, w)
        inp = set()
        for i, j in pairs:
            inp.add(i)
            inp.add(j)
        return {23: [1 if i in inp else 0 for i in range(n)]}
    if fam == "sep":
        seps = arc_scene.separators(g)
        panels = seps["panels"]
        sep_rows = {r for s, e, _c in seps["row_separators"] for r in range(s, e + 1)}
        sep_cols = {c for s, e, _c in seps["col_separators"] for c in range(s, e + 1)}
        idx, on = [], []
        for o in objs:
            found = -1
            for pi, (r0, c0, r1, c1) in enumerate(panels):
                if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in o["cells"]):
                    found = pi
                    break
            idx.append(found)
            on.append(1 if any(r in sep_rows or c in sep_cols for r, c in o["cells"]) else 0)
        return {24: idx, 25: on}
    if fam == "locus":
        crossed = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                for k in arc_scene.crossings(objs[i], objs[j], objs):
                    if k != i and k != j:
                        crossed[k] += 1
        return {26: crossed}
    if fam == "sym":
        axes = arc_scene.symmetry_axes(g)
        if not axes:
            return {27: [0] * n}
        best = max(axes, key=lambda a: (a["score"], a["transform"]))
        img = {}
        for a, b in best["correspondence"]:
            img[a] = b
            img[b] = a
        return {27: [sum(1 for c in o["cells"] if img.get(c) in o["cells"]) for o in objs]}
    if fam == "rel":
        # arc_scene.relations() emits one record per related (i<j) pair with the flags it found:
        # same_row / same_col (bbox axis overlap), bbox_adjacent, contains -> (outer, inner).
        aligned, adjacent, inner, outer = [0] * n, [0] * n, [0] * n, [0] * n
        for rec in arc_scene.relations(objs):
            i, j = rec["i"], rec["j"]
            if rec.get("same_row") or rec.get("same_col"):
                aligned[i] += 1
                aligned[j] += 1
            if rec.get("bbox_adjacent"):
                adjacent[i] += 1
                adjacent[j] += 1
            if "contains" in rec:
                out_i, in_i = rec["contains"]
                outer[out_i] = 1
                inner[in_i] = 1
        return {28: aligned, 29: adjacent, 30: inner, 31: outer}
    return {}


def descriptor_rows(g, families=None):
    """Per-object descriptor rows for `g`: a list (in components() order) of N_DESCRIPTORS-tuples.

    `families` restricts the computation to the families actually needed -- at APPLY time a record
    only references its selector's and key's families, so an unreferenced O(n^3) measurement is
    never paid for. Unrequested slots read None and no literal can match them."""
    want = None if families is None else frozenset(families)
    key = (g, want)
    hit = _DESC_CACHE.get(key)
    if hit is not None:
        return hit
    objs = _objs(g)
    n = len(objs)
    cols = {}
    for fam in sorted(set(DESCRIPTOR_FAMILY)):
        if want is not None and fam not in want:
            continue
        try:
            cols.update(_family_rows(fam, g, objs))
        except Exception:  # noqa: BLE001 -- a measurement that cannot read this grid is simply absent
            continue
    rows = tuple(
        tuple(
            cols[pid][i] if pid in cols and i < len(cols[pid]) else None
            for pid in range(N_DESCRIPTORS)
        )
        for i in range(n)
    )
    _DESC_CACHE[key] = rows
    return rows


def families_for(selector, key_pid=None):
    """The descriptor families a compiled record must compute -- its literals' plus its key's."""
    fams = {DESCRIPTOR_FAMILY[pid] for pid, _v in selector}
    if key_pid is not None:
        fams.add(DESCRIPTOR_FAMILY[key_pid])
    return frozenset(fams)


# ====================================================================================================
# the SELECTOR: an induced conjunction of (predicate id, value) literals
# ====================================================================================================


def selector_matches(row, selector):
    """True iff every literal holds on this unit's descriptor row. The empty conjunction is TRUE --
    that is the degenerate always-true selector, and it is exactly what the pre-registered falsifier
    demands a claimed win NOT depend on."""
    for pid, val in selector:
        if row[pid] != val:
            return False
    return True


def induce_selector(pos_rows, neg_rows, cap=SELECTOR_LITERAL_CAP):
    """Greedy sequential covering over the descriptor alphabet (docs/INVENTION.md B4).

    pos_rows / neg_rows: descriptor rows of the units that MUST and MUST NOT be selected, pooled
    over every train pair. Adds one literal at a time; a literal is admissible only if it holds on
    EVERY positive (the conjunction may never drop a positive), and is kept only if it STRICTLY
    reduces the number of negatives still matched. Returns the literal tuple -- () when there is
    nothing to exclude -- or None when the alphabet cannot separate the two sets.

    No numeric threshold arbitrates: the stopping rule is strict improvement, ties break on
    predicate id, and `cap` is a declared resource guard that bounds length, never choice."""
    if not pos_rows:
        return None
    sel = []
    live_neg = list(neg_rows)
    while live_neg and len(sel) < cap:
        best = None
        for pid in range(N_DESCRIPTORS):
            vals = {row[pid] for row in pos_rows}
            if len(vals) != 1:
                continue  # not shared by every positive -> the conjunction would drop a positive
            val = next(iter(vals))
            if val is None:
                continue
            rest = [row for row in live_neg if row[pid] == val]
            if len(rest) >= len(live_neg):
                continue  # no STRICT improvement -> not a literal worth its length
            cand = (len(rest), pid)
            if best is None or cand < best[0]:
                best = (cand, pid, val, rest)
        if best is None:
            return None  # the alphabet cannot separate these units
        _cand, pid, val, rest = best
        sel.append((pid, val))
        live_neg = rest
    if live_neg:
        return None  # still selecting a unit that must not be written
    return tuple(sel)


# ====================================================================================================
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def main():
    width = 96
    _emit("=" * width)
    _emit("ARC RESIDUAL -- versioned descriptor alphabet + induced-conjunction selector")
    _emit("=" * width)
    _emit(f"  descriptor alphabet version {DESCRIPTOR_VERSION}: {N_DESCRIPTORS} predicates")
    fams = sorted(set(DESCRIPTOR_FAMILY))
    _emit(f"  families ({len(fams)}): {', '.join(fams)}")

    # This block used to carry `ok = True`, never reassigned, and closed with
    # `_emit("RESULT: " + ("PASS" if ok else "FAIL"))` / `return 0 if ok else 1`. `ok` had no
    # reachable False, so "RESULT: PASS" was a constant wearing a signal's clothes (P3 shape 5) --
    # a printed verdict that could not report a failure. The `assert`s below are the actual gate:
    # they raise, the process exits non-zero, and run_selfchecks scores FAIL on the exit status.
    # Removed rather than wired up, because a second verdict channel beside the asserts is exactly
    # the extra moving part P2 forbids.

    # ---- the alphabet is a well-formed versioned enumeration ---------------------------------------
    assert [pid for pid, _n, _f in DESCRIPTORS] == list(range(N_DESCRIPTORS)), "ids must be dense"
    assert len(set(DESCRIPTOR_NAMES)) == N_DESCRIPTORS, "names must be unique"
    _emit(f"  [OK] predicate ids are dense 0..{N_DESCRIPTORS - 1} and names are unique")

    # ---- descriptors are measured, integer-valued and deterministic --------------------------------
    g = (
        (0, 0, 0, 0, 0, 0),
        (0, 3, 3, 3, 0, 5),
        (0, 3, 0, 3, 0, 0),
        (0, 3, 3, 3, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (7, 0, 0, 0, 0, 0),
    )
    clear_cache()
    rows = descriptor_rows(g)
    assert len(rows) == 3, f"three objects expected, got {len(rows)}"
    for row in rows:
        for v in row:
            assert v is None or isinstance(v, int), "descriptors are integer-valued or None"
    clear_cache()
    assert descriptor_rows(g) == rows, "descriptor rows must be deterministic"
    holed = [r for r in rows if r[9] and r[9] > 0]
    assert len(holed) == 1 and holed[0][1] == 8, "the ring object has 1 hole and 8 cells"
    _emit(
        f"  [OK] {len(rows)} units measured on {N_DESCRIPTORS} predicates; deterministic; ints only"
    )

    # counted, not assumed: how many predicates actually vary across these units
    varying = sum(1 for pid in range(N_DESCRIPTORS) if len({r[pid] for r in rows}) > 1)
    _emit(
        f"  [--] {varying} of {N_DESCRIPTORS} predicates are DIFFERENTIAL on this grid (telemetry)"
    )

    # ---- the selector separates, and refuses when it cannot ----------------------------------------
    sel = induce_selector([rows[0]], [rows[1], rows[2]])
    assert sel is not None and len(sel) >= 1, "a separable split must induce at least one literal"
    assert selector_matches(rows[0], sel), "the induced conjunction must hold on the positive"
    assert not any(selector_matches(r, sel) for r in rows[1:]), "and on no negative"
    _emit(f"  [OK] induced conjunction {sel} selects 1 of 3 units")

    # the empty conjunction is the degenerate always-true selector
    assert induce_selector(list(rows), []) == (), "nothing to exclude -> the empty conjunction"
    assert all(selector_matches(r, ()) for r in rows), "the empty conjunction is always true"
    _emit("  [OK] no negatives -> the EMPTY (always-true) conjunction, and it selects everything")

    # a positive and a negative with IDENTICAL descriptor rows cannot be separated: refuse, never guess
    assert induce_selector([rows[0]], [rows[0]]) is None, "identical rows must refuse"
    _emit("  [OK] indistinguishable units -> None (refuse), never a guess")

    # ---- the cap bounds length, never choice -------------------------------------------------------
    s1 = induce_selector([rows[0]], [rows[1], rows[2]], cap=1)
    assert s1 is None or len(s1) <= 1, "the cap bounds the conjunction length"
    _emit(f"  [OK] SELECTOR_LITERAL_CAP={SELECTOR_LITERAL_CAP} is a length guard (cap=1 -> {s1})")

    # ---- families restrict the work a compiled record pays for -------------------------------------
    clear_cache()
    cheap = descriptor_rows(g, families_for(((1, 8),), key_pid=0))
    assert all(r[26] is None for r in cheap), "an unreferenced family must not be computed"
    assert [r[1] for r in cheap] == [r[1] for r in rows], "a referenced family must still be exact"
    _emit("  [OK] families_for() pays only for the referenced measurements (unrequested -> None)")

    _emit("=" * width)
    _emit("RESULT: every assert above held (a failure raises and exits non-zero)")
    _emit("=" * width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
