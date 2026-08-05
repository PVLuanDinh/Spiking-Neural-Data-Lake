"""
ARC MINT -- Gf Phase-3 (first build): hypothesise out-of-closure grid->grid primitives FROM the arc_scene
perceptual vocabulary, induce their free parameters with MEMORISATION-RESISTANT induction, and keep only
programs that reproduce every training pair. This is the "mint": perception -> hypothesis -> verified named
transform. It is the genuine remaining fluid-intelligence move -- inducing a primitive OUTSIDE the
arc_solver DSL closure, from measurements, with no hand-authored answer.

Honest scope of THIS build (do not overclaim):
  * It targets the MEASUREMENT core: transforms expressible as "measure a per-object / per-region / grid
    attribute with arc_scene, then recolour / crop / reflect / tile / materialise". The ~30% operation-not-
    measurement family (sequential path, stateful transport) is NOT covered -- that is the deferred
    operation-synthesis arm (Phase-3 continues).
  * MERGED (2026-08-03, specs/006-fixed-point-gf): the mint's ONLY consumer-facing surface is
    mint_records(train) -> TypedPrimitiveRecord -> arc_solver._pool() as `mint:*` primitives; the
    ONE searcher composes and applies them (contract C10-C13). The former direct-application path
    (candidates()/mint(), depth-1 + a bounded depth-2 harness) is RETIRED -- measured at adoption:
    evaluation 12 -> 18 of 400 and training 43 -> 53 of 400, no lost ID unaccounted (verdict.md).

Policy compliance:
  * NON-ADDITIVE: perception is REUSED from arc_scene / arc_solver.components() -- no segmentation is
    reimplemented. The mint is a consumer of the vocabulary, not a fork of it.
  * NO HAND-AUTHORED PREDICATES (theater-guard #1): every candidate is a generic skeleton whose parameters
    are INDUCED from the task's own train pairs. Nothing here encodes a specific task's answer.
  * MEMORISATION-RESISTANT (the Phase-3 de-risk finding): a raw lookup map fit on train overfits (a
    lookup solved train but failed held-out in the de-risk). So induction PREFERS FUNCTIONAL RULES
    (const / mod-k / offset / affine -- parametric, generalise to unseen attribute values by construction)
    in MDL order, and validates each by LEAVE-ONE-OUT over the train pairs. MEMORISATION-class forms
    (`rank` ordinal table, `lookup` raw table) are used only if no functional form validates; they return
    None on an unseen attribute (no answer, never a wrong guess) and are FLAGGED as their kind so the
    held-out metric -- never train-fit -- is the arbiter. HONEST: on the 20-task dev set the functional
    apparatus contributes ZERO held-out solves today (the recolour generalisation is a flagged lookup);
    functional wins need more producers/attributes/composition. The mechanism is unit-verified, not the
    dev-set showcase.
  * HONEST METRIC: every record carries its rule_form and memo_flag, so a solve reports how it was
    induced; the harness measures HELD-OUT generalisation (arc_bench test outputs), never train-fit
    alone -- and scores THROUGH arc_solver.make_solver(), never by direct application.

Determinism (CLAUDE.md): pure functions of the task; no random; no builtin hash; sorted iteration; ASCII.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.arc_mint
"""

from collections import Counter, deque
from typing import NamedTuple

from snn_data_lake.reasoning import arc_residual
from snn_data_lake.reasoning.arc_scene import object_holes, period, shape_class
from snn_data_lake.reasoning.arc_solver import background, components

_T = tuple


# ---- helpers ---------------------------------------------------------------------------------------
def _ok(g):
    """Legal ARC grid: rectangular, sides 1..30, ints 0-9."""
    if not g or not isinstance(g, tuple):
        return False
    h = len(g)
    if not (1 <= h <= 30):
        return False
    w = len(g[0]) if h else 0
    if not (1 <= w <= 30):
        return False
    for row in g:
        if len(row) != w or any(not isinstance(v, int) or not (0 <= v <= 9) for v in row):
            return False
    return True


def _grid(rows):
    return _T(_T(int(v) for v in row) for row in rows)


def _objs(g):
    return components(g, 4, True, background(g))


def enclosed_regions(g):
    """Grid-level TOPO: background cells NOT reachable from the border by 4-connected bg moves, grouped
    into 4-connected regions. A grid-level companion to arc_scene.object_holes (which is per-object).
    Returns a list of frozenset cellsets (sorted, deterministic)."""
    h, w = len(g), len(g[0])
    bg = background(g)
    outside, q = set(), deque()
    for r in range(h):
        for c in (0, w - 1):
            if g[r][c] == bg and (r, c) not in outside:
                outside.add((r, c))
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r][c] == bg and (r, c) not in outside:
                outside.add((r, c))
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == bg and (nr, nc) not in outside:
                outside.add((nr, nc))
                q.append((nr, nc))
    enclosed = {
        (r, c) for r in range(h) for c in range(w) if g[r][c] == bg and (r, c) not in outside
    }
    regions, seen = [], set()
    for cell in sorted(enclosed):
        if cell in seen:
            continue
        comp, stack = [], [cell]
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc = (r + dr, c + dc)
                if nc in enclosed and nc not in seen:
                    seen.add(nc)
                    stack.append(nc)
        regions.append(frozenset(comp))
    return regions


def _side(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


# ---- typed attribute readers (arc_scene measurements on a unit) ------------------------------------
def _obj_attr(o, attr):
    if attr == "size":
        return o["size"]
    if attr == "side":
        return _side(o["cells"])
    if attr == "color":
        return o["color"]
    if attr == "shapecells":
        return len(o["shape"])
    if attr == "shapeclass":
        return shape_class(o["shape"])
    if attr == "nholes":
        return object_holes(o)[1]
    return None


def _region_attr(cells, attr):
    if attr == "size":
        return len(cells)
    if attr == "side":
        return _side(cells)
    return None


def _object_units(attr):
    return lambda g: [(o["cells"], _obj_attr(o, attr)) for o in _objs(g)]


def _region_units(attr):
    return lambda g: [(cells, _region_attr(cells, attr)) for cells in enclosed_regions(g)]


# ---- MEMORISATION-RESISTANT rule induction ---------------------------------------------------------
# A "rule" maps an integer attribute value -> a colour 0..9. Functional forms are parametric (they
# generalise to unseen attribute values); a lookup is a finite table (memorises). Each form: (name,
# mdl_len, fit(obs)->rule|None) where obs is a list of (attr, colour) and rule is attr->colour|None.
def _fit_const_modk(obs):
    for k in range(2, 11):
        if all(col == (a % k) for a, col in obs):
            return (lambda a, kk=k: a % kk), (k,)
    return None


def _fit_offset_mod10(obs):
    # colour = (attr + b) mod 10 for a single integer offset b
    a0, c0 = obs[0]
    b = (c0 - a0) % 10
    if all(col == (a + b) % 10 for a, col in obs):
        return (lambda a, bb=b: (a + bb) % 10), (b,)
    return None


def _fit_affine(obs):
    # colour = m*attr + b, staying in 0..9; need >=2 distinct attrs to fit the line
    ymap = {}
    for a, c in obs:
        if ymap.setdefault(a, c) != c:
            return None  # same attr -> two colours: not a function
    xs = sorted(ymap)
    if len(xs) < 2:
        return None
    x0, x1 = xs[0], xs[-1]
    num, den = ymap[x1] - ymap[x0], x1 - x0
    if den == 0 or num % den != 0:
        return None
    m = num // den
    b = ymap[x0] - m * x0
    if all(col == m * a + b and 0 <= m * a + b <= 9 for a, col in obs):
        return (lambda a, mm=m, bb=b: mm * a + bb), (m, b)
    return None


def _fit_rank(obs):
    # colour = the colour whose RANK (by value) equals the attribute's RANK among distinct attrs.
    attrs = sorted({a for a, _ in obs})
    cols = sorted({c for _, c in obs})
    if len(attrs) != len(cols):
        return None
    amap = {a: cols[i] for i, a in enumerate(attrs)}
    if all(col == amap[a] for a, col in obs):
        return (lambda a, mp=dict(amap): mp.get(a)), tuple(sorted(amap.items()))
    return None


def _fit_lookup(obs):
    table = {}
    for a, c in obs:
        if table.setdefault(a, c) != c:
            return None  # same attr -> two colours: not even a function
    return (lambda a, mp=dict(table): mp.get(a)), tuple(sorted(table.items()))


_RULE_FORMS = (
    # every fit returns (rule_fn, rule_params) or None; rule_params fully determines the rule and
    # is what crosses the typed bottleneck (specs/006 data-model "Typed primitive record")
    (
        "const_mod10",
        1,
        lambda obs: ((lambda a: a % 10), ()) if all(c == a % 10 for a, c in obs) else None,
    ),
    ("mod_k", 2, _fit_const_modk),
    ("offset_mod10", 2, _fit_offset_mod10),
    ("affine", 3, _fit_affine),
    ("rank", 3, _fit_rank),  # MEMORISATION-class: an ordinal permutation table, None on unseen attr
    ("lookup", 9, _fit_lookup),  # MEMORISATION-class: a raw table, only when nothing else fits
)

# compile a rule back from (rule_form, rule_params) -- the auditable half of the bottleneck:
# mint_records() builds fn THROUGH these, so the declarative fields must reproduce behaviour
_RULE_COMPILE = {
    "const_mod10": lambda p: lambda a: a % 10,
    "mod_k": lambda p: lambda a, kk=p[0]: a % kk,
    "offset_mod10": lambda p: lambda a, bb=p[0]: (a + bb) % 10,
    "affine": lambda p: lambda a, mm=p[0], bb=p[1]: mm * a + bb,
    "rank": lambda p: lambda a, mp=dict(p): mp.get(a),
    "lookup": lambda p: lambda a, mp=dict(p): mp.get(a),
}

# rank + lookup are finite maps that return None on an unseen attribute -- they DO NOT generalise, so they
# are flagged as memorisation-risk (never passed off as functional) and cost more MDL at candidate level.
_MEMO_KINDS = frozenset({"rank", "lookup"})


def induce_rule(obs_per_pair):
    """Induce an attribute->colour rule from the train-pair observations, memorisation-resistantly.
    obs_per_pair: list (one per train pair) of [(attr, colour), ...]. Returns (rule_fn, kind, mdl_len) or
    None. Two-tier, MDL-ordered:
      * a FUNCTIONAL form (const / mod-k / offset / affine -- parametric, GENERALISES to unseen attribute
        values by construction) is preferred, and is accepted only if it also survives LEAVE-ONE-OUT over
        the train pairs (fit on the others, correctly predict the held-out pair) -- so a functional form
        that only fits by luck is skipped in favour of the next form.
      * MEMORISATION-class forms (`rank` an ordinal permutation table, `lookup` a raw table -- see
        _MEMO_KINDS) are finite maps that return None on an UNSEEN attribute (they do NOT generalise; at
        apply time they yield no answer, never a wrong one). They are accepted only if consistent (same
        attr -> same colour), FLAGGED as their kind (memorisation-risk -- never passed off as functional),
        and NOT leave-one-out-rejected (LOO is too strict for a sparse-but-genuine table -- an attribute
        value seen in only one pair). The HELD-OUT metric is the honest arbiter of any memorisation rule."""
    flat = [oc for pair in obs_per_pair for oc in pair]
    if not flat:
        return None
    npairs = len(obs_per_pair)
    for name, mdl, fit in _RULE_FORMS:
        got = fit(flat)
        if got is None:
            continue
        rule, params = got
        if not all(rule(a) == c for a, c in flat):
            continue
        if name in _MEMO_KINDS:
            return (
                rule,
                name,
                mdl,
                params,
            )  # memorisation-class: flagged, held-out judges it, no LOO pre-rejection
        loo_ok = (
            True  # functional forms must survive leave-one-out (parametric -> should generalise)
        )
        if npairs >= 2:
            for i in range(npairs):
                rest = [oc for j, pair in enumerate(obs_per_pair) if j != i for oc in pair]
                held = obs_per_pair[i]
                if not rest or not held:
                    continue
                r2 = fit(rest)
                if r2 is None or any(r2[0](a) != c for a, c in held):
                    loo_ok = False
                    break
        if loo_ok:
            return rule, name, mdl, params
    return None


# ---- CONSUMERS (typed value -> grid op) ------------------------------------------------------------
def _apply_recolor_units(g, unit_fn, rule):
    out = [list(row) for row in g]
    for cells, aval in unit_fn(g):
        if aval is None:
            return None
        col = rule(aval)
        if col is None or not (0 <= col <= 9):
            return None
        for r, c in cells:
            out[r][c] = col
    return _T(_T(row) for row in out)


def _reflect(g, axis):
    h, w = len(g), len(g[0])
    if axis == "vertical_mirror":
        return _T(_T(row[::-1]) for row in g)
    if axis == "horizontal_mirror":
        return _T(g[h - 1 - r] for r in range(h))
    return _T(_T(g[h - 1 - r][w - 1 - c] for c in range(w)) for r in range(h))


def _tile(g, kh, kw, alt):
    h, w = len(g), len(g[0])
    if not (1 <= kh * h <= 30 and 1 <= kw * w <= 30):
        return None
    flip = _T(_T(row[::-1]) for row in g)
    rows = []
    for ti in range(kh):
        src = g if (not alt or ti % 2 == 0) else flip
        for r in range(h):
            rows.append(_T(src[r]) * kw)
    return _T(rows)


def _crop_sel(g, key):
    os_ = _objs(g)
    if not os_:
        return None
    cnt = Counter(v for row in g for v in row)
    keyfn = {
        "largest": lambda o: (o["size"], -o["bbox"][0]),
        "smallest": lambda o: (-o["size"], o["bbox"][0]),
        "rarest": lambda o: (-cnt.get(o["color"], 0), o["bbox"][0]),
    }[key]
    b = max(os_, key=keyfn)["bbox"]
    return _T(_T(g[r][b[1] : b[3] + 1]) for r in range(b[0], b[2] + 1))


def _render(objs, h, w, bg):
    """Render only `objs` (their cells + colours) on a background canvas -- keep/delete output."""
    out = [[bg] * w for _ in range(h)]
    for o in objs:
        for r, c in o["cells"]:
            out[r][c] = o["color"]
    return _T(_T(row) for row in out)


def _obj_has(o, attr, os_, h, w):
    """Boolean object predicate (structural selection -- no free colour rule to induce)."""
    if attr == "holes":
        return object_holes(o)[1] > 0
    if attr == "solid_rect":
        r0, c0, r1, c1 = o["bbox"]
        return o["size"] == (r1 - r0 + 1) * (c1 - c0 + 1)
    if attr == "largest":
        return o is max(os_, key=lambda x: (x["size"], -x["bbox"][0]))
    if attr == "border":
        return any(r in (0, h - 1) or c in (0, w - 1) for r, c in o["cells"])
    return False


def _filter_objs(g, attr, keep):
    """Keep (keep=True) or delete (keep=False) objects satisfying a boolean attribute, rendered on bg."""
    os_ = _objs(g)
    if not os_:
        return None
    bg = background(g)
    h, w = len(g), len(g[0])
    sel = [o for o in os_ if _obj_has(o, attr, os_, h, w) == keep]
    return _render(sel, h, w, bg)


def _perimeter_cells(g):
    h, w = len(g), len(g[0])
    ring = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)}
    ring |= {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
    return frozenset(ring)


def _all_holes_cells(g):
    cells = set()
    for o in _objs(g):
        for hs in object_holes(o)[0]:
            cells |= set(hs)
    return frozenset(cells)


def _induce_paint_cellset(train, cellset_fn):
    """Paint a global CELLSET (perimeter / all object-holes) a single colour induced from train. The
    colour must be CONSTANT across all pairs (parametric const -> generalises); returns the program fn
    (grid-general) or None. Keeps the grid shape."""
    color = None
    pairs = []
    for pair in train:
        gi, go = _grid(pair["in"]), _grid(pair["out"])
        if len(gi) != len(go) or len(gi[0]) != len(go[0]):
            return None
        cells = cellset_fn(gi)
        if not cells:
            return None
        cc = {go[r][c] for r, c in cells}
        if len(cc) != 1:
            return None
        col = next(iter(cc))
        if color is None:
            color = col
        elif color != col:
            return None  # not a constant colour across pairs
        pairs.append((gi, go))

    def fn(gg, col=color):
        cells = cellset_fn(gg)
        if not cells:
            return None
        out = [list(row) for row in gg]
        for r, c in cells:
            out[r][c] = col
        return _T(_T(row) for row in out)

    for gi, go in pairs:
        if fn(gi) != go:
            return None
    return fn, color


# ---- SKELETONS: candidate primitives (name, build(train)->fn|None, mdl_len, kind) ------------------
# A skeleton with a free rule INDUCES the rule from the train pairs' (attr, out-colour) observations and
# returns a grid-general program, or None if it does not fit+verify. Structural skeletons have no free
# rule. The mint enumerates these in MDL order.
# NOT RETIRED -- and docs/INVENTION.md 3.4 item 2 is WRONG about this, measured. Design B claimed
# the induced conjunction SUBSUMES this authored enumeration, so the six ("object", attr) tuples
# could be deleted. They cannot. The two inductions express DIFFERENT task shapes:
#   * this path fits ONE rule over EVERY unit, so a unit that does not change is still a legal
#     observation -- (attr, the colour it already has) -- and a lookup absorbs it with no selector;
#   * _build_selected_recolor() treats an unchanged unit as a NEGATIVE and must find a conjunction
#     that excludes it, refusing outright when the alphabet cannot separate them.
# Port-before-retire (research.md R6) caught the difference causally on evaluation task ae58858e:
# retiring these six lost it (induce_selector returns None on 3-of-7, 3-of-9, 2-of-6, 1-of-6
# changed-unit splits) while the authored path solves it via recolor_object_by[shapecells]:lookup.
# Neither mechanism supersedes the other, so P2's adopt-and-retire does not apply here; recorded as
# a measured delta against the design in specs/006-fixed-point-gf/designb/.
_RECOLOR_UNITS = (
    ("region", "side", _region_units("side")),
    ("region", "size", _region_units("size")),
    ("object", "shapecells", _object_units("shapecells")),
    ("object", "shapeclass", _object_units("shapeclass")),
    ("object", "size", _object_units("size")),
    ("object", "side", _object_units("side")),
    ("object", "color", _object_units("color")),
    ("object", "nholes", _object_units("nholes")),
)


def _build_recolor(train, unit_fn):
    """Induce the attr->colour rule for a recolour-units skeleton, then verify grid-general on all pairs.
    Returns (fn, kind) or None. Each unit MUST be monochrome in the output (else the skeleton is wrong)."""
    pairs = []
    for pair in train:
        gi, go = _grid(pair["in"]), _grid(pair["out"])
        if len(gi) != len(go) or len(gi[0]) != len(go[0]):
            return None
        pairs.append((gi, go))
    obs_per_pair = []
    for gi, go in pairs:
        units = unit_fn(gi)
        if not units:
            return None
        obs = []
        for cells, aval in units:
            outcols = {go[r][c] for r, c in cells}
            if len(outcols) != 1 or aval is None:
                return None
            obs.append((aval, next(iter(outcols))))
        obs_per_pair.append(obs)
    got = induce_rule(obs_per_pair)
    if got is None:
        return None
    rule, kind, _mdl, params = got
    fn = lambda gg: _apply_recolor_units(gg, unit_fn, rule)  # noqa: E731
    for gi, go in pairs:
        r = fn(gi)
        if r is None or not _ok(r) or r != go:
            return None
    return fn, kind, params


# (candidates() and mint() -- the mint's DIRECT end-to-end application path -- were retired
# 2026-08-03 per specs/006-fixed-point-gf verdict.md (P2, contract C13): the A/B measured the
# merged path at 18 of 400 evaluation / 53 of 400 training vs 12 / 43 baseline with every lost
# ID accounted, so mint_records() -> arc_solver._pool() is now the ONE consumer-facing surface
# and synthesize() the ONE end-to-end ARC solve path. The bounded depth-2 composition that only
# candidates() carried is deliberately NOT ported: composition is the searcher's job, and the
# record run showed every mint-earned solve is depth-1 through the searcher.)


# ---- DESIGN B: recolour a unit set INDUCED as a conjunction over measured descriptors ------------
# docs/INVENTION.md section 3. The authored (unit, attribute) enumeration above could only emit a
# record when EVERY unit was explained; this induces WHICH units get written, so a task where some
# objects change and others do not is now expressible. The empty conjunction is the always-true
# selector and reproduces the authored all-units case exactly -- that is why the six authored object
# tuples could be retired rather than kept beside this.


def _split_changed(gi, go):
    """Classify each object of `gi` against `go`: (positives, negatives) as (index, target_colour).

    components(..., same_color=True) yields MONOCHROME objects, so a unit is either recoloured (its
    output cells are one colour, different from its input colour), left alone (output cells identical
    to input), or NEITHER -- and `neither` means a unit-recolour skeleton is simply the wrong shape
    for this task, so the whole build refuses. Returns None on refusal. Cells outside every object
    must be untouched: a recolour writes units, and nothing else."""
    if len(gi) != len(go) or len(gi[0]) != len(go[0]):
        return None
    objs = _objs(gi)
    if not objs:
        return None
    pos, neg, covered = [], [], set()
    for idx, o in enumerate(objs):
        cells = o["cells"]
        covered |= set(cells)
        outcols = {go[r][c] for r, c in cells}
        if len(outcols) != 1:
            return None  # not a monochrome write and not untouched -> wrong skeleton
        col = next(iter(outcols))
        if col == o["color"]:
            neg.append((idx, col))
        else:
            pos.append((idx, col))
    for r in range(len(gi)):
        for c in range(len(gi[0])):
            if (r, c) not in covered and gi[r][c] != go[r][c]:
                return None  # a cell outside every unit moved -> this is not a unit recolour
    return pos, neg


def _apply_selected_recolor(g, key_pid, rule, selector):
    """Write rule(key(v)) into every unit v the induced conjunction selects. A selector that selects
    NOTHING returns None -- a primitive that cannot act must fail like any other, never hand back its
    input as if it had answered."""
    fams = arc_residual.families_for(selector, key_pid)
    rows = arc_residual.descriptor_rows(g, fams)
    objs = _objs(g)
    if not objs or len(rows) != len(objs):
        return None
    out = [list(row) for row in g]
    hit = 0
    for o, row in zip(objs, rows):
        if not arc_residual.selector_matches(row, selector):
            continue
        aval = row[key_pid]
        if aval is None:
            return None
        col = rule(aval)
        if col is None or not (0 <= col <= 9):
            return None
        hit += 1
        for r, c in o["cells"]:
            out[r][c] = col
    if hit == 0:
        return None
    return _T(_T(row) for row in out)


def _selector_survives_loo(splits, rows_per_pair):
    """Leave-one-out on the SELECTOR, the same discipline induce_rule applies to the value rule: a
    conjunction induced without one train pair must still classify that pair's units correctly. An
    EMPTY conjunction is vacuous here (it selects everything by definition) and is not tested.

    Applied only when the REFIT is still cross-pair constrained, i.e. n - 1 >= 2. That bound is
    definitional, not tuned: a conjunction induced from a single pair has no agreement to satisfy,
    so it will happily pick a literal true of that pair alone and the test would reject genuine
    selectors rather than memorising ones. At n = 2 the cross-pair constraint is already carried by
    induce_selector itself -- a literal must hold on EVERY positive of EVERY pair to be admissible,
    which is what stops a per-pair accident like `color` from entering the conjunction."""
    n = len(splits)
    if n - 1 < 2:
        return True
    for i in range(n):
        pos, neg = [], []
        for j in range(n):
            if j == i:
                continue
            p, g_ = splits[j]
            pos.extend(rows_per_pair[j][k] for k, _c in p)
            neg.extend(rows_per_pair[j][k] for k, _c in g_)
        if not pos:
            continue
        sel = arc_residual.induce_selector(pos, neg)
        if sel is None:
            return False
        hp, hn = splits[i]
        if not all(arc_residual.selector_matches(rows_per_pair[i][k], sel) for k, _c in hp):
            return False
        if any(arc_residual.selector_matches(rows_per_pair[i][k], sel) for k, _c in hn):
            return False
    return True


def _build_selected_recolor(train):
    """Induce (selector, key descriptor, value rule) for the object-recolour consumer.

    Returns a list of candidate tuples in the mint_records() shape. The selector is induced ONCE per
    task (it separates written units from untouched ones and does not depend on the value key); each
    integer descriptor is then tried as the rule KEY in the existing MDL order. A candidate is
    emitted only if the compiled program reproduces EVERY train pair."""
    splits, rows_per_pair = [], []
    for pair in train:
        gi, go = pair["in"], pair["out"]
        got = _split_changed(gi, go)
        if got is None:
            return []
        splits.append(got)
        rows_per_pair.append(arc_residual.descriptor_rows(gi))
    pos_rows, neg_rows = [], []
    for (pos, neg), rows in zip(splits, rows_per_pair):
        pos_rows.extend(rows[k] for k, _c in pos)
        neg_rows.extend(rows[k] for k, _c in neg)
    if not pos_rows:
        return []  # nothing is written -> not a recolour task
    selector = arc_residual.induce_selector(pos_rows, neg_rows)
    if selector is None:
        return []  # the alphabet cannot separate written from untouched units -> refuse
    if selector and not _selector_survives_loo(splits, rows_per_pair):
        return []
    # With an EMPTY conjunction this induction and the authored _RECOLOR_UNITS path compile to the
    # same program, so the keys the authored path already carries would be minted twice and charged
    # twice against the node budget. Derived from _RECOLOR_UNITS, never hardcoded, so the two stay
    # in step if either changes. A NON-empty conjunction is never a duplicate: the authored path
    # cannot express it at all.
    authored = {a for u, a, _f in _RECOLOR_UNITS if u == "object"} if not selector else set()
    cand = []
    for pid in range(arc_residual.N_DESCRIPTORS):
        if arc_residual.DESCRIPTOR_NAMES[pid] in authored:
            continue
        obs_per_pair = []
        ok = True
        for (pos, _neg), rows in zip(splits, rows_per_pair):
            obs = []
            for k, col in pos:
                aval = rows[k][pid]
                if aval is None:
                    ok = False
                    break
                obs.append((aval, col))
            if not ok:
                break
            if obs:
                obs_per_pair.append(obs)
        if not ok or not obs_per_pair:
            continue
        got = induce_rule(obs_per_pair)
        if got is None:
            continue
        _rule, kind, rule_mdl, params = got
        memo = kind in _MEMO_KINDS
        attr = arc_residual.DESCRIPTOR_NAMES[pid]
        lits = "&".join(f"{arc_residual.DESCRIPTOR_NAMES[p]}={v}" for p, v in selector)
        name = f"recolor_object_by[{attr}]" + (f"{{{lits}}}" if lits else "") + f":{kind}"
        # MDL = the skeleton's own pre-existing hand-assigned 3 (NOT retired here: retiring it would
        # change the pool ORDER and confound the selector measurement, so it is named as a surviving
        # P1 offender in specs/006-fixed-point-gf/designb/) plus two COUNTS -- the rule form's
        # description length from _RULE_FORMS and the number of induced selector literals. Both are
        # counted, not chosen, and the memo forms already carry their cost there (rank 3, lookup 9).
        cand.append(
            (
                3 + rule_mdl + len(selector),
                name,
                "object",
                attr,
                kind,
                params,
                "recolor_sel",
                memo,
                selector,
            )
        )
    if not cand:
        return []
    # MDL, applied ACROSS the key alphabet and not just within one key. With few written units the
    # value rule is weakly constrained, so a dozen descriptors fit it by coincidence and each would
    # enter the pool as a separate primitive -- and at max_depth 3 the node count scales with the
    # CUBE of the pool size, so an unbounded key sweep buys vocabulary by starving the budget that
    # has to search it. Keeping the MDL-minimal keys is the module's existing arbiter doing the work
    # it exists for (a min over counted description lengths), not a new cap: nothing numeric is
    # chosen, and a key survives only by being the cheapest explanation on offer.
    best = min(c[0] for c in cand)
    return [c for c in cand if c[0] == best]


def solves(fn, train):
    for pair in train:
        gi, go = _grid(pair["in"]), _grid(pair["out"])
        try:
            r = fn(gi)
        except Exception:  # noqa: BLE001
            return False
        if r is None or not _ok(r) or r != go:
            return False
    return True


# ---- the typed bottleneck: mint -> searcher (specs/006-fixed-point-gf, FR-005, contract C10) ----


class TypedPrimitiveRecord(NamedTuple):
    """The ONLY object that crosses the mint -> searcher boundary. The declarative fields fully
    determine the program: `fn` is compiled FROM them by compile_record(), so the bottleneck is
    auditable by construction -- no grid, train pair, or mint-internal state crosses. Memo-flagged
    records keep refuse-on-unseen: fn returns None on an attribute value outside the induced
    table, and the searcher treats that None exactly as a failed primitive."""

    name: str
    unit: str  # "object" / "region" / "grid"
    attribute: str  # arc_scene attribute, or the structural variant this consumer applies
    rule_form: str  # const_mod10 / mod_k / offset_mod10 / affine / rank / lookup / const
    rule_params: tuple
    consumer: str  # recolor / crop / reflect / tile / materialize / paint / keep / delete
    memo_flag: bool
    selector: tuple  # DESIGN B: induced conjunction, ((predicate id, value), ...) from
    # arc_residual.DESCRIPTORS at DESCRIPTOR_VERSION. Literal values only -- never a pointer into
    # mint-internal state -- so the record stays reconstructible from its own fields. () is the
    # always-true conjunction. A version bump invalidates every stored record by construction.
    fn: object


# structural skeleton templates: (mdl, name, unit, attribute, rule_params, consumer). Nothing is
# induced for these; the tile shape lives in rule_params, the variant in `attribute`. NOTE the
# consumer set extends data-model.md's six with keep/delete -- recorded as a measured delta: the
# mint's keep/delete skeletons are real capability and the port-before-retire inventory (R6)
# forbids dropping them silently.
_STRUCT_RECORDS = tuple(
    [
        (2, f"reflect[{ax}]", "grid", ax, (), "reflect")
        for ax in ("vertical_mirror", "horizontal_mirror", "rotate_180")
    ]
    + [(2, "materialize[period]", "grid", "period", (), "materialize")]
    + [
        (3, f"tile[{kh}x{kw}]", "grid", "tiling", (kh, kw, 0), "tile")
        for kh, kw in ((3, 3), (2, 2), (1, 3), (3, 1), (2, 1), (1, 2))
    ]
    + [
        (4, f"tile_alt[{kh}x{kw}]", "grid", "tiling", (kh, kw, 1), "tile")
        for kh, kw in ((3, 3), (2, 2), (1, 3), (3, 1), (2, 1), (1, 2))
    ]
    + [(3, f"crop[{k}]", "object", k, (), "crop") for k in ("largest", "smallest", "rarest")]
    + [
        (3, f"keep[{a}]", "object", a, (), "keep")
        for a in ("holes", "solid_rect", "largest", "border")
    ]
    + [
        (4, f"delete[{a}]", "object", a, (), "delete")
        for a in ("holes", "solid_rect", "largest", "border")
    ]
)

_UNIT_FNS = {"object": _object_units, "region": _region_units}
_PAINT_CELLSETS = {"perimeter": _perimeter_cells, "allholes": _all_holes_cells}


def compile_record(unit, attribute, rule_form, rule_params, consumer, selector=()):
    """Compile the DECLARATIVE fields of a typed record back into the grid -> grid-or-None
    program. mint_records() builds every emitted fn through here, and the reconstruction test
    re-runs it from the fields alone -- the audit that nothing beyond the record crosses.

    `selector` is the DESIGN B induced conjunction over arc_residual.DESCRIPTORS; the default ()
    is the always-true conjunction, so a record minted before Design B compiles unchanged."""
    if consumer == "recolor":
        unit_fn = _UNIT_FNS[unit](attribute)
        rule = _RULE_COMPILE[rule_form](rule_params)
        return lambda gg: _apply_recolor_units(gg, unit_fn, rule)
    if consumer == "recolor_sel":
        # DESIGN B: write only the units the induced conjunction selects. `attribute` names the KEY
        # descriptor in arc_residual.DESCRIPTORS at DESCRIPTOR_VERSION; `selector` is the literal
        # tuple. A DISTINCT consumer from "recolor" because it is a distinct induction, not a
        # generalisation of it -- see the _RECOLOR_UNITS note on why neither subsumes the other.
        key_pid = arc_residual.DESCRIPTOR_NAMES.index(attribute)
        rule = _RULE_COMPILE[rule_form](rule_params)
        sel = tuple((int(p), int(v)) for p, v in selector)
        return lambda gg: _apply_selected_recolor(gg, key_pid, rule, sel)
    if consumer == "paint":
        cs_fn = _PAINT_CELLSETS[attribute]
        col = rule_params[0]

        def paint(gg, cs_fn=cs_fn, col=col):
            cells = cs_fn(gg)
            if not cells:
                return None
            out = [list(row) for row in gg]
            for r, c in cells:
                out[r][c] = col
            return _T(_T(row) for row in out)

        return paint
    if consumer == "reflect":
        return lambda gg, ax=attribute: _reflect(gg, ax)
    if consumer == "tile":
        kh, kw, alt = rule_params
        return lambda gg, a=kh, b=kw, c=alt: _tile(gg, a, b, bool(c))
    if consumer == "materialize":
        return lambda gg: period(gg)["template"] if period(gg)["periodic"] else None
    if consumer == "crop":
        return lambda gg, k=attribute: _crop_sel(gg, k)
    if consumer in ("keep", "delete"):
        return lambda gg, a=attribute, kp=(consumer == "keep"): _filter_objs(gg, a, kp)
    return None


def mint_records(train):
    """The typed bottleneck (contract C10/C11): induce, validate and VERIFY primitives on
    `train` -- a sequence of (input_grid, output_grid) pairs -- and return them as
    TypedPrimitiveRecord in deterministic MDL-then-name order.

    Emission rules: a record is emitted only if its compiled fn reproduces EVERY train pair
    (the mint's own verification bar, unchanged); recolour rules additionally carry the
    induce_rule discipline (functional forms leave-one-out-validated; rank/lookup accepted
    without LOO but FLAGGED memo -- refuse-on-unseen). The mint's bounded depth-2 compositions
    are deliberately NOT exported: composition is the one searcher's job after the merge."""
    pairs = [{"in": _grid(i), "out": _grid(o)} for i, o in train]
    cand = [
        (mdl, name, unit, attribute, "const", params, consumer, False, ())
        for mdl, name, unit, attribute, params, consumer in _STRUCT_RECORDS
    ]
    for pname, cs_fn in sorted(_PAINT_CELLSETS.items()):
        got = _induce_paint_cellset(pairs, cs_fn)
        if got:
            cand.append(
                (3, f"paint_{pname}[const]", "grid", pname, "const", (got[1],), "paint", False, ())
            )
    for unit_name, attr, unit_fn in _RECOLOR_UNITS:
        built = _build_recolor(pairs, unit_fn)
        if built:
            _fn, kind, params = built
            memo = kind in _MEMO_KINDS
            cand.append(
                (
                    3 + (2 if memo else 0),
                    f"recolor_{unit_name}_by[{attr}]:{kind}",
                    unit_name,
                    attr,
                    kind,
                    params,
                    "recolor",
                    memo,
                    (),
                )
            )
    cand.extend(_build_selected_recolor(pairs))
    recs = []
    for mdl, name, unit, attribute, rule_form, params, consumer, memo, selector in sorted(
        cand, key=lambda c: (c[0], c[1])
    ):
        fn = compile_record(unit, attribute, rule_form, params, consumer, selector)
        if fn is None or not solves(fn, pairs):
            continue
        recs.append(
            TypedPrimitiveRecord(
                name, unit, attribute, rule_form, params, consumer, memo, selector, fn
            )
        )
    return recs


# ================================================================================================
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def main():
    import json
    import os

    # Repo root, DERIVED -- never entered. This used to be `os.chdir(<repo root>)`, whose only job
    # was to make the relative `scratchpad/oracle_dev.json` below resolve. A self-check that changes
    # the PROCESS working directory is the module-scope-side-effect defect P4c records against the
    # daemon (`os.chdir` / `sys.path.insert` / `os.environ.setdefault` at import): `run_selfchecks`
    # spawns subprocesses so it was invisible there, but `run_selfchecks.run_one_in_process()` (the
    # frozen-app path, which has no interpreter to spawn) runs modules with `runpy` in ONE process --
    # so every module after this one would have inherited a different cwd. Resolving the path
    # absolutely does the same work and leaves the process alone.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    width = 96
    _emit("=" * width)
    _emit("ARC MINT -- Gf Phase-3: memorisation-resistant functional induction over arc_scene")
    _emit("=" * width)

    # ---- unit tests: memorisation-resistant induction -----------------------------------------------
    # a genuine functional rule (colour = attr mod 10) is induced as FUNCTIONAL, generalises to unseen attr
    r = induce_rule([[(3, 3), (7, 7)], [(5, 5)]])
    assert r is not None and r[1] in ("const_mod10", "mod_k"), r
    assert r[0](9) == 9, "functional rule must generalise to an unseen attribute value"
    _emit(
        f"  [OK] functional rule induced as '{r[1]}' (mdl {r[2]}); generalises to unseen attr -> {r[0](9)}"
    )

    # a FUNCTIONAL form is PREFERRED over a lookup when both fit (MDL): mod-10 data is not tagged 'lookup'
    r_pref = induce_rule([[(3, 3), (7, 7)], [(5, 5), (3, 3)]])
    assert r_pref is not None and r_pref[1] != "lookup", r_pref
    _emit(
        f"  [OK] functional PREFERRED over a lookup (MDL): mod-10 data tagged '{r_pref[1]}', not memorised"
    )

    # a NON-functional but consistent table (colours {5,5,8} on attrs {1,2,3}: not const/mod/offset/affine/
    # rank) is accepted but FLAGGED 'lookup' -- the mint never passes a memorised table off as a general
    # rule; the HELD-OUT metric is the arbiter, and the lookup returns None (no answer) on an unseen attr.
    r3 = induce_rule([[(1, 5), (2, 5), (3, 8)], [(1, 5), (2, 5), (3, 8)]])
    assert r3 is not None and r3[1] == "lookup", r3
    assert r3[0](99) is None, (
        "a lookup must return None (no answer) on an unseen attribute, not a guess"
    )
    _emit(
        f"  [OK] non-functional table flagged '{r3[1]}' (memorisation-aware); returns None on unseen attr"
    )

    # inconsistent map (same attr -> two colours) is not even a function -> rejected outright
    assert induce_rule([[(1, 4)], [(1, 5)]]) is None, "same attr -> two colours must be rejected"
    _emit("  [OK] a non-function (same attr, two colours) is rejected outright")

    # ---- record reconstruction: the declarative fields alone must rebuild fn (C10 audit) -----------
    demo = [
        {"in": [[0, 7, 0], [7, 0, 7], [0, 7, 0]], "out": [[0, 7, 0], [7, 4, 7], [0, 7, 0]]},
        {"in": [[7, 7, 0], [7, 0, 7], [0, 7, 7]], "out": [[7, 7, 0], [7, 4, 7], [0, 7, 7]]},
    ]
    demo_pairs = [(p["in"], p["out"]) for p in demo]
    for rec in mint_records(demo_pairs):
        fn2 = compile_record(rec.unit, rec.attribute, rec.rule_form, rec.rule_params, rec.consumer)
        assert all(fn2(_grid(i)) == _grid(o) for i, o in demo_pairs), rec.name
        assert rec.name.isascii()
    _emit("  [OK] typed records rebuild from declarative fields alone (the auditable bottleneck)")

    # ---- end-to-end on the Phase-1 dev tasks, HELD-OUT metric, THROUGH the one searcher ------------
    # (specs/006 C13/Q4: the mint is the PRODUCER -- mint_records() -> arc_solver._pool() as
    # mint:* entries; every solve below is scored through synthesize(), no direct application.)
    try:
        from snn_data_lake.reasoning.arc_bench import load_tasks

        with open(
            os.path.join(_repo_root, "scratchpad", "oracle_dev.json"), encoding="utf-8"
        ) as _f:
            dev = json.load(_f)
        eval_tasks = dict(load_tasks("evaluation"))
    except Exception as e:  # noqa: BLE001
        _emit(
            f"  [skip] dev/eval corpus unavailable ({type(e).__name__}); unit tests above are the gate"
        )
        _emit("-" * width)
        _emit(
            "self-check OK (unit level): functional-preferring, leave-one-out-validated rule induction."
        )
        return 0

    from snn_data_lake.reasoning.arc_bench import _grids_equal
    from snn_data_lake.reasoning.arc_solver import synthesize

    minted, held_out = [], []
    for task in dev:
        tid = task["id"]
        train = [(p["in"], p["out"]) for p in task["train"]]
        if mint_records(train):
            minted.append(tid)
        t = eval_tasks.get(tid)
        if not (t and t.get("test")):
            continue
        ok_all, used = True, set()
        for p in t["test"]:
            cands, progs = synthesize(
                train, p["input"], attempts=2, node_budget=4000, time_budget=5.0
            )
            hits = [i for i, c in enumerate(cands) if _grids_equal(c, p["output"])]
            if not hits:
                ok_all = False
                break
            used.update(
                n
                for n in progs[hits[0]][0]
                if (n[4:] if n.startswith("fix:") else n).startswith("mint:")
            )
        if ok_all:
            held_out.append((tid, tuple(sorted(used))))
    n = len(dev)
    _emit("-" * width)
    _emit(
        f"dev tasks: {n} of {n} scored | minted records on {len(minted)} of {n} | "
        f"HELD-OUT solved through the ONE searcher: {len(held_out)} of {n}"
    )
    for tid, used in held_out:
        via = ",".join(used) if used else "base DSL (no mint step in the winning program)"
        _emit(f"    HELD-OUT SOLVE  {tid}  via {via}")
    # regression guards: the two de-risk held-out solves must still land -- now THROUGH the searcher
    ids = {tid for tid, _ in held_out}
    assert "00dbd492" in ids, "size-coded region fill must still generalize held-out"
    assert "00576224" in ids, "alt-mirror tile must still generalize held-out"
    kind_count = Counter()
    for _tid, used in held_out:
        for nm in used:
            base = nm[4:] if nm.startswith("fix:") else nm
            k = base.rsplit(":", 1)[-1] if base.startswith("mint:recolor") else "structural"
            kind_count[k] += 1
    n_func = sum(v for k, v in kind_count.items() if k not in _MEMO_KINDS and k != "structural")
    n_memo = sum(v for k, v in kind_count.items() if k in _MEMO_KINDS)
    n_struct = kind_count.get("structural", 0)
    _emit("-" * width)
    _emit(f"held-out mint-step kinds: {dict(kind_count)}   (functional held-out = {n_func})")
    _emit(
        "self-check OK (MECHANISM): functional-rule induction (const/mod-k/offset/affine) is unit-verified"
    )
    _emit(
        "above -- preferred over memorisation, leave-one-out-validated, generalises to unseen attrs;"
    )
    _emit(
        "memorisation-class rules (rank/lookup) are FLAGGED and return None on an unseen attr, never a guess."
    )
    _emit(
        "HONEST RESULT: on THIS 20-task dev set the functional apparatus contributes ZERO held-out solves --"
    )
    _emit(
        f"the {len(held_out)} held-out carry {n_struct} structural + {n_memo} flagged-memorisation +"
        f" {n_func} functional mint steps."
    )
    _emit(
        "The number moves only with more producers/attributes/composition. NO overclaim: the operation-"
    )
    _emit("synthesis arm + the arc_scene schema migration remain future Phase-3 steps.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
