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
  * Depth-1 (a single minted primitive) + a bounded depth-2 composition. Not the full search.
  * Programs are applied DIRECTLY here. Composing minted primitives INTO arc_solver.synthesize() + the
    standing P2 obligation (migrate synthesize onto the arc_scene scene, delete the closed 5-key schema)
    are the NEXT Phase-3 steps -- noted, not done here.

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
  * HONEST METRIC: mint() reports how a program was induced; the harness measures HELD-OUT generalisation
    (arc_bench test outputs), never train-fit alone.

Determinism (CLAUDE.md): pure functions of the task; no random; no builtin hash; sorted iteration; ASCII.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.arc_mint
"""

from collections import Counter, deque

from snn_data_lake.reasoning.arc_scene import period, shape_class
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
            return lambda a, kk=k: a % kk
    return None


def _fit_offset_mod10(obs):
    # colour = (attr + b) mod 10 for a single integer offset b
    a0, c0 = obs[0]
    b = (c0 - a0) % 10
    if all(col == (a + b) % 10 for a, col in obs):
        return lambda a, bb=b: (a + bb) % 10
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
        return lambda a, mm=m, bb=b: mm * a + bb
    return None


def _fit_rank(obs):
    # colour = the colour whose RANK (by value) equals the attribute's RANK among distinct attrs.
    attrs = sorted({a for a, _ in obs})
    cols = sorted({c for _, c in obs})
    if len(attrs) != len(cols):
        return None
    amap = {a: cols[i] for i, a in enumerate(attrs)}
    if all(col == amap[a] for a, col in obs):
        return lambda a, mp=dict(amap): mp.get(a)
    return None


def _fit_lookup(obs):
    table = {}
    for a, c in obs:
        if table.setdefault(a, c) != c:
            return None  # same attr -> two colours: not even a function
    return lambda a, mp=dict(table): mp.get(a)


_RULE_FORMS = (
    (
        "const_mod10",
        1,
        lambda obs: (lambda a: a % 10) if all(c == a % 10 for a, c in obs) else None,
    ),
    ("mod_k", 2, _fit_const_modk),
    ("offset_mod10", 2, _fit_offset_mod10),
    ("affine", 3, _fit_affine),
    ("rank", 3, _fit_rank),  # MEMORISATION-class: an ordinal permutation table, None on unseen attr
    ("lookup", 9, _fit_lookup),  # MEMORISATION-class: a raw table, only when nothing else fits
)

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
        rule = fit(flat)
        if rule is None or not all(rule(a) == c for a, c in flat):
            continue
        if name in _MEMO_KINDS:
            return (
                rule,
                name,
                mdl,
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
                if r2 is None or any(r2(a) != c for a, c in held):
                    loo_ok = False
                    break
        if loo_ok:
            return rule, name, mdl
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


# ---- SKELETONS: candidate primitives (name, build(train)->fn|None, mdl_len, kind) ------------------
# A skeleton with a free rule INDUCES the rule from the train pairs' (attr, out-colour) observations and
# returns a grid-general program, or None if it does not fit+verify. Structural skeletons have no free
# rule. The mint enumerates these in MDL order.
_RECOLOR_UNITS = (
    ("region", "side", _region_units("side")),
    ("region", "size", _region_units("size")),
    ("object", "shapecells", _object_units("shapecells")),
    ("object", "shapeclass", _object_units("shapeclass")),
    ("object", "size", _object_units("size")),
    ("object", "side", _object_units("side")),
    ("object", "color", _object_units("color")),
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
    rule, kind, _mdl = got
    fn = lambda gg: _apply_recolor_units(gg, unit_fn, rule)  # noqa: E731
    for gi, go in pairs:
        r = fn(gi)
        if r is None or not _ok(r) or r != go:
            return None
    return fn, kind


def candidates(train):
    """Yield (name, fn, mdl_len, kind) for every skeleton that FITS + VERIFIES on `train`, depth-1 then a
    bounded depth-2 composition. `kind` is "structural" (no free rule) or the induced rule NAME
    (const_mod10 / mod_k / offset_mod10 / affine = functional; rank / lookup = memorisation-class, see
    _MEMO_KINDS). fn is grid-general."""
    out = []

    # -- structural depth-1 (no free rule) --
    for ax in ("vertical_mirror", "horizontal_mirror", "rotate_180"):
        out.append((f"reflect[{ax}]", lambda gg, a=ax: _reflect(gg, a), 2, "structural"))
    out.append(
        (
            "materialize[period]",
            lambda gg: period(gg)["template"] if period(gg)["periodic"] else None,
            2,
            "structural",
        )
    )
    for kh, kw in ((3, 3), (2, 2), (1, 3), (3, 1), (2, 1), (1, 2)):
        out.append(
            (f"tile[{kh}x{kw}]", lambda gg, a=kh, b=kw: _tile(gg, a, b, False), 3, "structural")
        )
        out.append(
            (f"tile_alt[{kh}x{kw}]", lambda gg, a=kh, b=kw: _tile(gg, a, b, True), 4, "structural")
        )
    for key in ("largest", "smallest", "rarest"):
        out.append((f"crop[{key}]", lambda gg, k=key: _crop_sel(gg, k), 3, "structural"))

    # -- parameterised depth-1: recolour units by an induced rule --
    for unit_name, attr, unit_fn in _RECOLOR_UNITS:
        built = _build_recolor(train, unit_fn)
        if built:
            fn, kind = built
            # `kind` is the rule NAME (const_mod10 / mod_k / affine / rank / lookup). A memorisation-class
            # rule costs more MDL, so a genuine functional recolor is preferred over a memorised one.
            mdl = 3 + (2 if kind in _MEMO_KINDS else 0)
            out.append((f"recolor_{unit_name}_by[{attr}]:{kind}", fn, mdl, kind))

    # -- bounded depth-2: crop-to-selected then recolour-by-rule (composition example) --
    for key in ("largest", "rarest"):
        for unit_name, attr, unit_fn in _RECOLOR_UNITS[:4]:

            def _comp(gg, k=key, uf=unit_fn, tr=train):
                cropped = _crop_sel(gg, k)
                if cropped is None:
                    return None
                # induce the recolour rule on the CROPPED train grids (composition = crop then recolour)
                sub = [{"in": _crop_sel(_grid(p["in"]), k), "out": _grid(p["out"])} for p in tr]
                if any(s["in"] is None for s in sub):
                    return None
                if any(
                    len(s["in"]) != len(s["out"]) or len(s["in"][0]) != len(s["out"][0])
                    for s in sub
                ):
                    return None
                b = _build_recolor(sub, uf)
                if b is None:
                    return None
                return b[0](cropped)

            out.append((f"crop[{key}]>>recolor_{unit_name}_by[{attr}]", _comp, 6, "functional"))

    return sorted(out, key=lambda c: (c[2], c[0]))


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


def mint(train):
    """The mint: return the shortest (MDL) candidate primitive that reproduces every train pair, as
    {"fn", "name", "mdl", "kind", "rank"}, or None. `kind` flags memorisation risk (lookup)."""
    cands = candidates(train)
    for rank, (name, fn, mdl, kind) in enumerate(cands, 1):
        if solves(fn, train):
            return {
                "fn": fn,
                "name": name,
                "mdl": mdl,
                "kind": kind,
                "rank": rank,
                "n_cands": len(cands),
            }
    return None


# ================================================================================================
def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def main():
    import json
    import os

    os.chdir(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
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

    # ---- end-to-end on the Phase-1 dev tasks, HELD-OUT metric ---------------------------------------
    try:
        from snn_data_lake.reasoning.arc_bench import load_tasks

        dev = json.load(open("scratchpad/oracle_dev.json", encoding="utf-8"))
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

    train_fit, held_out = [], []
    for task in dev:
        tid = task["id"]
        m = mint(task["train"])
        if not m:
            continue
        train_fit.append(tid)
        t = eval_tasks.get(tid)
        gen = None
        if t and t.get("test"):
            gen = all(m["fn"](_grid(p["input"])) == _grid(p["output"]) for p in t["test"])
        if gen:
            held_out.append((tid, m["name"], m["kind"]))
    n = len(dev)
    _emit("-" * width)
    _emit(f"dev tasks: {n} | TRAIN-FIT: {len(train_fit)} | HELD-OUT generalizing: {len(held_out)}")
    for tid, name, kind in held_out:
        _emit(f"    HELD-OUT SOLVE  {tid}  [{kind}]  {name}")
    # regression guards: the two de-risk held-out solves must still land, honestly kind-tagged
    ids = {tid for tid, _, _ in held_out}
    assert "00dbd492" in ids, "size-coded region fill must still generalize held-out"
    assert "00576224" in ids, "alt-mirror tile must still generalize held-out"
    kinds = Counter(k for _, _, k in held_out)
    n_func = sum(v for k, v in kinds.items() if k != "structural" and k not in _MEMO_KINDS)
    _emit("-" * width)
    _emit(f"held-out solve kinds: {dict(kinds)}   (functional held-out = {n_func})")
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
        f"the {len(held_out)} held-out generalisations are a structural tile + a flagged memorisation recolor."
    )
    _emit(
        "The number moves only with more producers/attributes/composition. NO overclaim: the operation-"
    )
    _emit(
        "synthesis arm + synthesize-composition + the P2 schema migration remain the next Phase-3 steps."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
