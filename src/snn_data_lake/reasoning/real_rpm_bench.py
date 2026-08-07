r"""HONEST held-out Raven's-Progressive-Matrices benchmark with a RENDERED-RASTER front-end.

WHY THIS EXISTS (the inflation it corrects)
-------------------------------------------
The existing RPM number (~98% on raven_bench) is a GRAMMAR SELF-TEST: the solver is scored on
puzzles produced by the same generator grammar it decodes, and every panel is handed to it as a
pre-parsed attribute dict. The coverage audit rightly discounts both. This module closes the two
gaps at once and reports the TRUE number, whatever it is:

  (a) RENDER    : every panel of a 3x3 matrix (and all 8 answer candidates) is drawn as a small
                  RASTER GRID (11x11, colours 0..9, background 0) -- shapes stamped at slot
                  positions, exactly the kind of grid an ARC task hands the brain.
  (b) PERCEIVE  : the raster is parsed by the repo's connected-component perceiver -- arc_solver's
                  `components()` (the object parser the ARC pipeline uses) -- and the attribute
                  cells (shape, count, position mask, colour) are DERIVED from the pixels, never
                  handed in. A shape the perceiver cannot classify is an honest perception gap
                  (the attribute is omitted), not a guess.
  (c) ENCODE+SOLVE : the derived attributes become VSA role-filler panel codes (bind role to
                  filler, bundle -- cortex/vsa_reasoning's spike ops; `panel_code()` below builds
                  them and RpmSolver builds the same form internally for its ignition decision),
                  and the matrix is solved by the EXISTING brain-native inducers, unmodified:
                  reasoning/rpm_solver.RpmSolver first (NVSA menu: constant / progression /
                  arithmetic / distribute-three), reasoning/vsa_synthesize.VsaSynthesizer as the
                  out-of-grammar fallback when the menu gaps -- the same cascade RpmExpert uses.
  (d) SCORE     : on TEN hand-written matrices whose rules are HELD OUT from any generator --
                  constant-in-row, progression, distribute-three, union-across-columns,
                  XOR-across-columns, a per-row progression (c = 2b - a, outside the fixed menu),
                  and a shape-rotation matrix -- plus two controls that MUST sit near chance.

HONEST FRAMING (read before quoting any number)
-----------------------------------------------
* Chance is 1/8 = 12.5% (8 candidates). Accuracy is the fraction of matrices whose returned
  answer index equals the hand-written truth; `strict` additionally requires known=True (the
  solver committed, not a tie-break guess).
* The XOR matrix and the rotation matrix are DESIGNED to fail and are KEPT failing: the position
  code is an integer slot bitmask, and neither inducer owns a set-XOR op over it; the perceiver
  has no orientation attribute, so all rotated-L panels derive IDENTICAL cells and the answer set
  contains a deliberate derived-space twin AHEAD of the truth. Whatever a tie-break luckily
  returns, `known` stays False there -- asserted.
* The union matrix uses DISJOINT position sets, where union == bitmask addition, so it is
  honestly within the NVSA arithmetic rule over the derived posmask code -- stated, not hidden.
* The perceiver's shape catalogue is the same six stencils the renderer draws. That is the
  front-end's vocabulary, not a leak: attributes still travel pixel -> connected component ->
  classified cell, and the round-trip is asserted (and rotated stencils honestly fall outside it).
* This is NOT the RAVEN dataset and NOT full ARC. Ten matrices is a small hand-audited probe, not
  a population estimate; the value is that every rule is held out from any generator and both
  failure modes are kept on the books.

CONTROLS (both asserted near chance)
  * shuffled-answer : candidates are re-shuffled and the solve is scored against the STALE truth
    index -- a broken answer key. Any scoring pipeline that beats chance here is leaking.
  * scrambled-panel : the 8 context panels are shuffled before solving (row structure destroyed),
    scored against the true key. A solver that still scores well never used the rules.

Deterministic: per-instance random.Random(seed); codes crc32-seeded via vsa_reasoning.hv / gf.code.
ASCII-only output. Representation-only: imports no brainstem, routes no actuator.

  set PYTHONPATH=src & python -m snn_data_lake.reasoning.real_rpm_bench
"""

import random

from snn_data_lake.cortex.vsa_reasoning import bind, bundle, hv, sim
from snn_data_lake.reasoning.arc_solver import components
from snn_data_lake.reasoning.rpm_solver import RpmSolver
from snn_data_lake.reasoning.vsa_synthesize import VsaSynthesizer

CHANCE = 1.0 / 8.0  # 8 answer candidates
_PANEL = 11  # raster side: 3x3 slots of 3x3 px + 1 px gaps
_PITCH = 4  # slot pitch in px

# shape ids (ints so a shape progression can ride gf's ordinal line); render stencils in slot px.
DOT, HBAR, PLUS, TEE, LSH, SQR = 1, 2, 3, 4, 5, 6
_RENDER = {
    DOT: frozenset({(1, 1)}),
    HBAR: frozenset({(1, 0), (1, 1), (1, 2)}),
    PLUS: frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    TEE: frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    LSH: frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    SQR: frozenset({(r, c) for r in range(3) for c in range(3)}),
}


def _rot90(stencil):
    """Rotate a 3x3 stencil 90 degrees clockwise (used only to BUILD the rotation matrix; the
    perceiver has no rotation op, which is exactly the honest ceiling that matrix measures)."""
    return frozenset((c, 2 - r) for r, c in stencil)


_RENDER["L90"] = _rot90(_RENDER[LSH])
_RENDER["L180"] = _rot90(_RENDER["L90"])
_RENDER["L270"] = _rot90(_RENDER["L180"])


def _norm(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


# classification catalogue: normalized stencil -> shape id (the SIX canonical shapes only; the
# rotated-L stencils are deliberately absent -> an honest perception gap, not a guess).
_CLASS = {_norm(_RENDER[i]): i for i in (DOT, HBAR, PLUS, TEE, LSH, SQR)}


def render_panel(spec):
    """Draw one panel spec -- a tuple of (shape, colour, slot) objects -- as an 11x11 raster
    (tuple-of-tuples of ints, background 0). Slots 0..8 tile the panel 3x3 with 1 px gaps."""
    g = [[0] * _PANEL for _ in range(_PANEL)]
    for shape, colour, slot in spec:
        rr, cc = _PITCH * (slot // 3), _PITCH * (slot % 3)
        for dr, dc in _RENDER[shape]:
            g[rr + dr][cc + dc] = colour
    return tuple(tuple(row) for row in g)


def perceive_panel(grid):
    """DERIVE a panel's attribute cell from PIXELS via arc_solver's connected-component perceiver.
    Returns {'count', 'posmask'[, 'shape'][, 'colour' as 'color']}: count = number of components,
    posmask = bitmask of occupied slots, shape only when every component classifies to ONE known
    stencil, color only when the components agree. Unknown shape -> the attribute is honestly
    omitted (a perception gap), never invented."""
    objs = components(grid, conn=4, same_color=True, bg=0)
    mask = 0
    ids, cols = [], []
    for o in objs:
        r0, c0, _r1, _c1 = o["bbox"]
        mask |= 1 << ((r0 // _PITCH) * 3 + (c0 // _PITCH))
        ids.append(_CLASS.get(o["shape"]))
        cols.append(o["color"])
    cell = {"count": len(objs), "posmask": mask}
    if ids and all(i is not None for i in ids) and len(set(ids)) == 1:
        cell["shape"] = ids[0]
    if cols and len(set(cols)) == 1:
        cell["color"] = cols[0]
    return cell


def expected_cell(spec):
    """The attribute cell the perceiver SHOULD derive for a spec (computed from the spec alone,
    for the render->perceive round-trip assert -- never fed to the solver)."""
    cell = {"count": len(spec)}
    mask = 0
    for _shape, _colour, slot in spec:
        mask |= 1 << slot
    cell["posmask"] = mask
    ids = [s for s, _, _ in spec]
    if ids and all(isinstance(i, int) for i in ids) and len(set(ids)) == 1:
        cell["shape"] = ids[0]
    cols = [c for _, c, _ in spec]
    if cols and len(set(cols)) == 1:
        cell["color"] = cols[0]
    return cell


def panel_code(cell):
    """VSA role-filler PANEL CODE for a derived cell: bind each attribute ROLE to its value FILLER
    and bundle (spike superposition) -- cortex/vsa_reasoning ops. RpmSolver builds this same form
    internally for its workspace-ignition decision; this exposed copy is used by the self-check to
    verify the perceptual codes behave (identical percepts coincide, different ones do not)."""
    pairs = [bind(hv(f"RRB|role|{a}"), hv(f"RRB|val|{a}|{v!r}")) for a, v in sorted(cell.items())]
    return bundle(pairs)


# ====================================================================================================
# the held-out items -- WRITTEN BY HAND, no generator. Each: 8 context panels (row-major, (2,2)
# missing), 8 candidate panels, the truth index. Rules span the menu (constant / progression /
# distribute-three), the derived-encoding edge (union == disjoint bitmask add), out-of-menu
# synthesis (c = 2b - a), and two designed-to-fail probes (set-XOR, rotation).
# ====================================================================================================


def _dots(colour, slots):
    return tuple((DOT, colour, s) for s in slots)


def _s(shape, colour, slot):
    return ((shape, colour, slot),)


ITEMS = [
    {
        "name": "constant_rows_a",
        "family": "constant",
        "context": [_s(SQR, 2, 4)] * 3 + [_s(PLUS, 5, 4)] * 3 + [_s(HBAR, 7, 4)] * 2,
        "candidates": [
            _s(SQR, 7, 4),
            _s(HBAR, 5, 4),
            _s(PLUS, 7, 4),
            _s(HBAR, 7, 4),
            _s(HBAR, 7, 0),
            _s(DOT, 2, 4),
            _s(SQR, 2, 4),
            _s(HBAR, 9, 4),
        ],
        "answer": 3,
    },
    {
        "name": "constant_rows_b",
        "family": "constant",
        "context": [_s(DOT, 3, 0)] * 3 + [_s(LSH, 6, 4)] * 3 + [_s(TEE, 1, 8)] * 2,
        "candidates": [
            _s(TEE, 1, 4),
            _s(LSH, 1, 8),
            _s(TEE, 6, 8),
            _s(DOT, 1, 8),
            _s(TEE, 2, 8),
            _s(SQR, 3, 8),
            _s(TEE, 1, 8),
            _s(LSH, 6, 4),
        ],
        "answer": 6,
    },
    {
        "name": "progression_shape_colour",
        "family": "progression",
        "context": [
            _s(DOT, 2, 4),
            _s(HBAR, 3, 4),
            _s(PLUS, 4, 4),
            _s(HBAR, 5, 4),
            _s(PLUS, 6, 4),
            _s(TEE, 7, 4),
            _s(PLUS, 3, 4),
            _s(TEE, 4, 4),
        ],
        "candidates": [
            _s(TEE, 5, 4),
            _s(LSH, 5, 4),
            _s(LSH, 4, 4),
            _s(SQR, 6, 4),
            _s(PLUS, 5, 4),
            _s(LSH, 7, 4),
            _s(HBAR, 3, 4),
            _s(SQR, 5, 4),
        ],
        "answer": 1,
    },
    {
        "name": "progression_count",
        "family": "progression",
        "context": [
            _dots(8, (4,)),
            _dots(8, (0, 8)),
            _dots(8, (0, 4, 8)),
            _dots(8, (0, 8)),
            _dots(8, (0, 4, 8)),
            _dots(8, (0, 2, 6, 8)),
            _dots(8, (0, 4, 8)),
            _dots(8, (0, 2, 6, 8)),
        ],
        "candidates": [
            _dots(8, (0, 2, 6, 8)),
            _dots(8, (0, 4, 8)),
            _dots(8, (0, 1, 2, 6, 7, 8)),
            _dots(3, (0, 2, 6, 8)),
            _dots(8, (0, 8)),
            _dots(8, (0, 2, 4, 6, 8)),
            _dots(8, (4,)),
            _dots(5, (0, 1, 2, 6, 7, 8)),
        ],
        "answer": 5,
    },
    {
        "name": "distribute3_shape_colour",
        "family": "distribute3",
        "context": [
            _s(SQR, 2, 4),
            _s(PLUS, 4, 4),
            _s(DOT, 6, 4),
            _s(PLUS, 4, 4),
            _s(DOT, 6, 4),
            _s(SQR, 2, 4),
            _s(DOT, 6, 4),
            _s(SQR, 2, 4),
        ],
        "candidates": [
            _s(PLUS, 2, 4),
            _s(DOT, 4, 4),
            _s(SQR, 4, 4),
            _s(PLUS, 6, 4),
            _s(DOT, 2, 4),
            _s(SQR, 6, 4),
            _s(PLUS, 4, 0),
            _s(PLUS, 4, 4),
        ],
        "answer": 7,
    },
    {
        "name": "distribute3_shape_rowpos",
        "family": "distribute3",
        "context": [
            _s(HBAR, 9, 0),
            _s(TEE, 9, 0),
            _s(SQR, 9, 0),
            _s(SQR, 1, 4),
            _s(HBAR, 1, 4),
            _s(TEE, 1, 4),
            _s(TEE, 5, 8),
            _s(SQR, 5, 8),
        ],
        "candidates": [
            _s(TEE, 5, 8),
            _s(SQR, 5, 8),
            _s(HBAR, 5, 8),
            _s(HBAR, 1, 8),
            _s(HBAR, 5, 4),
            _s(DOT, 5, 8),
            _s(HBAR, 9, 8),
            _s(SQR, 1, 4),
        ],
        "answer": 2,
    },
    {
        "name": "union_disjoint_positions",
        "family": "union",
        "context": [
            _dots(6, (0,)),
            _dots(6, (4,)),
            _dots(6, (0, 4)),
            _dots(6, (1, 2)),
            _dots(6, (5,)),
            _dots(6, (1, 2, 5)),
            _dots(6, (3,)),
            _dots(6, (7, 8)),
        ],
        "candidates": [
            _dots(6, (7, 8)),
            _dots(2, (3, 7, 8)),
            _dots(6, (3, 5, 7, 8)),
            _dots(6, (1, 7, 8)),
            _dots(6, (3, 7, 8)),
            _dots(6, (8,)),
            _dots(9, (3, 7, 8)),
            _dots(6, (3, 6, 8)),
        ],
        "answer": 4,
    },
    {
        "name": "xor_overlapping_positions",
        "family": "xor",
        "context": [
            _dots(4, (0, 4)),
            _dots(4, (4, 8)),
            _dots(4, (0, 8)),
            _dots(4, (1, 4)),
            _dots(4, (4, 7)),
            _dots(4, (1, 7)),
            _dots(4, (2, 4)),
            _dots(4, (4, 6)),
        ],
        "candidates": [
            _dots(4, (0, 1, 2)),
            _dots(9, (5, 7)),
            _dots(4, (0, 6)),
            _dots(4, (4,)),
            _dots(4, (3, 5)),
            _dots(4, (0, 2, 6, 8)),
            _dots(4, (2, 6)),
            _dots(2, (1, 8)),
        ],
        "answer": 6,
    },
    {
        "name": "rowprog_2b_minus_a",
        "family": "rowprog",
        "context": [
            _s(SQR, 3, 4),
            _s(SQR, 4, 4),
            _s(SQR, 5, 4),
            _s(SQR, 2, 4),
            _s(SQR, 5, 4),
            _s(SQR, 8, 4),
            _s(SQR, 4, 4),
            _s(SQR, 6, 4),
        ],
        "candidates": [
            _s(SQR, 5, 4),
            _s(SQR, 6, 4),
            _s(SQR, 7, 4),
            _s(SQR, 8, 4),
            _s(SQR, 9, 4),
            _s(SQR, 2, 4),
            _s(SQR, 3, 4),
            _s(SQR, 1, 4),
        ],
        "answer": 3,
    },
    {
        "name": "rotation_l_shape",
        "family": "rotation",
        "context": [
            _s("L90", 7, 4),
            _s("L180", 7, 4),
            _s("L270", 7, 4),
            _s("L90", 7, 4),
            _s("L180", 7, 4),
            _s("L270", 7, 4),
            _s("L90", 7, 4),
            _s("L180", 7, 4),
        ],
        "candidates": [
            _s("L90", 7, 4),
            _s(SQR, 7, 4),
            _s("L270", 2, 4),
            _s("L180", 7, 0),
            _s("L270", 7, 4),
            _s(DOT, 7, 4),
            _s("L90", 3, 4),
            _dots(7, (0, 8)),
        ],
        "answer": 4,
    },
]


# ====================================================================================================
# the pipeline: raster -> components() -> derived cells -> RpmSolver, VsaSynthesizer fallback
# ====================================================================================================


def solve_matrix(ctx_cells, cand_cells, seed=0):
    """Solve one perceived matrix with the EXISTING inducers, unmodified: RpmSolver's NVSA menu
    first; when it gaps (known=False), VsaSynthesizer's out-of-grammar program search; when both
    gap, the honest fallback is RpmSolver's tie-break index with known=False (a guess, flagged as
    one). Returns {answer_idx, known, path, detail}."""
    rpm = RpmSolver(seed=seed).solve(list(ctx_cells), list(cand_cells))
    if rpm["known"]:
        return {
            "answer_idx": rpm["answer_idx"],
            "known": True,
            "path": "rpm",
            "detail": rpm["rule"],
        }
    syn = VsaSynthesizer(seed=seed).solve(list(ctx_cells), list(cand_cells))
    if syn["known"]:
        return {
            "answer_idx": syn["answer_idx"],
            "known": True,
            "path": "synth",
            "detail": syn["program"],
        }
    return {
        "answer_idx": rpm["answer_idx"],
        "known": False,
        "path": "fallback",
        "detail": rpm["rule"],
    }


def _perceived_items():
    """Render + perceive every item once: [(ctx_cells, cand_cells, truth_idx, item)]."""
    out = []
    for item in ITEMS:
        ctx = [perceive_panel(render_panel(p)) for p in item["context"]]
        cands = [perceive_panel(render_panel(p)) for p in item["candidates"]]
        out.append((ctx, cands, item["answer"], item))
    return out


def run_benchmark(seed=0):
    """Score the full held-out set. Returns {records, accuracy, strict_accuracy, chance, n_items}.
    accuracy       : fraction with answer_idx == truth (includes flagged tie-break guesses).
    strict_accuracy: fraction with answer_idx == truth AND known=True (the solver committed)."""
    records = []
    for ctx, cands, truth, item in _perceived_items():
        res = solve_matrix(ctx, cands, seed=seed)
        records.append(
            {
                "name": item["name"],
                "family": item["family"],
                "answer_idx": res["answer_idx"],
                "truth": truth,
                "correct": res["answer_idx"] == truth,
                "known": res["known"],
                "path": res["path"],
                "detail": res["detail"],
            }
        )
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    strict = sum(r["correct"] and r["known"] for r in records) / n
    return {
        "records": records,
        "accuracy": acc,
        "strict_accuracy": strict,
        "chance": CHANCE,
        "n_items": n,
    }


def shuffled_answer_control(repeats=8, seed=101):
    """BROKEN-KEY control: shuffle the candidate order, solve, score against the STALE index.
    Any pipeline that beats chance here is leaking the key. Must sit near 1/8."""
    rng = random.Random(seed)
    per = _perceived_items()
    hits = total = 0
    for _ in range(repeats):
        for ctx, cands, truth, _item in per:
            order = list(range(len(cands)))
            rng.shuffle(order)
            res = solve_matrix(ctx, [cands[i] for i in order])
            hits += res["answer_idx"] == truth
            total += 1
    return hits / total


def scrambled_panel_control(repeats=8, seed=202):
    """BROKEN-MATRIX control: shuffle the 8 context panels (row structure destroyed), solve,
    score against the TRUE key. A solver that still scores well never used the rules."""
    rng = random.Random(seed)
    per = _perceived_items()
    hits = total = 0
    for _ in range(repeats):
        for ctx, cands, truth, _item in per:
            sc = list(ctx)
            rng.shuffle(sc)
            res = solve_matrix(sc, cands)
            hits += res["answer_idx"] == truth
            total += 1
    return hits / total


# ---------------------------------------------------------------- self-check --------------------------


def main():
    print("=" * 78)
    print("REAL RPM BENCH -- rendered rasters, component perception, held-out hand-written rules")
    print("=" * 78)

    # (1) render -> perceive round-trip: every attribute cell is DERIVED from pixels and matches
    #     what the spec implies (incl. the rotated-L panels deriving NO shape -- perception gap).
    n_panels = n_match = 0
    rot_gap = True
    for item in ITEMS:
        for spec in list(item["context"]) + list(item["candidates"]):
            got = perceive_panel(render_panel(spec))
            n_panels += 1
            n_match += got == expected_cell(spec)
            if any(not isinstance(s, int) for s, _, _ in spec):
                rot_gap = rot_gap and ("shape" not in got)
    print(f"  perception round-trip: {n_match}/{n_panels} panels derive the expected cell")
    print(f"  rotated-L panels derive NO shape attribute (honest perception gap): {rot_gap}")

    # (2) VSA role-filler panel codes behave: identical percepts coincide, different ones do not.
    a = panel_code(perceive_panel(render_panel(_s(SQR, 2, 4))))
    b = panel_code(perceive_panel(render_panel(_s(SQR, 2, 4))))
    c = panel_code(perceive_panel(render_panel(_s(DOT, 7, 0))))
    code_ok = sim(a, b) == 1.0 and sim(a, c) < 0.6
    print(
        f"  role-filler panel codes: identical sim={sim(a, b):.2f}, different sim={sim(a, c):.2f}"
    )

    # (3) the benchmark itself.
    result = run_benchmark(seed=0)
    for r in result["records"]:
        print(
            "  [{}] {:<26} -> answer #{} (truth #{}) known={} path={}".format(
                "PASS" if r["correct"] else "FAIL",
                r["name"],
                r["answer_idx"],
                r["truth"],
                r["known"],
                r["path"],
            )
        )
    acc, strict = result["accuracy"], result["strict_accuracy"]
    print("-" * 78)
    print(
        f"  TRUE accuracy: {acc:.1%} ({int(round(acc * result['n_items']))}/{result['n_items']})"
        f"   strict (known=True and correct): {strict:.1%}   chance: {CHANCE:.1%}"
    )

    # (4) controls -- both must sit near chance (the broken key / broken matrix cannot be beaten).
    ctrl_shuf = shuffled_answer_control(repeats=8, seed=101)
    ctrl_scram = scrambled_panel_control(repeats=8, seed=202)
    print(
        f"  controls: shuffled-answer {ctrl_shuf:.1%}, scrambled-panel {ctrl_scram:.1%} "
        f"(chance {CHANCE:.1%}; both must be <= 30%)"
    )

    # (5) determinism: representative menu item and synthesis item solve identically twice.
    per = _perceived_items()
    det_ok = all(
        solve_matrix(per[i][0], per[i][1]) == solve_matrix(per[i][0], per[i][1]) for i in (0, 8)
    )
    print(f"  determinism: repeated solves identical = {det_ok}")

    # ---- asserts (the claims, exactly as measured -- failures are kept, not tuned away) ----
    assert n_match == n_panels, "every rendered panel must perceive back to its expected cell"
    assert rot_gap, "unclassifiable (rotated) shapes must derive NO shape attribute -- honest gap"
    assert code_ok, "role-filler panel codes must coincide for identical percepts only"
    by_name = {r["name"]: r for r in result["records"]}
    for name in (
        "constant_rows_a",
        "constant_rows_b",
        "progression_shape_colour",
        "progression_count",
        "distribute3_shape_colour",
        "distribute3_shape_rowpos",
        "union_disjoint_positions",
    ):
        r = by_name[name]
        assert r["correct"] and r["known"] and r["path"] == "rpm", (
            f"menu-rule item {name} must solve via RpmSolver from PERCEIVED cells"
        )
    r9 = by_name["rowprog_2b_minus_a"]
    assert r9["correct"] and r9["known"] and r9["path"] == "synth", (
        "the out-of-menu c=2b-a item must be solved by VsaSynthesizer (rpm menu gaps)"
    )
    # These two remain OUT OF THE SOLVER'S COMPETENCE and must never CLAIM to be in it. The
    # assertion is on the knowledge claim, not on the answer -- see the re-audit below, where one
    # of them started answering correctly while still (correctly) declining to claim it knew.
    for name in ("xor_overlapping_positions", "rotation_l_shape"):
        assert not by_name[name]["known"], (
            f"out-of-competence item {name} must NOT report known=True"
        )
    # RE-AUDITED 2026-08-06, which is exactly what the previous pin (0.8) demanded when it fired.
    # It fired because the number MOVED, and it moved UP -- which is the tripwire working, not a
    # regression. The audit, per item, HEAD vs now:
    #   xor_overlapping_positions   answer #2 (truth #6)  ->  answer #6   FIXED
    #   rotation_l_shape            answer #1 (truth #4)  ->  unchanged   still wrong
    # Attributable to the arbiter rank-key repair in thalamus/global_workspace.py (co-saturating
    # pools stopped being ranked by their source NAME). The fixed item resolves on the FALLBACK
    # path with known=False, so:
    #   TRUE accuracy  0.8 -> 0.9   (one more correct answer)
    #   STRICT         0.8 -> 0.8   (UNCHANGED -- no new knowledge claim was gained)
    # That gap is the honest reading and it is why both numbers are pinned separately below. The
    # solver did not learn to solve XOR; its fallback stopped being misdirected by a workspace head
    # that had been decided alphabetically. n=1, so this is an attributable improvement on one item
    # and NOT evidence of a general gain.
    assert abs(acc - 0.9) < 1e-9, (
        f"TRUE accuracy moved from the re-audited 9/10 -- re-audit, do not re-tune ({acc:.3f})"
    )
    assert abs(strict - 0.8) < 1e-9, (
        f"STRICT accuracy moved from the audited 8/10 -- a change here means the solver's KNOWLEDGE "
        f"CLAIMS moved, which is a bigger event than an answer changing ({strict:.3f})"
    )
    assert 0.0 <= ctrl_shuf <= 0.30, (
        f"shuffled-answer control must sit near chance: {ctrl_shuf:.3f}"
    )
    assert 0.0 <= ctrl_scram <= 0.30, (
        f"scrambled-panel control must sit near chance: {ctrl_scram:.3f}"
    )
    assert det_ok, "solves must be deterministic under a fixed seed"

    print("=" * 78)
    print(
        "self-check OK: real RPM bench -- rendered 11x11 rasters perceived by arc_solver "
        f"components() into derived cells, solved by RpmSolver+VsaSynthesizer: TRUE accuracy "
        f"{acc:.0%} ({strict:.0%} strict) on 10 held-out hand-written matrices vs 12.5% chance; "
        f"XOR and rotation both still CLAIM NOTHING (known=False), which is why STRICT did not "
        f"move; rotation still answers wrong, XOR now answers right on the fallback; controls at "
        f"{ctrl_shuf:.0%}/{ctrl_scram:.0%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
