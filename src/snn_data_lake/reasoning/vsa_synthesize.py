r"""OUT-OF-GRAMMAR RPM rule SYNTHESIZER -- fluid intelligence past the fixed NVSA rule list.

reasoning/rpm_solver.py scores ~98% on raven_bench but is developer-CHARGED: it SELECTS from a
pre-coded rule menu (constant / progression / arithmetic / distribute-three) over a hand-specified
attribute grammar. Chollet's developer-aware clause charges exactly that: the developer, not the
system, supplied the rules. This module removes the menu. VsaSynthesizer is given ONLY the example
rows of a matrix and SEARCHES a bounded COMPOSITIONAL space of VSA operation-programs --
compositions of bind / permute / permute-compose / bundle over the row's role slots -- for the
SMALLEST program that regenerates every example row (MDL, iterative deepening: gf.induce_program's
idiom lifted from integer expressions to spike-algebra programs). The winning program is EXECUTED
on the query row and the candidate matching its output is the answer. The rule is SYNTHESIZED at
test time, not selected: the four classic NVSA rules fall out as ordinary points of the program
space, and so do NOVEL compositions no fixed menu contains (col3 = col2 - col1; col3 = col1 + col2
+ 1; a per-row progression whose step DIFFERS row to row, i.e. col3 = 2*col2 - col1) -- the
self-check proves rpm_solver gaps on those very matrices while this module solves them.

THE OPERATION SPACE (the DSL; every op a spike op from cortex/vsa_reasoning + gf's ordinal line):
  terminals    A, B         the row's first / second cell spike code (role slots)
               R            the row INVARIANT: fold-bind of example row 0's three cell codes
                            (bind is self-inverse XOR, so R carries the row's multiset identity)
  perm(x, s)   literal spike-line shift, |s| <= shift_lim (gf's ordinal successor/step operator)
  bind(x, y)   spike-XOR coincidence gate (self-inverse: bind IS unbind)
  comp(x, y)   permute-compose: shift x by y's INDUCED anchor shift -- fractional-power exponent
               ADDITION, code(a) comp code(b) = code(a+b); the shift is read from y by exact spike
               coincidence against the ordinal codebook, so this stays a permute-family spike op
  icomp(x, y)  inverse permute-compose (exponent SUBTRACTION): code(a) icomp code(b) = code(a-b)
  bund(x, y)   spike superposition + majority WTA (in the vocabulary for completeness; exact-match
               consistency essentially never selects it -- stated, not hidden)
Programs are expression trees of depth <= 2 whose description length (node count; a perm costs its
shift magnitude) is the MDL yardstick. Enumeration is smallest-first with a deterministic
tie-break (the rendered program string), so the synthesizer commits to the SHORTEST consistent
program -- exactly induce_program's contract. Consistency is EXACT spike-code equality on every
example row: a 512-bit equation per row, so a spurious fit is ~2^-512 per free row. (For the full
R-unbind program the row-0 equation holds by construction -- R is built from row 0 -- so row 1
carries the real constraint there; stated, not hidden.)

HONEST CEILING (the Chollet clause, applied to ourselves). The op set, the depth bound, the shift
bound and the ordinal encoding are still developer-chosen, so the Gf credit is PARTIAL: this is
synthesis WITHIN a designed algebra, not open-ended concept invention. What it genuinely adds over
rpm_solver: the rule for a matrix is COMPOSED at test time from primitive spike ops, so matrices
whose rule is NOT in any pre-coded menu are solved -- and the self-check shows rpm_solver returning
its honest gap on those same matrices. What it still CANNOT do: rules needing depth > 2 or shifts
beyond the bound, value MULTIPLICATION (no op maps code(a),code(b) -> code(a*b)), conditional /
branching rules, multi-object panels, and any perception (it consumes parsed attribute cells).
Two example rows also genuinely UNDER-DETERMINE some matrices (e.g. identical distribute-three
example rows admit a shorter shift program that mis-generalises); MDL then commits to the shorter
reading and the candidate match reports known=False when the prediction fails to single out one
answer. Honest gap (known=False) whenever no program in the space fits -- never a bluffed choice.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.reasoning.vsa_synthesize
"""

# ruff: noqa: UP031  -- %-format is the hand-aligned house style of every sibling in reasoning/
# (which the repo excludes from `ruff check .`); kept for consistency, not reflowed to f-strings.
import random
import sys

from snn_data_lake.cortex.vsa_reasoning import bind, bundle, hv, permute
from snn_data_lake.reasoning.gf import code

MAX_SIZE = 6  # MDL cap: total description length of a program (nodes; perm costs |shift|)
MAX_DEPTH = 2  # compositional depth of the expression tree
SHIFT_LIM = 3  # literal perm shifts searched: -3..-1, +1..+3
_SHIFT_MAX = 64  # ordinal-line extent the comp/icomp shift induction reads exactly

_TERMS = (("A",), ("B",), ("R",))
_BINOPS = ("bind", "bund", "comp", "icomp")
_COMM = ("bind", "bund")  # commutative -> canonical operand order, no self-pairs
_SPACE_CACHE = {}  # (max_size, max_depth, shift_lim) -> ordered programs
_SHIFT_BOOK = {}  # tuple(code(s)) -> s, built lazily (0.._SHIFT_MAX)


def _render(expr):
    """Deterministic ASCII rendering of a program -- also the MDL tie-break key."""
    op = expr[0]
    if op in ("A", "B", "R"):
        return op
    if op == "perm":
        return "perm(%s, %+d)" % (_render(expr[2]), expr[1])
    return "%s(%s, %s)" % (op, _render(expr[1]), _render(expr[2]))


def _shift_of(vec):
    """Read a spike code's EXACT position on gf's ordinal line (which anchor permute it is), by
    codebook lookup over code(0.._SHIFT_MAX). None if the code is not on the line -- comp/icomp
    then honestly fail instead of guessing a nearest shift."""
    if not _SHIFT_BOOK:
        for s in range(_SHIFT_MAX + 1):
            _SHIFT_BOOK[tuple(code(s))] = s
    return _SHIFT_BOOK.get(tuple(vec))


def _program_space(max_size, max_depth, shift_lim):
    """Enumerate every program of the DSL up to the bounds, ordered smallest-first (MDL) with the
    rendered string as the deterministic tie-break -- gf._expressions_by_size, for spike programs.
    Returns [(size, render, expr)]; cached (a pure function of the bounds)."""
    key = (max_size, max_depth, shift_lim)
    if key in _SPACE_CACHE:
        return _SPACE_CACHE[key]
    size, depth = {}, {}
    for t in _TERMS:
        size[t], depth[t] = 1, 0
    layers = {0: list(_TERMS)}
    for d in range(1, max_depth + 1):
        built = []
        younger = [e for dd in range(d) for e in layers[dd]]
        for sub in layers[d - 1]:
            if sub[0] == "perm":
                continue  # nested perms collapse to one shift -- skip duplicates
            for s in range(-shift_lim, shift_lim + 1):
                if s == 0 or size[sub] + abs(s) > max_size:
                    continue
                e = ("perm", s, sub)
                size[e], depth[e] = size[sub] + abs(s), d
                built.append(e)
        for op in _BINOPS:
            for left in younger:
                for right in younger:
                    if depth[left] != d - 1 and depth[right] != d - 1:
                        continue  # at least one child on the frontier (no duplicates)
                    if op in _COMM and (left == right or _render(left) > _render(right)):
                        continue  # canonical order; bind(x,x)/bund(x,x) are degenerate
                    if size[left] + size[right] + 1 > max_size:
                        continue
                    e = (op, left, right)
                    size[e], depth[e] = size[left] + size[right] + 1, d
                    built.append(e)
        layers[d] = built
    every = [e for lay in layers.values() for e in lay]
    ordered = sorted(((size[e], _render(e), e) for e in every), key=lambda t: (t[0], t[1]))
    _SPACE_CACHE[key] = ordered
    return ordered


def _eval(expr, env, memo):
    """Execute a program on one row environment {A, B, R}. Memoised per row so shared subtrees are
    evaluated once across the whole enumeration. None = the op is undefined on these codes (e.g.
    comp on a code that is not on the ordinal line) -- an honest miss, never a guess."""
    if expr in memo:
        return memo[expr]
    op = expr[0]
    if op in ("A", "B", "R"):
        out = env[op]
    elif op == "perm":
        sub = _eval(expr[2], env, memo)
        out = None if sub is None else permute(sub, expr[1])
    else:
        lv = _eval(expr[1], env, memo)
        rv = _eval(expr[2], env, memo)
        if lv is None or rv is None:
            out = None
        elif op == "bind":
            out = bind(lv, rv)
        elif op == "bund":
            out = bundle([lv, rv])
        else:  # comp / icomp: permute-compose along the ordinal line
            s = _shift_of(rv)
            out = None if s is None else permute(lv, s if op == "comp" else -s)
    memo[expr] = out
    return out


def _enc(attr, v):
    """Spike code for one attribute value: integers ride gf's ordinal line (enables perm/comp/
    icomp), anything else gets a categorical code (bind/R programs still apply)."""
    if isinstance(v, bool) or not isinstance(v, int):
        return hv("SYNTH|%s|%r" % (attr, v))
    return code(v)


class VsaSynthesizer:
    """Test-time RPM rule SYNTHESIZER over the spiking VSA -- no fixed rule list.

    solve(context_cells, candidates) -> {answer_idx, confidence, program, known}
      context_cells : the 8 known cells row-major (rows 0,1 complete; (2,0),(2,1) the knowns);
                      each cell a dict {attribute: value} (ints ride the ordinal spike line).
      candidates    : >= 2 candidate cells (same dict shape); one completes cell (2,2).
    Per attribute the two complete rows are the EXAMPLES: the smallest program in the bounded
    compositional space (see module docstring) consistent with BOTH rows -- exact spike-code
    equality -- is synthesized and executed on the query row; candidates are matched against the
    predicted codes. known=True only when one candidate uniquely matches every synthesized
    attribute. Honest gap (known=False, program='gap:...') when nothing in the space fits.

    Deterministic: crc32-seeded codes, deterministic enumeration order, no wall-clock, no global
    RNG (seed kept for interface parity with RpmSolver). Guarded: never raises out of solve().
    """

    def __init__(self, seed=0, max_size=MAX_SIZE, max_depth=MAX_DEPTH, shift_lim=SHIFT_LIM):
        self.seed = seed
        self.max_size = max_size
        self.max_depth = max_depth
        self.shift_lim = shift_lim

    # -- synthesis (the Gf move: example rows -> program, smallest-first) -----------------------
    def _synthesize(self, ex_envs, ex_targets, q_env):
        """Smallest program consistent with every example row, executed on the query row.
        Returns (render, size, predicted_code) or None -- no program in the space fits."""
        space = _program_space(self.max_size, self.max_depth, self.shift_lim)
        memos = [{} for _ in ex_envs]
        q_memo = {}
        for sz, txt, expr in space:
            fit = True
            for env, memo, want in zip(ex_envs, memos, ex_targets):
                if _eval(expr, env, memo) != want:
                    fit = False
                    break
            if not fit:
                continue
            pred = _eval(expr, q_env, q_memo)
            if pred is None:
                continue  # fits the examples but is undefined on the query codes
            return txt, sz, pred
        return None

    # -- solve ----------------------------------------------------------------------------------
    def solve(self, context_cells, candidates):
        """Solve one RPM. Returns {answer_idx, confidence, program, known}. Guarded: any failure
        returns an honest gap (known=False, answer_idx=0), never raises."""
        try:
            return self._solve(context_cells, candidates)
        except Exception as exc:  # honest gap, never a crash
            return {
                "answer_idx": 0,
                "confidence": 0.0,
                "known": False,
                "program": "gap:%s" % type(exc).__name__,
            }

    def _solve(self, context_cells, candidates):
        if len(context_cells) != 8 or len(candidates) < 2:
            return {"answer_idx": 0, "confidence": 0.0, "known": False, "program": "gap:malformed"}
        rows = [context_cells[0:3], context_cells[3:6], context_cells[6:8]]
        attrs = sorted({a for c in list(context_cells) + list(candidates) for a in c})
        programs, predicted = {}, {}
        for a in attrs:
            if any(a not in c for c in context_cells):
                continue
            ex = [tuple(_enc(a, rows[r][i][a]) for i in range(3)) for r in (0, 1)]
            rvec = bind(bind(ex[0][0], ex[0][1]), ex[0][2])  # row invariant (fold-bind row 0)
            ex_envs = [{"A": e[0], "B": e[1], "R": rvec} for e in ex]
            ex_targets = [e[2] for e in ex]
            q_env = {"A": _enc(a, rows[2][0][a]), "B": _enc(a, rows[2][1][a]), "R": rvec}
            got = self._synthesize(ex_envs, ex_targets, q_env)
            if got is not None:
                programs[a], _sz, predicted[a] = got[0], got[1], got[2]
        if not predicted:  # nothing in the program space fits any attribute
            return {"answer_idx": 0, "confidence": 0.0, "known": False, "program": "gap:no-program"}

        n = len(predicted)
        matches = [
            sum(1 for a2, p in predicted.items() if a2 in cand and _enc(a2, cand[a2]) == p)
            for cand in candidates
        ]
        best = max(matches)
        winners = [i for i, m in enumerate(matches) if m == best]
        answer_idx = winners[0]
        second = max([m for i, m in enumerate(matches) if i != answer_idx] or [0])
        confidence = round((best - second) / float(n), 3)
        known = len(winners) == 1 and best == n
        prog_txt = "; ".join("%s=%s" % (a2, programs[a2]) for a2 in sorted(programs))
        return {
            "answer_idx": answer_idx,
            "confidence": confidence,
            "known": bool(known),
            "program": prog_txt if known else ("low-confidence (%s)" % prog_txt),
        }


class IdentityProgramSolver:
    """FAILING control: no synthesis at all -- FREEZE the single program C = A (copy the row's
    first cell) for every attribute. On a mixed-rule split this must collapse to ~chance; if it
    scored well the task set (not the synthesizer) would be broken."""

    def solve(self, context_cells, candidates):
        if len(context_cells) != 8 or len(candidates) < 2:
            return {"answer_idx": 0, "confidence": 0.0, "known": False, "program": "gap:malformed"}
        frozen = context_cells[6]  # row 2, col 0 -- program A's prediction per attribute
        matches = [sum(1 for a, v in frozen.items() if cand.get(a) == v) for cand in candidates]
        return {
            "answer_idx": matches.index(max(matches)),
            "confidence": 0.0,
            "known": False,
            "program": "frozen:A",
        }


# ---------------------------------------------------------------- self-check --------------------------
# Verified demo grids (same value grid replicated over type/size/color; per-attr candidates are
# rng-perturbed). Each grid was checked BY HAND so the intended rule is the unique smallest
# consistent program: every smaller-or-equal competitor dies on example row 1.

_ATTRS = ("type", "size", "color")

_CLASSIC = {  # the fixed menu's own regimes -- must fall out as SYNTHESIZED programs, and
    # rpm_solver must ALSO solve them (proving they are in-grammar for the fixed menu)
    "constant": ([[3, 3, 3], [5, 5, 5]], [2, 2], 2),
    "progression": ([[1, 3, 5], [2, 4, 6]], [3, 5], 7),
    "arithmetic": ([[2, 3, 5], [4, 1, 5]], [1, 4], 5),
    "distribute3": ([[1, 2, 3], [2, 3, 1]], [3, 1], 2),
}
_NOVEL = {  # HELD-OUT regimes: NOT expressible by rpm_solver's fixed menu (it must gap) --
    # out-of-grammar synthesis is proven on exactly these matrices
    "revsub c=b-a": ([[1, 4, 3], [2, 7, 5]], [1, 7], 6),
    "sumplus1 c=a+b+1": ([[1, 2, 4], [3, 1, 5]], [2, 3], 6),
    "rowprog c=2b-a": ([[2, 3, 4], [1, 4, 7]], [3, 5], 7),
}
_GARBAGE = ([[1, 5, 2], [4, 2, 7]], [3, 6])  # no program in the space fits -> must gap honestly


def _grid_task(rows, knowns, answer_val, rng):
    """Replicate one verified value grid across the three attributes; build 8 candidates with
    raven_bench's own distractor builder (reused, not re-invented). Returns (context, cands, idx)."""
    from snn_data_lake.reasoning.raven_bench import _candidates

    context = [{a: rows[r][c] for a in _ATTRS} for r in range(2) for c in range(3)]
    context += [{a: knowns[c] for a in _ATTRS} for c in range(2)]
    answer = {a: answer_val for a in _ATTRS}
    cands, idx = _candidates(answer, rng)
    return context, cands, idx


def main():
    from snn_data_lake.reasoning.raven_bench import RandomSolver, _candidates, score
    from snn_data_lake.reasoning.rpm_solver import RpmSolver

    print("=" * 78)
    print("OUT-OF-GRAMMAR VSA RULE SYNTHESIZER -- programs composed at test time, no rule menu")
    print("=" * 78)
    synth = VsaSynthesizer(seed=0)
    rpm = RpmSolver(seed=0)
    rng = random.Random(0)
    space = _program_space(MAX_SIZE, MAX_DEPTH, SHIFT_LIM)
    print(
        "program space: %d programs (depth<=%d, size<=%d, |shift|<=%d) over "
        "{A,B,R} x {perm,bind,comp,icomp,bund}" % (len(space), MAX_DEPTH, MAX_SIZE, SHIFT_LIM)
    )

    # (a) the fixed menu's own regimes, re-derived as SYNTHESIZED programs (rpm solves them too,
    #     proving these matrices are in-grammar for the fixed menu).
    print("-" * 78)
    print("(a) classic regimes -> synthesized programs (rpm_solver must ALSO solve = in-grammar)")
    classic_ok = []
    for kind, (rows, knowns, ans) in _CLASSIC.items():
        ctx, cands, idx = _grid_task(rows, knowns, ans, rng)
        res = synth.solve(ctx, cands)
        rr = rpm.solve(ctx, cands)
        ok = res["known"] and res["answer_idx"] == idx and rr["known"] and rr["answer_idx"] == idx
        classic_ok.append(ok)
        prog = res["program"].split(";")[0].split("=", 1)[-1].strip()
        print(
            "  [%s] %-12s -> #%d (truth #%d) known=%s  program: C = %s"
            % ("PASS" if ok else "FAIL", kind, res["answer_idx"], idx, res["known"], prog)
        )

    # (b) HELD-OUT regimes: rpm_solver must return its honest gap (out-of-grammar for the fixed
    #     menu) while the synthesizer composes a program and solves.
    print("-" * 78)
    print("(b) held-out regimes -> rpm_solver GAPS (out-of-grammar), synthesizer composes + solves")
    novel_ok = []
    novel_example = None
    for kind, (rows, knowns, ans) in _NOVEL.items():
        ctx, cands, idx = _grid_task(rows, knowns, ans, rng)
        res = synth.solve(ctx, cands)
        rr = rpm.solve(ctx, cands)
        ok = res["known"] and res["answer_idx"] == idx and not rr["known"]
        novel_ok.append(ok)
        prog = res["program"].split(";")[0].split("=", 1)[-1].strip()
        if kind.startswith("rowprog"):
            novel_example = prog
        print(
            "  [%s] %-17s -> synth #%d (truth #%d) known=%s | rpm known=%s  program: C = %s"
            % (
                "PASS" if ok else "FAIL",
                kind,
                res["answer_idx"],
                idx,
                res["known"],
                rr["known"],
                prog,
            )
        )

    # (bench) raven_bench's own synthetic split + its un-riggable failing controls.
    print("-" * 78)
    limit = 40
    r = score(VsaSynthesizer(seed=0), "synthetic", limit=limit)
    ctrl_r = score(RandomSolver(seed=0), "synthetic", limit=limit)
    ctrl_i = score(IdentityProgramSolver(), "synthetic", limit=limit)
    print("(c) raven_bench synthetic split (%d tasks, 8-choice, chance 12.5%%):" % limit)
    print(
        "  VsaSynthesizer   %3d/%-3d  %5.1f%%  (known %.1f%%)"
        % (r["solved"], r["n"], 100 * r["accuracy"], 100 * r["known_frac"])
    )
    print(
        "  random-ctrl      %3d/%-3d  %5.1f%%"
        % (ctrl_r["solved"], ctrl_r["n"], 100 * ctrl_r["accuracy"])
    )
    print(
        "  frozen-A-ctrl    %3d/%-3d  %5.1f%%"
        % (ctrl_i["solved"], ctrl_i["n"], 100 * ctrl_i["accuracy"])
    )

    # (gap) a grid no program in the space regenerates -> honest gap, never a bluffed choice.
    g_rows, g_knowns = _GARBAGE
    g_ctx = [{a: g_rows[r][c] for a in _ATTRS} for r in range(2) for c in range(3)]
    g_ctx += [{a: g_knowns[c] for a in _ATTRS} for c in range(2)]
    g_cands, _gidx = _candidates({a: 4 for a in _ATTRS}, random.Random(3))
    gap = synth.solve(g_ctx, g_cands)
    gap_ok = not gap["known"]
    print("-" * 78)
    print(
        "  [%s] garbage grid -> known=%s program=%r (must be an honest gap)"
        % ("PASS" if gap_ok else "FAIL", gap["known"], gap["program"])
    )

    # determinism: two fresh instances reproduce the identical result dict.
    ctx_d, cands_d, _i = _grid_task(*_CLASSIC["distribute3"], rng=random.Random(7))
    det_ok = VsaSynthesizer(seed=0).solve(ctx_d, cands_d) == VsaSynthesizer(seed=0).solve(
        ctx_d, cands_d
    )
    print(
        "  [%s] determinism  -> same task reproduces the identical result dict"
        % ("PASS" if det_ok else "FAIL")
    )

    print("-" * 78)
    print("honest ceiling: the op set / depth / shift bounds are developer-chosen, so Gf credit is")
    print("PARTIAL (synthesis WITHIN a designed algebra). CAN synthesize: any depth<=2 composition")
    print("of bind/perm/comp/icomp/bund -- incl. b-a, a+b+1, 2b-a, row-multiset unbinds. CANNOT:")
    print(
        "value multiplication, conditionals, depth>2 rules, shifts>|%d|, perception. Two example"
        % SHIFT_LIM
    )
    print("rows can under-determine a matrix; MDL then backs the shortest reading (known=False if")
    print("it fails to single out a candidate). No rule menu anywhere: delete a classic rule from")
    print("rpm_solver and nothing here changes -- the program space never enumerates rule NAMES.")

    example = novel_example or "(none)"
    assert all(classic_ok), "all four classic regimes must be re-derived as synthesized programs"
    assert all(novel_ok), (
        "every held-out regime must be solved by synthesis while rpm_solver "
        "gaps on it -- that asymmetry IS the out-of-grammar proof"
    )
    assert r["accuracy"] >= 0.60, (
        "the synthesizer must clear chance (12.5%%) by a wide margin, got %.1f%%"
        % (100 * r["accuracy"])
    )
    assert ctrl_r["accuracy"] <= 0.30 and ctrl_i["accuracy"] <= 0.30, (
        "both controls must genuinely fail near chance (%.1f%% / %.1f%%)"
        % (100 * ctrl_r["accuracy"], 100 * ctrl_i["accuracy"])
    )
    assert r["accuracy"] > ctrl_r["accuracy"] + 0.30, "the margin over control must discriminate"
    assert gap_ok, "a grid no program regenerates must return an HONEST gap (known=False)"
    assert det_ok, "the synthesizer must be deterministic"

    print("=" * 78)
    print("self-check OK: rules SYNTHESIZED as spike-op programs (no fixed menu) -- classics")
    print("re-derived, held-out regimes solved where rpm_solver honestly gaps (example synthesized")
    print(
        "out-of-grammar program: C = %s, i.e. col3 = 2*col2 - col1), %.1f%% on raven_bench"
        % (example, 100 * r["accuracy"])
    )
    print(
        "synthetic vs %.1f%% random / %.1f%% frozen-program controls; honest gap on a no-rule "
        "grid; deterministic." % (100 * ctrl_r["accuracy"], 100 * ctrl_i["accuracy"])
    )
    return 0


if __name__ == "__main__":
    main()
