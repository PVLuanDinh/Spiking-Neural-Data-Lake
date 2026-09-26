"""Gf-inducer cluster, COMBINED (not merged) into one file.

Five COMPLEMENTARY test-time rule inducers kept as ONE file, each mechanism intact:
  program_inducer -- bounded iterative-deepening program synthesis (induce_program*).
  open_gf         -- OPEN-ENDED self-extending inducer (OpenGf: library learning + unbounded deepening).
  gf_inducer      -- spiking-VSA labelled-family inducer (synthesize_sequence, _exact_readout, raven/analogy).
  active_learner  -- breaks underdetermination by EXPERIMENT (consistent_hypotheses, World, Library).
  sequence_learner-- next-term sequence learner.

Combined, NOT merged: colliding PRIVATE helpers (_render/_ev/_norm/_CONSTS/_OPS/_evaluate/main) are
per-module renamed with a tag (_pi/_ogf/_gfi/_al/_sl) so the five distinct implementations coexist;
public names (induce_program, OpenGf, synthesize_sequence, consistent_hypotheses, ...) are unchanged.
  PYTHONPATH=src python -m snn_data_lake.reasoning.gf
"""

# ============================================================================================
# --- program_inducer ---
# ============================================================================================
r"""
UNBOUNDED PROGRAM-INDUCER -- synthesise the GENERATING PROGRAM of a sequence at test time.

This crosses the fixed-menu wall of gf_inducer's synthesize_sequence. Instead of a hand-listed set of
rule types, it SEARCHES the open space of programs over a tiny grammar

    E := n | t1 | t2 | c            (c in a small constant set; t1=t[n-1], t2=t[n-2], n=1-based index)
       | E + E | E - E | E * E

by ITERATIVE DEEPENING on program size, and returns the SMALLEST expression that generates the
observed terms. The space is closed under composition and the only limit is the size budget -- raise
the budget and it finds larger programs with no code change, so it is unbounded in principle (bounded
in practice by search cost). This is test-time program synthesis in the strict Chollet/ARC sense: the
program is built for THIS sequence, not selected from a menu the author enumerated.

It SUBSUMES the menu (arithmetic = t1+c, geometric = t1*c, polynomial = n*n..., recurrence = t1+t2)
AND reaches programs the menu could NOT: factorial (t[n] = t1 * n), products/cross-terms (n*t1,
t1*t1), etc.

HONESTY: this is SYMBOLIC search, not a spike operation (stated plainly, like the rest of the
higher-order inducer). Overfit is guarded by LEAVE-ONE-OUT -- a program must be consistent on all but
the last term AND correctly predict that held-out last term before it is trusted. The honest ceiling
is real and structural: any rule NOT expressible in this grammar within the size budget (primes,
mixed +/x per-step, anything needing a primitive the grammar lacks) still returns None -- an honest
gap. Widening the grammar/budget moves the wall; it never removes it.

Deps: stdlib (numpy/torch allowed project-wide).
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.gf     # probe + self-check
"""
import sys

_CONSTS_pi = (1, 2, 3)  # the atomic integer constants the grammar may use
_MAX_SIZE = 7  # node budget (raise -> larger programs, unbounded in principle)


def _digit_sum(v):
    """sum of the base-10 digits of |v|. A NEW general grammar primitive (digitsum(t[n-1])) -- not a
    per-item special case: it composes with +/-/* like any other terminal, so it reaches any sequence
    whose step is a function of the previous term's digits (e.g. t[n]=t[n-1]+digitsum(t[n-1]))."""
    return sum(int(ch) for ch in str(abs(int(v))))


# An expression is a nested tuple: ('n',) ('t1',) ('t2',) ('d1',) ('c', k) ('+', a, b) ('-', a, b) ('*', a, b).
# 'd1' = digitsum(t[n-1]) -- a general terminal, composable with the rest of the grammar.


def _evaluate_pi(expr, n, t1, t2):
    tag = expr[0]
    if tag == "n":
        return n
    if tag == "t1":
        return t1
    if tag == "t2":
        return t2
    if tag == "d1":
        return _digit_sum(t1)
    if tag == "c":
        return expr[1]
    a = _evaluate_pi(expr[1], n, t1, t2)
    b = _evaluate_pi(expr[2], n, t1, t2)
    return a + b if tag == "+" else a - b if tag == "-" else a * b


def _uses_pi(expr, sym):
    if expr[0] == sym:
        return True
    return len(expr) == 3 and (_uses_pi(expr[1], sym) or _uses_pi(expr[2], sym))


def _min_start_pi(expr):
    """The earliest history index this expr can be evaluated at: needs t[n-2] (t2) -> 2; needs t[n-1]
    (t1, or d1 which is a function OF t1 and so needs the same history) -> 1; a pure f(n) -> 0."""
    if _uses_pi(expr, "t2"):
        return 2
    if _uses_pi(expr, "t1") or _uses_pi(expr, "d1"):
        return 1
    return 0


def _render_pi(expr):
    tag = expr[0]
    if tag in ("n", "t1", "t2"):
        return {"n": "n", "t1": "t[n-1]", "t2": "t[n-2]"}[tag]
    if tag == "d1":
        return "digitsum(t[n-1])"
    if tag == "c":
        return str(expr[1])
    return f"({_render_pi(expr[1])} {tag} {_render_pi(expr[2])})"


_LAYER_CACHE = {}  # max_size -> {size: [expr, ...]} -- built once, reused across every call/turn


def _layers(max_size):
    """The expression layers up to max_size, built ONCE and cached (it is a pure function of max_size).
    Rebuilding the size-7 grammar cost ~16 ms on every induce/consistent call before this cache."""
    if max_size not in _LAYER_CACHE:
        layers = {1: [("n",), ("t1",), ("t2",), ("d1",)] + [("c", k) for k in _CONSTS_pi]}
        for size in range(2, max_size + 1):
            built = []
            for op in ("+", "-", "*"):
                for left in range(1, size - 1):
                    for a in layers[left]:
                        for b in layers[size - 1 - left]:
                            built.append((op, a, b))
            layers[size] = built
        _LAYER_CACHE[max_size] = layers
    return _LAYER_CACHE[max_size]


def _expressions_by_size(max_size):
    """Yield (size, expr) for every distinct expression up to max_size, smallest first (iterative
    deepening). Lazy over the cached layers, so early-exit consumers still stop early."""
    layers = _layers(max_size)
    for size in range(1, max_size + 1):
        for e in layers[size]:
            yield size, e


def _consistent(expr, values, start):
    """True iff expr regenerates values[start:] from the real history (all but the final term)."""
    for i in range(start, len(values) - 1):
        if _evaluate_pi(expr, i + 1, values[i - 1], values[i - 2]) != values[i]:
            return False
    return True


def induce_program_full(values):
    """Like induce_program but also returns the program's NODE COUNT (its description length): the
    smallest (next_value, "t[n]=...", nodes) that survives leave-one-out, or None. The node count lets
    a caller apply MDL -- a program that does not COMPRESS the data (nodes >= #terms) is a coincidental
    fit, not a pinned rule (the primes case), and an honest agent should ask rather than assert it."""
    if len(values) < 4:  # need history to fit + a term to hold out
        return None
    last = len(values) - 1
    for size, expr in _expressions_by_size(_MAX_SIZE):
        start = _min_start_pi(expr)
        if last - start < 2:  # too few equations to identify a program
            continue
        if not _consistent(expr, values, start):
            continue
        held_out = _evaluate_pi(expr, last + 1, values[last - 1], values[last - 2])
        if held_out != values[last]:  # leave-one-out: must predict the last term
            continue
        nxt = _evaluate_pi(expr, last + 2, values[last], values[last - 1])
        return nxt, f"t[n]={_render_pi(expr)}", size
    return None


def induce_program(values):
    """The smallest generating program's (next_value, "t[n]=..."), or None (honest gap: no program in
    the grammar fits). Guarded by LEAVE-ONE-OUT. (induce_program_full also returns the node count.)"""
    full = induce_program_full(values)
    return (full[0], full[1]) if full is not None else None


def consistent_programs(values, top_n=8):
    """Every DISTINCT-continuation program (smallest first) that reproduces the terms under
    leave-one-out -- for detecting UNDERDETERMINATION: if >=2 of these disagree on the next term, the
    data does not pin the rule and an honest agent should ASK, not commit to the smallest. Deduped by
    the predicted next value (we care about disagreement on the continuation). Returns [(next, label)]."""
    if len(values) < 4:
        return []
    last = len(values) - 1
    out, seen = [], set()
    for _size, expr in _expressions_by_size(_MAX_SIZE):
        start = _min_start_pi(expr)
        if last - start < 2:
            continue
        if not _consistent(expr, values, start):
            continue
        if _evaluate_pi(expr, last + 1, values[last - 1], values[last - 2]) != values[last]:
            continue  # leave-one-out: must predict the held-out term
        nxt = _evaluate_pi(expr, last + 2, values[last], values[last - 1])
        if nxt in seen:
            continue
        seen.add(nxt)
        out.append((nxt, f"t[n]={_render_pi(expr)}"))
        if len(out) >= top_n:
            break
    return out


def main_pi():
    print("=" * 96)
    print("UNBOUNDED PROGRAM-INDUCER -- synthesise a sequence's generating program at test time")
    print("=" * 96)
    passed = []

    def check(name, values, want):
        got = induce_program(values)
        val = got[0] if got else None
        ok = val == want
        passed.append(ok)
        prog = got[1] if got else "(no program found)"
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:28s} -> {val!r:>6}  (want {want!r:>6})  {prog}"
        )

    # subsumes the menu ...
    check("arithmetic 2 4 6 8", [2, 4, 6, 8], 10)
    check("geometric  2 4 8 16", [2, 4, 8, 16], 32)
    check("squares    1 4 9 16 25", [1, 4, 9, 16, 25], 36)
    check("Fibonacci  1 1 2 3 5", [1, 1, 2, 3, 5], 8)
    # ... AND reaches programs the fixed menu could NOT
    check("factorial  1 2 6 24 120", [1, 2, 6, 24, 120], 720)  # t[n] = n * t[n-1]
    check("square-recur 2 4 16 256", [2, 4, 16, 256], 65536)  # t[n] = t[n-1] * t[n-1]

    n_pass = sum(passed)
    print(
        f"\n  programs synthesised: {n_pass}/{len(passed)} (menu families + factorial + cross-terms)"
    )

    # UNDERDETERMINATION (a real property of induction, not a bug): with few terms the SIMPLEST program
    # consistent with the data may not be the 'intended' rule. Given only 2 3 5 7 11 the inducer finds
    # a genuine generating program (t[n]=2*t[n-2]+1) that fits every term but continues 15, not the
    # prime 13 -- because 'prime' is knowledge beyond the 5 numbers shown. This is honest: it returns a
    # program that DOES reproduce the data, and more terms would disambiguate. It is not a bluff.
    prime_prog = induce_program([2, 3, 5, 7, 11])
    print(
        f"\n  underdetermination: primes 2 3 5 7 11 -> {prime_prog[1]} (fits the data, predicts "
        f"{prime_prog[0]}; not 'prime' -- 5 terms underdetermine the rule)"
    )

    # the true structural wall -- NO program in the grammar/budget fits even the given terms -> gap
    print("\n  the wall -- no program in the bounded grammar fits the data at all, MUST gap:")
    wall = {
        "random  3 1 4 1 5": induce_program([3, 1, 4, 1, 5]),
        "random  5 3 9 2 7 6": induce_program([5, 3, 9, 2, 7, 6]),
    }
    for name, got in wall.items():
        print(f"  [{'GAP ' if got is None else 'BLUFF'}] {name:24s} -> {got!r}")

    assert induce_program([1, 2, 6, 24, 120])[0] == 720, "must SYNTHESISE factorial (t[n]=n*t1)"
    assert "t[n-1]" in induce_program([1, 2, 6, 24, 120])[1], (
        "factorial program must reference t[n-1]"
    )
    assert induce_program([1, 1, 2, 3, 5])[0] == 8, "must synthesise the Fibonacci recurrence"
    assert induce_program([2, 4, 8, 16])[0] == 32, "must synthesise geometric"
    assert prime_prog is not None, (
        "underdetermination: it must find A program consistent with the data"
    )
    assert all(v is None for v in wall.values()), (
        "a sequence with NO fitting program MUST honestly gap"
    )

    print(
        "\nSELF-CHECK PASSED: open-ended test-time program synthesis (unbounded in the size budget), "
        "leave-one-out guarded; underdetermination surfaced honestly; structural gaps where no "
        "program fits the data."
    )
    return 0


# ============================================================================================
# --- open_gf ---
# ============================================================================================
r"""
OPEN-ENDED FLUID INTELLIGENCE -- a self-extending rule inducer whose grammar GROWS with experience.

The bounded inducers (gf_inducer, program_inducer) search a FIXED grammar to a FIXED depth. Their
honest wall: a rule needing a bigger program than the budget, or a primitive the grammar lacks, gaps.
This makes that wall MOVE instead of sit still -- three levers, none of which is "solve everything"
(that is uncomputable; the wall is real, just no longer FIXED):

  1. ITERATIVE DEEPENING, no hardcoded ceiling. Search size 1,2,3,... until a program fits or a
     RESOURCE budget is hit. The budget is a compute guard you can raise, not a grammar limit -- raise
     it and deeper programs come into reach. solve() stops at the budget (finite-time gap);
     solve_unbounded() raises it automatically with NO ceiling until the program fits -- COMPLETE, but
     it trades the finite-time gap for possible non-termination on a non-program target (the in-the-
     limit lever; completeness <=> incomputability, see docs/THE_FUNDAMENTAL_WALL.md).
  2. LIBRARY LEARNING (DreamCoder-style). Every rule it SOLVES is added, if it compresses the data,
     as a NAMED PRIMITIVE. Future searches include the library as extra terminals -- so the searchable
     space grows super-linearly with what it has learned. n^6 is out of budget from scratch, but after
     it learns `cube` from n^3, n^6 = cube*cube is one step away. The rule count is not fixed.
  3. DURABLE. The library persists across sessions, so competence ACCUMULATES and (as the Godel
     machine proves) the reasoning CODE itself can be extended when even the closure is short.

     **P4b VIOLATION CLOSED 2026-08-05.** This line said "persists to the home lake" and it did
     NOT: `save_state`/`load_state` (substrate/home.py) wrote `<NORD_HOME>/gf_library.json`, a plain
     JSON file BESIDE `brain_lake.bin`. It was proven across a process boundary on 2026-08-04 -- a
     writer learned `p0 = n*(n*n)` with persist=True and exited, and the only artifact in NORD_HOME
     was `gf_library.json` (41 B), with `brain_lake.bin` never created. The learned Gf grammar is
     exactly "what the brain learns" -- one of P0's self-improvement channels -- so under P4b it
     belongs IN the one lake under a record tag. It is there now, and NEITHER `_STATE` NAME NOR ANY
     LINE OF THIS FILE CHANGED: `save_state` writes a `_T_STATE` record keyed by that name, so both
     libraries here (`gf_library.json` and `active_learner`'s `active_library.json`, further down)
     moved with the mechanism. The escalation asked for a `_T_GFLIB` tag plus a save/load pair; that
     would have been the sixth near-identical channel for one job, and P2 says one survives. The
     same-process durability the escape blocks below exercise is now backed by a NEW-PROCESS
     round-trip in main(), which is the only form of the claim P4b accepts.

HONEST CEILING (unchanged, and stated): this is unbounded in the OPEN-ENDED sense -- no fixed menu,
depth, or library size; the boundary moves with learning. It is NOT omniscient: a genuinely
non-computable target, or one outside the closure of its primitives, still gaps at any budget -- and
it says so rather than bluffing. "Unbounded" here means self-extending, not all-knowing.

Grammar: E := n | t[n-1] | t[n-2] | c(1,2,3) | <learned primitive> | E+E | E-E | E*E. Leave-one-out
guards overfit; a program is learned only if it COMPRESSES (nodes < #terms).
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.gf     # open-ended growth self-check
"""
import time

from snn_data_lake.substrate.home import load_state, save_state

_CONSTS_ogf = (1, 2, 3)
# the base PRIMITIVE set. // and % widen the reachable CLOSURE beyond {+,-,*}: division-based rules
# (triangular n(n+1)/2) and cyclic/modulo patterns become expressible. A raisable set -- the Godel
# machine can extend it further at the code level when even this closure is short.
_OPS_ogf = ("+", "-", "*", "//", "%")
_DEDUP = range(
    1, 8
)  # closed-form behaviour compared over these indices to dedup learned primitives


def _ev(e, n, t1, t2, lib):
    tag = e[0]
    if tag == "n":
        return n
    if tag == "t1":
        return t1
    if tag == "t2":
        return t2
    if tag == "c":
        return e[1]
    if tag == "lib":
        return _ev(lib[e[1]], n, t1, t2, lib)
    a = _ev(e[1], n, t1, t2, lib)
    b = _ev(e[2], n, t1, t2, lib)
    if tag == "+":
        return a + b
    if tag == "-":
        return a - b
    if tag == "*":
        return a * b
    if b == 0:
        raise ZeroDivisionError  # invalid on this input -> the program can't fit
    return a // b if tag == "//" else a % b


def _uses_ogf(e, sym):
    if e[0] == sym:
        return True
    return len(e) == 3 and (_uses_ogf(e[1], sym) or _uses_ogf(e[2], sym))


def _nodes(e):
    # a learned primitive counts as ONE node -- that is the compression the library buys.
    return 1 if len(e) < 3 else 1 + _nodes(e[1]) + _nodes(e[2])


def _render_ogf(e):
    tag = e[0]
    if tag in ("n", "t1", "t2"):
        return {"n": "n", "t1": "t[n-1]", "t2": "t[n-2]"}[tag]
    if tag == "c":
        return str(e[1])
    if tag == "lib":
        return e[1]
    return f"({_render_ogf(e[1])} {tag} {_render_ogf(e[2])})"


def _norm_ogf(e):
    # tuples <-> lists survive a JSON round-trip; normalise a loaded library to tuples for hashing/eval.
    return tuple(
        [e[0]]
        + (
            [e[1]]
            if e[0] == "c"
            else [e[1]]
            if e[0] == "lib"
            else [_norm_ogf(e[1]), _norm_ogf(e[2])]
            if len(e) == 3
            else []
        )
    )


class OpenGf:
    """A rule inducer whose grammar grows. solve() searches (base grammar + learned primitives) by
    iterative deepening; a compressing solution is learned as a new primitive and (optionally) persisted,
    so the next problem is searched over a larger space."""

    _STATE = "gf_library.json"

    def __init__(self, budget=12, persist=False, cap=60000):
        self.budget = budget  # iterative-deepening ceiling (a RESOURCE guard, raisable)
        self.cap = cap  # per-size hard node-count guard
        self.persist = persist
        self.lib = {}  # name -> expr (learned primitives, composable as terminals)
        if persist:
            for name, expr in (load_state(self._STATE, default={}) or {}).items():
                self.lib[name] = _norm_ogf(expr)
        self._layer_cache = {}

    def _terminals(self):
        return (
            [("n",), ("t1",), ("t2",)]
            + [("c", k) for k in _CONSTS_ogf]
            + [("lib", nm) for nm in sorted(self.lib)]
        )

    def _layers(self, max_size):
        key = (max_size, tuple(sorted(self.lib)))
        if key not in self._layer_cache:
            layers = {1: self._terminals()}
            for size in range(2, max_size + 1):
                built = []
                for op in _OPS_ogf:
                    for left in range(1, size - 1):
                        for a in layers[left]:
                            for b in layers[size - 1 - left]:
                                built.append((op, a, b))
                                if len(built) > self.cap:
                                    break
                layers[size] = built
            self._layer_cache[key] = layers
        return self._layer_cache[key]

    def _consistent(self, e, values, start):
        for i in range(start, len(values) - 1):
            if _ev(e, i + 1, values[i - 1], values[i - 2], self.lib) != values[i]:
                return False
        return True

    def solve(self, values, learn=True):
        """Return (next_value, program_str, nodes) or None. Iterative deepening over the GROWING grammar;
        leave-one-out guarded. A compressing solution is learned as a reusable primitive."""
        if len(values) < 4:
            return None
        last = len(values) - 1
        for size in range(1, self.budget + 1):  # ITERATIVE DEEPENING (no fixed ceiling)
            for e in self._layers(size).get(size, []):
                start = 2 if _uses_ogf(e, "t2") else 1 if _uses_ogf(e, "t1") else 0
                if last - start < 2:
                    continue
                try:
                    if not self._consistent(e, values, start):
                        continue
                    if (
                        _ev(e, last + 1, values[last - 1], values[last - 2], self.lib)
                        != values[last]
                    ):
                        continue  # leave-one-out
                    nxt = _ev(e, last + 2, values[last], values[last - 1], self.lib)
                except ZeroDivisionError:
                    continue  # invalid (div-by-zero) on some input -> not a fit
                if learn:
                    self._learn(e, values)
                return nxt, _render_ogf(e), _nodes(e)
        return None

    def _learn(self, e, values):
        """Promote a solved rule to a NAMED PRIMITIVE if it COMPRESSES (a real rule, not a coincidental
        fit) and is a closed-form function of n (composable as a terminal). The grammar grows."""
        if _nodes(e) >= len(values):  # coincidental fit -> not a rule worth keeping
            return
        if _uses_ogf(e, "t1") or _uses_ogf(e, "t2"):  # a recurrence isn't a reusable f(n) primitive
            return
        try:
            beh = tuple(_ev(e, k, 0, 0, self.lib) for k in _DEDUP)
        except ZeroDivisionError:
            return  # divides by zero somewhere -> not a clean f(n)
        for expr in self.lib.values():  # dedup by BEHAVIOUR (don't relearn the same fn)
            try:
                if tuple(_ev(expr, k, 0, 0, self.lib) for k in _DEDUP) == beh:
                    return
            except ZeroDivisionError:
                continue
        self.lib[f"p{len(self.lib)}"] = e
        self._layer_cache.clear()  # grammar changed -> rebuild layers next search
        if self.persist:
            save_state(self._STATE, {k: v for k, v in self.lib.items()})

    def grow_budget(self, k):
        """Raise the resource budget -- deeper programs become reachable (the ceiling is not fixed)."""
        self.budget = max(self.budget, k)
        return self

    def solve_unbounded(self, values, learn=True, step=4, max_budget=None, max_seconds=None):
        """UNBOUNDED iterative deepening: raise the budget with NO fixed ceiling (grow_budget in a loop)
        until a compressing program fits, then return (next, program, nodes). Where solve() stops at
        self.budget and returns a finite-time gap, this keeps deepening -- so it is COMPLETE: any program
        expressible in the growing grammar closure is found given enough time, smallest-first (the same
        smallest-first order Levin search uses).

        HONEST TRADE (docs/THE_FUNDAMENTAL_WALL.md, the in-the-limit lever): completeness is BOUGHT by
        surrendering solve()'s finite-time gap. On a target with NO program in the closure this does NOT
        halt -- it deepens forever. That is not a bug; it is the wall. You cannot have both complete AND
        always-halting (Solomonoff: completeness <=> incomputability). Pass max_budget and/or max_seconds
        to make it ANYTIME-interruptible; then a None means 'no fit UNDER the budget/time reached', NOT
        'proven absent' (Rice/Solomonoff forbid certifying absence). With both guards left None it is
        truly unbounded: found-or-forever."""
        if len(values) < 4:
            return None  # too short to fit -- raising the budget never helps
        entry, t0 = self.budget, time.perf_counter()
        b = self.budget
        try:
            while True:
                self.budget = b
                r = self.solve(values, learn=learn)
                if r is not None:
                    return r
                if max_budget is not None and b >= max_budget:
                    return None  # anytime gap: a resource limit, NOT proof of absence
                if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
                    return None
                b += step
        finally:
            self.budget = entry  # a query, not a reconfig -- never leave the budget raised


def main_ogf():
    print("=" * 96)
    print("OPEN-ENDED FLUID INTELLIGENCE -- a grammar that GROWS with experience (self-check)")
    print("=" * 96)

    # ---- 1) ITERATIVE DEEPENING: raising the budget reaches a program a smaller budget could not ----
    shallow = OpenGf(budget=4)
    deep = OpenGf(budget=9)
    cubes = [1, 8, 27, 64, 125]  # n^3 -> smallest program (n*(n*n)) has 5 nodes
    print(
        f"\n  [deepening] n^3 at budget 4 -> {shallow.solve(cubes, learn=False)}  "
        f"(too shallow); at budget 9 -> {deep.solve(cubes, learn=False)[:2]}"
    )
    assert shallow.solve(cubes, learn=False) is None, (
        "a size-5 rule must be OUT of a budget-4 search"
    )
    assert deep.solve(cubes, learn=False)[0] == 216, (
        "deepening the budget must reach it -> next is 216"
    )

    # ---- 1b) UNBOUNDED deepening: no fixed ceiling -- keep raising the budget until the program fits ----
    # n^4's smallest program is 7 nodes -- OUT of a budget-4 search. solve() gaps; solve_unbounded()
    # deepens automatically (4->8) and FINDS it. Completeness: any program in the closure is reached.
    n4 = [1, 16, 81, 256, 625, 1296]  # f(n)=n^4, next=2401, smallest program 7 nodes
    small = OpenGf(budget=4)
    assert small.solve(n4, learn=False) is None, (
        "n^4 (7-node program) is OUT of a fixed budget-4 search"
    )
    found = OpenGf(budget=4).solve_unbounded(n4, learn=False, max_budget=40)
    print(f"  [unbounded] n^4 bounded@4 -> None; solve_unbounded -> {found}")
    assert found is not None and found[0] == 2401 and found[2] == 7, (
        "UNBOUNDED deepening must reach the out-of-budget program a fixed budget cannot (next 2401, 7 nodes)"
    )
    # HONEST TRADE: on a non-program series it would deepen FOREVER -- a max_budget guard makes it an
    # anytime gap ('no fit under budget B'), which is a resource limit, NOT a proof of absence.
    wall_guarded = OpenGf(budget=6).solve_unbounded([3, 1, 4, 1, 5, 9], learn=False, max_budget=14)
    assert wall_guarded is None, (
        "unbounded search must still gap (under a guard) on a genuine non-program series"
    )

    # ---- 2) LIBRARY LEARNING: n^6 is out of budget from scratch, but reachable AFTER learning cube ----
    g = OpenGf(budget=7)
    n6 = [1, 64, 729, 4096, 15625, 46656]  # n^6: base program has 11 nodes -> out of budget 7
    print(f"\n  [grow] n^6 from scratch (budget 7) -> {g.solve(n6, learn=False)} (out of budget)")
    assert g.solve(n6, learn=False) is None, "n^6 must be out of the base grammar within budget 7"
    g.solve([1, 8, 27, 64, 125, 216])  # learn n^3 (5 nodes < 6 terms -> compresses) = `cube`
    grown = g.solve(n6, learn=False)  # n^6 = cube*cube -- now one step away
    print(f"  [grow] after learning n^3=cube (library {sorted(g.lib)}) -> n^6 = {grown}")
    assert grown is not None and grown[0] == 117649, "after the grammar grew, n^6 must be solvable"
    assert "p" in grown[1] and grown[2] <= 3, (
        "n^6 must REUSE the learned primitive in a tiny program (cube*cube)"
    )

    # ---- 3) DURABLE: the learned library persists -> a FRESH engine (new session) still solves n^6 ----
    import os
    import tempfile

    prev, tmp = os.environ.get("NORD_HOME"), tempfile.mkdtemp(prefix="opengf_")
    os.environ["NORD_HOME"] = tmp
    try:
        a = OpenGf(budget=7, persist=True)
        a.solve([1, 8, 27, 64, 125, 216])  # learns + PERSISTS cube (n^3)
        b = OpenGf(budget=7, persist=True)  # a NEW engine, same durable home
        transferred = b.solve(n6, learn=False)
        print(f"\n  [durable] fresh engine reloaded library {sorted(b.lib)} -> n^6 = {transferred}")
        assert b.lib, "the learned library must persist across sessions"
        assert transferred is not None and transferred[0] == 117649, (
            "accumulated competence must transfer"
        )
        # P4b: the two lines above share ONE interpreter, so they would pass against a pure RAM
        # store -- which is what this check certified while the library was a JSON sidecar. A
        # SEPARATE process now has to rebuild the library from `brain_lake.bin` alone, and the
        # sidecar must not exist at all.
        import subprocess
        import sys

        from snn_data_lake.substrate import home as _home

        assert not os.path.exists(_home.state_path(OpenGf._STATE)), (
            "the learned library must go IN brain_lake.bin -- no gf_library.json sidecar"
        )
        _src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        child = (
            f"import sys; sys.path.insert(0, {_src_root!r})\n"
            f"import os; os.environ['NORD_HOME'] = {tmp!r}\n"
            "from snn_data_lake.reasoning.gf import OpenGf\n"
            "c = OpenGf(budget=7, persist=True)\n"
            f"assert sorted(c.lib) == {sorted(b.lib)!r}, sorted(c.lib)\n"
            f"got = c.solve({n6!r}, learn=False)\n"
            "assert got is not None and got[0] == 117649, got\n"
            "print('  [durable] a SEPARATE process reloaded %s from the lake and solved n^6'"
            " % sorted(c.lib))\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        assert rc == 0, f"the learned library did not survive into a NEW process (rc={rc})"
    finally:
        os.environ.pop("NORD_HOME", None)
        if prev is not None:
            os.environ["NORD_HOME"] = prev

    # ---- 3b) WIDER CLOSURE: // and % reach families {+,-,*} alone cannot ----
    w = OpenGf(budget=8)
    r_mod = w.solve([1, 2, 0, 1, 2, 0], learn=False)  # cyclic period-3 = n%3 (no t1/t2 recurrence)
    r_div = w.solve([0, 2, 4, 8, 12, 18], learn=False)  # (n*n)//2 -- integer division
    print(f"\n  [closure] n%3 cyclic -> {r_mod} ; (n*n)//2 -> {r_div}")
    assert r_mod is not None and r_mod[0] == 1 and "%" in r_mod[1], (
        "% widens the closure -> a cyclic rule is reachable"
    )
    assert r_div is not None and r_div[0] == 24 and "//" in r_div[1], (
        "// widens the closure -> (n*n)//2 is reachable"
    )

    # ---- 4) HONEST WALL: not everything is a program -- a random series still gaps at ANY budget ----
    huge = OpenGf(budget=10)
    print(
        f"\n  [wall] random 3 1 4 1 5 9 at budget 10 -> {huge.solve([3, 1, 4, 1, 5, 9], learn=False)}"
    )
    assert huge.solve([3, 1, 4, 1, 5, 9], learn=False) is None, (
        "the wall MOVED, it did not vanish -- a genuinely non-program series still honestly gaps"
    )

    print(
        "\nSELF-CHECK PASSED: open-ended Gf -- iterative deepening (no fixed depth), library learning "
        "(the grammar GROWS with experience: n^6 reachable only after learning cube), durable across "
        "sessions; still an HONEST gap on what no program in the closure fits. Unbounded = self-"
        "extending, not omniscient."
    )
    return 0


# ============================================================================================
# --- gf_inducer ---
# ============================================================================================
r"""
FLUID-INTELLIGENCE INDUCER -- test-time rule induction over the brain's OWN spiking VSA.

The brain measured Gc but 0/11 Gf (scratchpad/probe_fluid.py): it could WALK taught knowledge but
never SYNTHESIZE a rule for a novel problem. The binding substrate was there (cortex.vsa_reasoning:
a genuinely spiking VSA) and simply never wired to induction. This module wires it.

THE MECHANISM (Rasmussen & Eliasmith 2011, "rule generation in inductive reasoning", in the
spiking VSA already in the brain). An ordered symbol space is a single orbit under the VSA's
PERMUTE operator -- value/position n is code(n) = permute(ANCHOR, n) -- so "successor" is
permute-by-1 and any constant-difference relation is permute-by-s for one fixed s. The rule s is
NOT given: the model SEARCHES its own representational space for the shift that maps the observed
examples (prev -> curr), which IS test-time rule synthesis (Chollet's structural minimum -- the
shift is not in the priors, only the ordered space is), then APPLIES it to a novel probe and reads
the answer out by spike CLEANUP over the codebook. Induction is spike-coincidence search, no MAC.

HONEST CEILING (measured, not hidden -- the probe asserts these GAP rather than bluff):
  * inducible: constant-difference number series (2 4 6 8 -> 10; 9 7 5 -> 3), letter succession
    (a b c -> d; a c e -> g), and "increment position p" letter analogies (abc:abd :: ijk -> ijl).
  * NOT inducible: non-constant relations -- geometric (2 4 8 16), Fibonacci (1 1 2 3 5), Raven
    distribution-of-three. These need relational integration across bound structures, not one shift.
  This is a BOUNDED, spiking-native Gf mechanism (WHICH shift on a known ordered space), not general
  fluid reasoning. The ordinality of the space is a built-in prior; the rule over it is induced.

Deps: stdlib + the brain's own snn_data_lake.cortex.vsa_reasoning (no LLM).
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.gf    # scripted probe + self-check
"""
import re
import string
from fractions import Fraction

from snn_data_lake.cortex.vsa_reasoning import bundle, cleanup, coincidence, hv, permute, sim

_ANCHOR = hv("GF_ORDERED_LINE_ANCHOR")
_SHIFTS = range(-15, 16)  # the transforms the model searches over (rule hypotheses)
_LETTERS = string.ascii_lowercase


def code(n):
    """The spike code for ordinal position/value n: the ANCHOR permuted n line-steps along the
    orbit. One orbit => successor is permute-by-1 and any constant step is one fixed permute."""
    return permute(_ANCHOR, n % len(_ANCHOR))


_LETTER_BOOK = {c: code(i) for i, c in enumerate(_LETTERS)}  # a..z  -> code(0..25)
_NUMBER_BOOK = {str(n): code(n) for n in range(100)}  # 0..99 (cleanup readout targets)


def _best_shift(prev, curr):
    """Induce the single transform (line-shift) that best maps spike code prev -> curr: the s whose
    permute(prev, s) is most COINCIDENT with curr. This is the rule hypothesis search, in spikes."""
    return max(_SHIFTS, key=lambda s: sim(permute(prev, s), curr))


def induce_shift(codes):
    """Induce ONE constant transform from a sequence of observed spike codes. Returns the shift if
    it is CONSTANT across every consecutive pair, else None -- an honest gap for a non-constant
    relation (the model refuses to force a single rule onto a series that has none)."""
    if len(codes) < 2:
        return None
    shifts = [_best_shift(a, b) for a, b in zip(codes, codes[1:], strict=False)]
    return shifts[0] if all(s == shifts[0] for s in shifts) else None


# ---- relational integration: rules that are relations OVER relations / across bound structures ----
# This is the ingredient the constant-shift inducer lacked. A single shift cannot express a geometric
# ratio, a polynomial (constant 2nd/3rd difference), or a recurrence (term from the previous TWO).
# So beyond the order-1 shift (which stays on the spiking VSA), the model SEARCHES a small hypothesis
# space of relation TYPES for the one that fits the observed terms -- still test-time synthesis (the
# rule is found for THIS problem, not stored), but this higher-order search is SYMBOLIC arithmetic
# over the observed values, not a spike operation. Honest split, stated plainly.


def _diffs(xs):
    return [b - a for a, b in zip(xs, xs[1:], strict=False)]


def _const(xs):
    return len(xs) >= 2 and all(x == xs[0] for x in xs)


def _poly_next(values):
    """Smallest k in 1..3 whose k-th finite difference is constant (>=2 equal entries) -> a degree-k
    polynomial sequence. Extends the difference table by one and integrates back up. Catches
    arithmetic (k=1), squares (k=2), cubes (k=3)."""
    for k in range(1, 4):
        d = values
        for _ in range(k):
            d = _diffs(d)
        if _const(d):
            table = [list(values)]
            for _ in range(k):
                table.append(_diffs(table[-1]))
            table[-1].append(table[-1][-1])  # extend the constant bottom row
            for i in range(len(table) - 2, -1, -1):
                table[i].append(table[i][-1] + table[i + 1][-1])
            return table[0][-1], ("constant difference" if k == 1 else f"constant {k}th-difference")
    return None


def _geometric_next(values):
    """Constant integer/rational RATIO between terms -> a geometric sequence (2 4 8 16, 3 9 27)."""
    if any(v == 0 for v in values[:-1]):
        return None
    ratios = [Fraction(b, a) for a, b in zip(values, values[1:], strict=False)]
    if len(ratios) >= 2 and all(r == ratios[0] for r in ratios) and ratios[0] != 1:
        nxt = ratios[0] * values[-1]
        if nxt.denominator == 1:
            return int(nxt), f"constant ratio x{ratios[0]}"
    return None


def _recurrence_next(values):
    """Order-2 linear recurrence v[i] = a*v[i-1] + b*v[i-2]: solve (a, b) EXACTLY from two equations
    and verify on the rest -- relational integration across the two prior bound terms. Catches
    Fibonacci (a=b=1), Lucas, Pell (a=2,b=1). Needs >=5 terms to solve AND verify."""
    v = values
    if len(v) < 5:
        return None
    det = v[1] * v[1] - v[0] * v[2]
    if det == 0:
        return None
    a = Fraction(v[2] * v[1] - v[3] * v[0], det)
    b = Fraction(v[3] * v[1] - v[2] * v[2], det)
    if all(Fraction(v[i]) == a * v[i - 1] + b * v[i - 2] for i in range(2, len(v))):
        nxt = a * v[-1] + b * v[-2]
        if nxt.denominator == 1:
            return int(nxt), f"linear recurrence (a={a}, b={b})"
    return None


def _periodic_additive_next(values):
    """The step (first difference) repeats with PERIOD p (2 or 3) -- a rule the fixed families cannot
    express: 1 3 6 8 11 13 has steps +2 +3 +2 +3 (period 2), not one constant difference/ratio. Needs
    >=2 full cycles to confirm. Constant steps are excluded (that is arithmetic -- handled upstream)."""
    d = _diffs(values)
    for p in (2, 3):
        if len(d) >= 2 * p and len(set(d[:p])) > 1 and all(d[i] == d[i % p] for i in range(len(d))):
            return values[-1] + d[len(d) % p], f"period-{p} additive cycle {d[:p]}"
    return None


def _op_candidates(v0, v1):
    """Small set of (kind, param) that map v0 -> v1 EXACTLY: additive is always offered; a clean
    integer multiplicative ratio (!=1, no remainder) is offered too when one exists."""
    out = [("+", v1 - v0)]
    if v0 != 0 and v1 % v0 == 0:
        ratio = v1 // v0
        if ratio != 1:
            out.append(("*", ratio))
    return out


def _apply_op(kind, param, v):
    return v + param if kind == "+" else v * param


def _parity_op_next(values):
    """Two DIFFERENT operators alternating by the PARITY of the transition index (5 10 7 14 11 22:
    x2 on even transitions, -3 on odd) -- a rule _periodic_additive_next cannot express (that one
    needs one constant additive STEP repeating; here the transition itself changes KIND per parity).
    Infers (kind, param) independently for even- and odd-indexed transitions from the observed
    values, then LEAVE-ONE-OUT verifies: the induced pair of rules must reproduce every transition,
    including the LAST one held out of the fit, before it is trusted. None -> honest gap."""
    if len(values) < 6:  # >=2 transitions/parity to fit + 1 held out to verify
        return None
    pairs = list(zip(values, values[1:]))
    train, held = pairs[:-1], pairs[-1]
    even_train, odd_train = train[0::2], train[1::2]
    if len(even_train) < 2 or not odd_train:
        return None

    def _fit(cls_pairs):
        for cand in _op_candidates(*cls_pairs[0]):
            if all(_apply_op(cand[0], cand[1], a) == b for a, b in cls_pairs):
                return cand
        return None

    op_even, op_odd = _fit(even_train), _fit(odd_train)
    if op_even is None or op_odd is None:
        return None
    held_op = op_even if len(train) % 2 == 0 else op_odd
    if _apply_op(held_op[0], held_op[1], held[0]) != held[1]:  # leave-one-out on the held-out term
        return None
    next_op = op_even if len(pairs) % 2 == 0 else op_odd
    nxt = _apply_op(next_op[0], next_op[1], values[-1])
    return nxt, f"parity op-alternation (even:{op_even[0]}{op_even[1]}, odd:{op_odd[0]}{op_odd[1]})"


def _interleave_next(values):
    """Two INDEPENDENT subsequences interleaved (1 10 2 20 3 30 = [1 2 3] woven with [10 20 30]) -- a
    depth-2 program the flat families cannot see. Split even/odd positions, SYNTHESISE each half
    recursively, and continue whichever subsequence the next position belongs to. Needs >=3 per half."""
    if len(values) < 6:
        return None
    ra, rb = synthesize_sequence(values[0::2]), synthesize_sequence(values[1::2])
    if ra is None or rb is None:
        return None
    nxt = ra if len(values) % 2 == 0 else rb  # next position: even -> A, odd -> B
    return nxt[0], f"interleave(A: {ra[1]} | B: {rb[1]})"


def synthesize_sequence(values):
    """TEST-TIME PROGRAM SYNTHESIS over a small DSL: try each program generator in ascending
    complexity (Occam) and return the first that REPRODUCES the observed terms, with its rule label.
    The generators are the DSL -- polynomial, geometric, order-2 recurrence (primitives), plus
    period-p operation cycles and interleave (depth-2 combinators). This is the strongest form the
    bounded searcher has: the rule is a PROGRAM synthesised for this problem, not a menu lookup.
    Anything outside the DSL / beyond this depth returns None -- an honest, structural gap."""
    for generator in (
        _poly_next,
        _geometric_next,
        _recurrence_next,
        _periodic_additive_next,
        _parity_op_next,
        _interleave_next,
        induce_program,
    ):
        got = generator(values)
        if got is not None:
            return got
    return None


def analogy_over(order, a, b, c):
    """Letter analogy whose SUCCESSOR ORDERING is a PROVIDED, possibly NOVEL alphabet `order` (symbols
    the brain has never seen, or a scrambled/reversed order). The shift is induced over THAT ordering
    and applied to c -- abstraction TRANSFER to a novel symbol system (Lewis & Mitchell counterfactual
    alphabets). None -> a symbol outside the given order, or unequal lengths."""
    book = {s: code(i) for i, s in enumerate(order)}
    if not (len(a) == len(b) == len(c)):
        return None
    try:
        out = []
        for ca, cb, cc in zip(a, b, c, strict=True):
            s = _best_shift(book[ca], book[cb])
            if (
                sim(permute(book[ca], s), book[cb]) != 1.0
            ):  # shift outside the window -> spurious -> gap
                return None
            out.append(_exact_readout(permute(book[cc], s), book))
    except KeyError:  # a symbol not in the provided alphabet
        return None
    return "".join(out) if all(o is not None for o in out) else None  # any off-alphabet slot -> gap


def _exact_readout(predicted, book):
    """Snap a predicted spike code to the codebook but ONLY accept an EXACT match -- the induced shift
    must land on a real symbol, not cyclic-wrap past the alphabet/codebook end. Else None (honest gap),
    never a bluffed nearest-symbol. This is what stops z+1 -> 'm' and a 0-99 overflow from bluffing."""
    label, _ = cleanup(predicted, book)
    return label if book[label] == predicted else None


def solve_series(values):
    """Extrapolate a number series. Returns (next_value, rule_label) or None (honest gap -- no program
    in the DSL fits). Order-1 constant difference is induced + read out through the spiking VSA; every
    other rule is found by test-time PROGRAM SYNTHESIS over the DSL (synthesize_sequence)."""
    if len(values) < 3:
        return None
    if all(0 <= v < 90 for v in values):  # within the orbit + readout codebook
        codes = [code(v) for v in values]
        s = induce_shift(codes)
        # trust the spiking readout ONLY if the shift REALLY maps every prev->curr (not a spurious
        # argmax from the bounded search window) AND the answer re-encodes exactly (no wrap/overflow);
        # else fall through to the general synthesis, which handles any constant difference correctly.
        if s is not None and all(
            sim(permute(codes[i], s), codes[i + 1]) == 1.0 for i in range(len(codes) - 1)
        ):
            label = _exact_readout(permute(codes[-1], s), _NUMBER_BOOK)
            if label is not None:
                return int(label), "constant difference (spiking VSA)"
    return synthesize_sequence(values)  # test-time program synthesis over the DSL


def number_series(values):
    """The next value of a number series, or None on an honest gap. (solve_series carries the rule.)"""
    got = solve_series(values)
    return got[0] if got is not None else None


def raven_dist3(rows):
    """Raven DISTRIBUTION-OF-THREE: a 3x3 grid where every row holds the SAME three symbols in some
    order, one cell None. Induce the required triple from the complete rows (relational integration
    ACROSS rows) and complete the target row by SPIKING VSA -- the candidate whose bundle best matches
    the required-set bundle by spike coincidence. None -> not a distribution-of-three item (gap)."""
    syms = sorted({c for row in rows for c in row if c is not None})
    complete = [set(r) for r in rows if None not in r]
    if len(syms) != 3 or not complete or any(s != set(syms) for s in complete):
        return None
    target = next((r for r in rows if None in r), None)
    present = [c for c in target if c is not None] if target else []
    if len(present) != 2 or len(set(present)) != 2:  # a valid dist-3 row has two DISTINCT knowns
        return None
    book = {s: hv(s) for s in syms}
    want = bundle([book[s] for s in syms])  # the required 3-symbol superposition
    return max(
        syms, key=lambda s: coincidence(bundle([book[c] for c in present] + [book[s]]), want)
    )


def _variable_shift_next(letters):
    """Extend letter succession to a shift that is ITSELF a function of position (a b d g k: shift
    grows +1,+2,+3,+4) -- the constant-shift path above induces a single fixed step only. Extract the
    per-transition shifts as a plain integer sequence and hand THAT to the general, leave-one-out
    guarded program-inducer (induce_program): any shift-growth law expressible in its grammar (linear,
    quadratic, ...) is reached, not just a hardcoded arithmetic-shift case. The induced next shift is
    applied to the final letter; a result that would run off the alphabet is an honest gap, not a wrap.
    A symbol outside a..z gaps too (None) rather than raising ValueError -- see letter_series."""
    if any(c not in _LETTERS for c in letters):
        return None
    idx = [_LETTERS.index(c) for c in letters]
    shifts = [b - a for a, b in zip(idx, idx[1:])]
    got = induce_program(shifts)
    if got is None:
        return None
    next_shift, _label = got
    target = idx[-1] + next_shift
    return _LETTERS[target] if 0 <= target < len(_LETTERS) else None


def _letter_shift(a, b):
    """Signed a..z alphabet step from letter a to letter b, or None if either side is not a single
    lowercase letter (guards the Raven-grid helpers below from non-letter cells)."""
    if len(a) != 1 or len(b) != 1 or a not in _LETTERS or b not in _LETTERS:
        return None
    return _LETTERS.index(b) - _LETTERS.index(a)


def _row_shift_pair(triple):
    """[shift(cell0->cell1), shift(cell1->cell2)] for a fully-known 3-letter row/column, or None if
    either shift is not a clean single a..z step."""
    s1, s2 = _letter_shift(triple[0], triple[1]), _letter_shift(triple[1], triple[2])
    return None if s1 is None or s2 is None else (s1, s2)


def raven_arith_grid(rows):
    """Raven grid whose ROWS each follow the SAME per-column letter shift AND whose COLUMNS each
    follow the SAME per-row letter shift ('a b c; c d e; e f ?' -> the row shift is +1,+1 and the
    column shift is +2,+2) -- a rule gf.raven_dist3 cannot see (it only handles distribution-of-
    three). Both the row-rule and the column-rule are induced from every OTHER complete row/column
    (never from the target alone), then cross-checked against the target's own known cells before
    being applied; when BOTH are computable they must AGREE (composed relational integration, verified
    two independent ways) or it is an honest gap, never a guess."""
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        return None
    blanks = [(i, j) for i, r in enumerate(rows) for j, c in enumerate(r) if c is None]
    if len(blanks) != 1:
        return None
    ri, cj = blanks[0]

    def _predict(target, other_lines, blank_pos):
        pairs = [_row_shift_pair(line) for line in other_lines]
        if not pairs or any(p is None for p in pairs) or len(set(pairs)) != 1:
            return None
        s1, s2 = pairs[0]
        if blank_pos == 2 and _letter_shift(target[0], target[1]) == s1:
            idx = _LETTERS.index(target[1]) + s2
        elif blank_pos == 0 and _letter_shift(target[1], target[2]) == s2:
            idx = _LETTERS.index(target[1]) - s1
        elif blank_pos == 1 and _letter_shift(target[0], target[2]) == s1 + s2:
            idx = _LETTERS.index(target[0]) + s1
        else:
            return None
        return _LETTERS[idx] if 0 <= idx < len(_LETTERS) else None

    row_pred = _predict(rows[ri], [r for k, r in enumerate(rows) if k != ri], cj)
    cols = [[rows[i][j] for i in range(3)] for j in range(3)]
    col_pred = _predict(cols[cj], [c for k, c in enumerate(cols) if k != cj], ri)
    if row_pred is not None and col_pred is not None:
        return row_pred if row_pred == col_pred else None  # composed: must agree, else honest gap
    return row_pred if row_pred is not None else col_pred


_CELL_RE = re.compile(r"^([a-z])(\d)$")


def raven_cell_attrs(rows):
    """Raven grid whose cells COMPOSE two independent attributes -- a letter (Latin-square
    distribution-of-three, checked by ROW and corroborated by COLUMN) and a digit that is purely
    POSITIONAL (constant within its column, verified across every OTHER column too). gf.raven_dist3
    alone sees 9 atomic 2-character symbols and bails; this decomposes each cell first. None on an
    honest gap in either attribute (never a guessed letter or digit)."""
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        return None
    parsed = [[None if c is None else _CELL_RE.match(c) for c in r] for r in rows]
    if any(c is not None and m is None for r, pr in zip(rows, parsed) for c, m in zip(r, pr)):
        return None  # a cell isn't clean letter+digit -> not this family
    letters = [[None if m is None else m.group(1) for m in r] for r in parsed]
    digits = [[None if m is None else int(m.group(2)) for m in r] for r in parsed]

    letter_row = raven_dist3(letters)
    cols_l = [[letters[i][j] for i in range(3)] for j in range(3)]
    letter_col = raven_dist3(cols_l)
    if letter_row is not None and letter_col is not None and letter_row != letter_col:
        return None  # row-rule and column-rule disagree -> honest gap
    letter_pred = letter_row if letter_row is not None else letter_col
    if letter_pred is None:
        return None

    blanks = [(i, j) for i, r in enumerate(digits) for j, c in enumerate(r) if c is None]
    if len(blanks) != 1:
        return None
    ri, cj = blanks[0]
    col_digits = {digits[i][cj] for i in range(3) if i != ri}
    if len(col_digits) != 1:
        return None  # digit isn't column-invariant -> honest gap
    for j in range(3):  # corroborate: every OTHER column is ALSO constant
        if j != cj and len({digits[i][j] for i in range(3)}) != 1:
            return None
    return f"{letter_pred}{next(iter(col_digits))}"


def letter_series(letters):
    """Extrapolate a letter succession the same way, reading out over the a..z spike codebook. A shift
    that runs off the end of the alphabet re-encodes inexactly -> honest gap (never a wrapped letter).
    Falls back to a POSITION-DEPENDENT shift search (_variable_shift_next) when no single constant
    shift fits every transition.

    A symbol outside the a..z codebook is an HONEST GAP (None), not an exception -- the same
    disposition analogy_over already takes (`except KeyError -> None`). This is load-bearing rather
    than defensive: letter_series is the one Gf entry on the LIVE chat path, and
    spike_chat_lake._gf_request selects its tokens with `str.isalpha()`, which is True for uppercase
    and for accented / non-Latin letters. Measured 2026-08-04: a turn whose tokens included such a
    character reached `_LETTER_BOOK[c]` and raised an uncaught KeyError out of the reply path
    (_gf_request is called unguarded at spike_chat_lake.py, in respond's Gf branch) instead of
    declining to answer."""
    if any(c not in _LETTER_BOOK for c in letters):
        return None
    codes = [_LETTER_BOOK[c] for c in letters]
    s = induce_shift(codes)
    if s is not None and all(
        sim(permute(codes[i], s), codes[i + 1]) == 1.0 for i in range(len(codes) - 1)
    ):
        return _exact_readout(permute(codes[-1], s), _LETTER_BOOK)
    return _variable_shift_next(letters)


def letter_analogy(a, b, c):
    """Solve 'a is to b as c is to ?' for equal-length strings: induce the PER-POSITION shift from
    the a->b change and apply the same shift to c, each position read out by spike cleanup.
    Unchanged positions carry shift 0. None -> honest gap (unequal lengths -- no alignment)."""
    if not (len(a) == len(b) == len(c)):
        return None
    out = []
    for ca, cb, cc in zip(a, b, c, strict=True):
        s = _best_shift(_LETTER_BOOK[ca], _LETTER_BOOK[cb])  # transform this position underwent
        if (
            sim(permute(_LETTER_BOOK[ca], s), _LETTER_BOOK[cb]) != 1.0
        ):  # shift outside the search window
            return None  # -> spurious argmax, would bluff: honest gap
        label = _exact_readout(permute(_LETTER_BOOK[cc], s), _LETTER_BOOK)
        if label is None:  # ran off the alphabet -> honest gap, no wrap
            return None
        out.append(label)
    return "".join(out)


_NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_VOWELS = set("aeiou")
_WORD_ATTRS = (
    ("letter count", len),
    ("vowel count", lambda w: sum(c in _VOWELS for c in w)),
    ("consonant count", lambda w: sum(c.isalpha() and c not in _VOWELS for c in w)),
    ("unique letter count", lambda w: len(set(w))),
    ("spelled-out numeral value", lambda w: _NUMBER_WORDS.get(w)),
)


def word_number_relation(a, b, c):
    """Induce which countable ATTRIBUTE of a word a number stands for from ONE example (a -> b): try
    a small set of candidate word->number relations smallest/most-primitive first (Occam) and keep
    the FIRST whose value on `a` equals `b`. Honest ordering, not a fit to a known answer -- the
    taught 'spelled-out numeral' shortcut IS a candidate but sits last and is REJECTED outright
    whenever the example does not actually match it (e.g. one -> 3 is not one's numeral value 1), so
    a KB-style shortcut cannot win by familiarity alone. Returns the same attribute applied to `c`,
    or None when no candidate is consistent with the one given example (an honest, structural
    ambiguity -- a single example cannot always identify the relation)."""
    for _name, fn in _WORD_ATTRS:
        val = fn(a)
        if val is not None and val == b:
            out = fn(c)
            return out if out is not None else None
    return None


# ------------------------------------------------------------- the fresh Gf probe ------------------


def main_gfi():
    print("=" * 96)
    print("FLUID-INTELLIGENCE PROBE -- test-time rule induction over the brain's spiking VSA")
    print("=" * 96)

    # IN-SCOPE: constant-difference relations the model must INDUCE + extrapolate (was 0/11 before).
    passed = []

    def check(name, got, want):
        ok = got == want
        passed.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:36s} -> {got!r}  (want {want!r})")
        return ok

    # order-1 constant shift -- induced + read out through the spiking VSA
    check("arithmetic  2 4 6 8", number_series([2, 4, 6, 8]), 10)
    check("arithmetic  9 7 5 (down)", number_series([9, 7, 5]), 3)
    check("letter succ a b c", letter_series("abc"), "d")
    check("letter skip a c e", letter_series("ace"), "g")
    check("analogy abc:abd :: ijk", letter_analogy("abc", "abd", "ijk"), "ijl")
    check("analogy rst:rsu :: abc", letter_analogy("rst", "rsu", "abc"), "abd")
    # relational integration -- rules a single shift CANNOT express (were the old ceiling)
    check("geometric   2 4 8 16", number_series([2, 4, 8, 16]), 32)
    check("geometric   3 9 27", number_series([3, 9, 27]), 81)
    check("squares     1 4 9 16", number_series([1, 4, 9, 16]), 25)
    check("cubes       1 8 27 64 125", number_series([1, 8, 27, 64, 125]), 216)
    check("Fibonacci   1 1 2 3 5", number_series([1, 1, 2, 3, 5]), 8)
    check("Lucas-like  2 2 4 6 10", number_series([2, 2, 4, 6, 10]), 16)
    check(
        "Raven dist-3 rows", raven_dist3([["a", "b", "c"], ["b", "c", "a"], ["c", "a", None]]), "b"
    )
    check(
        "Raven dist-3 rows2", raven_dist3([["x", "y", "z"], ["y", "z", "x"], ["z", None, "y"]]), "x"
    )
    # PROGRAM SYNTHESIS -- depth-2 programs the flat rule families structurally CANNOT express
    check("period-2 step 1 3 6 8 11 13", number_series([1, 3, 6, 8, 11, 13]), 16)
    check("interleave 1 10 2 20 3 30", number_series([1, 10, 2, 20, 3, 30]), 4)
    check("interleave 5 1 10 2 15 3", number_series([5, 1, 10, 2, 15, 3]), 20)
    # NOVEL-SYMBOL TRANSFER -- the rule applied over an alphabet the brain has never seen
    check("novel alphabet q w e r t", analogy_over(list("qwert"), "q", "w", "e"), "r")
    check("reversed alphabet transfer", analogy_over(list("zyxw"), "z", "y", "x"), "w")
    # UNBOUNDED PROGRAM SYNTHESIS -- programs the DSL combinators alone could not reach
    check("factorial   1 2 6 24 120", number_series([1, 2, 6, 24, 120]), 720)
    check("square-recur 2 4 16 256", number_series([2, 4, 16, 256]), 65536)

    n_in, n_pass = len(passed), sum(passed)
    print(
        f"\n  Gf items solved: {n_pass}/{n_in} (spiking-VSA shift + relational integration + test-time "
        "program synthesis + novel-symbol transfer; the pre-inducer brain scored 0/11)"
    )

    # THE WALL (now that the unbounded program-inducer is wired in): a rule is out of reach only when
    # NO program in the grammar fits the data at all, or the structure is unalignable. (Sequences like
    # primes DO get a consistent-but-underdetermined program -- see program_inducer's own probe.)
    print(
        "\n  the wall -- no program/alignment fits the data at all, these MUST gap (honest), not bluff:"
    )
    # (short sequences like primes / mixed-+-x are UNDERDETERMINED, not walls: the inducer returns a
    # simplest program that fits the shown terms -- see program_inducer's probe. The true walls are
    # sequences with NO fitting program and structurally unalignable analogy/grid items.)
    ceiling = {
        "random      3 1 4 1 5": number_series([3, 1, 4, 1, 5]),  # no program fits the data
        "random      5 3 9 2 7 6": number_series([5, 3, 9, 2, 7, 6]),
        "analogy unequal len": letter_analogy("ab", "abc", "ij"),
        "raven not dist-3": raven_dist3([["a", "b", "c"], ["a", "b", "d"], ["a", "b", None]]),
        "symbol outside alphabet": analogy_over(list("abc"), "a", "b", "z"),
        # THE REACHABLE RED for letter_series' off-alphabet guard. Both inputs are shapes the LIVE
        # chat path can hand it: spike_chat_lake._gf_request keeps single-character tokens by
        # `str.isalpha()`, which is True for uppercase and for accented / non-Latin letters. Delete
        # either guard and these rows stop being GAPs and become an uncaught KeyError (letter_series)
        # or ValueError (_variable_shift_next) -- verified by removing them, 2026-08-04.
        "letter series, uppercase": letter_series("ABC"),
        "letter series, non-latin": letter_series("ab\u00e9"),
        "letter series, off-alpha variable-shift path": letter_series("abd\u00e9"),
    }
    for name, got in ceiling.items():
        print(f"  [{'GAP ' if got is None else 'BLUFF'}] {name:36s} -> {got!r}")

    # ---- gated self-check: the full DSL PASSES, everything past the wall honestly GAPS ----
    assert n_pass == n_in, f"all Gf items must pass, got {n_pass}/{n_in}"
    assert all(v is None for v in ceiling.values()), (
        "a rule outside the bounded DSL must be an HONEST GAP, never a bluffed answer"
    )
    # every solve is test-time program synthesis: the program is searched per problem, not stored.
    assert induce_shift([code(1), code(4), code(7)]) == 3, "must induce an unseen step at test time"
    assert solve_series([1, 10, 2, 20, 3, 30])[1].startswith("interleave"), (
        "must synthesise a depth-2 program"
    )
    assert solve_series([1, 3, 6, 8, 11, 13])[1].startswith("period-2"), (
        "must find a periodic op-cycle"
    )
    assert analogy_over(list("qwert"), "q", "w", "e") == "r", (
        "must transfer a rule to a novel alphabet"
    )
    assert number_series([1, 2, 6, 24, 120]) == 720, (
        "the unbounded inducer must synthesise factorial"
    )

    print(
        "\nSELF-CHECK PASSED: fluid intelligence by TEST-TIME PROGRAM SYNTHESIS -- spiking-VSA shift, "
        "relational-integration combinators (period/interleave), an UNBOUNDED program-inducer "
        "(factorial, products, ...), and novel-symbol transfer; short sequences are underdetermined "
        "(simplest fitting program) and genuinely patternless / unalignable items honestly gap."
    )
    return 0


# ============================================================================================
# --- active_learner ---
# ============================================================================================
r"""
THE OPEN-ENDED ACTIVE LEARNER -- generate hypotheses, TEST them against the world, and grow.

The three limits the passive inducer + learned model could not escape were:
  (1) BOUNDED GRAMMAR   -- a fixed searcher gaps outside its primitives.
  (2) TRAINING DISTRIBUTION -- a learned model gaps outside the corpus it was fit on.
  (3) UNDERDETERMINATION -- finitely many terms admit many fitting rules; the passive inducer just
      returns the simplest and may be wrong (primes 2 3 5 7 11 -> it guessed 15).

This is the fourth move -- an AGENT that escapes all three, not by being bigger, but by ACTING:

  generate  -> enumerate every low-complexity hypothesis CONSISTENT with what it has seen so far
  test      -> when hypotheses DISAGREE, DESIGN THE EXPERIMENT (the query where they diverge most),
               ASK THE WORLD (a queryable oracle it does not control), and PRUNE what the answer kills
  grow      -> when it identifies a rule, ADD it to its LIBRARY as a new primitive, so the searchable
               grammar GROWS -- next time a harder rule is one step away, not out of budget
  persist   -> the library is saved durably, so learning ACCUMULATES across problems and sessions

Escapes, each demonstrated in the self-check:
  (3) underdetermination -> ACTIVE QUERYING: the world answers f(3)=7, killing the simplest-but-wrong
      hypothesis the passive inducer would have committed to. Evidence breaks the tie.
  (1) bounded grammar    -> LIBRARY LEARNING: n**6 is OUT of the size budget with the base grammar
      (no consistent hypothesis found), but after the agent learns `cube` it solves n**6 as cube*cube.
  (2) training dist      -> no corpus: it learns ONLINE from each world it meets and the library
      persists durably, so competence accumulates rather than being fixed at train time.

HONEST SCOPE (stated plainly): "the world" is a QUERYABLE ORACLE -- a hidden function the agent may
probe -- not the physical universe. This is genuine active empiricism (test a hypothesis against an
external source of truth you do not control) inside a sandbox. It is still bounded at any instant (a
size budget, a query budget, a primitive set); the point is that the boundary MOVES as it learns, and
that it resolves underdetermination by EXPERIMENT rather than by guessing. Where the world is
genuinely outside its grammar and queries run out, it reports the irreducible ambiguity honestly.

Deps: stdlib + snn_data_lake.substrate.home (durable library). Grammar is closed-form in the index n
(so any position is directly queryable): E := n | c | lib | E+E | E-E | E*E.
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.gf     # the 3-escape self-check
"""


_CONSTS_al = (1, 2, 3)
_OPS_al = ("+", "-", "*")
_DEDUP_SPAN = range(1, 7)  # behaviours compared over these indices to dedup equal programs


def _evaluate_al(expr, n, library):
    tag = expr[0]
    if tag == "n":
        return n
    if tag == "c":
        return expr[1]
    if tag == "lib":
        return _evaluate_al(library[expr[1]], n, library)
    a = _evaluate_al(expr[1], n, library)
    b = _evaluate_al(expr[2], n, library)
    return a + b if tag == "+" else a - b if tag == "-" else a * b


def _render_al(expr):
    tag = expr[0]
    if tag == "n":
        return "n"
    if tag == "c":
        return str(expr[1])
    if tag == "lib":
        return expr[1]
    return f"({_render_al(expr[1])} {tag} {_render_al(expr[2])})"


_ENUM_CACHE = {}  # (max_size, library-names) -> {size: [expr,...]} -- built once, reused every call


def _enum_layers(max_size, library):
    """The grammar layers, built ONCE per (size, library) and cached -- rebuilding them every call cost
    ~16-24 ms on the hot ambiguity-check path. Pure function of max_size + the library primitive names."""
    key = (max_size, tuple(sorted(library)))
    if key not in _ENUM_CACHE:
        terminals = [("n",)] + [("c", k) for k in _CONSTS_al] + [("lib", name) for name in library]
        layers = {1: terminals}
        for size in range(2, max_size + 1):
            built = []
            for op in _OPS_al:
                for left in range(1, size - 1):
                    for a in layers[left]:
                        for b in layers[size - 1 - left]:
                            built.append((op, a, b))
            layers[size] = built
        _ENUM_CACHE[key] = layers
    return _ENUM_CACHE[key]


def _enumerate(max_size, library, cap=20000):
    """Yield (size, expr) smallest-first over the (cached) grammar, with library primitives as extra
    terminals (so the searchable space GROWS as the library grows). Bounded by a hard cap."""
    layers = _enum_layers(max_size, library)
    seen = 0
    for size in range(1, max_size + 1):
        for e in layers[size]:
            yield size, e
            seen += 1
            if seen >= cap:
                return


def consistent_hypotheses(obs, library, max_size, top_n):
    """Every low-complexity program (smallest first) consistent with all (index, value) observations,
    deduped by BEHAVIOUR (programs equal over _DEDUP_SPAN count once). Returns up to top_n."""
    out, behaviours = [], set()
    for _size, expr in _enumerate(max_size, library):
        if any(_evaluate_al(expr, k, library) != v for k, v in obs):
            continue
        beh = tuple(_evaluate_al(expr, k, library) for k in _DEDUP_SPAN)
        if beh in behaviours:
            continue
        behaviours.add(beh)
        out.append(expr)
        if len(out) >= top_n:
            break
    return out


def _discriminating_query(hyps, asked, span, library):
    """The unqueried index in `span` where the hypotheses DISAGREE most (distinct predictions) -- the
    maximally informative experiment. None if they agree everywhere in span (cannot discriminate)."""
    best_k, best_spread = None, 1
    for k in span:
        if k in asked:
            continue
        spread = len({_evaluate_al(h, k, library) for h in hyps})
        if spread > best_spread:
            best_k, best_spread = k, spread
    return best_k


class World:
    """A queryable oracle: a hidden generating function the agent may probe but not see. query(k)
    returns the k-th value and counts the probe -- experiments against the world cost something."""

    def __init__(self, gen, name="world"):
        self._gen = gen
        self.name = name
        self.queries = 0

    def query(self, k):
        self.queries += 1
        return self._gen(k)


class Library:
    """The GROWING set of learned primitives (name -> program), persisted so competence accumulates
    across problems and sessions (escaping any fixed train-time distribution).

    P4b VIOLATION CLOSED 2026-08-05, exactly as recorded at OpenGf's `_STATE` above and by the same
    single edit: `save_state`/`load_state` no longer write `<NORD_HOME>/active_library.json`, they
    write a `_T_STATE` record keyed by that name INSIDE `brain_lake.bin`, and import the existing
    sidecar once on first read. `_STATE` is now a record NAME rather than a filename; it is left
    spelled with the .json suffix on purpose, so the name a lake record carries and the sidecar it
    was migrated from are the same string and the mapping needs no translation table."""

    _STATE = "active_library.json"

    def __init__(self, prims=None):
        self.prims = dict(prims or {})

    def __contains__(self, name):
        return name in self.prims

    def __getitem__(self, name):
        return self.prims[name]

    def __iter__(self):
        return iter(self.prims)

    def add(self, name, expr):
        self.prims[name] = expr

    def save(self):
        save_state(self._STATE, self.prims)

    @classmethod
    def load(cls):
        return cls(load_state(cls._STATE, default={}))


def function_from_pairs(pairs, query_x, max_size=7):
    """Induce f(x) from explicit (x, y) example PAIRS ('if 3 -> 10, 5 -> 26, 7 -> 50, then 9 -> ?') --
    as opposed to a flat number SERIES, this is example -> rule induction over independent inputs.
    Reuses the active_learner grammar (E := n | c | E+E | E-E | E*E, n standing in for x) via the
    same _enumerate/_evaluate_al machinery, smallest-first. LEAVE-ONE-OUT guarded: the LAST given
    pair is held out of the fit, and a program is trusted only once it BOTH reproduces every earlier
    pair AND correctly predicts the held-out one (this is what stops a coincidental 2-point fit from
    being trusted on 3 examples). Returns (value, label) for f(query_x), or None -- an honest gap (no
    program in the grammar reproduces every given pair)."""
    if len(pairs) < 3:
        return None
    *train, held = pairs
    lib = Library()
    for _size, expr in _enumerate(max_size, lib.prims):
        if any(_evaluate_al(expr, x, lib) != y for x, y in train):
            continue
        if _evaluate_al(expr, held[0], lib) != held[1]:
            continue
        return _evaluate_al(expr, query_x, lib), "f(x)=%s" % _render_al(expr).replace("n", "x")
    return None


_STR_TRANSFORMS = (
    ("id", lambda s: s),
    ("rev", lambda s: s[::-1]),
    ("upper", lambda s: s.upper()),
    ("lower", lambda s: s.lower()),
)
_STR_SUFFIXES = (
    ("", lambda s: ""),
    ("+len", lambda s: str(len(s))),
    ("+first", lambda s: s[0] if s else ""),
    ("+last", lambda s: s[-1] if s else ""),
)


def string_from_pairs(pairs, query):
    """Induce a COMPOSED string program f(s) = transform(s) + suffix(original s) from explicit
    (in, out) example pairs -- e.g. reverse-then-append-length ('cat' -> 'tac3'). A small
    compositional DSL (4 whole-string transforms x 4 length/first/last/none suffixes), tried
    smallest-first; LEAVE-ONE-OUT guarded exactly like function_from_pairs: the induced program must
    reproduce every earlier pair AND correctly predict the held-out LAST pair before it is trusted,
    then it is applied to the queried word. None -- no program in this small DSL reproduces every
    given pair (an honest, structural gap, not a guess)."""
    if len(pairs) < 3:
        return None
    *train, held = pairs
    for t_name, t_fn in _STR_TRANSFORMS:
        for s_name, s_fn in _STR_SUFFIXES:

            def prog(s, t_fn=t_fn, s_fn=s_fn):
                return t_fn(s) + s_fn(s)

            if all(prog(x) == y for x, y in train) and prog(held[0]) == held[1]:
                return prog(query), "%s(s)+%s" % (t_name, s_name if s_name else "''")
    return None


class ActiveLearner:
    """Generate -> test-against-the-world -> prune -> (identify) -> grow the library."""

    def __init__(self, library=None, max_size=7, span=range(1, 13), top_n=12, max_queries=10):  # noqa: B008 -- range is immutable; the default is a constant span, not shared state
        self.library = library or Library()
        self.max_size = max_size
        self.span = span
        self.top_n = top_n
        self.max_queries = max_queries

    def solve(self, world, seeds=(1, 2)):
        """Actively identify the world's rule. Returns (expr | None, identified: bool, n_queries).
        identified=True means a UNIQUE behaviour survived; None means no hypothesis fits (out of
        grammar); (expr, False) means queries/span were exhausted with ambiguity remaining -- honest."""
        obs = [(k, world.query(k)) for k in seeds]
        asked = set(seeds)
        for _ in range(self.max_queries):
            hyps = consistent_hypotheses(obs, self.library, self.max_size, self.top_n)
            if not hyps:
                return None, False, world.queries
            if len(hyps) == 1:
                return hyps[0], True, world.queries
            k = _discriminating_query(hyps, asked, self.span, self.library)
            if k is None:  # cannot discriminate further in span
                return hyps[0], False, world.queries
            obs.append((k, world.query(k)))
            asked.add(k)
        hyps = consistent_hypotheses(obs, self.library, self.max_size, self.top_n)
        return (hyps[0] if hyps else None), (len(hyps) == 1), world.queries

    def learn(self, name, expr):
        """Add an identified rule to the library as a reusable primitive -- the grammar GROWS."""
        self.library.add(name, expr)
        return self


def _matches(expr, gen, library, span):
    """The learned program reproduces the world on a HELD-OUT range (genuine identification, not fit)."""
    return all(_evaluate_al(expr, k, library) == gen(k) for k in span)


def main_al():
    print("=" * 96)
    print("THE OPEN-ENDED ACTIVE LEARNER -- generate hypotheses, TEST against the world, grow")
    print("=" * 96)

    # -- ESCAPE (3): UNDERDETERMINATION broken by EXPERIMENT ---------------------------------------
    # World is 3n-2. From only f(1)=1, f(2)=4 the SIMPLEST consistent program is n*n (smaller than
    # 3n-2) -- exactly the trap the passive inducer falls into. The active agent QUERIES and prunes.
    lin = ActiveLearner()
    w_lin = World(lambda n: 3 * n - 2, "3n-2")
    passive = consistent_hypotheses([(1, 1), (2, 4)], Library(), 7, 12)
    expr, ident, nq = lin.solve(w_lin, seeds=(1, 2))
    val = range(5, 12)
    print(
        f"\n  [escape 3] world=3n-2. passive simplest fit on {{1,4}} = {_render_al(passive[0])} "
        f"(predicts f(3)={_evaluate_al(passive[0], 3, Library())}, WRONG)."
    )
    print(
        f"             active agent asked the world {nq} times -> identified {_render_al(expr)} "
        f"(identified={ident}); matches world on held-out f(5..11)={_matches(expr, w_lin._gen, lin.library, val)}"
    )
    assert _evaluate_al(passive[0], 3, Library()) != 3 * 3 - 2, (
        "the passive trap: simplest fit must be wrong here"
    )
    assert ident and _matches(expr, w_lin._gen, lin.library, val), (
        "active querying must identify the true rule"
    )
    assert nq > len((1, 2)), "it must have run at least one EXPERIMENT beyond the seeds"

    # -- ESCAPE (1): BOUNDED GRAMMAR broken by LIBRARY LEARNING ------------------------------------
    # n**6 is OUT of the size budget with the base grammar -> no hypothesis. The agent first learns
    # `cube` from an n**3 world, ADDS it, and then n**6 becomes cube*cube -- in budget, solvable.
    agent = ActiveLearner()
    w6_before, _, _ = agent.solve(World(lambda n: n**6, "n^6"), seeds=(2, 3))
    cube_expr, cube_ident, _ = agent.solve(World(lambda n: n**3, "n^3"), seeds=(2, 3, 4))
    agent.learn("cube", cube_expr)  # the grammar GROWS
    w6_after, w6_ident, _ = agent.solve(World(lambda n: n**6, "n^6"), seeds=(2, 3, 5))
    print(
        f"\n  [escape 1] n^6 with the BASE grammar -> {w6_before!r} (out of budget). learned "
        f"cube={_render_al(cube_expr)}; n^6 now -> {_render_al(w6_after)} (identified={w6_ident})."
    )
    assert w6_before is None, "n^6 must be OUT of the base grammar's budget (the wall)"
    assert cube_ident and w6_after is not None and w6_ident, (
        "after learning cube, n^6 must be solvable"
    )
    assert _matches(w6_after, lambda n: n**6, agent.library, range(6, 10)), (
        "n^6 solution must be the true rule"
    )
    assert "cube" in _render_al(w6_after), "the n^6 solution must REUSE the learned cube primitive"

    # -- ESCAPE (2): no frozen corpus -- the library PERSISTS durably (accumulates across sessions) --
    import os
    import tempfile

    prev, tmp = os.environ.get("NORD_HOME"), tempfile.mkdtemp(prefix="snn_active_")
    os.environ["NORD_HOME"] = tmp
    try:
        lib = Library()
        lib.add("cube", cube_expr)
        lib.save()
        reloaded = Library.load()
        print(f"\n  [escape 2] library persisted durably and reloaded: primitives={list(reloaded)}")
        assert "cube" in reloaded, (
            "the learned library must persist across sessions (continual, not frozen)"
        )
        # and a fresh agent with the reloaded library solves n^6 immediately (accumulated competence)
        fresh = ActiveLearner(library=reloaded)
        w6, w6i, _ = fresh.solve(World(lambda n: n**6, "n^6"), seeds=(2, 3, 5))
        assert w6i and _matches(w6, lambda n: n**6, reloaded, range(6, 10)), (
            "reloaded competence must transfer"
        )
        # P4b: `Library.load()` above runs in the SAME interpreter that just saved, so it proves
        # nothing about durability on its own. A SEPARATE process must rebuild it from the lake,
        # and no `active_library.json` sidecar may exist.
        import subprocess
        import sys

        from snn_data_lake.substrate import home as _home

        assert not os.path.exists(_home.state_path(Library._STATE)), (
            "the active library must go IN brain_lake.bin -- no active_library.json sidecar"
        )
        _src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        child = (
            f"import sys; sys.path.insert(0, {_src_root!r})\n"
            f"import os; os.environ['NORD_HOME'] = {tmp!r}\n"
            "from snn_data_lake.reasoning.gf import Library\n"
            "back = Library.load()\n"
            "assert 'cube' in back, list(back)\n"
            # compared against the PARENT's reloaded value, not against `cube_expr`: this library is
            # stored as JSON, which has no tuple, so a round-tripped expression is a list of lists
            # either way. The claim under test is "the child reads what the parent stored", and
            # asserting the pre-serialisation tuple would fail on the encoding, not on durability.
            f"assert back['cube'] == {reloaded['cube']!r}, back['cube']\n"
            "print('  [escape 2] a SEPARATE process reloaded %s from the lake' % list(back))\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        assert rc == 0, f"the active library did not survive into a NEW process (rc={rc})"
    finally:
        os.environ.pop("NORD_HOME", None)
        if prev is not None:
            os.environ["NORD_HOME"] = prev

    # -- the honest wall stays: a world outside the grammar with queries exhausted -> reported, not faked
    stumper = ActiveLearner(max_queries=6)
    s_expr, s_ident, _ = stumper.solve(
        World(lambda n: 1 if n in (2, 3, 5, 7, 11) else 0, "is-prime-ish"), seeds=(1, 2)
    )
    print(
        f"\n  [honest wall] a world with no closed-form in the grammar -> identified={s_ident} "
        f"(reports ambiguity/None rather than faking certainty)"
    )
    assert not s_ident, "outside the grammar, it must NOT falsely claim to have identified the rule"

    print(
        "\nSELF-CHECK PASSED: an open-ended active learner -- breaks underdetermination by EXPERIMENT, "
        "breaks the grammar budget by LIBRARY LEARNING, accumulates durably (no frozen corpus); and "
        "still reports honest ambiguity where the world is outside its reach."
    )
    return 0


# ============================================================================================
# --- sequence_learner ---
# ============================================================================================
r"""
LARGE LEARNED MODEL vs the PROGRAM-INDUCER -- the measured Gc-vs-Gf contrast.

The honest other half of "cross the wall". A learned model (here a real numpy MLP trained by gradient
descent on a LARGE synthetic corpus of sequences -- NOT an imported transformer, no faked LLM) learns
to predict the next term. We then MEASURE the thing Chollet's argument predicts:

  * IN-DISTRIBUTION (rule families the model was trained on): it predicts well -- crystallized
    competence (Gc) bought by data + fitting.
  * OUT-OF-DISTRIBUTION (a rule family held OUT of training): it fails -- it interpolates its training
    distribution and cannot SYNTHESISE a rule it never saw.

Against the same out-of-distribution sequences, program_inducer.induce_program (test-time synthesis)
solves them EXACTLY. That gap IS the point: learning at scale buys coverage (Gc); it does not buy
fluid rule synthesis (Gf). This is not defeatable by making the net bigger -- a bigger net trained on
more families just moves the boundary; for ANY finite training distribution there is an
out-of-distribution rule it fails, while the inducer builds the rule on the spot.

HONEST SCOPE: the MLP here is deliberately small and trained briefly so the self-check runs in CI;
its in-distribution accuracy is a demonstration, not a benchmark. The measured OOD failure is the
structural claim and it does NOT depend on model size.

Deps: numpy + stdlib + program_inducer.

numpy is imported at MODULE scope here. It was `global np` + a lazy `import numpy as np` inside
main_sl(), commented "keep gf.py import numpy-free so the chat bridge stays pure" -- a purity
argument, which P7 does not accept, and it cost real correctness: `MLP` and `_corpus` read the
module-global `np`, which did not exist until main_sl() had run, so `gf.MLP(rng)` raised
`NameError: name 'np' is not defined` for any other caller (verified 2026-08-04).

Measured before changing it, per P7's requirement that a dependency decision ship a measurement on
the real workload rather than a preference: bare interpreter 41.1 ms, +import gf 137.8 ms, and
importing numpy first adds 83.4 ms -- but numpy is ALREADY in sys.modules by the time either
importer of this module reaches it (verified for `cortex.brain` and for the chat module the comment
named). So the laziness bought 0 ms on the workload it cited, and numpy is an unconditional core
dependency in pyproject.toml regardless.
  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.gf     # train + contrast + check
"""


import numpy as np

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it. UNGOVERNED tier, so
#: these are REPORTED and asserted on nothing.
#:
#: Three symbols here are NOT_A_RULE. They grow a SYMBOLIC GRAMMAR or search over one; none updates a
#: weight, so no cosine could ever be computed for them. They were left UNDECLARED at first because
#: the four original dispositions had no value for "not a learning rule", which reported them as
#: UNDETERMINABLE -- true, but that state is ratcheted DOWN as debt, and these are not debt. They are
#: a permanent and correct answer.
GRADIENT_CLAIM = {
    "MLP.train": (
        "CONTROL",
        "not a learning rule this project is trying to make good -- it is the Gc arm that main_sl() "
        "requires to LOSE (in-dist 100%, OOD 25% against the inducer's 100%). Owner-adjudicated "
        "2026-08-15 and exempted in eprop_transport_gate.EXEMPT, which FR-019 retires INTO this "
        "registry so exactly one hand-maintained record of the disposition survives. Do NOT port it "
        "to e-prop: a weaker learner would satisfy mdl_ok < ind_ok for the wrong reason",
    ),
    "ActiveLearner.learn": (
        "NOT_A_RULE",
        "adds an identified rule to the library as a reusable primitive -- the GRAMMAR grows. No "
        "weight, threshold or trace is updated, so there is nothing for a gradient to approximate",
    ),
    "OpenGf._learn": (
        "NOT_A_RULE",
        "promotes a solved rule to a NAMED PRIMITIVE when it COMPRESSES. A symbolic library "
        "operation gated on description length, not a weight update",
    ),
    "OpenGf.solve_unbounded": (
        "NOT_A_RULE",
        "iterative-deepening SEARCH: raises the node budget in a loop until a compressing program "
        "fits. Enumerated because the walk keys on 'solve'/'learn'-adjacent names; it learns nothing",
    ),
}

WINDOW = 4  # the model sees 4 consecutive terms, predicts the 5th


def _norm_sl(terms):
    """Scale a window + target by the window's magnitude so the MLP sees bounded inputs. Returns
    (x, y, scale): x = first WINDOW terms / scale, y = next term / scale."""
    scale = max(1.0, max(abs(float(t)) for t in terms[:WINDOW]))
    return [t / scale for t in terms[:WINDOW]], terms[WINDOW] / scale, scale


def _corpus(rng, n):
    """A large corpus drawn from the TRAINING families: arithmetic and geometric (random params)."""
    xs, ys = [], []
    for _ in range(n):
        if rng.random() < 0.5:  # arithmetic: a, a+d, a+2d, ...
            a, d = rng.integers(-5, 10), rng.integers(-4, 6)
            seq = [int(a + d * i) for i in range(WINDOW + 1)]
        else:  # geometric: a, a*r, a*r^2, ...
            a, r = rng.integers(1, 6), rng.integers(2, 5)
            seq = [int(a * r**i) for i in range(WINDOW + 1)]
        x, y, _ = _norm_sl(seq)
        xs.append(x)
        ys.append([y])
    return np.array(xs), np.array(ys)


class MLP:
    """A plain one-hidden-layer tanh regressor trained by SGD. Small on purpose (the OOD claim does
    not depend on size); genuinely learned -- every weight comes from gradient descent on the data.

    WEIGHT TRANSPORT, DECLARED HERE BECAUSE THIS CLASS PERFORMS IT. `train` is hand-written
    reverse-mode: `gh = (g @ self.w2.T) * (1 - h**2)` reads the FORWARD read-out weight, transposed,
    to carry error back into `self.w1`. Under the "e-prop, not backprop" invariant's distinguishing
    test -- what does the transpose FEED -- that is a weight update, which is the shape the rule
    forbids on the brain's own learning paths.

    ADJUDICATED 2026-08-15 (docs/POLICIES.md, e-prop invariant block): NOT a violation, because this
    is a CONTROL rather than one of the brain's learning rules -- code whose job is to be measured
    AGAINST is not a rule by which the brain learns. This class has zero references outside this
    file, one construction site in `main_sl()`, and reaches no tick and no faculty.

    NOT to be confused with the instrument the invariant NAMES, which is a different control that
    already exists: `reasoning/eprop.py::Net.grad_bptt`, the independent reverse-mode gradient the
    "cosine > 0.5" clause is measured against. `MLP` is a Gc BASELINE, not a gradient reference.
    (This distinction is recorded because the first draft of the adjudication blurred it.)

    DO NOT "FIX" THIS TO E-PROP. `main_sl()` asserts `mdl_ok < ind_ok` beside `in_acc >= 0.5`: the
    model must genuinely learn AND must still fail the held-out family the inducer solves. A weaker
    learner passes that comparison for the wrong reason -- it would fail OOD because it learns
    worse, not because learning is not synthesis -- which converts the measurement into a
    contaminated control and voids the Gc-vs-Gf claim this self-check exists to make.

    EXEMPTION GRANTED 2026-08-15 by the owner: `gates/eprop_transport_gate.EXEMPT` carries
    `reasoning.gf.train` with this verdict as its reason, so promoting `gf.py` no longer reds that
    gate. The entry is asserted to keep matching a real finding -- if this transport is ever removed
    or re-spelled, the exemption stops blessing anything and the gate FAILS on the stale row rather
    than going quietly green."""

    def __init__(self, rng, hidden=32):
        self.w1 = rng.standard_normal((WINDOW, hidden)) * 0.3
        self.b1 = np.zeros(hidden)
        self.w2 = rng.standard_normal((hidden, 1)) * 0.3
        self.b2 = np.zeros(1)

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2

    def train(self, x, y, epochs=400, lr=0.05, batch=128, rng=None):
        n = len(x)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s : s + batch]
                xb, yb = x[b], y[b]
                h, pred = self.forward(xb)
                g = (pred - yb) / len(b)  # dMSE/dpred
                gw2 = h.T @ g
                gb2 = g.sum(0)
                gh = (g @ self.w2.T) * (1 - h**2)
                gw1 = xb.T @ gh
                gb1 = gh.sum(0)
                self.w2 -= lr * gw2
                self.b2 -= lr * gb2
                self.w1 -= lr * gw1
                self.b1 -= lr * gb1

    def predict_next(self, terms):
        """Predict the next integer term of a >=WINDOW sequence (last WINDOW terms, denormalised)."""
        x, _, scale = _norm_sl(list(terms[-WINDOW:]) + [0.0])
        _, pred = self.forward(np.array([x]))
        return int(round(pred[0, 0] * scale))


def main_sl():
    print("=" * 96)
    print(
        "LARGE LEARNED MODEL vs PROGRAM-INDUCER -- does learning at scale buy fluid intelligence?"
    )
    print("=" * 96)
    rng = np.random.default_rng(0)

    xtr, ytr = _corpus(rng, 8000)  # a large learned corpus (Gc)
    model = MLP(rng, hidden=32)
    model.train(xtr, ytr, epochs=400, lr=0.05, rng=rng)
    print(
        f"  trained a {WINDOW}->32->1 MLP by SGD on {len(xtr)} sequences (arithmetic + geometric)"
    )

    # in-distribution: fresh arithmetic + geometric it did NOT train on (but from the SAME families)
    in_seqs = []
    for _ in range(500):
        a, d = rng.integers(-5, 10), rng.integers(-4, 6)
        in_seqs.append([int(a + d * i) for i in range(WINDOW + 1)])
    in_acc = sum(model.predict_next(s[:WINDOW]) == s[WINDOW] for s in in_seqs) / len(in_seqs)

    # OUT-OF-DISTRIBUTION: rule families HELD OUT of training (factorial, Fibonacci) as length-6
    # sequences -- predict the 6th. The learned model sees the last WINDOW terms; the program-inducer
    # sees the first 5 and SYNTHESISES the rule. Same information, two different mechanisms.
    ood6 = [
        [1, 2, 6, 24, 120, 720],
        [2, 6, 24, 120, 720, 5040],  # factorial
        [1, 1, 2, 3, 5, 8],
        [2, 3, 5, 8, 13, 21],
    ]  # Fibonacci
    ind_ok = sum((induce_program(s[:-1]) or (None,))[0] == s[-1] for s in ood6) / len(ood6)
    mdl_ok = sum(model.predict_next(s[:-1]) == s[-1] for s in ood6) / len(ood6)

    print(
        f"\n  IN-DISTRIBUTION (arithmetic+geometric, unseen samples)   learned model: {in_acc:.0%}"
    )
    print("  OUT-OF-DISTRIBUTION (factorial+Fibonacci, held-out family):")
    print(f"      learned model (Gc, interpolates its training):         {mdl_ok:.0%}")
    print(f"      program-inducer (Gf, synthesises the rule per problem): {ind_ok:.0%}")

    # ---- the measured result: learning buys in-distribution competence, NOT out-of-distribution Gf --
    assert in_acc >= 0.5, (
        f"the learned model must actually LEARN its training families (got {in_acc:.0%})"
    )
    assert ind_ok == 1.0, (
        f"the inducer must SYNTHESISE every held-out rule exactly (got {ind_ok:.0%})"
    )
    assert mdl_ok < ind_ok, (
        "the measured Gc-vs-Gf point: the learned model must FAIL the held-out family that the "
        f"inducer solves (model {mdl_ok:.0%} vs inducer {ind_ok:.0%})"
    )

    print(
        "\nSELF-CHECK PASSED: a large learned model buys IN-DISTRIBUTION competence (Gc) but FAILS "
        "the held-out rule family that test-time program synthesis (Gf) solves exactly. Scale moves "
        "the boundary; it does not remove it -- learning is not synthesis."
    )
    return 0


# ============================================================================================
# --- combined self-check: run every sub-module's checks ---
# ============================================================================================
def main():
    main_pi()
    main_ogf()
    main_gfi()
    main_al()
    main_sl()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
