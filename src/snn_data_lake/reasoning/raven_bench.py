r"""Raven's Progressive Matrices (RPM) scoring harness -- the honest yardstick for the NVSA Gf lift.

Mirrors reasoning/arc_bench.py: load a split, score a solver, and keep an un-riggable FAILING control
(random guessing) so a real number is believable. RPM is 8-choice, so the control -- and the floor any
solver must clear -- is CHANCE = 12.5%. The grounded anchor is NVSA (Hersche et al. 2023,
arXiv:2203.04571), which reaches ~88% on the RAVEN benchmark; rpm_solver.RpmSolver runs that backend
over the brain's spiking VSA and is bounded to the same NVSA rule set.

DATA. If a real I-RAVEN / RAVEN dataset directory is present (data/raven or data/i-raven, one JSON per
task with {context, candidates, answer}), load_tasks reads it. Otherwise split='synthetic' GENERATES
deterministic 3x3 tasks: for each attribute one of the four NVSA rules (distribute-three / progression
/ constant / arithmetic) governs the row values, and 8 candidates (the answer + 7 perturbed
distractors) are built so exactly one completes cell (2,2).

Scoring is exact single-choice: the solver's answer_idx must equal the held-out answer index.

HONEST ANCHORS for reading any score this harness prints:
  - a random-guess solver scores ~12.5% (the self-check asserts this control genuinely fails, so a
    ~12.5% here means the harness measures rather than that it is broken);
  - NVSA on RAVEN is ~88% (Hersche 2023) -- the grounded ceiling for this rule set;
  - the solver is bounded to the NVSA rule set over clean attribute codes: NOT full ARC.

Imports no brainstem, routes no actuator.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.reasoning.raven_bench
"""

# %-format is the hand-aligned house style of every sibling in reasoning/; kept for consistency,
# not reflowed to f-strings. The file-level ruff blanket-suppression of UP031 that stood here was
# DELETED 2026-08-04 as doubly wrong: its parenthetical claimed the repo "excludes [reasoning/] from
# `ruff check .`", which has been FALSE since 2026-07-28 (pyproject's own comment records the
# change), and UP031 is already in the `src/snn_data_lake/reasoning/**` per-file-ignores list, so
# the directive suppressed nothing that was not suppressed anyway. Verified by running ruff on
# this file with the line removed: 0 findings.
import json
import os
import random
import sys

from snn_data_lake.reasoning.rpm_solver import RpmSolver

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA_DIRS = (os.path.join(_ROOT, "data", "i-raven"), os.path.join(_ROOT, "data", "raven"))

ATTRS = ("type", "size", "color")
CHANCE = 0.125  # 8-choice RPM -> random guessing floor
NVSA_RAVEN = 0.88  # cited grounded anchor (Hersche et al. 2023, arXiv:2203.04571)
RULES = ("distribute3", "progression", "constant", "arithmetic")
_POOL = tuple(range(1, 9))  # candidate value pool for synthetic distractors


# ---------------------------------------------------------------- real dataset (optional) ------------


def _load_real(limit=None):
    """Load tasks from a real I-RAVEN/RAVEN JSON dir if present. Each file: {context:[8 cells],
    candidates:[8 cells], answer:int}. Returns [(task_id, task)] or [] if no dataset is on disk."""
    for d in DATA_DIRS:
        if not os.path.isdir(d):
            continue
        out = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                t = json.load(f)
            if {"context", "candidates", "answer"} <= set(t):
                out.append((fn[:-5], t))
        if out:
            return out[:limit] if limit is not None else out
    return []


# ---------------------------------------------------------------- synthetic generation ---------------


def _rotate(seq, k):
    k %= len(seq)
    return seq[k:] + seq[:k]


def _grid_for(rule, rng):
    """Build a 3x3 integer value grid (rows x cols) obeying one NVSA rule. Values stay in 1..8."""
    if rule == "distribute3":
        triple = rng.sample(range(1, 7), 3)
        return [_rotate(triple, r + rng.randint(0, 2)) for r in range(3)]
    if rule == "progression":
        step = rng.choice((1, 2))
        rows = []
        for _ in range(3):
            start = rng.randint(1, 3)  # ONE start per row (fixed across the row's 3 cells)
            rows.append([start + step * c for c in range(3)])
        return rows
    if rule == "constant":
        return [[rng.randint(1, 6)] * 3 for _ in range(3)]
    if rule == "arithmetic":
        rows = []
        for _ in range(3):
            a, b = rng.randint(1, 4), rng.randint(1, 4)
            rows.append([a, b, a + b])
        return rows
    raise ValueError(rule)


def _candidates(answer, rng, pool=_POOL, n=8):
    """8 distinct candidate cells including the answer; distractors perturb 1..len(ATTRS) attributes."""
    cands = [dict(answer)]
    seen = {tuple(answer[a] for a in ATTRS)}
    guard = 0
    while len(cands) < n and guard < 3000:
        guard += 1
        c = dict(answer)
        for a in rng.sample(list(ATTRS), rng.randint(1, len(ATTRS))):
            c[a] = rng.choice([x for x in pool if x != answer[a]])
        key = tuple(c[a] for a in ATTRS)
        if key not in seen:
            seen.add(key)
            cands.append(c)
    rng.shuffle(cands)
    idx = next(i for i, c in enumerate(cands) if all(c[a] == answer[a] for a in ATTRS))
    return cands, idx


def _synth_task(task_id, seed):
    """One deterministic synthetic RPM: an independent NVSA rule per attribute + 8 candidates."""
    rng = random.Random(seed)
    grids = {a: _grid_for(rng.choice(RULES), rng) for a in ATTRS}
    context = []
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                continue
            context.append({a: grids[a][r][c] for a in ATTRS})
    answer = {a: grids[a][2][2] for a in ATTRS}
    candidates, idx = _candidates(answer, rng)
    return (task_id, {"context": context, "candidates": candidates, "answer": idx})


def load_tasks(split="synthetic", limit=None):
    """Return [(task_id, task), ...]. split='real' loads a dataset dir if present (else []);
    split='synthetic' (default) generates deterministic tasks. Each task is
    {context:[8 cells], candidates:[8 cells], answer:int}."""
    if split == "real":
        return _load_real(limit)
    if split == "synthetic":
        n = limit if limit is not None else 100
        return [_synth_task("syn%04d" % i, seed=1000 + i) for i in range(n)]
    real = _load_real(limit)
    return real if real else load_tasks("synthetic", limit)


# ---------------------------------------------------------------- scoring ----------------------------


def score_task(solver, task):
    """(correct, known) for ONE task, from a SINGLE solve. A solver that raises or returns junk
    scores (False, False) -- never crashes the run.

    This returned only `correct` and `score()` called `solve` a SECOND time to read `known`.
    Two defects in one (fixed 2026-08-04): it doubled the solve cost of every bench run, and --
    the reason it is a correctness bug and not just a slow one -- a solver carrying state sees a
    DIFFERENT internal state on the two calls, so `accuracy` and `known_frac` were measured on
    two different runs of the same task set. `RandomSolver` in this very file is exactly that:
    its `self.rng` advances per solve, so accuracy read draws 1,3,5,... and known_frac read
    2,4,6,.... One solve, one pair of numbers.

    WHAT THIS CHANGED, MEASURED AND RECORDED 2026-08-05 because the fix moved a PUBLISHED number
    and the fix itself did not say so. Both versions were run on this tree, same seed, limit=60:

        solve calls in main()   240 -> 120   (60 -> 30 per score() call for a stateful solver)
        RpmSolver accuracy      98.3% -> 98.3%   known 96.7% -> 96.7%   (stateless: unchanged)
        random-control accuracy 11.7% -> 8.3%    (RandomSolver draws 1,3,5,... -> 1,2,3,...)
        const-control accuracy  15.0% -> 15.0%   (stateless: unchanged)

    The new 8.3% is the honest number -- it is the control measured on the same draws the accuracy
    column reports -- and it sits further BELOW the 12.5% chance line, so every assert below keeps
    its margin. But `docs/DEPTH_CEILINGS.md` records "controls: random 11.7% | constant 15.0%" and
    that line is now unreproducible from this file. Correcting it is outside this module and is
    handed back rather than left implied: a bench fix that silently invalidates a published control
    is how an unreproducible number gets into the docs in the first place."""
    try:
        res = solver.solve(task["context"], task["candidates"])
        return int(res.get("answer_idx", -1)) == int(task["answer"]), bool(res.get("known"))
    except Exception:
        return False, False


def score(solver, split="synthetic", limit=50):
    """Score a solver over a split. Returns {split, solved, n, accuracy, known_frac}."""
    tasks = load_tasks(split, limit=limit)
    solved = 0
    known = 0
    for _tid, task in tasks:
        correct, is_known = score_task(solver, task)
        solved += correct
        known += is_known
    n = len(tasks)
    return {
        "split": split,
        "solved": solved,
        "n": n,
        "accuracy": (solved / n) if n else 0.0,
        "known_frac": (known / n) if n else 0.0,
    }


class RandomSolver:
    """FAILING control: ignore the matrix and guess uniformly among the 8 candidates. Must score
    ~12.5% -- if it scores well, the harness (or the task set) is not discriminating."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def solve(self, context_cells, candidates):
        idx = self.rng.randrange(len(candidates)) if candidates else 0
        return {"answer_idx": idx, "confidence": 0.0, "known": False, "rule": "random-guess"}


class ConstantSolver:
    """Second control: always pick candidate 0. Also ~12.5% since the answer index is shuffled."""

    def solve(self, context_cells, candidates):
        return {"answer_idx": 0, "confidence": 0.0, "known": False, "rule": "always-0"}


def main():
    limit = 60
    real = load_tasks("real")
    if real:
        print("raven_bench: %d real RPM tasks on disk" % len(real))

    solver = RpmSolver(seed=0)
    r = score(solver, "synthetic", limit=limit)
    ctrl = score(RandomSolver(seed=0), "synthetic", limit=limit)
    ctrl2 = score(ConstantSolver(), "synthetic", limit=limit)

    print("=" * 78)
    print("RAVEN'S PROGRESSIVE MATRICES BENCH (8-choice; chance 12.5%%, NVSA-RAVEN ~88%% cited)")
    print("=" * 78)
    print("solver         split      solved   accuracy   known")
    print(
        "  %-12s %-9s %3d/%-3d   %6.1f%%   %5.1f%%"
        % ("RpmSolver", r["split"], r["solved"], r["n"], 100 * r["accuracy"], 100 * r["known_frac"])
    )
    print(
        "  %-12s %-9s %3d/%-3d   %6.1f%%      -"
        % ("random-ctrl", ctrl["split"], ctrl["solved"], ctrl["n"], 100 * ctrl["accuracy"])
    )
    print(
        "  %-12s %-9s %3d/%-3d   %6.1f%%      -"
        % ("const-ctrl", ctrl2["split"], ctrl2["solved"], ctrl2["n"], 100 * ctrl2["accuracy"])
    )
    print("-" * 78)
    print(
        "anchors: chance=%.1f%%  random-control=%.1f%%  solver=%.1f%%  NVSA-RAVEN=%.0f%% (cited)"
        % (100 * CHANCE, 100 * ctrl["accuracy"], 100 * r["accuracy"], 100 * NVSA_RAVEN)
    )
    print("ceiling: bounded to the NVSA rule set over clean attribute codes -- NOT full ARC.")

    # ---- un-riggable self-check: the solver must beat chance by a wide margin AND the control must
    #      genuinely fail near chance (proving the bench discriminates). Deterministic, ASCII. ----
    assert r["n"] == limit and ctrl["n"] == limit, "both runs must cover the same task set"
    assert r["accuracy"] > 0.5, (
        "the NVSA solver must beat 12.5%% chance by a wide margin, got %.1f%%"
        % (100 * r["accuracy"])
    )
    assert ctrl["accuracy"] <= 0.30, (
        "the random control must fail near chance (<=30%%), got %.1f%%" % (100 * ctrl["accuracy"])
    )
    assert ctrl2["accuracy"] <= 0.30, (
        "the constant control must fail near chance (<=30%%), got %.1f%%"
        % (100 * ctrl2["accuracy"])
    )
    assert r["accuracy"] > ctrl["accuracy"] + 0.30, (
        "the solver must clear the control by a wide, discriminating margin (%.1f%% vs %.1f%%)"
        % (100 * r["accuracy"], 100 * ctrl["accuracy"])
    )
    # determinism: a fresh solver + fresh task set reproduce the identical accuracy.
    assert score(RpmSolver(seed=0), "synthetic", limit=limit)["solved"] == r["solved"], (
        "scoring must be deterministic under a fixed seed"
    )

    print("=" * 78)
    print(
        "self-check OK: RpmSolver %.1f%% vs random-control %.1f%% (chance 12.5%%) over %d synthetic "
        "8-choice RPMs -- the bench discriminates (solver >> control), deterministic; "
        "grounded anchor NVSA-RAVEN ~88%%, ceiling = the NVSA rule set."
        % (100 * r["accuracy"], 100 * ctrl["accuracy"], limit)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
