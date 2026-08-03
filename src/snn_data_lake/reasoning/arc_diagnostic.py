r"""ARC failure diagnostic -- is the solver BUDGET-exhausted or DSL-COVERAGE-limited?

WHY THIS EXISTS
---------------
Prior literature research concluded that NEURAL-GUIDED program search (learning a policy that
prioritises promising branches) would only help arc_solver if its failures are SEARCH-BUDGET
limited -- solutions exist in the program space but the blind enumeration cannot reach them in
budget -- and would be worth ~nothing if the failures are DSL-COVERAGE limited (no program in the
space at all, so no search order can find one). Nobody had measured which regime arc_solver is in.
This module measures it, without modifying arc_solver.py: it drives `synthesize` (the public entry
point) at TWO budgets and compares, per the two-budget method.

METHOD (and why the second budget is a PROOF, not a guess)
----------------------------------------------------------
`synthesize` truncates silently on its node/time budget -- it does not report exhaustion state.
But its node counter is externally boundable: nodes increments once per (frontier state, primitive)
expansion, |frontier_1| = 1 and |frontier_{d+1}| <= |frontier_d| * P for a pool of P primitives, so
a search to max_depth D can NEVER count past  P + P^2 + ... + P^D.  Running with a node budget
ABOVE that bound therefore provably cannot be node-truncated; if such a run also finishes under its
wall-clock safety cap, it has enumerated the ENTIRE (equivalence-pruned) depth<=D program space.
P is computed per task from arc_solver's primitive registry (read-only reflection; conservative
fallback constant if the internals move).

Each test pair the baseline (the solver's shipped default budget) fails is then classified:

  solved_baseline   not a failure.
  budget_proven     the provably-untruncated run SOLVES it: the failure was the budget, proven.
  budget_unresolved the provably-untruncated run hit its wall-clock cap: budget-limited in the
                    honest sense of "hit the (enlarged) budget without proving no program exists".
  coverage_proven   full enumeration completed and the wanted grid is NOT producible by ANY
                    training-consistent program at depth<=D: no budget or search order can help.
  misranked         full space contains a training-consistent program that DOES produce the wanted
                    grid, but >=2 other consistent-but-wrong candidate grids rank above it (MDL
                    smallest-first + pass@2), so the deterministic search returns the wrong two AT
                    ANY BUDGET. Budget cannot help; candidate RE-RANKING could. Found by re-running
                    the same provably-untruncated search with attempts=128 (the rank probe).
  rank_unresolved   the rank probe itself was capped or overflowed 128 candidates: unproven.

Task-level bin = coverage_proven if any pair is coverage-proven (that pair blocks the task at any
budget and any ranking); else unresolved if any pair is unresolved; else misranked if any pair is
misranked (budget alone provably cannot flip the task); else budget_proven (EVERY failing pair
flips with more budget -- the only bin where raising the budget fixes the task).

HONEST LIMITS (read before quoting the verdict)
-----------------------------------------------
- Everything is measured WITHIN the solver's own depth<=3 program space. "coverage_proven" means
  no depth<=3 program over THIS DSL works; deeper programs are not probed (the depth-4 space is
  ~P^4 = 5M nodes/task -- out of scope here, and neural guidance that reached depth 4+ would face
  the same DSL vocabulary wall).
- The wall-clock safety cap makes ONLY the *_unresolved bins machine-dependent; every proven bin
  is a deterministic replay (node budgets only, no RNG anywhere).
- attempts=2 is the official ARC pass@2; the misranked bin exists precisely because a determinstic
  MDL ranking at k=2 is part of what fails.
- If no ARC data is on disk the diagnostic falls back to the bundled synthetic control tasks and
  SAYS SO -- synthetic numbers say nothing about ARC.

  PYTHONPATH=src python -m snn_data_lake.reasoning.arc_diagnostic            # self-check
  PYTHONPATH=src python -m snn_data_lake.reasoning.arc_diagnostic --full     # whole training split
  PYTHONPATH=src python -m snn_data_lake.reasoning.arc_diagnostic --limit 50 --cap 30

Deterministic apart from the stated caps; ASCII-only output. Representation-only: imports no
brainstem, routes no actuator.
"""

import sys
import time

from snn_data_lake.reasoning import arc_bench, arc_solver

ATTEMPTS = 2  # official ARC pass@2
MAX_DEPTH = 3  # the solver's own default program depth
BASELINE_NODES = 4000  # the solver's shipped default node budget (make_solver)
EXHAUST_CAP_S = 30.0  # wall-clock safety valve on the provably-untruncated runs
RANK_ATTEMPTS = 128  # candidate cap for the rank probe
_P_FALLBACK = 64  # pool-size overestimate if reflection into arc_solver fails

# verdict threshold: guided search is "worth building" only if its measured ceiling could flip at
# least this fraction of the current failures (budget wins + unresolved + re-rankable).
WORTH_FRACTION = 0.15


def _tup(g):
    return tuple(tuple(int(v) for v in row) for row in g)


def _eq(a, b):
    """Exact grid equality, tolerant of tuple/list mixes; malformed -> False."""
    try:
        if len(a) != len(b):
            return False
        for ra, rb in zip(a, b):
            if len(ra) != len(rb) or any(int(x) != int(y) for x, y in zip(ra, rb)):
                return False
        return True
    except (TypeError, ValueError):
        return False


def _pool_size(train_pairs):
    """Exact primitive-pool size for this task (base + fitted + minted, no library) via
    READ-ONLY use of arc_solver's own _pool builder -- exact BY CONSTRUCTION, so the node
    bound tracks the pool even as intake changes (the specs/006 merge added per-task minted
    records). Falls back to a conservative overestimate if internals move -- an overestimate
    keeps the bound valid, it just wastes budget headroom."""
    try:
        tin = tuple(_tup(i) for i, _o in train_pairs)
        tout = tuple(_tup(o) for _i, o in train_pairs)
        return len(arc_solver._pool(tin, tout))
    except (AttributeError, TypeError, ValueError, IndexError):
        return _P_FALLBACK


def node_bound(train_pairs, max_depth=MAX_DEPTH):
    """A node budget PROVABLY above any node count `synthesize` can reach at this depth:
    nodes <= P + P^2 + ... + P^max_depth (see module docstring). +1 so `nodes > budget` can
    never fire."""
    p = _pool_size(train_pairs)
    total, term = 0, 1
    for _ in range(max_depth):
        term *= p
        total += term
    return total + 1


def probe_pair(
    train,
    test_input,
    want,
    baseline_nodes=BASELINE_NODES,
    attempts=ATTEMPTS,
    max_depth=MAX_DEPTH,
    exhaust_cap_s=EXHAUST_CAP_S,
):
    """Classify ONE (train, test) pair. Returns {'status': ..., evidence...}; statuses are
    solved_baseline / budget_proven / budget_unresolved / coverage_proven / misranked /
    rank_unresolved (see module docstring for the exact meaning and proof obligations)."""
    base_c, _bp = arc_solver.synthesize(
        train,
        test_input,
        attempts=attempts,
        max_depth=max_depth,
        node_budget=baseline_nodes,
        time_budget=None,
    )
    if any(_eq(c, want) for c in base_c[:attempts]):
        return {"status": "solved_baseline"}

    bound = node_bound(train, max_depth)
    t0 = time.monotonic()
    ex_c, ex_p = arc_solver.synthesize(
        train,
        test_input,
        attempts=attempts,
        max_depth=max_depth,
        node_budget=bound,
        time_budget=exhaust_cap_s,
    )
    ex_s = time.monotonic() - t0
    out = {"bound": bound, "exhaust_s": round(ex_s, 3), "n_cands": len(ex_c)}
    if any(_eq(c, want) for c in ex_c[:attempts]):
        out["status"] = "budget_proven"
        return out
    if ex_s >= exhaust_cap_s:
        out["status"] = "budget_unresolved"  # hit the enlarged budget; nothing proven
        return out
    # full enumeration completed. Did the search stop early because attempts filled with
    # consistent-but-wrong grids? (const-closer candidates are added AFTER the search ends and
    # never imply an early stop.)
    non_const = sum(1 for _names, closer in ex_p if closer != "const")
    if non_const < attempts:
        out["status"] = "coverage_proven"  # whole space, no (more) consistent program
        return out
    # rank probe: is the wanted grid ANYWHERE in the training-consistent candidate set?
    t0 = time.monotonic()
    rk_c, _rp = arc_solver.synthesize(
        train,
        test_input,
        attempts=RANK_ATTEMPTS,
        max_depth=max_depth,
        node_budget=bound,
        time_budget=exhaust_cap_s,
    )
    rk_s = time.monotonic() - t0
    out["rank_s"] = round(rk_s, 3)
    out["rank_cands"] = len(rk_c)
    if any(_eq(c, want) for c in rk_c):
        out["status"] = "misranked"  # expressible but ranked below the top-2
        return out
    if rk_s >= exhaust_cap_s or len(rk_c) >= RANK_ATTEMPTS:
        out["status"] = "rank_unresolved"
        return out
    out["status"] = "coverage_proven"  # whole space, wanted grid not producible
    return out


_TASK_BINS = ("solved", "budget_proven", "unresolved", "misranked", "coverage_proven")


def diagnose_task(task, **kw):
    """Classify one arc_bench-format task dict. Returns {'status', 'pairs': [pair dicts]}."""
    train = [(p["input"], p["output"]) for p in task.get("train", [])]
    pairs = []
    for pair in task.get("test", []):
        r = probe_pair(train, pair["input"], pair["output"], **kw)
        pairs.append(r)
        if r["status"] == "coverage_proven":
            break  # this pair blocks the task at ANY budget/ranking
    sts = [r["status"] for r in pairs]
    if all(s == "solved_baseline" for s in sts):
        status = "solved"
    elif "coverage_proven" in sts:
        status = "coverage_proven"
    elif "budget_unresolved" in sts or "rank_unresolved" in sts:
        status = "unresolved"
    elif "misranked" in sts:
        status = "misranked"  # budget alone provably cannot flip the task
    else:
        status = "budget_proven"  # every failing pair flips with more budget
    return {"status": status, "pairs": pairs}


def synthetic_tasks():
    """Bundled fallback/control tasks: (tid, task_dict, expected_task_bin). Each expectation was
    verified against the real solver -- they double as the genuinely-failing controls."""
    rot = {
        "train": [
            {"input": [[1, 2], [3, 0]], "output": [[0, 3], [2, 1]]},
            {"input": [[5, 0], [0, 6]], "output": [[6, 0], [0, 5]]},
        ],
        "test": [{"input": [[7, 0], [0, 8]], "output": [[8, 0], [0, 7]]}],
    }
    # crop_content -> scale2; genuinely unsolvable at a 20-node baseline, solved at the bound
    two = {
        "train": [
            {"input": [[0, 0], [0, 3]], "output": [[3, 3], [3, 3]]},
            {"input": [[5, 0], [0, 0]], "output": [[5, 5], [5, 5]]},
        ],
        "test": [{"input": [[0, 0], [9, 0]], "output": [[9, 9], [9, 9]]}],
    }
    # outputs of DIFFERENT shapes from same-shaped inputs, colours 8/9 never in any input:
    # full enumeration yields ZERO consistent programs (verified) -> coverage_proven, direct.
    cov = {
        "train": [
            {"input": [[1, 2, 1], [2, 1, 2]], "output": [[9, 9], [9, 9], [9, 9]]},
            {"input": [[2, 1, 2], [1, 2, 1]], "output": [[8, 8, 8], [8, 8, 8]]},
        ],
        "test": [{"input": [[1, 1, 2], [2, 2, 1]], "output": [[7, 7], [7, 7]]}],
    }
    # one ambiguous train pair: root colour-map + rot180 fill pass@2 first (both wrong on the
    # test); the true rule flip_h IS in the consistent set (verified) -> misranked.
    mis = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4], [5, 6]], "output": [[4, 3], [6, 5]]}],
    }
    # same ambiguity but the wanted grid uses colour 9 that no consistent program can produce:
    # rank probe exhausts the space without finding it (verified) -> coverage_proven via rank.
    cvr = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
        ],
        "test": [{"input": [[1, 5], [2, 6]], "output": [[9, 9], [9, 9]]}],
    }
    return [
        ("syn_rot180", rot, "solved"),
        ("syn_budget", two, "budget_proven"),
        ("syn_cover_direct", cov, "coverage_proven"),
        ("syn_misranked", mis, "misranked"),
        ("syn_cover_rank", cvr, "coverage_proven"),
    ]


def diagnose(split="training", limit=None, verbose=False, **kw):
    """Run the diagnostic over an ARC split (or the synthetic fallback when no data is on disk
    -- flagged in 'data_source'). Returns the honest report dict; prints nothing unless verbose."""
    tasks = arc_bench.load_tasks(split)
    data_source = "arc-" + split
    if not tasks:
        tasks = [(tid, t) for tid, t, _exp in synthetic_tasks()]
        data_source = "synthetic-fallback"
    if limit is not None:
        tasks = tasks[:limit]
    counts = {b: 0 for b in _TASK_BINS}
    detail = []
    for tid, task in tasks:
        r = diagnose_task(task, **kw)
        counts[r["status"]] += 1
        detail.append({"tid": tid, "status": r["status"], "pairs": r["pairs"]})
        if verbose:
            print("  %-24s %s" % (tid, r["status"]))
    n = len(tasks)
    failing = n - counts["solved"]
    budget_side = counts["budget_proven"] + counts["unresolved"]
    coverage_side = counts["coverage_proven"] + counts["misranked"]
    rep = {
        "data_source": data_source,
        "split": split,
        "n_tasks": n,
        "solved": counts["solved"],
        "failing": failing,
        "budget_proven": counts["budget_proven"],
        "unresolved": counts["unresolved"],
        "misranked": counts["misranked"],
        "coverage_proven": counts["coverage_proven"],
        "budget_exhausted": budget_side,
        "coverage_limited": coverage_side,
        "tasks": detail,
    }
    rep["verdict"] = _verdict(rep)
    return rep


def _verdict(rep):
    """One-line verdict from the measured numbers only. Ceilings, not promises: budget_ceiling is
    what infinite budget could flip; guided_ceiling adds the re-rankable and unproven tasks --
    the most any depth<=3 neural-guided search could POSSIBLY flip over this DSL."""
    f = rep["failing"]
    if f == 0:
        return "VERDICT: no failing tasks measured; nothing to diagnose."
    budget_ceiling = rep["budget_proven"] + rep["unresolved"]
    guided_ceiling = budget_ceiling + rep["misranked"]
    n = rep["n_tasks"]
    worth = (guided_ceiling / f) >= WORTH_FRACTION
    return (
        "VERDICT: neural-guided search over this DSL is %s -- of %d failures, %d (%.1f%%) are "
        "provably coverage-limited; ceiling from ANY budget increase is +%d task(s) (+%.1f%% "
        "absolute), plus %d re-rankable and %d unproven: guided-search ceiling +%d (+%.1f%% "
        "absolute)."
        % (
            "WORTH BUILDING" if worth else "NOT worth building",
            f,
            rep["coverage_proven"],
            100.0 * rep["coverage_proven"] / f,
            rep["budget_proven"],
            100.0 * rep["budget_proven"] / n,
            rep["misranked"],
            rep["unresolved"],
            guided_ceiling,
            100.0 * guided_ceiling / n,
        )
    )


def report_lines(rep):
    """ASCII summary lines for a diagnose() report."""
    lines = [
        "source=%s split=%s tasks=%d solved@baseline=%d failing=%d"
        % (rep["data_source"], rep["split"], rep["n_tasks"], rep["solved"], rep["failing"]),
        "%d failing tasks: %d budget-exhausted (%d proven fixed by budget, %d unresolved at cap), "
        "%d coverage-limited (%d no-consistent-program, %d consistent-but-misranked) (%.1f%% "
        "coverage-limited)"
        % (
            rep["failing"],
            rep["budget_exhausted"],
            rep["budget_proven"],
            rep["unresolved"],
            rep["coverage_limited"],
            rep["coverage_proven"],
            rep["misranked"],
            (100.0 * rep["coverage_limited"] / rep["failing"]) if rep["failing"] else 0.0,
        ),
        rep["verdict"],
    ]
    if rep["data_source"] == "synthetic-fallback":
        lines.insert(
            0,
            "NOTE: no ARC data on disk -- synthetic fallback tasks only; these "
            "numbers say NOTHING about ARC.",
        )
    return lines


# ====================================================================================================
# self-check -- controls whose expected classification was verified against the real solver
# ====================================================================================================


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cap = EXHAUST_CAP_S
    limit = 12
    full = "--full" in argv
    if "--cap" in argv:
        cap = float(argv[argv.index("--cap") + 1])
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
        full = True

    if full:
        # measurement run: the whole split (or --limit N), per-task lines, honest headline
        print("=" * 96)
        print(
            "ARC BUDGET-VS-COVERAGE DIAGNOSTIC -- full run (baseline=%d nodes, cap=%.0fs)"
            % (BASELINE_NODES, cap)
        )
        print("=" * 96)
        t0 = time.monotonic()
        rep = diagnose(
            "training",
            limit=None if "--limit" not in argv else limit,
            verbose=True,
            exhaust_cap_s=cap,
        )
        for ln in report_lines(rep):
            print(ln)
        print("elapsed: %.0fs" % (time.monotonic() - t0))
        return 0

    print("=" * 96)
    print("ARC DIAGNOSTIC -- budget-vs-coverage failure classifier for arc_solver")
    print("=" * 96)
    ok = True

    # ---- (a) control classifications (each expectation pre-verified against the solver) ---------
    for tid, task, expect in synthetic_tasks():
        kw = {}
        if tid == "syn_budget":
            kw["baseline_nodes"] = 20  # the reduced baseline that makes it fail first
        r = diagnose_task(task, **kw)
        tag = "OK " if r["status"] == expect else "FAIL"
        print("  %s control %-18s -> %-15s (expected %s)" % (tag, tid, r["status"], expect))
        if r["status"] != expect:
            ok = False
    # the coverage control must be a DIRECT full-enumeration proof (zero candidates), and the
    # rank-path control must show the rank probe actually ran
    cov_task = dict(synthetic_tasks()[2][1])
    cov_pair = diagnose_task(cov_task)["pairs"][0]
    if not (cov_pair["status"] == "coverage_proven" and cov_pair["n_cands"] == 0):
        print("  FAIL: direct coverage control should exhaust the space with 0 candidates")
        ok = False
    rank_pair = diagnose_task(synthetic_tasks()[4][1])["pairs"][0]
    if "rank_cands" not in rank_pair:
        print("  FAIL: rank-path coverage control never reached the rank probe")
        ok = False

    # ---- (b) the bound is a real fixed point: doubling it must change nothing -------------------
    train = [(p["input"], p["output"]) for p in cov_task["train"]]
    ti = cov_task["test"][0]["input"]
    b = node_bound(train)
    c1, p1 = arc_solver.synthesize(train, ti, node_budget=b, time_budget=None)
    c2, p2 = arc_solver.synthesize(train, ti, node_budget=2 * b, time_budget=None)
    if (c1, p1) != (c2, p2):
        print("  FAIL: node_bound is not a fixed point -- doubling the budget changed the result")
        ok = False
    print("node_bound fixed point: budget %d and %d give identical results" % (b, 2 * b))

    # ---- (c) spot diagnostic on real data (honest fallback note when absent) --------------------
    rep = diagnose("training", limit=limit)
    for ln in report_lines(rep):
        print(ln)
    if rep["solved"] + rep["failing"] != rep["n_tasks"]:
        print("  FAIL: task bins do not sum to the task count")
        ok = False
    if rep["budget_exhausted"] + rep["coverage_limited"] != rep["failing"]:
        print("  FAIL: failure bins do not sum to the failure count")
        ok = False

    print("-" * 96)
    print("HONEST: every 'proven' bin is a deterministic replay; only the *_unresolved bins depend")
    print("on the wall cap. All claims live INSIDE the solver's depth<=3 program space -- nothing")
    print("here says what a deeper DSL or a richer vocabulary would do.")
    if ok:
        print(
            "self-check OK: arc_diagnostic classified %d/%d controls correctly; %s[:%d] -> "
            "%d failing = %d budget / %d coverage-limited"
            % (
                5,
                5,
                rep["data_source"],
                rep["n_tasks"],
                rep["failing"],
                rep["budget_exhausted"],
                rep["coverage_limited"],
            )
        )
        return 0
    print("arc_diagnostic self-check: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
