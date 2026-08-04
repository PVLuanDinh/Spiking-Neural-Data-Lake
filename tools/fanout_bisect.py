"""Measure F* -- the smallest connections-per-unit a module keeps its function at.

WHY THIS IS A TOOL AND NOT A CONSTANT. specs/009 research measured that the fanout knee tracks
STORED LOAD, not width and not density. Both of those were asserted and both were falsified on the
same module with the same metric: an 8.3x knee-density spread at fixed width, and knee fanout
doubling at constant density. Every closed form attempted relocated the constant it was meant to
remove. So F is measured per deployment, here, and passed in -- a default in the module would be
exactly the magic constant P1 forbids.

THE TOLERANCE IS PRE-REGISTERED, not fitted. It was written to specs/009 research.md before the
sweep that measured the knee, and it is encoded here so it cannot drift to fit a result:

    F PASSES iff, over N seeds at the same width and stored load, the module's own task metric is
    within MEAN_TOL (0.02) of the dense baseline in the mean AND within MIN_TOL (0.05) at the worst
    seed. F* is the smallest passing F for which every larger grid point also passes, and it MUST be
    BRACKETED -- a failing F strictly below it. An F* at the grid floor is CENSORED, never reported
    as a value. A drop above CLIFF (0.20) between adjacent points is a CLIFF, not a knee.

TWO REFUSALS THIS TOOL MAKES, both from failures that happened during research:

  * A metric that cannot decrease is not function, it is arithmetic. Anything that rises with
    parameter count regardless of learning (a sum over surviving weights, a nearest-prototype
    similarity) would report a knee that is pure counting.
  * A dense baseline that is already failing makes the tolerance meaningless. "95 percent of dense"
    is worthless when dense is 0.150. This happened once and the result was reported before anyone
    noticed.

Run: python tools/fanout_bisect.py            (self-check)
     python tools/fanout_bisect.py --brain    (measure at the brain's real configuration)
"""

import sys

sys.path.insert(0, "src")

#: Pre-registered in specs/009 research.md BEFORE any sweep. Do not tune to fit a result.
MEAN_TOL = 0.02
MIN_TOL = 0.05
CLIFF = 0.20
SEEDS = (0, 1, 2, 3, 4)

#: Descending powers of two down to 8. Descending so bracketing is natural: the first FAILING F
#: brackets the last passing one from below.
GRID = (256, 224, 192, 176, 160, 144, 128, 112, 96, 64, 32, 16, 8)

#: A dense baseline below this cannot support a meaningful "within 0.02" claim.
MIN_USABLE_BASELINE = 0.5


def sweep(build, metric, grid=GRID, seeds=SEEDS):
    """Measure the metric at each F. `build(fanout, seed)` returns a module; `metric(m)` scores it.

    Returns (baseline, rows) where rows is [(F, mean, min, passed)] in grid order.
    """
    base = [metric(build(None, s)) for s in seeds]
    baseline = sum(base) / len(base)
    if baseline < MIN_USABLE_BASELINE:
        raise ValueError(
            f"dense baseline {baseline:.3f} is below {MIN_USABLE_BASELINE}: the module is not "
            "working before thinning, so no tolerance against it means anything"
        )
    rows = []
    for f in grid:
        got = [metric(build(f, s)) for s in seeds]
        mean, worst = sum(got) / len(got), min(got)
        ok = abs(mean - baseline) <= MEAN_TOL and (baseline - worst) <= MIN_TOL
        rows.append((f, mean, worst, ok))
    return baseline, rows


def f_star(rows):
    """(F*, status). Status is 'BRACKETED', 'CENSORED' or 'NONE'.

    F* is the smallest passing F with every LARGER grid point also passing, and it must have a
    failing F below it on the grid -- otherwise the grid simply ran out before function did, which
    is not a knee and must not be reported as one.
    """
    best = None
    for i, (f, _, _, ok) in enumerate(rows):
        if not ok:
            break
        best = (i, f)
    if best is None:
        return None, "NONE"
    i, f = best
    if i + 1 >= len(rows):
        return f, "CENSORED"  # grid floor reached with no failure below it
    return f, "BRACKETED"


def cliffs(rows):
    """Adjacent grid points whose mean drops by more than CLIFF -- reported, never smoothed."""
    out = []
    for (f0, m0, _, _), (f1, m1, _, _) in zip(rows, rows[1:], strict=False):
        if m0 - m1 > CLIFF:
            out.append((f0, f1, m0 - m1))
    return out


def report(name, baseline, rows):
    f, status = f_star(rows)
    print(f"\n=== {name}")
    print(f"  tolerance : mean {MEAN_TOL}, min {MIN_TOL} vs dense, seeds {list(SEEDS)}")
    print(f"  dense     : {baseline:.3f}")
    print(f"  {'F':>6} {'mean':>8} {'min':>8}  verdict")
    for fx, mean, worst, ok in rows:
        print(f"  {fx:>6} {mean:>8.3f} {worst:>8.3f}  {'PASS' if ok else 'FAIL'}")
    for a, b, d in cliffs(rows):
        print(f"  CLIFF: {a} -> {b} drops {d:.3f} (> {CLIFF}) -- not a knee")
    if status == "BRACKETED":
        below = next(fx for fx, _, _, ok in rows if not ok and fx < f)
        print(f"  F* = {f}   BRACKETED by F={below} which failed")
    elif status == "CENSORED":
        print(f"  F* CENSORED at F >= {f} -- the grid ran out before function did")
    else:
        print("  F* = NONE -- no fanout on the grid held the tolerance")
    return f, status


def _brain_config():
    """Measure F* at the configuration cortex/brain.py actually constructs."""
    from snn_data_lake.hippocampus.episode_consolidation import EpisodeConsolidator

    DIM, CAP = 256, 64  # brain.py:856

    def build(fanout, seed):
        e = EpisodeConsolidator(dim=DIM, capacity=CAP, seed=seed, fanout=fanout)
        for i in range(CAP):  # the brain's real load is its capacity
            ep = {"who": f"a{i}", "what": f"w{i}", "where": f"p{i % 7}", "when": str(i)}
            e.store(e.encode(ep), ep)
        return e

    baseline, rows = sweep(build, lambda m: m.retained())
    return report(f"EpisodeConsolidator dim={DIM} capacity={CAP} load={CAP}", baseline, rows)


def _selfcheck():
    print("FANOUT BISECT self-check")

    # f_star must refuse an unbracketed result rather than report the grid floor as a knee.
    all_pass = [(f, 1.0, 1.0, True) for f in GRID]
    assert f_star(all_pass) == (GRID[-1], "CENSORED"), f_star(all_pass)
    none_pass = [(f, 0.0, 0.0, False) for f in GRID]
    assert f_star(none_pass) == (None, "NONE"), f_star(none_pass)
    mixed = [(f, 1.0, 1.0, i < 3) for i, f in enumerate(GRID)]
    assert f_star(mixed) == (GRID[2], "BRACKETED"), f_star(mixed)

    # a cliff must be reported rather than smoothed into a knee
    steep = [(GRID[0], 1.0, 1.0, True), (GRID[1], 0.5, 0.5, False)]
    assert cliffs(steep) == [(GRID[0], GRID[1], 0.5)], cliffs(steep)

    # a broken dense baseline must raise, not silently anchor the comparison
    class _Fake:
        def retained(self):
            return 0.1

    try:
        sweep(lambda f, s: _Fake(), lambda m: m.retained(), grid=(8,), seeds=(0,))
    except ValueError as exc:
        assert "not working before thinning" in str(exc), exc
    else:
        raise AssertionError("a failing dense baseline must refuse the sweep")

    print("  bracketing, censoring, cliff detection, baseline refusal: OK")
    print("self-check OK: 4 of 4 properties hold")


if __name__ == "__main__":
    if "--brain" in sys.argv:
        _brain_config()
    else:
        _selfcheck()
