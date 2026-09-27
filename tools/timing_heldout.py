"""Held-out A/B for one added time cell: does it EARN its place, or is it free-rider arithmetic?

The question this answers is narrow on purpose. `cortex/time_cells.py` tiles an interval with
`n_cells` ramping-to-threshold cells; `grow("interior")` splits the widest log-gap. Whether the
added cell improves the population's timing has to be judged on intervals the population was NOT
taught, against a control that shows the improvement is not something ANY added cell would get.

What this instrument refuses to do is more important than what it does:

* It refuses to report a verdict on a readout that cannot see its own population. Before it measures
  anything it perturbs a mid threshold and requires the readout's output to move, and it runs the
  degenerate-ladder control (a 64-level ladder spanning 6.3e-5, carrying no timing information, on
  which the module's own `final_err < 5.0` assertion MUST fail). Either check failing prints
  NOT-ELIGIBLE and exits 2. Exit 2 is not a soft 0 -- a probe that cannot ask the question must not
  answer it. Under the pre-2026-08 ramp readout this instrument exits 2 by construction, which is
  the correct behaviour, not a bug.
* It refuses to call an effect inside its own dispersion a result. That is exit 1, an explicit null,
  and it is a valid end state for the whole feature.

Split, ported discipline not position: teach on one portion, judge on a disjoint remainder, refuse
when either is too small (`growth/ng_snn.py:178-201`, its held-out prune -- this said
`reasoning/ng_snn.py` until 2026-08-04, a path that has not existed since the v3.78 promotion;
the LINES still resolve, only the tier was wrong). The POSITION does not transfer onto the
raw interval axis -- MEASURED, a literal trailing-20% split leaves only 4 of 63 gaps able to move
the metric and a stride over the integer axis gives 15 of 63, while the gap-major grid used here
gives 63 of 63. The universe is derived from the population's OWN ladder: 10 points per gap at
`lo + (hi-lo)*(p+0.5)/10`, ordered gap-major, deterministic positional 1-in-5 stride
(`index % 5 == 4` -> TEST). No RNG, so determinism semantics survive.

Statistic ported from `scratchpad/grow_on_top_nonstationary.py:130-137` (population-agnostic
arithmetic; the file itself is gitignored and `tools/citation_guard.py`'s STRICT assert -- the
`assert not unexempt` in its `main()` -- forbids a tracked file citing an untracked .py as a
reproduction path, so it is re-derived here rather than imported. Named by symbol: this said
`:320`, which is the neighbouring stale-exemption assert).
Two deliberate changes: tie counts print UNCONDITIONALLY -- MEASURED 10 of 30 paired differences
were exactly 0.0, and a mean/se hiding 33% zero-inflation overstates its own power -- and every
count prints as `N of M`, never `N/M`, because the coverage-artifact fingerprint harvests bare
ratios positionally.

  python tools/timing_heldout.py --seeds 12
  python tools/timing_heldout.py --regimes          # the saturation census, writes regimes.md
"""

import argparse
import bisect
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Probes run with PYTHONPATH=<root>/src only, which drops the brain to a blind stdlib perception
# tier and leaves the repo root off the path entirely. Insert both ourselves.
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from snn_data_lake.cortex.time_cells import TimeCellClock, interval_grid  # noqa: E402

_REGIMES_MD = os.path.join(_ROOT, "specs", "003-grow-the-live-populations", "regimes.md")


# ------------------------------------------------------------------ the readout under test
def readout_geometric_mid(clock, tau):
    """R(clock, tau) = the population's own noise-free readout.

    Delegates to `TimeCellClock.nominal_estimate` rather than re-deriving it: a probe carrying its
    own copy of the readout it is judging can pass while the live one is broken, which is the exact
    shape of blindness the eligibility gate below exists to catch. One mechanism (P2)."""
    return clock.nominal_estimate(tau)


def readout_blind_ramp(clock, tau):
    """A readout that ignores the population entirely -- the negative control for the eligibility
    self-test. Kept here so `tests/test_timing_heldout.py` can assert the instrument refuses it."""
    return tau


# ------------------------------------------------------------------ the interval universe
def interval_universe(thresholds):
    """(teach, test, n_gaps_the_test_set_touches) over the population's own ladder.

    The grid itself is `cortex.time_cells.interval_grid` -- it is a property of the ladder, not of
    this probe. At n=64: 630 intervals, 504 teach and 126 test, all 63 gaps represented."""
    teach, test = interval_grid(thresholds)
    gaps_hit = len({bisect.bisect_right(thresholds, t) - 1 for t in test})
    return teach, test, gaps_hit


def rmse(clock, intervals, readout):
    if not intervals:
        raise ValueError("refusing to compute an RMSE over an empty interval set")
    total = sum((readout(clock, tau) - tau) ** 2 for tau in intervals)
    return math.sqrt(total / len(intervals))


# ------------------------------------------------------------------ admission gates on R
def eligible(readout, width):
    """FR-005 applied to the instrument itself. Returns (ok, reason).

    Two independent ways a readout can be blind, and both have been seen: it can ignore the
    thresholds (perturbation test), or it can read something that merely correlates with elapsed
    time -- the spike-tick anchor scored err 0.995 on a ladder carrying no information at all, which
    is why the degenerate control exists as a separate gate."""
    plain = TimeCellClock(n_cells=width, seed=0)
    poked = TimeCellClock(n_cells=width, seed=0)
    mid = width // 2
    poked.thresholds[mid] += 0.4
    probe_tau = poked.thresholds[mid]
    if readout(plain, probe_tau) == readout(poked, probe_tau):
        return False, "readout output is unchanged by a perturbed threshold"

    degenerate = TimeCellClock(n_cells=width, seed=0)
    degenerate.thresholds = [1.0 + 1e-6 * i for i in range(width)]
    if abs(readout(degenerate, 50.0) - 50.0) < 5.0:
        return False, "readout PASSES the degenerate-ladder control (it is not reading the ladder)"
    return True, ""


# ------------------------------------------------------------------ the arms
def teach_clock(seed, width, teach_set):
    """Drive one clock through the teach intervals the way brain.py drives it: reset at the cue,
    tick out the interval, teach at the reward. The clock's own RNG is the only source of variance
    across seeds, which is what makes the paired statistic meaningful."""
    clock = TimeCellClock(n_cells=width, seed=seed)
    for tau in teach_set:
        clock.reset()
        for _ in range(max(1, int(round(tau)))):
            clock.tick()
        clock.learn_interval(tau, reward=1.0)
    clock.reset()
    return clock


def _gap_of(clock, tau):
    j = bisect.bisect_right(clock.thresholds, tau) - 1
    return j if 0 <= j < clock.n_cells - 1 else None


def teach_residual_gap(clock, teach_set, test_set, readout):
    """The gap whose teach intervals the population already fits BEST -- the null arm's target.

    Inserting where nothing is wrong is the control: if a cell added there improves held-out RMSE by
    as much as one added at the widest gap, the main arm is measuring 'any extra cell helps' rather
    than contribution, and FR-006 forbids reporting that as earning.

    CANDIDATES ARE RESTRICTED TO GAPS THAT HOLD AT LEAST ONE TEST INTERVAL, and that restriction is
    the whole difference between a control and a decoration. Without it this function returned gaps
    like [3.3152, 3.3597) -- MEASURED across seeds 0/1/2, zero test intervals inside -- so the null
    arm scored exactly +0.000000 with 12 of 12 ties because NOTHING MEASURED IT, not because the
    insertion was worthless. A control whose zero is produced by the instrument's own blindness is
    worse than no control: it launders the main arm's result. Returns None if no measurable gap
    exists, and the caller must refuse rather than report."""
    measurable = {j for j in (_gap_of(clock, tau) for tau in test_set) if j is not None}
    per_gap = {}
    for tau in teach_set:
        j = _gap_of(clock, tau)
        if j is not None and j in measurable:
            per_gap.setdefault(j, []).append((readout(clock, tau) - tau) ** 2)
    if not per_gap:
        return None
    return min(per_gap, key=lambda j: sum(per_gap[j]) / len(per_gap[j]))


def paired_stats(diffs):
    """mean / sd / se / wins / ties over per-seed differences. Ties are counted, never suppressed."""
    n = len(diffs)
    if n < 2:
        raise ValueError("refusing to report a dispersion over fewer than 2 seeds")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    wins = sum(1 for d in diffs if d < 0.0)
    ties = sum(1 for d in diffs if d == 0.0)
    ratio = abs(mean / se) if se > 0.0 else float("inf") if mean != 0.0 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "se": se, "wins": wins, "ties": ties, "ratio": ratio}


def run_arm(seed, width, teach_set, test_set, readout, mode):
    """(ungrown_rmse, grown_rmse) for one seed under one insertion policy."""
    clock = teach_clock(seed, width, teach_set)
    before = rmse(clock, test_set, readout)
    if mode == "null":
        j = teach_residual_gap(clock, teach_set, test_set, readout)
        if j is None:
            raise ValueError("no gap holds a test interval; the control cannot be measured")
        added = clock.grow("interior", gap=j)
    else:
        added = clock.grow(mode)
    if added is None:
        return before, before, False
    return before, rmse(clock, test_set, readout), True


# ------------------------------------------------------------------ the saturation census
def regimes(seeds, ticks):
    """T018 / SC-003: what fraction of live clock-advancing ticks are UNSATURATED?

    Every later claim has to name the regime it was measured in, because the saturated branch of the
    readout is a ceiling rather than an estimate -- an effect measured entirely inside it would be
    an artefact of where the ceiling sits."""
    from snn_data_lake.cortex.brain import make_brain

    rows = []
    for seed in range(seeds):
        brain = make_brain(seed=seed, persist=False)
        clock = brain._time_clock
        if clock is None:
            brain.close()
            raise RuntimeError("brain constructed without a time clock; nothing to census")
        unsat = sat = 0
        for _ in range(ticks):
            brain.tick()
            if clock.n_fired >= clock.n_cells:
                sat += 1
            else:
                unsat += 1
        brain.close()
        rows.append((seed, unsat, sat, clock.n_cells))
    return rows


def write_regimes(rows, ticks, path=_REGIMES_MD):
    total_u = sum(r[1] for r in rows)
    total_s = sum(r[2] for r in rows)
    total = total_u + total_s
    lines = [
        "# Saturation regime census (T018, SC-003)",
        "",
        "Every claim this feature makes must name the regime it was measured in. The readout's",
        "saturated branch is a RANGE CEILING, not an estimate: once every cell has fired the",
        "population holds no further information, so an effect measured inside that branch is an",
        "artefact of where the ceiling sits and not of how the population tiles time.",
        "",
        f"Live brain via `make_brain(persist=False)`, {len(rows)} seeds x {ticks} ticks.",
        "",
        "| seed | width | unsaturated ticks | saturated ticks | unsaturated share |",
        "|---|---|---|---|---|",
    ]
    for seed, unsat, sat, width in rows:
        share = unsat / (unsat + sat) if (unsat + sat) else 0.0
        lines.append(f"| {seed} | {width} | {unsat} of {unsat + sat} | {sat} | {share:.4f} |")
    share = total_u / total if total else 0.0
    lines += [
        f"| **all** | | **{total_u} of {total}** | **{total_s}** | **{share:.4f}** |",
        "",
        f"**Verdict**: {share:.1%} of live clock-advancing ticks are unsaturated. The held-out",
        "instrument evaluates the ladder noise-free across the population's own gaps, so it reports",
        "the UNSATURATED regime; nothing here licenses a claim about the saturated one.",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return share


# ------------------------------------------------------------------ the durable-footprint census
def bounding(ticks, seed=0):
    """T049 / FR-014: how many bytes does autonomous growth commit per unit time, and does it stop?

    Runs the live brain with NORD_TIME_CELLS_GROW=1 against a throwaway NORD_HOME, so the operator's
    durable state is never touched. Reports growths, refusals, final width, and the persisted blob's
    size -- which is the whole footprint, because `save_state` OVERWRITES: there is no append, so
    there is no retention constant to tune and no unbounded tail to prune."""
    import tempfile

    from snn_data_lake.substrate import home

    prior_home = os.environ.get("NORD_HOME")
    prior_flag = os.environ.get("NORD_TIME_CELLS_GROW")
    work = tempfile.mkdtemp(prefix="tc_bounding_")
    try:
        os.environ["NORD_HOME"] = work
        os.environ["NORD_TIME_CELLS_GROW"] = "1"
        from snn_data_lake.cortex.brain import make_brain

        brain = make_brain(seed=seed, persist=True)
        clock = brain._time_clock
        w0 = clock.n_cells
        for _ in range(ticks):
            brain.tick()
        st = brain.state()
        brain.close()
        path = home.state_path("time_cells.json")
        size = os.path.getsize(path) if os.path.exists(path) else 0
        return {
            "ticks": ticks,
            "width_start": w0,
            "width_end": st["time_cells_width"],
            "grown": st["time_cells_grown"],
            "refused": st["time_cells_refused"],
            "failed": st["time_cells_failed"],
            "bytes": size,
        }
    finally:
        for key, val in (("NORD_HOME", prior_home), ("NORD_TIME_CELLS_GROW", prior_flag)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=12, help="paired seeds for the A/B (>=2)")
    ap.add_argument("--width", type=int, default=64, help="population width under test")
    ap.add_argument("--regimes", action="store_true", help="run the saturation census instead")
    ap.add_argument("--regime-seeds", type=int, default=3)
    ap.add_argument("--regime-ticks", type=int, default=600)
    ap.add_argument("--regimes-out", default=_REGIMES_MD)
    ap.add_argument("--bounding", action="store_true", help="durable-footprint census (FR-014)")
    ap.add_argument("--bounding-ticks", type=int, default=600)
    args = ap.parse_args(argv)

    if args.bounding:
        row = bounding(args.bounding_ticks)
        print(
            f"bounding: {row['ticks']} ticks -> width {row['width_start']} to {row['width_end']}, "
            f"grown {row['grown']}, refused {row['refused']}, failed {row['failed']}, "
            f"durable {row['bytes']} bytes (overwrite, not append)"
        )
        return 0

    if args.regimes:
        rows = regimes(args.regime_seeds, args.regime_ticks)
        share = write_regimes(rows, args.regime_ticks, args.regimes_out)
        print(f"saturation census: {share:.4f} unsaturated -> {args.regimes_out}")
        return 0

    if args.seeds < 2:
        print("NOT-ELIGIBLE: a dispersion needs at least 2 seeds; refusing to invent one")
        return 2

    readout = readout_geometric_mid
    ok, why = eligible(readout, args.width)
    print("=" * 78)
    print("HELD-OUT A/B -- does one inserted time cell earn its place?")
    print("=" * 78)
    if not ok:
        print(f"NOT-ELIGIBLE: {why}")
        print("no verdict is issued; a probe that cannot ask the question must not answer it")
        return 2

    probe = TimeCellClock(n_cells=args.width, seed=0)
    teach_set, test_set, gaps_hit = interval_universe(probe.thresholds)
    n_gaps = args.width - 1
    if not teach_set or not test_set:
        print("NOT-ELIGIBLE: the split left one side empty")
        return 2

    regime = "unsaturated (noise-free ladder evaluation; see specs/003/regimes.md)"
    print("readout      : geometric-mid population readout, eligible (perturbation + degenerate)")
    print("split        : gap-major grid, positional 1-in-5 stride, no RNG")
    print(
        f"               {len(teach_set) + len(test_set)} intervals -> "
        f"{len(teach_set)} of {len(teach_set) + len(test_set)} teach, "
        f"{len(test_set)} of {len(teach_set) + len(test_set)} test"
    )
    print(f"               test coverage {gaps_hit} of {n_gaps} gaps")
    print(f"regime       : {regime}")
    print("control      : null arm inserts at the LOWEST-teach-residual gap")
    print("-" * 78)

    results, per_seed = {}, {}
    for mode, label in (("interior", "main (widest log-gap)"), ("null", "null-control")):
        diffs, refused = [], 0
        for seed in range(args.seeds):
            before, after, grew = run_arm(seed, args.width, teach_set, test_set, readout, mode)
            if not grew:
                refused += 1
            diffs.append(after - before)
        st = paired_stats(diffs)
        results[mode] = (st, refused)
        per_seed[mode] = diffs
        print(f"{label:24s} mean d = {st['mean']:+.6f}  sd {st['sd']:.6f}  se {st['se']:.6f}")
        print(
            f"{'':24s} improves {st['wins']} of {st['n']}   "
            f"ties {st['ties']} of {st['n']}   refusals {refused} of {st['n']}"
        )
        print(
            f"{'':24s} |mean/se| = {st['ratio']:.2f}   "
            f"(2.0 bar {'MET' if st['ratio'] >= 2.0 else 'not met'}, "
            f"3.0 bar {'MET' if st['ratio'] >= 3.0 else 'not met'})"
        )
    print("-" * 78)

    main_st, _ = results["interior"]
    null_st, _ = results["null"]
    # THE CONTRAST IS THE VERDICT, not either arm against zero. With a noise-free readout ANY
    # measurable insertion moves some bracket, so a control that is required to be INDISTINGUISHABLE
    # FROM ZERO can never pass once it is placed where the metric can see it -- and one placed where
    # the metric CANNOT see it scores a fake zero (that was this instrument's first result: the
    # lowest-residual gap held 0 test intervals, so the control tied 12 of 12 for reasons that had
    # nothing to do with growth). The question FR-006 actually asks is whether the benefit is
    # attributable to PLACEMENT, and that is a paired contrast between the two arms at the same
    # seeds. Bars are the repo's ported 2.0 and 3.0; no new constant.
    contrast = paired_stats(
        [m - n for m, n in zip(per_seed["interior"], per_seed["null"], strict=True)]
    )
    print(
        f"{'main - null (contrast)':24s} mean d = {contrast['mean']:+.6f}  "
        f"sd {contrast['sd']:.6f}  se {contrast['se']:.6f}"
    )
    print(
        f"{'':24s} main better {contrast['wins']} of {contrast['n']}   "
        f"ties {contrast['ties']} of {contrast['n']}   "
        f"|mean/se| = {contrast['ratio']:.2f} "
        f"(2.0 bar {'MET' if contrast['ratio'] >= 2.0 else 'not met'}, "
        f"3.0 bar {'MET' if contrast['ratio'] >= 3.0 else 'not met'})"
    )
    print("-" * 78)

    if main_st["mean"] < 0.0 and contrast["mean"] < 0.0 and contrast["ratio"] >= 2.0:
        print(
            f"VERDICT: the inserted cell EARNS its place. Held-out RMSE falls {-main_st['mean']:.6f} "
            f"at the widest log-gap against {-null_st['mean']:.6f} at the best-fitting one; the "
            f"paired contrast is {contrast['mean']:+.6f} at |mean/se| {contrast['ratio']:.2f}, so "
            f"the benefit is attributable to WHERE the cell went, not to having one more cell."
        )
        return 0
    if main_st["mean"] >= 0.0 or main_st["ratio"] < 2.0:
        print(
            f"VERDICT: NULL. Main arm {main_st['mean']:+.6f} at |mean/se| {main_st['ratio']:.2f} "
            f"({main_st['wins']} of {main_st['n']} improve, {main_st['ties']} of {main_st['n']} "
            f"tie). The effect is not distinguishable from its own dispersion."
        )
        return 1
    print(
        f"VERDICT: NULL. The main arm improved ({main_st['mean']:+.6f}) but the control improved "
        f"comparably ({null_st['mean']:+.6f}); the contrast is {contrast['mean']:+.6f} at "
        f"|mean/se| {contrast['ratio']:.2f}. This is a benefit any added cell receives, which "
        f"FR-006 forbids reporting as contribution."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
