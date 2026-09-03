"""The world-red instrument (specs/022 PLAN-FINAL Phase 1 W3, P8 invention register #9).

The ONE place the question "does today's world still carry structure the brain's own
telemetry can see?" is measured. Phase 1's reds W-R1 and W-R2 are scored against what this
records; the plan states the order as gate-before-rule -- W3 lands before W1 touches the
world.

WHY THIS EXISTS: rank 2c refused a build on a measurement (evidence/s2c-rotating-stream-
refused.md). The audit's red assumed the |LP| facade the growth stop reads was at a floor;
the facade was ALIVE (about 190 distinct values per 200 wake rows) and the floor belonged to
a different chain entirely. A roadmap number is a claim about an instrument. This module is
that instrument, promoted out of the two throwaway probes archived beside the memory store
(archive/audit-2026-09-01/rank2c/) so the next such claim is re-derived, never quoted.

TWO CHAINS, MEASURED SEPARATELY AND NEVER AVERAGED (learning_progress.py's own docstring
records that the spiking drive was MEASURED INVALID as an |LP| basis, so these are two
mechanisms, not two views of one):
  * Chain A -- the arm's spiking curiosity over the _safe_ctx context: state()
    ["curiosity_mag"], Brain._lp_mean, state()["neuromod_acetylcholine"].
  * Chain B -- the rate-drive |LP| facade fed percept surprise, the signal GrowthStop reads:
    state()["learning_progress"]["cortex"].

Modes
  (bare)         the run_selfchecks path: a CHEAP wiring check on synthetic series. It
                 proves every criterion below can FAIL before any of them is believed, and
                 it never constructs a brain -- anything that observes the brain must be
                 cheap enough not to become the load (POLICIES cross-cutting).
  --self-check   the same criteria PLUS one scratch brain, to prove the five telemetry
                 reads still resolve against the live surface. Measured cost is dominated by
                 brain construction; the tick budget is deliberately small (--check-ticks).
  --record       the sweep that produces Phase 1's pre-registered baselines: per seed, the
                 last WINDOW wake rows at each checkpoint, plus category entropy and the
                 refit derivation. Hours, not seconds -- never wired into a gate.

WHAT IS REPORTED, AND THE THREE DECISIONS BEHIND IT

  1. ROUNDING IS LOAD-BEARING AND IS STATED. The two in-tree liveness idioms disagree --
     autonomous_loop.py::main rounds to 6 dp before set(), tools/ab_connectome_delay.py::
     _regime does not round at all -- and state() itself rounds curiosity_mag and
     acetylcholine to 4 dp while leaving the facade unrounded. Every channel is therefore
     reported as `distinct_raw / distinct_round` at 4 dp, the form s2c's tables already use,
     and THE LIVENESS VERDICT KEYS ON THE ROUNDED COUNT. An unrounded distinct-count over
     float telemetry inflates trivially: s2c measured channels at 200 / 2, meaning all 200
     rows differed at full precision and collapsed to two values at 4 dp. The raw count
     would have called that chain alive. Criterion 6 of the wiring check exists to keep this
     decision falsifiable.

  2. ABSENCE IS NOT ZERO. state()["learning_progress"] is {} until Brain._perceive first
     succeeds, and LearningProgress.lp() RAISES for an unfed channel rather than returning
     0.0 -- deliberately. The facade is read with .get("cortex"), a missing channel is
     recorded as None and counted in `missing`, and it never enters a mean. Likewise
     _last_res["unit"] is None on the reflex/E-STOP branch: those ticks are COUNTED in the
     entropy row's `none` field, never silently dropped.

  3. THE REFIT COUNT IS DERIVED, NOT INFERRED. NeurogenesisSelfImprove exposes no refit
     counter -- confirmed in-process, not only on the daemon's wire. Two derivations exist
     and both are reported: SCHEDULED calls (engine.ticks // engine.refit_every, exact,
     because tick() increments then tests the modulus) and EFFECTIVE calls
     (len(engine.trajectory), appended exactly once per _refit that passed its data guard).
     Their DIFFERENCE is itself the finding: it counts refits that early-returned on too few
     holdout rows or fewer than two categories. rank 2b's "no refit" reading of the live
     daemon was an INFERENCE with no control; this is arithmetic over two counters.

DISCLOSED HOLE, said here rather than found later: neither derivation survives a lake
restore. NeurogenesisSelfImprove.restore_growth_state REPLACES trajectory with the persisted
tail and baselines with a single-element list, so a probe reading them after a restore gets a
small number that is not a refit count. Every mode here runs persist=False under a throwaway
home, where no restore can occur, and _assert_scratch_home refuses to run otherwise -- but
the numbers must not be lifted onto a restored brain.

PLACEMENT: substrate/, not tools/ -- the p4c ratchet refuses +1 ungoverned file in tools/,
and an instrument is part of the brain (P4c). Same deviation port_charge_probe and
perf_baseline recorded before it.
"""

# distinct-from: substrate/perf_baseline.py -- measures the tick's COST (wake/sleep p50/p95,
#   disk IOPS) and ratchets it; this measures what the tick's telemetry CARRIES. Neither
#   subsumes the other: a tick can get faster while its signals go flat.
# distinct-from: substrate/port_charge_probe.py -- attributes a consumer's divergence to ONE
#   declared port by pinning its payload. This asks a prior question with no neuter at all:
#   does the signal vary before anyone asks who consumes it.
# distinct-from: tools/charge_probe.py -- the module-level neuter A/B. Same reason.
# distinct-from: basal_ganglia/learning_progress.py -- the Chain B PRODUCER this reads; it
#   computes |LP|, this only observes it. All sites read.

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys

WINDOW = 200  # wake rows summarised at each checkpoint (s2c's window, kept for comparability)
ENTROPY_WAKE = 1000  # wake rows in the category-entropy window
ND = 4  # the rounding the liveness verdict keys on (decision 1 above)
CHECKPOINTS = (3000, 6000)  # tick() calls, not wake rows (the plan's W-R1 wording)

#: The gradient-claim census captures `refit_counts` on the `fit` substring in its name and
#: cannot decide it, which lands as a governed UNDETERMINABLE -- a ratcheted debt it is not.
#: Declared rather than renamed: "refit" is the engine's own word for the batch solve this
#: counts, and renaming a function to slip past a detector is avoidance, not a disposition.
GRADIENT_CLAIM = {
    "refit_counts": (
        "NOT_A_RULE",
        "integer arithmetic over two counters the neurogenesis engine already keeps "
        "(ticks // refit_every, and len(trajectory)). It reports how often a batch solve RAN; it "
        "adapts nothing, holds no state and touches no weight. The solve it counts is "
        "NeurogenesisSelfImprove._refit, which carries its own disposition",
    ),
}


def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def series_liveness(values, nd=ND):
    """Liveness of one telemetry channel over a window of wake rows.

    `values` may contain None (a channel with no feed yet). None is COUNTED in `missing` and
    excluded from every statistic -- absence is not zero. `live` keys on the ROUNDED distinct
    count and on adjacent changes, never on the raw count (decision 1).
    """
    fin = [v for v in values if v is not None]
    rounded = [round(v, nd) for v in fin]
    changes = sum(1 for a, b in zip(rounded, rounded[1:]) if a != b)
    return {
        "n": len(values),
        "missing": len(values) - len(fin),
        "distinct_raw": len(set(fin)),
        "distinct_round": len(set(rounded)),
        "changes": changes,
        "mean": statistics.fmean(fin) if fin else None,
        "last": fin[-1] if fin else None,
        "live": bool(fin) and len(set(rounded)) > 1 and changes > 0,
    }


def category_entropy(cats):
    """Shannon entropy (bits) of the arm learner's per-tick prototype category.

    `cats` may contain None -- AutonomousArm.step returns {"unit": None} on the reflex /
    E-STOP branch. Those ticks are counted in `none` and excluded from the distribution; a
    run whose categories are mostly None is a run about the latch, not about the world.
    """
    none_n = sum(1 for c in cats if c is None)
    counts = collections.Counter(c for c in cats if c is not None)
    n = sum(counts.values())
    # len(counts) > 1 rather than n > 0: a single category gives -0.0 from the negated sum, and a
    # negative zero in a report invites a reader to look for a sign error that is not there.
    bits = -sum((k / n) * math.log2(k / n) for k in counts.values()) if len(counts) > 1 else 0.0
    return {
        "n": n,
        "none": none_n,
        "distinct": len(counts),
        "entropy_bits": bits,
        "max_bits": math.log2(len(counts)) if counts else 0.0,
        "dist": dict(sorted(counts.items())),
    }


def refit_counts(ticks, refit_every, n_trajectory):
    """The two derivations of the refit count, and the gap between them (decision 3)."""
    scheduled = ticks // refit_every if refit_every else 0
    return {
        "ticks": ticks,
        "refit_every": refit_every,
        "scheduled": scheduled,
        "effective": n_trajectory,
        "skipped_on_data_guard": scheduled - n_trajectory,
    }


def wake_row(b):
    """One wake row: the five observables, read the safe way (decision 2)."""
    st = b.state()
    lp = st.get("learning_progress") or {}
    res = getattr(b, "_last_res", None) or {}
    return {
        "cur_mag": st.get("curiosity_mag"),
        "lp_mean": getattr(b, "_lp_mean", None),
        "ach": st.get("neuromod_acetylcholine"),
        "facade": lp.get("cortex"),
        "unit": res.get("unit"),
    }


CHANNELS = ("cur_mag", "lp_mean", "ach", "facade")


def summarise(rows, window=WINDOW, entropy_wake=ENTROPY_WAKE):
    """Summarise a list of wake rows: per-channel liveness over the last `window`, category
    entropy over the FIRST `entropy_wake` rows (the window the pre-registered baseline uses,
    fixed so a treatment cannot pick a friendlier slice) and over the LAST `entropy_wake`."""
    tail = rows[-window:]
    out = {"wake_rows": len(rows), "window": len(tail)}
    for ch in CHANNELS:
        out[ch] = series_liveness([r[ch] for r in tail])
    out["entropy_first"] = category_entropy([r["unit"] for r in rows[:entropy_wake]])
    out["entropy_last"] = category_entropy([r["unit"] for r in rows[-entropy_wake:]])
    return out


def _assert_scratch_home():
    """Refuse to run against the live lake. The pre-mortem gate: a probe that appends to
    brain_lake.bin has contaminated the subject it measures (P4b's honesty guard, inverted)."""
    import pathlib

    from snn_data_lake.substrate.home import home

    live = pathlib.Path.home() / ".snn_data_lake"
    here = pathlib.Path(str(home())).resolve()
    assert here != live.resolve(), (
        f"world_red_probe refuses to run against the live lake home {here} -- "
        "wrap the call in substrate.home.throwaway_home()"
    )


def run_seed(seed, checkpoints=CHECKPOINTS, window=WINDOW, entropy_wake=ENTROPY_WAKE):
    """Tick ONE scratch brain to max(checkpoints) and summarise at each checkpoint.

    Caller owns the throwaway home. persist=False: this measures the world, not a restore.
    """
    from snn_data_lake.cortex.brain import make_brain

    _assert_scratch_home()
    b = make_brain(seed=seed, persist=False)
    try:
        rows, marks, last = [], {}, max(checkpoints)
        for t in range(1, last + 1):
            # The checkpoint is taken on tick t whether or not t was awake. Guarding it behind
            # the wake test is a MEASURED defect, not a hypothetical: the wake fraction is about
            # 0.49, so roughly half of all checkpoints land on a sleep tick, and a skipped
            # checkpoint silently reported the state at the END of the run instead. Two
            # checkpoints printing identical rows is that bug's tell -- it is what caught it.
            if b.tick() == "awake":
                rows.append(wake_row(b))
            if t in checkpoints:
                marks[t] = _mark(b, rows, window, entropy_wake)
        assert set(marks) == set(checkpoints), (
            f"every checkpoint must be marked at its own tick: {sorted(marks)} != {sorted(checkpoints)}"
        )
        seen = [marks[t]["wake_rows"] for t in sorted(marks)]
        assert all(a < b for a, b in zip(seen, seen[1:])), (
            f"wake rows must STRICTLY rise between checkpoints (got {seen}) -- equal counts mean a "
            "later checkpoint's state was reported for an earlier one, the defect this assert names"
        )
        return {"seed": seed, "checkpoints": marks}
    finally:
        b.close()


def _mark(b, rows, window, entropy_wake):
    eng = getattr(b, "neuro", None)
    mark = summarise(rows, window, entropy_wake)
    mark["refit"] = (
        refit_counts(
            getattr(eng, "ticks", 0),
            getattr(eng, "refit_every", 0),
            len(getattr(eng, "trajectory", ())),
        )
        if eng is not None
        else {"engine": "ABSENT"}
    )
    return mark


def _print_mark(seed, t, m):
    _emit(f"  seed {seed} @ {t} ticks: {m['wake_rows']} wake rows, window {m['window']}")
    for ch in CHANNELS:
        s = m[ch]
        mean = "None" if s["mean"] is None else f"{s['mean']:.5f}"
        _emit(
            f"    {ch:8s} distinct raw/{ND}dp = {s['distinct_raw']}/{s['distinct_round']}"
            f"  changes {s['changes']}  missing {s['missing']}  mean {mean}"
            f"  {'LIVE' if s['live'] else 'AT FLOOR'}"
        )
    for key in ("entropy_first", "entropy_last"):
        e = m[key]
        _emit(
            f"    {key:14s} n {e['n']} none {e['none']} distinct {e['distinct']}"
            f"  H {e['entropy_bits']:.4f} / max {e['max_bits']:.4f} bits  dist {e['dist']}"
        )
    r = m["refit"]
    _emit(f"    refit          {r}")


# ---------------------------------------------------------------------------
# The wiring check: every criterion above, proven able to fail, on synthetic
# series, in one command. No brain, no disk. Bare invocation runs exactly this.
# ---------------------------------------------------------------------------


def _wiring_check():
    # 1 RED -- a pinned constant must read AT FLOOR. This is the criterion the whole module
    #   rests on; the two in-tree idioms both pin a constant and assert distinct == 1 first.
    const = series_liveness([0.6] * 200)
    assert const["distinct_round"] == 1 and const["changes"] == 0 and not const["live"], (
        f"the RED control must read a pinned constant as dead ({const}) -- "
        "if it does not, this instrument cannot fail"
    )
    # 2 GREEN -- a varying channel must read LIVE, or the instrument fires on everything.
    ramp = series_liveness([i / 1000 for i in range(200)])
    assert ramp["distinct_round"] == 200 and ramp["changes"] == 199 and ramp["live"], (
        f"a varying channel must read live ({ramp})"
    )
    # 3 RED -- an unfed channel is MISSING, not 0.0 (decision 2). A mean of 0.0 here would
    #   put an absent facade and a silent one on the same row.
    absent = series_liveness([None] * 200)
    assert absent["missing"] == 200 and absent["mean"] is None and not absent["live"], (
        f"an unfed channel must be missing, never zero ({absent})"
    )
    # 4 -- entropy: exact on a uniform alphabet, zero on one category, None counted apart.
    uni = category_entropy([0, 1, 2, 3] * 50)
    one = category_entropy([7] * 200)
    withnone = category_entropy([1, 2, None, None])
    assert abs(uni["entropy_bits"] - 2.0) < 1e-12 and uni["distinct"] == 4, (
        f"uniform k=4 is 2 bits ({uni})"
    )
    assert one["entropy_bits"] == 0.0 and one["max_bits"] == 0.0, f"one category is 0 bits ({one})"
    assert withnone["n"] == 2 and withnone["none"] == 2, (
        f"a None category is counted, never dropped ({withnone}) -- "
        "an E-STOP run must not read as a low-entropy world"
    )
    # 5 -- the refit derivation, including a planted data-guard gap.
    r = refit_counts(1016, 20, 47)
    assert r["scheduled"] == 50 and r["effective"] == 47 and r["skipped_on_data_guard"] == 3, (
        f"the refit gap must be arithmetic, not an inference ({r})"
    )
    assert refit_counts(100, 0, 0)["scheduled"] == 0, "refit_every 0 must not divide by zero"
    # 6 RED -- the 4 dp choice must BITE. A channel varying only below the rounding floor is
    #   raw-distinct and rounded-dead; s2c measured real channels at 200 / 2 exactly here.
    fine = series_liveness([0.5 + 1e-9 * i for i in range(200)])
    assert fine["distinct_raw"] == 200 and fine["distinct_round"] == 1 and not fine["live"], (
        f"the {ND} dp rounding must decide liveness, not the raw count ({fine}) -- "
        "if this passes, the reported distinct_raw would call a dead chain alive"
    )
    # 7 -- summarise() wires the parts together and slices the two entropy windows apart.
    rows = [
        {"cur_mag": 0.0, "lp_mean": None, "ach": i / 100, "facade": 0.5, "unit": i % 3}
        for i in range(50)
    ]
    s = summarise(rows, window=10, entropy_wake=5)
    assert s["wake_rows"] == 50 and s["window"] == 10, (
        f"window slicing ({s['wake_rows']}, {s['window']})"
    )
    assert s["cur_mag"]["live"] is False and s["ach"]["live"] is True, (
        "per-channel verdicts must differ"
    )
    assert s["lp_mean"]["missing"] == 10, "a missing channel stays missing through summarise"
    assert s["entropy_first"]["dist"] == {0: 2, 1: 2, 2: 1}, (
        f"first window ({s['entropy_first']['dist']})"
    )
    assert s["entropy_last"]["dist"] == {0: 2, 1: 2, 2: 1}, (
        f"last window ({s['entropy_last']['dist']})"
    )
    _emit(
        "world_red_probe wiring OK: 7 criteria proven falsifiable "
        f"(pinned constant dead, ramp live, absent != zero, entropy exact, refit gap arithmetic, "
        f"{ND}dp rounding decisive, summarise wired) -- no brain, no disk"
    )


def _brain_check(ticks):
    """The expensive arm: prove the five telemetry reads still resolve on a real brain."""
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _assert_scratch_home()
        from snn_data_lake.cortex.brain import make_brain

        b = make_brain(seed=0, persist=False)
        try:
            rows = [wake_row(b) for _ in range(ticks) if b.tick() == "awake"]
            assert rows, f"{ticks} ticks produced no wake row -- the probe cannot sample this brain"
            for ch in ("cur_mag", "ach"):
                assert all(isinstance(r[ch], float) for r in rows), (
                    f"{ch} must be float on every wake row"
                )
            eng = getattr(b, "neuro", None)
            assert eng is not None, "no neurogenesis engine: the refit derivation has no source"
            r = refit_counts(eng.ticks, eng.refit_every, len(eng.trajectory))
        finally:
            b.close()
    assert r["scheduled"] >= r["effective"] >= 0, f"a refit gap cannot be negative ({r})"
    s = summarise(rows, window=min(WINDOW, len(rows)), entropy_wake=len(rows))
    _emit(f"world_red_probe brain arm: {len(rows)} wake rows over {ticks} ticks")
    _print_mark(0, ticks, {**s, "refit": r})
    _emit(
        f"world_red_probe self-check OK: the wiring reds fire AND all five reads resolve on a "
        f"live brain ({len(rows)} wake rows, refit {r['scheduled']} scheduled / {r['effective']} effective)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the world-red instrument (specs/022 W3)")
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--self-check", dest="self_check", action="store_true")
    mode.add_argument("--record", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--checkpoints", default=",".join(str(c) for c in CHECKPOINTS))
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--entropy-wake", dest="entropy_wake", type=int, default=ENTROPY_WAKE)
    ap.add_argument("--check-ticks", dest="check_ticks", type=int, default=60)
    ap.add_argument("--out", default=None, help="write the sweep as JSON (outside the repo)")
    a = ap.parse_args(argv)

    _wiring_check()
    if a.self_check:
        _brain_check(a.check_ticks)
        return 0
    if not a.record:
        return 0

    from snn_data_lake.substrate.home import throwaway_home

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    checkpoints = tuple(int(c) for c in a.checkpoints.split(",") if c.strip())
    _emit(f"world_red_probe --record: seeds {seeds} checkpoints {checkpoints} window {a.window}")
    out = []
    for seed in seeds:
        with throwaway_home():  # one home per seed: no seed inherits another's lake
            res = run_seed(seed, checkpoints, a.window, a.entropy_wake)
        out.append(res)
        for t in sorted(res["checkpoints"]):
            _print_mark(seed, t, res["checkpoints"][t])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        _emit(f"world_red_probe wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
