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

# distinct-from: gates/perf_baseline.py -- measures the tick's COST (wake/sleep p50/p95,
#   disk IOPS) and ratchets it; this measures what the tick's telemetry CARRIES. Neither
#   subsumes the other: a tick can get faster while its signals go flat.
# distinct-from: instruments/port_charge_probe.py -- it too has a did-it-move criterion
#   (margin_frozen: an arbitrate-join margin whose range never exceeds tol), so "this one asks
#   whether the signal varies" is NOT the distinction and the panel was right to say so. The real
#   one is subject and statistic: margin_frozen is a boolean over ONE manifest-declared port's
#   join margins, beside a neuter that ATTRIBUTES divergence to that port; this is a per-channel
#   distinct/mean/entropy report over arbitrary Brain.state() telemetry with None handling and no
#   neuter anywhere. Neither can produce the other's number.
# distinct-from: tools/charge_probe.py -- the module-level neuter A/B. Same reason.
# distinct-from: growth/learning_progress.py -- the Chain B PRODUCER this reads; it
#   computes |LP|, this only observes it.
# distinct-from: growth/neuron_zoo.py::_worker_ladder -- flagged by the duplicate-capability
#   check against refit_counts on the shared words "counts" and "derived". Read: it builds a
#   ladder of candidate THREAD counts to race for a performance choice. Nothing in common but
#   the vocabulary. All sites read.

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import types

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
    excluded from every statistic -- absence is not zero.

    NON-FINITE values are counted in `nonfinite` and excluded the same way. Without that,
    NaN INVERTS the verdict: NaN != NaN, so a channel of nothing but NaN reports maximal
    distinct counts and maximal changes and reads LIVE -- a dead channel scoring as the most
    alive thing in the report. Found by the arc's review panel, not by the criteria below,
    which is why criterion 8 now carries it.

    `live` keys on the ROUNDED distinct count alone. It previously also required
    `changes > 0`; that conjunct was VACUOUS -- over a sequence, more than one distinct
    rounded value implies at least one adjacent change -- so it is dropped rather than left
    reading as a second, independent condition. `changes` is still reported, because the
    NUMBER separates a channel that stepped once from one that oscillates.
    """
    fin = [v for v in values if v is not None and math.isfinite(v)]
    rounded = [round(v, nd) for v in fin]
    changes = sum(1 for a, b in zip(rounded, rounded[1:]) if a != b)
    out = {
        "n": len(values),
        "missing": sum(1 for v in values if v is None),
        "nonfinite": sum(1 for v in values if v is not None and not math.isfinite(v)),
        "distinct_raw": len(set(fin)),
        "distinct_round": len(set(rounded)),
        "changes": changes,
        "mean": statistics.fmean(fin) if fin else None,
        "last": fin[-1] if fin else None,
        "live": len(set(rounded)) > 1,
    }
    assert out["distinct_round"] <= out["distinct_raw"], (
        f"rounding can only merge values, never split them ({out}) -- a rounded count above the "
        "raw count means a non-finite value survived the filter"
    )
    return out


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


# distinct-from: growth/neuron_zoo.py::_worker_ladder -- see the header block; that races
#   candidate thread counts, this counts refits of a batch solve.
def refit_counts(ticks, refit_every, n_trajectory):
    """The two derivations of the refit count, and the gap between them (decision 3).

    ATTRIBUTION LIMIT, because the field name invites more than it earns: at the engine's
    default buffer_cap of 1000 the only REACHABLE branch of `_refit`'s data guard is
    `n_cat < 2` (holdout is min(60, n // 3), so `holdout < 3` cannot hold, and
    `(n - holdout) < n_cat` would need more than 940 categories). A non-zero
    `skipped_on_data_guard` therefore means "the window went single-category", which is the
    SAME fact a collapsed category entropy reports -- not an independent confirmation of it.
    """
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


# distinct-from: brainstem/so101_arm.py::inject_disturbance -- flagged by the duplicate-capability
#   check on "stand in for". Read: it shoves a joint's qpos to stand in for an EXTERNAL FORCE on
#   the physical arm, exercising the deviation reflex. This stands in for a READER's data source.
#   Different subject, different direction, and that file is envelope-gated. All sites read.
def _telemetry_double(lp, last_res, lp_mean=0.5):
    """Brain's telemetry surface, reduced to what wake_row touches, so wake_row can be
    exercised without constructing a Brain. Three attributes and one method are the whole
    surface. It lives beside the reader it doubles for rather than inside the check."""
    b = types.SimpleNamespace(_lp_mean=lp_mean, _last_res=last_res)
    b.state = lambda: {
        "curiosity_mag": 0.1,
        "neuromod_acetylcholine": 0.2,
        "learning_progress": lp,
    }
    return b


CHANNELS = ("cur_mag", "lp_mean", "ach", "facade")


def summarise(rows, window=WINDOW, entropy_wake=ENTROPY_WAKE):
    """Summarise a list of wake rows: per-channel liveness over the last `window`, category
    entropy over the FIRST `entropy_wake` rows (the window the pre-registered baseline uses,
    fixed so a treatment cannot pick a friendlier slice) and over the LAST `entropy_wake`.

    The two entropy windows OVERLAP whenever the run has fewer than 2 * entropy_wake wake
    rows, and `entropy_overlap` reports by how many. It is not zero in practice: at the 3000
    tick checkpoint the sweep had about 1483 wake rows, so the two 1000-row windows shared
    about 517 of them. A first-vs-last CONTRAST is only clean where that number is 0.
    """
    tail = rows[-window:]
    out = {"wake_rows": len(rows), "window": len(tail)}
    for ch in CHANNELS:
        out[ch] = series_liveness([r[ch] for r in tail])
    out["entropy_first"] = category_entropy([r["unit"] for r in rows[:entropy_wake]])
    out["entropy_last"] = category_entropy([r["unit"] for r in rows[-entropy_wake:]])
    out["entropy_overlap"] = max(0, min(entropy_wake, len(rows)) * 2 - len(rows))
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
    from snn_data_lake.engine.brain import make_brain

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
                marks[t]["at_tick"] = t
        # IDENTITY, not a side effect. Two earlier guards were both weak and the panel said so:
        # `set(marks) == set(checkpoints)` cannot fail (last = max(checkpoints), so the loop
        # visits every checkpoint >= 1 and marks only ever takes keys from checkpoints), and
        # "wake rows must STRICTLY rise" fires spuriously on adjacent checkpoints and says
        # nothing at all with one checkpoint. This asserts the property actually at risk: each
        # mark was taken on ITS OWN tick. It fires with a single checkpoint too.
        assert all(marks[t]["at_tick"] == t for t in checkpoints), (
            f"each checkpoint's mark must be taken at its own tick: "
            f"{ {t: marks[t]['at_tick'] for t in checkpoints} }"
        )
        seen = [marks[t]["wake_rows"] for t in sorted(marks)]
        assert all(a <= b for a, b in zip(seen, seen[1:])), (
            f"wake rows cannot FALL between checkpoints (got {seen})"
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
    # 7 -- summarise() must slice THE RIGHT ROWS into each of its three windows.
    #   The first version of this criterion was VACUOUS and the arc's panel proved it with four
    #   surviving mutants: its fixture used unit = i % 3 with entropy_wake = 5, so the first five
    #   and last five units were the same multiset, and a channel varying uniformly could not
    #   tell a head slice from a tail slice either. "entropy_last aliases entropy_first" and "the
    #   channel window is the FIRST 200 rows" both passed green -- the two defects that would
    #   have silently relocated every number in the evidence file. The fixture is now ASYMMETRIC
    #   in both dimensions: categories differ head-vs-tail, and `ach` is flat over the head and
    #   varying over the tail, so a window swap flips a verdict.
    rows = [
        {
            "cur_mag": 0.0,
            "lp_mean": None,
            "ach": 0.0 if i < 40 else i / 100,
            "facade": 0.5,
            "unit": 0 if i < 25 else 1,
        }
        for i in range(50)
    ]
    s = summarise(rows, window=10, entropy_wake=5)
    assert s["wake_rows"] == 50 and s["window"] == 10, (
        f"window slicing ({s['wake_rows']}, {s['window']})"
    )
    assert s["entropy_first"]["dist"] == {0: 5} and s["entropy_last"]["dist"] == {1: 5}, (
        f"the two entropy windows must come from DIFFERENT rows "
        f"(first {s['entropy_first']['dist']}, last {s['entropy_last']['dist']}) -- equal "
        "distributions here is what made this criterion vacuous"
    )
    assert s["ach"]["live"] is True and s["ach"]["last"] == rows[-1]["ach"], (
        f"the channel window must be the TAIL ({s['ach']}) -- `ach` is flat over the head, so a "
        "head or middle slice reads AT FLOOR and a wrong `last` names the wrong row"
    )
    assert s["cur_mag"]["live"] is False, "a constant channel must still read at floor here"
    assert s["lp_mean"]["missing"] == 10, "a missing channel stays missing through summarise"
    assert s["entropy_overlap"] == 0, f"50 rows, window 5: no overlap ({s['entropy_overlap']})"
    assert summarise(rows, window=10, entropy_wake=30)["entropy_overlap"] == 10, (
        "50 rows, window 30: the two entropy windows share 10 rows and must SAY so"
    )
    # 8 RED -- NaN must not INVERT the verdict. NaN != NaN, so an unfiltered NaN channel reads
    #   maximally distinct and maximally changing: the deadest possible channel scoring as the
    #   most alive thing in the report. Found by the panel, not by criteria 1-7.
    nan = series_liveness([float("nan")] * 200)
    assert nan["nonfinite"] == 200 and nan["mean"] is None and not nan["live"], (
        f"a NaN channel must read dead, not maximally live ({nan})"
    )
    mixed = series_liveness([1.0, float("inf"), None, 1.0])
    assert mixed["nonfinite"] == 1 and mixed["missing"] == 1 and mixed["distinct_raw"] == 1, (
        f"non-finite and absent are counted apart, and neither enters a statistic ({mixed})"
    )
    # 9 -- wake_row IS decision 2 ("absence is not zero"), and criterion 3 never reached it: it
    #   tested series_liveness on a hand-built [None] list, so inverting the decision INSIDE
    #   wake_row left the whole check green (the panel demonstrated exactly that). _telemetry_double
    #   reaches it with no Brain and no measurable cost.
    unfed = wake_row(_telemetry_double({}, {"unit": None}))
    assert unfed["facade"] is None and unfed["unit"] is None, (
        f"an unfed facade must read None, never 0.0 ({unfed}) -- decision 2 at its only "
        "implementation"
    )
    fed = wake_row(_telemetry_double({"cortex": 0.5}, {"unit": 3}))
    assert fed["facade"] == 0.5 and fed["unit"] == 3 and fed["lp_mean"] == 0.5, (
        f"a fed facade must carry its value through ({fed})"
    )
    # 10 -- the refit arithmetic on its BOUNDARY, where an off-by-one phase error lives.
    assert refit_counts(1000, 20, 47)["scheduled"] == 50, "1000 ticks at every 20 is 50 refits"
    assert refit_counts(999, 20, 47)["scheduled"] == 49, "999 is one short of the 50th refit"
    # 11 -- the reported derivations, pinned: `changes` counts ROUNDED steps, `mean` is UNROUNDED.
    step = series_liveness([0.5, 0.50001, 0.5])
    assert step["changes"] == 0, f"`changes` must count rounded steps, not raw ones ({step})"
    assert step["mean"] == statistics.fmean([0.5, 0.50001, 0.5]), (
        f"`mean` must be the unrounded mean ({step['mean']})"
    )
    _emit(
        "world_red_probe wiring OK: 11 criteria proven falsifiable "
        f"(pinned constant dead, ramp live, absent != zero, entropy exact, refit gap arithmetic, "
        f"{ND}dp rounding decisive, summarise slices the RIGHT rows, NaN dead not live, wake_row "
        "keeps absence, refit boundary, changes/mean derivations) -- no brain, no disk"
    )


def _brain_check(ticks):
    """The expensive arm: prove the five telemetry reads still resolve on a real brain."""
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _assert_scratch_home()
        from snn_data_lake.engine.brain import make_brain

        b = make_brain(seed=0, persist=False)
        try:
            rows = [wake_row(b) for _ in range(ticks) if b.tick() == "awake"]
            assert rows, f"{ticks} ticks produced no wake row -- the probe cannot sample this brain"
            # ALL FIVE reads, because the emitted line claims all five. It previously asserted
            # only cur_mag and ach while printing "all five reads resolve", and the panel drove
            # it green against a stub with lp_mean, the facade and unit all DETACHED. An arm
            # that names five and checks two is a harness that cannot fail on three of them.
            for ch in ("cur_mag", "ach", "lp_mean"):
                assert all(isinstance(r[ch], float) for r in rows), (
                    f"{ch} must be a float on every wake row -- read {[r[ch] for r in rows[:3]]}"
                )
            assert any(r["facade"] is not None for r in rows), (
                "the Chain B facade never resolved in this run: state()['learning_progress'] "
                "carried no 'cortex' key on any wake row"
            )
            assert any(r["unit"] is not None for r in rows), (
                "no wake row carried a prototype category: _last_res['unit'] was None throughout, "
                "which is the reflex/E-STOP shape, not a sampling failure"
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
