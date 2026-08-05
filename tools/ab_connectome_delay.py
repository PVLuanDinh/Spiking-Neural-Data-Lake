#!/usr/bin/env python3
"""A/B: does per-tract CONDUCTION DELAY variation change anything in the connectome router?

specs/010 follow-on. `connectome.py`'s `_e()` gained a `delay` field on 2026-08-04; before that every
long-range projection arrived in a uniform 1 tick. The router (`thalamus/connectome_router.py`,
`ConnectomeRouter.__init__`) was already built to consume per-edge delays and to size a ring buffer
to the longest one -- it had just never been handed a value.

THIS SCRIPT EXISTS TO BE ABLE TO SAY NO. Per P8 a criterion with no reachable RED is worthless, and
per P3 a wire that does not charge is REVERTED, not kept.

CORRECTED 2026-08-05, AND THE CORRECTION IS THE POINT. The first version of this file asked one
question and reported the answer to a different one. It compared `uniform(1)` against a VARIED table
drawn from [1, 6] -- which moves the delay MEAN from 1.0 to ~3.7 at the same time as it introduces
spread. Its headline observable ("26 of 40 first-arrival ticks MOVED") is reproduced almost exactly
by `uniform(3)`, a table with ZERO variation in it: 27 of 40. So the original comparison could not
come back negative-for-variation, because ANY delay change at all produces a positive. The treatment
confounded two variables and the verdict credited the wrong one.

The fix is one control: a UNIFORM delay MATCHED TO THE VARIED TABLE'S OWN MEAN. Magnitude is then
held constant and spread is the only thing that moves. Both questions are now answered separately,
because they have different consequences:

  VERDICT 1 -- does the `delay` FIELD carry anything at all?   (uniform(1) vs uniform(mean))
      This is the keep-or-revert question the field itself owes. A null here reverts the field.
  VERDICT 2 -- does per-tract VARIATION carry anything BEYOND magnitude?  (uniform(mean) vs varied)
      This is the question that bears on adopting LEARNED delays (DCLS-Delays, arXiv:2306.17670,
      whose entire claim is about the DISTRIBUTION of delays across connections). A null here says
      a single global delay constant buys everything the spread buys, and learning per-tract values
      would be fitting noise.

THE MEAN-MATCH IS NOT EXACT, AND SAYING SO IS THE POINT (added 2026-08-05). Delays are integers, so
"uniform matched to the varied table's mean" is a ROUNDED mean and a residual magnitude difference
survives: measured +0.309, -0.395, +0.259 and -0.321 ticks for spread seeds 0..3. That residual is
the same confound this file was corrected to remove, one order of magnitude smaller, and it is not
closed by asserting it is small. It is closed by BRACKETING: `_bracket_control` compares the varied
arm against BOTH neighbouring integer magnitudes, and the varied arm differs from floor(mean) AND
from ceil(mean) in 26 to 28 of the 40 drive regions on every swept seed. No single uniform magnitude
reproduces the varied arm, so the residual cannot be what VERDICT 2 is measuring.

WHAT IS COMPARED. Three observables, chosen because they are what a delay could plausibly move:
  1. per-region firing rate trajectory over the run (the raw signal)
  2. REVERBERATION -- activity persisting after drive stops, which is the loop-closure property the
     router exists to demonstrate
  3. ARRIVAL ORDER at convergence nodes -- the coincidence-detection observable. This is the one the
     biology predicts should move: delays are what make a target fire only when inputs CO-ARRIVE.
  plus RECRUITMENT / SILENCING -- regions that fire at all under one arm and never under the other.
     This is the strongest of the four because it is qualitative and cannot be produced by a latency
     shift. BOTH DIRECTIONS ARE REPORTED, added 2026-08-05: `_compare` measures one direction only
     (fires under B, silent under A), and this paragraph described it as symmetric while the code
     counted half of it. Measured on the VERDICT 2 sweep, the unreported half was not empty --
     20 recruited by variation and 10 SILENCED by it -- so a third of the qualitative events the
     harness's own headline rests on were being discarded by the observable that reports it.
     AND ITS CONTROL WAS ONE PAIR. "Magnitude cannot recruit" was read off VERDICT 1's incidental
     `recruited: 0` for the single pair uniform(1) vs uniform(4) -- an anecdote carrying the whole
     weight of VERDICT 2. `_magnitude_recruitment_control` now sweeps ALL 30 ordered uniform pairs
     in [1, 6] over all 40 drive regions: 1200 pairs, recruited 0, silenced 0. The null is a
     measurement and not a dead code path -- fed varied tables through the identical code path it
     returns 154 / 154.

FOUR INSTRUMENT DEFECTS ALSO FIXED, all of which suppressed the harness's ability to say no. (This
line read TWO while the list below carried four entries; the census of a gate's own defects is the
first number in it that goes stale, and `tools/instrument_guard.py` records being caught by exactly
that one level up.)

  * THE DRIVE REGION WAS AN ARBITRARY PICK, and it picked the worst case. The old code took
    `EDGES[0]["source"]`, which is `thal_relay` -- the most saturating hub in the graph. Driven
    there the network pins to a binary pattern (28 of 40 pools at rate 1.0, 12 at 0.0, tail values
    only {0.0, 1.0}) and observable 2 reads exactly 0.700000 in BOTH arms for every spread seed
    tried. Every difference lands in ticks 1..14 and there is NONE at or after the drive stops, so
    observable 2 -- defined over ticks 20..59 -- was measuring a window in which nothing could
    differ BY CONSTRUCTION. It was a dead term in the verdict disjunction. This version sweeps
    EVERY drive region instead of picking one, which gives the term a reachable red and removes the
    arbitrary choice. **RE-MEASURED 2026-08-05 AND RE-ATTRIBUTED, because the figure this sentence
    used to carry was produced by the comparison the file exists to replace.** It read "16 of 40
    drive regions do move it", and 16/40 is exactly `uniform(1)` vs `varied` -- the CONFOUNDED arm.
    Under the comparison this file actually runs (`uniform(mean)` vs `varied`) the count is 17, 10,
    19 and 12 of 40 for spread seeds 0..3, **21 of 40 in the union over the four and 8 of 40 in the
    intersection**; the VERDICT 1 arm (`uniform(1)` vs `uniform(4)`) moves it in 13 of 40. The
    number was reproducible and still wrong, which is the harder failure to see: citing a figure
    from the superseded comparison inside the paragraph that supersedes it.
  * THE VERDICT WAS A ONE-TERM TEST WEARING THREE TERMS' CLOTHES. `charged = (not identical) or
    moved or abs(tail_delta) > 1e-12` -- but identical trajectories imply identical tails AND
    identical first-arrival ticks, so the last two terms could never flip it. They are reported as
    diagnostics now, and the verdict says which comparison it rests on.
  * A FOURTH "CRITERION" THAT COULD NOT FAIL. `main` carried
    `assert _selfcheck(edges, regions), "self-check must return True"` against a `_selfcheck` whose
    only `return` was the literal `True`, so the assert had no reachable red at all: the pass was
    unconditional and the fail could only come from an AssertionError raised inside. F11 deletes
    such a criterion rather than weakening it -- the call stays, the assert is gone, and the file's
    assert count is 3, which is also the number that redden. See `_selfcheck`'s own docstring.
  * A GUARD THAT DID NOT GUARD, AND WOULD HAVE CORRUPTED A TOTAL. `_compare` tested
    `same_shape` and used it for three of its five fields; `moved` and `recruited` indexed the
    other trajectory unguarded, so a mismatch raised `KeyError` before any sentinel returned -- and
    the sentinel it would have returned was `cells = -1`, which `_sweep` SUMS into the aggregate
    the verdict is read off. Both shape and length are now explicit contracts that raise. The
    length one was found by feeding `_compare` a 4-tick trajectory: `b[DRIVE_TICKS:]` is then empty,
    `tail_delta` is NaN, and `abs(nan) > 1e-12` is False -- the reverberation term would have
    reported "did not move" for every pair while measuring nothing.

THE REGIME IS SATURATED, AND A FUTURE NULL FROM THIS HARNESS MUST NOT BE BELIEVED UNTIL THAT IS
RE-CHECKED (measured and added 2026-08-05). Over 576,000 cells sampled across both verdicts' arms,
the pool firing rate takes exactly TWO values: 0.0 and 1.0 -- and `_regime` re-derives the same
answer over 288,000 cells on every run and PRINTS it, so this is a line of output rather than a
number in a docstring that nothing re-checks. The router's stated design is
that *"per-neuron threshold heterogeneity turns a graded drive in [0, 1] into a graded pool firing
RATE"* -- at K=24 a graded rate would land on multiples of 1/24 -- and at DRIVE_AMP=1.0 not one
intermediate value occurs anywhere. Three consequences, and only the third is comfortable:

  * The router SEED is inert here. Sweeping it 0..4 returns byte-identical aggregates
    (magnitude 1200 pairs rec=0 sil=0; variation differ=112 rec=20 sil=10, every seed), because
    `thr` cannot matter when every pool is all-or-none. So "the result is not a seed artifact" is
    true but says less than it sounds like: the seed is not a live dimension at this drive.
  * COINCIDENCE DETECTION -- the mechanism this file names as the one biology predicts delays are
    FOR -- is a graded phenomenon, and it is being measured in the one regime that cannot express
    it gradually. `recruited` here means a pool went from zero neurons firing to all 24.
  * The verdicts are POSITIVES, so saturation does not invalidate them: a difference found in a
    hard-to-move regime is still a difference, and if anything understates the effect. **But a
    NULL from this harness would be uninterpretable** -- indistinguishable from "the regime cannot
    express it" -- and that is not a hazard that announces itself. DRIVE_AMP is de-saturable
    (measured: 0.25 -> 5 distinct rates, 0.10 -> 7, 0.05 -> silence), so re-running there is the
    check a null owes before it is reported as one. `main` prints the regime for this reason.

WHAT THIS IS NOT. It does not claim the varied delays are anatomically correct -- they are a spread,
not a measurement. Establishing real per-tract values is a P1 obligation that only arises if
VERDICT 2 comes back positive. The [1, 6] range is the treatment arm of an experiment and is not
defended, and neither is DRIVE_AMP.

Reproduce:
    PYTHONPATH=src python tools/ab_connectome_delay.py
"""

import os
import sys
from contextlib import contextmanager

import numpy as np

# Derived from __file__, not the literal "src": a relative sys.path entry is resolved against the
# cwd at import time, so this ran only from the repo root and silently imported nothing elsewhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from snn_data_lake import connectome  # noqa: E402
from snn_data_lake.thalamus.connectome_router import ConnectomeRouter  # noqa: E402

TICKS = 60
DRIVE_TICKS = 20
SEED = 0
#: Spread seeds swept so no claim rests on one draw of the delay table.
SPREAD_SEEDS = (0, 1, 2, 3)
#: The treatment range. ARBITRARY AND NOT DEFENDED -- this is an experiment, not anatomy.
DELAY_LO, DELAY_HI = 1, 7
#: External drive amplitude. Was an unnamed `1.0` inside `_run`; named 2026-08-05 because it is the
#: value that decides the REGIME every observable is measured in, and at 1.0 that regime is fully
#: saturated (pool rate takes only 0.0 / 1.0). Kept at 1.0 so no reported number moves -- naming it
#: is the fix, not changing it. Equally undefended as DELAY_LO/HI; see the module docstring.
DRIVE_AMP = 1.0


def _tracts(edges):
    """Distinct tract names. A delay is a property of a fibre bundle, not of a per-call draw."""
    return sorted({e["tract"] for e in edges})


def _varied_delays(edges, seed=SEED):
    """A deterministic spread of delays in [DELAY_LO, DELAY_HI), keyed by tract identity."""
    rng = np.random.default_rng(seed)
    return {t: int(rng.integers(DELAY_LO, DELAY_HI)) for t in _tracts(edges)}


def _uniform_delays(edges, value):
    """Every tract gets the SAME delay -- zero variation, magnitude only."""
    return {t: int(value) for t in _tracts(edges)}


def _mean_matched(table):
    """The uniform delay matched to `table`'s own MEAN -- the control that isolates spread.

    P1: derived from the treatment table rather than fixed, so it tracks whatever range the
    experiment uses. Checked at the degenerate inputs: EMPTY table -> 1 (the router's own default,
    not a division by zero); SINGLE element -> that element; an all-ZERO table -> 1, because the
    router floors delays at 1 itself (`np.maximum(..., 1)` in `ConnectomeRouter.__init__`) and a 0
    here would be silently rewritten downstream, making the control not the control it claims.
    """
    if not table:
        return 1
    return max(1, int(round(float(np.mean(list(table.values()))))))


@contextmanager
def _delays(edges, table):
    """Install `table` on the shared EDGES dicts, RESTORING the originals on exit (never popping).

    The router reads `connectome.EDGES` at CONSTRUCTION, so a router built inside this block sees
    the table and one built outside does not.
    """
    original = [e.get("delay", 1) for e in edges]
    try:
        for e in edges:
            e["delay"] = table[e["tract"]]
        yield
    finally:
        for e, d in zip(edges, original, strict=True):
            e["delay"] = d


def _run(drive_region, ticks=TICKS, drive_ticks=DRIVE_TICKS, seed=SEED, amp=DRIVE_AMP):
    """Drive one region, then release. Returns the full per-region rate trajectory."""
    router = ConnectomeRouter(seed=seed)
    router.reset_state()
    traj = []
    for t in range(ticks):
        out = router.step({drive_region: amp} if t < drive_ticks else {})
        traj.append([out[r] for r in router.regions])
    return np.asarray(traj, dtype=float)


def _regime(edges, regions, amp=DRIVE_AMP):
    """How many DISTINCT pool-rate values does the network produce? 2 means all-or-none.

    Reported beside the verdicts because it decides what a NULL from this harness would mean. The
    router turns drive into a GRADED pool rate via threshold heterogeneity (K=24, so a graded rate
    lands on multiples of 1/24); if only {0.0, 1.0} occur, nothing graded is being measured and a
    null could not be distinguished from a regime that cannot express the effect.
    """
    vals, cells = set(), 0
    for table in (_uniform_delays(edges, 1), _uniform_delays(edges, 4), _varied_delays(edges, 0)):
        with _delays(edges, table):
            for r in regions:
                t = _run(r, amp=amp)
                vals |= set(np.unique(t).tolist())
                cells += t.size
    return {"distinct": len(vals), "cells": cells, "saturated": sorted(vals) == [0.0, 1.0]}


def _first_rise(traj, thresh=1e-9):
    """First tick at which each region fires at all; -1 if it never does."""
    out = {}
    for j, col in enumerate(traj.T):
        nz = np.nonzero(col > thresh)[0]
        out[j] = int(nz[0]) if nz.size else -1
    return out


def _compare(a, b):
    """The observables, A vs B. Both trajectories must have the SAME shape.

    The mismatched-shape branch this used to carry was not a guard. It set `cells` to -1 and
    `tail_delta` to 0.0 while `moved` / `recruited` indexed `fb[j]` unguarded, so a mismatch raised
    `KeyError` before any sentinel could be returned -- and had it not, `_sweep` sums `cells`, so a
    -1 would have silently subtracted from the aggregate the verdict is read off. A contract that
    cannot be met quietly is better than a fallback that corrupts a total (2026-08-05).
    """
    if a.shape != b.shape:
        raise ValueError(f"trajectories must have the same shape (got {a.shape} vs {b.shape})")
    if a.shape[0] <= DRIVE_TICKS:
        # An empty tail slice makes `tail_delta` NaN, and `abs(nan) > 1e-12` is False -- so the
        # reverberation term would report "did not move" for every pair while measuring nothing.
        # Found by feeding _compare a trajectory shorter than DRIVE_TICKS (2026-08-05); unreachable
        # at TICKS=60 / DRIVE_TICKS=20, and one constant edit away from being reachable silently.
        raise ValueError(f"need more than DRIVE_TICKS={DRIVE_TICKS} ticks, got {a.shape[0]}")
    fa, fb = _first_rise(a), _first_rise(b)
    return {
        "identical": bool(np.array_equal(a, b)),
        "cells": int(np.sum(a != b)),
        "moved": sum(1 for j in fa if fa[j] != fb[j]),
        # qualitative, BOTH directions: a latency shift can produce neither.
        "recruited": sum(1 for j in fa if fa[j] == -1 and fb[j] != -1),  # silent in A, fires in B
        "silenced": sum(1 for j in fa if fa[j] != -1 and fb[j] == -1),  # fires in A, silent in B
        "tail_delta": float(b[DRIVE_TICKS:].mean() - a[DRIVE_TICKS:].mean()),
    }


def _sweep(edges, regions, table_of, seeds=SPREAD_SEEDS):
    """Run `A vs B` over every drive region and spread seed. `table_of(seed) -> (a_table, b_table)`.

    `seeds` is a parameter because the MAGNITUDE sweep's tables do not depend on the seed: running
    it over SPREAD_SEEDS would repeat one comparison four times and multiply its cell count by
    four, which invites a false side-by-side with the variation sweep's genuinely distinct draws.

    Returns (per-pair results, aggregate dict).
    """
    rows = []
    for region in regions:
        for seed in seeds:
            ta, tb = table_of(seed)
            with _delays(edges, ta):
                a = _run(region)
            with _delays(edges, tb):
                b = _run(region)
            r = _compare(a, b)
            r["region"], r["seed"] = region, seed
            rows.append(r)
    agg = {
        "pairs": len(rows),
        "differ": sum(1 for r in rows if not r["identical"]),
        "cells": sum(r["cells"] for r in rows),
        "moved": sum(r["moved"] for r in rows),
        "recruited": sum(r["recruited"] for r in rows),
        "silenced": sum(r["silenced"] for r in rows),
        "tail_moved": sum(1 for r in rows if abs(r["tail_delta"]) > 1e-12),
    }
    return rows, agg


def _magnitude_recruitment_control(edges, regions, lo=DELAY_LO, hi=DELAY_HI):
    """Can a PURE MAGNITUDE change ever recruit or silence a region? Sweeps every uniform pair.

    This is the control the recruitment result stands on, and it was standing on one pair.
    VERDICT 1 happens to compare uniform(1) against uniform(mean) and happens to report
    `recruited: 0`, which was read as "magnitude cannot recruit". One pair is an anecdote: it
    proves nothing about uniform(3) vs uniform(4), or about the reverse direction, and the whole
    weight of VERDICT 2 rests on the contrast. Sweeping all ordered pairs in [lo, hi) over every
    drive region turns the anecdote into a measurement, and a NON-ZERO result here would refute
    the qualitative claim outright -- which is what makes it a control rather than a decoration.
    """
    traj = {(k, r): None for k in range(lo, hi) for r in regions}
    for k in range(lo, hi):
        with _delays(edges, _uniform_delays(edges, k)):
            for r in regions:
                traj[(k, r)] = _run(r)
    pairs = rec = sil = 0
    for i in range(lo, hi):
        for k in range(lo, hi):
            if i == k:
                continue
            for r in regions:
                c = _compare(traj[(i, r)], traj[(k, r)])
                rec += c["recruited"]
                sil += c["silenced"]
                pairs += 1
    return {"pairs": pairs, "recruited": rec, "silenced": sil}


def _bracket_control(edges, regions, seeds=SPREAD_SEEDS):
    """Close the ROUNDING residual: does the varied arm differ from BOTH neighbouring magnitudes?

    `_mean_matched` rounds, so the control carries a residual magnitude offset (+0.31, -0.40,
    +0.26, -0.32 ticks for seeds 0..3). If some uniform magnitude reproduced the varied arm, the
    VERDICT 2 positive would be that residual rather than the spread. Bracketing answers it without
    a tolerance to tune: compare against floor(mean) and ceil(mean) and require the varied arm to
    differ from both.
    """
    out = []
    for s in seeds:
        vt = _varied_delays(edges, s)
        mu = float(np.mean(list(vt.values()))) if vt else 0.0
        lo, hi = max(1, int(np.floor(mu))), max(1, int(np.ceil(mu)))
        d = {"seed": s, "mean": mu, "lo": lo, "hi": hi, "differ_lo": 0, "differ_hi": 0}
        for r in regions:
            with _delays(edges, vt):
                v = _run(r)
            for key, m in (("differ_lo", lo), ("differ_hi", hi)):
                with _delays(edges, _uniform_delays(edges, m)):
                    d[key] += 0 if _compare(_run(r), v)["identical"] else 1
        out.append(d)
    return out


def _selfcheck(edges, regions):
    """Prove each criterion FIRES on a constructed RED, not merely that it passes on live input.

    Per P8 a criterion whose red cannot be reached is deleted, not weakened -- so each criterion
    below is evaluated TWICE: once on the input it must accept (green) and once on a constructed
    input it must reject (red). Asserting only the green half is how a criterion that can never
    fail passes for a measurement.

    RETURNS NOTHING, and that is the fix rather than an omission (2026-08-05). This function used
    to end `return True` -- a literal -- and `main` wrapped the call in
    `assert _selfcheck(edges, regions), "self-check must return True"`. That assert had NO
    reachable red by construction: the only two outcomes are an AssertionError raised from inside
    this function, or the literal True. It is the exact shape `tools/red_reachability.py` names
    TAUTOLOGICAL, in the file whose entire subject is criteria that can fail, and F11 says such a
    criterion is DELETED rather than weakened. The call does the work; the assert around it only
    looked like a fourth check.
    """
    region = regions[0]
    base = _run(region)
    with _delays(edges, _uniform_delays(edges, 2)):
        shifted = _run(region)
        twin = _run(region)

    # CRITERION 1 -- "a null (baseline-equal) treatment reproduces the baseline exactly".
    #   GREEN: the all-1 table (which IS the baseline) reproduces it.
    #   RED  : a delay-2 table does NOT. If this ever came back True the criterion would be
    #          satisfied by a treatment that demonstrably changed the network.
    with _delays(edges, _uniform_delays(edges, 1)):
        green1 = _compare(base, _run(region))["identical"]
    red1 = _compare(base, shifted)["identical"]
    assert green1 and not red1, f"criterion 1 has no reachable red (green={green1} red={red1})"

    # CRITERION 2 -- "the harness detects a delay change at all" (the sensitivity floor).
    #   GREEN: baseline vs delay-2 differs.
    #   RED  : delay-2 vs ITSELF does not. A harness reporting a difference here would be
    #          manufacturing positives out of nondeterminism or leaked delay state.
    green2 = not _compare(base, shifted)["identical"]
    red2 = not _compare(shifted, twin)["identical"]
    assert green2 and not red2, f"criterion 2 has no reachable red (green={green2} red={red2})"


def main():
    print("=" * 78)
    print("A/B -- does per-tract conduction delay variation charge the connectome router?")
    print("=" * 78)

    edges = connectome.EDGES
    regions = ConnectomeRouter(seed=SEED).regions
    baseline = [int(e.get("delay", 1)) for e in edges]
    ref = _varied_delays(edges)
    matched = _mean_matched(ref)
    mean_delay = float(np.mean(list(ref.values()))) if ref else 0.0

    print(f"edges: {len(edges)}   distinct tracts: {len(_tracts(edges))}   pools: {len(regions)}")
    print(f"BASELINE delays: min={min(baseline)} max={max(baseline)} distinct={len(set(baseline))}")
    print(
        f"VARIED   delays: range [{DELAY_LO}, {DELAY_HI - 1}] mean={mean_delay:.2f} "
        f"-> mean-matched control = uniform({matched})"
    )
    print(f"sweep: {len(regions)} drive regions x {len(SPREAD_SEEDS)} spread seeds, {TICKS} ticks")
    reg = _regime(edges, regions)
    print(
        f"REGIME at DRIVE_AMP={DRIVE_AMP}: {reg['distinct']} distinct pool-rate values over "
        f"{reg['cells']} cells -> {'SATURATED (all-or-none)' if reg['saturated'] else 'graded'}"
    )
    if reg["saturated"]:
        print("  a POSITIVE here is still a positive; a NULL here would NOT be interpretable")
        print("  without re-running de-saturated -- see the module docstring.")
    print()

    # ---------- controls, with their reds constructed and run ----------
    _selfcheck(edges, regions)  # raises on failure; NOT wrapped in an assert -- see its docstring
    with _delays(edges, _uniform_delays(edges, 1)):
        null = _compare(_run(regions[0]), _run(regions[0]))
    assert null["identical"], "control 1: a null (all-1) treatment must reproduce the baseline"
    print("controls: 3 criteria, each evaluated on its GREEN input and on a constructed RED;")
    print("          all three accepted the green and REJECTED the red (the reds are reachable)")

    mag_rec = _magnitude_recruitment_control(edges, regions)
    print(
        f"control: {mag_rec['pairs']} uniform-vs-uniform pairs (ZERO variation, magnitudes "
        f"{DELAY_LO}..{DELAY_HI - 1}) ->"
    )
    print(
        f"          recruited={mag_rec['recruited']}  silenced={mag_rec['silenced']}   "
        "(pure magnitude cannot make a region fire or fall silent)"
    )
    print()

    # ---------- VERDICT 1: does the delay FIELD carry anything? (magnitude only) ----------
    _, mag = _sweep(
        edges,
        regions,
        lambda s: (_uniform_delays(edges, 1), _uniform_delays(edges, matched)),
        seeds=(SEED,),  # the tables do not depend on the seed; see _sweep
    )
    # a ONE-tick magnitude step, so the two per-pair cell counts below are readable on one scale
    step = max(1, matched - 1)
    _, mag1 = _sweep(
        edges,
        regions,
        lambda s: (_uniform_delays(edges, step), _uniform_delays(edges, matched)),
        seeds=(SEED,),
    )
    print("--- VERDICT 1: does the `delay` FIELD carry anything? ---")
    print(f"  uniform(1) vs uniform({matched})  [ZERO variation in either arm]")
    print(
        f"  pairs differing: {mag['differ']}/{mag['pairs']}   cells: {mag['cells']} "
        f"({mag['cells'] / mag['pairs']:.1f}/pair)   first-arrival moves: {mag['moved']}   "
        f"recruited: {mag['recruited']}  silenced: {mag['silenced']}"
    )
    print(
        f"  SCALE, so the per-pair figures below are not read against a different distance: this "
        f"arm spans {matched - 1} ticks of"
    )
    print(
        f"  magnitude. ONE tick -- uniform({step}) vs uniform({matched}) -- is "
        f"{mag1['cells'] / mag1['pairs']:.1f} cells/pair over {mag1['pairs']} pairs."
    )
    print()

    # ---------- VERDICT 2: does VARIATION carry anything beyond magnitude? ----------
    rows, var = _sweep(
        edges,
        regions,
        lambda s: (
            _uniform_delays(edges, _mean_matched(_varied_delays(edges, s))),
            _varied_delays(edges, s),
        ),
    )
    print("--- VERDICT 2: does per-tract VARIATION carry anything BEYOND magnitude? ---")
    print("  uniform(mean) vs varied  [magnitude held constant, spread is the only difference]")
    print(
        f"  pairs differing: {var['differ']}/{var['pairs']}   cells: {var['cells']} "
        f"({var['cells'] / var['pairs']:.1f}/pair)   first-arrival moves: {var['moved']}"
    )
    print(
        f"  QUALITATIVE, both directions: recruited (fires under varied, never under uniform) = "
        f"{var['recruited']}; silenced (fires under uniform, never under varied) = {var['silenced']}"
    )
    print(f"  tail (reverberation) moved in {var['tail_moved']}/{var['pairs']} pairs")
    print("  BRACKET control -- varied vs BOTH neighbouring integer magnitudes (closes the")
    print("  rounding residual in the mean-match; see the module docstring):")
    for d in _bracket_control(edges, regions):
        print(
            f"    seed={d['seed']} mean={d['mean']:.3f}: differs from uniform({d['lo']}) in "
            f"{d['differ_lo']}/{len(regions)}, from uniform({d['hi']}) in "
            f"{d['differ_hi']}/{len(regions)}"
        )
    top = sorted(rows, key=lambda r: (-r["cells"], r["region"]))[:6]
    print("  largest variation effects by drive region:")
    for r in top:
        print(
            f"    {r['region']:<14} seed={r['seed']}  cells={r['cells']:>4}  "
            f"moved={r['moved']:>2}  recruited={r['recruited']}  silenced={r['silenced']}  "
            f"tail_delta={r['tail_delta']:+.6g}"
        )
    print()

    field_charged = mag["differ"] > 0
    variation_charged = var["differ"] > 0
    print("=" * 78)
    if field_charged:
        print("VERDICT 1: CHARGED -- the `delay` field changes the router's output. KEEP it.")
    else:
        print("VERDICT 1: INERT -- delay changes nothing. REVERT the field (P3: a wire that does")
        print("  not charge is reverted, not kept). Do not 'fix' this by picking louder numbers.")
    if variation_charged:
        print("VERDICT 2: CHARGED -- per-tract VARIATION changes the output beyond magnitude.")
        print("  Learned/per-tract delays are worth pursuing. NEXT OBLIGATION (P1): the values")
        print("  must stop being a spread and become anatomy or a learned parameter.")
    else:
        print("VERDICT 2: NULL -- spread buys nothing a single global delay constant does not.")
        print("  Per-tract delay values would be fitting noise; a ONE-constant delay is the")
        print("  honest form, and DCLS-style learned delays are not indicated by this router.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
