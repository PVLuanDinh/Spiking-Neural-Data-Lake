#!/usr/bin/env python3
"""A/B: does per-tract CONDUCTION DELAY variation change anything in the connectome router?

specs/010 follow-on. `connectome.py`'s `_e()` gained a `delay` field on 2026-08-04; before that every
long-range projection arrived in a uniform 1 tick. The router (`thalamus/connectome_router.py:83`)
was already built to consume per-edge delays and to size a ring buffer to the longest one -- it had
just never been handed a value.

THIS SCRIPT EXISTS TO BE ABLE TO SAY NO. Per P8 a criterion with no reachable RED is worthless, and
per P3 a wire that does not charge is REVERTED, not kept. So the question is asked in the form that
can fail:

    Does replacing the uniform 1-tick delay with a VARIED assignment change any observable
    of the live propagation layer?

If the answer is no, the `delay` field is a constant wearing a signal's clothes (P3 shape 5) and the
honest action is to revert it rather than keep a parameter that carries nothing.

WHAT IS COMPARED. Three observables, chosen because they are what a delay could plausibly move:
  1. per-region firing rate trajectory over the run (the raw signal)
  2. REVERBERATION -- activity persisting after drive stops, which is the loop-closure property the
     router exists to demonstrate
  3. ARRIVAL ORDER at convergence nodes -- the coincidence-detection observable. This is the one the
     biology predicts should move: delays are what make a target fire only when inputs CO-ARRIVE.

WHAT THIS IS NOT. It does not claim the varied delays are anatomically correct -- they are a spread,
not a measurement. Establishing real per-tract values is a P1 obligation that only arises if this
A/B comes back positive. A null here means the field goes, and no anatomy is needed.

Reproduce:
    PYTHONPATH=src python tools/ab_connectome_delay.py
"""

import os
import sys

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


def _varied_delays(edges, seed=SEED):
    """A deterministic spread of delays in [1, 6], keyed by tract identity.

    Keyed by the tract NAME so the same tract always gets the same delay -- a delay is a property of
    a fibre bundle, not of a random draw per call. The RANGE is arbitrary and is not defended: this
    is the treatment arm of an experiment, not a claim about anatomy.
    """
    rng = np.random.default_rng(seed)
    tracts = sorted({e["tract"] for e in edges})
    table = {t: int(rng.integers(1, 7)) for t in tracts}
    return table


def _run(router, drive_region, ticks=TICKS, drive_ticks=DRIVE_TICKS):
    """Drive one region, then release. Returns the full per-region rate trajectory."""
    router.reset_state()
    traj = []
    for t in range(ticks):
        drive = {drive_region: 1.0} if t < drive_ticks else {}
        out = router.step(drive)
        traj.append([out[r] for r in router.regions])
    return np.asarray(traj, dtype=float)


def main():
    print("=" * 78)
    print("A/B -- does per-tract conduction delay variation charge the connectome router?")
    print("=" * 78)

    edges = connectome.EDGES
    table = _varied_delays(edges)
    baseline_delays = [int(e.get("delay", 1)) for e in edges]
    print(f"edges: {len(edges)}   distinct tracts: {len(table)}")
    print(
        f"BASELINE delays: min={min(baseline_delays)} max={max(baseline_delays)} "
        f"distinct={len(set(baseline_delays))}"
    )
    print(
        f"VARIED   delays: min={min(table.values())} max={max(table.values())} "
        f"distinct={len(set(table.values()))}"
    )
    print()

    # pick a driven region that actually has downstream targets
    drive_region = None
    for e in edges:
        if e["source"] in connectome.NODES:
            drive_region = e["source"]
            break

    # --- A: as shipped (uniform) ---
    a = _run(ConnectomeRouter(seed=SEED), drive_region)

    # --- B: varied, applied by mutating the edge dicts the router reads at construction ---
    original = [e.get("delay", 1) for e in edges]
    try:
        for e in edges:
            e["delay"] = table[e["tract"]]
        b = _run(ConnectomeRouter(seed=SEED), drive_region)
    finally:
        for e, d in zip(edges, original, strict=True):
            e["delay"] = d

    # ---- observable 1: rate trajectory ----
    same_shape = a.shape == b.shape
    identical = bool(same_shape and np.array_equal(a, b))
    max_abs = float(np.max(np.abs(a - b))) if same_shape else float("nan")
    n_diff_cells = int(np.sum(a != b)) if same_shape else -1
    n_diff_ticks = int(np.sum(np.any(a != b, axis=1))) if same_shape else -1

    print(f"driven region: {drive_region!r}   ticks: {TICKS} (drive for {DRIVE_TICKS})")
    print("--- observable 1: per-region rate trajectory ---")
    print(f"  identical              : {identical}")
    print(f"  max |A-B|              : {max_abs:.6g}")
    print(f"  cells differing        : {n_diff_cells} of {a.size}")
    print(f"  ticks with any change  : {n_diff_ticks} of {TICKS}")
    print()

    # ---- observable 2: reverberation (post-drive persistence) ----
    tail_a = float(a[DRIVE_TICKS:].mean())
    tail_b = float(b[DRIVE_TICKS:].mean())
    print("--- observable 2: reverberation (mean activity after drive stops) ---")
    print(
        f"  A (uniform) = {tail_a:.6f}    B (varied) = {tail_b:.6f}    delta = {tail_b - tail_a:+.6f}"
    )
    print()

    # ---- observable 3: arrival order at convergence nodes ----
    def first_rise(traj, thresh=1e-9):
        out = {}
        for j, r in enumerate(traj.T):
            nz = np.nonzero(r > thresh)[0]
            out[j] = int(nz[0]) if nz.size else -1
        return out

    fa, fb = first_rise(a), first_rise(b)
    moved = [j for j in fa if fa[j] != fb[j]]
    print("--- observable 3: first-arrival tick per region (coincidence observable) ---")
    print(f"  regions whose first-arrival tick MOVED: {len(moved)} of {len(fa)}")
    if moved:
        for j in moved[:8]:
            print(f"    {ConnectomeRouter(seed=SEED).regions[j]:<14} {fa[j]} -> {fb[j]}")
    print()

    charged = (not identical) or moved or abs(tail_b - tail_a) > 1e-12
    print("=" * 78)
    if charged:
        print("VERDICT: CHARGED -- delay variation changes the router's output.")
        print(
            "  Keeping the `delay` field is justified. NEXT OBLIGATION (P1): the values must stop"
        )
        print(
            "  being a spread and become anatomy. A delay that arbitrates behaviour is exactly the"
        )
        print("  shape P1 forbids leaving as a magic constant.")
    else:
        print("VERDICT: INERT -- delay variation changes NOTHING measurable.")
        print("  REVERT the `delay` field. P3: a wire that does not charge is reverted, not kept,")
        print("  and a parameter that cannot move an outcome is a constant wearing a signal's")
        print("  clothes (shape 5). Do not 'fix' this by picking louder numbers.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
