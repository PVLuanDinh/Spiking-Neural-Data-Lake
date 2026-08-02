"""Measure which SATURATION REGIME the live time-cell clock actually runs in.

Why this exists (spec 003, T018 / SC-003). TimeCellClock's readout behaves differently once
every cell has fired: below saturation it reads the population's rank-order code, at and above
saturation the population is exhausted and can only report its top nominal latency. Those two
regimes have opposite consequences for a growth claim -- a cell added above the top threshold
widens a clamp rather than contributing -- so every claim about growth MUST name the regime it
was measured in. This probe supplies that number.

It reports the fraction of CLOCK-ADVANCING ticks (ticks on which the clock actually ran) that
sit in each regime, over several seeds of a live brain.

  PYTHONUTF8=1 PYTHONPATH=src python tools/regime_split.py
  PYTHONUTF8=1 PYTHONPATH=src python tools/regime_split.py --seeds 3 --ticks 600
"""

import argparse
import os
import sys

# Probes are run with PYTHONPATH=<root>/src only, so the repo root is NOT importable and the
# brain silently degrades to a blind stdlib perception tier. Put the root on the path first.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def measure(seed, ticks):
    """Drive one live brain and count the regime of every clock-advancing tick."""
    from snn_data_lake.cortex.brain import make_brain

    brain = make_brain(seed=seed, persist=False)
    clock = getattr(brain, "_time_clock", None)
    if clock is None:
        return None

    advancing = saturated = 0
    intervals = 1
    longest = 0
    prev_t = clock.t
    for _ in range(ticks):
        brain.tick()
        now_t = clock.t
        if now_t == prev_t:  # the clock did not run on this brain tick
            continue
        if now_t < prev_t:  # reset() fired: a new timed interval began
            intervals += 1
        else:
            longest = max(longest, now_t)
        advancing += 1
        if clock.n_fired >= clock.n_cells:
            saturated += 1
        prev_t = now_t
    return advancing, saturated, intervals, longest, clock.n_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (default 3)")
    ap.add_argument("--ticks", type=int, default=600, help="brain ticks per seed (default 600)")
    args = ap.parse_args()

    rows = []
    for seed in range(args.seeds):
        got = measure(seed, args.ticks)
        if got is None:
            print(f"seed {seed}: NO CLOCK -- the brain has no _time_clock; nothing to measure")
            return 2
        rows.append((seed,) + got)

    print(f"clock regime split over {args.ticks} brain ticks per seed")
    print("  seed  advancing  unsaturated   saturated  %unsat  intervals  longest  n_cells")
    tot_adv = tot_sat = 0
    for seed, adv, sat, ivals, longest, n_cells in rows:
        unsat = adv - sat
        pct = (100.0 * unsat / adv) if adv else 0.0
        tot_adv += adv
        tot_sat += sat
        print(
            f"  {seed:>4}  {adv:>9}  {unsat:>11}  {sat:>10}  {pct:>5.1f}%"
            f"  {ivals:>9}  {longest:>7}  {n_cells:>7}"
        )

    tot_unsat = tot_adv - tot_sat
    pct = (100.0 * tot_unsat / tot_adv) if tot_adv else 0.0
    # NOTE "N of M", never "N/M": the coverage recorder harvests every bare ratio in a probe's
    # stdout positionally into the artifact fingerprint.
    print(f"\n  unsaturated: {tot_unsat} of {tot_adv} clock-advancing ticks ({pct:.1f}%)")
    print(f"  saturated  : {tot_sat} of {tot_adv} clock-advancing ticks ({100.0 - pct:.1f}%)")
    print(
        "\n  Any growth claim must name which of these regimes it was measured in. A benefit\n"
        "  measured only in the saturated regime is clamp-widening, not a cell earning its\n"
        "  place, and spec 003 FR-006 forbids reporting it as growth."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
