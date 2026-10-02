"""
Stress-test the spiking associative-memory prototype: how many patterns can it
hold, and how much cue corruption can it tolerate, before recall breaks?

Reuses the prototype's functions directly (no duplication). Run:
    python test_prototype.py
"""

import random

from snn_data_lake.faculties.spiking_storage_prototype import (
    make_patterns,
    train,
    recall,
    corrupt,
    overlap,
)

N, K, STEPS = 256, 20, 12
TRIALS = 5  # average over a few random pattern sets for stable numbers


def mean_recall(num_patterns, drop, add, trials=TRIALS, rng=None):
    """avg recall overlap across `trials` random codebooks, recalling every
    stored pattern from a cue with `drop` bits removed + `add` bits added.

    `rng` is threaded into make_patterns/corrupt rather than seeded globally. Until
    2026-08-04 this function was deterministic only as a SIDE EFFECT of a module-scope
    `random.seed(0)` inside spiking_storage_prototype, which was removed (correctly --
    import-time global seeding is banned tree-wide, and run_selfchecks runs every
    self-check in ONE process, so one module's import-time seed silently steers every
    later one). Nothing here ever seeded itself, so removing it made this file
    entropy-seeded: three fresh processes measured random.random() at 0.132 / 0.687 /
    0.937. The asserted point was not flaky (mean_recall(10,4,4) == 1.0 on 60/60 distinct
    streams) but the PRINTED sweep was: mean_recall(80,8,8) ranged 0.9825-0.9975 over 12
    streams. A default of None keeps the module's own default behaviour for any other
    caller; main() passes an explicit Random(0) so the reported numbers reproduce."""
    total, count = 0.0, 0
    for t in range(trials):
        pats = make_patterns(num_patterns, N, K, rng=rng)
        W = train(pats, N, K)
        for p in pats:
            cue = corrupt(p, N, K, drop=drop, add=add, rng=rng)
            out, _, _ = recall(W, cue, N, K, STEPS)
            total += overlap(out, p, K)
            count += 1
    return total / count


def capacity_sweep(rng=None):
    """Fix moderate noise (6 of 20 bits), grow the codebook until recall falls.
    The FACTORED memory stores the patterns themselves (O(P*k)), so it holds far more
    than the classic Hebbian 'few % of N' -- recall stays high until inter-pattern
    crosstalk in the reconstructed field dominates. Swept well past N to find that knee."""
    print(f"CAPACITY SWEEP  (N={N}, K={K}, cue: drop 6 / add 6)")
    print(f"{'patterns':>9} | {'mean recall':>11} | bar")
    print("-" * 48)
    knee = None
    for num in (20, 40, 80, 128, 192, 256, 384, 512, 768):
        r = mean_recall(num, drop=6, add=6, rng=rng)
        bar = "#" * int(r * 20)
        print(f"{num:>9} | {r:>10.1%} | {bar}")
        if knee is None and r < 0.90:  # first load that drops below 90%
            knee = num
    if knee:
        print(f"knee (recall first < 90%): ~{knee} patterns = {knee / N:.1f}x N")
    else:
        print("knee: no cliff up to 3x N (768 patterns)")
    print()


def noise_sweep(rng=None):
    """Fix a comfortable load (15 patterns), grow cue corruption until recall
    fails. drop=add keeps the cue the same size as a stored pattern."""
    print(f"NOISE SWEEP  (N={N}, K={K}, 15 patterns stored)")
    print(f"{'corruption':>10} | {'mean recall':>11} | bar")
    print("-" * 48)
    for d in (0, 2, 4, 6, 8, 10, 12, 14, 16):
        r = mean_recall(15, drop=d, add=d, rng=rng)
        pct = d / K  # fraction of active bits swapped
        bar = "#" * int(r * 20)
        print(f"{pct:>9.0%} | {r:>10.1%} | {bar}")
    print()


def _cue_baseline(num_patterns, drop, add, trials=TRIALS, rng=None):
    """DO-NOTHING control: score the corrupted cue against the stored pattern WITHOUT
    running recall. Same codebooks, same corruption, attractor dynamics removed. This is
    the control that is not the method itself (P8) -- it is what `recall` has to beat."""
    total, count = 0.0, 0
    for _ in range(trials):
        pats = make_patterns(num_patterns, N, K, rng=rng)
        for p in pats:
            cue = corrupt(p, N, K, drop=drop, add=add, rng=rng)
            total += overlap(cue, p, K)
            count += 1
    return total / count


def main():
    print("=" * 48)
    print("SPIKING ASSOCIATIVE MEMORY -- STRESS TEST")
    print("=" * 48)
    # One RNG threaded through every sweep so the PRINTED numbers reproduce run to run.
    # Deliberately NOT random.seed(0): run_selfchecks executes every self-check in one
    # process, so a global seed here would steer whichever module runs next.
    capacity_sweep(rng=random.Random(0))
    noise_sweep(rng=random.Random(0))

    # Sanity floor, judged against a control rather than a literal. The old form was
    # `mean_recall(10, 4, 4) >= 0.95`, which measured 1.0 on 60/60 distinct global
    # streams -- so as a REGRESSION detector it was blind across the whole 0.80-0.95
    # band: the uncorrupted-cue control measures 0.800 HERE (measured at this fixture,
    # rng=Random(0); a prior review reported ~0.60 from a different fixture and that
    # figure does not reproduce at drop=add=4), so a memory that returned its input
    # unchanged scored 0.800 and the old assert was only 0.15 away from passing a
    # do-nothing implementation. Replaced with the control comparison plus an exact
    # bound, neither of which carries a tuned number.
    got = mean_recall(10, drop=4, add=4, rng=random.Random(0))
    cue = _cue_baseline(10, drop=4, add=4, rng=random.Random(0))
    assert got > cue, f"recall must beat the uncorrected cue ({got:.3f} vs {cue:.3f})"
    assert got == 1.0, f"easy case must recall every pattern exactly (got {got:.4f})"
    # RED, constructed and EXECUTED 2026-08-04, not asserted (P8 -- a criterion whose red
    # was never reached is not a criterion). Substituting the identity
    # `recall = lambda memory, cue_active, n, k, steps: (set(cue_active), 1, 0)`
    # -- attractor dynamics removed, cue returned unchanged -- measures:
    #     got = 0.8000   cue = 0.8000
    #     got > cue   -> False   (fires)
    #     got == 1.0  -> False   (fires)
    #     old >= 0.95 -> False   (also fires)
    # So the old literal was NOT blind to a total neuter, and the claim that it was is
    # withdrawn. What it was blind to is the PARTIAL band 0.80-0.95: any regression that
    # left the memory doing some work but not all of it scored above the do-nothing
    # control and below correctness, and passed. `got > cue` closes the bottom of that
    # band against a measured control instead of a guess, and `got == 1.0` closes the top
    # without introducing a tuned number -- 1.0 is the definitional maximum of an
    # overlap fraction, not a threshold anyone chose.
    print(f"self-check OK: 10 patterns @ 20% noise -> recall {got:.3f} vs cue {cue:.3f}")


if __name__ == "__main__":
    main()
