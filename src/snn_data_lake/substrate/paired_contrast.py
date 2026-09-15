r"""The plan's seed-paired statistic, in ONE place: a paired one-sided t over the S seeds against the bar,
with its exact sign-flip false-positive rate printed beside every contrast (PLAN-FINAL Part F preamble).

LIFTED 2026-09-15 (specs/022 T077, evidence/t077-metacognition.md 2.10) from `cortex/prior_drift_harness.py`,
where it was written for T063 (its 2.6) and reused by T070, VERBATIM -- the harness now imports these names
from here and re-exports them, so `prior_drift_harness.contrast` still resolves. Why it moved: the harness is
an EXEMPT instrument (a scratch-population harness, never on the live tick), and a LIVE module that imports
it -- even lazily, inside a function -- reads as "reachable from a live root" to `substrate/dormancy_gate`,
which then names five EXEMPT rows stale (determinism_gate, h01_ingest, kernel_state, kernel_tick,
prior_drift_harness; measured on T077's first campaign commit). Retiring those rows to green the gate would
be the lie T070's record refused (t070-integration-kernel.md 3.5); duplicating the statistic would be the P2
defect. So the statistic lives where any module may import it, and the instruments keep their exemptions.

  paired_t(diffs)              -- ties counted, never dropped; 0.0 when the sample has no variance
  permutation_fpr(diffs, bar)  -- the fraction of the 2^S sign flips whose t exceeds the bar (exact)
  contrast(name, diffs, bar)   -- one contrast decided BEFORE any number: a zero-variance sample is MET iff
                                  its common value is > 0 (recorded as degenerate), else NOT MET
  T_BAR = 1.895                -- paired one-sided t, 7 d.f., alpha = 0.05

Pure standard library; deterministic; ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.paired_contrast
"""

import math
from typing import NamedTuple

# consolidates: cortex/prior_drift_harness.py (merge)
#   -- T_BAR, Contrast, paired_t, permutation_fpr and contrast moved here verbatim; the harness imports and
#   re-exports them (one implementation survives; the harness keeps every other symbol).
T_BAR = 1.895  # paired one-sided t, 7 d.f., alpha = 0.05 (PLAN-FINAL Part F preamble)


class Contrast(NamedTuple):
    name: str
    diffs: tuple
    t: float
    fpr: float
    degenerate: bool
    met: bool


def paired_t(diffs):
    """Paired one-sided t over the S seeds; ties are counted, never dropped. 0.0 when the sample
    has no variance -- that case is DEGENERATE and `contrast` decides it explicitly."""
    n = len(diffs)
    if n < 2:
        return 0.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0.0:
        return 0.0
    return mean / math.sqrt(var / n)


def permutation_fpr(diffs, bar=T_BAR):
    """The false-positive rate of this test on THIS sample: the fraction of the 2^S sign flips
    whose t exceeds the bar. Printed beside every contrast (Part F's preamble)."""
    n = len(diffs)
    hits = 0
    for mask in range(1 << n):
        flipped = [(-d if (mask >> i) & 1 else d) for i, d in enumerate(diffs)]
        if paired_t(flipped) > bar:
            hits += 1
    return hits / float(1 << n)


def contrast(name, diffs, bar=T_BAR):
    """One contrast, decided BEFORE any number: a zero-variance set of differences is MET iff its
    common value is > 0 (recorded as degenerate) and NOT MET otherwise."""
    diffs = tuple(float(d) for d in diffs)
    degenerate = len(set(diffs)) <= 1
    if degenerate:
        return Contrast(name, diffs, 0.0, 0.0, True, bool(diffs and diffs[0] > 0.0))
    t = paired_t(diffs)
    return Contrast(name, diffs, t, permutation_fpr(diffs, bar), False, t > bar)


def main():
    """Self-check with its reds: the statistic must be able to say NOT MET, and its FPR must be exact."""
    import itertools

    up = contrast("all positive, 8 seeds", [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2])
    assert up.met and up.t > T_BAR and not up.degenerate, up
    # the FPR is EXACT: re-enumerate the 2^8 sign flips independently and count the t's above the bar.
    # (The first draft of this check asserted 1/256 from expectation; the enumeration reads 13/256 --
    # flipping one small element still leaves t above the bar -- so the assertion is the enumeration.)
    hits = sum(
        1
        for signs in itertools.product((1.0, -1.0), repeat=len(up.diffs))
        if paired_t([s * d for s, d in zip(signs, up.diffs)]) > T_BAR
    )
    assert up.fpr == hits / 256.0 and 0.0 < up.fpr < 0.1, (up.fpr, hits)
    down = contrast("all negative", [-d for d in up.diffs])
    assert not down.met and down.t < 0.0, down
    noise = contrast("sign-balanced noise", [0.1, -0.1, 0.12, -0.11, 0.09, -0.1, 0.1, -0.12])
    assert not noise.met, noise  # RED: a sign-balanced sample must NOT clear the bar
    deg_pos = contrast("degenerate positive", [0.5] * 8)
    deg_neg = contrast("degenerate zero", [0.0] * 8)
    assert deg_pos.degenerate and deg_pos.met and deg_pos.t == 0.0, deg_pos
    assert deg_neg.degenerate and not deg_neg.met, deg_neg
    assert paired_t([1.0]) == 0.0 and paired_t([]) == 0.0
    assert contrast(up.name, up.diffs) == up, "deterministic"
    print(
        f"self-check OK: paired t {up.t:.4f} > {T_BAR} on 8 same-sign seeds (exact FPR {up.fpr}); "
        f"reversed {down.t:.4f} NOT MET; sign-balanced noise t {noise.t:.4f} NOT MET; degenerate positive MET, "
        "degenerate zero NOT MET; deterministic"
    )


if __name__ == "__main__":
    main()
