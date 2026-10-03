r"""The plan's seed-paired statistic, in ONE place: a paired one-sided t over the S seeds against the bar,
with its exact sign-flip false-positive rate printed beside every contrast (PLAN-FINAL Part F preamble).

LIFTED 2026-09-15 (specs/022 T077, evidence/t077-metacognition.md 2.10) from `instruments/prior_drift_harness.py`,
where it was written for T063 (its 2.6) and reused by T070, VERBATIM -- the harness now imports these names
from here and re-exports them, so `prior_drift_harness.contrast` still resolves. Why it moved: the harness is
an EXEMPT instrument (a scratch-population harness, never on the live tick), and a LIVE module that imports
it -- even lazily, inside a function -- reads as "reachable from a live root" to `gates/dormancy_gate`,
which then names five EXEMPT rows stale (determinism_gate, h01_ingest, kernel_state, kernel_tick,
prior_drift_harness; measured on T077's first campaign commit). Retiring those rows to green the gate would
be the lie T070's record refused (t070-integration-kernel.md 3.5); duplicating the statistic would be the P2
defect. So the statistic lives where any module may import it, and the instruments keep their exemptions.

  paired_t(diffs)              -- ties counted, never dropped; 0.0 when the sample has no variance
  permutation_fpr(diffs, bar)  -- the fraction of the 2^S sign flips whose t exceeds the bar (exact; by
                                  meet-in-the-middle for every S up to 40, see below)
  permutation_fpr_detail(...)  -- the same count plus the population of its certified band
  contrast(name, diffs, bar)   -- one contrast decided BEFORE any number: a zero-variance sample is MET iff
                                  its common value is > 0 (recorded as degenerate), else NOT MET
  T_BAR = 1.895                -- paired one-sided t, 7 d.f., alpha = 0.05
  T_BAR_S31 = 1.6973           -- paired one-sided t, 30 d.f. (S = 31), alpha = 0.05

MERGED 2026-10-03 (G1(c) P1-A, specs/022 evidence/g1c-carried-state.md 2.3): `permutation_fpr` was a 2^S
loop (hours at S = 31). The loop's body is REPLACED, not joined by a second path, by the exact
meet-in-the-middle count below: one mechanism for every S; the 2^S loop survives only as the oracle in
tests/test_paired_contrast.py. The definition is unchanged: the fraction of the sign-flip masks with
`paired_t(flipped) > bar`.

Why the count is exact. A sign flip leaves Q = sum(d*d) fixed, so with S' the flipped total, m' = S'/n and
var' = (Q - n m'^2)/(n - 1), t' = m'/sqrt(var'/n). For bar > 0, t' > bar <=> S' > thr, thr = n * bar *
sqrt(Q / (n (n - 1 + bar^2))) (t' is increasing in S' for S' > 0). The flips split into a left and a right
half; each half's 2^h signed sums are enumerated, the right's sorted, and each left sum counts the right sums
above thr - s by bisection. The one exception to the threshold rule is var' = 0 (every flipped value equal,
possible only when every |d| is equal): `paired_t` returns 0.0 there, so those (at most two) flips are decided
by `paired_t` itself, wherever they fall. Floating point: every flip whose reconstructed total lies within a
band |S' - thr| <= eps of the threshold is ALSO decided by `paired_t` on its reconstructed vector, so the
result equals the definition exactly whenever eps bounds the combined rounding error (derived at `_band_eps`).

Pure standard library; deterministic; ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.paired_contrast
"""

import math
from bisect import bisect_left, bisect_right
from typing import NamedTuple

# consolidates: instruments/prior_drift_harness.py (merge)
#   -- T_BAR, Contrast, paired_t, permutation_fpr and contrast moved here verbatim; the harness imports and
#   re-exports them (one implementation survives; the harness keeps every other symbol).
T_BAR = 1.895  # paired one-sided t, 7 d.f., alpha = 0.05 (PLAN-FINAL Part F preamble)
# paired one-sided t, 30 d.f. (S = 31), alpha = 0.05: pool_selection.t_quantile(0.95, 30) = 1.6972609 rounded UP
# at the fourth place (3.9e-5 stricter), as T_BAR = 1.895 rounds 1.8945786 up at the third. This module does
# NOT import pool_selection (a live module importing an EXEMPT instrument makes dormancy_gate name EXEMPT
# rows stale -- see the docstring); the tie between the two lives in tests/test_pool_selection.py.
T_BAR_S31 = 1.6973

# above MITM_MAX_N the 2^(n/2) half-enumerations are refused, never approximated; more than MITM_MAX_BAND flips
# inside the band are refused, never a silent multi-hour loop; _U is the unit roundoff of an IEEE double;
# (_Q_MIN, _Q_MAX) is the sum-of-squares range inside which no square under- or overflows in the error bound
MITM_MAX_N = 40
MITM_MAX_BAND = 1 << 20
_U = 2.0**-53
_Q_MIN = 1e-250
_Q_MAX = 1e250


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


# distinct-from: faculties/action_gating.py, substrate/lake_recall.py, faculties/access_consciousness.py -- the hook's lexical matches (a top-k set, an index rebuild, an ignition threshold; also a LIF step and a salience readout) share only docstring vocabulary ("iff", "threshold") with the helpers below; none enumerates signed subset sums or counts sign flips and no module in src/ does (grep: no sign-flip counter, no meet-in-the-middle); the helpers REPLACE the body of this module's own permutation_fpr, they are not a second mechanism beside it
def _signed_sums(xs):
    """All 2^len(xs) signed sums of xs, accumulated left to right; bit k of an index is set iff xs[k] is
    negated (the mask convention of the original 2^S loop)."""
    sums = [0.0]
    for x in xs:
        sums = [s + x for s in sums] + [s - x for s in sums]
    return sums


def _threshold(n, bar, q):
    """The total S' a flip must exceed for t' > bar (bar > 0): n * bar * sqrt(Q / (n (n - 1 + bar^2)))."""
    # = sqrt(n Q) * bar / sqrt(bar^2 + n - 1), written with hypot so neither bar^2 overflows nor the quotient
    # underflows (fold of the P1 panel's F2: [1, 2] at bar 1e155 had read threshold 0.0 and fpr 0.5 against 0.0)
    return math.sqrt(n * q) * (bar / math.hypot(bar, math.sqrt(n - 1)))


def _band_eps(n, q):
    """Half-width of the band around the threshold inside which a flip is decided by `paired_t` itself.

    A conservative bound on the combined floating error, as an absolute error on S'. Write s = sqrt(n*Q) and
    u = 2^-53; s bounds S', thr and A = sum|d| (Cauchy-Schwarz; thr < s for n >= 2); the sums below add
    non-negative terms, so each carries a relative error of at most (number of additions) * u.
      (a) a half-sum of h terms plus the final s_left + s_right: at most n*u*A <= n*u*s;
      (b) thr: Q has relative error <= (n + 1)u, the remaining operations (n*Q, sqrt(n - 1), hypot, the
          division and the product, each <= 2u) add <= 10u, the square roots halve their part; at most
          (n + 12)u * thr <= (n + 12)u*s;
      (c) `paired_t` on the reconstructed vector: its mean carries n*u*A/S' + u, its centred variance sums n
          NON-negative squares (no cancellation: <= (n + 3)u, the second-order n*(mean error)^2/var is far
          smaller), the division, square root and division add 3u: a relative error eta <= n*u*A/S' +
          (n + 3)u/2 + 5u on t. t grows at least like S' (d ln t / dS' >= 1/S'), so eta moves the decision
          boundary by at most eta*S' <= (1.5 n + 7)u*s;
      (d) the two bisect keys (thr +- eps) - s: <= 4u*s.
    The total is <= (3.5 n + 23)u*s, below 4(n + 16)u*s, about 2e-14*s at n = 31; the bound is floored at
    1e-9*s (about 10^5 times that at n = 31; the cost is only a larger, still tiny, band re-decided by
    `paired_t`). The bound assumes no square under- or overflows: `permutation_fpr_detail` refuses a Q outside
    (_Q_MIN, _Q_MAX).
    """
    scale = math.sqrt(n * q)
    return max(4.0 * (n + 16) * _U * scale, 1e-9 * scale)


def _flipped(xs, mask):
    return [(-d if (mask >> i) & 1 else d) for i, d in enumerate(xs)]


def _decide(flipped, bar):
    """THE definition of a hit; every flip the threshold rule cannot certify is decided here."""
    return paired_t(flipped) > bar


def _equal_flip_masks(xs):
    """The masks of the flips whose values are all equal (possible only when every |d| is equal): all +m and
    all -m. `paired_t` returns 0.0 for them whatever the threshold says (zero variance)."""
    if len({abs(x) for x in xs}) != 1:
        return ()
    all_pos = sum(1 << i for i, x in enumerate(xs) if x < 0)
    return (all_pos, ((1 << len(xs)) - 1) ^ all_pos)


def permutation_fpr_detail(diffs, bar=T_BAR):
    """(fpr, band_population): the exact false-positive rate of this test on THIS sample -- the fraction of the
    2^n sign flips whose paired t exceeds the bar -- and the number of flips the threshold rule left to
    `paired_t` (the certified band). n < 2 or a zero sum of squares: (0.0, 0), as the 2^n loop gave. Refused
    (ValueError), never approximated: bar not in (0, inf), n above MITM_MAX_N, a sum of squares outside
    (_Q_MIN, _Q_MAX), a band of more than MITM_MAX_BAND flips. A non-finite difference makes every t nan:
    (0.0, 0), as the definition gives."""
    xs = [float(d) for d in diffs]
    n = len(xs)
    # the refusals come BEFORE the trivial answers (fold of the P1 panel's F3: [0, 0] at bar -1 had returned 0.0
    # where the definition gives 1.0, and 41 zeros had returned 0.0 instead of being refused)
    if not 0.0 < bar < math.inf:
        raise ValueError(f"bar must be in (0, inf) (the threshold needs bar > 0), got {bar!r}")
    if n > MITM_MAX_N:
        raise ValueError(f"n = {n} exceeds MITM_MAX_N = {MITM_MAX_N}: refused, never approximated")
    if n < 2 or not all(map(math.isfinite, xs)):
        return 0.0, 0
    q = sum(x * x for x in xs)
    if q == 0.0:
        return 0.0, 0
    if not _Q_MIN < q < _Q_MAX:
        raise ValueError(
            f"sum of squares {q!r} is outside ({_Q_MIN}, {_Q_MAX}): the bound needs it in range"
        )
    h = n // 2
    left = _signed_sums(xs[:h])
    right = _signed_sums(xs[h:])
    order = sorted(range(len(right)), key=right.__getitem__)
    rsorted = [right[j] for j in order]
    top = len(rsorted)
    thr = _threshold(n, bar, q)
    eps = _band_eps(n, q)
    hi, lo = thr + eps, thr - eps
    clear = 0
    spans = []  # (left index, first, last + 1) positions of the band in the sorted right sums
    for i, s in enumerate(left):
        a = bisect_right(rsorted, hi - s)  # right sums above this: S' > thr + eps, a hit
        b = bisect_left(rsorted, lo - s)  # right sums below this: S' < thr - eps, a miss
        clear += top - a
        if a > b:
            spans.append((i, b, a))
    band = sum(a - b for _, b, a in spans)
    if band > MITM_MAX_BAND:
        raise ValueError(f"{band} flips lie within the threshold band (> {MITM_MAX_BAND}): refused")
    hits = clear + sum(
        _decide(_flipped(xs, i | (order[j] << h)), bar) for i, b, a in spans for j in range(b, a)
    )
    # the zero-variance flips are decided by paired_t whichever path counted them: one in the band was
    # decided by paired_t already, one above it was counted as a hit (take that back), one below it was not
    for mask in _equal_flip_masks(xs):
        s, r = left[mask & ((1 << h) - 1)], right[mask >> h]
        if not lo - s <= r <= hi - s:
            hits += _decide(_flipped(xs, mask), bar) - (r > hi - s)
    return hits / float(1 << n), band


def permutation_fpr(diffs, bar=T_BAR):
    """The false-positive rate of this test on THIS sample: the fraction of the 2^S sign flips
    whose t exceeds the bar. Printed beside every contrast (Part F's preamble). Exact, by
    meet-in-the-middle, for every n up to MITM_MAX_N (`permutation_fpr_detail`)."""
    return permutation_fpr_detail(diffs, bar)[0]


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
    import random

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
    # the meet-in-the-middle count against an independent itertools enumeration at n = 12 (the n = 8 witness
    # above is the same enumeration at the recorded size)
    rng = random.Random(20261002)
    d12 = [rng.gauss(0.01, 0.02) for _ in range(12)]
    hits12 = sum(
        1
        for signs in itertools.product((1.0, -1.0), repeat=12)
        if paired_t([s * d for s, d in zip(signs, d12, strict=True)]) > T_BAR
    )
    assert permutation_fpr(d12) == hits12 / 4096.0, (permutation_fpr(d12), hits12)
    # n = 31 (no enumeration possible): the rate is invariant to reversing the sample and to negating it
    d31 = [rng.gauss(0.005, 0.02) for _ in range(31)]
    fpr31, band31 = permutation_fpr_detail(d31, T_BAR_S31)
    assert (
        permutation_fpr(d31[::-1], T_BAR_S31)
        == fpr31
        == permutation_fpr([-d for d in d31], T_BAR_S31)
    )
    # RED: a bar that is not positive, and a sample above MITM_MAX_N, are refused -- never a silent loop
    for bad_diffs, bad_bar in (
        (d12, -1.0),
        ([0.01 * (1 + i % 3) for i in range(MITM_MAX_N + 1)], T_BAR),
    ):
        try:
            permutation_fpr(bad_diffs, bad_bar)
        except ValueError:
            continue
        raise AssertionError("refusal did not fire")
    print(
        f"self-check OK: paired t {up.t:.4f} > {T_BAR} on 8 same-sign seeds (exact FPR {up.fpr}); "
        f"reversed {down.t:.4f} NOT MET; sign-balanced noise t {noise.t:.4f} NOT MET; degenerate positive MET, "
        "degenerate zero NOT MET; deterministic; "
        f"meet-in-the-middle == enumeration at n = 12 ({hits12}/4096); n = 31 at {T_BAR_S31}: FPR {fpr31:.6f} "
        f"(band {band31} flips), reversal- and negation-invariant; bar <= 0 and n = {MITM_MAX_N + 1} refused"
    )


if __name__ == "__main__":
    main()
