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

import hashlib
import json
import math
import random
import zlib
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
# the Monte-Carlo FPR beyond MITM_MAX_N (A0d): 200,000 draws bound the printed SE at .0005 at p = .05
MC_DRAWS = 200_000


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


# graph-exists 2026-10-28 ("seeded Monte-Carlo sign-flip permutation false positive rate"): one hit,
# permutation_fpr_detail above -- REUSE of its threshold rule (_threshold, _band_eps, _decide, the Q range); the
# exact path is untouched and still refuses n > MITM_MAX_N. This is the approximate mode BESIDE it (A0d, P2: one
# mechanism), a sampled count of the same event, labelled MC and never presented as exact.
def permutation_fpr_mc(diffs, bar=T_BAR, *, row, look, draws=MC_DRAWS):
    """(fpr, se): a seeded Monte-Carlo estimate of the fraction of sign flips whose paired t exceeds the bar,
    for any n >= 2. `draws` flips, seeded by `zlib.crc32` of the row and the look; se is the binomial standard
    error of the estimate (p floored at 1/draws so a zero count does not read as certain). Flips inside the
    exact path's band around the threshold are decided by `paired_t`, as there. Same trivial answers and
    refusals as `permutation_fpr_detail` except n above MITM_MAX_N."""
    xs = [float(d) for d in diffs]
    n = len(xs)
    if not 0.0 < bar < math.inf:
        raise ValueError(f"bar must be in (0, inf) (the threshold needs bar > 0), got {bar!r}")
    if n < 2 or not all(map(math.isfinite, xs)):
        return 0.0, 0.0
    q = sum(x * x for x in xs)
    if q == 0.0:
        return 0.0, 0.0
    if not _Q_MIN < q < _Q_MAX:
        raise ValueError(
            f"sum of squares {q!r} is outside ({_Q_MIN}, {_Q_MAX}): the bound needs it in range"
        )
    thr, eps = _threshold(n, bar, q), _band_eps(n, q)
    rng = random.Random(zlib.crc32(f"{row}|{look}".encode()))
    signs = (1.0, -1.0)
    equal = len({abs(x) for x in xs}) == 1  # zero-variance flips: paired_t reads 0.0, as in exact
    hits = 0
    for _ in range(draws):
        flips = rng.choices(signs, k=n)
        total = sum(map(float.__mul__, flips, xs))
        if total > thr + eps:
            hits += _decide([f * x for f, x in zip(flips, xs)], bar) if equal else 1
        elif total >= thr - eps:
            hits += _decide([f * x for f, x in zip(flips, xs)], bar)
    p = hits / draws
    return p, math.sqrt(max(p, 1.0 / draws) * (1.0 - p) / draws)


def mc_label(fpr, se, draws=MC_DRAWS):
    """The printed form of a Monte-Carlo FPR: always says MC, the draws and the SE."""
    return f"MC {fpr:.4f} (SE {se:.4f}, {draws} sign-flip draws; not exact)"


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
    assert _g9_constructed_selfcheck()


def _g9_profile():
    """The frozen constructed DATA profile, never an engine/resource attestation."""
    expected = {}
    for split, items in (("validation", 8), ("future", 12)):
        expected[split] = [
            {
                "presentation_id": f"{split}-presentation-{i:02d}",
                "source_item_id": f"{split}-item-{i % items:02d}",
                "target": ("GREET", "GOAL", "GOAL", "GREET")[i % 4],
            }
            for i in range(4 * items)
        ]
    return {
        "schema": "g9-constructed-v1",
        "origin": "CONSTRUCTED",
        "profile_id": "fixture-profile-v1",
        "source_revision": "fixture-source-v1",
        "protocol_id": "fixture-proposed-contract-v1",
        "primary": {"problem_id": "c1d", "transport_form": "copy", "judging_split": "future"},
        "pair_keys": [f"constructed-pair-{i}" for i in range(8)],
        "stable_epochs": 10,
        "t_bar": T_BAR,
        "tie_tolerance": 1e-6,
        "secondary_keys": [
            {"problem_id": "c1d", "transport_form": "activity", "judging_split": "future"}
        ],
        "expected": expected,
        "definitions": {
            "resource": {
                "inventory_id": "fixture-inventory-v1",
                "boundary_id": "fixture-final-pre-sleep",
                "signature": {
                    "grown": {
                        "allocated": 6,
                        "active_e": 4,
                        "active_i": 2,
                        "input_shape": [6, 4],
                        "dtype": "fixture-f64",
                    },
                    "pool": {
                        "allocated": 8,
                        "active": 8,
                        "recurrent_shape": [8, 8],
                        "dtype": "fixture-f64",
                    },
                    "feed": {
                        "shape": [8, 6],
                        "allocated_connections": 48,
                        "effective_connections": 48,
                        "dtype": "fixture-f64",
                    },
                    "readout": {"shape": [2, 8], "label_count": 2, "dtype": "fixture-f64"},
                    "return": {"shape": [2, 2], "allocated_connections": 4, "dtype": "fixture-f64"},
                    "state": {"shape": [8], "trace_shape": [6], "dtype": "fixture-f64"},
                },
            },
            "reset": {
                "policy_id": "fixture-reset-v1",
                "episode_boundary": "RESET",
                "between_turns": "CARRY",
                "initial_state_id": "fixture-zero-state",
            },
            "exposure": {
                "policy_id": "fixture-exposure-v1",
                "teaching_order": [f"fixture-teach-{i}" for i in range(4)],
                "candidate_order": [f"fixture-candidate-{i}" for i in range(3)],
            },
            "control": {
                "policy_id": "fixture-control-v1",
                "growth_off_arm": "A5",
                "source_family": "FIXTURE_FROZEN",
                "physical_credit": "NONE",
            },
            "retention": {
                "policy_id": "fixture-retention-v1",
                "old_task_ids": ["fixture-old-task"],
                "predicate": "EXACT",
                "measured_credit": "NONE",
            },
        },
    }


def _g9_same(value, expected):
    """Exact nested schema/value equality, including bool versus int versus float."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(
            _g9_same(value[k], v) for k, v in expected.items()
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _g9_same(a, b) for a, b in zip(value, expected, strict=True)
        )
    return value == expected


def _g9_require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _g9_metadata(profile, key, arm, form):
    definitions = profile["definitions"]
    return {
        "origin": "CONSTRUCTED",
        "profile_id": profile["profile_id"],
        "source_revision": profile["source_revision"],
        "protocol_id": profile["protocol_id"],
        "problem_id": "c1d",
        "transport_form": form,
        "judging_split": "future",
        "pair_key": key,
        "arm": arm,
        "terminal": "COMPLETED",
        "growth_off": arm == "A5",
        "attempt_id": "fixture-attempt-0",
        "resource_receipt": definitions["resource"],
        "reset_receipt": definitions["reset"],
        "exposure_receipt": definitions["exposure"],
        "control_policy_id": definitions["control"]["policy_id"],
        "retention_policy_id": definitions["retention"]["policy_id"],
    }


def _g9_correct(answers, expected):
    _g9_require(
        type(answers) is list and len(answers) == len(expected), "incomplete judged answer list"
    )
    correct = 0
    for answer, target in zip(answers, expected, strict=True):
        _g9_require(
            type(answer) is dict
            and answer.keys() == {"presentation_id", "source_item_id", "answer"},
            "answer schema must contain identities and an actual answer only",
        )
        _g9_require(
            type(answer["presentation_id"]) is str
            and answer["presentation_id"] == target["presentation_id"]
            and type(answer["source_item_id"]) is str
            and answer["source_item_id"] == target["source_item_id"],
            "judged presentation/source identities differ from the ordered fixed list",
        )
        _g9_require(
            type(answer["answer"]) is str and answer["answer"] in ("GREET", "GOAL"),
            "invalid answer label",
        )
        correct += answer["answer"] == target["target"]
    return correct


def _g9_record_read(record, profile):
    _g9_require(type(record) is dict, "record must be a plain dictionary")
    key, arm, form = record.get("pair_key"), record.get("arm"), record.get("transport_form")
    _g9_require(
        type(key) is str and key in profile["pair_keys"], "unknown constructed pair identity"
    )
    _g9_require(type(arm) is str and arm in ("A0", "A5"), "unknown constructed arm")
    _g9_require(
        type(form) is str and form in ("copy", "activity"), "unknown primary/secondary form"
    )
    metadata = _g9_metadata(profile, key, arm, form)
    _g9_require(
        record.keys() == metadata.keys() | {"validation", "future"},
        "unknown/incomplete record fields",
    )
    _g9_require(
        _g9_same({k: record[k] for k in metadata}, metadata),
        "record identity, terminal, growth-OFF or complete "
        "resource/reset/exposure/control/retention mismatch",
    )
    validation = record["validation"]
    _g9_require(
        type(validation) is list and len(validation) == 10,
        "missing ten consecutive validation reads",
    )
    stable = True
    for epoch, read in enumerate(validation, 1):
        _g9_require(
            type(read) is dict and read.keys() == {"attempt_id", "epoch", "answers"},
            "invalid epoch-end validation schema",
        )
        _g9_require(
            type(read["epoch"]) is int
            and read["epoch"] == epoch
            and type(read["attempt_id"]) is str
            and read["attempt_id"] == record["attempt_id"],
            "validation must be consecutive epochs 1..10 in one fixed attempt",
        )
        correct = _g9_correct(read["answers"], profile["expected"]["validation"])
        stable = stable and correct == len(profile["expected"]["validation"])
    future_correct = _g9_correct(record["future"], profile["expected"]["future"])
    return (form, key, arm), (stable, future_correct)


def evaluate_g9_primary_records(contract, records, *, purpose):
    """Pure fixed synthetic record logic; every real/deciding use is REFUSED.

    A constructed MET supplies no engine growth, physical attribution, producer
    provenance, retention or fresh-process evidence. The actual runtime contracts
    remain unresolved even when these fixture records pass.
    """
    report = {
        "scope": "G-R9(i) record logic only",
        "constructed_status": "REFUSED",
        "deciding_status": "REFUSED",
        "campaign_eligible": False,
        "unresolved_runtime": ["resource", "reset", "control", "retention"],
        "reasons": [],
    }
    if type(purpose) is not str or purpose != "CONSTRUCTED":
        report["reasons"] = ["this version unconditionally refuses real/deciding purposes"]
        return report
    profile = _g9_profile()
    try:
        _g9_require(
            _g9_same(contract, profile), "contract is not the complete fixed constructed profile"
        )
        _g9_require(
            type(records) is list and 16 <= len(records) <= 32,
            "incomplete/excess constructed arm coverage",
        )
        readings = {}
        for record in records:
            identity, reading = _g9_record_read(record, profile)
            _g9_require(identity not in readings, "duplicate constructed form/pair/arm identity")
            readings[identity] = reading
        _g9_require(
            all(
                ("copy", key, arm) in readings
                for key in profile["pair_keys"]
                for arm in ("A0", "A5")
            ),
            "missing primary A0/A5 pair coverage; secondary records cannot substitute",
        )
    except ValueError as error:
        report["reasons"] = [str(error)]
        return report
    rows, diffs = [], []
    count = len(profile["expected"]["future"])
    for key in profile["pair_keys"]:
        stable0, correct0 = readings[("copy", key, "A0")]
        stable5, correct5 = readings[("copy", key, "A5")]
        q0, q5 = correct0 / count, correct5 / count
        # The same q0-q5 contrast, from integer counts to avoid cancellation.
        diffs.append((correct0 - correct5) / count)
        rows.append(
            {
                "pair_key": key,
                "future_count": count,
                "correct0": correct0,
                "correct5": correct5,
                "q0": q0,
                "q5": q5,
                "engine_validation_stable": stable0,
                "twin_validation_stable": stable5,
                "engine_future_solved": correct0 == count,
                "twin_future_solved": correct5 == count,
            }
        )
    t = paired_t(diffs)
    degenerate = len(set(diffs)) == 1
    advantage = not degenerate and t > T_BAR
    engine_all8 = all(
        row["engine_validation_stable"] and row["engine_future_solved"] for row in rows
    )
    twin_failure = all(not row["twin_future_solved"] for row in rows)
    for flag, reason in (
        (engine_all8, "not every primary engine is validation-stable and future-exact"),
        (twin_failure, "not every valid growth-OFF twin fails its future exact predicate"),
        (advantage, "paired advantage requires nondegenerate t strictly above the inherited bar"),
    ):
        if not flag:
            report["reasons"].append(reason)
    report.update(
        {
            "constructed_status": (
                "MET" if engine_all8 and twin_failure and advantage else "NOT_MET"
            ),
            "primary": profile["primary"],
            "profile_id": profile["profile_id"],
            "source_revision": profile["source_revision"],
            "protocol_id": profile["protocol_id"],
            "contract_sha256": hashlib.sha256(
                json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "sample_size": 8,
            "secondary_count": sum(form == "activity" for form, _, _ in readings),
            "diffs": diffs,
            "t": t,
            "bar": T_BAR,
            "fpr": permutation_fpr(diffs, T_BAR),
            "degenerate": degenerate,
            "non_tied_count": sum(abs(d) > profile["tie_tolerance"] for d in diffs),
            "advantage": advantage,
            "engine_all8": engine_all8,
            "twin_future_failure_all8": twin_failure,
            "per_seed": rows,
        }
    )
    return report


def _g9_fixture_answers(expected, errors=0):
    return [
        {
            "presentation_id": row["presentation_id"],
            "source_item_id": row["source_item_id"],
            "answer": (
                ("GOAL" if row["target"] == "GREET" else "GREET") if i < errors else row["target"]
            ),
        }
        for i, row in enumerate(expected)
    ]


def _g9_selfcheck_fixture():
    profile, records = _g9_profile(), []
    for i, key in enumerate(profile["pair_keys"]):
        for arm in ("A0", "A5"):
            record = _g9_metadata(profile, key, arm, "copy")
            record["validation"] = [
                {
                    "attempt_id": "fixture-attempt-0",
                    "epoch": epoch,
                    "answers": _g9_fixture_answers(profile["expected"]["validation"]),
                }
                for epoch in range(1, 11)
            ]
            record["future"] = _g9_fixture_answers(
                profile["expected"]["future"], i + 1 if arm == "A5" else 0
            )
            records.append(record)
    return profile, records


def _g9_constructed_selfcheck():
    profile, records = _g9_selfcheck_fixture()
    before = json.dumps([profile, records], sort_keys=True)
    positive = evaluate_g9_primary_records(profile, records, purpose="CONSTRUCTED")
    assert positive["constructed_status"] == "MET" and positive["engine_all8"]
    assert positive["twin_future_failure_all8"] and math.isclose(positive["t"], math.sqrt(27))
    assert before == json.dumps([profile, records], sort_keys=True)
    assert positive == evaluate_g9_primary_records(
        profile, list(reversed(records)), purpose="CONSTRUCTED"
    )
    readings = [("positive", positive)]
    records[0]["future"][0]["answer"] = "GOAL"
    wrong = evaluate_g9_primary_records(profile, records, purpose="CONSTRUCTED")
    assert wrong["constructed_status"] == "NOT_MET" and not wrong["engine_all8"]
    readings.append(("one wrong engine answer", wrong))
    profile, records = _g9_selfcheck_fixture()
    records[1]["future"] = _g9_fixture_answers(profile["expected"]["future"])
    records[1]["validation"][-1]["answers"] = _g9_fixture_answers(
        profile["expected"]["validation"], 1
    )
    twin = evaluate_g9_primary_records(profile, records, purpose="CONSTRUCTED")
    assert twin["constructed_status"] == "NOT_MET" and not twin["twin_future_failure_all8"]
    assert (
        not twin["per_seed"][0]["twin_validation_stable"]
        and twin["per_seed"][0]["twin_future_solved"]
    )
    readings.append(("seven twin failures with one future solve", twin))
    profile, records = _g9_selfcheck_fixture()
    for record in records:
        if record["arm"] == "A5":
            record["future"] = _g9_fixture_answers(profile["expected"]["future"], 3)
    constant = evaluate_g9_primary_records(profile, records, purpose="CONSTRUCTED")
    assert (
        constant["constructed_status"] == "NOT_MET"
        and constant["degenerate"]
        and not constant["advantage"]
    )
    assert contrast("legacy positive constant", [0.25] * 8).met
    readings.append(("constant-positive known cost", constant))
    incomplete = evaluate_g9_primary_records(profile, records[:-1], purpose="CONSTRUCTED")
    deciding = evaluate_g9_primary_records(None, None, purpose="DECIDING")
    assert incomplete["constructed_status"] == deciding["constructed_status"] == "REFUSED"
    readings.extend((("missing primary arm", incomplete), ("deciding request", deciding)))
    for label, reading in readings:
        assert reading["deciding_status"] == "REFUSED" and reading["campaign_eligible"] is False
        print(
            f"G9 CONSTRUCTED {label}: {reading['constructed_status']} "
            f"t={reading.get('t')} FPR={reading.get('fpr')}"
        )
    print(
        "G9 CONSTRUCTED record checks OK; deciding REFUSED; engine/provenance/retention unresolved"
    )
    return True


if __name__ == "__main__":
    main()
