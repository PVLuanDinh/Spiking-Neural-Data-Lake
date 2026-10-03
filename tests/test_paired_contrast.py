"""The exact sign-flip false-positive rate by meet-in-the-middle, against brute force (G1(c) P1-A).

specs/022-unified-brain-arc/evidence/g1c-carried-state.md 2.3 and 2.10 (P1): `permutation_fpr`
counts the fraction of the 2^n sign flips whose paired t exceeds the bar. It was a 2^n loop (hours
at n = 31); it is now a meet-in-the-middle count whose definition is unchanged, so the loop survives
HERE, and only here, as the ORACLE: `itertools.product` plus the module's own `paired_t`, sharing
nothing with the fast path's internals.

  (a) 160 random samples at n 8-14 (ties included), at the two bars the plan uses: equality;
  (b) lattice samples shaped like the real statistic (a mean of three segment differences);
  (c) adversarial fixtures: cancellation, a 1e-8..1 dynamic range, a flip constructed to land
      EXACTLY on the threshold, all-equal magnitudes (the zero-variance flips), a non-finite
      sample, the trivial samples;
  (d) n = 18 and n = 20 against the oracle;
  (e) n = 31 (no oracle): invariance to reversal, a shuffle, negation, scaling by 2.0 and |d|;
  (f) REDS: four mutants of the fast path's internals, each of which must DISAGREE with the oracle
      on its fixture while the unmutated path agrees (a control that is not the method itself);
  (g) the refusals (bar not positive, n above 40, a sum of squares out of range, a huge band).

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_paired_contrast.py -q --no-cov
"""

import itertools
import math
import random
import time

import pytest

import snn_data_lake.instruments.paired_contrast as pc
from snn_data_lake.instruments.paired_contrast import (
    T_BAR,
    T_BAR_S31,
    paired_t,
    permutation_fpr,
    permutation_fpr_detail,
)

BARS = (T_BAR, T_BAR_S31)


# ---- the oracle: plain enumeration, the module's paired_t, nothing of the fast path ------------
def _flip_ts(d):
    """paired_t of each of the 2^n sign flips (the oracle's one expensive pass)."""
    return [
        paired_t([s * x for s, x in zip(signs, d, strict=True)])
        for signs in itertools.product((1.0, -1.0), repeat=len(d))
    ]


def _oracle(d, bar, ts=None):
    ts = _flip_ts(d) if ts is None else ts
    return sum(t > bar for t in ts) / float(1 << len(d))


def _assert_matches_oracle(d, bars=BARS):
    ts = _flip_ts(d)
    for bar in bars:
        assert permutation_fpr(d, bar) == _oracle(d, bar, ts), (d, bar)


# ---- (a) random samples ------------------------------------------------------------------------
def test_a_160_random_samples_equal_the_oracle_at_both_bars():
    rng = random.Random(20261002)
    checks = 0
    for n in (8, 10, 12, 14):
        for trial in range(40):
            d = [rng.gauss(0.01 * (trial % 4), 0.02) for _ in range(n)]
            if trial % 10 == 0:
                d = [round(x, 2) for x in d]  # ties on purpose
            _assert_matches_oracle(d)
            checks += 2
    assert checks == 320


# ---- (b) lattice samples like the real statistic -----------------------------------------------
def test_b_lattice_samples_equal_the_oracle():
    rng = random.Random(20261003)
    for _ in range(40):
        n = rng.randint(8, 14)
        d = [
            (rng.randint(-20, 20) / 463 + rng.randint(-20, 20) / 256 + rng.randint(-20, 20) / 210)
            / 3
            for _ in range(n)
        ]
        _assert_matches_oracle(d)


# ---- (c) adversarial fixtures ------------------------------------------------------------------
def test_c_cancellation_large_and_tiny_magnitudes_together():
    _assert_matches_oracle([1.0, 1e-9, -1.0, 1e-9, 1.0, -1e-9, 1e-9, -1.0, 1.0, 1e-9])
    _assert_matches_oracle([1.0, -1.0, 1.0, -1.0, 1.0, 1e-9, -1e-9, 1e-9, 1e-9, -1e-9, 1e-9, 1.0])
    _assert_matches_oracle([1e8, -1e8, 1.0, 1e-8, 1e8, 1e-8, -1.0, 1e8, -1e8, 1.0])


def test_c_dynamic_range_1e_8_to_1():
    geometric = [(-1.0) ** k * 10.0 ** (-8.0 * k / 11.0) for k in range(12)]
    _assert_matches_oracle(geometric)
    _assert_matches_oracle([abs(x) for x in geometric])
    rng = random.Random(20261004)
    for _ in range(6):
        d = [rng.choice((-1.0, 1.0)) * 10.0 ** rng.uniform(-8.0, 0.0) for _ in range(11)]
        _assert_matches_oracle(d)


def test_c_a_flip_constructed_to_land_exactly_on_the_threshold():
    # [3.0, 0.0] at bar 1.0: thr = 2 * sqrt(9 / 4) = 3.0 exactly and paired_t([3.0, 0.0]) = 1.0,
    # which is NOT > 1.0: no hit, and the two flips that sit on the threshold are in the band
    fpr, band = permutation_fpr_detail([3.0, 0.0], 1.0)
    assert fpr == 0.0 == _oracle([3.0, 0.0], 1.0), fpr
    assert band >= 1, band
    # [0.7, -1.4] at bar 1/3: the flip [-0.7, 1.4] has S' = 0.7 = thr in exact arithmetic and
    # paired_t reads it ABOVE the bar in floats (the oracle says 1/2): the band re-decision gets it
    fpr, band = permutation_fpr_detail([0.7, -1.4], 1.0 / 3.0)
    assert fpr == _oracle([0.7, -1.4], 1.0 / 3.0) == 0.5, fpr
    assert band >= 1, band


def test_c_tie_rich_samples_populate_the_band_and_still_equal_the_oracle():
    # equal magnitudes at a perfect-square n and bar 1.0 put thousands of flips exactly on the
    # threshold (thr = sqrt(n) * c; n = 16: C(16, 6) = 8008 of them), and whether paired_t reads
    # such a flip above the bar depends on the float rounding of c: the band is met in VOLUME
    # here, not only by the two constructed fixtures above
    rng = random.Random(20261006)
    band_total = 0
    for n in (4, 9, 16):
        for c in (0.1, 0.7, 1.0, 3.0):
            d = [rng.choice((-1.0, 1.0)) * c for _ in range(n)]
            ts = _flip_ts(d)
            for bar in (1.0, T_BAR):
                fpr, band = permutation_fpr_detail(d, bar)
                assert fpr == _oracle(d, bar, ts), (d, bar)
                band_total += band
    assert band_total > 30000, band_total


def test_c_all_equal_magnitudes_with_mixed_signs():
    # dyadic: the zero-variance flip is exactly zero-variance in floats
    _assert_matches_oracle([0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5])
    _assert_matches_oracle([0.5] * 8)
    _assert_matches_oracle([-0.5] * 8)
    # non-dyadic: 0.1 + ... + 0.1 does not divide back to 0.1, so paired_t sees a tiny nonzero
    # variance and reads the all-equal flip as a (huge) hit; the expected value is the oracle's
    _assert_matches_oracle([0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1])
    _assert_matches_oracle([0.1] * 8)
    _assert_matches_oracle([-0.1] * 8)
    _assert_matches_oracle([0.1] * 9)


def test_c_a_non_finite_sample_is_zero_as_the_definition_gives():
    for d in ([math.nan, 0.1, 0.2, 0.3], [math.inf, 0.1, 0.2, 0.3], [0.1, -math.inf, 0.2, 0.3]):
        _assert_matches_oracle(d)
        assert permutation_fpr_detail(d) == (0.0, 0)


def test_c_the_trivial_samples_give_zero_as_the_loop_gave():
    for d in ([], [0.3], [0.0] * 8, [0.0, 0.0]):
        assert permutation_fpr(d) == 0.0 == _oracle(d, T_BAR)
        assert permutation_fpr_detail(d) == (0.0, 0)


# ---- (d) n = 18 and n = 20 against the oracle --------------------------------------------------
@pytest.mark.parametrize("n", [18, 20])
def test_d_larger_n_equals_the_oracle(n):
    rng = random.Random(20261005 + n)
    _assert_matches_oracle([rng.gauss(0.008, 0.02) for _ in range(n)])


# ---- (e) n = 31: no oracle, so invariance ------------------------------------------------------
def test_e_n31_invariance_and_time():
    rng = random.Random(20261031)
    d = [rng.gauss(0.005, 0.02) for _ in range(31)]
    t0 = time.perf_counter()
    base = permutation_fpr(d, T_BAR_S31)
    assert time.perf_counter() - t0 < 5.0
    shuffled = list(d)
    rng.shuffle(shuffled)
    for name, variant in (
        ("reversed", d[::-1]),
        ("shuffled", shuffled),
        ("negated", [-x for x in d]),
        ("scaled by 2.0", [2.0 * x for x in d]),
        ("absolute values", [abs(x) for x in d]),
    ):
        t0 = time.perf_counter()
        assert permutation_fpr(variant, T_BAR_S31) == base, name
        assert time.perf_counter() - t0 < 5.0, name
    assert 0.0 < base < 1.0


# ---- (f) the reds: each mutant must DISAGREE with the oracle on its fixture --------------------
def _red(monkeypatch, fixture, bar, mutate):
    """The unmutated path equals the oracle (a sighted control); the mutated path must not."""
    expected = _oracle(fixture, bar)
    assert permutation_fpr(fixture, bar) == expected
    mutate(monkeypatch)
    return permutation_fpr(fixture, bar), expected


def test_f_red_a_greater_equal_boundary_mutant(monkeypatch):
    def mutate(mp):
        mp.setattr(pc, "_decide", lambda flipped, bar: paired_t(flipped) >= bar)

    got, expected = _red(monkeypatch, [3.0, 0.0], 1.0, mutate)
    assert expected == 0.0 and got == 0.5, (got, expected)


def test_f_red_the_bar_dropped_from_the_threshold(monkeypatch):
    fixture = [0.021, 0.034, -0.008, 0.015, 0.027, 0.011, -0.013, 0.019, 0.025, 0.002]
    threshold = pc._threshold

    def mutate(mp):
        mp.setattr(pc, "_threshold", lambda n, bar, q: threshold(n, 0.0, q))

    got, expected = _red(monkeypatch, fixture, T_BAR, mutate)
    assert got != expected, (got, expected)


def test_f_red_the_all_equal_redecision_removed(monkeypatch):
    def mutate(mp):
        mp.setattr(pc, "_equal_flip_masks", lambda xs: ())

    got, expected = _red(monkeypatch, [0.5] * 8, T_BAR, mutate)
    assert got - expected == 1 / 256, (got, expected)  # exactly the one all-positive flip over


def test_f_red_the_band_redecision_removed_with_eps_zero(monkeypatch):
    def mutate(mp):
        mp.setattr(pc, "_band_eps", lambda n, q: 0.0)
        mp.setattr(pc, "_decide", lambda flipped, bar: False)  # the strict threshold rule alone

    got, expected = _red(monkeypatch, [0.7, -1.4], 1.0 / 3.0, mutate)
    assert expected == 0.5 and got == 0.25, (got, expected)


# ---- (g) the refusals: never a silent loop, never an approximation -----------------------------
@pytest.mark.parametrize("bar", [0.0, -1.0, math.nan, math.inf])
def test_g_a_bar_that_is_not_positive_is_refused(bar):
    with pytest.raises(ValueError, match="bar"):
        permutation_fpr([0.01, 0.02, -0.01, 0.03], bar)


def test_g_n_above_40_is_refused():
    with pytest.raises(ValueError, match="MITM_MAX_N"):
        permutation_fpr([0.01 * (1 + i % 3) for i in range(41)])


def test_g_a_sum_of_squares_out_of_range_is_refused():
    for scale in (1e130, 1e-130):
        with pytest.raises(ValueError, match="sum of squares"):
            permutation_fpr([scale, 2 * scale, -scale, 3 * scale])


def test_g_a_band_of_billions_of_flips_is_refused_at_once():
    # 36 ones at bar 1.0: thr = 36 * sqrt(1 / 36) = 6.0 and C(36, 15) = 5.6e9 flips sit on it
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="threshold band"):
        permutation_fpr([1.0] * 36, 1.0)
    assert time.perf_counter() - t0 < 30.0
