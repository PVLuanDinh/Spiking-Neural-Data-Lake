"""MAP D0 (research/representational-drift/MAP.md section 4): the representational-drift
instrument's constructed reds, each run before any claim is read. The instrument is pure
measurement; these pin that every reading it can return is reachable both ways.

1. A/A: identical sessions read exactly zero drift.
2. A planted orthogonal rotation is recovered at graded doses, and drift grows with the dose.
3. Planted drift confined to null(W) reads a null fraction of 1, row-space drift reads 0, and a
   random readout of matched rank reads about (n - rank) / n.
4. A preferred-orientation shift across the 0/180 wrap reads its true size.
5. An over-parameterised decoder is refused by the capacity guard.
6. A non-orthogonal planted drift is NOT "recovered" by Procrustes.
"""

import numpy as np
import pytest

from snn_data_lake.instruments import representational_drift as rd


def _session(seed, n=40, s=12):
    return np.random.default_rng(seed).normal(size=(n, s))


def _rotation(n, dose, seed=0):
    """An orthogonal n x n matrix exp(dose * K) for a fixed random skew-symmetric K: dose 0 is the
    identity and the rotation grows smoothly with the dose."""
    from scipy.linalg import expm

    a = np.random.default_rng(seed).normal(size=(n, n))
    k = (a - a.T) / np.sqrt(n)
    return expm(dose * k)


def test_red1_identical_sessions_read_exactly_zero_drift():
    a = _session(1)
    assert rd.pv_correlation(a, a.copy()) == 1.0
    assert rd.drift_index(a, a.copy()) == 0.0
    assert np.all(rd.tuning_correlation(a, a.copy()) == 1.0)


def test_red2_a_planted_rotation_is_recovered_at_graded_doses():
    # samples x neurons, MORE samples than neurons: with fewer, R is not identifiable (only the
    # residual is), so the recovered-equals-planted check would be vacuous or wrong.
    a = _session(2, n=12, s=60).T
    drifts = []
    for dose in (0.05, 0.2, 0.5, 1.0):
        r = _rotation(a.shape[1], dose)
        b = a @ r
        realign, rel = rd.procrustes_realign(b, a)
        assert rd.recovered(rel), (dose, rel)
        assert np.allclose(realign, r.T, atol=1e-8), dose
        drifts.append(rd.drift_index(a.T, b.T))
    assert drifts == sorted(drifts) and drifts[0] > 0.0, drifts


def test_red3_null_space_drift_reads_one_row_space_reads_zero_random_reads_chance():
    rng = np.random.default_rng(3)
    n, k = 30, 4
    w = rng.normal(size=(k, n))
    _, _, vt = np.linalg.svd(w)
    row_basis, null_basis = vt[:k], vt[k:]
    in_null = null_basis.T @ rng.normal(size=(n - k, 5))
    in_row = row_basis.T @ rng.normal(size=(k, 5))
    assert rd.readout_null_split(in_null, w) == pytest.approx(1.0, abs=1e-12)
    assert rd.readout_null_split(in_row, w) == pytest.approx(0.0, abs=1e-12)
    isotropic = rng.normal(size=(n, 200))
    fracs = rd.random_readout_null_fractions(isotropic, rank=k, n_draws=200, seed=4)
    assert abs(float(np.mean(fracs)) - (n - k) / n) < 0.02, float(np.mean(fracs))


def test_red3b_a_zero_change_has_no_null_fraction():
    with pytest.raises(ValueError):
        rd.readout_null_split(np.zeros(10), np.ones((2, 10)))


def test_red4_orientation_shift_across_the_wrap_reads_its_true_size():
    assert rd.preferred_shift(170.0, 10.0) == pytest.approx(20.0)
    assert rd.preferred_shift(10.0, 170.0) == pytest.approx(-20.0)
    assert rd.preferred_shift(45.0, 45.0) == 0.0
    assert rd.preferred_shift(350.0, 10.0, period=360.0) == pytest.approx(20.0)


def test_red5_an_over_parameterised_decoder_is_refused():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(10, 10))  # 10 features + bias > 10 samples: it would interpolate anything
    y = np.arange(10) % 2
    with pytest.raises(rd.CapacityRefusal):
        rd.decoder_accuracy(x, y, x, y, k=2)


def test_red6_a_non_orthogonal_drift_is_not_recovered_by_procrustes():
    a = _session(6).T
    shear = np.eye(a.shape[1])
    shear[: a.shape[1] // 2] *= 1.5  # half the neurons gain 50%: no rotation can undo it
    realign, rel = rd.procrustes_realign(a @ shear, a)
    assert not rd.recovered(rel), rel


def test_reading_aa_is_null_and_a_planted_rotation_is_rescued_by_realignment():
    rng = np.random.default_rng(7)
    k, n, per = 3, 25, 40
    means = rng.normal(scale=3.0, size=(k, n))
    y = np.repeat(np.arange(k), per)
    x0 = means[y] + rng.normal(size=(k * per, n))
    same = rd.read_drift(x0, y, x0.copy(), y, k=k)
    assert same.drift_index == 0.0 and same.frozen_acc_day1 == same.frozen_acc_day0
    x1 = x0 @ _rotation(n, 1.0, seed=8)
    moved = rd.read_drift(x0, y, x1, y, k=k)
    assert moved.drift_index > 0.0
    assert moved.frozen_acc_day1 < moved.frozen_acc_day0
    assert moved.realigned_acc_day1 > moved.frozen_acc_day1
