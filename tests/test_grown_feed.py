"""specs/022 T111 (G3) P5: the feed -- admitted grown units project into the G1 reservoir.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.9 C15 (the norm-matched
random-feed arm must differ from the fed arm on the pool's carriers; the instrument's positive
control is that a "random" arm equal to the fed matrix does NOT differ) and C17's reservoir-segment
half (Dale's sign on every grown -> reservoir synapse: a planted deposit that would flip a sign is
clipped, a planted stored synapse that violates its unit's sign is refused at restore, the refusal
names the synapse). The port is OFF by default: percept_fed with a zero feed is byte-identical to
percept. Constructed fixtures on seeds 9900-9999 only; nothing here builds a Brain or a lake.
"""

import numpy as np
import pytest

from snn_data_lake.growth import grown_feed as gf
from snn_data_lake.growth.grown_code import GrownUnit, segment_psp
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features
from snn_data_lake.reservoir import recurrent_pool as rp

TAU = 8.0
T_POST = 14
LSM_PRE = [[1, 2], [0, 0], [1, 0]]
LSM_W = [[0.5, 0.25], [1.0, -0.5], [0.5, 0.5]]
LSM_CFG = {"lambda": 0.5, "g": 2, "T_in": 3}
N_POOL = 3


def _pool(window=2):
    return rp.KernelLsmPool(
        LSM_PRE, LSM_W, LSM_CFG, "carried", window=window, w_in=np.full((N_POOL, 64), 0.01)
    )


def _ranks(text="cat dog sun sky"):
    learner = SpikingInteractionLearner(["A", "B"], n_in=52, seed=9900, t_post=6)
    return tuple(learner.encode(text_features(text, 52)))


def _unit(uid, slot, sign=1):
    w = [0.0] * 52
    w[slot] = 2.0
    return GrownUnit(
        unit_id=uid, w_in=tuple(w), threshold=1.0, window=12, sign=sign, out=(0.0, 0.0)
    )


UNITS = (_unit(3, 0, +1), _unit(7, 1, -1), _unit(5, 2, +1))  # unit_id order: 3, 5, 7
# columns in unit_id order (3, 5, 7); E columns >= 0, the I column (id 7) <= 0
W_FEED = np.array([[2.0, 1.0, -1.5], [0.0, 3.0, -2.0], [1.0, 0.5, -3.0]])
X = np.linspace(0.0, 1.0, 64)


def test_the_witness_units_actually_spike_so_the_feed_is_not_vacuous():
    s = segment_psp(UNITS, _ranks(), TAU, T_POST)
    assert len(s) == 3 and all(v > 0.0 for v in s), s  # the detector fires on a known positive


def test_feed_drive_is_the_segment_through_the_declared_synapses_in_unit_id_order():
    s = np.array(segment_psp(UNITS, _ranks(), TAU, T_POST))
    got = gf.feed_drive(UNITS, _ranks(), TAU, T_POST, W_FEED)
    assert got.shape == (N_POOL,) and np.allclose(got, W_FEED @ s)
    shuffled = (UNITS[2], UNITS[0], UNITS[1])  # insertion order never matters
    assert gf.feed_drive(shuffled, _ranks(), TAU, T_POST, W_FEED).tolist() == got.tolist()
    silenced = (UNITS[0], UNITS[1].silenced(), UNITS[2])
    off = gf.feed_drive(silenced, _ranks(), TAU, T_POST, W_FEED)
    assert np.allclose(
        off, W_FEED[:, [0, 1]] @ s[[0, 1]]
    )  # silenced unit 7 (column 2) contributes nothing


def test_port_off_is_todays_bytes():
    base, fed = _pool(), _pool()
    a = base.percept(X)
    b = fed.percept_fed(X, np.zeros(N_POOL))
    assert a.tolist() == b.tolist() and base.fingerprint() == fed.fingerprint()


def test_port_refuses_a_bad_feed():
    for bad in (np.zeros(N_POOL + 1), np.zeros((N_POOL, 1)), np.array([0.0, np.nan, 0.0])):
        with pytest.raises(ValueError, match="REFUSED"):
            _pool().percept_fed(X, bad)


def _carriers_after(feed_matrix):
    pool = _pool(window=1)
    drive = gf.feed_drive(UNITS, _ranks(), TAU, T_POST, feed_matrix)
    pool.percept_fed(X, drive)
    pool.percept_fed(X, drive)
    return pool.fingerprint(), {k: v.tolist() for k, v in pool.carriers().items()}


def test_C15_random_feed_differs_from_fed_and_the_instrument_has_a_positive_control():
    fed = _carriers_after(W_FEED)
    rand = gf.random_feed(W_FEED, UNITS, 9901)
    assert _carriers_after(rand) != fed  # the norm-matched random arm is a different input
    # positive control: an arm equal to the fed matrix does NOT differ (not always-on)
    assert _carriers_after(W_FEED.copy()) == fed
    # a no-feed arm also differs from fed (the feed reaches the carriers at all)
    assert _carriers_after(np.zeros_like(W_FEED)) != fed


def test_random_feed_is_norm_matched_sign_consistent_and_seeded():
    r = gf.random_feed(W_FEED, UNITS, 9901)
    assert np.allclose(np.linalg.norm(r, axis=0), np.linalg.norm(W_FEED, axis=0))
    assert (r[:, 0] >= 0).all() and (r[:, 1] >= 0).all() and (r[:, 2] <= 0).all()
    assert r.tolist() == gf.random_feed(W_FEED, UNITS, 9901).tolist()
    assert r.tolist() != gf.random_feed(W_FEED, UNITS, 9902).tolist()
    assert r.tolist() != W_FEED.tolist()


def test_C17_clip_flips_to_the_unit_sign_and_sign_consistent_deposits_pass_unchanged():
    assert gf.clip_feed(W_FEED, UNITS).tolist() == W_FEED.tolist()
    bad = W_FEED.copy()
    bad[1, 2] = +4.0  # an I unit's deposit that would flip positive
    bad[2, 0] = -4.0  # an E unit's deposit that would flip negative
    clipped = gf.clip_feed(bad, UNITS)
    assert clipped[1, 2] == 0.0 and clipped[2, 0] == 0.0
    assert clipped.tolist() == [[2.0, 1.0, -1.5], [0.0, 3.0, 0.0], [0.0, 0.5, -3.0]]
    assert bad[1, 2] == 4.0  # the input is never mutated


def test_C17_restore_refuses_a_sign_violating_synapse_and_names_it():
    gf.check_feed(W_FEED, UNITS)  # sign-consistent: accepted
    bad = W_FEED.copy()
    bad[1, 2] = 4.0
    with pytest.raises(ValueError, match=r"unit 7.*pool unit 1"):
        gf.check_feed(bad, UNITS)
    with pytest.raises(ValueError, match="REFUSED"):
        gf.check_feed(W_FEED[:, :2], UNITS)  # one column per unit, no more, no fewer
    nan = W_FEED.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"unit 3.*pool unit 0"):
        gf.check_feed(nan, UNITS)


def test_a_non_real_feed_is_refused_before_conversion():
    # Codex b2 MAJOR: asarray(float64) dropped a complex part (1j became 0) and hid a complex NaN
    for bad in (
        np.full(N_POOL, 1j),
        np.array(["a", "b", "c"]),
        np.array([None, 1.0, 2.0], dtype=object),
    ):
        with pytest.raises(ValueError, match="REFUSED"):
            _pool().percept_fed(np.zeros(64), bad)
    wc = W_FEED.astype(complex)
    wc[0, 0] = complex(2.0, np.nan)
    for fn in (gf.check_feed, gf.clip_feed):
        with pytest.raises(ValueError, match="REFUSED"):
            fn(wc, UNITS)
    with pytest.raises(ValueError, match="REFUSED"):
        gf.random_feed(wc, UNITS, 9901)


def test_random_feed_is_scale_safe_and_finite_at_extreme_magnitudes():
    # Codex b2 MINOR: the squared-sum norm overflowed at 1e200 (and underflowed to 0 at 1e-200)
    for scale in (1e200, 1e-200):
        out = gf.random_feed(W_FEED * scale, UNITS, 9901)
        assert np.isfinite(out).all()
        for j, u in enumerate(sorted(UNITS, key=lambda u: u.unit_id)):
            assert (out[:, j] * u.sign >= 0).all()
            ratio = np.max(np.abs(out[:, j])) / np.max(np.abs(W_FEED[:, j] * scale))
            assert 0.0 < ratio < 1e3  # same magnitude order, not inf and not collapsed to zero


def test_random_feed_refuses_a_non_finite_source_column():
    # Codex b2 p2p3 m1: a NaN source column silently became a zero feed arm
    nan = W_FEED.copy()
    nan[0, 1] = np.nan
    with pytest.raises(ValueError, match="REFUSED"):
        gf.random_feed(nan, UNITS, 9901)


def test_random_feed_does_not_overflow_when_the_source_norm_is_representable():
    # Codex b2 safety m1: col * (norm_w / norm) overflowed to inf for a column at 1.7e308
    unit = _unit(0, 0, +1)
    out = gf.random_feed(np.array([[1.7e308], [0.0]]), (unit,), 9900)
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.7e308 * (1 + 1e-12)


def test_random_feed_requires_an_integer_seed_in_g3s_block():
    # Codex b2 safety m2: seed=None drew from OS entropy (non-reproducible); the block is 9000-9999
    for bad in (None, True, 9000.0, "9901", 8999, 10000, 301):
        with pytest.raises(ValueError, match="seed"):
            gf.random_feed(W_FEED, UNITS, bad)
    assert (
        gf.random_feed(W_FEED, UNITS, 9000).tolist() == gf.random_feed(W_FEED, UNITS, 9000).tolist()
    )


def test_random_feed_refuses_an_unrepresentable_column_norm():
    # Codex int2 closing: finite entries whose L2 norm exceeds the float max gave [[inf], [inf]]
    unit = _unit(0, 0, +1)
    with pytest.raises(ValueError, match="REFUSED"):
        gf.random_feed(np.array([[1.7e308], [1.7e308]]), (unit,), 9900)
