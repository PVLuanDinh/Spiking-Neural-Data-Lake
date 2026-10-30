"""G3 feed feature plumbing; constructed 9900-block fixtures, no trained claim."""

import numpy as np
import pytest

import snn_data_lake.reservoir.recurrent_pool as rp
from snn_data_lake.growth.feed_readout import (
    arm_latencies,
    feed_u_contrasts,
    join_fed_unit,
    pool_first_spikes,
    rebuild_joined_pool,
)
from snn_data_lake.growth.grown_code import GrownUnit, latencies
from snn_data_lake.growth.grown_feed import feed_drive
from snn_data_lake.growth.growth_loop import BudgetValues, Ledger, make_admission
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features
from snn_data_lake.reservoir.recurrent_pool import KernelLsmPool


def pool(form="carried", settle=0):
    return KernelLsmPool(
        [[1, 2], [0, 0], [1, 0]],
        [[0.5, 0.25], [1.0, -0.5], [0.5, 0.5]],
        {"lambda": 0.5, "g": 2, "T_in": 3},
        form,
        window=0,
        s_settle=settle,
        w_in=np.full((3, 64), 0.01),
    )


def units():
    out = []
    for uid, slot in ((3, 0), (7, 1), (5, 2)):
        weights = [0.0] * 52
        weights[slot] = 2.0
        out.append(GrownUnit(uid, tuple(weights), 1.0, 12, 1, (0.0, 0.0)))
    return tuple(out)


def ranks():
    unit = SpikingInteractionLearner(["A", "B"], 52, seed=9900, t_post=6)
    return tuple(unit.encode(text_features("cat dog sun sky", 52)))


def test_first_spikes_are_actual_counted_ticks_and_match_uninstrumented_state():
    observed, control = pool("settled", 4), pool("settled", 4)
    seen = []
    observed.observer = lambda _, a: seen.append(a.copy())
    observer = observed.observer
    x, feed = np.zeros(64), np.array([0.6, 0.0, 0.0])
    got = pool_first_spikes(observed, x, feed)
    control.percept_fed(x, feed)
    assert observed.fingerprint() == control.fingerprint()
    assert observed.observer is observer and len(seen) == 7
    counted = seen[-3:]
    expected = tuple(next((t for t, a in enumerate(counted, 1) if a[i]), None) for i in range(3))
    assert got == expected and any(t is not None for t in got)


def test_observer_is_restored_when_the_incumbent_port_refuses():
    p = pool()

    def observer(*_):
        pass

    p.observer = observer
    before = p.fingerprint()
    with pytest.raises(ValueError, match="REFUSED"):
        pool_first_spikes(p, np.zeros(64), np.zeros(4))
    assert p.observer is observer and p.fingerprint() == before


def test_direct_arm_keeps_unit_identity_and_does_not_touch_pool():
    p = pool()
    before = p.fingerprint()
    us, code = units(), ranks()
    got = arm_latencies("direct", us, code, 8.0, 14, seed=9900, pool=p)
    assert got == tuple(latencies(us, code, 8.0))
    assert p.fingerprint() == before
    assert got == arm_latencies("direct", tuple(reversed(us)), code, 8.0, 14, seed=9900)


def test_fed_arm_reuses_the_existing_projection_and_really_reaches_pool_spikes():
    us, code, x = units(), ranks(), np.linspace(0.0, 1.0, 64)
    w = np.array([[2.0, 1.0, 1.5], [0.0, 3.0, 2.0], [1.0, 0.5, 3.0]])
    fed, control = pool(), pool()
    got = arm_latencies("fed", us, code, 8.0, 14, seed=9900, pool=fed, observation=x, w_feed=w)
    expected = pool_first_spikes(control, x, feed_drive(us, code, 8.0, 14, w))
    assert got == expected and fed.fingerprint() == control.fingerprint()
    assert any(t is not None for t in got)


def test_random_feed_is_seeded_and_does_not_mutate_units_or_synapses():
    us, code, x = units(), ranks(), np.zeros(64)
    w = np.array([[0.5, 0.1, 0.2], [0.0, 0.3, 0.2], [0.1, 0.05, 0.3]])
    before = w.copy()
    a, b = pool(), pool()
    args = ("random-feed", us, code, 8.0, 14)
    assert arm_latencies(*args, seed=9900, pool=a, observation=x, w_feed=w) == arm_latencies(
        *args,
        seed=9900,
        pool=b,
        observation=x,
        w_feed=w,
    )
    assert a.fingerprint() == b.fingerprint() and np.array_equal(w, before)


@pytest.mark.parametrize("seed", [9100, 9200, 9300, 9108, 9208, 9308, 300, True, 9900.0])
def test_no_confirmatory_reserve_or_unregistered_seed_can_run_build_arms(seed):
    with pytest.raises(ValueError, match="build/dev fixture seed"):
        arm_latencies("direct", units(), ranks(), 8.0, 14, seed=seed)


def test_feed_superiority_rule_has_reachable_null_and_degeneracy_reds():
    fed = [9] * 8
    positive = feed_u_contrasts(
        fed, [18, 19, 18, 20, 19, 18, 20, 19], [22, 21, 23, 22, 21, 23, 22, 21]
    )
    assert all(row.met and row.t > 1.895 for row in positive.values())
    assert not any(row.met for row in feed_u_contrasts(fed, fed, fed).values())
    uniform = feed_u_contrasts(fed, [18] * 8, [18] * 8)
    assert all(row.degenerate and not row.met for row in uniform.values())
    assert not all(row.met for row in feed_u_contrasts(fed, [1, 2, 1, 2, 1, 2, 1, 2], fed).values())


def test_invalid_or_unpaired_u_scores_are_refused():
    for values in ([1] * 7, [1] * 9, [True] * 8, [1.0] * 8, [-1] * 8):
        with pytest.raises(ValueError, match="requires eight"):
            feed_u_contrasts(values, [9] * 8, [9] * 8)


@pytest.mark.parametrize("bad", [np.full(64, np.nan), np.zeros(63), np.zeros(64, dtype=complex)])
def test_invalid_observation_is_refused_before_pool_or_observer_effects(bad):
    p = pool()
    before = p.fingerprint()
    with pytest.raises(ValueError, match="finite real reservoir observation"):
        pool_first_spikes(p, bad, np.zeros(3))
    assert p.fingerprint() == before and p.observer is None


def test_invalid_frozen_rank_segment_is_refused():
    code = list(ranks())
    code[0] = code[1]
    with pytest.raises(ValueError, match="frozen 52-rank"):
        arm_latencies("direct", units(), code, 8.0, 14, seed=9900)


def test_a_planted_size_above_256_meets_the_feed_check_not_a_size_gate():
    p = pool()
    p.n = 257  # planted declared-size mismatch: no size gate, the feed-shape check refuses
    before = p.count
    with pytest.raises(ValueError, match="do not name exactly the pool units"):
        arm_latencies(
            "fed",
            units(),
            ranks(),
            8.0,
            14,
            seed=9900,
            pool=p,
            observation=np.zeros(64),
            w_feed=np.zeros((3, 3)),
        )
    assert p.count == before


def test_post_time_check_runs_above_256_before_any_pool_step():
    p = pool()
    p.n = 257  # planted declared size; no size gate precedes the post-time read
    before = p.count
    with pytest.raises(ValueError, match="finite positive grown post time"):
        arm_latencies(
            "fed",
            units(),
            ranks(),
            8.0,
            float("nan"),
            seed=9900,
            pool=p,
            observation=np.zeros(64),
            w_feed=None,
        )
    assert p.count == before


@pytest.mark.parametrize("n", [256, 257])
def test_no_size_gate_on_either_side_of_256(n):
    p = pool()
    p.n = n  # planted declared size: both sides of 256 reach the same feed-shape check
    with pytest.raises(ValueError, match="do not name exactly the pool units"):
        arm_latencies(
            "fed",
            units(),
            ranks(),
            8.0,
            14,
            seed=9900,
            pool=p,
            observation=np.zeros(64),
            w_feed=np.array([[2.0, 1.0, 1.5], [0.0, 3.0, 2.0], [1.0, 0.5, 3.0]]),
        )


# Measurement points of the N > 256 arm, never caps (g3-growth-loop.md DATED ADDENDUM 2026-10-29):
# 257 is the smallest N above the engine's 256; 512 is the first doubling of it.
GROWN_N = (257, 512)


def grown_pool(n, p_parts=1, threads=1, w=None):
    """A hosted pool at size n: R-b3's witness, its draw and encoder rule at n rows, dev seed."""
    config = rp.R_B3_WITNESS["lsm"]
    _, pre, drawn = rp._lsm_draw(
        rp.realization_seed("lsm", rp.DEV_SEED), n, config["k"], config["w"]
    )
    encoder = np.random.default_rng(rp.DEV_SEED * 31 + 7).standard_normal((n, rp.N_IN)) / 8
    return KernelLsmPool(
        pre,
        drawn if w is None else w,
        config,
        w_in=encoder,
        window=8,
        p_parts=p_parts,
        threads=threads,
    )


@pytest.mark.parametrize("n", GROWN_N)
def test_fed_arm_reads_a_hosted_pool_above_256(n):
    us, code, x = units(), ranks(), np.linspace(0.0, 1.0, 64)
    w = np.zeros((n, 3))
    w[0], w[-1] = [2.0, 1.0, 1.5], [1.0, 0.5, 3.0]
    fed, control = grown_pool(n), grown_pool(n)
    got = arm_latencies("fed", us, code, 8.0, 14, seed=9900, pool=fed, observation=x, w_feed=w)
    expected = pool_first_spikes(control, x, feed_drive(us, code, 8.0, 14, w))
    assert len(got) == n and got == expected
    assert fed.fingerprint() == control.fingerprint()


@pytest.mark.parametrize("n", GROWN_N)
def test_the_pools_determinism_red_reruns_at_each_grown_n(n):
    seq = rp.cycled_fixture(16)
    runs = [rp._record(grown_pool(n, p, t), seq, chain=True) for p, t in rp.R_B3_HOSTED]
    assert rp._runs_agree(runs)
    # the red: one incoming weight of the last unit (index n - 1 > 255) changed must disagree
    drawn = grown_pool(n).w.copy()
    drawn[n - 1, 0] = -drawn[n - 1, 0]
    planted = rp._record(grown_pool(n, w=drawn), seq, chain=True)
    assert not rp._runs_agree([runs[0], planted])


def _assert_random_arm_delivers_its_own_drive():
    """C2: the random arm must reach the pool with the SEEDED random projection, not the fed one."""
    import snn_data_lake.growth.feed_readout as consumer
    from snn_data_lake.growth.grown_feed import random_feed

    us, code, x = units(), ranks(), np.zeros(64)
    # Scaled so the constructed pool is NOT saturated: fed and random drives separate in latency.
    w = 0.2 * np.array([[2.0, 1.0, 1.5], [0.0, 3.0, 2.0], [1.0, 0.5, 3.0]])
    args = (us, code, 8.0, 14)
    # Independent derivation through the incumbent implementation, never the consumer's name.
    derived = random_feed(w, us, 9900)
    assert not np.array_equal(derived, w), "degenerate projection: random equals the fed weights"
    assert np.allclose(np.linalg.norm(derived, axis=0), np.linalg.norm(w, axis=0))
    drive_random = feed_drive(us, code, 8.0, 14, derived)
    drive_fed = feed_drive(us, code, 8.0, 14, w)
    assert not np.allclose(drive_random, drive_fed), "the two arms must be driven differently"
    expected = pool_first_spikes(pool(), x, drive_random)
    unrandomized = pool_first_spikes(pool(), x, drive_fed)
    assert expected != unrandomized, "constructed pool must separate the random and fed drives"
    got = consumer.arm_latencies(
        "random-feed", *args, seed=9900, pool=pool(), observation=x, w_feed=w
    )
    fed = consumer.arm_latencies("fed", *args, seed=9900, pool=pool(), observation=x, w_feed=w)
    assert got == expected, "the random arm did not deliver ITS drive to the pool"
    assert got != fed, "the random arm is indistinguishable from the unrandomized feed"
    # A second seed: a consumer that ignores its seed delivers one fixed drive for both.
    expected_b = pool_first_spikes(
        pool(), x, feed_drive(us, code, 8.0, 14, random_feed(w, us, 9901))
    )
    got_b = consumer.arm_latencies(
        "random-feed", *args, seed=9901, pool=pool(), observation=x, w_feed=w
    )
    assert got_b == expected_b, "the random arm ignored its seed (9901)"
    assert got_b != got, "the random arm delivers the same drive for different seeds"


def test_random_feed_arm_delivers_its_seeded_projection_to_the_pool():
    _assert_random_arm_delivers_its_own_drive()


def test_identity_random_feed_mutant_is_caught(monkeypatch):
    import snn_data_lake.growth.feed_readout as consumer

    monkeypatch.setattr(consumer, "random_feed", lambda weights, units, seed: weights)
    with pytest.raises(AssertionError, match="random arm"):
        _assert_random_arm_delivers_its_own_drive()


# ---- G3 grow round 2: the feed hands over at join (RULING items 5-7) ----------------------------

TUNED = {"k": 8, "w": 0.25, "lambda": 0.9, "g": 4, "T_in": 1}
JOIN_VALUES = BudgetValues(first=52, beta=1.5, k_max=3)
# FIXTURE site/centre/density: tests/test_growth_loop.py's admission values (G7's test values)
JOIN_ADMISSION = make_admission(
    site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0), density_per_mm2=2.7e4, seed=9000
)


def _col(ids):
    """One COL batch in unit_id order: E units, then exactly one I unit."""
    return tuple(
        GrownUnit(uid, (0.0,) * 52, 1.0, 12, -1 if i == len(ids) - 1 else 1, (0.0,))
        for i, uid in enumerate(ids)
    )


def _admitted(tmp_path, ids=(40, 41, 42)):
    """A ledger with the batch's birth rows, through GrowthLoop's admission (labels, G7 gate)."""
    ledger = Ledger(tmp_path / "lake.bin", JOIN_VALUES)
    batch = _col(ids)
    labels, _ticket = JOIN_ADMISSION(
        batch, "p", 0, resolve_problem=lambda p: True, known_unit=lambda u: False
    )
    ledger.record_units(batch, labels, "p", 0)
    return ledger, batch


def _born_pool():
    _, pre, w = rp._lsm_draw(rp.realization_seed("lsm", rp.DEV_SEED), rp.N, TUNED["k"], TUNED["w"])
    return KernelLsmPool(pre, w, TUNED, w_in=rp.w_in(rp.DEV_SEED), window=0)


def _feed(rows, batch):
    """A Dale-respecting feed: |N(0,1)| scaled by each column's unit sign, unit_id order."""
    magnitude = np.abs(np.random.default_rng(5).standard_normal((rows, len(batch))))
    return magnitude * np.array([u.sign for u in sorted(batch, key=lambda u: u.unit_id)])


def _row(pool):
    return rp.w_in(rp.DEV_SEED, pool.n + 1)[pool.n]


def _joins(ledger):
    return [r for r in ledger.events() if r["kind"] == "join"]


def test_joined_unit_leaves_external_feed(tmp_path):
    ledger, batch = _admitted(tmp_path)
    pool, w = _born_pool(), _feed(rp.N, batch)
    grown, rest, record = join_fed_unit(pool, w, batch, 41, _row(pool), ledger)
    assert [u.unit_id for u in rest] == [40, 42] and grown.shape == (rp.N + 1, 2)
    assert record["unit_id"] == 41 and record["retired_feed"] == w[:, 1].tolist()
    assert record["sign"] == 1 and pool.signs[rp.N] == 1 and pool.n == rp.N + 1
    rows = _joins(ledger)
    assert len(rows) == 1 and rows[0]["unit_id"] == 41 and rows[0]["n"] == rp.N
    assert rows[0]["retired_feed"] == w[:, 1].tolist()  # retired from the network, kept in the lake
    assert any(r["unit_id"] == 41 for r in ledger.units())  # its birth row stays


def test_old_feed_columns_never_resampled_at_join(tmp_path):
    ledger, batch = _admitted(tmp_path)
    pool, w = _born_pool(), _feed(rp.N, batch)
    grown, rest, _ = join_fed_unit(pool, w, batch, 41, _row(pool), ledger)
    grown, rest, _ = join_fed_unit(pool, grown, rest, 42, _row(pool), ledger)
    assert [u.unit_id for u in rest] == [40] and grown.shape == (rp.N + 2, 1)
    assert np.array_equal(grown[: rp.N, 0], w[:, 0])  # the kept column, bit for bit
    assert not np.any(grown[rp.N :])  # the joined units' rows are zero, never drawn
    assert [r["unit_id"] for r in _joins(ledger)] == [41, 42]
    assert list(pool.signs[rp.N :]) == [1, -1]


def test_a_join_follows_only_from_the_admission(tmp_path):
    ledger, batch = _admitted(tmp_path)
    stranger = GrownUnit(99, (0.0,) * 52, 1.0, 12, 1, (0.0,))
    pool, w = _born_pool(), _feed(rp.N, batch + (stranger,))
    before = pool.params_fingerprint()
    with pytest.raises(ValueError, match="REFUSED"):  # no birth row: never admitted
        join_fed_unit(pool, w, batch + (stranger,), 99, _row(pool), ledger)
    with pytest.raises(ValueError, match="REFUSED"):  # admitted, but not a fed unit here
        join_fed_unit(pool, _feed(rp.N, batch[:2]), batch[:2], 42, _row(pool), ledger)
    with pytest.raises(ValueError, match="REFUSED"):  # a feed that does not name the pool's units
        join_fed_unit(pool, _feed(rp.N - 1, batch), batch, 41, _row(pool), ledger)
    assert not _joins(ledger) and pool.n == rp.N and pool.params_fingerprint() == before


def test_restart_rebuilds_grown_pool_from_join_rows(tmp_path):
    ledger, batch = _admitted(tmp_path)
    live, w = _born_pool(), _feed(rp.N, batch)
    w, rest, _ = join_fed_unit(live, w, batch, 40, _row(live), ledger)
    w, rest, _ = join_fed_unit(live, w, rest, 42, _row(live), ledger)
    restarted, _, _ = rebuild_joined_pool(_born_pool(), _feed(rp.N, batch), batch, ledger)
    assert restarted.n == live.n == rp.N + 2
    assert restarted.params_fingerprint() == live.params_fingerprint()
    assert restarted.frozen() and restarted.boundary == live.boundary
    assert np.array_equal(restarted.signs, live.signs)
    assert np.array_equal(restarted.w_in, live.w_in)
    # the red: a journal holding only unit N's row (no retargeted list) rebuilds another pool
    stripped = _born_pool()
    for row in _joins(ledger):
        stripped.apply_join(dict(row, retargeted=[]))
    assert stripped.params_fingerprint() != live.params_fingerprint()


def test_restart_rebuilds_the_feed_handover_from_join_rows(tmp_path):
    """RULING items 5-6 (enforcer #7299 MAJOR-1): the restart rebuilds the feed from the same join
    rows as the pool, so the restarted feed names the restarted pool's units and a joined unit
    drives only as a pool unit, never also as a feed column."""
    ledger, batch = _admitted(tmp_path)
    live, w = _born_pool(), _feed(rp.N, batch)
    w, rest, _ = join_fed_unit(live, w, batch, 40, _row(live), ledger)
    w, rest, _ = join_fed_unit(live, w, rest, 42, _row(live), ledger)
    born = _feed(rp.N, batch)
    pool, feed, units = rebuild_joined_pool(_born_pool(), born, batch, ledger)
    assert pool.n == rp.N + 2 and feed.shape == w.shape == (rp.N + 2, 1)
    assert [u.unit_id for u in units] == [u.unit_id for u in rest] == [41]  # 40, 42 retired
    assert np.array_equal(feed, w)  # the restarted feed equals the live one, every column
    assert np.array_equal(feed[: rp.N, 0], born[:, 1]) and not np.any(feed[rp.N :])
    # the red: a born feed that is not the one the joins retired columns from is REFUSED
    with pytest.raises(ValueError, match="REFUSED"):
        rebuild_joined_pool(_born_pool(), born * 2, batch, ledger)
