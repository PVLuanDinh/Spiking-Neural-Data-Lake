"""G3 feed feature plumbing; constructed 9900-block fixtures, no trained claim."""

import numpy as np
import pytest

from snn_data_lake.growth.feed_readout import arm_latencies, feed_u_contrasts, pool_first_spikes
from snn_data_lake.growth.grown_code import GrownUnit, latencies
from snn_data_lake.growth.grown_feed import feed_drive
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


def test_unratified_reservoir_growth_is_refused_before_any_pool_step():
    p = pool()
    p.n = 257  # planted declared-size mismatch only; no larger pool is built or admitted
    before = p.count
    with pytest.raises(ValueError, match="N>256 growth target unratified"):
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
