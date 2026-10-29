"""G3 feed-arm tests (red-first, 2026-10-29, not run; see the row's ARMS.md).

Arm A: the fed / direct / norm-matched random-feed readout contrast (g3-growth-loop.md 2.10).
Arm B: the N > 256 fed-unit-becomes-reservoir arm (same section, POLICIES P4d).
Constructed 9900-block fixtures, no brain, no lake, no trained claim.
"""

import numpy as np
import pytest

from snn_data_lake.growth import feed_readout
from snn_data_lake.growth.feed_readout import arm_latencies, feed_u_contrasts
from snn_data_lake.growth.grown_code import GrownUnit, latencies
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features
from snn_data_lake.reservoir import recurrent_pool
from snn_data_lake.reservoir.recurrent_pool import KernelLsmPool

T_IN = 3
W_FEED = np.array([[2.0, 1.0, 1.5], [0.0, 3.0, 2.0], [1.0, 0.5, 3.0]])


def pool():
    return KernelLsmPool(
        [[1, 2], [0, 0], [1, 0]],
        [[0.5, 0.25], [1.0, -0.5], [0.5, 0.5]],
        {"lambda": 0.5, "g": 2, "T_in": T_IN},
        "carried",
        window=0,
        s_settle=0,
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
    learner = SpikingInteractionLearner(["A", "B"], 52, seed=9900, t_post=6)
    return tuple(learner.encode(text_features("cat dog sun sky", 52)))


# ---- Arm A: fed / direct / random-feed readout contrast ----


def test_feed_contrast_is_one_sided_in_fed_favour():
    """A fed arm needing MORE units than direct and random feed is NOT MET, not degenerate."""
    fed = [18] * 8
    better = [9, 10, 9, 10, 9, 10, 9, 10]
    rows = feed_u_contrasts(fed, better, better)
    assert not any(row.met for row in rows.values())
    assert not any(row.degenerate for row in rows.values())


def test_three_arms_give_one_latency_per_unit_each_a_first_spike_or_none():
    us, code, x = units(), ranks(), np.zeros(64)
    direct = arm_latencies("direct", us, code, 8.0, 14, seed=9900)
    assert direct == tuple(latencies(us, code, 8.0)) and len(direct) == len(us)
    for arm in ("fed", "random-feed"):
        got = arm_latencies(
            arm, us, code, 8.0, 14, seed=9900, pool=pool(), observation=x, w_feed=W_FEED
        )
        assert len(got) == 3
        assert all(t is None or t in range(1, T_IN + 1) for t in got)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="G3 arm unbuilt (red-first, 2026-10-29)",
)
def test_trained_arm_runner_reads_the_taught_label_neurons_for_all_three_arms():
    """Absent: a runner feeding each arm's latency segment to the SAME taught label neurons."""
    scores = feed_readout.run_feed_arms(units(), ranks(), seed=9900, pool=pool(), w_feed=W_FEED)
    assert set(scores) == {"fed", "direct", "random-feed"}


# ---- Arm B: N > 256 fed-unit-becomes-reservoir ----


def test_n_above_256_is_refused_before_the_feed_or_post_time_is_read():
    p = pool()
    p.n = 257  # planted declared-size mismatch only; no larger pool is built
    before = p.count
    with pytest.raises(ValueError, match="N>256 growth target unratified"):
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


def test_the_size_gate_is_strictly_above_256():
    p = pool()
    p.n = 256  # at the bound: the size gate passes, the feed-shape check refuses next
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
            w_feed=W_FEED,
        )


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="G3 arm unbuilt (red-first, 2026-10-29)",
)
def test_a_fed_unit_can_become_a_reservoir_unit_above_256():
    """Absent: admit a fed unit as a reservoir unit; the determinism red re-runs at the new N."""
    grown = recurrent_pool.admit_fed_unit_as_reservoir(pool(), units()[0], W_FEED)
    assert grown.n == 4
