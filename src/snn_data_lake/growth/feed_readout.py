"""G3 feed-arm latency segments and the registered paired-U superiority rule.

Contract: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.10,
2.12 and ruling 9a. These are inputs to the SAME taught label neurons, not
another classifier. No teaching rule, pool growth target, or campaign is
invented here. The G9 consumer must bind these separate latencies to its
label-neuron port; that consumer is absent at base d4c18ce0.
"""

import numpy as np

from snn_data_lake.growth.grown_code import apply_unit_row, latencies
from snn_data_lake.growth.grown_feed import check_feed, feed_drive, random_feed
from snn_data_lake.instruments.paired_contrast import T_BAR, contrast
from snn_data_lake.reservoir.recurrent_pool import KernelLsmPool


def _build_seed(seed):
    if type(seed) is not int or not (9000 <= seed < 9100 or 9900 <= seed < 10000):
        raise ValueError(f"REFUSED: build/dev fixture seed required, got {seed!r}")


def pool_first_spikes(pool, observation, feed):
    """Collect actual hosted-kernel FIRST spikes in the counted percept steps.

    No rate-to-rank conversion, reranking of the frozen 52-code, or second pool.
    Settling ticks are excluded, as they are from percept_drive's counted read.
    The existing observer still sees every tick and is restored on any exit.
    A read advances the pool's carriers once, exactly like percept_fed.
    """
    if type(pool) is not KernelLsmPool:
        raise ValueError("REFUSED: G1(c)'s hosted KernelLsmPool required")
    x = np.asarray(observation)
    if (
        pool.w_in is None
        or x.dtype.kind not in "fiu"
        or x.shape != (pool.w_in.shape[1],)
        or not np.isfinite(x).all()
    ):
        raise ValueError("REFUSED: finite real reservoir observation of encoder width required")
    observer, spikes = pool.observer, []

    def collect(current, activity):
        spikes.append(np.asarray(activity).copy())
        if observer is not None:
            observer(current, activity)

    pool.observer = collect
    try:
        pool.percept_fed(observation, feed)
    finally:
        pool.observer = observer
    counted = spikes[-pool.t_in :]
    if len(counted) != pool.t_in:
        raise ValueError("REFUSED: incomplete counted reservoir spike window")
    return tuple(
        next((t for t, activity in enumerate(counted, 1) if activity[i] != 0), None)
        for i in range(pool.n)
    )


def arm_latencies(
    arm, units, ranks, tau, t_post_g, *, seed, pool=None, observation=None, w_feed=None
):
    """Direct grown latencies or fed/random-feed reservoir first-spike latencies.

    Units remain in their immortal unit_id order; a fed unit keeps its identity
    and is not relabeled as a reservoir unit. N>256 admission remains gated by
    the owner-ratified reservoir growth target in the frozen contract.
    """
    _build_seed(seed)
    if arm not in ("direct", "fed", "random-feed"):
        raise ValueError(f"REFUSED: unknown feed arm {arm!r}")
    if (
        len(ranks) != 52
        or any(type(t) is not int for t in ranks)
        or sorted(ranks) != list(range(52))
        or not np.isfinite(tau)
        or tau <= 0
    ):
        raise ValueError("REFUSED: frozen 52-rank code and positive finite tau required")
    segment = ()
    for unit in units:
        if len(unit.w_in) != 52:
            raise ValueError(f"REFUSED: unit {unit.unit_id}: frozen input width is not 52")
        segment = apply_unit_row(segment, {"op": "admit", "unit": unit})
    if arm == "direct":
        return tuple(latencies(segment, ranks, tau))
    if type(pool) is not KernelLsmPool or pool.n > 256:
        raise ValueError("REFUSED: hosted pool N<=256 required; N>256 growth target unratified")
    if not np.isfinite(t_post_g) or t_post_g <= 0:
        raise ValueError("REFUSED: finite positive grown post time required")
    check_feed(w_feed, segment)
    if np.asarray(w_feed).shape[0] != pool.n:
        raise ValueError("REFUSED: feed synapses do not name exactly the pool units")
    weights = random_feed(w_feed, segment, seed) if arm == "random-feed" else w_feed
    external = feed_drive(segment, ranks, tau, t_post_g, weights)
    return pool_first_spikes(pool, observation, external)


def feed_u_contrasts(fed, direct, random):
    """The two U contrasts, S=8, t>1.895; superiority degeneracy is NOT MET.

    Each caller-supplied U is an observed matched-budget score; this function
    does not generate solve sizes or license a campaign result from fixtures.
    """
    arms = {"fed": tuple(fed), "direct": tuple(direct), "random-feed": tuple(random)}
    for name, values in arms.items():
        if len(values) != 8 or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"REFUSED: {name} requires eight non-negative integer U scores")
    rows = {}
    for name in ("direct", "random-feed"):
        diffs = [u - f for u, f in zip(arms[name], arms["fed"], strict=True)]
        row = contrast(f"{name} U - fed U", diffs, T_BAR)
        rows[name] = row._replace(met=not row.degenerate and row.t > T_BAR)
    return rows


def main():
    """Constructed rule red/control only; no trained arm or deciding seed."""
    fed = [9] * 8
    positive = feed_u_contrasts(
        fed, [18, 19, 18, 20, 19, 18, 20, 19], [22, 21, 23, 22, 21, 23, 22, 21]
    )
    assert all(row.met for row in positive.values())
    tied = feed_u_contrasts(fed, fed, fed)
    assert not any(row.met for row in tied.values())
    uniform = feed_u_contrasts(fed, [18] * 8, [18] * 8)
    assert not any(row.met for row in uniform.values()), "ruling 9a: uniform superiority is NOT MET"
    print("feed_readout: paired-U null/degenerate reds fire; synthetic variable-win control passes")


if __name__ == "__main__":
    main()
