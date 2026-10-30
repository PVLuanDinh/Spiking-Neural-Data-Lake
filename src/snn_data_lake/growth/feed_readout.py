"""G3 feed-arm latency segments and the registered paired-U superiority rule.

Contract: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.10,
2.12 and ruling 9a. These are inputs to the SAME taught label neurons, not
another classifier. No teaching rule, pool growth target, or campaign is
invented here. The G9 consumer must bind these separate latencies to its
label-neuron port; that consumer is absent at base d4c18ce0.

The join seam (G3 grow round 2, lanes/engine-1024/g3-grow-research/RULING.md items 5-7): an admitted
fed unit becomes pool unit N through `join_fed_unit` (its feed column retired, one join row per join
on the ONE grower's Ledger), and `rebuild_joined_pool` replays those rows on a restart, the pool and
its feed handover both (P4b).
"""

import numpy as np

from snn_data_lake.growth.growable import grow_rows
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
    and is not relabeled as a reservoir unit. The pool's N has no ceiling (owner
    growth target, g3-growth-loop.md DATED ADDENDUM 2026-10-29); P4d's budget
    paces growth.
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
    if type(pool) is not KernelLsmPool:
        raise ValueError("REFUSED: G1(c)'s hosted KernelLsmPool required")
    if not np.isfinite(t_post_g) or t_post_g <= 0:
        raise ValueError("REFUSED: finite positive grown post time required")
    check_feed(w_feed, segment)
    if np.asarray(w_feed).shape[0] != pool.n:
        raise ValueError("REFUSED: feed synapses do not name exactly the pool units")
    weights = random_feed(w_feed, segment, seed) if arm == "random-feed" else w_feed
    external = feed_drive(segment, ranks, tau, t_post_g, weights)
    return pool_first_spikes(pool, observation, external)


# distinct-from: growth/grown_feed.py::random_feed -- redraws every column's direction for the random
#   arm; this retires one column from the live feed when its unit joins the pool
def _feed_column(pool, w_feed, units, unit_id):
    """(w_feed as float64, the fed units in unit_id order, the column of `unit_id`). REFUSED: a unit
    that is not among the fed `units`, or a feed that does not name the pool's units."""
    fed = sorted(units, key=lambda u: u.unit_id)
    column = [i for i, u in enumerate(fed) if u.unit_id == unit_id]
    if not column:
        raise ValueError(f"REFUSED: unit {unit_id} is not one of the fed units")
    check_feed(w_feed, fed)
    w = np.asarray(w_feed, dtype=np.float64)
    if w.shape[0] != pool.n:
        raise ValueError("REFUSED: feed synapses do not name exactly the pool units")
    return w, fed, column[0]


def _retire_column(pool, w, fed, j):
    """The handover after unit fed[j] became pool unit N: its column leaves the live feed, every
    other column keeps its entries bit for bit and gains a zero row for N (growable.grow_rows)."""
    kept = np.delete(w, j, axis=1)
    grown = np.array(grow_rows(kept.tolist(), pool.n, ncols=kept.shape[1]), dtype=np.float64)
    return grown, tuple(fed[:j] + fed[j + 1 :])


def join_fed_unit(pool, w_feed, units, unit_id, encoder_row, ledger):
    """An admitted fed unit becomes pool unit N (RULING items 5-7). The join follows only from
    GrowthLoop's admission: a unit with no birth row on `ledger` (Ledger.known_unit) is REFUSED, as
    is a unit that is not among the fed `units` or a feed that does not name the pool's units. The
    join row is drawn from the declared join stream with join_seed = unit_id (minted once, never
    reused, so no two joins share a stream), checked by a dry apply, journaled write-ahead as ONE
    "join" row through Ledger.record_event (with the retired feed column and the params fingerprint
    the dry apply staged, the re-boundary a restart checks against), then applied. The unit's
    feed column leaves the live network; every other column keeps its entries bit for bit and gains
    a zero row for N (growable.grow_rows). Returns (w_feed, the fed units left, the applied row)."""
    if type(pool) is not KernelLsmPool:
        raise ValueError("REFUSED: G1(c)'s hosted KernelLsmPool required")
    if not ledger.known_unit(unit_id):
        raise ValueError(f"REFUSED: unit {unit_id!r} has no birth row; a join follows admission")
    w, fed, j = _feed_column(pool, w_feed, units, unit_id)
    record = pool.draw_join(fed[j].sign, unit_id, encoder_row)
    params = pool.apply_join(record, dry=True)
    record = dict(record, unit_id=unit_id, retired_feed=w[:, j].tolist(), params=params)
    ledger.record_event({"kind": "join", **record})
    applied = pool.apply_join(record)
    grown, rest = _retire_column(pool, w, fed, j)
    return grown, rest, applied


def rebuild_joined_pool(pool, w_feed, units, ledger):
    """P4b: `pool` and its born feed (`w_feed` over the born fed `units`) rebuilt from every
    journaled "join" row in lake order, the pool through KernelLsmPool.apply_join (the one apply
    path; a row for another N is REFUSED there) and the feed by the same handover the live join
    made (RULING items 5-6, enforcer #7299 MAJOR-1). REFUSED, before that row's apply: a row with no
    journaled params (unverifiable), a joined unit that is not a fed unit, a column that is not the
    one the row retired, or a sign that is not the unit's; apply_join refuses a rebuild whose staged
    params differ from the journaled ones (#7299 MINOR-1). That check covers LsmPool.params()
    ({pre, w}) and the compiled table only, NOT w_in or the unit signs: a restart over a born encoder
    drawn from another seed passes it (the existing boundary definition; enforcer #7333 NIT).
    Returns (pool, w_feed, the units left)."""
    for row in ledger.events():
        if row.get("kind") != "join":
            continue
        if row.get("params") is None:
            raise ValueError(f"REFUSED: join row for unit {row.get('unit_id')} has no params")
        w_feed, fed, j = _feed_column(pool, w_feed, units, row["unit_id"])
        if w_feed[:, j].tolist() != row["retired_feed"] or fed[j].sign != row["sign"]:
            raise ValueError(f"REFUSED: unit {row['unit_id']}'s born feed is not the one it left")
        pool.apply_join(row)
        w_feed, units = _retire_column(pool, w_feed, fed, j)
    return pool, np.asarray(w_feed, dtype=np.float64), tuple(units)


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
