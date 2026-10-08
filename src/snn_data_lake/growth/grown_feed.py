"""specs/022 T111 (G3) P5: the feed -- admitted grown units project into the G1 reservoir.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.10 (Feed, connection 2) and
2.9 C15 / C17. Every admitted unit keeps its unit_id and projects into the pool through declared input
synapses: one column of `w_feed` (pool units x grown units) per unit, columns in unit_id order (the
order `grown_code.segment_psp` returns), so a synapse is named by (unit_id, pool unit) and resolves to
the unit's label record through the journal -- no second label store. The drive handed to
`recurrent_pool.percept_fed` is `w_feed @ segment_psp(...)`.

- `feed_drive`   -- the pool's external drive from the segment (a silenced or non-spiking unit gives 0);
- `clip_feed`    -- C17's deposit half on the reservoir segment: E columns >= 0, I columns <= 0 (D9);
- `check_feed`   -- C17's restore half: a sign-violating or non-finite synapse is REFUSED, named;
- `random_feed`  -- the norm-matched random-feed arm: per-column L2 norm and Dale sign kept, direction
                    redrawn from a seeded stream.
"""

import numpy as np

from snn_data_lake.growth.grown_code import GrownUnit, clip_to_sign, segment_psp
from snn_data_lake.growth.growth_loop import SEED_BLOCK

# distinct-from: growth/grown_code.py -- the unit contract and the label-neuron segment; this module is the
#   second consumer of the same segment (the reservoir input), reusing its clip and PSP
# distinct-from: faculties/developmental_envelope.py -- an EMA of a competence signal into a maturity scalar
# distinct-from: gates/perf_baseline.py -- the marginal bytes of persisted state per grown unit
# distinct-from: growth/placement.py -- position, layer and sign of newly grown cells
# distinct-from: instruments/episode_driver.py -- one sparse stimulus pattern drawn from its stream
# distinct-from: instruments/prior_drift_harness.py -- a planted wiring control over a connectivity table
# (all read 2026-10-07: phrase overlap on 'drive' / 'units' only; none projects grown units into a pool)


def _by_id(units):
    return sorted(units, key=lambda u: u.unit_id)


def _matrix(w_feed, units):
    w = np.asarray(w_feed)
    if (
        w.dtype.kind not in "fiu"
    ):  # never cast a complex/object matrix (the imaginary part would vanish)
        raise ValueError(f"REFUSED: w_feed must be real numbers, got dtype {w.dtype}")
    w = w.astype(np.float64)
    if w.ndim != 2 or w.shape[1] != len(units):
        raise ValueError(
            f"REFUSED: w_feed {w.shape} needs one column per grown unit ({len(units)} units)"
        )
    return w


def feed_drive(units, ranks, tau, t_post_g, w_feed):
    """The reservoir's external drive from the grown segment: w_feed @ segment_psp, unit_id order."""
    w = _matrix(w_feed, units)
    return w @ np.asarray(segment_psp(units, ranks, tau, t_post_g), dtype=np.float64)


def clip_feed(w_feed, units):
    """Dale's sign at a deposit onto the reservoir segment; returns a new matrix, never mutates."""
    w = _matrix(w_feed, units)
    out = w.copy()
    for j, u in enumerate(_by_id(units)):
        out[:, j] = clip_to_sign(w[:, j].tolist(), u.sign, u.unit_id)
    return out


def check_feed(w_feed, units):
    """Restore: every synapse finite and on its unit's Dale side; the refusal names unit and synapse."""
    w = _matrix(w_feed, units)
    for j, u in enumerate(_by_id(units)):
        for i, x in enumerate(w[:, j]):
            if not np.isfinite(x) or x * u.sign < 0:
                raise ValueError(
                    f"REFUSED: unit {u.unit_id} (sign {u.sign:+d}): synapse to pool unit {i} is {x!r}"
                )


def _l2(x):
    """L2 norm without squaring the raw magnitudes (1e200 overflows, 1e-200 underflows to 0)."""
    m = np.max(np.abs(x))
    return float(m * np.linalg.norm(x / m)) if m > 0 else 0.0


def random_feed(w_feed, units, seed):
    """The norm-matched random arm: each column keeps its L2 norm and its unit's Dale sign."""
    lo, hi = SEED_BLOCK
    if isinstance(seed, bool) or not isinstance(seed, int) or not lo <= seed < hi:
        raise ValueError(f"REFUSED: seed {seed!r} must be an int in G3's block {lo}-{hi - 1}")
    w = _matrix(w_feed, units)
    check_feed(w, units)  # a non-finite or sign-violating source must not become a silent zero arm
    rng = np.random.default_rng(seed)
    out = np.zeros_like(w)
    for j, u in enumerate(_by_id(units)):
        col = np.abs(rng.standard_normal(w.shape[0])) * u.sign
        norm = np.linalg.norm(col)
        if norm > 0:
            target = _l2(w[:, j])
            if not np.isfinite(target):
                raise ValueError(
                    f"REFUSED: unit {u.unit_id}: the source column's L2 norm is not representable"
                )
            out[:, j] = (col / norm) * target  # unit direction first: nothing above the target norm
    if not np.isfinite(out).all():
        raise ValueError("REFUSED: the random arm is not finite")
    return out


def main():
    units = (GrownUnit(0, (0.0,), 1.0, 4, +1, (0.0,)), GrownUnit(1, (0.0,), 1.0, 4, -1, (0.0,)))
    w = np.array([[1.0, -2.0], [0.5, -1.0]])
    check_feed(w, units)  # sign-consistent: accepted

    def refused(fn, *args):
        try:
            fn(*args)
        except ValueError as exc:
            return "REFUSED" in str(exc)
        return False

    flip = w.copy()
    flip[0, 1] = 3.0  # an I unit's synapse flipped positive
    assert refused(check_feed, flip, units), "red: a sign-violating synapse must be refused"
    assert clip_feed(flip, units)[0, 1] == 0.0 and flip[0, 1] == 3.0, (
        "red: clip flips, never mutates"
    )
    assert refused(check_feed, w.astype(complex), units), "red: a complex matrix must be refused"
    r = random_feed(w, units, 9900)
    assert np.allclose(np.linalg.norm(r, axis=0), np.linalg.norm(w, axis=0)), "red: norm-matched"
    assert r.tolist() == random_feed(w, units, 9900).tolist() and r.tolist() != w.tolist()
    assert np.isfinite(random_feed(w * 1e200, units, 9900)).all(), "red: finite at 1e200"
    assert refused(random_feed, w, units, None), "red: an unseeded stream must be refused"
    print("grown_feed: C17 sign/dtype reds fire; C15 random arm norm-matched, seeded, scale-safe")


if __name__ == "__main__":
    main()
