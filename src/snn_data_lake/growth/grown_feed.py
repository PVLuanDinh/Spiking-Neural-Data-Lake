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

from snn_data_lake.growth.grown_code import clip_to_sign, segment_psp

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
    w = _matrix(w_feed, units)
    check_feed(w, units)  # a non-finite or sign-violating source must not become a silent zero arm
    rng = np.random.default_rng(seed)
    out = np.zeros_like(w)
    for j, u in enumerate(_by_id(units)):
        col = np.abs(rng.standard_normal(w.shape[0])) * u.sign
        norm = np.linalg.norm(col)
        if norm > 0:
            out[:, j] = col * (_l2(w[:, j]) / norm)
    return out
