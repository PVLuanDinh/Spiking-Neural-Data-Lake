"""specs/022 T111 (G3) P4a: the growth-unit arms of T1 -- COL, SINGLE, BATCH9 (pre-registration 2.7, 2.10).

COL is a batch of k excitatory units plus ONE shared inhibitory unit with the causal E/I coupling of
`grown_code.column_latencies`; BATCH9 is the same k + 1 units at coupling 0; SINGLE grows one unit at a time.
They differ only in unit size and coupling, so T1 isolates the unit (size) and the structure (coupling).

- `gamma` is a REQUIRED parameter, never defaulted: its value and sensitivity pair are PENDING Q-COL (G7's
  amendment (i)); this module decides nothing about it.
- A batch record is `batch_id` + member unit ids only. The E/I role is read off the label's Dale sign
  (`placement.label_of`): one store, so a role/sign disagreement cannot exist (2.7, D8).
- Labels, sites and kernels are not here: `growth_loop.make_admission` calls G7's `place_labeled(roles=...)`
  for every batch, and attachment stays kernel-free until G4's T3 (2.7).
- `worst_residual` is the ONE targeting rule both arms share: the label neurons with the largest residual.

distinct-from: growth/grown_code.py::column_latencies -- that is the coupled evaluation itself; this module
only names the arms, assembles a source's units into a column and groups label rows into batches.
distinct-from: growth/placement.py -- labels and positions are G7's; nothing here places a unit.
"""

import math
from dataclasses import dataclass, replace

from snn_data_lake.growth.grown_code import GrownUnit, clip_to_sign, column_latencies, latency

ARMS = ("COL", "SINGLE", "BATCH9")
K_GRID = (4, 8, 12)  # excitatory units per batch, shared with G7
K_PRIMARY = 8


@dataclass(frozen=True)
class ColumnBatch:
    batch_id: int
    members: tuple  # unit ids, ascending


def _require(arm, k):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if k not in K_GRID:
        raise ValueError(f"k must be one of {K_GRID}, got {k!r}")


def batch_size(arm, k=K_PRIMARY):
    """Units admitted per batch; U counts every admitted unit, E and the shared I alike."""
    _require(arm, k)
    return 1 if arm == "SINGLE" else k + 1


def coupling(arm, gamma):
    """The E/I coupling the arm evaluates with: gamma for COL, 0 for BATCH9 and SINGLE."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(gamma)
        or gamma < 0
    ):
        raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}")
    return float(gamma) if arm == "COL" else 0.0


def as_column(units, k):
    """k excitatory units then one inhibitory unit: signs set by position, output synapses clipped to the sign."""
    units = list(units)
    if len(units) != k + 1:
        raise ValueError(f"a column of k={k} holds {k + 1} units, got {len(units)}")
    signs = [1] * k + [-1]
    return [
        replace(u, sign=s, out=tuple(clip_to_sign(u.out, s, u.unit_id)))
        for u, s in zip(units, signs)
    ]


def column_source(source, k):
    """Wrap a GrowthLoop source (units, or (units, evaluations)) so every batch is one column of k E + 1 I."""

    def column(n):
        out = source(n)
        units, evals = out if isinstance(out, tuple) else (out, None)
        col = as_column(units, k)
        return col if evals is None else (col, evals)

    return column


def batches(label_rows):
    """Label rows -> ColumnBatch records, grouped by batch_id, ids and members ascending; no role is kept."""
    grouped = {}
    for row in label_rows:
        grouped.setdefault(row["batch_id"], []).append(row["unit_id"])
    return tuple(ColumnBatch(b, tuple(sorted(m))) for b, m in sorted(grouped.items()))


def worst_residual(residuals, n):
    """Indices of the n label neurons with the largest |residual|, ties to the lowest index."""
    if n > len(residuals):
        raise ValueError(f"asked for {n} label neurons of {len(residuals)}")
    order = sorted(range(len(residuals)), key=lambda i: (-abs(residuals[i]), i))
    return tuple(order[:n])


def latencies(arm, units, ranks, tau, gamma):
    """Each unit's first-spike step in unit_id order (None = no spike) under the arm's coupling."""
    g = coupling(arm, gamma)
    if arm == "SINGLE":
        if len(units) != 1:
            raise ValueError(f"SINGLE reads one unit, got {len(units)}")
        return [latency(units[0], ranks, tau)]
    got = column_latencies(units, ranks, tau, g)
    return [got[u.unit_id] for u in sorted(units, key=lambda u: u.unit_id)]


def main():
    """Self-check: BATCH9 is COL at gamma 0, and a planted early I spike makes COL differ."""
    e = GrownUnit(0, (0.65, 0.65), 1.0, 12, 1, (0.5,))
    i = GrownUnit(1, (2.0, 0.0), 1.0, 12, -1, (-0.5,))
    ranks = (1, 3)
    b9 = latencies("BATCH9", [e, i], ranks, 8.0, 1.0)
    col = latencies("COL", [e, i], ranks, 8.0, 1.0)
    assert b9 == [4, 2] and col[0] != 4, (b9, col)
    assert as_column([e] * 8 + [i], 8)[-1].sign == -1
    print(
        "column_batch: BATCH9 == COL at gamma 0, COL coupling changes an E latency, column assembled"
    )


if __name__ == "__main__":
    main()
