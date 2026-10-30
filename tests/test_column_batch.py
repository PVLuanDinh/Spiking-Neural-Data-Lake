"""specs/022 T111 (G3) P4a: the growth-unit arms COL / SINGLE / BATCH9 (pre-registration 2.7, 2.10).

Charged here: batch sizes, the one coupling rule (gamma is a required parameter, never defaulted:
pending Q-COL), the role-free batch record (roles are read off the label's sign), the column
assembly's refusals, the worst-residual targeting rule, and the isolating reds (BATCH9 is COL at
gamma 0; COL's coupling changes a latency on a planted fixture). Nothing builds a Brain or a lake.
"""

import pytest

from snn_data_lake.growth import column_batch as cb
from snn_data_lake.growth.grown_code import GrownUnit, latency

TAU = 8.0


def _unit(uid, w, threshold=1.0, sign=1, out=(0.5, 0.5), window=12):
    return GrownUnit(uid, tuple(w), threshold, window, sign, tuple(out))


def test_batch_sizes_count_every_admitted_unit():
    assert cb.batch_size("COL") == cb.batch_size("BATCH9") == cb.K_PRIMARY + 1 == 9
    assert cb.batch_size("COL", 4) == 5 and cb.batch_size("COL", 12) == 13
    assert cb.batch_size("SINGLE") == 1


def test_unknown_arm_or_k_is_refused():
    with pytest.raises(ValueError):
        cb.batch_size("RAND")
    with pytest.raises(ValueError):
        cb.batch_size("COL", 5)


def test_coupling_is_gamma_only_for_col_and_gamma_has_no_default():
    assert cb.coupling("COL", 0.25) == 0.25
    assert cb.coupling("BATCH9", 0.25) == 0.0 and cb.coupling("SINGLE", 0.25) == 0.0
    with pytest.raises(TypeError):
        cb.coupling("COL")
    for bad in (-0.1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            cb.coupling("COL", bad)


def _eleven(n):
    return [_unit(i, [0.1], out=(0.5, -0.5)) for i in range(n)]


def test_as_column_makes_k_excitatory_and_one_inhibitory_and_clips_to_sign():
    col = cb.as_column(_eleven(9), 8)
    assert [u.sign for u in col] == [1] * 8 + [-1]
    assert all(min(u.out) >= 0 for u in col[:8]) and all(max(u.out) <= 0 for u in col[8:])
    assert [u.unit_id for u in col] == list(range(9))


@pytest.mark.parametrize("n", [8, 10])
def test_as_column_refuses_a_wrong_size(n):
    with pytest.raises(ValueError):
        cb.as_column(_eleven(n), 8)


def test_column_source_wraps_a_plain_and_a_searching_source():
    plain = cb.column_source(lambda n: _eleven(n), 8)
    assert [u.sign for u in plain(9)] == [1] * 8 + [-1]
    searching = cb.column_source(lambda n: (_eleven(n), 41), 8)
    units, evals = searching(9)
    assert evals == 41 and len(units) == 9
    with pytest.raises(ValueError):  # the loop's batch must be the column size
        plain(5)


def test_batches_groups_label_rows_by_batch_id_and_keeps_no_role():
    rows = [
        {"unit_id": 3, "batch_id": 1, "sign": -1},
        {"unit_id": 1, "batch_id": 0, "sign": 1},
        {"unit_id": 2, "batch_id": 1, "sign": 1},
        {"unit_id": 0, "batch_id": 0, "sign": 1},
    ]
    got = cb.batches(rows)
    assert got == (cb.ColumnBatch(0, (0, 1)), cb.ColumnBatch(1, (2, 3)))
    assert not hasattr(got[0], "role") and not hasattr(got[0], "sign")


def test_worst_residual_picks_the_largest_magnitude_ties_to_the_lowest_index():
    assert cb.worst_residual([0.1, -0.9, 0.9, 0.0], 2) == (1, 2)
    assert cb.worst_residual([0.5, 0.5, 0.5], 2) == (0, 1)
    with pytest.raises(ValueError):
        cb.worst_residual([0.1], 2)
    with pytest.raises(ValueError):  # a negative n would slice all but |n| entries
        cb.worst_residual([0.1, 0.2, 0.3], -1)


# ---- the isolating reds: a planted fixture where the shared I spikes before an E unit would
RANKS = (1, 3)


def _fixture():
    e = _unit(0, [0.65, 0.65], threshold=1.0, sign=1)
    i = _unit(1, [2.0, 0.0], threshold=1.0, sign=-1, out=(-0.5, -0.5))
    return [e, i]


def test_batch9_is_col_at_gamma_zero_byte_identically():
    units = _fixture()
    alone = [latency(u, RANKS, TAU) for u in units]
    assert cb.latencies("BATCH9", units, RANKS, TAU, gamma=0.5) == alone
    assert cb.latencies("COL", units, RANKS, TAU, gamma=0.0) == alone
    assert alone == [4, 2]


def test_col_coupling_changes_an_e_latency():
    # the liveness red: without it COL == BATCH9 and T1-struct is vacuous
    units = _fixture()
    assert cb.latencies("COL", units, RANKS, TAU, gamma=1.0)[0] != 4


def test_single_arm_reads_each_unit_alone():
    e = _fixture()[0]
    assert cb.latencies("SINGLE", [e], RANKS, TAU, gamma=1.0) == [4]
    with pytest.raises(ValueError):
        cb.latencies("SINGLE", _fixture(), RANKS, TAU, gamma=1.0)
