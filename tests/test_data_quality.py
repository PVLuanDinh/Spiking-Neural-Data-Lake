"""
Gate tests for data_quality: schema rejects bad t/channel/non-int, gold-sanity rejects
NaN/inf/out-of-range, duplicates + round-trip + immutability are caught, and valid
Bronze+Gold passes. These gates block promotion to the Gold layer, so a false pass is a
data-integrity hole; the boundaries (t in [0,duration), channel in [0,n)) are the point.

Run: pytest tests/test_data_quality.py
"""

import pytest

from snn_data_lake import data_quality as dq
from snn_data_lake.substrate.data_quality import DataQualityError

N, DUR = 8, 1000


def test_error_is_assertionerror_subclass():
    assert issubclass(DataQualityError, AssertionError)


def test_valid_pipeline_passes():
    rows = [(t, t % N) for t in range(0, DUR, 5)]
    gold = {"icr": 0.3, "rates": [0.1, 0.2, 0.0], "synchrony": 0.5}
    rep = dq.gate(rows, N, DUR, gold=gold)
    assert rep["rows"] == len(rows) and rep["gold_checked"] is True


@pytest.mark.parametrize(
    "rows",
    [
        [(-1, 0)],  # t negative
        [(DUR, 0)],  # t == duration (upper bound is exclusive)
        [(1.5, 0)],  # non-int t
        [(0, N)],  # channel == n_channels (exclusive)
        [(0, -1)],  # channel negative
    ],
)
def test_check_schema_rejects_bad_rows(rows):
    with pytest.raises(DataQualityError):
        dq.check_schema(rows, N, DUR)


def test_check_schema_accepts_valid_edges():
    assert dq.check_schema([(0, 0), (DUR - 1, N - 1)], N, DUR) is True


def test_duplicates_rejected():
    with pytest.raises(DataQualityError):
        dq.check_no_duplicates([(5, 1), (5, 1)])
    assert dq.check_no_duplicates([(5, 1), (5, 2), (6, 1)]) is True


def test_encodable_roundtrip():
    assert dq.check_encodable([(t, t % N) for t in range(50)]) is True


def test_immutable_detects_mutation():
    rows = [(1, 0), (2, 1)]
    h = dq.bronze_hash(rows)
    assert dq.check_immutable(rows, h) is True
    with pytest.raises(DataQualityError):
        dq.check_immutable(rows + [(3, 2)], h)
    assert dq.check_immutable(rows, None) is True  # no prior hash -> skip


@pytest.mark.parametrize(
    "gold",
    [
        {"icr": 0.0, "rates": [], "synchrony": 0.0},  # icr not > 0
        {"icr": 1.5, "rates": [], "synchrony": 0.0},  # icr > 1
        {"icr": float("nan"), "rates": [], "synchrony": 0.0},  # NaN icr
        {"icr": 0.3, "rates": [-1.0], "synchrony": 0.0},  # negative rate
        {"icr": 0.3, "rates": [float("inf")], "synchrony": 0.0},  # inf rate
        {"icr": 0.3, "rates": [], "synchrony": float("inf")},  # inf synchrony
        {"icr": 0.3, "rates": [], "synchrony": -1.0},  # negative synchrony
    ],
)
def test_check_gold_rejects_insane(gold):
    with pytest.raises(DataQualityError):
        dq.check_gold(gold)


def test_check_gold_accepts_boundary_icr_one():
    assert dq.check_gold({"icr": 1.0, "rates": [0.0, 5.0], "synchrony": 0.0}) is True


def test_check_event_stream_parity():
    assert dq.check_event({"t": 1, "channel": 0}, N) == {"t": 1, "channel": 0}
    with pytest.raises(DataQualityError):
        dq.check_event({"t": 1.5, "channel": 0}, N)  # non-int
    with pytest.raises(DataQualityError):
        dq.check_event({"t": -1, "channel": 0}, N)  # negative t
    with pytest.raises(DataQualityError):
        dq.check_event({"t": 1, "channel": N}, N)  # channel out of range
