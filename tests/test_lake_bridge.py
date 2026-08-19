"""specs/022 T005b: the format bridge's round-trip red, its control, and its refusals."""

import struct

import pytest

from snn_data_lake.substrate import lake_bridge
from snn_data_lake.substrate.binlog import BinLog


def _mklake(root, n):
    log = BinLog(str(root / "brain_lake.bin"), dedup=False)
    for i in range(n):
        log.append(f"rec-{i}".encode())


def test_round_trip_equality_with_mutation_control(tmp_path):
    _mklake(tmp_path, 9)
    ok, msg = lake_bridge.round_trip_ok(tmp_path)
    assert ok, msg
    assert "mutation control moved it" in msg, "the equality is only evidence WITH its control"


def test_bridge_yields_binlog_records_in_order(tmp_path):
    _mklake(tmp_path, 4)
    assert list(lake_bridge.iter_records(tmp_path)) == [b"rec-0", b"rec-1", b"rec-2", b"rec-3"]


def test_empty_home_round_trips_trivially(tmp_path):
    ok, msg = lake_bridge.round_trip_ok(tmp_path)
    assert ok and "empty home" in msg


def test_torn_tail_does_not_diverge_the_bridge(tmp_path):
    # The bridge inherits BinLog's torn-tail semantics rather than reimplementing them:
    # after a torn frame, bridge and direct replay must still agree (both see the heal).
    _mklake(tmp_path, 5)
    with open(tmp_path / "brain_lake.bin", "ab") as f:
        f.write(struct.pack("<I", 999) + b"torn")
    ok, msg = lake_bridge.round_trip_ok(tmp_path)
    assert ok, msg
    assert len(list(lake_bridge.iter_records(tmp_path))) == 5


def test_premature_segment_refused_loudly(tmp_path):
    # A segment existing before its writer (T005) is a defect, not data.
    _mklake(tmp_path, 2)
    seg = tmp_path / lake_bridge.SEGMENT_DIR
    seg.mkdir()
    (seg / "seg-00000001.lake").write_bytes(b"not-a-real-segment")
    with pytest.raises(NotImplementedError, match="before its writer"):
        list(lake_bridge.iter_records(tmp_path))


def test_digest_is_boundary_sensitive():
    a = lake_bridge.stream_digest(iter([b"ab", b"c"]))
    b = lake_bridge.stream_digest(iter([b"a", b"bc"]))
    assert a != b, "length-prefixing must distinguish record boundaries"
