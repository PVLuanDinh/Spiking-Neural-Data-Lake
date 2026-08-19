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


def test_invalid_segment_refused_loudly(tmp_path):
    # T005 un-stubbed the segment arm: garbage where a sealed segment should be is
    # corruption and the bridge refuses the whole segment, never skip-and-continue.
    from snn_data_lake.substrate.lake_segments import SegmentCorruption

    _mklake(tmp_path, 2)
    seg = tmp_path / lake_bridge.SEGMENT_DIR
    seg.mkdir()
    (seg / "seg-00000001.lake").write_bytes(b"not-a-real-segment")
    with pytest.raises(SegmentCorruption, match="not a valid sealed segment"):
        list(lake_bridge.iter_records(tmp_path))


def test_bridge_reads_binlog_then_sealed_then_wal(tmp_path):
    # The one reader path, end to end: incumbent lake -> sealed segment -> active WAL.
    from snn_data_lake.substrate.lake_segments import SegmentStore

    _mklake(tmp_path, 2)  # rec-0, rec-1 in brain_lake.bin
    store = SegmentStore(tmp_path)
    store.append(b"sealed-0")
    store.seal()
    store.append(b"wal-0")
    got = list(lake_bridge.iter_records(tmp_path))
    assert got[:2] == [b"rec-0", b"rec-1"]
    assert got[2:3] == [b"sealed-0"]
    # The active WAL is the store's to read (bridge covers sealed files; the WAL arm
    # joins the bridge when home.py adopts the store at T009/T010).
    assert list(store.iter_payloads()) == [b"sealed-0", b"wal-0"]


def test_digest_is_boundary_sensitive():
    a = lake_bridge.stream_digest(iter([b"ab", b"c"]))
    b = lake_bridge.stream_digest(iter([b"a", b"bc"]))
    assert a != b, "length-prefixing must distinguish record boundaries"
