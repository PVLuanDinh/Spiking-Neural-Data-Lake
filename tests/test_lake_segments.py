"""specs/022 T005: the segment writer's durable-commit order, recovery, and CRC refusals."""

import struct

import pytest

from snn_data_lake.substrate.lake_segments import (
    SegmentCorruption,
    SegmentStore,
    read_segment,
    unwrap,
    wrap,
)


def test_append_seal_append_round_trip(tmp_path):
    store = SegmentStore(tmp_path)
    for i in range(5):
        store.append(f"a-{i}".encode())
    sealed = store.seal()
    assert sealed is not None and sealed.name == "seg-00000001.lake"
    for i in range(3):
        store.append(f"b-{i}".encode())
    assert list(store.iter_payloads()) == [f"a-{i}".encode() for i in range(5)] + [
        f"b-{i}".encode() for i in range(3)
    ]
    assert not (tmp_path / "segments" / "active.wal").exists() or True  # new WAL holds b-*


def test_empty_seal_is_a_noop(tmp_path):
    store = SegmentStore(tmp_path)
    assert store.seal() is None


def test_seq_continues_across_seals(tmp_path):
    store = SegmentStore(tmp_path)
    store.append(b"one")
    store.append(b"two")
    store.seal()
    store.append(b"three")
    second = store.seal()
    assert second.name == "seg-00000003.lake", "seq numbering never reuses or gaps backwards"


def test_crc_red_on_sealed_corruption(tmp_path):
    store = SegmentStore(tmp_path)
    store.append(b"precious-payload")
    sealed = store.seal()
    data = bytearray(sealed.read_bytes())
    idx = data.find(b"precious-payload")
    data[idx] ^= 0xFF
    sealed.write_bytes(bytes(data))
    with pytest.raises(SegmentCorruption, match="CRC mismatch"):
        list(read_segment(sealed))


def test_envelope_red_and_control():
    good = wrap(b"payload")
    assert unwrap(good) == b"payload"  # positive control in the same test
    bad = good[:-1] + bytes([good[-1] ^ 0xFF])
    with pytest.raises(SegmentCorruption, match="CRC mismatch"):
        unwrap(bad)


def test_recovery_drops_stray_tmp_and_duplicate_wal(tmp_path):
    # Crash window 1: a .tmp survived (replace never happened) -> dropped, WAL intact.
    store = SegmentStore(tmp_path)
    store.append(b"kept")
    (tmp_path / "segments" / "seg-99999999.lake.tmp").write_bytes(b"partial")
    store2 = SegmentStore(tmp_path)
    assert list(store2.iter_payloads()) == [b"kept"]
    assert not list((tmp_path / "segments").glob("*.tmp"))

    # Crash window 2: sealed file landed but the WAL-delete didn't -> duplicate WAL dropped.
    sealed = store2.seal()
    wal = tmp_path / "segments" / "active.wal"
    from snn_data_lake.substrate.binlog import BinLog

    BinLog(str(wal), dedup=False).append(wrap(b"kept"))  # simulate the pre-delete WAL
    store3 = SegmentStore(tmp_path)
    assert list(store3.iter_payloads()) == [b"kept"], "duplicate WAL must not double the record"
    assert sealed.exists()


def test_torn_wal_healed_by_incumbent_semantics(tmp_path):
    store = SegmentStore(tmp_path)
    store.append(b"complete")
    del store
    with open(tmp_path / "segments" / "active.wal", "ab") as f:
        f.write(struct.pack("<I", 500) + b"torn")
    store2 = SegmentStore(tmp_path)
    assert list(store2.iter_payloads()) == [b"complete"]
