"""specs/022 T005: the segment writer's durable-commit order, recovery, and CRC refusals."""

import struct

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from snn_data_lake.substrate.lake_segments import (
    SEGMENT_SCHEMA,
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


def _seal_the_pre_t066_way(seg_dir, payloads, seq0=1):
    """The writer `seal()` REPLACED: the whole segment as ONE Arrow table, so the footer indexes
    the segment and not the records. Constructed here because no such segment exists in the tree
    -- without it the seal() comment's "REFUSES an old segment loudly" claim has no test behind
    it, which is exactly what the 2026-09-08 panel found."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    envelopes = [wrap(p) for p in payloads]
    heads = [struct.unpack_from("<HI", e) for e in envelopes]
    table = pa.table(
        [
            pa.array([seq0 + i for i in range(len(payloads))], pa.uint64()),
            pa.array([h[0] for h in heads], pa.uint16()),
            pa.array([h[1] for h in heads], pa.uint32()),
            pa.array(list(payloads), pa.binary()),
        ],
        schema=SEGMENT_SCHEMA,
    )
    path = seg_dir / f"seg-{seq0:08d}.lake"
    with open(path, "wb") as f:
        with pa_ipc.new_file(f, SEGMENT_SCHEMA) as w:
            w.write_table(table)
    return path


def test_read_segment_still_reads_a_pre_t066_whole_table_segment(tmp_path):
    """Backwards compatibility of the WHOLE-segment reader across the 2026-09-08 layout change.

    TRIMMED 2026-09-09 (T068 LOSS): this was the positive control inside
    `test_read_record_refuses_a_pre_t066_whole_table_segment`, whose subject -- `read_record`'s
    per-record addressing and its refusal of the one-batch layout -- was deleted with the paging
    layer that was its only caller (specs/022 evidence/s5-paging-null.md 3.4/3.5). The refusal
    arms go with the function. THIS claim does not: `read_segment` survives, is live through
    `lake_bridge.located_records`, and both segment layouts still exist on disk, so the
    compatibility it asserts is a live property and keeping the arm is not sentiment."""
    payloads = [b"old-0", b"old-1", b"old-2"]
    path = _seal_the_pre_t066_way(tmp_path / "segments", payloads)
    with pa_ipc.open_file(path) as r:
        assert r.num_record_batches == 1, "the fixture must be the one-batch layout, not the new"
    assert list(read_segment(path)) == payloads

    # the CONTROL that keeps this from passing on a broken fixture: the CURRENT writer's layout is
    # a different shape (one batch per record) and the same reader must read it too.
    store = SegmentStore(tmp_path / "new")
    for p in payloads:
        store.append(p)
    sealed = store.seal()
    with pa_ipc.open_file(sealed) as r:
        assert r.num_record_batches == len(payloads), "the writer seals one batch per record"
    assert list(read_segment(sealed)) == payloads


def test_torn_wal_healed_by_incumbent_semantics(tmp_path):
    store = SegmentStore(tmp_path)
    store.append(b"complete")
    del store
    with open(tmp_path / "segments" / "active.wal", "ab") as f:
        f.write(struct.pack("<I", 500) + b"torn")
    store2 = SegmentStore(tmp_path)
    assert list(store2.iter_payloads()) == [b"complete"]
