"""Index rebuild gate (specs/022 T008) -- the index is DERIVED, and this proves it.

Contract (contracts/lake-records.md SS Index): snapshot the logical dump -> DELETE the index
entirely (db + WAL sidecars) -> rebuild from a forward scan of the lake (lake_bridge, the one
reader path) -> the rebuilt LOGICAL dump must equal the snapshot. Logical, not byte: page
layout is engine noise, rows are semantics.

Keying at this stage is mechanical and total: tag = first payload byte (the incumbent lake's
own dispatch convention), key = global ordinal, container = "lake". Precise per-container
addressing arrives when home.py adopts the segment store (T010); THIS gate pins the property
that must survive that adoption -- same lake in, same rows out, from nothing but the lake.

Reds, both proven by firing in --self-check, controls in the same run:
  * PLANTED DIVERGENCE: a row inserted behind the gate's back must fail the equality.
  * LAKE-TRACKING control: appending one record to the lake must CHANGE the rebuilt digest
    (an equality that survives a changed lake would be vacuous).

Bare invocation (run_selfchecks) == --self-check on a constructed temp lake; never the real
home (writer-vs-probe hazard).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from snn_data_lake.substrate.binlog import BinLog
from snn_data_lake.substrate.home import _T_BLOCKDELTA, _T_UNITBLOCK
from snn_data_lake.substrate.lake_bridge import located_records
from snn_data_lake.substrate.lake_engine import BlockDeltaRecord, UnitBlockRecord
from snn_data_lake.substrate.lake_index import LakeIndex

INDEX_NAME = "lake_index.db"
_TAG_BLOCK = _T_UNITBLOCK[0]
_TAG_DELTA = _T_BLOCKDELTA[0]


def build_index(root: str | os.PathLike, index_path: str | os.PathLike) -> int:
    """Rebuild the pointer index from nothing but the lake. Returns records scanned.

    TWO KEYINGS, and the split is the point (specs/022 T067, closing the opening T066 left owed).

      * THE BLOCK CLASSES ARE TAG-AWARE. A 0x0C record names its own `block_id`, so its key is
        that id, its container is the SEGMENT FILE it lives in and its seq is the record's own
        seq -- the LATEST record per block wins by forward-scan position. A 0x0D delta is keyed
        `<block_id>/<n>` by its order within its block, with a per-block meta row
        `<block_id>/count` whose container carries the SUPERSEDED watermark (how many of that
        block's deltas were already folded when the newest 0x0C record for it was written -- a
        forward scan sees exactly that) and a per-block `<block_id>/gen` row under 0x0C holding
        the merge count. Those are precisely the rows `UnitBlockStore` writes, so D1 holds for the
        block tier: the store's rows are DERIVED, and a reorder that renumbers seqs stays
        re-derivable because the block id is IN the record rather than recoverable by arithmetic.
      * EVERY OTHER TAG KEEPS THE PROVISIONAL GLOBAL-ORDINAL KEYING, unchanged, because nothing
        else in the lake carries a key of its own yet. The self-check below runs on exactly those
        tags, so its rebuild equality, its planted-row red and its lake-tracking control read the
        same as they did before this change.
    """
    idx = LakeIndex(index_path)
    try:
        batch: list[tuple] = []
        n = 0
        newest_block: dict[int, tuple[str, int]] = {}
        block_records: dict[int, int] = {}
        delta_rows: dict[int, list[tuple[str, int]]] = {}
        superseded: dict[int, int] = {}
        for ordinal, (container, seq, payload) in enumerate(located_records(root)):
            tag = payload[0] if payload else 0
            where = (container, ordinal if seq is None else seq)
            if tag == _TAG_BLOCK:
                bid = UnitBlockRecord.decode(payload).block_id
                newest_block[bid] = where
                block_records[bid] = block_records.get(bid, 0) + 1
                superseded[bid] = len(delta_rows.get(bid, ()))
            elif tag == _TAG_DELTA:
                bid = BlockDeltaRecord.decode(payload).block_id
                delta_rows.setdefault(bid, []).append(where)
            else:
                batch.append((tag, str(ordinal), "lake", ordinal))
            n += 1
            if len(batch) >= 10_000:
                idx.put_batch(batch)
                batch.clear()
        for bid, (container, seq) in sorted(newest_block.items()):
            batch.append((_TAG_BLOCK, str(bid), container, seq))
            batch.append((_TAG_BLOCK, f"{bid}/gen", "meta", block_records[bid] - 1))
        for bid, rows in sorted(delta_rows.items()):
            batch.append(
                (_TAG_DELTA, f"{bid}/count", f"superseded={superseded.get(bid, 0)}", len(rows))
            )
            for k, (container, seq) in enumerate(rows):
                batch.append((_TAG_DELTA, f"{bid}/{k}", container, seq))
        idx.put_batch(batch, watermark=n)
        return n
    finally:
        idx.close()


def _delete_index(index_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(index_path) + suffix)
        if p.exists():
            p.unlink()


def assert_rebuild(root: str | os.PathLike, index_path: str | os.PathLike) -> tuple[bool, str]:
    """The gate: build -> snapshot digest -> DELETE -> rebuild -> digests equal."""
    index_path = Path(index_path)
    build_index(root, index_path)
    idx = LakeIndex(index_path)
    before, rows = idx.logical_dump_digest(), idx.row_count()
    idx.close()
    _delete_index(index_path)
    build_index(root, index_path)
    idx = LakeIndex(index_path)
    after = idx.logical_dump_digest()
    idx.close()
    if before != after:
        return False, f"REBUILD DIVERGED: {before[:16]} -> {after[:16]} over {rows} rows"
    return True, f"rebuild equality OK over {rows} rows (digest {before[:16]}...)"


def self_check() -> int:
    with tempfile.TemporaryDirectory(prefix="lake_rebuild_gate_") as td:
        root = Path(td)
        log = BinLog(str(root / "brain_lake.bin"), dedup=False)
        for i in range(50):
            log.append(bytes([i % 7 + 1]) + f"payload-{i}".encode())
        del log

        ok, msg = assert_rebuild(root, root / INDEX_NAME)
        if not ok:
            print(f"lake_rebuild_gate self-check FAILED (green path): {msg}")
            return 1

        # RED 1: a planted row behind the gate's back must fail the equality.
        idx = LakeIndex(root / INDEX_NAME)
        clean = idx.logical_dump_digest()
        idx.put_batch([(9, "planted-behind-the-gates-back", "nowhere", 0)])
        planted = idx.logical_dump_digest()
        idx.close()
        _delete_index(root / INDEX_NAME)
        build_index(root, root / INDEX_NAME)
        idx = LakeIndex(root / INDEX_NAME)
        rebuilt = idx.logical_dump_digest()
        idx.close()
        if planted == rebuilt or clean != rebuilt:
            print("lake_rebuild_gate self-check FAILED: planted divergence not detected")
            return 1

        # RED 2 / lake-tracking control: one appended record must move the rebuilt digest.
        BinLog(str(root / "brain_lake.bin"), dedup=False).append(b"\x05new-record")
        _delete_index(root / INDEX_NAME)
        build_index(root, root / INDEX_NAME)
        idx = LakeIndex(root / INDEX_NAME)
        grown = idx.logical_dump_digest()
        idx.close()
        if grown == clean:
            print("lake_rebuild_gate self-check FAILED: a changed lake did not change the index")
            return 1

        print(
            "lake_rebuild_gate self-check OK: rebuild equality; planted-divergence red fired; "
            "lake-tracking control moved the digest"
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--root":
        # Explicit-root assert (quiesced home or backup ONLY -- never the running daemon's).
        root = Path(argv[1])
        with tempfile.TemporaryDirectory(prefix="lake_rebuild_assert_") as td:
            ok, msg = assert_rebuild(root, Path(td) / INDEX_NAME)
        print(f"lake_rebuild_gate --root: {msg}")
        return 0 if ok else 1
    return self_check()


if __name__ == "__main__":
    sys.exit(main())
