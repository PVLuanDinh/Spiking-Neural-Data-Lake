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
from snn_data_lake.substrate.lake_bridge import located_records
from snn_data_lake.substrate.lake_index import LakeIndex

INDEX_NAME = "lake_index.db"


def build_index(root: str | os.PathLike, index_path: str | os.PathLike) -> int:
    """Rebuild the pointer index from nothing but the lake. Returns records scanned.

    ONE KEYING: every tag is keyed by its GLOBAL ORDINAL, which is provisional because nothing in
    the lake carries a key of its own.

    THE TAG-AWARE BLOCK BRANCH IS DELETED (2026-09-09, T068 LOSS). Until then this function had a
    second keying for the block tier -- the 0x0D ADOPT-record fence (adoption is a RECORD, never a
    position) and `<block_id>/<n>` delta rows -- written for the paging layer that specs/022
    evidence/s5-paging-null.md 3.4 measured at a 35.2x tick cost against a 3.0x bound and deleted
    on its null. It goes with the tags it keyed: 0x0C and 0x0D no longer exist in RETENTION or in
    TAG_DISPOSITIONS, so the branch could only ever have been dead code.

    IT WAS NEVER EXERCISED BY THIS MODULE'S OWN SELF-CHECK, measured rather than assumed: the
    check below constructs tags 1-7 only, so the branch's equality, its planted-row red and its
    lake-tracking control never entered it. That is stated here because it is the honest reason
    its deletion changes no gate reading -- not evidence that the branch was correct. A successor
    paging row that re-registers a block tier owes the fence AND a self-check arm that reaches it.
    """
    idx = LakeIndex(index_path)
    try:
        batch: list[tuple] = []
        n = 0
        for ordinal, (_container, _seq, payload) in enumerate(located_records(root)):
            tag = payload[0] if payload else 0
            batch.append((tag, str(ordinal), "lake", ordinal))
            n += 1
            if len(batch) >= 10_000:
                idx.put_batch(batch)
                batch.clear()
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
