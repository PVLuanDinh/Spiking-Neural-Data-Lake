"""Derived pointer index over the lake (specs/022 T007) -- provably rebuildable, never trusted.

# distinct-from: streaming_hub.py -- its coarse (time, offset) index maps TIME to seek points
# inside per-channel .spk telemetry files; this maps (record tag, key) to a CONTAINER+SEQ
# location across the whole durable lake, with an applied-watermark contract. Different key
# space, different files, different consumer. The 022 audit (evidence: lake-internals verify)
# measured that NO on-disk index of any kind exists over brain_lake.bin -- this is that gap.
# distinct-from: proximity_index / LAESA -- content-addressable SIMILARITY search; this is
# exact pointer lookup. The recall contract (T057) composes THAT; this serves record fetch.

research.md D3 deferred the engine choice to a measured decision with the criteria pinned:
single-writer + N readers, cross-process readers, short-lived snapshots, a recorded
amplification trade. RESOLVED: **sqlite3 in WAL mode (stdlib)** --

  * single-writer/N-reader is native (WAL readers never block the writer and vice versa);
  * cross-process locking is native (busy_timeout, OS-level);
  * snapshots are per-connection read transactions -- opened per batch, never held across
    ticks (the MVCC-growth hazard D3 names);
  * zero new dependencies on cp314t (P7 favours the dependency that already shipped);
  * amplification trade, WRITTEN DOWN: B-tree write amplification on random upserts, bought
    knowingly -- the write shape is BATCHED pointer appends (mostly-increasing keys), a
    B-tree's cheap case. MEASURED on the pinned host at landing (evidence/s0-lake-index.md);
    REVISIT at S2 paging widths per D3 if the shape turns random-heavy.

THE INDEX IS DERIVED STATE. The lake is the only source of truth (P4b): every row here must
be reconstructible by a forward scan of the lake, and `applied_watermark` records how far the
lake has been folded in. On open, a stale watermark means replay-the-tail, never trust-the-
rows. T008's rebuild gate asserts delete -> rebuild -> logical-dump equality with a planted-
divergence red.

Bare invocation (run_selfchecks): temp index -- put/get round-trip, watermark persistence
across reopen, logical-dump determinism, and the divergence red (a planted extra row must
change the dump digest).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    tag  INTEGER NOT NULL,
    key  TEXT    NOT NULL,
    container TEXT NOT NULL,
    seq  INTEGER NOT NULL,
    PRIMARY KEY (tag, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
) WITHOUT ROWID;
"""


class LakeIndex:
    """One writer, many readers; every row derivable from the lake."""

    def __init__(self, path: str | os.PathLike, *, read_only: bool = False):
        self.path = Path(path)
        uri = f"file:{self.path.as_posix()}?mode={'ro' if read_only else 'rwc'}"
        self._db = sqlite3.connect(uri, uri=True, timeout=30.0)
        self._db.execute("PRAGMA busy_timeout=30000")
        if not read_only:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    # -- write path (the ONE writer thread/process) ------------------------------------
    # distinct-from: growth/h01_ingest.py -- a JSON pin of H01 stage records; no lake index. Site read.
    # distinct-from: instruments/carried_state_report.py -- report rows; no lake index. Site read.
    def put_batch(self, entries, *, watermark: int | None = None, meta: dict | None = None) -> None:
        """Fold a batch of (tag, key, container, seq) pointer rows, optionally advancing the
        applied watermark IN THE SAME TRANSACTION -- rows and the how-far marker move
        together or not at all, so a crash can never record progress it did not make.
        `meta` pairs (e.g. a compaction anchor) ride in the same transaction."""
        pairs = dict(meta or {})
        if watermark is not None:
            pairs["applied_watermark"] = str(int(watermark))
        with self._db:
            self._db.executemany(
                "INSERT INTO records (tag, key, container, seq) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tag, key) DO UPDATE SET container=excluded.container, seq=excluded.seq",
                entries,
            )
            self._db.executemany(
                "INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                [(k, str(v)) for k, v in pairs.items()],
            )

    # -- read path ----------------------------------------------------------------------
    def get(self, tag: int, key: str):
        row = self._db.execute(
            "SELECT container, seq FROM records WHERE tag=? AND key=?", (tag, key)
        ).fetchone()
        return None if row is None else (row[0], int(row[1]))

    def clear(self, tag: int, meta_keys) -> None:
        """Delete every row of `tag` AND the named meta keys in ONE transaction: both or neither."""
        with self._db:
            self._db.execute("DELETE FROM records WHERE tag=?", (tag,))
            self._db.executemany("DELETE FROM meta WHERE k=?", [(k,) for k in meta_keys])

    def data_version(self) -> int:
        """SQLite's PRAGMA data_version: it changes when ANOTHER connection commits to this file (never
        for this connection's own commits), so a caller that cached a verdict can tell the rows moved."""
        return int(self._db.execute("PRAGMA data_version").fetchone()[0])

    def iter_range(self, tag: int, lo: str, hi: str, *, reverse: bool = False):
        """(key, seq) pairs of [lo, hi) in key order (descending when `reverse`), streamed from the
        cursor (no list). A caller that stops early should close the generator."""
        order = "DESC" if reverse else "ASC"
        cur = self._db.execute(
            f"SELECT key, seq FROM records WHERE tag=? AND key>=? AND key<? ORDER BY key {order}",
            (tag, lo, hi),
        )
        for key, seq in cur:
            yield key, int(seq)

    def count_range(self, tag: int, lo: str, hi: str) -> int:
        sql = "SELECT COUNT(*) FROM records WHERE tag=? AND key>=? AND key<?"
        return int(self._db.execute(sql, (tag, lo, hi)).fetchone()[0])

    def meta(self, k: str):
        row = self._db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return None if row is None else row[0]

    def applied_watermark(self) -> int:
        row = self._db.execute("SELECT v FROM meta WHERE k='applied_watermark'").fetchone()
        return int(row[0]) if row else 0

    def logical_dump_digest(self) -> str:
        """Canonical LOGICAL dump (sorted rows), hashed -- the rebuild gate's comparator.
        Deliberately not a byte-compare of the .db file: page layout is engine noise; the
        ROWS are the semantics (analyze finding, round 2 engineering lens)."""
        h = hashlib.sha256()
        for row in self._db.execute(
            "SELECT tag, key, container, seq FROM records ORDER BY tag, key"
        ):
            h.update(repr(row).encode())
        h.update(str(self.applied_watermark()).encode())
        for k, v in self._db.execute(
            "SELECT k, v FROM meta WHERE k != 'applied_watermark' ORDER BY k"
        ):  # derived markers (a compaction anchor) are logical state too
            h.update(repr((k, v)).encode())
        return h.hexdigest()

    def row_count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def close(self) -> None:
        self._db.close()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lake_index_") as td:
        p = os.path.join(td, "index.db")
        idx = LakeIndex(p)
        idx.put_batch([(1, f"k{i}", "seg-00000001.lake", i) for i in range(100)], watermark=100)
        if idx.get(1, "k42") != ("seg-00000001.lake", 42):
            print("lake_index self-check FAILED: round-trip")
            return 1
        d1 = idx.logical_dump_digest()
        idx.close()
        idx2 = LakeIndex(p)  # persistence across reopen
        if idx2.applied_watermark() != 100 or idx2.logical_dump_digest() != d1:
            print("lake_index self-check FAILED: persistence/determinism across reopen")
            return 1
        # THE DIVERGENCE RED: a planted extra row must move the logical dump digest.
        idx2.put_batch([(9, "planted", "nowhere", 0)])
        if idx2.logical_dump_digest() == d1:
            print("lake_index self-check FAILED: planted divergence did not move the digest")
            return 1
        idx2.close()
        print("lake_index self-check OK: round-trip, reopen persistence, divergence red fired")
        return 0


if __name__ == "__main__":
    sys.exit(main())
