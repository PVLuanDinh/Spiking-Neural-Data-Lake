"""Segment writer for the lake engine (specs/022 T005) -- WAL in the incumbent format,
sealed segments in Arrow IPC.

Design (research.md D2/D4, contracts/lake-records.md), COMPOSING the incumbent rather than
replacing it (P2):

  * The ACTIVE segment is a BinLog (`segments/active.wal`) -- the one framing whose
    torn-tail semantics are already proven and gate-tested (tests/test_cutover.py). Every
    append goes through it, so a crash mid-append is healed by machinery that predates this
    module.
  * Each record is wrapped in a CRC envelope before it enters the WAL:
        [u16 version][u32 crc32(payload)][payload]
    BinLog's u32 length completes the contract framing (magic lives at file level in both
    formats). The envelope travels INTO the sealed segment, so corruption is detectable in
    either format, at read time, per record.
  * `seal()` converts the WAL into an immutable Arrow IPC *file* (`seg-NNNNNNNN.lake`,
    schema seq:u64 / version:u16 / crc:u32 / payload:binary) -- footer-indexed, mmap-able,
    lock-free for parallel readers. THE DURABLE COMMIT ORDER IS THE POINT:
        write tmp -> fsync tmp -> os.replace to final -> RE-READ final and digest-compare
        against the WAL -> only then delete the WAL.
    The WAL is never deleted before the sealed copy has been read back equal, so no crash
    window loses records; a crash between replace and WAL-delete leaves BOTH, and recovery
    prefers the sealed file and drops the duplicate WAL (same digest = same records).
  * Read side: `read_segment()` verifies each record's CRC and REFUSES the segment loudly on
    a mismatch -- a sealed file is immutable, so a bad CRC is corruption, never a race.

`lake_bridge.iter_records` consumes `read_segment` for its segment arm (T005 un-stubs T005b's
refusal), keeping ONE reader path: brain_lake.bin, then sealed segments in filename order,
then the active WAL.

Bare invocation (run_selfchecks): construct a temp store, append/seal/append, verify the
bridge round-trip AND the CRC red (a flipped byte in a sealed file must refuse) -- the red and
its positive control in one run.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc

from snn_data_lake.substrate.binlog import BinLog

ENVELOPE_VERSION = 1
_ENV_HEAD = struct.Struct("<HI")  # version:u16, crc32:u32
SEGMENT_DIR = "segments"
WAL_NAME = "active.wal"

#: The sealed-segment schema, named once because `seal()` now writes batch by batch and every
#: batch must carry the identical schema for the footer to index them as one file.
SEGMENT_SCHEMA = pa.schema(
    [
        ("seq", pa.uint64()),
        ("version", pa.uint16()),
        ("crc", pa.uint32()),
        ("payload", pa.binary()),
    ]
)


class SegmentCorruption(RuntimeError):
    """A sealed segment failed its per-record CRC or structural check. Sealed files are
    immutable; this is corruption (or tampering), never a race -- refuse, never skip."""


def wrap(payload: bytes) -> bytes:
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError("segment payload must be non-empty bytes")
    payload = bytes(payload)
    return _ENV_HEAD.pack(ENVELOPE_VERSION, zlib.crc32(payload) & 0xFFFFFFFF) + payload


def unwrap(record: bytes, *, where: str = "?") -> bytes:
    if len(record) < _ENV_HEAD.size:
        raise SegmentCorruption(f"{where}: record shorter than its envelope header")
    version, crc = _ENV_HEAD.unpack_from(record)
    payload = record[_ENV_HEAD.size :]
    if version != ENVELOPE_VERSION:
        raise SegmentCorruption(f"{where}: unknown envelope version {version}")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise SegmentCorruption(f"{where}: CRC mismatch (stored {crc:#010x})")
    return payload


class SegmentStore:
    """Append records durably (WAL), seal immutable Arrow IPC segments on demand."""

    def __init__(self, root: str | os.PathLike):
        self.dir = Path(root) / SEGMENT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_seal()
        self._wal = BinLog(str(self.dir / WAL_NAME), dedup=False)

    # -- write path ------------------------------------------------------------------
    def append(self, payload: bytes) -> None:
        self._wal.append(wrap(payload))

    # distinct-from: binlog.py -- NOT a second bulk mechanism: a one-line DELEGATE to
    # BinLog.extend (the incumbent's own bulk path) that only adds this store's CRC envelope.
    def append_many(self, payloads) -> int:
        """Bulk append with ONE fsync via BinLog.extend (measured there at 21x-1341x over
        per-record fsync). The migrator's path: 156k records, one fsync each, would be the
        wrong tool."""
        return self._wal.extend([wrap(p) for p in payloads])

    def seal(self) -> Path | None:
        """Seal the active WAL into the next seg-NNNNNNNN.lake. Returns the sealed path,
        or None when the WAL holds no records (an empty seal is a no-op, not an error)."""
        records = list(self._wal.replay())
        if not records:
            return None
        seq0 = self._next_seq()
        final = self.dir / f"seg-{seq0:08d}.lake"
        tmp = final.with_suffix(".lake.tmp")
        # ONE ARROW RECORD BATCH PER RECORD (specs/022 T066). The first form wrote the whole
        # sealed segment as ONE table, so the footer indexed the segment and a record inside it
        # was reachable only by reading the segment whole -- which is not a paged read. One batch
        # per record makes the footer index the RECORDS, which is what `read_record` seeks by.
        # The BYTE LAYOUT OF A SEALED SEGMENT CHANGES, and it changes for every tenant because a
        # per-tenant layout would be two writers (P2). The change is FORWARD-ONLY and that is the
        # load-bearing part rather than the pre-registration's "no production segment exists yet",
        # which this commit did not re-measure: every existing READER is unaffected, because
        # `read_all()` concatenates batches whatever their width, so `read_segment`,
        # `iter_payloads`, `lake_bridge.iter_records` and `lake_rebuild_gate` read an old
        # whole-table segment and a new one identically. Only `read_record` requires the new
        # layout, and it REFUSES an old segment loudly -- at EVERY seq, with the same message --
        # rather than returning a wrong record. Asserted, not assumed, in
        # tests/test_lake_segments.py::test_read_record_refuses_a_pre_t066_whole_table_segment
        # (which constructs one with write_table, the writer this seal replaced) and in
        # tests/test_neuron_pager.py (the new-layout round trip and its out-of-range refusal).
        with open(tmp, "wb") as f:
            with pa_ipc.new_file(f, SEGMENT_SCHEMA) as w:
                for i, rec in enumerate(records):
                    version, crc = _ENV_HEAD.unpack_from(rec)
                    w.write_batch(
                        pa.record_batch(
                            [
                                pa.array([seq0 + i], pa.uint64()),
                                pa.array([version], pa.uint16()),
                                pa.array([crc], pa.uint32()),
                                pa.array([rec[_ENV_HEAD.size :]], pa.binary()),
                            ],
                            schema=SEGMENT_SCHEMA,
                        )
                    )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        # Verify-then-delete: the WAL dies only after the sealed copy READS BACK equal.
        sealed_payloads = list(read_segment(final))
        wal_payloads = [unwrap(r, where="wal") for r in records]
        if sealed_payloads != wal_payloads:
            raise SegmentCorruption(f"{final.name}: read-back != WAL; WAL retained")
        wal_path = self.dir / WAL_NAME
        del self._wal
        os.remove(wal_path)
        self._wal = BinLog(str(wal_path), dedup=False)
        return final

    # -- recovery --------------------------------------------------------------------
    def _recover_interrupted_seal(self) -> None:
        """A crash between os.replace and WAL-delete leaves both; the sealed file already
        read back equal (or the replace never happened), so the duplicate WAL is dropped
        IFF its records match the newest segment. A stray .tmp is always dropped (the
        replace never happened; the WAL still holds everything)."""
        for tmp in self.dir.glob("*.lake.tmp"):
            tmp.unlink()
        wal_path = self.dir / WAL_NAME
        segs = sorted(self.dir.glob("seg-*.lake"))
        if not (wal_path.exists() and segs):
            return
        wal_records = [unwrap(r, where="wal") for r in BinLog(str(wal_path), dedup=False).replay()]
        if wal_records and wal_records == list(read_segment(segs[-1]))[-len(wal_records) :]:
            os.remove(wal_path)

    def _next_seq(self) -> int:
        last = 0
        for seg in self.dir.glob("seg-*.lake"):
            with pa_ipc.open_file(seg) as r:
                n = r.read_all().num_rows
            last = max(last, int(seg.stem.split("-")[1]) + n - 1)
        return last + 1 if last else 1

    # -- read path -------------------------------------------------------------------
    def iter_payloads(self):
        """Sealed segments in order, then the active WAL -- one reader, oldest first."""
        for seg in sorted(self.dir.glob("seg-*.lake")):
            yield from read_segment(seg)
        for rec in self._wal.replay():
            yield unwrap(rec, where="wal")


def read_segment(path: str | os.PathLike):
    """Yield payloads from one sealed segment, verifying every record's CRC. Refuses the
    whole segment on any mismatch (immutable file, so corruption -- never skip-and-continue)."""
    path = Path(path)
    try:
        with pa_ipc.open_file(path) as r:
            table = r.read_all()
    except pa.ArrowInvalid as e:
        raise SegmentCorruption(f"{path.name}: not a valid sealed segment ({e})") from e
    crcs = table.column("crc").to_pylist()
    payloads = table.column("payload").to_pylist()
    for i, (crc, payload) in enumerate(zip(crcs, payloads)):
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise SegmentCorruption(f"{path.name}: record {i} CRC mismatch")
        yield payload


# DELETED 2026-09-09 (T068 LOSS): `read_record(path, seq)` and `read_records(path, lo, hi)`, the
# seq-addressed random-access readers over a sealed segment, were built for the paging layer's
# per-block page-in (specs/022 T066) and go with it -- that layer was measured at a 35.2x tick
# cost against a 3.0x bound fixed before the run and deleted on its null
# (evidence/s5-paging-null.md 3.4/3.5). Measured before removing them: the pager was their only
# production caller, and with it gone they had none.
#
# THE SEGMENT TIER KEEPS ITS READERS, so `seal` does not become write-only: `read_segment` above
# streams a whole segment CRC-checked (used live by `lake_bridge.located_records`, and through it
# by the migrator and the rebuild gate), and `SegmentStore.iter_payloads` reads the store's own
# WAL + sealed spans. What is gone is only O(1) addressing of ONE record by its global seq, which
# nothing but a pager ever wanted. A successor paging row rebuilds it with its own reds.


def main() -> int:
    # Bare self-check: round-trip + the CRC red, one run, constructed store.
    with tempfile.TemporaryDirectory(prefix="lake_segments_") as td:
        store = SegmentStore(td)
        for i in range(5):
            store.append(f"pre-seal-{i}".encode())
        sealed = store.seal()
        for i in range(3):
            store.append(f"post-seal-{i}".encode())
        got = list(store.iter_payloads())
        want = [f"pre-seal-{i}".encode() for i in range(5)] + [
            f"post-seal-{i}".encode() for i in range(3)
        ]
        if got != want:
            print(f"lake_segments self-check FAILED: order/content mismatch ({len(got)} records)")
            return 1
        # THE RED: flip one byte inside the sealed file's payload region; read must REFUSE.
        data = bytearray(sealed.read_bytes())
        idx = data.find(b"pre-seal-3")
        data[idx] ^= 0xFF
        sealed.write_bytes(bytes(data))
        try:
            list(read_segment(sealed))
        except SegmentCorruption as e:
            print(f"lake_segments self-check OK: round-trip 8/8; CRC red fired ({e})")
            return 0
        print("lake_segments self-check FAILED: corrupted segment was read without refusal")
        return 1


if __name__ == "__main__":
    sys.exit(main())
