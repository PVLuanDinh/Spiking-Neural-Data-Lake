"""Lake format bridge -- read-old/write-new across BinLog and IPC segments (specs/022 T005b).

The FIRST of the two persistence-continuity commits (SYNTHESIS 3.5): before any new engine
writes a byte, one reader must yield every record from BOTH formats in append order, so old
records are never stranded behind new code. `iter_records(root)` yields payload bytes from:

  1. the incumbent `brain_lake.bin` (BinLog v1 frames, `SNDLBIN1` magic) -- delegated to
     `BinLog.replay()`, which owns torn-tail semantics; the bridge NEVER reimplements framing
     (P2: one mechanism per job);
  2. any sealed segment files under `segments/` (`seg-*.lake`), in filename order -- the new
     engine's output once T005 lands. Until then the directory is absent and this arm yields
     nothing, which is the correct read of a pre-022 home.

THE ROUND-TRIP RED (proven in tests/test_lake_bridge.py and run once against the LIVE lake on
the pinned host, recorded in specs/022-unified-brain-arc/evidence/): the bridge's payload
stream over a BinLog-only home must be BYTE-IDENTICAL to BinLog.replay() -- same records, same
order, same bytes -- verified as a rolling sha256 over length-prefixed payloads, with a
mutated-payload control arm in the same command (the hash MUST diverge when one byte changes,
or the equality proves nothing). `--live` runs it against $NORD_HOME -- point that at a
QUIESCED home or the barrier backup, NEVER the running daemon's home: constructing a second
BinLog runs tail-repair, and a frame the daemon is mid-appending reads as torn and would be
TRUNCATED under it (the writer-vs-probe hazard tests/test_cutover.py exists to police).

Bare invocation (run_selfchecks) is that same round-trip on a small CONSTRUCTED lake in a temp
dir -- cheap, self-contained, never touches the real home.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from snn_data_lake.substrate.binlog import BinLog

SEGMENT_DIR = "segments"
SEGMENT_GLOB = "seg-*.lake"


def iter_records(root: str | os.PathLike) -> Iterator[bytes]:
    """Yield every stored payload under `root`, oldest first: BinLog first (the incumbent
    history), then sealed segments in filename order (the future). Absent pieces are skipped
    silently -- an empty home yields nothing, which is a fact, not an error.

    A ONE-LINE DELEGATE to `located_records` since specs/022 T067, so the payload stream and the
    located stream can never drift into two forward readers (P2). The order, the skips and the
    torn-tail semantics are unchanged, which is what `tests/test_lake_bridge.py` pins."""
    for _container, _seq, payload in located_records(root):
        yield payload


# distinct-from: iter_records directly above -- the SAME single forward walk; that one drops the
#   location and this one carries it. The rebuild gate needs (container, seq) to re-derive the
#   block tier's own pointer rows, which a payload-only stream cannot supply, and a second walk
#   beside this one would be the P2 defect this delegation exists to avoid.
# distinct-from: substrate/lake_segments.py::SegmentStore.iter_payloads -- the STORE's own read
#   over its segments AND its active WAL; this one deliberately does not read the WAL (that arm is
#   the store's, pinned by tests/test_lake_bridge.py) and it starts at brain_lake.bin.
#   Both sites read 2026-09-08.
def located_records(root: str | os.PathLike):
    """`(container, seq, payload)` for every stored record under `root`, oldest first.

    `container` is "lake" for `brain_lake.bin` and the segment FILE NAME for a sealed segment;
    `seq` is the segment's global record seq, or None inside the BinLog, which has no seq space of
    its own (a caller that needs one there uses the record's forward-scan ordinal and says so)."""
    root = Path(root)
    lake = root / "brain_lake.bin"
    if lake.exists():
        for payload in BinLog(str(lake), dedup=False).replay():
            yield "lake", None, payload
    seg_dir = root / SEGMENT_DIR
    if seg_dir.is_dir():
        for seg in sorted(seg_dir.glob(SEGMENT_GLOB)):
            base = int(seg.stem.split("-")[1])
            for i, payload in enumerate(_iter_segment(seg)):
                yield seg.name, base + i, payload


def _iter_segment(path: Path) -> Iterator[bytes]:
    """Sealed-segment reading, delegated to the writer's own verified reader (T005) --
    CRC-checked per record; a mismatch refuses the segment loudly (SegmentCorruption)."""
    from snn_data_lake.substrate.lake_segments import read_segment  # noqa: PLC0415

    yield from read_segment(path)


def stream_digest(records: Iterator[bytes]) -> str:
    """Order- and boundary-sensitive digest: sha256 over u32-length-prefixed payloads.
    Length prefixes make [b"ab", b"c"] and [b"a", b"bc"] distinct, so equality means
    same records, same order, same bytes."""
    h = hashlib.sha256()
    for payload in records:
        h.update(struct.pack("<I", len(payload)))
        h.update(payload)
    return h.hexdigest()


def round_trip_ok(root: str | os.PathLike) -> tuple[bool, str]:
    """The T005b red, with its control IN THE SAME CALL: bridge digest must equal the direct
    BinLog digest, AND a single-byte mutation of the last payload must move the digest."""
    root = Path(root)
    lake = root / "brain_lake.bin"
    if not lake.exists():
        return True, "no lake at root (empty home round-trips trivially)"
    direct = list(BinLog(str(lake), dedup=False).replay())
    d_direct = stream_digest(iter(direct))
    d_bridge = stream_digest(iter_records(root))
    if d_bridge != d_direct:
        return False, f"BRIDGE DIVERGES: bridge {d_bridge[:16]} != direct {d_direct[:16]}"
    if direct:
        mutated = direct[:-1] + [bytes([direct[-1][0] ^ 0xFF]) + direct[-1][1:]]
        d_control = stream_digest(iter(mutated))
        if d_control == d_direct:
            return False, "CONTROL FAILED: a mutated payload did not move the digest"
    return (
        True,
        f"round-trip OK over {len(direct)} records (digest {d_direct[:16]}...; mutation control moved it)",
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--live":
        from snn_data_lake.substrate.home import home  # noqa: PLC0415

        ok, msg = round_trip_ok(home())
        print(f"lake_bridge --live: {msg}")
        return 0 if ok else 1
    # Bare (run_selfchecks): constructed lake in a temp dir, never the real home.
    with tempfile.TemporaryDirectory(prefix="lake_bridge_") as td:
        log = BinLog(os.path.join(td, "brain_lake.bin"), dedup=False)
        for i in range(7):
            log.append(f"bridge-selfcheck-{i}".encode())
        ok, msg = round_trip_ok(td)
        print(f"lake_bridge self-check: {msg}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
