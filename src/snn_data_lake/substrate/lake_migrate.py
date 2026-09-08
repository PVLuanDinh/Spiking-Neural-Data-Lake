"""Per-tag migration dispositions for the lake cutover (specs/022 T009, SYNTHESIS 3.5).

Every persisted record tag carries an explicit disposition BEFORE the new engine adopts
anything, so no record class is migrated by accident or stranded by silence. The engine
stores raw payload bytes (lake_segments wraps them in a CRC envelope and unwraps on read),
so for every legacy tag the honest disposition is **read-as-is**: migration is BYTE-IDENTITY,
and the round-trip red proves exactly that -- every record, bridge -> SegmentStore ->
seal -> read-back, per-tag digest equality at full backup scale.

An UNKNOWN tag in the source lake fails migration loudly (a tag with no disposition row is a
record class nobody decided about -- the analyze finding this module exists to close). New
record classes born in the new engine (snapshot/delta/goal/proposal, T010+) register HERE in
the same commit that introduces them, per the tasks.md standing clause.

Bare invocation (run_selfchecks): constructed multi-tag lake -> migrate -> per-tag digests
equal + the unknown-tag red fires. `--verify-backup <src>` runs the full-scale identity proof
against a quiesced copy (never the running home).
"""

from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
from pathlib import Path

from snn_data_lake.substrate.binlog import BinLog
from snn_data_lake.substrate.lake_bridge import iter_records
from snn_data_lake.substrate.lake_segments import SegmentStore

# Dispositions keyed by home.py's OWN tag constants (cited by SYMBOL -- the first draft of
# this table mapped growth/episodes to swapped bytes and put _T_STATE at 0x00; the unknown-tag
# red caught all three against the real lake BEFORE a byte migrated, which is the red's job).
# Census of the 2026-08-19 backup: 0x01=21,119  0x03=83,405  0x04=51,780  0x05=1  0x07=22.
# read-as-is: the engine stores raw bytes; migration is byte-identity per record.
TAG_DISPOSITIONS: dict[int, tuple[str, str]] = {
    0x01: ("read-as-is", "_T_FACT; byte-identity"),
    0x02: ("read-as-is", "_T_SPIKE (reserved AER exemplar; zero records measured); byte-identity"),
    0x03: (
        "read-as-is",
        "_T_GROWTH whole-substrate snapshots (the reconstructible delta CHAIN at T010 is a NEW class, never a rewrite of these)",
    ),
    0x04: ("read-as-is", "_T_EPISODE; byte-identity"),
    0x05: ("read-as-is", "_T_SELFBUILD; byte-identity"),
    0x06: ("read-as-is", "_T_CALIB (zero records in the backup; constant exists); byte-identity"),
    0x07: (
        "read-as-is",
        "_T_STATE name-keyed frames; byte-identity. The kernel_state.v1 frame (specs/022 T013) "
        "is ADDITIVE within this class: the layout version lives in the NAME, an old lake lacks "
        "the name and cold-starts, so no upcaster exists to write -- a future layout ships "
        "kernel_state.v2 with its own disposition (SYNTHESIS 3.5)",
    ),
    0x08: (
        "read-as-is",
        "_T_GROWTH_DELTA reconstructible growth chain (FULL/DELTA, specs/022 T010-ii); "
        "byte-identity -- the upcaster is IDENTITY because chain semantics live in the reader "
        "(home.load_growth verifies link hashes on replay), so migrating the bytes migrates the chain",
    ),
    0x09: (
        "read-as-is",
        "_T_SYNDELTA spike-derived synapse deltas (lake_engine.SynapseDelta, specs/022 T010a); "
        "byte-identity -- upcaster is IDENTITY, record semantics live in SynapseDelta.entries",
    ),
    0x0A: (
        "read-as-is",
        "_T_COALITION settled workspace coalitions (global_workspace.Coalition, specs/022 "
        "T031); byte-identity -- upcaster is IDENTITY, record semantics live in "
        "Coalition.from_bytes (JSON body; zero records predate this tag)",
    ),
    0x0B: (
        "read-as-is",
        "_T_GOAL goal event rows (learning_progress.GoalStore, specs/022 T046a); v1 "
        "READ-AS-IS -- JSON body rows folded on replay, zero records predate this tag; "
        "the round-trip red over live-written records runs in the store's own main()",
    ),
    0x0C: (
        "read-as-is",
        "_T_UNITBLOCK per-block out-edge records (lake_engine.UnitBlockRecord, specs/022 T066); "
        "v1 byte-identity -- upcaster is IDENTITY, record semantics live in "
        "UnitBlockRecord.decode (a `<BIIH` header then four parallel LE arrays, refused WHOLE on "
        "any malformation). Zero records predate this tag; the round-trip red over written "
        "records runs in substrate/neuron_pager's own main() and in tests/test_neuron_pager.py",
    ),
    0x0D: (
        "read-as-is",
        "_T_BLOCKDELTA, and TWO record shapes live under it: v1 per-block out-edge deltas "
        "(lake_engine.BlockDeltaRecord, specs/022 T067 -- a `<BIIQ` header then five parallel LE "
        "arrays) and v2 ADOPT records (lake_engine.BlockAdoptRecord, the adoption fence). Both are "
        "read-as-is and the upcaster is IDENTITY for both; delta semantics live in "
        "BlockDeltaRecord.decode (refused WHOLE on any malformation) and in "
        "lake_engine.apply_deltas, which the reader runs against the block's CURRENT generation. "
        "THE HAZARD, stated because migrating the bytes DOES NOT migrate the tier here the way "
        "0x08's chain semantics travel with home.load_growth: an ADOPT record names its adopted "
        "0x0C record by (container, seq) and a container is a SEGMENT FILENAME, valid only inside "
        "the root that wrote it. A migration re-containerises, so it can report equal=True per tag "
        "over a destination root whose block tier then refuses with FileNotFoundError at the first "
        "page-in -- byte-identity per record is exactly what this table promises and it is not "
        "enough for this tag. Migration is NOT made to refuse: the fix belongs to the reclaimer "
        "row, which anchors the fence on the ADOPT record's POSITION (the 0x0C record and the "
        "deltas preceding it) rather than on the address it carries -- position is invariant under "
        "any order-preserving copy, and RETENTION[0x0D] carries the same reading. Zero records "
        "predate this tag; the round-trip red over written records runs in "
        "substrate/lake_compactor's own main() and in tests/test_lake_compactor.py",
    ),
}
# Anything else is UNKNOWN and refuses -- a record class nobody decided about.
_OBSERVED_EXTRA: dict[int, str] = {}


class UnknownTag(RuntimeError):
    """A record class nobody decided about. Migration refuses rather than guessing."""


def _tag_of(payload: bytes) -> int:
    return payload[0] if payload else 0


def _disposition(tag: int) -> str:
    if tag in TAG_DISPOSITIONS:
        return TAG_DISPOSITIONS[tag][0]
    if tag in _OBSERVED_EXTRA:
        return "read-as-is"
    raise UnknownTag(
        f"tag {tag:#04x} has no disposition row -- decide it in TAG_DISPOSITIONS before migrating"
    )


def migrate(src_root: str | Path, dst_root: str | Path, *, seal_every: int = 50_000) -> dict:
    """Stream every source record into a SegmentStore at dst_root (identity migration),
    sealing every `seal_every` records. Returns per-tag counts + digests for both sides;
    raises UnknownTag on any undecided record class BEFORE writing it."""
    store = SegmentStore(dst_root)
    src_digests: dict[int, hashlib._Hash] = {}
    counts: dict[int, int] = {}
    batch: list[bytes] = []
    for payload in iter_records(src_root):
        tag = _tag_of(payload)
        _disposition(tag)  # refuses UNKNOWN before the byte is written
        h = src_digests.setdefault(tag, hashlib.sha256())
        h.update(struct.pack("<I", len(payload)))
        h.update(payload)
        counts[tag] = counts.get(tag, 0) + 1
        batch.append(payload)
        if len(batch) >= seal_every:
            store.append_many(batch)
            store.seal()
            batch.clear()
    if batch:
        store.append_many(batch)
        store.seal()

    dst_digests: dict[int, hashlib._Hash] = {}
    dst_counts: dict[int, int] = {}
    for payload in store.iter_payloads():
        tag = _tag_of(payload)
        h = dst_digests.setdefault(tag, hashlib.sha256())
        h.update(struct.pack("<I", len(payload)))
        h.update(payload)
        dst_counts[tag] = dst_counts.get(tag, 0) + 1

    report = {}
    for tag in sorted(set(counts) | set(dst_counts)):
        s = src_digests.get(tag).hexdigest() if tag in src_digests else "-"
        d = dst_digests.get(tag).hexdigest() if tag in dst_digests else "-"
        report[tag] = {
            "disposition": _disposition(tag),
            "src_n": counts.get(tag, 0),
            "dst_n": dst_counts.get(tag, 0),
            "equal": s == d and counts.get(tag) == dst_counts.get(tag),
        }
    return report


def _verdict(report: dict) -> bool:
    return all(r["equal"] for r in report.values()) and bool(report)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--verify-backup":
        src = Path(argv[1])
        with tempfile.TemporaryDirectory(prefix="lake_migrate_") as td:
            report = migrate(src, td)
        for tag, r in sorted(report.items()):
            print(
                f"  tag {tag:#04x} [{r['disposition']}] src={r['src_n']} dst={r['dst_n']} equal={r['equal']}"
            )
        ok = _verdict(report)
        print(
            f"lake_migrate --verify-backup: {'IDENTITY HOLDS per tag' if ok else 'MIGRATION DIVERGED'}"
        )
        return 0 if ok else 1

    # Bare self-check: constructed multi-tag lake -> identity + the unknown-tag red.
    with tempfile.TemporaryDirectory(prefix="lake_migrate_") as td:
        src = Path(td) / "src"
        src.mkdir()
        log = BinLog(str(src / "brain_lake.bin"), dedup=False)
        for tag in (0x01, 0x03, 0x04, 0x05):
            for i in range(20):
                log.append(bytes([tag]) + f"rec-{tag}-{i}".encode())
        del log
        report = migrate(src, Path(td) / "dst")
        if not _verdict(report) or sorted(report) != [0x01, 0x03, 0x04, 0x05]:
            print(f"lake_migrate self-check FAILED: identity did not hold ({report})")
            return 1
        # THE RED: an undecided tag must refuse BEFORE anything is written.
        BinLog(str(src / "brain_lake.bin"), dedup=False).append(b"\x7fundecided-class")
        try:
            migrate(src, Path(td) / "dst2")
        except UnknownTag as e:
            print(f"lake_migrate self-check OK: 4-tag identity held; unknown-tag red fired ({e})")
            return 0
        print("lake_migrate self-check FAILED: an undecided tag migrated silently")
        return 1


if __name__ == "__main__":
    sys.exit(main())
