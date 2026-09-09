# Contract: lake records and the engine

The durable interface (P4b). The lake is the ONE store; every index is derived and disposable.

## Framing (every record)
```
[magic][version:u16][length:u32][payload][crc32]
```
- Recovery: forward scan, halt at first failed CRC, truncate to last-known-good offset. Tested with injected torn writes INCLUDING an uncorrupted positive-control run in the same command.
- Commit: bytes flushed → durable watermark published (atomic-rename manifest or commit record). File length is NEVER the commit signal.

## Segments
- Sealed immutable Arrow IPC files; footer = block offsets ("replay from tick T" is a lookup).
- Compaction: epoch-fenced, COMPACTABLE classes only, relocation proof precedes reclamation.

## Record classes and retention
| class | retention | notes |
|---|---|---|
| UnitBlock | IMMORTAL (current + delta chain) | block-aligned for one-read paging | **DELETED 2026-09-09 (T068 LOSS): tag 0x0C, together with the 0x0D delta class. No lake ever held a record under either -- the live home has no `segments/` and no `lake_index.db` -- so no durable record changes class. The tag bytes are free.**
| SynapseDelta | IMMORTAL as reconstructible | spike-derived dirty set; quantize-compare-drop |
| Episode / Coalition | IMMORTAL | recall contract operates here |
| Goal / Proposal / Provenance | IMMORTAL | the archive is the rollback mechanism |
| Snapshot | COMPACTABLE | bounded recovery aid; includes RNG stream positions (G5) |
| Derived read-models | COMPACTABLE | drop-and-rebuild is a supported, exercised operation |
- Every IMMORTAL class carries a bytes/tick ceiling with a red-capable assert.

## Index contract
- Rebuildable: snapshot index → delete → rebuild from lake → canonical logical dumps EQUAL.
- The index persists its applied watermark; on open it replays the lake tail.
- Single writer thread, N readers; snapshots short-lived (per tick / per batch); cross-process file lock on the writer path.

## Paging contract (L2) *DELETED ON ITS NULL 2026-09-09 (T068 LOSS, evidence/s5-paging-null.md 3.4/3.5) -- the paging contract below is the record of a deleted mechanism. Note that the 128-byte alignment other rows cite as "the paging contract" (data-model.md) is untouched: SoA block reads justify it independently of any pager.*
- `fetch(units) -> one async batch submission`; per-tick integer ROUNDS budget asserted (red-capable via constructed overload).
- A spike arriving at a paged-out unit: never silently dropped; bounded staleness; the constructed red exists BEFORE the mechanism.
- Residency: activation-driven custom policy (never OS/mmap eviction); pinned hot set for executive/gating units.
- Null clause: paging must beat forced-total residency at the derived binding width, or the layer is deleted and the deletion recorded. *DATED NOTE 2026-09-09: this null clause is the one that FIRED. Paging did not beat forced-total residency under the pre-registered rule, so "the layer is deleted and the deletion recorded" is the branch executed (T068).*

## Migration (cutover, §3.5)
- One disposition per persisted tag: read-as-is | upcast-on-read | one-time-rewrite — each with a round-trip red over LIVE records of that tag.
- Old segments replay under new code via an upcaster registry keyed on record version.
