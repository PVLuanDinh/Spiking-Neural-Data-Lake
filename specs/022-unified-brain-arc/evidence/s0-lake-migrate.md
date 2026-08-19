# T009 evidence — per-tag migration identity at full scale

**Census of the backup lake** (D:\backups\snn_lake_pre022_20260819, 2026-08-19):
`0x01 _T_FACT 21,119 · 0x03 _T_GROWTH 83,405 · 0x04 _T_EPISODE 51,780 · 0x05 _T_SELFBUILD 1 ·
0x07 _T_STATE 22` — 156,327 records.

**Round-trip**: every record streamed bridge → SegmentStore (bulk WAL, seal per 50k) →
sealed-segment read-back. **Per-tag length-prefixed digests equal on all five classes**
(`--verify-backup` output committed above). Disposition: read-as-is for every legacy tag —
the engine stores raw bytes, so migration is byte-identity; the delta chain replacing
`_T_GROWTH` snapshots at T010 is a NEW class, never a rewrite of these.

**The red earned its keep at first contact**: the draft table mapped growth/episodes to
swapped bytes and placed `_T_STATE` at 0x00; `UnknownTag` refused on the real lake (0x07)
before a byte migrated, and the corrected table is keyed to `home.py`'s own `_T_*` constants
by symbol. An undecided tag can never migrate silently.
