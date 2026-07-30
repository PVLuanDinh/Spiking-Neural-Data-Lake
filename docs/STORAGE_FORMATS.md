# Spike storage formats

The repo uses several on-disk formats, each tuned to a different access pattern. This is the
one place that says which is which, so the `.spk / .spc / .spkl` sprawl isn't guesswork.

| Ext | Written by | Read by | Layout | Why it exists |
|-----|-----------|---------|--------|---------------|
| `.spk` | `spike_telemetry_hub.py` | `paradigm_b_{engine,matcher,genn}.py` | Sparse binary, magic header + per-channel index (offset+count) | **Partial reads** — query one channel without scanning the file (in-storage SNN query) |
| `.spkl` | `streaming_hub.py` | `streaming_hub.py` | Durable append-only event log, flushed per spike | **Streaming** — record *while* querying; events land as they fire, replayable |
| `.spc` | `signal_loop.py`, `spike_preprocessing.py` | same | gzip-compressed encoded windows | **Storage-cheap lake** — compact at-rest store of encoded signal windows / preprocessing cache |
| `.parquet` | `lakehouse/medallion.py` (polars) | medallion, polars SQL | Columnar (Bronze/Silver/Gold) | **Analytics** — predicate pushdown + column pruning; the lakehouse data path |
| Delta | `lakehouse/delta_demo.py` (delta-rs) | polars `read_delta` | Parquet + transaction log | **Streaming lakehouse** — transaction log + `OPTIMIZE` compaction kills the small-file problem; ACID snapshots + time-travel. Needs `deltalake` in `.venv-lake`. |
| `.jsonl` | `signal_loop.py` | `signal_loop.py` | JSON lines | **Unknowns** — non-matching windows queued for clustering / continual learning |

## Common invariants
- **Bronze events are immutable** — `(t:int, channel:int)`, append-only. `data_quality.bronze_hash`
  pins content; `data_quality.gate()` blocks Gold promotion on schema / duplicate / encodability /
  immutability / Gold-sanity failure (in `medallion.py`).
- `.spk` and `.spkl` both store raw spike events; `.spk` is **index-optimized for reads**, `.spkl`
  is **append-optimized for writes**. They are not interchangeable — pick by access pattern.

## Scope
Convergence to a single format isn't worth it: each format earns its place by a distinct access
pattern (partial-read vs append vs compress vs columnar). The cost was *documentation*, not *count* —
that's what this file fixes.

## `.spkl` record widths and the packed-store header (2026-07-29)

A `.spkl` log is a sequence of fixed-width records; two widths exist:

- **Default (8 B/event):** `<II` little-endian `(channel u32, time u32)`. A default store is a raw
  headerless record log — byte-identical to every `.spkl` ever written, including the live
  `brain_data/*.spkl` stores.
- **Packed (4 B/event, opt-in):** one u32 = `(channel << 26) | time`, lossless for `channel < 64`
  and `time < 2^26`; out-of-range values raise at append.

**The header.** The packed width used to live only in the caller's `packed=True` flag; reopening a
packed store without it silently misframed 4 B records as 8 B ones (measured: a 100-event packed
store read back as **1** event). Every packed store written since 2026-07-29 therefore begins with an
8 B header:

```
offset 0: 4s  magic     b"SPKL"
offset 4: B   version   1
offset 5: B   record kind   0 = 8 B default, 1 = 4 B packed
offset 6: H   reserved  0
```

Records start at byte 8. On open, `StreamingHub` probes the first 8 bytes: a valid header **decides**
the record width (an explicit `packed=True` against a header that says 8 B raises `ValueError`
instead of misreading); an absent/invalid magic means a **legacy headerless** store, read exactly as
before under the caller's flag. Default stores are still written headerless, so no existing size
invariant or reader changes. No migration is needed or performed: legacy files stay legacy for their
lifetime and remain fully readable.

Residual detection ambiguity is negligible by construction: a legacy 8 B store would need first-record
channel `0x4C4B5053` (~1.28e9; channels are bounded by `n_channels`, at most a few hundred here), and a
legacy 4 B packed store would need two specific consecutive records. Neither occurs in any real store.

## The one brain lake (`brain_lake.bin`) record tags (2026-07-29)

`BinLog` frames (`SNDLBIN1` magic + u32 length prefixes, fsync per append, torn-tail repair) carry a
1-byte type tag per record. `_T_FACT = 0x01` is the symbolic view (`KnowledgeLog`), the only tag with
a live writer (19,775 records on the production lake, 2026-07-29). `_T_SPIKE = 0x02` is **reserved**:
its writer (`SpikeLake`, durable AER exemplars) was retired per P3 after measuring 0 records ever
written, zero constructors outside its own self-check, and no live consumer — see the tag comment in
`substrate/home.py`. Unknown/foreign tags are skipped by every view and preserved by
`KnowledgeLog.consolidate`, so a historical lake containing them replays fine.
