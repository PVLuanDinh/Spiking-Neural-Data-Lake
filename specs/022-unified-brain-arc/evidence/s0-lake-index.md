# T007 evidence — index engine choice (research.md D3 addendum)

**Chosen**: sqlite3 WAL mode (stdlib). Criteria from D3, all satisfied natively:
single-writer/N-reader (WAL), cross-process locks (busy_timeout), short-lived per-batch
snapshots, zero new deps on cp314t.

**Amplification trade, measured on the pinned host (2026-08-19)**: batched pointer upserts
**1,250,788 rows/s**; singleton upserts **204,683 rows/s** (6.1× batch advantage — hence
`put_batch` is the only write API, watermark folded into the same transaction). The write
shape is batched, mostly-increasing keys — a B-tree's cheap case. REVISIT per D3 at S2 paging
widths if the shape turns random-heavy; the LSM alternative stays recorded in research.md.

Self-check (bare, run_selfchecks): round-trip, reopen persistence, logical-dump determinism,
planted-divergence red — green at landing. Close-discipline note: connections must be closed
before their directory is removed (Windows locks the .db; the landing measurement script
tripped exactly this and the module's own self-check does not).
