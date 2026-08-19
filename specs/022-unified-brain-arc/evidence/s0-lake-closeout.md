# T012 — lake-block close-out evidence (2026-08-19, pinned host LUAN_AND_ANNIE)

## 1. The daemon's HEAD-poll restart lands on the watermark under the NEW engine

T005a's barrier, re-proven on the LIVE daemon across five push-triggered restarts in one
afternoon (each push is a live cutover rehearsal): heads `c73b14c8 → 1ce8951f → 7959d3e0 →
b25aea54 → 47eaece7 → ff999684`, every boot logging `watermark: clean handoff: lake length
matches the last barrier`. Watermark at close-out: `head=ff999684…, clean=True,
lake_bytes=626,459,331`; cockpit answering HTTP 200 on :8137 (liveness = the service's own
surface, never launcher-log mtime).

## 2. The live cutover, measured on the lake itself (read-only walker, never a BinLog —
its tail-repair writes)

Full-file frame walk, byte-accounted to a clean end both runs (walked bytes == file size at
open — the walker's own positive control):

| tag | records | bytes | mean B |
|---|---|---|---|
| 0x01 _T_FACT | 22,859 | 8,454,396 | 370 |
| 0x03 _T_GROWTH (legacy) | 86,095 | 597,532,595 | 6,940 |
| 0x04 _T_EPISODE | 60,502 | 6,658,790 | 110 |
| 0x05 _T_SELFBUILD | 665 | 99,549 | 150 |
| 0x07 _T_STATE | 9 | 262,822 | 29,202 |
| 0x08 _T_GROWTH_DELTA | 2,620 | 12,546,198 | 4,789 |

Two censuses ~40 s apart: **+14 chain records, +0 legacy** (0x03 frozen at 86,095 since the
T010-ii cutover at head `47eaece7`) — the sharpest live proof the flush path switched.

Honest accounting of the savings: member-granularity deltas run **31% under** legacy
(mean 4,789 vs 6,940 B; would-have-cost ≈18.2 MB vs 12.5 MB actual), NOT the ~96% the unit
pathology test shows — because the live engine genuinely rewrites `W/b/beta` on every refit,
and new weights are new information the chain rightly carries. What the chain kills outright
is the byte-identical repeat (writes NOTHING, tested); what buys the deeper live reduction is
per-synapse granularity — `_T_SYNDELTA` 0x09 (T010a), wired at the kernel block.

## 3. The 0x08 append baseline, measured as promised at T010-ii

Scratch-brain method (T010-i verbatim: `make_brain(seed=0)` × 120 ticks, wrapped
`BinLog.append`+`extend`, p95 over nonzero per-tick sums): **0x08 = 2,902 B/tick p95**,
0x04 = 113, 0x07 = 12,381; **zero legacy-growth ticks**; even the close-flush chains (666 B
delta). `tools/lake_append_baseline.json` updated; 0x03's ceiling CARRIED not re-measured
(the tag is closed to npz writes — a 0x03 flood is a regression the budget should still
catch). 0x09 stays unbaselined until a live writer exists (honest: a budget nobody measured
is not a budget).

## 4. T001's ratchet, flipped to a pinned-host pre-push assert

- `perf_baseline --assert` now DECLINES JURISDICTION off the pinned host (exit-0 SKIPPED
  notice, never a verdict — cross-host comparison is noise judging noise). CI row stays
  report-only permanently.
- Tracked hook `tools/hooks/pre-push` (canonical), installed per-host by copy to
  `.git/hooks/pre-push` — NOT `core.hooksPath`, which would silently disable the graphify
  post-commit/post-checkout hooks already living there. Foreign clones without the gate env
  skip loudly.
- **The flip's own first assert was a FALSE RED and the fix is recorded in the gate**:
  same binary, three back-to-back runs beside the live daemon measured wake p95
  118 → 131 → 205 ms (p50 stable 103–111). Timing interference is one-sided (only ever adds
  time), so the assert takes the best of ≤3 runs — pass on first green, red only when every
  attempt regresses. A real regression moves every run, so the red stays reachable
  (`--self-check`'s halved-baseline arm constructs it). Post-fix assert: green, first
  attempt (p50 109.3/p95 128.6 vs 107.6/121.0).

## 5. Exemption re-evaluations (the corrections promised at 0d0a7745)

T012 landed as **verify + flip** — the live lake deliberately stays BinLog-resident and no
journaller cadence was wired. Accordingly `lake_bridge / lake_engine / lake_migrate /
lake_index / lake_segments / signal_ring` dormancy reasons were re-anchored to their ONE
named consumer, the kernel block's two-phase tick and paging tier (T013+), **bound by S0
exit (T039) and not movable again** — a third extension is a defect, not a correction.
