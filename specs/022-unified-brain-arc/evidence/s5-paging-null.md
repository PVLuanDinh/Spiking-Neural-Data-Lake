# S5 / T068 — the paging null clause (SC-007), pre-registered with its metric chosen first (2026-09-08)

> Plan row (PLAN-FINAL-2026-09-02.md, Phase 3): **S5 = T068 (SC-007)** the paging null clause at the
> derived comparison width vs forced-total residency — WIN keep, LOSS delete the layer; gated on T061
> (closed) only.
>
> Task row (tasks.md T068): Paging null-clause measurement at a derived COMPARISON width just under the
> host-RAM ceiling (where both arms genuinely run — forced-total residency cannot execute at the >RAM
> binding width, which remains the deployment regime with its own red); WIN → keep, LOSS → delete the
> layer and record the deletion (both outcomes close this task).
>
> SC-007 (spec.md): At the derived binding width, paging meets its integer rounds budget and beats
> forced-total residency — or the layer is deleted and the deletion recorded (both outcomes satisfy
> this criterion; silence satisfies nothing). Red S-R3 (Part F): paging loses to forced residency at
> the comparison width; control: forced residency. INVENTION_REGISTER row 1: CONTROL forced-total
> residency at a derived comparison width under the host-RAM ceiling; DISCLOSURE the comparison width
> is NOT the deployment width (>RAM), stated wherever the win is claimed; NULL paging loses at the
> comparison width → the layer is DELETED.

## 1. Territory, measured before the run (HEAD after T067's push)

- **No governing document names the metric "beats" refers to.** SC-007, FR-010/FR-011, US3, quickstart
  §S2, the S5 bullet and S-R3 all say beats / loses to with no quantity (84-fact reader survey, every
  sentence quoted). Choosing the metric is this row's own act, and it is made HERE, before the run,
  with the numbers already seen stated in the next bullet so the choice cannot be read as made after
  them.
- **What has already been measured, and it points two ways.** T066's width arm (32,768 units,
  saturating drive, five cells): on EXACT resident edge bytes the pager TIES the twin (1.000, evictions
  0 in every cell); on wall-clock tick time the pager is SLOWER in every cell (1.70–3.47× contended;
  2.61× in a quiet pilot; 3.59× at 4,096 units). T066's own panel controls at 16,384 units: under
  working-set-limited drive resident edge bytes fell to 0.250, under transient drive to 0.164. T067's
  reorder cut reads per round 4.00→1.00 on a co-fetched index-distant stream. So: a criterion on tick
  time alone pre-registers a near-certain LOSS; on resident edge bytes under saturating drive a TIE
  (silence); on resident edge bytes under working-set-limited drive a likely WIN. **The metric is
  therefore chosen from what the layer EXISTS to do, not from which number looks best**: a paging
  layer exists so the substrate can run where forced residency CANNOT — its purpose is core memory,
  and its cost is time. The primary is memory; time is a bounded secondary; feasibility at the
  ceiling is the deployment claim, tested directly.
- **The exact instrument understates real core by ~23×.** `resident_edge_bytes` counts 16 B/edge (two
  int64 arrays per bundle); the live representation measured this session costs **374 B/edge**
  (bundle dict, tuple keys, ndarray headers, the `by_pre` index): 11,982 B/unit at 32 edges/unit,
  consistent with the archived cells' filtered tracemalloc (393 B/edge). A comparison width derived
  from 16 B/edge would put the forced-total arm ~23× over the ceiling and it would not execute. **The
  primary metric is therefore the whole-process footprint** (peak RSS of the arm's process, and traced
  total as the second instrument that fails differently), with exact resident edge bytes PRINTED beside
  it as the mechanism's own reading — never the criterion.
- **T065's binding width is the wrong quantity for this row and its figures are withdrawn**
  (`271 + 8·m`, the grower's blob, in memory, 11-unit top end; budgets hard-coded 64/256/1024 MB —
  no landed instrument has ever computed a width against host RAM). This row derives its own.
- **The host, measured:** `LUAN_AND_ANNIE` (the pinned baseline host), 31.79 GB total; with the live
  daemon UP (~5.1 GB working set) and the battery quiet, available RAM is measured at run time and
  RECORDED — the width is a function of that state. **The daemon is not paused for the verdict** (P4c:
  killing it stops the brain; the only precedent is a disclosed suspend A/B for a timing control): the
  comparison width is derived against AVAILABLE RAM with the daemon up and the battery quiet — the
  honest deployment ambient. A daemon-paused sensitivity arm, if run, runs last, resumes immediately,
  and never decides.
- **Timing on a contended host owes the ambient-ratio control** `perf_baseline` already carries (a
  crc32 reference sample beside every tick, the RATIO reported, best-of-N, the pinned host); the T066
  probe used none of it and said so. Every tick number here is a RATIO to the reference, best of 3,
  with the host state recorded beside it, and the tracemalloc pass and the timing pass stay separate
  (tracemalloc inflates the paged arm 5.8× and the twin 2.3× — a single pass would manufacture the
  loss).
- **Feasibility bounds the design.** T066's five cells at 32,768 units cost 3 h 44 m; a cell at
  ~330,000 units projects to 2.5–4 h paged, and an S = 8 sweep multiplies by eight. So the four
  regime cells run at an AFFORDABLE width with S = 8 seeds (Part F's bar), and the "just under the
  host-RAM ceiling" clause is discharged by a separate FEASIBILITY arm at the ceiling with one seed
  (a construction fact, not a statistic) — the drive, not the width, is what creates any difference
  (1.000 saturating vs 0.250/0.164 working-set-limited, measured).
- **The deletion inventory is written BEFORE the run** (2.6), so a LOSS does not get negotiated after
  the number is known.
- **Two wording debts, discharged in this row's first commit rather than argued about:** `spec.md`'s
  SC-007 says *binding* width while the task row, INVENTION_REGISTER row 1 and S-R3 say *comparison*
  width — a dated amendment to SC-007 states the comparison width under the ceiling as the verdict's
  width and the >RAM deployment width as arm F's and S-R5's; and `tasks.md` names T068's evidence as
  `evidence/s2-paging.md`, a pre-S-numbering path — this file is `evidence/s5-paging-null.md` (S3 →
  `s3-neuron-pager.md`, S4 → `s4-lake-compactor.md`), and the tasks row is amended to it.
- **Not swept, disclosed:** the pinned hot set is EMPTY in every cell (T066's width arm ran
  `pinned=frozenset()` too; a pinned set is counted inside `capacity_blocks`); `p_parts` is fixed at
  4 everywhere measured, and it multiplies the un-pageable resident floor (21 + 16·P + 1 B/unit), so a
  larger P makes paging structurally LESS able to win — a P sweep is S6's business, not this null's.

## 2. Pre-registration (written before the run)

### 2.1 The arms

Both arms share the edge set, the drive, the seed, `frac_bits`, `P`, delays 1..8 and the tick count,
built as the T066 probe built them. **TWIN** = `KernelTick` over the full `Synapses` (forced-total
residency; no pager code). **PAGED** = `UnitBlockStore` + `Synapses.empty` + `NeuronPager` at capacity
`c` (an INPUT stated per cell), budget derived by `_derive_rounds` on the cell's own store, `max_stale_ticks
= 2`, no pinned set; T067's `co_fetch_order` reorder applied in the REORDERED cells only.

### 2.2 The regimes (four, each a cell family; every cell S = 8 seeds, seeds 0..7)

R1 SATURATING drive (~99 % of units fire every tick — T066's width fixture); R2 WORKING-SET-LIMITED
drive (a fixed subset of blocks driven, the activated working set ≤ ¼ of the graph); R3 TRANSIENT drive
(tick 0 only); R4 CO-FETCHED index-distant groups, before and after T067's reorder. Width for R1–R4:
**16,384 units × 32 edges** (cells measured at minutes each; 8 seeds × 4 regimes × 2 arms ≈ 64 cells + 8
reordered ≈ 5–7 h), capacity 0.25 of blocks. FEASIBILITY arm F: at the derived ceiling width `W*` (the
largest width at which the twin's measured footprint fits available RAM with the daemon up and the
battery quiet, derived from the 374 B/edge + floor regression re-measured on this host at run time)
the twin CONSTRUCTS and ticks; at `2·W*` the twin FAILS to construct (MemoryError or refusal, recorded);
the paged arm at capacity 0.10 constructs and ticks at BOTH — one seed, 50 ticks, the construction
outcomes and footprints recorded. At `W*` and above the store exceeds the OS page cache, so the paged
arm's reads are DISK reads and `_derive_rounds`' cache-warming disclosure changes character — arm F's
tick ratio is REPORTED with that caveat and never decides. SENSITIVITY cell R2-refrac: R2 at one seed
with `refrac_ticks = 10 > max_delay = 8` (T066's third post-green defect's regime, never run at width —
every archived cell ran `REFRAC_TICKS = 0`), reported beside R2 and never deciding; a paged/twin
byte-identity check runs in that cell because it is the regime the defect lived in.

### 2.3 The metric, the verdict rule, and the reds (written first)

**Primary (memory):** peak process RSS of the arm (measured by a subprocess-level monitor of the
cell's own process) and traced total (tracemalloc, whole process) as the second instrument; both
reported as paged / twin ratios per seed; exact resident edge bytes printed beside them. **Secondary
(time):** mean tick time as the RATIO to the crc32 reference sample, best of 3, paged / twin per seed.
**Tertiary (reads):** reads per round, from T067's instrument, before and after reorder in R4.

**Verdict rule (anti-rigging: one primary, bounded secondary, feasibility as the deployment claim):**
- A regime is a WIN when the paired one-sided t over S = 8 of (twin − paged) primary ratio-to-one
  clears 1.895 at 7 d.f. with the exact sign-flip p ≤ 0.05 (ties within 1 % folded, DEGENERATE below
  5 non-tied; the permutation false-positive rate printed), on BOTH memory instruments, AND the
  secondary ratio's median does not exceed **3.0×** (the cost bound, chosen from the measured 1.7–3.6×
  band BEFORE this run so that a cost at the band's top is still within it — disclosed as chosen).
- The layer's verdict: **WIN** iff R2 (the regime it exists for) is a WIN AND arm F shows the paged
  arm executes at `2·W*` where the twin cannot; **LOSS** iff R2 is not a WIN OR arm F's paged arm
  cannot execute where the twin cannot. R1's expected TIE-or-loss is recorded as the layer's regime
  bound, never smoothed; R3 and R4 are reported, and R4's reorder must reproduce T067's reads-per-round
  reduction (a control on T067, not a verdict input).
- **Reds:** S-R3 fires on LOSS (the pre-registered red). The instrument reds: a NEUTERED paged arm
  (capacity = all blocks, no eviction) must read ratio 1.000 on both memory instruments (the harness
  can see "no saving"); the twin must construct at `W*` and fail at `2·W*` (the feasibility instrument
  discriminates); the ambient-ratio control must print a reference ratio within 20 % across the two
  arms' runs (else the timing arm is refused as contended and re-run); every paged cell calls BOTH
  exact residency instruments (`resident_edge_bytes` and `resident_edge_bytes_via_routes`) and asserts
  byte agreement with `dangling == 0` — the module's own two-instrument rule, which the archived T066
  probe did not apply (it read one).

### 2.4 The control that is not the method

The twin contains no paging code. The neutered paged arm (capacity = all blocks) is the method with
its eviction removed. The transient regime (R3) is a positive control for the memory instruments (the
largest expected saving, 0.164 at 16,384 units) — if R3 does not read a saving, the instrument is
blind and no verdict is reported.

### 2.5 Attribution and disclosure

Per-item: eviction neutered → memory ratio 1.000 (R2); reorder neutered (identity order) → reads per
round unchanged (R4); the daemon-paused sensitivity arm, if run, printed beside the verdict and never
deciding. Disclosed: the comparison width is NOT the deployment width; `W*` is a function of host state
and the state is recorded; tick ratios are contended-host measurements corrected by a reference, not
benchmarks; resident edge bytes are the mechanism's own count and understate core ~23×; the cost bound
3.0× was chosen from the seen band.

### 2.6 Disposition on a null (fixed in advance)

| outcome | disposition |
|---|---|
| LOSS | the layer is DELETED: `neuron_pager.py`, its tests, its REGIONS/CORE rows and `wire` EXEMPT row, INVENTION_REGISTER row 1 marked DELETED ON ITS NULL, S3/S5 annotations, S-R1/2/3/5, quickstart §S2 item 3, SYNTHESIS §6 L2; `lake_compactor.py`'s pager-dependent halves (`co_fetch_order` over `plan()`, `_paged` fixtures, reads per round) go with it and its merge/reorder over the store are RE-HOMED as record maintenance without a pager consumer or DELETED per T064's re-armed §2.3 clause — decided here: deleted, unless T067's merge has a non-pager consumer by then (it does not); `unit_blocks` goes with them (T064's deletion condition re-arms). RE-HOMED, not deleted (load-bearing independently of paging): `compile_bundles` as the one compiler, `Synapses.empty`, `read_records`, `located_records`, the ADOPT-record fence and the tag-aware rebuild, the 0x0C/0x0D record classes with their retention rows (records already written are IMMORTAL/COMPACTABLE and stay), `observe_append`, KernelTick's `resident=None` path (the mask branch is deleted with the pager). `tests/test_lake_compactor.py` imports `NeuronPager` and `UnitBlockStore` at module top and builds a pager in its `_paged()` helper, so the whole T067 suite fails to COLLECT without the pager — a LOSS is not a file deletion: the pager-free arms (the record classes, the fence, the rebuild, `read_records`) are re-homed into a pager-free test module in the same stacked commit series, and the pager-dependent arms go with the pager. The T067 panel's C1 finding is inherited as a fact about any survivor: a merge undoes a reorder's placement. The evidence files survive the code. |
| WIN | the layer stays; S-R5 (S6) inherits the R1 regime bound and the 3.0× cost as stated limits; the deployment width claim is scoped to arm F's `2·W*` observation |
| R3 reads no saving | instruments blind — no verdict; fix the instrument first |
| the reference ratio differs > 20 % across arms | the timing arm is contended — re-run quiet; never read |
| arm F's twin constructs at `2·W*` | `W*` was mis-derived — re-derive, re-run F; no verdict until F discriminates |

### 2.7 What this row does NOT claim

No claim about widths above `2·W*`; no tick-time benchmark; no claim that the co-fetch order is the
right locality signal; no live-tick claim (S6). It claims: a metric chosen before the run for the
layer's purpose, a verdict rule that cannot be satisfied by silence or by picking the friendliest
number, and a deletion inventory written before the number is known.

### 2.8 Commit order

1. this file, with the SC-007 dated amendment in `spec.md` and the tasks row's evidence path — 2. the
harness (an archived session probe under `memory/archive/t068-<date>/`, cited by short path, plus
in-tree: the two instrument reds in `tests/` where they touch tree code) — 3. the
cells, one subprocess each, run.log with the host state — 4. the record (§3) and the verdict's
disposition applied (a WIN's records, or a LOSS's deletion as its own stacked commits) — panel — fold —
ONE battery — closing record — push.

### 2.9 Amendments BEFORE the campaign (2026-09-08, after the harness's instrument reds; numbers seen stated beside each)

The harness was built as 2.8 orders (an archived probe, `memory/archive/t068-2026-09-08/`) and its
instrument reds were run at 4,096 and 16,384 units BEFORE any campaign cell. Three of them fired on THIS
pre-registration's own apparatus — not on the mechanism — and each is amended here, dated, with the
reading that forced it. No verdict number has been seen: every cell run so far is a neutered, control
or pilot cell, and its numbers are quoted in full so the amendments cannot be read as made after them.

**A1 — the memory instruments (2.3), fired: the NEUTERED arm did not read 1.000.** Measured: peak
working set 1.0431 / 1.0528 (4,096, R1 / R2) and 0.9767 (16,384, R1); traced peak 0.8808 / 0.8804 /
0.8830 — while the arm's EXACT resident edge bytes equalled the twin's to the byte (2,097,152 and
8,388,608), evictions 0, fully resident. Two instrument defects, both of the "a replacement instrument
reads a different quantity" shape the T066 panel already convicted once: (i) `tracemalloc`'s PEAK is a
compile-path transient — the twin compiles the whole table in ONE `compile_bundles` call whose buffers
set the high-water mark, the paged arm compiles per block — so the constant −12 % is construction, not
residency; (ii) `peak_wset` includes the same construction transient on the twin's side and the store's
roughly constant machinery on the paged side, so its zero point MOVES with width and crosses 1.000
between 4,096 and 16,384 (+4.3 % → −2.3 %) — a verdict at one width would read that crossing as a
saving. AMENDED: both instruments become STEADY-STATE readings — after every tick the process's current
working set (`psutil` `wset`) and the current traced total (`tracemalloc.get_traced_memory()[0]`) are
sampled, and the cell reports the MAX over ticks of each, construction excluded. The neutered red is
re-specified as a calibration that can fail rather than a physical identity the two representations do
not share: (a) exact edge bytes ratio 1.000 (the arm is the twin's graph); (b) the neutered arm's
steady-state ratio is ≥ 0.99 on BOTH instruments — a "saving" with no eviction and no lazy loading is
the instrument lying; (c) two neutered runs at the same width and seed agree within 1 % (repeatability);
and the neutered ratio is REPORTED per width as the layer's zero point (its machinery's cost with no
eviction). Pre-stated branch: if (b) fails on the traced instrument under steady-state sampling, the
traced instrument is reading REPRESENTATION (the twin double-holds the raw edge arrays beside its
bundles) and is DROPPED as a verdict instrument, replaced by committed private bytes (`psutil`
`private`) as the second instrument that fails differently from the working set; the drop is recorded
with the number. The WIN rule is unchanged: paged / twin < 1 on both verdict instruments, paired over
S = 8 seeds, on the layer's whole-process cost — the machinery is what deployment pays, so it stays
inside the ratio.

**A2 — the regimes' topology and parameters (2.1, 2.2), fired: R2 read exact ratio 1.0000, not
0.250, with evictions 0 in EVERY cell of EVERY regime at both widths.** Cause, measured: 2.1 fixed
delays 1..8 and `max_stale_ticks = 2`, and 2.2 named a DRIVE per regime but no TOPOLOGY; under T066's
random-post edge set any driven subset activates the whole graph, and under an 8-tick ring window the
ring-safety clause is false for essentially every block — T066 §3.5's own finding, reproduced. The
panel controls whose 0.250 / 0.164 this row cites ran a LOCAL topology (post inside the pre's own block
or the next) with `max_delay = 3`, `max_stale_ticks = 1`, `blocks_per_round = 16`, `max_rounds = 8`,
`pinned = {0}`. AMENDED: topology and the pager parameters are recorded INPUTS per cell, defaulted per
regime to the fixture whose number this row cites — R1 keeps T066's width fixture exactly (random
posts, delays 1..8, `max_stale_ticks = 2`, no pinned set: it is the regime bound and must reproduce
T066's 1.000); R2, R3 and R4 take the panel's local fixture and parameters exactly as listed (pinned
{0} included, overriding 2.1's "no pinned set" for those three, disclosed), and R2 must reproduce
0.250 ± 0.02 on the exact instrument as the fixture's check before its cells count. The tick count for
R1–R4 is 20 (the panel's; 2.2 omitted it); arm F's 50 stands. R1's eviction neuter is disclosed
VACUOUS (eviction never fires under R1 in either arm — the neutered and incumbent arms read
0.9767 / 0.9765, identical); the per-item attribution in 2.5 is carried by R2, where the neuter must
return the paged ratio to the zero point of A1. The neutered red runs in R2 (the verdict regime) and
R1 (the bound), both at 16,384.

**A3 — R4's fixture (2.2), fired: the reorder refused all 32 blocks** (`block 31 carries the edge
(2, 61, 7) twice -- the delta tier addresses an edge by (pre_local, post, delay)`), reads per round
4.0 → 4.0, `moved = 0`. T066's edge builder draws random posts × random delays and collides inside
every block; T067's own fixture is duplicate-free by construction because the delta tier needs it. The
order function is correct (the R4 due sets are 8 groups of 4 index-distant blocks; `co_fetch_order`
places them with mean contiguous runs 4.000 → 1.000). AMENDED: the R2–R4 edge sets are generated
duplicate-free in `(pre_local, post, delay)` (rejection on collision), as T067's fixture is; R1 keeps
T066's builder unchanged. The probe now records `reorder_refusal_last` and `reorder_had_effect` so a
whole-store refusal can never again read as "the reorder did not help".

**Disclosed, not amended.** The ambient-ratio control's 20 % refusal is REACHABLE (a 28-core hog
drifts the reference 61.1 %), but a 4-core hog on this 28-thread host drifts it only 13.1 % and does
not refuse — the control is coarse below roughly a fifth of the machine, stated here. Quiet drift
3.83 % across four runs; cross-arm 0.93 %. One R1 timing cell at 16,384 read a secondary of 3.135×
(296.9 vs 923.8 ms per tick) — above the 3.0× bound; R1 is the bound regime, not the verdict's, and
the bound is NOT moved. The paged arm's steady-state working set still carries the store's machinery
(sqlite index, Arrow segments) — that is the zero point A1 reports, and it is deployment's cost.
`verdict.py` applies the 1.895 critical value as written and prints a `df_note` when tie-folding
leaves fewer than 7 d.f. Spike trains were byte-identical between arms in every 4,096 pilot cell and
NOT at 16,384 under R1 (T066's forced-round deferral at tick 2, reproduced) — the byte-identity claim
is T066's under calm drive and is not re-claimed here.

**What the amendments cannot do.** They do not change the metric (whole-process footprint, primary),
the verdict rule, the S = 8 paired test, the 3.0× cost bound, arm F, or the disposition table. They
change how two instruments sample and which fixture each regime runs — the fixtures whose numbers
this row cited from the start.
