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

### 2.10 Second amendment BEFORE the campaign (2026-09-08, after the amended reds; numbers stated beside)

The amended reds were re-run at 16,384 units, seed 0. **(a) the neutered calibration PASSES** under
steady-state sampling: exact ratio 1.0000 in R1 and R2; zero points R1 1.0289 footprint / 1.0010
traced, R2 1.0588 / 1.0026 (all ≥ 0.99); repeatability 0.56 % / 0.005 % / 0.002 % / 0.018 % — and the
RETIRED peak readings in the same cells still fail (0.9724 / 0.8830), so A1 changed the instrument, not
the luck. The pre-stated drop branch did NOT apply (traced ≥ 1.000); private bytes was measured anyway
and would have been unusable (repeatability 7.39 % / 5.40 %; no saving in R3, 1.106) — recorded so the
branch cannot be re-proposed. **(c) R3 PASSES** (footprint 0.5048, traced 0.1757, exact 0.164062).
**(d) R4 PASSES** (refusals 0, moved 128 of 128, reads per round 3.95 → 1.00; the "before" is 3.95, not
4.00, because one due set carries an incidental adjacency — reported as read). The worker also found a
manufactured zero in its own probe (`reorder` returns `merged`, the probe read `moved`) and fixed it.

**(b) and (e) FAIL, on a fixture truth measured with the module's own `_TripCensus`:** R2 at capacity
0.25 reads exact ratio **1.0000** (page-ins 127 of 128, evictions 0, over capacity 19 of 20 ticks), and
the eviction neuter cannot move (paged 1.0502 vs neutered 1.0588 footprint, evictions 0 in both).
189/189, 741/741 and 636/636 eviction refusals are the RING-SAFETY clause: under `panel_local` the
recurrent weight (1.5) exceeds the threshold (1.0), so a block once lit re-fires itself forever and
nothing lit ever goes quiet. Consequence, and it reaches back into §1: **the exact ratio under every
fixture so far is the ACTIVATED-BLOCK FRACTION** (R3 21/128 = 0.164, R4 80/128, R2 128/128) — a
lazy-loading count, never a capacity effect — and the panel's 0.250 that §1 cited as "working-set-
limited" evidence was the same fraction. This row misread that number as evidence of eviction; it was
evidence of blocks never loaded. Stated here rather than smoothed.

**A4 — R2's drive (2.2), amended to REALIZE the condition the regime was named for.** R2 is defined as
"the activated working set ≤ ¼ of the graph"; a fixed driven subset with self-sustaining recurrence
never satisfies it (the set grows to the graph). AMENDED: R2 is a SLIDING working set — each tick a
contiguous window of 32 blocks (¼ of 128) receives the external drive, the window slides 4 blocks per
tick, 20 ticks (coverage 32 + 19·4 = 108 of 128 blocks, the lazy-loading floor 108/128 = 0.844); the
R2 fixture's recurrent weight is 0.6 < threshold 1.0 (a unit fires only under drive or two coincident
inputs), so a block the window has left falls silent inside the 3-tick ring window and becomes
evictable; topology `panel_local`, duplicate-free, the panel's pager parameters. The R2 FIXTURE CHECK
becomes: the paged arm at capacity 0.25 must EVICT (evictions > 0) and read an exact ratio ≤ 0.50 —
well below the 0.844 coverage floor, so a saving that clears the check is attributable to eviction by
construction; the neutered arm (capacity = all) must read the coverage floor (≈ 0.844 exact) as R2's
lazy-loading zero point. If the check fails, the campaign does not start and the failure is recorded
as a third fixture red — no further tuning without a dated amendment quoting its numbers.

**The verdict rule gains a STRICTER conjunct (added before any campaign number; it can only make a WIN
harder).** R2's WIN now also requires **paged < neutered** on the primary instrument by the same paired
one-sided test over S = 8 — the invented residency policy (eviction) must charge beyond what lazy
loading alone buys. Three outcomes replace two: **WIN** (R2 paged < twin AND paged < neutered on both
verdict instruments, secondary ≤ 3.0×, arm F discriminates) → the layer stays; **STORE WINS, POLICY
NULL** (paged < twin but NOT paged < neutered: the saving is lazy loading's) → the eviction policy is
DELETED ON ITS NULL (P8 clause 5: `NeuronPager`'s capacity parameter, `_evictable`, the demote path and
the rounds-vs-capacity accounting go; the pager becomes the lazy page-in over the block store;
INVENTION_REGISTER row 1's residency-policy half is marked DELETED ON ITS NULL; the store, `read_records`,
the delta tier, the fence and the reorder stay as measured winners; S-R1/S-R2's eviction-shaped reds
are retired with the policy and the drop red S-R2 is re-homed to the lazy loader if it still applies);
**LOSS** (NOT paged < twin, or arm F's paged arm cannot run where the twin cannot) → 2.6's deletion
inventory applies in full. R1 stays the regime bound (its neuter is vacuous under saturating drive and
is reported as such); R3 and R4 are reported, R4's reorder control passed.

**Not changed.** The metric, the instruments as amended in A1, S = 8, the 3.0× cost bound (the one R1
timing cell at 3.135× is still not a verdict), arm F, the disposition on LOSS, and the tick count. The
lazy-loading floor is stated as a floor, not claimed as the policy's. *Appended 2026-09-09, after the panel (3.7 item 13): T066's record stated the attribution in the same sentence these numbers were taken from (`s3-neuron-pager.md`: "— from never LOADING, not from eviction"); §1's citation dropped that clause, so this was a dropped disclaimer at the point of citation rather than an ambiguous number, and A2's "R2 must reproduce 0.250" was built on the dropped half. What this row measured for itself is the wider fact: the exact ratio equals the activated-block fraction under every fixture.*

### 2.11 Third amendment BEFORE the campaign (2026-09-08; the A4 check's numbers stated beside)

**A4's check FAILED, twice, identically:** the paged arm at capacity 0.25 read exact ratio **0.843750**
(7,077,888 / 8,388,608) against the required ≤ 0.50 with **evictions 0**; every one of the 1,273
candidates at the 19 capacity trips was refused on the ring-safety clause (`_TripCensus`), because the
activated-block count climbed monotonically 32 → 107 of 128 and no block the window left ever fell
silent. The neutered arm read the coverage floor **exactly, 0.843750 = 108/128**, in both runs (that
conjunct met), and paged / neutered = 1.000000. R3 and R4 unchanged to the digit (0.164062; refusals 0,
moved 128, 3.95 → 1.00). **A4's premise was wrong and is withdrawn:** "recurrent weight 0.6 < threshold
1.0 silences an abandoned block" ignored that the membrane INTEGRATES — at decay 0.5 the sustained gain
is 2×, and each post carries ~16 own-block in-edges, so a lit block is self-sustaining far below the
per-edge threshold. The worker's positive control in the same harness at weight 0.1: **63 evictions,
exact 0.351562 against its own neutered 0.843750** — the eviction path charges once activity dies. The
regime the pager exists for (a working set that MOVES and whose abandoned regions GO QUIET) is
realizable on this substrate; a hand-picked recurrence constant is what failed, twice. Per P1 a picked
constant is replaced by a derivation. Disclosed: these check cells necessarily produced R2's first
paged / twin ratios at seed 0 under the FAILED fixture (0.862–0.864 footprint, 0.821 traced, the whole
saving present in the neutered arm too) — the "STORE WINS, POLICY NULL" shape at one seed of a fixture
that did not realize the regime; no verdict is read from it, and it is quoted so it cannot be read as
unseen.

**A5 — R2's recurrence is DERIVED by a silence probe, not picked.** On the R2 fixture at 16,384 (sliding
window 32, slide 4, `panel_local`, duplicate-free, the panel's pager parameters), for the recurrent
weight w over the descending grid {0.6, 0.4, 0.3, 0.2, 0.15, 0.1}: drive ONE 32-block window for 5
ticks, remove the drive, and count the ticks until no unit in those blocks fires (the block's SILENCE
TIME). R2's w is the LARGEST grid value whose silence time is ≤ 3 ticks — the ring window `max_delay`,
the clause that must clear for eviction — and the whole table is printed and recorded in every R2 cell's
JSON as the derivation. The fixture check of §2.10 is unchanged (evictions > 0; exact ≤ 0.50; the
neutered arm at the 0.844 floor) and must pass under the derived w before any cell counts. Pre-stated
branch: if NO grid value silences within 3 ticks, the regime is UNREALIZABLE on this substrate's
membrane and the campaign runs without R2's eviction conjunct being reachable — the verdict is then
"STORE WINS, POLICY NULL" or LOSS by construction (eviction cannot charge in any realizable regime),
and it is recorded that way, never as a WIN.

**`verdict.py` is brought up to §2.10 before the campaign** (three outcomes; the paged < neutered
conjunct by the same paired test; the R2-refrac and R1-bound rows reported, never deciding), and its
four synthetic self-test sets become five (WIN, STORE-WINS/POLICY-NULL, LOSS, DEGENERATE, R3-blind).

**Not changed:** everything §2.10 lists as unchanged, and §2.10's three outcomes and their dispositions.
Three fixture reds in a row are the instrument reds doing their job before the number is known; each
is dated here with the reading that forced it, and the campaign has not started.

## 3. Results (on the stacked head, before the panel)

### 3.1 The harness, and the instrument reds' final state under the final fixture (`memory/archive/t068-2026-09-08/`)

Built as 2.8 orders: `t068_probe.py` (one cell per subprocess, `NORD_HOME` on a scratch dir, every input
recorded in the cell JSON — topology, pager parameters, drive, recurrence and its derivation, tick
count), `run_cells.py` (resumable driver, one cell at a time, host state per line in `run.log`),
`silence_probe.py` (A5's derivation), `derive_wstar.py` + `run_f.py` (arm F), `verdict.py` (§2.10's
three outcomes, a five-set synthetic self-test each with an asserted expected outcome, and four rule
mutations that each break it). The tree was not changed by the harness (the archived-probe precedent of
T066's width arm). Three instrument reds fired on this row's own apparatus before any campaign cell,
each amended and dated in 2.9–2.11; their FINAL readings under the final fixture, at 16,384 units × 32
edges, 128 blocks, seed 0:

| red (2.3, as amended) | reading | verdict |
|---|---|---|
| neutered calibration (a) identity | exact edge bytes ratio 1.0000 in R1 and R2 (8,388,608 B both arms) | MET |
| neutered calibration (b) no false saving, steady state | R1 1.0289 footprint / 1.0010 traced; R2 (fully resident form) 1.0588 / 1.0026 — the retired PEAK readings in the same cells still fail (0.9724 / 0.8830), so A1 changed the instrument, not the luck | MET |
| neutered calibration (c) repeatability | 0.56 % / 0.005 % / 0.002 % / 0.018 % | MET |
| the pre-stated drop branch | did NOT apply (traced ≥ 1.000); private bytes measured anyway and unusable (repeatability 7.39 % / 5.40 %, no saving in R3: 1.106) — recorded so it cannot be re-proposed | — |
| two exact residency instruments | `edge_bytes_table == edge_bytes_routes`, `dangling == 0`, in every paged cell (dozens) | MET |
| R3 positive control | footprint 0.5048, traced 0.1757, exact 0.164062 = 21 of 128 blocks ever loaded (the panel's 0.164 to three digits; evictions 0 — a lazy-loading saving) | MET |
| R4 reorder control | refusals 0, moved 128 of 128 (`is_permutation`), reads per round **3.95 → 1.00** (one due set carries an incidental adjacency; reported as read), bytes appended 6,292,992 | MET |
| ambient-ratio control | quiet drift 3.83 % over four runs, cross-arm 0.93 %; a 28-core hog drifts the reference 61.1 % (the red is reachable); a 4-core hog only 13.1 % on this 28-thread host (coarse below a fifth of the machine, disclosed) | MET |
| A5 silence probe (R2's recurrence derived) | w ∈ {0.6, 0.4, 0.3}: NEVER silent within the 20-tick cap (at 0.3 the abandoned window is still recruiting, 40 → 908 units); w = **0.2**: silent 1 tick after the drive stops (2 ticks at seeds 1–3; stays silent to the cap); 0.15 and 0.1: 1 tick. Known-positive (never driven) silent at tick 1 at all six weights; known-negative (drive sustained) never silent at any — the detector discriminates. **Derived w = 0.2**, the largest grid value inside the 3-tick ring window; the cliff sits between 0.3 and 0.2 where the sustained gain w/(1 − decay) = 2w times the realized coincidence stops clearing threshold 1.0 | derived |
| R2 fixture check (2.10), two runs | paged at capacity 0.25: exact **0.367188** (3,080,192 B), **61 evictions**, page-ins 107, 19 capacity trips, over capacity 19 of 20 ticks; the neutered arm reads the coverage floor **0.843750 = 108/128 exactly**, evictions 0; the neuter discriminates: paged / neutered 0.435 exact, 0.718 / 0.720 footprint, 0.483 traced; repeatability: exact identical, footprint ≤ 0.62 %, traced ≤ 0.002 %. `_TripCensus`: 19 trips, 750 refusals of which 748 are the ring-safety clause, 61 evictions (under A4's 0.6 the same census read 1,273 of 1,273 refusals, 0 evictions). The activated working set, MEASURED per tick: 32 … 34 of 128, peak 0.266 — marginally above R2's "≤ ¼" because the drive's leading edge recruits one or two blocks, recorded rather than rounded | MET, all four conjuncts |

Twin, paged and neutered spike trains were byte-identical in every check cell (`ca6db5c22801d9b6`),
so the twin's silence is the silence in the arm that evicts. The check cells necessarily produced R2's
paged / twin ratios at seed 0 under the campaign's fixture — 0.612 / 0.608 footprint, 0.395 traced —
before the campaign; they are quoted in 2.11 and here so that seed 0 cannot be read as unseen; A5's
rule, the check's conjuncts and `verdict.py`'s outcomes were fixed before they existed (`f61911c`).

One reading that is not a red: the neutered arm's steady-state footprint under the sliding drive is
0.845–0.853, below A1(b)'s "≥ 0.99" — A1(b) was written for a FULLY RESIDENT neutered arm (R1's form,
exact 1.000, where it reads 1.0289); under the sliding drive the neutered arm lazily loads 108 of 128
blocks and 2.10 re-specifies its reading as the coverage floor, which it hits to the digit.

Defects found in the harness by its own builders, each fixed before the campaign: `LakeCompactor.reorder`
returns `merged`/`merged_ids`, and the probe had read `moved` — a manufactured zero in every reordered
cell; `verdict.py`'s cell key omitted `refrac_ticks`, so the R2-refrac sensitivity cell would have
silently replaced the seed-0 R2 verdict cell; one of the builder's own rule mutations moved a symbol
the fixture builder shares and was itself defective, reported.

### 3.2 The campaign (226 cells, 16,384 units, S = 8, one cell at a time, the live daemon up)

Launched 23:10 UTC on 2026-09-08 after the A5 check, one cell per subprocess, `NORD_HOME` on a scratch
dir, every cell `rc 0` (226 of 226 by 00:13 UTC); the 24 R2 timing cells were then re-run pinned to the
fast core class (3.6, item 3) and the campaign's own unpinned R2 timing cells are kept beside them under
`campaign/refused_timing_unpinned/`. `verdict.py` over the cells (`campaign/verdict_pre_f.json`),
paired over S = 8 seeds, the one-sided t against 1.895 at 7 d.f., the exact sign-flip p over all 2^8
assignments (the permutation false-positive rate 0.047 at this S), ties within 1 % folded:

| regime | steady working set, paged / twin (median; t; p) | steady traced total (median; t; p) | vs NEUTERED (2.10), footprint / traced | secondary: tick cost paged / twin, RAW wall-clock (median; range) and reference-corrected | regime WIN |
|---|---|---|---|---|---|
| R1 saturating (the bound) | **1.028**; t −21.2; p 1.0 — the paged arm costs MORE (the zero point of A1) | 1.000134; DEGENERATE (0 non-tied of 8 under 2.3's 1 % tie fold — not identity: the paged arm reads LARGER on 8 of 8 seeds, +0.0076 % to +0.0245 %; the EXACT instrument is the one that ties to the byte, 8,388,608 B in both arms) | — (vacuous: evictions 0 in both arms) | **3.07×** (2.89–3.36); corrected 3.12× | no (no saving; cost at the bound's edge) |
| **R2 sliding working set (the verdict regime)** | **0.602**; t 203.8; p 0.0039 (8 of 8) | **0.399**; t 229.8; p 0.0039 | **0.709** (neutered / twin 0.847); t 107.8; p 0.0039 — and **0.488** (0.818); t 159.6; p 0.0039: **eviction charges beyond lazy loading on both instruments** | **35.2×** (34.3–37.6); corrected 26.0× — twin 0.55 ms per tick, paged 20 ms | **no — memory WIN on every conjunct, the cost conjunct FAILS by an order of magnitude** |
| R3 transient (the positive control) | 0.454; t 50.5; p 0.0039 | 0.184; t 11,089; p 0.0039 | — | 4.51× (4.05–4.62); corrected 4.27× | no (cost) |
| R4 co-fetched, reordered | 0.766; t 112.0; p 0.0039 | 0.628; t 6,465; p 0.0039 | — | 3.81× (3.68–3.93); corrected 3.84× | no (cost) |

Reported, never deciding: the R4 reorder control reproduced T067 on all 8 seeds (reads per round 3.95 →
1.00, refusals 0, moved 128 of 128, `had_effect`); the R2-refrac sensitivity cell (`refrac_ticks = 10 >
max_delay 3`, seed 0) read paged 156,123,136 vs twin 223,559,680 B (0.698) with 63 evictions and trains
byte-identical to the twin — T066's third defect's regime, clean at width; the ambient-ratio control
passed within 20 % in R1, R3 and R4 on every seed (drift 0.2–9.5 %) and REFUSED R2 on every seed (3.6,
item 3); the R3 positive control read a saving on both instruments (the instruments are not blind).
The raw wall-clock instrument's own control (corrected at the fold, 3.7 item 2): taking spread as (slowest run − fastest run) / best-of-3, the spread is ≤ 9 % in 57 of the 64 verdict-arm timing cells (64 of 72 including R2's neutered arm); the exceedances are R2 twin s2 36.2 %, R1 paged s2 17.7 %, R3 twin s3 14.2 %, R1 paged s3 12.4 %, R3 twin s5 11.5 %, R3 twin s2 10.6 %, R4 twin s3 10.0 % (R2 paged_neutered s6 9.3 % when the neutered arm is counted). The twin's best-of-3 raw tick spans across seeds R1 7.8 %, R2 6.2 %, R3 13.8 %, R4 6.3 %. The worst spread (1.36×, the R2 twin that is the 35.2× ratio's denominator) taken entirely against the ratio leaves 25.9×.

### 3.3 Arm F — the derived ceiling width

`derive_wstar.py` (R1, seed 0): twin peak working set at 2,048 / 4,096 / 8,192 / 16,384 units =
98.2 / 130.2 / 194.0 / 321.0 MB; fit bytes = 66,503,725 + 15,538.66 × units (485.6 B/edge at 32
edges/unit; max residual 0.05 %); available RAM with the daemon up 11.83 GB of 34.14 (decimal GB throughout this section; the harness's console labels GiB as "GB" — corrected at the fold, 3.7 item 5); budget 0.8 × available = 9.47 GB → **W* = 604,800 units** (predicted twin 9.46 GB), **2·W* = 1,209,600** (predicted 18.86 GB — above available, so by 2.2 the twin "must FAIL"). `run_f.py` (R1 drive — T066's width
fixture, the harness default; 2.2 left F's drive unstated; 50 ticks; capacity 0.10; one seed):

| cell | arm | units | constructed | wall | steady working set | available before → after |
|---|---|---|---|---|---|---|
| F1 | twin | 604,800 | **yes**, ticked 50 | 785 s | 10.03 GB (predicted 9.46: 5.9 % OVER, 10,025,857,024 vs 9,464,285,818 B) | 11.51 → 7.34 GB |
| F2 | twin | 1,209,600 | **yes — it did NOT fail**, ticked 50 | 1,780 s (2.27× F1 for 2× the units) | 17.70 GB (predicted 18.86: 6.1 % UNDER, 17,702,244,352 vs 18,862,067,912 B) | 14.81 → 7.18 GB |
| F3 | paged, capacity 0.10 (472 of 4,725 blocks) | 604,800 | **yes**, ticked 50 — page-ins 4,725 (every block), evictions 0, over capacity 48 of 50 ticks, rounds mean 11.8 (585 at tick 0), resident fraction 1.0 | 2,872 s (store build 153 s) | 9.57 GB (0.954 of the twin) | 20.78 → 8.71 GB |
| F4 | paged, capacity 0.10 (945 of 9,450) | 1,209,600 | **yes**, ticked 50 — page-ins 9,450 (every block), evictions 0, over capacity 48 of 50, rounds mean 23.6 (1,176 at tick 0), resident 1.0 | 6,443 s (build 674 s) | 17.55 GB (0.992 of the twin) | 16.36 → 6.79 GB |

Both paged cells are T066's width bound reproduced at 37× and 74× its width: under saturating drive the
pager evicts NOTHING (the ring-safety clause is false for every block), pages every block in over forced
rounds, and holds the twin's footprint — capacity 0.10 was honoured on 2 of 50 ticks. "Executes where
the twin cannot" was never reached because the twin executed everywhere F looked. Both exact residency
instruments agreed in every F cell (309,657,600 and 619,315,200 B, dangling 0). Per tick (one seed, one run, `mean_tick_ms_uncontrolled` — NOT the campaign's best-of-3 reference-corrected secondary; store build excluded, which the wall column includes): F1 14.9 s, F2 33.7 s, F3 53.8 s, F4 113.9 s — paged / resident twin at W* = 3.62×, paged / OS-paged twin at 2·W* = 3.38×. 2.2 pre-registered that arm F's tick ratio is REPORTED with the disk-read caveat and NEVER DECIDES; it enters no conjunct and is not read against the 3.0× bound. The successor row inherits 3.38× as its starting comparison, at one seed.

**Arm F did not discriminate, and the reason is the derivation's premise, not the twin.** 2.6 row 5
pre-stated it: *"arm F's twin constructs at 2·W* → W* was mis-derived — re-derive, re-run F; no verdict
until F discriminates."* W* was derived against AVAILABLE RAM; the twin at 2·W* ran with a working set 5.87 GB above the availability the derivation used (11.83 GB at 00:16 UTC; 2.89 GB above F2's own recorded start of 14.81 GB) because Windows trimmed other processes (the daemon among them — it
stayed up) and backed the rest with the pagefile. Forced-total residency's ceiling on this host is the
COMMIT limit (RAM + pagefile), not available RAM, and below it the OS supplies paging of its own at a
cost — F2's wall was 2.27× F1's for 2× the units, i.e. the OS-paged twin slowed by ~14 % per tick, not
by a wall. The pre-stated re-derivation (against the commit limit, tens of GB, a twin that thrashes
for hours before it fails) was NOT executed: F is a WIN conjunct only ("WIN iff R2 is a WIN AND F…"),
and the layer's verdict is already LOSS on R2's cost (3.4), which no F outcome can change; re-running
F could only decorate a LOSS, and it is recorded as not run rather than run for show (3.6). What F
leaves behind for the successor row: the deployment claim "runs where forced residency cannot" has NO
width on this host below the commit limit; the honest successor comparison is the OS-paged twin's
tick against the pager's at the same width above RAM — F2 and F4 are that pair, one seed, reported.

### 3.4 The verdict, in the pre-registration's words

2.3: *"A regime is a WIN when the paired one-sided t … clears 1.895 at 7 d.f. with the exact sign-flip
p ≤ 0.05 … on BOTH memory instruments, AND the secondary ratio's median does not exceed 3.0×."* 2.10:
*"R2's WIN now also requires paged < neutered on the primary instrument by the same paired one-sided
test."* 2.3: *"The layer's verdict: WIN iff R2 is a WIN AND arm F shows the paged arm executes at 2·W*
where the twin cannot; LOSS iff R2 is not a WIN OR arm F's paged arm cannot execute where the twin
cannot."*

| R2 conjunct | required | measured | verdict |
|---|---|---|---|
| paged beats twin, both instruments | t > 1.895, p ≤ 0.05, both | 0.602 (t 203.8, p 0.0039) and 0.399 (t 229.8, p 0.0039) | **MET** |
| paged beats neutered (2.10) | same test, primary (and traced reported) | 0.709 (t 107.8, p 0.0039); traced 0.488 (t 159.6, p 0.0039) | **MET** |
| eviction conjunct reachable (2.11) | the derived w realizes the regime | w = 0.2; 61 evictions per cell; the neuter discriminates | **MET** |
| secondary cost ≤ 3.0× | median paged / twin tick ratio | RAW 35.2× (34.3–37.6); reference-corrected 26.0× — the most pager-favourable reading is 8.7× over the bound | **NOT MET** |
| arm F discriminates | twin fails at 2·W*, paged runs at both | the twin CONSTRUCTED and ticked at 2·W* (17.7 GB working set; the OS trimmed and paged for it) — F does not discriminate on this host below the commit limit (3.3) | **NOT MET** (a WIN conjunct only; it cannot make or unmake the LOSS) |

**What the pre-registered machinery prints, literally** (`verdict.py --arm-f`, `campaign/verdict_final.json`):
`LAYER VERDICT: NO VERDICT`, on two refusal rows of 2.6 — row 4 (R2's timing arm: the reference level
differs more than 20 % across the arms; "re-run quiet, never read") and row 5 (arm F's twin constructed
at 2·W*; "re-derive, re-run F; no verdict until F discriminates"). The quiet re-run was executed (pinned,
both arms) and the drift persisted for a measured, arm-intrinsic reason (3.6 item 3); the F re-derivation
was not executed for the reason 3.3 states. **The session's ruling, put to the panel as its first
question and to Codex as an independent second model before anything is deleted:** the two refusals
guard the WIN path (a contended cost reading could hide a real cost; a non-discriminating F could
decorate a win) and neither can convert a cost measured at 35× on an instrument with a passing control
of its own into a reading under 3.0×; the LOSS below is therefore read under 2.3, with both deviations
quoted, and a panel finding that this is a relabelling REVERTS the disposition to NO VERDICT (instrument
fix + F re-derivation before any deletion). The deletion follows the panel, not this paragraph.

**R2 is NOT a WIN**: the memory claim the layer exists for is MET on every conjunct, on both instruments,
with eviction charging beyond lazy loading on all 8 seeds — and the bounded secondary is NOT MET by an
order of magnitude. Under 2.3, "LOSS iff R2 is not a WIN": **the layer's verdict is LOSS.** 2.10's
three-outcome restatement carried a drafting gap — its LOSS clause named only "NOT paged < twin, or arm
F", omitting the cost conjunct that 2.10 itself said was "not changed" and that WIN still required — so
this case fell in no 2.10 outcome as literally written; it is resolved to 2.3's rule (the stricter
reading, the one that pre-dates every number) and the gap is recorded here as the pre-registration's
own defect, not smoothed. Not a "STORE WINS, POLICY NULL": the policy charges (paged < neutered). Not a
WIN on any reading of the cost: the twin's tick under the sliding drive is 0.55 ms at 16,384 units and
the paged arm's is 20 ms — four page-ins per tick at ≈ 4.5 ms each (index lookup, range read, decode,
per-block compile), a cost the ratio method cannot hide and the bound cannot absorb.

What the LOSS is and is not. It IS the pre-registered finding that THIS pager, in the regime it exists
for, buys a 40 % / 60 % footprint saving at a 35× tick cost — "it was measured and it did not work" for
a residency layer whose cost bound was set at 3.0× before the run. It is NOT a finding that paging cannot
work: the cost is THIS implementation's (3.7 item 4): with page-ins and evictions ablated to zero the residency-mask branch's own tick is 2.5–3.4× the twin's (three runs: 2.485 / 3.046 / 3.388, trains byte-identical), and on the live paged arm at R2's resident fraction the paged `kt.tick()` alone is 3.2–6.0× — the mask branch consumes the whole 3.0× budget before any page-in; `pager.step()` is 91.8 % of the paged tick (compile ~76 %, read ~4 %). The successor does not inherit the mask branch (2.6 deletes it); what it inherits is this decomposition and the 3.38× OS-paged comparison of 3.3.

### 3.5 The disposition applied

2.6, row LOSS: *"the layer is DELETED"*, with the inventory written before the number was known
(`neuron_pager.py`, its tests, its REGIONS/CORE rows and `wire` EXEMPT row, INVENTION_REGISTER row 1
marked DELETED ON ITS NULL, the S3/S5 annotations, S-R1/2/3/5, quickstart §S2 item 3, SYNTHESIS §6 L2;
`lake_compactor.py`'s pager-dependent halves and — as decided there — its merge/reorder over the store
unless a non-pager consumer exists (it does not); `unit_blocks`; KernelTick's mask branch; the
pager-free arms re-homed FIRST: `compile_bundles` as the one compiler, `Synapses.empty`,
`read_records`, `located_records`, the ADOPT fence and the tag-aware rebuild, the 0x0C/0x0D record
classes with their retention rows, `observe_append`, KernelTick's `resident=None` path; the evidence
files survive). **Applied AFTER the panel** (3.6 item 4): the panel's verdict on the null is §4; the
deletion is the fold's stacked commits, one logical change each, re-homing before removal (P2), each
gate re-run, battery-verified only. P8 clause 5 is the licence: a null this clean — the memory claim
met on every conjunct, the cost failed by an order of magnitude in the regime the layer exists for —
is the method being deleted because it was measured, which is the policy working. What the deletion
does NOT touch: records already written under 0x0C/0x0D (IMMORTAL/COMPACTABLE — the lake's schema
outlives its writer), T067's delta-tier record classes and fence (re-homed), the kernel (S6 wires it,
resident-only, until a successor row re-registers paging with the successor comparison 3.3 names).
The deletion commits, if the panel does not block the ruling, are listed in §4 with their hashes; if it
does, §4 records the reversion to NO VERDICT and the instrument work it orders instead.

### 3.6 Deviations from the pre-registration, each with its reason

1. Three amendments before the campaign (2.9–2.11), each dated with the readings that forced it; the
metric, the verdict rule's direction, S = 8, the 3.0× bound, arm F and the LOSS inventory never moved;
the WIN rule gained a STRICTER conjunct and a third outcome. 2. Arm F ran after the campaign, not
before it (the campaign's timing cells needed a quiet host, and F needs the whole RAM); 2.2 left F's
DRIVE unstated, and it ran under R1 (T066's width fixture, the harness's own default and the harshest
for the paged arm — everything pages in), stated rather than chosen for a friendlier number. 3. **The
secondary is read on RAW wall-clock per tick, with the reference-corrected ratio beside it, and the
pre-registered refusal is recorded as an instrument FINDING.** The ambient-ratio control refused R2's
timing arm on every seed: the twin's crc32 reference read 0.41–0.46 ms while every paged cell's read
0.58–0.69 ms (a 35–40 % drift by ARM). A per-core probe showed this host is hybrid (logical CPUs 0–15
at 0.050 ms per sample, 16–27 at 0.092), and the pre-registered "re-run quiet" was executed as a
re-run of all 24 R2 timing cells pinned to the fast class, both arms alike (`--affinity`, recorded per
cell): the drift PERSISTED (0.455 vs 0.615 ms). It is not placement and not contention: the R2 twin
is a process whose whole tick is 0.55 ms, and its adjacent reference sample runs faster than every
other cell's on this host (every other regime's twin AND paged references sit at 0.59–0.69 ms), so
the reference is not arm-independent when one arm is that light — the assumption the ratio method
rests on fails, and the pre-registered refusal cannot be lifted by any quiet re-run. The quantity the
secondary names is tick time; the raw wall-clock per tick on a quiet, pinned host is that quantity
without the failed correction, and it carries its own control (the best-of-3 spread and cross-seed
stability, 3.2). Both readings are printed; the verdict is the same under either; the corrected one
is the more pager-favourable and is 8.7× over the bound. 4. **The deletion follows the panel instead
of preceding it** (2.8 step 4 ordered the disposition before the panel): a LOSS deletes two arcs of
code, so the panel refutes the null BEFORE anything is destroyed; the deletion commits are then the
fold's, battery-verified only, and the panel's verdict on the null is in §4. 5. The R2 timing cells
were re-run outside the driver (its command template, a scratch `NORD_HOME` per cell, one at a time);
the unpinned originals are kept beside them. 6. `run_f.py` refused a numeric `--wstar` (the width is
derived, never picked) — it takes the derivation file; recorded because the first launch was the
orchestrator's mistake, not the harness's. 7. 2.5's SECOND per-item neuter — the identity-order reorder (R4) — was not run in the campaign: no cell in the archive carries `reorder.order_source == "identity"` (13 reordered cells, all `co_fetch`), so R4's 3.95 → 1.00 is attributed by the positive arm alone and the campaign did not separate the co-fetch ORDER from the rewrite that carries it; R4 is tertiary and never deciding, and 2.5's first neuter (eviction) ran in R2 and charges.

### 3.7 Corrections from the panel, dated 2026-09-09 (each quotes the sentence it corrects; §1–2 are left verbatim)

The panel (§4) confirmed 31 findings and refuted 6; 31 of the 31 confirmed findings carried a proposed fix
the refuter showed would introduce a defect, so every correction below is the refuter's minimal fix, never
the panel's. Nothing here moves a number the verdict rests on; §3.4's ruling is restated in item 1 as what
it is.

1. **The ruling is a selection among conflicting pre-registered provisions, and says so (Codex, thread
   `01a084e4`, and panel C5/C14/C27).** 2.6 row 4 says "re-run quiet; never read" and row 5 "no verdict
   until F discriminates"; neither limits itself to the WIN path; 2.10's LOSS clause names only "NOT
   paged < twin, or arm F", omitting cost-only failure; 2.3 says "LOSS iff R2 is not a WIN" and 2.10 says
   the 3.0× bound is "not changed". Reading LOSS under 2.3 after the number therefore SELECTS 2.3 over the
   refusal rows for the LOSS direction. The selection criterion, stated: the refusal rows exist so that a
   contended cost reading cannot hide a real cost and a non-discriminating F cannot decorate a WIN — both
   protect the WIN path — and the quiet re-run row 4 orders was in fact performed by the panel's instrument
   lens (both R2 arms re-run pinned in one session, reference drift 9.0 %, the ambient control PASSING):
   raw 25.7×, reference-corrected 28.5× — the cost conjunct read on the pre-registered instrument with its
   control green is 9.5× over the bound. Row 4's condition is discharged on the row's own terms; row 5 is a
   WIN conjunct's guard. `verdict.py` never produced the label: its `elif refusals` branch short-circuits to
   NO VERDICT before its cost branch, and neither that cost branch (`elif not cost_ok`, the GAP-(i) route)
   nor the arm-F LOSS disjunct is exercised by its five-set self-test (all five run the secondary at 2.1×);
   the panel exercised the cost branch for the first time by lifting the two refusal rows with the readings
   untouched and it printed LOSS at median 26.03×. 2.11's "`verdict.py` is brought up to §2.10 before the
   campaign" is UNVERIFIED, not verified: the file is untracked (no version record) and its only timestamp,
   23:12:12 UTC, is 116 s after the campaign's first cell (five R1-twin cells had completed, one arm, no
   ratio); the archived `t068_probe.py` (last write 00:17:32 UTC, the `--affinity` option) is the version
   that produced the 24 pinned R2 timing cells and the four arm-F cells, not the other 202; no cell records
   a harness hash. What IS witnessed before every number: the outcome set and the conjuncts (`ca8000d`,
   `f61911c`, 22:54:54 UTC, before the A5 check cells at 23:04–23:05 UTC). §3.1's "`verdict.py`'s outcomes
   were fixed before they existed" is withdrawn as to verdict.py. For the successor row: hash the probe and
   the verdict tool into every cell JSON beside the git head.
2. **The raw wall-clock control was misreported (C2, C7).** §3.2's control sentence is replaced in place:
   taking spread as (slowest run − fastest run) / best-of-3, ≤ 9 % holds in 57 of the 64 verdict-arm timing
   cells (64 of 72 with R2's neutered arm), with eight exceedances named there; "the twin's raw tick is stable
   across seeds within 8 % in every regime" is STRUCK — measured on the reported statistic: R1 7.8 %, R2 6.2 %,
   R3 13.8 %, R4 6.3 %. No 32-cell population reproduces the original "30 of 32", and none is constructed.
   The worst spread (1.36×, the R2 twin that is the 35.2× ratio's denominator) taken entirely against the
   ratio leaves 25.9×. The control constrains the reading; it does not rescue it.
3. **The secondary's raw reading lives outside the pre-registered machinery (C20).** `verdict.py` computes
   `secondary_within_bound` from each cell's `reading` = `tick_ratio_to_reference_best_of_3` — the corrected
   quantity — and `campaign/verdict_final.json` stores those medians (R1 3.118, R2 26.030, R3 4.267, R4
   3.844). The RAW ratios in §3.2/§3.4 were computed by a reader outside the archived harness: median over the
   8 seeds of `paged.best_run.mean_tick_ms / twin.best_run.mean_tick_ms`; `best_run` is `min(runs,
   key=ratio)`, so the raw number inherits a best-of-3 selection made on the corrected instrument. Both
   readings are printed everywhere the verdict is stated; neither is under 3.0×.
4. **§3.4's counterfactual is struck (C3).** "so the ratio is set by how many ticks a block stays resident
   between page-ins (8 here, the window ÷ the slide); a working set that moves 100× more slowly would sit
   inside the bound" is withdrawn. Measured by the panel with page-ins and evictions ablated to zero (two
   KernelTicks over the same table, state and drive, `resident=None` against an all-True mask, trains
   byte-identical `589b81547a4c03fe`): the masked path's own tick is 2.5–3.4× the twin's across three runs
   (2.485 / 3.046 / 3.388), and on the live paged arm at R2's resident fraction the paged `kt.tick()` alone
   is 3.2–6.0× the twin's — the residency-mask branch consumes the whole 3.0× budget before any page-in.
   The page-in decomposition, weighted: `pager.step()` 91.8 % of the paged tick (compile ~76 %, read ~4 %),
   `kt.tick()` 8.2 %. "It is NOT a finding that paging cannot work" stands: it is a finding about THIS
   implementation's mask branch and page-in path. The successor does not inherit the mask branch (2.6
   deletes it), so the floor is a fact about the deleted mechanism, not a constraint on a successor.
5. **Arm F's units and the sign of the F2 comparison (C4, C8).** §3.3 mixed GiB (the harness's console
   labels `/ 2**30` as "GB") with decimal GB (the table). Converted in place to DECIMAL throughout (the
   minority change): available 11.83 GB of 34.14, budget 9.47 GB, predicted 9.46 GB at W* and 18.86 GB at
   2·W*. Unit-free ratios: measured / predicted 1.059 at W* (5.9 % OVER, 10,025,857,024 vs 9,464,285,818 B)
   and 0.938 at 2·W* (6.1 % UNDER, 17,702,244,352 vs 18,862,067,912 B) — the original pairing read the
   second as at-or-over. "6 GB above what was available" is 5.87 GB against the availability the derivation
   used (11.83 GB at 00:16 UTC) and 2.89 GB against F2's own recorded start (14.81 GB). The measured twin at
   2·W* exceeded availability under either prediction, so the non-discrimination has ONE cause (the OS pages
   what available RAM does not hold), not two.
6. **Arm F's per-tick number, reported as 2.2 required (C16).** Per tick (one seed, one run,
   `mean_tick_ms_uncontrolled` — NOT the campaign's best-of-3 reference-corrected secondary; store build
   excluded, which the wall column includes): F1 14.9 s, F2 33.7 s, F3 53.8 s, F4 113.9 s — paged / resident
   twin at W* = 3.62×, paged / OS-paged twin at 2·W* = 3.38×. 2.2 pre-registered that this ratio is REPORTED
   with the disk-read caveat and NEVER DECIDES; it enters no conjunct and is not read against the 3.0× bound.
   The successor row inherits 3.38× as its starting comparison, at one seed.
7. **R1's traced cell (C18).** "identical to the byte" is replaced in place: DEGENERATE under 2.3's 1 % tie
   fold — not identity: the paged arm reads LARGER on 8 of 8 seeds, +0.0076 % to +0.0245 %; the EXACT
   instrument is the one that ties to the byte (8,388,608 B in both arms).
8. **What "construction excluded" excludes (C17).** A1's sampling after tick 0 removes the construction
   TRANSIENT (the compile-path peak) but not the LEVEL construction leaves. In R2 the twin's primary reading
   is fixed before the first tick (`wset_per_tick` 223,772,672 → 223,801,344 B, 0.013 % over 20 ticks;
   traced 119,963,548 → 119,973,554, 0.008 %) while the paged arm's climbs on both (126.7 → 135.0 MB; 32.2 →
   47.4 MB); R1's twin is not flat (+2.3 %), so this is a property of the verdict regime. The primary is a
   RESIDENT-set instrument: 0.602 is a claim about RAM the twin occupies, not pages it touches; with a
   working-set trim applied at the same point in both arms (after build, before tick 0) the twin touches
   18.5 MB over 20 ticks and the paged arm 91.7 MB — a touched-set reading inverts the ordering. That
   reading is post-hoc, one seed, one width, control-free, and was measured NON-stationary; it is NOT a
   verdict instrument and A1's pre-stated replacement (private bytes) was already rejected on repeatability.
   The memory WIN is what the pre-registered instrument measures — occupancy — and it is stated as that.
9. **The neutered calibration's repeatability at 4,096 (C19).** (c) was verified at the campaign width. A
   post-record re-run at 4,096 reads footprint repeatability 1.53 % on the first pair and 0.17–3.14 % over
   four runs (4 of 6 pairs outside the 1 % bar); traced 0.114 %; (a) and (b) met at 4,096 (exact 2,097,152 B
   both arms; footprint 1.031–1.064, traced 0.9974). Cause, measured: the twin is stable at both widths
   (0.15 %) while host working-set jitter is roughly constant in bytes (~4.3 MB at 4,096, ~3.1 MB at 16,384)
   over denominators of 131 and 327 MB — the footprint instrument's PRECISION improves with width. Degraded
   precision below the campaign width, not the moving, 1.000-crossing bias A1 convicted the peak of.
10. **2.5's second neuter was not run (C15; §3.6 item 7).** The identity-order reorder neuter for R4 ran in
    no cell (13 reordered cells archive-wide, all `co_fetch`); R4's 3.95 → 1.00 is attributed by the positive
    arm alone and the campaign did not separate the co-fetch ORDER from the rewrite that carries it. R4 is
    tertiary and never deciding; the eviction neuter (2.5's first) ran in R2 and charges.
11. **2.9's "its numbers are quoted in full" was FALSE when written (C6; a dated correction, 2.9 left
    verbatim).** 2.9 quotes the readings that forced A1/A2/A3 (1.0431 / 1.0528, 0.8808 / 0.8804, 0.9767 /
    0.9765, 0.8830, 3.135×) and omits eleven pre-amendment arm ratios, all head `41acd12`: pilot at 4,096
    footprint R1 1.0751, R2 1.0572, R3 0.8564, R4 1.0549, R4-reorder 1.1270; reds at 4,096 R3 footprint
    0.8538 / traced 0.5865; reds16k at 16,384 R3 paged 0.3424 / 0.1518 and R3 neutered 0.3699 / 0.1518 — the
    omitted set is the WIN-shaped one, which is why the omission is not cosmetic. The verdict regime R2 at
    16,384 has no pre-2.9 reading. **The pre-2.11 w = 0.1 positive control (C23)** also produced a
    PRIMARY-instrument reading not quoted in 2.11: steady working set 132,825,088 B — 0.696 of its own
    w = 0.1 neutered arm (a matched pair) and 0.591 against the w = 0.6 twin run four minutes earlier (no
    w = 0.1 twin was run; that ratio is cross-fixture), against the campaign's medians 0.602 and 0.709. No R2
    TIMING cell existed before `f61911c` (the only earlier timing cells are six R1 ambient runs and two R1
    reds16k cells), and `f61911c` was 42 insertions, no deletions — it added A5's rule and moved no conjunct.
12. **2.3's rationale for the 3.0× bound was arithmetically false (C22; 2.3 left verbatim).** "chosen from
    the measured 1.7–3.6× band BEFORE this run so that a cost at the band's top is still within it": the band
    recomputed from T066's archived cells is 1.70 / 3.47 / 3.00 / 2.61 / 3.59, so its top is 3.59× and the
    bound does not contain it — present verbatim in `840dba5` and never edited, a pre-run drafting error,
    not a post-hoc move. The bound sits BELOW the band's top, stricter than the worst cost already seen, so
    the error cuts against the pager. The bound is not moved: no regime's verdict changes under 3.6× (R1 fails
    on memory regardless; R2 35.2× / 26.0×, R3 4.51× / 4.27×, R4 3.81× / 3.84× all exceed it).
13. **§1's citation dropped its source's disclaimer (C21; appended inside 2.10 as a dated sentence).**
    T066's record stated the attribution in the same sentence the 0.250 / 0.164 came from ("— from never
    LOADING, not from eviction"); §1's citation dropped that clause, so A2's "R2 must reproduce 0.250" was
    built on the dropped half. The resident-edge-byte WIN expectation was available from the source; only the
    eviction attribution was not. What this row measured for itself is the wider fact: the exact ratio equals
    the activated-block fraction under every fixture.
14. **§3.6 item 3's reference bands (C25)** are restated on one field and one run: on
    `ref_ms_median_across_runs` (the field the ambient control reads), the pinned re-run's R2 twin reference
    read 0.415–0.442 ms and R2's paged cells 0.580–0.598 ms (a 32.9–40.7 % drift by arm, refusing on 8 of 8
    seeds; the unpinned originals: twin 0.409–0.425, paged 0.575–0.602, drift 37.6–44.8 %); every other
    regime's twin and paged references sit at 0.557–0.648 ms on the same field. The separation argument is
    unchanged: R2 twin's ceiling 0.442 lies below every other cell's floor 0.557. The "0.455 vs 0.615 ms" pair
    in item 3 is the across-seed median of `best_run.mean_ref_ms`, a third field, and is so labelled.
15. **Verdict.py's self-test coverage (C14, C27; record only, the archived harness untouched).** All five
    sets run the secondary at 2.1×, so two outcome branches were never exercised by the battery: the
    cost-fails-with-memory-met route and the arm-F LOSS disjunct. Neither fired on the campaign's cells (the
    refusals short-circuit). A sixth set, if any, belongs to a successor row's pre-registration.
16. **The disposition's inventory, corrected by census (C9, C10, C12, C28–C31; 2.6 left verbatim, §4
    carries the census).** 2.6's "RE-HOMED, not deleted (load-bearing independently of paging)" measured
    FALSE for the block tier: the 0x0C/0x0D record classes' ONE writer (`UnitBlockStore.append_adoptions`),
    all five block-tier reds, `read_records`/`read_record`'s sole production caller, `apply_deltas`,
    `edge_table`, `compare_drop`, `record_from_edges` and the ADOPT fence's rebuild branch all live in or
    are exercised only by the deleted set; `lake_rebuild_gate`'s self-check never scanned a 0x0C/0x0D
    record (tags 1–7 only, 0 decodes) so no gate would ever report the dormancy; and the live lake home has
    NO segment tier, so "records already written are IMMORTAL/COMPACTABLE and stay" binds nothing. The
    disposition taken (P3, no dormancy — recorded here first, then executed in §4's commits): the block-tier
    schema goes with the pager, the immortal-tag count returns to 9, and the S6 kernel wire stays. KernelTick's
    mask branch takes `replay_gate`'s paged refusal and its only test with it (a criterion whose input can no
    longer be constructed is deleted, not weakened; measured: with the guard left standing `replay_gate
    --assert` is green). `p4c_ratchet` reads the git INDEX, so each module's REGIONS/CORE rows and its file
    leave the index in one staged step. The documentation census (about twelve sites 2.6 did not name,
    including `spec.md` and Part F S-R6/S-R7/S-R8; "SYNTHESIS §6 L2" is §2's L2 row) is applied as DATED
    notes, never as edits to pre-registered text. The archived T066/T067/T068 harnesses import the deleted
    modules: tag `t068-last-with-pager` (= `36b2b6a`) is the head to check out to re-run them (Codex).
17. **`41acd12`'s commit message was wrong about 0x0A (C11; record only, the wire is correct).** 0x0A was
    never unbaselined — `tools/lake_append_baseline.json` has carried key 10 = 434 B since `b6465d8`
    (2026-08-21) and `RETENTION[0x0A]` says the flood assert is live — so the journaller wire ARMS a live
    assert on the running daemon (bar 3,472 B/tick from the first write; on a live 40-tick brain 24 writing
    ticks, 0 floods); and a tag with no row self-baselines after 32 of its own writing ticks, so "never
    trips" was wrong in general. The seed is not stale: re-run today, 0x0A per-tick bytes read 193–434 against
    the recorded p95 434 / max 435.

## 4. The panel's verdict on the null, and the fold (2026-09-09, on `36b2b6a`; the fold commits are stacked on it)

**Shape.** Three lenses (instrument / pre-registration honesty / disposition), each filing only findings with a
reproducing command on the frozen head; every finding sent to an independent refuter (Opus) instructed to
refute the CLAIM and, separately, the proposed FIX (obs #2146). 40 agents, 2.8 h. 37 filed: **31 stood** (1
BLOCKER, 12 MAJOR, 18 MINOR), 6 refuted — and **31 of the 31 standing findings carried a proposed fix the
refuter showed would introduce a defect**; every correction in 3.7 and every commit below is the refuter's
minimal fix. Beside the panel, Codex (an independent second model, `codex-converse`, threads `01a083c6` and
`01a084e4`, read-only) was asked to refute the ruling twice: its first apparatus hypothesis (allocation
tracing left active during the timing pass) was REFUTED by the panel at runtime (`tracemalloc.is_tracing()`
False at all 60 tick boundaries; the 35× survives); its second reading — that the ruling selects among
conflicting provisions and that deleting the layer strands the archived harnesses — is folded into 3.7 item 1
and the tag `t068-last-with-pager`.

**The first question, answered.** Is the LOSS a relabelling of a pre-registered condition after its number?
The instrument lens: NO — every number the record quotes re-derives exactly from the archived cells under a
script sharing no code with `verdict.py`; the resolution of 2.10's drafting gap to 2.3 is the only rule
covering the case and pre-dates every number; and an independent paired re-run of both R2 arms, pinned,
with the ambient control PASSING (drift 9.0 %), read raw 25.7× / corrected 28.5× — "the cost finding does
not depend on the refused instrument"; the eviction policy contributes nothing to the cost (neutered 19.92 ms
vs paged 19.98 ms per tick, the same 107 page-ins). The pre-registration-honesty lens found no relabelling
either; what it found (3.7) were wrong numbers in the record's own controls, a false "quoted in full", a
false rationale for the bound that cuts against the pager, and an untracked verdict tool — none moving the
verdict. **The LOSS stands.** Its honest form is 3.7 item 1: a ruling that selects 2.3 over the refusal rows
for the LOSS direction, with the quiet re-run the row ordered performed and passing.

**The second question, answered by measurement.** 2.6's "RE-HOMED, not deleted (load-bearing independently
of paging)" was FALSE for the block tier (3.7 item 16): one writer, five reds, one production caller, all in
the deleted set; a rebuild self-check that never scanned the class; a live lake with no segment tier. The
disposition taken is the P3 one — the schema goes with its writer — recorded in 3.7 before it was executed
here. The BLOCKER (the full suite RED at `36b2b6a` on `doc_budget_gate`, SECURITY.md 713 words against a
488 ceiling after `f441949`) was closed first, in its own commit with the WHY the gate demands (`ac91ed3`).

**The fold, as commits (each one logical change; each gate re-run; the git-index staging rule of 3.7 item 16):**

| commit | change | what the worker found that the brief did not say |
|---|---|---|
| `ac91ed3` | the BLOCKER: `doc_budget_gate` ceiling for SECURITY.md 488 → 713 with the WHY | — |
| `2b11b2d` | the record's §3.7 (the panel's corrections, dated) | — |
| `343cd48` | `lake_compactor.py` + its 34 tests, its REGIONS/CORE/EXEMPT rows — deleted (no non-pager consumer, as 2.6 decided) | — |
| `229cdf5` | `neuron_pager.py` (NeuronPager, UnitBlockStore, both residency instruments, `_derive_rounds`, the width instrument) + its 27 tests, its REGIONS/CORE/`wire` rows — deleted | no other test named it (positive-controlled grep) |
| `327c184` | `unit_blocks.py` retired — its ONE surviving capability ported FIRST: `UnitBlocks.group` → `lake_engine.SynapseDelta.group_by_block` (400 randomized cases, 0 mismatches; the live consumer driven 79 records, identical grouping; a wrong partition IS detected) | **`interaction_learning._record_synapse_delta` imported `UnitBlocks` inside `try … except Exception: return None`** — deleting the module would have reddened nothing and silently turned the 0x09 writer into a no-op; the duplicate-capability hook forced port + retire into one commit (P2's own grammar); 5 of `test_unit_blocks_writer.py`'s 7 arms pin the surviving 0x09 writer and moved to `tests/test_synapse_delta_writer.py` |
| `5b3fa46` | `KernelTick`'s residency-mask branch (`resident`, `pending`, the mask arms in `_propagate` / `_deliver_and_rotate`) + `replay_gate._PAGED_REFUSAL` and its docstring — deleted together (P8 clause 1) | the paged-refusal test's positive control was the suite's ONLY capture/restore round-trip pin — kept as its own test |
| `0f48588` | the block-tier schema: `UnitBlockRecord` / `BlockDeltaRecord` / `BlockAdoptRecord`, `apply_deltas`, `edge_table`, `compare_drop`, `record_from_edges`, `is_block_adoption`, `RETENTION[0x0C]`/`[0x0D]`, `TAG_DISPOSITIONS[0x0C]`/`[0x0D]`, `build_index`'s block branch, `read_record`/`read_records`, `home`'s two tag constants — deleted; immortal tags 10 → 9 | "the 0x0D pruner" named no symbol (nothing orphaned); `test_lake_segments`' pre-T066 read-back arm re-homed with a control of its own |
| `22db40a` | the documentation census (395 hits in 37 files): dated notes at every site, never an edit to pre-registered text; the `INVENTION_REGISTER` row 1 → DELETED ON ITS NULL (its ceiling 1543 → 1595 after condensing, measured + 0) | three sites the brief did not name (`contracts/lake-records.md`, `data-model.md`, `percept_surprise.py`'s prefetch disposition → VOID) |

Gates after every commit, all rc 0: `p4c_ratchet --assert`, `import_manifest`, `dormancy_gate --assert`,
`method_dormancy_gate --assert`, `replication_guard`, `immortal_retention --assert` (9 tags),
`red_reachability --assert`, `replay_gate --assert`, `determinism_gate --assert`; ruff clean; the re-homed
reds proven REACHABLE by mutation (three mutants of the ported method caught by 3/5, 5/5 and 4/5 applicable
arms, the unmutated control 5/5). 6,761 lines deleted, 314 added, across 34 files.

**Documentation census (C28), applied as dated notes, never as edits to pre-registered text:** 395 hits in 37 files; 16 files annotated with dated notes (PLAN-FINAL: S3, S4, S5 → CLOSED, the 2026-09-08 S6
amendment RETRACTED — S6 keeps the kernel wire — T062's rationale, Part F S-R1/R2/R3/R5/R6/R7/R8 → "RETIRED
WITH THE LAYER 2026-09-09 (S-R3 FIRED)", S-R4 deliberately untouched as S6's; tasks T064/T066/T067 (+ the
0x0A correction)/T089, T068 → DONE; spec.md SC-007's second dated line and eight sites; SYNTHESIS seven sites;
quickstart; PROPOSED-ROWS L5; `contracts/lake-records.md`; `data-model.md`; the two evidence headers); left
with a reason: this file, the dated evidence records, `plan.md`/`research.md` (superseded by PLAN-FINAL),
`kernel_state.py` and its test (the 128-byte alignment stands on contiguous SoA reads; the note lives at the
contract they cite), `lake_index.py`'s "REVISIT at S2 paging widths" (a conditional that now never
triggers). The worker's own boundary sweep logged obs #2152 (a test file is a mixture — delete by arm,
not by file), #2153 (a consumer behind a swallowed dynamic import fails silently, never red) and #2154 (a
mutation harness catching `Exception` misses pytest's `Failed`).

**What the deletion leaves.** The kernel (`KernelTick` on the `resident=None` path only, `compile_bundles`,
`Synapses.empty`), the segment tier (`lake_segments.SegmentStore`, `lake_index`) as they were before T066 with
their own `wire` rows, `lake_bridge.located_records` and `home.observe_append` with their live callers, the
evidence files of T066, T067 and T068 (each now headed by a pointer to this section and to the tag), and the
successor comparison 3.7 item 6 names. Nine immortal tags. The S6 kernel wire is unchanged; its pager-wire
amendment and S-R5 are retracted by dated note. T064's deliverable (`unit_blocks`) is void until a successor
paging row; the row stays open and says so.

**Verification of the fold.** The deletion commits were executed by one worker from the panel-corrected
brief and read by the orchestrator; the records commits are the orchestrator's; **neither was re-panelled —
the fold is battery-verified only** (the arc's one battery runs on the stacked head that includes it, and its
record commit names it).
