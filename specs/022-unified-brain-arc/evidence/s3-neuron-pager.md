# S3 / T066 — activation-driven paging, both reds before the mechanism (2026-09-08)

> **THE MECHANISM WAS DELETED ON ITS NULL 2026-09-09** (`substrate/neuron_pager.py` and its tests) -- see `s5-paging-null.md` 3.4/3.5/4. The last head carrying it is tag `t068-last-with-pager`. Everything below is left VERBATIM: the measurements stand, and an evidence file outlives the code it measured.

> Plan row (PLAN-FINAL-2026-09-02.md, Phase 3): **S3 = T066** activation-driven paging, BOTH reds built
> before the mechanism (FR-010/FR-011: no silent drop of a spike to a paged-out unit; the integer rounds
> budget trips on constructed overload).
>
> Task row (tasks.md T066): Activation-driven paging — P8 invention #1: residency policy (custom, never
> mmap), pinned hot set, one async batch per tick, integer ROUNDS budget assert (constructed-overload red),
> two-resolution substrate; paged-out-spike red (no silent drop, bounded staleness) built BEFORE the
> mechanism; in `substrate/neuron_pager.py`.
>
> Acceptance (spec.md US3 scenario 2): Given lake-resident neurons at the binding width, When a tick's
> fetch set is assembled, Then it issues as one async batch within the integer rounds budget, and the
> budget assert can go red (constructed overload). FR-011: a spike arriving at a paged-out neuron MUST
> NOT be silently dropped (constructed red exists before the mechanism).

Filename note: Phase-3 stage naming gives S1 = `s1-unit-blocks.md`, S2 = `s2-state-per-unit.md`, so this
row is `s3-`. The T068 row names its evidence `s2-paging.md` in the older US-stage numbering (US3 = S2);
T068 keeps that name when it lands. Not a conflict, disclosed.

## 1. Territory, measured before the build (HEAD `d1b229b`)

- **`substrate/neuron_pager.py` is ABSENT.** `graph-exists.js "activation-driven paging of units: page
  out inactive units, route spikes to paged-out units without silent drop, integer rounds budget
  overload"` → NO EXISTING IMPLEMENTATION. A 38-file census of `resident|evict|prefetch|paging|page_`
  over `src/` found NO per-unit residency, no hot set, no prefetch queue and no rounds concept
  (`region_wiring.sync_rounds` is a barrier-cost term). Every near-match was adjudicated distinct-from
  (four sites read 2026-09-08):
  - `substrate/neuro_chip_stream.py` — the 2026-09-01 audit's `GAP_A3_neuro_chip_stream_is_an_unadjudicated_pager`.
    Adjudicated here: it streams a RECORDED chip's spike shards (input data, one core resident); it holds
    no substrate unit, no activation-driven policy and no delivery path. Different job, not a rival
    mechanism; nothing to adopt or merge. Its `max_resident_cores` bookkeeping-integer defect is the
    shape this row's residency proof must NOT repeat (see 2.6).
  - `substrate/proximity_index.py` — the tree's only mmap store (`mmap.mmap(..., ACCESS_READ)` + LRU).
    It is the incumbent the row's "never mmap" clause refuses: OS eviction is blind to activation
    (research.md D1). It pages medoid PATTERNS for a similarity index, not units. Not adopted.
  - `substrate/unit_blocks.py` (T064) — the partition (`block_of`, `units_in`, `group`) IS adopted as
    this row's block arithmetic (P2: one partition). Its `max_resident` counts context-manager
    nesting, not memory (its own panel's correction) and is NOT used as a residency instrument here.
  - `substrate/lake_memory.py`, `cortex/theta_gamma_code.py`, `hippocampus/affective_memory.py`,
    `hippocampus/episode_consolidation.py` — cognitive-level eviction of traces / slots / episodes;
    none touches substrate units.
- **The lake tier T066 is the named live consumer of already exists and is uncharged:** `lake_segments`
  (Arrow IPC segments, footer-indexed, CRC envelope), `lake_index` (sqlite `(tag,key) -> (container,seq)`
  pointers, whose own docstring says "REVISIT at S2 paging widths per D3"), `lake_bridge` (the ONE
  forward reader). Their `wire` dormancy exemptions were RETIRED 2026-09-06 with the comment that their
  consumer is T066 and "BOUND EXPIRED AT T039 UNMET". `pyarrow 25.0.1` is importable in `.venv-lake`.
  `SegmentStore.seal()` writes a whole sealed segment as ONE Arrow table (`write_table`), so today a
  record inside a segment is reachable only by reading the segment whole (`read_segment` CRC-checks
  every record). Per-record random access needs one Arrow record batch per record and a
  `read_record(container, seq)` that seeks by the footer — an ADDITION to the incumbent writer/reader,
  never a second store (P4b).
- **The substrate tick.** `KernelState` (T013) holds five 128-byte-aligned SoA arrays: v `<f4`, thr
  `<f4`, refrac `<i4`, last_spike `<i8`, sign `|i1` = 21 B/unit. `KernelTick` (T014) addresses posts by
  DENSE index over `[0, n_units)`: `_propagate(p)` integrates every member of partition p;
  `_deliver_and_rotate(p)` drains slot `(tick+1) % (max_delay+1)` with `np.add.at(acc, posts, q)` over a
  FULL-width int64 accumulator per partition; there is no mask, no residency bit, no bounds check at
  delivery. `Synapses` compiles ALL edges at construction into `bundles[(pre, part, delay)] ->
  (posts, q)` and `by_pre[pre] -> [(part, delay, key)]` via one `np.lexsort`; there is no partial table
  and no per-unit edge record in the lake. Determinism (T015) rests on partition-owned INTEGER
  accumulation at fixed P; replay (G5) captures the whole KernelState frame, every `acc_<p>` and every
  ring slot's key list verbatim.
- **Where the bytes are, measured on the incumbent representation** (this decides what pages): per unit,
  state is 21 B and the per-partition accumulator 8 B; each out-edge costs 16 B in bundle arrays
  (post int64 + q int64) plus Python dict/array overhead per bundle. At 32 out-edges per unit the edge
  arrays alone are 512 B/unit — over 94 % of resident bytes before overhead. **The paged quantity is the
  block's out-edges and weights; unit STATE stays resident** (its durable owner is the `kernel_state.v1`
  `_T_STATE` frame — a second copy of state inside the block record would be a P2/P4 duplicate).
- **The constructed width.** T065's `bytes_per_unit(m) = 271 + 8·m` is the GROWER's blob (NGSNN units
  with a beta column per class), measured in memory; its widths are withdrawn as figures. It does not
  describe the kernel representation above. This row therefore reports the kernel's own bytes/block
  (measured, 2.6) and runs its width arm at a CONSTRUCTED width stated as an input; no figure from T065
  is reused as a kernel quantity.
- **What the design text left undefined** (eight files read; the reader's list): what a ROUND is; the
  staleness BOUND; the contents of the two-resolution summary; the hot set's membership rule; the tier
  policies; whether "evict" means residency or record destruction. Every one is DEFINED in 2.1 below,
  before the build, so the reds can be constructed against a definition rather than a phrase.
- **Reds already pre-registered elsewhere, honoured here:** Part F S-R1/R2 ("rounds budget never trips on
  constructed overload; a paged-out spike is dropped silently"; controls "constructed overload; a
  resident twin"); INVENTION_REGISTER row 1 (RED / CONTROL forced-total residency / ATTRIBUTION rounds
  counter vs budget / DISCLOSURE comparison width ≠ deployment width / NULL delete). The NULL (S-R3,
  SC-007) is T068's row and is NOT pre-judged here.
- **T028's forward disposition** (surprise-driven prefetch, "discharged at T066") is adjudicated in 2.8:
  NOT discharged this row, re-dated with its reason, and the edge is NOT declared.

## 2. Pre-registration (written before `neuron_pager.py` existed)

### 2.1 The mechanism, with every open term defined

**Unit block.** `UnitBlocks(n_units, block_size)` partitions units into contiguous blocks (T064's
arithmetic, adopted). `block_size` has NO default and must be a positive multiple of 128 so every SoA
field's block start stays 128-byte aligned (measured: the `|i1` sign array is the binding field; 32
suffices for the 4- and 8-byte fields, only multiples of 128 align all five). The caller derives
`block_size` from its bytes-per-block target and the measured bytes per unit.

**The lake record — `_T_UNITBLOCK = 0x0C`** (both registries checked: highest tag in use is 0x0B;
`lake_engine.RETENTION[0x0C]` and `lake_migrate.TAG_DISPOSITIONS[0x0C]` land together;
`registry_consistent()` asserted). One record per block: `struct '<BIIH'` (version=1, block_id,
n_units_in_block, block_size) followed by the block's out-edges as four parallel arrays
`pre_local <u2`, `post <u4`, `weight <f4`, `delay <u2` (little-endian, lengths equal). Payload only,
CRC-enveloped by `lake_segments.wrap`. Label **IMMORTAL, reclaimable=False**: the block is the unit's
synaptic identity; there is no snapshot to reconstruct it from. Malformed payloads are refused WHOLE
(bad version, short header, length mismatch) — the SynapseDelta idiom.

**The store — `UnitBlockStore(root)`** lives INSIDE the lake (P4b): records are appended through
`lake_segments.SegmentStore` and sealed with ONE ARROW RECORD BATCH PER RECORD (an addition to
`seal()`, applied to every tenant — the byte layout of a sealed segment changes and no production
segment exists yet; `read_segment`/`lake_bridge.iter_records`/`lake_rebuild_gate` must stay green on
the new layout, tested), and pointers are folded into `LakeIndex` as `(0x0C, block_id) -> (container,
seq)` with the watermark advanced in the same transaction. `read_record(container, seq)` (new in
`lake_segments`) opens the file plainly — `pa.ipc.open_file` on a regular file handle, NEVER
`memory_map` — reads exactly one batch by the footer index and verifies that ONE record's CRC.
`fetch(block_ids) -> {block_id: decoded}` submits the reads of one ROUND as one batch to a thread pool
(the "async batch") and counts `submissions` — the store-side rounds instrument.

**A ROUND (defined).** One `UnitBlockStore.fetch` submission of at most `blocks_per_round` block
reads. A tick whose fetch set holds F blocks costs `rounds = ceil(F / blocks_per_round)` submissions,
issued sequentially. The budget is `max_rounds: int ≥ 1` per tick. Both constants are INPUTS with NO
DEFAULT; the self-check derives them in the same command from a measured per-block read latency and a
per-tick I/O allowance it states (the recorded wake-tick p50 from `tools/perf_baseline.json` is the
reference it names), and prints the derivation. Rounds are counted TWICE, by instruments that fail
differently: the pager's planner (`rounds_used[tick]`) and the store's `submissions` counter; the
self-check asserts they agree on every tick.

**Residency (custom, never mmap).** `NeuronPager(kt, store, *, blocks, capacity_blocks,
blocks_per_round, max_rounds, max_stale_ticks, pinned)` owns `resident: bool[n_units]` (a view shared
with the KernelTick) and the per-block `last_active[block]` (max over the block's `last_spike` and the
tick of the block's last pending arrival). A block is RESIDENT when its units integrate and its
compiled bundles sit in `syn.bundles` / `syn.by_pre`; PAGED-OUT when its bundles are removed and its
units are masked out of propagate. `capacity_blocks` has no default. **Two-resolution substrate:**
resident units carry full state + edges; a paged-out unit carries exactly the lossy resident summary —
`pending[p][unit]` (int64, per partition, the single-writer contract) and `last_active` of its block —
and that summary DECIDES fetches: a paged-out unit with `pending ≠ 0` is DUE.

**The tick with the pager.** At the START of tick t, before propagate: (1) `plan()`: the due set =
blocks holding any paged-out unit with pending ≠ 0, ordered (overdue first, then oldest pending
arrival, then largest |pending|, then lowest block id — deterministic); (2) `fetch` in rounds of
`blocks_per_round` until the due set is empty or `max_rounds` is reached; a due block whose oldest
pending arrival is older than `max_stale_ticks` ticks is FORCED into an extra round beyond the budget
rather than deferred — a budget TRIP is counted (`budget_trips += 1`, with the tick and the overflow
recorded) and nothing is dropped; (3) for each block paged in: compile its record's edges with the SAME
compiler `Synapses.__init__` uses (factored out as `compile_bundles(...)`, one compiler — P2), insert
into `syn.bundles`/`syn.by_pre`, set `resident[units] = True`, drain `pending[p][units]` into
`kt._acc[p][units]` exactly (integer), zero the pending; (4) eviction to make room: among non-pinned
resident blocks NOT in this tick's due set, choose by smallest `last_active`, then lowest block id, and
only a block whose units all have `last_spike < tick − max_delay` and zero pending (its bundle keys can
be in no ring slot, so removal cannot orphan a scheduled delivery); if no block is eligible the fetch
proceeds over capacity and `capacity_trips += 1` is counted — never a drop. Then `kt.tick()`: propagate
integrates only resident members; `_deliver_and_rotate` splits each bundle's posts by the mask —
resident posts into the accumulator, paged-out posts into `pending[p]` — a delivery is never
discarded.

**KernelTick hook (the incumbent's one residency branch, P2):** `KernelTick(..., resident=None)`. With
`resident is None` every phase runs the code it runs today (no new branch executes); with a mask,
`_propagate` restricts `m` to resident members and `_deliver_and_rotate` splits posts. Bit-identity of
`resident=None` against an all-True mask is asserted (rings, accumulators, trains). `Synapses.empty(n_units,
p_parts, frac_bits, max_delay)` constructs the empty table the pager fills; the refusals (out-of-range
endpoint, zero delay, delay > max_delay) live in `compile_bundles` and bind the per-block compile too.

**Bounded staleness (defined).** Staleness of a delivery = ticks between its arrival in `pending` and
its drain into the accumulator. Within budget it is ZERO by construction (arrival at the end of tick
t−1, drain at the start of tick t, integrated at t — exactly when the resident twin integrates it), so
a calm paged run is byte-identical to the twin. Under overload it is ≤ `max_stale_ticks` (an input
with no default) by the forced-round rule above; exceeding it is impossible by construction and the
cost of that guarantee is visible as trips.

**Pinned hot set.** `pinned: frozenset[block]` — never evicted, resident from construction, counted in
capacity. The pager does not choose them (the contract says executive/gating units; the caller names
them — P1). Refused if a pinned block id is out of range or if `len(pinned) > capacity_blocks`.

**Tiers.** Host (resident blocks + the always-resident state/accumulator/summary floor) and lake (the
store) land here. The DEVICE tier is S6's cutover with its own workload name and is NOT stubbed: a
declared-empty tier would be P3 shape 5. FR-010's third tier is therefore carried as S6's debt, stated.

**Prefetch.** None this row (2.8).

### 2.2 ONE mechanism, N callers (P2 / P4)

- One partition: `UnitBlocks` (T064). One bundle compiler: `compile_bundles` (factored out of
  `Synapses.__init__`, used by the constructor and by the pager's page-in). One delivery phase:
  `KernelTick._deliver_and_rotate` gains the residency split; no second scatter exists. One lake: the
  block store is `lake_segments` + `lake_index` under a new tag — no sidecar file, no second durable
  store. One state owner: `KernelState` / `kernel_state.v1` — the block record carries NO state.
- P2 pin: a test asserts `compile_bundles` is the only site that builds `bundles`/`by_pre` (a census over
  `kernel_tick.py` and `neuron_pager.py`, anchored on `__file__`, not a single literal).

### 2.3 The reds, each with the input that makes it fail (built and COUNTED RED before the mechanism)

- **S-R2 — a spike to a paged-out unit is dropped silently.** Fixture: n units in `n/block_size`
  blocks, random out-edges, delays 1..D, external drive on a subset. Arms: the RESIDENT TWIN (plain
  `KernelTick` with the full `Synapses`, today's mechanism, no pager) and the PAGED run (same seed,
  `capacity_blocks < n_blocks`, calm budget). Criteria: (i) CALM arm — spike trains byte-identical to the
  twin over T ticks (`fired.tobytes()` per tick), ring-slot key lists identical per tick, and the run
  actually paged (`page_ins > 0`, `evictions > 0`, resident fraction strictly between 0 and 1, printed);
  (ii) OVERLOAD arm — per-unit delivered charge over the run (accumulator drains + pending drains)
  equals the twin's exactly, max observed staleness ≤ `max_stale_ticks`, every pending drains to zero by
  a stated tail, and `budget_trips > 0`. **The red:** a DROPPING twin — a test-constructed `KernelTick`
  subclass whose deliver phase discards non-resident posts (the naive pager) — must show a charge
  shortfall EQUAL to its own count of discarded deliveries (not merely > 0) and trains that differ from
  the resident twin. Runs IN THE SAME COMMAND as the criterion, in the self-check and in pytest.
- **S-R1 — the rounds budget never trips on constructed overload.** Constructed overload: `capacity_blocks`
  small, a drive pulse that makes every block due in one tick, `blocks_per_round × max_rounds <
  n_blocks`. Criterion: `budget_trips ≥ 1` on that tick, `rounds_used` on that tick > `max_rounds`, the
  trip record names the tick and the overflow, and `assert_within_budget()` RAISES
  `RoundsBudgetExceeded`. CONTROL in the same command: a calm arm (due set ≤ `blocks_per_round ×
  max_rounds` every tick) with `budget_trips == 0`, `rounds_used ≤ max_rounds` on every tick and
  `max(rounds_used) ≥ 1` (the budget is exercised, not idle). **The red:** a BUDGET-BLIND variant — a
  test-constructed pager subclass that submits the whole due set in one round regardless of
  `blocks_per_round` — reads `budget_trips == 0` on the identical overload, so the criterion is seen to
  fail. Both rounds instruments (planner vs store submissions) must agree on every tick in every arm.
- **Commit order (the rule the row is named for):** the tests carrying both reds land in their own
  commit BEFORE the mechanism and are RUN on that tree; their failure output is quoted in §3 as the
  counted red. The mechanism commit turns them green with the two constructed broken arms still failing
  in-command.

### 2.4 Statistic and bar

Substrate row: arm-table assertions (exact integer equalities, byte-identity, counter inequalities), no
seed sweep. Where a quantity is measured rather than asserted (bytes, latency, rounds per tick) it is
printed with its instrument named. No paired-t is claimed; none is owed by a bit-identity criterion.

### 2.5 The control that is not the method (P8 clause 2)

The RESIDENT TWIN is today's `KernelTick` with all edges compiled — it contains no pager code. Expected
readings before the run: calm paged ≡ twin (bit-identical); overload paged ≠ twin in trains but equal
in delivered charge; dropping twin < twin in delivered charge by exactly its discard count. Which arm
can FAIL: the dropping twin fails S-R2; the budget-blind pager fails S-R1; the calm arm fails if either
the mask or the drain is wrong.

### 2.6 Attribution and disclosure (P8 clauses 3 and 4)

- **Attribution by per-item neuter:** neuter the pending accumulator → the drop shortfall appears (2.3);
  neuter the rounds accounting → trips vanish on overload (2.3); neuter eviction (capacity = n_blocks)
  → page-ins after warm-up fall to zero and resident bytes equal the twin's (a third arm, printed).
- **Residency proof that measures MEMORY, not nesting** (the T064 debt): `resident_edge_bytes()` = the
  exact sum of `nbytes` over resident bundle arrays, AND `tracemalloc` peak over the same run as a
  second instrument that fails differently; both must place the paged run BELOW the twin, and both
  numbers are printed with their ratio. Neither is a win claim.
- **Disclosed preconditions and limits:** (a) state (21 B/unit), the per-partition accumulator
  (8 B/unit) and the pager's summary (pending 8 B/unit/partition + a resident flag) stay resident for
  EVERY unit — paging reduces edge bytes only, and the resident floor per unit is printed; (b) a
  paged-out unit's membrane does not decay while paged out (its inputs are conserved as integers) —
  identical to the twin only because drain precedes integration; (c) "async" means one thread-pool
  submission per round overlapping the reads with each other, not with compute — no prefetch overlaps
  I/O with the tick this row; (d) random connectivity has no locality: at a capacity below the
  connectivity's reach the due set approaches every block every tick and the pager thrashes — that
  measurement is REPORTED in the width arm as the regime the locality reorder (T067) exists for, not
  smoothed; (e) the store's writes are a one-time build, not per-tick appends, and are not routed
  through `_observe_append` — the append-budget class for per-tick growth deltas is T067's; (f) the
  `_T_UNITBLOCK` class is IMMORTAL and the pager's page-out DESTROYS NO RECORD — it changes residency
  in RAM only; L-R6's "paging pass" clause is discharged by a lake-digest identity arm (the store's
  bytes before and after a full page-out/page-in cycle are identical), and the "MORTAL class evicted in
  the same pass" control is NOT APPLICABLE to a pass that destroys nothing — stated, not faked; (g) the
  pager is NOT on the live tick until S6 — it lands with a `dormancy_gate.EXEMPT` `wire` row dated to
  S6 and its consumers today are its self-check and tests; `lake_segments`/`lake_index`/`lake_bridge`
  gain a charged in-tree consumer but remain off the live tick for the same reason; (h) the cockpit
  row "paging binding-width status" is owed by the S6 wire, not here.

### 2.7 P2 adjudication (stated before the build)

`neuro_chip_stream` and `proximity_index`: named, read, not rivals (1.). `UnitBlocks`: adopted. The
block record carries no state (KernelState owns it). The T064 §2.3 deletion condition ("if the first
real consumer of grouped records does not use the block structure, the blocks go") is answered by THIS
row: the pager is the first real consumer of `UnitBlocks` and it uses `block_of`/`units_in` as the
residency grain — `group()`/`stream()` (the delta-writer half) remain T064's debt, untouched here.

### 2.8 The T028 prefetch disposition — NOT discharged, re-dated with the reason

A surprise scalar names no block. Without a percept→block map or a block adjacency with locality
meaning, a `surprise_in` port on the pager would be an edge whose consumer changes nothing — the
force-wiring P6 forbids and P3 shape 4. The edge is NOT declared; `manifest_gate.REGISTRANTS` is not
touched. The disposition is re-dated to **T067** (the locality reorder gives block adjacency meaning,
and surprise then widens the prefetch radius over that adjacency with spare rounds), annotated at both
sites where it was dated (SYNTHESIS §6 and `percept_surprise.declare_ports`'s docstring), per the
T034/M10 convention.

### 2.9 Disposition on a null (fixed in advance)

| outcome | disposition |
|---|---|
| S-R1's trip cannot be constructed at any capacity/budget | the budget criterion is DELETED, not weakened (F11) |
| S-R2's calm identity fails on the mechanism | the mechanism does not land; the failing tick and the first differing unit are recorded |
| the dropping twin does NOT show a shortfall | the conservation instrument is blind; fix the instrument first, no verdict |
| the two rounds instruments disagree on any tick | no rounds number is reported until they agree |
| paging loses to forced residency | NOT this row's call — T068 (S-R3 / SC-007) decides and owns the deletion |

### 2.10 What this row does NOT claim

No paging WIN, no tick-time claim, no bytes-fetched/bytes-used claim (T067's metric), no live-tick
charge, no device tier, no prefetch, no persisted-bytes figure for the grower's units. It claims: the
two reds are reachable and counted; a calm paged run is byte-identical to the resident twin; delivered
charge is conserved under overload with staleness bounded by an input; rounds are integer, budgeted,
double-counted and trip on constructed overload; resident edge bytes fall, measured two ways.

### 2.11 Commit order

1. this file (§1–2) — 2. the reds: `tests/test_neuron_pager.py`, RUN RED and quoted — 3. the
mechanism (kernel hook + `compile_bundles` + `Synapses.empty`, `_T_UNITBLOCK` in both registries,
`read_record` + one-batch-per-record sealing, `neuron_pager.py` with its self-check, the registration
triad, the `wire` row) — 4. records (INVENTION_REGISTER row 1 line, SYNTHESIS §6 + `percept_surprise`
re-dating, tasks.md row, plan S3 annotation, §3 of this file) — panel (3 lenses) — fold — ONE battery
on the stacked head — closing record commit — push.

## 3. Results (on the stacked head, before the panel)

Five commits on `d1b229b`: `2a3be20` (this file, §1–2) → `2214dd0` (the reds, RUN RED) → `53e4bc3` (the
mechanism) → `1b0bc1c` (two defects the orchestrator's read found, each red-first) → `0901a02` (a third,
red-first). The records commit follows. Nothing was pushed before the panel ran.

### 3.1 The counted red (`2214dd0`, the tree carrying only the test file)

`pytest tests/test_neuron_pager.py -q -p no:cacheprovider --no-cov` →
`ModuleNotFoundError: No module named 'snn_data_lake.substrate.neuron_pager'` — **23 test functions
defined, 0 collected, 1 collection error, rc 2.** Nothing in the file could pass until the mechanism
existed.

### 3.2 The mechanism (`53e4bc3`) — the self-check's arms, one command, `throwaway_home()`

| arm | reading |
|---|---|
| derivation (P1) | per-block read ≈1.0–1.1 ms (median of 24 single-block rounds); `blocks_per_round` = the widest round still within 2× the single-block latency (read 8 / 8 / 4 across runs); allowance 5 % of the recorded wake-tick p50 119.6 ms = 5.98 ms → `max_rounds` (read 2 / 1 / 3). The derivation is a measurement and moves between runs; the calm arm's criterion (byte-identity) does not depend on which sufficient budget it lands on |
| S-R2 calm | **45 ticks byte-identical** to the resident twin in trains AND ring-slot key lists; `page_ins` 14, `evictions` 8, resident fraction 0.875 of 8 blocks; rounds trips 0; **capacity trips 8** (below) |
| S-R2 red | the DROPPING twin lost exactly **28,311,552** fixed-point units of charge across 46 units — shortfall equal to its own discard count per unit; trains differ |
| S-R2 overload, `max_stale_ticks = 0` | every due block forced; staleness 0; byte-identical to the twin; charge equal to the twin's exactly; budget trips > 0 |
| S-R2 overload, `max_stale_ticks = 2` | 1 trip, max observed staleness 2 ≤ 2, delivered == integrated per unit, pending drained to zero by the stated tail |
| S-R1 overload | tick 1 needed 4 rounds against a 2-round budget (overflow 2); the trip record names tick and overflow; `assert_within_budget()` RAISED; nothing left paged out |
| S-R1 calm control | 0 trips, max rounds 2 ≥ 1 |
| S-R1 red | the BUDGET-BLIND pager read 0 trips on the identical overload |
| rounds instruments | planner `rounds_used` and store `submissions` agree on every tick of all arms — a NARROW separation (one call apart), stated as narrow |
| residency | resident edge bytes **49,152 vs 57,344** (ratio 0.857) — since the fold READ TWICE by two EXACT instruments that must agree to the byte (the sum over the bundle table, and the sum over the same arrays reached by walking `by_pre`'s routes; 0/0 dangling routes); the reduction is **one demoted block of the seven that own edges**, at peak 7 of 8 resident, time-mean 5.90, 16 of 20 ticks above `capacity_blocks = 4`; resident floor **86.0 B/unit** for every unit (state 21 B + accumulators + pending + a flag) — paging reduces edge bytes only. The tracemalloc readings (filtered 0.790, unfiltered peak 0.931) are printed as allocator footprints at ~34× the exact quantity and are used for NO criterion (4.2) |
| lake | 14 page-ins and 8 page-outs left the store's segment bytes IDENTICAL (sha256 `935a61ec0d576446…`); the MORTAL-class control is NOT APPLICABLE to a pass that destroys nothing |
| tests / gates | `tests/test_neuron_pager.py` 23 passed; the kernel + unit_blocks + lake set 71 passed; `tests/test_selfchecks.py` 288 passed; every listed substrate gate and the six tools gates rc 0; self-check 1.2–1.5 s wall |

**The capacity trips are the arm's honest half.** Residency stood ABOVE `capacity_blocks = 4` on **37 of
45 ticks (82 %, peak 7 of 8, time-mean 5.76)**; `capacity_trips = 8` counts only the ticks on which a fetch
found too few evictable blocks (the fold prints both quantities and the counter's definition beside
them — the first record read the counter as the tick count, 4.2). On those ticks the fetch went OVER
capacity rather than deferring a block (never a drop). A per-trip census with the eviction predicate itself: 8 block-visits were pinned or in
that tick's due set, **30 were refused, and all 30 had spiked inside the ring window** `tick −
max_delay`. A block whose units fired within `max_delay` ticks may still have bundle keys in a delay
slot and cannot be demoted; a chain-wavefront fixture keeps exactly those blocks hot. **The cause is the
ring-safety clause under a firing wavefront — NOT locality: the fixture is a nearest-neighbour chain, the
most local topology possible, and the first record's "no-locality regime (T067)" attribution at this site
was wrong (4.2).** It means
**capacity is a target the pager honours when a block is evictable, not a bound it can enforce
under sustained firing**. T068 inherits that as a measured property, not a surprise.

### 3.3 Three defects found AFTER the mechanism was green, each constructed red first

All three were found by the orchestrator's read of the head, not by the 23 tests or the 24 green
gates; each was constructed as a failing test on the tree that had it, quoted, then fixed.

- **A — external drive to a DEFERRED paged-out block was dropped** (`1b0bc1c`). `_propagate` restricted
  members to resident ones, so a non-resident unit's `ext_q` this tick was added nowhere; the next
  tick's plan saw only the next tick's drive. On `53e4bc3`: `18874368 fixed-point units lost across
  896 units; the independent counter of drive that arrived at non-resident units reads 18874368` —
  the two counters agree to the unit, attributing the loss to that port and no other. Fix: the
  membrane update runs over EVERY member (a paged-out unit ages exactly as the twin's does, `inj` is
  exactly zero for it, so the decay is bit-identical rather than approximately right); residency now
  gates two things — only resident members FIRE, and a non-resident member's `ext_q` goes to
  `pending[p]` under the same single-writer contract delivery uses. **Disclosure 2.6(b) is
  WITHDRAWN**: a paged-out membrane DOES decay.
- **B — eviction demanded an exactly-zero membrane** (`1b0bc1c`). The clause existed only because the
  membrane was frozen; with A fixed it had no reason. On `53e4bc3`: `the block holding sub-threshold
  charge was never demoted (evicted []); a membrane that only decays toward zero pinned it in core`.
  `_at_rest` → `_evictable`: ring safety, `refrac == 0`, `acc == 0`, `pending == 0`. The
  membrane-equality assert in that arm was probed for its own red (decay half neutered: `evicted [1,
  2, 3], membranes identical to twin = False, 3 units differing`) — it measures the decay.
- **C — a unit at threshold with its refractory just released could be demoted before it fired**
  (`0901a02`), reachable when `refrac_ticks > max_delay`. On `1b0bc1c`, `max_delay 2`, `refrac_ticks
  5`: `the twin fired [128, 256] and the paged run fired [256] — a unit at threshold with its
  refractory released was demoted before propagate and its spike was DROPPED, not deferred`. Fix: one
  clause — a block with any unit at `v ≥ thr` is not evictable; the cost is a capacity trip, asserted.

The self-check's calm arm moved by ZERO on every counter across A/B/C: that fixture's edges all clear
threshold, so every unit resets to exactly 0.0 and none of the three clauses bound it. Their cost is
visible only in the constructed arms — which is what a red is for.

### 3.4 Deviations from the pre-registration, each with its reason

1. **The record leads with its class tag byte** (`0x0C` before the `<BIIH` header): the lake's dispatch
   convention is `payload[0] == tag`; without it the `TAG_DISPOSITIONS[0x0C]` row is unreachable and a
   migration would read the record as `_T_FACT`.
2. **External drive makes a block due** (2.1 named pending alone) — the LATENCY clause that keeps the
   calm arm at zero staleness; conservation of that drive is A above.
3. **S-R2's overload arm runs in two regimes** (2.3(ii) and 2.5 were not jointly satisfiable: charge
   cannot equal the twin's while trains differ) — `max_stale_ticks = 0` for identity-with-trips,
   `max_stale_ticks = 2` for deferral-with-conservation; both in the same command.
4. **Every block gets a record**, including edge-less ones (an unwritten block is an unfetchable one).
5. **`page_ins` excludes the pinned hot set's construction load.**
6. **The rounds double-count is a narrow separation**, stated as narrow.
7. **The seal layout change is forward-only**: `read_all()` concatenates batches whatever their width, so
   every existing reader is unaffected; only `read_record` needs one-batch-per-record and refuses an
   old whole-table segment loudly. No production segment existed to migrate.
8. **The second residency instrument, as first landed, was a FILTERED tracemalloc snapshot — and the
   panel measured it reading a DIFFERENT quantity (4.2); the fold replaced it with a second EXACT reading.
   The first record's account, kept for the trail:** The peak was built and run and measured to be measuring something else: 0.888
   standalone, **1.262 in-process under `run_selfchecks`** (it counts the sqlite index, the Arrow
   batches and the thread pool that exist only because the run pages, and it depends on process
   history). The filtered snapshot was then found to read ~34× the exact quantity, to track bundle COUNT,
   and to disagree with itself across process history — the same defect (4.2). Neither allocator reading
   is a criterion now; both are printed as footprints. **The honest reading of the unfiltered peak stands
   beside them: at this fixture's size, paging's own machinery costs more than the edge bytes it saves
   in-process.** Whether that inverts at width is the width arm's question (3.5), not this arm's.
11. **The third pre-registered attribution arm (2.6, eviction neutered at `capacity = n_blocks`) was DROPPED
    at build time and this list omitted it** (panel, C6). Built in the fold: resident edge bytes 57,344 ==
    the twin's with eviction off, 49,152 with eviction on — the reduction is eviction's and nothing
    else's; the page-ins-after-warm-up clause reads 0/0 on both arms and is printed as non-discriminating.
12. **`UnitBlockStore.build` no longer advances the pointer index's watermark** (`watermark=None`; 2.1's
    "advanced in the same transaction" is withdrawn): the marker belongs to the lake's own fold, and the
    store shares the index's key space with a provisional rebuild keying that does not reconstruct `0x0C`
    rows — owed at S6 (4.2).
9. **The `resident is None` path shares the LIF step with the mask path** rather than duplicating it
   into two branches: the incumbent's arithmetic is untouched, each residency-specific step sits behind
   an `is None` test the unpaged path never enters, and byte-identity is asserted (trains, rings,
   accumulators, membranes) and re-run by the determinism and replay gates.
10. **A pre-existing lint red** (`ruff` I001 in the test file at `53e4bc3`, verified by stashing) was
    cleared in `1b0bc1c`; the mechanism commit's own report had recorded `ruff check .` rc 0 — one of the
    two readings was taken on a different tree state and the panel is asked to say which.

### 3.5 The width arm

Measured on the pre-fold head `4d09202` (the fold `99ed4be` added instruments and refusals; the
residency rule, the planner and the budget are unchanged), by an archived session probe
(`memory/archive/t066-2026-09-08/t066_probe.py`, cells under `cells/`), **at 32,768 units** (the
sanctioned reduction from 65,536 — the paged cells took 57–71 min each even so), 256 blocks of 128,
32 out-edges per unit, delays 1..8, 200 ticks, seed 0, `max_stale_ticks = 2`, no pinned set:

| arm | topology | capacity | resident edge bytes, exact | page-ins | evictions | budget trips | capacity trips | rounds mean / max | trains = twin |
|---|---|---|---|---|---|---|---|---|---|
| twin | random | — | 16,777,216 | — | — | — | — | — | — |
| paged | random | 0.10 (26 / 256) | 16,777,216 (**1.000**) | 256 | **0** | 1 | 1 | 0.64 / 126 | no |
| twin | local (±2 blocks) | — | 16,777,216 | — | — | — | — | — | — |
| paged | local | 0.10 | 16,777,216 (**1.000**) | 256 | **0** | 1 | 1 | 0.32 / 58 | no |
| paged | local | 0.50 (128 / 256) | 16,777,216 (**1.000**) | 256 | **0** | 1 | 1 | 0.32 / 56 | no |

**What it measured.** The fixture's drive saturates the substrate — ~99.2 % of units fire on every tick —
so the ring-safety clause is false for every block on every tick: **evictions 0 in every cell, residency
1.000, capacity (26 or 128 blocks) never enforceable, and the 0.10-vs-0.50 contrast measured nothing about
capacity.** Exact resident edge bytes read the twin's 16,777,216 (32,768 × 32 × 16 B) in all five cells.
The single budget trip per paged cell is the forced round at tick 2: all 256 blocks are due at tick 0
against a 2–12-block budget, so byte-identity with the twin was unreachable by construction (a spike
deficit of 0.65–0.70 %, all in the first ticks, staleness saturated at 2) — deferral, not loss. Both
rounds instruments agreed on every tick and matched the store's submission delta; each cell ran twice
in-process with identical train hashes and counters.

**What it does NOT support.** No random-vs-local comparison of rounds or trips: the disk was contended
for the random cell (18.8 ms per block read → a 2 × 1 budget) and quiet for the local ones (1.2 ms →
4 × 3 and 4 × 4), so the derived budgets are not comparable. No tick-time figure: the panel's suites and
the live daemon ran throughout (the same twin read 456 ms/tick in a quiet pilot and 1,490 ms/tick in the
cell). No residency verdict from the allocator readings: the filtered snapshot read 0.946 where the exact
instruments read 1.000, ~24.5× the exact quantity (4.2), and the unfiltered peak separated nothing.

**The boundary, from the panel's own controls** (its refuter for C13, 16,384 units): with a strict-
locality drive whose activated working set is smaller than the graph, exact resident bytes fell to
**0.250** of the twin's, and with a transient drive (tick 0 only) to **0.164** — from never LOADING, not
from eviction. **Resident edge bytes fall exactly when the activated working set is smaller than the graph,
and do not fall at all when sustained drive covers it.** 2.6(d) predicted thrash for the no-locality case;
the measured failure mode is the opposite — total residency with zero evictions and the rounds budget
untouched after tick 2. 2.10's "resident edge bytes fall, measured two ways" is therefore scoped: TRUE on
the 1,024-unit chain fixture (0.857) and on working-set-limited drive (0.250, 0.164); FALSE under
saturating drive at width. **No win or loss is claimed; T068 owns that — and its comparison must include
a sustained-drive cell, where this pager saves nothing, and must read residency from the over-capacity
tick count, never from the capacity argument.**

### 3.6 What this row does NOT claim, and what its completion owes

Restating 2.10: no paging WIN, no tick-time claim, no bytes-fetched/bytes-used figure, no live-tick
charge, no device tier, no prefetch, no persisted-bytes figure for the grower's units. **Owed:** the S6
wire onto the live tick (the pager stays off it under its own `wire` row; of the lake modules it consumes,
only `lake_index` still carries a `wire` row — `lake_segments`, `lake_bridge`, `lake_engine` and
`lake_migrate` had theirs retired on 09-06 — corrected from the first record, 4.2); the cockpit's "paging
binding-width status" row (S6); a paged snapshot for replay (`replay_gate` now REFUSES a masked kernel
instead of silently dropping its `pending` — the format that carries residency is S6's);
the surprise-driven prefetch, re-dated to T067 (2.8); and the null itself, T068's S-R3. Capacity is
honoured, not enforced, under sustained firing (3.2) — T068's comparison must read that from the
capacity-trip counter, never from the capacity argument.

## 4. THE PANEL'S FOLD (2026-09-08)

Three lenses (instrument / policy-honesty / integration) over the frozen head `4d09202`, 40 findings
filed, each attacked by an independent refuter that re-ran its command: **25 stand** (1 BLOCKER, 13
MAJOR after severity corrections, 11 MINOR), 15 refuted. Nothing was pushed before the panel ran. The
fold is two stacked commits — code (`99ed4be`) and this record — and the fold is
**battery-verified only**, stated here rather than implied. Every serious finding was in the arc's OWN
apparatus and records; no red the row pre-registered was found unreachable, and the calm identity, the
dropping twin's shortfall, the budget trip and the budget-blind arm all reproduced to the digit on three
independent runs.

### 4.1 The BLOCKER — the full suite was RED and no gate saw it

`tests/test_ratified_row_gates.py` pins the immortal-tag count at 9; `RETENTION[0x0C]` made it 10 at
`53e4bc3`. Every subset this arc ran was green and the full suite was never run before the panel; the
gate whose census changed (`immortal_retention --assert`) is green and prints "3 of 10" without
complaint. Fixed by bumping the pin and naming `0x0C` (never by deriving the count, which would zero the
detector). **The rule this re-teaches:** a row's own subset is not the gate; the A.4 battery would have
halted here.

### 4.2 Claims corrected downward (each with the refuter's correction, not the filer's)

- **The second residency instrument measured a different quantity** (C2/C14). The filtered
  tracemalloc snapshot read 34.5× the exact edge bytes at the module fixture and 14.3× at width; it
  tracks bundle COUNT (count moved 5.0× at fixed bytes; bytes moved 1.57× at fixed count), and it
  disagreed with ITSELF across process history (0.788 / 0.605 / 0.605 in one process) — the exact
  defect that retired the unfiltered peak it replaced. 3.4(8)'s "the same quantity by a different
  mechanism" was FALSE. Replaced in the fold by a second EXACT reading of the same quantity that fails
  differently (nbytes reached through `by_pre`'s routes rather than `bundles`' values); both must
  agree exactly. The allocator readings stay printed, labelled as footprints, used for nothing.
  **Transferable:** an instrument that replaces a retired one owes the red that retired it.
- **"Exceeded on 8 ticks" was the wrong number** (C3/C15). `capacity_trips` counts ticks on which a
  fetch found too few evictable blocks; residency actually stood above `capacity_blocks = 4` on **37 of
  45 ticks (82 %, peak 7 of 8)** in the calm arm and 16 of 20 in the residency arm, whose 0.857 ratio is
  ONE demoted block of seven. The fold prints peak, time-mean and over-capacity tick counts beside the
  trips in both arms. The row's closing sentence — capacity is honoured, not enforced — was right; the
  number under it was not.
- **The "no locality" attribution was false at six sites** (C9). The calm fixture is a nearest-
  neighbour CHAIN, the most local topology possible; the census had already named the cause two
  sentences earlier — the ring-safety clause under a firing wavefront (30 of 30 refusals inside `tick −
  max_delay`). Locality was never varied in the row. Corrected in the module's print, §3.2/3.3 here,
  tasks.md and the plan's S3 bullet; 2.6(d)'s regime is addressed in 4.3.
- **The rounds double-count does not catch the budget-blind arm** (C4). Both counters read 1 on it;
  what it violates is submission WIDTH (7 blocks in a 2-block round), which nothing measured.
  `blocks_read` was written and never read (C18, P3 shape 4). The fold adds the width instrument
  (per planner round, width ≤ `blocks_per_round`, the pinned load excluded) and gives `blocks_read`
  its consumer; the budget-blind arm now reads a width violation in-command.
- **The lake-digest arm could not go red** (C7): it passed with every store write path removed, and
  the digest glob would read sha256-of-nothing on an empty root. The fold gives it the positive
  control its sibling `immortal_retention.round_trip` already had (a flipped byte on a scratch copy
  moves the digest; a zero-file root is refused). The identity itself was true — the pager writes
  nothing — and remains asserted.
- **The third attribution arm was dropped at build time and 3.4 omitted it** (C6). Built in the
  fold: eviction neutered (`capacity = n_blocks`) reads resident bytes EQUAL to the twin's, the
  incumbent reads below — so the reduction is caused by eviction and nothing else; its page-ins-
  after-warm-up clause does not discriminate on this fixture (both arms read 0) and is printed as such.
- **§3.2 credited one command with two of pytest's readings** (C8): the ring-slot identity and the
  `max_stale_ticks = 0` identity lived only in pytest, and 3.4(3)'s "both in the same command" was
  false. The fold adds both arms to the self-check so the sentences become true.
- **`read_record`'s old-layout diagnostic and a false "asserted in tests" sentence** (C10). Every
  seq of an old whole-table segment was refused (never a wrong record), but only the first with the
  intended message; and the sentence naming two test files that assert the refusal was false — no test
  constructed a whole-table segment. Fold: the diagnostic on the error path, and the test.
- **The pager's dated wire pointed at a row that does not contain the work** (C12). S6's text names
  neither the pager nor paging and its reds do not cover a paged substrate; S6 could land green while
  the pager stays off the tick — the exact shape the dormancy comment block warns of. Amended as a
  VISIBLE dated addition to S6 (the pager wire, the cockpit "paging binding-width status" row, and a
  third red: the live tick over a paged substrate is output-identical to the resident twin), with
  **S-R5** added to Part F.
- **§1 said lake_index's `wire` exemption was retired; it is live** (C11/C25). Four rows were retired
  on 09-06 (lake_engine, lake_migrate, lake_bridge, lake_segments); lake_index's survives T066 because
  the pager is itself off the live tick, and the gate's standing-TODO arm still prints its unmet T039
  bound every run. §2.6(g) and §3.6 carried the mirror error. Corrected here and in the EXEMPT row's
  text; §1–2 stay as written, per convention, with this section as their correction.
- **SYNTHESIS §6 asserted a width-arm measurement that did not exist** (C5). The records commit
  converted a forward promise into a past-tense result. Struck; the T028 re-dating rests on the
  structural argument alone (2.8), which needs no measurement. The width arm's readings are in 3.5.
- **The identity test carried no paging guard** (C17): its four assertions all pass with eviction
  switched off; the guard lived in a sibling with a hand-duplicated config. Folded into one config.
- **3.4(10) answered** (C16): the I001 at `53e4bc3` was REAL and the mechanism commit's "rc 0" was the
  false reading — not from stashing: the test file is byte-identical at `2214dd0` and `53e4bc3`;
  ruff's isort classified the not-yet-existing module as third-party until the file appeared.
- **Grain mismatch accepted silently** (C23): the pager validated every input except the one that
  decides which units the others are about. Fold: refusals ADDED (never moved) for a block_size that
  is not a multiple of 128 and for a store whose partition disagrees; a reopened store is accepted.
- **`replay_gate` accepted a paged kernel and lost its pending** (C20): 262,144 units in `pending`
  vanished across capture/restore with no refusal, against the module's own contract. Unreachable on
  this tree (no caller snapshots a paged kernel) — fold: capture and restore REFUSE a masked kernel,
  naming the S6 work that owes a paged snapshot.
- **A shared key space in the pointer index** (C21/C24): the store's `(0x0C, block_id)` rows and
  `lake_rebuild_gate.build_index`'s provisional `(tag, ordinal)` rows share one table; a rebuild
  clobbers the pointers and `fetch` raises. Not reachable today (no caller rebuilds a real home's
  index; the pager's rows ARE reconstructible by a forward scan — the provisional keying is what does
  not reconstruct them). The store now passes `watermark=None` (2.1's "advanced" was the wrong word
  for a marker the store does not own; a max() fix was refused as unsafe in the direction that
  matters). Owed at S6: teach the rebuild the `0x0C` keying. Stated in the EXEMPT row.
- **`SegmentStore._recover_interrupted_seal` infers a duplicate WAL from payload equality** (C22): a
  pre-existing T005 property, untouched here. Only MULTIPLICITY can be lost, never content, and
  through `UnitBlockStore.build` the only window is a crash between two adjacent statements, where the
  drop restores exactly the state the index describes. Disclosed; the proposed sentinel fix was refused
  by measurement (a surviving WAL after an interrupted rebuild is the worse state).
- **A stale inline comment** on the doc-budget line (C19), fixed.

### 4.3 The width arm, and what it changes in the row's claims

The width arm ran on `4d09202` during the panel (3.5): at 32,768 units under a drive that makes ~99 %
of units fire every tick, **the pager evicted nothing in any cell** — exact resident bytes 1.000 of the
twin's, capacity unenforceable, one forced round at tick 2 and a 0.65–0.70 % spike deficit confined to the
first ticks. The panel's own controls at 16,384 units put the boundary where it is: bytes fall (0.250,
0.164) exactly when the activated working set is smaller than the graph. Consequences folded into the
record: 2.6(d) is CORRECTED (the no-locality failure mode is total residency, not thrash); 2.10's
"resident edge bytes fall" is SCOPED to working-set-limited drive; the "no locality" attribution of the
calm arm's trips is withdrawn (4.2); and T068's null clause inherits two requirements — a sustained-drive
cell and residency read from the over-capacity tick count. The numbers describe the pre-fold mechanism;
the fold changed instruments and refusals, not the residency rule, the planner or the budget.

### 4.4 Corrected disposition

**LANDED, with the claims scoped by measurement.** What stands: both reds reachable and counted
before the mechanism; a calm paged run byte-identical to the resident twin while genuinely paging
(determinism held at thread-pool widths 1/4/8 and across the T = 1..16 sweep, byte-for-byte); the
staleness bound measured, saturated at every setting, and confirmed by an independent timestamp
instrument; S-R1 not satisfiable without the mechanism; resident edge bytes fall exactly when the
activated working set is smaller than the graph — and do NOT fall, at all, when sustained drive covers
it (4.3). What does not stand: any unscoped "resident edge bytes fall" (2.10 is corrected by 4.3); the
second instrument as pre-registered; the "no locality" attribution; the 8-ticks number.

**What the completion owes** (each a row, none silent): the S6 wire with S-R5; the cockpit row; a
paged snapshot for replay (S6); the rebuild's `0x0C` keying (S6); the T067 prefetch; **and the null
itself — T068 must read residency from the over-capacity tick count, never from the capacity argument,
and must include a sustained-drive cell, where this pager saves nothing.**
