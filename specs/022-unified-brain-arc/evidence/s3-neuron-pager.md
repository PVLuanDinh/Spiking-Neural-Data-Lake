# S3 / T066 — activation-driven paging, both reds before the mechanism (2026-09-08)

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
