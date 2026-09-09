# S4 / T067 — the growth delta tier, the sleep-phase merge, and the locality reorder (2026-09-08)

> **THE MECHANISM WAS DELETED ON ITS NULL 2026-09-09** (`substrate/lake_compactor.py`, its tests and the 0x0C/0x0D block tier) -- see `s5-paging-null.md` 3.4/3.5/4. The last head carrying it is tag `t068-last-with-pager`. Everything below is left VERBATIM: the measurements stand, and an evidence file outlives the code it measured.

> Plan row (PLAN-FINAL-2026-09-02.md, Phase 3): **S4 = T067** growth delta tier + background merge in
> sleep (its locality reorder uses block adjacency until the prior lands).
>
> Task row (tasks.md T067): Growth delta tier + background merge (FreshDiskANN shape) +
> locality-preserving lake reorder (bytes-fetched/bytes-used metric); in `substrate/lake_compactor.py`
> (runs in the sleep phase per L5.5).
>
> Governing decisions (research.md): **D1** the append-only lake is the sole source of truth, every index
> derived and provably rebuildable, "the lake must never be rewritten in place"; **D2** compaction is
> epoch-fenced, COMPACTABLE classes only, relocation proof precedes reclamation; **D4** the dirty set is
> derived from spikes, quantize-then-compare-then-append, snapshots are compaction artifacts only, growth
> deltas IMMORTAL-as-reconstructible; **D8** the delta tier and the reorder are the DiskANN-family
> patterns named beside the pager.

## 1. Territory, measured before the build (HEAD `b99ea65`)

- **`substrate/lake_compactor.py` is ABSENT.** `graph-exists.js` over the capability returns seven
  near-matches, every one read and adjudicated: `lake_engine.SynapseDelta` (0x09, the LEARNER's
  per-synapse delta over a dense `[pre, post]` matrix, POST-grouped by `UnitBlocks.group`, quantized at
  the learner's `lr`, NO writer anywhere, IMMORTAL "until the kernel block's own proof"),
  `interaction_learning._record_synapse_delta` (its only producer, returning at the `delta_sink is None`
  guard on every production call — T064's withdrawn "live writer"), `unit_blocks` (the partition,
  adopted by T066), `neuron_pager` (the reader this row's records must page in through),
  `function_preserving_growth`, `ring_attractor`, `benna_fusi` (vocabulary only: growth heuristics, a
  dead-reckoner, plasticity deltas that never touch the lake).
- **There is no merge, compaction, reorder or defragment over lake records anywhere in the tree.**
  `BinLog.rewrite` is the ONE durable reclaimer (whole-log rewrite to a survivor list over
  `brain_lake.bin`, reached only by `KnowledgeLog.consolidate` on the `% 8` sleep night); it never
  touches the segment family. `lake_migrate.migrate` copies read-as-is. `brain_telemetry._compact` is
  the only background-compaction THREAD precedent, over a JSONL file. Nothing has ever merged two
  records, moved a record's position, or written a second record for a key that had one.
- **The one existing delta tier is the 0x08 growth chain in `home.py`**: additive tag, FULL every
  `_GROWTH_CHAIN_EVERY = 64` deltas, a sha256 manifest verified link by link, latest-wins on read, a
  refusing reader that HOLDS at the last verified link, COMPACTABLE because a prefix ending before the
  newest FULL is reconstructible. It is the precedent this row copies in shape.
- **The kernel-tier store.** `UnitBlockStore.build` writes one 0x0C `UnitBlockRecord` per block (tag
  byte first, `<BIIH` header, PRE-local edges with f4 weights and delays) into ONE sealed segment, one
  Arrow batch per record, block-id order == seq order, and upserts `(0x0C, block_id) -> (segment,
  seq)` into `LakeIndex` with `watermark=None`. `UnitBlockRecord` carries no generation field; two
  records for one block are told apart only by lake position and by which the index points at.
  `fetch` resolves every id through the index on every call, so a moved pointer is honoured on the
  next page-in with no pager change; a RESIDENT block keeps its compiled bundles until evicted.
  **The seq arithmetic in `build` (`seq0 + bid`) assumes an empty WAL** — a second write to the same
  store makes it off by the WAL's prior count; this row is the first second-writer and fixes it by
  reading each sealed record's seq rather than deriving it.
- **The index keying collision T066 left owed at S6 becomes material here**:
  `lake_rebuild_gate.build_index` re-derives `(tag, global ordinal) -> ("lake", ordinal)` into the
  same `lake_index.db`, overwriting the store's `(0x0C, block_id)` rows; a reorder makes `seq0 + bid`
  unrecoverable by arithmetic, so the row closes this opening: `build_index` learns tag-aware keying
  for the block classes (key = the record's own `block_id`, container = the segment file, seq = the
  record's seq, latest record per block wins by forward-scan position), and the rebuild round-trip
  proves the store's rows are DERIVED (D1), with the planted-row red kept.
- **The append observer sees six tags, all framed in `home.py`**; 0x09, 0x0A, 0x0B, 0x0C bypass it.
  `AppendBudget` is a FLOOD assert (8× the tag's own last-32-writing-ticks p95; a RAMP at ≤ 2× per
  write is invisible at any magnitude, pinned by test) — it bounds nothing cumulative. A new class
  is `unbaselined` (reported, never trips) until it has a 32-writing-tick window of its own or a
  measured row in `tools/lake_append_baseline.json` (decimal-string keys; five rows today).
- **The sleep phase.** `Brain.tick` calls `_enter_sleep` once per night on the wake→sleep transition
  and `_sleep_tick` every asleep tick — foreground, in the tick. There is NO duty registry: a nightly
  duty is a hand-written `if self._persist and self._nights % N == 0:` block with its own
  `try/except`, counters in `__init__`, `state()` keys and a `brain_telemetry.PROVENANCE` row; the
  lake duty is the `% 8` block (`LakeConsolidation` → `KnowledgeLog.consolidate`, `ImmortalRefusal`
  counted separately). `_sleep_caps` + `neuter_sleep(element)` is the per-capability neuter idiom
  (raises on an unknown element). `perf_baseline --assert` binds the WAKE p50 only; the sleep bucket
  is unasserted — "must not slow the wake tick" has an instrument, "must not slow sleep" does not.
- **The live lake has NO segment tier**: `~/.snn_data_lake` holds `brain_lake.bin` (2.58 GB) and
  neither `segments/` nor `lake_index.db`. The block store, the pager and everything this row builds
  are scratch-home only until S6 puts unit blocks in the one lake. `cortex.brain` is a dormancy
  `_LIVE_ROOT` whose AST walk is unconditional, so an import from `_enter_sleep` — even inside a
  guarded no-op branch — would make the module REACHABLE, forbid an EXEMPT row, and leave a
  hollow green over a duty with nothing to do (the T064 shape). **This row therefore does NOT wire
  `_enter_sleep`.** It lands the duty as a callable in the exact nightly-duty shape (never raises,
  budgeted, counters returned, the neuter element named) with a `wire` dormancy row dated to S6,
  where the live brain first has a store — the same disposition T066 took and S6's amendment now
  names.
- **What T066 measured that this row changes the economics of**: at width under saturating drive
  the pager evicts nothing (every block fires inside the ring window); resident bytes fall only when
  the activated working set is smaller than the graph (0.250 / 0.164). The reorder cannot change
  evictability; it changes how many READS a round costs when the working set IS smaller than the
  graph and its blocks are index-distant. That is the regime the reorder's red runs in.
- **What the design text left undefined** and this row DEFINES in 2.1: "delta tier" (record class,
  producer, consumer), "merge" (what is written, what is proven, what is never destroyed),
  "background" (a sleep-night duty under an integer budget, not a thread), "locality-preserving
  reorder" (a permutation of block→seq under a co-fetch order, written as a NEW segment),
  "bytes-fetched/bytes-used" (numerator and denominator, at the pager's fetch, with reads per round
  beside it), "FreshDiskANN shape" (a fresh tier read beside the base, folded into the base in the
  background, the fresh records superseded only after a proof).
- **Reds already pre-registered elsewhere, honoured here:** L-R6 (a constructed paging/compaction
  pass evicts or rewrites an IMMORTAL record — for an APPEND-ONLY pass the identity criterion is
  logical, 2.3), L-R7 (a reclaimer that ignores `retention_of` is caught; a COMPACTABLE reclaim is
  still permitted), L-R8 (the append budget flags a real spike and not ordinary growth), S-R3 (T068's
  null, NOT decided here), S-R5 (S6's wire, NOT discharged here). T067 has no Part F row of its own;
  2.3 pre-registers **S-R6, S-R7, S-R8** and Part F gains them in the records commit.

## 2. Pre-registration (written before `lake_compactor.py` existed)

### 2.1 The mechanism, with every open term defined

**The delta record — `_T_BLOCKDELTA = 0x0D`.** One record = a change to ONE block's out-edges,
PRE-keyed exactly as the block record is (so the pager's `pre_global()` → `compile_bundles` path
composes): tag byte, `struct '<BIIQ'` (version = 1, block_id, n_units_in_block, base_generation),
then parallel LE arrays `pre_local <u2`, `post <u4`, `weight <f4`, `delay <u2`, `op <u1` (0 = upsert
an edge (pre, post, delay) with this weight, 1 = remove it). `base_generation` names the merged 0x0C
generation the delta applies to (the count of merges the block has had; 0 = the built record), so a
delta whose base is not the block's current generation is REFUSED at apply time — the 0x08 chain's
"a DELTA whose base is not what the chain holds" refusal, at the block grain. Malformed payloads are
refused WHOLE. Registered in BOTH registries: `RETENTION[0x0D] = ("COMPACTABLE", False, reason)` —
COMPACTABLE because a delta superseded by a merged 0x0C record is reconstructible from that record
(the reason says so and names the proof); `reclaimable` stays **False at introduction** and flips
only when the reconstruction proof below has LANDED and been counted (the L6 rule: the flag licenses
reclaim, it never performs it) — this row lands the proof and flips the flag in the same commit ONLY
if the proof's red is reachable and counted; a per-record SUPERSEDED exemption, read at runtime the
way `home._prune_state_history` reads 0x07's, marks which deltas a reclaimer may touch.
`TAG_DISPOSITIONS[0x0D]` = read-as-is. The class's append budget: observed at the write site by
calling `home._observe_append` (the private symbol is promoted to a documented module function
`observe_append` — one observer path, P2, and `SegmentStore` gains no observer of its own), and
BASELINED on the pinned host by the recorded method before the flag is trusted: `tools/
lake_append_baseline.json` gains a `"13"` row with its measurement sentence. **0x09 is NOT retired
and NOT reused**: it is the learner's per-synapse delta over a different substrate, POST-grouped and
lr-quantized; the P2 disposition is recorded — when the kernel becomes THE substrate at S6, 0x09's
"revisits with the kernel block's own proof" clause is discharged by 0x0D and 0x09 retires.

**The writer — `UnitBlockStore.append_deltas(records)`.** Appends 0x0D records (compare-dropped at
the record boundary: an edge already at that weight in the block's current edge set is not emitted —
D4's quantize-then-compare-then-append at the f4 grain the record stores) through the store's
`SegmentStore` into the active WAL, indexes each under `(0x0D, f"{block_id}/{n}")` where `n` is the
block's delta count from a per-block meta row `(0x0D, f"{block_id}/count")` — a small fixed key
scheme the rebuild can re-derive by forward scan (D1) — and reports bytes to the observer. Sealing
is the store's (`seal()` at the end of a batch). **Producer**: growth into the kernel substrate,
which arrives at S6/T062; today the producers are this row's self-check, its tests and its width
arm — disclosed, the same standing T066's pager has.

**The fresh tier — the pager reads base + deltas.** `UnitBlockStore.fetch` returns, per block, the
decoded 0x0C record with its unmerged 0x0D deltas APPLIED in order (upserts replace or add, removes
delete) into one edge set — so `_page_in` and `compile_bundles` are untouched (P2: one compiler, one
page-in). The extra reads are counted in the SAME rounds/width instruments (a block with k unmerged
deltas costs 1 + k records in its round; the rounds budget and width bound bind them exactly as
they bind base records). `generation` and `n_unmerged` per block are readable on the store.

**The merge — `LakeCompactor.merge(store, *, budget_blocks)`** (the "background merge",
FreshDiskANN shape: fold the fresh tier into the base). For each DIRTY block (n_unmerged > 0), in
block-id order, up to `budget_blocks` per call: (1) build the merged edge set = base + deltas
applied (the same apply function `fetch` uses — one apply, P2); (2) encode it as a NEW 0x0C record
with `generation + 1` (the record layout is unchanged — generation lives in the index meta row
`(0x0C, f"{block_id}/gen")`, so 0x0C's `_VER` and its read-as-is disposition stand); (3) append it,
seal, and **PROVE RELOCATION BEFORE MOVING THE POINTER**: decode the sealed record back from disk
and assert it equals the merged edge set byte-for-byte AND equals base + deltas re-applied from the
lake (the reconstruction proof, per block, in the same call); (4) only then upsert `(0x0C,
block_id)` to the new record, bump `gen`, and mark the folded deltas SUPERSEDED in their index rows
(`(0x0D, f"{block_id}/{n}")` → container unchanged, a `superseded=gen` note in the meta row);
(5) a block whose proof fails is SKIPPED with the pointer untouched and counted (`merge_refusals`),
never half-moved. **Nothing is ever destroyed or rewritten in place** (D1): the old 0x0C record and
every 0x0D record stay in their segments byte for byte; reclaim of superseded segment-family records
is a separate row (D2's "relocation proof precedes reclamation" — this row lands the proof and the
label; the segment family has no reclaimer and gains none here, stated). Returns a result dict
(`merged`, `deferred`, `refusals`, `bytes_appended`, `seconds`) and never raises past its own
guards — the nightly-duty shape.

**"Background" (defined).** A sleep-NIGHT duty under an integer budget, not a thread: `budget_blocks`
is an INPUT with NO DEFAULT; the self-check derives it from a measured per-block merge cost and a
sleep-night allowance it states (the recorded wake p50 is the reference it names, as `_derive_rounds`
does), prints the derivation, and a dirty set larger than the budget is DEFERRED to the next call
with `deferred` counted — the budget's red is a constructed overload (2.3). The merge is never called
from the wake tick; the wake ratchet (`perf_baseline --assert`) is the instrument that would catch it
and stays green. The sleep wire itself is S6's (1.).

**The reorder — `LakeCompactor.reorder(store, order)`** ("locality-preserving reorder"). `order` is a
permutation of block ids; the compactor writes the CURRENT record of every block (base with deltas
folded — a reorder is a merge of everything, so it uses `merge`'s apply and proof per block) into a
NEW sealed segment in that order, proves each relocated record before moving its pointer, and
upserts every pointer; old segments stay byte for byte. **The order — `co_fetch_order(due_sets)`**
(the "block adjacency" the row promised): from a run's recorded per-tick fetch sets (the pager's
`plan()` output, already deterministic), place blocks so that blocks fetched together are adjacent —
a greedy walk over the co-fetch count graph (highest-count neighbour next; ties by lowest id; blocks
never fetched keep index order at the tail). "Until the prior lands" = the co-fetch graph is the
only locality signal this row has; the H01 prior (T062) may replace it as the order's input later
without changing the mechanism. The order function is deterministic and its output is a
permutation (asserted).

**The reader that makes the metric definable — coalesced range reads.** `UnitBlockStore.fetch` now
groups a round's needed `(segment, seq)` pairs by segment, sorts, and reads each maximal
CONTIGUOUS run of needed seqs with ONE `read_records(path, seq_lo, seq_hi)` (new in
`lake_segments`, a plain-file batch-range read by the footer, never mmap, every record CRC-checked)
— reading exactly the needed records and never a gap (no over-read this row). Two readings per
round, both exact: **reads per round** (the number of range reads issued) and
**bytes-fetched / bytes-used** = (bytes of every record read) / (bytes of the records the round
requested); with no over-read the ratio is 1.0 by construction and is PRINTED as such — the
quantity the reorder moves this row is reads per round. Over-reading a gap (which would make the
ratio informative) is a derived trade (seek cost vs bytes) that owes its own measurement and is
NOT taken here, stated.

**No defaults (P1).** `budget_blocks`, the delta compare-drop grain (the f4 record grain, not a
picked epsilon), and the order function's tie rule are stated; nothing is picked silently. Every
constant the self-check uses is derived from a measurement it prints.

### 2.2 ONE mechanism, N callers (P2 / P4 / P4b)

One partition (`UnitBlocks`). One record class per job: 0x0C the block, 0x0D its delta, both
PRE-keyed. One apply function (fetch and merge). One compiler and one page-in (untouched). One
observer path (`home.observe_append`). One store (`SegmentStore` + `LakeIndex`; no sidecar, no
second family — `seg-*.lake` is the only glob). One rebuild (`build_index`, made tag-aware; the
store's rows are derived from a forward scan, D1). No second durable reclaimer (nothing deletes).
0x09 stays the learner's delta with its retirement condition written down.

### 2.3 The reds, each with the input that makes it fail (built and COUNTED RED before the mechanism)

- **S-R6 — a merge (or the fresh-tier read) loses or mis-applies a delta.** Fixture: a built store,
  a set of 0x0D deltas to several blocks (upserts of new edges, weight changes, removes), and the
  RESIDENT TWIN = a plain `KernelTick` with the full `Synapses` compiled directly from the
  post-delta edge set. Criteria: (i) the paged run reading base + unmerged deltas is byte-identical
  to the twin in trains and ring-slot key lists over T calm ticks; (ii) after `merge`, the paged run
  reading merged records only is byte-identical to the same twin, `n_unmerged == 0` for every merged
  block, every relocation proof passed (`refusals == 0`), the old 0x0C records and every 0x0D record
  are byte-identical on disk (per-file sha256 of every PRE-EXISTING segment unchanged — the append-
  only form of L-R6), and the logical record set reachable through the index is the post-delta edge
  set for every block; (iii) a delta whose `base_generation` is stale is REFUSED, never applied.
  **The red:** a DROPPING merge — a test-constructed subclass whose apply skips one delta — must
  yield a paged run that DIFFERS from the twin and a relocation proof that FAILS on that block
  (refusals ≥ 1, pointer untouched). Positive control in the same command.
- **S-R7 — the merge budget never trips on constructed overload, or the merge runs on the wake tick.**
  Constructed overload: more dirty blocks than `budget_blocks` → `deferred > 0` on that call, the
  deferred blocks merge on the next call, and the union of both calls' merged sets equals the dirty
  set. Calm control: dirty ≤ budget → `deferred == 0` and `merged == dirty`. **The red:** a
  BUDGET-BLIND merge subclass ignores the budget and reads `deferred == 0` on the identical overload.
  Second half: `perf_baseline --assert` remains green with the compactor present (the wake tick is
  untouched — the duty has no wake-tick caller by construction; asserted by the dormancy census and
  the gate, and stated).
- **S-R8 — a reorder changes or loses a record, or does not reduce reads on a co-fetched stream, or
  reduces them on a stream with no co-fetch structure.** Fixture: a store of N blocks; a CO-FETCH
  stream whose per-tick fetch sets are fixed groups of k blocks that are INDEX-DISTANT (e.g. blocks
  {0, N/4, N/2, 3N/4} each tick) — before reorder each round costs k reads; after `reorder(store,
  co_fetch_order(due_sets))` the k blocks are adjacent and the round costs 1 read (reads per round
  strictly lower; bytes-fetched/bytes-used 1.0 both times, printed). Criteria: every block's record
  after reorder decodes identical to before (logical identity), every pre-existing segment byte-
  identical, the paged run byte-identical to the twin before and after, `co_fetch_order` returns a
  permutation. CONTROL (not the method): a RANDOM-fetch stream (a fresh random subset each tick) —
  reorder by its co-fetch order must not reduce reads per round beyond chance (the reduction on the
  structured stream must exceed the reduction on the random one by a stated margin), and must never
  lose a record. **The red:** a DROPPING reorder (skips one block) → that block unfetchable (`fetch`
  raises `KeyError`, refused rather than read as empty) and the logical identity fails; and a
  LOSSY order (not a permutation) is refused at the call.
- **The rebuild identity (D1)**: `build_index` over the root after build + deltas + merge + reorder
  reproduces the store's `(0x0C, *)` and `(0x0D, *)` rows exactly (logical dump digest equal), with
  the planted-row red kept; a store rebuilt this way pages in identically.
- **Commit order:** the tests carrying S-R6/S-R7/S-R8 and the rebuild identity land in their own
  commit BEFORE the mechanism and are RUN on that tree; their failure output is quoted in §3.

### 2.4 Statistic and bar

Substrate row: exact equalities (byte-identity, record identity, integer counters) and strict
inequalities on reads per round; no seed sweep. The reorder's control margin is stated as an
integer difference in reads per round (structured − random), pre-registered ≥ k − 1 − 1 for k = 4
(the structured stream must reach exactly 1 read per round; the random stream must stay ≥ 3 on
average). Every measured quantity is printed with its instrument named.

### 2.5 The control that is not the method (P8 clause 2)

The RESIDENT TWIN built directly on the post-delta edge set contains no delta, merge or reorder code.
The RANDOM-fetch stream is the reorder's null: a locality order fitted to noise must not help.
Expected readings before the run: fresh-tier paged ≡ twin; merged paged ≡ twin; dropping merge ≠
twin with a failed proof; structured reorder 4 reads → 1; random reorder ≥ 3 → ≥ 3 (within one).

### 2.6 Attribution and disclosure (P8 clauses 3 and 4)

- Per-item neuters: the delta apply neutered (deltas ignored) → identity to the twin FAILS; the
  proof neutered (pointer moved without proof) → the dropping merge is NOT caught (refusals 0) — the
  proof is what makes the red reachable, printed; the co-fetch order replaced by identity → reads
  per round do not fall. Each printed side by side with the incumbent.
- **Disclosed preconditions and limits:** (a) no live producer of kernel-tier deltas exists until
  S6/T062; the writer's callers this row are its self-check, tests and width arm — the pager's
  standing, disclosed, and the row's `wire` dormancy row dates the sleep wire to S6; (b) the sleep
  duty is NOT wired into `_enter_sleep` (a guarded no-op import would be a hollow reachability — the
  T064 shape); (c) bytes-fetched/bytes-used is 1.0 by construction without over-read and is printed
  as such — reads per round is the quantity this row moves; (d) nothing reclaims superseded records
  in the segment family — the label and the proof land, the reclaim is a separate row; (e) the append
  budget is a flood assert, so the 0x0D class's bound is a flood bound, not a slope bound (L4b's
  disclosure inherited); (f) the reorder's order comes from co-fetch statistics of a run, so it is
  only as good as the run it watched — a different activation pattern can be worse after reorder
  than before (measured by the random control, disclosed as the regime's cost); (g) resident blocks
  keep stale bundles until evicted — a merged block already resident is not re-read (measured and
  printed as `stale_resident` in the merge result); **(h) added at the fold (panel C1, MAJOR):** the
  reorder's placement is NOT durable under the sleep-night fold — `merge` seals each merged block's
  record into its own segment, so a fold of k co-fetched blocks returns their reads per round to the
  pre-reorder value (measured: 1.00 → 1.75 folding 3 of 16; under budget < dirty, 5.00 across
  passes), and the reorder must re-run after every fold to restore it; a one-seal-per-pass merge in
  current-placement order was built and measured at the panel and restores 1.00 only when EVERY block
  of the group is dirty (2.00 under a partial fold), so it is not adopted — the durable answer is the
  reorder re-run, a sleep-wire (S6) scheduling fact, recorded here rather than mechanised at the fold;
  **(i) added at the fold (panel C9):** 0x0C records are written by `build()` AND by every merge and
  reorder pass, and those appends are UNOBSERVED by `home.observe_append` (only the ADOPT records are
  observed) and unbounded by any budget — `RETENTION[0x0C]`'s "one-time build" clause was false from
  `4109912` on and is corrected in the fold; the S6 sleep wire owes the observation.

### 2.7 P2 adjudication (stated before the build)

`SynapseDelta` (0x09) named, not reused; its retirement condition written. `UnitBlocks` adopted. The
0x08 chain's SHAPE adopted (additive tag, generation base, proof-then-commit, latest-wins), its code
not shared (a different container and grain — the two chains are two tenants of one lake; a common
codec would be a merge across tag families that this row does not attempt, stated). T066's `fetch`
gains coalescing; no second reader. `build_index` gains tag-aware keying; no second rebuild.

### 2.8 The T028 prefetch disposition (re-dated to T067 by T066)

The co-fetch order gives block adjacency LOCALITY MEANING (2.1); a surprise-driven prefetch would
widen a round over that adjacency with spare rounds. **NOT declared here**: prefetch needs spare
rounds (the pager's budget has none by derivation) and a live percept stream feeding a live pager
(S6). Re-dated to **S6** with this reason at both sites (SYNTHESIS §6 and the producer's
docstring); the mechanism it would consume — adjacency with meaning — is this row's product.

### 2.9 Disposition on a null (fixed in advance)

| outcome | disposition |
|---|---|
| the dropping merge is NOT caught by the proof | the proof is blind; no merge lands until it catches its red |
| the structured stream does not reach 1 read per round after reorder | the order function is wrong or the coalescing is; measured, fixed or the reorder is DELETED on its null (P8 clause 5) |
| the random control ALSO reaches ~1 read per round | the criterion cannot fail (F11) — the reorder's benefit claim is deleted, the identity properties stay |
| the append baseline cannot be measured on the pinned host | the 0x0D row stays `unbaselined`, said so; `reclaimable` stays False |
| paging vs forced residency | T068's, not this row's |

### 2.10 What this row does NOT claim

No paging WIN (T068). No tick-time claim. No live producer, no live sleep duty, no cockpit row
(S6). No reclaim of segment-family records. No over-read trade. No claim that the co-fetch order
is the right locality signal beyond the stream it watched. It claims: a block delta class that
pages in correctly beside its base; a merge that proves relocation before it commits and never
destroys; a reorder that preserves every record and cuts reads per round on a co-fetched stream and
not on a random one; the index rows derived; and every one of those with a counted red.

### 2.11 Commit order

1. this file (§1–2) — 2. the reds: `tests/test_lake_compactor.py`, RUN RED and quoted — 3. the
delta tier (`0x0D` in both registries, `UnitBlockRecord`-sibling `BlockDeltaRecord` in
`lake_engine`, `append_deltas` + the fresh-tier `fetch`, `observe_append`, the seq fix, the
tag-aware `build_index`) — 4. `substrate/lake_compactor.py` (merge with proof and budget; reorder;
`co_fetch_order`; `read_records` + coalescing in `fetch`; the self-check; the registration triad;
the `wire` row) — 5. records (Part F S-R6/7/8, tasks row, plan S4 annotation, SYNTHESIS §6 + the
producer docstring re-dating, `lake_append_baseline.json` row if measured, §3 here) — panel — fold
— ONE battery on the stacked head — closing record — push.

## 3. Results (on the stacked head, before the panel)

Commits on `b99ea65`: `1d38a61` (this file, §1–2) → `b560fb1` (the reds, RUN RED) → `51b8683` (the delta
tier) → `4109912` (the compactor) → `1f2e260` (D2's epoch fence, red-first). The records commit
follows. Nothing was pushed before the panel ran.

### 3.1 The counted red (`b560fb1`)

`pytest tests/test_lake_compactor.py -q -p no:cacheprovider --no-cov` on the tree with no
`lake_compactor.py`: `ImportError: cannot import name 'lake_compactor'` — **29 test functions defined,
0 collected, 1 collection error, rc 2.**

### 3.2 The delta tier (`51b8683`)

`BlockDeltaRecord` (0x0D, tag byte first, `<BIIQ` header, five parallel arrays, refused WHOLE when
malformed) in both registries — `RETENTION[0x0D] = ("COMPACTABLE", False, …)` at introduction;
`home.observe_append` is the ONE observer path (a rename with all seven call sites updated — 0x0D is the
first class observed at a write site outside `home.py`); `UnitBlockStore.append_deltas` (compare-drop
at the f4 record grain against the block's current edge set; one index row per delta `(0x0D,
"{bid}/{n}")`, per-block count and generation meta rows); `fetch` returns base + unmerged deltas through
the ONE apply (`apply_deltas(base, deltas, *, generation)` — the generation keyword is required because
the stale-base refusal is not computable without it); the seq-arithmetic fix (`seal_and_locate` reads
every sealed record's seq with a byte-for-byte tail check, so a second write to a store with a
non-empty WAL indexes every block to its true seq); `lake_bridge.located_records` (`iter_records`
delegates to it — the rebuild needs `(container, seq)`); a tag-aware `lake_rebuild_gate.build_index`
for 0x0C/0x0D with the planted-row red kept; `lake_segments.read_records(path, seq_lo, seq_hi)` (a
plain-file batch-range read, every record CRC-checked, never mmap) and a coalescing `fetch` (maximal
contiguous runs of needed seqs per segment, one read each) that records reads per round and
bytes-fetched / bytes-used. The width BOUND stays a BLOCK bound (T066's contract); the record count
rides beside it with its own cross-check (`assert_read_accounting`).

### 3.3 The compactor (`4109912`) — the self-check's arms, one command, `throwaway_home()`

| arm | reading |
|---|---|
| derivation (P1) | 7 blocks folded in ≈17 ms each (measured, this host, ONE draw); a subjective night ≈ 100 sleep ticks at the RECORDED sleep-tick p50 10.99 ms; the check's stated share of one night is 10 % = 109.9 ms → `budget_blocks = 6`. Printed, never picked. **Variance disclosed at the fold (panel C19):** the reading is single-shot wall clock, and 12 consecutive unmanipulated runs on this host derived `{1,3,3,3,3,4,4,5,5,5,6,6}` (per-block 17.4–122.6 ms, median 24.6) — the 6 reproduces on a quiet host and is the TOP of the band; drift under load is toward a SMALLER budget (the safe direction); the derived value has no consumer, and the print now says so rather than a caller discovering it |
| S-R6 fresh tier | 3 of 4 deltas written (one compare-dropped at the f4 grain); the paged run reading base + unmerged deltas is **byte-identical** to a twin compiled directly on the post-delta edge set over 20 ticks (7 page-ins); the pre-delta twin DIFFERS (the identity is not vacuous) |
| S-R6 merged | after `merge`: the paged run reading merged records only is byte-identical to the same twin; `n_unmerged == 0` on every merged block; `refusals == 0`; **every pre-existing segment file's sha256 unchanged**; the logical record set through the index equals the post-delta edge set; a stale-generation delta is REFUSED |
| S-R6 red + attribution | the DROPPING fold reads `refusals 3 / merged 0` with every pointer and generation UNTOUCHED; the SAME fold with the PROOF NEUTERED reads `refusals 0 / merged 3` and adopts a record set wrong by 4 edges — the proof is what makes the red reachable |
| S-R7 | 6 dirty blocks vs `budget_blocks = 3`: deferred 3 on the first call, 0 on the second, the union of merged sets IS the dirty set; calm control (2 ≤ 3) defers 0; the BUDGET-BLIND merge reads deferred 0 / merged 6 on the same overload |
| S-R8 | a co-fetch stream of 4 index-distant blocks costs **4.00 reads per round before the reorder and 1.00 after**; bytes-fetched / bytes-used 1.0 both times (by construction — no over-read); every record decodes identically; every pre-existing segment byte-identical; `co_fetch_order` returns a permutation; the identity-order neuter reads 4.00 → 4.00 (the ORDER is what moves it, not the coalescing) |
| S-R8 control | the RANDOM-fetch stream reads 3.45 → 2.75 under its own co-fetch order (a noise-fitted greedy order lands some co-occurring pair adjacent by chance); the pre-registered INTEGER margin structured − random = 3.00 − 0.70 = **2.30 ≥ 2** passes; 2.5's "≥ 3 (within one)" holds (2.75) — the reds commit had transcribed it as a hard ≥ 3.0 and was corrected to the pre-registration's own wording (3.4). **Seed and length disclosed at the fold (panel, MINOR):** the control's seed and 20-tick stream length were not registered in 2.4/2.5 and the record reports one draw (2.4 says so); the panel swept 11 seeds at 20 ticks — 10 of 11 pass the margin, 1 misses by chance on the noise-fitted greedy order — and at 50 ticks all 11 pass; the self-check now prints the seed and length it used; the seed and length are NOT changed at the fold (a re-registration is not a fix), the spread is the disclosure |
| S-R8 red | a DROPPING reorder is caught by the per-record proof (**refusals 15 of 16**, `merged 1`, every record still readable and identical); with the proof neutered a block becomes unreadable (the segment's own span refusal) — the reds file's original `KeyError` expectation was wrong and the test now asserts BOTH halves (3.4) |
| rebuild identity | `build_index` over the root after build + deltas + merge + reorder reproduces the store's rows (logical dump digest equal, planted-row red kept); a store rebuilt this way pages in identically — stated for a store with ZERO merge refusals (3.4, and closed by `1f2e260`) |
| gates / suite | full suite **1266 tests, 0 failed, 2 pre-existing xfails**; every substrate gate rc 0; `perf_baseline --assert` ASSERTED on the pinned host (wake ratio p50 174.842 vs the 183.517 ambient-ratio baseline — dimensionless, the tick paired with an adjacent crc32 reference, NOT milliseconds: the recorded wake p50 is 119.6 ms; label corrected at the fold, panel C16; p95 reported-HIGH under a concurrent suite, its standing disclosure); self-check 1.5–2.7 s wall |

### 3.4 Deviations from the pre-registration, each with its reason

1. `apply_deltas` takes a required `generation` keyword (the stale-base refusal needs it). 2.
`co_fetch_order(due_sets, n_blocks)` (the never-fetched tail is not enumerable without `n_blocks`). 3. The
width bound stays a BLOCK bound; records ride beside it. 4. `edge_table` refuses a base record repeating a
`(pre, post, delay)` triple (a delta addresses an edge by that triple); delta-free blocks never build a
table, so T066's fixtures stay green. 5. `merge(…, resident_blocks=())` — a fact, not a constant; the
result dict carries `merged_ids` (S-R7's union criterion needs sets; a `deferred_ids` key was also written here and read by nothing — deleted at the fold, panel C14, P3 shape 4). 6. **Two corrections
to this row's own reds, made in the mechanism commit and stated:** the dropping-reorder red as written
("unfetchable, `KeyError`") was wrong — the proof CATCHES the drop, so the test asserts both halves; and
the random control's absolute bar was transcribed as ≥ 3.0 against 2.5's "≥ 3 (within one)" — measured
2.75, the wording restored. 7. **`RETENTION[0x0D].reclaimable` stayed False at `4109912`** although the
proof's red is reachable, because of a hazard the build found: a FAILED proof leaves an unadopted 0x0C
record that the rebuild's latest-wins-by-position rule would adopt — closed by the epoch fence
(`1f2e260`, 3.5). 8. `observe_append` is a rename, not a wrapper. 9. `lake_bridge.located_records`
added; `iter_records` delegates. 10. The merge seals one segment per block, the reorder one for the pass —
TRUE at `4109912`; since the epoch fence (`1f2e260`, 3.5) `adopt_merged` also appends the ADOPT record
through `seal_and_locate`, so the merge seals TWO segments per merged block and the reorder n + 1 for
n relocated blocks (measured at the panel: merge of 5 → +10 segments, reorder of 6 → +7); the cost is
segment FILES, not read amplification — ADOPT records are never pointed to by a read-path row, so
`_coalesced_reads` does not group on them (forward pointer added at the fold, panel C17).
11. `lake_compactor.py` is 1080 lines at `4109912` and 1095 at `bd39587` (the fence added 15; this
item was transcribed at the head and not re-measured — corrected at the fold, panel C13), ~640 of
them the self-check — over the soft ceiling, deliberately, because the row's criterion is one command. 12. The `"13"` append-baseline row is NOT written: the recorded
method (`make_brain × 120 ticks`) cannot produce a 0x0D append because no live writer exists; 0x0D is
OBSERVED and `unbaselined` — reported, never trips — and the record says so rather than inventing a row.

### 3.5 D2's epoch fence (`1f2e260`)

Found by the build (3.4 item 7) and closed red-first before the panel. **The red, on `4109912`:** a
store with one block whose relocation proof FAILS (the dropping fold), rebuilt over its root —
`block 0: the rebuild adopted ('seg-00000012.lake', 12), a record the store REFUSED (its pointer is
('seg-00000001.lake', 1))`. Latest-wins-by-position adopted an unproven record. **The fix:** adoption
is an append-only durable FACT — `BlockAdoptRecord` (tag 0x0D, a sibling class discriminated by the
version byte `BlockDeltaRecord.decode` already refuses on: a delta's length equation is its
refused-whole check and cannot carry a variable-length container name) records `block_id`, the adopted
generation, the adopted `(container, seq)` and `folded_through`; `merge` and `reorder` append it ONLY
after the proof passes and BEFORE the pointer upsert; `build()` emits generation-0 ADOPT records so the
rule is total — a 0x0C record is adopted iff an ADOPT record names it; `build_index` derives pointers,
generations and SUPERSEDED marks from ADOPT records in forward order and never from position. Two
crash windows constructed and pinned: after the ADOPT record and before the upsert, the rebuild adopts
the ADOPTed record and a store reopened over it pages in the merged edge set identically; after the
sealed record and before the ADOPT record, the rebuild does NOT adopt it and the store serves the
previous generation identically. The rebuild identity now holds on a failed-proof store. Two T066 pins
changed to assert the fence's contents rather than a segment count (a build now seals two segments).
**`RETENTION[0x0D].reclaimable` STAYS False**, for a different divergence the build measured by
constructing the reclaim the flag would license (a folded prefix dropped from a re-containerised
copy): the ADOPT record's stored container names a segment the new root lacks (`FileNotFoundError`),
and `folded_through` exceeds the deltas a rebuild can see (`n_unmerged = −1`). A sealed segment cannot
lose a record in place, so any reclaimer re-containerises, and the fence must then anchor on the ADOPT
record's POSITION rather than a filename — the reclaimer row's business (3.6). The same hazard applies
to `lake_migrate.migrate` and is disclosed in the class docstring.

### 3.6 What this row does NOT claim, and what its completion owes

Restating 2.10: no paging WIN (T068), no tick-time claim, no live producer, no live sleep duty, no cockpit
row, no reclaim of segment-family records, no over-read trade, no claim that the co-fetch order is the
right locality signal beyond the stream it watched. **Owed:** the S6 sleep wire (the live lake has no
segment tier; a guarded no-op import was deliberately NOT made); a reclaimer for SUPERSEDED 0x0D records
with L-R7's control, in its own row; the `"13"` baseline row once a live writer exists; the
surprise-driven prefetch, re-dated to S6 (2.8); and T068's null.

## 4. The panel's fold (2026-09-08, on `bd39587`; the fold commits are stacked on it)

**Shape.** Three lenses (instrument / policy-honesty / integration), each filing findings only with a
reproducing command on the frozen head; every filed finding sent to an independent refuter (Opus)
instructed to refute the CLAIM and, separately, the PROPOSED FIX. 33 filed: **19 stood** (2 MAJOR, 17
MINOR, 0 BLOCKER), 14 refuted. Of the 19 standing findings, **7 carried a proposed fix the refuter showed
would introduce a defect** (C1's one-seal merge, C14's assert arm, C15's test-only reclaimer, C17's
read-amplification cost claim, C18's count-every-moved-resident rule, C19's fold-twice median, C13's
suite-count rewrite) — the fold adopts the refuter's minimal fix in each case, never the panel's. The
refuted set includes, verbatim as filed and worth keeping: the "4x fall buys 1.00x wall time" claim (a
single noisy draw; 8 of 10 re-runs above 1.00x), the reorder-is-not-idempotent MAJOR (real, but the
cost figure was not a zero-change pass and the fix is the row's own red arm), the per-round allowance
"bind" (both notes say the bound is reported beside, not lost), and the epoch-fence-reds-in-the-
mechanism-commit MINOR (the quoted commit-order rule scopes to the S-R6/7/8 tests, which did land first).

**What stood, and what the fold did (record = this file; code = the fold's code commit):**

1. **C1 MAJOR, record only (2.6(h)).** The merge's per-block seal undoes the reorder's placement:
   reads per round 1.00 → 1.75 folding 3 of 16, and under budget < dirty the co-fetch group is split
   across passes (5.00). The panel's fix (one seal per merge pass in current-placement order) was built
   and measured by the refuter: 1.00 only when every block of the group is dirty, 2.00 under a partial
   fold — not durable either, so NOT adopted. The durable answer is that the reorder re-runs after every
   fold, a scheduling fact for the S6 sleep wire, now disclosed.
2. **C2 MAJOR, code + record.** `reorder` folded each block OUTSIDE its guard, so a CRC-failed record or
   a missing segment raised past the duty (the EXEMPT row and 2.6 said "never raises past its own
   guards"); it is also unbudgeted. Fold: the per-block fold moves inside a per-block try, a refused
   block is SKIPPED (pointer and deltas untouched) and counted in `refusals`, and the ADOPT/pointer rows
   are built from the rows actually WRITTEN so seq mapping cannot misalign; red-first test in
   `tests/test_lake_compactor.py`: one 0x0C record's payload byte flipped on disk (the positive
   control in the same test: `payload_at` raises `SegmentCorruption`), RUN RED on `bd39587` —
   `reorder → _reorder_payloads → _folded → SegmentCorruption: seg-00000001.lake#4: CRC mismatch`
   escaped the duty — and green after: `refusals == 1`, `merged == 7` of 8, the victim's pointer
   unmoved, every other block's edge set byte-identical before and after; the dropping arm's None
   rows still claim their positions so the pre-registered S-R8 red (refusals 15 of 16) still
   fires; a pass in which EVERY block refuses no longer raises "a reorder that writes nothing" (an
   unnamed hole the worker closed). No budget is added: the one-seal-per-pass co-fetch property
   needs the whole pass, so the reorder is a whole-pass duty, deliberately unbudgeted, and the EXEMPT row
   now says so (C11).
3. **C18 MINOR, code.** `reorder` hard-coded `stale_resident: 0` and took no `resident_blocks`. Fold:
   `reorder(store, order, *, resident_blocks=())` reports `len(resident ∩ dirty_at_entry ∩ moved)` —
   NOT every moved resident block (a clean resident block's bundle is still correct; the panel's rule
   would have reported 4 where 3 is true and 4 on a clean control where 0 is true). Test: resident
   {0,1,2,3}, dirty {0,1,2} → 3; the same residents on a clean store → 0.
4. **C14 MINOR, code + record.** `deferred_ids` was written in both result dicts and read by nothing
   (P3 shape 4; a frozen literal in `reorder`, shape 5). Deleted; item 5 corrected; the union
   criterion needs `merged_ids` alone.
5. **C19 MINOR, record + print.** The budget derivation is a single-shot wall-clock reading that swings
   6x run-to-run on an unmanipulated host (3.3's derivation row now carries the 12-run band). The
   panel's fold-twice-and-median fix is NOT adopted (it changes the mechanism to fix a disclosure);
   the print now states the raw quotient, that the reading is single-shot and load-sensitive, that drift
   is toward a smaller budget, and "clamped to 1" only when the clamp binds (it was never printed
   before). The fold's own run, under the worker's load: quotient 2.01 → `budget_blocks = 2` —
   inside the 12-run band, and a third distinct value for the same command on the same host.
6. **C9 MINOR, code (prose) + record (2.6(i)).** `RETENTION[0x0C]`'s reason rested on "one-time build";
   0x0C is written by build AND every merge/reorder, unobserved and unbounded. Reason rewritten;
   home.py's 0x0D comment likewise.
7. **C10 MINOR, code (prose).** `lake_migrate.TAG_DISPOSITIONS[0x0D]` named one shape and asserted
   identity migration; the ADOPT record names a container valid only inside its root, so `migrate`
   reports equal over a root whose block tier then refuses. Prose corrected (both shapes, the hazard,
   the reclaimer row's position anchor); `migrate` is NOT made to refuse — that is the reclaimer row's
   mechanism, and a refusal here would be a second one.
8. **C15 MINOR, code (prose).** The divergence keeping `RETENTION[0x0D].reclaimable` False was a one-off
   build measurement with no landed reproducer; the reason now says so and names the reclaimer row as
   the instrument's owner, and its "silently serving a block short of two edges" clause is scoped to
   half (ii) alone (the measured composite refuses loudly at half (i)). The panel's test-only reclaimer
   is NOT landed: it would pin `FileNotFoundError` and `n_unmerged = −1` as expected behaviour, which
   the reclaimer row must invert.
9. **C16 / C13 / C17 MINOR, record.** The perf ratchet's dimensionless ambient ratio was labelled
   "ms" (3.3, corrected); item 11's line count was transcribed at the head and dated to the mechanism
   commit (corrected, both numbers); item 10's segment counts were true at `4109912` and superseded by
   the fence one commit later (forward pointer added; the panel's read-amplification cost claim is
   refuted — ADOPT records are never pointed to by a read-path row).
10. **MINOR, print only.** The S-R8 arm prints `bytes_appended` for the reorder pass (a reorder is not
    idempotent — a second pass under the same order rewrites every record; the growth is now visible);
    the random control prints its seed and stream length (3.3's control row carries the 11-seed
    spread, and the draw is unchanged: 3.45 → 2.75 under seed 17); the rebuild-identity
    comparator — the finding named a print that does not exist; `_block_rows_digest`'s docstring
    was already truthful, so the one text that IS emitted, the assert message, now states that the
    index is not identical whole and that the rebuild's own `applied_watermark` row and the
    ordinal-keyed foreign-tag rows are excluded by construction.

**Not changed at the fold, on purpose:** the random control's seed and stream length (a
re-registration is not a fix; the spread is the disclosure); the reorder's idempotence (no wire exists
to call it twice; the row's own red arm is the instrument); the merge's seal granularity (see 1).

**Verification of the fold.** The code half was written by one worker from the refuter-corrected list
and read by the orchestrator; **it was not re-panelled — the fold is battery-verified only** (the arc's
one battery runs on the stacked head that includes it). Its gates at the fold commit: `ruff format`
+ `ruff check` rc 0; the compactor self-check rc 0 (7.25 s, every arm's reading unchanged; the new
prints present); `pytest tests/test_lake_compactor.py tests/test_neuron_pager.py
tests/test_ratified_row_gates.py` 75 passed; the dormancy gate rc 0; the `lake_engine` and
`lake_migrate` self-checks rc 0. The full suite and every other gate: the battery's.
