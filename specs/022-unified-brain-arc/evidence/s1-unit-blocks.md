# S1 / T064 — unit blocks, block-aligned records, and the first live writer of `SynapseDelta` (2026-09-07)

The row (plan Phase 3): *"**S1 = T064** unit blocks + block-aligned records
(`substrate/lake_engine.py`, new `substrate/unit_blocks.py`), the first live writer of
`SynapseDelta` (rewiring becomes lake-resident before unit growth does); port
`neuro_chip_stream`'s one-core-resident shard discipline (P2)."*

## 1. Territory, measured before the build

- **`SynapseDelta` (tag `0x09`) is complete and has NO live writer.** `git grep` finds it referenced
  only from `lake_engine.py` (its definition), `lake_migrate.py` and `replay_gate.py` (both naming
  the tag), and its own test. That matches L6's census: `0x09` is one of four declared lake classes
  that **have never held a record**. It is built and uncharged — P3 shape 1, "method never called".
- **The tree already anticipates this row.** `RETENTION[0x09]` reads: *"Append budget: unbaselined
  (reported, never trips) **until a live writer exists to measure**"*. Landing the writer is what
  makes that budget measurable.
- **`substrate/unit_blocks.py` is ABSENT.** `graph-exists` returns 20 term-overlap matches and none
  is a unit-block partition: `neurogenesis_engine.placement` is per-unit (depth, layer, sign)
  metadata with no blocks or residency; `learning_progress.GoalStore` is `_T_GOAL` event rows with a
  cosine cache; `arc_solver._unscale` is ARC grid geometry. All three read.
- **The live weight-update site is `cortex/interaction_learning.py::SpikingInteractionLearner._deposit`**,
  where `n.w[i]` moves per-synapse for one label neuron `li`. The natural `[pre, post]` mapping is
  pre = input synapse index, post = label index — and it is genuinely SPIKE-DERIVED, which is the
  precondition `SynapseDelta` is built on: only the deposited neuron and the inputs that carried
  spikes are candidates.
- **The P2 source: `neuro_chip_stream.StreamingChip`** persists one shard per core and proves
  `max_resident_cores == 1` in its own self-check. That discipline — not its code, which is
  chip-specific — is what this row ports.

## 2. Pre-registration (written before `unit_blocks.py` existed)

### 2.1 The mechanism

`UnitBlocks(n_units, block_size)` partitions units into fixed-size blocks and frames records
block-aligned. **`block_size` has NO default (P1)**, for the same reason `SynapseDelta.q_step` has
none: the right block is a property of the caller's state-per-unit and its memory budget, and
T065 is the row that measures state-per-unit. A class that picked one here would be choosing the
number the next row exists to derive.

The ported discipline: `resident(block)` is a context manager, and the class tracks
`max_resident` so **exactly one block is resident at a time** — the property
`neuro_chip_stream` proves for cores, asserted here for blocks in the same shape.

### 2.2 The reds, each with the input that makes it fail

- **R-S1a — one block resident.** A traversal over every block must leave `max_resident == 1`. The
  red: a deliberately nested residency raises, and the self-check constructs it. A counter that only
  ever sees 1 because nothing ever nests is not evidence, so the nesting attempt is made.
- **R-S1b — block alignment is a partition.** Every unit belongs to exactly one block, blocks tile
  `range(n_units)` with no gap and no overlap, and the last block is short rather than padded. The
  red: an entry outside `[0, n_units)` is refused, not silently bucketed.
- **R-S1c — the frozen-substrate red, inherited and re-run at the block layer.** A byte-frozen weight
  matrix whose endpoints keep firing must produce **zero** grouped records — the 97.64 %-repeated
  snapshot pathology refused at the door. The positive control in the same command: a single changed
  synapse produces exactly one entry, in exactly one block.
- **R-S1d — the live writer charges the tag.** `0x09` has never held a record. After the writer runs
  on a real deposit, it holds records whose replay reconstructs the weights that moved. The control:
  a deposit that moves no weight past `q_step` writes nothing.

### 2.3 Disposition on a null

If block-aligned grouping cannot be shown to change anything a flat record does not already do, the
blocks are DELETED and `SynapseDelta` is wired flat — the row's value is the live writer, and the
blocking is only justified if it is load-bearing for residency.

### 2.4 What this row does NOT claim

It does not claim a paging benefit (that is S3/T066 with its own reds), it does not baseline the
append budget (a measurement the writer makes POSSIBLE, taken in its own row), and it does not touch
unit growth — rewiring becomes lake-resident here, growth does not.

## 3. THE PANEL'S FOLD — T064 is HALF-LANDED, and the "first live writer" claim is WITHDRAWN

Three lenses on the frozen arc. The verdict on this row is substantially negative and the claims are
corrected downward rather than defended. **Nothing was pushed before the panel ran.**

### 3.1 WITHDRAWN — this is not a live writer, and 0x09 is still uncharged

`delta_sink` is set to a non-None value in **exactly one place in the tree: the tests.** No
production path attaches one, so `_record_synapse_delta` returns at its first line on every real
deposit. Worse, and found by the panel rather than by me: **nothing in the tree ever emits the byte
`0x09`.** The method produces a *payload* and hands it to a callable; it never writes a TAGGED lake
record, so even with a sink attached the tag would still hold nothing.

`SynapseDelta` therefore remains exactly what L6's census found it to be — declared, tested, and
never written. The commit's headline "the FIRST LIVE WRITER of SynapseDelta" is **false as
committed**, and is withdrawn here. What landed is a writer that is *capable* of being live: P3
shape 7, the registry nobody registered with.

**The dormancy green is hollow too.** The gate is satisfied by an `import` statement that sits
behind the `sink is None` early return and never executes in production — a static AST walk cannot
tell that branch from a live one. The module passes the gate while never being imported at runtime.

### 3.2 WITHDRAWN — `max_resident == 1` does not measure peak resident state

R-S1a asserted the ported one-resident discipline via a counter incremented in `resident()`. **The
counter measures CONTEXT-MANAGER NESTING, not memory.** `stream()` yields inside the manager and the
consumer is free to retain every yielded group, so measured overshoot equals the block count exactly:
peak resident state is all blocks while the counter reads 1 throughout.

This is the arc's own apparatus failing in the way this project keeps finding — a counter that
reports the quantity it was incremented for rather than the quantity it is named for. The residency
claim is withdrawn; what the counter honestly asserts is that `resident()` is never nested.

### 3.3 The blocking is VACUOUS at its only wire, and §2.3 pre-registered what to do

With `delta_block_size = 1` and exactly one post neuron per deposit, `group()` returns **exactly one
block, every time** — measured: `blocks per record = 1.00` over 400 deposits. The partition is the
identity, and the residency discipline is never exercised by the only caller.

§2.3 pre-registered: *"If block-aligned grouping cannot be shown to change anything a flat record
does not already do, the blocks are DELETED."* That disposition is **not executed here**, and the
reason is stated rather than assumed: the blocking is untested at this wire, not measured useless at
the wire it was built for (a grown substrate, which does not exist yet — Phase 2 stops at 2 units).
It is carried as **OPEN DEBT with a deletion condition**: if the first real consumer of grouped
records does not use the block structure, the blocks go.

### 3.4 What the arc DID measure, and it is the useful part

The append rate a production sink would produce, driven through the real conversational path
(400 deposits, scratch home):

| quarter of the record stream | records | bytes |
|---|---|---|
| q1 | 192 | 15,984 |
| q2 | 192 | 16,032 |
| q3 | 192 | 15,168 |
| q4 | 195 | 15,967 |

**771 records from 400 deposits (192.8 %)** — `interact` deposits twice per turn (reinforce the
correct neuron, depress the wrongly-chosen one) — at **~158 B per deposit, and the rate is FLAT**.
Compare-drop does not quiet it, because the learner keeps moving weights past `q_step = lr` on every
deposit.

**That is a flood shape, and it is the reason the production wire is NOT made in this arc.** L4b's
incident was a tag appended at a rate nobody had measured; `RETENTION[0x09]` says the budget stays
unbaselined "until a live writer exists to measure". The measurement now exists; the wire owes a
baselined budget first.

### 3.5 Corrected disposition

**T064 is HALF-LANDED.** What stands: `UnitBlocks` (partition, POST-keyed grouping, a non-nesting
guard), the delta-emitting method with a derived `q_step`, their tests, and the append-rate
measurement above. What does NOT stand: the live-writer claim, the peak-residency claim, and any
suggestion that `0x09` is charged.

**What the completion owes:** a production sink that writes a genuinely TAGGED `0x09` lake record; a
baselined append budget for that rate; a residency proof that measures memory rather than nesting;
and a first real consumer of grouped records, or the blocks are deleted per §2.3.
