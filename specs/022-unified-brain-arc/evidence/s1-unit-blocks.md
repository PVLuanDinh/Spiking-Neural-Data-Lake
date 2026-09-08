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
