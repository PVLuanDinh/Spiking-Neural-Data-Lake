# Stage 05 continuation: class-block interference pilot

2026-11-02, America/Chicago. Artur authorized this continuation after the proposal
to split A_c1d by its existing labels. Source remains clean detached
c4c10b285f1741aa798d5ee9c5296ae371fabecf. Original pilot-1 and its protocol remain
unchanged. Only external planning artifacts are edited.

## Question and fixed substrate

Does sequential single-class training create measurable interference in the current
fixed-pool delta baseline, and can the already specified offline procedures recover
the first class without losing the second?

Reuse A_c1d, its existing generators/splits and carry feature path. A = examples
labelled GREET, B = examples labelled GOAL; labels themselves never change. Each
class has 16 unique train, 4 dev and 6 future texts; evaluation repeats each four
times, so n=16/24 per class does not mean 16/24 independent examples. Joint train
contains the original 32 texts, with no task-id input. Single-class A/B accuracy
alone can be achieved by a constant answer; the combined balanced read and joint
positive are therefore essential. This is a class-order diagnostic, not two
independently learned general skills or evidence about biological sleep.

Existing fixed 256-neuron pool and DeltaAdaptor/OnlineReadout, CPU mu=1, labels
[GOAL,GREET], zero initial beta, same host/parameters as stage 04/05. No growth,
pruning, native build, production Brain, restored retired sleep capability or
shared-source change. NGSNN._solve_beta is the same existing batch mechanism.

## Bounded development discovery, before any offline comparison

Use all four already exposed development seeds 8000-8003, never select a subset.
Generate feature rows once per seed, compare them to preserved stage-05 A_c1d
records, assert fixed parameter hashes and train/dev/future text disjointness.
Use the existing original shuffled epoch order, filtered to each class; no new
generator. Max teaching-order generation is 300 epochs. All 32 train texts remain
represented. Future_d and historical scientific cohorts remain unopened.

1. Joint positive: original mixed A_c1d epochs, exactly 30, no early stopping;
   both classes must be fully correct at every one of the last 10 dev reads.
2. Sequential discovery: train A for 30 epochs (480 presentations), save/read A/B.
   Then train B for up to 300 epochs; save/read after exactly 30, 100, 300 B epochs
   (480, 1600, 4800 B presentations). These are three declared development looks.
3. Choose the FIRST budget in (30,100,300) for which EVERY seed has A-before-B>=.90,
   B-after-B>=.90 and A accuracy loss>=.10 on dev. Selection uses dev only, and
   is written to selection.json before offline workers start. Do not inspect
   future outcomes or enlarge the ladder to choose a result. If no budget qualifies,
   stop with NO_SENSITIVE_REGIME and report all looks; do not run sleep comparisons.

The three budgets are a finite engineering ladder, not a statistically powered
sequential design or three scientific hypothesis tests. All discovery results,
including unused budgets/seeds, remain reported. Continuing from B30 to B100/300
must equal the declared single learning trajectory; no resets between looks.

## Frozen offline comparison after discovery

For each seed, make fresh copies of its selected persisted after-B lake. Every arm
loads exactly that same full beta/counters/replay; no retraining or reinitialization.
Memory contains only all 32 train examples, not dev/future. Offline procedures:

- none: skip updates entirely;
- replay: 10 original mixed training epochs, 320 delta updates, all known rows;
- batch: the same 320 rows and true labels, one existing regularized solve;
- batch_shuffle: identical rows/solve, existing marginal-preserving label shuffle.

No parameters, epoch count or mode change after observing these comparisons. Read
dev and future once after freeze (then repeat only for the fixed restart proof).
Report A recovery, B loss and balanced mean accuracy relative to none; also batch
minus replay and true-label minus shuffle. Conditional useful effect for this
diagnostic is >=.10 A recovery with <=.05 B loss. This is descriptive at N=4.
No p-value, equivalence, general scientific NULL, deletion or P8 retest is licensed.

## Lake, sensitivity, reproducibility and stop conditions

Reuse guarded pure reads and full state snapshots from the original stage-05
harness. Learn and batch solve refuse during evaluation; learner/replay hashes and
lake bytes must not change. Targets remain with scorer; restore receives only
texts/roles/feature receipts. Checkpoint at episode boundaries, with full readout,
training memory, seed, pool recipe, phase and the experiment profile in the existing
named-state tenant of brain_lake.bin. A versioned envelope/receipt detects missing,
stale and corrupt state. A foreign named record must survive. No sidecar state.

Before task discovery run the prior constructed guard positives/reds, plus new
checkpoint schema/receipt controls and a changed real-pool-recipe refusal on scratch
8900. During final comparison, each finished arm exits; a new process must load its
lake, rebuild the fixed pool, recompute both classes' features and reproduce full
readout/replay and all answers exactly. Repeat seed-8000 discovery and all selected
offline arms once in fresh processes; repeats do not increase N. Any failed positive,
hash, budget, schema, transport or restore check HALTs; missing/failed arms are not
negative scientific results. Preserve failures and disclose repairs separately.

One worker/seed/arm per subprocess, isolated guarded NORD_HOME, CPython 3.14t GIL-off,
torch CPU threads=1, bytecode off, <=120 seconds/worker and <=20 minutes/block.
No installs planned. Record commands, source/protocol/adapter hashes, all feature
rows/epoch orders/answers, states, lake bytes, wall/CPU/RSS and failure reasons.
Use a stdlib offline reader, constructed corruptions and two identical recalculations.

## Sample size (P1) and limits

N=4 is the full already exposed diagnostic set, not a derived confirmatory sample
size. Seeds and future outcomes were exposed previously under different training
orders. This work can establish an instrument-sensitive regime on these fixed
cases, not population generalization. Report per-seed headroom and paired SD;
neither a zero SD nor a successful finite selection licenses a powered claim.
A new confirmation needs separately justified practical effect, pilot variability,
MDE/power, multiplicity and simulated sequential boundaries with licensed untouched
seeds. The earlier stage-03 calculation is not transferable. The fixed engineering
ladder above is not that confirmation.

## Pre-run implementation clarification

The discovery lake retains four separately named checkpoints: after_a, b30,
b100 and b300, plus a foreign tenant. Each offline home receives a byte-identical
copy of this lake and explicitly loads ONLY the preselected bN name/receipt. Thus
unused later-budget training checkpoints remain preserved in the copy; they are
not loaded into the learner. The final result is a fifth experiment checkpoint
under its own name. The reader verifies the selected full state and the result
state separately. This is one lake with named snapshots per isolated trajectory,
not a separate learned-state sidecar. Restore stimuli are flat dev/final lists,
without target fields or class-group labels.

Pre-data preparation repair, 2026-11-02: class-pilot-1 was interrupted with code
130 while its source-receipt preparation was blocked in file reads, before any
worker/teaching. The cause of slow filesystem reads is not established. The retry
records the clean pinned source Git tree identity and SHA256 receipts/copies of
the seven external harness/protocol files, rather than reading all 400 unrelated
working source files. Source HEAD/clean checks still run before and after; no
task, budget, criterion, feature, learning rule or dataset changes. The failed
preparation is retained in class-pilot-1/preparation-error.json.

Second pre-data runtime repair, 2026-11-02: class-pilot-2 HALTed at 120 seconds in
the first controls worker, with no task teaching. A separate faulthandler probe
also timed out during Torch source imports. `ls -lO` showed compressed,dataless
files in the checkout, Torch and its cached archive. To avoid reading unavailable
placeholders, export exactly HEAD's src/tools from Git to /private/tmp and verify
every exported blob against git ls-tree (483 matched). Recover the exact twelve
versions of stage-04 runtime-requirements.txt as cp314t/arm64 or pure wheels into
a separate temporary site directory, checking wheel SHA256 against PyPI metadata.
The project venv is not modified. Workers assert actual SNDL/Torch/NumPy import
paths and GIL-off. Export verification and wheel receipts enter the manifest.
Only execution location changes; original source bytes, library versions, feature
checks, scientific conditions and 120-second worker cap stay fixed. The first two
failed preparations and import trace are retained. No outcome data existed when
these repairs were specified.
