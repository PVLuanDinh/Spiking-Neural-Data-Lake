# Stage 05: diagnostic protocol, before task runs

2026-11-01, America/Chicago. Artur authorized stage 05 after stage 04. Source
`c4c10b285f1741aa798d5ee9c5296ae371fabecf`, clean detached checkout. Only external
planning artifacts are edited; no upstream implementation, frozen campaign or seed
registration changes. This is a local exploratory component experiment.

## Question and incumbents

Can a fixed 256-neuron reservoir and stage-04 DeltaAdaptor retain two compatible
tasks, improve after offline consolidation, and restore the exact learned state
from the one isolated brain_lake.bin in a new process?

A = existing i.S0C0; B = existing ac1d. Their vocabularies/episodes differ and the
same label vocabulary is used. Joint solvability must be MEASURED by an interleaved
positive using the same reservoir/features and delta rule; individual success is
insufficient. No task identifier, new encoder or generator is added. S0C0 evaluation
repeats known episodes; ac1d uses existing disjoint train/validation/future texts.
All task seeds 8000-8003 are exposed development seeds from stages 02-04; scratch
8900 is already exposed. No historical confirmatory/reserve cohort is used.

Adopt `_Drive.walk`, stage-04 state/parameter readers, DeltaAdaptor/OnlineReadout.learn,
and NGSNN._solve_beta (current activity-regularized batch readout solve, defaults
c1=1, k=.1, c0=.0001). The batch solve consumes already frozen reservoir features;
NGSNN.fit, rates, growth and pruning are never called. This is DIRECT_SOLVE, with
no propagated gradient or weight transport into a learning update. It is a different
schedule/rule after delta wake training; a gain cannot be attributed to biological
sleep. The retired sliding-window online ridge is not restored.

R2/R3/G6 history: automatic interaction replay/deposit, Benna-Fusi rest/rebinding
and AttractorNet.sleep learning were retired by owner1039; do not restore them.
Retained structural rewiring would change topology and is outside this stage.
R3 F5 recorded no stage advantage over flat, with some zero-headroom seeds. We
therefore measure each seed's headroom and do not invent sleep stages. G6's full
eligible-brain, mask, pressure and durable-consumption gaps are not solved here.

## Frozen design and budget

One feature preparation per seed using the unchanged stage-04 carry reservoir,
same parameters/host recipe. Generate the existing 30 teaching epochs (A:240,
B:960 presentations). Read validation before, after A, after B, after offline phase,
then read future once after freeze. No checkpoint selection or early stopping.
Joint positive uses the same 30 epochs interleaved A then B in each epoch.

All arms start with identical zero beta, receive identical A then B training and
exactly the same stored training rows. Offline arms:

1. none: no update calls at all;
2. replay: 10 passes over the first teaching epoch's 8 A + 32 B rows, using the
   existing delta rule (400 presentations; no additional real examples);
3. batch: NGSNN._solve_beta once over that same 400-row matrix, assign beta;
4. batch_shuffle: same batch solve and rows, marginal-preserving permutation of
   the labels (existing taught_campaign._shuffled_labels), as a signal control.

Repeated rows weight batch regularization; the solve is not cost-matched to 400
delta updates. Report wall/CPU/RSS and presentations/solves separately. The replay
mix is uniform over examples (A:B = 1:4), not task-balanced. No tuning after data.

Record all answers, feature rows/texts/targets, state hashes and full beta/counters,
replay membership, commands, runtime, source/protocol hashes and errors. Before/after
every evaluation, hash full adaptor state AND replay and lake bytes; patch learning
and batch solve to refuse. No evaluation examples/labels enter replay. Replay comes
from train only; ac1d text intersections must be empty. S0C0 overlap is disclosed.

## Persistence and reachable failures

Use existing home.save_named_state/load_named_state on explicit isolated lake paths,
tag 0x07 (existing COMPACTABLE named-state tenant). One versioned envelope contains
beta, rule counters, labels, readiness/buffer, replay rows and fixed-pool seed/recipe/
parameter hashes. SHA256 envelope and expected latest checkpoint detect corruption
and a missing final write. Save after A, B and offline. Keep a foreign named record
and prove it survives. An independent process loads only the lake state, rebuilds
the fixed pool from its persisted seed and source recipe, recomputes evaluation
features from stimulus texts and evaluates without training. Verify exact complete
state, features and answers, not accuracy alone. These are explicit harness saves,
NOT proof of the production Brain flush or R2 transactional crash recovery.

Before task runs, constructed controls must detect: learning during read, direct
beta mutation, replay mutation, wrong score, missing checkpoint, corrupt envelope,
missing last write/stale state and changed pool recipe. A planted separable matrix
must be solved by the adopted batch code and its sign-reversed beta must fail.
Missing/invalid arms or broken controls HALT, never a scientific negative result.

## Sample size (P1), outcomes and stopping

N=4 is the finite already exposed diagnostic set, no power or population claim.
First seed 8000 checks cost/technical controls, then 8001-8003 only if valid. Each
worker <=120 s, whole block <=20 min. Repeat 8000 in new processes once for exact
determinism, without increasing N. No additional seed or larger budget is selected
after seeing outcomes. Dependencies/a new runtime error may be repaired with a
dated disclosure and original failed records preserved.

Descriptive endpoints: A forgetting after B, A headroom, offline minus none for A/B
accuracy and macro accuracy, batch minus replay, B loss, train-only loss diagnostic,
exact reload. Conditional practical effect for future planning: +.10 A accuracy
with <=.05 B loss. Report paired differences/sample SD only. P1 confirmatory sample
size, uncertainty in SD, multiplicity and sequential boundaries remain unfilled;
stage-03 power is not transferable. Low N or ceiling gives INCONCLUSIVE for broad
benefit, not scientific NULL, no deletion or P8 scale retest is licensed. A scoped
positive is credited only on this finite exposed set and fixed budget.

Completion: joint positive is valid; all three primary modes measured; leakage and
persistence controls fire; exact new-process restoration; honest effect/limitations
and reusable records for stage 06. Failure of joint positive stops the comparison
and records a bounded dependency diagnosis. Full Brain/G6/G9 success is not claimed.
