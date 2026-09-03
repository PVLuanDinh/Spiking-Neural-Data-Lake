# Rank 2b: the GrowingLearner lake record (2026-09-02)

Audit §8 rank 2, commit 2 of 3: *"persist the experience-driven grower under its own tag with the
cross-process restart red (`tests/test_brain_growth_restart.py` shape)."* Owner directive: "Do rank 2."
The audit's §3 finding: *"the experience-driven grower that does see real vision and audio
(`GrowingLearner`, 24 units) has no persistence path anywhere in the tree (CONFIRMED BLOCKER) — it
re-grows from zero at every restart, and the daemon restarts on every commit."*

## Why it was expected to be load-bearing (a claim the Result below WITHDRAWS)

Since rank 2a (`2867ed64`) the neurogenesis grower's categories ARE the arm loop's `GrowingLearner`
prototype ids, and the NG-SNN readout persisted under `_T_GROWTH_DELTA` is keyed to them. The
expectation written before the measurement: a learner that re-grows from zero on every daemon relaunch
re-keys the vocabulary the substrate was fit on, so persisting it would change the first refit after a
restart. The measurement (Result) could not show that, for a reason of its own protocol; the sentence
stands here as the pre-registered expectation, not as a finding.

## What was observed on the live daemon before this commit (the relaunch onto `2867ed64`)

Clean handoff; restored 31 units at accuracy 0.6333 with floors `None`; then **881 wake percepts and 900
sleep ticks with NO refit** (re-read by the review panel at 3,539 wake ticks: still no refit, floors still
`None`). `NeurogenesisSelfImprove._refit` refuses a buffer it cannot tell two categories apart in
(`n_cat < 2`). Read off `/api/state` at the same time: `arm_stopped: True` (the durable E-STOP latch from
2026-08-31; `estop.latch` mtime 07:40 that day), `perceiving_real`/`perceiving_audio` True,
`real_perception_units` 24, `neurogenesis.n_modes` 2. **Inference, labelled as one (no control run):** a
stopped arm has constant proprioception, and the attention-pooled real channels reach the arm's context
learner as near-constant scalars, which at vigilance 0.85 recruits very few prototypes — CONSISTENT with
a buffer holding one category, which `/api/state` does not expose directly (`n_modes` counts every
category ever shown, including the first percept, which has no predecessor and never enters the buffer:
`n_modes` 2 and "one category in the buffer" coexist, reproduced in-process by the panel). The learner
had re-grown from zero at the relaunch (no record yet). The pre-registered expectation from 2a ("the
restored substrate prunes at the first real-stream refit") could not be tested: its precondition never
held. What would settle the causal claim is a latch-cleared brain on the same stream — an owner action
(audit §8 rank 8); until then this is the E-STOP's *likely* P0 cost, not its measured one.

## Mechanism (P2: reuse the ONE named-state record — no new tag)

- `hippocampus/growing_learner.GrowingLearner.state_dict()` / `load_state_dict(d)`: a JSON-able dict
  (`fmt` = `growing_learner.v1`, vigilance, lr, dim, prototypes); every field is ADOPTED on restore (a
  field the restore does not read is a claim the lake keeps forever). `load_state_dict` REFUSES rather
  than half-restores: a wrong format tag, a prototype of the wrong width, a non-finite value, a
  vigilance or lr outside [0, 1], or a live learner that has already recruited (ids in use are never
  overwritten). The record's dim wins for an empty learner; whether it matches the live producer is the
  consumer's question (next bullet).
- `Brain._flush_prototypes()`: ONE mechanism, TWO callers (the night's persist block beside
  `_flush_growth`, and `close()`), writing `save_state("growing_learners.json", {"arm": …, "vis": …,
  "aud": …})` — the existing `_T_STATE` named-state record (`substrate/home.save_state`), which drops
  superseded history on consolidation and returns False when unchanged. A brain that has recruited
  nothing writes nothing. Measured by the panel: ~9 ms per night's flush at 26 prototypes/learner, the
  record ~135 KB there and ~520 KB at 100 prototypes; a refinement on any familiar hit changes it, so a
  night is rarely "unchanged" — once a night that is negligible.
- Restore in `Brain.__init__`, gated on `persist` (the same gate as growth), after the arm exists: the
  arm's context learner from `"arm"` — REFUSED and named if the record's width differs from the arm's
  producer width (the panel measured a 32-wide record restored into the 64-wide arm raising out of every
  wake tick); the vision/audio learners are CONSTRUCTED from their records up front (they are otherwise
  created lazily at the first real percept), so `real_perception_units` at tick 0 is what the lake
  holds, and at the first real percept `_consume_perception` GROWS a narrower learner (`grow_dim`) or
  DROPS a wider one, counted on `state()["prototype_faults"]` and named on `state()["prototypes_restore"]`
  (top-level, so a host with no torch tier sees it too; a `PROVENANCE` row says what it covers). The old
  bare `except: pass` on that path is gone: every swallow is counted.
- **What this record covers, exactly: the three PERCEPT learners** — the arm's context learner and the
  vision and audio prototype learners. **Not persisted, said plainly (open P4b debt):** the arm's Q-table
  over these ids (`arm.q`, re-created lazily, so the ids are stable and the policy rows over them are
  not); the affect integrator's SOM (`AffectIntegrator._som`, `n_emotions`) and the lifelong
  consolidator's learner (`LifelongConsolidator`, `lifelong_units`) — two more live `GrowingLearner`
  instances the panel found re-growing from zero across the same process boundary (both are constructed
  after the restore block, so persisting them is a code move, not this commit). The audit's §3 row
  ("used at eight live sites … no persistence anywhere") is therefore closed for three of the live
  instances, not all.

## Protocol — PRE-REGISTERED before the first run (written first; results follow)

**Red 1 (P4b: "it persists" is a claim about a NEW process).** `tests/test_growing_learner_restart.py`:
a parent `make_brain(seed=7, persist=True)` under an isolated `NORD_HOME` ticks until the arm's learner
holds ≥ 2 prototypes, records `crc32` of each learner's prototypes and their counts, closes, exits; a
SEPARATE interpreter over the same home constructs a brain and reports the same counts and crcs at tick
0, and the lake holds ≥ 1 `_T_STATE` record named `growing_learners.json` (counted by `BinLog` replay,
not by the writer's own state). CONTROL: a child over an EMPTY home comes back with 0 prototypes.
REFUSAL: a record with a wrong width leaves the learner empty (no half-restore).

**Measurement (P3: does the restored vocabulary change an outcome?).** 6 seeds. Parent: `persist=True`
for 400 ticks (grows prototypes AND a substrate). Two children over the same home, 200 ticks each:
LEARNER+BLOB (this commit) vs BLOB-ONLY (the learner record deleted from the home before the child
boots — 2a's behaviour). Statistic: the child's FIRST refit's held-out accuracy and the number of
distinct categories its buffer holds; the persisted NG-SNN's `m_out` vs the child's `n_modes`.
Pre-registered reading: LEARNER+BLOB's first-refit accuracy ≥ BLOB-ONLY's on a majority of seeds is a
WIN; the reverse or a tie is a NULL and is recorded — the restart red above is the commit's claim, this
measurement is its attribution, and a NULL here would mean the readout re-keys itself harmlessly at the
first refit (the panel's OPEN-gate observation), which leaves the P4b claim intact and the "load-bearing"
sentence above withdrawn.

## Result

**Red 1 (the restart) — GREEN, across a real process boundary.** `tests/test_growing_learner_restart.py`:
the parent recruits ≥ 2 arm prototypes AND ≥ 1 vision + 1 audio prototype (asserted, so a byte-equal
comparison is never vacuous on a `None`), the lake holds the `growing_learners.json` `_T_STATE` record
(BinLog replay through the lake's own record parser), the child comes back with the same counts and
byte-equal crc32s on all three learners at tick 0, `real_perception_units` at tick 0 equals what the
lake holds, the empty-home child holds 0 prototypes; a parent KILLED with `os._exit` after a night's
flush (no `close()`) comes back with exactly what that night flushed — the sleep-tick caller's own red,
since the panel measured the clean parent reaching two prototypes at tick 2 and only `close()` ever
flushing; a width-mismatched arm record is refused and named and the brain still ticks, a wider vision
record is dropped and counted; and the wrong-width / wrong-format / non-finite / out-of-range records
are refused whole. The panel's mutants (flush a no-op; restore skipping `arm`; `load_state_dict`
returning True without setting; a different record name) each go RED. The module self-check adds the
JSON round trip (vigilance and lr adopted from the record) and five refusals.

**Measurement (attribution), 6 seeds, warm 400 / continue 200, each role its own interpreter:**

| seed | parent arm units / grown / readout classes | LEARNER+BLOB: start arm units → first-refit acc (cats in buffer, units) | BLOB-ONLY: start arm units → first-refit acc (cats, units) |
|---|---|---|---|
| 0 | 5 / 1 / 5 | 5 → 0.000 (2, 1) | 0 → 0.000 (2, 1) |
| 1 | 3 / 2 / 3 | 3 → 0.143 (3, 1) | 0 → 0.000 (2, 1) |
| 2 | 4 / 2 / 4 | 4 → 0.250 (2, 3) | 0 → 0.000 (2, 2) |
| 3 | 4 / 1 / 4 | 4 → 0.167 (2, 1) | 0 → 0.167 (2, 1) |
| 4 | 4 / 5 / 4 | 4 → 0.333 (3, 5) | 0 → 0.333 (3, 5) |
| 5 | 4 / 2 / 3 | 4 → 0.750 (2, 2) | 0 → 0.750 (2, 2) |

**Verdict: NULL on the attribution — and the protocol, not the mechanism, is why.** Two disclosures
first. (i) The pre-registration's tie wording was a criterion with no reachable red: read literally,
"≥ on a majority of seeds is a WIN" scores two IDENTICAL arms as a win. The harness's rule (wins > losses
AND wins ≥ half the seeds, in the script before the run) is P8 clause 1 applied — the reading with a red
is the one adopted — and it scores 2–0–4 as NULL. (ii) The first-refit "accuracy" is a 3–7-row held-out
slice (`holdout = min(60, n // 3)` on an 11–16-row child buffer): each "gain" above is one row (1/7,
1/4).

What the panel then showed by instrumenting both children (per-tick learner ids, the buffer at the first
refit, seeds 0/6/7): **the four ties are vacuous comparisons.** Both children replay the parent's stream
from t = 0 — the perception pointers (`_real_ptr`, `_audio_ptr`) and the arm's RNG are not in the lake
and reset to zero — so the blob-only child re-recruited the SAME prototype ids at the parent's OWN recruit
ticks (seed 0: [0, 1, 3, 39, 86] both), and the two children's category buffers were IDENTICAL when the
first refit was scored (observe #12–17), while their id streams first diverge at observe #39 / #57 /
never within 100. Nothing was re-keyed on the ties, so "the readout re-keys itself harmlessly" is not what
was measured and is withdrawn. The two "gains" are one-row differences on a 4-row slice and are not
evidence either (the earlier "met a THIRD category" gloss was wrong for seed 2, where both buffers held
2 categories). The record's value, if any, lies in the LATE prototypes (recruited at ticks 39/86/115,
refined over 400 ticks) that a first-refit statistic at tick 26–44 cannot see by construction — and the
daemon, `make_brain(seed=0)` with the same pointer resets, replays its early stream at every relaunch
too. A protocol that could attribute would score AFTER the measured divergence or de-correlate the
child's stream; it is not run here. Also disclosed: the blob-only home was the parent's `brain_lake.bin`
copied alone (the parent's `skills/`, `skills.spkl` sidecars were not), which did not bite because the
buffers were identical.

So: the "load-bearing for this wire" sentence in 2a's evidence is WITHDRAWN as a claim about accuracy —
not because re-keying was measured harmless, but because the measurement could not reach the question.
2b's claim is P4b's (a restart is not amnesia for what the brain perceived — its prototype ids and
vectors survive a new process), and the readout being keyed to stable ids is a consistency property.

Also visible in the table: every child, restored or not, sits near the floor at its first refit
(0.00–0.33 on five seeds) with 2–3 categories in a fresh buffer — the engine's `_buf_x/_buf_y` are not
in the lake, so a restarted grower starts its buffer from zero whatever else it restores; that is 2a's
regime-shift shape restated, not a new finding.

Reproduce: `rank2b_charge.py` and `rank2b_charge.json` are archived with the session's audit record
beside the memory store (`archive/audit-2026-09-01/rank2b/`).
