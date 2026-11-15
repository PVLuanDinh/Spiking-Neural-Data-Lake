# Engine drift study E-PC (MAP D3, source S4): pre-registration, written before any run

Owner 2026-11-15: "Start it, and include it in the final project report" (the final project's stretch goal,
unlocked because RQ1 H1 and RQ4 E1 passed on held-out mice).

## What is measured

The SNDL engine's live predictive-coding layer (`readout/predictive_coding.PredictiveCodingNetwork`) in the exact
configuration `engine/live_brain.LiveBrain.tick` uses: input = `encode_multimodal(visual, temperature, audio,
distance)` (64-d), dims (64, 16, 8), settle `infer(x, steps=60, lr=0.05)`, one `learn_step()` (lr_w 0.02) per wake
tick. **Component-level**: the network is driven directly, not through a constructed Brain (disclosed limit).

- **Population:** the 24 cause neurons (r1 16 + r2 8), read after a settle.
- **Stimuli:** 30 fixed probe contexts (random raw values per modality, seed-fixed). A presentation adds Gaussian
  sensory noise (sd 0.05) to the raw values before encoding: the "trials".
- **Imaging session:** each probe presented 10 times, measurement only (the settle runs, no learning step), giving
  (24 neurons, 10 repeats, 30 probes): the same shape as the Allen grid.
- **Between sessions:** 300 wake ticks on a background stream of fresh random contexts. Sessions 0..3, so 0, 300,
  600 and 900 ticks.
- **Arms:** LEARN (learn_step every wake tick, as live) and FROZEN (the twin forked from the same snapshot: identical
  stream, no learn_step). The frozen twin is the control that is not the method.
- **Units:** 10 seeds (network seed = probe seed = stream seed).

## Statistics (the Allen pipeline, unchanged functions)

Per seed and arm, over the 6 session pairs: d1 = within - mean between (PV correlation on odd/even repeat halves),
d2 = mean [RSA geometry r - median tuning r], d3 = slope of between-session r on ticks apart. Plus, from
`representational_drift.read_drift` on session 0 vs session 3 (probe identity, k = 30): frozen-readout accuracy on
day 0 and day 3, the realigned accuracy, and the readout null fraction beside its matched-rank random control.

## Hypotheses and decisions (exact one-sided sign test over 10 seeds, MET at >= 9 of 10, p = 0.0107)

| id | prediction | why |
|---|---|---|
| E-H1 | d1(LEARN) - d1(FROZEN) > 0 | local learning makes the code drift (the frozen twin removes noise-only drift) |
| E-H2 | d2(LEARN) > 0 | Qin et al. (2023): Hebbian drift leaves geometry more stable than single units |
| E-H3 | d3(LEARN) < 0 | similarity falls with time |

Reported, not decided: all arms' d1-d3, the readout numbers. Reachable reds: FROZEN d1 near 0 (sign test on
d1(FROZEN) > 0 must NOT be MET); the instrument's self-tests (self-paired session = 0) already pass.

Not claimed: anything about the whole brain, sleep, or other drift sources (S1-S3, S5-S7); the 10-seed n is a
fixed one-shot count chosen for the deadline, not a power-derived P1 design (disclosed).

## Disclosure before the run (smoke test on unregistered seed 99, 50 ticks)

- The readout null-space split is VACUOUS here: a 30-class readout on 24 neurons spans all 24 dimensions, so the
  null fraction is 0 for the readout and its random control alike. It is reported as vacuous, never read.
- The smoke test found and fixed one ordering bug in the script (trials must be probe-major so `read_drift`'s
  even/odd split falls on repeats). No registered seed was run before this commit.

## Results, 2026-11-15 (10 registered seeds, run once at b878ee89)

| test | k / n | exact p | verdict |
|---|---|---|---|
| E-H1 d1(LEARN) - d1(FROZEN) > 0 | 10/10 | 0.00098 | MET |
| E-H2 d2(LEARN) > 0 | 0/10 | 1.0 | NOT MET (all 10 negative) |
| E-H3 d3(LEARN) < 0 | 10/10 | 0.00098 | MET |
| RED d1(FROZEN) > 0 (must not be met) | 5/10 | 0.62 | not met, as required |

- d1: LEARN mean +0.0627 (sd 0.0154); FROZEN +0.00005 (sd 0.0017). d2: LEARN -0.0645 (sd 0.023); FROZEN +0.0005.
  d3: LEARN -1.53e-4 per tick; FROZEN -1.1e-6.
- Frozen readout (probe identity, 30 classes, chance 0.033): session 0 accuracy 0.647 in both arms; at session 3
  LEARN 0.419 vs FROZEN 0.642; after Procrustes realignment LEARN 0.545, FROZEN 0.582.
- Null-space split: vacuous (0 in both arms and in the random control), as disclosed before the run.
- Full per-seed numbers: research/representational-drift/engine_pc_drift_result.json.


## Move note, 2026-11-15 (P4c; history above is unchanged)

The scripts named above moved into the governed package: the Allen pipeline into
`src/snn_data_lake/sensorimotor/allen_drift.py` (the frozen code kept byte-exact as sha256-checked records in
`tools/d0_allen/frozen-code/*.py.txt`; the module self-check proves its copies AST-identical to them), and the
agreement-rule simulation and the E-PC engine study into `src/snn_data_lake/instruments/drift_studies.py`
(`rule-sim`, `engine-pc`). No result was re-read or re-run for the move.

- Correction, same day: `drift_studies.py` was folded away before landing (native_ratchet: one fewer numpy
  importer). The agreement-rule simulation now lives in `instruments/d_r0_part_a.py` (`rule-sim`), the E-PC
  engine study in `instruments/representational_drift.py` (`engine-pc`).

## Sample size (P1)

One-shot: a fixed count of 10 seeds, because the owner's course deadline left no time for a sequential campaign.
Written 2026-11-15 AFTER the registered run (disclosed): the section was missing at registration and the sample-size
gate flagged it. No number below was used to choose n.

- Measured SD (from the registered run itself, not a pilot): d1(LEARN) - d1(FROZEN) SD 0.0154; d2(LEARN) SD 0.023.
- Effect claimed: the sign test's p+ >= 0.9 (9 of 10 seeds in the predicted direction); the observed effect was
  d1 difference mean +0.063 (about 4 SD) and d2 mean -0.064.
- Target: power 0.74 at p+ = 0.9 for MET at >= 9 of 10 (exact binomial; 0.996 at p+ = 0.99); below that the reading is
  INCONCLUSIVE, never a null. A proper P1 design for a follow-up takes these SDs as its pilot.

## Amendment S10, 2026-11-15 (owner "Start next steps"): the 10x re-test on units, written before any scaled run

P8 amendment item 2: E-H2 read NOT MET (0/10) at 24 cause neurons; geometry may stabilise only in larger populations,
so the one 10x re-test runs on the **units** axis.

- **Change:** dims (64, 160, 80), so 240 cause neurons. Everything else as registered (inputs, settle, learning rule,
  sessions, ticks, noise, 10 seeds, frozen twin), except REPEATS = 20 per probe per session: the decoder needs more
  training trials than its 241 parameters (capacity guard), so 300 even-trial samples train it.
- **Same hypotheses and rule:** E-H1, E-H2, E-H3, exact one-sided sign test, MET at >= 9 of 10; the frozen-twin red.
- **Now readable:** the readout null-space split (30 classes in 240 dimensions leaves a 210-dimensional null space),
  reported beside its matched-rank random control, Rule et al. (2020). Reported, not decided.
- **Disposition:** E-H2 still NOT MET at 240 neurons -> a scale-tested null for this layer (the network's geometry is
  not preserved at 24 or 240 units). E-H2 MET at 240 -> geometry preservation is scale-dependent, credited at 240 only.
- **Sample size (P1):** one-shot 10 seeds, the same fixed count as the registered run so the two scales compare
  directly; measured SD from the 24-unit run (d2 SD 0.023); effect p+ >= 0.9; target power 0.74 at p+ = 0.9.
- Output: research/representational-drift/engine_pc_drift_scale10_result.json.
- Disclosure before the scaled run (smoke test, unregistered seed 99, 30 ticks): the Procrustes re-alignment is
  ILL-POSED at this scale -- it is fitted on 30 class means in 240 dimensions, so the rotation is underdetermined
  outside their span (smoke: realigned accuracy 0.097). The realigned accuracy is reported as uninterpretable at
  scale 10, never read. The registered scale-1 code path is unchanged (R1, R2, REPEATS default to 16, 8, 10).

## Results, amendment S10 (10x: 240 cause neurons, 10 registered seeds, run once at f6bbe750)

| test | k / n | exact p | verdict |
|---|---|---|---|
| E-H1 d1(LEARN) - d1(FROZEN) > 0 | 9/10 | 0.011 | MET |
| E-H2 d2(LEARN) > 0 | 10/10 | 0.00098 | MET by the registered rule -- see the disclosure below |
| E-H3 d3(LEARN) < 0 | 10/10 | 0.00098 | MET |
| RED d1(FROZEN) > 0 (must not be met) | 6/10 | 0.38 | not met, as required |

- d1: LEARN +0.0067 (sd 0.0030), FROZEN +0.0001: drift is ~10x smaller than at 24 units (the same lr_w spread over
  10x more units). d3: LEARN -1.7e-5 per tick, FROZEN -3e-7.
- **Disclosure on E-H2 (unprompted, P8 item 4):** the FROZEN twin reads the same d2 (+0.1166 vs LEARN +0.1173). At 240
  units the population geometry is simply measured more reliably than noisy single-unit tuning, learning or not. The
  registered E-H2 asks only d2(LEARN) > 0, so it is MET, but the reading is NOT evidence that learning preserves
  geometry: d2(LEARN) - d2(FROZEN) is +0.0007 here and -0.065 at 24 units. The rule should have been the contrast
  against the frozen twin, as E-H1 is; recorded as a design lesson, the verdict is not re-scored.
- Frozen readout: 0.531 -> 0.429 under learning (FROZEN 0.531 -> 0.543). Re-alignment ill-posed at this scale (0.08,
  disclosed before the run), not read.
- Null-space split (now readable, reported only): learning-driven drift lies 94.8% in the readout's null space vs
  87.9% for matched-rank random readouts (FROZEN arm 98.2%, noise only).
- Per-seed numbers: research/representational-drift/engine_pc_drift_scale10_result.json.
