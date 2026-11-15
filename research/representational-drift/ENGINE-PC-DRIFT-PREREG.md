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
`src/snn_data_lake/instruments/allen_drift.py` (the frozen code kept byte-exact as sha256-checked records in
`tools/d0_allen/frozen-code/*.py.txt`; the module self-check proves its copies AST-identical to them), and the
agreement-rule simulation and the E-PC engine study into `src/snn_data_lake/instruments/drift_studies.py`
(`rule-sim`, `engine-pc`). No result was re-read or re-run for the move.
