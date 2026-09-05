# Phase 2's L1b — synapses learn online (2026-09-05)

The plan row (`PLAN-FINAL-2026-09-02.md` Phase 2, L1b; the red L-R2 in Part F), the owner's standing
authorization of 2026-09-05 ("Unblock everything, you are authorized to do everything to finish the plan":
every decision below is the session's call under it, disclosed where it departs from the row's letter), and the
lessons carried from the parallel-arms arc: a mechanism that redefines the measurement slice confounds its own
before/after (every arm here is scored on the SAME rows, verified by identical floors); a red both arms fail is
not a red (P8 clause 1); a gate's positive control runs BEFORE the cells; a bar set from the seeds it is scored
on is met by construction; the fold of a review is unreviewed code.

## 1. Territory (code at `1eb37df`, read 2026-09-05)

- **The grown units' input weights never move today.** `hippocampus/ng_snn.NGSNN` holds `W` (L, d_in = 64),
  `b` (L,) and the readout `beta` (L, m). `W`/`b` are drawn at random at admission (`fit`, the candidate blocks at
  scale Λ ∈ {0.1, 1, 5}) and the only writers in the tree are `fit`'s whole-assign, `prune_unearned`'s row
  select and `restore_growth_state`; only `beta` is ever re-solved. The LIF rate `s = spikes / T` (T = 15)
  is the unit's response to the constant current `x @ W.T + b`.
- **The wake hook.** `Brain._wake_tick` → `neuro.observe(percept, unit)` once per wake percept inside a
  try/except that counts refusals (`_neuro_feed_refusals`); `_tick_ensemble` steps ACh / LC / 5-HT BEFORE the
  arm tick, so `self._ach.plasticity_scale()` is fresh at the feed. `observe` today only buffers
  (percept_{t-1} → category_t). The batch solve (`NGSNN.fit` via `_refit`) runs ONLY on the sleep tick
  (`_sleep_tick` → `neuro.tick()`, every 20 sleep ticks); `fit(init_W=…)` keeps inherited units as the LEADING
  rows byte-identical, appends new ones, and `prune_unearned` selects rows — so a value written into the
  retained model's `W` between refits SURVIVES the next refit.
- **The third factor.** `neuromodulation/acetylcholine.Acetylcholine.plasticity_scale()` = `PLAST_MAX · level`,
  level a leaky EMA (τ 10) of the brain's novelty drive (1.0 on a prototype recruit, else clip(`_lp_mean`)),
  declared NOT_A_RULE "a scalar a rule may multiply by". LC's level was measured PINNED at 1.0 (its own
  docstring) and is excluded; the arm's RPE is a motor-coverage signal (a semantic mismatch for perceptual
  input weights) and is the disclosed non-choice. The ACh gate's live range is unmeasured on this stream —
  Stage 0 measures it first.
- **The floors and the accuracy.** `_refit` scores the held-out marginal-covering slice (`marginal_holdout`,
  L1a) with its majority and persistence floors; the L cells read the last-third mean of the refit rows.
  Under L1a + L3 LIVE sits BELOW its persistence floor on 8 of 8 seeds (0.31–0.95 vs 0.91–1.00,
  `l-parallel-arms.md` §3.3) and above its majority floor on 7 of 8 (mean +0.133, t 1.49, not at the bar).
- **Governance a new rule joins.** FR-018: a GRADIENT_CLAIM row, declared read/write sets (a RuleSpec over an
  AuthoredRule), per-rule neuter attribution; `tools/rule_census.json` is regenerated only through
  `rule_composer --regen-census` and a new LOCAL row needs a `_CENSUS_ADJUDICATED` entry or the census ratchet
  reds (UNDISPOSITIONED 57 > 56). A LOCAL rule owes no gradient cosine (POLICIES e-prop SCOPE);
  `cortex/eprop_kernel` stays the reference. `method_dormancy_gate` needs a live call site for a learn-verb
  method outside its own `__main__`.
- **The plan's control precedent, corrected.** The row cites "the gf.py::MLP.train precedent: its own
  GRADIENT_CLAIM row and a test that it is unreachable from the tick": the row exists; no such test does (the
  only reachability red keys on `.backward()` + optimizer step). The batch solve carries no `.backward()`, is
  ALREADY off the wake tick by construction, and is the GROWER (admission) as well as the readout solver — so
  "kept ONLY as a declared control off the live tick, else deleted" is scoped here to what it can mean: the
  incumbent "input rows frozen from admission" is the CONTROL ARM (the rule's neuter), not a code path to keep
  or delete; `NGSNN.fit` stays the admission mechanism (L2's) and the readout solver (until L1c), its
  DIRECT_SOLVE row unchanged. The session's call, disclosed.

## 2. Pre-registration (written before any cell runs; the build follows it)

### 2.1 The mechanism — one family, one factor swapped (P2)

`plasticity/grown_input_rule.GrownInputRule.learn(model, x, gate)`, called from `observe` after the transition
is buffered, on the RETAINED model's `W` in place. For unit i: `u_i = W_i / ‖W_i‖`, `xn = x / ‖x‖`, `s_i` the
unit's LIF rate on THIS percept (computed before the update), `g` the gate:
`u_i ← normalize(u_i + μ·g·post_i·xn)`, `W_i ← ‖W_i‖·u_i`. **Candidate OJA:** `post_i = s_i` (Hebb in
direction; the renormalization is Oja's decay to first order). **Candidate BCM:** `post_i = s_i(s_i − θ_i)`,
`θ_i ← θ_i + (s_i² − θ_i)/τ_θ`, θ_i initialized at the first observed s_i² (no move on first touch), τ_θ = 40.0
(`bcm_plasticity`'s default, cited). The row norm (the admission scale Λ_i) is invariant by construction; `b`
does not move (it sets the operating point; disclosed). Competitive-Hebbian is NOT a candidate: it collides
(P2) with the live ART recruit `GrowingLearner.observe` that already produces the grower's categories, and
`plasticity/competitive_learning` is exempt-dormant. **μ = 0.05 is CHOSEN** (the angular step per percept at
g = post = 1; the P1 debt this arc carries, named in the module; not tuned against any red). The gate is
`Acetylcholine.plasticity_scale()` passed by the brain at the feed; `None` = ungated (g = 1), disclosed.
The rule reads x, s_i, its own row, its own θ_i and g — never `beta`, never an error, never a transpose:
transport-free by construction, declared LOCAL.

State: OJA holds none beyond `W`; BCM's θ (L,) lives on the model, follows the inherited units across a
refit (leading rows) and a prune (`last_keep`) as placement does, and rides in the growth record as an
OPTIONAL npz member `bcm_theta` (no format bump). The record's `acc` keeps meaning "the last refit's held-out
accuracy of the weights AT THAT FIT" — with `W` moving between refits the weights beside it can differ; the
LIVE instrument below is the honest number for the moving weights (disclosed).

The LIVE instrument (reported, not the bar): in `observe`, BEFORE any learning on the tick, the retained
model predicts this category from the previous percept; `live_accuracy` and `live_persistence`
(= mean(cat_t == cat_{t-1})) accumulate on state() beside the rule's own charge (`moved`, `cum_abs_dw`,
`mean_gate`). A rule that raises inside `observe` reads as a feed refusal — every cell reports refusals.

### 2.2 Stage 0 — the gate's positive control (before any rule cell)

GATE0 × 3 seeds × 800 ticks: the ACh `plasticity_scale()` the brain passes on every fed tick. **Usable iff
on every seed: distinct values ≥ 20 AND the mean over the last 200 fed ticks ≥ 0.05** (a multiplicative
gate at ~0 makes the rule inert — a two-factor rule wearing a name). **If not usable:** the gate is
WITHDRAWN for this arc (g = 1, the three-factor claim dropped and disclosed; the ACh reading recorded as the
instrument's null), and the cells run ungated. No other fallback is pre-registered.

### 2.3 The cells — 8 seeds × 800 ticks, 4 at a time, one command, thread-pinned

Arms per seed (`l1b_probe.py`): **BATCH** (no rule: the incumbent = Stage A = the control), **BATCH_B** (the
A/A pair), **OJA**, **BCM**, **OJA_G1**, **BCM_G1** (gate pinned at 1.0: the gate's attribution), **TWIN_OJA**,
**TWIN_BCM** (the rule on, the readout UNTRAINED: `NGSNN.predict` scores through a seeded random beta of the
solved beta's Frobenius norm, one per model object — the plan's untrained twin). *Read after the cells (panel
I-1 / I-4 / M-5): the readout is RE-DRAWN at every open-gate refit (a new model object each refit; 5–8 draws per
cell), not frozen; the class-level patch of `predict` also feeds `_refit`'s held-out accuracy into the T047 growth
gate, so the twin's ADMISSION PATH diverges from its candidate's (W hashes differ on OJA seeds 0, 2, 3, 6 and BCM
seeds 4, 5) — a two-item neuter (random scoring AND a growth gate fed random accuracy), and the Frobenius-norm
matching is a no-op for an argmax. The twin conjunct is therefore a FLOOR check (a random readout cannot beat
the majority on any substrate), not a dissociation; the discriminating half is the candidate's. A future twin
must patch scoring only (the trajectory accuracy with the solved beta; randomize the reported prediction) and
key the draw by refit index, not `id()`.* `torch.set_num_threads(1)` in every
worker (CHOSEN; the A/A pair reports the residual). Same rows: every arm at one seed sees the same stream (a read path from
the engine's accuracy to the metacognitive HOLD exists in code — `self_model.observe(self_predict_acc=…)` —
so same rows is a MEASURED property, not a structural one), asserted by identical majority AND persistence
floors across the arms at every seed; a seed whose floors differ across arms is EXCLUDED and the exclusion counted.

### 2.4 The reds — each conjunct quoted, MET / NOT MET one by one

Statistic: the plan's (S = 8, paired one-sided t > 1.895 AND exact sign-flip p ≤ 0.05, ties at |d| ≤ 1e-6 as
zeros, DEGENERATE below 5 non-tied; the planted/random-sign control printed beside every verdict is a
tie-pattern check, not power — panel m8).

- **L-R2(1), the row's letter: "L1b beats the control on live accuracy at the bar."** → paired
  (acc_CAND − acc_BATCH) over the last-third refit rows clears, per candidate. Reported beside it: the same
  on the LIVE instrument (live_accuracy − live_persistence, per arm), and units.
- **L-R2(2), the row's letter: "an untrained twin (frozen random readout) sits BELOW its own persistence floor
  while L1b does not."** As written the incumbent fails it (it sits 0.05–0.64 below a 0.91–1.00
  persistence floor on 8 of 8; §1). **The letter is SCORED in §3.3 as written. Beside it, a LOWERED bar was
  pre-registered as the session's call under the authorization:** paired (acc_CAND − majority floor) clears
  the bar AND paired (acc_TWIN_CAND − majority floor) does NOT clear. The incumbent reads +0.133 (t 1.49) on
  this conjunct: a reachable red for the candidate, and the twin's random readout is expected at or below
  the majority. *(Relabelled in the fold, panel PH-1: the first writing justified the lowered bar by P8
  clause 1 — "a red both arms fail is not a red" — which is the wrong clause: the letter HAD a reachable red
  and lacked a reachable green for the incumbent; P8 says such a criterion is deleted, not loosened. The
  lowered bar is a session choice, stated as one; the disposition is the same under both readings.)*
- **Non-regression:** paired (acc_CAND − acc_BATCH) does not fall past −1.895.
- **Attribution (P8 clause 3):** rule on vs off = L-R2(1); the gate: paired (CAND − CAND_G1), reported, no
  bar; the A/B: paired (OJA − BCM), reported.
- **The A/A:** max |acc_BATCH − acc_BATCH_B| per seed; if it exceeds 1e-6 on any seed the tie tolerance is
  re-derived from it before any verdict is read.

### 2.5 Decision rule (pre-registered)

A candidate LANDS if L-R2(1) and the restated L-R2(2) both clear for it; if both candidates clear, the one
with the larger paired t on L-R2(1) lands and the other is DELETED from the module (P2, one mechanism); if
exactly one clears, it lands and the other is deleted; **if neither clears, the rule is DELETED (P8 clause
5), the record stands, the engine's hook and the live instrument stay** (an instrument, not a method), and L2's
trigger ("L1b's synaptic plateau") is restated in the plan by the session. The brain constructs the survivor
in `_make_neuro`; the strict L-R2 test in `tests/test_real_percept_stream.py` carries the reading either way.

### 2.6 Instrument control

BATCH's per-seed floors and units against `l-parallel-arms.md` §3.4 (stageB3, the same head's src, the same
day): the floors are expected to agree to 1e-6 on every seed (the perception packs have not moved today); a
disagreement is reported as the instrument moving, not as a finding.

## 3. Results

### 3.1 Stage 0 — the gate's positive control (GATE0 × 3 seeds × 800 ticks, 23:16–23:17 UTC, `stage0/`, head `4b77571`)

| seed | fed ticks | distinct gate values | mean | max | fraction > 0.05 | last-200 mean | §2.2 rule |
|---|---|---|---|---|---|---|---|
| 0 | 393 | 392 | 0.071 | 0.244 | 0.567 | **0.030** | NOT usable |
| 1 | 399 | 340 | 0.084 | 0.300 | 0.461 | **0.026** | NOT usable |
| 2 | 399 | 390 | 0.072 | 0.225 | 0.782 | 0.053 | usable |

The ACh plasticity scale is ALIVE at the feed (hundreds of distinct values, 0.0–0.30) — the modulator is not a
pinned gauge — but it DECAYS as the stream stabilizes (prototype recruits stop, the arm's learning-progress
flattens): the last-200 mean sits below the pre-registered 0.05 on 2 of 3 seeds. **Per §2.2 the gate is
WITHDRAWN for this arc: the candidates run UNGATED (g = 1; the gate the brain passes is still recorded per cell),
the three-factor claim is dropped for this delivery, and the reading stands as the instrument's null — a rule
multiplied by 0.03 would have been a two-factor rule wearing a name.** No other fallback was pre-registered and
none is taken. Consequences: the `_G1` arms are the candidate arms (the pinned gate IS the withdrawn gate), so the
cells are 6 arms × 8 seeds (BATCH, BATCH_B, OJA, BCM, TWIN_OJA, TWIN_BCM, all `--ungated`); the gate-attribution
row of §2.4 is VOID. Also read here: the live instrument on the incumbent (BATCH = GATE0's brain): live accuracy
0.547 / 0.849 / 0.807 against live persistence 0.932 / 0.963 / 0.952; units 2 on every seed; refusals 0.
Owed onward (not this arc): a gate that tracks its own running scale (g / EMA(g)) or the arm's baseline-
subtracted RPE, each with its positive control — the three-factor row returns when one of them reads usable.

### 3.2 The cells — 6 arms × 8 seeds × 800 ticks, ungated, 4 at a time (23:19–23:30 UTC; `stage1/`; head(s) 4f26c9c; wall 51–63 s per cell; the daemon was up)

Same rows: majority AND persistence floors identical across the six arms on every seed to 1e-6 — no seed excluded; seeds used [0, 1, 2, 3, 4, 5, 6, 7]. A/A (BATCH vs BATCH_B): max |Δacc| = 0.00e+00; W hashes equal on 8 of 8 seeds (the tie tolerance 1e-6 holds). Perception unified on every cell: True; feed refusals: none. **§2.6's instrument control: BATCH's majority
and persistence floors, units and last-third accuracy equal the same-day incumbent cells of `l-parallel-arms.md`
§3.3 (LIVE at `308f6db`) to 0.0 on 8 of 8 seeds — the perception input did not move today.**

| arm | seed | units (last-third) | acc last third | persistence | majority | live acc / live pers | rule moved / Σ|dW| | gate mean (recorded, unused) | W hash |
|---|---|---|---|---|---|---|---|---|---|
| BATCH | 0 | 2 (1.83) | 0.311 | 0.953 | 0.744 | 0.547 / 0.932 |  /  | 0.071 | 1fd01989 |
| BATCH | 1 | 2 (2.00) | 0.917 | 0.983 | 0.739 | 0.849 / 0.963 |  /  | 0.084 | d655e464 |
| BATCH | 2 | 2 (1.83) | 0.825 | 0.953 | 0.694 | 0.807 / 0.952 |  /  | 0.072 | f2879fd9 |
| BATCH | 3 | 7 (7.00) | 0.656 | 0.911 | 0.319 | 0.653 / 0.902 |  /  | 0.102 | 60750bab |
| BATCH | 4 | 6 (6.00) | 0.628 | 0.950 | 0.508 | 0.590 / 0.907 |  /  | 0.138 | 708075a4 |
| BATCH | 5 | 3 (2.67) | 0.833 | 0.947 | 0.758 | 0.541 / 0.934 |  /  | 0.269 | 5b73e151 |
| BATCH | 6 | 3 (2.50) | 0.867 | 0.933 | 0.506 | 0.630 / 0.951 |  /  | 0.085 | 914883ca |
| BATCH | 7 | 2 (2.00) | 0.947 | 1.000 | 0.653 | 0.771 / 0.992 |  /  | 0.101 | 292414f4 |
| BATCH_B | 0 | 2 (1.83) | 0.311 | 0.953 | 0.744 | 0.547 / 0.932 |  /  | 0.071 | 1fd01989 |
| BATCH_B | 1 | 2 (2.00) | 0.917 | 0.983 | 0.739 | 0.849 / 0.963 |  /  | 0.084 | d655e464 |
| BATCH_B | 2 | 2 (1.83) | 0.825 | 0.953 | 0.694 | 0.807 / 0.952 |  /  | 0.072 | f2879fd9 |
| BATCH_B | 3 | 7 (7.00) | 0.656 | 0.911 | 0.319 | 0.653 / 0.902 |  /  | 0.102 | 60750bab |
| BATCH_B | 4 | 6 (6.00) | 0.628 | 0.950 | 0.508 | 0.590 / 0.907 |  /  | 0.138 | 708075a4 |
| BATCH_B | 5 | 3 (2.67) | 0.833 | 0.947 | 0.758 | 0.541 / 0.934 |  /  | 0.269 | 5b73e151 |
| BATCH_B | 6 | 3 (2.50) | 0.867 | 0.933 | 0.506 | 0.630 / 0.951 |  /  | 0.085 | 914883ca |
| BATCH_B | 7 | 2 (2.00) | 0.947 | 1.000 | 0.653 | 0.771 / 0.992 |  /  | 0.101 | 292414f4 |
| OJA | 0 | 6 (6.00) | 0.211 | 0.953 | 0.744 | 0.629 / 0.932 | 369 / 1616.739634 | 0.071 | 5513c74c |
| OJA | 1 | 5 (5.00) | 0.933 | 0.983 | 0.739 | 0.910 / 0.963 | 377 / 1953.853016 | 0.084 | 3adf9e82 |
| OJA | 2 | 4 (4.00) | 0.089 | 0.953 | 0.694 | 0.176 / 0.952 | 374 / 2890.998973 | 0.072 | 52872921 |
| OJA | 3 | 6 (4.67) | 0.703 | 0.911 | 0.319 | 0.426 / 0.902 | 378 / 3078.102853 | 0.102 | e1fd65e0 |
| OJA | 4 | 15 (15.00) | 0.539 | 0.950 | 0.508 | 0.516 / 0.907 | 376 / 3767.908355 | 0.138 | 46536b40 |
| OJA | 5 | 4 (2.83) | 0.872 | 0.947 | 0.758 | 0.607 / 0.934 | 377 / 814.542108 | 0.269 | c79a8c3c |
| OJA | 6 | 6 (5.67) | 0.608 | 0.933 | 0.506 | 0.570 / 0.951 | 370 / 1995.648565 | 0.085 | 799fe58b |
| OJA | 7 | 2 (2.00) | 0.458 | 1.000 | 0.653 | 0.840 / 0.992 | 376 / 163.928363 | 0.101 | 10d75a4b |
| BCM | 0 | 4 (4.00) | 0.211 | 0.953 | 0.744 | 0.664 / 0.932 | 369 / 372.306008 | 0.071 | 97f56271 |
| BCM | 1 | 2 (2.00) | 0.917 | 0.983 | 0.739 | 0.849 / 0.963 | 377 / 203.003524 | 0.084 | a61df3b0 |
| BCM | 2 | 3 (3.00) | 0.783 | 0.953 | 0.694 | 0.775 / 0.952 | 374 / 528.428595 | 0.072 | 860f6713 |
| BCM | 3 | 3 (3.00) | 0.214 | 0.911 | 0.319 | 0.405 / 0.902 | 378 / 905.460525 | 0.102 | fbf2b6a2 |
| BCM | 4 | 5 (5.00) | 0.939 | 0.950 | 0.508 | 0.638 / 0.907 | 376 / 1289.526463 | 0.138 | cedb5ed4 |
| BCM | 5 | 2 (1.67) | 0.842 | 0.947 | 0.758 | 0.724 / 0.934 | 377 / 286.327029 | 0.269 | 7f5cdf5a |
| BCM | 6 | 4 (3.67) | 0.608 | 0.933 | 0.506 | 0.597 / 0.951 | 370 / 650.018584 | 0.085 | bd0f2eb3 |
| BCM | 7 | 2 (2.00) | 0.472 | 1.000 | 0.653 | 0.840 / 0.992 | 376 / 70.200062 | 0.101 | d0849ffa |
| TWIN_OJA | 0 | 6 (6.00) | 0.000 | 0.953 | 0.744 | 0.192 / 0.932 | 369 / 1616.402137 | 0.071 | 48ad7191 |
| TWIN_OJA | 1 | 5 (5.00) | 0.300 | 0.983 | 0.739 | 0.239 / 0.963 | 377 / 1953.853016 | 0.084 | 3adf9e82 |
| TWIN_OJA | 2 | 2 (2.17) | 0.139 | 0.953 | 0.694 | 0.152 / 0.952 | 374 / 2975.794252 | 0.072 | ce89f5b7 |
| TWIN_OJA | 3 | 5 (4.50) | 0.167 | 0.911 | 0.319 | 0.190 / 0.902 | 378 / 3014.42398 | 0.102 | 6aac78f7 |
| TWIN_OJA | 4 | 15 (15.00) | 0.508 | 0.950 | 0.508 | 0.399 / 0.907 | 376 / 3767.908355 | 0.138 | 46536b40 |
| TWIN_OJA | 5 | 4 (2.83) | 0.231 | 0.947 | 0.758 | 0.215 / 0.934 | 377 / 814.542108 | 0.269 | c79a8c3c |
| TWIN_OJA | 6 | 9 (6.67) | 0.306 | 0.933 | 0.506 | 0.286 / 0.951 | 370 / 2341.328958 | 0.085 | 20ee60cd |
| TWIN_OJA | 7 | 2 (2.00) | 0.331 | 1.000 | 0.653 | 0.551 / 0.992 | 376 / 163.928363 | 0.101 | 10d75a4b |
| TWIN_BCM | 0 | 4 (4.00) | 0.000 | 0.953 | 0.744 | 0.320 / 0.932 | 369 / 372.306008 | 0.071 | 97f56271 |
| TWIN_BCM | 1 | 2 (2.00) | 0.250 | 0.983 | 0.739 | 0.342 / 0.963 | 377 / 203.003524 | 0.084 | a61df3b0 |
| TWIN_BCM | 2 | 3 (3.00) | 0.033 | 0.953 | 0.694 | 0.056 / 0.952 | 374 / 528.428595 | 0.072 | 860f6713 |
| TWIN_BCM | 3 | 3 (3.00) | 0.000 | 0.911 | 0.319 | 0.159 / 0.902 | 378 / 905.460525 | 0.102 | fbf2b6a2 |
| TWIN_BCM | 4 | 4 (4.33) | 0.022 | 0.950 | 0.508 | 0.090 / 0.907 | 376 / 1466.907081 | 0.138 | a72a60e2 |
| TWIN_BCM | 5 | 2 (1.83) | 0.053 | 0.947 | 0.758 | 0.090 / 0.934 | 377 / 286.327014 | 0.269 | f2015609 |
| TWIN_BCM | 6 | 4 (3.67) | 0.022 | 0.933 | 0.506 | 0.065 / 0.951 | 370 / 650.018584 | 0.085 | bd0f2eb3 |
| TWIN_BCM | 7 | 2 (2.00) | 0.331 | 1.000 | 0.653 | 0.527 / 0.992 | 376 / 70.200062 | 0.101 | d0849ffa |

### 3.3 Verdict — each conjunct of §2.4 against its wording

- **OJA, L-R2(1) ("beats the control on live accuracy at the bar"): paired (acc_OJA − acc_BATCH) clears** — t = -1.959, exact p = 0.9648, 3 / 5 / 0, mean -0.196 (planted t inf, null clears 0.045) → **NOT MET**.
- **OJA, L-R2(2) restated: paired (acc_OJA − majority floor) clears** — t = -0.511, exact p = 0.6953, 5 / 3 / 0, mean -0.064 (planted t inf, null clears 0.050) → **NOT MET**; **AND TWIN_OJA does NOT clear** — t = -4.248, exact p = 1.0000, 0 / 7 / 1, mean -0.368 (planted t 7.00, null clears 0.055) → twin holds (does not clear).
- OJA, non-regression (not below −1.895): t -1.959 → NOT MET. Live instrument: (live acc − live pers) t = -4.658, exact p = 1.0000, 0 / 8 / 0, mean -0.357 (planted t inf, null clears 0.040); (live acc − BATCH's live acc) t = -1.038, exact p = 0.7969, 4 / 4 / 0, mean -0.089 (planted t inf, null clears 0.010). Units [6, 5, 4, 6, 15, 4, 6, 2]; rule moved per cell [369, 377, 374, 378, 376, 377, 370, 376]; Σ|dW| [1616.739634, 1953.853016, 2890.998973, 3078.102853, 3767.908355, 814.542108, 1995.648565, 163.928363]; the recorded (unused) gate mean [0.0652, 0.0768, 0.0672, 0.0974, 0.1311, 0.2714, 0.0784, 0.0971].
- **BCM, L-R2(1) ("beats the control on live accuracy at the bar"): paired (acc_BCM − acc_BATCH) clears** — t = -1.358, exact p = 0.8984, 2 / 5 / 1, mean -0.125 (planted t 7.00, null clears 0.040) → **NOT MET**.
- **BCM, L-R2(2) restated: paired (acc_BCM − majority floor) clears** — t = 0.079, exact p = 0.4922, 5 / 3 / 0, mean +0.008 (planted t inf, null clears 0.050) → **NOT MET**; **AND TWIN_BCM does NOT clear** — t = -9.105, exact p = 1.0000, 0 / 8 / 0, mean -0.526 (planted t inf, null clears 0.050) → twin holds (does not clear).
- BCM, non-regression (not below −1.895): t -1.358 → MET. Live instrument: (live acc − live pers) t = -5.815, exact p = 1.0000, 0 / 8 / 0, mean -0.255 (planted t inf, null clears 0.035); (live acc − BATCH's live acc) t = 0.283, exact p = 0.3906, 4 / 3 / 1, mean +0.013 (planted t 7.00, null clears 0.055). Units [4, 2, 3, 3, 5, 2, 4, 2]; rule moved per cell [369, 377, 374, 378, 376, 377, 370, 376]; Σ|dW| [372.306008, 203.003524, 528.428595, 905.460525, 1289.526463, 286.327029, 650.018584, 70.200062]; the recorded (unused) gate mean [0.0652, 0.0768, 0.0672, 0.0974, 0.1311, 0.2714, 0.0784, 0.0971].
- **L-R2(2) BY THE LETTER ("an untrained twin sits BELOW its own persistence floor while L1b does not"):**
  the twins sit below persistence on 8 of 8 (TWIN_OJA t −13.56, mean −0.706; TWIN_BCM t −23.59, mean −0.865)
  and so do BOTH CANDIDATES on 8 of 8 (OJA acc − persistence t −3.81, mean −0.402; BCM t −3.22, mean −0.331;
  the incumbent t −2.90, mean −0.206) → **NOT MET** for both, harder than under the lowered bar.
- The counters: "moved" counts learn() calls that REACHED the update (a model present, an open gate, a
  non-silent percept), not rows that changed; Σ|dW| carries the charge claim. The two gate means differ by
  denominator: §3.2's column is the probe's mean over every fed tick (0.065–0.271 across seeds), the rule's
  `mean_gate` is over post-first-model calls only.
- **A/B (OJA − BCM):** t = -0.583, exact p = 0.6562, 3 / 3 / 2, mean -0.071 (planted t 4.58, null clears 0.050).
- **The incumbent (BATCH):** units [2, 2, 2, 7, 6, 3, 3, 2]; acc [0.3111, 0.9167, 0.825, 0.6556, 0.6278, 0.8333, 0.8667, 0.9472]; (acc − majority) t = 1.487, exact p = 0.0898, 7 / 1 / 0, mean +0.133 (planted t inf, null clears 0.065); (acc − persistence) t = -2.898, exact p = 1.0000, 0 / 8 / 0, mean -0.206 (planted t inf, null clears 0.045). Floors per seed (majority, persistence): {'0': [0.7444, 0.9528], '1': [0.7389, 0.9833], '2': [0.6944, 0.9528], '3': [0.3194, 0.9111], '4': [0.5083, 0.95], '5': [0.7583, 0.9472], '6': [0.5056, 0.9333], '7': [0.6528, 1.0]}.
- The gate-attribution row of §2.4 is VOID (the gate withdrawn at Stage 0, §3.1).

## 4. Verdict and disposition — the pre-registered decision rule (§2.5), applied as written

- **L-R2(1), both candidates: NOT MET.** OJA (acc − BATCH) t -1.959, p 0.9648, 3/5/0, mean -0.196; BCM t -1.358, p 0.8984, 2/5/1, mean -0.125.
- **L-R2(2) restated, both candidates: NOT MET** on the candidate half (OJA t -0.511, p 0.6953, 5/3/0, mean -0.064; BCM
  t 0.079, p 0.4922, 5/3/0, mean +0.008); the twin half held on both (TWIN_OJA t -4.248, p 1.0000, 0/7/1, mean -0.368; TWIN_BCM
  t -9.105, p 1.0000, 0/8/0, mean -0.526) — the frozen random readout sits far below the majority, so the conjunct's red
  was reachable and the candidates did not reach the green.
- **Non-regression: OJA FELL PAST THE BAR** (t −1.959 < −1.895; 5 losses, seeds 2 and 7 at −0.74 and −0.49; the
  mirrored exact p in the harm direction 0.039) — a measured HARM of the rule AS DEPLOYED. **Its cause, read from the
  refit rows (panel M-1, verified by the session): input rows moving under a readout the growth gate holds STALE.**
  Every collapse follows a growth row: the T047 gate then CLOSES (its stop is sticky) and each closed-gate refit
  scores the RETAINED model — whose `W` the rule keeps moving on every wake percept while `beta` stays at the last
  open-gate solve — so the predictions degenerate to a CONSTANT non-majority class: OJA seed 2 reads 0.083–0.100 for
  six rows (a ~0.08 minority class under a 0.69 majority), seed 7 falls 0.983 → 0.267 within one refit interval, and on
  seeds 0 and 7 OJA and BCM read IDENTICAL accuracies for the last 4–6 rows with different substrates (acc + majority
  ≈ 1 − the minor classes). A readout re-solved on the moved rows could not produce that; the incumbent shows the
  SAME mode with frozen `W` (BATCH seed 0 rows 15–18 at 0.067–0.167 under a 0.72–0.78 majority, recovering only at
  the next open refit). The first writing's cause — "aligned rows lose the feature diversity the readout uses" — is
  therefore an UNMEASURED hypothesis (no diversity statistic was taken in the cells), retracted as a cause. **BCM did
  not fall past the bar** (t −1.358) and did not gain. The live instrument (which reads the stale-readout interaction
  on every tick): OJA live acc − the incumbent's live acc mean −0.089; BCM +0.013 (t 0.28); every arm's live accuracy
  sits below its live persistence floor (OJA t −4.66; BCM t −5.82).
- **Side reading (reported):** OJA ADMITTED MORE UNITS than the incumbent on 6 of 8 seeds ([6, 5, 4, 6, 15, 4, 6, 2] vs
  [2, 2, 2, 7, 6, 3, 3, 2]): the moved rows change the admission path (a fit seeded from rotated rows leaves residual the
  greedy pursuit fills with new units), so the accuracy comparison is between substrates that differ in width as well
  as in weights — attributable to the rule (the only difference between the arms) but not to the moved rows ALONE. BCM
  admitted [4, 2, 3, 3, 5, 2, 4, 2].
- **A/B:** (OJA − BCM) t -0.583, p 0.6562, 3/3/2, mean -0.071 — not resolvable; moot under the rule.
- **Instrument:** same rows on 8 of 8 seeds (floors identical across the six arms to 1e-6, no seed excluded); the A/A
  pair bit-identical (max |Δacc| 0.0, W hashes equal on 8 of 8) under the thread pin, so the 1e-6
  tie tolerance holds; the rule ran on every rule cell (369–378 moves of ~395 fed percepts; Σ|dW| OJA
  164–3768, BCM 70–1290); the withdrawn gate was recorded (mean 0.065–0.271,
  consumed by nothing); perception unified on every cell; refusals none; one head (`4f26c9c`). The incumbent
  (BATCH): units [2, 2, 2, 7, 6, 3, 3, 2], (acc − majority) t 1.487, p 0.0898, 7/1/0, mean +0.133, (acc − persistence) t -2.898, p 1.0000, 0/8/0, mean -0.206.

**DISPOSITION — the rule is DELETED (P8 clause 5, the licence working), as §2.5 pre-registered: neither candidate
cleared, under the lowered bar or the letter.** What stays: the LIVE instrument (`live_accuracy` / `live_persistence`
on state(), scored from the PREVIOUS percept before this tick's transition is buffered — the percept it predicts
from was seen by the model one tick earlier, unsupervised, no label leak, not unseen) with its own brain-free test
asserting that ordering — an instrument, not a method. What goes (the fold completed the deletion the first writing
left half done, panel I-2 / I-5 / PH-6): `plasticity/grown_input_rule.py`, its tests, its registration rows, its
census adjudication (the census regenerated), the engine's kind-name constructor branch — AND, departing from
§2.5's "what stays" as the session's call because P3 does not exempt a dormant wire by disclosure: the engine's
`online_rule` hook and its charge counter, the `gate` kwarg and the brain's ACh pass at the feed (a write-only wire
with no consumer), and BCM's theta bookkeeping (a refit carry-over, a prune re-index, an always-empty npz member
and a restore read with no writer left). The plug point is described in the engine's docstring; the next rule
re-adds it with its consumer. The pre-registered in-tree strict L-R2 test (§2.5, "carries the reading either
way") is NOT carried on a deletion — its arms would import the deleted rule; this record and Part F's L-R2 row
carry the reading (panel I-3 / PH-3 / M-2). **The negative to keep (the re-proposal guard), SCOPED to what was
measured (panel PH-7 / M-1):** at μ = 0.05 (CHOSEN, untuned), T = 15, ungated, 800 ticks, with the readout
re-solved only at open-gate refits, on this stream, a norm-preserving Hebbian step on the grown units' input rows
HARMS next-category accuracy past the bar and BCM neither harms nor helps; the harm's measured mechanism is the
stale-readout interaction above, and "diversity erosion" is a hypothesis without a statistic. Before that negative
is generalized it owes one cell: a rule arm with the growth-gate verdict logged per refit row, or a
re-solve-beta-on-every-refit control. **What it points at:** the READOUT must co-adapt with any moving input
weights — L1c (the readout online) precedes any second input-weight rule, and that rule's candidate (a scalar
broadcast of the live prediction's correctness gating the same local step, the R-STDP shape) is pre-registered in
its own arc after L1c. Owed onward: the μ derivation (P1), a gate that is alive on the tail (§3.1). **The plan's L2 trigger ("L1b's
synaptic plateau") is restated by the session (Part H's authorization): L2 reads the online READOUT's residual
plateau if L1c lands, else the T047 growth gate's own refit-accuracy plateau (already live) and the row reduces to
its attribution red.**

## 5. Commits of this arc (stacked on `cdf9454`)

| commit | what | landed on |
|---|---|---|
| `0fc7a89` 1 | this record §1–2 (pre-registration) | before any cell |
| `4b77571` 2 | the build: `plasticity/grown_input_rule`, the engine's hook + live instrument + theta bookkeeping, the ACh gate at the feed, the census row, four brain-free reds | the module self-check + 9 brain-free tests + both claim gates |
| `4f26c9c` 3 | Stage 0 read: the gate withdrawn per §2.2 (`gated=False`), §3.1 | 3 GATE0 cells |
| `34583a0` 4 | the cells (§3.2–3.3), the verdict (§4), the DELETION on the null, the plan's L1b / L2 / L-R2 rows | 48 cells; P8 clause 5 |
| 5 (this commit, the fold) | the record panel's fold: the deletion completed (theta plumbing, the hook and the gate wire — P3), the live-instrument test asserts the ordering, the twin restated, the letter scored, the harm's cause rewritten and the negative scoped, §6 | battery-verified only; the fold of a review is unreviewed code |

The archived `grown_input_rule.py` under `archive/l1b-2026-09-05/` is a pre-commit DRAFT (its first-form reds); the
measured artifact is `git show 4f26c9c:src/snn_data_lake/plasticity/grown_input_rule.py` (panel M-7).

## 6. Record panel (three lenses, Workflow; `record_panel.json`) — ledger

All three lenses: the disposition STANDS (deletion on the null). Findings and dispositions; "folded" = commit 5.

| id | finding (one line) | disposition |
|---|---|---|
| I-1 MAJOR | the twin's `predict` patch leaks into the growth gate; the twin's substrate diverged on 6 of 16 twin cells | folded: §2.3 restated as a two-item neuter; the twin half is a floor check; a scoring-only twin prescribed |
| I-2 MAJOR | BCM's theta plumbing survived with no writer (P3) | folded: deleted from the engine |
| I-3 / PH-3 / M-2 | the pre-registered in-tree strict L-R2 test was not delivered and not disclosed | folded: §4 states why |
| I-4 | "frozen" is per model object — re-drawn at each open refit | folded: §2.3 wording; the plan's row |
| I-5 / PH-6 | the hook and the ACh gate wire are a write-only wire with no consumer (P3 shapes 4 and 7) | folded: REMOVED (P3 over §2.5's "stays"; the session's call) |
| I-6 | the instrument test's hook assertion was vacuous; the ordering not asserted | folded: the test asserts the prediction precedes the buffer, hand-computed |
| I-7..I-10 NOTE | the live path, the gates, the tick cost (~0.3 % of the wake tick), the state surfaces as claimed | none |
| PH-1 MAJOR | the restated conjunct is a LOWERED bar justified by the wrong P8 clause | folded: the letter scored (NOT MET harder); relabelled as the session's call |
| PH-2 | two digits (+0.11 / t 1.29; 0.05–0.6) | folded: +0.133 / t 1.49; 0.05–0.64 |
| PH-4 / M-5 | the twin controls more than the record said | folded with I-1 |
| PH-5 / M-6 | "moved" counts calls reaching the update; two gate means, one label | folded: §3.3 |
| PH-7 | the negative overreached one operating point; μ derivation owed | folded: scoped; owed onward |
| PH-8..PH-10 NOTE | the pre-registration unedited after commit 1; the control scoping honest; every §3 digit reproduces | none |
| M-1 MAJOR | the harm's stated cause is wrong: rows moving under a readout the closed gate holds stale; the incumbent shows the same mode | folded: §4 rewritten; the negative scoped; the owed cell named; the next candidate re-pointed at the readout (L1c first) |
| M-3 | §2.6's control never reported | folded: §3.2 (identical to 0.0 on 8 of 8) |
| M-4 | "the engine feeds nothing back" false by code; same rows by measurement | folded: §2.3 |
| M-7 | the archived module is a draft; the git hash is the artifact | folded: §5 note |
| M-8..M-12 NOTE | same rows, A/A, every statistic reproduced (the harm also clears the mirrored p 0.039), Stage 0, the instrument's ordering, the brain-free reds | none |
