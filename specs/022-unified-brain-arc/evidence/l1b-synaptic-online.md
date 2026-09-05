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
  `l-parallel-arms.md` §3.3) and above its majority floor on 7 of 8 (mean +0.11, t 1.29, not at the bar).
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
**TWIN_BCM** (the rule on, the readout FROZEN RANDOM: `NGSNN.predict` scores through a seeded random beta of the
solved beta's Frobenius norm, one per model — the plan's untrained twin). `torch.set_num_threads(1)` in every
worker (CHOSEN; the A/A pair reports the residual). Same rows: every arm at one seed sees the same stream (the
engine feeds nothing back to the arm), asserted by identical majority AND persistence floors across the
arms at every seed; a seed whose floors differ across arms is EXCLUDED and the exclusion counted.

### 2.4 The reds — each conjunct quoted, MET / NOT MET one by one

Statistic: the plan's (S = 8, paired one-sided t > 1.895 AND exact sign-flip p ≤ 0.05, ties at |d| ≤ 1e-6 as
zeros, DEGENERATE below 5 non-tied; the planted/random-sign control printed beside every verdict is a
tie-pattern check, not power — panel m8).

- **L-R2(1), the row's letter: "L1b beats the control on live accuracy at the bar."** → paired
  (acc_CAND − acc_BATCH) over the last-third refit rows clears, per candidate. Reported beside it: the same
  on the LIVE instrument (live_accuracy − live_persistence, per arm), and units.
- **L-R2(2), the row's letter: "an untrained twin (frozen random readout) sits BELOW its own persistence floor
  while L1b does not."** As written this is failed by EVERY live arm (all sit 0.05–0.6 below a 0.91–1.00
  persistence floor; §1) — a red both arms fail is not a red. **RESTATED at pre-registration (the session's
  call under the authorization):** paired (acc_CAND − majority floor) clears the bar AND paired
  (acc_TWIN_CAND − majority floor) does NOT clear. The incumbent reads +0.11 (t 1.29) on this conjunct: a
  reachable red for the candidate, and the twin's random readout is expected at or below the majority.
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
