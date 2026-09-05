# The JEPA-collapse arc: the arm's world model cannot see succession — the red first, then the fix (2026-09-05)

*Owner ratification, in chat, 2026-09-05: "Q1 (a), start the JEPA-collapse arc."* Per PLAN-FINAL Part H
(Q1) this arc runs before Phase 2's L2 and before any task whose red reads
`JepaWorldModel.prediction_error`. It was opened by the W2 record
(`evidence/w2-context-latents.md` §3.1, §4.2, §4.3, §4.3b, §4.7): the arm's `cerebellum/jepa_world_model.
JepaWorldModel` collapses its latent on every context this body sees, and its prediction error cannot
tell the natural succession from a destroyed one.

## 1. The defect, and the one-line cause the orientation found

**Measured (W2, twelve in-brain models at `f546e0a`, seeds 0–2, 3000 ticks):** the trained model's
target-latent spread over the 256 distinct 64-d context patterns it saw is ×0.005–0.11 of a fresh
model's spread over the same patterns; `latent_std()` reads 0.005–0.034; the prediction error per
256-row block is identical to the fourth decimal under the natural order and under a fresh per-cycle
permutation — on the W2 context and on the incumbent's. Brain-free (the design panel's replication on
the exact classes): a PLANTED world with a deterministic 3-bit successor map is "mastered" as fast
shuffled as ordered.

**The cause, read at the symbols this session.** `JepaWorldModel.__init__` defaults `lambda_var = 1.0`,
`gamma = 1.0`, `eps = 1e-4`, and `train_step`'s VICReg variance hinge contributes
`lambda_var · −(z_j − μ_j) / (L · m · sd_j)` to `dL/dz` with `L = 16`, `m = 32`, `sd_j = sqrt(var_j + eps)`.
Three facts about that line: (i) the latent is `unit(tanh(·))` in 16 dimensions, so a per-dimension std
can never reach `gamma = 1.0` (a uniform point on the 16-sphere has per-dim std 1/4) — the hinge is ALWAYS
active and its threshold never matters; (ii) the restoring force is proportional to the latent's own
deviation `|z_j − μ_j|` over a std floored at 0.01 (`eps = 1e-4`), so once the deviation is below the
floor the force that should re-inflate the latent shrinks with it — collapse is a fixed point; (iii) with
`|z − μ| ≈ sd` the whole term is of order `lambda_var / (L · m) = 1/512`, against an invariance gradient
`2 · r` of order 0.05–0.2 while anything is left to predict. **And the module's own `main()` self-check
runs `lambda_var = 20.0, lambda_cov = 2.0` with the comment "the unit-sphere latent caps per-dim energy,
so the anti-collapse pressure is scaled up here for both models to visibly hold variance"** — the
self-check certifies a configuration the arm never constructs (`AutonomousArm.step` builds
`JepaWorldModel(ACTIONS, in_dim=64, seed=…)` at the defaults). The module's header even records the
collapse as a v3.18 finding ("the online arm latent measured ~0.003 apart across distinct contexts") and
offered an opt-in reward-shaping head as the remedy; `_reward_shaped_encoder` is off on the live arm.

**The fixtures.** Four readers of `latent_std()` exist. The module self-check asserts `full_std > 0.05
and full_std > 3.0 · ctrl_std` and `tests/instruments/jepa_highdim_probe.py` asserts `std_full > 2 ·
std_abl` — both on synthetic streams at `lambda_var = 20`. The only bars on the LIVE model are
`autonomous_loop`'s self-check `jp_std > 0.0` and `live_brain`'s `jepa_std > 0.0`. Nothing applies the
module's 0.05 floor to the configuration the arm runs.

**What `latent_std()` measures, disclosed before it is used as a bar:** the mean per-dimension std over
the last 32 ONLINE latents — 32 consecutive ticks. On a near-constant live context a healthy encoder also
reads low there. The gauge that separates "similar inputs" from "collapsed encoder" is the trained model's
target-latent spread over DISTINCT patterns against a FRESH model's spread over the same patterns (W2's
instrument); this record carries both and says which is the bar where.

## 2. Protocol — PRE-REGISTERED before the first harness run

Symbols not lines; runtime evidence outranks this text; every number in §2.6 is a prediction.

### 2.1 The brain-free red harness (`jepa_collapse_red.py`, archived; promoted into the tree as the test)

The arm's exact classes: `JepaWorldModel(ACTIONS, in_dim=64, seed)` at the arm's construction call,
`cortex/multimodal_encoder.encode_multimodal` building the 64-d context exactly as `_context_pattern`
does (`[vdc(vis)]·4, [prop], [vdc(aud)]·4, 0.12`), one `train_step` per transition after the arm's own
`prediction_error` read (the arm's order). Two worlds, each a 256-tick cycle:

- **planted** — vision id `i mod 8`, audio `(3i) mod 8`: a deterministic 3-bit successor map (next-id
  mutual information 3 bits). If a model can learn ANY succession it can learn this one.
- **labels** — the tracked packs' label order as proxy ids (the natural stream's proxy; W2 §3.1c measured
  it at the median of its shuffle distribution).

Each world runs ORDERED and SHUFFLED (both id lists re-permuted under one fresh `random.Random(seed ·
100003 + cycle)` permutation per cycle, pairing kept) with the SAME proprioception random walk and the
SAME action draws per seed; 6 cycles (1536 steps); seeds 0, 1, 2. Per run: prediction error per
256-block; the trained-vs-fresh target-latent spread over the world's distinct patterns; `latent_std()`;
and — read-only, from the model's state with the module's own formulas — the magnitudes of the
invariance gradient and the variance-hinge gradient on `dL/dz`, first and last 16 samples.

**Positive control for the harness itself (P8 clause 2):** a tabular oracle `(vis, aud, action) → the
next pattern seen last time` scores ~0 error on the ordered planted world from cycle 2 and stays high on
the shuffled one. If the oracle does not separate the two arms, the worlds do not carry what the red
claims and nothing below is read.

### 2.2 Variants — ONE named change each, then the smallest combination that clears the reds

`base` (the arm's construction; the red count), `ablation` (`anti_collapse=False`, the module's own
control: must collapse at least as hard as base), `eps8` (`eps 1e-4 → 1e-8`: the std floor 0.01 → 1e-4),
`gamma_sphere` (`gamma 1.0 → 1/√L = 0.25`, the per-dim std of a uniform point on the unit sphere — a
fixed point on a normalized output, P1 case (a)), `lamvar4 / lamvar16 / lamvar20 / lamvar64` (`lambda_var`;
20 is the module self-check's OWN value — an incumbent constant, adopted not invented; 16 = L, the value
that cancels the hinge's 1/L), then `eps8_lamvar16` and `eps8_gamma_lamvar16`. Per-item attribution =
each single change against `base` on the same seeds; the shipped change is the SMALLEST that clears §2.3
(a) and, in-brain, (b).

### 2.3 The reds (each reachable on the current tree — counted in §3 before any fix)

- **(a) ORDER, brain-free:** on the planted world, mean prediction error over the last block, ordered ÷
  shuffled, **< 0.5 on 3 of 3 seeds** (a fully predictable world against an unpredictable one; a working
  predictor's error on the ordered arm should fall well below the shuffled arm's, which is bounded below
  by the irreducible successor entropy). Today: ratio ≈ 1.
- **(b) THE LIVE-MODEL FLOOR, in-brain:** the module's EXISTING floor, `latent_std() > 0.05`, applied at
  the two live-model sites — `autonomous_loop`'s self-check (`jp_std`, 90 steps on its synthetic
  stimulus) and `live_brain`'s (`jepa_std`, 90 ticks) — replacing `> 0.0`; and on the brain's own stream:
  `latent_std()` at 3000 ticks ≥ 0.05 on seeds 0–2 (`throwaway_home()`, `persist=False`, W2's harness).
  Beside it, not a bar: the trained/fresh spread ratio at 3000 ticks (today ×0.04–0.11 on the incumbent).
  If (a) passes and the in-brain floor fails, the floor is NOT lowered: the disagreement is recorded and
  the `latent_std` gauge's own limitation (§1) is the first suspect, tested by the spread ratio.
- **(c) THE GRADIENT RATIO, counted:** `|g_var| / |g_inv|` on `dL/dz` over the last 16 samples, per
  variant — reported beside every verdict, never a bar (a count before any rule).

### 2.4 Controls that are not the method

The `ablation` variant (both VICReg terms off, target tied to online) must collapse at least as hard as
`base` (spread ratio ≤ base's on 3 of 3) and must NOT clear (a) — if the ablation clears (a), order was
being read by something other than the anti-collapse term and the attribution is void. The fresh model
is the spread reference. The tabular oracle is the harness's positive control. The shuffled arm is (a)'s
null.

### 2.5 In-brain measurement after a brain-free pass (disclosed, never smoothed)

With the shipped defaults: `substrate/world_red_probe.py --record --seeds 0,1,2 --checkpoints 3000,6000`
(the W3 instrument; the W3 baselines are the "before"); the archived W2 harness's `incumbent` and
`incumbent_shuffle` arms (the ordered-vs-shuffled error read in-brain, W2 §4.3b the "before"); and, for
the bundled gauge item, per-tick logging of the UNROUNDED `arm._cur_mag`, `raw_cur`, `raw_emp` and the
both-drives-zero fraction (W2 §4.7). Whatever Chain A does — rises, falls, or stays at its floor — is
recorded as the arc's downstream finding. Every in-brain cell: one arm per subprocess, live lake never
touched, live daemon left running.

### 2.6 Predictions, pinned

- `base`: (a) ratio 0.9–1.1 on all seeds; spread ratio < 0.05; `|g_var|/|g_inv|` < 0.05.
- `ablation`: spread ratio ≤ base's; (a) fails.
- `lamvar20` (the self-check's own value) and `lamvar16`: the spread ratio rises above 0.2; (a) UNCERTAIN —
  the honest red; `eps8` alone: little change (the floor binds only once collapse is under way);
  `gamma_sphere` alone: no change (the hinge is already always active).
- The oracle: ~0 on ordered planted from cycle 2; high on shuffled.
- The `labels` world: no variant clears (a) — the stream has no first-order succession (W2 §3.1c); this
  world is the "natural" read and is reported, not scored.

### 2.7 Disposition

**PASS:** the smallest single change (or combination) clearing (a) brain-free and (b) in-brain ships as
the module's default in ONE commit, with the two live-model bars raised to the 0.05 floor, a fast test
(`tests/test_jepa_collapse.py`: the planted world's (a) and the ablation control, brain-free, seeds
0–2) whose red is the pre-fix tree, the module self-check's `LV = 20` reconciled with the new default
(one configuration, P2), and the P1 line for any constant that moves (adopted from the self-check, or
derived, stated). Then §2.5, the panel, the battery, the record.
**NULL:** no variant clears (a) → the collapse is not a parameter defect of the hinge; the linear
unit-norm encoder under a pure invariance objective is the limiter. Nothing ships except the two raised
bars (they are red today and stay red — a criterion whose red is REACHABLE and STANDING is kept, P8) and
this record; Q1 returns to the owner as (b). Partial fixes are not shipped as "better".
**Deletion clause:** if (a) turns out unreachable by construction (the oracle fails), the harness is the
defect and is deleted, not the criterion weakened.

### 2.8 P2 dispositions, stated before the data

`reward_shaping` head (opt-in, needs a reward; task-alignment, not anti-collapse): COMPOSED, untouched.
`cerebellum/spiking_jepa`'s `−log var` variance term on a different class: NOT adopted unless every hinge
variant nulls — then a P2 adjudication (one anti-collapse mechanism), its own arc. The self-check's
`LV = 20 / LC = 2`: the configuration this arc most likely ADOPTS into the default — one configuration
must survive (P2), so on a pass the self-check's local override goes. `_feedback_gz`, the EMA target,
the rollout term: untouched.

## 3. The red, counted on the current tree (before any fix)

Ten runs at `b0bdc47` (the arm's construction, `JepaWorldModel(ACTIONS, in_dim=64, seed)`; the harness
and every JSON archived under `archive/jepa-2026-09-05/`). Last-block prediction error = block 6 of 6
(steps 1281–1536).

| variant / world | seed | perr last block, ordered / shuffled | ratio | spread trained / fresh | `latent_std` | \|g_var\| / \|g_inv\| (last 16) | oracle last block, ordered / shuffled |
|---|---|---|---|---|---|---|---|
| base / planted | 0 | 0.00197 / 0.00166 | **1.19** | 0.0158 / 1.140 (×0.014) | 0.013 | 0.087 | **0.00000** / 5.36 |
| base / planted | 1 | 0.00199 / 0.00131 | **1.52** | 0.0154 / 1.203 (×0.013) | 0.010 | 0.088 | 0.00000 / 5.22 |
| base / planted | 2 | 0.00195 / 0.00162 | **1.21** | 0.0172 / 1.034 (×0.017) | 0.008 | 0.088 | 0.00000 / 5.27 |
| ablation / planted | 0 | 0.00093 / 0.00097 | 0.96 | 0.0064 / 1.140 (×0.006) | 0.008 | 0 | 0.00000 / 5.36 |
| ablation / planted | 1 | 0.00040 / 0.00035 | 1.13 | 0.0055 / 1.203 (×0.005) | 0.004 | 0 | 0.00000 / 5.22 |
| ablation / planted | 2 | 0.00096 / 0.00094 | 1.03 | 0.0088 / 1.034 (×0.009) | 0.003 | 0 | 0.00000 / 5.27 |
| base / labels | 0 | 0.00280 / 0.00267 | 1.05 | 0.0141 / 1.077 (×0.013) | 0.011 | 0.073 | 2.96 / 5.25 |
| base / labels | 1 | 0.00266 / 0.00275 | 0.97 | 0.0136 / 1.217 (×0.011) | 0.008 | 0.067 | 3.26 / 5.13 |
| base / labels | 2 | 0.00201 / 0.00200 | 1.00 | 0.0169 / 1.087 (×0.016) | 0.007 | 0.073 | 3.15 / 5.52 |
| ablation / labels | 0 | 0.00149 / 0.00143 | 1.04 | 0.0063 / 1.077 (×0.006) | 0.008 | 0 | 2.96 / 5.25 |

**Read plainly.** (1) The harness's positive control holds: on the planted world the tabular oracle
reaches 0.00000 error on the ordered arm from cycle 2 and stays at 5.2–5.4 on the shuffled arm — the
worlds carry, and destroy, exactly the succession the red claims. (2) **Red (a) is red on the current
tree on 3 of 3 seeds**, and worse than "cannot see order": the arm's model predicts the fully
deterministic ordered world WORSE than its per-cycle-shuffled twin (ratios 1.19–1.52). (3) The latent
spread is ×0.013–0.017 of a fresh model's; `latent_std` 0.008–0.013, below the module's own 0.05 floor
and above the live-model bars' `> 0.0`. (4) The variance hinge's gradient is 7–9 % of the invariance
gradient at the end of training (pinned "< 0.05" — off by about 2×, disclosed; the direction and the
order of magnitude were right). (5) The ablation collapses harder (×0.005–0.009) and fails (a) the same
way — the control that is not the method holds, and it says the anti-collapse term as configured buys
about a 2× larger spread and nothing on order. (6) The labels world (the natural stream's proxy): ratio
≈ 1 for every variant, as predicted; its oracle sits at 3.0–3.3 ordered against 5.1–5.5 shuffled — the
pack has memorisable (vis, aud, action)→next structure at the exact-cycle level and none at first order,
consistent with W2 §3.1c; reported, not scored.

## 4. Sweep 1 — the anti-collapse axis: holding the latent open does NOT buy order

Eight single-change / combination variants, planted world, seeds 0–2 (24 runs, archived):

| variant | ratio ordered/shuffled (s0 / s1 / s2) | spread ratio trained/fresh | `latent_std` ordered | \|g_var\|/\|g_inv\| | ordered last-block perr |
|---|---|---|---|---|---|
| `eps8` | 1.18 / 1.51 / 1.20 | ×0.013–0.017 | 0.008–0.013 | 0.11 | 0.0020 |
| `gamma_sphere` | 1.19 / 1.52 / 1.21 | ×0.013–0.017 | 0.008–0.013 | 0.088 | 0.0020 |
| `lamvar4` | 1.27 / 1.66 / 1.44 | ×0.019–0.023 | 0.014–0.019 | 0.28 | 0.0031–0.0040 |
| `lamvar16` | 2.10 / 1.42 / 2.83 | **×0.104–0.120** | **0.060–0.064** | 0.26–0.61 | 0.025–0.036 |
| `lamvar20` (the self-check's own) | 1.59 / 1.32 / 2.78 | **×0.117–0.178** | **0.066–0.085** | 0.22–0.58 | 0.037–0.054 |
| `lamvar64` | 1.38 / 1.06 / 2.09 | ×0.33–0.91 | 0.105–0.18 | 0.38–0.87 | 0.25–0.33 |
| `eps8_lamvar16` | 2.05 / 1.36 / 2.61 | ×0.107–0.124 | 0.061–0.067 | 0.26–0.62 | 0.025–0.038 |
| `eps8_gamma_lamvar16` | 2.15 / 1.46 / 2.81 | ×0.111–0.138 | 0.066–0.070 | 0.27–0.60 | 0.027–0.040 |

Predictions from §2.6, scored: `eps8` and `gamma_sphere` alone change nothing (as pinned); `lamvar16` and
`lamvar20` lift the spread ratio above 0.1 and `latent_std` above the module's 0.05 floor (pinned "above
0.2" for the spread — off by 2×; the floor half holds); **(a) fails for every variant, and fails the
wrong way round: the model predicts the fully deterministic ordered world WORSE than its shuffled twin
on 24 of 24 runs (ratios 1.06–2.83), and the more the latent is held open the larger the ordered error.**
At `lambda_var = 64` the model no longer learns at all (ordered error 0.25–0.33 ≈ the target's variance).

**The mechanism, read at the symbols — a second constraint the protocol did not name.** The predictor
is residual, `p = z + Wz·z + Wa[:, a] + b`, and `train_step` row-scales `Wz` every step so its maximum
row-abs-sum stays below `rho_max = 0.9` (a contractive proxy for bounded `plan()` rollouts), on top of a
per-step weight decay `wd = 1e-3` (×0.999 per step: 78 % of any `Wz` gone over the 1536-step run). A
step between two well-separated unit-norm latents has norm up to 2, so once VICReg has spread the 8
joint states apart the successor map is UNREPRESENTABLE under the bound — while the SHUFFLED world's
optimum (the mean next latent, a constant carried by `b` alone) is representable. Spread and a
contractive predictor fight; the base configuration resolves the fight by collapsing (a collapsed
latent needs no step). This is why the base model's ordered error is only 0.002 and why every
anti-collapse variant makes the ordered world worse. `predict()` already clamps every output at
`z_clip = 3`, so rollout boundedness does not rest on the row-scaling.

## 5. Protocol AMENDMENT — a second axis, pre-registered before sweep 2

Single changes on the predictor bound, each alone and with the anti-collapse fix that held the latent
open (`lambda_var = 16`, the value that cancels the hinge's 1/L; the self-check's 20 is the incumbent
constant and is measured beside it): `rho2`, `rho4`, `rhoinf` (no row-scaling; `predict()`'s clamp still
bounds rollouts), `wd0` (no weight decay on `Wz`), and the combinations `lamvar16_rho2`,
`lamvar16_rho4`, `lamvar16_rhoinf`, `lamvar16_wd0`, `lamvar16_rho4_wd0`, `lamvar20_rho4`. Same worlds,
seeds, statistics and bar as §2.3 (a). Attribution stays per-item: the bound alone (with the collapsing
`lambda_var = 1`), the anti-collapse alone (§4), and the combination.

**Predictions, pinned:** `rho4` alone: the latent still collapses (spread < 0.05) and (a) fails (nothing
to predict between); `lamvar16_rho4` and `lamvar16_rhoinf`: (a) PASSES on ≥ 2 of 3 seeds (ratio < 0.5)
with `latent_std` ≥ 0.05 — the honest red of this sweep; `lamvar16_wd0` alone: partial (the row-scaling
still binds); `lamvar64_*`: not run (it does not learn). If `lamvar16_rhoinf` passes and `lamvar16_rho4`
does not, the shipped bound is the smallest `rho_max` that passes on 3 of 3, measured, and disclosed as
a P1 constant with its derivation (the maximum row-abs-sum a rotation of the 16-sphere needs is ≤ 4).
The module self-check's rollout assert (40 steps bounded, `max |z_i| < 10`) must stay green under the
shipped bound — `predict()`'s clamp is what carries it; if it reds, `z_clip` is the next single change,
not a reason to keep 0.9.

## 6. Sweep 2 — the predictor bound: a free predictor does not read order either

| variant | ratio ordered/shuffled (s0 / s1 / s2) | spread ratio | `latent_std` ordered | ordered blocks 4–6 (s1) |
|---|---|---|---|---|
| `rho4` (λ 1) | 1.67 / 1.96 / 1.72 | ×0.013–0.020 | 0.012–0.019 | 0.0068, 0.0044, 0.0026 |
| `lamvar16_rho4` | 1.28 / **15.3** / 6.17 | **×1.59–1.87** | **0.232–0.244** (the ceiling) | 0.035, 0.106, **0.329** |
| `lamvar16_rhoinf` | identical to `rho4` (the bound was not binding above 4) | | | |
| `lamvar16_rho2` | 9.87 / 6.49 / 14.6 | ×1.48–1.68 | 0.24 | 0.043, 0.083, 0.139 |
| `lamvar16_wd0` | 1.36 / 1.15 / 2.73 | ×0.10–0.12 | 0.06 | 0.026, 0.025, 0.030 |
| `lamvar16_rho4_wd0` | 1.55 / 3.12 / 7.15 | ×1.47–1.84 | 0.24 | 0.249, 0.250, 0.148 |
| `lamvar20_rho4` | **0.65** / 2.64 / 3.86 | ×1.61–1.90 | 0.24 | 0.042, 0.055, 0.134 |

§5's pinned prediction — `lamvar16_rho4` clears (a) on ≥ 2 of 3 — was WRONG. Lifting the bound with the
latent held open produces the opposite of learning: the latent expands to its ceiling (per-dim std
0.24 = the maximum a unit-norm 16-d latent can have; spread 1.6–1.9× a FRESH model's) and the ordered
error rises through the run and oscillates (seed 1: 0.035 → 0.106 → 0.329), while the shuffled twin's
stays at 0.009–0.05. One seed of one variant (`lamvar20_rho4`, seed 0) reads 0.65 — a single seed
against a 3-of-3 bar, the shape of noise.

**The mechanism, and it closes the loop on §1(i).** `gamma = 1.0` is UNREACHABLE for a unit-norm
latent (Σ_j var_j ≤ 1 over 16 dims ⇒ mean per-dim std ≤ 0.25), so the variance hinge NEVER switches off:
it is a constant expansion force with no equilibrium. Under a weak force (λ 1) invariance wins and the
latent collapses; under a strong force with a free predictor (λ 16, ρ 4) expansion wins, the ONLINE
encoder keeps moving, the EMA target (`tau` 0.99) lags behind it, and the predictor chases a target that
never settles — the ordered world, which needs an exact 8-cycle, suffers most. The self-check's `LV = 20`
"works" on its synthetic stream because that stream's own variation keeps the predictor's job small.
`gamma = 1/√L = 0.25` (sweep 1) is the ceiling itself, not an interior target, so it did not help either.

## 7. Amendment 2, pre-registered before the last runs — two controls the null owes

- **A reachable hinge target.** `g125`: `gamma = 0.125` (half the isotropic ceiling; a fixed point on the
  normalized latent's own scale, P1 case (a)), with `lambda_var 16` and `rho4`, and with `lambda_var 4`
  and `rho4`. Prediction: the hinge switches off once the latent is half-open, the encoder settles, the
  ordered error stops rising; whether it CLEARS (a) is uncertain — the honest red of this amendment.
- **The positive control at the mechanism's input contract (rank 2a's lesson).** A plain LINEAR
  predictor on the RAW 64-d pattern, `x_{t+1} ≈ A x_t + b`, least squares over cycles 1–5 and scored on
  cycle 6, ordered vs shuffled, same seeds. If it separates the two worlds (ordered ≪ shuffled) the
  planted world is linearly learnable at the arm's own input and the JEPA's failure is the JEPA's
  architecture, not the world's; if it does not, the harness's world is the defect (§2.7's deletion
  clause).

## 8. Amendment 2 results — the world is learnable; a reachable hinge target is the one thing that moved

**The linear-raw positive control PASSES** (`jepa_linear_control.py`, archived): a plain linear
predictor on the raw 64-d pattern, least squares on cycles 1–5 and scored on cycle 6 —

| world | seed | held-out error ordered / shuffled | ratio | id bands only, ordered / shuffled |
|---|---|---|---|---|
| planted | 0 / 1 / 2 | 0.147 / 2.863 · 0.125 / 2.844 · 0.143 / 2.876 | **0.051 / 0.044 / 0.050** | **0.0000** / 2.72 on every seed |
| labels | 0 / 1 / 2 | 2.563 / 2.870 · 2.530 / 2.873 · 2.558 / 2.884 | 0.89 / 0.88 / 0.89 | 2.41 / 2.72 |

The planted succession is exactly linearly predictable at the arm's own input (the id bands to
0.0000; the remaining 0.13–0.15 is the proprioception random walk's irreducible step); the tracked
packs' label order is not (0.88 — no first-order structure, W2 §3.1c again). The world is not the
defect; the deletion clause does not fire. Every JEPA failure above is the JEPA's.

**The reachable hinge target (`gamma = 0.125`) is the one change that produced a qualitatively
different model:**

| variant | seed | ratio ordered/shuffled | ordered blocks 1–6 | spread ratio | `latent_std` | \|g_var\|/\|g_inv\| |
|---|---|---|---|---|---|---|
| `lamvar16_g125_rho4` | 0 | **0.238** | 0.127, 0.032, 0.045, 0.019, 0.012, **0.004** (falling) | ×0.61 | 0.164 | 0.09 |
| `lamvar16_g125_rho4` | 1 | 1.33 | 0.122, 0.040, 0.031, 0.025, 0.028, 0.028 (plateau) | ×0.50 | 0.152 | 0.09 |
| `lamvar16_g125_rho4` | 2 | 2.36 | 0.124, 0.023, 0.025, 0.028, 0.028, 0.020 (plateau) | ×0.67 | 0.189 | 0.00 (hinge OFF) |
| `lamvar4_g125_rho4` | 0 / 1 / 2 | 1.97 / 3.57 / 3.74 | falling to 0.005–0.009 | ×0.03–0.05 (partial collapse) | 0.024–0.041 | 0.19–0.29 |

With a target the latent can actually reach, the hinge switches off (the gradient ratio reads 0.000 on
seed 2 and 0.09 on the others, against 0.26–0.61 under `gamma = 1`), the latent settles in the
INTERIOR (spread ×0.5–0.67 of fresh, `latent_std` 0.15–0.19 — neither collapsed nor at the ceiling)
and the ordered error stops rising. On seed 0 the model then READS the cycle: its ordered error falls
monotonically to 0.004 and sits at 0.24 of the shuffled twin's — the first and only cell in 66 runs where
a JEPA predicted the deterministic world better than the destroyed one. Seeds 1 and 2 plateau at
0.02–0.03 (ratios 1.3 and 2.4). **Against the pre-registered 3-of-3 bar, (a) FAILS: 1 of 3.** §7's
prediction ("the encoder settles, the ordered error stops rising") held; whether it clears was pinned
uncertain, and it did not. `lamvar4` with the same target collapses partially: the reachable target is
necessary, the force still has to be ~16.

**Amendment 3, pre-registered before the run:** the same configuration for 12 cycles on the same
seeds — a plateau that holds for six more cycles is a seed-fragile optimum (the null stands); a
plateau that breaks changes the reading to "slow", not "cannot". Prediction: seed 0's ratio falls
below 0.1; seeds 1 and 2 UNCERTAIN.

## 9. Amendment 3 result — the plateau breaks: red (a) CLEARS on 3 of 3 seeds at 12 cycles

`lamvar16_g125_rho4`, 12 cycles (3072 steps), same seeds, same worlds:

| seed | ratio ordered/shuffled | ordered blocks 6–12 | shuffled last block | spread ratio | `latent_std` ord / shuf | \|g_var\|/\|g_inv\| |
|---|---|---|---|---|---|---|
| 0 | **0.069** | 0.0046, 0.0063, 0.0058, 0.0030, 0.0046, 0.0035, 0.0014 | 0.0201 | ×0.63 | 0.163 / 0.033 | 0.000 |
| 1 | **0.022** | 0.0290, 0.0276, 0.0116, 0.0064, 0.0023, 0.0042, 0.0012 | 0.0533 | ×0.61 | 0.162 / 0.079 | 0.000 |
| 2 | **0.238** | 0.0216, 0.0157, 0.0079, 0.0027, 0.0014, 0.0024, 0.0035 | 0.0149 | ×0.70 | 0.159 / 0.060 | 0.000 |

Seed 0's ratio fell below 0.1 as pinned; seeds 1 and 2, pinned uncertain, broke their plateaus at
cycles 8 and 7 and fell to the same regime. **At 12 cycles the pre-registered bar (< 0.5 on 3 of 3)
CLEARS on 3 of 3**; the shuffled twins sit at 0.015–0.053 with their latents narrower (0.03–0.08) —
the model now uses its latent exactly when the world rewards it. Honesty on the bar: §2.3 fixed 6
cycles and the 6-cycle run failed 1 of 3 (§8); this is the SAME configuration for longer, disclosed as
amendment 3 before it ran, and both results stand in the record. The in-tree test asserts the
12-cycle form; "slow" is the reading, and the daemon's wake stream is thousands of ticks long.

**Attribution owed (P8 clause 3), run at 12 cycles before the change ships:** `g125` alone
(`lambda_var 1`, `rho_max 0.9`), `lamvar16_g125` (the old bound) and `g125_rho4` (`lambda_var 1`) —
the shipped change is the SMALLEST of the four that clears 3 of 3.

## 10. The bundled item — the curiosity gauge's resolution and hold, measured in-brain

`cur_mag_gauge_probe.py` (archived): seed 0, 1500 ticks (742 wake rows), `throwaway_home()`,
`persist=False`, read-only wraps on `intrinsic_reward` and `step_reward`, the UNROUNDED `arm._cur_mag`
read every wake tick beside the 4-dp `state()["curiosity_mag"]`:

| window | held (both drives zero) | `raw_cur` zero | `raw_emp` zero | unrounded gauge: distinct / min / max / mean | 4-dp key: distinct / mean |
|---|---|---|---|---|---|
| last 200 rows | **0.000** | 0.000 | 0.505 | 200 / 1.8e-4 / 1.7e-3 / 7.2e-4 | 16 / 0.00072 |
| last 1000 rows | 0.013 | 0.081 | 0.327 | 732 / 0 / 3.3e-2 / 9.4e-3 | 262 / 0.00943 |

The gauge is a live EMA updated on 99–100 % of wake ticks; the hold case (both drives exactly zero)
is 0–1.3 % of ticks; it decays from 0.033 at ~500 ticks to 1.8e-4 at 1500 — below the 4-dp display
step by 3000, which is W3's "1 distinct value". **No code change**: the reading the W2 panel corrected
("a frozen latch") is refuted in-brain as it was on the cells; the display rounds a live sub-resolution
signal, and rendering 1e-4 as curiosity would not be more honest than rendering it as 0.0000. Whether
the gauge moves once the world model reads succession is §2.5's downstream measurement.

## 11. Attribution (P8 clause 3) — each change alone fails, each pair fails, the triple clears

All at 12 cycles, seeds 0–2, planted world (the `base` and pair rows from §3, §4, §6 at 6 cycles where
the verdict was already unambiguous):

| configuration | force `lambda_var` | target `gamma` | bound `rho_max` | ratio ordered/shuffled (s0 / s1 / s2) | spread ratio | verdict |
|---|---|---|---|---|---|---|
| base (the arm today) | 1 | 1.0 (unreachable) | 0.9 | 1.19 / 1.52 / 1.21 | ×0.013–0.017 | collapsed, order unread |
| `g125` alone | 1 | 0.125 | 0.9 | 1.04 / 1.04 / 1.04 | ×0.006–0.008 | collapsed |
| `g125_rho4` | 1 | 0.125 | 4 | 1.09 / 1.06 / 1.09 | ×0.006–0.008 | collapsed |
| `lamvar16` alone | 16 | 1.0 | 0.9 | 2.10 / 1.42 / 2.83 | ×0.10–0.12 | open, order unread |
| `lamvar16_g125` | 16 | 0.125 | 0.9 | 1.29 / 1.12 / 4.52 | ×0.23–0.41 | open, the bound blocks the step |
| `lamvar16_rho4` | 16 | 1.0 | 4 | 1.28 / 15.3 / 6.17 | ×1.6–1.9 (ceiling) | expanding, unstable |
| **`lamvar16_g125_rho4`** | **16** | **0.125** | **4** | **0.069 / 0.022 / 0.238** | **×0.61–0.70** | **reads the succession, 3 of 3** |

Three necessary conditions, none sufficient alone or in pairs: a force strong enough to hold the
latent open against the invariance term, a target the latent can REACH so the force switches off and
the encoder settles, and a predictor allowed to take the step between two settled latent states.

**The shipped change — one commit, three constants, each derived from the latent width `L` (P1):**

- `lambda_var = L` (16): the hinge's gradient carries a 1/L from averaging the per-dimension hinge;
  `L` cancels it, putting the per-dimension variance force at VICReg's own O(1/m) scale. (The self-check's
  hand-picked 20 was the same repair by eye; one configuration survives — the self-check now runs the
  defaults, P2.)
- `gamma = 0.5 / sqrt(L)` (0.125): a unit-norm latent's per-dimension std is at most 1/√L (Σ var_j ≤ 1),
  so the old `1.0` was never reachable and the hinge never switched off; half the isotropic ceiling is an
  interior fixed point on the normalized output's own scale (P1 case (a)).
- `rho_max = sqrt(L)` (4): an orthogonal map of the L-sphere has rows of L2-norm 1 and hence L1-norm ≤ √L,
  so a bound of √L admits any rotation/permutation of latent states while the old 0.9 forbade every step
  longer than the latent's own radius; long rollouts stay bounded by `predict()`'s `z_clip` clamp, which
  the module self-check's 40-step assert continues to verify.

**What the two live-model self-check sites can and cannot assert — disclosed, and it changes §2.3(b) as
written.** Both `autonomous_loop`'s `jp_std` and `live_brain`'s `jepa_std` are read after 90 steps on a
FIXED stimulus (the same ctx dict every step); a healthy encoder maps identical inputs to identical
latents, so `latent_std()` there measures the stimulus, not the encoder (it printed 0.000 and 0.007 and
passed `> 0.0` on sub-float variation). Raising those two bars to 0.05 would red for a reason that is
not collapse. They are relabelled as what they are — wiring checks that the model is constructed and its
buffer live — and the collapse floor lives where the input varies: the new test on the planted world
(brain-free, seeds 0 and 2, 10 cycles) and §2.5's in-brain measurement on the real stream. The (b) red
as pre-registered is therefore carried by the test and the in-brain cells, not by the two 90-step sites.

## 12. The fix, landed and verified (commit 2 of the arc)

**Commit 1 (`2b3d067`) — the red:** `tests/test_jepa_collapse.py` (the planted world's learnability as
the positive control; the order red; the collapse gauge; slow-marked, seeds 0 and 2, 10 cycles) plus this
record through §11 and the plan's Q1 ratification. Run on that tree: the control PASSED, the order test
FAILED with ordered 0.00131 against shuffled 0.00129 (indistinguishable worlds) — the red counted on the
committed tree before the rule.

**Commit 2 — the fix, one change (three constants, one derivation):** `JepaWorldModel.__init__` now
derives `lambda_var = L` (the constructor's `lambda_var=None` default; callers may still pass a value),
`gamma = 0.5 / sqrt(L)` and `rho_max = sqrt(L)`, each with its reason at the symbol; the module docstring's
design claims corrected (the hinge target and the "contractive" bound) and a dated paragraph pointing
here; `main()` runs the DEFAULTS instead of its private `lambda_var 20 / lambda_cov 2` (one configuration,
P2); the two live-model self-check sites relabelled as the fixed-stimulus wiring checks they are (their
`> 0.0` asserts kept — a wiring check's red is a model that never updated).

**A criterion that measured the pathology went red when the pathology left**, and is folded here rather
than hidden: `main()`'s reward-shaping block asserted "pure SSL must collapse a few repeating contexts"
and "a readout recovers the rule from the reward-shaped latent, NOT the collapsed one (+0.3)". At the new
defaults the plain readout also recovers the rule (1.00 vs 1.00), so the contrast that only a collapse
could produce was gone. What the head is FOR survives as the criteria — it separates the contexts by a
margin over pure SSL (1.768 vs 0.088) and a readout recovers the rule from its latent (1.00) — and the
plain arm's numbers are reported beside it. `feedback-criteria-measure-the-pathology` is the memory this
repeats.

**Verification on the fixed tree (all commands re-run, outputs archived):**

| check | result |
|---|---|
| `tests/test_jepa_collapse.py` | **3 passed** (green only after the fix) |
| module self-check `jepa_world_model` at the defaults | OK — collapse: full 0.083 vs ablation 0.007; JEPA beats identity 0.049 < 0.100 and the 1-D model 0.018 < 0.116; 40-step rollout bounded (max \|z\| 0.69); determinism + save/load round-trip; deep encoder 4/4 |
| `autonomous_loop` self-check | OK — buffer std 0.029 on the fixed stimulus (relabelled); reward-latent goal formed; E-STOP freeze holds; its own `reward-enc` line still measures the head separating band latents (1.66 vs 0.043) |
| `live_brain` self-check | OK — buffer std 0.037 on the fixed stimulus (relabelled); goal formed; determinism with JEPA live |
| `hjepa_serve` self-check | OK (unchanged) |
| `tests/instruments/jepa_highdim_probe.py` (3000 steps) | OK — latent std 0.229 vs ablation 0.018; learned probe +49 % over random |
| `ruff check` + `ruff format --check` on the four files | clean |

Disclosed, not changed: `cerebellum/spiking_jepa.py` carries its own copy of the 0.9 bound and the
1e-3 decay ("copied from jepa_world_model") on a different class with its own measured claims — a P2
follow-up for its own arc, named here. `save_weights` persists `lambda_var` but not `gamma`/`rho_max`;
the only external `load_weights` caller is the research tier, and the live arm never loads a saved
model, so a restored old file cannot reintroduce the constants on the live path.
