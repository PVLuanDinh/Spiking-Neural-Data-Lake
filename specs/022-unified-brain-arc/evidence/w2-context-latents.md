# W2: the real latents reach the arm context — the premise probe, pre-registered (2026-09-05)

*The plan of record, Phase 1: "**W2 — the real latents reach the context without breaking the width.**
`_safe_ctx` pools the real vision/audio latents to one scalar per band. W2 delivers them as their
prototype ids (the categories the learners already produce) and the encoder's surprise through the
declared afferent lanes (FR-008) WITHOUT widening the 64-wide arm context — a width change would orphan
the daemon's 64-wide prototypes and the 31-unit growth blob (rank 2b refuses a record of another width;
`NGSNN` refuses a widened percept). If a widening is ever chosen it is a SYNTHESIS §3.5 migration event
(FR-030) with its own P4b restart red and a fresh-engine disposition, stated in advance."*

W1 was retired on 2026-09-04 (owner decision, P8 deletion-on-a-null; `evidence/w1-goal-curriculum.md`),
so W2 runs ALONE and the plan's "W1+W2" reds are scored on W2 by itself. The reds, from Part F (its rows
say "under W1+W2"; read here as W2 alone, paraphrased):
**W-R1** — mean `curiosity_mag` over the last 200 wake rows at 3000 and at 6000 ticks does not exceed
the time-shuffled-context world, per seed (control: the same world, context order shuffled). **W-R2**
(amended 2026-09-03 on W3's measurement) — category entropy not above the W3 baseline at the NAMED
window (the FIRST 1000 wake rows), PER SEED, or live next-category accuracy not above its
marginal-preserving shuffle (floors reported within-arm). **On a null:** W2 reverts; the finding is that
the curiosity chain's floor is intrinsic to the world model over this body, and Phase 2's L2 is
re-targeted at prediction error.

## Why this file opens with a probe and not a build

Part C of the plan: **"The world before the wire. No growth or learning task starts until its stimulus
has been shown, on the exact signal the task names, to carry structure its red can detect (rank 2c's
in-process probe is the template: probe, then build)."** Rank 2c itself, refusing the rotating stream,
wrote W2's premise before W2 existed: *"the pooled real channels enter that context as scalars a world
model masters in a few hundred ticks whatever frames feed them"* and *"a richer CONTEXT (not richer
frames) is the pre-registration worth writing: its red is Chain A's `curiosity_mag` leaving 1 distinct
value at 3000 ticks under the same seeds."*

So this file first measures what the arm's context carries TODAY (§1), states the premise as something
that can fail (§2), pre-registers the protocol (§3), runs the in-process premise probe with no tree
change (§4), and only then builds — or refuses on measurement, as rank 2c did.

## 1. What the arm's context carries today — measured at HEAD `f546e0a`, before any code

Every number below is from in-process brains under `substrate.home.throwaway_home()` with
`make_brain(seed, persist=False)`; nothing touched the live lake. Seed 0 unless stated. Scripts are
archived with this arc's session record beside the memory store.

**The chain, read at the symbols (not from the plan's summary).** `Brain._wake_tick` builds
`ctx = self._safe_ctx()`, hands it to `self.brain.tick(ctx, ...)`, and only AFTER the arm has stepped
perceives the next frames (`_perceive_real`, `_perceive_audio`) and consumes them
(`_consume_perception`). The context on tick *t* therefore carries the percept perceived at the end of
tick *t−1*. `_safe_ctx` returns `visual` and `audio` as 4-vectors from `_attend_real_stream`: the
learned reticular gate (`cortex/learned_attention`) gates the 64-d latent and the gated vector is
averaged over four 16-cell slices. `AutonomousArm._context_pattern` → `encode_multimodal` then turns
each 4-vector into its 16-cell band by AVERAGING the `PopulationEncoder(n=16, lo=0, hi=1)` Gaussian
activations of its four elements (σ = 1/15). The 64-d pattern is what the arm's `GrowingLearner`
(cosine, vigilance 0.85) categorises and what `JepaWorldModel.prediction_error` scores; that error
drives `SpikingCuriosityDrive` (rectified learning progress) → `_cur_mag` → `state()["curiosity_mag"]`
(Chain A).

**1a. The prototype ids exist and are thrown away.** `Brain._consume_perception` runs
`_, grew = learner.observe(percept)` on the vision learner and on the audio learner every wake tick:
the prototype id is computed and DISCARDED (P3's shape 2, a return value discarded); only `grew`
survives, to lift `real_novelty`. Nothing downstream ever sees which category the brain just perceived.

**1b. The id stream is rich.** Both packs are 256 frames and their pointers advance in lockstep
(`_real_ptr == _audio_ptr ==` the wake-tick count), so the joint stimulus has period 256. After 294
wake rows (600 ticks) the vision learner holds **9** prototypes and the audio learner **15**; over 393
wake rows (800 ticks) the (vision, audio) id pair takes **102 distinct values**. Vision-id marginal over
one cycle (seed 0): `{0:33, 1:33, 2:22, 3:26, 4:59, 5:17, 6:36, 7:28, 8:2}`; audio: 15 ids, the
largest 36 of 256. The encoder's surprise (`_surprise_signal`, the Oja/Sanger residual) is alive: 72
distinct values at 4 dp over 294 rows, mean 0.082, range [0.0000, 1.0000].

**1c. The band the arm sees destroys that structure.** Over 393 wake rows:

| band (16 cells) | cells that ever leave 0 | consecutive-row cosine, mean / min | per-cell std, max |
|---|---|---|---|
| encoded visual | 0–4 carry the bump; 5–6 read < 0.003; 7–15 identically 0.0 at 4 dp | **0.9938 / 0.9695** | 0.066 |
| encoded audio | 0–3 carry the bump; 4–5 read < 0.003; 6–15 identically 0.0 | **0.9969 / 0.9733** | 0.045 |
| encoded temperature (proprioception) | 0–13 | 0.9575 / 0.0000 | 0.401 |

The raw 4-vectors are small and nearly constant — visual means `[0.074, 0.098, 0.035, 0.035]` with
stds ≤ 0.035, audio means `[0.020, 0.021, 0.032, 0.040]` — so every frame lands as the same Gaussian
bump against the band's low edge. The test that matters: **the visual band's consecutive-row cosine is
0.9933 when the vision prototype id CHANGED between the rows (n = 340) and 0.9969 when it stayed the
same (n = 52).** A code whose distance between different categories equals its distance within one
category carries no category. The only band that moves is the body's own joint fold. This is rank 2c's
sentence with a number on it: a world model that has learned the proprioception band has learned the
context, and Chain A goes to its floor.

**1d. The floor, as W3 recorded it** (`evidence/w3-world-baselines.md`, the pre-registered "before"):
`curiosity_mag` at 4 dp is AT FLOOR (1 distinct value, mean 0.0000) on seeds 0 and 2 at both 3000 and
6000 ticks; seed 1 is alive at 3000 (127 distinct, mean 0.02001) and collapsing at 6000 (5 distinct,
mean 0.00026). Category entropy over the first 1000 wake rows: seed 0 **1.8801** bits / 7 categories,
seed 1 **1.0369** / 4, seed 2 **2.0415** / 6 — the W-R2 baselines, per seed, window pinned.

## 2. The premise, stated as something that can fail

**Premise.** If the visual and audio bands carry the prototype ids (a categorical code that separates
categories) and the encoder's surprise instead of the near-constant pooled bump, the arm's context
acquires the structure §1b shows exists upstream, and (i) the arm's learner forms more categories over
the first 1000 wake rows than the W3 baseline, (ii) the world model has something to learn, so Chain A
leaves its floor while it learns.

**How it fails, and each failure is a recordable finding.** (a) The world model masters the id
sequence as fast as it mastered the bump — the pack is periodic with period 256, and a first-order
predictor over ~9×15 id pairs plateaus within a few cycles; Chain A's floor is then intrinsic to an LP
drive over ANY stationary world, exactly the plan's null disposition. (b) The categorical code recruits
more arm units (entropy rises) while the world model learns nothing — W-R2's first half passes and
W-R1 fails; that is a finding about the learner, not the world, and W-R2's second half (next-category
accuracy above its shuffle) is what separates "more units" from "more structure". (c) The change moves
`curiosity_mag` through a collateral wire — the generalizing-novelty pseudo-count reads the ctx dict —
rather than through the world model; the protocol's control and attribution arms exist for this.

## 3. Protocol — PRE-REGISTERED before the premise probe runs

Written before the 9-cell probe of §3.4 is launched. Symbols, not line numbers. Runtime evidence outranks
this text: a run that contradicts a claim below is the finding, and the text is corrected in §4.

The protocol was drafted by an eleven-agent design panel (three independent designs, three judges, a
synthesis, a pre-mortem, two red-team lenses, an integrator; archived beside this arc's session record)
and then cut to what the plan's rules actually need. Every finding the panel raised that changed a rule
is named where the rule is stated. One finding reshapes the reds and is stated first.

### 3.0 The reading of the plan sentence, and two deviations disclosed

(i) "their prototype ids" = the ids `Brain._vis_learner` / `_aud_learner` already return from
`observe()` in `_consume_perception` and discard (§1a). (ii) "through the declared afferent lanes" =
every delivered id's frame entered through `afferent_boundary.vision_out → brain.vision_in` /
`audition_out → brain.audition_in`; the surprise already rides `percept_surprise.surprise_out →
brain.salience_surprise_in` and the T046a lane into `LearningProgress.consume_surprise`. (iii) "without
widening" = the ctx dict keeps its `(4, 1, 4, 1)` shape and `encode_multimodal`'s fixed 4×16 tiling;
`MultiModalEncoder.add_modality` is never called.

**Deviation 1 — surprise is not placed inside the 16-cell band.** Under the unchanged `_encode_band`
average, a second element in `ctx["visual"]` halves the id bump and adds a bump wandering mostly in
[0, 0.2] (§1b: mean 0.082), the positions ids 0, 8, 4 and 12 occupy — a category/surprise confound in
the very band W-R2 scores; and the surprise already has two declared consumers, so a third is a P2
question. If W2 proceeds, an arm `w2+surprise` (the design panel's arm E; §3.5) measures it and the slot
ships only if it wins. **The surprise half entered no probe cell** (every cell is `--code vdc`, ids
only); it is the subject of Q2 (§5).
**Deviation 2 — the encoding is a POSITION, not a sparse code.** The id enters its band as one scalar,
`symbol_position(id)` = the base-2 van der Corput value (0 → 0, 1 → .5, 2 → .25, 3 → .75, 4 → .125 …),
so `ctx["visual"] = [p] * 4` and the unchanged encoder produces ONE unit-amplitude Gaussian bump — the
same energy as the proprioception bump — with consecutive ids never closer than 0.25 (3.75σ) and no
metric following recruitment order. Every other reader of the ctx dict (the generalizing-novelty hash,
`_c1_tick`'s 9-d affect feature, the predictive-coding front end) keeps its shape. Capacity: ~8–10
resolvable positions per band; ids beyond alias to a neighbour — the vision learner holds 9, the audio
learner 15 (§1b), so audio aliases. The alternative — splicing a 16-cell sparse code into the band on
a side channel the arm reads inside `_context_pattern` — was built into the probe as a comparison arm
(`--code sparse`) and is NOT the build candidate: it changes the meaning of 32 of 64 cells the daemon's
restored prototypes were learned on, and its only advantage (capacity) is exactly what FR-030 reserves
for a migration event.

### 3.1 Stage −1 and A0 — the world before the wire, on the exact signal (plan Part C)

**Stage −1, brain-free, on the exact classes** (the panel's control red-team, replicated independently
by its integrator; two scratch scripts, archived): `JepaWorldModel(ACTIONS, in_dim=64, latent_dim=16,
seed)` at the arm's own defaults (lr 0.05, buffer 32, γ 1.0, anti_collapse True), fed the build's exact
context shape `[vdc(vis)]*4 + [prop] + [vdc(aud)]*4 + [1.0]` with the tracked packs' labels as proxy ids.
Three results, each a reason on its own: (a) the target-latent spread over 64 percept patterns falls
0.728 → 0.0009 in 1500 steps (`latent_std` 0.006) — the latent COLLAPSES; (b) a PLANTED world with a
deterministic 3-bit successor map (next-id mutual information 3 bits) is "mastered" as fast under a
per-cycle permutation as in order — the prediction error cannot see order even when order is all there
is; (c) the packs' own label order carries no first-order structure: cyclic next-label MI 0.250 (vision)
/ 1.085 (audio) sits inside the distribution of 200 whole-sequence shuffles (0.259 / 1.096). **(c)
replicated by this session on the tracked packs** (`models/nmnist_pack.npz`, `models/shd_pack.npz`, 256
frames each, 10 and 20 classes): ordered MI 0.2498 vs shuffle mean 0.2579 ± 0.042 with **122 of 200
shuffles at or above the ordered value**; audio 1.0851 vs 1.1015 ± 0.048, 119 of 200 above; adjacent
frames share a label 7.4 % / 5.9 % of the time. The ordered pack is the median shuffle. **The
mechanism, read at the symbol:** in `JepaWorldModel.train_step` the VICReg variance hinge pushes each
latent dimension away from the batch mean by `lambda_var · −(z_j − μ_j) / (L · m · sd)`, a restoring
force proportional to the latent's own deviation over a floored std — so once the deviation is small the
force that should re-inflate the latent is itself small, and the invariance term (`2·r`) wins. **Four
readers of `latent_std()` exist, and the sharper finding is which of them sees the live model.** The
module's own `__main__` self-check asserts an ABSOLUTE floor — `full_std > 0.05 and full_std > 3.0 *
ctrl_std` — and `tests/instruments/jepa_highdim_probe.py` asserts `std_full > std_abl * 2`; both run on
synthetic streams and never on the arm's context. The only bars on the LIVE model are `autonomous_loop`'s
self-check `jp_std > 0.0` and `live_brain`'s `jepa_std > 0.0`, which a std of 0.006 clears. Every
in-brain value in this record (0.005–0.034, §4.2) would FAIL the module's own 0.05 floor; nothing applies
that floor where it matters. (Corrected by the arc's panel, PH-2: an earlier draft said "no consumer
outside two self-checks".) **These are owed as a record regardless of W2's outcome** (§3.7).

**A0, in-brain, no tree change** (this session; seed 0, 1500 ticks = 742 wake rows ≈ 2.9 cycles; instance
wraps on `jepa.prediction_error`, `curiosity.intrinsic_reward`, `empower.step_reward`,
`arm._context_pattern`; the trained model's target-latent spread over the 256 distinct 64-d patterns it
saw, beside a FRESH model's spread over the same patterns):

| arm (seed 0) | spread trained / fresh @1500 | `latent_std` | perr per 256-row block | `curiosity_mag` last-200 mean @1500 | arm units @742 rows |
|---|---|---|---|---|---|
| incumbent | 0.040 / 0.264 | 0.034 | 0.0165, 0.0024, 0.0017 | 0.00072 (16 distinct) | 5 |
| w2 (vdc) | 0.0136 / 0.769 | 0.009 | 0.130, 0.0087, 0.0052 | 0.0588 | 55 |
| w2 (sparse, comparison) | 0.0063 / 0.916 | 0.010 | 0.166, 0.0122, 0.0064 | 0.0209 | 93 |
| w2 (sparse) per-cycle-shuffled | 0.0062 / 0.912 | 0.012 | 0.166, 0.0128, 0.0064 | 0.0546 | 90 |

Read plainly: the richer the context, the harder the latent collapses (ratio 0.15 incumbent, 0.018 vdc,
0.007 sparse); the ordered and the per-cycle-shuffled contexts give the SAME error trajectory; curiosity
under W2 leaves the incumbent's floor by 30–80× — and the shuffled control leaves it equally or more.
What moves `curiosity_mag` under W2 at 1500 ticks coincides with the recruitment of 55–93 new arm units
against 5 (§4.4 says how far that reading holds), not with anything the world model learned about
succession. **The design panel's A0 was a two-branch rule and, applied as written, it did NOT confirm
collapse** (the arc's panel, PH-6): "CONFIRMED iff spread < 0.01 AND `raw_cur == 0` on > 95 % of wake
ticks after 300" — the spread half held for the sparse arms (0.0063) and not for the build candidate
(0.0136, still falling), and the zero-fraction half read 0.58–0.78. Its REFUTED branch prescribed a G3
cell — the INCUMBENT context ordered against the incumbent frame-shuffled — which the nine-cell probe
did not run; the probe's G-ORDER measured succession on the W2 context only, and its w2 seed-0 cell at
1500 ticks is number-for-number A0's vdc seed-0 cell (the same seed, code and head, re-run). The fold
ran G3 (§4.3b): the incumbent context reads no succession either. The operational fact that decides
W-R1's reachability is the identical error trajectory on both contexts, which needs no threshold.

### 3.2 The premise, restated after A0, as the probe's decision rule

W-R1 asks whether the arm's curiosity is higher under W2 than under the same world with the context's
temporal structure destroyed. That contrast can only be positive if the world model reads succession. A0
says it does not (identical error under order and permutation) and stage −1 says why (collapse) and that
the pack has no first-order succession to read anyway. **Pre-registered prediction for the probe:
W-R1's premise FAILS** — `curiosity_mag(w2)` does not exceed `curiosity_mag(w2-shuffled)` at 3000 ticks on
3 of 3 seeds, and the last-block prediction error is equal to 3 dp between the two arms on every seed.
If that prediction is WRONG on any seed, the brain-free proxy diverged from the brain and W-R1 stands as
a red for an S = 8 sweep at 3000 and 6000 (§3.5).

W-R2 asks whether the arm's world has category structure the arm learner and the grower can use.
**Pre-registered predictions:** (a) first-1000-row entropy under W2 exceeds the W3 baseline on 3 of 3
seeds by recruitment alone (A0: 5.4–6.0 bits / 55–93 units against 1.7 / 5) — a pass that says "more
units", which is why (a) is only the P3 charge twin and never the verdict; (b) the grower's held-out
next-category accuracy under W2 does NOT beat its persistence floor on ≥ 2 of 3 seeds, and does not beat
the shuffled arm's — an alphabet of 55–93 categories on a 1000-row buffer is more classes than rows per
class; W-R2(b) is the honest red and it is expected to null.

### 3.3 Arms, seeds, rows

Every cell: `with throwaway_home(): b = make_brain(seed, persist=False)`; one arm per subprocess, 4
workers; live daemon left running (it moves wall-clock only); live lake never touched; JSON outside the
repo. Monkeypatches on the INSTANCE only, no tree change (rank 2c's idiom):

- **incumbent** — the untouched brain (the instrument control; must reproduce W3 at summary level on
  seeds 0–2, with any drift explained by the src commits since `43d06f7e`).
- **w2** — `--code vdc`: `b._safe_ctx` wrapped to deliver `[vdc(vis_id)] * 4` / `[vdc(aud_id)] * 4`
  where an id exists (the id = `learner.classify(percept)` after `_consume_perception`'s `observe()`
  refined it, which equals `observe()`'s own return), the incumbent bands where none does (the first
  tick). `_attend_real_stream` left running so novelty/affect see what they see today. The splice count
  must equal the wake rows minus the no-percept ticks.
- **w2-shuffled** — the same wrapper, and at every cycle boundary (`_real_ptr % 256 == 0`, once per
  cycle) BOTH packs re-permuted under ONE fresh `random.Random(seed·100003 + cycle)` permutation, so the
  vision frame i stays paired with audio frame i (the joint pair marginal kept per cycle) and only
  succession is destroyed. Fresh per cycle because a FIXED permutation of a 256-periodic stream is another
  256-periodic stream (vacuous). Disclosed: this is the FRAME shuffle — it permutes what the learners,
  readouts and `PerceptSurprise` experience too, and could carry a difference through
  `_route_percept_to_action`'s class-keyed `_ext_drive`, an action route that never touches the context.
  It is the right control for a NO-TREE-CHANGE probe (a delivery-level shuffle needs a pre-filled latent
  cache and a delivery-time `classify`, both instrument code); if W2 proceeds to an S = 8 sweep the
  control is the delivery-level shuffle the panel specified, with this frame shuffle as its
  difference-in-differences fallback. Under A0's identical error trajectories the distinction cannot
  change the verdict; that is why the cheaper control is enough to decide the premise.

Seeds 0, 1, 2 (W3's, so the direction of each is comparable). 3000 ticks, marks at 1500 and 3000. Rows
per wake tick: `world_red_probe.wake_row` (+ arm units, `_novelty_live`, `real_perception_units`);
per call: `prediction_error`, `intrinsic_reward`, `step_reward`; per mark: `summarise` (last-200
liveness at 4 dp, first/last-1000 entropy with the overlap count), `latent_std`, the trained/fresh
spread, perr per 256-row block, `raw_cur`/`raw_emp` zero fractions after wake row 300, and the grower's
`predict_accuracy`, refit count, last-third mean accuracy beside its majority and persistence floors.
Cost: 9 cells × ~4 min (3000 ticks at the measured 0.08 s/tick with four brains contending) ≈ 12 min
wall at 4 workers (an earlier draft said 5; the panel's PH-8).
Three plumbing smoke runs (300 / 700 / 300 ticks, seed 0, the sparse code) and the four A0 cells were
run BEFORE this section was written; their numbers are in §3.1 and are not scored; the probe below is
scored against the predictions above and nothing in it was tuned to them.

### 3.4 Decision rule (in force before the first cell)

- **G5 instrument:** incumbent seeds 0–2 reproduce W3 at 3000 ticks at summary level (`curiosity_mag`
  distinct@4dp / mean; `entropy_first` 1.8801 / 1.0369 / 2.0415). A miss is explained by `git log
  43d06f7e..f546e0a -- src` or it halts the probe.
- **G-ORDER (W-R1's premise):** the world model reads succession iff last-block perr(w2) <
  perr(w2-shuffled) by more than 3 dp on ≥ 2 of 3 seeds. Predicted: fails.
- **G1 (W-R1 at probe power, FP 1/8):** mean `curiosity_mag` (last 200 wake rows) w2 > w2-shuffled at
  3000 on 3 of 3 seeds. Predicted: fails.
- **G-ID (the P3 charge twin, W-R2a):** H_first1000(w2) > H_first1000(incumbent) on ≥ 2 of 3.
  Predicted: passes by recruitment.
- **G-W-R2b:** last-third mean held-out accuracy under w2 > its own persistence floor on ≥ 2 of 3 seeds
  AND > the shuffled arm's on ≥ 2 of 3. Predicted: fails.
- **PROCEED TO BUILD iff G5 AND G-ORDER AND G1 AND G-W-R2b.** Otherwise W2 is REFUSED ON MEASUREMENT
  before a build (the rank-2c outcome): §4 records it, `tasks.md`'s W2 row and the plan's W2 row close as
  refused, and the disposition of §3.6 applies.

### 3.5 If the probe passes — the build and its reds at S = 8 (recorded so the bar exists before the data)

Encoding and anatomy as §3.0; `Brain._consume_perception` keeps `observe()`'s return; `_safe_ctx` drops
the `_attend_real_stream` pooling for the two real bands; two OUT ports on `brain` (`context_visual_out`,
`context_audio_out`) and two IN ports + two edges on `autonomous_loop` (census 36/19 → 40/21, artifact
regenerated, `tests/test_wiring_manifest.py` premise updated in the same commit); `world_red_probe` gains
the `--arm` switch and the delivery-level shuffle; the charge attribution written once through
`substrate/port_charge_probe.record_attribution`; `cortex/learned_attention` retired in its own commit
once its output has no reader. Reds: W-R1 at 3000 AND 6000, W-R2(a)/(b), each per seed over S = 8
against the delivery-level shuffle, statistic = paired one-sided t > 1.895 (7 d.f.) **AND the exact
one-sided sign-flip p over all 2⁸ patterns ≤ 0.05** (the plan's α), ties kept as zeros — the panel's
arithmetic: with five ties `d = [x, x, x, 0, 0, 0, 0, 0]` gives t = 2.049 and exact p = 1/8, so t alone
is a 12.5 % bar; no verdict is possible with fewer than 5 non-tied seeds (printed DEGENERATE-TIES).
Positive control in the same command: a planted paired effect on the OBSERVED tie pattern must clear the
bar beside a null sample that must not. P3 charge proof: `symbol_position` pinned to 0.5 (the id's
INFORMATION removed, the bump's ENERGY kept) must change the unit AND action series against the wired run,
and `w2` must beat that pinned arm on entropy at S = 8 — otherwise "the charge is the bump" and W2 is not
adopted as an id lane.

### 3.6 Disposition on a null

The plan's own words apply: W2 reverts (here: is not built); the finding is recorded. **The finding's
wording is corrected by A0:** the curiosity chain's floor is intrinsic to the world MODEL — a latent
that collapses on any context this body sees, and a pack with no first-order succession — rather than to
"the world model over this body" in the sense the plan meant (a body-limited stimulus). Consequence for
Phase 2: "L2 re-targeted at prediction error" must not target THIS prediction error until a
JEPA-collapse arc with its own red (stage −1(b) must flip; `latent_std` ≥ a pre-registered fraction of
the init spread; the hinge/invariance gradient ratio counted) is green. Two questions only the owner can
answer are raised in §5.

### 3.7 Owed regardless of W2's outcome

The JEPA collapse record (stage −1 and A0 numbers), the live-model `> 0.0` fixture finding (the
module's own 0.05 floor never sees the live context), the re-reading of s2c's "masters in a few hundred
ticks" as collapse, and the RESOLUTION of the curiosity gauge: `state()["curiosity_mag"]` is
`round(_cur_mag, 4)`, and inside `AutonomousArm.step`'s `if raw_cur or raw_emp:` guard the gauge is an
EMA (τ 0.02) that DECAYS whenever either drive is nonzero and is held only when both are zero — so "1
distinct value at 4 dp" is a gauge below 5e-5, and the instrument cannot tell held from decayed without the
unrounded series. (An earlier draft called this "a frozen latch"; the arc's panel — I-1/PH-1 — refuted it
on the record's own cells: the incumbent's `raw_cur` is nonzero on 91–98 % of ticks at 1e-6–1e-10 while
the gauge reads 0.0000.)

## 4. Result — W2 REFUSED ON MEASUREMENT, before a build (2026-09-05)

Nine cells at HEAD `f546e0a` (tree clean but for this file), 3000 ticks each, four brains at a time,
the live daemon running throughout (~0.08 s/tick under contention; ~4 min per cell); raw JSON and the
probe script archived beside the session record (`archive/w2-2026-09-05/`). Every number below is read
from those files; nothing was re-run to a friendlier value. **Provenance, stated plainly (the panel's
I-7):** the ordering "protocol, then cells" rests on this record's own text and on the design panel's
`panel_final.md` §3.11 (which pre-states a stricter prediction set the record then cut to four gates);
the evidence file has one commit carrying both, the archived files' mtimes are the copy time, and the
cell JSONs carry `wall_s` but no head or clock. The harness was amended in the fold to write the git
head and start/end timestamps into every cell (the G3 cells of §4.3b carry them); the nine original cells
do not.

### 4.1 G5 — the instrument reproduces W3 exactly, with the wraps on

| seed | `curiosity_mag` @3000 distinct@4dp, mean — probe / W3 | `_lp_mean` distinct@4dp — probe / W3 | `entropy_first` bits / categories — probe / W3 |
|---|---|---|---|
| 0 | 1, 0.0000 / 1, 0.0000 | 2 / 2 | 1.8801 / 7 / 1.8801 / 7 |
| 1 | 127, 0.0200 / 127, 0.02001 | 58 / 58 | 1.0369 / 4 / 1.0369 / 4 |
| 2 | 1, 0.0000 / 1, 0.0000 | 1 / 1 | 2.0415 / 6 / 2.0415 / 6 |

Digit-for-digit, on a head eleven `src` commits past W3's and with four read-only instance wraps
installed on every arm (`prediction_error`, `intrinsic_reward`, `step_reward`, `_context_pattern`; the W2
arms additionally REPLACE `_safe_ctx`, the treatment). The wraps are a polled == unpolled control: an
instrument that had written what it reports would have moved these. G5 PASSES at 3000 ticks; W3's
6000-tick columns were not re-run.

### 4.2 The world model's latent collapses on every arm, hardest under W2

Target-latent spread over the 256 distinct 64-d patterns each brain saw, the TRAINED model against a
FRESH model of the same seed on the same patterns, at 3000 ticks:

| seed | incumbent trained / fresh | w2 trained / fresh | w2-shuffled trained / fresh | `latent_std` inc / w2 / shuf |
|---|---|---|---|---|
| 0 | 0.0288 / 0.264 (×0.11) | 0.0088 / 0.769 (×0.011) | 0.0169 / 0.764 (×0.022) | 0.024 / 0.013 / 0.005 |
| 1 | 0.0126 / 0.271 (×0.047) | 0.0044 / 0.831 (×0.005) | 0.0046 / 0.828 (×0.006) | 0.021 / 0.006 / 0.009 |
| 2 | 0.0052 / 0.122 (×0.042) | 0.0126 / 0.976 (×0.013) | 0.0104 / 0.978 (×0.011) | 0.018 / 0.006 / 0.005 |

A context three to eight times richer (fresh spread 0.77–0.98 against 0.12–0.27) is compressed to a
latent spread of 0.004–0.013 — forty-five- to two-hundredfold (the ×-ratios in the table) — while the
incumbent's already-narrow context loses a further nine- to twenty-four-fold. The live-model bars
(`jp_std > 0.0`, `jepa_std > 0.0`) clear on every one of these nine models; the module's own 0.05 floor
would fail all nine. Stage −1's brain-free 0.0009 was an underestimate of the in-brain spread but not of
the mechanism.

### 4.3 G-ORDER — the world model does not read succession

Prediction error per 256-wake-row block, ordered (w2) against per-cycle-shuffled (w2-shuffled):

| seed | w2, blocks 4–6 | w2-shuffled, blocks 4–6 | last-block difference (w2 − shuffled) |
|---|---|---|---|
| 0 | 0.00253, 0.00199, **0.00139** | 0.00303, 0.00198, **0.00134** | +0.00005 |
| 1 | 0.00286, 0.00176, **0.00135** | 0.00356, 0.00169, **0.00178** | −0.00043 |
| 2 | 0.00276, 0.00193, **0.00191** | 0.00374, 0.00292, **0.00179** | +0.00012 |

The pre-registered wording ("lower by more than 3 dp on ≥ 2 of 3") was ambiguous and is scored both
ways: as a difference ≥ 0.0005, 0 of 3 seeds; as a sign-consistent advantage over the last three
blocks, 0 of 3 (every seed flips sign at least once). Under a stream whose succession has been
destroyed the model's error is the same, to the fourth decimal, as under the natural one. G-ORDER FAILS.

### 4.3b G3 (the fold) — the incumbent context reads no succession either

Three cells added after the panel (the design panel's A0 refuted-branch cell, PH-6): the UNTOUCHED brain
with both packs re-permuted under one fresh permutation per cycle (`--arm incumbent_shuffle`, seeds 0–2,
3000 ticks; these cells carry the git head `6a1a437` and UTC start/end stamps in their JSON):

| seed | incumbent, blocks 4–6 | incumbent-shuffled, blocks 4–6 | last-block difference | spread trained / fresh, shuffled | `curiosity_mag` @3000, incumbent / shuffled |
|---|---|---|---|---|---|
| 0 | 0.00423, 0.00132, 0.00157 | 0.00123, 0.00097, **0.00083** | +0.00074 (shuffled LOWER on every block) | 0.032 / 0.264 | 0.0000 / 0.0000 |
| 1 | 0.00104, 0.00105, 0.00074 | 0.00117, 0.00097, 0.00081 | −0.00007 | 0.013 / 0.271 | **0.0200 / 0.0000** |
| 2 | 0.00143, 0.00085, 0.00070 | 0.00208, 0.00130, 0.00090 | −0.00020 | 0.0044 / 0.122 | 0.0000 / 0.0000 |

No sign-consistent advantage for the natural order (seed 0 is the wrong way on all three blocks; the two
"ordered lower" seeds are within 0.0002), the same collapse (ratios ×0.036–0.12), and seed 1 — the one
incumbent seed W3 recorded "alive" at 3000 (127 distinct, 0.0200) — reads 0.0000 with one distinct value
under the frame shuffle while its learners hold 10 / 17 prototypes against 9 / 15. G3 FAILS as G-ORDER
did: on the incumbent context, as on W2's, the world model cannot tell the natural succession from a
destroyed one. The grower under the shuffled incumbent reads 0.67 / 0.84 / 0.12 against persistence
0.95 / 0.96 / 0.98 — the same "below persistence" shape as the ordered incumbent.

### 4.4 G1 — W-R1's premise: curiosity under W2 moves without the world model learning succession

| seed | arm units @1500 → @3000, w2 / shuffled | `curiosity_mag` last-200 mean @3000, w2 / shuffled | `raw_cur == 0` fraction after row 300, w2 / shuffled |
|---|---|---|---|
| 0 | 55 → 80 / 54 → 57 | **0.1322** / 0.0131 | 0.54 / 0.57 |
| 1 | 51 → **51** / 43 → 56 | **0.0000 (1 distinct)** / 0.0395 | 0.87 / 0.41 |
| 2 | 87 → 102 / 59 → 91 | 0.0919 / 0.0528 | 0.62 / 0.70 |

Two of three seeds have w2 above its shuffled control; the rule needed three (probe power, FP 1/8).
G1 FAILS the bar — **and the pinned direction (§3.2: w2 does NOT exceed its shuffle) was REVERSED on
seeds 0 and 2**, which §3.2's escalation clause said would send W-R1 to an S = 8 sweep. Adjudicated
here, not skipped: that clause's premise was "the brain-free proxy diverged from the brain" — the world
model reads succession after all — and §4.3 shows it did not (identical error under order and
permutation on all three seeds), so a w2-above-shuffle difference on two seeds is not the world model
reading order; an S = 8 W-R1 column would test a contrast §3.4's rule already cannot turn into a build
(G-ORDER and G-W-R2b fail regardless). It was NOT run; it costs ~35 min (16 cells at 6000 ticks, 4
workers) and is available on request. This is the one place the pre-registration and the outcome
disagree, and it is recorded as such rather than folded into "as predicted".

What the six cells DO support, stated at their strength: recruitment is SUFFICIENT for a reading and its
absence returns the gauge to the 4-dp floor — seed 1's w2 arm stopped recruiting between 1500 and 3000
ticks (51 units at both marks) and read 0.0000 with one distinct value while its shuffled twin, still
recruiting (43 → 56), read 0.0395. It does NOT order the six cells: second-half recruits (25, 3, 0, 13,
15, 32) against `curiosity_mag` (0.1322, 0.0131, 0.0000, 0.0395, 0.0919, 0.0528) give Spearman ≈ 0.83
with one within-pair inversion — on seed 2 w2 recruited 15 against the shuffle's 32 and read HIGHER
(0.0919 vs 0.0528) — and the incumbent pair does not fit at all (seed 1 recruited one unit and reads
0.0200 with `raw_cur` nonzero on 96 % of ticks; seed 0 recruited two and reads 0.0000). "Every alive
reading is a recruitment transient" is therefore a HYPOTHESIS with seed 1 as its evidence and seed 2 and
the incumbent seed 1 as its counter-cases (the panel's I-2/PH-4; an earlier draft stated it as
established). What is established is narrower and enough for the verdict: the W2 contrast the plan's
W-R1 control was built to isolate — treatment above its own time-shuffled twin — does not hold on 3 of 3.

Against the incumbent, the honest pairs are: w2 @3000 over the incumbent's @1500 mean, 183× (seed 0)
and 37× (seed 2) — the incumbent reads 0.0000 at 3000 on those seeds, so no same-checkpoint ratio
exists; and on seed 1 the OPPOSITE, w2 0.0000 against the incumbent's 0.0200. (An earlier draft's
"30–190×" was a cross-checkpoint, cross-code composite; the panel's I-3/PH-10.)

Measured consequence of the frame shuffle (the panel's I-6): the control's vision/audio learners
recruited a different alphabet on every seed — 9–10 / 17 prototypes against w2's 9 / 15 — so the
control kept the FRAME marginal, not the ID marginal. Harmless to G-ORDER and G1 under identical error
trajectories, and exactly what the delivery-level shuffle of §3.5 removes if W2 ever proceeds.

### 4.5 W-R2 — entropy rises by recruitment; the grower cannot use the alphabet

| seed | `entropy_first` bits / categories: incumbent / w2 / w2-shuffled | grower last-third held-out accuracy / majority / persistence — w2 | same — w2-shuffled | `n_modes` w2 / shuffled |
|---|---|---|---|---|
| 0 | 1.8801 / 7 · **5.3851 / 56** · 5.2398 / 57 | 0.017 / 0.062 / 0.051 | 0.039 / 0.045 / 0.041 | 80 / 57 |
| 1 | 1.0369 / 4 · **5.1823 / 51** · 4.8502 / 46 | 0.013 / 0.132 / 0.062 | 0.058 / 0.081 / 0.048 | 51 / 56 |
| 2 | 2.0415 / 6 · **6.0820 / 97** · 5.6943 / 76 | 0.017 / 0.048 / 0.039 | 0.015 / 0.019 / 0.075 | 102 / 91 |

G-ID (W-R2a's twin) PASSES 3 of 3, by 3–5 bits — the RBF-position code under the arm learner's cosine
vigilance of 0.85 makes nearly every (vision position, audio position, joint fold) combination its own
prototype: 51–102 units against 5–7. G-W-R2b FAILS 0 of 3 against persistence and 1 of 3 against the
shuffled arm: on a 51–102-way alphabet with a 1000-row buffer (10–20 rows per class) the grower's
next-category accuracy is 1.3–1.7 %, below its own majority floor (5–13 %) and below persistence
(4–6 %). The shuffled arm does no worse (1.5–5.8 %): the delivered order carries nothing the grower can
read either. For reference the incumbent grower reads 0.12 / 0.78 / 0.44 against persistence 0.94–0.99
(s2's "below persistence" finding, unchanged).

### 4.6 Verdict

PROCEED required G5 ∧ G-ORDER ∧ G1 ∧ G-W-R2b. G5 passes; G-ORDER and G-W-R2b fail on the direction
§3.2 pinned before the run; G1 fails the 3-of-3 bar with the pinned direction REVERSED on seeds 0 and 2
(adjudicated in §4.4). **W2 is refused on measurement before a build** — the rank-2c outcome. No module,
port, edge, test or plan constant was written; the tree is unchanged but for this record.

What the refusal establishes, stated so it is not re-derived:

1. **The arm's curiosity chain cannot be moved by a richer context, because its world model reads no
   succession.** `JepaWorldModel`'s latent collapses on every context this body sees (§4.2), its
   prediction error is identical under order and permutation (§4.3), and the tracked packs have no
   first-order succession to read anyway (§3.1c, replicated). W-R1 as written — treatment above its
   time-shuffled control — is unreachable on this chain and this world. That is a fact about the
   INSTRUMENT (the world model) and the STIMULUS (the pack), not about the body: the plan's null
   sentence, "intrinsic to the world model over this body", is corrected to "intrinsic to the world
   MODEL — a collapsed latent — and to a world with no succession".
2. **A richer context moves the curiosity gauge without the world model learning anything about
   succession.** Under W2 the gauge rose to 0.09–0.13 on seeds 0 and 2 (and to 0.0000 on seed 1) while
   §4.3 shows the ordered and permuted worlds indistinguishable to the model; recruitment of new arm
   prototypes is sufficient for a reading and its absence returns the gauge to the 4-dp floor (seed 1),
   but recruitment does not order the six cells (§4.4) — what drives the remaining variance is not
   measured here. The gauge itself resolves 4 dp of an EMA that decays whenever either drive is nonzero,
   so "1 distinct value" means "below 5e-5", not "frozen" (§3.7).
3. **Delivering ids as one RBF position per band fragments the arm's alphabet ten- to twentyfold**, and
   nothing downstream of the arm learner — the Q rows, the curiosity keys, the grower's task — is sized
   for it. A different code (a sparse 16-cell splice, measured in A0) fragments harder (93 units at 742
   rows). The remedy would be a wider or hierarchical context — FR-030's migration event, stated in
   advance by the plan and not taken here.
4. **The plan's disposition "Phase 2's L2 is re-targeted at prediction error" cannot be applied as
   written**: the prediction error it names is the collapsed one. Applying it would aim the growth
   trigger at a signal that is flat by construction. This needs the owner (§5).

### 4.7 Findings owed regardless of W2, now measured in-brain

- `cerebellum/jepa_world_model.JepaWorldModel` collapses its latent under the arm's own defaults on
  every seed and every context in this record (nine of nine models; spread ratios 0.005–0.11). The
  VICReg hinge's restoring force is proportional to the latent's own deviation over a floored std, so
  the collapse is a fixed point. The module's own self-check already asserts `full_std > 0.05` (and 3×
  its ablation) on a synthetic stream; the only bars on the LIVE model are `autonomous_loop`'s
  `jp_std > 0.0` and `live_brain`'s `jepa_std > 0.0`, which 0.005–0.034 clears (§3.1). **Filed as its
  own arc** with the red already specified: stage −1(b)'s planted 3-bit successor world must give
  perr(ordered) < perr(shuffled); the module's EXISTING 0.05 floor applied to the live model (not a new
  fraction); the hinge/invariance gradient ratio counted before the rule.
- The curiosity gauge's RESOLUTION and its unmeasured hold: `state()` rounds `_cur_mag` to 4 dp; inside
  the `if raw_cur or raw_emp:` guard the EMA decays whenever either drive is nonzero and is held only when
  both are zero — a case that exists in code and was not measured here (no both-zero fraction, no
  unrounded series logged). **Filed** with the JEPA arc as "gauge resolution + unmeasured hold" (a fix
  moves the number on every seed and owes its own P8 item).
- s2c's sentence "the pooled real channels enter that context as scalars a world model masters in a few
  hundred ticks" described collapse, not mastery. The s2c record stands; this file is the correction.
- The incumbent grower on seed 0 at 3000 ticks reports `predict_accuracy` 0.0 and a last-third majority
  floor of 0.0 beside a persistence floor of 0.94 — a floor of exactly zero is a claim about the
  instrument (`_heldout_baselines` on a held-out slice whose majority class is absent from the training
  slice?) and is noted here for the L1a arc, whose subject it is.

## 5. Questions only the owner can answer

- **Q1 — Part F.** W-R1's stimulus fails plan Part C on the exact signal it names (§4.3) and the JEPA
  collapse is an incumbent defect (§4.7). The plan's null disposition sends Phase 2's L2 at this
  prediction error. Ratify one: (a) the JEPA-collapse arc runs BEFORE L2 (and before any task whose red
  reads `prediction_error`), with L2's re-target held until that arc's red is green; or (b) L2 is
  re-targeted at a different learning signal named by the owner; or (c) the disposition stands as
  written and L2 targets the collapsed error (not recommended: a growth trigger on a flat signal).
- **Q2 — W2's remaining half.** The plan's W2 sentence also asked for the encoder's surprise in the
  context. With the id lane refused, is a surprise-only context change (one scalar, no alphabet
  fragmentation) worth a probe of its own, or is W2 closed whole? This record closes it whole unless
  the owner reopens the surprise half.

## 6. Disposition

- W2: **REFUSED ON MEASUREMENT** (this record; the id lane — the surprise half is Q2); nothing built;
  `PLAN-FINAL` W2 row and Part E updated; a W2 row ADDED to `tasks.md` in the W3/A2c shape (the fold;
  §3.4's "tasks.md's W2 row closes" presumed a row that did not exist — the panel's INT-3); Phase 1's
  exit record (`evidence/w-exit.md`) written beside it.
- Open, filed, escalated (the completion gate's "file it and escalate the queue" clause): the
  JEPA-collapse arc; the `_cur_mag` held gauge (bundled). Q1 and Q2 above.
- The three-lens panel for this arc reviewed THIS RECORD (the arc lands no code); its findings and the
  fold are §7. The fold is a stacked commit; per the plan's A.4 step 4 it is stated here as
  battery-verified only, not re-panelled.

## 7. The arc's panel — three lenses on the record at `6a1a437`, and the fold

Three read-only lenses (instrument / policy-honesty / integration), every finding reproduced by a
command the lens ran; the full reports are archived (`archive/w2-2026-09-05/panel/record_panel.json`).
All three verdicts: the refusal follows from the cells as recorded. What they found wrong was my
overreach around it. Ledger (dispositions applied in this fold unless marked):

| id | severity | finding | disposition |
|---|---|---|---|
| I-1 / PH-1 | MAJOR | "`_cur_mag` is a frozen latch": false — `state()` rounds to 4 dp and the EMA inside the guard decays whenever `raw_emp` is nonzero; the incumbent's `raw_cur` is nonzero on 91–98 % of ticks at 1e-6–1e-10 while the gauge reads 0.0000 | ACCEPTED: §3.7, §4.6(2), §4.7 rewritten; the filed item re-scoped to "gauge resolution + unmeasured hold"; w-exit and the plan row corrected |
| I-2 / PH-4 | MAJOR | "curiosity follows recruitment": seed 2 inverts within the pair (w2 +15 units / 0.0919 vs shuffled +32 / 0.0528); Spearman 0.83 over six cells; the incumbent pair does not fit | ACCEPTED: §4.4 restated at strength (sufficient, not ordering; a hypothesis with counter-cases); w-exit, plan row (b), and the memory entry corrected |
| I-3 / PH-10 / INT-5 | MAJOR/MINOR | "30–190×" is a cross-checkpoint composite with no denominator; at 3000 the incumbent reads 0.0000 on seeds 0 and 2 and seed 1 is the opposite | ACCEPTED: replaced everywhere by the actual pairs (183× / 37× against the incumbent's @1500 mean; seed 1 reversed) |
| PH-2 | MAJOR | "no consumer of `latent_std()` outside two self-checks": false — the module's own self-check asserts `full_std > 0.05 and > 3× ctrl` and `tests/instruments/jepa_highdim_probe.py` asserts `> 2× abl`, both on synthetic streams; only the live-model bars are `> 0.0` | ACCEPTED: §3.1, §4.2, §4.7 corrected; the JEPA arc's red now reuses the existing 0.05 floor on the live model; the chip re-filed |
| PH-3 | MAJOR | G1's pinned direction was REVERSED on seeds 0 and 2 and §3.2's escalation clause was never adjudicated | ACCEPTED: §4.4 adjudicates it (the clause's premise — the proxy diverged — is contradicted by G-ORDER; the S = 8 column not run, ~35 min on request); §4.6 no longer says "each on the pinned direction" |
| PH-5 | MAJOR | the plan edit enacted a sequencing rule ("no task whose red reads `prediction_error` starts before…") while Part H lists Q1 open | ACCEPTED: the plan sentence now RECOMMENDS Q1(a) and does not enact it |
| PH-6 | MAJOR | the design panel's two-branch A0 rule was not confirmed; its refuted branch's G3 was never run; "reads no succession" measured on the W2 context only; probe seed 0 @1500 replays A0's cell | ACCEPTED: §3.1 discloses; G3 run in the fold (§4.3b: fails the same way on the incumbent context); plan row (a) qualified |
| INT-1 / I-9 | MAJOR/MINOR | w-exit widened G5 to W3's 6000-tick columns, which were not re-run | ACCEPTED: w-exit caption and §4.1 scoped to 3000 |
| INT-2 | MAJOR | w-exit Table 2 cited the W1 evidence file for two rows that live in the plan's W1 row and commit `17e4613` | ACCEPTED: sources corrected; "bandit and buffer untouched" traced to the commit message |
| INT-3 | MAJOR | §3.4 promised to close a `tasks.md` W2 row that does not exist | ACCEPTED: W2 row ADDED to `tasks.md` in the W3/A2c shape; §6 says so |
| INT-4 | MAJOR | SC-011 and L-R2 "need Phase 1's stream" and Phase 1 delivered none; SC-011's stimulus measured absent | ACCEPTED as a RECORD (not a rule): one sentence at Part D's Phase 2 head and in w-exit; the re-scoping is the owner's |
| INT-9 | MINOR | pre-existing T076 inconsistencies in `tasks.md` (row names the deleted module; the O5 sentence describes it landing) and a Part D pointer to a Part E row that does not exist | ACCEPTED: `tasks.md` T076 row and O5 sentence marked WITHDRAWN with pointers; Part D pointer fixed |
| I-4 / PH-9 / INT-6 | MINOR | four wraps, not three; the W2 arms also replace `_safe_ctx` | ACCEPTED (§4.1) |
| I-5 | MINOR | §4.2 prose ratios not derived from its table | ACCEPTED (3–8×, 45–200×, 9–24×) |
| I-6 | MINOR | the frame shuffle changes the control's id alphabet (9–10 / 17 vs 9 / 15) — measured, undisclosed | ACCEPTED (§4.4) |
| I-7 | MINOR | ordering "protocol then cells" not verifiable from artefacts; cells carry no head or clock | ACCEPTED: disclosed at §4's head; the harness now writes head + timestamps (the G3 cells carry them) |
| I-8 / INT-7 | MINOR | §1c "cells 5–15 identically 0.0": cells 5–6 read < 0.003 | ACCEPTED |
| PH-7 | MINOR | the surprise half entered no cell; the plan row header did not say so; a cross-reference pointed at the wrong section | ACCEPTED (§3.0, the plan row header) |
| PH-8 | MINOR | "≈ 5 min wall": 12 | ACCEPTED |
| INT-8 | MINOR | "verbatim from Part F" was a paraphrase | ACCEPTED |

Not changed by the panel: the verdict, the four gates, every table cell in §4.1–4.5 (each lens
recomputed them from the archived JSON and found them exact), Q1 and Q2.
