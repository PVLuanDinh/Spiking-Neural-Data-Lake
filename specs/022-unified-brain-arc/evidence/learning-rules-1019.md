# Learning rules for the P4d engine: research note (2026-10-19/20) and E1, the charge of the live RPE rules

**Status.** Read-only research plus ONE measurement lane (E1). No code change came out of either; nothing here is a
claim the tree can make until its own gate runs. Owner, 2026-10-23 (verbatim): *"Yes: Session A adds the learning-rule
brief, appendix and E1 report as a docs-only evidence file in specs/022-unified-brain-arc/evidence/, stacked into the P8
arc's one battery."* This file is that evidence file. It condenses three lane documents written in session 286f18b6
("SNDL learning rule optimization"); the originals and the E1 harness are lane artefacts, untracked, on the owner's
machine, and are named here as provenance only, not as a reproduction path in this repository:

- brief: `D:\AI-project\lanes\research-1019\learning-rules\LEARNING-RULES-BRIEF.md` (2026-10-20);
- appendix: `D:\AI-project\lanes\research-1019\learning-rules\LEARNING-RULES.md` (2026-10-19);
- E1: `D:\AI-project\lanes\e1-rpe-charge-1019-work\` (`REPORT.md`, `PREREG.md` with addenda 1-6, `REPORT-TABLES.md`,
  `results\*.json`, and the lane's own harness scripts); board #4245 (lane result), #4246 (integrator re-derivation).

Two corrections carried from the brief: the **perception-tier caveat** (E1 measured the no-encoder brain; every E1
verdict below is tagged "no-encoder tier") and the `_plan_w` count (**6 of 8 seeds**, not 7 of 8; seed 1 is an exact tie).

## 1. Bottom line

1. **Do not add learning rules.** All four tiers of the owner's plasticity table (2026-10-19: STDP + WTA, e-prop,
   three-factor, homeostatic + structural) already exist in some form. The gaps are CHARGE (live rules with no measured
   effect on their objective; for the two RPE rules, measured on the no-encoder tier) and the STREAM (inputs with no first-order succession, W2), not a missing rule. Adding
   rules now adds P3 dormancy and P2 duplicates.
2. **E1 measured the two live three-factor rules** (valid run: 8 seeds, N = 600 ticks, repeat floor 0 flips, positive
   control 379 / 439 flips):
   - `ValenceLearner.learn` charges behaviour, NOT its objective (no-encoder tier). E-V1 t -1.02 / +1.98; E-V2 (same-stream replay)
     t +0.54 / +1.66; the zero-weight frozen predictor beat the learner on 6 of 8 seeds in the replay.
   - The `AutonomousArm._plan_w` update (no-encoder tier) charges behaviour (E-P1, 7 of 8 seeds above the floor), but neutering it RAISED
     held-out reward on 6 of 8 seeds (seed 1 an exact tie), t(pfrozen - real) = +2.82, past the mirrored bar. That is
     a HARM SIGNAL, not a pre-registered harm endpoint: replicate (E1b) before any disposition.
   - **TIER CAVEAT (found by the SNDL protocol session 2026-10-20, confirmed by file check):** E1 ran with the working
     directory set to its worktree, whose `models\` folder lacked the gitignored `eprop_nmnist.pt` and `eprop_shd.pt`.
     `engine/brain.py` loads both by cwd-relative defaults (`SNN_EPROP_CKPT`, `SNN_EPROP_AUDIO_CKPT`) and the harness
     set neither, so E1 measured the NO-ENCODER brain (`perception_tier` "stdlib"), not the production brain the
     daemon, int3 and the battery run (both encoders, sha256 prefixes 9b471619e39acfa0 and 6b32f9eb38beeca3). The two
     tiers are measurably different brains (seed 0, 200 ticks: neither encoder 28/0/0 vs both 6/9/20, repeat pair 0
     ticks apart). E1's verdicts hold for the no-encoder tier only, until E1b says otherwise.
3. **One rule per P4d job (P2), as the tree should stand:**

| P4d job | Rule | Status | Next gate |
|---|---|---|---|
| Grown units' input rows | admit, then FREEZE | incumbent (research-1016 G1 rank 1; L1b Oja harm t -1.96) | a challenger must beat freeze (E3) |
| Taught recurrent units | `growth_loop.trace_train` (random-e-prop shape, LOCAL) | not live; lands with the merged grower | grower merge (research-1016 DESIGN.md T0-T13) |
| Readout tuning | `OnlineReadout.learn` delta (LOCAL) | live, charge 0.0 on the deployed brain (one silent unit) | E2 after the merge |
| Third factor | RPE (`ValenceLearner`, `_plan_w`) | live; objective NOT charged (E1, no-encoder tier) | E1b |
| Reservoir | FIXED `KernelLsmPool` | G1(b) winner; G1(c) G-R1 red | E4, doubly gated |
| Stability | fast IP / iSTDP + slow scaling + Benna-Fusi | live on the chat path | none to add |
| Structure | grow on a failed solved predicate; sleep commits / prunes | owner #3423; merge pending | the merged grower's reds |
| Sleep plasticity | downscaling on Hebbian parts only (literature: helps STDP, not surrogate gradient) | R3 F1 null | E6 after E1b |

## 2. Plan, in order (each item is a lane brief; none has run except E1)

1. **E1b: replicate as a HARM test** (cheapest, highest information). Fresh seeds 8-15, same harness, pre-registered:
   `_plan_w` red = t(pfrozen - real) on held-out reward > 1.895 (the harm replicates); ValenceLearner red = frozen beats
   learning in same-stream replay. **Both perception tiers:** the production tier (the two encoder files copied into the
   worktree, or `SNN_EPROP_CKPT` / `SNN_EPROP_AUDIO_CKPT` exported as absolute paths) and the no-encoder tier E1 used;
   the harness prints `perception_tier` beside every result and refuses a run whose tier differs from the one requested.
   A harm that replicates on both tiers is the finding; one that appears on a single tier is a tier finding. Needs
   `resource:timing`. About 35 min of brain runs (40 runs at about 45 s each) per tier. E1's harness is lane scripts,
   untracked; E1b owes its harness committed with its pre-registration before it runs.
   *Integrator note, 2026-10-23, under the P8 null-scale amendment (landed in the same arc as this file):* a harm past a
   pre-registered bar is deleted without a scale re-test (amendment item 4), so E1b's `_plan_w` harm endpoint, if it
   replicates, is a harm disposition, not a null. ValenceLearner's objective, if it fails again, is a null only if E1b
   is powered for the claimed effect; an underpowered failure is INCONCLUSIVE and licenses nothing (item 4). A powered
   null owes a NULL-RECORD and the one 10x re-test (axis = ticks for a learning rule) before any deletion. Neither rule
   is an owner-drawn P4d part (those are the two networks, the reservoir and the return path), so the no-deletion clause
   for owner-drawn parts does not apply to them. The brief's earlier wording ("a P8 deletion candidate") predates the
   amendment.
   *Integrator note, 2026-10-23, under the P1 sample-size amendment (owner, verbatim: "switch E1b to a sequential design,
   P1 policy, make sure all sessions understand that"):* "Fresh seeds 8-15" above is superseded. Seeds 8-15 become look 1
   of a sequential design; the looks, alpha spending, futility bound and caps live in pre-data addenda 2-9 of
   `specs/022-unified-brain-arc/evidence/e1b-rpe-harm.md`, which govern (a later addendum rules where it differs from an
   earlier one: 3 makes the production tier decide, 4 sets H-V1's cap 152 (19 looks) at the power target under E1's
   resampled per-seed shape; 5 set H-P1's cap 184, which 7 replaces (owner "Smoothed shape") with 40 (5 looks) at the
   power target under a kernel-smoothed resample of that shape; 6 is the arc-2 panel's erratum and supersession list;
   8 states H-V1's cap 152 as powered on that kernel-smoothed shape too, owner 2026-10-25 "I approve the H-V1 under
   smoothed shape"; 9 raises H-P1's cap to 176 (22 looks) so it is powered on the shrunk, unshrunk and half-bandwidth
   smoothed shapes, owner 2026-10-25 "raise it in a new pre-data addendum").
2. **Grower merge** (research-1016 `design/DESIGN.md` T0-T13; owner #3403 / #3423 / O-D). It is what makes grown units
   fire.
3. **E2:** `OnlineReadout` charge on a brain whose grown units fire (blocked until 2; a re-run today returns 0.0).
4. **E3:** three-factor input-row rule (M = live prediction correctness, L1b's named candidate) vs FREEZE, an M-shuffle
   control, and the gate verdict logged per refit row. Only after E2.
5. **E4: reservoir plasticity** (LIF STDP + IP + normalization, or DA-BCM) vs the fixed pool at equal synapse count.
   Blocked on BOTH a stream with measured succession and an LIF implementation. Candidate stream: the owner's planned
   event-camera percept layer; test any stream with W2's ordered next-label MI vs shuffle first.
6. **E5: tier-1 STDP + WTA vs ART**, only after the owner answers Q-T1 (section 8).
7. **E6:** sleep downscaling on the Hebbian / R-STDP chat learner vs a no-sleep twin, 10-20% sleep fraction.

## 3. The answer per owner tier (appendix section 0)

| Owner's tier | Rule as drawn | Verdict for the SNDL engine | Why |
|---|---|---|---|
| 1. Primary representation | STDP + WTA | Do not add to the brain network now; park Q-T1. No unsupervised rule changes tissue weights on the live tick. The sensory front-end is the FROZEN offline e-prop encoders (`EpropEncoder.load`, only when the perception tier is armed); on top sit ART prototypes (`growth.growing_learner::GrowingLearner.observe`, not a spiking mechanism), NG-SNN residual admission and predictive coding (`readout.predictive_coding::PredictiveCodingNetwork.learn_step`, every wake tick). | STDP + WTA beside ART is a P2 adopt-or-merge event that needs a benchmark that exists; none does, so build the benchmark first (E5). Whether the representation layer MUST be spiking is an owner decision, not a benchmark. |
| 2. Credit assignment | e-prop | Use it only where a recurrent network needs multi-step credit. The readout keeps the online delta rule (`readout.online_readout::OnlineReadout.learn`, LOCAL). Grown taught units: `growth.growth_loop::trace_train` (LOCAL, fixed-random broadcast) is the merged grower's rule, not live today. Symmetric e-prop is forbidden (weight transport). | Delta: +0.143 live accuracy, t 4.13 (L1c), but charge 0.0 on the deployed brain (one silent unit, `l-exit.md`, not re-measured). e-prop proper is APPROXIMATES_GRADIENT and owes cosine > 0.5. |
| 3. Neuromodulation | R-STDP / dopaminergic e-prop | Already built and LIVE; what is missing is its charge. `plasticity.valence_stdp::ValenceLearner.learn` and the `_plan_w` RPE update run every arm tick; R-STDP (`readout.interaction_learning::SpikingInteractionLearner`) runs on chat turns. | E1 (section 7) measured the first two, on the no-encoder tier. |
| 4. Stability and capacity | homeostatic scaling + structural plasticity | Already LIVE in parts; open items are charge and sleep. Scaling + IP + STP + Benna-Fusi (`plasticity.homeostatic_plasticity_stack` via `PlasticityRuleStack`, chat turns); four sleep pruners; `sleep.structural_rewiring::StructuralRewiring.rewire` LIVE but INERT (turnover 0 on 8/8 seeds). | G6(a)'s pruner neuter-A/B returned no result. The literature says slow scaling alone is too slow; the fast partners (IP, iSTDP) are already in the stack. |

**Why not the full four-tier stack as drawn.** The literature does say several plasticity forms must act together
(Zenke, Agnes and Gerstner 2015 found that blocking any one component broke the memory network; SORN worked only with
STDP, intrinsic plasticity and normalization together, Lazar 2009). SNDL already has every tier in some form; its
measured gap is uncharged live rules and a stream with no first-order succession, so new rules would add dormancy, not
capability.

## 4. What the tree had (code map, SNDL `main`, 2026-10-19)

Reachability from an AST import closure proves "imported", not "charged".

- **Tier 1:** no unsupervised rule changes tissue weights on `Brain.tick`. Live learning: ART prototypes (wake),
  predictive coding (wake), Oja residual in `faculties.percept_surprise` (only when the perception tier is armed).
  Research-only or EXEMPT: `reasoning/stdp_wta_bayes.py`, `reasoning/triplet_stdp.py`, `GrowableSTDP` (slated RETIRE
  by DESIGN.md T8), the MNIST STDP instruments. L1b's Oja / BCM on grown rows were removed after measurement (both are
  triaged in `null-scale-triage.md`, block 11 and U4).
- **Tier 2:** `NGSNN.fit` (DIRECT_SOLVE, sleep refit every 20 ticks) + `OnlineReadout.learn` (delta, LOCAL, live,
  uncharged); `LaminarLocalReadout.observe` (delta + fixed-random feedback, only with the perception checkpoint); e-prop
  kernels are offline trainers whose frozen weights the brain loads. `reasoning/reservoir_eprop.py`: hybrid 54.4% vs
  pure e-prop about 55% (no gain from training the reservoir).
- **Tier 3:** `ValenceLearner.learn` and `_plan_w` live every arm tick; R-STDP live on chat turns; NACA dormant; ACh
  `plasticity_scale` dormant (L1b: its tail mean is 0.03, a gate that vanishes where a rule would act);
  `plasticity/eligibility_algebra.py` is the authoring contract, bit-equivalent to R-STDP.
- **Reservoir:** weights fixed; `engine/` had no pool construction, so `Brain.tick` did not step the reservoir. The G1
  reservoir readout is a fixed ridge `ReferenceReadout`, not a learning rule: P4d's "readout neurons tuned by the brain
  through a learning rule" is not built on the live tick.
- **Gradient census** (`gates/gradient_claim_gate.py`, 211 rows on 2026-10-19; re-derive before relying on it): 82
  LOCAL, 97 NOT_A_RULE, 20 APPROXIMATES_GRADIENT (every e-prop row), 6 CONTROL, 4 DIRECT_SOLVE, 2 EXACT_GRADIENT.
  `trace_train`, `OnlineReadout.learn`, `ValenceLearner.learn`, `RSTDPNeuron.reward`, BCM, Oja and iSTDP are LOCAL. No
  row for `stdp_wta_bayes`, `triplet_stdp`, `delay_learning`, `reservoir_eprop`.

## 5. What the literature says, mapped to SNDL (paraphrased; sources in section 9)

1. **Combinations, not single rules, give stable learning** (Zenke, Agnes and Gerstner 2015: triplet STDP + fast
   heterosynaptic plasticity + slow homeostasis + consolidation formed recallable assemblies; Zenke and Gerstner 2017,
   read from search snippets only: scaling is far slower than Hebbian change, so fast compensatory processes are
   needed). SNDL: the chat-path stack already pairs fast (IP, iSTDP trim) with slow (scaling) and a cascade
   (Benna-Fusi). Nothing to add; measure it.
2. **STDP + WTA needs homeostasis inside the rule** (Diehl and Cook 2015: unsupervised MNIST from STDP + lateral
   inhibition + an adaptive threshold). SNDL: the re-proposal guard already shows accuracy tracks learner scale, not the
   plasticity strategy; a brain-network STDP + WTA owes a new regime or a benchmark against ART.
3. **Three-factor rules need a seconds-scale eligibility trace** (Gerstner et al. 2018: about 0.2-2 s predicted; about 1
   s measured in striatum, several seconds in cortex). SNDL: a new third-factor rule states its trace constant in TICKS
   against the delay to its factor.
4. **e-prop is the three-factor rule for recurrent spiking nets** (Bellec et al. 2020; random e-prop needs no
   transport; Liu et al. 2021 MDGL, read from a snippet only, reports cell-type-specific signals doing better). SNDL:
   `trace_train` is the random-e-prop shape and the rule for taught recurrent units; MDGL is a research direction only.
5. **Reservoir plasticity pays only when the input has temporal structure** (Lazar 2009 SORN; the evolvable-LSM paper,
   Sci Rep 2023, found STDP in the liquid a small gain and DA-modulated BCM a larger one). SNDL territory:
   `reasoning/sorn.py` reproduced the SORN gain on structured words (89.1% vs 84.4%, `g1b-pool-selection.md`), but on
   the live percept stream SORN's best configuration had plasticity OFF, and SORN was non-selectable (binary units).
   Both preconditions for reservoir plasticity were measured unmet: a stream with succession and an LIF implementation.
6. **Sleep renormalization helps Hebbian learners, not gradient learners** (arXiv 2601.08447, 2026: weight decay toward
   a baseline at 10-20% sleep fraction stabilized an STDP SNN and did nothing for a surrogate-gradient one; arXiv
   2405.16922: continual learning needs state over several timescales). SNDL: target the Hebbian / R-STDP chat learner,
   not the delta readout or the DIRECT_SOLVE refit.
7. **Growth literature freezes what it grows** (research-1016 G1 rank 1), in line with L1b's harm: rows moving under a
   readout held stale by a closed growth gate collapsed predictions (Oja t -1.96).

## 6. Experiments E1-E6 (pre-registerable; statistic: paired t over 8 seeds against 1.895, live-vs-live baseline)

| # | Experiment | Arms | Red (must be reachable) | Non-method control | Unblocks |
|---|---|---|---|---|---|
| E1 | Charge of the live three-factor rules | `ValenceLearner.learn` and `_plan_w` neutered vs live; objective = what each rule serves (held-out reward-prediction error; held-out reward), not arm coverage | neuter raises held-out RPE at the bar | magnitude-preserving time shuffle of DA / RPE | any new tier-3 proposal |
| E2 | `OnlineReadout` charge on a brain whose grown units fire | live vs neutered readout | a unit fires on the live stream and the neuter moves live accuracy | frozen-beta twin | E3; BLOCKED today (one silent unit, charge 0.0 on record, `l-exit.md`) |
| E3 | Grown-unit input rows: three-factor rule vs freeze | freeze; three-factor with M = live prediction correctness; M time-shuffled | three-factor beats freeze at the bar AND the M-shuffle arm does not | M-shuffle (same magnitude) | the growth tier's rule |
| E4 | Reservoir plasticity vs fixed `KernelLsmPool` | equal synapse count | blocked until a stream shows ordered next-label MI above its shuffle; then carried-state D and separation must beat the fixed pool | random liquid at equal count | the reservoir tier |
| E5 | Tier-1 benchmark | ART vs STDP + WTA (adaptive threshold) on held-out category structure | STDP + WTA beats ART; otherwise not P2-eligible | label-shuffle null | a P2 adopt-or-merge of tier 1 |
| E6 | Sleep downscaling on Hebbian-trained parts | ON vs no-sleep twin, 10-20% sleep fraction | held-out retention improves at the bar | downscaling on random synapses at equal total | the owner's 2026-09-27 sleep-plasticity directive |

**The unifying algebra is a research direction, not a recommendation.** Every rule above fits
`dw = eta * M_j(t) * e_ij(t)` (pure STDP: M constant; R-STDP: a global scalar; delta readout: M = the error; e-prop: a
neuron-specific learning signal), and `plasticity/eligibility_algebra.py` states it as the authoring contract. Making
it THE single implementation would retire four code paths and owes an A/B per retirement (P2); no benchmark compares
them as one family today.

## 7. E1 result (lane e1-rpe, 2026-10-19; board #4245, integrator re-derivation #4246)

Read-only measurement on an origin/main d501c7aa worktree; N = 600 ticks per run (tick rule: 220 ticks took 17.7 s, so
1,200 ticks would exceed the hour); held-out window H = ticks 301-600; 8 seeds; paired t bar 1.895 (df 7) and at least
5 of 8 seeds in the direction. PREREG frozen (sha256 d69cd655...ca38fa) with addenda 1-6: arm-instance scoping of the
wrappers, the tick rule, two pauses and one halt that yielded `resource:timing` to batteries; no arm, endpoint, bar or
disposition changed.

**Harness validity: ALL HOLD.** 0 errors and 0 swallowed exceptions; series length = learn calls in every run; every
patched arm's counter > 0; pfrozen learning rate 0.0 in every run; `pred == r - rpe` (< 1e-9) on every real call; a
repeat arm (real seed 0 again) had 0 action flips and an identical series; the positive control flipped 379 (seed 0)
and 439 (seed 1) actions; the replay's planted control passed (t 13447.7 / 39.5). Disclosed: 564-592 valence and 24
learn calls per run came from OTHER `ValenceLearner` instances (passed through; the instance scoping was necessary);
the shuffle arm's swapped RPE differed from the native one on nearly every call, with no modulo wrap.

| Endpoint | Pre-registered test | Result | Verdict |
|---|---|---|---|
| E-V1 | t(MAE vfrozen - real) > 1.895 AND t(MAE vshuf - real) > 1.895 | t -1.02 (3/8) and +1.98 (6/8) | NOT MET |
| E-V2 | replay: t(frozen - learn) > 1.895 AND t(shuffled - learn) > 1.895 | t +0.54 (2/8) and +1.66 (6/8) | NOT MET |
| E-V3 | action flips vfrozen vs real (reported) | 112-387 of 600 on every seed, floor 0 | charges behaviour |
| E-P1 | pfrozen flips above the 0-flip floor on >= 6/8 seeds | 7/8 (seed 1: 0 flips) | MET |
| E-P2 | t(reward real - pfrozen) > 1.895 AND t(real - pshuf) > 1.895 | t -2.82 (1/8) and -1.83 (2/8) | NOT MET, opposite direction |

Per-seed values (lower MAE is better; reward is mean r over H):

| seed | MAE real | MAE vfrozen | MAE vshuf | replay learn | replay frozen | replay shuf | r real | r pfrozen | r pshuf | pfrozen flips |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4307 | 0.4515 | 0.3920 | 0.4307 | 0.3902 | 0.4344 | 0.2265 | 0.2196 | 0.2068 | 162 |
| 1 | 0.5284 | 0.7914 | 1.0304 | 0.5284 | 1.0639 | 0.5673 | -0.9935 | -0.9935 | -0.9923 | 0 |
| 2 | 0.6753 | 0.6843 | 0.8656 | 0.6753 | 0.6375 | 0.6459 | 0.0634 | 0.2198 | 0.2198 | 130 |
| 3 | 0.5057 | 0.4294 | 0.5143 | 0.5057 | 0.4974 | 0.5285 | 0.4067 | 0.4805 | 0.4805 | 136 |
| 4 | 0.4157 | 0.2222 | 0.6302 | 0.4157 | 0.3566 | 0.4806 | 0.3041 | 0.4843 | 0.3128 | 321 |
| 5 | 0.4595 | 0.3435 | 0.4329 | 0.4595 | 0.5037 | 0.4697 | 0.4737 | 0.5127 | 0.6029 | 90 |
| 6 | 0.4878 | 0.3009 | 0.5125 | 0.4878 | 0.4169 | 0.4758 | 0.4013 | 0.4882 | 0.3449 | 161 |
| 7 | 0.3093 | 0.1496 | 0.4463 | 0.3093 | 0.2557 | 0.4167 | 0.1665 | 0.1909 | 0.2703 | 90 |

Seed 1 is degenerate on reward (mean r -0.99 in every arm; `_plan_w` minimum 0.0033) and is the outlier in E-V1 / E-V2.
`_plan_w` moved well off its 0.1 initial value on every seed (max |deviation| 0.165-0.277).

**Pre-registered dispositions (all on the no-encoder tier).** ValenceLearner: charges behaviour, not its objective.
`_plan_w`: charges behaviour, objective unresolved. Disclosed without a pre-registered branch: neutering `_plan_w` raised held-out reward on 6 of 8
seeds (seed 1 tied), t +2.82 for pfrozen - real; in the replay the zero-weight frozen predictor beat the learning
ValenceLearner on 6 of 8 seeds. Both are harm SIGNALS, not confirmatory findings: E1b, pre-registered as a harm test on
both perception tiers, is owed before either drives a change. Consequence: E3 and E6 assumed a tier-3 rule with a
charged objective; neither live RPE rule has one on the no-encoder tier, and the production tier is unmeasured.

**Scale of these results (P8 null-scale amendment item 1, landed 2026-10-23 in the same arc as this file).** Every arm acted on the arm's own
single `ValenceLearner` instance and its `_plan_w` vector (units: n/a for a rule's objective); 600 ticks per run, 300
held out; 8 seeds. Power: for E-P2 the SD of the paired difference was 0.0695 against a mean reward of 0.1311, so a
10% effect needs about 174 seeds (one-sided .05, power .8); E-P2 is therefore UNDERPOWERED for a 10% effect and is
INCONCLUSIVE on the objective under amendment item 4, not a null. No power figure was pre-registered for E-V1 / E-V2.
No disposition is taken in this file, so it carries no NULL-RECORD line; the record is written with the disposition,
after E1b.

## 8. Owner question and re-proposal guard

**Q-T1 (parked, not decided): must the brain's primary representation layer be SPIKING?** P4d's reservoir rule says
"Spiking only can win". Today the representation is frozen e-prop encoders plus ART prototypes. If the owner extends
"spiking only" to the representation layer, a spiking STDP + WTA layer (STDP + lateral inhibition + adaptive threshold)
becomes the candidate to replace ART under P2, and E5 decides it; if not, tier 1 stays and E5 runs only on demand. No
measurement can answer this; it defines what the brain is.

**Measured negatives that bind this question** (memory ledger `project-snn-refuted-and-negatives`):
- L1b: Oja-in-direction on grown units' input rows HARMED live accuracy (t -1.96); BCM neither gained nor cleared
  (t -1.36). Mechanism: rows moving under a stale readout.
- G5 Part A: the trace-delta readout challenger won 1 of 12 on discovery and 0 replicated, and was removed.
- L2 / T092: the grow-on-plateau trigger measured +0.625 units per seed at t 0.141 and was removed (triaged in
  `null-scale-triage.md`, blocks 15-16).
- G1(b): SORN's tuned configuration had plasticity off on the live stream; the plastic ports were removed.
- Pruning in the sparse kWTA regime: 88.8% to 83.5%; criticality-guided pruning tied activity pruning. REGIME-bound
  (kWTA = 1 sparse coders, single-layer MNIST), not a ban on P4d's sleep prune.
- Accuracy tracks the learner's scale, not the plasticity strategy (v3.43-v3.48).
- `reasoning/reservoir_eprop.py`: training the reservoir by e-prop gained nothing (54.4% vs about 55%).

**What any new rule owes** (copy into its lane brief): a `GRADIENT_CLAIM` row and a pass of
`gates/eprop_transport_gate.py` (a gradient claim owes cosine > 0.5 against the independent BPTT reference); a
pre-registered red with an input on which it must lose and a control that is not the method (shuffle, frozen twin, or a
`tools/charge_probe.py` neuter-A/B with its positive control); per-item neuter attribution, with a third factor
positive-controlled on the run's TAIL magnitude, not on liveness; unprompted disclosure; P8 clause 5 and its null-scale
amendment; a check against the guard above.

**Instrument lessons from E1** (each would have voided it): the working directory is a silent input (pin the perception
tier and print it with every result); a class-level monkeypatch hit foreign instances (scope every patch to the arm's
own instance); the dopamine block's `except Exception: pass` hides a broken patch as a perfect neuter (wrappers record,
then re-raise); a board lease expiry is not a resource release (check the holder's process before taking
`resource:timing`).

## 9. Sources (fetched 2026-10-19 with webgrab unless marked)

- Zenke, Agnes, Gerstner 2015, Nat Commun 6:6922, "Diverse synaptic plasticity mechanisms orchestrated to form and
  retrieve memories in spiking neural networks".
- Zenke, Gerstner 2017, Phil Trans R Soc B, "Hebbian plasticity requires compensatory processes on multiple
  timescales" (full text blocked; search snippets only).
- Lazar, Pipa, Triesch 2009, Front Comput Neurosci, "SORN: a self-organizing recurrent neural network".
- Bellec et al. 2020, Nat Commun 11:3625, "A solution to the learning dilemma for recurrent networks of spiking
  neurons".
- Gerstner, Lehmann, Liakoni, Corneil, Brea 2018, Front Neural Circuits, "Eligibility traces and plasticity on
  behavioral time scales".
- Diehl, Cook 2015, Front Comput Neurosci, "Unsupervised learning of digit recognition using spike-timing-dependent
  plasticity".
- Liu et al. 2021, PNAS, "Cell-type-specific neuromodulation guides synaptic credit assignment in a spiking neural
  network" (blocked; search snippet only).
- Three-factor learning in SNNs: an overview from a machine-learning perspective, 2025, arXiv 2504.05341.
- Sleep-based homeostatic regularization for stabilizing spike-timing-dependent plasticity, 2026, arXiv 2601.08447.
- Theories of synaptic memory consolidation and intelligent plasticity for continual learning, 2024, arXiv 2405.16922.
- Adaptive structure evolution and biologically plausible synaptic plasticity for recurrent spiking neural networks,
  Sci Rep 2023.

SNDL evidence cited: `g1b-pool-selection.md`, `l1b-synaptic-online.md`, `l1c-online-readout.md`, `l-exit.md`,
`g5-readout-variants.md`, `g1c-carried-state.md`, `null-scale-triage.md` (all in this directory).
