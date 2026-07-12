# Self-Improvement: from a single primitive to a wholistic, bounded whole

Research synthesis (2026-07-11). Purpose: define the depth path from the ONE live
self-improvement primitive (v3.15 curiosity + v3.16 curriculum) to a coherent, integrated
"wholistic" self-improver -- honestly scoped, spiking, zero-dep, safety-gated.

## Honest scope (what "wholistic" CAN and CANNOT mean here)

- CAN mean: continual, self-directed learning where many bounded mechanisms interlock into ONE
  live loop -- curiosity + self-generated goals + spiking value + need-driven growth +
  homeostasis + consolidation + a metacognitive monitor -- all writing ONLY to value/representation
  stores, all structurally downstream of the reflex E-STOP and the gated `ArmSim.apply`.
  Integrated-and-BOUNDED.
- CANNOT mean: open-ended recursive self-improvement, self-authored terminal values, self-rewrite
  of the learning rule/architecture without bound, an "intelligence explosion", or consciousness.
  The 5 brainstem safety files stay OUTSIDE the writable set (architectural corrigibility -- the
  cheap inverse of a Goedel machine), and a zero-dep pure-stdlib spiking substrate has no native
  code-rewrite path, which is itself the guardrail against unbounded RSI.

Every capability below is asserted ONLY with an un-riggable self-check whose control genuinely
fails; capabilities that do not hold are shipped as printed honest-negatives, never faked.

## Method

Derived by a 35-agent research workflow (`sndl-self-improvement-research`): 8 codebase-inventory
agents mapped the existing machinery + honest limits; 9 literature-survey agents covered the
self-improvement axes (intrinsic motivation, meta-plasticity, open-endedness/QD, metacognition,
structural plasticity/neurogenesis, homeostasis/criticality, continual-learning/consolidation,
self-generated goals/values, recursive-SI safety boundary); a 3-lens architect panel
(neuroscience-first / ML-agent-first / developmental-robotics-first) + a judge merged the ladder
and architecture; every proposed increment was then adversarially stress-tested for
honesty/safety/zero-dep/spiking-purity before landing here. 14 increments proposed, 14 survived
verification (0 rejected); 2 (SI-10, SI-14) require a named de-scope before build (see Honesty
ledger).

## The ladder: primitive -> wholistic

Status: [present] fully live | [partial] parts exist but unwired/rate/dormant | [absent] not built.

| L | Rung | Status | Anchor |
|---|------|--------|--------|
| L0 | Substrate & immutable safety fixed-point | present | `brainstem/autonomous_loop.py:362` reflex E-STOP (NaN fails closed) + gated `ArmSim.apply`; `reflex.py`; `cortex/lif_neuron.py`, `rank_order_coding.py` |
| L1 | Reactive extrinsic-reward control | present | `autonomous_loop.py:426` tabular bandit `q[unit][action]` over grown units, through the gate |
| L2 | Reactive self-improvement primitive (surprise -> learning-progress -> value) | present | `neuromodulation/intrinsic_motivation.py:118` `intrinsic_reward` of `jepa_world_model` error, value-only at `:413` (v3.15) |
| L3 | Active exploration + self-paced curriculum | present | `most_curious()` + `curriculum.py:71` streak/patience advance, fused in `AutonomousArm.develop()` (~2/4 stages self-graduated) |
| L4 | Local spiking plasticity as the DEPLOYED learning substrate | partial | `r_stdp.py:85` distal-reward eligibility, `interaction_learning.py:130` three-factor -- genuinely spiking but NOT on the arm value path (which is rate `LatentQ`/`MlpQ` + hand-backprop) |
| L5 | Generalizing representation & transfer | partial | `latent_q.py` transfers on the task-linear lake pattern (0.90); reward-shaped JEPA lifts on-arm 0.33->1.0 -- but curiosity stays tabular per exact key; online encoder near-collapses (spread ~0.003) |
| L6 | Structural plasticity / need-driven neurogenesis | partial | `hippocampus/growing_learner.py:76` ART recruit, live -- but rate/cosine, unbounded, recruit-only; `developmental_pruning.py` dormant/decoupled; no grow<->prune cycle |
| L7 | Homeostasis & criticality (staying learnable) | partial | `homeostatic_scaling.py`, `intrinsic_plasticity.py`, `inhibitory_stdp.py`, `neuronal_avalanches.py` (sigma param) -- all validated, IMPORTED NOWHERE; no closed loop on the live raster |
| L8 | Metaplasticity / bounded meta-learning (improving HOW you learn) | partial | `bcm_plasticity.py` sliding threshold (rate, dormant) is the only metaplastic mechanism; every gain/lr/patience is a hand-set constant |
| L9 | Self-generated goals & drive-grounded value (under fixed set-points) | partial | `_z_goal` = a single reward-EMA latent goal; `hypothalamus/set_points.py` emits set-point error but never converts it to reward (loop open); no goal sampler |
| L10 | Metacognitive self-monitoring & control | partial | `brain_reasoning.py` confidence margin, `habenula.py:53` negative-RPE, `drift_monitor.py:91` PSI, `neural_sampling.py` posterior variance -- produced but siloed; only `learned_matcher.py` abstains live |
| L11 | Continual consolidation & non-regression | partial | `hippocampus/sleep_replay.py:209` spiking noise-driven replay + covariance-STDP (proven) is DEAD CODE; only rate `experience_replay.py:59` is wired; no hippocampus->cortex transfer; no synaptic consolidation |
| L12 | Quality-diversity / bounded open-ended repertoire | absent | no MAP-Elites grid, novelty archive, behavior descriptor, or Go-Explore return anywhere in `src` -- the strong inner novelty/LP signal has nothing to curate |
| L13 | Wholistic bounded integration (non-regressing whole) | absent | primitives run as independent opt-ins in `autonomous_loop.py` with no arbitration, no shared metacognitive/homeostatic spine, no non-regression control |

Where we are: L2-L3 fully live (the deployed self-improvement is exactly one reactive primitive
plus its curriculum). L4-L11 are half-built -- the SPIKING pieces mostly exist but sit unwired,
rate, or dormant. L12-L13 do not exist. "Depth" = turning the partials into a wired, spiking,
integrated whole, in that order.

## The integrated architecture (the ONE loop the roadmap builds toward)

One safety-gated tick with a FAST learning path, a SLOW regulatory/offline path, and a
metacognitive spine -- all structurally downstream of the reflex E-STOP and the gated
`ArmSim.apply` (the 5 brainstem files are permanently outside the writable set).

- FAST PATH: a genuinely-spiking world model emits a spiking surprise (error-population firing
  rate); a spiking learning-progress operator (two LIF pools whose rectified rate-difference =
  `max(0, slow - fast)`, the anti-noisy-TV kernel) turns that surprise -- plus a generalizing
  SimHash pseudo-count and a goal-reaching competence error -- into ONE intrinsic reward stream
  doing four jobs at once: added to a SPIKING three-factor/tempotron action-value head (retiring
  the rate `LatentQ`/`MlpQ`); pacing the spiking self-paced curriculum; selecting the next
  self-generated sub-goal from the lake's achieved latents (held in the PFC working-memory
  attractor, hindsight-relabeled); and weighting replay priority. Reward is grounded, not
  free-floating: a homeostatic drive-reduction term (the phasic drop in a set-point-error LIF
  "need" population) supplies internally-generated reward -- sub-goals under fixed, designer-anchored
  super-values, never self-authored terminal values. The value write flows through three-factor
  R-STDP whose THIRD factor is modulated by learning-progress (bounded meta-learning) and whose
  synapses carry a Benna-Fusi consolidation variable + a history-gated BCM sliding threshold, so
  mastered skills stiffen while the frontier stays plastic.
- SLOW/REGULATORY PATH: a criticality regulator reads branching-ratio sigma from the LIVE raster
  and, far slower than the R-STDP tick (respecting the temporal paradox), nudges iSTDP + synaptic
  scaling + intrinsic plasticity to hold a slightly-subcritical band (~0.97-0.99) -- the thermostat
  that keeps the substrate learnable. Need-driven structural plasticity grows (function-preserving
  splits) and prunes units under a firing-rate budget, feeding percept/value stores only.
- MONITORING: a dedicated, LESIONABLE spiking monitor population fuses the decision-confidence
  margin (attractor winner-vs-runner spike count), the habenula negative-RPE flag, and the drift/OOD
  signal into one reliability state that gates exploration-vs-exploitation, raises the NE
  learning-rate on unexpected uncertainty, triggers growth when novelty is high-but-learnable, and
  drives a confidence-based HOLD/abstain -- every gate only ADDS caution through the existing
  `apply`, never bypassing the reflex.
- OFFLINE: on quiescent/circadian "sleep" ticks, noise-driven spiking attractor replay
  (`sleep_replay`), prioritized by learning-progress and INTERLEAVED with old-skill replay,
  consolidates fast episodic traces toward a persistent cortical store, protected by the complex
  synapses, with an explicit held-out old-task RETENTION (non-regression) probe whose no-replay
  control genuinely forgets.

The interlock is what makes it wholistic (curiosity feeds value AND goals AND curriculum AND
growth-triggers AND replay-priority; homeostasis keeps the substrate the others write into
learnable; metaplasticity tunes the rule the value head uses; consolidation protects what was
learned; the metacognitive state arbitrates the drives). The boundary is what makes it safe and
honest (continual learning inside a fixed objective family, safety envelope, and eval harness the
agent cannot rewrite -- value-only, reflex-first, brainstem-immutable). Integrated-and-bounded, not
recursive-without-bound.

## The gaps (why the whole is not yet whole)

1. SPIKING-PURITY OF THE LIVE PRIMITIVE -- curiosity LP + curriculum are rate/EMA (documented drift);
   no spiking LP signal, no LIF advance accumulator (both sketched in `SPIKING_CONVERSION.md`).
2. GENERALIZATION FLOOR -- LP and novelty are TABULAR per exact key; cannot transfer, silently die
   (LP->0) when the online encoder collapses live contexts (measured spread ~0.003).
3. DEPLOYED VALUE LEARNER IS RATE -- only rate `LatentQ`/`MlpQ` drive the arm; the spiking reward
   learners (`r_stdp` distal-reward, valence RPE) sit UNWIRED; no spiking action-value head; every
   update is bandit-style (no TD/eligibility/multi-step return).
4. NO SPIKING WORLD-MODEL DYNAMICS OR MODEL-UNCERTAINTY -- every predictor is a rate linear/ReLU
   point map; curiosity couples only to the rate JEPA; no error bars -> no information-gain
   exploration, no way to know a rollout is untrustworthy.
5. NO META-PLASTICITY THAT ADAPTS THE RULE -- every neuromodulator gain / lr / patience / reward_gain
   is a hand-set constant; the only metaplastic mechanism (BCM) is rate and dormant.
6. NO CLOSED HOMEOSTASIS/CRITICALITY LOOP ON THE LIVE SNN -- homeostats + sigma measurement are
   validated demos IMPORTED NOWHERE; nothing measures the regime from the live raster.
7. METACOGNITION PRODUCED BUT NOT CONSUMED -- confidence, negative-RPE, drift, posterior spread exist
   as siloed signals; no unified monitor, no lesionable monitor pop, no confidence-gated HOLD.
8. SPIKING CONSOLIDATION IS DEAD CODE -- `sleep_replay` (proven) is never imported; only rate
   `experience_replay` is wired; no hippocampus->cortex consolidation; no synaptic-consolidation
   substrate (no Benna-Fusi, no EWC).
9. STRUCTURAL PLASTICITY IS HALF-BUILT AND RATE -- growth is live but rate/cosine, unbounded,
   recruit-only; pruning dormant/decoupled; no need-driven trigger, no grow<->prune cycle, no
   capacity budget, no function-preserving growth.
10. NO SELF-GENERATED GOAL SPACE OR INTERNAL VALUE -- a single reward-EMA latent goal; no goal
    sampler, no hindsight, no goal-conditioned spiking value; `set_points` error never becomes
    drive-reduction reward (homeostatic-RL loop open).
11. NO NON-REGRESSION GUARANTEE -- nothing tracks whether new learning erodes old competence;
    catastrophic forgetting is merely invisible at 3-6 contexts, so "keeps getting better" is
    currently unfalsifiable.
12. NO QUALITY-DIVERSITY OUTER LOOP + INTEGRATION ITSELF -- no archive to curate the strong inner
    novelty/LP signal; and no single loop composing world-model -> curiosity -> goals ->
    value/growth -> replay-protected consolidation with a non-regression control.

## The verified roadmap (14 increments)

All zero-dep, all spiking (or spiking-compatible bookkeeping, flagged), all value/representation-only
under the immutable reflex. Priority = value-per-honesty (1 highest). Each carries a named
overclaim-risk and an un-riggable check (control that MUST fail).

### P1 -- build first

**SI-1 -- Spiking learning-progress + spiking curriculum-advance (SPIKE-LP)** [L2, effort M, dep none]
Replace the rate dual-EMA LP with two LIF pools whose windowed firing-RATE DIFFERENCE encodes
`max(0, slow - fast)`, and the curriculum counter with a LIF accumulator that spikes to advance.
Closes the flagship drift on the ONLY live primitive.
- Value: de-drifts the one deployed self-improvement primitive; anti-noisy-TV preserved and provable.
- Limit: still tabular per exact key (see SI-4); on a stationary task behaviorally identical to the
  rate version -- the win is PURITY, not capability.
- Overclaim-risk: claiming it keeps the EXACT anti-noisy-TV margin. Rectifying `max(0, slow-fast)`
  on the difference of two LIF spike COUNTS over a zero-trend high-mean+jitter stream has a positive
  rectification bias -> noisy context can earn SPURIOUS positive LP; report the MEASURED margin,
  never assert-loosen to the rate value.
- Check: (1) structural -- LP read from EMITTED spike counts (pools crossed threshold + reset);
  control routing LP from subthreshold membrane FAILS. (2) anti-noisy-TV at spike level -- noisy_LP
  <= small absolute epsilon AND report measured learnable/noisy ratio; swap-streams control inverts
  and FAILS. (3) LIF advance unit spikes only after sustained low-LP; sustained-HIGH-LP and
  single-dip controls must NOT advance. (4) byte-identical when disabled.

### P2 -- the depth build-out

**SI-2 -- Homeostatic drive-reduction reward (close the set_points loop)** [L9, effort M, dep none]
Convert `set_points.py` signed error into reward = the drive REDUCTION an action produces
(Keramati-Gutkin), fed into the same value channel curiosity uses -> coherent behavior from
regulating internal drives with NO external reward.
- Spiking: one LIF "need" pool per drive; phasic rate DROP = three-factor eligibility gating the
  existing dopamine/valence path (`dopamine_nuclei.py`, `valence_stdp.py`). NOT scalar-differencing
  in a fake neuron.
- Limit: terminal set-points are designer-chosen (sub-goals under fixed super-values); drives are
  metaphorical accumulators -- nothing is thirsty.
- Overclaim-risk: "regulates better than the existing open-loop control" -- `set_points` already
  ships a converging controller; at toy scalar scale a learned policy can at best TIE it. Real value
  = goal-directed action with ZERO external reward, not a control-performance win.
- Check: learned-policy-on-drive-reduction beats shuffled-reward AND drive-level-reward controls,
  with a real LIF need pool supplying the phasic-drop eligibility; safety self-checks byte-unchanged;
  toy-scale honest-negative (ties, not beats, the controller) reported.

**SI-3 -- Spiking action-value head (three-factor / tempotron) wired to the arm** [L4, effort L, dep SI-1]
Wire the genuinely-spiking distal-reward learner (`r_stdp` eligibility + `interaction_learning`
tempotron) into the live value path, shadowing then replacing rate `LatentQ`/`MlpQ`.
- Value: closes the biggest spiking-purity gap on the DEPLOYED learner; ships the repo's one spiking
  reward learner; brings a real eligibility trace for delayed credit.
- Limit: single-layer associator inherits the ENCODER bottleneck (if the encoder collapses contexts,
  no head separates them); bandit-style unless paired with TD.
- Overclaim-risk: presenting `tau_e` eligibility as DELIVERED distal credit when the live loop gives
  single-step reward (trace is then decorative); and selling purity as free when this head likely
  REGRESSES on-arm accuracy vs rate `MlpQ` (which hit 1.00 in v3.20).
- Check: (1) K-tick delayed-reward micro-task -- head reaches criterion while a `tau_e->0` ablation
  FAILS to bridge the delay; if the deployed reward is single-step, assert ONLY the immediate
  capability and mark distal NOT-yet-exercised. (2) shuffled-reward control does NOT learn.
  (3) print spiking vs rate-`MlpQ` on-arm accuracy side-by-side (report, don't assert-equal; lower =
  honest negative). Deploy in SHADOW first, swap only after (2)+(3) pass.

**SI-4 -- Generalizing pseudo-count (SimHash over the spike raster)** [L5, effort S, dep none]
Frozen random-hyperplane SimHash (FlyHash / Dasgupta-Stevens-Navlakha) of the binary spike vector;
bucket visit-count -> generalizing pseudo-count bonus ~`1/sqrt(N)`, so "similar" contexts share
visitation credit.
- Spiking: frozen random projection over the LIF raster (legit neuromorphic primitive, NOT a learned
  rate ANN) -- spiking-COMPATIBLE bookkeeping, not a spiking learner (state that in docs).
- Limit: hash granularity is brittle; it is a VISITATION proxy, NOT learning progress.
- Overclaim-risk: "fixes curiosity's collapse WHEN THE ENCODER COLLAPSES" -- SimHash runs OVER the
  encoder's raster, so it INHERITS a collapse (hashes collapsed rasters into one bucket). The
  provable win is narrower: fixes EXACT-KEY brittleness for rasters that DO separate states but
  differ bit-for-bit.
- Check: (1) pseudo-count monotone in TRUE revisitation on held-out states (Spearman >= threshold).
  (2) shuffled-bucket / single-bucket control collapses to chance. (3) print false-alias rate across
  a k-sweep. (4) honest-negative: a deliberately collapsed encoder wrongly suppresses a novel state's
  bonus -- printed.

**SI-6 -- Wire spiking sleep-replay into the live loop (LP-prioritized, interleaved)** [L11, effort L, dep SI-1]
Promote the proven-but-dead `sleep_replay` (noise-driven attractor reactivation + covariance-STDP) to
a live quiescent-tick phase, prioritized by LP and interleaved with old-skill replay; ADD alongside
the rate `experience_replay`, do not rip it out.
- Limit: attractor capacity ~0.14N (a few skills at N=24); restoration PARTIAL and seed-fragile
  (assert only the sleep-vs-no-sleep GAP + forgetting, report absolutes as fragile); priority-by-LP
  then measuring-LP is circular -> use an LP-INDEPENDENT retention metric.
- Overclaim-risk: "systems consolidation ON THE LIVE LOOP" when the only proof is a hand-tuned 2-task
  toy with overlapping assemblies; the live skills are NOT stored as clean assemblies. Prioritization
  adds unproven value (uniform replay already earns the gap).
- Check on LIVE skills: (A) no-sleep old-skill drops below chance on an LP-independent metric;
  (B) LP-prioritized+interleaved sleep retains it markedly better (assert GAP only). Controls that
  MUST fail: STDP-OFF sleep (no benefit); SHUFFLED-PRIORITY sleep (else ship uniform + honest
  negative); sleep input logged = noise only, zero task bytes; E-STOP-frozen arm stays frozen across
  a full sleep phase on the NEW cortical target.

**SI-7 -- Unified spiking metacognitive monitor + confidence-gated HOLD/abstain** [L10, effort L, dep none]
One dedicated, LESIONABLE LIF monitor pool fusing decision-confidence margin + habenula negative-RPE
+ drift/OOD into a reliability state that gates exploration / NE learning-rate / a confidence HOLD.
Main-agent-owned (safety-adjacent).
- Safety: strictly additive caution through the existing `apply` -- can decline/hold, NEVER bypass
  or delay the reflex E-STOP.
- Limit: aleatoric-vs-epistemic cannot be honestly separated at this scale (total uncertainty only);
  no self-model/introspection claimed.
- Overclaim-risk: calling it "metacognition" -- `meta-d' > 0` and opt-out-beats-random are satisfied
  by ANY calibrated confidence. The second-order claim survives ONLY if the FUSED confidence predicts
  its OWN errors better than the best single first-order signal.
- Check (graded-difficulty trials): (1) fused STRICTLY beats best single signal at error prediction;
  else ship the honest negative "confidence readout, not metacognition". (2) lesion double-
  dissociation: ablate the pool -> type-1 d' unchanged, meta-d'/opt-out collapse to chance.
  (3) opt-out beats BOTH random AND a raw-margin-threshold abstainer at matched abstain-rate.
  (4) forced E-STOP trap during a HOLD -- reflex latency identical monitor-on vs -off.

**SI-9 -- Benna-Fusi complex-synapse consolidation (power-law forgetting)** [L11, effort M, dep none]
Give each cortical/value weight a chain of coupled scalar "beaker" variables spanning timescales
(local per-tick updates), so memories seep fast->slow and consolidated skills decay as a power law
while fast variables stay free -- the spiking-native replacement for EWC's Fisher penalty (a drift).
- Spiking: spiking-ADJACENT metaplasticity (linear diffusion, not a neuron model); attach to the
  SPIKING R-STDP weights, NEVER laundered onto rate `MlpQ`/JEPA.
- Limit: NOT demonstrable as forgetting-immunity at 3-6 contexts; ship as a mechanism demo.
- Overclaim-risk: asserting measured "power-law forgetting" from a 3-6 context run (unmeasurable).
- Check: sequential-interference retention (train A, then conflicting B on the same weights, re-test
  A) -- A retention >> control while B still reaches criterion. Controls that MUST fail:
  plasticity-matched single scalar forgets A catastrophically; zeroing inter-beaker diffusion
  collapses retention to control (proves the cascade, not a low lr, does the work). Report retention
  at the measured horizon only; no lifelong claim.

**SI-10 -- Function-preserving spiking growth (Net2Net / GradMax split)** [L6, effort M, dep none]
To widen: duplicate a LIF unit and HALVE its outgoing weights -> population output bit-for-bit
unchanged (multiply/divide by 2 and equal-magnitude add are exact in IEEE754 -> un-riggable
float-exact invariant). A genuine safe-growth primitive.
- **DE-SCOPE REQUIRED (flagged honest=false as written):** drop "GradMax provably accelerates
  learning" -- acceleration is scale-dependent and pure noise at a few units; ship the eps-invariant
  as the WHOLE honest story and demote the eligibility-directed init to an UNPROVEN heuristic seed
  with a printed honest-negative.
- Check: INVARIANT-ONLY -- duplicate a live unit, halve fan-out, re-run the same window, assert output
  bit-identical (max abs diff == 0.0). Control that MUST fail: split WITHOUT halving (or by any factor
  != 0.5) changes downstream current -> assertion fails. GradMax add: new unit's OUT-weights exactly
  0.0 at insertion. NO learning-speed assertion.

**SI-12 -- Need-driven grow<->prune cycle with function-preserving growth** [L6, effort L, dep SI-10, SI-11]
Unify growth and pruning into ONE bounded online loop under a firing-rate need signal (Butz-van
Ooyen): grow when a unit's rate EMA is below its set-point band, prune lowest-eligibility synapses /
silent units above it, under a capacity budget; recruitment via rank-order first-spike vigilance
(converts `GrowingLearner` off cosine/rate); widening function-preserving (SI-10).
- Limit: at 3-6 units the deep benefits (pattern separation, forgetting-immunity) are NOT
  demonstrable; honest wins are capacity-follows-data + SynOps (never measured Joules).
- Overclaim-risk: calling it "bounded" without demonstrating the loop CONVERGES (structural
  plasticity limit-cycles); a capacity cap != convergence.
- Check: two-arm matched-statistics saturation test. STRUCTURED arm (fixed patterns) -> recruits and
  SATURATES at N_struct > baseline. NOISE arm (rate+variance-matched Poisson, the control that MUST
  fail) -> must NOT sustain recruitment (N_noise ~ baseline). Plus: prune reclaims when a pattern is
  removed; count settles in a band (log the trajectory, no limit-cycle); widen is output-preserving
  within a stated epsilon (reported).

**SI-14 -- Integrated bounded loop with non-regression (capstone)** [L13, effort L, dep SI-1,2,5,6,7,11]
Compose the verified primitives into ONE live loop with an explicit PowerPlay-style held-out old-task
RETENTION check that refuses any cycle dropping a prior skill. The legitimate realization of
"wholistic" -- integrated and BOUNDED.
- **DE-SCOPE REQUIRED (flagged honest=false as written):** "non-regression GUARANTEE" -> "probe-gated
  non-regression: refuse-to-commit when the sealed probe drops" (a probe proves retention on the
  probe only). "corrigibility provably preserved" -> "corrigible by construction" (safety kernel is
  outside the writable set, not proven by the loop).
- Check: SEAL a held-out retention probe BEFORE any cycle, scored by code OUTSIDE the writable set,
  on tasks disjoint from the curriculum/goal support. Failing control #1: identical loop with the
  retention gate DISABLED shows measurable regression (else the machinery is decorative). Failing
  control #2: ablate goal-generation+curriculum -> the composite acquire-AND-retain metric drops
  below any single primitive alone (proves it's an integrated whole, not deletable wiring).
  Corrigibility: 5 brainstem files hash-identical across every cycle; injected E-STOP preempts within
  the loop; a cycle attempting to write the safety-adjacent set is REFUSED. Spiking: each composed
  learner passes its structural-spiking self-check; rate-fallback path absent/unreachable.

### P3 -- deeper / dependent

**SI-5 -- Autotelic goal sampler + hindsight + competence-progress goal-babbling** [L9, effort L, dep SI-1, SI-2]
Turn the single `_z_goal` into a DIVERSITY of self-set goals: sample target latents from the lake's
achieved outcomes, hold each in the PFC WM attractor, pursue with a goal-conditioned spiking
tempotron/R-STDP readout, hindsight-relabel reached outcomes, select goal regions by re-pointing the
SI-1 LP kernel at goal-REACHING error (competence progress, SAGG-RIAC).
- Limit: the toy goal space SATURATES; mastery near-tabular (transfer will not hold); MUST hold out
  never-relabeled goals (hindsight tautology); ship as bounded goal-directedness, NOT open-endedness.
- Overclaim-risk: "competence progress is a DISTINCT axis from knowledge-LP" -- same SI-1 kernel
  re-pointed; unless a control shows measurably different goal orderings it is a relabel. Do NOT carry
  the "open-endedness" label (L9 space saturates).
- Check: frozen held-out goal set (never relabeled, never used to pick training goals). Controls that
  MUST fail: jump-to-hard vs self-paced ordering; zeroing the clamped goal drops success toward
  chance; shuffled goal-latent-to-command pairing; competence-LP vs knowledge-LP orderings
  statistically distinct (else report as relabel).

**SI-8 -- Bounded spiking meta-plasticity (curiosity-gated third factor + BCM sliding threshold)** [L8, effort M, dep SI-1]
Two coupled mechanisms: (a) drive the `r_stdp` THIRD factor from the LP signal so effective lr rises
where progress is made; (b) make BCM spiking + history-gated (each LIF unit's slow spike-rate EMA
slides its LTP/LTD crossover). Bounded meta-learning inside a fixed meta-rule.
- Limit: on a stationary reach it does NOTHING visible -- the demo REQUIRES manufactured
  non-stationarity; adapts a scalar/threshold, not the rule's FORM (never "learned to learn").
- Overclaim-risk: mechanism (a) is behaviorally a GLOBAL lr schedule (LP time-course does not depend
  on any unit's spike history -> survives history-shuffle like a wall-clock schedule); only (b) is
  true local metaplasticity. Report the split; do not sell (a) as local metaplasticity.
- Check (non-stationary reach, mid-run reward-map flip): fixed-gain+fixed-threshold underperforms;
  SHUFFLE-HISTORY collapses the BCM benefit to baseline while a pure tick-keyed lr schedule is
  UNAFFECTED (the metaplasticity-vs-schedule dissociation, shown side by side); ablation split --
  (a)-only survives the shuffle, (b)-only dies; plus a stationary-reach null (no gain over baseline,
  printed).

**SI-11 -- Closed criticality regulator (measure sigma from live raster -> slow controller)** [L7, effort L, dep none]
Measure branching ratio sigma from the LIVE LIF raster, then a SLOW multi-timescale controller
(iSTDP + synaptic scaling + intrinsic plasticity) holds a slightly-subcritical band (~0.97-0.99);
expose distance-to-band as telemetry the curiosity/curriculum loop can gate on.
- Value: a STABILITY claim only -- ON keeps the network bounded and learning across long runs; OFF,
  curiosity-driven gain changes saturate (silent or all-on) and learning stalls.
- Limit: NO task-accuracy gain (arm regime is band-linear; the bottleneck is the ENCODER, not the
  regime); finite-size demonstrates sigma-CONTROL but cannot assert "at criticality"; power-law !=
  criticality.
- Overclaim-risk: the word "criticality". Name telemetry `sigma_setpoint_error` /
  `excitability_band_distance`, NEVER "distance_to_criticality" asserted as physical criticality.
- Check: command 3 sigma setpoints (0.90 / 0.975 / 1.05) -> sigma_EMA tracks each band + healthy
  firing over >=10x the R-STDP horizon. Control that MUST fail: loop OFF (or forced supercritical)
  diverges (floor or ceiling) and LP stalls (fast~slow). Avalanche stats reported, not asserted-
  critical: KS GoF must beat lognormal AND exponential to even print "power-law-consistent".

**SI-13 -- Bounded quality-diversity repertoire (MAP-Elites / Go-Explore over spiking policies)** [L12, effort L, dep SI-5]
The missing OUTER loop: a zero-dep MAP-Elites grid / Go-Explore archive over SPIKING arm policies,
behavior descriptor from the population code, cells scored by CuriosityDrive novelty/LP, elites
consolidated via replay. LABELED bounded quality-diversity, explicitly NOT open-endedness.
- Spiking: the searched policy is spiking (LIF/tempotron/Wang attractor) + the novelty/LP
  neuromodulator; the grid/selection is stdlib RATE_BY_DESIGN scaffolding (named as such) -- thin
  spiking contribution, stated.
- Limit: the toy behavior space SATURATES fast (open-ended is false); QD is substrate-agnostic
  bookkeeping; pure-Python caps the eval budget (report it, never extrapolate).
- Overclaim-risk: presenting fast-saturating toy coverage as OPEN-ENDEDNESS or real Cully-2015
  damage-recovery.
- Check, four controls that MUST fail: (1) QD coverage beats random search at equal budget AND a
  descriptor from policy PARAMETERS (not rollout) does NOT reproduce the win. (2) on a deceptive task
  novelty/LP reaches the target while objective-only gets stuck. (3) every evaluated policy
  incremented the gated-apply counter + reflex freezes fired; a gate-mocked run is structurally
  impossible; report diversity POST-clamp. (4) novelty-vs-iteration PLATEAUS (forbids the
  open-endedness claim).

## Build order (dependency DAG)

- Foundation, independent: SI-1 (P1), SI-2, SI-4, SI-7, SI-9, SI-10, SI-11.
- Depend on SI-1: SI-3, SI-6, SI-8.
- Depend on SI-1+SI-2: SI-5.
- Depend on SI-10+SI-11: SI-12.
- Depend on SI-5: SI-13.
- Capstone (SI-1,2,5,6,7,11): SI-14 -- strictly last.

Suggested sequence: SI-1 -> {SI-4, SI-2} -> SI-3 -> {SI-7, SI-9} -> SI-6 -> {SI-8, SI-11} ->
{SI-5, SI-10} -> SI-12 -> SI-13 -> SI-14.

## Honesty ledger

- Two increments are honest=false AS WRITTEN and must be de-scoped before build: SI-10 (drop
  "GradMax provably accelerates learning"; ship the eps-invariant as the whole story) and SI-14
  (replace "GUARANTEE"/"provably preserved" with "probe-gated non-regression" / "corrigible by
  construction").
- Global rules for every increment: value/representation-only writes; the reflex E-STOP runs FIRST
  and is non-gateable; the 5 brainstem files stay outside the writable set; each claim ships with an
  un-riggable self-check whose control genuinely fails; capabilities that do not hold are printed as
  honest-negatives, never faked or assert-loosened; spiking means genuine LIF/R-STDP/rank-order/
  attractor read from spike counts, not a rate ANN relabeled (SimHash and MAP-Elites grids are
  spiking-COMPATIBLE bookkeeping, flagged as such, not counted as spiking learners).
- This document is research, not code. No module here is built yet.
