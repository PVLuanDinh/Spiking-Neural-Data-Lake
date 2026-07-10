# Brain-anatomy completeness

The library is organized by brain anatomy (see `snn_data_lake.REGIONS` / `brain_atlas.py` for the
region → module map, and `connectome.py` for the region → region wiring). This file tracks how
complete that anatomy is against the Allen Brain Atlas top-level structural ontology, and what
remains. It is a **mesoscale functional** model — each region models a structure's *computation*,
not its cell-level microcircuitry or phenomenology.

## Coverage (Allen top-level structures)

| Canonical structure | Status | In the project |
|---|---|---|
| Cerebral cortex | present (function) | `cortex/` — encoders, 5 neuron models, microcircuit, readouts, language, reasoning (`brain_reasoning`, `vsa_reasoning`) + `prefrontal_executive` (working-memory goal executive) + `live_brain` (the whole brain, running) |
| Hippocampal formation | present | `hippocampus/` — DG separation → CA3 completion, grid cells, replay, sleep |
| Cerebellum | present | `cerebellum/` — forward model, Kalman, spiking PID, timing, planner |
| Basal ganglia | present | `basal_ganglia/` — RPE learning + evidence decision + Go/NoGo `action_gating` |
| Thalamus | present | `thalamus/` — `global_workspace` (GWT) + `reticular_gate` (TRN attentional gate) |
| Amygdala | present | `amygdala/fear_conditioning` — CS-US threat learning (BLA→CeA), extinction |
| Hypothalamus | present | `hypothalamus/` — `circadian_clock` (SCN spiking clock) + `set_points` (thermo/osmo homeostats the SCN paces); HPA in `neuromodulation/cortisol` |
| Midbrain | present | `midbrain/` — `dopamine_nuclei` (VTA/SNc RPE source) + `superior_colliculus` (orienting) + `sc_deep_layers` (gaze burst generator, gated) |
| Pons | present | `pons/` — `pontine_relay` (cortico-ponto-cerebellar) + `reticular_formation` (ARAS arousal) |
| Epithalamus | present | `epithalamus/habenula` — lateral habenula anti-reward (negative RPE, dopamine opponent via RMTg) |
| Olfactory pathway | present | `olfactory/` — `olfactory_bulb` (decorrelation) + `piriform_cortex` (association / completion) |
| Claustrum | present | `claustrum/claustrum` — coincidence-gated cortical synchrony / salience hub |
| Basal forebrain / neuromodulators | present | `neuromodulation/` — ACh, LC-noradrenaline, raphe-serotonin, astrocyte, cortisol; dopamine in `midbrain` |
| Reticular activating system (ARAS) | present | `pons/reticular_formation` — bistable sleep-wake arousal switch |
| White-matter connectome | present (schematic) | `connectome.py` — typed/directed/signed/loop-aware edge list (37 nodes, 11 closed loops) |
| Medulla | present | `medulla/` — `vital_nuclei` (pre-Botzinger respiratory CPG + NTS/RVLM baroreflex, homeostatic set-point loops onto the viscera) + `interoceptive_arousal` (visceral error -> bounded caution/gain, wired live) |

**Roughly:** the Allen top-level structural ontology is now fully covered — 15 anatomical regions
present (each a functional model) plus a schematic connectome; no top-level structure remains partial.
The medulla's autonomic vital nuclei (the last partial) are now a real homeostatic pattern generator.

## Canonical loops (asserted closed in `connectome.py`)

thalamocortical · cortico-basal-ganglia-thalamo-cortical (CBGTC) · hippocampal trisynaptic ·
cortico-cerebellar (via the pontine relay) · Papez (limbic) · tecto-thalamo-cortical orienting ·
habenula reward-opponent · olfactory bulb↔piriform · claustro-cortical · medullary respiratory ·
baroreflex (both autonomic homeostatic loops onto the viscera). Plus the amygdala threat
pathway, the ascending neuromodulatory broadcast (dopamine, noradrenaline/LC, serotonin/raphe, ACh,
ARAS), and the descending corticospinal path. A self-check asserts the arm actuator is reachable
**only** through the gated brainstem edge — reinforcing the arm-safety envelope.

The amygdala threat detector, the basal-ganglia Go/NoGo gate, and the interoceptive-arousal drive are
wired **live** into the running arm loop (`brainstem/autonomous_loop.py`, opt-in): the amygdala learns
which contexts predict danger, the BG gate selects the released action, and the medullary/hypothalamic
homeostats turn the body's `homeostatic_load` into a bounded caution (`medulla/interoceptive_arousal`
via the NTS→LC ascending edge) — all **caution-only**, and the reflex/E-STOP still preempts every
motor command.

The **prefrontal executive** (`cortex/prefrontal_executive`) sits at the centre: it deliberates over
the action-values (reusing the drift-diffusion race), holds a **goal** in working memory, and biases
selection top-down toward it (`ctx_pfc→ctx_motor`), so the arm is **goal-directed**, not merely
reactive — a top-down *preference* that still yields to caution and the reflex/E-STOP.
`cortex/live_brain` runs the whole stack together as one self-driving loop: the brain is **live**,
centred on reasoning, acting only through the gate.

The running brain is drivable interactively — `substrate/live_brain_tab` adds a "Live Brain" tab to the
control panel that surfaces the functional signatures (the Global Workspace broadcast, the held goal,
threat / interoceptive caution) and the gated arm, presented as *functional correlates* and explicitly
**not** a claim of consciousness. Its working memory is backed by the **spike data lake**: with
`lake_wm` on, each tick is stored in and recalled from the durable `.spkl` lake
(`substrate/lake_working_memory`), so the data lake literally holds the working memory.

## Roadmap (next honest primitives, ranked)

The Allen top-level ontology is complete, the medullary/hypothalamic homeostats are wired live as
caution-only arousal, and the **prefrontal executive** now runs the whole brain **live** as a
goal-directed loop (`cortex/live_brain`). The next honest primitives deepen the reasoning centre:

1. ~~**Deeper deliberation** — give the executive multi-step / model-based reasoning (planning or
   lookahead over the cerebellar forward model) rather than single-shot evidence commitment, and a
   richer working memory (multiple simultaneous goals / sub-goals).~~ **Done (v3.10):**
   `PrefrontalExecutive.plan()` deliberates over `horizon`-step rollouts of an injected forward model
   (reactive `h=1` takes a trap, planned `h>=2` rejects it); working memory is now a bounded goal
   stack (`push_subgoal` / `.goals`, sub-goal pops back to its parent). Surfaced on the live brain as
   `LiveBrain.deliberate_ahead` — a deliberative faculty that biases a goal and never drives the arm.
2. **Deep sub-nuclei** — the Allen ontology goes below the top level (hypothalamic nuclei, amygdalar
   subnuclei, thalamic relay/association nuclei). Add the load-bearing ones for existing loops.
3. **Spinal cord as its own region** — split the ventral-horn CPG / stretch-reflex out of `brainstem/`
   into a distinct caudal region, completing the neuraxis below the medulla.

Every addition stays pure-stdlib, offline-deterministic, self-checked with a fair control, lives in
an anatomical region subpackage, and touches none of the 5 safety-envelope files.

## Learning

Beyond the anatomy, the brain also carries **learning mechanisms**. As of v3.11 the cerebellum holds a
**JEPA latent world model** (`cerebellum/jepa_world_model.py`) beside the input-space `forward_model`:
a self-supervised encoder + action-conditioned latent predictor + EMA target + VICReg anti-collapse,
which the prefrontal executive plans over in latent space and which self-trains online from the live
tick stream. See `docs/JEPA.md`. v3.14 adds a served hierarchical H-JEPA (`cerebellum/hjepa_serve.py`) the
executive can plan over at the abstract level.

As of v3.15 the brain also has a **curiosity drive** (`neuromodulation/intrinsic_motivation.py`,
`CuriosityDrive`) -- a **learning-progress** intrinsic-motivation signal (Oudeyer; Schmidhuber): the JEPA
world-model's prediction error becomes an intrinsic reward that rewards *learnable* novelty (not noise --
it rejects the "noisy-TV" trap), so the arm explores and learns SELF-DIRECTED, even without an external
reward (opt-in `AutonomousArm(curiosity=True)`). This is the engine of self-directed, cumulative learning
-- one primitive, not a claim of general developmental intelligence; the reflex/E-STOP + caution still
preempt every command.

v3.16 adds a **self-paced curriculum** (`neuromodulation/curriculum.py`, `Curriculum`): an ordered
sequence of stages of increasing difficulty that advances AUTONOMOUSLY when the agent's own learning
progress on a stage dries up (mastery, not a clock -- Elman "starting small"; Oudeyer). Wired as
`AutonomousArm.develop(stages, curriculum)`: the arm graduates itself through progressively harder stages
driven by its own curiosity, learning through the same gated `step()`. Curiosity + curriculum together are
the self-directed, cumulative-learning loop -- master easy, unlock harder, at the frontier of competence.

v3.17 adds the **transfer substrate** (`basal_ganglia/latent_q.py`, `LatentQ`): a SHARED linear
action-value `Q(x,a) = W[a].x` (opt-in `AutonomousArm(latent_q=True)`) replacing the tabular
per-context `q[unit][action]`. One weight set across all contexts, so a reward update generalizes -- the
arm reaches competence 1.00 on genuinely-novel contexts where the tabular baseline scores 0.00. The
substrate rides the task-linear lake-stored `[0,1]^64` pattern.

v3.18 adds the **nonlinear head** (`basal_ganglia/mlp_q.py`, `MlpQ`): a SHARED one-hidden-layer action-value
`Q(z,a) = W2[a].relu(W1 z + b1) + b2[a]` over the self-supervised JEPA latent (opt-in `AutonomousArm(mlp_q=True)`).
Its own self-check proves the head recovers a NONLINEAR (XOR) rule a linear Q provably cannot (0.99 vs 0.47).
**Honest finding (corrects v3.17):** v3.17 called the SSL latent "not task-aligned" (a linear online Q over it
scored 0.33 ~ chance). That was over-pessimistic -- the SSL latent DOES encode the task (an offline probe
recovers the arm's rule at 0.80 linear / 0.88 MLP). The real cause of the on-arm chance result is the arm's
**online JEPA encoder near-collapsing** its few live contexts (latent spread ~0.003), so on-robot no head --
linear or nonlinear -- can separate them. The honest next lever is a better ENCODER (reward-shaped / deeper /
more diverse experience), not a bigger Q head.

v3.19 takes exactly that lever: **reward-shaping the encoder** (`cerebellum/jepa_world_model.py`, opt-in
`reward_shaping=True`; `AutonomousArm(reward_shaped_encoder=True)`). A reward-predictive auxiliary head
`r_hat[a] = Wr[a].z + br[a]` adds a weak supervised gradient (from the observed reward) into the SAME encoder
backprop, so the latent encodes task/reward structure -- "task-shaped SSL", not pure SSL. This breaks the
few-context collapse (band latent spread 0.003 -> 1.32) so the SAME nonlinear MLP-Q now recovers the arm's rule
on-robot. v3.20 tunes the reward-shaping strength (`reward_gain`, default 5.0): a firm task pull lifts on-arm
held-out competence the whole way -- **0.33 (plain SSL) -> 0.83 (v3.19) -> 1.00 (v3.20)**, robust across seeds.
The task-aligned LEARNED substrate now FULLY recovers the rule, matching the purely task-linear pattern substrate
(v3.17's 1.00) while being a learned representation -- by fixing the encoder exactly as the v3.18 finding predicted.

v3.21 adds **spiking interaction learning** (`cortex/interaction_learning.py`, `SpikingInteractionLearner`), and
corrects course after a spiking-purity audit (`docs/SPIKING_AUDIT.md`): the v3.11-v3.20 learning tier above is
**rate-level** (linear / MLP / EMA over the lake's `[0,1]^64` pattern), a drift from the project's spiking purpose.
This module improves the live brain **by being talked to**, the spiking way -- rank-order spike encoding + reward-
modulated spike-timing plasticity (R-STDP), reusing the real spiking primitives, not a new ANN. Opt-in
`LiveBrain(interaction_learning=True).teach(text, intent)` / `.route(text)`: talking to the brain lifts its spiking
intent routing from chance to 1.00, proven from spike timing (a shuffled-teaching control stays at chance), and it
adds no actuator route (arm commands still go through the NL firewall + gated arm; the reflex E-STOP still preempts).
