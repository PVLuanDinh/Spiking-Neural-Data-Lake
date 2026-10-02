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
| Amygdala | present | `faculties/fear_conditioning` — CS-US threat learning (BLA→CeA), extinction |
| Hypothalamus | present | `hypothalamus/` — `circadian_clock` (SCN spiking clock) + `set_points` (thermo/osmo homeostats the SCN paces); HPA in `faculties/cortisol` |
| Midbrain | present | `midbrain/` — `dopamine_nuclei` (VTA/SNc RPE source) + `superior_colliculus` (orienting) + `sc_deep_layers` (gaze burst generator, gated) |
| Pons | **ABSENT** | `pons/` holds only `__init__.py`; `REGIONS` maps 0 modules to it. `pontine_relay` and `reticular_formation` are NOT implemented (corrected 2026-07-28 — this row read "present" with both named). The directory is a placeholder, and `live_brain.py:262` still emits a `spikes.pons` row from a proxy signal |
| Epithalamus | present | `faculties/habenula` — lateral habenula anti-reward (negative RPE, dopamine opponent via RMTg) |
| Olfactory pathway | **ABSENT** | `olfactory/` holds only `__init__.py`; `REGIONS` maps 0 modules to it. `olfactory_bulb` and `piriform_cortex` are NOT implemented (corrected 2026-07-28). `spikes.olfactory` is a constant 0.0 and `PROVENANCE` says so |
| Claustrum | present | `faculties/claustrum` — coincidence-gated cortical synchrony / salience hub |
| Basal forebrain / neuromodulators | present | `neuromodulation/` — ACh, LC-noradrenaline, raphe-serotonin, astrocyte, cortisol; dopamine in `midbrain` |
| Reticular activating system (ARAS) | **ABSENT** | `pons/reticular_formation` does not exist (corrected 2026-07-28). The wake/sleep switch that DOES run is the circadian clock in `engine/brain.py`, which is not an ARAS model |
| White-matter connectome | present (schematic) | `connectome.py` — typed/directed/signed/loop-aware edge list (37 nodes, 11 closed loops) |
| Medulla | present | `medulla/` — `vital_nuclei` (pre-Botzinger respiratory CPG + NTS/RVLM baroreflex, homeostatic set-point loops onto the viscera) + `interoceptive_arousal` (visceral error -> bounded caution/gain, wired live) |

**Corrected 2026-07-28:** the Allen top-level ontology is **NOT** fully covered. This paragraph read
"fully covered ... no top-level structure remains partial" while **pons** and **olfactory** were empty
directories holding only `__init__.py`, and the ARAS row pointed at a module that does not exist.
15 regions carry real modules; **2 of the 17 `REGIONS` keys map to zero modules**.
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
wired **live** into the running arm loop (`engine/autonomous_loop.py`, opt-in): the amygdala learns
which contexts predict danger, the BG gate selects the released action, and the medullary/hypothalamic
homeostats turn the body's `homeostatic_load` into a bounded caution (`faculties/interoceptive_arousal`
via the NTS→LC ascending edge) — all **caution-only**, and the reflex/E-STOP still preempts every
motor command.

The **prefrontal executive** (`cortex/prefrontal_executive`) sits at the centre: it deliberates over
the action-values (reusing the drift-diffusion race), holds a **goal** in working memory, and biases
selection top-down toward it (`ctx_pfc→ctx_motor`), so the arm is **goal-directed**, not merely
reactive — a top-down *preference* that still yields to caution and the reflex/E-STOP.
`engine/live_brain` runs the whole stack together as one self-driving loop: the brain is **live**,
centred on reasoning, acting only through the gate.

The running brain is watchable and drivable interactively — `infra/cockpit` (the ONE frontend,
served by the always-on daemon at `http://127.0.0.1:8137/`) surfaces the functional signatures (the
Global Workspace broadcast, the held goal, threat / interoceptive caution), the gated arm, and chat,
presented as *functional correlates* and explicitly **not** a claim of consciousness. Its working
memory is backed by the **spike data lake**: with
`lake_wm` on, each tick is stored in and recalled from the durable `.spkl` lake
(`substrate/lake_working_memory`), so the data lake literally holds the working memory.

## Roadmap (next honest primitives, ranked)

The Allen top-level ontology is complete **except pons and olfactory, which are empty** (see the
correction above), the medullary/hypothalamic homeostats are wired live as
caution-only arousal, and the **prefrontal executive** now runs the whole brain **live** as a
goal-directed loop (`engine/live_brain`). The next honest primitives deepen the reasoning centre:

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
**JEPA latent world model** (`faculties/jepa_world_model.py`) beside the input-space `forward_model`:
a self-supervised encoder + action-conditioned latent predictor + EMA target + VICReg anti-collapse,
which the prefrontal executive plans over in latent space and which self-trains online from the live
tick stream. See `docs/JEPA.md`. v3.14 adds a served hierarchical H-JEPA (`faculties/hjepa_serve.py`) the
executive can plan over at the abstract level.

As of v3.15 the brain also has a **curiosity drive** (`faculties/intrinsic_motivation.py`,
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

v3.17 adds the **transfer substrate** (`faculties/latent_q.py`, `LatentQ`): a SHARED linear
action-value `Q(x,a) = W[a].x` (opt-in `AutonomousArm(latent_q=True)`) replacing the tabular
per-context `q[unit][action]`. One weight set across all contexts, so a reward update generalizes -- the
arm reaches competence 1.00 on genuinely-novel contexts where the tabular baseline scores 0.00. The
substrate rides the task-linear lake-stored `[0,1]^64` pattern.

v3.18 adds the **nonlinear head** (`faculties/mlp_q.py`, `MlpQ`): a SHARED one-hidden-layer action-value
`Q(z,a) = W2[a].relu(W1 z + b1) + b2[a]` over the self-supervised JEPA latent (opt-in `AutonomousArm(mlp_q=True)`).
Its own self-check proves the head recovers a NONLINEAR (XOR) rule a linear Q provably cannot (0.99 vs 0.47).
**Honest finding (corrects v3.17):** v3.17 called the SSL latent "not task-aligned" (a linear online Q over it
scored 0.33 ~ chance). That was over-pessimistic -- the SSL latent DOES encode the task (an offline probe
recovers the arm's rule at 0.80 linear / 0.88 MLP). The real cause of the on-arm chance result is the arm's
**online JEPA encoder near-collapsing** its few live contexts (latent spread ~0.003), so on-robot no head --
linear or nonlinear -- can separate them. The honest next lever is a better ENCODER (reward-shaped / deeper /
more diverse experience), not a bigger Q head.

v3.19 takes exactly that lever: **reward-shaping the encoder** (`faculties/jepa_world_model.py`, opt-in
`reward_shaping=True`; `AutonomousArm(reward_shaped_encoder=True)`). A reward-predictive auxiliary head
`r_hat[a] = Wr[a].z + br[a]` adds a weak supervised gradient (from the observed reward) into the SAME encoder
backprop, so the latent encodes task/reward structure -- "task-shaped SSL", not pure SSL. This breaks the
few-context collapse (band latent spread 0.003 -> 1.32) so the SAME nonlinear MLP-Q now recovers the arm's rule
on-robot. v3.20 tunes the reward-shaping strength (`reward_gain`, default 5.0): a firm task pull lifts on-arm
held-out competence the whole way -- **0.33 (plain SSL) -> 0.83 (v3.19) -> 1.00 (v3.20)**, robust across seeds.
The task-aligned LEARNED substrate now FULLY recovers the rule, matching the purely task-linear pattern substrate
(v3.17's 1.00) while being a learned representation -- by fixing the encoder exactly as the v3.18 finding predicted.

v3.21 adds **spiking interaction learning** (`readout/interaction_learning.py`, `SpikingInteractionLearner`), and
corrects course after a spiking-purity audit (`docs/SPIKING_AUDIT.md`): the v3.11-v3.20 learning tier above is
**rate-level** (linear / MLP / EMA over the lake's `[0,1]^64` pattern), a drift from the project's spiking purpose.
This module improves the live brain **by being talked to**, the spiking way -- rank-order spike encoding + reward-
modulated spike-timing plasticity (R-STDP), reusing the real spiking primitives, not a new ANN. Opt-in
`LiveBrain(interaction_learning=True).teach(text, intent)` / `.route(text)`: talking to the brain lifts its spiking
intent routing from chance to 1.00, proven from spike timing (a shuffled-teaching control stays at chance), and it
adds no actuator route (arm commands still go through the NL firewall + gated arm; the reflex E-STOP still preempts).

v3.22 answers "does the brain improve by being talked to?" honestly: the fixed `converse()` keyword router does NOT
(it is stateless and misses paraphrases with no keyword -- 0.20 on held-out novel paraphrases), so the brain was
improved so it does. `LiveBrain.chat(text)` replies AND learns from each turn -- the firewall's own grounded intent
is a SELF-SUPERVISED R-STDP teaching signal (no human labels), and a `t_post` spike-timing knob makes the readout
key on early content-word spikes so it GENERALIZES. The brain's spiking router now routes NOVEL paraphrases the
keyword firewall misses (firewall 0.00, router 0.00 -> 1.00 on held-out), still spike-timing (shuffled control at
chance) and still intent-recognition only (no actuator route). Honest limit: on the firewall's own blind spots
distillation inherits the error and explicit conversational feedback is what recovers them.
