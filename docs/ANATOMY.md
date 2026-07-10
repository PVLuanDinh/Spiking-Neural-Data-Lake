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

1. **Deeper deliberation** — give the executive multi-step / model-based reasoning (planning or
   lookahead over the cerebellar forward model) rather than single-shot evidence commitment, and a
   richer working memory (multiple simultaneous goals / sub-goals).
2. **Deep sub-nuclei** — the Allen ontology goes below the top level (hypothalamic nuclei, amygdalar
   subnuclei, thalamic relay/association nuclei). Add the load-bearing ones for existing loops.
3. **Spinal cord as its own region** — split the ventral-horn CPG / stretch-reflex out of `brainstem/`
   into a distinct caudal region, completing the neuraxis below the medulla.

Every addition stays pure-stdlib, offline-deterministic, self-checked with a fair control, lives in
an anatomical region subpackage, and touches none of the 5 safety-envelope files.
