# Brain-anatomy completeness

The library is organized by brain anatomy (see `snn_data_lake.REGIONS` / `brain_atlas.py` for the
region → module map, and `connectome.py` for the region → region wiring). This file tracks how
complete that anatomy is against the Allen Brain Atlas top-level structural ontology, and what
remains. It is a **mesoscale functional** model — each region models a structure's *computation*,
not its cell-level microcircuitry or phenomenology.

## Coverage (Allen top-level structures)

| Canonical structure | Status | In the project |
|---|---|---|
| Cerebral cortex | present (function) | `cortex/` — encoders, 5 neuron models, microcircuit dynamics, readouts, language |
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
| Medulla | present | `medulla/vital_nuclei` — pre-Botzinger respiratory CPG + NTS/RVLM baroreflex, each a homeostatic set-point loop onto the viscera (autonomic effectors, never the arm) |

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

The amygdala threat detector and the basal-ganglia Go/NoGo gate are also wired **live** into the
running arm loop (`brainstem/autonomous_loop.py`, opt-in): the amygdala learns which contexts predict
danger and the BG gate selects the released action — both **caution-only**, and the reflex/E-STOP
still preempts every motor command.

## Roadmap (next honest primitives, ranked)

The Allen **top-level** structural ontology is now complete (regions *and* connectome). The next
honest primitives move from breadth to depth and live integration:

1. **Autonomic integration (live, caution-only)** — wire the medullary vital nuclei + hypothalamic
   set-points into the running loop the way the amygdala threat detector and BG Go/NoGo gate already
   are: visceral/homeostatic state modulates arousal and gain, never the gated arm path.
2. **Deep sub-nuclei** — the Allen ontology goes below the top level (hypothalamic nuclei, amygdalar
   subnuclei, thalamic relay/association nuclei). Add the load-bearing ones for existing loops.
3. **Spinal cord as its own region** — split the ventral-horn CPG / stretch-reflex out of `brainstem/`
   into a distinct caudal region, completing the neuraxis below the medulla.

Every addition stays pure-stdlib, offline-deterministic, self-checked with a fair control, lives in
an anatomical region subpackage, and touches none of the 5 safety-envelope files.
