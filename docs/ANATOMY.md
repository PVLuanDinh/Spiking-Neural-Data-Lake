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
| Hypothalamus | present | `hypothalamus/circadian_clock` — SCN spiking clock (free-run + light entrainment); HPA in `neuromodulation/cortisol` |
| Midbrain | present | `midbrain/` — `dopamine_nuclei` (VTA/SNc RPE source) + `superior_colliculus` (orienting) |
| Pons | present | `pons/` — `pontine_relay` (cortico-ponto-cerebellar) + `reticular_formation` (ARAS arousal) |
| Epithalamus | present | `epithalamus/habenula` — lateral habenula anti-reward (negative RPE, dopamine opponent via RMTg) |
| Olfactory pathway | present | `olfactory/` — `olfactory_bulb` (decorrelation) + `piriform_cortex` (association / completion) |
| Claustrum | present | `claustrum/claustrum` — coincidence-gated cortical synchrony / salience hub |
| Basal forebrain / neuromodulators | present | `neuromodulation/` — ACh, LC-noradrenaline, raphe-serotonin, astrocyte, cortisol; dopamine in `midbrain` |
| Reticular activating system (ARAS) | present | `pons/reticular_formation` — bistable sleep-wake arousal switch |
| White-matter connectome | present (schematic) | `connectome.py` — typed/directed/signed/loop-aware edge list (33 nodes, 9 closed loops) |
| Medulla | partial | `brainstem/` reflex + loops; no autonomic vital nuclei (respiratory / cardiac) |

**Roughly:** the Allen top-level structural ontology is now covered — 14 anatomical regions present
(each a functional model) plus a schematic connectome; only the medullary autonomic vital nuclei
remain partial.

## Canonical loops (asserted closed in `connectome.py`)

thalamocortical · cortico-basal-ganglia-thalamo-cortical (CBGTC) · hippocampal trisynaptic ·
cortico-cerebellar (via the pontine relay) · Papez (limbic) · tecto-thalamo-cortical orienting ·
habenula reward-opponent · olfactory bulb↔piriform · claustro-cortical. Plus the amygdala threat
pathway, the ascending neuromodulatory broadcast (dopamine, noradrenaline/LC, serotonin/raphe, ACh,
ARAS), and the descending corticospinal path. A self-check asserts the arm actuator is reachable
**only** through the gated brainstem edge — reinforcing the arm-safety envelope.

The amygdala threat detector and the basal-ganglia Go/NoGo gate are also wired **live** into the
running arm loop (`brainstem/autonomous_loop.py`, opt-in): the amygdala learns which contexts predict
danger and the BG gate selects the released action — both **caution-only**, and the reflex/E-STOP
still preempts every motor command.

## Roadmap (next honest primitives, ranked)

1. **Medulla autonomic nuclei** — a spiking respiratory / cardiac pattern generator (pre-Bötzinger)
   with homeostatic set-points; the last partial Allen top-level structure.
2. **Hypothalamic set-points beyond the clock** — thermo / osmo / energy regulators that the SCN paces.
3. **Superior-colliculus deep motor layers** — turn the orienting map into a gaze command *through*
   the gated brainstem path (never a new route to the actuator).

Every addition stays pure-stdlib, offline-deterministic, self-checked with a fair control, lives in
an anatomical region subpackage, and touches none of the 5 safety-envelope files.
