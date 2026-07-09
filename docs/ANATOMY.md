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
| Basal forebrain / neuromodulators | partial | `neuromodulation/` — ACh, astrocyte, cortisol; dopamine in `basal_ganglia` |
| Hypothalamus | partial | HPA output only (`neuromodulation/cortisol`); no SCN clock / set-points |
| Medulla | partial | `brainstem/` reflex + loops; no autonomic vital nuclei |
| White-matter connectome | present (schematic) | `connectome.py` — typed/directed/signed/loop-aware edge list |
| Midbrain | missing | dopamine appears as a neuromodulator, not as VTA/SNc/colliculi nuclei |
| Pons | missing | only `cerebellum/cpg_oscillator`; no pontine relay |
| Olfactory pathway | partial | expansion-coding primitive exists (`research/fly_hash`); no bulb/piriform |
| Claustrum, epithalamus, reticular/ARAS | missing | low priority / hard to build honestly |

**Roughly:** 6 structures present + a schematic connectome, 4 partial, ~4 low-priority missing.

## Canonical loops (asserted closed in `connectome.py`)

thalamocortical · cortico-basal-ganglia-thalamo-cortical (CBGTC) · hippocampal trisynaptic ·
cortico-cerebellar · Papez (limbic). Plus the amygdala threat pathway, neuromodulatory broadcast,
and the descending corticospinal path. A self-check asserts the arm actuator is reachable **only**
through the gated brainstem edge — reinforcing the arm-safety envelope.

## Roadmap (next honest primitives, ranked)

1. **Hypothalamus `circadian_clock` (SCN)** — a zero-dep spiking phase oscillator that free-runs at
   intrinsic period and entrains to a light zeitgeber; gates consolidation to the rest phase and sets
   a diurnal cortisol baseline. Fair control: free-runs in constant darkness (endogenous rhythm).
2. **Neuromodulation `locus_coeruleus` (LC-noradrenaline)** — a broadcast gain node (arousal); the
   connectome's `broadcast` edge class wants it. Raphe-serotonin is the sibling.
3. **Midbrain / pons relay nuclei** — VTA/SNc as nuclei, superior colliculus (orienting), the
   cortico-ponto-cerebellar relay.

Every addition stays pure-stdlib, offline-deterministic, self-checked with a fair control, lives in
an anatomical region subpackage, and touches none of the 5 safety-envelope files.
