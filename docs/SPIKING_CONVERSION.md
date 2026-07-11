# Spiking-conversion program -- turn the entire neural atlas spiking

Goal (user directive): make **every brain-region module genuinely spiking** and the **live loop spike-based
end-to-end**, verified. Run as staged, self-checked slices. The ~30 `substrate/` modules are the data-lake
plumbing (sketches, telemetry, IO) -- **not neurons**, excluded. Rate-by-design neural-mass models get a
spiking ANALOG, not a forced fake.

Baseline inventory (2026-07-10, parallel classification of 103 brain-region modules):
**SPIKING 34 | RATE 47 (convert) | RATE_BY_DESIGN 22 (spiking analog).**
After slice 1 (v3.23): **SPIKING 39 | RATE 42 | RATE_BY_DESIGN 22.**

## Converted to spiking (progress)

- **Slice 1 (v3.23):** `cerebellum/spiking_pid` (P/I/D as LIF populations; integral = running spike count),
  `midbrain/superior_colliculus` (LIF first-spike WTA race; latency = first-spike time),
  `midbrain/sc_deep_layers` (LIF burst pool + gain-1 line-attractor spiking integrator; displacement =
  accumulated spikes), `cortex/cue_integration` (spiking probabilistic population code; fusion = spike
  addition), `neuromodulation/locus_coeruleus` (LIF NE pool; NE = population spike rate, phasic burst -> gain).
  All API-preserved, un-riggable spiking self-checks, live loop untouched. Honest cost: spiking read-outs add
  quantization/Poisson noise (SC latency integer-quantized; SC-deep ~4.4%; cue-integration approaches MLE within
  tolerance) -- reported, not faked.

## Already SPIKING (34) -- no work

lif_neuron, dendritic_neuron, izhikevich_neuron, hodgkin_huxley, alif_neuron, adaptive_spiking,
stochastic_resonance, rank_order_coding, spike_preprocessing, spike_slicer, lt_gate, is_snn_forward,
neural_sampling, trained_brain, tempotron, paradigm_b_matcher, paradigm_b_engine, spiking_lm_serve,
interaction_learning, htm_sequence_memory, sleep_replay, r_stdp, action_gating, synfire_chain, reflex,
closed_loop, signal_loop, dopamine_nuclei, vital_nuclei, habenula, claustrum, fear_conditioning,
inhibitory_stdp, short_term_plasticity.

## RATE -- to convert (47), with the spiking design

| module | spiking conversion |
|---|---|
| cortex/population_encoding | RBF units -> LIF/Poisson neurons firing at Gaussian rate; decode from windowed spike counts |
| cortex/population_vector | cosine-tuned cells emit spike trains at rate r_i; vector-sum decode from spike counts |
| cortex/multimodal_encoder | TTFS/latency-encode each band into per-feature spike trains (not a continuous [0,1]^64 vector) |
| cortex/spike_pooling | spiking WTA: first-spike-wins lateral inhibition per window |
| cortex/lca_sparse_coding | each atom a LIF neuron; threshold spikes drive inhibition; coeff = spike rate (S-LCA) |
| cortex/predictive_coding | causes/errors as LIF populations; error spike-rate carries mismatch; spiking recurrence settles |
| cortex/restricted_boltzmann | stochastic LIF sampling; event-driven STDP instead of CD gradient (Neftci 2014) |
| cortex/cue_integration | Poisson population per cue, gain = reliability; sum populations (spiking PPC, Ma 2006) |
| cortex/snn_classifier | LIF output neurons (membrane/threshold/reset); STDP / surrogate-gradient on spike times |
| cortex/learned_matcher | tempotron/STDP spike-time classifier instead of the delta-rule rate readout |
| cortex/gesture_recognition | latency/rank-encode the distance stream; spiking sequence/tempotron detector end-to-end |
| cortex/nl_interface | rank-order spike-encode words -> R-STDP intent classifier (keep the fixed allow-list) |
| cortex/brain_reasoning | read class by spiking WTA over classifier output spikes; confidence from spike-count margin |
| cortex/vsa_reasoning | phasor/spike-polarity hypervectors; bind = phase-multiply, bundle = spike superposition |
| hippocampus/hippocampus | DG spiking expansion + lateral-inhibition WTA; CA3 recurrent spiking attractor completion |
| hippocampus/grid_cells | 3 velocity-controlled spiking oscillators; grid firing from spike-phase interference |
| hippocampus/frequency_memory | LIF neurons; firing-rate homeostat gates STDP on/off (FIP/FID) to form assemblies |
| hippocampus/experience_replay | three-factor R-STDP actor; replay = spontaneous spike-sequence reactivation |
| hippocampus/growing_learner | spiking prototype/RBF units matched by spike coincidence; recruit-on-weak-response; Hebbian |
| hippocampus/legendre_memory | realize the A/B delay state-space in spiking neurons via NEF (spiking LMU) |
| basal_ganglia/evidence_accumulation | Wang-2002 spiking attractor: LIF pools integrate to threshold via recurrent E + I |
| basal_ganglia/valence_stdp | LIF valence neuron; pre/post spike-timing eligibility gated by dopamine (three-factor) |
| basal_ganglia/latent_q | spiking linear readout over latent spikes, reward-modulated (three-factor) plasticity |
| basal_ganglia/mlp_q | spiking MLP: LIF hidden units, surrogate-gradient / reward-STDP; first-spike action-value |
| cerebellum/forward_model | granule->Purkinje spiking forward model; climbing-fiber error drives spike-timing plasticity |
| cerebellum/kalman_filter | NEF/PPC spiking net carrying state + covariance in population activity |
| cerebellum/spiking_pid | **named spiking but rate** -> genuine LIF integral/derivative populations |
| cerebellum/jepa_world_model | spiking JEPA: LIF encoder/predictor (surrogate grad); slow spiking EMA target; homeostatic VICReg |
| cerebellum/hjepa_serve | serve a spiking feedforward net: LIF layers, rate/first-spike latent readout |
| brainstem/autonomous_loop | spiking WTA action selection + dopamine-modulated STDP on action-value synapses (SAFETY: mine) |
| thalamus/global_workspace | LIF specialist pools, recurrent self-excitation + global lateral inhibition; ignition = spike threshold |
| hypothalamus/set_points | comparator as opponent spiking populations; antagonistic effector spike-rate difference drives plant |
| midbrain/superior_colliculus | LIF per location + lateral inhibition; first-to-spike wins, spike time = latency |
| midbrain/sc_deep_layers | spiking EBN burst pool + recurrent line-attractor spiking integrator -> eye displacement |
| pons/pontine_relay | LIF granule neurons on sparse random weights; k-WTA lateral inhibition; active = spiking in window |
| medulla/interoceptive_arousal | LC-like LIF pool driven by aggregated error current; caution = normalized firing rate |
| olfactory/piriform_cortex | spiking attractor: LIF, Hebbian recurrent synapses, k-WTA; completion via recurrent settling |
| neuromodulation/acetylcholine | spiking attractor (LIF) + STDP writes; ACh spike-rate gates STDP + recurrent gain |
| neuromodulation/cortisol | slow spiking neuromodulatory pool integrating aversive spikes; rate scales threshold + STDP |
| neuromodulation/locus_coeruleus | LC spiking neurons bursting on salient events; NE spike-rate scales downstream gain |
| neuromodulation/raphe_serotonin | tonic raphe spike-rate sets a spiking accumulator's leak; competing spiking integrators |
| neuromodulation/intrinsic_motivation | two spiking populations, different time constants; firing-rate difference = learning-progress |
| neuromodulation/curriculum | LIF accumulator that spikes to advance a stage when sustained low-progress crosses threshold |
| plasticity/bcm_plasticity | spiking triplet-STDP (Pfister-Gerstner); sliding threshold tracks postsynaptic spike rate |
| plasticity/homeostatic_scaling | wrap a LIF neuron; windowed spike count scales weights toward target spike-rate |
| plasticity/intrinsic_plasticity | adapt a spiking neuron's threshold/gain from its spike-rate distribution toward the target |
| plasticity/developmental_pruning | LIF readout over input spikes; surrogate-gradient; prune synapses by weight magnitude |

## RATE_BY_DESIGN (22) -- spiking analog, not a forced rewrite

neuronal_avalanches (critical-branching stats), theta_gamma_code (phase-slot buffer), divisive_normalization
(canonical rate gain-control), wilson_cowan (E/I mean-rate mass), language_brain (facade), prefrontal_executive
(LCA drift-diffusion by design), live_brain (integration facade), ring_attractor (continuous-attractor bump),
kuramoto_sync (mean-field phase model), wavefront_planner (BFS path algorithm), reticular_gate (divisive-norm),
circadian_clock (phase-oscillator), reticular_formation (Wilson-Cowan flip-flop), olfactory_bulb
(divisive-normalization whitening), astrocyte_modulation (glia are non-spiking by biology), cpg_oscillator
(Matsuoka rate oscillator), + the safety envelope / IO / config (serial_arm, arm_sim, arm_config, interpreter,
live_arm, arm_bridge -- physics/byte-IO/config, correctly not neural and NEVER changed).

## Progress

- **Slice 1 (v3.23, in progress):** convert 5 self-contained non-safety RATE modules ->
  `cerebellum/spiking_pid`, `midbrain/superior_colliculus`, `midbrain/sc_deep_layers`,
  `cortex/cue_integration`, `neuromodulation/locus_coeruleus`. Each in-place to genuine LIF/spike-timing with an
  un-riggable spiking self-check; live-loop wiring untouched (no ripple).
- Remaining RATE after slice 1: 42. Live-loop-wired conversions (global_workspace, growing_learner,
  multimodal_encoder, the value substrate, autonomous_loop selection) are later, safety-adjacent slices done by
  the main agent.
