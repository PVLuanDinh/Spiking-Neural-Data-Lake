# Spiking-conversion program -- turn the entire neural atlas spiking

Goal (user directive): make **every brain-region module genuinely spiking** and the **live loop spike-based
end-to-end**, verified. Run as staged, self-checked slices. The ~30 `substrate/` modules are the data-lake
plumbing (sketches, telemetry, IO) -- **not neurons**, excluded. Rate-by-design neural-mass models get a
spiking ANALOG, not a forced fake.

Baseline inventory (2026-07-10, parallel classification of 103 brain-region modules):
**SPIKING 34 | RATE 47 (convert) | RATE_BY_DESIGN 22 (spiking analog).**
After slice 1 (v3.23): **SPIKING 39 | RATE 42 | RATE_BY_DESIGN 22.**
After slice 2 (v3.24): **SPIKING 44 | RATE 38 | RATE_BY_DESIGN 21.**
After slice 3 (v3.25): **SPIKING 48 | RATE 36 | RATE_BY_DESIGN 19.**
After slice 4 (v3.26): **SPIKING 49** (new module `cerebellum/spiking_jepa`; rate `jepa_world_model` kept as the fair baseline).
After slice 5 (v3.29): **SPIKING 50** (new module `cortex/pc_cortex_serve`, the DEEP spiking predictive-coding cortex server; the 2-level spiking seed `cortex/predictive_coding` kept as the small baseline).

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

## Rational-thinking focus (slice 2+) + the "rate-by-design is convertible" method

Per directive, the program now prioritises the modules that contribute to **rational thinking** -- decision,
working memory, reasoning, inference, attention, memory-for-thought -- and ignores the rest (motor: gaze /
PID / CPG / path-planning; autonomic / homeostatic; peripheral sensory encoding; pure spatial: grid / ring;
safety IO). Those ignored modules can be converted later or left as rate; they are not on the thinking path.

**RATE_BY_DESIGN is NOT an exemption from spiking.** Each canonical "rate" model has a well-known SPIKING
network implementation; the analog is genuine spiking, not a fake:
- drift-diffusion / LCA decision -> **Wang-2002 spiking decision attractor** (LIF pools, recurrent excitation +
  shared inhibition, integrate-to-sustained-firing; chronometric RT = time-to-threshold).
- persistent-activity working memory -> a **bistable spiking attractor** (recurrent excitation > leak keeps a
  pool firing across an input-free delay).
- WTA / argmax -> **LIF lateral inhibition** (first-to-sustained-spike wins; margin = spike-count gap).
- E/I neural mass (Wilson-Cowan) -> an actual **spiking E/I network** whose population rates match the mass model.
- divisive normalization -> **spiking normalization** (inhibitory-pool feedback gain control, Loihi-style).
- continuous attractor (ring) -> a **spiking ring attractor** (recurrent bump of spiking neurons).
So the rational-thinking RATE_BY_DESIGN modules (prefrontal_executive, reticular_gate, theta_gamma_code,
divisive_normalization, wilson_cowan) are convertible and will be redesigned into spikes, starting with the
prefrontal executive.

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
- **Slice 2 (v3.24) -- the RATIONAL-THINKING core (5 modules):** `cortex/prefrontal_executive` (RATE_BY_DESIGN ->
  spiking: Wang-2002 spiking WTA decision attractor + persistent-activity bistable spiking working memory; the
  flagship rate-by-design redesign; still passes autonomous_loop sec12 + live_brain + E-STOP), `cortex/vsa_reasoning`
  (spike-polarity hypervectors; bind=spike-XOR, bundle=superposition-WTA, sim=coincidence), `cortex/predictive_coding`
  (signed LIF error populations; inference = spiking settling; Hebbian-on-spikes), `cortex/restricted_boltzmann`
  (stochastic LIF sampling + event-driven CD/STDP), `cortex/brain_reasoning` (LIF WTA decision + spike-margin
  confidence). All API-preserved, un-riggable spiking self-checks. Honest gaps reported (pfc WM metastable/drive-gated
  release; spike-quantized inference approximate; WTA ties can differ from argmax). Remaining RATE after slice 2: 38.
- **Slice 3 (v3.25) -- more rational-thinking (4 modules):** `basal_ganglia/evidence_accumulation` (RATE ->
  Wang-2002 spiking decision attractor: LIF pools + shared inhibition; psychometric/chronometric/speed-accuracy
  hold), `thalamus/global_workspace` (RATE -> spiking Dehaene-Changeux IGNITION: LIF pools + global lateral
  inhibition + self-excitation; all-or-none ignition; LIVE-LOOP-WIRED -- verified autonomous_loop sec8 broadcast
  contract + E-STOP + live_brain), `thalamus/reticular_gate` (RATE_BY_DESIGN -> spiking TRN attentional gate:
  LIF relay + inhibitory pool = spiking divisive normalization), `cortex/theta_gamma_code` (RATE_BY_DESIGN ->
  spiking Lisman-Idiart phase code: LIF populations fire at item-specific gamma phases; capacity = theta/gamma
  ratio ~7). All API-preserved, un-riggable spiking self-checks; honest gaps reported (attractor RT units differ;
  global-workspace N-way exact-tie metastable; theta-gamma is a reduced discrete-step oscillator). Remaining
  RATE after slice 3: 36.
- **Slice 4 (v3.26) -- SpikingJEPA (NEW module, honest study):** `cerebellum/spiking_jepa.py` = a genuinely-spiking
  (LIF backbone, {0,1} signals, latent=integer counts), genuinely-LOCAL (three-factor rule, no backprop -- freeze-top
  proof) JEPA world model synthesizing LPL + event-stream + RPL. It genuinely learns action-conditioned latent
  DYNAMICS (predict beats identity, time-shuffle destroys it). HONEST NEGATIVES (printed, not asserted, not hidden):
  it does NOT beat a random SNN encoder at representation, and no-collapse can't be shown (a random sparse LIF
  backbone keeps rank even with anti-collapse off) -- matching the research prediction. The rate `jepa_world_model`
  stays the fair baseline (NOT converted in place). Atlas SPIKING 49. Separately, a GPU + disk-streaming TIER
  optimization landed (streaming_hub + proximity_index out-of-core in stdlib core; GPU chip resident-caching + mnist
  DataLoader + cu128 doc in the torch tiers) -- see CHANGELOG v3.26 + docs/PERFORMANCE.md; core stays zero-dep.
- **Slice 5 (v3.29) -- deep spiking predictive-coding cortex (NEW module, GPU-verified, honest):** the 2-level seed
  `cortex/predictive_coding` scaled DEEP under the predictive-coding framework (the user directive: use the predictive-
  coding node to build out the rest of the brain, torch GPU allowed). NEW CORE `cortex/pc_cortex_serve.py` = a zero-dep
  forward SERVER for an L-level (default 3) generative spiking hierarchy exported as `snn_pcnet/1` -- signed ON/OFF LIF
  error populations, discrete-spike settling (the seed's exact `_lif` generalized to L levels: `r_j += lr*(W_j^T s_{j-1}
  - s_j)`, top `r_L += lr*(W_L^T s_{L-1})`), an optional linear readout on the settled top cause. Trust-boundary `load()`
  validates the `snn_pcnet/1` geometry/chaining/readout; un-riggable self-check proves faithful replay vs an independent
  reference, genuinely-predictive settling (error-spike activity + recon residual fall vs a frozen-cause control),
  discrete {-1,0,1} spikes + real leak + no torch/numpy in core. TRAINER `research/pc_cortex_torch.py` (torch/GPU, ruff-
  excluded, NOT imported by core) learns the hierarchy by LOCAL Hebbian-on-spikes predictive coding -- **no global
  backprop across levels** (settling under no_grad; M-step `W_l += lr_w * outer(rate_{l-1}, r_l)`); ridge readout on the
  settled top cause. GPU-VERIFIED on the RTX 5070 (`.venv-cu128`, torch 2.11.0+cu128): N-MNIST (60k/10k, dims
  [2312,128,64,32]) test acc **0.748**, export fidelity server-vs-trainer top-cause **max_err 0.00e+00** (byte-faithful
  serve). HONEST NEGATIVES (printed, not asserted): LOCAL-PC learning adds only **+0.026** over random generative weights
  + a trained readout at this scale (most of the accuracy is architecture + linear readout, not the local rule); and the
  depth-earns-its-place XOR toy did NOT separate (3-level 0.75 == raw-avg 0.75) -- depth does not yet beat shallow. The
  value delivered is a faithful zero-dep spiking GENERATIVE trunk + the honest measurement, not a SOTA number. Rate/2-level
  seed `cortex/predictive_coding` kept as the small baseline (mirrors spiking_jepa keeping jepa_world_model). No live
  wiring this slice (zero safety surface); Slice 6 = opt-in checkpoint-gated perception-tier wiring. Atlas SPIKING 50.
- **Slice 6 (v3.30) -- PC-cortex perception tier WIRED into the live brain (safety-verified):** `cortex/pc_cortex_serve`
  wired into `cortex/brain.py` as an opt-in, ZERO-DEP, checkpoint-gated perception tier (`SNN_PC_CKPT` ->
  models/pc_cortex_nmnist.json; no torch needed to serve; degrades to stdlib), exposed via `perceive_pc(x)` -> settled
  top-cause latent, reported `state()["pc_cortex_tier"]` ("pc-cortex" | "stdlib"). The live tick is UNCHANGED (settling
  runs only on the explicit call, never in a tick); no actuator route. New brain self-check claim 8 (opt-in + right-length
  latent when loaded + degrade-to-None on bad input + arm untouched); autonomous_loop E-STOP/freeze + live_brain
  gated-danger freeze verified to hold. No new module -- Atlas SPIKING stays 50. brain.py commit isolated from the
  entangled uncommitted nengo-bridge working-tree changes (left untouched).
- **Slice 7 (v3.31) -- PC-cortex duck-types into PrefrontalExecutive.plan() (honest negative):** gave
  `cortex/pc_cortex_serve` the JEPA/HJepaServe planning interface (`encode`/`generate` top-down/`predict`
  action-AGNOSTIC/`make_goal_value` negative-L1) so a REAL `PrefrontalExecutive.plan()` consumes its top-down
  predictions (self-check: duck-types + deterministic + top-down generate beats a null cause). HONEST NEGATIVE
  asserted structurally: PC has NO action channel, so predict IGNORES command -> plan() TIES across options
  (perception != control; action-conditioned planning stays with JEPA/HJepaServe). Self-contained in
  pc_cortex_serve.py -- no live wiring, no actuator route. No new module -- Atlas SPIKING stays 50. The genuinely-
  useful PC planning role is active-inference goal_value/surprise-minimization (a follow-up), not a dynamics model.
- **Slice 8 (v3.32) -- PC active inference: surprise / expected-free-energy value (honest POSITIVE):** the
  self-consistent resolution of slice 7's negative -- PC is the PERCEPTION half of active inference, so its
  planning role is the SURPRISE/EFE value, NOT a dynamics model. Added `surprise(x)` (settled free energy: low=
  on-manifold/explained, high=anomalous) + `make_efe_value(x_goal,beta)` (= -(latent goal-dist + beta*surprise),
  duck-types into plan() over OBSERVATIONS). Un-riggable HEAD-TO-HEAD: a co-kernel look-alike TRAP CLOSEST to the
  goal in latent (0.315 < 0.455) but off-manifold (surprise 4.64 vs 0.03) -> a REAL PrefrontalExecutive.plan()
  with the goal-distance-ONLY value WALKS INTO the trap, while the EFE value (with PC surprise) AVOIDS it and picks
  the trusted good option. PC surprise adds discrimination distance alone cannot. HONEST scope: EFE scores
  observations, so a JepaWorldModel pairing needs a shared observation space (decoder / same-space training) --
  stated, not silently assumed. Self-contained in pc_cortex_serve.py, no live wiring, no actuator. No new module --
  Atlas SPIKING stays 50. Completes the PC-planning story: slice 7 = honest negative (no action channel), slice 8 =
  honest positive (surprise IS the value).
- **Slice 9 (v3.33) -- active-inference BRIDGE: JEPA dynamics + PC surprise compose (the "next lever"):** closes
  slice 7's gap. NEW zero-dep demo `research/pc_active_inference.py` (RESEARCH gate). A ridge DECODER
  `D(z_jepa)->observation` (stdlib, fit on `(JEPA.encode(x),x)`) maps JepaWorldModel's action-conditioned predicted
  latents back into PC's observation space where PC.surprise lives; composed value = `-(JEPA goal-dist +
  beta*PC.surprise(D(z)))`. JEPA supplies the action channel PC lacks; PC supplies the epistemic value distance
  lacks; shared synthetic observation manifold (the slice-8 prerequisite). Un-riggable: (1) JEPA.predict
  discriminates actions; (2) decoder recon 0.82 vs mean 7.0; (3) surprise survives encode+decode (on 0.005 << off
  3.478); (4) a REAL PrefrontalExecutive.plan() with the composed EFE value AVOIDS an off-manifold look-alike
  (JEPA-dist 1.86, closer than good 2.82) that the goal-distance-ONLY value WALKS INTO. Zero-dep, no live wiring,
  no actuator. Total self-checks 203->204. COMPLETES the arc: slice 7 negative (PC no action channel) -> slice 8
  positive (PC=surprise value) -> slice 9 (JEPA+decoder+PC compose = action-conditioned active inference).
