"""
v0.50 â€” single canonical runner for every module self-check.

(Said "every pure-stdlib self-check" until 2026-07-29, contradicting its own paragraph four lines
down. That was backlog M52: the phrase outlived the rule by nine days.)

The CI module list used to live hand-typed in BOTH .github/workflows/ci.yml and
docs/RUNNING.md â€” add or rename a module and the two silently drift. This is the ONE
source of truth: CI runs `python run_selfchecks.py`, docs point here. Each module's
`if __name__ == "__main__"` block is its own assert-based test; this runs them as
subprocesses (isolation) and fails loudly if any non-zero exits.

numpy + torch are core deps project-wide (the zero-dependency rule was removed 2026-07-20); modules may
import them freely and there is no pure-stdlib purity assertion anymore. Two tiers: CORE = the product
package (src/snn_data_lake/, run as `-m snn_data_lake.<mod>`); RESEARCH = benchmarks / storage / KG demos
(research/). Both CI-green; the MNIST/GPU/cloud scripts need torch/beam and are excluded by design.

  python run_selfchecks.py          # run all, exit non-zero on any failure
  python run_selfchecks.py --list   # print the module list, one per line
"""

import os
import sys
import subprocess
import tempfile

# children print Unicode (Ï†, âˆ’, â€¦); force UTF-8 so a Windows cp1252 console can't crash them.
# PYTHONPATH="src" so `snn_data_lake` (the package) + research/ modules importing it resolve.
#
# NORD_HOME added 2026-07-31: an ISOLATED home for the whole gate run. Without it every child
# inherited the operator's `~/.snn_data_lake` -- the same lake the always-on daemon appends to -- so
# the gate both wrote into the live brain and read a file that was changing underneath it. That is
# what made `cortex.brain`'s claim 4 ("two seed=0 Brains must be byte-identical") an intermittent
# ~42% failure rather than a stable one; subprocess isolation was UNDER-SAMPLING the race, not
# preventing it, since a cold subprocess leaves a wider window between the two constructions than a
# warm in-process build does. NORD_PERSIST is deliberately NOT set to 0: several self-checks exist to
# prove persistence works, and one shared isolated home lets them do that without touching the
# operator's brain.
_HOME = os.path.join(tempfile.gettempdir(), "snn_selfchecks_home")
os.makedirs(_HOME, exist_ok=True)
_ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": "src",
    "NORD_HOME": _HOME,
}

# CORE â€” the product: arm-loop + data-lake, clone-and-run (root). As of 3.0 organized by
# anatomical brain region (src/snn_data_lake/<region>/<module>.py); run as
# `-m snn_data_lake.<region>.<module>`. snn_data_lake.brain_atlas renders + verifies the map.
CORE = [
    "brain_atlas",
    "connectome",
    # cortex â€” sensing/encoding, cortical neuron models, microcircuit dynamics, readout, language
    "cortex.rank_order_coding",
    "cortex.population_encoding",
    "cortex.multimodal_encoder",
    "cortex.spike_preprocessing",
    "cortex.theta_gamma_code",
    "cortex.rsnn_encoder",
    "cortex.lif_neuron",
    "cortex.dendritic_neuron",
    "cortex.izhikevich_neuron",
    "cortex.hodgkin_huxley",
    "cortex.alif_neuron",
    "cortex.divisive_normalization",
    "cortex.neuronal_avalanches",
    "cortex.cortical_microcircuit",
    "cortex.cortical_prior",  # H01 human reference data: every entry carries its four attributes
    "cortex.placement",  # grown units ordered into cortical layers by measured H01 occupancy
    "cortex.lca_sparse_coding",
    "cortex.predictive_coding",
    "cortex.pc_cortex_serve",
    "cortex.is_snn_forward",
    "cortex.neural_sampling",
    "cortex.snn_classifier",
    # PROMOTED 2026-08-13 from reasoning/ -- and moved RESEARCH -> CORE with it. It had been
    # registered in RESEARCH, which runs out-of-process with root on PYTHONPATH; leaving a governed
    # cortex/ module there would register it without holding it to the CORE bar (in-process under
    # pytest), which is the bar the promotion exists to impose.
    "cortex.snn_moe_classifier",
    "cortex.trained_brain",
    "cortex.learned_matcher",
    "cortex.tempotron",
    "cortex.paradigm_b_matcher",
    "cortex.paradigm_b_engine",
    "cortex.gesture_recognition",
    "cortex.nl_interface",
    "cortex.spiking_lm_serve",
    "cortex.vsa_reasoning",
    "cortex.prefrontal_executive",
    "cortex.live_brain",
    "cortex.interaction_learning",
    "cortex.recognizer_bank",
    "cortex.neuron_zoo",
    "cortex.access_consciousness",
    "cortex.integration_measure",
    "cortex.self_model",
    "cortex.reasoning_curriculum",
    "cortex.autapse_neuron",
    "cortex.adex_neuron",
    "cortex.resonate_and_fire",
    "cortex.srm_neuron",
    "cortex.metacognitive_monitor",
    # PROMOTED from reasoning/brain_adapters/ 2026-08-12 (#10 lane 4c): live-tick percept
    # readout; numpy-only, real cached N-MNIST pack, measured ~2-5s. CORE, not TORCH_GATED.
    "cortex.laminar_local_readout",
    "cortex.wholistic_loop",
    "cortex.autotelic_goals",
    "cortex.quality_diversity",
    "cortex.self_improvement_suite",
    "cortex.theory_of_mind",
    "cortex.agency_monitor",
    "cortex.affect_core",
    "cortex.crossmodal_binder",  # W6: categorical emotion + 2nd-order ToM; cross-modal super-additive binding
    "cortex.executive_control",  # W8: ACC conflict + EVC + tree/sequence planning
    "cortex.competence_executive",  # PROMOTED+RENAMED 2026-08-12 (#10 lane 4f): dopamine-RPE lr gate + public competence-planning objects (wrappers retired same day); stdlib-only, ~1s
    "cortex.nef_compute",  # PROMOTED 2026-08-12 (PROMOTE lane): NEF reference implementation / nengo_bridge ground truth; numpy-only, ~0.2s
    "cortex.vsa_structure_mapping",  # PROMOTED 2026-08-12 (PROMOTE lane): Gentner structure-mapping analogy (VSA/resonator); stdlib, ~0.1s
    "cortex.rose_language",  # PROMOTED 2026-08-12 (PROMOTE lane): ROSE fluent-language pipeline (VSA constituency + grounded semantics + SLDA competence); ~0.5s
    "cortex.time_cells",
    "cortex.task_set",
    "cortex.generative_self_model",  # C1: spiking clock / task-set-shift / learned generative self-model
    "cortex.recursive_tom",
    "cortex.counterfactual_sim",  # C1 batch-2: N-agent recursive ToM / counterfactual do()-operator
    "cortex.learned_attention",  # C1 batch-3: learned spiking reticular attention gate
    "cortex.learned_evc",  # C1 batch-4: learned expected-value-of-control (dopamine-gated cost/value)
    "cortex.second_order_access",  # C2 r2: LIF metacognition OF access reliability (knowing THAT it knows)
    "cortex.heterogeneous_microcircuit",  # Brain-Cog-adopted: LIF+Izhikevich+AdEx composed live under a local rule
    "cortex.percept_surprise",  # loop C2: online unsupervised Oja-PCA reconstruction-residual surprise (orienting drive)
    "cortex.kb_ranking",  # PROMOTED 2026-08-11 (#10 batch-1): DistMult KB completion; offline fallback keeps it deterministic
    "cortex.corpus_stream",  # PROMOTED 2026-08-11 (#10 family C2): real corpus + CMUdict ingestion; offline fallback deterministic
    "cortex.world_kb_stream",  # PROMOTED 2026-08-11 (#10 family C2b): masked-slot fact completion vs per-relation baseline (0.4s)
    "cortex.language_self_build",  # PROMOTED 2026-08-11 (#10 family C6): real corpus + controls; NORD_EVAL_N bounds only the eval sample
    "cortex.language_readout_slda",  # PROMOTED 2026-08-11 (#10 family C5): equal-budget n-gram bar (fair_compare's port) + full-train bar
    "cortex.nightly_skills",  # PROMOTED 2026-08-11 (#10 family C9a, from the RESEARCH list): the ONE standing skills loop, sleep-tick entry
    # hippocampus â€” memory formation, spatial coding, replay
    "hippocampus.grid_cells",
    # MOVED RESEARCH -> CORE 2026-08-13. It was PROMOTED into hippocampus/ on 2026-08-11 (#10 family
    # C7) and registered in REGIONS, but its run_selfchecks entry stayed in RESEARCH -- so for two
    # days it was a GOVERNED module running out-of-process, outside the in-process pytest set, while
    # every surface said it was governed. Found by census, not by eye: the identical defect had just
    # been fixed for snn_moe_classifier by accident, so the class got mechanized (see the note at
    # RESEARCH's head). Runs in 2.84 s, in line with its CORE neighbours.
    "hippocampus.open_world_loop",
    "hippocampus.experience_replay",
    "hippocampus.sleep_replay",
    "hippocampus.ring_attractor",
    "hippocampus.growing_learner",
    "hippocampus.function_preserving_growth",
    "hippocampus.sleep_consolidation",
    "hippocampus.affective_memory",
    "hippocampus.contrastive_episode_analysis",
    "hippocampus.allocentric_map",
    "hippocampus.grid_place_map",  # W6: live grid + place cells binding a percept to an allocentric location
    "hippocampus.episodic_recall",  # W8: reconstructive pattern-completion autobiographical recall
    "hippocampus.spiking_grid_can",  # C1: continuous-attractor SPIKING grid network (bump path integration)
    "hippocampus.content_signature",  # PROMOTED 2026-08-11 (#10 family C1): moved here from the RESEARCH list
    "hippocampus.proximity_replay",  # PROMOTED+RENAMED 2026-08-11 (#10 family C3): P4b cross-process readback + 2 constructed reds (0.5s)
    "hippocampus.skill_builder",  # PROMOTED 2026-08-11 (#10 family C4): 2 domains, competence >= baseline, router 99% vs 50% chance (11.6s)
    "hippocampus.hippocampal_memory",  # PROMOTED 2026-08-11 (#10 lane 4a): balance/no-leak/consolidate/determinism asserts (seconds)
    "hippocampus.episode_consolidation",  # C1 batch-4: systems-consolidation of real episodes (sparse attractor + sleep replay)
    "hippocampus.lake_consolidation",  # sleep-time PRUNE of the symbolic durable lake (cap re-derivable bulk, keep learned) -> live + bounded
    "hippocampus.spiking_storage_prototype",  # factored associative memory; criteria judged vs a do-nothing cue control + permutation null, each with an executed red
    # basal ganglia â€” action selection, reward-modulated learning, decision, Go/NoGo gating
    "basal_ganglia.evidence_accumulation",
    "basal_ganglia.r_stdp",
    "basal_ganglia.valence_stdp",
    "basal_ganglia.action_gating",
    "basal_ganglia.latent_q",
    "basal_ganglia.mlp_q",
    # PROMOTED from reasoning/ 2026-08-12 (#10 lane 4d): selection stage over latent_q. Pure
    # stdlib composition (no torch), sub-second; carries the four-arm fusion A/B whose
    # value-channel verdict is a recorded NULL at the shipped rule (printed, not asserted).
    "basal_ganglia.brain_decision",
    # PROMOTED from RESEARCH 2026-08-04. Reward-modulated action selection over its own candidate
    # set: the three-factor RSTDP weights are driven by the GATE verdict (kept +1 / rolled_back -1).
    # Reachable on the live sleep path but only under NORD_SELFMOD=1, so charge is UNMEASURED here.
    "basal_ganglia.self_build_loop",
    "basal_ganglia.brain_native_selfbuild",
    # amygdala â€” aversive/threat learning (CS-US fear conditioning)
    "amygdala.fear_conditioning",
    "amygdala.affective_appraisal",  # C1: learned multi-dim affective appraisal (three-factor)
    # cerebellum â€” motor coordination, timing, forward/inverse models, oscillators
    "cerebellum.forward_model",
    "cerebellum.spiking_pid",
    "cerebellum.synfire_chain",
    "cerebellum.wavefront_planner",
    "cerebellum.jepa_world_model",
    "cerebellum.hjepa_serve",
    "cerebellum.spiking_jepa",
    "cerebellum.gain_adaptation",
    # PROMOTED from RESEARCH 2026-08-04. It was "reasoning.polychronization" in the RESEARCH list
    # below, which runs it as an isolated subprocess and leaves it outside all three P3 gates. It
    # holds the only delay-matched coincidence detector in the tree; keeping it ungoverned is how
    # two prior audits could record it dormant and nothing was obliged to happen.
    "cerebellum.polychronization",
    # brainstem + spinal cord â€” reflexes, actuation, sensorimotor loops, motor pattern generation
    # (the safety envelope lives here)
    "brainstem.reflex",
    "brainstem.serial_arm",
    "brainstem.so101_arm",
    "brainstem.arm_hardware",  # torque budget for the arm being built (asserts the shoulder shortfall)
    "brainstem.arm_config",
    "brainstem.interpreter",
    "brainstem.live_arm",
    "brainstem.arm_bridge",
    "brainstem.closed_loop",
    "brainstem.autonomous_loop",
    "brainstem.signal_loop",
    "brainstem.cpg_oscillator",
    # thalamus â€” the relay + global-broadcast hub (GWT) + reticular attentional gate
    "thalamus.global_workspace",
    "thalamus.reticular_gate",
    "thalamus.reentrant_workspace",  # W8: de-saturated competitive ignition + re-entry
    "thalamus.connectome_router",  # C1 batch-3: live signed/delayed spike propagation over the connectome graph
    "thalamus.gwt_deep",  # phase-binding demonstration (spiking-vs-rate control); promoted 2026-08-11
    # hypothalamus â€” homeostatic timing (the SCN circadian clock)
    "hypothalamus.circadian_clock",
    "hypothalamus.set_points",
    "hypothalamus.homeostatic_drives",  # W8: drives as percepts
    # midbrain â€” dopamine source nuclei (VTA/SNc RPE) + superior-colliculus orienting + deep gaze layers
    "midbrain.dopamine_nuclei",
    "midbrain.superior_colliculus",
    "midbrain.sc_deep_layers",
    "midbrain.multisensory_integration",
    # pons â€” cortico-ponto-cerebellar relay + the ascending reticular activating system (ARAS)
    # medulla â€” the autonomic vital nuclei (respiratory / cardiac CPG) + interoceptive arousal
    "medulla.vital_nuclei",
    "medulla.interoceptive_arousal",
    # epithalamus â€” the lateral habenula (anti-reward / negative RPE)
    "epithalamus.habenula",
    # olfactory â€” bulb decorrelation + piriform association
    # claustrum â€” cortical synchrony / salience hub
    "claustrum.claustrum",
    # neuromodulation â€” diffuse chemical modulators + glia (LC noradrenaline, raphe serotonin)
    "neuromodulation.acetylcholine",
    "neuromodulation.astrocyte_modulation",
    "neuromodulation.cortisol",
    "neuromodulation.locus_coeruleus",
    "neuromodulation.raphe_serotonin",
    "neuromodulation.intrinsic_motivation",
    "neuromodulation.spiking_curiosity",
    "neuromodulation.generalizing_novelty",
    "neuromodulation.drive_reduction",
    "neuromodulation.naca",
    "neuromodulation.empowerment",
    "neuromodulation.developmental_envelope",
    "neuromodulation.affect_integrator",  # C1 batch-2: spiking neuromod fusion -> coupled affect + self-organized emotions
    # plasticity â€” Hebbian/STDP, homeostatic, developmental learning rules
    "plasticity.bcm_plasticity",
    "plasticity.oja_pca",
    "plasticity.homeostatic_scaling",
    "plasticity.inhibitory_stdp",
    "plasticity.intrinsic_plasticity",
    "plasticity.short_term_plasticity",
    "plasticity.developmental_pruning",
    "plasticity.competitive_learning",
    "plasticity.lifelong_consolidation",
    "plasticity.criticality_regulator",
    "plasticity.meta_plasticity",
    "plasticity.benna_fusi",
    "plasticity.homeostatic_plasticity_stack",
    "plasticity.plasticity_rule_stack",
    "plasticity.snn_mnist_stdp_genn",  # specs/012 GeNN port: codegen-integrity check every box; GPU train is GENN_RUN-gated
    # substrate â€” the spiking data-lake, telemetry, and infra (not a brain region)
    "substrate._mathutils",
    "substrate.spike_stream_sketch",
    "substrate.spike_cardinality",
    "substrate.reservoir_sampling",
    "substrate.lake_memory",
    "substrate.stream_sketch_panel",
    "substrate.sliding_window_count",
    "substrate.frequency_moment",
    "substrate.streaming_quantile",
    "substrate.minhash",
    "substrate.morris_counter",
    "substrate.proximity_index",
    "substrate.energy",
    "substrate.data_quality",
    "substrate.drift_monitor",
    "substrate.spike_telemetry_hub",
    "substrate.streaming_hub",
    "substrate.neuro_chip",
    "substrate.neuro_chip_stream",
    "substrate.nir_export",
    "substrate.sensor_csv_ingest",
    "substrate.esp32_encoder_path",
    "substrate.make_sensor_dataset",
    "substrate.demo",
    "substrate.brain_bundle",
    "substrate.web_ingest",
    "substrate.lake_working_memory",
    "substrate.chat_memory",
    "substrate.signal_calibrator",
    "substrate.device",
    # PROMOTED from RESEARCH 2026-08-04. Infra, not a brain region: subprocesses + tree copies +
    # crc32, no weight anywhere. Owns _PROTECTED (the 9-file firewall) and the anchored _repo_root().
    "substrate.self_mod_harness",
    "substrate.home",
    "substrate.binlog",
    "substrate.growable",
    "substrate.region_wiring",  # the always-on daemon's faculty-wiring registry -> charge_all + honest coverage report
    "substrate.arm_view",  # stdlib PNG encoder + demand-gated offscreen arm frames
    "substrate.brain_telemetry",  # snapshot contract + recorder + /api/state (no-orphan-panel gate)
    "substrate.brain_geometry",  # anatomical layout (hand-authored coords, REAL connectome tracts)
    "substrate.cockpit",  # THE one frontend: page + contract gate + queued-not-raced chat turn
    # P4c 2026-08-04: the brain's DRIVE, promoted from the repo root where no gate reached it. Its
    # __main__ is a self-check that never starts the loop (--serve does that), so this entry is safe.
    "substrate.learn_daemon",
    "substrate.dormancy_gate",  # Policy 3: assert every brain module is reachable from a live path or EXEMPT
    "substrate.replication_guard",  # Policy 4: assert one canonical Brain/LiveBrain/AutonomousArm (no replica)
    "substrate.import_manifest",  # Policy 3: assert every reasoning/ module the live brain lazily imports still imports
    "substrate.method_dormancy_gate",  # Policy 3 amendment: assert no update rule on a live module lacks a live caller
    "substrate.eprop_transport_gate",  # e-prop invariant: assert no governed module transposes a weight into a weight update
    "substrate.doc_budget_gate",  # word ceilings on the standing docs: a document may not grow unnoticed
    "substrate.gradient_claim_gate",  # e-prop SCOPE: assert every governed learning rule is classified as owing a cosine or not
    "substrate.policy_manifest",  # POLICIES.md manifest: derivation reconciles, committed copy in sync, drift never worsens
    "substrate.perf_baseline",  # 022 T001 wiring check: baseline parses + scorer discriminates (cheap; never measures)
    "substrate.lake_bridge",  # 022 T005b: bridge round-trip red on a constructed temp lake (never the real home)
    "substrate.lake_segments",  # 022 T005: seal round-trip + CRC red on a constructed temp store (never the real home)
    "substrate.lake_index",  # 022 T007: pointer-index round-trip + watermark persistence + divergence red (temp db)
    "substrate.lake_rebuild_gate",  # 022 T008: rebuild equality + planted-divergence red + lake-tracking control (temp)
    "substrate.lake_migrate",  # 022 T009: per-tag identity migration + unknown-tag red (constructed multi-tag lake)
    "substrate.lake_engine",  # 022 T010-i: retention/disposition registries agree + constructed flood red
    "substrate.signal_ring",  # 022 T011: journaller replay A/B + loud-loss overrun red on scratch rings
    "substrate.safety_envelope",  # THE five approval-gated brainstem files: assert code and SECURITY.md still name the same five
    "substrate.ingest_guard",  # outside-data content boundary: licence deny-by-default + injection flagging
    "substrate.hf_stream",  # streamed HF datasets -> features; offline self-check (no network, no datasets dep)
    "substrate.region_coherence",  # C1 batch-3: cross-region Kuramoto coherence + integration -> ignition bias
    "substrate.basic_components",  # Brain-Cog-adopted composition interface (node + local-rule + connection palette)
    "substrate.similarity_sized_growth",  # Brain-Cog SCA-adopted: novelty-sized recruit cap + frozen-expert reuse
]
# RESEARCH â€” benchmarks / storage / KG demos (research/), run with root on PYTHONPATH.
RESEARCH = [
    "reasoning.test_prototype",
    "reasoning.hopfield_baseline",
    "reasoning.dense_associative_memory",
    "reasoning.sparse_distributed_memory",
    "reasoning.fly_hash",
    "reasoning.spiking_som",
    "reasoning.temporal_coding_storage",
    "reasoning.spike_compression",
    "reasoning.event_quadtree_compression",
    "reasoning.index_compression",
    "reasoning.spike_bloom",
    "reasoning.spiking_ssm",
    "reasoning.spike_kg",
    "reasoning.resonator_network",
    "reasoning.mcr_hdc",
    "reasoning.bayesian_snn",
    "reasoning.nmnist_ingest",
    "reasoning.nef_integrator",
    "reasoning.stats_bounds",
    "reasoning.loop_latency",
    "reasoning.delay_learning",
    "reasoning.eprop",
    # "reasoning.polychronization" -> PROMOTED to CORE as "cerebellum.polychronization" (2026-08-04)
    "reasoning.sn_p_system",
    "reasoning.free_energy_control",
    "reasoning.pc_active_inference",
    "reasoning.dishbrain_learning",
    "reasoning.triplet_stdp",
    "reasoning.cellular_neural_network",
    "reasoning.random_boolean_network",
    "reasoning.neural_heterogeneity",
    "reasoning.burst_multiplexing",
    "reasoning.stdp_wta_bayes",
    "reasoning.sorn",
    "reasoning.resume_learning",
    "reasoning.weight_perturbation",
    "reasoning.make_results_plot",
    # durable never-forget chat: contradiction reasoning, Hebbian skill wiring, autonomous scholarly
    # enrichment, and the lake-backed chat (all self-checks are deterministic + OFFLINE).
    "reasoning.lake_reason",
    "reasoning.autonomous_enrich",
    "reasoning.spike_chat_lake",
    "reasoning.train_of_thought",
    "reasoning.concept_reflection",
    "reasoning.model_builder",
    # the 5 Gf inducers COMBINED (not merged) into one file: reasoning.gf
    "reasoning.gf",
    # HONEST held-out RPM bench (rendered rasters -> spiking component perception -> derived cells),
    # replacing the old grammar self-test: 80% measured vs 12.5% chance, broken controls near chance.
    # (reasoning.arc_diagnostic is deliberately NOT here -- a ~27 s offline instrument, covered by
    # tests/test_arc_diagnostic.py; the smoke gate stays fast.)
    "reasoning.real_rpm_bench",
    "reasoning.reasoning_cortex",  # MoE router: ~8 dead reasoning studies made live on the converse path
    "reasoning.derivation",  # multi-hop transitive DEDUCTION (proof-carrying BFS over taught facts)
    "reasoning.reasoning_compose",  # MULTI-STEP composition over the MoE router (bounded expert grammar)
    "reasoning.pc_abduction",  # ABDUCTION (min-free-energy cause selection) + Pearl 3-step counterfactual
    "reasoning.rpm_solver",  # spiking NVSA Raven's-matrix solver: role-filler + example->rule induction (Gf)
    "reasoning.raven_bench",  # honest RPM benchmark (synthetic tasks + failing random control)
    "reasoning.vsa_synthesize",  # OUT-OF-GRAMMAR Gf: synthesizes novel rules as VSA spike-op programs (no fixed menu)
    "reasoning.semantic_memory",  # W6: VSA meaning-based recall (paraphrase-invariant, beyond the lexical crc32 gate)
    "reasoning.pragmatics",  # W8: speech-act recognition + Gricean implicature + indirect-request detection
    "reasoning.impact_analysis",  # pre-mutation blast radius (harness LSP principle, native ast + graphify) -> self-build prefers surgical changes
    "reasoning.ar_spiking_lm_serve",  # forward-only Spiking-RWKV serve: order-sensitive logprob (the BLiMP lever)
    "reasoning.eprop_lm_serve",  # BRAIN-NATIVE e-prop recurrent-spiking LM serve: order-sensitive logprob (no backprop)
    "reasoning.eprop_lm_alif_serve",  # deep-ALIF e-prop LM serve: adaptive-threshold long-range, order-sensitive logprob
    "reasoning.self_improve",
    "reasoning.godel_machine",
    # the two OUTSIDE yardsticks the depth ceilings are measured against. Both SKIP-with-notice when
    # their dataset is not downloaded (data/ is gitignored), so CI stays green either way -- which is
    # also why neither can be treated as a regression test for the ceilings it measures.
    "reasoning.arc_bench",  # ARC-AGI: exact-match pass@2 over 800 tasks; controls must score ~0
    "reasoning.blimp_bench",  # BLiMP: 67 paradigms x 1000 minimal pairs; chance 50%, human 88.6%
    "reasoning.arc_solver",  # test-time program synthesis over 2-D grids (object DSL + iterative deepening)
    "reasoning.arc_scene",  # Gf Phase-2 perceptual VOCABULARY: extensible attribute/relation registry (palette/morph/topo/shape-eq/align) over components() -- measurement only, feeds the Phase-3 mint
    "reasoning.arc_mint",  # Gf Phase-3 MINT: hypothesize out-of-closure primitives from arc_scene + memorization-resistant functional induction (LOO-validated); held-out metric
    "reasoning.arc_residual",  # Design B: the induced descriptor alphabet the mint selects over
]
# TORCH/NUMPY-GATED CORE -- modules that import a heavy numeric dep (torch or numpy); run ONLY where
# it is importable and SKIPPED-with-notice otherwise, so the pure-stdlib CI matrix + the in-process
# pytest purity asserts (which require NO numpy/torch in sys.modules) stay green. torch is an optional
# extra ([gpu]/[train]). hippocampus.ng_snn is the original documented exception; the BRAIN TIER
# (cortex.brain the unified brain + cortex.neurogenesis_engine) is the user-authorized relaxation
# of the zero-dep-core rule for "the best-performing brain" -- the zero-dep core still runs with no
# torch (Brain degrades to its deterministic narrator + no growth). SpikeGPT was SCRAPPED (v3.77): no
# imported language model; conversation is the narrator + the brain-native cortex.language_acquisition
# (numpy predictive-coding + local plasticity, gated here so it never pollutes the pure in-process test).
TORCH_GATED = [
    "hippocampus.ng_snn",
    "hippocampus.growable_readout",  # PROMOTED 2026-08-11 (#10 lane 4b): grow-preservation 1.9e-06, freeze retention delta (8.4s)
    "cortex.brain",
    "cortex.neurogenesis_engine",
    "cortex.language_acquisition",
    "cortex.eprop_nmnist",  # PROMOTED 2026-08-12 (PROMOTE lane 4/4): N-MNIST e-prop perception encoder; bare check ~6s CPU incl torch import
    "cortex.eprop_shd",  # PROMOTED 2026-08-13 (specs/013 C3): SHD e-prop audio perception encoder, the auditory twin
    "cortex.eprop_kernel",  # NEW 2026-08-13 (specs/013 C4): the ONE e-prop engine both encoders import; two demonstrated reds
    "reasoning.schema_induction",
]
# NENGO-GATED -- the Nengo bridge (substrate.nengo_bridge) exposes the NEF/LIF/spiking substrate to
# Nengo (sim + NengoLoihi deploy target); it imports `nengo` (optional). Run only where nengo is
# importable, SKIPPED-with-notice otherwise (its own __main__ also self-SKIPs). Zero-dep core unaffected.
NENGO_GATED = ["substrate.nengo_bridge"]

# [train]-extra modules (torchvision at module top, the recorded P7 adjudication: torchvision is
# REJECTED for core because a bare pip resolve replaces the cu128 torch with the CPU build -- see
# plasticity/snn_mnist_stdp.py's docstring). Run only where torchvision is importable (the gate
# env carries torchvision+cu128), SKIPPED-with-notice otherwise (CI's torch-tier job installs no
# extras, so these skip there -- same contract as NENGO_GATED). Added 2026-08-11 with the
# deep_stdp-lineage promotions (#10 lane 2).
TRAIN_GATED = [
    "plasticity.snn_mnist_stdp",  # PROMOTED 2026-08-11 (lane 2a): the Diehl-Cook rule hub (full real train, GPU, 58.8s)
    "plasticity.deep_stdp",  # PROMOTED 2026-08-11 (lane 2b): library mechanics check; DEEPSTDP_RUN=1 = the full 8-min benchmark
    "cortex.deep_stdp_latency",  # PROMOTED 2026-08-11 (lane 2c): latency machinery mechanics; DSLATENCY_RUN=1 = the full 6.6-min benchmark
    "hippocampus.developmental_loop_brain",  # PROMOTED 2026-08-12 (lane 4e): wiring mechanics (~5s); DLB_RUN=1 = the full 5.8-min measurement
]

# The repo root, for resolving ARTIFACT_GATED paths regardless of the cwd the runner was invoked from.
_ROOT = os.path.dirname(os.path.abspath(__file__))

# ARTIFACT-GATED: module -> the repo-relative input it cannot run without.
#
# For a module whose self-check needs a gitignored binary that NOTHING in the tree can produce. Such a
# module must still raise rather than bare-return when its input is missing (a bare return exits 0 and
# is scored PASS -- a gate that cannot fail), so the decision to tolerate the absence is made HERE,
# once and visibly, as a skip that is COUNTED SEPARATELY and never folded into the pass tally.
#
# Writing the producer removes entries from this dict. That is the fix; this is the accounting.
#
# EMPTY as of 2026-07-29, and by the fix rather than by giving up on it: the sole entry was
# `cortex.percept_surprise` -> `models/nmnist_pack.npz`, one of the two perception packs that had five
# reader sites and no producer. `training/make_packs.py` is now that producer -- it rebuilds
# nmnist_pack.npz and shd_pack.npz from tonic, byte-for-byte identical to the artifacts every
# perception number in this repo was measured against (verified: 0 of 8,878,080 and 0 of 17,920,000
# cells differ, crc32 0xb1bd2e57 / 0xaca9f8b2). A fresh clone can satisfy the gate, so it should have
# to. The mechanism stays for the next artifact that genuinely has no producer.
ARTIFACT_GATED = {}


def _torch_ok():
    # torch is a core dependency, so this is normally True and TORCH_GATED runs. It stays a probe
    # rather than an assumption only so a broken install fails with a clear skip notice instead of 6
    # confusing import errors. The NORD_TORCH=0 off-switch was removed 2026-07-29: skipping the torch
    # tier on demand was the zero-dependency rule's last mechanism, and that rule was retired
    # 2026-07-20 when torch became unconditional in pyproject.
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _nengo_ok():
    try:
        import nengo  # noqa: F401

        return True
    except Exception:
        return False


def _train_ok():
    # torchvision is the [train]-extra discriminator (snntorch is core); where it is absent the
    # TRAIN_GATED modules cannot even import (module-top import, the recorded deliberate shape).
    try:
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


_TORCH = _torch_ok()
_NENGO = _nengo_ok()
_TRAIN = _train_ok()
MODULES = (
    CORE
    + (TORCH_GATED if _TORCH else [])
    + (NENGO_GATED if _NENGO else [])
    + (TRAIN_GATED if _TRAIN else [])
    + RESEARCH
)


def _assert_tiers_agree_with_regions():
    """A GOVERNED module may not sit in RESEARCH, and every governed module must be registered.

    RESEARCH runs out-of-process and is excluded from the in-process pytest set
    (`tests/test_selfchecks.py`: `_IN_PROCESS = CORE + TORCH_GATED`). So a module that is in
    `REGIONS`/`TORCH_TIER` but listed in RESEARCH *looks* governed at every surface -- right
    directory, REGIONS entry, p4c count falling -- while never being held to the CORE bar. Nothing
    detected that, because each half is individually correct.

    WHY THIS EXISTS, and the count that earned it. Two promotions landed this defect:
    `hippocampus.open_world_loop` (2026-08-11) and `reasoning.snn_moe_classifier` -> cortex/
    (2026-08-13). The second was caught only because someone happened to read which list the entry
    was in; the first had sat governed-but-out-of-process for two days. A census over REGIONS x the
    tier lists then found exactly ONE remaining (open_world_loop) -- so the class is real, recurring,
    and small enough to close. Output when this check landed: 1, now 0.

    Reachable red: move any REGIONS entry into RESEARCH, or delete a governed module's tier entry.
    Both raise here. (The census that produced this first reported a SECOND finding,
    `substrate.nengo_bridge` "registered nowhere" -- a FALSE POSITIVE from its own incomplete tier
    list, which did not know NENGO_GATED existed. Hence `_ALL_TIERS` below is built from every tier,
    not an enumerated guess: a checker with an incomplete denominator manufactures findings.)
    """
    try:
        from snn_data_lake import REGIONS, TORCH_TIER
    except Exception:  # pragma: no cover - import guard raises on a GIL build; not this check's job
        return

    governed = {f"{r}.{m}" for r, mods in REGIONS.items() for m in mods}
    governed |= {f"{r}.{m}" for r, mods in TORCH_TIER.items() for m in mods}
    _ALL_TIERS = set(CORE) | set(TORCH_GATED) | set(NENGO_GATED) | set(TRAIN_GATED) | set(RESEARCH)

    in_research = sorted(governed & set(RESEARCH))
    assert not in_research, (
        "GOVERNED modules registered in RESEARCH -- they run out-of-process and escape the "
        f"in-process pytest set while every other surface calls them governed: {in_research}"
    )
    unregistered = sorted(governed - _ALL_TIERS)
    assert not unregistered, (
        f"governed modules in NO run_selfchecks tier -- they are never self-checked: {unregistered}"
    )


_assert_tiers_agree_with_regions()


def run_all():
    if not _TORCH:
        # torch is a CORE dependency, so this is a BROKEN INSTALL, not a supported configuration.
        # The old message said "zero-dep matrix stays green" and reassured the reader that skipping
        # 6 modules was fine -- true while the zero-dependency rule existed, misleading since it was
        # retired. Left as a skip rather than promoted to a hard failure because that is a separate
        # decision with its own blast radius; the message now says what the skip means.
        for m in TORCH_GATED:
            print(
                f"SKIP  {m}  (torch NOT IMPORTABLE -- broken install: torch is a core dependency)"
            )
    if not _NENGO:
        # nengo genuinely IS optional (the [nengo] extra), so this skip is a supported path.
        for m in NENGO_GATED:
            print(f"SKIP  {m}  (nengo not importable -- optional [nengo] extra not installed)")
    failed, artifact_skipped = [], []
    for m in MODULES:
        missing = ARTIFACT_GATED.get(m)
        if missing and not os.path.exists(os.path.join(_ROOT, missing)):
            # ARTIFACT-GATED, added 2026-07-29. A module's own self-check should RAISE when its input
            # artifact is absent -- printing SKIP and bare-returning exits 0 and is scored PASS on exit
            # status alone, which is a CORE gate entry that cannot fail. But when nothing in the tree
            # can produce that artifact, the correct raise turns the whole suite red on every fresh
            # clone for a reason no contributor can act on.
            # So the decision is made HERE, once, visibly, instead of by a bare-return inside the
            # module: a missing artifact is an explicit SKIP that is COUNTED SEPARATELY and never
            # folded into the pass tally. The module keeps its raise for the case that actually
            # matters -- the artifact is present but unreadable. ARTIFACT_GATED is empty today (the
            # perception packs it was built for now have a producer); this stays for the next one.
            print(f"SKIP  {m}  (missing artifact {missing} -- no producer exists in this tree)")
            artifact_skipped.append(m)
            continue
        cmd = [sys.executable, "-m", f"snn_data_lake.{m}"]  # every module runs as a package module
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_ENV)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {m}")
        if not ok:
            failed.append(m)
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
    ran = len(MODULES) - len(artifact_skipped)
    print("=" * 56)
    print(f"{ran - len(failed)}/{ran} self-checks passed")
    if artifact_skipped:
        # Printed as its own line, not merged into the total, so "277/277" can never quietly mean
        # "276 ran and one was skipped".
        print(f"ARTIFACT-SKIPPED ({len(artifact_skipped)}): " + ", ".join(artifact_skipped))
    if failed:
        print("FAILED: " + ", ".join(failed))
    return not failed


def run_one_in_process(mod):
    """Run one module's self-check IN-PROCESS via runpy, capturing its output. Returns (ok, text).

    run_all() uses subprocesses, which is the better isolation and stays the default. This exists for
    one situation subprocesses cannot serve: a frozen PyInstaller build has no interpreter to spawn,
    so `python -m snn_data_lake.<mod>` is unavailable and the packaged app would otherwise have no way
    to verify that the core it bundled is actually runnable.

    sys.argv is isolated to [mod] so a module sees no stray caller flags -- several modules dispatch on
    argv (spike_telemetry_hub runs its CLI when given any, control_panel used to run --smoke on none),
    and leaking the parent's arguments would silently run the wrong entry point.
    """
    import io
    import runpy
    from contextlib import redirect_stderr, redirect_stdout

    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = [mod]
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
        return True, buf.getvalue()
    except SystemExit as e:  # a module may sys.exit(0) on success
        return (e.code in (0, None)), buf.getvalue()
    except BaseException as e:  # AssertionError / any self-check failure
        return False, f"{buf.getvalue()}\n{type(e).__name__}: {e}"
    finally:
        sys.argv = saved_argv


def run_all_in_process(emit=print):
    """Every CORE module's self-check, in-process. True iff all passed. See run_one_in_process."""
    failed = []
    for m in CORE:
        ok, out = run_one_in_process(m)
        emit(f"{'PASS' if ok else 'FAIL'}  {m}")
        if not ok:
            failed.append(m)
            emit(out[-2000:])
    emit("=" * 56)
    emit(
        f"{len(CORE) - len(failed)}/{len(CORE)} core self-checks passed (in-process)"
        + (f"   FAILED: {', '.join(failed)}" if failed else "")
    )
    return not failed


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(MODULES))
        sys.exit(0)
    if "--in-process" in sys.argv:
        sys.exit(0 if run_all_in_process() else 1)
    sys.exit(0 if run_all() else 1)
