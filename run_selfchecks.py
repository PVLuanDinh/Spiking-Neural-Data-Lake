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

import contextlib
import os
import sys
import subprocess
import tempfile
import time

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
    "cortex.h01_ingest",  # T062: the c3 synapse graph -- 8 gates, each with its red, offline
    "cortex.arealization",  # T040+T041: exhaustive GRN censuses + prosomeric grid + graded arm
    "cortex.growth_program",  # T042: sealed view, shared-program counter, dilution, refusal reds
    "cortex.prior_drift_harness",  # 022 T063: both reds, the PLANTED positive control, the WRONG validity precondition and every nuisance refusal, on a 64-unit synthetic population
    "cortex.placement",  # grown units ordered into cortical layers by measured H01 occupancy
    "cortex.growth_stop",  # T047: overgrowth red + four regimes + |LP| veto + measured null clause
    "cortex.memory_diagnostic",  # G1 causal contract synthetic reds
    "cortex.recurrent_pool",  # G1(b): R-b0 parity, R-b3, R-b4 pool noise, R-b5 and every red
    "cortex.pool_selection",  # G1(b) campaign: ordering, power, R-b7 reds, step order, exit codes
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
    "cortex.attention_schema",  # specs/022 T078 Part I: G3/G4's six reds on constructed streams (no brain, no lake); the campaign is --t078-campaign only
    "cortex.report_order_probe",  # specs/022 T081 [US4]: 2.4's six reds on constructed workspaces (no brain, no lake); the campaign is --t081-campaign only
    "cortex.everyday_battery",  # specs/022 Phase 4 C0: 2.5's five constructed reds + (k)/(s)/(p)'s constructed arms (no brain, no lake); the campaign is --c0-campaign only
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
    "hippocampus.lake_recall",  # T057: recall(cue) -> k episodes with margins from the lake (FR-017)
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
    "basal_ganglia.learning_progress",  # T046a: |LP| four regimes + discharged consumer + goal-store reds
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
    "cerebellum.microcircuit",  # T054-i: the tissue (granule/PF/Purkinje/MLI; cb_ctx + cb_deep)
    "pons.mossy_relay",  # T054-i: the pn stage (the pons region's first computing module)
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
    "thalamus.interoceptive_relay",  # T039j: thal_relay -- the declared tract hop that had no module
    "thalamus.ignition_instrument",  # T033: dual-derived ignition scorer over the re-entry stage
    "thalamus.binding_instrument",  # T080: the live cross-modal binder's report vs its own re-paired features
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
    "plasticity.wm_capacity",  # T056: K = T_max/t_s derived + five guards (wm_attractor retired)
    "plasticity.developmental_pruning",
    "plasticity.structural_rewiring",  # T048: A/A + budget-conservation + MSP sign arms
    "plasticity.eligibility_algebra",  # T050: incumbent bit-equivalence + mis-declaration red
    "plasticity.rule_composer",  # T051a: naive-sum red + causal/measured LOO + refusals by message
    "plasticity.espp_rule",  # T052-i: delegation-live + BIT/FULL/DELTA equivalence + coefficient pins
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
    "substrate.paired_contrast",  # T077 2.10: the seed-paired t + exact FPR, lifted from the T063 harness
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
    "substrate.brian2_backend",  # Brian2 reference dynamics == differentiable Torch bridge
    # PROMOTED from RESEARCH 2026-08-04. Infra, not a brain region: subprocesses + tree copies +
    # crc32, no weight anywhere. Owns _PROTECTED (the 9-file firewall) and the anchored _repo_root().
    "substrate.self_mod_harness",
    "substrate.home",
    "substrate.binlog",
    "substrate.growable",
    "substrate.spontaneous_activity",  # T045: OU stationarity + same-path A/B + nulls + refusals
    "substrate.growth_coverage_gate",  # T046: coverage ratchet + refusals + sharpness + nulls + Allen
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
    "substrate.policy_manifest",
    "substrate.capability_bar",  # X1: 4 constructed reds + the REFUSED arm + the surface check
    "substrate.immortal_retention",  # L5: label reds, the reclaimer scan, the cross-process round trip
    "substrate.conclusion_protocol",  # C5: row reds, the verdict/aggregate refusals, the prose control  # POLICIES.md manifest: derivation reconciles, committed copy in sync, drift never worsens
    "substrate.perf_baseline",  # 022 T001 wiring check: baseline parses + scorer discriminates (cheap; never measures)
    "substrate.amdahl_gate",  # specs/008 Gate Zero / 022 S6: the three reds, then the neuter-and-measure ceiling reading
    "substrate.lake_bridge",  # 022 T005b: bridge round-trip red on a constructed temp lake (never the real home)
    "substrate.lake_segments",  # 022 T005: seal round-trip + CRC red on a constructed temp store (never the real home)
    "substrate.lake_index",  # 022 T007: pointer-index round-trip + watermark persistence + divergence red (temp db)
    "substrate.lake_rebuild_gate",  # 022 T008: rebuild equality + planted-divergence red + lake-tracking control (temp)
    "substrate.lake_migrate",  # 022 T009: per-tag identity migration + unknown-tag red (constructed multi-tag lake)
    "substrate.lake_engine",  # 022 T010-i: retention/disposition registries agree + constructed flood red
    "substrate.signal_ring",  # 022 T011: journaller replay A/B + loud-loss overrun red on scratch rings
    "substrate.kernel_state",  # 022 T013: SoA alignment/contiguity + scratch-lake round-trip + refusal reds
    "substrate.kernel_tick",  # 022 T014: delay-ring arrival + T=1==T=4 train equality + imbalance + refusals
    "substrate.determinism_gate",  # 022 T015: f32-diverges/int64-agrees control arms + a thread pair (full sweep in CI --assert)
    "substrate.replay_gate",  # 022 T017: G5 resume byte-identical + omit-positions/mismatch refusals + re-seed divergence control
    "substrate.wiring_manifest",  # 022 T018: five checks + constructed reds + NIR-HDF5 round-trip on a scratch manifest
    "substrate.manifest_gate",  # 022 T019: committed-artifact checks + divergence/join/region reds through the gate's verdict path
    "substrate.report_channel",  # 022 T034: dark-record honesty + moving strength + release + determinism
    "substrate.episode_driver",  # 022 T035: port drives + trial structure/oracle + G5 restore + refusals
    "substrate.afferent_boundary",  # 022 T036: sentinel->MISSING + 014-normalizer reuse + proprio neuter A/B
    "substrate.efferent_boundary",  # 022 T037: passage counting + paired copy-cut charge + missing-body honesty
    "substrate.port_charge_probe",  # 022 T021: sensitivity flip + specificity hold + margin varied-vs-frozen separation
    "insula.interoceptive_integration",  # 022 T021a: deviation-from-self salience + frozen-baseline red + habituation
    "substrate.envelope_hash_gate",  # 022 T021b: live five-file control + tamper/stale-approval/unbaselined reds
    "substrate.region_charge_gate",  # 022 T022: registration-never-greens + constructed attribution + non-diverged-charges-nothing
    "substrate.atlas_node_gate",  # 022 T024: constructed-phantom red + 7 reference-rule probes incl. both landing false-positives
    "substrate.selfcheck_cache",  # carry-forward cache (envelope v3): dependency-flip + poison-refusal + envelope + FAIL-never-carries reds, plus the roster-split / data-input / named-env / tree-scanner precision reds and their key-invariance controls
    "substrate.safety_envelope",  # THE five approval-gated brainstem files: assert code and SECURITY.md still name the same five
    "substrate.ingest_guard",  # outside-data content boundary: licence deny-by-default + injection flagging
    "substrate.hf_stream",  # streamed HF datasets -> features; offline self-check (no network, no datasets dep)
    "substrate.region_coherence",  # C1 batch-3: cross-region Kuramoto coherence + integration -> ignition bias
    "substrate.basic_components",  # Brain-Cog-adopted composition interface (node + local-rule + connection palette)
    "substrate.similarity_sized_growth",  # Brain-Cog SCA-adopted: novelty-sized recruit cap + frozen-expert reuse
    "substrate.world_red_probe",  # specs/022 W3: 7 wiring reds (pinned constant dead, absent != zero, 4dp decisive, refit gap arithmetic)
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
    "hippocampus.online_readout",  # L1c: delta beats batch-only on a drifting chain, sign flip below, neutered bit-identical
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


def latch_in_window(mtime, w0, w1):
    """True iff the E-STOP sidecar's mtime falls inside the running check's own wall-clock slot.

    Attribution rule (chip task_100cb4d5, adjudicated 2026-08-31): batteries had attributed a
    found latch to whatever check's slot it SURFACED in -- existence alone -- which named
    innocent checks: only an arm-driving process tree writes this file (so101_arm owns it),
    and a straggler from an earlier check's tree can leave one during a later check's slot or
    a carried stretch. So a latch is attributed to the running check ONLY when its mtime sits
    inside [w0, w1]; anything else prints as a background straggler, unattributed, instead of
    the runner inventing a name. The counted adjudication record (three batteries' sets, the
    13-run standalone sweeps) lives in the commit that landed this, not here -- re-derive,
    never cite.

    Disclosed limits, each measured: (a) a straggler writing INSIDE the window is
    indistinguishable from the check's own write -- the window NARROWS the smear (inter-slot
    gaps, carried stretches, the first slot), it cannot erase it; full attribution needs
    writer identity inside the latch, which is envelope-owned and out of runner scope.
    (b) NTFS mtime can lag time.time() by ~1 ms, so both boundaries are soft at millisecond
    scale. (c) subprocess.run(capture_output=True) drains pipes to EOF, so it WAITS for
    grandchildren that inherit stdio -- only a grandchild spawned with detached stdio can
    outlive its slot; the straggler class is narrow, not generic. VERDICT SCOPE: this
    variance was never a determinism-contract violation -- the contract binds the substrate
    tick and G5 replay (docs/POLICIES.md, THE DETERMINISM CONTRACT); battery-runner REPORTING
    sits outside it, and the defect was the report naming innocent checks.
    """
    return w0 <= mtime <= w1


def _sweep_latch(latch, m, w0, w1, latch_left, latch_stray):
    """The classify-and-remove step after every LIVE check: read the sidecar's mtime BEFORE
    removing it, then attribute (latch_left) or report unattributed (latch_stray) by
    latch_in_window. Extracted so the attribution WIRING is testable against a real file --
    the pre-fix defect (attribution by existence alone) lived exactly here. TWO production
    call seams consume it (run_all's and run_all_in_process's per-check calls); neither is
    driven by a test, both are stated in the test, and the in-process seam was run-verified
    end-to-end 2026-08-31 (a sentinel-home --in-process run attributed the one real
    latch-leaver in-slot with the sentinel left untouched)."""
    if not os.path.exists(latch):
        return
    latch_mtime = os.path.getmtime(latch)  # read BEFORE remove
    os.remove(latch)
    if latch_in_window(latch_mtime, w0, w1):
        latch_left.append(m)
    else:
        # surfaced during m's slot but written outside it -- see latch_in_window
        latch_stray.append(f"{m}@{latch_mtime - w0:+.1f}s")


def _latch_report_lines(latch_left, latch_stray):
    """The sidecar report, ONE formatter for both runners (subprocess and in-process) so the
    two paths cannot drift into different claims about the same signal."""
    lines = []
    if latch_left:
        lines.append(
            f"GATE-HOME E-STOP SIDECAR left by {len(latch_left)} check(s) and cleared between "
            "checks (a check that exits stopped is information, not a defect; "
            "attribution is mtime-in-slot -- latch_in_window): " + ", ".join(latch_left)
        )
    if latch_stray:
        lines.append(
            f"GATE-HOME E-STOP SIDECAR cleared {len(latch_stray)} time(s) with mtime OUTSIDE "
            "the running check's slot -- background straggler from an earlier arm-driving "
            "check's process tree, UNATTRIBUTED (slot-it-surfaced-in@mtime-offset-from-that-"
            "slot-start): " + ", ".join(latch_stray)
        )
    return lines


@contextlib.contextmanager
def _gate_home():
    """Point NORD_HOME at the runner's shared isolated home for the duration, then RESTORE the
    caller's value -- never a bare pop of a value that was set (substrate.home records why: a
    bare pop deletes the caller's own isolation). What the 2026-08-31 probe established: home()
    and so101_arm._latch_path() read NORD_HOME at USE time (set-after-import redirects, re-set
    re-resolves), so this redirects every home()-derived path resolved INSIDE the span. What it
    does NOT cover, said plainly: a thread or atexit handler resolving home() AFTER the span
    exits reads whatever the caller's env then holds -- which is why run_all_in_process spans
    its WHOLE loop with one gate and the --in-process CLI dispatch sets the env at PROCESS
    scope (atexit fires after every context manager has restored). Distinct from
    substrate.home.throwaway_home(), which gates a FRESH temp dir per use; this points at the
    runner's one SHARED home so persistence checks can prove persistence (the _ENV rationale)."""
    prev = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = _HOME
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prev


# H1 (2026-09-02, owner decision O3): the six `tools/` gates ran only in the CI lint job,
# which has been dark since 2026-08-20 -- `p4c_ratchet --assert` had NO local execution path
# (audit section 8 rank 10). This stage gives every tools gate one, so `run_selfchecks` is
# the whole local gate. Never carried, never run --in-process (a frozen bundle ships no tools/).
TOOLS_GATES = [
    ["tools/p4c_ratchet.py", "--assert"],
    ["tools/p6_gate.py", "--assert"],
    ["tools/dependency_ledger.py", "--assert"],
    ["tools/red_reachability.py", "--assert"],
    ["tools/citation_guard.py"],
    ["tools/doc_command_guard.py", "--assert"],
]


def run_tools_gates(gates=None, emit=print):
    """Run the tools/ gates as subprocesses in the isolated env; return the failed names.

    The red is reachable by construction and asserted by tests/test_run_selfchecks_tools_stage.py:
    a gate argv that exits non-zero comes back in the failed list, one that exits zero does not."""
    gates = TOOLS_GATES if gates is None else gates
    failed = []
    for argv in gates:
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, *argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=_ENV,
            cwd=_ROOT,
        )
        ok = r.returncode == 0
        name = "tools/" + os.path.basename(argv[0])
        emit(f"{'PASS' if ok else 'FAIL'}  {name}  [{time.perf_counter() - t0:.1f}s]")
        if not ok:
            failed.append(name)
            emit(((r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]).strip())
    emit(f"{len(gates) - len(failed)}/{len(gates)} tools gates passed")
    return failed


def run_all(carry=False):
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
    # --carry (specs: the owner-authorized carry-forward cache, 2026-08-21): a PASS whose
    # ENTIRE static import closure, the data files / named env vars / root-package roster /
    # tree surface its closure reads, and the global envelope are byte-identical to a prior
    # green on this host is CARRIED -- a hash verification, never a skip. LOCAL batteries only: nothing
    # in .github/ passes --carry, and exit-certification batteries must not either. Any
    # FAIL in a carrying run is re-run FULL before diagnosis (the cache's own banner and
    # substrate/selfcheck_cache.py's docstring carry the full blind-spot disclosure).
    cache = None
    carried = []
    sys.path.insert(0, "src")
    if carry:
        from snn_data_lake.substrate.selfcheck_cache import SelfcheckCache

        cache = SelfcheckCache()
        print(cache.banner())
    # T037a hygiene: a self-check that E-STOPs its So101Arm and exits leaves the DURABLE latch
    # sidecar in this run's SHARED isolated home; the next subprocess (a different pid) would
    # honour it and come up frozen -- the shared-home sequence coupling, made safety-shaped.
    # Cleared between checks, and SAID: the check that left it is named -- but ONLY when the
    # latch mtime falls inside that check's own wall-clock slot (latch_in_window; a straggler
    # from an earlier arm-driving check's tree is reported unattributed, never named). The
    # real brain's home is never touched by THIS path (NORD_HOME=_HOME rides in _ENV for the
    # child's whole lifetime). The --in-process runners carry their own gates since 2026-08-31
    # (_gate_home per call + a loop-wide span + a process-scope set at the CLI dispatch --
    # three scopes because the per-call gate alone does not cover stragglers that outlive a
    # call; each scope is named where it lives).
    from snn_data_lake.brainstem.so101_arm import ESTOP_LATCH  # the ONE spelling, in the envelope

    latch = os.path.join(_HOME, ESTOP_LATCH)
    latch_left, latch_stray = [], []
    if os.path.exists(latch):
        os.remove(latch)
        print("GATE-HOME E-STOP SIDECAR from a previous run cleared before this battery")
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
        if cache is not None:
            hit = cache.consult(m)
            if hit is not None:
                saved = float(hit.get("secs", 0.0))
                print(
                    f"CARRY {m}  (inputs unchanged; passed {hit['when']}, {saved:.1f}s not re-spent)"
                )
                carried.append(saved)
                continue
            # the miss is attributed, so a battery log says WHY each module re-ran
            print(f"RERUN {m}  ({cache.why(m)})")
        cmd = [sys.executable, "-m", f"snn_data_lake.{m}"]  # every module runs as a package module
        t0 = time.perf_counter()
        w0 = time.time()  # wall-clock slot start -- latch mtimes compare against THIS clock
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_ENV)
        secs = time.perf_counter() - t0
        w1 = time.time()
        ok = r.returncode == 0
        # the wall-clock rides on every line so a battery log IS the per-module histogram
        # (the battery-speed research needed one and none existed), and the cache stores it
        # so a later carry can say what it did not spend -- measured, never claimed.
        print(f"{'PASS' if ok else 'FAIL'}  {m}  [{secs:.1f}s]")
        _sweep_latch(latch, m, w0, w1, latch_left, latch_stray)
        if not ok:
            failed.append(m)
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
        elif cache is not None:
            import datetime

            cache.record(m, datetime.datetime.now().isoformat(timespec="seconds"), secs)
    ran = len(MODULES) - len(artifact_skipped)
    print("=" * 56)
    if carried:
        # the split is stated, never folded: "N passed" where some were carried would be
        # the coverage-shaped silence PROVENANCE exists to prevent. live = everything not
        # carried, INCLUDING failures (the first form subtracted failed from live and
        # under-reported live runs by exactly the failure count -- review minor, fixed)
        live = ran - len(carried)
        print(
            f"{ran - len(failed)}/{ran} self-checks passed "
            f"({len(carried)} CARRIED on hash-verified unchanged inputs, "
            f"{sum(carried) / 60:.1f} min of live time not re-spent; "
            f"{live} run live of which {len(failed)} failed)"
        )
    else:
        print(f"{ran - len(failed)}/{ran} self-checks passed")
    if artifact_skipped:
        # Printed as its own line, not merged into the total, so "277/277" can never quietly mean
        # "276 ran and one was skipped".
        print(f"ARTIFACT-SKIPPED ({len(artifact_skipped)}): " + ", ".join(artifact_skipped))
    for line in _latch_report_lines(latch_left, latch_stray):
        print(line)
    tools_failed = run_tools_gates()
    if tools_failed:
        print("FAILED (tools gates): " + ", ".join(tools_failed))
    if failed:
        if carried:
            print(
                "CARRY-RUN FAIL PROTOCOL: rerun this battery WITHOUT --carry before "
                "diagnosing -- a carried-run red is not evidence until the full run "
                "confirms it (shared-home sequence coupling is a disclosed blind spot)."
            )
        print("FAILED: " + ", ".join(failed))
    return not failed and not tools_failed


def run_one_in_process(mod):
    """Run one module's self-check IN-PROCESS via runpy, capturing its output. Returns (ok, text).

    run_all() uses subprocesses, which is the better isolation and stays the default. This exists for
    one situation subprocesses cannot serve: a frozen PyInstaller build has no interpreter to spawn,
    so `python -m snn_data_lake.<mod>` is unavailable and the packaged app would otherwise have no way
    to verify that the core it bundled is actually runnable.

    sys.argv is isolated to [mod] so a module sees no stray caller flags -- several modules dispatch on
    argv (spike_telemetry_hub runs its CLI when given any, control_panel used to run --smoke on none),
    and leaking the parent's arguments would silently run the wrong entry point.

    NORD_HOME is isolated per call (_gate_home, added 2026-08-31): until then a check run
    through THESE runners hit the operator's REAL ~/.snn_data_lake -- the exact hazard
    run_all's _ENV exists to prevent -- and an E-STOP sidecar landed there uncleared (the
    pytest in-process sibling was never exposed: tests/conftest.py's _hermetic_home has
    isolated it since 2026-07-31). The gate covers home()-derived paths resolved DURING the
    call; a straggler thread or atexit handler outliving the call resolves the CALLER's env,
    so a caller wanting run-wide or process-wide isolation owns it (run_all_in_process spans
    its loop; the --in-process dispatch sets the env at process scope).
    """
    import io
    import runpy
    from contextlib import redirect_stderr, redirect_stdout

    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = [mod]
    try:
        with _gate_home(), redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
        return True, buf.getvalue()
    except SystemExit as e:  # a module may sys.exit(0) on success
        return (e.code in (0, None)), buf.getvalue()
    except BaseException as e:  # AssertionError / any self-check failure
        return False, f"{buf.getvalue()}\n{type(e).__name__}: {e}"
    finally:
        sys.argv = saved_argv


def run_all_in_process(emit=print):
    """Every CORE module's self-check, in-process. True iff all passed. See run_one_in_process.

    Carries the same E-STOP sidecar discipline as run_all (added 2026-08-31): pre-clear, a
    per-check _sweep_latch against the isolated home, and the shared report lines -- until
    then a latch left by a check run this way sat in the operator's real home, uncleared and
    unattributed. The WHOLE loop runs inside one _gate_home() span, so inter-check windows
    and threads outliving a single check still resolve the isolated home; atexit handlers
    fire after this function returns, which is the --in-process dispatch's process-scope set.
    If the arm stack cannot import (a frozen bundle without MuJoCo -- the case this runner
    exists for), the sweep is disabled LOUDLY and the checks still run: so101_arm is the only
    latch writer, so no in-process check can write one either in that configuration."""
    try:
        from snn_data_lake.brainstem.so101_arm import ESTOP_LATCH  # ONE spelling, in the envelope
    except Exception as e:  # pre-diff this import would have aborted the runner with no output
        ESTOP_LATCH = None
        emit(
            f"E-STOP sidecar sweep DISABLED: arm stack unimportable ({type(e).__name__}: {e}); "
            "no in-process check can write a latch without it (so101_arm is the only writer)"
        )
    latch = os.path.join(_HOME, ESTOP_LATCH) if ESTOP_LATCH else None
    latch_left, latch_stray = [], []
    failed = []
    with _gate_home():
        if latch and os.path.exists(latch):
            os.remove(latch)
            emit("GATE-HOME E-STOP SIDECAR from a previous run cleared before this in-process run")
        for m in CORE:
            w0 = time.time()
            ok, out = run_one_in_process(m)
            w1 = time.time()
            emit(f"{'PASS' if ok else 'FAIL'}  {m}")
            if latch:
                _sweep_latch(latch, m, w0, w1, latch_left, latch_stray)
            if not ok:
                failed.append(m)
                emit(out[-2000:])
    emit("=" * 56)
    emit(
        f"{len(CORE) - len(failed)}/{len(CORE)} core self-checks passed (in-process)"
        + (f"   FAILED: {', '.join(failed)}" if failed else "")
    )
    for line in _latch_report_lines(latch_left, latch_stray):
        emit(line)
    return not failed


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(MODULES))
        sys.exit(0)
    if "--in-process" in sys.argv:
        # PROCESS-scope isolation: atexit handlers registered by in-process checks fire after
        # every context manager has restored, so the env must still point at the gate home
        # when they run. This process exits right after, so no restore is owed.
        os.environ["NORD_HOME"] = _HOME
        sys.exit(0 if run_all_in_process() else 1)
    sys.exit(0 if run_all(carry="--carry" in sys.argv) else 1)
