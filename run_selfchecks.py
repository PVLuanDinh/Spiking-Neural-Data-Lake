"""
v0.50 — single canonical runner for every pure-stdlib self-check.

The CI module list used to live hand-typed in BOTH .github/workflows/ci.yml and
docs/RUNNING.md — add or rename a module and the two silently drift. This is the ONE
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

# children print Unicode (φ, −, …); force UTF-8 so a Windows cp1252 console can't crash them.
# PYTHONPATH="src" so `snn_data_lake` (the package) + research/ modules importing it resolve.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": "src"}

# CORE — the product: arm-loop + data-lake, clone-and-run (root). As of 3.0 organized by
# anatomical brain region (src/snn_data_lake/<region>/<module>.py); run as
# `-m snn_data_lake.<region>.<module>`. snn_data_lake.brain_atlas renders + verifies the map.
CORE = [
    "brain_atlas",
    "connectome",
    # cortex — sensing/encoding, cortical neuron models, microcircuit dynamics, readout, language
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
    "cortex.lca_sparse_coding",
    "cortex.predictive_coding",
    "cortex.pc_cortex_serve",
    "cortex.is_snn_forward",
    "cortex.neural_sampling",
    "cortex.snn_classifier",
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
    "cortex.metacognitive_monitor",
    "cortex.wholistic_loop",
    "cortex.autotelic_goals",
    "cortex.quality_diversity",
    "cortex.self_improvement_suite",
    "cortex.theory_of_mind",
    "cortex.agency_monitor",
    "cortex.affect_core",
    "cortex.crossmodal_binder",  # W6: categorical emotion + 2nd-order ToM; cross-modal super-additive binding
    "cortex.executive_control",  # W8: ACC conflict + EVC + tree/sequence planning
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
    # hippocampus — memory formation, spatial coding, replay
    "hippocampus.grid_cells",
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
    "hippocampus.episode_consolidation",  # C1 batch-4: systems-consolidation of real episodes (sparse attractor + sleep replay)
    "hippocampus.lake_consolidation",  # sleep-time PRUNE of the symbolic durable lake (cap re-derivable bulk, keep learned) -> live + bounded
    # basal ganglia — action selection, reward-modulated learning, decision, Go/NoGo gating
    "basal_ganglia.evidence_accumulation",
    "basal_ganglia.r_stdp",
    "basal_ganglia.valence_stdp",
    "basal_ganglia.action_gating",
    "basal_ganglia.latent_q",
    "basal_ganglia.mlp_q",
    # amygdala — aversive/threat learning (CS-US fear conditioning)
    "amygdala.fear_conditioning",
    "amygdala.affective_appraisal",  # C1: learned multi-dim affective appraisal (three-factor)
    # cerebellum — motor coordination, timing, forward/inverse models, oscillators
    "cerebellum.forward_model",
    "cerebellum.spiking_pid",
    "cerebellum.synfire_chain",
    "cerebellum.wavefront_planner",
    "cerebellum.jepa_world_model",
    "cerebellum.hjepa_serve",
    "cerebellum.spiking_jepa",
    "cerebellum.gain_adaptation",
    # brainstem + spinal cord — reflexes, actuation, sensorimotor loops, motor pattern generation
    # (the safety envelope lives here)
    "brainstem.reflex",
    "brainstem.serial_arm",
    "brainstem.arm_sim",
    "brainstem.arm_hardware",  # torque budget for the arm being built (asserts the shoulder shortfall)
    "brainstem.arm_config",
    "brainstem.interpreter",
    "brainstem.live_arm",
    "brainstem.arm_bridge",
    "brainstem.closed_loop",
    "brainstem.autonomous_loop",
    "brainstem.signal_loop",
    "brainstem.cpg_oscillator",
    # thalamus — the relay + global-broadcast hub (GWT) + reticular attentional gate
    "thalamus.global_workspace",
    "thalamus.reticular_gate",
    "thalamus.reentrant_workspace",  # W8: de-saturated competitive ignition + re-entry
    "thalamus.connectome_router",  # C1 batch-3: live signed/delayed spike propagation over the connectome graph
    # hypothalamus — homeostatic timing (the SCN circadian clock)
    "hypothalamus.circadian_clock",
    "hypothalamus.set_points",
    "hypothalamus.homeostatic_drives",  # W8: drives as percepts
    # midbrain — dopamine source nuclei (VTA/SNc RPE) + superior-colliculus orienting + deep gaze layers
    "midbrain.dopamine_nuclei",
    "midbrain.superior_colliculus",
    "midbrain.sc_deep_layers",
    "midbrain.multisensory_integration",
    # pons — cortico-ponto-cerebellar relay + the ascending reticular activating system (ARAS)
    # medulla — the autonomic vital nuclei (respiratory / cardiac CPG) + interoceptive arousal
    "medulla.vital_nuclei",
    "medulla.interoceptive_arousal",
    # epithalamus — the lateral habenula (anti-reward / negative RPE)
    "epithalamus.habenula",
    # olfactory — bulb decorrelation + piriform association
    # claustrum — cortical synchrony / salience hub
    "claustrum.claustrum",
    # neuromodulation — diffuse chemical modulators + glia (LC noradrenaline, raphe serotonin)
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
    # plasticity — Hebbian/STDP, homeostatic, developmental learning rules
    "plasticity.bcm_plasticity",
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
    # substrate — the spiking data-lake, telemetry, and infra (not a brain region)
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
    "substrate.reaction_diffusion_compute",
    "substrate.web_ingest",
    "substrate.lake_working_memory",
    "substrate.chat_memory",
    "substrate.signal_calibrator",
    "substrate.home",
    "substrate.binlog",
    "substrate.growable",
    "substrate.region_wiring",  # the always-on daemon's faculty-wiring registry -> charge_all + honest coverage report
    "substrate.brain_telemetry",  # snapshot contract + recorder + /api/state (no-orphan-panel gate)
    "substrate.brain_geometry",  # anatomical layout (hand-authored coords, REAL connectome tracts)
    "substrate.cockpit",  # THE one frontend: page + contract gate + queued-not-raced chat turn
    "substrate.dormancy_gate",  # Policy 3: assert every brain module is reachable from a live path or EXEMPT
    "substrate.replication_guard",  # Policy 4: assert one canonical Brain/LiveBrain/AutonomousArm (no replica)
    "substrate.import_manifest",  # Policy 3: assert every reasoning/ module the live brain lazily imports still imports
    "substrate.method_dormancy_gate",  # Policy 3 amendment: assert no update rule on a live module lacks a live caller
    "substrate.region_coherence",  # C1 batch-3: cross-region Kuramoto coherence + integration -> ignition bias
    "substrate.basic_components",  # Brain-Cog-adopted composition interface (node + local-rule + connection palette)
    "substrate.similarity_sized_growth",  # Brain-Cog SCA-adopted: novelty-sized recruit cap + frozen-expert reuse
]
# RESEARCH — benchmarks / storage / KG demos (research/), run with root on PYTHONPATH.
RESEARCH = [
    "reasoning.spiking_storage_prototype",
    "reasoning.test_prototype",
    "reasoning.hopfield_baseline",
    "reasoning.dense_associative_memory",
    "reasoning.sparse_distributed_memory",
    "reasoning.fly_hash",
    "reasoning.spiking_som",
    "reasoning.snn_moe_classifier",
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
    "reasoning.srm_neuron",
    "reasoning.nef_compute",
    "reasoning.nef_integrator",
    "reasoning.stats_bounds",
    "reasoning.loop_latency",
    "reasoning.delay_learning",
    "reasoning.eprop",
    "reasoning.polychronization",
    "reasoning.sn_p_system",
    "reasoning.oja_pca",
    "reasoning.resonate_and_fire",
    "reasoning.free_energy_control",
    "reasoning.pc_active_inference",
    "reasoning.dishbrain_learning",
    "reasoning.adex_neuron",
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
    "reasoning.open_world_loop",
    "reasoning.open_meta_loop",
    "reasoning.model_builder",
    # the 5 Gf inducers COMBINED (not merged) into one file: reasoning.gf
    "reasoning.gf",
    # HONEST held-out RPM bench (rendered rasters -> spiking component perception -> derived cells),
    # replacing the old grammar self-test: 80% measured vs 12.5% chance, broken controls near chance.
    # (reasoning.arc_diagnostic is deliberately NOT here -- a ~27 s offline instrument, covered by
    # tests/test_arc_diagnostic.py; the smoke gate stays fast.)
    "reasoning.real_rpm_bench",
    "reasoning.vsa_structure_mapping",  # C1 batch-4: Gentner relational structure-mapping analogy (VSA/resonator)
    "reasoning.reasoning_cortex",  # MoE router: ~8 dead reasoning studies made live on the converse path
    "reasoning.derivation",  # multi-hop transitive DEDUCTION (proof-carrying BFS over taught facts)
    "reasoning.reasoning_compose",  # MULTI-STEP composition over the MoE router (bounded expert grammar)
    "reasoning.pc_abduction",  # ABDUCTION (min-free-energy cause selection) + Pearl 3-step counterfactual
    "reasoning.rose_language",  # ROSE fluent-language pipeline: VSA constituency + grounded semantics + SLDA competence
    "reasoning.rpm_solver",  # spiking NVSA Raven's-matrix solver: role-filler + example->rule induction (Gf)
    "reasoning.raven_bench",  # honest RPM benchmark (synthetic tasks + failing random control)
    "reasoning.vsa_synthesize",  # OUT-OF-GRAMMAR Gf: synthesizes novel rules as VSA spike-op programs (no fixed menu)
    "reasoning.semantic_memory",  # W6: VSA meaning-based recall (paraphrase-invariant, beyond the lexical crc32 gate)
    "reasoning.pragmatics",  # W8: speech-act recognition + Gricean implicature + indirect-request detection
    "reasoning.brain_native_selfbuild",  # OPEN-ENDED brain-native self-build: three-factor-learned de-hardcode proposer (retired the numeric tuner)
    "reasoning.impact_analysis",  # pre-mutation blast radius (harness LSP principle, native ast + graphify) -> self-build prefers surgical changes
    "reasoning.ar_spiking_lm_serve",  # forward-only Spiking-RWKV serve: order-sensitive logprob (the BLiMP lever)
    "reasoning.eprop_lm_serve",  # BRAIN-NATIVE e-prop recurrent-spiking LM serve: order-sensitive logprob (no backprop)
    "reasoning.eprop_lm_alif_serve",  # deep-ALIF e-prop LM serve: adaptive-threshold long-range, order-sensitive logprob
    "reasoning.self_improve",
    "reasoning.godel_machine",
    "reasoning.self_mod_harness",
    "reasoning.self_build_loop",
    "reasoning.content_signature",
    # the two OUTSIDE yardsticks the depth ceilings are measured against. Both SKIP-with-notice when
    # their dataset is not downloaded, so the zero-dep CI matrix stays green either way.
    "reasoning.arc_bench",  # ARC-AGI: exact-match pass@2 over 800 tasks; controls must score ~0
    "reasoning.blimp_bench",  # BLiMP: 67 paradigms x 1000 minimal pairs; chance 50%, human 88.6%
    "reasoning.arc_solver",  # test-time program synthesis over 2-D grids (object DSL + iterative deepening)
    "reasoning.arc_scene",  # Gf Phase-2 perceptual VOCABULARY: extensible attribute/relation registry (palette/morph/topo/shape-eq/align) over components() -- measurement only, feeds the Phase-3 mint
    "reasoning.arc_mint",  # Gf Phase-3 MINT: hypothesize out-of-closure primitives from arc_scene + memorization-resistant functional induction (LOO-validated); held-out metric
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
    "cortex.brain",
    "cortex.neurogenesis_engine",
    "cortex.language_acquisition",
    "reasoning.nightly_skills",
    "reasoning.schema_induction",
]
# NENGO-GATED -- the Nengo bridge (substrate.nengo_bridge) exposes the NEF/LIF/spiking substrate to
# Nengo (sim + NengoLoihi deploy target); it imports `nengo` (optional). Run only where nengo is
# importable, SKIPPED-with-notice otherwise (its own __main__ also self-SKIPs). Zero-dep core unaffected.
NENGO_GATED = ["substrate.nengo_bridge"]


def _torch_ok():
    # TORCH TIER ON BY DEFAULT: run the torch-gated modules whenever torch is importable. MANUAL
    # OFF-SWITCH: env NORD_TORCH=0 forces the torch tier off (skip them) even if torch is installed.
    if os.environ.get("NORD_TORCH") == "0":
        return False
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


_TORCH = _torch_ok()
_NENGO = _nengo_ok()
MODULES = CORE + (TORCH_GATED if _TORCH else []) + (NENGO_GATED if _NENGO else []) + RESEARCH


def run_all():
    if not _TORCH:
        for m in TORCH_GATED:
            print(f"SKIP  {m}  (torch not importable -- zero-dep matrix stays green)")
    if not _NENGO:
        for m in NENGO_GATED:
            print(f"SKIP  {m}  (nengo not importable -- zero-dep matrix stays green)")
    failed = []
    for m in MODULES:
        cmd = [sys.executable, "-m", f"snn_data_lake.{m}"]  # every module runs as a package module
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_ENV)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {m}")
        if not ok:
            failed.append(m)
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
    print("=" * 56)
    print(f"{len(MODULES) - len(failed)}/{len(MODULES)} self-checks passed")
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
