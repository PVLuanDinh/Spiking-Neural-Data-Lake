"""
v0.50 — single canonical runner for every pure-stdlib self-check.

The CI module list used to live hand-typed in BOTH .github/workflows/ci.yml and
docs/RUNNING.md — add or rename a module and the two silently drift. This is the ONE
source of truth: CI runs `python run_selfchecks.py`, docs point here. Each module's
`if __name__ == "__main__"` block is its own assert-based test; this runs them as
subprocesses (isolation) and fails loudly if any non-zero exits.

All ZERO-DEP (pure stdlib). Two tiers: CORE = the product package (src/snn_data_lake/, run as
`-m snn_data_lake.<mod>`); RESEARCH = benchmarks / storage / KG demos (research/). Both CI-green;
the MNIST/GPU/cloud scripts need torch/beam and are excluded by design.

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
    "brain_atlas", "connectome",
    # cortex — sensing/encoding, cortical neuron models, microcircuit dynamics, readout, language
    "cortex.rank_order_coding", "cortex.population_encoding", "cortex.population_vector", "cortex.multimodal_encoder",
    "cortex.spike_preprocessing", "cortex.theta_gamma_code", "cortex.spike_slicer", "cortex.adaptive_spiking",
    "cortex.cue_integration", "cortex.spike_pooling",
    "cortex.lif_neuron", "cortex.dendritic_neuron", "cortex.izhikevich_neuron", "cortex.hodgkin_huxley", "cortex.alif_neuron",
    "cortex.divisive_normalization", "cortex.wilson_cowan", "cortex.neuronal_avalanches", "cortex.stochastic_resonance",
    "cortex.lt_gate", "cortex.lca_sparse_coding", "cortex.predictive_coding", "cortex.is_snn_forward", "cortex.neural_sampling",
    "cortex.restricted_boltzmann",
    "cortex.snn_classifier", "cortex.trained_brain", "cortex.learned_matcher", "cortex.tempotron", "cortex.paradigm_b_matcher",
    "cortex.paradigm_b_engine", "cortex.gesture_recognition",
    "cortex.language_brain", "cortex.nl_interface", "cortex.brain_reasoning", "cortex.spiking_lm_serve", "cortex.vsa_reasoning",
    # hippocampus — memory formation, spatial coding, replay
    "hippocampus.hippocampus", "hippocampus.grid_cells", "hippocampus.htm_sequence_memory", "hippocampus.frequency_memory",
    "hippocampus.experience_replay", "hippocampus.sleep_replay", "hippocampus.ring_attractor", "hippocampus.growing_learner",
    "hippocampus.legendre_memory",
    # basal ganglia — action selection, reward-modulated learning, decision, Go/NoGo gating
    "basal_ganglia.evidence_accumulation", "basal_ganglia.r_stdp", "basal_ganglia.valence_stdp",
    "basal_ganglia.action_gating",
    # amygdala — aversive/threat learning (CS-US fear conditioning)
    "amygdala.fear_conditioning",
    # cerebellum — motor coordination, timing, forward/inverse models, oscillators
    "cerebellum.forward_model", "cerebellum.kalman_filter", "cerebellum.spiking_pid", "cerebellum.kuramoto_sync",
    "cerebellum.synfire_chain", "cerebellum.wavefront_planner",
    # brainstem + spinal cord — reflexes, actuation, sensorimotor loops, motor pattern generation
    # (the safety envelope lives here)
    "brainstem.reflex", "brainstem.serial_arm", "brainstem.arm_sim", "brainstem.arm_config", "brainstem.interpreter",
    "brainstem.live_arm", "brainstem.arm_bridge", "brainstem.closed_loop", "brainstem.autonomous_loop",
    "brainstem.signal_loop", "brainstem.cpg_oscillator",
    # thalamus — the relay + global-broadcast hub (GWT) + reticular attentional gate
    "thalamus.global_workspace", "thalamus.reticular_gate",
    # hypothalamus — homeostatic timing (the SCN circadian clock)
    "hypothalamus.circadian_clock",
    # midbrain — dopamine source nuclei (VTA/SNc RPE) + superior-colliculus orienting
    "midbrain.dopamine_nuclei", "midbrain.superior_colliculus",
    # pons — cortico-ponto-cerebellar relay + the ascending reticular activating system (ARAS)
    "pons.pontine_relay", "pons.reticular_formation",
    # epithalamus — the lateral habenula (anti-reward / negative RPE)
    "epithalamus.habenula",
    # olfactory — bulb decorrelation + piriform association
    "olfactory.olfactory_bulb", "olfactory.piriform_cortex",
    # claustrum — cortical synchrony / salience hub
    "claustrum.claustrum",
    # neuromodulation — diffuse chemical modulators + glia (LC noradrenaline, raphe serotonin)
    "neuromodulation.acetylcholine", "neuromodulation.astrocyte_modulation", "neuromodulation.cortisol",
    "neuromodulation.locus_coeruleus", "neuromodulation.raphe_serotonin",
    # plasticity — Hebbian/STDP, homeostatic, developmental learning rules
    "plasticity.bcm_plasticity", "plasticity.homeostatic_scaling", "plasticity.inhibitory_stdp", "plasticity.intrinsic_plasticity",
    "plasticity.short_term_plasticity", "plasticity.developmental_pruning",
    # substrate — the spiking data-lake, telemetry, and infra (not a brain region)
    "substrate._mathutils", "substrate.spike_stream_sketch", "substrate.spike_cardinality", "substrate.reservoir_sampling",
    "substrate.sliding_window_count", "substrate.frequency_moment", "substrate.streaming_quantile", "substrate.minhash",
    "substrate.morris_counter", "substrate.entropy_coder", "substrate.proximity_index", "substrate.energy",
    "substrate.data_quality", "substrate.drift_monitor", "substrate.spike_telemetry_hub", "substrate.streaming_hub",
    "substrate.neuro_chip", "substrate.neuro_chip_stream", "substrate.nir_export", "substrate.sensor_csv_ingest",
    "substrate.esp32_encoder_path", "substrate.make_sensor_dataset", "substrate.sensor_demo", "substrate.demo",
    "substrate.control_panel", "substrate.brain_bundle", "substrate.reaction_diffusion_compute",
    "substrate.web_ingest",
]
# RESEARCH — benchmarks / storage / KG demos (research/), run with root on PYTHONPATH.
RESEARCH = [
    "research/spiking_storage_prototype", "research/test_prototype", "research/hopfield_baseline",
    "research/dense_associative_memory", "research/sparse_distributed_memory", "research/fly_hash",
    "research/spiking_som",
    "research/snn_moe_classifier",
    "research/temporal_coding_storage", "research/spike_compression",
    "research/event_quadtree_compression", "research/index_compression", "research/spike_bloom",
    "research/spiking_ssm",
    "research/spike_kg",
    "research/resonator_network", "research/mcr_hdc", "research/bayesian_snn", "research/nmnist_ingest",
    "research/srm_neuron", "research/nef_compute", "research/nef_integrator", "research/stats_bounds",
    "research/loop_latency",
    "research/delay_learning", "research/eprop",
    "research/polychronization", "research/sn_p_system", "research/oja_pca", "research/resonate_and_fire",
    "research/free_energy_control", "research/dishbrain_learning", "research/adex_neuron", "research/triplet_stdp",
    "research/cellular_neural_network", "research/random_boolean_network", "research/neural_heterogeneity",
    "research/burst_multiplexing", "research/stdp_wta_bayes", "research/sorn",
    "research/resume_learning", "research/weight_perturbation",
    "research/make_results_plot",
]
MODULES = CORE + RESEARCH


def run_all():
    failed = []
    for m in MODULES:
        cmd = ([sys.executable, f"{m}.py"] if m.startswith("research/")   # research/: run as a script
               else [sys.executable, "-m", f"snn_data_lake.{m}"])         # CORE: run as a package module
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


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(MODULES))
        sys.exit(0)
    sys.exit(0 if run_all() else 1)
