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

# CORE — the product: arm-loop + data-lake, clone-and-run (root).
CORE = [
    "_mathutils",
    "snn_classifier", "trained_brain", "brain_bundle", "spiking_lm_serve", "nir_export", "spike_telemetry_hub", "streaming_hub", "short_term_plasticity",
    "data_quality", "drift_monitor",
    "lif_neuron", "dendritic_neuron", "spike_slicer", "is_snn_forward", "lt_gate", "spike_pooling", "ring_attractor", "grid_cells",
    "htm_sequence_memory", "theta_gamma_code", "legendre_memory", "predictive_coding", "spike_stream_sketch", "spike_cardinality", "reservoir_sampling", "sliding_window_count", "frequency_moment", "streaming_quantile", "minhash", "morris_counter", "entropy_coder",
    "paradigm_b_matcher", "paradigm_b_engine", "vsa_reasoning", "spike_preprocessing", "adaptive_spiking", "proximity_index", "energy",
    "neuro_chip", "neuro_chip_stream",
    "signal_loop", "learned_matcher", "rank_order_coding", "tempotron", "lca_sparse_coding", "evidence_accumulation",
    "reflex", "valence_stdp", "r_stdp", "bcm_plasticity", "homeostatic_scaling", "inhibitory_stdp",
    "divisive_normalization",
    "intrinsic_plasticity", "izhikevich_neuron", "hodgkin_huxley", "alif_neuron", "cortisol", "acetylcholine", "astrocyte_modulation",
    "neural_sampling", "restricted_boltzmann",
    "interpreter", "closed_loop", "make_sensor_dataset", "sensor_demo",
    "arm_sim", "arm_config", "population_encoding", "population_vector", "multimodal_encoder", "cue_integration", "growing_learner",
    "autonomous_loop", "hippocampus", "frequency_memory", "experience_replay", "sleep_replay", "nl_interface", "brain_reasoning", "language_brain", "sensor_csv_ingest", "esp32_encoder_path",
    "gesture_recognition", "live_arm", "serial_arm", "arm_bridge", "spiking_pid", "cpg_oscillator", "kuramoto_sync", "wilson_cowan", "neuronal_avalanches", "synfire_chain", "stochastic_resonance", "forward_model", "kalman_filter", "demo",
    "reaction_diffusion_compute", "wavefront_planner",
    "control_panel",
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
