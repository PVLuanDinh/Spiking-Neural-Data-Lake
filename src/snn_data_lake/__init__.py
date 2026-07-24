"""snn_data_lake -- a zero-dependency neuromorphic library organized by brain anatomy.

As of 3.0 the modules live in anatomical region subpackages -- cortex, hippocampus,
basal_ganglia, cerebellum, brainstem, neuromodulation, plasticity, and substrate (the
non-anatomical data-lake / telemetry / infra tier). REGIONS below is the authoritative map.

The flat pre-3.0 import paths stay valid: a lazy alias finder + module __getattr__ (installed
at the bottom) make `import snn_data_lake.lif_neuron` and `from snn_data_lake.lif_neuron import
LIFNeuron` resolve to the moved module (snn_data_lake.cortex.lif_neuron). See
snn_data_lake.brain_atlas for the rendered map and a back-compat self-check.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys

# region -> modules  (single source of truth for the anatomical layout)
REGIONS = {
    "cortex": [
        "rank_order_coding",
        "population_encoding",
        "population_vector",
        "multimodal_encoder",
        "spike_preprocessing",
        "theta_gamma_code",
        "spike_slicer",
        "spike_pooling",
        "lif_neuron",
        "dendritic_neuron",
        "izhikevich_neuron",
        "hodgkin_huxley",
        "alif_neuron",
        "divisive_normalization",
        "neuronal_avalanches",
        "cortical_microcircuit",
        "stochastic_resonance",
        "lca_sparse_coding",
        "predictive_coding",
        "is_snn_forward",
        "neural_sampling",
        "snn_classifier",
        "trained_brain",
        "learned_matcher",
        "tempotron",
        "paradigm_b_matcher",
        "paradigm_b_engine",
        "gesture_recognition",
        "nl_interface",
        "spiking_lm_serve",
        "vsa_reasoning",
        "prefrontal_executive",
        "live_brain",
        "interaction_learning",
        "recognizer_bank",
        "sensory_encoder_bank",
        "neuron_zoo",  # merges: recognizer ensemble / encoder bank / neuron population
        # higher cognition / self-model / self-improvement (v3.14+)
        "access_consciousness",
        "self_model",
        "metacognitive_monitor",
        "theory_of_mind",
        "reasoning_curriculum",
        "autotelic_goals",
        "quality_diversity",
        "self_improvement_suite",
        "wholistic_loop",
        "spiking_recurrent_cell",
        "autapse_neuron",
        "integration_measure",
        "agency_monitor",
        "affect_core",
        "crossmodal_binder",
        "executive_control",
        "time_cells",
        "task_set",
        "generative_self_model",  # C1: spiking clock / task-set-shift / learned generative self-model
        "recursive_tom",
        "counterfactual_sim",  # C1 batch-2: N-agent recursive ToM / counterfactual do()-operator simulation
        "learned_attention",  # C1 batch-3: learned spiking reticular attention gate (local plasticity)
        "learned_evc",  # C1 batch-4: learned expected-value-of-control (dopamine-gated cost/value)
        "second_order_access",  # C2 r2: LIF metacognition OF access reliability (knowing THAT it knows)
        "heterogeneous_microcircuit",  # Brain-Cog-adopted: LIF+Izhikevich+AdEx composed live under a local rule
        "percept_surprise",  # loop C2: online unsupervised Oja-PCA reconstruction-residual surprise (orienting drive)
        # torch-trained tiers with a zero-dep serve/integration path (import cleanly without torch)
        "pc_cortex_serve",
        "brain",
        "rsnn_encoder",
    ],
    "hippocampus": [
        "hippocampus",
        "grid_cells",
        "experience_replay",
        "sleep_replay",
        "ring_attractor",
        "growing_learner",
        "legendre_memory",
        "function_preserving_growth",
        "sleep_consolidation",
        "affective_memory",
        "contrastive_episode_analysis",
        "allocentric_map",
        "grid_place_map",
        "episodic_recall",
        "spiking_grid_can",  # C1: continuous-attractor SPIKING grid network (bump path integration)
        "episode_consolidation",  # C1 batch-4: systems-consolidation of real episodes (sparse attractor + sleep replay)
        "lake_consolidation",  # sleep-time prune of the symbolic durable lake (cap re-derivable bulk, keep learned)
    ],
    "basal_ganglia": [
        "evidence_accumulation",
        "r_stdp",
        "valence_stdp",
        "action_gating",
        "latent_q",
        "mlp_q",
    ],
    "amygdala": ["fear_conditioning", "affective_appraisal"],
    "cerebellum": [
        "forward_model",
        "spiking_pid",
        "synfire_chain",
        "wavefront_planner",
        "jepa_world_model",
        "hjepa_serve",
        "spiking_jepa",
        "gain_adaptation",
    ],
    "brainstem": [
        "reflex",
        "serial_arm",
        "arm_sim",
        "arm_config",
        "interpreter",
        "live_arm",
        "arm_bridge",
        "closed_loop",
        "autonomous_loop",
        "signal_loop",
        "cpg_oscillator",
    ],
    "thalamus": ["global_workspace", "reticular_gate", "reentrant_workspace", "connectome_router"],
    "hypothalamus": ["circadian_clock", "set_points", "homeostatic_drives"],
    "midbrain": [
        "dopamine_nuclei",
        "superior_colliculus",
        "sc_deep_layers",
        "multisensory_integration",
    ],
    "pons": ["pontine_relay", "reticular_formation"],
    "medulla": ["vital_nuclei", "interoceptive_arousal"],
    "epithalamus": ["habenula"],
    "olfactory": [],
    "claustrum": ["claustrum"],
    "neuromodulation": [
        "acetylcholine",
        "astrocyte_modulation",
        "cortisol",
        "locus_coeruleus",
        "raphe_serotonin",
        "intrinsic_motivation",
        "spiking_curiosity",
        "generalizing_novelty",
        "drive_reduction",
        "naca",
        "empowerment",
        "developmental_envelope",
        "affect_integrator",  # C1 batch-2: spiking neuromod fusion -> coupled affect + self-organized emotions
    ],
    "plasticity": [
        "bcm_plasticity",
        "homeostatic_scaling",
        "inhibitory_stdp",
        "intrinsic_plasticity",
        "short_term_plasticity",
        "developmental_pruning",
        "competitive_learning",
        "criticality_regulator",
        "meta_plasticity",
        "benna_fusi",
        "lifelong_consolidation",
        "homeostatic_plasticity_stack",
        "plasticity_rule_stack",
    ],
    "substrate": [
        "_mathutils",
        "spike_stream_sketch",
        "spike_cardinality",
        "reservoir_sampling",
        "lake_memory",
        "stream_sketch_panel",
        "sliding_window_count",
        "frequency_moment",
        "streaming_quantile",
        "minhash",
        "morris_counter",
        "entropy_coder",
        "proximity_index",
        "energy",
        "data_quality",
        "drift_monitor",
        "spike_telemetry_hub",
        "streaming_hub",
        "neuro_chip",
        "neuro_chip_stream",
        "nir_export",
        "sensor_csv_ingest",
        "esp32_encoder_path",
        "make_sensor_dataset",
        "demo",
        "control_panel",
        "brain_bundle",
        "reaction_diffusion_compute",
        "web_ingest",
        "lake_working_memory",
        "chat_memory",  # durable content-addressable chat memory on the spike lake (noise-filtered)
        "signal_calibrator",  # durable running-quantile of a live signal (Step-0 of data self-recalibration)
        "live_brain_tab",
        "nengo_bridge",
        "home",
        "binlog",
        "growable",
        "region_coherence",  # C1 batch-3: cross-region Kuramoto coherence + integration -> GWT ignition bias
        "region_wiring",  # the always-on daemon's faculty-wiring registry -> charge_all + honest coverage report
        "dormancy_gate",  # Policy 3: the no-silent-dormancy CI gate (reachable-or-EXEMPT over REGIONS)
        "replication_guard",  # Policy 4: the one-brain guard (canonical stack classes defined once)
        "basic_components",  # Brain-Cog-adopted composition interface: node + local-rule + connection palette
        "similarity_sized_growth",  # Brain-Cog SCA-adopted: novelty-sized recruit cap + frozen-expert reuse
    ],
}

# TORCH-TIER ANNEX -- region -> modules that import torch AT MODULE LOAD (so they cannot be imported
# in the zero-dep CI matrix). They are part of the anatomical map (region_of + the atlas count include
# them), but are NOT aliased into the zero-dep back-compat layer and are import-verified only when torch
# is present. Every OTHER torch-trained organ (pc_cortex_serve, brain, jepa/hjepa serve,
# neurogenesis) imports lazily and lives in REGIONS above. See brain_atlas for the gated verification.
TORCH_TIER = {
    "cortex": ["neurogenesis_engine", "language_acquisition"],
    "hippocampus": ["ng_snn"],
}

# old flat name -> "region.module". The hippocampus module shares its region's name, so it is
# reached through the real snn_data_lake.hippocampus package (which re-exports it) -- not aliased.
_MOVED = {m: f"{r}.{m}" for r, mods in REGIONS.items() for m in mods if m != r}


class _AliasLoader(importlib.abc.Loader):
    """Binds a legacy flat name to the real regional module (same object -- a true alias).

    create_module returns the already-imported regional module, but the import machinery's
    _init_module_attrs then stamps THIS alias spec onto that shared object -- which would corrupt
    the real module's __spec__ (its loader would become _AliasLoader, with no get_code, breaking
    runpy/get_code on the regional module). exec_module runs after that stamping, so we restore
    the real module's own spec/loader there.
    """

    def __init__(self, target):
        self._target = target
        self._real_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        if self._real_spec is not None:
            module.__spec__ = self._real_spec
            module.__loader__ = self._real_spec.loader


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves legacy `snn_data_lake.<flat>` imports to their new regional location."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + "."
        if not fullname.startswith(prefix):
            return None
        tail = fullname[len(prefix) :]
        if "." in tail or tail not in _MOVED:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(f"{__name__}.{_MOVED[tail]}"))


sys.meta_path.append(_AliasFinder())


def __getattr__(name):
    # supports `from snn_data_lake import lif_neuron` and `snn_data_lake.lif_neuron` attribute access
    if name in _MOVED:
        return importlib.import_module(f"{__name__}.{_MOVED[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
