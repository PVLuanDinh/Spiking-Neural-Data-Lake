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
        "rank_order_coding", "population_encoding", "population_vector", "multimodal_encoder",
        "spike_preprocessing", "theta_gamma_code", "spike_slicer", "adaptive_spiking",
        "cue_integration", "spike_pooling",
        "lif_neuron", "dendritic_neuron", "izhikevich_neuron", "hodgkin_huxley", "alif_neuron",
        "divisive_normalization", "wilson_cowan", "neuronal_avalanches", "stochastic_resonance",
        "lt_gate", "lca_sparse_coding", "predictive_coding", "is_snn_forward", "neural_sampling",
        "restricted_boltzmann",
        "snn_classifier", "trained_brain", "learned_matcher", "tempotron", "paradigm_b_matcher",
        "paradigm_b_engine", "gesture_recognition",
        "language_brain", "nl_interface", "brain_reasoning", "spiking_lm_serve", "vsa_reasoning",
    ],
    "hippocampus": [
        "hippocampus", "grid_cells", "htm_sequence_memory", "frequency_memory",
        "experience_replay", "sleep_replay", "ring_attractor", "growing_learner", "legendre_memory",
    ],
    "basal_ganglia": ["evidence_accumulation", "r_stdp", "valence_stdp"],
    "cerebellum": [
        "forward_model", "kalman_filter", "spiking_pid", "kuramoto_sync", "synfire_chain",
        "wavefront_planner",
    ],
    "brainstem": [
        "reflex", "serial_arm", "arm_sim", "arm_config", "interpreter", "live_arm", "arm_bridge",
        "closed_loop", "autonomous_loop", "signal_loop", "cpg_oscillator",
    ],
    "neuromodulation": ["acetylcholine", "astrocyte_modulation", "cortisol"],
    "plasticity": [
        "bcm_plasticity", "homeostatic_scaling", "inhibitory_stdp", "intrinsic_plasticity",
        "short_term_plasticity", "developmental_pruning",
    ],
    "substrate": [
        "_mathutils", "spike_stream_sketch", "spike_cardinality", "reservoir_sampling",
        "sliding_window_count", "frequency_moment", "streaming_quantile", "minhash",
        "morris_counter", "entropy_coder", "proximity_index", "energy", "data_quality",
        "drift_monitor", "spike_telemetry_hub", "streaming_hub", "neuro_chip", "neuro_chip_stream",
        "nir_export", "sensor_csv_ingest", "esp32_encoder_path", "make_sensor_dataset",
        "sensor_demo", "demo", "control_panel", "brain_bundle", "reaction_diffusion_compute",
        "web_ingest",
    ],
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
        tail = fullname[len(prefix):]
        if "." in tail or tail not in _MOVED:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(f"{__name__}.{_MOVED[tail]}"))


sys.meta_path.append(_AliasFinder())


def __getattr__(name):
    # supports `from snn_data_lake import lif_neuron` and `snn_data_lake.lif_neuron` attribute access
    if name in _MOVED:
        return importlib.import_module(f"{__name__}.{_MOVED[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
