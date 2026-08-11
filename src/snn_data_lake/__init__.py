"""snn_data_lake -- a neuromorphic library organized by brain anatomy.

REQUIRES THE FREE-THREADED (PEP 703) INTERPRETER. Owner decision 2026-08-05: the brain runs on
CPython 3.14 built with --disable-gil (ABI tag cp314t), and `_require_free_threaded()` below
REFUSES TO IMPORT under a GIL-enabled build. `pyproject.toml` carries `requires-python = ">=3.14"`,
which is the most the packaging metadata can say -- free-threading is a BUILD, not a language
level, so there is no marker for it and the check has to be at runtime.

The guard is loud on purpose. A silent fallback to the GIL build is the failure this project would
NOT notice: every self-check would pass, single-threaded, the parallelism would simply be absent,
and the only symptom would be a performance number nobody was watching. That is the same shape as
the defects this tree keeps finding -- a thing that looks live, passes its gate, and does nothing.
Verified before the floor moved: torch 2.13.0, mujoco 3.11.0 and numpy 2.5.1 all publish
cp314t win_amd64 wheels, and a same-seed 25-tick brain produced a BYTE-IDENTICAL state hash under
both builds, so the determinism invariants survive removing the GIL.

Core dependencies are torch, numpy and mujoco. This line described the package as
"zero-dependency" until 2026-08-04, fifteen days after that contract was abolished project-wide
(2026-07-20) -- the single most-read description of the package advertised a property it no
longer had. Per P7, any dependency that helps this become a phenomenal brain is accepted.


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
import os
import sys
import sysconfig


def _require_free_threaded():
    """Refuse to import under a GIL-enabled interpreter. See the module docstring for WHY loud.

    WHY THIS IS A RUNTIME CHECK AND NOT PACKAGING METADATA: free-threading is a BUILD of CPython
    (PEP 703, `--disable-gil`, ABI tag cp314t), not a language level, so `requires-python` cannot
    express it. `pip install` on a GIL-enabled 3.14 succeeds and every self-check passes -- the
    parallelism is simply absent, silently. This project's recurring defect is exactly that shape,
    so the requirement is enforced where it can actually be observed.

    ESCAPE HATCH, deliberate and narrow: `NORD_ALLOW_GIL=1` downgrades the refusal to a stderr
    warning. It exists because the SNN work is the reason for free-threading, not the whole tree --
    a contributor bisecting an unrelated bug on a GIL build should not be blocked, and CI may need
    to prove the guard itself fires. It is NOT a supported runtime mode: the brain's parallel paths
    fall back to serial under it, so any timing measured with it set is measuring the wrong thing.

    REACHABLE RED, constructed and run 2026-08-05: executing this under the standard (GIL-enabled)
    3.14 build raises RuntimeError; under the cp314t build it returns silently. Both were observed.
    """
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    msg = (
        "snn_data_lake requires the FREE-THREADED CPython build (PEP 703, ABI tag cp314t).\n"
        f"  running: {sys.version.splitlines()[0]}\n"
        f"  Py_GIL_DISABLED={sysconfig.get_config_var('Py_GIL_DISABLED')!r} "
        f"(1 = free-threaded, 0/None = GIL build)\n"
        "  install:  uv python install 3.14+freethreaded\n"
        "            <that python> -m venv .venv-lake && .venv-lake/Scripts/pip install -e '.[dev]'\n"
        "  A GIL build imports and passes every self-check with the parallelism silently absent,\n"
        "  which is why this refuses instead of warning. Set NORD_ALLOW_GIL=1 to downgrade to a\n"
        "  warning for unrelated work -- never for a timing measurement."
    )
    if os.environ.get("NORD_ALLOW_GIL") == "1":
        print(
            f"[snn_data_lake] WARNING: GIL build; parallel paths run SERIAL.\n{msg}",
            file=sys.stderr,
        )
        return
    raise RuntimeError(msg)


_require_free_threaded()

# region -> modules  (single source of truth for the anatomical layout)
REGIONS = {
    "cortex": [
        "rank_order_coding",
        "population_encoding",
        "multimodal_encoder",
        "spike_preprocessing",
        "theta_gamma_code",
        "lif_neuron",
        "dendritic_neuron",
        "izhikevich_neuron",
        "hodgkin_huxley",
        "alif_neuron",
        "divisive_normalization",
        "neuronal_avalanches",
        "cortical_microcircuit",
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
        "autapse_neuron",
        # PROMOTED from reasoning/ 2026-08-04: the three neuron models `neuron_zoo`
        # composes. They fed a GOVERNED module from outside the gate.
        "adex_neuron",
        "resonate_and_fire",
        "srm_neuron",
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
        "heterogeneous_microcircuit",  # Brain-Cog-adopted: LIF+Izhikevich+AdEx worked EXAMPLE of the basic_components palette (not on the live tick)
        "percept_surprise",  # loop C2: online unsupervised Oja-PCA reconstruction-residual surprise (orienting drive)
        "kb_ranking",  # PROMOTED from reasoning/ 2026-08-11 (#10 batch-1): open-vocab DistMult KB completion head
        "corpus_stream",  # PROMOTED 2026-08-11 (#10 family C2): real-scale corpus/CMUdict ingestion for the language loop
        "world_kb_stream",  # PROMOTED 2026-08-11 (#10 family C2b): the curated world-KB (605 real triples) + masked-slot adapter
        # torch-trained tiers with a zero-dep serve/integration path (import cleanly without torch)
        "pc_cortex_serve",
        "brain",
        "rsnn_encoder",
    ],
    "hippocampus": [
        "grid_cells",
        "experience_replay",
        "sleep_replay",
        "ring_attractor",
        "growing_learner",
        "function_preserving_growth",
        "sleep_consolidation",
        "affective_memory",
        "contrastive_episode_analysis",
        "allocentric_map",
        "grid_place_map",
        "episodic_recall",
        "spiking_grid_can",  # C1: continuous-attractor SPIKING grid network (bump path integration)
        "episode_consolidation",  # C1 batch-4: systems-consolidation of real episodes (sparse attractor + sleep replay)
        "content_signature",  # PROMOTED 2026-08-11 (#10 family C1): the 64-dim crc32 content key for lake/skill routing
        "lake_consolidation",  # sleep-time prune of the symbolic durable lake (cap re-derivable bulk, keep learned)
        # Promoted from reasoning/ 2026-08-04 after an in-place P7/P1/P8 redesign. FACTORED
        # k-WTA associative memory: the P sparse patterns ARE the memory (O(P*k)), so there is
        # no N x N weight matrix to store. WIRED AND CHARGED, measured not assumed -- neutering
        # S.recall to a pass-through costs its live consumer (cortex/recognizer_bank.py:233
        # StorageRecognizer) 3 of its 64 out-of-distribution rejections, 0.844 -> 0.797. That is
        # P6 one-way as well: a hippocampus faculty whose output moves a cortex faculty's output.
        # Honest qualifier: the bank builds lazily and fires only on an explicit "recognize:"
        # query, so the path is charged but query-gated, not exercised on an ordinary tick.
        "spiking_storage_prototype",  # factored k-WTA associative memory; denoising content-addressable recall
    ],
    "basal_ganglia": [
        "evidence_accumulation",
        "r_stdp",
        "valence_stdp",
        "action_gating",
        "latent_q",
        "mlp_q",
        # Promoted from reasoning/ 2026-08-04. Both belong in basal_ganglia because both LEARN FROM
        # OUTCOME: self_build_loop is the correctness-gated apply/rollback path, and
        # brain_native_selfbuild picks WHICH candidate to propose using the brain's own three-factor
        # RSTDP rule driven by the gate verdict (kept +1 / rolled_back -1), which is reward-modulated
        # action selection in the literal sense -- not a census, not a heuristic ranking.
        #
        # LIVE, and the honest limit of that claim: brain.py::_enter_wake reaches both once every
        # SELFBUILD_EVERY=8 nights, but ONLY under NORD_SELFMOD=1 (the daemon sets it, CI does not),
        # so under `pytest` / `run_selfchecks.py` neither fires and CHARGE IS UNMEASURED. Registered
        # as reachable-and-operator-gated, NOT as charge-proven.
        #
        # brain_native_selfbuild's learned weights moved INTO the one lake under _T_SELFBUILD as part
        # of this promotion (P4b) -- they were in <repo>/data/selfbuild_state.json, a second durable
        # store. Cross-process round-trip proven: writer pid wrote w=0.77/0.11, a separate reader
        # process read them back after the writer exited.
        "self_build_loop",
        "brain_native_selfbuild",
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
        # PROMOTED from reasoning/ 2026-08-04 (owner directive: every build is promoted into the
        # governed tier the moment it lands; reasoning/ is outside all three P3 gates, which is
        # where dormancy hides). Cerebellum is the anatomically correct home -- this is the timing
        # structure, and synfire_chain.py:37 already cross-references it to distinguish zero-lag
        # synchrony from delay-matched convergence.
        "polychronization",
    ],
    "brainstem": [
        "reflex",
        "serial_arm",
        "so101_arm",  # the MuJoCo SO-101 gate; REPLACED arm_sim (specs/004)
        "arm_hardware",  # physical-arm build TARGET: spec + static torque budget + the sim-to-real gap
        "arm_config",
        "interpreter",
        "live_arm",
        "arm_bridge",
        "closed_loop",
        "autonomous_loop",
        "signal_loop",
        "cpg_oscillator",
    ],
    "thalamus": [
        "global_workspace",
        "reticular_gate",
        "reentrant_workspace",
        "connectome_router",
        "gwt_deep",
    ],
    "hypothalamus": ["circadian_clock", "set_points", "homeostatic_drives"],
    "midbrain": [
        "dopamine_nuclei",
        "superior_colliculus",
        "sc_deep_layers",
        "multisensory_integration",
    ],
    "pons": [],
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
        # PROMOTED from reasoning/ 2026-08-04, AFTER its P7 redesign (owner rule: redesign
        # first, promote second). Oja (1982) is a HEBBIAN RULE with a normalizing decay term,
        # w += lr*y*(x - y*w): no membrane, no threshold, no spike. A plasticity rule, not a
        # neuron model. Sole governed importer: cortex/percept_surprise.py.
        #
        # P7: the compromise was at the BOUNDARY -- percept_surprise held a numpy residual and
        # called .tolist() per sample, at in_dim=2312, purely so a list-based OjaNeuron could
        # eat it. Both sides are numpy now. NOT bit-exact (pairwise vs left-to-right sums):
        # measured max |dw| = 1.47e-15, relative error 3.8e-16 (one ULP of float64), and the
        # consumer's four reported statistics are unchanged (first20 0.2218, last20 0.0442,
        # offdist 0.2036, novelty_corr 0.2594).
        #
        # NOT CLAIMED LIVE. docs/coverage/artifacts/surprise_60.txt records the neuter-A/B:
        # the orienting signal changed 0 of 60 act/hold decisions. Promotion makes it VISIBLE
        # to the P3 gates; it does not make it charged, and P3 forbids force-wiring it.
        "oja_pca",
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
        # PROMOTED from reasoning/ 2026-08-07 as specs/012's MANDATED BUILD (owner call: finish
        # the GeNN port). The tree's only GPU plasticity codegen: the CPU-verified v0.17
        # burst+x_tar rule as GeNN C++. Governance lands BEFORE the finish so the port's
        # completion happens inside the gates; dormancy disposition = EXEMPT (opt-in GPU
        # training harness, see dormancy_gate).
        "snn_mnist_stdp_genn",
    ],
    "substrate": [
        # PROMOTED 2026-08-04: P2 merge of training/device.py (GPU-first, measured) with the
        # retired reasoning/torch_device.py (had the users + enable()). Infra, not a brain region.
        "device",
        # Promoted from reasoning/ 2026-08-04. SUBSTRATE, not basal_ganglia (an earlier plan said
        # basal_ganglia): this module spawns subprocesses, copies a directory tree and checksums
        # files. It contains no neuron, no weight and no learning rule -- its own docstring says it
        # "gates source edits, it does not update a weight" -- so it is infra by the same test that
        # puts `device` here. It also owns _PROTECTED, the 9-file firewall the self-build path
        # enforces, and _repo_root(), which three other modules import.
        "self_mod_harness",
        "_mathutils",
        "arm_view",
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
        "brain_bundle",
        "web_ingest",
        "lake_working_memory",
        "chat_memory",  # durable content-addressable chat memory on the spike lake (noise-filtered)
        "signal_calibrator",  # durable running-quantile of a live signal (Step-0 of data self-recalibration)
        "nengo_bridge",
        "home",
        "binlog",
        "growable",
        "region_coherence",  # C1 batch-3: cross-region Kuramoto coherence + integration -> GWT ignition bias
        "region_wiring",  # the always-on daemon's faculty-wiring registry -> charge_all + honest coverage report
        "brain_telemetry",  # the ONE snapshot contract + JSONL recorder + provenance table
        "brain_geometry",  # hand-authored anatomical layout + the REAL connectome tracts (stdlib)
        "cockpit",  # THE one frontend: live brain + arm + hardware target + chat, one page
        # PROMOTED from the repo root 2026-08-04 under P4c: the daemon is the brain's DRIVE (wake/
        # sleep cycle, tick loop, continuity), not a launcher beside it. It was 370 lines outside the
        # package in no gate, with no self-check and not one assert. `learn_daemon.py` at the root is
        # now a thin shim so the documented command still works. Its __main__ is a self-check, NOT
        # the drive loop -- run_selfchecks executes __main__ and a forever-loop would hang the gate.
        "learn_daemon",
        "dormancy_gate",  # Policy 3: the no-silent-dormancy CI gate (reachable-or-EXEMPT over REGIONS)
        "replication_guard",  # Policy 4: the one-brain guard (canonical stack classes defined once)
        "import_manifest",  # Policy 3: the cross-tier canary (governed -> ungoverned lazy imports)
        "method_dormancy_gate",  # Policy 3 amendment: per-METHOD dormancy (update rules with no live caller)
        "ingest_guard",  # content boundary for outside data: licence deny-by-default + injection flagging
        "hf_stream",  # streamed HF datasets -> brain-consumable features (licence-gated, bounded)
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

# old flat name -> "region.module". The `m != r` guard used to exempt the same-named hippocampus module
# (region name == module name); that module was retired in the Policy-3 wave, so no such collision
# remains today -- the guard is kept because it is what makes a future same-named module safe.
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
