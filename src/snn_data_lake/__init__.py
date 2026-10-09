"""snn_data_lake -- a neuromorphic library organized by brain anatomy.

REQUIRES THE FREE-THREADED (PEP 703) INTERPRETER. Owner decision 2026-08-05: the brain runs on
CPython 3.14 built with --disable-gil (ABI tag cp314t), and `require_free_threaded()` below
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
LIFNeuron` resolve to the moved module (snn_data_lake.growth.lif_neuron). See
snn_data_lake.brain_atlas for the rendered map and a back-compat self-check.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import sysconfig


# distinct-from: sensorimotor/wavefront_planner.py, faculties/quality_diversity.py,
#   sleep/sleep_replay.py, sleep/circadian_clock.py,
#   gates/gradient_claim_gate.py -- the hook matched shared verbs (run/refuse/free);
#   none guards the interpreter. This EDIT EXTENDS the incumbent guard in place (one
#   mechanism, two tiers) -- it does not add a sibling. All sites read.
def require_free_threaded(where="import", strict=False):
    """Refuse to run under a GIL-enabled interpreter. See the module docstring for WHY loud.

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

    TWO DIMENSIONS SINCE specs/022 T016 (FR-006), one mechanism: the BUILD flag
    (Py_GIL_DISABLED -- is this a cp314t interpreter at all) AND the RUNTIME state
    (sys._is_gil_enabled() -- is the GIL actually off RIGHT NOW). The build flag alone was the
    measured gap: on cp314t under PYTHON_GIL=1 the flag still reads 1 while the GIL runs, so
    the pre-T016 guard passed a de-parallelized interpreter. On builds without
    `sys._is_gil_enabled` (pre-3.13) the honest answer is "enabled", which the getattr default
    supplies.

    TWO TIERS, deliberate: plain import keeps the narrow `NORD_ALLOW_GIL=1` escape (downgrades
    to a stderr warning -- a contributor bisecting an unrelated bug on a GIL build should not
    be blocked; NOT a supported runtime mode, and any timing measured under it is measuring the
    wrong thing). `strict=True` -- the daemon boot and `make_brain` construction path -- IGNORES
    the escape and fails hard always (FR-006: the brain may never run de-parallelized, because
    nothing would crash; every partition would silently serialize while every gate stayed
    green).

    REACHABLE REDS, both constructed and run: the GIL-build refusal on standard 3.14
    (2026-08-05); the RUNTIME refusal under PYTHON_GIL=1 on cp314t incl. strict-ignores-escape
    (tests/test_gil_gate.py, real interpreters, T016).
    """
    build_ok = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    runtime_gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    if build_ok and not runtime_gil:
        return
    msg = (
        f"[{where}] snn_data_lake requires the FREE-THREADED CPython build (PEP 703, cp314t) "
        "with the GIL genuinely OFF.\n"
        f"  running: {sys.version.splitlines()[0]}\n"
        f"  Py_GIL_DISABLED={sysconfig.get_config_var('Py_GIL_DISABLED')!r} "
        "(1 = free-threaded build, 0/None = GIL build)\n"
        f"  sys._is_gil_enabled()={runtime_gil!r} (False = genuinely off; PYTHON_GIL=1 or an "
        "extension without free-threading support re-enables it at runtime)\n"
        "  install:  uv python install 3.14+freethreaded\n"
        "            <that python> -m venv .venv-lake && .venv-lake/Scripts/pip install -e '.[dev]'\n"
        "  A GIL'd interpreter imports and passes every self-check with the parallelism silently\n"
        "  absent, which is why this refuses instead of warning. Set NORD_ALLOW_GIL=1 to downgrade\n"
        "  to a warning for unrelated work -- never for a timing measurement, and never for the\n"
        "  brain itself (the construction path is strict and ignores the escape; FR-006)."
    )
    if not strict and os.environ.get("NORD_ALLOW_GIL") == "1":
        print(
            f"[snn_data_lake] WARNING: GIL active; parallel paths run SERIAL.\n{msg}",
            file=sys.stderr,
        )
        return
    raise RuntimeError(msg)


require_free_threaded()

# region -> modules  (single source of truth for the anatomical layout)
REGIONS = {
    "brainstem": [
        "reflex",
        "serial_arm",
        "so101_arm",  # the MuJoCo SO-101 gate; REPLACED arm_sim (specs/004)
        "arm_config",
        "interpreter",
    ],
    # specs/022 T054-i: the pons region's first computing module -- the pn mossy relay
    # (the second pons disposition: the package had been anatomy prose only)
    # specs/022 T021a: the insular region opens registration-complete -- the interoceptive
    # cortical terminus of viscera -> nts_rvlm -> thal_relay -> insula -> {gwt, ctx_pfc}.
    "plasticity": [
        # PROMOTED from reasoning/ 2026-08-04, AFTER its P7 redesign (owner rule: redesign
        # first, promote second). Oja (1982) is a HEBBIAN RULE with a normalizing decay term,
        # w += lr*y*(x - y*w): no membrane, no threshold, no spike. A plasticity rule, not a
        # neuron model. Sole governed importer: faculties/percept_surprise.py.
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
        # neuron model. Sole governed importer: faculties/percept_surprise.py.
        "oja_pca",
        "bcm_plasticity",
        "homeostatic_scaling",
        "inhibitory_stdp",
        "intrinsic_plasticity",
        "short_term_plasticity",
        # specs/022 T056: WM capacity derived, never picked -- K = T_max/t_s from the
        # executive's own synaptic constants + the five FR-020 guards (wm_attractor,
        # T055's standalone store, was ported into prefrontal_executive and DELETED)
        "eligibility_algebra",  # specs/022 T050: sum-of-products authoring contract (three-factor rules)
        "rule_composer",  # specs/022 T051a: typed sets + naive-sum red + causal attribution (P8 #3 core)
        "espp_rule",  # specs/022 T052-i: the extracted ESPP palette element (jepa's espp branch delegates)
        "competitive_learning",
        "meta_plasticity",
        "benna_fusi",
        "homeostatic_plasticity_stack",
        "plasticity_rule_stack",
        # PROMOTED from reasoning/ 2026-08-07 as specs/012's MANDATED BUILD (owner call: finish
        # the GeNN port). The tree's only GPU plasticity codegen: the CPU-verified v0.17
        # burst+x_tar rule as GeNN C++. Governance lands BEFORE the finish so the port's
        # completion happens inside the gates; dormancy disposition = EXEMPT (opt-in GPU
        # training harness, see dormancy_gate).
        # snn_mnist_stdp + deep_stdp (lane 2a/2b promotions) moved to the TRAIN_TIER annex
        # below on 2026-08-11 -- their module-top torchvision imports fail the alias-layer
        # round-trip wherever the [train] extra is absent (the first post-promotion CI run
        # proved it), which is exactly what the annex tier exists for.
        "naca",
        "r_stdp",
        "valence_stdp",
    ],
    "substrate": [
        # PROMOTED 2026-08-04: P2 merge of training/device.py (GPU-first, measured) with the
        # retired reasoning/torch_device.py (had the users + enable()). Infra, not a brain region.
        # Promoted from reasoning/ 2026-08-04. SUBSTRATE, not basal_ganglia (an earlier plan said
        # basal_ganglia): this module spawns subprocesses, copies a directory tree and checksums
        # files. It contains no neuron, no weight and no learning rule -- its own docstring says it
        # "gates source edits, it does not update a weight" -- so it is infra by the same test that
        # puts `device` here. It also owns _PROTECTED, the 9-file firewall the self-build path
        # enforces, and _repo_root(), which three other modules import.
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
        "data_quality",
        "spike_telemetry_hub",
        "streaming_hub",
        "lake_working_memory",
        "chat_memory",  # durable content-addressable chat memory on the spike lake (noise-filtered)
        "signal_calibrator",  # durable running-quantile of a live signal (Step-0 of data self-recalibration)
        "home",
        "binlog",
        # PROMOTED from the repo root 2026-08-04 under P4c: the daemon is the brain's DRIVE (wake/
        # sleep cycle, tick loop, continuity), not a launcher beside it. It was 370 lines outside the
        # package in no gate, with no self-check and not one assert. `learn_daemon.py` at the root is
        # now a thin shim so the documented command still works. Its __main__ is a self-check, NOT
        # the drive loop -- run_selfchecks executes __main__ and a forever-loop would hang the gate.
        "lake_bridge",  # specs/022 T005b: read-old/write-new format bridge; bare == constructed round-trip red
        "lake_segments",  # specs/022 T005: WAL(BinLog) + sealed Arrow IPC segments; bare == round-trip + CRC red
        "lake_index",  # specs/022 T007: derived pointer index (sqlite WAL); bare == round-trip + divergence red
        "lake_migrate",  # specs/022 T009: per-tag dispositions + identity migration; bare == identity + unknown-tag red
        "lake_engine",  # specs/022 T010-i: retention registry + IMMORTAL append budget; bare == consistency + flood red
        # PROMOTED 2026-08-11 (#10 family C1): the 64-dim crc32 content key for lake/skill routing
        "content_signature",
        "episodic_recall",
        # specs/022 T057: recall(cue) -> k episodes with margins over THE ONE LAKE's episode
        # records (LAESA/proximity_index composed; margin vs a measured scrambled-cue null)
        "lake_recall",
        "paradigm_b_engine",
        "paradigm_b_matcher",
        # PROMOTED+RENAMED 2026-08-11 (#10 family C3, was reasoning lake_memory): durable proximity-addressable replay
        "proximity_replay",
    ],
    # G8 (specs/022 evidence g8-reorganization.md 2.2, stage S1a): the engine-function package of
    # every policy and correctness gate. It opens with G8's own G-R8 instrument; the substrate
    # gates join it at M1.
    "gates": [
        "move_gate",  # G8 G-R8: the gate census, its diff against a declared delta, and its reds
        # specs/008 Gate Zero / 022 S6: the Amdahl ceiling of a delivery-path conversion
        "amdahl_gate",
        # specs/022 T024: every connectome node id named by its region or dated-EXEMPT; bare == census + probe reds
        "atlas_node_gate",
        # specs/022 X1 (T088): a capability claim carries its protocol and provenance
        "capability_bar",
        # specs/022 C5 (T090): what a consciousness instrument may conclude  # POLICIES.md manifest + release audit: derive rows, audit gates, certify (bare == --assert)
        "conclusion_protocol",
        # specs/022 T015: T=1..16 byte-equal trains + measured frac_bits; bare == control arms + thread pair
        "determinism_gate",
        # word ceilings on the standing documents: prose growth stays visible
        "doc_budget_gate",
        "docs_only_battery",
        # Policy 3: the no-silent-dormancy CI gate (reachable-or-EXEMPT over REGIONS)
        "dormancy_gate",
        # specs/022 T021b: five envelope files' bytes vs approved baseline; bare == positive control + 3 reds
        "envelope_hash_gate",
        # e-prop invariant: a transposed weight may not reach a weight update
        "eprop_transport_gate",
        # e-prop SCOPE: which rules OWE a cosine at all (census asserts on the governed tier)
        "gradient_claim_gate",
        # specs/022 T046: register row 7's instruments, counted (coverage red 1 of 10)
        "growth_coverage_gate",
        # specs/022 L5 (T089): what the lake never evicts, and the proof
        "immortal_retention",
        # Policy 3: the cross-tier canary (governed -> ungoverned lazy imports)
        "import_manifest",
        # specs/022 T008: delete->rebuild->logical-equality from the lake alone; bare == self-check
        "lake_rebuild_gate",
        # specs/022 T019: the five contract checks + divergence red over the committed artifact; bare == assert + reds
        "manifest_gate",
        # Policy 3 amendment: per-METHOD dormancy (update rules with no live caller)
        "method_dormancy_gate",
        # specs/022 T001: pinned-host perf ratchet; bare == cheap wiring check, never a measurement
        "perf_baseline",
        "policy_manifest",
        # specs/022 T022: reachable AND port-charged per region (BINDING at the T039 floor since 2026-08-22); bare == census + reds
        "region_charge_gate",
        # specs/022 T017: G5 snapshot/restore with RNG stream positions; bare == all four arms, reduced ticks
        "replay_gate",
        # Policy 4: the one-brain guard (canonical stack classes defined once)
        "replication_guard",
        # THE five approval-gated brainstem files, one definition + a SECURITY.md agreement check
        "safety_envelope",
        # battery carry-forward: closure-hash-verified PASS carries; bare == dependency/poison/envelope/FAIL reds
        "selfcheck_cache",
        # specs/022 G7 (T115): P4d's provenance check (red G-R7) -- unlabeled grown units refused; bare == R6 + ten reds
        "unit_label_gate",
    ],
    "instruments": [
        # specs/022 T078 Part I: the G3 task-binding (MI vs a permutation floor) and G4 content-leakage (held-out decode vs a shuffle band) instrument over the live loop's own attention state; no ACS, no wire to brain.py
        "attention_schema",
        "binding_instrument",
        # G1(c) carried-state campaign: the frozen constants, the contrast row and the oracle check (g1c-carried-state.md 2.2-2.4)
        "carried_state_campaign",
        # G1(c) campaign driver, part 2b: the controls, transport and separation stages (g1c-carried-state.md 2.6-2.8)
        "carried_state_controls",
        # G1(c) campaign driver: preflight, CLI and the twelve stages' wiring (g1c-carried-state.md 2.13)
        "carried_state_driver",
        # G1(c) campaign driver, part 2c: the report stage and the golden check (g1c-carried-state.md 2.3, 2.5, 2.11)
        "carried_state_report",
        # G1(c) campaign driver, part 2a: the migrate, causality and primary stages and the helpers every pool stage shares (2.9, 2.10, 2.3)
        "carried_state_stages",
        # G2 / T110 P1: the problem generators ((i)'s delayed-cue 2x2, C1-d, A_c1d, A_ii, U, U_S) and their construction checks (evidence g2-taught-problems.md 2.2, 2.8, 2.15)
        "delayed_cue",
        "energy",
        # 022 T035: stimulus port + trial schedule (fixation/saccade) + G5-replayable trials
        "episode_driver",
        # specs/022 Phase 4 C0: the L12 audit's 20-item everyday battery as a GOVERNED SCORER -- the frozen answer key, 2.3's ordered classifier (abstention decided by route AND preamble before any key is tested) and the five-seed campaign on scratch brains; never the live lake
        "everyday_battery",
        # specs/022 T113 (G5): the retained cell HARNESS (frozen generator, reading, in-brain cell machinery, PA3/PA4/PA8/PA10 reds, persisted records); Parts A and B were deleted on their nulls (4ecc90d9 / 0fba900c keep them as run). Reuse by Part D is PENDING, so the harness is exempt-by-registration, not charged (P3); brain-free bare self-check
        "g5_readout_cells",
        "ignition_instrument",
        # G2(ii)'s inseparable-mapping generator and G-R2's construction check on it (g1c-carried-state.md 2.8)
        "inseparable_mapping",
        "integration_measure",
        # specs/022 G6 (T114) part (b): joint-mask search contract + decide_b + THE mask_digest (R2 imports it)
        "joint_mask_search",
        # G1 causal contract diagnostic harness
        "memory_diagnostic",
        # PROMOTED 2026-08-12 (PROMOTE lane): the NEF (Eliasmith & Anderson) reference implementation -- REFERENCE IMPLEMENTATION / cross-engine ground truth for the gated [nengo] tier (infra/nengo_bridge:471 reaches it only when nengo is installed, which no gate env carries); its only gate-executed path is its own __main__ self-check. Reachable via nengo_bridge (dormancy green, no EXEMPT); recorded per the polychronization-entry pattern so "reachable" is not read as "charged".
        "nef_compute",
        "neuro_chip",
        "neuro_chip_stream",
        "neuronal_avalanches",
        # the plan's seed-paired t + exact sign-flip FPR (lifted from the T063 harness, T077 2.10)
        "paired_contrast",
        # G1(b) pool-selection campaign: section 2's steps, ordering and taxonomy
        "pool_selection",
        # specs/022 T021: pinned-payload neuter + margin surveillance + replay-twice A/B; bare == both controls
        "port_charge_probe",
        # specs/022 T063: the three-arm seeded/random/wrong drift harness with SC-006's two reds, PLANTED and FROZEN; bare == the offline self-check
        "prior_drift_harness",
        # specs/022 T081 [US4]: the C5 report_order instrument -- the live arm's report channel read against LESIONED CLONES of the workspace state that produced it (named pool vs a scrambled one); the lesion never touches the live brain
        "report_order_probe",
        # specs/022 G6 (T114) part (c): the sleep-pressure HARNESS (rescheduling, A6 shuffle, A7 meter); never a scheduler
        "sleep_pressure",
        # specs/022 G6 (T114) part (a): the live sleep-pruner census + per-pruner neuter-A/B (R0-R4, E)
        "sleep_pruner_census",
        # specs/022 G6 (T114) P4: the transactional mask application -- the `sleep_mask_prepared` producer (A12 step 1), the commit row (step 5) and the `applied_mask_digest` check ruling 33 owes; retires when R2's learner_events imports it and the mask rows live on the live path
        "sleep_mask_apply",
        # specs/022 R3 / T104 P1: the 2026-09-27 sleep-plasticity arms F1-F4 (two-process scratch-home arm-seeds, the presentation ledger, K1-K14, rulings 9 / 9a); never the live lake
        "sleep_plasticity_ab",
        "snn_mnist_stdp_genn",
        # G2 / T110 P3: the taught campaign -- the canonical incumbent through G0's channel, the presized and NGSNN front-ends, the explicit-history control, B3-B8, the G-R2 reading and the driver (evidence g2-taught-problems.md 2.4, 2.5, 2.8, 2.9, 2.15)
        "taught_campaign",
        # G2 / T110 P2: the Problem contract G3/G5/G6/G9 import -- splits as episodes, the future_d guard, the solved predicate, the budget fields, learning OFF (evidence g2-taught-problems.md 2.6, 2.7, 2.15)
        "taught_problems",
        # specs/022 G6 (T114) part (e): the staged sleep-bout runner (R3's run_bout), participation census, fabrication guard
        "staged_sleep_bout",
        # specs/022 W3: the world-red instrument -- two-chain liveness at 4dp, category entropy, the derived refit gap (P8 #9)
        "world_red_probe",
    ],
    "sensorimotor": [
        # 022 T036: transduction-in; idle sentinel handled at the door; proprio lane
        "afferent_boundary",
        "arm_bridge",
        # physical-arm build TARGET: spec + static torque budget + the sim-to-real gap
        "arm_hardware",
        "closed_loop",
        # PROMOTED 2026-08-11 (#10 family C2): real-scale corpus/CMUdict ingestion for the language loop
        "corpus_stream",
        # 022 T037: efference copy around the envelope; cerebellar forward-model charge
        "efferent_boundary",
        "esp32_encoder_path",
        "gesture_recognition",
        # streamed HF datasets -> brain-consumable features (licence-gated, bounded)
        "hf_stream",
        # content boundary for outside data: licence deny-by-default + injection flagging
        "ingest_guard",
        "lca_sparse_coding",
        "learned_matcher",
        "live_arm",
        "make_sensor_dataset",
        "multimodal_encoder",
        "nl_interface",
        # torch-trained tiers with a zero-dep serve/integration path (import cleanly without torch)
        "pc_cortex_serve",
        "population_encoding",
        "rank_order_coding",
        "recognizer_bank",
        "rsnn_encoder",
        "sensor_csv_ingest",
        "signal_loop",
        "spike_preprocessing",
        "trained_brain",
        "wavefront_planner",
        "web_ingest",
        # PROMOTED 2026-08-11 (#10 family C2b): the curated world-KB (605 real triples) + masked-slot adapter
        "world_kb_stream",
    ],
    "infra": [
        "_mathutils",
        "brain_bundle",
        # hand-authored anatomical layout + the REAL connectome tracts (stdlib)
        "brain_geometry",
        "brain_native_selfbuild",
        # the ONE snapshot contract + JSONL recorder + provenance table
        "brain_telemetry",
        # THE one frontend: live brain + arm + hardware target + chat, one page
        "cockpit",
        "demo",
        "device",
        "drift_monitor",
        "nengo_bridge",
        "nir_export",
        # the always-on daemon's faculty-wiring registry -> charge_all + honest coverage report
        "region_wiring",
        # Promoted from reasoning/ 2026-08-04. Both belong in basal_ganglia because both LEARN FROM
        # OUTCOME: self_build_loop is the correctness-gated apply/rollback path, and
        # brain_native_selfbuild picks WHICH candidate to propose using the brain's own three-factor
        # RSTDP rule driven by the gate verdict (kept +1 / rolled_back -1), which is reward-modulated
        # action selection in the literal sense -- not a census, not a heuristic ranking.
        # LIVE, and the honest limit of that claim: brain.py::_enter_wake reaches both once every
        # SELFBUILD_EVERY=8 nights, but ONLY under NORD_SELFMOD=1 (the daemon sets it, CI does not),
        # so under `pytest` / `run_selfchecks.py` neither fires and CHARGE IS UNMEASURED. Registered
        # as reachable-and-operator-gated, NOT as charge-proven.
        # brain_native_selfbuild's learned weights moved INTO the one lake under _T_SELFBUILD as part
        # of this promotion (P4b) -- they were in <repo>/data/selfbuild_state.json, a second durable
        # store. Cross-process round-trip proven: writer pid wrote w=0.77/0.11, a separate reader
        # process read them back after the writer exited.
        "self_build_loop",
        "self_mod_harness",
    ],
    "growth": [
        # PROMOTED from reasoning/ 2026-08-04: the three neuron models `neuron_zoo`
        # composes. They fed a GOVERNED module from outside the gate.
        "adex_neuron",
        "alif_neuron",
        # specs/022 T040: organizer fields + repression-GRN region emission (P8 #7)
        "arealization",
        "autapse_neuron",
        # Brain-Cog-adopted composition interface: node + local-rule + connection palette
        "basic_components",
        # Brian2 LIF authority + differentiable Torch bridge
        "brian2_backend",
        # specs/022 T111 (G3) P1: the growth trigger -- contradiction count + exact lift certificate
        "capacity_certificate",
        # H01 HUMAN cortex reference data: value + units + provenance + bound
        "cortical_prior",
        "dendritic_neuron",
        "function_preserving_growth",
        "growable",
        "growing_learner",
        # specs/022 T042: one-shared-program locality + scaffold refusal (P8 #7)
        "growth_program",
        # specs/022 T047: |LP|+charge growth gates, executive-independent (P8 #4)
        "growth_stop",
        # specs/022 T111 (G3) P3: the grow-attempt loop, the budget pause, the ledger + unit journal
        "growth_loop",
        # specs/022 T111 (G3) P1: the taught learner's grown latency segment (frozen code never re-ranked)
        "grown_code",
        # specs/022 T111 (G3) P5: the grown units' projection into the G1 reservoir (connection 2)
        "grown_feed",
        # specs/022 T062: the H01 c3 SYNAPSE graph, 8 gates, a seeded population
        "h01_ingest",
        # Brain-Cog-adopted: LIF+Izhikevich+AdEx worked EXAMPLE of the basic_components palette (not on the live tick)
        "heterogeneous_microcircuit",
        "hodgkin_huxley",
        "izhikevich_neuron",
        # specs/022 T046a: |LP| third factor (adopts CuriosityDrive) + the _T_GOAL store
        "learning_progress",
        "lif_neuron",
        # merges: recognizer ensemble / encoder bank / neuron population
        "neuron_zoo",
        # every grown unit gets a cortical depth, a layer and an immutable Dale sign
        "placement",
        "resonate_and_fire",
        # Brain-Cog SCA-adopted: novelty-sized recruit cap + frozen-expert reuse
        "similarity_sized_growth",
        "srm_neuron",
    ],
    "reservoir": [
        "cortical_microcircuit",
        "criticality_regulator",
        # specs/022 T013: SoA substrate state, 128B-aligned, lake-backed; bare == alignment + round-trip + refusal reds
        "kernel_state",
        # specs/022 T014: two-phase deterministic tick (P fixed, integer delivery); bare == delay + thread-equality reds
        "kernel_tick",
        "neural_sampling",
        # G1(b) pool-selection benchmark: the winner, g1 LSM, as a streaming pool
        "recurrent_pool",
        # specs/022 T045: learnable OU bootstrap through the SAME plasticity path (FR-033)
        "spontaneous_activity",
    ],
    "readout": [
        # PROMOTED 2026-08-11 (#10 lane 4b): function-preserving Net2Net growth + freeze; the specs/007 T022 origin copy
        "growable_readout",
        "interaction_learning",
        # PROMOTED from reasoning/brain_adapters/ 2026-08-12 (#10 lane 4c): the ONE live-tick
        # percept readout (feedback alignment + fixed-random L6, no weight transport) --
        # engine/brain.py constructs it lazily on every real percept; numpy-only, CORE tier.
        "laminar_local_readout",
        # PROMOTED 2026-08-11 (#10 family C5): SLDA-RFF competence head; absorbed fair_compare's equal-budget bar
        "language_readout_slda",
        "predictive_coding",
        "snn_classifier",
        # PROMOTED 2026-08-13 (was reasoning/snn_moe_classifier, pre-promotion path): firing-rate ROUTED
        # mixture-of-experts over snn_classifier's spiking experts. CORE, not TORCH_TIER -- it
        # imports only math/random/numpy plus readout.snn_classifier and infra._mathutils.
        # Reached from sensorimotor/recognizer_bank.py. Its blocker was an UNRESTORED process-global
        # `random.seed(0)`, fixed with infra/brain_bundle.py's snapshot-seed-restore pattern.
        "snn_moe_classifier",
        "tempotron",
    ],
    "sleep": [
        "circadian_clock",
        "developmental_pruning",
        # C1 batch-4: systems-consolidation of real episodes (sparse attractor + sleep replay)
        "episode_consolidation",
        "experience_replay",
        # PROMOTED 2026-08-11 (#10 lane 4a): the loop's replay adapter over ReplayBuffer + SleepConsolidator
        "hippocampal_memory",
        # sleep-time prune of the symbolic durable lake (cap re-derivable bulk, keep learned)
        "lake_consolidation",
        "lifelong_consolidation",
        # PROMOTED 2026-08-11 (#10 family C9a): the ONE standing skills loop (absorbed meta_skill_loop + open_meta_loop)
        "nightly_skills",
        "sleep_consolidation",
        "sleep_replay",
        # specs/022 T048: DEEP-R budget + MSP sign steering, slow clock
        "structural_rewiring",
    ],
    "engine": [
        "autonomous_loop",
        "brain",
        "learn_daemon",
        "live_brain",
        # 022 T034: typed workspace-report snapshot; the broadcast's declared consumer
        "report_channel",
        # specs/022 T011: ONE-writer multicast ring + lake journaller; bare == replay A/B + overrun red
        "signal_ring",
        # specs/022 T018: TypedPort + WiringManifest (declared anatomy); bare == five checks + five reds + NIR round-trip
        "wiring_manifest",
    ],
    "faculties": [
        "claustrum",
        "affective_appraisal",
        "cpg_oscillator",
        "fear_conditioning",
        "habenula",
        "homeostatic_drives",
        "interoceptive_arousal",
        "interoceptive_integration",
        "mossy_relay",
        # C1 batch-3: cross-region Kuramoto coherence + integration -> GWT ignition bias
        "region_coherence",
        "set_points",
        "vital_nuclei",
        "wm_capacity",
        "acetylcholine",
        # C1 batch-2: spiking neuromod fusion -> coupled affect + self-organized emotions
        "affect_integrator",
        "affective_memory",
        "allocentric_map",
        "astrocyte_modulation",
        "contrastive_episode_analysis",
        "cortisol",
        "developmental_envelope",
        "drive_reduction",
        "empowerment",
        "generalizing_novelty",
        "grid_cells",
        "grid_place_map",
        "intrinsic_motivation",
        "locus_coeruleus",
        # PROMOTED 2026-08-11 (#10 family C7): DP-means signature discovery (feeds nightly_skills' open frontier)
        "open_world_loop",
        "raphe_serotonin",
        "ring_attractor",
        # PROMOTED 2026-08-11 (#10 family C4): domain-agnostic skill formation + SkillRegistry proximity router
        "skill_builder",
        "spiking_curiosity",
        # C1: continuous-attractor SPIKING grid network (bump path integration)
        "spiking_grid_can",
        # Promoted from reasoning/ 2026-08-04 after an in-place P7/P1/P8 redesign. FACTORED
        # k-WTA associative memory: the P sparse patterns ARE the memory (O(P*k)), so there is
        # no N x N weight matrix to store. WIRED AND CHARGED, measured not assumed -- neutering
        # S.recall to a pass-through costs its live consumer (sensorimotor/recognizer_bank.py:233
        # StorageRecognizer) 3 of its 64 out-of-distribution rejections, 0.844 -> 0.797. That is
        # P6 one-way as well: a hippocampus faculty whose output moves a cortex faculty's output.
        # Honest qualifier: the bank builds lazily and fires only on an explicit "recognize:"
        # query, so the path is charged but query-gated, not exercised on an ordinary tick.
        # factored k-WTA associative memory; denoising content-addressable recall
        "spiking_storage_prototype",
        "action_gating",
        # PROMOTED from reasoning/ 2026-08-12 (#10 lane 4d): the SELECTION stage over latent_q's
        # VALUE table (fuses value + metacognition on the loop's curiosity). Composition-only;
        # the fusion A/B verdict at promotion is a recorded NULL at the shipped rule -- see the
        # module docstring. EXEMPT-"wire" until developmental_loop_brain promotes.
        "brain_decision",
        "connectome_router",
        "dopamine_nuclei",
        "evidence_accumulation",
        "forward_model",
        "gain_adaptation",
        "global_workspace",
        "gwt_deep",
        "hjepa_serve",
        "interoceptive_relay",
        "jepa_world_model",
        "latent_q",
        # specs/022 T054-i: the anatomical TISSUE layer (granule expansion + PF bundle +
        # Purkinje + MLI opposite-sign pathway) -- names cb_ctx and cb_deep
        "microcircuit",
        "mlp_q",
        "multisensory_integration",
        # PROMOTED from reasoning/ 2026-08-04 (owner directive: every build is promoted into the
        # governed tier the moment it lands; reasoning/ is outside all three P3 gates, which is
        # where dormancy hides). Cerebellum is the anatomically correct home -- this is the timing
        # structure, and synfire_chain.py:37 already cross-references it to distinguish zero-lag
        # synchrony from delay-matched convergence.
        "polychronization",
        "reentrant_workspace",
        "reticular_gate",
        "sc_deep_layers",
        "spiking_jepa",
        "spiking_pid",
        "superior_colliculus",
        "synfire_chain",
        # higher cognition / self-model / self-improvement (v3.14+)
        "access_consciousness",
        "affect_core",
        "agency_monitor",
        "autotelic_goals",
        # PROMOTED+RENAMED 2026-08-12 (#10 lane 4f, was reasoning/brain_adapters/executive_control -- the LAST brain_adapters module): dopamine-RPE lr gate + the PUBLIC CompetenceDynamics/goal_value planning objects (the WorkspaceBroadcast/ExecutivePlanner wrappers RETIRED later the same day -- retirement record in-file); renamed because the basename is taken by faculties/executive_control (brain_atlas.py:81 asserts uniqueness)
        "competence_executive",
        # C1 batch-2: N-agent recursive ToM / counterfactual do()-operator simulation
        "counterfactual_sim",
        "crossmodal_binder",
        "divisive_normalization",
        "executive_control",
        # C1: spiking clock / task-set-shift / learned generative self-model
        "generative_self_model",
        "is_snn_forward",
        # PROMOTED from reasoning/ 2026-08-11 (#10 batch-1): open-vocab DistMult KB completion head
        "kb_ranking",
        # PROMOTED 2026-08-11 (#10 family C6): autonomous R-STDP language assemblies; LIVE on the sleep tick
        "language_self_build",
        # C1 batch-3: learned spiking reticular attention gate (local plasticity)
        "learned_attention",
        # C1 batch-4: learned expected-value-of-control (dopamine-gated cost/value)
        "learned_evc",
        "metacognitive_monitor",
        # loop C2: online unsupervised Oja-PCA reconstruction-residual surprise (orienting drive)
        "percept_surprise",
        "prefrontal_executive",
        "quality_diversity",
        "reasoning_curriculum",
        "recursive_tom",
        # PROMOTED 2026-08-12 (PROMOTE lane): the ROSE fluent-language pipeline (VSA constituency parse + KB-grounded comprehend + SLDA competence), LIVE via brain._rose_serve's lazy reach. P3/P4b debt carried visibly, NOT discharged by the move (AUDIT_BACKLOG H7, runtime-reproduced at the promotion audit): nightly_refit moves the SLDA sufficient statistics while competence() returns the build-time cached acc byte-identical; the refit filter accepts a novel HEAD coded OOV-zero; _readout is per-Brain RAM reaching brain_lake.bin nowhere. P2 direction recorded: rose's derived parameterisation (rff from in_dim, ctx from data, relations from the KB ratio) is THE mechanism; sensorimotor/world_kb_stream main()'s four-literal demo copy consumes it or retires at the family's next lane.
        "rose_language",
        # C2 r2: LIF metacognition OF access reliability (knowing THAT it knows)
        "second_order_access",
        "self_improvement_suite",
        "self_model",
        "task_set",
        "theory_of_mind",
        "theta_gamma_code",
        "time_cells",
        "vsa_reasoning",
        # PROMOTED 2026-08-12 (PROMOTE lane): Gentner relational structure-mapping over VSA spike codes; LIVE via reasoning_cortex's structure-map solver (reached from brain.py's lazy :924 -- the lane's own I001 sort moved it from :928 -- + reasoning_cortex:705). P3 debt carried visibly, NOT fixed here: brain.py:967's `_structure_mapper` attribute is write-only (AUDIT_BACKLOG M11) -- retire-or-read is its own commit. P2 direction recorded in-file: the dim-parameterised _cleanup_project is THE mechanism; reasoning/resonator_network's hardcoded copy (measured 300/300 bit-identical at dim=512) delegates or retires when its lane arrives.
        "vsa_structure_mapping",
        "wholistic_loop",
    ],
}

# TORCH-TIER ANNEX -- region -> modules that import torch AT MODULE LOAD (so they cannot be imported
# in the zero-dep CI matrix). They are part of the anatomical map (region_of + the atlas count include
# them), but are NOT aliased into the zero-dep back-compat layer and are import-verified only when torch
# is present. Every OTHER torch-trained organ (pc_cortex_serve, brain, jepa/hjepa serve,
# neurogenesis) imports lazily and lives in REGIONS above. See brain_atlas for the gated verification.
TORCH_TIER = {
    "sensorimotor": [
        # PROMOTED 2026-08-12 (PROMOTE lane 4/4, was training/eprop_nmnist, pre-promotion path): the brain's
        # N-MNIST perception encoder (e-prop, cosine>0.5 vs BPTT asserted in-check). TORCH_TIER
        # because torch is imported at module top; reached on Brain's CONSTRUCTION path
        # (brain.py's perception-tier try block). Its e-prop engine lives in `eprop_kernel` below,
        # which it and `eprop_shd` both import.
        "eprop_nmnist",
        # PROMOTED 2026-08-13 (specs/013 C3, was training/eprop_shd, pre-promotion path): the brain's SHD audio
        # perception encoder, the auditory twin of eprop_nmnist. TORCH_TIER for the same reason
        # (torch at module top), reached on the same construction path via brain.py's audio try.
        "eprop_shd",
    ],
    "growth": [
        # specs/022 T111 (G3) P2: NG-SNN's candidate lifecycle, shared by every consumer of the one grower
        "candidate_lifecycle",
        "neurogenesis_engine",
        "ng_snn",
    ],
    "plasticity": [
        # The ONE e-prop engine (specs/013 C4). Holds the nine definitions that were byte-identical
        # across eprop_nmnist and eprop_shd -- surrogate spike fn, the independent BPTT reference,
        # the cosine comparator, the Adam step, the evaluation loop, and the LIF base. Both
        # consumers import from it; neither carries a copy. Verified bit-exact across the merge
        # (12/12 pinned tensors identical, tests/pin/), which is the check that matters here --
        # the cosine assert is measurably blind to the one judgment call it contains.
        "eprop_kernel",
    ],
    "readout": [
        "online_readout",
    ],
    "faculties": [
        "language_acquisition",
    ],
}

# TRAIN-TIER ANNEX (added 2026-08-11, #10 lane-2 CI regression fix) -- region -> modules that
# import TORCHVISION at module load (the [train] extra; torch alone is not enough). Same
# contract as TORCH_TIER: part of the anatomical map (region_of + atlas counts), NOT aliased
# into the back-compat layer, and import-verified only where torchvision is present (the gate
# env carries torchvision+cu128; CI's torch-tier job installs no extras and skips with notice).
# The first CI run after these modules entered REGIONS failed exactly as the hub's own
# docstring predicted -- the alias round-trip imported them on a box with no torchvision.
TRAIN_TIER = {
    # PROMOTED 2026-08-12 (#10 lane 4e): the brain-integrated developmental loop -- continual
    # learning via balanced hippocampal replay (its identity mechanism). torchvision at module
    # load via the load_mnist hub import, hence this annex and not REGIONS.
    "instruments": [
        "deep_stdp",
        "deep_stdp_latency",
        "developmental_loop_brain",
        "snn_mnist_stdp",
    ],
}

# G8 (specs/022 evidence g8-reorganization.md 2.1 decision 4; 2.4 S1a): the FROZEN pre-G8 label
# tables -- copies of the S0-head (393411b) REGIONS / TORCH_TIER / TRAIN_TIER, every key and
# member string byte-identical and in the same order (C6 asserts them AST-equal to the S0 head's
# tables). Their provenance comments stay with the live tables above: copied here they would
# re-cite five retired files and raise citation_guard's BROAD count in a live file, which G8's C7
# forbids (measured at S1a: exactly +5). Region-keyed gates take LABELS and MEMBERSHIP from these
# and every module PATH from module_path() below (F5): once G8 moves a module, its label is not a
# directory. NEVER edited after S1a; re-targeting the region rules to grown-neuron labels is later
# G7/G4 work, and these tables are never merged with that vocabulary (2.2).
LEGACY_REGIONS = {
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
        "laminar_local_readout",
        "neuronal_avalanches",
        "cortical_microcircuit",
        "cortical_prior",
        "h01_ingest",
        "arealization",
        "growth_program",
        "prior_drift_harness",
        "placement",
        "growth_stop",
        "memory_diagnostic",
        "recurrent_pool",
        "pool_selection",
        "lca_sparse_coding",
        "predictive_coding",
        "is_snn_forward",
        "neural_sampling",
        "snn_classifier",
        "snn_moe_classifier",
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
        "neuron_zoo",
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
        "generative_self_model",
        "recursive_tom",
        "counterfactual_sim",
        "learned_attention",
        "attention_schema",
        "report_order_probe",
        "everyday_battery",
        "learned_evc",
        "second_order_access",
        "heterogeneous_microcircuit",
        "percept_surprise",
        "kb_ranking",
        "corpus_stream",
        "world_kb_stream",
        "language_self_build",
        "language_readout_slda",
        "nightly_skills",
        "competence_executive",
        "nef_compute",
        "vsa_structure_mapping",
        "rose_language",
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
        "lake_recall",
        "spiking_grid_can",
        "episode_consolidation",
        "content_signature",
        "proximity_replay",
        "skill_builder",
        "open_world_loop",
        "hippocampal_memory",
        "growable_readout",
        "lake_consolidation",
        "spiking_storage_prototype",
    ],
    "basal_ganglia": [
        "evidence_accumulation",
        "learning_progress",
        "r_stdp",
        "valence_stdp",
        "action_gating",
        "latent_q",
        "mlp_q",
        "brain_decision",
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
        "polychronization",
        "microcircuit",
    ],
    "brainstem": [
        "reflex",
        "serial_arm",
        "so101_arm",
        "arm_hardware",
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
        "ignition_instrument",
        "binding_instrument",
        "interoceptive_relay",
    ],
    "hypothalamus": ["circadian_clock", "set_points", "homeostatic_drives"],
    "midbrain": [
        "dopamine_nuclei",
        "superior_colliculus",
        "sc_deep_layers",
        "multisensory_integration",
    ],
    "pons": ["mossy_relay"],
    "medulla": ["vital_nuclei", "interoceptive_arousal"],
    "epithalamus": ["habenula"],
    "olfactory": [],
    "claustrum": ["claustrum"],
    "insula": ["interoceptive_integration"],
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
        "affect_integrator",
    ],
    "plasticity": [
        "oja_pca",
        "bcm_plasticity",
        "homeostatic_scaling",
        "inhibitory_stdp",
        "intrinsic_plasticity",
        "short_term_plasticity",
        "wm_capacity",
        "developmental_pruning",
        "structural_rewiring",
        "eligibility_algebra",
        "rule_composer",
        "espp_rule",
        "competitive_learning",
        "criticality_regulator",
        "meta_plasticity",
        "benna_fusi",
        "lifelong_consolidation",
        "homeostatic_plasticity_stack",
        "plasticity_rule_stack",
        "snn_mnist_stdp_genn",
    ],
    "substrate": [
        "device",
        "brian2_backend",
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
        "chat_memory",
        "paired_contrast",
        "signal_calibrator",
        "nengo_bridge",
        "home",
        "binlog",
        "growable",
        "spontaneous_activity",
        "growth_coverage_gate",
        "region_coherence",
        "region_wiring",
        "brain_telemetry",
        "brain_geometry",
        "cockpit",
        "learn_daemon",
        "dormancy_gate",
        "replication_guard",
        "import_manifest",
        "method_dormancy_gate",
        "eprop_transport_gate",
        "doc_budget_gate",
        "amdahl_gate",
        "gradient_claim_gate",
        "policy_manifest",
        "capability_bar",
        "immortal_retention",
        "conclusion_protocol",
        "perf_baseline",
        "lake_bridge",
        "lake_segments",
        "lake_index",
        "lake_rebuild_gate",
        "lake_migrate",
        "lake_engine",
        "signal_ring",
        "kernel_state",
        "kernel_tick",
        "determinism_gate",
        "replay_gate",
        "wiring_manifest",
        "manifest_gate",
        "port_charge_probe",
        "envelope_hash_gate",
        "region_charge_gate",
        "atlas_node_gate",
        "selfcheck_cache",
        "report_channel",
        "episode_driver",
        "afferent_boundary",
        "efferent_boundary",
        "safety_envelope",
        "ingest_guard",
        "hf_stream",
        "basic_components",
        "similarity_sized_growth",
        "world_red_probe",
    ],
}

LEGACY_TORCH_TIER = {
    "cortex": [
        "neurogenesis_engine",
        "language_acquisition",
        "eprop_nmnist",
        "eprop_kernel",
        "eprop_shd",
    ],
    "hippocampus": [
        "ng_snn",
        "online_readout",
    ],
}

LEGACY_TRAIN_TIER = {
    "plasticity": ["snn_mnist_stdp", "deep_stdp"],
    "cortex": ["deep_stdp_latency"],
    "hippocampus": ["developmental_loop_brain"],
}


# The package's ONE basename-to-path mapping (G8 S1a): the call-time accessors the region-keyed
#   gates join on (F5), and since G8 S1e the source of gates/dormancy_gate.py's _CANON, which
#   used to compute the same map itself (P2). brain_atlas.region_of answers the frozen LABEL.
def package_of(module):
    """The package that holds `module` (a basename) NOW, read from the three live tables; None
    when no live table lists it. The live half of G8's join -- LEGACY_* holds the label half."""
    for table in (REGIONS, TORCH_TIER, TRAIN_TIER):
        for package, members in table.items():
            if module in members:
                return package
    return None


def module_path(module):
    """`module`'s current import path, `snn_data_lake.<package>.<module>`. Region-keyed gates
    build every PATH through this and take only LABELS and MEMBERSHIP from LEGACY_* (F5): a label
    is not a directory once G8 moves a module. KeyError for a name no live table lists."""
    package = package_of(module)
    if package is None:
        raise KeyError(f"{module!r} is in no live registry table")
    return f"{__name__}.{package}.{module}"


# G8 S1b (evidence g8-reorganization.md 2.5, "The extended alias layer"). `_LEGACY` maps each old
# `pkg.mod` to its current `pkg.mod`, derived from LEGACY_* and the live tables and holding only the
# pairs that differ: EMPTY until a module moves, then growing by itself with every re-keyed table.
# `_LEGACY_PKGS` is every LEGACY_* package key minus every live package key, from the tables and
# never from the filesystem (F4): a directory left behind is C6's red, not a silent shadow.
_LABEL = {
    m: p
    for t in (LEGACY_REGIONS, LEGACY_TORCH_TIER, LEGACY_TRAIN_TIER)
    for p, ms in t.items()
    for m in ms
}
_LEGACY = {
    f"{_LABEL[m]}.{m}": f"{p}.{m}"
    for t in (REGIONS, TORCH_TIER, TRAIN_TIER)
    for p, ms in t.items()
    for m in ms
    if m in _LABEL and _LABEL[m] != p
}
_LEGACY_PKGS = frozenset(
    k for t in (LEGACY_REGIONS, LEGACY_TORCH_TIER, LEGACY_TRAIN_TIER) for k in t
) - frozenset(k for t in (REGIONS, TORCH_TIER, TRAIN_TIER) for k in t)

# old flat name -> "region.module". The `m != r` guard used to exempt the same-named hippocampus module
# (region name == module name); that module was retired in the Policy-3 wave, so no such collision
# remains today -- the guard is kept because it is what makes a future same-named module safe. G8
# S1b: a `_LEGACY_PKGS` name never becomes a flat alias (`claustrum` once its package empties), or it
# would shadow the synthetic legacy package of the same name.
_MOVED = {
    m: f"{r}.{m}" for r, mods in REGIONS.items() for m in mods if m != r and m not in _LEGACY_PKGS
}


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


# distinct-from: sensorimotor/eprop_shd.py -- _self_ref derives one module's own identity; these extend
#   this file's own alias layer in place (one layer: the flat alias plus G8's legacy paths). Also
#   read: faculties/spiking_jepa.py (target_encode), faculties/vsa_reasoning.py (coincidence),
#   faculties/grid_place_map.py (_pearson), substrate/lake_recall.py (_find_cell_collision).
class _LegacyLoader(_AliasLoader):
    """G8 S1b: a legacy two-level path's alias -- the same identity as the flat alias, plus
    `get_code` delegating to the target's real loader UNDER THE TARGET'S NAME (a SourceFileLoader
    refuses any other name, 1.6), so `python -m <legacy path>` runs the real module; its spec carries
    the target's origin, from which runpy sets __file__ (F2). The flat alias keeps no get_code: it
    serves imports only, and doc_command_guard's premise that it cannot run under -m stays true."""

    def get_code(self, fullname):
        return importlib.util.find_spec(self._target).loader.get_code(self._target)


class _LegacyPackageLoader(importlib.abc.Loader):
    """G8 S1b: the synthetic package standing where an emptied legacy package was (F4)."""

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__doc__ = (
            f"{module.__name__} -- a SYNTHETIC legacy package (specs/022 G8): every module it held "
            "moved to an engine-function package; its old two-level paths resolve through _LEGACY"
        )


def _strict(fullname, what):
    """The G8 tripwire (F3(c)): with SNN_ALIAS_STRICT=1 a legacy path or a synthetic package never
    resolves -- live code may not use legacy names -- while a flat alias always does (the envelope's
    own self-check imports `snn_data_lake.signal_loop` through one)."""
    if os.environ.get("SNN_ALIAS_STRICT") == "1":
        raise ImportError(
            f"SNN_ALIAS_STRICT: {fullname} resolves only through G8's legacy alias ({what})"
        )


def _legacy_spec(fullname, target):
    spec = importlib.machinery.ModuleSpec(
        fullname, _LegacyLoader(target), origin=importlib.util.find_spec(target).origin
    )
    spec.has_location = True
    return spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves legacy `snn_data_lake.<flat>` imports to their new regional location, and since G8
    S1b, in this order: (a) a `_LEGACY_PKGS` name to a synthetic package (is_package, no search
    locations); (b) a `_LEGACY` two-level path to an identity alias carrying the target's origin;
    (c) a flat name exactly as before. It stays LAST on sys.meta_path, so a real file always wins."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + "."
        if not fullname.startswith(prefix):
            return None
        tail = fullname[len(prefix) :]
        if tail in _LEGACY_PKGS:
            _strict(fullname, "a synthetic legacy package")
            spec = importlib.machinery.ModuleSpec(fullname, _LegacyPackageLoader(), is_package=True)
            spec.submodule_search_locations = []
            return spec
        if tail in _LEGACY:
            _strict(fullname, f"now {__name__}.{_LEGACY[tail]}")
            return _legacy_spec(fullname, f"{__name__}.{_LEGACY[tail]}")
        if "." in tail or tail not in _MOVED:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(f"{__name__}.{_MOVED[tail]}"))


sys.meta_path.append(_AliasFinder())


def __getattr__(name):
    # supports `from snn_data_lake import lif_neuron` and `snn_data_lake.lif_neuron` attribute access
    if name in _MOVED:
        return importlib.import_module(f"{__name__}.{_MOVED[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
