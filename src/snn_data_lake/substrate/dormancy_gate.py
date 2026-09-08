"""Policy-3 dormancy gate -- no part of the brain may remain SILENTLY dormant.

Policy 3 (user, 2026-07-24): "every part of the brain must never remain dormant; all present AND
future builds must be wired and utilized appropriately." This gate MECHANIZES that, the same way
brain_atlas mechanizes "every module on disk is in the map": it builds the static import graph over
snn_data_lake, computes which REGIONS/TORCH_TIER modules are REACHABLE from the live-behaviour roots
(the autonomous tick) or a shipped console entrypoint, and ASSERTS that every brain module is either
REACHABLE or explicitly EXEMPT (with a reason). A new build that is never imported by any live path
and is not exempted FAILS this self-check in CI -- so dormancy cannot ship silently.

Honest framing (this is the guard that keeps Policy 3 from manufacturing theater): reachability proves
a module is IMPORTED by a live path; it does NOT prove the module's output changes behaviour. "Utilized
appropriately" (does the wire actually charge behaviour) is proven separately, per faculty, by the
runtime neuter-A/B probe (`tools/charge_probe.py`) during a wiring wave -- too expensive for CI. So
this gate is the CHEAP first tier (catch built-but-never-imported); the A/B is the second tier.

(That citation read `scratchpad/smoke_drive_charge.py` until 2026-08-04. `scratchpad/` is gitignored,
so the file named here as the runtime instrument is ABSENT FROM A CLONE -- the same dead citation
POLICIES.md records correcting in its own text and in this file, where it survived. It is the exact
defect `tools/citation_guard.py`'s narrow rule exists for, and the guard does not catch it here
because a prose mention is outside its STRICT scope. The tracked instrument is `tools/charge_probe.py`.)

The EXEMPT allowlist is the honest record of "not reachable from a live path, and why": every entry is
one of three dispositions -- "retire" (superseded, pending deletion in a Policy-2/3 wave), "wire"
(valuable, pending a verify-first wiring wave), or "permanent" (legitimately not on the live tick:
[train]-tier offline trainers, benchmark/measurement harnesses, arm/GUI/serve entrypoints reached only
by a console script, or a faculty awaiting a prerequisite such as a real sensor). The gate FAILS if a
brain module is neither reachable nor exempt; it also reports (soft) any EXEMPT entry that has since
become reachable (retire the exemption) so the list shrinks honestly as waves land.

  python -m snn_data_lake.substrate.dormancy_gate   # (also run by run_selfchecks.py)
"""

import ast
import os

from snn_data_lake import REGIONS, TORCH_TIER, TRAIN_TIER

_PKG = "snn_data_lake"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../snn_data_lake

# canonical path per REGIONS/torch basename (the module shares its region name only for hippocampus)
_CANON = {}
for _region, _mods in REGIONS.items():
    for _m in _mods:
        _CANON[_m] = f"{_PKG}.{_region}.{_m}" if _m != _region else f"{_PKG}.{_region}.{_region}"
for _tier in (TORCH_TIER, TRAIN_TIER):  # both annexes are brain modules (2026-08-11: +TRAIN_TIER)
    for _region, _mods in _tier.items():
        for _m in _mods:
            _CANON[_m] = f"{_PKG}.{_region}.{_m}"
BRAIN_MODULES = set(_CANON.values())

# The live-behaviour roots (the autonomous brain) + the shipped console-script entrypoints. A brain
# module reachable from ANY of these is "wired"; reachable only from tests/__main__ is dormant.
_LIVE_ROOTS = (
    f"{_PKG}.cortex.brain",
    f"{_PKG}.cortex.live_brain",
    f"{_PKG}.brainstem.autonomous_loop",
    f"{_PKG}.thalamus.reentrant_workspace",
)
_ENTRYPOINTS = (  # pyproject [project.scripts] + the always-on daemon's in-package deps
    f"{_PKG}.substrate.demo",
    f"{_PKG}.brainstem.live_arm",
    f"{_PKG}.brainstem.closed_loop",
    f"{_PKG}.substrate.cockpit",  # THE frontend (served by the daemon); replaced control_panel
    f"{_PKG}.training.cli",
    f"{_PKG}.substrate.web_ingest",
    f"{_PKG}.substrate.nengo_bridge",
    f"{_PKG}.substrate.region_wiring",  # the daemon charges faculties through this registry
)

# ------------------------------------------------------------------------------------------------
# EXEMPT: brain modules that are legitimately NOT reachable from a live path, each with a disposition
# and a reason. Dispositions: "retire" (pending deletion), "wire" (pending a verify-first wave),
# "permanent" (train-tier / benchmark / entry-serve / awaiting-prerequisite). The gate shrinks this
# list as Policy-3 waves land; a "retire"/"wire" entry that outlives its wave is a standing TODO.
# ------------------------------------------------------------------------------------------------
# key = module basename (unique across REGIONS per brain_atlas); value = (disposition, reason).
EXEMPT = {
    # so101_arm's "wire" exemption is GONE, as its own text predicted it should be: the gate is
    # on the live tick now (autonomous_loop, serial_arm's mirror, live_arm, cpg_oscillator,
    # spiking_pid, ring_attractor and the planner all drive it), so it is reachable and needs no
    # exemption. Leaving the entry would have been the quiet lie the soft-report exists to catch.
    # reaction_diffusion_compute: RETIRED AND DELETED 2026-08-02, as its own entry said it should be.
    # The exemption existed for one reason -- so the deletion would be a decision taken on its merits
    # rather than a side effect of the specs/004 planner rewrite -- and that decision is now taken.
    # What was checked before deleting, because "no consumer" is a claim and not an observation:
    #   * the only historical consumer, cerebellum/wavefront_planner.py, is now a bidirectional RRT;
    #     the grid it needed is 36**6 ~ 2.2e9 cells at six joints against 2592 at two, so the wave
    #     cannot run on the arm this repo actually has. Dimension retired it, not preference.
    #   * there is no OTHER 2-D occupancy grid anywhere live. The brain's only spatial surface is the
    #     allocentric landmark map behind `map_position`, which carries positions and landmarks and no
    #     obstacles, so there is nothing over which a wavefront would compute a geodesic.
    # Wiring it to the landmark map to keep it alive would have been a fake consumer, which P3 names
    # explicitly. The primitive was correct, self-checked and cross-validated against an independent
    # BFS; none of that is a reason to keep an unused mechanism (P2). It is recoverable from history.
    # NOTE: the 17 "retire"-tagged modules (wilson_cowan/RBM/kalman/kuramoto/cue_integration/olfactory x2/
    # 3 redundant seq-memories/neurogenesis-stdlib/language_brain+brain_reasoning/adaptive_spiking/lt_gate/
    # feedback_alignment/sensor_demo/coordinated_growth) were DELETED in the Policy-2/3 retirement wave --
    # they are gone from REGIONS, so they are no longer brain modules and need no exemption.
    # ---- "permanent": legitimately not a tick faculty, each with the evidence that settled it ----
    # afferent_boundary (T036) needs NO entry: Brain._safe_ctx's import makes it
    # REACHABLE and the gate proved it (a first draft added an exemption claiming the
    # walk could not see the lazy import -- the stale-EXEMPT guard fired the same run,
    # which is that guard earning its keep). Its neuter A/B lives in its self-check.
    "episode_driver": (
        "permanent",
        "TRIAL DRIVER (specs/022 T035): owns the trial-phase schedule and the stimulus "
        "port over scratch KernelTick subjects; the same/different oracle is ESPP's "
        "supervisory source. Consumers: its __main__ self-check (in-regime charge, "
        "oracle-truth, G5-restore and refusal reds), T052's ESPP adoption benchmark "
        "(names this oracle in its task row), and T081's report-order probe windowing "
        "over its trials -- dated forward clients, each landing WITH its task (an "
        "earlier draft cited 'T077+', which does not route through this driver -- "
        "corrected at review). Wiring it into the live tick would make the experiment "
        "part of the organism it measures",
    ),
    "ignition_instrument": (
        "permanent",
        "MEASUREMENT INSTRUMENT (specs/022 T033, P8 invention #6's ignition scorer): drives "
        "scratch ReentrantWorkspace subjects through the T032 actuator and scores the "
        "all-or-none transition (dual-derived gate + external controls). Its consumers are "
        "its __main__ self-check (five reachable failure-mode reds), tools/"
        "ignition_baseline.json (drift-checked every self-check run), and T039's exit "
        "certification. Wiring it into the live tick would make the observer part of the "
        "observed -- the same reason charge probes are not tick faculties",
    ),
    "gwt_deep": (
        "permanent",
        "BENCHMARK DEMONSTRATION (promoted reasoning/ -> thalamus/ 2026-08-11): the tree's "
        "phase-binding spiking-vs-rate control, credited 2026-08-07 on the benchmark that killed "
        "the workspace bind_* family. Nothing on the live tick binds K (color,shape) objects; its "
        "consumers are its __main__ self-check (4 reachable reds) and "
        "tests/instruments/gwt_deep_split_ab.py. Wiring it to the tick would be the fake "
        "consumer P3 names",
    ),
    "growable_readout": (
        "permanent",
        "TORCH-TIER growable readout (promoted 2026-08-11, #10 lane 4b): function-preserving "
        "Net2Net (1,0) growth + bit-exact freeze. T022 DISCHARGED HERE 2026-08-12: the one "
        "torch (1,0)-split helper lives in this file, and LaminarColumns (per-task isolated "
        "columns) was ported in at the laminar merge (laminar_readout.py deleted). The old "
        "'converts to reachable when its consumers promote' clause was MEASURED FALSE at the "
        "dlb promotion (2026-08-12): its importers are reasoning/autonomous_brain (the kept "
        "A/B control) and the now-governed TRAIN-TIER developmental_loop_brain -- whose "
        "DEFAULT path never constructs it (NORD_GROWABLE/NORD_COLUMNS opt-in; the full 348.8s "
        "run printed 'readout grew 0x'). Honest disposition: permanent train-tier library "
        "+ the instrument tests/instruments/ensemble_rbf_ab.py",
    ),
    "brain_decision": (
        "permanent",
        "SELECTION-turned-CONTROL stage over basal_ganglia/latent_q's value table (promoted "
        "2026-08-12, #10 lane 4d): fuses value + metacognition on the loop's curiosity; "
        "composition-only. The EXEC-bypass P2 this entry used to record was ADJUDICATED "
        "2026-08-12: the consumer (hippocampus/developmental_loop_brain) deleted NORD_EXEC "
        "and the fused choose() branch on a measured tie plus this module's recorded fusion "
        "null -- choose() keeps exactly ONE external call site, the consumer's "
        "mechanics-check split criterion, executed by gate command #1. On that consumer's "
        "default path observe() trains a table nothing reads (bit-identical A/B, 3 seeds) "
        "-- presence-not-participation, recorded plainly with NO conversion clause: the "
        "consumer is itself EXEMPT-permanent TRAIN-TIER and reachable() seeds only from "
        "live roots/entrypoints (measured 2026-08-12: its promotion converts NOTHING to "
        "reachable). Wiring anything else would be the fake consumer P3 names",
    ),
    "hippocampal_memory": (
        "permanent",
        "ADAPTER over this region's own governed mechanisms (promoted 2026-08-11, #10 lane 4a): "
        "per-digit ReplayBuffers + a SleepConsolidator pass for the developmental training loop. "
        "Its ONE consumer (hippocampus/developmental_loop_brain, all three methods on its main "
        "path) promoted 2026-08-12 -- and the 'converts to reachable the commit that loop "
        "promotes' clause this entry used to carry was MEASURED FALSE at that commit: "
        "reachable() seeds only from _LIVE_ROOTS/_ENTRYPOINTS/_external_roots, and the "
        "promoted consumer is EXEMPT-permanent TRAIN-TIER, not a live root. Adding it to the "
        "roots would launder dormancy for its whole import closure. Honest disposition: "
        "permanent, same shape as deep_stdp_latency",
    ),
    "developmental_loop_brain": (
        "permanent",
        "TRAIN-TIER brain-integrated developmental loop (promoted 2026-08-12, #10 lane 4e): "
        "continual MNIST learning via balanced hippocampal replay, offline by design (Adam "
        "readout -- the specs/010 e-prop adjudication permits it ONLY while no live root "
        "reaches it; its docstring carries the binding constraint). Consumers: none by "
        "design -- it IS the train-tier measurement loop (bare __main__ = 5s wiring "
        "mechanics, DLB_RUN=1 = the 5.8-min benchmark reproducing 93.3% / -0.2% vs "
        "DEEP_LOOP_REF). Wiring it to a live root would convert the permitted offline "
        "trainer into an e-prop violation",
    ),
    "competence_executive": (
        "permanent",
        "EXECUTIVE ADAPTERS over a developmental COMPETENCE vector (promoted+RENAMED "
        "reasoning/brain_adapters/executive_control.py -> cortex/competence_executive.py "
        "2026-08-12, #10 lane 4f -- the LAST brain_adapters module; renamed because "
        "cortex/executive_control holds the basename and brain_atlas.py:81 asserts "
        "uniqueness). Three stdlib wrappers over governed mechanisms: dopamine RPE -> "
        "bounded lr multiplier (midbrain/dopamine_nuclei), objective broadcast "
        "(thalamus/global_workspace), competence planning (cortex/prefrontal_executive). "
        "Its ONE consumer is hippocampus/developmental_loop_brain, lazily inside main() "
        "and _mechanics_check(), and that consumer is itself EXEMPT-permanent TRAIN-TIER "
        "and offline. NO CONVERSION CLAUSE IS WRITTEN HERE, and that is a measurement, "
        "not caution: reachable() seeds only from _LIVE_ROOTS/_ENTRYPOINTS/_external_roots, "
        "and the consumer has ZERO importers tree-wide (measured at promotion against the "
        "unfiltered graph walk -- neither this module nor its consumer is in the raw "
        "closure), so there is no downstream promotion that could ever convert this entry. "
        "The three 'converts to reachable' clauses corrected wire->permanent on 2026-08-12 "
        "are the precedent. DO NOT force-wire: the lr multiplier feeds an OFFLINE Adam "
        "readout (no live consumer asks for one), and the workspace wrapper's live repoint "
        "is blocked on a contract fix -- broadcast() steps a clock it does not own, "
        "measured to perturb the arm's own (content, hold) on 20 of 20 ticks at thr*1.6 "
        "and 3 of 20 even sub-threshold. A wire built to green this gate is exactly the "
        "fake consumer P3 names",
    ),
    "deep_stdp_latency": (
        "permanent",
        "TRAIN-TIER perception feature learner (promoted 2026-08-11, #10 lane 2c, region "
        "settled cortex/ -- its consumer set IS the perception tier): ConvLatencyStdp is "
        "consumed by the reasoning-tier loops (autonomous_brain, developmental_*, "
        "fast_features) until their own rungs; nothing governed reaches it yet. Mechanics "
        "check in TRAIN_GATED; DSLATENCY_RUN=1 = the full benchmark with both measured bars",
    ),
    "deep_stdp": (
        "permanent",
        "TRAIN-TIER LIBRARY + recorded honest negative (promoted 2026-08-11, #10 lane 2b): "
        "FcStdp/OneLayerNet/assign_labels/evaluate are imported verbatim by the lineage's "
        "superseders (deep_stdp_latency, deep_stdp_2stage -- reasoning-tier until their own "
        "rungs); nothing on the live tick trains it. Full real self-check in TRAIN_GATED "
        "wherever torchvision imports",
    ),
    "snn_mnist_stdp": (
        "permanent",
        "TRAIN-TIER HARNESS (promoted 2026-08-11, #10 lane 2a): the Diehl-Cook rule-family hub; "
        "nothing on the live tick trains MNIST. Its constants ARE the GeNN port's oracle source "
        "of truth (snn_mnist_stdp_genn.oracle() lazy-imports them -- a function-level import "
        "inside an EXEMPT module, which the reachability walk does not traverse). Its full real "
        "self-check (58.8s GPU train) runs in the run_selfchecks TRAIN_GATED tier wherever "
        "torchvision is importable",
    ),
    "snn_mnist_stdp_genn": (
        "permanent",
        "TRAIN-TIER HARNESS (specs/012, MANDATED BUILD 2026-08-07): opt-in GPU training run via "
        "__main__ + GENN_RUN=1 on a GeNN box; nothing on the live tick trains MNIST. The gate "
        "self-check covers its codegen integrity on every box, toolchain or not",
    ),
    "heterogeneous_microcircuit": (
        "permanent",
        "PALETTE EXAMPLE: a worked demo of the basic_components composition palette (itself permanent). "
        "The REGIONS comment claiming it is 'composed live under a local rule' was FALSE and is corrected",
    ),
    # arealization's "wire" row RETIRED 2026-08-25 by its own stated condition: T043 landed
    # cortex/placement.py's module-top import, which is what this gate walks (placement is
    # itself counted reachable through neurogenesis_engine's function-body imports -- the
    # PERMISSIVE conditional-import walk the banner discloses). Reachability, not charge:
    # region_prior's GRN arm has no live-path caller yet (the sleep-tick grower takes the
    # cortex arm); its callers today are placement's own __main__, which every gate run
    # executes. No waived invariant was recorded in the row beyond that TODO.
    #
    # growth_program's "wire" row RETIRED 2026-08-25 the same way, with a REAL consumer this
    # time: placement.place_grown BRANCHES on growth_program.SCAFFOLD (the admission refusal
    # inherits T042's closure) and the self-check asserts every grower route sits inside
    # GROWABLE -- functional consumption of the ratified partition, not import-only. Recorded
    # because this row's TODO print dies with it: the program's step/divide machinery remains
    # __main__-only. RE-ADJUDICATED at T047: growth_stop.py gates the REFIT grain and does
    # not compose the division machinery (the earlier "until T047 composes it" was a promise
    # at the wrong grain); composition is owed to the task that wires region-wide growth
    # from seed -- the growth-coverage gate's 1-of-10 red is its standing driver.
    # espp_rule's `wire` row RETIRED at T053-ii (2026-08-31): the module is REACHABLE through
    # rule_composer's census imports (which inhibitory_stdp's composed production path now
    # reaches), so an EXEMPT row would be the stale-exemption red. The T052-ii MIXED-NULL
    # adjudication (no production wire on that measurement; the exact in-gate assert re-opens
    # it the day espp starts helping) lives in the module's own benchmark section and the
    # T052 tasks.md rows -- reachability changes none of it.
    # rule_composer's `wire` row RETIRED at T053-ii (2026-08-31): InhibitorySTDP's production
    # step() now applies THROUGH the kernel (the first tick-charged composed rule -- the very
    # consumer this row dated), so the module is reachable from a live path and the row's
    # excuse is discharged, not merely stale. Still OPEN and carried where they live: P2
    # merge-must-reduce (both stacks remain production -- module docstring + tasks.md T051b-iv
    # row) and the T051c-wave disposition WAVE (its own tasks.md row). check_producers /
    # rule_charge remain module-level functions invisible to method_dormancy_gate's ClassDef
    # scan -- the module docstring says so; rule_charge's tick-live reading landed with this
    # same commit's stack harness.
    # eligibility_algebra's `wire` row RETIRED at T053-i (2026-08-31): istdp_synapse_rule
    # (plasticity/inhibitory_stdp.py, a reachable registered module) now imports the algebra
    # on its authoring path, so module-grade reachability is SATISFIED and an EXEMPT row here
    # would be the stale-exemption red this gate exists to catch. What reachability does NOT
    # discharge, said where the gate reader looks: the algebra's first TICK-live composition
    # is T053-ii's, and its P1 harness-constant exception carries that date in its own
    # DISCLOSURES (one expiry, one date, both siblings -- the rule_composer row below still
    # holds the composer's half).
    # structural_rewiring's `wire` row RETIRED at T053-ii (2026-08-31): reachable through
    # rule_composer's harness imports, so an EXEMPT row would be the stale-exemption red.
    # THE WIRE LANDED AT T058 (2026-09-01), mandated by FR-024's sleep-owned structural
    # slow clock: Brain._sleep_tick rewires the IXN rows per sleep tick, MSP-steered by
    # the stack's meta theta, with the T051b-ii cascade-adoption fix
    # (PlasticityRuleStack.adopt_structural) in the same block and rewire refusals
    # COUNTED into state(). T048's live-regime measurement stands as the EXPECTATION,
    # not a refusal: an all-positive-birth substrate crosses sign rarely, so live
    # turnover ~0 is the mechanism holding; the reachable red is the constructed
    # sign-flip arm in tests/test_sleep_capabilities.py. T054-ii's tissue-side refusal
    # is unmoved (pf_w is its own state of record; nothing writes it -- that half stays
    # re-dated to FR-019's cerebellar rule). Both module docstrings carry the record.
    "microcircuit": (
        "wire",
        "specs/022 T054-i: the cerebellar TISSUE (granule codon expansion + PF bundle + "
        "Purkinje + the MLI opposite-sign pathway; names cb_ctx and cb_deep, retiring "
        "their atlas phantoms). Charged surface today: its own registered self-check "
        "(regenerable basis, exact-arithmetic codon gate, sparsity + supra-input capacity "
        "each against its own ablation control, the FR-019 window pinned to MEASURED "
        "channel latency with a discriminating weakened-twin red, the MLI weight neuter, "
        "the deep readout held to its identity per step, and FR-018's per-element "
        "neuter A/B). NO DATED CONSUMER -- said plainly rather than dressed as one. "
        "T054-ii MEASURED AND REFUSED the structural wire this row previously dated "
        "(nothing writes pf_w: 0 sign crossings and 0 of 512 weights changed over 2200 "
        "driven steps, so DEEP-R's prune trigger cannot fire and wiring anyway is P6 "
        "force-wiring). Both remaining consumers are undated by construction: the "
        "structural wire waits on the first mechanism that WRITES pf_w, and the "
        "brain-side consumer -- FR-019's cerebellar rule in the live motor loop -- has "
        "no task id in specs/022 tasks.md yet (the T044 panel form).",
    ),
    "mossy_relay": (
        "wire",
        "specs/022 T054-i: the pn stage (cortico-ponto-cerebellar mossy fan-out; the "
        "pons region's first computing module -- names pn, retiring its atlas phantom). "
        "Charged surface today: its own registered self-check (regenerable structure, "
        "fan-in normalization, determinism, one-channel locality, discrimination vs a "
        "full-fan-out control, refusals) plus the tissue self-check's end-to-end "
        "ctx->pn->cb_ctx->cb_deep arm. Same disposition as the microcircuit row above "
        "and for the same measured reason: NO dated consumer -- T054-ii refused the "
        "structural wire on measurement, and the live-brain consumer carries no task "
        "id yet, both said plainly.",
    ),
    # wm_attractor: RETIRED AND DELETED (specs/022 T056). The T055 row that stood here
    # carried the settled P2 verdict (the delayed-match benchmark could not separate the
    # two stores; two wrong verdicts recorded); T056 executed the row's own disposition --
    # the unique capability (the decoupled u/tau_f retention vs R/tau_d release pair, its
    # elapsed-time bookkeeping, and the neuter idiom) is PORTED into
    # cortex/prefrontal_executive, the capacity is DERIVED there (K = T_max/t_s,
    # plasticity/wm_capacity, five guards), and the file is gone. Exactly one mechanism
    # survives; the full record lives in wm_capacity's docstring and specs/022 tasks.md.
    # genomic_bottleneck left this table 2026-09-02: the FR-033 wire it named landed at
    # 847c5f6a (the engine imported it; NGSNN.fit read the map through a candidate hook), its
    # paired A/B at 24 seeds was a NULL, and the module was DELETED in the following commit per
    # its own pre-registered disposition. Record: specs/022 evidence/s1-genome-fanin-ab.md.
    "cortical_microcircuit": (
        "permanent",
        "REFERENCE MODEL: Potjans-Diesmann 2014 validation dynamics. CORRECTED 2026-08-16 -- this "
        "reason read 'with no input port' and that was FALSE: `thalamic_drive(thalamic_hz)` IS an "
        "input port, added 2026-08-04, and `simulate(..., thalamic_hz=)` carries a caller's drive "
        "into it. It is anatomically targeted (CONN_PROB_TH is nonzero for L4e/L4i/L6e/L6i ONLY, so "
        "a caller cannot inject into L2/3 or L5 -- the stimulus must travel the intracortical "
        "route), it contributes EXACTLY 0.0 at 0 Hz so an undriven run stays bit-identical, and it "
        "refuses a negative rate. The TRUE reason for the exemption is that NOTHING DRIVES IT: zero "
        "references to `thalamic_drive` or `CorticalMicrocircuit` exist outside this module. The "
        "port carries a SCALAR rate, so a caller can modulate drive intensity but not inject a "
        "spatial pattern -- that, not the port, is what specs/011 Phase A should be scoped against. "
        "Wiring a live driver is specs/019 [B2.08], and this exemption is retired the day it lands",
    ),
    # NOTE: plasticity_rule_stack + homeostatic_plasticity_stack + bcm_plasticity + inhibitory_stdp +
    # homeostatic_scaling + intrinsic_plasticity + short_term_plasticity are WIRED (Brain.__init__ attaches
    # PlasticityRuleStack to the live intent learner; _sleep_tick relaxes the cascades) -> exemptions retired.
    "competitive_learning": (
        "permanent",
        "LIBRARY: a whole competitive LAYER (own LIF neurons + lateral weights), NOT a .pre_gain/.stabilize "
        "wrapper -- contract-incompatible with the learner hook, and neither stack composes it; its one "
        "plausible consumer (recognizer_bank) is itself not live and the brain already has clustering paths",
    ),
    "developmental_pruning": (
        "permanent",
        "BENCHMARK: no class and no reusable surface -- only run(seed)/main() over module-level constants "
        "hard-tied to a synthetic task; it already delivered its finding (magnitude +53% over random at "
        "11.9% density). Wiring it would mean writing a NEW module, not wiring this one",
    ),
    # NOTE: 11 more "wire" candidates were RETIRED after probing (entropy_coder, population_vector,
    # stochastic_resonance, hippocampus, legendre_memory, spiking_recurrent_cell, sensory_encoder_bank,
    # spike_slicer, spike_pooling, pontine_relay, reticular_formation) -- each had no honest consumer or
    # duplicated a live+charged path. They are gone from REGIONS, so they need no exemption.
    "neural_sampling": (
        "permanent",
        "LIBRARY: posterior sampling from a DEFINED energy model; the live tick has no such model and "
        "no query asking for samples from one (it is also the retired RBM's replacement, on demand)",
    ),
    "sensor_csv_ingest": (
        "permanent",
        "TOOL: offline CLI ingest aimed at signal_loop, not the brain tick",
    ),
    # esp32_encoder_path's EXEMPT row retired at T036 on THIS GATE'S OWN BAR ONLY --
    # import-graph reachability (the afferent boundary imports its `normalize`, and this
    # gate's stale-EXEMPT check fires on a reachable module with a standing row). Said
    # exactly, because the first retirement comment overclaimed 'reachable from the live
    # tick' and the review measured the call level: transduce_temperature has no live
    # caller yet, so 014's SEPARATE clear-the-exemption bar (SC-008: a live consumer,
    # decided by FR-010's neuter-A/B) remains OPEN -- the milestone is NOT claimed, and
    # its dated record plus the awaiting-hardware invariant live in specs/022 SYNTHESIS
    # section 6. Reachability satisfied the gate; charge still owes its A/B.
    "sc_deep_layers": (
        "permanent",
        "AWAITING EFFECTOR: a gaze MOTOR PLANT (burst pool -> neural integrator -> displacement); the "
        "brain has no eyes/gaze motor, so there is nothing for it to drive",
    ),
    "superior_colliculus": (
        "permanent",
        "AWAITING EFFECTOR: orienting/saccade WTA map; its selection role on the percept is already "
        "filled by the live attention gate, and there is no gaze effector to orient",
    ),
    "neuro_chip": (
        "permanent",
        "HARDWARE TWIN: software model of Loihi/SpiNNaker silicon, targeted for real via nir_export",
    ),
    "neuro_chip_stream": (
        "permanent",
        "HARDWARE TWIN: out-of-core shard-streaming sibling of neuro_chip",
    ),
    "energy": (
        "permanent",
        "BENCHMARK: analytical SynOps->Joules / NeuroBench harness; an accounting instrument, not a faculty",
    ),
    "rsnn_encoder": (
        "permanent",
        "TRAIN-TIER: recurrent-spiking encoder trained offline and SERVED frozen; emits no live-tick value",
    ),
    # spiking_jepa + is_snn_forward `permanent` rows RETIRED at T053-ii (2026-08-31): both
    # reachable through rule_composer's census imports, so EXEMPT rows would be the
    # stale-exemption red. The adjudications stand where they live: jepa is NOT a drop-in
    # for the live rate JepaWorldModel (encode() invariant across 24 live-width patterns --
    # its own docstring's record; kept, unadopted, honestly), and is_snn_forward remains a
    # library primitive tied to it. Reachability is import-topology, not adoption.
    # ---- "permanent": legitimately not on the live tick, with a reason (infra / tool / library) ----
    "basic_components": (
        "permanent",
        "Brain-Cog composition PALETTE (a library/interface, not a tick faculty)",
    ),
    "data_quality": ("permanent", "lake data-quality monitor (substrate infra util)"),
    "make_sensor_dataset": ("permanent", "offline dataset-builder tool (invoked out-of-band)"),
    "lake_memory": (
        "permanent",
        "substrate lake primitive used by reasoning tools/apps, not the tick",
    ),
    "nir_export": ("permanent", "neuromorphic NIR export -- an offline hardware-export tool"),
    # dormancy_gate needs no exemption: region_wiring.live_coverage() delegates to it -> reachable.
    "replication_guard": (
        "permanent",
        "the Policy-4 guard itself -- a CI self-check, not a tick faculty",
    ),
    "eprop_transport_gate": (
        "permanent",
        "the e-prop invariant's tree-level gate -- a CI self-check, not a tick faculty. Asserts no "
        "GOVERNED module reads a weight transposed into a weight update; reports the ungoverned "
        "tier without asserting on it (1 finding at reasoning/gf.py::train, hand-written backprop "
        "with no .backward() anywhere, which is why promotion_sweep's marker grep cannot see it)",
    ),
    # placement: EXEMPTION RETIRED 2026-08-16, the wire landed. `cortex/neurogenesis_engine` calls
    # place() for every unit a refit adds, reindexes by the model's survivor mask after each prune,
    # and persists (depth, layer, sign) with the substrate. It is reachable from a live root and
    # needs no exemption -- which is the disposition P3 actually wants, rather than a standing TODO.
    # cortical_prior: EXEMPTION RETIRED 2026-08-16, and the GATE ASKED FOR IT -- once placement was
    # wired it reported "1 EXEMPT entries are now reachable (retire them): ['cortical_prior']". The
    # chain neurogenesis_engine -> placement -> cortical_prior reaches a live root, so the H01 human
    # reference data is consumed rather than merely present. P4b still forbids it entering the lake:
    # it is what a microscope saw, not what the brain learned.
    "doc_budget_gate": (
        "permanent",
        "word ceilings on the STANDING documents -- a CI self-check, not a tick faculty. Budgets "
        "five docs read every session; deliberately does NOT budget the historical records "
        "(CHANGELOG, specs/, docs/coverage/, vendored papers) where growth is CORRECT, nor "
        "CLAUDE.md, which is untracked and would report MISSING in every clone",
    ),
    "hf_stream": (
        "permanent",
        "TOOL: offline data-prep CLI -- streams a licensed HF dataset into feature rows for a "
        "trainer. Same disposition as sensor_csv_ingest: it feeds the [train] tier, not the tick, "
        "and the brain must never make a blocking network call mid-control",
    ),
    # method_dormancy_gate + gradient_claim_gate `permanent` rows RETIRED at T053-ii
    # (2026-08-31): both reachable through rule_composer's cross-check imports (check_claim
    # holds claim_keys to gradient_claim_gate's own surface), so EXEMPT rows would be the
    # stale-exemption red. Both remain CI self-checks, never tick faculties -- their own
    # docstrings say so; nothing about their disposition moves with reachability.
    "policy_manifest": (
        "permanent",
        "the POLICIES.md manifest + release-audit instrument (specs/021) -- a CI self-check, not "
        "a tick faculty. Derives the per-policy enforcement manifest from the governing document, "
        "audits gate commands, certifies releases. Same disposition as gradient_claim_gate above: "
        "reachable from run_selfchecks and CI, never from a tick",
    ),
    "capability_bar": (
        "permanent",
        "specs/022 X1 (T088): the capability-claim register -- every claim of a human-comparable "
        "or superhuman capability carries its protocol, both numbers and both provenances, or it "
        "is REFUSED and no tracked surface may compare. A CI/run_selfchecks instrument, never on "
        "a tick -- the policy_manifest disposition. RETIRES when a tick faculty reads the "
        "register, which nothing should: a claim is about the brain, not an input to it",
    ),
    "immortal_retention": (
        "permanent",
        "specs/022 L5 (T089): the never-forgets gate -- every lake tag labelled IMMORTAL or "
        "COMPACTABLE with its reason, no durable reclaimer outside the scratch homes, and a "
        "cross-process byte-identity round trip per IMMORTAL tag. Reads lake_engine.RETENTION and "
        "lake_migrate.TAG_DISPOSITIONS rather than re-declaring them (P2). A CI/run_selfchecks "
        "instrument, never on a tick -- the policy_manifest disposition. RETIRES if the tick ever "
        "consults retention to decide what to write, which would make the label load-bearing live",
    ),
    "conclusion_protocol": (
        "permanent",
        "specs/022 C5 (T090): what a consciousness instrument may and may not conclude -- one "
        "licensed sentence, one refused sentence and one dissociation input per C4 instrument, no "
        "aggregate across two of them, and no surface printing a reading without its licensed "
        "sentence in view. The sentences are written BEFORE C4 measures anything; that is the "
        "point of the row. A CI/run_selfchecks instrument, never on a tick -- the policy_manifest "
        "disposition. RETIRES when C4's instruments import it to gate their own reporting, which "
        "would make it live-reachable and the row the stale-exemption red",
    ),
    "growth_coverage_gate": (
        "permanent",
        "specs/022 T046: invention register row 7's four instruments (coverage ratchet + "
        "refusals, three-way partition, boundary sharpness via the PRODUCTION labelers, "
        "spatial null, Allen elementwise external control) -- a CI/run_selfchecks "
        "instrument, never on a tick -- the policy_manifest disposition. The coverage red "
        "it reports (1 of 10, cortex-only) is the OPEN work; the gate is its counter",
    ),
    "lake_rebuild_gate": (
        "permanent",
        "specs/022 T008: the index-is-derived gate (delete -> rebuild -> logical equality, "
        "planted-divergence red, lake-tracking control). A CI/run_selfchecks instrument, never "
        "on a tick -- the policy_manifest disposition",
    ),
    # ---- FOUR `wire` ROWS RETIRED 2026-09-06 by the stale-exemption arm (specs/022 L6, T091) --------
    # lake_engine, lake_migrate, lake_bridge and lake_segments all went reachable in one commit, and the
    # gate refused their rows the same run. What made it happen: L6 put `lake_engine.retention_of` on the
    # SLEEP TICK -- `hippocampus/lake_consolidation.consolidate` and `substrate/home.KnowledgeLog.consolidate`
    # both read the retention label before any record is dropped -- and `lake_engine` imports
    # `lake_migrate.TAG_DISPOSITIONS` at module top, which carries the other three in behind it.
    #
    # THE HONEST PART, and the reason this comment exists instead of four deleted lines. Only ONE of the
    # four is CHARGED: `lake_engine`. Its `retention_of` decides whether a learned fact survives the night,
    # which is an outcome, and L-R7 in `lake_consolidation.main()` is the constructed red that proves it --
    # flip 0x01 to reclaimable and the same call caps again.
    #
    # `lake_migrate`, `lake_bridge` and `lake_segments` are reachable through an IMPORT and nothing more.
    # Their rows promised a live consumer at the US3 kernel paging tier (T064/T066) and that consumer has
    # NOT arrived; the BOUND EXPIRED AT T039 UNMET is still true of all three, and their standing-TODO
    # lines are gone with the rows, so it is recorded here instead. Reachability is not charge (P3), and a
    # retired exemption must not be read as a satisfied promise. The debt these rows carried:
    #   lake_migrate  (T009) per-tag dispositions + identity migration -- consumers are still instruments
    #                 (lake_engine's registry-consistency assert, the T009 full-scale proof); the live
    #                 cutover migration belongs to T066.
    #   lake_bridge   (T005b) read-old/write-new format bridge -- consumers still instruments
    #                 (lake_migrate.iter_records); the live path is T013+'s paging tier.
    #   lake_segments (T005) segment writer (WAL + sealed Arrow IPC) -- the migrator and the bridge read it
    #                 as instruments; the live-path consumer is T066.
    # All three keep BOUND EXPIRED AT T039 UNMET, recorded and not re-dated.
    "lake_index": (
        "wire",
        "specs/022 T007: the derived pointer index (sqlite WAL, stdlib). EXPIRY RE-EVALUATED "
        "AT T012: the rebuild gate (T008) consumes it as a CI instrument and the close-out "
        "did not cut the live lake over, so the live-path consumer is the US3 kernel block's "
        "paged reads (T064/T066). BOUND EXPIRED AT T039 UNMET (US1 audit B3, 2026-08-25; "
        "recorded, not re-dated). Printed by the standing-TODO arm every run",
    ),
    "region_charge_gate": (
        "permanent",
        "specs/022 T022: the region-granularity charge verdict (reachable AND measured "
        "port-edge attribution; registration never greens). BINDING since T039 wrote the "
        "floor (2026-08-22; asserts charged_count >= tools/region_charge_floor.json -- this "
        "row said report-only for three days after, US1 audit m4); a CI/run_selfchecks "
        "instrument, never on a tick -- the policy_manifest disposition",
    ),
    "atlas_node_gate": (
        "permanent",
        "specs/022 T024: connectome-node naming census (every NODES id named by >=1 module "
        "in its region or dated-EXEMPT; 23 day-one phantoms recorded, ratchet down only). "
        "A CI/run_selfchecks instrument, never on a tick -- the policy_manifest disposition",
    ),
    # selfcheck_cache needs NO exemption: run_selfchecks.py is an external root and its
    # --carry import makes the module live-reachable -- the stale-exemption arm proved it
    # by refusing the row this comment replaces (the gate working, twice in one spec).
    # envelope_hash_gate's `permanent` row RETIRED at T053-ii (2026-08-31): reachable through
    # the same rule_composer import cluster, so an EXEMPT row would be the stale-exemption
    # red. It remains a CI/run_selfchecks instrument that reads and compares the five
    # brainstem files' bytes -- never on a tick, the envelope never its target (its docstring
    # and the T021b record carry that); reachability changes nothing about the approval act.
    # interoceptive_integration: EXEMPTION RETIRED 2026-08-22 (specs/022 T039f), and it is the
    # measurement that retired it, not the wire. The row promised a coalition consumer at
    # T030/T031; T039b measured that sink OUT OF RANGE and refused it. The row then named the
    # sink that does work -- threshold modulation, the producer's own claim -- and bound its own
    # retirement to "the measurement that shows a consumer's own output changing, and by nothing
    # else". That measurement is tests/test_interoceptive_ignition.py. DISCLOSED because it would
    # otherwise read as work: the single import in cortex/brain.py flips this region to CHARGED on
    # its own, consumer or no consumer. The import is the cheap half; the charge is the measured
    # one, and only the second is evidence.
    "replay_gate": (
        "permanent",
        "specs/022 T017: the G5 snapshot/replay criterion (RNG stream positions captured, "
        "restore asserts position equality, continuation byte-identical, omit-positions and "
        "parameter-mismatch refusals). A CI/run_selfchecks instrument that L7's battery will "
        "consume; never on a tick -- the policy_manifest disposition",
    ),
    "determinism_gate": (
        "permanent",
        "specs/022 T015: the determinism criterion over kernel_tick (T=1..16 byte-equal at a "
        "constructed fixed P, V_c/V_ermv zero, measured frac_bits, float-accumulation broken "
        "arm in-command). A CI/run_selfchecks instrument, never on a tick -- the "
        "policy_manifest disposition, same as lake_rebuild_gate",
    ),
    "kernel_tick": (
        "wire",
        "specs/022 T014: the two-phase deterministic tick over kernel_state (P fixed per "
        "lineage, integer delivery, delay rings). Consumers in this same kernel block: the "
        "determinism gate (T015) and replay gate (T017) drive it as instruments. THE PROMISED "
        "T033+ CONSUMER LANDED WITHOUT IMPORTING THIS MODULE -- episode_driver drives the "
        "brain path and carries only distinct-from prose about KernelTick (measured zero "
        "imports, US1 audit B3). The live consumer is the US3 kernel cutover (T064/T066/T070). "
        "BOUND EXPIRED AT T039 UNMET (recorded, not re-dated). Printed by the standing-TODO arm",
    ),
    "neuron_pager": (
        "wire",
        "specs/022 T066: activation-driven paging of unit blocks (residency policy, pinned hot "
        "set, one async batch per tick, integer ROUNDS budget, no silent drop of a spike to a "
        "paged-out unit). ITS CONSUMERS TODAY ARE ITS OWN SELF-CHECK AND tests/test_neuron_pager "
        ".py -- the pager is NOT on the live tick, and that is the row's own disclosure 2.6(g), "
        "not an omission: the live consumer is the S6 kernel cutover, which is where a paged "
        "substrate first carries traffic and where the cockpit's 'paging binding-width' row is "
        "owed. It charges lake_segments/lake_index/lake_bridge with an in-tree consumer, which "
        "their four retired rows named T066 for, and those three stay off the live tick for the "
        "same reason this does. Printed by the standing-TODO arm every run",
    ),
    "kernel_state": (
        "wire",
        "specs/022 T013: SoA substrate state (128B-aligned blocks, contiguous faculty slices, "
        "_T_STATE-backed). Consumer is the two-phase tick in the SAME kernel block "
        "(kernel_tick, T014, the very next task); the P2 retirement of the per-object neuron "
        "state it supersedes executes at the cutover, faculty by faculty, with parity A/Bs -- "
        "now the US3 kernel cutover (T064/T066/T070). BOUND EXPIRED AT T039 UNMET (US1 audit "
        "B3, 2026-08-25; recorded, not re-dated). Printed by the standing-TODO arm every run",
    ),
    "perf_baseline": (
        "permanent",
        "the pinned-host performance ratchet (specs/022 T001) -- a measurement instrument, not a "
        "tick faculty. Records/asserts wake-tick p50/p95 + D: IOPS against tools/perf_baseline.json; "
        "bare invocation is a cheap wiring check. Same disposition as policy_manifest above: "
        "reachable from run_selfchecks and CI, never from a tick",
    ),
    "import_manifest": (
        "permanent",
        "the cross-tier canary -- a CI self-check covering the governed -> ungoverned lazy imports "
        "this gate cannot see (reasoning/ and training/ are outside BRAIN_MODULES), not a tick faculty",
    ),
    "world_red_probe": (
        "permanent",
        "the world-red instrument (specs/022 Phase 1 W3) -- a measurement instrument, not a tick "
        "faculty. Same disposition as perf_baseline above: reachable from run_selfchecks (bare "
        "invocation is a 0.07 s wiring check over synthetic series), never from a tick. Wiring it "
        "into a live path would be the dead consumer P3's own honesty guard forbids -- it OBSERVES "
        "the two curiosity chains and must not perturb them",
    ),
}


def _mpath(pyfile):
    return (
        os.path.relpath(pyfile, os.path.dirname(_PKG_DIR)).replace("\\", "/")[:-3].replace("/", ".")
    )


def _all_paths():
    """Every in-package module path on disk (packages excluded -- an `__init__` is not a target)."""
    paths = set()
    for root, _d, files in os.walk(_PKG_DIR):
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                paths.add(_mpath(os.path.join(root, f)))
    return paths


# The second element of the old (paths, pkgs) return was bound to `_PKG_PATHS` and read by nothing
# tree-wide -- a computed value with no consumer, which is the shape this very gate exists to name.
_ALL_PATHS = _all_paths()


def _resolve(dotted, names):
    out = set()
    if not (dotted and dotted.startswith(_PKG)):
        return out
    if dotted in _ALL_PATHS:
        out.add(dotted)
    for nm in names:  # from PKG.a.b import c  -> submodule PKG.a.b.c
        if f"{dotted}.{nm}" in _ALL_PATHS:
            out.add(f"{dotted}.{nm}")
    if dotted == _PKG:  # from snn_data_lake import <flat alias>
        out |= {_CANON[nm] for nm in names if nm in _CANON}
    last = dotted.split(".")[-1]  # import snn_data_lake.<flat alias>
    if dotted not in _ALL_PATHS and last in _CANON:
        out.add(_CANON[last])
    return out


def _imports(pyfile):
    try:
        tree = ast.parse(open(pyfile, encoding="utf-8").read())
    except (OSError, SyntaxError):
        return set()
    cur_pkg = ".".join(_mpath(pyfile).split(".")[:-1])
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out |= _resolve(a.name, [])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = cur_pkg.split(".")
                base = (
                    ".".join(parts[: len(parts) - (node.level - 1)]) if node.level > 1 else cur_pkg
                )
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            out |= _resolve(mod, [n.name for n in node.names])
    return out


def _graph():
    g = {}
    for root, _d, files in os.walk(_PKG_DIR):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                mp = _mpath(p)
                key = mp[: -len(".__init__")] if f == "__init__.py" else mp
                g.setdefault(key, set())
                g[key] |= _imports(p)
    return g


# distinct-from: substrate/method_dormancy_gate.py -- its per-file import list serves the
#   METHOD gate's own narrow scan; this exposes the whole-package graph _graph() already
#   builds. Alias, not a second walk. Read.
def import_graph():
    """THE static import graph (module -> in-package imports), public. Consumers reuse
    this ONE walk rather than re-deriving it: selfcheck_cache's closure hashing rides on
    it (P2 -- one graph)."""
    return _graph()


def _external_roots():
    """In-package modules imported by REAL consumers that live OUTSIDE the scanned package.

    Why this exists: `_ENTRYPOINTS` is hand-maintained, so anything the daemon or an app under
    `integrations/` imports had to be remembered by a person. It was not, and the gate paid for it --
    `integrations/chat_app/web_server.py` held the ONLY working attachment of the plasticity stack
    while this gate reported that stack dormant. A hand-maintained allowlist cannot see a wire nobody
    wrote down, so derive the roots from the source instead of trusting memory.

    Scanned: repo-root scripts (learn_daemon.py and friends, non-recursive) + everything under
    `integrations/`. Deliberately NOT tests/ or research/ -- reachable only from a test is dormant,
    which is the distinction this whole gate exists to draw.
    """
    repo = os.path.dirname(os.path.dirname(_PKG_DIR))
    out = set()
    try:
        for f in os.listdir(repo):
            if f.endswith(".py"):
                out |= _imports(os.path.join(repo, f))
    except OSError:
        pass
    integ = os.path.join(repo, "integrations")
    for root, _d, files in os.walk(integ):
        for f in files:
            if f.endswith(".py"):
                out |= _imports(os.path.join(root, f))
    return out


def reachable():
    """Brain modules reachable (transitively) from any live root, console entrypoint, or an
    out-of-package consumer (root scripts + integrations/) discovered by _external_roots()."""
    g = _graph()
    seen = set(r for r in _LIVE_ROOTS + _ENTRYPOINTS + tuple(_external_roots()) if r in g)
    stack = list(seen)
    while stack:
        for dep in g.get(stack.pop(), ()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen & BRAIN_MODULES


def conditional_import_count():
    """How many in-package imports across the tree sit INSIDE a conditional.

    THIS GATE IS PERMISSIVE BY CONSTRUCTION, and the number is here so the headline `dormant 0`
    is read with that in mind. `_imports()` walks the AST unconditionally, so an import nested in
    `if <flag>:` confers reachability whether or not the flag is ever true -- and this tree has
    genuinely dead flags: `gain_adapt` is never passed True ANYWHERE, and `develop_env` is only
    passed True inside `cortex/live_brain.py`'s own `main()`.

    Making the walk guard-aware was tried and rejected: naively dropping every conditional import
    moves the count from `reachable 169 / dormant 0` to `reachable 94 / dormant 75`, which is
    nonsense -- most of those guards are flags the Brain DOES pass. Deciding which guards are dead
    needs per-flag constructor tracing, and a static pass that guesses would fail in the direction
    that retires live faculties. So the honest move is to report the surface, not to pretend to a
    precision this analysis does not have.

    An audit (2026-07-28, backlog H2) claimed honouring the two dead flags turns `dormant 0` into
    `dormant 9`. It does not reproduce: NEITHER dead flag guards an import. The mechanism it
    describes is real; the specific number was not."""
    n = 0
    for dp, _dn, fn in os.walk(_PKG_DIR):
        for f in fn:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dp, f), encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                for c in ast.walk(node):
                    if isinstance(c, (ast.Import, ast.ImportFrom)):
                        mod = c.module if isinstance(c, ast.ImportFrom) else c.names[0].name
                        if mod and _PKG in str(mod):
                            n += 1
    return n


def coverage():
    reach = reachable()
    exempt = {_CANON[m] for m in EXEMPT if m in _CANON}
    dormant = sorted(BRAIN_MODULES - reach - exempt)
    stale = sorted(
        m for m in EXEMPT if m in _CANON and _CANON[m] in reach
    )  # exempt but now reachable
    return {
        "total": len(BRAIN_MODULES),
        "reachable": len(reach),
        "exempt": len(exempt),
        "dormant": [d.replace(_PKG + ".", "") for d in dormant],
        "stale_exempt": stale,
        "conditional_imports": conditional_import_count(),
        # US1 audit B3: the module gate's STANDING-TODO arm, the method gate's shape. A
        # non-permanent row is open work; seven of them carried a bound ("S0 exit (T039),
        # not movable again") that came due with the gate silent -- the docstring's own
        # contract ("a 'retire'/'wire' entry that outlives its wave is a standing TODO")
        # had no printer. Report-only, never asserted: 7 rows at landing.
        "open_dispositions": sorted(
            (m, EXEMPT[m][0]) for m in EXEMPT if EXEMPT[m][0] != "permanent"
        ),
    }


def main():
    cov = coverage()
    print("=" * 66)
    print("DORMANCY GATE -- Policy 3: no brain module may be silently dormant")
    print("=" * 66)
    print(
        f"brain modules {cov['total']}  reachable {cov['reachable']}  "
        f"exempt {cov['exempt']}  dormant {len(cov['dormant'])}"
    )
    if cov["stale_exempt"]:
        print(
            f"NOTE: {len(cov['stale_exempt'])} EXEMPT entries are now reachable (retire them): "
            f"{cov['stale_exempt']}"
        )
    print(
        f"NOTE: reachability counts {cov['conditional_imports']} in-package imports that sit inside a "
        f"conditional. This gate is PERMISSIVE by construction -- an import under a never-taken guard "
        f"still confers reachability. See conditional_import_count()."
    )
    if cov["dormant"]:
        print("--- DORMANT (unreached AND unexempted) ---")
        for d in cov["dormant"]:
            print("  ", d)
    print("=" * 66)
    assert not cov["dormant"], (
        f"Policy 3: {len(cov['dormant'])} brain module(s) are dormant (not reachable from any live path "
        f"or entrypoint) and not in EXEMPT -- wire them, retire them, or add an EXEMPT reason: "
        f"{cov['dormant']}"
    )
    # ASSERTED 2026-08-15, having been COMPUTED and merely NOTE-printed since this gate was written.
    # An exemption that has stopped being true is not a harmless leftover: several EXEMPT reasons in
    # this table are the ONLY thing recording why a rule is being waived, and `growable_readout` /
    # `developmental_loop_brain` say in so many words that their Adam trainers are permitted "ONLY
    # while no live root reaches" them. Wiring one flips its exemption stale -- and the assert above
    # cannot see it, because becoming reachable makes `dormant` SMALLER, never larger. MEASURED by
    # adding one edge from cortex.brain to developmental_loop_brain: reachable 199 -> 207,
    # stale_exempt 0 -> 8, dormant 0 -> 0. The gate was green through all of it.
    # Currently 0, so this is a work list of length zero rather than an allowlist. The reachable red
    # is constructed in tests/test_eprop_train_tier_unreachable.py, which also holds the e-prop half
    # directly (this assert catches the CLASS; that one names the constraint).
    assert not cov["stale_exempt"], (
        f"Policy 3: {len(cov['stale_exempt'])} stale EXEMPT entry/entries -- now REACHABLE, so their "
        f"stated reason no longer holds -- retire the entry, or revert the wire that made it "
        f"reachable. Check the reason before deleting it: an exemption may be the only record of a "
        f"waived invariant. {cov['stale_exempt']}"
    )
    if cov["open_dispositions"]:
        print(
            f"STANDING TODO -- {len(cov['open_dispositions'])} non-permanent exemption(s) still "
            "open (module-gate arm, US1 audit B3; report-only):"
        )
        for m, disp in cov["open_dispositions"]:
            print(f"    {m} ({disp})")
    print("dormancy gate OK: every brain module is reachable from a live path or explicitly exempt")


if __name__ == "__main__":
    main()
