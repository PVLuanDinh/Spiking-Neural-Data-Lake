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

  python -m snn_data_lake.gates.dormancy_gate   # (also run by run_selfchecks.py)
"""

import ast
import os

from snn_data_lake import REGIONS, TORCH_TIER, TRAIN_TIER, module_path

_PKG = "snn_data_lake"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../snn_data_lake

# canonical path per governed basename, over all three live tables (both annexes are brain modules,
# 2026-08-11: +TRAIN_TIER). G8 S1e (P2): derived from module_path(), the package's ONE
# basename-to-path mapping, which this gate used to compute a second time; basenames are unique
# (G8 invariant 3), so the two agreed entry for entry and one survives.
_CANON = {
    m: module_path(m) for t in (REGIONS, TORCH_TIER, TRAIN_TIER) for ms in t.values() for m in ms
}
BRAIN_MODULES = set(_CANON.values())

# The live-behaviour roots (the autonomous brain) + the shipped console-script entrypoints. A brain
# module reachable from ANY of these is "wired"; reachable only from tests/__main__ is dormant.
_LIVE_ROOTS = (
    f"{_PKG}.engine.brain",
    f"{_PKG}.engine.live_brain",
    f"{_PKG}.engine.autonomous_loop",
    f"{_PKG}.faculties.reentrant_workspace",
)
_ENTRYPOINTS = (  # pyproject [project.scripts] + the always-on daemon's in-package deps
    f"{_PKG}.infra.demo",
    f"{_PKG}.sensorimotor.live_arm",
    f"{_PKG}.sensorimotor.closed_loop",
    f"{_PKG}.infra.cockpit",  # THE frontend (served by the daemon); replaced control_panel
    f"{_PKG}.training.cli",
    f"{_PKG}.sensorimotor.web_ingest",
    f"{_PKG}.infra.nengo_bridge",
    f"{_PKG}.infra.region_wiring",  # the daemon charges faculties through this registry
)

# The S6 panel's dated correction, appended VERBATIM to both kernel `wire` rows. It is a SUCCESSION,
# never a rewrite: each row keeps its earlier reading word for word, because a row that quietly swaps
# in a new justification erases the evidence that the old one was wrong -- the discipline those two
# rows already apply to their own T064/T066/T070 reading. Full entries: evidence/s6-kernel-wire.md
# S3.7 (C1, C2, C3, C6, C8, C11, C12, C20, C21, C27).
_S6_PANEL_CORRECTION = (
    "CORRECTED 2026-09-09 (PANEL FOLD; everything above is kept VERBATIM). "
    "(1) T062 PLACES NO UNITS. PLAN-FINAL scopes its H01 ingest to the c3 SYNAPSE GRAPH composed "
    "onto growth/cortical_prior.py (:649-651), its decision table reads 'placement has no reader' "
    "(:774, :290), and its own same-arc dated note names GROWTH as the placer -- 'T062's OTHER "
    "reader stands unchanged (growth placing units at scale)' (:653-655); ~16k is a measured "
    "INGEST MILESTONE only (~5.0 MB at 302 B/unit, three orders under the 64 MB binding width; "
    "evidence/s2-state-per-unit.md 61-63, tasks.md T065). T062's own pre-registration DOES define "
    "an ingested population -- a KernelState + Synapses built from the H01 graph, off the live "
    "tick, consumed by T063 -- but this row's consumer is the first population ON THE TICK, which "
    "T062 does not put there. The live consumer is THE FIRST TICK POPULATION AT OR ABOVE N*, which "
    "today only growth can produce; T062 supplies the prior, not the population. This row is now "
    "dated to that CONDITION and not to a task id -- the third naming of this clause in two days "
    "-- so it names a measurable state that discharges itself instead of a row whose landing would "
    "leave it stale again. "
    "(2) N* by the profile derivation 2,504-2,828 across the five recorded readings (2,565 on the "
    "quiet extra), by the neuter derivation 642-6,316 (1,788 on the quiet extra), the two "
    "1.43-4.40x apart across those five (a panel re-read on a loaded host read 1.34). Each is its "
    "upstream statistic restated in width units -- N*_profile = K*((1-s)/s)*N with s the "
    "profiler's object-path share, N*_neuter = K*((1-c)/c)*N with c the Amdahl ceiling, "
    "K = tol/(1-tol) -- so the neuter's N* inherits the ceiling's 95% band, which is unbounded "
    "above on every determinate seed (seed 0 boot maps to [2,049, +inf), seed 1 to [2,045, +inf)); "
    "seed 2 boot's 642 is that formula at its INDETERMINATE ceiling 0.049. A POINT ESTIMATE at one "
    "measured width, its bound direction UNTESTED (c_u is measured at one width; the LOWER-bound "
    "reading holds only while the object path grows at most linearly in N). The committed "
    "'2,500-6,300 / 1.5-2.5x' above dropped seed 2 boot's 642 and 4.40 and the quiet extra's 1,788 "
    "and 1.43 while keeping 2,828 from that same boot reading; no single reading rule produced it. "
    "(3) THE CEILING, restated as what reproduces: measured at the deployed width (177-189 live "
    "LIF units) BELOW WHAT THE INSTRUMENT RESOLVES -- 0.005 raw on the two determinate recorded "
    "seeds (bands to 0.015), 0.017 on an UNREGISTERED quiet re-read of seed 0 (band to 0.038); the "
    "point estimate is not stable across readings (six panel re-readings of the same seeds read "
    "-0.03 to +0.03) and is NOT quotable as a number; the instrument's own matched A/A null reads "
    "0.01-0.03 run to run; every determinate 95% band's upper bound sits far under the 0.15 bar. "
    "The clause '95% bands inside 0.04' above is WITHDRAWN -- it is a property of one unlogged "
    "reading, and five of six fresh determinate bands breach it. Per-reading table: "
    "evidence/s6-kernel-wire.md S3.2. "
    "(4) The DECLINE, the 0.15 bar and this row's disposition are UNCHANGED by all of it."
)

# ------------------------------------------------------------------------------------------------
# EXEMPT: brain modules that are legitimately NOT reachable from a live path, each with a disposition
# and a reason. Dispositions: "retire" (pending deletion), "wire" (pending a verify-first wave),
# "permanent" (train-tier / benchmark / entry-serve / awaiting-prerequisite). The gate shrinks this
# list as Policy-3 waves land; a "retire"/"wire" entry that outlives its wave is a standing TODO.
# ------------------------------------------------------------------------------------------------
# key = module basename (unique across REGIONS per brain_atlas); value = (disposition, reason).
EXEMPT = {
    "memory_diagnostic": (
        "permanent",
        "G1's diagnostic harness (a governed benchmark); consumers: its CLI campaign and tests/test_memory_diagnostic.py",
    ),
    "recurrent_pool": (
        "permanent",
        "G1(b)'s selected pool, g1 LSM, a governed benchmark component until G1(c)'s disposition (g1b section 2.9); consumers: instruments/memory_diagnostic.py's pool arm, instruments/pool_selection.py's campaign, tests/test_recurrent_pool.py",
    ),
    "pool_selection": (
        "permanent",
        "G1(b)'s selection campaign (a governed benchmark; its record is g1b section 3.4) until G1(c)'s disposition; consumers: its CLI and tests/test_pool_selection.py",
    ),
    "carried_state_campaign": (
        "permanent",
        "G1(c)'s campaign module (a governed benchmark; evidence g1c-carried-state.md) until G1(c)'s disposition; consumers: its CLI and tests/test_carried_state_campaign.py",
    ),
    "brian2_backend": (
        "permanent",
        "the CI-asserted Brian2 reference simulator, the CPU parity oracle (owner decision 2, 2026-10-02: 'CPU ref + Torch/kernel GPU'); its last live-rooted importers (training/trainer, training/spiking_lm) retired 2026-10-02 (specs/022 evidence trainer-retirement.md); consumers: its __main__ self-check (run_selfchecks, ci.yml) and the call sites still listed in its MIGRATED_CALL_SITES",
    ),
    # so101_arm's "wire" exemption is GONE, as its own text predicted it should be: the gate is
    # on the live tick now (autonomous_loop, serial_arm's mirror, live_arm, cpg_oscillator,
    # spiking_pid, ring_attractor and the planner all drive it), so it is reachable and needs no
    # exemption. Leaving the entry would have been the quiet lie the soft-report exists to catch.
    # reaction_diffusion_compute: RETIRED AND DELETED 2026-08-02, as its own entry said it should be.
    # The exemption existed for one reason -- so the deletion would be a decision taken on its merits
    # rather than a side effect of the specs/004 planner rewrite -- and that decision is now taken.
    # What was checked before deleting, because "no consumer" is a claim and not an observation:
    #   * the only historical consumer, sensorimotor/wavefront_planner.py, is now a bidirectional RRT;
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
        "part of the organism it measures. "
        "CURRENT READING 2026-09-16 (T081 landed; evidence/t081-report-order.md 2.0 and "
        "fact 6, quoted): 'the row's wording says the probe runs on the T034/T035 report "
        "channel + EpisodeDriver. The report channel yes; the driver no -- fact 6 measures "
        "that the driver drives a kernel the composite does not include, and routing its "
        "trial patterns into the arm's bus would be a second stimulus mechanism beside the "
        "percept packs (P2).' Measured there: the composite the report channel reads has no "
        "KernelTick and no EpisodeDriver (grep over engine/brain.py and engine/live_brain.py "
        "found neither). T081 therefore did NOT route this driver, and the forward-client "
        "sentence above -- 'T081's report-order probe windowing over its trials' -- is "
        "SUPERSEDED, not deleted: it stands as the dated expectation this row disposed of in "
        "the open. The remaining forward client is T052's ESPP adoption benchmark",
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
    "binding_instrument": (
        "permanent",
        "MEASUREMENT INSTRUMENT (specs/022 T080 [US4], the C5 row temporal_binding): reads the "
        "LIVE CrossModalBinder's own per-tick report off a SCRATCH brain (throwaway_home, "
        "persist=False) whose audio pack is re-ordered at the STIMULUS, and scores it against a "
        "membership-matched permutation of that same recording at matched K. Its consumers are "
        "its __main__ self-check (2.4's six reachable reds on constructed inputs), "
        "tests/test_binding_instrument.py, and the T080 campaign (--t080-campaign, which pins "
        "tools/t080_temporal_binding.json once). Wiring it to the tick would make the observer "
        "part of the observed -- the same reason ignition_instrument is not a tick faculty",
    ),
    # `report_order_probe` RETIRED from EXEMPT 2026-09-16 (T081's PASS surfaces,
    # evidence/t081-report-order.md 2.6): it is REACHABLE now, because `engine/brain.py::Brain.state`
    # imports `_t081_top_margin` from it to publish `report_order_margin`. The reason is kept here
    # rather than deleted, because the gate's own advice is that an exemption may be the only record
    # of a waived invariant -- VERBATIM: "MEASUREMENT INSTRUMENT (specs/022 T081 [US4], the C5 row
    # report_order): reads the LIVE arm's report channel off a SCRATCH brain (throwaway_home,
    # persist=False) and scores it against LESIONED CLONES of the very workspace state that produced
    # it -- the reported content's attractor charge zeroed on one clone, an unreported pool's on
    # another. THE LESION IS APPLIED TO A COPY: the live workspace is never perturbed, and a clone's
    # subscribers and output ring are detached, so no copy can deliver a broadcast or write a lake
    # byte. Its consumers are its __main__ self-check (2.4's six reachable reds on constructed
    # workspaces), tests/test_report_order_probe.py, and the T081 campaign (--t081-campaign, which
    # pins tools/t081_report_order.json once). Wiring it to the tick would make the observer part of
    # the observed -- the same reason ignition_instrument and binding_instrument are not tick
    # faculties."
    #
    # WHAT CHANGED AND WHAT DID NOT, stated exactly rather than waved through. The RECORDER, the
    # CLONES and the CAMPAIGN are untouched: they still run on a scratch brain and still lesion only
    # copies, so "the lesion is applied to a copy" stands. What is now reachable is ONE pure reader,
    # `_t081_top_margin(ws)`, which takes `ws._slot()` and returns a ratio -- it mutates nothing,
    # clones nothing, and submits nothing. It is read on `Brain.state()`, the TELEMETRY view, and
    # its value is consumed by exactly two surfaces (brain_telemetry.PROVENANCE and the cockpit's
    # report-order row, measured by grep at this commit) and by no decision anywhere. So the
    # observer has NOT been wired into the tick; a number it computes is printed on the state the
    # cockpit renders, which is what 2.6's PASS branch requires and what P4c requires of any
    # reported quantity. If a future caller feeds this margin back into an action, THAT is the wire
    # the sentence above forbids, and it owes its own adjudication.
    "gwt_deep": (
        "permanent",
        "BENCHMARK DEMONSTRATION (promoted reasoning/ -> the thalamus package 2026-08-11; now faculties/): the tree's "
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
        "SELECTION-turned-CONTROL stage over faculties/latent_q's value table (promoted "
        "2026-08-12, #10 lane 4d): fuses value + metacognition on the loop's curiosity; "
        "composition-only. The EXEC-bypass P2 this entry used to record was ADJUDICATED "
        "2026-08-12: the consumer (instruments/developmental_loop_brain) deleted NORD_EXEC "
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
        "Its ONE consumer (instruments/developmental_loop_brain, all three methods on its main "
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
        "reasoning/brain_adapters/executive_control.py -> faculties/competence_executive.py "
        "2026-08-12, #10 lane 4f -- the LAST brain_adapters module; renamed because "
        "faculties/executive_control holds the basename and brain_atlas.py:81 asserts "
        "uniqueness). Three stdlib wrappers over governed mechanisms: dopamine RPE -> "
        "bounded lr multiplier (faculties/dopamine_nuclei), objective broadcast "
        "(faculties/global_workspace), competence planning (faculties/prefrontal_executive). "
        "Its ONE consumer is instruments/developmental_loop_brain, lazily inside main() "
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
        "settled in the cortex package, now faculties/ -- its consumer set IS the perception "
        "tier): ConvLatencyStdp is "
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
    # growth/placement.py's module-top import, which is what this gate walks (placement is
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
    # faculties/prefrontal_executive, the capacity is DERIVED there (K = T_max/t_s,
    # faculties/wm_capacity, five guards), and the file is gone. Exactly one mechanism
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
    # placement: EXEMPTION RETIRED 2026-08-16, the wire landed. `growth/neurogenesis_engine` calls
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
    "docs_only_battery": (
        "permanent",
        "a battery RUNNER for docs-only pushes (owner 2026-10-02), invoked from the command line by the "
        "operator before a push -- not a tick faculty; its self-check runs in run_selfchecks",
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
    # `conclusion_protocol` RETIRED from EXEMPT 2026-09-16 (T081, evidence/t081-report-order.md 2.6)
    # ON ITS OWN STATED CONDITION, kept here verbatim rather than paraphrased: "RETIRES when C4's
    # instruments import it to gate their own reporting, which would make it live-reachable and the
    # row the stale-exemption red". That is exactly what landed -- `reasoning/spike_chat_lake.py`'s
    # "are you conscious" reply now reads CONCLUSION for the rows that CARRY A READING and prints
    # each one's licensed AND refused sentence together, which is the row's own "no surface printing
    # a reading without its licensed sentence in view" clause enforced at the one surface where a
    # consciousness claim reaches a human. The gate read the stale entry on the same commit; the
    # entry is retired, never the gate.
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
    # SLEEP TICK -- `sleep/lake_consolidation.consolidate` and `substrate/home.KnowledgeLog.consolidate`
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
    # otherwise read as work: the single import in engine/brain.py flips this region to CHARGED on
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
        "imports, US1 audit B3). PRIOR READING, SUPERSEDED 2026-09-09 -- kept verbatim because "
        "this row's value is that it records its own past falsehood, and an exemption that "
        "quietly swaps in a new justification erases the evidence that the old one was wrong: "
        "'The live consumer is the US3 kernel cutover (T064/T066/T070).' None of the three "
        "carries the wire -- T064's deliverable is void, T066 was DELETED ON ITS NULL (T068 "
        "LOSS), T070 is an instrument. CURRENT READING 2026-09-09, dated to T062: the live "
        "consumer is the row that FIRST PLACES UNITS AT SCALE -- T062's H01 ingest, ~16k signed "
        "neurons -- or growth reaching the MEASURED width trigger N*, a LOWER BOUND of "
        "2,500-6,300 units at 0.15 of the tick (the profile and the neuter derivations disagree "
        "1.5-2.5x and both are reported, the instrument's own finding; evidence/s6-kernel-wire.md "
        "S3). S6 (2026-09-09) measured this wire's Amdahl ceiling AT THE DEPLOYED WIDTH (177-189 "
        "live LIF units) at 0.005-0.017 raw, 95% bands inside 0.04 on two determinate seeds (one "
        "seed INDETERMINATE twice), and DECLINED the conversion under specs/008 FR-010 / SC-006 "
        "(declined-with-measurement, 0 sites unaccounted), FR-004 read contrapositively, plan.md "
        "Standing Invariants 1 and 3, and contracts/conversion.md C1.3 -- 'a site whose ceiling "
        "is below the noise floor of the tick is DECLINED, and the decline carries its number'. "
        "CITATION CORRECTED 2026-09-09 (panel): this row cited FR-005, whose antecedent is a site "
        "measured SLOWER AFTER vectorizing. No array version was built, so the Conversion-site "
        "entity's measured array cost and equivalence result do not exist for this site and "
        "FR-005 is not reached; the rule applied is the one written first, only its citation was "
        "narrow. BOUND EXPIRED AT T039 UNMET (recorded, not re-dated). Printed by the "
        "standing-TODO arm. " + _S6_PANEL_CORRECTION,
    ),
    "kernel_state": (
        "wire",
        "specs/022 T013: SoA substrate state (128B-aligned blocks, contiguous faculty slices, "
        "_T_STATE-backed). Consumer is the two-phase tick in the SAME kernel block "
        "(kernel_tick, T014, the very next task); the P2 retirement of the per-object neuron "
        "state it supersedes executes at the cutover, faculty by faculty, with parity A/Bs. "
        "PRIOR READING, SUPERSEDED 2026-09-09 -- kept verbatim for the same reason as "
        "kernel_tick's, that a row which rewrites its own justification destroys the record of "
        "having been wrong: 'now the US3 kernel cutover (T064/T066/T070)'. None of the three "
        "carries the cutover -- T064's deliverable is void, T066 was DELETED ON ITS NULL (T068 "
        "LOSS), T070 is an instrument. CURRENT READING 2026-09-09, dated to T062: the live "
        "consumer is the row that FIRST PLACES UNITS AT SCALE -- T062's H01 ingest, ~16k signed "
        "neurons -- or growth reaching the MEASURED width trigger N*, a LOWER BOUND of "
        "2,500-6,300 units at 0.15 of the tick (two derivations, 1.5-2.5x apart, both reported; "
        "evidence/s6-kernel-wire.md S3). S6 (2026-09-09) measured the wire's Amdahl ceiling at "
        "the DEPLOYED width (177-189 live LIF units) at 0.005-0.017 raw, bands inside 0.04 on "
        "two determinate seeds, and DECLINED the conversion under specs/008 FR-010 / SC-006 "
        "(declined-with-measurement, 0 sites unaccounted), FR-004 read contrapositively, plan.md "
        "Standing Invariants 1 and 3, and contracts/conversion.md C1.3 -- 'a site whose ceiling "
        "is below the noise floor of the tick is DECLINED, and the decline carries its number'. "
        "CITATION CORRECTED 2026-09-09 (panel): this row cited FR-005, whose antecedent is a site "
        "measured SLOWER AFTER vectorizing, and no array version was built, so FR-005 is not "
        "reached. The SoA state therefore waits on the same trigger its consumer does. BOUND "
        "EXPIRED AT T039 UNMET (US1 audit B3, 2026-08-25; recorded, not re-dated). Printed by "
        "the standing-TODO arm every run. " + _S6_PANEL_CORRECTION,
    ),
    "h01_ingest": (
        "wire",
        "specs/022 T062, DATED 2026-09-09: the H01 c3 synapse-graph ingest and the population it "
        "constructs. Reachable from run_selfchecks (the offline gate: 8 refusals, each with its "
        "constructed red, on synthetic fixtures plus the tracked real-release header), NEVER from "
        "a tick -- the artifact is read ONCE at construction of a population and nothing on a live "
        "path reads it again, which is the specs/011 data-model.md 3 rule that makes a seeded "
        "value MOVABLE rather than pinned. THE NAMED CONSUMER IS T063, the three-arm "
        "seeded/random/wrong drift harness (instruments/prior_drift_harness.py), which is the row that "
        "turns this population into a beatable prior; this exemption retires the day T063 lands. "
        "Stated as a wire and not as 'permanent' deliberately: an ingest whose output nothing ever "
        "consumes is exactly the dormancy P3 names, and cortical_prior's own exemption is the "
        "precedent for how such a row is RETIRED once a live root reaches the module. Note what "
        "this row does NOT claim -- it is not the kernel wire. kernel_state's and kernel_tick's "
        "rows name T062's ingested population as their live consumer; constructing that population "
        "here does not by itself re-arm those, because a population built inside a self-check is "
        "an instrument, not a tick. "
        "APPENDED 2026-09-09, the text above kept VERBATIM (a row that rewrites its own "
        "justification destroys the record of what it used to claim): T063 LANDED AND ITS ANSWER "
        "WAS NO, so this row is RE-DATED rather than retired. The campaign read the PLANTED "
        "control MET at t = +9.2601 -- the instrument is NOT blind, so the verdict is read -- and "
        "then conjunct (b) NOT MET at t = -0.2277: the H01 WIRING BEYOND ITS MARGINALS does not "
        "beat its degree-, sign- and magnitude-matched null on that task. 2.9's clause for that "
        "combination is exact, and it is applied here: US3's beatable prior is recorded NOT MET "
        "for that component; `cortical_prior` and T062's artifact remain REFERENCE DATA, "
        "untouched; and THE SEEDED POPULATION CONSTRUCTOR `h01_ingest.population()` LOSES ITS "
        "PRIOR ROLE. What the sentence above calls 'the row that turns this population into a "
        "beatable prior' is the row that measured it is not one. THE CONSUMER IS UNCHANGED AND "
        "THE ROLE IS NOT: `instruments/prior_drift_harness` still reaches this module, now as the "
        "harness's FIXTURE -- the population an instrument constructs and scores -- and no longer "
        "as a prior anything is seeded from. This exemption therefore does NOT retire the day "
        "T063 lands, as the text above promised: T063 landed, and what it changed is the role, "
        "not the reachability. The narrowness of the claim is 2.2's own and is restated because "
        "it bounds what was just recorded -- the null RETAINS H01's degrees, signs and weight "
        "magnitudes, so this cannot separate 'H01 contributes nothing' from 'H01's useful "
        "information is exactly what both arms share', and nothing beyond that component is "
        "claimed. Also unchanged: this is still not the kernel wire, and kernel_state's and "
        "kernel_tick's rows are untouched by any of it -- an instrument that ticks a scratch "
        "population is not the live consumer those rows name (T063 2.8)",
    ),
    "prior_drift_harness": (
        "permanent",
        "specs/022 T063, DATED 2026-09-09: the three-arm seeded/random/wrong drift harness with "
        "SC-006's two reds, the PLANTED positive control and the FROZEN twin. A "
        "CI/run_selfchecks MEASUREMENT INSTRUMENT on a SCRATCH population -- it constructs its "
        "own artifact and its own KernelState/Synapses/KernelTick under a throwaway NORD_HOME and "
        "is never constructed BY a brain, which is the amdahl_gate / determinism_gate / "
        "perf_baseline shape exactly. Reachable from run_selfchecks (the offline check: red 1 on "
        "the constructed lr = 0 pin, red 2's predicate on its constructed input, the PLANTED "
        "control, the WRONG arm's measured validity precondition, and every nuisance refusal), "
        "never from a tick. It is the CONSUMER kernel_plasticity's row names; it is NOT the live "
        "consumer kernel_tick's and kernel_state's rows name, and T063 2.8 records that an "
        "instrument that ticks a scratch population discharges neither. "
        "APPENDED 2026-09-09 (the text above is kept VERBATIM, including the sentence about "
        "kernel_plasticity's row, because a row that rewrites its own justification destroys the "
        "record of what it used to say): THAT ROW IS GONE. `substrate/kernel_plasticity` was "
        "DELETED on its own retention criterion -- T063's campaign read t(d_f) = -13.0154 with "
        "the FROZEN twin ahead of the learning arm on every one of the 8 seeds, and 2.9 says a "
        "learner that moves weights without an attributed benefit is a null. The exemption's own "
        "warning was the thing that came true: it was stated as a wire and not as permanent "
        "BECAUSE a learner whose only caller is a measurement instrument is exactly the dormancy "
        "P3 names, and it retired by DELETION rather than by a live path reaching it. This module "
        "is unchanged in disposition -- still a CI/run_selfchecks instrument on a scratch "
        "population, still never on a tick -- but it is now a NO-LEARNER instrument: its offline "
        "check reads the no-learner refusal in red 1's slot (never the word PINNED), measures per "
        "arm that four diagnostic lines and both distances are zero, and keeps PLANTED, the WRONG "
        "arm's validity precondition, the D ladder, the projection, the settling clause, the "
        "campaign guard and every nuisance refusal. Conjunct (a) reads NOT READ from here on: "
        "drift is not measurable without a learner, and specs/011's FR-008/FR-009 stay OPEN",
    ),
    "attention_schema": (
        "permanent",
        "specs/022 T078 Part I instrument, DATED 2026-09-15: the G3 task-binding gate (MI between "
        "the attention control state at t and the committed action at t+1, against a "
        "2,000-permutation floor) and the G4 content-leakage gate (a held-out decode of the frame "
        "label from the candidate u_t against a 200-draw label-shuffle band, beside the matched "
        "content decode), plus their six reds on CONSTRUCTED streams. A CI/run_selfchecks "
        "MEASUREMENT INSTRUMENT on a SCRATCH brain (make_brain under throwaway_home, "
        "persist=False) -- it constructs its own brain and is never constructed BY one, which is "
        "the amdahl_gate / determinism_gate / perf_baseline shape exactly. Reachable from "
        "run_selfchecks (the six reds run on constructed streams with no brain and no lake), "
        "never from a tick. PART I ADDS NO WIRE: there is no attention schema in it, and "
        "brain.py's `_attend_real_stream` keeps `goal=None` byte-identical. The ONE live wire "
        "(Head A's endogenous band preference fed to LearnedAttention.attend as the top-down "
        "goal) lands in Part II (2.11) -- or the row STOPS at G3 / VOIDS at G4, in which case no "
        "ACS is ever built and THIS EXEMPTION IS RETIRED WITH THE ROW. It is stated as permanent "
        "rather than as a wire because the instrument is legitimately an instrument on either "
        "branch; what is conditional is whether a SECOND, live module follows it. "
        "2026-09-15, AFTER the campaign (the sentences above were written at 3485853, BEFORE it): "
        "the STOP branch fired -- G3 STOP beside a G4 PASS, "
        "specs/022-unified-brain-arc/evidence/t078-attention-schema.md 3.4 -- so Part II (2.11) is "
        "not built and no live wire follows this instrument",
    ),
    "amdahl_gate": (
        "permanent",
        "specs/008 Gate Zero / specs/022 S6: the Amdahl ceiling of a delivery-path conversion, read "
        "by neutering the per-object neuron path on a SCRATCH brain (throwaway NORD_HOME, "
        "persist=False) and re-timing paired wake ticks. A CI/run_selfchecks MEASUREMENT "
        "INSTRUMENT, never on a tick -- it constructs its own brains and is never constructed BY "
        "one. Same disposition as determinism_gate and perf_baseline: reachable from "
        "run_selfchecks, never from a tick",
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
    "everyday_battery": (
        "wire",
        "specs/022 Phase 4 row C0, DATED 2026-09-17: the L12 audit's 20-item everyday battery as a "
        "governed SCORER -- the frozen answer key, 2.3's ordered classifier and 2.5's five "
        "constructed reds. A CI/run_selfchecks MEASUREMENT INSTRUMENT: its bare invocation touches "
        "no brain, no lake and no network, and its only brain-touching path (`run_seed`, reached "
        "from `--c0-campaign` alone) constructs its own scratch brains -- make_brain(persist=False) "
        "under throwaway_home -- and is never constructed BY one. That is the attention_schema / "
        "report_order_probe / prior_drift_harness shape exactly. "
        "IT CANNOT BE WIRED TO A TICK, and the reason is P4b, not convenience: the answer path it "
        "scores is `Brain.respond(q, human=True)`, the chat lane, which INGESTS every question into "
        "the one durable lake. Twenty-four synthetic questions per run written into brain_lake.bin "
        "would be the instrument contaminating the thing it measures (C0 pre-registration fact 4), "
        "so its number reaches a surface only as a PINNED campaign reading that says so. "
        "THE RETIREMENT CONDITION, and it is a wire rather than a permanent BECAUSE one exists "
        "(2.6): the plan's C-R2 is the named consumer -- this scorer is what makes that red "
        "reachable at all, C-R2 CONSUMES it, and THIS EXEMPTION RETIRES THE DAY C2 LANDS. A scorer "
        "whose only caller is its own self-check past that day is exactly the dormancy P3 names",
    ),
    "move_gate": (
        "permanent",
        "specs/022 G8's G-R8 instrument (evidence g8-reorganization.md 2.6, landed at S1a): the gate "
        "census of a tree, its diff against a stage's pre-registered delta, and the planted reds "
        "that prove it can see. A CI/run_selfchecks instrument run once per G8 stage, never on a "
        "tick and never constructed by one: it is stdlib-only and imports nothing of the package "
        "it measures, because the tree it censuses is an export, not the running brain -- the "
        "policy_manifest disposition",
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


# distinct-from: gates/method_dormancy_gate.py -- its per-file import list serves the
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
    passed True inside `engine/live_brain.py`'s own `main()`.

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


# distinct-from: brain_atlas.py, gates/selfcheck_cache.py -- phrase overlap only, as with
#   sensorimotor/arm_hardware.py, instruments/everyday_battery.py and growth/ng_snn.py (a region file
#   list, a keying word, a torque, a concurrency conjunct, a unit prune); all five read. The nearest
#   real sibling, gates/eprop_transport_gate.py's stale_exemptions(), checks THAT gate's transport
#   registry against gradient findings; this checks THIS table's keys against _CANON.
# G8 S0b (specs/022 evidence g8-reorganization.md 2.4): `coverage()` reads only the EXEMPT keys that
# resolve (`m in _CANON`), so a key naming NO brain module -- a rename, a retirement, a typo --
# exempts nothing and was invisible. That is the phantom row, and main() asserts it.
def _phantom_rows(exempt):
    """EXEMPT keys absent from `_CANON`: rows that name no brain module at all."""
    return sorted(m for m in exempt if m not in _CANON)


def _assert_no_phantom(rows):
    assert not rows, (
        f"Policy 3: {len(rows)} phantom EXEMPT row(s) name no brain module (absent from REGIONS "
        f"and the torch/train tiers) -- a key that exempts nothing hides the module it once named; "
        f"re-key the row or retire it: {rows}"
    )


def _phantom_selfcheck():
    """The phantom assert's reachable red: a row naming a module that does not exist, planted in a
    COPY of EXEMPT (never the real table), must red the gate; the unplanted table must stay green."""
    planted = {**EXEMPT, "g8_s0b_planted_phantom": ("permanent", "planted; names no module")}
    try:
        _assert_no_phantom(_phantom_rows(planted))
    except AssertionError as exc:
        assert "g8_s0b_planted_phantom" in str(exc), f"the planted red names no row: {exc}"
    else:
        raise AssertionError("dormancy self-check: a planted phantom EXEMPT row did not red")
    _assert_no_phantom(_phantom_rows(EXEMPT))


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
        "phantom_exempt": _phantom_rows(EXEMPT),  # G8 S0b: asserted in main()
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
    # adding one edge from engine.brain to developmental_loop_brain: reachable 199 -> 207,
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
    _assert_no_phantom(cov["phantom_exempt"])  # G8 S0b, with its own planted red next
    _phantom_selfcheck()
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
