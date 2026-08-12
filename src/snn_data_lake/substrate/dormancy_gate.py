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
        "wire",
        "TORCH-TIER growable readout (promoted 2026-08-11, #10 lane 4b): function-preserving "
        "Net2Net (1,0) growth + bit-exact freeze -- the ORIGIN copy the specs/007 T022 "
        "consolidation collapses the laminar duplicates into. Consumers are reasoning-tier "
        "(autonomous_brain the kept A/B control at :50 unconditional; developmental_loop_brain "
        "opt-in) until their own lanes; same conversion condition as hippocampal_memory",
    ),
    "brain_decision": (
        "wire",
        "SELECTION stage over basal_ganglia/latent_q's value table (promoted 2026-08-12, #10 "
        "lane 4d): fuses value + metacognition on the loop's curiosity; composition-only. Its "
        "ONE consumer (reasoning/developmental_loop_brain, which also BYPASSES choose() under "
        "NORD_EXEC=1 default -- the recorded P2 vs ExecutivePlanner) is p6_gate-pinned -- this "
        "exemption converts to reachable the commit that loop promotes; wiring anything else "
        "to satisfy the gate would be the fake consumer P3 names",
    ),
    "hippocampal_memory": (
        "wire",
        "ADAPTER over this region's own governed mechanisms (promoted 2026-08-11, #10 lane 4a): "
        "per-digit ReplayBuffers + a SleepConsolidator pass for the developmental training loop. "
        "Its ONE consumer (reasoning/developmental_loop_brain, all three methods on its main "
        "path) is p6_gate-pinned and its own promotion is specs/010-blocked -- this exemption "
        "converts to reachable the commit that loop promotes; wiring anything else to satisfy "
        "the gate would be the fake consumer P3 names",
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
    "cortical_microcircuit": (
        "permanent",
        "REFERENCE MODEL: Potjans-Diesmann 2014 validation dynamics with no input port -- its output is a "
        "deterministic constant given its build parameters, so it can drive nothing",
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
    "esp32_encoder_path": (
        "permanent",
        "AWAITING HARDWARE: real-sensor adapter; pyserial absent and arm_config SAFETY_CALIBRATED=False",
    ),
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
    "spiking_jepa": (
        "permanent",
        "TRAIN-TIER world model. Probed as an adopt-or-retire vs the live rate JepaWorldModel and it is "
        "NOT a drop-in: measured encode() output does not vary across 24 distinct live-width patterns, so "
        "swapping it in would silently deaden the planner's transition_fn. Kept, unadopted, honestly",
    ),
    "is_snn_forward": (
        "permanent",
        "LIBRARY/BENCHMARK: spiking-forward primitive tied to spiking_jepa, not a tick faculty",
    ),
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
    "hf_stream": (
        "permanent",
        "TOOL: offline data-prep CLI -- streams a licensed HF dataset into feature rows for a "
        "trainer. Same disposition as sensor_csv_ingest: it feeds the [train] tier, not the tick, "
        "and the brain must never make a blocking network call mid-control",
    ),
    "method_dormancy_gate": (
        "permanent",
        "the Policy-3 per-METHOD amendment's gate -- a CI self-check, not a tick faculty",
    ),
    "import_manifest": (
        "permanent",
        "the cross-tier canary -- a CI self-check covering the governed -> ungoverned lazy imports "
        "this gate cannot see (reasoning/ and training/ are outside BRAIN_MODULES), not a tick faculty",
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
    print("dormancy gate OK: every brain module is reachable from a live path or explicitly exempt")


if __name__ == "__main__":
    main()
