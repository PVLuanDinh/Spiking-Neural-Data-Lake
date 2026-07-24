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
runtime neuter-A/B probe (scratchpad/smoke_drive_charge.py) during a wiring wave -- too expensive for
CI. So this gate is the CHEAP first tier (catch built-but-never-imported); the A/B is the second tier.

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

from snn_data_lake import REGIONS, TORCH_TIER

_PKG = "snn_data_lake"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../snn_data_lake

# canonical path per REGIONS/torch basename (the module shares its region name only for hippocampus)
_CANON = {}
for _region, _mods in REGIONS.items():
    for _m in _mods:
        _CANON[_m] = f"{_PKG}.{_region}.{_m}" if _m != _region else f"{_PKG}.{_region}.{_region}"
for _region, _mods in TORCH_TIER.items():
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
    f"{_PKG}.substrate.control_panel",
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
    # NOTE: the 17 "retire"-tagged modules (wilson_cowan/RBM/kalman/kuramoto/cue_integration/olfactory x2/
    # 3 redundant seq-memories/neurogenesis-stdlib/language_brain+brain_reasoning/adaptive_spiking/lt_gate/
    # feedback_alignment/sensor_demo/coordinated_growth) were DELETED in the Policy-2/3 retirement wave --
    # they are gone from REGIONS, so they are no longer brain modules and need no exemption.
    # ---- "wire": valuable, just unplugged -> wired verify-first (neuter-A/B must CHARGE) in a wave ----
    "heterogeneous_microcircuit": (
        "wire",
        "Brain-Cog composed microcircuit; also fix its false 'live' claim",
    ),
    "cortical_microcircuit": ("wire", "canonical L2/3-L5 microcircuit, unplugged"),
    "plasticity_rule_stack": ("wire", "composed local-learning stack -> always-learning brain"),
    "homeostatic_plasticity_stack": ("wire", "homeostatic stack, unplugged"),
    "bcm_plasticity": ("wire", "BCM rule, part of the plasticity stack"),
    "inhibitory_stdp": ("wire", "iSTDP, part of the plasticity stack"),
    "homeostatic_scaling": ("wire", "synaptic scaling, part of the plasticity stack"),
    "intrinsic_plasticity": ("wire", "intrinsic excitability, part of the plasticity stack"),
    "short_term_plasticity": ("wire", "STP, part of the plasticity stack"),
    "competitive_learning": ("wire", "competitive learning, part of the plasticity stack"),
    "developmental_pruning": ("wire", "synaptic pruning, part of the plasticity stack"),
    "sensory_encoder_bank": ("wire", "opt-in multi-sensor front-end"),
    "spike_slicer": ("wire", "encoder-bank component"),
    "spike_pooling": ("wire", "encoder-bank component"),
    "neural_sampling": ("wire", "spiking Boltzmann sampler (also the RBM replacement)"),
    "sensor_csv_ingest": ("wire", "real-sensor CSV ingest pipeline"),
    "esp32_encoder_path": ("wire", "ESP32 real-sensor encoder path"),
    "stochastic_resonance": ("wire", "noise-benefit encoding, plausible live consumer"),
    "sc_deep_layers": ("wire", "superior-colliculus deep layers (saccade/orienting)"),
    "superior_colliculus": ("wire", "orienting/saccade map"),
    "population_vector": ("wire", "population-vector readout"),
    "entropy_coder": ("wire", "lake spike-stream compression"),
    "neuro_chip": ("wire", "neuromorphic chip energy model"),
    "neuro_chip_stream": ("wire", "streaming chip energy model"),
    "energy": ("wire", "spike-energy accounting"),
    "legendre_memory": ("wire", "kept sequence-memory (Legendre delay network)"),
    "hippocampus": ("wire", "core hippocampal memory, unplugged from the tick"),
    "rsnn_encoder": ("wire", "recurrent-spiking encoder"),
    "spiking_recurrent_cell": ("wire", "RSNN cell primitive"),
    "spiking_jepa": ("wire", "spiking world-model; adopt-or-retire vs rate jepa_world_model in B3"),
    "is_snn_forward": ("wire", "spiking-JEPA forward primitive; tied to spiking_jepa"),
    "pontine_relay": ("wire", "pontine relay (lower confidence)"),
    "reticular_formation": ("wire", "reticular arousal (lower confidence)"),
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
    "dormancy_gate": (
        "permanent",
        "the Policy-3 gate itself -- a CI self-check, not a tick faculty",
    ),
    "replication_guard": (
        "permanent",
        "the Policy-4 guard itself -- a CI self-check, not a tick faculty",
    ),
}


def _mpath(pyfile):
    return (
        os.path.relpath(pyfile, os.path.dirname(_PKG_DIR)).replace("\\", "/")[:-3].replace("/", ".")
    )


def _all_paths():
    paths, pkgs = set(), set()
    for root, _d, files in os.walk(_PKG_DIR):
        for f in files:
            if f.endswith(".py"):
                mp = _mpath(os.path.join(root, f))
                (pkgs.add(mp[: -len(".__init__")]) if f == "__init__.py" else paths.add(mp))
    return paths, pkgs


_ALL_PATHS, _PKG_PATHS = _all_paths()


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


def reachable():
    """Brain modules reachable (transitively) from any live root or console entrypoint."""
    g = _graph()
    seen = set(r for r in _LIVE_ROOTS + _ENTRYPOINTS if r in g)
    stack = list(seen)
    while stack:
        for dep in g.get(stack.pop(), ()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen & BRAIN_MODULES


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
