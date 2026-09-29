"""NengoGUI model -- the whole SNDL brain running LIVE as a spiking Nengo network.

Open the interactive browser GUI with the TORCH venv (so the torch-tier neurogenesis actually GROWS
neurons -- a nengo-only venv has no torch, so grown_units stays flat there):
    .venv-cu128\\Scripts\\nengo.exe -P 8765 src\\snn_data_lake\\reasoning\\nengo_gui_brain.py
(.venv-cu128 has both torch + nengo). Watch `grown_units` / `reasoning_units` climb during the SLEEP
phases -- growth is sleep-gated, so this model uses a short day_period to reach sleep quickly.

NengoGUI loads the `model` network below, draws the brain-tap Node + one labeled spiking ENSEMBLE per
brain signal, and lets you press play to watch every organ's signal evolve as spikes in real time. Each
GUI simulation step calls brain.tick() once (the Node), so the custom brain runs under Nengo's clock;
every numeric brain.state() signal (including the torch/nengo/pc TIER channels, exported as scalars)
is re-encoded into its own spiking NEF population. Observation-only -- no actuator route; deployable
to NengoLoihi. Right-click an ensemble in the GUI for a spike raster / value / firing-rate plot.

FIXED 2026-08-04 -- P4 VIOLATION ON THE CONSTRUCTION PATH, plus an import-time side effect.

  * `Brain(seed=0, day_period=15)` was called DIRECTLY at module scope. P4 names exactly one
    sanctioned construction path, and `engine.brain.make_brain`'s own docstring names this consumer
    in its list: "Every consumer -- the always-on daemon, the chat servers, demos, THE NENGO BRIDGE
    -- obtains the brain through here instead of calling ``Brain(...)`` directly, so there is
    exactly one construction path to evolve and the replication guard can enforce that no second
    brain stack is stood up." This file was the counter-example to a policy that names it. The swap
    is behaviour-identical by inspection -- `make_brain(**kwargs)` is a bare `return Brain(**kwargs)`
    -- so it costs nothing and puts the last direct caller behind the guard.

  * That call ran at MODULE SCOPE, so merely IMPORTING this module constructed a whole Brain (which
    opens the durable lake) as a side effect. `run_selfchecks.py` executes part of the suite in ONE
    process, and P4c records the same shape as three latent bugs in `learn_daemon.py`. Construction
    is now deferred behind a module-level `__getattr__` (PEP 562): `model` is built on FIRST ACCESS,
    so `import snn_data_lake.reasoning.nengo_gui_brain` is inert and `module.model` still resolves
    for NengoGUI, which looks the attribute up on the loaded module.

  * Dead code removed: `_here` was assigned from `os.path.dirname(...)` and never read (the only
    use of `import os`), and a `pass  # migrated: sys.path bootstrap removed` no-op sat beside it.
    Both were residue of the package migration. E402 is already in the per-file ignore list for
    `reasoning/**`, so the `# noqa: E402` markers those lines existed to justify are gone too.

VERIFICATION LIMIT, stated plainly rather than implied: **nengo is NOT installed in `.venv-lake`**,
the deterministic gate environment, so nothing in this file could be executed here. What WAS
verified: the laziness, by AST -- no module-level statement constructs a Brain or touches nengo
(main() re-derives that check and asserts it, so it has a reachable red). What was NOT verified:
that NengoGUI still discovers `model` through PEP 562 on the GUI path, which needs `.venv-cu128`
and a browser. If that discovery turns out to require an eager module-scope binding, the fix is to
build eagerly under an explicit entry rather than to restore the direct `Brain(...)` call -- the P4
half stands either way.

`nengo` is imported INSIDE build_model rather than at module top, for the same reason: the gate
environment has no nengo, and a top-level import would make this file unimportable there, so the
two static criteria in main() -- which are the only ones checkable without the optional dependency
-- could not run at all. P5: the suite stays runnable without the optional tier.
"""

import ast

from snn_data_lake.engine.brain import make_brain
from snn_data_lake.infra.nengo_bridge import brain_signals

# a curated, readable subset so the GUI graph stays legible (falls back to the first 12 signals). The
# GROWTH signals lead: grown_units (torch neurogenesis) + reasoning_units climb during the sleep phases.
_SHOW = (
    "grown_units",
    "reasoning_units",
    "brain_alive",
    "access_index",
    "access_ignited",
    "novelty_live",
    "criticality",
    "phase",
    "wake_ticks",
    "reflex_armed",
    "arm_stopped",
    "nengo_tier",
    "pc_cortex_tier",
    "perception_tier",
)
# count/growth signals grow past 1.0, so give them a generous NEF radius (else the ensemble saturates)
_COUNTS = {
    "grown_units",
    "reasoning_units",
    "wake_ticks",
    "sleep_ticks",
    "nights_elapsed",
    "lifelong_units",
}

# P1 disposition of the three fixed values below, stated rather than left silent. None arbitrates a
# brain decision -- this module is observation-only and routes no actuator; they are DISPLAY scale
# for the NEF populations that re-encode the brain's own signals. `_COUNT_RADIUS` is a rendering
# headroom for unbounded counters, `_RADIUS_HEADROOM` keeps a bounded signal off the saturation
# edge, and `_N_NEURONS` is a compute-shape cap on how many spikes the GUI has to draw per signal.
_COUNT_RADIUS = 25.0
_RADIUS_HEADROOM = 1.5
_N_NEURONS = 40

_model = None


def build_model(seed=0, day_period=15):
    """Construct the one canonical Brain (via make_brain -- P4) and the Nengo network that taps it.

    Deferred out of module scope on purpose: see the header. `day_period` is short so the
    SLEEP-gated neurogenesis is reached quickly and `grown_units` visibly climbs in the GUI.
    """
    import nengo

    brain = make_brain(seed=seed, day_period=day_period)
    sig0 = brain_signals(brain)
    keys = [k for k in _SHOW if k in sig0] or list(sig0)[:12]
    scales = {
        k: (_COUNT_RADIUS if k in _COUNTS else max(1.0, _RADIUS_HEADROOM * abs(sig0[k])))
        for k in keys
    }

    net = nengo.Network(label="SNDL whole brain (live under Nengo)")
    with net:

        def _tap(t):
            brain.tick()  # one GUI sim step = one brain tick
            s = brain_signals(brain)
            return [s.get(k, 0.0) for k in keys]

        brain_node = nengo.Node(_tap, size_out=len(keys), label="brain.tick() -> state()")
        for i, k in enumerate(keys):
            ens = nengo.Ensemble(n_neurons=_N_NEURONS, dimensions=1, radius=scales[k], label=k)
            nengo.Connection(brain_node[i], ens, synapse=0.01)
    net.sndl_keys = keys  # what the GUI is showing, for the self-check and for callers
    return net


def __getattr__(name):
    """PEP 562: `model` is built on FIRST ACCESS, so importing this module constructs no Brain."""
    global _model
    if name == "model":
        if _model is None:
            _model = build_model()
        return _model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _module_scope_is_inert():
    """AST check: no top-level statement in THIS file constructs a Brain or calls into nengo.

    This is the criterion for the import-side-effect half of the fix, and it is checkable without
    nengo installed -- which matters, because the gate environment does not have nengo and could
    otherwise verify nothing here at all. Returns the list of offending (lineno, source) pairs.

    `build_model` ADDED to the watched set 2026-08-05, because the criterion was BLIND TO ITS OWN
    STATED RED. main() said "REACHABLE RED: move `_model = build_model()` to module scope and this
    lists it"; that mutation was constructed and RUN and this function returned `[]` -- the name
    `build_model` was not in the tuple below, so the one regression the check exists to prevent
    (calling the deferred constructor eagerly again) was exactly the one it could not see. Only the
    absent `nengo` import made the mutated file exit non-zero here, so in `.venv-cu128`, where the
    GUI actually runs, it would have passed green with a Brain built at import. A watch list is a
    claim about what it watches: the helper that constructs the thing has to be on it, not only the
    constructor it wraps.
    """
    tree = ast.parse(open(__file__, encoding="utf-8").read())
    bad = []
    watched = ("make_brain", "Brain", "build_model", "Network", "Ensemble", "Node", "Connection")
    for node in tree.body:  # TOP LEVEL only -- function bodies are exactly where this belongs
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                nm = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if nm in watched:
                    bad.append((sub.lineno, nm))
    return bad


def main():
    offenders = _module_scope_is_inert()
    print("=" * 88)
    print("NENGO GUI BRAIN -- the whole brain as a live spiking Nengo network (observation-only)")
    print("=" * 88)
    print(f"module-scope Brain/nengo construction sites: {len(offenders)} {offenders}")

    # (1) the import-side-effect fix, checked statically so it holds with or without nengo.
    #     REACHABLE RED, constructed and RUN 2026-08-05 (it did NOT fire before the watch-list fix
    #     recorded in _module_scope_is_inert -- see that docstring): move `_model = build_model()`
    #     to module scope, after build_model's def, and this exits 1 listing `(N, 'build_model')`.
    #     A module-scope `make_brain(...)` fires it too (also run).
    assert not offenders, (
        f"importing this module must construct nothing -- found {offenders}. A Brain built at "
        f"import time opens the durable lake for anyone who merely imports the file."
    )

    # (2) the P4 fix: this file must not name the Brain constructor directly anywhere.
    #     REACHABLE RED: restore `from ... import Brain` / `Brain(seed=0, ...)` and this fires.
    src = open(__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    direct = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Brain"
    ]
    assert not direct, (
        f"P4: construct the brain only via make_brain(); direct Brain(...) at lines {direct}"
    )
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "make_brain" in imported and "Brain" not in imported, (
        f"P4: this module must import make_brain, not Brain (imports: {sorted(imported)})"
    )

    # (3) the curated signal list must be a real subset of what the brain exports -- otherwise the
    #     GUI silently falls back to `list(sig0)[:12]` and shows something else entirely. This is
    #     the only check that needs a live Brain, so it is SKIPPED (loudly) without nengo, which is
    #     the state of the deterministic gate env. REACHABLE RED: rename a key in _SHOW -- verified
    #     2026-08-05 that this red is NOT reachable in `.venv-lake` (the mutation exits 0, because
    #     the skip below is taken first). It needs an env with nengo; stated rather than implied.
    #
    #     `except Exception` NARROWED to the nengo-absent case, 2026-08-05. It swallowed EVERY
    #     failure -- a broken build_model, a Brain that would not construct, a nengo API change --
    #     and printed "SKIPPED ... self-check OK (STATIC ONLY)" and `return 0`. That is the
    #     silent-degradation shape this tree records: a stage fails, writes a placeholder, exits 0.
    #     The skip is legitimate for exactly one cause (the optional dependency is not installed),
    #     so that is the only cause it now catches; anything else propagates and reddens the gate.
    try:
        net = build_model()
    except ModuleNotFoundError as exc:  # ONLY the optional-dependency case is a legitimate skip
        if exc.name != "nengo":
            raise
        print(f"live-model check SKIPPED: {type(exc).__name__}: {exc}")
        print(
            "self-check OK (STATIC ONLY): import is inert (0 module-scope construction sites) and "
            "the brain is obtained via make_brain (P4). The live-model check needs nengo and was "
            "NOT run -- this is reported, not counted as a pass."
        )
        return 0
    assert net.sndl_keys, "the model exposed no signals"
    assert set(net.sndl_keys) <= set(_SHOW), (
        f"the GUI fell back off the curated list -- showing {net.sndl_keys}"
    )
    assert len(net.ensembles) == len(net.sndl_keys), (
        f"{len(net.ensembles)} ensembles for {len(net.sndl_keys)} signals"
    )
    print(f"live model: {len(net.ensembles)} labeled ensembles over {len(net.sndl_keys)} signals")
    print(
        f"self-check OK: import inert, brain via make_brain (P4), {len(net.ensembles)} spiking "
        f"ensembles over the curated signal set {net.sndl_keys}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
