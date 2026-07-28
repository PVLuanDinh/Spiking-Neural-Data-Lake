"""NengoGUI model -- the whole SNDL brain running LIVE as a spiking Nengo network.

Open the interactive browser GUI with the TORCH venv (so the torch-tier neurogenesis actually GROWS
neurons -- .venv-nengo has no torch, so grown_units stays flat there):
    D:\\AI-project\\spiking-neural-data-lake\\.venv-cu128\\Scripts\\nengo.exe -P 8765 research\\nengo_gui_brain.py
(.venv-cu128 has both torch + nengo). Watch `grown_units` / `reasoning_units` climb during the SLEEP
phases -- growth is sleep-gated, so this model uses a short day_period to reach sleep quickly.

NengoGUI loads the `model` network below, draws the brain-tap Node + one labeled spiking ENSEMBLE per
brain signal, and lets you press play to watch every organ's signal evolve as spikes in real time. Each
GUI simulation step calls brain.tick() once (the Node), so the custom stdlib+torch brain runs under
Nengo's clock; every numeric brain.state() signal (including the torch/nengo/pc TIER channels, exported
as scalars) is re-encoded into its own spiking NEF population. Observation-only -- no actuator route;
deployable to NengoLoihi. Right-click an ensemble in the GUI for a spike raster / value / firing-rate plot.
"""

import os

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
pass  # migrated: sys.path bootstrap removed (now an importable package)

import nengo  # noqa: E402

from snn_data_lake.cortex.brain import Brain  # noqa: E402
from snn_data_lake.substrate.nengo_bridge import brain_signals  # noqa: E402

# short day_period so the SLEEP-gated neurogenesis is reached quickly (grown_units grows during sleep).
_brain = Brain(seed=0, day_period=15)
_sig0 = brain_signals(_brain)

# a curated, readable subset so the GUI graph stays legible (falls back to the first 12 signals). The
# GROWTH signals lead: grown_units (torch neurogenesis) + reasoning_units climb during the sleep phases.
_SHOW = [
    k
    for k in (
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
    if k in _sig0
]
_keys = _SHOW or list(_sig0)[:12]
# count/growth signals grow past 1.0, so give them a generous NEF radius (else the ensemble saturates)
_COUNTS = {
    "grown_units",
    "reasoning_units",
    "wake_ticks",
    "sleep_ticks",
    "nights_elapsed",
    "lifelong_units",
}
_scales = {k: (25.0 if k in _COUNTS else max(1.0, 1.5 * abs(_sig0[k]))) for k in _keys}

model = nengo.Network(label="SNDL whole brain (live under Nengo)")
with model:

    def _tap(t):
        _brain.tick()  # one GUI sim step = one brain tick
        s = brain_signals(_brain)
        return [s.get(k, 0.0) for k in _keys]

    brain_node = nengo.Node(_tap, size_out=len(_keys), label="brain.tick() -> state()")
    for i, k in enumerate(_keys):
        ens = nengo.Ensemble(n_neurons=40, dimensions=1, radius=_scales[k], label=k)
        nengo.Connection(brain_node[i], ens, synapse=0.01)


if __name__ == "__main__":
    print(
        f"NengoGUI model ready: {len(model.ensembles)} labeled spiking ensembles over {len(_keys)} brain signals."
    )
    print(
        r"Launch the GUI:  D:\AI-project\.venv-nengo\Scripts\nengo.exe research\nengo_gui_brain.py"
    )
