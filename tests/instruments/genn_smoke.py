"""specs/012 T004 pre-flight smoke: minimal GeNN model - build (nvcc compile), load, step.

Proves on THIS box, in .venv-lake: pygenn imports; GeNN's code generator, the host C++
compiler and CUDA agree; a compiled model steps and records spikes. Output is the evidence
(specs/012-finish-genn-port/evidence.md #preflight). ASCII only.

Run:  .venv-lake\\Scripts\\python tests\\instruments\\genn_smoke.py
(Lived at specs/012-finish-genn-port/ until 2026-08-11; moved because a spec-bundle .py is a
build the p4c ratchet counts, while tests/** is the instrument category it excludes.)
"""

import sys
import warnings

import numpy as np


def main():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from pygenn import GeNNModel, init_postsynaptic, init_weight_update

        gil_warns = [str(w.message) for w in caught if "GIL" in str(w.message)]
    import pygenn

    print("pygenn version: " + getattr(pygenn, "__version__", "?"))
    print("GIL warnings at import: " + (repr(gil_warns) if gil_warns else "none"))

    model = GeNNModel("float", "genn_smoke")
    model.dt = 1.0

    # 4 source neurons, one spike each at t = 1, 2, 3, 4 ms
    spike_times = np.array([1.0, 2.0, 3.0, 4.0])
    ssa = model.add_neuron_population(
        "In",
        4,
        "SpikeSourceArray",
        {},
        {"startSpike": np.arange(4), "endSpike": np.arange(1, 5)},
    )
    # pygenn 5 EGP init; older spelling kept as fallback so the smoke reports API drift
    # instead of crashing on it (the module docstring's own warning about field names).
    try:
        ssa.extra_global_params["spikeTimes"].set_init_values(spike_times)
    except (AttributeError, KeyError, TypeError) as e:
        print("EGP init via extra_global_params[] failed (" + repr(e) + "); trying legacy")
        ssa.set_extra_global_param("spikeTimes", spike_times)

    lif = model.add_neuron_population(
        "Out",
        2,
        "LIF",
        {
            "C": 1.0,
            "TauM": 10.0,
            "Vrest": 0.0,
            "Vreset": 0.0,
            "Vthresh": 0.5,
            "Ioffset": 0.0,
            "TauRefrac": 1.0,
        },
        {"V": 0.0, "RefracTime": 0.0},
    )
    lif.spike_recording_enabled = True

    model.add_synapse_population(
        "InOut",
        "DENSE",
        ssa,
        lif,
        init_weight_update("StaticPulse", {}, {"g": 1.0}),
        init_postsynaptic("DeltaCurr"),
    )

    print("building (nvcc compile)...")
    model.build()
    model.load(num_recording_timesteps=10)
    for _ in range(10):
        model.step_time()
    model.pull_recording_buffers_from_device()
    times, ids = lif.spike_recording_data[0]
    print("backend: " + type(getattr(model, "_backend", model)).__name__)
    print("out spikes: t=" + repr(list(np.asarray(times))) + " id=" + repr(list(np.asarray(ids))))
    n = len(np.asarray(times))
    assert n > 0, "no output spikes recorded - model ran but is inert"
    assert not np.any(np.isnan(np.asarray(times))), "NaN in recorded spike times"
    print("SMOKE OK: compiled, stepped on device, " + str(n) + " spikes recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
