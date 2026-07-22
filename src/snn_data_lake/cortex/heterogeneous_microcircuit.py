"""
Heterogeneous microcircuit -- MULTIPLE neuron models + a local rule + population encoding, composed at
once through the Basic-Components contract (substrate/basic_components).

Real cortex is not one neuron type: it mixes regular-spiking, fast-spiking, bursting and adapting cells.
This circuit does the same -- it wires THREE different published neuron models into a single live-runnable
population and learns their afferent weights with a LOCAL plasticity rule:

    input scalar  --population encode-->  presynaptic vector
                  --Connection.propagate (masked matmul)-->  per-neuron current
                  --HeterogeneousLayer.step-->  spikes from {LIF, Izhikevich, AdEx} cells
                  --STDPRule.delta (local outer product) + Connection.update-->  weights learn

The neuron cells have DIFFERENT native interfaces, so each is wrapped in a one-line shim to the Node
contract `step(current)->spike, reset()` (we do NOT edit the cell files):
  - cortex/lif_neuron.LIFNeuron           : already step(current)->0/1, reset_state -> reset shim
  - cortex/izhikevich_neuron              : only a simulate() fn + REGIMES params -> a stepping shim that
                                            mirrors its per-dt (v,u) dynamics, sub-stepped within a tick
  - reasoning/adex_neuron.AdEx            : step(I)->0/1, reset(); sub-stepped within a tick

Population encoding reuses cortex/population_encoding (its RBF receptive fields). Learning reuses the
local STDP/covariance rule from substrate/basic_components -- no loss.backward, no autograd, no surrogate
gradients: the weights autonomously diverge from init under the local rule.

  python -m snn_data_lake.cortex.heterogeneous_microcircuit        # self-check
"""

import random

import numpy as np

# REGIMES = canonical Izhikevich parameter sets, reused from the existing cell (do NOT re-hardcode them)
from snn_data_lake.cortex.izhikevich_neuron import REGIMES
from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.population_encoding import PopulationEncoder
from snn_data_lake.reasoning.adex_neuron import AdEx
from snn_data_lake.substrate.basic_components import (
    Connection,
    STDPRule,
    compose_step,
)


# --------------------------------------------------------------------------------------------------
# Encoder adapter: reuse PopulationEncoder's RBF fields but expose bounded [0,1] activations as the
# presynaptic vector (an Encoder: encode(x) -> vector).
# --------------------------------------------------------------------------------------------------
class _PopRateEncoder:
    """Wraps cortex/population_encoding.PopulationEncoder; encode(x) returns its Gaussian activations."""

    def __init__(self, n, lo=0.0, hi=1.0):
        self.pop = PopulationEncoder(n=n, lo=lo, hi=hi, max_spikes=10)
        self.n = n

    def encode(self, x):
        return self.pop.activations(
            x
        )  # bounded RBF activations, reuses the population's receptive fields


# --------------------------------------------------------------------------------------------------
# Node shims: adapt each existing cell to step(scalar_current)->0/1 + reset(). A per-type input GAIN maps
# the dimensionless connection drive into each model's native current units (LIF ~O(1), Izhikevich ~O(10),
# AdEx ~O(100s) pA); stiff cells sub-step their own dt within one circuit tick. These are unit
# conversions / operating-envelope bounds, not behaviour-arbitrating thresholds.
# --------------------------------------------------------------------------------------------------
class _LIFCell:
    """LIFNeuron shim -> Node. reset_state() -> reset()."""

    def __init__(self, gain, tau_m=8.0, v_threshold=1.0, refractory=0):
        self.gain = gain
        self.cell = LIFNeuron(tau_m=tau_m, v_threshold=v_threshold, refractory=refractory)

    def reset(self):
        self.cell.reset_state()

    def step(self, current):
        return self.cell.step(self.gain * float(current))


class _IzhCell:
    """Izhikevich (v,u) stepping shim -> Node. Mirrors izhikevich_neuron.simulate's per-dt dynamics,
    sub-stepped within a tick; params reused from REGIMES. Input current clamped to the model's regime."""

    def __init__(self, regime, gain, substeps=40, dt=0.1, i_max=15.0):
        self.a, self.b, self.c, self.d = REGIMES[regime]
        self.gain, self.substeps, self.dt, self.i_max = gain, substeps, dt, i_max
        self.reset()

    def reset(self):
        self.v = -65.0
        self.u = self.b * -65.0

    def step(self, current):
        i = min(self.i_max, max(0.0, self.gain * float(current)))
        fired = 0
        for _ in range(self.substeps):
            self.v += self.dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i)
            self.u += self.dt * (self.a * (self.b * self.v - self.u))
            if self.v >= 30.0:  # spike + reset, exactly as izhikevich_neuron.simulate
                self.v = self.c
                self.u += self.d
                fired = 1
        return fired


class _AdExCell:
    """AdEx shim -> Node. Sub-steps AdEx.step within a tick; input current clamped to a physiological pA envelope."""

    def __init__(self, gain, substeps=40, i_max=1200.0, **adex_kwargs):
        self.gain, self.substeps, self.i_max = gain, substeps, i_max
        self.cell = AdEx(**adex_kwargs)

    def reset(self):
        self.cell.reset()

    def step(self, current):
        i = min(self.i_max, max(0.0, self.gain * float(current)))
        fired = 0
        for _ in range(self.substeps):
            if self.cell.step(i):
                fired = 1
        return fired


# --------------------------------------------------------------------------------------------------
# HeterogeneousLayer: a Node whose neurons are DIFFERENT types. step(current_vec) -> spike_vec.
# --------------------------------------------------------------------------------------------------
class HeterogeneousLayer:
    """A population of mixed neuron types stepped through one current vector -> one spike vector."""

    def __init__(self, cells):
        self.cells = cells
        self.n = len(cells)

    def reset(self):
        for c in self.cells:
            c.reset()

    def step(self, current):
        cur = np.asarray(current, dtype=float)
        return np.array([c.step(cur[i]) for i, c in enumerate(self.cells)], dtype=float)


# --------------------------------------------------------------------------------------------------
# Build + run the circuit.
# --------------------------------------------------------------------------------------------------
# neuron index -> (label, coarse type) so the self-check can prove EACH type fires
_LAYOUT = [
    ("LIF-regular", "LIF"),
    ("LIF-refractory", "LIF"),
    ("Izh-regular-spiking", "IZH"),
    ("Izh-fast-spiking", "IZH"),
    ("AdEx-adapting", "ADEX"),
    ("AdEx-tonic", "ADEX"),
]


def _build(seed):
    n_pre = 8
    enc = _PopRateEncoder(n_pre, lo=0.0, hi=1.0)
    cells = [
        _LIFCell(gain=2.0, tau_m=8.0, v_threshold=1.0),
        _LIFCell(gain=2.4, tau_m=6.0, v_threshold=1.0, refractory=3),
        _IzhCell("regular-spiking", gain=24.0),
        _IzhCell("fast-spiking", gain=24.0),
        _AdExCell(gain=900.0, a=2.0, b=60.0, tau_w=300.0),  # adapting
        _AdExCell(gain=900.0, a=0.0, b=0.0),  # tonic
    ]
    layer = HeterogeneousLayer(cells)
    conn = Connection(n_pre, len(cells), seed=seed, w_scale=0.5, w_max=1.0)
    rule = STDPRule(lr=0.02, decay=0.9, tau_baseline=20.0)
    return enc, layer, conn, rule


def run_circuit(seed=0, ticks=250):
    """Drive the heterogeneous circuit for `ticks` ticks; return (per-neuron spike counts, connection)."""
    enc, layer, conn, rule = _build(seed)
    layer.reset()
    rule.reset()
    rng = random.Random(seed + 777)
    counts = [0] * layer.n
    for _ in range(ticks):
        x = rng.random()  # deterministic input stream in [0,1]
        spike = compose_step(enc, layer, conn, rule, x)
        for i in range(layer.n):
            counts[i] += int(spike[i])
    return counts, conn


def main():
    counts, conn = run_circuit(seed=0)

    # per-type totals: prove every neuron TYPE actually fires
    type_totals = {}
    for (_label, typ), c in zip(_LAYOUT, counts):
        type_totals[typ] = type_totals.get(typ, 0) + c

    div = conn.divergence()
    norm0 = float(np.linalg.norm(conn.W0))
    normf = float(np.linalg.norm(conn.W))
    within_bounds = bool(((conn.W >= conn.w_min - 1e-12) & (conn.W <= conn.w_max + 1e-12)).all())

    # determinism: identical seed -> identical spikes AND identical learned weights
    counts_b, conn_b = run_circuit(seed=0)
    deterministic = bool(counts_b == counts and np.array_equal(conn.W, conn_b.W))

    print("=" * 70)
    print("HETEROGENEOUS MICROCIRCUIT -- 3 neuron models + local rule + population encoding")
    print("=" * 70)
    print(f"pre={conn.n_pre} (RBF population)  post={conn.n_post} (mixed types)  ticks=250")
    print(f"{'neuron':<24}{'type':>6}{'spikes':>9}")
    for (label, typ), c in zip(_LAYOUT, counts):
        print(f"{label:<24}{typ:>6}{c:>9}")
    print("-" * 70)
    print(f"per-type totals   : {type_totals}")
    print(f"weight norm       : ||W0||={norm0:.4f} -> ||Wf||={normf:.4f}")
    print(
        f"divergence        : ||Wf - W0||={div:.4f} (afferent weights learned under the local rule)"
    )
    print(f"bounded           : all weights in [{conn.w_min}, {conn.w_max}] = {within_bounds}")
    print(f"deterministic     : same seed -> identical spikes + weights = {deterministic}")
    print("=" * 70)

    # ---- self-check: deterministic asserts (real numbers from the runs above) ----
    # (a) all three neuron TYPES actually fire
    assert type_totals.get("LIF", 0) > 0, f"LIF neurons must fire ({type_totals})"
    assert type_totals.get("IZH", 0) > 0, f"Izhikevich neurons must fire ({type_totals})"
    assert type_totals.get("ADEX", 0) > 0, f"AdEx neurons must fire ({type_totals})"
    assert all(c > 0 for c in counts), f"every neuron must fire at least once ({counts})"
    # (b) the connection weights LEARN (diverge from init)
    assert div > 1e-2, (
        f"afferent weights must diverge from init under the local rule (div={div:.5f})"
    )
    assert within_bounds, "weights must stay soft-bounded in [w_min, w_max]"
    # (c) deterministic across two seed-equal runs
    assert deterministic, "same seed must reproduce identical spikes and weights"
    print(
        f"self-check OK: 3 neuron types all fire {type_totals}; weights learn ||dW||={div:.4f} "
        f"(||W0||={norm0:.4f}->||Wf||={normf:.4f}); bounded; deterministic"
    )


if __name__ == "__main__":
    main()
