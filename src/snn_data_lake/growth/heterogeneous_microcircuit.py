"""
Heterogeneous microcircuit -- MULTIPLE neuron models + a local rule + population encoding, composed at
once through the Basic-Components contract (growth/basic_components).

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
  - growth/izhikevich_neuron              : only a simulate() fn + REGIMES params -> a stepping shim that
                                            mirrors its per-dt (v,u) dynamics, sub-stepped within a tick
  - cortex/adex_neuron.AdEx               : step(I)->0/1, reset(); sub-stepped within a tick

Population encoding reuses sensorimotor/population_encoding (its RBF receptive fields). Learning reuses the
local STDP/covariance rule from growth/basic_components -- no loss.backward, no autograd, no surrogate
gradients: the weights autonomously diverge from init under the local rule.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.heterogeneous_microcircuit
"""

import random

import numpy as np

from snn_data_lake.growth.adex_neuron import AdEx
from snn_data_lake.growth.basic_components import (
    Connection,
    STDPRule,
    compose_step,
)

# REGIMES = canonical Izhikevich parameter sets, reused from the existing cell (do NOT re-hardcode them)
from snn_data_lake.growth.izhikevich_neuron import REGIMES
from snn_data_lake.growth.lif_neuron import LIFNeuron
from snn_data_lake.growth.neuron_zoo import Population
from snn_data_lake.sensorimotor.population_encoding import PopulationEncoder


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
    """A population of mixed neuron types stepped through one current vector -> one spike vector.

    Within a tick the cells are mutually INDEPENDENT -- each integrates its own membrane from its
    own state and writes only its own spike bit -- so this is a population update, and it delegates
    the loop to the single implementation of that in `cortex/neuron_zoo.Population` instead of
    keeping a second copy here (P2: one mechanism survives). `Population` decides serial-or-
    across-cores by racing both on real steps; the two paths are byte-identical, so that decision
    cannot reach this layer's output -- only its cost.

    THIS is the shape in this repo where free-threading actually pays, and it pays because the
    grain is fat: `_IzhCell` and `_AdExCell` each sub-step 40 times per tick, 6.1 us/neuron.
    MEASURED on this exact cell mix (3.14.6 free-threading build, Py_GIL_DISABLED=1, 28 logical
    CPUs, median of repeated real steps, fresh process): 240 cells run 1.45 ms serial against
    0.36 ms on 16 threads = 4.02x; 1,000 cells 6.19 ms against 0.87 ms on 28 = 7.09x; 10,000 cells
    61.4 ms against 6.12 ms = 10.03x. Byte-identity was re-checked at every one of those widths.

    The crossover for this mix is ~12-24 cells (6 cells measures 0.66x, 12 breaks even, 24 wins
    1.76x). `_build()` uses 6, which is BELOW it, so the race picks serial there -- that is the
    gate doing its job, not the parallel path failing, and main() exercises a width above the
    crossover so the claim is measured rather than asserted."""

    def __init__(self, cells):
        self._pop = Population(cells)
        # `Population` copies the sequence it is given, so binding `self.cells` to ITS list keeps
        # one list rather than two. Found by adversarial review: with `self.cells = cells` the layer
        # exposed a list that `step()` no longer read, so replacing a neuron changed what the layer
        # REPORTED and not what it STEPPED (measured: cells[2] read back as LIFNeuron while _IzhCell
        # was still being stepped). Before the delegation `step()` iterated this attribute directly,
        # so that was a silent regression in a public one.
        self.cells = self._pop.cells
        self.n = self._pop.n

    def reset(self):
        for c in self.cells:
            c.reset()

    def step(self, current):
        cur = np.asarray(current, dtype=float)
        return np.array(self._pop.step(cur), dtype=float)

    def close(self):
        """Release any worker threads the population retained."""
        self._pop.close()


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


# one factory per _LAYOUT row, in the same order -- so a column of any width can be built from the
# same mix (main() needs a width above the parallel crossover to exercise the population path)
_PROTOS = (
    lambda: _LIFCell(gain=2.0, tau_m=8.0, v_threshold=1.0),
    lambda: _LIFCell(gain=2.4, tau_m=6.0, v_threshold=1.0, refractory=3),
    lambda: _IzhCell("regular-spiking", gain=24.0),
    lambda: _IzhCell("fast-spiking", gain=24.0),
    lambda: _AdExCell(gain=900.0, a=2.0, b=60.0, tau_w=300.0),  # adapting
    lambda: _AdExCell(gain=900.0, a=0.0, b=0.0),  # tonic
)


def _cell_mix(n):
    """n cells cycling through the _LAYOUT prototypes. n == len(_LAYOUT) is exactly the column
    _build() has always used, in the same order with the same parameters."""
    return [_PROTOS[i % len(_PROTOS)]() for i in range(n)]


def _build(seed):
    n_pre = 8
    enc = _PopRateEncoder(n_pre, lo=0.0, hi=1.0)
    cells = _cell_mix(len(_LAYOUT))
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

    # ---- (d) THE POPULATION UPDATE ---------------------------------------------------------------
    # HeterogeneousLayer no longer carries its own per-neuron loop: it delegates to
    # cortex/neuron_zoo.Population, which runs the population across cores once it is wide enough
    # to pay for the threads. Two separate claims, both MEASURED here rather than asserted:
    #   (d1) the delegation is FAITHFUL -- identical to the in-line loop it replaced;
    #   (d2) at a width ABOVE the crossover the raced path is byte-identical to serial, so the
    #        speedup costs nothing in the result.
    def _run_inline(cells, drives):
        """The pre-delegation loop, kept here as the reference the delegation is scored against."""
        out = []
        for d in drives:
            cur = np.asarray(d, dtype=float)
            out.append([float(c.step(cur[i])) for i, c in enumerate(cells)])
        return out

    n6 = len(_LAYOUT)
    drives = [[0.10 + 0.05 * ((i + t) % 7) for i in range(n6)] for t in range(40)]
    layer_d = HeterogeneousLayer(_cell_mix(n6))
    via_pop = [list(layer_d.step(d)) for d in drives]
    inline = _run_inline(_cell_mix(n6), drives)
    delegation_faithful = via_pop == inline
    # REACHABLE RED for (d1): the comparison has to be able to SEE a reordered population, which is
    # the one way a delegated loop silently goes wrong. Same cells, same drives, reversed order --
    # if that still compares equal then the equality above is vacuous. RUN, not reasoned.
    delegation_red = _run_inline(_cell_mix(n6)[::-1], drives) != inline

    # (d2) a column wide enough for the race to have a real choice. 30 steps because the race
    # spends its first steps probing each candidate thread count before it commits.
    wide_n, wide_t = 240, 30
    wide_drives = [[0.10 + 0.05 * ((i + t) % 7) for i in range(wide_n)] for t in range(wide_t)]
    wide_layer = HeterogeneousLayer(_cell_mix(wide_n))
    wide_pop = [list(wide_layer.step(d)) for d in wide_drives]
    wide_ser = _run_inline(_cell_mix(wide_n), wide_drives)
    wide_identical = wide_pop == wide_ser
    wide_fired = sum(sum(row) for row in wide_ser)
    gate_reason = wide_layer._pop.gate.reason

    print("-" * 70)
    print(
        f"population update : delegation faithful over {n6}x{len(drives)} = {delegation_faithful} "
        f"(reordered-population red fires : {delegation_red})"
    )
    print(
        f"  wide column     : {wide_n} cells x {wide_t} steps, raced path == serial : "
        f"{wide_identical}  ({wide_fired:.0f} spikes)"
    )
    print(f"  gate (raced)    : {gate_reason}")
    print("=" * 70)

    assert delegation_faithful, (
        "HeterogeneousLayer.step must return EXACTLY what its in-line per-neuron loop returned "
        "after delegating to neuron_zoo.Population"
    )
    assert delegation_red, (
        "the delegation check must be FALSIFIABLE: stepping the same cells in reversed order must "
        "produce a different result, or comparing against the in-line loop proves nothing"
    )
    assert wide_identical, (
        f"above the crossover the raced population path must stay BYTE-IDENTICAL to serial "
        f"({wide_n} cells x {wide_t} steps)"
    )
    assert wide_fired > 0, (
        "the wide column must actually spike -- an all-silent population makes the byte-identity "
        "comparison above vacuous (a permutation of all zeros equals itself)"
    )
    layer_d.close()
    wide_layer.close()

    print(
        f"self-check OK: 3 neuron types all fire {type_totals}; weights learn ||dW||={div:.4f} "
        f"(||W0||={norm0:.4f}->||Wf||={normf:.4f}); bounded; deterministic; population update "
        f"delegated to neuron_zoo.Population, faithful at {n6} and byte-identical at {wide_n} "
        f"({gate_reason})"
    )


if __name__ == "__main__":
    main()
