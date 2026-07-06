"""
NeuroChip -- a functional/architectural emulator of an event-driven neuromorphic processor.

Software twin of the neuromorphic silicon this repo already targets via nir_export (the
Loihi / SpiNNaker deploy path): multiple event-driven CORES, each holding a population of
LIFNeuron spiking cells, wired together by an on-chip SPIKE ROUTER that moves address-event
(AER) packets between cores. run() executes an existing spiking workload on the emulated chip
-- time-stepped, one router hop per tick -- and accounts spikes / SynOps / energy / latency, so
the whole neuromorphic stack runs locally in software WITHOUT physical silicon. That is the point.

Concept credit: the AER / mesh-routing execution model is Intel Loihi and BrainChip Akida; the
per-neuron dynamics reuse lif_neuron.LIFNeuron and the energy figures reuse energy.py verbatim.

CEILING (honest): this is a FUNCTIONAL/architectural emulator, NOT cycle-accurate silicon. No
analog circuit dynamics, no real place-and-route, no timing/thermal model; the workload is a small
synthetic spiking net. It reproduces the ARCHITECTURE (cores, AER routing, SynOp/energy/latency
accounting), not the physics. Every synapse is a fixed 1-tick router hop (same for intra- and
inter-core), which is exactly what makes the mapped computation invariant to core placement.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuro_chip
"""
# ponytail: additive/opt-in emulator, changes no existing behaviour; reuses LIFNeuron + energy.py.
# Ceiling: functional/architectural twin (AER-routed, time-stepped), NOT cycle-accurate silicon --
# no analog dynamics, no place-and-route, synthetic workload; software side of the nir_export target.
from snn_data_lake import energy
from snn_data_lake.lif_neuron import LIFNeuron


class Core:
    """One event-driven core: a population of LIFNeurons + a per-tick input-current accumulator."""

    def __init__(self, core_id, n_neurons, **lif_kwargs):
        self.id = core_id
        self.neurons = [LIFNeuron(**lif_kwargs) for _ in range(n_neurons)]
        self.currents = [0.0] * n_neurons        # current injected into each neuron THIS tick

    def inject(self, j, amount):
        """Accumulate synaptic/external current into local neuron j for the current tick."""
        self.currents[j] += amount

    def step(self):
        """Advance every neuron with its accumulated current; return local indices that spiked."""
        fired = [j for j, nrn in enumerate(self.neurons) if nrn.step(self.currents[j])]
        self.currents = [0.0] * len(self.neurons)    # drain the accumulator for the next tick
        return fired

    def reset_state(self):
        for nrn in self.neurons:
            nrn.reset_state()
        self.currents = [0.0] * len(self.neurons)


class NeuroChip:
    """Software twin of a neuromorphic chip: cores + an AER spike router (the routing table)."""

    def __init__(self):
        self.cores = []
        self.routes = {}         # (src_core, src_neuron) -> [(dst_core, dst_neuron, weight), ...]

    def add_core(self, n_neurons, **lif_kwargs):
        """Add a core of n_neurons LIFNeurons (shared **lif_kwargs); return its core id."""
        core_id = len(self.cores)
        self.cores.append(Core(core_id, n_neurons, **lif_kwargs))
        return core_id

    def connect(self, src_core, src_neuron, dst_core, dst_neuron, weight):
        """Add one AER fan-out edge: a spike at src is routed (1-tick hop) as `weight` into dst."""
        self.routes.setdefault((src_core, src_neuron), []).append((dst_core, dst_neuron, weight))

    def map_feedforward(self, layer_sizes, weights, cores_per_layer=1, **lif_kwargs):
        """Map a layered feedforward spiking net onto cores and build the AER table from the weight
        matrices (weights[l][i][j] = layer-l neuron i -> layer-(l+1) neuron j). Each layer's neurons
        are split contiguously across `cores_per_layer` cores. Returns (input_addrs, output_addrs),
        the global (core, neuron) addresses of the first and last layer."""
        layer_addr = []                                   # layer_addr[l][i] = global (core, local)
        for size in layer_sizes:
            c = max(1, min(cores_per_layer, size))
            base, rem = divmod(size, c)
            addrs = []
            for k in range(c):
                n_here = base + (1 if k < rem else 0)     # spread the remainder over the first cores
                cid = self.add_core(n_here, **lif_kwargs)
                addrs.extend((cid, local) for local in range(n_here))
            layer_addr.append(addrs)
        for li, w in enumerate(weights):
            for i, row in enumerate(w):
                sc, sn = layer_addr[li][i]
                for j, wij in enumerate(row):
                    if wij != 0.0:
                        dc, dn = layer_addr[li + 1][j]
                        self.connect(sc, sn, dc, dn, wij)
        return layer_addr[0], layer_addr[-1]

    def run(self, input_spikes, ticks, output_addrs=None):
        """Time-stepped AER event loop. input_spikes: {tick: [(core, neuron, current), ...]} of
        scheduled external drive. Each tick: (1) deliver AER packets emitted last tick as currents
        (the 1-tick router hop), (2) inject scheduled external inputs, (3) step every core, (4) for
        each spike emit AER packets to its routed targets, queued for next tick. Returns a metrics
        dict; if output_addrs is given, latency + a per-tick output_trace are tracked for them."""
        for core in self.cores:
            core.reset_state()
        out_index = {addr: k for k, addr in enumerate(output_addrs or [])}
        queue = []                                        # AER packets emitted last tick
        spikes = synops = 0
        per_core_spikes = [0] * len(self.cores)
        latency = -1
        output_trace = []
        for t in range(ticks):
            for dc, dn, w in queue:                       # (1) 1-tick router hop -> dst currents
                self.cores[dc].inject(dn, w)
            for c, n, cur in input_spikes.get(t, []):     # (2) scheduled external input spikes
                self.cores[c].inject(n, cur)
            queue = []
            row = [0] * len(out_index)
            for core in self.cores:                       # (3) advance every core this tick
                for j in core.step():
                    spikes += 1
                    per_core_spikes[core.id] += 1
                    targets = self.routes.get((core.id, j), ())   # (4) emit AER packets, queue them
                    synops += len(targets)
                    queue.extend(targets)
                    addr = (core.id, j)
                    if addr in out_index:
                        row[out_index[addr]] = 1
                        if latency < 0:
                            latency = t
            if out_index:
                output_trace.append(row)
        return {
            "spikes": spikes,
            "synops": synops,
            "energy_ac_j": energy.snn_energy(synops),                      # idealised 45nm AC lens
            "energy_loihi_j": energy.snn_energy(synops, energy.E_SYNOP_LOIHI),   # Loihi per-SynOp lens
            "ticks": ticks,
            "latency_to_first_output": latency,
            "per_core_spikes": per_core_spikes,
            "output_trace": output_trace,
        }


def main():
    print("=" * 60)
    print("NEUROCHIP -- functional emulator of a neuromorphic processor")
    print("=" * 60)

    # 1. ROUTING: a spike on core A crosses the AER router to core B after the 1-tick hop.
    chip = NeuroChip()
    a = chip.add_core(1)
    b = chip.add_core(1)
    chip.connect(a, 0, b, 0, 1.5)
    m = chip.run({0: [(a, 0, 1.5)]}, ticks=4, output_addrs=[(b, 0)])
    assert m["per_core_spikes"][a] == 1, "source core neuron must fire once"
    assert m["per_core_spikes"][b] == 1, "AER packet must cross to the OTHER core and fire it"
    assert m["latency_to_first_output"] == 1, "cross-core delivery is a 1-tick router hop"
    print(f"routing     : core {a} spike -> AER -> core {b} fires at tick "
          f"{m['latency_to_first_output']} (packet crossed cores)")

    # 2. FUNCTIONAL: a 2-layer net across >= 2 cores computes a specific input->output mapping.
    #    W[i][j] = input i -> output j ; here in0->out1, in1->out0 ; spiking in0 must fire out1.
    w = [[0.0, 1.5], [1.5, 0.0]]
    chip = NeuroChip()
    ins, outs = chip.map_feedforward([2, 2], [w], cores_per_layer=2)
    m = chip.run({0: [(ins[0][0], ins[0][1], 1.5)]}, ticks=4, output_addrs=outs)
    fired_out = [k for k, bit in enumerate(m["output_trace"][m["latency_to_first_output"]]) if bit]
    assert len(chip.cores) >= 2, "feedforward net must be mapped across >= 2 cores"
    assert fired_out == [1], "spiking in0 must fire output neuron 1 only (the intended computation)"
    print(f"functional  : 2-layer net on {len(chip.cores)} cores, in0 -> out{fired_out[0]} (as designed)")

    # 3. MAPPING-INVARIANCE: the SAME net + SAME input on 1 core vs distributed across N cores must
    #    give IDENTICAL spike behaviour -- routing/placement must not change the computation.
    def single_core():
        ch = NeuroChip()
        c0 = ch.add_core(4)                 # in0,in1 = local 0,1 ; out0,out1 = local 2,3
        ch.connect(c0, 0, c0, 3, 1.5)       # in0 -> out1
        ch.connect(c0, 1, c0, 2, 1.5)       # in1 -> out0
        return ch, [(c0, 0), (c0, 1)], [(c0, 2), (c0, 3)]

    def distributed():
        ch = NeuroChip()
        i2, o2 = ch.map_feedforward([2, 2], [w], cores_per_layer=2)      # each neuron on its own core
        return ch, i2, o2

    metrics = []
    ch = None
    for build in (single_core, distributed):
        ch, i_addr, o_addr = build()
        src = i_addr[0]                     # spike logical input 0 (address (0, 0) in both builds)
        metrics.append(ch.run({0: [(src[0], src[1], 1.5)]}, ticks=5, output_addrs=o_addr))
    keys = ("spikes", "synops", "latency_to_first_output", "output_trace")
    assert all(metrics[0][k] == metrics[1][k] for k in keys), \
        "1-core and N-core mappings must yield identical spike behaviour"
    print(f"invariance  : 1 core vs {len(ch.cores)} cores -> identical output_trace + spikes/synops "
          "(mapping-invariant)")

    # 4. ENERGY / SYNOPS accounting: one spike with fan-out 3 -> synops == 3 ; both energy lenses > 0.
    chip = NeuroChip()
    s = chip.add_core(1)
    d = chip.add_core(3)
    for j in range(3):
        chip.connect(s, 0, d, j, 0.1)       # sub-threshold fan-out: exactly 3 deliveries, no re-spike
    m = chip.run({0: [(s, 0, 1.5)]}, ticks=3)
    routed_spikes, fan_out = 1, 3
    assert m["synops"] == routed_spikes * fan_out, "SynOps must equal routed-spikes x fan-out"
    assert m["energy_ac_j"] == energy.snn_energy(m["synops"]), "AC energy must match energy.snn_energy"
    assert m["energy_loihi_j"] == energy.snn_energy(m["synops"], energy.E_SYNOP_LOIHI), \
        "Loihi-lens energy must match energy.snn_energy with the Loihi per-SynOp figure"
    assert m["energy_ac_j"] > 0 and m["energy_loihi_j"] > 0, "both energy figures must be positive"
    print(f"energy      : {m['synops']} SynOps -> AC {m['energy_ac_j']:.3e} J , "
          f"Loihi-lens {m['energy_loihi_j']:.3e} J (both > 0, via energy.py)")

    # 5. DETERMINISM: same net + same input -> identical metrics dict across two fresh runs.
    def build_run():
        ch2 = NeuroChip()
        i5, _ = ch2.map_feedforward([2, 2], [w], cores_per_layer=2)
        return ch2.run({0: [(i5[0][0], i5[0][1], 1.5)]}, ticks=6, output_addrs=[i5[0]])
    r1, r2 = build_run(), build_run()
    assert r1 == r2, "same input + mapping must give identical metrics"
    print("determinism : two fresh runs -> identical metrics dict")

    print("=" * 60)
    print("self-check OK: AER routing crosses cores; feedforward net computes correctly; "
          "1-core == N-core (mapping-invariant); SynOps/energy accounted; deterministic")


if __name__ == "__main__":
    main()
