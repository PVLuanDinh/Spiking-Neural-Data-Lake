"""
NeuroChipStream -- an OUT-OF-CORE (disk-streaming) emulator of a neuromorphic processor.

AirLLM (github.com/lyogavin/airllm) runs a model far larger than RAM by LAYER-STREAMING: load one
transformer layer's weights from disk, compute, free it, load the next -- so peak resident memory is
the LARGEST SINGLE LAYER, not the whole model. This applies that idea to neuro_chip.NeuroChip: the
chip's CORES are persisted to disk (one shard file per core) and streamed in ONE AT A TIME each tick
-- load a core's weights + membrane state, apply the AER packets addressed to it, step its LIFNeurons,
save the state back, FREE it, move to the next core. Peak resident memory is therefore one core's
weights + state, not the whole chip, so the emulated neuron count is bounded by DISK, not RAM
(max_resident_cores == 1 -- proven in the self-check).

It reproduces neuro_chip's EXACT tick semantics (one 1-tick router hop per synapse; cores stepped in
id order; packets emitted this tick delivered next tick), so results are bit-for-bit identical to the
in-memory NeuroChip -- streaming changes only WHERE the weights live, never the computation.

Concept credit: AirLLM (layer streaming); the AER / mesh-routing model is Intel Loihi + BrainChip
Akida; per-neuron dynamics reuse lif_neuron.LIFNeuron, energy figures reuse energy.py, and the
in-memory reference + tick semantics are neuro_chip.NeuroChip (imported for the invariance check).

CEILING (honest): same functional/architectural ceiling as neuro_chip -- NOT cycle-accurate silicon,
no analog dynamics, no place-and-route, synthetic spiking workload. Streaming trades MEMORY for DISK
I/O (a shard load+save per core per tick -> slower per step), buying the ability to emulate chips far
larger than RAM; it adds no new physical fidelity.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.neuro_chip_stream
"""

# additive/opt-in out-of-core wrapper over neuro_chip; changes no existing behaviour and
# reuses LIFNeuron + energy + NeuroChip. Ceiling: functional/architectural twin, NOT cycle-accurate
# silicon -- no analog dynamics/place-and-route; trades RAM for per-tick disk I/O (AirLLM streaming).
import os
import shutil
import struct
import tempfile
from array import array

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate import energy
from snn_data_lake.substrate.neuro_chip import NeuroChip

_MAGIC = b"NCS1"
_HDR = struct.Struct("<4sIII")  # magic, core_id, n_neurons, n_edges
_LIF = struct.Struct("<ddddi")  # tau_m, v_threshold, v_reset, v_rest, refractory
_EDGE = struct.Struct("<IIId")  # src_local, dst_core, dst_neuron, weight
_DBL = array("d").itemsize
_INT = array("i").itemsize


class StreamingChip:
    """Out-of-core neuromorphic chip: per-core shards on disk, exactly ONE core resident at a time."""

    def __init__(self, shard_dir=None):
        self._dir = shard_dir or tempfile.mkdtemp(prefix="ncstream_")
        self._own_dir = shard_dir is None
        self._ncores = 0
        self._resident = 0
        self._max_resident = 0

    def __len__(self):
        return self._ncores

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @property
    def shard_dir(self):
        return self._dir

    # ---- shard binary I/O (array/struct, mirrors spike_telemetry_hub) -------
    def _path(self, cid):
        return os.path.join(self._dir, f"core_{cid:06d}.bin")

    def _write_shard(self, cid, lif, v, refr, edges):
        with open(self._path(cid), "wb") as f:
            f.write(_HDR.pack(_MAGIC, cid, len(v), len(edges)))
            f.write(
                _LIF.pack(
                    lif["tau_m"],
                    lif["v_threshold"],
                    lif["v_reset"],
                    lif["v_rest"],
                    lif["refractory"],
                )
            )
            f.write(array("d", v).tobytes())
            f.write(array("i", refr).tobytes())
            for src, dc, dn, w in edges:
                f.write(_EDGE.pack(src, dc, dn, w))

    def _read_shard(self, cid):
        with open(self._path(cid), "rb") as f:
            magic, core_id, n, ne = _HDR.unpack(f.read(_HDR.size))
            if magic != _MAGIC:
                raise ValueError(f"not an NCS shard (bad magic {magic!r})")
            tau_m, vth, vres, vrest, refrac = _LIF.unpack(f.read(_LIF.size))
            v = array("d")
            v.frombytes(f.read(_DBL * n))
            refr = array("i")
            refr.frombytes(f.read(_INT * n))
            edges = [_EDGE.unpack(f.read(_EDGE.size)) for _ in range(ne)]
        lif = {
            "tau_m": tau_m,
            "v_threshold": vth,
            "v_reset": vres,
            "v_rest": vrest,
            "refractory": refrac,
        }
        return core_id, n, lif, v, refr, edges

    # ---- build API (mirrors NeuroChip; writes/updates shards on disk) -------
    def add_core(self, n_neurons, **lif_kwargs):
        """Add a core of n_neurons LIFNeurons; write its initial state shard to disk; return its id."""
        cid = self._ncores
        lif = {
            "tau_m": lif_kwargs.get("tau_m", 10.0),
            "v_threshold": lif_kwargs.get("v_threshold", 1.0),
            "v_reset": lif_kwargs.get("v_reset", 0.0),
            "v_rest": lif_kwargs.get("v_rest", 0.0),
            "refractory": lif_kwargs.get("refractory", 0),
        }
        v = array("d", [lif["v_rest"]] * n_neurons)
        refr = array("i", [0] * n_neurons)
        self._write_shard(cid, lif, v, refr, [])
        self._ncores += 1
        return cid

    def _append_edges(self, src_core, new_edges):
        """Append outgoing AER edges to a source core's shard (load, extend, rewrite -- 1 resident)."""
        _, _, lif, v, refr, edges = self._read_shard(src_core)
        edges.extend(new_edges)
        self._write_shard(src_core, lif, v, refr, edges)

    def connect(self, src_core, src_neuron, dst_core, dst_neuron, weight):
        """Record one AER fan-out edge in the SOURCE core's shard (routed as a 1-tick hop into dst)."""
        self._append_edges(src_core, [(src_neuron, dst_core, dst_neuron, weight)])

    def map_feedforward(self, layer_sizes, weights, cores_per_layer=1, **lif_kwargs):
        """Map a layered feedforward net onto cores + build the AER table (weights[l][i][j] = layer-l
        neuron i -> layer-(l+1) neuron j), exactly like NeuroChip.map_feedforward. Edges are grouped
        per source core so each shard is rewritten once. Returns (input_addrs, output_addrs)."""
        layer_addr = []
        for size in layer_sizes:
            c = max(1, min(cores_per_layer, size))
            base, rem = divmod(size, c)
            addrs = []
            for k in range(c):
                n_here = base + (1 if k < rem else 0)  # spread the remainder over the first cores
                cid = self.add_core(n_here, **lif_kwargs)
                addrs.extend((cid, local) for local in range(n_here))
            layer_addr.append(addrs)
        buf = {}  # src_core -> [(src_local, dc, dn, w), ...]
        for li, w in enumerate(weights):
            for i, wrow in enumerate(w):
                sc, sn = layer_addr[li][i]
                for j, wij in enumerate(wrow):
                    if wij != 0.0:
                        dc, dn = layer_addr[li + 1][j]
                        buf.setdefault(sc, []).append((sn, dc, dn, wij))
        for sc, elist in buf.items():
            self._append_edges(sc, elist)
        return layer_addr[0], layer_addr[-1]

    # ---- out-of-core time-stepped run --------------------------------------
    def _reset_all(self):
        """Reset every core's membrane state on disk (v=v_rest, refr=0), one shard at a time."""
        for k in range(self._ncores):
            _, n, lif, v, refr, edges = self._read_shard(k)
            for j in range(n):
                v[j] = lif["v_rest"]
                refr[j] = 0
            self._write_shard(k, lif, v, refr, edges)

    def _step_core(self, k, q_pkts, in_pkts, out_index):
        """LOAD core k, apply currents, step its neurons, SAVE state, then FREE it (locals released on
        return). Returns (n_spikes, emitted_AER_packets, fired_output_indices)."""
        _, n, lif, v, refr, edges = self._read_shard(k)  # LOAD one shard into RAM
        self._resident += 1
        self._max_resident = max(self._max_resident, self._resident)
        routes = {}
        for src, rdc, rdn, rw in edges:
            routes.setdefault(src, []).append((rdc, rdn, rw))
        currents = [0.0] * n
        for dn, w in q_pkts:  # (1) 1-tick router hop -> dst currents
            currents[dn] += w
        for dn, cur in in_pkts:  # (2) scheduled external input spikes
            currents[dn] += cur
        neurons = [LIFNeuron(**lif) for _ in range(n)]  # reconstruct LIFNeuron state (v, refr)
        for j in range(n):
            neurons[j].v = v[j]
            neurons[j]._refr = refr[j]
        n_spikes, emitted, fired_out = 0, [], []
        for j in range(n):  # (3) advance EVERY neuron this tick
            if neurons[j].step(currents[j]):
                n_spikes += 1
                emitted.extend(routes.get(j, ()))  # (4) emit AER packets (queue for next tick)
                if (k, j) in out_index:
                    fired_out.append(out_index[(k, j)])
        for j in range(n):
            v[j] = neurons[j].v
            refr[j] = neurons[j]._refr
        self._write_shard(k, lif, v, refr, edges)  # SAVE updated membrane state back to disk
        self._resident -= 1  # FREE core k (frame locals released on return)
        return n_spikes, emitted, fired_out

    def run(self, input_spikes, ticks, output_addrs=None):
        """Out-of-core AER event loop reproducing NeuroChip.run bit-for-bit. Each tick, cores are
        streamed IN ORDER: load shard k, apply last tick's AER packets + scheduled inputs addressed to
        k, step, emit packets (queued for next tick), save state, free k before loading k+1. A small
        sparse packet queue lives in RAM (spikes are sparse). Returns NeuroChip's metrics dict plus
        max_resident_cores (== 1 -- only one core's weights are ever resident)."""
        self._reset_all()
        out_index = {addr: k for k, addr in enumerate(output_addrs or [])}
        queue = []  # AER packets emitted last tick
        spikes = synops = 0
        per_core_spikes = [0] * self._ncores
        latency = -1
        output_trace = []
        self._resident = self._max_resident = 0
        for t in range(ticks):
            q_by_core = {}
            for dc, dn, w in queue:  # bucket last-tick packets by dst core
                q_by_core.setdefault(dc, []).append((dn, w))
            in_by_core = {}
            for c, n, cur in input_spikes.get(t, []):  # bucket scheduled inputs by core
                in_by_core.setdefault(c, []).append((n, cur))
            queue = []
            row = [0] * len(out_index)
            for k in range(self._ncores):  # stream cores in id order (== NeuroChip)
                n_spk, emitted, fired_out = self._step_core(
                    k, q_by_core.get(k, ()), in_by_core.get(k, ()), out_index
                )
                spikes += n_spk
                per_core_spikes[k] += n_spk
                synops += len(emitted)  # SynOps = routed-spikes x fan-out
                queue.extend(emitted)
                for oi in fired_out:
                    row[oi] = 1
                    if latency < 0:
                        latency = t
            if out_index:
                output_trace.append(row)
        return {
            "spikes": spikes,
            "synops": synops,
            "energy_ac_j": energy.snn_energy(synops),  # idealised 45nm AC lens
            "energy_loihi_j": energy.snn_energy(
                synops, energy.E_SYNOP_LOIHI
            ),  # Loihi per-SynOp lens
            "ticks": ticks,
            "latency_to_first_output": latency,
            "per_core_spikes": per_core_spikes,
            "output_trace": output_trace,
            "max_resident_cores": self._max_resident,  # peak cores held in RAM at once (must be 1)
        }

    def close(self):
        """Delete this chip's shard files (and the temp dir it owns). Safe to call more than once."""
        if self._own_dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        else:
            for k in range(self._ncores):
                try:
                    os.remove(self._path(k))
                except OSError:
                    pass
        self._ncores = 0


def main():
    print("=" * 64)
    print("NEUROCHIP-STREAM -- out-of-core (disk-streaming) neuromorphic emulator")
    print("=" * 64)

    # small multi-core feedforward net shared by the invariance + determinism checks
    ls = [2, 3, 2]
    w = [
        [[1.5, 0.0, 1.5], [0.0, 1.5, 0.0]],  # layer0(2) -> layer1(3)
        [[1.5, 0.0], [1.5, 1.5], [0.0, 1.5]],  # layer1(3) -> layer2(2)
    ]

    def drive(ins):
        return {0: [(ins[0][0], ins[0][1], 1.5), (ins[1][0], ins[1][1], 1.5)]}

    # 1. STREAMING-INVARIANCE: same net + input on in-memory NeuroChip and out-of-core StreamingChip.
    mem = NeuroChip()
    ins_m, outs_m = mem.map_feedforward(ls, w, cores_per_layer=2)
    m_mem = mem.run(drive(ins_m), ticks=8, output_addrs=outs_m)
    stream = StreamingChip()
    inv_dir = stream.shard_dir
    ins_s, outs_s = stream.map_feedforward(ls, w, cores_per_layer=2)
    m_str = stream.run(drive(ins_s), ticks=8, output_addrs=outs_s)
    inv_keys = ("spikes", "synops", "latency_to_first_output", "output_trace", "per_core_spikes")
    assert ins_m == ins_s and outs_m == outs_s, "address maps must match in-memory vs streaming"
    for key in inv_keys:
        assert m_mem[key] == m_str[key], f"streaming changed {key}: {m_mem[key]} != {m_str[key]}"
    assert m_mem["energy_ac_j"] == m_str["energy_ac_j"], "energy must match (identical synops)"
    assert m_str["spikes"] > 0, "invariance net must actually spike (else the test is trivial)"
    print(
        f"invariance  : in-memory vs streaming -> spikes {m_str['spikes']} (==), "
        f"synops {m_str['synops']} (==), output_trace + per_core identical"
    )

    # 2. BOUNDED MEMORY: only ONE core's weights+state are ever resident -> chip can exceed RAM.
    assert m_str["max_resident_cores"] == 1, (
        f"peak resident cores must be 1, got {m_str['max_resident_cores']}"
    )
    print(
        f"bounded mem : max_resident_cores = {m_str['max_resident_cores']} "
        f"(peak RAM = ONE core of {len(stream)}; neuron count bounded by DISK, not RAM)"
    )

    # 3. SCALE: tens of cores / hundreds of neurons -- wasteful to hold densely in RAM -- run streaming.
    n_layers, width, cpl = 12, 16, 4
    big_ls = [width] * n_layers
    diag = [[1.5 if i == j else 0.0 for j in range(width)] for i in range(width)]
    big_w = [diag for _ in range(n_layers - 1)]  # identity forwarding -> a spiking cascade
    big = StreamingChip()
    ins_b, outs_b = big.map_feedforward(big_ls, big_w, cores_per_layer=cpl)
    drive_b = {0: [(ins_b[i][0], ins_b[i][1], 1.5) for i in range(8)]}
    m_big = big.run(drive_b, ticks=n_layers + 4, output_addrs=outs_b)
    n_cores_big, n_neurons_big = len(big), n_layers * width
    assert m_big["max_resident_cores"] == 1, "scale run must also stay at 1 resident core"
    assert m_big["spikes"] > 0, "scale run produced no spikes"
    print(
        f"scale       : {n_neurons_big} neurons across {n_cores_big} cores -> "
        f"{m_big['spikes']} spikes, {m_big['synops']} synops"
    )
    print(
        f"              energy AC {m_big['energy_ac_j']:.3e} J , "
        f"Loihi-lens {m_big['energy_loihi_j']:.3e} J (via energy.py); peak resident 1 core"
    )

    # 4. DETERMINISM: same net + input -> identical metrics across two fresh streaming runs.
    def fresh_run():
        s = StreamingChip()
        i2, o2 = s.map_feedforward(ls, w, cores_per_layer=2)
        r = s.run(drive(i2), ticks=8, output_addrs=o2)
        s.close()
        return r

    r1, r2 = fresh_run(), fresh_run()
    assert r1 == r2, "two fresh streaming runs must give identical metrics"
    print("determinism : two fresh streaming runs -> identical metrics dict")

    # ---- clean up all shard files (no temp leak) ----
    inv_dir_existed = os.path.isdir(inv_dir)
    big_dir = big.shard_dir
    stream.close()
    big.close()
    assert inv_dir_existed and not os.path.isdir(inv_dir), "temp shard dir must be removed on close"
    assert not os.path.isdir(big_dir), "scale-chip shard dir must be removed on close"

    print("=" * 64)
    print(
        "self-check OK: streaming == in-memory (bit-for-bit invariant), max_resident_cores=1 "
        f"(RAM-bounded), scaled to {n_neurons_big} neurons/{n_cores_big} cores, deterministic; "
        "temp shards removed"
    )


if __name__ == "__main__":
    main()
