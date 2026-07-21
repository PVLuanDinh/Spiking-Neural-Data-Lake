"""Connectome router -- turn the STATIC connectome schematic into a LIVE spiking propagation layer.

`connectome.py` is a static typed / directed / signed region->region graph: it asserts topological
facts (loop closure, the single gated actuator path) but nothing PROPAGATES spikes along it. This
module makes each connectome edge a signed, delayed synaptic projection between small LIF region
pools, so a spike injected into one region flows to its downstream targets, inhibitory edges
subtract from their targets, neuromodulatory edges scale their target's gain, and signed positive
loops REVERBERATE (activity persists after the drive is removed) while feed-forward chains decay.

Each connectome NODE (a canonical brain sub-structure such as `ec`, `thal_relay`, `ctx_motor`) is
one small pool of leaky integrate-and-fire neurons -- the same LIF dynamics as
`cortex/lif_neuron.py` (leak toward rest, integrate current, spike + reset at threshold), here
numpy-vectorized across the whole connectome. Per-neuron threshold heterogeneity turns a graded
drive in [0, 1] into a graded pool firing RATE. The wiring is read from `connectome.py`'s public
`NODES` / `EDGES` (never edited).

Representation-only: this simulates spike propagation over the wiring schematic. It imports no arm /
brainstem module and drives no hardware -- the effector nodes (`actuator`, `viscera`) are inert LIF
sink pools like any other, integrating spikes but commanding nothing.

  python -m snn_data_lake.thalamus.connectome_router   # self-check, exit 0
"""

import math

import numpy as np

from snn_data_lake import connectome

_SIGN_EXC = "exc"
_SIGN_INH = "inh"
_NEUROMOD = ("da", "ne", "5ht", "ach")  # diffuse chemical classes -> scale target gain


class ConnectomeRouter:
    """Spike-propagation router over the static connectome.

    One small LIF pool per connectome node; every connectome edge becomes a signed, delayed
    synaptic projection (exc adds, inh subtracts, neuromod scales the target's gain).

    API:
      ConnectomeRouter(seed=0)
      .regions                       -> list of node ids (one LIF pool each)
      .step(region_drive: dict)      -> {region: spike_rate in [0,1]} for this tick
      .reverberation(region, ticks)  -> float: mean network activity persisting after drive stops
    """

    def __init__(
        self,
        seed=0,
        pool_size=24,
        tau_m=8.0,
        v_threshold=1.0,
        thr_spread=0.4,
        syn_scale=2.0,
        drive_scale=1.4,
        nmod_gain=0.6,
        default_delay=1,
    ):
        self.rng = np.random.default_rng(seed)
        self.regions = list(connectome.NODES.keys())
        self.n = len(self.regions)
        self._idx = {r: i for i, r in enumerate(self.regions)}
        self.K = int(pool_size)
        self.decay = math.exp(-1.0 / tau_m)  # membrane leak per tick
        self.syn_scale = float(syn_scale)
        self.drive_scale = float(drive_scale)
        self.nmod_gain = float(nmod_gain)

        # per-neuron threshold heterogeneity (deterministic from seed) -> a graded pool rate
        u = self.rng.random((self.n, self.K))
        self.thr = v_threshold * (1.0 + thr_spread * (u - 0.5))

        # vectorize the edge list into parallel arrays (source, target, weight, delay, sign masks)
        src, tgt, w, dly, sgn = [], [], [], [], []
        for e in connectome.EDGES:
            s, t = e["source"], e["target"]
            if s not in self._idx or t not in self._idx:
                continue
            src.append(self._idx[s])
            tgt.append(self._idx[t])
            w.append(float(e.get("weight", 1.0)))
            dly.append(int(e.get("delay", default_delay)) or default_delay)
            sgn.append(e["sign"])
        self._src = np.asarray(src, dtype=np.intp)
        self._tgt = np.asarray(tgt, dtype=np.intp)
        self._w = np.asarray(w, dtype=np.float64)
        self._delay = np.maximum(np.asarray(dly, dtype=np.intp), 1)
        sgn = np.asarray(sgn)
        self._exc = sgn == _SIGN_EXC
        self._inh = sgn == _SIGN_INH
        self._nmod = np.isin(sgn, _NEUROMOD)
        self._maxdelay = int(self._delay.max()) if self._delay.size else 1

        self.reset_state()

    def reset_state(self):
        """Silence every pool and clear the conduction (delay) buffers."""
        self.v = np.zeros((self.n, self.K), dtype=np.float64)
        # ring buffer of per-node population rates, one row per past tick up to the longest delay
        self._hist = np.zeros((self._maxdelay + 1, self.n), dtype=np.float64)
        self._t = 0
        self._rate = np.zeros(self.n, dtype=np.float64)

    def _node_current(self, drive):
        """Signed, delayed synaptic input + external drive -> per-node input current."""
        exc_in = np.zeros(self.n, dtype=np.float64)
        inh_in = np.zeros(self.n, dtype=np.float64)
        nmod_in = np.zeros(self.n, dtype=np.float64)
        if self._src.size:
            L = self._maxdelay + 1
            rows = (self._t - self._delay) % L  # each edge reads its source `delay` ticks ago
            pre = self._hist[rows, self._src] * self._w  # delayed, weighted presynaptic rate
            np.add.at(exc_in, self._tgt[self._exc], pre[self._exc])
            np.add.at(inh_in, self._tgt[self._inh], pre[self._inh])
            np.add.at(nmod_in, self._tgt[self._nmod], pre[self._nmod])
        gain = 1.0 + self.nmod_gain * nmod_in  # neuromodulation scales excitatory gain
        return gain * (self.drive_scale * drive + self.syn_scale * exc_in) - self.syn_scale * inh_in

    def step(self, region_drive=None):
        """Advance one tick. `region_drive` maps region->drive in [0,1]. Returns {region: rate}."""
        drive = np.zeros(self.n, dtype=np.float64)
        if region_drive:
            for r, d in region_drive.items():
                i = self._idx.get(r)
                if i is not None:
                    drive[i] = float(d)

        current = self._node_current(drive)

        # LIF pool update (vectorized over n nodes x K neurons): leak toward rest + integrate,
        # spike and reset the neurons that cross their (heterogeneous) threshold.
        self.v = self.v * self.decay + current[:, None]
        spikes = self.v >= self.thr
        self.v = np.where(spikes, 0.0, self.v)
        rate = spikes.mean(axis=1)  # fraction of the pool that fired this tick

        L = self._maxdelay + 1
        self._hist[self._t % L] = rate  # publish this tick's rates for delayed downstream reads
        self._t += 1
        self._rate = rate
        return {r: float(rate[i]) for i, r in enumerate(self.regions)}

    def reverberation(self, seed_region, ticks, drive_ticks=None):
        """Mean network activity that PERSISTS after drive to `seed_region` stops.

        Drive the seed briefly, remove the drive, then average the whole network's firing rate over
        `ticks` free-running ticks. A positive (signed excitatory) loop re-enters and sustains, so
        this stays well above baseline; a feed-forward node or pure sink decays back toward zero.
        Side-effect free: the router's live state is snapshotted and restored.
        """
        snap = (self.v.copy(), self._hist.copy(), self._t, self._rate.copy())
        try:
            self.reset_state()
            if drive_ticks is None:
                drive_ticks = max(2, ticks // 3)
            for _ in range(drive_ticks):
                self.step({seed_region: 1.0})
            total = 0.0
            for _ in range(ticks):
                out = self.step(None)  # drive removed -- free-running
                total += sum(out.values())
            return total / (ticks * self.n)
        finally:
            self.v, self._hist, self._t, self._rate = snap


def main():
    # ---------- (a) drive ONE region -> its connectome-downstream targets fire; isolated stay quiet
    r = ConnectomeRouter(seed=0)
    out = {}
    for _ in range(6):
        out = r.step({"ec": 1.0})  # entorhinal cortex -> trisynaptic loop head
    # ec -> dg (perforant path) -> ca3 (mossy fibres): both downstream fire within a few ticks
    downstream_fire = out["dg"] > 0.2 and out["ca3"] > 0.2
    # the olfactory pair {ob, pir} has NO in-edge from the rest of the graph -> stays silent
    isolated_quiet = out["ob"] < 0.02 and out["pir"] < 0.02
    assert downstream_fire, (
        f"driving ec must fire its downstream dg/ca3 (got {out['dg']:.2f}/{out['ca3']:.2f})"
    )
    assert isolated_quiet, (
        f"connectome-unconnected ob/pir must stay quiet (got {out['ob']:.2f}/{out['pir']:.2f})"
    )

    # ---------- (b) an inhibitory edge REDUCES the target vs the excitatory-only condition
    def mean_target_rate(router, drive, target, ticks=14, tail=6):
        router.reset_state()
        rates = [router.step(drive)[target] for _ in range(ticks)]
        return sum(rates[-tail:]) / tail

    a = ConnectomeRouter(seed=1)
    exc_only = mean_target_rate(a, {"cb_deep": 1.0}, "thal_relay")  # dentatothalamic (exc)
    b = ConnectomeRouter(seed=1)
    # add the pallidothalamic gate gpi_snr -> thal_relay (inh); gpi_snr's ONLY out-edge is this
    # inhibitory projection, so it is a clean single-variable manipulation of the same target.
    exc_plus_inh = mean_target_rate(b, {"cb_deep": 1.0, "gpi_snr": 1.0}, "thal_relay")
    assert exc_plus_inh < exc_only, (
        f"inhibitory edge must reduce the target (exc={exc_only:.3f} inh+exc={exc_plus_inh:.3f})"
    )

    # ---------- (c) a positive loop reverberates (> baseline after drive removed); no-loop decays
    rev_loop = ConnectomeRouter(seed=2).reverberation(
        "ec", 40
    )  # hippocampal loop: ec->dg->ca3->ca1->sub->ec
    rev_none = ConnectomeRouter(seed=2).reverberation("actuator", 40)  # pure sink, no out-edge
    baseline = 0.01
    assert rev_loop > baseline, (
        f"positive loop must persist above baseline (rev_loop={rev_loop:.4f})"
    )
    assert rev_loop > 3.0 * rev_none, (
        f"loop must reverberate far more than the no-loop sink (loop={rev_loop:.4f} sink={rev_none:.4f})"
    )

    # ---------- (d) determinism: same seed -> identical dynamics
    d1 = ConnectomeRouter(seed=7)
    d2 = ConnectomeRouter(seed=7)
    seq = [{"ctx_sensory": 0.8}, {"ctx_motor": 0.5}, None, {"vta_snc": 1.0}, None]
    same_step = all(d1.step(s) == d2.step(s) for s in seq)
    same_rev = ConnectomeRouter(seed=3).reverberation("ec", 25) == ConnectomeRouter(
        seed=3
    ).reverberation("ec", 25)
    assert same_step and same_rev, "same seed must give identical outputs"

    n_edges = int(r._src.size)
    print("=" * 70)
    print("CONNECTOME ROUTER -- signed, delayed spike propagation over the wiring schematic")
    print("=" * 70)
    print(f"pools={r.n} (one LIF pool per connectome node)  projections={n_edges}")
    print(
        f"(a) drive ec: dg={out['dg']:.2f} ca3={out['ca3']:.2f} fire; ob={out['ob']:.2f} pir={out['pir']:.2f} quiet"
    )
    print(
        f"(b) thal_relay rate  exc-only={exc_only:.3f}  +inh(gpi_snr)={exc_plus_inh:.3f} (inhibition reduces)"
    )
    print(f"(c) reverberation    loop(ec)={rev_loop:.4f}  no-loop(actuator)={rev_none:.4f}")
    print("(d) determinism      same seed -> identical step + reverberation")
    print("=" * 70)
    print(
        f"self-check OK: driving ec propagates to dg/ca3 (ob/pir isolated); inhibitory edge reduces "
        f"thal_relay {exc_only:.3f}->{exc_plus_inh:.3f}; loop reverberates {rev_loop:.4f} vs no-loop "
        f"{rev_none:.4f}; deterministic; {n_edges} projections over {r.n} pools"
    )


if __name__ == "__main__":
    main()
