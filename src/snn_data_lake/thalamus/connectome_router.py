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
import threading
import time

import numpy as np

from snn_data_lake import connectome

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "ConnectomeRouter._lif_update": (
        "NOT_A_RULE",
        "the LIF pool update: leak toward rest, integrate, spike and reset. Membrane dynamics for one "
        "tick; no synapse is modified and nothing is learned",
    ),
}

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
        # running-EMA cost of one neuron's LIF update, seconds -- the P1 baseline the parallel gate
        # weighs against this machine's measured thread floor. 0.0 means "not measured yet", and the
        # gate treats that as SERIAL rather than guessing.
        self._lif_per_elem_s = 0.0
        self._lif_par_eff = 1.0  # running-EMA fraction of IDEAL parallel savings actually achieved

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

    def _lif_update(self, current):
        """The LIF pool update: leak toward rest + integrate, then spike and reset the neurons that
        cross their (heterogeneous) threshold. Returns the new membrane array and the per-node rate.

        ROW-PARTITIONED ABOVE A MEASURED CROSSOVER. Every node's pool is independent here -- row i
        reads only `v[i]`, `current[i]`, `thr[i]` and writes only `v[i]` and `rate[i]`. There is no
        cross-row reduction at all (`mean(axis=1)` reduces WITHIN a row), so threads own disjoint
        contiguous row blocks and the result is elementwise identical to the whole-array form; there
        is no float-addition order to change. The one genuinely cross-node step -- the delayed,
        signed synaptic scatter-add in `_node_current` -- stays serial, and it is not the cost: at
        width 960,000 it measured 12 us against this function's 5,577 us.

        The crossover is measured, not written down: `region_wiring.pick_thread_count` weighs this
        instance's running-EMA cost per neuron against this machine's measured thread floor and the
        efficiency this kernel has actually been achieving. Measured end-to-end through `step()` on a
        28-core free-threaded 3.14.6 box, this is a MODEST win that arrives LATE -- width 960 stays
        serial, 480,000 1.27x, 960,000 1.6-2.0x, 4,800,000 2.5-2.8x, 10,000,000 2.5-2.9x. It
        saturates near 2.5-2.9x because the kernel is memory-bandwidth bound, not compute bound, and
        the back-solved efficiency sits around 0.5-0.75.

        Width 96,000 is the honest edge case and this row used to read "0.99x (gate: serial)". That
        is true only of a WARM gate: re-measured cold, the gate picks 2-4 threads there and runs
        0.62-0.66x -- a 35-40% REGRESSION -- and it is `realized_efficiency` feeding that result
        back that returns it to serial afterwards. The cost is bounded (one mispriced call per fresh
        instance) and self-correcting, but stating it as "serial" hid a loss that is real,
        reproducible, and sits only ~100x above the live width rather than safely beyond the
        crossover.

        The live brain builds this router at pool_size=24 over 40 nodes = width 960, which is ~500x
        below the crossover, so in production today the gate says serial and that is the correct
        answer. The path exists because the gate is what makes outgrowing it safe -- and note the
        width knob here is `pool_size`, which nothing in the live tree currently raises, so this is
        the weakest-standing of the three parallel paths on present evidence."""
        n = self.v.shape[0]
        nt = 1
        est = self.v.size * self._lif_per_elem_s
        if n >= 2 and self._lif_per_elem_s > 0.0:
            from snn_data_lake.substrate import region_wiring as _rw

            nt = min(_rw.pick_thread_count(est, mode="threads", efficiency=self._lif_par_eff), n)
        if nt < 2:
            t0 = time.perf_counter()
            v = self.v * self.decay + current[:, None]
            spikes = v >= self.thr
            v = np.where(spikes, 0.0, v)
            rate = spikes.mean(axis=1)  # fraction of the pool that fired this tick
            dt = (time.perf_counter() - t0) / max(1, self.v.size)
            self._lif_per_elem_s = (
                dt
                if self._lif_per_elem_s <= 0.0
                else 0.75 * self._lif_per_elem_s + 0.25 * dt  # running baseline (P1)
            )
            return v, rate

        from snn_data_lake.substrate import region_wiring as _rw

        t0 = time.perf_counter()
        out_v = np.empty_like(self.v)
        rate = np.empty(n, dtype=np.float64)

        bounds = _rw.blocks(n, nt)
        errors = [None] * len(bounds)

        # `out_v` / `rate` are UNINITIALISED memory. A worker that raises before writing its rows
        # dies inside threading, which prints a traceback and swallows the exception -- `join()`
        # then returns normally and this function would hand back whatever those bytes happened to
        # hold. Measured before this guard: killing one worker returned pool rates of [0, 0, 0]
        # where the truth was [0.766, 0.766, 0.0] -- no exception, no NaN, nothing out of range,
        # and the wrong rates go straight into `_hist` and drive every downstream node. Silent
        # plausible garbage is strictly worse than a crash, so every block records its exception
        # and the caller re-raises rather than returning a partially-written buffer.
        def block(i, a, b):  # writes ONLY rows [a, b) of out_v / rate -- disjoint across threads
            try:
                v = self.v[a:b] * self.decay + current[a:b, None]
                spikes = v >= self.thr[a:b]
                np.copyto(out_v[a:b], np.where(spikes, 0.0, v))
                rate[a:b] = spikes.mean(axis=1)
            except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised by the caller
                errors[i] = exc

        threads = [
            threading.Thread(target=block, args=(i, a, b), name=f"cr-lif-{i}")
            for i, (a, b) in enumerate(bounds)
            if i > 0
        ]
        for th in threads:
            th.start()
        block(0, *bounds[0])  # the caller takes the first block rather than idling
        for th in threads:
            th.join()
        for exc in errors:  # ascending block order -> deterministic choice when several fail
            if exc is not None:
                raise exc
        # Feed the ACHIEVED scaling back: this kernel is memory-bandwidth bound, so ideal scaling is
        # never reached and assuming it made the gate thread a 96,000-neuron update for a 0.67x loss.
        eff = _rw.realized_efficiency(est, time.perf_counter() - t0, nt, mode="threads")
        self._lif_par_eff = 0.5 * self._lif_par_eff + 0.5 * eff
        return out_v, rate

    def step(self, region_drive=None):
        """Advance one tick. `region_drive` maps region->drive in [0,1]. Returns {region: rate}."""
        drive = np.zeros(self.n, dtype=np.float64)
        if region_drive:
            for r, d in region_drive.items():
                i = self._idx.get(r)
                if i is not None:
                    drive[i] = float(d)

        current = self._node_current(drive)
        self.v, rate = self._lif_update(current)

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

    # ---------- (e) the ROW-PARTITIONED LIF update is BYTE-IDENTICAL to the whole-array form
    # Forced both ways rather than left to the measured gate, so the comparison happens on every run
    # at a width where the gate itself would (correctly) choose serial.
    from snn_data_lake.substrate import region_wiring as _rw

    # The drive is deliberately WEAK (0.2, not 1.0). At saturating drive every neuron in a pool
    # clears every threshold, the per-neuron threshold spread stops mattering, and a comparison of
    # the two paths passes even if one of them ignores `thr` entirely -- measured, not assumed. The
    # graded regime below is the one where the heterogeneity is load-bearing, and the assert on
    # `n_graded` keeps it that way if the dynamics ever shift.
    graded_drive = {"ctx_sensory": 0.2, "ec": 0.14}
    p = ConnectomeRouter(seed=11, pool_size=64)
    p._lif_per_elem_s = 0.0  # cold EMA -> the gate must refuse to guess and take the serial path
    for _ in range(5):
        p.step(graded_drive)  # settle into the graded regime
    pdrive = np.zeros(p.n, dtype=np.float64)
    for _r, _d in graded_drive.items():
        pdrive[p._idx[_r]] = _d
    cur = p._node_current(pdrive)
    v_ser, rate_ser = p._lif_update(cur)
    n_graded = int(((rate_ser > 0.02) & (rate_ser < 0.98)).sum())
    assert n_graded >= 3, (
        f"the byte-identity comparison must run where per-neuron thresholds MATTER; only "
        f"{n_graded} pools are firing at a graded rate, so the comparison would be vacuous"
    )
    assert p._lif_per_elem_s > 0.0, "a serial LIF update must record its own per-neuron cost (P1)"
    assert _rw.pick_thread_count(0.0, mode="threads") == 1, (
        "with no measured cost the gate must return 1 (serial), never a guess"
    )
    p._lif_per_elem_s = 1.0  # an absurd cost -> the gate must now choose threads
    forced_nt = _rw.pick_thread_count(p.v.size * p._lif_per_elem_s, mode="threads")
    v_par, rate_par = p._lif_update(cur)
    identical = v_par.tobytes() == v_ser.tobytes() and rate_par.tobytes() == rate_ser.tobytes()
    assert forced_nt >= 2, f"a huge measured cost must select >= 2 threads, got {forced_nt}"
    assert identical, (
        "the row-partitioned LIF update must be BYTE-identical to the whole-array form -- rows are "
        "independent and mean(axis=1) reduces within a row, so any difference is a real defect"
    )

    # REACHABLE RED for the worker-failure path. `out_v` / `rate` are UNINITIALISED memory, so a
    # worker that dies before writing its rows leaves them holding whatever bytes were there --
    # and threading swallows the exception, so `join()` returns and the caller gets plausible
    # garbage. MEASURED before the re-raise guard: killing one worker returned pool rates of
    # [0.0, 0.0, 0.0] where the truth was [0.765625, 0.765625, 0.0], with no exception, no NaN and
    # nothing out of range -- wrong rates that flow straight into `_hist` and drive every
    # downstream node. Poison exactly one block's threshold slice, and a WORKER's rather than the
    # caller's, so this tests the thread that cannot raise into the caller by itself.
    class _PoisonThr:
        """Delegates to the real threshold array, but raises for ONE row block."""

        def __init__(self, real, lo):
            self._real, self._lo = real, lo

        def __getitem__(self, k):
            if isinstance(k, slice) and k.start == self._lo:
                raise MemoryError("simulated mid-tick failure in one worker block")
            return self._real[k]

    q = ConnectomeRouter(seed=11, pool_size=64)
    q._lif_per_elem_s = 1.0
    q_nt = min(_rw.pick_thread_count(q.v.size * 1.0, mode="threads"), q.v.shape[0])
    q_bounds = _rw.blocks(q.v.shape[0], q_nt)
    assert len(q_bounds) >= 2, f"this RED needs a real worker thread to kill: {q_bounds}"
    q_real_thr = q.thr
    q.thr = _PoisonThr(q_real_thr, q_bounds[1][0])  # block 1 runs on a worker, never the caller
    try:
        q._lif_update(q._node_current(np.zeros(q.n)))
        raise AssertionError(
            "a worker block that raises must PROPAGATE -- never return a partially-written buffer"
        )
    except MemoryError:
        pass
    finally:
        q.thr = q_real_thr

    # zero / one: a single-node partition and an empty drive must survive the same path
    p1 = ConnectomeRouter(seed=5, pool_size=1)
    p1._lif_per_elem_s = 1.0
    v1, r1 = p1._lif_update(p1._node_current(np.zeros(p1.n)))
    assert v1.shape == p1.v.shape and r1.shape == (p1.n,), (v1.shape, r1.shape)

    # and the whole tick agrees end to end, serial vs forced-parallel, over a real trajectory
    def _trace(force):
        rr = ConnectomeRouter(seed=11, pool_size=64)
        rr._lif_per_elem_s = 1.0 if force else 0.0
        out = []
        for i in range(12):
            out.append(rr.step(graded_drive if i < 8 else None))
            if force:
                rr._lif_per_elem_s = 1.0  # keep the gate pinned to parallel across the trajectory
        return out

    t_ser, t_par = _trace(False), _trace(True)
    assert t_ser == t_par, (
        "12 ticks of step() must agree exactly between the serial and parallel LIF"
    )
    assert any(0.02 < x < 0.98 for tick in t_ser for x in tick.values()), (
        "the 12-tick trajectory comparison must include graded rates, or it too is vacuous"
    )

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
    print(
        f"(e) parallel LIF     row-partitioned on {forced_nt} threads == whole-array, byte-identical "
        f"(v + rate); 12 ticks agree end to end"
    )
    print("=" * 70)
    print(
        f"self-check OK: driving ec propagates to dg/ca3 (ob/pir isolated); inhibitory edge reduces "
        f"thal_relay {exc_only:.3f}->{exc_plus_inh:.3f}; loop reverberates {rev_loop:.4f} vs no-loop "
        f"{rev_none:.4f}; deterministic; {n_edges} projections over {r.n} pools"
    )


if __name__ == "__main__":
    main()
