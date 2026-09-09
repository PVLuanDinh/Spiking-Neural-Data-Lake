"""KernelTick -- the two-phase deterministic tick over KernelState (specs/022 T014, FR-005):
propagate -> deliver -> rotate, fixed partition count P per lineage, integer synaptic
accumulation, bundles keyed (pre, partition, delay), per-partition delay rings.

THE DETERMINISM SHAPE (research.md D6, NEST virtual-process form). Every unit belongs to
exactly one of P partitions (unit_index % P -- static, documented, and grown units join
EXISTING partitions by the same rule). Each partition OWNS its units' delivery: a spike is
routed to the TARGET partition's delay ring between phases, and only the owner drains its own
ring into its own accumulators -- single-writer per accumulator, so threads never contend, and
the accumulator is INTEGER (fixed-point weights at `frac_bits`), so the sum is order-invariant
outright. The membrane update is elementwise float32 (no cross-unit reduction), deterministic
at any thread count by construction. Byte-identical trains across T in {1..16} at fixed P is
T015's gate (`substrate/determinism_gate.py`); this module carries the mechanism.

P-VS-GROWTH RULE (SYNTHESIS L1, verbatim intent): P is FIXED for the life of a lineage.
Grown units are assigned to existing partitions by the same index rule; the resulting
imbalance is MEASURED (`partition_imbalance()`, provenance-tagged) and rises honestly rather
than being rebalanced silently -- renegotiating P is a NEW lineage with a recorded baseline
break. P itself is DERIVED at landing from a measured saturation sweep on the pinned host
(`--derive-p`, recorded to tools/kernel_partition_baseline.json), never picked (P1).

THE COCKPIT ROW IS DEFERRED TO CUTOVER, on purpose: the cockpit renders the LIVE brain's
snapshot through provenance-tagged accessors, and this kernel is not on the live tick yet --
rendering a bench number beside live rows is exactly the dashboard lie PROVENANCE exists to
prevent (the dead-E-STOP incident's sibling). The metric + its provenance string ship HERE;
the row lands when the kernel joins the live snapshot.

EQUATION OWNERSHIP (the T013 P2 disposition, honored): the leaky integrate-and-fire step
below is the VECTORIZED form of `cortex/lif_neuron.py`'s per-instance equation -- that module
remains the cited equation owner and its scalar form retires at cutover with a parity A/B.
No constant is picked here: decay, thresholds (state.thr), reset and frac_bits are the
caller's, and frac_bits' binding value is T015's measurement (D6: fractional bits MEASURED
vs FP64).

THE RESIDENCY HOOK IS DELETED (2026-09-09, T068 LOSS). This tick had one residency branch --
`KernelTick(..., resident=<bool mask>)` gated FIRING and routed a non-resident target's charge into
`pending[p]` -- for the activation-driven paging layer of specs/022 T066. That layer was measured
(evidence/s5-paging-null.md 3.4) at a 35.2x median tick cost against a 3.0x bound fixed before the
run and DELETED on its null; the mask branch went with it, because with the pager gone `resident`
could only ever be `None`. The measurement that made this deletion non-optional rather than tidy-up:
with page-ins and evictions ablated to zero, the mask branch's OWN tick was still 2.5-3.4x the
incumbent's -- it consumed the entire cost budget before a single page-in. A successor paging row
does NOT inherit this branch; it inherits that decomposition. What remains below is the incumbent
path, byte for byte, which is what `resident=None` always ran. `compile_bundles` is untouched and
remains ONE compiler, used by `Synapses.__init__`.
"""

# distinct-from: cortex/cortical_microcircuit.py -- a fixed 8-population Potjans-Diesmann
#   circuit with its own internal stepping; this is the general SoA substrate tick.
# distinct-from: substrate/signal_ring.py -- transport of lake-bound RECORDS between faculties;
#   the delay rings here carry spike BUNDLES inside one kernel.
# distinct-from: cortex/neuron_zoo.py -- per-object model population (see T013 disposition).
# distinct-from: basal_ganglia/r_stdp.py -- its _FanOut/Calibration machinery parallelizes its
#   OWN plasticity math with a measured serial/parallel crossover; a learning-rule execution
#   detail, never a substrate tick or delivery transport.
# distinct-from: cortex/access_consciousness.py -- workspace BROADCAST delivery (_head_only is
#   that contract's red); content routing between faculties, not spikes between units.
# distinct-from: substrate/lake_migrate.py, cortex/neurogenesis_engine.py,
#   cortex/counterfactual_sim.py -- matched on shared vocabulary (dispositions, crc identity);
#   none steps a substrate. All sites read.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PARTITION_BASELINE = REPO / "tools" / "kernel_partition_baseline.json"

#: the cockpit provenance string for the imbalance metric (rendered at cutover, not before)
IMBALANCE_PROVENANCE = (
    "kernel_tick.partition_imbalance: max/mean unit count over the lineage's fixed P "
    "(unit_index % P; grown units join existing partitions -- imbalance rises honestly, "
    "never silently rebalanced)"
)


# distinct-from: cortex/percept_surprise.py::declare_ports, neuromodulation/raphe_serotonin.py::
#   declare_ports -- manifest REGISTRANTS: they declare typed faculty ports on the wiring manifest
#   and compile nothing; no weight, no delay, no ring key.
# distinct-from: substrate/basic_components.py::Connections.update -- an in-place bounded weight
#   CHANGE (add dw, mask, clip) on a dense matrix; this quantizes a static edge list ONCE into
#   routing bundles and never moves a weight.
# distinct-from: substrate/brain_bundle.py::BrainBundle.predict -- "bundle" there is a named
#   collection of trained brains being routed to; here a bundle is (pre, partition, delay) ->
#   posts/qweights inside one kernel.
# distinct-from: hippocampus/ng_snn.py::decide -- the ONE argmax decision rule over an NG-SNN rate
#   matrix; a readout, not a delivery table. All five sites read 2026-09-08.
def compile_bundles(pre, post, weight, delay, n_units, p_parts, frac_bits, max_delay=None):
    """THE ONE BUNDLE COMPILER (specs/022 T066, P2). Factored out of `Synapses.__init__` so the
    constructor and the pager's per-block page-in share one compiler rather than two that drift:
    a block compiled on page-in must produce bundles byte-identical to the ones the whole-table
    constructor would have produced, or a paged run could not be byte-identical to a resident
    twin. Returns `(bundles, by_pre)`; the caller owns where they land.

    Every refusal lives here, so it binds the per-block compile too: parallel arrays, an endpoint
    outside [0, n_units), a delay below 1 (it would deliver inside its own propagate phase), and a
    delay above `max_delay` (which would collide in a ring slot before its drain). `max_delay=None`
    derives it from the edges, which is the constructor's default; the pager passes the table's
    own value so a block with short delays cannot silently shrink the rings.
    """
    import numpy as np

    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    delay = np.asarray(delay, dtype=np.int64)
    if not (len(pre) == len(post) == len(weight) == len(delay)):
        raise ValueError("pre/post/weight/delay must be parallel arrays")
    if len(pre) and (
        pre.min() < 0 or pre.max() >= n_units or post.min() < 0 or post.max() >= n_units
    ):
        raise ValueError("synapse endpoint out of range")
    if len(delay) and delay.min() < 1:
        raise ValueError(
            "delays are in ticks and start at 1 (a 0-delay edge would deliver inside "
            "its own propagate phase)"
        )
    md = int(max_delay if max_delay is not None else (delay.max() if len(delay) else 1))
    if len(delay) and delay.max() > md:
        raise ValueError("a delay exceeds max_delay")
    q = np.rint(weight * (1 << int(frac_bits))).astype(np.int64)
    part = post % int(p_parts)
    #: (pre, partition, delay) -> (post_idx array, qweight array)
    bundles: dict[tuple[int, int, int], tuple] = {}
    order = np.lexsort((post, delay, part, pre))
    keys = np.stack([pre[order], part[order], delay[order]], axis=1)
    if len(order):
        edges = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
        starts = [0] + edges.tolist() + [len(order)]
        for a, b in zip(starts[:-1], starts[1:]):
            k = (int(keys[a, 0]), int(keys[a, 1]), int(keys[a, 2]))
            sel = order[a:b]
            bundles[k] = (post[sel].copy(), q[sel].copy())
    #: pre -> [(partition, delay, key)] for O(out-bundles) routing on a spike
    by_pre: dict[int, list[tuple[int, int, tuple]]] = {}
    for k in bundles:
        by_pre.setdefault(k[0], []).append((k[1], k[2], k))
    return bundles, by_pre


class Synapses:
    """Static synapse table compiled into delivery bundles keyed (pre, target partition, delay).

    A bundle is the unit of routing: when `pre` fires, one (partition, delay) bundle reference
    lands in the target partition's ring -- delivery then walks the bundle's posts/weights with
    integer adds. Weights are quantized ONCE at `frac_bits` (fixed-point int64), so every later
    accumulation is exact integer arithmetic. The compile itself is `compile_bundles` above --
    ONE compiler, this constructor and the pager's page-in being its two callers."""

    def __init__(self, pre, post, weight, delay, n_units, p_parts, frac_bits, max_delay=None):
        import numpy as np

        d = np.asarray(delay, dtype=np.int64)
        self.n_units, self.p_parts, self.frac_bits = int(n_units), int(p_parts), int(frac_bits)
        self.max_delay = int(max_delay if max_delay is not None else (d.max() if len(d) else 1))
        self.bundles, self.by_pre = compile_bundles(
            pre, post, weight, delay, self.n_units, self.p_parts, self.frac_bits, self.max_delay
        )

    @classmethod
    def empty(cls, n_units, p_parts, frac_bits, max_delay):
        """The empty (0-edge) table. DATED NOTE 2026-09-09 (specs/022 T068): this line used to
        read "the empty table a pager fills block by block (specs/022 T066)"; the paging layer was
        DELETED ON ITS NULL, so there is no pager and no block-by-block fill. What survives is the
        constructor a later INCREMENTAL COMPILE would start from. It has ZERO call sites in the
        tree today -- measured, not assumed, by git grep for the token `.empty(` over the tracked
        *.py set, where every remaining hit is a numpy or torch `.empty`; tests/test_kernel_tick.py
        builds `Synapses(...)` directly (its `_build` helper, and the refusal cases) and never
        reaches this classmethod. `max_delay` is REQUIRED here and has no default: the rings are
        sized from it at KernelTick construction, before a single edge exists, so deriving it from
        the (absent) edges would size the rings to one slot and refuse every later block whose
        delays are longer."""
        return cls([], [], [], [], n_units, p_parts, frac_bits, max_delay)


class KernelTick:
    """propagate -> deliver -> rotate over a KernelState. Single mechanism, any thread count."""

    def __init__(self, state, synapses, decay, v_reset, refrac_ticks, threads=1):
        import numpy as np

        if not (0.0 < float(decay) <= 1.0):
            raise ValueError("decay is the caller's per-tick membrane retention in (0, 1]")
        self.state, self.syn = state, synapses
        self.decay, self.v_reset = float(decay), float(v_reset)
        self.refrac_ticks = int(refrac_ticks)
        self.threads = int(threads)
        self.p_parts = synapses.p_parts
        self.tick_index = 0
        n = state.n_units
        if n != synapses.n_units:
            raise ValueError("state and synapse table disagree on unit count")
        idx = np.arange(n, dtype=np.int64)
        #: static partition membership (the lineage's layout; grown units join by the same rule)
        self._members = [idx[idx % self.p_parts == p] for p in range(self.p_parts)]
        #: per-partition delay rings: ring[p][slot] = list of bundle keys due that slot
        self._rings = [[[] for _ in range(synapses.max_delay + 1)] for _ in range(self.p_parts)]
        #: per-partition INTEGER accumulators over the full index space (owner-written only)
        self._acc = [np.zeros(n, dtype=np.int64) for _ in range(self.p_parts)]
        self._np = np

    # ---- the three phases ------------------------------------------------------------------
    def _propagate(self, p, ext_q):
        """Integrate partition p's units (elementwise, deterministic at any T); return fired.

        THE RESIDENCY MASK IS GONE (2026-09-09, T068 LOSS -- see the module header). What stands
        here is the incumbent arithmetic the `resident is None` path always ran: every member's
        membrane decays by one shared elementwise op, external drive lands in the membrane, and a
        member fires when it crosses threshold outside refraction. The LIF step is not duplicated
        into branches -- this module's header names it as the vectorized form it owns, and a second
        copy inside a residency branch was the drift P2 refuses. The determinism and replay gates
        re-run this path."""
        np = self._np
        m = self._members[p]
        scale = float(1 << self.syn.frac_bits)
        v = self.state.arrays["v"]
        thr = self.state.arrays["thr"]
        refrac = self.state.arrays["refrac"]
        last = self.state.arrays["last_spike"]
        inj = self._acc[p][m]
        if ext_q is not None:
            inj = inj + ext_q[m]
        v[m] = (v[m] * self.decay + (inj / scale)).astype(np.float32)
        self._acc[p][m] = 0
        gated = refrac[m] > 0
        refrac[m] = np.maximum(refrac[m] - 1, 0)
        can_fire = (v[m] >= thr[m]) & ~gated
        fired = m[can_fire]
        v[fired] = self.v_reset
        refrac[fired] = self.refrac_ticks
        last[fired] = self.tick_index
        return fired

    def _route(self, fired_sorted):
        """Between phases, single-threaded and in sorted order: each spike drops its bundle
        keys into the TARGET partitions' rings at (now + delay). Order here cannot affect the
        result -- delivery sums are integer -- but a deterministic route order keeps the ring
        contents themselves byte-comparable for the replay gate (T017)."""
        slots = self.syn.max_delay + 1
        for u in fired_sorted.tolist():
            for part, delay, key in self.syn.by_pre.get(u, ()):
                self._rings[part][(self.tick_index + delay) % slots].append(key)

    # distinct-from: cortex/time_cells.py -- first_passage_sd is a Wiener first-passage
    #   variance statistic. distinct-from: reasoning/rsnn_delays_torch.py -- verify_transfer
    #   bridges a torch export into the zero-dep core. distinct-from:
    #   reasoning/train_of_thought.py -- summon() ignites a workspace slot. distinct-from:
    #   substrate/cockpit.py -- Acts queues arm HTTP commands for the tick-owning thread.
    #   distinct-from: cortex/brain.py -- _route_percept_to_action is W5 top-down drive.
    #   Vocabulary overlap only; none drains a delay-ring slot. All sites read.
    def _deliver_and_rotate(self, p):
        """Partition p drains ITS OWN slot for the NEXT tick into ITS OWN integer accumulator
        (the single-writer contract), then clears the slot (the rotate). Draining (t+1) rather
        than (t) is what makes a delay of d ticks land at EXACTLY t+d: the accumulator this
        fills is integrated by the next tick's propagate. The first draft drained (t) and every
        delivery arrived one tick late -- caught by the self-check's delay-arrival arm on its
        first run. Ring safety at slots = max_delay+1: an event that could collide in a slot
        before its drain would need delay > max_delay, which Synapses refuses.

        THE RESIDENCY SPLIT IS DELETED (2026-09-09, T068 LOSS -- see the module header). Every
        bundle's posts go into the partition's own accumulator, which is what the unpaged path
        always did; there is no second destination for a delivery any more."""
        np = self._np
        slot = self._rings[p][(self.tick_index + 1) % (self.syn.max_delay + 1)]
        acc = self._acc[p]
        for key in slot:
            posts, q = self.syn.bundles[key]
            np.add.at(acc, posts, q)
        slot.clear()

    def tick(self, ext_q=None):
        """One two-phase tick. `ext_q` is optional external drive, already fixed-point int64
        over the full index space (the caller quantizes with its own frac_bits -- P1).
        Returns the fired unit indices, sorted ascending (the canonical train byte-form is
        `fired.tobytes()`)."""
        np = self._np
        parts = range(self.p_parts)
        if self.threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                fired = np.concatenate(list(ex.map(lambda p: self._propagate(p, ext_q), parts)))
        else:
            fired = np.concatenate([self._propagate(p, ext_q) for p in parts])
        fired.sort()
        self._route(fired)
        if self.threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                list(ex.map(self._deliver_and_rotate, parts))
        else:
            for p in parts:
                self._deliver_and_rotate(p)
        self.tick_index += 1
        return fired

    def partition_imbalance(self):
        """max/mean unit count across partitions (1.0 = perfectly balanced). The number the
        cockpit renders at cutover under IMBALANCE_PROVENANCE."""
        counts = [len(m) for m in self._members]
        mean = sum(counts) / len(counts)
        return (max(counts) / mean) if mean else float("inf")


# ---- the P derivation (run at landing on the pinned host; recorded, never picked) -----------
def _bench_kernel(n_units, fan_out, p_parts, threads, n_ticks, seed=7):
    """ticks/s for one (P, threads) cell on a disclosed synthetic network: `n_units` LIF units,
    `fan_out` random out-edges each, delays uniform in 1..8, drive strong enough that a real
    fraction fires per tick (activity, not silence, is what delivery costs)."""
    import numpy as np

    from snn_data_lake.substrate.determinism_gate import binding_frac_bits
    from snn_data_lake.substrate.kernel_state import KernelState

    nprng = np.random.default_rng(seed)
    pre = nprng.integers(0, n_units, size=n_units * fan_out)
    post = nprng.integers(0, n_units, size=n_units * fan_out)
    weight = nprng.normal(0.05, 0.02, size=n_units * fan_out)
    delay = nprng.integers(1, 9, size=n_units * fan_out)
    frac_bits = binding_frac_bits(default=16)  # T015's MEASURED value once recorded (12)
    state = KernelState([("bench", n_units)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(pre, post, weight, delay, n_units, p_parts, frac_bits)
    kt = KernelTick(state, syn, decay=0.9, v_reset=0.0, refrac_ticks=2, threads=threads)
    ext = np.zeros(n_units, dtype=np.int64)
    drive_ix = nprng.integers(0, n_units, size=max(1, n_units // 10))
    ext[drive_ix] = int(1.2 * (1 << frac_bits))
    t0 = time.perf_counter()
    for _k in range(n_ticks):
        kt.tick(ext)
    return n_ticks / (time.perf_counter() - t0)


def derive_p(n_units=4096, fan_out=40, n_ticks=30):
    """Sweep P and pick the SMALLEST P within 5% of peak throughput across the swept thread
    counts -- the knee, preferring fewer partitions because imbalance under growth scales with
    P. The rule is written here so the derivation is repeatable; the VALUE is recorded to
    tools/kernel_partition_baseline.json with the whole sweep beside it."""
    sweep = {}
    for p_parts in (1, 2, 4, 8, 16, 32):
        for threads in (1, 4, 8):
            key = f"P{p_parts}/T{threads}"
            sweep[key] = round(_bench_kernel(n_units, fan_out, p_parts, threads, n_ticks), 2)
    peak = max(sweep.values())
    eligible = sorted(int(k.split("/")[0][1:]) for k, v in sweep.items() if v >= 0.95 * peak)
    return eligible[0], sweep, peak


def main(argv=None) -> int:
    import os

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--derive-p"]:
        p_parts, sweep, peak = derive_p()
        record = {
            "rule": (
                "smallest P within 5% of peak ticks/s (knee; fewer partitions -> less "
                "growth imbalance)"
            ),
            "bench": (
                "4096 LIF units x 40 random out-edges, delays 1..8, ~10% driven, "
                "30 ticks/cell, seed 7"
            ),
            "host": os.environ.get("COMPUTERNAME", "?"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sweep_ticks_per_s": sweep,
            "peak_ticks_per_s": peak,
            "P": p_parts,
        }
        PARTITION_BASELINE.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0

    # Bare invocation (run_selfchecks): mechanism checks on a tiny constructed net -- delay
    # arrival, thread-count train equality, imbalance metric, refusals.
    import numpy as np

    from snn_data_lake.substrate.kernel_state import KernelState

    def build(threads):
        state = KernelState([("a", 6), ("b", 6)])
        state.arrays["thr"][:] = 1.0
        syn = Synapses(
            pre=[0, 0, 1, 5],
            post=[3, 7, 8, 2],
            weight=[2.0, 2.0, 2.0, 2.0],
            delay=[1, 3, 2, 1],
            n_units=12,
            p_parts=4,
            frac_bits=16,
        )
        return state, KernelTick(
            state, syn, decay=0.5, v_reset=0.0, refrac_ticks=1, threads=threads
        )

    _state, kt = build(1)
    ext = np.zeros(12, dtype=np.int64)
    ext[[0, 1, 5]] = 2 << 16  # drive three pres over threshold on tick 0
    trains = [kt.tick(ext)]
    for _ in range(4):
        trains.append(kt.tick())
    if trains[0].tolist() != [0, 1, 5]:
        print(f"kernel_tick self-check FAILED: driven units did not fire ({trains[0]})")
        return 1
    # delay arrival: 0->3 d=1 and 5->2 d=1 land tick 1; 1->8 d=2 lands tick 2; 0->7 d=3 tick 3
    got = [t.tolist() for t in trains[1:4]]
    if got != [[2, 3], [8], [7]]:
        print(f"kernel_tick self-check FAILED: delay rings misdelivered ({got})")
        return 1
    _state4, kt4 = build(4)
    trains4 = [kt4.tick(ext)] + [kt4.tick() for _ in range(4)]
    if any(a.tobytes() != b.tobytes() for a, b in zip(trains, trains4)):
        print("kernel_tick self-check FAILED: T=1 and T=4 trains diverged at fixed P")
        return 1
    if abs(kt.partition_imbalance() - 1.0) > 1e-9:
        print(
            "kernel_tick self-check FAILED: 12 units over P=4 must balance "
            f"({kt.partition_imbalance()})"
        )
        return 1
    try:
        Synapses([0], [1], [1.0], [0], 2, 1, 16)
        print("kernel_tick self-check FAILED: a 0-tick delay was accepted")
        return 1
    except ValueError:
        pass
    baseline = (
        "recorded"
        if PARTITION_BASELINE.exists()
        else "NOT RECORDED (run --derive-p on the pinned host)"
    )
    print(
        "kernel_tick self-check OK: propagate/deliver/rotate delivered every delay on its "
        "tick; T=1 == T=4 trains byte-equal at fixed P; imbalance metric exact; 0-delay "
        f"refused; partition baseline {baseline}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
