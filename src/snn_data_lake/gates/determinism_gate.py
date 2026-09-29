"""Determinism gate (specs/022 T015, FR-005, gates.md 'Determinism gate' row): byte-identical
spike trains across T in {1..16} at fixed P, V_c == 0 and V_ermv == 0, fractional bits MEASURED
against an FP64 reference (research.md D6), and the float-accumulation broken-control arm in
the SAME command.

WHAT THE TWO ZEROS MEAN. V_c is the variance across thread counts of the total spike count;
V_ermv is the worst end-run membrane divergence across thread counts (max over T of
max|v_final(T) - v_final(1)|). The train bytes subsume V_c, and V_ermv adds the STATE half:
two runs whose trains agree while their membranes drift apart are diverging with a delay, and
only the state comparison sees it before it spikes.

THE SWEEP RUNS AT A CONSTRUCTED FIXED P=8, NOT THE LINEAGE P -- deliberately. The lineage's
measured P is 1 (tools/kernel_partition_baseline.json: every parallel cell lost to P1/T1 at
the bench width), and at P=1 thread-invariance has NO REACHABLE RED: one partition means one
task regardless of the pool size, so the criterion would pass a kernel whose multi-partition
delivery races. P=8 is the gate's own constructed exercise rig, recorded here, and it makes
the red reachable -- the broken arm proves it fires.

THE BROKEN-CONTROL ARM (same command, gates.md): float32 accumulation over a constructed
magnitude-spread bundle set ((1e8 + 1.0) - 1e8 rounds the 1.0 away in one order and keeps it
in another), driven through THE SAME train comparator -- forward and reordered float32 arms
MUST produce different trains (the red fires), and the SAME values through int64 fixed-point
MUST agree (the mechanism, not luck, removes the order dependence). A gate whose red was
never constructed is a gate nobody has seen work.

FRACTIONAL BITS ARE MEASURED, NEVER PICKED (D6): --measure sweeps frac_bits and records the
smallest value whose trains equal the FP64-reference trains (raw unquantized weights, fixed
order, T=1) over the disclosed bench, into tools/determinism_baseline.json -- WITH the
coarse-bits red beside it (at very few bits the trains MUST diverge from the reference,
proving the sweep can fail). `binding_frac_bits()` is the read path for callers; the bench
placeholder inside kernel_tick defers to it once recorded.
"""

# distinct-from: reservoir/kernel_tick.py -- the MECHANISM this gate exercises; the gate owns
#   the criterion, the kernel owns the tick. distinct-from: gates/eprop_transport_gate.py
#   and gates/gradient_claim_gate.py -- learning-rule transport criteria, not tick
#   determinism. distinct-from: gates/replication_guard.py -- class-uniqueness census.
# All sites read.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "tools" / "determinism_baseline.json"

#: the gate's constructed exercise rig (see module docstring for why NOT the lineage P)
GATE_P = 8
THREAD_SWEEP = tuple(range(1, 17))


def binding_frac_bits(default=None):
    """The measured binding frac_bits, or `default` when no measurement is recorded yet."""
    if BASELINE.exists():
        return int(json.loads(BASELINE.read_text())["frac_bits"])
    return default


def _bench_net(seed=11, n=96, edges=768):
    """The disclosed bench: n LIF units, `edges` random synapses, delays 1..5, 8 units driven."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return {
        "n": n,
        "pre": rng.integers(0, n, edges),
        "post": rng.integers(0, n, edges),
        "weight": rng.normal(0.35, 0.25, edges),
        "delay": rng.integers(1, 6, edges),
        "drive_ix": rng.integers(0, n, 8),
    }


def _run_kernel(net, threads, frac_bits, ticks=40):
    """One bench run -> (train_bytes, final_v copy, total_spikes)."""
    import numpy as np

    from snn_data_lake.reservoir.kernel_state import KernelState
    from snn_data_lake.reservoir.kernel_tick import KernelTick, Synapses

    state = KernelState([("bench", net["n"])])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(
        net["pre"], net["post"], net["weight"], net["delay"], net["n"], GATE_P, frac_bits
    )
    kt = KernelTick(state, syn, decay=0.8, v_reset=0.0, refrac_ticks=2, threads=threads)
    ext = np.zeros(net["n"], dtype=np.int64)
    ext[net["drive_ix"]] = int(1.5 * (1 << frac_bits))
    total, blobs = 0, []
    for t in range(ticks):
        fired = kt.tick(ext if t % 7 == 0 else None)  # periodic re-drive keeps activity alive
        total += len(fired)
        blobs.append(fired.tobytes())
    return b"".join(blobs), state.arrays["v"].copy(), total


def _run_fp64_reference(net, ticks=40):
    """The FP64 reference: same dynamics, RAW unquantized weights, float64 everywhere, fixed
    order, T=1. What the frac_bits measurement is judged against."""
    import numpy as np

    n = net["n"]
    v = np.zeros(n, dtype=np.float64)
    thr = np.ones(n)
    refrac = np.zeros(n, dtype=np.int64)
    max_d = int(net["delay"].max())
    rings = [np.zeros(n, dtype=np.float64) for _ in range(max_d + 1)]
    by_pre: dict[int, list[tuple[int, float, int]]] = {}
    for p, q, w, d in zip(net["pre"], net["post"], net["weight"], net["delay"]):
        by_pre.setdefault(int(p), []).append((int(q), float(w), int(d)))
    drive = np.zeros(n, dtype=np.float64)
    drive[net["drive_ix"]] = 1.5
    blobs = []
    for t in range(ticks):
        inj = rings[t % (max_d + 1)].copy()
        rings[t % (max_d + 1)][:] = 0.0
        if t % 7 == 0:
            inj = inj + drive
        v = v * 0.8 + inj
        gated = refrac > 0
        refrac = np.maximum(refrac - 1, 0)
        fired = np.nonzero((v >= thr) & ~gated)[0].astype(np.int64)
        v[fired] = 0.0
        refrac[fired] = 2
        for u in fired.tolist():
            for q, w, d in by_pre.get(u, ()):
                rings[(t + d) % (max_d + 1)][q] += w
        blobs.append(fired.tobytes())
    return b"".join(blobs)


def thread_sweep(frac_bits, threads=THREAD_SWEEP):
    """-> (ok, V_c, V_ermv, detail). Byte-equal trains + zero count variance + zero end-run
    membrane divergence across the sweep, all against the T=1 arm."""
    import numpy as np

    net = _bench_net()
    base_train, base_v, base_count = _run_kernel(net, 1, frac_bits)
    counts, worst_v, all_equal = [base_count], 0.0, True
    for t_count in threads[1:]:
        train, v_final, count = _run_kernel(net, t_count, frac_bits)
        counts.append(count)
        worst_v = max(worst_v, float(np.max(np.abs(v_final - base_v))))
        all_equal = all_equal and (train == base_train)
    v_c = float(np.var(np.asarray(counts, dtype=np.float64)))
    ok = all_equal and v_c == 0.0 and worst_v == 0.0
    return ok, v_c, worst_v, f"T={threads[0]}..{threads[-1]} at P={GATE_P}, counts={counts[0]}"


def broken_control():
    """The float-accumulation red, through the same comparator (gates.md's arm). Returns
    (float_diverged, int_agreed) -- the gate is trusted only when BOTH are True."""
    import numpy as np

    # constructed magnitude spread: (1e8 + 1.0) - 1e8 == 0.0 in f32 one way, 1.0 the other
    contributions = np.array([1e8, 1.0, -1e8], dtype=np.float64)
    orders = ([0, 1, 2], [0, 2, 1])

    def trains(mode):
        out = []
        for order in orders:
            if mode == "f32":
                acc = np.float32(0.0)
                for i in order:
                    acc = np.float32(acc + np.float32(contributions[i]))
                total = float(acc)
            else:
                frac = 16
                acc = 0
                for i in order:
                    acc += int(round(contributions[i] * (1 << frac)))
                total = acc / (1 << frac)
            out.append(b"\x01" if total >= 0.5 else b"\x00")  # threshold -> a 1-unit "train"
        return out

    f32 = trains("f32")
    i64 = trains("int")
    return f32[0] != f32[1], i64[0] == i64[1]


def measure_frac_bits(candidates=(4, 6, 8, 10, 12, 14, 16, 20, 24)):
    """-> (bits, sweep, red_bits). Smallest candidate whose trains equal the FP64 reference,
    the full sweep, and a coarse-bits value PROVEN to diverge (the measurement's red)."""
    net = _bench_net()
    ref = _run_fp64_reference(net)
    sweep = {}
    for bits in candidates:
        train, _v, _c = _run_kernel(net, 1, bits)
        sweep[bits] = train == ref
    passing = [b for b, ok in sweep.items() if ok]
    if not passing:
        raise AssertionError(f"no candidate matched the FP64 reference: {sweep}")
    red_bits = None
    for bits in (1, 2, 3):
        train, _v, _c = _run_kernel(net, 1, bits)
        if train != ref:
            red_bits = bits
            break
    if red_bits is None:
        raise AssertionError(
            "even 1..3 fractional bits matched the FP64 reference -- the measurement cannot "
            "fail on this bench, so its answer means nothing (vacuous sweep)"
        )
    return min(passing), sweep, red_bits


def main(argv=None) -> int:
    import os

    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""

    if mode == "--measure":
        bits, sweep, red_bits = measure_frac_bits()
        record = {
            "frac_bits": bits,
            "rule": "smallest candidate whose 40-tick trains equal the FP64 reference (raw weights, T=1)",
            "bench": "96 LIF units x 768 random edges, delays 1..5, 8 driven, seed 11, P=8",
            "sweep_train_equals_fp64": {str(k): v for k, v in sweep.items()},
            "red_bits_diverged": red_bits,
            "host": os.environ.get("COMPUTERNAME", "?"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        BASELINE.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0

    if mode in ("--assert", "--self-check"):
        f32_red, i64_green = broken_control()
        if not f32_red or not i64_green:
            print(
                f"determinism gate FAILED its own control: float-diverged={f32_red}, "
                f"int-agreed={i64_green} -- the comparator or the mechanism is not doing "
                "what the criterion claims"
            )
            return 1
        bits = binding_frac_bits()
        if bits is None:
            print("determinism gate: no frac_bits baseline; run --measure on the pinned host")
            return 1
        ok, v_c, v_ermv, detail = thread_sweep(bits)
        print(
            f"determinism gate [{mode[2:]}] trains {'BYTE-EQUAL' if ok else 'DIVERGED'} "
            f"({detail}); V_c={v_c} V_ermv={v_ermv}; broken control: f32 order arms diverged, "
            f"int64 arms agreed; frac_bits={bits} (measured)"
        )
        return 0 if ok else 1

    # Bare invocation (run_selfchecks): the cheap wiring check -- both control arms + one
    # tiny thread pair, never the full sweep.
    f32_red, i64_green = broken_control()
    if not f32_red or not i64_green:
        print("determinism gate wiring-check FAILED: the constructed control arms misbehaved")
        return 1
    bits = binding_frac_bits(default=16)
    ok, v_c, v_ermv, _detail = thread_sweep(bits, threads=(1, 4))
    if not ok:
        print(
            f"determinism gate wiring-check FAILED: T=1 vs T=4 diverged (V_c={v_c}, V_ermv={v_ermv})"
        )
        return 1
    state = "measured" if BASELINE.exists() else "NOT MEASURED (run --measure on the pinned host)"
    print(
        "determinism gate wiring-check OK: f32 order arms diverged / int64 agreed; T=1==T=4 "
        f"at P={GATE_P}; V_c=0 V_ermv=0; frac_bits baseline {state}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
