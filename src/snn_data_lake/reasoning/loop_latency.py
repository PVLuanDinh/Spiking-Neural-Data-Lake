"""
End-to-end loop latency — quantify the real-time claim (sensor → command).

The loop advertises real-time control (signal_loop's `--window`, `STRIDE < W` "for lower live
latency") but ships NO latency numbers (roadmap D2). For a control loop the sensor→command latency —
and its TAIL — matters as much as accuracy. This instruments the recognition path with
time.perf_counter per stage (encode → in-lake match) over many windows and reports p50/p95/p99. It is
pure instrumentation: no model changes, reuses the existing template matcher. Refs: real-time
neuromorphic control ~20 ms (arXiv:2503.09636; iCub closed-loop ~20–23 ms).

  python loop_latency.py        # (also run by run_selfchecks.py)
"""

import time

from snn_data_lake import signal_loop
from snn_data_lake.spike_preprocessing import encode_latency, van_rossum_distance

N_WINDOWS = 300
BUDGET_MS = 50.0  # generous ceiling — timing is machine-dependent; the check is STRUCTURAL


def percentile(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p / 100.0 * len(sorted_vals)))]


def main():
    lib = signal_loop.build_default_library()
    refs = [encode_latency(v) for v in lib.values()]  # enrolled references (encoded once)
    src = signal_loop.simulated_stream(lib, N_WINDOWS, seed=0)

    enc_ms, match_ms, total_ms = [], [], []
    for _, window in src:
        t0 = time.perf_counter()
        q = encode_latency(window)  # stage 1: spike encoding
        t1 = time.perf_counter()
        _ = min(van_rossum_distance(q, r) for r in refs)  # stage 2: in-lake nearest match
        t2 = time.perf_counter()
        enc_ms.append((t1 - t0) * 1e3)
        match_ms.append((t2 - t1) * 1e3)
        total_ms.append((t2 - t0) * 1e3)

    tot = sorted(total_ms)
    p50, p95, p99 = percentile(tot, 50), percentile(tot, 95), percentile(tot, 99)
    enc_mean = sum(enc_ms) / len(enc_ms)
    match_mean = sum(match_ms) / len(match_ms)

    print("=" * 58)
    print("END-TO-END LOOP LATENCY (sensor -> command, template matcher)")
    print("=" * 58)
    print(f"windows={N_WINDOWS}  library={len(lib)} signatures")
    print(f"  stage means  : encode {enc_mean:.3f} ms | match {match_mean:.3f} ms")
    print(f"  sensor->cmd  : p50 {p50:.3f} ms | p95 {p95:.3f} ms | p99 {p99:.3f} ms")
    print("  (a regression check can flag if p50 crosses a per-rig budget)")
    print("=" * 58)

    # ---- self-check (timing is non-deterministic -> assert STRUCTURE, not a flaky number) ----
    assert p50 <= p95 <= p99, "percentiles must be monotone"
    assert p50 > 0.0, "latency must be positive"
    assert p50 < BUDGET_MS, (
        f"median sensor->command latency {p50:.3f} ms over budget {BUDGET_MS} ms"
    )
    assert abs((enc_mean + match_mean) - (sum(total_ms) / len(total_ms))) < 1e-6 + 0.5, (
        "stage times should sum to ~total"
    )
    print(f"self-check OK: p50/p95/p99 = {p50:.2f}/{p95:.2f}/{p99:.2f} ms, monotone, under budget")


if __name__ == "__main__":
    main()
