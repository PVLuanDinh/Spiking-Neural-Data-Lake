"""
End-to-end loop latency — quantify the real-time claim (sensor → command).

The loop advertises real-time control (signal_loop's `--window`, `STRIDE < W` "for lower live
latency") but ships NO latency numbers (roadmap D2). For a control loop the sensor→command latency —
and its TAIL — matters as much as accuracy. This instruments the recognition path with
time.perf_counter per stage (encode → in-lake match) over many windows and reports p50/p95/p99. It is
pure instrumentation: no model changes, reuses the existing template matcher. Refs: real-time
neuromorphic control ~20 ms (arXiv:2503.09636; iCub closed-loop ~20–23 ms).

  PYTHONPATH=src python -m snn_data_lake.reasoning.loop_latency   # (also run by run_selfchecks.py)
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

    enc_ms, match_ms, total_ms, best = [], [], [], []
    for _, window in src:
        t0 = time.perf_counter()
        q = encode_latency(window)  # stage 1: spike encoding
        t1 = time.perf_counter()
        d = min(van_rossum_distance(q, r) for r in refs)  # stage 2: in-lake nearest match
        t2 = time.perf_counter()
        enc_ms.append((t1 - t0) * 1e3)
        match_ms.append((t2 - t1) * 1e3)
        total_ms.append((t2 - t0) * 1e3)
        best.append(d)  # the matcher's OWN output -- was discarded into `_` (P3 shape 2)

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
    #
    # TWO CRITERIA WERE DELETED HERE 2026-08-04, not weakened, because neither could go red (P8/F11):
    #   * `p50 <= p95 <= p99` -- `percentile()` indexes ONE sorted list at non-decreasing positions,
    #     so this is monotone by construction of the helper above, for any input whatsoever.
    #   * `abs((enc_mean + match_mean) - mean(total_ms)) < 1e-6 + 0.5` -- per window,
    #     total = (t2-t0) and enc + match = (t1-t0) + (t2-t1), the SAME sum reassociated, so the
    #     difference is float rounding (~1e-13 ms here) against a 0.5 ms tolerance. It restated its
    #     own arithmetic. (The tolerance `1e-6 + 0.5` was also just 0.500001 written oddly.)
    # What replaces them are checks a real defect can fail: a truncated stream, a stage that stops
    # doing work, and the budget regression this file exists to catch.
    assert len(total_ms) == N_WINDOWS, (
        f"instrumented {len(total_ms)} windows, expected {N_WINDOWS} -- the stream was truncated, "
        "so the percentiles below describe a different run than the banner claims"
    )
    # A THIRD criterion was deleted here 2026-08-05, for the same F11 reason as the two above and
    # after the same mistake: `assert enc_mean > 0.0 and match_mean > 0.0` was introduced as "a
    # stage that stops doing work", but a mean of 300 `perf_counter` deltas around a real call is
    # never 0.0. MEASURED: the encode stage was neutered to a single dict lookup returning a cached
    # spike train and the assert STAYED GREEN; an empty-body timing reads exactly 0.0 ms only ~11%
    # of the time (217 of 2000), so a MEAN of 300 needs all 300 to be zero. No reachable red.
    #
    # What replaces it reads the matcher's OWN OUTPUT, which this loop previously threw away into
    # `_`. A neutered encoder (constant code) or a neutered matcher (constant distance) collapses
    # `best` to a single value, so this DOES go red for "a stage stopped doing work" -- verified by
    # constructing both. Timing stays out of it: the tail numbers are the report, not the criterion.
    assert len(set(best)) > 1, (
        f"the matcher returned the same distance {best[0]!r} on all {len(best)} windows -- a stage "
        "is emitting a constant, so the latency below is timing a no-op"
    )
    assert p50 > 0.0, "latency must be positive"
    assert p50 < BUDGET_MS, (
        f"median sensor->command latency {p50:.3f} ms over budget {BUDGET_MS} ms"
    )
    print(
        f"self-check OK: {len(total_ms)} windows, {len(set(best))} distinct match distances "
        f"(no stage emitting a constant), p50/p95/p99 = {p50:.2f}/{p95:.2f}/{p99:.2f} ms, "
        f"under the {BUDGET_MS:.0f} ms budget"
    )


if __name__ == "__main__":
    main()
