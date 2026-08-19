"""Pinned-host performance baseline + ratchet (specs/022 T001, FR/SC-002).

The ONE place performance claims are measured and asserted. CI registers the
--report mode only ("CI never hosts performance claims", plan.md): the assert
runs on the pinned measurement host (the owner's box) as a pre-push gate,
flipped on at T012.

Modes
  --record       measure and write tools/perf_baseline.json (the baseline)
  --report       measure and print current-vs-baseline; ALWAYS exits 0
  --assert       ratchet: current wake-tick p50/p95 must be within TOLERANCE of
                 the baseline (per-unit-normalized when unit counts are known
                 on both sides, so growth does not read as regression); exits 1
                 on regression
  --self-check   prove the assert can FAIL: score the live measurement against
                 a synthetically-halved baseline (must red) AND against the
                 real baseline in the same run (must green) -- the positive
                 control and the red share one command, per the standing rule.

What is measured
  * wake-tick p50/p95 and sleep-tick p50/p95 over N ticks of a scratch brain
    (NORD_HOME is pointed at a temp dir for the whole run: a probe must never
    touch the live lake), with the wake/sleep duty cycle reported beside them
    -- the circadian oscillator free-runs, so the ratio is data, not a constant.
  * D: random-read IOPS at 4 KiB with OS buffering DISABLED
    (FILE_FLAG_NO_BUFFERING via ctypes; a buffered figure would measure the
    page cache, not the disk).

Re-baselining is an owner-visible act: --record refuses to overwrite an
existing baseline unless --stage NAME differs from the recorded stage, so a
stage exit re-baseline is explicit in the diff.

Bare invocation (the run_selfchecks path) is a cheap WIRING check only: the
baseline file parses and the scorer discriminates on synthetic numbers. It
never constructs a brain or touches the disk benchmark. The baseline JSON
lives at tools/perf_baseline.json (data beside the legacy instruments; this
module is governed, P4c).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "tools" / "perf_baseline.json"
TOLERANCE = 0.15  # 15% headroom over baseline before the ratchet reds
N_TICKS = 120
IOPS_FILE_MB = 256
IOPS_READS = 1500
SECTOR = 4096


def _measure_ticks(n_ticks: int = N_TICKS) -> dict:
    """Tick a scratch brain n times; classify wake/sleep by which phase method ran."""
    scratch = tempfile.mkdtemp(prefix="perf_baseline_")
    os.environ["NORD_HOME"] = scratch  # before any snn_data_lake import
    from snn_data_lake.cortex.brain import make_brain  # noqa: PLC0415

    brain = make_brain(seed=0)
    phases: list[str] = []
    orig_wake, orig_sleep = brain._wake_tick, brain._sleep_tick

    def wake(*a, **k):
        phases.append("wake")
        return orig_wake(*a, **k)

    def sleep(*a, **k):
        phases.append("sleep")
        return orig_sleep(*a, **k)

    brain._wake_tick, brain._sleep_tick = wake, sleep
    durations: list[tuple[str, float]] = []
    for _ in range(n_ticks):
        n_before = len(phases)
        t0 = time.perf_counter()
        brain.tick()
        dt = (time.perf_counter() - t0) * 1000.0
        phase = phases[-1] if len(phases) > n_before else "other"
        durations.append((phase, dt))

    def pcts(vals: list[float]) -> dict:
        if not vals:
            return {"p50_ms": None, "p95_ms": None, "n": 0}
        vals = sorted(vals)
        return {
            "p50_ms": round(statistics.median(vals), 3),
            "p95_ms": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
            "n": len(vals),
        }

    wake_ms = [d for p, d in durations if p == "wake"]
    sleep_ms = [d for p, d in durations if p == "sleep"]
    n_units = None
    for attr in ("n_units", "unit_count"):
        n_units = getattr(getattr(brain, "neuro", None), attr, None)
        if isinstance(n_units, int):
            break
    return {
        "wake": pcts(wake_ms),
        "sleep": pcts(sleep_ms),
        "duty_wake_fraction": round(len(wake_ms) / max(1, len(durations)), 3),
        "n_ticks": n_ticks,
        "n_units": n_units if isinstance(n_units, int) else None,
    }


def _measure_iops(drive_dir: str | None = None) -> dict:
    """Unbuffered 4 KiB random reads on D: (FILE_FLAG_NO_BUFFERING)."""
    import random  # noqa: PLC0415

    rng = random.Random(0)
    target_dir = drive_dir or ("D:\\" if os.path.exists("D:\\") else tempfile.gettempdir())
    path = os.path.join(target_dir, "_perf_baseline_iops.tmp")
    size = IOPS_FILE_MB * 1024 * 1024
    with open(path, "wb") as f:
        block = os.urandom(1024 * 1024)
        for _ in range(IOPS_FILE_MB):
            f.write(block)
    GENERIC_READ, FILE_SHARE_READ = 0x80000000, 0x1
    OPEN_EXISTING, NO_BUFFERING = 3, 0x20000000
    k32 = ctypes.windll.kernel32
    handle = k32.CreateFileW(
        path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, NO_BUFFERING, None
    )
    if handle == wt.HANDLE(-1).value:
        os.remove(path)
        raise OSError("CreateFileW failed for unbuffered IOPS measurement")
    try:
        buf = ctypes.create_string_buffer(SECTOR)
        read = wt.DWORD(0)
        t0 = time.perf_counter()
        for _ in range(IOPS_READS):
            offset = rng.randrange(0, size // SECTOR) * SECTOR
            k32.SetFilePointerEx(handle, ctypes.c_longlong(offset), None, 0)
            if not k32.ReadFile(handle, buf, SECTOR, ctypes.byref(read), None):
                raise OSError("ReadFile failed during IOPS measurement")
        elapsed = time.perf_counter() - t0
    finally:
        k32.CloseHandle(handle)
        os.remove(path)
    return {
        "random_read_iops_4k_unbuffered": round(IOPS_READS / elapsed),
        "reads": IOPS_READS,
        "file_mb": IOPS_FILE_MB,
        "target_dir": target_dir,
    }


def _normalized(metrics: dict, baseline: dict) -> tuple[float, float, float, float, str]:
    """Return (cur_p50, cur_p95, base_p50, base_p95, basis)."""
    cur, base = metrics["wake"], baseline["ticks"]["wake"]
    cu, bu = metrics.get("n_units"), baseline["ticks"].get("n_units")
    if cu and bu:
        return (
            cur["p50_ms"] / cu,
            cur["p95_ms"] / cu,
            base["p50_ms"] / bu,
            base["p95_ms"] / bu,
            "per-unit",
        )
    return (cur["p50_ms"], cur["p95_ms"], base["p50_ms"], base["p95_ms"], "raw")


def _score(metrics: dict, baseline: dict) -> tuple[bool, str]:
    c50, c95, b50, b95, basis = _normalized(metrics, baseline)
    ok50, ok95 = c50 <= b50 * (1 + TOLERANCE), c95 <= b95 * (1 + TOLERANCE)
    msg = (
        f"wake p50 {c50:.3f} vs baseline {b50:.3f} ({basis}) -> {'OK' if ok50 else 'REGRESSION'}; "
        f"p95 {c95:.3f} vs {b95:.3f} -> {'OK' if ok95 else 'REGRESSION'} "
        f"(duty now {metrics['duty_wake_fraction']}, baseline {baseline['ticks']['duty_wake_fraction']})"
    )
    return ok50 and ok95, msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--assert", dest="assert_", action="store_true")
    mode.add_argument("--self-check", dest="self_check", action="store_true")
    ap.add_argument("--stage", default="S0-pre")
    args = ap.parse_args(argv)

    if not (args.record or args.report or args.assert_ or args.self_check):
        # Bare invocation (run_selfchecks path): a CHEAP wiring check -- the baseline
        # parses and the scoring math discriminates on SYNTHETIC numbers. Deliberately
        # no brain construction and no disk benchmark: anything that observes the brain
        # must be cheap enough not to become the load (POLICIES cross-cutting).
        if not BASELINE.exists():
            print(
                "perf_baseline wiring-check: no baseline recorded yet (run --record on the pinned host)"
            )
            return 0
        baseline = json.loads(BASELINE.read_text())
        fast = {"wake": {"p50_ms": 1.0, "p95_ms": 1.0}, "n_units": None, "duty_wake_fraction": 0.5}
        slow = {"wake": {"p50_ms": 1e6, "p95_ms": 1e6}, "n_units": None, "duty_wake_fraction": 0.5}
        ok_fast, _ = _score(fast, baseline)
        ok_slow, _ = _score(slow, baseline)
        if not ok_fast or ok_slow:
            print(
                "perf_baseline wiring-check FAILED: scoring does not discriminate (fast should pass, slow should fail)"
            )
            return 1
        print(
            f"perf_baseline wiring-check OK: baseline stage={baseline.get('stage')!r} parses; scorer discriminates; iops={baseline['iops']['random_read_iops_4k_unbuffered']}"
        )
        return 0

    if args.record:
        if BASELINE.exists():
            prior = json.loads(BASELINE.read_text())
            if prior.get("stage") == args.stage:
                print(
                    f"perf_baseline: REFUSING overwrite -- baseline for stage {args.stage!r} exists; re-baselining needs a NEW --stage name (an owner-visible act)"
                )
                return 1
        ticks = _measure_ticks()
        iops = _measure_iops()
        record = {
            "stage": args.stage,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.environ.get("COMPUTERNAME", "?"),
            "ticks": ticks,
            "iops": iops,
        }
        BASELINE.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0

    if not BASELINE.exists():
        print("perf_baseline: no baseline recorded; run --record on the pinned host first")
        return 0 if args.report else 1
    baseline = json.loads(BASELINE.read_text())

    if args.report or args.assert_:
        # THE ASSERT DECLINES JURISDICTION OFF THE PINNED HOST (specs/022 T012, the pre-push
        # flip). A cross-host comparison is noise judging noise -- a slower laptop would red
        # every push and a faster box would wave regressions through -- so this exit-0 is a
        # SKIPPED notice, never a verdict. CI stays report-only for the same reason
        # ("CI never hosts performance claims", plan.md); the binding assert runs only where
        # the baseline was recorded.
        here = os.environ.get("COMPUTERNAME", "?")
        if args.assert_ and baseline.get("host") != here:
            print(
                f"perf_baseline [assert] SKIPPED, not a verdict: host {here!r} is not the "
                f"pinned baseline host {baseline.get('host')!r} -- the ratchet binds on the "
                "pinned host only"
            )
            return 0
        # TIMING NOISE IS ONE-SIDED (interference only ever ADDS time), so the assert takes
        # the best of up to 3 runs: pass on the first green, red only when EVERY attempt
        # regresses. Measured at the T012 flip, same binary, three back-to-back runs beside
        # the live daemon: p95 118 -> 131 -> 205 ms -- a single-run assert would block pushes
        # at random, and a flaky gate trains overrides (this gate's own first assert was
        # exactly such a false red). A real regression moves every run, so the red stays
        # reachable; --self-check's halved-baseline arm constructs it.
        ok, msgs = False, []
        for _ in range(3 if args.assert_ else 1):
            ticks = _measure_ticks()
            ok, msg = _score({**ticks}, baseline)
            msgs.append(msg)
            if ok:
                break
        tail = f" (best of {len(msgs)} attempts)" if args.assert_ and len(msgs) > 1 else ""
        print(f"perf_baseline [{'assert' if args.assert_ else 'report'}] {msgs[-1]}{tail}")
        return (0 if ok else 1) if args.assert_ else 0

    # --self-check: red against a halved baseline, green against the real one, ONE command.
    # Full baseline length: a p95 over a short run is a different (noisier) statistic and
    # produced a false regression when this used 40 ticks -- lengths must match to compare.
    ticks = _measure_ticks()
    halved = json.loads(json.dumps(baseline))
    halved["ticks"]["wake"]["p50_ms"] /= 2.0
    halved["ticks"]["wake"]["p95_ms"] /= 2.0
    red_ok, red_msg = _score({**ticks}, halved)
    green_ok, green_msg = _score({**ticks}, baseline)
    print(f"perf_baseline self-check RED arm (halved baseline, must fail): {red_msg}")
    print(f"perf_baseline self-check GREEN arm (real baseline, must pass): {green_msg}")
    if red_ok:
        print(
            "SELF-CHECK FAILED: the constructed regression did not red -- the ratchet cannot fail"
        )
        return 1
    if not green_ok:
        print(
            "SELF-CHECK: green arm regressed vs baseline (real regression or noisy host) -- investigate before trusting the ratchet"
        )
        return 1
    print("perf_baseline self-check OK: red arm reds, green arm greens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
