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

# THE AMBIENT-RATIO CONTROL (T012 follow-up, owner-approved 2026-08-20). The absolute form
# blocked three clean pushes in its first day: the pinned host's dominant variable is the LIVE
# DAEMON's own activity phase (boot refit bursts; per-tick cost growing with its 650 MB lake),
# proven by a suspend A/B -- daemon paused -> p95 126 ms green, daemon churning -> 199 ms red,
# SAME CODE. So every tick sample is now paired with an adjacent REFERENCE sample: crc32 over a
# fixed buffer -- single-threaded pure-C CPU work of deterministic size, the same execution
# profile as the tick (deliberately NOT a BLAS op, whose thread pool answers ambient load
# differently). Ambient that slows the brain slows the reference equally, and the per-sample
# RATIO divides it out. The scorer prefers ratio percentiles whenever the baseline carries
# them; a pre-ratio baseline falls back to the absolute compare (legacy, ambient-exposed).
REF_BYTES = 8 * 1024 * 1024
_REF_BUF: list[bytes] = []


def _ref_ms() -> float:
    """One reference sample, in ms."""
    import zlib  # noqa: PLC0415

    if not _REF_BUF:
        _REF_BUF.append(b"\x5a" * REF_BYTES)
    t0 = time.perf_counter()
    zlib.crc32(_REF_BUF[0])
    return (time.perf_counter() - t0) * 1000.0


def _measure_ticks(n_ticks: int = N_TICKS) -> dict:
    """Tick a scratch brain n times; classify wake/sleep by which phase method ran. Every tick
    sample is paired with an ADJACENT reference sample (see the ambient-ratio note above), so
    the wake ratios carry the ambient-corrected signal the scorer prefers."""
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
    durations: list[tuple[str, float, float]] = []
    for _ in range(n_ticks):
        n_before = len(phases)
        ref = _ref_ms()  # adjacent, so a load burst hits this pair together
        t0 = time.perf_counter()
        brain.tick()
        dt = (time.perf_counter() - t0) * 1000.0
        phase = phases[-1] if len(phases) > n_before else "other"
        durations.append((phase, dt, ref))

    def pcts(vals: list[float]) -> dict:
        if not vals:
            return {"p50_ms": None, "p95_ms": None, "n": 0}
        vals = sorted(vals)
        return {
            "p50_ms": round(statistics.median(vals), 3),
            "p95_ms": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
            "n": len(vals),
        }

    wake_ms = [d for p, d, _r in durations if p == "wake"]
    sleep_ms = [d for p, d, _r in durations if p == "sleep"]
    ref_ms = [r for _p, _d, r in durations]
    wake_ratios = [d / r for p, d, r in durations if p == "wake" and r > 0]
    n_units = None
    for attr in ("n_units", "unit_count"):
        n_units = getattr(getattr(brain, "neuro", None), attr, None)
        if isinstance(n_units, int):
            break

    def rpcts(vals: list[float]) -> dict:
        if not vals:
            return {"p50": None, "p95": None, "n": 0}
        vals = sorted(vals)
        return {
            "p50": round(statistics.median(vals), 3),
            "p95": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
            "n": len(vals),
        }

    return {
        "wake": pcts(wake_ms),
        "sleep": pcts(sleep_ms),
        "ref": pcts(ref_ms),
        "wake_ratio": rpcts(wake_ratios),
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


def _measure_state_per_unit(ticks: int = 600, caps: tuple = (1, 2, 4, 8, 16)) -> dict:
    """The MARGINAL bytes of persisted state per grown unit -- specs/022 T065's binding width.

    MEASURED AS A SLOPE, NOT A RATIO, and the difference is the whole instrument. The naive
    `blob_bytes / units` is confounded twice over:
      * with ticks varying and width fixed, the ratio moves entirely with the TRAJECTORY tail --
        measured: across 200-1200 ticks the unit count never leaves 2 while the blob grows
        5550 -> 6214 B, so the "per-unit" figure rises 2775 -> 3107 B/unit while no unit is added;
      * with width varying, the ratio charges the blob's FIXED overhead to the units -- measured:
        5503 B/unit at one unit against 776 B/unit at eleven, a factor of seven that is all
        intercept.
    So ticks are held FIXED, width is forced by the admission cap (`_l_max`, with `_eps_tol` driven
    to zero so the cap actually binds -- the inertness measured in evidence/sc-012a-width.md), and
    blob bytes are regressed on units. The SLOPE is the marginal cost of a unit; the INTERCEPT is
    the fixed overhead a ratio would have silently blamed on the units.

    The binding width follows by division and is reported at DISCLOSED budgets rather than at one
    picked here -- the budget is an input, not a property of the brain (P1)."""
    import tempfile as _tf  # noqa: PLC0415

    scratch = _tf.mkdtemp(prefix="perf_state_per_unit_")
    os.environ["NORD_HOME"] = scratch
    from snn_data_lake.cortex.brain import make_brain  # noqa: PLC0415

    pts = []
    for cap in caps:
        brain = make_brain(seed=0, persist=False)
        eng = getattr(brain, "neuro", None)
        if eng is None:
            brain.close()
            return {"slope_bytes_per_unit": None, "reason": "no growth engine (torch tier absent)"}
        eng._l_max = int(cap)
        eng._eps_tol = 0.0
        for _ in range(ticks):
            brain.tick()
        blob = eng.growth_state_bytes()
        pts.append((int(eng.grown_units), len(blob) if blob else 0))
        brain.close()

    xs = [u for u, _ in pts if u]
    ys = [n for u, n in pts if u]
    if len(set(xs)) < 2:
        # absence is a fact: with no width variation the slope is NOT measurable, and this
        # refuses rather than dividing and calling the ratio a per-unit cost
        return {
            "slope_bytes_per_unit": None,
            "reason": f"width did not vary (units {sorted(set(xs))}) -- slope not measurable",
            "points": pts,
        }
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    return {
        "slope_bytes_per_unit": round(slope, 2),
        "fixed_overhead_bytes": round(intercept, 2),
        "max_abs_residual_bytes": round(max(abs(r) for r in resid), 2),
        "points": pts,
        "ticks": ticks,
        # reported, never asserted: the width a budget buys. The budget is an INPUT.
        "binding_width_at_budget": {
            f"{mb}MB": (int((mb * 1024 * 1024 - intercept) / slope) if slope > 0 else None)
            for mb in (64, 256, 1024)
        },
    }


def _score_state_per_unit(metrics: dict, baseline: dict) -> tuple[bool, str]:
    """RATCHET: the marginal bytes-per-unit may not RISE past tolerance.

    A growing per-unit cost is the regression that matters for a substrate meant to scale -- it
    shrinks the binding width at every budget at once. Absent on either side is REPORTED, never a
    red: a baseline recorded before this metric existed, or a run without the growth engine, is a
    missing measurement rather than a regression (the standing rule that a zero is a claim about
    the instrument first)."""
    cur = (metrics.get("state_per_unit") or {}).get("slope_bytes_per_unit")
    base = (baseline.get("state_per_unit") or {}).get("slope_bytes_per_unit")
    if cur is None or base is None:
        return True, f"state-per-unit: cur={cur} baseline={base} -> reported, not asserted"
    ok = cur <= base * (1 + TOLERANCE)
    return ok, (
        f"state-per-unit {cur:.1f} B/unit vs baseline {base:.1f} "
        f"-> {'OK' if ok else 'REGRESSION'} (a rising per-unit cost shrinks the binding width)"
    )


def _normalized(metrics: dict, baseline: dict) -> tuple[float, float, float, float, str]:
    """Return (cur_p50, cur_p95, base_p50, base_p95, basis). Ambient-ratio basis whenever the
    baseline carries wake_ratio fields (see the module-top note); absolute otherwise."""
    base_t = baseline["ticks"]
    if base_t.get("wake_ratio", {}).get("p50") and metrics.get("wake_ratio", {}).get("p50"):
        cur, base = metrics["wake_ratio"], base_t["wake_ratio"]
        return (cur["p50"], cur["p95"], base["p50"], base["p95"], "ambient-ratio")
    cur, base = metrics["wake"], base_t["wake"]
    cu, bu = metrics.get("n_units"), base_t.get("n_units")
    if cu and bu:
        return (
            cur["p50_ms"] / cu,
            cur["p95_ms"] / cu,
            base["p50_ms"] / bu,
            base["p95_ms"] / bu,
            "per-unit",
        )
    return (cur["p50_ms"], cur["p95_ms"], base["p50_ms"], base["p95_ms"], "raw")


# distinct-from: substrate/gradient_claim_gate.py -- its reported_nulls publishes statistical
#   nulls asserted on by nothing; same reported-not-asserted PATTERN, different instrument and
#   different quantity. Site read; the pattern reuse is deliberate and named.
def _score(metrics: dict, baseline: dict) -> tuple[bool, str]:
    """On the ambient-ratio basis, p50 BINDS and p95 is REPORTED, not asserted -- a criterion
    deletion, recorded rather than smoothed (the gradient_claim_gate reported_nulls pattern).
    Measured on the flip's first day: the ratio made the p50 rock-stable beside the churning
    daemon (183.5 -> 178.7 -> 178.2) while the p95 tail stayed ambient-dominated (a load burst
    shorter than one ~110 ms tick lands INSIDE it and misses the 0.6 ms adjacent reference,
    and the daemon's refit cadence guarantees a few contended ticks per run) -- it red-ed on
    roughly half of all runs with ZERO real regressions behind it. A red that fires without
    cause cannot distinguish, and a flaky gate trains overrides; the tail stays visible in
    every report and in CI's trend row, and a genuine heavy-tail code change moves the median
    too. The legacy absolute basis keeps both asserts (pre-ratio baselines, unchanged)."""
    c50, c95, b50, b95, basis = _normalized(metrics, baseline)
    ok50, ok95 = c50 <= b50 * (1 + TOLERANCE), c95 <= b95 * (1 + TOLERANCE)
    ratio_basis = basis == "ambient-ratio"
    ok = ok50 if ratio_basis else (ok50 and ok95)
    if ratio_basis:
        p95_verdict = "reported" if ok95 else "reported-HIGH (tail is ambient-dominated here)"
    else:
        p95_verdict = "OK" if ok95 else "REGRESSION"
    msg = (
        f"wake p50 {c50:.3f} vs baseline {b50:.3f} ({basis}) -> {'OK' if ok50 else 'REGRESSION'}; "
        f"p95 {c95:.3f} vs {b95:.3f} -> {p95_verdict} "
        f"(duty now {metrics['duty_wake_fraction']}, baseline {baseline['ticks']['duty_wake_fraction']})"
    )
    return ok, msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--assert", dest="assert_", action="store_true")
    mode.add_argument("--self-check", dest="self_check", action="store_true")
    mode.add_argument(
        "--state-per-unit",
        dest="spu",
        action="store_true",
        help="T065: measure the marginal bytes-per-unit slope and score it against the baseline",
    )
    ap.add_argument("--stage", default="S0-pre")
    args = ap.parse_args(argv)

    if args.spu:
        # T065's OWN MODE, and it is separate for a measured reason. The slope needs five brains x
        # 600 ticks to force width with the admission cap; folding that into --assert would put
        # minutes onto every pre-push gate, and "anything that observes the brain must be cheap
        # enough not to become the load" is a standing invariant here. So the expensive
        # measurement runs on demand and at --record; the per-push gate keeps the tick ratchet.
        # DISCLOSED: that means the state-per-unit ratchet is NOT in the pre-push path -- it binds
        # when this mode is run, and its absence from --assert is stated rather than implied.
        if not BASELINE.exists():
            print("perf_baseline: no baseline recorded; run --record on the pinned host first")
            return 1
        baseline = json.loads(BASELINE.read_text())
        t0 = time.perf_counter()
        spu = _measure_state_per_unit()
        print(json.dumps(spu, indent=2))
        if not (baseline.get("state_per_unit") or {}).get("slope_bytes_per_unit"):
            # FIRST measurement establishes this metric's own baseline and NOTHING ELSE. A full
            # --record would re-measure ticks and iops too, silently resetting the reference the
            # tick ratchet has asserted against since S0-kernel -- a side effect nobody asked for
            # and the kind of quiet re-baseline this module's --stage refusal exists to prevent.
            if spu.get("slope_bytes_per_unit") is None:
                print(f"perf_baseline [state-per-unit] NOT MEASURABLE: {spu.get('reason')}")
                return 1
            baseline["state_per_unit"] = spu
            BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
            print(
                f"perf_baseline [state-per-unit] BASELINE ESTABLISHED at "
                f"{spu['slope_bytes_per_unit']} B/unit (ticks and iops UNTOUCHED)  "
                f"({time.perf_counter() - t0:.1f}s)"
            )
            return 0
        ok, msg = _score_state_per_unit({"state_per_unit": spu}, baseline)
        print(f"perf_baseline [state-per-unit] {msg}  ({time.perf_counter() - t0:.1f}s)")
        return 0 if ok else 1

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
        fast = {
            "wake": {"p50_ms": 1.0, "p95_ms": 1.0},
            "wake_ratio": {"p50": 0.001, "p95": 0.001},
            "n_units": None,
            "duty_wake_fraction": 0.5,
        }
        slow = {
            "wake": {"p50_ms": 1e6, "p95_ms": 1e6},
            "wake_ratio": {"p50": 1e6, "p95": 1e6},
            "n_units": None,
            "duty_wake_fraction": 0.5,
        }
        ok_fast, _ = _score(fast, baseline)
        ok_slow, _ = _score(slow, baseline)
        # T065: the state-per-unit ratchet is checked on the SAME cheap synthetic basis -- a
        # cheap-and-lean ratchet must still be shown to discriminate, or its green means nothing.
        _base_spu = baseline.get("state_per_unit") or {}
        if _base_spu.get("slope_bytes_per_unit"):
            _b = float(_base_spu["slope_bytes_per_unit"])
            ok_spu_fast, _ = _score_state_per_unit(
                {"state_per_unit": {"slope_bytes_per_unit": _b * 0.5}}, baseline
            )
            ok_spu_slow, _ = _score_state_per_unit(
                {"state_per_unit": {"slope_bytes_per_unit": _b * 100.0}}, baseline
            )
            ok_spu_absent, _ = _score_state_per_unit({}, baseline)
            if not ok_spu_fast or ok_spu_slow or not ok_spu_absent:
                print(
                    "perf_baseline wiring-check FAILED: the state-per-unit ratchet does not "
                    "discriminate (lean should pass, fat should fail, absent should report)"
                )
                return 1
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
        state_per_unit = _measure_state_per_unit()  # T065: the binding width's slope
        record = {
            "stage": args.stage,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.environ.get("COMPUTERNAME", "?"),
            "ticks": ticks,
            "iops": iops,
            "state_per_unit": state_per_unit,
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
    # halve WHICHEVER basis the scorer will use -- an absolute-only halving under the
    # ambient-ratio basis would leave the red arm comparing untouched ratio fields (vacuous)
    if halved["ticks"].get("wake_ratio", {}).get("p50"):
        halved["ticks"]["wake_ratio"]["p50"] /= 2.0
        halved["ticks"]["wake_ratio"]["p95"] /= 2.0
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
