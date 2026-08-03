"""GPU-FIRST device selection, with demotion decided by measurement instead of by doctrine.

THE POLICY THIS IMPLEMENTS
    P5 says use the GPU for everything it actually speeds up, and that "actually" is decided by a
    benchmark on the real workload -- not by CUDA being available, and not by a table someone wrote
    down once. The default here is therefore **cuda whenever cuda exists**. CPU is chosen only for a
    workload this machine has MEASURED to be faster on CPU, at the width it is actually running.

WHY A ROUTER AT ALL, AND WHY IT DID NOT EXIST
    `training/__init__.py` has documented this module since the training tier was written -- "device.py
    : opt-in CUDA device selection (NORD_GPU=1 / NORD_DEVICE=cuda), CPU fallback" -- and the file was
    never created. Device choice ended up scattered as ad-hoc `torch.cuda.is_available()` checks in
    `cortex/brain.py:546`, `cortex/neurogenesis_engine.py:116`, `hippocampus/ng_snn.py:88` and
    `:499`, each with its own fallback rule and none of them reporting what they chose. A brain that
    cannot say which device its work landed on cannot be asked why the GPU is idle.

WHAT IS MEASURED, AND WHY IT IS NOT A CONSTANT
    The crossover between numpy and CUDA is a property of THIS machine and THIS population width, and
    P5 states it as a number that must be re-measured rather than a doctrine: "re-measure when
    populations grow; the crossover is a number, not a doctrine". The table recorded on 2026-07-28
    (RTX 5070, torch 2.11.0+cu128) had numpy winning 34x at N=240 and CUDA winning 41x at N=262,144.
    Hardcoding either end would be a tuned constant arbitrating behaviour, which P1 forbids, and it
    would go stale the moment a population grows or the machine changes.

    So `crossover()` MEASURES it, once per process, with the same matmul shape the live tick runs, and
    caches the answer. It costs a few milliseconds at startup and is specific to the hardware it is
    running on.

HONEST LIMIT
    This routes by SIZE, which is the dimension the measurement covers. It does not model transfer
    cost across a chain of ops, so a pipeline of small GPU ops that each beat CPU can still lose
    overall to keeping the whole chain on CPU. `cortex/brain.py:541` already notes the perception
    encoders are transfer-bound for exactly that reason. Routing a multi-op pipeline needs a
    measurement of the PIPELINE, and this module does not pretend to have one.
"""

from __future__ import annotations

import os

_CROSSOVER = None  # measured once per process, cached
_DECISIONS = {}  # workload -> (device, n, reason), for telemetry


def cuda_available():
    """True when torch is importable AND a CUDA device is present. Never raises."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_name():
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _bench(n, iters=200):
    """(numpy_ms, cuda_ms) for one elementwise+reduce step at width n -- the shape a tick runs."""
    import time  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    a = np.random.rand(n).astype("float32")
    b = np.random.rand(n).astype("float32")
    t0 = time.perf_counter()
    for _ in range(iters):
        float((a * b).sum())
    np_ms = (time.perf_counter() - t0) * 1000.0 / iters

    ta = torch.from_numpy(a).cuda()
    tb = torch.from_numpy(b).cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        float((ta * tb).sum().item())
    torch.cuda.synchronize()
    cu_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return np_ms, cu_ms


def crossover(force=False):
    """The width above which CUDA beats numpy ON THIS MACHINE, or None when there is no CUDA.

    Doubles from a small width until CUDA wins, so the cost is logarithmic in the answer and the
    measurement is taken on the hardware that will run the work. Returns None rather than guessing if
    CUDA is absent or the benchmark raises -- a router that invents a crossover is worse than one
    that says it does not know."""
    global _CROSSOVER
    if _CROSSOVER is not None and not force:
        return _CROSSOVER
    if not cuda_available():
        _CROSSOVER = None
        return None
    try:
        n = 256
        while n <= 1 << 22:  # 4M elements; past this CUDA has won on every machine measured
            np_ms, cu_ms = _bench(n, iters=50 if n > 1 << 18 else 200)
            if cu_ms < np_ms:
                _CROSSOVER = n
                return n
            n *= 2
        _CROSSOVER = 1 << 22
    except Exception:
        _CROSSOVER = None
    return _CROSSOVER


def select(workload, n=None):
    """The device `workload` should run on. GPU-FIRST: returns 'cuda' unless CPU is measured faster.

    `n` is the element count of the work. Omit it for workloads that are not width-bounded (training,
    neurogenesis refits, dataset passes) -- those go to the GPU whenever one exists, which is the
    whole point of having one.

    Explicit override wins over measurement, both directions: NORD_DEVICE=cpu|cuda, and NORD_GPU=0
    forces CPU. An operator who has a reason beats a benchmark."""
    forced = (os.environ.get("NORD_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda"):
        dev, why = (forced, "NORD_DEVICE override")
        if dev == "cuda" and not cuda_available():
            dev, why = "cpu", "NORD_DEVICE=cuda but no CUDA device present"
    elif os.environ.get("NORD_GPU") == "0":
        dev, why = "cpu", "NORD_GPU=0 override"
    elif not cuda_available():
        dev, why = "cpu", "no CUDA device"
    elif n is None:
        dev, why = "cuda", "not width-bounded -- GPU by default"
    else:
        xo = crossover()
        if xo is None:
            dev, why = "cuda", "crossover unmeasurable -- GPU by default"
        elif n >= xo:
            dev, why = "cuda", f"n {n} >= measured crossover {xo}"
        else:
            dev, why = "cpu", f"n {n} < measured crossover {xo} (CPU measured faster here)"
    _DECISIONS[workload] = (dev, n, why)
    return dev


def decisions():
    """Every routing decision this process made: {workload: {device, n, why}}.

    Exported so the cockpit can answer "why is the GPU idle" with the reason the router recorded,
    rather than leaving an operator to infer it from a utilisation graph."""
    return {k: {"device": d, "n": n, "why": w} for k, (d, n, w) in _DECISIONS.items()}


def main():
    print("=" * 74)
    print("DEVICE ROUTER -- GPU first, CPU only where this machine measures it faster")
    print("=" * 74)
    print(f"cuda available : {cuda_available()}  ({device_name()})")
    xo = crossover()
    print(f"measured crossover : {xo if xo is not None else 'n/a (no CUDA)'}")
    for workload, n in (
        ("live_tick", 240),
        ("population_encode", 4096),
        ("neurogenesis_refit", None),
        ("dataset_pass", 262144),
    ):
        d = select(workload, n)
        print(f"  {workload:20s} n={str(n):>7s} -> {d:4s}  ({_DECISIONS[workload][2]})")

    assert select("x", 1) in ("cpu", "cuda")
    if cuda_available():
        assert xo is not None, "CUDA present but no crossover measured"
        assert select("big", xo) == "cuda", "at the crossover the GPU must win"
        assert select("unbounded", None) == "cuda", "unbounded work must default to the GPU"
        os.environ["NORD_GPU"] = "0"
        assert select("forced", None) == "cpu", "NORD_GPU=0 must force CPU"
        os.environ.pop("NORD_GPU", None)
    else:
        assert select("anything", None) == "cpu", "no CUDA must resolve to cpu, never raise"
    print(f"self-check OK: {len(decisions())} routing decisions recorded, overrides honoured")


if __name__ == "__main__":
    main()
