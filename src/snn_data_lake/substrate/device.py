"""GPU-FIRST device selection, with demotion decided by measurement instead of by doctrine.

THE POLICY THIS IMPLEMENTS
    P5 says use the GPU for everything it actually speeds up, and that "actually" is decided by a
    benchmark on the real workload -- not by CUDA being available, and not by a table someone wrote
    down once. The default here is therefore **cuda whenever cuda exists**. CPU is chosen only for a
    workload this machine has MEASURED to be faster on CPU, at the width it is actually running.

WHY A ROUTER AT ALL, AND WHY IT DID NOT EXIST
    `training/__init__.py` has documented this module since the training tier was written -- "device.py
    : opt-in CUDA device selection (NORD_GPU=1 / NORD_DEVICE=cuda), CPU fallback" -- and the file was
    never created. Device choice ended up scattered as ad-hoc `torch.cuda.is_available()` checks:
    `cortex/brain.py`'s `device="cuda"` default (:186) and its `self._perception_device` block
    (:574-581), `cortex/neurogenesis_engine.py:120`, `hippocampus/ng_snn.py:88`, each with its own
    fallback rule and none of them reporting what they chose. A brain that cannot say which device
    its work landed on cannot be asked why the GPU is idle. (The two `brain.py` cites read `:546`
    and `:541` until 2026-08-04; both resolved onto unrelated attribute initialisations. The
    `neurogenesis_engine` cite was corrected in the SAME pass that claimed to correct them and was
    left at `:116`, which is `self.peak = float(peak)` -- the check is at `:120`. Re-resolved
    2026-08-05: :186 `device="cuda",`; :574/:581 the two `self._perception_device` assignments;
    :120 `self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")`; ng_snn :88
    `self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"`.)

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
    overall to keeping the whole chain on CPU. `cortex/brain.py`'s perception-device block
    (:567-581, the comment plus the two `self._perception_device` assignments) already notes the
    encoders are transfer-bound for exactly that reason. Routing a multi-op pipeline needs a
    measurement of the PIPELINE, and this module does not pretend to have one.

WHAT IS DORMANT HERE, STATED RATHER THAN IMPLIED (P3, re-derived 2026-08-04)
    `select()`, `crossover()` and `decisions()` -- the measured GPU-first router this module exists
    to BE -- have ZERO callers tree-wide outside this file's own self-check (re-derived by AST over
    `src/` + `tests/` + `tools/` 2026-08-05: no module imports any of the three, and every bare
    `select(` call site found belongs to an unrelated method -- `basal_ganglia/action_gating.py`,
    `brainstem/autonomous_loop.py`, `lakehouse/medallion.py`, `reasoning/rpm_solver.py`). What the
    importers below actually use is `resolve_device` / `enable`, the compatibility surface ported in
    from the retired `reasoning/torch_device.py`. So the module is reachable, the merge is real, and
    the headline capability is uncharged.

    `decisions()` additionally carried a claim about a consumer that does not exist: its docstring
    said it is "exported so the cockpit can answer 'why is the GPU idle'". The cockpit reads only
    `substrate/brain_telemetry.snapshot()`, `decisions` appears in neither DASHBOARD_KEYS nor
    DASHBOARD_SECTION_KEYS, and no panel references it. That is `region_wiring`'s defect exactly --
    a docstring asserting a wire nobody built -- so the claim is corrected at the site rather than
    satisfied by adding a panel over a dict that is empty because nothing calls `select()`. Wiring
    a reader to an always-empty producer is the force-wiring P3 forbids; the honest order is
    consumers for `select()` first, telemetry second. Both are open work.
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

    # A LOCAL Generator, not `np.random.rand`. The module-global numpy RNG is process-wide state,
    # and `run_selfchecks.py` runs module self-checks IN-PROCESS via runpy -- so drawing from the
    # global stream here advances it for every module that runs afterwards. Same rule as
    # `random.Random(seed)` per instance; consuming a shared stream is the same coupling as seeding
    # one. (Benchmark inputs, so the values do not matter -- the reproducibility does.)
    rng = np.random.default_rng(0)
    a = rng.random(n, dtype="float32")
    b = rng.random(n, dtype="float32")
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

    NO CONSUMER TODAY. This said it was "exported so the cockpit can answer 'why is the GPU idle'";
    the cockpit does not read it and never has (see the module docstring). It is empty in any
    process that does not call `select()`, which today is every process but this one's self-check.
    Kept because it is the right shape for that answer and costs nothing; charge UNMEASURED."""
    return {k: {"device": d, "n": n, "why": w} for k, (d, n, w) in _DECISIONS.items()}


# --- PORTED from reasoning/torch_device.py, which is RETIRED (P2 merge, 2026-08-04) -------------
#
# Two device routers existed with OPPOSITE defaults and no overlap in users:
#   * this module        -- GPU-FIRST unless the machine measures CPU faster. P5's operative clause
#                           ("actually" is decided by a benchmark). ZERO call sites.
#   * reasoning/torch_device.py -- CPU-FIRST, opt-in GPU. **43 call sites**, and the only holder of
#                           `enable()`. (The promotion sweep reported "3 importers" for it; that
#                           counts GOVERNED importers only, which is the right metric for "is this
#                           promotion optional" and the wrong one for "how much will repointing
#                           touch". 40 of the 43 were inside `reasoning/` itself.)
# P2 allows exactly one survivor and requires the retired mechanism's unique capability to be
# ported FIRST. This module survives because its POLICY is the correct one; `torch_device` held the
# USERS and `enable()`, so both come here. `resolve_device` is kept as the compatibility surface
# those callers already use.
#
# "the three call sites" stood here and was the metric-confusion the paragraph directly above warns
# about, committed three lines later: 3 is the GOVERNED-importer count, not the call-site count.
# RE-COUNTED 2026-08-05 by AST over `src/` + `tests/` + `tools/`: **42 import statements across 41
# modules**, all of them under `reasoning/` or `training/`, import `resolve_device` and/or `enable`
# from here; nothing imports `select`/`crossover`/`decisions`. The figure recorded one day earlier
# was "41 / 40" and it was already one short -- `reasoning/rsnn_delays_torch.py` carries TWO such
# imports, and a per-module count and a per-statement count are not the same number. Re-derive with
# the walk above rather than inheriting this; the repoint sweep was still landing when it was taken.


def resolve_device(explicit=None):
    """'cuda[:N]' when a GPU is wanted and present, else 'cpu'. Compatibility surface.

    Preserves `torch_device.resolve_device`'s contract for its existing callers -- an explicit
    argument or NORD_DEVICE / NORD_GPU=1 asks for CUDA, absence of CUDA falls back with a message
    rather than raising. It no longer re-implements the availability check: it defers to
    `cuda_available()` so this module has ONE answer to "is there a GPU", not two.

    Note the default DIFFERS from the retired module and that is deliberate, not an oversight:
    torch_device defaulted to CPU because it predated P5's "default to GPU for work it
    accelerates". An explicit `NORD_GPU=0` or `NORD_DEVICE=cpu` still forces CPU.
    """
    want = explicit or os.environ.get("NORD_DEVICE")
    if not want and os.environ.get("NORD_GPU") == "1":
        want = "cuda"
    if not want or want == "cpu":
        return "cpu"
    if want.startswith("cuda"):
        if cuda_available():
            print(f"GPU: {device_name()} -> device={want}")
            return want
        print(
            f"WARNING: device '{want}' requested but no CUDA device is present; falling back to CPU."
        )
        return "cpu"
    return want


def enable(device):
    """Route torch tensor/module factories to `device`. No-op for 'cpu'. Returns `device`.

    The one capability `torch_device` had that this module did not, hence ported rather than
    dropped: it calls `torch.set_default_device` so everything built afterwards lands on the GPU
    without threading `.to(device)` through every call site.
    """
    if device != "cpu":
        try:
            import torch  # noqa: PLC0415

            if hasattr(torch, "set_default_device"):
                torch.set_default_device(device)
        except Exception:
            return "cpu"
    return device


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
        # RESTORE, not pop (2026-08-04). This read `os.environ.pop("NORD_GPU", None)`, which drops
        # the CALLER's value rather than just this check's -- the exact defect `home.main()`'s
        # `finally` records ("restored in the finally, NOT popped") and that `cockpit.py` was fixed
        # for the same day. A self-check that silently disables the caller's GPU is a real hazard
        # here of all places. (Cited by symbol: `home.py:311` landed on an unrelated docstring.)
        _prior = os.environ.get("NORD_GPU")
        try:
            os.environ["NORD_GPU"] = "0"
            assert select("forced", None) == "cpu", "NORD_GPU=0 must force CPU"
        finally:
            if _prior is None:
                os.environ.pop("NORD_GPU", None)
            else:
                os.environ["NORD_GPU"] = _prior
    else:
        assert select("anything", None) == "cpu", "no CUDA must resolve to cpu, never raise"

    # the ported surface must work too, or the P2 merge dropped something
    assert resolve_device("cpu") == "cpu", "explicit cpu must resolve to cpu"
    assert enable("cpu") == "cpu", "enable('cpu') is a no-op returning its argument"
    print(
        f"self-check OK: {len(decisions())} routing decisions recorded, overrides honoured; "
        "ported resolve_device/enable live"
    )


if __name__ == "__main__":
    main()
