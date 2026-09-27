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
    (:574-581), `growth/neurogenesis_engine.py:120`, `growth/ng_snn.py:88`, each with its own
    fallback rule and none of them reporting what they chose. A brain that cannot say which device
    its work landed on cannot be asked why the GPU is idle. (The two `brain.py` cites read `:546`
    and `:541` until 2026-08-04; both resolved onto unrelated attribute initialisations. The
    `neurogenesis_engine` cite was corrected in the SAME pass that claimed to correct them and was
    left at `:116`, which is `self.peak = float(peak)` -- the check is at `:120`. Re-resolved
    2026-08-05: :186 `device="cuda",`; :574/:581 the two `self._perception_device` assignments;
    ng_snn :88 `self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"`.
    The `neurogenesis_engine` cite is now given by SYMBOL -- `NeurogenesisSelfImprove.__init__`'s
    `self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")` -- because :120 went
    stale the same day it was re-resolved: adding an import and four docstring lines to that file
    moved it to :124. It is also no longer the whole story there, and that is the point of this
    pass: the attribute still records an availability guess, but `_refit` no longer routes on it.)

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

WHAT WAS DORMANT HERE, AND WHAT IS CHARGED NOW (P3 / P5a, 2026-08-05)
    Until 2026-08-05 `select()`, `crossover()` and `decisions()` -- the measured GPU-first router
    this module exists to BE -- had ZERO callers tree-wide outside this file's own self-check
    (derived by AST over `src/` + `tests/` + `tools/`: no module imported any of the three, and
    every bare `select(` call site found belonged to an unrelated method --
    `basal_ganglia/action_gating.py`, `brainstem/autonomous_loop.py`, `lakehouse/medallion.py`,
    `reasoning/rpm_solver.py`). What the importers used was `resolve_device` / `enable`, the
    compatibility surface ported in from the retired `reasoning/torch_device.py`, and that surface
    routed on AVAILABILITY. So the module was reachable, the merge was real, and the headline
    capability was uncharged.

    CLOSED on the producer side by two edits, not by adding importers to 41 modules:
      * `resolve_device()` no longer decides -- it resolves INTENT and calls `select()`. Every
        module that already imported it now routes through the measured path, and every one of
        their choices lands in `decisions()`.
      * `cortex/neurogenesis_engine._refit()` passes a REAL width (`x_tr.numel()`), so at least one
        live path is width-routed rather than availability-routed. Before that edit the refit ran on
        the card at ~60k elements against a measured crossover of 262,144 -- the 0.01x regime P5a
        was written to prevent.
    STILL OPEN, stated so it is not read as finished: one width-routed call site is one, not the
    tick path in general. ("nothing CONSUMES `decisions()`" stood here too and was FALSE WHEN
    WRITTEN -- `growth/neurogenesis_engine.py:51` has imported it since 6b33a45d, well before this
    paragraph was added. It is the same error as the `decisions()` docstring it cross-referenced,
    and it survived that docstring's correction on 2026-08-06 because only the sentence that was
    looked for got fixed. Corrected here; see `decisions()` for both consumers.)

HONEST ABOUT WHAT THIS BUYS TODAY: NOTHING VISIBLE, AND THAT IS THE CORRECT RESULT
    Every live width in this tree (24-240 for populations, ~60k elements for a neurogenesis refit)
    is below the crossover measured on this machine, so `select()` answers 'cpu' everywhere it is
    asked. The gain is not speed -- it is that the answer is now MEASURED instead of ASSUMED, and it
    flips to 'cuda' on its own when a population grows past the crossover, which P5a says is the
    expectation. Lowering the crossover, or adding a width constant, to make the flip visible today
    would be fitting a threshold to produce a demo; that is the defect this project deletes.

    `decisions()` additionally carried a claim about a consumer that did not exist: its docstring
    said it is "exported so the cockpit can answer 'why is the GPU idle'". That was corrected at
    the site rather than satisfied by adding a panel over a dict that was empty because nothing
    called `select()` -- "wiring a reader to an always-empty producer is the force-wiring P3
    forbids; the honest order is consumers for `select()` first, telemetry second."

    THE ORDER WAS HONOURED AND THE SECOND HALF IS NOW DONE (2026-08-06). The precondition was
    measured before the reader was built, which is the only thing that distinguishes this from the
    force-wire the paragraph above refuses: in a `make_brain()` process the dict is NON-EMPTY from
    tick 40 onward (`neurogenesis_refit`, via `_refit`'s `select()` call), and over 1,200 ticks
    sampled every 20 it takes 30 distinct widths and 30 distinct reasons. So there is something
    real to read. `infra/cockpit.py` reads it and draws the DEVICE ROUTING plate.
    **That is a REPORT, not a P6 faculty-to-faculty charge** -- no faculty's output changes because
    a page renders this -- and it is not counted as one.

    WHAT THE READER IS NOT ALLOWED TO CLAIM, and this is the half that needed measuring rather than
    wiring. "Why is the GPU idle" is more than this dict can answer. Measured in one 200-tick
    process: exactly ONE workload reaches `decisions()`, while `Brain._perception_device` is set to
    'cuda' by an availability check that never calls `select()`, and `torch.cuda`'s peak allocation
    over those ticks is 3,147,264 bytes. The GPU is not idle, and one 'cpu' row does not say it is.
    The plate is therefore captioned DEVICE ROUTING rather than with that question, tagged
    "router log, not a GPU census", and carries a note naming `_perception_device` as the choice
    made outside this router.
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


def select(workload, n=None, explicit=None):
    """The device `workload` should run on. GPU-FIRST: returns 'cuda' unless CPU is measured faster.

    `n` is the element count of the work. Omit it for workloads that are not width-bounded (training,
    dataset passes) -- those go to the GPU whenever one exists, which is the whole point of having
    one. ("neurogenesis refits" stood in that list until 2026-08-05 and was wrong: a refit fits an
    NG-SNN over a CONCRETE buffer, so its element count is `x_tr.numel()` and it is width-bounded
    like anything else. `cortex/neurogenesis_engine._refit` now passes it.)

    Explicit override wins over measurement, both directions: the `explicit` ARGUMENT first, then
    NORD_DEVICE=cpu|cuda, then NORD_GPU=0 forces CPU. An operator who has a reason beats a benchmark.

    `explicit` exists so `resolve_device` -- the compatibility surface 41 modules already import --
    can route THROUGH this function instead of re-deciding availability and precedence itself (P2:
    one mechanism, not two). It occupies the same slot as NORD_DEVICE and OUTRANKS it, because that
    is `resolve_device`'s pre-existing precedence (its argument beat the environment) preserved
    rather than changed. Passing `explicit` pins the answer, so the width test does not run -- which
    is the same rule NORD_DEVICE already had, applied to the argument that stands in front of it."""
    src = "explicit argument" if explicit else "NORD_DEVICE"
    forced = (explicit or os.environ.get("NORD_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda"):
        dev, why = (forced, f"{src} override")
        if dev == "cuda" and not cuda_available():
            dev, why = "cpu", f"{src}=cuda but no CUDA device present"
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

    READ SIDE CLOSED 2026-08-06. There are TWO consumers and they are different in kind; saying
    which is which is the point, because collapsing them is how "it has a reader" becomes a claim
    nobody measured:

      * A CRITERION reads it, and always did. `cortex/neurogenesis_engine.main()`'s claim 6 asserts
        the recorded width IS the fit tensor's own element count and that the recorded device
        follows the measured crossover -- the only proof in this tree that `_refit` routes on
        WIDTH rather than on availability, and its own comment records it firing red against a
        width taken from the wrong entry. **That consumer already existed while this docstring
        said "STILL NO TELEMETRY CONSUMER"**, which was true as written (no *telemetry* consumer)
        and false as it was read (nothing at all reads it). It is why retiring this function was
        REFUSED: deleting it deletes a criterion that has a reachable red, and P8 deletes criteria
        that have NO red, never ones that do.
      * A REPORT reads it, as of 2026-08-06. `infra/cockpit.py` serves it as the `device`
        section and draws the DEVICE ROUTING plate. A page rendering a number is not a faculty
        consuming a signal, so this is NOT a P6 read and no count is raised by it.

    WHAT THIS DICT IS NOT -- measured, because the sentence it used to carry ("so the cockpit can
    answer 'why is the GPU idle'") promises more than it can pay. Over 200 ticks of a `make_brain()`
    process: ONE workload lands here, `Brain._perception_device` is 'cuda' from an availability
    check that never calls `select()`, and peak CUDA allocation is 3,147,264 bytes. This is a LOG
    OF THE CHOICES THAT ASKED THE ROUTER, not a census of GPU use; a 'cpu' row does not license
    "the GPU is idle". The cockpit tags it "router log, not a GPU census" and its plate note names
    `_perception_device` as a choice made outside this router, so the page cannot imply otherwise.

    READS A COPY, and that is not defensive habit. The tick thread writes `_DECISIONS` while the
    cockpit's HTTP handler thread reads it, on a free-threaded interpreter (Py_GIL_DISABLED=1), so
    comprehending over the live dict is a "dictionary changed size during iteration" waiting for
    the first new workload key -- which is precisely what a growing brain produces. `.copy()` runs
    under the dict's own lock; the comprehension then walks a private snapshot."""
    return {k: {"device": d, "n": n, "why": w} for k, (d, n, w) in _DECISIONS.copy().items()}


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
# from here. ("nothing imports `select`/`crossover`/`decisions`" followed that clause and was
# already FALSE when written: `growth/neurogenesis_engine.py:51` imports all three, added in
# 6b33a45d. Re-derived 2026-08-06 -- the governed importers of those three are
# `growth/neurogenesis_engine.py` and, since the DEVICE ROUTING plate landed,
# `infra/cockpit.py`.) The figure recorded one day earlier
# was "41 / 40" and it was already one short -- `reasoning/rsnn_delays_torch.py` carries TWO such
# imports, and a per-module count and a per-statement count are not the same number. Re-derive with
# the walk above rather than inheriting this; the repoint sweep was still landing when it was taken.


def resolve_device(explicit=None, workload="resolve_device", n=None):
    """'cuda[:N]' when a GPU is wanted and `select()` agrees, else 'cpu'. Compatibility surface.

    Preserves `torch_device.resolve_device`'s contract for its existing callers -- an explicit
    argument or NORD_DEVICE / NORD_GPU=1 asks for CUDA, absence of CUDA falls back with a message
    rather than raising.

    WHAT CHANGED 2026-08-05, AND WHY IT IS THIS FUNCTION THAT CHANGED (P5a). This used to decide for
    itself: it read the env, called `cuda_available()`, and returned. `select()` -- the router that
    decides the same question by MEASUREMENT -- had zero importers tree-wide, so the whole tree
    routed on AVAILABILITY while the measured router sat dormant (P3: computed, never consumed).
    Repointing 41 modules at `select()` would have been 41 edits; repointing THIS function is one,
    because it is what all 41 already import. So it now resolves INTENT only (which device is being
    asked for, and by whom) and hands the actual decision to `select()`, which owns the override
    precedence, the availability check and the width test. P2: one mechanism, not two.

    PRECEDENCE, PRESERVED EXACTLY RATHER THAN TIDIED. The argument outranks NORD_DEVICE, which
    outranks NORD_GPU. That was already true here -- `resolve_device("cuda")` returned 'cuda' under
    NORD_DEVICE=cpu and under NORD_GPU=0 -- and it survives because the argument is passed into
    `select(explicit=...)`, the slot that stands in front of NORD_DEVICE. It is deliberately NOT
    "fixed" to let the env win: 41 callers depend on today's answer, and changing an override rule
    while claiming a behaviour-preserving refactor is how a silent regression ships.

    NORD_GPU=1 IS AN ENABLE, NOT A PIN, and that distinction is what makes the width test reachable
    from here at all. An argument or NORD_DEVICE pins the device (measurement skipped, per the rule
    `select()` already had). NORD_GPU=1 only expresses "a GPU is wanted", so it is NOT forwarded as
    a pin, and `select()` still applies the measured crossover to any caller that supplies `n`.

    THE DEFAULT IS CPU, and the paragraph that used to stand here said the opposite -- "the default
    DIFFERS from the retired module ... torch_device defaulted to CPU because it predated P5". The
    code it described has always returned 'cpu' when nothing asks for a GPU (`if not want ...`), so
    the claimed difference did not exist. Corrected rather than implemented: making the default GPU
    would silently move 41 modules onto the card, which is the opposite of what P5a is for.

    `workload` names the entry in `decisions()`; `n` is the element count when the caller knows it.
    """
    want = explicit or os.environ.get("NORD_DEVICE")
    pinned = bool(want)  # an argument or NORD_DEVICE is an OVERRIDE; NORD_GPU=1 is only an ENABLE
    if not want and os.environ.get("NORD_GPU") == "1":
        want = "cuda"
    if not want:
        want, pinned = "cpu", True  # the documented default: CPU unless something asks otherwise
    base = "cuda" if want.startswith("cuda") else want
    if base not in ("cpu", "cuda"):
        return want  # an unknown accelerator string ('mps', ...) passes through, as it always did
    dev = select(workload, n, explicit=base if pinned else None)
    if dev == "cuda":
        print(f"GPU: {device_name()} -> device={want}")
        return want
    if base == "cuda":
        if cuda_available():
            # A card exists and the router still said CPU -- only reachable when the caller passed a
            # width. The old text ("no CUDA device is present") would be a lie here, so it is not used.
            print(f"NOTE: device '{want}' requested; routed to CPU -- {_DECISIONS[workload][2]}")
        else:
            print(
                f"WARNING: device '{want}' requested but no CUDA device is present; falling back to CPU."
            )
    return "cpu"


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
        (
            "training_epoch",
            None,
        ),  # was "neurogenesis_refit" -- that one IS width-bounded, see select()
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

    # --- resolve_device DEFERS to select(), and behaves exactly as it did before it did (P5a) ----
    # EIGHT assertions, not the full 45-cell override matrix, and the difference is deliberate. The
    # matrix was proven byte-identical before/after by running every cell -- 45 with a card and 45
    # with `cuda_available` patched False -- which is evidence a self-check cannot carry cheaply.
    # What stays here is one assertion per DISTINCT rule, each with its own constructed red: a
    # summary assertion that only restates its neighbours has no independent red, which is the
    # tautological shape P8 forbids, so the one that did (narrow.device != wide.device) was cut.
    # `_gpu` is 'cpu' where no card exists, so this block runs unchanged on CI -- P5's hard
    # constraint, satisfied by construction rather than by a skip.
    _cuda = cuda_available()
    _gpu = "cuda" if _cuda else "cpu"
    _prior_dev, _prior_gpu = os.environ.get("NORD_DEVICE"), os.environ.get("NORD_GPU")
    try:
        for _k in ("NORD_DEVICE", "NORD_GPU"):
            os.environ.pop(_k, None)
        assert resolve_device() == "cpu", "no request and no env must stay on CPU (the default)"
        assert resolve_device("cuda") == _gpu, "an explicit cuda request follows the card"
        assert resolve_device("cuda:1") == ("cuda:1" if _cuda else "cpu"), (
            "the ':N' suffix must survive the round trip through select(), which only speaks "
            "'cpu'/'cuda' -- losing it would silently move work to device 0"
        )
        os.environ["NORD_DEVICE"] = "cpu"
        assert resolve_device("cuda") == _gpu, "the ARGUMENT outranks NORD_DEVICE, as it always did"
        del os.environ["NORD_DEVICE"]
        os.environ["NORD_GPU"] = "0"
        assert resolve_device("cuda") == _gpu, "the ARGUMENT outranks NORD_GPU=0, as it always did"
        os.environ["NORD_GPU"] = "1"
        assert resolve_device() == _gpu, "NORD_GPU=1 asks for the card"

        # NORD_GPU=1 is an ENABLE, not a pin -- so a caller that supplies a WIDTH is still measured.
        # These are the assertions about the new capability rather than the preserved one.
        assert resolve_device(workload="narrow_probe", n=1) == "cpu", (
            "width 1 must route to CPU even under NORD_GPU=1: one element is below any crossover, "
            "and an enable that pinned the device would make the measurement unreachable"
        )
        assert decisions()["narrow_probe"]["n"] == 1, (
            "the width must reach decisions(), not be lost"
        )
        if _cuda:  # no card -> no crossover to straddle, and nothing this can say
            assert resolve_device(workload="wide_probe", n=crossover()) == "cuda", (
                f"at the measured crossover ({crossover()}) the same call must reach the card; a "
                "router whose answer never changes with n is not routing"
            )
    finally:
        for _k, _v in (("NORD_DEVICE", _prior_dev), ("NORD_GPU", _prior_gpu)):
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v

    print(
        f"self-check OK: {len(decisions())} routing decisions recorded, overrides honoured; "
        "resolve_device defers to select() and its width reaches decisions()"
    )


if __name__ == "__main__":
    main()
