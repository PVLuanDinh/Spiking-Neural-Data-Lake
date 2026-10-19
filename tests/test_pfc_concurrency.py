"""Codex B2 pfc-p2 MAJOR 1: `PrefrontalExecutive._decide_spiking` is `rng.getstate()` ->
`_native.pfc_decide` -> `rng.setstate()`. Without a transaction, two threads (free-threaded
3.14t) that both snapshot before either restores lose one advance: the final state equals ONE
call. The deleted Python body advanced the `random.Random` through atomic method calls, so two
overlapping calls always left two advances. Contract: the final state after two overlapping
calls equals two sequential calls. Oracle start state: chain A of
`native/sndl-core/tests/fixtures_pfc.json` (shared with tests/test_native_pfc.py).
"""

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

from test_native_pfc import FIX, _executive

import snn_data_lake.faculties.prefrontal_executive as pe

EVIDENCE = [1.0, 0.5] + [0.0] * 11  # the review's reproducer: a real race, not the zero path


def _chain_a():
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    return _executive(ch)


class _Rendezvous:
    """`_native` stand-in whose `pfc_decide` waits for the other thread to reach it (both have
    taken their getstate snapshot by then) before running the real binding. A thread the lock keeps
    out never arrives: the wait times out, the barrier stays `broken`, and the thread inside simply
    continues."""

    def __init__(self, real):
        self.real = real
        self.barrier = threading.Barrier(2, timeout=1.0)

    def pfc_decide(self, *args):
        with contextlib.suppress(threading.BrokenBarrierError):
            self.barrier.wait()
        return self.real.pfc_decide(*args)


def _two_overlapping(p, monkeypatch, real):
    """Both workers are running (start barrier) before either enters `_decide_spiking`, so the
    second is at most a lock away from the binding. Returns (outputs, rendezvous)."""
    met = _Rendezvous(real)
    monkeypatch.setattr(pe, "_native", met)
    start = threading.Barrier(2)

    def worker():
        start.wait()
        return list(p._decide_spiking(EVIDENCE))

    with ThreadPoolExecutor(2) as pool:
        outs = [f.result() for f in [pool.submit(worker) for _ in range(2)]]
    return outs, met


def test_two_overlapping_decisions_advance_the_rng_twice(monkeypatch):
    real = pe._native
    p = _chain_a()
    sequential = [list(p._decide_spiking(EVIDENCE)) for _ in range(2)]
    two_calls = p.rng.getstate()
    p = _chain_a()
    p._decide_spiking(EVIDENCE)
    one_call = p.rng.getstate()
    assert one_call != two_calls

    # control: the lockless handoff (the B2 shape) on this schedule loses an advance -- the red the
    # lock is judged against; if the two threads do not meet, the schedule was not established
    p = _chain_a()
    p._rng_lock = contextlib.nullcontext()
    _, met = _two_overlapping(p, monkeypatch, real)
    assert not met.barrier.broken, "both threads must meet inside the binding"
    assert p.rng.getstate() == one_call

    p = _chain_a()
    outs, met = _two_overlapping(p, monkeypatch, real)
    assert met.barrier.broken, "the second call entered the binding while the first was inside"
    assert p.rng.getstate() == two_calls
    assert sorted(outs) == sorted(sequential)
