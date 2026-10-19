"""Codex B2 pfc-p2 MAJOR 1: `PrefrontalExecutive._decide_spiking` is `rng.getstate()` ->
`_native.pfc_decide` -> `rng.setstate()`. Without a transaction, two threads (free-threaded
3.14t) that both snapshot before either restores lose one advance: the final state equals ONE
call. The deleted Python body advanced the `random.Random` through atomic method calls, so two
overlapping calls always left two advances. Contract: the final state after two overlapping
calls equals two sequential calls. Oracle start state: chain A of
`native/sndl-core/tests/fixtures_pfc.json` (as tests/test_native_pfc.py).
"""

import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import snn_data_lake.faculties.prefrontal_executive as pe
from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive

FIX = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "native"
        / "sndl-core"
        / "tests"
        / "fixtures_pfc.json"
    ).read_text()
)
EVIDENCE = [1.0, 0.5] + [0.0] * 11  # the review's reproducer: a real race, not the zero path


def _chain_a_executive():
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    p = PrefrontalExecutive(**ch["ctor"])
    p.rng.setstate((3, tuple(ch["state0"]), None))
    return p


class _Rendezvous:
    """`_native` stand-in whose `pfc_decide` waits for the other thread to reach it (so both have
    taken their getstate snapshot) before running the real binding. A thread the fix keeps out
    never arrives: the wait times out and the thread inside simply continues."""

    def __init__(self, real):
        self.real = real
        self.barrier = threading.Barrier(2, timeout=0.5)

    def pfc_decide(self, *args):
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return self.real.pfc_decide(*args)


def test_two_overlapping_decisions_advance_the_rng_twice(monkeypatch):
    p = _chain_a_executive()
    sequential = [list(p._decide_spiking(EVIDENCE)) for _ in range(2)]
    two_calls = p.rng.getstate()
    p = _chain_a_executive()
    p._decide_spiking(EVIDENCE)
    one_call = p.rng.getstate()
    assert one_call != two_calls

    p = _chain_a_executive()
    monkeypatch.setattr(pe, "_native", _Rendezvous(pe._native))
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(p._decide_spiking, EVIDENCE) for _ in range(2)]
        overlapping = [list(f.result()) for f in futures]

    final = p.rng.getstate()
    assert final != one_call, "one advance was lost: the final state is ONE call's"
    assert final == two_calls
    assert sorted(overlapping) == sorted(sequential)
