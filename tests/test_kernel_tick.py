"""KernelTick (specs/022 T014, FR-005) -- delivery exactness, thread-count train equality,
integer order-invariance, and the P-vs-growth imbalance metric.

The delay contract is EXACT: a spike at tick t through an edge of delay d moves the post
membrane at tick t+d -- the first draft delivered at t+d+1 and the self-check's arrival arm
caught it. Thread equality here is the smoke form; the T∈{1..16} gate with its broken-control
arm is T015's (`gates/determinism_gate.py`), which must run at a constructed fixed P>1
because the MEASURED lineage P is 1 (tools/kernel_partition_baseline.json: P1/T1 = 784 ticks/s,
every parallel cell 40-60% slower at 4,096 units) and thread-invariance at P=1 has no
reachable red.
"""

import numpy as np
import pytest

from snn_data_lake.substrate.kernel_state import KernelState
from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses

FRAC = 16


def _drive(n, ix, level=2.0):
    ext = np.zeros(n, dtype=np.int64)
    ext[ix] = int(level * (1 << FRAC))
    return ext


def _build(pre, post, weight, delay, n, p_parts, threads, thr=1.0, decay=0.5):
    state = KernelState([("net", n)])
    state.arrays["thr"][:] = thr
    syn = Synapses(pre, post, weight, delay, n, p_parts, FRAC)
    return KernelTick(state, syn, decay=decay, v_reset=0.0, refrac_ticks=1, threads=threads)


def test_delay_arrival_is_exact():
    # a 3-link chain with distinct delays: 0 -(2)-> 1 -(3)-> 2 -(1)-> 3
    kt = _build([0, 1, 2], [1, 2, 3], [2.0, 2.0, 2.0], [2, 3, 1], n=4, p_parts=2, threads=1)
    fired = {0: kt.tick(_drive(4, [0])).tolist()}
    for t in range(1, 8):
        fired[t] = kt.tick().tolist()
    assert fired[0] == [0], fired
    assert fired[2] == [1], f"delay 2 must land at tick 2 exactly: {fired}"
    assert fired[5] == [2], f"the chain hop (2+3) must land at tick 5: {fired}"
    assert fired[6] == [3], f"the final hop (5+1) must land at tick 6: {fired}"
    quiet = [t for t in (1, 3, 4, 7) if fired[t]]
    assert not quiet, f"spikes on ticks where nothing is due: {fired}"


def test_trains_byte_equal_across_thread_counts_at_fixed_p():
    rng = np.random.default_rng(3)
    n, edges = 64, 512
    args = (
        rng.integers(0, n, edges),
        rng.integers(0, n, edges),
        rng.normal(0.4, 0.2, edges),
        rng.integers(1, 6, edges),
    )
    # ONE drive drawn BEFORE the loop: the first draft drew it inside, so each thread count ran
    # a different input and the "divergence" was the test's own rng advancing (instrument bug,
    # caught on first run -- validate the validator).
    drive = _drive(n, rng.integers(0, n, 8))
    runs = {}
    for threads in (1, 2, 8):
        kt = _build(*args, n=n, p_parts=4, threads=threads)
        runs[threads] = b"".join(kt.tick(drive if t == 0 else None).tobytes() for t in range(30))
    assert runs[1] == runs[2] == runs[8], (
        "trains diverged across thread counts at fixed P -- the FR-005 invariant the "
        "partition-owned integer accumulators exist to guarantee"
    )


def test_integer_accumulation_is_exact():
    # two pres converge on one post the same tick; the fixed-point sum must be exact
    kt = _build([0, 1], [2, 2], [0.6, 0.6], [1, 1], n=3, p_parts=2, threads=1, thr=1.1)
    kt.tick(_drive(3, [0, 1]))
    fired = kt.tick().tolist()
    assert fired == [2], (
        "0.6 + 0.6 in fixed-point must clear a 1.1 threshold -- convergent delivery lost charge"
    )


def test_refractory_gates_firing():
    kt = _build([0], [1], [2.0], [1], n=2, p_parts=1, threads=1)
    assert kt.tick(_drive(2, [0])).tolist() == [0]
    # drive unit 0 again immediately: refrac_ticks=1 must gate it this tick
    assert 0 not in kt.tick(_drive(2, [0])).tolist(), "a refractory unit fired"
    assert kt.tick(_drive(2, [0])).tolist()[0:1] == [0], "refractory never released"


def test_imbalance_metric_is_exact():
    kt = _build([0], [1], [1.0], [1], n=10, p_parts=4, threads=1)
    # 10 units over 4 partitions -> counts (3, 3, 2, 2); max/mean = 3 / 2.5
    assert abs(kt.partition_imbalance() - 3 / 2.5) < 1e-12


def _plain_kernel(n=4):
    state = KernelState([("net", n)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses([0], [2], [2.0], [1], n, 1, FRAC)
    return KernelTick(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=1)


def test_the_kernel_round_trips_through_capture_and_restore():
    """G5 (T017): a captured kernel resumes THE SAME trajectory.

    This was the POSITIVE CONTROL inside a paged-kernel refusal test until 2026-09-09, when the
    paging layer was deleted on its null (T068 LOSS, evidence/s5-paging-null.md 3.4/3.5) and
    `replay_gate._PAGED_REFUSAL` went with it: `KernelTick` no longer takes a `resident` mask, so
    that criterion's red can no longer be CONSTRUCTED, and a criterion with no reachable red is
    deleted rather than weakened (P8 clause 1 / F11). The control outlives the claim it
    controlled, deliberately -- it is the only round-trip pin `capture` and `restore` have in the
    suite, so deleting it with the refusal would have removed live coverage of a surviving
    mechanism."""
    from snn_data_lake.gates.replay_gate import capture, restore

    n = 4
    plain = _plain_kernel(n)
    plain.tick(_drive(n, [0]))
    blob = capture(plain, {})
    resumed = _plain_kernel(n)
    assert restore(blob, resumed) == {}
    assert resumed.tick_index == plain.tick_index
    assert np.array_equal(resumed.state.arrays["v"], plain.state.arrays["v"])


def test_refusals():
    with pytest.raises(ValueError):
        Synapses([0], [1], [1.0], [0], 2, 1, FRAC)  # 0-tick delay
    with pytest.raises(ValueError):
        Synapses([0], [5], [1.0], [1], 2, 1, FRAC)  # endpoint out of range
    with pytest.raises(ValueError):
        _build([0], [1], [1.0], [1], n=2, p_parts=1, threads=1, decay=0.0)  # decay bound
    state = KernelState([("net", 3)])
    syn = Synapses([0], [1], [1.0], [1], 2, 1, FRAC)
    with pytest.raises(ValueError):
        KernelTick(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=1)  # unit-count mismatch
