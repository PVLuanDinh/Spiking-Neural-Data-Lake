"""SignalRing + LakeJournaller (specs/022 T011) -- the faculty ring's three contract points.

Single writer is ENFORCED (a second thread's publish refuses), multicast consumers see the same
bytes through independent cursors, and loss is LOUD: an overrun poll reports exactly what
vanished, and the lake journaller refuses to journal around a hole -- before consuming anything,
so refusal can never manufacture the very gap it refuses (P4b: a gapped lake is amnesia).
The charge proof is the replay A/B: the same produced stream lands byte-equal through the
journaller and not at all through the neutered arm.
"""

import threading

import numpy as np
import pytest

from snn_data_lake.engine.signal_ring import (
    DeliveringProducer,
    GradedUnbound,
    JoinConsumer,
    LakeJournaller,
    RingOverrun,
    SignalRing,
    UndeclaredJoin,
)
from snn_data_lake.engine.wiring_manifest import Port
from snn_data_lake.substrate.binlog import BinLog


def _records(n, salt=""):
    return [bytes([0x7F]) + f"rec-{salt}{k}".encode("ascii") for k in range(n)]


def test_replay_ab_charge(tmp_path):
    produced = _records(20)
    ring = SignalRing(slots=64)
    journal = LakeJournaller(ring, BinLog(str(tmp_path / "a.bin")))
    for p in produced:
        ring.publish(p)
    assert journal.drain() == 20
    assert list(BinLog(str(tmp_path / "a.bin")).replay()) == produced, (
        "the journaller must land the stream byte-equal, oldest first"
    )
    # Arm B: journaller neutered (never attached) -> the lake holds ZERO of the same stream.
    ring_b = SignalRing(slots=64)
    for p in produced:
        ring_b.publish(p)
    assert list(BinLog(str(tmp_path / "b.bin")).replay()) == [], (
        "the neutered arm reached the lake -- the wire's charge is fake"
    )


def test_multicast_independent_cursors():
    ring = SignalRing(slots=64)
    eager = ring.attach("eager")
    lazy = ring.attach("lazy")
    produced = _records(12)
    seen_eager = []
    for p in produced:
        ring.publish(p)
        recs, lost = eager.poll()  # polls after every publish
        assert lost == 0
        seen_eager += [payload for _s, payload in recs]
    recs, lost = lazy.poll()  # polls once at the end
    assert lost == 0
    assert seen_eager == [payload for _s, payload in recs] == produced, (
        "two consumers at different cadences must see the SAME stream"
    )


def test_attach_sees_only_the_future():
    ring = SignalRing(slots=8)
    for p in _records(5):
        ring.publish(p)
    late = ring.attach("late")
    recs, lost = late.poll()
    assert (recs, lost) == ([], 0), "history a consumer never subscribed to is not loss"


def test_overrun_reports_exact_loss_then_recovers():
    ring = SignalRing(slots=4)
    lagger = ring.attach("lagger")
    produced = _records(11)
    for p in produced:
        ring.publish(p)
    recs, lost = lagger.poll()
    assert lost == 7, f"11 publishes over 4 slots must report 7 lost, got {lost}"
    assert [payload for _s, payload in recs] == produced[7:], (
        "after an overrun the survivors must be the newest `slots` records, in order"
    )
    recs2, lost2 = lagger.poll()
    assert (recs2, lost2) == ([], 0), "loss must be reported ONCE, not re-counted forever"


def test_journaller_refuses_hole_without_consuming(tmp_path):
    ring = SignalRing(slots=4)
    journal = LakeJournaller(ring, BinLog(str(tmp_path / "lake.bin")))
    for p in _records(7):
        ring.publish(p)
    with pytest.raises(RingOverrun):
        journal.drain()
    assert list(BinLog(str(tmp_path / "lake.bin")).replay()) == [], (
        "a refused drain must append NOTHING -- a partial batch is a silently gapped lake"
    )
    with pytest.raises(RingOverrun):
        journal.drain()  # the hole does not heal by itself; refusal repeats loudly


def test_journaller_drain_reports_every_appended_payload_to_the_one_append_observer(tmp_path):
    """The ring journaller is the ONLY writer of the workspace coalition class (0x0A) and it wrote
    around `home.observe_append` -- the append budget's census could never see the tag that is
    80 % of the operator's lake by bytes. RUN RED first: the observer saw nothing across a drain.
    The drain reports each payload it appended, tag byte and tag-prefixed length, AFTER the
    BinLog extend (accounting never precedes or breaks a durable write); an empty drain and a
    refused drain report nothing (the control, in the same test)."""
    from snn_data_lake.substrate import home as home_mod

    ring = SignalRing(slots=8)
    journal = LakeJournaller(ring, BinLog(str(tmp_path / "lake.bin")))
    payloads = [home_mod._T_COALITION + f"coalition-{k}".encode("ascii") for k in range(3)]
    for p in payloads:
        ring.publish(p)
    seen = []
    prev = home_mod.set_append_observer(lambda tag, n: seen.append((tag, n)))
    try:
        assert journal.drain() == 3
        assert journal.drain() == 0  # the control: an empty drain observes nothing more
    finally:
        home_mod.set_append_observer(prev)
    assert seen == [(0x0A, len(p)) for p in payloads], (
        f"the journaller's appends are off the one observer path: {seen}"
    )
    assert list(BinLog(str(tmp_path / "lake.bin")).replay()) == payloads
    # A REFUSED drain appends nothing and must observe nothing (accounting follows the write).
    small = SignalRing(slots=2)
    j2 = LakeJournaller(small, BinLog(str(tmp_path / "lake2.bin")))
    for p in payloads:
        small.publish(p)
    seen2 = []
    prev = home_mod.set_append_observer(lambda tag, n: seen2.append((tag, n)))
    try:
        with pytest.raises(RingOverrun):
            j2.drain()
    finally:
        home_mod.set_append_observer(prev)
    assert seen2 == [], f"a refused drain observed an append that never happened: {seen2}"


def test_single_writer_is_structural():
    ring = SignalRing(slots=4)
    ring.publish(b"\x7fowner")
    box = {}

    def interloper():
        try:
            ring.publish(b"\x7fsecond-thread")
            box["raised"] = False
        except RuntimeError:
            box["raised"] = True

    t = threading.Thread(target=interloper)
    t.start()
    t.join()
    assert box["raised"], "a second thread published without refusal -- single-writer is prose"


def test_construction_and_payload_validation():
    for bad in (0, 1, -3, 2.5, "8"):
        with pytest.raises(ValueError):
            SignalRing(slots=bad)
    ring = SignalRing(slots=2)
    with pytest.raises(TypeError):
        ring.publish("not-bytes")


# ---- T020: JoinPolicy + DeliveryPolicy enforcement (P8 invention #2's enforcement half) ----
def _in_port(policy, shape=(4,)):
    return Port(
        name=f"d_{policy}",
        region="thalamus",
        shape=shape,
        dtype="f4",
        unit="rate_hz",
        encoding="VEC_DENSE",
        kind="IN",
        declaring_module="global_workspace",
        join_policy=policy,
    )


def test_all_five_joins_combine_correctly():
    ring = SignalRing(slots=64)
    policies = ("sum", "max", "concat", "typed-merge")
    consumers = {p: JoinConsumer(ring, _in_port(p)) for p in policies}
    ring.publish(JoinConsumer.frame(1, "alpha", [1.0, 2.0, 3.0, 4.0]))
    ring.publish(JoinConsumer.frame(1, "beta", [4.0, 3.0, 2.0, 1.0]))
    ((t, s),) = consumers["sum"].poll_joined()[0]
    assert t == 1 and np.allclose(s, [5.0, 5.0, 5.0, 5.0])
    ((_, mx),) = consumers["max"].poll_joined()[0]
    assert np.allclose(mx, [4.0, 3.0, 3.0, 4.0])
    ((_, cat),) = consumers["concat"].poll_joined()[0]
    assert cat.shape == (8,) and np.allclose(cat[:4], [1.0, 2.0, 3.0, 4.0])  # producer-sorted
    ((_, tm),) = consumers["typed-merge"].poll_joined()[0]
    assert set(tm) == {"alpha", "beta"} and np.allclose(tm["beta"], [4.0, 3.0, 2.0, 1.0])


def test_arbitrate_carries_the_margin_record():
    ring = SignalRing(slots=16)
    j = JoinConsumer(ring, _in_port("arbitrate"))
    ring.publish(JoinConsumer.frame(1, "quiet", [1.0, 0.0, 0.0, 0.0]))
    ring.publish(JoinConsumer.frame(1, "loud", [5.0, 5.0, 5.0, 5.0]))
    ((_, winner),) = j.poll_joined()[0]
    assert np.allclose(winner, [5.0, 5.0, 5.0, 5.0])
    assert j.last_margin["winner"] == "loud" and j.last_margin["margin_to_runner_up"] > 0
    # a consumer whose margin never moves is not consuming -- the record must UPDATE
    ring.publish(JoinConsumer.frame(2, "quiet", [9.0, 9.0, 9.0, 9.0]))
    ring.publish(JoinConsumer.frame(2, "loud", [1.0, 0.0, 0.0, 0.0]))
    j.poll_joined()
    assert j.last_margin["winner"] == "quiet", "the margin record froze"


def test_silent_sum_red_policy_stripped_port_refuses():
    good = _in_port("sum")
    stripped = object.__new__(Port)
    for f_name, val in good.__dict__.items():
        object.__setattr__(stripped, f_name, val)
    object.__setattr__(stripped, "join_policy", None)
    with pytest.raises(UndeclaredJoin):
        JoinConsumer(SignalRing(slots=4), stripped)


def test_graded_refuses_unbound_and_scales_bound():
    out = Port(
        name="g_out",
        region="thalamus",
        shape=(4,),
        dtype="f4",
        unit="rate_hz",
        encoding="VEC_DENSE",
        kind="OUT",
        declaring_module="reentrant_workspace",
        delivery_policy=("graded", "reentrant_workspace._assoc"),
    )
    with pytest.raises(GradedUnbound):
        DeliveringProducer(SignalRing(slots=4), "ws", out).publish(1, [1.0, 1.0, 1.0, 1.0])
    ring = SignalRing(slots=8)
    sink = JoinConsumer(ring, _in_port("sum"))
    DeliveringProducer(ring, "ws", out, association_source=lambda: 0.25).publish(
        1, [4.0, 4.0, 4.0, 4.0]
    )
    ((_, v),) = sink.poll_joined()[0]
    assert np.allclose(v, [1.0, 1.0, 1.0, 1.0])
