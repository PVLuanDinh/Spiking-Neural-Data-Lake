"""SignalRing + LakeJournaller (specs/022 T011) -- the faculty ring's three contract points.

Single writer is ENFORCED (a second thread's publish refuses), multicast consumers see the same
bytes through independent cursors, and loss is LOUD: an overrun poll reports exactly what
vanished, and the lake journaller refuses to journal around a hole -- before consuming anything,
so refusal can never manufacture the very gap it refuses (P4b: a gapped lake is amnesia).
The charge proof is the replay A/B: the same produced stream lands byte-equal through the
journaller and not at all through the neutered arm.
"""

import threading

import pytest

from snn_data_lake.substrate.binlog import BinLog
from snn_data_lake.substrate.signal_ring import LakeJournaller, RingOverrun, SignalRing


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
