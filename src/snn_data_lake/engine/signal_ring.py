"""SignalRing -- the faculty ring: ONE writer, multicast consumers, pre-allocated slots
(specs/022 T011), plus the LakeJournaller that makes the durable-lake writer a CONSUMER of it.

WHY A RING, and why the lake sits on it as a consumer rather than being called inline: P4b says
the lake is part of the brain, and today every faculty that persists calls the lake DIRECTLY from
inside its own tick -- persistence cost lands on the producing faculty, and a second consumer of
the same event (telemetry, the append budget) means a second call site that can drift. On the
ring, a faculty PUBLISHES once; the journaller drains to the ONE BinLog on its own cadence, and
any other consumer reads the SAME bytes through its own cursor. Single-writer stays structural on
both ends: one publisher (enforced, see below) and one durable-lake appender (the journaller).

THE THREE CONTRACT POINTS, each with a red in main():
  * SINGLE WRITER, enforced not documented: the first `publish` binds the ring to that thread;
    a publish from any other thread raises. A multi-producer ring is a different data structure
    with different loss semantics, and quietly becoming one is how interleaved frames corrupt a
    replay.
  * MULTICAST: every consumer owns its cursor and sees the SAME byte stream in the SAME order.
  * LOSS IS LOUD: slots are pre-allocated and the writer NEVER blocks -- a consumer that falls
    more than `slots` behind is overrun, and `poll()` reports exactly how many records it lost.
    A ring that silently resumes from the oldest surviving slot hands its consumer a
    coherent-looking stream with a hole in it, which for the lake consumer is P4b amnesia with
    green telemetry. The journaller therefore REFUSES on loss (raises) rather than journaling
    around a hole.

Slot payloads are opaque bytes: framing/tags belong to the producer and the lake, never to the
transport. The slot ARRAY is allocated once at construction (pre-allocated slots == bounded
memory, the writer overwrites in place); payload buffers are the producer's own bytes objects.

distinct-from: substrate/streaming_hub.py -- the .spkl spike-channel store (a different lake,
  persistent); this is an in-process transport with no durability of its own.
distinct-from: substrate/spike_telemetry_hub.py -- fan-out of derived TELEMETRY to sockets;
  this carries lake-bound records and enforces loss-loudness instead of best-effort delivery.
distinct-from: substrate/lake_working_memory.py -- reserved-channel working memory over the
  .spkl store. distinct-from: hippocampus ring/replay buffers -- learned content, not transport.
All sites read.

Charge (replay A/B, run in main() and tests/test_signal_ring.py): the SAME event stream produced
WITH the journaller attached lands byte-equal in a scratch BinLog, oldest first; produced with the
journaller NEUTERED (detached), the lake holds ZERO of them. The difference is the wire's charge,
constructed the way tools/charge_probe.py constructs its neuter A/B.

Live-path status, stated honestly (the lake_* EXEMPT corrections of 0d0a7745 are the cautionary
tale): TODAY nothing on the tick publishes here. The live producer is the two-phase kernel tick
(T013+) and the daemon journaller cadence (T012); the dormancy exemption names that expiry, and
if those land WITHOUT adopting this module the exemption gets corrected the same way, not
quietly extended.
"""

from __future__ import annotations

import sys
import threading

from snn_data_lake.substrate.home import observe_append


class RingOverrun(RuntimeError):
    """A consumer lost records (writer lapped it). For the lake consumer this is never survivable
    silence -- the journaller raises this rather than journaling around a hole (P4b)."""


class SignalRing:
    """ONE-writer multicast ring over `slots` pre-allocated positions."""

    def __init__(self, slots):
        if not isinstance(slots, int) or slots < 2:
            raise ValueError("slots must be an int >= 2 (a 1-slot ring cannot hold head and tail)")
        self._slots = slots
        self._buf = [None] * slots  # allocated ONCE; publish overwrites in place
        self._head = 0  # next seq to write; seq k lives at _buf[k % slots]
        self._writer_ident: int | None = None

    @property
    def head(self):
        return self._head

    @property
    def slots(self):
        return self._slots

    def publish(self, payload):
        """Append one opaque record; returns its seq. Never blocks, never drops ITS OWN write --
        the slot it overwrites is the oldest, and any consumer still needing that slot discovers
        the loss at its next poll(). Binds the ring to the FIRST publishing thread."""
        ident = threading.get_ident()
        if self._writer_ident is None:
            self._writer_ident = ident
        elif ident != self._writer_ident:
            raise RuntimeError(
                "SignalRing is single-writer: publish() from a second thread refused -- a "
                "multi-producer ring has different loss semantics and must be chosen on purpose"
            )
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes (framing belongs to the producer, not the ring)")
        seq = self._head
        self._buf[seq % self._slots] = bytes(payload)
        self._head = seq + 1
        return seq

    def attach(self, name):
        """A new consumer cursored at the CURRENT head (it sees what is published after it
        attaches; history it never saw is not loss)."""
        return _Consumer(self, str(name), self._head)


class _Consumer:
    """One cursor over the ring. poll() -> (records, lost): records oldest-first from the cursor,
    lost = how many the writer overwrote before this consumer reached them. Loss is REPORTED,
    never smoothed -- the caller decides whether a hole is survivable (telemetry) or fatal (lake)."""

    def __init__(self, ring, name, cursor):
        self._ring = ring
        self.name = name
        self._cursor = cursor

    @property
    def lag(self):
        return self._ring.head - self._cursor

    def poll(self, max_n=None):
        ring = self._ring
        head = ring.head
        oldest = max(0, head - ring.slots)
        lost = 0
        if self._cursor < oldest:
            lost = oldest - self._cursor
            self._cursor = oldest
        stop = head if max_n is None else min(head, self._cursor + int(max_n))
        out = []
        while self._cursor < stop:
            out.append((self._cursor, ring._buf[self._cursor % ring.slots]))
            self._cursor += 1
        return out, lost


class LakeJournaller:
    """THE durable-lake writer, as a ring consumer: drains published records into one BinLog.

    Loss is fatal here by contract: a journaller that skipped a hole would republish a stream
    that LOOKS complete, which is exactly the P4b failure ('retained() said 1.000 while 0
    persisted'). On overrun it raises RingOverrun BEFORE appending anything from the poll, so
    the lake never holds a silently-gapped suffix."""

    def __init__(self, ring, log):
        self._ring = ring
        self._consumer = ring.attach("lake-journaller")
        self._log = log

    @property
    def lag(self):
        return self._consumer.lag

    # REUSE, not a second bulk path: the drain hands its batch to BinLog.extend (ONE fsync for
    # the whole batch -- the existing bulk-ingest mechanism), never a per-record append loop.
    # distinct-from: substrate/binlog.py and substrate/home.py -- they ARE the append mechanisms
    #   this consumer calls; this class only decides WHAT reaches them and refuses holes.
    # distinct-from: sleep/lake_consolidation.py -- prunes fact history, no transport.
    def drain(self, max_n=None):
        """Append pending records to the lake, oldest first. Returns how many were appended.

        The overrun check runs BEFORE the poll, deliberately: poll() consumes (advances the
        cursor), so a check-after-consume would refuse the batch and then SKIP it on the next
        drain -- the journaller manufacturing exactly the hole it exists to refuse. A refused
        hole also does not heal: every subsequent drain raises again until the caller decides,
        loudly, what to do about the loss."""
        lost = self._consumer.lag - self._ring.slots
        if lost > 0:
            raise RingOverrun(
                f"lake journaller lost {lost} record(s) to overrun -- refusing to journal around "
                "the hole; size the ring or drain more often (P4b: a gapped lake is amnesia)"
            )
        records, lost_now = self._consumer.poll(max_n)
        if lost_now:  # unreachable single-threaded (pre-checked); real under a concurrent
            raise RingOverrun(  # publisher (TOCTOU), so the second gate stays
                f"lake journaller lost {lost_now} record(s) DURING the drain -- a concurrent "
                "publisher lapped the poll; the polled batch was not appended"
            )
        if not records:
            return 0
        payloads = [payload for _seq, payload in records]
        n = self._log.extend(payloads)
        # The ONE append-observation path (home.observe_append, specs/022 T067), AFTER the durable
        # write: this drain is the only writer of the workspace coalition class (0x0A), and until
        # 2026-09-08 it wrote around the observer, so the append budget's census could never see
        # the tag that is most of the operator's lake by bytes. Each payload is tag-prefixed as the
        # producer framed it, which is exactly the length the observer is documented to take.
        for payload in payloads:
            observe_append(payload[:1], len(payload))
        return n


# ---- T020: JoinPolicy + DeliveryPolicy ENFORCEMENT on the delivery path (P8 invention #2,
# ---- docs/INVENTION_REGISTER.md row 2 -- the register indexes this, the reds live here) ----
# distinct-from: engine/wiring_manifest.py -- DECLARES the policies; this ENFORCES them
#   where payloads actually combine. distinct-from: faculties/global_workspace.py -- workspace
#   COMPETITION among coalitions; this is typed fan-in combination at a port. All sites read.
class UndeclaredJoin(RuntimeError):
    """Two producers reached one consumer with no declared join -- the silent-sum defect,
    REFUSED instead of summed. Both major neuromorphic IRs default to elementwise summation
    here, and that default is the measured one-scalar collapse mechanism (invention #2's
    register row); this exception is the enforcement half of the manifest red."""


class GradedUnbound(RuntimeError):
    """A graded OUT port delivered before its association source was bound. The DECLARED
    incumbent source is `reentrant_workspace._assoc`; the FINAL runtime binding lands with
    T030's workspace instance resolution -- until then a graded port without a live source
    REFUSES rather than silently delivering all-or-none (a policy downgrade nobody chose)."""


class JoinConsumer:
    """A ring consumer that enforces a declared IN port's join_policy over same-tick payloads.

    Producers publish framed records ([tick u64][nlen u8][producer utf8][float32 vector]);
    poll_joined() groups them by tick and combines each group under the port's policy:
      sum / max     elementwise, accumulated in float64 over PRODUCER-SORTED order (sorted so
                    the float sum is deterministic; disclosed, not accidental)
      concat        axis-0 concatenation in producer-sorted order
      typed-merge   {producer: vector} -- contributions kept apart by name, no arithmetic
      arbitrate     ONE winner by largest L2 norm (ties break by producer name, deterministic)
                    and every decision carries the contract's cleanup-margin record
                    {winner, margin_to_runner_up} via last_margin -- a consumer whose margin
                    never moves is not consuming (the charge contract's own tell)
    A port with kind != IN or a policy outside the declared set REFUSES AT CONSTRUCTION
    (UndeclaredJoin) -- the strict=False deserialization path is the only door such a port
    can arrive through, and this is the door closing."""

    def __init__(self, ring, port):
        from snn_data_lake.engine.wiring_manifest import JOIN_POLICIES

        if (
            getattr(port, "kind", None) != "IN"
            or getattr(port, "join_policy", None) not in JOIN_POLICIES
        ):
            raise UndeclaredJoin(
                f"port {getattr(port, 'name', port)!r}: kind={getattr(port, 'kind', None)!r} "
                f"join_policy={getattr(port, 'join_policy', None)!r} -- a fan-in with no "
                "declared join is REFUSED, never summed"
            )
        self.port = port
        self._consumer = ring.attach(f"join:{port.name}")
        self.last_margin = None  # {"winner": str, "margin_to_runner_up": float} after arbitrate

    @staticmethod
    def frame(tick, producer, vector) -> bytes:
        import struct as _struct

        import numpy as np

        raw = str(producer).encode("utf-8")
        return (
            _struct.pack("<QB", int(tick), len(raw))
            + raw
            + np.asarray(vector, dtype=np.float32).tobytes()
        )

    @staticmethod
    def _unframe(payload):
        import struct as _struct

        import numpy as np

        tick, nlen = _struct.unpack_from("<QB", payload, 0)
        producer = payload[9 : 9 + nlen].decode("utf-8")
        vec = np.frombuffer(payload[9 + nlen :], dtype=np.float32)
        return tick, producer, vec

    def poll_joined(self):
        """-> list of (tick, combined) oldest-first; loss propagates from the underlying
        consumer as (records, lost) does -- a lost producer contribution is REPORTED, and the
        join runs over what survived (stated, never silent)."""
        import numpy as np

        records, lost = self._consumer.poll()
        by_tick: dict = {}
        for _seq, payload in records:
            tick, producer, vec = self._unframe(payload)
            by_tick.setdefault(tick, []).append((producer, vec))
        out = []
        for tick in sorted(by_tick):
            group = sorted(by_tick[tick], key=lambda pv: pv[0])
            policy = self.port.join_policy
            if policy == "sum":
                combined = sum((v.astype(np.float64) for _p, v in group), start=np.float64(0.0))
            elif policy == "max":
                combined = np.max(np.stack([v for _p, v in group]), axis=0)
            elif policy == "concat":
                combined = np.concatenate([v for _p, v in group], axis=0)
            elif policy == "typed-merge":
                combined = {p: v for p, v in group}
            else:  # arbitrate
                norms = sorted(
                    ((float(np.linalg.norm(v)), p, v) for p, v in group),
                    key=lambda t: (-t[0], t[1]),
                )
                combined = norms[0][2]
                self.last_margin = {
                    "winner": norms[0][1],
                    "margin_to_runner_up": norms[0][0] - (norms[1][0] if len(norms) > 1 else 0.0),
                }
            out.append((tick, combined))
        return out, lost


class DeliveringProducer:
    """The OUT side: applies the declared delivery_policy at publish time.

    all-or-none publishes the whole vector. ('graded', source_name) SCALES the vector by the
    bound association weight in [0, 1] -- and REFUSES (GradedUnbound) while unbound:
    silently downgrading graded to all-or-none would be a policy nobody declared. THE
    binding landed at T030: `reentrant_workspace.association_level` is the incumbent
    source -- a zero-argument bound method returning the learned association support of
    the last ignited winner (0.0 while the workspace is dark), satisfying the
    `w = float(self._assoc())` contract below; the self-check proves the graded scale
    against that REAL source, not a lambda."""

    def __init__(self, ring, faculty, port, association_source=None):
        if getattr(port, "kind", None) != "OUT":
            raise ValueError(f"DeliveringProducer needs an OUT port, got {port.kind!r}")
        self.ring, self.faculty, self.port = ring, str(faculty), port
        self._assoc = association_source

    def publish(self, tick, vector):
        import numpy as np

        vec = np.asarray(vector, dtype=np.float32)
        if self.port.delivery_policy != "all-or-none":
            if self._assoc is None:
                raise GradedUnbound(
                    f"OUT port {self.port.name!r} declares "
                    f"{self.port.delivery_policy!r} but no association source is bound "
                    "(THE source: reentrant_workspace.association_level, bound at T030)"
                )
            w = float(self._assoc())
            vec = vec * min(1.0, max(0.0, w))
        return self.ring.publish(JoinConsumer.frame(tick, self.faculty, vec))


def main() -> int:
    import tempfile
    from pathlib import Path

    from snn_data_lake.substrate.binlog import BinLog

    work = Path(tempfile.mkdtemp(prefix="signal_ring_"))

    # ---- replay A/B: the journaller wire IS the difference (constructed neuter A/B) ----
    produced = [bytes([0x7F]) + f"ring-record-{k}".encode("ascii") for k in range(16)]
    ring = SignalRing(slots=32)
    journal = LakeJournaller(ring, BinLog(str(work / "arm_a.bin")))
    telemetry = ring.attach("telemetry-twin")  # multicast: a second consumer of the same bytes
    for p in produced:
        ring.publish(p)
    n = journal.drain()
    got = list(BinLog(str(work / "arm_a.bin")).replay())
    if n != 16 or got != produced:
        print(f"signal_ring self-check FAILED: arm A journaled {n}, replay equal={got == produced}")
        return 1
    seen, lost = telemetry.poll()
    if lost or [p for _s, p in seen] != produced:
        print("signal_ring self-check FAILED: multicast consumer saw a different stream")
        return 1
    # Arm B: same producer, journaller NEUTERED (never attached) -> the lake holds ZERO.
    ring_b = SignalRing(slots=32)
    for p in produced:
        ring_b.publish(p)
    got_b = list(BinLog(str(work / "arm_b.bin")).replay())
    if got_b:
        print("signal_ring self-check FAILED: the neutered arm reached the lake somehow")
        return 1

    # ---- loss is LOUD: overrun reports exactly what vanished, journaller refuses ----
    small = SignalRing(slots=4)
    lagger = small.attach("lagger")
    for k in range(9):  # 9 publishes over 4 slots: the 5 oldest are gone
        small.publish(bytes([k]))
    recs, lost = lagger.poll()
    if lost != 5 or [p for _s, p in recs] != [bytes([k]) for k in range(5, 9)]:
        print(f"signal_ring self-check FAILED: overrun reported lost={lost}, {len(recs)} records")
        return 1
    small2 = SignalRing(slots=4)
    j2 = LakeJournaller(small2, BinLog(str(work / "arm_c.bin")))
    for k in range(6):
        small2.publish(bytes([k]))
    try:
        j2.drain()
        print("signal_ring self-check FAILED: journaller journaled around a hole (P4b)")
        return 1
    except RingOverrun:
        pass
    if list(BinLog(str(work / "arm_c.bin")).replay()):
        print("signal_ring self-check FAILED: a refused drain still appended something")
        return 1

    # ---- T020: join enforcement, the silent-sum red, graded-unbound red, margin record ----
    import numpy as np

    from snn_data_lake.engine.wiring_manifest import Port

    def _in_port(policy):
        return Port(
            name=f"d_{policy}",
            region="thalamus",
            shape=(4,),
            dtype="f4",
            unit="rate_hz",
            encoding="VEC_DENSE",
            kind="IN",
            declaring_module="global_workspace",
            join_policy=policy,
        )

    ring_j = SignalRing(slots=64)
    jsum = JoinConsumer(ring_j, _in_port("sum"))
    jarb = JoinConsumer(ring_j, _in_port("arbitrate"))
    ring_j.publish(JoinConsumer.frame(1, "alpha", [1.0, 2.0, 3.0, 4.0]))
    ring_j.publish(JoinConsumer.frame(1, "beta", [10.0, 20.0, 30.0, 40.0]))
    joined, lost_j = jsum.poll_joined()
    if lost_j or len(joined) != 1 or not np.allclose(joined[0][1], [11.0, 22.0, 33.0, 44.0]):
        print(f"signal_ring self-check FAILED: sum join wrong ({joined})")
        return 1
    ring_j.publish(JoinConsumer.frame(2, "alpha", [1.0, 0.0, 0.0, 0.0]))
    ring_j.publish(JoinConsumer.frame(2, "beta", [9.0, 9.0, 9.0, 9.0]))
    _ = jarb.poll_joined()  # one poll drains both ticks; last_margin holds tick 2's decision
    m = jarb.last_margin
    if not m or m["winner"] != "beta" or m["margin_to_runner_up"] <= 0:
        print(f"signal_ring self-check FAILED: arbitrate margin record wrong ({m})")
        return 1
    # the silent-sum red: a policy-stripped IN port (the strict=False door) REFUSES
    stripped = object.__new__(Port)
    for f_name, val in _in_port("sum").__dict__.items():
        object.__setattr__(stripped, f_name, val)
    object.__setattr__(stripped, "join_policy", None)
    try:
        JoinConsumer(ring_j, stripped)
        print("signal_ring self-check FAILED: an undeclared join was ACCEPTED (silent sum)")
        return 1
    except UndeclaredJoin:
        pass
    # graded-unbound red + the bound green control in the same arm
    out_graded = Port(
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
    try:
        DeliveringProducer(SignalRing(slots=4), "ws", out_graded).publish(1, [1, 1, 1, 1])
        print("signal_ring self-check FAILED: graded delivered while UNBOUND")
        return 1
    except GradedUnbound:
        pass
    ring_g = SignalRing(slots=8)
    gsink = JoinConsumer(ring_g, _in_port("sum"))
    DeliveringProducer(ring_g, "ws", out_graded, association_source=lambda: 0.5).publish(
        3, [2.0, 2.0, 2.0, 2.0]
    )
    gj, _ = gsink.poll_joined()
    if not np.allclose(gj[0][1], [1.0, 1.0, 1.0, 1.0]):
        print(f"signal_ring self-check FAILED: graded scaling wrong ({gj})")
        return 1

    # (T030: the graded binding against THE REAL SOURCE -- both regimes, dark->0 and
    # ignited->learned-weight -- is proven in reentrant_workspace's OWN self-check, where
    # the source lives: a workspace construction here would register this substrate module
    # as a private faculty bus in the p6 walk, which is exactly the census it must not
    # pollute. This arm keeps the generic contract reds: unbound refusal + lambda scale.)

    # ---- single-writer is enforced, not documented ----
    box = {}

    def _second_thread():
        try:
            ring.publish(b"\x7finterloper")
            box["raised"] = False
        except RuntimeError:
            box["raised"] = True

    t = threading.Thread(target=_second_thread)
    t.start()
    t.join()
    if not box.get("raised"):
        print("signal_ring self-check FAILED: a second thread published without refusal")
        return 1

    print(
        "signal_ring self-check OK: replay A/B charged (16 records byte-equal through the "
        "journaller, ZERO through the neutered arm); multicast twin saw the same stream; "
        "overrun reported 5 lost of 9 and the journaller REFUSED the hole; second-thread "
        "publish refused (single-writer structural); T020 joins ENFORCED -- sum exact, "
        "arbitrate carried its {winner, margin} record, the policy-stripped port REFUSED "
        "(silent-sum red), graded refused UNBOUND and scaled correctly bound"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
