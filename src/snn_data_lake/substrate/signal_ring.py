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
    # distinct-from: hippocampus/lake_consolidation.py -- prunes fact history, no transport.
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
        return self._log.extend([payload for _seq, payload in records])


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
        "publish refused (single-writer structural)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
