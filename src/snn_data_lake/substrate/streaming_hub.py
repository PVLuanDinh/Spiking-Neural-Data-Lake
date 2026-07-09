"""
v0.41 — streaming telemetry hub: live append + query on one machine.

spike_telemetry_hub.py is BATCH — it builds the whole store in memory, then `save()` writes
the `.spk` with its channel index at save time. This is the STREAMING counterpart:

  - events are appended to a durable append-only log (`.spkl`) AS THEY FIRE (flushed)
  - an in-memory per-channel index grows on each append
  - a windowed query runs at ANY time over the growing store (record-WHILE-query)
  - restart -> `replay()` rebuilds the index by scanning the log once (crash recovery)

8 bytes/event on disk (channel u32, time u32). The distributed version of this is the cloud
path (Pub/Sub -> Dataflow -> GCS Bronze); this is the single-machine one.

  python streaming_hub.py
"""
import os
import struct
from array import array
from bisect import bisect_left

_REC = struct.Struct("<II")     # (channel, time) = 8 bytes per event


class StreamingHub:
    def __init__(self, path, n_channels, resume=False, flush_every=1):
        """`flush_every` (opt-in group-commit knob; DEFAULT 1 = the original behaviour, unchanged):

          - flush_every == 1 : flush() after EVERY append -> durable-per-event crash recovery
            (at most 0 events lost on a hard crash). This is the default and is byte-for-byte
            identical to the pre-v2.5 hub.
          - flush_every == K > 1 : buffer writes and flush() only once per K appends (and always
            on query_window / hot_motifs / close). Trades per-event durability (up to K-1 recently
            appended events can be lost on a HARD crash before the group flush) for ~K-fold fewer
            flush syscalls -> higher append throughput. The in-memory index still updates on every
            append, so a query is never stale; a clean close() always flushes the buffered tail
            (no data lost on a normal shutdown)."""
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        self.path = path
        self.n = n_channels
        self.flush_every = int(flush_every)
        self._pending = 0                                   # appends written but not yet flushed
        self.ch = [array("I") for _ in range(n_channels)]   # in-memory sorted spike times
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if resume and os.path.exists(path):
            self._replay()
        self._f = open(path, "ab")                          # durable append-only handle

    def _flush_pending(self):
        """Flush buffered appends (no-op when nothing is pending -> the default path never adds a
        flush at a query/close point, keeping flush_every=1 byte-for-byte identical to before)."""
        if self._pending:
            self._f.flush()
            self._pending = 0

    # ---- record (live) ------------------------------------------------------
    def append(self, channel, t):
        """Record one spike now: write to the durable log, update the live index.
        Flush cadence is governed by `flush_every` (default 1 = flush every event)."""
        if not 0 <= channel < self.n:
            raise IndexError(f"channel {channel} out of range [0,{self.n})")
        t = int(t)
        self._f.write(_REC.pack(channel, t))
        self._pending += 1
        if self._pending >= self.flush_every:               # default flush_every=1 -> durable per event
            self._f.flush()
            self._pending = 0
        a = self.ch[channel]
        if a and t < a[-1]:                                 # out-of-order arrival
            a.insert(bisect_left(a, t), t)                  # O(n) insert; spikes are
        else:                                               #   normally monotonic so this is rare
            a.append(t)

    # ---- query (any time, while still recording) ----------------------------
    def query_window(self, t0, t1, channels=None):
        """events with t0 <= t < t1; touches only the requested channels (no full scan)."""
        self._flush_pending()                               # a query never misses buffered events
        out = []
        for c in (channels if channels is not None else range(self.n)):
            a = self.ch[c]
            for k in range(bisect_left(a, t0), bisect_left(a, t1)):
                out.append((a[k], c))
        return out

    def n_events(self):
        return sum(len(a) for a in self.ch)

    # ---- opt-in fixed-memory analytics (read-only) --------------------------
    def hot_motifs(self, window, top_k=3, depth=4, width=256):
        """Top-`top_k` most frequent channel-COACTIVATION motifs (the set of channels firing together within a
        `window`-length time bin) over the recorded stream, via a Count-Min heavy-hitter sketch. Fixed memory
        regardless of how many distinct motifs occur -> the scale-out analytics primitive (the single-machine
        index could exact-count; the sketch is what carries to the cloud path). Read-only: does not touch the
        log, index, or any append/query behaviour. Returns a list of channel frozensets, most frequent first."""
        self._flush_pending()                               # durable up to this analytics read
        from snn_data_lake.substrate.spike_stream_sketch import HeavyHitters
        bins = {}
        for c in range(self.n):
            for t in self.ch[c]:
                bins.setdefault(t // window, set()).add(c)
        hh = HeavyHitters(k=top_k, depth=depth, width=width)
        for b in sorted(bins):
            motif = frozenset(bins[b])
            if motif:
                hh.add(motif)
        return hh.heavy()

    def close(self):
        self._flush_pending()                               # clean close always flushes the tail
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()                                       # release the append handle even on an exception
        return False

    # ---- recovery -----------------------------------------------------------
    def _replay(self):
        """Rebuild the in-memory index from the durable log (called on resume)."""
        with open(self.path, "rb") as f:
            data = f.read()
        for i in range(0, len(data) - _REC.size + 1, _REC.size):
            c, t = _REC.unpack(data[i:i + _REC.size])
            if 0 <= c < self.n:
                self.ch[c].append(t)
        for c in range(self.n):
            a = self.ch[c]
            if any(a[i] > a[i + 1] for i in range(len(a) - 1)):
                self.ch[c] = array("I", sorted(a))


def main():
    path = os.path.join(".", "data", "stream.spkl")
    if os.path.exists(path):
        os.remove(path)
    N, DUR = 16, 10_000
    hub = StreamingHub(path, N)

    # ---- record WHILE querying: two bursts, query between them --------------
    import random
    rng = random.Random(0)
    # burst 1 on ch3 in [1000,2000)
    for t in sorted(rng.sample(range(1000, 2000), 200)):
        hub.append(3, t)
    q1 = hub.query_window(1000, 2000, channels=[3])          # query mid-stream
    # burst 2 on ch3 in [3000,4000) — recorded AFTER the first query
    for t in sorted(rng.sample(range(3000, 4000), 150)):
        hub.append(3, t)
    q2_old = hub.query_window(1000, 2000, channels=[3])      # old window unchanged
    q2_new = hub.query_window(3000, 4000, channels=[3])      # new events now visible
    # background traffic on other channels
    for c in range(N):
        for t in sorted(rng.sample(range(DUR), 30)):
            hub.append(c, t)
    partial = hub.query_window(0, DUR, channels=[3])         # only ch3, not all 16
    q_final = hub.query_window(1000, 2000, channels=[3])     # final state of that window
    disk_bytes = os.path.getsize(path)
    total = hub.n_events()
    hub.close()

    # ---- opt-in analytics: a recurring channel-coactivation motif surfaces as a hot motif -----
    motif_path = os.path.join(".", "data", "motif.spkl")
    if os.path.exists(motif_path):
        os.remove(motif_path)
    mhub = StreamingHub(motif_path, N)
    W = 10
    tt = 0
    for _ in range(40):                                      # channels {1,4,7} co-fire, 40 times, distinct bins
        for c in (1, 4, 7):
            mhub.append(c, tt + rng.randint(0, W - 1))
        tt += W
    for _ in range(300):                                    # background: lone random channels in later bins
        mhub.append(rng.randrange(N), tt + rng.randint(0, W - 1))
        tt += W
    hot = mhub.hot_motifs(window=W, top_k=3)
    mhub.close()

    # ---- restart: replay the durable log -----------------------------------
    hub2 = StreamingHub(path, N, resume=True)
    replayed = hub2.n_events()
    q_after = hub2.query_window(1000, 2000, channels=[3])
    hub2.close()

    print("=" * 60)
    print("STREAMING HUB  (live append + record-while-query)")
    print("=" * 60)
    print(f"recorded     : {total:,} events -> {disk_bytes:,} B durable log (8 B/event)")
    print(f"mid-stream q : window [1000,2000) ch3 -> {len(q1)} events (during recording)")
    print(f"after burst2 : same window still {len(q2_old)}; new window [3000,4000) -> {len(q2_new)}")
    print(f"partial query: ch3 over full range -> {len(partial)} events (touched 1 of {N} channels)")
    print(f"hot motifs   : top coactivation {sorted(hot[0])} (recurring {{1,4,7}}) via fixed-memory sketch")
    print(f"replay       : restart from log -> {replayed:,} events, window q -> {len(q_after)}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert len(q1) == 200, f"mid-stream query wrong: {len(q1)}"
    assert len(q2_old) == 200, "earlier window changed after later appends"
    assert len(q2_new) == 150, "events recorded after a query are not visible"
    assert len(partial) == 200 + 150 + 30, "partial query miscounted ch3"
    assert disk_bytes == total * _REC.size, "durable log size != events x 8"
    assert replayed == total and q_after == q_final, "replay did not reconstruct the store"
    assert hot and set(hot[0]) == {1, 4, 7}, f"hot_motifs must surface the recurring coactivation motif ({hot})"

    # ---- v2.5 opt-in group-commit: default byte-identical; K>1 loses nothing on query/close ----
    gc_events = [(rng.randrange(N), rng.randrange(DUR)) for _ in range(500)]
    p1 = os.path.join(".", "data", "gc_default.spkl")       # flush_every=1 (current behaviour)
    pk = os.path.join(".", "data", "gc_group.spkl")         # flush_every=K (opt-in group-commit)
    for p in (p1, pk):
        if os.path.exists(p):
            os.remove(p)
    h1 = StreamingHub(p1, N)                                 # DEFAULT: flush every event
    hk = StreamingHub(pk, N, flush_every=32)                 # K=32: buffer, flush every 32
    for c, t in gc_events:
        h1.append(c, t)
        hk.append(c, t)
    q1 = h1.query_window(0, DUR)                             # query flushes hk's buffered tail
    qk = hk.query_window(0, DUR)                             # ...so K>1 misses nothing on a query
    tail_pending = len(gc_events) % 32                       # 500 % 32 = 20 buffered before close
    h1.close(); hk.close()                                   # clean close flushes hk's last group
    with open(p1, "rb") as f:
        b1 = f.read()
    with open(pk, "rb") as f:
        bk = f.read()
    hk2 = StreamingHub(pk, N, resume=True)                   # reload K-hub log from disk
    replayed_k = hk2.n_events()
    hk2.close()
    expected = sorted((t, c) for c, t in gc_events)
    print(f"group-commit : default {len(b1):,}B vs K=32 {len(bk):,}B "
          f"(byte-identical={b1 == bk}); {tail_pending}-event tail flushed on close; "
          f"query saw all {len(qk)} events")

    assert len(b1) == len(gc_events) * _REC.size, "default (flush_every=1) on-disk log size drifted"
    assert b1 == bk, "group-commit changed the on-disk bytes vs flush-every-event default"
    assert sorted(q1) == sorted(qk) == expected, "group-commit dropped/reordered events in a query"
    assert replayed_k == len(gc_events), "clean close did not flush the buffered tail to the durable log"
    print("self-check OK: record-while-query works, queries are partial + live, "
          "durable log replays exactly; hot_motifs surfaces the recurring coactivation motif; "
          "group-commit default is byte-identical and flush_every=K loses nothing on query/close")


if __name__ == "__main__":
    main()
