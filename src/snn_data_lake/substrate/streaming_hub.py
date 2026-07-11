"""
v0.41 — streaming telemetry hub: live append + query on one machine.

spike_telemetry_hub.py is BATCH — it builds the whole store in memory, then `save()` writes
the `.spk` with its channel index at save time. This is the STREAMING counterpart:

  - events are appended to a durable append-only log (`.spkl`) AS THEY FIRE (flushed)
  - a BOUNDED resident structure tracks each channel (a recent tail + a coarse on-disk index)
  - a windowed query runs at ANY time over the growing store (record-WHILE-query)
  - restart -> `_replay()` rebuilds that bounded structure by scanning the log once (crash recovery)

8 bytes/event on disk (channel u32, time u32). The distributed version of this is the cloud
path (Pub/Sub -> Dataflow -> GCS Bronze); this is the single-machine one.

OUT-OF-CORE (so it scales as the lake GROWS): the hub never holds O(total-events) resident and
never slurps the whole `.spkl`. Per channel it keeps only (a) a RECENT TAIL of the current
sub-K block in RAM (bounded by K events/channel) and (b) a COARSE offset index -- one
(time, byte-offset) sample every K events, size O(total/K). A recent-window query is answered
straight from the resident tail (no I/O); an older range binary-searches the coarse index and
SEEKS the log, scanning only the overlapping block(s) in bounded chunks -- reusing the
seek-per-channel indexed-reader idea from spike_telemetry_hub.disk_query. Small lakes (< K
events/channel) live entirely in the tail, so nothing regresses for them.

  python streaming_hub.py
"""
import os
import struct
from array import array
from bisect import bisect_left, bisect_right

_REC = struct.Struct("<II")     # (channel, time) = 8 bytes per event
_INDEX_STRIDE = 1024            # coarse offset index: one (time, offset) sample every K events/channel
_SCAN_CHUNK = 1 << 20          # bytes read per disk block while scanning (bounds resident memory)
_MAX_T = 1 << 32               # exclusive upper bound over all u32 spike times (whole-lake scan)


class _ChannelStream:
    """Lazy, out-of-core view of ONE channel's spike times in ascending order.

    Preserves the historical `hub.ch[c]` read surface (iterate / len / index) WITHOUT keeping
    every channel resident: the channel's events are materialized transiently from the durable
    log (the resident tail + a chunked disk scan of its blocks) only when actually read, then
    discarded. A small channel never touches disk -- it lives entirely in the resident tail."""

    __slots__ = ("_hub", "_c", "_cache")

    def __init__(self, hub, c):
        self._hub, self._c, self._cache = hub, c, None

    def _materialize(self):
        if self._cache is None:
            ev = self._hub.query_window(0, _MAX_T, channels=[self._c])
            self._cache = sorted(t for t, _ in ev)          # ascending times, like the old in-RAM index
        return self._cache

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self):
        return self._hub._count[self._c]                    # O(1): resident per-channel counter

    def __getitem__(self, k):
        return self._materialize()[k]


class _ChannelViews:
    """Back-compat stand-in for the old `hub.ch` list: `hub.ch[c]` -> a lazy `_ChannelStream`."""

    __slots__ = ("_hub",)

    def __init__(self, hub):
        self._hub = hub

    def __getitem__(self, c):
        if not 0 <= c < self._hub.n:
            raise IndexError(f"channel {c} out of range [0,{self._hub.n})")
        return _ChannelStream(self._hub, c)

    def __len__(self):
        return self._hub.n


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
        self._K = _INDEX_STRIDE                             # coarse-index / tail block size
        self._nrec = 0                                      # total records on disk (== events appended)
        self._count = [0] * n_channels                      # events per channel (O(1) len())
        self._itimes = [array("I") for _ in range(n_channels)]  # coarse index: per-block start TIME
        self._ioff = [array("Q") for _ in range(n_channels)]    # coarse index: per-block start OFFSET
        self._tail = [[] for _ in range(n_channels)]        # current (< K) block per channel, RESIDENT
        self._tail_off = [0] * n_channels                   # byte offset where the current block begins
        self._last_t = [-1] * n_channels                    # last time appended (monotonicity check)
        self._mono = [True] * n_channels                    # channel's on-disk times are non-decreasing
        self._last_query_bytes = 0                          # bytes read by the most recent query (diag)
        self.ch = _ChannelViews(self)                       # lazy out-of-core per-channel read view
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

    def _record_index(self, channel, t, off):
        """Update the BOUNDED resident index for one on-disk record (shared by append + _replay,
        so the reconstructed state is identical whether written live or replayed from disk)."""
        cnt = self._count[channel]
        if cnt % self._K == 0:                              # this event starts a new K-block -> checkpoint
            self._itimes[channel].append(t)
            self._ioff[channel].append(off)
            self._tail[channel] = []                        # prior block is durable + seekable; drop RAM
            self._tail_off[channel] = off
        self._tail[channel].append(t)
        self._count[channel] = cnt + 1
        lt = self._last_t[channel]
        if lt >= 0 and t < lt:                              # out-of-order arrival on this channel
            self._mono[channel] = False
        self._last_t[channel] = t

    # ---- record (live) ------------------------------------------------------
    def append(self, channel, t):
        """Record one spike now: write to the durable log, update the bounded live index.
        Flush cadence is governed by `flush_every` (default 1 = flush every event)."""
        if not 0 <= channel < self.n:
            raise IndexError(f"channel {channel} out of range [0,{self.n})")
        t = int(t)
        off = self._nrec * _REC.size                        # byte offset of the record about to be written
        self._f.write(_REC.pack(channel, t))
        self._nrec += 1
        self._pending += 1
        if self._pending >= self.flush_every:               # default flush_every=1 -> durable per event
            self._f.flush()
            self._pending = 0
        self._record_index(channel, t, off)

    # ---- disk scan (bounded, seek-per-channel) ------------------------------
    def _scan(self, c, t0, t1, start, end, out):
        """Read the byte range [start, end) of the durable log in bounded chunks, appending the
        channel-c events inside [t0, t1) to `out`. Returns bytes read -- a block, never the file."""
        read = 0
        with open(self.path, "rb") as f:
            f.seek(start)
            pos = start
            while pos < end:
                buf = f.read(min(_SCAN_CHUNK, end - pos))
                if not buf:
                    break
                m = (len(buf) // _REC.size) * _REC.size
                for i in range(0, m, _REC.size):
                    cc, tt = _REC.unpack_from(buf, i)
                    if cc == c and t0 <= tt < t1:
                        out.append((tt, cc))
                read += m
                pos += m
        return read

    # ---- query (any time, while still recording) ----------------------------
    def query_window(self, t0, t1, channels=None):
        """events with t0 <= t < t1; touches only the requested channels (no full scan).

        Recent windows are served from the resident tail (no I/O). Older ranges binary-search the
        coarse per-channel offset index and SEEK the log, scanning only the overlapping block(s)."""
        self._flush_pending()                               # a query never misses buffered events
        t0 = int(t0)
        t1 = int(t1)
        out = []
        self._last_query_bytes = 0
        for c in (channels if channels is not None else range(self.n)):
            itimes = self._itimes[c]
            ioff = self._ioff[c]
            j = len(itimes)                                 # index entries; entry j-1 is the current block
            if j >= 2:                                      # completed blocks 0..j-2 live on disk
                if self._mono[c]:                           # keys sorted -> seek only overlapping blocks
                    p = bisect_right(itimes, t0)
                    q = bisect_left(itimes, t1)
                    j0 = p - 1 if p > 0 else 0
                    j1 = q - 1
                    if j1 > j - 2:
                        j1 = j - 2
                    if j0 <= j1:
                        self._last_query_bytes += self._scan(c, t0, t1, ioff[j0], ioff[j1 + 1], out)
                else:                                       # rare out-of-order channel: scan its blocks
                    self._last_query_bytes += self._scan(c, t0, t1, ioff[0], ioff[j - 1], out)
            for tt in self._tail[c]:                        # current block (resident): recent fast path
                if t0 <= tt < t1:
                    out.append((tt, c))
        return out

    def n_events(self):
        return sum(self._count)

    # ---- opt-in fixed-memory analytics (read-only) --------------------------
    def hot_motifs(self, window, top_k=3, depth=4, width=256):
        """Top-`top_k` most frequent channel-COACTIVATION motifs (the set of channels firing together within a
        `window`-length time bin) over the recorded stream, via a Count-Min heavy-hitter sketch. Fixed memory
        regardless of how many distinct motifs occur -> the scale-out analytics primitive (the single-machine
        index could exact-count; the sketch is what carries to the cloud path). Read-only: streams the durable
        log in bounded chunks (does not slurp it) and touches no append/query state. Returns a list of channel
        frozensets, most frequent first."""
        self._flush_pending()                               # durable up to this analytics read
        from snn_data_lake.substrate.spike_stream_sketch import HeavyHitters
        bins = {}
        end = self._nrec * _REC.size
        with open(self.path, "rb") as f:
            pos = 0
            while pos < end:
                buf = f.read(min(_SCAN_CHUNK, end - pos))
                if not buf:
                    break
                m = (len(buf) // _REC.size) * _REC.size
                for i in range(0, m, _REC.size):
                    c, t = _REC.unpack_from(buf, i)
                    if 0 <= c < self.n:
                        bins.setdefault(t // window, set()).add(c)
                pos += m
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
        """Rebuild the bounded resident index from the durable log (called on resume).

        Scans the log in bounded chunks (NEVER `f.read()` of the whole file) and replays records
        in append order through the SAME `_record_index` path used live, so the reconstructed tail,
        coarse index and monotonicity flags -- and therefore all query results -- are identical."""
        size = os.path.getsize(self.path)
        nrec = size // _REC.size
        end = nrec * _REC.size
        with open(self.path, "rb") as f:
            pos = 0
            while pos < end:
                buf = f.read(min(_SCAN_CHUNK, end - pos))
                if not buf:
                    break
                m = (len(buf) // _REC.size) * _REC.size
                for i in range(0, m, _REC.size):
                    c, t = _REC.unpack_from(buf, i)
                    if 0 <= c < self.n:
                        self._record_index(c, t, pos + i)
                pos += m
        self._nrec = nrec


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

    # ---- OUT-OF-CORE: bounded resident memory as the lake GROWS -------------
    # Write a large MONOTONIC stream; show resident structures stay bounded (NOT O(total events))
    # while an OLD-range query SEEKS one coarse block off disk and a small lake stays fully in-RAM.
    def _ooc_probe(total_events, n_ch):
        p = os.path.join(".", "data", f"ooc_{total_events}.spkl")
        if os.path.exists(p):
            os.remove(p)
        h = StreamingHub(p, n_ch, flush_every=256)
        for tk in range(total_events):
            h.append(tk % n_ch, tk)                          # round-robin -> each channel monotonic
        recent = h.query_window(total_events - 40, total_events, channels=[0])  # recent -> RAM tail
        recent_bytes = h._last_query_bytes
        old_ev = h.query_window(0, 8 * n_ch, channels=[0])   # earliest ticks -> must SEEK a block
        old_bytes = h._last_query_bytes
        r = {
            "path": p,
            "total": h.n_events(),
            "resident": sum(len(a) for a in h._tail),        # RAM tail: the only per-event resident store
            "index": sum(len(a) for a in h._itimes),         # coarse offset index: O(total / K)
            "recent_n": len(recent),
            "recent_bytes": recent_bytes,
            "old": [t for t, _ in old_ev],
            "old_bytes": old_bytes,
            "file_bytes": os.path.getsize(p),
            "old_true": [t for t in range(0, 8 * n_ch) if t % n_ch == 0],
        }
        h.close()
        return r

    BN = 8
    lo = _ooc_probe(1_000, BN)
    hi = _ooc_probe(100_000, BN)
    cap = _INDEX_STRIDE * BN                                  # tail cap: K events/channel, total-independent

    big2 = StreamingHub(hi["path"], BN, resume=True)          # reopen the big lake purely from disk
    reopened_total = big2.n_events()
    reopened_old = [t for t, _ in big2.query_window(0, 8 * BN, channels=[0])]
    big2.close()

    print("-" * 60)
    print("OUT-OF-CORE (bounded resident memory as the lake grows):")
    print(f"  1k events   : resident tail {lo['resident']} events, index {lo['index']} entries, "
          f"old query {lo['old_bytes']} B (fully in-RAM)")
    print(f"  100k events : resident tail {hi['resident']} events, index {hi['index']} entries, "
          f"old query SEEK {hi['old_bytes']:,} B of {hi['file_bytes']:,} B "
          f"({100 * hi['old_bytes'] / hi['file_bytes']:.1f}% of file)")
    print(f"  tail cap    : {cap} events (K={_INDEX_STRIDE} x {BN} channels) -- independent of total; "
          f"recent query read {hi['recent_bytes']} B")
    print("=" * 60)

    # (b) BOUNDED MEMORY: the resident per-event store is capped independent of total events.
    assert lo["resident"] <= cap and hi["resident"] <= cap, "resident tail is not bounded by the cap"
    assert hi["total"] == 100_000, "big probe did not record all events"
    # the coarse offset index is O(total/K) -- small, NOT O(total events).
    assert hi["index"] <= hi["total"] // _INDEX_STRIDE + BN, "coarse index larger than O(total/K)"
    assert hi["index"] < hi["total"] // 8, "index grew like O(total), not O(total/K)"
    # small lake stays fully in-RAM; big lake SEEKS a bounded block (never the whole file).
    assert lo["old_bytes"] == 0, "small lake should answer from RAM (no disk read)"
    assert hi["recent_bytes"] == 0, "a recent-window query should hit the RAM tail (no disk read)"
    assert 0 < hi["old_bytes"] < hi["file_bytes"] // 4, "old-range query did not SEEK a bounded block"
    # (a) CORRECTNESS unchanged at scale + survives reopen from the durable lake.
    assert lo["old"] == lo["old_true"], "small-lake old-range query returned wrong events"
    assert hi["old"] == hi["old_true"], "seeked old-range query returned wrong events"
    assert reopened_total == hi["total"], "reopen from disk lost events"
    assert reopened_old == hi["old"], "reopen changed the seeked query result"
    # (c) DETERMINISM: same construction -> same resident shape + same query answers.
    again = _ooc_probe(100_000, BN)
    assert (again["resident"], again["index"], again["old"], again["old_bytes"]) == \
           (hi["resident"], hi["index"], hi["old"], hi["old_bytes"]), "out-of-core path is not deterministic"

    print("self-check OK: record-while-query works, queries are partial + live, "
          "durable log replays exactly; hot_motifs surfaces the recurring coactivation motif; "
          "group-commit default is byte-identical and flush_every=K loses nothing on query/close; "
          "OUT-OF-CORE: resident memory bounded (tail cap + O(total/K) index), old-range queries "
          "SEEK the log, results unchanged, deterministic, survive reopen")


if __name__ == "__main__":
    main()
