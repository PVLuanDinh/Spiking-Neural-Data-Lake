"""
Lake-backed working memory -- the spike data lake IS the brain's short-term store.

Today a brain's working memory lives in volatile in-RAM neural state while the data lake
(the append-only `.spkl` spike log in streaming_hub.StreamingHub) is a SEPARATE, durable thing.
This module removes that gap: every tick's activity is APPENDED to the durable lake, and
"what the brain is currently holding" is read back by a recent-window query over that same log.
So the working memory does not live beside the lake -- it lives INSIDE it, and survives a restart.

Mechanism (deterministic, storage-only):
  1. write(t, active_channels): for each active channel index, StreamingHub.append(channel, t)
     records one spike (channel u32, time u32 = 8 B) to the durable append-only `.spkl` log and
     the live index. `t` is the monotonic tick index; the latest `t` is tracked as the "now" cursor.
  2. recall(window): query_window(last_t-window+1, last_t+1) reads the last `window` ticks back
     out of the durable lake and groups them per tick -> the bounded "currently held" contents.
  3. most_recent(lo, hi): scans the durable lake for the most-recently written channel in a
     reserved slot range [lo, hi) -> read back a one-hot value (the current goal / action slot).
  4. reopen(): close the handle, reopen the SAME file with resume=True (StreamingHub replays the
     log), and rebuild the "now" cursor purely from disk -> the WM is reconstructed from the lake.

Fair control (un-riggable):
  - The WM lives in the LAKE, not RAM: after reopen() throws away the in-memory object and replays
    only the durable `.spkl`, recall() returns the SAME recent contents. Contents that survive
    discarding RAM and replaying the file genuinely live in the data lake, not in the process.
  - Recency window: recall(window=k) returns ONLY the last k ticks -- an older tick's channels are
    ABSENT and a recent tick's channels are PRESENT (a bounded store, not the whole history).
  - Durability accounting: n_events() equals the number of appends and the on-disk `.spkl` is
    non-empty (size == events x 8 B) -- the lake actually persisted the working memory, no in-RAM
    shortcut. Same inputs -> identical recall (determinism).

Honest framing: a reduced recency / short-term episodic store whose contents are DURABLY held in
the spike data lake -- each tick's activity is appended to the `.spkl` log and read back by a
recent-window query, so "the data lake IS the working memory". It is a windowed read of a durable
append-only log, not a biological working-memory model; its only job is to make the lake<->WM
coupling real, durable, and inspectable. Storage only -- it imports nothing from brainstem/ and
touches no actuator.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.lake_working_memory
"""
import os
import random
import tempfile

from snn_data_lake.substrate.streaming_hub import StreamingHub

_REC_BYTES = 8          # StreamingHub stores (channel u32, time u32) = 8 B/event
_MAX_T = 1 << 32        # exclusive upper bound over all u32 tick times (whole-lake query)


class LakeWorkingMemory:
    """Working memory whose contents live in the durable spike lake (StreamingHub).

    write(t, active_channels) appends one tick of activity to the `.spkl` log; recall(window)
    reads the last `window` ticks back out of the lake as `(t, frozenset(channels))` (newest last);
    most_recent(lo, hi) reads back a one-hot value from a reserved slot range; reopen() reconstructs
    the whole store purely from the durable file, proving the WM survives a restart."""

    def __init__(self, n_channels, path=None, seed=0):
        """`n_channels` is the width of the brain's activity vector PLUS any reserved value-slots.

        If `path` is None a unique temp `.spkl` file is created (and removed by close(remove=True)
        or by the caller); otherwise the given path is used. `seed` drives a per-instance RNG for
        deterministic demo activity (no import-time seeding)."""
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".spkl")
            os.close(fd)                                    # StreamingHub opens its own handle
            self._created_temp = True
        else:
            self._created_temp = False
        self.n = int(n_channels)
        self.seed = seed
        self.rng = random.Random(seed)
        self.path = path
        self.last_t = None                                  # the "now" cursor (latest tick written)
        self.hub = StreamingHub(path, self.n, resume=False)

    # ---- write one tick into the durable lake -------------------------------
    def write(self, t, active_channels):
        """Record tick `t`: append one spike per active channel to the durable `.spkl` log.

        Advances the "now" cursor to the latest tick seen (t is expected monotonic, but max() keeps
        the cursor correct even if a caller writes out of order)."""
        t = int(t)
        for c in active_channels:
            self.hub.append(int(c), t)
        self.last_t = t if self.last_t is None else max(self.last_t, t)

    # ---- read the current working-memory contents back out of the lake ------
    def recall(self, window):
        """The last `window` ticks' contents, read from the durable lake and grouped per tick.

        Returns a list of `(t, frozenset(channels))` sorted by t ascending (newest LAST). Empty if
        nothing has been written or `window` <= 0. This is the bounded "what the brain is currently
        holding" view -- a recent-window query over the durable append-only log."""
        if self.last_t is None or window <= 0:
            return []
        t0 = max(0, self.last_t - window + 1)
        t1 = self.last_t + 1
        by_tick = {}
        for t, c in self.hub.query_window(t0, t1):
            by_tick.setdefault(t, set()).add(c)
        return [(t, frozenset(chans)) for t, chans in sorted(by_tick.items())]

    # ---- read back a one-hot value from a reserved slot range ---------------
    def most_recent(self, lo, hi):
        """The most-recently written channel in the reserved range [lo, hi), or None.

        Scans the durable lake for the latest event whose channel is in [lo, hi) and returns that
        channel index (ties broken by highest channel for determinism). With a one-hot value written
        on a reserved range this reads back the CURRENT value (e.g. the current goal/action slot)."""
        chans = [c for c in range(lo, hi) if 0 <= c < self.n]
        if not chans or self.last_t is None:
            return None
        events = self.hub.query_window(0, self.last_t + 1, channels=chans)
        if not events:
            return None
        _, channel = max(events, key=lambda tc: (tc[0], tc[1]))
        return channel

    # ---- reconstruct the WM purely from the durable lake --------------------
    def reopen(self):
        """Close and reopen the SAME `.spkl` with resume=True, rebuilding the "now" cursor from disk.

        This discards the in-RAM index and replays the durable log, proving the working memory is
        reconstructable from the lake alone. Returns self."""
        self.hub.close()
        self.hub = StreamingHub(self.path, self.n, resume=True)
        events = self.hub.query_window(0, _MAX_T)
        self.last_t = max((t for t, _ in events), default=None)
        return self

    def n_events(self):
        """Total spikes recorded in the durable lake (== number of append()s)."""
        return self.hub.n_events()

    def close(self, remove=False):
        """Release the lake handle. If `remove` is True and this instance created a temp file,
        delete it (leaves no litter)."""
        self.hub.close()
        if remove and self._created_temp and os.path.exists(self.path):
            os.remove(self.path)


def main():
    N = 16                      # 12 activity channels + 4 reserved value slots
    ACT = 12                    # activity channels [0, 12)
    VLO, VHI = 12, 16           # reserved one-hot value slots [12, 16)
    created = []                # temp paths to clean up at the end

    try:
        # ---- write a handful of ticks of activity into the durable lake -----
        wm = LakeWorkingMemory(N, seed=0)
        created.append(wm)
        plan = {                # tick -> the channels active on that tick
            0: [1, 4],
            1: [2, 7],
            2: [0, 9],
            3: [3, 5],
            4: [6, 8],
            5: [10, 11],
        }
        for t, chans in plan.items():
            wm.write(t, chans)
        appends = sum(len(c) for c in plan.values())

        # recall the last 3 ticks BEFORE and AFTER reopen (reconstruct from disk)
        W = 3
        recall_before = wm.recall(W)
        wm.reopen()                                     # throw away RAM, replay durable log
        recall_after = wm.recall(W)
        disk_bytes = os.path.getsize(wm.path)
        events = wm.n_events()

        # ---- one-hot value round-trip on the reserved slot range ------------
        vm = LakeWorkingMemory(N, seed=0)
        created.append(vm)
        values = [3, 1, 2, 0, 3, 2]                     # a sequence of goal/action values
        for t, v in enumerate(values):
            vm.write(t, [t % ACT, VLO + v])             # some activity + the one-hot value
        read_channel = vm.most_recent(VLO, VHI)
        read_value = read_channel - VLO

        # ---- determinism: same seed -> same activity -> identical recall ----
        da = LakeWorkingMemory(N, seed=123)
        db = LakeWorkingMemory(N, seed=123)
        created += [da, db]
        for t in range(8):
            ca = sorted(da.rng.sample(range(ACT), 2))
            cb = sorted(db.rng.sample(range(ACT), 2))
            da.write(t, ca)
            db.write(t, cb)
        recall_da = da.recall(4)
        recall_db = db.recall(4)

        print("=" * 68)
        print("LAKE WORKING MEMORY -- the spike data lake IS the working memory")
        print("=" * 68)
        print(f"channels     : {N} ({ACT} activity + {VHI - VLO} reserved value slots "
              f"[{VLO},{VHI}))")
        print(f"wrote        : {len(plan)} ticks, {appends} spikes -> durable {os.path.basename(wm.path)}")
        print(f"durable log  : {disk_bytes} B on disk ({events} events x {_REC_BYTES} B)")
        print("-" * 68)
        print(f"recall(3) BEFORE reopen : {[(t, sorted(s)) for t, s in recall_before]}")
        print(f"recall(3) AFTER  reopen : {[(t, sorted(s)) for t, s in recall_after]}")
        print("  (identical -> the WM was reconstructed from the durable lake, not RAM)")
        print("-" * 68)
        print(f"value stream {values} -> most_recent([{VLO},{VHI})) = ch {read_channel} "
              f"= value {read_value} (the last one written)")
        print(f"determinism  : two seed=123 stores -> recall equal = {recall_da == recall_db}")
        print("=" * 68)

        # ---- un-riggable self-checks (deterministic asserts) ----------------
        # (1) The WM lives in the LAKE, not RAM: contents survive discarding RAM + replaying disk.
        assert recall_before == recall_after, "reopen from the durable lake changed the WM contents"
        assert recall_after, "recall after reopen must be non-empty (WM did not survive the restart)"

        # (2) Recency window: recall(k) holds ONLY the last k ticks (bounded store).
        recalled_ticks = {t for t, _ in recall_after}
        assert recalled_ticks == {3, 4, 5}, f"recall(3) must be the last 3 ticks, got {recalled_ticks}"
        old_channels = set(plan[0])                     # tick 0 is outside the window
        recent_channels = {c for _, s in recall_after for c in s}
        assert not (old_channels & recent_channels), "an old tick's channels leaked into recall(3)"
        assert set(plan[5]).issubset(recent_channels), "the most recent tick's channels are missing"

        # (3) Value round-trip: most_recent tracks the LAST written value.
        assert read_value == values[-1], f"most_recent must read back the last value {values[-1]}"
        assert read_channel == VLO + values[-1], "most_recent returned the wrong reserved channel"

        # (4) Durability accounting: n_events == appends and the .spkl actually persisted the WM.
        assert events == appends, f"n_events {events} != appends {appends}"
        assert disk_bytes > 0, "durable .spkl is empty -- the WM was not persisted to the lake"
        assert disk_bytes == events * _REC_BYTES, "durable log size != events x 8 B"

        # (5) Determinism: identical inputs -> identical recall.
        assert recall_da == recall_db, "same seed produced different recall results"

        print("self-check OK: WM lives in the durable lake (survives reopen); recall is a bounded "
              "recency window; one-hot value round-trips; n_events == appends and the .spkl is "
              "non-empty; recall is deterministic")
    finally:
        for w in created:
            w.close(remove=True)                        # release handles + delete temp files


if __name__ == "__main__":
    main()
