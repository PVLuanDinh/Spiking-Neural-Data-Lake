"""
v0.12 — Spike Telemetry Hub  (Paradigm A).

Completes the assessment's Paradigm A: a hub that ingests, stores, manages and
queries multi-channel spike-train telemetry — the kind streamed by high-density
BCIs (e.g. Neuralink, 1024 channels) or large neural simulations — as SPARSE
events, never as dense waveforms/rasters.

Core idea (the data-lake bit): a spike train is mostly silence. Store only the
events (per-channel sorted spike times = Address-Event Representation), persist
them to a self-describing file with a per-channel index, and answer
"which spikes fired on channels X in window [t0,t1)?" by SEEKING to just those
channels and binary-searching the window — reading neither other channels nor the
silent gaps into memory.

Capabilities:
  - ingest multi-channel spike times
  - .spk file format with a channel index (offset+count) -> partial reads
  - windowed range query (in-memory and disk-partial), O(log n + hits)
  - bin / firing-rate / inter-spike-interval (ISI) / burst (anomaly) detection
  - storage + query-efficiency measurement vs a dense raster

Run:  python spike_telemetry_hub.py
"""

import os
import random
import struct
from array import array
from bisect import bisect_left

# no import-time random.seed() (a process-global side effect); main() seeds locally for its own
# synthetic-store generation. The store format + query paths themselves use no randomness.

_MAGIC = b"SPK1"
_HEADER = struct.Struct("<4sII")  # magic, n_channels, duration
_INDEX = struct.Struct("<QI")  # per-channel: byte offset, spike count


class SpikeTelemetryHub:
    """In-memory sparse store: one sorted uint32 spike-time array per channel."""

    def __init__(self, n_channels, duration):
        if n_channels <= 0 or duration <= 0:
            raise ValueError("n_channels and duration must be positive")
        self.n = n_channels
        self.dur = duration
        self.ch = [array("I") for _ in range(n_channels)]

    # ---- ingest -------------------------------------------------------------
    def ingest(self, channel, times):
        if not 0 <= channel < self.n:
            raise IndexError(f"channel {channel} out of range [0,{self.n})")
        a = self.ch[channel]
        prev_last = a[-1] if a else None  # O(1) max: the stored array is kept sorted
        new = [int(t) for t in times]
        a.extend(new)
        # v2.4 perf: the stored array is ALWAYS sorted (invariant maintained here), so an
        # appended batch keeps it sorted iff the batch is itself sorted AND its first element
        # is >= the previous last. Only re-sort when that boundary is actually broken --
        # incremental ingest is O(batch) per call, not an O(n) full rescan every time (which
        # was O(n^2) overall). The stored array is byte-identical to the old full-rescan path.
        in_order = all(new[i] <= new[i + 1] for i in range(len(new) - 1))
        if new and not (in_order and (prev_last is None or prev_last <= new[0])):
            self.ch[channel] = array("I", sorted(a))

    def n_events(self):
        return sum(len(a) for a in self.ch)

    # ---- persistence (.spk with channel index for partial reads) ------------
    def save(self, path):
        index, offset = [], _HEADER.size + self.n * _INDEX.size
        for a in self.ch:
            index.append((offset, len(a)))
            offset += len(a) * 4
        with open(path, "wb") as f:
            f.write(_HEADER.pack(_MAGIC, self.n, self.dur))
            for off, cnt in index:
                f.write(_INDEX.pack(off, cnt))
            for a in self.ch:
                f.write(a.tobytes())
        return os.path.getsize(path)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            magic, n, dur = _HEADER.unpack(f.read(_HEADER.size))
            if magic != _MAGIC:
                raise ValueError(f"not a .spk file (bad magic {magic!r})")
            idx = [_INDEX.unpack(f.read(_INDEX.size)) for _ in range(n)]
            hub = SpikeTelemetryHub(n, dur)
            for c, (off, cnt) in enumerate(idx):
                f.seek(off)
                a = array("I")
                a.frombytes(f.read(cnt * 4))
                hub.ch[c] = a
        return hub

    # ---- query / analyse ----------------------------------------------------
    def query_window(self, t0, t1, channels=None):
        """events with t0 <= t < t1; binary-search per channel (no full scan)."""
        out = []
        for c in channels if channels is not None else range(self.n):
            a = self.ch[c]
            for k in range(bisect_left(a, t0), bisect_left(a, t1)):
                out.append((a[k], c))
        return out

    def rates(self):
        return [len(a) / self.dur for a in self.ch]

    def bin(self, channel, window):
        a = self.ch[channel]
        nb = (self.dur + window - 1) // window
        counts = [0] * nb
        for t in a:
            counts[t // window] += 1
        return counts

    def isi(self, channel):
        a = self.ch[channel]
        return [a[i + 1] - a[i] for i in range(len(a) - 1)]

    def bursts(self, channel, window, k):
        """bins (anomalies) where spike count >= k — simple rate-threshold detector."""
        return [(b * window, c) for b, c in enumerate(self.bin(channel, window)) if c >= k]


def disk_query(path, t0, t1, channels):
    """Windowed query that reads ONLY the requested channels from disk — the
    'don't load the whole dataset' path. Returns (events, bytes_read)."""
    bytes_read = 0
    with open(path, "rb") as f:
        magic, n, _ = _HEADER.unpack(f.read(_HEADER.size))
        if magic != _MAGIC:
            raise ValueError("not a .spk file")
        out_events = []
        for c in channels:
            if not 0 <= c < n:
                raise IndexError(f"channel {c} out of range")
            f.seek(_HEADER.size + c * _INDEX.size)
            off, cnt = _INDEX.unpack(f.read(_INDEX.size))
            f.seek(off)
            a = array("I")
            a.frombytes(f.read(cnt * 4))
            bytes_read += cnt * 4
            for k in range(bisect_left(a, t0), bisect_left(a, t1)):
                out_events.append((a[k], c))
    return out_events, bytes_read


def pct_read(bytes_read, total_bytes):
    """Windowed-query efficiency: % of the .spk file a partial read actually touched."""
    return 100.0 * bytes_read / total_bytes if total_bytes else 0.0


# ---- bring your own data -----------------------------------------------------
# These two were the only "CSV -> queryable spike store" on-ramp in the project, and they lived
# inside a tkinter control panel (substrate/control_panel.py) behind an Import button. Moved here
# when that panel was retired: they build a SpikeTelemetryHub and nothing else, so a GUI was never
# their home, and a capability reachable only by clicking is a capability most users never find.
# substrate/sensor_csv_ingest.py is a different thing (timestamped serial CSV -> signal_loop windows).


def csv_to_hub(path):
    """A CSV of spike EVENTS -- each row 'channel,time' (integers; ';' also accepted; header /
    comment / blank rows skipped). Channel count and duration are inferred from the data."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(";", ",")
            if not line or not (line[0].isdigit() or line[0] in "+-"):
                continue  # skip header / comment / blank
            parts = line.split(",")
            if len(parts) >= 2:
                events.append((int(float(parts[0])), int(float(parts[1]))))
    if not events:
        raise ValueError("no 'channel,time' rows found")
    n_ch = max(c for c, _ in events) + 1
    dur = max(t for _, t in events) + 1
    by_ch = {}
    for c, t in events:
        by_ch.setdefault(c, []).append(t)
    hub = SpikeTelemetryHub(n_ch, dur)
    for c in range(n_ch):
        hub.ingest(c, sorted(by_ch.get(c, [])))
    return hub


def dense_csv_to_hub(path, threshold=0.5):
    """A DENSE sensor CSV: one row per time step, one column per channel; a value >= threshold
    emits a spike at (column, row). Non-numeric header rows are skipped. This is the
    'I have a table of sensor readings over time' path -- the dense->sparse conversion that makes
    the rest of this module apply to data that did not arrive as events."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(";", ",")
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                continue  # header / non-numeric row -> skip
    if not rows:
        raise ValueError("no numeric rows found")
    n_ch = max(len(r) for r in rows)
    hub = SpikeTelemetryHub(n_ch, len(rows))
    for c in range(n_ch):
        hub.ingest(c, [t for t, r in enumerate(rows) if c < len(r) and r[c] >= threshold])
    return hub


# ---- demo + measure ---------------------------------------------------------
def synth(n_channels, duration, rate, burst_channels, burst_win):
    """simulate sparse multi-channel telemetry; inject a dense burst on a few
    channels inside burst_win (the 'anomaly' to be detected/queried)."""
    hub = SpikeTelemetryHub(n_channels, duration)
    for c in range(n_channels):
        k = max(1, int(rate * duration))
        times = sorted(random.sample(range(duration), k))
        if c in burst_channels:  # add a dense burst
            b0, b1 = burst_win
            # set(): a neuron fires at most once per step (canonical AER) — the burst sample
            # can land on a baseline time, and a duplicate (t,channel) is not a real 2nd spike.
            times = sorted(set(times + random.sample(range(b0, b1), (b1 - b0) // 3)))
        hub.ingest(c, times)
    return hub


def main():
    random.seed(0)  # local, deterministic synthetic store (not an import-time side effect)
    N, DUR, RATE = 256, 100_000, 0.004  # 256 channels, ~400 spikes each
    BURST_CH, BURST = {7, 99}, (50_000, 52_000)
    hub = synth(N, DUR, RATE, BURST_CH, BURST)
    path = os.path.join(__import__("tempfile").mkdtemp(prefix="snn_sc_"), "telemetry.spk")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbytes = hub.save(path)
    ev = hub.n_events()

    print("=" * 60)
    print("SPIKE TELEMETRY HUB  (Paradigm A)")
    print("=" * 60)
    print(f"channels={N}  duration={DUR:,} steps  total spikes={ev:,}")
    print()

    # storage: sparse AER file vs dense rasters
    dense_byte = N * DUR  # 1 byte / (channel,step)
    dense_bit = N * DUR // 8  # 1 bit  / (channel,step)
    print("STORAGE (lower = less space):")
    print(f"  sparse .spk file : {nbytes:,} B")
    print(f"  dense 1-bit raster: {dense_bit:,} B  -> {dense_bit / nbytes:.1f}x larger")
    print(f"  dense 1-byte raster: {dense_byte:,} B  -> {dense_byte / nbytes:.0f}x larger")
    print()

    # windowed partial-read query: only the 2 burst channels in the burst window
    events, read = disk_query(path, BURST[0], BURST[1], channels=sorted(BURST_CH))
    print("WINDOWED QUERY (channels {7,99}, window [50000,52000)):")
    print(f"  events returned : {len(events):,}")
    print(f"  bytes read      : {read:,}  ({100 * read / nbytes:.1f}% of the file)")
    print()

    # analyse: firing rate + burst (anomaly) detection
    rts = hub.rates()
    print("ANALYSIS:")
    print(f"  mean firing rate : {1000 * sum(rts) / len(rts):.2f} mHz/step (avg over channels)")
    found = hub.bursts(7, window=2000, k=300)
    print(
        f"  burst detector on ch7 (>=300 spikes/2000-step bin): {len(found)} hit(s) at {found[:1]}"
    )
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    hub2 = SpikeTelemetryHub.load(path)  # roundtrip integrity
    assert hub2.n_events() == ev, "save/load lost events"
    brute = [(t, c) for c in sorted(BURST_CH) for t in hub.ch[c] if BURST[0] <= t < BURST[1]]
    assert sorted(events) == sorted(brute), "disk windowed query != brute force"
    assert read < nbytes, "partial read touched the whole file"
    assert nbytes < dense_bit, "sparse store not smaller than 1-bit raster"
    assert len(found) >= 1, "injected burst not detected"
    # ---- the bring-your-own-data path, exercised end to end ------------------
    # csv_to_hub / dense_csv_to_hub were reachable only through a GUI button before they moved here,
    # which is why they had no gate. They do now: a CSV must survive the whole round trip -- parse,
    # ingest, save, partial disk query -- and the dense path must agree with the event path on the
    # same data, because "spike at (col,row) where value >= threshold" is a claim about equivalence.
    import tempfile

    fd, csv_path = tempfile.mkstemp(prefix="sndl_hub_selfcheck_", suffix=".csv")
    os.close(fd)
    fd, dense_path = tempfile.mkstemp(prefix="sndl_hub_selfcheck_dense_", suffix=".csv")
    os.close(fd)
    fd, spk_path = tempfile.mkstemp(prefix="sndl_hub_selfcheck_", suffix=".spk")
    os.close(fd)
    try:
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("channel,time\n0,1\n0,5\n2,3\n2,9\n")  # header row must be skipped
        h = csv_to_hub(csv_path)
        assert h.n == 3 and h.dur == 10, f"inferred shape wrong: {h.n}x{h.dur}"
        assert h.n_events() == 4, f"csv_to_hub lost events: {h.n_events()}"
        assert list(h.ch[0]) == [1, 5] and list(h.ch[2]) == [3, 9], "csv event times wrong"

        # the same four events written densely: 10 rows x 3 columns, 1.0 where a spike belongs
        grid = [[0.0] * 3 for _ in range(10)]
        for c, t in ((0, 1), (0, 5), (2, 3), (2, 9)):
            grid[t][c] = 1.0
        with open(dense_path, "w", encoding="utf-8") as fh:
            fh.write("a,b,c\n" + "\n".join(",".join(str(v) for v in row) for row in grid) + "\n")
        hd = dense_csv_to_hub(dense_path, threshold=0.5)
        assert [list(hd.ch[c]) for c in range(3)] == [list(h.ch[c]) for c in range(3)], (
            "dense and event CSV paths disagree on identical data"
        )

        h.save(spk_path)
        got, nread = disk_query(spk_path, 0, 10, [0])
        assert sorted(t for t, _ in got) == [1, 5], f"windowed query over an imported CSV: {got}"
        assert 0.0 < pct_read(nread, os.path.getsize(spk_path)) < 100.0, "partial read not partial"
    finally:
        for p in (csv_path, dense_path, spk_path):
            if os.path.exists(p):
                os.remove(p)

    print(
        "self-check OK: roundtrip intact, windowed query correct & partial, "
        "sparse < dense, burst detected, CSV import (event + dense) round-trips and agrees"
    )


def cli(argv=None):
    """Bring your own data: turn a CSV into a queryable .spk store, or query one you already have.

    This is the surface that replaced the retired control panel's Lake tab. The engine was always
    here; only the way in was trapped behind a GUI.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="spike_telemetry_hub", description=cli.__doc__)
    ap.add_argument("--ingest", metavar="CSV", help="CSV to import ('channel,time' rows)")
    ap.add_argument(
        "--dense", action="store_true", help="treat CSV as dense rows=time, cols=channels"
    )
    ap.add_argument("--threshold", type=float, default=0.5, help="dense: value >= T emits a spike")
    ap.add_argument("--out", metavar="SPK", help="write the imported store here")
    ap.add_argument("--query", metavar="SPK", help="windowed query against an existing .spk")
    ap.add_argument("--window", default="", metavar="T0,T1", help="query window (default: all)")
    ap.add_argument("--channels", default="", metavar="A,B", help="channels (default: all)")
    a = ap.parse_args(argv)

    if a.ingest:
        hub = dense_csv_to_hub(a.ingest, a.threshold) if a.dense else csv_to_hub(a.ingest)
        print(f"imported {a.ingest}: {hub.n} channels, duration {hub.dur}, {hub.n_events()} events")
        if a.out:
            hub.save(a.out)
            print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes)")
        else:
            print("(no --out given; nothing written)")
        return 0

    if a.query:
        with open(a.query, "rb") as f:
            _, n_ch, dur = _HEADER.unpack(f.read(_HEADER.size))
        t0, t1 = (int(x) for x in a.window.split(",")) if a.window else (0, dur)
        chans = [int(x) for x in a.channels.split(",")] if a.channels else list(range(n_ch))
        events, nread = disk_query(a.query, t0, t1, chans)
        total = os.path.getsize(a.query)
        print(f"{a.query}: {n_ch} channels, duration {dur}, {total} bytes")
        print(f"window [{t0},{t1}) over {len(chans)} channel(s) -> {len(events)} events")
        print(f"bytes read: {nread} ({pct_read(nread, total):.1f}% of the file)")
        for t, c in events[:20]:
            print(f"  t={t} ch={c}")
        if len(events) > 20:
            print(f"  ... {len(events) - 20} more")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    import sys

    # bare invocation runs the self-check (this module is in run_selfchecks CORE, which passes no
    # args); anything else is a user asking for the ingest/query CLI.
    sys.exit(cli() if len(sys.argv) > 1 else (main() or 0))
