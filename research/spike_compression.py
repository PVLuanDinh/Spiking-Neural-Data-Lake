"""
Spike-train compression — a spike-native lossless codec vs the fixed-precision store + gzip.

The storage thesis is the whole point, and the lake's sparse `.spk`/`.spc` formats
(spike_telemetry_hub.py, signal_loop.flush_lake) store spike timestamps at FIXED precision (uint32,
4 B/spike) — sparse vs a dense raster (61× there), but not compressed among themselves. Spikes carry
information in their TIMING, and inter-spike intervals are small and skewed, so a spike-native codec —
delta-code the sorted times per channel, then LEB128 varint the deltas (small ISI -> 1 byte) — beats
both the fixed 4-byte store and generic gzip, losslessly. Refs: lossless event coding (arXiv:2411.07155,
2411.03010); spike-only 100–1000× data reduction (Nature Comms Eng 2025).

  python spike_compression.py        # (also run by run_selfchecks.py)
"""
import gzip
import random

random.seed(0)


def synth_spikes(n_channels=64, duration=20000, active=0.04):
    """Sparse per-channel spike trains: sorted unique integer timestamps (~4% of the duration)."""
    rng = random.Random(0)
    spikes = {}
    for ch in range(n_channels):
        k = max(1, int(duration * active * rng.uniform(0.5, 1.5)))
        spikes[ch] = sorted(set(rng.randrange(duration) for _ in range(k)))
    return spikes


def _leb128(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_leb128(buf, pos):
    n = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, pos
        shift += 7


def encode(spikes):
    """Delta + LEB128 varint per channel: [ch][count][ISI deltas...] — the spike-native codec."""
    buf = bytearray()
    for ch in sorted(spikes):
        times = spikes[ch]
        buf += _leb128(ch) + _leb128(len(times))
        prev = 0
        for t in times:
            buf += _leb128(t - prev)        # sorted -> non-negative deltas, usually tiny
            prev = t
    return bytes(buf)


def decode(buf):
    spikes, pos, n = {}, 0, len(buf)
    while pos < n:
        ch, pos = _read_leb128(buf, pos)
        count, pos = _read_leb128(buf, pos)
        times, prev = [], 0
        for _ in range(count):
            d, pos = _read_leb128(buf, pos)
            prev += d
            times.append(prev)
        spikes[ch] = times
    return spikes


def raw_bytes(spikes):
    """Fixed-precision baseline: 4-byte uint32 per (ch header, count, each timestamp)."""
    buf = bytearray()
    for ch in sorted(spikes):
        buf += ch.to_bytes(4, "little") + len(spikes[ch]).to_bytes(4, "little")
        for t in spikes[ch]:
            buf += t.to_bytes(4, "little")
    return bytes(buf)


def main():
    spikes = synth_spikes()
    n_spikes = sum(len(v) for v in spikes.values())

    raw = raw_bytes(spikes)
    var = encode(spikes)
    gz_raw = gzip.compress(raw, 9)
    gz_var = gzip.compress(var, 9)

    print("=" * 60)
    print("SPIKE-TRAIN COMPRESSION (spike-native delta+varint, lossless)")
    print("=" * 60)
    print(f"{n_spikes:,} spikes over {len(spikes)} channels")
    print(f"  fixed uint32 (.spk-style) : {len(raw):>8,} B  ({len(raw) / n_spikes:.2f} B/spike)")
    print(f"  gzip(fixed)               : {len(gz_raw):>8,} B  ({len(raw) / len(gz_raw):.1f}× vs fixed)")
    print(f"  delta+varint (spike-native): {len(var):>8,} B  ({len(raw) / len(var):.1f}× vs fixed)")
    print(f"  gzip(delta+varint)        : {len(gz_var):>8,} B  ({len(raw) / len(gz_var):.1f}× vs fixed)")
    print("=" * 60)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert decode(var) == spikes, "codec not lossless (round-trip mismatch)"
    assert len(var) < len(raw), "delta+varint should beat the fixed-precision store"
    assert len(gz_var) < len(gz_raw), "spike-native codec should beat generic gzip on the fixed store"
    print(f"self-check OK: lossless; delta+varint {len(raw) / len(var):.1f}× < fixed and "
          f"gzip(varint) {len(gz_raw) / len(gz_var):.1f}× smaller than gzip(fixed)")


if __name__ == "__main__":
    main()
