"""
Quadtree event compression — a 2-D spatial axis for the lossless store (scan #6).

spike_compression.py is 1-D: it delta+varint-codes each channel's spike times independently, so it
cannot exploit SPATIAL structure — bursts where adjacent channels co-fire in the same time window
(exactly the co-firing the Gold synchrony view surfaces). A quadtree over the channel×time event grid
encodes empty regions in one bit and only recurses into occupied quadrants, so a clustered event field
compresses far below the per-channel codec. Lossless. Refs: arXiv:2411.07155 (low-complexity lossless
event compression; the learned octree/hyperprior variant arXiv:2411.03010 is the torch-tier ceiling).

  Run: python event_quadtree_compression.py       # (also run by run_selfchecks.py)
"""

import gzip
import random

import snn_data_lake.reasoning.spike_compression as sc  # research/ on sys.path; reuse the 1-D delta+varint baseline

random.seed(0)

C, T = 64, 256  # channels × time bins (both powers of two)


def clustered_events(seed, n_bursts=10):
    """Bursty co-firing field: each burst = a block of adjacent channels active over a time window."""
    rng = random.Random(seed)
    evs = set()
    for _ in range(n_bursts):
        c0 = rng.randrange(0, C - 8)
        t0 = rng.randrange(0, T - 12)
        for c in range(c0, c0 + rng.randint(4, 8)):
            for t in range(t0, t0 + rng.randint(6, 12)):
                if rng.random() < 0.7:
                    evs.add((c, t))
    return evs


def _split(c0, t0, cs, ts):
    cr = [(c0, cs // 2), (c0 + cs // 2, cs - cs // 2)] if cs > 1 else [(c0, cs)]
    tr = [(t0, ts // 2), (t0 + ts // 2, ts - ts // 2)] if ts > 1 else [(t0, ts)]
    return [(cc, csz, tt, tsz) for cc, csz in cr for tt, tsz in tr]


def quad_encode(evs, c0, t0, cs, ts, bits):
    if not evs:
        bits.append(0)
        return
    bits.append(1)
    if cs == 1 and ts == 1:
        return  # leaf: the '1' is the spike
    for cc, csz, tt, tsz in _split(c0, t0, cs, ts):
        sub = [e for e in evs if cc <= e[0] < cc + csz and tt <= e[1] < tt + tsz]
        quad_encode(sub, cc, tt, csz, tsz, bits)


def quad_decode(bits, pos, c0, t0, cs, ts, out):
    b = bits[pos]
    pos += 1
    if b == 0:
        return pos
    if cs == 1 and ts == 1:
        out.append((c0, t0))
        return pos
    for cc, csz, tt, tsz in _split(c0, t0, cs, ts):
        pos = quad_decode(bits, pos, cc, tt, csz, tsz, out)
    return pos


def pack(bits):
    ba = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            ba[i >> 3] |= 1 << (i & 7)
    return bytes(ba), len(bits)


def main():
    evs = clustered_events(0)
    n = len(evs)

    bits = []
    quad_encode(evs, 0, 0, C, T, bits)
    qbytes, nbits = pack(bits)
    # lossless round-trip
    out = []
    quad_decode(bits, 0, 0, 0, C, T, out)
    lossless = set(out) == evs

    # baselines
    raw_bytes = n * ((C - 1).bit_length() + (T - 1).bit_length()) / 8  # fixed bits per (c,t)
    gz_raw = len(
        gzip.compress(
            b"".join(c.to_bytes(1, "little") + t.to_bytes(2, "little") for c, t in sorted(evs)), 9
        )
    )
    spikes = {}
    for c, t in evs:
        spikes.setdefault(c, []).append(t)
    for c in spikes:
        spikes[c].sort()
    onedim = len(sc.encode(spikes))  # spike_compression delta+varint (1-D)

    print("=" * 60)
    print("QUADTREE EVENT COMPRESSION (2-D spatial, lossless)")
    print("=" * 60)
    print(f"{n} events on a {C}x{T} grid (bursty co-firing)")
    print(f"  fixed-bit raw     : {raw_bytes:>7.0f} B")
    print(f"  gzip(raw)         : {gz_raw:>7} B  ({raw_bytes / gz_raw:.1f}x)")
    print(
        f"  1-D delta+varint  : {onedim:>7} B  ({raw_bytes / onedim:.1f}x)  (spike_compression.py)"
    )
    print(
        f"  2-D quadtree      : {len(qbytes):>7} B  ({raw_bytes / len(qbytes):.1f}x)  [{nbits} bits]"
    )
    print("quadtree exploits spatial burst clustering the per-channel 1-D codec cannot.")
    print("=" * 60)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert lossless, "quadtree codec must be lossless (decoded event set != original)"
    assert len(qbytes) < gz_raw, f"quadtree ({len(qbytes)} B) should beat gzip(raw) ({gz_raw} B)"
    assert len(qbytes) < onedim, (
        f"on bursty data quadtree ({len(qbytes)} B) should beat 1-D ({onedim} B)"
    )
    print(
        f"self-check OK: lossless; quadtree {len(qbytes)} B beats gzip {gz_raw} B and 1-D {onedim} B "
        "on bursty co-firing data"
    )


if __name__ == "__main__":
    main()
