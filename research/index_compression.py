"""
Index / edge-list compression — shrink the lookup METADATA, not the spike payload (scan #8).

spike_compression.py compresses spike TIMESTAMPS (the payload). A distinct target is the store's
lookup metadata: the id / edge / adjacency lists the content-addressable query path walks (Paradigm B
match stamps, Paradigm C KG edges, telemetry channel indexes). These are SETS — neighbour order is
irrelevant — so sorting then delta-coding the gaps + varint (order-invariant coding) shrinks them far
below fixed 4-byte ids, losslessly, at zero query cost. Refs: arXiv:2501.10479 (Meta, "Lossless
Compression of Vector IDs for ANN Search", ~7x on ids).

  Run: python index_compression.py       # (also run by run_selfchecks.py)
"""
import gzip
import random

import spike_compression as sc  # research/ on sys.path; reuse the LEB128 varint

random.seed(0)

ID_SPACE = 100_000


def adjacency(n_lists=200, seed=0):
    """Clustered neighbour-id lists (ids cluster in id-space -> small d-gaps), like a KG's edges."""
    rng = random.Random(seed)
    lists = []
    for _ in range(n_lists):
        base = rng.randrange(ID_SPACE - 2000)
        k = rng.randint(8, 30)
        lists.append(sorted({base + rng.randrange(2000) for _ in range(k)}))
    return lists


def encode(lists):
    """Order-invariant: per list, LEB128(count) then LEB128 d-gaps of the sorted ids."""
    buf = bytearray()
    for ids in lists:
        buf += sc._leb128(len(ids))
        prev = 0
        for x in ids:
            buf += sc._leb128(x - prev)          # sorted -> small non-negative gaps -> ~1 byte each
            prev = x
    return bytes(buf)


def decode(buf):
    lists, pos, n = [], 0, len(buf)
    while pos < n:
        count, pos = sc._read_leb128(buf, pos)
        ids, prev = [], 0
        for _ in range(count):
            g, pos = sc._read_leb128(buf, pos)
            prev += g
            ids.append(prev)
        lists.append(ids)
    return lists


def raw_bytes(lists):
    buf = bytearray()
    for ids in lists:
        buf += len(ids).to_bytes(4, "little")
        for x in ids:
            buf += x.to_bytes(4, "little")       # fixed 4-byte id
    return bytes(buf)


def main():
    lists = adjacency()
    n_ids = sum(len(x) for x in lists)
    raw = raw_bytes(lists)
    var = encode(lists)
    gz = len(gzip.compress(raw, 9))

    print("=" * 58)
    print("INDEX / EDGE-LIST COMPRESSION (order-invariant d-gap+varint)")
    print("=" * 58)
    print(f"{len(lists)} lists, {n_ids:,} ids over an id-space of {ID_SPACE:,}")
    print(f"  fixed 4-byte ids : {len(raw):>7,} B  ({len(raw) / n_ids:.2f} B/id)")
    print(f"  gzip(fixed)      : {gz:>7,} B  ({len(raw) / gz:.1f}x)")
    print(f"  d-gap + varint   : {len(var):>7,} B  ({len(raw) / len(var):.1f}x)  ({len(var) / n_ids:.2f} B/id)")
    print("compresses the query INDEX metadata (edges/ids), complementary to spike_compression's payload.")
    print("=" * 58)

    # ---- self-check (ponytail: one runnable assert-based check; deterministic) ----
    assert decode(var) == lists, "index codec must be lossless (round-trip mismatch)"
    assert len(var) < len(raw), "d-gap+varint should beat fixed 4-byte ids"
    assert len(var) < gz, f"order-invariant codec should beat generic gzip ({len(var)} !< {gz})"
    print(f"self-check OK: lossless; d-gap+varint {len(raw) / len(var):.1f}x < fixed and beats gzip "
          f"({gz} -> {len(var)} B)")


if __name__ == "__main__":
    main()
