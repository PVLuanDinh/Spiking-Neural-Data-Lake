"""
Arithmetic / range entropy coder -- lossless compression of a symbol stream to the Shannon bound.

The lake already has a spike-native INTEGER codec (research/spike_compression.py): delta + LEB128
varint over a sorted list of spike ids, which shrinks the GAPS between consecutive integers. That
exploits monotone integer STRUCTURE; it is not an entropy coder and does not approach the Shannon
bound for a general symbol distribution. THIS module is the complementary tool: a static-model
range coder (the byte-renormalized form of arithmetic coding; Rissanen 1976; Witten, Neal & Cleary,
"Arithmetic Coding for Data Compression", CACM 30:520, 1987). Given any symbol distribution it packs
a length-N stream into about N*H bits, where H = -sum_i p_i*log2(p_i) is the entropy: each symbol
costs ~ -log2(p_i) bits, so a frequent symbol costs a fraction of a bit. Reaches the information-
theoretic limit that no fixed-width or byte-per-symbol layout can.

How the two differ (state it plainly): spike_compression.py = delta + varint (LEB128) INTEGER codec,
compact because sorted spike ids have small gaps -- probability skew is irrelevant to it. This coder
= ENTROPY coder, compact because the symbol probabilities are skewed -- integer monotonicity is
irrelevant to it. Different structure, different technique; complementary compressors.

Core (integer range coder, 32-bit low/range, byte renormalization, carryless underflow handling
after Subbotin): to encode symbol s the interval narrows by its cumulative-frequency slice --
  range //= total ; low += cumfreq[s] * range ; range *= freq[s]
-- and whenever the top byte of `low` is settled (or `range` underflows below 2^16) that byte is
emitted and low/range shift up by 8. The decoder mirrors it exactly: it holds a `code` word, finds
the symbol whose cumulative interval contains (code - low) // (range // total), then narrows and
renormalizes with the identical arithmetic, so encoder and decoder stay in lockstep. Lossless
round-trip is the non-negotiable core signature; near-N*H size is the point.

The ceiling here is a classic STATIC-model integer range coder (byte renormalization) on
synthetic symbol streams -- the entropy-coding level. Static frequency model (count queries only);
no adaptive / context modelling, no arithmetic beyond the standard renorm. Not a spiking
implementation, not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.entropy_coder
"""

import math
import random

MASK = 0xFFFFFFFF  # 32-bit working precision
TOP = 1 << 24  # top-byte-settled renormalization threshold
BOT = 1 << 16  # underflow threshold (also the max total frequency the coder tolerates)
MAX_TOTAL = 1 << 16  # scale any model down to this so total frequency stays within the coder


def _scale_counts(counts, limit):
    """Scale integer counts so their sum is <= limit while keeping every positive count >= 1."""
    total = sum(counts)
    if total <= limit:
        return list(counts)
    scaled = [max(1, c * limit // total) for c in counts]
    over = sum(scaled) - limit
    i = 0
    n = len(scaled)
    while over > 0:  # trim the surplus off buckets that can spare it
        if scaled[i] > 1:
            scaled[i] -= 1
            over -= 1
        i = (i + 1) % n
    return scaled


def shannon_entropy(symbols):
    """Empirical entropy of a sequence in bits/symbol: H = -sum_i p_i * log2(p_i)."""
    n = len(symbols)
    if n == 0:
        return 0.0
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


class RangeCoder:
    """Static-model integer range coder. `frequencies` maps each symbol to a positive count.

    encode(symbols) -> bytes ; decode(data, n_symbols) -> list ; the round-trip is bit-exact and the
    encoded length is close to n_symbols * entropy(). Deterministic: holds no RNG, so the same model
    and symbols always yield identical bytes."""

    def __init__(self, frequencies):
        if isinstance(frequencies, dict):
            items = [(k, int(v)) for k, v in frequencies.items()]
        else:
            items = [(i, int(v)) for i, v in enumerate(frequencies)]
        items = [(s, c) for s, c in items if c > 0]
        if not items:
            raise ValueError("model needs at least one symbol with a positive count")
        items.sort(key=lambda kv: kv[0])  # stable, deterministic symbol order
        scaled = _scale_counts([c for _, c in items], MAX_TOTAL)
        self._symbols = [s for s, _ in items]
        self._freq = {}
        self._cum = {}
        self._cum_arr = []  # cumulative starts, ascending (for search)
        self._sym_arr = []  # symbols aligned with _cum_arr
        run = 0
        for s, f in zip(self._symbols, scaled):
            self._freq[s] = f
            self._cum[s] = run
            self._cum_arr.append(run)
            self._sym_arr.append(s)
            run += f
        self._total = run

    # ---- model queries ----------------------------------------------------------------------
    def entropy(self):
        """Shannon entropy of the (integer) model distribution, in bits/symbol."""
        h = 0.0
        for s in self._symbols:
            p = self._freq[s] / self._total
            h -= p * math.log2(p)
        return h

    def encoded_bits(self, symbols):
        """Encoded size of `symbols` in bits (= 8 * len(encode(symbols)))."""
        return 8 * len(self.encode(symbols))

    # ---- carryless renormalization (shared shape for encode/decode) -------------------------
    def _settled(self):
        return (self._low ^ ((self._low + self._range) & MASK)) < TOP

    def _underflow(self):
        self._range = (-self._low) & (BOT - 1)
        return True

    # ---- encoder ----------------------------------------------------------------------------
    def encode(self, symbols):
        """Encode a sequence of symbols (all must be in the model) to a bytes object."""
        self._low = 0
        self._range = MASK
        out = bytearray()
        total = self._total
        for s in symbols:
            self._range //= total
            self._low = (self._low + self._cum[s] * self._range) & MASK
            self._range = (self._range * self._freq[s]) & MASK
            while self._settled() or (self._range < BOT and self._underflow()):
                out.append((self._low >> 24) & 0xFF)
                self._low = (self._low << 8) & MASK
                self._range = (self._range << 8) & MASK
        for _ in range(4):  # flush the remaining low bytes
            out.append((self._low >> 24) & 0xFF)
            self._low = (self._low << 8) & MASK
        return bytes(out)

    # ---- decoder ----------------------------------------------------------------------------
    def _find(self, dv):
        """Index of the symbol whose cumulative interval [cum, cum+freq) contains dv."""
        lo, hi = 0, len(self._cum_arr) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cum_arr[mid] <= dv:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def decode(self, data, n_symbols):
        """Decode exactly n_symbols from `data`; the exact inverse of encode()."""
        pos = 0
        n = len(data)
        self._low = 0
        self._range = MASK
        code = 0
        for _ in range(4):  # prime the code word
            code = ((code << 8) | (data[pos] if pos < n else 0)) & MASK
            pos += 1
        total = self._total
        out = []
        for _ in range(n_symbols):
            self._range //= total
            dv = ((code - self._low) & MASK) // self._range
            if dv >= total:
                dv = total - 1
            idx = self._find(dv)
            sym = self._sym_arr[idx]
            self._low = (self._low + self._cum_arr[idx] * self._range) & MASK
            self._range = (self._range * self._freq[sym]) & MASK
            while self._settled() or (self._range < BOT and self._underflow()):
                code = ((code << 8) | (data[pos] if pos < n else 0)) & MASK
                pos += 1
                self._low = (self._low << 8) & MASK
                self._range = (self._range << 8) & MASK
            out.append(sym)
        return out


# ---- demo helpers (deterministic; RNG lives only here, never in the coder) -------------------
def _weighted(weights, n, seed):
    """Draw n symbols from range(len(weights)) with the given weights (seeded, reproducible)."""
    return random.Random(seed).choices(range(len(weights)), weights=weights, k=n)


def _counts(stream):
    c = {}
    for s in stream:
        c[s] = c.get(s, 0) + 1
    return c


def main():
    print("=" * 64)
    print("RANGE / ARITHMETIC ENTROPY CODER (static model, lossless)")
    print("=" * 64)

    # ---- 1. lossless round-trip on several stream shapes (THE test) ----
    urng = random.Random(12345)
    skew_w = [1000, 300, 150, 80, 40, 20, 8, 2]  # skewed 8-symbol model, H ~ 1.7 bits
    skewed = _weighted(skew_w, 40000, seed=7)
    uniform = [urng.randrange(16) for _ in range(40000)]
    allone = [5] * 4000
    short = [2, 7, 2, 2, 7, 3, 2]

    rt = []
    for name, stream in (
        ("skewed", skewed),
        ("uniform", uniform),
        ("all-one", allone),
        ("short", short),
    ):
        coder = RangeCoder(_counts(stream))
        back = coder.decode(coder.encode(stream), len(stream))
        ok = back == stream
        rt.append(ok)
        print(
            f"  round-trip {name:8s}: {'OK' if ok else 'FAIL'}  "
            f"(N={len(stream)}, alphabet={len(set(stream))})"
        )
    print(f"  all round-trips bit-exact: {all(rt)}")

    # ---- 2. near the entropy bound on the skewed stream (THE point) ----
    coder = RangeCoder(_counts(skewed))
    n = len(skewed)
    h = shannon_entropy(skewed)
    enc = coder.encode(skewed)
    achieved = 8 * len(enc) / n
    overhead = achieved - h
    print("-" * 64)
    print(f"  skewed stream: N={n}, H={h:.4f} bits/symbol")
    print(f"    achieved = {achieved:.4f} bits/symbol  (overhead {overhead:+.4f} bits/symbol)")
    print(f"    model entropy() = {coder.entropy():.4f} bits/symbol")

    # ---- 3. beats fixed-width AND byte-per-symbol on the skewed stream ----
    k = len(set(skewed))
    fixed_w = max(1, math.ceil(math.log2(k)))
    fixed_bits = n * fixed_w
    byte_bits = n * 8
    enc_bits = 8 * len(enc)
    print("-" * 64)
    print(f"  skewed alphabet k={k}: fixed width = {fixed_w} bits/symbol")
    print(f"    entropy-coded : {len(enc):>7,} B  ({enc_bits / n:.3f} bits/symbol)")
    print(
        f"    fixed-width   : {fixed_bits // 8:>7,} B  ({fixed_w}.000 bits/symbol)"
        f"  -> {fixed_bits / enc_bits:.2f}x smaller"
    )
    print(
        f"    byte/symbol   : {byte_bits // 8:>7,} B  (8.000 bits/symbol)"
        f"  -> {byte_bits / enc_bits:.2f}x smaller"
    )

    # ---- 4. no free lunch on a uniform source ----
    ku = 16
    ucoder = RangeCoder(_counts(uniform))
    uenc = ucoder.encode(uniform)
    u_achieved = 8 * len(uenc) / len(uniform)
    log2k = math.log2(ku)
    print("-" * 64)
    print(f"  uniform over k={ku}: log2(k) = {log2k:.4f} bits/symbol")
    print(f"    achieved = {u_achieved:.4f} bits/symbol  (entropy coding cannot beat log2(k))")

    # ---- 5. determinism: two fresh coders, same model + symbols -> identical bytes ----
    enc_a = RangeCoder(skew_w).encode(skewed)
    enc_b = RangeCoder(skew_w).encode(skewed)
    print("-" * 64)
    print(f"  determinism: two fresh coders -> identical bytes: {enc_a == enc_b}")
    print("=" * 64)

    # ---- asserts (none rigged: real streams, real entropy) ----
    assert all(rt), "a stream failed to round-trip bit-exactly"
    assert achieved < h + 0.1, f"entropy overhead too large ({achieved:.4f} vs H={h:.4f})"
    assert achieved > h - 0.02, f"cannot beat the entropy bound ({achieved:.4f} vs H={h:.4f})"
    assert enc_bits < fixed_bits, "must beat fixed-width coding on a skewed source"
    assert enc_bits < 0.8 * fixed_bits, "should be clearly below fixed-width"
    assert enc_bits < byte_bits, "must beat one byte per symbol"
    assert abs(u_achieved - log2k) < 0.1, "uniform must land near log2(k) (no free lunch)"
    assert enc_a == enc_b, "coder must be deterministic"
    print(
        f"self-check OK: lossless bit-exact round-trips; skewed near H "
        f"({achieved:.3f} vs {h:.3f} bits/symbol, overhead {overhead:+.3f}); "
        f"{fixed_bits / enc_bits:.2f}x < fixed-width, {byte_bits / enc_bits:.2f}x < byte/symbol; "
        f"uniform {u_achieved:.3f} ~ log2(k) {log2k:.3f}"
    )


if __name__ == "__main__":
    main()
