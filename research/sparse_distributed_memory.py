"""
Sparse Distributed Memory (Kanerva) — distributed, noise-robust content-addressable store
(fresh research sweep). Pure-stdlib.

The store's associative recall (dense_associative_memory, spiking_storage_prototype) uses
Hopfield-style energies. Kanerva's SDM (1988) is the classical distributed alternative — and the
natural memory *behind* the v0.68 FlyHash index: a large set of random D-bit HARD-LOCATION
addresses, each holding D integer bit-counters. Write activates every hard location within Hamming
radius r of the address and nudges its counters by the data bits; read sums the activated counters
and thresholds -> a majority vote spread across the r-close locations. Storage is distributed, so
recall degrades gracefully under a corrupted cue instead of failing, and a 1-2 step read iteration
cleans the cue toward the nearest stored word. Refs: Kanerva, "Sparse Distributed Memory" (MIT
Press, 1988); Bricken & Pehlevan, "Attention Approximates Sparse Distributed Memory" (NeurIPS 2021,
arXiv:2111.05498).

  Run: python sparse_distributed_memory.py        # (also run by run_selfchecks.py)
"""
import random


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


class SDM:
    """Kanerva SDM over D-bit words: N random hard locations, Hamming-radius r activation."""

    def __init__(self, d=128, n=1000, radius=52, seed=0):
        self.d, self.n, self.r = d, n, radius
        rng = random.Random(seed)
        self.addr = [[rng.randint(0, 1) for _ in range(d)] for _ in range(n)]   # hard-location addresses
        self.ctr = [[0] * d for _ in range(n)]                                   # per-location bit counters

    def _active(self, address):
        return [i for i in range(self.n) if _hamming(self.addr[i], address) <= self.r]

    def write(self, address, data):
        for i in self._active(address):
            row = self.ctr[i]
            for j, bit in enumerate(data):
                row[j] += 1 if bit else -1

    def read(self, address):
        active = self._active(address)
        if not active:
            return [0] * self.d
        out = []
        for j in range(self.d):
            s = sum(self.ctr[i][j] for i in active)
            out.append(1 if s >= 0 else 0)        # majority vote across activated locations; ties -> 1
        return out

    def recall(self, cue, iters=2):
        """Iterated read — SDM converges toward the nearest stored word, cleaning a noisy cue."""
        x = cue
        for _ in range(iters):
            x = self.read(x)
        return x


def _word(rng, d):
    return [rng.randint(0, 1) for _ in range(d)]


def _corrupt(rng, w, frac):
    w = list(w)
    for i in rng.sample(range(len(w)), int(len(w) * frac)):
        w[i] ^= 1
    return w


def _bit_acc(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main():
    rng = random.Random(1)
    d, k = 128, 40
    mem = SDM(d=d, n=1000, radius=52, seed=0)
    words = [_word(rng, d) for _ in range(k)]
    for w in words:
        mem.write(w, w)                            # autoassociative store (address == data)

    n_active = len(mem._active(words[0]))
    clean = sum(_bit_acc(mem.read(w), w) for w in words) / k
    noisy = sum(_bit_acc(mem.recall(_corrupt(rng, w, 0.15)), w) for w in words) / k
    light = sum(_bit_acc(mem.recall(_corrupt(rng, w, 0.10)), w) for w in words) / k
    heavy = sum(_bit_acc(mem.recall(_corrupt(rng, w, 0.30)), w) for w in words) / k

    print("=" * 66)
    print("SPARSE DISTRIBUTED MEMORY (Kanerva) — distributed content-addressable store")
    print("=" * 66)
    print(f"D={d} bits, N={mem.n} hard locations, radius r={mem.r} (~{n_active} active/access)")
    print(f"stored {k} words autoassociatively (address == data)")
    print(f"clean recall bit-accuracy    : {clean:.1%}")
    print(f"recall from 15%-corrupted cue : {noisy:.1%}  (2 read iterations)")
    print(f"graceful degradation 10%->30% : {light:.1%} -> {heavy:.1%}")
    print("=" * 66)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert 0 < n_active < mem.n, f"activation must be sparse, not all/none (got {n_active})"
    assert clean > 0.99, f"clean recall at the exact address must be near-perfect ({clean:.3f})"
    assert noisy > 0.88, f"SDM must clean a 15%-corrupted cue back toward the word ({noisy:.3f})"
    assert light >= heavy, f"recall should degrade with corruption ({light:.3f} !>= {heavy:.3f})"
    assert SDM(seed=0).addr == SDM(seed=0).addr, "same seed -> identical hard-location addresses"
    print(f"self-check OK: {n_active} active/access; clean {clean:.0%}; 15%-cue -> {noisy:.0%}; "
          f"degrades {light:.0%}->{heavy:.0%}; deterministic")


if __name__ == "__main__":
    main()
