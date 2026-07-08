"""
Sparse Distributed Memory (Kanerva) — a distributed associative-recall matcher option.

PROMOTED from research/ into the product (v0.83, additive). Kanerva's SDM (1988) stores data across
many random hard-location addresses, so recall degrades gracefully under a corrupted cue instead of
failing. `SDMMatcher` uses it as a recognition option: it writes each enrolled command's binarized
`encode_latency`-domain window to a hard-location set keyed by the window bits, with the command's
one-hot id as the stored data; a query window is recognized by reading back the distributed vote.
A distributed, noise-robust alternative to the Van Rossum / delta-rule / tempotron matchers — a
self-contained option, not the live default. Refs: Kanerva, "Sparse Distributed Memory" (MIT Press,
1988); Bricken & Pehlevan, "Attention Approximates SDM" (NeurIPS 2021, arXiv:2111.05498).

  python -m snn_data_lake.sparse_distributed_memory        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.signal_loop import build_default_library
from snn_data_lake.spike_preprocessing import N


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _binarize(vec, thr=0.30):
    return [1 if v >= thr else 0 for v in vec]


class SDM:
    """Kanerva SDM: n random addr_dim-bit hard locations, Hamming-radius r activation, data_dim counters."""

    def __init__(self, addr_dim, data_dim, n=1500, radius=22, seed=0):
        rng = random.Random(seed)
        self.n, self.r, self.data_dim = n, radius, data_dim
        self.addr = [[rng.randint(0, 1) for _ in range(addr_dim)] for _ in range(n)]
        self.ctr = [[0] * data_dim for _ in range(n)]

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
            return [0] * self.data_dim
        return [sum(self.ctr[i][j] for i in active) for j in range(self.data_dim)]   # summed votes


class SDMMatcher:
    """Recognize a window by distributed SDM recall of the enrolled command id."""

    def __init__(self, library, per_class=25, seed=0):
        self.labels = list(library)
        c = len(self.labels)
        self.mem = SDM(addr_dim=N, data_dim=c, seed=seed)
        rng = random.Random(seed)
        for li, lab in enumerate(self.labels):
            onehot = [1 if k == li else 0 for k in range(c)]
            for _ in range(per_class):
                jit = [min(1.0, max(0.0, v + rng.uniform(-0.08, 0.08))) for v in library[lab]]
                self.mem.write(_binarize(jit), onehot)

    def label(self, window):
        votes = self.mem.read(_binarize(window))
        return self.labels[max(range(len(self.labels)), key=lambda c: votes[c])]


def main():
    rng = random.Random(1)
    library = build_default_library()
    labels = list(library)
    matcher = SDMMatcher(library)

    def jitter(v, n):
        return [min(1.0, max(0.0, x + rng.uniform(-n, n))) for x in v]

    trials = 150
    ok = sum(matcher.label(jitter(library[labels[i % len(labels)]], 0.10)) == labels[i % len(labels)]
             for i in range(trials))
    acc = ok / trials
    clean = sum(matcher.label(library[lab]) == lab for lab in labels)

    print("=" * 60)
    print("SDM MATCHER — distributed associative-recall gesture recognition")
    print("=" * 60)
    print(f"{len(labels)} commands, {matcher.mem.n} hard locations, radius {matcher.mem.r}")
    print(f"recognition accuracy: {acc:.0%} on {trials} jittered windows; clean {clean}/{len(labels)}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert acc >= 0.9, f"SDM matcher must recognize the commands ({acc:.2f})"
    assert clean == len(labels), "every clean command must be recognized exactly"
    assert SDMMatcher(library).label(library[labels[0]]) == \
        SDMMatcher(library).label(library[labels[0]]), "deterministic"
    print(f"self-check OK: SDM matcher {acc:.0%} on jittered windows, {clean}/{len(labels)} clean; "
          f"distributed associative recall wired to the command library; deterministic")


if __name__ == "__main__":
    main()
