"""
Spiking self-organizing map (SOM) — an additive clustering backend for gesture discovery.

PROMOTED from research/ into the product (v0.84, additive). signal_loop.cluster_unknowns groups
recurring unknown windows by greedy single-link (Van Rossum distance) to grow the gesture vocabulary.
`som_cluster` offers an ALTERNATIVE: lay the unknown windows onto a Kohonen map (competitive WTA BMU +
lateral neighbourhood) and group them by best-matching unit — a topographic organization where similar
gestures land in neighbouring cells. Additive and opt-in: the default single-link path in signal_loop
is unchanged. Refs: Kohonen, "The self-organizing map", Proc. IEEE 78 (1990).

  python -m snn_data_lake.spiking_som        # (also run by run_selfchecks.py)
"""

import math
import random

from snn_data_lake.signal_loop import build_default_library


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


class SOM:
    """G x G competitive map; BMU (WTA winner) + decaying Gaussian-neighbourhood update."""

    def __init__(self, g, d, seed=0):
        self.g, self.d = g, d
        rng = random.Random(seed)
        self.w = [[rng.random() for _ in range(d)] for _ in range(g * g)]

    def _coord(self, i):
        return divmod(i, self.g)

    def bmu(self, x):
        return min(range(self.g * self.g), key=lambda i: _dist2(self.w[i], x))

    def train(self, data, epochs=15, lr0=0.5, sigma0=None):
        sigma0 = sigma0 or self.g / 2.0
        steps = epochs * len(data)
        t = 0
        for _ in range(epochs):
            for x in data:
                lr = lr0 * math.exp(-t / steps)
                sigma = sigma0 * math.exp(-t / steps)
                br, bc = self._coord(self.bmu(x))
                two_s2 = 2 * sigma * sigma
                for i in range(self.g * self.g):
                    r, c = self._coord(i)
                    h = math.exp(-((r - br) ** 2 + (c - bc) ** 2) / two_s2)
                    if h < 1e-3:
                        continue
                    wi = self.w[i]
                    for j in range(self.d):
                        wi[j] += lr * h * (x[j] - wi[j])
                t += 1


def som_cluster(vectors, g=4, epochs=15, seed=0):
    """Cluster vectors via a SOM: train, then group each vector under its BMU.
    Returns (som, {bmu_index: [vector, ...]}) — an alternative to single-link cluster_unknowns."""
    som = SOM(g, len(vectors[0]), seed=seed)
    som.train(vectors, epochs=epochs)
    groups = {}
    for v in vectors:
        groups.setdefault(som.bmu(v), []).append(v)
    return som, groups


def main():
    rng = random.Random(1)
    library = build_default_library()
    gestures = list(library)[:3]  # three distinct gestures to rediscover
    per = 30

    labelled = []
    for gi, lab in enumerate(gestures):
        for _ in range(per):
            jit = [min(1.0, max(0.0, v + rng.uniform(-0.08, 0.08))) for v in library[lab]]
            labelled.append((jit, gi))
    vectors = [v for v, _ in labelled]

    som, groups = som_cluster(vectors)

    # purity: each BMU group's majority gesture; fraction of vectors matching their group's majority
    bmu_of = {id(v): som.bmu(v) for v, _ in labelled}
    from collections import Counter

    votes = {}
    for v, gi in labelled:
        votes.setdefault(bmu_of[id(v)], Counter())[gi] += 1
    majority = {b: cnt.most_common(1)[0][0] for b, cnt in votes.items()}
    correct = sum(majority[bmu_of[id(v)]] == gi for v, gi in labelled)
    purity = correct / len(labelled)

    print("=" * 60)
    print("SPIKING SOM — additive clustering backend for gesture discovery")
    print("=" * 60)
    print(f"{len(gestures)} gestures x {per} unknown windows -> {som.g}x{som.g} map")
    print(f"active BMU cells: {len(groups)};  cluster purity: {purity:.0%}")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert purity >= 0.9, f"SOM clustering must separate the gestures ({purity:.2f})"
    assert len(groups) >= len(gestures), "distinct gestures must land in distinct BMU cells"
    assert som_cluster(vectors)[1].keys() == groups.keys(), "same seed -> identical clustering"
    print(
        f"self-check OK: SOM groups {len(gestures)} gestures at {purity:.0%} purity over {len(groups)} "
        f"BMU cells; additive backend, single-link cluster_unknowns unchanged; deterministic"
    )


if __name__ == "__main__":
    main()
