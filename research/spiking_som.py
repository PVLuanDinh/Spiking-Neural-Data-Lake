"""
Spiking self-organizing map (SOM) — unsupervised topographic organization of the store
(fresh research sweep). Pure-stdlib.

FlyHash (v0.68) indexes the store and SDM (v0.69) stores it; neither ORGANIZES it. A SOM is
competitive Hebbian learning built from the repo's own primitives — a winner-take-all "BMU" spike
plus a lateral neighbourhood update — that lays stored patterns onto a 2-D map so nearby map units
answer to similar inputs. That gives the lake an unsupervised index / cluster structure the drift
monitor and interpreter can read (which region a live window lands in). Refs: Kohonen, "The
self-organizing map", Proc. IEEE 78 (1990); spiking/neuromorphic SOM variants (Rumbell et al.,
Neural Networks 2014).

  Run: python spiking_som.py        # (also run by run_selfchecks.py)
"""
import math
import random
from collections import Counter


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


class SOM:
    """G x G competitive map; BMU (WTA winner) + decaying Gaussian-neighbourhood update."""

    def __init__(self, g, d, seed=0):
        self.g, self.d = g, d
        rng = random.Random(seed)
        self.w = [[rng.random() for _ in range(d)] for _ in range(g * g)]   # a weight vector per unit

    def _coord(self, i):
        return divmod(i, self.g)

    def bmu(self, x):
        return min(range(self.g * self.g), key=lambda i: _dist2(self.w[i], x))   # winner-take-all

    def train(self, data, epochs=15, lr0=0.5, sigma0=None):
        sigma0 = sigma0 or self.g / 2.0
        steps = epochs * len(data)
        t = 0
        for _ in range(epochs):
            for x in data:
                lr = lr0 * math.exp(-t / steps)                  # anneal learning rate
                sigma = sigma0 * math.exp(-t / steps)            # shrink the neighbourhood
                br, bc = self._coord(self.bmu(x))
                two_s2 = 2 * sigma * sigma
                for i in range(self.g * self.g):
                    r, c = self._coord(i)
                    h = math.exp(-((r - br) ** 2 + (c - bc) ** 2) / two_s2)   # lateral neighbourhood
                    if h < 1e-3:
                        continue
                    wi = self.w[i]
                    for j in range(self.d):
                        wi[j] += lr * h * (x[j] - wi[j])
                t += 1

    def quant_error(self, data):
        return sum(math.sqrt(_dist2(self.w[self.bmu(x)], x)) for x in data) / len(data)


def _clusters(rng, d, n_clusters=3, per=40, spread=0.06):
    centers = [[rng.random() for _ in range(d)] for _ in range(n_clusters)]
    data, labels = [], []
    for c, ctr in enumerate(centers):
        for _ in range(per):
            data.append([min(1.0, max(0.0, v + rng.uniform(-spread, spread))) for v in ctr])
            labels.append(c)
    return data, labels


def _topographic_ratio(som):
    """Mean weight-distance between grid-NEIGHBOUR units vs RANDOM unit pairs. A well-ordered map
    has neighbours much more similar than random pairs (ratio << 1)."""
    g = som.g
    neigh = []
    for i in range(g * g):
        r, c = som._coord(i)
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if nr < g and nc < g:
                neigh.append(math.sqrt(_dist2(som.w[i], som.w[nr * g + nc])))
    rng = random.Random(7)
    rand = [math.sqrt(_dist2(som.w[rng.randrange(g * g)], som.w[rng.randrange(g * g)])) for _ in range(200)]
    return (sum(neigh) / len(neigh)) / (sum(rand) / len(rand))


def main():
    rng = random.Random(1)
    d, g = 6, 5
    data, labels = _clusters(rng, d)
    som = SOM(g, d, seed=0)

    qe0 = som.quant_error(data)
    som.train(data, epochs=15)
    qe1 = som.quant_error(data)
    topo = _topographic_ratio(som)

    # class organization: each map unit's majority label -> purity of the BMU assignment
    votes = {}
    for x, y in zip(data, labels):
        votes.setdefault(som.bmu(x), Counter())[y] += 1
    unit_label = {u: cnt.most_common(1)[0][0] for u, cnt in votes.items()}
    purity = sum(unit_label[som.bmu(x)] == y for x, y in zip(data, labels)) / len(data)

    print("=" * 62)
    print("SPIKING SELF-ORGANIZING MAP — unsupervised topographic store index")
    print("=" * 62)
    print(f"{g}x{g} map, d={d}; {len(data)} points / {len(set(labels))} clusters")
    print(f"quantization error : {qe0:.3f} -> {qe1:.3f}")
    print(f"topographic ratio  : {topo:.2f}  (neighbour/random weight-dist; <1 = ordered)")
    print(f"BMU class purity   : {purity:.1%}")
    print("=" * 62)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert qe1 < qe0 * 0.6, f"training must sharpen the map (QE {qe0:.3f} -> {qe1:.3f})"
    assert topo < 0.8, f"map must be topographically ordered (neighbours similar), ratio {topo:.2f}"
    assert purity > 0.9, f"BMU regions must be class-pure on separable clusters ({purity:.2f})"
    assert SOM(g, d, seed=0).w == SOM(g, d, seed=0).w, "same seed -> identical init (deterministic)"
    print(f"self-check OK: QE {qe0:.2f}->{qe1:.2f}; topographic ratio {topo:.2f}; purity {purity:.0%}; deterministic")


if __name__ == "__main__":
    main()
