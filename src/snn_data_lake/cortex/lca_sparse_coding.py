"""
LCA sparse coding + sparse-coding matcher — lateral-inhibition recognition.

Rozell et al. (2008) LCA finds a SPARSE representation of a signal over a dictionary using only neuromorphic-
native operations: leaky integration, thresholding, and recurrent LATERAL INHIBITION. Each atom is a leaky
"neuron" driven by how well it matches the input (b = Phi^T x); active atoms inhibit their neighbours in
proportion to dictionary overlap (G - I), so correlated atoms compete and only a few survive -> a sparse code
x ~ Phi a (Loihi 2: arXiv:2606.08584).

PRODUCT integration (additive, opt-in, recognition-only): `LCAMatcher` uses the enrolled command references AS
the LCA dictionary and reads the command off the DOMINANT atom of the sparse code -- recognition by lateral
competition. It is an alternative matcher OPTION (cf. RankOrderMatcher / TempotronMatcher), not the live default,
enrolled on the same `build_default_library()`; it touches only recognition, never the arm / control / safety
path. On-thesis with sparse_distributed_memory / fly_hash / rank_order_coding.

  python -m snn_data_lake.cortex.lca_sparse_coding        # (also run by run_selfchecks.py)
"""

import random

from snn_data_lake.brainstem.signal_loop import build_default_library
from snn_data_lake.substrate._mathutils import dot


def _soft(x, lam):
    """Soft-threshold: shrink toward 0 by lam, clamp small values to exactly 0 (drives sparsity)."""
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def _unit(v):
    n = sum(c * c for c in v) ** 0.5 or 1.0
    return [c / n for c in v]


class LCA:
    """Locally Competitive Algorithm: sparse coding by leaky integration + lateral inhibition."""

    def __init__(self, dictionary, lam=0.1, dt=0.1):
        self.phi = dictionary  # list of M atoms, each a D-vector (columns of Phi)
        self.m = len(dictionary)
        self.lam = lam
        self.dt = dt
        # Gram matrix with zeroed diagonal = lateral-inhibition weights (overlap between atoms)
        self.g_off = [
            [0.0 if i == j else dot(dictionary[i], dictionary[j]) for j in range(self.m)]
            for i in range(self.m)
        ]

    def encode(self, x, steps=80):
        # v2.4 perf: default steps 300 -> 80 (~3.5x faster). Same algorithm, fewer LCA
        # iterations. Measured flat: LCAMatcher accuracy is 100% clean + jittered at
        # steps 300/150/80, and the sparse-code support/reconstruction asserts still hold.
        b = [dot(atom, x) for atom in self.phi]  # driving input: match of each atom to the signal
        u = [0.0] * self.m  # membrane potentials
        a = [0.0] * self.m  # active coefficients (thresholded)
        for _ in range(steps):
            for i in range(self.m):
                inhib = sum(self.g_off[i][j] * a[j] for j in range(self.m))
                u[i] += self.dt * (
                    b[i] - u[i] - inhib
                )  # leaky integrate: drive - leak - lateral inhibition
            a = [_soft(ui, self.lam) for ui in u]
        return a

    def reconstruct(self, a):
        d = len(self.phi[0])
        return [sum(a[i] * self.phi[i][k] for i in range(self.m)) for k in range(d)]


class LCAMatcher:
    """Opt-in sparse-coding recognizer: sparse-code a window over the enrolled references, read the command off
    the dominant atom. An alternative matcher OPTION (not the live default); recognition-only, no control path."""

    def __init__(self, library, lam=0.05):
        self.labels = list(library)
        self.lca = LCA([_unit(library[lab]) for lab in self.labels], lam=lam)

    def label(self, window):
        a = self.lca.encode(window)
        return self.labels[max(range(len(a)), key=lambda i: abs(a[i]))]


def main():
    rng = random.Random(7)

    # ---- 1. LCA sparse coding: lateral inhibition recovers the sparse code over an incoherent dictionary ----
    d, n_rand, max_coherence = 16, 8, 0.35
    phi = [[1.0 if k == i else 0.0 for k in range(d)] for i in range(d)]
    while len(phi) < d + n_rand:
        cand = _unit([rng.uniform(-1, 1) for _ in range(d)])
        if all(abs(dot(cand, atom)) < max_coherence for atom in phi):  # incoherent-frame condition
            phi.append(cand)
    support, coeffs = [1, 3, 6], [1.0, 0.8, 0.6]
    x = [0.0] * d
    for idx, c in zip(support, coeffs):
        for k in range(d):
            x[k] += c * phi[idx][k]
    lca = LCA(phi, lam=0.08)
    a = lca.encode(x)
    active = [i for i, ai in enumerate(a) if abs(ai) > 0.05]
    corr_active = [
        i for i in range(len(phi)) if abs(dot(phi[i], x)) > 0.05
    ]  # dense correlation baseline
    err = sum((x[k] - lca.reconstruct(a)[k]) ** 2 for k in range(d)) ** 0.5

    # ---- 2. PRODUCT integration: sparse-coding matcher over the enrolled command library ----
    library = build_default_library()
    labels = list(library)
    matcher = LCAMatcher(library)

    def jitter(v, n):
        return [min(1.0, max(0.0, xi + rng.uniform(-n, n))) for xi in v]

    clean = sum(matcher.label(library[lab]) == lab for lab in labels)
    trials, ok = 200, 0
    for i in range(trials):
        lab = labels[i % len(labels)]
        ok += matcher.label(jitter(library[lab], 0.10)) == lab
    acc = ok / trials

    print("=" * 60)
    print("LCA SPARSE CODING — sparse code + sparse-coding command matcher")
    print("=" * 60)
    print(
        f"sparse code : support {active} (true {support}); {len(active)}-sparse vs dense {len(corr_active)}, err {err:.3f}"
    )
    print(
        f"LCA matcher : {len(labels)} commands; clean {clean}/{len(labels)}, jittered {acc:.0%} over {trials}"
    )
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert set(active) == set(support), (
        f"LCA must recover the true 3-atom support ({active} vs {support})"
    )
    assert len(active) < len(corr_active) and err < 0.2, (
        f"LCA must be sparse + accurate ({len(active)}/{err:.3f})"
    )
    assert clean == len(labels), (
        f"sparse-coding matcher must recognize every clean reference ({clean}/{len(labels)})"
    )
    assert acc >= 0.9, f"sparse-coding matcher must be robust to jitter ({acc:.2f})"
    assert LCAMatcher(library).label(library[labels[0]]) == labels[0], "deterministic recognition"
    print(
        f"self-check OK: LCA recovers {len(active)}-sparse code (err {err:.3f}); LCAMatcher recognizes "
        f"{len(labels)} commands (clean {clean}/{len(labels)}, jittered {acc:.0%}); deterministic"
    )


if __name__ == "__main__":
    main()
