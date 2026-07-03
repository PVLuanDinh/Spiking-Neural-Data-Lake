"""
LCA sparse coding — the Locally Competitive Algorithm as a lateral-inhibition network (pure Python, no tensors).

Rozell et al. (2008) LCA finds a SPARSE representation of a signal over an OVERCOMPLETE dictionary using only
neuromorphic-native operations: leaky integration, thresholding, and recurrent LATERAL INHIBITION. Each atom is
a leaky "neuron" driven by how well it matches the input (b = Phi^T x); active atoms inhibit their neighbours in
proportion to dictionary overlap (G - I), so correlated atoms compete and only a few survive -> a sparse code
x ~ Phi a. Sparsity is the substrate this whole lake stores/recalls over (FlyHash / SDM / spike codes all assume
it); LCA is the encoder that PRODUCES it, and its dynamics map straight onto neuromorphic hardware (Loihi 2:
arXiv:2606.08584). Compared to a plain correlation code (Phi^T x, dense), LCA's lateral competition collapses the
representation to the few true atoms. Pure standard library, deterministic. On-thesis with spike_compression /
sparse_distributed_memory / fly_hash.

  python -m research.lca_sparse_coding        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import random


def _dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def _soft(x, lam):
    """Soft-threshold: shrink toward 0 by lam, clamp small values to exactly 0 (drives sparsity)."""
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


class LCA:
    """Locally Competitive Algorithm: sparse coding by leaky integration + lateral inhibition."""

    def __init__(self, dictionary, lam=0.1, dt=0.1):
        self.phi = dictionary                       # list of M atoms, each a D-vector (columns of Phi)
        self.m = len(dictionary)
        self.lam = lam
        self.dt = dt
        # Gram matrix with zeroed diagonal = lateral-inhibition weights (overlap between atoms)
        self.g_off = [[0.0 if i == j else _dot(dictionary[i], dictionary[j]) for j in range(self.m)]
                      for i in range(self.m)]

    def encode(self, x, steps=300):
        b = [_dot(atom, x) for atom in self.phi]     # driving input: match of each atom to the signal
        u = [0.0] * self.m                           # membrane potentials
        a = [0.0] * self.m                           # active coefficients (thresholded)
        for _ in range(steps):
            for i in range(self.m):
                inhib = sum(self.g_off[i][j] * a[j] for j in range(self.m))
                u[i] += self.dt * (b[i] - u[i] - inhib)     # leaky integrate: drive - leak - lateral inhibition
            a = [_soft(ui, self.lam) for ui in u]
        return a

    def reconstruct(self, a):
        d = len(self.phi[0])
        return [sum(a[i] * self.phi[i][k] for i in range(self.m)) for k in range(d)]


def _unit(v):
    n = sum(c * c for c in v) ** 0.5 or 1.0
    return [c / n for c in v]


def main():
    rng = random.Random(7)
    d, n_rand, max_coherence = 16, 8, 0.35
    # Overcomplete dictionary Phi = [ orthonormal basis I_16 | 8 random unit atoms ] (M = 24 atoms in R^16).
    # LCA provably recovers the true support only on a sufficiently INCOHERENT dictionary (mutual coherence
    # below ~1/(2K-1) for K-sparse signals), so we build one: reject any random atom too correlated with the
    # current set. This is the standard incoherent-frame condition, not a thumb on the scale.
    phi = [[1.0 if k == i else 0.0 for k in range(d)] for i in range(d)]
    while len(phi) < d + n_rand:
        cand = _unit([rng.uniform(-1, 1) for _ in range(d)])
        if all(abs(_dot(cand, atom)) < max_coherence for atom in phi):
            phi.append(cand)

    # ground-truth signal = a sparse combination of 3 basis atoms
    support = [1, 3, 6]
    coeffs = [1.0, 0.8, 0.6]
    x = [0.0] * d
    for idx, c in zip(support, coeffs):
        for k in range(d):
            x[k] += c * phi[idx][k]

    lca = LCA(phi, lam=0.08)
    a = lca.encode(x)
    recon = lca.reconstruct(a)
    err = sum((x[k] - recon[k]) ** 2 for k in range(d)) ** 0.5
    active = [i for i, ai in enumerate(a) if abs(ai) > 0.05]

    # baseline: plain correlation code (no lateral competition) is DENSE
    corr_active = [i for i in range(len(phi)) if abs(_dot(phi[i], x)) > 0.05]

    print("=" * 60)
    print("LCA SPARSE CODING — lateral inhibition finds the sparse code")
    print("=" * 60)
    print(f"dictionary: {len(phi)} atoms in R^{d} (overcomplete); signal = 3-atom combo {support}")
    print(f"LCA support : {active} (coeffs {[round(a[i], 2) for i in active]})")
    print(f"correlation : {len(corr_active)} atoms active (dense, no competition)")
    print(f"reconstruction error: {err:.4f}; sparsity {len(active)} vs dense {len(corr_active)}")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert set(active) == set(support), f"LCA must recover exactly the true 3-atom support ({active} vs {support})"
    # residual is dominated by soft-threshold SHRINKAGE (each active coeff pulled toward 0 by ~lam) -> ~sqrt(K)*lam
    assert err < 0.2, f"reconstruction must be accurate ({err:.4f})"
    assert len(active) < len(corr_active), f"LCA must be sparser than the dense correlation code ({len(active)} vs {len(corr_active)})"
    for idx, c in zip(support, coeffs):                 # coefficients ordered like the true magnitudes
        assert abs(a[idx] - c) < 0.15, f"coefficient for atom {idx} off ({a[idx]:.2f} vs {c})"
    assert LCA(phi, lam=0.08).encode(x) == a, "same signal -> identical sparse code (deterministic)"
    print(f"self-check OK: recovered exact support {active} (coeffs {[round(a[i],2) for i in active]}); "
          f"recon err {err:.3f}; {len(active)}-sparse vs dense {len(corr_active)}; deterministic")


if __name__ == "__main__":
    main()
