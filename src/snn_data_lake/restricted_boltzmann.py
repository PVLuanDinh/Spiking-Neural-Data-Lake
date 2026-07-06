"""
Restricted Boltzmann machine trained by contrastive divergence -- unsupervised LEARNING of an
undirected energy-based generative model (pure stdlib, no tensors).

The RBM (Smolensky 1986; Hinton, "Training Products of Experts by Minimizing Contrastive
Divergence", Neural Comput 14:1771, 2002) is a bipartite Markov random field: visible units
v in {0,1}^nv and hidden units h in {0,1}^nh with weights W (nv x nh) and biases b_v, b_h. Its
joint is Boltzmann, p(v, h) prop exp(-E(v, h)), with energy

    E(v, h) = -b_v . v - b_h . h - v^T W h .

Because there are no visible-visible or hidden-hidden edges the conditionals FACTORIZE, so a whole
layer is sampled in one shot:
    p(h_j = 1 | v) = sigmoid(b_h_j + sum_i v_i W_ij)
    p(v_i = 1 | h) = sigmoid(b_v_i + sum_j W_ij h_j) .

LEARNING by contrastive divergence (CD-1), per data vector v0:
    positive phase   ph0 = p(h | v0)                      (hidden probs from the data)
    one Gibbs recon  h0 ~ ph0 -> v1 ~ p(v | h0) -> ph1 = p(h | v1)
    W   += lr * (outer(v0, ph0) - outer(v1, ph1))
    b_v += lr * (v0 - v1)
    b_h += lr * (ph0 - ph1)
CD approximates the gradient of the log-likelihood by contrasting the data statistics against a
one-step reconstruction, so training LOWERS the reconstruction error and pulls the free energy of
the data below that of random vectors. The free energy of a visible vector (hidden units summed out)
is
    F(v) = -b_v . v - sum_j softplus(b_h_j + sum_i v_i W_ij) ,   softplus(x) = log(1 + exp(x))
and lower F(v) means higher modelled probability p(v) prop exp(-F(v)).

DISTINCT from siblings in this repo (stated, having read them):
  * neural_sampling.py's BoltzmannSampler is a FIXED-weight single-site Gibbs sampler of a GIVEN,
    fully-visible Boltzmann/Ising joint p(z): its whole job is to MATCH a known analytic
    distribution -- nothing is learned, there are no hidden units, and there is no CD. THIS module
    is a bipartite visible/hidden RBM whose weights are LEARNED from data by contrastive divergence;
    its signature is that training lowers reconstruction error and drives the data's free energy
    below random. Different object (learned latent-variable generative model vs fixed known joint),
    different capability (unsupervised generative LEARNING vs sampling a given distribution).
  * predictive_coding.py is a DIRECTED generative hierarchy trained by gradient on prediction ERROR
    (top-down prediction, bottom-up error message-passing). An RBM is an UNDIRECTED energy-based
    model trained by CD -- a different model class AND a different learning rule.
  * research/hopfield_baseline.py and research/dense_associative_memory.py are DETERMINISTIC
    associative-recall attractors (one-shot energy-descent recall of stored patterns). THIS is a
    STOCHASTIC bipartite generative model that is SAMPLED (Gibbs) and LEARNED (CD), not one-shot
    recall of hand-stored memories.

ponytail: the ceiling is a small binary RBM trained by CD-1 on synthetic prototype patterns -- the
energy-based-generative-LEARNING level, NOT a deep belief net; no persistent-CD, no annealed
importance sampling for the true partition function; dims kept tiny so it trains in a couple
seconds and the signatures are stable at a fixed seed. Not cognition.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.restricted_boltzmann   # (also run by selfchecks)
"""
import math
import random


def _sigmoid(u):
    """Numerically stable logistic 1 / (1 + exp(-u))."""
    if u >= 0.0:
        return 1.0 / (1.0 + math.exp(-u))
    e = math.exp(u)
    return e / (1.0 + e)


def _softplus(x):
    """Numerically safe log(1 + exp(x))."""
    if x > 30.0:
        return x
    if x < -30.0:
        return 0.0
    return math.log1p(math.exp(x))


class RBM:
    """Restricted Boltzmann machine over v in {0,1}^nv and h in {0,1}^nh, learned by CD-1.

    Weights W (nv x nh) and biases b_v, b_h define the bipartite Boltzmann joint; the conditionals
    factorize so each layer is sampled in one shot. train() runs contrastive divergence; free_energy
    scores a visible vector (lower = higher modelled probability). Deterministic given seed: one
    random.Random(seed) drives both weight init and all CD sampling."""

    def __init__(self, n_visible, n_hidden, seed=0):
        self.nv = n_visible
        self.nh = n_hidden
        self.rng = random.Random(seed)
        # small symmetric-range init; biases start at zero (Hinton's practical default).
        self.W = [[self.rng.uniform(-0.1, 0.1) for _ in range(n_hidden)] for _ in range(n_visible)]
        self.b_v = [0.0] * n_visible
        self.b_h = [0.0] * n_hidden

    # ---- conditionals (the factorized layer distributions) -------------------------------------
    def hidden_probs(self, v):
        """p(h_j = 1 | v) for every hidden unit j."""
        return [_sigmoid(self.b_h[j] + sum(v[i] * self.W[i][j] for i in range(self.nv)))
                for j in range(self.nh)]

    def visible_probs(self, h):
        """p(v_i = 1 | h) for every visible unit i."""
        return [_sigmoid(self.b_v[i] + sum(self.W[i][j] * h[j] for j in range(self.nh)))
                for i in range(self.nv)]

    def _sample(self, probs):
        """Bernoulli sample a binary vector from per-unit probabilities (uses self.rng)."""
        return [1 if self.rng.random() < p else 0 for p in probs]

    def sample_hidden(self, v):
        """Return (hidden probs ph, binary hidden sample h) given a visible vector v."""
        ph = self.hidden_probs(v)
        return ph, self._sample(ph)

    def sample_visible(self, h):
        """Return (visible probs pv, binary visible sample v) given a hidden vector h."""
        pv = self.visible_probs(h)
        return pv, self._sample(pv)

    # ---- generation / denoising ----------------------------------------------------------------
    def reconstruct(self, v):
        """One mean-field pass v -> h -> v (deterministic): hidden probs, then thresholded visible.

        Uses probabilities (not samples) so a trained RBM cleans a corrupted input toward the
        nearest learned mode without sampling noise."""
        ph = self.hidden_probs(v)
        pv = self.visible_probs(ph)
        return [1 if p >= 0.5 else 0 for p in pv]

    def gibbs(self, v0, k):
        """k full steps of block Gibbs sampling starting from v0; returns a binary visible sample."""
        v = list(v0)
        for _ in range(k):
            _ph, h = self.sample_hidden(v)
            _pv, v = self.sample_visible(h)
        return v

    # ---- energy -------------------------------------------------------------------------------
    def free_energy(self, v):
        """Free energy F(v) = -b_v.v - sum_j softplus(b_h_j + sum_i v_i W_ij); lower F = higher p(v)."""
        vis = -sum(self.b_v[i] * v[i] for i in range(self.nv))
        hid = 0.0
        for j in range(self.nh):
            hid += _softplus(self.b_h[j] + sum(v[i] * self.W[i][j] for i in range(self.nv)))
        return vis - hid

    # ---- learning (contrastive divergence) -----------------------------------------------------
    def train(self, data, epochs=40, lr=0.1):
        """CD-1 training over `data` (list of binary vectors). Returns the per-epoch mean
        reconstruction error (mean squared error between each v0 and its reconstruction probs).

        Deterministic data order (no shuffle) so the same data/seed give an identical curve."""
        curve = []
        for _ in range(epochs):
            epoch_err = 0.0
            for v0 in data:
                ph0 = self.hidden_probs(v0)            # positive phase: p(h | data)
                h0 = self._sample(ph0)
                pv1 = self.visible_probs(h0)           # one Gibbs reconstruction of the visible
                v1 = self._sample(pv1)
                ph1 = self.hidden_probs(v1)            # negative phase: p(h | reconstruction)

                for i in range(self.nv):
                    vi0, vi1 = v0[i], v1[i]
                    row = self.W[i]
                    for j in range(self.nh):
                        row[j] += lr * (vi0 * ph0[j] - vi1 * ph1[j])
                    self.b_v[i] += lr * (vi0 - vi1)
                for j in range(self.nh):
                    self.b_h[j] += lr * (ph0[j] - ph1[j])

                epoch_err += sum((v0[i] - pv1[i]) ** 2 for i in range(self.nv)) / self.nv
            curve.append(epoch_err / len(data))
        return curve


# ---- helpers for the self-check ---------------------------------------------------------------
def _hamming(a, b):
    """Number of differing bits between two binary vectors."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _noisy_copies(proto, n, flips, rng):
    """n copies of a binary prototype, each with `flips` distinct random bits flipped."""
    out = []
    nv = len(proto)
    for _ in range(n):
        v = list(proto)
        for idx in rng.sample(range(nv), flips):
            v[idx] = 1 - v[idx]
        out.append(v)
    return out


def main():
    nv, nh = 16, 8
    data_rng = random.Random(123)                       # fixed dataset (separate from the model rng)

    # three well-separated 16-bit prototypes with clear structure.
    proto_a = [1] * 8 + [0] * 8                          # left block on
    proto_b = [0] * 8 + [1] * 8                          # right block on
    proto_c = [i % 2 for i in range(nv)]                 # alternating 0101...
    protos = [proto_a, proto_b, proto_c]

    data = []
    for p in protos:
        data.extend(_noisy_copies(p, n=8, flips=2, rng=data_rng))   # 24 noisy training vectors

    # ---- 1. CD learning lowers reconstruction error ----
    rbm = RBM(nv, nh, seed=0)
    curve = rbm.train(data, epochs=40, lr=0.1)
    first_err, last_err = curve[0], curve[-1]

    # ---- 2. the learned model scores data above random (lower free energy) ----
    f_data = sum(rbm.free_energy(v) for v in data) / len(data)
    rand_rng = random.Random(999)
    randoms = [[rand_rng.randint(0, 1) for _ in range(nv)] for _ in range(200)]
    f_rand = sum(rbm.free_energy(v) for v in randoms) / len(randoms)
    f_margin = f_rand - f_data                           # positive => data lower F => more probable

    # ---- 3. denoising: corrupt a prototype, reconstruct toward the true mode ----
    corrupt = list(proto_a)
    for idx in (2, 5, 9):                                # flip 3 bits (deterministic)
        corrupt[idx] = 1 - corrupt[idx]
    before = _hamming(corrupt, proto_a)
    cleaned = rbm.reconstruct(corrupt)
    after = _hamming(cleaned, proto_a)

    # ---- 4. determinism: same data/seed -> identical trained model + identical curve ----
    twin = RBM(nv, nh, seed=0)
    twin_curve = twin.train(data, epochs=40, lr=0.1)
    same_curve = twin_curve == curve
    same_weights = twin.W == rbm.W and twin.b_v == rbm.b_v and twin.b_h == rbm.b_h

    print("=" * 68)
    print("RESTRICTED BOLTZMANN MACHINE -- CD-1 learning of an energy-based model")
    print("=" * 68)
    print(f"recon-error curve head {[round(v, 4) for v in curve[:3]]}")
    print(f"recon-error curve tail {[round(v, 4) for v in curve[-3:]]}")
    print(f"  epoch 1 {first_err:.4f} -> final {last_err:.4f}  ({first_err / last_err:.1f}x lower)")
    print(f"free energy: F(data) {f_data:+.3f}   F(random) {f_rand:+.3f}   margin {f_margin:+.3f}")
    print("  (lower F(v) = higher modelled p(v); data must sit BELOW random)")
    print(f"denoise: corrupt proto_a (flip 3 bits) Hamming {before} -> reconstruct {after}")
    print(f"determinism: identical curve {same_curve}; identical weights {same_weights}")
    print("=" * 68)

    # ---- self-check (ponytail: deterministic asserts, zero-dep stdlib, ASCII-only prints) ----
    # 1. CD lowers reconstruction error clearly (the headline learning test).
    assert last_err < 0.5 * first_err, \
        f"CD must clearly lower reconstruction error ({first_err:.4f} -> {last_err:.4f})"
    # 2. the model learned the distribution: data free energy well below random.
    assert f_data < f_rand, \
        f"F(data) must be below F(random) ({f_data:.3f} vs {f_rand:.3f})"
    assert f_margin > 1.0, f"free-energy margin must be clear ({f_margin:.3f})"
    # 3. denoising pulls a corrupted pattern back toward the true learned mode.
    assert after < before, f"reconstruct must reduce distance to the true prototype ({before} -> {after})"
    assert after <= 1, f"reconstruction must land on (or next to) the true mode ({after})"
    # 4. determinism: same data/seed -> identical trained model and identical curve.
    assert same_curve and same_weights, "same data/seed must give an identical trained model"

    print(f"self-check OK: CD recon error {first_err:.3f} -> {last_err:.3f}; "
          f"F(data) {f_data:.2f} < F(random) {f_rand:.2f} (margin {f_margin:.2f}); "
          f"denoise Hamming {before} -> {after}; deterministic")


if __name__ == "__main__":
    main()
