"""
Hierarchical predictive coding (Rao & Ballard 1999) -- perceptual inference by prediction-error
minimization over a 2-level generative hierarchy (pure stdlib, no tensors).

A layered GENERATIVE model: each level predicts the level below and ONLY the prediction ERROR
propagates upward. For representations r1 (layer-1 causes) and r2 (layer-2 causes) with generative
weights W1 (predicts the input x) and W2 (predicts r1):

    x_hat  = W1 @ r1     e0 = x  - x_hat        (input prediction error)
    r1_hat = W2 @ r2     e1 = r1 - r1_hat        (layer-1 prediction error)

INFERENCE (fast, weights fixed): descend the total error energy E = |e0|^2 + |e1|^2 --
    r1 += lr * ( W1^T @ e0 - e1 )    # bottom-up error in, MINUS its own top-down error
    r2 += lr * ( W2^T @ e1 )         # driven ONLY by the layer-1 error e1
iterate to settle; E -> ~0 for a representable input and x_hat reconstructs x. This is exact
gradient descent on the convex quadratic E (a reduced free-energy / Friston view), so E is
monotonically non-increasing across inference steps.

LEARNING (slow, optional): after inference a Hebbian/gradient step on the weights --
    W1 += lrW * outer(e0, r1) ;  W2 += lrW * outer(e1, r2)
over a small dataset lowers the average post-inference reconstruction error across epochs (it
learns a generative dictionary). LINEAR generative model (identity nonlinearity) -- chosen so the
representable-input test drives E to 0 exactly and stays verifiable; a rectifying nonlinearity on r
would be a drop-in monotone option but is not needed here.

DISTINCT from siblings in this repo (stated, having read them):
  * lca_sparse_coding.py: a SINGLE-LAYER lateral-inhibition sparse coder (one dictionary, atoms
    compete via the Gram matrix). THIS is a HIERARCHY (>=2 levels) whose signature is top-down
    prediction + bottom-up error propagation BETWEEN levels -- message-passing over a generative
    hierarchy, a different architecture from lateral inhibition inside one layer.
  * valence_stdp.py / r_stdp: REWARD prediction error (dopamine RPE = reward - predicted value)
    driving VALUE learning. THIS is SENSORY/perceptual prediction error -- the mismatch between a
    top-down generative prediction and the actual input, driving INFERENCE of latent causes.
    Different error, different purpose (perception vs reward).
  * htm_sequence_memory.py: TEMPORAL next-SDR prediction over time (a sequence model). THIS is a
    static generative hierarchy explaining a single input, not prediction across time.
  * dendritic_neuron.py: a single neuron's apical/basal coincidence ("top-down" = one cell's tuft).
    THIS is a NETWORK generative model, not one cell.
  (research/free_energy_control.py is a spiking CONTROLLER for action selection; this is perceptual
  representation / inference -- a different domain of the free-energy idea.)

A small linear (rectified-linear optional) 2-level Rao-Ballard predictive-coding network
on synthetic data with gradient inference + Hebbian weight learning -- a reduced model, NOT spiking
biophysics, no precision/attention weighting, tiny dims so the representable-input test is exact;
not cognition.
Refs: Rao & Ballard, Nat Neurosci 2:79 (1999); Friston, free-energy view.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.predictive_coding
"""
import random


def _matvec(m, v):
    """M @ v : m is rows x cols, v has length cols -> result length rows."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def _matvec_t(m, v):
    """M^T @ v : m is rows x cols, v has length rows -> result length cols."""
    rows, cols = len(m), len(m[0])
    return [sum(m[i][j] * v[i] for i in range(rows)) for j in range(cols)]


class PredictiveCodingNetwork:
    """2-level Rao-Ballard predictive coder. dims = (input, r1, r2).

    Generative weights W1 (input x r1) predict the input from r1; W2 (r1 x r2) predict r1 from r2.
    infer() settles the causes r1, r2 to minimize the total prediction-error energy E; learn()
    then adapts W1, W2. Deterministic given seed; pure stdlib (only random for init)."""

    def __init__(self, dims=(8, 4, 2), seed=0):
        self.d_in, self.d_r1, self.d_r2 = dims
        rng = random.Random(seed)
        # W1: predicts input x from r1 (d_in rows x d_r1 cols); W2: predicts r1 from r2.
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(self.d_r1)] for _ in range(self.d_in)]
        self.w2 = [[rng.uniform(-0.5, 0.5) for _ in range(self.d_r2)] for _ in range(self.d_r1)]
        self.x = [0.0] * self.d_in
        self.reset()

    def reset(self):
        """Clear the inferred causes r1, r2 to zero (call before inferring a new input)."""
        self.r1 = [0.0] * self.d_r1
        self.r2 = [0.0] * self.d_r2

    @property
    def state(self):
        """Current inferred causes as (r1, r2) copies."""
        return (list(self.r1), list(self.r2))

    def set_state(self, r1, r2):
        """Force the causes (used to probe the message-passing directly)."""
        self.r1 = list(r1)
        self.r2 = list(r2)

    def reconstruct(self):
        """Top-down reconstruction of the input from r1: x_hat = W1 @ r1."""
        return _matvec(self.w1, self.r1)

    def predict_r1(self):
        """Top-down prediction of r1 from r2: r1_hat = W2 @ r2."""
        return _matvec(self.w2, self.r2)

    def errors(self):
        """Prediction errors (e0, e1): e0 = x - W1 @ r1, e1 = r1 - W2 @ r2."""
        x_hat = self.reconstruct()
        r1_hat = self.predict_r1()
        e0 = [self.x[k] - x_hat[k] for k in range(self.d_in)]
        e1 = [self.r1[j] - r1_hat[j] for j in range(self.d_r1)]
        return e0, e1

    def energy(self):
        """Total prediction-error energy E = |e0|^2 + |e1|^2 at the current state."""
        e0, e1 = self.errors()
        return sum(v * v for v in e0) + sum(v * v for v in e1)

    def infer_step(self, x, lr):
        """One gradient-inference step. Returns E BEFORE the update (weights held fixed).

        r1 += lr * (W1^T @ e0 - e1) ;  r2 += lr * (W2^T @ e1). The r2 update is driven ONLY by the
        layer-1 error e1 -- the hierarchical message-passing signature."""
        self.x = list(x)
        e0, e1 = self.errors()
        pre_energy = sum(v * v for v in e0) + sum(v * v for v in e1)
        g1 = _matvec_t(self.w1, e0)        # bottom-up error W1^T @ e0 (length d_r1)
        g2 = _matvec_t(self.w2, e1)        # bottom-up error W2^T @ e1 (length d_r2)
        self.r1 = [self.r1[j] + lr * (g1[j] - e1[j]) for j in range(self.d_r1)]
        self.r2 = [self.r2[m] + lr * g2[m] for m in range(self.d_r2)]
        return pre_energy

    def infer(self, x, steps=400, lr=0.05):
        """Settle r1, r2 for input x by descending E. Returns the E trace (length steps+1)."""
        self.x = list(x)
        trace = [self.infer_step(x, lr) for _ in range(steps)]
        trace.append(self.energy())        # final settled E after the last update
        return trace

    def learn(self, dataset, epochs=12, steps=120, lr=0.05, lr_w=0.02):
        """Train W1, W2 on a dataset. Returns the per-epoch average post-inference reconstruction
        MSE. Each sample: infer causes, then Hebbian weight step on the residual errors."""
        curve = []
        for _ in range(epochs):
            epoch_mse = 0.0
            for x in dataset:
                self.reset()
                self.infer(x, steps=steps, lr=lr)
                e0, e1 = self.errors()
                epoch_mse += sum(v * v for v in e0) / self.d_in
                for k in range(self.d_in):                     # W1 += lr_w * outer(e0, r1)
                    for j in range(self.d_r1):
                        self.w1[k][j] += lr_w * e0[k] * self.r1[j]
                for j in range(self.d_r1):                     # W2 += lr_w * outer(e1, r2)
                    for m in range(self.d_r2):
                        self.w2[j][m] += lr_w * e1[j] * self.r2[m]
            curve.append(epoch_mse / len(dataset))
        return curve

    def representable_input(self, r2_true):
        """Build an input the hierarchy can explain EXACTLY: x = W1 @ (W2 @ r2_true)."""
        r1_true = _matvec(self.w2, list(r2_true))
        return _matvec(self.w1, r1_true)


def synth_dataset(n=8, d_in=8, rank=2, noise=0.02, seed=7):
    """A handful of vectors from a low-rank generative source (rank latent factors) + small noise."""
    rng = random.Random(seed)
    basis = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(d_in)]   # d_in x rank
    data = []
    for _ in range(n):
        coeff = [rng.uniform(-1.0, 1.0) for _ in range(rank)]
        data.append([sum(basis[k][r] * coeff[r] for r in range(rank)) + rng.uniform(-noise, noise)
                     for k in range(d_in)])
    return data


def main():
    # ---- 1. INFERENCE minimizes prediction error: E -> ~0 for a representable input ----
    net = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    r2_true = [0.9, -0.6]
    x = net.representable_input(r2_true)          # x = W1 @ (W2 @ r2_true) -> exactly representable
    net.reset()
    trace = net.infer(x, steps=800, lr=0.2)
    e0, e1 = net.errors()
    recon_mse = sum(v * v for v in e0) / net.d_in
    final_e = trace[-1]
    monotone = all(trace[t + 1] <= trace[t] + 1e-9 for t in range(len(trace) - 1))

    # ---- 2. HIERARCHICAL message-passing: r2 moves ONLY when driven by the layer-1 error e1 ----
    net2 = PredictiveCodingNetwork(dims=(8, 4, 2), seed=1)
    r2_fixed = [0.5, -0.3]
    net2.set_state([0.0] * 4, r2_fixed)
    r1_match = net2.predict_r1()                  # r1 = W2 @ r2 => e1 == 0 (matched top-down)
    net2.set_state(r1_match, r2_fixed)
    r2_before = list(net2.r2)
    net2.infer_step([0.0] * 8, lr=0.1)            # any input; r2 step = lr * W2^T @ e1 = 0
    frozen_move = max(abs(a - b) for a, b in zip(r2_before, net2.r2))
    net2.set_state([v + 0.5 for v in r1_match], r2_fixed)   # residual e1 = +0.5 per component
    r2_before2 = list(net2.r2)
    net2.infer_step([0.0] * 8, lr=0.1)
    residual_move = max(abs(a - b) for a, b in zip(r2_before2, net2.r2))

    # ---- 3. LEARNING reduces post-inference reconstruction error over epochs ----
    data = synth_dataset(n=8, d_in=8, rank=2, noise=0.02, seed=7)
    net3 = PredictiveCodingNetwork(dims=(8, 4, 2), seed=3)
    curve = net3.learn(data, epochs=12, steps=120, lr=0.05, lr_w=0.02)
    first_mse, last_mse = curve[0], curve[-1]

    # ---- 4. DETERMINISM: same seed/dataset -> identical weights and identical error traces ----
    a = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    b = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    same_weights = a.w1 == b.w1 and a.w2 == b.w2
    xa = a.representable_input([0.4, -0.7])
    trace_a = a.infer(xa, steps=60, lr=0.05)
    trace_b = b.infer(b.representable_input([0.4, -0.7]), steps=60, lr=0.05)
    same_trace = trace_a == trace_b

    print("=" * 68)
    print("PREDICTIVE CODING -- 2-level Rao-Ballard hierarchy (top-down predict, bottom-up error)")
    print("=" * 68)
    print(f"inference : E trace head {[round(v, 4) for v in trace[:3]]}")
    print(f"            E trace tail {[round(v, 8) for v in trace[-3:]]}")
    print(f"            final E {final_e:.3e}  recon MSE |x - x_hat|^2/d {recon_mse:.3e}  "
          f"monotone {monotone}")
    print(f"hierarchy : r2 move with e1==0 (matched top-down) {frozen_move:.3e} -> frozen; "
          f"r2 move with residual e1 {residual_move:.4f} -> moves")
    print("learning  : per-epoch post-inference recon MSE ->")
    print("            " + " ".join(f"{v:.4f}" for v in curve))
    print(f"            epoch 1 {first_mse:.4f} -> final {last_mse:.4f} "
          f"({first_mse / last_mse:.1f}x lower)")
    print(f"determinism: identical weights {same_weights}; identical E traces {same_trace}")
    print("=" * 68)

    # ---- self-check: deterministic asserts, zero-dep stdlib, ASCII-only prints ----
    assert monotone, "inference must be monotone: E non-increasing across steps"
    assert final_e < 1e-6, f"inference must drive E to ~0 for a representable input ({final_e:.3e})"
    assert recon_mse < 1e-6, f"reconstruction of a representable input must be near-exact ({recon_mse:.3e})"
    assert frozen_move < 1e-12, \
        f"with e1==0 the layer-1 error vanishes so r2 must NOT move ({frozen_move:.3e})"
    assert residual_move > 1e-3, \
        f"with a residual layer-1 error e1, r2 MUST move to reduce it ({residual_move:.4f})"
    assert last_mse < first_mse * 0.5, \
        f"learning must lower post-inference recon MSE ({first_mse:.4f} -> {last_mse:.4f})"
    assert all(curve[i + 1] <= curve[i] + 1e-9 for i in range(len(curve) - 1)), \
        "learning curve must be monotonically non-increasing"
    assert same_weights and same_trace, "same seed/dataset must give identical weights and traces"
    print(f"self-check OK: inference drives E -> {final_e:.2e} (recon MSE {recon_mse:.2e}, monotone); "
          f"r2 frozen at e1==0 ({frozen_move:.1e}) but moves on residual e1 ({residual_move:.3f}); "
          f"learning {first_mse:.3f} -> {last_mse:.3f} recon MSE; deterministic")


if __name__ == "__main__":
    main()
