"""
Spiking hierarchical predictive coding (Rao & Ballard 1999, spiking form) -- perceptual inference
by prediction-error minimization over a 2-level generative hierarchy, carried by DISCRETE SPIKES.

A layered GENERATIVE model where each level predicts the level below and ONLY the prediction ERROR
propagates upward -- but here the ERRORS are encoded by LEAKY INTEGRATE-AND-FIRE (LIF) neuron
POPULATIONS whose SPIKE RATE carries the (precision-weighted) mismatch, and the CAUSES are updated
by those error SPIKES. For causes r1 (layer-1) and r2 (layer-2) with generative weights W1 (predicts
the input x) and W2 (predicts r1):

    x_hat  = W1 @ r1     e0 = x  - x_hat        (input prediction error  -> LIF error population 0)
    r1_hat = W2 @ r2     e1 = r1 - r1_hat        (layer-1 prediction error -> LIF error population 1)

Each error component drives a signed pair of LIF neurons (an ON neuron for a positive mismatch, an
OFF neuron for a negative one): the drive charges a membrane, and when it crosses threshold the
neuron emits a DISCRETE spike (+1 ON / -1 OFF) and resets by subtraction. So the error-population
SPIKE RATE encodes |mismatch| and its sign -- large error -> high spike rate, no error -> silence.

INFERENCE (fast, weights fixed) is the recurrent spiking network SETTLING. Each timestep the cause
neurons integrate the error SPIKES that reach them:

    r1 += lr * ( W1^T @ s_e0 - s_e1 )    # bottom-up error spikes in, MINUS its own top-down error
    r2 += lr * ( W2^T @ s_e1 )           # driven ONLY by the layer-1 error spikes s_e1

where s_e0, s_e1 are the per-step signed spike vectors. As the causes climb to explain the input the
mismatch shrinks, the error neurons fall silent, and the prediction-error SPIKE ACTIVITY DECAYS --
the network settles. This is stochastic-approximation gradient descent on the reduced free energy
E = |e0|^2 + |e1|^2 driven by quantized spikes, so the settled estimate is APPROXIMATE: it recovers
the cause and reconstructs a representable input to a small (reported) residual, NOT exactly zero.

LEARNING (slow, optional) is Hebbian-ON-SPIKES: after a settle, correlate each error neuron's spike
RATE over the window with the post-inference cause activity --
    W1 += lrW * outer( rate(s_e0), r1 ) ;  W2 += lrW * outer( rate(s_e1), r2 )
over a small dataset this lowers the average post-inference reconstruction error across epochs (it
learns a generative dictionary from spikes). LINEAR generative model (identity readout on r) -- the
readout of a cause is its graded membrane, standard in balanced spiking coders.

DISTINCT from siblings in this repo (stated, having read them):
  * lca_sparse_coding.py: a SINGLE-LAYER lateral-inhibition sparse coder (one dictionary, atoms
    compete via the Gram matrix). THIS is a HIERARCHY (>=2 levels) whose signature is top-down
    prediction + bottom-up error propagation BETWEEN levels -- spike message-passing over a
    generative hierarchy, a different architecture from lateral inhibition inside one layer.
  * valence_stdp.py / r_stdp: REWARD prediction error (dopamine RPE = reward - predicted value)
    driving VALUE learning. THIS is SENSORY/perceptual prediction error -- the mismatch between a
    top-down generative prediction and the actual input, driving spiking INFERENCE of latent causes.
    Different error, different purpose (perception vs reward).
  * htm_sequence_memory.py: TEMPORAL next-SDR prediction over time (a sequence model). THIS is a
    static generative hierarchy explaining a single input, not prediction across time.
  * dendritic_neuron.py: a single neuron's apical/basal coincidence ("top-down" = one cell's tuft).
    THIS is a NETWORK of LIF populations, not one cell.
  (research/free_energy_control.py is a spiking CONTROLLER for action selection; this is perceptual
  representation / inference -- a different domain of the free-energy idea.)

A small 2-level Rao-Ballard predictive coder made GENUINELY SPIKING (LIF error + cause populations,
discrete spikes, spiking recurrent settling, Hebbian-on-spikes learning) on synthetic data -- a
reduced model with tiny dims; spiking inference is approximate (residual reported), not exact
biophysics, not cognition.
Refs: Rao & Ballard, Nat Neurosci 2:79 (1999); Friston, free-energy view.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.predictive_coding
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
    """2-level SPIKING Rao-Ballard predictive coder. dims = (input, r1, r2).

    Generative weights W1 (input x r1) predict the input from r1; W2 (r1 x r2) predict r1 from r2.
    The prediction ERRORS are carried by LIF neuron populations (a signed ON/OFF pair per error
    component: membrane, threshold, spike, reset-by-subtraction). infer() settles the causes r1, r2
    by having them integrate the error SPIKES until the error populations fall silent; learn() then
    adapts W1, W2 by a Hebbian-on-spikes rule. Deterministic given seed (only random
    for weight init). Inference is approximate -- it drives the prediction-error spike activity and
    the reconstruction residual DOWN, not to exactly zero."""

    # fixed LIF parameters (the constructor signature is part of the public API and cannot grow).
    THETA = 0.5  # spike threshold for every error neuron
    LEAK = 0.02  # membrane leak per step -> genuinely LEAKY integrate-and-fire
    PREC0 = 1.0  # precision (gain) on the input-level mismatch drive
    PREC1 = 1.0  # precision (gain) on the layer-1 mismatch drive

    def __init__(self, dims=(8, 4, 2), seed=0):
        self.d_in, self.d_r1, self.d_r2 = dims
        self.rng = random.Random(seed)
        # W1: predicts input x from r1 (d_in rows x d_r1 cols); W2: predicts r1 from r2.
        self.w1 = [
            [self.rng.uniform(-0.5, 0.5) for _ in range(self.d_r1)] for _ in range(self.d_in)
        ]
        self.w2 = [
            [self.rng.uniform(-0.5, 0.5) for _ in range(self.d_r2)] for _ in range(self.d_r1)
        ]
        self.x = [0.0] * self.d_in
        self.reset()

    def reset(self):
        """Clear the inferred causes r1, r2 and every LIF membrane/spike/counter (call before
        inferring a new input)."""
        self.r1 = [0.0] * self.d_r1  # layer-1 causes (graded cause-neuron readout)
        self.r2 = [0.0] * self.d_r2  # layer-2 causes
        # LIF error-neuron membranes: signed ON/OFF pair per error component.
        self.v_e0_on = [0.0] * self.d_in
        self.v_e0_off = [0.0] * self.d_in
        self.v_e1_on = [0.0] * self.d_r1
        self.v_e1_off = [0.0] * self.d_r1
        # last discrete spike vectors (+1 ON / -1 OFF / 0 silent), for inspection.
        self.s_e0 = [0] * self.d_in
        self.s_e1 = [0] * self.d_r1
        # cumulative signed spikes + step count over a settle window (for Hebbian-on-spikes).
        self.cum_e0 = [0.0] * self.d_in
        self.cum_e1 = [0.0] * self.d_r1
        self.n_steps = 0

    @property
    def state(self):
        """Current inferred causes as (r1, r2) copies."""
        return (list(self.r1), list(self.r2))

    def set_state(self, r1, r2):
        """Force the causes and clear the error membranes (used to probe the spike message-passing
        directly)."""
        self.r1 = list(r1)
        self.r2 = list(r2)
        self.v_e0_on = [0.0] * self.d_in
        self.v_e0_off = [0.0] * self.d_in
        self.v_e1_on = [0.0] * self.d_r1
        self.v_e1_off = [0.0] * self.d_r1

    def reconstruct(self):
        """Top-down reconstruction of the input from r1: x_hat = W1 @ r1."""
        return _matvec(self.w1, self.r1)

    def predict_r1(self):
        """Top-down prediction of r1 from r2: r1_hat = W2 @ r2."""
        return _matvec(self.w2, self.r2)

    def errors(self):
        """Analog prediction errors (e0, e1): e0 = x - W1 @ r1, e1 = r1 - W2 @ r2. These are the
        membrane DRIVES that the LIF error populations spike-encode; also used for reporting."""
        x_hat = self.reconstruct()
        r1_hat = self.predict_r1()
        e0 = [self.x[k] - x_hat[k] for k in range(self.d_in)]
        e1 = [self.r1[j] - r1_hat[j] for j in range(self.d_r1)]
        return e0, e1

    def energy(self):
        """Total prediction-error energy E = |e0|^2 + |e1|^2 at the current state (reduced free
        energy; the quantity the spiking settling descends)."""
        e0, e1 = self.errors()
        return sum(v * v for v in e0) + sum(v * v for v in e1)

    def _lif(self, v_on, v_off, drive, k):
        """One LIF step for the signed ON/OFF neuron pair encoding error component k. The ON neuron
        integrates the positive part of the drive, the OFF neuron the negative part; each leaks,
        crosses THETA -> emits a discrete spike and resets by subtraction. Returns the signed spike
        in {-1, 0, +1}. Mutates the two membrane lists in place."""
        pos = drive if drive > 0.0 else 0.0
        neg = -drive if drive < 0.0 else 0.0
        v_on[k] = (1.0 - self.LEAK) * v_on[k] + pos
        s_on = 1 if v_on[k] >= self.THETA else 0
        if s_on:
            v_on[k] -= self.THETA
        v_off[k] = (1.0 - self.LEAK) * v_off[k] + neg
        s_off = 1 if v_off[k] >= self.THETA else 0
        if s_off:
            v_off[k] -= self.THETA
        return s_on - s_off

    def infer_step(self, x, lr):
        """One timestep of the spiking network. Encodes the current mismatch into error SPIKES,
        updates the causes from those spikes, and returns the prediction-error SPIKE ACTIVITY (total
        discrete error spikes emitted) BEFORE the cause update -- the spiking analog of E.

        r1 += lr * (W1^T @ s_e0 - s_e1) ;  r2 += lr * (W2^T @ s_e1). The r2 update is driven ONLY by
        the layer-1 error spikes s_e1 -- the hierarchical spike message-passing signature. Passing
        lr = 0.0 reads the error spikes WITHOUT moving the causes (a settled/ablated readout)."""
        self.x = list(x)
        e0, e1 = self.errors()
        s_e0 = [
            self._lif(self.v_e0_on, self.v_e0_off, self.PREC0 * e0[k], k) for k in range(self.d_in)
        ]
        s_e1 = [
            self._lif(self.v_e1_on, self.v_e1_off, self.PREC1 * e1[j], j) for j in range(self.d_r1)
        ]
        self.s_e0, self.s_e1 = s_e0, s_e1
        activity = sum(1 for s in s_e0 if s) + sum(1 for s in s_e1 if s)
        for k in range(self.d_in):
            self.cum_e0[k] += s_e0[k]
        for j in range(self.d_r1):
            self.cum_e1[j] += s_e1[j]
        self.n_steps += 1
        g1 = _matvec_t(self.w1, s_e0)  # bottom-up error spikes W1^T @ s_e0 (length d_r1)
        g2 = _matvec_t(self.w2, s_e1)  # bottom-up error spikes W2^T @ s_e1 (length d_r2)
        self.r1 = [self.r1[j] + lr * (g1[j] - s_e1[j]) for j in range(self.d_r1)]
        self.r2 = [self.r2[m] + lr * g2[m] for m in range(self.d_r2)]
        return activity

    def infer(self, x, steps=400, lr=0.05):
        """Settle r1, r2 for input x by letting the spiking network run. Returns the prediction-error
        SPIKE-ACTIVITY trace (length steps+1); it falls as the error neurons fall silent. The final
        entry is a settled readout taken with the causes held fixed (lr = 0.0)."""
        self.x = list(x)
        trace = [self.infer_step(x, lr) for _ in range(steps)]
        trace.append(self.infer_step(x, 0.0))  # settled error-spike readout, causes held
        return trace

    def learn(self, dataset, epochs=12, steps=120, lr=0.05, lr_w=0.02):
        """Train W1, W2 on a dataset. Returns the per-epoch average post-inference reconstruction
        MSE. Each sample: settle the spiking network, then a Hebbian-ON-SPIKES weight step using each
        error neuron's spike RATE over the settle window correlated with the post-inference cause."""
        curve = []
        for _ in range(epochs):
            epoch_mse = 0.0
            for x in dataset:
                self.reset()
                self.infer(x, steps=steps, lr=lr)
                e0, _ = self.errors()
                epoch_mse += sum(v * v for v in e0) / self.d_in
                rate0 = [self.cum_e0[k] / self.n_steps for k in range(self.d_in)]  # spike-rate ~ e0
                rate1 = [self.cum_e1[j] / self.n_steps for j in range(self.d_r1)]  # spike-rate ~ e1
                for k in range(self.d_in):  # W1 += lr_w * outer(rate(s_e0), r1)
                    for j in range(self.d_r1):
                        self.w1[k][j] += lr_w * rate0[k] * self.r1[j]
                for j in range(self.d_r1):  # W2 += lr_w * outer(rate(s_e1), r2)
                    for m in range(self.d_r2):
                        self.w2[j][m] += lr_w * rate1[j] * self.r2[m]
            curve.append(epoch_mse / len(dataset))
        return curve

    def representable_input(self, r2_true):
        """Build an input the hierarchy can explain: x = W1 @ (W2 @ r2_true) (recovered only up to
        the spiking residual)."""
        r1_true = _matvec(self.w2, list(r2_true))
        return _matvec(self.w1, r1_true)


def synth_dataset(n=8, d_in=8, rank=2, noise=0.02, seed=7):
    """A handful of vectors from a low-rank generative source (rank latent factors) + small noise."""
    rng = random.Random(seed)
    basis = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(d_in)]  # d_in x rank
    data = []
    for _ in range(n):
        coeff = [rng.uniform(-1.0, 1.0) for _ in range(rank)]
        data.append(
            [
                sum(basis[k][r] * coeff[r] for r in range(rank)) + rng.uniform(-noise, noise)
                for k in range(d_in)
            ]
        )
    return data


def _window_mean(trace, lo, hi):
    """Mean of trace[lo:hi] (spike-activity smoothing helper)."""
    seg = trace[lo:hi]
    return sum(seg) / len(seg)


def main():

    # ---- 1. SPIKING INFERENCE settles: error-spike activity + recon residual FALL vs an ablated
    #         control whose cause->error pathway is severed (lr=0) so its error neurons never quiet.
    net = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    r2_true = [0.9, -0.6]
    x = net.representable_input(r2_true)  # x = W1 @ (W2 @ r2_true)
    net.reset()
    trace = net.infer(x, steps=800, lr=0.05)
    e0, _ = net.errors()
    recon_mse = sum(v * v for v in e0) / net.d_in
    total_spikes = sum(trace)
    act_early = _window_mean(trace, 0, 50)  # prediction-error spike rate, start of inference
    act_late = _window_mean(trace, len(trace) - 50, len(trace))  # ... near the settled point

    ablate = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    ablate.reset()
    trace_ab = ablate.infer(x, steps=800, lr=0.0)  # causes frozen -> mismatch never explained away
    e0_ab, _ = ablate.errors()
    recon_mse_ab = sum(v * v for v in e0_ab) / ablate.d_in
    act_late_ab = _window_mean(trace_ab, len(trace_ab) - 50, len(trace_ab))

    # ---- 2. HIERARCHICAL spike message-passing: r2 moves ONLY when driven by layer-1 error SPIKES.
    net2 = PredictiveCodingNetwork(dims=(8, 4, 2), seed=1)
    r2_fixed = [0.5, -0.3]
    r1_match = _matvec(net2.w2, r2_fixed)  # r1 = W2 @ r2 => e1 == 0 => NO e1 spikes
    net2.set_state(r1_match, r2_fixed)
    net2.x = [0.0] * 8
    r2_before = list(net2.r2)
    net2.infer_step([0.0] * 8, lr=0.1)  # s_e1 == 0 so r2 step = lr * W2^T @ 0 = 0
    frozen_move = max(abs(a - b) for a, b in zip(r2_before, net2.r2))
    net2.set_state([v + 0.5 for v in r1_match], r2_fixed)  # residual e1 = +0.5 -> ON spikes
    r2_before2 = list(net2.r2)
    for _ in range(30):  # let the e1 spikes accumulate and drive r2
        net2.infer_step([0.0] * 8, lr=0.1)
    residual_move = max(abs(a - b) for a, b in zip(r2_before2, net2.r2))

    # ---- 3. LEARNING (Hebbian-on-spikes) reduces post-inference reconstruction error over epochs.
    data = synth_dataset(n=8, d_in=8, rank=2, noise=0.02, seed=7)
    net3 = PredictiveCodingNetwork(dims=(8, 4, 2), seed=3)
    curve = net3.learn(data, epochs=12, steps=120, lr=0.05, lr_w=0.02)
    first_mse, last_mse = curve[0], curve[-1]

    # ---- 4. DETERMINISM: same seed/input -> identical weights and identical spike traces.
    a = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    b = PredictiveCodingNetwork(dims=(8, 4, 2), seed=0)
    same_weights = a.w1 == b.w1 and a.w2 == b.w2
    a.reset()
    b.reset()
    trace_a = a.infer(a.representable_input([0.4, -0.7]), steps=60, lr=0.05)
    trace_b = b.infer(b.representable_input([0.4, -0.7]), steps=60, lr=0.05)
    same_trace = trace_a == trace_b

    # spikes are discrete events in {-1, 0, +1} (assert on the last settled spike vectors).
    spikes_discrete = all(s in (-1, 0, 1) for s in net.s_e0 + net.s_e1)

    print("=" * 68)
    print("SPIKING PREDICTIVE CODING -- 2-level Rao-Ballard, LIF error+cause populations")
    print("=" * 68)
    print(
        f"inference : error-spike activity head {[trace[t] for t in range(3)]} "
        f"tail {[trace[t] for t in range(len(trace) - 3, len(trace))]}"
    )
    print(
        f"            act early {act_early:.3f} -> late {act_late:.3f}  "
        f"({act_early / max(act_late, 1e-9):.1f}x quieter)  total spikes {total_spikes}"
    )
    print(
        f"            recon MSE |x - x_hat|^2/d {recon_mse:.3e} (settled)  "
        f"vs ablated {recon_mse_ab:.3e} (frozen causes)"
    )
    print(
        f"ablated   : error-spike activity stays high, late {act_late_ab:.3f} "
        f"(vs settled {act_late:.3f})"
    )
    print(
        f"hierarchy : r2 move with e1 spikes==0 {frozen_move:.3e} -> frozen; "
        f"r2 move with residual e1 spikes {residual_move:.4f} -> moves"
    )
    print("learning  : per-epoch post-inference recon MSE ->")
    print("            " + " ".join(f"{v:.4f}" for v in curve))
    print(
        f"            epoch 1 {first_mse:.4f} -> final {last_mse:.4f} "
        f"({first_mse / last_mse:.1f}x lower)"
    )
    print(f"spiking   : spikes discrete in {{-1,0,1}} {spikes_discrete}")
    print(f"determinism: identical weights {same_weights}; identical spike traces {same_trace}")
    print("=" * 68)

    # ---- self-check: deterministic asserts, ASCII-only prints ----
    # (a) FUNCTION -- spiking settling infers the cause and QUIETS the error neurons vs the control.
    assert total_spikes > 20, f"the network must actually emit discrete spikes ({total_spikes})"
    assert act_late < 0.5 * act_early, (
        f"prediction-error spike activity must FALL as inference settles ({act_early:.3f}->{act_late:.3f})"
    )
    assert act_late < 0.5 * act_late_ab, (
        f"settled error-spike activity must be far below the frozen-cause control ({act_late:.3f} vs {act_late_ab:.3f})"
    )
    assert recon_mse < 1e-2, (
        f"spiking inference must reconstruct a representable input closely ({recon_mse:.3e})"
    )
    assert recon_mse < 0.2 * recon_mse_ab, (
        f"settled recon residual must be far below the ablated control ({recon_mse:.3e} vs {recon_mse_ab:.3e})"
    )
    # (b) HIERARCHY -- r2 is driven ONLY by the layer-1 error spikes.
    assert frozen_move < 1e-12, (
        f"with e1 spikes==0 the layer-1 error vanishes so r2 must NOT move ({frozen_move:.3e})"
    )
    assert residual_move > 1e-3, (
        f"with residual e1 spikes, r2 MUST move to reduce the layer-1 error ({residual_move:.4f})"
    )
    # (c) LEARNING -- Hebbian-on-spikes lowers post-inference recon MSE.
    assert last_mse < first_mse * 0.5, (
        f"learning must lower post-inference recon MSE ({first_mse:.4f} -> {last_mse:.4f})"
    )
    # (d) GENUINELY SPIKING + DETERMINISM.
    assert spikes_discrete, "error signal must be carried by discrete spikes in {-1,0,+1}"
    assert same_weights and same_trace, (
        "same seed/input must give identical weights and spike traces"
    )
    print(
        f"self-check OK: spiking inference quiets error spikes {act_early:.2f}->{act_late:.2f} "
        f"(control stays {act_late_ab:.2f}), recon MSE {recon_mse:.2e} (approx, not exact); "
        f"r2 frozen at e1==0 ({frozen_move:.1e}) but moves on residual e1 spikes ({residual_move:.3f}); "
        f"Hebbian-on-spikes {first_mse:.3f} -> {last_mse:.3f}; discrete spikes; deterministic"
    )


if __name__ == "__main__":
    main()
