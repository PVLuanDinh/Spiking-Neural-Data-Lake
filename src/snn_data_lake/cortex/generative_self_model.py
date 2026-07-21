"""
Generative self-model -- a LEARNED spiking predictive-coding / e-prop forward model that predicts
the brain's OWN next self-signal vector from its current one; the prediction ERROR is a learned
SELF-SURPRISE. This replaces the hand-coded proxy aggregate in cortex/self_model.py -- that module's
SelfModel tracks signals with fixed-decimal EMAs and computes "confidence" as a hand-picked PRODUCT
of three uncertainty proxies (a fixed formula the designer chose, never learned). This module instead
LEARNS to predict itself: a real generative model of the self-signal trajectory, whose own prediction
error becomes the self-surprise signal, adapting with experience.

MECHANISM (reuses primitives from predictive_coding.py and eprop.py, does not reimplement them):

  1. LEAKY ELIGIBILITY TRACE (reasoning/eprop.py's readout-trace pattern). Each tick folds the
     current self-vector into a per-channel leaky trace:
         zbar <- KAPPA * zbar + self_vec
     -- exactly eprop.py's `zbar[j] = KAPPA*zbar[j] + z[t][j]`, here applied to the continuous
     self-signal instead of a binary spike (this repo's self-signals -- goal, valence, surprise,
     learning_progress, ... -- are graded values, not spikes; see cortex/self_model.py TRACKED).
     This trace is the synapse-local ELIGIBILITY: computed forward in time, no stored history.

  2. LEAKY-READOUT PREDICTION (eprop.py's non-recurrent readout: exact, no approximation needed).
         next_hat = W @ zbar
     -- a linear generative readout of the eligibility trace, structurally identical to eprop.py's
     `Wout` readout (`yr = KAPPA*yr_prev + sum(Wout[j]*z[t][j])`).

  3. SPIKING PREDICTION ERROR (predictive_coding.py's LIF error-population pattern). Once the true
     next self-vector arrives, the mismatch `error = next_vec - next_hat` drives a population of
     signed ON/OFF LEAKY INTEGRATE-AND-FIRE neurons -- predictive_coding.py's `_lif` cell, one pair
     per channel -- settled for a few steps with the error held as a constant drive (rate-coding the
     mismatch magnitude into discrete spikes, exactly as predictive_coding.py rate-codes its
     prediction-error populations). self_surprise is the settled population's spike rate: a
     genuinely SPIKING, non-negative measure of how wrong the prediction was -- not a hand-picked
     formula, a MEASURED one.

  4. LOCAL E-PROP WEIGHT UPDATE (eprop.py's exact readout-gradient rule, reused verbatim in form).
         dW[j][k] = lr * error[j] * zbar[k]
     -- broadcast error (the "third factor" / learning signal, eprop.py's `err[t]`) times the
     eligibility trace (eprop.py's `zbar[j]`), EXACTLY eprop.py's `dWout[j] += err[t] * zbar[j]`.
     This is a one-hop readout, so -- as eprop.py itself notes for its own readout weights -- the
     local rule IS the exact gradient of the squared-error loss w.r.t. W; no unrolled history, no
     backward pass, NO BPTT.

reset() clears only the TRANSIENT state (eligibility trace, tick counter) -- mirroring
predictive_coding.PredictiveCodingNetwork.reset(), which clears causes/membranes but leaves the
LEARNED weights (w1, w2) untouched. Here, `w` (the learned generative weights) is likewise left
untouched by reset() -- the brain does not forget what it has learned about predicting itself just
because a trial boundary passed.

API: GenerativeSelfModel(dim, seed=0).predict(self_vec) -> next_hat (list[float], length dim);
.observe(next_vec) -> self_surprise (float >= 0, and applies the local update); .reset().
Deterministic given seed (rng used only for weight init).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.generative_self_model
"""

import random


class GenerativeSelfModel:
    """Spiking predictive-coding / e-prop forward model of the brain's OWN self-signal trajectory.

    dim: length of the self-signal vector observed each tick (e.g. the tracked-signal vector from
    cortex/self_model.py). predict(self_vec) folds self_vec into a leaky eligibility trace and reads
    out a linear prediction of the NEXT self-vector. observe(next_vec) compares that prediction to
    the actual next self-vector, rate-codes the mismatch through a settled LIF error population into
    self_surprise (>=0), and applies a LOCAL e-prop weight update (broadcast error x eligibility
    trace, no BPTT). Deterministic given seed."""

    THETA = 0.15  # LIF spike threshold for the error population (predictive_coding.py style)
    LEAK = 0.10  # LIF membrane leak per settle step
    KAPPA = 0.5  # eligibility-trace / readout leak (eprop.py style)
    K_SETTLE = 6  # LIF settle steps used to rate-code one tick's error into self_surprise
    LR = 0.15  # local e-prop learning rate
    W_SCALE = 0.3  # weight init half-range

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance RNG, weight init ONLY (deterministic path)
        self.w = [
            [self.rng.uniform(-self.W_SCALE, self.W_SCALE) for _ in range(dim)] for _ in range(dim)
        ]
        self._init_transient()

    def _init_transient(self):
        self.zbar = [0.0] * self.dim  # leaky eligibility trace over the self-signal
        self._zbar_at_predict = [0.0] * self.dim
        self._last_hat = [0.0] * self.dim
        self.tick = 0

    def reset(self):
        """Clear the TRANSIENT state (eligibility trace, stored prediction, tick) -- mirrors
        predictive_coding.PredictiveCodingNetwork.reset(): the LEARNED weights `w` are left
        untouched, exactly as that module's reset() clears causes/membranes but keeps w1/w2."""
        self._init_transient()

    def predict(self, self_vec):
        """Fold self_vec into the leaky eligibility trace and return a linear readout prediction of
        the NEXT self-vector (eprop.py's non-recurrent readout structure: next_hat = W @ zbar)."""
        if len(self_vec) != self.dim:
            raise ValueError(f"self_vec width {len(self_vec)} != dim {self.dim}")
        self.zbar = [self.KAPPA * zb + float(x) for zb, x in zip(self.zbar, self_vec)]
        next_hat = [
            sum(self.w[j][k] * self.zbar[k] for k in range(self.dim)) for j in range(self.dim)
        ]
        self._zbar_at_predict = list(self.zbar)  # eligibility snapshot this prediction rests on
        self._last_hat = next_hat
        return list(next_hat)

    @staticmethod
    def _lif_step(v_on, v_off, drive, k, theta, leak):
        """One LIF settle step for error channel k (signed ON/OFF pair, the same cell used by
        predictive_coding.PredictiveCodingNetwork._lif): mutates v_on/v_off in place, returns the
        discrete spike in {-1, 0, +1}."""
        pos = drive if drive > 0.0 else 0.0
        neg = -drive if drive < 0.0 else 0.0
        v_on[k] = (1.0 - leak) * v_on[k] + pos
        s_on = 1 if v_on[k] >= theta else 0
        if s_on:
            v_on[k] -= theta
        v_off[k] = (1.0 - leak) * v_off[k] + neg
        s_off = 1 if v_off[k] >= theta else 0
        if s_off:
            v_off[k] -= theta
        return s_on - s_off

    def observe(self, next_vec):
        """Compare the actual next_vec to the last predict()ed next_hat; rate-code the mismatch
        through a FRESH settled LIF error population (self.K_SETTLE steps, membranes start at 0 each
        tick so self_surprise measures THIS transition only) into self_surprise (>=0); apply the
        local e-prop weight update (broadcast error x eligibility trace, no BPTT); advance the tick.
        Returns self_surprise."""
        if len(next_vec) != self.dim:
            raise ValueError(f"next_vec width {len(next_vec)} != dim {self.dim}")
        error = [float(a) - float(b) for a, b in zip(next_vec, self._last_hat)]

        v_on = [0.0] * self.dim
        v_off = [0.0] * self.dim
        total_spikes = 0
        for _ in range(self.K_SETTLE):
            for k in range(self.dim):
                total_spikes += abs(self._lif_step(v_on, v_off, error[k], k, self.THETA, self.LEAK))
        self_surprise = total_spikes / float(self.dim * self.K_SETTLE)

        # local e-prop update: dW[j][k] = lr * error[j] * zbar[k]  (eprop.py's readout-gradient rule)
        zb = self._zbar_at_predict
        for j in range(self.dim):
            ej = error[j]
            if ej == 0.0:
                continue
            row = self.w[j]
            for k in range(self.dim):
                row[k] += self.LR * ej * zb[k]

        self.tick += 1
        return self_surprise


# ============================ un-riggable self-check ==========================
def _rotation_trajectory(n_cycles, dim):
    """A repeating/structured self-signal trajectory: a period-`dim` one-hot-ish rotation, so
    `next_vec` at tick t is always `base[(t+1) % dim]` given `self_vec` at tick t is `base[t % dim]`
    -- a genuinely learnable next-state relationship, not noise. Returns [(self_vec, next_vec), ...]
    of length n_cycles*dim."""
    base = [[0.85 if i == k else 0.05 for i in range(dim)] for k in range(dim)]
    seq = []
    for _ in range(n_cycles):
        for k in range(dim):
            seq.append((base[k % dim], base[(k + 1) % dim]))
    return seq, base


def main():
    import sys

    dim = 4
    lines = []

    def emit(s):
        lines.append(s)

    emit("=" * 78)
    emit("GENERATIVE SELF-MODEL -- learned spiking predictive-coding / e-prop forward model")
    emit("=" * 78)

    # (1) LEARNING: self_surprise on a REPEATING/structured trajectory falls as the model learns
    #     to predict its own next self-signal. ------------------------------------------------
    traj, base = _rotation_trajectory(n_cycles=40, dim=dim)  # 160 ticks, period-4 rotation
    model = GenerativeSelfModel(dim, seed=0)
    surprises = []
    for cur, nxt in traj:
        model.predict(cur)
        surprises.append(model.observe(nxt))

    early = surprises[:16]
    late = surprises[-16:]
    mean_early = sum(early) / len(early)
    mean_late = sum(late) / len(late)
    emit("(1) LEARNING (repeating period-4 self-signal rotation)")
    emit(f"    self_surprise early (first 16 ticks) = {mean_early:.4f}")
    emit(f"    self_surprise late  (last 16 ticks)  = {mean_late:.4f}")
    assert mean_late < mean_early, (
        f"self_surprise must DECREASE as the model learns to predict itself ({mean_early:.4f} -> {mean_late:.4f})"
    )
    assert mean_late < 0.5 * mean_early, (
        f"self_surprise must fall substantially, not just marginally ({mean_early:.4f} -> {mean_late:.4f})"
    )

    # (2) NOVELTY: a genuinely mismatched transition (does not follow the learned rotation) raises
    #     self_surprise back above the learned baseline. ---------------------------------------
    novel_cur = base[0]  # a familiar state ...
    novel_next = [0.05, 0.05, 0.85, 0.85]  # ... but NOT followed by its learned successor
    model.predict(novel_cur)
    s_novel = model.observe(novel_next)
    emit("-" * 78)
    emit("(2) NOVELTY (perturbed transition, breaks the learned rotation)")
    emit(
        f"    self_surprise on the novel transition = {s_novel:.4f}  (learned late-baseline {mean_late:.4f})"
    )
    assert s_novel > mean_late, (
        f"self_surprise must RISE on a novel/perturbed transition ({s_novel:.4f} !> {mean_late:.4f})"
    )

    # (3) RESET keeps the LEARNED weights but clears transient state (predictive_coding.py contract):
    #     a fresh predict() right after reset() depends only on w (learned) and the just-given input,
    #     not on any residual trace from the training run. ---------------------------------------
    w_before = [row[:] for row in model.w]
    model.reset()
    w_after_reset = model.w
    probe = [0.4, 0.1, 0.1, 0.1]
    hat_after_reset = model.predict(probe)
    expected = [
        sum(model.w[j][k] * (model.KAPPA * 0.0 + probe[k]) for k in range(dim)) for j in range(dim)
    ]
    emit("-" * 78)
    emit("(3) RESET clears transient state, keeps learned weights")
    emit(f"    weights unchanged by reset(): {w_before == w_after_reset}")
    emit(
        f"    fresh predict() after reset matches W @ probe (zbar cleared): "
        f"{all(abs(a - b) < 1e-9 for a, b in zip(hat_after_reset, expected))}"
    )
    assert w_before == w_after_reset, "reset() must not touch the LEARNED weights"
    assert all(abs(a - b) < 1e-9 for a, b in zip(hat_after_reset, expected)), (
        "reset() must clear the eligibility trace (predict() right after must equal W @ probe)"
    )

    # (4) DETERMINISM: two fresh seed=0 models fed the identical trajectory produce identical
    #     self_surprise sequences and identical learned weights. -------------------------------
    ma = GenerativeSelfModel(dim, seed=0)
    mb = GenerativeSelfModel(dim, seed=0)
    seq_a, seq_b = [], []
    for cur, nxt in traj[:40]:
        ma.predict(cur)
        seq_a.append(ma.observe(nxt))
        mb.predict(cur)
        seq_b.append(mb.observe(nxt))
    det_ok = (seq_a == seq_b) and (ma.w == mb.w)
    emit("-" * 78)
    emit("(4) DETERMINISM")
    emit(
        f"    two seed=0 models, identical trajectory -> identical self_surprise + weights: {det_ok}"
    )
    assert det_ok, (
        "identical seed/trajectory must produce identical self_surprise sequence and weights"
    )

    # (5) SPIKING + NO BPTT + NON-NEGATIVE. --------------------------------------
    all_nonneg = all(s >= 0.0 for s in surprises + [s_novel])
    no_tensors = "numpy" not in sys.modules and "torch" not in sys.modules
    emit("-" * 78)
    emit("(5) SPIKING SELF-SURPRISE + NO BPTT")
    emit(f"    every self_surprise >= 0: {all_nonneg}; numpy/torch absent: {no_tensors}")
    assert all_nonneg, "self_surprise must be a non-negative scalar (it is a settled spike rate)"

    emit("=" * 78)
    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        f"self-check OK: self_surprise learns to predict its own next self-signal "
        f"({mean_early:.3f}->{mean_late:.3f} on a repeating trajectory, <0.5x), "
        f"rises to {s_novel:.3f} on a novel transition; reset() clears eligibility, keeps learned "
        f"weights; local e-prop update (broadcast error x eligibility trace, no BPTT); "
        f"deterministic"
    )


if __name__ == "__main__":
    main()
