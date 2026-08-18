"""
Learned spiking attention gate -- a reticular/pulvinar attention gate over the real percept stream
whose synapses ADAPT by LOCAL three-factor plasticity (no backprop).

GAP this closes: brain._attend_real_stream pools the high-D encoder latent into the arm's sensory
channel with a FIXED quarter-sum divisive-normalization gate (a hardcoded 0.4*gain contrast term). The
gate weights never learn -- the brain cannot come to attend HARDER to feature bands that reliably predict
reward, nor bias its gate toward a top-down goal it is holding. This module makes that gate LEARNED.

The gate is brain-native, not a rate-ANN attention head:
  - bottom-up SALIENCE  -- per-band feature magnitude (what is loud right now).
  - bottom-up SURPRISE   -- positive deviation of salience from a running expectation (reticular/pulvinar
                            novelty), so an unexpected band draws the gate.
  - top-down GOAL bias   -- per-band similarity to a goal vector currently held (Reynolds-Heeger feature
                            attention: the goal multiplicatively boosts goal-similar bands).
  - LEARNED synapse w[b] -- one plastic reticular-gate weight per band; this is what adapts.
  - reticular GATE       -- divisive normalization of the per-band drive (Reynolds-Heeger normalization
                            model of attention): graded, competitive, bounded gain control in [0,1).
  - spiking WINNER       -- a LIF unit per band races on the gate; the band that spikes most is the
                            attended focus (a spiking winner-take-all readout).

LOCAL learning (learn(reward)): a THREE-FACTOR / reward-modulated Hebbian rule. The eligibility of each
band is its post-gate x pre-salience co-activation from the last attend(); reward (the third,
neuromodulatory factor) gates a baseline-subtracted update, so bands that were gated ON and salient when
reward arrived are AMPLIFIED and the rest relatively depressed. No global error signal, no backprop --
pre x post x neuromodulator, exactly the substrate the repo's R-STDP / three-factor modules use.

Representation-ONLY: this is a perception/attention gate. It produces a gated sensory representation and a
focus readout. It imports no arm, touches no brainstem/E-STOP/reflex path, and adds no actuator route.

  PYTHONPATH=src python -m snn_data_lake.cortex.learned_attention   # (asserting self-check)
"""

import zlib

import numpy as np

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "LearnedAttention.learn": (
        "LOCAL",
        "the module states it outright: LOCAL three-factor update, dw[b] = lr * reward * (elig[b] - "
        "mean_elig). Eligibility times a scalar reward, baseline-subtracted",
    ),
}


def _cos(a, b):
    """Cosine similarity clipped to [0,1]; a zero vector on either side yields 0 (no bias)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return max(0.0, float(np.dot(a, b) / (na * nb)))


class LearnedAttention:
    """A learned spiking reticular attention gate: divisive-normalized gain per band, plastic synapses."""

    def __init__(self, dim, n_bands=4, seed=0):
        if dim < n_bands:
            raise ValueError("dim must be >= n_bands")
        self.dim = int(dim)
        self.n_bands = int(n_bands)
        self.band_dim = max(1, self.dim // self.n_bands)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        # learned reticular-gate synapses (one per band). A tiny seeded jitter breaks symmetry
        # deterministically (same seed -> same init), without affecting the learned separation.
        self.w = np.ones(self.n_bands) + 1e-3 * self._rng.standard_normal(self.n_bands)
        self.w_min, self.w_max = 0.2, 6.0  # gate synapse bounds (stability)
        self.lr = 0.35  # three-factor learning rate
        self.sigma = 0.5  # divisive semi-saturation (Reynolds-Heeger)
        self.alpha_goal = 1.5  # top-down goal-bias strength
        self.alpha_surprise = 0.6  # bottom-up surprise strength
        self.exp_decay = 0.2  # salience-expectation EMA rate
        self.lif_gain = 5.0  # gate -> LIF input current scale (spiking WTA readout)
        self._exp = None  # per-band running salience expectation (lazy)
        self._gate = [0.0] * self.n_bands  # last per-band gate in [0,1]
        self._elig = np.zeros(self.n_bands)  # last eligibility (post-gate x pre-salience)
        self._focus_band = 0
        self._focus_strength = 0.0

    def _bands(self, x):
        """Split a length-dim vector into n_bands contiguous chunks (last band absorbs any remainder)."""
        out = []
        for b in range(self.n_bands):
            lo = b * self.band_dim
            hi = self.dim if b == self.n_bands - 1 else (b + 1) * self.band_dim
            out.append(np.asarray(x[lo:hi], dtype=float))
        return out

    def _band_of(self, i):
        """Which band input index i belongs to (matching _bands' contiguous layout)."""
        return min(i // self.band_dim, self.n_bands - 1)

    def _goal_gain(self, goal, feat_bands):
        """Per-band top-down bias in [0,1]. A dim-length goal -> per-band cosine to the features; an
        n_bands-length goal -> a direct per-band preference. None -> no bias."""
        if goal is None:
            return np.zeros(self.n_bands)
        g = np.asarray(goal, dtype=float).ravel()
        if g.size == self.n_bands and g.size != self.dim:
            gg = np.clip(g, 0.0, None)
            m = float(gg.max()) or 1.0
            return gg / m
        if g.size != self.dim:
            g = np.resize(g, self.dim)
        gb = self._bands(g)
        return np.array([_cos(gb[b], feat_bands[b]) for b in range(self.n_bands)])

    @staticmethod
    def _lif_rate(current, steps=20, tau=10.0, thr=1.0):
        """Deterministic leaky integrate-and-fire spike rate for a constant input current (no randomness).
        Monotonic + saturating in the current -- the spiking readout of the winner-take-all race."""
        v = 0.0
        spikes = 0
        for _ in range(steps):
            v += (-v + float(current)) / tau
            if v >= thr:
                spikes += 1
                v = 0.0
        return spikes / steps

    def attend(self, features, goal=None):
        """Gate the input by learned spiking attention. Returns the gated feature list (features * a
        per-band [0,1] gate). The gate = reticular divisive normalization of a per-band drive built from
        the LEARNED synapse * bottom-up salience * top-down goal similarity * bottom-up surprise."""
        x = np.asarray(features, dtype=float).ravel()
        if x.size < self.dim:
            x = np.resize(x, self.dim)
        elif x.size > self.dim:
            x = x[: self.dim]
        feat_bands = self._bands(x)
        sal = np.array([float(np.mean(np.abs(fb))) if fb.size else 0.0 for fb in feat_bands])
        # bottom-up surprise: positive deviation of salience from the running expectation, normalized
        if self._exp is None:
            self._exp = sal.copy()
        surprise = np.clip(sal - self._exp, 0.0, None)
        sscale = float(surprise.max()) or 1.0
        surprise = surprise / sscale
        self._exp = (1.0 - self.exp_decay) * self._exp + self.exp_decay * sal
        goal_gain = self._goal_gain(goal, feat_bands)
        # per-band drive: learned synapse * salience * top-down goal * bottom-up surprise
        drive = (
            self.w
            * sal
            * (1.0 + self.alpha_goal * goal_gain)
            * (1.0 + self.alpha_surprise * surprise)
        )
        drive = np.clip(drive, 0.0, None)
        # reticular gate: divisive normalization (Reynolds-Heeger gain control), graded and bounded
        denom = self.sigma + float(drive.sum())
        gate = drive / denom if denom > 0 else np.zeros(self.n_bands)
        # spiking winner-take-all: a LIF unit per band races on its gate; most spikes -> attended focus
        rates = np.array([self._lif_rate(self.lif_gain * float(g)) for g in gate])
        w_band = int(
            np.argmax(rates + 1e-9 * gate)
        )  # tie-break by the graded gate, then lowest index
        self._focus_band = w_band
        self._focus_strength = float(rates[w_band])
        self._gate = [float(g) for g in gate]
        self._elig = gate * sal  # post-gate x pre-salience eligibility (for three-factor learning)
        return [float(x[i] * gate[self._band_of(i)]) for i in range(self.dim)]

    def learn(self, reward):
        """LOCAL three-factor update: dw[b] = lr * reward * (elig[b] - mean_elig), from the last attend().
        Reward is the neuromodulatory third factor; the baseline subtraction makes the update competitive
        so bands co-active with reward are amplified and the rest relatively depressed. Weights clipped."""
        r = float(reward)
        base = float(self._elig.mean())
        dw = self.lr * r * (self._elig - base)
        self.w = np.clip(self.w + dw, self.w_min, self.w_max)

    @property
    def focus(self):
        """Current attended (winning) band and its spiking strength in [0,1]."""
        return (self._focus_band, round(self._focus_strength, 3))

    @property
    def gate(self):
        """Last per-band gate values in [0,1]."""
        return list(self._gate)

    def signature(self):
        """CRC32 fingerprint of the persisted gate weights (zlib.crc32, never builtin hash)."""
        return zlib.crc32(",".join(f"{v:.4f}" for v in self.w).encode("ascii"))


def main():
    dim, n_bands = 8, 4
    probe = [1.0] * dim  # uniform probe: gated output reveals the per-band gate directly

    # ---- (a) LOCAL learning: reward a specific band, it gets gated higher ----------------------------
    rewarded, unrewarded = 0, 2
    att = LearnedAttention(dim, n_bands=n_bands, seed=0)
    att.attend(probe)
    pre = att.gate
    train_feat = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # rewarded band 0 strongly present
    for _ in range(60):
        att.attend(train_feat)
        att.learn(1.0)  # reward co-occurs with attention to band 0
    att.attend(probe)
    post = att.gate
    att.attend(train_feat)
    focus_band, focus_strength = att.focus

    # ---- (b) TOP-DOWN goal biases the gate toward goal-similar features ------------------------------
    feats = [1.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.1, 0.1]  # band0 and band1 equally salient (0.5 each)
    goal = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # goal matches band1's pattern
    a_no = LearnedAttention(dim, n_bands=n_bands, seed=0)
    a_go = LearnedAttention(dim, n_bands=n_bands, seed=0)
    a_no.attend(feats)
    a_go.attend(feats, goal=goal)
    gate_no, gate_go = a_no.gate, a_go.gate

    # ---- (c) determinism: two fresh same-seed gates agree bit-for-bit --------------------------------
    d1 = LearnedAttention(dim, n_bands=n_bands, seed=7).attend(train_feat)
    d2 = LearnedAttention(dim, n_bands=n_bands, seed=7).attend(train_feat)

    print("=" * 62)
    print("LEARNED ATTENTION -- spiking reticular gate, local three-factor plasticity")
    print("=" * 62)
    print(f"bands={n_bands} dim={dim}; rewarded band={rewarded}, unrewarded band={unrewarded}")
    print(
        f"(a) learn : gate[{rewarded}] {pre[rewarded]:.3f} (pre) -> {post[rewarded]:.3f} (post); "
        f"unrewarded gate[{unrewarded}] {post[unrewarded]:.3f}"
    )
    print(f"    focus : winning band {focus_band} strength {focus_strength:.3f} (spiking WTA)")
    print(
        f"(b) goal  : band1 gate {gate_no[1]:.3f} (no goal) -> {gate_go[1]:.3f} (goal); "
        f"competing band0 {gate_go[0]:.3f}"
    )
    print(f"(c) det   : two seed=7 gates agree = {d1 == d2}; weight sig crc32={att.signature()}")
    print("=" * 62)

    # ---- self-check: deterministic asserts, ASCII-only ----
    assert all(0.0 <= g <= 1.0 for g in post), f"gate must stay in [0,1] ({post})"
    assert post[rewarded] > pre[rewarded] + 0.05, (
        f"reward must AMPLIFY the rewarded band's gate ({pre[rewarded]:.3f} -> {post[rewarded]:.3f})"
    )
    assert post[rewarded] > post[unrewarded], (
        f"learned gate must favor the rewarded over the unrewarded band "
        f"({post[rewarded]:.3f} vs {post[unrewarded]:.3f})"
    )
    assert focus_band == rewarded and focus_strength > 0.0, (
        f"spiking focus must land on the learned/present band ({focus_band}, {focus_strength:.3f})"
    )
    assert gate_go[1] > gate_no[1] + 0.05, (
        f"a top-down goal must bias the gate toward goal-similar features "
        f"({gate_no[1]:.3f} -> {gate_go[1]:.3f})"
    )
    assert gate_go[1] > gate_go[0], (
        f"under the goal, the goal-similar band must out-gate the competing band "
        f"({gate_go[1]:.3f} vs {gate_go[0]:.3f})"
    )
    assert d1 == d2, "same seed + same input must produce identical gated output (determinism)"
    print(
        f"self-check OK: learn gate[{rewarded}] {pre[rewarded]:.2f}->{post[rewarded]:.2f} "
        f"(unrewarded {post[unrewarded]:.2f}); goal biases band1 {gate_no[1]:.2f}->{gate_go[1]:.2f}; "
        f"focus band {focus_band}; deterministic"
    )


if __name__ == "__main__":
    main()
