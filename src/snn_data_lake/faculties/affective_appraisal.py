"""
affective_appraisal -- a LEARNED multi-dimensional affective appraisal net (fixes: appraisal is
not learned).

Generalizes faculties/fear_conditioning.py's single CS-US three-factor Hebbian BLA readout from ONE
scalar (threat) to a full (valence, arousal) appraisal of an arbitrary latency-encoded feature
vector (percept + context, concatenated by the caller before it reaches this module). Where
fear_conditioning binds a CS to one aversive teaching signal (the US) via
`w_exc[i] += lr * US * cs_active[i] * bound`, this module binds a (percept + context) pattern to a
TARGET (valence, arousal) via the reward-prediction-error rule plasticity/valence_stdp.py uses
for its scalar valence (`delta = reward - predicted; w += lr * delta * activity`) -- but read out
through spiking LIF integrate-and-fire pools (fear_conditioning's BLA circuit) instead of one
linear unit, and with THREE pools instead of one.

Three pools, one per read-out channel, each a leaky integrate-and-fire neuron driven by the SAME
latency-encoded input over its own plastic weights (exactly fear_conditioning's `_bla_spikes`
mechanism, generalized from one channel to three):
  w_pos -- approach / positive-valence drive
  w_neg -- avoid / negative-valence drive     (valence = saturate(w_pos) - saturate(w_neg), [-1,1])
  w_aro -- arousal drive                      (arousal = saturate(w_aro), [0,1])

learn(features, target_valence, target_arousal, neuromod=1.0) computes the appraisal error (target
- current appraisal -- an RPE exactly like valence_stdp's `dopamine = reward - predicted`) and
applies a three-factor Hebbian/anti-Hebbian update GATED by that error and by `neuromod`, a scalar
standing in for a diffuse modulator (pass neuromodulation/cortisol.Cortisol.learn_rate_scale(signed)
as `neuromod` to make aversive appraisals learn faster under stress, mirroring cortisol.py's own
worked example of scaling a learning rate):
  err_v = target_valence - valence
  if err_v > 0: potentiate w_pos on the ACTIVE features, decay w_neg toward 0  (need MORE approach)
  if err_v < 0: potentiate w_neg on the ACTIVE features, decay w_pos toward 0  (need MORE avoid)
  (the same pattern, single-pool, for arousal: it only ever needs to be raised or lowered)
Every update touches only features ACTIVE in the presented pattern (Hebbian pre-synaptic gating,
the third factor is the error x neuromod product) and is bounded in [0, W_MAX] via the same soft
`(1 - w/W_MAX)` / `(w/W_MAX)` saturation shape fear_conditioning uses. This is entirely LOCAL
three-factor plasticity: no backprop, no gradient through the readout, no global loss. A context
that never co-occurs with a teaching signal keeps zero weight on its own (disjoint) features and
stays at the pre-training (neutral) appraisal -- the same un-riggable "unpaired CS" control
fear_conditioning uses, generalized to two dimensions (see self-check).

step(decay) applies slow homeostatic weight decay (forgetting) -- call it between episodes, not
after every appraise(); learn() already does its own local decay on the antagonist pool.

Honest framing: this is a learned input -> (valence, arousal) FUNCTION -- a spiking three-factor
associative readout. It does not feel valence or arousal, and nothing here experiences anything.

Deterministic: the class holds no RNG of its own (appraise/learn/step are pure functions of their
inputs and current weights); the training PROTOCOL in the self-check seeds its own local
random.Random(seed), so two runs with the same seed are identical.

  python -m snn_data_lake.faculties.affective_appraisal
"""

import random

from snn_data_lake.sensorimotor.spike_preprocessing import T, encode_latency

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "AffectiveAppraisal.learn": (
        "LOCAL",
        "three-factor update: a teaching signal (target - current appraisal) that is an RPE, gated "
        "by a diffuse neuromodulator. Nothing is propagated backward through a network, so it owes "
        "no cosine",
    ),
    "AffectiveAppraisal._potentiate": (
        "LOCAL",
        "bounded Hebbian potentiation on the ACTIVE features only, the rate already folding in the "
        "three-factor gate. Coincidence-driven and synapse-local",
    ),
    "AffectiveAppraisal._decay": (
        "LOCAL",
        "bounded ANTI-Hebbian decay toward 0 on the active features only. The mirror half of the same "
        "local rule -- depression is plasticity, not its absence",
    ),
    "AffectiveAppraisal.step": (
        "LOCAL",
        "slow homeostatic forgetting: every weight relaxes toward 0 by a fixed factor. Each weight's "
        "own value decides its change",
    ),
}

# ---- LIF / plasticity constants (hand-tuned; same style as fear_conditioning.py) --------------
LEAK = 0.90  # LIF membrane leak per step (matches fear_conditioning's BLA)
TH = 1.0  # LIF firing threshold (subtractive reset)
SAT = 16.0  # pool spike count that saturates its readout to 1.0
LR = 0.10  # three-factor learning rate
W_MAX = 1.0  # soft weight bound


def _by_step(vec):
    """Latency-encode a feature vector and bucket its spikes by timestep (same helper as
    fear_conditioning._by_step). Returns (events_by_step, active_features)."""
    ev = encode_latency(vec, T)
    by = [[] for _ in range(T)]
    for t, i in ev:
        by[t].append(i)
    return by, sorted({i for _, i in ev})


def _lif_spikes(by_step, w):
    """Leaky integrate-and-fire over T steps, driven by plastic weights `w` on whichever features
    are active at each step (identical mechanism to fear_conditioning.Amygdala._bla_spikes,
    generalized to an arbitrary readout pool). Returns the output spike count."""
    v = 0.0
    spikes = 0
    for t in range(T):
        cur = 0.0
        for i in by_step[t]:
            cur += w[i]
        v = v * LEAK + cur
        if v >= TH:
            spikes += 1
            v -= TH
    return spikes


class AffectiveAppraisal:
    """Learned multi-dimensional appraisal: a latency-encoded (percept + context) feature vector ->
    (valence in [-1,1], arousal in [0,1]), via three spiking LIF readout pools trained online by
    three-factor (pre-synaptic activity x error x neuromodulator) local Hebbian/anti-Hebbian
    plasticity. No backprop, no gradient -- generalizes fear_conditioning's single CS-US BLA readout
    to a full dimensional appraisal."""

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.seed = seed  # kept for API symmetry with the rest of amygdala/*; unused by
        self.w_pos = [0.0] * dim  # appraise()/learn(), which are deterministic pure functions of
        self.w_neg = [0.0] * dim  # (weights, input) -- see module docstring.
        self.w_aro = [0.0] * dim

    def appraise(self, features):
        """(valence, arousal) readout for a feature vector of length `dim` -- no learning."""
        by, _ = _by_step(features)
        pos = min(1.0, _lif_spikes(by, self.w_pos) / SAT)
        neg = min(1.0, _lif_spikes(by, self.w_neg) / SAT)
        aro = min(1.0, _lif_spikes(by, self.w_aro) / SAT)
        valence = max(-1.0, min(1.0, pos - neg))
        arousal = max(0.0, min(1.0, aro))
        return valence, arousal

    def learn(self, features, target_valence, target_arousal, neuromod=1.0):
        """Three-factor update: teaching signal = (target - current appraisal), an RPE exactly like
        valence_stdp's dopamine, gated by `neuromod` (a diffuse-modulator scalar, e.g.
        cortisol.Cortisol.learn_rate_scale(...); floored at 0 so a modulator can only scale learning,
        never invert it). Potentiates whichever pool needs to grow, on the ACTIVE features of
        `features` only, and decays its antagonist pool toward 0 on those same features -- pure
        local Hebbian/anti-Hebbian, no backprop. Returns (err_valence, err_arousal), the appraisal
        error BEFORE this update (the RPE)."""
        by, active = _by_step(features)
        valence, arousal = self.appraise(features)
        err_v = target_valence - valence
        err_a = target_arousal - arousal
        gate = max(0.0, float(neuromod))
        if err_v > 0.0:
            self._potentiate(self.w_pos, active, LR * gate * err_v)
            self._decay(self.w_neg, active, LR * gate * err_v)
        elif err_v < 0.0:
            self._potentiate(self.w_neg, active, LR * gate * -err_v)
            self._decay(self.w_pos, active, LR * gate * -err_v)
        if err_a > 0.0:
            self._potentiate(self.w_aro, active, LR * gate * err_a)
        elif err_a < 0.0:
            self._decay(self.w_aro, active, LR * gate * -err_a)
        return err_v, err_a

    @staticmethod
    def _potentiate(w, active, rate):
        """Bounded Hebbian potentiation on the active features only. `rate` already folds in the
        three factors (pre-synaptic activity is the `active` gating itself; error x neuromodulator
        is `rate`)."""
        if rate <= 0.0:
            return
        for i in active:
            w[i] += rate * (1.0 - w[i] / W_MAX)

    @staticmethod
    def _decay(w, active, rate):
        """Bounded anti-Hebbian decay toward 0 on the active features only."""
        if rate <= 0.0:
            return
        for i in active:
            w[i] -= rate * (w[i] / W_MAX)
            if w[i] < 0.0:
                w[i] = 0.0

    def step(self, decay=0.01):
        """Optional slow homeostatic forgetting: every weight relaxes toward 0 by a factor `decay`
        per call, independent of any learn() signal. Use between episodes, not after every
        appraise()."""
        keep = 1.0 - decay
        for w in (self.w_pos, self.w_neg, self.w_aro):
            for i in range(self.dim):
                w[i] *= keep


# ---- feature-vector helpers for the self-check (same style as fear_conditioning._tone) ---------
def _context(seed, lo, hi, dim, p_active=0.6):
    """A sparse "context" feature vector: features in [lo, hi) are active with probability
    `p_active`, at random intensity; everything outside the band is exactly 0. Two contexts built
    from DISJOINT bands share no features, so training one cannot leak into the other."""
    r = random.Random(seed)
    v = [0.0] * dim
    for i in range(lo, hi):
        if r.random() < p_active:
            v[i] = r.random()
    return v


def _jitter(vec, rng, noise=0.05):
    """Small trial-to-trial variability, matching fear_conditioning._jitter."""
    return [min(1.0, max(0.0, x + rng.uniform(-noise, noise))) if x > 0 else 0.0 for x in vec]


def main():
    print("=" * 74)
    print("AFFECTIVE APPRAISAL -- learned multi-dimensional (valence, arousal) net")
    print("spiking three-factor readout / associative FUNCTION -- not felt affect")
    print("=" * 74)

    dim = 48
    half = dim // 2
    ctx_trained = _context(1, 0, half, dim)  # the context that gets paired with a target
    ctx_other = _context(2, half, dim, dim)  # a DIFFERENT, disjoint-band, never-trained context
    target_v, target_a = -0.70, 0.80  # a fear-like appraisal: negative valence, high arousal

    net = AffectiveAppraisal(dim=dim, seed=0)

    pre_v, pre_a = net.appraise(ctx_trained)
    pre_err = abs(target_v - pre_v) + abs(target_a - pre_a)
    other_pre_v, other_pre_a = net.appraise(ctx_other)

    trng = random.Random(7)
    EPOCHS = 25
    for _ in range(EPOCHS):
        net.learn(_jitter(ctx_trained, trng), target_v, target_a, neuromod=1.0)

    post_v, post_a = net.appraise(ctx_trained)
    post_err = abs(target_v - post_v) + abs(target_a - post_a)
    other_post_v, other_post_a = net.appraise(ctx_other)

    print(
        f"trained context : pre  (v={pre_v:+.3f}, a={pre_a:.3f})  err={pre_err:.3f}  "
        f"(naive/pre-training baseline)"
    )
    print(
        f"                  post (v={post_v:+.3f}, a={post_a:.3f})  err={post_err:.3f}  "
        f"target (v={target_v:+.2f}, a={target_a:.2f})"
    )
    print(f"other context   : pre  (v={other_pre_v:+.3f}, a={other_pre_a:.3f})")
    print(
        f"                  post (v={other_post_v:+.3f}, a={other_post_a:.3f})  "
        f"(never trained -- must stay neutral)"
    )

    # ---- determinism: same seed + same training protocol -> identical weights/output -----------
    net2 = AffectiveAppraisal(dim=dim, seed=0)
    trng2 = random.Random(7)
    for _ in range(EPOCHS):
        net2.learn(_jitter(ctx_trained, trng2), target_v, target_a, neuromod=1.0)
    det = net2.appraise(ctx_trained) == (post_v, post_a)
    print(f"determinism: two identically-seeded runs match = {det}")

    # ---- step() homeostatic decay: weights relax toward 0 with no accompanying learn() ----------
    pre_decay_err = abs(target_v - net.appraise(ctx_trained)[0])
    for _ in range(60):
        net.step(decay=0.02)
    decayed_v, decayed_a = net.appraise(ctx_trained)
    post_decay_err = abs(target_v - decayed_v)
    print(
        f"step() decay x60 -> trained context (v={decayed_v:+.3f}, a={decayed_a:.3f})  "
        f"(error to target grew {post_decay_err:.3f} vs {pre_decay_err:.3f} pre-decay)"
    )
    print("=" * 74)

    # ---- un-riggable self-check asserts ----------------------------------------------------------
    # (1) learning genuinely moves the trained context CLOSER to its target than the naive baseline.
    assert post_err < pre_err, (
        f"post-training error ({post_err:.3f}) must be smaller than the pre-training baseline ({pre_err:.3f})"
    )
    assert post_err < 0.5 * pre_err, (
        f"post-training error ({post_err:.3f}) must improve substantially over baseline ({pre_err:.3f})"
    )
    # (2) the learned appraisal actually points the right DIRECTION (not just a smaller magnitude).
    assert post_v < pre_v, "trained valence must move toward the negative target, not away from it"
    assert post_a > pre_a, "trained arousal must move toward the higher target, not away from it"
    # (3) a DIFFERENT (disjoint-feature) context is untouched by training on the first one.
    assert (other_post_v, other_post_a) == (other_pre_v, other_pre_a) == (0.0, 0.0), (
        f"an unrelated context must stay exactly neutral, got pre={(other_pre_v, other_pre_a)} "
        f"post={(other_post_v, other_post_a)}"
    )
    # (4) determinism.
    assert det, "identical seed + identical training protocol must reproduce identical output"
    # (5) step()-only decay moves the trained appraisal AWAY from the target again (pure forgetting,
    #     no relearning) -- proves step() is doing something and it is NOT the same as learn().
    assert post_decay_err > pre_decay_err, (
        "step() decay must relax weights (and the appraisal) back toward neutral"
    )

    print(
        f"self-check OK: trained-context error {pre_err:.3f} -> {post_err:.3f} vs target "
        f"(v={target_v:+.2f}, a={target_a:.2f}); a different context stayed exactly neutral "
        f"({other_pre_v:+.2f},{other_pre_a:.2f}); deterministic; step() decay relaxes learning "
        f"({pre_decay_err:.3f} -> {post_decay_err:.3f})"
    )


if __name__ == "__main__":
    main()
