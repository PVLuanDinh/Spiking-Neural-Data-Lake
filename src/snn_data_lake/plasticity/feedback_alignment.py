"""
Random FEEDBACK ALIGNMENT + local plasticity -- global credit assignment without weight transport.

A new member of the plasticity family (additive, opt-in; changes no existing behaviour). This is
BrainCog's BackEISNN idea in miniature and Lillicrap et al.'s "random feedback alignment" (Nat.
Commun. 2016): a two-layer spiking net (input -> hidden LIF -> output) is trained WITHOUT the
biologically implausible weight-transport of backprop. The hidden layer's error signal is delivered
by a FIXED RANDOM feedback matrix B (not the transpose of the forward output weights W2), and that
per-neuron error is the third factor that GATES a LOCAL synaptic update. This closes the repo's
global-credit-assignment gap: a purely local rule cannot tell a hidden neuron whether it helped or
hurt the output; a feedback CONNECTION can, and -- surprisingly -- a random one aligns well enough.

MECHANISM (feedback alignment fused with a local three-factor rule):
    forward   hidden j fires as a POPULATION of LIF neurons; rate r_j = spikes / (K*T)   [spiking]
              output   o_k = W2[k] . r + b2[k];   p = softmax(o)
    output    delta_out = p - target                    (softmax + cross-entropy gradient)
    W2 update (local delta rule)   W2[k][i] -= lr * delta_out[k] * r[i]
    hidden error (NO weight transport)   e_hidden[j] = sum_k B[j][k] * delta_out[k]    (B fixed random)
    W1 update (local, gated by e_hidden)   dw_ji = -lr * e_hidden[j] * phi_j * x_i
              where phi_j is a self-limiting surrogate of the LIF rate's slope.
The W1 update is delivered through RSTDPNeuron.reward() (basal_ganglia.r_stdp): its reward-modulated
form  w += lr * modulator * eligibility  IS the three-factor local rule. Feedback alignment supplies
the PER-NEURON modulator (the credit signal) that R-STDP normally gets as a single global dopamine
scalar; the eligibility here is a rate-coded Hebbian coincidence (presynaptic x_i times a
postsynaptic surrogate) rather than a spike-timing trace, because the hidden units are read as
population firing rates. It reuses cortex.lif_neuron.LIFNeuron for the spiking forward pass and
substrate._mathutils for softmax/dot; the local-rule family is BCM / R-STDP.

HONESTY -- this is a MECHANISM PoC, not a trained deep-net benchmark. Random feedback alignment
APPROACHES exact backprop; it does NOT reproduce it -- the fixed random feedback direction only
partially aligns with the true gradient (cosine < 1, reported below). At this toy scale (3-bit
parity, 4 hidden units) the wins that ARE real and load-bearing:
  (1) global feedback + a local rule LEARNS the nonlinear task; a LOCAL-ONLY net (no error to the
      hidden layer) stays near chance -- the feedback credit assignment is doing the work;
  (2) a RANDOM, weight-transport-free feedback still assigns useful credit, but only when it is a
      STATIONARY matrix the forward weights can align to: a per-step reshuffled feedback degrades
      back toward local-only.
No claim of matching backprop at scale, no deep multi-layer credit assignment, no cognition.
Refs: Lillicrap, Cownden, Tweed & Akerman, "Random synaptic feedback weights support error
backpropagation for deep learning", Nat. Commun. 7 (2016); Zeng et al., BrainCog, Patterns 2023
(BackEISNN, feedback-connection learning); Bienenstock, Cooper & Munro, J. Neurosci. 2 (1982).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.feedback_alignment
"""

import math
import random

from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron
from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate._mathutils import dot, softmax

N_IN = 3  # 3-bit parity: the smallest task random hidden features reliably FAIL on
N_HIDDEN = 4  # small enough that a frozen random hidden layer cannot solve it
N_OUT = 2  # even / odd parity
K_SUB = 6  # LIF sub-neurons per hidden unit -> a smooth POPULATION rate code
T_STEPS = 20  # timesteps the hidden LIF population is run per forward pass
TAU_M = 8.0
SUB_OFFSETS = [0.15 * (k / (K_SUB - 1)) for k in range(K_SUB)]  # heterogeneous fixed bias offsets
BIAS0 = 0.30  # baseline drive so the hidden population sits in its graded firing band
GAIN = 0.35  # maps hidden pre-activation to input current
W_SCALE = 1.0  # init range for all weights and the fixed feedback matrix
B1_INIT = 0.10  # init range for the learnable per-unit current bias
CUR_LO, CUR_HI = 0.03, 1.3  # clamp the drive out of the permanent-dead / hard-saturated zones
B1_LO, B1_HI = -0.25, 0.45  # keep the learnable bias in a live regime (no dead-neuron runaway)
RATE_MAX = 0.7  # population rate ceiling (used only to shape the surrogate)
SURR_LEAK = 0.06  # surrogate floor so a silent unit can still recover
OUT_LR = 0.25
HID_LR = 0.45
B1_LR = 0.13
EPOCHS = 1200
N_SEEDS = 8  # average over a fixed seed range -> stable, un-cherry-picked means

MODES = ("fa", "local", "shuffled", "backprop")


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def parity_dataset():
    """The 2**N_IN input patterns of N-bit parity (label = XOR of the bits). Not linearly
    separable: it genuinely needs a learned hidden layer, so a frozen random one fails."""
    data = []
    for n in range(2**N_IN):
        x = [float((n >> i) & 1) for i in range(N_IN)]
        target = [0.0] * N_OUT
        target[int(sum(x)) % 2] = 1.0
        data.append((x, target))
    return data


def _pop_lif_rate(base):
    """Genuinely spiking forward: a population of K_SUB LIF neurons with heterogeneous fixed bias
    offsets is driven at current `base` for T_STEPS; the averaged spike count is a SMOOTH graded
    rate (a single LIF has a coarse staircase F-I curve; the population smooths it)."""
    total = 0
    for off in SUB_OFFSETS:
        total += sum(LIFNeuron(tau_m=TAU_M).run([base + off] * T_STEPS))
    return total / (K_SUB * T_STEPS)


class FeedbackAlignmentSNN:
    """input -> hidden LIF population -> softmax output, trained by random feedback alignment.

    `mode` selects the hidden-layer error signal (everything else is identical -- a clean control):
      "fa"       : e_hidden = B . delta_out          (FIXED random feedback -- weight-transport-free)
      "local"    : e_hidden = 0                       (no feedback -- the local-only control)
      "shuffled" : e_hidden = B_reshuffled . delta    (feedback re-randomized every step -- no
                                                       stationary target to align to)
      "backprop" : e_hidden = W2^T . delta_out        (exact weight transport -- the reference)
    """

    def __init__(self, seed=0, mode="fa"):
        self.mode = mode
        rng = random.Random(seed)
        # hidden afferent weights live inside RSTDPNeuron cells (reused as the local plastic synapse);
        # overwrite the default init so every mode shares an identical start for a given seed.
        self.hidden = [RSTDPNeuron(N_IN, lr=HID_LR, seed=seed * 100 + j) for j in range(N_HIDDEN)]
        for cell in self.hidden:
            cell.w = [rng.uniform(-W_SCALE, W_SCALE) for _ in range(N_IN)]
        self.b1 = [rng.uniform(-B1_INIT, B1_INIT) for _ in range(N_HIDDEN)]
        self.W2 = [[rng.uniform(-W_SCALE, W_SCALE) for _ in range(N_HIDDEN)] for _ in range(N_OUT)]
        self.b2 = [0.0] * N_OUT
        self.B = [[rng.uniform(-W_SCALE, W_SCALE) for _ in range(N_OUT)] for _ in range(N_HIDDEN)]
        self._shuf_rng = random.Random(seed * 7 + 3)

    def _hidden_rates(self, x):
        rates = []
        for j, cell in enumerate(self.hidden):
            base = _clamp(BIAS0 + GAIN * dot(cell.w, x) + self.b1[j], CUR_LO, CUR_HI)
            rates.append(_pop_lif_rate(base))
        return rates

    def forward(self, x):
        r = self._hidden_rates(x)
        o = [dot(self.W2[k], r) + self.b2[k] for k in range(N_OUT)]
        return softmax(o), r

    def _hidden_error(self, delta_out):
        """The per-hidden-neuron error signal -- the ONLY thing that differs across modes."""
        if self.mode == "local":
            return [0.0] * N_HIDDEN
        if self.mode == "backprop":
            return [
                sum(self.W2[k][j] * delta_out[k] for k in range(N_OUT)) for j in range(N_HIDDEN)
            ]
        if self.mode == "shuffled":
            flat = [self.B[j][k] for j in range(N_HIDDEN) for k in range(N_OUT)]
            self._shuf_rng.shuffle(flat)
            fb = [[flat[j * N_OUT + k] for k in range(N_OUT)] for j in range(N_HIDDEN)]
        else:  # "fa": the fixed random feedback matrix
            fb = self.B
        return [sum(fb[j][k] * delta_out[k] for k in range(N_OUT)) for j in range(N_HIDDEN)]

    def train_step(self, x, target):
        p, r = self.forward(x)
        delta_out = [p[k] - target[k] for k in range(N_OUT)]
        e_hidden = self._hidden_error(delta_out)  # uses current W2 (must precede the W2 update)
        # output layer: local delta rule
        for k in range(N_OUT):
            for i in range(N_HIDDEN):
                self.W2[k][i] -= OUT_LR * delta_out[k] * r[i]
            self.b2[k] -= OUT_LR * delta_out[k]
        # hidden layer: local three-factor rule via RSTDPNeuron.reward (w += lr * modulator * elig).
        # elig = surrogate-Hebbian coincidence (post slope phi * pre x); modulator = -e_hidden (descent).
        for j, cell in enumerate(self.hidden):
            phi = 4.0 * r[j] * (RATE_MAX - r[j]) / (RATE_MAX * RATE_MAX) + SURR_LEAK
            cell.e = [phi * GAIN * x[i] for i in range(N_IN)]
            cell.reward(-e_hidden[j])
            self.b1[j] = _clamp(self.b1[j] - B1_LR * e_hidden[j] * phi, B1_LO, B1_HI)
        return -sum(target[k] * math.log(p[k] + 1e-12) for k in range(N_OUT))


def _accuracy(net, data):
    ok = 0
    for x, target in data:
        p, _ = net.forward(x)
        if max(range(N_OUT), key=p.__getitem__) == max(range(N_OUT), key=target.__getitem__):
            ok += 1
    return ok / len(data)


def feedback_alignment_cosine(net, data):
    """cosine( B . delta , W2^T . delta ) averaged over the data -- how close the random-feedback
    credit direction is to the exact-backprop gradient. In (0, 1): > 0 aligns (approaches),
    < 1 never matches. This IS the defining approximation of feedback alignment."""
    coss = []
    for x, target in data:
        p, _ = net.forward(x)
        d = [p[k] - target[k] for k in range(N_OUT)]
        g_fa = [sum(net.B[j][k] * d[k] for k in range(N_OUT)) for j in range(N_HIDDEN)]
        g_bp = [sum(net.W2[k][j] * d[k] for k in range(N_OUT)) for j in range(N_HIDDEN)]
        den = math.sqrt(dot(g_fa, g_fa) * dot(g_bp, g_bp))
        if den > 1e-12:
            coss.append(dot(g_fa, g_bp) / den)
    return sum(coss) / len(coss) if coss else 0.0


def train_condition(mode, seed, epochs=EPOCHS):
    """Train one net; return (net, accuracy, final_epoch_loss). Deterministic in (mode, seed)."""
    net = FeedbackAlignmentSNN(seed=seed, mode=mode)
    data = parity_dataset()
    order = random.Random(seed * 13 + 5)
    loss = 0.0
    for _ in range(epochs):
        idx = list(range(len(data)))
        order.shuffle(idx)
        loss = sum(net.train_step(*data[k]) for k in idx) / len(data)
    return net, _accuracy(net, data), loss


def _run(mode, keep_align=False):
    data = parity_dataset()
    accs, losses, aligns = [], [], []
    for seed in range(N_SEEDS):
        net, acc, loss = train_condition(mode, seed)
        accs.append(acc)
        losses.append(loss)
        if keep_align:
            aligns.append(feedback_alignment_cosine(net, data))
    mean = sum(accs) / len(accs)
    return mean, sum(losses) / len(losses), accs, (sum(aligns) / len(aligns) if aligns else 0.0)


def main():
    chance = 1.0 / N_OUT
    fa_acc, fa_loss, fa_seeds, align = _run("fa", keep_align=True)
    lo_acc, lo_loss, lo_seeds, _ = _run("local")
    sh_acc, sh_loss, sh_seeds, _ = _run("shuffled")
    bp_acc, bp_loss, bp_seeds, _ = _run("backprop")

    print("=" * 78)
    print("FEEDBACK ALIGNMENT -- random weight-transport-free credit assignment + local plasticity")
    print(
        f"(input -> hidden LIF population -> output; {N_IN}-bit parity, {N_HIDDEN} hidden, chance {chance:.2f})"
    )
    print("=" * 78)
    print(f"{'condition':<26}{'mean acc':>10}{'mean loss':>11}   per-seed")
    print(
        f"{'feedback + local (FA)':<26}{fa_acc:>10.3f}{fa_loss:>11.3f}   {[round(a, 2) for a in fa_seeds]}"
    )
    print(
        f"{'local-only (no feedback)':<26}{lo_acc:>10.3f}{lo_loss:>11.3f}   {[round(a, 2) for a in lo_seeds]}"
    )
    print(
        f"{'shuffled feedback':<26}{sh_acc:>10.3f}{sh_loss:>11.3f}   {[round(a, 2) for a in sh_seeds]}"
    )
    print(
        f"{'exact backprop (W2^T ref)':<26}{bp_acc:>10.3f}{bp_loss:>11.3f}   {[round(a, 2) for a in bp_seeds]}"
    )
    print("-" * 78)
    print(f"FA-vs-backprop feedback alignment  cos(B.delta, W2^T.delta) = {align:.3f}  (in (0,1))")
    print(
        f"gap to exact backprop: FA reaches {fa_acc:.3f} vs the W2^T reference {bp_acc:.3f}; the random"
    )
    print(
        "feedback only PARTIALLY aligns with the true gradient (cos < 1) -- it approaches, does not match."
    )
    print("=" * 78)

    # ---- self-check: deterministic asserts, ASCII. Claim + CONTROLS that FAIL. ----
    # (1) feedback + local LEARNS the nonlinear task, well above chance.
    assert fa_acc > 0.85, f"feedback+local must learn parity well above chance ({fa_acc:.3f})"
    assert fa_acc - chance > 0.30, (
        f"feedback+local must beat chance by a wide margin ({fa_acc:.3f})"
    )
    # (2) LOCAL-ONLY control fails -- stuck near chance -> the feedback credit assignment is load-bearing.
    assert lo_acc < 0.72, f"local-only must stay near chance ({lo_acc:.3f})"
    assert fa_acc - lo_acc > 0.25, (
        f"feedback must be load-bearing: FA >> local-only ({fa_acc:.3f} vs {lo_acc:.3f})"
    )
    # (3) SHUFFLED feedback degrades toward local-only -> it is the ALIGNED STATIONARY random feedback,
    #     not per-step noise, that assigns credit.
    assert sh_acc < 0.72, f"per-step-reshuffled feedback must stay near chance ({sh_acc:.3f})"
    assert fa_acc - sh_acc > 0.25, (
        f"stationary random feedback must beat reshuffled ({fa_acc:.3f} vs {sh_acc:.3f})"
    )
    assert sh_acc < lo_acc + 0.12, (
        f"shuffled must degrade toward local-only ({sh_acc:.3f} vs {lo_acc:.3f})"
    )
    # (4) gap to exact backprop -- FA approaches (aligns with) but does NOT match the true gradient.
    assert 0.15 < align < 0.97, (
        f"FA feedback must align with but not equal exact backprop ({align:.3f})"
    )
    # (5) determinism: same (mode, seed) -> identical weights.
    a = train_condition("fa", 0)[0]
    b = train_condition("fa", 0)[0]
    same = (
        all(a.hidden[j].w == b.hidden[j].w for j in range(N_HIDDEN))
        and a.W2 == b.W2
        and a.b1 == b.b1
    )
    assert same, "same (mode, seed) must give identical weights (determinism)"

    print(
        f"self-check OK: feedback+local {fa_acc:.3f} vs local-only {lo_acc:.3f} vs shuffled {sh_acc:.3f} "
        f"(chance {chance:.2f}); backprop ref {bp_acc:.3f}; alignment {align:.2f} (approaches, not matches); "
        f"random weight-transport-free credit assignment; additive; deterministic"
    )


if __name__ == "__main__":
    main()
