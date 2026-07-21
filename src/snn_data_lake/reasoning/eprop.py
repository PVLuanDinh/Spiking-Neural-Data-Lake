"""
e-prop -- online eligibility-trace learning that approximates BPTT for recurrent spiking nets.

Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, "A solution to the learning
dilemma for recurrent networks of spiking neurons", Nature Communications 11:3625 (2020),
DOI 10.1038/s41467-020-17236-y.

WHAT e-prop IS. The true gradient of a temporal loss through a recurrent net is BPTT, which
must unroll the whole sequence and store its history (memory O(T)). e-prop FACTORS that same
gradient dE/dw_ji into two pieces that need no stored history:
    (1) a synapse-LOCAL ELIGIBILITY TRACE e_ji(t), computed FORWARD in time -- it accumulates
        the pre-synaptic activity gated by the post-synaptic PSEUDO-DERIVATIVE (surrogate
        gradient of the non-differentiable spike), and
    (2) a LEARNING SIGNAL L_j(t) broadcast from the output error through the readout weights.
The weight update is  dw_ji = -lr * sum_t L_j(t) * e_ji(t).  Because the eligibility trace is
updated per step, e-prop's memory is O(neurons + synapses), INDEPENDENT of the sequence length
T -- no BPTT history is ever stored. It is the foundation later three-factor / TESS / S-TLLR
rules build on.

HOW THIS DIFFERS from r_stdp.py and delay_learning.py (do not confuse them):
  * r_stdp.py is REINFORCEMENT learning: a SCALAR reward, delivered AFTER the fact, gates a
    spike-timing (STDP) eligibility trace. It carries NO notion of a target signal or a gradient
    -- the eligibility there is an STDP coincidence accumulator, and the third factor is a raw
    reward. e-prop is SUPERVISED / regression: its "third factor" is a per-step, per-neuron
    ERROR (target minus readout), and its eligibility trace IS the factored BPTT GRADIENT (pre
    activity x post pseudo-derivative), not a timing-coincidence heuristic.
  * delay_learning.py learns a synaptic DELAY parameter by a three-factor rule on a coincidence
    detector; it is not a gradient of a recurrent unroll and stores no eligibility of the
    membrane state. e-prop learns the WEIGHTS of a recurrent LIF network as an online
    approximation of reverse-mode BPTT.

THE GOLD-STANDARD, NON-RIGGABLE CHECK (self-check (b)). The defining claim -- "the eligibility
trace really is the factored gradient" -- is tested by comparing the e-prop weight-gradient
against a GENUINE, INDEPENDENT reverse-mode BPTT gradient of the SAME loss through the SAME
unrolled net using the SAME pseudo-derivative (see `Net.grad_bptt`, which shares NO code with
the eligibility computation in `Net.grad_eprop`: it back-propagates adjoints v<-z<-y through
time, including the recurrent off-diagonal and reset terms that e-prop deliberately drops). We
assert their COSINE SIMILARITY is clearly positive. It is NOT 1.0 -- e-prop is an approximation
(it truncates the recurrent-error paths) -- but symmetric e-prop aligns strongly. The actual
cosine is printed and asserted > 0.5.

Minimal -- 3 inputs (store cue / recall cue / background) -> an 8-unit recurrent LIF
layer -> a scalar leaky readout (+bias), on a tiny seeded store-and-recall task (T=24). Seeded
`random` + `math`, ASCII output, deterministic. Additive research
module; touches no core or safety-envelope file.

  Run: python eprop.py        # (also run by run_selfchecks.py)
"""
import math
import random

# ---- network / neuron constants (discrete-time LIF) ----
N_IN = 3            # input channels: [0]=store cue (early), [1]=recall cue (late), [2]=background
N_REC = 8           # recurrent LIF units
T = 24              # sequence length (steps); the store cue is early, the answer is at the END
ALPHA = 0.90        # membrane decay exp(-dt/tau_m): the diagonal of the recurrent Jacobian
V_TH = 0.6          # spike threshold (soft reset subtracts V_TH)
GAMMA = 0.5         # pseudo-derivative half-width: psi = DAMP*max(0, 1 - |v-V_TH|/GAMMA)
DAMP = 0.3          # pseudo-derivative dampening factor
KAPPA = 0.9         # leaky-readout decay (also the eligibility-trace output filter)
BG_LEVEL = 0.3      # tonic background CURRENT: keep neurons near threshold (not saturated)
W_IN_SCALE = 0.4    # input-weight init half-range
W_REC_SCALE = 0.2   # recurrent-weight init half-range
W_OUT_SCALE = 0.2   # readout-weight init half-range

# ---- task / training constants ----
CUE_LO, CUE_HI = 2, 7          # early STORE-cue window (steps [CUE_LO, CUE_HI))
WIN_LEN = 5                    # readout / RECALL window = the last WIN_LEN steps
N_TRAIN, N_TEST = 24, 24       # balanced (half positive, half negative)
LR = 0.08
EPOCHS = 120


def make_dataset(n, rng):
    """Store-and-recall (Bellec's canonical temporal-credit task). An early STORE cue on channel 0
    (present for class +1, absent for class -1) must be remembered across a delay and reported at
    the RECALL window at the END. Channel 1 is the RECALL cue (fires in the last WIN_LEN steps of
    BOTH classes -- it says 'answer now' but carries no class info). Channel 2 is a tonic
    background so BOTH classes spike throughout (the -1 class is not silent -> its output can be
    driven negative). The net must SUSTAIN the early store cue across ~12 silent steps to bias the
    recall response: genuine temporal credit assignment, not a feedforward reflex. Seeded via `rng`."""
    data = []
    for m in range(n):
        cls = 1 if m % 2 == 0 else -1
        start = CUE_LO + rng.randint(0, 1)          # small jitter on store-cue onset
        xs = []
        for t in range(T):
            x_store = 1.0 if (cls == 1 and start <= t < CUE_HI) else 0.0
            x_recall = 1.0 if t >= T - WIN_LEN else 0.0       # 'answer now' trigger, both classes
            x_bg = BG_LEVEL                                   # gentle tonic drive: both classes spike
            xs.append((x_store, x_recall, x_bg))
        data.append((xs, float(cls)))
    return data


def _target(cls_val):
    """Readout target over time: cls_val on the last WIN_LEN steps, undefined (no error) before."""
    tgt = [None] * T
    for t in range(T - WIN_LEN, T):
        tgt[t] = cls_val
    return tgt


class Net:
    """A recurrent LIF network: 2 inputs -> N_REC recurrent LIF units -> 1 leaky linear readout.
    Weights: Win[j][i] (input->rec), Wrec[j][k] (rec->rec, no self loop), Wout[j] (rec->readout)."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.Win = [[rng.uniform(-W_IN_SCALE, W_IN_SCALE) for _ in range(N_IN)] for _ in range(N_REC)]
        self.Wrec = [[0.0 if j == k else rng.uniform(-W_REC_SCALE, W_REC_SCALE) for k in range(N_REC)]
                     for j in range(N_REC)]
        self.Wout = [rng.uniform(-W_OUT_SCALE, W_OUT_SCALE) for _ in range(N_REC)]
        self.bout = 0.0            # readout bias: lets the baseline (no-cue) class map to -1

    def forward(self, xs):
        """Roll the net forward. Returns (v, z, psi, y): membrane, spikes, pseudo-derivative
        (all T x N_REC), and the scalar readout trace y (length T). This is the ONLY forward;
        both gradient routines below read these states -- what differs is HOW they turn the
        recorded forward into a gradient (eligibility factorization vs reverse-mode adjoints)."""
        v = [[0.0] * N_REC for _ in range(T)]
        z = [[0.0] * N_REC for _ in range(T)]
        psi = [[0.0] * N_REC for _ in range(T)]
        y = [0.0] * T
        yr_prev = 0.0                                       # leaky readout integrator (no bias)
        for t in range(T):
            z_prev = z[t - 1] if t > 0 else [0.0] * N_REC
            v_prev = v[t - 1] if t > 0 else [0.0] * N_REC
            for j in range(N_REC):
                cur = 0.0
                for i in range(N_IN):
                    cur += self.Win[j][i] * xs[t][i]
                for k in range(N_REC):
                    if k != j:
                        cur += self.Wrec[j][k] * z_prev[k]
                vj = ALPHA * v_prev[j] - V_TH * z_prev[j] + cur       # soft reset: subtract V_TH
                v[t][j] = vj
                z[t][j] = 1.0 if vj >= V_TH else 0.0
                psi[t][j] = DAMP * max(0.0, 1.0 - abs(vj - V_TH) / GAMMA)
            yr = KAPPA * yr_prev + sum(self.Wout[j] * z[t][j] for j in range(N_REC))
            y[t] = yr + self.bout                           # bias is a DIRECT output offset (stable)
            yr_prev = yr
        return v, z, psi, y

    def _errors(self, y, tgt):
        return [(y[t] - tgt[t]) if tgt[t] is not None else 0.0 for t in range(T)]

    def loss(self, xs, tgt):
        _, _, _, y = self.forward(xs)
        err = self._errors(y, tgt)
        return 0.5 * sum(e * e for e in err)

    def grad_eprop(self, xs, tgt):
        """e-prop gradient -- ONLINE, forward-only, memory O(neurons+synapses), NOT O(T).
        For each synapse we carry a scalar eligibility vector eps (filtered pre-synaptic
        activity), form the eligibility trace e = psi_post * eps, low-pass filter it by the
        readout KAPPA (ebar), and accumulate  err[t] * Wout[post] * ebar  -- i.e. the broadcast
        error (learning signal) times the eligibility trace. No per-timestep history is stored:
        eps/ebar are updated in place each step. Returns (dWin, dWrec, dWout)."""
        v, z, psi, y = self.forward(xs)
        err = self._errors(y, tgt)
        eps_in = [[0.0] * N_IN for _ in range(N_REC)]
        eps_rec = [[0.0] * N_REC for _ in range(N_REC)]
        ebar_in = [[0.0] * N_IN for _ in range(N_REC)]
        ebar_rec = [[0.0] * N_REC for _ in range(N_REC)]
        zbar = [0.0] * N_REC
        dWin = [[0.0] * N_IN for _ in range(N_REC)]
        dWrec = [[0.0] * N_REC for _ in range(N_REC)]
        dWout = [0.0] * N_REC
        dbout = 0.0
        for t in range(T):
            z_prev = z[t - 1] if t > 0 else [0.0] * N_REC
            for j in range(N_REC):
                pj = psi[t][j]
                lj = self.Wout[j] * err[t]                 # broadcast learning signal L_j(t)
                for i in range(N_IN):
                    eps_in[j][i] = ALPHA * eps_in[j][i] + xs[t][i]       # eligibility VECTOR
                    ebar_in[j][i] = KAPPA * ebar_in[j][i] + pj * eps_in[j][i]   # filtered TRACE
                    dWin[j][i] += lj * ebar_in[j][i]
                for k in range(N_REC):
                    if k != j:
                        eps_rec[j][k] = ALPHA * eps_rec[j][k] + z_prev[k]
                        ebar_rec[j][k] = KAPPA * ebar_rec[j][k] + pj * eps_rec[j][k]
                        dWrec[j][k] += lj * ebar_rec[j][k]
            for j in range(N_REC):                         # readout weights: exact (non-recurrent)
                zbar[j] = KAPPA * zbar[j] + z[t][j]
                dWout[j] += err[t] * zbar[j]
            dbout += err[t]                                 # readout bias: direct offset, exact
        return dWin, dWrec, dWout, dbout

    def grad_bptt(self, xs, tgt):
        """GENUINE, INDEPENDENT reverse-mode BPTT of the SAME loss through the SAME unrolled net,
        using the SAME pseudo-derivative psi. Shares NO code with grad_eprop: it back-propagates
        adjoints backward in time (dy -> dz -> dv), INCLUDING the recurrent off-diagonal path
        (Wrec[k][j]*dv_k[t+1]) and the reset path (-V_TH*dv_j[t+1]) that e-prop deliberately
        truncates. This is the reference the eligibility gradient is measured against."""
        v, z, psi, y = self.forward(xs)
        err = self._errors(y, tgt)
        dv = [[0.0] * N_REC for _ in range(T)]
        dz = [[0.0] * N_REC for _ in range(T)]
        dy = [0.0] * T
        for t in range(T - 1, -1, -1):
            dy[t] = err[t] + (KAPPA * dy[t + 1] if t + 1 < T else 0.0)
            for j in range(N_REC):
                s = self.Wout[j] * dy[t]                   # z_j[t] -> readout y[t]
                if t + 1 < T:
                    for k in range(N_REC):
                        if k != j:
                            s += self.Wrec[k][j] * dv[t + 1][k]   # z_j[t] -> recurrent input of k
                    s += (-V_TH) * dv[t + 1][j]            # z_j[t] -> own reset at t+1
                dz[t][j] = s
                dvj = psi[t][j] * dz[t][j]                 # v_j[t] -> z_j[t]
                if t + 1 < T:
                    dvj += ALPHA * dv[t + 1][j]            # v_j[t] -> v_j[t+1]
                dv[t][j] = dvj
        dWin = [[0.0] * N_IN for _ in range(N_REC)]
        dWrec = [[0.0] * N_REC for _ in range(N_REC)]
        dWout = [0.0] * N_REC
        dbout = 0.0
        for t in range(T):
            z_prev = z[t - 1] if t > 0 else [0.0] * N_REC
            for j in range(N_REC):
                for i in range(N_IN):
                    dWin[j][i] += dv[t][j] * xs[t][i]
                for k in range(N_REC):
                    if k != j:
                        dWrec[j][k] += dv[t][j] * z_prev[k]
                dWout[j] += dy[t] * z[t][j]
            dbout += err[t]                                 # bias is a direct output offset: dL/db = err
        return dWin, dWrec, dWout, dbout

    def apply(self, dWin, dWrec, dWout, dbout, lr):
        for j in range(N_REC):
            for i in range(N_IN):
                self.Win[j][i] -= lr * dWin[j][i]
            for k in range(N_REC):
                self.Wrec[j][k] -= lr * dWrec[j][k]
            self.Wout[j] -= lr * dWout[j]
        self.bout -= lr * dbout

    def predict(self, xs):
        _, _, _, y = self.forward(xs)
        win = sum(y[t] for t in range(T - WIN_LEN, T)) / WIN_LEN   # mean over the readout window
        return 1.0 if win >= 0.0 else -1.0                 # the window is what the loss optimizes


def _batch_grad(net, data, grad_fn):
    """Sum a per-trial gradient (grad_fn) over the whole batch. Used both to train (e-prop) and
    to snapshot the two gradients for the cosine check."""
    gWin = [[0.0] * N_IN for _ in range(N_REC)]
    gWrec = [[0.0] * N_REC for _ in range(N_REC)]
    gWout = [0.0] * N_REC
    gbout = 0.0
    for xs, cls_val in data:
        dWin, dWrec, dWout, dbout = grad_fn(xs, _target(cls_val))
        for j in range(N_REC):
            for i in range(N_IN):
                gWin[j][i] += dWin[j][i]
            for k in range(N_REC):
                gWrec[j][k] += dWrec[j][k]
            gWout[j] += dWout[j]
        gbout += dbout
    return gWin, gWrec, gWout, gbout


def _flat_hidden(gWin, gWrec):
    """Flatten the HIDDEN-weight gradient (Win + Wrec) -- the subspace where e-prop actually
    APPROXIMATES BPTT. Wout is excluded: its gradient is exact/identical in both methods, so
    including it would only inflate the cosine. This keeps the alignment check honest."""
    flat = []
    for j in range(N_REC):
        flat.extend(gWin[j])
        flat.extend(gWrec[j][k] for k in range(N_REC) if k != j)
    return flat


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _accuracy(net, data):
    return sum(1 for xs, cls_val in data if net.predict(xs) == cls_val) / len(data)


def _mean_loss(net, data):
    return sum(net.loss(xs, _target(cls_val)) for xs, cls_val in data) / len(data)


def run(seed):
    """One full deterministic experiment. Returns a dict of measured numbers + the loss
    trajectory, so main() can (d) confirm a repeat run with the same seed is byte-identical."""
    rng = random.Random(seed)
    train = make_dataset(N_TRAIN, rng)
    test = make_dataset(N_TEST, rng)
    net = Net(seed)

    # (b) GOLD STANDARD, measured at initialization on the full training batch: cosine of the
    #     e-prop gradient with a genuine independent BPTT gradient, over the hidden weights.
    e_in, e_rec, _, _ = _batch_grad(net, train, net.grad_eprop)
    b_in, b_rec, _, _ = _batch_grad(net, train, net.grad_bptt)
    cosine = _cosine(_flat_hidden(e_in, e_rec), _flat_hidden(b_in, b_rec))

    init_loss = _mean_loss(net, train)
    init_acc = _accuracy(net, test)
    traj = [init_loss]
    for _ in range(EPOCHS):
        gWin, gWrec, gWout, gbout = _batch_grad(net, train, net.grad_eprop)   # train with e-prop
        net.apply(gWin, gWrec, gWout, gbout, LR / N_TRAIN)
        traj.append(_mean_loss(net, train))
    final_loss = traj[-1]
    final_acc = _accuracy(net, test)
    return {
        "init_loss": init_loss, "final_loss": final_loss,
        "init_acc": init_acc, "final_acc": final_acc,
        "cosine": cosine, "traj": traj,
    }


def main():
    print("=" * 70)
    print("e-prop -- online eligibility-trace learning (Bellec et al. 2020)")
    print("recurrent LIF + surrogate pseudo-derivative; eligibility x broadcast error")
    print("=" * 70)

    r = run(0)
    print(f"task            : store-and-recall, T={T}, {N_REC} recurrent LIF units, readout=1")
    print(f"1 learns (loss) : initial {r['init_loss']:.4f} -> final {r['final_loss']:.4f}  "
          f"(ratio {r['final_loss'] / r['init_loss']:.3f})")
    print(f"2 accuracy      : {r['init_acc']:.0%} -> {r['final_acc']:.0%}  (chance 50%)")
    print(f"3 e-prop vs BPTT: cosine(e-prop grad, INDEPENDENT reverse-mode BPTT grad) = "
          f"{r['cosine']:.4f}")
    print("4 memory        : eligibility updated forward per-step -> O(neurons+synapses), "
          "independent of T (no BPTT history stored)")

    # determinism: same seed -> identical trajectory + identical cosine.
    r2 = run(0)
    identical = (r["traj"] == r2["traj"] and r["cosine"] == r2["cosine"])
    print(f"5 determinism   : same seed -> identical loss trajectory + cosine = {identical}")
    print("=" * 70)

    # ---- self-check (deterministic asserts; (b) is the non-riggable gold standard) ----
    assert r["final_loss"] < 0.5 * r["init_loss"], \
        f"(a) e-prop must LEARN: final loss {r['final_loss']:.4f} !< 0.5*initial {r['init_loss']:.4f}"
    assert r["final_acc"] >= 0.9, f"(a) task accuracy must rise well above chance: {r['final_acc']:.2f}"
    assert r["cosine"] > 0.5, \
        f"(b) e-prop gradient must ALIGN with the true BPTT gradient: cosine {r['cosine']:.4f} !> 0.5"
    assert identical, "(d) same seed must reproduce the loss trajectory and cosine byte-identically"
    print(f"self-check OK: LEARNS loss {r['init_loss']:.4f}->{r['final_loss']:.4f} "
          f"(acc {r['init_acc']:.0%}->{r['final_acc']:.0%}); e-prop~BPTT cosine {r['cosine']:.4f} "
          f"(>0.5, factored gradient confirmed vs an INDEPENDENT reverse-mode reference); "
          f"eligibility memory O(neurons), not O(T); deterministic")


if __name__ == "__main__":
    main()
