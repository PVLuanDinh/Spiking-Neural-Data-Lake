"""
SORN -- a self-organizing recurrent network that BEATS a static random reservoir on sequence
prediction (Lazar, Pipa & Triesch, Front. Comput. Neurosci. 3:23, 2009).

The repo already has a fixed random spiking reservoir -- research/spiking_ssm.py, a Liquid State
Machine whose recurrent weights and thresholds are frozen at random init and only a linear readout
is trained. It also has the individual plasticity primitives (intrinsic_plasticity, homeostatic_scaling,
valence_stdp / bcm_plasticity) as standalone single-cell/single-rule demos. What it did NOT have is the
INTEGRATED self-organizing network: a reservoir that reshapes its OWN recurrent weights and thresholds
online via three interacting plasticity rules and thereby represents an input sequence's temporal
statistics far better than a frozen random reservoir of the same size.

SORN is a recurrent net of N_E excitatory + N_I inhibitory BINARY threshold units. Discrete update
(Theta = Heaviside step):
    x_E(t+1) = Theta( W_EE x_E(t) - W_EI x_I(t) - T_E + u(t) )
    x_I(t+1) = Theta( W_IE x_E(t) - T_I )
Three plasticity rules run together during a self-organization phase and drive the random net toward a
structured, homeostatic regime:
  (1) STDP on the sparse excitatory W_EE:  dW_ij = eta_stdp * ( x_i(t)*x_j(t-1) - x_i(t-1)*x_j(t) );
  (2) synaptic normalization: each unit's incoming excitatory weights re-scale to sum to 1 (bounded,
      competitive);
  (3) intrinsic plasticity (IP): T_E_i += eta_ip * ( x_i(t) - H_IP ) -- pushes every unit toward a target
      mean firing rate H_IP (homeostasis).

DISTINCT from spiking_ssm.py: that is a STATIC LSM (the control here). SORN's win comes only from letting
those three rules run. Faithful to Hartmann, Lazar, Nessler & Triesch: the SORN update itself carries NO
noise -- its richness comes from LEARNING in a deterministic net -- so this runs fully deterministically
(one per-instance random.Random(seed) threads wiring / init / the input stream; no module-level seeding).

Task: a structured symbol stream built from two 4-symbol "words" that SHARE their middle --
  word1 = A B C D   (0 1 2 3),   word2 = E B C F   (4 1 2 5).
Both go '... B C ...', so the current symbol C is ambiguous: the next symbol is D or F depending on whether
the word started with A or E, three steps earlier. Predicting it REQUIRES recurrent memory of the word
identity; a memoryless code caps at 3/4 = 75% (the word boundary D/F -> next start is a genuine 50/50).
An identical-architecture STATIC reservoir (same size, same random wiring/init, plasticity OFF) with the
SAME ridge readout trained on the SAME data is the fair control -- the ONLY difference is whether SORN's
three rules ran.

  python research/sorn.py       # (also run by run_selfchecks.py, root on PYTHONPATH)
"""

import math
import random

# ---- network / task sizes (small: whole experiment runs twice in a few seconds) ----
N_E = 100  # excitatory binary threshold units
N_I = 20  # inhibitory units (~0.2 * N_E)
P_EE = 0.10  # E->E connection probability (sparse recurrent excitation, plastic)
P_IE = 0.20  # E->I connection probability (fixed)
P_EI = 0.30  # I->E connection probability (fixed)
T_E_MAX = 0.5  # initial excitatory thresholds ~ U(0, T_E_MAX) -- heterogeneous on purpose
T_I_MAX = 0.8  # inhibitory thresholds ~ U(0, T_I_MAX), fixed
I_IN = 1.0  # input drive to a symbol's E units (> T_E_MAX -> forces them to spike)
IN_PER_SYM = 8  # E units driven by each symbol (disjoint blocks)

ETA_STDP = 0.004  # STDP learning rate on W_EE
ETA_IP = 0.01  # intrinsic-plasticity rate on T_E
H_IP = 0.10  # IP target mean firing rate (homeostatic set-point)

SELFORG_WORDS = 800  # words streamed through the plasticity (self-organization) phase
TRAIN_WORDS = 220  # words for readout training (features + next-symbol targets)
TEST_WORDS = 220  # held-out words for evaluation
WASHOUT = 20  # leading steps dropped so the readout sees settled state
RIDGE = 3.0  # ridge (L2) regularization for the least-squares readout

# Two words with a DISTINCT shared middle: word1 = A [g0..g(L-1)] D, word2 = E [same middle] F. The middle
# transitions are all deterministic (no counting); only the last-middle-symbol -> end branch is ambiguous
# and needs memory of the START, SHARED_LEN+1 steps back. The stream also carries seeded INPUT NOISE
# (NOISE_UNITS spurious spikes per step, identical for both networks). That is the regime where SORN's
# self-organization matters: a static random reservoir sits in a high-activity CHAOTIC regime that
# AMPLIFIES the noise and scrambles the stored word identity, whereas SORN's three rules pull it to a
# sparse, homeostatic, NEAR-CRITICAL regime (branching ~1) that carries the signal through the noise.
SHARED_LEN = 4
NOISE_UNITS = 4  # spurious input spikes injected per step (seeded; shared by both nets)


def _build_task(shared_len):
    start1, start2 = 0, 1
    shared = tuple(range(2, 2 + shared_len))
    end1, end2 = 2 + shared_len, 3 + shared_len
    w1 = (start1,) + shared + (end1,)
    w2 = (start2,) + shared + (end2,)
    return w1, w2, 4 + shared_len


WORD1, WORD2, N_SYMBOLS = _build_task(SHARED_LEN)


def set_task(shared_len):
    """(Re)build the task words for a given shared-middle length (used to tune memory depth)."""
    global WORD1, WORD2, N_SYMBOLS, SHARED_LEN
    SHARED_LEN = shared_len
    WORD1, WORD2, N_SYMBOLS = _build_task(shared_len)


def make_stream(n_words, rng):
    """Random concatenation of WORD1 / WORD2 with per-step seeded input noise.

    Returns a list of (symbol, noise_units): noise_units is a tuple of E indices given a spurious input
    spike this step. Baked into the stream so BOTH networks see the identical noise realization (fair)."""
    out = []
    for _ in range(n_words):
        word = WORD1 if rng.random() < 0.5 else WORD2
        for sym in word:
            noise = tuple(rng.sample(range(N_E), NOISE_UNITS)) if NOISE_UNITS else ()
            out.append((sym, noise))
    return out


def build_net(rng):
    """Random SORN: sparse E->E adjacency (incoming lists), fixed I<->E, heterogeneous thresholds.

    Returns a dict of mutable state. W_EE is stored as, per post-unit i, parallel lists (pre[i], w[i]);
    rows are normalized to sum 1 at init (synaptic-normalization invariant)."""
    # E->E incoming synapses per post-unit i (no self-connections)
    ee_pre, ee_w = [], []
    for i in range(N_E):
        pre, w = [], []
        for j in range(N_E):
            if j != i and rng.random() < P_EE:
                pre.append(j)
                w.append(rng.random())
        if not pre:  # guarantee >=1 incoming so normalization is defined
            j = rng.randrange(N_E - 1)
            j = j + 1 if j >= i else j
            pre.append(j)
            w.append(rng.random())
        s = sum(w)
        w = [x / s for x in w]  # normalize incoming to sum 1
        ee_pre.append(pre)
        ee_w.append(w)
    # I->E incoming (fixed), per E post-unit i: list of (k, weight); rows normalized to sum 1
    ei = []
    for i in range(N_E):
        pre = [k for k in range(N_I) if rng.random() < P_EI]
        wk = [rng.random() for _ in pre]
        s = sum(wk) or 1.0
        ei.append(list(zip(pre, [x / s for x in wk])))
    # E->I incoming (fixed), per I post-unit k: list of (j, weight); rows normalized to sum 1
    ie = []
    for k in range(N_I):
        pre = [j for j in range(N_E) if rng.random() < P_IE]
        if not pre:
            pre = [rng.randrange(N_E)]
        wj = [rng.random() for _ in pre]
        s = sum(wj) or 1.0
        ie.append(list(zip(pre, [x / s for x in wj])))
    t_e = [rng.uniform(0, T_E_MAX) for _ in range(N_E)]
    t_i = [rng.uniform(0, T_I_MAX) for _ in range(N_I)]
    # disjoint input block per symbol (per-symbol size shrinks if the alphabet is large)
    per = min(IN_PER_SYM, max(2, N_E // N_SYMBOLS))
    units = list(range(N_E))
    rng.shuffle(units)
    in_units = [units[s * per : (s + 1) * per] for s in range(N_SYMBOLS)]
    return {
        "ee_pre": ee_pre,
        "ee_w": ee_w,
        "ei": ei,
        "ie": ie,
        "t_e": t_e,
        "t_i": t_i,
        "in_units": in_units,
    }


def _step(net, x_e, x_i, sym, noise=()):
    """One SORN update using the OLD (x_e, x_i), input symbol `sym`, and noise units. Returns new states."""
    ee_pre, ee_w, ei = net["ee_pre"], net["ee_w"], net["ei"]
    t_e, in_units = net["t_e"], net["in_units"]
    drive_in = [0.0] * N_E
    for u in in_units[sym]:
        drive_in[u] = I_IN
    for u in noise:
        drive_in[u] = I_IN
    new_e = [0] * N_E
    for i in range(N_E):
        r = drive_in[i] - t_e[i]
        pre, w = ee_pre[i], ee_w[i]
        for idx in range(len(pre)):
            if x_e[pre[idx]]:
                r += w[idx]
        for k, wk in ei[i]:
            if x_i[k]:
                r -= wk
        new_e[i] = 1 if r >= 0.0 else 0
    ie, t_i = net["ie"], net["t_i"]
    new_i = [0] * N_I
    for k in range(N_I):
        r = -t_i[k]
        for j, wj in ie[k]:
            if x_e[j]:
                r += wj
        new_i[k] = 1 if r >= 0.0 else 0
    return new_e, new_i


def self_organize(net, stream):
    """Run STDP + synaptic normalization + intrinsic plasticity over the stream (mutates net)."""
    ee_pre, ee_w, t_e = net["ee_pre"], net["ee_w"], net["t_e"]
    x_e, x_i = [0] * N_E, [0] * N_I
    for sym, noise in stream:
        new_e, new_i = _step(net, x_e, x_i, sym, noise)
        # (1) STDP on existing E->E synapses: potentiate i<-j if i fired after j; depress if j after i
        for i in range(N_E):
            pre, w = ee_pre[i], ee_w[i]
            changed = False
            for idx in range(len(pre)):
                j = pre[idx]
                d = ETA_STDP * (new_e[i] * x_e[j] - x_e[i] * new_e[j])
                if d:
                    nv = w[idx] + d
                    w[idx] = nv if nv > 0.0 else 0.0  # excitatory: clip at 0
                    changed = True
            # (2) synaptic normalization: re-scale incoming weights to sum 1
            if changed:
                s = sum(w)
                if s > 0.0:
                    ee_w[i] = [x / s for x in w]
        # (3) intrinsic plasticity: push each unit's rate toward H_IP by moving its threshold
        for i in range(N_E):
            t_e[i] += ETA_IP * (new_e[i] - H_IP)
        x_e, x_i = new_e, new_i


def collect(net, stream):
    """Run the net (plasticity OFF) over the stream; return (features, labels) for next-symbol readout.

    Feature = the E-state after seeing symbol t (binary) + bias; label = symbol t+1. Drops WASHOUT."""
    x_e, x_i = [0] * N_E, [0] * N_I
    states = []
    for sym, noise in stream:
        x_e, x_i = _step(net, x_e, x_i, sym, noise)
        states.append(x_e)
    feats, labels = [], []
    for t in range(WASHOUT, len(stream) - 1):
        act = [i for i in range(N_E) if states[t][i]]
        act.append(N_E)  # bias feature index
        feats.append(act)
        labels.append(stream[t + 1][0])  # next symbol (drop its noise)
    return feats, labels


def probe_rates(net, stream):
    """E-unit firing rates over the stream (plasticity OFF) + mean spikes/step (branching numerator)."""
    x_e, x_i = [0] * N_E, [0] * N_I
    counts = [0] * N_E
    prev_tot, ratios = None, []
    for sym, noise in stream:
        x_e, x_i = _step(net, x_e, x_i, sym, noise)
        tot = sum(x_e)
        for i in range(N_E):
            counts[i] += x_e[i]
        if prev_tot:
            ratios.append(tot / prev_tot)  # spikes(t)/spikes(t-1): branching-ratio estimate
        prev_tot = tot
    n = len(stream)
    rates = [c / n for c in counts]
    branching = sum(ratios) / len(ratios) if ratios else 0.0
    return rates, branching


def _solve(a, b):
    """Solve A X = B (A n x n, B n x m) by Gaussian elimination with partial pivoting."""
    n, m = len(a), len(b[0])
    aug = [a[r][:] + b[r][:] for r in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(aug[r][c]))
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c] or 1e-12
        inv = 1.0 / piv
        row = aug[c]
        for k in range(c, n + m):
            row[k] *= inv
        for r in range(n):
            if r != c and aug[r][c]:
                f = aug[r][c]
                ar = aug[r]
                for k in range(c, n + m):
                    ar[k] -= f * row[k]
    return [[aug[r][n + col] for col in range(m)] for r in range(n)]


def train_readout(feats, labels):
    """Ridge least-squares readout over the (sparse binary) E-state features -> one-hot next symbol.

    Optimal linear readout (no epochs / learning-rate to tune) -> the control cannot be under-trained.
    Builds A = X^T X + RIDGE*I and B = X^T Y exploiting binary sparsity, then solves A W = B."""
    nf = N_E + 1
    a = [[0.0] * nf for _ in range(nf)]
    b = [[0.0] * N_SYMBOLS for _ in range(nf)]
    for act, lab in zip(feats, labels):
        for p in act:
            row = a[p]
            for q in act:
                row[q] += 1.0
            b[p][lab] += 1.0
    for p in range(nf):
        a[p][p] += RIDGE
    return _solve(a, b)


def accuracy(w, feats, labels):
    ok = 0
    for act, lab in zip(feats, labels):
        best, bs = 0, None
        for c in range(N_SYMBOLS):
            sc = 0.0
            for p in act:
                sc += w[p][c]
            if bs is None or sc > bs:
                bs, best = sc, c
        ok += best == lab
    return ok / len(labels)


def _mean_var(v):
    m = sum(v) / len(v)
    return m, sum((x - m) ** 2 for x in v) / len(v)


def _weight_std(net):
    ws = [w for row in net["ee_w"] for w in row]
    m = sum(ws) / len(ws)
    return math.sqrt(sum((x - m) ** 2 for x in ws) / len(ws))


def run_experiment(seed):
    """Build one random net; the STATIC control uses it as-is, SORN self-organizes a COPY. Same wiring,
    same init, same readout, same data -- the only difference is whether the three rules ran."""
    rng = random.Random(seed)
    net = build_net(rng)
    selforg = make_stream(SELFORG_WORDS, rng)
    train = make_stream(TRAIN_WORDS, rng)
    test = make_stream(TEST_WORDS, rng)
    probe = make_stream(60, rng)

    # STATIC control: frozen random reservoir (plasticity never runs)
    static_std = _weight_std(net)
    r0, br0 = probe_rates(net, probe)
    fx, ly = collect(net, train)
    ftx, lty = collect(net, test)
    w0 = train_readout(fx, ly)
    static_acc = accuracy(w0, ftx, lty)

    # SORN: same net, but run the three plasticity rules over the self-organization stream first
    sorn = build_net(random.Random(seed))  # identical net (same seed -> same wiring/init)
    self_organize(sorn, selforg)
    sorn_std = _weight_std(sorn)
    r1, br1 = probe_rates(sorn, probe)
    fx2, ly2 = collect(sorn, train)
    ftx2, lty2 = collect(sorn, test)
    w1 = train_readout(fx2, ly2)
    sorn_acc = accuracy(w1, ftx2, lty2)

    m0, v0 = _mean_var(r0)
    m1, v1 = _mean_var(r1)
    return {
        "static_acc": static_acc,
        "sorn_acc": sorn_acc,
        "static_rate_mean": m0,
        "static_rate_var": v0,
        "sorn_rate_mean": m1,
        "sorn_rate_var": v1,
        "static_std": static_std,
        "sorn_std": sorn_std,
        "static_branch": br0,
        "sorn_branch": br1,
        "static_silent": sum(1 for r in r0 if r < 0.005),
        "sorn_silent": sum(1 for r in r1 if r < 0.005),
    }


def main():
    seed = 7
    res = run_experiment(seed)
    chance = 1.0 / N_SYMBOLS
    # a code with no memory of the word identity must guess the run's end -> best it can do on this stream
    memoryless = (SHARED_LEN + 0.5) / (SHARED_LEN + 2)
    ceiling = (SHARED_LEN + 1.5) / (SHARED_LEN + 2)

    print("=" * 66)
    print("SORN -- self-organizing recurrent net vs a static reservoir (Lazar 2009)")
    print("=" * 66)
    print(f"net: N_E={N_E} E + N_I={N_I} I binary threshold units; W_EE sparse p={P_EE} (plastic)")
    print(
        f"task: word1 = A [{SHARED_LEN} shared syms] D, word2 = E [same {SHARED_LEN}] F -> next-after-shared"
    )
    print(
        f"      needs {SHARED_LEN + 1}-step memory of the start; {NOISE_UNITS} noise spikes/step; H_IP={H_IP}"
    )
    print(
        f"self-org {SELFORG_WORDS}w  train {TRAIN_WORDS}w  test {TEST_WORDS}w (held out)  seed={seed}"
    )
    print("-" * 66)
    print(f"{'net':<10}{'next-symbol acc':>18}{'mean E-rate (target ' + f'{H_IP:.2f})':>26}")
    print(f"{'static':<10}{res['static_acc']:>17.1%}{res['static_rate_mean']:>25.3f}")
    print(f"{'SORN':<10}{res['sorn_acc']:>17.1%}{res['sorn_rate_mean']:>25.3f}")
    print("-" * 66)
    margin = res["sorn_acc"] - res["static_acc"]
    print(
        f"chance = {chance:.1%}   memoryless = {memoryless:.1%}   full ceiling = {ceiling:.1%}   "
        f"SORN - static = {margin:+.1%}"
    )
    print(
        f"rate variance   static {res['static_rate_var']:.4f} -> SORN {res['sorn_rate_var']:.4f} "
        "(IP homeostasis tightens the rate distribution)"
    )
    print(
        f"silent E units  static {res['static_silent']:>3d} -> SORN {res['sorn_silent']:>3d}  |  "
        f"W_EE weight std {res['static_std']:.3f} -> {res['sorn_std']:.3f} (STDP broadens)"
    )
    print(
        f"branching ratio static {res['static_branch']:.2f} -> SORN {res['sorn_branch']:.2f} "
        "(~1 == near-critical)"
    )
    print("=" * 66)

    # ---- self-checks (deterministic; the paper's comparison, nothing rigged) ----
    # (a) self-organization helps: SORN clearly beats the fair static reservoir, both above chance
    assert res["static_acc"] > chance, f"static reservoir at/under chance: {res['static_acc']:.3f}"
    assert res["sorn_acc"] > chance, f"SORN at/under chance: {res['sorn_acc']:.3f}"
    assert res["sorn_acc"] > res["static_acc"] + 0.03, (
        f"SORN must clearly beat the static reservoir: {res['sorn_acc']:.3f} vs {res['static_acc']:.3f}"
    )
    # (b) homeostasis signature: IP pulled mean E-rate to ~H_IP and dropped the rate variance
    assert abs(res["sorn_rate_mean"] - H_IP) < 0.05, (
        f"IP should drive mean E-rate to ~H_IP={H_IP}: got {res['sorn_rate_mean']:.3f}"
    )
    assert res["sorn_rate_var"] < res["static_rate_var"], (
        f"IP should tighten the rate variance: {res['sorn_rate_var']:.4f} !< {res['static_rate_var']:.4f}"
    )
    # (c) structure signature: STDP broadened the (initially normalized) W_EE weight distribution
    assert res["sorn_std"] > res["static_std"], (
        f"STDP should broaden W_EE weights: {res['sorn_std']:.3f} !> {res['static_std']:.3f}"
    )
    # (d) determinism: same seed -> identical accuracies on a repeat run
    res2 = run_experiment(seed)
    assert res2["sorn_acc"] == res["sorn_acc"] and res2["static_acc"] == res["static_acc"], (
        "same seed must reproduce identical accuracies"
    )
    print(
        f"self-check OK: SORN {res['sorn_acc']:.1%} > static {res['static_acc']:.1%} "
        f"(+{margin:.1%}); mean E-rate -> {res['sorn_rate_mean']:.3f} ~ H_IP {H_IP}; "
        "rate var down; deterministic"
    )


if __name__ == "__main__":
    main()
