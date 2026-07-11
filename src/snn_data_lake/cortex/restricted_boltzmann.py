"""
Spiking Boltzmann machine -- an undirected energy-based generative model whose units are STOCHASTIC
LIF neurons and whose learning is SPIKE-DRIVEN (event-driven contrastive divergence). Pure stdlib.

The RBM (Smolensky 1986; Hinton, "Training Products of Experts by Minimizing Contrastive Divergence",
Neural Comput 14:1771, 2002) is a bipartite Markov random field: visible units v in {0,1}^nv and hidden
units h in {0,1}^nh with weights W (nv x nh) and biases b_v, b_h, joint p(v, h) prop exp(-E(v, h)),

    E(v, h) = -b_v . v - b_h . h - v^T W h .

Because there are no visible-visible or hidden-hidden edges the conditionals FACTORIZE, so a whole layer
is sampled at once. In the ORIGINAL rate RBM a layer was drawn by evaluating sigmoid(b + Wx) and flipping
Bernoulli coins, and learning was an outer product of those continuous sigmoids (CD-1). Sampling was
incidental; the model was rate-based.

THIS module makes the machine GENUINELY SPIKING, following Neftci, Das, Pedroni, Kreutz-Delgado & Cauwenberghs,
"Event-Driven Contrastive Divergence for Spiking Neuromorphic Systems" (Front. Neurosci. 7:272, 2013/2014),
which builds on the neural-sampling result (Buesing, Bill, Nessler & Maass 2011) that a network of stochastic
spiking neurons samples a Boltzmann distribution by SPIKING:

  * Each unit is a stochastic leaky-integrate-and-fire neuron. Its membrane V leaks toward rest
    (decay = exp(-1/tau_m)), integrates the weighted INPUT SPIKES from the opposite layer plus its bias as a
    synaptic current, and receives per-step membrane noise. When V crosses threshold it emits a DISCRETE SPIKE
    and resets. A spike is a Monte-Carlo sample of that unit being ON; over a short window the network samples
    the joint distribution by spiking (block-Gibbs realised as two spiking populations driving each other).
  * Learning is EVENT-DRIVEN contrastive divergence == spike-timing coincidence plasticity (STDP). Weight
    W_ij is changed only by COINCIDENT pre/post spikes: in the data (clamped) phase co-spikes POTENTIATE, in
    the free-running reconstruction phase co-spikes DEPRESS, and the net update is the contrast of the two
    integrated coincidence counts. A synapse whose pre or post neuron never fires in either phase receives
    EXACTLY zero update -- the defining event-driven property, impossible for an outer product of continuous
    sigmoids (which touches every synapse). Integrated over the sampling window the STDP contrast estimates
    <v_i h_j>_data - <v_i h_j>_recon, i.e. the CD-1 gradient -- now carried entirely by spikes.

The free energy of a visible vector (hidden units summed out) is the standard RBM score
    F(v) = -b_v . v - sum_j softplus(b_h_j + sum_i v_i W_ij) ,   softplus(x) = log(1 + exp(x))
and lower F(v) means higher modelled probability p(v) prop exp(-F(v)); it is a deterministic function of the
LEARNED weights and needs no sampling. Training (by spikes) drives the data's free energy below random and
lets the spiking network denoise a corrupted pattern back toward the nearest learned mode.

HONEST COST: spiking-CD is a Monte-Carlo estimator on a finite window, so it is NOISIER than exact CD-1. The
reconstruction-error curve floors at the residual spike-sampling variance (it does not reach ~0 like exact
mean-field), and denoising is stochastic-but-seeded. The self-check reports the REAL numbers; nothing is faked.

DISTINCT from siblings in this repo (stated, having read them):
  * neural_sampling.py's BoltzmannSampler is a FIXED-weight single-site Gibbs sampler of a GIVEN, fully-visible
    Boltzmann/Ising joint p(z): its whole job is to MATCH a known analytic distribution -- nothing is learned,
    there are no hidden units, no CD. THIS module is a bipartite visible/hidden spiking RBM whose weights are
    LEARNED from data by spike-driven contrastive divergence; its signature is that spiking training lowers
    reconstruction error and drives the data's free energy below random. Different object (learned latent-variable
    generative model vs fixed known joint), different capability (unsupervised generative LEARNING by spikes vs
    sampling a given distribution).
  * predictive_coding.py is a DIRECTED generative hierarchy trained by gradient on prediction ERROR (top-down
    prediction, bottom-up error message-passing). A spiking RBM is an UNDIRECTED energy-based model trained by
    event-driven CD -- a different model class AND a different learning rule.
  * research/hopfield_baseline.py and research/dense_associative_memory.py are DETERMINISTIC associative-recall
    attractors (one-shot energy-descent recall of stored patterns). THIS is a STOCHASTIC bipartite generative
    model that is SAMPLED BY SPIKING and LEARNED by spike coincidences, not one-shot recall of hand-stored memories.

The ceiling is a small binary spiking RBM trained by event-driven CD on synthetic prototype patterns -- the
spike-based energy-based-generative-LEARNING level, NOT a deep belief net; no persistent-CD, no annealed
importance sampling for the true partition function; dims and the sampling window kept tiny so it trains in a
couple seconds and the signatures are stable at a fixed seed. Not cognition.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.restricted_boltzmann   # (also run by selfchecks)
"""
import math
import random

from snn_data_lake.substrate._mathutils import sigmoid


def _softplus(x):
    """Numerically safe log(1 + exp(x))."""
    if x > 30.0:
        return x
    if x < -30.0:
        return 0.0
    return math.log1p(math.exp(x))


class RBM:
    """Spiking Boltzmann machine over v in {0,1}^nv and h in {0,1}^nh, learned by event-driven CD.

    Weights W (nv x nh) and biases b_v, b_h define the bipartite Boltzmann joint. Both layers are populations
    of STOCHASTIC LIF neurons: a unit's membrane leaks, integrates the weighted input spikes from the opposite
    layer plus its bias, takes per-step noise, and emits a DISCRETE SPIKE (a sample of the unit being ON) when
    it crosses threshold. sample_hidden/sample_visible/gibbs sample the model BY SPIKING; train() runs
    event-driven contrastive divergence == spike-coincidence STDP (co-spikes in the clamped-data phase minus
    co-spikes in the free-running reconstruction phase). free_energy scores a visible vector (lower = higher
    modelled probability) from the learned weights. Deterministic given seed: one random.Random(seed) drives
    weight init AND every stochastic spike, so the same data/seed reproduce the run exactly."""

    def __init__(self, n_visible, n_hidden, seed=0):
        self.nv = n_visible
        self.nh = n_hidden
        self.rng = random.Random(seed)
        # small symmetric-range init; biases start at zero (Hinton's practical default).
        self.W = [[self.rng.uniform(-0.1, 0.1) for _ in range(n_hidden)] for _ in range(n_visible)]
        self.b_v = [0.0] * n_visible
        self.b_h = [0.0] * n_hidden
        # ---- stochastic-LIF sampling parameters (the spiking substrate) ----
        self.tau_m = 4.0                          # membrane time constant (steps)
        self.decay = math.exp(-1.0 / self.tau_m)  # leak-toward-rest factor per step
        self.v_threshold = 1.0                    # spike threshold
        self.v_reset = 0.0                         # post-spike reset potential
        self.noise = 0.30                          # per-step membrane noise (drives stochastic firing)
        self.gain = 1.5                            # synaptic-current gain into the membrane
        self.window = 30                           # sampling window: micro-steps per block-Gibbs half-step

    # ---- stochastic LIF substrate ---------------------------------------------------------------
    def _lif_step(self, u, drive):
        """Advance one LIF micro-step for a layer of membranes `u` (mutated in place) under synaptic `drive`.

        V <- decay*V + gain*current + noise; on threshold crossing emit a DISCRETE spike (1) and reset.
        Returns the layer's binary spike vector for this step (each entry in {0, 1})."""
        spikes = [0] * len(u)
        g = self.gain
        thr = self.v_threshold
        for k in range(len(u)):
            uk = u[k] * self.decay + g * drive[k] + self.rng.gauss(0.0, self.noise)
            if uk >= thr:
                spikes[k] = 1
                uk = self.v_reset
            u[k] = uk
        return spikes

    # ---- conditionals (the factorized layer distributions the spiking sampler targets) ----------
    def hidden_probs(self, v):
        """Analytic p(h_j = 1 | v) = sigmoid(b_h_j + sum_i v_i W_ij) -- the ON-probability the stochastic
        LIF hidden units approximate by their windowed spike rate."""
        return [sigmoid(self.b_h[j] + sum(v[i] * self.W[i][j] for i in range(self.nv)))
                for j in range(self.nh)]

    def visible_probs(self, h):
        """Analytic p(v_i = 1 | h) = sigmoid(b_v_i + sum_j W_ij h_j) -- the ON-probability the stochastic
        LIF visible units approximate by their windowed spike rate."""
        return [sigmoid(self.b_v[i] + sum(self.W[i][j] * h[j] for j in range(self.nh)))
                for i in range(self.nv)]

    def _sample(self, probs):
        """Bernoulli sample a binary vector from per-unit probabilities (uses self.rng). Kept for the
        analytic conditionals; the model's own layer sampling is done by _spike_layer (LIF spikes)."""
        return [1 if self.rng.random() < p else 0 for p in probs]

    def _spike_layer(self, drive, u):
        """Sample a whole layer BY SPIKING: run the stochastic LIF over `self.window` micro-steps with the
        given per-unit `drive` (bias + weighted input spikes from the opposite layer). Membranes start from
        `u`. Returns (rate, spikes): rate[k] = spike_count_k / window (the spike-rate readout, an estimate of
        p(unit ON)); spikes[k] = 1 if unit k fired at least once in the window else 0 (the discrete sample)."""
        counts = [0] * len(drive)
        for _ in range(self.window):
            s = self._lif_step(u, drive)
            for k in range(len(drive)):
                counts[k] += s[k]
        inv = 1.0 / self.window
        rate = [c * inv for c in counts]
        spikes = [1 if c > 0 else 0 for c in counts]
        return rate, spikes

    def sample_hidden(self, v):
        """Sample the hidden layer given visible vector v BY SPIKING. Returns (rate, h): rate = per-unit
        windowed spike rate (estimate of p(h|v)); h = discrete binary spike sample (each entry in {0, 1})."""
        drive = [self.b_h[j] + sum(v[i] * self.W[i][j] for i in range(self.nv)) for j in range(self.nh)]
        return self._spike_layer(drive, [0.0] * self.nh)

    def sample_visible(self, h):
        """Sample the visible layer given hidden vector h BY SPIKING. Returns (rate, v): rate = per-unit
        windowed spike rate (estimate of p(v|h)); v = discrete binary spike sample (each entry in {0, 1})."""
        drive = [self.b_v[i] + sum(self.W[i][j] * h[j] for j in range(self.nh)) for i in range(self.nv)]
        return self._spike_layer(drive, [0.0] * self.nv)

    # ---- generation / denoising (by spiking) ----------------------------------------------------
    def reconstruct(self, v):
        """One spiking pass v -> h -> v that denoises toward the nearest learned mode.

        Samples the hidden layer's spike RATE from v, then drives the visible layer with that rate and reads
        out its spike rate over a doubled window, thresholding at 0.5. Averaging the spikes suppresses the
        Monte-Carlo sampling noise so a trained machine cleans a corrupted input; it is still spike-based
        (the message between layers is a windowed spike rate), not a noiseless mean-field pass."""
        wide = self.window * 2
        # hidden spike-rate from v (longer window for a stable readout)
        uh = [0.0] * self.nh
        ch = [0] * self.nh
        drive_h = [self.b_h[j] + sum(v[i] * self.W[i][j] for i in range(self.nv)) for j in range(self.nh)]
        for _ in range(wide):
            s = self._lif_step(uh, drive_h)
            for j in range(self.nh):
                ch[j] += s[j]
        h_rate = [ch[j] / wide for j in range(self.nh)]
        # visible spike-rate driven by the hidden rate
        uv = [0.0] * self.nv
        cv = [0] * self.nv
        drive_v = [self.b_v[i] + sum(self.W[i][j] * h_rate[j] for j in range(self.nh)) for i in range(self.nv)]
        for _ in range(wide):
            s = self._lif_step(uv, drive_v)
            for i in range(self.nv):
                cv[i] += s[i]
        return [1 if cv[i] / wide >= 0.5 else 0 for i in range(self.nv)]

    def gibbs(self, v0, k):
        """k full steps of block Gibbs sampling starting from v0, realised BY SPIKING (each half-step is a
        stochastic-LIF spike sample of a layer). Returns a binary visible spike sample."""
        v = list(v0)
        for _ in range(k):
            _rh, h = self.sample_hidden(v)
            _rv, v = self.sample_visible(h)
        return v

    # ---- energy -------------------------------------------------------------------------------
    def free_energy(self, v):
        """Free energy F(v) = -b_v.v - sum_j softplus(b_h_j + sum_i v_i W_ij); lower F = higher p(v). A
        deterministic score of the LEARNED weights (no sampling needed)."""
        vis = -sum(self.b_v[i] * v[i] for i in range(self.nv))
        hid = 0.0
        for j in range(self.nh):
            hid += _softplus(self.b_h[j] + sum(v[i] * self.W[i][j] for i in range(self.nv)))
        return vis - hid

    # ---- spiking phases + event-driven CD (the spike-driven learning) ---------------------------
    def _data_phase(self, v0):
        """Clamped (positive) phase. Visible units are CLAMPED to the data: an ON data bit spikes tonically
        (every step), an OFF bit stays silent. Hidden units are stochastic-LIF neurons driven by those visible
        spikes. Accumulates, over `self.window` steps, the integer coincidence counts C[i][j] = #steps in which
        visible i AND hidden j spiked TOGETHER (the STDP pre/post coincidence), plus per-unit spike counts.
        Returns (coinc, cv, ch) -- all integer spike statistics."""
        uh = [0.0] * self.nh
        coinc = [[0] * self.nh for _ in range(self.nv)]
        cv = [0] * self.nv
        ch = [0] * self.nh
        for _ in range(self.window):
            vs = v0                                   # clamped tonic data spikes
            drive_h = [self.b_h[j] + sum(vs[i] * self.W[i][j] for i in range(self.nv)) for j in range(self.nh)]
            hs = self._lif_step(uh, drive_h)
            for i in range(self.nv):
                if vs[i]:
                    cv[i] += 1
                    row = coinc[i]
                    for j in range(self.nh):
                        if hs[j]:
                            row[j] += 1                # co-spike -> STDP coincidence event
            for j in range(self.nh):
                ch[j] += hs[j]
        return coinc, cv, ch

    def _recon_phase(self, h0):
        """Free-running (negative) phase: a spiking block-Gibbs chain. Starting from the data-phase hidden
        spikes h0, each step samples visible spikes from the current hidden spikes then hidden spikes from
        those visible spikes (both stochastic LIF). Accumulates the same integer coincidence + spike-count
        statistics from the RECONSTRUCTION spikes. Returns (coinc, cv, ch)."""
        uv = [0.0] * self.nv
        uh = [0.0] * self.nh
        hs = list(h0)
        coinc = [[0] * self.nh for _ in range(self.nv)]
        cv = [0] * self.nv
        ch = [0] * self.nh
        for _ in range(self.window):
            drive_v = [self.b_v[i] + sum(self.W[i][j] * hs[j] for j in range(self.nh)) for i in range(self.nv)]
            vs = self._lif_step(uv, drive_v)
            drive_h = [self.b_h[j] + sum(vs[i] * self.W[i][j] for i in range(self.nv)) for j in range(self.nh)]
            hs = self._lif_step(uh, drive_h)
            for i in range(self.nv):
                if vs[i]:
                    cv[i] += 1
                    row = coinc[i]
                    for j in range(self.nh):
                        if hs[j]:
                            row[j] += 1                # co-spike -> STDP coincidence event
            for j in range(self.nh):
                ch[j] += hs[j]
        return coinc, cv, ch

    def _cd_step(self, v0, lr):
        """One event-driven contrastive-divergence update for a single data vector, driven ENTIRELY by spike
        coincidences. Runs the clamped-data spiking phase and the free-running reconstruction spiking phase,
        then W_ij += lr * (coinc_data - coinc_recon) / window and the analogous bias updates from spike counts.
        Every term is an integer spike/coincidence count, so a synapse with no co-spike in EITHER phase gets
        exactly zero update (the event-driven property). Returns a stats dict for inspection/self-check."""
        cd, cvd, chd = self._data_phase(v0)
        h0 = [1 if chd[j] > 0 else 0 for j in range(self.nh)]     # data-phase hidden spike sample
        cr, cvr, chr_ = self._recon_phase(h0)
        inv = lr / self.window
        for i in range(self.nv):
            Wi = self.W[i]
            cdi, cri = cd[i], cr[i]
            for j in range(self.nh):
                Wi[j] += inv * (cdi[j] - cri[j])                  # STDP: potentiate on data co-spikes, depress on recon co-spikes
            self.b_v[i] += inv * (cvd[i] - cvr[i])
        for j in range(self.nh):
            self.b_h[j] += inv * (chd[j] - chr_[j])
        return {"coinc_data": cd, "coinc_recon": cr, "cv_data": cvd, "cv_recon": cvr,
                "ch_data": chd, "ch_recon": chr_, "recon_rate": [cvr[i] / self.window for i in range(self.nv)]}

    # ---- learning (event-driven contrastive divergence) -----------------------------------------
    def train(self, data, epochs=40, lr=0.1):
        """Event-driven CD training over `data` (list of binary vectors). Each vector runs a clamped-data
        spiking phase and a free-running reconstruction spiking phase; weights move on the CONTRAST of spike
        coincidences (STDP). Returns the per-epoch mean reconstruction error (mean squared error between each
        v0 and the reconstruction-phase visible spike RATE). Deterministic data order (no shuffle) so the same
        data/seed give an identical curve."""
        curve = []
        nv = self.nv
        for _ in range(epochs):
            epoch_err = 0.0
            for v0 in data:
                stats = self._cd_step(v0, lr)
                rr = stats["recon_rate"]
                epoch_err += sum((v0[i] - rr[i]) ** 2 for i in range(nv)) / nv
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


def _recon_accuracy(model, protos):
    """Mean per-bit reconstruction accuracy of a model over clean prototypes (spiking reconstruct)."""
    correct = 0
    total = 0
    for p in protos:
        r = model.reconstruct(p)
        correct += sum(1 for x, y in zip(r, p) if x == y)
        total += len(p)
    return correct / total


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

    # ---- 1. event-driven-CD (spike-driven) learning lowers reconstruction error ----
    rbm = RBM(nv, nh, seed=0)
    curve = rbm.train(data, epochs=50, lr=0.1)
    first_err, last_err = curve[0], curve[-1]

    # ---- 2. FUNCTION: trained machine reconstructs prototypes far above an untrained control ----
    trained_acc = _recon_accuracy(rbm, protos)
    control = RBM(nv, nh, seed=7)                        # untrained (random weights) baseline
    control_acc = _recon_accuracy(control, protos)

    # ---- 3. the learned model scores data above random (lower free energy) ----
    f_data = sum(rbm.free_energy(v) for v in data) / len(data)
    rand_rng = random.Random(999)
    randoms = [[rand_rng.randint(0, 1) for _ in range(nv)] for _ in range(200)]
    f_rand = sum(rbm.free_energy(v) for v in randoms) / len(randoms)
    f_margin = f_rand - f_data                           # positive => data lower F => more probable

    # ---- 4. denoising: corrupt a prototype, reconstruct (by spiking) toward the true mode ----
    corrupt = list(proto_a)
    for idx in (2, 5, 9):                                # flip 3 bits (deterministic)
        corrupt[idx] = 1 - corrupt[idx]
    before = _hamming(corrupt, proto_a)
    cleaned = rbm.reconstruct(corrupt)
    after = _hamming(cleaned, proto_a)

    # ---- 5. GENUINELY SPIKING: discrete spikes + event-driven (spike-coincidence) learning ----
    probe = RBM(nv, nh, seed=3)
    _rh, h_spikes = probe.sample_hidden(proto_a)         # sample the hidden layer BY SPIKING
    _rv, v_spikes = probe.sample_visible(h_spikes)       # sample the visible layer BY SPIKING
    spikes_binary = set(h_spikes) <= {0, 1} and set(v_spikes) <= {0, 1}
    spikes_fire = sum(h_spikes) > 0 and sum(v_spikes) > 0
    # one CD step: the update is driven purely by integer spike coincidences.
    stats = probe._cd_step(proto_a, lr=0.1)
    cd_int = all(isinstance(stats["coinc_data"][i][j], int) and isinstance(stats["coinc_recon"][i][j], int)
                 for i in range(nv) for j in range(nh))
    both_zero = [(i, j) for i in range(nv) for j in range(nh)
                 if stats["coinc_data"][i][j] == 0 and stats["coinc_recon"][i][j] == 0]
    any_coincident = any(stats["coinc_data"][i][j] > 0 or stats["coinc_recon"][i][j] > 0
                         for i in range(nv) for j in range(nh))
    import sys as _sys
    no_tensors = "numpy" not in _sys.modules and "torch" not in _sys.modules

    # ---- 6. determinism: same data/seed -> identical trained model + identical curve ----
    twin = RBM(nv, nh, seed=0)
    twin_curve = twin.train(data, epochs=50, lr=0.1)
    same_curve = twin_curve == curve
    same_weights = twin.W == rbm.W and twin.b_v == rbm.b_v and twin.b_h == rbm.b_h

    print("=" * 72)
    print("SPIKING BOLTZMANN MACHINE -- event-driven contrastive divergence (Neftci 2014)")
    print("=" * 72)
    print(f"recon-error curve head {[round(v, 4) for v in curve[:3]]}")
    print(f"recon-error curve tail {[round(v, 4) for v in curve[-3:]]}")
    print(f"  epoch 1 {first_err:.4f} -> final {last_err:.4f}  ({first_err / last_err:.2f}x lower)")
    print(f"reconstruction accuracy: trained {trained_acc:.3f}  vs  untrained control {control_acc:.3f}")
    print(f"free energy: F(data) {f_data:+.3f}   F(random) {f_rand:+.3f}   margin {f_margin:+.3f}")
    print("  (lower F(v) = higher modelled p(v); data must sit BELOW random)")
    print(f"denoise (spiking): corrupt proto_a (flip 3 bits) Hamming {before} -> reconstruct {after}")
    print(f"spiking: hidden spikes {h_spikes} (discrete, fired={sum(h_spikes)})")
    print(f"  event-driven learning: {len(both_zero)}/{nv * nh} synapses had NO spike coincidence "
          f"in either phase -> exactly zero update")
    print(f"determinism: identical curve {same_curve}; identical weights {same_weights}")
    print("=" * 72)

    # ---- self-check: deterministic asserts, zero-dep stdlib, ASCII-only prints ----
    # (a) FUNCTION -- spike-driven CD lowers reconstruction error; the machine completes patterns above a
    #     fair untrained baseline and assigns the data lower free energy than random vectors.
    assert last_err < 0.75 * first_err, \
        f"spiking CD must clearly lower reconstruction error ({first_err:.4f} -> {last_err:.4f})"
    assert trained_acc >= 0.80, f"trained machine must reconstruct prototypes well ({trained_acc:.3f})"
    assert trained_acc - control_acc >= 0.30, \
        f"trained must beat untrained control clearly ({trained_acc:.3f} vs {control_acc:.3f})"
    assert f_data < f_rand, f"F(data) must be below F(random) ({f_data:.3f} vs {f_rand:.3f})"
    assert f_margin > 0.8, f"free-energy margin must be clear ({f_margin:.3f})"
    assert after < before, f"spiking reconstruct must reduce distance to the true prototype ({before} -> {after})"
    assert after <= 2, f"spiking reconstruction must land on/near the true mode ({after})"
    # (b) GENUINELY SPIKING -- units sample by DISCRETE spikes; learning is spike-coincidence driven
    #     (integer counts; no-coincidence => no-update); no tensor libraries imported.
    assert spikes_binary, "layer samples must be DISCRETE spikes (each unit in {0, 1})"
    assert spikes_fire, "the stochastic LIF units must actually fire"
    assert cd_int, "spike coincidences must be integer counts of co-firing events"
    assert any_coincident, "some spike coincidences must occur to drive learning"
    assert len(both_zero) > 0, "event-driven test needs synapses with no co-spike in either phase"
    W_after = probe._cd_step.__self__.W  # (probe.W already updated by the CD step above)
    # re-derive: a fresh probe, snapshot, one CD step, confirm both-zero synapses are byte-identical.
    ev = RBM(nv, nh, seed=3)
    W_before = [row[:] for row in ev.W]
    st = ev._cd_step(proto_a, lr=0.1)
    bz = [(i, j) for i in range(nv) for j in range(nh)
          if st["coinc_data"][i][j] == 0 and st["coinc_recon"][i][j] == 0]
    assert all(ev.W[i][j] == W_before[i][j] for (i, j) in bz), \
        "event-driven: synapses with no spike coincidence must receive EXACTLY zero update"
    assert any(ev.W[i][j] != W_before[i][j] for i in range(nv) for j in range(nh)), \
        "spike-coincident synapses must move (learning happened)"
    assert no_tensors, "must be pure stdlib -- numpy/torch must NOT be imported"
    _ = W_after  # (silence: probe state already exercised)
    # (c) DETERMINISM: same data/seed -> identical trained model and identical curve.
    assert same_curve and same_weights, "same data/seed must give an identical trained model"

    print(f"self-check OK: spiking-CD recon error {first_err:.3f} -> {last_err:.3f}; "
          f"recon acc {trained_acc:.2f} (control {control_acc:.2f}); "
          f"F(data) {f_data:.2f} < F(random) {f_rand:.2f} (margin {f_margin:.2f}); "
          f"denoise Hamming {before} -> {after}; discrete spikes + event-driven STDP; deterministic")


if __name__ == "__main__":
    main()
