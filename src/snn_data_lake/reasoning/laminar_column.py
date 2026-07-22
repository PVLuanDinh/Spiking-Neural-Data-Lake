r"""
laminar_column -- four NO-BACKPROP readouts on ONE feature set, ranked head to head.

This module asks a single honest question with a fair, four-way controlled comparison: does the
canonical neocortical laminar motif (L4 -> L2/3 -> L5, with an L6 top-down gain loop) buy anything
over a flat linear readout -- and does a REAL spiking cortical column carry class information in its
firing, or does its resting-state dynamics wash the stimulus out? Every variant learns with LOCAL
plasticity only. There is NO gradient tape, NO weight transport, NO Adam anywhere: the hidden layer
learns by FEEDBACK ALIGNMENT (a fixed random feedback matrix, Lillicrap et al. 2016) and the output
by a local delta rule; the single-layer baselines learn by an error-modulated Hebbian rule. numpy
and stdlib only.

The four variants (all on the IDENTICAL feature matrix + the IDENTICAL held-out split):

  (1) BASELINE  -- FrozenExpert, a single-layer LOCAL error-modulated Hebbian readout on the raw
                   features (substrate.similarity_sized_growth.FrozenExpert). The flat reference.

  (2) B' LAMINAR-LOCAL -- the L4 -> L2/3 -> L5 -> L6 four-role motif, brain-native:
        L4   : a FIXED random tanh expansion of the thalamic input (read-only).
        L2/3 : the hidden FEATURE layer, trained by FEEDBACK ALIGNMENT -- the per-neuron credit
               e_hidden = B . delta_out is delivered by a FIXED RANDOM feedback matrix B (never
               W2 transposed), gating a local three-factor update. This is the mode="fa" mechanism
               of plasticity.feedback_alignment lifted into a sized numpy layer (that class hard-
               wires 3-bit parity; the mechanism is what transfers). NEUROGENESIS-FREEZE lives here:
               a per-hidden-unit read-only mask -- old units are frozen bit-exact, freshly grown
               units stay plastic (the FrozenExpert freeze idea, per unit).
        L5   : the OUTPUT layer, trained by a LOCAL delta rule  W2 -= lr * delta_out (x) pre.
        L6   : a FIXED random top-down feedback producing a per-L2/3 sigmoid GAIN, applied on one
               second forward pass (h_mod = h * gain) -- the reference's L6 role, weight-transport-
               free.

  (3) A' COLUMN-DYNAMICS -- inject the feature vector as a per-neuron thalamic->L4 external current
        into the Potjans-Diesmann spiking column (cortex.cortical_microcircuit.CorticalMicrocircuit),
        simulate, and read the per-neuron L2/3e + L5e spike counts as the laminar feature; then a
        FrozenExpert local readout on those firing responses. The injection sets only the external
        drive (mu_ext); the recurrent synapses are never touched. The injection GAIN is fixed on a
        regime criterion, NOT tuned to accuracy: it is the strongest drive that keeps the stimulated
        L5e pool inside the column's certified asynchronous-irregular band (rate < ~30 Hz), the
        regime the microcircuit's own self-check validates.

  (4) SYNTHESIS -- the A' column-read firing features fed into the B' feedback-alignment laminar-
        local readout. Does laminar structure rescue the column features, or is the column the
        bottleneck?

FEATURE SET: a deterministic synthetic K=6 task in R^12 where each class is the union of TWO
Gaussian sub-blobs at random centers. Two blobs per class makes it NOT linearly separable (a single
per-class linear score cannot own two separated regions), so a hidden layer has room to help and the
baseline-vs-laminar comparison can actually discriminate. Chance = majority-class frequency.

HONESTY: a variant that ties or loses the flat baseline is a VALID result and is reported as-is; no
number is tuned to win. The spiking column is a resting-state DYNAMICS model, and whether an injected
stimulus survives its recurrent background is exactly the open question this file measures.

Refs: Lillicrap, Cownden, Tweed & Akerman, Nat. Commun. 7 (2016) (random feedback alignment);
Potjans & Diesmann, Cereb. Cortex 24 (2014) (the microcircuit). Deterministic (seeded); ASCII-only.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.laminar_column
"""

import math
import random

import numpy as np

from snn_data_lake.cortex.cortical_microcircuit import CorticalMicrocircuit
from snn_data_lake.substrate.similarity_sized_growth import FrozenExpert

# --- Potjans-Diesmann population indices (canonical order in cortical_microcircuit.POP_NAMES) ----
L23E, L4E, L4I, L5E = 0, 2, 3, 4

# A' injection gain, fixed on a REGIME criterion (see module docstring): the strongest per-neuron
# L4 drive that keeps the stimulated L5e pool inside the column's certified asynchronous-irregular
# band (< ~30 Hz). Chosen independent of downstream accuracy; the self-check re-verifies the band.
INJECT_GAIN = 0.12
SIM_T_MS = 100.0
SIM_SETTLE_MS = 40.0
COLUMN_SCALE = 0.008
COLUMN_SEED = 0


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# =====================================================================================================
# Variant B' -- the laminar-local readout (feedback alignment + local delta + L6 gain + freeze).
# =====================================================================================================
class LaminarLocalReadout:
    """L4 -> L2/3 -> L5 -> L6 cortical-column readout, LOCAL learning only (no gradient tape).

    L4 is a fixed random tanh expansion (read-only thalamocortical transform). L2/3 is the hidden
    feature layer, learned by FEEDBACK ALIGNMENT: the hidden credit e_hidden = B . delta_out uses a
    FIXED RANDOM feedback matrix B (weight-transport-free), gating a local three-factor update. L5 is
    the output layer, learned by a local delta rule. L6 is a fixed random top-down feedback producing
    a per-L2/3 sigmoid gain applied on a second forward pass. NEUROGENESIS-FREEZE: grow() freezes the
    existing L2/3 units read-only (bit-exact) and appends fresh plastic units."""

    def __init__(self, in_dim, n_classes, n4=64, n23=32, lr_hid=0.08, lr_out=0.08, seed=0):
        self.in_dim = int(in_dim)
        self.K = int(n_classes)
        self.n4 = int(n4)
        self.lr_hid = float(lr_hid)
        self.lr_out = float(lr_out)
        rng = np.random.default_rng(seed)

        # L4: fixed random tanh input expansion (read-only). std ~ 1/sqrt(in_dim) keeps pre-acts sane.
        self.W4 = rng.normal(0.0, 1.0 / math.sqrt(self.in_dim), size=(self.n4, self.in_dim))
        self.b4 = np.zeros(self.n4)

        # L2/3: plastic hidden layer (feedback alignment). n23 units to start; grow() adds more.
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(self.n4), size=(n23, self.n4))
        self.b1 = np.zeros(n23)
        self._W1_init = self.W1.copy()  # birth snapshot -> prove the local rule MOVES the weights

        # L5: plastic output layer (local delta rule).
        self.W2 = rng.normal(0.0, 1.0 / math.sqrt(n23), size=(self.K, n23))
        self.b2 = np.zeros(self.K)

        # B: FIXED random feedback for L2/3 credit (feedback alignment -- NOT W2 transposed).
        self.B = rng.normal(0.0, 1.0, size=(n23, self.K))
        # L6: FIXED random top-down feedback -> per-L2/3 sigmoid gain.
        self.W6 = rng.normal(0.0, 1.0, size=(n23, self.K))
        self.b6 = np.zeros(n23)

        self._frozen = np.zeros(n23, dtype=bool)  # per-L2/3-unit read-only mask
        self._grow_seed = int(seed) * 100 + 7

    @property
    def n_hidden(self):
        return self.W1.shape[0]

    def forward(self, x, use_l6=True):
        a4 = np.tanh(self.W4 @ x + self.b4)  # L4 fixed expansion
        pre = self.W1 @ a4 + self.b1  # L2/3 pre-activation
        h = np.tanh(pre)  # L2/3 feature activations
        logits1 = self.W2 @ h + self.b2  # L5 first pass
        if not use_l6:
            return _softmax(logits1), (a4, h, h, logits1)
        gain = 1.0 / (1.0 + np.exp(-(self.W6 @ logits1 + self.b6)))  # L6 sigmoid gain in (0,1)
        h_mod = h * gain
        logits = self.W2 @ h_mod + self.b2  # L5 second pass
        return _softmax(logits), (a4, h, h_mod, logits1)

    def train_step(self, x, target):
        p, (a4, h, h_mod, _logits1) = self.forward(x, use_l6=True)
        delta_out = p - target  # softmax + cross-entropy output error

        # L5 output: LOCAL delta rule (presynaptic = the actual second-pass input h_mod).
        self.W2 -= self.lr_out * np.outer(delta_out, h_mod)
        self.b2 -= self.lr_out * delta_out

        # L2/3 hidden: FEEDBACK ALIGNMENT credit through the FIXED random B (no weight transport),
        # gating a local three-factor update. phi' is the tanh surrogate slope.
        e_hidden = self.B @ delta_out
        phip = 1.0 - h * h
        mod = e_hidden * phip
        dW1 = -self.lr_hid * np.outer(mod, a4)
        db1 = -self.lr_hid * mod
        if self._frozen.any():  # freeze: read-only units do not move
            dW1[self._frozen] = 0.0
            db1[self._frozen] = 0.0
        self.W1 += dW1
        self.b1 += db1

    def fit(self, X, y, epochs, seed=0):
        X = np.asarray(X, dtype=float)
        eye = np.eye(self.K)
        rng = random.Random(seed)
        order = list(range(len(y)))
        for _ in range(int(epochs)):
            rng.shuffle(order)
            for i in order:
                self.train_step(X[i], eye[int(y[i])])
        return self

    def grow(self, n_new, freeze_old=True):
        """NEUROGENESIS-FREEZE: mark every current L2/3 unit read-only (frozen bit-exact), then
        append n_new fresh PLASTIC units. New L5 fan-in columns start at zero (the grown units add
        nothing at the instant of growth); their B / W6 feedback rows are fresh random."""
        rng = np.random.default_rng(self._grow_seed)
        if freeze_old:
            self._frozen[:] = True
        new_W1 = rng.normal(0.0, 1.0 / math.sqrt(self.n4), size=(n_new, self.n4))
        self.W1 = np.vstack([self.W1, new_W1])
        self.b1 = np.concatenate([self.b1, np.zeros(n_new)])
        self._W1_init = np.vstack([self._W1_init, new_W1.copy()])
        self.W2 = np.hstack(
            [self.W2, np.zeros((self.K, n_new))]
        )  # zero fan-in: function-preserving
        self.B = np.vstack([self.B, rng.normal(0.0, 1.0, size=(n_new, self.K))])
        self.W6 = np.vstack([self.W6, rng.normal(0.0, 1.0, size=(n_new, self.K))])
        self.b6 = np.concatenate([self.b6, np.zeros(n_new)])
        self._frozen = np.concatenate([self._frozen, np.zeros(n_new, dtype=bool)])
        return self.n_hidden

    def predict(self, x):
        return int(np.argmax(self.forward(x, use_l6=True)[0]))

    def accuracy(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray([int(v) for v in y])
        preds = np.array([self.predict(X[i]) for i in range(len(y))])
        return float(np.mean(preds == y))

    def hidden_diverged(self):
        """Max abs change of the L2/3 weights from their birth init -- proof the local rule LEARNED."""
        return float(np.max(np.abs(self.W1 - self._W1_init)))


def train_laminar_local(Xtr, ytr, K, in_dim, seed=0, phase1=40, phase2=40, n23_init=32, n_new=16):
    """Two-phase B' training: learn L2/3 by feedback alignment, then GROW+FREEZE (old units read-only,
    fresh plastic units) and keep learning. Returns the trained readout plus a freeze audit."""
    net = LaminarLocalReadout(in_dim, K, n23=n23_init, seed=seed)
    net.fit(Xtr, ytr, epochs=phase1, seed=seed * 3 + 1)
    frozen_snapshot = net.W1.copy()  # value of the to-be-frozen rows at the moment of freeze
    n_old = net.n_hidden
    net.grow(n_new, freeze_old=True)
    net.fit(Xtr, ytr, epochs=phase2, seed=seed * 3 + 2)
    frozen_now = net.W1[:n_old]
    frozen_diff = float(np.max(np.abs(frozen_now - frozen_snapshot)))  # must be 0.0 (read-only)
    new_diff = float(np.max(np.abs(net.W1[n_old:] - net._W1_init[n_old:])))  # must be > 0 (plastic)
    audit = {
        "n_old": n_old,
        "n_hidden": net.n_hidden,
        "frozen_diff": frozen_diff,
        "new_diff": new_diff,
    }
    return net, audit


# =====================================================================================================
# Variant A' -- inject the feature into the real spiking column, read L2/3e + L5e firing.
# =====================================================================================================
def _pop_range(net, p):
    lo = net.pop_start[p]
    return range(lo, lo + net.n_pop[p])


def _recurrent_fingerprint(net):
    """A cheap deterministic fingerprint of the recurrent wiring + weights -- to PROVE A' leaves the
    column's recurrent synapses untouched (only the external drive mu_ext carries the stimulus)."""
    total = 0.0
    for src in range(net.n_total):
        for _tgt, w in net.out_adj[src]:
            total += w
    return (net.n_synapses, round(net.w_exc, 9), round(net.w_inh, 9), round(total, 6))


class ColumnEncoder:
    """Wraps one Potjans-Diesmann column. encode(x) injects a fixed random projection of x as a
    per-neuron external current onto the L4 pools, simulates, and returns the per-neuron L2/3e + L5e
    spike counts. The recurrent synapses are never modified; simulate() re-seeds its own noise from
    the column seed, so the background is identical across samples and the encoding is deterministic."""

    def __init__(self, in_dim, scale=COLUMN_SCALE, seed=COLUMN_SEED, gain=INJECT_GAIN, proj_seed=1):
        self.net = CorticalMicrocircuit(scale=scale, seed=seed)
        self.net.simulate(t_ms=SIM_T_MS, settle_ms=SIM_SETTLE_MS)  # prime mu_ext / spike buffers
        self.gain = float(gain)
        self._base_mu = list(self.net.mu_ext)  # the resting external drive, restored per sample
        self._l4 = list(_pop_range(self.net, L4E)) + list(_pop_range(self.net, L4I))
        rng = np.random.default_rng(proj_seed)
        # unit-variance rows: P @ x_standardized has std ~ 1, so `gain` sets the injected current std.
        self.P = rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(len(self._l4), in_dim))
        self.feat_pops = (L23E, L5E)
        self.feat_dim = sum(self.net.n_pop[p] for p in self.feat_pops)

    def encode(self, x):
        inj = self.gain * (self.P @ np.asarray(x, dtype=float))
        mu = list(self._base_mu)
        for k, idx in enumerate(self._l4):
            mu[idx] = self._base_mu[idx] + inj[k]
        self.net.mu_ext = mu
        self.net.simulate(t_ms=SIM_T_MS, settle_ms=SIM_SETTLE_MS)
        counts = []
        for p in self.feat_pops:
            for i in _pop_range(self.net, p):
                counts.append(len(self.net.spike_times[i]))
        return np.asarray(counts, dtype=float)

    def stimulated_l5e_rate(self, feats_matrix):
        """Mean L5e firing rate (Hz) across encoded samples -- the regime check for INJECT_GAIN."""
        n_l23e = self.net.n_pop[L23E]
        l5e_counts = feats_matrix[:, n_l23e:]
        return float(np.mean(l5e_counts) / (SIM_T_MS / 1000.0))


def encode_all(encoder, X, progress_label=""):
    feats = np.empty((len(X), encoder.feat_dim), dtype=float)
    for i in range(len(X)):
        feats[i] = encoder.encode(X[i])
        if progress_label and (i + 1) % 60 == 0:
            print(f"    [{progress_label}] encoded {i + 1}/{len(X)} column responses")
    return feats


# =====================================================================================================
# Shared: dataset, standardization, single-layer baseline readout.
# =====================================================================================================
def make_dataset(seed=7, D=12, K=6, n_train=35, n_test=15, n_sub=2, sep=3.0, noise=1.0):
    """K classes in R^D; each class = union of n_sub Gaussian sub-blobs at random centers -> NOT
    linearly separable. Balanced. Returns raw (unstandardized) train/test arrays."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, sep, size=(K, n_sub, D))
    Xtr, ytr, Xte, yte = [], [], [], []
    for c in range(K):
        for _ in range(n_train):
            s = int(rng.integers(0, n_sub))
            Xtr.append(centers[c, s] + rng.normal(0.0, noise, size=D))
            ytr.append(c)
        for _ in range(n_test):
            s = int(rng.integers(0, n_sub))
            Xte.append(centers[c, s] + rng.normal(0.0, noise, size=D))
            yte.append(c)
    return (np.asarray(Xtr), np.asarray(ytr), np.asarray(Xte), np.asarray(yte))


def standardize(train, test):
    """Standardize on TRAIN statistics only (no test leakage), applied to both."""
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (test - mean) / std


def baseline_accuracy(Xtr, ytr, Xte, yte, K, seed=0):
    """Variant (1)/(3) readout: FrozenExpert single-layer LOCAL error-modulated Hebbian."""
    exp = FrozenExpert(K, Xtr.shape[1])
    exp.learn(Xtr, list(ytr), epochs=40, seed=seed)
    return exp.accuracy(Xte, yte), exp


def majority_chance(y):
    vals, counts = np.unique(np.asarray(y), return_counts=True)
    return float(np.max(counts) / len(y))


# =====================================================================================================
# Self-check: the four held-out accuracies, the winner, and every honesty assert. Deterministic, ASCII.
# =====================================================================================================
def main():
    seed = 7
    Xtr_raw, ytr, Xte_raw, yte = make_dataset(seed=seed)
    K = int(max(ytr.max(), yte.max()) + 1)
    in_dim = Xtr_raw.shape[1]
    chance = majority_chance(yte)

    print("=" * 82)
    print("LAMINAR COLUMN -- four NO-BACKPROP readouts on ONE feature set, ranked head to head")
    print("=" * 82)
    print(
        f"task: K={K} classes in R^{in_dim}, each class = 2 Gaussian sub-blobs (NOT linearly "
        f"separable)"
    )
    print(f"      train N={len(ytr)}  held-out N={len(yte)}  chance (majority class)={chance:.3f}")
    print("-" * 82)

    # Standardize raw features on TRAIN stats (leak-free), used by variants (1) and (2).
    Xtr, Xte = standardize(Xtr_raw, Xte_raw)

    # ---- (1) BASELINE: flat single-layer local Hebbian readout --------------------------------
    acc1, _ = baseline_accuracy(Xtr, ytr, Xte, yte, K, seed=1001)
    print(
        f"(1) BASELINE  flat FrozenExpert (local Hebbian) on raw features   held-out acc = {acc1:.3f}"
    )

    # ---- (2) B' LAMINAR-LOCAL: FA hidden + delta out + L6 gain + neurogenesis-freeze ----------
    net2, audit2 = train_laminar_local(Xtr, ytr, K, in_dim, seed=2002)
    acc2 = net2.accuracy(Xte, yte)
    print(
        f"(2) B' LAMINAR-LOCAL  L4->L2/3(FA)->L5(delta)->L6 gain on raw       held-out acc = {acc2:.3f}"
    )
    print(
        f"      neurogenesis-freeze: L2/3 {audit2['n_old']} frozen (bit-exact, diff="
        f"{audit2['frozen_diff']:.1e}) + {audit2['n_hidden'] - audit2['n_old']} plastic grown "
        f"(moved, diff={audit2['new_diff']:.2e})"
    )

    # ---- Encode ALL samples through the REAL spiking column (shared by variants 3 and 4) -------
    print("-" * 82)
    print(
        f"    injecting features -> L4 of a {CorticalMicrocircuit(scale=COLUMN_SCALE).n_total}-neuron "
        f"Potjans-Diesmann column (gain={INJECT_GAIN}, {SIM_T_MS:.0f}ms x {len(ytr) + len(yte)} sims"
        f", ~1-2 min)..."
    )
    # Inject the STANDARDIZED features (unit-variance): the projection P has unit-variance rows, so
    # `gain` then sets the injected-current std -- the exact regime the gain criterion was fixed on.
    encoder = ColumnEncoder(in_dim, proj_seed=1)
    fp_before = _recurrent_fingerprint(encoder.net)
    Ctr_raw = encode_all(encoder, Xtr, progress_label="train")
    Cte_raw = encode_all(encoder, Xte, progress_label="test")
    fp_after = _recurrent_fingerprint(encoder.net)
    l5e_rate = encoder.stimulated_l5e_rate(np.vstack([Ctr_raw, Cte_raw]))
    print(
        f"    column feature dim = {encoder.feat_dim} (L2/3e {encoder.net.n_pop[L23E]} + L5e "
        f"{encoder.net.n_pop[L5E]} per-neuron spike counts); stimulated L5e rate = {l5e_rate:.1f} Hz"
    )

    # Standardize column features on TRAIN stats (leak-free), used by variants (3) and (4).
    Ctr, Cte = standardize(Ctr_raw, Cte_raw)

    # ---- (3) A' COLUMN-DYNAMICS: flat local readout on the column firing features -------------
    acc3, _ = baseline_accuracy(Ctr, ytr, Cte, yte, K, seed=3003)
    print(
        f"(3) A' COLUMN-DYNAMICS  flat FrozenExpert on column L2/3e+L5e firing  held-out acc = {acc3:.3f}"
    )

    # ---- (4) SYNTHESIS: column firing features -> B' laminar-local readout --------------------
    net4, audit4 = train_laminar_local(Ctr, ytr, K, Ctr.shape[1], seed=4004)
    acc4 = net4.accuracy(Cte, yte)
    print(
        f"(4) SYNTHESIS   column firing -> B' laminar-local (FA) readout        held-out acc = {acc4:.3f}"
    )

    # ---- winner ----
    accs = {
        "(1) baseline flat": acc1,
        "(2) B' laminar-local": acc2,
        "(3) A' column-dynamics": acc3,
        "(4) synthesis": acc4,
    }
    winner = max(accs, key=accs.__getitem__)
    print("-" * 82)
    print(
        f"chance = {chance:.3f}   |   "
        + "   ".join(f"{k.split()[0]}={v:.3f}" for k, v in accs.items())
    )
    print(f"WINNER: {winner}  (held-out acc = {accs[winner]:.3f})")

    # honest verdicts
    lam_vs_flat = acc2 - acc1
    col_vs_flat = acc3 - acc1
    syn_vs_lam = acc4 - acc2
    print("-" * 82)
    print("HONEST VERDICT:")
    print(
        f"  laminar structure vs flat baseline : B' {acc2:.3f} vs {acc1:.3f} "
        f"({'+' if lam_vs_flat >= 0 else ''}{lam_vs_flat:.3f}) -> "
        f"{'laminar HELPS' if lam_vs_flat > 0.01 else 'TIE / no gain' if abs(lam_vs_flat) <= 0.01 else 'laminar HURTS'}"
    )
    print(
        f"  spiking column vs raw features     : A' {acc3:.3f} vs {acc1:.3f} "
        f"({'+' if col_vs_flat >= 0 else ''}{col_vs_flat:.3f}) -> "
        f"{'column carries info, above raw' if col_vs_flat > 0.01 else 'column WASHES OUT some class info' if col_vs_flat < -0.01 else 'column neutral vs raw'} "
        f"(but well above chance {chance:.3f} either way: {acc3 > chance + 0.05})"
    )
    print(
        f"  synthesis vs laminar-on-raw        : (4) {acc4:.3f} vs (2) {acc2:.3f} "
        f"({'+' if syn_vs_lam >= 0 else ''}{syn_vs_lam:.3f}) -> column features "
        f"{'help' if syn_vs_lam > 0.01 else 'do not help' if syn_vs_lam <= 0.01 else ''} the laminar readout"
    )

    # =============================== ASSERTS (real numbers, ASCII) ==============================
    # every variant beats chance
    for name, a in accs.items():
        assert a > chance + 1e-9, f"{name} did not beat chance: {a:.3f} <= {chance:.3f}"

    # B' / synthesis hidden weights LEARN (diverge from init under the LOCAL feedback-alignment rule)
    assert net2.hidden_diverged() > 0.0, "B' L2/3 weights did not move under the local rule"
    assert net4.hidden_diverged() > 0.0, "synthesis L2/3 weights did not move under the local rule"
    # neurogenesis-freeze: old units bit-exact read-only, grown units actually plastic
    for tag, aud in (("B'", audit2), ("synthesis", audit4)):
        assert aud["frozen_diff"] == 0.0, (
            f"{tag} frozen L2/3 units must be bit-exact, got {aud['frozen_diff']:.2e}"
        )
        assert aud["new_diff"] > 0.0, (
            f"{tag} grown L2/3 units must be plastic, got {aud['new_diff']:.2e}"
        )

    # A' / synthesis leave the column's RECURRENT weights fixed (only the external drive carried info)
    assert fp_before == fp_after, (
        f"column recurrent weights changed during A': {fp_before} != {fp_after}"
    )

    # regime check: INJECT_GAIN kept the stimulated L5e pool inside the certified AI band (< ~30 Hz),
    # so A' was measured in the column's validated operating regime, not driven into saturation.
    assert l5e_rate < 30.0, f"injection drove L5e out of the AI band ({l5e_rate:.1f} Hz >= 30 Hz)"

    # DETERMINISM: an identical B' rebuild reproduces the accuracy + hidden weights; one column encode
    # reproduces its spike counts bit-for-bit.
    net2b, _ = train_laminar_local(Xtr, ytr, K, in_dim, seed=2002)
    assert net2b.accuracy(Xte, yte) == acc2, "B' training is not deterministic (accuracy differs)"
    assert np.array_equal(net2b.W1, net2.W1), "B' training is not deterministic (weights differ)"
    enc2 = ColumnEncoder(in_dim, proj_seed=1)
    assert np.array_equal(enc2.encode(Xte[0]), Cte_raw[0]), "column encode is not deterministic"

    print("-" * 82)
    print(
        f"self-check OK: all four variants beat chance ({chance:.3f}); B'/synthesis L2/3 learned by "
        f"feedback alignment (diverged {net2.hidden_diverged():.2f}/{net4.hidden_diverged():.2f}); "
        f"neurogenesis-freeze bit-exact; column recurrent weights FIXED; L5e in-band ({l5e_rate:.1f} "
        f"Hz < 30); deterministic. Winner: {winner} ({accs[winner]:.3f}). NO backprop -- learning is "
        f"feedback alignment + local delta/Hebbian only."
    )


if __name__ == "__main__":
    main()
