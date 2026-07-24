r"""
laminar_local_readout -- a NO-BACKPROP drop-in replacement for LaminarReadout.

LaminarReadout (brain_adapters/laminar_readout.py) is a torch nn.Module trained by Adam via a
reverse-mode gradient tape through the L4 -> L2/3 -> L5 -> L6 laminar motif. This module ports
the SAME four-role architecture but every layer learns LOCALLY -- the exact mechanism measured
as the winning variant B' of a four-way local-learning-readout study, here re-exposed
behind LaminarReadout's public interface (n_hidden / train_step / predict / grow) so it is a
true drop-in. Pure numpy. NO torch dependency, NO gradient tape, NO Adam, NO reverse-mode
differentiation call:

  L4   : a FIXED random tanh expansion of the input (read-only thalamocortical transform).
  L2/3 : the plastic hidden layer, learned by FEEDBACK ALIGNMENT (Lillicrap et al. 2016) --
         the hidden credit e_hidden = delta_out @ B is delivered through a FIXED RANDOM
         feedback matrix B (never W2 transposed -- weight-transport-free), gating a local
         three-factor update.
  L5   : the plastic output layer, learned by a LOCAL delta rule  W2 -= lr * delta_out (x) pre.
  L6   : a FIXED random top-down feedback producing a per-L2/3 sigmoid GAIN, applied on a
         SECOND forward pass (h_mod = h * gain) -- weight-transport-free, like B.
  NEUROGENESIS-FREEZE: a per-L2/3-unit boolean read-only mask. grow() freezes every existing
  L2/3 unit bit-exact, then appends fresh PLASTIC units with a ZERO L5 fan-in column (function-
  preserving: the column's output is unchanged at the instant of growth).

train_step() batches the source's per-sample local rule into one vectorized minibatch update
(average delta/credit over the batch); this is a numerically different but mechanism-identical
pass -- the self-check verifies it by confirming loss decreases and weights move.

Refs: Lillicrap, Cownden, Tweed & Akerman, "Random synaptic feedback weights support error
backpropagation for deep learning", Nat. Commun. 7 (2016) (feedback alignment). The mechanism
this ports: the variant-B' LaminarLocalReadout (L4 -> L2/3 -> L5 -> L6, local learning only).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.brain_adapters.laminar_local_readout
"""

import math
import sys

import numpy as np


def _softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LaminarLocalReadout:
    """No-backprop drop-in for LaminarReadout. Same public shape (n_hidden property,
    train_step(X, y, lr, steps) -> loss, predict(X) -> logits, grow(k, freeze_old) -> n_hidden)
    but every parameter update is LOCAL: L2/3 by feedback alignment through a fixed random
    matrix B, L5 by a local delta rule, L6 is a fixed random top-down gain. See module
    docstring for the full mechanism (the variant-B' L4 -> L2/3 -> L5 -> L6 local-learning readout)."""

    def __init__(self, in_dim, n_classes=10, n4=96, n23=64, seed=0, use_l6=True):
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.n4 = int(n4)
        self.use_l6 = bool(use_l6)
        self._seed = int(seed)
        self._lr = 0.08  # default local learning rate used by observe(); train_step() always
        # uses the CALLER-supplied lr and never reads this.
        rng = np.random.default_rng(seed)

        # L4: fixed random tanh expansion (read-only thalamocortical transform).
        self.W4 = rng.normal(0.0, 1.0 / math.sqrt(self.in_dim), size=(self.n4, self.in_dim))
        self.b4 = np.zeros(self.n4)

        # L2/3: plastic hidden layer, learned by feedback alignment. n23 units to start;
        # grow() adds more.
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(self.n4), size=(n23, self.n4))
        self.b1 = np.zeros(n23)
        self._W1_init = self.W1.copy()  # birth snapshot -> prove the local rule MOVES the weights

        # L5: plastic output layer, learned by a local delta rule.
        self.W2 = rng.normal(0.0, 1.0 / math.sqrt(n23), size=(self.n_classes, n23))
        self.b2 = np.zeros(self.n_classes)

        # B: FIXED random feedback for L2/3 credit (feedback alignment -- NOT W2 transposed).
        self.B = rng.normal(0.0, 1.0, size=(n23, self.n_classes))
        # L6: FIXED random top-down feedback -> per-L2/3 sigmoid gain.
        self.W6 = rng.normal(0.0, 1.0, size=(n23, self.n_classes))
        self.b6 = np.zeros(n23)

        self._frozen = np.zeros(n23, dtype=bool)  # per-L2/3-unit read-only mask
        self._grow_seed = self._seed * 100 + 7  # separate stream, mirrors the source exactly

    @property
    def n_hidden(self):
        """Current L2/3 width (the growable dimension)."""
        return self.W1.shape[0]

    def _forward_batch(self, X, use_l6):
        """Batched two-pass forward. Returns (a4, h, h_mod, logits) -- h_mod == h and logits ==
        logits1 when use_l6 is False (pure feedforward, L6 path skipped)."""
        a4 = np.tanh(X @ self.W4.T + self.b4)  # [N,n4]  L4 fixed expansion
        h = np.tanh(a4 @ self.W1.T + self.b1)  # [N,n23] L2/3 feature activations
        logits1 = h @ self.W2.T + self.b2  # [N,K]   L5 first pass
        if not use_l6:
            return a4, h, h, logits1
        gain = 1.0 / (1.0 + np.exp(-(logits1 @ self.W6.T + self.b6)))  # [N,n23] L6 sigmoid gain
        h_mod = h * gain
        logits = h_mod @ self.W2.T + self.b2  # [N,K]   L5 second pass
        return a4, h, h_mod, logits

    def train_step(self, X, y, lr, steps=1):
        """`steps` local-update passes over the whole minibatch (X, y). Uses the CALLER-
        supplied lr for every layer (overrides any constructor default). Always trains through
        the L6 second pass (matches the source's train_step, which hard-wires use_l6=True
        regardless of the flag used at predict time). Returns the final mean cross-entropy
        loss as a python float."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n = X.shape[0]
        onehot = np.eye(self.n_classes)[y]
        lr = float(lr)
        last_loss = None
        for _ in range(int(steps)):
            a4, h, h_mod, logits = self._forward_batch(X, use_l6=True)
            p = _softmax_rows(logits)
            delta = p - onehot  # [N,K] softmax + cross-entropy output error

            # L5 output: LOCAL delta rule (presynaptic = the actual second-pass input h_mod).
            self.W2 -= lr * (delta.T @ h_mod) / n
            self.b2 -= lr * delta.mean(axis=0)

            # L2/3 hidden: FEEDBACK ALIGNMENT credit through the FIXED random B (no weight
            # transport), gating a local three-factor update. phip is the tanh surrogate slope.
            e_hidden = delta @ self.B.T  # [N,n23]
            phip = 1.0 - h * h
            mod = e_hidden * phip
            dW1 = -lr * (mod.T @ a4) / n
            db1 = -lr * mod.mean(axis=0)
            dW1[self._frozen] = 0.0  # freeze: read-only units do not move
            db1[self._frozen] = 0.0
            self.W1 += dW1
            self.b1 += db1

            eps = 1e-12
            last_loss = float(-np.mean(np.sum(onehot * np.log(p + eps), axis=1)))
        return last_loss

    def predict(self, X):
        """Return raw LOGITS [N, n_classes] (no softmax, no argmax -- caller does argmax).
        Respects self.use_l6 (unlike train_step, which always trains through L6)."""
        X = np.asarray(X, dtype=float)
        _, _, _, logits = self._forward_batch(X, use_l6=self.use_l6)
        return logits

    def grow(self, k=1, freeze_old=False):
        """NEUROGENESIS-FREEZE: freeze_old=True marks every EXISTING L2/3 unit read-only
        (frozen bit-exact) before the new units are appended, so a new context can only
        recruit the fresh units. k fresh PLASTIC units are then appended with a ZERO L5
        fan-in column -- function-preserving: the column's output at the instant of growth
        is unchanged, because the new units contribute h_new * 0 == 0 to every logit. Their
        B / W6 feedback rows are fresh random. Returns the new n_hidden."""
        rng = np.random.default_rng(self._grow_seed)
        k = int(k)
        if freeze_old:
            self._frozen[:] = True
        new_W1 = rng.normal(0.0, 1.0 / math.sqrt(self.n4), size=(k, self.n4))
        self.W1 = np.vstack([self.W1, new_W1])
        self.b1 = np.concatenate([self.b1, np.zeros(k)])
        self._W1_init = np.vstack([self._W1_init, new_W1.copy()])
        self.W2 = np.hstack([self.W2, np.zeros((self.n_classes, k))])  # zero fan-in
        self.B = np.vstack([self.B, rng.normal(0.0, 1.0, size=(k, self.n_classes))])
        self.W6 = np.vstack([self.W6, rng.normal(0.0, 1.0, size=(k, self.n_classes))])
        self.b6 = np.concatenate([self.b6, np.zeros(k)])
        self._frozen = np.concatenate([self._frozen, np.zeros(k, dtype=bool)])
        return self.n_hidden

    # --- live-tick single-sample methods for a running brain ------------------------------
    def observe(self, x, y):
        """One local update on a single sample x ([in_dim]) with integer label y. Same rule as
        train_step (respects the freeze mask), single-sample batch, using the instance's
        default learning rate self._lr."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        self.train_step(x, [int(y)], lr=self._lr, steps=1)

    def infer(self, x):
        """No-learning, deterministic single-sample inference. Returns
        (int argmax class, float max-softmax confidence in [0,1])."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        p = _softmax_rows(self.predict(x))[0]
        cls = int(np.argmax(p))
        conf = float(np.max(p))
        return cls, conf


# =====================================================================================================
# Self-check: real N-MNIST features if available, else the source's deterministic K-blob task.
# =====================================================================================================
def _repo_root():
    import pathlib

    # this file: <root>/src/snn_data_lake/reasoning/brain_adapters/laminar_local_readout.py
    return pathlib.Path(__file__).resolve().parents[4]


def _load_nmnist_split(seed=7, test_frac=0.2):
    """Load models/nmnist_pack.npz if present, rate-code (sum spikes over time) + flatten, and
    split deterministically. Returns (Xtr, ytr, Xte, yte, K, in_dim, desc) or None if the file
    is missing or fails to load cheaply."""
    npz_path = _repo_root() / "models" / "nmnist_pack.npz"
    if not npz_path.exists():
        return None
    try:
        d = np.load(npz_path)
        frames = d["frames"]  # [N, T, 2, 34, 34] float32 spike frames (on/off polarity)
        labels = d["labels"].astype(int)
    except Exception:
        return None
    x = frames.sum(axis=1).reshape(frames.shape[0], -1).astype(float)  # rate code, flatten
    y = labels
    k = int(y.max()) + 1
    n = len(y)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_test = max(k, int(round(test_frac * n)))
    te_idx, tr_idx = idx[:n_test], idx[n_test:]
    desc = (
        f"real cached feature pack models/nmnist_pack.npz (N-MNIST rate-coded spike counts, "
        f"N={n}, K={k}, in_dim={x.shape[1]})"
    )
    return x[tr_idx], y[tr_idx], x[te_idx], y[te_idx], k, x.shape[1], desc


def _make_blob_dataset(seed=7, dim=12, k=6, n_train=35, n_test=15, n_sub=2, sep=3.0, noise=1.0):
    """A deterministic K-blob task: K classes in R^D, each class = union of n_sub Gaussian sub-blobs
    at random centers -> NOT linearly separable. Balanced. Self-contained and torch-free."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, sep, size=(k, n_sub, dim))
    xtr, ytr, xte, yte = [], [], [], []
    for c in range(k):
        for _ in range(n_train):
            s = int(rng.integers(0, n_sub))
            xtr.append(centers[c, s] + rng.normal(0.0, noise, size=dim))
            ytr.append(c)
        for _ in range(n_test):
            s = int(rng.integers(0, n_sub))
            xte.append(centers[c, s] + rng.normal(0.0, noise, size=dim))
            yte.append(c)
    desc = f"synthetic K-blob task (K={k} classes in R^{dim}, 2 Gaussian sub-blobs each, NOT linearly separable)"
    return (
        np.asarray(xtr),
        np.asarray(ytr),
        np.asarray(xte),
        np.asarray(yte),
        k,
        dim,
        desc,
    )


def _standardize(train, test):
    """Standardize on TRAIN statistics only (no test leakage), applied to both."""
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (test - mean) / std


def _run_once(seed=7, n_epochs=300, lr=0.05, n23_init=64, n_grow=16):
    """One full deterministic train+grow+train+eval run. Returns a dict of everything the
    self-check asserts on, so main() can call this TWICE for the determinism check."""
    loaded = _load_nmnist_split(seed=seed)
    if loaded is not None:
        xtr_raw, ytr, xte_raw, yte, k, in_dim, desc = loaded
    else:
        xtr_raw, ytr, xte_raw, yte, k, in_dim, desc = _make_blob_dataset(seed=seed)
    xtr, xte = _standardize(xtr_raw, xte_raw)

    net = LaminarLocalReadout(in_dim, n_classes=k, n23=n23_init, seed=seed)
    w1_birth = net._W1_init.copy()

    losses = []
    for _ in range(n_epochs):
        losses.append(net.train_step(xtr, ytr, lr=lr, steps=1))

    weights_moved = float(np.max(np.abs(net.W1 - w1_birth)))

    # --- grow: function-preserving check (argmax unchanged at the instant of growth) -------
    probe = xte[: min(8, len(xte))]
    logits_before = net.predict(probe)
    pred_before = np.argmax(logits_before, axis=1)
    n_old = net.n_hidden
    net.grow(n_grow, freeze_old=True)
    logits_after = net.predict(probe)
    pred_after = np.argmax(logits_after, axis=1)
    frozen_snapshot = net.W1[:n_old].copy()  # value of the just-frozen rows, right after grow

    # --- continue training post-grow, then verify the frozen rows never moved --------------
    for _ in range(n_epochs // 2):
        net.train_step(xtr, ytr, lr=lr, steps=1)
    freeze_bit_exact = bool(np.array_equal(net.W1[:n_old], frozen_snapshot))

    acc = float(np.mean(np.argmax(net.predict(xte), axis=1) == yte))
    chance = 1.0 / k

    return {
        "desc": desc,
        "k": k,
        "in_dim": in_dim,
        "n_train": len(ytr),
        "n_test": len(yte),
        "losses": losses,
        "weights_moved": weights_moved,
        "grow_preserving": bool(np.array_equal(pred_before, pred_after)),
        "freeze_bit_exact": freeze_bit_exact,
        "accuracy": acc,
        "chance": chance,
    }


def main():
    assert "torch" not in sys.modules, "laminar_local_readout must stay a torch-free module"

    print("=" * 88)
    print("LAMINAR LOCAL READOUT -- no-backprop drop-in for LaminarReadout (self-check)")
    print("=" * 88)

    r1 = _run_once(seed=7)
    print(f"dataset: {r1['desc']}")
    print(
        f"  train N={r1['n_train']}  held-out N={r1['n_test']}  K={r1['k']}  in_dim={r1['in_dim']}"
    )
    print(f"  loss[0]={r1['losses'][0]:.4f}  loss[-1]={r1['losses'][-1]:.4f}")
    print(f"  weights moved (max abs W1 - W1_init) = {r1['weights_moved']:.6f}")
    print(
        f"  grow function-preserving (argmax unchanged at instant of growth) = {r1['grow_preserving']}"
    )
    print(
        f"  freeze bit-exact (frozen rows unchanged after post-grow training) = {r1['freeze_bit_exact']}"
    )
    print(f"  held-out accuracy = {r1['accuracy']:.4f}   chance (1/K) = {r1['chance']:.4f}")

    # (a) no torch anywhere in this process as a result of this module.
    assert "torch" not in sys.modules, "torch leaked into sys.modules"
    # (b) weights move.
    assert r1["weights_moved"] > 0.0, "L2/3 weights did not move -- local rule is not learning"
    # (c) freeze is bit-exact.
    assert r1["freeze_bit_exact"], "frozen L2/3 rows drifted after a post-grow training run"
    # (d) grow is function-preserving.
    assert r1["grow_preserving"], "predict() argmax changed at the instant of growth"
    # (e) beats chance.
    assert r1["accuracy"] > r1["chance"], (
        f"held-out accuracy {r1['accuracy']:.4f} did not beat chance {r1['chance']:.4f}"
    )
    # gradient-free sanity check: the batched local rule actually descends the loss.
    assert r1["losses"][-1] < r1["losses"][0], "mean cross-entropy loss did not decrease"

    # (f) deterministic: a second fresh run with the same seed reproduces the SAME accuracy.
    r2 = _run_once(seed=7)
    assert r1["accuracy"] == r2["accuracy"], (
        f"non-deterministic: run1 acc={r1['accuracy']:.6f} vs run2 acc={r2['accuracy']:.6f}"
    )

    print("-" * 88)
    print(
        f"laminar_local_readout OK: no-backprop drop-in verified -- held-out acc={r1['accuracy']:.4f} "
        f"vs chance={r1['chance']:.4f} on {'N-MNIST' if 'N-MNIST' in r1['desc'] else 'synthetic K-blob'} "
        f"data; weights moved, freeze bit-exact, grow function-preserving, deterministic across runs."
    )


if __name__ == "__main__":
    main()
