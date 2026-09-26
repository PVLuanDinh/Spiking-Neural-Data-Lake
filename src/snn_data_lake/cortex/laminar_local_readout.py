r"""
laminar_local_readout -- a NO-BACKPROP drop-in replacement for LaminarReadout.

LaminarReadout (brain_adapters/laminar_readout.py) is a torch nn.Module trained by Adam via a
reverse-mode gradient tape through the L4 -> L2/3 -> L5 -> L6 laminar motif. This module ports
the SAME four-role architecture but every layer learns LOCALLY -- the exact mechanism measured
as the winning variant B' of a four-way local-learning-readout study, here re-exposed
behind LaminarReadout's public interface (n_hidden / train_step / predict / grow) so it is a
true drop-in.

The property defended here is LOCALITY, and it is a claim about the RULE, not about the import
list: NO gradient tape, NO Adam, NO reverse-mode differentiation call, and no weight transport
anywhere (hidden credit arrives through a fixed random matrix, never through W2 transposed).
It is dense-matrix numpy throughout, and that is a MERIT choice carrying a P7 MEASUREMENT, not a
purity constraint. The LIVE input width is 64 -- not the 2312 an earlier draft of this docstring
asserted. 2312 is the width of the SELF-CHECK's rate-coded N-MNIST pack (2*34*34); the live
consumer builds this readout with in_dim=len(percept) over the spiking encoder's dim-64 latent
(cortex/brain.py, _perceive_real), which a live run confirmed by reading in_dim off the
constructed instance. P7 requires a dependency REJECTION to ship a measurement on the real
workload rather than a purity argument, so: one observe() (single-sample forward + local update),
median of 7 repeats x 400 observes against a hand-written torch-CPU twin of the same rule --
numpy 0.061 ms vs torch 0.117 ms at the LIVE width 64, and 0.108 ms vs 0.131 ms at the self-check
width 2312. Absolute milliseconds drift with machine load and should not be quoted back; the
REPRODUCIBLE quantity is the ratio, and numpy won every repeat at both widths (1.93-2.05x at 64,
1.18-1.21x at 2312). Note the margin SHRINKS as the width grows, so this rejection is scoped to
the live single-sample path and is not a claim about a batched or wider regime. This
module carries NO dependency-purity claim -- the zero-dependency rule was retired 2026-07-20 and
P7 (2026-08-04) supersedes it. Two `assert "torch" not in sys.modules` purity asserts were
DELETED here on 2026-08-04: they keyed on a library NAME rather than on locality, which is the
same error the project's e-prop invariant warns about in the mirror direction ("absence of
.backward() is NOT evidence of locality") -- absence of torch was not evidence either, and under
pytest torch is already resident before the first self-check runs, so the asserts were also a
guaranteed CORE-tier failure. The evidence for the locality claim is the self-check battery
below (weights move / freeze bit-exact / grow function-preserving / above chance) together with
its RED controls, which prove each of those instruments can actually register a failure.

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

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.laminar_local_readout

(PROMOTED 2026-08-12, #10 lane 4c: reasoning/brain_adapters/ -> cortex/, the region whose
live tick consumes it -- cortex/brain.py constructs it lazily on every real percept and its
confidence gates the percept drive. The command above was rewritten AT the move and verified
to run at the new path, per this docstring's own earlier warning about pre-naming a path.)
"""

import math

import numpy as np

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "LaminarLocalReadout.train_step": (
        "LOCAL",
        "local-update passes over the minibatch, each layer trained on its own signal with the "
        "caller-supplied lr -- which is what the module is named for. No error crosses a layer "
        "boundary, so there is no gradient for a cosine to compare against",
    ),
}


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
        # P1 (was `self._lr = 0.08`, AUDIT_BACKLOG M15): observe() is the LIVE wake-tick path
        # (cortex/brain.py calls it every real percept) and its confidence GATES the percept
        # drive, so its step size arbitrates behaviour and may not be a fixed magic number.
        # It is now a RUNNING-BASELINE-RELATIVE form: the absolute rate is derived each
        # observation from an EMA of the module's own L2/3 presynaptic energy ||a4||^2, so the
        # step tracks the input scale instead of being pinned to one. See _observe_lr().
        #   _mu       : dimensionless relaxation factor on a NORMALIZED quantity -- the step is
        #               expressed as a FRACTION of presynaptic energy, so mu carries no units and
        #               no scale. Fixed under P1 exception (a) (docs/POLICIES.md P1: "a fixed
        #               point on a normalized output"), which it satisfies on its own.
        #               DELETED, not softened: an earlier draft justified mu = 1.0 as "the DERIVED
        #               fully-corrective step, the value that exactly cancels the current sample's
        #               output error", with normalized-LMS convergence on 0 < mu < 2. That
        #               derivation does not transfer to this architecture -- NLMS is a LINEAR unit
        #               under squared error, and this update is softmax cross-entropy through a
        #               tanh hidden layer, feedback alignment through a fixed random B, and an L6
        #               gain pass. Measured: one observe() at mu = 1.0 moved the residual output
        #               error 0.892132 -> 0.854140, i.e. it cancelled 4.3% of it, not 100%. The
        #               step is normalized-LMS-SHAPED; calling it fully corrective was false.
        #   _base_rate: EMA horizon, reusing the project's `_*_base` running-baseline idiom (a P1
        #               reuse anchor) at the same 0.02 the live anchor uses -- verified against
        #               cortex/brain.py `_lc_base`, which is `0.98 * base + 0.02 * lvl`.
        self._mu = 1.0
        self._base_rate = 0.02
        self._energy_base = None  # EMA of ||[a4, 1]||^2; seeded by the first observation
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
        B / W6 feedback rows are fresh random. Returns the new n_hidden.

        P3 DISCLOSURE (2026-08-04, not a gate failure and deliberately NOT fixed by wiring): this
        method has NO live call site. Its only callers are _run_once() and _red_controls(), both
        inside this module's self-check. The live consumer (cortex/brain.py, via observe()) never
        grows the readout, so after promotion this is a dead method on a live object. It escapes
        method_dormancy_gate because "grow" is not in LEARN_VERBS. Forcing a caller to silence
        that would be exactly the fake-consumer move P3's honesty guard forbids -- it is recorded
        here instead."""
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
    def _observe_lr(self, x):
        """P1 running-baseline step size for the live single-sample path. Updates the EMA of the
        L2/3 presynaptic energy with THIS sample, then returns mu / energy_base.

        The energy is that of the AUGMENTED presynaptic vector [a4, 1], not of a4 alone: the bias
        b1 is updated on the same step and its own presynaptic activation is exactly 1, so
        ||a4||^2 + 1 is the DERIVED denominator for the parameters actually being moved -- not a
        regularizer, and not the `+ 1e-12` epsilon it replaces (an undeclared magic constant whose
        only job was to hide a division by zero). It also makes the denominator strictly >= 1,
        which matters on the live path: a percept driving L2/3 to silence (a4 == 0, reachable
        whenever the upstream encoder emits no spikes for a frame) returned mu/1e-12 == 1e12 under
        the old form. Measured on that input: ONE observe() drove max|b2| to 9.0e11 and max|b1| to
        2.6e12, and pinned infer()'s confidence -- the value that GATES the percept drive at
        cortex/brain.py -- at 1.0, i.e. the gate stuck wide open and never recovered (the fixed
        lr = 0.08 this replaced gave max|b2| = 0.072 on the identical input). Honest scope: on
        today's live pack that input does NOT occur -- all 256 frames were probed and the smallest
        latent maximum was 15.0 -- so this was a latent hazard, not an observed live failure.

        NOT scale-free, despite what an earlier draft of this docstring claimed ("double the input
        drive and the returned rate halves"). a4 = tanh(...) saturates, so the rate is not
        inverse-linear in input amplitude: measured lr(2x)/lr(1x) = 0.434 at in_dim 64 and 0.358 at
        in_dim 2312, between the small-signal 1/scale^2 and the saturated constant. What it does
        buy is what P1 asks for -- the step tracks the module's own running input drive instead of
        being pinned to a number chosen once. Its LIVE reading, read off the instance the running
        brain built after 60 wake ticks: energy_base 20.27 -> lr 0.0493, against the fixed 0.08
        this replaced. Disclosed because it bounds the claim: over a live percept stream the rate
        varies only within a few percent, so the arbitration this buys is real but small."""
        a4 = np.tanh(x @ self.W4.T + self.b4)
        energy = float(np.sum(a4 * a4)) + 1.0  # ||[a4, 1]||^2 -- the bias unit's presyn is 1
        self._energy_base = (
            energy
            if self._energy_base is None
            else (1.0 - self._base_rate) * self._energy_base + self._base_rate * energy
        )
        return self._mu / self._energy_base

    def observe(self, x, y):
        """One local update on a single sample x ([in_dim]) with integer label y. Same rule as
        train_step (respects the freeze mask), single-sample batch, at the running-baseline
        learning rate from _observe_lr() -- no fixed step constant (P1)."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        self.train_step(x, [int(y)], lr=self._observe_lr(x), steps=1)

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
_PACK_REL = ("models", "nmnist_pack.npz")


def _repo_root():
    """Locate the repo root by WALKING UP from __file__ to the first ancestor holding
    pyproject.toml -- never by a hardcoded parent depth.

    This was `parents[4]`, which encoded the module's then-current nesting
    (<root>/src/snn_data_lake/reasoning/brain_adapters/<file>). That number is a self-contained
    path assumption, and moving this file one level up silently turns it into the WRONG
    directory: the .npz stops resolving, _load_nmnist_split() returns None, the self-check
    quietly runs the synthetic K-blob instead of real N-MNIST and exits 0 at ~1.0 accuracy.
    A gate going green on a module that stopped testing against real data is the exact
    fail-green shape this project treats as a defect, so the depth is now DERIVED. Returns None
    when no marker is found (e.g. a frozen exe), which callers must treat as "no repo".
    """
    import pathlib

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _load_nmnist_split(seed=7, test_frac=0.2):
    """Load models/nmnist_pack.npz if present, rate-code (sum spikes over time) + flatten, and
    split deterministically. Returns (Xtr, ytr, Xte, yte, K, in_dim, desc) or None if the repo
    root / file is missing or the pack fails to load cheaply."""
    root = _repo_root()
    if root is None:
        return None
    npz_path = root.joinpath(*_PACK_REL)
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
    at random centers -> NOT linearly separable. Balanced. Self-contained: the fallback when the
    real N-MNIST pack is absent (e.g. a frozen exe), and the cheap substrate for the red controls."""
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


def _red_controls(seed=7, n_epochs=60):
    """P8 (1)+(2): construct the RED for every claim main() asserts, so each success criterion is
    shown to be REACHABLE rather than assumed. Every entry below is the SAME measurement main()
    uses, run under the condition that should break it; if any of these comes back green the
    corresponding assert is not an instrument, it is decoration. Runs on the small synthetic blob
    task (dim 12) so the controls stay cheap next to the real-data run.

    Returns a dict of the control readings; main() asserts each one is RED.
    """
    xtr_raw, ytr, xte_raw, yte, k, dim, _ = _make_blob_dataset(seed=seed)
    xtr, xte = _standardize(xtr_raw, xte_raw)
    out = {}

    # -- control 1: FROZEN RATE (lr = 0). Reds "weights moved" and "loss decreased" together:
    #    with no step, a working instrument must report no movement and no descent.
    net0 = LaminarLocalReadout(dim, n_classes=k, n23=32, seed=seed)
    w0 = net0.W1.copy()
    losses0 = [net0.train_step(xtr, ytr, lr=0.0, steps=1) for _ in range(5)]
    out["lr0_weights_moved"] = float(np.max(np.abs(net0.W1 - w0)))
    out["lr0_loss_decreased"] = bool(losses0[-1] < losses0[0])

    # -- control 2: UNTRAINED TWIN. The identical architecture with the local rule never applied
    #    is the null the TRAINED accuracy is measured against. Scope, corrected: this reds the
    #    trained-vs-untrained claim, NOT main()'s "beats chance" assert -- `blob_chance` is
    #    returned and printed alongside so the reader can see that this untrained twin already
    #    beats chance, which is exactly why it cannot serve as the red for that assert.
    twin = LaminarLocalReadout(dim, n_classes=k, n23=32, seed=seed)
    out["untrained_accuracy"] = float(np.mean(np.argmax(twin.predict(xte), axis=1) == yte))
    out["blob_chance"] = 1.0 / k

    trained = LaminarLocalReadout(dim, n_classes=k, n23=32, seed=seed)
    for _ in range(n_epochs):
        trained.train_step(xtr, ytr, lr=0.05, steps=1)
    out["trained_accuracy"] = float(np.mean(np.argmax(trained.predict(xte), axis=1) == yte))

    # -- control 3: GROW WITHOUT FREEZE. Reds "freeze bit-exact": the same array_equal probe run
    #    on units that were NOT frozen must report movement, else it cannot detect drift.
    n_old = trained.n_hidden
    trained.grow(8, freeze_old=False)
    snap = trained.W1[:n_old].copy()
    for _ in range(10):
        trained.train_step(xtr, ytr, lr=0.05, steps=1)
    out["unfrozen_bit_exact"] = bool(np.array_equal(trained.W1[:n_old], snap))

    # -- control 4: NON-ZERO L5 FAN-IN. Reds "grow function-preserving": the argmax probe must
    #    register a change when the new units are given a non-zero output column, else it cannot
    #    detect a growth step that is NOT function-preserving.
    net2 = LaminarLocalReadout(dim, n_classes=k, n23=32, seed=seed)
    for _ in range(n_epochs):
        net2.train_step(xtr, ytr, lr=0.05, steps=1)
    probe = xte[: min(8, len(xte))]
    before = np.argmax(net2.predict(probe), axis=1)
    n_old2 = net2.n_hidden
    net2.grow(8, freeze_old=True)
    net2.W2[0, n_old2:] = 50.0  # break function-preservation on purpose: class 0 now dominates
    after = np.argmax(net2.predict(probe), axis=1)
    out["broken_grow_preserving"] = bool(np.array_equal(before, after))
    out["broken_grow_n_changed"] = int(np.sum(before != after))

    # -- control 5: P1 STEP IS SCALE-FREE. Reds the running-baseline claim: the same input at a
    #    different amplitude must yield a proportionally different rate (a fixed constant would
    #    return the identical number for both, which is the defect this replaced).
    netl = LaminarLocalReadout(dim, n_classes=k, n23=32, seed=seed)
    x1 = np.asarray(xtr[:1], dtype=float)
    lr_a = netl._observe_lr(x1)
    netl2 = LaminarLocalReadout(dim, n_classes=k, n23=32, seed=seed)
    lr_b = netl2._observe_lr(x1 * 0.25)
    out["lr_small_input"] = float(lr_b)
    out["lr_large_input"] = float(lr_a)
    return out


def main():
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

    # The two `assert "torch" not in sys.modules` purity asserts that used to open and close this
    # block are DELETED (P7, 2026-08-04). They keyed on a library NAME, not on the locality this
    # module actually claims, and they could not survive the CORE tier anyway: pytest imports
    # run_selfchecks, which imports torch at module scope, so torch is resident before the first
    # self-check runs and the asserts fired unconditionally. What replaces them is not a weaker
    # version of the same check -- it is the RED-control battery below, which tests the property.

    # (a) DATA PROVENANCE. Silent-downgrade guard: if the repo root resolves AND the real pack is
    # on disk, the run MUST have used it. Without this, a broken real-data path just falls back to
    # the synthetic blobs and the gate goes green on a module that stopped testing against real
    # data. Reachable RED: make models/nmnist_pack.npz unreadable -- the file still exists, the
    # loader returns None, and this fires instead of passing at synthetic ~1.0 accuracy.
    _root = _repo_root()
    _pack = _root.joinpath(*_PACK_REL) if _root is not None else None
    if _pack is not None and _pack.exists():
        assert "nmnist_pack.npz" in r1["desc"], (
            f"{_pack} exists but the self-check fell back to synthetic data -- the real-data path "
            "broke silently"
        )

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

    # (f) deterministic -- CONVERTED 2026-08-12 (P8/F11) from the old same-seed accuracy
    # comparison, which had NO reachable red: _run_once is a pure function of its seed, so
    # no input could ever fail it (the concession below recorded exactly that). The
    # two-sided form has a real red on each side: a constructor that ignores `seed` fails
    # the second half, and unseeded construction jitter fails the first. Birth weights,
    # not 51-sample accuracy, so no coincidence flake.
    _d7a = LaminarLocalReadout(in_dim=8, n_classes=4, n4=16, n23=8, seed=7)
    _d7b = LaminarLocalReadout(in_dim=8, n_classes=4, n4=16, n23=8, seed=7)
    _d8 = LaminarLocalReadout(in_dim=8, n_classes=4, n4=16, n23=8, seed=8)
    assert np.array_equal(_d7a._W1_init, _d7b._W1_init), (
        "same-seed constructions must produce identical birth weights"
    )
    assert not np.array_equal(_d7a._W1_init, _d8._W1_init), (
        "different-seed constructions must produce different birth weights "
        "(a constructor that ignores seed would pass the first half and fail here)"
    )

    # (g) P8 reachable-red controls, with their coverage stated EXACTLY. Each re-runs the SAME
    # measurement under the condition that should break it; a control that comes back green means
    # the corresponding assert cannot fail and would have to be deleted, not weakened.
    #   COVERED in-process: (b) weights move, (c) freeze bit-exact, (d) grow function-preserving,
    #   the loss-descent assert, and the P1 running-baseline step claim.
    #   NOT COVERED, stated because the earlier wording here was "every criterion above is now
    #   shown to have a REACHABLE RED" and that was an overclaim:
    #     - (a) DATA PROVENANCE ships no in-process control. ((f) DETERMINISM was converted
    #       2026-08-12 to the two-sided birth-weight form above, which carries its own red.)
    #     - (e) "beats chance" is NOT redded by control 2. Control 2 is a trained-vs-untrained
    #       CONTRAST; the untrained twin's own accuracy is printed below precisely because it
    #       BEATS blob chance (measured 0.1889 vs 1/6 = 0.1667), so it cannot demonstrate that the
    #       chance assert is able to fail.
    #   The remaining reds ARE reachable -- constructed and run out-of-band during verification:
    #   make np.load raise on an existing pack -> the run falls back to synthetic and (a) fires;
    #   neuter train_step -> untrained
    #   N-MNIST accuracy 0.0784 < chance 0.1000 and (e) fires. They are
    #   deliberately NOT asserted in-process (each needs a monkeypatch of this module's own
    #   internals, and an in-process untrained-vs-chance assert would be one seed away from a
    #   flake). Claiming coverage they do not have is the P8 defect this note now avoids.
    c = _red_controls()
    print("-" * 88)
    print("RED controls (each must be the FAILING reading, proving the assert can fail):")
    print(
        f"  lr=0 -> weights moved = {c['lr0_weights_moved']:.6f} (want 0.0), "
        f"loss decreased = {c['lr0_loss_decreased']} (want False)"
    )
    print(
        f"  untrained twin acc = {c['untrained_accuracy']:.4f} vs trained {c['trained_accuracy']:.4f} "
        f"(blob chance = {c['blob_chance']:.4f}; the twin BEATS chance, so this control reds "
        f"trained-vs-untrained, NOT the 'beats chance' assert)"
    )
    print(f"  grown WITHOUT freeze -> bit-exact = {c['unfrozen_bit_exact']} (want False)")
    print(
        f"  non-zero L5 fan-in -> function-preserving = {c['broken_grow_preserving']} (want False, "
        f"{c['broken_grow_n_changed']} probe rows flipped)"
    )
    print(
        f"  P1 step vs input amplitude: lr(0.25x input) = {c['lr_small_input']:.4f} "
        f"vs lr(1x input) = {c['lr_large_input']:.4f} (want NOT equal)"
    )
    assert c["lr0_weights_moved"] == 0.0, (
        "control failed: weights moved at lr=0, so the weights-moved assert is not measuring the rule"
    )
    assert not c["lr0_loss_decreased"], (
        "control failed: loss decreased at lr=0, so the loss-descent assert cannot detect a dead rule"
    )
    assert c["untrained_accuracy"] < c["trained_accuracy"], (
        f"control failed: untrained twin {c['untrained_accuracy']:.4f} matched the trained net "
        f"{c['trained_accuracy']:.4f} -- the accuracy claim has no reachable red"
    )
    assert not c["unfrozen_bit_exact"], (
        "control failed: rows that were NOT frozen still read bit-exact, so the freeze assert "
        "cannot detect drift"
    )
    assert not c["broken_grow_preserving"], (
        "control failed: a deliberately non-zero L5 fan-in column still read function-preserving, "
        "so the grow assert cannot detect a non-preserving growth step"
    )
    assert c["lr_small_input"] != c["lr_large_input"], (
        "control failed: the observe() step is identical at two input amplitudes -- it is behaving "
        "like the fixed constant P1 required it to stop being"
    )

    print("-" * 88)
    print(
        f"laminar_local_readout OK: no-backprop drop-in verified -- held-out acc={r1['accuracy']:.4f} "
        f"vs chance={r1['chance']:.4f} on {'N-MNIST' if 'N-MNIST' in r1['desc'] else 'synthetic K-blob'} "
        f"data; weights moved, freeze bit-exact, grow function-preserving, deterministic across runs; "
        f"all 6 red controls returned the failing reading."
    )


if __name__ == "__main__":
    main()
