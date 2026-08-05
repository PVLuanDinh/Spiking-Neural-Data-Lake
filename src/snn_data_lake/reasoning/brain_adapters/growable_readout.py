r"""
growable_readout -- re-add NEUROGENESIS to the developmental loop's READOUT.

Honest framing (read this first). The loop's classifier is a plain
`nn.Linear(feat_dim, 10)` on frozen unsupervised conv1 features (93.5%). The v3.54
swap DROPPED the loop's original neurogenesis -- "the brain grows its own neurons."
This adapter re-adds it WITHOUT degrading accuracy: a GROWABLE-width MLP readout
whose hidden capacity is GROWN as new digits arrive, using the brain's real
FUNCTION-PRESERVING growth principle (Net2Net), so a growth event never disrupts the
learned function -- accuracy is bit-continuous across the grow, then training
specialises the new capacity.

The growth principle is REUSED (not reimplemented) from the shipped core module
`hippocampus/function_preserving_growth.py` (`SpikingGrowthLayer.duplicate_unit`):
to widen a layer, DUPLICATE a hidden unit's INCOMING weights and HALVE both copies'
OUTGOING weights, so the two half-weight fan-outs recombine to the parent's original
fan-out and the layer's output is UNCHANGED at the instant of growth:

    out_parent = w_out * h_p
    after split (h_new == h_p): (w_out/2) * h_p + (w_out/2) * h_new == w_out * h_p

That is the whole guarantee: the readout GROWS mid-training with ~zero functional
disruption. Here the layer is a torch nn.Module (backprop-trained, so it stays
accurate) and the units are ReLU, which makes the split identity hold: an exact copy
of the incoming weights gives an identical pre-activation, ReLU is deterministic, so
h_new == h_p and the halved fan-outs sum back to the original.

ONE honest correction vs a naive halving split. A backprop MLP with ReLU has the
classic Net2Net symmetry trap: the EQUAL (0.5, 0.5) fan-out halve preserves the output
ONLY if the two copies keep IDENTICAL incoming weights -- but two units with identical
incoming receive identical gradients and NEVER differentiate under Adam, so the "added"
capacity is a PHANTOM (the pair acts as one unit and loss never drops below the
plateau). The function-preserving split that actually adds usable capacity is the
(1, 0) split: the parent KEEPS its full fan-out and the duplicate starts at ZERO
fan-out. That still preserves the output exactly (the duplicate contributes 0 * h_new
== 0), but now the duplicate's incoming is free to be a genuinely DISTINCT feature
(seeded from the parent's incoming plus a full-scale perturbation), which training
recruits by growing its fan-out up from zero. This is precisely the zero-fan-out,
function-preserving growth the core module also ships as `add_unit_gradmax`. So the
guarantee here is EXACT to floating point (< 1e-6 in practice, asserted < 1e-4) AND the
capacity is real -- both, honestly, rather than a halve that looks tidy but adds nothing.

Torch/GPU tier. Built on `resolve_device()` (`snn_data_lake.substrate.device`) with an
explicit `.to(device)` on every parameter -- it does NOT rely on set_default_device
(that quirk lands `torch.normal` / freshly-wrapped `nn.Parameter` on CPU even under a
cuda default). Deterministic from PER-INSTANCE generators alone. ASCII-only prints.

P4c (2026-08-04). This header used to read "CORE is untouched: this file lives under
research/, imports nothing from src/, and edits neither run_selfchecks nor pyproject."
All three clauses were false and the framing was the defect P4c names: the file lives at
`src/snn_data_lake/reasoning/brain_adapters/`, it imports `snn_data_lake.substrate.device`,
and "it is only a research helper" is not a disposition. It is part of the brain; its open
debt is that it is registered in neither `REGIONS` nor `run_selfchecks.py` (see the P3/P4c
note on `LaminarReadout` -- both readouts carry the same debt, and P2 owes a decision
between this backprop readout and `laminar_local_readout.LaminarLocalReadout`).

Determinism (2026-08-04): `torch.manual_seed(seed)` used to run inside `__init__`, which
re-seeded the PROCESS-GLOBAL torch RNG for every module constructed afterwards --
`run_selfchecks.py` runs part of the suite in ONE process, so a constructor was silently
reaching outside itself. It was also redundant: every random draw in this class already
passes `generator=self._init_gen` / `self._grow_gen`. Removed; the self-check output is
byte-identical before and after (verified by A/B on the full main()).

  PYTHONPATH=src python -m snn_data_lake.reasoning.brain_adapters.growable_readout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.substrate.device import resolve_device


class GrowableReadout(nn.Module):
    """An MLP readout `in_dim -> hidden -> n_classes` (one ReLU hidden layer) whose
    hidden width can be GROWN function-preservingly (Net2Net split). Backprop-trained
    with Adam so it stays accurate; grow() adds capacity without disrupting the learned
    function. Parameters are raw tensors (not nn.Linear) so growth is a simple
    cat/slice on the weight matrices and the Adam moment buffers can be extended in
    lockstep."""

    def __init__(self, in_dim, n_classes=10, hidden=64, seed=0):
        super().__init__()
        # NO torch.manual_seed() here: a constructor may not re-seed the process-global RNG
        # (run_selfchecks runs part of the suite in one process). Every draw below is
        # per-instance via generator=.
        self.in_dim = in_dim
        self.n_classes = n_classes
        self._seed = seed
        # per-instance CPU generators -> deterministic init AND deterministic
        # symmetry-breaking noise, independent of the global RNG.
        self._init_gen = torch.Generator().manual_seed(seed)
        self._grow_gen = torch.Generator().manual_seed(seed + 0x5EED)

        dev = resolve_device()
        self.device = torch.device(dev)

        # Kaiming-ish init, built on CPU via an explicit generator then moved to device.
        # We do NOT use set_default_device: torch.normal(generator=...) and nn.Parameter
        # wrapping can silently stay on CPU under a cuda default, so .to(device) is explicit.
        std1 = (2.0 / in_dim) ** 0.5
        std2 = (2.0 / hidden) ** 0.5
        w1 = torch.normal(0.0, std1, (hidden, in_dim), generator=self._init_gen).to(self.device)
        b1 = torch.zeros(hidden).to(self.device)
        w2 = torch.normal(0.0, std2, (n_classes, hidden), generator=self._init_gen).to(self.device)
        b2 = torch.zeros(n_classes).to(self.device)

        self.W1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(b1)
        self.W2 = nn.Parameter(w2)
        self.b2 = nn.Parameter(b2)

        self._lr = 1e-2
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr)

        # NEUROGENESIS-WITH-FREEZE state (parameter isolation for continual learning). One
        # bool per hidden unit; True == this unit's INCOMING weights (its W1 row + b1 entry)
        # are a FIXED feature detector -- their gradient is masked to zero every train_step so
        # a new context cannot overwrite them. All units start plastic (False).
        self._frozen = torch.zeros(self.n_hidden, dtype=torch.bool, device=self.device)
        # OPTIONAL output-pathway isolation (PROPER parameter isolation). One bool per OUTPUT
        # class; True == this class's OUTPUT row (W2 row + b2 entry) is locked. Freezing the
        # old features alone is NOT enough to stop forgetting, because the shared softmax head
        # keeps being retrained on the new task (recency bias) -- see the self-check's 3-way
        # comparison. Locking the OLD classes' output rows too (while the NEW classes' rows
        # stay trainable, so the new class is still learned) is what makes retention decisive.
        # Empty by default -> behavior is exactly the hidden-only freeze unless opted into.
        self._frozen_out = torch.zeros(self.n_classes, dtype=torch.bool, device=self.device)

    @property
    def n_hidden(self):
        return self.W1.shape[0]

    def forward(self, X):
        h = F.relu(X @ self.W1.t() + self.b1)
        return h @ self.W2.t() + self.b2

    def predict(self, X):
        """Return logits [N, n_classes] (no grad)."""
        X = X.to(self.device)
        with torch.no_grad():
            return self.forward(X)

    def train_step(self, X, y, lr, steps=1):
        """steps Adam cross-entropy step(s) on a feature batch. Returns the last loss
        (float). `lr` sets the Adam learning rate for this call (and is remembered so a
        subsequent grow() rebuilds the optimizer at the same rate)."""
        X = X.to(self.device)
        y = y.to(self.device)
        self._lr = lr
        for g in self.opt.param_groups:
            g["lr"] = lr
        last = None
        for _ in range(steps):
            self.opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.forward(X), y)
            loss.backward()
            # NEUROGENESIS-WITH-FREEZE: frozen hidden units are FIXED feature detectors --
            # zero the gradient of their INCOMING weights (W1 rows + b1 entries) BEFORE
            # opt.step() so those weights cannot change. New (unfrozen) hidden units and the
            # OUTPUT layer (W2, b2) are never masked, so they update normally: the output can
            # still route the new units to the new class and, with replay, keep old-class
            # boundaries. Gradient masking on the single Parameter (no split); guard grad None.
            if bool(self._frozen.any()):
                if self.W1.grad is not None:
                    self.W1.grad[self._frozen] = 0.0
                if self.b1.grad is not None:
                    self.b1.grad[self._frozen] = 0.0
            # PROPER parameter isolation (opt-in via grow(freeze_out_classes=...)): also lock
            # the OLD classes' OUTPUT pathway (W2 rows + b2 entries) so Task B cannot rewrite
            # the old readout. NEW classes' rows are never masked, so the new class is still
            # learned. This is the completion that makes retention decisive -- freezing only
            # the hidden features leaves the shared softmax head to forget via recency bias.
            if bool(self._frozen_out.any()):
                if self.W2.grad is not None:
                    self.W2.grad[self._frozen_out] = 0.0
                if self.b2.grad is not None:
                    self.b2.grad[self._frozen_out] = 0.0
            self.opt.step()
            last = float(loss.detach())
        return last

    def grow(self, k=1, freeze_old=False, freeze_out_classes=None):
        """Add k hidden units FUNCTION-PRESERVINGLY (Net2Net split). Returns the new
        hidden width. Each unit is grown by duplicating an existing unit's INCOMING
        weights (plus a perturbation, so it is a distinct feature) and splitting the
        parent's OUTGOING fan-out as (parent-keeps-all, duplicate-starts-at-zero) so the
        two fan-outs SUM to the parent's original -> the network output is unchanged at
        the instant of growth. Zero duplicate fan-out (rather than an equal halve) is what
        lets the new capacity be real instead of a phantom -- see the module docstring.
        The Adam moment buffers are extended in lockstep so training resumes with the
        existing units' momentum intact and each new unit's at zero.

        NEUROGENESIS-WITH-FREEZE. When `freeze_old=True`, mark EVERY currently-existing
        hidden unit as frozen BEFORE the k new units are added, so a new context can ONLY
        recruit the fresh (unfrozen) units and physically cannot overwrite the old feature
        detectors (parameter isolation; the dynamic-architecture fix for catastrophic
        forgetting, e.g. progressive nets / PackNet). The k new units are added exactly as
        before (function-preserving (1,0) split) and are born PLASTIC. `freeze_old=False`
        (default) is the original all-plastic behavior. `_frozen` grows by k False entries.

        `freeze_out_classes` (optional list of class indices) additionally locks those
        classes' OUTPUT rows (W2 + b2) -> PROPER parameter isolation. Pass the OLD task's
        class ids to preserve its readout EXACTLY while the new classes' rows stay trainable.
        Default None keeps the output layer fully trainable (the plain hidden-only freeze)."""
        if freeze_old:
            # Lock all pre-existing units now, before growth appends the new plastic ones.
            self._frozen[:] = True
        if freeze_out_classes is not None:
            # Lock the OLD classes' output rows so Task B cannot rewrite their readout.
            for c in freeze_out_classes:
                self._frozen_out[int(c)] = True
        for _ in range(int(k)):
            self._grow_one()
        # Residual Adam momentum would drift a frozen row even under a zero gradient (m, v
        # decay geometrically but are non-zero after Task-A training), so zero those moments
        # once. Combined with the per-step gradient mask (grad stays 0 -> m, v stay 0), the
        # frozen incoming weights then stay EXACTLY constant. See _zero_frozen_moments.
        self._zero_frozen_moments()
        return self.n_hidden

    def _grow_one(self):
        # --- snapshot the current Adam state BEFORE we replace the Parameter objects ---
        old_state = {}
        for name in ("W1", "b1", "W2", "b2"):
            prm = getattr(self, name)
            st = self.opt.state.get(prm, None)
            if st:
                old_state[name] = {
                    "step": st["step"].clone() if torch.is_tensor(st["step"]) else st["step"],
                    "exp_avg": st["exp_avg"].detach().clone(),
                    "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
                }

        with torch.no_grad():
            # Pick the parent: the unit carrying the most OUTGOING weight (its fan-out
            # L2 norm) -- the most-used feature, a good seed for a new one.
            p = int(torch.argmax(torch.linalg.vector_norm(self.W2.data, dim=0)))

            # OUTGOING SPLIT. Net2Net widening splits the parent's fan-out between the two
            # copies so they SUM to the parent's original fan-out -> the output is
            # preserved. There is a whole simplex of preserving splits; which point we pick
            # is NOT free, because of a ReLU+Adam trap:
            #
            #   * the EQUAL (0.5, 0.5) halve preserves output only if the copies have
            #     IDENTICAL incoming (so h_new == h_p and the two halves recombine to
            #     w_out * h_p). But two units with identical incoming get identical
            #     gradients and NEVER differentiate under Adam -- the "added" capacity is a
            #     phantom (the pair acts as one unit; loss will not drop below the plateau).
            #   * the (1, 0) split -- parent KEEPS its full fan-out, the duplicate starts
            #     at ZERO fan-out -- also preserves output (the duplicate contributes 0 *
            #     anything == 0), but now the duplicate's incoming is FREE: it can be a
            #     genuinely distinct feature, because its (currently zero) fan-out is what
            #     zeros its output contribution, NOT an identical activation.
            #
            # We use the (1, 0) split (this is exactly the zero-fan-out, function-preserving
            # growth the core module ships as `SpikingGrowthLayer.add_unit_gradmax`). The
            # duplicate SEEDS from the parent's incoming (a real duplication) plus a full-
            # scale perturbation, so it is a distinct feature from birth; training then
            # recruits it by growing its fan-out from zero. This is the honest fix for the
            # phantom trap: preservation is EXACT (< 1e-6) AND the capacity is real.
            w1_p = self.W1.data[p].clone()
            std1 = (2.0 / self.in_dim) ** 0.5
            noise = (
                torch.normal(0.0, 1.0, (self.in_dim,), generator=self._grow_gen).to(self.device)
                * std1
            )
            new_row = (w1_p + noise).unsqueeze(0)  # duplicate parent's incoming + fresh feature
            new_W1 = torch.cat([self.W1.data, new_row], dim=0)
            new_b1 = torch.cat([self.b1.data, self.b1.data[p : p + 1].clone()], dim=0)

            # OUTGOING: parent's column UNCHANGED, duplicate's column ZERO -> the (1, 0)
            # split. Sum of the two fan-outs == the parent's original fan-out, so the
            # downstream logit is preserved exactly (the duplicate adds 0 * h_new == 0).
            zero_col = torch.zeros(self.n_classes, 1, device=self.W2.device)
            new_W2 = torch.cat([self.W2.data, zero_col], dim=1)
            new_b2 = self.b2.data.clone()  # output biases unchanged

        # Re-register the widened parameters (nn.Module.__setattr__ handles the swap).
        self.W1 = nn.Parameter(new_W1)
        self.b1 = nn.Parameter(new_b1)
        self.W2 = nn.Parameter(new_W2)
        self.b2 = nn.Parameter(new_b2)

        # Rebuild Adam over the new params and restore the extended moment buffers.
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr)
        self._restore_adam(old_state)

        # The new unit is born PLASTIC (trainable). grow(freeze_old=True) has already frozen
        # the pre-existing units before this call, so appending False keeps the new one free.
        self._frozen = torch.cat(
            [self._frozen, torch.zeros(1, dtype=torch.bool, device=self._frozen.device)]
        )

    def _restore_adam(self, old_state):
        """Re-seed the fresh optimizer's state so persistent units keep their Adam
        moments and each new unit's row/column starts at zero momentum. The overlapping
        top-left block of exp_avg / exp_avg_sq is copied; padded entries stay zero.
        (W1 gains a row, W2 gains a column, b1 gains an entry, b2 is unchanged.)"""
        if not old_state:
            return
        for name in ("W1", "b1", "W2", "b2"):
            if name not in old_state:
                continue
            prm = getattr(self, name)
            src = old_state[name]
            exp_avg = torch.zeros_like(prm.data)
            exp_avg_sq = torch.zeros_like(prm.data)
            block = tuple(slice(0, s) for s in src["exp_avg"].shape)
            exp_avg[block] = src["exp_avg"].to(exp_avg.device)
            exp_avg_sq[block] = src["exp_avg_sq"].to(exp_avg_sq.device)
            self.opt.state[prm] = {
                "step": src["step"],
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }

    def _zero_frozen_moments(self):
        """Zero the Adam first/second-moment buffers (exp_avg, exp_avg_sq) for frozen rows --
        the incoming weights of frozen hidden units (W1 rows + b1) and, when output isolation
        is on, the frozen classes' output rows (W2 rows + b2). A frozen row's gradient is
        masked to zero every step, so once its moments are zero they stay zero (m = b1*0 +
        (1-b1)*0 = 0, likewise v) and Adam's update lr * m_hat / (sqrt(v_hat) + eps) = 0 ->
        the row is bit-for-bit unchanged. WITHOUT this, momentum accumulated during Task-A
        training keeps nudging the frozen rows during Task-B training (momentum drift), and
        the "exactly unchanged" guarantee fails. This is why the moments must be zeroed too,
        not just the gradients."""
        if bool(self._frozen.any()):
            idx = self._frozen
            for prm in (self.W1, self.b1):
                st = self.opt.state.get(prm, None)
                if st and "exp_avg" in st:
                    st["exp_avg"][idx] = 0.0
                    st["exp_avg_sq"][idx] = 0.0
        if bool(self._frozen_out.any()):
            odx = self._frozen_out
            for prm in (self.W2, self.b2):
                st = self.opt.state.get(prm, None)
                if st and "exp_avg" in st:
                    st["exp_avg"][odx] = 0.0
                    st["exp_avg_sq"][odx] = 0.0


# --------------------------------------------------------------------------------------
# Self-check: on small SYNTHETIC data, prove (1) grow() is function-preserving (logits
# unchanged < 1e-4), (2) the grown capacity is USABLE (loss drops far below a converged
# bottleneck baseline), (3) the readout beats chance on a toy task.
# --------------------------------------------------------------------------------------
def _make_bands(n=600, in_dim=8, n_bands=6, seed=0):
    """A 2-class band-parity task: one informative coordinate in [0, n_bands) whose label
    is floor(x0) % 2 (n_bands alternating stripes), plus label-irrelevant noise dims. A
    1-hidden-layer ReLU net with H units makes at most H kinks, so it needs H >= n_bands-1
    to separate all stripes. With n_bands=6 a hidden=4 readout UNDERFITS (a bottleneck) and
    MUST grow past ~5 units to solve it -- which is exactly what makes a capacity gain
    observable rather than a phantom."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(n, 1, generator=g) * n_bands  # [0, n_bands)
    y = (x0.squeeze(1).floor().long()) % 2
    noise = torch.normal(0.0, 1.0, (n, in_dim - 1), generator=g)
    X = torch.cat([x0, noise], 1)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-8)
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]


def _accuracy(model, X, y):
    pred = model.predict(X).argmax(dim=1).cpu()
    return float((pred == y).float().mean())


def _make_task_blobs(n_per=150, in_dim=16, n_classes=6, sep=1.6, seed=0):
    """6 OVERLAPPING, INTERLEAVED Gaussian blobs (classes 0..5). The class centers are placed
    along one informative axis in the INTERLEAVED order [0,3,1,4,2,5], so Task-A classes
    {0,1,2} (positions 0,2,4) and Task-B classes {3,4,5} (positions 1,3,5) SHARE the feature
    region -- each B class sits between two A classes. With overlap (noise std ~ sep) the
    hidden units are SHARED across tasks, so training Task B necessarily reshapes the decision
    boundaries Task A depends on. That interference is exactly the condition under which
    parameter isolation matters (well-separated blobs would not forget with or without freeze,
    so they cannot demonstrate the fix). Deterministic; CPU tensors (torch.normal(float,float,
    size) lands on CPU regardless of a cuda default) -- the model moves batches to its device."""
    order = [0, 3, 1, 4, 2, 5]  # position -> class label; A and B classes alternate
    pos_of_class = {c: i for i, c in enumerate(order)}
    g = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for c in range(n_classes):
        center = torch.zeros(in_dim)
        center[0] = pos_of_class[c] * sep  # informative axis
        center[1] = ((pos_of_class[c] % 2) * 2 - 1) * 0.5 * sep  # mild zig-zag on a 2nd axis
        Xs.append(center + torch.normal(0.0, 1.0, (n_per, in_dim), generator=g))
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    X = torch.cat(Xs, 0)
    y = torch.cat(ys, 0)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-8)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


def _accuracy_subset(model, X, y, classes):
    """Task-aware (Task-Incremental) accuracy: argmax over ONLY `classes`. Standard continual-
    learning protocol -- it isolates whether the task's own FEATURES/READOUT survived, without
    the shared-softmax recency bias between tasks (which no feature/output freeze can fix
    without a task id at test time; see the self-check's afterALL numbers)."""
    idx = torch.tensor(classes, device=model.device)
    logits = model.predict(X)[:, idx]
    pred = idx[logits.argmax(dim=1)].cpu()
    return float((pred == y).float().mean())


def _cf_run(mode, X, y, in_dim, n_classes, hidden0, grow_k, seed, replay_n=30):
    """One catastrophic-forgetting trial. Train Task A (classes {0,1,2}) to convergence, grow
    at the A->B boundary, then train Task B (classes {3,4,5}) WITH a small replay of A
    (mirroring the loop). `mode` selects the growth policy at the boundary:
      'nofreeze' -> grow(freeze_old=False)                          (all-plastic; the baseline)
      'hidden'   -> grow(freeze_old=True)                           (freeze old FEATURES only)
      'proper'   -> grow(freeze_old=True, freeze_out_classes=[0,1,2]) (features + old READOUT)
    Identical seed/data across the three modes -- only the policy differs. Returns a dict of
    task-aware/all-class accuracies and the frozen-vs-new weight diffs."""
    task_a = [0, 1, 2]
    a_mask = (y.unsqueeze(1) == torch.tensor(task_a)).any(1)
    Xa, ya = X[a_mask], y[a_mask]
    Xb, yb = X[~a_mask], y[~a_mask]

    # SMALL replay of A (deterministic subset), the same handful the developmental loop keeps.
    g = torch.Generator().manual_seed(seed + 7)
    rp = torch.randperm(Xa.shape[0], generator=g)[:replay_n]
    Xb_tr = torch.cat([Xb, Xa[rp]], 0)
    yb_tr = torch.cat([yb, ya[rp]], 0)

    model = GrowableReadout(in_dim, n_classes=n_classes, hidden=hidden0, seed=seed)
    for _ in range(800):
        model.train_step(Xa, ya, lr=5e-3, steps=1)
    before_ta = _accuracy_subset(model, Xa, ya, task_a)

    if mode == "nofreeze":
        model.grow(k=grow_k, freeze_old=False)
    elif mode == "hidden":
        model.grow(k=grow_k, freeze_old=True)
    elif mode == "proper":
        model.grow(k=grow_k, freeze_old=True, freeze_out_classes=task_a)
    else:
        raise ValueError(mode)
    W1_at_grow = model.W1.detach().clone()
    W2_at_grow = model.W2.detach().clone()
    frozen_idx = model._frozen.clone()

    for _ in range(1200):
        model.train_step(Xb_tr, yb_tr, lr=5e-3, steps=1)

    W1_now = model.W1.detach()
    if bool(frozen_idx.any()):
        w1_frozen_diff = float((W1_now[frozen_idx] - W1_at_grow[frozen_idx]).abs().max())
        w1_new_diff = float((W1_now[~frozen_idx] - W1_at_grow[~frozen_idx]).abs().max())
    else:
        w1_frozen_diff = float("nan")  # nothing frozen in the no-freeze baseline
        w1_new_diff = float((W1_now - W1_at_grow).abs().max())
    if bool(model._frozen_out.any()):
        odx = model._frozen_out
        w2_out_diff = float((model.W2.detach()[odx] - W2_at_grow[odx]).abs().max())
    else:
        w2_out_diff = float("nan")
    return {
        "before_ta": before_ta,
        "after_ta": _accuracy_subset(model, Xa, ya, task_a),
        "after_all": _accuracy(model, Xa, ya),
        "accB_ta": _accuracy_subset(model, Xb, yb, [3, 4, 5]),
        "w1_frozen_diff": w1_frozen_diff,
        "w1_new_diff": w1_new_diff,
        "w2_out_diff": w2_out_diff,
    }


def main():
    # No global torch.manual_seed(): every dataset builder and every model below draws from
    # its own torch.Generator, so the run is deterministic without touching the global RNG.
    in_dim, n_classes = 8, 2
    X, y = _make_bands(n=600, in_dim=in_dim, n_bands=6, seed=0)
    chance = 1.0 / n_classes

    print("=" * 70)
    print("GROWABLE READOUT -- function-preserving neurogenesis (Net2Net) -- research")
    print("=" * 70)
    print(
        f"task: 6-band parity  N={X.shape[0]}  in_dim={in_dim}  classes={n_classes}  "
        f"chance={chance:.3f}"
    )

    # Bottleneck baseline: 4 hidden units (< 5 kinks) CANNOT separate 6 stripes -> it
    # converges to a high-loss plateau. Any further loss drop after growth is therefore
    # attributable to added capacity, not to more training of the same net.
    model = GrowableReadout(in_dim, n_classes=n_classes, hidden=4, seed=0)
    loss_base = None
    for _ in range(1500):
        loss_base = model.train_step(X, y, lr=1e-2, steps=1)
    acc_base = _accuracy(model, X, y)
    print(
        f"baseline : hidden={model.n_hidden}  loss={loss_base:.4f}  acc={acc_base:.3f}  "
        f"(4-unit ReLU has too few kinks for 6 stripes -> plateau)"
    )

    # (1) FUNCTION-PRESERVING GROWTH: logits ~identical immediately after grow().
    logits_pre = model.predict(X).clone()
    new_width = model.grow(k=12)  # 4 -> 16 hidden units
    logits_post = model.predict(X)
    max_logit_diff = float((logits_pre - logits_post).abs().max())
    assert new_width == 16, f"grow must widen 4 -> 16, got {new_width}"
    assert model.n_hidden == 16, f"n_hidden must be 16 after grow, got {model.n_hidden}"
    assert max_logit_diff < 1e-4, (
        f"grow() must be function-preserving: max abs logit diff {max_logit_diff:.3e} >= 1e-4"
    )
    loss_at_grow = float(F.cross_entropy(logits_post, y.to(model.device)))
    print(
        f"(1) grow : hidden 4 -> {new_width}  max|dlogit|={max_logit_diff:.3e} (< 1e-4)  "
        f"loss_at_grow={loss_at_grow:.4f} (~loss_base, preserved)"
    )

    # (2) CAPACITY USABLE: train the grown net; loss must drop FAR below the converged
    # 1-unit plateau (only possible if the split units genuinely differentiated).
    loss_grown = None
    for _ in range(3000):
        loss_grown = model.train_step(X, y, lr=5e-3, steps=1)
    acc_grown = _accuracy(model, X, y)
    assert loss_grown < loss_base - 0.10, (
        f"grown capacity must reduce loss below the bottleneck plateau: "
        f"grown {loss_grown:.4f} vs base {loss_base:.4f}"
    )
    assert acc_grown > acc_base + 0.10, (
        f"grown net must classify better than the bottleneck: {acc_grown:.3f} vs {acc_base:.3f}"
    )
    print(
        f"(2) usable: after grow+train  loss={loss_grown:.4f} (< base-0.10)  "
        f"acc={acc_grown:.3f} (> base+0.10)  -> new capacity is real, not phantom"
    )

    # (3) BEATS CHANCE on the toy task.
    assert acc_grown > 0.85, f"grown readout must beat chance decisively: {acc_grown:.3f} <= 0.85"
    print(f"(3) chance: acc={acc_grown:.3f} >> chance {chance:.3f}")

    # ----------------------------------------------------------------------------------
    # (4) CATASTROPHIC FORGETTING: neurogenesis + parameter isolation vs all-plastic
    # neurogenesis, on a 2-task problem (Task A={0,1,2}, Task B={3,4,5}, interleaved so the
    # tasks SHARE the feature region and genuinely interfere). Identical seed/data for all
    # three runs; only the growth policy at the A->B boundary differs:
    #   nofreeze -> grow(freeze_old=False)                            (all-plastic baseline)
    #   hidden   -> grow(freeze_old=True)                             (freeze old FEATURES only)
    #   proper   -> grow(freeze_old=True, freeze_out_classes=[0,1,2]) (features + old READOUT)
    # Retention is scored TASK-AWARE (over the Task-A classes) -- the standard Task-IL metric.
    # HONEST finding baked into this test: freezing the FEATURES alone (hidden) recovers only
    # a few points, because the shared softmax head keeps getting retrained on Task B (recency
    # bias). PROPER parameter isolation -- also locking the old classes' output rows, while the
    # new classes' rows stay trainable so the new class is still learned -- is what retains
    # Task A decisively. (afterALL, the class-incremental accuracy with NO task id, stays low
    # for every mode: that confound is unsolvable by any freeze scheme without a task id, and
    # we report it rather than hide it.)
    # ----------------------------------------------------------------------------------
    cf_in, cf_nc, cf_h0, cf_k, cf_seed = 16, 6, 12, 12, 0
    Xcf, ycf = _make_task_blobs(n_per=150, in_dim=cf_in, n_classes=cf_nc, sep=2.0, seed=cf_seed)
    r_nf = _cf_run("nofreeze", Xcf, ycf, cf_in, cf_nc, cf_h0, cf_k, cf_seed)
    r_hd = _cf_run("hidden", Xcf, ycf, cf_in, cf_nc, cf_h0, cf_k, cf_seed)
    r_pr = _cf_run("proper", Xcf, ycf, cf_in, cf_nc, cf_h0, cf_k, cf_seed)
    delta_hidden = r_hd["after_ta"] - r_nf["after_ta"]
    delta_proper = r_pr["after_ta"] - r_nf["after_ta"]

    print("-" * 70)
    print("(4) catastrophic forgetting -- Task A={0,1,2} -> Task B={3,4,5}, small A-replay")
    print("    retention = task-aware acc on Task-A classes (Task-IL); acc_B confirms B learned")
    print(
        f"    nofreeze : acc_A {r_nf['before_ta']:.3f} -> {r_nf['after_ta']:.3f}   "
        f"(all-class acc_A={r_nf['after_all']:.3f}, acc_B={r_nf['accB_ta']:.3f})"
    )
    print(
        f"    hidden   : acc_A {r_hd['before_ta']:.3f} -> {r_hd['after_ta']:.3f}   "
        f"(all-class acc_A={r_hd['after_all']:.3f}, acc_B={r_hd['accB_ta']:.3f})  freeze features only"
    )
    print(
        f"    proper   : acc_A {r_pr['before_ta']:.3f} -> {r_pr['after_ta']:.3f}   "
        f"(all-class acc_A={r_pr['after_all']:.3f}, acc_B={r_pr['accB_ta']:.3f})  features + old readout"
    )
    print(
        f"    frozen W1 rows max|diff| over Task B = {r_pr['w1_frozen_diff']:.3e} (must be 0.0)  |  "
        f"new-unit rows max|diff| = {r_pr['w1_new_diff']:.3e} (> 0)"
    )
    print(
        f"    frozen old-class W2 rows max|diff| = {r_pr['w2_out_diff']:.3e} (must be 0.0, exact readout)"
    )
    print(
        f"    retention delta vs nofreeze: hidden-only={delta_hidden:+.3f} (partial)  "
        f"proper={delta_proper:+.3f} (must be > +0.15)"
    )
    assert r_pr["w1_frozen_diff"] == 0.0, (
        f"frozen units' W1 rows must be EXACTLY unchanged across Task B, got {r_pr['w1_frozen_diff']:.3e}"
    )
    assert r_pr["w1_new_diff"] > 0.0, (
        f"at least one NEW unit's W1 row must change during Task B, got {r_pr['w1_new_diff']:.3e}"
    )
    assert r_pr["w2_out_diff"] == 0.0, (
        f"frozen old-class W2 rows must be EXACTLY unchanged (exact readout), got {r_pr['w2_out_diff']:.3e}"
    )
    assert r_pr["accB_ta"] > 0.85, (
        f"the NEW task must still be learned under isolation, got acc_B={r_pr['accB_ta']:.3f}"
    )
    assert delta_proper > 0.15, (
        f"PROPER isolation must retain Task A markedly better than nofreeze: "
        f"proper {r_pr['after_ta']:.3f} vs nofreeze {r_nf['after_ta']:.3f} (delta {delta_proper:+.3f} <= 0.15)"
    )
    print(
        "    -> proper parameter isolation prevents Task B from overwriting Task A "
        "(features frozen exactly, old readout frozen exactly, new class still learned)"
    )

    # keep the summary line's variable names stable
    after_nf, after_fz, fdiff_fz = r_nf["after_ta"], r_pr["after_ta"], r_pr["w1_frozen_diff"]
    retention_delta = delta_proper

    print("-" * 70)
    print("HONEST NOTE: growth preserves output EXACTLY via a (1,0) fan-out split (parent")
    print("  keeps its fan-out, duplicate starts at zero). An EQUAL (0.5,0.5) halve would")
    print("  also preserve output but force IDENTICAL incoming -> identical gradients ->")
    print("  the duplicate never specialises under Adam (phantom capacity; loss would NOT")
    print("  drop below the plateau). Zero duplicate fan-out frees the duplicate's incoming")
    print("  to be a distinct feature that training recruits -- so the capacity is REAL.")
    print("=" * 70)
    print(
        f"self-check OK: function-preserving grow (max|dlogit|={max_logit_diff:.3e}), "
        f"capacity usable (loss {loss_base:.3f} -> {loss_grown:.3f}), "
        f"acc {acc_base:.3f} -> {acc_grown:.3f} (chance {chance:.3f})"
    )
    print(
        f"  + neurogenesis-with-FREEZE beats forgetting: Task-A retention "
        f"nofreeze {after_nf:.3f} vs freeze {after_fz:.3f} (delta {retention_delta:+.3f}), "
        f"frozen rows exactly fixed (max|diff|={fdiff_fz:.3e})"
    )


if __name__ == "__main__":
    main()
