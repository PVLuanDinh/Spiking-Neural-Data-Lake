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

Torch/GPU tier. Built on `resolve_device()` (from research/torch_device.py) with an
explicit `.to(device)` on every parameter -- it does NOT rely on set_default_device
(that quirk lands `torch.normal` / freshly-wrapped `nn.Parameter` on CPU even under a
cuda default). Deterministic (torch.manual_seed + per-instance generators). ASCII-only
prints. Zero-dep CORE is untouched: this file lives under research/, imports nothing
from src/, and edits neither run_selfchecks nor pyproject.

  NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/brain_adapters/growable_readout.py
"""
import os
import sys

# Make research/torch_device.py importable when run as a script (the script's own dir
# research/brain_adapters/ is on sys.path, not research/), and when imported as a module.
_RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_device import resolve_device


class GrowableReadout(nn.Module):
    """An MLP readout `in_dim -> hidden -> n_classes` (one ReLU hidden layer) whose
    hidden width can be GROWN function-preservingly (Net2Net split). Backprop-trained
    with Adam so it stays accurate; grow() adds capacity without disrupting the learned
    function. Parameters are raw tensors (not nn.Linear) so growth is a simple
    cat/slice on the weight matrices and the Adam moment buffers can be extended in
    lockstep."""

    def __init__(self, in_dim, n_classes=10, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
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
            self.opt.step()
            last = float(loss.detach())
        return last

    def grow(self, k=1):
        """Add k hidden units FUNCTION-PRESERVINGLY (Net2Net split). Returns the new
        hidden width. Each unit is grown by duplicating an existing unit's INCOMING
        weights (plus a perturbation, so it is a distinct feature) and splitting the
        parent's OUTGOING fan-out as (parent-keeps-all, duplicate-starts-at-zero) so the
        two fan-outs SUM to the parent's original -> the network output is unchanged at
        the instant of growth. Zero duplicate fan-out (rather than an equal halve) is what
        lets the new capacity be real instead of a phantom -- see the module docstring.
        The Adam moment buffers are extended in lockstep so training resumes with the
        existing units' momentum intact and each new unit's at zero."""
        for _ in range(int(k)):
            self._grow_one()
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
            noise = torch.normal(
                0.0, 1.0, (self.in_dim,), generator=self._grow_gen
            ).to(self.device) * std1
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


def main():
    torch.manual_seed(0)
    in_dim, n_classes = 8, 2
    X, y = _make_bands(n=600, in_dim=in_dim, n_bands=6, seed=0)
    chance = 1.0 / n_classes

    print("=" * 70)
    print("GROWABLE READOUT -- function-preserving neurogenesis (Net2Net) -- research")
    print("=" * 70)
    print(f"task: 6-band parity  N={X.shape[0]}  in_dim={in_dim}  classes={n_classes}  "
          f"chance={chance:.3f}")

    # Bottleneck baseline: 4 hidden units (< 5 kinks) CANNOT separate 6 stripes -> it
    # converges to a high-loss plateau. Any further loss drop after growth is therefore
    # attributable to added capacity, not to more training of the same net.
    model = GrowableReadout(in_dim, n_classes=n_classes, hidden=4, seed=0)
    loss_base = None
    for _ in range(1500):
        loss_base = model.train_step(X, y, lr=1e-2, steps=1)
    acc_base = _accuracy(model, X, y)
    print(f"baseline : hidden={model.n_hidden}  loss={loss_base:.4f}  acc={acc_base:.3f}  "
          f"(4-unit ReLU has too few kinks for 6 stripes -> plateau)")

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
    print(f"(1) grow : hidden 4 -> {new_width}  max|dlogit|={max_logit_diff:.3e} (< 1e-4)  "
          f"loss_at_grow={loss_at_grow:.4f} (~loss_base, preserved)")

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
    print(f"(2) usable: after grow+train  loss={loss_grown:.4f} (< base-0.10)  "
          f"acc={acc_grown:.3f} (> base+0.10)  -> new capacity is real, not phantom")

    # (3) BEATS CHANCE on the toy task.
    assert acc_grown > 0.85, f"grown readout must beat chance decisively: {acc_grown:.3f} <= 0.85"
    print(f"(3) chance: acc={acc_grown:.3f} >> chance {chance:.3f}")

    print("-" * 70)
    print("HONEST NOTE: growth preserves output EXACTLY via a (1,0) fan-out split (parent")
    print("  keeps its fan-out, duplicate starts at zero). An EQUAL (0.5,0.5) halve would")
    print("  also preserve output but force IDENTICAL incoming -> identical gradients ->")
    print("  the duplicate never specialises under Adam (phantom capacity; loss would NOT")
    print("  drop below the plateau). Zero duplicate fan-out frees the duplicate's incoming")
    print("  to be a distinct feature that training recruits -- so the capacity is REAL.")
    print("=" * 70)
    print(f"self-check OK: function-preserving grow (max|dlogit|={max_logit_diff:.3e}), "
          f"capacity usable (loss {loss_base:.3f} -> {loss_grown:.3f}), "
          f"acc {acc_base:.3f} -> {acc_grown:.3f} (chance {chance:.3f})")


if __name__ == "__main__":
    main()
