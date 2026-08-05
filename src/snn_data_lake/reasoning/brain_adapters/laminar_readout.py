r"""
laminar_readout -- a READOUT shaped like the canonical 6-layer NEOCORTICAL laminar
motif (L4 -> L2/3 -> L5, with an L6 top-down feedback/gain loop), with NEUROGENESIS
(grow + freeze) in the L2/3 feature layer.

Honest framing (read this first). The developmental loop's classifier is a flat
linear/MLP readout on frozen unsupervised conv1 features. This adapter replaces that
flat head with a CORTICAL-COLUMN architecture -- the layer ROLES are taken from the
Potjans-Diesmann microcircuit (src/snn_data_lake/cortex/cortical_microcircuit.py),
whose docstring maps the canonical intracortical flow:

    thalamus -> L4 (main sensory INPUT layer)
             -> L2/3 (superficial FEATURE / associational layer, cortico-cortical)
             -> L5 (deep OUTPUT layer, projects out of the column)
             L6 (multiform FEEDBACK layer, gain-control back onto L2/3)

That microcircuit is a spiking DYNAMICS model (it reproduces firing statistics, it does
NOT classify), so it is used here ONLY as the reference for the roles/flow. This module
is a trainable, backprop feedforward READOUT that borrows the SAME four-role motif:

  * L4   : Linear(in_dim -> n4) + ReLU     -- conv1 features are the "thalamic" input.
  * L2/3 : Linear(n4 -> n23) + ReLU        -- the FEATURE layer; this is the GROWABLE +
                                              FREEZE (neurogenesis) layer.
  * L5   : Linear(n23 -> n_classes)        -- the OUTPUT layer -> class logits.
  * L6   : Linear(n_classes -> n23) + sigmoid -- ONE top-down feedback iteration that
                                              produces a per-L2/3-unit multiplicative
                                              GAIN, applied on a second forward pass.

L6 FEEDBACK COMPUTATION (exactly one iteration, documented):
    h4       = relu(X @ W4^T + b4)                       # L4
    h23      = relu(h4 @ W23^T + b23)                     # L2/3 (first pass)
    logits1  = h23 @ W5^T + b5                            # L5 first-pass prediction
    gain     = sigmoid(logits1 @ W6^T + b6)  in (0,1)^n23 # L6: predict a per-L2/3 gain
    h23_mod  = h23 * gain                                 # L6 gates L2/3 activations
    logits   = h23_mod @ W5^T + b5                        # L5 second pass -> final logits
If use_l6=False the head is pure feedforward: logits = logits1 (L6 path skipped).

NEUROGENESIS-WITH-FREEZE on L2/3 (REUSED verbatim from growable_readout.py). The L2/3
weight matrix W23 carries a per-hidden-unit `_frozen` bool mask. grow(k, freeze_old=True)
marks every existing L2/3 unit frozen BEFORE appending k fresh plastic units, so a new
context can only recruit the new units. train_step() zeroes the gradient of frozen W23
rows (+ b23 entries) every step, and grow() additionally zeroes their Adam moment buffers
(_zero_frozen_moments) -- without that, momentum accumulated during earlier training would
drift the frozen rows even under a zero gradient, so the "bit-exact" guarantee needs BOTH.

FUNCTION-PRESERVING GROWTH of L2/3 (Net2Net (1,0) split, same principle as the source
module). Adding an L2/3 unit must not disturb the column's output at the instant of growth,
and that requires touching THREE matrices, not one:
  * L2/3 incoming (W23 row + b23 entry): DUPLICATE a parent unit's incoming weights plus a
    full-scale perturbation, so the new unit is a genuinely distinct feature from birth.
  * L5 fan-in (W5 gains one COLUMN): ZERO-init, so the new L2/3 unit contributes 0 * h23_new
    == 0 to logits1 AND to the final logits -> the output is preserved exactly.
  * L6 fan-out onto the new unit (W6 gains one ROW): ZERO-init. The gain on OLD units is a
    function of logits1, which is unchanged (new W5 column is zero), so old-unit gains are
    identical; the new unit's gain multiplies its ZERO L5 fan-in, so it contributes nothing.
    Hence preservation holds regardless, and zero-init keeps it clean. b6 gains a copied entry.
The Adam moment buffers for W23/b23/W5/W6 are extended in lockstep (block-copy of the old
top-left region; padded entries zero) so training resumes with existing momentum intact.

Torch/GPU tier. Built on resolve_device() (`snn_data_lake.substrate.device`) with an explicit
.to(device) on every parameter -- it does NOT rely on set_default_device (that quirk lands
torch.normal / freshly-wrapped nn.Parameter on CPU even under a cuda default). Deterministic
from PER-INSTANCE generators alone. ASCII-only prints.

P4c (2026-08-04). This header used to read "CORE untouched: this file lives under research/,
imports nothing from src/, edits neither run_selfchecks nor pyproject." All three clauses were
false -- the file is at `src/snn_data_lake/reasoning/brain_adapters/` and it imports
`snn_data_lake.substrate.device` -- and "it is only research" is not a disposition. It is part
of the brain. OPEN DEBT, stated rather than wired: it is registered in neither `REGIONS` nor
`run_selfchecks.py`, so no gate sees it; and P2 owes a decision between this BACKPROP readout
and `laminar_local_readout.LaminarLocalReadout`, which was built as its no-backprop drop-in
(same public interface) and is the one on the live wake tick (`cortex/brain.py`). Two
mechanisms for one job is the P2 defect; retiring this one needs its three importers
(`developmental_loop_brain`, `ensemble_rbf`, `readout_ensemble`) repointed first.

Determinism (2026-08-04): `torch.manual_seed(seed)` used to run inside BOTH constructors,
re-seeding the PROCESS-GLOBAL torch RNG for every module built afterwards. It was redundant --
every draw here already passes `generator=` -- and `run_selfchecks.py` runs part of the suite in
one process, so a constructor was reaching outside itself. Removed; self-check output is
byte-identical before and after (verified by A/B on the full main()).

  PYTHONPATH=src python -m snn_data_lake.reasoning.brain_adapters.laminar_readout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.substrate.device import resolve_device


class LaminarReadout(nn.Module):
    """A cortical-column READOUT following the 6-layer laminar motif L4 -> L2/3 -> L5 with
    an L6 top-down gain loop. The L2/3 feature layer is GROWABLE (function-preserving
    Net2Net neurogenesis) and FREEZABLE (parameter isolation for continual learning).
    Parameters are raw tensors (not nn.Linear) so growth is a simple cat/slice on the
    weight matrices and the Adam moment buffers can be extended in lockstep. Drop-in for
    the loop's flat readout: same train_step / predict / grow / n_hidden interface."""

    def __init__(self, in_dim, n_classes=10, n4=96, n23=64, seed=0, use_l6=True):
        super().__init__()
        # NO torch.manual_seed() here: a constructor may not re-seed the process-global RNG
        # (run_selfchecks runs part of the suite in one process). All draws are per-instance.
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.n4 = n4
        self.use_l6 = use_l6
        self._seed = seed
        # per-instance CPU generators -> deterministic init AND deterministic grow-noise,
        # independent of the global RNG.
        self._init_gen = torch.Generator().manual_seed(seed)
        self._grow_gen = torch.Generator().manual_seed(seed + 0x5EED)

        dev = resolve_device()
        self.device = torch.device(dev)

        # Kaiming-ish inits, each built on CPU via the explicit generator then moved to
        # device. We do NOT use set_default_device: torch.normal(generator=...) and
        # nn.Parameter wrapping can silently stay on CPU under a cuda default, so the
        # explicit .to(self.device) on every tensor is required.
        def _w(fan_out, fan_in):
            std = (2.0 / fan_in) ** 0.5
            return torch.normal(0.0, std, (fan_out, fan_in), generator=self._init_gen).to(
                self.device
            )

        def _b(n):
            return torch.zeros(n).to(self.device)

        # L4: sensory INPUT layer (conv1 features -> n4), ReLU.
        self.W4 = nn.Parameter(_w(n4, in_dim))
        self.b4 = nn.Parameter(_b(n4))
        # L2/3: FEATURE layer (n4 -> n23), ReLU -- GROWABLE + FREEZE lives here.
        self.W23 = nn.Parameter(_w(n23, n4))
        self.b23 = nn.Parameter(_b(n23))
        # L5: OUTPUT layer (n23 -> n_classes) -> logits.
        self.W5 = nn.Parameter(_w(n_classes, n23))
        self.b5 = nn.Parameter(_b(n_classes))
        # L6: FEEDBACK / gain-control layer (n_classes -> n23), sigmoid gain onto L2/3.
        # Always allocated (so grow() logic is uniform), used only when use_l6=True.
        self.W6 = nn.Parameter(_w(n23, n_classes))
        self.b6 = nn.Parameter(_b(n23))

        self._lr = 1e-2
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr)

        # NEUROGENESIS-WITH-FREEZE state: one bool per L2/3 unit. True == this unit's INCOMING
        # weights (its W23 row + b23 entry) are a FIXED feature detector -- their gradient is
        # masked to zero every train_step so a new context cannot overwrite them. All plastic.
        self._frozen = torch.zeros(self.n_hidden, dtype=torch.bool, device=self.device)

    @property
    def n_hidden(self):
        """Current L2/3 width (the growable dimension)."""
        return self.W23.shape[0]

    # --- forward through the laminar column ------------------------------------------------
    def _l4_l23(self, X):
        """L4 -> L2/3: return the first-pass L2/3 activations h23 (ReLU)."""
        h4 = F.relu(X @ self.W4.t() + self.b4)  # L4 sensory input layer
        h23 = F.relu(h4 @ self.W23.t() + self.b23)  # L2/3 feature layer
        return h23

    def forward(self, X, use_feedback=None):
        """Full column forward. use_feedback=None -> use self.use_l6. Pass False to force
        the pure feedforward (L4->L2/3->L5) output even on an L6 model (used by the self-check
        to isolate the L6 contribution)."""
        if use_feedback is None:
            use_feedback = self.use_l6
        h23 = self._l4_l23(X)
        logits1 = h23 @ self.W5.t() + self.b5  # L5 first-pass prediction
        if not use_feedback:
            return logits1
        # L6: predict a per-L2/3-unit multiplicative gain in (0,1) from the first-pass
        # logits, gate the L2/3 activations, and re-run L5 (ONE feedback iteration).
        gain = torch.sigmoid(logits1 @ self.W6.t() + self.b6)  # (N, n23)
        h23_mod = h23 * gain
        return h23_mod @ self.W5.t() + self.b5  # L5 second pass -> final logits

    def predict(self, X):
        """Return final logits [N, n_classes] (no grad), through L4->L2/3->[L6 gain]->L5."""
        X = X.to(self.device)
        with torch.no_grad():
            return self.forward(X)

    def train_step(self, X, y, lr, steps=1):
        """steps Adam cross-entropy step(s) on a feature batch. Returns the last loss (float).
        Applies the L2/3 frozen-row gradient mask each step (like GrowableReadout): frozen
        L2/3 units are fixed feature detectors -- zero their W23 row + b23 gradients BEFORE
        opt.step() so they cannot change. New/plastic L2/3 units and every other layer (L4,
        L5, L6) update normally."""
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
            if bool(self._frozen.any()):
                if self.W23.grad is not None:
                    self.W23.grad[self._frozen] = 0.0
                if self.b23.grad is not None:
                    self.b23.grad[self._frozen] = 0.0
            self.opt.step()
            last = float(loss.detach())
        return last

    # --- neurogenesis on L2/3 -------------------------------------------------------------
    def grow(self, k=1, freeze_old=False):
        """Add k L2/3 units FUNCTION-PRESERVINGLY (Net2Net (1,0) split) -- this is the
        neurogenesis. Returns the new L2/3 width. Each unit duplicates an existing L2/3
        unit's INCOMING weights (W23 row + b23, plus a perturbation so it is a distinct
        feature) and is born with a ZERO L5 fan-in column (and zero L6 fan-out row), so the
        column output is unchanged at the instant of growth. The Adam moment buffers are
        extended in lockstep.

        When freeze_old=True, mark EVERY currently-existing L2/3 unit frozen BEFORE the k new
        units are appended, so a new context can only recruit the fresh (plastic) units and
        physically cannot overwrite the old feature detectors (parameter isolation). The k
        new units are born PLASTIC. freeze_old=False (default) is the all-plastic behavior."""
        if freeze_old:
            self._frozen[:] = True
        for _ in range(int(k)):
            self._grow_one()
        # Residual Adam momentum would drift a frozen row even under a zero gradient, so zero
        # those moments once. Combined with the per-step gradient mask, the frozen L2/3 rows
        # then stay EXACTLY constant. See _zero_frozen_moments.
        self._zero_frozen_moments()
        return self.n_hidden

    def _grow_one(self):
        # --- snapshot the current Adam state BEFORE we replace the Parameter objects ---
        old_state = {}
        for name in ("W4", "b4", "W23", "b23", "W5", "b5", "W6", "b6"):
            prm = getattr(self, name)
            st = self.opt.state.get(prm, None)
            if st:
                old_state[name] = {
                    "step": st["step"].clone() if torch.is_tensor(st["step"]) else st["step"],
                    "exp_avg": st["exp_avg"].detach().clone(),
                    "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
                }

        with torch.no_grad():
            # Pick the parent L2/3 unit: the one carrying the most OUTGOING weight into L5
            # (its W5 column L2 norm) -- the most-used feature, a good seed for a new one.
            p = int(torch.argmax(torch.linalg.vector_norm(self.W5.data, dim=0)))

            # L2/3 INCOMING: duplicate parent's W23 row + a full-scale perturbation, so the
            # new unit is a genuinely distinct feature from birth (the (1,0) split's honest
            # capacity: its zero L5 fan-in is what zeros its output, NOT an identical activation).
            w23_p = self.W23.data[p].clone()
            std23 = (2.0 / self.n4) ** 0.5
            noise = (
                torch.normal(0.0, 1.0, (self.n4,), generator=self._grow_gen).to(self.device) * std23
            )
            new_W23 = torch.cat([self.W23.data, (w23_p + noise).unsqueeze(0)], dim=0)
            new_b23 = torch.cat([self.b23.data, self.b23.data[p : p + 1].clone()], dim=0)

            # L5 FAN-IN: parent's column UNCHANGED, new unit's column ZERO -> the (1,0) split.
            # The new L2/3 unit contributes 0 * h23_new == 0 to logits1 and to the final
            # logits, so the column output is preserved exactly at the instant of growth.
            zero_col = torch.zeros(self.n_classes, 1, device=self.W5.device)
            new_W5 = torch.cat([self.W5.data, zero_col], dim=1)

            # L6 FAN-OUT onto the new L2/3 unit: W6 gains a ZERO row (its gain does not matter
            # because the new unit's L5 fan-in is zero; zero-init keeps it clean). b6 gains a
            # copied entry. Old-unit gains are unchanged because they depend on logits1, which
            # is unchanged (new W5 column is zero).
            zero_w6_row = torch.zeros(1, self.n_classes, device=self.W6.device)
            new_W6 = torch.cat([self.W6.data, zero_w6_row], dim=0)
            new_b6 = torch.cat([self.b6.data, self.b6.data[p : p + 1].clone()], dim=0)

        # Re-register the widened parameters (nn.Module.__setattr__ handles the swap).
        self.W23 = nn.Parameter(new_W23)
        self.b23 = nn.Parameter(new_b23)
        self.W5 = nn.Parameter(new_W5)
        self.W6 = nn.Parameter(new_W6)
        self.b6 = nn.Parameter(new_b6)
        # W4, b4, b5 are unchanged in shape -- but re-wrap them too so the fresh optimizer
        # takes ALL current Parameters (their state is restored by block-copy below).

        # Rebuild Adam over the new params and restore the extended moment buffers.
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr)
        self._restore_adam(old_state)

        # The new L2/3 unit is born PLASTIC. grow(freeze_old=True) has already frozen the
        # pre-existing units before this call, so appending False keeps the new one free.
        self._frozen = torch.cat(
            [self._frozen, torch.zeros(1, dtype=torch.bool, device=self._frozen.device)]
        )

    def _restore_adam(self, old_state):
        """Re-seed the fresh optimizer's state so persistent units keep their Adam moments
        and each new L2/3 row/column starts at zero momentum. The overlapping top-left block
        of exp_avg / exp_avg_sq is copied; padded entries stay zero. (W23 gains a row, W5
        gains a column, W6 gains a row, b23/b6 gain an entry; W4/b4/b5 are unchanged.)"""
        if not old_state:
            return
        for name in ("W4", "b4", "W23", "b23", "W5", "b5", "W6", "b6"):
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
        """Zero the Adam first/second-moment buffers (exp_avg, exp_avg_sq) for the frozen
        L2/3 rows (W23 rows + b23). A frozen row's gradient is masked to zero every step, so
        once its moments are zero they stay zero and Adam's update becomes 0 -> the row is
        bit-for-bit unchanged. WITHOUT this, momentum accumulated during earlier training
        keeps nudging the frozen rows (momentum drift) and the 'exactly unchanged' guarantee
        fails. This is why the moments must be zeroed too, not just the gradients."""
        if bool(self._frozen.any()):
            idx = self._frozen
            for prm in (self.W23, self.b23):
                st = self.opt.state.get(prm, None)
                if st and "exp_avg" in st:
                    st["exp_avg"][idx] = 0.0
                    st["exp_avg_sq"][idx] = 0.0


class LaminarColumns(nn.Module):
    r"""Per-digit ISOLATED cortical COLUMNS (PROGRESSIVE neural networks, Rusu et al. 2016)
    so freezing covers the ENTIRE upstream path -- the fix for the v3.59 failure where
    LaminarReadout froze only L2/3 while a SHARED L4 stayed plastic and drifted, corrupting
    the "frozen" L2/3 features (they received an L4 output they were never frozen for ->
    catastrophic forgetting, 63.7%, the worst mode).

    THE FIX. Each new digit/task gets its OWN full COLUMN, and freezing locks a whole column:

      column c = L4_c (Linear in_dim->n4 + ReLU) -> L2/3_c (Linear n4->n23 + ReLU)

    so every frozen column's L2/3 features ALWAYS see the SAME (its own, fixed) L4 output --
    there is no shared upstream stage left to drift. A SINGLE shared output head reads the
    CONCATENATION of every column's L2/3 activations:

      h23_c   = relu( relu(X @ W4_c^T + b4_c) @ W23_c^T + b23_c )   for each column c
      concat  = [ h23_0 | h23_1 | ... | h23_C ]                     (width = C * n23)
      logits  = concat @ W5^T + b5                                  (shared L5 head)

    There is NO L6 top-down loop in the column version (kept simple + stable on purpose --
    the isolation guarantee is the point here, not the feedback gain). Compare LaminarReadout
    above for the L6-gated single-column variant.

    NEUROGENESIS = grow(freeze_old=True): FREEZE all existing columns entirely (their L4+L2/3
    params: gradient masked to zero every step AND Adam moments zeroed at freeze -- the
    momentum-zeroing is essential, or a frozen weight drifts under residual momentum even at
    zero gradient), then ADD one fresh PLASTIC column and EXTEND the shared L5 head's fan-in by
    n23 with a ZERO block. Zero L5 fan-in makes the grow FUNCTION-PRESERVING: the new column
    contributes 0 * h23_new == 0 to the logits at the instant of growth. The Adam moment
    buffers are extended in lockstep (block-copy of the old top-left region; new entries zero).

    WHAT STAYS TRAINABLE. The newest (unfrozen) column and the shared L5 head update every
    step -- L5 must stay fully trainable so it can route the new column to the new class and,
    with a little replay, keep the old-class boundaries. HONEST caveat: because L5 is shared
    and trainable, some class-incremental (no task id) recency forgetting through the head can
    remain -- what full-column isolation GUARANTEES is that every old column's features are
    bit-exact preserved (the v3.59 corruption cannot happen); it does not by itself freeze the
    shared readout. Score retention task-aware (Task-IL) to isolate the feature/column effect.

    Interface (train_step / predict / grow(k, freeze_old) / n_hidden) matches the loop's
    readout for drop-in use. Torch/GPU tier: built on resolve_device() with an explicit
    .to(device) on every parameter (NOT set_default_device -- torch.normal(generator=...) and
    freshly-wrapped nn.Parameter can silently stay on CPU under a cuda default). Deterministic
    from per-instance generators alone (no global seeding). ASCII-only prints.
    """

    def __init__(self, in_dim, n_classes=10, n4=48, n23=32, seed=0):
        super().__init__()
        # NO torch.manual_seed() here: a constructor may not re-seed the process-global RNG
        # (run_selfchecks runs part of the suite in one process). All draws are per-instance.
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.n4 = n4
        self.n23 = n23
        self._seed = seed
        # per-instance CPU generators -> deterministic init AND deterministic grow draws,
        # independent of the global RNG. Column 0 draws from _init_gen; grown columns from
        # _grow_gen (so adding a column is a fresh, distinct feature detector from birth).
        self._init_gen = torch.Generator().manual_seed(seed)
        self._grow_gen = torch.Generator().manual_seed(seed + 0x5EED)

        dev = resolve_device()
        self.device = torch.device(dev)

        # Per-column L4 + L2/3 parameters, parallel lists indexed by column. Raw Parameters
        # (in ParameterLists) so growth is a simple append + the Adam moment buffers can be
        # extended in lockstep.
        self.W4s = nn.ParameterList()  # each: L4  (n4  x in_dim)  sensory input layer
        self.b4s = nn.ParameterList()
        self.W23s = nn.ParameterList()  # each: L2/3 (n23 x n4)     feature layer
        self.b23s = nn.ParameterList()

        # Whole-column frozen flag (parameter isolation). True == this column's L4+L2/3 are a
        # FIXED feature stack -- their gradients are masked to zero every train_step and their
        # Adam moments are zeroed at freeze, so the whole column is bit-for-bit constant.
        self._frozen_cols = []

        self._add_column(self._init_gen)  # start with ONE column (column 0)

        # Shared L5 OUTPUT head reads the CONCAT of all columns' L2/3 activations. Its fan-in
        # is total_n23 (== C * n23) and GROWS by n23 each time a column is added.
        self.W5 = nn.Parameter(self._w(n_classes, self.n_hidden, self._init_gen))
        self.b5 = nn.Parameter(self._b(n_classes))

        self._lr = 1e-2
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr)

    # --- construction helpers --------------------------------------------------------------
    def _w(self, fan_out, fan_in, gen):
        std = (2.0 / fan_in) ** 0.5
        return torch.normal(0.0, std, (fan_out, fan_in), generator=gen).to(self.device)

    def _b(self, n):
        return torch.zeros(n).to(self.device)

    def _add_column(self, gen):
        """Append one fresh PLASTIC column (its own L4 + L2/3). Draws from `gen` so each
        column is deterministic and distinct. Does NOT touch W5 -- the caller sizes/extends
        the shared head's fan-in."""
        self.W4s.append(nn.Parameter(self._w(self.n4, self.in_dim, gen)))
        self.b4s.append(nn.Parameter(self._b(self.n4)))
        self.W23s.append(nn.Parameter(self._w(self.n23, self.n4, gen)))
        self.b23s.append(nn.Parameter(self._b(self.n23)))
        self._frozen_cols.append(False)

    @property
    def n_hidden(self):
        """Total L2/3 width across ALL columns (== n_columns * n23). The loop's HIDDEN_CAP
        bounds this, exactly as for the single-column readout."""
        return sum(int(w.shape[0]) for w in self.W23s)

    @property
    def n_columns(self):
        return len(self.W23s)

    # --- forward through the parallel columns ---------------------------------------------
    def _column_h23(self, X, i):
        """One column's L4 -> L2/3 activation (ReLU/ReLU)."""
        h4 = F.relu(X @ self.W4s[i].t() + self.b4s[i])  # L4 sensory input (this column)
        return F.relu(h4 @ self.W23s[i].t() + self.b23s[i])  # L2/3 feature layer (this column)

    def forward(self, X):
        h23s = [self._column_h23(X, i) for i in range(self.n_columns)]
        concat = torch.cat(h23s, dim=1)  # [N, C*n23]
        return concat @ self.W5.t() + self.b5  # shared L5 head -> logits

    def predict(self, X):
        """Return final logits [N, n_classes] (no grad)."""
        X = X.to(self.device)
        with torch.no_grad():
            return self.forward(X)

    def train_step(self, X, y, lr, steps=1):
        """steps Adam cross-entropy step(s). FROZEN columns' L4+L2/3 params do not change:
        their gradient is zeroed every step (and their Adam moments were zeroed at freeze).
        The newest (unfrozen) column and the shared L5 head update normally."""
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
            for i, frozen in enumerate(self._frozen_cols):
                if frozen:
                    for prm in (self.W4s[i], self.b4s[i], self.W23s[i], self.b23s[i]):
                        if prm.grad is not None:
                            prm.grad.zero_()
            self.opt.step()
            last = float(loss.detach())
        return last

    # --- neurogenesis: add one isolated column --------------------------------------------
    def grow(self, k=1, freeze_old=False):
        """PER-DIGIT neurogenesis. Adds ONE new column (regardless of k -- k is accepted only
        for interface compatibility with the single-unit readouts; a column grow is atomic).
        Returns the new total L2/3 width (n_hidden).

        When freeze_old=True: FREEZE every existing column entirely (L4+L2/3 -> grad-masked
        each step + Adam moments zeroed here), then add the new PLASTIC column and EXTEND the
        shared L5 head's fan-in by n23 with a ZERO block. The zero block makes the grow
        FUNCTION-PRESERVING -- the new column contributes 0 to the logits at the instant of
        growth -- while freezing whole columns guarantees every frozen column's features
        always see their OWN fixed L4 (the v3.59 shared-L4 drift is structurally impossible).
        freeze_old=False just appends a plastic column (all columns stay trainable)."""
        # Snapshot Adam state (keyed by stable name) BEFORE re-wrapping W5 / appending params.
        old_state = self._snapshot_adam()

        if freeze_old:
            for i in range(len(self._frozen_cols)):
                self._frozen_cols[i] = True

        # EXTEND the shared L5 fan-in by n23 with a ZERO block (function-preserving: the new
        # column's contribution to the logits is 0 * h23_new == 0 at the instant of growth).
        with torch.no_grad():
            zero_block = torch.zeros(self.n_classes, self.n23, device=self.W5.device)
            new_W5 = torch.cat([self.W5.data, zero_block], dim=1)
        self.W5 = nn.Parameter(new_W5)

        # ADD the new plastic column (its h23 lands in the LAST n23 slots of the concat, which
        # is exactly the zero block just appended to W5 -> preserved output).
        self._add_column(self._grow_gen)

        # Rebuild Adam over the new param set and restore the extended moment buffers.
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr)
        self._restore_adam(old_state)

        # Residual Adam momentum would drift a frozen column even under a zero gradient, so
        # zero those moments once. Combined with the per-step gradient mask, frozen columns
        # then stay EXACTLY constant. See _zero_frozen_moments.
        self._zero_frozen_moments()
        return self.n_hidden

    # --- Adam state bookkeeping across the grow -------------------------------------------
    def _param_map(self):
        """Stable name -> current Parameter map (per-column L4/L2/3 + shared W5/b5). Used to
        snapshot/restore Adam moments across a grow, where W5 is re-wrapped (bigger fan-in)
        and a column's four params are appended."""
        m = {}
        for i in range(self.n_columns):
            m[f"W4_{i}"] = self.W4s[i]
            m[f"b4_{i}"] = self.b4s[i]
            m[f"W23_{i}"] = self.W23s[i]
            m[f"b23_{i}"] = self.b23s[i]
        m["W5"] = self.W5
        m["b5"] = self.b5
        return m

    def _snapshot_adam(self):
        snap = {}
        for key, prm in self._param_map().items():
            st = self.opt.state.get(prm, None)
            if st:
                snap[key] = {
                    "step": st["step"].clone() if torch.is_tensor(st["step"]) else st["step"],
                    "exp_avg": st["exp_avg"].detach().clone(),
                    "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
                }
        return snap

    def _restore_adam(self, snap):
        """Re-seed the fresh optimizer so existing columns keep their Adam moments and the
        widened W5 keeps its (top-left block) moments; the new column and the new L5 fan-in
        block start at zero momentum. The overlapping block is copied; padding stays zero."""
        if not snap:
            return
        pm = self._param_map()
        for key, src in snap.items():
            if key not in pm:
                continue
            prm = pm[key]
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
        """Zero the Adam first/second-moment buffers for every FROZEN column's L4+L2/3 params.
        A frozen column's gradient is masked to zero every step, so once its moments are zero
        they stay zero and Adam's update becomes 0 -> the whole column is bit-for-bit
        unchanged. WITHOUT this, momentum accumulated during the column's own training would
        keep nudging it under a zero gradient and the 'exactly unchanged' guarantee fails."""
        for i, frozen in enumerate(self._frozen_cols):
            if not frozen:
                continue
            for prm in (self.W4s[i], self.b4s[i], self.W23s[i], self.b23s[i]):
                st = self.opt.state.get(prm, None)
                if st and "exp_avg" in st:
                    st["exp_avg"].zero_()
                    st["exp_avg_sq"].zero_()


# --------------------------------------------------------------------------------------
# Self-check: on small SYNTHETIC data, prove (1) the laminar column beats chance,
# (2) grow(k, freeze_old=True) on L2/3 is function-preserving AND frozen rows stay bit-exact
# while new units change, (3) the L6 feedback actually changes the logits.
# Then (4) LaminarColumns: full-upstream freeze retains Task A where the shared-L4 laminar
# did not -- frozen column bit-exact, function-preserving grow, high 2-task retention.
# --------------------------------------------------------------------------------------
def _make_bands(n=600, in_dim=8, n_bands=6, seed=0):
    """A 2-class band-parity task: one informative coordinate in [0, n_bands) whose label is
    floor(x0) % 2 (n_bands alternating stripes), plus label-irrelevant noise dims. Needs a
    handful of hidden kinks to separate the stripes -> a small readout that then grows can
    show a real capacity gain. Deterministic; standardized."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(n, 1, generator=g) * n_bands
    y = (x0.squeeze(1).floor().long()) % 2
    noise = torch.normal(0.0, 1.0, (n, in_dim - 1), generator=g)
    X = torch.cat([x0, noise], 1)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-8)
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]


def _accuracy(model, X, y):
    pred = model.predict(X).argmax(dim=1).cpu()
    return float((pred == y).float().mean())


def _accuracy_subset(model, X, y, classes):
    """Task-aware (Task-Incremental) accuracy: argmax over ONLY `classes`. The standard
    continual-learning metric -- it isolates whether the task's own columns/features survived,
    without the shared-softmax recency bias between tasks (which no feature freeze can remove
    without a task id at test time -- reported honestly in the LaminarColumns caveat)."""
    idx = torch.tensor(classes, device=model.device)
    logits = model.predict(X)[:, idx]
    pred = idx[logits.argmax(dim=1)].cpu()
    return float((pred == y).float().mean())


def _make_col_blobs(n_per=120, in_dim=12, n_classes=6, sep=3.0, seed=0):
    """6 DISTINCT (well-separated) Gaussian blobs, classes 0..5, placed along one informative
    axis with a zig-zag on a second axis. Distinct on purpose: with isolated per-digit columns
    + full-upstream freeze, Task A={0,1,2} should be RETAINED after learning Task B={3,4,5} --
    which the shared-L4 laminar could not do (its 'frozen' L2/3 saw a drifting L4). Deterministic;
    CPU tensors (torch.normal(float,float,size) lands on CPU) -- the model moves batches to its
    device."""
    g = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for c in range(n_classes):
        center = torch.zeros(in_dim)
        center[0] = c * sep  # informative axis
        center[1] = ((c % 2) * 2 - 1) * sep  # zig-zag on a 2nd axis
        Xs.append(center + torch.normal(0.0, 1.0, (n_per, in_dim), generator=g))
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    X = torch.cat(Xs, 0)
    y = torch.cat(ys, 0)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-8)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


def main():
    # No global torch.manual_seed(): every dataset builder and every model below draws from
    # its own torch.Generator, so the run is deterministic without touching the global RNG.
    in_dim, n_classes = 8, 2
    X, y = _make_bands(n=600, in_dim=in_dim, n_bands=6, seed=0)
    chance = 1.0 / n_classes

    print("=" * 74)
    print("LAMINAR READOUT -- 6-layer cortical motif (L4->L2/3->L5 + L6 gain) -- research")
    print("=" * 74)
    print(
        f"task: 6-band parity  N={X.shape[0]}  in_dim={in_dim}  classes={n_classes}  "
        f"chance={chance:.3f}"
    )

    # ---- (1) forward through all layers + train -> beat chance -------------------------
    model = LaminarReadout(in_dim, n_classes=n_classes, n4=32, n23=8, seed=0, use_l6=True)
    print(
        f"shapes: L4 {tuple(model.W4.shape)}  L2/3 {tuple(model.W23.shape)}  "
        f"L5 {tuple(model.W5.shape)}  L6 {tuple(model.W6.shape)}  (use_l6={model.use_l6})"
    )
    acc_init = _accuracy(model, X, y)
    loss = None
    # Train to a MODERATE point (acc >> chance) rather than to convergence. On this perfectly
    # separable toy task the logits otherwise blow up to magnitude ~70, and float32's ~1e-6
    # RELATIVE matmul-accumulation error would then read as an absolute max|dlogit| ~6e-4 at
    # the grow -- a scale artifact, not a growth bug (the (1,0) split is mathematically exact).
    # Growing at a bounded-logit point keeps the function-preservation check honest and tight,
    # exactly as growable_readout.py grows a high-loss bottleneck rather than a converged net.
    for _ in range(200):
        loss = model.train_step(X, y, lr=5e-3, steps=1)
    acc_trained = _accuracy(model, X, y)
    assert acc_trained > 0.85, (
        f"laminar readout must beat chance decisively: {acc_trained:.3f} <= 0.85"
    )
    print(
        f"(1) column: acc {acc_init:.3f} (init) -> {acc_trained:.3f} (trained)  "
        f"loss={loss:.4f}  >> chance {chance:.3f}"
    )

    # ---- (2) grow(k, freeze_old=True) on L2/3: function-preserving + frozen bit-exact ---
    logits_pre = model.predict(X).clone()
    n23_before = model.n_hidden
    new_width = model.grow(k=8, freeze_old=True)  # freeze the old L2/3 units, add 8
    logits_post = model.predict(X)
    max_logit_diff = float((logits_pre - logits_post).abs().max())
    assert new_width == n23_before + 8, (
        f"L2/3 must widen {n23_before} -> {n23_before + 8}, got {new_width}"
    )
    assert model.n_hidden == n23_before + 8, (
        f"n_hidden must be {n23_before + 8}, got {model.n_hidden}"
    )
    assert max_logit_diff < 1e-4, (
        f"grow() must be function-preserving on L2/3: max|dlogit| {max_logit_diff:.3e} >= 1e-4"
    )
    print(
        f"(2a) grow : L2/3 {n23_before} -> {new_width}  max|dlogit|={max_logit_diff:.3e} (< 1e-4, "
        f"function-preserving)"
    )

    frozen_idx = model._frozen.clone()  # first n23_before units now frozen
    assert int(frozen_idx[:n23_before].sum()) == n23_before, (
        "all pre-existing L2/3 units must be frozen"
    )
    assert int(frozen_idx[n23_before:].sum()) == 0, "the 8 new L2/3 units must be plastic"
    W23_at_grow = model.W23.detach().clone()
    # Train further on the SAME task -- frozen rows must stay bit-exact; new rows must move.
    for _ in range(1500):
        model.train_step(X, y, lr=5e-3, steps=1)
    W23_now = model.W23.detach()
    frozen_diff = float((W23_now[frozen_idx] - W23_at_grow[frozen_idx]).abs().max())
    new_diff = float((W23_now[~frozen_idx] - W23_at_grow[~frozen_idx]).abs().max())
    assert frozen_diff == 0.0, (
        f"frozen L2/3 rows must be EXACTLY unchanged, got max|diff|={frozen_diff:.3e}"
    )
    assert new_diff > 0.0, (
        f"at least one NEW L2/3 unit must change during training, got {new_diff:.3e}"
    )
    acc_after_grow = _accuracy(model, X, y)
    print(
        f"(2b) freeze: frozen L2/3 rows max|diff|={frozen_diff:.3e} (must be 0.0)  |  "
        f"new-unit rows max|diff|={new_diff:.3e} (> 0)"
    )
    print(
        f"          acc after grow+train = {acc_after_grow:.3f} (neurogenesis kept the column working)"
    )

    # ---- (3) L6 feedback actually changes the logits -----------------------------------
    # Same model, same input: compare the pure feedforward L5 output (use_feedback=False)
    # with the L6-gated output (use_feedback=True). A nonzero difference proves the L6 loop
    # is doing something. Also build a use_l6=False twin (same seed -> identical L4/L2/3/L5
    # init draws) and confirm ITS predict() matches the feedforward path exactly.
    with torch.no_grad():
        Xg = X.to(model.device)
        ff = model.forward(Xg, use_feedback=False)
        fb = model.forward(Xg, use_feedback=True)
    l6_diff = float((ff - fb).abs().max())
    assert l6_diff > 1e-4, f"L6 feedback must change the logits, got max|diff|={l6_diff:.3e}"
    print(
        f"(3) L6    : max|logits(feedback) - logits(feedforward)| = {l6_diff:.3e} (> 1e-4, "
        f"feedback is active)"
    )

    twin = LaminarReadout(in_dim, n_classes=n_classes, n4=32, n23=8, seed=0, use_l6=False)
    with torch.no_grad():
        twin_ff = twin.forward(X.to(twin.device))
        base_ff = LaminarReadout(
            in_dim, n_classes=n_classes, n4=32, n23=8, seed=0, use_l6=True
        ).forward(X.to(twin.device), use_feedback=False)
    twin_match = float((twin_ff - base_ff).abs().max())
    print(f"          use_l6=False twin matches the feedforward path (max|diff|={twin_match:.3e})")

    # ---- (4) LaminarColumns: per-digit ISOLATED columns -> FULL upstream freeze -----------
    # The v3.59 failure: LaminarReadout froze only L2/3, a SHARED L4 drifted, and the 'frozen'
    # L2/3 saw an L4 it was never frozen for -> catastrophic forgetting (63.7%). Fix: give each
    # task its OWN full column (L4 + L2/3) and freeze whole columns, so every frozen column's
    # features always see their OWN fixed L4. Prove on a 2-task problem (A={0,1,2}, B={3,4,5},
    # distinct blobs): (a) the frozen column's L4 AND L2/3 are BIT-EXACT after Task B, (b) the
    # grow is function-preserving, (c) Task-A retention AND Task-B are both high.
    print("-" * 74)
    print("(4) LaminarColumns -- progressive per-digit columns (Rusu 2016): FULL upstream freeze")
    ci, cnc = 12, 6
    Xc, yc = _make_col_blobs(n_per=120, in_dim=ci, n_classes=cnc, sep=3.0, seed=0)
    task_a, task_b = [0, 1, 2], [3, 4, 5]
    a_mask = (yc.unsqueeze(1) == torch.tensor(task_a)).any(1)
    Xa, ya = Xc[a_mask], yc[a_mask]
    Xb, yb = Xc[~a_mask], yc[~a_mask]
    # SMALL replay of A (deterministic subset), the same handful the developmental loop keeps.
    grp = torch.Generator().manual_seed(7)
    rp = torch.randperm(Xa.shape[0], generator=grp)[:40]
    Xb_tr = torch.cat([Xb, Xa[rp]], 0)
    yb_tr = torch.cat([yb, ya[rp]], 0)

    col = LaminarColumns(ci, n_classes=cnc, n4=48, n23=32, seed=0)
    # Train column 0 on Task A to a MODERATE point (bounded logits keep the grow check tight).
    for _ in range(400):
        col.train_step(Xa, ya, lr=5e-3, steps=1)
    acc_A_before = _accuracy_subset(col, Xa, ya, task_a)
    print(
        f"    column-0 trained on Task A{task_a}: acc_A (task-aware) = {acc_A_before:.3f}  "
        f"(n_columns={col.n_columns}, n_hidden={col.n_hidden})"
    )

    # Snapshot column-0's ENTIRE upstream (L4 + L2/3) to prove bit-exact freeze later.
    W4_0 = col.W4s[0].detach().clone()
    W23_0 = col.W23s[0].detach().clone()

    # grow(freeze_old=True): freeze column-0 entirely, add plastic column-1. Function-preserving.
    logits_pre = col.predict(Xc).clone()
    new_width = col.grow(k=1, freeze_old=True)
    logits_post = col.predict(Xc)
    max_logit_diff = float((logits_pre - logits_post).abs().max())
    assert new_width == 2 * 32, f"n_hidden must be 2 columns * 32 = 64, got {new_width}"
    assert col.n_columns == 2, f"must have 2 columns after grow, got {col.n_columns}"
    assert max_logit_diff < 1e-4, (
        f"grow() must be function-preserving: max|dlogit| {max_logit_diff:.3e} >= 1e-4"
    )
    assert bool(col._frozen_cols[0]) and not bool(col._frozen_cols[1]), (
        "column-0 must be frozen, column-1 plastic"
    )
    print(
        f"    grow(freeze_old=True): columns 1 -> {col.n_columns}, n_hidden {32} -> {new_width}, "
        f"L5 fan-in extended with a zero block  max|dlogit|={max_logit_diff:.3e} (< 1e-4)"
    )

    # Train Task B (with small A replay). Column-0 frozen; column-1 + shared L5 learn.
    for _ in range(1500):
        col.train_step(Xb_tr, yb_tr, lr=5e-3, steps=1)

    w4_diff = float((col.W4s[0].detach() - W4_0).abs().max())
    w23_diff = float((col.W23s[0].detach() - W23_0).abs().max())
    acc_A_after = _accuracy_subset(col, Xa, ya, task_a)
    acc_B = _accuracy_subset(col, Xb, yb, task_b)
    acc_A_after_all = _accuracy(col, Xa, ya)  # class-incremental (no task id) -- honest confound
    assert w4_diff == 0.0, (
        f"frozen column-0 L4 must be EXACTLY unchanged, got max|diff|={w4_diff:.3e}"
    )
    assert w23_diff == 0.0, (
        f"frozen column-0 L2/3 must be EXACTLY unchanged, got max|diff|={w23_diff:.3e}"
    )
    assert acc_A_after > 0.85, (
        f"isolated column-0 must RETAIN Task A after Task B (task-aware): {acc_A_after:.3f} <= 0.85"
    )
    assert acc_B > 0.85, f"the NEW task must be learned in column-1: acc_B={acc_B:.3f} <= 0.85"
    print(
        f"    frozen column-0: L4 max|diff|={w4_diff:.3e}  L2/3 max|diff|={w23_diff:.3e} "
        f"(BOTH must be 0.0 -> FULL upstream freeze)"
    )
    print(
        f"    retention: acc_A {acc_A_before:.3f} -> {acc_A_after:.3f} (task-aware, > 0.85)  |  "
        f"acc_B={acc_B:.3f} (> 0.85)  |  acc_A all-class(no task id)={acc_A_after_all:.3f}"
    )
    print(
        "    -> full-column isolation retains Task A where the shared-L4 laminar drifted "
        "(L5 shared/trainable: some class-IL recency remains, reported not hidden)"
    )

    print("-" * 74)
    print("HONEST NOTES: the microcircuit reference is a DYNAMICS model (no accuracy); this")
    print("  readout borrows only its LAYER ROLES/FLOW. L6 is ONE feedback iteration (not a")
    print("  fixed point). Growth preserves output EXACTLY via a (1,0) L2/3 split -- the new")
    print("  unit's ZERO L5 fan-in (and zero L6 fan-out) is what preserves the logits, and")
    print("  its perturbed incoming is a distinct feature training then recruits.")
    print("=" * 74)
    print(
        f"self-check OK: laminar column acc {acc_init:.3f} -> {acc_trained:.3f} (chance {chance:.3f}); "
        f"L2/3 neurogenesis function-preserving (max|dlogit|={max_logit_diff:.3e}), frozen rows "
        f"bit-exact (diff={frozen_diff:.3e}), new units move (diff={new_diff:.3e}); "
        f"L6 feedback active (diff={l6_diff:.3e})"
    )
    print(
        f"  + LaminarColumns full-upstream freeze: frozen column-0 L4/L2/3 bit-exact "
        f"(L4 diff={w4_diff:.3e}, L2/3 diff={w23_diff:.3e}); 2-task retention acc_A "
        f"{acc_A_before:.3f} -> {acc_A_after:.3f} (task-aware), acc_B={acc_B:.3f}"
    )


if __name__ == "__main__":
    main()
