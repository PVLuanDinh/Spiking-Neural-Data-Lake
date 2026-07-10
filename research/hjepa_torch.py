r"""Hierarchical (2-level) nonlinear H-JEPA (PyTorch trainer) -- designed to BEAT flat.

**UNVERIFIED ON HARDWARE -- run on the RTX 5070 to measure whether the hierarchy beats
flat.** There is no GPU/torch in the environment this file was written in, so NOTHING below
was executed end-to-end here. Every torch-specific behaviour is flagged ``# VERIFY(torch):``.
No loss/accuracy/MSE numbers are hard-coded anywhere -- the banner prints only values a real
run computed. Target: the user's RTX 5070 (``NORD_GPU=1 python research/hjepa_torch.py``).

Honest starting point (the negative result this file answers, not repeats). A minimal
2-level *LINEAR* H-JEPA (once in the zero-dep core, now removed) did NOT beat flat baselines,
in two measured ways:
  (1) ROLLOUT SATURATION -- both levels used spectrally-CONTRACTIVE linear predictors, so a
      multi-step rollout collapses to a fixed point; neither level extrapolated action-driven
      regime moves, and L0 fast-roll actually beat L1 coarse-roll.
  (2) TRIVIAL ABSTRACTION -- the slow regime was a LINEAR band in the observation, so raw
      windowed AVERAGING (no learning) decoded it better than the learned L1 abstraction;
      the learned hierarchy added nothing.
This file addresses BOTH with three concrete changes (all UNVERIFIED until run on the 5070):
  * NONLINEAR encoders at BOTH levels -- deep ReLU MLPs (>=2 hidden layers), latent
    L2-normalized onto the unit sphere. L0 encodes an observation; L1 encodes a WINDOW
    (pool of K L0 latents) into an abstract latent. A nonlinear hierarchy HAS the capacity
    to represent an abstraction a linear map provably cannot (fixes failure mode 2 by
    construction of the world below).
  * EXPRESSIVE, NON-CONTRACTIVE predictors -- the abstract (L1) predictor is a pure MLP over
    the COARSE timescale (NOT a contractive residual linear map). It is NOT spectrally
    clamped; divergence is bounded by L2-normalizing every predicted latent onto the unit
    sphere (a projection, not a contraction toward a fixed point), so it can EXTRAPOLATE
    regime moves over a long horizon instead of saturating (fixes failure mode 1). The L0
    fine predictor is a residual NONLINEAR MLP (small fine steps), also unclamped + normalized
    -- residual around a NONLINEAR map, which is not the contractive-linear failure.
  * A GENUINELY NONLINEAR abstraction -- a synthetic two-timescale world whose slow REGIME is
    XOR(b0, b1) of two binary observation factors. Each factor is rendered in its OWN band, so
    a pooled observation is ADDITIVE in (b0, b1); XOR is NOT expressible as g0(b0) + g1(b1),
    hence NO linear readout of the pooled observation (and no raw windowed average) can decode
    it -- the linear-additive least-squares floor on balanced XOR{0,1} is MSE = 0.25 (the
    constant 0.5). A nonlinear L1 CAN fold the XOR interaction into a linearly-decodable
    abstract latent. Coarse actions (FLIP0 / FLIP1 / KEEP) drive the regime; fine actions
    (ADVANCE / HOLD / REVERSE) drive a within-window phase.

Method (I-JEPA / V-JEPA in a two-timescale action setting):
  * ENCODERS  f0, f1: deep ReLU MLPs; z = unit(MLP(.)). f1 mean-pools K (stop-gradient) L0
    latents then applies its own MLP, so L1 sits on top of L0 features (a probe/head geometry).
  * PREDICTORS g0 (fine, residual MLP), g1 (coarse, pure MLP); both re-normalize to the unit
    sphere -- non-contractive, bounded, action-conditioned.
  * TARGETS   EMA (momentum tau) copies of f0 and f1's head, no gradient; the prediction target
    is the stop-gradient EMA latent of the NEXT observation (L0) / NEXT window (L1).
  * LOSS at BOTH levels: invariance ||g(f(x), a) - sg(f_ema(x_next))||^2 + VICReg anti-collapse
    (variance hinge gamma=1 + off-diagonal covariance). Standard autograd; optional AMP on CUDA.

torch is imported LAZILY inside the functions/main (never at module top), so this file imports
cleanly with no torch and ``python research/hjepa_torch.py`` exits 0 with a guidance banner.
The nn.Module is defined inside build_hjepa() for the same reason. research/ is ruff-excluded
and NOT in CI; this is a trainer, not a shipped core module -- the core serves single-level
snn_jepa/1,2 only, so no core loader is required for the snn_hjepa/1 artifact export below.

NOTE ON AirLLM: deliberately NOT used. AirLLM is disk-streaming INFERENCE for VRAM-too-big
LLMs; it has no training path and would only slow a compute-bound H-JEPA. The only performance
levers here are CUDA via resolve_device(), AMP autocast/GradScaler, and (if a much deeper
encoder is ever needed) torch gradient-checkpointing -- NOT AirLLM.

References:
  H-JEPA  -- LeCun, "A Path Towards Autonomous Machine Intelligence", 2022 (the Hierarchical
             JEPA proposal: stacked JEPAs predicting at successively coarser timescales).
  V-JEPA  -- Bardes et al., "V-JEPA: Video Joint-Embedding Predictive Architectures", 2024.
  VICReg  -- Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization
             for Self-Supervised Learning", ICLR 2022.

Run (own venv, torch installed):  NORD_GPU=1 .venv-arm/Scripts/python research/hjepa_torch.py
Run (no torch):                   python research/hjepa_torch.py  -> guidance, exit 0
"""
import json
import math

IN_DIM = 48                 # 3 bands (phase, b0, b1) x BAND
BAND = 16                   # neurons per rendered factor band  (IN_DIM == 3 * BAND)
K = 8                       # fine ticks per coarse window (the L1 pooling width)
LATENT0 = 16                # fine (L0) latent dim
LATENT1 = 8                 # abstract (L1) latent dim
HIDDEN0 = 64                # L0 encoder MLP width
HIDDEN1 = 64                # L1 head MLP width
HIDDEN_P = 64               # predictor MLP width

FINE_ACTIONS = ("ADVANCE", "HOLD", "REVERSE")     # drive the within-window phase (fast)
COARSE_ACTIONS = ("FLIP0", "FLIP1", "KEEP")       # drive the XOR regime (slow)
N_FINE = len(FINE_ACTIONS)
N_COARSE = len(COARSE_ACTIONS)

HJEPA_FORMAT = "snn_hjepa/1"
DEFAULT_TAU = 0.99          # EMA target momentum
DEFAULT_VAR_COEF = 1.0      # VICReg variance-hinge weight
DEFAULT_COV_COEF = 0.04     # VICReg covariance weight (standard VICReg scale)
VICREG_GAMMA = 1.0          # target per-dim std for the variance hinge

DEFAULT_STEPS = 400         # coarse optimization steps (scaffold budget, not a full run)
DEFAULT_BATCH = 128
H_ROLL = 6                  # coarse horizon for the long-rollout claim (b): H*K fine ticks
DECODE_N = 1024             # samples for the regime-decode datasets (claim a)
FINE_STEP = 0.15            # phase advance per fine action
XOR_LINEAR_FLOOR = 0.25     # provable MSE floor for a linear-additive readout on balanced XOR{0,1}


def build_hjepa(in_dim=IN_DIM, latent0=LATENT0, latent1=LATENT1, hidden0=HIDDEN0,
                hidden1=HIDDEN1, hidden_p=HIDDEN_P, n_fine=N_FINE, n_coarse=N_COARSE,
                tau=DEFAULT_TAU):
    """Build the 2-level nonlinear H-JEPA. torch/nn imported lazily; the nn.Module is defined
    INSIDE so importing this file never requires torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def mlp(sizes):
        # ReLU MLP with len(sizes)-2 hidden layers; NO final activation (encoders normalize,
        # predictors normalize). sizes=[in, h, h, out] -> 2 hidden layers (>=2 as required).
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class _HJepa(nn.Module):
        def __init__(self):
            super().__init__()
            # --- online encoders (deep, nonlinear; latent L2-normalized in forward) ---
            self.enc0 = mlp([in_dim, hidden0, hidden0, latent0])          # L0: obs -> fine latent
            self.enc1 = mlp([latent0, hidden1, hidden1, latent1])         # L1 head: pooled L0 -> abstract
            # --- predictors (action-conditioned; non-contractive, bounded by normalize) ---
            self.pred0 = mlp([latent0 + n_fine, hidden_p, hidden_p, latent0])    # fine (residual in forward)
            self.pred1 = mlp([latent1 + n_coarse, hidden_p, hidden_p, latent1])  # coarse (pure MLP, no skip)
            # --- EMA target encoders (frozen copies; updated only by ema_update) ---
            self.enc0_t = mlp([in_dim, hidden0, hidden0, latent0])
            self.enc1_t = mlp([latent0, hidden1, hidden1, latent1])
            for tgt, onl in ((self.enc0_t, self.enc0), (self.enc1_t, self.enc1)):
                for pt, po in zip(tgt.parameters(), onl.parameters()):
                    pt.data.copy_(po.data)
                    pt.requires_grad_(False)                             # # VERIFY(torch): no-grad target
            self.tau = float(tau)

        # ---- encoders --------------------------------------------------------------------
        def encode0(self, x):
            return F.normalize(self.enc0(x), dim=-1)                     # unit(MLP(x))

        def encode0_target(self, x):
            with torch.no_grad():                                       # stop-gradient EMA branch
                return F.normalize(self.enc0_t(x), dim=-1)

        def _stack0(self, window, target=False):
            # window [B, K, in_dim] -> per-tick L0 latents [B, K, latent0]
            b, k, d = window.shape
            flat = window.reshape(b * k, d)
            z = self.encode0_target(flat) if target else self.encode0(flat)
            return z.reshape(b, k, -1)

        def encode1(self, window):
            # L1 sits on top of DETACHED L0 features (probe/head geometry: enc0 is trained by
            # the fine loss only; enc1 learns the abstraction on top). Mean-pool over K averages
            # out the fast phase and keeps the slow (b0,b1) -> the XOR-carrying signal.
            z0s = self._stack0(window).detach()                         # # VERIFY(torch): sg L0 -> L1
            return F.normalize(self.enc1(z0s.mean(dim=1)), dim=-1)

        def encode1_from_latents(self, z0s):
            # same L1 head but from ALREADY-computed L0 latents [B, m, latent0] (used to decode a
            # rolled-out L0 prediction through the coarse abstraction).
            return F.normalize(self.enc1(z0s.mean(dim=1)), dim=-1)

        def encode1_target(self, window):
            with torch.no_grad():
                z0s = self._stack0(window, target=True)
                return F.normalize(self.enc1_t(z0s.mean(dim=1)), dim=-1)

        # ---- predictors ------------------------------------------------------------------
        def predict0(self, z0, a_fine):
            # residual NONLINEAR fine predictor (small phase steps); re-normalized -> bounded,
            # NOT spectrally clamped, NOT a contractive linear map.
            d = self.pred0(torch.cat([z0, a_fine], dim=-1))
            return F.normalize(z0 + d, dim=-1)

        def predict1(self, z1, a_coarse):
            # coarse abstract predictor: a PURE MLP (no residual skip), re-normalized onto the
            # unit sphere. Expressive + non-contractive so it can EXTRAPOLATE regime moves over a
            # long horizon rather than collapse to a fixed point (the linear-core failure mode 1).
            return F.normalize(self.pred1(torch.cat([z1, a_coarse], dim=-1)), dim=-1)

        # ---- EMA target update -----------------------------------------------------------
        def ema_update(self):
            with torch.no_grad():
                for tgt, onl in ((self.enc0_t, self.enc0), (self.enc1_t, self.enc1)):
                    for pt, po in zip(tgt.parameters(), onl.parameters()):
                        pt.mul_(self.tau).add_(po, alpha=1.0 - self.tau)   # # VERIFY(torch): momentum update

    return _HJepa()


def vicreg_terms(z, gamma=VICREG_GAMMA, eps=1e-4):
    """VICReg anti-collapse on an embedding batch z [batch, latent]. Returns (var_loss, cov_loss):
    variance hinge mean(relu(gamma - std_j)) keeps each dim's batch std up; covariance sum of
    squared OFF-diagonal covariances / latent decorrelates the dims. Blocks the constant latent."""
    import torch

    std = torch.sqrt(z.var(dim=0) + eps)                 # [latent] per-dim std over the batch
    var_loss = torch.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    n = max(1, z.shape[0] - 1)
    cov = (zc.t() @ zc) / n
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / z.shape[1]
    return var_loss, cov_loss


class TwoTimescaleWorld:
    """Self-contained two-timescale world with a NONLINEAR (XOR) slow regime.

    Hidden state per trajectory: a fast phase p in [0,1) and two slow binary factors b0, b1 in
    {0,1}. Each observation is a banded [0,1]^in_dim RBF rendering of (p, pos(b0), pos(b1)) with
    b -> {0.25, 0.75}; the three factors occupy DISJOINT bands, so a pooled observation is
    ADDITIVE in (b0, b1). The slow REGIME is r = XOR(b0, b1) -- NOT expressible as g0(b0)+g1(b1),
    so no linear readout of the pooled observation (nor a raw windowed average) can decode it,
    while a nonlinear L1 can. Fine actions advance/hold/reverse the phase (fast); coarse actions
    FLIP0 / FLIP1 / KEEP move the regime (slow). Deterministic given `gen`. torch imported lazily."""

    def __init__(self, gen, device, batch=DEFAULT_BATCH, k=K, band=BAND,
                 n_fine=N_FINE, n_coarse=N_COARSE, step=FINE_STEP, noise=0.02):
        import torch

        self.gen, self.device = gen, device
        self.batch, self.k, self.band = batch, k, band
        self.n_fine, self.n_coarse, self.noise = n_fine, n_coarse, noise
        self.centers = torch.linspace(0.0, 1.0, band, device=device)      # [band] band centers
        self.sigma = 1.5 / max(1, band - 1)                               # RBF width ~ band spacing
        self.eff_fine = torch.tensor([step, 0.0, -step], device=device)   # ADVANCE / HOLD / REVERSE
        # running trajectory state for next_coarse_batch (contiguous coarse stream)
        self.phase = torch.rand(batch, generator=gen, device=device)
        self.b0 = (torch.rand(batch, generator=gen, device=device) < 0.5).float()
        self.b1 = (torch.rand(batch, generator=gen, device=device) < 0.5).float()
        self._cur = None                                                  # cached current window/actions

    # ---- rendering -----------------------------------------------------------------------
    def _band(self, pos):
        # pos [n] -> RBF activations [n, band]
        import torch

        diff = self.centers.unsqueeze(0) - pos.unsqueeze(1)
        return torch.exp(-0.5 * (diff / self.sigma) ** 2)

    def _render(self, phase, b0, b1):
        import torch

        pos0 = 0.25 + 0.5 * b0                                            # b in {0,1} -> {0.25,0.75}
        pos1 = 0.25 + 0.5 * b1
        x = torch.cat([self._band(phase), self._band(pos0), self._band(pos1)], dim=1)
        jitter = (torch.rand(x.shape, generator=self.gen, device=self.device) * 2.0 - 1.0) * self.noise
        return (x + jitter).clamp(0.0, 1.0)

    def _gen_window(self, phase, b0, b1):
        # roll K fine ticks with random fine actions; return (window[.,K,in], a_fine[.,K,n_fine], phase_end)
        import torch

        n = phase.shape[0]
        obs_list, act_list = [], []
        p = phase
        for _ in range(self.k):
            obs_list.append(self._render(p, b0, b1))
            a_idx = torch.randint(0, self.n_fine, (n,), generator=self.gen, device=self.device)
            a_oh = torch.nn.functional.one_hot(a_idx, self.n_fine).float()   # # VERIFY(torch)
            eff = self.eff_fine[a_idx]
            act_list.append(a_oh)
            p = torch.remainder(p + eff, 1.0)                            # # VERIFY(torch): wrap phase into [0,1)
        return torch.stack(obs_list, dim=1), torch.stack(act_list, dim=1), p

    def _apply_coarse(self, b0, b1, a_idx):
        # FLIP0 -> flip b0; FLIP1 -> flip b1; KEEP -> unchanged.
        import torch
        nb0 = torch.where(a_idx == 0, 1.0 - b0, b0)
        nb1 = torch.where(a_idx == 1, 1.0 - b1, b1)
        return nb0, nb1

    # ---- streams -------------------------------------------------------------------------
    def next_coarse_batch(self):
        """One coarse transition on the running trajectory: returns (window, a_fine, a_coarse_onehot,
        next_window). window/next_window are [batch, K, in_dim]; a_fine [batch, K, n_fine];
        a_coarse_onehot [batch, n_coarse]."""
        import torch

        if self._cur is None:
            self._cur = self._gen_window(self.phase, self.b0, self.b1)
        cur_window, cur_afine, self.phase = self._cur
        a_idx = torch.randint(0, self.n_coarse, (self.batch,), generator=self.gen, device=self.device)
        a_coarse = torch.nn.functional.one_hot(a_idx, self.n_coarse).float()
        nb0, nb1 = self._apply_coarse(self.b0, self.b1, a_idx)
        next_window, next_afine, self.phase = self._gen_window(self.phase, nb0, nb1)
        self._cur = (next_window, next_afine, self.phase)
        self.b0, self.b1 = nb0, nb1
        return cur_window, cur_afine, a_coarse, next_window

    def sample_windows(self, n):
        """A fresh, BALANCED decode dataset: draw (phase, b0, b1) uniformly and render one K-window
        each. Returns (window [n, K, in_dim], regime [n] = XOR(b0, b1) in {0.,1.})."""
        import torch

        phase = torch.rand(n, generator=self.gen, device=self.device)
        b0 = (torch.rand(n, generator=self.gen, device=self.device) < 0.5).float()
        b1 = (torch.rand(n, generator=self.gen, device=self.device) < 0.5).float()
        window, _a, _p = self._gen_window(phase, b0, b1)
        regime = (b0 != b1).float()                                       # XOR
        return window, regime

    def rollout_episode(self, h):
        """A fresh H-coarse-step episode for the long-rollout claim (b). Returns
        (window0 [B,K,in_dim], fine_actions [B,H,K,n_fine] for windows 0..H-1,
        coarse_actions [B,H,n_coarse] for steps 0..H-1, regime_H [B] = XOR after H coarse moves)."""
        import torch

        phase = torch.rand(self.batch, generator=self.gen, device=self.device)
        b0 = (torch.rand(self.batch, generator=self.gen, device=self.device) < 0.5).float()
        b1 = (torch.rand(self.batch, generator=self.gen, device=self.device) < 0.5).float()
        window0, af0, phase = self._gen_window(phase, b0, b1)
        fine_windows = [af0]                                              # window0 fine actions
        coarse_list = []
        cur_b0, cur_b1 = b0, b1
        for step in range(h):
            a_idx = torch.randint(0, self.n_coarse, (self.batch,), generator=self.gen, device=self.device)
            coarse_list.append(torch.nn.functional.one_hot(a_idx, self.n_coarse).float())
            cur_b0, cur_b1 = self._apply_coarse(cur_b0, cur_b1, a_idx)
            w_next, af_next, phase = self._gen_window(phase, cur_b0, cur_b1)
            if step < h - 1:                                             # collect fine actions for windows 1..H-1
                fine_windows.append(af_next)
        regime_h = (cur_b0 != cur_b1).float()
        fine_actions = torch.stack(fine_windows, dim=1)                  # [B, H, K, n_fine]
        coarse_actions = torch.stack(coarse_list, dim=1)                 # [B, H, n_coarse]
        return window0, fine_actions, coarse_actions, regime_h


def train_step(model, opt, batch, device="cpu", scaler=None,
               var_coef=DEFAULT_VAR_COEF, cov_coef=DEFAULT_COV_COEF):
    """One optimization step: invariance + VICReg at BOTH levels, then an EMA target update.
    Optional AMP on CUDA via `scaler`. Returns the REAL loss components (no fabricated numbers)."""
    import torch

    window, a_fine, a_coarse, next_window = batch
    b, k, d = window.shape
    is_cuda = str(device).startswith("cuda")
    dev_type = "cuda" if is_cuda else "cpu"
    use_amp = bool(scaler is not None and is_cuda)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev_type, enabled=use_amp):          # # VERIFY(torch): AMP autocast
        # --- L0 fine level: within-window consecutive (obs_i, a_fine_i, obs_{i+1}) pairs ---
        obs_t = window[:, :-1, :].reshape(b * (k - 1), d)
        obs_n = window[:, 1:, :].reshape(b * (k - 1), d)
        af = a_fine[:, :-1, :].reshape(b * (k - 1), a_fine.shape[-1])
        z0 = model.encode0(obs_t)
        p0 = model.predict0(z0, af)
        t0 = model.encode0_target(obs_n)
        inv0 = (p0 - t0).pow(2).sum(dim=1).mean()
        var0, cov0 = vicreg_terms(z0)
        # --- L1 coarse level: (window_t, a_coarse_t, window_{t+1}) ---
        z1 = model.encode1(window)
        p1 = model.predict1(z1, a_coarse)
        t1 = model.encode1_target(next_window)
        inv1 = (p1 - t1).pow(2).sum(dim=1).mean()
        var1, cov1 = vicreg_terms(z1)
        loss = inv0 + inv1 + var_coef * (var0 + var1) + cov_coef * (cov0 + cov1)
    if use_amp:
        scaler.scale(loss).backward()                                    # # VERIFY(torch): scaled backward
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    model.ema_update()                                                   # momentum targets AFTER the online step
    return {
        "loss": float(loss.detach().cpu()),
        "inv0": float(inv0.detach().cpu()), "inv1": float(inv1.detach().cpu()),
        "var": float((var0 + var1).detach().cpu()), "cov": float((cov0 + cov1).detach().cpu()),
    }


# ---- linear readout helpers (closed-form ridge; the FAIR flat/averaging baselines) ----------
def _ridge_fit(x, y, lam=1e-2):
    """Closed-form ridge readout on CPU float32 (augmented with a bias). Returns W [d+1, 1]."""
    import torch

    x = x.detach().float().cpu()
    y = y.detach().float().cpu().reshape(-1, 1)
    n, d = x.shape
    xa = torch.cat([x, torch.ones(n, 1, device="cpu")], dim=1)
    a = xa.t() @ xa + lam * torch.eye(d + 1, device="cpu")
    return torch.linalg.solve(a, xa.t() @ y)                             # # VERIFY(torch): linalg.solve


def _ridge_pred(w, x):
    import torch

    x = x.detach().float().cpu()
    xa = torch.cat([x, torch.ones(x.shape[0], 1, device="cpu")], dim=1)
    return (xa @ w).reshape(-1)


def _mse(pred, y):
    y = y.detach().float().cpu().reshape(-1)                             # pred is already CPU float from _ridge_pred
    return float(((pred - y) ** 2).mean())


def _band_means(pooled, band):
    """Raw windowed averaging reduced to the two factor-band means [n, 2] -- 'just average the
    window and read the factors'. Additive in (b0, b1) -> provably cannot decode XOR."""
    import torch

    f0 = pooled[:, band:2 * band].mean(dim=1)                            # factor-0 band mean
    f1 = pooled[:, 2 * band:3 * band].mean(dim=1)                        # factor-1 band mean
    return torch.stack([f0, f1], dim=1)


def evaluate_hierarchy_vs_flat(model, world, h=H_ROLL, n=DECODE_N):
    """Measure the two claims the linear H-JEPA FAILED. Returns a dict of REAL numbers.

    (a) hierarchy-vs-flat abstraction: the learned L1 abstract latent decodes the XOR regime
        better than (i) a flat linear pooled-obs readout AND (ii) raw windowed averaging.
    (b) long-horizon regime rollout: L1's coarse MLP predictor (H coarse jumps) decodes the
        horizon regime better than the L0 fine predictor rolled the same H*K ticks (which
        saturates and is blind to the coarse regime moves)."""
    import torch

    model.eval()
    with torch.no_grad():
        # ----- claim (a): fit readouts on a train split, evaluate on a held-out split -----
        w_tr, r_tr = world.sample_windows(n)
        w_te, r_te = world.sample_windows(n)
        z1_tr, z1_te = model.encode1(w_tr), model.encode1(w_te)          # learned L1 abstraction
        pooled_tr, pooled_te = w_tr.mean(dim=1), w_te.mean(dim=1)        # mean-pooled observation
        raw_tr, raw_te = _band_means(pooled_tr, world.band), _band_means(pooled_te, world.band)

        w_hier = _ridge_fit(z1_tr, r_tr)
        w_flat = _ridge_fit(pooled_tr, r_tr)
        w_raw = _ridge_fit(raw_tr, r_tr)
        mse_hier = _mse(_ridge_pred(w_hier, z1_te), r_te)
        mse_flat = _mse(_ridge_pred(w_flat, pooled_te), r_te)
        mse_raw = _mse(_ridge_pred(w_raw, raw_te), r_te)
        claim_a = (mse_hier < mse_flat) and (mse_hier < mse_raw)

        # ----- claim (b): long-horizon regime prediction, coarse-roll vs fine-roll -----
        w0, fine_actions, coarse_actions, regime_h = world.rollout_episode(h)
        # L1 coarse roll: H applications of the coarse MLP predictor (each re-normalized).
        z1 = model.encode1(w0)
        for step in range(h):
            z1 = model.predict1(z1, coarse_actions[:, step, :])
        mse_l1 = _mse(_ridge_pred(w_hier, z1), regime_h)
        # L0 fine roll: H*K applications of the fine predictor from obs0's L0 latent, then decode
        # the drifted latent through the SAME coarse abstraction (encode1_from_latents).
        z0 = model.encode0(w0[:, 0, :])
        fa = fine_actions.reshape(w0.shape[0], h * world.k, fine_actions.shape[-1])
        for t in range(h * world.k):
            z0 = model.predict0(z0, fa[:, t, :])
        z1_from_l0 = model.encode1_from_latents(z0[:, None, :])
        mse_l0 = _mse(_ridge_pred(w_hier, z1_from_l0), regime_h)
        claim_b = mse_l1 < mse_l0

    finite = all(math.isfinite(v) for v in (mse_hier, mse_flat, mse_raw, mse_l1, mse_l0))
    return {
        "mse_hier": mse_hier, "mse_flat": mse_flat, "mse_raw": mse_raw, "claim_a": claim_a,
        "mse_l1_roll": mse_l1, "mse_l0_roll": mse_l0, "claim_b": claim_b,
        "h": h, "fine_ticks": h * world.k, "finite": finite,
    }


def export_hjepa(model, path):
    """Dump BOTH levels' weights as an snn_hjepa/1 JSON research artifact (no core loader is
    required -- the core serves single-level snn_jepa only). Weights are nn.Linear [out][in]
    via .tolist() (no transpose). Returns the written dict."""
    import torch.nn as nn

    def _seq(seq):
        return [[m.weight.detach().cpu().tolist(), m.bias.detach().cpu().tolist()]
                for m in seq if isinstance(m, nn.Linear)]

    blob = {
        "format": HJEPA_FORMAT,
        "in_dim": IN_DIM, "K": K, "latent0": LATENT0, "latent1": LATENT1,
        "fine_actions": list(FINE_ACTIONS), "coarse_actions": list(COARSE_ACTIONS),
        "enc0": _seq(model.enc0), "enc1": _seq(model.enc1),               # online encoders (application order)
        "enc0_target": _seq(model.enc0_t), "enc1_target": _seq(model.enc1_t),
        "pred0": _seq(model.pred0), "pred1": _seq(model.pred1),           # fine (residual) / coarse (pure MLP)
        "tau": float(model.tau),
        "predictors": {"pred0": "residual_mlp_unit", "pred1": "mlp_unit"},
        "meta": {"trained_by": "hjepa_torch", "unverified_on_hardware": True,
                 "abstraction": "xor(b0,b1)"},
    }
    with open(path, "w", encoding="ascii") as f:
        json.dump(blob, f)
    return blob


def main():
    # --- no-torch guard: import lazily so this file is safe to run with no torch installed ---
    try:
        import torch
    except ImportError:
        print("=" * 72)
        print("HIERARCHICAL H-JEPA TORCH TRAINER (research/, not CI) -- torch NOT installed")
        print("=" * 72)
        print("A 2-level NONLINEAR H-JEPA (LeCun 2022 H-JEPA / V-JEPA 2024 / VICReg 2022).")
        print("Needs a GPU + torch to train and measure -- nothing to run without torch.")
        print("needs torch: pip install '.[gpu]'")
        print("run on the RTX 5070: NORD_GPU=1 python research/hjepa_torch.py")
        print("")
        print("Two claims this file MEASURES on the GPU (the minimal LINEAR H-JEPA FAILED BOTH):")
        print("  (a) the learned nonlinear L1 abstract latent decodes the XOR regime BETTER than")
        print("      (i) a flat linear pooled-obs baseline AND (ii) raw windowed averaging.")
        print("      Linear core: raw averaging BEAT the learned abstraction (trivial abstraction).")
        print("  (b) long-horizon regime prediction via L1's coarse MLP predictor BEATS the L0 fine")
        print("      predictor rolled the SAME number of ticks.")
        print("      Linear core: contractive predictors saturated -> L0 fast-roll BEAT L1 coarse-roll.")
        print("")
        print("How this addresses them (capacity is present by CONSTRUCTION; realization UNVERIFIED):")
        print("  - nonlinear ReLU-MLP encoders at both levels (a linear map provably cannot do XOR;")
        print("    the world's slow regime is XOR(b0,b1), additive-in-obs, so flat/averaging are floored)")
        print("  - non-contractive MLP predictors, bounded by unit-sphere normalization (NOT spectrally")
        print("    clamped), so the coarse predictor can EXTRAPOLATE regime moves instead of saturating.")
        print("Exiting cleanly (0).")
        return 0

    import os
    import tempfile

    from torch_device import enable, resolve_device                      # shared cpu/cuda selector

    device = enable(resolve_device())                                    # # VERIFY(torch): NORD_GPU -> cuda
    is_cuda = str(device).startswith("cuda")
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)                  # # VERIFY(torch): device-side generator

    model = build_hjepa().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)               # # VERIFY(torch): AMP (no-op on cpu)
    world = TwoTimescaleWorld(gen, device)

    last = {}
    for _ in range(DEFAULT_STEPS):
        last = train_step(model, opt, world.next_coarse_batch(), device=device, scaler=scaler)

    res = evaluate_hierarchy_vs_flat(model, world)

    export_path = os.path.join(tempfile.gettempdir(), "snn_hjepa1_scaffold.json")
    blob = export_hjepa(model, export_path)

    print("=" * 72)
    print("HIERARCHICAL H-JEPA TORCH TRAINER -- 2-level nonlinear H-JEPA, UNVERIFIED")
    print("=" * 72)
    print(f"backend      : torch {torch.__version__}  device={device}  amp={is_cuda}")
    print(f"world        : two-timescale, slow regime = XOR(b0,b1); fine {list(FINE_ACTIONS)}, "
          f"coarse {list(COARSE_ACTIONS)}")
    print(f"encoders     : L0 MLP [{IN_DIM}->{HIDDEN0}->{HIDDEN0}->{LATENT0}]  "
          f"L1 head MLP [{LATENT0}->{HIDDEN1}->{HIDDEN1}->{LATENT1}] (both unit-normalized, 2 hidden)")
    print(f"predictors   : L0 residual-MLP (fine), L1 pure-MLP (coarse); unit-sphere bounded, NOT clamped")
    print(f"train ({DEFAULT_STEPS} steps): loss={last['loss']:.4f} inv0={last['inv0']:.4f} "
          f"inv1={last['inv1']:.4f} var={last['var']:.4f} cov={last['cov']:.4f}   (real, from this run)")
    print("-" * 72)
    print("[GPU-VERIFY] claim (a) NONLINEAR abstraction (fixes the trivial-abstraction failure):")
    print(f"[GPU-VERIFY]   learned L1 abstract  XOR-regime decode MSE = {res['mse_hier']:.4f}")
    print(f"[GPU-VERIFY]   flat linear pooled-obs          decode MSE = {res['mse_flat']:.4f}")
    print(f"[GPU-VERIFY]   raw windowed averaging          decode MSE = {res['mse_raw']:.4f}")
    print(f"[GPU-VERIFY]   (linear-additive floor on balanced XOR = {XOR_LINEAR_FLOOR:.2f})")
    print(f"[GPU-VERIFY]   claim (a) HOLDS (L1 beats BOTH flat and averaging): {res['claim_a']}")
    print("[GPU-VERIFY] claim (b) long-horizon regime rollout (fixes the rollout-saturation failure):")
    print(f"[GPU-VERIFY]   L1 coarse-roll ({res['h']} coarse steps) regime MSE = {res['mse_l1_roll']:.4f}")
    print(f"[GPU-VERIFY]   L0 fine-roll  ({res['fine_ticks']} fine ticks) regime MSE = {res['mse_l0_roll']:.4f}")
    print(f"[GPU-VERIFY]   claim (b) HOLDS (L1 coarse-roll beats L0 fine-roll): {res['claim_b']}")
    print("-" * 72)
    print(f"exported     : {HJEPA_FORMAT} -> {export_path} "
          f"(enc0/enc1/pred0/pred1 + EMA targets; research artifact, no core loader)")
    print("NOTE: UNVERIFIED ON HARDWARE. Capacity to beat flat is present by construction; whether")
    print("the SSL objective REALIZES both claims is what THIS GPU run measures (see [GPU-VERIFY]).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
