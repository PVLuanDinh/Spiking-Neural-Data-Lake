r"""ARM-NATIVE 2-level nonlinear H-JEPA (PyTorch trainer) -- trained so it can be SERVED.

**RUN 2026-08-04, AND THE PRIMARY CLAIM MEASURED FALSE.** The "UNVERIFIED ON HARDWARE / there is
no GPU/torch in the environment this file was written in, so NOTHING below was executed" banner
that stood here was STALE -- torch is an unconditional core dependency now (pyproject.toml), and
this file runs end-to-end and exits 0. Executed on torch 2.13.0+cpu, device=cpu, at the shipped
defaults (450 steps, DECODE_N=1024):

    learned L1 abstract (K-pool)  regime decode  MSE=0.0985  acc=0.923
    flat single-frame z0          regime decode  MSE=0.0966  acc=0.815
    raw windowed averaging (64d)  regime decode  MSE=0.0000  acc=1.000
    PRIMARY claim (mse_hier < mse_flat) -> FALSE          secondary (vs raw pool) -> FALSE

So the temporal-abstraction win this file was built to show DOES NOT HOLD on MSE, and the
un-learned raw window average decodes the regime PERFECTLY (MSE 0.0000, acc 1.000) while the
learned abstraction does not. Note also that the two observables DISAGREE in direction -- z1 is
worse on MSE (0.0985 vs 0.0966) and better on accuracy (0.923 vs 0.815) -- so which claim you
report is decided by which of two available numbers you pick, and this file picks MSE. Read the
"EXPECTED TO HOLD by construction" paragraph below against that: the pooling CAPACITY is present
by construction, and the SSL objective demonstrably does not realise it here.

Every torch-specific behaviour is still flagged ``# VERIFY(torch):``. No loss/accuracy/MSE numbers
are hard-coded in the code path -- the [GPU-VERIFY] banner prints only values a real run computed;
the figures above are a recorded result, re-derivable with the command at the bottom of this
docstring. The GPU (RTX 5070) run has NOT been done; the numbers above are CPU.

WHY THIS FILE (vs reasoning/hjepa_torch.py). The v3.13 H-JEPA (hjepa_torch.py) is trained on a
toy XOR world with FLIP0/FLIP1/KEEP actions -- USELESS for the arm, whose action set is
{ENGAGE, RELEASE, HOLD} and whose observations are the encode_multimodal
[0,1]^64 banded vector. This trainer reuses that file's architecture WHOLESALE (the _HJepa
nn.Module: deep ReLU-MLP encode0/encode1, unit-normalized latents, EMA targets; VICReg
anti-collapse; the residual-fine / pure-coarse predictors; the resolve_device/AMP/no-torch
pattern; the snn_hjepa/1 export layout) but swaps in an ARM-NATIVE two-timescale world so
the exported snn_hjepa/1 artifact can be loaded by the zero-dep core server
``src/snn_data_lake/faculties/hjepa_serve.py`` and served in the live arm brain.

THE ARM-NATIVE WORLD (ArmTwoTimescaleWorld):
  * Observations are [0,1]^64 BANDED exactly like sensorimotor/multimodal_encoder.encode_multimodal:
    four modality bands of 16 -- visual[0:16], temperature[16:32], audio[32:48],
    distance[48:64]. (The torch world builds the banded [0,1] shape with a GPU RBF; it MIRRORS
    the encode_multimodal layout rather than calling that stdlib/CPU function on GPU tensors.)
  * A SLOW REGIME r in {0,1} lives on the K-clock: it is CONSTANT within a K=8 window and can
    change only at window boundaries (a slow per-window Markov flip, prob P_FLIP), so it
    persists across several windows -- genuinely SLOW relative to the fast per-tick nuisance.
    r is EXOGENOUS: the arm cannot flip it (it is task context, e.g. "is a graspable object
    present"). r is rendered into the DISTANCE band and it also names the optimal arm action:
    r=0 -> RELEASE, r=1 -> ENGAGE, HOLD is never optimal (served downstream via the
    meta.regime_to_action map; the SSL trainer itself is reward-free).
  * A FAST within-window nuisance corrupts the regime band per FRAME so a SINGLE frame is
    ambiguous about r, but the K-window POOL reveals it: the distance-band position is
    0.25 + 0.5*r + A*cos(2*pi*tick/K + phi). Over exactly K ticks that cosine sums to ~0 (roots
    of unity), so the pooled distance band recenters on 0.25/0.75 (=> r) while any one frame is
    smeared. This is the whole point: pooling (z1 / raw window-average) recovers r; one frame
    (z0) cannot.
  * Actions drive a SEPARATE fast factor so obs_t -(arm action)-> obs_{t+1} is a real,
    learnable transition: the arm action advances an AUDIO-band phase q (ENGAGE +step,
    RELEASE -step, HOLD 0), so the fine predictor has a genuine action-conditioned target.
    The regime is untouched by the action (exogenous, on the slow clock).
  * FINE_ACTIONS == COARSE_ACTIONS == the 3 ARM actions: the arm has ONE action set. The
    hierarchy here is TEMPORAL (z1 abstracts K ticks of z0), NOT an action decomposition, so
    n_fine == n_coarse == 3.

HONEST EXPECTATION FOR THE [GPU-VERIFY] CLAIM (do not rig it). The measured question is:
does the learned ABSTRACT z1 (pooled over the window) decode the slow regime r better than
(i) the flat single-frame z0 and (ii) raw windowed averaging of the observation?
  * z1 vs flat z0  -- EXPECTED TO HOLD by construction: z0 sees one nuisance-corrupted frame
    (ambiguous), z1 pools K frames (nuisance cancels). This is the temporal-abstraction win the
    file is built to show.
  * z1 vs raw window-average -- HONESTLY NOT EXPECTED TO BE A CLEAN WIN. Unlike the XOR world
    (where the slow regime was XOR(b0,b1), provably NOT linearly decodable from the pooled obs,
    flooring raw averaging at MSE 0.25), the ARM regime is a SINGLE binary factor rendered
    LINEARLY in one band. Raw window-averaging removes the same nuisance z1 does and a linear
    readout of the pooled band decodes r well -- a strong baseline. So z1 may TIE or LOSE to raw
    averaging. The banner reports all three; the primary claim printed as HOLDS/FAILS is
    z1-beats-flat-z0. Whether the SSL objective (which never sees r) makes z1 encode r at all is
    exactly what the GPU run measures.

torch is imported LAZILY inside functions/main (module top imports stdlib json/math/os/tempfile
only), so this file imports cleanly with no torch and ``python -m
snn_data_lake.reasoning.hjepa_arm_torch`` exits 0 with a guidance banner. (Said ``python
research/hjepa_arm_torch.py`` until 2026-08-05 -- no such file exists in a clone; the same stale
command was also being PRINTED by the no-torch banner below.)
The nn.Module is defined INSIDE build_hjepa_arm() for the same
reason. (The claim "research/ is ruff-excluded and NOT in CI" was removed 2026-08-04: this file
lives in reasoning/, which HAS been ruff-linted since 2026-07-28. Not in run_selfchecks, though --
so nothing gates the measurement above, which is why it is written down here.)

References:
  H-JEPA  -- LeCun, "A Path Towards Autonomous Machine Intelligence", 2022 (Hierarchical JEPA:
             stacked JEPAs predicting at successively coarser timescales).
  V-JEPA  -- Bardes et al., "V-JEPA: Video Joint-Embedding Predictive Architectures", 2024.
  VICReg  -- Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization for
             Self-Supervised Learning", ICLR 2022.

Run:  NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python -m snn_data_lake.reasoning.hjepa_arm_torch
CPU:  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.hjepa_arm_torch
"""

import json
import math
import os
import tempfile

IN_DIM = 64  # 4 modality bands x BAND (mirrors encode_multimodal's N=64)
BAND = 16  # neurons per modality band  (IN_DIM == 4 * BAND)
K = 8  # fine ticks per coarse window (the L1 pooling width == regime clock)
LATENT0 = 16  # fine (L0) latent dim
LATENT1 = 8  # abstract (L1) latent dim
HIDDEN0 = 64  # L0 encoder MLP width
HIDDEN1 = 64  # L1 head MLP width
HIDDEN_P = 64  # predictor MLP width

# The arm has ONE action set; the hierarchy is TEMPORAL, so fine == coarse == the arm actions.
ARM_ACTIONS = ("ENGAGE", "RELEASE", "HOLD")  # brainstem/interpreter.py command set
FINE_ACTIONS = ARM_ACTIONS
COARSE_ACTIONS = ARM_ACTIONS
N_FINE = len(FINE_ACTIONS)
N_COARSE = len(COARSE_ACTIONS)
# r names the optimal arm action (served downstream; the SSL trainer is reward-free).
REGIME_TO_ACTION = {"0": "RELEASE", "1": "ENGAGE"}  # HOLD is never optimal

HJEPA_FORMAT = "snn_hjepa/1"
DEFAULT_TAU = 0.99  # EMA target momentum
DEFAULT_VAR_COEF = 1.0  # VICReg variance-hinge weight
DEFAULT_COV_COEF = 0.04  # VICReg covariance weight (standard VICReg scale)
VICREG_GAMMA = 1.0  # target per-dim std for the variance hinge

DEFAULT_STEPS = 450  # coarse optimization steps (scaffold budget, not a full sweep)
DEFAULT_BATCH = 128
DECODE_N = 1024  # samples per split for the regime-decode datasets
REGIME_AMP = 0.33  # fast within-window nuisance amplitude on the regime (distance) band
AUDIO_STEP = 0.13  # arm-action phase advance on the audio band (the learnable transition)
P_FLIP = 0.20  # per-window (K-clock) regime flip prob -> slow, persistent regime
BRIDGE_TOL = 1e-4  # torch-vs-core snn_hjepa/1 transfer tolerance


def build_hjepa_arm(
    in_dim=IN_DIM,
    latent0=LATENT0,
    latent1=LATENT1,
    hidden0=HIDDEN0,
    hidden1=HIDDEN1,
    hidden_p=HIDDEN_P,
    n_fine=N_FINE,
    n_coarse=N_COARSE,
    tau=DEFAULT_TAU,
):
    """Build the 2-level nonlinear H-JEPA (arm action/obs dims). Architecture is REUSED wholesale
    from reasoning/hjepa_torch.py's _HJepa. torch/nn imported lazily; the nn.Module is defined
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
            self.enc0 = mlp([in_dim, hidden0, hidden0, latent0])  # L0: obs -> fine latent
            self.enc1 = mlp([latent0, hidden1, hidden1, latent1])  # L1 head: pooled L0 -> abstract
            # --- predictors (action-conditioned; non-contractive, bounded by normalize) ---
            self.pred0 = mlp(
                [latent0 + n_fine, hidden_p, hidden_p, latent0]
            )  # fine (residual in forward)
            self.pred1 = mlp(
                [latent1 + n_coarse, hidden_p, hidden_p, latent1]
            )  # coarse (pure MLP, no skip)
            # --- EMA target encoders (frozen copies; updated only by ema_update) ---
            self.enc0_t = mlp([in_dim, hidden0, hidden0, latent0])
            self.enc1_t = mlp([latent0, hidden1, hidden1, latent1])
            for tgt, onl in ((self.enc0_t, self.enc0), (self.enc1_t, self.enc1)):
                for pt, po in zip(tgt.parameters(), onl.parameters()):
                    pt.data.copy_(po.data)
                    pt.requires_grad_(False)  # # VERIFY(torch): no-grad target
            self.tau = float(tau)

        # ---- encoders --------------------------------------------------------------------
        def encode0(self, x):
            return F.normalize(self.enc0(x), dim=-1)  # unit(MLP(x))

        def encode0_target(self, x):
            with torch.no_grad():  # stop-gradient EMA branch
                return F.normalize(self.enc0_t(x), dim=-1)

        def _stack0(self, window, target=False):
            # window [B, K, in_dim] -> per-tick L0 latents [B, K, latent0]
            b, k, d = window.shape
            flat = window.reshape(b * k, d)
            z = self.encode0_target(flat) if target else self.encode0(flat)
            return z.reshape(b, k, -1)

        def encode1(self, window):
            # L1 sits on top of DETACHED L0 features (probe/head geometry). Mean-pool over K
            # averages out the fast per-tick nuisance and keeps the slow regime (the distance-band
            # signal that survives pooling).
            z0s = self._stack0(window).detach()  # # VERIFY(torch): sg L0 -> L1
            return F.normalize(self.enc1(z0s.mean(dim=1)), dim=-1)

        def encode1_target(self, window):
            with torch.no_grad():
                z0s = self._stack0(window, target=True)
                return F.normalize(self.enc1_t(z0s.mean(dim=1)), dim=-1)

        # ---- predictors ------------------------------------------------------------------
        def predict0(self, z0, a_fine):
            # residual NONLINEAR fine predictor (small audio-phase steps); re-normalized -> bounded,
            # NOT spectrally clamped, NOT a contractive linear map.
            d = self.pred0(torch.cat([z0, a_fine], dim=-1))
            return F.normalize(z0 + d, dim=-1)

        def predict1(self, z1, a_coarse):
            # coarse abstract predictor: a PURE MLP (no residual skip), re-normalized onto the unit
            # sphere. Non-contractive; the arm action does NOT drive the exogenous regime, so this
            # mostly learns regime-persistence (correct while the slow r holds across windows).
            return F.normalize(self.pred1(torch.cat([z1, a_coarse], dim=-1)), dim=-1)

        # ---- EMA target update -----------------------------------------------------------
        def ema_update(self):
            with torch.no_grad():
                for tgt, onl in ((self.enc0_t, self.enc0), (self.enc1_t, self.enc1)):
                    for pt, po in zip(tgt.parameters(), onl.parameters()):
                        pt.mul_(self.tau).add_(
                            po, alpha=1.0 - self.tau
                        )  # # VERIFY(torch): momentum update

    return _HJepa()


def vicreg_terms(z, gamma=VICREG_GAMMA, eps=1e-4):
    """VICReg anti-collapse on an embedding batch z [batch, latent]. Returns (var_loss, cov_loss):
    variance hinge mean(relu(gamma - std_j)) keeps each dim's batch std up; covariance sum of
    squared OFF-diagonal covariances / latent decorrelates the dims. Blocks the constant latent."""
    import torch

    std = torch.sqrt(z.var(dim=0) + eps)  # [latent] per-dim std over the batch
    var_loss = torch.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    n = max(1, z.shape[0] - 1)
    cov = (zc.t() @ zc) / n
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / z.shape[1]
    return var_loss, cov_loss


class ArmTwoTimescaleWorld:
    """Arm-native two-timescale world. Observations are [0,1]^64 banded like encode_multimodal
    (visual|temperature|audio|distance, 16 each). A SLOW binary regime r lives on the K-clock
    (constant within a window, slow per-window Markov flip P_FLIP, EXOGENOUS to the arm) and is
    rendered into the DISTANCE band, corrupted per-FRAME by a fast within-window cosine nuisance
    that sums to ~0 over K ticks -> one frame is ambiguous about r, the K-window POOL reveals it.
    The arm action advances the AUDIO-band phase (ENGAGE +step / RELEASE -step /
    HOLD 0), a real learnable obs_t -> obs_{t+1} transition. torch imported lazily."""

    def __init__(
        self,
        gen,
        device,
        batch=DEFAULT_BATCH,
        k=K,
        band=BAND,
        n_fine=N_FINE,
        n_coarse=N_COARSE,
        amp=REGIME_AMP,
        audio_step=AUDIO_STEP,
        p_flip=P_FLIP,
        noise=0.02,
    ):
        import torch

        self.gen, self.device = gen, device
        self.batch, self.k, self.band = batch, k, band
        self.n_fine, self.n_coarse = n_fine, n_coarse
        self.amp, self.audio_step, self.p_flip, self.noise = amp, audio_step, p_flip, noise
        self.centers = torch.linspace(0.0, 1.0, band, device=device)  # [band] band centers
        self.sigma = 1.5 / max(1, band - 1)  # RBF width ~ band spacing
        # audio-band phase advance per arm action: ENGAGE +step, RELEASE -step, HOLD 0.
        # (Said "CLOSE / OPEN" until 2026-08-04 -- neither is in ARM_ACTIONS, so the comment named
        # commands the interpreter does not have; the ORDER it describes is correct.)
        self.eff_audio = torch.tensor([audio_step, -audio_step, 0.0], device=device)
        # running trajectory state for the contiguous coarse stream (next_coarse_batch)
        self.regime = (torch.rand(batch, generator=gen, device=device) < 0.5).float()  # r in {0,1}
        self.q = torch.rand(batch, generator=gen, device=device)  # audio phase in [0,1)
        self._cur = None  # cached current window/actions

    # ---- rendering -----------------------------------------------------------------------
    def _band(self, pos):
        # pos [n] -> RBF activations [n, band] in [0,1]
        import torch

        diff = self.centers.unsqueeze(0) - pos.unsqueeze(1)
        return torch.exp(-0.5 * (diff / self.sigma) ** 2)

    def _render(self, regime, q, phi, ctx, tick):
        # One frame [n, 64]: visual (const context) | temperature (per-window nuisance) | audio
        # (action phase) | distance (regime + fast within-window nuisance). ctx is INDEPENDENT of
        # phi so the temperature band never leaks the distance-nuisance phase (keeps a single frame
        # genuinely ambiguous about r -- no side channel).
        import torch

        n = regime.shape[0]
        vis_pos = torch.full((n,), 0.5, device=self.device)  # constant context band
        temp_pos = 0.25 + 0.5 * ctx  # per-window nuisance, NO regime info
        audio_pos = 0.5 + 0.4 * torch.sin(2.0 * math.pi * q)  # arm-action-driven fast band
        nuis = self.amp * torch.cos(2.0 * math.pi * (tick / self.k) + 2.0 * math.pi * phi)
        dist_pos = (0.25 + 0.5 * regime + nuis).clamp(0.0, 1.0)  # regime, per-frame corrupted
        x = torch.cat(
            [
                self._band(vis_pos),
                self._band(temp_pos),
                self._band(audio_pos),
                self._band(dist_pos),
            ],
            dim=1,
        )
        jitter = (
            torch.rand(x.shape, generator=self.gen, device=self.device) * 2.0 - 1.0
        ) * self.noise
        return (x + jitter).clamp(0.0, 1.0)

    def _gen_window(self, regime, q):
        # roll K fine ticks with random arm actions; the action advances the audio phase q, the
        # regime is held constant across the window. Returns (window[.,K,64], a_fine[.,K,3], q_end).
        import torch

        n = regime.shape[0]
        phi = torch.rand(
            n, generator=self.gen, device=self.device
        )  # fresh per-window nuisance phase
        ctx = torch.rand(
            n, generator=self.gen, device=self.device
        )  # independent per-window temperature nuisance
        obs_list, act_list = [], []
        for tick in range(self.k):
            obs_list.append(self._render(regime, q, phi, ctx, tick))
            a_idx = torch.randint(0, self.n_fine, (n,), generator=self.gen, device=self.device)
            a_oh = torch.nn.functional.one_hot(a_idx, self.n_fine).float()  # # VERIFY(torch)
            act_list.append(a_oh)
            q = torch.remainder(q + self.eff_audio[a_idx], 1.0)  # # VERIFY(torch): wrap phase
        return torch.stack(obs_list, dim=1), torch.stack(act_list, dim=1), q

    def _flip_regime(self, regime):
        # slow K-clock Markov flip: with prob p_flip the exogenous regime toggles at the boundary.
        import torch

        flip = (
            torch.rand(regime.shape[0], generator=self.gen, device=self.device) < self.p_flip
        ).float()
        return torch.remainder(regime + flip, 2.0)

    # ---- streams -------------------------------------------------------------------------
    def next_coarse_batch(self):
        """One coarse transition on the running trajectory: (window, a_fine, a_coarse_onehot,
        next_window). window/next_window [batch, K, 64]; a_fine [batch, K, 3]; a_coarse_onehot
        [batch, 3]. The coarse action is a representative arm action for the window transition; it
        does NOT drive the regime (the regime flips exogenously on the K-clock)."""
        import torch

        if self._cur is None:
            self._cur = self._gen_window(self.regime, self.q)
        cur_window, cur_afine, self.q = self._cur
        a_idx = torch.randint(
            0, self.n_coarse, (self.batch,), generator=self.gen, device=self.device
        )
        a_coarse = torch.nn.functional.one_hot(a_idx, self.n_coarse).float()
        self.regime = self._flip_regime(self.regime)  # exogenous slow flip
        next_window, next_afine, self.q = self._gen_window(self.regime, self.q)
        self._cur = (next_window, next_afine, self.q)
        return cur_window, cur_afine, a_coarse, next_window

    def sample_windows(self, n):
        """A fresh, BALANCED decode dataset: draw regime r uniformly in {0,1} and a random audio
        phase, render one K-window each with random arm actions. Returns
        (window [n, K, 64], regime [n] in {0.,1.})."""
        import torch

        regime = (torch.rand(n, generator=self.gen, device=self.device) < 0.5).float()
        q = torch.rand(n, generator=self.gen, device=self.device)
        window, _a, _q = self._gen_window(regime, q)
        return window, regime


def train_step(
    model,
    opt,
    batch,
    device="cpu",
    scaler=None,
    var_coef=DEFAULT_VAR_COEF,
    cov_coef=DEFAULT_COV_COEF,
):
    """One optimization step: invariance + VICReg at BOTH levels, then an EMA target update.
    Optional AMP on CUDA via `scaler`. Returns the REAL loss components (no fabricated numbers)."""
    import torch

    window, a_fine, a_coarse, next_window = batch
    b, k, d = window.shape
    is_cuda = str(device).startswith("cuda")
    dev_type = "cuda" if is_cuda else "cpu"
    use_amp = bool(scaler is not None and is_cuda)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev_type, enabled=use_amp):  # # VERIFY(torch): AMP autocast
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
        scaler.scale(loss).backward()  # # VERIFY(torch): scaled backward
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    model.ema_update()  # momentum targets AFTER the online step
    return {
        "loss": float(loss.detach().cpu()),
        "inv0": float(inv0.detach().cpu()),
        "inv1": float(inv1.detach().cpu()),
        "var": float((var0 + var1).detach().cpu()),
        "cov": float((cov0 + cov1).detach().cpu()),
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
    return torch.linalg.solve(a, xa.t() @ y)  # # VERIFY(torch): linalg.solve


def _ridge_pred(w, x):
    import torch

    x = x.detach().float().cpu()
    xa = torch.cat([x, torch.ones(x.shape[0], 1, device="cpu")], dim=1)
    return (xa @ w).reshape(-1)


def _mse(pred, y):
    y = y.detach().float().cpu().reshape(-1)  # pred is already CPU float from _ridge_pred
    return float(((pred - y) ** 2).mean())


def _acc(pred, y):
    # threshold the ridge readout at 0.5 -> binary regime accuracy
    y = y.detach().float().cpu().reshape(-1)
    return float(((pred > 0.5).float() == y).float().mean())


def evaluate_abstract_vs_flat(model, world, n=DECODE_N):
    """Measure the ARM abstraction claim with REAL numbers: fit a linear readout latent->r on a
    train split, evaluate held-out. Compares the learned pooled L1 abstraction z1 against
    (i) the flat single-frame z0 and (ii) raw windowed averaging of the observation."""
    import torch

    model.eval()
    with torch.no_grad():
        w_tr, r_tr = world.sample_windows(n)
        w_te, r_te = world.sample_windows(n)
        z1_tr, z1_te = model.encode1(w_tr), model.encode1(w_te)  # learned pooled abstraction
        z0_tr, z0_te = (
            model.encode0(w_tr[:, 0, :]),
            model.encode0(w_te[:, 0, :]),
        )  # flat single frame
        raw_tr, raw_te = w_tr.mean(dim=1), w_te.mean(dim=1)  # raw windowed averaging (64-dim)

        w_hier = _ridge_fit(z1_tr, r_tr)
        w_flat = _ridge_fit(z0_tr, r_tr)
        w_raw = _ridge_fit(raw_tr, r_tr)
        p_hier, p_flat, p_raw = (
            _ridge_pred(w_hier, z1_te),
            _ridge_pred(w_flat, z0_te),
            _ridge_pred(w_raw, raw_te),
        )
        mse_hier, mse_flat, mse_raw = _mse(p_hier, r_te), _mse(p_flat, r_te), _mse(p_raw, r_te)
        acc_hier, acc_flat, acc_raw = _acc(p_hier, r_te), _acc(p_flat, r_te), _acc(p_raw, r_te)

    finite = all(math.isfinite(v) for v in (mse_hier, mse_flat, mse_raw))
    return {
        "mse_hier": mse_hier,
        "mse_flat": mse_flat,
        "mse_raw": mse_raw,
        "acc_hier": acc_hier,
        "acc_flat": acc_flat,
        "acc_raw": acc_raw,
        "claim_beats_flat": mse_hier < mse_flat,  # primary: abstract z1 beats flat single-frame z0
        "beats_raw_pool": mse_hier
        < mse_raw,  # honest secondary (NOT expected to hold; see docstring)
        "finite": finite,
    }


def export_hjepa_arm(model, path):
    """Dump BOTH levels' weights as an snn_hjepa/1 JSON artifact for the zero-dep core server
    src/snn_data_lake/faculties/hjepa_serve.py. Layout matches reasoning/hjepa_torch.export_hjepa:
    nn.Linear weights are [out][in] via .tolist() (no transpose); enc0/enc1/pred0/pred1 in
    application order, plus EMA targets. Keys are arm-native (in_dim=64, arm action sets, regime
    map). Returns the written dict."""
    import torch.nn as nn

    def _seq(seq):
        return [
            [m.weight.detach().cpu().tolist(), m.bias.detach().cpu().tolist()]
            for m in seq
            if isinstance(m, nn.Linear)
        ]

    blob = {
        "format": HJEPA_FORMAT,
        "in_dim": IN_DIM,
        "K": K,
        "latent0": LATENT0,
        "latent1": LATENT1,
        "fine_actions": list(FINE_ACTIONS),
        "coarse_actions": list(COARSE_ACTIONS),
        "enc0": _seq(model.enc0),
        "enc1": _seq(model.enc1),  # online encoders (application order)
        "enc0_target": _seq(model.enc0_t),
        "enc1_target": _seq(model.enc1_t),
        "pred0": _seq(model.pred0),
        "pred1": _seq(model.pred1),  # fine (residual) / coarse (pure MLP)
        "tau": float(model.tau),
        "predictors": {"pred0": "residual_mlp_unit", "pred1": "mlp_unit"},
        "meta": {
            "trained_by": "hjepa_arm_torch",
            "unverified_on_hardware": True,
            "world": "arm_two_timescale",
            "regime": "slow_binary_distance_band",
            "regime_to_action": REGIME_TO_ACTION,
            "obs_bands": {
                "visual": [0, 16],
                "temperature": [16, 32],
                "audio": [32, 48],
                "distance": [48, 64],
            },
        },
    }
    with open(path, "w", encoding="ascii") as f:
        json.dump(blob, f)
    return blob


def verify_transfer_hjepa(model, path):
    """Bridge check: load the exported artifact into the zero-dep core HJepaServe and assert its
    encode0 / encode1 / predict1 match a CPU float64 clone of the torch model within BRIDGE_TOL.

    CRITICAL DEVICE RULE (this bit us before): the core is pure-Python (all-CPU). Under NORD_GPU,
    torch_device.enable() called torch.set_default_device('cuda'), so a bare torch factory lands on
    CUDA. Build a float64 CPU CLONE and create every probe tensor with EXPLICIT device='cpu' so the
    reference matches the CPU core instead of crashing on a device mismatch. If the core server is
    absent, print 'core-no-hjepa-serve' and skip. UNVERIFIED ON HARDWARE."""
    import torch

    try:
        from snn_data_lake.faculties.hjepa_serve import HJepaServe
    except ImportError:
        print(
            "[bridge] core-no-hjepa-serve (src/snn_data_lake/faculties/hjepa_serve.py absent) -- skipped"
        )
        return None

    # float64 CPU clone carrying the trained weights (state_dict copies values across devices).
    cpu_model = build_hjepa_arm()
    cpu_model.load_state_dict(model.state_dict())
    cpu_model = cpu_model.to(
        device="cpu", dtype=torch.float64
    )  # # VERIFY(torch): explicit CPU float64
    cpu_model.eval()

    serve = HJepaServe.load(path)
    gcpu = torch.Generator(device="cpu").manual_seed(
        1
    )  # explicit CPU generator (NOT the cuda default)
    max_err = 0.0
    with torch.no_grad():
        for _ in range(4):
            obs = torch.rand(
                IN_DIM, generator=gcpu, device="cpu", dtype=torch.float64
            )  # explicit CPU
            window = torch.rand(K, IN_DIM, generator=gcpu, device="cpu", dtype=torch.float64)
            a_idx = int(torch.randint(0, N_COARSE, (1,), generator=gcpu, device="cpu").item())
            a_oh = torch.zeros(N_COARSE, device="cpu", dtype=torch.float64)
            a_oh[a_idx] = 1.0

            z0_ref = cpu_model.encode0(obs.unsqueeze(0))[0]
            z1_ref = cpu_model.encode1(window.unsqueeze(0))[0]
            z1p_ref = cpu_model.predict1(z1_ref.unsqueeze(0), a_oh.unsqueeze(0))[0]

            z0_srv = torch.tensor(serve.encode0(obs.tolist()), device="cpu", dtype=torch.float64)
            z1_srv = torch.tensor(serve.encode1(window.tolist()), device="cpu", dtype=torch.float64)
            z1p_srv = torch.tensor(
                serve.predict1(z1_ref.tolist(), a_idx), device="cpu", dtype=torch.float64
            )

            for ref, srv in ((z0_ref, z0_srv), (z1_ref, z1_srv), (z1p_ref, z1p_srv)):
                max_err = max(max_err, float((ref - srv).abs().max()))

    ok = max_err < BRIDGE_TOL
    print(
        f"[bridge] core HJepaServe vs torch CPU-float64 clone: max_abs_err={max_err:.2e} "
        f"tol={BRIDGE_TOL:.0e} -> {'MATCH' if ok else 'MISMATCH'}"
    )
    assert ok, f"snn_hjepa/1 transfer mismatch: max_abs_err={max_err:.2e} >= {BRIDGE_TOL:.0e}"
    return max_err


def main():
    # --- no-torch guard: import lazily so this file is safe to run with no torch installed ---
    try:
        import torch
    except ImportError:
        print("=" * 72)
        print(
            "ARM-NATIVE HIERARCHICAL H-JEPA TORCH TRAINER (not in run_selfchecks) -- torch NOT installed"
        )
        print("=" * 72)
        print(
            "A 2-level NONLINEAR H-JEPA (LeCun 2022 H-JEPA / V-JEPA 2024 / VICReg 2022) trained on"
        )
        print("the ARM's observation ([0,1]^64 banded) + action ({ENGAGE,RELEASE,HOLD})")
        print("space, so the exported snn_hjepa/1 artifact can be SERVED in the live arm brain.")
        print("Needs a GPU + torch to train and measure -- nothing to run without torch.")
        print("needs torch: pip install '.[gpu]'")
        print(
            "run on the RTX 5070: NORD_GPU=1 PYTHONPATH=src python -m "
            "snn_data_lake.reasoning.hjepa_arm_torch"
        )
        print("")
        print("What the GPU run MEASURES ([GPU-VERIFY]): does the learned pooled abstraction z1")
        print("decode the SLOW arm regime r better than (i) the flat single-frame z0 and (ii) raw")
        print("windowed averaging? A fast per-tick nuisance makes ONE frame ambiguous but cancels")
        print(
            "over the K-window pool, so z1 is EXPECTED to beat flat z0 by construction. Honestly,"
        )
        print(
            "z1 is NOT expected to clearly beat raw windowed averaging: unlike the XOR world, the"
        )
        print("arm regime is a single band-LINEAR factor, so raw pooling is a strong baseline.")
        print("Exiting cleanly (0).")
        return 0

    from snn_data_lake.infra.device import (
        enable,
        resolve_device,
    )  # shared cpu/cuda selector

    device = enable(resolve_device())  # # VERIFY(torch): NORD_GPU -> cuda (set_default_device)
    is_cuda = str(device).startswith("cuda")
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)  # # VERIFY(torch): device-side generator

    model = build_hjepa_arm().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)  # # VERIFY(torch): AMP (no-op on cpu)
    world = ArmTwoTimescaleWorld(gen, device)

    last = {}
    for _ in range(DEFAULT_STEPS):
        last = train_step(model, opt, world.next_coarse_batch(), device=device, scaler=scaler)

    res = evaluate_abstract_vs_flat(model, world)

    export_path = os.path.join(tempfile.gettempdir(), "snn_hjepa_arm.json")
    export_hjepa_arm(model, export_path)

    print("=" * 72)
    print("ARM-NATIVE HIERARCHICAL H-JEPA TORCH TRAINER -- 2-level nonlinear H-JEPA")
    print("=" * 72)
    print(f"backend      : torch {torch.__version__}  device={device}  amp={is_cuda}")
    print(
        f"world        : arm two-timescale; slow regime r in distance band (K-clock, exogenous); "
        f"actions {list(ARM_ACTIONS)}"
    )
    print(
        f"encoders     : L0 MLP [{IN_DIM}->{HIDDEN0}->{HIDDEN0}->{LATENT0}]  "
        f"L1 head MLP [{LATENT0}->{HIDDEN1}->{HIDDEN1}->{LATENT1}] (both unit-normalized, 2 hidden)"
    )
    print(
        "predictors   : L0 residual-MLP (fine audio phase), L1 pure-MLP (coarse); unit-sphere bounded"
    )
    print(
        f"train ({DEFAULT_STEPS} steps): loss={last['loss']:.4f} inv0={last['inv0']:.4f} "
        f"inv1={last['inv1']:.4f} var={last['var']:.4f} cov={last['cov']:.4f}   (real, from this run)"
    )
    print("-" * 72)
    print("[GPU-VERIFY] does the abstract pooled z1 decode the SLOW regime r better than flat/raw?")
    print(
        f"[GPU-VERIFY]   learned L1 abstract (K-pool)  regime decode  MSE={res['mse_hier']:.4f}  acc={res['acc_hier']:.3f}"
    )
    print(
        f"[GPU-VERIFY]   flat single-frame z0          regime decode  MSE={res['mse_flat']:.4f}  acc={res['acc_flat']:.3f}"
    )
    print(
        f"[GPU-VERIFY]   raw windowed averaging (64d)  regime decode  MSE={res['mse_raw']:.4f}  acc={res['acc_raw']:.3f}"
    )
    print(
        f"[GPU-VERIFY]   PRIMARY claim -- abstract z1 beats flat single-frame z0: {res['claim_beats_flat']}"
    )
    print(
        f"[GPU-VERIFY]   (honest secondary) z1 also beats raw windowed averaging: {res['beats_raw_pool']}"
    )
    print(
        "[GPU-VERIFY]   NOTE: z1>flat is expected by construction (nuisance cancels over the pool);"
    )
    print("[GPU-VERIFY]   z1>raw-average is NOT expected -- the arm regime is band-LINEAR, so raw")
    print(
        "[GPU-VERIFY]   pooling is a strong baseline (no XOR floor here). Not rigged -- report as-is."
    )
    if not res["claim_beats_flat"]:
        # P8 (d): report the null, and report it where a reader cannot miss it. This module has NO
        # assert on its own primary claim and is not in run_selfchecks, so without this line it
        # prints "False" once in a table and exits 0 -- indistinguishable from a pass.
        print("[GPU-VERIFY]   *** NULL: the PRIMARY claim FAILED on this run. The learned pooled")
        print(
            f"[GPU-VERIFY]   *** abstraction (MSE {res['mse_hier']:.4f}) did NOT beat the flat single"
        )
        print(f"[GPU-VERIFY]   *** frame (MSE {res['mse_flat']:.4f}), and raw windowed averaging")
        print(
            f"[GPU-VERIFY]   *** (MSE {res['mse_raw']:.4f}) beat both. The pooling CAPACITY is present"
        )
        print(
            "[GPU-VERIFY]   *** by construction; the SSL objective did not realise it. Not a crash,"
        )
        print("[GPU-VERIFY]   *** not a pass -- a measured null. See the docstring.")
    print("-" * 72)
    verify_transfer_hjepa(model, export_path)
    print("-" * 72)
    print(
        f"exported     : {HJEPA_FORMAT} -> {export_path} "
        f"(in_dim=64, arm actions, enc0/enc1/pred0/pred1 + EMA targets; for faculties/hjepa_serve.py)"
    )
    print(
        "NOTE: the pooling capacity to beat a flat single frame is present by construction; whether"
    )
    print(
        "the SSL objective (which never sees r) REALIZES it is what the [GPU-VERIFY] block measures."
    )
    print(
        "Measured 2026-08-04 on torch-CPU at the shipped defaults: it did NOT -- see the docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
