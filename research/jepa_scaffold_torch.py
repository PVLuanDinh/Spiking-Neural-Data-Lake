"""Deeper action-conditioned latent JEPA (PyTorch trainer) -> exports the zero-dep core.

**UNVERIFIED ON HARDWARE -- best-effort trainer.** There is no GPU/torch in the
environment this file was written in, so NOTHING below was executed end-to-end here.
Every torch-specific behaviour is flagged with a ``# VERIFY(torch):`` comment. No loss
or accuracy numbers are hard-coded anywhere -- the banner prints only values a real run
computed. Intended target: the user's RTX 5070 (``NORD_GPU=1 python research/jepa_scaffold_torch.py``).

This is the SCALE tier for the zero-dep core cerebellum/jepa_world_model.py (v3.11 shipped
a minimal *linear* JEPA there). It trains a DEEPER latent JEPA in PyTorch -- a hidden-layer
MLP encoder instead of a single linear read-in -- and bridges the GPU-trained weights into a
NEW zero-dep core format ``snn_jepa/2`` so the pure-stdlib core serves the deeper model WITH
NO TORCH. This mirrors how snn_data_lake.training.mnist exports a torch MLP into the zero-dep
``trained_brain`` (export_brain/verify_transfer): train on GPU, serve on CPU with no deps.
It lives in research/ (ruff-excluded, NOT in CI); it is a trainer, not a shipped core module.

Method (action-conditioned latent JEPA -- I-JEPA / V-JEPA in a robot-action setting):
  * ENCODER   f_theta:  x in R^in_dim -> h in R^hidden -> z in R^latent_dim, a two-layer MLP
                        z = unit(tanh(W2 @ tanh(W1 @ x + b1) + b2)), final latent L2-normalized
                        to the unit sphere (matches the core encoder geometry).
  * PREDICTOR g_phi:    residual, action-conditioned:  z_hat' = z + Wz.z + Wa.a + b
                        (a = one-hot action). Predict the NEXT latent from (z_t, a_t).
  * TARGET    f_xi:     an EMA (momentum tau) copy of the FULL encoder, no gradient -- the
                        prediction target is sg(f_xi(x_{t+1})) (stop-gradient), so the model
                        cannot collapse both branches to a constant.
  * LOSS:     invariance  ||g_phi(f_theta(x_t), a_t) - sg(f_xi(x_{t+1}))||^2
              + VICReg anti-collapse on the online embeddings (variance hinge gamma=1 +
              off-diagonal covariance), which is what blocks the trivial constant latent.
              Standard autograd -- no surrogate gradient is needed (this is a rate/latent model,
              not a spiking one), unlike snn_data_lake.training.mnist.

The deeper encoder is the whole point of ``snn_jepa/2``: the core ``snn_jepa/1`` format serves a
single linear read-in ``We[latent][in]``; ``snn_jepa/2`` serves a LIST of encoder layers
(``enc_layers``), so a hidden-layer MLP round-trips faithfully (the core applies the layers in
order, exactly the way ``trained_brain`` already serves a multi-layer torch MLP). The core
``JepaWorldModel.load_weights`` is being extended in parallel to read this format.

Persisted core format ('snn_jepa/2'), shapes with in_dim=64, hidden=H, latent_dim=16, 3 actions:
  enc_layers = [[W1[H][64], b1[H]], [W2[16][H], b2[16]]]   online encoder, APPLICATION order
  tgt_layers = [[.......][H], [.......][16]]               EMA target encoder, same shapes
  Wz [16][16]   Wa [16][3]  b [16]                         residual action-conditioned predictor
  actions = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")

torch is imported LAZILY inside the functions/main (never at module top), so this file imports
cleanly with no torch installed and ``python research/jepa_scaffold_torch.py`` exits 0 with a
guidance banner. The nn.Module is defined inside build_model() for the same reason.

Real corpus (optional, documented not required): SyntheticStream below is a self-contained
action->next-state generator so the trainer runs standalone. To train on a real robot-arm log
instead, replace SyntheticStream.next_batch() with an iterator over (x_t, a_onehot, x_next)
tensors drawn from recorded (observation, action, next-observation) triples in [0,1]^in_dim --
nothing else in the loss/export path changes.

References:
  I-JEPA  -- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
             Predictive Architecture", CVPR 2023.
  V-JEPA  -- Bardes et al., "V-JEPA: Video Joint-Embedding Predictive Architectures", 2024.
  VICReg  -- Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization
             for Self-Supervised Learning", ICLR 2022.
  JEPA    -- LeCun, "A Path Towards Autonomous Machine Intelligence", 2022 (the JEPA proposal).

Run (own venv, torch installed):  NORD_GPU=1 .venv-arm/Scripts/python research/jepa_scaffold_torch.py
Run (no torch):                   python research/jepa_scaffold_torch.py  -> guidance, exit 0
"""
import json
import math

IN_DIM = 64
LATENT_DIM = 16
DEFAULT_HIDDEN = 64                 # H: hidden width of the MLP encoder (the snn_jepa/2 scale knob)
ACTIONS = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")
N_ACT = len(ACTIONS)

JEPA_FORMAT = "snn_jepa/2"
DEFAULT_TAU = 0.99                  # EMA target momentum: xi <- tau*xi + (1-tau)*theta
DEFAULT_VAR_COEF = 1.0             # VICReg variance-hinge weight
DEFAULT_COV_COEF = 0.04            # VICReg covariance weight (standard VICReg default scale)
VICREG_GAMMA = 1.0                 # target per-dim std for the variance hinge

DEFAULT_STEPS = 300               # a few optimization steps -- this is a scaffold, not a full run
DEFAULT_BATCH = 128
K_FACTORS = 4                      # hidden factors in the synthetic world (multi-factor by construction)


def build_model(in_dim=IN_DIM, latent_dim=LATENT_DIM, hidden=DEFAULT_HIDDEN,
                n_act=N_ACT, tau=DEFAULT_TAU):
    """Build the deeper action-conditioned latent JEPA. torch/nn imported lazily; the nn.Module
    is defined INSIDE here so importing this file never requires torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _Jepa(nn.Module):
        def __init__(self):
            super().__init__()
            # online encoder: MLP read-in x -> h -> z  (exports to enc_layers, application order)
            self.enc = nn.ModuleList([nn.Linear(in_dim, hidden), nn.Linear(hidden, latent_dim)])
            # residual action-conditioned predictor: z_hat' = z + Wz.z + Wa.a + b
            self.pred_z = nn.Linear(latent_dim, latent_dim, bias=False)     # Wz
            self.pred_a = nn.Linear(n_act, latent_dim, bias=True)           # Wa (+ shared bias b)
            # EMA target encoder: a frozen copy of the FULL encoder, updated by ema_update() only.
            self.tgt = nn.ModuleList([nn.Linear(in_dim, hidden), nn.Linear(hidden, latent_dim)])
            for pt, po in zip(self.tgt.parameters(), self.enc.parameters()):
                pt.data.copy_(po.data)                                      # start = online encoder
                pt.requires_grad_(False)                                    # # VERIFY(torch): no-grad target
            self.tau = float(tau)

        @staticmethod
        def _run_layers(layers, x):
            # z = unit(tanh(W_last @ tanh(... tanh(W1 @ x + b1) ...) + b_last)); tanh EVERY layer,
            # then L2-normalize the FINAL latent onto the unit sphere (matches the core geometry).
            h = x
            for lin in layers:
                h = torch.tanh(lin(h))                                      # # VERIFY(torch): tanh per layer
            return F.normalize(h, dim=-1)                                   # unit(.) == core _mathutils.unit

        def encode(self, x):
            return self._run_layers(self.enc, x)

        def encode_target(self, x):
            # stop-gradient target branch (sg): no autograd through the EMA encoder.
            with torch.no_grad():
                return self._run_layers(self.tgt, x)

        def predict(self, z, a_onehot):
            # residual predictor -- predicts the CHANGE and adds it back to z.
            return z + self.pred_z(z) + self.pred_a(a_onehot)

        def ema_update(self):
            # xi <- tau*xi + (1-tau)*theta, momentum-encoder update (I-JEPA / BYOL style).
            with torch.no_grad():
                for pt, po in zip(self.tgt.parameters(), self.enc.parameters()):
                    pt.mul_(self.tau).add_(po, alpha=1.0 - self.tau)        # # VERIFY(torch)

    return _Jepa()


def vicreg_terms(z, gamma=VICREG_GAMMA, eps=1e-4):
    """VICReg anti-collapse on an embedding batch z [batch, latent].

    Returns (var_loss, cov_loss):
      * variance hinge: push each latent dim's std up to `gamma` so the batch does not
        collapse to a point -- mean(relu(gamma - std_j)).
      * covariance: decorrelate latent dims -- sum of squared OFF-diagonal covariances / dim.
    """
    import torch

    std = torch.sqrt(z.var(dim=0) + eps)                 # [latent] per-dim std over the batch
    var_loss = torch.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    n = max(1, z.shape[0] - 1)
    cov = (zc.t() @ zc) / n                               # [latent, latent] covariance matrix
    off = cov - torch.diag(torch.diagonal(cov))          # zero the diagonal (variance handled above)
    cov_loss = off.pow(2).sum() / z.shape[1]
    return var_loss, cov_loss


class SyntheticStream:
    """A self-contained action-conditioned (x_t, a_onehot, x_next) generator.

    A hidden multi-factor latent u in [0,1]^K evolves under N_ACT discrete actions (each action
    drifts the factors in a fixed direction), and each observation is a BANDED [0,1]^in_dim RBF
    rendering of u (one contiguous band of RBF population activations per factor). x_next is the
    rendering of u AFTER the action, so the residual predictor sees a genuine action->next-state
    signal -- exactly the structure the core's own synthetic world uses, lifted to torch tensors.
    Deterministic given `gen`. torch is imported lazily so the module stays torch-free at import.
    """

    def __init__(self, gen, device, in_dim=IN_DIM, n_act=N_ACT, k_factors=K_FACTORS,
                 batch=DEFAULT_BATCH, step=0.15, noise=0.02):
        import torch

        assert in_dim % k_factors == 0, "in_dim must divide evenly into k_factors bands"
        self.gen, self.device = gen, device
        self.in_dim, self.n_act, self.k = in_dim, n_act, k_factors
        self.batch, self.noise = batch, noise
        band = in_dim // k_factors
        self.centers = torch.linspace(0.0, 1.0, band, device=device)               # [band] in [0,1]
        self.sigma = 1.5 / max(1, band - 1)                                          # RBF width ~ band spacing
        # per-action drift on each factor, drawn ONCE (uniform in [-step, step]).  # VERIFY(torch)
        self.effects = (torch.rand(n_act, k_factors, generator=gen, device=device) * 2.0 - 1.0) * step
        self.u = torch.rand(batch, k_factors, generator=gen, device=device)         # hidden latent state

    def _render(self, u):
        import torch

        acts = []
        for k in range(self.k):
            diff = self.centers.unsqueeze(0) - u[:, k:k + 1]                          # [batch, band]
            acts.append(torch.exp(-0.5 * (diff / self.sigma) ** 2))
        x = torch.cat(acts, dim=1)                                                    # [batch, in_dim]
        jitter = (torch.rand(x.shape, generator=self.gen, device=self.device) * 2.0 - 1.0) * self.noise
        return (x + jitter).clamp(0.0, 1.0)

    def next_batch(self):
        import torch

        a_idx = torch.randint(0, self.n_act, (self.batch,), generator=self.gen, device=self.device)
        a_onehot = torch.nn.functional.one_hot(a_idx, self.n_act).float()
        x_t = self._render(self.u)
        u_next = (self.u + self.effects[a_idx]).clamp(0.0, 1.0)
        x_next = self._render(u_next)
        self.u = u_next.detach()                                                     # carry trajectory forward
        return x_t, a_onehot, x_next


def train_step(model, opt, batch, device="cpu", scaler=None,
               var_coef=DEFAULT_VAR_COEF, cov_coef=DEFAULT_COV_COEF):
    """One optimization step: invariance loss + VICReg, then an EMA target update. Optional AMP
    on CUDA via `scaler`. Returns the REAL loss components from this step (no fabricated numbers)."""
    import torch

    x_t, a_onehot, x_next = batch
    is_cuda = str(device).startswith("cuda")
    dev_type = "cuda" if is_cuda else "cpu"
    use_amp = bool(scaler is not None and is_cuda)                                    # AMP optional
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev_type, enabled=use_amp):                      # # VERIFY(torch): AMP
        z_t = model.encode(x_t)                          # online embedding of x_t
        pred = model.predict(z_t, a_onehot)              # predicted next latent
        target = model.encode_target(x_next)             # sg(EMA target) -- already no-grad
        inv_loss = (pred - target).pow(2).sum(dim=1).mean()   # invariance (JEPA prediction loss)
        var_loss, cov_loss = vicreg_terms(z_t)           # anti-collapse on online embeddings
        loss = inv_loss + var_coef * var_loss + cov_coef * cov_loss
    if use_amp:
        scaler.scale(loss).backward()                    # # VERIFY(torch): scaled backward
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    model.ema_update()                                   # momentum target AFTER the online step
    return {
        "loss": float(loss.detach().cpu()),
        "inv": float(inv_loss.detach().cpu()),
        "var": float(var_loss.detach().cpu()),
        "cov": float(cov_loss.detach().cpu()),
    }


def save_checkpoint(model, opt, path):
    """Save a torch training checkpoint (model + optimizer state)."""
    import torch

    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict() if opt is not None else None}, path)         # # VERIFY(torch)


def load_checkpoint(model, opt, path, map_location="cpu"):
    """Load a torch training checkpoint into `model` (and `opt` if given). Returns `model`."""
    import torch

    ckpt = torch.load(path, map_location=map_location)                               # # VERIFY(torch)
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    return model


def export_jepa(model, path):
    """THE BRIDGE: write the core ``snn_jepa/2`` JSON so the zero-dep core serves these latents.

    nn.Linear.weight is [out][in], so enc[0].weight is [hidden][in] and enc[1].weight is
    [latent][hidden] -- a direct ``.tolist()`` with NO transpose, exactly like trained_brain's
    export. Layers are emitted in APPLICATION order (input->hidden first, hidden->latent last).
    EVERY shape is ASSERTED before writing. Returns the written dict.
    """
    def _layers(module_list):
        return [[lin.weight.detach().cpu().tolist(), lin.bias.detach().cpu().tolist()]
                for lin in module_list]

    enc_layers = _layers(model.enc)                      # [[W1[H][64], b1[H]], [W2[16][H], b2[16]]]
    tgt_layers = _layers(model.tgt)                      # same shapes, EMA target encoder
    Wz = model.pred_z.weight.detach().cpu().tolist()     # [latent][latent]
    Wa = model.pred_a.weight.detach().cpu().tolist()     # [latent][n_act]
    b = model.pred_a.bias.detach().cpu().tolist()        # [latent]

    hidden = len(enc_layers[0][1])                       # H = out-features of the first encoder layer

    def _assert_encoder(name, layers):
        assert len(layers) == 2, f"{name} must be a 2-layer MLP encoder, got {len(layers)} layers"
        (W1, b1), (W2, b2) = layers
        assert len(W1) == hidden and all(len(r) == IN_DIM for r in W1), \
            f"{name}[0] W must be [hidden={hidden}][in={IN_DIM}]"
        assert len(b1) == hidden, f"{name}[0] b must be [hidden={hidden}]"
        assert len(W2) == LATENT_DIM and all(len(r) == hidden for r in W2), \
            f"{name}[1] W must be [latent={LATENT_DIM}][hidden={hidden}]"
        assert len(b2) == LATENT_DIM, f"{name}[1] b must be [latent={LATENT_DIM}]"

    _assert_encoder("enc_layers", enc_layers)
    _assert_encoder("tgt_layers", tgt_layers)
    assert len(Wz) == LATENT_DIM and all(len(r) == LATENT_DIM for r in Wz), "Wz must be [latent][latent]"
    assert len(Wa) == LATENT_DIM and all(len(r) == N_ACT for r in Wa), "Wa must be [latent][n_act]"
    assert len(b) == LATENT_DIM, "b must be [latent]"

    blob = {
        "format": JEPA_FORMAT,
        "in_dim": IN_DIM,
        "latent_dim": LATENT_DIM,
        "hidden": hidden,
        "actions": list(ACTIONS),
        "enc_layers": enc_layers,                        # online encoder, application order
        "tgt_layers": tgt_layers,                        # EMA target encoder, same shapes
        "Wz": Wz, "Wa": Wa, "b": b,                      # residual predictor (as core snn_jepa/1)
        "tau": float(model.tau),
        "predictor": "residual",                         # z' = z + Wz.z + Wa.a + b
        "meta": {"trained_by": "jepa_scaffold_torch", "unverified_on_hardware": True},
    }
    with open(path, "w", encoding="ascii") as f:
        json.dump(blob, f)
    return blob


def verify_transfer(path, model, n_probe=5, tol=1e-4):
    """Export->core bridge check: load the zero-dep core from `path` and confirm its forward
    encode/predict matches the torch model's forward on a few probe inputs (within `tol`).

    Runs entirely on CPU (no CUDA needed). To make the comparison faithful across dtypes, a
    float64 CPU clone of the trained torch model is used as the reference (torch trains in
    float32; the core computes in Python float64). If torch is absent, skips gracefully.
    Depends on the parallel core extension that teaches JepaWorldModel.load_weights the
    ``snn_jepa/2`` format -- if the installed core predates it, this reports that and returns.
    """
    try:
        import torch
    except ImportError:
        print("verify_transfer: skipped (torch not installed)")
        return None

    from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel

    # float64 CPU reference built from the LIVE trained weights (via state_dict) -- verifies both
    # that export captured the live weights AND that the core reads them back correctly.
    in_dim = model.enc[0].in_features
    hidden = model.enc[0].out_features
    latent = model.enc[-1].out_features
    n_act = model.pred_a.in_features
    ref = build_model(in_dim, latent, hidden, n_act, model.tau)
    ref.load_state_dict(model.state_dict())
    ref = ref.double().to("cpu")                                                     # # VERIFY(torch)
    ref.eval()

    core = JepaWorldModel(ACTIONS, in_dim=in_dim, latent_dim=latent)
    try:
        core.load_weights(path)                          # requires the parallel snn_jepa/2 core support
    except ValueError as e:
        print(f"verify_transfer: core rejected {JEPA_FORMAT} ({e}); "
              "ensure the parallel snn_jepa/2 core extension is installed.")
        return {"ok": False, "reason": "core-no-v2"}

    torch.manual_seed(123)
    # the bridge compares the CPU float64 ref vs the zero-dep (CPU) core, so the probe tensors MUST be on
    # CPU explicitly -- with NORD_GPU the default device is cuda, which would mismatch the CPU ref.
    probes = torch.rand(n_probe, in_dim, dtype=torch.float64, device="cpu")           # probe inputs in [0,1]
    max_err = 0.0
    with torch.no_grad():
        z_ref = ref.encode(probes)                       # [n_probe, latent] float64
        for i in range(n_probe):
            x = [float(v) for v in probes[i]]
            z_core = core.encode(x)                       # list[latent] from the zero-dep core
            for a, b in zip(z_core, z_ref[i].tolist()):
                assert math.isfinite(a) and math.isfinite(b), "non-finite latent in bridge check"
                max_err = max(max_err, abs(a - b))
            # predict: compare core.predict(z, action) to the torch residual predictor forward.
            for aj, action in enumerate(ACTIONS):
                a_onehot = torch.zeros(1, n_act, dtype=torch.float64, device="cpu")
                a_onehot[0, aj] = 1.0
                p_ref = ref.predict(z_ref[i:i + 1], a_onehot)[0].tolist()
                p_core = core.predict(z_core, action)
                for a, b in zip(p_core, p_ref):
                    max_err = max(max_err, abs(a - b))

    assert max_err < tol, (
        f"export->core bridge mismatch: max |core - torch| = {max_err:.3e} exceeds tol {tol:.1e}"
    )
    return {"ok": True, "max_err": max_err, "n_probe": n_probe}


def main():
    # --- no-torch guard: import lazily so this file is safe to run with no torch installed ---
    try:
        import torch
    except ImportError:
        print("=" * 66)
        print("JEPA TORCH TRAINER (research/, not CI) -- torch is NOT installed")
        print("=" * 66)
        print("This is the torch SCALE tier for the zero-dep core")
        print("cerebellum/jepa_world_model.py (deeper MLP encoder -> snn_jepa/2).")
        print("Without torch there is nothing to train or export; the zero-dep")
        print("core is what CI covers.")
        print("needs torch: pip install '.[gpu]'")
        print("Exiting cleanly (0).")
        return 0

    import os
    import tempfile

    from torch_device import enable, resolve_device      # reuse the shared cpu/cuda selector

    device = enable(resolve_device())                    # # VERIFY(torch): NORD_GPU -> cuda, else cpu
    is_cuda = str(device).startswith("cuda")
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)  # # VERIFY(torch): device-side generator

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)  # # VERIFY(torch): AMP optional (no-op on cpu)
    stream = SyntheticStream(gen, device)

    last = {}
    for _ in range(DEFAULT_STEPS):
        batch = stream.next_batch()
        last = train_step(model, opt, batch, device=device, scaler=scaler)

    # checkpoint save/load round-trip (real: reloaded weights must export byte-identically) ----
    ckpt_path = os.path.join(tempfile.gettempdir(), "snn_jepa2_ckpt.pt")
    save_checkpoint(model, opt, ckpt_path)
    reloaded = load_checkpoint(build_model().to(device), None, ckpt_path,
                               map_location=device)

    export_path = os.path.join(tempfile.gettempdir(), "snn_jepa2_scaffold.json")
    blob = export_jepa(model, export_path)
    reloaded_blob = export_jepa(reloaded,
                                os.path.join(tempfile.gettempdir(), "snn_jepa2_reload.json"))
    ckpt_ok = (reloaded_blob["enc_layers"] == blob["enc_layers"]
               and reloaded_blob["Wz"] == blob["Wz"] and reloaded_blob["b"] == blob["b"])

    verify = verify_transfer(export_path, model)          # export->core bridge check (CPU)

    W1, b1 = blob["enc_layers"][0]
    W2, b2 = blob["enc_layers"][1]
    Wz, Wa, b = blob["Wz"], blob["Wa"], blob["b"]
    print("=" * 66)
    print("JEPA TORCH TRAINER -- deeper action-conditioned latent JEPA, UNVERIFIED")
    print("=" * 66)
    print(f"backend      : torch {torch.__version__}  device={device}  amp={is_cuda}")
    print(f"actions      : {list(ACTIONS)}")
    print(f"encoder MLP  : enc_layers[0] W [{len(W1)}][{len(W1[0])}] b [{len(b1)}]  ->  "
          f"enc_layers[1] W [{len(W2)}][{len(W2[0])}] b [{len(b2)}]  (z = unit(tanh(.)))")
    print(f"target MLP   : tgt_layers same shapes (EMA tau={blob['tau']})")
    print(f"predictor    : residual  Wz [{len(Wz)}][{len(Wz[0])}]  "
          f"Wa [{len(Wa)}][{len(Wa[0])}]  b [{len(b)}]")
    print(f"train ({DEFAULT_STEPS} steps): loss={last['loss']:.4f} inv={last['inv']:.4f} "
          f"var={last['var']:.4f} cov={last['cov']:.4f}   (real, from this run)")
    print(f"checkpoint   : save/load round-trip byte-identical export: {ckpt_ok}")
    if verify is None:
        print("bridge check : skipped (no torch)")
    elif verify.get("ok"):
        print(f"bridge check : core serves {JEPA_FORMAT} == torch forward "
              f"(max |diff| {verify['max_err']:.2e} over {verify['n_probe']} probes)")
    else:
        print(f"bridge check : NOT confirmed ({verify.get('reason')})")
    print(f"exported     : {JEPA_FORMAT} -> {export_path}")
    print("=" * 66)
    print("NOTE: trainer only -- shapes/format verified, dynamics UNVERIFIED on hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
