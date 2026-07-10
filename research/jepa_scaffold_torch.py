"""Latent action-conditioned JEPA (torch SCALE scaffold) -> exports the zero-dep core.

**UNVERIFIED ON HARDWARE -- best-effort scaffold.** There is no GPU/torch in the
environment this file was written in, so NOTHING below was executed end-to-end here.
Every torch-specific behaviour is flagged with a ``# VERIFY(torch):`` comment. No loss
or accuracy numbers are hard-coded anywhere -- the banner prints only values computed
by a real run.

This is the SCALE tier for the zero-dep core cerebellum/jepa_world_model.py: it trains a
faithful I-JEPA / V-JEPA-style latent JEPA in PyTorch (autograd, no surrogate needed) and
bridges the learned weights into the core ``snn_jepa/1`` JSON format so the pure-stdlib
core can serve the torch-trained latents WITH NO TORCH -- exactly mirroring how
snn_data_lake.training.mnist exports into cortex/trained_brain (export_brain/verify_transfer).
It lives in research/ (ruff-excluded, NOT in CI); it is a scaffold, not a shipped core module.

Method (action-conditioned latent JEPA):
  * ENCODER   f_theta:  x in R^in_dim -> z in R^latent_dim  (linear read-in; see NOTE below).
  * PREDICTOR g_phi:    residual, action-conditioned:  z_hat' = z + Wz.z + Wa.a + b
                        (a = one-hot action). Predict the NEXT latent from (z_t, a_t).
  * TARGET    f_xi:     an EMA (momentum tau) copy of the encoder, no gradient -- the
                        prediction target is sg(f_xi(x_{t+1})) (stop-gradient), so the model
                        cannot cheat by collapsing both branches.
  * LOSS:     invariance  ||g_phi(f_theta(x_t), a_t) - sg(f_xi(x_{t+1}))||^2
              + VICReg anti-collapse on the online embeddings (variance hinge + covariance),
              which is what stops the trivial constant-latent solution a JEPA is prone to.

NOTE on the encoder shape (the one real design tension, flagged honestly):
  The core ``snn_jepa/1`` format serves a LINEAR read-in ``We[latent_dim][in_dim]`` + ``be``,
  so the export is byte-faithful only when the torch encoder is itself linear (this is the
  same "torch dynamics == core dynamics" invariant that makes the mnist export faithful).
  A hidden-layer MLP encoder is the natural SCALE extension, but it cannot round-trip into a
  single ``We[16][64]`` without loss -- it would need the core to grow a hidden layer too
  (the way trained_brain already supports multi-layer brains). Kept linear here for a faithful
  bridge; the MLP is left as a flagged scale knob rather than a lossy export.  # VERIFY(torch)

Persisted core format ('snn_jepa/1'), shapes with in_dim=64, latent_dim=16, 3 actions:
  We [16][64]   be [16]              online encoder (linear read-in)
  Wz [16][16]   Wa [16][3]  b [16]   residual action-conditioned latent predictor
  actions = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")

torch is imported LAZILY inside the functions/main (never at module top), so this file
imports cleanly with no torch installed and ``python research/jepa_scaffold_torch.py`` exits
0 with a guidance banner. The nn.Module is defined inside build_model() for the same reason.

References:
  I-JEPA  -- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
             Predictive Architecture", CVPR 2023.
  V-JEPA  -- Bardes et al., "V-JEPA: Video Joint-Embedding Predictive Architectures", 2024.
  VICReg  -- Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization
             for Self-Supervised Learning", ICLR 2022.
  JEPA    -- LeCun, "A Path Towards Autonomous Machine Intelligence", 2022 (the JEPA proposal).

Run (own venv, torch installed):  .venv-arm/Scripts/python research/jepa_scaffold_torch.py
Run (no torch):                   python research/jepa_scaffold_torch.py  -> guidance, exit 0
"""
import json

IN_DIM = 64
LATENT_DIM = 16
ACTIONS = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")
N_ACT = len(ACTIONS)

JEPA_FORMAT = "snn_jepa/1"
DEFAULT_TAU = 0.99          # EMA target momentum: xi <- tau*xi + (1-tau)*theta
DEFAULT_VAR_COEF = 1.0      # VICReg variance-hinge weight
DEFAULT_COV_COEF = 0.04     # VICReg covariance weight (standard VICReg default scale)
VICREG_GAMMA = 1.0          # target per-dim std for the variance hinge


def build_model(in_dim=IN_DIM, latent_dim=LATENT_DIM, n_act=N_ACT, tau=DEFAULT_TAU):
    """Build the action-conditioned latent JEPA. torch/nn imported lazily; the nn.Module is
    defined INSIDE here so importing this file never requires torch."""
    import torch
    import torch.nn as nn

    class _Jepa(nn.Module):
        def __init__(self):
            super().__init__()
            # online encoder: linear read-in x -> z  (exports to We/be, [latent][in]/[latent])
            self.enc = nn.Linear(in_dim, latent_dim)
            # residual action-conditioned predictor: z_hat' = z + Wz.z + Wa.a + b
            self.pred_z = nn.Linear(latent_dim, latent_dim, bias=False)   # Wz
            self.pred_a = nn.Linear(n_act, latent_dim, bias=True)          # Wa (+ shared bias b)
            # EMA target encoder: a frozen copy, updated by ema_update(), never by grad.
            self.target = nn.Linear(in_dim, latent_dim)
            self.target.load_state_dict(self.enc.state_dict())            # start = online encoder
            for p in self.target.parameters():
                p.requires_grad_(False)                                    # # VERIFY(torch): no-grad target
            self.tau = float(tau)

        def encode(self, x):
            return self.enc(x)

        def encode_target(self, x):
            # stop-gradient target branch (sg): no autograd through the EMA encoder.
            with torch.no_grad():
                return self.target(x)

        def predict(self, z, a_onehot):
            # residual predictor -- predicts the CHANGE and adds it back to z.
            return z + self.pred_z(z) + self.pred_a(a_onehot)

        def ema_update(self):
            # xi <- tau*xi + (1-tau)*theta, momentum-encoder update (I-JEPA / BYOL style).
            with torch.no_grad():
                for pt, po in zip(self.target.parameters(), self.enc.parameters()):
                    pt.mul_(self.tau).add_(po, alpha=1.0 - self.tau)

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


def synthetic_batch(gen, batch, device, in_dim=IN_DIM, n_act=N_ACT):
    """A tiny synthetic (x_t, action, x_{t+1}) stream so the predictor has real signal.

    x_next is a smooth action-conditioned shift of x_t (each action nudges the observation a
    little in a fixed direction), which is exactly the kind of action->next-state structure the
    residual predictor is meant to capture. Deterministic given `gen`.  # VERIFY(torch)
    """
    import torch

    x_t = torch.rand(batch, in_dim, generator=gen, device=device)
    a_idx = torch.randint(0, n_act, (batch,), generator=gen, device=device)
    a_onehot = torch.nn.functional.one_hot(a_idx, n_act).float()
    shift = (a_idx.float().unsqueeze(1) - (n_act - 1) / 2.0) * 0.05        # per-action drift
    x_next = (x_t + shift).clamp(0.0, 1.0)
    return x_t, a_onehot, x_next


def train_step(model, opt, batch, var_coef=DEFAULT_VAR_COEF, cov_coef=DEFAULT_COV_COEF):
    """One optimization step: invariance loss + VICReg, then an EMA target update.

    Returns a dict of the REAL loss components from this step (no fabricated numbers)."""
    import torch  # noqa: F401  (kept explicit; VERIFY: autograd graph built by the ops below)

    x_t, a_onehot, x_next = batch
    z_t = model.encode(x_t)                              # online embedding of x_t
    pred = model.predict(z_t, a_onehot)                  # predicted next latent
    target = model.encode_target(x_next)                 # sg(EMA target) -- already no-grad
    inv_loss = (pred - target).pow(2).sum(dim=1).mean()  # invariance (JEPA prediction loss)
    var_loss, cov_loss = vicreg_terms(z_t)               # anti-collapse on online embeddings
    loss = inv_loss + var_coef * var_loss + cov_coef * cov_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    model.ema_update()                                   # momentum target AFTER the online step
    return {
        "loss": float(loss.detach().cpu()),
        "inv": float(inv_loss.detach().cpu()),
        "var": float(var_loss.detach().cpu()),
        "cov": float(cov_loss.detach().cpu()),
    }


def export_jepa(model, path):
    """THE BRIDGE: write the core ``snn_jepa/1`` JSON so the zero-dep core serves these latents.

    nn.Linear.weight is [out][in], so enc.weight is [latent][in] = We[16][64] and pred_z.weight
    is [latent][latent] = Wz[16][16] -- a direct ``.tolist()`` with no transpose, exactly like
    trained_brain's export. Shapes are ASSERTED before writing. Returns the written dict.
    """
    We = model.enc.weight.detach().cpu().tolist()        # [latent_dim][in_dim]
    be = model.enc.bias.detach().cpu().tolist()          # [latent_dim]
    Wz = model.pred_z.weight.detach().cpu().tolist()     # [latent_dim][latent_dim]
    Wa = model.pred_a.weight.detach().cpu().tolist()     # [latent_dim][n_act]
    b = model.pred_a.bias.detach().cpu().tolist()        # [latent_dim]

    assert len(We) == LATENT_DIM and all(len(r) == IN_DIM for r in We), "We must be [latent][in]"
    assert len(be) == LATENT_DIM, "be must be [latent]"
    assert len(Wz) == LATENT_DIM and all(len(r) == LATENT_DIM for r in Wz), "Wz must be [latent][latent]"
    assert len(Wa) == LATENT_DIM and all(len(r) == N_ACT for r in Wa), "Wa must be [latent][n_act]"
    assert len(b) == LATENT_DIM, "b must be [latent]"

    # VERIFY(assumption): the sibling core cerebellum/jepa_world_model.py is being built in
    # parallel and was not present when this was written. Weight KEYS (We/be/Wz/Wa/b) and the
    # residual predictor form are placed at the top level; if the core nests them under
    # "encoder"/"predictor", move them to match -- the shapes above are the contract.
    blob = {
        "format": JEPA_FORMAT,
        "in_dim": IN_DIM,
        "latent_dim": LATENT_DIM,
        "actions": list(ACTIONS),
        "predictor": "residual",                         # z' = z + Wz.z + Wa.a + b
        "We": We, "be": be,
        "Wz": Wz, "Wa": Wa, "b": b,
        "meta": {
            "source": "research/jepa_scaffold_torch.py (torch SCALE scaffold, UNVERIFIED)",
            "encoder": "linear read-in",
            "loss": "invariance + VICReg(var hinge + covariance)",
            "target": "EMA momentum encoder (stop-gradient)",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    return blob


def main():
    # --- no-torch guard: import lazily so this file is safe to run with no torch installed ---
    try:
        import torch
    except ImportError:
        print("=" * 64)
        print("JEPA SCALE SCAFFOLD (research/, not CI) -- torch is NOT installed")
        print("=" * 64)
        print("This is the torch SCALE tier for the zero-dep core")
        print("cerebellum/jepa_world_model.py. Without torch there is nothing to")
        print("train or export; the zero-dep core is what CI covers.")
        print("To run this scaffold, install torch:  pip install '.[gpu]'")
        print("Exiting cleanly (0).")
        return 0

    import os
    import tempfile

    from torch_device import enable, resolve_device  # reuse the shared cpu/cuda selector

    device = enable(resolve_device())                    # # VERIFY(torch): NORD_GPU -> cuda/cpu
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    steps = 2                                            # a COUPLE of steps -- this is a scaffold
    last = {}
    for _ in range(steps):
        batch = synthetic_batch(gen, batch=64, device=device)
        last = train_step(model, opt, batch)

    export_path = os.path.join(tempfile.gettempdir(), "snn_jepa_scaffold.json")
    blob = export_jepa(model, export_path)

    print("=" * 64)
    print("JEPA SCALE SCAFFOLD -- action-conditioned latent JEPA (torch), UNVERIFIED")
    print("=" * 64)
    print(f"backend      : torch {torch.__version__}  device={device}")
    print(f"actions      : {list(ACTIONS)}")
    print(f"encoder We   : [{len(blob['We'])}][{len(blob['We'][0])}]  be [{len(blob['be'])}]")
    print(f"predictor    : residual  Wz [{len(blob['Wz'])}][{len(blob['Wz'][0])}]  "
          f"Wa [{len(blob['Wa'])}][{len(blob['Wa'][0])}]  b [{len(blob['b'])}]")
    print(f"train ({steps} steps): loss={last['loss']:.4f} inv={last['inv']:.4f} "
          f"var={last['var']:.4f} cov={last['cov']:.4f}   (real, from this run)")
    print(f"exported     : {JEPA_FORMAT} -> {export_path}")
    print("=" * 64)
    print("NOTE: scaffold only -- shapes/format verified, dynamics UNVERIFIED on hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
