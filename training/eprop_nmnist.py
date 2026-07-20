"""
training/eprop_nmnist.py -- BRAIN-NATIVE (e-prop) spiking N-MNIST classifier / encoder for the
Spiking Neural Data Lake. This is the e-prop counterpart of training/surrogate_snn.py: same
perception-tier ROLE (event frame -> latent + class), same encoder interface, but a fundamentally
different LEARNING RULE.

WHAT THIS IS (honest framing). A REAL recurrent spiking network -- input frame -> input weights ->
a RECURRENT spiking (LIF) hidden layer -> a 10-class leaky readout -- trained by e-prop (Bellec,
Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, Nature Communications 11:3625, 2020), NOT by
backpropagation-through-time. The hidden-weight gradient is computed ONLINE, forward-in-time, from a
per-synapse ELIGIBILITY TRACE (filtered pre-synaptic activity gated by the post-synaptic surrogate
pseudo-derivative) times a BROADCAST learning signal (the readout error routed back through the
readout weights). No BPTT history is unrolled or stored -- e-prop's memory is O(neurons + synapses),
independent of the sequence length T. `loss.backward()` is NEVER called for the hidden weights; the
factored gradient is built by hand in torch tensors, vectorized over batch and units, EXACTLY the way
src/snn_data_lake/reasoning/eprop.py factors it (that pure-stdlib module is the proven reference).
Any accuracy printed is the number ACTUALLY obtained at the small scale it was run at; nothing here
is extrapolated to full-dataset numbers, and where e-prop trails BPTT it is said so.

NON-RIGGABLE GOLD STANDARD (self-check). --smoke computes the e-prop hidden-weight gradient AND an
INDEPENDENT reverse-mode BPTT gradient of the SAME loss through the SAME unrolled recurrent net using
the SAME surrogate pseudo-derivative -- and the two share NO gradient code (the BPTT reference is
torch autograd flowing through a differentiable rebuild of the forward, with the surrogate injected in
a custom autograd Function; the e-prop path is a hand-written eligibility loop under no_grad). We
assert their COSINE SIMILARITY over the hidden weights is clearly positive (> 0.5). It is NOT 1.0 --
e-prop truncates the recurrent-error and reset paths BPTT keeps -- but a positive cosine proves the
eligibility trace really IS the factored gradient (not backprop mislabeled). --smoke also asserts a
tiny CPU e-prop train REDUCES the loss and raises accuracy above chance. This mirrors eprop.py's
grad_eprop-vs-grad_bptt gold standard.

WHAT THIS IS NOT (dependency direction, non-negotiable). OPTIONAL torch [train]/[gpu] tier. Imports
NOTHING from src/snn_data_lake/ (in particular nothing from brainstem/ or any safety-envelope file);
importing this module has NO side effects (data loading + training are lazy, only --smoke/CLI or an
explicit call runs anything). The N-MNIST loader is REUSED from research/ng_snn/data.py, imported
lazily so this module (and EpropEncoder) stay importable without tonic / the dataset. Runs in
.venv-cu128 on the RTX 5070; GPU kernels are not bit-exact (expected for the torch tier).

  PYTHONUTF8=1 .venv-cu128/Scripts/python training/eprop_nmnist.py --smoke
      # tiny synthetic: e-prop-vs-BPTT cosine gold standard + tiny e-prop train (CPU-ok, no dataset)
  PYTHONUTF8=1 .venv-cu128/Scripts/python training/eprop_nmnist.py --epochs 1 --limit 4000 --save models/eprop_nmnist.pt
      # short real N-MNIST e-prop train on a subset -> prints test accuracy (needs tonic + dataset)

ASCII-only output (Windows cp1252-safe). torch + any RNG are seeded per --seed; GPU is not bit-exact.
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# N-MNIST event geometry (matches research/ng_snn/data.py SENSOR = (34, 34, 2)).
SENSOR_H = 34
SENSOR_W = 34
IN_CH = 2
NUM_CLASSES = 10
DEFAULT_T = 15          # event time-bins; matches data.py DEFAULT_T
DEFAULT_HIDDEN = 64     # recurrent LIF width == latent dim the encoder emits (matches existing wiring)

# ---- LIF / e-prop dynamics constants (the PROVEN values from reasoning/eprop.py) ----
DEFAULT_ALPHA = 0.90    # membrane decay exp(-dt/tau_m): the diagonal of the recurrent Jacobian
DEFAULT_V_TH = 0.60     # spike threshold (soft reset subtracts V_TH)
DEFAULT_GAMMA = 0.50    # pseudo-derivative half-width: psi = DAMP*max(0, 1 - |v-V_TH|/GAMMA)
DEFAULT_DAMP = 0.30     # pseudo-derivative dampening factor
DEFAULT_KAPPA = 0.90    # leaky-readout decay (also the eligibility-trace output filter)


# ----------------------------------------------------------------------------
# The recurrent spiking network: input frame -> input weights -> recurrent LIF hidden -> leaky readout.
# Weights mirror eprop.py's Win / Wrec / Wout (Wrec has NO self-loop: zero diagonal, kept zero).
# nn.Module purely to hold parameters + a serving forward; TRAINING gradients are computed by hand
# (eprop_grads), NOT by autograd -- see the module docstring.
# ----------------------------------------------------------------------------
class EpropNet(nn.Module):
    def __init__(self, in_ch=IN_CH, num_classes=NUM_CLASSES, hidden=DEFAULT_HIDDEN,
                 sensor_h=SENSOR_H, sensor_w=SENSOR_W, alpha=DEFAULT_ALPHA, v_th=DEFAULT_V_TH,
                 gamma=DEFAULT_GAMMA, damp=DEFAULT_DAMP, kappa=DEFAULT_KAPPA, seed=0,
                 g_in=4.0, g_rec=0.6, g_out=0.6):
        super().__init__()
        self.in_ch = int(in_ch)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.sensor_h = int(sensor_h)
        self.sensor_w = int(sensor_w)
        self.alpha = float(alpha)
        self.v_th = float(v_th)
        self.gamma = float(gamma)
        self.damp = float(damp)
        self.kappa = float(kappa)
        self.n_in = self.in_ch * self.sensor_h * self.sensor_w

        gen = torch.Generator().manual_seed(int(seed))       # per-instance seeded init (deterministic)
        h, n_in, k = self.hidden, self.n_in, self.num_classes
        w_in = (torch.rand(h, n_in, generator=gen) * 2 - 1) * (g_in / (n_in ** 0.5))
        w_rec = (torch.rand(h, h, generator=gen) * 2 - 1) * (g_rec / (h ** 0.5))
        w_rec.fill_diagonal_(0.0)                            # NO self-loop (eprop.py Wrec convention)
        w_out = (torch.rand(k, h, generator=gen) * 2 - 1) * (g_out / (h ** 0.5))
        self.W_in = nn.Parameter(w_in)
        self.W_rec = nn.Parameter(w_rec)
        self.W_out = nn.Parameter(w_out)
        self.b_out = nn.Parameter(torch.zeros(k))

    # -- shared helpers (the ONE forward math; the three code paths -- serving, e-prop, BPTT -- each
    #    re-run these equations in their own loop, sharing constants but not gradient logic) --
    def psi(self, v):
        """Surrogate pseudo-derivative of the spike: DAMP * max(0, 1 - |v - V_TH| / GAMMA)."""
        return self.damp * torch.clamp(1.0 - (v - self.v_th).abs() / self.gamma, min=0.0)

    def flatten_frames(self, x):
        """[..., C, H, W] event frames -> [..., n_in] per-timestep input vectors."""
        return x.reshape(*x.shape[:-3], self.n_in)

    def offdiag_mask(self, device):
        return 1.0 - torch.eye(self.hidden, device=device)

    @torch.no_grad()
    def forward(self, x):
        """Serve: x [B, T, C, H, W] event frames -> (logits_sum [B, num_classes], latent_sum [B, hidden]).
        logits_sum is the time-summed leaky readout (the class scores); latent_sum is the hidden
        spike-COUNT activity over the T window (the encoder latent). Manual LIF recurrence, no autograd,
        no dataset/tonic needed -- torch-only-to-serve."""
        x = self.flatten_frames(x.float())                  # [B, T, n_in]
        b, t = x.shape[0], x.shape[1]
        dev = x.device
        v = torch.zeros(b, self.hidden, device=dev)
        z = torch.zeros(b, self.hidden, device=dev)
        y = torch.zeros(b, self.num_classes, device=dev)    # leaky readout integrator
        logits_sum = torch.zeros(b, self.num_classes, device=dev)
        latent_sum = torch.zeros(b, self.hidden, device=dev)
        w_rec = self.W_rec * self.offdiag_mask(dev)
        for step in range(t):
            cur = x[:, step] @ self.W_in.t() + z @ w_rec.t()
            v = self.alpha * v - self.v_th * z + cur         # soft reset: subtract V_TH on last spike
            z = (v >= self.v_th).float()
            y = self.kappa * y + z @ self.W_out.t()
            logits_sum = logits_sum + (y + self.b_out)
            latent_sum = latent_sum + z
        return logits_sum, latent_sum

    def config(self):
        return dict(in_ch=self.in_ch, num_classes=self.num_classes, hidden=self.hidden,
                    sensor_h=self.sensor_h, sensor_w=self.sensor_w, alpha=self.alpha, v_th=self.v_th,
                    gamma=self.gamma, damp=self.damp, kappa=self.kappa)


# ----------------------------------------------------------------------------
# e-prop gradient -- ONLINE, forward-only, hand-built (NO autograd for the hidden weights).
# Factors dLoss/dw EXACTLY as reasoning/eprop.py does:
#   eligibility VECTOR   eps = ALPHA*eps + pre_activity            (per pre-synaptic unit, shared over post)
#   filtered TRACE       ebar = KAPPA*ebar + psi_post * eps        (KAPPA is the readout filter)
#   learning signal      L_j(t) = sum_k W_out[k,j] * err_k(t)      (broadcast error through readout weights)
#   hidden update        dW_ji = sum_t L_j(t) * ebar_ji(t)
# Readout uses its EXACT (non-recurrent) gradient: dW_out = sum_t err(t) (x) zbar(t), zbar = KAPPA*zbar + z.
# Loss = per-timestep softmax cross-entropy against the constant label, summed over t and batch (so err(t)
# = softmax(logits(t)) - onehot, the direct analogue of eprop.py's regression err(t) = y(t) - target(t)).
# ----------------------------------------------------------------------------
@torch.no_grad()
def eprop_grads(net, x, y):
    """x [B, T, C, H, W], y [B] int labels. Returns (grads, loss, spike_rate) where grads is a dict of
    dLoss/dw for W_in, W_rec, W_out, b_out (the e-prop-factored hidden grads + exact readout grads). No
    per-timestep history is stored: eps/ebar/zbar are updated in place -> O(batch * synapses) memory,
    independent of T."""
    x = net.flatten_frames(x.float())                       # [B, T, n_in]
    b, t = x.shape[0], x.shape[1]
    dev = x.device
    h, n_in, k = net.hidden, net.n_in, net.num_classes
    onehot = F.one_hot(y.long(), k).float().to(dev)         # [B, K]
    mask = net.offdiag_mask(dev)
    w_rec = net.W_rec * mask

    v = torch.zeros(b, h, device=dev)
    z = torch.zeros(b, h, device=dev)
    yint = torch.zeros(b, k, device=dev)                    # leaky readout integrator
    eps_in = torch.zeros(b, n_in, device=dev)              # alpha-filtered input (shared over post j)
    eps_rec = torch.zeros(b, h, device=dev)               # alpha-filtered z_prev (shared over post j)
    ebar_in = torch.zeros(b, h, n_in, device=dev)         # kappa-filtered eligibility trace [B, post, pre]
    ebar_rec = torch.zeros(b, h, h, device=dev)
    zbar = torch.zeros(b, h, device=dev)                   # kappa-filtered spikes (exact readout eligibility)

    dW_in = torch.zeros(h, n_in, device=dev)
    dW_rec = torch.zeros(h, h, device=dev)
    dW_out = torch.zeros(k, h, device=dev)
    db_out = torch.zeros(k, device=dev)
    loss_sum = 0.0
    spikes = 0.0

    for step in range(t):
        z_prev = z
        xt = x[:, step]
        cur = xt @ net.W_in.t() + z_prev @ w_rec.t()
        v = net.alpha * v - net.v_th * z_prev + cur
        z = (v >= net.v_th).float()
        pj = net.psi(v)                                     # [B, H]
        spikes += float(z.mean().item())

        yint = net.kappa * yint + z @ net.W_out.t()
        logits = yint + net.b_out                           # [B, K]
        err = torch.softmax(logits, dim=1) - onehot         # [B, K] softmax-CE error (per sample)
        loss_sum += float(F.cross_entropy(logits, y.long(), reduction="sum").item())

        # eligibility: eps = ALPHA*eps + pre ; ebar = KAPPA*ebar + psi_post * eps  (mirrors eprop.py)
        eps_in = net.alpha * eps_in + xt
        eps_rec = net.alpha * eps_rec + z_prev
        ebar_in = net.kappa * ebar_in + pj.unsqueeze(2) * eps_in.unsqueeze(1)     # [B, H, n_in]
        ebar_rec = net.kappa * ebar_rec + pj.unsqueeze(2) * eps_rec.unsqueeze(1)  # [B, H, H]

        lsig = err @ net.W_out                              # broadcast learning signal L_j(t) = [B, H]
        dW_in += torch.einsum("bj,bji->ji", lsig, ebar_in)
        dW_rec += torch.einsum("bj,bjk->jk", lsig, ebar_rec)

        zbar = net.kappa * zbar + z                         # readout weights: exact (non-recurrent)
        dW_out += torch.einsum("bk,bj->kj", err, zbar)
        db_out += err.sum(0)

    dW_rec = dW_rec * mask                                  # no self-loop gradient
    grads = {"W_in": dW_in, "W_rec": dW_rec, "W_out": dW_out, "b_out": db_out}
    return grads, loss_sum / max(t, 1), spikes / max(t, 1)


# ----------------------------------------------------------------------------
# Independent reverse-mode BPTT reference (the gold standard). Shares NO gradient code with eprop_grads:
# a differentiable rebuild of the SAME forward with the SAME surrogate (injected via _SpikeFn.backward),
# then torch autograd. Autograd keeps the recurrent off-diagonal and soft-reset (-V_TH*z_prev) paths that
# e-prop truncates -> this is the TRUE gradient the eligibility gradient is measured against.
# ----------------------------------------------------------------------------
class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th, gamma, damp):
        ctx.save_for_backward(v)
        ctx.v_th, ctx.gamma, ctx.damp = v_th, gamma, damp
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        psi = ctx.damp * torch.clamp(1.0 - (v - ctx.v_th).abs() / ctx.gamma, min=0.0)
        return grad_out * psi, None, None, None             # SAME surrogate as net.psi / eprop.py


def bptt_grads(net, x, y):
    """True reverse-mode BPTT of the SAME summed-per-timestep softmax-CE loss through a differentiable
    rebuild of the SAME unrolled net, via autograd. Returns a dict of grads for W_in, W_rec, W_out,
    b_out. Does NOT touch net.*.grad (uses torch.autograd.grad)."""
    x = net.flatten_frames(x.float())
    b, t = x.shape[0], x.shape[1]
    dev = x.device
    mask = net.offdiag_mask(dev)
    w_rec = net.W_rec * mask
    v = torch.zeros(b, net.hidden, device=dev)
    z = torch.zeros(b, net.hidden, device=dev)
    yint = torch.zeros(b, net.num_classes, device=dev)
    loss = torch.zeros((), device=dev)
    for step in range(t):
        cur = x[:, step] @ net.W_in.t() + z @ w_rec.t()
        v = net.alpha * v - net.v_th * z + cur
        z = _SpikeFn.apply(v, net.v_th, net.gamma, net.damp)
        yint = net.kappa * yint + z @ net.W_out.t()
        logits = yint + net.b_out
        loss = loss + F.cross_entropy(logits, y.long(), reduction="sum")
    g_in, g_rec, g_out, g_b = torch.autograd.grad(loss, (net.W_in, net.W_rec, net.W_out, net.b_out))
    return {"W_in": g_in, "W_rec": g_rec * mask, "W_out": g_out, "b_out": g_b}


def _cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


# ----------------------------------------------------------------------------
# The perception interface the brain's hook calls -- mirrors training/surrogate_snn.SurrogateEncoder.
# Lazy: constructing/loading an encoder does NOT train and needs only torch + the saved weights.
# ----------------------------------------------------------------------------
class EpropEncoder:
    """Wraps a trained EpropNet as a perception encoder (same interface as SurrogateEncoder).

    encode(event_frame) -> latent vector (the recurrent hidden layer's spike-COUNT activity over T):
      * event_frame [T, C, H, W]      -> latent [hidden]        (single sample; brain-hook path)
      * event_frame [B, T, C, H, W]   -> latent [B, hidden]     (batched)
    classify(event_frame) -> predicted class label(s): [T,C,H,W] -> int ; [B,T,C,H,W] -> list[int].
    Accepts a numpy array or a torch tensor. Latent values are spike counts over the T window (>= 0).

    load(path, device=...) rebuilds the net from the checkpoint's saved config + weights -- no training,
    no dataset needed. Checkpoint format matches surrogate_snn: {"model": state_dict, "config": cfg}."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt.get("config", {})
        net = EpropNet(**cfg)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    def _prep(self, event_frame, what):
        x = event_frame
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().to(self.device)
        single = (x.ndim == 4)                              # [T, C, H, W] -> add batch dim
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError("%s expects [T,C,H,W] or [B,T,C,H,W]; got shape %s"
                             % (what, tuple(x.shape)))
        return x, single

    @torch.no_grad()
    def encode(self, event_frame):
        x, single = self._prep(event_frame, "encode")
        _, latent = self.net(x)
        return latent[0] if single else latent

    @torch.no_grad()
    def classify(self, event_frame):
        x, single = self._prep(event_frame, "classify")
        logits, _ = self.net(x)
        pred = logits.argmax(dim=1)
        return int(pred[0]) if single else [int(v) for v in pred]


# ----------------------------------------------------------------------------
# Data (reused from research/ng_snn/data.py, imported lazily so this module stays torch-only-importable).
# ----------------------------------------------------------------------------
def _load_data_module():
    here = os.path.dirname(os.path.abspath(__file__))
    ng = os.path.abspath(os.path.join(here, "..", "research", "ng_snn"))
    if ng not in sys.path:
        sys.path.insert(0, ng)
    import data as nmnist_data   # research/ng_snn/data.py -- disk-streaming N-MNIST loader (tonic)
    return nmnist_data


def _apply_grads(net, grads, opt):
    """Load the e-prop-computed grads into .grad and take one optimizer step (Adam on the e-prop grads).
    No loss.backward() anywhere -- the hidden grads are the eligibility factorization."""
    opt.zero_grad(set_to_none=True)
    net.W_in.grad = grads["W_in"]
    net.W_rec.grad = grads["W_rec"]
    net.W_out.grad = grads["W_out"]
    net.b_out.grad = grads["b_out"]
    opt.step()
    with torch.no_grad():
        net.W_rec.copy_(net.W_rec * net.offdiag_mask(net.W_rec.device))   # keep the diagonal at zero


def evaluate(net, loader, device, max_samples=None):
    """Test accuracy = argmax of the time-summed leaky readout. Caps at max_samples for quick runs."""
    net.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, yb in loader:
            x = x.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            logits, _ = net(x)
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += int(yb.shape[0])
            if max_samples is not None and seen >= max_samples:
                break
    return (100.0 * correct / seen) if seen else 0.0, seen


def train_and_eval(args):
    device = args.device
    torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = _load_data_module()
    print("device=%s  T=%d  batch=%d  hidden=%d  limit=%s  epochs=%d  lr=%.4g"
          % (device, args.T, args.batch_size, args.hidden, str(args.limit), args.epochs, args.lr))
    if device == "cuda" and torch.cuda.is_available():
        print("gpu=%s  torch=%s" % (torch.cuda.get_device_name(0), torch.__version__))

    train_loader, test_loader = data.make_nmnist_loaders(
        n_time_bins=args.T, batch_size=args.batch_size, num_workers=args.workers)
    print("train_batches=%d  test_batches=%d" % (len(train_loader), len(test_loader)))

    net = EpropNet(hidden=args.hidden, seed=args.seed).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print("model: input -> recurrent LIF(%d) -> leaky readout(%d) | e-prop (no BPTT) | params=%d"
          % (args.hidden, NUM_CLASSES, n_params))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        net.train()
        seen = 0
        running = 0.0
        nb = 0
        for x, yb in train_loader:
            x = x.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()      # tonic labels are int32 -> int64
            grads, loss, rate = eprop_grads(net, x, yb)        # e-prop factored gradient (no backward)
            _apply_grads(net, grads, opt)
            running += loss
            nb += 1
            seen += int(yb.shape[0])
            if nb % 20 == 0:
                print("  epoch %d  batch %d  seen %d  loss %.4f  spike_rate %.3f"
                      % (epoch, nb, seen, running / nb, rate))
            if args.limit is not None and seen >= args.limit:
                break
        acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
        print("epoch %d done: train_samples=%d  avg_loss=%.4f  test_acc=%.2f%% (on %d test samples)"
              % (epoch, seen, running / max(nb, 1), acc, ntest))

    acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
    print("=" * 74)
    print("EPROP-SNN (brain-native perception tier, e-prop -- NO BPTT) | device=%s" % device)
    print("test accuracy %.2f%% on %d N-MNIST test samples (T=%d, %d train samples, %d epoch(s))"
          % (acc, ntest, args.T, min(args.limit or 10 ** 12, 60000), args.epochs))
    print("HONEST: small-scale foundation number, not SOTA and not extrapolated. Hidden weights learned")
    print("by e-prop eligibility traces + broadcast error (loss.backward NEVER called for them); readout")
    print("uses its exact gradient. Torch [train]/[gpu] tier ONLY -- the zero-dep core never imports this.")
    print("Encoder latent dim = %d (recurrent hidden spike-count activity)." % args.hidden)
    print("=" * 74)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save({"model": net.state_dict(), "config": net.config()}, args.save)
        print("saved encoder checkpoint -> %s" % args.save)
    return acc


# ----------------------------------------------------------------------------
# Smoke: synthetic class-separable events (no dataset). Proves (b) the e-prop-vs-BPTT gold-standard
# cosine and (a) that a tiny e-prop train reduces loss + raises accuracy above chance, then round-trips
# the encoder through the real load(path) interface. CPU-ok.
# ----------------------------------------------------------------------------
def _make_synthetic(n_per_class, n_classes, t_steps, in_ch, h, w, seed, density=0.28, noise=0.03):
    """Class-separable event frames [N, T, C, H, W] + labels [N]. Each class has a FIXED spatial
    template (SHARED across train/test so the task is generalizable), persistent over time, plus a
    little per-timestep flip noise. The `seed` varies only the noise + sample order, NOT the templates."""
    n_in = in_ch * h * w
    tgen = torch.Generator().manual_seed(7000 + n_classes)  # template seed FIXED (shared train<->test)
    templates = (torch.rand(n_classes, n_in, generator=tgen) < density).float()
    gen = torch.Generator().manual_seed(int(seed))          # per-split noise + shuffle only
    xs, ys = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            base = templates[c].unsqueeze(0).repeat(t_steps, 1)          # [T, n_in] constant class pattern
            flip = (torch.rand(t_steps, n_in, generator=gen) < noise).float()
            frame = (base + flip - 2 * base * flip).clamp(0, 1)          # XOR-style noise flips
            xs.append(frame.reshape(t_steps, in_ch, h, w))
            ys.append(c)
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


def run_smoke(device):
    torch.manual_seed(0)
    n_classes, t_steps, h, w = 3, 8, 8, 8                    # tiny + fast on CPU
    hidden = 12
    x_tr, y_tr = _make_synthetic(10, n_classes, t_steps, IN_CH, h, w, seed=1)
    x_te, y_te = _make_synthetic(5, n_classes, t_steps, IN_CH, h, w, seed=2)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    print("smoke: device=%s  synthetic [N=%d, T=%d, C=%d, %dx%d] %d classes  hidden=%d (no dataset)"
          % (device, x_tr.shape[0], t_steps, IN_CH, h, w, n_classes, hidden))

    net = EpropNet(in_ch=IN_CH, num_classes=n_classes, hidden=hidden, sensor_h=h, sensor_w=w,
                   seed=0).to(device)

    # (b) GOLD STANDARD: e-prop hidden grad vs an INDEPENDENT reverse-mode BPTT grad, SAME loss/net/psi.
    ge, _, rate = eprop_grads(net, x_tr, y_tr)
    gb = bptt_grads(net, x_tr, y_tr)
    mask = net.offdiag_mask(device).bool()
    e_hidden = torch.cat([ge["W_in"].reshape(-1), ge["W_rec"][mask].reshape(-1)])
    b_hidden = torch.cat([gb["W_in"].reshape(-1), gb["W_rec"][mask].reshape(-1)])
    cos_hidden = _cosine(e_hidden, b_hidden)
    cos_readout = _cosine(ge["W_out"], gb["W_out"])
    print("  hidden spike_rate at init = %.3f (must be > 0 for a meaningful gradient)" % rate)
    print("  e-prop vs BPTT cosine (HIDDEN W_in+W_rec) = %.4f   [gold standard, must be > 0.5]"
          % cos_hidden)
    print("  e-prop vs BPTT cosine (readout W_out, exact) = %.4f   [plumbing check, must be ~1]"
          % cos_readout)

    # (a) tiny e-prop train: loss must fall, accuracy must rise above chance (= 1/n_classes).
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    _, init_loss, _ = eprop_grads(net, x_tr, y_tr)
    init_acc, _ = evaluate(net, [(x_te, y_te)], device)
    for _ in range(80):
        grads, _, _ = eprop_grads(net, x_tr, y_tr)
        _apply_grads(net, grads, opt)
    _, final_loss, _ = eprop_grads(net, x_tr, y_tr)
    final_acc, _ = evaluate(net, [(x_te, y_te)], device)
    chance = 100.0 / n_classes
    print("  e-prop train: loss %.4f -> %.4f ; test acc %.1f%% -> %.1f%% (chance %.1f%%)"
          % (init_loss, final_loss, init_acc, final_acc, chance))

    # encoder round-trip via the real load(path) interface (tempfile, then deleted).
    enc_dir = tempfile.mkdtemp()
    ckpt = os.path.join(enc_dir, "smoke_eprop_encoder.pt")
    torch.save({"model": net.state_dict(), "config": net.config()}, ckpt)
    enc = EpropEncoder.load(ckpt, device=device)
    single = enc.encode(x_te[0])                            # [T,C,H,W] -> [hidden]
    batched = enc.encode(x_te)                              # [B,T,C,H,W] -> [B,hidden]
    label = enc.classify(x_te[0])
    labels = enc.classify(x_te)
    os.remove(ckpt)
    os.rmdir(enc_dir)
    print("  encoder.encode([T,C,H,W]) -> %s ; encode([B,T,C,H,W]) -> %s ; classify -> int %s, list len %d"
          % (tuple(single.shape), tuple(batched.shape), label, len(labels)))

    # ---- self-check asserts (the non-riggable gold standard is cos_hidden > 0.5) ----
    assert rate > 0.0, "hidden units must spike at init (else psi=0 and the gradient is undefined)"
    assert cos_hidden > 0.5, ("(b) e-prop gradient must ALIGN with the true BPTT gradient: cosine %.4f !> 0.5"
                              % cos_hidden)
    assert cos_readout > 0.98, ("readout grad must match BPTT (exact): cosine %.4f !> 0.98" % cos_readout)
    assert final_loss < init_loss, ("(a) e-prop must reduce loss: %.4f !< %.4f" % (final_loss, init_loss))
    assert final_acc > chance + 15.0, ("(a) accuracy must rise well above chance: %.1f%% vs chance %.1f%%"
                                       % (final_acc, chance))
    assert tuple(single.shape) == (hidden,), "single-sample latent must be [hidden]"
    assert tuple(batched.shape) == (x_te.shape[0], hidden), "batched latent must be [B, hidden]"
    assert isinstance(label, int) and isinstance(labels, list), "classify types"
    print("SMOKE OK: e-prop~BPTT hidden cosine %.4f (>0.5, factored gradient confirmed vs an INDEPENDENT"
          % cos_hidden)
    print("reverse-mode reference); e-prop LEARNS loss %.4f->%.4f (acc %.1f%%->%.1f%%, chance %.1f%%);"
          % (init_loss, final_loss, init_acc, final_acc, chance))
    print("encoder load/encode/classify path verified (no dataset, %s). loss.backward NEVER called for"
          % device)
    print("the hidden weights -- they are trained purely by the online eligibility factorization.")


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Brain-native (e-prop) recurrent spiking classifier / "
                                            "encoder on N-MNIST -- online eligibility traces, NO BPTT.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny synthetic: e-prop-vs-BPTT cosine gold standard + tiny train (CPU-ok, no dataset)")
    p.add_argument("--epochs", type=int, default=1, help="training epochs")
    p.add_argument("--limit", type=int, default=None,
                   help="cap training samples per epoch (small-subset quick run)")
    p.add_argument("--eval-limit", type=int, default=2000, dest="eval_limit",
                   help="cap test samples during eval (quick runs)")
    p.add_argument("--T", type=int, default=DEFAULT_T, help="event time-bins per sample")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="recurrent LIF width == latent dim")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=2, help="dataloader workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None, help="path to save the encoder checkpoint")
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_dev, choices=["cuda", "cpu"])
    return p


def main():
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[note] cuda requested but unavailable -- falling back to cpu")
        args.device = "cpu"
    if args.smoke:
        run_smoke(args.device)
        return
    train_and_eval(args)


if __name__ == "__main__":
    main()
