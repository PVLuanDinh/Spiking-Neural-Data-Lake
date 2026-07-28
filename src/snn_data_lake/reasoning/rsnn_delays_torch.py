"""Tier-3 deeper spiking encoder: DCLS LEARNABLE DELAYS + ADAPTIVE-LIF (PyTorch trainer) -> zero-dep core.

**Runs on the RTX 5070 (NORD_GPU=1 .venv-cu128); CPU-smoke-verified only in the authoring env.** This is
the torch TRAINING tier for the zero-dep core cortex/rsnn_encoder.py (served-frozen `snn_rsnn/1`). It
trains a recurrent-free feedforward SNN whose per-synapse conduction DELAYS are learned with Dilated
Convolutions with Learnable Spacings (DCLS) alongside the weights, with adaptive-LIF neurons carrying
heterogeneous TRAINABLE membrane time constants -- the ICLR-2024 recipe (Hammouamri, Khalfaoui-Hassani &
Masquelier, "Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable
Spacings", arXiv:2306.17670) that reaches SOTA on the temporal SHD/SSC benchmarks WITHOUT recurrence.
Delays SHINE on temporal (audio) data, so the target is SHD (Spiking Heidelberg Digits, 20-class, via
tonic); it FALLS BACK to N-MNIST (via research/ng_snn/data.py) and to a self-contained SYNTHETIC
temporal dataset (no download) so the trainer runs standalone for the CPU smoke.

Bridge to the zero-dep core (mirrors jepa_scaffold_torch.py -> cerebellum/jepa_world_model.py):
  * the DCLS delay is a 1-D conv over TIME whose kernel is a Gaussian bump at a learnable position P
    (the delay). sigma is annealed during training; at FINALIZE the bump becomes a Dirac at round(P),
    i.e. an INTEGER tap -- exactly the integer delay the pure-stdlib core reads from its ring buffer.
  * the adaptive-LIF dynamics are written ONCE and mirrored bit-for-bit by RsnnEncoder in the core:
        I[t] = sum_i W[j][i]*x_i[t-d[j][i]];  u=beta*u+I;  thr=v_th+a*b;  s=(u>=thr);  u-=thr*s;  b=rho*b+s
    (torch uses a surrogate gradient ONLY in the backward pass; the forward Heaviside is identical.)
  * export writes `snn_rsnn/1` JSON: per layer W[out][in] (Linear.weight is [out][in], NO transpose),
    integer delays[out][in], per-neuron beta/a/v_th, max_delay; layers in APPLICATION order.
  * verify_transfer loads the export into the core and asserts the core forward == a float64 CPU torch
    reference on probe samples (spike counts EXACT, currents within tol; the reference is pinned to CPU).

CONTROLS reported (honest, un-riggable) -- all on the SAME held-out set, features -> a generalized
SLDA-RFF readout (the readout research/developmental_loop_slda.py uses, generalized to n_cls):
  * DELAYS ON vs OFF   -- freeze every delay to 0 and retrain (expect >= +3% from learned delays).
  * ADAPTIVE tau ON vs OFF -- heterogeneous trainable beta+adaptation vs a homogeneous fixed-tau plain LIF.
  * RANDOM-INIT encoder -- a FROZEN UNTRAINED RSNN's features + SLDA-RFF must LOSE to the trained one by a
    large margin (the critical honesty check; spiking_jepa's negative was a random encoder holding rank).
  * CEILING-MOVED -- trained temporal features + SLDA-RFF vs a TIMING-COLLAPSED (rate) baseline on the
    same held-out (the faithful temporal-data stand-in for a frozen conv-STDP spatial encoder, which is
    image-only and cannot process audio SHD; see the report caveat).

torch is imported LAZILY inside the functions/main (never at module top), so this file imports cleanly
with no torch installed and `python research/rsnn_delays_torch.py` exits 0 with a guidance banner. All
printed numbers are REAL (computed by the run); nothing is hard-coded.

Run (GPU, real dataset):  NORD_GPU=1 NORD_DATASET=shd PYTHONPATH=src .venv-cu128/Scripts/python research/rsnn_delays_torch.py
Run (CPU smoke, tiny):    NORD_SMOKE=1 PYTHONPATH=src .venv-cu128/Scripts/python research/rsnn_delays_torch.py
Run (no torch):           python research/rsnn_delays_torch.py   -> guidance, exit 0
Knobs: NORD_DATASET(synthetic|shd|nmnist) NORD_T NORD_HIDDEN NORD_MAXDELAY NORD_EPOCHS NORD_BATCH
       NORD_TRAIN NORD_TEST NORD_RFF NORD_LR NORD_SMOKE(=1 tiny) NORD_GPU NORD_DATA_DIR
Deps: torch (+ tonic/torchvision for shd/nmnist). research/GPU tier; the served core stays zero-dep.
"""

import json
import math
import os

RSNN_FORMAT = "snn_rsnn/1"
RHO_ADAPT = 0.9  # shared spike-triggered-adaptation decay (fixed hyperparam, mirrored in core)
V_TH = 1.0  # fixed base firing threshold (exported per-neuron as a constant vector)
SURR_BETA = 10.0  # fast-sigmoid surrogate sharpness (backward only)


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


# =====================================================================================================
# Model (torch imported lazily inside build_model so the module is import-safe with no torch)
# =====================================================================================================
def build_model(n_in, n_class, hidden, max_delay, learn_delays=True, adaptive=True, seed=0):
    """Feedforward DCLS-delay + adaptive-LIF SNN + a count-readout head. `hidden` is a list of layer
    widths (the served spiking layers); the last one is the penultimate feature layer. Tensors are built
    on CPU with a CPU generator then moved by the caller's .to(device) -- NEVER via set_default_device
    (torch.randn/normal with a CPU generator under a cuda default errors/misplaces; the documented trap)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _SpikeFn(torch.autograd.Function):
        """Heaviside forward (u>=thr -> spike); fast-sigmoid surrogate gradient in backward ONLY, so the
        forward is IDENTICAL to the zero-dep core's hard threshold (the bridge check depends on this)."""

        @staticmethod
        def forward(ctx, x):  # x = u - thr
            ctx.save_for_backward(x)
            return (x >= 0).to(x.dtype)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g / (1.0 + SURR_BETA * x.abs()) ** 2

    class _DclsDelay(nn.Module):
        """DCLS delay layer: a dense synapse W[out][in] plus a per-synapse learnable delay position
        P[out][in] rendered as a Gaussian bump along a length-max_delay 1-D temporal conv kernel. `hard`
        replaces the bump with a Dirac at round(P) -> the integer tap the core serves."""

        def __init__(self, nin, nout, maxd, learn, g):
            super().__init__()
            self.nin, self.nout, self.maxd = nin, nout, maxd
            # weight-gain so sparse binary spike volleys actually cross the LIF threshold at init (a dead,
            # never-firing init gives no surrogate gradient); NORD_WGAIN tunes it for denser inputs (SHD).
            se = _envf("NORD_WGAIN", 5.0) / math.sqrt(nin)
            W0 = (torch.rand(nout, nin, generator=g, device="cpu") * 2.0 - 1.0) * se
            self.W = nn.Parameter(W0)
            # per-neuron trainable bias CURRENT -- the learnable knob that sets each neuron's operating
            # point (firing rate); exported per layer and mirrored in the zero-dep core's I[t].
            self.bias = nn.Parameter(torch.full((nout,), 0.2, device="cpu"))
            # delay positions in [0, maxd-1]; heterogeneous init when learned, all-zero (plain synapse) when not.
            if learn:
                P0 = torch.rand(nout, nin, generator=g, device="cpu") * (maxd - 1)
                self.P = nn.Parameter(P0)
                self.learn = True
            else:
                self.register_buffer("P", torch.zeros(nout, nin))
                self.learn = False
            self.sigma = max(0.5, (maxd - 1) / 2.0)  # annealed toward ~0.5 during training

        def _kernel(self, hard):
            dev = self.W.device
            dcoord = torch.arange(self.maxd, device=dev, dtype=self.W.dtype)  # delay axis 0..maxd-1
            P = self.P.clamp(0.0, self.maxd - 1).to(self.W.dtype)
            if hard:
                Pr = P.round()
                bump = (dcoord.view(1, 1, -1) == Pr.unsqueeze(-1)).to(
                    self.W.dtype
                )  # Dirac at round(P)
            else:
                d = dcoord.view(1, 1, -1) - P.unsqueeze(-1)  # [out,in,maxd]
                bump = torch.exp(-0.5 * (d / self.sigma) ** 2)
                bump = bump / (bump.sum(-1, keepdim=True) + 1e-8)  # unit-mass -> W is the gain
            kern_d = self.W.unsqueeze(-1) * bump  # indexed by delay d
            return torch.flip(kern_d, dims=[-1])  # conv tau = maxd-1-d

        def forward(self, x, hard=False):
            # x [B,T,nin] -> currents [B,T,nout]; causal left-pad so I[t] = sum_i W*x[t-d] + bias.
            xc = x.transpose(1, 2)  # [B,nin,T]
            k = self._kernel(hard)  # [nout,nin,maxd]
            y = F.conv1d(F.pad(xc, (self.maxd - 1, 0)), k)  # [B,nout,T]
            return y.transpose(1, 2) + self.bias  # + per-neuron bias current

        def int_delays(self):
            return self.P.detach().clamp(0.0, self.maxd - 1).round().to(torch.int64).cpu()

    class _Alif(nn.Module):
        """Adaptive-LIF population: per-neuron membrane decay beta=sigmoid(beta_raw) (heterogeneous,
        trainable when adaptive) and per-neuron adaptation coupling a=softplus(a_raw). adaptive=False ->
        a homogeneous FIXED beta and a=0 (a plain LIF), the tau-ablation control."""

        def __init__(self, nout, adaptive, g):
            super().__init__()
            self.nout, self.adaptive = nout, adaptive
            if adaptive:
                # heterogeneous betas spread across (0.5,0.95) via the logit of a uniform spread.
                b_target = 0.5 + torch.rand(nout, generator=g, device="cpu") * 0.45
                self.beta_raw = nn.Parameter(
                    torch.log(b_target / (1.0 - b_target))
                )  # on CPU (b_target is)
                self.a_raw = nn.Parameter(
                    torch.full((nout,), -2.0, device="cpu")
                    + torch.rand(nout, generator=g, device="cpu") * 0.5
                )
            else:
                self.register_buffer(
                    "beta_raw", torch.full((nout,), math.log(0.9 / 0.1), device="cpu")
                )  # fixed beta=0.9
                self.a_raw = None  # a == 0
            self.register_buffer("v_th", torch.full((nout,), V_TH, device="cpu"))

        def beta(self):
            import torch

            return torch.sigmoid(self.beta_raw)

        def a(self):
            import torch
            import torch.nn.functional as F

            if self.a_raw is None:
                return torch.zeros(self.nout, device=self.v_th.device, dtype=self.v_th.dtype)
            return F.softplus(self.a_raw)

        def forward(self, I):
            # I [B,T,nout] currents -> spikes [B,T,nout]. Sequential ALIF recurrence over T (mirrors core).
            import torch

            B, T, N = I.shape
            beta = self.beta().to(I.dtype)
            a = self.a().to(I.dtype)
            v_th = self.v_th.to(I.dtype)
            u = torch.zeros(B, N, device=I.device, dtype=I.dtype)
            b = torch.zeros(B, N, device=I.device, dtype=I.dtype)
            out = []
            for t in range(T):
                u = beta * u + I[:, t, :]
                thr = v_th + a * b
                s = _SpikeFn.apply(u - thr)
                u = u - thr * s
                b = RHO_ADAPT * b + s
                out.append(s)
            return torch.stack(out, dim=1)

    class _Rsnn(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator(device="cpu").manual_seed(seed)
            self.delays = nn.ModuleList()
            self.alifs = nn.ModuleList()
            prev = n_in
            for w in hidden:
                self.delays.append(_DclsDelay(prev, w, max_delay, learn_delays, g))
                self.alifs.append(_Alif(w, adaptive, g))
                prev = w
            self.readout = nn.Linear(
                prev, n_class, device="cpu"
            )  # count-readout head (NOT served; SLDA replaces it)
            with torch.no_grad():
                self.readout.weight.mul_(0.0).add_(
                    (torch.rand(n_class, prev, generator=g, device="cpu") * 2.0 - 1.0)
                    / math.sqrt(prev)
                )
                self.readout.bias.zero_()
            self.max_delay = max_delay
            self.learn_delays = learn_delays
            self.adaptive = adaptive

        def encode(self, x, hard=False):
            """Return (penultimate spike counts [B,Hlast], list of per-layer spike tensors)."""
            spikes = None
            per_layer = []
            h = x
            for dl, al in zip(self.delays, self.alifs):
                spikes = al(dl(h, hard=hard))
                per_layer.append(spikes)
                h = spikes
            counts = spikes.sum(dim=1)  # [B, Hlast] spike counts over time
            return counts, per_layer

        def features(self, x, hard=False):
            return self.encode(x, hard=hard)[0]

        def forward(self, x, hard=False):
            return self.readout(self.features(x, hard=hard))  # logits

        def anneal_sigma(self, frac):
            """Anneal every DCLS sigma from its init toward ~0.5 as training progresses (frac in [0,1])."""
            for dl in self.delays:
                init = max(0.5, (dl.maxd - 1) / 2.0)
                dl.sigma = max(0.5, init * (1.0 - frac) + 0.5 * frac)

    return _Rsnn()


# =====================================================================================================
# Generalized SLDA-on-RFF readout (the developmental_loop_slda.SLDA_RFF math, parametrized to n_cls)
# =====================================================================================================
def slda_rff_accuracy(Ftr, ytr, Fte, yte, n_cls, rff=1024, gamma=0.02, shrink=0.1, seed=0):
    """Streaming-LDA on Random Fourier Features -- a linear head that approximates the static RBF readout
    (RanDumb). Fit sufficient statistics on train features, score test. Generalized to n_cls (the shared
    SLDA_RFF hardcodes 10). GPU-safe: RFF map drawn on CPU then moved (the generator/device trap)."""
    import torch

    dev = Ftr.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    D = Ftr.shape[1]
    W = (torch.randn(D, rff, generator=g, device="cpu") * gamma).to(dev)
    bph = (torch.rand(rff, generator=g, device="cpu") * 6.283185).to(dev)
    scale = (2.0 / rff) ** 0.5

    def _rff(Fm):
        return scale * torch.cos(Fm @ W + bph)

    Ztr = _rff(Ftr)
    sum_c = torch.zeros(n_cls, rff, device=dev)
    cnt_c = torch.zeros(n_cls, device=dev)
    for c in range(n_cls):
        m = ytr == c
        if m.any():
            sum_c[c] = Ztr[m].sum(0)
            cnt_c[c] = float(m.sum().item())
    ZZt = Ztr.t() @ Ztr
    mu = sum_c / cnt_c.clamp_min(1e-6).unsqueeze(1)
    n = cnt_c.sum().clamp_min(1)
    S = ZZt - (cnt_c.unsqueeze(1) * mu).t() @ mu
    cov = S / n
    cov = cov + shrink * torch.eye(rff, device=dev) * cov.diagonal().mean()
    cinv = torch.linalg.inv(cov)
    Wd = mu @ cinv
    bias = -0.5 * (Wd * mu).sum(1)
    scores = _rff(Fte) @ Wd.t() + bias
    return float((scores.argmax(1) == yte).float().mean())


# =====================================================================================================
# Datasets: a self-contained temporal SYNTHETIC (no download, for the CPU smoke) + SHD + N-MNIST wrappers
# =====================================================================================================
class _Synthetic:
    """A temporal spiking classification task where classes differ ONLY in the TIMING pattern of the same
    active input channels -- so a timing-blind (rate) encoder is near chance while a delay-aware encoder
    can separate them. Deterministic. Held fully in memory (tiny)."""

    def __init__(self, gen, n_in, n_class, T, n_train, n_test, k_active=4, jitter=1, noise=0.02):
        import torch

        self.n_in, self.n_class, self.T = n_in, n_class, T
        # the dataset lives on CPU (like a DataLoader); training moves batches to device -- NEVER build it
        # on the cuda default device, since `gen` is a CPU generator (the set_default_device trap).
        chans = (
            torch.arange(k_active, device="cpu") * max(1, n_in // (k_active + 1)) + 1
        )  # fixed active channels
        tmpl = []
        for c in range(n_class):  # per-class spike TIMES
            tmpl.append(
                (torch.rand(k_active, generator=gen, device="cpu") * (T - 2 - jitter) + 1).long()
            )
        self.tmpl, self.chans, self.jitter, self.noise = tmpl, chans, jitter, noise
        self.Xtr, self.ytr = self._make(gen, n_train)
        self.Xte, self.yte = self._make(gen, n_test)

    def _make(self, gen, n):
        import torch

        X = torch.zeros(n, self.T, self.n_in, device="cpu")
        y = torch.randint(0, self.n_class, (n,), generator=gen, device="cpu")
        for s in range(n):
            c = int(y[s])
            for i, ch in enumerate(self.chans):
                jt = int(
                    torch.randint(-self.jitter, self.jitter + 1, (1,), generator=gen, device="cpu")
                )
                t = min(self.T - 1, max(0, int(self.tmpl[c][i]) + jt))
                X[s, t, int(ch)] = 1.0
            noise = (
                torch.rand(self.T, self.n_in, generator=gen, device="cpu") < self.noise
            ).float()
            X[s] = torch.clamp(X[s] + noise, 0.0, 1.0)
        return X, y

    def train_iter(self, bs):
        yield from self._iter(self.Xtr, self.ytr, bs)

    def test_iter(self, bs):
        yield from self._iter(self.Xte, self.yte, bs)

    @staticmethod
    def _iter(X, y, bs):
        for i in range(0, len(X), bs):
            yield X[i : i + bs], y[i : i + bs]


def load_dataset(name, T, n_train, n_test, data_dir, seed=0):
    """Return a dataset object exposing .n_in/.n_class/.T and .train_iter(bs)/.test_iter(bs) (CPU tensors:
    x[B,T,nin] float spikes, y[B] long). SHD/N-MNIST stream from disk via tonic; synthetic is in-memory."""
    import torch

    if name == "synthetic":
        g = torch.Generator(device="cpu").manual_seed(seed)
        n_in = _envi("NORD_NIN", 24)
        n_class = _envi("NORD_NCLASS", 6)
        return _Synthetic(g, n_in, n_class, T, n_train or 2000, n_test or 500)

    if name == "nmnist":
        from snn_data_lake.reasoning.ng_snn.data import (
            make_nmnist_loaders,
        )  # disk-streaming N-MNIST frames [B,T,2,34,34]

        tr, te = make_nmnist_loaders(save_to=data_dir, n_time_bins=T, batch_size=128, num_workers=0)
        return _LoaderDataset(tr, te, n_in=2 * 34 * 34, n_class=10, T=T, flatten=True)

    if name == "shd":
        import tonic
        import tonic.transforms as TT
        from torch.utils.data import DataLoader

        sensor = tonic.datasets.SHD.sensor_size  # (700, 1, 1)
        to_frame = TT.ToFrame(sensor_size=sensor, n_time_bins=T)
        tf = TT.Compose([to_frame, _ToBinTensor()])
        tr_ds = tonic.datasets.SHD(save_to=data_dir, train=True, transform=tf)
        te_ds = tonic.datasets.SHD(save_to=data_dir, train=False, transform=tf)
        common = dict(batch_size=128, num_workers=0, drop_last=False)
        # DataLoader(shuffle=True)'s RandomSampler calls torch.randperm(generator=...); under
        # set_default_device(cuda) that generator must be on cuda too (the documented device trap).
        from snn_data_lake.reasoning.torch_device import resolve_device as _rdev

        gen = torch.Generator(device=_rdev())
        tr = DataLoader(tr_ds, shuffle=True, generator=gen, **common)
        te = DataLoader(te_ds, shuffle=False, **common)
        return _LoaderDataset(tr, te, n_in=700, n_class=20, T=T, flatten=True)

    raise ValueError(f"unknown NORD_DATASET={name!r} (expected synthetic|shd|nmnist)")


class _ToBinTensor:
    """Picklable frame -> binarized float32 spike tensor (module-level class so Windows num_workers>0 can
    pickle it). tonic ToFrame emits event COUNTS per bin; binarize to {0,1} to keep the input spike-driven."""

    def __call__(self, frames):
        import numpy as np
        import torch

        return (torch.from_numpy(np.asarray(frames, dtype="float32")) > 0).float()


class _LoaderDataset:
    def __init__(self, tr, te, n_in, n_class, T, flatten):
        self.tr, self.te = tr, te
        self.n_in, self.n_class, self.T, self.flatten = n_in, n_class, T, flatten

    def _wrap(self, loader, bs_ignored):
        for x, y in loader:
            xf = x.float()
            if self.flatten:
                xf = xf.reshape(xf.shape[0], xf.shape[1], -1)  # [B,T,*] -> [B,T,nin]
                xf = (xf > 0).float()
            yield xf, y.long()

    def train_iter(self, bs):
        yield from self._wrap(self.tr, bs)

    def test_iter(self, bs):
        yield from self._wrap(self.te, bs)


# =====================================================================================================
# Train / extract-features / controls
# =====================================================================================================
def train_model(model, data, device, epochs, bs, lr, scaler=None, log=""):
    """Supervised surrogate-gradient BPTT with a cross-entropy count-readout head. sigma is annealed over
    training. Optional AMP on CUDA. Returns the final test accuracy of the CE head."""
    import torch
    import torch.nn.functional as F

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    is_cuda = str(device).startswith("cuda")
    for ep in range(epochs):
        model.train()
        model.anneal_sigma((ep + 1) / max(1, epochs))
        seen = 0
        for x, y in data.train_iter(bs):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if is_cuda else "cpu", enabled=bool(scaler and is_cuda)
            ):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if scaler and is_cuda:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            seen += len(y)
        if log:
            print(
                f"  [{log}] epoch {ep + 1}/{epochs}  seen={seen}  last-loss={float(loss.detach()):.4f}"
            )
    return ce_accuracy(model, data, device, bs)


def ce_accuracy(model, data, device, bs, hard=False):
    import torch

    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, y in data.test_iter(bs):
            x, y = x.to(device), y.to(device)
            ok += int((model(x, hard=hard).argmax(1) == y).sum().item())
            tot += len(y)
    return ok / max(1, tot)


def extract_features(model, data, device, bs, hard=False):
    """Penultimate spike-count features + labels for train and test (what SLDA-RFF consumes)."""
    import torch

    model.eval()

    def _collect(it):
        Fs, ys = [], []
        with torch.no_grad():
            for x, y in it:
                Fs.append(model.features(x.to(device), hard=hard))
                ys.append(y.to(device))
        return torch.cat(Fs), torch.cat(ys)

    Ftr, ytr = _collect(data.train_iter(bs))
    Fte, yte = _collect(data.test_iter(bs))
    return Ftr, ytr, Fte, yte


def rate_collapsed_accuracy(data, device, bs, n_proj, seed=0):
    """TIMING-COLLAPSED baseline: sum input spikes over time (destroy timing) -> a fixed RANDOM ReLU
    projection -> SLDA-RFF. The faithful temporal-data stand-in for a frozen spatial (conv-STDP) encoder,
    which ignores fine timing. The trained temporal encoder should beat this = the ceiling moved."""
    import torch
    import torch.nn.functional as F

    g = torch.Generator(device="cpu").manual_seed(seed)
    Wp = None

    def _collect(it):
        nonlocal Wp
        Fs, ys = [], []
        for x, y in it:
            x = x.to(device)
            r = x.sum(dim=1)  # [B, nin] rate (timing destroyed)
            if Wp is None:
                Wp = (
                    (torch.rand(r.shape[1], n_proj, generator=g, device="cpu") * 2 - 1)
                    / math.sqrt(r.shape[1])
                ).to(device)
            Fs.append(F.relu(r @ Wp))
            ys.append(y.to(device))
        return torch.cat(Fs), torch.cat(ys)

    Ftr, ytr = _collect(data.train_iter(bs))
    Fte, yte = _collect(data.test_iter(bs))
    return Ftr, ytr, Fte, yte


# =====================================================================================================
# Export + bridge verify (mirrors jepa_scaffold_torch.export_jepa / verify_transfer)
# =====================================================================================================
def export_rsnn(model, path):
    """Write the core `snn_rsnn/1` JSON: per layer W[out][in] (NO transpose), FINALIZED integer delays,
    per-neuron beta/a/v_th, max_delay; layers in application order. Every shape asserted. Returns the dict."""

    layers = []
    for dl, al in zip(model.delays, model.alifs):
        W = dl.W.detach().cpu().tolist()
        delays = dl.int_delays().tolist()
        bias = dl.bias.detach().cpu().tolist()
        beta = al.beta().detach().cpu().tolist()
        a = al.a().detach().cpu().tolist()
        v_th = al.v_th.detach().cpu().tolist()
        nout, nin = len(W), len(W[0])
        assert all(len(r) == nin for r in W), "W rows must be [in]"
        assert len(delays) == nout and all(len(r) == nin for r in delays), (
            "delays must be [out][in]"
        )
        assert all(isinstance(d, int) and 0 <= d < dl.maxd for r in delays for d in r), (
            "delays must be int taps"
        )
        assert len(bias) == nout and len(beta) == nout and len(a) == nout and len(v_th) == nout, (
            "bias/beta/a/v_th must be [out]"
        )
        layers.append(
            {
                "W": W,
                "delays": delays,
                "max_delay": int(dl.maxd),
                "bias": bias,
                "beta": beta,
                "a": a,
                "v_th": v_th,
            }
        )
    blob = {
        "format": RSNN_FORMAT,
        "n_in": int(model.delays[0].nin),
        "rho": RHO_ADAPT,
        "layers": layers,
        "meta": {
            "trained_by": "rsnn_delays_torch",
            "learn_delays": bool(model.learn_delays),
            "adaptive": bool(model.adaptive),
        },
    }
    with open(path, "w", encoding="ascii") as f:
        json.dump(blob, f)
    return blob


def verify_transfer(path, model, n_probe=4, tol=1e-6):
    """Bridge check: load the export into the zero-dep core and confirm its forward matches a float64 CPU
    torch reference (FINALIZED integer delays, hard Heaviside) on probe spike trains. Spike COUNTS must be
    EXACT; the reference is pinned to CPU explicitly (the set_default_device trap). Skips if torch absent."""
    try:
        import torch
    except ImportError:
        print("verify_transfer: skipped (torch not installed)")
        return None
    from snn_data_lake.cortex.rsnn_encoder import RsnnEncoder

    ref = model.double().to("cpu")
    ref.eval()
    core = RsnnEncoder().load_weights(path)

    T = _envi("NORD_PROBE_T", 12)
    n_in = model.delays[0].nin
    g = torch.Generator(device="cpu").manual_seed(321)
    probes = (
        torch.rand(n_probe, T, n_in, generator=g, device="cpu") < 0.15
    ).double()  # sparse spike trains, CPU
    max_err = 0.0
    with torch.no_grad():
        counts_ref = ref.features(probes, hard=True)  # [n_probe, feat_dim] float64
    for s in range(n_probe):
        frames = [[float(v) for v in probes[s, t]] for t in range(T)]
        c_core = core.features(frames)
        c_ref = counts_ref[s].tolist()
        assert len(c_core) == len(c_ref), "feature dim mismatch core vs torch"
        for a, b in zip(c_core, c_ref):
            assert math.isfinite(a) and math.isfinite(b), "non-finite feature in bridge check"
            max_err = max(max_err, abs(a - b))
    assert max_err < tol, (
        f"export->core bridge mismatch: max |core-torch| spike-count diff = {max_err:.3e} > tol {tol:.1e}"
    )
    return {"ok": True, "max_err": max_err, "n_probe": n_probe, "feat_dim": len(c_ref)}


# =====================================================================================================
# main -- full control protocol; sizes scale from tiny (NORD_SMOKE) to real (NORD_GPU + NORD_DATASET)
# =====================================================================================================
def main():
    try:
        import torch
    except ImportError:
        print("=" * 74)
        print("RSNN DELAYS TORCH TRAINER (research/, not CI) -- torch is NOT installed")
        print("=" * 74)
        print("Torch training tier for the zero-dep core cortex/rsnn_encoder.py (snn_rsnn/1):")
        print("DCLS learnable delays + adaptive-LIF, surrogate-gradient BPTT on SHD/N-MNIST.")
        print("Without torch there is nothing to train; the zero-dep core is what CI covers.")
        print("needs torch (+tonic): pip install '.[gpu]'  (and '.[nmnist]' for shd/nmnist)")
        print("Exiting cleanly (0).")
        return 0

    import tempfile
    from snn_data_lake.reasoning.torch_device import resolve_device, enable

    smoke = os.environ.get("NORD_SMOKE") == "1"
    device = enable(resolve_device())
    is_cuda = str(device).startswith("cuda")
    torch.manual_seed(0)

    dataset = os.environ.get("NORD_DATASET", "synthetic")
    if smoke:  # tiny, standalone (no download), fast on CPU
        dataset = os.environ.get("NORD_DATASET", "synthetic")
        T = _envi("NORD_T", 16)
        hidden = [_envi("NORD_HIDDEN", 24)]
        maxd = _envi("NORD_MAXDELAY", 6)
        epochs = _envi("NORD_EPOCHS", 2)
        bs = _envi("NORD_BATCH", 64)
        n_train = _envi("NORD_TRAIN", 240)
        n_test = _envi("NORD_TEST", 120)
        rff = _envi("NORD_RFF", 256)
    else:
        T = _envi("NORD_T", 100 if dataset == "shd" else 20)
        hidden = [int(w) for w in os.environ.get("NORD_HIDDEN", "256,256").split(",")]
        maxd = _envi("NORD_MAXDELAY", 25)
        epochs = _envi("NORD_EPOCHS", 30)
        bs = _envi("NORD_BATCH", 128)
        n_train = _envi("NORD_TRAIN", 0)
        n_test = _envi("NORD_TEST", 0)
        rff = _envi("NORD_RFF", 2048)
    lr = _envf("NORD_LR", 3e-3)
    data_dir = os.environ.get("NORD_DATA_DIR", "data")

    data = load_dataset(dataset, T, n_train, n_test, data_dir)
    n_in, n_class = data.n_in, data.n_class

    print("=" * 74)
    print("RSNN DELAYS -- DCLS learnable delays + adaptive-LIF (torch trainer -> zero-dep core)")
    print("=" * 74)
    print(f"backend  : torch {torch.__version__}  device={device}  amp={is_cuda}  smoke={smoke}")
    print(f"dataset  : {dataset}  n_in={n_in}  n_class={n_class}  T={T}")
    print(
        f"model    : hidden={hidden}  max_delay={maxd}  epochs={epochs}  bs={bs}  rff={rff}  lr={lr}"
    )
    print("-" * 74)

    def _scaler():
        return torch.amp.GradScaler("cuda", enabled=is_cuda)

    def _slda(Ftr, ytr, Fte, yte):
        return slda_rff_accuracy(Ftr, ytr, Fte, yte, n_class, rff=rff)

    # ---- FULL model: delays ON + adaptive ON ----
    full = build_model(n_in, n_class, hidden, maxd, learn_delays=True, adaptive=True, seed=0).to(
        device
    )
    ce_full = train_model(full, data, device, epochs, bs, lr, _scaler(), log="full")
    ce_full_hard = ce_accuracy(full, data, device, bs, hard=True)  # finalized integer-delay re-eval
    Ftr, ytr, Fte, yte = extract_features(full, data, device, bs, hard=True)
    slda_full = _slda(Ftr, ytr, Fte, yte)

    # ---- CONTROL: RANDOM-INIT encoder (frozen, untrained) -- the critical honesty check ----
    rnd = build_model(n_in, n_class, hidden, maxd, learn_delays=True, adaptive=True, seed=7).to(
        device
    )
    Rtr, ry, Rte, rye = extract_features(rnd, data, device, bs, hard=True)
    slda_random = _slda(Rtr, ry, Rte, rye)

    # ---- CONTROL: DELAYS OFF (all delays frozen to 0), retrained ----
    nod = build_model(n_in, n_class, hidden, maxd, learn_delays=False, adaptive=True, seed=0).to(
        device
    )
    train_model(nod, data, device, epochs, bs, lr, _scaler(), log="no-delay")  # trains in place
    Dtr, dy, Dte, dye = extract_features(nod, data, device, bs, hard=True)
    slda_nodelay = _slda(Dtr, dy, Dte, dye)

    # ---- CONTROL: ADAPTIVE tau OFF (homogeneous fixed-tau plain LIF), retrained ----
    noa = build_model(n_in, n_class, hidden, maxd, learn_delays=True, adaptive=False, seed=0).to(
        device
    )
    train_model(noa, data, device, epochs, bs, lr, _scaler(), log="no-adapt")  # trains in place
    Atr, ay, Ate, aye = extract_features(noa, data, device, bs, hard=True)
    slda_noadapt = _slda(Atr, ay, Ate, aye)

    # ---- CONTROL: TIMING-COLLAPSED rate baseline (ceiling comparator) ----
    Ctr, cy, Cte, cye = rate_collapsed_accuracy(data, device, bs, n_proj=max(64, hidden[-1]))
    slda_rate = _slda(Ctr, cy, Cte, cye)

    # ---- export + bridge verify (CPU float64 reference) ----
    export_path = os.path.join(tempfile.gettempdir(), "snn_rsnn1_export.json")
    blob = export_rsnn(full, export_path)
    # bridge reference = full's TRAINED weights copied into a fresh CPU model (verify converts it to float64
    # in place, so it must be a copy of `full`, not `full` itself, and not a fresh untrained model).
    ref_cpu = build_model(n_in, n_class, hidden, maxd, learn_delays=True, adaptive=True, seed=0)
    ref_cpu.load_state_dict(full.state_dict())
    bridge = verify_transfer(export_path, ref_cpu)

    d_delay = slda_full - slda_nodelay
    d_adapt = slda_full - slda_noadapt
    d_random = slda_full - slda_random
    d_ceiling = slda_full - slda_rate

    print("-" * 74)
    print("RESULTS -- SLDA-RFF accuracy on penultimate spike-count features (held-out)")
    print(f"  CE head (soft delays)                 : {ce_full:.1%}")
    print(f"  CE head (FINALIZED integer delays)    : {ce_full_hard:.1%}")
    print("-" * 74)
    print(f"  TRAINED encoder (delays+adaptive)     : {slda_full:.1%}   <- ceiling")
    print(
        f"  control: delays OFF (retrained)       : {slda_nodelay:.1%}   delta(delays)   = {d_delay:+.1%}"
    )
    print(
        f"  control: adaptive tau OFF (retrained) : {slda_noadapt:.1%}   delta(adaptive) = {d_adapt:+.1%}"
    )
    print(
        f"  control: RANDOM-INIT (frozen)         : {slda_random:.1%}   delta(vs random)= {d_random:+.1%}"
    )
    print(
        f"  control: TIMING-COLLAPSED rate        : {slda_rate:.1%}   delta(ceiling)  = {d_ceiling:+.1%}"
    )
    print("-" * 74)
    fd = bridge["feat_dim"] if bridge and bridge.get("ok") else "?"
    if bridge and bridge.get("ok"):
        print(
            f"  bridge   : core snn_rsnn/1 == torch forward (feat_dim={fd}, max count-diff "
            f"{bridge['max_err']:.1e} over {bridge['n_probe']} probes)"
        )
    else:
        print("  bridge   : NOT confirmed")
    print(f"  exported : {RSNN_FORMAT} -> {export_path}  ({len(blob['layers'])} layers)")
    print("=" * 74)
    print(
        "HONEST NOTES: expect delta(delays) >= +3% and delta(vs random) large on TEMPORAL data (SHD); on"
    )
    print(
        "static/near-linear tasks delays and the trained-vs-random margin can be small -- reported as-is."
    )
    print(
        "The rate baseline is the temporal stand-in for a frozen conv-STDP spatial encoder (image-only)."
    )

    # asserts: only the RIGGABLE-if-broken plumbing invariants (never the honest deltas, which are data-dependent)
    assert bridge is not None and bridge.get("ok"), "export->core bridge must match (plumbing)"
    assert 0.0 <= slda_full <= 1.0 and 0.0 <= slda_random <= 1.0, (
        "accuracies must be valid fractions"
    )
    assert ce_full_hard > 0.0, "finalized-delay CE head must run"
    print(
        f"self-check OK: bridge matches (max count-diff {bridge['max_err']:.1e}); trained SLDA-RFF "
        f"{slda_full:.1%} vs random {slda_random:.1%} (delta {d_random:+.1%}), delays delta {d_delay:+.1%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
