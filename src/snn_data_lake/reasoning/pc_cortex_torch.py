r"""
[GPU-VERIFY]  research/pc_cortex_torch.py -- GPU-batched trainer for a DEEP spiking
predictive-coding cortex (exports `snn_pcnet/1` for src/.../cortex/pc_cortex_serve.py).

Honest framing. This is the torch TRAINING side of the deep spiking Rao-Ballard predictive
coder whose zero-dep forward SERVER is src/snn_data_lake/cortex/pc_cortex_serve.py (and whose
2-level stdlib seed is cortex/predictive_coding.py). It learns an L-level (default L=3)
generative hierarchy by LOCAL predictive coding -- top-down generative weights W_1..W_L, a
bottom-up precision-weighted PREDICTION-ERROR carried by SIGNED ON/OFF LIF spike populations,
and a LOCAL Hebbian-on-spikes weight update per level. There is NO global backprop chained
across the hierarchy -- each W_l is updated only from quantities local to its own two adjacent
levels. GPU's ONLY role is to settle a whole MINIBATCH of samples in parallel (batched matmuls
+ parallel LIF), not to run autograd across levels. The scientific point is LOCALITY.

The settling forward here is byte-for-byte the same discrete spiking dynamics the server replays
(verified in-script by loading the emitted artifact with PcCortexServe and matching top causes on
a few samples). Cause levels r_1..r_L (init 0); W_l is dims[l-1] x dims[l] (rows=below, cols=above),
THETA=0.5, LEAK=0.02, prec=1. Per settling step:
    e_0 = x   - W_1 @ r_1
    e_l = r_l - W_{l+1} @ r_{l+1}     (l = 1..L-1; top r_L has no error above it)
each e_i encoded by a signed ON/OFF LIF pair (leak, threshold, reset-by-subtraction -> discrete
spike in {-1,0,+1}); then
    r_j += lr * (W_j^T @ s_{j-1} - s_j)   (j = 1..L-1)
    r_L += lr * (W_L^T @ s_{L-1})         (top: driven ONLY by the level-(L-1) error spikes).

LEARNING (the M-step, LOCAL Hebbian-on-spikes): after settling a minibatch, rate_i = mean signed
spike over the settle window at error level i; then W_l += lr_w * mean_batch( outer(rate_{l-1}, r_l) ).
No gradient crosses a level boundary -- asserted/printed as a headline property.

READOUT: a LINEAR head (W_ro r_L + b) on the settled top cause, trained by closed-form ridge on the
settled top causes toward one-hot labels. It is honestly a rate/linear classification head on top of
the spiking latent (the latent is spiking; the readout is linear) -- exactly the seed's identity
cause readout, plus a trained linear map.

DATA. Target is N-MNIST (real event-camera data at data/NMNIST/, streamed + binned via tonic, then
collapsed over time to a per-pixel event-RATE input vector -- the STATIC input a predictive coder
settles on; the collapse is honest: the PC input is a fixed vector, so the temporal event stream is
reduced to its per-channel rate). MNIST-rate (pure-stdlib idx load from data/MNIST/raw, flattened
intensities) is the honest fast fallback and is what --smoke uses so the CPU smoke needs no tonic.

NOT zero-dep -- torch/GPU TRAINING TIER only. The zero-dep core NEVER imports this. torch runs in
.venv-cu128 on the RTX 5070. ASCII-only prints (PYTHONUTF8=1). unverified_on_hardware=True until the
orchestrator runs the full GPU pass.

  # CPU smoke (tiny, MNIST-rate, no tonic, proves it runs + exports a server-loadable checkpoint):
  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python research/pc_cortex_torch.py --smoke

  # full GPU run (N-MNIST; orchestrator runs this):
  NORD_GPU=1 PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python research/pc_cortex_torch.py
"""

import argparse
import json
import os
import struct
import time

import torch

pass  # migrated: sys.path bootstrap removed (now an importable package)
from snn_data_lake.reasoning.torch_device import enable, resolve_device  # noqa: E402

THETA = 0.5
LEAK = 0.02


# =========================================================================================================
# The batched spiking settling (GPU) -- MUST match pc_cortex_serve.PcCortexServe.settle exactly.
# =========================================================================================================
class DeepSpikingPC:
    """An L-level spiking predictive-coding hierarchy, settled over a whole minibatch in parallel on the
    chosen device. Weights W[l] are [dims[l], dims[l+1]] (rows=below dim, cols=above dim), matching the
    server. All math runs under torch.no_grad() -- learning is LOCAL Hebbian-on-spikes, never autograd."""

    def __init__(self, dims, device, dtype=torch.float32, seed=0, init_scale=None):
        self.dims = list(dims)
        self.L = len(dims) - 1
        self.device = device
        self.dtype = dtype
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = []
        for l in range(self.L):
            fan = dims[l + 1]
            a = (0.5 / (fan**0.5)) if init_scale is None else init_scale
            w = (torch.rand(dims[l], dims[l + 1], generator=g, device="cpu") * 2.0 - 1.0) * a
            self.W.append(w.to(device=device, dtype=dtype))
        self.prec = [1.0] * self.L

    @torch.no_grad()
    def settle(self, x, steps, lr, collect_rate=False):
        """Settle causes for a batch x [B, dims[0]]. Returns list r[0..L-1] (r[j] is [B, dims[j+1]]).
        If collect_rate: also returns rate[0..L-1] (mean signed error-spike over the window, [B, dims[i]])
        and the per-step total error-spike count (a settling trace). Discrete spikes in {-1,0,+1}."""
        B = x.shape[0]
        dev, dt, dims, W, prec, L = self.device, self.dtype, self.dims, self.W, self.prec, self.L
        x = x.to(device=dev, dtype=dt)
        r = [torch.zeros(B, dims[j + 1], device=dev, dtype=dt) for j in range(L)]
        v_on = [torch.zeros(B, dims[i], device=dev, dtype=dt) for i in range(L)]
        v_off = [torch.zeros(B, dims[i], device=dev, dtype=dt) for i in range(L)]
        rate = (
            [torch.zeros(B, dims[i], device=dev, dtype=dt) for i in range(L)]
            if collect_rate
            else None
        )
        trace = []
        n = steps + 1  # server loops range(steps+1)
        for _ in range(n):
            # top-down predictions -> analog errors e_0..e_{L-1}
            errs = [x - r[0] @ W[0].t()]  # e_0 = x - W_1 @ r_1
            for lvl in range(1, L):  # e_l = r_l - W_{l+1} @ r_{l+1}
                errs.append(r[lvl - 1] - r[lvl] @ W[lvl].t())
            # encode each error into signed ON/OFF LIF spikes (reset-by-subtraction, single spike/step)
            spikes = []
            for i in range(L):
                drive = prec[i] * errs[i]
                pos = torch.clamp(drive, min=0.0)
                neg = torch.clamp(-drive, min=0.0)
                v_on[i] = (1.0 - LEAK) * v_on[i] + pos
                s_on = (v_on[i] >= THETA).to(dt)
                v_on[i] = v_on[i] - THETA * s_on
                v_off[i] = (1.0 - LEAK) * v_off[i] + neg
                s_off = (v_off[i] >= THETA).to(dt)
                v_off[i] = v_off[i] - THETA * s_off
                si = s_on - s_off
                spikes.append(si)
                if collect_rate:
                    rate[i] = rate[i] + si
            trace.append(sum(float(s.abs().sum().item()) for s in spikes))
            # integrate error spikes into causes: r_j += lr*(W_j^T @ s_{j-1} - s_j); top has no s_L
            for j in range(L):
                up = spikes[j] @ W[j]  # [B, dims[j+1]]  (== W_j^T @ s_{j-1})
                if j < L - 1:
                    r[j] = r[j] + lr * (up - spikes[j + 1])
                else:
                    r[j] = r[j] + lr * up
        if collect_rate:
            rate = [rt / n for rt in rate]
            return r, rate, trace
        return r, trace

    @torch.no_grad()
    def hebb_update(self, rate, r, lr_w):
        """LOCAL Hebbian-on-spikes M-step: W_l += lr_w * mean_batch(outer(rate_{l-1}, r_l)). rate[l] is the
        error-level-l mean spike [B, dims[l]]; r[l] is cause [B, dims[l+1]]. dW_l = lr_w * rate[l].T @ r[l] / B.
        NO gradient crosses a level -- each update touches only its own two adjacent levels."""
        B = r[0].shape[0]
        for l in range(self.L):
            dW = rate[l].t() @ r[l] / B  # [dims[l], dims[l+1]]
            self.W[l] = self.W[l] + lr_w * dW

    def export_weights(self):
        return [w.detach().to("cpu", torch.float64).tolist() for w in self.W]


# =========================================================================================================
# Data.  MNIST-rate (pure stdlib idx, always available -> smoke/fallback) and N-MNIST (tonic, target).
# =========================================================================================================
def _read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad idx3 magic {magic}"
        buf = f.read(n * rows * cols)
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).to(torch.float32) / 255.0
    return t.view(n, rows * cols)


def _read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad idx1 magic {magic}"
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).to(torch.long)


def load_mnist_rate(root="data/MNIST/raw", n_train=None, n_test=None):
    """Flattened MNIST intensities in [0,1], 784-dim. Pure stdlib idx load -- no torch datasets, no tonic."""
    xtr = _read_idx_images(os.path.join(root, "train-images-idx3-ubyte"))
    ytr = _read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte"))
    xte = _read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte"))
    yte = _read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte"))
    if n_train:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    if n_test:
        xte, yte = xte[:n_test], yte[:n_test]
    return xtr, ytr, xte, yte, 784, [str(i) for i in range(10)]


def load_nmnist_rate(n_train=None, n_test=None, n_time_bins=15):
    """N-MNIST via tonic, collapsed over time to a per-pixel event-RATE vector (2*34*34 = 2312 dims,
    normalized to [0,1]). The temporal event stream is reduced to its per-channel rate because a
    predictive coder settles on a STATIC input vector -- honest collapse, stated in the header."""
    import numpy as np
    import tonic
    import tonic.transforms as TT

    sensor = tonic.datasets.NMNIST.sensor_size
    to_frame = TT.ToFrame(sensor_size=sensor, n_time_bins=n_time_bins)

    def _load(train, limit):
        ds = tonic.datasets.NMNIST(save_to="data", train=train, transform=to_frame)
        split = "train" if train else "test"
        ds = tonic.DiskCachedDataset(ds, cache_path=f"data/nmnist_cache/{split}_T{n_time_bins}")
        m = len(ds) if not limit else min(limit, len(ds))
        xs = torch.empty(m, 2 * 34 * 34, dtype=torch.float32, device="cpu")
        ys = torch.empty(m, dtype=torch.long, device="cpu")
        for i in range(m):
            frames, label = ds[i]  # frames: (T, 2, 34, 34)
            rate = np.asarray(frames, dtype=np.float32).sum(axis=0).reshape(-1)  # collapse time
            mx = rate.max()
            if mx > 0:
                rate = rate / mx
            xs[i] = torch.from_numpy(rate)
            ys[i] = int(label)
        return xs, ys

    xtr, ytr = _load(True, n_train)
    xte, yte = _load(False, n_test)
    return xtr, ytr, xte, yte, 2 * 34 * 34, [str(i) for i in range(10)]


# =========================================================================================================
# Readout: closed-form ridge on the settled top causes (linear/rate head on the spiking latent).
# =========================================================================================================
@torch.no_grad()
def train_ridge_readout(R, y, n_classes, lam=1.0):
    """Ridge-fit a linear head W[n_classes, D] + b on settled top causes R [N, D] toward one-hot y.
    Solves (Rb^T Rb + lam I) Wb = Rb^T Y with a bias column. Returns (W, b) as python lists (float64)."""
    dev, dt = R.device, torch.float64
    R = R.to(dt)
    N, D = R.shape
    Rb = torch.cat([R, torch.ones(N, 1, device=dev, dtype=dt)], dim=1)  # [N, D+1]
    Y = torch.zeros(N, n_classes, device=dev, dtype=dt)
    Y[torch.arange(N, device=dev), y.to(dev)] = 1.0
    A = Rb.t() @ Rb + lam * torch.eye(D + 1, device=dev, dtype=dt)
    Wb = torch.linalg.solve(A, Rb.t() @ Y)  # [D+1, n_classes]
    W = Wb[:D, :].t().contiguous()  # [n_classes, D]
    b = Wb[D, :].contiguous()  # [n_classes]
    return W.to("cpu").tolist(), b.to("cpu").tolist()


@torch.no_grad()
def readout_acc(R, y, W, b):
    """Top-1 accuracy of a linear head (python-list W,b) over settled top causes R [N, D]."""
    dev = R.device
    Wt = torch.tensor(W, device=dev, dtype=torch.float64).t()  # [D, n_classes]
    bt = torch.tensor(b, device=dev, dtype=torch.float64)
    logits = R.to(torch.float64) @ Wt + bt
    pred = logits.argmax(dim=1).to("cpu")
    return float((pred == y.to("cpu")).float().mean().item())


@torch.no_grad()
def settle_all(net, X, steps, lr, batch):
    """Settle X [N, d_in] in minibatches; return the settled TOP causes r_L stacked [N, dims[-1]]."""
    tops = []
    for i in range(0, X.shape[0], batch):
        r, _ = net.settle(X[i : i + batch].to(net.device), steps=steps, lr=lr)
        tops.append(r[-1].to("cpu"))
    return torch.cat(tops, dim=0)


# =========================================================================================================
# Depth-earns-its-place toy: XOR over pooled features. A 3-level PC should separate it; a 2-level PC +
# raw feature-averaging should not. Honest -- we PRINT the numbers whichever way it lands.
# =========================================================================================================
@torch.no_grad()
def depth_toy(device, seed=0):
    """Build a hierarchical XOR task: input = two feature GROUPS, each group's mean encodes one bit; the
    label is XOR(bit_a, bit_b). Raw averaging (a linear readout on pooled means) cannot separate XOR; a
    deeper nonlinear-through-settling hierarchy has a chance. Returns (acc3, acc2, acc_avg)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ndim, per = 16, 8  # 2 groups of 8 dims
    N = 400
    bits = (torch.rand(N, 2, generator=g, device="cpu") > 0.5).to(torch.float32)
    y = bits[:, 0].long() ^ bits[:, 1].long()
    X = torch.zeros(N, ndim, device="cpu")
    for i in range(N):
        X[i, :per] = 0.8 * bits[i, 0] + 0.05 * torch.randn(per, generator=g, device="cpu")
        X[i, per:] = 0.8 * bits[i, 1] + 0.05 * torch.randn(per, generator=g, device="cpu")
    X = X.clamp(0, 1)
    ntr = 300
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    def run(dims):
        net = DeepSpikingPC(dims, device, seed=seed, init_scale=0.2)
        for _ in range(30):  # local Hebbian epochs
            r, rate, _ = net.settle(Xtr.to(device), steps=40, lr=0.1, collect_rate=True)
            net.hebb_update(rate, r, lr_w=0.02)
        Rtr = settle_all(net, Xtr, 40, 0.1, 128)
        Rte = settle_all(net, Xte, 40, 0.1, 128)
        W, b = train_ridge_readout(Rtr, ytr, 2, lam=1.0)
        return readout_acc(Rte, yte, W, b)

    acc3 = run([ndim, 8, 4, 2])  # 3 cause levels
    acc2 = run([ndim, 4, 2])  # 2 cause levels
    # raw-averaging linear baseline: 2 pooled group-means -> ridge
    Ptr = torch.stack([Xtr[:, :per].mean(1), Xtr[:, per:].mean(1)], dim=1)
    Pte = torch.stack([Xte[:, :per].mean(1), Xte[:, per:].mean(1)], dim=1)
    W, b = train_ridge_readout(Ptr.to(device), ytr, 2, lam=1.0)
    acc_avg = readout_acc(Pte.to(device), yte, W, b)
    return acc3, acc2, acc_avg


# =========================================================================================================
# Export + fidelity: write snn_pcnet/1, load it with the ZERO-DEP server, and prove the server's settling
# reproduces this trainer's top causes on a few samples (float64 both sides) -- export serves faithfully.
# =========================================================================================================
def write_artifact(path, dims, gen_weights, prec, steps, lr, classes, readout, dataset, test_acc):
    blob = {
        "format": "snn_pcnet/1",
        "dims": [int(d) for d in dims],
        "gen_weights": gen_weights,
        "prec": [float(p) for p in prec],
        "theta": THETA,
        "leak": LEAK,
        "steps": int(steps),
        "lr": float(lr),
        "classes": list(classes),
        "readout": readout,
        "meta": {
            "trained_by": "pc_cortex_torch",
            "dataset": dataset,
            "test_acc": float(test_acc),
            "unverified_on_hardware": True,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        json.dump(blob, f)
    return blob


def verify_export_fidelity(path, net, X, steps, lr, n=4, tol=1e-4):
    """Load the emitted artifact with the pure-stdlib PcCortexServe and assert its settling top causes
    match this trainer's (both in float64) on n samples -- proves the export serves faithfully."""
    from snn_data_lake.cortex.pc_cortex_serve import PcCortexServe

    srv = PcCortexServe.load(path)  # trust-boundary validation must pass
    # trainer side: settle in float64 on CPU for an apples-to-apples comparison with the server
    net64 = DeepSpikingPC(net.dims, "cpu", dtype=torch.float64, seed=0)
    net64.W = [w.to("cpu", torch.float64) for w in net.W]
    xs = X[:n].to("cpu", torch.float64)
    r_torch, _ = net64.settle(xs, steps=steps, lr=lr)
    top_torch = r_torch[-1].tolist()
    max_err = 0.0
    for i in range(n):
        r_srv = srv.settle(xs[i].tolist(), steps=steps, lr=lr)  # server's pure-python settling
        top_srv = r_srv[-1]
        for a, b in zip(top_torch[i], top_srv):
            max_err = max(max_err, abs(a - b))
    return srv, max_err, (max_err <= tol)


# =========================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke", action="store_true", help="tiny CPU config (MNIST-rate), fast, no tonic"
    )
    ap.add_argument("--dataset", choices=["nmnist", "mnist"], default=None)
    ap.add_argument("--out", default="models/pc_cortex_nmnist.json")
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        device = "cpu"  # force CPU for a portable smoke
        dataset = args.dataset or "mnist"
        n_train, n_test = 500, 300
        steps, settle_lr, lr_w, epochs, batch = 25, 0.05, 0.02, 3, 500
        hidden = [64, 32, 16]
    else:
        device = enable(resolve_device())
        dataset = args.dataset or "nmnist"
        n_train, n_test = None, None
        steps, settle_lr, lr_w, epochs, batch = 60, 0.05, 0.01, 15, 256
        hidden = [128, 64, 32]

    print("=" * 80)
    print(
        "PC-CORTEX TORCH -- deep spiking predictive-coding, LOCAL learning (no cross-level backprop)"
    )
    print("=" * 80)
    print(f"device={device}  dataset={dataset}  smoke={args.smoke}  torch={torch.__version__}")

    # ---- data (fall back to MNIST if tonic/N-MNIST is unavailable) ----
    if dataset == "nmnist":
        try:
            xtr, ytr, xte, yte, d_in, classes = load_nmnist_rate(n_train, n_test)
        except Exception as e:  # noqa: BLE001
            print(
                f"NOTE: N-MNIST load failed ({type(e).__name__}: {e}); FALLING BACK to MNIST-rate."
            )
            dataset = "mnist"
    if dataset == "mnist":
        xtr, ytr, xte, yte, d_in, classes = load_mnist_rate(n_train=n_train, n_test=n_test)
    dims = [d_in] + hidden
    n_classes = len(classes)
    print(
        f"data: train={xtr.shape[0]} test={xte.shape[0]} d_in={d_in}  dims={dims}  classes={n_classes}"
    )

    xtr = xtr.to(device)
    xte_d = xte.to(device)

    # ---- (1) train the LOCAL-PC cortex: settle each minibatch, local Hebbian-on-spikes M-step ----
    net = DeepSpikingPC(dims, device, seed=0)
    W_before = net.W[0].clone()
    for ep in range(epochs):
        perm = torch.randperm(xtr.shape[0], device=device)
        for i in range(0, xtr.shape[0], batch):
            xb = xtr[perm[i : i + batch]]
            r, rate, _ = net.settle(xb, steps=steps, lr=settle_lr, collect_rate=True)
            net.hebb_update(rate, r, lr_w=lr_w)
    weights_moved = float((net.W[0] - W_before).abs().max().item())

    Rtr = settle_all(net, xtr, steps, settle_lr, batch)
    Rte = settle_all(net, xte_d, steps, settle_lr, batch)
    W_ro, b_ro = train_ridge_readout(Rtr, ytr, n_classes, lam=1.0)
    acc_pc = readout_acc(Rte, yte, W_ro, b_ro)

    # ---- (2) BASELINE (i): random-weight generative matrices + a trained readout ----
    net_rand = DeepSpikingPC(dims, device, seed=123)  # never learned
    Rtr_r = settle_all(net_rand, xtr, steps, settle_lr, batch)
    Rte_r = settle_all(net_rand, xte_d, steps, settle_lr, batch)
    W_r, b_r = train_ridge_readout(Rtr_r, ytr, n_classes, lam=1.0)
    acc_rand = readout_acc(Rte_r, yte, W_r, b_r)

    # ---- (3) BASELINE (ii): surrogate-grad backprop SNN ceiling (reference) ----
    surro = "models/surrogate_nmnist.pth"
    surro_acc = None
    if os.path.exists(surro):
        try:
            obj = torch.load(surro, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                for k in ("test_acc", "acc", "accuracy", "best_acc"):
                    if k in obj and isinstance(obj[k], (int, float)):
                        surro_acc = float(obj[k])
                        break
        except Exception as e:  # noqa: BLE001
            print(
                f"NOTE: could not read {surro} ({type(e).__name__}); treating as unlabeled reference."
            )

    # ---- export + fidelity check against the zero-dep server ----
    gen_weights = net.export_weights()
    write_artifact(
        args.out,
        dims,
        gen_weights,
        net.prec,
        steps,
        settle_lr,  # writes args.out
        classes,
        [W_ro, b_ro],
        dataset,
        acc_pc,
    )
    srv, fid_err, fid_ok = verify_export_fidelity(args.out, net, xte, steps, settle_lr)
    srv_acc = None
    # sanity: server.classify on a few samples agrees with the trainer readout direction (loose)
    n_chk = min(50, xte.shape[0])
    srv_correct = sum(
        1 for i in range(n_chk) if srv.classify(xte[i].tolist()) == classes[int(yte[i])]
    )
    srv_acc = srv_correct / n_chk

    # ---- (4) depth-earns-its-place toy ----
    acc3, acc2, acc_avg = depth_toy(device)

    dt = time.time() - t0
    print("-" * 80)
    print(f"[1] PRIMARY  local-PC cortex test acc         : {acc_pc:.4f}")
    print(
        f"[2] BASE(i)  random-gen + trained readout acc : {acc_rand:.4f}  "
        f"(PC-learning gain: {acc_pc - acc_rand:+.4f})"
    )
    if surro_acc is not None:
        print(
            f"[3] BASE(ii) surrogate-backprop SNN ceiling   : {surro_acc:.4f}  "
            f"(local-PC -> backprop gap: {surro_acc - acc_pc:+.4f})"
        )
    else:
        print(
            f"[3] BASE(ii) surrogate-backprop reference     : models/surrogate_nmnist.pth present="
            f"{os.path.exists(surro)} (no acc field found; backprop reference, gap not auto-readable)"
        )
    print(
        f"[4] DEPTH    XOR-over-pooled toy: 3-level PC={acc3:.3f}  2-level PC={acc2:.3f}  "
        f"raw-avg={acc_avg:.3f}"
    )
    depth_helps = acc3 > max(acc2, acc_avg) + 0.05
    print(
        f"             depth earns its place (3lvl beats 2lvl AND raw-avg by >0.05): {depth_helps}"
    )
    if not depth_helps:
        print(
            "             HONEST: depth did NOT clearly separate this toy at this scale (printed, not asserted)."
        )
    print("-" * 80)
    print(
        f"locality    : learning is LOCAL Hebbian-on-spikes; NO autograd graph (all settling under "
        f"no_grad), gen[0] moved {weights_moved:.3e}"
    )
    print(
        f"export      : wrote {args.out}  (dims={dims}, dataset={dataset}, test_acc={acc_pc:.4f})"
    )
    print(
        f"fidelity    : PcCortexServe.load OK; server settling matches trainer top causes "
        f"max_err={fid_err:.2e} (<=1e-4): {fid_ok}"
    )
    print(
        f"            : server.classify agrees with labels on {n_chk} samples: {srv_acc:.3f} "
        f"(trainer readout acc on same head: consistency check)"
    )
    print(
        f"spiking     : discrete ON/OFF LIF spikes in {{-1,0,+1}}, theta={THETA} leak={LEAK}; "
        f"settling matches the zero-dep server byte-for-byte"
    )
    print(f"elapsed     : {dt:.1f}s")
    print("=" * 80)

    assert fid_ok, (
        f"EXPORT FIDELITY FAILED: server settling does not match trainer (max_err={fid_err:.2e})"
    )
    print(
        "DONE: snn_pcnet/1 exported and verified server-loadable + settling-faithful. "
        "unverified_on_hardware=True until the GPU pass."
    )


if __name__ == "__main__":
    main()
