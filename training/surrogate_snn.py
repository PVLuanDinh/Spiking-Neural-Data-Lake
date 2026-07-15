"""
training/surrogate_snn.py -- surrogate-gradient deep spiking classifier on N-MNIST (event data),
the FOUNDATION of the hybrid PyTorch perception tier for the Spiking Neural Data Lake brain.

WHAT THIS IS (honest framing). A REAL surrogate-gradient SNN trainer: a small spiking conv net
(snntorch Leaky/LIF layers, fast-sigmoid surrogate) trained by BPTT-through-time on N-MNIST event
frames, with a rate/spike-count cross-entropy readout, plus a clean `SurrogateEncoder` the brain's
perception hook can call to turn an event frame into a latent vector (the penultimate spiking layer's
activity). It is the trained-perception FOUNDATION for the hybrid tier -- a working trainer + encoder
interface -- NOT a SOTA result. Any accuracy printed is the number ACTUALLY obtained at the small
scale it was run at; nothing here is extrapolated to full-dataset / long-training numbers.

WHAT THIS IS NOT (dependency direction, non-negotiable). This is the OPTIONAL torch [train]/[gpu]
tier. The zero-dep stdlib safety-core NEVER imports this module; the dependency is one-directional
(perception tier -> torch/snntorch, and nothing in src/snn_data_lake/ or brain.py imports back). When
torch or trained weights are absent, the brain degrades to its stdlib perception path -- this tier is
a bolt-on, not a requirement. This module imports NOTHING from brainstem/ or the zero-dep core, and
importing it has NO side effects (data loading + training are lazy; only `--smoke`/CLI or an explicit
call runs anything). Runs in .venv-cu128 on the RTX 5070 (sm_120); GPU training is inherently
stochastic (expected + fine for the torch tier, unlike the deterministic core).

Data loading is REUSED, not reinvented: the N-MNIST disk-streaming loader lives in
research/ng_snn/data.py (tonic DiskCachedDataset -> [B, T, 2, 34, 34] frames). It is imported lazily
so this module (and `SurrogateEncoder`) stay importable without tonic / the dataset present.

  PYTHONUTF8=1 .venv-cu128/Scripts/python training/surrogate_snn.py --smoke
      # 1 tiny synthetic batch through fwd/bwd + encoder round-trip (CPU-ok, no dataset needed)
  PYTHONUTF8=1 .venv-cu128/Scripts/python training/surrogate_snn.py --epochs 1 --limit 2000
      # short real N-MNIST train on a small subset -> prints test accuracy (needs tonic + dataset)

ASCII-only output (Windows cp1252-safe). Seeds are set where possible; GPU kernels are not bit-exact.
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# N-MNIST event geometry (matches research/ng_snn/data.py SENSOR = (34, 34, 2)).
SENSOR_H = 34
SENSOR_W = 34
IN_CH = 2
NUM_CLASSES = 10
DEFAULT_T = 15          # event time-bins; matches data.py DEFAULT_T (set from the paper for real runs)
DEFAULT_HIDDEN = 128    # penultimate spiking-layer width == the latent dimension the encoder emits
DEFAULT_BETA = 0.9      # LIF membrane decay
DEFAULT_SLOPE = 25.0    # fast-sigmoid surrogate steepness


# ----------------------------------------------------------------------------
# The spiking classifier: 2 spiking conv blocks -> spiking FC (latent) -> spiking readout.
# init_hidden=False: membrane state is threaded explicitly so the encoder can read the penultimate
# layer's spike activity cleanly. Readout = spike COUNT per class over T (rate-coded), fed to CE.
# ----------------------------------------------------------------------------
class SpikingConvNet(nn.Module):
    def __init__(self, in_ch=IN_CH, num_classes=NUM_CLASSES, hidden=DEFAULT_HIDDEN,
                 beta=DEFAULT_BETA, slope=DEFAULT_SLOPE, sensor_h=SENSOR_H, sensor_w=SENSOR_W):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.in_ch = int(in_ch)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.beta = float(beta)
        self.slope = float(slope)
        self.sensor_h = int(sensor_h)
        self.sensor_w = int(sensor_w)

        self.conv1 = nn.Conv2d(in_ch, 12, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        flat = self._flat_dim()                              # derive FC input from a shape-only pass
        self.fc1 = nn.Linear(flat, self.hidden)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)   # PENULTIMATE spiking layer == latent
        self.fc2 = nn.Linear(self.hidden, self.num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)   # spiking readout

    def _flat_dim(self):
        """Flatten width after the two conv+pool blocks (shape-only, no LIF, no grad)."""
        with torch.no_grad():
            d = torch.zeros(1, self.in_ch, self.sensor_h, self.sensor_w)
            d = self.pool1(self.conv1(d))
            d = self.pool2(self.conv2(d))
            return int(d.flatten(1).shape[1])

    def forward(self, x):
        """x: [B, T, C, H, W] event frames. Returns (spk_out_sum [B, num_classes],
        latent_sum [B, hidden]) -- both are spike COUNTS summed over the T time-bins. spk_out_sum are
        the class logits (rate readout); latent_sum is the penultimate spiking-layer activity."""
        b, t = x.shape[0], x.shape[1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk_out_sum = torch.zeros(b, self.num_classes, device=x.device)
        latent_sum = torch.zeros(b, self.hidden, device=x.device)
        for step in range(t):
            xt = x[:, step]
            cur1 = self.pool1(self.conv1(xt))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)               # penultimate spikes -> latent
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk_out_sum = spk_out_sum + spk4
            latent_sum = latent_sum + spk3
        return spk_out_sum, latent_sum

    def config(self):
        return dict(in_ch=self.in_ch, num_classes=self.num_classes, hidden=self.hidden,
                    beta=self.beta, slope=self.slope, sensor_h=self.sensor_h, sensor_w=self.sensor_w)


# ----------------------------------------------------------------------------
# The perception interface the brain's hook calls. Lazy: constructing/loading an encoder does NOT
# train and does NOT require tonic or the dataset -- only torch + the saved weights.
# ----------------------------------------------------------------------------
class SurrogateEncoder:
    """Wraps a trained SpikingConvNet as a perception encoder.

    encode(event_frame) -> latent vector (the penultimate spiking layer's spike-count activity):
      * event_frame [T, C, H, W]      -> latent [hidden]        (single sample; brain-hook path)
      * event_frame [B, T, C, H, W]   -> latent [B, hidden]     (batched)
    Accepts a numpy array or a torch tensor. Values are spike counts over the T window (>= 0).

    load(path, device=...) rebuilds the net from the checkpoint's saved config and weights -- no
    training, no dataset needed. This is the interface the brain perception hook imports and calls;
    when torch or the checkpoint is absent the brain uses its stdlib perception path instead."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt.get("config", {})
        net = SpikingConvNet(**cfg)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    @torch.no_grad()
    def encode(self, event_frame):
        x = event_frame
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().to(self.device)
        single = (x.ndim == 4)                               # [T, C, H, W] -> add batch dim
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError("encode expects [T,C,H,W] or [B,T,C,H,W]; got shape %s" % (tuple(x.shape),))
        _, latent = self.net(x)
        return latent[0] if single else latent


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


# ----------------------------------------------------------------------------
# Train / eval.
# ----------------------------------------------------------------------------
def evaluate(net, loader, device, max_samples=None):
    """Test accuracy = argmax of the spike-count readout. Caps at max_samples for quick runs."""
    net.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            spk_out, _ = net(x)
            correct += int((spk_out.argmax(1) == y).sum().item())
            seen += int(y.shape[0])
            if max_samples is not None and seen >= max_samples:
                break
    return (100.0 * correct / seen) if seen else 0.0, seen


def train_and_eval(args):
    device = args.device
    torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = _load_data_module()
    print("device=%s  T=%d  batch=%d  hidden=%d  limit=%s  epochs=%d"
          % (device, args.T, args.batch_size, args.hidden, str(args.limit), args.epochs))
    if device == "cuda" and torch.cuda.is_available():
        print("gpu=%s  torch=%s" % (torch.cuda.get_device_name(0), torch.__version__))

    train_loader, test_loader = data.make_nmnist_loaders(
        n_time_bins=args.T, batch_size=args.batch_size, num_workers=args.workers)
    print("train_batches=%d  test_batches=%d" % (len(train_loader), len(test_loader)))

    net = SpikingConvNet(hidden=args.hidden, beta=args.beta, slope=args.slope).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print("model: 2 spiking-conv blocks -> spiking FC(%d) latent -> spiking readout | params=%d"
          % (args.hidden, n_params))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        net.train()
        seen = 0
        running = 0.0
        nb = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()        # tonic labels are int32; CE needs int64
            spk_out, _ = net(x)
            loss = loss_fn(spk_out, y)                        # CE on spike-count logits (rate readout)
            opt.zero_grad()
            loss.backward()                                   # BPTT through the T unrolled time-bins
            opt.step()
            running += float(loss.item())
            nb += 1
            seen += int(y.shape[0])
            if nb % 20 == 0:
                print("  epoch %d  batch %d  seen %d  loss %.4f" % (epoch, nb, seen, running / nb))
            if args.limit is not None and seen >= args.limit:
                break
        acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
        print("epoch %d done: train_samples=%d  avg_loss=%.4f  test_acc=%.2f%% (on %d test samples)"
              % (epoch, seen, running / max(nb, 1), acc, ntest))

    acc, ntest = evaluate(net, test_loader, device, max_samples=args.eval_limit)
    print("=" * 72)
    print("SURROGATE-SNN (hybrid perception tier FOUNDATION) | device=%s" % device)
    print("test accuracy %.2f%% on %d N-MNIST test samples (T=%d, %d train samples, %d epoch(s))"
          % (acc, ntest, args.T, min(args.limit or 10 ** 12, 60000), args.epochs))
    print("HONEST: small-scale foundation number, not SOTA and not extrapolated. Torch [train]/[gpu]")
    print("tier ONLY -- the zero-dep core never imports this; brain falls back to stdlib perception")
    print("when torch/weights are absent. Encoder latent dim = %d (penultimate spiking activity)."
          % args.hidden)
    print("=" * 72)

    if args.save:
        torch.save({"model": net.state_dict(), "config": net.config()}, args.save)
        print("saved encoder checkpoint -> %s" % args.save)
    return acc


# ----------------------------------------------------------------------------
# Smoke: prove fwd/bwd + encoder round-trip on 1 tiny synthetic batch, NO dataset, CPU-ok.
# ----------------------------------------------------------------------------
def run_smoke(device):
    torch.manual_seed(0)
    t_steps, b = 4, 2
    print("smoke: device=%s  synthetic batch=[B=%d, T=%d, C=%d, %dx%d] (no dataset needed)"
          % (device, b, t_steps, IN_CH, SENSOR_H, SENSOR_W))
    net = SpikingConvNet(hidden=32).to(device)               # tiny latent for a fast smoke
    x = (torch.rand(b, t_steps, IN_CH, SENSOR_H, SENSOR_W, device=device) < 0.1).float()  # sparse events
    y = torch.randint(0, NUM_CLASSES, (b,), device=device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    spk_out, latent = net(x)
    loss = loss_fn(spk_out, y)
    opt.zero_grad()
    loss.backward()                                          # proves surrogate-grad backward path
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    opt.step()
    print("  forward: logits=%s  latent=%s  loss=%.4f  finite-grads=%s"
          % (tuple(spk_out.shape), tuple(latent.shape), float(loss.item()), grad_ok))

    # encoder round-trip via the real load(path) interface (tempfile, then deleted).
    enc_dir = tempfile.mkdtemp()
    ckpt = os.path.join(enc_dir, "smoke_encoder.pt")
    torch.save({"model": net.state_dict(), "config": net.config()}, ckpt)
    enc = SurrogateEncoder.load(ckpt, device=device)
    single = enc.encode(x[0])                                # [T,C,H,W] -> [hidden]
    batched = enc.encode(x)                                  # [B,T,C,H,W] -> [B,hidden]
    os.remove(ckpt)
    os.rmdir(enc_dir)
    print("  encoder.encode([T,C,H,W]) -> %s ; encode([B,T,C,H,W]) -> %s"
          % (tuple(single.shape), tuple(batched.shape)))

    assert spk_out.shape == (b, NUM_CLASSES), "logits shape"
    assert latent.shape == (b, 32), "latent shape"
    assert grad_ok, "backward produced no finite grads"
    assert tuple(single.shape) == (32,), "single-sample latent must be [hidden]"
    assert tuple(batched.shape) == (b, 32), "batched latent must be [B, hidden]"
    print("SMOKE OK: surrogate-grad fwd/bwd + encoder load/encode path verified (no dataset, %s)."
          % device)


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Surrogate-gradient spiking classifier on N-MNIST "
                                            "(hybrid perception tier foundation).")
    p.add_argument("--smoke", action="store_true",
                   help="run 1 tiny synthetic batch through fwd/bwd + encoder (CPU-ok, no dataset)")
    p.add_argument("--epochs", type=int, default=1, help="training epochs")
    p.add_argument("--limit", type=int, default=None,
                   help="cap training samples per epoch (small-subset quick run)")
    p.add_argument("--eval-limit", type=int, default=2000, dest="eval_limit",
                   help="cap test samples during eval (quick runs); None-like large value for full")
    p.add_argument("--T", type=int, default=DEFAULT_T, help="event time-bins per sample")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="penultimate/latent width")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="LIF membrane decay")
    p.add_argument("--slope", type=float, default=DEFAULT_SLOPE, help="fast-sigmoid surrogate slope")
    p.add_argument("--lr", type=float, default=1e-3)
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
