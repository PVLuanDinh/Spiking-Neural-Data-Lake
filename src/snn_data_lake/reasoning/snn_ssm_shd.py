"""
Spiking State-Space Model (deep, trained) — the SOTA-track companion to the zero-dep LSM.

RESEARCH TRACK — needs torch+snntorch (SHD path also needs tonic); runs in its own venv, NOT in CI.
Validated on CPU: the synthetic long-range task reaches 100%; real SHD via Tonic reaches 32% (20-class,
chance 5%) on a capped 500-sample / 12-epoch run — full-SHD SOTA needs a GPU box with uncapped,
longer training. The zero-dep, CI-covered mechanism is spiking_ssm.py.

Deep spiking SSMs are the current best for long event sequences (SpikingSSMs, AAAI 2025; P-SpikeSSM,
ICLR 2025; FLAMES, arXiv:2504.01257): a linear state-space recurrence h_t = A⊙h_{t-1} + B·x_t carries
long-range memory cheaply, a LIF spiking nonlinearity on C·h_t keeps activations sparse/binary, and the
whole stack is trained by BPTT with snnTorch's surrogate gradient. This file trains a small stacked
spiking-SSM classifier and reports accuracy + spike SPARSITY + SynOps (piped through energy.py).

Data: a synthetic long-range temporal task by default (delayed-cue classification — memory must span the
sequence), or the real **SHD** (Spiking Heidelberg Digits, 20-class spoken digits) via Tonic with --shd.
Env knobs: SSM_DIM, SSM_LAYERS, SSM_STEPS, SSM_EPOCHS.

Run (own venv, from repo root so snn_data_lake resolves):
  PYTHONPATH=src .venv-gpu/Scripts/python research/snn_ssm_shd.py            # synthetic long-range task
  PYTHONPATH=src .venv-gpu/Scripts/python research/snn_ssm_shd.py --shd      # real SHD via Tonic
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import snntorch as snn
from snntorch import surrogate
from snn_data_lake.reasoning.torch_device import resolve_device, enable

_DEVICE = enable(resolve_device())

DIM = int(os.environ.get("SSM_DIM", 128))
LAYERS = int(os.environ.get("SSM_LAYERS", 2))
STEPS = int(os.environ.get("SSM_STEPS", 50))      # sequence length (time bins)
EPOCHS = int(os.environ.get("SSM_EPOCHS", 25))
BATCH = int(os.environ.get("SSM_BATCH", 32))
SHD_CAP = int(os.environ.get("SSM_SHD_CAP", 500))   # cap SHD samples for a feasible CPU validation run
SEED = 0


class SpikingSSMBlock(nn.Module):
    """Diagonal linear state-space recurrence + LIF spiking readout (surrogate-grad trainable).
    A = exp(-softplus(·)) keeps the per-channel decay in (0,1) — the long-range memory knob."""

    def __init__(self, dim, spike_grad):
        super().__init__()
        self.log_a = nn.Parameter(torch.rand(dim) * 2 - 1)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)
        self.lif = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):                         # x: [batch, steps, dim] -> spikes [batch, steps, dim]
        a = torch.exp(-F.softplus(self.log_a))    # diagonal decay in (0,1)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        mem = self.lif.init_leaky()
        out, self.spike_count = [], 0.0
        for t in range(x.size(1)):
            h = a * h + self.B(x[:, t])           # SSM linear recurrence (carries long-range memory)
            spk, mem = self.lif(self.C(h), mem)   # binary spiking nonlinearity
            self.spike_count = self.spike_count + spk.sum().item()
            out.append(spk)
        return torch.stack(out, dim=1)


class SpikingSSM(nn.Module):
    def __init__(self, in_dim, n_classes, dim=DIM, layers=LAYERS):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.embed = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([SpikingSSMBlock(dim, spike_grad) for _ in range(layers)])
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, x):                          # x: [batch, steps, in_dim]
        h = self.embed(x)
        self.synops = 0.0
        for blk in self.blocks:
            h = blk(h)
            self.synops += blk.spike_count * blk.C.out_features   # spikes × fan-out
        return self.readout(h.mean(dim=1))         # pool spikes over time -> logits


def synthetic_task(n, steps, in_dim, n_classes, gen):
    """Delayed-cue classification: a class-specific cue appears early, then noise — the label can
    only be read out if the model REMEMBERS the cue across the whole sequence (long-range memory)."""
    x = torch.rand(n, steps, in_dim, generator=gen) * 0.1
    y = torch.randint(0, n_classes, (n,), generator=gen)
    for i in range(n):
        x[i, :5, y[i] % in_dim] += 1.5             # early cue in a class-specific channel
    return x, y


def load_shd(train):
    """Real SHD via Tonic (guarded). Returns (frames [N,steps,channels], labels). Capped to
    SHD_CAP samples so a CPU validation run is feasible (full SHD needs a GPU box)."""
    import tonic  # noqa: PLC0415
    ds = tonic.datasets.SHD(save_to="./data", train=train,
                            transform=tonic.transforms.ToFrame(
                                sensor_size=tonic.datasets.SHD.sensor_size, n_time_bins=STEPS))
    n = min(SHD_CAP if train else max(1, SHD_CAP // 4), len(ds))
    xs, ys = [], []
    for i in range(n):
        frame, lab = ds[i]
        xs.append(torch.as_tensor(frame, dtype=torch.float32).reshape(STEPS, -1))  # -> [steps, channels]
        ys.append(int(lab))
    return torch.stack(xs), torch.tensor(ys)


def run(x, y, xt, yt, n_classes):
    torch.manual_seed(SEED)
    g = torch.Generator(device=_DEVICE).manual_seed(SEED)
    model = SpikingSSM(x.size(2), n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(EPOCHS):                        # minibatch SGD (many updates, not one/epoch)
        perm = torch.randperm(x.size(0), generator=g)
        for k in range(0, x.size(0), BATCH):
            idx = perm[k:k + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = model(xt)
        acc = (logits.argmax(1) == yt).float().mean().item()
        sparsity = 1.0 - model.synops / (xt.size(0) * STEPS * DIM * LAYERS * DIM)
    return acc, model.synops / xt.size(0), sparsity


def main():
    gen = torch.Generator(device=_DEVICE).manual_seed(SEED)
    if "--shd" in sys.argv:
        mode = "real SHD (Tonic)"
        x, y = load_shd(True)
        xt, yt = load_shd(False)
        n_classes = int(max(y.max(), yt.max()).item()) + 1
    else:
        mode = "synthetic long-range"
        in_dim, n_classes = 16, 4
        x, y = synthetic_task(256, STEPS, in_dim, n_classes, gen)
        xt, yt = synthetic_task(64, STEPS, in_dim, n_classes, gen)

    acc, synops, sparsity = run(x, y, xt, yt, n_classes)

    try:
        from snn_data_lake.energy import compare   # noqa: PLC0415
        # fair baseline = the dense-SSM recurrence a non-spiking model runs EVERY step for ALL units
        # (the C matmul: layers × steps × dim × dim); the spiking model only pays for actual spikes.
        e = compare(int(synops), LAYERS * STEPS * DIM * DIM)
        energy = f"{e['snn_nj']:.1f} nJ vs dense {e['ann_nj']:.1f} nJ ({e['energy_ratio']:.1f}x)"
    except ImportError:
        energy = "(run with PYTHONPATH=src to pipe SynOps through energy.py)"

    print("=" * 60)
    print(f"SPIKING STATE-SPACE MODEL (deep, trained) — {mode}")
    print("=" * 60)
    print(f"dim={DIM} layers={LAYERS} steps={STEPS} epochs={EPOCHS} classes={n_classes}")
    print(f"test accuracy : {acc:.1%}   (chance {1 / n_classes:.0%})")
    print(f"spike sparsity: {sparsity:.1%}   SynOps/seq: {synops:,.0f}")
    print(f"energy        : {energy}")
    print("=" * 60)

    # ---- self-check (runs only in a torch venv; UNVERIFIED on the CI/dev box) ----
    assert acc > 1.5 / n_classes, f"spiking SSM should beat chance on the memory task: {acc:.2f}"
    assert 0.0 <= sparsity <= 1.0, "sparsity must be a fraction"
    assert synops > 0, "no synaptic operations recorded"
    print("self-check OK: trained spiking SSM classifies the long-range task above chance")


if __name__ == "__main__":
    main()
