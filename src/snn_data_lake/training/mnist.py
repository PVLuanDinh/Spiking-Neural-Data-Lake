"""Fast surrogate-gradient MNIST brain trainer (torch + snnTorch) for the GPU tier.

Trains a 784 -> H -> 10 spiking network whose forward dynamics match
``snn_data_lake.cortex.trained_brain`` EXACTLY, then exports a zero-dep JSON "brain" the
core can serve with NO torch present. Because the trainer's LIF hidden layer and
accumulate-current output are the SAME dynamics ``trained_brain.predict`` runs, the
learned weights transfer faithfully -- and that is proven after export by re-scoring a
held-out MNIST subset through the zero-dep ``trained_brain.accuracy`` (the honesty win:
both the torch test accuracy AND the core-served accuracy are reported).

Dynamics matched to trained_brain (beta = DEFAULT_BETA = 0.9, threshold = DEFAULT_THRESHOLD = 1.0):
  * input pixels in [0, 1] are RATE-CODED over STEPS: pixel i spikes with prob x_i each step.
  * HIDDEN layer is leaky integrate-and-fire: m = m*beta + W1 . spikes_in each step; it emits a
    spike and RESETS TO ZERO when m >= threshold (snnTorch Leaky, reset_mechanism="zero").
  * OUTPUT layer ACCUMULATES weighted hidden-spike currents over all steps (no leak, no
    threshold): out[c] = sum_t sum_h W2[c][h] * hidden_spike_h(t); prediction = argmax(out).

``nn.Linear.weight`` is [out][in], so ``fc1.weight`` is [H][784] and ``fc2.weight`` is
[10][H] -- exactly the ``layers=[W1, W2]`` chain (784 -> H -> 10) that
``trained_brain.load_brain`` validates. ``.tolist()`` on each maps 1:1 into a brain with no
transpose; the shapes are asserted before saving.

SPEED: mixed precision (``torch.amp`` autocast + GradScaler -> fp16 tensor cores on the
5070), Adam, and a minibatched DataLoader. This trains in MINUTES on a 5070, not days.

torch / snntorch / torchvision are imported at module top HERE on purpose: this module is
only ever imported when actually training (from cli.main() behind an import guard), never at
package import time -- so ``import snn_data_lake`` in a plain zero-dep install stays clean.
"""
import os

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms

from snn_data_lake.cortex.trained_brain import (
    DEFAULT_BETA,
    DEFAULT_THRESHOLD,
    accuracy as core_accuracy,
    make_brain,
    save_brain,
)

DEFAULT_HIDDEN = 256
DEFAULT_STEPS = 25
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 128
IN_FEATURES = 784
OUT_CLASSES = 10


# ---- model: 784 -> H (LIF) -> 10 (accumulate) -- IS trained_brain.predict's forward -------
class MnistBrain(nn.Module):
    """Two-layer spiking net whose forward reproduces ``trained_brain.predict`` exactly.

    ``fc1.weight`` is [H][784] and ``fc2.weight`` is [10][H] -- the ``layers=[W1, W2]``
    chain (784 -> H -> 10) that ``trained_brain.load_brain`` validates, so exporting is a
    direct ``.tolist()`` per layer with no transpose.
    """

    def __init__(self, hidden=DEFAULT_HIDDEN, beta=DEFAULT_BETA, threshold=DEFAULT_THRESHOLD):
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, hidden, bias=False)
        # hidden LIF: leaky integrate, spike + reset-to-zero at threshold -- exactly the
        # core's `m = m*beta + cur; if m >= thr: fire; m = 0`.
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism="zero",
            spike_grad=surrogate.fast_sigmoid(),
        )
        self.fc2 = nn.Linear(hidden, OUT_CLASSES, bias=False)

    def forward(self, spikes):
        """spikes: [STEPS, batch, 784] binary. Returns [batch, 10] accumulated output currents."""
        mem = None
        out = None
        for t in range(spikes.shape[0]):
            cur1 = self.fc1(spikes[t])                   # [batch, H] hidden LIF input current
            if mem is None:
                mem = torch.zeros_like(cur1)
            spk1, mem = self.lif(cur1, mem)              # hidden spikes; membrane resets to 0 on spike
            cur2 = self.fc2(spk1)                        # [batch, 10] weighted hidden-spike current
            out = cur2 if out is None else out + cur2    # OUTPUT accumulates over all steps (no leak/thr)
        return out


# ---- helpers ----------------------------------------------------------------
def _generator(device, seed):
    """A device-bound Generator so the stochastic rate coding is reproducible."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def rate_encode(X, steps, gen, device):
    """Rate-code X[n, 784] in [0, 1] into a spike train [steps, n, 784].

    Pixel i fires with prob x_i per timestep -- EXACTLY ``trained_brain``'s rate coding
    (spike probability == input value), so the trained net sees the input statistics the
    zero-dep core will serve.
    """
    prob = X.unsqueeze(0).expand(steps, -1, -1)          # [steps, n, 784] (broadcast view)
    noise = torch.rand(prob.shape, generator=gen, device=device)
    return (noise < prob).float()


def _datasets(root, download=True):
    """(train, test) torchvision MNIST datasets. ToTensor scales pixels to [0, 1]."""
    tf = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=download, transform=tf)
    test = datasets.MNIST(root=root, train=False, download=download, transform=tf)
    return train, test


def _flatten(images, device, non_blocking=False):
    """[b, 1, 28, 28] -> [b, 784] float in [0, 1] on `device`.

    `non_blocking` enables an async H2D copy; only helps when the source batch is in
    pinned host memory (pin_memory=True), so callers pass it True only on CUDA.
    """
    return images.view(images.shape[0], -1).to(device, non_blocking=non_blocking)


# ---- train / evaluate -------------------------------------------------------
def train_model(epochs=DEFAULT_EPOCHS, hidden=DEFAULT_HIDDEN, steps=DEFAULT_STEPS,
                device="cpu", seed=0, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE,
                beta=DEFAULT_BETA, threshold=DEFAULT_THRESHOLD, root=None, download=True):
    """Train the 784 -> H -> 10 spiking brain with surrogate-gradient BPTT + Adam + AMP.

    Mixed precision is enabled on CUDA (fp16 tensor cores); on CPU the autocast / GradScaler
    become no-ops so the exact same code path runs (just in fp32). Returns (model, metrics)
    and is deterministic for a given seed/device.
    """
    torch.manual_seed(int(seed))
    root = root or os.path.join(".", "data", "mnist")
    use_amp = str(device).startswith("cuda")

    train_ds, test_ds = _datasets(root, download=download)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(seed))
    # DataLoader tuning is only worth it on CUDA: worker processes decode/collate batches
    # while the GPU computes, and pinned host buffers give faster (async) H2D copies. On CPU
    # keep num_workers=0 / pin_memory=False so the path is byte-identical to before. The main
    # process still owns the seeded `generator`, so the shuffle order is unchanged either way.
    num_workers = 4 if use_amp else 0                 # GPU-VERIFY
    pin_memory = use_amp                              # GPU-VERIFY
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen,
        num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    model = MnistBrain(hidden=hidden, beta=beta, threshold=threshold).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    amp_dev = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler(amp_dev, enabled=use_amp)
    gen = _generator(device, seed)

    last_loss_t = None
    for _epoch in range(int(epochs)):
        model.train()
        for images, targets in train_loader:
            X = _flatten(images, device, non_blocking=use_amp)   # pinned async H2D on CUDA  # GPU-VERIFY
            y = targets.to(device, non_blocking=use_amp)                                      # GPU-VERIFY
            spikes = rate_encode(X, steps, gen, device)          # [steps, b, 784]
            opt.zero_grad()
            with torch.amp.autocast(amp_dev, enabled=use_amp):   # fp16 on the 5070 tensor cores
                out = model(spikes)
                loss = loss_fn(out, y)                           # CrossEntropy on accumulated output
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            last_loss_t = loss.detach()                          # keep on-device; no per-step host sync
    # Sync the final loss to host ONCE after all training (was a per-step .cpu() that stalled
    # the GPU every step). Value is identical: the last batch's loss of the final epoch.
    last_loss = float("nan") if last_loss_t is None else float(last_loss_t)

    train_acc = evaluate(model, train_loader, steps, device=device, seed=seed, limit=10000)
    test_acc = evaluate(model, test_loader, steps, device=device, seed=seed + 1)

    metrics = {
        "epochs": int(epochs),
        "device": str(device),
        "hidden": int(hidden),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "amp": bool(use_amp),
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "final_loss": round(last_loss, 6),
    }
    return model, metrics


def evaluate(model, loader, steps, device="cpu", seed=1234, limit=None):
    """Accuracy of `model` over a DataLoader. Uses its own seeded generator so eval is
    reproducible independent of training; `limit` caps images scored (fast train-set sanity)."""
    model.eval()
    nb = str(device).startswith("cuda")                          # GPU-VERIFY: async H2D only on CUDA
    gen = _generator(device, seed)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            X = _flatten(images, device, non_blocking=nb)
            y = targets.to(device, non_blocking=nb)
            spikes = rate_encode(X, steps, gen, device)
            pred = model(spikes).argmax(dim=1)
            correct += int((pred == y).sum().cpu())
            total += int(y.shape[0])
            if limit is not None and total >= limit:
                break
    return correct / max(1, total)


# ---- checkpointing (torch.save / torch.load) --------------------------------
def save_checkpoint(model, path):
    """Persist the torch model (state_dict + shape metadata) for resuming."""
    torch.save(
        {
            "format": "snn_training.MnistBrain/1",
            "state_dict": model.state_dict(),
            "in_features": model.fc1.in_features,
            "hidden": model.fc1.out_features,
            "out_classes": model.fc2.out_features,
        },
        path,
    )
    return path


def load_checkpoint(path, beta=DEFAULT_BETA, threshold=DEFAULT_THRESHOLD):
    """Rebuild an MnistBrain from a checkpoint written by save_checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model = MnistBrain(hidden=ckpt["hidden"], beta=beta, threshold=threshold)
    model.load_state_dict(ckpt["state_dict"])
    return model


# ---- THE BRIDGE: export torch weights into a zero-dep trained_brain ---------
def export_brain(model, path, steps=DEFAULT_STEPS, beta=DEFAULT_BETA,
                 threshold=DEFAULT_THRESHOLD, meta=None):
    """Export the trained weights as a zero-dep ``trained_brain`` JSON at `path`.

    ``fc1.weight`` is [H][784] and ``fc2.weight`` is [10][H] (nn.Linear.weight is [out][in]),
    exactly the ``layers=[W1, W2]`` chain (784 -> H -> 10) trained_brain validates. The shapes
    are ASSERTED before writing, then ``make_brain`` + ``save_brain`` write the file the core
    serves with no torch. Returns the brain dict.
    """
    W1 = model.fc1.weight.detach().cpu().tolist()        # [H][784]
    W2 = model.fc2.weight.detach().cpu().tolist()        # [10][H]
    hidden = model.fc1.out_features
    assert len(W1) == hidden and all(len(row) == IN_FEATURES for row in W1), (
        f"fc1.weight must be [{hidden}][{IN_FEATURES}], got [{len(W1)}][{len(W1[0]) if W1 else 0}]"
    )
    assert len(W2) == OUT_CLASSES and all(len(row) == hidden for row in W2), (
        f"fc2.weight must be [{OUT_CLASSES}][{hidden}], got [{len(W2)}][{len(W2[0]) if W2 else 0}]"
    )
    m = {"source": "snn_data_lake.training.mnist surrogate-gradient GPU tier",
         "task": "mnist", "hidden": hidden, "steps": int(steps)}
    if meta:
        m.update(meta)
    brain = make_brain([W1, W2], steps=steps, beta=beta, threshold=threshold, meta=m)
    save_brain(brain, path)
    return brain


def verify_transfer(brain, n=200, root=None, download=True):
    """Prove the export transfers into the core: score `n` held-out MNIST test images through
    the ZERO-DEP ``trained_brain.accuracy`` and return that core-served accuracy.

    Compare it to the torch test accuracy -- they should be CLOSE. They are not required to be
    identical: the trainer and the core each rate-code the input with their own RNG, and rate
    coding is robust to that (the underlying weights + dynamics are the same). Each image is
    handed over as an ``(x_list, label)`` pair with x_list a length-784 vector in [0, 1].
    """
    root = root or os.path.join(".", "data", "mnist")
    _, test_ds = _datasets(root, download=download)
    n = min(int(n), len(test_ds))
    subset = []
    for i in range(n):
        image, label = test_ds[i]                        # image [1, 28, 28] in [0, 1]
        subset.append(([float(v) for v in image.view(-1)], int(label)))
    return core_accuracy(brain, subset)
