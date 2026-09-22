"""Surrogate-gradient spiking classifier (Torch + Brian2 equation) for the GPU tier.

Targets the EXACT task the zero-dep core defines. The data + task are REUSED from
``snn_data_lake.cortex.snn_classifier`` (they are zero-dep, so importing them is free):
8x8 binary shapes, GRID=8 / N=64 inputs, C=4 classes, STEPS=20 rate-coded
timesteps (ON_RATE=0.9 lit / OFF_RATE=0.05 dark per step).

Model: a SINGLE spiking linear readout -- ``nn.Linear(N, C, bias=False)`` feeding
``BrianLIF`` membranes, unrolled over STEPS timesteps on rate-coded input
spikes. The class score is the summed output spikes. Because the readout is a
single input->class linear map, ``fc.weight`` has shape [C][N] and maps 1:1 onto
the core's ``W[c][i]`` matrix -- that is what makes ``export_to_core`` a faithful
bridge back into the zero-dep install.

Training uses the Brian2-backed fast-sigmoid surrogate bridge with BPTT and Adam. Seeding is
deterministic: ``torch.manual_seed`` plus a device-bound ``torch.Generator`` for
the stochastic rate coding.

Torch/Brian2 are imported at module top HERE on purpose: this module is only
ever imported when actually training (from cli.main() behind an import guard, or
from offload.py), never at package import time.
"""
import random

import torch
import torch.nn as nn
from snn_data_lake.substrate.brian2_backend import BrianLIF, fast_sigmoid

from snn_data_lake.cortex.snn_classifier import (
    templates,
    dataset,
    save_weights,
    GRID,
    N,
    STEPS,
    ON_RATE,
    OFF_RATE,
    TRAIN_PER_CLASS,
    TEST_PER_CLASS,
)

DEFAULT_BETA = 0.9      # Leaky membrane decay
DEFAULT_LR = 0.05
DEFAULT_EPOCHS = 50


# ---- model: single spiking linear readout (weight is [C][N] -> maps 1:1 to core W)
class SpikingReadout(nn.Module):
    """One input->class linear synapse feeding Leaky membranes, unrolled in time.

    ``self.fc.weight`` has shape [n_classes, n_inputs] == [C][N], the same layout as
    the core classifier's W[c][i], so the export bridge is a direct copy.
    """

    def __init__(self, n_inputs=N, n_classes=4, beta=DEFAULT_BETA):
        super().__init__()
        self.fc = nn.Linear(n_inputs, n_classes, bias=False)
        self.lif = BrianLIF(beta=beta, spike_grad=fast_sigmoid())

    def forward(self, spikes):
        """spikes: [STEPS, batch, N] binary. Returns [batch, C] summed output spikes."""
        mem = None
        spk_sum = None
        for t in range(spikes.shape[0]):
            cur = self.fc(spikes[t])                 # [batch, C]
            if mem is None:
                mem = torch.zeros_like(cur)
            spk, mem = self.lif(cur, mem)
            spk_sum = spk if spk_sum is None else spk_sum + spk
        return spk_sum


# ---- helpers ----------------------------------------------------------------
def _generator(device, seed):
    """A device-bound Generator so the stochastic rate coding is reproducible."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _to_tensors(data, device):
    """(sample, label) list -> (X[n,N] float, y[n] long) on `device`."""
    X = torch.tensor([s for s, _ in data], dtype=torch.float32, device=device)
    y = torch.tensor([lab for _, lab in data], dtype=torch.long, device=device)
    return X, y


def rate_encode(X, steps, gen, device):
    """Rate-code X[n,N] in {0,1} into a spike train [steps, n, N].

    Each pixel fires with prob ON_RATE if lit else OFF_RATE per timestep -- exactly
    the core's encode_step probabilities, so the trained readout sees the same input
    statistics the zero-dep core will serve.
    """
    prob = X * ON_RATE + (1.0 - X) * OFF_RATE            # [n, N]
    prob = prob.unsqueeze(0).expand(steps, -1, -1)       # [steps, n, N]
    noise = torch.rand(prob.shape, generator=gen, device=device)
    return (noise < prob).float()


# ---- train / evaluate -------------------------------------------------------
def train_model(epochs=DEFAULT_EPOCHS, device="cpu", seed=0, lr=DEFAULT_LR, beta=DEFAULT_BETA):
    """Train the spiking readout with surrogate-gradient BPTT + Adam.

    Returns (model, metrics). Deterministic for a given seed/device.
    """
    torch.manual_seed(int(seed))
    random.seed(int(seed))                                # core dataset() uses the global RNG

    temps = templates()
    n_classes = len(temps)
    train_data = dataset(TRAIN_PER_CLASS, temps)
    test_data = dataset(TEST_PER_CLASS, temps)
    Xtr, ytr = _to_tensors(train_data, device)

    model = SpikingReadout(N, n_classes, beta=beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = _generator(device, seed)

    last_loss = float("nan")
    for _epoch in range(int(epochs)):
        model.train()
        spikes = rate_encode(Xtr, STEPS, gen, device)    # fresh stochastic spikes / epoch
        opt.zero_grad()
        out = model(spikes)
        loss = loss_fn(out, ytr)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    train_acc = evaluate(model, Xtr, ytr, device=device)
    Xte, yte = _to_tensors(test_data, device)
    test_acc = evaluate(model, Xte, yte, device=device)

    metrics = {
        "epochs": int(epochs),
        "device": str(device),
        "n_classes": n_classes,
        "n_inputs": N,
        "steps": STEPS,
        "grid": GRID,
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "final_loss": round(last_loss, 6),
    }
    return model, metrics


def evaluate(model, X, y, device="cpu", seed=1234):
    """Accuracy of `model` on (X, y). Uses its own seeded generator so eval is
    reproducible independent of how many training epochs ran."""
    model.eval()
    gen = _generator(device, seed)
    with torch.no_grad():
        spikes = rate_encode(X, STEPS, gen, device)
        pred = model(spikes).argmax(dim=1)
        return float((pred == y).float().mean().cpu())


# ---- checkpointing (torch.save / torch.load) --------------------------------
def save_checkpoint(model, path):
    """Persist the torch model (state_dict + shape metadata) for resuming."""
    torch.save(
        {
            "format": "snn_training.SpikingReadout/1",
            "state_dict": model.state_dict(),
            "n_inputs": model.fc.in_features,
            "n_classes": model.fc.out_features,
        },
        path,
    )
    return path


def load_checkpoint(path, beta=DEFAULT_BETA):
    """Rebuild a SpikingReadout from a checkpoint written by save_checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model = SpikingReadout(ckpt["n_inputs"], ckpt["n_classes"], beta=beta)
    model.load_state_dict(ckpt["state_dict"])
    return model


# ---- THE BRIDGE: export torch weights into the zero-dep core -----------------
def export_to_core(model, path, meta=None):
    """Shape the trained ``nn.Linear(N, C)`` weight to the core's W = list[C][N] of
    python floats and write it with the core's ``save_weights`` (imported above), so
    ``snn_classifier.load_weights(path)`` can serve it in a zero-dep, torch-free
    install.

    ``fc.weight`` is [out_features, in_features] == [n_classes, n_inputs] == [C][N],
    so ``.tolist()`` already yields exactly the core's W[c][i] layout -- a 1:1 map,
    no transpose needed. The shape is asserted before writing.

    HONEST CAVEAT on dynamics: the export is FORMAT-faithful, but the two inference
    paths are not identical. This trainer optimizes LIF output-spike dynamics
    (leaky membrane + threshold), while the zero-dep core serves a LINEAR
    spike-count readout (``currents[c] = sum_i W[c][i] * count[i]``, argmax). So the
    accuracy the core gets from these weights may differ from the trainer's test
    accuracy -- treat the export as a GPU-optimized weight INIT and VERIFY it in the
    core after export: ``W = snn_classifier.load_weights(path)`` then evaluate with
    ``snn_classifier.forward``/``evaluate``.
    """
    W = model.fc.weight.detach().cpu().tolist()          # [C][N]
    n_classes, n_inputs = model.fc.out_features, model.fc.in_features
    assert len(W) == n_classes, f"expected {n_classes} rows, got {len(W)}"
    assert all(len(row) == n_inputs == N for row in W), (
        f"each row must be N={N} inputs; got {[len(r) for r in W][:3]}"
    )
    m = {"source": "snn_data_lake.training surrogate-gradient GPU tier"}
    if meta:
        m.update(meta)
    return save_weights(W, path, meta=m)
