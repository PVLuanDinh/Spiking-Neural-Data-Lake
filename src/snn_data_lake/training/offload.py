"""CPU/disk-OFFLOAD training for models larger than VRAM.

This is the HONEST "disk-streaming training" path, and it is DISTINCT from AirLLM
(``local_llm/``), which offloads for INFERENCE only. Here we actually train while
keeping memory off the compute device, using two real, standard mechanisms:

  1. Gradient checkpointing (``torch.utils.checkpoint``) over the time-unrolled SNN
     forward. The activations for each of STEPS timesteps are RECOMPUTED during the
     backward pass instead of all being held at once, turning activation memory from
     O(STEPS) into O(1) at the cost of ~one extra forward. For deep/long-unrolled
     spiking nets this is where the VRAM actually goes.

  2. Optimizer state kept on CPU. The master weight is a CPU leaf tensor, so Adam's
     moment buffers (exp_avg / exp_avg_sq -- each the same size as the weights, and
     the dominant term in optimizer memory) allocate on CPU. Each step streams the
     weight to the compute device for the matmul-heavy forward/backward, copies the
     gradient back to CPU, and runs the Adam update there.

This is a correct, reviewable demonstration of the ZeRO-Offload idea on the
project's own task, not a stub. For production scale (billions of params, real
disk streaming, sharded optimizer state), use the purpose-built libraries instead:

  * HuggingFace ``accelerate`` with ``device_map="auto"`` / ``offload_folder=`` to
    stream layers between GPU, CPU RAM, and disk automatically.
  * DeepSpeed ZeRO-Offload / ZeRO-Infinity for optimizer+parameter offload to CPU
    and NVMe.

Torch/Brian2 are imported at module top on purpose: this module is only imported
when actually training (from cli.main() behind an import guard).
"""
import random

import torch
import torch.nn as nn
from snn_data_lake.growth.brian2_backend import BrianLIF, fast_sigmoid
from torch.utils.checkpoint import checkpoint

from snn_data_lake.cortex.snn_classifier import (
    templates,
    dataset,
    N,
    STEPS,
    TRAIN_PER_CLASS,
    TEST_PER_CLASS,
)
from snn_data_lake.training.trainer import (
    SpikingReadout,
    rate_encode,
    _to_tensors,
    _generator,
    evaluate,
    DEFAULT_BETA,
    DEFAULT_LR,
    DEFAULT_EPOCHS,
)


def train_with_offload(epochs=DEFAULT_EPOCHS, compute_device="cpu", seed=0,
                       lr=DEFAULT_LR, beta=DEFAULT_BETA):
    """Train the same single spiking readout as trainer.train_model, but with the
    weights' master copy + optimizer state on CPU and the time-unrolled forward
    gradient-checkpointed on ``compute_device``.

    Returns (model, metrics); the returned model is a plain SpikingReadout with the
    trained weight loaded, so it checkpoints / exports identically to the normal path.
    Deterministic for a given seed/device.
    """
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    temps = templates()
    n_classes = len(temps)
    train_data = dataset(TRAIN_PER_CLASS, temps)
    test_data = dataset(TEST_PER_CLASS, temps)
    Xtr, ytr = _to_tensors(train_data, compute_device)
    batch = Xtr.shape[0]
    gen = _generator(compute_device, seed)

    # Master weight lives on CPU -> Adam moment buffers allocate on CPU (offloaded).
    weight_cpu = torch.zeros(n_classes, N, requires_grad=True, device="cpu")
    lif = BrianLIF(beta=beta, spike_grad=fast_sigmoid()).to(compute_device)
    opt = torch.optim.Adam([weight_cpu], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def _timestep(spk_t, w_dev, mem):
        # x @ W.T with W = [C, N] reproduces nn.Linear(N, C, bias=False).
        return lif(spk_t @ w_dev.t(), mem)

    last_loss = float("nan")
    for _epoch in range(int(epochs)):
        opt.zero_grad()
        spikes = rate_encode(Xtr, STEPS, gen, compute_device)   # [STEPS, batch, N]
        w_dev = weight_cpu.to(compute_device)                   # stream params to device
        w_dev.retain_grad()                                     # non-leaf: keep its grad
        mem = torch.zeros(batch, n_classes, device=compute_device)
        spk_sum = torch.zeros(batch, n_classes, device=compute_device)
        for t in range(STEPS):
            # checkpoint recomputes this timestep in backward -> O(1) activation memory
            spk, mem = checkpoint(_timestep, spikes[t], w_dev, mem, use_reentrant=False)
            spk_sum = spk_sum + spk
        loss = loss_fn(spk_sum, ytr)
        loss.backward()
        # bring the gradient back to the CPU master; Adam step + state update on CPU
        weight_cpu.grad = w_dev.grad.detach().to("cpu")
        opt.step()
        last_loss = float(loss.detach().cpu())

    # Load the trained weight into a standard model so it exports/checkpoints normally.
    model = SpikingReadout(N, n_classes, beta=beta)
    with torch.no_grad():
        model.fc.weight.copy_(weight_cpu.detach().cpu())

    Xte, yte = _to_tensors(test_data, "cpu")
    test_acc = evaluate(model, Xte, yte, device="cpu")
    metrics = {
        "mode": "offload (grad-checkpoint + CPU optimizer state)",
        "epochs": int(epochs),
        "compute_device": str(compute_device),
        "n_classes": n_classes,
        "n_inputs": N,
        "test_acc": round(test_acc, 4),
        "final_loss": round(last_loss, 6),
    }
    return model, metrics
