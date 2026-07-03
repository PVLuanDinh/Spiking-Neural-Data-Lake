"""
Phase 3 — IS-SNN embedding: Intrinsically-Stable SNN trained natively by surrogate-gradient STBP.

An event embedding network trained end-to-end with Spatio-Temporal Backpropagation (surrogate
gradients) -- NO ANN-to-SNN conversion. It is "intrinsically stable": a variance-preserving
(orthogonal) weight init keeps the per-layer firing rate stable across depth WITHOUT dynamic
BatchNorm, avoiding catastrophic firing-rate decay. Multi-step SpikingJelly LIF; defaults to the
`cupy` backend (custom CUDA kernels) when available and falls back to `torch` on CPU. The rate-coded
readout is the embedding a downstream Victor-Purpura / Van Rossum index (core, stdlib) can store.
Refs: STBP (Wu et al., Front. Neurosci. 2018); intrinsically-stable SNN init.

Quarantined: needs torch + SpikingJelly (own venv). NOT part of the zero-dep core / CI.

  python embedding/is_snn.py        # tiny CPU STBP smoke (cupy backend used automatically if present)
"""
import math

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


def pick_backend():
    """Prefer SpikingJelly's cupy backend (CUDA kernels); fall back to torch if cupy/CUDA absent."""
    try:
        import cupy  # noqa: F401
        if torch.cuda.is_available():
            return "cupy"
    except Exception:
        pass
    return "torch"


class ISSNN(nn.Module):
    """Intrinsically-stable spiking embedder. Variance-preserving init -> stable firing rate across
    depth without BatchNorm; trained by surrogate-gradient STBP."""

    def __init__(self, c_in, hidden=64, depth=3, out=16, backend=None):
        super().__init__()
        self.backend = backend or pick_backend()
        self.lins = nn.ModuleList()
        self.lifs = nn.ModuleList()
        dims = [c_in] + [hidden] * depth
        for a, b in zip(dims, dims[1:]):
            lin = layer.Linear(a, b)
            nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))   # variance-preserving (intrinsically stable)
            nn.init.zeros_(lin.bias)
            self.lins.append(lin)
            self.lifs.append(neuron.LIFNode(surrogate_function=surrogate.ATan(),
                                            detach_reset=True, backend=self.backend))
        self.head = layer.Linear(hidden, out)
        functional.set_step_mode(self, "m")

    def forward(self, x):                       # x: (T, B, C) spikes
        functional.reset_net(self)
        h = x
        rates = []
        for lin, lif in zip(self.lins, self.lifs):
            h = lif(lin(h))
            rates.append(h.mean().item())       # per-layer mean firing rate (stability diagnostic)
        emb = self.head(h).mean(0)              # (B, out) rate-coded embedding (mean over time)
        return emb, rates


def smoke():
    torch.manual_seed(0)
    T, B, C, K = 15, 24, 10, 3
    class_rates = torch.rand(K, C) * 0.5 + 0.1
    ys = torch.randint(0, K, (B,))
    x = torch.bernoulli(class_rates[ys].unsqueeze(0).expand(T, B, C).contiguous())   # (T,B,C) spikes

    model = ISSNN(C, hidden=64, depth=3, out=16)
    readout = nn.Linear(16, K)
    opt = torch.optim.Adam(list(model.parameters()) + list(readout.parameters()), lr=1e-2)

    losses, layer_rates = [], []
    for _ in range(80):
        emb, layer_rates = model(x)
        logits = readout(emb)
        loss = torch.nn.functional.cross_entropy(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    acc = (logits.argmax(1) == ys).float().mean().item()

    assert losses[-1] < losses[0], f"STBP must reduce loss ({losses[0]:.2f}->{losses[-1]:.2f})"
    assert acc >= 0.8, f"IS-SNN embedding must separate classes ({acc:.2f})"
    assert min(layer_rates) > 0.0 and layer_rates[-1] > 0.4 * layer_rates[0], \
        f"firing rate must stay stable across depth (no catastrophic decay): {layer_rates}"
    print(f"IS-SNN smoke OK: STBP loss {losses[0]:.2f}->{losses[-1]:.2f}, acc {acc:.0%}; "
          f"layer firing rates {[round(r, 3) for r in layer_rates]} (stable, no decay); backend={model.backend}")


if __name__ == "__main__":
    smoke()
