"""
Phase 2 — SpikeSlicer + SPA-Loss: dynamic event-stream slicing (torch + SpikingJelly).

A lightweight spiking network that learns WHERE to cut a continuous event stream into semantically
coherent segments, trained with a Spiking Position-aware Loss (SPA-Loss). The slicer reads an
event-density / feature stream (produced upstream by the stdlib encoders — snn_data_lake.encode_latency
for TTFS or population_encoding for receptive fields), integrates it in LIF neurons, and emits a spike
at each optimal slice boundary. SPA-Loss has two terms: a membrane-potential-driven term (push the
readout membrane above threshold at target boundaries, below it elsewhere) and a count regularizer
(match the number of cuts to the target). Refs: SpikeSlicer (Cao et al., NeurIPS 2024).

Needs torch (a core dependency) + SpikingJelly, installed in this subproject's own venv rather than
in the deterministic gate env. SpikingJelly is an ADOPTED dependency (P7 -- docs/POLICIES.md records
the adoption into this directory, 2026-07-02), not a quarantine, and under P4c there is no "outside
the brain" category: the venv split exists because CI has no GPU and the gate must stay
deterministic, not because this file is exempt from anything.

  python encoding/spikeslicer.py        # tiny CPU smoke train (no cupy needed)
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class SpikeSlicer(nn.Module):
    """SNN over an event-feature stream (T, B, C) -> per-step cut spikes (T, B) + readout membrane.
    Multi-step mode so the readout LIF stores its membrane sequence (v_seq) for SPA-Loss."""

    def __init__(self, c_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            layer.Linear(c_in, hidden),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(hidden, 1),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, store_v_seq=True),
        )
        functional.set_step_mode(self, "m")           # multi-step: process the whole (T,B,C) at once

    def forward(self, x):
        functional.reset_net(self)
        spikes = self.net(x).squeeze(-1)              # (T, B) cut spikes
        v_seq = self.net[-1].v_seq.squeeze(-1)        # (T, B) readout membrane (for SPA-Loss)
        return spikes, v_seq


def spa_loss(v_seq, spikes, targets, thr=1.0, alpha=1.0, beta=0.1):
    """Spiking Position-aware Loss. targets: (T, B) in {0,1} = optimal cut positions.
    MPD term: at a target, want v >= thr; off a target, want v <= 0. Count term: #cuts ~ #targets."""
    want_high = (thr - v_seq).clamp(min=0) * targets          # push membrane UP at boundaries
    want_low = v_seq.clamp(min=0) * (1.0 - targets)           # push membrane DOWN elsewhere
    mpd = (want_high + want_low).mean()
    count = ((spikes.sum(0) - targets.sum(0)) ** 2).mean()
    return alpha * mpd + beta * count


def smoke():
    torch.manual_seed(0)
    T, B, C = 20, 4, 8
    x = torch.rand(T, B, C)
    targets = (torch.rand(T, B) < 0.15).float()               # synthetic optimal cut positions
    model = SpikeSlicer(C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        spikes, v = model(x)
        loss = spa_loss(v, spikes, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"SPA-loss should decrease with training ({losses[0]:.3f}->{losses[-1]:.3f})"
    print(f"SpikeSlicer smoke OK: SPA-loss {losses[0]:.3f} -> {losses[-1]:.3f}; "
          f"cuts/step {spikes.mean().item():.2f}; device=cpu (cupy backend optional)")


if __name__ == "__main__":
    smoke()
