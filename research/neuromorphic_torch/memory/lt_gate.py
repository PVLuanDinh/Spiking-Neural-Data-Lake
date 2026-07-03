"""
Phase 4a — LT-Gate: Local Timescale Gating dual-time-constant neuron (anti-forgetting).

A spiking neuron with TWO membrane branches on different timescales -- a FAST branch that tracks
recent input and a SLOW branch that retains older context -- combined by a learnable gate. During
continuous ingestion the slow branch preserves long-range structure, mitigating catastrophic
forgetting that a single fast time-constant would suffer. Surrogate-gradient trainable (the gate and
downstream weights). Refs: LT-Gate dual-timescale neurons for continual spiking learning.

Quarantined: needs torch + SpikingJelly (own venv). NOT part of the zero-dep core / CI.

  python memory/lt_gate.py        # CPU smoke: slow branch retains input longer than a fast-only neuron
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate


class LTGateNeuron(nn.Module):
    """Dual-time-constant LIF: v = g*fast + (1-g)*slow, g learnable. Input x: (T, B, N)."""

    def __init__(self, tau_fast=2.0, tau_slow=20.0, v_threshold=1.0, gate_init=0.0):
        super().__init__()
        self.decay_fast = 1.0 - 1.0 / tau_fast
        self.decay_slow = 1.0 - 1.0 / tau_slow
        self.thr = v_threshold
        self.sg = surrogate.ATan()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))   # sigmoid(gate) mixes fast vs slow

    def forward(self, x):
        g = torch.sigmoid(self.gate)
        vf = torch.zeros_like(x[0])
        vs = torch.zeros_like(x[0])
        outs = []
        for t in range(x.shape[0]):
            vf = self.decay_fast * vf + x[t]
            vs = self.decay_slow * vs + x[t]
            v = g * vf + (1.0 - g) * vs
            s = self.sg(v - self.thr)
            vf = vf - s * self.thr        # reset only the FAST branch; the SLOW branch is a decaying
            outs.append(s)                # memory trace (not reset) -> retains context -> anti-forgetting
        return torch.stack(outs)          # (T, B, N)


def _retention_horizon(spikes, stop_t):
    """Last timestep (>= stop_t) with any spike, minus stop_t = how long activity persists after input."""
    active = (spikes.sum(dim=(1, 2)) > 0.5).nonzero(as_tuple=True)[0]
    late = [int(t) for t in active if int(t) >= stop_t]
    return (max(late) - stop_t) if late else 0


def smoke():
    torch.manual_seed(0)
    T, B, N, stop_t = 70, 3, 6, 12
    x = torch.zeros(T, B, N)
    x[:stop_t] = 1.2                       # an input burst (charges the slow trace), then silence

    lt = LTGateNeuron(tau_fast=2.0, tau_slow=25.0, gate_init=-1.0)     # gate toward the SLOW branch
    fast_only = LTGateNeuron(tau_fast=2.0, tau_slow=2.0, gate_init=5.0)  # effectively fast-only

    horizon_lt = _retention_horizon(lt(x), stop_t)
    horizon_fast = _retention_horizon(fast_only(x), stop_t)

    # surrogate-gradient trainability: a loss on the output must reach the gate parameter
    out = lt(x)
    loss = (out.mean() - 0.1) ** 2
    loss.backward()
    grad_ok = lt.gate.grad is not None and torch.isfinite(lt.gate.grad).all()

    assert horizon_lt > horizon_fast, \
        f"LT-Gate slow branch must retain input longer than fast-only ({horizon_lt} vs {horizon_fast})"
    assert grad_ok, "surrogate gradient must reach the LT-Gate parameters"
    print(f"LT-Gate smoke OK: post-input retention {horizon_lt} steps (dual-timescale) vs "
          f"{horizon_fast} (fast-only) -> anti-forgetting; surrogate grad reaches gate")


if __name__ == "__main__":
    smoke()
