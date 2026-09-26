"""
Basic-Components composition interface -- the swappable (encoder, neuron, connection, rule) contract.

This is the architectural idea Brain-Cog exposes and this repo lacked: a *learning rule is a function of a
node and a connection*, so any neuron model, any synaptic connection, any local plasticity rule and any
encoder plug together through a small duck-typed contract. Brain-Cog's braincog/base/learningrule/STDP.py
is the reference PATTERN -- `STDP(node, connection)` runs `i = connection(x); s = node(i)`, low-passes a
presynaptic `trace`, and returns `(spike, dw)`. We keep that shape but replace its
`torch.autograd.grad(...)` weight update with a LOCAL Hebbian outer product computed in numpy:

    dw_ij = lr * (pre_trace_i - baseline_i) * post_spike_j

i.e. a pre-synaptic covariance / STDP-eligibility rule. It is LOCAL (only the pre trace and the post
spike enter it), the baseline is a running mean (no hand-set threshold that arbitrates behaviour), and the
weights are soft-bounded in `Connection.update`. There is NO loss.backward, NO autograd, NO surrogate
gradient anywhere -- the weights autonomously diverge from their init under the local rule.

Protocols (duck-typed, documented -- our existing cells satisfy them, some via a one-line shim):

    Node     : step(current) -> spike ,  reset()          # lif_neuron (reset_state), adex_neuron, ...
    Rule     : delta(pre, post) -> dw                      # this module's STDPRule, plasticity/*
    Encoder  : encode(x) -> vector                         # cortex/population_encoding, ...

`compose_step(encoder, neuron, connection, rule, x)` runs one tick through the whole contract.

  python -m snn_data_lake.substrate.basic_components        # self-check
"""

import math
import random
from typing import Protocol, runtime_checkable

import numpy as np

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "Connection.update": (
        "NOT_A_RULE",
        "applies a SUPPLIED dw in place, masks and clips. A primitive carrying out whatever change it "
        "is handed -- the claim belongs to whichever rule computed dw, exactly like "
        "reasoning/eprop::_batch_grad",
    ),
}


# --------------------------------------------------------------------------------------------------
# Protocols -- the composition contract. These are documentation + optional runtime isinstance checks;
# any object with the right methods (duck typing) plugs in, no inheritance required.
# --------------------------------------------------------------------------------------------------
@runtime_checkable
class Node(Protocol):
    """A neuron (or a population of them). Canonical method names: step + reset.

    `step(current)` advances one timestep and returns the spike(s); `current` and the returned spike may
    be scalars (single neuron) or vectors/arrays (a layer). Cells whose reset is named differently
    (e.g. LIFNeuron.reset_state) are adapted with a tiny shim -- see cortex/heterogeneous_microcircuit.
    """

    def step(self, current): ...
    def reset(self): ...


@runtime_checkable
class Rule(Protocol):
    """A local plasticity rule. `delta(pre, post)` returns the weight change dw for the connection it
    drives -- computed ONLY from presynaptic activity `pre` and postsynaptic spike `post` (never a
    global loss/gradient)."""

    def delta(self, pre, post): ...


@runtime_checkable
class Encoder(Protocol):
    """A stimulus encoder. `encode(x)` maps an input `x` to a presynaptic activity vector."""

    def encode(self, x): ...


# --------------------------------------------------------------------------------------------------
# Connection -- the swappable synaptic primitive: a masked weight matrix with propagate + bounded update.
# --------------------------------------------------------------------------------------------------
class Connection:
    """A dense weight matrix W (n_pre x n_post) with an optional binary mask.

    propagate(pre) -> post current vector (masked matmul); update(dw) -> in-place, mask-respecting,
    soft-bounded weight change. This is the primitive you swap to change connectivity (dense/sparse/
    structured) without touching the neuron or the rule.
    """

    def __init__(self, n_pre, n_post, seed=0, w_scale=0.1, mask=None, w_min=0.0, w_max=1.0):
        if n_pre < 1 or n_post < 1:
            raise ValueError("need n_pre>=1 and n_post>=1")
        if w_max <= w_min:
            raise ValueError("need w_max > w_min")
        self.n_pre, self.n_post = n_pre, n_post
        self.w_min, self.w_max = float(w_min), float(w_max)
        # per-instance deterministic init (random.Random per the repo determinism invariant)
        rng = random.Random(seed)
        self.W = np.array(
            [[rng.uniform(0.0, w_scale) for _ in range(n_post)] for _ in range(n_pre)],
            dtype=float,
        )
        if mask is None:
            self.mask = np.ones((n_pre, n_post), dtype=float)
        else:
            self.mask = np.asarray(mask, dtype=float)
            if self.mask.shape != (n_pre, n_post):
                raise ValueError("mask shape must equal (n_pre, n_post)")
            self.W *= self.mask  # absent synapses start (and stay) at zero
        self.W0 = self.W.copy()  # frozen init, so callers can measure divergence

    def propagate(self, pre):
        """Presynaptic activity vector (len n_pre) -> postsynaptic current vector (len n_post)."""
        pre = np.asarray(pre, dtype=float)
        if pre.shape != (self.n_pre,):
            raise ValueError(f"pre must have length {self.n_pre}, got {pre.shape}")
        return pre @ (self.W * self.mask)

    def update(self, dw):
        """In-place bounded weight change: add dw, zero the masked-out synapses, clip to [w_min, w_max]."""
        dw = np.asarray(dw, dtype=float)
        if dw.shape != self.W.shape:
            raise ValueError(f"dw must have shape {self.W.shape}, got {dw.shape}")
        self.W += dw
        self.W *= self.mask
        np.clip(self.W, self.w_min, self.w_max, out=self.W)

    def divergence(self):
        """Frobenius norm ||W - W_init|| -- how far the weights have autonomously moved from init."""
        return float(np.linalg.norm(self.W - self.W0))


# --------------------------------------------------------------------------------------------------
# STDPRule -- the local Hebbian rule (Brain-Cog STDP shape, autograd replaced by a numpy outer product).
# --------------------------------------------------------------------------------------------------
class STDPRule:
    """Local pre-synaptic-covariance / STDP-eligibility rule.

        dw = lr * outer(trace - baseline, post_spike)

    `trace` is a low-pass (decay) accumulation of presynaptic activity -- the STDP eligibility trace, held
    inside the rule exactly like Brain-Cog's STDP.cal_trace. `baseline` is a running mean of that trace, so
    presynaptic activity ABOVE its own baseline potentiates and BELOW it depresses (a covariance rule): the
    threshold is data-driven, not a hand-set constant, which keeps learning stable and lets weights
    differentiate instead of all saturating. No gradients, no backprop.
    """

    def __init__(self, lr=0.02, decay=0.9, tau_baseline=20.0):
        self.lr = float(lr)
        self.decay = float(decay)
        self.tau_baseline = float(tau_baseline)
        self.trace = None
        self.baseline = None

    def reset(self):
        self.trace = None
        self.baseline = None

    def delta(self, pre, post):
        """Local weight change from presynaptic activity `pre` (len n_pre) and post spike `post` (len n_post)."""
        pre = np.asarray(pre, dtype=float)
        post = np.asarray(post, dtype=float)
        if self.trace is None:
            self.trace = np.zeros_like(pre)
            self.baseline = np.zeros_like(pre)
        # eligibility trace: leaky accumulation of presynaptic activity
        self.trace = self.decay * self.trace + pre
        # running baseline of the trace (EMA) -- the sliding covariance reference
        self.baseline += (self.trace - self.baseline) / self.tau_baseline
        return self.lr * np.outer(self.trace - self.baseline, post)


# --------------------------------------------------------------------------------------------------
# compose_step -- one tick through the whole contract (the thin composition helper).
# --------------------------------------------------------------------------------------------------
def compose_step(encoder, neuron, connection, rule, x):
    """Run one tick: encode -> propagate -> spike -> local weight update. Returns the post spike(s).

        current = connection.propagate(encoder.encode(x))
        spike   = neuron.step(current)
        dw      = rule.delta(pre=<encoded activity>, post=spike)   # local, no backprop
        connection.update(dw)

    Any Encoder / Node / Connection / Rule (this module's, or an existing cell + a shim) composes here.
    """
    pre = np.asarray(encoder.encode(x), dtype=float)
    current = connection.propagate(pre)
    spike = neuron.step(current)
    post = np.asarray(spike, dtype=float)
    dw = rule.delta(pre=pre, post=post)
    connection.update(dw)
    return post


# --------------------------------------------------------------------------------------------------
# Minimal demo Encoder + Node so this module self-checks the interface end-to-end on its own (the real
# heterogeneous cells are composed in cortex/heterogeneous_microcircuit).
# --------------------------------------------------------------------------------------------------
class _DemoEncoder:
    """Tiny Gaussian population rate encoder (a demo Encoder): scalar x in [0,1] -> n-vector of rates."""

    def __init__(self, n):
        self.centers = [(i + 0.5) / n for i in range(n)]
        self.sigma = 1.0 / n

    def encode(self, x):
        s2 = 2.0 * self.sigma * self.sigma
        return [math.exp(-((x - c) ** 2) / s2) for c in self.centers]


class _DemoLIFLayer:
    """Minimal vectorized LIF population (a demo Node): step(current_vec) -> spike_vec, reset()."""

    def __init__(self, n, tau_m=8.0, v_threshold=0.5):
        self.n = n
        self.decay = math.exp(-1.0 / tau_m)
        self.v_threshold = v_threshold
        self.reset()

    def reset(self):
        self.v = np.zeros(self.n, dtype=float)

    def step(self, current):
        self.v = self.v * self.decay + np.asarray(current, dtype=float)
        spike = (self.v >= self.v_threshold).astype(float)
        self.v = self.v * (1.0 - spike)  # reset the neurons that fired
        return spike


def _drive(n_pre, n_post, seed, mask=None, steps=300):
    """Drive a Connection through compose_step for `steps` ticks; return (connection, total post spikes)."""
    enc = _DemoEncoder(n_pre)
    layer = _DemoLIFLayer(n_post)
    conn = Connection(n_pre, n_post, seed=seed, mask=mask)
    rule = STDPRule(lr=0.02, decay=0.9, tau_baseline=20.0)
    rng = random.Random(seed + 100)
    total = 0
    for _ in range(steps):
        x = rng.random()  # deterministic input stream in [0,1]
        spike = compose_step(enc, layer, conn, rule, x)
        total += int(spike.sum())
    return conn, total


def main():
    n_pre, n_post = 6, 4

    # 1) fully-connected: drive it and prove the weights autonomously diverge from init
    conn, spikes = _drive(n_pre, n_post, seed=1)
    norm0 = float(np.linalg.norm(conn.W0))
    normf = float(np.linalg.norm(conn.W))
    div = conn.divergence()
    within_bounds = bool(((conn.W >= conn.w_min - 1e-12) & (conn.W <= conn.w_max + 1e-12)).all())

    # 2) masked connection: absent synapses must stay exactly zero through learning
    checker = np.array([[(i + j) % 2 for j in range(n_post)] for i in range(n_pre)], dtype=float)
    conn_m, _ = _drive(n_pre, n_post, seed=2, mask=checker)
    masked_stay_zero = bool((conn_m.W[checker == 0] == 0.0).all())
    live_synapses_moved = float(np.linalg.norm(conn_m.W[checker == 1] - conn_m.W0[checker == 1]))

    # 3) determinism: identical seeds -> byte-identical learned weights
    conn_a, _ = _drive(n_pre, n_post, seed=1)
    deterministic = bool(np.array_equal(conn.W, conn_a.W))

    print("=" * 66)
    print("BASIC COMPONENTS -- (encoder, neuron, connection, rule) composition contract")
    print("=" * 66)
    print(
        f"connection      : {n_pre} pre -> {n_post} post, local STDP/covariance rule, no backprop"
    )
    print(f"post spikes      : {spikes} over 300 ticks (rule was driven)")
    print(f"weight norm      : ||W0||={norm0:.4f} -> ||Wf||={normf:.4f}")
    print(
        f"divergence       : ||Wf - W0||={div:.4f}  (weights moved from init under the local rule)"
    )
    print(f"bounded          : all weights within [{conn.w_min}, {conn.w_max}] = {within_bounds}")
    print(
        f"masked synapses  : absent stay zero = {masked_stay_zero}; live moved = {live_synapses_moved:.4f}"
    )
    print(f"deterministic    : same seed -> identical weights = {deterministic}")
    print("=" * 66)

    # ---- self-check: deterministic asserts (real numbers from the runs above) ----
    assert isinstance(_DemoLIFLayer(3), Node), "demo layer must satisfy the Node protocol"
    assert isinstance(STDPRule(), Rule), "STDPRule must satisfy the Rule protocol"
    assert isinstance(_DemoEncoder(3), Encoder), "demo encoder must satisfy the Encoder protocol"
    assert spikes > 0, "the composed circuit must actually spike (drive the rule)"
    assert div > 1e-3, f"weights must diverge from init under the local rule (div={div:.5f})"
    assert within_bounds, "weights must stay soft-bounded in [w_min, w_max]"
    assert masked_stay_zero, "masked-out synapses must remain exactly zero through learning"
    assert live_synapses_moved > 1e-3, "live (unmasked) synapses must still learn"
    assert deterministic, "same seed must reproduce identical learned weights"
    print(
        f"self-check OK: composition contract runs; weights diverge ||dW||={div:.4f} "
        f"(||W0||={norm0:.4f}->||Wf||={normf:.4f}); bounded; mask honored; deterministic"
    )


if __name__ == "__main__":
    main()
