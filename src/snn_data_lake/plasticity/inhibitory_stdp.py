"""
Inhibitory STDP (Vogels-Sprekeler) -- additive spike-timing plasticity of INHIBITORY synapses for E/I balance.

The canonical rule of Vogels, Sprekeler, Zenke, Clopath & Gerstner, "Inhibitory Plasticity Balances
Excitation and Inhibition in Sensory Pathways and Memory Networks", Science 334:1569 (2011). A single
postsynaptic LIF neuron gets a FIXED excitatory drive plus N plastic INHIBITORY synapses. Each inhibitory
synapse carries a presynaptic trace; the neuron carries a postsynaptic trace; both jump +1 on a spike and
leak with tau. On a presynaptic spike w += eta*(x_post - alpha); on a postsynaptic spike w += eta*x_pre for
every synapse; w is clamped >= 0. With alpha = 2*rho0*tau the mean weight change is 2*eta*tau*r_pre*(r_post
- rho0), a negative-feedback controller: inhibition GROWS while the neuron fires above rho0 and SHRINKS
while it fires below, so the rate is driven to the target rho0 and excitation is met by co-tuned inhibition
(detailed E/I balance / network stability).

Genuinely DISTINCT from the existing core plasticity family (the headline differentiator):
- homeostatic_scaling.py: slow MULTIPLICATIVE scaling of EXCITATORY weights toward a rate set-point; global,
  NOT spike-timing. iSTDP is SPIKE-TIMING plasticity of INHIBITORY synapses producing detailed balance.
- intrinsic_plasticity.py: adapts the NEURON'S OWN gain/bias (excitability). iSTDP adapts inhibitory SYNAPSE
  weights, leaving the neuron fixed.
- bcm_plasticity / stdp / r_stdp / valence_stdp: EXCITATORY / reward-modulated potentiation. iSTDP is
  inhibitory and balance-seeking, not reward-driven.
So: inhibitory-synapse spike-timing plasticity that drives the postsynaptic rate to a target and balances
excitation with co-tuned inhibition. Additive/opt-in -- a new member of the plasticity family that changes
no existing default. Ref: Vogels et al., Science 334 (2011).

This is a single postsynaptic neuron with small, Poisson-synthetic presynaptic input -- a rate/
timing caricature of the Vogels 2011 rule, NOT a full recurrent E/I network and NOT cognition/AGI. It is
inhibitory plasticity for E/I balance and network stability only. Deterministic (seeded);
additive (no existing behaviour touched).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.inhibitory_stdp        # (also run by run_selfchecks.py)
"""

import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron


class InhibitorySTDP:
    """N plastic inhibitory synapses onto one LIF neuron; iSTDP drives the post rate to rho0 (Vogels 2011)."""

    def __init__(self, n_syn=60, rho0=0.05, tau=20.0, eta=0.005, w_init=0.0):
        self.n = n_syn
        self.rho0 = rho0
        self.eta = eta
        self.decay = math.exp(-1.0 / tau)  # synaptic-trace leak per step
        self.alpha = 2.0 * rho0 * tau  # Vogels depression offset -> fixed point at rho0
        self.w = [w_init] * n_syn  # inhibitory conductance magnitudes (clamped >= 0)
        self.x_pre = [0.0] * n_syn  # per-synapse presynaptic traces
        self.x_post = 0.0  # postsynaptic trace
        self.neuron = LIFNeuron(tau_m=10.0, v_threshold=1.0)

    def step(self, exc_drive, inh_spikes, learn=True):
        """One timestep. Net input = exc_drive - sum_i(w[i]*inh_spikes[i]); apply iSTDP; return 1/0 post spike."""
        self.x_post *= self.decay
        inh = 0.0
        for i in range(self.n):  # leak pre traces + gather this step's inhibitory current
            self.x_pre[i] *= self.decay
            if inh_spikes[i]:
                inh += self.w[i]
        post = self.neuron.step(
            exc_drive - inh
        )  # inhibition subtracts from the fixed excitatory drive
        if learn:
            d = self.eta * (
                self.x_post - self.alpha
            )  # pre spike: potentiate by x_post, depress by alpha
            for i in range(self.n):
                if inh_spikes[i]:
                    w = self.w[i] + d
                    self.w[i] = w if w > 0.0 else 0.0
        for i in range(self.n):  # pre spikes bump their own traces (after the update)
            if inh_spikes[i]:
                self.x_pre[i] += 1.0
        if post:
            if learn:
                for i in range(self.n):  # post spike: potentiate every synapse by its pre trace
                    w = self.w[i] + self.eta * self.x_pre[i]
                    self.w[i] = w if w > 0.0 else 0.0
            self.x_post += 1.0
        return post

    def mean_w(self):
        return sum(self.w) / self.n


def simulate(
    exc_drive, seed, steps=20000, n_syn=60, pre_rate=0.05, rho0=0.05, w_init=0.0, learn=True
):
    """Drive an InhibitorySTDP with fixed excitation + seeded Poisson inhibitory spike trains.
    Return (tail firing rate over the last quarter, final mean inhibitory weight, the trained object)."""
    rng = random.Random(seed)
    net = InhibitorySTDP(n_syn=n_syn, rho0=rho0, w_init=w_init)
    rnd = rng.random
    tail = steps // 4
    tail_spikes = 0
    for t in range(steps):
        inh_spikes = [
            1 if rnd() < pre_rate else 0 for _ in range(n_syn)
        ]  # seeded Poisson presynaptic input
        post = net.step(exc_drive, inh_spikes, learn=learn)
        if t >= steps - tail:
            tail_spikes += post
    return tail_spikes / tail, net.mean_w(), net


def main():
    rho0 = 0.05
    exc_strong, exc_weak = 0.20, 0.10  # strong drive fires >> rho0; weak drive fires < rho0
    w_prime = 0.02  # weak case starts pre-inhibited, to show weights melt

    # (1) BALANCING: strong drive un-inhibited fires far above rho0; iSTDP grows inhibition down to rho0.
    uninhib_strong, _, _ = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=False)
    rate_strong, wmean_strong, _ = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=True)

    # (2) BIDIRECTIONAL: weak drive fires below rho0; pre-inhibited weights shrink toward 0 (not over-suppressed).
    uninhib_weak, _, _ = simulate(exc_weak, seed=11, rho0=rho0, w_init=0.0, learn=False)
    rate_weak, wmean_weak, _ = simulate(exc_weak, seed=11, rho0=rho0, w_init=w_prime, learn=True)

    # (4) DETERMINISM: a second fresh strong run must reproduce the weights and rate exactly.
    rate_strong2, _, net2 = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=True)
    _, _, net1 = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=True)

    print("=" * 72)
    print("INHIBITORY STDP (Vogels 2011) -- spike-timing plasticity of inhibition for E/I balance")
    print("=" * 72)
    print(
        f"target rho0            : {rho0:.3f} spikes/step   (alpha = 2*rho0*tau = {2 * rho0 * 20.0:.2f})"
    )
    print(
        f"strong drive {exc_strong:.2f}      : un-inhibited {uninhib_strong:.4f} -> post-iSTDP {rate_strong:.4f} (-> rho0)"
    )
    print(
        f"weak drive   {exc_weak:.2f}      : un-inhibited {uninhib_weak:.4f} (below rho0), post-iSTDP {rate_weak:.4f}"
    )
    print(
        f"mean inhib weight      : strong {wmean_strong:.4f} (grew from 0)  vs  weak {wmean_weak:.4f} (shrank from {w_prime:.2f})"
    )
    print("=" * 72)

    # ---- self-check: deterministic asserts, ASCII only ----
    assert abs(rate_strong - rho0) < 0.01, (
        f"iSTDP must drive the strong-drive rate to rho0 ({rate_strong:.4f} vs {rho0})"
    )
    assert abs(rate_strong - rho0) < 0.25 * abs(uninhib_strong - rho0), (
        f"post-iSTDP rate must be far closer to rho0 than the un-inhibited rate ({rate_strong:.4f})"
    )
    assert rate_strong < 0.5 * uninhib_strong, (
        f"balanced rate must fall well below the un-inhibited runaway ({rate_strong:.4f} vs {uninhib_strong:.4f})"
    )
    assert wmean_strong > 0.0, (
        f"strong drive must GROW inhibitory weights from 0 ({wmean_strong:.4f})"
    )
    assert uninhib_weak < rho0, (
        f"weak drive must fire below the target ({uninhib_weak:.4f} < {rho0})"
    )
    assert wmean_weak < w_prime, (
        f"weak drive must SHRINK the pre-inhibited weights ({wmean_weak:.4f} < {w_prime})"
    )
    assert wmean_weak < wmean_strong, (
        f"mean inhib weight must be lower under weak than strong drive ({wmean_weak:.4f} < {wmean_strong:.4f})"
    )
    assert net1.w == net2.w and rate_strong == rate_strong2, (
        "same seed -> identical weights and rate"
    )
    print(
        f"self-check OK: un-inhibited {uninhib_strong:.3f} -> iSTDP {rate_strong:.3f} ~ rho0 {rho0:.3f}; "
        f"weights strong {wmean_strong:.3f} > weak {wmean_weak:.3f}; additive; deterministic"
    )


if __name__ == "__main__":
    main()
