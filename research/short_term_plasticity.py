"""
Short-term plasticity (Tsodyks-Markram dynamic synapses) — transient synaptic filters
(synaptic-dynamics axis). Pure-stdlib.

Every synapse in the repo has a static weight; strength changes only through learning. But real
synapses are DYNAMIC on the millisecond timescale: each spike transiently depletes a finite pool of
release resources (depression) and raises release probability (facilitation), so a synapse's gain
depends on the recent spike history. That makes a single synapse a temporal filter for free -- a
depressing synapse is a high-pass/novelty detector (responds to onsets, fades on steady input), a
facilitating one is a low-pass integrator (builds up over a burst). It's the mechanism behind
adaptation and gain control the temporal codes here assume. Refs: Tsodyks & Markram, PNAS 94 (1997);
Markram, Wang & Tsodyks, PNAS 95 (1998).

  Run: python short_term_plasticity.py        # (also run by run_selfchecks.py)
"""
import math


class DynamicSynapse:
    """Tsodyks-Markram synapse: utilization u (facilitation) + resources R (depression)."""

    def __init__(self, U, tau_f, tau_d):
        self.U, self.tau_f, self.tau_d = U, tau_f, tau_d
        self.u, self.R = 0.0, 1.0

    def spike(self, dt):
        """Advance by dt (ms) since the last spike, release, return the EPSC amplitude."""
        self.u *= math.exp(-dt / self.tau_f)                 # utilization decays toward 0
        self.R = 1.0 - (1.0 - self.R) * math.exp(-dt / self.tau_d)   # resources recover toward 1
        self.u += self.U * (1.0 - self.u)                    # facilitation: each spike raises u
        epsc = self.u * self.R                               # released = u * R
        self.R -= epsc                                       # depression: deplete the released fraction
        return epsc

    def train(self, isi, n):
        self.u, self.R = 0.0, 1.0
        return [self.spike(1e9 if i == 0 else isi) for i in range(n)]   # first spike: fully recovered


def main():
    isi, n = 20.0, 8                                          # 8 spikes at 50 Hz
    dep = DynamicSynapse(U=0.5, tau_f=50.0, tau_d=800.0).train(isi, n)    # strong depression
    fac = DynamicSynapse(U=0.15, tau_f=500.0, tau_d=50.0).train(isi, n)   # strong facilitation

    # recovery: after a long silent gap the depressing synapse returns to baseline
    s = DynamicSynapse(U=0.5, tau_f=50.0, tau_d=800.0)
    s.train(isi, n)
    recovered = s.spike(3000.0)

    print("=" * 60)
    print("SHORT-TERM PLASTICITY (Tsodyks-Markram dynamic synapses)")
    print("=" * 60)
    print(f"train of {n} spikes at {1000 / isi:.0f} Hz; EPSC amplitudes:")
    print(f"  depressing  : {[round(e, 3) for e in dep]}  (last/first {dep[-1] / dep[0]:.2f})")
    print(f"  facilitating: {[round(e, 3) for e in fac]}  (peak/first {max(fac) / fac[0]:.2f})")
    print(f"  depressing synapse recovers to {recovered:.3f} after a 3 s gap (baseline {dep[0]:.3f})")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert dep[-1] < 0.6 * dep[0], f"depressing synapse must fade over a train ({dep[-1] / dep[0]:.2f})"
    assert all(dep[i + 1] <= dep[i] + 1e-9 for i in range(n - 1)), "depression must be monotone non-increasing"
    assert fac[1] > fac[0], f"facilitating synapse must grow (2nd > 1st spike: {fac[1]:.3f} > {fac[0]:.3f})"
    assert max(fac) > 1.3 * fac[0], f"facilitation must build up meaningfully ({max(fac) / fac[0]:.2f})"
    assert recovered > 0.8 * dep[0], f"a long gap must restore baseline release ({recovered:.3f} vs {dep[0]:.3f})"
    assert DynamicSynapse(0.5, 50, 800).train(isi, n) == dep, "same params -> identical train (deterministic)"
    print(f"self-check OK: depression to {dep[-1] / dep[0]:.2f}x, facilitation to {max(fac) / fac[0]:.2f}x, "
          f"recovery {recovered / dep[0]:.2f}x; deterministic")


if __name__ == "__main__":
    main()
