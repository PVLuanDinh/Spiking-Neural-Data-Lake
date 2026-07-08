"""
Triplet STDP — spike-timing plasticity that captures the FREQUENCY dependence pair-STDP gets wrong (pure stdlib).

Classical pair-based STDP updates a synapse from pre/post spike PAIRS only, so for a fixed pre-before-post lag its
prediction barely depends on how fast the pairs repeat -- and where it does depend, it depends the WRONG way:
cross-pair post-before-pre interactions pile up at high rate and drive net LTD. But the cortical data (Sjostrom,
Turrigiano & Nelson 2001) show the opposite -- the SAME +10 ms pairing flips from little change at low rate to
strong LTP at high rate. Pfister & Gerstner (2006) fixed this by making LTP require a TRIPLET (post-pre-post): a
second slow post-synaptic trace o2 (tau_y=125 ms) gates potentiation, so LTP accumulates with pairing frequency
exactly as measured, and the rule maps onto a BCM-like sliding threshold. Four eligibility traces -- fast/slow pre
(r1 tau_+, r2 tau_x) and fast/slow post (o1 tau_-, o2 tau_y): a pre spike drives LTD w -= o1*(A2- + A3-*r2), a post
spike drives LTP w += r1*(A2+ + A3+*o2), each triplet term reading the OTHER slow trace just before its own spike
updates it. Event-driven exact-exponential decay -> fully deterministic, zero randomness, offline.

Distinct from the repo's existing plasticity: bcm_plasticity is the rate/threshold formulation of the same
selectivity, this is the microscopic spike-TIMING model that derives it; valence_stdp / r_stdp / inhibitory_stdp
are all PAIR-based (two-factor) rules; oja_pca learns variance directions. On-thesis with bcm_plasticity /
valence_stdp / homeostatic_scaling.

Paper: Pfister, J.-P. & Gerstner, W. (2006), "Triplets of Spikes in a Model of Spike Timing-Dependent
Plasticity", Journal of Neuroscience 26(38):9673-9682, DOI 10.1523/JNEUROSCI.1425-06.2006 (all-to-all,
visual-cortex parameter set, Table 3).

  python -m research.triplet_stdp        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import math


class TripletSTDP:
    """Pfister-Gerstner minimal triplet rule: 4 eligibility traces, additive weight change, event-driven."""

    def __init__(self, a2p, a2m, a3p, a3m, tau_p=16.8, tau_m=33.7, tau_x=101.0, tau_y=125.0):
        self.a2p, self.a2m, self.a3p, self.a3m = a2p, a2m, a3p, a3m   # pair LTP/LTD, triplet LTP/LTD amplitudes
        self.tau_p, self.tau_m = tau_p, tau_m                        # fast pre / fast post trace time constants (ms)
        self.tau_x, self.tau_y = tau_x, tau_y                        # slow pre / slow post trace time constants (ms)

    @classmethod
    def triplet(cls):
        """Minimal triplet model: LTP is purely triplet-gated (A2+ = 0), LTD mostly pair-based."""
        return cls(a2p=0.0, a2m=7.1e-3, a3p=6.5e-3, a3m=2.3e-4)

    @classmethod
    def pair(cls):
        """Classical pair-based STDP control: nonzero pair LTP/LTD, no triplet terms."""
        return cls(a2p=8.8e-3, a2m=7.1e-3, a3p=0.0, a3m=0.0)

    def pairing_dw(self, freq_hz, n_pairs=60, dt_ms=10.0):
        """Net additive weight change from n_pairs pre->post pairs (dt_ms apart) repeated at freq_hz."""
        r1 = r2 = o1 = o2 = 0.0                       # fast pre, slow pre, fast post, slow post eligibility traces
        w = 0.0
        t_last = 0.0
        period = 1000.0 / freq_hz
        events = []
        for k in range(n_pairs):                      # pre at k*period, post dt_ms later
            events.append((k * period, 0))            # 0 = presynaptic spike
            events.append((k * period + dt_ms, 1))    # 1 = postsynaptic spike
        events.sort()
        for t, kind in events:
            dt = t - t_last                           # decay every trace to the current event time
            r1 *= math.exp(-dt / self.tau_p)
            r2 *= math.exp(-dt / self.tau_x)
            o1 *= math.exp(-dt / self.tau_m)
            o2 *= math.exp(-dt / self.tau_y)
            t_last = t
            if kind == 0:                             # PRE: LTD, slow-pre trace r2 read BEFORE this spike bumps it
                w -= o1 * (self.a2m + self.a3m * r2)
                r1 += 1.0
                r2 += 1.0
            else:                                     # POST: LTP, slow-post trace o2 read BEFORE this spike bumps it
                w += r1 * (self.a2p + self.a3p * o2)
                o1 += 1.0
                o2 += 1.0
        return w


def main():
    freqs = [1, 10, 20, 40, 50]                       # Sjostrom 2001 pairing-frequency sweep (Hz)
    tri, pair = TripletSTDP.triplet(), TripletSTDP.pair()
    tri_dw = [tri.pairing_dw(f) for f in freqs]
    pair_dw = [pair.pairing_dw(f) for f in freqs]
    i40 = freqs.index(40)

    print("=" * 64)
    print("TRIPLET STDP  --  frequency dependence pair-STDP predicts backwards")
    print("=" * 64)
    print("pairing protocol: 60 pre->post pairs, +10 ms lag, repeated at each rate")
    print(f"freq (Hz)      : {[f'{f:>7}' for f in freqs]}")
    print(f"triplet dw     : {[f'{v:>+7.3f}' for v in tri_dw]}  (LTP grows with rate)")
    print(f"pair    dw     : {[f'{v:>+7.3f}' for v in pair_dw]}  (LTP -> LTD, wrong way)")
    print(f"@40 Hz         : triplet {tri_dw[i40]:+.3f} vs pair {pair_dw[i40]:+.3f}  (opposite sign)")
    print("=" * 64)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    mono_up = all(tri_dw[i] < tri_dw[i + 1] for i in range(len(freqs) - 1))
    mono_down = all(pair_dw[i] > pair_dw[i + 1] for i in range(len(freqs) - 1))
    assert mono_up, f"defining property: triplet LTP must INCREASE with pairing frequency ({tri_dw})"
    assert tri_dw[-1] > 0.4, f"triplet rule must reach decisive net LTP at high frequency ({tri_dw[-1]:.3f})"
    assert mono_down, f"pair-STDP control must predict the OPPOSITE (decreasing) trend ({pair_dw})"
    assert tri_dw[i40] > 0.0 > pair_dw[i40], (
        f"sign divergence at 40 Hz: triplet potentiates where pair depresses "
        f"({tri_dw[i40]:+.3f} vs {pair_dw[i40]:+.3f})")
    reduced = TripletSTDP(a2p=8.8e-3, a2m=7.1e-3, a3p=0.0, a3m=0.0)      # triplet terms off + pair's A2
    assert [reduced.pairing_dw(f) for f in freqs] == pair_dw, "triplet rule must reduce to pair-STDP when A3 = 0"
    assert [TripletSTDP.triplet().pairing_dw(f) for f in freqs] == tri_dw, "same protocol -> identical dw (deterministic)"
    print(f"self-check OK: triplet LTP {tri_dw[0]:+.3f}->{tri_dw[-1]:+.3f} rises with rate, pair "
          f"{pair_dw[0]:+.3f}->{pair_dw[-1]:+.3f} falls; opposite sign @40 Hz; reduces to pair @A3=0; deterministic")


if __name__ == "__main__":
    main()
