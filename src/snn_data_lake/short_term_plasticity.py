"""
Short-term plasticity (Tsodyks-Markram) — an additive onset/novelty filter for streamed channels.

PROMOTED from research/ into the product (v0.86, additive). streaming_hub stores per-channel spike
streams; a DEPRESSING synapse turns one into an onset detector for free — each spike transiently
depletes release resources, so a burst of closely-spaced spikes fades while a spike arriving after a
gap (an onset) rings out at full strength. `novelty_filter` runs a channel's spike stream through a
Tsodyks-Markram depressing synapse and returns a per-spike novelty weight, so a downstream query can
weight onsets over sustained activity. Additive and opt-in: streaming_hub's default store/query path
is untouched. Refs: Tsodyks & Markram, PNAS 94 (1997); Markram, Wang & Tsodyks, PNAS 95 (1998).

  python -m snn_data_lake.short_term_plasticity        # (also run by run_selfchecks.py)
"""
import math
import os
import tempfile

from snn_data_lake.streaming_hub import StreamingHub


class DynamicSynapse:
    """Tsodyks-Markram synapse: utilization u (facilitation) + resources R (depression)."""

    def __init__(self, U, tau_f, tau_d):
        self.U, self.tau_f, self.tau_d = U, tau_f, tau_d
        self.u, self.R = 0.0, 1.0

    def spike(self, dt):
        self.u *= math.exp(-dt / self.tau_f)
        self.R = 1.0 - (1.0 - self.R) * math.exp(-dt / self.tau_d)
        self.u += self.U * (1.0 - self.u)
        epsc = self.u * self.R
        self.R -= epsc
        return epsc


def novelty_filter(spike_times, U=0.6, tau_f=1.0, tau_d=60.0):
    """Per-spike novelty weight for a channel stream: onsets (after a gap) stay strong, sustained
    bursts fade. Returns [(t, weight), ...]. Apply to StreamingHub.ch[c] or query_window output."""
    syn = DynamicSynapse(U, tau_f, tau_d)
    out, prev = [], None
    for t in spike_times:
        dt = 1e9 if prev is None else (t - prev)
        out.append((t, syn.spike(dt)))
        prev = t
    return out


def main():
    path = os.path.join(tempfile.mkdtemp(), "stream.spkl")
    hub = StreamingHub(path, n_channels=1)
    baseline = [0, 40, 80, 120]                       # sparse: each spike is an onset
    burst = [200, 202, 204, 206, 208, 210]            # dense: sustained activity after an onset
    for t in baseline + burst:
        hub.append(0, t)

    weights = dict(novelty_filter(hub.ch[0]))
    base_w = [weights[t] for t in baseline]
    burst_w = [weights[t] for t in burst]

    print("=" * 60)
    print("SHORT-TERM PLASTICITY — onset/novelty filter over a streamed channel")
    print("=" * 60)
    print(f"stream: {len(baseline)} sparse + {len(burst)} burst spikes on channel 0")
    print(f"sparse (onset) weights : {[round(w, 3) for w in base_w]}")
    print(f"burst  weights (fade)  : {[round(w, 3) for w in burst_w]}")
    print(f"burst onset {burst_w[0]:.3f} -> sustained {burst_w[-1]:.3f} "
          f"(depression); mean sparse {sum(base_w) / len(base_w):.3f} > mean burst {sum(burst_w) / len(burst_w):.3f}")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert burst_w[0] > burst_w[-1] * 1.5, f"a sustained burst must fade (onset >> tail: {burst_w[0]:.3f} vs {burst_w[-1]:.3f})"
    assert sum(base_w) / len(base_w) > sum(burst_w) / len(burst_w), \
        "sparse onsets must carry more novelty weight than sustained burst spikes"
    assert burst_w[0] > 0.8 * base_w[0], "the burst ONSET (after a gap) must ring out near full strength"
    assert dict(novelty_filter(hub.ch[0])) == weights, "same stream -> identical weights (deterministic)"
    print(f"self-check OK: onset {burst_w[0]:.2f} >> sustained {burst_w[-1]:.2f}; sparse novelty > burst; "
          f"additive filter, streaming_hub default path unchanged; deterministic")


if __name__ == "__main__":
    main()
