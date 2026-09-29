"""
v0.64 — online drift + novelty monitor: warn BEFORE recognition silently collapses.

arm_config.py states the ceiling plainly: recognition is tuned to one specific reference sensor mount —
"move the rig + the 95% recognition collapses". The loop already REJECTS novel signals (the Van Rossum
gate) and derives bins from a capture (arm_config.bins_from_capture), but nothing watches the LIVE stream
for slow distribution drift — so a gradual rig/lighting shift degrades accuracy silently. This adds that
watch (roadmap D3).

Two cheap, complementary online signals over the encoded feature windows:
  - drift  : Population Stability Index (PSI = Σ (a−e)·ln(a/e) over bins) between the enrolled REFERENCE
             distribution and a sliding window of recent live windows. PSI > ~0.2 = the input distribution
             has moved (the standard credit-risk drift threshold).
  - novelty: OeSNN-style — a live window is novel if its distance to the reference centroid exceeds a
             radius calibrated from the reference spread; a rising novelty RATE is early drift.
When either trips, `assess()` warns and the caller re-derives bins with arm_config.bins_from_capture and
re-enrolls. Distribution-agnostic (works on any normalised feature vectors); a light leaf (imports only
arm_config). Refs: OeSNN streaming anomaly (PeerJ 2025 HABCOeSNN; Springer 2022 s10994-022-06129-4).

  python drift_monitor.py        # self-check
"""

import math

from snn_data_lake.brainstem import arm_config

_EPS = 1e-6


def _proportions(windows, dim, n_bins):
    """Per-feature histogram proportions over [0,1]; returns [dim][n_bins] summing to 1 per feature."""
    hist = [[0.0] * n_bins for _ in range(dim)]
    for w in windows:
        for i in range(dim):
            b = min(n_bins - 1, max(0, int(w[i] * n_bins)))  # clamp into [0, n_bins)
            hist[i][b] += 1.0
    for i in range(dim):
        tot = sum(hist[i]) or 1.0
        hist[i] = [c / tot for c in hist[i]]
    return hist


# distinct-from: sensorimotor/spike_preprocessing.py::van_rossum_distance -- a metric between two SPIKE
#   TRAINS (L2 between exponentially-filtered waveforms). PSI below is a divergence between two
#   BINNED PROPORTION VECTORS, it is not symmetric, and it takes no time axis. Read 2026-09-09.
def psi(actual, reference, eps=_EPS):
    """THE Population Stability Index between two binned proportion vectors: sum (a - e) * ln((a +
    eps) / (e + eps)) over paired bins. 0 when the two agree; larger as they separate.

    LIFTED TO MODULE SCOPE 2026-09-09 (specs/022 T063, the P2 ADOPT/MERGE decision specs/019
    [B10.13] demanded before a second "drift" mechanism could land). The formula lived once, inside
    `DriftMonitor.psi`, and T063's drift statistic needs the same distance for a band-pair weight
    matrix. Two copies of a distance is the additive shape P2 forbids, so the formula moves here
    and `DriftMonitor.psi` becomes its per-feature caller.

    WHAT THIS MERGE DOES NOT DO. `DriftMonitor` is NOT retired and is NOT T063's instrument: it
    compares an ENROLLED REFERENCE distribution against a sliding window of live sensor features
    and its zero is a PASS (a quiet stream), which is the exact inversion of specs/011 FR-009,
    where a zero distance is a FAILURE of the seeding approach. What the two share is one formula,
    which is all that moves."""
    return sum(
        (a - e) * math.log((a + eps) / (e + eps)) for a, e in zip(actual, reference, strict=True)
    )


def _centroid(windows, dim):
    n = len(windows)
    return [sum(w[i] for w in windows) / n for i in range(dim)]


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class DriftMonitor:
    """Enroll a reference set of normalised windows, then push() live windows and assess() drift."""

    def __init__(self, reference, n_bins=10, psi_warn=0.20, novelty_k=2.5, window=40):
        self.dim = len(reference[0])
        self.n_bins = n_bins
        self.psi_warn = psi_warn
        self.window = window
        self.ref_hist = _proportions(reference, self.dim, n_bins)
        self.centroid = _centroid(reference, self.dim)
        spread = [_dist(w, self.centroid) for w in reference]
        self.radius = (
            sum(spread) / len(spread)
        ) * novelty_k  # novelty threshold from reference spread
        self.novelty_warn = 0.30  # rate of novel windows that signals drift
        self.recent = []

    def is_novel(self, w):
        return _dist(w, self.centroid) > self.radius

    def push(self, w):
        self.recent.append(list(w))
        if len(self.recent) > self.window:
            self.recent.pop(0)
        return self.is_novel(w)

    # distinct-from: substrate/stream_sketch_panel.py::_recent_resemblance -- a MinHash Jaccard
    #   over a motif VOCABULARY (set overlap), not a binned-histogram divergence; it is this
    #   monitor's own panel and reads this number alongside its own.
    # distinct-from: sensorimotor/learned_matcher.py::nearest_dist_enc -- nearest Van Rossum distance to
    #   an enrolled reference TRAIN; "reference" there is a stored exemplar, not a histogram.
    # distinct-from: reservoir/cortical_microcircuit.py::firing_rates,
    #   cortex/language_acquisition.py::developmental_curve, readout/tempotron.py::TempotronMatcher
    #   -- a mean rate, a mean surprise and a one-vs-rest classifier: means over a window, none of
    #   them a distance between two distributions. All five read 2026-09-09.
    def psi(self):
        """Mean per-feature PSI between the reference and the recent-window distribution. The
        formula itself is the module-level `psi` above (the P2 merge, T063) -- what stays here is
        the per-feature loop and the mean, which is what makes this monitor's reading its own."""
        if len(self.recent) < self.n_bins:
            return 0.0
        act = _proportions(self.recent, self.dim, self.n_bins)
        total = 0.0
        for i in range(self.dim):
            total += psi(act[i], self.ref_hist[i], _EPS)
        return total / self.dim

    def novelty_rate(self):
        if not self.recent:
            return 0.0
        return sum(1 for w in self.recent if self.is_novel(w)) / len(self.recent)

    def assess(self):
        """Return (psi, novelty_rate, warn). warn -> re-derive bins + re-enroll before accuracy drops."""
        psi, nov = self.psi(), self.novelty_rate()
        return psi, nov, (psi > self.psi_warn or nov > self.novelty_warn)


def main():
    import random

    rng = random.Random(0)
    dim = 16

    def sample(shift=0.0):
        # a normalised feature window: a base profile + jitter, optionally shifted (a "rig move")
        return [
            min(1.0, max(0.0, 0.5 + 0.3 * math.sin(i) + shift + rng.uniform(-0.05, 0.05)))
            for i in range(dim)
        ]

    reference = [sample() for _ in range(60)]

    # 1. a stable live stream must NOT warn
    mon = DriftMonitor(reference)
    for _ in range(40):
        mon.push(sample())
    psi_stable, nov_stable, warn_stable = mon.assess()

    # 2. a drifted live stream ("move the rig" -> feature means shift) MUST warn
    mon2 = DriftMonitor(reference)
    for _ in range(40):
        mon2.push(sample(shift=0.35))
    psi_drift, nov_drift, warn_drift = mon2.assess()

    # 3. re-derivation: shifted raw distance captures -> arm_config.bins_from_capture gives new cut points
    stable_d = [0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9]
    shifted_d = [d + 0.5 for d in stable_d]
    bins_before = arm_config.bins_from_capture(stable_d)
    bins_after = arm_config.bins_from_capture(shifted_d)

    print("=" * 60)
    print("DRIFT + NOVELTY MONITOR (warn before recognition collapses)")
    print("=" * 60)
    print(f"stable stream : PSI={psi_stable:.3f} novelty={nov_stable:.0%} warn={warn_stable}")
    print(f"drifted stream: PSI={psi_drift:.3f} novelty={nov_drift:.0%} warn={warn_drift}")
    print(
        f"re-derive bins: {tuple(round(b, 2) for b in bins_before)} -> "
        f"{tuple(round(b, 2) for b in bins_after)}  (shifted capture -> new cut points)"
    )
    print("=" * 60)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert not warn_stable, (
        f"stable stream should not warn (PSI={psi_stable:.3f}, nov={nov_stable:.2f})"
    )
    assert warn_drift, f"drifted stream should warn (PSI={psi_drift:.3f}, nov={nov_drift:.2f})"
    assert psi_drift > psi_stable, "drift PSI should exceed stable PSI"
    assert bins_after != bins_before, "a shifted capture should re-derive different bins"
    print("self-check OK: stable stream quiet, rig-move drift warns, bins re-derive on the shift")


if __name__ == "__main__":
    main()
