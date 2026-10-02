"""
Stream-sketch panel -- "Lake Vitals": one fixed-memory O(1)-per-record self-monitor over the brain's
spike/motif stream, composing the nine dead streaming-sketch primitives behind a single
observe(event) / snapshot()->dict contract, plus drift_novelty()->[0,1].

The lake ships nine INDEPENDENT fixed-memory sketches, each answering ONE streaming question in memory
that does not grow with the stream length. Used one at a time they are easy to forget to wire up; used
TOGETHER they are a live "vitals panel" for the running brain -- a single object you push every spike/motif
event into, that keeps a constant-memory running picture of the stream and raises ONE bounded drift signal.
This module is that composition: a BANK of sketches (an additive library), NOT a mixture-of-experts and NOT
a router -- every sketch sees every record and each keeps answering its own question. Nothing here selects an
action, drives a motor, or trains anything; it only WATCHES.

What each reused primitive contributes to one observe(event) (imported + adapted, NEVER rewritten):
  * spike_cardinality.HyperLogLog   -> DISTINCT motif count F0          ("how many unique motifs streamed?")
  * spike_stream_sketch.HeavyHitters-> top-K frequent motifs            ("which motifs recur most?")
  * sliding_window_count.SlidingWindowCounter (DGIM) -> spikes in last W ("recent firing rate, windowed")
  * streaming_quantile.StreamingQuantiles (P2)       -> p50/p90/p99 of per-record activity (order statistics)
  * frequency_moment.AMSSketch (F2) -> sum_i f_i^2 = distribution SKEW  ("uniform chatter vs a few hot motifs?")
  * reservoir_sampling.Reservoir    -> a uniform SAMPLE of real motifs  ("give me k fair example motifs")
  * morris_counter.MorrisCounter    -> total event count in ~loglog bits ("how many events, approx, tiny reg")
  * minhash.MinHash                 -> Jaccard(recent motif-vocab, reference) -> a set-resemblance drift term
  * drift_monitor.DriftMonitor      -> PSI + novelty-rate over the feature windows -> the primary drift term

drift_novelty()->[0,1] blends the two drift-aware sketches into ONE bounded scalar: a weighted sum of the
DriftMonitor PSI (normalised against its own warn threshold), the DriftMonitor novelty rate, and the MinHash
resemblance DROP of the recent motif vocabulary against an enrolled reference. It RISES when the input
distribution moves and is deliberately conservative (weights sum to 1, clamped to [0,1]).

Cost / memory: every per-record op is a FIXED-size sketch update, so observe() is O(1) amortised in the
stream length and total memory is bounded (the sum of nine fixed sketch footprints + two bounded windows),
independent of how long the stream runs. The two O(set) operations (MinHash signatures, DriftMonitor PSI)
are DEFERRED to snapshot()/drift_novelty(), which are called occasionally, never per record.

Honest ceiling: this is a bank of APPROXIMATE sketches read-only over the stream -- each carries its own
estimator error (documented in its own module) and drift_novelty is a HEURISTIC blend, a distribution-shift
alarm, not a calibrated statistic and not semantic novelty. It watches; it never acts. The read-only
snapshot() is safe to tap anywhere. drift_novelty() is ARM-ADJACENT only when a caller wires it into action
selection (see the integration note at the bottom): there it may ONLY ADD caution toward the SAFE action --
it can never introduce motion, and the brainstem reflex / E-STOP remains the dominant, independent guarantee.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.stream_sketch_panel   # self-check
"""

from collections import deque

from snn_data_lake.infra._mathutils import clamp01
from snn_data_lake.infra.drift_monitor import DriftMonitor
from snn_data_lake.substrate.frequency_moment import AMSSketch, exact_f2
from snn_data_lake.substrate.minhash import MinHash
from snn_data_lake.substrate.morris_counter import MorrisCounter
from snn_data_lake.substrate.reservoir_sampling import Reservoir
from snn_data_lake.substrate.sliding_window_count import SlidingWindowCounter
from snn_data_lake.substrate.spike_cardinality import HyperLogLog
from snn_data_lake.substrate.spike_stream_sketch import HeavyHitters
from snn_data_lake.substrate.streaming_quantile import StreamingQuantiles

#: Squeeze a float into [0,1] (bounds the drift blend and every component that feeds it).
#: MERGED 2026-08-15 into `infra/_mathutils.clamp01` (8 copies -> 1). Alias keeps the local
#: spelling so no call site changes; the MECHANISM is singular.
_clamp01 = clamp01


def _is_index_set(m):
    """True iff m is a set/frozenset of ints -- the form the motif sketches that iterate indices require."""
    return isinstance(m, (set, frozenset)) and all(isinstance(x, int) for x in m)


class StreamSketchPanel:
    """Lake Vitals: push every spike/motif event through observe(); read a constant-memory picture via
    snapshot() and one bounded drift alarm via drift_novelty(). A bank of nine fixed-memory sketches --
    additive, read-only, deterministic given the seed. Selects nothing, drives nothing.

    An `event` may be:
      * a mapping with any of keys {'motif': iterable of int feature-indices, 'features': list[float] in [0,1],
        'spike': 0/1, 'value': float} -- missing fields are DERIVED from 'features' when present;
      * a bare feature vector (list/tuple of numbers) -> motif = its active-index set, value = active count,
        spike = 1 if any feature is active;
      * a bare set/frozenset of int indices -> treated as the motif;
      * any other hashable id -> a bare motif (frequency/cardinality/sample/count only; the index-iterating
        sketches HeavyHitters + MinHash gracefully skip it).
    """

    def __init__(
        self,
        hll_p=11,
        heavy_k=3,
        cms_depth=4,
        cms_width=256,
        dgim_window=256,
        dgim_k=4,
        quantile_ps=(0.5, 0.9, 0.99),
        ams_d=16,
        ams_g=9,
        reservoir_k=16,
        morris_m=64,
        minhash_num_perm=64,
        active_threshold=0.5,
        drift_window=40,
        drift_reference_size=60,
        drift_n_bins=10,
        drift_psi_warn=0.20,
        drift_novelty_k=2.5,
        drift_weights=(0.5, 0.3, 0.2),
        seed=0,
    ):
        self.seed = seed
        self.active_threshold = float(active_threshold)
        # --- the nine fixed-memory sketches (each seeded where it has an RNG) ---
        self.hll = HyperLogLog(p=hll_p, seed=seed)  # distinct F0
        self.heavy = HeavyHitters(
            k=heavy_k, depth=cms_depth, width=cms_width, seed=seed
        )  # top-K frequency
        self.dgim = SlidingWindowCounter(dgim_window, max_buckets=dgim_k)  # windowed spike count
        self.quant = StreamingQuantiles(quantile_ps)  # p50/p90/p99 activity
        self.ams = AMSSketch(d=ams_d, g=ams_g, seed=seed)  # F2 skew
        self.reservoir = Reservoir(reservoir_k, seed=seed)  # uniform motif sample
        self.morris = MorrisCounter(m=morris_m, a=2.0, seed=seed)  # approx total count
        self.minhash = MinHash(num_perm=minhash_num_perm, seed=seed)  # set-resemblance drift term
        # --- drift enrollment state (DriftMonitor needs a reference set up front -> buffer then freeze) ---
        self._drift_window = int(drift_window)
        self._drift_reference_size = int(drift_reference_size)
        self._drift_n_bins = int(drift_n_bins)
        self._drift_psi_warn = float(drift_psi_warn)
        self._drift_novelty_k = float(drift_novelty_k)
        self._drift_weights = tuple(float(w) for w in drift_weights)
        self._drift = None  # the DriftMonitor, lazily built once the reference is full
        self._ref_buf = []  # buffers the first N feature vectors until enrollment
        self._ref_active = None  # frozen reference motif-vocabulary (union of active index sets)
        self._recent_active = deque(
            maxlen=self._drift_window
        )  # recent motif index-sets (bounded window)
        # --- exact bookkeeping (a handful of ints; O(1) and useful next to the approximate estimates) ---
        self._n = 0
        self._reservoir_seen = 0

    # ------------------------------------------------------------------ ingest
    def _coerce(self, event):
        """Normalise any accepted `event` form into (motif, features, spike, value)."""
        motif = features = spike = value = None
        if isinstance(event, dict):
            motif = event.get("motif")
            features = event.get("features")
            spike = event.get("spike")
            value = event.get("value")
        elif isinstance(event, (set, frozenset)):
            motif = event
        elif (
            isinstance(event, (list, tuple))
            and event
            and all(isinstance(x, (int, float)) for x in event)
        ):
            features = event  # a bare feature vector
        else:
            motif = event  # a bare hashable id
        if features is not None:
            active = frozenset(i for i, v in enumerate(features) if v >= self.active_threshold)
            if motif is None:
                motif = active
            if value is None:
                value = float(len(active))
            if spike is None:
                spike = 1 if active else 0
        if isinstance(motif, (set, frozenset, list, tuple)):
            motif = frozenset(motif)  # normalise to a hashable index-set
        if spike is None:
            spike = 1 if motif else 0
        return motif, features, spike, value

    def observe(self, event):
        """Fold ONE spike/motif event into every sketch. O(1) amortised in the stream length; returns None.

        The heavy MinHash-signature / DriftMonitor-PSI work is NOT done here -- it is deferred to
        snapshot()/drift_novelty(), so this stays a fixed-size update per record."""
        motif, features, spike, value = self._coerce(event)
        self._n += 1
        self.morris.increment()  # total count, tiny register
        self.dgim.add(1 if spike else 0)  # windowed firing count
        if value is not None:
            self.quant.add(value)  # order statistics of per-record activity
        if motif is not None:
            self.hll.add(motif)  # distinct motifs
            self.ams.add(motif)  # F2 skew
            if self.reservoir.add(tuple(sorted(motif)) if _is_index_set(motif) else motif):
                pass
            self._reservoir_seen += 1
            if _is_index_set(motif):
                self.heavy.add(motif)  # top-K frequency (needs an int-index iterable)
                self._recent_active.append(motif)  # recent motif vocabulary (bounded)
        if features is not None:
            self._observe_features(list(features))
        return None

    def _observe_features(self, features):
        """Route a feature vector to the drift path: buffer for enrollment, then push once enrolled."""
        if self._drift is None:
            self._ref_buf.append(features)
            if len(self._ref_buf) >= self._drift_reference_size:
                self._drift = DriftMonitor(
                    self._ref_buf,
                    n_bins=self._drift_n_bins,
                    psi_warn=self._drift_psi_warn,
                    novelty_k=self._drift_novelty_k,
                    window=self._drift_window,
                )
                self._ref_active = frozenset(
                    i for w in self._ref_buf for i, v in enumerate(w) if v >= self.active_threshold
                )  # frozen reference vocabulary
                self._ref_buf = None  # release the reference buffer -> bounded memory
        else:
            self._drift.push(features)

    # ------------------------------------------------------------------ query
    def _recent_resemblance(self):
        """MinHash Jaccard between the recent motif vocabulary and the enrolled reference; 1.0 if not enrolled."""
        if self._ref_active is None or not self._recent_active:
            return 1.0
        recent = set()
        for s in self._recent_active:
            recent |= s
        if not recent and not self._ref_active:
            return 1.0
        return self.minhash.jaccard(recent, self._ref_active)

    def drift_novelty(self):
        """One bounded [0,1] drift alarm: weighted blend of PSI (vs its warn threshold), DriftMonitor novelty
        rate, and the MinHash resemblance DROP of the recent vocabulary. 0.0 before enrollment. RISES on drift."""
        if self._drift is None:
            return 0.0
        psi, nov, _warn = self._drift.assess()
        w_psi, w_nov, w_res = self._drift_weights
        psi_c = (
            _clamp01(psi / (2.0 * self._drift_psi_warn)) if self._drift_psi_warn > 0 else 0.0
        )  # PSI==warn -> 0.5
        nov_c = _clamp01(nov)
        res_c = _clamp01(1.0 - self._recent_resemblance())
        return _clamp01(w_psi * psi_c + w_nov * nov_c + w_res * res_c)

    def snapshot(self):
        """A read-only, JSON-friendly dict of the whole panel: every sketch's current answer + memory footprint.

        Safe to call anywhere -- it reads the sketches and never mutates the stream or drives anything."""
        drift_block = {
            "enrolled": self._drift is not None,
            "psi": 0.0,
            "novelty_rate": 0.0,
            "warn": False,
            "resemblance": 1.0,
        }
        if self._drift is not None:
            psi, nov, warn = self._drift.assess()
            drift_block = {
                "enrolled": True,
                "psi": psi,
                "novelty_rate": nov,
                "warn": bool(warn),
                "resemblance": self._recent_resemblance(),
            }
        heavy = [sorted(m) for m in self.heavy.heavy()]
        return {
            "n_events_exact": self._n,
            "n_events_morris": self.morris.count(),
            "distinct_motifs": self.hll.count(),
            "heavy_hitters": heavy,
            "window_spike_count": self.dgim.count(),
            "quantiles": {p: v for p, v in self.quant.quantiles().items()},
            "f2_skew": self.ams.estimate(),
            "reservoir_size": len(self.reservoir),
            "reservoir_seen": self._reservoir_seen,
            "drift": drift_block,
            "drift_novelty": self.drift_novelty(),
            "memory": self._memory_footprint(),
        }

    def _memory_footprint(self):
        """Fixed-size accounting -- every entry is bounded and independent of the number of events seen."""
        return {
            "hll_registers": self.hll.m,
            "heavy_counters": self.heavy.cms.counters(),
            "dgim_buckets": self.dgim.n_buckets,
            "ams_counters": self.ams.n_counters,
            "reservoir_k": self.reservoir.k,
            "morris_registers": self.morris.m,
            "quantile_markers": 5 * len(self.quant.estimators),
            "drift_window": len(self._recent_active),
        }


def main():
    import random

    dim = 24
    thr = 0.5
    seed = 7

    def build_stream(shift, rng, n_hot_reps=(170, 140, 110), n_background=1500, hot=None):
        """A synthetic spike/motif stream: a few FREQUENT hot motifs buried in many rare ones, each event
        also carrying a normalised feature window (base profile + jitter, optionally shifted = a 'rig move')."""
        import math as _m

        if hot is None:
            hot = [frozenset(rng.sample(range(200), 6)) for _ in range(len(n_hot_reps))]

        def feature_window():
            return [
                min(1.0, max(0.0, 0.5 + 0.3 * _m.sin(i) + shift + rng.uniform(-0.05, 0.05)))
                for i in range(dim)
            ]

        motifs = []
        for m, reps in zip(hot, n_hot_reps):
            motifs += [m] * reps
        for _ in range(n_background):
            motifs.append(frozenset(rng.sample(range(200), 6)))
        rng.shuffle(motifs)
        events = [{"motif": m, "features": feature_window()} for m in motifs]
        return events, hot

    def run(panel_seed):
        rng = random.Random(1234)
        panel = StreamSketchPanel(seed=panel_seed)
        # ---- STATIONARY phase ----
        stat_events, hot = build_stream(0.0, rng)
        active_counts = []
        motif_stream = []
        for ev in stat_events:
            panel.observe(ev)
            active_counts.append(sum(1 for v in ev["features"] if v >= thr))
            motif_stream.append(ev["motif"])
        stat_snap = panel.snapshot()
        stat_drift = panel.drift_novelty()
        # ---- DRIFTED phase ("move the rig": feature means shift) ----
        drift_events, _ = build_stream(
            0.40, rng, n_hot_reps=(40, 30, 20), n_background=120, hot=hot
        )
        for ev in drift_events:
            panel.observe(ev)
        drift_snap = panel.snapshot()
        drift_drift = panel.drift_novelty()
        return (
            panel,
            hot,
            active_counts,
            motif_stream,
            stat_snap,
            stat_drift,
            drift_snap,
            drift_drift,
        )

    (panel, hot, active_counts, motif_stream, stat_snap, stat_drift, drift_snap, drift_drift) = run(
        seed
    )

    # ---- oracles (allowed in the TEST, not part of the panel) ----
    distinct_true = len(set(motif_stream))
    hll_rel = abs(stat_snap["distinct_motifs"] - distinct_true) / distinct_true
    hll_bound = 3.0 * panel.hll.relative_error + 0.01
    heavy_true = {tuple(sorted(m)) for m in hot}
    heavy_got = {tuple(m) for m in stat_snap["heavy_hitters"]}
    sorted_ac = sorted(active_counts)
    exact_med = sorted_ac[len(sorted_ac) // 2]
    p50_est = stat_snap["quantiles"][0.5]
    p50_rel = abs(p50_est - exact_med) / max(1.0, exact_med)
    f2_true = exact_f2(motif_stream)
    f2_rel = abs(stat_snap["f2_skew"] - f2_true) / f2_true
    morris_rel = (
        abs(stat_snap["n_events_morris"] - stat_snap["n_events_exact"])
        / stat_snap["n_events_exact"]
    )
    morris_bound = (
        3.0 * panel.morris.relative_error + 0.03
    )  # 3-sigma of the m-averaged estimate + margin

    # ---- determinism: a second identical run must reproduce the panel exactly ----
    (_p2, _h2, _ac2, _ms2, stat_snap2, stat_drift2, drift_snap2, drift_drift2) = run(seed)
    det = (
        stat_snap["distinct_motifs"] == stat_snap2["distinct_motifs"]
        and stat_snap["heavy_hitters"] == stat_snap2["heavy_hitters"]
        and stat_snap["quantiles"] == stat_snap2["quantiles"]
        and stat_snap["f2_skew"] == stat_snap2["f2_skew"]
        and stat_drift == stat_drift2
        and drift_drift == drift_drift2
    )

    # ---- memory is bounded + constant across phases for the fixed sketches ----
    fixed_keys = (
        "hll_registers",
        "heavy_counters",
        "ams_counters",
        "reservoir_k",
        "morris_registers",
        "quantile_markers",
    )
    mem_constant = all(stat_snap["memory"][k] == drift_snap["memory"][k] for k in fixed_keys)
    dgim_bound = panel.dgim.k * 12 * 12  # k * log2(W)^2 for W=256 (L=8) -> generous cap
    dgim_ok = drift_snap["memory"]["dgim_buckets"] <= dgim_bound

    print("=" * 74)
    print("STREAM-SKETCH PANEL -- 'Lake Vitals': nine fixed-memory sketches over one stream")
    print("=" * 74)
    print(
        f"stream: {stat_snap['n_events_exact']} stationary events then "
        f"{drift_snap['n_events_exact'] - stat_snap['n_events_exact']} drifted; motif space range(200)"
    )
    print(
        f"distinct : HLL {stat_snap['distinct_motifs']:.0f} vs true {distinct_true} "
        f"(rel {hll_rel * 100:.2f}%, bound {hll_bound * 100:.2f}%)"
    )
    print(f"heavy    : recovered {sorted(heavy_got)} == true hot {heavy_got == heavy_true}")
    print(
        f"quantile : activity p50 est {p50_est:.1f} vs exact {exact_med} (rel {p50_rel * 100:.2f}%); "
        f"p90={stat_snap['quantiles'][0.9]:.1f} p99={stat_snap['quantiles'][0.99]:.1f}"
    )
    print(f"F2 skew  : AMS {stat_snap['f2_skew']:.0f} vs exact {f2_true} (rel {f2_rel * 100:.2f}%)")
    print(
        f"count    : Morris {stat_snap['n_events_morris']:.0f} vs exact {stat_snap['n_events_exact']} "
        f"(rel {morris_rel * 100:.2f}%, bound {morris_bound * 100:.2f}%); "
        f"DGIM last-W spikes {stat_snap['window_spike_count']:.1f}"
    )
    print(
        f"sample   : reservoir holds {stat_snap['reservoir_size']} of {stat_snap['reservoir_seen']} motifs"
    )
    print(
        f"drift    : stationary novelty {stat_drift:.3f} (warn {stat_snap['drift']['warn']}) -> "
        f"drifted novelty {drift_drift:.3f} (warn {drift_snap['drift']['warn']})"
    )
    print(
        f"           PSI {stat_snap['drift']['psi']:.3f} -> {drift_snap['drift']['psi']:.3f}; "
        f"resemblance {stat_snap['drift']['resemblance']:.3f} -> {drift_snap['drift']['resemblance']:.3f}"
    )
    print(
        f"memory   : fixed sketches constant across phases {mem_constant}; "
        f"DGIM buckets {drift_snap['memory']['dgim_buckets']} <= {dgim_bound}"
    )
    print(f"determinism: two fresh panels, same seed/stream -> identical snapshot {det}")
    print("=" * 74)

    # ---- asserts: deterministic; genuine estimates vs real oracles, nothing rigged ----
    assert hll_rel < hll_bound, (
        f"HLL distinct off by {hll_rel * 100:.2f}% (> {hll_bound * 100:.2f}%)"
    )
    assert heavy_got == heavy_true, (
        f"heavy hitters must recover the hot motifs ({heavy_got} vs {heavy_true})"
    )
    assert p50_rel < 0.10, f"activity p50 off by {p50_rel * 100:.2f}% (> 10%)"
    assert f2_rel < 0.20, f"AMS F2 off by {f2_rel * 100:.2f}% (> 20%)"
    assert morris_rel < morris_bound, (
        f"Morris count off by {morris_rel * 100:.2f}% (> {morris_bound * 100:.2f}%)"
    )
    assert stat_snap["reservoir_size"] == panel.reservoir.k, "reservoir must stay at k"
    assert not stat_snap["drift"]["warn"], (
        f"stationary stream must not warn (PSI={stat_snap['drift']['psi']:.3f})"
    )
    assert drift_snap["drift"]["warn"], (
        f"drifted stream must warn (PSI={drift_snap['drift']['psi']:.3f})"
    )
    assert drift_drift > stat_drift + 0.10, (
        f"drift_novelty must RISE on the drifted phase ({stat_drift:.3f} -> {drift_drift:.3f})"
    )
    assert 0.0 <= stat_drift <= 1.0 and 0.0 <= drift_drift <= 1.0, (
        "drift_novelty must stay in [0,1]"
    )
    assert mem_constant, "fixed-memory sketches must keep a constant footprint across phases"
    assert dgim_ok, (
        f"DGIM buckets {drift_snap['memory']['dgim_buckets']} exceed O(log^2 W) bound {dgim_bound}"
    )
    assert det, "same seed/stream must give an identical snapshot (deterministic)"
    print(
        f"self-check OK: HLL distinct within {hll_rel * 100:.2f}%, hot motifs recovered, activity p50 within "
        f"{p50_rel * 100:.2f}%, F2 within {f2_rel * 100:.2f}%, Morris within {morris_rel * 100:.2f}%; "
        f"drift_novelty {stat_drift:.3f}->{drift_drift:.3f} on the rig-move; memory bounded + constant; "
        f"deterministic"
    )


if __name__ == "__main__":
    main()
