"""
Cross-region coherence / synchrony over LIVE region activity -- a Kuramoto phase-order parameter plus a
tractable integration (PCI-style co-activation complexity) signal, yielding one scalar `coherence` that
can gate a GWT ignition threshold.

WHAT THIS MEASURES -- READ THIS FIRST (honest, agnostic framing)
===============================================================
Global-workspace theory says a content reaches "global availability" when many regions IGNITE together --
a large-scale, integrated, synchronized coalition. This module turns that into a measured substrate/dynamics
signal over the brain's per-region spike activity each tick. It combines two complementary primitives:

  * PHASE SYNCHRONY -- the Kuramoto order parameter R = |mean_j e^{i*phase_j}| in [0,1] (Kuramoto 1975;
    Strogatz, Physica D 143:1, 2000; the self-contained live Kuramoto primitive -- it superseded and retired
    the standalone cerebellum.kuramoto_sync). Each region's
    instantaneous PHASE is read from its recent activity via the FFT-based analytic (Hilbert) signal of the
    mean-removed window -- the standard "Hilbert-like" instantaneous-phase estimate, robust to the oscillation
    frequency. R ~ 1 when the regions are phase-locked (synchronized), R near the finite-N floor ~1/sqrt(N)
    when their phases are scattered.

  * INTEGRATION / DIFFERENTIATION -- a lightweight perturb-free proxy: the normalized entropy of the spatial
    CO-ACTIVATION patterns across regions over the window (how many distinct region-coalitions the tick
    visits), cross-checked with the Casali normalized Lempel-Ziv complexity of the binarized co-activation
    raster (the primitive reused from cortex.integration_measure). This rewards a rich, differentiated
    coalition structure, not a single frozen pattern.

The scalar `coherence` = a phase-weighted blend of the two (synchrony dominates; the complexity term
modulates), clamped to [0,1]. `ignition_bias(base)` maps it to a GWT rheobase: higher coherence LOWERS the
ignition threshold (a synchronized, integrated brain ignites its global workspace more readily), monotone
and clamped. The SIZE of that discount is a single CALIBRATED constant, `RegionCoherence.IGNITION_GAIN`,
measured against the coherence distribution the live brain actually produces (and against the coalition
drives it is compared with) -- read its docstring for the numbers, the defensible band, and the honest
limit. It is calibrated, NOT learned: an adaptive rheobase would destroy cross-tick comparability of
"did it ignite?", which is the whole point of having a fixed yardstick.

This is a DYNAMICS / SUBSTRATE property -- large-scale phase coherence with a tractable integration proxy --
REPORTED as an ignition-gating signal. It is NEVER asserted to equal consciousness, and it is NOT IIT Phi
(intractable). The project's stance on consciousness is AGNOSTIC: this neither denies nor claims it; it
reports a coherence number and names it exactly. An honest anti-rig guard is built in: a bank of SILENT or
CONSTANT regions has no oscillation to phase-lock, so each is assigned a crc32-scattered surrogate phase and
the bank reads as INCOHERENT (low R) rather than spuriously synchronized -- coherence must be earned by real,
differentiated, phase-locked activity, not by everyone sitting still at zero.

Representation-only: no arm import, nothing under brainstem/. numpy is a core dependency and used freely.
Deterministic: per-region surrogate phases via zlib.crc32 (never builtin hash()); no global seeding. ASCII.

  PYTHONPATH=src python -m snn_data_lake.substrate.region_coherence        # (self-check, exit 0)
"""

import math
import zlib

import numpy as np

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "RegionCoherence.update": (
        "NOT_A_RULE",
        "consumes {region: window|rate} and returns {coherence, phase_order, n_regions, integration, "
        "integration_lz}. A METRIC computed deterministically from its inputs -- no weight, "
        "threshold or trace is updated, so there is nothing a cosine could measure. Enumerated only "
        "because the walk keys on the verb 'update'",
    ),
}

# Reuse the tractable integration primitive from the PCI module (Casali normalized Lempel-Ziv complexity).
try:
    from snn_data_lake.cortex.integration_measure import normalized_lz
except Exception:  # pragma: no cover - defensive fallback keeps this module self-contained

    def normalized_lz(s):
        """Fallback normalized-LZ (Casali) if the cortex primitive is unavailable at import time."""
        L = len(s)
        if L == 0 or s.count("1") == 0:
            return 0.0
        # Lempel-Ziv 1976 phrase count c(L).
        i, k, l, k_max, c = 0, 1, 1, 1, 1
        while True:
            if s[i + k - 1] == s[l + k - 1]:
                k += 1
                if l + k > L:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > L:
                        break
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        return c * math.log2(L) / L


class RegionCoherence:
    """Cross-region coherence over live per-region activity.

    update(region_rasters) -- region_rasters is a dict {region_name: window|rate}, where the value is either
    an array-like of recent spike counts / rates (a per-region raster window) OR a single scalar rate. Scalar
    rates accumulate into a bounded per-region rolling window internally, so the caller may feed one number
    per region per tick. Returns {coherence, phase_order, n_regions, integration, integration_lz}.

    `coherence` (property) is the last scalar in [0,1]. `ignition_bias(base_threshold)` returns an adjusted
    GWT rheobase: higher coherence lowers the threshold (easier ignition), monotone and clamped.
    """

    # -- THE single definition of the ignition-bias gain -----------------------------------------
    # Never re-type this number: `_adjust` takes `gain` as a REQUIRED argument and `__init__`
    # defaults to this constant, so there is exactly one place to change it. (A duplicated default
    # here previously let an edit in one place silently no-op the other.)
    IGNITION_GAIN = 0.5
    """How strongly coherence lowers the GWT rheobase: thr = base_threshold * (1 - IGNITION_GAIN*c).

    CALIBRATED, and deliberately NOT LEARNED. The rheobase is the YARDSTICK that decides "did this
    tick ignite?"; making the yardstick itself adaptive would destroy cross-tick comparability of the
    answer. So the FORM stays fixed (memoryless, bounded, monotone, deterministic) and only the SIZE
    of the discount is calibrated -- against the coherence this brain actually produces.

    MEASURED (throwaway probe: cortex.brain.Brain wake sessions, seeds {0,1,2,3,5} x 120 forced-day
    ticks with a varying synthetic percept, n=600 samples of Brain._region_coherence_val):
        min 0.138 | p05 0.259 | p25 0.390 | median 0.500 | mean 0.488 | p75 0.571 | p95 0.697
        | max 0.896 | sd 0.134   -- mass concentrated in [0.3, 0.7]; c=0 and c=1 NEVER occur.
    Coalition drives from the same sessions (the values this rheobase is compared against): the
    WINNING coalition's bottom-up salience spans [0.668, 1.000] (median 0.815); the MASKED/losing
    coalitions span [0.086, 0.500] (median 0.121, p75 0.167).

    WHY 0.5 -- over the coherence that actually occurs it sweeps the rheobase across [0.304, 0.512]
    from the 0.55 base. That band sits strictly BELOW every winning drive ever observed (0.668 -- the
    workspace never darkens) and strictly ABOVE the p75 of the masked drives (0.167 -- a weak
    coalition is not handed access for free). Across the p05..p95 coherence bulk the threshold moves
    0.479 -> 0.358, a 25% relative discount: big enough that the gate demonstrably covaries with
    coherence, small enough that ignition still has to be earned. Those two constraints bound a
    defensible band of gain in [0.40, 0.78): under 0.40 the bulk discount drops below 20% relative
    and the gate barely does anything; at ~0.78 and above, the threshold at the highest observed
    coherence falls under the masked-drive p75 and stops discriminating (at gain 1.0 it reaches
    0.057 -- below EVERY drive ever observed, i.e. fully open and carrying no information). 0.5 sits
    inside that band on the conservative side, so the previously unjustified 0.5 SURVIVES the
    measurement and is kept here WITH its evidence rather than re-picked.

    HONEST LIMIT -- in the sessions measured the workspace ignited on 100% of wake ticks at EVERY
    gain in [0.0, 1.0], including gain 0.0 (no bias at all), with exactly one coalition sustaining
    and the winner's spike count saturated. At those drive levels single-winner selectivity is
    enforced by the workspace's lateral inhibition, not by the rheobase. This constant is therefore
    calibrated on the threshold's MARGIN against the measured drives, NOT on a measured change in
    ignition outcome; no value in [0.0, 1.0] was observed to change what the brain does today.
    """

    def __init__(self, seed=0, window=32, min_window=4, w_phase=0.7, w_integ=0.3, gain=None):
        self.seed = int(seed)
        self.window = int(window)  # rolling-history cap for scalar-rate inputs
        self.min_window = int(min_window)  # shortest array treated as a direct phase window
        self.w_phase = float(w_phase)  # weight on synchrony (dominant)
        self.w_integ = float(w_integ)  # weight on the integration/complexity proxy
        self.gain = float(self.IGNITION_GAIN if gain is None else gain)  # see IGNITION_GAIN
        # THE GAIN'S RANGE IS ENFORCED AT ITS ONE DOOR (the T032 theta-authority pattern):
        # _adjust's positivity bound derives entirely from gain < 1 (result in
        # [base*(1-gain), base]), so gain >= 1 makes the adjusted rheobase <= 0 at full
        # coherence -- the competition dissolves, the structural death set_base_rheobase's
        # own range names -- and gain < 0 RAISES the threshold with coherence, inverting
        # the mechanism's stated direction (a sign error upstream, refused not absorbed).
        if not (0.0 <= self.gain < 1.0):
            raise ValueError(
                f"RegionCoherence(gain={self.gain}): gain must be in [0, 1) -- at >= 1 the "
                "adjusted rheobase reaches <= 0 at full coherence (structurally dead), and "
                "below 0 coherence would RAISE the threshold (the mechanism inverted)"
            )
        self._hist = {}  # region -> rolling list of scalar rates
        self._coherence = 0.0
        self._phase_order = 0.0
        self._n_regions = 0

    # ---- properties -------------------------------------------------------------------------------
    @property
    def coherence(self):
        """Last computed coherence in [0,1]."""
        return self._coherence

    @property
    def phase_order(self):
        """Last computed Kuramoto order parameter R in [0,1]."""
        return self._phase_order

    @property
    def n_regions(self):
        """Number of regions in the last update."""
        return self._n_regions

    # ---- per-region window + phase ----------------------------------------------------------------
    def _window_for(self, region, value):
        """Return a float window for `region`. An array-like of length >= min_window is used directly
        (its mean is also recorded into the rolling history); a scalar (or a too-short array) is appended
        to the bounded rolling history and that history is returned."""
        hist = self._hist.setdefault(region, [])
        if np.ndim(value) > 0:
            arr = [float(x) for x in np.asarray(value, dtype=float).ravel()]
            if len(arr) >= self.min_window:
                hist.append(sum(arr) / len(arr))
                if len(hist) > self.window:
                    hist.pop(0)
                return arr
            scal = arr[-1] if arr else 0.0
        else:
            scal = float(value)
        hist.append(scal)
        if len(hist) > self.window:
            hist.pop(0)
        return list(hist)

    def _surrogate_phase(self, region):
        """A deterministic crc32-scattered surrogate phase in [-pi, pi) for a silent/constant region
        (no oscillation to lock). Distinct regions scatter, so a still/silent bank reads as INCOHERENT
        rather than spuriously synchronized. Uses zlib.crc32 (never builtin hash())."""
        h = zlib.crc32(f"{self.seed}:{region}".encode("ascii"))
        return (h / 4294967296.0) * (2.0 * math.pi) - math.pi

    @staticmethod
    def _phase(window):
        """Instantaneous phase at the latest sample from the FFT-based analytic (Hilbert) signal of the
        mean-removed window -- the standard "Hilbert-like" instantaneous-phase estimate, robust to the
        oscillation frequency. Returns None for a constant/silent window (no phase defined)."""
        x = np.asarray(window, dtype=float)
        n = x.size
        if n < 2:
            return None
        xc = x - x.mean()
        if not np.any(np.abs(xc) > 1e-12):
            return None
        xf = np.fft.fft(xc)
        hfilt = np.zeros(n)
        hfilt[0] = 1.0
        if n % 2 == 0:
            hfilt[n // 2] = 1.0
            hfilt[1 : n // 2] = 2.0
        else:
            hfilt[1 : (n + 1) // 2] = 2.0
        z = np.fft.ifft(xf * hfilt)[-1]
        if abs(z) < 1e-12:
            return None
        return float(np.angle(z))

    @staticmethod
    def _order(phases):
        """Kuramoto order parameter R = |mean_j e^{i*phase_j}| in [0,1] (reused primitive)."""
        if not phases:
            return 0.0
        z = np.mean(np.exp(1j * np.asarray(phases, dtype=float)))
        return float(np.abs(z))

    # ---- integration / differentiation proxy ------------------------------------------------------
    def _integration(self, windows):
        """A lightweight integration/differentiation proxy over the co-activation of the region windows.
        Binarize each region over time (active iff above its own window mean); read the spatial pattern
        across regions at each timestep. Return (entropy, lz): the normalized Shannon entropy of the
        distinct spatial co-activation patterns, and the Casali normalized-LZ of the time-major binarized
        raster (reused from cortex.integration_measure) as a compression cross-check. Both in [0,1]."""
        usable = [w for w in windows if len(w) >= 2]
        if len(usable) < 2:
            return 0.0, 0.0
        length = min(len(w) for w in usable)
        if length < 2:
            return 0.0, 0.0
        bitrows = []
        for w in usable:
            ww = np.asarray(w[:length], dtype=float)
            m = ww.mean()
            bitrows.append([1 if v > m else 0 for v in ww])
        patterns = [tuple(row[t] for row in bitrows) for t in range(length)]
        counts = {}
        for p in patterns:
            counts[p] = counts.get(p, 0) + 1
        probs = [c / length for c in counts.values()]
        h = -sum(p * math.log2(p) for p in probs if p > 0.0)
        h_max = math.log2(length)  # at most `length` distinct patterns observable
        entropy = h / h_max if h_max > 0.0 else 0.0
        s = "".join("".join(str(row[t]) for row in bitrows) for t in range(length))
        lz = normalized_lz(s)
        return float(min(1.0, entropy)), float(min(1.0, lz))

    # ---- the tick update --------------------------------------------------------------------------
    def update(self, region_rasters):
        """Consume a dict {region: window|rate}; return {coherence, phase_order, n_regions, integration,
        integration_lz}. Deterministic given the inputs (and seed, for silent-region surrogate phases)."""
        regions = sorted(region_rasters)  # deterministic order
        phases, windows = [], []
        for r in regions:
            w = self._window_for(r, region_rasters[r])
            windows.append(w)
            ph = self._phase(w)
            if ph is None:
                ph = self._surrogate_phase(
                    r
                )  # silent/constant -> scattered, not spuriously in-phase
            phases.append(ph)
        n = len(regions)
        r_order = self._order(phases) if n else 0.0
        entropy, lz = self._integration(windows) if n >= 2 else (0.0, 0.0)
        coherence = float(np.clip(self.w_phase * r_order + self.w_integ * entropy, 0.0, 1.0))
        self._phase_order = r_order
        self._coherence = coherence
        self._n_regions = n
        return {
            "coherence": round(coherence, 6),
            "phase_order": round(r_order, 6),
            "n_regions": n,
            "integration": round(entropy, 6),
            "integration_lz": round(lz, 6),
        }

    # ---- ignition gating --------------------------------------------------------------------------
    @staticmethod
    def _adjust(base_threshold, coherence, gain, interoceptive=0.0):
        """Map coherence -> adjusted GWT rheobase. Higher coherence LOWERS the threshold (easier
        ignition), strictly monotone decreasing in coherence.

        TWO REASONS TO IGNITE, ONE LAW (specs/022 T039f). `interoceptive` is the insula's
        feeling_salience -- how far the body sits from its OWN running baseline -- and a body far
        from baseline should find it easier to ignite the workspace, which is that producer's own
        stated claim. It joins HERE rather than through a second actuator because the tree already
        modulates this threshold through this expression and P2 forbids a mechanism beside the one
        that exists.

        THE COMBINATION IS NOISY-OR AND CARRIES NO NEW CONSTANT:

            c_eff = 1 - (1 - c) * (1 - s)

        i.e. the threshold is eased to the degree that AT LEAST ONE independent reason to ignite
        is present. This is the only form that satisfies all four constraints IGNITION_GAIN's
        docstring imposes on this yardstick -- bounded, monotone in each input, memoryless,
        deterministic -- while introducing nothing to calibrate. A weighted sum would have needed a
        second calibrated weight, and `max()` would have discarded the weaker reason entirely.

        THE BOUND SURVIVES UNCHANGED, which is why this is safe to fold in here: c_eff lies in
        [0, 1] whenever c and s do, so the result still lies in [base * (1 - gain), base] and is
        still strictly positive IFF gain < 1 -- the same precondition, enforced at the same one
        door (__init__). Nothing downstream widens.

        WITH s = 0 THIS IS TODAY'S EXPRESSION EXACTLY (c_eff == c), so an absent or unfed insula
        leaves every threshold byte-identical -- the term is additive from zero, and that is what
        makes its attribution measurable rather than argued.

        THE BOUND IS A PRECONDITION, NOT A CLAMP, and it is stated exactly (an earlier
        docstring said 'clamped to (0, base]' while the body clamps nothing -- the same
        claim-wearing-a-derivation's-clothes shape the T032 review reddened in
        reentrant_workspace, corrected here the same way). With `c` clamped to [0, 1]
        below, the result is `base * (1 - gain*c)`, which lies in
        [base * (1 - gain), base] -- strictly positive IFF gain < 1. That precondition is
        ENFORCED at the constant's one door (__init__ refuses gain outside [0, 1)), never
        silently absorbed here: a result that somehow escaped the band would raise loudly
        downstream, where reentrant_workspace.coherence_theta range-refuses (0, _COAL_CAP]
        out of Brain.tick() -- the loud path is the guard, a quiet clamp would hide the
        upstream computation error it exists to expose.

        `gain` is REQUIRED -- it has NO default here on purpose. The one definition lives in
        RegionCoherence.IGNITION_GAIN (with its measured justification), so re-calibrating it cannot
        be silently no-opped by a stale second default sitting in this signature."""
        c = min(1.0, max(0.0, float(coherence)))
        s = min(1.0, max(0.0, float(interoceptive)))
        c_eff = 1.0 - (1.0 - c) * (1.0 - s)
        return float(base_threshold * (1.0 - gain * c_eff))

    def ignition_bias(self, base_threshold, interoceptive=0.0, coherence=None):
        """Adjusted ignition threshold: higher coherence OR a body further from its own baseline
        -> lower threshold.

        `coherence=None` uses the last measured value, which is every pre-T039f caller unchanged.
        A caller that has ALREADY GATED coherence passes the gated value explicitly -- cortex.brain
        does, because its synfire binding gate decides whether a coherence volley was reliable
        enough to count, and that decision belongs where the synfire chain is, not here. Passing
        0.0 is how the caller says "coherence contributes nothing this tick", and the
        interoceptive term is unaffected by it: the body does not stop deviating from baseline
        because a cross-region volley failed to bind."""
        c = self._coherence if coherence is None else coherence
        return self._adjust(base_threshold, c, self.gain, interoceptive)


#: the declared edge this module CONSUMES -- the insula's OUT port, edge-less since T026, lands here
INTEROCEPTION_EDGE = (
    "interoceptive_integration",
    "interoception_out",
    "region_coherence",
    "interoception_in",
)


def declare_ports(manifest):
    """Manifest registrant (specs/022 T039j): the ignition law's interoceptive input.

    This DISCHARGES the edge-less disposition `interoceptive_integration.interoception_out` has
    carried since T026. The port sits here rather than on the `gwt` node because this is where the
    salience is READ -- `_adjust` folds it into the threshold by noisy-OR -- and the workspace
    downstream receives the threshold, never the salience. join_policy is `max`: a single producer
    today, and if a second interoceptive source is ever declared the strongest body signal should
    set the discount rather than two sources summing past the bound the law depends on."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "region_coherence",
        Port(
            name="interoception_in",
            region="substrate",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="IN",
            declaring_module="region_coherence",
            join_policy="max",
        ),
    )
    manifest.declare_edge(*INTEROCEPTION_EDGE)


def _interoception_consumer_factory():
    """Fresh ring + JoinConsumer over `interoception_in`, one pair per replay arm."""
    from snn_data_lake.insula.interoceptive_integration import (
        declare_ports as _producer_ports,
    )
    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import WiringManifest

    m = WiringManifest()
    _producer_ports(m)
    declare_ports(m)
    port = m.ports[("region_coherence", "interoception_in")]
    ring = SignalRing(slots=256)
    return ring, JoinConsumer(ring, port)


def consume_interoception(consumer):
    """The downstream computation under measurement: each joined salience frame is folded into the
    ignition law and the resulting threshold recorded. Coherence is held at a fixed value across
    both arms so the ONLY thing that can move the output is the interoceptive payload."""
    rc = RegionCoherence(seed=0)
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        thr = rc.ignition_bias(0.55, interoceptive=float(vec[0]), coherence=0.4)
        out.append((tick, round(thr, 9)))
    return out


def _measure_edge_charge():
    """The replay-twice A/B for `interoception_out -> interoception_in` -> (live, pinned, reads,
    responded). `responded` = the threshold FELL at the challenge, which is the direction the
    producer's own docstring claims (a body far from baseline finds it EASIER to ignite)."""
    from snn_data_lake.substrate.port_charge_probe import (
        interoception_challenge_frames,
        replay_twice,
    )

    frames, spike_at = interoception_challenge_frames()
    live, pinned_out, reads = replay_twice(
        frames, _interoception_consumer_factory, consume_interoception, [0.2]
    )
    pre = min(v for t, v in live if spike_at - 5 <= t < spike_at)
    post = min(v for t, v in live if t >= spike_at)
    return live, pinned_out, reads, post < pre


def record_charge():
    """Measure the A/B and record its attribution via THE one writer."""
    from snn_data_lake.substrate.port_charge_probe import record_attribution

    live, pinned_out, reads, responded = _measure_edge_charge()
    if live == pinned_out or not responded:
        print(f"record_charge REFUSED for {INTEROCEPTION_EDGE}: no divergence or no response")
        return 1
    path = record_attribution(
        INTEROCEPTION_EDGE,
        {"reads": reads, "n_frames": len(live)},
        "replay_twice: a REAL insula salience trace vs a pinned constant, coherence held fixed "
        "across both arms so only the interoceptive payload can move the output; the live "
        "threshold diverged and FELL at the challenge -- the direction the producer claims. "
        "Discharges the edge-less disposition interoception_out carried since T026; self-check "
        "re-measures and requires `measured` to reproduce",
    )
    print(f"recorded {INTEROCEPTION_EDGE} into {path}")
    return 0


# ======================================================================================================
# self-check
# ======================================================================================================
def _sine_window(length, phase, freq=1.0):
    """A single-cycle sine window at a given phase offset (a clean per-region oscillation)."""
    n = np.arange(length)
    return list(np.sin(2.0 * np.pi * freq * n / length + phase))


def main():
    length, k = 32, 6

    # (a) SYNCHRONIZED: every region gets the SAME in-phase oscillation -> phases identical -> R ~ 1.
    sync = {f"r{i}": _sine_window(length, 0.0) for i in range(k)}
    sync_out = RegionCoherence(seed=0).update(sync)

    # (b) DESYNCHRONIZED: region i at phase offset 2*pi*i/k -> phases spread as k-th roots of unity -> R ~ 0.
    desync = {f"r{i}": _sine_window(length, 2.0 * math.pi * i / k) for i in range(k)}
    desync_out = RegionCoherence(seed=0).update(desync)

    # honesty guard: a bank of SILENT regions must NOT read as synchronized (crc32-scattered surrogate phases).
    silent = {f"r{i}": [0.0] * length for i in range(k)}
    silent_out = RegionCoherence(seed=0).update(silent)

    # (c) ignition_bias moves monotonically with coherence: strictly decreasing over a coherence sweep,
    #     and unchanged (== base) at coherence 0; the two live states obey the same ordering.
    base = 0.55
    sweep = [0.0, 0.25, 0.5, 0.75, 1.0]
    gain = RegionCoherence.IGNITION_GAIN  # the ONE definition (calibrated; see its docstring)
    biases = [RegionCoherence._adjust(base, c, gain) for c in sweep]
    monotone = all(b2 < b1 for b1, b2 in zip(biases, biases[1:], strict=False))

    # (e) CALIBRATION: the gain must keep the rheobase inside the band that the MEASURED live signals
    #     bound. Numbers from the wake-session probe recorded in RegionCoherence.IGNITION_GAIN --
    #     coherence range actually observed, and the coalition drives the rheobase is compared with.
    MEAS_C_MIN, MEAS_C_P05, MEAS_C_P95, MEAS_C_MAX = 0.138, 0.259, 0.697, 0.896
    MEAS_MASKED_P75 = 0.167  # p75 of the LOSING coalitions' bottom-up drive
    MEAS_WINNER_MIN = 0.668  # weakest WINNING coalition drive ever observed
    thr_c_min = RegionCoherence._adjust(base, MEAS_C_MIN, gain)
    thr_c_max = RegionCoherence._adjust(base, MEAS_C_MAX, gain)
    thr_p05 = RegionCoherence._adjust(base, MEAS_C_P05, gain)
    thr_p95 = RegionCoherence._adjust(base, MEAS_C_P95, gain)
    bulk_discount = (thr_p05 - thr_p95) / thr_p05  # relative rheobase drop over the coherence bulk

    rc = RegionCoherence(seed=0)
    rc.update(sync)
    c_hi, bias_hi = rc.coherence, rc.ignition_bias(base)
    rc.update(desync)
    c_lo, bias_lo = rc.coherence, rc.ignition_bias(base)

    # (d) determinism: same seed + identical input -> identical output.
    det_a = RegionCoherence(seed=0).update(sync)
    det_b = RegionCoherence(seed=0).update(sync)

    print("=" * 84)
    print(
        "REGION COHERENCE -- Kuramoto phase-order R + a tractable integration proxy over live region activity"
    )
    print("=" * 84)
    print(f"regions per case = {k}, window = {length}")
    print(
        f"  synchronized (in-phase) : phase_order R = {sync_out['phase_order']:.3f}  "
        f"coherence = {sync_out['coherence']:.3f}  integ = {sync_out['integration']:.3f}"
    )
    print(
        f"  desynchronized (spread) : phase_order R = {desync_out['phase_order']:.3f}  "
        f"coherence = {desync_out['coherence']:.3f}  integ = {desync_out['integration']:.3f}"
    )
    print(
        f"  silent bank (honesty)   : phase_order R = {silent_out['phase_order']:.3f}  "
        f"(surrogate-scattered, not spuriously synchronized)"
    )
    print(
        f"  ignition_bias(base={base}): coherence {c_hi:.3f} -> {bias_hi:.3f}  "
        f"vs coherence {c_lo:.3f} -> {bias_lo:.3f}   (higher coherence -> lower threshold)"
    )
    print(f"  bias sweep over coherence {sweep} : {[round(b, 3) for b in biases]}")
    print(
        f"  CALIBRATION (gain = {gain}, measured live coherence {MEAS_C_MIN}..{MEAS_C_MAX}): rheobase "
        f"sweeps {thr_c_max:.3f}..{thr_c_min:.3f}, above masked-drive p75 {MEAS_MASKED_P75} and below "
        f"the weakest winning drive {MEAS_WINNER_MIN}; bulk (p05..p95) discount {100 * bulk_discount:.1f}%"
    )

    # ---- asserts ----
    # (a) synchronized -> high phase order.
    assert sync_out["phase_order"] > 0.95, (
        f"in-phase regions must give HIGH phase_order (got {sync_out['phase_order']:.3f})"
    )
    # (b) desynchronized -> low phase order, and far below the synchronized case.
    assert desync_out["phase_order"] < 0.20, (
        f"phase-spread regions must give LOW phase_order (got {desync_out['phase_order']:.3f})"
    )
    assert sync_out["phase_order"] > 4.0 * desync_out["phase_order"] + 0.5, (
        f"synchronized R must dwarf desynchronized R ({sync_out['phase_order']:.3f} vs {desync_out['phase_order']:.3f})"
    )
    # honesty: a silent bank is not spuriously coherent.
    assert silent_out["phase_order"] < sync_out["phase_order"], (
        f"a silent bank must NOT read as synchronized as a real coalition "
        f"({silent_out['phase_order']:.3f} vs {sync_out['phase_order']:.3f})"
    )
    # (c) ignition_bias monotone with coherence; unchanged at coherence 0; live states obey the ordering.
    assert monotone, f"ignition_bias must strictly decrease with coherence (sweep {biases})"
    assert biases[0] == base, f"coherence 0 must leave the threshold unchanged (got {biases[0]})"
    assert c_hi > c_lo, (
        f"synchronized coherence must exceed desynchronized ({c_hi:.3f} vs {c_lo:.3f})"
    )
    assert bias_hi < bias_lo, (
        f"higher coherence must lower the ignition threshold ({bias_hi:.3f} vs {bias_lo:.3f})"
    )
    assert 0.0 < bias_hi <= base and 0.0 < bias_lo <= base, (
        f"adjusted thresholds must stay in (0, base]: {bias_hi:.3f}, {bias_lo:.3f}"
    )
    # (e) the gain stays inside the empirically defensible band (see IGNITION_GAIN for the probe).
    assert thr_c_max > MEAS_MASKED_P75, (
        f"gain {gain} drops the rheobase to {thr_c_max:.3f} at the highest observed coherence, at or "
        f"under the masked coalitions' p75 drive {MEAS_MASKED_P75} -- the gate stops discriminating"
    )
    assert thr_c_min < MEAS_WINNER_MIN, (
        f"gain {gain} leaves the rheobase at {thr_c_min:.3f} at the lowest observed coherence, above "
        f"the weakest winning drive {MEAS_WINNER_MIN} -- the workspace would darken"
    )
    assert bulk_discount >= 0.20, (
        f"gain {gain} discounts the rheobase only {100 * bulk_discount:.1f}% across the p05..p95 "
        f"coherence bulk -- the gate would barely covary with coherence"
    )
    # (d) determinism.
    assert det_a == det_b, "same seed + identical input must reproduce an identical result"
    # returned contract.
    for key in ("coherence", "phase_order", "n_regions"):
        assert key in sync_out, f"update() must return '{key}'"
    assert sync_out["n_regions"] == k, "n_regions must equal the number of regions supplied"
    assert 0.0 <= sync_out["coherence"] <= 1.0, "coherence must lie in [0,1]"

    # THE GAIN RANGE'S REDS, both directions (the _adjust bound is a PRECONDITION enforced
    # at construction, not a clamp -- see _adjust's docstring for the derivation): gain >= 1
    # dissolves the competition at full coherence; gain < 0 inverts the mechanism.
    for bad_gain in (1.0, 1.5, -0.1):
        try:
            RegionCoherence(seed=0, gain=bad_gain)
            raise AssertionError(f"RegionCoherence(gain={bad_gain}) must refuse -- red unreachable")
        except ValueError:
            pass  # the RED, reached: out-of-band gain refused at the constant's one door
    assert RegionCoherence(seed=0, gain=0.0).gain == 0.0, (
        "gain=0.0 (bias OFF, threshold never moves) is inside the band and must construct"
    )

    # ---- THE INTEROCEPTIVE EDGE IS CHARGED, measured (specs/022 T039j) -------------------
    # `interoception_out` carried an edge-less disposition from T026 until this module declared
    # the consumer end. A declared edge that carries nothing is the defect the disposition was
    # honest about, so the edge asserts its own attribution here rather than being taken on the
    # declaration.
    from snn_data_lake.substrate.port_charge_probe import read_attribution

    live, pinned_out, reads, responded = _measure_edge_charge()
    assert reads == len(live), f"{reads} reads != {len(live)} frames"
    assert live != pinned_out, (
        "pinning the interoceptive payload did not change the ignition threshold -- a declared, "
        "not a charged, edge"
    )
    assert responded, (
        "the interoceptive challenge did not LOWER the threshold -- divergence without the "
        "direction the producer claims attributes nothing"
    )
    _rec = read_attribution(INTEROCEPTION_EDGE)
    assert _rec is not None and _rec.get("diverged") is True, (
        f"no committed attribution for {INTEROCEPTION_EDGE} -- regenerate with --record-charge"
    )
    assert _rec.get("measured") == {"reads": reads, "n_frames": len(live)}, (
        f"committed attribution does not reproduce: {_rec.get('measured')}"
    )

    print("")
    print(
        f"interoception edge CHARGED: pinning the insula payload held the threshold while the "
        f"live arm diverged and FELL at the challenge, over {len(live)} frames; committed "
        f"attribution reproduces."
    )
    print(
        f"self-check OK: synchronized phase_order {sync_out['phase_order']:.3f} (coherence "
        f"{sync_out['coherence']:.3f}) >> desynchronized {desync_out['phase_order']:.3f}; silent bank "
        f"{silent_out['phase_order']:.3f} not spuriously coherent; ignition_bias monotone (base {base} -> "
        f"{bias_hi:.3f} at coherence {c_hi:.3f} < {bias_lo:.3f} at {c_lo:.3f}); deterministic; NOT IIT Phi, "
        f"NOT consciousness."
    )


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(record_charge() if "--record-charge" in _sys.argv else (main() or 0))
