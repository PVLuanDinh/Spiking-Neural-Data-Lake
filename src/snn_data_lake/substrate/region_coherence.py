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
    Strogatz, Physica D 143:1, 2000; the exact primitive reused from cerebellum.kuramoto_sync). Each region's
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
and clamped.

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

    def __init__(self, seed=0, window=32, min_window=4, w_phase=0.7, w_integ=0.3, gain=0.5):
        self.seed = int(seed)
        self.window = int(window)  # rolling-history cap for scalar-rate inputs
        self.min_window = int(min_window)  # shortest array treated as a direct phase window
        self.w_phase = float(w_phase)  # weight on synchrony (dominant)
        self.w_integ = float(w_integ)  # weight on the integration/complexity proxy
        self.gain = float(gain)  # how strongly coherence lowers the ignition threshold
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
    def _adjust(base_threshold, coherence, gain=0.5):
        """Map coherence -> adjusted GWT rheobase. Higher coherence LOWERS the threshold (easier
        ignition), strictly monotone decreasing in coherence, clamped to (0, base]."""
        c = min(1.0, max(0.0, float(coherence)))
        return float(base_threshold * (1.0 - gain * c))

    def ignition_bias(self, base_threshold):
        """Adjusted ignition threshold given the last coherence: higher coherence -> lower threshold."""
        return self._adjust(base_threshold, self._coherence, self.gain)


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
    biases = [RegionCoherence._adjust(base, c, 0.5) for c in sweep]
    monotone = all(b2 < b1 for b1, b2 in zip(biases, biases[1:], strict=False))

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
    # (d) determinism.
    assert det_a == det_b, "same seed + identical input must reproduce an identical result"
    # returned contract.
    for key in ("coherence", "phase_order", "n_regions"):
        assert key in sync_out, f"update() must return '{key}'"
    assert sync_out["n_regions"] == k, "n_regions must equal the number of regions supplied"
    assert 0.0 <= sync_out["coherence"] <= 1.0, "coherence must lie in [0,1]"

    print("")
    print(
        f"self-check OK: synchronized phase_order {sync_out['phase_order']:.3f} (coherence "
        f"{sync_out['coherence']:.3f}) >> desynchronized {desync_out['phase_order']:.3f}; silent bank "
        f"{silent_out['phase_order']:.3f} not spuriously coherent; ignition_bias monotone (base {base} -> "
        f"{bias_hi:.3f} at coherence {c_hi:.3f} < {bias_lo:.3f} at {c_lo:.3f}); deterministic; NOT IIT Phi, "
        f"NOT consciousness."
    )


if __name__ == "__main__":
    main()
