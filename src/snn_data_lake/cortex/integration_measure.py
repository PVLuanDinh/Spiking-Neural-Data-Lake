"""
Perturbational Complexity Index (PCI) -- a TRACTABLE, HONESTLY-NAMED integration proxy over a small
spiking network.

WHAT THIS MEASURES -- READ THIS FIRST (non-negotiable framing)
=============================================================
This is a PCI-style perturb-and-compress measure after Casali et al. (Sci Transl Med 5:198ra105, 2013):
perturb a small spiking network with a brief stimulus, record the spatiotemporal spike RESPONSE that the
perturbation CAUSED, and take the Lempel-Ziv complexity of that binarized response. PCI is high only when
the causal response is BOTH:
  * INTEGRATED  -- the perturbation spreads network-wide through the recurrent wiring (not a local echo), AND
  * DIFFERENTIATED -- that spread is complex / non-stereotyped (not a single saturated reverberation).
A response that is widespread but stereotyped (a synchronous saturated burst) is integrated-but-not-
differentiated and scores LOW; a response that is differentiated but confined (independent noisy units)
is differentiated-but-not-integrated and also scores LOW. Only a response that is both scores HIGH. That
is the whole point: it rewards differentiation AND integration together.

This measure is a SUBSTRATE / DYNAMICS property -- perturbational complexity -- REPORTED as an integration
proxy. It is NEVER asserted to equal consciousness, and it is NOT IIT Phi. True integrated information
(IIT Phi) is intractable: it is super-exponential in the number of units (a search over every bipartition
of every mechanism), uncomputable at any real scale. PCI is the deliberately TRACTABLE stand-in -- a
single perturb-and-compress number, not Phi. A high PCI means "a local perturbation caused a complex,
network-wide causal response", nothing more; it does NOT mean the system feels, experiences, or is aware.
There is NO consciousness threshold here, no "distance_to_consciousness", no claim that any value means
the system is or is not conscious. This is an honest, agnostic step of the project's consciousness program
-- it neither denies nor claims consciousness; it reports a dynamics property and names it exactly.

ONE PCI NUMBER, ON ITS OWN, IS NOT A REPORTABLE QUANTITY. ONLY CONTRASTS ARE. (Added 2026-08-05 after the
defect below.) There is no calibration set for perturbational complexity on an ARTIFICIAL substrate --
Casali's scale was normed on human sleep, anaesthesia and disorders of consciousness, and none of that
transfers to a twelve-unit LIF net. No calibration set means no cutoff, which means an absolute value
supports no verdict about ANOTHER system at all: it can only ever be telemetry. What this module is FOR is
the comparison -- which is why everything below it is built as one: the load-bearing assertions in the
self-check are ORDERINGS between regimes that share a substrate (integrated vs independent-noise vs
synchronous vs silent) plus a firing-rate control proving the gap is integration and not activity. Read
`pci()`'s return that way: it is one arm of an experiment, not a score.

That sentence said "EVERY assertion in the self-check is an ORDERING" until it was checked against the
self-check, which is the failure this whole paragraph is about, committed in the paragraph about it.
COUNTED 2026-08-05: `main()` carries 13 asserts, and SIX of them are absolute cutoffs on a single number
-- `mean_pci_integ > 0.25`, `pci_indep < 0.15`, `pci_silent < 1e-9`, `rate_silent == 0.0`,
`pci_sync < 0.05`, `rate_sync > 0.2`. They are legitimate and they are NOT a counter-example to the rule
above, for a reason worth stating rather than assuming: each is a cutoff on a FIXED synthetic regime
built by this module's own factories with fixed seeds, so it is a regression check on a constant this
file owns, not a verdict about a system it did not construct. An absolute number is barred as a claim
ABOUT SOMETHING ELSE. It is fine as a tripwire on your own fixture.

THE WORKED CASE, because it was live in this tree for the life of the feature. `cortex/brain.py` built a
fresh 10-unit instance of this class every 4th night and published its `pci()` as the brain's own `pci`
telemetry. Two things were wrong and the second is the general one: (1) the coupling it passed was
`1.0 + max(0.0, min(0.6, criticality))` where the converged regulator holds `criticality` just under 1.0
(MEASURED [0.98, 1.003] across seeds; nothing clamps it there, so this is empirical and not structural),
so the clamp pinned coupling at 1.6 and the "reading" was a pure function of the brain's seed -- MEASURED,
1 distinct value (0.4068) over 900 ticks and 37 nights at seed 0; (2) even had it varied, an absolute PCI off a
synthetic net says nothing about the system reporting it, because THIS CLASS HAS NO PORT through which an
external system's state can enter -- `_run()` builds its own neurons from its own weight matrix and
pre-draws its own noise. Deleted 2026-08-05. A future caller wanting to say something about the brain owes
the port first, and a contrast second; neither is optional.

WHY IT IS UN-RIGGABLE (the noise trap, and how the causal-footprint step defeats it)
====================================================================================
The classic failure mode of any "complexity" measure is the NOISE TRAP: a bank of INDEPENDENT random
units has MAXIMAL differentiation (each unit does its own thing, maximal joint entropy) yet ZERO
integration -- and a naive complexity/entropy score rewards it. A pure entropy coder falls straight in
(it rewards randomness). Two things keep PCI out of the trap:

  1. LEMPEL-ZIV, NOT AN ENTROPY CODER. LZ rewards repeated STRUCTURE, not marginal randomness; and we
     normalize by the Casali normalized-LZ, whose ceiling for a given firing density is the source
     entropy H(p) (reported alongside) -- we do NOT divide by H, which would inflate a sparse near-random
     local response back into the trap.
  2. THE CAUSAL FOOTPRINT. The response we compress is not the raw spike raster -- it is the CAUSAL
     footprint of the perturbation: spikes that DIFFER between the perturbed run and an otherwise-
     identical unperturbed run driven by the SAME background-noise realization. Spontaneous noise is
     shared by both runs and cancels EXACTLY in the difference; only what the perturbation causally
     changed survives. This is the deterministic-simulation analog of Casali's significance-vs-baseline
     thresholding. In an INDEPENDENT-noise bank the perturbation cannot spread (there are no wires), so
     its causal footprint stays confined to the stimulated units -> LOW PCI, even though the bank is
     more active and more differentiated than the integrated network. Integration, not mere complexity,
     is what PCI captures.

The self-check asserts the full ordering: PCI(integrated) > PCI(independent-noise) > PCI(silent) ~ 0, plus
a synchronous/saturated control (integrated-but-undifferentiated) that also scores ~0 -- and it does so
while the independent-noise bank fires AT LEAST AS MUCH as the integrated network, so a skeptic cannot
attribute the gap to activity level.

SUBSTRATE. A small recurrent LIF subnet (n=12; reuses cortex.lif_neuron.LIFNeuron, the canonical
spiking primitive). The DYNAMICAL REGIME is set by the recurrent COUPLING and background NOISE; PCI peaks
near the edge of chaos, the same slightly-subcritical band that plasticity.criticality_regulator holds a
spiking pool in (branching ratio ~1, maximal stimulus separability). This module measures that band's
perturbational signature; it does not regulate it.

Per-instance random.Random(seed); zlib.crc32 for child seeds (never
hash()); zlib for the gzip cross-check compressor; deterministic; ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.integration_measure   # (run_selfchecks.py)
"""

import math
import random
import zlib

from snn_data_lake.cortex.lif_neuron import LIFNeuron


def _child_seed(base, tag):
    """Deterministic child seed from a base seed + a string tag via crc32 (never hash())."""
    return zlib.crc32(f"{base}:{tag}".encode("ascii"))


# ======================================================================================================
# The compression primitives: Lempel-Ziv (LZ76) + a gzip cross-check. LZ rewards STRUCTURE, not
# marginal randomness (unlike an entropy coder) -- the anti-noise-trap choice for the compression step.
# ======================================================================================================
def lz76_complexity(s):
    """Lempel-Ziv 1976 complexity c(L): the number of distinct phrases in the exhaustive left-to-right
    LZ parse of the binary string `s` (Kaspar & Schuster 1987 form). This is the complexity term of
    Casali's PCI. A stereotyped/periodic string yields few phrases (low c); a structurally rich string
    yields many. Deterministic."""
    n = len(s)
    if n < 2:
        return 1
    i, k, l = 0, 1, 1
    k_max = 1
    c = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def source_entropy(s):
    """Shannon source entropy H(p) in bits of the binary string `s` (p = fraction of 1s). This is the
    CEILING of the normalized-LZ complexity for a random source at that density -- reported, not divided
    by (dividing would reward a sparse near-random response: the noise trap)."""
    n = len(s)
    if n == 0:
        return 0.0
    p = s.count("1") / n
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def normalized_lz(s):
    """Casali normalized Lempel-Ziv complexity of a binary string: c(L) * log2(L) / L. Its ceiling for a
    random source of the same density is the source entropy H(p). Returns 0.0 for an empty or all-zero
    string (no causal response -> no complexity)."""
    L = len(s)
    if L == 0 or s.count("1") == 0:
        return 0.0
    return lz76_complexity(s) * math.log2(L) / L


def gzip_complexity(s):
    """Cross-check compressor: the compressed length (bytes) of the packed bitstring under zlib (an
    LZ77+Huffman, i.e. gzip's DEFLATE). Reported to show PCI's ordering is robust to the exact LZ
    variant -- gzip, like LZ76, rewards STRUCTURE, not marginal randomness."""
    if not s:
        return 0
    packed = int(s, 2).to_bytes((len(s) + 7) // 8, "big") if "1" in s else b"\x00"
    return len(zlib.compress(packed, 9))


class PerturbationalComplexity:
    """A small recurrent LIF subnet plus a perturb-and-compress probe (PCI).

    n leaky integrate-and-fire units (reusing cortex.lif_neuron.LIFNeuron). A sparse random weight
    matrix (scaled by `coupling`) wires them recurrently; every unit also receives independent
    background-noise current (scaled by `noise`). `coupling`, `noise` and `weight_mode` select the
    DYNAMICAL REGIME (see the module factories `integrated_network`, `independent_noise`,
    `synchronous_network`, `silent_network`).

    `pci()` perturbs a subset of units with a brief current pulse, extracts the CAUSAL footprint (the
    spikes the perturbation changed relative to a same-noise unperturbed run), and returns the mean
    normalized-LZ complexity of that binarized footprint over a battery of perturbations. Deterministic
    given `seed`."""

    def __init__(
        self,
        n=12,
        coupling=1.1,
        noise=0.45,
        weight_mode="balanced",
        density=0.5,
        tau_m=10.0,
        refractory=1,
        seed=0,
    ):
        self.n = n
        self.coupling = coupling
        self.noise = noise
        self.tau_m = tau_m
        self.refractory = refractory
        self.seed = seed
        rng = random.Random(seed)
        # recurrent weight matrix W[i][j] = weight from j -> i.
        self.w = [[0.0] * n for _ in range(n)]
        if weight_mode == "excitatory":
            # fully-connected strong POSITIVE weights -> a kick drives a synchronous saturated burst
            # (integrated but NOT differentiated).
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.w[i][j] = abs(rng.gauss(1.2, 0.2))
        else:
            # sparse SIGNED (balanced E/I) weights -> near-critical spreading dynamics.
            k = max(1, int(density * (n - 1)))
            for i in range(n):
                pres = rng.sample([j for j in range(n) if j != i], k)
                for j in pres:
                    self.w[i][j] = rng.gauss(0.0, 1.0)

    # ---- the live spiking substrate -------------------------------------------------------------
    def _run(self, noise_seed, perturb, pulse_t, pulse_len, pulse_current, steps):
        """One run of the subnet. `perturb` (a set of unit indices) or None; when set, those units get
        `pulse_current` added during [pulse_t, pulse_t+pulse_len). The background noise is PRE-DRAWN from
        `noise_seed` so a perturbed and an unperturbed run share the EXACT same noise realization (their
        difference is then the pure causal footprint of the perturbation). Returns spikes[t] = [0/1]*n."""
        rng = random.Random(_child_seed(self.seed, f"noise:{noise_seed}"))
        noise_arr = [
            [rng.gauss(0.0, 1.0) * self.noise for _ in range(self.n)] for _ in range(steps)
        ]
        neurons = [
            LIFNeuron(tau_m=self.tau_m, v_threshold=1.0, refractory=self.refractory)
            for _ in range(self.n)
        ]
        pset = set(perturb) if perturb else set()
        spikes = []
        prev = [0] * self.n
        for t in range(steps):
            cur = list(noise_arr[t])
            if self.coupling:
                for i in range(self.n):
                    wi = self.w[i]
                    s = 0.0
                    for j in range(self.n):
                        if prev[j]:
                            s += wi[j]
                    cur[i] += self.coupling * s
            if pset and pulse_t <= t < pulse_t + pulse_len:
                for i in pset:
                    cur[i] += pulse_current
            sp = [neurons[i].step(cur[i]) for i in range(self.n)]
            spikes.append(sp)
            prev = sp
        return spikes

    def causal_response(
        self, noise_seed, perturb, pulse_t=20, pulse_len=3, pulse_current=5.0, steps=80
    ):
        """The binarized CAUSAL footprint of one perturbation: response[t'][i] = 1 iff the perturbed run
        and the same-noise unperturbed run differ at unit i, time t, over the POST-stimulus window
        t in [pulse_t+pulse_len, steps). Spontaneous noise cancels in the difference; only what the
        perturbation causally changed remains."""
        base = self._run(noise_seed, None, pulse_t, pulse_len, pulse_current, steps)
        pert = self._run(noise_seed, perturb, pulse_t, pulse_len, pulse_current, steps)
        w0 = pulse_t + pulse_len
        return [
            [1 if pert[t][i] != base[t][i] else 0 for i in range(self.n)] for t in range(w0, steps)
        ]

    @staticmethod
    def _serialize(response):
        """Serialize the response matrix time-major (each timestep's spatial pattern, concatenated over
        time) into one binary string for the LZ parse -- the spatiotemporal scan."""
        return "".join("".join(str(b) for b in row) for row in response)

    def pci(self, trials=8, n_perturb=3, pulse_t=20, pulse_len=3, pulse_current=5.0, steps=80):
        """Mean PCI over a battery of `trials` perturbations (each a distinct crc32-seeded subset of
        `n_perturb` units, each with its own background-noise realization). Returns (mean_pci, per_trial)
        where per_trial is a list of (pci, lz_complexity, n_ones, source_entropy, gzip_len)."""
        per = []
        for t in range(trials):
            perturb = random.Random(_child_seed(self.seed, f"perturb:{t}")).sample(
                range(self.n), n_perturb
            )
            resp = self.causal_response(t, perturb, pulse_t, pulse_len, pulse_current, steps)
            s = self._serialize(resp)
            per.append(
                (
                    normalized_lz(s),
                    lz76_complexity(s) if "1" in s else 0,
                    s.count("1"),
                    source_entropy(s),
                    gzip_complexity(s),
                )
            )
        mean = sum(p[0] for p in per) / len(per)
        return mean, per

    def gzip_pci(self, trials=8, n_perturb=3, pulse_t=20, pulse_len=3, pulse_current=5.0, steps=80):
        """Cross-check: mean gzip-compressed length (bytes) of the causal footprint over the same battery.
        Larger = a more complex, less compressible causal response. Same ordering as `pci` -> the result
        is robust to the compressor choice (LZ76 vs DEFLATE)."""
        total = 0.0
        for t in range(trials):
            perturb = random.Random(_child_seed(self.seed, f"perturb:{t}")).sample(
                range(self.n), n_perturb
            )
            resp = self.causal_response(t, perturb, pulse_t, pulse_len, pulse_current, steps)
            total += gzip_complexity(self._serialize(resp))
        return total / trials

    def mean_firing_rate(self, steps=80, noise_seed=0):
        """Mean fraction of the pool spiking per step in a free (unperturbed) run -- the ACTIVITY level,
        reported so a low PCI cannot be blamed on the network simply being quiet."""
        sp = self._run(noise_seed, None, 20, 3, 5.0, steps)
        return sum(sum(row) for row in sp) / (self.n * steps)


# ---- regime factories: same subnet, different DYNAMICS -----------------------------------------------
def integrated_network(seed=0):
    """Coupled, near-edge-of-chaos, active: a perturbation spreads network-wide and differentiatedly."""
    return PerturbationalComplexity(coupling=1.1, noise=0.45, weight_mode="balanced", seed=seed)


def independent_noise(seed=0):
    """The noise trap: uncoupled noisy units -- maximal differentiation, ZERO integration. Deliberately
    LOUDER (higher noise) than the integrated network so its low PCI cannot be blamed on low activity."""
    return PerturbationalComplexity(coupling=0.0, noise=1.2, weight_mode="balanced", seed=seed)


def synchronous_network(seed=0):
    """Integrated but NOT differentiated: strong all-excitatory coupling saturates into a stereotyped
    synchronous burst, so a perturbation leaves almost no differentiated causal trace -> PCI ~ 0."""
    return PerturbationalComplexity(coupling=0.4, noise=0.45, weight_mode="excitatory", seed=seed)


def silent_network(seed=0):
    """Frozen: no coupling, no noise -> sub-threshold, no dynamics. A perturbation evokes nothing -> PCI ~ 0."""
    return PerturbationalComplexity(coupling=0.0, noise=0.0, weight_mode="balanced", seed=seed)


def main():
    trials = 8
    integ = integrated_network(seed=0)
    indep = independent_noise(seed=0)
    sync = synchronous_network(seed=0)
    silent = silent_network(seed=0)

    pci_integ, per_integ = integ.pci(trials=trials)
    pci_indep, per_indep = indep.pci(trials=trials)
    pci_sync, _ = sync.pci(trials=trials)
    pci_silent, _ = silent.pci(trials=trials)

    rate_integ = integ.mean_firing_rate()
    rate_indep = indep.mean_firing_rate()
    rate_sync = sync.mean_firing_rate()
    rate_silent = silent.mean_firing_rate()

    gz_integ = integ.gzip_pci(trials=trials)
    gz_indep = indep.gzip_pci(trials=trials)
    gz_silent = silent.gzip_pci(trials=trials)

    # mean source-entropy ceiling + causal spread (mean # of units the perturbation reached) per regime.
    h_integ = sum(p[3] for p in per_integ) / trials
    h_indep = sum(p[3] for p in per_indep) / trials

    def spread(net):
        tot = 0.0
        for t in range(trials):
            perturb = random.Random(_child_seed(net.seed, f"perturb:{t}")).sample(range(net.n), 3)
            resp = net.causal_response(t, perturb)
            reached = sum(1 for i in range(net.n) if any(row[i] for row in resp))
            tot += reached
        return tot / trials

    spread_integ = spread(integ)
    spread_indep = spread(indep)

    # PCI and the firing rates vary far more ACROSS NETWORKS than across perturbation trials, so the
    # two clauses below that compare them are averaged over NETS independently-drawn subnets. Measured
    # over 30 seeds at trials=8: single-seed pci_integ ranges 0.153-0.558 and clears 0.25 on only 22/30
    # (it is 0.153 at seed 4, 0.184 at seed 5), and rate_indep >= rate_integ holds on 25/30. Raising
    # `trials` does NOT fix this -- at trials=32 seed 4 is still 0.215 -- because the draw that matters
    # is the subnet, not the perturbation. Over 8-seed windows the mean PCI never fell below 0.2937 and
    # the rate ordering never inverted, across all 23 windows in those 30 seeds. Costs ~0.5 s.
    NETS = 8
    _integs = [integrated_network(seed=s) for s in range(NETS)]
    _indeps = [independent_noise(seed=s) for s in range(NETS)]
    mean_pci_integ = sum(n.pci(trials=trials)[0] for n in _integs) / NETS
    mean_pci_indep = sum(n.pci(trials=trials)[0] for n in _indeps) / NETS
    mean_rate_integ = sum(n.mean_firing_rate() for n in _integs) / NETS
    mean_rate_indep = sum(n.mean_firing_rate() for n in _indeps) / NETS

    print("=" * 84)
    print(
        "PERTURBATIONAL COMPLEXITY INDEX (PCI) -- a tractable integration proxy over a spiking subnet"
    )
    print("=" * 84)
    print(
        "perturb a small recurrent LIF subnet (n=12); compress the CAUSAL footprint (perturbed run"
    )
    print(
        "minus same-noise baseline run) with Lempel-Ziv; PCI = c(L)*log2(L)/L. NOT consciousness, NOT"
    )
    print("IIT Phi -- a substrate/dynamics property, reported as an integration proxy.")
    print("")
    print(
        "  regime          PCI       gzip(bytes)   firing-rate   causal-spread(units)   H-ceiling"
    )
    print(
        f"  integrated    {pci_integ:6.4f}     {gz_integ:7.2f}       {rate_integ:6.3f}          "
        f"{spread_integ:5.1f} / {integ.n}            {h_integ:5.3f}"
    )
    print(
        f"  independent   {pci_indep:6.4f}     {gz_indep:7.2f}       {rate_indep:6.3f}          "
        f"{spread_indep:5.1f} / {indep.n}            {h_indep:5.3f}"
    )
    print(
        f"  synchronous   {pci_sync:6.4f}     {'':7}       {rate_sync:6.3f}          (saturated burst)"
    )
    print(
        f"  silent        {pci_silent:6.4f}     {gz_silent:7.2f}       {rate_silent:6.3f}          "
        f"(no dynamics)"
    )
    print("")
    print(
        f"  -> integrated PCI {pci_integ:.3f} is {pci_integ / max(pci_indep, 1e-9):.1f}x the independent-"
        f"noise bank {pci_indep:.3f}"
    )
    print(
        f"  -> and the independent bank fires MORE ({rate_indep:.3f} >= {rate_integ:.3f}) yet scores far "
        f"lower: the gap is INTEGRATION, not activity"
    )
    print(
        f"  -> the perturbation reached {spread_integ:.1f}/{integ.n} units when coupled vs "
        f"{spread_indep:.1f}/{indep.n} uncoupled (causal spread)"
    )

    # determinism: a fresh identical build reproduces the PCI exactly.
    pci_integ2, _ = integrated_network(seed=0).pci(trials=trials)
    deterministic = pci_integ2 == pci_integ

    # ============================ genuine self-checks (not rigged) ==============================
    # (1) THE NOISE TRAP: an integrated network scores strictly, substantially higher than a bank of
    #     independent noisy units -- which has MAXIMAL differentiation but ZERO integration.
    assert mean_pci_integ > 0.25, (
        f"an integrated near-critical net must reach a high PCI (mean {mean_pci_integ:.3f} over {NETS} "
        f"subnets; this one {pci_integ:.3f})"
    )
    assert pci_indep < 0.15, (
        f"independent noise must score LOW PCI (max differentiation, no integration): {pci_indep:.3f}"
    )
    # same single-subnet variance as the clause above -- this one was MASKED by it until that was fixed
    assert mean_pci_integ > 3.0 * mean_pci_indep, (
        f"PCI(integrated) must dwarf PCI(independent-noise): mean {mean_pci_integ:.3f} vs "
        f"{mean_pci_indep:.3f} over {NETS} subnets (this pair {pci_integ:.3f} vs {pci_indep:.3f})"
    )
    # (2) the gap is INTEGRATION, not activity: the independent bank fires at least as much, yet loses.
    assert mean_rate_indep >= mean_rate_integ, (
        f"the independent-noise control must be AT LEAST as active (mean rate {mean_rate_indep:.3f} vs "
        f"{mean_rate_integ:.3f} over {NETS} subnets; this pair {rate_indep:.3f} vs {rate_integ:.3f})"
    )
    assert pci_indep > pci_silent, (
        f"independent noise must still be genuinely ACTIVE (PCI {pci_indep:.3f} > silent {pci_silent:.3f})"
    )
    # (3) SILENT ~ 0: a frozen network evokes no causal response.
    assert pci_silent < 1e-9, f"a silent/frozen network must score ~0 PCI (got {pci_silent:.3g})"
    assert rate_silent == 0.0, f"the silent network must not spike (rate {rate_silent:.3f})"
    # (4) DIFFERENTIATION is required too: a synchronous/saturated (integrated-but-stereotyped) network
    #     scores ~0 despite firing hard -- PCI rewards differentiation AND integration TOGETHER.
    assert pci_sync < 0.05, (
        f"a synchronous saturated net must score ~0 PCI (undifferentiated): {pci_sync:.3f}"
    )
    assert rate_sync > 0.2, (
        f"the synchronous control must actually be saturated/active (rate {rate_sync:.3f})"
    )
    assert pci_integ > 3.0 * pci_sync, (
        f"PCI(integrated) must dwarf PCI(synchronous): {pci_integ:.3f} vs {pci_sync:.3f}"
    )
    # (5) causal spread confirms the mechanism: the perturbation reaches many units when coupled, few when not.
    assert spread_integ > 2.0 * spread_indep, (
        f"the perturbation must spread further when coupled ({spread_integ:.1f} vs {spread_indep:.1f} units)"
    )
    # (6) the gzip cross-check reproduces the SAME ordering -> robust to the compressor choice.
    assert gz_integ > gz_indep > gz_silent, (
        f"gzip cross-check must give the same ordering (integ {gz_integ:.1f} > indep {gz_indep:.1f} > silent {gz_silent:.1f})"
    )
    # (7) spiking + determinism.
    assert deterministic, "same seed must reproduce an identical PCI"

    print("")
    print(
        f"self-check OK: PCI integrated {pci_integ:.3f} > independent-noise {pci_indep:.3f} "
        f"({pci_integ / max(pci_indep, 1e-9):.1f}x) > silent {pci_silent:.3f}; synchronous(saturated) "
        f"{pci_sync:.3f} ~0 (differentiation required); independent fires MORE ({rate_indep:.3f}>="
        f"{rate_integ:.3f}) so the gap is INTEGRATION; causal spread {spread_integ:.1f}vs{spread_indep:.1f}"
        f"/{integ.n} units; gzip cross-check agrees; deterministic. NOT consciousness, NOT IIT Phi."
    )


if __name__ == "__main__":
    main()
