"""
Potjans-Diesmann 6-layer / 8-population cortical MICROCIRCUIT -- a SPIKING dynamics model.

This is the canonical cortical column model of Potjans & Diesmann 2014 ("The Cell-Type Specific
Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model",
Cerebral Cortex 24:785-806) -- the gold-standard, experimentally-grounded connectivity map of ~1 mm^2
of early sensory cortex. It has EIGHT leaky-integrate-and-fire (LIF) populations: an excitatory (e)
and an inhibitory (i) pool in each of the four cortical layers L2/3, L4, L5, L6 (layer 1 has no cell
bodies). The layer-specific, cell-type-specific connection probabilities are the paper's Table 5.

It is a DYNAMICS model, NOT a classifier: it reproduces the layer-specific FIRING STATISTICS of
cortex (per-layer rates, the asynchronous-irregular ground state), not any label/accuracy. There are
no targets, no training, no accuracy claim -- the honest acceptance test is that the KNOWN biological
signatures emerge from the connectivity + LIF physics alone (see main()).

Canonical intracortical flow the connectivity encodes: thalamus -> L4 (main input layer) -> L2/3
(superficial associational) -> L5 (deep output / subcortical projection, the highest-firing
excitatory layer) -> L6 (feedback to thalamus). Every layer's e-pool is reined in by a co-located
i-pool; the whole circuit is inhibition-dominated (relative inhibitory weight g = 4), which is what
holds it in the low-rate asynchronous-irregular (AI) state instead of runaway synchrony.

Neuron / synapse model (Potjans-Diesmann values): LIF, tau_m = 10 ms, C_m = 250 pF, refractory
2 ms, resting = reset = -65 mV, threshold = -50 mV (15 mV above rest), dt = 0.1 ms. Synapses are
CURRENT-based DELTA synapses expressed directly in membrane-voltage jumps: an excitatory spike bumps
the post-synaptic membrane by an EPSP of ~0.15 mV, an inhibitory spike by g x that (g = 4).
Transmission delay ~1.5 ms. External input is the paper's layer-specific bank of 8 Hz Poisson
synapses (k_ext), here injected via the diffusion (mean + variance) approximation.

SCALING APPROXIMATION (honest): the full model is 77169 neurons -- intractable in pure Python -- so
each population is scaled down by `scale` (default 0.008 -> ~620 neurons total). Downscaling would
distort the input statistics, so we apply the standard Potjans-Diesmann VARIANCE-PRESERVING weight
correction w -> w / sqrt(scale) (which keeps the recurrent input FLUCTUATIONS -- and hence the
irregular AI dynamics -- at full-scale magnitude) together with a tuned external drive that holds the
mean input near threshold. This is a documented SIMPLIFICATION of the paper's full mean-and-variance
matching (we tune the external drive rather than derive an analytic per-population DC current from the
predicted rates), chosen so the qualitative layer signatures below survive the ~120x downscaling. The
per-population rates are therefore approximate; the QUALITATIVE signatures (all layers low-rate,
asynchronous-irregular, superficial-low / L5-elevated) are what this scaled twin reproduces.

THE INPUT PORT (added 2026-08-04, closing P0's clearest open avenue). Until now this circuit took no
signal from anywhere: `__init__` configured it and `.simulate(t_ms)` ran it, so no faculty could drive
the column and nothing could read a response to anything -- which is exactly why
`substrate/dormancy_gate.py` exempts it permanently with the reason *"it has no input port"*.
`simulate(..., thalamic_hz=)` is that port: the paper's own thalamo-cortical projection (N_TH = 902
relay cells onto L4e/L4i/L6e/L6i with the Table-5 probabilities; L2/3 and L5 receive NO direct
thalamic afferent), folded into the same diffusion approximation the background bank uses. Driving it
raises the DIRECT targets and leaves L2/3 to see the stimulus only through the intracortical
thalamus -> L4 -> L2/3 route, which is a layer-specific claim the circuit can fail -- and the
self-check runs the input that makes it fail. An undriven port contributes exactly 0.0 to both the
mean and the variance, so the pre-port trajectory is unchanged.

Per-instance random.Random(seed);
ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.cortical_microcircuit
  (back-compat alias also resolves:  python -m snn_data_lake.cortical_microcircuit)
"""

import math
import random

# --- the eight populations, in canonical order --------------------------------------------------
POP_NAMES = ["L2/3e", "L2/3i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
EXC_POPS = (0, 2, 4, 6)  # indices of the excitatory pools (the rest are inhibitory)

# Full-scale population sizes (Potjans-Diesmann 2014, Table 5). Total = 77169.
N_FULL = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]

# Connection PROBABILITY matrix C[target_row][source_col] in the population order above (Table 5).
CONN_PROB = [
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
    [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
    [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
    [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443],
]

# External Poisson in-degrees per population (bank of 8 Hz background synapses), Table 5.
K_EXT = [1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]

# --- THE THALAMIC INPUT PORT (Potjans-Diesmann 2014, thalamic column of Table 5) ------------------
# WHY THIS EXISTS. `docs/POLICIES.md` P0 lists "cortical column construction" as its clearest OPEN
# avenue, and `substrate/dormancy_gate.py` exempts this module permanently with the reason *it has no
# input port*: `__init__(scale, seed, g, bg_rate, v_op, noise_gain)` and `.simulate(t_ms)` took no
# signal from anywhere, so no faculty could drive the column and nothing could read a response to
# anything. A column that cannot be driven is not a faculty; it is a screensaver. This is that port.
#
# It is the paper's OWN thalamic projection, not an invented one: a population of N_TH = 902
# thalamo-cortical relay cells projecting with layer-specific probability onto L4 and L6 (L2/3 and
# L5 receive NO direct thalamic input -- they see the stimulus only through the intracortical
# thalamus -> L4 -> L2/3 -> L5 flow the connectivity above already encodes). That asymmetry is what
# makes the port measurable rather than decorative: driving it must lift L4 FIRST and L2/3 only
# derivatively, which is a claim the circuit can fail.
#
# The drive enters through the SAME diffusion approximation as the background bank (mean + variance
# folded into the per-neuron external kick), so a rate of 0 Hz adds exactly 0.0 to both terms and
# draws exactly the same number of random numbers -- the undriven trajectory is bit-identical to the
# pre-port module, which the self-check asserts rather than assumes.
N_TH = 902  # thalamo-cortical relay population size (full scale)
CONN_PROB_TH = [0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196]  # -> L4e/L4i/L6e/L6i only
TH_TARGETS = tuple(p for p, c in enumerate(CONN_PROB_TH) if c > 0.0)

# --- LIF + synapse parameters (Potjans-Diesmann values), voltages measured RELATIVE to rest -----
TAU_M = 10.0  # membrane time constant (ms)
DT = 0.1  # integration step (ms)
V_TH = 15.0  # spike threshold (mV above the -65 mV rest / reset)
V_RESET = 0.0  # reset = rest (relative)
T_REF_MS = 2.0  # absolute refractory period (ms)
DELAY_MS = 1.5  # synaptic transmission delay (ms)
PSP = 0.15  # excitatory PSP amplitude (mV) -- the delta-synapse voltage jump
G_INH = 4.0  # relative inhibitory weight (inhibition-dominated): IPSP = -G_INH x PSP
BG_RATE = 8.0  # external background rate per synapse (Hz)
V_OP = 6.0  # sub-threshold mean operating point (mV) set by the external drive
NOISE_GAIN = 4.0  # multiplier on the external diffusion std (sets the fluctuation size)


class CorticalMicrocircuit:
    """The Potjans-Diesmann 8-population cortical column as a scaled spiking LIF network.

    Build a downscaled instance (`scale` in (0, 1]; full model is scale=1.0 -> 77169 neurons, far
    too large for stdlib), then call `.simulate(t_ms)`. `.firing_rates()` returns the per-population
    firing rate (Hz) and `.cv_isi()` the per-population coefficient of variation of the inter-spike
    intervals (CV ~ 1 == Poisson-irregular). Connectivity is drawn once from a per-instance
    random.Random(seed); the drawn network + drive is a deterministic function of (scale, seed)."""

    def __init__(
        self, scale=0.008, seed=0, g=G_INH, bg_rate=BG_RATE, v_op=V_OP, noise_gain=NOISE_GAIN
    ):
        if not (0.0 < scale <= 1.0):
            raise ValueError("scale must be in (0, 1]")
        self.scale = scale
        self.seed = seed
        self.g = g
        self.bg_rate = bg_rate
        self.v_op = v_op
        self.noise_gain = noise_gain
        self.decay = math.exp(-DT / TAU_M)  # membrane leak factor per step (used in simulate too)
        self.rng = random.Random(seed)  # per-instance RNG (connectivity build)

        # scaled population sizes (>=1 each so no layer vanishes) and the global index ranges
        self.n_pop = [max(1, int(round(N_FULL[p] * scale))) for p in range(8)]
        self.n_total = sum(self.n_pop)
        self.pop_start = []
        acc = 0
        for p in range(8):
            self.pop_start.append(acc)
            acc += self.n_pop[p]
        self.pop_of = [0] * self.n_total  # global index -> population index
        for p in range(8):
            for k in range(self.n_pop[p]):
                self.pop_of[self.pop_start[p] + k] = p

        # variance-preserving weight correction: w -> w / sqrt(scale)  (Potjans-Diesmann scaling)
        corr = 1.0 / math.sqrt(scale)
        self.w_exc = PSP * corr  # excitatory delta-synapse voltage jump (mV)
        self.w_inh = -g * PSP * corr  # inhibitory delta-synapse voltage jump (mV)

        self._build_connectivity()

        # External drive as an operating-point current plus per-layer diffusion noise. Naively
        # downscaling the network shrinks the recurrent MEAN (the net-inhibitory drive that balances
        # cortex) to a factor sqrt(scale) of full while the variance correction w -> w/sqrt(scale)
        # keeps the recurrent FLUCTUATIONS full-size. Rather than an analytic per-population DC term
        # (which, from the paper's rates, lands the mean near REST -- ~15 mV below threshold -- so the
        # superficial layers fall silent), we set the external MEAN so each neuron's leak-steady
        # membrane sits at a common sub-threshold operating point v_op (a few mV below V_th): the
        # network then lives in the fluctuation-driven regime where the large, net-inhibitory
        # recurrent fluctuations produce low-rate, irregular firing. The external VARIANCE is still
        # the paper's per-layer 8 Hz x k_ext Poisson bank (diffusion approximation), so the layers
        # keep distinct input statistics; the layer-specific RATE ordering then emerges from the
        # recurrent connectivity. This is a documented simplification of the paper's full
        # mean-and-variance matching, tuned so the qualitative signatures survive the downscaling.
        dt_s = DT / 1000.0
        mu_op = v_op * (
            1.0 - self.decay
        )  # external mean / step that holds the leak-steady v = v_op
        self.mu_ext = [0.0] * self.n_total  # per-neuron external mean voltage kick / step (mV)
        self.sigma_ext = [0.0] * self.n_total  # per-neuron external std of the kick / step (mV)
        for i in range(self.n_total):
            p = self.pop_of[i]
            rate = K_EXT[p] * bg_rate * dt_s  # expected external spikes this step (per-layer)
            self.mu_ext[i] = mu_op
            self.sigma_ext[i] = noise_gain * math.sqrt(rate) * PSP

        # Thalamic in-degree per population: CONN_PROB_TH[p] * N_TH, NOT downscaled.
        #
        # This is deliberate and it is the paper's own treatment of an EXTERNAL bank: K_EXT above is
        # likewise used at full scale for every `scale`, because downscaling an external in-degree
        # would shrink the very input statistics the variance correction exists to preserve.
        # MEASURED while building this: scaling it (round(p*N_TH*scale)) collapses the whole
        # projection at the default width -- k_th became [0,0,1,0,0,0,0,0], so three of the four
        # thalamic target populations had NO afferent at all and the fourth had one, and driving the
        # port moved L4e by -0.008 relative at 15 Hz (noise, not signal). Kept as a float in-degree
        # for the same reason: rounding 0.709 to 1 (or to 0) is a quantisation the diffusion
        # approximation does not need and cannot afford at these widths.
        self.k_th = [CONN_PROB_TH[p] * N_TH for p in range(8)]

        # results (populated by simulate())
        self.spike_times = [
            [] for _ in range(self.n_total)
        ]  # per-neuron spike times (ms) in window
        self.meas_ms = 0.0
        self.thalamic_hz = 0.0  # rate the last simulate() drove the port at (0 = undriven)

    def _build_connectivity(self):
        """Draw the recurrent synapses with the Potjans-Diesmann fixed-in-degree rule. For each
        target neuron in population i we draw K_ij = round(C[i][j] * N_j_scaled) presynaptic partners
        uniformly (with replacement, so multapses/autapses can occur, exactly as in the paper) from
        each source population j. Stored as an out-adjacency list out_adj[source] = [(target, w), ...]
        for event-driven propagation. Weight sign is set by whether the SOURCE pool is exc or inh."""
        rng = self.rng
        self.out_adj = [[] for _ in range(self.n_total)]
        self.in_degree = [
            [0] * 8 for _ in range(8)
        ]  # K_ij: recurrent in-degree matrix (inspectable)
        n_syn = 0
        for i in range(8):  # target population
            tgt_lo = self.pop_start[i]
            for j in range(8):  # source population
                c = CONN_PROB[i][j]
                if c <= 0.0:
                    continue
                k = int(round(c * self.n_pop[j]))  # in-degree from j into each i-neuron
                self.in_degree[i][j] = k
                if k == 0:
                    continue
                w = self.w_exc if j in EXC_POPS else self.w_inh
                src_lo = self.pop_start[j]
                src_n = self.n_pop[j]
                for t in range(self.n_pop[i]):
                    tgt = tgt_lo + t
                    for _ in range(k):
                        src = src_lo + rng.randrange(src_n)
                        self.out_adj[src].append((tgt, w))
                        n_syn += 1
        self.n_synapses = n_syn

    def thalamic_drive(self, thalamic_hz):
        """THE INPUT PORT, as (mean, std) voltage kick per step per neuron -- the signal a caller
        injects, in the same diffusion form as the background bank.

        `thalamic_hz` is the firing rate of each thalamo-cortical relay cell (the paper's stimulus
        transient uses 15 Hz). Returns two length-n_total lists to be ADDED to the background mean
        and (in quadrature) to the background variance. At 0 Hz both are exactly 0.0, which is what
        makes an undriven run bit-identical to the pre-port module.

        Only L4 and L6 have a nonzero thalamic in-degree (CONN_PROB_TH), so a caller cannot inject
        straight into L2/3 or L5: the stimulus has to travel the intracortical route. Refuses a
        negative rate rather than silently flipping the sign of a mean current."""
        hz = float(thalamic_hz)
        if hz < 0.0:
            raise ValueError("thalamic_hz must be >= 0 (a firing rate, not a current)")
        dt_s = DT / 1000.0
        mu = [0.0] * self.n_total
        var = [0.0] * self.n_total
        if hz == 0.0:
            return mu, var
        mu_op = self.v_op * (1.0 - self.decay)  # the background bank's own per-step mean kick
        for i in range(self.n_total):
            p = self.pop_of[i]
            th_rate = self.k_th[p] * hz * dt_s  # expected thalamic spikes / step into this neuron
            bg_rate = K_EXT[p] * self.bg_rate * dt_s  # ... and the background bank's, same units
            # MEAN. The background mean is TUNED (it holds the leak-steady membrane at v_op) rather
            # than derived from PSP, so the only honest way to add a second bank to it is in the
            # same currency: the thalamic bank contributes mu_op scaled by its share of the
            # background bank's spike rate. Deriving `th_rate * PSP` instead would mix a tuned mean
            # with an untuned one and the ratio between them would be an artifact of v_op.
            mu[i] = mu_op * (th_rate / bg_rate) if bg_rate > 0.0 else 0.0
            # VARIANCE. Independent Poisson banks -> variances add; same diffusion form the
            # background uses (sigma = noise_gain * sqrt(rate) * PSP, so var = rate*(gain*PSP)^2).
            var[i] = th_rate * (self.noise_gain * PSP) ** 2
        return mu, var

    def simulate(self, t_ms=400.0, settle_ms=100.0, thalamic_hz=0.0):
        """Integrate the network for `settle_ms` (discarded transient) + `t_ms` (measurement window).
        Spikes in the measurement window are recorded per neuron in self.spike_times (ms, window-
        relative). Deterministic: the noise RNG is re-seeded from `seed` here, so repeated calls with
        the same instance reproduce identically regardless of prior calls. Returns self.

        `thalamic_hz` DRIVES THE INPUT PORT (see thalamic_drive): the per-relay-cell rate of the
        thalamo-cortical afferent population. 0.0 (the default) is the undriven column and is
        bit-identical to the port-less module -- the drive is folded into the existing per-neuron
        external mean/std, so the RNG draw order does not change."""
        th_mu, th_var = self.thalamic_drive(thalamic_hz)
        self.thalamic_hz = float(thalamic_hz)
        decay = math.exp(-DT / TAU_M)
        tref_steps = int(round(T_REF_MS / DT))
        delay_steps = max(1, int(round(DELAY_MS / DT)))
        ring = delay_steps + 1
        settle_steps = int(round(settle_ms / DT))
        meas_steps = int(round(t_ms / DT))
        total_steps = settle_steps + meas_steps
        self.meas_ms = meas_steps * DT

        n = self.n_total
        v = [0.0] * n
        ref = [0] * n
        # Background + thalamic, combined in the diffusion approximation: means ADD, variances add
        # (independent Poisson banks). At thalamic_hz == 0 every th_* term is exactly 0.0, and the
        # combination below is exact there -- `a + 0.0 == a` for every finite a, and sigma passes
        # straight through when its added variance is exactly 0.0. So the undriven trajectory is
        # bit-identical AND still flows through the port, which is what makes the inertness assert
        # in main() falsifiable.
        #
        # This was FIRST written as `mu = self.mu_ext if thalamic_hz == 0.0 else [...]`, which made
        # that assert TRUE BY CONSTRUCTION: with the port short-circuited out of the hz==0 path, no
        # defect in thalamic_drive could reach it. Caught by constructing the red (a port leaking a
        # nonzero mean at 0 Hz) and watching the check pass anyway. Do not reintroduce the branch --
        # `sqrt(s*s + 0.0)` is NOT guaranteed to round back to s, which is the only reason the
        # variance line tests `vv == 0.0` instead of always calling sqrt.
        mu = [a + b for a, b in zip(self.mu_ext, th_mu)]
        sigma = [s if vv == 0.0 else math.sqrt(s * s + vv) for s, vv in zip(self.sigma_ext, th_var)]
        out_adj = self.out_adj
        theta = V_TH
        reset = V_RESET
        spikes = [[] for _ in range(n)]
        pending = [[0.0] * n for _ in range(ring)]  # ring buffer of delayed synaptic input (mV)

        # dedicated, reproducible noise RNG (independent of the connectivity draw sequence)
        rng = random.Random((self.seed * 1000003 + 101) & 0x7FFFFFFF)
        gauss = rng.gauss

        # THIS LOOP IS NOT PARALLELISED, AND THAT IS A MEASURED RESULT, NOT AN OVERSIGHT.
        #
        # It is the largest population update in the tree (618 neurons at the default scale, 77169
        # at scale 1.0) and therefore the obvious first target for the free-threaded build, so the
        # numbers are recorded here to stop the next reader paying for the experiment twice.
        # A BYTE-IDENTICAL parallel form exists and was built and run: three phases per step --
        # (A) draw gauss serially for exactly the non-refractory neurons in index order, so the
        # draw STREAM is unchanged; (B) integrate + threshold across threads over disjoint index
        # ranges; (C) scatter the fired synapses serially in index order, so the float-addition
        # order into `fut` is unchanged. It reproduced the serial spike trains exactly.
        #
        # It loses anyway, structurally. MEASURED (3.14.6, Py_GIL_DISABLED=1, 28 logical CPUs):
        #   phase split      scale 0.008 (n=618)   scale 0.1 (n=7717)
        #   rng.gauss             73.6%                 69.0%       <- MUST stay serial for identity
        #   synaptic scatter       0.9%                  5.2%       <- MUST stay serial for identity
        #   parallelisable        25.5%                 25.8%
        #   Amdahl ceiling        1.34x                 1.35x       <- at INFINITE threads
        # and end to end, against serial simulate(): 0.39x / 0.18x / 0.09x at 4 / 8 / 14 threads
        # (scale 0.008), 0.39x / 0.13x / 0.07x (scale 0.1). Widening the column does not rescue it:
        # the ceiling is a property of the ALGORITHM -- one shared RNG stream consumed in neuron
        # order -- not of the population size, so it is 1.34x at every scale including full.
        #
        # Per-partition RNG streams would lift that ceiling and are the obvious next idea. They
        # would also change every number this module produces, and its layer-specificity assert was
        # calibrated over six columns of THIS stream (see PORT_SEEDS), so that is a re-baselining
        # exercise with a fitted criterion attached, not a speedup. Not done, and not smuggled in.
        # The population update that DID pay is the independent-cell one in cortex/neuron_zoo.py.
        for step in range(total_steps):
            slot = step % ring
            buf = pending[slot]
            fut = pending[(step + delay_steps) % ring]
            recording = step >= settle_steps
            for i in range(n):
                r = ref[i]
                if r > 0:
                    ref[i] = r - 1  # in refractory: membrane clamped at reset
                    continue
                s = sigma[i]
                vi = v[i] * decay + buf[i] + mu[i] + (gauss(0.0, s) if s > 0.0 else 0.0)
                if vi >= theta:
                    v[i] = reset
                    ref[i] = tref_steps
                    if recording:
                        spikes[i].append((step - settle_steps) * DT)
                    for tgt, w in out_adj[i]:
                        fut[tgt] += w
                else:
                    v[i] = vi
            pending[slot] = [0.0] * n  # clear the consumed slot for reuse

        self.spike_times = spikes
        return self

    def _pop_range(self, p):
        lo = self.pop_start[p]
        return range(lo, lo + self.n_pop[p])

    def firing_rates(self):
        """Per-population mean firing rate (Hz) over the measurement window."""
        if self.meas_ms <= 0.0:
            raise RuntimeError("call simulate() before firing_rates()")
        secs = self.meas_ms / 1000.0
        rates = []
        for p in range(8):
            total = sum(len(self.spike_times[i]) for i in self._pop_range(p))
            rates.append(total / (self.n_pop[p] * secs))
        return rates

    def cv_isi(self):
        """Per-population coefficient of variation of the inter-spike intervals. ISIs are POOLED
        across all neurons in a population (each neuron contributes the intervals within its own
        spike train), a standard estimator at these low rates / short windows. CV = std/mean of the
        pooled ISIs; ~1.0 for a Poisson (asynchronous-irregular) train, ~0 for clock-like regular
        firing. Returns a list of 8 CV values (float('nan') if a population had too few intervals)."""
        cvs = []
        for p in range(8):
            isis = []
            for i in self._pop_range(p):
                st = self.spike_times[i]
                for a in range(1, len(st)):
                    isis.append(st[a] - st[a - 1])
            if len(isis) >= 2:
                m = sum(isis) / len(isis)
                if m > 0.0:
                    var = sum((x - m) ** 2 for x in isis) / len(isis)
                    cvs.append(math.sqrt(var) / m)
                else:
                    cvs.append(float("nan"))
            else:
                cvs.append(float("nan"))
        return cvs


def _mean_finite(xs):
    vals = [x for x in xs if x == x]  # drop NaNs (NaN != NaN)
    return sum(vals) / len(vals) if vals else float("nan")


# --- the INPUT-PORT experiment ---------------------------------------------------------------------
# SEED-AVERAGED on purpose. Both the connectivity draw and the noise realisation move the per-layer
# rates by more than the stimulus does at 15 Hz at this width: MEASURED at seed 0 alone, driving the
# port at 15 Hz moved L2/3i by -12.8% one way and L5e by +2.9% the other (both k_th == 0), so a
# single-seed reading of any non-target layer is a coin flip.
#
# SIX COLUMNS, NOT THREE -- RAISED 2026-08-05 ON A MEASURED FALSE-RED RATE, and the way three was
# arrived at is the lesson. Three was chosen, the layer-specificity assert was seen to pass at
# PORT_SEEDS = (0, 1, 2), and that was taken as "the claim holds". Swept properly instead -- 12
# columns (seeds 0..11) simulated at 0/15/30 Hz, scored over all 220 three-seed triples:
#
#   assert                                    3-seed triples passing   min margin @15 Hz
#   L4e > L2/3e + 0.05 at BOTH rates              201 / 220 (91.4%)         -0.036
#   l4e_rel[1] > 0.10                             220 / 220                  --
#   l4e_rel[1] > l4e_rel[0] > 0 (monotone)        220 / 220                  --
#
# So the criterion this comment claimed "then holds at both rates" fails on roughly one triple in
# twelve WITH NO DEFECT PRESENT -- it was fitted to the seeds it was first run on, which is the
# instrument-tuned-on-its-own-sample failure this project keeps finding. Over the 200 SIX-seed
# windows from the same sweep it passes 200/200 with a minimum 15 Hz margin of +0.080, i.e. 0.030
# clear of the 0.05 threshold rather than 0.086 under it. The threshold is UNCHANGED; only the
# sample count moved, so this is a strictly stronger claim, not a relaxed one.
#
# The 30 Hz half was already robust (min margin +0.091 over all 220 triples) -- the fragility is
# entirely at 15 Hz, where the L4e signal (~+0.15) is the same size as the chaotic divergence a
# non-target population shows (L5i, N=9 at this scale, moved +20.3% at PORT_SEEDS=(0,1,2)).
# Cost: 18 short simulations, MEASURED ~10 s of a ~16 s self-check. Do not trim it back to 3.
PORT_SEEDS = (0, 1, 2, 3, 4, 5)
PORT_RATES = (0.0, 15.0, 30.0)  # 15 Hz is the paper's own thalamic transient rate
PORT_T_MS, PORT_SETTLE_MS = 200.0, 40.0


def _port_rates(hz, scale=0.008):
    """Mean per-population firing rate over PORT_SEEDS independently-drawn columns, driven at `hz`."""
    per = []
    for s in PORT_SEEDS:
        net = CorticalMicrocircuit(scale=scale, seed=s)
        net.simulate(t_ms=PORT_T_MS, settle_ms=PORT_SETTLE_MS, thalamic_hz=hz)
        per.append(net.firing_rates())
    return [sum(r[p] for r in per) / len(per) for p in range(8)]


def main():
    scale = 0.008
    net = CorticalMicrocircuit(scale=scale, seed=0)
    net.simulate(t_ms=400.0, settle_ms=100.0)
    rates = net.firing_rates()
    cvs = net.cv_isi()
    mean_cv = _mean_finite(cvs)

    print("=" * 72)
    print("POTJANS-DIESMANN CORTICAL MICROCIRCUIT -- 8-population spiking LIF column")
    print("=" * 72)
    print(
        f"scale {scale:.4f}  ->  {net.n_total} neurons  |  {net.n_synapses} recurrent synapses  "
        f"(full model = {sum(N_FULL)} neurons)"
    )
    print(
        f"variance-preserving weights: EPSP {net.w_exc:.3f} mV  IPSP {net.w_inh:.3f} mV  "
        f"(g = {net.g:.0f}, base PSP {PSP} mV / sqrt(scale))"
    )
    print(f"measurement window: {net.meas_ms:.0f} ms after a 100 ms settle,  dt = {DT} ms")
    print("-" * 72)
    print("pop      N     rate(Hz)    CV_ISI     kind")
    for p in range(8):
        cv = cvs[p]
        cv_s = f"{cv:6.3f}" if cv == cv else "   n/a"
        kind = "exc" if p in EXC_POPS else "inh"
        print(f"{POP_NAMES[p]:6s} {net.n_pop[p]:5d}   {rates[p]:8.2f}    {cv_s}     {kind}")
    print("-" * 72)
    print(f"mean CV_ISI across populations: {mean_cv:.3f}   (Poisson-irregular ~ 1.0)")
    r23e, r5e = rates[0], rates[4]
    print(
        f"layer signature: rate(L5e) {r5e:.2f} Hz  >  rate(L2/3e) {r23e:.2f} Hz  "
        f"(superficial-low, L5 elevated)"
    )
    print("=" * 72)

    # ---- self-checks: assert the KNOWN biological SIGNATURES (dynamics model, not accuracy) ----
    # (1) every population is active at a plausible LOW rate -- none silent, none runaway/saturated.
    for p in range(8):
        assert rates[p] > 0.0, f"{POP_NAMES[p]} is silent (0 Hz) -- lost its drive"
        assert rates[p] < 30.0, f"{POP_NAMES[p]} rate {rates[p]:.1f} Hz is runaway/saturated"
    # (2) ASYNCHRONOUS-IRREGULAR ground state: spiking is irregular, not clock-like or synchronized.
    assert mean_cv == mean_cv, "CV_ISI undefined -- too few spikes to estimate irregularity"
    assert mean_cv > 0.7, (
        f"mean CV_ISI {mean_cv:.2f} too regular -- not the asynchronous-irregular state"
    )
    # (3) canonical LAYER RATE SIGNATURE: superficial L2/3e is low, L5e is the elevated excitatory
    #     layer (the robust qualitative ordering that survives the downscaling).
    assert rates[4] > rates[0], (
        f"layer signature broken: L5e {rates[4]:.2f} Hz must exceed L2/3e {rates[0]:.2f} Hz"
    )
    # structural sanity: genuinely spiking, deterministic.
    net2 = CorticalMicrocircuit(scale=scale, seed=0)
    net2.simulate(t_ms=400.0, settle_ms=100.0)
    assert net2.firing_rates() == rates, "same (scale, seed) must reproduce identical rates"

    # ================= (4) THE INPUT PORT: the column can be DRIVEN, and it answers ==================
    # P0's clearest open avenue (docs/POLICIES.md): this circuit was EXEMPT-permanent in
    # substrate/dormancy_gate.py because "it has no input port". It has one now, and the claim is not
    # that the port exists -- it is that driving it changes the column in the LAYER-SPECIFIC way the
    # thalamic projection predicts.
    port = {hz: _port_rates(hz) for hz in PORT_RATES}
    undriven = port[0.0]

    def _rel(hz, p):
        return (port[hz][p] - undriven[p]) / undriven[p] if undriven[p] else float("inf")

    l4e_rel = [_rel(hz, 2) for hz in PORT_RATES[1:]]
    l23e_rel = [_rel(hz, 0) for hz in PORT_RATES[1:]]

    # (4a) an UNDRIVEN port is INERT -- it contributes exactly 0.0 to both the mean and the variance,
    #      so the default trajectory is unchanged by the port's existence.
    #
    #      Stated on thalamic_drive's OWN output rather than by diffing two simulate() runs. The
    #      obvious form -- run with thalamic_hz=0.0 and compare against the default-arg run -- is
    #      TRUE BY CONSTRUCTION: both calls take the same code path with the same value, so a port
    #      leaking a nonzero mean at 0 Hz leaks it into BOTH and they still agree. MEASURED: with a
    #      +1e-3 mV/step leak injected, that diff stayed True while the layer-specificity claim went
    #      red. The zeros below are the property that actually makes the combination in simulate()
    #      the identity (a + 0.0 == a exactly; sigma passes through untouched at vv == 0.0).
    z_mu, z_var = CorticalMicrocircuit(scale=scale, seed=0).thalamic_drive(0.0)
    port_zero_identical = all(m == 0.0 for m in z_mu) and all(v == 0.0 for v in z_var)

    # (4c) the port refuses a physically meaningless drive rather than flipping a mean current.
    try:
        CorticalMicrocircuit(scale=scale, seed=0).thalamic_drive(-1.0)
        refuses_negative = False
    except ValueError:
        refuses_negative = True

    print("-" * 72)
    print(
        f"INPUT PORT (thalamic, k_th = {[round(k, 1) for k in net.k_th]} onto "
        f"{'/'.join(POP_NAMES[p] for p in TH_TARGETS)}), {len(PORT_SEEDS)} columns x {PORT_T_MS:.0f} ms:"
    )
    print("  pop        0 Hz      15 Hz      30 Hz      rel@15    rel@30")
    for p in range(8):
        print(
            f"  {POP_NAMES[p]:6s} {undriven[p]:8.3f} {port[15.0][p]:10.3f} {port[30.0][p]:10.3f}"
            f"   {_rel(15.0, p):+8.3f} {_rel(30.0, p):+9.3f}"
        )
    print(
        f"  -> L4e (direct thalamic target) {l4e_rel[0]:+.3f} -> {l4e_rel[1]:+.3f}; "
        f"L2/3e (NO thalamic afferent) {l23e_rel[0]:+.3f} -> {l23e_rel[1]:+.3f}"
    )
    print(f"  -> undriven port contributes exactly 0 mean and 0 variance: {port_zero_identical}")
    print("=" * 72)

    assert port_zero_identical, (
        "thalamic_drive(0.0) must contribute EXACTLY 0.0 to both the mean and the variance -- an "
        "undriven port must be inert, so the port-less trajectory is unchanged by its existence"
    )
    assert l4e_rel[1] > 0.10, (
        f"driving the thalamic port must raise L4e, its direct target (rel {l4e_rel[1]:+.3f} at "
        f"{PORT_RATES[-1]:.0f} Hz over {len(PORT_SEEDS)} columns)"
    )
    assert l4e_rel[1] > l4e_rel[0] > 0.0, (
        f"L4e's response must GROW with the thalamic rate ({l4e_rel[0]:+.3f} at 15 Hz -> "
        f"{l4e_rel[1]:+.3f} at 30 Hz) -- a port whose output does not track its input is a switch"
    )
    assert all(a > b + 0.05 for a, b in zip(l4e_rel, l23e_rel)), (
        f"the response must be LAYER-SPECIFIC: L4e (thalamic in-degree {net.k_th[2]:.1f}) must rise "
        f"more than L2/3e (in-degree {net.k_th[0]:.1f}, no thalamic afferent) at every rate "
        f"(L4e {[round(x, 3) for x in l4e_rel]} vs L2/3e {[round(x, 3) for x in l23e_rel]})"
    )
    assert refuses_negative, "thalamic_drive must refuse a negative firing rate"

    print(
        f"self-check OK: 8/8 populations active {min(rates):.2f}-{max(rates):.2f} Hz, "
        f"asynchronous-irregular (mean CV {mean_cv:.2f} > 0.7), "
        f"layer signature L5e > L2/3e ({rates[4]:.2f} > {rates[0]:.2f} Hz); deterministic; "
        f"INPUT PORT drives L4e {l4e_rel[0]:+.0%} -> {l4e_rel[1]:+.0%} across 15/30 Hz while "
        f"L2/3e (no thalamic afferent) stays {l23e_rel[1]:+.0%}; undriven port inert"
    )


if __name__ == "__main__":
    main()
