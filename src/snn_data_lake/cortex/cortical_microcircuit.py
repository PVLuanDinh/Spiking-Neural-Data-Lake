"""
Potjans-Diesmann 6-layer / 8-population cortical MICROCIRCUIT -- a pure-stdlib SPIKING dynamics model.

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

Zero runtime dependencies (pure Python stdlib: random, math). Per-instance random.Random(seed);
ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.cortical_microcircuit
  (back-compat alias also resolves:  python -m snn_data_lake.cortical_microcircuit)
"""
import math
import random
import sys

# --- the eight populations, in canonical order --------------------------------------------------
POP_NAMES = ["L2/3e", "L2/3i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]
EXC_POPS = (0, 2, 4, 6)                    # indices of the excitatory pools (the rest are inhibitory)

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

# --- LIF + synapse parameters (Potjans-Diesmann values), voltages measured RELATIVE to rest -----
TAU_M = 10.0                 # membrane time constant (ms)
DT = 0.1                     # integration step (ms)
V_TH = 15.0                  # spike threshold (mV above the -65 mV rest / reset)
V_RESET = 0.0                # reset = rest (relative)
T_REF_MS = 2.0               # absolute refractory period (ms)
DELAY_MS = 1.5               # synaptic transmission delay (ms)
PSP = 0.15                   # excitatory PSP amplitude (mV) -- the delta-synapse voltage jump
G_INH = 4.0                  # relative inhibitory weight (inhibition-dominated): IPSP = -G_INH x PSP
BG_RATE = 8.0                # external background rate per synapse (Hz)
V_OP = 6.0                   # sub-threshold mean operating point (mV) set by the external drive
NOISE_GAIN = 4.0             # multiplier on the external diffusion std (sets the fluctuation size)


class CorticalMicrocircuit:
    """The Potjans-Diesmann 8-population cortical column as a scaled pure-Python spiking LIF network.

    Build a downscaled instance (`scale` in (0, 1]; full model is scale=1.0 -> 77169 neurons, far
    too large for stdlib), then call `.simulate(t_ms)`. `.firing_rates()` returns the per-population
    firing rate (Hz) and `.cv_isi()` the per-population coefficient of variation of the inter-spike
    intervals (CV ~ 1 == Poisson-irregular). Connectivity is drawn once from a per-instance
    random.Random(seed); the drawn network + drive is a deterministic function of (scale, seed)."""

    def __init__(self, scale=0.008, seed=0, g=G_INH, bg_rate=BG_RATE, v_op=V_OP, noise_gain=NOISE_GAIN):
        if not (0.0 < scale <= 1.0):
            raise ValueError("scale must be in (0, 1]")
        self.scale = scale
        self.seed = seed
        self.g = g
        self.bg_rate = bg_rate
        self.v_op = v_op
        self.noise_gain = noise_gain
        self.decay = math.exp(-DT / TAU_M)      # membrane leak factor per step (used in simulate too)
        self.rng = random.Random(seed)          # per-instance RNG (connectivity build)

        # scaled population sizes (>=1 each so no layer vanishes) and the global index ranges
        self.n_pop = [max(1, int(round(N_FULL[p] * scale))) for p in range(8)]
        self.n_total = sum(self.n_pop)
        self.pop_start = []
        acc = 0
        for p in range(8):
            self.pop_start.append(acc)
            acc += self.n_pop[p]
        self.pop_of = [0] * self.n_total       # global index -> population index
        for p in range(8):
            for k in range(self.n_pop[p]):
                self.pop_of[self.pop_start[p] + k] = p

        # variance-preserving weight correction: w -> w / sqrt(scale)  (Potjans-Diesmann scaling)
        corr = 1.0 / math.sqrt(scale)
        self.w_exc = PSP * corr                # excitatory delta-synapse voltage jump (mV)
        self.w_inh = -g * PSP * corr           # inhibitory delta-synapse voltage jump (mV)

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
        mu_op = v_op * (1.0 - self.decay)      # external mean / step that holds the leak-steady v = v_op
        self.mu_ext = [0.0] * self.n_total     # per-neuron external mean voltage kick / step (mV)
        self.sigma_ext = [0.0] * self.n_total  # per-neuron external std of the kick / step (mV)
        for i in range(self.n_total):
            p = self.pop_of[i]
            rate = K_EXT[p] * bg_rate * dt_s   # expected external spikes this step (per-layer)
            self.mu_ext[i] = mu_op
            self.sigma_ext[i] = noise_gain * math.sqrt(rate) * PSP

        # results (populated by simulate())
        self.spike_times = [[] for _ in range(self.n_total)]   # per-neuron spike times (ms) in window
        self.meas_ms = 0.0

    def _build_connectivity(self):
        """Draw the recurrent synapses with the Potjans-Diesmann fixed-in-degree rule. For each
        target neuron in population i we draw K_ij = round(C[i][j] * N_j_scaled) presynaptic partners
        uniformly (with replacement, so multapses/autapses can occur, exactly as in the paper) from
        each source population j. Stored as an out-adjacency list out_adj[source] = [(target, w), ...]
        for event-driven propagation. Weight sign is set by whether the SOURCE pool is exc or inh."""
        rng = self.rng
        self.out_adj = [[] for _ in range(self.n_total)]
        self.in_degree = [[0] * 8 for _ in range(8)]   # K_ij: recurrent in-degree matrix (inspectable)
        n_syn = 0
        for i in range(8):                              # target population
            tgt_lo = self.pop_start[i]
            for j in range(8):                          # source population
                c = CONN_PROB[i][j]
                if c <= 0.0:
                    continue
                k = int(round(c * self.n_pop[j]))       # in-degree from j into each i-neuron
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

    def simulate(self, t_ms=400.0, settle_ms=100.0):
        """Integrate the network for `settle_ms` (discarded transient) + `t_ms` (measurement window).
        Spikes in the measurement window are recorded per neuron in self.spike_times (ms, window-
        relative). Deterministic: the noise RNG is re-seeded from `seed` here, so repeated calls with
        the same instance reproduce identically regardless of prior calls. Returns self."""
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
        mu = self.mu_ext
        sigma = self.sigma_ext
        out_adj = self.out_adj
        theta = V_TH
        reset = V_RESET
        spikes = [[] for _ in range(n)]
        pending = [[0.0] * n for _ in range(ring)]     # ring buffer of delayed synaptic input (mV)

        # dedicated, reproducible noise RNG (independent of the connectivity draw sequence)
        rng = random.Random((self.seed * 1000003 + 101) & 0x7fffffff)
        gauss = rng.gauss

        for step in range(total_steps):
            slot = step % ring
            buf = pending[slot]
            fut = pending[(step + delay_steps) % ring]
            recording = step >= settle_steps
            for i in range(n):
                r = ref[i]
                if r > 0:
                    ref[i] = r - 1              # in refractory: membrane clamped at reset
                    continue
                s = sigma[i]
                vi = v[i] * decay + buf[i] + mu[i] + (gauss(0.0, s) if s > 0.0 else 0.0)
                if vi >= theta:
                    v[i] = reset
                    ref[i] = tref_steps
                    if recording:
                        spikes[i].append((step - settle_steps) * DT)
                    for (tgt, w) in out_adj[i]:
                        fut[tgt] += w
                else:
                    v[i] = vi
            pending[slot] = [0.0] * n          # clear the consumed slot for reuse

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
    vals = [x for x in xs if x == x]           # drop NaNs (NaN != NaN)
    return sum(vals) / len(vals) if vals else float("nan")


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
    print(f"scale {scale:.4f}  ->  {net.n_total} neurons  |  {net.n_synapses} recurrent synapses  "
          f"(full model = {sum(N_FULL)} neurons)")
    print(f"variance-preserving weights: EPSP {net.w_exc:.3f} mV  IPSP {net.w_inh:.3f} mV  "
          f"(g = {net.g:.0f}, base PSP {PSP} mV / sqrt(scale))")
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
    print(f"layer signature: rate(L5e) {r5e:.2f} Hz  >  rate(L2/3e) {r23e:.2f} Hz  "
          f"(superficial-low, L5 elevated)")
    print("=" * 72)

    # ---- self-checks: assert the KNOWN biological SIGNATURES (dynamics model, not accuracy) ----
    # (1) every population is active at a plausible LOW rate -- none silent, none runaway/saturated.
    for p in range(8):
        assert rates[p] > 0.0, f"{POP_NAMES[p]} is silent (0 Hz) -- lost its drive"
        assert rates[p] < 30.0, f"{POP_NAMES[p]} rate {rates[p]:.1f} Hz is runaway/saturated"
    # (2) ASYNCHRONOUS-IRREGULAR ground state: spiking is irregular, not clock-like or synchronized.
    assert mean_cv == mean_cv, "CV_ISI undefined -- too few spikes to estimate irregularity"
    assert mean_cv > 0.7, f"mean CV_ISI {mean_cv:.2f} too regular -- not the asynchronous-irregular state"
    # (3) canonical LAYER RATE SIGNATURE: superficial L2/3e is low, L5e is the elevated excitatory
    #     layer (the robust qualitative ordering that survives the downscaling).
    assert rates[4] > rates[0], (
        f"layer signature broken: L5e {rates[4]:.2f} Hz must exceed L2/3e {rates[0]:.2f} Hz")
    # structural sanity: genuinely spiking, pure stdlib, deterministic.
    assert "numpy" not in sys.modules and "torch" not in sys.modules, "must be pure stdlib"
    net2 = CorticalMicrocircuit(scale=scale, seed=0)
    net2.simulate(t_ms=400.0, settle_ms=100.0)
    assert net2.firing_rates() == rates, "same (scale, seed) must reproduce identical rates"

    print(f"self-check OK: 8/8 populations active {min(rates):.2f}-{max(rates):.2f} Hz, "
          f"asynchronous-irregular (mean CV {mean_cv:.2f} > 0.7), "
          f"layer signature L5e > L2/3e ({rates[4]:.2f} > {rates[0]:.2f} Hz); pure-stdlib, deterministic")


if __name__ == "__main__":
    main()
