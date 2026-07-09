"""
Hodgkin-Huxley neuron -- the conductance-based BIOPHYSICAL model where the action potential EMERGES.

Every other neuron in this package IMPOSES the spike. lif_neuron is leaky integrate-and-fire: a fixed
threshold plus a hard reset -- the "spike" is a rule the code applies, there are no ion channels and no
real action-potential shape. izhikevich_neuron is a 2-variable REDUCED phenomenological model (membrane v
plus recovery u) tuned to mimic firing patterns; it still resets v by hand at a cutoff. dendritic_neuron's
own docstring says it is "NOT Hodgkin-Huxley, no conductances, no real ion channels" -- a reduced rate/spike
caricature. THIS module is the real thing: the classic single-compartment space-clamped squid-axon model
(Hodgkin & Huxley, J Physiol 117:500, 1952). Nothing is imposed. The membrane voltage V and three gating
variables -- m, h (sodium activation / inactivation) and n (potassium activation) -- obey coupled ODEs, and
the overshoot, the sharp repolarization, the afterhyperpolarization, the all-or-none threshold, the
refractory period and the f-I curve ALL fall out of the nonlinear Na (m^3 h) and K (n^4) channel kinetics.

    C dV/dt = I_ext - g_Na*m^3*h*(V - E_Na) - g_K*n^4*(V - E_K) - g_L*(V - E_L)
    dx/dt   = alpha_x(V)*(1 - x) - beta_x(V)*x            for x in {m, h, n}

Standard squid parameters (per cm^2): C=1.0 uF, g_Na=120, g_K=36, g_L=0.3 mS; E_Na=50, E_K=-77,
E_L=-54.4 mV; resting V ~ -65 mV. Integrated with forward Euler at dt=0.01 ms (stable for these ODEs).
The alpha_n rate has a removable 0/0 singularity at V=-55 mV and alpha_m at V=-40 mV (denominator -> 0);
both are guarded with their L'Hopital limits (alpha_n -> 0.1, alpha_m -> 1.0) so the model never divides
by zero. m, h, n start at their steady-state values x_inf = alpha/(alpha+beta) at the resting voltage.

Honesty ceiling: the classic single-compartment space-clamped HH squid-axon ODEs, forward-Euler
integrated -- the biophysical-neuron level. NOT multi-compartment; no temperature / Q10 scaling; no channel
noise (deterministic); a caricature of one axon patch, not cognition. Pure standard library, zero deps.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.hodgkin_huxley        # (also run by run_selfchecks.py)
"""
import math

V_REST = -65.0        # mV, approximate resting potential the gates are initialised at


# --- voltage-dependent gating rate functions (V in mV); singularities guarded with L'Hopital limits ---
def alpha_n(v):
    x = v + 55.0
    if abs(x) < 1e-6:                       # removable 0/0 at V = -55 mV: L'Hopital limit
        return 0.1
    return 0.01 * x / (1.0 - math.exp(-x / 10.0))


def beta_n(v):
    return 0.125 * math.exp(-(v + 65.0) / 80.0)


def alpha_m(v):
    x = v + 40.0
    if abs(x) < 1e-6:                       # removable 0/0 at V = -40 mV: L'Hopital limit
        return 1.0
    return 0.1 * x / (1.0 - math.exp(-x / 10.0))


def beta_m(v):
    return 4.0 * math.exp(-(v + 65.0) / 18.0)


def alpha_h(v):
    return 0.07 * math.exp(-(v + 65.0) / 20.0)


def beta_h(v):
    return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))


def _x_inf(a, b):
    """Steady-state gate value x_inf = alpha / (alpha + beta)."""
    return a / (a + b)


class HodgkinHuxley:
    """Space-clamped Hodgkin-Huxley squid-axon neuron. step(i_ext) -> membrane V (mV).

    Conductance-based and biophysical: the action potential emerges from the Na (m^3 h) and K (n^4) gate
    kinetics, nothing is imposed. State V, m, h, n and the params/reversals are exposed as attributes.
    Deterministic (no randomness; `seed` is kept only for API symmetry with the other neuron models)."""

    def __init__(self, dt=0.01, seed=0):
        self.dt = dt                        # ms, forward-Euler step (0.01 ms is stable for these ODEs)
        self.seed = seed                    # unused: model is deterministic, kept for API symmetry
        # membrane capacitance and maximal conductances (per cm^2)
        self.c_m = 1.0                      # uF
        self.g_na = 120.0                   # mS
        self.g_k = 36.0                     # mS
        self.g_l = 0.3                      # mS
        # reversal potentials (mV)
        self.e_na = 50.0
        self.e_k = -77.0
        self.e_l = -54.4
        self.reset()

    def reset(self):
        """Reset to rest: V at V_REST, gates at their steady-state values there."""
        v = V_REST
        self.v = v
        self.m = _x_inf(alpha_m(v), beta_m(v))
        self.h = _x_inf(alpha_h(v), beta_h(v))
        self.n = _x_inf(alpha_n(v), beta_n(v))

    def step(self, i_ext):
        """Advance one dt under injected current i_ext (uA/cm^2); return the new membrane V (mV)."""
        v, m, h, n, dt = self.v, self.m, self.h, self.n, self.dt
        i_na = self.g_na * m ** 3 * h * (v - self.e_na)     # sodium current (m^3 h gating)
        i_k = self.g_k * n ** 4 * (v - self.e_k)            # potassium current (n^4 gating)
        i_l = self.g_l * (v - self.e_l)                     # leak
        dv = (i_ext - i_na - i_k - i_l) / self.c_m
        # forward Euler; gates evaluated at the pre-step voltage v
        self.v = v + dt * dv
        self.m = m + dt * (alpha_m(v) * (1.0 - m) - beta_m(v) * m)
        self.h = h + dt * (alpha_h(v) * (1.0 - h) - beta_h(v) * h)
        self.n = n + dt * (alpha_n(v) * (1.0 - n) - beta_n(v) * n)
        return self.v

    def run(self, currents):
        """Integrate a sequence of injected currents; return the V trace (list of mV, one per dt)."""
        return [self.step(c) for c in currents]

    def run_const(self, i_ext, ms):
        """Integrate a constant current i_ext for `ms` milliseconds; return the V trace."""
        return self.run([i_ext] * int(round(ms / self.dt)))

    @staticmethod
    def spike_count(trace, level=0.0):
        """Count action potentials = upward crossings of `level` mV (default 0 mV) in a V trace."""
        return sum(1 for i in range(1, len(trace))
                   if trace[i - 1] < level <= trace[i])


def _pulse(amp, dur_ms, total_ms, dt, delay_ms):
    """A brief rectangular current: `amp` from delay_ms for dur_ms, else 0, over total_ms."""
    n = int(round(total_ms / dt))
    lo = int(round(delay_ms / dt))
    hi = lo + int(round(dur_ms / dt))
    return [amp if lo <= i < hi else 0.0 for i in range(n)]


def _two_pulse(amp, dur_ms, gap_ms, total_ms, dt, delay_ms):
    """Two identical brief pulses separated by gap_ms (onset to onset); returns (current, gap_index)."""
    n = int(round(total_ms / dt))
    dd = int(round(dur_ms / dt))
    s1 = int(round(delay_ms / dt))
    s2 = int(round((delay_ms + gap_ms) / dt))
    cur = [0.0] * n
    for i in range(s1, min(s1 + dd, n)):
        cur[i] = amp
    for i in range(s2, min(s2 + dd, n)):
        cur[i] = amp
    return cur, s2


def main():
    dt = 0.01
    sc = HodgkinHuxley.spike_count

    # 1. ACTION POTENTIAL: a brief suprathreshold pulse -> a full biphasic AP (overshoot then AHP).
    ap = HodgkinHuxley(dt).run(_pulse(10.0, 1.0, 40.0, dt, 5.0))
    ap_peak, ap_trough = max(ap), min(ap)

    # 2. ALL-OR-NONE THRESHOLD (brief 0.5 ms pulse): a small stimulus step flips 0 spikes -> one full,
    #    stereotyped spike whose height does not grow much with a much stronger stimulus.
    sub = HodgkinHuxley(dt).run(_pulse(13.0, 0.5, 40.0, dt, 5.0))     # just sub-threshold
    supra = HodgkinHuxley(dt).run(_pulse(15.0, 0.5, 40.0, dt, 5.0))   # just supra-threshold
    strong = HodgkinHuxley(dt).run(_pulse(30.0, 0.5, 40.0, dt, 5.0))  # much stronger
    sub_spikes, supra_spikes = sc(sub), sc(supra)
    supra_h, strong_h = max(supra), max(strong)

    # 3. REFRACTORY PERIOD: two identical brief pulses. Short gap -> the second fails (refractory);
    #    long gap -> both fire full APs.
    cur_s, gi_s = _two_pulse(10.0, 1.0, 8.0, 60.0, dt, 5.0)
    cur_l, gi_l = _two_pulse(10.0, 1.0, 18.0, 60.0, dt, 5.0)
    short_trace = HodgkinHuxley(dt).run(cur_s)
    long_trace = HodgkinHuxley(dt).run(cur_l)
    short_spikes, long_spikes = sc(short_trace), sc(long_trace)
    short_peak2 = max(short_trace[gi_s:])       # peak during/after the second pulse
    long_peak2 = max(long_trace[gi_l:])

    # 4. f-I CURVE: silent below a rheobase, then repetitive firing whose rate rises with current.
    win = 150.0
    silent = sc(HodgkinHuxley(dt).run_const(2.0, win))              # below rheobase -> silent
    fi_currents = (7.0, 12.0, 20.0)                                 # above rheobase -> repetitive
    fi_rates = [sc(HodgkinHuxley(dt).run_const(i, win)) / (win / 1000.0) for i in fi_currents]

    # 5. DETERMINISM: same dt + current -> identical trace across two fresh instances.
    det_a = HodgkinHuxley(dt).run(_pulse(10.0, 1.0, 20.0, dt, 5.0))
    det_b = HodgkinHuxley(dt).run(_pulse(10.0, 1.0, 20.0, dt, 5.0))

    print("=" * 68)
    print("HODGKIN-HUXLEY NEURON -- conductance-based biophysical action potential")
    print("=" * 68)
    print(f"action potential    : peak {ap_peak:+.1f} mV (overshoot), trough {ap_trough:+.1f} mV (AHP)")
    print(f"all-or-none (0.5 ms): sub-thresh {sub_spikes} spike (peak {max(sub):+.1f} mV) -> "
          f"supra {supra_spikes} spike (peak {supra_h:+.1f} mV)")
    print(f"  stereotyped height: supra {supra_h:+.1f} mV vs strong {strong_h:+.1f} mV "
          f"(diff {abs(supra_h - strong_h):.1f} mV)")
    print(f"refractory          : short gap 8 ms -> {short_spikes} spike (2nd peak {short_peak2:+.1f} mV); "
          f"long gap 18 ms -> {long_spikes} spikes (2nd peak {long_peak2:+.1f} mV)")
    print(f"f-I curve           : I=2 -> {silent} Hz (silent); "
          + ", ".join(f"I={int(c)} -> {r:.0f} Hz" for c, r in zip(fi_currents, fi_rates)))
    print("=" * 68)

    # ---- self-check: a new BIOPHYSICAL member of the neuron-model family; conductance-based,
    # nothing imposed. Deterministic asserts, zero-dep stdlib, ASCII-only prints ----
    # 1. a real biphasic action potential: overshoot well above 0, then afterhyperpolarization below rest
    assert ap_peak > 20.0, f"AP must overshoot well above 0 mV (got {ap_peak:.1f})"
    assert ap_trough < V_REST, f"AP must repolarize below rest -- afterhyperpolarization (got {ap_trough:.1f})"
    # 2. all-or-none: sub-threshold -> no spike, supra-threshold -> one full spike; amplitude stereotyped
    assert sub_spikes == 0, f"sub-threshold brief pulse must evoke NO spike (got {sub_spikes})"
    assert supra_spikes == 1, f"supra-threshold brief pulse must evoke one full spike (got {supra_spikes})"
    assert supra_h > 20.0, f"the evoked spike must be a full-amplitude AP (peak {supra_h:.1f})"
    assert abs(supra_h - strong_h) < 6.0, \
        f"amplitude must be stereotyped, not graded (supra {supra_h:.1f} vs strong {strong_h:.1f})"
    # 3. refractory: second pulse fails at a short gap, both fire at a long gap
    assert short_spikes == 1 and short_peak2 < 0.0, \
        f"neuron must be refractory at an 8 ms gap: second pulse evokes no AP ({short_spikes}, {short_peak2:.1f})"
    assert long_spikes == 2 and long_peak2 > 20.0, \
        f"neuron must recover by an 18 ms gap: two full APs ({long_spikes}, {long_peak2:.1f})"
    # 4. f-I: silent below rheobase, repetitive above, rate monotone-increasing with current
    assert silent == 0, f"neuron must be silent below rheobase (got {silent} spikes at I=2)"
    assert fi_rates[0] >= 30.0, f"neuron must fire REPETITIVELY above rheobase (got {fi_rates[0]:.0f} Hz at I=7)"
    assert all(fi_rates[i + 1] >= fi_rates[i] for i in range(len(fi_rates) - 1)) and fi_rates[-1] > fi_rates[0], \
        f"f-I must be monotone-increasing above threshold ({fi_rates})"
    # 5. determinism
    assert det_a == det_b, "same dt/current -> identical V trace (deterministic)"

    print(f"self-check OK: AP peak {ap_peak:+.1f}/trough {ap_trough:+.1f} mV; all-or-none "
          f"{sub_spikes}->{supra_spikes} spike (stereotyped {supra_h:.0f}~{strong_h:.0f} mV); refractory "
          f"{short_spikes} vs {long_spikes}; f-I {fi_rates[0]:.0f}->{fi_rates[-1]:.0f} Hz; deterministic")


if __name__ == "__main__":
    main()
