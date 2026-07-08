"""
Wilson-Cowan neural mass model -- mesoscale excitatory/inhibitory population firing-rate dynamics
with the full bifurcation repertoire: a stable fixed point, a limit-cycle oscillation, OR bistable
hysteresis-memory, depending only on the parameters (pure stdlib).

Two coupled mean firing rates -- E (excitatory pool) and I (inhibitory pool) -- each relaxing (time
constants tau_E, tau_I) toward a sigmoid of its net input. Connection weights w_EE, w_EI, w_IE, w_II
and external drives P (to E), Q (to I) set the operating regime:

    tau_E * dE/dt = -E + f( w_EE*E - w_EI*I + P )
    tau_I * dI/dt = -I + f( w_IE*E - w_II*I + Q )

with a logistic transfer f(x) = 1/(1+exp(-a*(x-theta))) (a = slope, theta = half-activation). We
integrate with a fixed forward-Euler step (small dt). Because E_new = (1-dt)*E + dt*f(.) with f in
(0,1), the unit square [0,1]^2 is an invariant region, so both rates stay bounded in [0,1] -- the
qualitative differences between regimes are genuine attractor structure, not numerical blow-up. This
is the canonical mesoscale cortical E/I rate model (Wilson & Cowan, Biophys J 12:1, 1972) whose SAME
equations, under three hand-picked parameter presets, exhibit three qualitatively different attractors.

DISTINCTNESS (read cpg_oscillator / kuramoto_sync / divisive_normalization before writing this):
- `cpg_oscillator.py` is a Matsuoka HALF-CENTER: a SPECIFIC two-cell mutual-inhibition-plus-adaptation
  circuit engineered to emit ONE locomotor rhythm to drive the arm. It has an adaptation state variable
  and produces a single designed oscillation. Wilson-Cowan is the GENERAL mesoscale E/I rate field: no
  adaptation variable, and the SAME equations give a stable fixed point, a limit cycle, OR bistable
  hysteresis under different parameters -- a bifurcation repertoire, not one designed rhythm. Here the
  oscillation comes from the E->I->E loop (with slower inhibition), the bistability from E
  self-excitation (w_EE) -- neither mechanism exists in the Matsuoka model.
- `kuramoto_sync.py` couples N PHASE oscillators and measures synchronization via the order parameter r
  (a phase-transition in phase-locking). This is FIRING-RATE dynamics of two interacting pools -- a
  different state variable and a different phenomenon (attractor structure vs phase coherence).
- `divisive_normalization.py` is an INSTANTANEOUS algebraic gain rule -- no time evolution, no
  attractors. This is a time-evolving dynamical system with fixed points and limit cycles. And it is a
  POPULATION rate model, not a single-neuron model (`lif_neuron` / `izhikevich`).

The classic 2-population Wilson-Cowan mean-field ODEs with a fixed forward-Euler step and
hand-picked parameter presets for each regime -- a REDUCED rate model, not spiking biophysics, no
spatial extent and no conduction delays; presets are chosen so each bifurcation regime is clearly
exhibited. Not cognition/AGI.

  python -m snn_data_lake.wilson_cowan        # (also run by run_selfchecks.py)
"""
import math
import random


def _sigmoid(x, a, theta):
    """Logistic transfer f(x) = 1/(1+exp(-a*(x-theta))), output in (0,1)."""
    return 1.0 / (1.0 + math.exp(-a * (x - theta)))


class WilsonCowan:
    """Two coupled rate populations E (excitatory) and I (inhibitory); forward-Euler integration.

    step() advances one Euler update and returns (E, I); run(steps) returns the (E,I) time series;
    settle()/fixed_point() run past the transient and return the steady (E,I); sweep_drive() ramps the
    external drive P point-by-point, each settled from the previous state, to expose hysteresis."""

    def __init__(self, w_ee=10.0, w_ei=10.0, w_ie=10.0, w_ii=2.0, tau_e=1.0, tau_i=1.0,
                 p=1.0, q=0.0, a=1.0, theta=4.0, dt=0.1, seed=0):
        self.w_ee, self.w_ei, self.w_ie, self.w_ii = w_ee, w_ei, w_ie, w_ii
        self.tau_e, self.tau_i = tau_e, tau_i
        self.p, self.q = p, q
        self.a, self.theta = a, theta
        self.dt = dt
        self.seed = seed
        self.e = 0.1
        self.i = 0.1

    # ---- named parameter presets, one per bifurcation regime ----
    @classmethod
    def fixed_point_regime(cls, dt=0.1, seed=0):
        """Moderate E/I gain, no self-excitation runaway -> one global stable fixed point (attractor)."""
        return cls(w_ee=10.0, w_ei=10.0, w_ie=10.0, w_ii=2.0, tau_e=1.0, tau_i=1.0,
                   p=1.0, q=0.0, a=1.0, theta=4.0, dt=dt, seed=seed)

    @classmethod
    def limit_cycle_regime(cls, dt=0.1, seed=0):
        """Strong E->I->E loop with slower inhibition (tau_I > tau_E) -> sustained limit-cycle."""
        return cls(w_ee=16.0, w_ei=12.0, w_ie=12.0, w_ii=3.0, tau_e=1.0, tau_i=2.0,
                   p=1.25, q=0.0, a=1.0, theta=4.0, dt=dt, seed=seed)

    @classmethod
    def bistable_regime(cls, dt=0.1, seed=0):
        """Strong E self-excitation (w_EE), weaker inhibition, steep transfer -> two stable states."""
        return cls(w_ee=11.0, w_ei=6.0, w_ie=8.0, w_ii=2.0, tau_e=1.0, tau_i=1.0,
                   p=0.0, q=0.0, a=1.6, theta=5.0, dt=dt, seed=seed)

    @property
    def state(self):
        return (self.e, self.i)

    def reset(self, e0=0.1, i0=0.1):
        """Set the (E, I) state; returns it."""
        self.e, self.i = e0, i0
        return self.state

    def _derivatives(self, e, i):
        """(dE/dt, dI/dt) of the Wilson-Cowan field at state (e, i)."""
        de = (-e + _sigmoid(self.w_ee * e - self.w_ei * i + self.p, self.a, self.theta)) / self.tau_e
        di = (-i + _sigmoid(self.w_ie * e - self.w_ii * i + self.q, self.a, self.theta)) / self.tau_i
        return de, di

    def rate_of_change(self):
        """(|dE/dt|, |dI/dt|) at the current state -- ~0 at a settled fixed point."""
        de, di = self._derivatives(self.e, self.i)
        return (abs(de), abs(di))

    def step(self):
        """One forward-Euler update; returns the new (E, I)."""
        de, di = self._derivatives(self.e, self.i)
        self.e += self.dt * de
        self.i += self.dt * di
        return self.state

    def run(self, steps):
        """Advance `steps` Euler updates; return the (E, I) time series."""
        return [self.step() for _ in range(steps)]

    def settle(self, steps=2000):
        """Run past the transient; return the steady (E, I)."""
        for _ in range(steps):
            self.step()
        return self.state

    def fixed_point(self, steps=2000):
        """Alias for settle(): the steady (E, I) reached after transients from the current state."""
        return self.settle(steps)

    def sweep_drive(self, p_values, up=True, settle_steps=1200):
        """Ramp external drive P over p_values, settling each point FROM THE PREVIOUS state (not a
        fresh start), so the returned steady E depends on history -> a rising vs falling ramp traces
        the two branches of a hysteresis loop. `up` ramps ascending, else descending; the returned
        list of settled E is aligned to the input order of p_values either way."""
        seq = list(p_values) if up else list(reversed(p_values))
        out = []
        for pv in seq:
            self.p = pv
            self.settle(settle_steps)
            out.append(self.e)
        return out if up else list(reversed(out))

    @staticmethod
    def oscillation_amplitude(series, tail_frac=0.4):
        """Peak-to-peak amplitude of E over the last `tail_frac` of a run's (E, I) series."""
        tail = series[-max(1, int(len(series) * tail_frac)):]
        es = [e for e, _ in tail]
        return max(es) - min(es)

    @staticmethod
    def mean_crossings(series, tail_frac=0.4):
        """Count times E crosses its own tail mean over the last `tail_frac` -- periodicity witness."""
        tail = series[-max(1, int(len(series) * tail_frac)):]
        es = [e for e, _ in tail]
        m = sum(es) / len(es)
        return sum(1 for x, y in zip(es, es[1:]) if (x - m) * (y - m) < 0.0)


def main():
    # ---- 1. STABLE FIXED POINT: many initial conditions converge to ONE steady (E,I) ----
    fp = WilsonCowan.fixed_point_regime()
    rng = random.Random(fp.seed)                       # deterministic jittered ICs (no module-level seed)
    settled, final_rates = [], []
    for _ in range(6):
        fp.reset(rng.random(), rng.random())           # random start in the unit square
        settled.append(fp.settle(2500))
        final_rates.append(max(fp.rate_of_change()))   # per-step change at the end -> ~0 if converged
    e0, i0 = settled[0]
    ic_spread = max(math.hypot(e - e0, i - i0) for e, i in settled)   # do all ICs agree?
    final_change = max(final_rates)

    # ---- 2. LIMIT CYCLE: strong E->I->E loop sustains a bounded oscillation (no decay, no blow-up) ----
    lc = WilsonCowan.limit_cycle_regime()
    lc.reset(0.1, 0.1)
    series = lc.run(4000)
    amp = WilsonCowan.oscillation_amplitude(series, tail_frac=0.4)
    crossings = WilsonCowan.mean_crossings(series, tail_frac=0.4)
    e_all = [e for e, _ in series]
    e_lo, e_hi = min(e_all), max(e_all)
    amp_early = WilsonCowan.oscillation_amplitude(series[1600:2400], tail_frac=1.0)   # mid-run window
    amp_late = WilsonCowan.oscillation_amplitude(series[3200:4000], tail_frac=1.0)    # end window

    # ---- 3. BISTABILITY + HYSTERESIS: same P range, up-ramp vs down-ramp settle to DIFFERENT E ----
    bi = WilsonCowan.bistable_regime()
    p_values = [round(0.25 * k, 3) for k in range(25)]     # 0.00 .. 6.00
    bi.reset(0.0, 0.0)
    up_curve = bi.sweep_drive(p_values, up=True, settle_steps=1200)     # ramp P up from the low state
    bi.reset(1.0, 1.0)
    down_curve = bi.sweep_drive(p_values, up=False, settle_steps=1200)  # ramp P down from the high state
    gaps = [abs(u - d) for u, d in zip(up_curve, down_curve)]
    max_gap = max(gaps)
    gap_idx = gaps.index(max_gap)
    p_gap = p_values[gap_idx]
    n_bistable = sum(1 for g in gaps if g > 0.2)           # P values with two coexisting stable states

    # ---- 4. DETERMINISM: same params -> identical trajectories across fresh instances ----
    det_a = WilsonCowan.limit_cycle_regime()
    det_a.reset(0.1, 0.1)
    det_b = WilsonCowan.limit_cycle_regime()
    det_b.reset(0.1, 0.1)
    identical = det_a.run(500) == det_b.run(500)

    print("=" * 60)
    print("WILSON-COWAN neural mass -- E/I rate dynamics + bifurcation repertoire")
    print("=" * 60)
    print("tau: forward-Euler dt=0.1; transfer f(x)=1/(1+exp(-a*(x-theta)))")
    print(f"1 fixed point : 6 ICs -> steady (E,I)=({e0:.4f},{i0:.4f}); "
          f"spread across ICs={ic_spread:.2e}, final |dE/dt|={final_change:.2e}")
    print(f"2 limit cycle : tail peak-to-peak amp={amp:.3f}, mean-crossings={crossings}, "
          f"E in [{e_lo:.3f},{e_hi:.3f}] (bounded)")
    print(f"                mid-run amp={amp_early:.3f} -> end amp={amp_late:.3f} (sustained, no decay)")
    print(f"3 bistable    : max hysteresis gap={max_gap:.3f} at P={p_gap:.2f} "
          f"(up-branch E={up_curve[gap_idx]:.3f} vs down-branch E={down_curve[gap_idx]:.3f})")
    print(f"                {n_bistable} of {len(p_values)} P values have two stable states (history-dependent)")
    print(f"4 determinism : identical limit-cycle series across fresh instances = {identical}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    assert ic_spread < 1e-6, f"all ICs must converge to ONE fixed point (spread {ic_spread:.2e})"
    assert final_change < 1e-6, f"a true fixed point must have per-step change ~0 (|dE/dt|={final_change:.2e})"
    assert amp > 0.05, f"limit cycle must sustain a bounded oscillation (amp {amp:.3f})"
    assert crossings >= 4, f"limit cycle must be periodic (mean-crossings {crossings})"
    assert 0.0 <= e_lo and e_hi <= 1.0, f"limit cycle must stay bounded in [0,1] ({e_lo:.3f}..{e_hi:.3f})"
    assert amp_late > 0.05 and amp_late > 0.5 * amp_early, \
        f"oscillation must not decay to a fixed point (mid {amp_early:.3f} -> end {amp_late:.3f})"
    assert max_gap > 0.2, f"up and down drive ramps must trace a hysteresis loop (gap {max_gap:.3f})"
    assert n_bistable >= 3, f"a non-trivial P range must be bistable ({n_bistable} points)"
    assert identical, "same params/seed must give identical trajectories"
    print(f"self-check OK: fixed point ({e0:.3f},{i0:.3f}) [spread {ic_spread:.0e}]; "
          f"limit cycle amp {amp:.2f}/{crossings} crossings in [{e_lo:.2f},{e_hi:.2f}]; "
          f"hysteresis gap {max_gap:.2f} over {n_bistable} P; deterministic")


if __name__ == "__main__":
    main()
