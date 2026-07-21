"""
Medullary vital nuclei -- a spiking respiratory + cardiac CPG under homeostatic set-point control.

The caudal medulla is the autonomic homeostat: a pre-Botzinger-complex rhythm generator paces
breathing, and the NTS -> RVLM baroreflex paces the heart/vasomotor tone. Neither runs open-loop --
both are servos closing a NEGATIVE-FEEDBACK loop onto a regulated variable (blood CO2, arterial
pressure). This module models that FUNCTION at rate/phase level: a bursting relaxation oscillator
for the breath, and two proportional reflex arcs that hold a chemical/mechanical variable at its
set-point against a load or a disturbance.

Mechanism (rate/phase-level, deterministic):
  1. Pre-Botzinger respiratory rhythm -- a relaxation (integrate-to-threshold-and-reset) oscillator.
     A phase variable ramps up at rate proportional to respiratory DRIVE; when it crosses threshold
     it emits one inspiratory BURST and resets. Constant drive -> a regular train of bursts with a
     fixed inter-burst interval. This is the rhythm, not the biophysics: no Na_p/leak currents, just
     a phase that fires and resets.
  2. Chemoreceptor homeostasis -- a "CO2" pool rises from a metabolic PRODUCTION load and falls by a
     fixed tidal amount on every inspiratory burst (a breath clears CO2). Central chemoreceptors set
     respiratory drive by NEGATIVE feedback: drive = drive0 + k_chemo * (CO2 - set_point). When CO2
     climbs, drive rises, the oscillator breathes FASTER, clearance rises, and CO2 is pulled back to
     the set-point. The loop self-selects the breathing rate that makes clearance equal production.
  3. Baroreflex -- an arterial "pressure" pool has an open-loop equilibrium that a disturbance shifts
     away from the set-point (a change in resistance/volume). The NTS senses pressure and the RVLM
     drives heart-rate/vasomotor tone by NEGATIVE feedback: control = k_baro * (set_point - P). With
     high loop gain the reflex rejects the disturbance and holds P near the set-point.

Fair control (un-riggable -- the regulation must do the work, not a hardcoded output):
  - The chemoreflex only converges to the set-point WITH feedback. Run open-loop (drive frozen at
    drive0) and CO2 drifts away, because baseline clearance no longer tracks the load -- proving the
    set-point is held by the closed loop, not baked in.
  - The output is genuinely oscillatory: many inspiratory bursts with a roughly constant, non-zero
    period. A tonic/flat drive that never resets would emit no bursts and fail this outright.
  - The chemoreflex answers demand: raise the metabolic CO2 load and the steady-state breathing rate
    rises (breathe faster under load) -- the servo, not a constant, sets the rate.
  - The baroreflex drives a step pressure disturbance back toward the set-point WITH feedback and
    leaves it stranded at the disturbed equilibrium WITHOUT it.

Honest framing: a reduced rate/phase-level model of medullary rhythm generation and homeostatic
reflex control -- NOT ion-channel Butera-Rinzel pre-Botzinger bursting and NOT a real baroreflex
arc with vagal/sympathetic conduction. It models the FUNCTION (rhythmic autonomic output held on a
closed-loop set-point) rather than the biophysics. These are AUTONOMIC effectors (breathing, heart
rate); they never touch the voluntary arm actuator or the brainstem safety envelope.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.medulla.vital_nuclei
"""

import random


class MedullaVitalNuclei:
    """Spiking respiratory/cardiac CPG with chemoreceptor and baroreflex homeostatic set-points.

    run_chemoreflex(...) paces a pre-Botzinger relaxation oscillator whose rate is set by a
    negative-feedback CO2 loop; run_baroreflex(...) regulates arterial pressure against a step
    disturbance via an NTS -> RVLM proportional reflex. With feedback each regulated variable settles
    at its set-point; without feedback it drifts (chemo) or stays stranded at the disturbance (baro).
    """

    def __init__(
        self,
        seed=0,
        # respiratory relaxation oscillator (pre-Botzinger)
        resp_gain=0.05,  # phase ramp per step per unit drive (sets breathing frequency)
        resp_threshold=1.0,  # phase at which an inspiratory burst fires and the phase resets
        tidal=1.0,  # CO2 cleared by one inspiratory burst (tidal volume)
        # chemoreceptor set-point loop
        co2_set_point=40.0,  # regulated arterial-CO2 target (mmHg-like units)
        drive0=1.0,  # baseline respiratory drive at the set-point (open-loop drive too)
        k_chemo=0.5,  # chemoreceptor feedback gain (drive per unit CO2 error)
        drive_min=0.05,  # drive floor so the oscillator never fully stalls
        # baroreflex set-point loop
        map_set_point=90.0,  # regulated mean-arterial-pressure target (mmHg-like units)
        k_baro=2.0,  # NTS -> RVLM baroreflex gain (control per unit pressure error)
        baro_leak=0.05,  # open-loop pressure relaxation toward its (disturbed) equilibrium
        baro_dt=0.1,  # integration step for the pressure plant (keeps the loop stable)
    ):
        self.rng = random.Random(seed)
        self.resp_gain = resp_gain
        self.resp_threshold = resp_threshold
        self.tidal = tidal
        self.co2_set_point = co2_set_point
        self.drive0 = drive0
        self.k_chemo = k_chemo
        self.drive_min = drive_min
        self.map_set_point = map_set_point
        self.k_baro = k_baro
        self.baro_leak = baro_leak
        self.baro_dt = baro_dt

    # ---- respiratory drive (central chemoreceptor) --------------------------
    def _resp_drive(self, co2, feedback):
        """Respiratory drive from CO2. WITH feedback: proportional to the CO2 error above set-point
        (breathe harder when CO2 is high). WITHOUT feedback: frozen at the baseline drive0."""
        if not feedback:
            return self.drive0
        return max(self.drive_min, self.drive0 + self.k_chemo * (co2 - self.co2_set_point))

    def run_chemoreflex(
        self, steps=3000, co2_production=0.06, co2_init=55.0, feedback=True, noise=0.0
    ):
        """Run the pre-Botzinger oscillator under the CO2 set-point loop.

        Each step: CO2 accrues the metabolic production load; the phase ramps at resp_gain * drive;
        crossing threshold fires an inspiratory burst that clears one tidal volume of CO2 and resets
        the phase. Returns the CO2 trace and the step indices of every inspiratory burst.
        `noise` adds a small deterministic (rng-seeded) jitter to production so the trace exercises
        the per-instance rng -- the determinism self-check relies on this."""
        co2 = float(co2_init)
        phase = 0.0
        co2_trace = []
        burst_steps = []
        for t in range(steps):
            prod = co2_production
            if noise:
                prod += self.rng.uniform(-noise, noise) * co2_production
            co2 += prod
            drive = self._resp_drive(co2, feedback)
            phase += self.resp_gain * drive
            if phase >= self.resp_threshold:
                phase -= self.resp_threshold
                co2 = max(0.0, co2 - self.tidal)
                burst_steps.append(t)
            co2_trace.append(co2)
        return {"co2": co2_trace, "bursts": burst_steps}

    def run_baroreflex(self, steps=800, disturbance=30.0, feedback=True, noise=0.0):
        """Regulate arterial pressure against a step disturbance via the NTS -> RVLM baroreflex.

        A disturbance shifts the plant's open-loop equilibrium off the set-point. WITH feedback the
        RVLM control term pulls pressure back toward the set-point; WITHOUT it, pressure relaxes to
        the disturbed equilibrium and stays there. Returns the pressure trace."""
        equilibrium = self.map_set_point + disturbance
        p = float(self.map_set_point)
        trace = []
        for _ in range(steps):
            control = self.k_baro * (self.map_set_point - p) if feedback else 0.0
            dp = -self.baro_leak * (p - equilibrium) + control
            if noise:
                dp += self.rng.uniform(-noise, noise)
            p += self.baro_dt * dp
            trace.append(p)
        return {"map": trace}

    # ---- trace summaries ----------------------------------------------------
    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    @staticmethod
    def _intervals(burst_steps):
        """Inter-burst intervals (steps between successive inspiratory bursts)."""
        return [b - a for a, b in zip(burst_steps, burst_steps[1:])]

    def steady_co2(self, res, tail=0.5):
        """Mean CO2 over the last `tail` fraction of a chemoreflex run (drop the transient)."""
        co2 = res["co2"]
        start = int(len(co2) * (1.0 - tail))
        return self._mean(co2[start:])

    def steady_rate(self, res, tail=0.5):
        """Steady-state breathing rate = inspiratory bursts per step over the run's tail window."""
        n = len(res["co2"])
        start = int(n * (1.0 - tail))
        bursts = [b for b in res["bursts"] if b >= start]
        return len(bursts) / max(1, n - start)

    def steady_map(self, res, tail=0.5):
        """Mean arterial pressure over the last `tail` fraction of a baroreflex run."""
        m = res["map"]
        start = int(len(m) * (1.0 - tail))
        return self._mean(m[start:])


def main():
    core = MedullaVitalNuclei(seed=0)

    # (1) chemoreceptor set-point regulation: closed loop settles at the set-point; open loop drifts.
    closed = core.run_chemoreflex(feedback=True)
    open_loop = core.run_chemoreflex(feedback=False)
    co2_closed = core.steady_co2(closed)
    co2_open = core.steady_co2(open_loop)
    err_closed = abs(co2_closed - core.co2_set_point)
    err_open = abs(co2_open - core.co2_set_point)

    # (2) genuinely oscillatory: many bursts with a roughly constant period.
    n = len(closed["co2"])
    steady_bursts = [b for b in closed["bursts"] if b >= n // 2]
    ivals = core._intervals(steady_bursts)
    mean_ival = core._mean(ivals)
    max_dev = max((abs(v - mean_ival) for v in ivals), default=0.0)

    # (3) chemoreflex answers demand: more metabolic load -> faster steady-state breathing.
    low = core.run_chemoreflex(co2_production=0.05, co2_init=40.0)
    high = core.run_chemoreflex(co2_production=0.09, co2_init=40.0)
    rate_low = core.steady_rate(low)
    rate_high = core.steady_rate(high)

    # (4) baroreflex: a step pressure disturbance is corrected WITH feedback, not without.
    baro_on = core.run_baroreflex(disturbance=30.0, feedback=True)
    baro_off = core.run_baroreflex(disturbance=30.0, feedback=False)
    map_on = core.steady_map(baro_on)
    map_off = core.steady_map(baro_off)
    baro_err_on = abs(map_on - core.map_set_point)
    baro_err_off = abs(map_off - core.map_set_point)

    print("=" * 72)
    print("MEDULLA VITAL NUCLEI -- respiratory + cardiac CPG on homeostatic set-points")
    print("=" * 72)
    print(
        f"CO2 set-point: {core.co2_set_point:.1f} | MAP set-point: {core.map_set_point:.1f} | "
        f"breath threshold: {core.resp_threshold:.1f}"
    )
    print("-" * 72)
    print("(1) chemoreceptor CO2 regulation (start 55.0, metabolic load 0.06):")
    print(f"    closed loop -> steady CO2 {co2_closed:6.2f}  (error {err_closed:5.2f})")
    print(
        f"    open loop   -> steady CO2 {co2_open:6.2f}  (error {err_open:5.2f}, drifts off target)"
    )
    print("-" * 72)
    print(
        f"(2) respiratory rhythm: {len(steady_bursts)} inspiratory bursts, "
        f"mean period {mean_ival:.1f} steps, max deviation {max_dev:.1f} (regular)"
    )
    print("-" * 72)
    print("(3) chemoreflex vs metabolic demand (breaths per step at steady state):")
    print(f"    low load  (0.05) -> rate {rate_low:.4f}")
    print(f"    high load (0.09) -> rate {rate_high:.4f}  (breathes faster under load)")
    print("-" * 72)
    print("(4) baroreflex vs a +30 pressure disturbance:")
    print(
        f"    with feedback    -> steady MAP {map_on:6.2f}  (error {baro_err_on:5.2f}, corrected)"
    )
    print(
        f"    without feedback -> steady MAP {map_off:6.2f}  (error {baro_err_off:5.2f}, stranded)"
    )
    print("=" * 72)

    # ---- un-riggable fair-control self-checks (deterministic asserts) --------
    # (1) the set-point is held by the loop: closed-loop error small, open-loop error large.
    assert err_closed < 2.0, (
        f"closed-loop CO2 must settle near the set-point (error {err_closed:.2f})"
    )
    assert err_open > 8.0, f"open-loop CO2 must drift off the set-point (error {err_open:.2f})"
    assert err_open > 4.0 * err_closed, "feedback must regulate far tighter than open loop"
    # (2) genuinely oscillatory: multiple bursts, non-zero and roughly constant period (not tonic).
    assert len(steady_bursts) >= 10, "the respiratory CPG must emit many bursts (not tonic/flat)"
    assert mean_ival > 1.0, "inter-burst interval must be non-zero (a real rhythm, not every-step)"
    assert max_dev <= 0.25 * mean_ival + 1.0, (
        "the breathing period must be roughly constant (regular)"
    )
    # (3) chemoreflex responds to demand: heavier CO2 load -> faster steady breathing.
    assert rate_high > rate_low, "a higher metabolic CO2 load must raise the steady breathing rate"
    # (4) baroreflex rejects the disturbance only with feedback.
    assert baro_err_on < 2.0, (
        f"the baroreflex must correct the disturbance (error {baro_err_on:.2f})"
    )
    assert baro_err_off > 20.0, (
        f"without feedback pressure must stay disturbed (error {baro_err_off:.2f})"
    )
    assert baro_err_off > 5.0 * baro_err_on, (
        "the baroreflex must reject the disturbance it is given"
    )
    # Determinism: same seed -> identical traces (rng-seeded metabolic/vasomotor jitter reproduces).
    a = MedullaVitalNuclei(seed=7)
    b = MedullaVitalNuclei(seed=7)
    ra = a.run_chemoreflex(noise=0.02)
    rb = b.run_chemoreflex(noise=0.02)
    assert ra == rb, "same seed must reproduce identical chemoreflex traces"
    assert a.run_baroreflex(noise=0.5) == b.run_baroreflex(noise=0.5), (
        "same seed must reproduce identical baroreflex traces"
    )

    print(
        f"self-check OK: CO2 closed-loop error {err_closed:.2f} vs open-loop {err_open:.2f}; "
        f"{len(steady_bursts)} bursts @ period {mean_ival:.1f}; "
        f"rate {rate_low:.3f}->{rate_high:.3f} under load; "
        f"baroreflex error {baro_err_on:.2f} vs {baro_err_off:.2f}; deterministic"
    )


if __name__ == "__main__":
    main()
