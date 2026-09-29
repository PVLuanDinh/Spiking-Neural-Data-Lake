"""
Hypothalamic set-points -- negative-feedback homeostats whose reference is paced by the SCN clock.

The hypothalamus holds the body's regulated variables at target values. Each regulated variable
(core temperature, plasma osmolarity, energy/glucose) has a SET-POINT; a comparator subtracts the
sensed value from the set-point to form an ERROR, and an antagonistic EFFECTOR drives the variable
back toward the set-point. That is a classic negative-feedback loop: the loop -- not the passive
physics of the tissue -- is what pins the variable to its reference. The temperature set-point is
not fixed: the suprachiasmatic nucleus (SCN, see circadian_clock.py) PACES it, so core body temp
runs warm in the subjective day (active phase) and cool in the subjective night (rest phase).

Mechanism (rate-level, deterministic):
  1. Comparator: error = set_point - sensed, where sensed = variable (+ optional sensor noise).
  2. Antagonistic effector: drive = gain * error. A POSITIVE error (too low) drives a raising
     effector (shiver/vasoconstrict, ADH/thirst); a NEGATIVE error (too high) drives the opposite
     lowering effector (sweat/vasodilate). One signed drive == two opposed effectors.
  3. Plant update: variable += leak * (ambient - variable) + drive + disturbance. The `leak` term
     is the passive pull toward the environment (heat loss, water loss) that the effector must
     oppose; with gain 0 (open loop) the variable simply relaxes to `ambient`, far from the set-point.
  4. SCN pacing: the temperature set-point is set_point(t) = temp_setpoint(clock.phase(t)), read from
     the real CircadianClock phase. Warmest at the middle of subjective DAY (phase 0.25), coolest at
     the middle of subjective NIGHT (phase 0.75) -- a reduced ~0.8 C diurnal swing about 37 C.

Fair control (un-riggable):
  - FEEDBACK does the work: the closed loop (gain > 0) converges the variable to the set-point;
    the OPEN loop (gain 0, identical plant + ambient) relaxes to ambient and stays far off. If the
    physics alone regulated, the open loop would converge too -- it does not.
  - DISTURBANCE REJECTION: a cold-shock impulse knocks the variable far off, then the loop drives
    it back -- the post-shock error shrinks over time to the small closed-loop residual.
  - SCN PACES the set-point: with the clock ADVANCING, the set-point at subjective night differs
    from subjective day (it tracks the phase); with the clock FROZEN (phase held) the set-point is
    flat. The rhythm comes from the SCN, not from the homeostat.
  - BIDIRECTIONAL: both a too-cold and a too-hot start are corrected to the set-point by opposite-
    sign effectors -- the raising effector fires for cold, the lowering effector for hot.

Honest framing: a reduced negative-feedback homeostat with a circadian-modulated reference. It is
NOT the real neuroendocrine cascade (POA/OVLT thermoreceptors, ADH/vasopressin osmoregulation,
leptin/ghrelin energy balance). It models the control FUNCTION -- a regulated variable tracking an
SCN-paced set-point under disturbance -- on toy scalar plants. Nothing here is warm, thirsty, or
hungry; "temperature" is a number in an accumulator. Additive/opt-in; reuses circadian_clock.py.
Credit: Cannon 1929 (homeostasis); Pittendrigh & Daan 1976 (SCN pacing of diurnal set-points).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hypothalamus.set_points
"""

import math
import random

from snn_data_lake.sleep.circadian_clock import CircadianClock

# ---- temperature set-point constants (reduced core-body-temperature model) ---
SET_POINT_TEMP = 37.0  # mean core body temperature set-point in C
TEMP_AMP = 0.4  # circadian half-swing in C (peak-to-trough ~= 0.8 C, physiological)
AMBIENT_TEMP = 24.0  # room temperature the body passively relaxes toward (the disturbance sink)
DAY_PHASE = 0.25  # middle of subjective day (active phase): warmest set-point
NIGHT_PHASE = 0.75  # middle of subjective night (rest phase): coolest set-point

# ---- osmolarity set-point constants (reduced thirst/ADH model) ---------------
SET_POINT_OSMO = 285.0  # plasma osmolarity set-point in mOsm/kg
AMBIENT_OSMO = 305.0  # osmolarity drifts UP toward this with no ADH/thirst (ongoing water loss)


def temp_setpoint(phase, base=SET_POINT_TEMP, amp=TEMP_AMP):
    """Circadian core-body-temperature set-point paced by the SCN phase in [0, 1). Warmest at the
    middle of subjective DAY (phase 0.25, active), coolest at the middle of subjective NIGHT (phase
    0.75, rest). A reduced ~2*amp C peak-to-trough diurnal swing about `base`."""
    return base + amp * math.cos(2.0 * math.pi * (phase - DAY_PHASE))


def scn_phase_trace(steps, seed=0, zeitgeber=None):
    """Run a real SCN CircadianClock for `steps` steps and return its phase at each step. The phase
    that PACES the temperature set-point comes from the actual oscillator (circadian_clock.py), not
    a stand-in -- so the "SCN paces the set-point" claim is real. `zeitgeber` None == free-run."""
    clock = CircadianClock(seed=seed)
    phases = []
    for t in range(steps):
        light = 0.0 if zeitgeber is None else zeitgeber(t)
        clock.step(light)
        phases.append(clock.phase)
    return phases


class Homeostat:
    """Generic negative-feedback homeostat: comparator (set_point - sensed) -> error -> antagonistic
    effector drive that pushes a scalar regulated variable back to its set-point.

    The same regulator serves temperature, osmolarity, or energy -- only the units, gain, and the
    passive `ambient` sink differ. gain 0 makes it an OPEN loop (no effector); the plant then relaxes
    to `ambient`, proving the closed loop, not the physics, is what holds the set-point."""

    def __init__(self, gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, sensor_noise=0.0, seed=0):
        self.gain = gain  # proportional feedback gain (0 == open loop, no effector)
        self.leak = leak  # passive relaxation of the variable toward `ambient`
        self.ambient = ambient  # value the plant drifts to with no effector (loss sink)
        self.sensor_noise = sensor_noise  # sd of gaussian sensor noise on the sensed value
        self.rng = random.Random(seed)

    def effector(self, sensed, set_point):
        """Antagonistic corrective drive. error = set_point - sensed. A positive error (variable too
        LOW) yields a positive (raising) effector; a negative error (too HIGH) yields a negative
        (lowering) effector -- one signed drive standing in for two opposed effectors. gain 0 ==
        open loop -> zero drive regardless of error."""
        return self.gain * (set_point - sensed)

    def step(self, x, set_point, disturbance=0.0):
        """Advance the regulated variable one step under negative feedback. Returns (x_next, error)
        where error is set_point - sensed at this step. `disturbance` is an additive external shock."""
        sensed = x + (self.rng.gauss(0.0, self.sensor_noise) if self.sensor_noise else 0.0)
        drive = self.effector(sensed, set_point)
        x_next = x + self.leak * (self.ambient - x) + drive + disturbance
        return x_next, set_point - sensed

    def run(self, x0, set_point, steps, disturb=None):
        """Run the loop for `steps`. `set_point` may be a float (constant reference) or a callable
        set_point(t) (an SCN-paced moving reference). `disturb` may be a callable disturb(t)->additive
        shock (None == none). Returns the trace of the regulated variable (length steps + 1)."""
        x = x0
        trace = [x]
        for t in range(steps):
            sp = set_point(t) if callable(set_point) else set_point
            d = disturb(t) if disturb else 0.0
            x, _ = self.step(x, sp, d)
            trace.append(x)
        return trace


def main():
    width = 74
    print("=" * width)
    print("HYPOTHALAMIC SET-POINTS -- negative-feedback homeostats, SCN-paced reference")
    print("control FUNCTION only -- not thermoreceptors, ADH, or leptin/ghrelin")
    print("=" * width)

    # ---- SCN pacing of the temperature set-point (real CircadianClock phase) -----
    steps = 300
    phases = scn_phase_trace(steps, seed=0)  # free-running SCN phase at each step
    live_sp = [temp_setpoint(p) for p in phases]  # phase advances -> set-point oscillates
    frozen = CircadianClock(seed=0)  # never stepped -> phase held constant
    frozen_sp = [temp_setpoint(frozen.phase) for _ in range(steps)]
    sp_day = temp_setpoint(DAY_PHASE)  # midday (active) -> warmest
    sp_night = temp_setpoint(NIGHT_PHASE)  # midnight (rest) -> coolest

    print(
        f"set-point mean {SET_POINT_TEMP:.1f} C, circadian half-swing +/-{TEMP_AMP:.1f} C, "
        f"ambient {AMBIENT_TEMP:.1f} C"
    )
    print(
        f"SCN-paced set-point:  subjective DAY (phase {DAY_PHASE:.2f}) = {sp_day:.3f} C   "
        f"subjective NIGHT (phase {NIGHT_PHASE:.2f}) = {sp_night:.3f} C"
    )
    print(
        f"  clock ADVANCING -> set-point swings {min(live_sp):.3f}..{max(live_sp):.3f} C   "
        f"clock FROZEN -> flat {frozen_sp[0]:.3f} C"
    )

    # ---- the homeostat TRACKS the moving SCN-paced set-point across the day ------
    thermo = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, seed=0)
    track = thermo.run(x0=live_sp[0], set_point=lambda t: live_sp[t], steps=steps)
    tail = range(steps // 2, steps)  # after the warm-up transient
    track_err = max(abs(track[t] - live_sp[t]) for t in tail)
    print(
        f"thermostat tracking the paced set-point: max tail error {track_err:.3f} C "
        "(follows the SCN rhythm)"
    )

    # ---- (1) feedback does the work: closed loop converges, open loop does not ----
    sp = sp_day
    closed = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, seed=0)
    openl = Homeostat(gain=0.0, leak=0.02, ambient=AMBIENT_TEMP, seed=0)
    t_closed = closed.run(x0=30.0, set_point=sp, steps=80)
    t_open = openl.run(x0=30.0, set_point=sp, steps=80)
    err_closed = abs(sp - t_closed[-1])
    err_open = abs(sp - t_open[-1])
    print("-" * width)
    print(
        f"(1) FEEDBACK: closed-loop final error {err_closed:.3f} C  vs  "
        f"open-loop final error {err_open:.3f} C (relaxed to ambient)"
    )

    # ---- (2) disturbance rejection: a cold shock is driven back to the set-point --
    shock_t, shock = 40, 5.0
    dist = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, seed=0)
    t_dist = dist.run(
        x0=sp, set_point=sp, steps=100, disturb=lambda t: -shock if t == shock_t else 0.0
    )
    err_after = abs(sp - t_dist[shock_t + 1])
    err_recovered = abs(sp - t_dist[-1])
    print(
        f"(2) DISTURBANCE: -{shock:.0f} C cold shock -> error {err_after:.3f} C, "
        f"recovers to {err_recovered:.3f} C"
    )

    # ---- (4) bidirectional: too-cold and too-hot starts both corrected -----------
    cold = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, seed=0)
    hot = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, seed=0)
    t_cold = cold.run(x0=sp - 4.0, set_point=sp, steps=80)
    t_hot = hot.run(x0=sp + 4.0, set_point=sp, steps=80)
    err_cold = abs(sp - t_cold[-1])
    err_hot = abs(sp - t_hot[-1])
    drive_cold = cold.effector(sp - 4.0, sp)  # too cold -> raising effector (> 0)
    drive_hot = hot.effector(sp + 4.0, sp)  # too hot  -> lowering effector (< 0)
    print(
        f"(4) BIDIRECTIONAL: cold start final {err_cold:.3f} C (effector {drive_cold:+.2f}), "
        f"hot start final {err_hot:.3f} C (effector {drive_hot:+.2f})"
    )

    # ---- a second regulated variable: osmolarity (thirst / ADH) ------------------
    osmo_closed = Homeostat(gain=0.9, leak=0.03, ambient=AMBIENT_OSMO, seed=0)
    osmo_open = Homeostat(gain=0.0, leak=0.03, ambient=AMBIENT_OSMO, seed=0)
    o_closed = osmo_closed.run(
        x0=SET_POINT_OSMO,
        set_point=SET_POINT_OSMO,
        steps=120,
        disturb=lambda t: 10.0 if t == 30 else 0.0,
    )  # a salt load
    o_open = osmo_open.run(x0=SET_POINT_OSMO, set_point=SET_POINT_OSMO, steps=120)
    err_osmo_closed = abs(SET_POINT_OSMO - o_closed[-1])
    err_osmo_open = abs(SET_POINT_OSMO - o_open[-1])
    print("-" * width)
    print(
        f"OSMOLARITY (thirst/ADH): set-point {SET_POINT_OSMO:.0f} mOsm/kg  ->  "
        f"closed-loop error {err_osmo_closed:.3f}  vs  open-loop error {err_osmo_open:.3f}"
    )
    print("=" * width)

    # ---- un-riggable self-checks (deterministic asserts) ------------------------
    # (1) FEEDBACK does the work: closed converges to the set-point, open relaxes to ambient.
    assert err_closed < 0.5, f"closed loop must converge to the set-point (err {err_closed:.3f})"
    assert err_open > 5.0, (
        f"open loop must relax to ambient, far off the set-point (err {err_open:.3f})"
    )
    assert err_open > 10.0 * err_closed, "feedback, not the plant, must be what holds the set-point"
    # (2) DISTURBANCE REJECTION: the post-shock error shrinks back toward the set-point.
    assert err_after > 3.0, (
        f"a -{shock:.0f} C cold shock must knock the variable far off ({err_after:.3f})"
    )
    assert err_recovered < 0.5, (
        f"the loop must drive the shock back to the set-point ({err_recovered:.3f})"
    )
    assert err_recovered < 0.25 * err_after, "the post-disturbance error must shrink over time"
    # (3) SCN PACES the set-point: night != day, and a frozen clock gives a flat set-point.
    assert sp_night != sp_day, (
        "the SCN-paced set-point must differ between subjective night and day"
    )
    assert sp_night < sp_day, (
        "core temperature must run cooler at subjective night than subjective day"
    )
    assert max(live_sp) - min(live_sp) > 0.2, "an advancing clock must make the set-point oscillate"
    assert max(frozen_sp) - min(frozen_sp) == 0.0, "a frozen clock must give a flat set-point"
    assert track_err < 0.5, f"the homeostat must track the SCN-paced set-point ({track_err:.3f})"
    # (4) BIDIRECTIONAL: both a too-cold and a too-hot start converge via opposite-sign effectors.
    assert err_cold < 0.5 and err_hot < 0.5, (
        "both a cold and a hot start must be corrected to the set-point"
    )
    assert drive_cold > 0.0 > drive_hot, (
        "the effectors must be antagonistic (raise if cold, lower if hot)"
    )
    # osmolarity: the same generic homeostat regulates a second variable (closed converges, open drifts).
    assert err_osmo_closed < 2.0, (
        f"osmolarity must converge to its set-point ({err_osmo_closed:.3f})"
    )
    assert err_osmo_open > 10.0, f"open-loop osmolarity must drift to ambient ({err_osmo_open:.3f})"
    # Determinism: same seed -> identical traces; a different seed -> a different trace (seed matters).
    a = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, sensor_noise=0.05, seed=7)
    b = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, sensor_noise=0.05, seed=7)
    c = Homeostat(gain=1.2, leak=0.02, ambient=AMBIENT_TEMP, sensor_noise=0.05, seed=8)
    ta = a.run(x0=30.0, set_point=sp, steps=60)
    tb = b.run(x0=30.0, set_point=sp, steps=60)
    tc = c.run(x0=30.0, set_point=sp, steps=60)
    assert ta == tb, "same seed must reproduce an identical trace"
    assert tc != ta, "a different seed must produce a different trace (the rng is really seeded)"

    print(
        f"self-check OK: closed loop -> {err_closed:.2f} C vs open loop -> {err_open:.2f} C (feedback "
        f"regulates); cold shock {err_after:.2f} -> {err_recovered:.2f} C (rejected);"
    )
    print(
        f"  SCN set-point night {sp_night:.2f} < day {sp_day:.2f} C, frozen clock flat; cold+hot both "
        f"-> set-point via antagonistic effectors; osmolarity regulates too; deterministic"
    )


if __name__ == "__main__":
    main()
