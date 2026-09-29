"""
Suprachiasmatic nucleus (SCN) -- the master circadian clock as a phase oscillator.

The SCN is the body's master clock: a small population of coupled neurons that keeps time
even with no external cue. This module reduces that network to a single PHASE OSCILLATOR that
carries the one thing the SCN exports downstream -- a phase of the subjective day in [0, 1) --
and drives three functions from it:

  FREE-RUNNING RHYTHM -- the phase advances by an intrinsic frequency f0 = 1 / tau every step
      (one "step" == one abstract hour). With no light the clock completes a full cycle every
      tau steps: the rhythm is ENDOGENOUS, generated inside the oscillator, not read off the
      environment. tau is deliberately NOT 24 (here 24.2, the measured human free-run) so that
      the endogenous period is visibly distinct from the 24-unit solar day.

  LIGHT ENTRAINMENT via a phase-response curve -- a light zeitgeber (an LD light/dark cycle)
      pulls the phase toward subjective day (a restoring PRC): when the clock is running behind
      the light it ADVANCES, when ahead it DELAYS, and in the middle of subjective day the pull
      is ~0 (the PRC "dead zone"). This is exactly "light near subjective dawn/dusk advances or
      delays the phase." Over a cycle the pull cancels the intrinsic tau-vs-Tz frequency error,
      so the measured period LOCKS to the zeitgeber period Tz -- but only inside an entrainment
      range set by the coupling strength (a classic Arnold tongue).

  DIURNAL OUTPUTS -- a rest/active gate (subjective night == rest, the phase that GATES memory
      consolidation / replay) and a cortisol-like baseline scalar that peaks at subjective dawn
      (phase 0) and troughs at dusk. Both are pure functions of the current phase.

Fair control (un-riggable -- the intrinsic advance f0 is IDENTICAL in every condition; only the
light schedule differs, so any period change must come from the light, and the free-run proves
the rhythm is not merely stimulus-driven):
  (a) CONSTANT DARKNESS: no zeitgeber. The clock free-runs -- measured period ~= tau (endogenous).
  (b) ENTRAINING LD cycle (Tz near tau): the measured period LOCKS to Tz and therefore DIFFERS
      from the constant-darkness period. The light, not the intrinsic clock, now sets the period.
  (c) FAR-OUT-OF-RANGE LD cycle (Tz far from tau): the pull cannot cover the frequency gap, so
      the clock IGNORES it and free-runs back at ~= tau -- entrainment is selective, not automatic.
  (d) JET LAG: advance the light schedule by several hours and the locked phase re-aligns to the
      new schedule over several cycles (the transient a real circadian system shows after travel).

Honest framing: a reduced single-phase-oscillator abstraction of the SCN's coupled-neuron network.
It models the TIMING function -- free-run, entrainment range, phase resetting -- not the molecular
transcription-translation loop, and nothing here is awake, asleep, or aware. Additive/opt-in.
Credit: Pittendrigh & Daan 1976 (free-run + PRC entrainment); Czeisler et al. 1999 (human tau ~24.2).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sleep.circadian_clock
"""

import math
import random

# ---- oscillator constants (hand-tuned; see the self-check anchors) -----------
INTRINSIC = 24.2  # free-running period tau in abstract hour-steps (human free-run ~24.2h)
COUPLING = 0.012  # light PRC amplitude; sets the entrainment (Arnold-tongue) half-width
DAY_TARGET = 0.25  # phase the light pulls toward: the middle of subjective day (active phase)
MOL_NOISE = 0.0003  # tiny per-step molecular jitter (deterministic via the per-instance rng)
REST_ONSET = 0.5  # subjective night starts here: phase in [0.5, 1.0) == rest / consolidation
SOLAR_DAY = 24  # the environmental light-cycle period used as the entraining zeitgeber


def circular_diff(target, phase):
    """Signed shortest phase distance target - phase, wrapped to (-0.5, 0.5]. Positive means the
    target is AHEAD of the clock (the clock is running behind and should advance)."""
    d = (target - phase) % 1.0
    return d - 1.0 if d > 0.5 else d


def light_cycle(period, duty=0.5, offset=0):
    """A square-wave LD zeitgeber: returns a schedule(t) that is light-ON (1.0) for the first
    `duty` fraction of each `period`-step cycle and dark (0.0) otherwise. `offset` shifts the
    lights-on onset in time -- advancing it simulates eastward jet-lag travel."""
    on = duty * period
    return lambda t: 1.0 if ((t - offset) % period) < on else 0.0


class CircadianClock:
    """SCN master clock: a phase oscillator that free-runs at tau and entrains to a light zeitgeber.

    Phase convention: phase in [0, 1). 0.0 == subjective dawn, 0.5 == subjective dusk. Subjective
    day [0, 0.5) is the active phase; subjective night [0.5, 1.0) is the rest / consolidation phase.
    """

    def __init__(
        self, intrinsic=INTRINSIC, coupling=COUPLING, target=DAY_TARGET, noise=MOL_NOISE, seed=0
    ):
        self.tau = intrinsic
        self.f0 = 1.0 / intrinsic  # intrinsic angular frequency (cycles per step)
        self.k = coupling
        self.target = target
        self.noise = noise
        self.rng = random.Random(seed)
        self.phase = self.rng.random()  # seed-dependent starting phase (deterministic)

    # ---- dynamics -----------------------------------------------------------
    def prc(self, light):
        """Light phase-response curve: a restoring pull toward subjective day, scaled by light
        intensity. Behind the light (target ahead) -> advance; ahead of it -> delay; dead zone in
        mid subjective day where target ~= phase. Zero in darkness. Returns a delta-phase."""
        if light <= 0.0:
            return 0.0
        return self.k * light * math.sin(2.0 * math.pi * circular_diff(self.target, self.phase))

    def step(self, light=0.0):
        """Advance one hour-step under `light` (0/1 or intensity). The phase moves by the intrinsic
        frequency plus the light PRC plus tiny molecular jitter. Returns True on a DAWN TICK -- the
        SCN's once-per-cycle population pulse, emitted when the phase wraps past subjective dawn."""
        delta = self.f0 + self.prc(light) + self.rng.gauss(0.0, self.noise)
        prev = self.phase
        self.phase = (self.phase + delta) % 1.0
        return self.phase < prev  # wrapped through dawn == one subjective day elapsed

    def run(self, steps, zeitgeber=None):
        """Advance `steps` steps under an optional zeitgeber schedule(t)->intensity (None == constant
        darkness). Returns the list of step indices at which a dawn tick fired."""
        ticks = []
        for t in range(steps):
            light = 0.0 if zeitgeber is None else zeitgeber(t)
            if self.step(light):
                ticks.append(t)
        return ticks

    # ---- diurnal outputs (pure functions of the phase) ----------------------
    def is_rest(self, phase=None):
        """Rest/active gate: True during subjective night [0.5, 1.0). Subjective night is the rest
        phase that gates offline memory consolidation / replay downstream."""
        p = self.phase if phase is None else phase
        return p >= REST_ONSET

    def consolidation_gate(self, phase=None):
        """Memory-consolidation gate: open during the subjective-night rest phase (same as is_rest).
        Downstream replay/consolidation is permitted only while this gate is open."""
        return self.is_rest(phase)

    def baseline(self, phase=None):
        """Cortisol-like diurnal baseline in [0, 1]: peaks at subjective dawn (phase 0) and troughs
        at subjective dusk (phase 0.5). A slow scalar drive that anticipates the active phase."""
        p = self.phase if phase is None else phase
        return 0.5 * (1.0 + math.cos(2.0 * math.pi * p))


def measure_period(clock, steps, zeitgeber=None, warm=1000):
    """Run `clock` for `steps` steps under `zeitgeber` and return the mean inter-dawn interval
    (its measured period) over the ticks AFTER the `warm`-step transient. Integer per-cycle
    intervals average to the true period over many cycles."""
    ticks = [t for t in clock.run(steps, zeitgeber) if t >= warm]
    intervals = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
    return sum(intervals) / len(intervals)


def _jetlag(seed, period, shift, warm=2000, recover_cycles=18):
    """Entrain a fresh clock to an LD cycle, then advance the schedule by `shift` steps and track
    the clock's phase at each new lights-on onset. Returns (locked_phase, err_after_shift,
    err_after_recovery) -- the alignment error right after the shift and after re-entraining."""
    clock = CircadianClock(seed=seed)
    base = light_cycle(period, offset=0)
    for t in range(warm):  # settle into the pre-travel lock
        clock.step(base(t))
    locked = []  # steady phase sampled at lights-on
    t = warm
    for _ in range(6 * period):
        onset = ((t) % period) == 0
        clock.step(base(t))
        if onset:
            locked.append(clock.phase)
        t += 1
    p_lock = sum(locked) / len(locked)

    shifted = light_cycle(period, offset=shift)  # the travelled (advanced) schedule
    errs = []
    for _ in range(recover_cycles):
        onset_phase = None
        for _ in range(period):
            onset = ((t - shift) % period) == 0
            clock.step(shifted(t))
            if onset:
                onset_phase = clock.phase
            t += 1
        errs.append(abs(circular_diff(p_lock, onset_phase)))
    return p_lock, errs[0], errs[-1]


def main():
    width = 74
    print("=" * width)
    print("SCN CIRCADIAN CLOCK -- a master phase oscillator (free-run + light entrainment)")
    print("timing FUNCTION only -- not sleep, not wakefulness, not awareness")
    print("=" * width)
    print(
        f"intrinsic period tau = {INTRINSIC} hour-steps (free-run), zeitgeber Tz = {SOLAR_DAY} (solar day)"
    )
    print(f"coupling k = {COUPLING}, light pulls phase toward subjective day (target {DAY_TARGET})")

    # ---- diurnal profile across one subjective day (pure functions of phase) ----
    ref = CircadianClock(seed=0)
    print("-" * width)
    print("subjective day  |  phase  gate      cortisol-like baseline")
    marks = [(0.0, "dawn"), (0.25, "midday"), (0.5, "dusk"), (0.75, "midnight")]
    for p, label in marks:
        gate = "REST " if ref.is_rest(p) else "ACTIVE"
        print(f"  {label:<12}  |  {p:.2f}   {gate}   baseline = {ref.baseline(p):.3f}")

    # ---- fair control: constant darkness vs entraining vs far-out-of-range ----
    dd = measure_period(CircadianClock(seed=1), 3600, None)
    ld = measure_period(CircadianClock(seed=1), 3600, light_cycle(SOLAR_DAY))
    far_period = 18
    far = measure_period(CircadianClock(seed=1), 3600, light_cycle(far_period))
    print("-" * width)
    print("FAIR CONTROL -- same intrinsic clock, only the light schedule differs:")
    print(
        f"  (a) constant darkness      : measured period = {dd:.3f}  (~= tau {INTRINSIC}, ENDOGENOUS free-run)"
    )
    print(
        f"  (b) entraining LD (Tz={SOLAR_DAY})   : measured period = {ld:.3f}  (LOCKED to Tz, differs from DD)"
    )
    print(
        f"  (c) far LD (Tz={far_period}, no lock) : measured period = {far:.3f}  (IGNORES it, free-runs at ~= tau)"
    )

    # ---- jet lag: advance the light schedule and watch the phase re-align ----
    p_lock, err0, err_end = _jetlag(seed=3, period=SOLAR_DAY, shift=8)
    print("-" * width)
    print(f"JET LAG -- advance the lights-on schedule by 8 hour-steps (locked phase {p_lock:.3f}):")
    print(f"  phase-alignment error right after the shift : {err0:.3f}")
    print(
        f"  phase-alignment error after re-entraining   : {err_end:.3f}  (re-aligned to the new day)"
    )
    print("=" * width)

    # ---- un-riggable anchors (deterministic asserts) ------------------------
    # (1) constant darkness: the rhythm persists at ~= the intrinsic period (endogenous free-run).
    assert abs(dd - INTRINSIC) < 0.15, (
        f"constant-darkness period must ~= intrinsic tau ({dd:.3f} vs {INTRINSIC})"
    )
    # (2) an entraining zeitgeber LOCKS the period to Tz, and that differs from the free-run.
    assert abs(ld - SOLAR_DAY) < 0.10, f"entrained period must lock to Tz ({ld:.3f} vs {SOLAR_DAY})"
    assert abs(dd - ld) > 0.12, (
        f"entrained period must DIFFER from the free-run ({ld:.3f} vs DD {dd:.3f})"
    )
    # (3) a far-out-of-range zeitgeber does NOT entrain -- it free-runs back at the intrinsic period.
    assert abs(far - far_period) > 3.0, (
        f"far Tz must NOT capture the clock ({far:.3f} vs Tz {far_period})"
    )
    assert abs(far - dd) < 0.10, (
        f"an un-entraining clock must free-run at ~= tau ({far:.3f} vs DD {dd:.3f})"
    )
    # (4) diurnal outputs: baseline peaks at dawn, troughs at dusk; night is the rest/gate phase.
    assert ref.baseline(0.0) > 0.99 and ref.baseline(0.5) < 0.01, (
        "baseline must peak at dawn, trough at dusk"
    )
    assert ref.baseline(0.0) > ref.baseline(0.25) > ref.baseline(0.5), (
        "baseline must fall from dawn to dusk"
    )
    assert not ref.is_rest(0.25) and ref.is_rest(0.75), (
        "subjective day == active, subjective night == rest"
    )
    assert ref.consolidation_gate(0.75) and not ref.consolidation_gate(0.25), (
        "consolidation gates to the rest phase"
    )
    # (5) jet lag: the locked phase re-aligns to the shifted schedule over several cycles.
    assert err0 > 0.25, (
        f"an 8-step schedule shift must first knock the clock out of phase ({err0:.3f})"
    )
    assert err_end < 0.25 * err0 and err_end < 0.05, (
        f"the clock must re-entrain to the new schedule ({err_end:.3f})"
    )

    # (6) determinism: two clocks with the same seed evolve identically.
    a = CircadianClock(seed=5)
    b = CircadianClock(seed=5)
    zeit = light_cycle(SOLAR_DAY)
    pa = [(a.step(zeit(t)), round(a.phase, 12)) for t in range(600)]
    pb = [(b.step(zeit(t)), round(b.phase, 12)) for t in range(600)]
    assert pa == pb, "same seed must produce an identical trajectory"

    print(
        f"self-check OK: constant-dark free-run {dd:.2f} ~= tau {INTRINSIC} (endogenous); "
        f"an LD cycle locked the period to Tz {ld:.2f} (differs from free-run); a far Tz={far_period} did NOT "
        f"entrain ({far:.2f} ~= tau); jet-lag re-aligned {err0:.2f} -> {err_end:.2f}; deterministic"
    )


if __name__ == "__main__":
    main()
