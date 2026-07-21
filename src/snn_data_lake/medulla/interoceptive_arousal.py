"""
Interoceptive arousal -- visceral homeostatic error mapped to a bounded caution / arousal gain.

The medulla and hypothalamus hold the body's regulated variables (blood CO2, arterial pressure,
core temperature) on negative-feedback set-points. When a metabolic or thermal CHALLENGE loads
those loops, the regulated variables drift off their set-points; the size of that residual error
IS the body's felt "how far from homeostasis am I" signal. The nucleus of the solitary tract (NTS)
integrates that visceral error and drives the ascending arousal system (locus coeruleus / LC
noradrenaline): a bigger homeostatic debt -> more arousal, more caution. This module models that
ascending NTS -> LC map at the FUNCTIONAL level -- it turns visceral homeostatic error into a
bounded caution scalar and an arousal gain, nothing else.

Mechanism (rate-level, deterministic):
  1. Compose the REAL homeostats. A metabolic/thermal `load` is injected into (a) the medullary
     chemoreflex (MedullaVitalNuclei.run_chemoreflex: extra metabolic CO2 production) and (b) a
     hypothalamic temperature Homeostat (an added thermal disturbance the loop must reject). This
     module does NOT reinvent a homeostatic model -- it instantiates the two servos and reads their
     residual set-point error after they respond.
  2. Aggregate error. stress = a normalized, saturating sum of the two residual set-point errors,
     measured as the EXCESS over the resting (load=0) regulated residual, so homeostasis reads zero.
  3. Ascending map. caution = a saturating map of stress in [0, 1] (what the arm loop ADDS to its
     safe-action bias); gain = an LC-like arousal gain in [1, cap] that rises with stress.

Fair control (un-riggable -- the bounded caution must come from the REAL regulation, not a curve):
  - NO SPURIOUS AROUSAL: at load 0 (homeostasis) the composed loops sit on their set-points, so
    stress ~ 0, caution ~ 0, gain ~ 1. Arousal is not manufactured from nothing.
  - GRADED: rising load raises the residual homeostatic error, so stress, caution and gain all rise
    monotonically. The map is graded in the challenge, not a threshold flip.
  - FEEDBACK BOUNDS THE STRESS: for the SAME load, running the internal homeostats with feedback
    ENABLED yields far lower stress than with feedback DISABLED (chemoreflex drive frozen, thermal
    gain 0). This proves the bounded caution comes from the closed-loop regulation -- exactly the
    fair control that vital_nuclei and set_points use for their own loops -- not from a hardcoded
    saturating curve. If the physics alone bounded the error, the open loop would too; it does not.
  - PURE MODULATOR: every caution stays in [0, 1] and every gain in [1, cap]; the module returns
    only these scalars. It imports nothing from brainstem/ and touches no actuator.

Honest framing: a reduced interoceptive-arousal map -- visceral homeostatic error (medullary
chemoreflex + a hypothalamic thermal set-point) -> a bounded caution / gain modulator, the
ascending NTS -> LC-style arousal drive. It MODULATES (biases caution, scales gain); it never
selects or commands an action and touches no actuator. "stress" is a normalized number, not a
feeling. The main loop consumes `caution` CAUTION-ONLY (it biases action selection toward the safe
action); this module computes only bounded scalars. Additive/opt-in; reuses vital_nuclei.py and
set_points.py.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.medulla.interoceptive_arousal
"""

import math

from snn_data_lake.hypothalamus.set_points import SET_POINT_TEMP, Homeostat
from snn_data_lake.medulla.vital_nuclei import MedullaVitalNuclei


class InteroceptiveArousal:
    """Ascending NTS -> LC map: visceral homeostatic error -> bounded caution + arousal gain.

    sense(load) composes a medullary chemoreflex and a hypothalamic thermal Homeostat, injects the
    load into both, reads their residual set-point error, and returns {"stress", "caution", "gain"}.
    At load 0 the loops sit on their set-points (stress ~ 0, caution ~ 0, gain ~ 1); rising load
    raises all three monotonically; feedback ENABLED bounds the stress far below feedback DISABLED.
    Caution is always in [0, 1] and gain in [1, cap]. It commands no movement and touches no
    actuator -- the arm loop ADDS `caution` to its safe-action bias (caution-only)."""

    def __init__(
        self,
        seed=0,
        # thermal homeostat (hypothalamic temperature set-point)
        thermo_gain=1.2,  # closed-loop proportional gain (0 == open loop, no effector)
        thermo_leak=0.02,  # passive relaxation of temperature toward ambient
        thermal_scale=1.0,  # C of per-step thermal disturbance per unit load
        thermo_steps=40,  # small settle budget for the thermal loop (keep sense() cheap)
        # medullary chemoreflex (CO2 set-point)
        base_prod=0.05,  # baseline metabolic CO2 production (load 0 -> steady CO2 == set-point)
        chemo_scale=0.02,  # extra CO2 production per unit load
        chemo_steps=60,  # small settle budget for the chemoreflex (keep sense() cheap)
        # aggregation + ascending map
        thermo_weight=1.0,  # weight of the thermal residual in the aggregate error
        chemo_weight=1.0,  # weight of the CO2 residual in the aggregate error
        stress_scale=5.0,  # aggregate error that maps to full stress (normalizer)
        caution_k=2.5,  # saturation sharpness of the stress -> caution map
        gain_cap=3.0,  # maximum LC-like arousal gain (gain is bounded to [1, gain_cap])
    ):
        self.seed = seed
        self.thermo_gain = thermo_gain
        self.thermo_leak = thermo_leak
        self.thermal_scale = thermal_scale
        self.thermo_steps = thermo_steps
        self.base_prod = base_prod
        self.chemo_scale = chemo_scale
        self.chemo_steps = chemo_steps
        self.thermo_weight = thermo_weight
        self.chemo_weight = chemo_weight
        self.stress_scale = stress_scale
        self.caution_k = caution_k
        self.gain_cap = gain_cap
        # resting (load 0) regulated residual, so homeostasis reads exactly zero stress. Computed
        # once from the REAL loops (noise-free) for both feedback settings; cheap.
        self._base_closed = self._raw_error(0.0, feedback=True, noise=0.0)
        self._base_open = self._raw_error(0.0, feedback=False, noise=0.0)

    # ---- residual set-point error from the composed homeostats ----------------
    def _thermo_error(self, load, feedback, noise):
        """Residual |set-point - temperature| after a hypothalamic Homeostat rejects a load-sized
        thermal disturbance. feedback False -> gain 0 (open loop): the disturbance is not rejected.
        ambient == the set-point, so at load 0 the loop sits exactly on target (zero residual)."""
        gain = self.thermo_gain if feedback else 0.0
        homeostat = Homeostat(
            gain=gain,
            leak=self.thermo_leak,
            ambient=SET_POINT_TEMP,
            sensor_noise=noise,
            seed=self.seed,
        )
        disturbance = self.thermal_scale * load
        trace = homeostat.run(
            x0=SET_POINT_TEMP,
            set_point=SET_POINT_TEMP,
            steps=self.thermo_steps,
            disturb=lambda _t: disturbance,
        )
        return abs(SET_POINT_TEMP - trace[-1])

    def _chemo_error(self, load, feedback, noise):
        """Residual |CO2 set-point - steady CO2| after the medullary chemoreflex responds to a
        load-sized extra metabolic CO2 production. feedback False -> respiratory drive frozen at
        baseline (open loop): clearance cannot track the load and CO2 drifts up. base_prod is tuned
        so at load 0 clearance == production and CO2 sits on the set-point (zero residual)."""
        core = MedullaVitalNuclei(seed=self.seed)
        production = self.base_prod + self.chemo_scale * load
        res = core.run_chemoreflex(
            steps=self.chemo_steps,
            co2_production=production,
            co2_init=core.co2_set_point,
            feedback=feedback,
            noise=noise,
        )
        return abs(core.steady_co2(res) - core.co2_set_point)

    def _raw_error(self, load, feedback, noise):
        """Weighted aggregate residual set-point error across the two composed homeostats."""
        thermo = self._thermo_error(load, feedback, noise)
        chemo = self._chemo_error(load, feedback, noise)
        return self.thermo_weight * thermo + self.chemo_weight * chemo

    # ---- ascending NTS -> LC map ---------------------------------------------
    def sense(self, load=0.0, feedback=True, noise=0.0):
        """Map a metabolic/thermal `load` (>= 0, 0 == homeostasis) to a bounded caution + arousal
        gain by composing the real homeostats and reading their residual set-point error.

        Returns at least {"stress", "caution", "gain"}:
          stress  in [0, 1] -- normalized EXCESS homeostatic error over the resting residual.
          caution in [0, 1] -- saturating map of stress (the arm loop ADDS this to its safe bias).
          gain    in [1, cap] -- LC-noradrenaline-like arousal gain that rises with stress.

        feedback False runs the internal loops open (chemo drive frozen, thermal gain 0) for the
        fair control; `noise` seeds a small rng-driven jitter in both loops (determinism check)."""
        load = max(0.0, load)
        base = self._base_closed if feedback else self._base_open
        raw = self._raw_error(load, feedback, noise)
        excess = max(0.0, raw - base)
        stress = min(1.0, excess / self.stress_scale)
        caution = 1.0 - math.exp(-self.caution_k * stress)  # saturating, 0 at stress 0, < 1
        gain = 1.0 + (self.gain_cap - 1.0) * caution  # in [1, gain_cap]
        return {"stress": stress, "caution": caution, "gain": gain, "load": load}


def main():
    ia = InteroceptiveArousal(seed=0)

    # (1) homeostasis: load 0 -> no spurious arousal.
    rest = ia.sense(load=0.0)

    # (2) graded: a rising load raises stress, caution and gain together.
    sweep = [0.0, 0.4, 0.8, 1.2, 1.6]
    graded = [ia.sense(load=x) for x in sweep]
    stresses = [r["stress"] for r in graded]
    cautions = [r["caution"] for r in graded]
    gains = [r["gain"] for r in graded]

    # (3) feedback bounds the stress: same load, real loops closed vs open.
    probe = 1.0
    closed = ia.sense(load=probe, feedback=True)
    openl = ia.sense(load=probe, feedback=False)

    width = 72
    print("=" * width)
    print("INTEROCEPTIVE AROUSAL -- visceral homeostatic error -> caution / arousal gain")
    print("ascending NTS -> LC map; caution-only, commands no movement, no actuator")
    print("=" * width)
    print(
        f"composed loops: chemoreflex (CO2 set-point {MedullaVitalNuclei().co2_set_point:.0f}) + "
        f"thermal Homeostat (set-point {SET_POINT_TEMP:.0f} C)"
    )
    print(
        f"gain cap: {ia.gain_cap:.1f} | thermo settle steps: {ia.thermo_steps} | "
        f"chemo settle steps: {ia.chemo_steps}"
    )
    print("-" * width)
    print(
        f"(1) homeostasis (load 0.0): stress {rest['stress']:.3f}  caution {rest['caution']:.3f}  "
        f"gain {rest['gain']:.3f}  (no spurious arousal)"
    )
    print("-" * width)
    print("(2) graded response (rising metabolic/thermal load):")
    for x, s, c, g in zip(sweep, stresses, cautions, gains):
        print(f"    load {x:.1f} -> stress {s:.3f}  caution {c:.3f}  gain {g:.3f}")
    print("-" * width)
    print(f"(3) feedback bounds the stress (same load {probe:.1f}):")
    print(f"    feedback ON  -> stress {closed['stress']:.3f}  (real regulation bounds it)")
    print(f"    feedback OFF -> stress {openl['stress']:.3f}  (open loop drifts, high stress)")
    print("=" * width)

    # ---- un-riggable fair-control self-checks (deterministic asserts) --------
    # (1) NO SPURIOUS AROUSAL: homeostasis -> baseline.
    assert rest["stress"] < 0.02, f"load 0 must give ~0 stress (got {rest['stress']:.3f})"
    assert rest["caution"] < 0.02, f"load 0 must give ~0 caution (got {rest['caution']:.3f})"
    assert abs(rest["gain"] - 1.0) < 0.02, f"load 0 must give gain ~1 (got {rest['gain']:.3f})"
    # (2) GRADED: stress, caution and gain all rise strictly with load.
    assert all(b > a for a, b in zip(stresses, stresses[1:])), "stress must rise with load"
    assert all(b > a for a, b in zip(cautions, cautions[1:])), "caution must rise with load"
    assert all(b > a for a, b in zip(gains, gains[1:])), "gain must rise with load"
    # (3) FEEDBACK BOUNDS THE STRESS: closed-loop stress << open-loop stress at the same load.
    assert closed["stress"] < openl["stress"], "feedback must yield lower stress than open loop"
    assert closed["stress"] < 0.5 * openl["stress"], (
        "the bounded caution must come from the real closed-loop regulation, not a hardcoded curve"
    )
    # (4) PURE MODULATOR, BOUNDED: every caution in [0, 1], every gain in [1, cap]; no motor command.
    allr = graded + [rest, closed, openl]
    assert all(0.0 <= r["caution"] <= 1.0 for r in allr), "caution must stay in [0, 1]"
    assert all(1.0 <= r["gain"] <= ia.gain_cap for r in allr), "gain must stay in [1, cap]"
    # (5) DETERMINISM: same seed -> identical noisy sense() traces; a different seed -> different.
    a = InteroceptiveArousal(seed=7)
    b = InteroceptiveArousal(seed=7)
    c = InteroceptiveArousal(seed=8)
    trace_a = [a.sense(load=x, noise=0.05) for x in sweep]
    trace_b = [b.sense(load=x, noise=0.05) for x in sweep]
    trace_c = [c.sense(load=x, noise=0.05) for x in sweep]
    assert trace_a == trace_b, "same seed must reproduce an identical sense() trace"
    assert trace_a != trace_c, "a different seed must produce a different trace (the rng is seeded)"

    print(
        f"self-check OK: homeostasis stress {rest['stress']:.3f}/caution {rest['caution']:.3f}/"
        f"gain {rest['gain']:.3f}; stress {stresses[0]:.2f}->{stresses[-1]:.2f}, "
        f"gain {gains[0]:.2f}->{gains[-1]:.2f} rise with load;"
    )
    print(
        f"  feedback bounds stress {closed['stress']:.2f} vs open-loop {openl['stress']:.2f}; "
        f"caution in [0,1], gain in [1,{ia.gain_cap:.0f}]; deterministic"
    )


if __name__ == "__main__":
    main()
