"""
v0.52 -> genuinely-spiking rewrite: LIF-population PID controller.

The old version was mislabelled: it NEF-decoded the error to a scalar and then ran a plain
arithmetic PID (kp*e + ki*sum + kd*delta). Nothing spiked. This version makes the control
signal EMERGE from leaky integrate-and-fire spike dynamics -- the name is now honest.

Every term is a population of LIF neurons (membrane that leaks + integrates an input current,
crosses a threshold, emits a discrete SPIKE, soft-resets). The control command is read out of
spike COUNTS, not scalar algebra:

  Proportional : matched ON/OFF LIF populations, membrane reset each control step so the
                 firing RATE tracks the live |error|. net = (ON spikes - OFF spikes).
  Integral     : an ON/OFF pair of near-PERFECT integrators (leak ~ 1). A LIF neuron with no
                 leak IS a perfect integrator: it accumulates the error current and every
                 threshold crossing sheds exactly one quantum of charge. The CUMULATIVE net
                 spike count (ON - OFF, persisted across steps) literally equals the integral
                 of the error. Sign flips (overshoot) fire the OFF side and wind the count back
                 down -- spiking anti-windup.
  Derivative   : an adapting fast/slow OPPONENT LIF pair per polarity. Both track |error|; the
                 fast membrane (small time-constant) leads, the slow one (the adaptation trace)
                 lags. Their spike-rate DIFFERENCE is zero at steady state and positive on a
                 rising edge, i.e. it encodes d|error|/dt. ON/OFF gives it the sign.

The scalars left in the readout (kp/ki/kd and the per-population decode weights) are synaptic
gains -- the standard NEF "solve a decoder for the population's spike train" step. The values
themselves are carried by discrete spike events through real membranes.

Front-end: the error is still fed through the RBF population code (population_encoding) for
noise-robust transduction, but the PID itself is now spikes end-to-end.

Deps: math, random (numpy/torch allowed project-wide). Deterministic: random.Random(seed) per instance.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.spiking_pid   # self-check
"""

import random

from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.so101_arm import So101Arm
from snn_data_lake.cortex.population_encoding import PopulationEncoder

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "SpikingPID._calibrate": (
        "NOT_A_RULE",
        "measures this controller's gain against a reference response. A calibration measurement, not "
        "a rule that learns from experience",
    ),
}


class _LIFPopulation:
    """A population of leaky integrate-and-fire neurons sharing one input drive `d`.

    Genuine LIF dynamics, one micro-step per advance():
        v[i] <- v[i]*leak + gain[i]*d + bias[i]        (leak in (0,1]; 1.0 == perfect integrator)
        if v[i] >= theta[i]:  emit a spike, v[i] -= theta[i]   (soft reset keeps the remainder)
        elif v[i] < 0:        v[i] = 0                          (rectify: one-sided current)
    Per-neuron gain/bias/threshold spread gives the population diverse tuning curves, so the
    spike-count readout is smooth over a wide input range and robust to per-neuron noise.
    Membrane state PERSISTS between calls unless reset() is called.
    """

    def __init__(self, leak, gain, bias, theta):
        self.leak = leak
        self.gain, self.bias, self.theta = gain, bias, theta
        self.n = len(gain)
        self.v = [0.0] * self.n

    def reset(self):
        self.v = [0.0] * self.n

    def advance(self, d):
        """Integrate one micro-step at drive `d`; return the population spike count (0..n)."""
        spikes = 0
        v, gain, bias, theta, leak = self.v, self.gain, self.bias, self.theta, self.leak
        for i in range(self.n):
            vi = v[i] * leak + gain[i] * d + bias[i]
            if vi >= theta[i]:
                vi -= theta[i]
                spikes += 1
            elif vi < 0.0:
                vi = 0.0
            v[i] = vi
        return spikes

    def run(self, d, window):
        """Run `window` micro-steps at constant drive `d`; return total spikes emitted."""
        return sum(self.advance(d) for _ in range(window))


def _tuning(n, rng, gain_spread=0.0, bias_max=0.0, theta=1.0, theta_spread=0.0):
    """Draw diverse per-neuron LIF parameters (deterministic given rng)."""
    gain = [1.0 + gain_spread * (rng.random() - 0.5) for _ in range(n)]
    bias = [bias_max * rng.random() for _ in range(n)]
    th = [theta * (1.0 + theta_spread * (rng.random() - 0.5)) for _ in range(n)]
    return gain, bias, th


class SpikingPID:
    """LIF-population PID. step(error_deg) -> control command (delta degrees).

    The control signal emerges from spike dynamics: proportional = ON/OFF LIF firing rate,
    integral = cumulative net integrator spikes, derivative = fast/slow opponent spike-rate
    difference. Public API (class name, __init__ signature, step) is unchanged.
    """

    def __init__(self, kp=0.6, ki=0.05, kd=0.15, dt=1.0, err_max=90.0, n=15, seed=0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.err_max = err_max
        self.n = n
        self.window = 40  # LIF micro-steps integrated per control step
        self.rng = random.Random(seed)
        self.enc = PopulationEncoder(n=n, lo=-err_max, hi=err_max, max_spikes=20)

        # map |error| (deg) -> per-micro-step input current (kept below threshold so LIF stays linear)
        self._g = 1.0 / err_max

        # --- proportional: MATCHED ON/OFF populations (shared tuning => exact cancel at e=0) ---
        p_tune = _tuning(n, self.rng, gain_spread=0.6, bias_max=0.25, theta=0.6, theta_spread=0.4)
        self.p_on = _LIFPopulation(0.85, *p_tune)
        self.p_off = _LIFPopulation(0.85, *p_tune)

        # --- integral: near-perfect integrator ON/OFF pair (leak ~ 1, no bias), state PERSISTS ---
        i_tune = _tuning(n, self.rng, gain_spread=0.15, bias_max=0.0, theta=0.5, theta_spread=0.1)
        self.i_on = _LIFPopulation(0.9995, *i_tune)
        self.i_off = _LIFPopulation(0.9995, *i_tune)
        self.integral_spikes = 0  # cumulative net (ON-OFF) integrator spikes == the integral

        # --- derivative: fast/slow opponent (adapting) LIF pair per polarity, state PERSISTS ---
        d_tune = _tuning(n, self.rng, gain_spread=0.3, bias_max=0.0, theta=0.5, theta_spread=0.3)
        self.d_fast_on = _LIFPopulation(0.2, *d_tune)
        self.d_slow_on = _LIFPopulation(0.9, *d_tune)
        self.d_fast_off = _LIFPopulation(0.2, *d_tune)
        self.d_slow_off = _LIFPopulation(0.9, *d_tune)

        self._calibrate()

    # -- NEF decode weights: measure each population's spike response to a known reference,
    #    then read out spikes/weight so the estimate is an unbiased spiking version of the term.
    def _calibrate(self):
        ref, w, g = self.err_max, self.window, self._g

        self.p_on.reset()
        self._Kp = (self.p_on.run(g * ref, w) or 1) / ref  # P spikes per deg (per step)
        self.p_on.reset()

        self.i_on.reset()
        self._Ki = (self.i_on.run(g * ref, w) or 1) / ref  # integrator spikes per deg per step
        self.i_on.reset()

        # fast/slow steady-state rates, each normalized to |e|
        self.d_fast_on.reset()
        self.d_fast_on.run(g * ref, 6 * w)
        self._Kf = (self.d_fast_on.run(g * ref, w) or 1) / ref
        self.d_fast_on.reset()
        self.d_slow_on.reset()
        self.d_slow_on.run(g * ref, 6 * w)
        self._Ks = (self.d_slow_on.run(g * ref, w) or 1) / ref
        self.d_slow_on.reset()

        # derivative scale: response to a unit ramp (de = 1 deg/step) -> read D out as ~ de/dt
        self.d_fast_on.reset()
        self.d_slow_on.reset()
        d_ramp, e_r = 0.0, 0.0
        for _ in range(40):
            e_r += 1.0
            d_ramp = (
                self.d_fast_on.run(g * e_r, w) / self._Kf
                - self.d_slow_on.run(g * e_r, w) / self._Ks
            )
        self._d_scale = 1.0 / d_ramp if d_ramp else 1.0
        for pop in (self.d_fast_on, self.d_slow_on, self.d_fast_off, self.d_slow_off):
            pop.reset()

    def _encode_error(self, e):
        """NEF front-end: error -> population spikes -> decoded error (robust to per-neuron noise)."""
        e = max(-self.err_max + 1e-6, min(self.err_max - 1e-6, e))  # clamp into the coded range
        est = self.enc.decode(self.enc.encode(e))
        return e if est is None else est

    def step(self, error):
        e = self._encode_error(error)
        e_on, e_off = max(0.0, e), max(0.0, -e)
        w, g = self.window, self._g

        # proportional: reset each step so firing RATE (not history) tracks |error|
        self.p_on.reset()
        self.p_off.reset()
        p_net = (self.p_on.run(g * e_on, w) - self.p_off.run(g * e_off, w)) / self._Kp

        # integral: cumulative net integrator spikes == integral of the error (persists, self-winds)
        self.integral_spikes += self.i_on.run(g * e_on, w) - self.i_off.run(g * e_off, w)
        integral_est = self.integral_spikes / self._Ki

        # derivative: fast/slow opponent spike-rate difference == d(error)/dt (signed via ON/OFF)
        d_on = (
            self.d_fast_on.run(g * e_on, w) / self._Kf - self.d_slow_on.run(g * e_on, w) / self._Ks
        )
        d_off = (
            self.d_fast_off.run(g * e_off, w) / self._Kf
            - self.d_slow_off.run(g * e_off, w) / self._Ks
        )
        d_net = (d_on - d_off) * self._d_scale

        return self.kp * p_net + self.ki * integral_est + self.kd * d_net


#: The joint the controller drives. Named, not indexed: index 0 is shoulder_pan on this body,
#: a yaw joint, where the incumbent's joint A was the shoulder pitch.
_PID_JOINT = "shoulder_lift"
_PID_INDEX = next(j.qpos_index for j in arm_config.JOINTS if j.name == _PID_JOINT)


def drive_to(
    target_deg, steps=60, noise=0.0, spiking=True, seed=0, perturb_at=None, perturb_deg=25.0
):
    """Drive the So101Arm's `_PID_JOINT` to target_deg; return (error trace, final angle, breaches).
    If perturb_at is set, an external shove is injected at that step and the arm's
    trajectory-deviation comparator flags the off-path event -- the controller then
    drives the joint back. Controller + edge safety reflex working together.

    (Named `arm_sim joint A` until 2026-08-04. `brainstem/arm_sim.py` does not exist -- it was
    REPLACED by `so101_arm.py` under specs/004 -- and "joint A" predates the named-joint move that
    put `_PID_JOINT = "shoulder_lift"` below. A stale citation to a deleted module sends a reader
    looking for something that is not there; `citation_guard` does not catch prose like this.)"""
    rng = random.Random(seed)
    arm = So101Arm()
    # One simulated second per control step -- the body's own derived ceiling. This demo
    # measures the CONTROLLER, so the plant's settling lag is deliberately removed rather than
    # folded into the error signal. The live brain adapts this rate instead of pinning it.
    arm.set_substeps(10**9)
    arm.apply("HOME")
    pid = SpikingPID(seed=seed) if spiking else None
    kp, ki, kd, integ, prev = 0.6, 0.05, 0.15, 0.0, 0.0
    errs, breaches = [], 0
    for step in range(steps):
        measured = arm.state()["theta_deg"][_PID_INDEX] + (
            rng.uniform(-noise, noise) if noise else 0.0
        )
        e = target_deg - measured
        errs.append(target_deg - arm.state()["theta_deg"][_PID_INDEX])  # true error (no noise)
        if spiking:
            u = pid.step(e)
        else:  # plain numeric PID baseline
            integ += e * 1.0
            u = kp * e + ki * integ + kd * (e - prev)
            prev = e
        u = max(-15.0, min(15.0, u))  # per-step slew limit
        arm.apply(arm_config.command_for(_PID_JOINT, round(u, 4)))  # relative: PID output
        if perturb_at is not None and step == perturb_at:
            arm.inject_disturbance(_PID_JOINT, perturb_deg)  # external shove off the path
            if arm.trajectory_breach():  # comparator catches it
                breaches += 1
    return errs, arm.state()["theta_deg"][_PID_INDEX], breaches


def _rmse(errs):
    return (sum(e * e for e in errs) / len(errs)) ** 0.5


def _disturbed_plant(mode, steps=90, load=2.0, target=30.0, seed=0):
    """First-order plant under a CONSTANT disturbance: x <- x + u - load.
    P-only leaves a steady-state offset (e_ss = load/(kp)) that only the integral term can
    cancel -- the textbook case where full PID beats a fair P-only / no-control baseline, and
    the case that exercises the spiking INTEGRAL (cumulative net integrator spikes -> drive)."""
    pid = SpikingPID(seed=seed) if mode == "spiking" else None
    x, errs = 0.0, []
    for _ in range(steps):
        e = target - x
        errs.append(e)
        if mode == "spiking":
            u = pid.step(e)
        elif mode == "p":
            u = 0.6 * e
        else:
            u = 0.0
        u = max(-15.0, min(15.0, u))
        x += u - load
    return errs, target - x


def main():
    # DERIVED from the joint's measured ground-free travel, not transcribed. The incumbent's
    # +30 deg sits within a degree or two of this arm's floor contact (the band tops out at
    # ~37), so a +25 deg shove on top of it would drive the gripper through the ground and the
    # run would measure contact recovery rather than disturbance rejection. The band midpoint
    # leaves room for the shove in either direction.
    _band_lo, _band_hi = arm_config.ground_free_band(_PID_JOINT)
    target = round((_band_lo + _band_hi) / 2.0, 1)

    errs_s, final_s, _ = drive_to(target, spiking=True)
    errs_n, final_n, _ = drive_to(target, spiking=False)
    # noise robustness: 4 deg measurement jitter
    errs_sn, final_sn, _ = drive_to(target, noise=4.0, spiking=True)
    # perturbation rejection: external shove mid-run, trajectory comparator flags it, PID recovers
    errs_pr, final_pr, breaches = drive_to(target, spiking=True, perturb_at=20, perturb_deg=25)
    # disturbance rejection on a first-order plant: the case where full PID beats fair controls
    _, dist_s = _disturbed_plant("spiking")
    _, dist_p = _disturbed_plant("p")
    _, dist_none = _disturbed_plant("none")

    print("=" * 66)
    print(f"SPIKING PID (LIF-population) -- drive So101Arm {_PID_JOINT} to setpoint")
    print("=" * 66)
    print(f"target             : {target} deg")
    print(f"spiking PID        : final {final_s:.2f} deg | settle RMSE {_rmse(errs_s):.2f}")
    print(f"numeric PID (ref)  : final {final_n:.2f} deg | settle RMSE {_rmse(errs_n):.2f}")
    print(
        f"+4deg noise spk    : final {final_sn:.2f} deg | steady |err| {abs(target - final_sn):.2f}"
    )
    print(f"perturb +25deg@20  : comparator breaches {breaches} -> recovered to {final_pr:.2f} deg")
    print("-- disturbance-rejection plant (x += u - load), final |error| ---------------------")
    print(f"spiking PID        : {abs(dist_s):.2f} deg   (integral cancels the load)")
    print(f"P-only   (fair)    : {abs(dist_p):.2f} deg   (steady-state offset it cannot remove)")
    print(f"no-control (fair)  : {abs(dist_none):.2f} deg")
    print("=" * 66)

    # ---- (a) FUNCTION: the spiking PID drives error down; fair controls do worse ------------
    assert abs(final_s - target) < 1.0, f"spiking PID did not converge: {final_s}"
    assert abs(errs_s[0]) > abs(errs_s[-1]), "error did not shrink over the run"
    # Overshoot = how far past the setpoint the response travelled, measured along the
    # direction of travel so it is meaningful for a target of either sign. With
    # measured = target - err, going past the target by d gives err = -d * sign(target).
    _s = 1.0 if target >= 0 else -1.0
    overshoot = max(0.0, max(-e * _s for e in errs_s))
    assert overshoot < 0.4 * abs(target), f"excessive overshoot: {overshoot:.1f} deg"
    assert abs(final_sn - target) < 2.0, f"spiking PID not noise-robust: {final_sn}"
    assert breaches >= 1, "trajectory comparator missed the injected perturbation"
    assert abs(final_pr - target) < 2.0, (
        f"controller did not recover from the perturbation: {final_pr}"
    )
    # the spiking integral rejects a disturbance that a fair P-only / no-control baseline cannot
    assert abs(dist_s) < abs(dist_p) < abs(dist_none), (
        f"spiking PID must beat fair controls under load: {abs(dist_s):.2f} !< {abs(dist_p):.2f} !< {abs(dist_none):.2f}"
    )
    assert abs(dist_s) < 1.0, f"spiking PID did not reject the disturbance: {abs(dist_s):.2f}"

    # ---- (b) GENUINELY SPIKING: real LIF membranes emitting discrete 0/1 spike events -------
    pid = SpikingPID()
    assert isinstance(pid.p_on.v, list) and pid.p_on.theta, "LIF membrane + threshold must exist"
    single = _LIFPopulation(0.9, *_tuning(1, random.Random(1), bias_max=0.4, theta=0.5))
    events = {single.advance(0.3) for _ in range(200)}  # one neuron: spikes each micro-step
    assert events and events <= {0, 1}, (
        f"per-neuron spikes must be discrete 0/1, got {sorted(events)}"
    )
    counts = [pid.p_on.advance(0.4) for _ in range(50)]  # population: integer spike counts
    assert all(isinstance(c, int) and 0 <= c <= pid.n for c in counts), (
        "spike counts must be discrete"
    )
    assert any(c > 0 for c in counts), "LIF population must actually spike"

    # ---- (c) DETERMINISM: same seed => byte-identical --------------------------------------
    a = SpikingPID(seed=7)
    b = SpikingPID(seed=7)
    seq = [3.0, 2.0, -1.5, 0.5, 4.0, -2.0, 1.0]
    ra = [repr(a.step(x)) for x in seq]
    rb = [repr(b.step(x)) for x in seq]
    assert ra == rb, "same seed must be byte-identical"
    d1 = drive_to(target, seed=3)
    d2 = drive_to(target, seed=3)
    assert repr(d1) == repr(d2), "drive_to must be deterministic for a fixed seed"

    print(
        f"self-check OK: spiking PID -> {final_s:.2f}deg (target {target}), overshoot {overshoot:.1f}deg; "
        f"under load spiking {abs(dist_s):.2f} < P-only {abs(dist_p):.2f} < none {abs(dist_none):.2f} deg; "
        f"noise-robust {abs(target - final_sn):.2f}deg, perturb caught ({breaches}) + recovered to {final_pr:.2f}deg; "
        f"LIF membranes emit discrete 0/1 spikes; deterministic per seed"
    )


if __name__ == "__main__":
    main()
