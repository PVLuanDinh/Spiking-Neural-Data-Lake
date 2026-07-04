"""
Ring attractor + spiking proprioception — encoder-free joint-angle estimate for the arm (pure stdlib).

A continuous attractor network (CAN): N rate units on a ring hold a single self-sustaining bump of activity via
local excitation + global inhibition. With no input the bump persists (analog memory of one angle); a velocity
input shifts it at a rate proportional to the angular velocity, so the bump position INTEGRATES velocity into
absolute angle -- dead-reckoning with no encoder (head-direction cells / arXiv:2604.14021 joint-state estimation).

PRODUCT integration (additive, opt-in, read-only): `ProprioceptiveEstimator` drives the ring from the arm's
COMMANDED joint deltas, giving a spiking estimate of the joint angle that never reads the joint sensor. Because
an uncommanded disturbance (`ArmSim.perturb`) is never fed to it, the estimate then DIVERGES from the true angle
-> a joint-level slip/disturbance signal that complements the end-effector-level `ArmSim.deviation()`. It is a
pure observer: it issues no commands and never touches the collision / EMERGENCY_STOP / trajectory-reflex path,
so the arm's safety envelope and default behaviour are unchanged. Deterministic.

  python -m snn_data_lake.ring_attractor        # (also run by run_selfchecks.py)
"""
import math

from snn_data_lake.arm_sim import ArmSim


class RingAttractor:
    """CAN ring: persistent bump (angle memory) + velocity-driven shift (angular path integration)."""

    def __init__(self, n=80, kappa=3.0, inhibition=0.9, total=50.0, vel_gain=0.2):
        self.n = n
        self.total = total
        self.inhibition = inhibition
        self.vel_gain = vel_gain
        self.theta = [2.0 * math.pi * i / n for i in range(n)]
        # symmetric local-excitation kernel + its angular derivative (the velocity-shift / offset weights)
        self.w_sym = [[math.exp(kappa * math.cos(self.theta[i] - self.theta[j])) for j in range(n)]
                      for i in range(n)]
        self.w_asym = [[-kappa * math.sin(self.theta[i] - self.theta[j])
                        * math.exp(kappa * math.cos(self.theta[i] - self.theta[j])) for j in range(n)]
                       for i in range(n)]
        self.reset(0.0)

    def reset(self, angle=0.0):
        a = [math.exp(3.0 * math.cos(self.theta[i] - angle)) for i in range(self.n)]
        s = sum(a)
        self.a = [x * self.total / s for x in a]

    def step(self, vel=0.0):
        n, a = self.n, self.a
        mean_a = sum(a) / n
        out = []
        for i in range(n):
            rec = sum(self.w_sym[i][j] * a[j] for j in range(n)) / n           # sustain the bump
            shift = vel * self.vel_gain * sum(self.w_asym[i][j] * a[j] for j in range(n)) / n   # move it
            v = rec + shift - self.inhibition * mean_a                          # global inhibition
            out.append(v * v if v > 0.0 else 0.0)                              # relu + supralinear (sharpen)
        s = sum(out) or 1.0
        self.a = [x * self.total / s for x in out]                             # divisive normalization (stable)
        return self.decode()

    def decode(self):
        """Bump center = circular mean of activity over the ring (the represented angle, radians)."""
        s = sum(self.a[i] * math.sin(self.theta[i]) for i in range(self.n))
        c = sum(self.a[i] * math.cos(self.theta[i]) for i in range(self.n))
        return math.atan2(s, c)

    def bump_width(self):
        """Angular std of activity around the bump center (degrees) -- small => localized single bump."""
        mu = self.decode()
        tot = sum(self.a) or 1.0
        var = sum(self.a[i] * math.atan2(math.sin(self.theta[i] - mu),
                                         math.cos(self.theta[i] - mu)) ** 2 for i in range(self.n)) / tot
        return math.degrees(var ** 0.5)


def _circ_diff_deg(a, b):
    """Signed smallest-angle difference a - b in degrees, wrapped to (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


class ProprioceptiveEstimator:
    """Spiking dead-reckoner: integrate the arm's COMMANDED joint deltas into an encoder-free angle estimate."""

    def __init__(self, steps_per_command=3):
        self.ring = RingAttractor()
        self.steps = steps_per_command
        # self-calibrate: how many degrees the decoded bump shifts per (velocity * step)
        self.ring.reset(0.0)
        before = math.degrees(self.ring.decode())
        cal_v, cal_s = 0.2, 10
        for _ in range(cal_s):
            self.ring.step(cal_v)
        self._gain = _circ_diff_deg(math.degrees(self.ring.decode()), before) / (cal_v * cal_s)
        self.ring.reset(0.0)

    def command(self, delta_deg):
        """Feed one commanded joint delta; the bump integrates it (no sensor read)."""
        vel = delta_deg / (self._gain * self.steps) if delta_deg else 0.0
        for _ in range(self.steps):
            self.ring.step(vel)
        return self.estimate_deg()

    def estimate_deg(self):
        return math.degrees(self.ring.decode())

    def mismatch_deg(self, actual_deg):
        """|estimate - actual| in degrees -> grows when the joint moved WITHOUT a command (slip/disturbance)."""
        return abs(_circ_diff_deg(self.estimate_deg(), actual_deg))


def main():
    # ---- 1. the ring as a bare angular integrator (persist / integrate / reverse) ----
    ring = RingAttractor()
    ring.reset(0.0)
    for _ in range(60):
        ring.step(0.0)
    persist_drift = abs(math.degrees(ring.decode()))
    width = ring.bump_width()
    ring.reset(0.0)
    for _ in range(30):
        ring.step(0.2)
    turned = math.degrees(ring.decode())

    # ---- 2. PRODUCT integration: spiking proprioception on ArmSim (read-only observer) ----
    arm = ArmSim()
    est = ProprioceptiveEstimator()
    for _ in range(6):                                    # six commanded +12 deg rotations of joint A
        arm.apply("JOINT_A_ROTATE(+12deg)")
        est.command(+12.0)
    actual = arm.state()["theta_deg"][0]
    tracked_mismatch = est.mismatch_deg(actual)           # estimate should track the commanded motion

    arm.perturb(da_deg=30.0)                              # uncommanded disturbance -> NOT fed to the estimator
    slipped = arm.state()["theta_deg"][0]
    slip_mismatch = est.mismatch_deg(slipped)             # now estimate and truth diverge -> slip detected

    print("=" * 60)
    print("RING ATTRACTOR — angular integrator + spiking arm proprioception")
    print("=" * 60)
    print(f"ring: persist {persist_drift:.2f} deg drift/60 steps, width {width:.0f} deg; +vel turned {turned:.0f} deg")
    print(f"arm proprioception: after 6x commanded +12deg -> estimate {est.estimate_deg():.1f} vs actual {actual:.1f} "
          f"(mismatch {tracked_mismatch:.1f})")
    print(f"uncommanded +30deg perturb -> actual {slipped:.1f}, estimate {est.estimate_deg():.1f} "
          f"(mismatch {slip_mismatch:.1f}) -> SLIP DETECTED")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert persist_drift < 5.0 and width < 45.0, f"bump must persist as a single localized attractor ({persist_drift=}, {width=})"
    assert tracked_mismatch < 8.0, f"estimate must track the commanded joint motion ({tracked_mismatch:.1f} deg)"
    assert slip_mismatch > 20.0, f"an uncommanded disturbance must diverge estimate from truth ({slip_mismatch:.1f} deg)"
    assert arm.state()["theta_deg"][0] == slipped, "estimator is read-only: it must not move the arm"
    est2 = ProprioceptiveEstimator()
    assert est2.command(12.0) == ProprioceptiveEstimator().command(12.0), "same command -> identical estimate (deterministic)"
    print(f"self-check OK: ring integrates angle; proprioception tracks commanded motion (mismatch "
          f"{tracked_mismatch:.1f} deg) and flags uncommanded slip ({slip_mismatch:.1f} deg); read-only; deterministic")


if __name__ == "__main__":
    main()
