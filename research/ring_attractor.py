"""
Ring attractor — angular path integration / proprioceptive joint-state estimation (pure Python, no tensors).

A continuous attractor network (CAN): N rate units arranged on a ring hold a single self-sustaining bump of
activity via local excitation + global inhibition. With NO input the bump persists (a continuous line of
attractors -> analog memory of one angle). A velocity input shifts the bump at a rate proportional to the
angular velocity, so the bump position INTEGRATES velocity into absolute angle -- dead-reckoning with no
encoder. This is the mechanism behind head-direction cells and the recent spiking ring attractor for robot
joint-state estimation (arXiv:2604.14021, 2026): a zero-dep angle tracker the arm can use to estimate a
revolute joint from its angular velocity. The velocity shift uses the standard McNaughton/Zhang asymmetric
weight = the angular DERIVATIVE of the symmetric bump kernel. Pure standard library, deterministic. On-thesis
with population_encoding / arm_sim / spiking_pid.

  python -m research.ring_attractor        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import math


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
        """Angular std of activity around the bump center (degrees) — small => localized single bump."""
        mu = self.decode()
        tot = sum(self.a) or 1.0
        var = sum(self.a[i] * math.atan2(math.sin(self.theta[i] - mu),
                                         math.cos(self.theta[i] - mu)) ** 2 for i in range(self.n)) / tot
        return math.degrees(var ** 0.5)


def _unwrap(deg):
    out, off = [], 0.0
    for k, d in enumerate(deg):
        if k and d + off - out[-1] > 180:
            off -= 360
        elif k and d + off - out[-1] < -180:
            off += 360
        out.append(d + off)
    return out


def _r2(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return (sxy * sxy) / (sxx * syy) if sxx and syy else 0.0


def main():
    ring = RingAttractor()

    # 1. persistence: no input -> bump holds its angle (analog memory), stays a single localized bump
    ring.reset(0.0)
    for _ in range(60):
        ring.step(0.0)
    persist_drift = abs(math.degrees(ring.decode()))
    width = ring.bump_width()

    # 2. path integration: a constant angular velocity -> decoded angle ramps LINEARLY (integrates velocity)
    ring.reset(0.0)
    ramp = [math.degrees(ring.step(0.2)) for _ in range(40)]
    ramp = _unwrap(ramp)
    r2 = _r2(list(range(len(ramp))), ramp)
    slope = (ramp[-1] - ramp[0]) / len(ramp)

    # 3. reversibility (the dead-reckoning hallmark): +v for T then -v for T returns to the start
    ring.reset(0.0)
    for _ in range(40):
        ring.step(0.2)
    turned = math.degrees(ring.decode())
    for _ in range(40):
        ring.step(-0.2)
    return_err = abs(((math.degrees(ring.decode()) - 0.0 + 180) % 360) - 180)

    # 4. direction: +v and -v move the bump opposite ways
    ring.reset(0.0)
    for _ in range(15):
        ring.step(0.2)
    plus = math.degrees(ring.decode())
    ring.reset(0.0)
    for _ in range(15):
        ring.step(-0.2)
    minus = math.degrees(ring.decode())

    print("=" * 60)
    print("RING ATTRACTOR — angular path integration (joint-state dead-reckoning)")
    print("=" * 60)
    print(f"persistence : bump held {persist_drift:.2f} deg drift over 60 steps, width {width:.1f} deg (localized)")
    print(f"integration : constant vel -> {slope:.2f} deg/step ramp, linear R^2={r2:.4f}")
    print(f"reversible  : +40 then -40 steps (turned {turned:.0f} deg) -> return err {return_err:.2f} deg")
    print(f"direction   : +v -> {plus:.1f} deg, -v -> {minus:.1f} deg (opposite)")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert persist_drift < 5.0 and width < 45.0, f"bump must persist as a single localized attractor ({persist_drift=}, {width=})"
    assert r2 > 0.99 and abs(slope) > 0.3, f"decoded angle must integrate velocity linearly (R^2={r2:.3f}, slope={slope:.2f})"
    assert return_err < 10.0, f"out-and-back must return to start (path integration): err {return_err:.2f} deg"
    assert (plus > 2.0 > -2.0 > minus) or (plus < -2.0 < 2.0 < minus), f"+v and -v must move opposite ways ({plus=}, {minus=})"
    r2b = RingAttractor(); r2b.reset(0.0)
    once = [r2b.step(0.2) for _ in range(10)]
    r2c = RingAttractor(); r2c.reset(0.0)
    twice = [r2c.step(0.2) for _ in range(10)]
    assert once == twice, "same drive -> identical trajectory (deterministic)"
    print(f"self-check OK: persistent bump (drift {persist_drift:.1f} deg, width {width:.0f} deg); velocity "
          f"integrated linearly (R^2 {r2:.3f}); out-and-back returns ({return_err:.1f} deg); deterministic")


if __name__ == "__main__":
    main()
