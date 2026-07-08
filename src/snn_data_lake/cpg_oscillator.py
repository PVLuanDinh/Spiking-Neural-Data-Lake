"""
Spiking CPG (Matsuoka half-center oscillator) — a rhythmic motor primitive for the arm.

PROMOTED from research/ into the product (v0.80). The arm loop has reactive control (spiking_pid,
reflex, Interpreter commands) but no RHYTHM generator — the circuit that produces a sustained
oscillatory sweep/wave from one tonic command, the way a biological CPG drives locomotion without
per-step planning. This wires the Matsuoka oscillator into ArmSim: its two mutually-inhibiting
half-centers give an anti-phase rhythm, and `cpg_sweep` maps that to rate-limited JOINT_A_ROTATE
commands so the arm sweeps back and forth through a bounded, collision-free band — driven entirely
through the existing (safety-checked) command API, so collision detection and EMERGENCY_STOP still
apply. Refs: Matsuoka, Biol. Cybern. 52 (1985); Ijspeert, "Central pattern generators for
locomotion", Neural Networks 21 (2008).

  python -m snn_data_lake.cpg_oscillator        # (also run by run_selfchecks.py)
"""
import math

from snn_data_lake.arm_sim import ArmSim


class MatsuokaCPG:
    """Two mutually-inhibiting adapting neurons -> a stable anti-phase oscillation."""

    def __init__(self, tau_r=0.1, tau_a=1.0, beta=2.5, w=2.5, dt=0.1):
        self.tau_r, self.tau_a, self.beta, self.w, self.dt = tau_r, tau_a, beta, w, dt
        self.reset()

    def reset(self):
        self.x1, self.v1, self.x2, self.v2 = 1.0, 0.0, 0.0, 0.0   # asymmetric start seeds the rhythm

    def step(self, drive):
        y1, y2 = max(0.0, self.x1), max(0.0, self.x2)
        dx1 = (-self.x1 - self.beta * self.v1 - self.w * y2 + drive) / self.tau_r
        dv1 = (-self.v1 + y1) / self.tau_a
        dx2 = (-self.x2 - self.beta * self.v2 - self.w * y1 + drive) / self.tau_r
        dv2 = (-self.v2 + y2) / self.tau_a
        self.x1 += self.dt * dx1
        self.v1 += self.dt * dv1
        self.x2 += self.dt * dx2
        self.v2 += self.dt * dv2
        return max(0.0, self.x1), max(0.0, self.x2)


def cpg_sweep(arm, steps=1200, drive=1.0, centre_deg=45.0, amp_deg=25.0, max_step_deg=8.0, seed_cpg=None):
    """Drive ArmSim joint A with the CPG rhythm: joint A tracks centre +- amp*(y1-y2), via
    rate-limited relative JOINT_A_ROTATE commands (bounded joint velocity). Returns the joint-A angle
    trace (deg). Goes through arm.apply(), so collision detection + EMERGENCY_STOP still gate motion."""
    cpg = seed_cpg or MatsuokaCPG(tau_r=0.5, tau_a=6.0)   # slow -> smooth graceful sweeps
    cpg.reset()
    arm.apply("HOME")
    trace = []
    for _ in range(steps):
        y1, y2 = cpg.step(drive)
        target = centre_deg + amp_deg * max(-1.0, min(1.0, y1 - y2))   # bounded target band
        cur = arm.theta[0] / (math.pi / 180.0)
        delta = max(-max_step_deg, min(max_step_deg, target - cur))    # rate limit (joint velocity)
        arm.apply(f"JOINT_A_ROTATE({delta:+.3f}deg)")
        trace.append(arm.theta[0] / (math.pi / 180.0))
        if arm.collision():                              # safety: never keep driving into a collision
            break
    return trace


def _reversals(trace):
    """Count direction changes in the angle trace — one per half-cycle of the sweep."""
    n, prev = 0, 0
    for a, b in zip(trace, trace[1:]):
        d = b - a
        s = 1 if d > 1e-6 else (-1 if d < -1e-6 else 0)
        if s and prev and s != prev:
            n += 1
        if s:
            prev = s
    return n


def main():
    arm = ArmSim()
    steps = 1200
    trace = cpg_sweep(arm, steps=steps)
    steady = trace[150:]                          # drop the ramp-up from HOME; measure the steady sweep
    rev = _reversals(steady)
    lo, hi = min(steady), max(steady)
    collision_free = len(trace) == steps          # cpg_sweep breaks on collision -> full length = clean

    # safety: an EMERGENCY_STOP mid-sweep must freeze the arm (motion goes through the gated API)
    arm2 = ArmSim()
    cpg_sweep(arm2, steps=200)
    arm2.apply("EMERGENCY_STOP")
    frozen_angle = arm2.theta[0]
    cpg2 = MatsuokaCPG()
    for _ in range(200):                      # keep driving after the stop
        y1, y2 = cpg2.step(1.0)
        arm2.apply("JOINT_A_ROTATE(+5deg)")
    stayed = abs(arm2.theta[0] - frozen_angle) < 1e-9

    print("=" * 60)
    print("SPIKING CPG -> ARM — rhythmic motor primitive (Matsuoka half-center)")
    print("=" * 60)
    print(f"joint A steady sweep over {len(steady)} steps: {rev} direction reversals (rhythmic)")
    print(f"angle band: {lo:.1f} deg .. {hi:.1f} deg (target 20..70, collision-free)")
    print(f"collision-free whole sweep: {collision_free};  EMERGENCY_STOP freezes the sweep: {stayed}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert rev >= 4, f"CPG must drive a sustained rhythmic sweep ({rev} reversals)"
    assert hi - lo >= 20.0, f"the sweep must have real amplitude, not a twitch ({lo:.1f}..{hi:.1f})"
    assert 15.0 <= lo and hi <= 75.0, f"sweep must stay in the safe joint band ({lo:.1f}..{hi:.1f})"
    assert collision_free, "the bounded sweep must stay collision-free end to end"
    assert stayed, "EMERGENCY_STOP must freeze the CPG-driven sweep (safety API respected)"
    assert cpg_sweep(ArmSim(), steps=300) == cpg_sweep(ArmSim(), steps=300), "deterministic"
    print(f"self-check OK: {rev} reversals in [{lo:.0f},{hi:.0f}]deg, collision-free, "
          f"E-STOP-safe; CPG wired to ArmSim via the command API; deterministic")


if __name__ == "__main__":
    main()
