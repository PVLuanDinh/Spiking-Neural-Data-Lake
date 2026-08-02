"""
Spiking CPG (Matsuoka half-center oscillator) — a rhythmic motor primitive for the arm.

PROMOTED from research/ into the product (v0.80). The arm loop has reactive control (spiking_pid,
reflex, Interpreter commands) but no RHYTHM generator — the circuit that produces a sustained
oscillatory sweep/wave from one tonic command, the way a biological CPG drives locomotion without
per-step planning. This wires the Matsuoka oscillator into So101Arm: its two mutually-inhibiting
half-centers give an anti-phase rhythm, and `cpg_sweep` maps that to rate-limited JOINT_A_ROTATE
commands so the arm sweeps back and forth through a bounded, collision-free band — driven entirely
through the existing (safety-checked) command API, so collision detection and EMERGENCY_STOP still
apply. Refs: Matsuoka, Biol. Cybern. 52 (1985); Ijspeert, "Central pattern generators for
locomotion", Neural Networks 21 (2008).

  python -m snn_data_lake.brainstem.cpg_oscillator        # (also run by run_selfchecks.py)
"""

from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.so101_arm import So101Arm


class MatsuokaCPG:
    """Two mutually-inhibiting adapting neurons -> a stable anti-phase oscillation."""

    def __init__(self, tau_r=0.1, tau_a=1.0, beta=2.5, w=2.5, dt=0.1):
        self.tau_r, self.tau_a, self.beta, self.w, self.dt = tau_r, tau_a, beta, w, dt
        self.reset()

    def reset(self):
        self.x1, self.v1, self.x2, self.v2 = 1.0, 0.0, 0.0, 0.0  # asymmetric start seeds the rhythm

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


#: The joint the rhythm drives, and its index into theta_deg. Named, not indexed: the
#: incumbent drove "joint A", which was the shoulder PITCH; index 0 on this body is
#: shoulder_pan, a YAW joint, so an unchanged index would have swept the arm sideways while
#: every assertion about the sweep still passed.
_CPG_JOINT = "shoulder_lift"
_CPG_INDEX = next(j.qpos_index for j in arm_config.JOINTS if j.name == _CPG_JOINT)


def _angle(arm):
    return arm.state()["theta_deg"][_CPG_INDEX]


def cpg_sweep(
    arm, steps=1200, drive=1.0, centre_deg=None, amp_deg=None, max_step_deg=8.0, seed_cpg=None
):
    """Drive the shoulder-lift joint with the CPG rhythm: the joint tracks centre +- amp*(y1-y2)
    via rate-limited relative ROTATE commands (bounded joint velocity). Returns the joint angle
    trace (deg). Goes through arm.apply(), so collision detection + EMERGENCY_STOP still gate
    motion."""
    # The band DEFAULTS to the joint's own ground-free travel. It used to default to centre 45
    # +/- 25, which described where the previous arm's joint A had clear air; on this body
    # positive shoulder_lift points DOWN and that band is through the floor. A caller may still
    # pass a narrower band, but it can no longer inherit one measured on a different arm.
    band_lo, band_hi = arm_config.ground_free_band(_CPG_JOINT)
    if centre_deg is None:
        centre_deg = (band_lo + band_hi) / 2.0
    if amp_deg is None:
        amp_deg = (band_hi - band_lo) / 2.0
    cpg = seed_cpg or MatsuokaCPG(tau_r=0.5, tau_a=6.0)  # slow -> smooth graceful sweeps
    cpg.reset()
    arm.apply("HOME")
    trace = []
    for _ in range(steps):
        y1, y2 = cpg.step(drive)
        target = centre_deg + amp_deg * max(-1.0, min(1.0, y1 - y2))  # bounded target band
        # ABSOLUTE, not relative. `target` is already an absolute angle -- the incumbent
        # differenced it into a delta only because its body had no absolute verb. On a servo
        # that lags, feeding back a delta computed from an angle the arm has not reached yet
        # makes the commanded setpoint run away from the achieved one: measured, the sweep
        # drove itself into the floor and reported GROUND after 85 of 1200 steps. The rate
        # limit still bounds joint VELOCITY, by limiting how far the target may move from the
        # angle the arm is actually at.
        cur = _angle(arm)
        want = max(cur - max_step_deg, min(cur + max_step_deg, target))
        arm.apply(arm_config.moveto_for(_CPG_JOINT, round(want, 3)))
        trace.append(_angle(arm))
        if arm.collision():  # safety: never keep driving into a collision
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
    arm = So101Arm()
    steps = 1200
    trace = cpg_sweep(arm, steps=steps)
    steady = trace[150:]  # drop the ramp-up from HOME; measure the steady sweep
    rev = _reversals(steady)
    lo, hi = min(steady), max(steady)
    collision_free = len(trace) == steps  # cpg_sweep breaks on collision -> full length = clean

    # safety: an EMERGENCY_STOP mid-sweep must freeze the arm (motion goes through the gated API)
    arm2 = So101Arm()
    cpg_sweep(arm2, steps=200)
    arm2.apply("EMERGENCY_STOP")
    frozen_angle = _angle(arm2)
    cpg2 = MatsuokaCPG()
    for _ in range(200):  # keep driving after the stop
        y1, y2 = cpg2.step(1.0)
        arm2.apply(arm_config.moveto_for(_CPG_JOINT, 5))
    stayed = _angle(arm2) == frozen_angle

    print("=" * 60)
    print("SPIKING CPG -> ARM — rhythmic motor primitive (Matsuoka half-center)")
    print("=" * 60)
    band_lo, band_hi = arm_config.ground_free_band(_CPG_JOINT)
    print(
        f"{_CPG_JOINT} steady sweep over {len(steady)} steps: {rev} direction reversals (rhythmic)"
    )
    print(
        f"angle band: {lo:.1f} deg .. {hi:.1f} deg "
        f"(ground-free band {band_lo:.0f}..{band_hi:.0f}, collision-free)"
    )
    print(
        f"collision-free whole sweep: {collision_free};  EMERGENCY_STOP freezes the sweep: {stayed}"
    )
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert rev >= 4, f"CPG must drive a sustained rhythmic sweep ({rev} reversals)"
    # Amplitude as a FRACTION of the travel available, not an absolute degree count: the old
    # 20-degree floor was a tenth of the previous arm's band and would be a different demand on
    # any other body. A tenth of the clear air is the property that transfers.
    assert (hi - lo) >= 0.1 * (band_hi - band_lo), (
        f"the sweep must have real amplitude, not a twitch ({lo:.1f}..{hi:.1f} "
        f"of {band_lo:.0f}..{band_hi:.0f})"
    )
    assert band_lo <= lo and hi <= band_hi, (
        f"sweep left the measured ground-free band: {lo:.1f}..{hi:.1f} "
        f"outside {band_lo:.0f}..{band_hi:.0f}"
    )
    assert collision_free, "the bounded sweep must stay collision-free end to end"
    assert stayed, "EMERGENCY_STOP must freeze the CPG-driven sweep (safety API respected)"
    assert cpg_sweep(So101Arm(), steps=300) == cpg_sweep(So101Arm(), steps=300), "deterministic"
    print(
        f"self-check OK: {rev} reversals in [{lo:.0f},{hi:.0f}]deg, collision-free, "
        f"E-STOP-safe; CPG wired to So101Arm via the command API; deterministic"
    )


if __name__ == "__main__":
    main()
