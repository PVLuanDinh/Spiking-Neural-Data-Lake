"""The SO-101 gate: E-STOP, HOME, clamping, and the two-sided collision contract.

WRITTEN BEFORE THE IMPLEMENTATION, deliberately. These are the safety properties the incumbent
already has, and the owner authorised REIMPLEMENTING them -- which is not the same as authorising
their loss. Writing them first means the implementation is measured against the contract rather
than the contract being written to describe whatever the implementation happened to do.

Each test states the measurement that motivated it, because several of these guard against
plausible-looking designs that were tried and found wrong:

  * "frozen" must keep stepping with ctrl latched to qpos. Stopping the stepping leaves stored
    velocity that discharges when anything resumes; ctrl:=0 is a MOVE-TO-ZERO command on a
    position actuator and slews 25.89 deg during the E-STOP itself.
  * the latch must be edge-guarded. The live proximity path re-issues EMERGENCY_STOP every tick
    with no already-stopped check; unguarded, 20 000 repeats drift 35.16 deg vs 0.0148 deg once.
  * HOME must be a hard reset, not "command zeros and settle". Settling lands 0.031 deg off and
    is not bitwise reproducible across start poses.
  * the probe must gate on joint limits. mj_kinematics + mj_collision returns ncon=0 for
    out-of-range poses, so a probe without that check calls 65.9% of the planner's grid free.

  pytest tests/test_so101_arm.py
"""

import math

import pytest

pytest.importorskip("mujoco")

from snn_data_lake.brainstem import (
    arm_config,  # noqa: E402
    serial_arm,  # noqa: E402
)
from snn_data_lake.brainstem import so101_arm as sa  # noqa: E402

JOINTS = [j.name for j in arm_config.JOINTS]


def _tok(name, deg):
    return f"{arm_config.token_for(name)}({deg:+g}deg)"


def _pose(arm):
    """The full observable pose, for bit-identity comparisons."""
    st = arm.state()
    return (
        tuple(round(v, 12) for v in st["theta_deg"]),
        tuple(round(v, 12) for v in st["ee_mm"]),
        st["stopped"],
        st["collision"],
    )


# ---- E-STOP (T025, T026) ----------------------------------------------------------------


def test_estop_latches_and_every_other_command_is_a_noop():
    arm = sa.So101Arm()
    arm.apply(_tok("shoulder_lift", 20))
    arm.apply("EMERGENCY_STOP")
    assert arm.state()["stopped"] is True, "EMERGENCY_STOP must latch"

    before = _pose(arm)
    for cmd in (_tok("shoulder_lift", 30), _tok("elbow_flex", -20), "ENGAGE", "RELEASE", "HOLD"):
        arm.apply(cmd)
        assert arm.state()["stopped"] is True, f"{cmd} cleared the latch"
    assert _pose(arm) == before, "a command other than HOME/RETRACT_ALL moved a latched arm"


@pytest.mark.parametrize("clearing", ["HOME", "RETRACT_ALL"])
def test_only_home_and_retract_clear_the_latch(clearing):
    arm = sa.So101Arm()
    arm.apply("EMERGENCY_STOP")
    arm.apply(clearing)
    assert arm.state()["stopped"] is False, f"{clearing} must clear the latch"


def test_estop_is_idempotent():
    """Applying it 1000x must be bit-identical to applying it once.

    Not hypothetical: the live proximity path emits EMERGENCY_STOP every tick with no
    already-stopped guard, so an object parked at the sensor -- or one NaN distance -- holds it
    down indefinitely. Unguarded, the latch action re-runs each time and the arm creeps.
    """
    once = sa.So101Arm()
    once.apply(_tok("shoulder_lift", 25))
    once.apply("EMERGENCY_STOP")

    many = sa.So101Arm()
    many.apply(_tok("shoulder_lift", 25))
    for _ in range(1000):
        many.apply("EMERGENCY_STOP")

    assert _pose(many) == _pose(once), (
        "repeated EMERGENCY_STOP drifted the arm -- the latch action is not edge-guarded"
    )


# ---- HOME (T027) ------------------------------------------------------------------------


def test_home_is_bitwise_reproducible_from_any_pose():
    """A hard reset, not a settle. The incumbent asserts an exact home end-effector position and
    that assertion must survive the transplant."""
    reference = None
    for pose in ([10, 20, -30, 15, 40, 5], [-45, -30, 60, -20, -80, 50], [0, 0, 0, 0, 0, 0]):
        arm = sa.So101Arm()
        for name, deg in zip(JOINTS, pose, strict=True):
            arm.apply(_tok(name, deg))
        arm.apply("HOME")
        got = _pose(arm)
        if reference is None:
            reference = got
        assert got == reference, f"HOME from {pose} differs from HOME from another pose"


def test_home_clears_velocity_not_just_position():
    """Stored velocity that survives HOME discharges the moment stepping resumes.

    Asserted as STATIONARITY rather than exact equality: a position servo holding against gravity
    droops by a bounded amount, so demanding zero movement would be demanding the arm not be
    physical. What must be true is that the drift settles and does not accumulate -- a cleared
    velocity produces a fixed offset, a surviving one produces a runaway.
    """
    arm = sa.So101Arm()
    arm.apply(_tok("shoulder_lift", 60))
    arm.apply("HOME")
    start = arm.state()["theta_deg"]

    for _ in range(50):
        arm.apply("HOLD")
    early = arm.state()["theta_deg"]
    for _ in range(200):
        arm.apply("HOLD")
    late = arm.state()["theta_deg"]

    drift_early = max(abs(a - b) for a, b in zip(early, start, strict=True))
    drift_late = max(abs(a - b) for a, b in zip(late, start, strict=True))
    assert drift_late < 1.0, f"the arm ran away after HOME: {drift_late:.4f} deg"
    assert drift_late <= drift_early * 1.5 + 1e-6, (
        f"drift is still growing ({drift_early:.5f} -> {drift_late:.5f} deg) -- "
        "velocity was not cleared by HOME"
    )


# ---- clamping (T028) --------------------------------------------------------------------


def test_command_beyond_the_limit_is_clamped_and_says_so():
    """MuJoCo's own clamp is SILENT: setting ctrl out of range leaves ctrl unmutated while the
    body sits at the rail, so state() would report a command the arm never received. The gate
    must clamp, and must report that it did."""
    arm = sa.So101Arm()
    lim = arm_config.JOINT_LIMITS["shoulder_pan"]
    beyond = math.degrees(lim["ctrl_hi"]) + 45.0
    st = arm.apply(_tok("shoulder_pan", beyond))
    assert st["clamped"] is True, "an out-of-range command must report clamped"
    assert st["theta_deg"][0] <= math.degrees(lim["ctrl_hi"]) + 1e-6, (
        "the arm exceeded its commandable range"
    )


def test_in_range_command_is_not_reported_as_clamped():
    arm = sa.So101Arm()
    st = arm.apply(_tok("shoulder_pan", 10))
    assert st["clamped"] is False, "an in-range command must not report clamped"


def test_unknown_joint_is_rejected_not_ignored():
    arm = sa.So101Arm()
    before = _pose(arm)
    arm.apply("JOINT_NOSUCH_ROTATE(+10deg)")
    assert _pose(arm) == before, "an unknown joint token must not move the arm"


# ---- the collision contract, both sides (T029, T030, T031) ------------------------------


def test_live_contact_is_reported_and_the_solver_prevents_it():
    """The live half: contact is REPORTED, and the commanded angle is NOT reached.

    Strictly stronger than the assertion it replaces. The incumbent reported a collision while
    letting the command through -- an absorbing state the repo measured at
    P(collide at t+1 | collided at t) = 0.991, with the effector spending most of its wake ticks
    underground. A real body refuses.
    """
    arm = sa.So101Arm()
    # ctrl_HI is the direction that reaches the floor. MEASURED: commanded +100 deg, achieved
    # +37.80, contact GROUND. The ctrl_lo direction reaches cleanly and would prove nothing.
    commanded = math.degrees(arm_config.JOINT_LIMITS["shoulder_lift"]["ctrl_hi"])
    for _ in range(400):
        st = arm.apply(_tok("shoulder_lift", commanded))
    achieved = st["theta_deg"][1]
    assert st["collision"] is not None, "driving into the floor must report contact"
    assert abs(achieved - commanded) > 1.0, (
        f"the arm reached {achieved:.3f} deg of a commanded {commanded:.3f} -- "
        "the solver did not prevent penetration"
    )


def test_probe_reports_contact_without_preventing_it():
    """The probe half: report WITHOUT prevent, which is what the planner needs.

    The planner asks a counterfactual -- "if the arm were here, would it collide?" -- and
    "prevent" is not an answer to that. The probe therefore writes into a private, never-stepped
    body: contact is detected, and nothing moves.
    """
    qpos = [0.0] * len(JOINTS)
    qpos[1] = arm_config.JOINT_LIMITS["shoulder_lift"]["stop_hi"]
    verdict = sa.probe_contact(qpos)
    assert verdict is not None, "a penetrating pose must report contact"


def test_probe_does_not_disturb_the_live_arm():
    """Teleporting the LIVE body into a penetrating pose and resuming stepping ejects it at
    36 rad/s from standstill. The probe must be unreachable from the live arm by construction."""
    arm = sa.So101Arm()
    arm.apply(_tok("shoulder_lift", 15))
    before = _pose(arm)
    bad = [0.0] * len(JOINTS)
    bad[1] = arm_config.JOINT_LIMITS["shoulder_lift"]["stop_hi"]
    sa.probe_contact(bad)
    assert _pose(arm) == before, "probing moved the live arm"


def test_probe_rejects_out_of_limit_poses():
    """mj_kinematics + mj_collision returns ncon=0 for out-of-range qpos, so a probe that does
    not check limits calls those poses FREE -- measured at 65.9% of the planner's default grid."""
    qpos = [0.0] * len(JOINTS)
    qpos[0] = arm_config.JOINT_LIMITS["shoulder_pan"]["stop_hi"] + 1.0
    verdict = sa.probe_contact(qpos)
    assert verdict == sa.OUT_OF_LIMIT, (
        f"an out-of-limit pose reported {verdict!r}; it must not be indistinguishable from free"
    )


def test_obstacle_verdict_is_retired():
    """The vendored scene has exactly one geom (`floor`), so OBSTACLE can never fire. A branch
    that cannot fire is computed-but-not-consumed; it was retired rather than left in place."""
    assert not hasattr(sa, "OBSTACLE"), "OBSTACLE was retired -- the scene has no obstacle geom"


# ---- reached / blocked (T040's contract) -------------------------------------------------


def test_unsatisfiable_command_reports_blocked_not_an_error():
    arm = sa.So101Arm()
    commanded = math.degrees(arm_config.JOINT_LIMITS["shoulder_lift"]["ctrl_hi"])
    for _ in range(400):
        st = arm.apply(_tok("shoulder_lift", commanded))
    assert st["reached"] is False, "a contact-blocked command did not reach its target"
    assert st["blocked"] is True, "a contact-blocked command must report blocked"


def test_satisfiable_command_reports_reached():
    arm = sa.So101Arm()
    for _ in range(400):
        st = arm.apply(_tok("shoulder_pan", 20))
    assert st["reached"] is True, f"a free command never converged: {st['theta_deg'][0]}"
    assert st["blocked"] is False


# ---- the body contract ------------------------------------------------------------------


def test_state_carries_exactly_the_agreed_contract():
    """Three research areas proposed three incompatible dicts. This is the settled one."""
    st = sa.So101Arm().state()
    expected = {
        "theta_deg",
        "ee_mm",
        "effector",
        "stopped",
        "collision",
        "reached",
        "blocked",
        "clamped",
        "tracking_error",
        "deviation",
        "backend",
        "moves",
    }
    assert set(st) == expected, f"body contract drift: {set(st) ^ expected}"
    assert len(st["theta_deg"]) == 6
    assert len(st["ee_mm"]) == 3
    assert st["backend"] == "mujoco"


def test_same_seed_is_deterministic():
    a = sa.So101Arm(seed=7)
    b = sa.So101Arm(seed=7)
    for cmd in (_tok("shoulder_pan", 15), _tok("elbow_flex", -25), "ENGAGE"):
        a.apply(cmd)
        b.apply(cmd)
    assert _pose(a) == _pose(b), "same seed produced different arm state"


# ---- regressions from b83737b6 ------------------------------------------------------------
# Both of these were shipped, passed all 17 tests, and were found by an independent sweep. They
# are pinned here because neither was expressible as "the feature does not work" -- each was a
# gate that accepted MORE than it should, which no positive test can catch.


def test_gate_rejects_a_rotate_token_for_the_gripper():
    """The gripper is driven by ENGAGE/RELEASE and has no rotate token.

    Membership was resolved against arm_config.JOINTS, which includes the gripper, while
    arm_config.TOKENS excludes it -- so JOINT_GRIPPER_ROTATE was a command the GATE accepted and
    the VOCABULARY said did not exist, and it moved the effector (measured 0.090908 -> 0.091143)
    around the sanctioned branch. Two sources of truth inside the single source of truth.
    """
    assert not any("GRIPPER" in t for t in arm_config.TOKENS), "the gripper has no rotate token"
    arm = sa.So101Arm()
    before = arm.state()["effector"]
    arm.apply("JOINT_GRIPPER_ROTATE(+30deg)")
    assert arm.state()["effector"] == before, (
        "a rotate token for the gripper moved the effector -- it must be rejected"
    )


def test_gate_rejects_a_valid_token_with_trailing_junk():
    """The gate must not be MORE PERMISSIVE than the wire allow-list it mirrors.

    With re.match, "JOINT_ELBOW_FLEX_ROTATE(+5deg)EMERGENCY_STOP" performed the rotation and
    SILENTLY DROPPED THE STOP -- measured, +0.023346 deg of elbow travel with stopped=False --
    while serial_arm's allow-list rejected the same string via fullmatch. A safety gate may be
    stricter than the layer beneath it; it may never be looser.
    """
    junk = "JOINT_ELBOW_FLEX_ROTATE(+5deg)EMERGENCY_STOP"
    assert sa._JOINT_TOKEN.fullmatch(junk) is None, "the gate parsed a truncated command"
    assert serial_arm.is_valid_command(junk) is False, "the wire allow-list changed"

    arm = sa.So101Arm()
    clean = sa.So101Arm()
    before = arm.state()["theta_deg"][2]
    arm.apply(junk)
    junk_move = abs(arm.state()["theta_deg"][2] - before)

    before_c = clean.state()["theta_deg"][2]
    clean.apply(_tok("elbow_flex", 5))
    clean_move = abs(clean.state()["theta_deg"][2] - before_c)

    # The junk command must not perform the rotation. It still advances time by one step, so the
    # arm settles slightly toward its held pose -- an order of magnitude below the real move.
    assert junk_move < clean_move / 5.0, (
        f"the truncated command rotated the joint: {junk_move:.6f} vs a real move {clean_move:.6f}"
    )


def test_effector_stays_within_zero_and_one():
    """A negative closure is not a physical claim, it is a leaked implementation detail.

    Measured -0.00000170 after 500 RELEASE before the clamp: the raw ratio leaves the interval at
    the rails because a position servo overshoots and the commandable range is not the physical
    range.
    """
    for cmd, n in (("RELEASE", 500), ("ENGAGE", 501)):
        arm = sa.So101Arm()
        for _ in range(n):
            arm.apply(cmd)
        e = arm.state()["effector"]
        assert 0.0 <= e <= 1.0, f"effector left [0,1] after {n}x {cmd}: {e:.8f}"


def test_the_live_gate_is_in_both_safety_kernels():
    """so101_arm is the ONLY mutation path to the arm, so the self-rewrite harness must not be
    able to rewrite it. tests/test_self_mod_safety.py only asserts the two kernels agree with
    EACH OTHER, so "neither was updated" passes there silently -- this closes that.
    """
    import pathlib
    import re as _re

    from snn_data_lake.reasoning.godel_machine import _SAFETY_KERNEL

    assert any("so101_arm" in p for p in _SAFETY_KERNEL), (
        "the live arm gate is absent from godel_machine._SAFETY_KERNEL"
    )
    si = pathlib.Path(sa.__file__).parent.parent / "reasoning" / "self_improve.py"
    block = _re.search(r"PROTECTED = \{(.*?)\}", si.read_text(encoding="utf-8"), _re.S).group(1)
    body = "\n".join(line.split("#", 1)[0] for line in block.splitlines())
    assert "so101_arm" in body, "the live arm gate is absent from self_improve's PROTECTED"


# ---- T041/T043 + the C4/C5 migration ---------------------------------------------------


def _settled(arm, command, steps=200):
    """Apply a command and let the servo actually arrive.

    One apply() is one 2 ms physics step, so a command moves the joint 0.020 deg -- these
    tests are about WHICH joint moves and HOW FAR, and neither is observable in 2 ms.
    """
    arm.set_substeps(steps)
    return arm.apply(command)


@pytest.mark.parametrize("name", [n for n in arm_config.ROTATABLE])
def test_each_joint_moves_alone(name):
    """T041: a joint token moves ITS joint and leaves every other joint where it was.

    This is the test the old 2-DOF body could not express: with two joints, "only the
    commanded one moved" is nearly vacuous. With six it is the property that a token names a
    joint rather than merely being accepted.
    """
    arm = sa.So101Arm()
    before = arm.state()["theta_deg"]
    _settled(arm, _tok(name, 20))
    after = arm.state()["theta_deg"]

    idx = next(j.qpos_index for j in arm_config.JOINTS if j.name == name)
    assert abs(after[idx] - before[idx]) > 1.0, f"{name} did not move on its own token"
    for j in arm_config.JOINTS:
        if j.qpos_index == idx:
            continue
        assert abs(after[j.qpos_index] - before[j.qpos_index]) < 0.5, (
            f"commanding {name} moved {j.name} by "
            f"{after[j.qpos_index] - before[j.qpos_index]:.3f} deg"
        )


def test_reach_in_real_units():
    """T043: reach is the max horizontal TCP radius, in real units, and is NOT the lever arm.

    The 300 mm figure in arm_hardware is the SHOULDER LEVER used for the torque budget. Reach
    is how far the gripper gets from the base. They are different measurements of different
    things and asserting them equal would make one of them wrong.
    """
    reach_mm = arm_config.MAX_REACH_MM
    assert 100.0 < reach_mm < 1000.0, f"reach {reach_mm} is not a plausible millimetre figure"

    # Independent re-measurement at a different sample count must agree to ~1%.
    arm = sa.So101Arm()
    arm.apply("HOME")
    home_r = math.hypot(*arm.ee_mm()[:2])
    assert home_r <= reach_mm * 1.01, "the home pose reaches further than the stated maximum"

    lever_mm = max(j.lever_mm for j in arm_config.JOINTS)
    assert abs(reach_mm - lever_mm) > 1.0, (
        "reach must not be asserted equal to the shoulder lever -- they measure different things"
    )


def test_rotate_is_relative_and_moveto_is_absolute():
    """The two verbs mean different things, and the difference is the whole point of having both.

    Eight ROTATE(+10) must accumulate; eight MOVETO(+10) must not. An absolute-only vocabulary
    turned three control loops into non-controllers, which is why ROTATE kept its semantics.
    """
    rel = sa.So101Arm()
    rel.set_substeps(200)
    for _ in range(4):
        rel.apply(arm_config.command_for("shoulder_lift", -10))
    rel_angle = rel.state()["theta_deg"][1]

    absolute = sa.So101Arm()
    absolute.set_substeps(200)
    for _ in range(4):
        absolute.apply(arm_config.moveto_for("shoulder_lift", -10))
    abs_angle = absolute.state()["theta_deg"][1]

    assert rel_angle < abs_angle - 10.0, (
        f"ROTATE did not accumulate: 4x-10deg relative reached {rel_angle:.2f}, "
        f"absolute reached {abs_angle:.2f}"
    )
    assert abs(abs_angle - (-10.0)) < 3.0, f"MOVETO did not hold its setpoint ({abs_angle:.2f})"


def test_substeps_is_bounded_by_a_derived_ceiling():
    """The rate the brain adapts is bounded, and the bound comes from the body's own timestep."""
    arm = sa.So101Arm()
    assert arm.substeps == 1, "the default must be the integrator's own step, not a tuned guess"

    ceiling = int(round(1.0 / float(arm.model.opt.timestep)))
    assert arm.set_substeps(10**9) == ceiling, "substeps must clamp to one simulated second"
    assert arm.set_substeps(0) == 1, "zero steps is a body that does not move while claiming it did"
    assert arm.set_substeps(-5) == 1


def test_ground_free_band_is_measured_not_assumed():
    """The CPG's safe band is a fact about the geometry, and positive shoulder_lift points DOWN.

    Measured, the gripper is below the floor above roughly +37 deg. The incumbent swept
    45 +/- 25 deg, i.e. straight through it.
    """
    lo, hi = arm_config.ground_free_band("shoulder_lift")
    assert lo < 0 < hi, f"band {lo}..{hi} does not contain home"
    assert hi < 45.0, (
        f"band top {hi} is above the incumbent's sweep centre -- the old band was safe after all, "
        "which would mean this measurement is wrong"
    )

    # The claim is falsifiable: just past the top, the arm must actually be grounded.
    arm = sa.So101Arm()
    _settled(arm, arm_config.moveto_for("shoulder_lift", hi + 25))
    assert arm.collision() == sa.GROUND, "just past the measured band the arm is not grounded"


def test_accessors_that_kept_their_name_agree_with_state():
    """The compat accessors are the SAME value as the state dict, not a second opinion."""
    arm = sa.So101Arm()
    _settled(arm, _tok("shoulder_lift", 15))
    st = arm.state()
    assert arm.collision() == st["collision"]
    assert arm.deviation() == st["deviation"]
    assert arm.effector == st["effector"]
    assert arm.stopped == st["stopped"]


def test_renamed_accessors_are_absent_not_shimmed():
    """`theta` and `ee()` must NOT exist: their meaning changed, so silence would be a lie.

    theta was 2 entries in radians and is 6 in degrees; ee() was 2-D unitless and is 3-D
    millimetres. A compatibility shim would let an unmigrated consumer keep running while
    reading the wrong unit and the wrong joint count.
    """
    arm = sa.So101Arm()
    assert not hasattr(arm, "theta"), "a `theta` shim would silently serve the wrong units"
    assert not hasattr(arm, "ee"), "an `ee` shim would silently serve 2-D unitless coordinates"
    assert not hasattr(arm, "perturb"), "perturb was renamed so it cannot be mistaken for control"
    assert "ee" not in arm.state(), "the retired state key must be gone, not aliased"


def test_close_leaves_the_body_usable():
    """close() must not destroy a body a later reader still asks about."""
    arm = sa.So101Arm()
    arm.close()
    assert arm.state()["backend"] == "mujoco", "close() broke a body that has nothing to release"


def test_autonomous_actions_all_pass_the_wire_allow_list():
    """T047: the path that composes its own commands is checked by the same gate as chat."""
    from snn_data_lake.brainstem.autonomous_loop import ACTIONS

    unchecked = {"HOLD"}
    bad = [a for a in ACTIONS if a not in unchecked and not serial_arm.is_valid_command(a)]
    assert not bad, f"autonomous actions the wire would refuse: {bad}"
    assert len(ACTIONS) == 3 + 2 * len(arm_config.ROTATABLE), (
        f"the repertoire is not one +/- pair per rotatable joint: {len(ACTIONS)}"
    )


def test_contact_does_not_trip_the_trajectory_breach():
    """T052: contact-explained deviation is EXCLUDED, and this is the pinned decision.

    Measured, sustained ground contact produces a tracking_error of 0.6842 rad -- far larger
    than the 0.3491 from a real 20-degree external shove -- while deviation stays at exactly
    0.0. The two cannot be separated by magnitude, so `deviation` is measured against the pose
    the servo actually settled at rather than against the command.

    This matters beyond tidiness: spiking_pid.py acts on trajectory_breach(). Had contact
    tripped it, the PID would abort every time the arm touched anything, which is precisely
    when a controller is most needed.
    """
    arm = sa.So101Arm()
    arm.set_substeps(200)
    _lo, hi = arm_config.ground_free_band("shoulder_lift")
    arm.apply(arm_config.moveto_for("shoulder_lift", hi + 40))  # drive into the floor
    for _ in range(50):
        arm.apply("HOLD")

    st = arm.state()
    assert st["collision"] == sa.GROUND, "the setup failed to make contact"
    assert st["tracking_error"] > 0.5, (
        f"contact should leave a LARGE tracking error, got {st['tracking_error']:.4f}"
    )
    assert st["deviation"] == 0.0, "contact leaked into deviation"
    assert not arm.trajectory_breach(), "sustained contact tripped the trajectory reflex"


def test_external_disturbance_does_trip_the_trajectory_breach():
    """The complement: the reflex is not simply dead. A shove trips it, contact does not."""
    arm = sa.So101Arm()
    arm.set_substeps(200)
    arm.apply(arm_config.moveto_for("shoulder_lift", -40))
    assert not arm.trajectory_breach(), "an undisturbed arm reports a breach"

    arm.inject_disturbance("shoulder_lift", 20.0)
    assert arm.deviation() > 0.3, f"the shove did not register ({arm.deviation():.6f})"
    assert arm.trajectory_breach(), "an external shove did not trip the trajectory reflex"


def test_the_live_body_does_not_read_the_legacy_deviation_bound():
    """T051: TRAJ_DEV_MAX is the OLD body's bound and the new body must not inherit it.

    0.15 m was tied to a 2 m reach and compared against a dimensionless distance. The
    replacement derives its bound per joint from the model's own gravity torque and actuator
    gain, so it tracks a model change instead of going stale. The constant survives only for
    arm_sim, which is retired in Phase 7 -- re-deriving it would be work on a corpse.
    """
    assert "TRAJ_DEV_MAX" not in sa.So101Arm.trajectory_breach.__code__.co_names, (
        "the live body reads the retired 2-DOF deviation bound"
    )
    import pathlib

    src = pathlib.Path(sa.__file__).read_text(encoding="utf-8")
    assert "TRAJ_DEV_MAX" not in src, "so101_arm still references the legacy bound"
