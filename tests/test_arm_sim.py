"""
pytest suite for arm_sim.py — 2-link planar arm simulation.

Tests mirror the self-check in arm_sim.py:main() to verify:
- HOME pose and collision detection
- Joint rotation and end-effector motion
- Effector state
- EMERGENCY_STOP freeze behavior
- Trajectory-deviation reflex (audit P1: no-op commands must NOT clear breach)
"""

from snn_data_lake.brainstem.arm_sim import ArmSim


def test_home_pose():
    """HOME pose is (2.0, 0.0) with no collision."""
    arm = ArmSim()
    home = arm.apply("HOME")
    assert home["ee"] == (2.0, 0.0), f"HOME ee should be (2.0, 0.0), got {home['ee']}"
    assert home["collision"] is None, f"HOME should have no collision, got {home['collision']}"


def test_joint_a_rotate_raises_arm():
    """JOINT_A_ROTATE(+15deg) from HOME raises joint A and lifts end-effector (y > 0)."""
    arm = ArmSim()
    arm.apply("HOME")
    s = arm.apply("JOINT_A_ROTATE(+15deg)")
    assert s["theta_deg"][0] == 15.0, f"Joint A should be 15.0 deg, got {s['theta_deg'][0]}"
    assert s["ee"][1] > 0, f"End-effector y should be > 0 after raising, got {s['ee'][1]}"


def test_effector_close_and_open():
    """ENGAGE sets effector 1.0; RELEASE sets 0.0."""
    arm = ArmSim()
    arm.apply("HOME")

    arm.apply("ENGAGE")
    assert arm.effector == 1.0, f"Effector should be 1.0 after CLOSE, got {arm.effector}"

    arm.apply("RELEASE")
    assert arm.effector == 0.0, f"Effector should be 0.0 after OPEN, got {arm.effector}"


def test_ground_collision_on_negative_rotate():
    """JOINT_A_ROTATE(-90deg) from HOME drives into ground, collision detected."""
    arm = ArmSim()
    arm.apply("HOME")
    coll = arm.apply("JOINT_A_ROTATE(-90deg)")
    assert coll["collision"] == "GROUND", f"Expected GROUND collision, got {coll['collision']}"


def test_emergency_stop_freezes_motion():
    """EMERGENCY_STOP freezes motion; motion commands are ignored until RETRACT_ALL."""
    arm = ArmSim()
    arm.apply("HOME")
    arm.apply("EMERGENCY_STOP")

    # Try to move while stopped
    frozen = arm.apply("JOINT_A_ROTATE(+30deg)")
    assert frozen["theta_deg"][0] == 0.0, f"frozen at 0deg, got {frozen['theta_deg'][0]}"
    assert frozen["stopped"] is True, f"stopped flag should be True, got {frozen['stopped']}"

    # Recover with RETRACT_ALL
    back = arm.apply("RETRACT_ALL")
    assert not back["stopped"], f"RETRACT_ALL must clear stopped, got {back['stopped']}"
    assert back["ee"] == (2.0, 0.0), f"ee should be back at HOME (2.0, 0.0), got {back['ee']}"


def test_trajectory_breach_audit_p1():
    """Audit P1: only real joint motion recommits trajectory reference.

    A no-op command (HOLD / unknown token / effector) must NOT clear a trajectory breach
    after an external perturbation. Only real joint motion recommits the path.

    Sequence:
    1. HOME, perturb +20deg -> trajectory_breach() = True
    2. HOLD (no-op) -> breach STAYS True
    3. ZZZ_UNKNOWN (unknown token) -> breach STAYS True
    4. ENGAGE (effector, no joint motion) -> breach STAYS True
    5. HOME (real motion) -> breach clears to False
    """
    arm = ArmSim()

    # Start at HOME, perturb externally (no command)
    arm.apply("HOME")
    assert not arm.trajectory_breach(), "At HOME, deviation should be near-zero (no breach)"

    arm.perturb(da_deg=20)
    assert arm.trajectory_breach(), "After perturb +20deg, trajectory_breach should be True"

    # No-op: HOLD must NOT clear the breach
    arm.apply("HOLD")
    assert arm.trajectory_breach(), "HOLD (no-op) must NOT clear trajectory breach"

    # No-op: unknown token must NOT clear the breach
    arm.apply("ZZZ_UNKNOWN")
    assert arm.trajectory_breach(), "Unknown token (no-op) must NOT clear trajectory breach"

    # No-op: effector command (no joint motion) must NOT clear the breach
    arm.apply("ENGAGE")
    assert arm.trajectory_breach(), "ENGAGE (no motion) must NOT clear breach"

    # Real joint motion: HOME recommits the path and clears the breach
    arm.apply("HOME")
    assert not arm.trajectory_breach(), "Corrective HOME (real motion) must clear trajectory breach"


def test_trajectory_deviation():
    """trajectory_breach() returns True when deviation exceeds TRAJ_DEV_MAX."""
    arm = ArmSim()
    arm.apply("HOME")

    # Initially on-path (deviation near 0)
    assert arm.deviation() < 1e-9, f"Deviation at HOME should be near-zero, got {arm.deviation()}"
    assert not arm.trajectory_breach(), "At HOME, trajectory_breach should be False"

    # Perturb: deviation grows, breach triggered
    arm.perturb(da_deg=20)
    dev = arm.deviation()
    assert dev > 0, f"Deviation after perturb should be > 0, got {dev}"
    assert arm.trajectory_breach(), f"With deviation {dev}, trajectory_breach should be True"

    # Corrective motion: path recommitted, deviation resets
    arm.apply("HOME")
    assert arm.deviation() < 1e-9, f"After HOME, deviation should reset, got {arm.deviation()}"
    assert not arm.trajectory_breach(), "After HOME, trajectory_breach should be False"
