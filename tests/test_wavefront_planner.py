"""The six-joint sampling planner.

The pairs below are MEASURED, not invented. A planner test whose query happens to be solvable
by the straight-line fast path never executes the search at all, and would pass against an RRT
that does not work -- so the blocked pair here was found by sampling the free space and keeping
one whose direct segment is genuinely obstructed.
"""

import pytest

pytest.importorskip("mujoco")

from snn_data_lake.brainstem import arm_config  # noqa: E402
from snn_data_lake.brainstem.so101_arm import So101Arm  # noqa: E402
from snn_data_lake.cerebellum import wavefront_planner as wp  # noqa: E402

HOME = [0.0] * len(wp.JOINTS)

# Measured 2026-08-01: straight-line segment blocked, RRT solves it in 9 waypoints.
BLOCKED_START = [95.7, -12.2, 39.0, 79.9, 50.0]
BLOCKED_GOAL = [-5.3, 14.2, 82.2, -80.9, -118.2]


def test_planner_finds_collision_free_path():
    """T068a: a query the straight line cannot answer is answered by the search."""
    assert not wp.collides(BLOCKED_START), "the measured start is no longer free"
    assert not wp.collides(BLOCKED_GOAL), "the measured goal is no longer free"
    assert not wp._segment_clear(BLOCKED_START, BLOCKED_GOAL), (
        "the direct segment is clear, so this query no longer exercises the RRT"
    )

    path = wp.plan(BLOCKED_START, BLOCKED_GOAL, seed=0)
    assert path is not None, "no path found for a connectable query"
    assert len(path) > 2, "the search returned a straight line for a blocked query"
    assert path[0] == BLOCKED_START and path[-1] == BLOCKED_GOAL, "path must span the query"
    for a, b in zip(path, path[1:], strict=False):
        assert wp._segment_clear(a, b), "the returned path sweeps through an obstacle"


def test_shortcutting_preserves_clearance():
    path = wp.plan(BLOCKED_START, BLOCKED_GOAL, seed=0)
    short = wp.shortcut(path, seed=0)
    assert len(short) <= len(path)
    assert short[0] == path[0] and short[-1] == path[-1], "shortcutting moved the endpoints"
    for a, b in zip(short, short[1:], strict=False):
        assert wp._segment_clear(a, b), "shortcutting produced a colliding segment"


def test_straight_line_is_not_searched():
    """A clear query must not pay for a search it does not need."""
    path = wp.plan(HOME, [-20.0, -20.0, 0.0, 0.0, 0.0], seed=0)
    assert path is not None and len(path) == 2


def test_colliding_goal_is_refused_not_approximated():
    """A path that stops short still looks like a plan to execute(). It must be None."""
    _lo, hi = arm_config.ground_free_band("shoulder_lift")
    buried = [0.0, hi + 40.0, 0.0, 0.0, 0.0]
    assert wp.collides(buried), "the setup pose is not actually in contact"
    assert wp.plan(HOME, buried, seed=0) is None
    assert wp.plan(buried, HOME, seed=0) is None, "a colliding START must also be refused"


def test_planner_is_deterministic():
    a = wp.plan(BLOCKED_START, BLOCKED_GOAL, seed=11)
    b = wp.plan(BLOCKED_START, BLOCKED_GOAL, seed=11)
    assert a == b, "the same seed produced a different path"


def test_ik_converges_and_avoids_contact():
    arm = So101Arm()
    x, y, z = arm.ee_mm()
    cfg = wp.ik(arm, x - 40.0, y + 30.0, z - 20.0)
    assert cfg is not None, "IK failed on a target a few centimetres from home"
    assert len(cfg) == len(wp.JOINTS)
    assert not wp.collides(cfg), "IK returned a colliding configuration"

    # Unreachable targets must be refused, not silently clamped to the nearest pose.
    assert wp.ik(arm, 5000.0, 5000.0, 5000.0) is None, "IK claimed an unreachable target"


def test_planner_respects_the_estop_latch():
    """THE safety property: a latched arm is not moved by a planner."""
    arm = So101Arm()
    arm.set_substeps(10**9)
    arm.apply("EMERGENCY_STOP")
    before = arm.state()["theta_deg"]

    res = wp.plan_to_config(arm, [35.0, -45.0, 30.0, -20.0, 15.0], seed=0)

    assert arm.state()["theta_deg"] == before, "a planner moved a latched arm"
    assert arm.state()["stopped"] is True, "the latch did not survive the plan"
    assert res["reached"] == 0, "a latched arm reported reaching waypoints"


def test_planner_uses_only_existing_vocabulary():
    """The planner must add no new command token: it moves via MOVETO, already allow-listed."""
    import pathlib
    import re

    from snn_data_lake.brainstem import serial_arm

    src = pathlib.Path(wp.__file__).read_text(encoding="utf-8")
    literals = set(re.findall(r'"(JOINT_[A-Z_]+\([^"]*\))"', src))
    assert not literals, f"the planner hand-builds command strings: {literals}"

    for name in wp.JOINTS:
        assert serial_arm.is_valid_command(arm_config.moveto_for(name, 10)), (
            f"the planner's MOVETO token for {name} is not wire-valid"
        )


def test_execute_drives_through_the_gate():
    arm = So101Arm()
    arm.set_substeps(10**9)
    goal = [35.0, -45.0, 30.0, -20.0, 15.0]
    res = wp.plan_to_config(arm, goal, seed=0)
    assert res["planned"] is True
    assert res["reached"] >= 1, f"nothing was reached: {res}"
    here = [arm.state()["theta_deg"][wp._index(n)] for n in wp.JOINTS]
    assert max(abs(a - b) for a, b in zip(here, goal, strict=True)) < 10.0, (
        f"the arm did not end near the goal: {here} vs {goal}"
    )
