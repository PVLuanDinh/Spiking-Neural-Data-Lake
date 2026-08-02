"""Collision-free motion planning for the six-joint arm.

WHY THIS IS NO LONGER A WAVEFRONT. The previous planner discretised configuration space into a
grid and floated an excitable reaction-diffusion wave from the goal back to the start. That is a
good algorithm for two joints: at 5 degrees the grid is 36 x 72 = 2592 cells and the wave sweeps
it in milliseconds. The same grid over six joints is 36**6 -- about 2.2 billion cells -- so the
wave cannot be run at all, and coarsening it enough to fit would make it miss every gap it was
supposed to find. Dimension, not preference, is what retires it (P2: the better mechanism
becomes the path, the inferior one goes).

WHAT REPLACES IT. A bidirectional RRT over the joint angles. Sampling does not enumerate the
space, so its cost tracks the difficulty of the query rather than the volume of the workspace.

WHAT DID NOT CHANGE, deliberately:
  * Every motion still leaves through `arm.apply()`. The planner computes waypoints; it never
    touches qpos. A latched arm therefore refuses a plan exactly as it refuses anything else.
  * Collision checking is `so101_arm.probe_contact`, the one collision authority, on a body that
    is never stepped -- so asking "would this pose collide?" cannot move the real arm.
  * The planner adds NO new command token. It moves via the MOVETO verb already in the
    vocabulary.

DETERMINISM. Seeded per call. Two runs with the same seed produce the same path, because a
planner that silently returns a different route each time cannot be regression-tested.
"""

import math
import random

from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.so101_arm import So101Arm, probe_contact

DEG = math.pi / 180.0

#: Joints the planner moves. The gripper is not a pose degree of freedom -- it opens and closes
#: via ENGAGE/RELEASE -- so planning over it would search a dimension that cannot help.
JOINTS = arm_config.ROTATABLE


def _index(name):
    return next(j.qpos_index for j in arm_config.JOINTS if j.name == name)


def _limits():
    """(lo_deg, hi_deg) per planned joint, from the range that BINDS A POSITION COMMAND.

    ctrl range, not jnt_range: a plan whose waypoints the actuator would silently clamp is not
    a plan, it is a suggestion. The two are measured unequal on this model.
    """
    out = []
    for name in JOINTS:
        lim = arm_config.JOINT_LIMITS[name]
        out.append((lim["ctrl_lo"] / DEG, lim["ctrl_hi"] / DEG))
    return out


def _full_qpos(cfg_deg):
    """Planned joints -> a full qpos vector in radians (gripper held at home)."""
    q = list(arm_config.HOME_QPOS)
    for name, deg in zip(JOINTS, cfg_deg):
        q[_index(name)] = deg * DEG
    return q


def _here(arm):
    st = arm.state()
    return [st["theta_deg"][_index(n)] for n in JOINTS]


def collides(cfg_deg):
    """Would the arm be in contact at this configuration? Reports; never moves anything."""
    return probe_contact(_full_qpos(cfg_deg)) is not None


def _segment_clear(a, b, step_deg=4.0):
    """Is the straight line between two configurations collision-free?

    Checked at a fixed angular resolution rather than only at the endpoints. Endpoint-only
    checking is the classic way a planner returns a path that sweeps straight through an
    obstacle: both ends are fine and everything between them is not.
    """
    span = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    n = max(1, int(math.ceil(span / step_deg)))
    for i in range(1, n + 1):
        t = i / n
        if collides([x + (y - x) * t for x, y in zip(a, b)]):
            return False
    return True


def _steer(frm, to, max_step_deg):
    """Move from `frm` toward `to` by at most max_step_deg on the fastest-moving joint."""
    d = [y - x for x, y in zip(frm, to)]
    span = max((abs(v) for v in d), default=0.0)
    if span <= max_step_deg:
        return list(to)
    k = max_step_deg / span
    return [x + v * k for x, v in zip(frm, d)]


def plan(start_deg, goal_deg, max_step_deg=12.0, iters=600, goal_bias=0.2, seed=0):
    """Bidirectional RRT from start to goal. Returns a waypoint list, or None if it fails.

    Returning None rather than a partial path is deliberate: a path that stops short still
    looks like a plan to `execute()`, which would drive the arm somewhere nobody asked for.
    """
    start_deg, goal_deg = list(start_deg), list(goal_deg)
    if collides(start_deg) or collides(goal_deg):
        return None
    if _segment_clear(start_deg, goal_deg):  # the easy case is very common; do not RRT it
        return [start_deg, goal_deg]

    rng = random.Random(seed)
    lim = _limits()
    na, nb = [start_deg], [goal_deg]
    ta, tb = {0: None}, {0: None}
    swapped = False

    def nearest(nodes, q):
        best, bi = None, 0
        for i, n in enumerate(nodes):
            dist = sum((x - y) ** 2 for x, y in zip(n, q))
            if best is None or dist < best:
                best, bi = dist, i
        return bi

    def extend(nodes, tree, q):
        i = nearest(nodes, q)
        new = _steer(nodes[i], q, max_step_deg)
        if not _segment_clear(nodes[i], new):
            return None
        nodes.append(new)
        tree[len(nodes) - 1] = i
        return len(nodes) - 1

    def chain(nodes, tree, i):
        out = []
        while i is not None:
            out.append(nodes[i])
            i = tree[i]
        return out

    for _ in range(iters):
        q = list(nb[0]) if rng.random() < goal_bias else [rng.uniform(lo, hi) for lo, hi in lim]
        ia = extend(na, ta, q)
        if ia is not None:
            ib = extend(nb, tb, na[ia])
            if ib is not None and _segment_clear(na[ia], nb[ib]):
                fwd, back = list(reversed(chain(na, ta, ia))), chain(nb, tb, ib)
                path = fwd + back
                return list(reversed(path)) if swapped else path
        na, nb, ta, tb = nb, na, tb, ta  # grow both trees, alternately
        swapped = not swapped
    return None


def shortcut(waypoints, rounds=40, seed=0):
    """Randomised shortcutting. An RRT path is collision-free but wanders; this straightens it."""
    if not waypoints or len(waypoints) < 3:
        return [list(p) for p in waypoints]
    rng = random.Random(seed)
    pts = [list(p) for p in waypoints]
    for _ in range(rounds):
        if len(pts) < 3:
            break
        i = rng.randrange(0, len(pts) - 2)
        j = rng.randrange(i + 2, len(pts))
        if _segment_clear(pts[i], pts[j]):
            pts = pts[: i + 1] + pts[j:]
    return pts


def ik(arm, x_mm, y_mm, z_mm, iters=300, tol_mm=5.0, damping=0.05):
    """Damped-least-squares Jacobian IK to a Cartesian target, in millimetres.

    Iterative and Jacobian-based because a six-joint arm has no closed form worth writing: the
    incumbent's two-link `ik()` was a pair of arccosines, which is exactly the thing that does
    not generalise. Returns a configuration in degrees, or None if it does not converge or the
    converged pose is in contact.

    Runs on ITS OWN MjData, never the live arm's -- solving is a question, not a motion.
    """
    import mujoco
    import numpy as np

    m = arm_config._model()
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    lim = _limits()
    idx = [_index(n) for n in JOINTS]

    try:
        start = _here(arm)
    except Exception:
        start = [0.0] * len(JOINTS)
    for k, qi in enumerate(idx):
        d.qpos[qi] = start[k] * DEG

    target = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0
    jacp = np.zeros((3, m.nv))
    ok = False
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        err = target - d.site_xpos[sid]
        if float(np.linalg.norm(err)) * 1000.0 <= tol_mm:
            ok = True
            break
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        jac = jacp[:, idx]
        # Damped least squares: a plain pseudo-inverse blows up near a singular configuration,
        # which for this arm is simply "reaching straight out" -- a pose a user will ask for.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(3), err)
        for k, qi in enumerate(idx):
            lo, hi = lim[k]
            d.qpos[qi] = min(max(d.qpos[qi] + float(dq[k]), lo * DEG), hi * DEG)
    if not ok:
        return None

    cfg = [float(d.qpos[qi]) / DEG for qi in idx]
    return None if collides(cfg) else cfg


def execute(arm, waypoints, tol_deg=3.0):
    """Drive the arm along the waypoints THROUGH the safety gate.

    Uses MOVETO, the absolute verb: a waypoint is a pose, and expressing it as a relative delta
    would make every step depend on the servo having finished the previous one.
    """
    applied, reached = 0, 0
    for wp in waypoints:
        for name, deg in zip(JOINTS, wp):
            arm.apply(arm_config.moveto_for(name, round(deg, 3)))
            applied += 1
        if arm.state()["stopped"]:  # a latched arm: stop pushing motion into a freeze
            break
        if max(abs(a - b) for a, b in zip(_here(arm), wp)) <= tol_deg:
            reached += 1
    return {"waypoints": len(waypoints), "applied": applied, "reached": reached}


def plan_to_config(arm, goal_deg, seed=0, **kw):
    """Plan and execute a move to a joint configuration (degrees, one per planned joint)."""
    wps = plan(_here(arm), list(goal_deg), seed=seed, **kw)
    if wps is None:
        return {"planned": False, "reason": "no collision-free path found", "reached": 0}
    out = execute(arm, shortcut(wps, seed=seed))
    out["planned"] = True
    return out


def plan_to_xyz(arm, x_mm, y_mm, z_mm, seed=0, **kw):
    """Plan and execute a move to a Cartesian end-effector target, in millimetres."""
    goal = ik(arm, x_mm, y_mm, z_mm)
    if goal is None:
        return {"planned": False, "reason": "no reachable collision-free IK solution", "reached": 0}
    out = plan_to_config(arm, goal, seed=seed, **kw)
    out["goal_cfg"] = [round(g, 2) for g in goal]
    return out


def main():
    home = [0.0] * len(JOINTS)

    # ---- 1. a clear straight line is answered without search -----------------------------
    easy = plan(home, [-20.0, -20.0, 0.0, 0.0, 0.0], seed=0)
    assert easy is not None and len(easy) == 2, "a clear straight line should not need an RRT"

    # ---- 2. the collision check is real, and it is the ONE authority ----------------------
    _lo, hi = arm_config.ground_free_band("shoulder_lift")
    buried = [0.0, hi + 40.0, 0.0, 0.0, 0.0]
    assert collides(buried), "a pose through the floor was not reported as contact"
    assert not collides(home), "the home pose must not be in contact"

    # ---- 3. a plan INTO a colliding goal is refused, not partially executed ---------------
    assert plan(home, buried, seed=0) is None, "planned a path into a colliding goal"

    # ---- 4. sweeping is checked, not just the endpoints ----------------------------------
    assert not _segment_clear(home, buried), "the swept segment was not checked"

    # ---- 5. a real query: plan, shortcut, execute through the gate ------------------------
    goal = [35.0, -45.0, 30.0, -20.0, 15.0]
    wps = plan(home, goal, seed=0)
    assert wps is not None, "no path found to a reachable free configuration"
    short = shortcut(wps, seed=0)
    assert len(short) <= len(wps), "shortcutting lengthened the path"
    for a, b in zip(short, short[1:]):
        assert _segment_clear(a, b), "shortcutting produced a colliding segment"

    arm = So101Arm()
    arm.set_substeps(10**9)  # let each waypoint complete; this checks the PLANNER, not the lag
    res = plan_to_config(arm, goal, seed=0)
    assert res["planned"] and res["reached"] >= 1, f"execution reached nothing: {res}"

    # ---- 6. determinism ------------------------------------------------------------------
    assert plan(home, goal, seed=7) == plan(home, goal, seed=7), "the planner is not deterministic"

    # ---- 7. IK lands where it was asked to, collision-free -------------------------------
    probe = So101Arm()
    tgt = probe.ee_mm()
    solved = ik(probe, tgt[0] - 40.0, tgt[1] + 30.0, tgt[2] - 20.0)
    assert solved is not None, "IK failed on a target a few centimetres from home"
    assert not collides(solved), "IK returned a colliding configuration"

    # ---- 8. THE SAFETY PROPERTY: a latched arm is not moved by a planner ------------------
    frozen = So101Arm()
    frozen.set_substeps(10**9)
    frozen.apply("EMERGENCY_STOP")
    before = frozen.state()["theta_deg"]
    plan_to_config(frozen, goal, seed=0)
    assert frozen.state()["theta_deg"] == before, "a planner moved a LATCHED arm"
    assert frozen.state()["stopped"] is True, "the latch did not survive a plan"

    print("=" * 62)
    print("MOTION PLANNER - six-joint sampling planner (bidirectional RRT)")
    print("=" * 62)
    print(f"planned joints  : {len(JOINTS)} ({', '.join(JOINTS)})")
    print(f"query           : home -> {goal}")
    print(f"path            : {len(wps)} waypoints -> {len(short)} after shortcutting")
    print(f"execution       : {res['reached']}/{res['waypoints']} waypoints reached via MOVETO")
    print(f"IK              : {[round(v, 1) for v in solved]} deg, collision-free")
    print("safety          : E-STOP latch survives a plan; the arm did not move")
    print("=" * 62)
    print(
        "self-check OK: straight-line fast path; swept-segment collision checking; colliding "
        "goal refused; shortcut stays clear; deterministic per seed; Jacobian IK converges; "
        "a latched arm is not moved by the planner"
    )


if __name__ == "__main__":
    main()
