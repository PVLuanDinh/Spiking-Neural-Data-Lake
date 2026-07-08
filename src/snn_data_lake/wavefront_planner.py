"""
Configuration-space wavefront path planner for the 2-DOF arm -- reaction-diffusion computing, applied.

This is the real home for reaction_diffusion_compute (v2.58): the excitable-medium wave that solves mazes
there plans collision-free arm motion here. The arm's CONFIGURATION SPACE (theta_A, theta_B) is discretized
into a grid; a cell is a WALL iff placing the arm at that joint pair puts the end-effector in collision,
reusing arm_sim.ArmSim.collision() -- the ONE collision authority (EE point vs the OBSTACLE circle and the
floor). A wave ignited at the GOAL config floods the free configs (arrival time == geodesic distance);
steepest descent from the START config down that field is a collision-free joint-space path that routes
AROUND the C-space obstacle. Each waypoint is executed as bounded, rate-limited relative JOINT_[AB]_ROTATE
commands THROUGH the existing ArmSim.apply() gate, so the E-STOP freeze / calibration gate / command
allow-list all stay in force. This module adds NO new command token and modifies NONE of the five
safety-envelope files (reflex, serial_arm, arm_sim, arm_config, interpreter) -- it only calls their public
surface read-only and drives motion through apply().

Cartesian goal: ik(x, y) is the closed-form 2-link inverse kinematics (law of cosines, elbow up/down) so a
caller can request an end-effector POINT; the planner picks the reachable, collision-free elbow branch whose
wavefront path is shortest.

Safety ceiling (honest): collision labelling INHERITS ArmSim.collision() verbatim -- an EE-POINT vs a point
obstacle, with no link-body sweep and no joint limits, sampled only at grid cells (the continuous single-joint
arc between two adjacent free cells is not swept). Small step_deg shrinks the gap; the per-step collision()
guard re-confirms each endpoint during execution; and the day ArmSim.collision() gains link-body / swept-volume
checking, this planner improves for free. It deliberately does NOT implement its own divergent collision model.

Pure standard library, deterministic (no RNG). ASCII-only output.

  python -m snn_data_lake.wavefront_planner        # (also run by run_selfchecks.py)

Reference: Steinbock, Toth & Showalter (1995), excitable-medium maze solving; the wave engine is
snn_data_lake.reaction_diffusion_compute (bfs_dist / excite_wave / backtrace / neighbours).
"""
import math

from snn_data_lake.arm_sim import ArmSim
from snn_data_lake.reaction_diffusion_compute import backtrace, bfs_dist, excite_wave, neighbours

DEG = math.pi / 180.0


def ik(x, y, l1=1.0, l2=1.0):
    """Closed-form 2-link planar inverse kinematics. Returns up to two (theta_A, theta_B) configs in
    RADIANS -- elbow-down then elbow-up -- that place the end-effector at (x, y); [] if unreachable
    (|EE| outside [|l1-l2|, l1+l2]). Mirrors the FK in ArmSim.ee()."""
    c = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if c < -1.0 or c > 1.0:
        return []
    c = max(-1.0, min(1.0, c))
    sols = []
    for sign in (1.0, -1.0):                                   # elbow-down, elbow-up
        tb = sign * math.acos(c)
        ta = math.atan2(y, x) - math.atan2(l2 * math.sin(tb), l1 + l2 * math.cos(tb))
        sols.append((ta, tb))
    return sols


class CSpaceGrid:
    """Discretized (theta_A, theta_B) configuration space. A cell (i, j) maps to joint angles and is FREE
    iff the arm placed there does not collide -- labelled by ArmSim.collision(), the collision authority."""

    def __init__(self, step_deg=5.0, a_range=(0.0, 180.0), b_range=(-180.0, 180.0)):
        self.step = step_deg
        self.a_lo, self.a_hi = a_range
        self.b_lo, self.b_hi = b_range
        self.na = int(round((self.a_hi - self.a_lo) / step_deg)) + 1
        self.nb = int(round((self.b_hi - self.b_lo) / step_deg)) + 1
        probe = ArmSim()                                       # one scratch sim, reused read-only
        self.free = set()
        for i in range(self.na):
            for j in range(self.nb):
                ta, tb = self.angles(i, j)
                probe.theta = [ta, tb]
                if probe.collision() is None:
                    self.free.add((i, j))

    def angles(self, i, j):
        """Cell -> (theta_A, theta_B) in RADIANS."""
        return ((self.a_lo + i * self.step) * DEG, (self.b_lo + j * self.step) * DEG)

    def cell(self, cfg):
        """(theta_A, theta_B) radians -> nearest grid cell, clamped in range."""
        i = round((cfg[0] / DEG - self.a_lo) / self.step)
        j = round((cfg[1] / DEG - self.b_lo) / self.step)
        return (max(0, min(self.na - 1, i)), max(0, min(self.nb - 1, j)))

    def snap(self, cfg):
        """Nearest FREE cell to cfg (breadth-first outward through blocked cells); None if none free."""
        start = self.cell(cfg)
        if start in self.free:
            return start
        seen, frontier = {start}, [start]
        while frontier:
            nxt = []
            for cell in frontier:
                for nb in neighbours(cell):
                    if 0 <= nb[0] < self.na and 0 <= nb[1] < self.nb and nb not in seen:
                        if nb in self.free:
                            return nb
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        return None


def plan(start_cfg, goal_cfg, step_deg=5.0, a_range=(0.0, 180.0), b_range=(-180.0, 180.0)):
    """Wavefront plan from start config to goal config (both (theta_A, theta_B) in RADIANS). Ignites the
    excitable wave at the goal cell (reaction_diffusion_compute.excite_wave) and back-traces from the start
    cell down the arrival-time field. Returns (status, waypoints): ('OK', [(ta, tb), ...] start->goal) or
    ('NO_FREE_CELL', []) / ('NO_PATH', []) if start/goal snap fails or the goal is unreachable in C-space."""
    grid = CSpaceGrid(step_deg, a_range, b_range)
    s = grid.snap(start_cfg)
    t = grid.snap(goal_cfg)
    if s is None or t is None:
        return ("NO_FREE_CELL", [])
    arrival, _ = excite_wave(grid.free, [t])
    if s not in arrival:                                       # goal cell walled off from the start cell
        return ("NO_PATH", [])
    cells = backtrace(arrival, s, t)                           # [s, ..., t] down the gradient
    return ("OK", [grid.angles(i, j) for (i, j) in cells])


def execute(arm, waypoints, slew_deg=8.0, tol_deg=1.0, max_steps=4000):
    """Drive the arm through the waypoint configs via bounded relative JOINT_[AB]_ROTATE commands THROUGH the
    gated ArmSim.apply(). Stops on E-STOP (state()['stopped']), collision, step budget, or arrival. Returns
    {'reached', 'stopped', 'collided', 'steps'}. Never commands the wire directly and adds no new token."""
    steps = 0
    for (ta, tb) in waypoints:
        for j, target_rad in ((0, ta), (1, tb)):
            axis = "A" if j == 0 else "B"
            target_deg = target_rad / DEG
            while True:
                if arm.state()["stopped"]:                    # mid-plan E-STOP: a frozen arm no-ops apply()
                    return {"reached": False, "stopped": True, "collided": False, "steps": steps}
                if steps >= max_steps:
                    return {"reached": False, "stopped": False, "collided": False, "steps": steps}
                err = target_deg - arm.theta[j] / DEG
                if abs(err) <= tol_deg:
                    break
                delta = max(-slew_deg, min(slew_deg, err))     # rate limit (bounded joint velocity)
                arm.apply(f"JOINT_{axis}_ROTATE({delta:+.4f}deg)")
                steps += 1
                if arm.collision() is not None:                # independent guard (mirrors cpg_sweep)
                    return {"reached": False, "stopped": arm.state()["stopped"], "collided": True, "steps": steps}
    reached = (abs(arm.theta[0] - waypoints[-1][0]) <= tol_deg * DEG
               and abs(arm.theta[1] - waypoints[-1][1]) <= tol_deg * DEG)
    return {"reached": reached, "stopped": arm.state()["stopped"],
            "collided": arm.collision() is not None, "steps": steps}


def _finish(arm, waypoints):
    """Execute a planned path and label the outcome status."""
    res = execute(arm, waypoints)
    res["status"] = ("REACHED" if res["reached"] else "ESTOPPED" if res["stopped"]
                     else "COLLIDED" if res["collided"] else "INCOMPLETE")
    res["waypoints"] = len(waypoints)
    return res


def plan_to_config(arm, ta_deg, tb_deg, step_deg=5.0, a_range=(0.0, 180.0), b_range=(-180.0, 180.0)):
    """Plan + execute to a target JOINT config (degrees), from the arm's live pose. Returns a status dict."""
    status, wp = plan((arm.theta[0], arm.theta[1]), (ta_deg * DEG, tb_deg * DEG), step_deg, a_range, b_range)
    if status != "OK":
        return {"status": status, "reached": False, "waypoints": 0, "steps": 0}
    return _finish(arm, wp)


def plan_to_xy(arm, x, y, step_deg=5.0, a_range=(0.0, 180.0), b_range=(-180.0, 180.0)):
    """Plan + execute so the end-effector reaches Cartesian (x, y). IK gives up to two elbow branches; the
    planner keeps the reachable, collision-free branch with the shortest wavefront path. Returns a status
    dict; 'UNREACHABLE' if IK has no solution, 'BLOCKED_GOAL' if no branch yields a collision-free plan."""
    goals = ik(x, y)
    if not goals:
        return {"status": "UNREACHABLE", "reached": False, "waypoints": 0, "steps": 0}
    start = (arm.theta[0], arm.theta[1])
    best = None
    probe = ArmSim()
    for gcfg in goals:
        probe.theta = [gcfg[0], gcfg[1]]
        if probe.collision() is not None:                     # the goal pose itself must be collision-free
            continue
        status, wp = plan(start, gcfg, step_deg, a_range, b_range)
        if status == "OK" and (best is None or len(wp) < len(best)):
            best = wp
    if best is None:
        return {"status": "BLOCKED_GOAL", "reached": False, "waypoints": 0, "steps": 0}
    return _finish(arm, best)


def _ee(cfg):
    """Forward kinematics via the authoritative ArmSim.ee() (avoids a second FK implementation)."""
    a = ArmSim()
    a.theta = [cfg[0], cfg[1]]
    return a.ee()


def main():
    print("=" * 64)
    print("WAVEFRONT ARM PLANNER -- reaction-diffusion path planning in joint space")
    print("=" * 64)

    # A verified detour: START and GOAL are just outside the obstacle circle, but the straight wrist sweep
    # between them passes the end-effector THROUGH the obstacle -- so the plan must route around it.
    start_cfg = (10.0 * DEG, 100.0 * DEG)
    goal_cfg = (10.0 * DEG, 115.0 * DEG)
    a_range, b_range, step = (0.0, 90.0), (0.0, 180.0), 5.0
    grid = CSpaceGrid(step, a_range, b_range)
    s_cell, g_cell = grid.cell(start_cfg), grid.cell(goal_cfg)
    status, wp = plan(start_cfg, goal_cfg, step, a_range, b_range)
    manhattan = abs(s_cell[0] - g_cell[0]) + abs(s_cell[1] - g_cell[1])
    print(f"start cfg (10,100) EE {tuple(round(v, 3) for v in _ee(start_cfg))}  free={start_cfg and s_cell in grid.free}")
    print(f"goal  cfg (10,115) EE {tuple(round(v, 3) for v in _ee(goal_cfg))}  free={g_cell in grid.free}")
    print(f"plan status {status}: {len(wp)} waypoints, geodesic {len(wp) - 1} vs Manhattan {manhattan} (detour)")

    # straight cell-interpolation start->goal, to show the direct path is blocked
    straight = [(round(s_cell[0] + (g_cell[0] - s_cell[0]) * k / manhattan),
                 round(s_cell[1] + (g_cell[1] - s_cell[1]) * k / manhattan)) for k in range(manhattan + 1)]
    blocked_on_straight = sum(1 for c in straight if c not in grid.free)
    print(f"straight interpolation crosses {blocked_on_straight} blocked cell(s) -> a real obstacle to route around")

    # Cartesian interface: reach the goal end-effector POINT via IK + the wavefront plan, through apply()
    goal_xy = _ee(goal_cfg)
    arm = ArmSim()
    arm.theta = [start_cfg[0], start_cfg[1]]
    arm.expected_ee = arm.ee()
    res = plan_to_xy(arm, goal_xy[0], goal_xy[1], step, a_range, b_range)
    print(f"plan_to_xy {tuple(round(v, 3) for v in goal_xy)}: status {res['status']}, "
          f"{res['waypoints']} waypoints, {res['steps']} gated commands, "
          f"final EE {tuple(round(v, 3) for v in arm.ee())}")
    print("=" * 64)

    # -------------------- self-check (deterministic; emergent, nothing hardcoded) --------------------
    # (h) inverse kinematics: FK(IK(x,y)) round-trips; out-of-reach returns no solution
    for (tx, ty) in [(1.4, 0.5), (0.4, 0.99), (-0.5, 1.2), (0.0, 1.5)]:
        sols = ik(tx, ty)
        assert sols, f"ik unexpectedly empty for reachable ({tx},{ty})"
        for cfg in sols:
            ex, ey = _ee(cfg)
            assert abs(ex - tx) < 1e-9 and abs(ey - ty) < 1e-9, f"ik round-trip failed {cfg} -> {(ex, ey)}"
    assert ik(3.0, 0.0) == [] and ik(0.0, 2.5) == [], "ik must reject out-of-reach targets"

    # (a) every planned waypoint is collision-free (labelled by the collision authority)
    assert status == "OK" and len(wp) >= 2, f"plan failed: {status}"
    for cfg in wp:
        probe = ArmSim(); probe.theta = [cfg[0], cfg[1]]
        assert probe.collision() is None, f"planned waypoint collides: {cfg}"
    # (b) connectivity: consecutive waypoints are one grid step apart in exactly one joint (4-adjacent)
    for (a, b) in zip(wp, wp[1:]):
        da, db = round((b[0] - a[0]) / DEG / step), round((b[1] - a[1]) / DEG / step)
        assert (abs(da), abs(db)) in ((1, 0), (0, 1)), f"waypoints not 4-adjacent: {a}->{b}"
    # (c) it is a genuine DETOUR: longer than the Manhattan straight line, which itself is blocked
    assert (len(wp) - 1) > manhattan, "path is not a detour"
    assert blocked_on_straight >= 1, "the straight path was not actually blocked -- no obstacle to route"

    # (d) executing through apply() reaches the goal, never collides, preserves the trajectory invariants
    assert res["status"] == "REACHED", f"execution did not reach goal: {res}"
    assert abs(arm.ee()[0] - goal_xy[0]) < 0.05 and abs(arm.ee()[1] - goal_xy[1]) < 0.05, "EE not at goal"
    assert arm.collision() is None and not arm.stopped, "arm ended collided or stopped"
    assert arm.deviation() < 1e-6 and not arm.trajectory_breach(), "trajectory-deviation invariant broken"

    # (e) a mid-plan E-STOP freezes the arm and aborts execution (motion goes through the gated apply())
    estop_arm = ArmSim(); estop_arm.theta = [start_cfg[0], start_cfg[1]]
    frozen = list(estop_arm.theta)
    estop_arm.apply("EMERGENCY_STOP")
    er = execute(estop_arm, wp)
    assert er["stopped"] and not er["reached"] and estop_arm.theta == frozen, "E-STOP did not freeze/abort"

    # (f) a disconnected goal is reported, not crashed (the NO_PATH guard the planner relies on)
    disconnected = {(0, 0), (0, 1), (9, 9)}                    # (9,9) is isolated from (0,0)
    arr_dc, _ = excite_wave(disconnected, [(0, 0)])
    assert (9, 9) not in arr_dc, "excite_wave reached a disconnected cell"

    # (g) determinism + the wave computes the geodesic: excite_wave arrival == independent BFS, repeatable
    arrival, _ = excite_wave(grid.free, [g_cell])
    assert arrival == bfs_dist(grid.free, [g_cell]), "excitable wave != BFS geodesic"
    assert plan(start_cfg, goal_cfg, step, a_range, b_range) == (status, wp), "planner is nondeterministic"

    print(f"self-check OK: IK round-trips + rejects out-of-reach; wavefront plan is collision-free, "
          f"4-adjacent, a real detour (geodesic {len(wp) - 1} > Manhattan {manhattan}, straight path blocked); "
          f"reaches the Cartesian goal through gated apply() with trajectory invariants intact; mid-plan "
          f"E-STOP freezes the arm; disconnected goal handled; wave == BFS geodesic; deterministic")


if __name__ == "__main__":
    main()
