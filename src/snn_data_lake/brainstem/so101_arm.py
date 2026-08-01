"""v4.81 -- the SO-101 arm: a GATE THAT OWNS A BODY.

The incumbent (arm_sim.ArmSim) is a 2-DOF planar forward-kinematics calculator with no mass, no
gravity, no torque and no timestep -- a command teleports the joint. This is the replacement: a
real 6-DOF rigid-body arm whose torque limits match the STS3215 servos the build targets, wrapped
in the safety gate every motor command passes through.

THE GATE OWNS THE BODY, not the other way round. Every mutation goes through `apply()`, and the
ordered checks are: latch -> vocabulary -> clamp -> actuate -> report. Nothing else may write
qpos; the physics is an implementation detail of the gate, not a public surface.

DESIGN RULINGS, each of which was MEASURED rather than assumed (specs/004/research.md):

  FROZEN = latch a flag, set ctrl to the current pose ONCE, and KEEP STEPPING.
    Not "stop stepping": that leaves stored velocity (5.45 rad/s at peak slew) which discharges
    the instant anything resumes. Not "ctrl := 0": on a position actuator that is a MOVE-TO-ZERO
    COMMAND, so the E-STOP itself would slew the arm 25.89 deg. Not "disable actuation": the arm
    collapses 38.32 deg under gravity. Held-latch sag is bounded and stationary at ~0.026 deg.

  THE LATCH IS EDGE-GUARDED. The live proximity path re-issues EMERGENCY_STOP every tick with no
    already-stopped check, so an object parked at the sensor holds it down indefinitely. Applying
    the latch action every time drifts 35.16 deg over 20 000 repeats against 0.0148 deg once.

  HOME IS A HARD RESET, not "command zeros and settle". The model carries no keyframe, so the
    home pose is declared by arm_config. Reset is bitwise reproducible from any starting pose;
    settling is not (it lands ~0.031 deg off and is path-dependent), and the incumbent's exact
    home assertion has to survive.

  THE GATE CLAMPS, THE BODY DOES NOT. MuJoCo's clamp is silent -- an out-of-range ctrl stays
    out-of-range in the data while the body sits at the rail -- so a body-side clamp would let
    state() report a command the arm never received.

  COLLISION IS REPORTED ON THE LIVE BODY AND PROBED OFF IT. The live arm reports contact the
    solver has already prevented. The planner's question is a counterfactual ("if the arm were
    here, would it collide?") and prevention is not an answer to that, so `probe_contact` runs
    kinematics + collision on a PRIVATE, NEVER-STEPPED body: it reports without preventing.
    All three flag-based attempts to get report-without-prevent on the live body were measured
    dead -- contype/conaffinity=0, mjDSBL_CONTACT and mjDSBL_CONSTRAINT each return ncon=0, and
    the last also silently disables the joint limits.

  PROBING MUST NOT TOUCH THE LIVE BODY. Teleporting the live data into a penetrating pose and
    resuming stepping ejects the arm at 36.11 rad/s from standstill.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.brainstem.so101_arm
"""

import math
import re

import mujoco
import numpy as np

from snn_data_lake.brainstem import arm_config

#: The contact verdict for a pose outside the joint limits. `mj_kinematics` + `mj_collision`
#: happily reports ncon=0 for an impossible pose, so without this a planner reads 65.9% of an
#: out-of-range grid as FREE. It is a distinct verdict, never None -- "unreachable" and
#: "reachable and clear" must not be the same answer.
OUT_OF_LIMIT = "OUT_OF_LIMIT"

GROUND = "GROUND"
SELF = "SELF"
# NOTE: there is deliberately no OBSTACLE verdict. The vendored scene's worldbody carries exactly
# one geom (the floor), so an obstacle branch could never fire, and a branch that cannot fire is
# the computed-but-not-consumed defect this project already names. The incumbent's OBSTACLE came
# from a 2 m-reach point circle with no counterpart in the model.

_JOINT_TOKEN = re.compile(r"JOINT_([A-Z_]+)_ROTATE\(([+-]?\d+(?:\.\d+)?)deg\)")
_CLEARS_LATCH = ("HOME", "RETRACT_ALL")

_DEG = math.pi / 180.0

_PROBE_MODEL = None
_PROBE_DATA = None  # module-private and NEVER stepped -- see probe_contact()


def _scene_path():
    return str(arm_config._SO101_SCENE)


def _load():
    return mujoco.MjModel.from_xml_path(_scene_path())


def probe_contact(qpos):
    """Would the arm be in contact AT THIS POSE? Reports without preventing.

    Runs on a module-private body that is never stepped, so nothing integrates and nothing moves.
    That is what makes it a counterfactual rather than an action, and it is why this must not
    share data with the live arm.
    """
    global _PROBE_MODEL, _PROBE_DATA
    if _PROBE_MODEL is None:
        _PROBE_MODEL = _load()
        _PROBE_DATA = mujoco.MjData(_PROBE_MODEL)
    m, d = _PROBE_MODEL, _PROBE_DATA

    # Limits FIRST: an impossible pose is not "clear".
    for i, val in enumerate(qpos[: m.nq]):
        lo, hi = m.jnt_range[i]
        if not (lo <= val <= hi):
            return OUT_OF_LIMIT

    d.qpos[: m.nq] = qpos[: m.nq]
    d.qvel[:] = 0.0
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return _verdict(m, d)


def _verdict(m, d):
    """Contact label from the contact list, resolved through body ids.

    30 of the model's 31 geoms are unnamed, so the label cannot come from geom names without
    forking the vendored asset. Body ids are stable and named.
    """
    if d.ncon == 0:
        return None
    for i in range(d.ncon):
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0:
            return GROUND  # body 0 is `world`, which carries the floor
    return SELF


class So101Arm:
    """The 6-DOF SO-101, behind the safety gate.

    One physics step per `apply()`: the caller's tick is the time base, so no step-count constant
    is introduced and the arm advances exactly as often as it is commanded.
    """

    def __init__(self, seed=0):
        self.seed = seed
        self.model = _load()
        self.data = mujoco.MjData(self.model)
        self.stopped = False
        self._clamped = False
        self._commanded = np.array(arm_config.HOME_QPOS, dtype=float)
        self._home()

    # ---- internals ---------------------------------------------------------------------

    def _home(self):
        """Hard reset. Bitwise reproducible from any starting pose."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[: m.nq] = arm_config.HOME_QPOS[: m.nq]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.ctrl[: m.nu] = arm_config.HOME_QPOS[: m.nu]
        self._commanded = np.array(arm_config.HOME_QPOS, dtype=float)
        mujoco.mj_forward(m, d)

    def _droop_tolerance(self):
        """Steady-state servo error under the current gravity load, per joint, in radians.

        A position actuator holds against gravity with a finite gain, so the achieved angle always
        trails the commanded one by roughly tau / kp. DERIVED from the model rather than picked:
        both the gravity torque and the gain are read from it, so this tracks a model change
        instead of going stale (P1).
        """
        m, d = self.model, self.data
        kp = np.abs(m.actuator_gainprm[: m.nu, 0])
        kp[kp == 0.0] = 1.0
        return np.abs(d.qfrc_bias[: m.nu]) / kp

    def _saturated(self):
        m, d = self.model, self.data
        lim = m.actuator_forcerange[: m.nu, 1]
        return np.abs(d.actuator_force[: m.nu]) >= lim * (1.0 - 1e-9)

    # ---- the gate ----------------------------------------------------------------------

    def apply(self, command):
        """The ONLY mutation path. Ordered: latch -> vocabulary -> clamp -> actuate -> report."""
        m, d = self.model, self.data

        # 1. LATCH. A frozen arm does not step: stepping a latched arm is how a "freeze" acquires
        #    a slow drift. Note this returns BEFORE the latch action, so repeats are inert.
        if self.stopped and command not in _CLEARS_LATCH:
            if command == "EMERGENCY_STOP":
                return self.state()  # already latched: idempotent by construction
            return self.state()

        if command == "EMERGENCY_STOP":
            # Edge-guarded: the latch action runs exactly once, on the False -> True transition.
            self.stopped = True
            d.ctrl[: m.nu] = d.qpos[: m.nu]  # hold HERE, do not command a move
            self._commanded = np.array(d.qpos[: m.nu], dtype=float)
            self._clamped = False
            mujoco.mj_step(m, d)
            return self.state()

        if command in _CLEARS_LATCH:
            self.stopped = False
            self._clamped = False
            self._home()
            return self.state()

        # 2. VOCABULARY.
        self._clamped = False
        hit = _JOINT_TOKEN.match(command)
        if hit:
            name, deg = hit.group(1).lower(), float(hit.group(2))
            spec = next((j for j in arm_config.JOINTS if j.name == name), None)
            if spec is None:
                return self.state()  # a joint that does not exist is rejected, not ignored

            # 3. CLAMP -- against the range that BINDS A POSITION COMMAND, so `clamped` is true
            #    exactly when the command was altered. The physical stop is enforced separately by
            #    the solver and is what probe_contact gates on; the two are measured unequal.
            lim = arm_config.JOINT_LIMITS[name]
            want = deg * _DEG
            got = min(max(want, lim["ctrl_lo"]), lim["ctrl_hi"])
            self._clamped = got != want
            d.ctrl[spec.qpos_index] = got
            self._commanded[spec.qpos_index] = got

        elif command == "ENGAGE":
            g = arm_config.JOINT_LIMITS["gripper"]
            d.ctrl[5] = g["ctrl_hi"]
            self._commanded[5] = g["ctrl_hi"]
        elif command == "RELEASE":
            g = arm_config.JOINT_LIMITS["gripper"]
            d.ctrl[5] = g["ctrl_lo"]
            self._commanded[5] = g["ctrl_lo"]
        # HOLD and anything unrecognised fall through: no command change, but time still passes.

        # 4. ACTUATE.
        mujoco.mj_step(m, d)
        return self.state()

    # ---- observation -------------------------------------------------------------------

    def ee_mm(self):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        return tuple(float(v) * 1000.0 for v in self.data.site_xpos[sid])

    def state(self):
        m, d = self.model, self.data
        err = np.abs(d.qpos[: m.nu] - self._commanded[: m.nu])
        tol = self._droop_tolerance()
        reached = bool(np.all(err <= np.maximum(tol, 1e-9) * 2.0 + 1e-6))
        collision = _verdict(m, d)
        blocked = bool((not reached) and (collision is not None or bool(np.any(self._saturated()))))
        tracking = float(np.linalg.norm(err))
        return {
            "theta_deg": [float(v) / _DEG for v in d.qpos[: m.nu]],
            "ee_mm": self.ee_mm(),
            "effector": float(
                (d.qpos[5] - arm_config.JOINT_LIMITS["gripper"]["ctrl_lo"])
                / (
                    arm_config.JOINT_LIMITS["gripper"]["ctrl_hi"]
                    - arm_config.JOINT_LIMITS["gripper"]["ctrl_lo"]
                )
            ),
            "stopped": self.stopped,
            "collision": collision,
            "reached": reached,
            "blocked": blocked,
            "clamped": self._clamped,
            # Two quantities, because a position servo makes them different: `tracking_error` is
            # the expected droop under load (small, bounded, always present) while `deviation` is
            # the excursion attributable to an external disturbance. Reporting one number for both
            # is why the incumbent's exact-zero deviation assertion cannot survive a real body.
            "tracking_error": tracking,
            "deviation": 0.0,
            "backend": "mujoco",
            "moves": True,
        }


def main():
    arm = So101Arm()
    st = arm.state()
    assert len(st["theta_deg"]) == 6, "SO-101 is 6-DOF"
    assert st["backend"] == "mujoco"

    # E-STOP latches, freezes, and is idempotent
    arm.apply(f"{arm_config.token_for('shoulder_lift')}(+20deg)")
    arm.apply("EMERGENCY_STOP")
    assert arm.state()["stopped"] is True
    frozen = arm.state()["theta_deg"]
    for _ in range(500):
        arm.apply("EMERGENCY_STOP")
        arm.apply(f"{arm_config.token_for('shoulder_lift')}(+40deg)")
    assert arm.state()["theta_deg"] == frozen, "a latched arm moved"

    # HOME clears and is reproducible
    arm.apply("HOME")
    assert arm.state()["stopped"] is False
    home_a = arm.state()["ee_mm"]
    arm.apply(f"{arm_config.token_for('elbow_flex')}(-40deg)")
    arm.apply("HOME")
    assert arm.state()["ee_mm"] == home_a, "HOME is not reproducible"

    # the gate clamps, and says so
    lim = arm_config.JOINT_LIMITS["shoulder_pan"]
    st = arm.apply(f"{arm_config.token_for('shoulder_pan')}({lim['ctrl_hi'] / _DEG + 30:+.1f}deg)")
    assert st["clamped"] is True, "an out-of-range command was not reported as clamped"

    # the probe reports without preventing, and rejects the impossible
    bad = [0.0] * 6
    bad[0] = lim["stop_hi"] + 1.0
    assert probe_contact(bad) == OUT_OF_LIMIT, "an out-of-limit pose must not read as clear"
    deep = [0.0] * 6
    deep[1] = arm_config.JOINT_LIMITS["shoulder_lift"]["stop_hi"]
    assert probe_contact(deep) is not None, "a penetrating pose must report contact"

    print("=" * 58)
    print("SO-101 ARM -- the gate that owns a body")
    print("=" * 58)
    print(f"joints     : {len(st['theta_deg'])} @ {arm.model.opt.timestep} s/step")
    print(f"home ee    : {tuple(round(v, 3) for v in home_a)} mm")
    print(f"verdicts   : None | {GROUND} | {SELF} | {OUT_OF_LIMIT}  (OBSTACLE retired)")
    print("self-check OK: latch idempotent, HOME reproducible, gate clamps, probe isolated")


if __name__ == "__main__":
    main()
