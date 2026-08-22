"""v4.81 -- the SO-101 arm: a GATE THAT OWNS A BODY.

The incumbent (arm_sim.ArmSim) is a 2-DOF planar forward-kinematics calculator with no mass, no
gravity, no torque and no timestep -- a command teleports the joint. This is the replacement: a
real 6-DOF rigid-body arm whose torque limits match the STS3215 servos the build targets, wrapped
in the safety gate every motor command passes through.

THE GATE OWNS THE BODY, not the other way round. Every mutation goes through `apply()`, and the
ordered checks are: latch -> vocabulary -> clamp -> actuate -> report. Nothing else may write
qpos; the physics is an implementation detail of the gate, not a public surface.

DESIGN RULINGS, each of which was MEASURED rather than assumed (specs/004/research.md):

  FROZEN = latch a flag, set ctrl to the current pose once on the EDGE, then refuse to step.
    A latched arm is BITWISE frozen -- measured, 200 further applies leave theta_deg and ee_mm
    bit-identical -- because apply() returns before mj_step while stopped. Note the latch EDGE
    itself steps once (0.106 mm of TCP travel), so a freeze proof must capture its baseline AFTER
    the EMERGENCY_STOP, not before.
    The alternatives were measured and rejected: "ctrl := 0" is a MOVE-TO-ZERO command on a
    position actuator (25.89 deg of slew during the E-STOP itself); "disable actuation" collapses
    the arm 38.32 deg under gravity; and holding ctrl while continuing to step leaves a bounded
    ~0.026 deg sag, which is why this code does not step at all while latched.

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

  THE LATCH IS DURABLE ACROSS RESTART (specs/022 T037a; owner decision OD-4, delegated
    2026-08-19). The daemon restarts on every commit, so an in-process-only latch is cleared by
    a routine automated event -- which is not a latch. On the E-STOP edge `apply()` also writes
    a sidecar (ESTOP_LATCH under `substrate.home.home()`, beside the lake and never in it:
    safety state must be readable when the lake is not, and it is not learned content; the NAME
    lives in this envelope file so an edit outside the approval gate cannot move or void it);
    a So101Arm built by a DIFFERENT process reads it at construction and comes up stopped, with
    `latch_source == "restored"` and the writer named.
    WHO MAY CLEAR WHAT, decided at this gate and nowhere else (`apply(cmd, deliberate=...)`):
      * a stop THIS process tripped: HOME / RETRACT_ALL from anyone, as before;
      * a stop RESTORED from a previous process: ONLY a DELIBERATE reset -- `deliberate=True`,
        which the cockpit's actuation ticket and its chat route pass (a human) and nothing
        automated does; an automated HOME is REFUSED, the arm stays frozen, and the refusal is
        counted (`latch_refusals`). The review built the bypasses this closes: the loop's
        self-clear, a SerialArm mirror's construction HOME, the reflex's RETRACT_ALL, the chat
        route's raw apply -- every one now meets the same rule in the one gate;
      * the sidecar is OWNED: a body that never held the stop (a twin, or a process whose arm
        was built before the stop existed) leaves it untouched on HOME -- the review erased a
        live stop with a never-latched body's routine HOME, and that is closed too.
    SCOPE, stated: the sidecar is honoured across PROCESSES (a per-process random token is
    recorded -- not the pid, which a reboot reuses); a body rebuilt inside the same process is a
    new body, not a restart, and starts clear -- the reflex re-evaluates on its first tick.
    FAIL CLOSED: an unreadable home or a corrupt sidecar is a stop; a sidecar the gate cannot
    write leaves the in-process latch standing and is COUNTED (`latch_persist_errors`), never
    silent. The record is written directly (a torn read already fails closed, so a temp-and-
    rename would buy nothing and, on Windows, fails under a concurrent reader). Every self-check
    that drives this gate runs under `substrate.home.throwaway_home()`, so a by-hand run can
    never leave the live brain's next restart frozen. Reds in main(): restart-persists (a real
    second process), automated-clear-refused-and-counted, deliberate-reset-clears, the same-
    process boundary pinned, a never-latched body's HOME leaves a foreign stop intact,
    unreadable-fails-closed, persist-failure-counted.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.brainstem.so101_arm
"""

import json
import math
import os
import re
import time

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

#: ROTATE is RELATIVE (add to the current setpoint), MOVETO is ABSOLUTE (drive to that angle).
#: The verb is part of the token, so a caller cannot get the semantics it did not ask for.
_JOINT_TOKEN = re.compile(r"JOINT_([A-Z_]+)_(ROTATE|MOVETO)\(([+-]?\d+(?:\.\d+)?)deg\)")
_CLEARS_LATCH = ("HOME", "RETRACT_ALL")

_DEG = math.pi / 180.0

#: THIS process's identity in the sidecar (T037a). A random token, not the pid: a pid is reused
#: after a reboot -- the very restart the latch exists to survive -- so a pid-keyed "same
#: process" test could read a previous boot's stop as its own and drop it. A sidecar without a
#: token (any other writer) is foreign, which is the fail-closed direction.
_PROCESS_TOKEN = os.urandom(8).hex()


#: The durable E-STOP sidecar's name (T037a) -- defined INSIDE the envelope on purpose: a rename
#: anywhere else would silently void every stop on disk while the envelope gate stayed green.
ESTOP_LATCH = "estop.latch"


# distinct-from: reasoning/hjepa_torch.py, reasoning/jepa_scaffold_torch.py -- those defer a
#   torch import so a file stays importable without torch; this defers substrate.home so the
#   ENVELOPE's module-level import list is unchanged and NORD_HOME is read at use. Read.
def _latch_path():
    """Where the durable E-STOP sidecar lives: ESTOP_LATCH under substrate.home.home(), resolved
    at the moment it is needed under whatever NORD_HOME holds (home() supplies the directory
    only; the name is this file's)."""
    from snn_data_lake.substrate.home import home

    return os.path.join(home(), ESTOP_LATCH)


#: The joints a ROTATE token may name. Derived from arm_config.TOKENS, so it excludes the
#: gripper by construction rather than by a second hand-written list.
_ROTATABLE = {
    j.name: j for j in arm_config.JOINTS if arm_config.token_for(j.name) in arm_config.TOKENS
}

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


def _effector_ratio(d):
    lo = arm_config.JOINT_LIMITS["gripper"]["ctrl_lo"]
    hi = arm_config.JOINT_LIMITS["gripper"]["ctrl_hi"]
    return (d.qpos[5] - lo) / (hi - lo)


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
        #: Physics substeps per apply() -- HOW LONG one motor command is allowed to act.
        #:
        #: THE BRAIN OWNS THIS. It is deliberately not a tuned constant: see set_substeps().
        #: Starts at 1, the integrator's own step, which is the only value that is a measured
        #: fact rather than a choice. A caller that never adapts it gets an arm that crawls --
        #: visibly, in `reached`, rather than silently.
        self.substeps = 1
        self._view = None  # lazily-created offscreen picture of THIS body; see the `view` property
        self._commanded = np.array(arm_config.HOME_QPOS, dtype=float)
        #: The pose the servo believes it is holding. deviation is measured against THIS, so an
        #: external shove shows up and ordinary servo droop does not.
        self._settled = np.array(arm_config.HOME_QPOS, dtype=float)
        self._home()
        #: None | "EMERGENCY_STOP" (tripped by this process) | "restored" (read from the sidecar
        #: another process left). `latch_restored_from` names the writer; `latch_persist_errors`
        #: counts sidecar writes/removes that failed (the in-process latch stands regardless).
        self.latch_source = None
        self.latch_restored_from = None
        self.latch_persist_errors = 0
        #: automated clears REFUSED because the latch was restored from another process
        self.latch_refusals = 0
        self._honour_sidecar()

    # ---- internals ---------------------------------------------------------------------

    # distinct-from: hippocampus/proximity_replay.py -- cross_process_durability_check proves
    #   LEARNED content survives in the lake (P4b); this reads a SAFETY flag kept beside it.
    # distinct-from: cortex/neurogenesis_engine.py -- _restart_probe restores the grown
    #   substrate from the lake; same distinction. Both read.
    # distinct-from: substrate/home.py -- owns the sidecar's path; this is its one reader. Read.
    def _honour_sidecar(self):
        """T037a: come up STOPPED when a different process left the E-STOP sidecar. Fail
        closed: an unreadable home or a corrupt sidecar is a stop."""
        try:
            path = _latch_path()
            if not os.path.exists(path):
                return
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            token = str(rec.get("token", ""))
            writer = rec.get("pid")
            writer = writer if isinstance(writer, int) else "?"  # file-derived: never raw text
            when = str(rec.get("utc", "?"))
        except Exception as e:
            self.stopped = True
            self.latch_source = "restored"
            self.latch_restored_from = f"unreadable sidecar ({type(e).__name__})"
            return
        if token == _PROCESS_TOKEN:
            return  # same process: a new body, not a restart (stated in the module docstring)
        self.stopped = True
        self.latch_source = "restored"
        self.latch_restored_from = f"pid {writer} at {when}"

    def _persist_latch(self, latched):
        """Write the sidecar on the E-STOP edge, remove it on the deliberate reset. A failure
        is counted and the in-process latch is unaffected -- persistence never weakens a stop."""
        try:
            path = _latch_path()
            if latched:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "token": _PROCESS_TOKEN,
                            "pid": os.getpid(),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "source": "EMERGENCY_STOP",
                        },
                        f,
                    )
            else:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        except Exception:
            self.latch_persist_errors += 1

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
        self._settled = np.array(d.qpos[: m.nu], dtype=float)

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

    def inject_disturbance(self, joint, deg, *, allow_latched=False):
        """TEST-ONLY: shove a joint without commanding it, to stand in for an external force.

        The live arm has no exafference channel -- nothing in the simulation pushes it. This is
        how the trajectory-deviation reflex is exercised anyway, and it is deliberately NOT part
        of the command path: it writes qpos directly, leaves `_commanded` untouched (so the gap
        it opens IS the deviation), and runs mj_forward rather than mj_step so the servo does not
        immediately correct it within the same call.

        Named `inject_disturbance` and not `perturb` precisely so it cannot be mistaken for the
        incumbent's control-adjacent helper. Nothing on the live tick calls this.
        """
        spec = _ROTATABLE.get(joint)
        if spec is None:
            return self.state()
        # THE LATCH APPLIES HERE TOO, added 2026-08-03. MEASURED before this: with the arm
        # E-STOPPED, the gated path moved 0.000000 deg and this moved 20.000000 -- it wrote qpos
        # without ever consulting `self.stopped`, making it THE ONLY ACTUATOR IN THE REPO that moves
        # an emergency-stopped arm. That was contained while nothing on the live tick called it and
        # it stayed test-only. It stopped being contained the moment a cockpit actuation route was
        # authorised, because a scene disturbance is exactly what a browser would reach for and this
        # is the only function that can apply an external force.
        #
        # `allow_latched` is the deliberate escape, and it is narrow: the trajectory-deviation reflex
        # self-check must shove a latched arm to prove the reflex fires, and refusing there would
        # delete the test rather than the hazard. It is a keyword-only argument so it can never be
        # passed by accident, and every caller that sets it is a self-check, not a live path.
        if self.stopped and not allow_latched:
            return self.state()
        self.data.qpos[spec.qpos_index] += deg * _DEG
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def view(self):
        """Offscreen picture of this body, created on first use.

        Lazy because most bodies are never looked at -- tests, the serial mirror, the benchmark
        -- and building a GL context for them would cost a context each and could fail on a
        headless box for no benefit. Rendering itself is NOT done here: the renderer belongs to
        whichever thread draws first, and that must be the daemon's tick thread, so the caller
        drives it. See substrate/arm_view.py.
        """
        if self._view is None:
            from snn_data_lake.substrate.arm_view import ArmView

            self._view = ArmView()
        return self._view

    def set_substeps(self, n):
        """Set how many physics steps one apply() runs. Bounded, not validated away.

        The ceiling is DERIVED: one second of simulated time. Past that the arm is not slow,
        it is stuck -- a command that has acted for a full second and still has not arrived is
        blocked or mis-specified, and letting the count grow without limit would turn that
        into an unbounded tick instead of a visible failure. The floor is one step, because
        zero steps is not a slower arm, it is a body that does not move while reporting that
        it was commanded.
        """
        ceiling = int(round(1.0 / float(self.model.opt.timestep)))
        self.substeps = max(1, min(int(n), ceiling))
        return self.substeps

    def _saturated(self):
        m, d = self.model, self.data
        lim = m.actuator_forcerange[: m.nu, 1]
        return np.abs(d.actuator_force[: m.nu]) >= lim * (1.0 - 1e-9)

    # ---- the gate ----------------------------------------------------------------------

    def apply(self, command, *, deliberate=False):
        """The ONLY mutation path. Ordered: latch -> vocabulary -> clamp -> actuate -> report.

        `deliberate` (T037a) marks a human's reset -- the cockpit's ticket or chat -- and is the
        only thing that clears a latch RESTORED from a previous process; everything automated
        leaves the default and is refused on such a latch. It changes nothing else."""
        m, d = self.model, self.data

        # 0. TYPE. Anything that is not a string is not a command. Refused as a no-op rather
        #    than raised, so a malformed call cannot be mistaken upstream for a transport error
        #    -- and so the gate matches serial_arm.is_valid_command, which type-checks first.
        if not isinstance(command, str):
            return self.state()

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
            self.latch_source = "EMERGENCY_STOP"
            self.latch_restored_from = None
            self._persist_latch(True)  # T037a: the stop outlives this process
            return self.state()

        if command in _CLEARS_LATCH:
            # T037a: a stop this process did not witness clears only deliberately; refused
            # clears stay frozen and are counted (never silent).
            if self.latch_source == "restored" and not deliberate:
                self.latch_refusals += 1
                return self.state()
            held = self.stopped  # did THIS body hold the stop? only then does it own the sidecar
            self.stopped = False
            self._clamped = False
            self._home()
            self.latch_source = None
            self.latch_restored_from = None
            if held:
                self._persist_latch(False)  # the reset of a stop this body held clears the disk
            return self.state()

        # 2. VOCABULARY.
        self._clamped = False
        # fullmatch, NOT match. With .match, "JOINT_ELBOW_FLEX_ROTATE(+5deg)EMERGENCY_STOP"
        # performed the rotation and SILENTLY DROPPED THE STOP -- measured, stopped stayed False
        # while the elbow moved +0.023346 deg. serial_arm's wire allow-list already used
        # .fullmatch, so the arm's own gate was strictly more permissive than the allow-list it
        # mirrors, which is the one direction a safety gate must never differ.
        hit = _JOINT_TOKEN.fullmatch(command)
        if hit:
            name, verb, deg = hit.group(1).lower(), hit.group(2), float(hit.group(3))
            # Membership comes from TOKENS, not JOINTS. JOINTS includes the gripper; TOKENS does
            # not, because the gripper is driven by ENGAGE/RELEASE. Resolving against JOINTS made
            # JOINT_GRIPPER_ROTATE(+30deg) a valid command that moved the effector (measured
            # 0.090908 -> 0.091143), bypassing the sanctioned branch -- a token the gate accepted
            # and the vocabulary said did not exist.
            spec = _ROTATABLE.get(name)
            if spec is None:
                return self.state()  # a joint that does not exist is rejected, not ignored

            # 3. CLAMP -- against the range that BINDS A POSITION COMMAND, so `clamped` is true
            #    exactly when the command was altered. The physical stop is enforced separately by
            #    the solver and is what probe_contact gates on; the two are measured unequal.
            lim = arm_config.JOINT_LIMITS[name]
            # RELATIVE accumulates on the COMMANDED value, not the achieved one. Accumulating on
            # qpos would fold each tick's servo droop back into the next setpoint, so a repeated
            # command would creep short -- the incumbent had no droop to fold because it
            # teleported, and reproducing its behaviour means accumulating on the command.
            want = self._commanded[spec.qpos_index] + deg * _DEG if verb == "ROTATE" else deg * _DEG
            got = min(max(want, lim["ctrl_lo"]), lim["ctrl_hi"])
            self._clamped = bool(got != want)  # a python bool: the contract says bool
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

        # 4. ACTUATE for as long as the brain has decided a command should act.
        for _ in range(self.substeps):
            mujoco.mj_step(m, d)
        # The arm has settled where the command put it; any later gap is exafference.
        self._settled = np.array(d.qpos[: m.nu], dtype=float)
        return self.state()

    # ---- observation -------------------------------------------------------------------

    @property
    def effector(self):
        """Gripper closure in [0, 1]. Same meaning and same range as ArmSim.effector."""
        return self.state()["effector"]

    def collision(self):
        """The ONE collision authority: REPORTS contact without PREVENTING it.

        Returns a reason string (GROUND / SELF / OUT_OF_LIMIT) or None, exactly as ArmSim did.
        MuJoCo's solver does stop the links interpenetrating, but that is the body being solid
        rather than this gate refusing a command: the command still executes, the arm still
        ends up wherever physics puts it, and the contact is still reported. That is the
        property the incumbent guaranteed and it survives unchanged.
        """
        return self.state()["collision"]

    def deviation(self):
        """Excursion attributable to an external disturbance. See state()['deviation']."""
        return self.state()["deviation"]

    def trajectory_breach(self):
        """True when the arm has moved further from its held pose than droop can explain.

        Compared PER JOINT against the derived droop envelope, not against the scalar norm:
        `_droop_tolerance()` returns one tolerance per joint, so reducing either side to a
        single number would let a large allowance on the shoulder mask a real breach at the
        wrist. ArmSim could compare against one fixed number because it teleported and had no
        droop at all; a position servo does, and the tolerance has to be shaped like the arm.
        """
        m, d = self.model, self.data
        excess = np.abs(d.qpos[: m.nu] - self._settled) - self._droop_tolerance()
        return bool(np.any(excess > 1e-9))

    def close(self):
        """No-op: there is nothing to release.

        Present so lifecycle callers need no edit. MuJoCo's model and data are freed when the
        Python objects are collected, and there is no socket, process or device behind this
        body the way there was behind ArmSim's optional pybullet backend. Dropping the model
        here would be worse than doing nothing -- a consumer that calls close() and then reads
        state() would crash on a body that is in fact still perfectly usable.
        """

    def ee_mm(self):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        return tuple(float(v) * 1000.0 for v in self.data.site_xpos[sid])

    def _effector(self):
        return _effector_ratio(self.data)

    def state(self):
        m, d = self.model, self.data
        err = np.abs(d.qpos[: m.nu] - self._commanded[: m.nu])
        tol = self._droop_tolerance()
        reached = bool(np.all(err <= np.maximum(tol, 1e-9) * 2.0 + 1e-6))
        collision = _verdict(m, d)
        # A saturated actuator means "full force", which is what a servo uses to ACCELERATE.
        # Without the velocity term this reported blocked=True for the whole of an ordinary
        # unobstructed move -- measured, while per-tick displacement was still growing -- and
        # coverage_reward penalises blocked, so the objective paid -0.30 for moving and +0.15
        # for standing still. The arm learned exactly that.
        #
        # Stopped-ness is derived from the droop envelope rather than picked: a joint that
        # cannot clear its own steady-state error within a second is not travelling anywhere.
        # Magnitudes, not per-joint: shoulder_pan's tolerance is exactly 0 (a yaw joint carries
        # no gravity moment), so a per-joint test let 4.4e-12 rad/s of float noise count as
        # motion and cleared `blocked` on an arm sitting in the floor.
        moving = bool(np.linalg.norm(d.qvel[: m.nu]) > np.linalg.norm(self._droop_tolerance()))
        blocked = bool(
            (not reached)
            and (collision is not None or bool(np.any(self._saturated())))
            and not moving
        )
        tracking = float(np.linalg.norm(err))
        return {
            "theta_deg": [float(v) / _DEG for v in d.qpos[: m.nu]],
            "ee_mm": self.ee_mm(),
            # Clamped to [0, 1]. The raw ratio leaves the interval at the rails -- measured
            # -0.00000170 after 500 RELEASE -- because a position servo overshoots slightly and
            # the gripper's commandable range is not its physical range. A negative "closure"
            # is not a physical claim, it is a leaked implementation detail.
            "effector": float(min(1.0, max(0.0, _effector_ratio(d)))),
            "stopped": self.stopped,
            # T037a: a latched arm is never silent about WHY. None / "EMERGENCY_STOP" (this
            # process) / "restored" (another process's stop, honoured at construction).
            "latch_source": self.latch_source,
            "latch_restored_from": self.latch_restored_from,
            "latch_persist_errors": self.latch_persist_errors,
            "latch_refusals": self.latch_refusals,
            "collision": collision,
            "reached": reached,
            "blocked": blocked,
            "clamped": self._clamped,
            # Two quantities, because a position servo makes them different.
            #
            # tracking_error is the expected droop under load: small, bounded, ALWAYS present, and
            # not a fault. deviation is the excursion attributable to an EXTERNAL disturbance.
            # They cannot be separated by magnitude -- measured, an ordinary commanded move peaks
            # at 0.52325 rad of tracking error, LARGER than the 0.34961 from a 20 deg external
            # shove -- so deviation is measured against the pose the servo is actually holding,
            # and is zero whenever nothing has pushed the arm.
            #
            # In normal operation that means deviation == 0.0, because the simulated arm has no
            # external disturbance source. That is a true statement about this body, not a stub;
            # inject_disturbance() is how the reflex is exercised.
            "tracking_error": tracking,
            "deviation": float(np.linalg.norm(d.qpos[: m.nu] - self._settled)),
            "backend": "mujoco",
            "moves": True,
        }


# distinct-from: thalamus/gwt_deep.py -- its _reds fire the binding criteria's reds.
# distinct-from: thalamus/reentrant_workspace.py -- its _reds fire the workspace claims' reds.
# distinct-from: substrate/selfcheck_cache.py -- fires the cache key's reds.
# distinct-from: substrate/manifest_gate.py -- fires the manifest gate's reds.
# distinct-from: cortex/autotelic_goals.py -- a chance baseline, not a red.
# distinct-from: cerebellum/gain_adaptation.py -- a converge-then-reverse run, not a red.
#   Each fires what ITS OWN criteria owe; these are the durable latch's five. All read.
def _durable_latch_checks(tok):
    """T037a: what the durable latch owes, each case constructed and run. Executes under the
    throwaway NORD_HOME main() sets, so the shared gate home is never written."""
    import subprocess
    import sys

    latch = _latch_path()
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    env["PYTHONUTF8"] = "1"

    def other_process_stops():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from snn_data_lake.brainstem.so101_arm import So101Arm; "
                "So101Arm().apply('EMERGENCY_STOP')",
            ],
            check=True,
            env=env,
        )
        assert os.path.exists(latch), "the other process's E-STOP left no sidecar"

    # (f) OWNERSHIP: a body built BEFORE the stop existed never held it, so its routine HOME
    #     leaves the foreign sidecar intact and the next process still comes up stopped
    bystander = So101Arm()
    other_process_stops()
    bystander.apply("HOME")
    assert os.path.exists(latch), "a never-latched body's HOME erased another process's stop"
    assert So101Arm().state()["stopped"] is True, "the stop was lost to a bystander's HOME"

    # (a) RESTART PERSISTS: a stop tripped by ANOTHER process is honoured here, and the
    #     restored arm is frozen to every motion command
    restored = So101Arm()
    st = restored.state()
    assert st["stopped"] is True and st["latch_source"] == "restored", st
    assert st["latch_restored_from"].startswith("pid "), st["latch_restored_from"]
    frozen = st["theta_deg"]
    restored.apply(tok(+30))
    assert restored.state()["theta_deg"] == frozen, "a restored latch let the arm move"

    # (g) AN AUTOMATED CLEAR IS REFUSED AND COUNTED; the sidecar stays
    for cmd in _CLEARS_LATCH:
        restored.apply(cmd)
    st = restored.state()
    assert st["stopped"] is True and st["latch_source"] == "restored", (
        "an automated HOME cleared a restored stop"
    )
    assert st["latch_refusals"] == len(_CLEARS_LATCH), st["latch_refusals"]
    assert os.path.exists(latch), "a refused clear removed the sidecar"

    # (b) THE DELIBERATE RESET CLEARS IT, for the next process too
    restored.apply("HOME", deliberate=True)
    assert restored.state()["stopped"] is False and restored.state()["latch_source"] is None
    assert not os.path.exists(latch), "the deliberate HOME left the sidecar behind"
    assert So101Arm().state()["stopped"] is False, "a cleared latch was restored"

    # (c) THE BOUNDARY, PINNED: a body built in the SAME process after a stop is a new body,
    #     and its HOME does not touch the stop it never held
    first = So101Arm()
    first.apply("EMERGENCY_STOP")
    twin = So101Arm()
    assert twin.state()["stopped"] is False and twin.state()["latch_source"] is None, (
        "a same-process twin restored the latch: the process-token rule is gone"
    )
    twin.apply("HOME")
    assert os.path.exists(latch), "a same-process twin's HOME erased the stop it never held"
    first.apply("HOME")  # the body that tripped it clears it, deliberate or not
    assert not os.path.exists(latch)

    # (d) FAIL CLOSED: an unreadable sidecar is a stop, and says so
    with open(latch, "w", encoding="utf-8") as f:
        f.write("{not json")
    bad = So101Arm()
    assert bad.state()["stopped"] is True, "a corrupt sidecar read as clear"
    assert bad.state()["latch_restored_from"].startswith("unreadable"), bad.state()
    os.remove(latch)

    # (e) A PERSIST FAILURE IS COUNTED, and the in-process latch stands
    fresh = So101Arm()
    os.makedirs(latch)  # a DIRECTORY where the sidecar goes: the write must fail
    try:
        fresh.apply("EMERGENCY_STOP")
        assert fresh.state()["stopped"] is True, "a persist failure weakened the stop"
        assert fresh.state()["latch_persist_errors"] == 1, fresh.state()["latch_persist_errors"]
    finally:
        os.rmdir(latch)


def main():
    # The whole self-check runs under a THROWAWAY home: its E-STOPs write the T037a sidecar,
    # and neither a battery's shared gate home nor the live brain's home may receive it.
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _main()


def _main():
    arm = So101Arm()
    st = arm.state()
    assert len(st["theta_deg"]) == 6, "SO-101 is 6-DOF"
    assert st["backend"] == "mujoco"
    assert st["stopped"] is False and st["latch_source"] is None, "a fresh home came up stopped"

    _durable_latch_checks(lambda deg: f"{arm_config.token_for('shoulder_lift')}({deg:+d}deg)")

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
    print(
        "self-check OK: latch idempotent, HOME reproducible, gate clamps, probe isolated; "
        "T037a durable latch: restart-persists (real second process), automated clear refused "
        "and counted, deliberate reset clears, same-process twin is a new body and owns no "
        "sidecar, a bystander's HOME leaves a foreign stop intact, unreadable sidecar fails "
        "closed, persist failure counted"
    )


if __name__ == "__main__":
    main()
