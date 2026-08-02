"""
v0.50 — robot-arm calibration in ONE place.

Every distance/IR threshold the arm pipeline depends on was hardcoded across
gesture_recognition.py and live_arm.py, tuned to one specific reference sensor mount.
Move the rig + the 95% recognition collapses. This centralizes them so a re-mount is a
one-file edit, and `bins_from_capture()` can derive them from real data instead of guessing.

Units: distance in metres (ultrasonic / Sensor_1). IR = Sensor_2 (raw analog, polarity
sensor-specific — see IR_PRESENT below).

  python arm_config.py        # self-check
"""

import pathlib as _pathlib
from collections import namedtuple as _namedtuple
from functools import cache as _cache
from functools import lru_cache as _lru_cache

# ---- distance bins (metres) ----------------------------------------------------
# NEAR/MID/FAR channels feed Paradigm B's order-aware sequence detector.
NEAR_MAX = 0.30  # d <  NEAR_MAX            -> NEAR
MID_MAX = 0.60  # NEAR_MAX <= d < MID_MAX  -> MID   (else FAR)
IN_RANGE_MAX = 2.0  # d >= IN_RANGE_MAX        -> no object present (idle)

# ---- reflex --------------------------------------------------------------------
CONTACT = 0.04  # d < CONTACT -> imminent-contact reflex EMERGENCY_STOP
TRAJ_DEV_MAX = 0.15  # end-effector deviation (m) from the intended pose that trips the
# trajectory-deviation reflex (the "differential position comparator":
# an external perturbation knocks the arm off its commanded path).
# tied to arm reach (l1+l2=2m); re-tune per real rig/effector.

# ---- real-hardware safety gate -------------------------------------------------
# CONTACT / joint-limit / reflex values here are PLACEHOLDERS tuned to the sim, NOT your rig.
# serial_arm.SerialArm refuses to drive real servos until this is True. Flip it ONLY after
# replacing them with your arm's measured limits (docs/arduino_requirements.md #3).
SAFETY_CALIBRATED = False

# ---- windowing -----------------------------------------------------------------
W = 8  # samples per recognition window
STRIDE = W  # W = non-overlapping; < W = sliding (lower live latency)

# ---- IR fusion (Sensor_2) ------------------------------------------------------
# IR polarity/scale is sensor-specific and uncalibrated, so fusion is OFF by
# default (None) — distance alone decides presence, exactly as before. Ceiling: idle vs
# far-but-present object is distance-only. Upgrade: capture IR with/without a hand in
# range, set IR_PRESENT to (">", v) or ("<", v) so IR confirms presence.
IR_PRESENT = None  # e.g. (">", 500) means "present when Sensor_2 > 500"

NEAR, MID, FAR = 0, 1, 2


def dbin(d):
    """Distance -> NEAR/MID/FAR channel, or None when out of range (idle)."""
    if d >= IN_RANGE_MAX:
        return None
    if d < NEAR_MAX:
        return NEAR
    if d < MID_MAX:
        return MID
    return FAR


def present(d, ir=None):
    """Is an object in range this sample? Distance-primary; IR confirms iff calibrated."""
    by_dist = d < IN_RANGE_MAX
    if IR_PRESENT is None or ir is None:
        return by_dist
    op, thr = IR_PRESENT
    by_ir = ir > thr if op == ">" else ir < thr
    return by_dist and by_ir  # both agree -> present (IR rejects distance ghosts)


def bins_from_capture(distances, lo=0.33, hi=0.66):
    """Derive NEAR/MID cut points from a real capture's in-range distances (percentiles),
    so the bins match the rig instead of magic constants. Returns (near_max, mid_max)."""
    xs = sorted(d for d in distances if d < IN_RANGE_MAX)
    if len(xs) < 3:
        return NEAR_MAX, MID_MAX
    return xs[int(lo * (len(xs) - 1))], xs[int(hi * (len(xs) - 1))]


# ---- the SO-101 body: joint vocabulary, limits, home pose ------------------------------
# v4.81 -- the arm becomes the MuJoCo SO-101 (specs/004). This is the ONE place the joint
# vocabulary and the joint limits are defined, so a token, a limit and a servo row cannot
# drift apart.
#
# P1: the LIMITS ARE NOT WRITTEN HERE. They are read from the vendored model, which is the
# only thing that actually constrains the body -- transcribing six pairs of floats into this
# file would create a second source of truth that silently goes stale when the model is bumped.
# Only alias / servo / lever_mm are hand-written, because they describe the PHYSICAL build
# target and have no counterpart in the simulation model.
#
# LOADED LAZILY, on purpose. Loading the model costs ~134 ms and this module is imported by a
# wide fan-in (arm_sim, serial_arm, autonomous_loop, live_arm, fear_conditioning, ...), while
# run_selfchecks spawns one subprocess per module. Paying 134 ms at import would tax every one
# of them for a value most never read. Provenance is what P1 asks for, not import timing.

_SO101_SCENE = _pathlib.Path(__file__).with_name("so101") / "scene.xml"

#: One row per joint, in qpos order. `name` is the MJCF joint name and is the CANONICAL
#: identity -- it is the only name that also exists outside this repo and must resolve through
#: mj_name2id, so renaming it would fork the vendored asset. `alias` is the build-target name
#: used by arm_hardware's torque table.
JointSpec = _namedtuple("JointSpec", "qpos_index name alias servo lever_mm")

JOINTS = (
    JointSpec(0, "shoulder_pan", "base_yaw", "ST3215-1:345", 0.0),
    JointSpec(1, "shoulder_lift", "shoulder_pitch", "ST3215-1:345", 300.0),
    JointSpec(2, "elbow_flex", "elbow_pitch", "ST3215-1:191", 180.0),
    JointSpec(3, "wrist_flex", "wrist_pitch", "ST3215-1:191", 60.0),
    JointSpec(4, "wrist_roll", "wrist_roll", "ST3215-1:147", 0.0),
    JointSpec(5, "gripper", "gripper", "SC-series", 0.0),
)


#: The rotate token for a joint. The REGEX that parses these is syntax; THIS TABLE is the
#: vocabulary. A site that matches the shape without checking membership here can mint a token
#: for a joint that does not exist, which is how the 2-DOF vocabulary came to be defined in
#: eight places that disagreed.
def token_for(name):
    """MJCF joint name -> its RELATIVE rotate token (prefix only; add the (+/-Ndeg) suffix)."""
    return f"JOINT_{name.upper()}_ROTATE"


def moveto_token_for(name):
    """MJCF joint name -> its ABSOLUTE setpoint token (prefix only)."""
    return f"JOINT_{name.upper()}_MOVETO"


def command_for(name, deg):
    """A complete RELATIVE command: rotate this joint BY `deg` degrees.

    Producers must call this rather than token_for(): token_for returns the PREFIX only, and a
    bare prefix falls through the gate's no-op branch -- a silently ignored command, which is
    the exact failure this vocabulary exists to remove. It also fixes the suffix format in one
    place instead of four different float formats across the call sites.
    """
    return f"{token_for(name)}({deg:+g}deg)"


def moveto_for(name, deg):
    """A complete ABSOLUTE command: drive this joint TO `deg` degrees."""
    return f"{moveto_token_for(name)}({deg:+g}deg)"


#: The joints a JOINT_* token may name. Excludes the gripper, which is driven by
#: ENGAGE/RELEASE and has no angle vocabulary. Producers import THIS rather than
#: re-deriving `!= "gripper"`, so "which joints are addressable" is decided once.
ROTATABLE = tuple(j.name for j in JOINTS if j.name != "gripper")

#: The default step for a producer with no reason to choose its own magnitude.
#:
#: The incumbent scattered these: +15 deg at the interpreter, -10 for the second joint, 6.0
#: in the autonomous loop, each formatted differently. They were arbitrary -- the interpreter
#: said so in its own comment -- and being arbitrary in four places made them four things to
#: keep in agreement instead of one. This is a command magnitude, not a behaviour-arbitrating
#: threshold, so a single shared default is the honest form; the thresholds that DO arbitrate
#: live in JOINT_LIMITS and are measured off the model.
DEFAULT_STEP_DEG = 6.0

#: The ROTATE tokens. Excludes the gripper, which is driven by ENGAGE/RELEASE.
TOKENS = tuple(token_for(n) for n in ROTATABLE)

#: The MOVETO tokens. Same joints, absolute semantics.
#:
#: TWO VERBS, DELIBERATELY. ROTATE is relative -- it adds to the current setpoint, which is what
#: the incumbent arm did and what four control loops assume when they compute delta =
#: target - current and send it as a token. MOVETO is absolute, for a caller that genuinely wants
#: a setpoint. Collapsing them into one verb was a silent semantic change: an absolute-only
#: ROTATE turns every one of those loops into a non-controller, and reduces the autonomous
#: repertoire to two fixed setpoints per joint (measured: 5 distinct 100 mm cells over 3000
#: ticks, so the arm's only dense reward can never fire).
MOVETO_TOKENS = tuple(moveto_token_for(n) for n in ROTATABLE)

#: The model carries no keyframe (nkey == 0), so the home pose is DECLARED by this repo rather
#: than read from the asset. All-zeros is the model's own rest configuration. HOME is applied as
#: a hard reset, never as "command zeros and let it settle" -- settling is not bitwise
#: reproducible across start poses, and the home assertion depends on it being so.
HOME_QPOS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@_lru_cache(maxsize=1)
def _model():
    """The vendored SO-101 model. Cached; see the lazy-loading note above."""
    import mujoco  # local: keeps the ~134 ms off every import of this module

    return mujoco.MjModel.from_xml_path(str(_SO101_SCENE))


@_lru_cache(maxsize=1)
def _joint_limits():
    """Per-joint limits, read from the model.

    TWO ranges, deliberately kept apart because they are MEASURED UNEQUAL on this model
    (shoulder_pan: stop +/-1.91986218 vs ctrl +/-1.91986):

      ctrl_lo/ctrl_hi -- actuator_ctrlrange: what BINDS A POSITION COMMAND. The gate clamps
                         to this, so `clamped` is true exactly when the command was altered.
      stop_lo/stop_hi -- jnt_range: the PHYSICAL STOP, enforced by the solver as a constraint.
                         The contact probe gates on this, because a probe that ignores it
                         reports out-of-range poses as collision-free.

    Conflating them is a defect: clamping to the stop lets a command through that the actuator
    then silently pins at the rail, which is the silent acceptance the spec forbids.
    """
    m = _model()
    out = {}
    for j in JOINTS:
        out[j.name] = {
            "ctrl_lo": float(m.actuator_ctrlrange[j.qpos_index][0]),
            "ctrl_hi": float(m.actuator_ctrlrange[j.qpos_index][1]),
            "stop_lo": float(m.jnt_range[j.qpos_index][0]),
            "stop_hi": float(m.jnt_range[j.qpos_index][1]),
        }
    return out


@_lru_cache(maxsize=1)
def _max_reach_mm():
    """Greatest horizontal distance from the base the gripper site can reach, in mm.

    SAMPLED from the model, not transcribed. There is no closed form: the joint limits stop
    the chain straightening, so the sum of the link lengths is an upper bound the arm cannot
    actually attain. A sweep over the limits is the honest answer, and it is what any
    workspace discretisation downstream has to be scaled against -- a cell size in metres
    means nothing except as a fraction of the reach it divides up.

    Deterministic (fixed seed, fixed count) so two runs discretise the workspace identically.
    A coverage count that moved between runs because the grid shifted under it would not be
    readable as progress. Lazy and cached, because most importers never touch the workspace.

    mj_kinematics only: this places the bodies without integrating or solving contact, so it
    cannot disturb any live arm and costs a fraction of a step.
    """
    import mujoco
    import numpy as np

    m = _model()
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    lo, hi = m.jnt_range[: m.nu, 0], m.jnt_range[: m.nu, 1]
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(4000):
        d.qpos[: m.nu] = lo + rng.random(m.nu) * (hi - lo)
        mujoco.mj_kinematics(m, d)
        x, y = d.site_xpos[sid][0], d.site_xpos[sid][1]
        best = max(best, float((x * x + y * y) ** 0.5))
    return best * 1000.0


@_cache
def ground_free_band(name):
    """(lo_deg, hi_deg): the range over which this joint ALONE keeps the gripper off the floor.

    Swept from the model at one-degree resolution with every other joint at home, which is
    exactly the condition a single-joint rhythm runs under. MEASURED rather than chosen: which
    way a joint points, and how far it can go before the gripper reaches the floor, are facts
    about the geometry that change with the body. The incumbent's equivalent numbers were
    written for a 2-link arm and silently described the wrong half of this one's travel.

    mj_forward only -- placement and contact, no integration, so this cannot disturb anything.
    """
    import mujoco
    import numpy as np

    m = _model()
    d = mujoco.MjData(m)
    spec = next(j for j in JOINTS if j.name == name)
    lim = _joint_limits()[name]
    lo_deg = int(np.ceil(np.rad2deg(lim["ctrl_lo"])))
    hi_deg = int(np.floor(np.rad2deg(lim["ctrl_hi"])))
    ok = []
    for deg in range(lo_deg, hi_deg + 1):
        d.qpos[:] = 0.0
        d.qpos[spec.qpos_index] = np.deg2rad(deg)
        mujoco.mj_forward(m, d)
        grounded = any(
            0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) for c in d.contact[: d.ncon]
        )
        if not grounded:
            ok.append(deg)
    if not ok:  # a joint with no clear air is a fact worth raising, not silently defaulting
        raise ValueError(f"{name}: no ground-free angle in [{lo_deg}, {hi_deg}] deg")
    return (float(min(ok)), float(max(ok)))


def __getattr__(name):
    """PEP 562 lazy attribute so `arm_config.JOINT_LIMITS` reads naturally without the model
    being loaded by every importer of this module."""
    if name == "JOINT_LIMITS":
        return _joint_limits()
    if name == "MAX_REACH_MM":
        return _max_reach_mm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main():
    assert dbin(0.1) == NEAR and dbin(0.45) == MID and dbin(0.9) == FAR
    assert dbin(3.0) is None, "out-of-range must read as idle"
    assert present(0.5) and not present(2.5), "distance presence broken"
    # IR disabled by default -> distance alone decides
    assert present(0.5, ir=0) is True, "IR must be ignored while uncalibrated"
    near, mid = bins_from_capture([0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9])
    assert 0.0 < near <= mid < IN_RANGE_MAX, f"derived bins invalid: {near},{mid}"
    print("=" * 56)
    print("ARM CONFIG — calibration in one place")
    print("=" * 56)
    print(f"bins (m)   : NEAR<{NEAR_MAX}  MID<{MID_MAX}  in-range<{IN_RANGE_MAX}")
    print(f"reflex     : CONTACT={CONTACT} m   window W={W} stride={STRIDE}")
    print(f"IR fusion  : {'OFF (distance-only)' if IR_PRESENT is None else IR_PRESENT}")
    print(f"derived bins from sample capture: NEAR<{near:.3f} MID<{mid:.3f}")
    # ---- SO-101 joint table -------------------------------------------------------
    assert len(JOINTS) == 6, f"expected 6 joints, got {len(JOINTS)}"
    assert [j.qpos_index for j in JOINTS] == list(range(6)), "JOINTS must be in qpos order"
    assert len(set(j.name for j in JOINTS)) == 6, "duplicate MJCF joint name"
    assert len(HOME_QPOS) == 6, "HOME_QPOS must have one entry per joint"
    lim = _joint_limits()
    assert set(lim) == {j.name for j in JOINTS}, "limits and vocabulary disagree"
    for j in JOINTS:
        r = lim[j.name]
        assert r["ctrl_lo"] < r["ctrl_hi"], f"{j.name}: empty ctrl range"
        assert r["stop_lo"] < r["stop_hi"], f"{j.name}: empty stop range"
        assert r["ctrl_lo"] <= HOME_QPOS[j.qpos_index] <= r["ctrl_hi"], (
            f"{j.name}: home pose is outside the commandable range"
        )
    print(f"joints     : {len(JOINTS)} (MJCF names, limits read from the model)")
    print(f"tokens     : {len(TOKENS)} rotate tokens + ENGAGE/RELEASE for the gripper")
    _pan = lim["shoulder_pan"]
    print(
        "limits     : ctrl and stop are DISTINCT -- shoulder_pan ctrl_hi="
        f"{_pan['ctrl_hi']:.8f} vs stop_hi={_pan['stop_hi']:.8f}"
    )
    print("self-check OK: bins, presence, IR-gating, capture-derived bins, SO-101 joint table")


if __name__ == "__main__":
    main()
