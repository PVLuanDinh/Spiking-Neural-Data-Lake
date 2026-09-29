"""
demo.py -- Spiking Neural Data Lake -> robot arm, the whole loop in ONE runnable file.

    PYTHONPATH=src python -m snn_data_lake.infra.demo

It walks the full reaction loop on embedded sample sensor data (an ultrasonic-distance +
IR-temperature rig, the reference format):

    sensor (distance, temp)  ->  window  ->  ORDER-AWARE gesture recognition
        ->  Interpreter (gesture -> robot command)  ->  arm motion
        ->  contact reflex (object at the sensor -> EMERGENCY_STOP)

The RECOGNITION half is a self-contained mirror of the real modules (sensorimotor/signal_loop,
sensorimotor/gesture_recognition, substrate/paradigm_b_engine, brainstem/interpreter); the ARM half is not a
mirror at all -- it drives the ONE real arm, `brainstem/so101_arm.So101Arm`.

CORRECTED 2026-08-04. This header opened "Download this single file and run it. No dependencies, no
other files, no hardware", and named `arm_sim` as one of the modules it mirrors:
  * it imports `snn_data_lake.brainstem.so101_arm`, so it is neither single-file nor
    dependency-free (mujoco is an unconditional core dependency -- see P7);
  * `arm_sim` does not exist anywhere in `src/`. That is the same phantom module CLAUDE.md records
    correcting out of its safety-envelope list, surviving here;
  * "no dependencies" as a selling point is exactly the retired zero-dependency rule P7 abolished,
    and this file is where its removal was PAID FOR -- the private arm it used to carry was a third
    E-STOP implementation (POLICIES.md P7, incident 1).

Optional live hardware (needs `pip install pyserial`):
    python -m snn_data_lake.infra.demo --serial COM8
"""

import sys

from snn_data_lake.brainstem import arm_config as _arm_config
from snn_data_lake.brainstem.arm_config import CONTACT  # the ONE contact threshold

W = 8  # samples per window
# CONTACT is IMPORTED, not redefined. This file carried its own `CONTACT = 0.04` -- a second copy of
# a safety-envelope constant (`brainstem/arm_config.py:27`), which P1's fourth exception protects
# precisely BECAUSE it is a safety constant, and which P2 says must exist once. Two copies of a
# reflex threshold is one copy that can drift out of the envelope without anything noticing.
NEAR, MID, FAR = 0, 1, 2

# gesture -> robot command (approach grabs the incoming object, retreat releases)
COMMANDS = {"HAND_APPROACH": "ENGAGE", "HAND_RETREAT": "RELEASE", "IDLE": "HOLD"}


# ---- order-aware recognition (Paradigm B sequence detector) ------------------
def dbin(d):
    if d >= 2.0:
        return None  # nothing in range
    return NEAR if d < 0.30 else MID if d < 0.60 else FAR


def match_sequence(events, ordered, window):
    """Do `ordered` channels fire IN ORDER within `window`? (the Paradigm B detector:
    far->near matches approach, near->far matches retreat; reverse does not match)."""
    L, matches, cands = len(ordered), [], []
    for t, ch in events:
        cands = [c for c in cands if t - c[0] <= window]
        nxt = []
        for t0, stage in cands:
            if ch == ordered[stage]:
                (matches.append(t) if stage + 1 == L else nxt.append((t0, stage + 1)))
            else:
                nxt.append((t0, stage))
        cands = nxt
        if ch == ordered[0]:
            matches.append(t) if L == 1 else cands.append((t, 1))
    return matches


def recognize(window):
    present = [(i, dbin(d)) for i, (d, _) in enumerate(window) if dbin(d) is not None]
    if len(present) < max(2, len(window) // 2):
        return "IDLE"
    appr = match_sequence(present, [FAR, MID, NEAR], W)
    retr = match_sequence(present, [NEAR, MID, FAR], W)
    if appr and not retr:
        return "HAND_APPROACH"
    if retr and not appr:
        return "HAND_RETREAT"
    pd = [window[i][0] for i, _ in present]  # noisy: fall back to net direction
    return "HAND_APPROACH" if pd[-1] < pd[0] else "HAND_RETREAT"


# ---- 2-link arm (forward kinematics + effector + collision) ------------------
class _RetiredArm:
    """RETIRED (specs/004). Kept only as a marker so the reason survives the deletion.

    This was a private 2-link arm with its OWN apply(), ee(), effector and stopped -- a third
    implementation of the safety gate, with a second E-STOP that no test ever exercised. It
    existed so this demo could run without dependencies; the zero-dependency rule was removed
    project-wide in 2026-07 and mujoco is a core dependency now, so that reason no longer
    describes anything. The demo drives the one real arm instead.

    THE BODIES ARE GONE TOO, 2026-08-04. `ee()` and `apply()` survived the retirement -- including
    the `if cmd == "EMERGENCY_STOP": self.stopped = True` branch, i.e. the second E-STOP itself,
    still on disk inside the marker announcing its removal. `__init__` raising makes them
    unreachable by construction, not absent: `_RetiredArm.apply(obj, "EMERGENCY_STOP")` still ran
    them. P2 says count the survivors, and a retirement marker needs its reason, not its code.
    """

    def __init__(self, *_a, **_k):
        raise RuntimeError(
            "demo.Arm is retired: there is exactly one arm (brainstem/so101_arm.So101Arm). "
            "Constructing a private arm here would re-implement the safety gate."
        )


# ---- sample data (embedded: the reference gesture shapes) ---------------
def sample_stream():
    """A scripted run: idle, an approach, a retreat, then a contact event."""
    idle = [(3.13, -1.0)] * W
    approach = [(round(0.9 - 0.115 * i, 2), 295.0 + i * 0.4) for i in range(W)]  # distance falling
    retreat = [(round(0.05 + 0.115 * i, 2), 298.0 - i * 0.4) for i in range(W)]  # distance rising
    contact = [(0.02, 298.6)]  # object AT sensor
    for label, win in (
        ("(quiet)", idle),
        ("(hand approaching)", approach),
        ("(hand retreating)", retreat),
    ):
        for s in win:
            yield label, s
    for s in contact:
        yield "(contact!)", s


def serial_stream(port, baud=115200):
    import serial  # only for --serial

    ser = serial.Serial(port, baud)
    while True:
        parts = ser.readline().decode("ascii", "ignore").strip().split(",")
        if len(parts) >= 2:
            try:
                yield "(live)", (float(parts[-2]), float(parts[-1]))
            except ValueError:
                continue


def run(stream, arm, narrate=True):
    buf, acted = [], []
    for note, (d, t) in stream:
        if d < CONTACT:  # reflex preempts everything
            arm.apply("EMERGENCY_STOP")
            acted.append(("REFLEX", "EMERGENCY_STOP"))
            if narrate:
                print(f"  !! {note} distance={d}m -> REFLEX EMERGENCY_STOP (arm frozen)")
            buf = []
            continue
        buf.append((d, t))
        if len(buf) == W:
            gesture = recognize(buf)
            # apply() returns the arm STATE, not the command. The retired demo Arm returned
            # the string, which is why this looked like a one-liner.
            cmd = COMMANDS[gesture]
            arm.apply(cmd)
            acted.append((gesture, cmd))
            if narrate:
                print(
                    f"  {note:20} -> recognize {gesture:14} -> command {cmd:14} "
                    f"(effector={arm.effector:.3f}, ee_mm={tuple(round(v, 1) for v in arm.ee_mm())})"
                )
            buf = []
    return acted


def main():
    # T037a: this self-check drives a So101Arm through EMERGENCY_STOP / HOME, which now writes
    # the durable latch sidecar -- under a throwaway home, never the live brain's.
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _main()


def _main():
    args = sys.argv[1:]
    from snn_data_lake.brainstem.so101_arm import So101Arm

    arm = So101Arm()
    # Let a command complete: this demo shows the gesture -> command -> motion loop, and a
    # single 2 ms physics step would move the joint 0.02 deg and make the loop look inert.
    arm.set_substeps(10**9)
    print("=" * 70)
    print("SPIKING NEURAL DATA LAKE  ->  ROBOT ARM   (one-file demo)")
    print("=" * 70)
    print("sensor (distance, temp) -> window -> recognize -> command -> arm  (+ reflex)\n")

    if "--serial" in args:
        port = args[args.index("--serial") + 1]
        print(f"live off {port}. Ctrl-C to stop.\n")
        run(serial_stream(port), arm)
        return

    acted = run(sample_stream(), arm)
    print("\n" + "=" * 70)
    cmds = [c for _, c in acted]
    print(f"actions: {', '.join(cmds)}")
    print(
        "approach grabbed (ENGAGE), retreat released (RELEASE), "
        "contact stopped the arm (EMERGENCY_STOP)."
    )
    print("=" * 70)

    # ---- self-check (so the demo is verifiable) -----------------------------
    seen = {c for _, c in acted}
    assert "ENGAGE" in seen, "approach did not grab"
    assert "RELEASE" in seen, "retreat did not release"
    assert "EMERGENCY_STOP" in seen and arm.stopped, "contact did not trip the reflex"

    # THE RETIRED ARM CARRIES NO BODY. `_RetiredArm.apply()` and `.ee()` survived the retirement
    # with their code intact -- including `if cmd == "EMERGENCY_STOP": self.stopped = True`, the
    # SECOND E-STOP that POLICIES.md P7 names as its sharpest incident -- inside the marker
    # announcing their removal. A raising `__init__` made them unreachable by construction, not
    # absent: `_RetiredArm.apply(obj, "EMERGENCY_STOP")` still executed them. Deleted 2026-08-04;
    # this assert is what stops them coming back, because `tests/test_arm_vocabulary.py` only
    # checks that CONSTRUCTION raises and would pass with either body present.
    # RED, CONSTRUCTED AND RUN: restoring either method makes this fire.
    _leftover = [m for m in ("apply", "ee", "theta", "l1", "l2") if m in vars(_RetiredArm)]
    assert not _leftover, (
        f"_RetiredArm still carries {_leftover} -- a retirement marker keeps its REASON, never its"
        " code, and the code here is a second implementation of the arm safety gate"
    )
    assert CONTACT is _arm_config.CONTACT, (
        "CONTACT must stay the ONE imported safety constant, never a local copy that can drift"
    )

    print(
        "self-check OK: recognized gestures, mapped to commands, moved the ONE arm, reflex stopped "
        "on contact; the retired private arm carries no body and CONTACT is imported, not copied."
    )


if __name__ == "__main__":
    main()
