"""
v0.47 — live real-hardware loop: ESP32 sensor -> recognize -> Interpreter -> arm sim.

Closes the loop end to end on the reference sensor:

  (d, t) samples  ->  buffer W  ->  ORDER-AWARE gesture recognition (Paradigm B, v0.46)
                  ->  Interpreter (gesture -> robot command)  ->  arm_sim (joint / effector motion)
  + a contact reflex that EMERGENCY_STOPs the arm the instant the object is at the sensor.

Sources (same downstream, sim-to-real):
  python live_arm.py                 # replay the reference labeled CSVs (hardware-free) + self-check
  python live_arm.py --serial COM8   # live off the ESP32 (needs `pip install pyserial`)
  python live_arm.py --csv FILE      # replay one capture

Opt-in arm primitives (OFF by default; every motion still goes through the safety-gated apply()):
  python live_arm.py --serial COM8 --proprioception  # + encoder-free joint estimate & slip watchdog (read-only)
  python live_arm.py --reach 30      # spiking-PID drive joint A to 30 deg (spiking_pid.drive_to)
  python live_arm.py --sweep         # CPG rhythmic bounded joint sweep (cpg_oscillator.cpg_sweep)
python live_arm.py --plan-xyz X Y Z  # sampling planner to a Cartesian goal, in millimetres

Gesture -> command map (sensor domain): approach = grab the incoming object, retreat = release.
"""

import math
import os
import sys
import time

from snn_data_lake import signal_loop as S
from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.arm_config import (  # reflex distance + window stride: calibration in arm_config
    CONTACT,
    STRIDE,
)
from snn_data_lake.brainstem.interpreter import Interpreter
from snn_data_lake.brainstem.so101_arm import So101Arm
from snn_data_lake.sensorimotor.efferent_boundary import intended_deltas
from snn_data_lake.sensorimotor.gesture_recognition import W, classify, load_csv
from snn_data_lake.substrate import home

SENSOR_COMMANDS = {
    "HAND_APPROACH": "ENGAGE",  # grab the approaching object
    "HAND_RETREAT": "RELEASE",  # release as it leaves
    "IDLE": "HOLD",
}

SLIP_DEG = 15.0  # |estimate - actual| above this flags an uncommanded slip
#: The joint the proprioceptive observer tracks. The observer estimates ONE joint angle, so
#: it has to name which; the incumbent's joint A was the shoulder pitch, and shoulder_lift is
#: that same joint on this body. Bound by name because the qpos INDEX moved: index 0 is now
#: shoulder_pan, a yaw joint, and an unchanged `[0]` would have silently watched the wrong one.
_OBS_JOINT = "shoulder_lift"
_OBS_INDEX = next(j.qpos_index for j in arm_config.JOINTS if j.name == _OBS_JOINT)


# consolidates: sensorimotor/efferent_boundary.py -- the private shoulder_lift regex this
#   function carried is DELETED and the parse is that module's `intended_deltas`, which
#   parses every rotatable joint. One parser survives (P2, specs/022 T079 2.1).
# distinct-from: faculties/ring_attractor.py::ProprioceptiveEstimator -- the CONSUMER of
#   this number (it integrates commanded deltas into an encoder-free angle estimate). This
#   only reads the delta out of a token; it integrates nothing and holds no state. Read.
def _joint_obs_delta(command):
    """Commanded delta (deg) for the observed joint; 0.0 for effector / HOLD / stop.

    ONE PARSER (P2, specs/022 T079 2.1): this was a private `JOINT_SHOULDER_LIFT_(ROTATE|
    MOVETO)` regex; the per-joint intended delta T079 needs is the same parse over every
    joint, so the parse moved to `substrate/efferent_boundary.intended_deltas` and this is
    a projection of it. Every ROTATE token reads bit-identically to the retired regex.

    ONE BEHAVIOUR CHANGED, and it is a correction, disclosed rather than preserved: the old
    regex also matched `JOINT_SHOULDER_LIFT_MOVETO(<N>deg)` and returned N as a DELTA.
    MOVETO is the ABSOLUTE verb (arm_config's two-verb note), so N there is a target ANGLE,
    and handing it to `ProprioceptiveEstimator.command(delta)` -- this function's one call
    site, in `run()` -- integrated a setpoint as a movement. It now reads 0.0. Unreachable
    on this module's own path either way: `run()` issues only SENSOR_COMMANDS (ENGAGE /
    RELEASE / HOLD) and the reflex's EMERGENCY_STOP, none of which is a joint token."""
    return intended_deltas(command)[_OBS_JOINT]


def _make_observer():
    """Opt-in read-only proprioceptive observer: encoder-free joint-angle estimate + slip watchdog."""
    from snn_data_lake.faculties.ring_attractor import ProprioceptiveEstimator

    return ProprioceptiveEstimator()


def serial_samples(port, baud=115200, reconnect=True):
    """(distance, ir) from the ESP32; strips banner / non-numeric lines. Real hardware
    drops the link — reconnect with backoff instead of crashing the arm loop."""
    backoff = 0.5
    while True:
        try:
            for line in S.serial_lines(port, baud):
                v = S.parse_line(line)
                if v and len(v) >= 2:
                    yield v[0], v[1]
            return  # generator ended cleanly (e.g. EOF)
        except Exception as e:  # SerialException, decode, device unplug
            if not reconnect:
                raise
            sys.stderr.write(f"serial {port} dropped ({e}); reconnecting in {backoff:.1f}s\n")
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)  # cap backoff at 5s


def _serial_samples_from(ser):
    """(d, ir) read from an ALREADY-OPEN serial handle — used when sensor-out and command-in
    share ONE UART (live_arm --serial COMx --actuate COMx opens the port once, reads + writes
    on the same handle). readline() returns b'' on timeout; keep waiting for the next line."""
    while True:
        raw = ser.readline()
        if not raw:
            continue
        v = S.parse_line(raw.decode("utf-8", "ignore"))
        if v and len(v) >= 2:
            yield v[0], v[1]


def replay_samples(paths):
    for p in paths:
        yield from load_csv(p)


def run(samples, arm, interp, emit, stride=STRIDE, observer=None):
    """Stream (d,ir) -> windowed recognition -> command -> arm. Returns action stats.
    stride < W slides the window (acts every `stride` samples) for lower live latency;
    stride == W (default) is non-overlapping, identical to before.

    observer (opt-in, default None -> loop is byte-for-byte unchanged): a read-only
    ProprioceptiveEstimator. Each COMMANDED joint delta is fed to it and the encoder-free
    joint-angle estimate + slip mismatch are logged. It issues NO command and never touches
    the arm, so collision / EMERGENCY_STOP / reflex still gate every motion."""
    buf = []
    stats = {"windows": 0, "ENGAGE": 0, "RELEASE": 0, "HOLD": 0, "reflex": 0}
    for d, t in samples:
        if (
            not math.isfinite(d) or d < CONTACT
        ):  # reflex preempts cognition (fail-closed on non-finite)
            cmd, _ = interp.interpret({"reflex": "STOP"})
            arm.apply(cmd)
            stats["reflex"] += 1
            buf = []
            emit({"reflex": "STOP", "command": cmd, "ee_mm": arm.state()["ee_mm"]})
            continue
        buf.append((d, t))
        if len(buf) == W:
            gesture = classify(buf)
            cmd, _ = interp.interpret({"match": gesture})
            arm.apply(cmd)
            stats["windows"] += 1
            stats[cmd] = stats.get(cmd, 0) + 1
            ev = {
                "gesture": gesture,
                "command": cmd,
                "effector": arm.effector,
                "ee_mm": arm.state()["ee_mm"],
            }
            if observer is not None:  # opt-in read-only proprioception
                est = observer.command(_joint_obs_delta(cmd))
                mism = observer.mismatch_deg(arm.state()["theta_deg"][_OBS_INDEX])
                ev["joint_est_deg"] = round(est, 1)
                ev["joint_mismatch_deg"] = round(mism, 1)
                ev["slip"] = mism > SLIP_DEG  # joint moved WITHOUT a command
            emit(ev)
            buf = buf[stride:]  # slide (stride==W -> full reset)
    return stats


def main():
    # T037a: this self-check drives a So101Arm through EMERGENCY_STOP / HOME, which now writes
    # the durable latch sidecar -- under a throwaway home, never the live brain's.
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _main()


def _main():
    args = sys.argv[1:]
    interp = Interpreter(commands=SENSOR_COMMANDS)

    # opt-in (snn-arm --persist): attach the shared durable home so the arm boots aware of its lifelong
    # PERCEPTUAL memory. SAFETY: this attaches memory only -- the safety-gated apply() / run() path is
    # untouched and NO remembered motor command is ever replayed at boot (an arm must never move itself
    # from memory). Banners go to stderr so the stdout JSON event stream stays clean.
    if "--persist" in args:
        sys.stderr.write(home.banner("live-arm", knowledge=False) + "\n")
        sys.stderr.write(
            "[persist] durable home attached for perceptual memory ONLY; the safety-gated "
            "apply() is untouched and no remembered motor command is ever replayed (safety).\n"
        )

    def emit(d):
        print(__import__("json").dumps(d))
        sys.stdout.flush()

    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else STRIDE
    actuate = args[args.index("--actuate") + 1] if "--actuate" in args else None
    proprio = "--proprioception" in args  # opt-in read-only observer (default off)

    # opt-in: drive joint A to a target with the spiking PID, THROUGH the safety-gated apply() (off by default)
    if "--reach" in args:
        from snn_data_lake.faculties.spiking_pid import drive_to

        target = float(args[args.index("--reach") + 1])
        errs, final, breaches = drive_to(target)  # drive_to uses So101Arm().apply() (gated)
        print(
            f"reach: spiking PID -> joint A {final:.2f} deg (target {target:.1f}) via gated apply(); "
            f"start err {errs[0]:+.1f} deg, trajectory breaches {breaches}"
        )
        return

    # opt-in: rhythmic bounded joint sweep from the CPG, THROUGH the safety-gated apply() (off by default)
    if "--sweep" in args:
        from snn_data_lake.faculties.cpg_oscillator import cpg_sweep

        trace = cpg_sweep(So101Arm())  # cpg_sweep drives arm.apply() (gated); stops on collision
        print(
            f"sweep: CPG rhythm -> joint A in [{min(trace):.1f}, {max(trace):.1f}] deg via gated apply() "
            f"({len(trace)} steps)"
        )
        return

    # opt-in: plan a collision-free path to a Cartesian end-effector goal in MILLIMETRES
    # (--plan-xyz X Y Z) or to a joint configuration (--plan a b c d e, degrees), executed
    # THROUGH the safety-gated apply(). A sampling planner in configuration space; off by
    # default, adds no new command token, moves via the MOVETO verb already allow-listed.
    #
    # xy became xyz because the workspace did: three of the arm's joints move the gripper
    # vertically, so a planar target names a line rather than a point.
    if "--plan-xyz" in args or "--plan" in args:
        from snn_data_lake.sensorimotor.wavefront_planner import (
            JOINTS as PLAN_JOINTS,
        )
        from snn_data_lake.sensorimotor.wavefront_planner import (
            plan_to_config,
            plan_to_xyz,
        )

        arm = So101Arm()
        arm.set_substeps(10**9)  # a CLI move should complete, not crawl one 2 ms step
        if "--plan-xyz" in args:
            i = args.index("--plan-xyz")
            gx, gy, gz = (float(args[i + 1]), float(args[i + 2]), float(args[i + 3]))
            res = plan_to_xyz(arm, gx, gy, gz)
            goal = f"EE ({gx:.1f}, {gy:.1f}, {gz:.1f}) mm"
        else:
            i = args.index("--plan")
            cfg = [float(args[i + 1 + k]) for k in range(len(PLAN_JOINTS))]
            res = plan_to_config(arm, cfg)
            goal = "config " + ", ".join(f"{v:.1f}" for v in cfg) + " deg"
        ex, ey, ez = arm.ee_mm()
        if not res.get("planned"):
            print(f"plan: {goal} -> REFUSED ({res.get('reason')})")
            return
        print(
            f"plan: {goal} -> {res['reached']}/{res['waypoints']} waypoints reached, "
            f"{res['applied']} gated commands; final EE "
            f"({ex:.1f}, {ey:.1f}, {ez:.1f}) mm via gated apply()"
        )
        return

    if "--serial" in args:
        port = args[args.index("--serial") + 1]
        baud = int(args[args.index("--baud") + 1]) if "--baud" in args else 115200
        if actuate:
            from snn_data_lake.brainstem import serial_arm

            serial_arm.assert_calibrated()  # refuse real servos w/ placeholder limits
            if actuate == port:  # one UART: share ONE handle (read + write)
                import serial

                ser = serial.Serial(port, baud, timeout=1)
                arm = serial_arm.SerialArm(writer=ser)
                samples = _serial_samples_from(ser)
            else:  # command on a separate port
                arm = serial_arm.SerialArm(actuate, baud)
                samples = serial_samples(port, baud)
        else:
            arm = So101Arm()
            samples = serial_samples(port, baud)
        sys.stderr.write(
            f"live: sensor={port}@{baud} actuate={actuate or 'sim'} stride={stride} "
            f"proprioception={proprio}\n"
        )
        run(
            samples, arm, interp, emit, stride, observer=_make_observer() if proprio else None
        )  # until interrupted
        return

    arm = So101Arm()  # stdlib backend (pybullet if installed)
    if "--csv" in args:
        run(
            replay_samples([args[args.index("--csv") + 1]]),
            arm,
            interp,
            emit,
            stride,
            observer=_make_observer() if proprio else None,
        )
        return

    # default: replay the reference labeled captures (hardware-free) + self-check
    gestures = {g: os.path.join("robot arm", f"{g}.csv") for g in SENSOR_COMMANDS}
    have_real = all(os.path.exists(p) for p in gestures.values())
    quiet = lambda d: None

    if have_real:
        stats_a = run(replay_samples([gestures["HAND_APPROACH"]]), So101Arm(), interp, quiet)
        stats_r = run(replay_samples([gestures["HAND_RETREAT"]]), So101Arm(), interp, quiet)
        a_arm = So101Arm()
        run(replay_samples([gestures["HAND_APPROACH"]]), a_arm, interp, quiet)
        source = "real capture (robot arm/*.csv)"
    else:  # CI without data: synth streams
        appr = [(0.9 - 0.11 * i, 295.0) for i in range(W)] * 3 + [(0.02, 298.0)]
        retr = [(0.1 + 0.11 * i, 295.0) for i in range(W)] * 3
        stats_a = run(iter(appr), So101Arm(), interp, quiet)
        stats_r = run(iter(retr), So101Arm(), interp, quiet)
        a_arm = So101Arm()
        run(iter(appr), a_arm, interp, quiet)
        source = "synthetic fallback"

    # audit P1 regression: a non-finite sensor sample must trip the contact reflex (fail-closed),
    # not slip past `d < CONTACT` (NaN < x is False). Runs in the CI/default path.
    nan_stats = run(iter([(float("nan"), 298.0)]), So101Arm(), interp, quiet)
    assert nan_stats["reflex"] == 1, (
        "a NaN distance sample must trigger the contact reflex (fail-closed)"
    )

    print("=" * 62)
    print("LIVE ARM LOOP — sensor -> recognize -> Interpreter -> arm sim")
    print("=" * 62)
    print(f"source        : {source}")
    print(
        f"APPROACH file : {stats_a.get('ENGAGE', 0)} ENGAGE, "
        f"{stats_a.get('reflex', 0)} reflex-STOP (object reached the sensor)"
    )
    print(f"RETREAT file  : {stats_r.get('RELEASE', 0)} RELEASE")
    print(
        f"arm after approach run: effector={a_arm.effector} ee_mm={a_arm.state()['ee_mm']} "
        f"stopped={a_arm.stopped}"
    )
    print("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert stats_a.get("ENGAGE", 0) >= 1, "approach did not drive ENGAGE"
    assert stats_r.get("RELEASE", 0) >= 1, "retreat did not drive RELEASE"
    assert stats_a.get("reflex", 0) >= 1, "contact never tripped the reflex STOP"

    # duplex smoke test: sensor-read + command-write share ONE serial handle (no hardware).
    from snn_data_lake.brainstem.serial_arm import SerialArm

    approach = [f"{0.9 - 0.11 * i}, 295.0\n".encode() for i in range(W)]
    # approach (x3) -> contact (E-STOP) -> approach again: the trailing approach must NOT re-drive the wire
    lines = (approach * 3) + [b"0.02, 298.0\n"] + approach

    class _FakeDuplex:
        def __init__(self, ls):
            self._it = iter(ls)
            self.sent = []

        def readline(self):
            return next(self._it, b"")  # b"" at end of the canned stream

        def write(self, b):
            self.sent.append(b.decode("ascii").strip())

        def flush(self):
            pass

        def close(self):
            pass

    fd = _FakeDuplex(lines)
    darm = SerialArm(
        writer=fd, allow_uncalibrated=True
    )  # bench duplex: opt in past the calibration gate

    def _bounded():  # _serial_samples_from, but stops at EOF
        while True:
            raw = fd.readline()
            if not raw:
                return
            v = S.parse_line(raw.decode("utf-8", "ignore"))
            if v and len(v) >= 2:
                yield v[0], v[1]

    run(_bounded(), darm, interp, quiet)  # read + recognize + write on one handle
    assert "ENGAGE" in fd.sent, f"duplex: approach did not send ENGAGE ({fd.sent})"
    assert "EMERGENCY_STOP" in fd.sent, "duplex: contact did not send EMERGENCY_STOP over serial"
    # the E-STOP freeze must hold end-to-end: the post-contact approach must NOT reach the wire
    estop_i = fd.sent.index("EMERGENCY_STOP")
    assert "ENGAGE" not in fd.sent[estop_i + 1 :], (
        f"a command after EMERGENCY_STOP reached the wire through live_arm ({fd.sent})"
    )

    print(
        "self-check OK: approach->ENGAGE, retreat->RELEASE, contact->reflex STOP, "
        "post-STOP commands frozen off the wire, one-UART duplex read+write (loop closed)"
    )

    # ---- opt-in arm-primitive wiring (v2.2): defaults above are unchanged; each is proven safe ----
    from snn_data_lake.faculties.ring_attractor import ProprioceptiveEstimator

    # 1. --proprioception observer is READ-ONLY: the SAME stream, with vs without the observer,
    #    must drive the arm identically (it issues no command and never moves the joint).
    obs_stream = [(0.9 - 0.11 * i, 295.0) for i in range(W)] * 3 + [(0.02, 298.0)]
    base_arm = So101Arm()
    base_stats = run(iter(obs_stream), base_arm, interp, quiet)
    obs_arm = So101Arm()
    events = []
    obs_stats = run(
        iter(obs_stream), obs_arm, interp, events.append, observer=ProprioceptiveEstimator()
    )
    assert obs_stats == base_stats, "proprioception observer changed the loop's actions"
    assert obs_arm.state() == base_arm.state(), (
        "proprioception observer moved the arm (must be read-only)"
    )
    assert any("joint_est_deg" in e for e in events), (
        "observer logged no encoder-free joint estimate"
    )
    # slip watchdog: a joint that moves with NO command must diverge estimate from truth
    watchdog = ProprioceptiveEstimator()  # estimate stays home (no commands issued to it)
    slipped_arm = So101Arm()
    slipped_arm.inject_disturbance(_OBS_JOINT, 25.0)  # joint moved WITHOUT a command
    assert watchdog.mismatch_deg(slipped_arm.state()["theta_deg"][_OBS_INDEX]) > SLIP_DEG, (
        "slip watchdog did not flag an uncommanded joint move"
    )

    # 2. --reach: spiking PID drives joint A to the target THROUGH the gated apply().
    from snn_data_lake.faculties.spiking_pid import drive_to

    _errs, reach_final, _ = drive_to(30.0)
    assert abs(reach_final - 30.0) < 1.5, (
        f"--reach did not converge through gated apply(): {reach_final:.2f}"
    )

    # 3. --sweep: CPG rhythmic bounded motion THROUGH the gated apply(), and E-STOP still freezes it.
    from snn_data_lake.faculties.cpg_oscillator import cpg_sweep

    sweep_trace = cpg_sweep(So101Arm(), steps=400)
    assert max(sweep_trace) - min(sweep_trace) > 15.0, "--sweep produced no rhythmic joint motion"
    # E-STOP freezes a CPG-driven arm. THIS ASSERT WAS VACUOUS UNTIL 2026-08-05 and it is the only
    # end-to-end E-STOP proof in the repo, so the failure is recorded rather than quietly patched.
    #
    # It drove the latched arm with the literal "JOINT_A_ROTATE(+5deg)". There is no joint "A": the
    # arm has 6 joints and that exact string is the ONLY `JOINT_...` token anywhere in
    # brainstem/interpreter.py, i.e. a placeholder that parses to nothing. MEASURED with the control
    # this test never had -- an UNLATCHED arm, driven 50x with that literal, moved 0 of 6 joints
    # (theta_deg identical to 4 decimal places), while JOINT_0_ROTATE(+5deg), SHOULDER_PAN(+5deg)
    # and MOVE(+5,0,0) each moved it. So the old assert compared a frozen arm against a command that
    # freezes nothing: it passed for every possible state of the E-STOP latch, including a broken
    # one, and would have gone on passing if the latch were deleted outright.
    #
    # The fix is the CONTROL, not a different literal: assert first that the command moves an
    # UNLATCHED arm (which makes the red reachable and keeps it reachable if the command vocabulary
    # ever drifts again), then that the same command moves nothing once latched.
    _MOTION_CMD = "JOINT_0_ROTATE(+5deg)"  # verified to move an unlatched arm, by the control below

    control_arm = So101Arm()
    cpg_sweep(control_arm, steps=120)
    before_ctrl = list(control_arm.state()["theta_deg"])
    for _ in range(50):
        control_arm.apply(_MOTION_CMD)
    assert list(control_arm.state()["theta_deg"]) != before_ctrl, (
        f"CONTROL FAILED: {_MOTION_CMD} moves an UNLATCHED arm not at all, so the E-STOP assert "
        "below would pass vacuously. Fix the command, not this assert -- a frozen arm and an inert "
        "command are indistinguishable, which is exactly the defect this control exists to catch."
    )

    estop_arm = So101Arm()
    cpg_sweep(estop_arm, steps=120)
    estop_arm.apply("EMERGENCY_STOP")
    frozen = estop_arm.state()["theta_deg"]
    for _ in range(50):
        estop_arm.apply(_MOTION_CMD)
    assert estop_arm.state()["theta_deg"] == frozen, (
        "EMERGENCY_STOP failed to freeze the CPG-driven arm"
    )

    print(
        "opt-in self-check OK: proprioception observer read-only + slip watchdog; "
        f"--reach PID -> {reach_final:.1f}deg; --sweep CPG rhythmic + E-STOP-safe (all via the gated arm API)"
    )


if __name__ == "__main__":
    main()
