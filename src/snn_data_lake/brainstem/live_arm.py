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

Gesture -> command map (sensor domain): approach = grab the incoming object, retreat = release.
"""

import math
import os
import re
import sys
import time

from snn_data_lake import signal_loop as S
from snn_data_lake.brainstem.arm_config import (  # reflex distance + window stride: calibration in arm_config
    CONTACT,
    STRIDE,
)
from snn_data_lake.brainstem.arm_sim import ArmSim
from snn_data_lake.brainstem.interpreter import Interpreter
from snn_data_lake.cortex.gesture_recognition import W, classify, load_csv
from snn_data_lake.substrate import home

SENSOR_COMMANDS = {
    "HAND_APPROACH": "ENGAGE",  # grab the approaching object
    "HAND_RETREAT": "RELEASE",  # release as it leaves
    "IDLE": "HOLD",
}

SLIP_DEG = 15.0  # |estimate - actual| above this flags an uncommanded slip
_JOINT_A = re.compile(r"JOINT_A_ROTATE\(([+-]?\d+(?:\.\d+)?)deg\)")


def _joint_a_delta(command):
    """Commanded joint-A delta (deg) from an Interpreter command; 0.0 for effector / HOLD / stop."""
    m = _JOINT_A.fullmatch(command or "")
    return float(m.group(1)) if m else 0.0


def _make_observer():
    """Opt-in read-only proprioceptive observer: encoder-free joint-angle estimate + slip watchdog."""
    from snn_data_lake.hippocampus.ring_attractor import ProprioceptiveEstimator

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
            emit({"reflex": "STOP", "command": cmd, "ee": arm.state()["ee"]})
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
                "ee": arm.state()["ee"],
            }
            if observer is not None:  # opt-in read-only proprioception
                est = observer.command(_joint_a_delta(cmd))
                mism = observer.mismatch_deg(arm.state()["theta_deg"][0])
                ev["joint_est_deg"] = round(est, 1)
                ev["joint_mismatch_deg"] = round(mism, 1)
                ev["slip"] = mism > SLIP_DEG  # joint moved WITHOUT a command
            emit(ev)
            buf = buf[stride:]  # slide (stride==W -> full reset)
    return stats


def main():
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
        from snn_data_lake.cerebellum.spiking_pid import drive_to

        target = float(args[args.index("--reach") + 1])
        errs, final, breaches = drive_to(target)  # drive_to uses ArmSim().apply() (gated)
        print(
            f"reach: spiking PID -> joint A {final:.2f} deg (target {target:.1f}) via gated apply(); "
            f"start err {errs[0]:+.1f} deg, trajectory breaches {breaches}"
        )
        return

    # opt-in: rhythmic bounded joint sweep from the CPG, THROUGH the safety-gated apply() (off by default)
    if "--sweep" in args:
        from snn_data_lake.brainstem.cpg_oscillator import cpg_sweep

        trace = cpg_sweep(ArmSim())  # cpg_sweep drives arm.apply() (gated); stops on collision
        print(
            f"sweep: CPG rhythm -> joint A in [{min(trace):.1f}, {max(trace):.1f}] deg via gated apply() "
            f"({len(trace)} steps)"
        )
        return

    # opt-in: plan a collision-free path around the obstacle to a Cartesian end-effector goal (--plan-xy X Y)
    # or a joint config (--plan A B), executed THROUGH the safety-gated apply() -- a reaction-diffusion
    # wavefront in configuration space (off by default; adds no new command token, moves via JOINT_*_ROTATE)
    if "--plan-xy" in args or "--plan" in args:
        from snn_data_lake.cerebellum.wavefront_planner import plan_to_config, plan_to_xy

        arm = ArmSim()
        if "--plan-xy" in args:
            gx = float(args[args.index("--plan-xy") + 1])
            gy = float(args[args.index("--plan-xy") + 2])
            res = plan_to_xy(arm, gx, gy)
            goal = f"EE ({gx:.3f}, {gy:.3f})"
        else:
            ga = float(args[args.index("--plan") + 1])
            gb = float(args[args.index("--plan") + 2])
            res = plan_to_config(arm, ga, gb)
            goal = f"config ({ga:.1f}, {gb:.1f}) deg"
        ex, ey = arm.ee()
        print(
            f"plan: {goal} -> status {res['status']}, {res['waypoints']} waypoints, "
            f"{res['steps']} gated commands; final EE ({ex:.3f}, {ey:.3f}) via gated apply()"
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
            arm = ArmSim()
            samples = serial_samples(port, baud)
        sys.stderr.write(
            f"live: sensor={port}@{baud} actuate={actuate or 'sim'} stride={stride} "
            f"proprioception={proprio}\n"
        )
        run(
            samples, arm, interp, emit, stride, observer=_make_observer() if proprio else None
        )  # until interrupted
        return

    arm = ArmSim()  # stdlib backend (pybullet if installed)
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
        stats_a = run(replay_samples([gestures["HAND_APPROACH"]]), ArmSim(), interp, quiet)
        stats_r = run(replay_samples([gestures["HAND_RETREAT"]]), ArmSim(), interp, quiet)
        a_arm = ArmSim()
        run(replay_samples([gestures["HAND_APPROACH"]]), a_arm, interp, quiet)
        source = "real capture (robot arm/*.csv)"
    else:  # CI without data: synth streams
        appr = [(0.9 - 0.11 * i, 295.0) for i in range(W)] * 3 + [(0.02, 298.0)]
        retr = [(0.1 + 0.11 * i, 295.0) for i in range(W)] * 3
        stats_a = run(iter(appr), ArmSim(), interp, quiet)
        stats_r = run(iter(retr), ArmSim(), interp, quiet)
        a_arm = ArmSim()
        run(iter(appr), a_arm, interp, quiet)
        source = "synthetic fallback"

    # audit P1 regression: a non-finite sensor sample must trip the contact reflex (fail-closed),
    # not slip past `d < CONTACT` (NaN < x is False). Runs in the CI/default path.
    nan_stats = run(iter([(float("nan"), 298.0)]), ArmSim(), interp, quiet)
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
        f"arm after approach run: effector={a_arm.effector} ee={a_arm.state()['ee']} "
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
    from snn_data_lake.hippocampus.ring_attractor import ProprioceptiveEstimator

    # 1. --proprioception observer is READ-ONLY: the SAME stream, with vs without the observer,
    #    must drive the arm identically (it issues no command and never moves the joint).
    obs_stream = [(0.9 - 0.11 * i, 295.0) for i in range(W)] * 3 + [(0.02, 298.0)]
    base_arm = ArmSim()
    base_stats = run(iter(obs_stream), base_arm, interp, quiet)
    obs_arm = ArmSim()
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
    slipped_arm = ArmSim()
    slipped_arm.perturb(da_deg=25.0)  # joint moved WITHOUT a command
    assert watchdog.mismatch_deg(slipped_arm.state()["theta_deg"][0]) > SLIP_DEG, (
        "slip watchdog did not flag an uncommanded joint move"
    )

    # 2. --reach: spiking PID drives joint A to the target THROUGH the gated apply().
    from snn_data_lake.cerebellum.spiking_pid import drive_to

    _errs, reach_final, _ = drive_to(30.0)
    assert abs(reach_final - 30.0) < 1.5, (
        f"--reach did not converge through gated apply(): {reach_final:.2f}"
    )

    # 3. --sweep: CPG rhythmic bounded motion THROUGH the gated apply(), and E-STOP still freezes it.
    from snn_data_lake.brainstem.cpg_oscillator import cpg_sweep

    sweep_trace = cpg_sweep(ArmSim(), steps=400)
    assert max(sweep_trace) - min(sweep_trace) > 15.0, "--sweep produced no rhythmic joint motion"
    estop_arm = ArmSim()
    cpg_sweep(estop_arm, steps=120)
    estop_arm.apply("EMERGENCY_STOP")
    frozen = estop_arm.theta[0]
    for _ in range(50):
        estop_arm.apply("JOINT_A_ROTATE(+5deg)")
    assert abs(estop_arm.theta[0] - frozen) < 1e-9, (
        "EMERGENCY_STOP failed to freeze the CPG-driven arm"
    )

    print(
        "opt-in self-check OK: proprioception observer read-only + slip watchdog; "
        f"--reach PID -> {reach_final:.1f}deg; --sweep CPG rhythmic + E-STOP-safe (all via the gated arm API)"
    )


if __name__ == "__main__":
    main()
