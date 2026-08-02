"""
v0.60 — real-arm actuator sink. The Phase-2 command-OUT path.

live_arm.py already does: sensor -> recognize -> Interpreter -> command -> arm.apply(cmd).
Until now `arm` was ArmSim (a simulation). SerialArm is a DROP-IN with the same surface
live_arm uses off ArmSim — .apply(cmd), .state(), .effector — but each command line also goes
out over serial to the ESP32 servo sketch (deploy/command_arm.ino), moving the REAL arm.
See docs/arduino_contract.md ("Command channel").

SAFETY (three guards enforced on the WIRE, before the write — not just in the mirror):
  1. CALIBRATION — SerialArm refuses to construct against real servos until arm_config
     .SAFETY_CALIBRATED is True. The check runs for EVERY writer (an injected `writer=` no longer
     bypasses it); a bench/echo test must opt in explicitly with allow_uncalibrated=True.
  2. EMERGENCY_STOP FREEZE — once EMERGENCY_STOP latches, NOTHING is transmitted to the servos
     except EMERGENCY_STOP / HOME / RETRACT_ALL until an explicit HOME/RETRACT clears the latch.
     The freeze gates the serial write itself, so it does not depend on the ESP32 firmware.
  3. COMMAND ALLOW-LIST — only recognised command tokens (the fixed set + JOINT_[AB]_ROTATE(...))
     are ever put on the wire; an unmapped/None command is dropped, never sent as raw bytes.

Fail-safe: any serial write error raises, so the caller stops the loop rather than driving blind.
pyserial is imported lazily (only when opening a real port) so this module + its self-check stay
pure-stdlib and run in CI.

  python serial_arm.py        # self-check (fake serial, no hardware, no pyserial)
"""

import re

from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.so101_arm import So101Arm

# The wire allow-list, built from the arm's own joint names -- INDEPENDENTLY of the mirror,
# which is the property that caught a malformed token the mirror had accepted. Names are
# escaped and alternated rather than matched with a loose [A-Z_]+ class, so a token naming a
# joint this arm does not have is rejected here rather than reaching the hardware.
# The two joints the self-check drives. Named once, so the command that is SENT and the
# command that is ASSERTED cannot drift apart -- the incumbent spelled each out twice.
_DEMO_MOVE = arm_config.command_for("shoulder_lift", 15)
_DEMO_BLOCKED = arm_config.command_for("elbow_flex", -5)

_JOINT = re.compile(
    r"JOINT_(?:"
    + "|".join(re.escape(n.upper()) for n in arm_config.ROTATABLE)
    + r")_(?:ROTATE|MOVETO)\([+-]?\d+(?:\.\d+)?deg\)"
)
_FIXED = {"EMERGENCY_STOP", "HOME", "RETRACT_ALL", "ENGAGE", "RELEASE", "HOLD"}
_ALLOWED_WHEN_STOPPED = {
    "EMERGENCY_STOP",
    "HOME",
    "RETRACT_ALL",
}  # the only tokens allowed on the wire once latched


def is_valid_command(command):
    """True only for a recognised actuator token (fixed set or a JOINT rotate) — the wire allow-list."""
    return isinstance(command, str) and (command in _FIXED or _JOINT.fullmatch(command) is not None)


def assert_calibrated(allow_uncalibrated=False):
    """Guard every real-servo entry point: raise unless the sim's placeholder safety limits
    have been replaced with the rig's real values (arm_config.SAFETY_CALIBRATED)."""
    if not (arm_config.SAFETY_CALIBRATED or allow_uncalibrated):
        raise RuntimeError(
            "SerialArm refuses to drive real servos with placeholder safety limits. "
            "Set arm_config.SAFETY_CALIBRATED=True after replacing CONTACT/joint limits "
            "with your rig's real values (docs/arduino_requirements.md #3), or pass "
            "allow_uncalibrated=True TOGETHER WITH a fake writer= for a bench/echo test."
        )


class SerialArm:
    """Command sink: mirrors ArmSim (for state()/effector logging) AND writes each command line
    to the ESP32. writer = an object with .write(bytes)/.flush()/.close() — a real serial.Serial
    on hardware, or a fake for bench tests. writer=None opens a real pyserial port."""

    def __init__(self, port=None, baud=115200, writer=None, allow_uncalibrated=False):
        # calibration is checked for EVERY construction, not just the auto-open branch, so an
        # injected `writer=` (e.g. the shared-UART live path) can no longer bypass the gate.
        # THE ESCAPE HATCH CANNOT OPEN A REAL PORT. The docstring always said
        # allow_uncalibrated was for a bench/echo test "(with a fake writer=)", but nothing
        # enforced the pairing -- so `SerialArm(port=..., allow_uncalibrated=True)` passed the
        # gate and then opened a real pyserial port, which is the single path to the servos that
        # calibration exists to close. Enforced here rather than in assert_calibrated() because
        # this is the only place that opens a port; the bare guard is also called by live_arm
        # where no port is involved. A legitimate bench test already injects a writer, so this
        # cannot break one.
        if allow_uncalibrated and writer is None:
            raise RuntimeError(
                "allow_uncalibrated=True requires an injected writer= (a bench/echo test). "
                "It is not a way to open a REAL port with placeholder safety limits."
            )
        assert_calibrated(allow_uncalibrated)
        if writer is None:
            import serial  # pyserial — only needed on real hardware

            writer = serial.Serial(port, baud, timeout=1)
        self.w = writer
        # Local twin so state()/effector still work. NOT the wire validator: `is_valid_command`
        # builds its allow-list from arm_config independently, which is what caught a token the
        # mirror had accepted. Two readings of one source, not one reading trusted twice.
        self.mirror = So101Arm()
        self.mirror.apply("HOME")

    def apply(self, command):
        """Gate the command against the E-STOP freeze + the allow-list, THEN (if it survives) send it
        to the real arm (flushed, so EMERGENCY_STOP is not buffered) and mirror it. A blocked command
        touches only the mirror; nothing goes to the servos. A serial failure raises (fail-safe)."""
        blocked_by_stop = self.mirror.stopped and command not in _ALLOWED_WHEN_STOPPED
        if blocked_by_stop or not is_valid_command(command):
            return self.mirror.apply(
                command
            )  # frozen or unrecognised -> mirror only, WIRE stays silent
        try:
            self.w.write((command + "\n").encode("ascii"))
            self.w.flush()
        except Exception as e:
            raise RuntimeError(f"actuator serial write failed on '{command}': {e}") from e
        return self.mirror.apply(command)

    def state(self):
        return self.mirror.state()

    @property
    def effector(self):
        return self.mirror.effector

    def close(self):
        try:
            self.w.close()
        except Exception:
            pass


def main():
    # bench self-check: a fake writer records the lines. No hardware, no pyserial.
    class FakeSerial:
        def __init__(self):
            self.lines = []
            self.fail = False

        def write(self, b):
            if self.fail:
                raise OSError("cable yanked")
            self.lines.append(b.decode("ascii").strip())

        def flush(self):
            pass

        def close(self):
            pass

    fake = FakeSerial()
    arm = SerialArm(writer=fake, allow_uncalibrated=True)  # bench must opt in explicitly now

    arm.apply("ENGAGE")
    arm.apply(_DEMO_MOVE)
    arm.apply("EMERGENCY_STOP")
    # --- once stopped, the WIRE must stay silent for anything but stop/home/retract ---
    arm.apply("ENGAGE")  # blocked by the E-STOP freeze -> NOT transmitted
    arm.apply(_DEMO_BLOCKED)  # blocked by the E-STOP freeze -> NOT transmitted
    lines_during_stop = list(fake.lines)
    arm.apply("HOME")  # allowed while stopped -> transmitted, clears the latch
    arm.apply("RELEASE")  # latch cleared -> transmitted again
    # --- allow-list: an unrecognised command is never put on the wire ---
    arm.apply("DROP_EVERYTHING")  # invalid token -> dropped, NOT transmitted

    print("=" * 58)
    print("SERIAL ARM — command-out sink (fake serial, no hardware)")
    print("=" * 58)
    print(f"lines sent     : {fake.lines}")
    print(
        f"during E-STOP  : nothing new transmitted -> {lines_during_stop[-1]} was last before HOME"
    )
    print(f"mirror effector : {arm.effector}   stopped: {arm.state()['stopped']}")

    # calibration gate: an INJECTED writer with no calibration + no opt-in must now be blocked too
    try:
        SerialArm(writer=FakeSerial())  # writer= no longer bypasses the calibration gate
        writer_gate_blocked = False
    except RuntimeError:
        writer_gate_blocked = True
    # calibration gate: opening a REAL port while uncalibrated must be blocked
    try:
        SerialArm(port="COM_TEST")  # writer=None -> would open a real port
        port_gate_blocked = False
    except RuntimeError:
        port_gate_blocked = True

    # fail-safe: a serial error must raise, not be swallowed
    fake.fail = True
    try:
        arm.apply("ENGAGE")  # (arm is not stopped now; valid) -> attempts a write -> raises
        failsafe = False
    except RuntimeError:
        failsafe = True

    print(
        f"calib gate     : injected-writer open blocked = {writer_gate_blocked}; real-port blocked = {port_gate_blocked}"
    )
    print(f"fail-safe      : serial error raises           = {failsafe}")
    print("=" * 58)

    # ---- self-checks --------------------------------------------------------
    assert fake.lines[:3] == ["ENGAGE", _DEMO_MOVE, "EMERGENCY_STOP"], (
        f"pre-stop commands not sent verbatim: {fake.lines}"
    )
    assert fake.lines.count("ENGAGE") == 1, (
        f"a command after EMERGENCY_STOP reached the wire — the freeze is not enforced on serial: {fake.lines}"
    )
    assert _DEMO_BLOCKED not in fake.lines, "a joint move after E-STOP reached the wire"
    assert "HOME" in fake.lines and "RELEASE" in fake.lines, (
        "HOME must clear the latch and re-enable the wire"
    )
    assert "DROP_EVERYTHING" not in fake.lines, (
        "an unrecognised command was transmitted (allow-list defeated)"
    )
    assert not arm.state()["stopped"], "HOME after E-STOP should have cleared the freeze"
    assert writer_gate_blocked, "an injected writer with no calibration/opt-in must be refused"
    assert port_gate_blocked or arm_config.SAFETY_CALIBRATED, (
        "uncalibrated real-port open must be blocked"
    )
    assert failsafe, "serial write failure must raise (fail-safe)"
    print(
        "self-check OK: pre-stop verbatim; E-STOP freezes the WIRE (post-stop commands dropped) + HOME clears it; "
        "allow-list blocks unknown tokens; calibration gate covers injected writers; fail-safe active"
    )


if __name__ == "__main__":
    main()
