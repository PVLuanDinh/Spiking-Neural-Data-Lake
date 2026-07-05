"""
v0.60 — real-arm actuator sink. The Phase-2 command-OUT path.

live_arm.py already does: sensor -> recognize -> Interpreter -> command -> arm.apply(cmd).
Until now `arm` was ArmSim (a simulation). SerialArm is a DROP-IN with the same surface
live_arm uses off ArmSim — .apply(cmd), .state(), .gripper — but each command line also goes
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

from snn_data_lake import arm_config
from snn_data_lake.arm_sim import ArmSim

_JOINT = re.compile(r"JOINT_[AB]_ROTATE\([+-]?\d+(?:\.\d+)?deg\)")
_FIXED = {"EMERGENCY_STOP", "HOME", "RETRACT_ALL", "GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD"}
_ALLOWED_WHEN_STOPPED = {"EMERGENCY_STOP", "HOME", "RETRACT_ALL"}   # the only tokens allowed on the wire once latched


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
            "allow_uncalibrated=True (with a fake writer=) for a bench/echo test.")


class SerialArm:
    """Command sink: mirrors ArmSim (for state()/gripper logging) AND writes each command line
    to the ESP32. writer = an object with .write(bytes)/.flush()/.close() — a real serial.Serial
    on hardware, or a fake for bench tests. writer=None opens a real pyserial port."""

    def __init__(self, port=None, baud=115200, writer=None, allow_uncalibrated=False):
        # calibration is checked for EVERY construction, not just the auto-open branch, so an
        # injected `writer=` (e.g. the shared-UART live path) can no longer bypass the gate.
        assert_calibrated(allow_uncalibrated)
        if writer is None:
            import serial  # pyserial — only needed on real hardware
            writer = serial.Serial(port, baud, timeout=1)
        self.w = writer
        self.mirror = ArmSim()                   # local twin so state()/gripper still work
        self.mirror.apply("HOME")

    def apply(self, command):
        """Gate the command against the E-STOP freeze + the allow-list, THEN (if it survives) send it
        to the real arm (flushed, so EMERGENCY_STOP is not buffered) and mirror it. A blocked command
        touches only the mirror; nothing goes to the servos. A serial failure raises (fail-safe)."""
        blocked_by_stop = self.mirror.stopped and command not in _ALLOWED_WHEN_STOPPED
        if blocked_by_stop or not is_valid_command(command):
            return self.mirror.apply(command)    # frozen or unrecognised -> mirror only, WIRE stays silent
        try:
            self.w.write((command + "\n").encode("ascii"))
            self.w.flush()
        except Exception as e:
            raise RuntimeError(f"actuator serial write failed on '{command}': {e}") from e
        return self.mirror.apply(command)

    def state(self):
        return self.mirror.state()

    @property
    def gripper(self):
        return self.mirror.gripper

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
    arm = SerialArm(writer=fake, allow_uncalibrated=True)      # bench must opt in explicitly now

    arm.apply("GRIPPER_CLOSE")
    arm.apply("JOINT_A_ROTATE(+15deg)")
    arm.apply("EMERGENCY_STOP")
    # --- once stopped, the WIRE must stay silent for anything but stop/home/retract ---
    arm.apply("GRIPPER_CLOSE")           # blocked by the E-STOP freeze -> NOT transmitted
    arm.apply("JOINT_B_ROTATE(-5deg)")   # blocked by the E-STOP freeze -> NOT transmitted
    lines_during_stop = list(fake.lines)
    arm.apply("HOME")                    # allowed while stopped -> transmitted, clears the latch
    arm.apply("GRIPPER_OPEN")            # latch cleared -> transmitted again
    # --- allow-list: an unrecognised command is never put on the wire ---
    arm.apply("DROP_EVERYTHING")         # invalid token -> dropped, NOT transmitted

    print("=" * 58)
    print("SERIAL ARM — command-out sink (fake serial, no hardware)")
    print("=" * 58)
    print(f"lines sent     : {fake.lines}")
    print(f"during E-STOP  : nothing new transmitted -> {lines_during_stop[-1]} was last before HOME")
    print(f"mirror gripper : {arm.gripper}   stopped: {arm.state()['stopped']}")

    # calibration gate: an INJECTED writer with no calibration + no opt-in must now be blocked too
    try:
        SerialArm(writer=FakeSerial())       # writer= no longer bypasses the calibration gate
        writer_gate_blocked = False
    except RuntimeError:
        writer_gate_blocked = True
    # calibration gate: opening a REAL port while uncalibrated must be blocked
    try:
        SerialArm(port="COM_TEST")           # writer=None -> would open a real port
        port_gate_blocked = False
    except RuntimeError:
        port_gate_blocked = True

    # fail-safe: a serial error must raise, not be swallowed
    fake.fail = True
    try:
        arm.apply("GRIPPER_CLOSE")           # (arm is not stopped now; valid) -> attempts a write -> raises
        failsafe = False
    except RuntimeError:
        failsafe = True

    print(f"calib gate     : injected-writer open blocked = {writer_gate_blocked}; real-port blocked = {port_gate_blocked}")
    print(f"fail-safe      : serial error raises           = {failsafe}")
    print("=" * 58)

    # ---- self-checks --------------------------------------------------------
    assert fake.lines[:3] == ["GRIPPER_CLOSE", "JOINT_A_ROTATE(+15deg)", "EMERGENCY_STOP"], \
        f"pre-stop commands not sent verbatim: {fake.lines}"
    assert fake.lines.count("GRIPPER_CLOSE") == 1, \
        f"a command after EMERGENCY_STOP reached the wire — the freeze is not enforced on serial: {fake.lines}"
    assert "JOINT_B_ROTATE(-5deg)" not in fake.lines, "a joint move after E-STOP reached the wire"
    assert "HOME" in fake.lines and "GRIPPER_OPEN" in fake.lines, "HOME must clear the latch and re-enable the wire"
    assert "DROP_EVERYTHING" not in fake.lines, "an unrecognised command was transmitted (allow-list defeated)"
    assert not arm.state()["stopped"], "HOME after E-STOP should have cleared the freeze"
    assert writer_gate_blocked, "an injected writer with no calibration/opt-in must be refused"
    assert port_gate_blocked or arm_config.SAFETY_CALIBRATED, "uncalibrated real-port open must be blocked"
    assert failsafe, "serial write failure must raise (fail-safe)"
    print("self-check OK: pre-stop verbatim; E-STOP freezes the WIRE (post-stop commands dropped) + HOME clears it; "
          "allow-list blocks unknown tokens; calibration gate covers injected writers; fail-safe active")


if __name__ == "__main__":
    main()
