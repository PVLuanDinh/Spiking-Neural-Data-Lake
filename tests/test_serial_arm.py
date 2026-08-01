"""
Safety tests for SerialArm (the real-servo command-out sink): the calibration gate refuses
an uncalibrated real port, a serial write error is fail-safe (raises rather than driving
blind), and commands are sent verbatim + mirrored. A fake writer stands in for the ESP32 —
no hardware, no pyserial (the module stays pure-stdlib until a real port is opened).

Run: pytest tests/test_serial_arm.py
"""

import pytest

from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.serial_arm import SerialArm, assert_calibrated


class FakeWriter:
    """Records the lines that would go out over serial; can be told to fail a write."""

    def __init__(self):
        self.lines, self.fail, self.closed = [], False, False

    def write(self, b):
        if self.fail:
            raise OSError("cable yanked")
        self.lines.append(b.decode("ascii").strip())

    def flush(self):
        pass

    def close(self):
        self.closed = True


def test_uncalibrated_real_port_is_refused():
    # writer=None -> opens a real port; must be blocked while SAFETY_CALIBRATED is False
    assert arm_config.SAFETY_CALIBRATED is False  # guard the test's own premise
    with pytest.raises(RuntimeError, match="refuses to drive real servos"):
        SerialArm(port="COM_TEST")


def test_assert_calibrated_gate_toggles(monkeypatch):
    with pytest.raises(RuntimeError):
        assert_calibrated()
    assert_calibrated(allow_uncalibrated=True)  # explicit bench override does not raise
    monkeypatch.setattr(arm_config, "SAFETY_CALIBRATED", True)
    assert_calibrated()  # calibrated -> no raise


def test_commands_sent_verbatim_and_mirrored():
    w = FakeWriter()
    arm = SerialArm(writer=w, allow_uncalibrated=True)  # bench: opt in past the gate
    arm.apply("ENGAGE")
    arm.apply("JOINT_A_ROTATE(+15deg)")
    arm.apply("EMERGENCY_STOP")
    assert w.lines == ["ENGAGE", "JOINT_A_ROTATE(+15deg)", "EMERGENCY_STOP"]
    assert arm.effector == 1.0  # mirror closed the effector
    assert arm.state()["stopped"] is True  # EMERGENCY_STOP froze the mirror


def test_injected_writer_is_still_calibration_gated():
    # closing the bypass: an injected writer with no calibration + no opt-in must be refused too
    w = FakeWriter()
    with pytest.raises(RuntimeError, match="refuses to drive real servos"):
        SerialArm(writer=w)


def test_emergency_stop_freezes_the_wire():
    # once latched, nothing but stop/home/retract reaches the servos — enforced before the write
    w = FakeWriter()
    arm = SerialArm(writer=w, allow_uncalibrated=True)
    arm.apply("EMERGENCY_STOP")
    arm.apply("ENGAGE")  # blocked by the freeze -> not transmitted
    arm.apply("JOINT_A_ROTATE(+10deg)")  # blocked -> not transmitted
    assert w.lines == ["EMERGENCY_STOP"]
    arm.apply("HOME")  # allowed while stopped -> clears the latch
    arm.apply("RELEASE")  # latch cleared -> transmitted again
    assert w.lines == ["EMERGENCY_STOP", "HOME", "RELEASE"]
    assert arm.state()["stopped"] is False


def test_unknown_command_is_not_transmitted():
    # allow-list: only recognised tokens ever reach the wire
    w = FakeWriter()
    arm = SerialArm(writer=w, allow_uncalibrated=True)
    arm.apply("DROP_EVERYTHING")
    arm.apply(None)
    assert w.lines == []


def test_write_error_is_failsafe():
    w = FakeWriter()
    arm = SerialArm(writer=w, allow_uncalibrated=True)
    w.fail = True
    with pytest.raises(RuntimeError, match="actuator serial write failed"):
        arm.apply("RELEASE")


def test_close_swallows_writer_errors():
    w = FakeWriter()
    arm = SerialArm(writer=w, allow_uncalibrated=True)
    arm.close()
    assert w.closed is True


# ---- T032 / FR-003: the calibration gate keeps its no-bypass property ----------------------
# The owner authorised REIMPLEMENTING the arm's safety gate for the SO-101. That is authorisation
# to rebuild it, not to weaken it. These pin the property independently of which body is
# underneath, so swapping the arm cannot quietly open a path to the wire.


def test_no_command_reaches_the_wire_while_uncalibrated():
    """Refusal is at construction and applies to every port shape, not just the first tried."""
    assert arm_config.SAFETY_CALIBRATED is False  # guard the test's own premise
    for port in ("COM_TEST", "/dev/ttyUSB0", "COM9"):
        with pytest.raises(RuntimeError, match="refuses to drive real servos"):
            SerialArm(port=port)


def test_injected_writer_does_not_bypass_the_gate():
    """A fake writer bypasses the PORT, not the CALIBRATION CHECK.

    This was once true the other way round -- an injected writer skipped the gate entirely -- and
    the module docstring records the fix. Pinned so it cannot regress.
    """
    with pytest.raises(RuntimeError, match="refuses to drive real servos"):
        SerialArm(writer=FakeWriter())


def test_the_escape_hatch_is_the_only_one():
    """`allow_uncalibrated` is a deliberate, documented bench-test opt-in. It must stay the ONLY
    parameter that can stand in for calibration.

    If a second one appears -- force=, unsafe=, skip_checks= -- this fails. One reviewed escape
    hatch is a decision; two is a habit.
    """
    import inspect

    params = set(inspect.signature(SerialArm.__init__).parameters)
    forbidden = {"force", "unsafe", "skip_checks", "bypass", "no_gate", "ignore_calibration"}
    assert not (params & forbidden), (
        f"SerialArm gained a second calibration bypass: {sorted(params & forbidden)}"
    )


def test_escape_hatch_still_requires_an_explicit_opt_in():
    """It must never be the default, and must not be reachable by omission."""
    import inspect

    sig = inspect.signature(SerialArm.__init__)
    assert sig.parameters["allow_uncalibrated"].default is False, (
        "the uncalibrated escape hatch became the default"
    )


def test_estop_freeze_keeps_the_wire_silent():
    """The freeze gates the serial write itself, so it does not depend on firmware behaviour."""
    arm = SerialArm(writer=FakeWriter(), allow_uncalibrated=True)
    arm.apply("EMERGENCY_STOP")
    sent_before = len(arm.w.lines)
    for cmd in ("JOINT_A_ROTATE(+15deg)", "ENGAGE", "RELEASE"):
        arm.apply(cmd)
    assert len(arm.w.lines) == sent_before, (
        f"a latched arm transmitted to the wire: {arm.w.lines[sent_before:]}"
    )
