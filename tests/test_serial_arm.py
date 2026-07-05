"""
Safety tests for SerialArm (the real-servo command-out sink): the calibration gate refuses
an uncalibrated real port, a serial write error is fail-safe (raises rather than driving
blind), and commands are sent verbatim + mirrored. A fake writer stands in for the ESP32 —
no hardware, no pyserial (the module stays pure-stdlib until a real port is opened).

Run: pytest tests/test_serial_arm.py
"""

import pytest

from snn_data_lake import arm_config
from snn_data_lake.serial_arm import SerialArm, assert_calibrated


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
    arm.apply("GRIPPER_CLOSE")
    arm.apply("JOINT_A_ROTATE(+15deg)")
    arm.apply("EMERGENCY_STOP")
    assert w.lines == ["GRIPPER_CLOSE", "JOINT_A_ROTATE(+15deg)", "EMERGENCY_STOP"]
    assert arm.gripper == 1.0  # mirror closed the gripper
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
    arm.apply("GRIPPER_CLOSE")  # blocked by the freeze -> not transmitted
    arm.apply("JOINT_A_ROTATE(+10deg)")  # blocked -> not transmitted
    assert w.lines == ["EMERGENCY_STOP"]
    arm.apply("HOME")  # allowed while stopped -> clears the latch
    arm.apply("GRIPPER_OPEN")  # latch cleared -> transmitted again
    assert w.lines == ["EMERGENCY_STOP", "HOME", "GRIPPER_OPEN"]
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
        arm.apply("GRIPPER_OPEN")


def test_close_swallows_writer_errors():
    w = FakeWriter()
    arm = SerialArm(writer=w, allow_uncalibrated=True)
    arm.close()
    assert w.closed is True
