"""
pytest suite for interpreter.py — matched signal to robot command + outcome feedback.

Tests mirror the self-check in interpreter.py:selfcheck() to verify:
- Reflex preemption (STOP, WITHDRAW)
- Instinct AVOID veto beats match
- Command matching and mapping
- Unmapped labels and novel events
- Outcome echo passthrough (None, None)
- Reward policy: default 0.0 vs assume_success +0.5
"""

from snn_data_lake.brainstem.interpreter import Interpreter


def test_reflex_stop():
    """Reflex STOP event -> ("EMERGENCY_STOP", -1.0)."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"reflex": "STOP", "preempt": True})
    assert cmd == "EMERGENCY_STOP", f"reflex STOP should map to EMERGENCY_STOP, got {cmd}"
    assert reward == -1.0, f"reflex should have reward -1.0, got {reward}"


def test_reflex_withdraw():
    """Reflex WITHDRAW event -> ("RETRACT_ALL", -1.0)."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"reflex": "WITHDRAW"})
    assert cmd == "RETRACT_ALL", f"reflex WITHDRAW should map to RETRACT_ALL, got {cmd}"
    assert reward == -1.0, f"reflex should have reward -1.0, got {reward}"


def test_match_gripper_close():
    """Match GRIPPER_CLOSE label -> command "GRIPPER_CLOSE"."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"match": "GRIPPER_CLOSE", "instinct": None})
    assert cmd == "GRIPPER_CLOSE", f"match should return GRIPPER_CLOSE, got {cmd}"


def test_instinct_avoid_vetoes_match():
    """Instinct AVOID veto -> ("HOLD", -0.3), beats any match."""
    itp = Interpreter()
    # AVOID preempts the match
    cmd, reward = itp.interpret({"match": "HOME", "instinct": "AVOID"})
    assert cmd == "HOLD", f"AVOID veto should return HOLD, got {cmd}"
    assert reward == -0.3, f"AVOID veto should have reward -0.3, got {reward}"


def test_match_none_novel_event():
    """Match None (novel event) -> (None, 0.0)."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"match": None})
    assert cmd is None, f"match None should return cmd None, got {cmd}"
    assert reward == 0.0, f"match None should have reward 0.0, got {reward}"


def test_unmapped_label():
    """Unmapped label (no command entry) -> (None, 0.0)."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"match": "UNMAPPED_LABEL"})
    assert cmd is None, f"unmapped label should return cmd None, got {cmd}"
    assert reward == 0.0, f"unmapped label should have reward 0.0, got {reward}"


def test_outcome_echo_passthrough():
    """Outcome echo from loop -> (None, None); not processed by Interpreter."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"outcome": -1.0})
    assert cmd is None, f"outcome echo should return cmd None, got {cmd}"
    assert reward is None, f"outcome echo should return reward None, got {reward}"


def test_executed_command_reward_default():
    """Executed command reward is 0.0 without assume_success."""
    itp = Interpreter(assume_success=False)
    cmd, reward = itp.interpret({"match": "HOME"})
    assert cmd == "HOME", f"match should return HOME, got {cmd}"
    assert reward == 0.0, f"without assume_success, reward should be 0.0, got {reward}"


def test_executed_command_reward_with_assume_success():
    """Executed command reward is 0.5 with assume_success=True."""
    itp = Interpreter(assume_success=True)
    cmd, reward = itp.interpret({"match": "HOME"})
    assert cmd == "HOME", f"match should return HOME, got {cmd}"
    assert reward == 0.5, f"with assume_success, reward should be 0.5, got {reward}"


def test_command_priority_reflex_beats_avoid():
    """Reflex priority: STOP/WITHDRAW beats AVOID instinct."""
    itp = Interpreter()
    # Reflex takes priority over instinct
    cmd, reward = itp.interpret({"reflex": "STOP", "instinct": "AVOID", "match": "HOME"})
    assert cmd == "EMERGENCY_STOP", f"reflex STOP should beat AVOID, got {cmd}"
    assert reward == -1.0, f"reflex should have reward -1.0, got {reward}"


def test_command_priority_avoid_beats_match():
    """Instinct priority: AVOID beats match."""
    itp = Interpreter()
    # AVOID vetoes the match
    cmd, reward = itp.interpret({"instinct": "AVOID", "match": "GRIPPER_CLOSE"})
    assert cmd == "HOLD", f"AVOID should beat match, got {cmd}"
    assert reward == -0.3, f"AVOID should have reward -0.3, got {reward}"


def test_joint_b_rotate_match():
    """Match JOINT_B_ROTATE -> command "JOINT_B_ROTATE(-10deg)" (default mapping)."""
    itp = Interpreter()
    cmd, reward = itp.interpret({"match": "JOINT_B_ROTATE"})
    assert cmd == "JOINT_B_ROTATE(-10deg)", f"JOINT_B_ROTATE mapped wrong, got {cmd}"


def test_custom_commands_dict():
    """Interpreter can use custom command mapping."""
    custom_cmds = {
        "TEST_CMD": "CUSTOM_TEST_COMMAND",
    }
    itp = Interpreter(commands=custom_cmds)
    cmd, reward = itp.interpret({"match": "TEST_CMD"})
    assert cmd == "CUSTOM_TEST_COMMAND", f"custom command should map, got {cmd}"
