"""The arm's command vocabulary is defined in several places, and they must not disagree.

`interpreter.COMMANDS` produces tokens, `serial_arm` allow-lists them on the wire, the arm's own
parser dispatches them, and `autonomous_loop` mints them directly. Each holds its own idea of
which joints exist. Nothing checks that those ideas match, which is how a two-joint vocabulary
came to be spelled out independently in eight places.

WHAT THIS ASSERTS, AND WHY IT IS SHAPED THIS WAY. It does not hardcode the joint set. It extracts
each site's set and asserts they are all EQUAL to each other. That invariant is true today (all
two-joint) and must still be true after the six-joint migration, so the test is meaningful in both
states -- and the state it fails in is exactly the dangerous one: a PARTIAL migration where one
site learned about the new joints and another did not.

  pytest tests/test_arm_vocabulary.py
"""

import re

from snn_data_lake.brainstem import arm_sim, interpreter, serial_arm

_JOINT_TOKEN = re.compile(r"JOINT_([A-Z_]+)_ROTATE")


def _joints_in(strings):
    """Every joint identifier mentioned by a collection of command strings."""
    found = set()
    for s in strings:
        found.update(_JOINT_TOKEN.findall(s))
    return found


def _interpreter_joints():
    return _joints_in(interpreter.COMMANDS.values())


def _autonomous_joints():
    from snn_data_lake.brainstem import autonomous_loop

    return _joints_in(autonomous_loop.ACTIONS)


def _parser_accepts(token):
    """Does the arm's own dispatch parser recognise this token as a joint move?"""
    return arm_sim._JOINT.match(token) is not None


def _wire_accepts(token):
    """Does the wire allow-list admit this token?"""
    return serial_arm._JOINT.match(token) is not None


def test_every_site_names_the_same_joints():
    """A joint known to one site and not another is a partial migration."""
    sites = {
        "interpreter.COMMANDS": _interpreter_joints(),
        "autonomous_loop.ACTIONS": _autonomous_joints(),
    }
    sets = list(sites.values())
    assert all(s for s in sets), f"a site names no joints at all: {sites}"
    assert all(s == sets[0] for s in sets), (
        f"sites disagree about which joints exist: {sites}. "
        "One of them has been migrated and the others have not."
    )


def test_every_produced_token_is_accepted_by_parser_and_wire():
    """A site must not be able to mint a token that the gate or the wire would reject."""
    produced = set(interpreter.COMMANDS.values()) | set(_autonomous_joints_tokens())
    rejected_by_parser = [t for t in produced if _JOINT_TOKEN.search(t) and not _parser_accepts(t)]
    rejected_by_wire = [t for t in produced if _JOINT_TOKEN.search(t) and not _wire_accepts(t)]
    assert not rejected_by_parser, (
        f"tokens produced but NOT parseable by the arm: {rejected_by_parser}"
    )
    assert not rejected_by_wire, (
        f"tokens produced but NOT on the wire allow-list: {rejected_by_wire}"
    )


def _autonomous_joints_tokens():
    from snn_data_lake.brainstem import autonomous_loop

    return [a for a in autonomous_loop.ACTIONS if _JOINT_TOKEN.search(a)]


def test_hold_asymmetry_is_deliberate_and_documented():
    """HOLD is on the wire allow-list but is not an interpreter command and not a parser case.

    That asymmetry is intentional -- HOLD falls through to the no-op branch so the commanded
    trajectory reference stays put. Pinned here so it stays a decision rather than becoming an
    accident somebody 'fixes'.
    """
    assert "HOLD" in serial_arm._FIXED, "HOLD left the wire allow-list"
    assert "HOLD" not in interpreter.COMMANDS, "HOLD became an interpreter command"
    assert not _parser_accepts("HOLD"), "HOLD became a joint-move token"


def test_joint_table_is_the_single_vocabulary_source():
    """arm_config owns the joint table; it must be non-empty, ordered, and uniquely named."""
    from snn_data_lake.brainstem import arm_config

    names = [j.name for j in arm_config.JOINTS]
    assert names, "arm_config.JOINTS is empty"
    assert len(set(names)) == len(names), f"duplicate joint name in the table: {names}"
    assert [j.qpos_index for j in arm_config.JOINTS] == list(range(len(names))), (
        "JOINTS must be in qpos order -- the index is used to address the model"
    )
