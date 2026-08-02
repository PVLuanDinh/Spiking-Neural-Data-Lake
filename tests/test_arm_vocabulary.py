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

from snn_data_lake.brainstem import interpreter, serial_arm, so101_arm

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
    """Does the arm's own dispatch parser recognise this token as a joint move?

    Asks so101_arm, not arm_sim: the incumbent is being retired and this gate must not go offline
    in that commit. The regex accepts both the two-joint and six-joint shapes, so this is a SHAPE
    check -- membership against arm_config.JOINTS is the gate's own job at apply().
    """
    return so101_arm._JOINT_TOKEN.fullmatch(token) is not None


def _wire_accepts(token):
    """Does the wire allow-list admit this token?

    fullmatch, matching serial_arm.is_valid_command. With .match this helper was LOOSER than the
    gate it checks, so a truncated command would pass the test and be rejected in production.
    """
    return serial_arm._JOINT.fullmatch(token) is not None


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


def test_exactly_one_arm():
    """P4: one body implementation. Checked structurally, not by name.

    An "arm implementation" here means a class that owns a pose and mutates it: it defines
    both `apply` and `state`. Searching for the string "arm" would have found the wrappers and
    missed a body called something else, which is the failure mode this test exists to catch --
    demo.py's private `Arm` sat in the tree for months precisely because nothing looked for it
    structurally.

    Two classes legitimately match, and the difference is that only ONE owns joint state:

      So101Arm  -- the body. Integrates physics, owns qpos, enforces the latch.
      SerialArm -- a command SINK. It has apply/state because it forwards them to a So101Arm
                   mirror; it owns no pose of its own, and the assertion below proves that by
                   requiring it to hold a mirror.
    """
    import ast
    import importlib
    import pathlib

    import pytest as _pytest

    root = pathlib.Path(so101_arm.__file__).resolve().parents[1]
    found = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # not our problem here; the lint gate owns syntax
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if {"apply", "state"} <= methods:
                found[node.name] = path.relative_to(root).as_posix()

    assert set(found) == {"So101Arm", "SerialArm"}, (
        f"expected exactly one body and one sink, found {found}"
    )
    assert found["So101Arm"] == "brainstem/so101_arm.py"

    # The sink must DELEGATE, not own a pose. If SerialArm ever grew its own joint state it
    # would be a second body wearing a wrapper's name, and the set check above would not notice.
    sink = pathlib.Path(root / found["SerialArm"]).read_text(encoding="utf-8")
    assert "self.mirror = So101Arm()" in sink, "the serial sink no longer delegates to the body"

    # And the retired one is really gone. Checked by IMPORTING it and by asking the atlas --
    # not by grepping, because the files that retired it say its name in their comments, so a
    # text search fails on the very note that records the retirement.
    assert not (root / "brainstem" / "arm_sim.py").exists(), "arm_sim.py is still on disk"
    with _pytest.raises(ImportError):
        importlib.import_module("snn_data_lake.brainstem.arm_sim")

    from snn_data_lake import brain_atlas

    listed = {m for mods in brain_atlas.REGIONS.values() for m in mods}
    assert "arm_sim" not in listed, "the atlas still lists the retired arm"
    assert "so101_arm" in listed, "the live arm is not in the atlas"


def test_the_retired_demo_arm_cannot_be_constructed():
    """demo.py kept a private 2-link arm with its own E-STOP. Constructing it must fail loudly."""
    import pytest as _pytest

    from snn_data_lake.substrate import demo

    assert not hasattr(demo, "Arm"), "the private demo arm is back under its old name"
    with _pytest.raises(RuntimeError, match="exactly one arm"):
        demo._RetiredArm()
