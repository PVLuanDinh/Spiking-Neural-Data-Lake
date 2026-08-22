"""specs/022 T039g: p6_gate must not be blind to a PUBLIC ACCESSOR.

THE DEFECT, measured before the fix through `p6_gate.measure(overlay=...)` -- four arms, one new
FACULTY module each, nothing written to disk:

    A  self.brain._reentrant.broadcast()    -> 1 site, CROSS, owner cortex/brain.py, fac_cross 0->1
    B  self.brain.reentrant.broadcast()     -> NO SITE AT ALL
    C  self.brain.reentrant.broadcast_to()  -> NO SITE AT ALL
    D  self.brain._nonesuch.broadcast()     -> NO SITE AT ALL      <- the NEGATIVE CONTROL

B and C were indistinguishable from D: reading a real workspace through a clean accessor looked
exactly like referencing a name nothing constructs, while reaching past the accessor into the
owner's PRIVATE attribute scored. A gate that can only see the worse of two identical routes is
pushing the tree toward worse structure, and SC-001 was reachable only by writing the worse one.

This is the same class of change the walk already took once for local and attribute ALIASES, and
its header records why that one mattered even at zero delta: a behaviour-preserving hoist must not
change what the gate can see. The tests below are the standing version of that argument.

Run: pytest tests/test_p6_accessor_attribution.py
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

p6_gate = pytest.importorskip("p6_gate")

PROBE = "cortex/_probe_accessor.py"
HDR = '"""overlay-only probe module -- never written to disk."""\n\n'


def _mod(body):
    return HDR + "class Probe:\n    def __init__(self, brain):\n        self.brain = brain\n" + body


def _measure(body):
    res = p6_gate.measure(ROOT, overlay={PROBE: _mod(body)})
    mine = {
        "cross": [c for c in res["cross"] if c[0] == PROBE],
        "unattr": [c for c in res["unattributable"] if c[0] == PROBE],
        "self": [c for c in res["selfread"] if c[0] == PROBE],
    }
    mine["sites"] = len(mine["cross"]) + len(mine["unattr"]) + len(mine["self"])
    mine["fac_cross"] = len(res["fac_cross"])
    return mine


@pytest.fixture(scope="module")
def base():
    return len(p6_gate.measure(ROOT)["fac_cross"])


def test_the_private_route_is_attributed(base):
    """The route that ALREADY worked. Pinned so the fix cannot be shown to work by breaking it."""
    m = _measure("    def step(self):\n        return self.brain._reentrant.broadcast()\n")
    assert m["sites"] == 1, m
    assert len(m["cross"]) == 1 and m["cross"][0][4] == "cortex/brain.py", m["cross"]
    assert m["fac_cross"] == base + 1


def test_the_PUBLIC_accessor_route_is_now_attributed_the_same_way(base):
    """THE FIX. Same object, same call, same route -- through the accessor instead of past it."""
    m = _measure("    def step(self):\n        return self.brain.reentrant.broadcast()\n")
    assert m["sites"] == 1, (
        "reading a workspace through a public accessor still produces no site -- the gate is "
        "blind to the clean route and rewards poking a private attribute"
    )
    assert len(m["cross"]) == 1 and m["cross"][0][4] == "cortex/brain.py", m["cross"]
    assert m["fac_cross"] == base + 1


def test_the_accessor_route_works_for_the_drain_shape_a_specialist_uses(base):
    """broadcast_to(name) is how a subscribed specialist actually reads. Checked separately from
    broadcast() because the API table keys on (method, arity) and one can be live while the
    other is not."""
    m = _measure("    def step(self):\n        return self.brain.reentrant.broadcast_to('x')\n")
    assert m["sites"] == 1, m
    assert len(m["cross"]) == 1 and m["cross"][0][4] == "cortex/brain.py", m["cross"]
    assert m["fac_cross"] == base + 1


def test_THE_NEGATIVE_CONTROL_still_produces_nothing(base):
    """A name nothing constructs must stay invisible. Without this, 'the accessor route is now
    visible' could be satisfied by a walk that has simply started counting everything -- which is
    precisely how B and C were indistinguishable from D before the fix, only in the other
    direction."""
    m = _measure("    def step(self):\n        return self.brain._nonesuch.broadcast()\n")
    assert m["sites"] == 0, m
    assert m["fac_cross"] == base


def test_a_computed_property_is_NOT_treated_as_a_rename(base):
    """THE REACHABLE RED FOR THE EXTRACTOR ITSELF. Only `return self.<attr>` is a rename of an
    attribute. A body that computes or walks a getattr chain returns a DIFFERENT object, and
    `Brain.workspace` is the live example -- it returns the ARM's GlobalWorkspace, not
    `self._reentrant`, and its own docstring says so. Treating it as an alias for `_reentrant`
    would attribute reads to the wrong owner, which is worse than not seeing them."""
    probe = HDR + (
        "class Probe:\n"
        "    def __init__(self, brain):\n"
        "        self.brain = brain\n"
        "        self._x = None\n"
        "    @property\n"
        "    def computed(self):\n"
        "        return getattr(getattr(self, '_x', None), 'reentrant', None)\n"
        "    def step(self):\n"
        "        return self.computed.broadcast()\n"
    )
    res = p6_gate.measure(ROOT, overlay={PROBE: probe})
    mine = [c for c in res["cross"] if c[0] == PROBE]
    assert not mine, f"a computed property was resolved as if it renamed an attribute: {mine}"


def test_an_accessor_defined_in_two_modules_is_ambiguous_not_guessed(base):
    """Ambiguity returns None, exactly as it does for construction tokens. Two modules exposing the
    same accessor name is not an answer to 'who owns this instance', and guessing one would be a
    false attribution -- the failure mode the walk already refuses for the token `workspace`."""
    twin = HDR + (
        "class Twin:\n"
        "    def __init__(self):\n"
        "        self._reentrant = None\n"
        "    @property\n"
        "    def reentrant(self):\n"
        "        return self._reentrant\n"
    )
    body = "    def step(self):\n        return self.brain.reentrant.broadcast()\n"
    res = p6_gate.measure(ROOT, overlay={PROBE: _mod(body), "cortex/_probe_twin.py": twin})
    mine_cross = [c for c in res["cross"] if c[0] == PROBE]
    mine_un = [c for c in res["unattributable"] if c[0] == PROBE]
    assert not mine_cross, f"an ambiguous accessor name was attributed anyway: {mine_cross}"
    assert mine_un, "the ambiguous read vanished entirely instead of being reported"


def test_brain_exposes_the_accessor_this_gate_can_now_see():
    """The tree half. A gate able to see accessors buys nothing while the owner exposes none, and
    a reader must not have to reach past it."""
    from snn_data_lake.cortex.brain import Brain

    assert isinstance(Brain.reentrant, property), (
        "Brain.reentrant must be a property so a subscribed specialist can reach the re-entrant "
        "workspace without touching Brain._reentrant"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
