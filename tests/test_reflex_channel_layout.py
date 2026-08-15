"""The reflex's channel LAYOUT is a safety contract, and nothing in the tree pinned it.

`brainstem/reflex.py::DEFAULT_RULES` maps a raw frame INDEX to a danger band and an action:

    6: (-50.0, 800.0, "STOP")       # motor current: collision / stall spike
    7: (-300.0, 300.0, "WITHDRAW")  # force/torque: over-range in either direction

Two different physical quantities, two different bands, two different protective actions. The index
is the only thing connecting a wire to its rule -- so **renumbering the frame changes what STOP
means**, silently, with every gate green. Measured before writing this file:
`grep -c DEFAULT_RULES tests/test_reflex.py` = 0, against a positive control of a file that does
exercise `Reflex`. The constant existed, was safety-critical, and was pinned nowhere.

WHY PINNING THE VALUES IS NOT ENOUGH, and it is the whole point of this file. A test that asserts
the numbers still catches an edit to the numbers; it does not catch a LAYOUT SHIFT, where the bands
stay identical and the wires move underneath them. So the meanings are pinned BEHAVIOURALLY, by two
readings chosen to be asymmetric across the two channels:

    +500.0  ->  SAFE on ch6 (inside -50..800)      DANGEROUS on ch7 (outside -300..300)
    -100.0  ->  DANGEROUS on ch6 (below -50)       SAFE on ch7 (inside -300..300)

Each is safe on one channel and dangerous on the other, so a 6<->7 swap inverts BOTH outcomes and
cannot pass. That asymmetry is not incidental -- it is what the bands mean. ch6's band is
asymmetric because motor current is mostly positive with a small reverse allowance; ch7's is
symmetric because an over-range force is equally dangerous in either direction.

`reflex.py` is one of the five approval-gated safety-envelope files and is NOT modified by this
work. This file only observes it.

Run: pytest tests/test_reflex_channel_layout.py
"""

from snn_data_lake.brainstem.reflex import DEFAULT_RULES, Reflex

_WIDTH = 8  # a frame wide enough to carry both rule channels
_SAFE_ON_6_DANGEROUS_ON_7 = 500.0
_DANGEROUS_ON_6_SAFE_ON_7 = -100.0


def _frame(channel, value):
    f = [0.0] * _WIDTH
    f[channel] = value
    return f


def _fires(rules, channel, value, steps=15):
    """The action this reading provokes, or None. Stepped `steps` times because a MARGINAL breach
    fires only after it persists -- the persistence latch is deliberate and must not be defeated by
    a single-step probe reporting a false negative."""
    r = Reflex(rules=rules)
    for _ in range(steps):
        action = r.step(_frame(channel, value))
        if action:
            return action
    return None


def test_default_rules_values_are_pinned():
    """The bands and actions themselves. Catches an edit to the numbers."""
    assert DEFAULT_RULES == {
        6: (-50.0, 800.0, "STOP"),
        7: (-300.0, 300.0, "WITHDRAW"),
    }


def test_channel_meanings_are_behaviourally_distinct():
    """The MEANINGS: each probe reading is safe on one channel and dangerous on the other."""
    assert _fires(None, 6, _SAFE_ON_6_DANGEROUS_ON_7) is None
    assert _fires(None, 7, _SAFE_ON_6_DANGEROUS_ON_7) == "WITHDRAW"
    assert _fires(None, 6, _DANGEROUS_ON_6_SAFE_ON_7) == "STOP"
    assert _fires(None, 7, _DANGEROUS_ON_6_SAFE_ON_7) is None


def test_a_layout_shift_is_DETECTABLE():
    """THE RED, demonstrated rather than asserted: swap the two channels' bands and every one of
    the four readings above inverts. A renumber cannot be silent while this test exists."""
    swapped = {6: DEFAULT_RULES[7], 7: DEFAULT_RULES[6]}
    assert _fires(swapped, 6, _SAFE_ON_6_DANGEROUS_ON_7) == "WITHDRAW"  # was None
    assert _fires(swapped, 7, _SAFE_ON_6_DANGEROUS_ON_7) is None  # was WITHDRAW
    assert _fires(swapped, 6, _DANGEROUS_ON_6_SAFE_ON_7) is None  # was STOP
    assert _fires(swapped, 7, _DANGEROUS_ON_6_SAFE_ON_7) == "STOP"  # was None


def test_a_narrow_frame_disables_a_safety_channel_LOUDLY():
    """A frame too narrow to carry ch7 leaves its rule INACTIVE -- reported, never silently
    skipped. This is the layout failure needing no edit at all: the wires did not move, the
    frame shrank."""
    r = Reflex()
    r.step([0.0] * 7)  # carries ch6 but not ch7
    assert 7 in r.missing, "a disabled safety channel must be surfaced, not hidden"
    assert 6 not in r.missing
