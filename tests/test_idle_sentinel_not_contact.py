"""A negative distance is a MISSING reading, not contact -- it may not bind a fear association.

The ESP32 firmware sends a -1 idle sentinel when nothing is within range
(`sensorimotor/esp32_encoder_path.py` documents it in `normalize`'s own docstring, where the
TEMPERATURE half of the same sentinel IS handled and the distance half is not).
`AutonomousArm.step()`'s safety early-return split the breach two ways -- NaN was a fail-closed
read, anything finite was "real contact" -- and -1.0 is finite and below `CONTACT`, so it fell on
the contact side. Every idle sample bound the amygdala to a context in which nothing was there:
a fabricated US, learned as real.

WHAT IS MEASURED, and the first choice was wrong in a way worth recording. The obvious observable
is `Amygdala.threat()`, and it is VACUOUS here: `threat()` is a thresholded CeA readout that
integrates `w_exc` against `TH_BLA`, so a single `pair()` -- five of them, measured -- leaves it at
exactly 0.0. A "nothing was bound" assertion read through it would have passed with the defect
fully live. The direct observable is what `pair()` WRITES: `w_exc`, the BLA association weights.

The three cases are in ONE file and asserted as a DISSOCIATION, not three separate markers, because
a negative alone cannot distinguish "the guard works" from "the harness is deaf":

    real contact (0.02 m)  -> E-STOP  AND  w_exc rises 0.0 -> 0.30   (positive control)
    idle sentinel (-1.0)   -> E-STOP  AND  w_exc unmoved              (the fix)
    clear (0.12 m)         -> no E-STOP, no binding                   (negative control)

NOT changed, and pinned so it stays that way: the E-STOP still fires on a negative reading. "No
object" and "the sensor said nothing" are different claims, and only the first is a reason to move.
The guard narrows what is LEARNED, never what STOPS.

Run: pytest tests/test_idle_sentinel_not_contact.py
"""

from snn_data_lake.engine.autonomous_loop import CONTACT, AutonomousArm

IDLE = -1.0  # the firmware idle sentinel: finite, below CONTACT, and NOT an object
NEAR = CONTACT * 0.5  # a real contact
FAR = CONTACT * 3.0  # comfortably clear


def _tick(distance):
    """One tick at this distance on a fresh arm. Returns (action, bound_weight_delta).

    Fresh arm per call so the E-STOP latch of one case cannot alter the control flow of another.
    """
    arm = AutonomousArm(seed=0, amygdala=True)
    pat = arm._context_pattern(None, None, None, distance)
    arm._amygdala_for(pat)  # size the detector first, exactly as the live path does
    before = sum(arm.amyg.w_exc[0])
    action = arm.step(None, None, None, distance)["action"]
    return action, sum(arm.amyg.w_exc[0]) - before


def test_real_contact_binds_an_association():
    """POSITIVE CONTROL. Without this the test below proves only that the harness cannot see."""
    action, delta = _tick(NEAR)
    assert action == "EMERGENCY_STOP"
    assert delta > 0.0


def test_idle_sentinel_binds_nothing():
    """THE FIX. -1.0 is finite and < CONTACT, so before the guard it reached `am.pair(pat)`."""
    assert _tick(IDLE)[1] == 0.0


def test_idle_sentinel_still_stops_the_arm():
    """The guard narrows what is LEARNED, never what STOPS -- an untrusted reading still stops."""
    assert _tick(IDLE)[0] == "EMERGENCY_STOP"


def test_a_clear_reading_neither_stops_nor_binds():
    """NEGATIVE CONTROL: the binding is specific to the breach, not a per-tick side effect."""
    action, delta = _tick(FAR)
    assert action != "EMERGENCY_STOP"
    assert delta == 0.0
