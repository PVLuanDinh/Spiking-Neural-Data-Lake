"""The safety path reads RAW METRES; the spike path reads the normalized feats. Nothing pinned it.

`engine/autonomous_loop.py::AutonomousArm.step` compares its `distance` argument against
`arm_config.CONTACT`, which is **0.04 metres** -- a raw physical limit inside the safety envelope,
fixed by design (POLICIES P1, fourth case).
`sensorimotor/esp32_encoder_path.py::normalize` divides the same reading by 3.0 to produce
the [0,1] feature the multimodal encoder consumes. Two units, one
quantity, and until this file NOTHING asserted which one crosses which boundary -- so a future
sensor wiring could hand the encoder's output to the safety comparison and every gate would stay
green.

MEASURED DIRECTION, and it CORRECTS the claim this task was filed under. specs/019 [B1.04] says the
substitution "loosens the trip point 3x". The 3x is exact and reproduced below; the word "loosens"
is wrong. `normalized < CONTACT` is `raw/3 < 0.04`, i.e. `raw < 0.12` -- the arm trips at THREE
TIMES the distance, so it stops MORE often, never less. Swept across the sensor range there is no
reading where the raw path trips and the normalized path does not.

So the honest disposition is the opposite of the filed one: **this is a capability defect (a 3x
false-stop radius), not a safety loosening, and the envelope is not breached by it.** It is still
worth pinning -- a silent 3x move in a safety threshold is a defect whichever way it points, and an
unasserted units contract is how it would arrive. The direction is asserted here too, so the record
cannot drift back.

Run: pytest tests/test_distance_units_contract.py
"""

from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.engine.autonomous_loop import AutonomousArm
from snn_data_lake.sensorimotor.esp32_encoder_path import normalize

_VALID_TEMP_K = 300.0  # inside the [20,40] C band, so the temp modality is present, not missing
_SWEEP = (0.005, 0.02, 0.039, 0.041, 0.05, 0.10, 0.119, 0.121, 0.5, 1.0, 2.0, 2.9)


def _trips(distance):
    """Does ONE tick at this distance trip the reflex?

    Fresh arm per call so no latch carries between cases."""
    return AutonomousArm(seed=0).step(None, None, None, distance)["action"] == "EMERGENCY_STOP"


def _norm(raw):
    return normalize(raw, _VALID_TEMP_K)[0]


def test_safety_path_reads_raw_metres():
    """The trip boundary is CONTACT in METRES -- pinned at runtime, on both sides of it."""
    assert _trips(CONTACT * 0.975)  # 0.039 m -- inside contact
    assert not _trips(CONTACT * 1.25)  # 0.050 m -- outside contact


def test_spike_path_receives_normalized_feats():
    """The encoder's output is a [0,1] feature across the whole sensor range, never metres."""
    for raw in _SWEEP:
        n = _norm(raw)
        assert 0.0 <= n <= 1.0, f"{raw} m normalized to {n}, outside [0,1]"
    # and it is genuinely a DIFFERENT number from the raw reading wherever the distinction matters
    assert _norm(0.10) != 0.10


def test_substituting_normalized_for_metres_moves_the_trip_point_3x():
    """THE RED. Feed the encoder's feature to the safety comparison and the boundary moves 3x."""
    ratio = 3.0  # normalize() divides by the 3.0 m object-present gate
    assert not _trips(0.10), "0.10 m is outside CONTACT and must not trip"
    assert _trips(_norm(0.10)), "the normalized 0.10 m reading trips -- the substitution is visible"
    # the boundary itself, stated as the arithmetic rather than as a single sampled point
    assert _norm(CONTACT * ratio) >= CONTACT  # 0.12 m is the first reading that does NOT trip
    assert _norm(CONTACT * ratio * 0.99) < CONTACT  # just inside it, it does


def test_the_substitution_is_strictly_MORE_conservative_not_less():
    """Pins the DIRECTION, because the task was filed with it backwards.

    If any reading existed where the raw path trips and the normalized path does not, the
    substitution would be a genuine safety loosening. Swept: there is none."""
    looser = [raw for raw in _SWEEP if _trips(raw) and not _trips(_norm(raw))]
    assert looser == [], f"readings where the substitution would SKIP a real stop: {looser}"
    stricter = [raw for raw in _SWEEP if not _trips(raw) and _trips(_norm(raw))]
    assert stricter, "the sweep must contain at least one extra false stop, or it proves nothing"
