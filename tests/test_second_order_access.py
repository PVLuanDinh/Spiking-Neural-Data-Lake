"""
Edge tests for cortex/second_order_access.py: the SECOND-ORDER access monitor reads high
when content reliably reached the workspace, reads zero when ignition FAILED (even under a
loud first-order report), holds on weak/masked access, calibrates above chance on a mixed
stream, and is deterministic. The full un-riggable protocol with the failing controls and
the printed numbers lives in the module's own __main__ self-check; this file is the
narrower pytest-style coverage the rest of tests/ uses.

Run: pytest tests/test_second_order_access.py
"""

import random

from snn_data_lake.cortex.second_order_access import _BANNED, SecondOrderAccess


def test_strong_reliable_access_reads_high_and_does_not_hold():
    m = SecondOrderAccess(seed=0)
    conf = m.observe(availability=1.0, access_strength=0.9, ignited=True, first_order_conf=0.9)

    assert 0.0 <= conf <= 1.0
    assert conf > 0.6  # clean ignition + sustained + confident -> high second-order confidence
    assert not m.hold()
    assert m.second_order_confidence() == conf


def test_failed_ignition_silences_the_pool_even_with_a_loud_first_order_report():
    m = SecondOrderAccess(seed=0)
    conf = m.observe(availability=0.0, access_strength=0.0, ignited=False, first_order_conf=0.95)

    assert conf == 0.0  # no ignition -> the first-order read is about nothing
    assert m.hold()


def test_weak_and_masked_access_reads_below_clean_access():
    strong = SecondOrderAccess(seed=0).observe(1.0, 0.9, True, 0.9)
    masked = SecondOrderAccess(seed=0).observe(0.25, 0.10, True, 0.2)
    weak = SecondOrderAccess(seed=0).observe(1.0, 0.05, True, 0.1)

    assert masked < strong  # reached a minority of taps
    assert weak < strong  # broadcast but the re-entrant sustain is gone


def test_confidence_rises_monotonically_with_access_strength():
    confs = [
        SecondOrderAccess(seed=0).observe(1.0, s, True, 0.5) for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(confs[i + 1] >= confs[i] - 1e-9 for i in range(len(confs) - 1)), confs
    assert confs[-1] > confs[0]


def test_confidence_is_an_exact_spike_count_fraction():
    m = SecondOrderAccess(seed=0)
    conf = m.observe(1.0, 0.8, True, 0.8)
    product = conf * m.pop_size * m.steps
    assert abs(product - round(product)) < 1e-9  # spike-derived, not a continuous membrane read
    assert round(product) == m._spikes


def test_calibration_starts_at_an_honest_zero_and_tracks_agreement():
    m = SecondOrderAccess(seed=0)
    assert m.calibration() == 0.0  # no evidence yet -- an honest gap, not a score
    assert m.n_observed() == 0

    for _ in range(10):  # clearly available content
        m.observe(1.0, 0.9, True, 0.9)
    for _ in range(10):  # clearly failed access
        m.observe(0.0, 0.0, False, 0.2)

    assert m.n_observed() == 20
    assert m.calibration() == 1.0  # the easy poles are called correctly


def test_calibration_on_a_mixed_stream_beats_chance():
    rng = random.Random(0)
    m = SecondOrderAccess(seed=0)
    truths = []
    for _ in range(60):
        if rng.random() < 0.5:
            m.observe(1.0, rng.uniform(0.7, 0.95), True, rng.uniform(0.7, 0.95))
            truths.append(True)
        else:
            m.observe(0.0, 0.0, False, rng.uniform(0.0, 0.95))
            truths.append(False)

    assert m.calibration() > 0.75  # clearly better than the 0.5 chance of a shuffled pairing
    assert 0.0 <= m.calibration() <= 1.0


def test_state_is_functional_only_with_no_phenomenal_key():
    m = SecondOrderAccess(seed=0)
    m.observe(1.0, 0.9, True, 0.9)
    st = m.state()

    assert set(st) == {
        "second_order_confidence",
        "second_order_hold",
        "second_order_calibration",
        "second_order_n",
    }
    for key in st:
        assert all(token not in key for token in _BANNED)
    assert st["second_order_hold"] is False
    assert st["second_order_n"] == 1


def test_inputs_are_clamped_and_confidence_stays_in_range():
    m = SecondOrderAccess(seed=0)
    for avail, strength, conf1 in ((5.0, 5.0, 5.0), (-3.0, -3.0, -3.0), (0.5, 0.5, 0.5)):
        for ignited in (True, False):
            c = m.observe(avail, strength, ignited, conf1)
            assert 0.0 <= c <= 1.0


def test_same_seed_and_protocol_is_deterministic():
    def build(seed):
        m = SecondOrderAccess(seed=seed)
        rng = random.Random(1)
        out = []
        for _ in range(15):
            out.append(m.observe(rng.random(), rng.random(), rng.random() < 0.6, rng.random()))
        return out, m.state()

    assert build(0) == build(0)
    assert build(0) != build(3)  # a different seed gives a different bias heterogeneity
