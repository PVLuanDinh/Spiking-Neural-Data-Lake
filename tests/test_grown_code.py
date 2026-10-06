"""specs/022 T111 (G3) P1: the encoding contract's grown segment.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.3 (the frozen 52-rank
code is never re-ranked; grown units add a SEPARATE latency segment; a capacity unit's latency is a
pure function of the frozen code and its own parameters; `grown_version` changes on every admission,
silencing or weight change; `apply_unit_row` moves the segment with no RNG), 2.9 C1 (old codes
unchanged; widening n_in is the red) and C17 (Dale's sign holds on every grown output synapse: a
planted flip is clipped at deposit and refused at restore). Fixture seed 9900; nothing here builds a
Brain or touches a lake.
"""

import pytest

from snn_data_lake.growth.grown_code import (
    GrownUnit,
    apply_unit_row,
    clip_to_sign,
    grown_version,
    latency,
    segment_psp,
)
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features

TAU = 8.0
TEXTS = [
    "cat dog sun sky",
    "cat dog sky sun",
    "dog cat sun sky",
    "dog cat sky sun",
    "hello there",
    "sky",
]


def _ranks(n_in, text):
    learner = SpikingInteractionLearner(["A", "B"], n_in=n_in, seed=9900, t_post=6)
    return tuple(learner.encode(text_features(text, n_in)))


def _unit(uid, w, threshold=1.0, sign=1, out=(0.0, 0.0), window=12):
    return GrownUnit(
        unit_id=uid, w_in=tuple(w), threshold=threshold, window=window, sign=sign, out=tuple(out)
    )


def _one_hot(n, i, v=2.0):
    w = [0.0] * n
    w[i] = v
    return w


# ---- C1: old codes unchanged


def test_frozen_encoder_read_twice_is_identical():
    assert [_ranks(52, t) for t in TEXTS] == [_ranks(52, t) for t in TEXTS]


def test_widening_n_in_changes_the_old_code():
    # the red: re-bucketing by a wider n_in changes the first-52-synapse code of every probe text
    changed = sum(_ranks(52, t) != _ranks(62, t)[:52] for t in TEXTS)
    assert changed == len(TEXTS)


# ---- the latency segment


def test_latency_is_deterministic_and_pure():
    ranks = _ranks(52, TEXTS[0])
    u = _unit(1, _one_hot(52, ranks.index(0)))
    assert latency(u, ranks, TAU) == latency(u, ranks, TAU)
    assert (
        latency(u, ranks, TAU) == 1
    )  # the rank-0 input lands at t=0, its PSP at t=1 is 2*exp(-1/8) >= 1


def test_no_spike_contributes_nothing():
    ranks = _ranks(52, TEXTS[0])
    silent = _unit(1, [0.0] * 52)
    assert latency(silent, ranks, TAU) is None
    assert segment_psp((silent,), ranks, TAU, t_post_g=12) == [0.0]


def test_segment_is_in_unit_id_order_not_insertion_order():
    ranks = _ranks(52, TEXTS[0])
    a = _unit(2, _one_hot(52, ranks.index(0)))
    b = _unit(1, _one_hot(52, ranks.index(3)))
    assert segment_psp((a, b), ranks, TAU, 12) == segment_psp((b, a), ranks, TAU, 12)


def test_inactive_unit_is_silenced_in_the_segment():
    ranks = _ranks(52, TEXTS[0])
    u = _unit(1, _one_hot(52, ranks.index(0)))
    assert segment_psp((u.silenced(),), ranks, TAU, 12) == [0.0]
    assert segment_psp((u,), ranks, TAU, 12)[0] > 0.0


# ---- the version and the journal step


def test_version_changes_on_admission_silencing_and_weight_change():
    u = _unit(1, _one_hot(52, 0), out=(0.5, 0.0))
    v0 = grown_version(())
    v1 = grown_version((u,))
    v2 = grown_version((u.silenced(),))
    v3 = grown_version((_unit(1, _one_hot(52, 0), out=(0.6, 0.0)),))
    assert len({v0, v1, v2, v3}) == 4


def test_apply_unit_row_admits_silences_and_updates_without_rng():
    units = apply_unit_row((), {"op": "admit", "unit": _unit(3, _one_hot(52, 0), out=(0.1, 0.0))})
    units = apply_unit_row(units, {"op": "admit", "unit": _unit(1, _one_hot(52, 1))})
    assert [u.unit_id for u in units] == [1, 3]
    units = apply_unit_row(units, {"op": "out", "unit_id": 3, "out": (0.4, 0.2)})
    assert units[1].out == (0.4, 0.2)
    units = apply_unit_row(units, {"op": "silence", "unit_id": 1})
    assert not units[0].active


def test_apply_unit_row_refuses_a_reused_unit_id():
    units = apply_unit_row((), {"op": "admit", "unit": _unit(1, _one_hot(52, 0))})
    with pytest.raises(ValueError):
        apply_unit_row(units, {"op": "admit", "unit": _unit(1, _one_hot(52, 2))})


# ---- C17: Dale's sign on the grown output synapses


def test_clip_to_sign_clips_a_planted_flip():
    assert clip_to_sign([0.3, -0.2], sign=1) == [
        0.3,
        0.0,
    ]  # an E unit's negative synapse is clipped
    assert clip_to_sign([0.3, -0.2], sign=-1) == [
        0.0,
        -0.2,
    ]  # an I unit's positive synapse is clipped
    assert clip_to_sign([0.3, 0.1], sign=1) == [0.3, 0.1]  # sign-consistent deposits pass unchanged


def test_restore_refuses_a_stored_synapse_violating_its_sign():
    inhibitory = _unit(7, _one_hot(52, 0), sign=-1, out=(-0.1, 0.0))
    units = apply_unit_row((), {"op": "admit", "unit": inhibitory})
    with pytest.raises(ValueError, match="unit 7"):
        apply_unit_row(units, {"op": "out", "unit_id": 7, "out": (0.2, 0.0)})
    with pytest.raises(ValueError, match="unit 8"):
        apply_unit_row(
            (), {"op": "admit", "unit": _unit(8, _one_hot(52, 0), sign=1, out=(-0.3, 0.0))}
        )


# ---- cp1 #17: sign exactly +-1, every synapse finite, refused with ids


def test_a_sign_other_than_plus_or_minus_one_is_refused():
    with pytest.raises(ValueError, match="unit 4"):
        apply_unit_row(
            (), {"op": "admit", "unit": _unit(4, _one_hot(52, 0), sign=0, out=(0.1, -0.1))}
        )
    with pytest.raises(ValueError):
        clip_to_sign([0.1, -0.1], sign=0)


def test_non_finite_synapses_are_refused_at_admit_deposit_and_restore():
    nan = float("nan")
    with pytest.raises(ValueError, match=r"unit 5.*synapse 1"):
        apply_unit_row((), {"op": "admit", "unit": _unit(5, _one_hot(52, 0), out=(0.1, nan))})
    w = _one_hot(52, 0)
    w[3] = float("inf")
    with pytest.raises(ValueError, match="unit 6"):
        apply_unit_row((), {"op": "admit", "unit": _unit(6, w)})
    units = apply_unit_row((), {"op": "admit", "unit": _unit(9, _one_hot(52, 0))})
    with pytest.raises(ValueError, match=r"unit 9.*synapse 0"):
        apply_unit_row(units, {"op": "out", "unit_id": 9, "out": (nan, 0.0)})
    with pytest.raises(ValueError, match="synapse 1"):
        clip_to_sign([0.1, nan], sign=1)


# ---- admit_rows: the single append of the grown segment (2026-10-04 amendment; G7's AdmissionTicket) ----


def _labeled_batch(ids, signs=None, seed=13010):
    """Real G7 label records (cortex) and matching units; the fixture density is G7's, not a measured one."""
    from snn_data_lake.growth import placement as P

    signs = signs or [1] * (len(ids) - 1) + [-1]
    roles = ["E" if s > 0 else "I" for s in signs]
    labels = P.place_labeled(
        len(ids), unit_ids=ids, region="cortex", site=(0.5 / 14, 1.5 / 4), problem_id="p", batch_id=0,
        seed=seed, roles=roles, centre_um=(0.0, 0.0), density_per_mm2=2.7e4,
    )
    units = [GrownUnit(i, (0.0,) * 52, 1.0, 12, s, (0.0,)) for i, s in zip(ids, signs, strict=True)]
    return units, labels


def _ticket(labels):
    from snn_data_lake.gates.unit_label_gate import require_admissible

    return require_admissible(labels, resolve_problem=lambda p: True, known_unit=lambda u: False)


def test_admit_rows_appends_a_ticketed_batch_and_spends_the_ticket():
    from snn_data_lake.growth.grown_code import admit_rows

    units, labels = _labeled_batch([0, 1, 2])
    ticket = _ticket(labels)
    seg = admit_rows((), units, labels, ticket)
    assert [u.unit_id for u in seg] == [0, 1, 2]
    with pytest.raises(Exception, match="already consumed"):  # a spent ticket admits nothing
        admit_rows((), units, labels, ticket)


def test_admit_rows_refuses_what_the_ticket_does_not_cover_and_leaves_the_segment_unchanged():
    from snn_data_lake.growth.grown_code import admit_rows

    units, labels = _labeled_batch([0, 1, 2])
    other_units, other_labels = _labeled_batch([7, 8, 9])
    ticket = _ticket(labels)
    with pytest.raises(Exception, match="not covered"):  # a stranger's rows under this ticket
        admit_rows((), other_units, other_labels, ticket)
    ticket = _ticket(labels)
    changed = [dict(labels[0], layer="Layer 6")] + list(labels[1:])
    with pytest.raises(Exception, match="differs"):  # a row that changed since validation
        admit_rows((), units, changed, ticket)
    ticket = _ticket(labels)
    with pytest.raises(Exception, match="missing"):  # a validated record left out of the append
        admit_rows((), units[:2], labels[:2], ticket)
    ticket = _ticket(labels)  # control: the untouched batch still goes through on a fresh ticket
    assert len(admit_rows((), units, labels, ticket)) == 3


def test_admit_rows_refuses_a_unit_whose_sign_is_not_its_labels_and_does_not_spend_the_ticket():
    from snn_data_lake.growth.grown_code import admit_rows

    units, labels = _labeled_batch([0, 1, 2])
    flipped = [units[0], units[1], GrownUnit(2, (0.0,) * 52, 1.0, 12, 1, (0.0,))]  # label says I (-1)
    ticket = _ticket(labels)
    with pytest.raises(ValueError, match="Dale"):
        admit_rows((), flipped, labels, ticket)
    assert len(admit_rows((), units, labels, ticket)) == 3  # the refusal did not consume the ticket
