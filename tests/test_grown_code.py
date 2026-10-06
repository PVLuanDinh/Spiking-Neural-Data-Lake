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


# ---- admit_rows: the single append of the grown segment (2026-10-04 amendment) ----


def _labeled_batch(ids, signs=None, seed=13010):
    """Real G7 label records (cortex) and matching units; the density is a fixture value."""
    from snn_data_lake.growth import placement as P

    signs = signs or [1] * (len(ids) - 1) + [-1]
    roles = ["E" if s > 0 else "I" for s in signs]
    labels = P.place_labeled(
        len(ids),
        unit_ids=ids,
        region="cortex",
        site=(0.5 / 14, 1.5 / 4),
        problem_id="p",
        batch_id=0,
        seed=seed,
        roles=roles,
        centre_um=(0.0, 0.0),
        density_per_mm2=2.7e4,
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
    flipped = [
        units[0],
        units[1],
        GrownUnit(2, (0.0,) * 52, 1.0, 12, 1, (0.0,)),
    ]  # label says I (-1)
    ticket = _ticket(labels)
    with pytest.raises(ValueError, match="Dale"):
        admit_rows((), flipped, labels, ticket)
    assert len(admit_rows((), units, labels, ticket)) == 3  # the refusal did not consume the ticket


def test_admit_rows_refuses_a_planted_unlabeled_row():
    # ruling 29's charge proof: the gate's ticket is consumed INSIDE admit_rows, so a row
    # stripped of its label after validation is refused there (an import proves reachability only)
    from snn_data_lake.gates.unit_label_gate import require_admissible
    from snn_data_lake.growth.grown_code import admit_rows

    units, labels = _labeled_batch([0, 1, 2])
    ticket = _ticket(labels)
    stripped = [{k: v for k, v in labels[0].items() if k != "layer"}] + list(labels[1:])
    with pytest.raises(Exception, match="differs"):
        admit_rows((), units, stripped, ticket)
    with pytest.raises(
        Exception, match="layer"
    ):  # and no ticket exists for an unlabeled batch at all
        require_admissible(stripped, resolve_problem=lambda p: True, known_unit=lambda u: False)


# ---- COL's wiring (G7's amendment (h)): the causal column evaluation; gamma = 0 IS BATCH9 ----


def _column(k=3, thresholds=(1.0, 1.0, 1.0, 1.0)):
    """k E units (sign +1) and one I unit (sign -1); unit j reads synapse j with weight 2.0."""
    units = [
        GrownUnit(j, tuple(2.0 if i == j else 0.0 for i in range(52)), thresholds[j], 12, 1, (0.0,))
        for j in range(k)
    ]
    units.append(
        GrownUnit(
            k, tuple(2.0 if i == k else 0.0 for i in range(52)), thresholds[k], 12, -1, (0.0,)
        )
    )
    return units


def _steps(*steps):
    """The frozen code's integer spike ranks: synapse j fires at step steps[j] (others: rank 99)."""
    return tuple(list(steps) + [99] * (52 - len(steps)))


def test_column_with_gamma_zero_is_batch9_byte_identically():
    from snn_data_lake.growth.grown_code import column_latencies

    units = _column()
    for ranks in (_steps(1, 2, 3, 2), _steps(5, 1, 7, 1), _steps(99, 99, 99, 99)):
        assert column_latencies(units, ranks, TAU, 0.0) == {
            u.unit_id: latency(u, ranks, TAU) for u in units
        }


def test_the_coupling_is_live_in_both_directions():
    from snn_data_lake.growth.grown_code import column_latencies

    # I->E: the I unit spikes at step 2; the E drive at step 10 (2*psp(1) = 1.76) sits just over
    # their threshold 1.7, and the inhibition (gamma*theta*psp(8), about 0.19) takes it under
    units = _column(thresholds=(1.7, 1.7, 1.7, 1.0))
    ranks = _steps(9, 9, 9, 1)
    plain = column_latencies(units, ranks, TAU, 0.0)
    coupled = column_latencies(units, ranks, TAU, 1 / 3)
    assert (
        plain[3] == coupled[3] == 2
    )  # the I unit's own latency is unchanged (no E has spiked before it)
    assert all(plain[e] == 10 for e in (0, 1, 2)) and all(coupled[e] != 10 for e in (0, 1, 2))
    # E->I: E units spike at step 1; the I unit (threshold 1.9, drive 2*psp(6) = 0.95 at step 7)
    # is lifted by gamma*theta_I*sum psp(t - 1)
    units = _column(thresholds=(1.0, 1.0, 1.0, 1.9))
    ranks = _steps(1, 1, 1, 7)
    plain = column_latencies(units, ranks, TAU, 0.0)
    coupled = column_latencies(units, ranks, TAU, 1 / 3)
    assert plain[3] is None or coupled[3] < plain[3]
    assert coupled[3] is not None


def test_the_evaluation_is_causal_a_later_spike_never_changes_an_earlier_latency():
    from snn_data_lake.growth.grown_code import column_latencies

    units = _column()
    base = column_latencies(units, _steps(1, 2, 3, 2), TAU, 1 / 3)
    # delay the LAST-spiking unit's afferent: units that spiked earlier keep their latencies
    moved = column_latencies(units, _steps(1, 2, 8, 2), TAU, 1 / 3)
    last = max((u for u in base if base[u] is not None), key=lambda u: base[u])
    earlier = {uid: t for uid, t in base.items() if t is not None and t < base[last]}
    assert earlier  # the fixture must have earlier spikes for this to be a test
    assert all(moved[uid] == t for uid, t in earlier.items())


def test_a_column_needs_one_i_unit_and_one_window_and_silenced_units_never_spike():
    from snn_data_lake.growth.grown_code import column_latencies

    units = _column()
    with pytest.raises(ValueError, match="exactly one"):
        column_latencies(units[:3], _steps(1, 1, 1), TAU, 0.1)
    odd = units[:3] + [GrownUnit(3, units[3].w_in, 1.0, 9, -1, (0.0,))]
    with pytest.raises(ValueError, match="window"):
        column_latencies(odd, _steps(1, 1, 1, 1), TAU, 0.1)
    silent = units[:2] + [units[2].silenced(), units[3]]
    assert column_latencies(silent, _steps(1, 1, 1, 1), TAU, 0.1)[2] is None


def test_column_latencies_do_not_depend_on_the_callers_unit_order_down_to_the_bytes():
    import json

    from snn_data_lake.growth.grown_code import column_latencies

    units = _column()
    ranks = _steps(1, 2, 3, 2)
    base = json.dumps(column_latencies(units, ranks, TAU, 1 / 3))
    for perm in (units[::-1], units[2:] + units[:2], [units[1], units[3], units[0], units[2]]):
        assert json.dumps(column_latencies(perm, ranks, TAU, 1 / 3)) == base
