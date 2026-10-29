"""specs/022 T102 (R1) -- static-lane tests: the A2 current-schema C8 fixture (r1-bounded-recall.md
A2) and, below it, the A4 generator/oracle core.

Brain-free and lake-free: nothing here builds a Brain or opens a lake. The fixture is judged by the
REAL admission check (`learner_events._refusal`), never by a copy of its rules.
"""

import pytest
from r1_c8_fixture import (
    N_LEGACY_TEACH,
    distractor_events,
    legacy_teach_records,
    refusal_arm_records,
)

from snn_data_lake.substrate import home as H
from snn_data_lake.substrate import learner_events as LE


def test_distractors_cover_every_supported_kind_exactly_once():
    kinds = [e["kind"] for e in distractor_events()]
    assert sorted(kinds) == sorted(LE.KIND_SPECS)
    assert set(kinds) <= H.LEARNER_EVENT_KINDS


@pytest.mark.parametrize("event", distractor_events(), ids=lambda e: e["kind"])
def test_each_distractor_is_complete_yet_classified_invalid(event):
    # A2: every current schema / identity / encoder field is present, so the ONLY reason to drop
    # the row is valid="0" -- classified invalid, not a critical admission refusal
    assert LE._refusal(event, LE.KIND_SPECS[event["kind"]], False) == "invalid"


def test_distractors_are_excluded_from_recall_by_the_one_predicate():
    assert not any(H.is_autobiographical_episode(e) for e in distractor_events())


def test_legacy_teach_upcasts_and_is_admitted_under_the_current_contract():
    teach = legacy_teach_records()
    assert len(teach) == N_LEGACY_TEACH
    for e in teach:
        up = LE._upcast(e)
        assert up["schema"] == LE.SCHEMA and up["split"] == "train"
        assert LE._refusal(up, LE.KIND_SPECS["teach"], False) is None
    assert not any(H.is_autobiographical_episode(e) for e in teach)


@pytest.mark.parametrize(("reason", "record"), refusal_arm_records())
def test_refusal_arm_rows_are_refused_for_the_named_reason(reason, record):
    spec = LE.KIND_SPECS[record["kind"]]
    assert LE._refusal(LE._upcast(record), spec, False) == reason


def test_refusal_arm_has_the_minimal_old_form_for_every_kind():
    minimal = [r for reason, r in refusal_arm_records() if reason == "missing_slot"]
    assert sorted(r["kind"] for r in minimal) == sorted(LE.KIND_SPECS)
    assert all(set(r) == {"kind", "event_id", "session", "seq"} for r in minimal)
