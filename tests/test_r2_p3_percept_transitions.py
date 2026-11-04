"""R2 part P3 (r2-typed-replay-hydration.md 2.10): the grower's percept-transition record.
Dev seeds 15000-15007 only; NEVER a seed >= 15010.

Exercises the shipped percept schema and hydrate branch, including in restart children.
Missing-integration mutants must refuse; the tests never repair production functionality.
"""

import ast
import inspect
import json
import os
import random
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

DEV_SEED = 15000
WIDTH = 64


# ---- constructed inputs ------------------------------------------------------------------------
def _vec(seed, width=WIDTH):
    rng = random.Random(seed)
    return [rng.random() for _ in range(width)]


def _ev(
    session,
    seq,
    pseq,
    x,
    cat,
    *,
    split="train",
    valid="1",
    provenance="self",
    encoder=None,
    boot="9",
    flushed="9",
):
    from snn_data_lake.growth.neurogenesis_engine import pack_percept

    return {
        "kind": "percept",
        "event_id": f"{session}-{seq}",
        "stream": "wake",
        "session": str(session),
        "seq": str(seq),
        "schema": "r2.v1",
        "valid": valid,
        "split": split,
        "x": pack_percept(x),
        "category": str(cat),
        "n_protos": "9",
        "proto_boot": boot,
        "proto_flushed": flushed,
        "pseq": str(pseq),
        "provenance": provenance,
        "encoder": encoder or f"arm.percept gained width={len(x)}",
    }


def _hydrate(events, **kw):
    from snn_data_lake.growth.neurogenesis_engine import hydrate_transitions

    last = max((int(e["session"]) for e in events if e.get("session")), default=0)
    kw = {"width": WIDTH, "n_units": 9, "buffer_cap": 1000, "current_session": last + 1, **kw}
    return hydrate_transitions(events, **kw)


def _naive(events, *, across_sessions=True, dedupe=False, filtered=False):
    """The red arms: pair the list as given, with one guard removed at a time."""
    from snn_data_lake.growth.neurogenesis_engine import unpack_percept

    pairs, prev, seen = [], None, set()
    for e in events:
        if filtered and (e["split"] != "train" or e.get("category") in (None, "")):
            continue
        if dedupe and e["event_id"] in seen:
            continue
        seen.add(e["event_id"])
        if prev is not None and (across_sessions or prev["session"] == e["session"]):
            pairs.append((unpack_percept(prev["x"]), int(e["category"])))
        prev = e
    return pairs


def _pairs(out):
    return list(zip(out["x"], out["y"], strict=True))


def _chain(session, start_seq, cats, pseq0=1):
    return [
        _ev(session, start_seq + i, pseq0 + i, _vec(session * 100 + i), c)
        for i, c in enumerate(cats)
    ]


# ---- the reds of 2.10 / Codex 3: each constructed input must be refused or detected --------------
def test_cross_session_chain_never_pairs_across_a_restart():
    a, b = _chain(1, 1, [0, 1, 2]), _chain(2, 4, [3, 0, 1])
    out = _hydrate(a + b)
    assert len(_pairs(out)) == 4, "two chains of three make two pairs each, never five"
    assert len(_naive(a + b)) == 5, (
        "the CROSS arm joins the boundary: one pair more than the twin's"
    )
    assert _pairs(out) == _naive(a, across_sessions=False) + _naive(b, across_sessions=False)
    # an adversarial pseq that ALSO looks consecutive across the boundary is still not joined
    c = _chain(2, 4, [3, 0, 1], pseq0=4)
    assert len(_pairs(_hydrate(a + c))) == 4


def test_o_buf_without_hydration_the_buffer_is_empty_and_the_refit_differs():
    twin, restored, events = _drive_pair()
    assert events, "the twin's sink must have recorded its percepts"
    assert twin._buf_x and not restored._buf_x
    for _ in range(4):
        twin.tick()
        restored.tick()
    assert twin._model is not None and restored._model is None, "O_buf: no hydration, no model"


def test_dup_repeated_event_id_is_refused_and_forms_no_pair():
    chain = _chain(1, 1, [0, 1, 2, 0])
    planted = _ev(1, 2, 2, _vec(999), 3)  # byte-distinct, repeats event_id 1-2
    events = chain[:2] + [planted] + chain[2:]
    out = _hydrate(events)
    assert out["refusals"].get("duplicate_id") == 1
    assert _pairs(out) == _pairs(_hydrate(chain))
    assert _naive(events) != _naive(chain), "the guard-neutered arm differs from the twin's"


def test_reorder_a_regressing_seq_is_refused():
    chain = _chain(1, 1, [0, 1, 2, 0])
    shuffled = [chain[0], chain[2], chain[1], chain[3]]
    out = _hydrate(shuffled)
    assert out["refusals"].get("seq_regression", 0) >= 1
    assert _pairs(out) != _pairs(_hydrate(chain)), "shuffled order must not reproduce the twin's"


def test_heldout_endpoint_breaks_the_chain_no_bridge():
    a, mid, c = _chain(1, 1, [0, 1, 2])
    mid = dict(mid, split="heldout")
    out = _hydrate([a, mid, c])
    assert out["refusals"] == {"heldout": 1}
    assert _pairs(out) == [], "no (x_A, category_C) pair is ever formed across a held-out endpoint"
    assert len(_naive([a, mid, c], filtered=True)) == 1, "the filtered-list arm forms exactly that"


def test_missing_slot_endpoint_is_refused_and_breaks_the_chain():
    a, mid, c = _chain(1, 1, [0, 1, 2])
    broken = {k: v for k, v in mid.items() if k != "category"}
    out = _hydrate([a, broken, c])
    assert out["refusals"] == {"missing_slot": 1} and _pairs(out) == []
    assert len(_naive([a, broken, c], filtered=True)) == 1


def test_width_a_63_dim_percept_is_refused():
    a, mid, c = _chain(1, 1, [0, 1, 2])
    short = _ev(1, 2, 2, _vec(7, 63), 1, encoder="arm.percept gained width=64")
    out = _hydrate([a, short, c])
    assert out["refusals"] == {"width": 1} and _pairs(out) == []
    # width known to the engine wins over the event's own spelling
    assert _hydrate([a, mid, c], width=32)["refusals"] == {"width": 3}


def test_vocab_a_category_the_restored_arm_does_not_hold_is_refused():
    a, mid, c = _chain(1, 1, [0, 5, 2])
    out = _hydrate([a, mid, c], n_units=5)
    assert out["refusals"] == {"vocab": 1} and _pairs(out) == []
    assert len(_hydrate([a, mid, c], n_units=6)["x"]) == 2
    assert 5 not in out["categories"]


def test_instrument_provenance_is_refused():
    a, mid, c = _chain(1, 1, [0, 1, 2])
    mid = dict(mid, provenance="instrument")
    out = _hydrate([a, mid, c])
    assert out["refusals"] == {"instrument": 1} and _pairs(out) == []


def test_stale_proto_an_unflushed_crashed_tail_is_refused_not_hydrated():
    """2.11 D2. Session A recruits prototype 0 and writes percepts with it, then crashes before
    the flush; session B recruits a DIFFERENT prototype under the same id 0, flushes and closes;
    session C hydrates. VOCAB alone passes A's events (0 < n_units = 1): only the flushed-count
    proof refuses them."""
    a = [_ev(1, i + 1, i + 1, _vec(10 + i), 0, boot="0", flushed="0") for i in range(3)]
    b = [
        _ev(2, 4, 1, _vec(20), 0, boot="0", flushed="0"),
        _ev(2, 5, 2, _vec(21), 0, boot="0", flushed="0"),
        _ev(2, 6, 3, _vec(22), 0, boot="0", flushed="1"),  # B's flush landed before this event
        _ev(2, 7, 4, _vec(23), 0, boot="0", flushed="1"),
    ]
    twin = _hydrate(b, n_units=1, current_session=3)
    out = _hydrate(a + b, n_units=1, current_session=3)
    assert out["refusals"] == {"stale_proto": 3}, out["refusals"]
    assert _pairs(out) == _pairs(twin) and len(_pairs(out)) == 3
    assert len(_naive(a + b, across_sessions=False)) == 5, "VOCAB alone also hydrates A's tail"
    # no flush at all in B: the restoring process is session B+1 and its restored count proves it
    unflushed_b = [dict(e, proto_flushed="0") for e in b]
    assert len(_pairs(_hydrate(unflushed_b, n_units=1, current_session=3))) == 3
    # ... but a process two sessions on has no proof for B, and B's boot is never read for A
    c = [
        _ev(3, 8, 1, _vec(30), 0, boot="1", flushed="1"),
        _ev(3, 9, 2, _vec(31), 0, boot="1", flushed="1"),
    ]
    assert _hydrate(a + c, n_units=1, current_session=4)["refusals"] == {"stale_proto": 3}
    # C booted holding 1 prototype: that is B's proof (its recruit was durable after all) ...
    assert _hydrate(unflushed_b + c, n_units=1, current_session=4)["refusals"] == {}
    # ... while a C that booted with none leaves B's whole tail unproven
    c0 = [dict(e, proto_boot="0", proto_flushed="0") for e in c]
    assert _hydrate(unflushed_b + c0, n_units=1, current_session=4)["refusals"] == {
        "stale_proto": 4
    }


@pytest.mark.parametrize("boot,flushed", [("2", "2"), ("2", "0"), ("0", "2")])
def test_own_durable_proof_survives_a_teach_only_intervening_session(boot, flushed):
    """Panel A1: the last endpoint's OWN proof must survive arbitrary later restarts."""
    events = [_ev(1, i, i, _vec(i), i - 1, boot=boot, flushed=flushed) for i in (1, 2)]
    first = _hydrate(events, n_units=2, current_session=2)
    events.append({"kind": "teach", "session": "2"})
    third = _hydrate(events, n_units=2, current_session=3)
    assert first["y"] == [1] and not first["refusals"]
    assert third == first


def test_chat_boot_proves_only_the_immediately_preceding_session():
    """Panel B1's exact zero-count input, with refused/reused-id evidence controls."""
    events = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    chat = {
        "kind": "teach",
        "event_id": "2-4",
        "session": "2",
        "seq": "4",
        "stream": "chat",
        "schema": "r2.v1",
        "valid": "1",
        "split": "train",
        "source": "hello private r2",
        "target": "GREET",
        "provenance": "self",
        "encoder": "text_features n_in=52",
        "proto_boot": "1",
        "proto_flushed": "1",
    }
    twin = _hydrate(events, n_units=1, current_session=2)
    assert twin["y"] == [0, 0] and not twin["refusals"]
    assert _hydrate(events + [chat], n_units=1, current_session=3) == twin
    for changes in (
        {"proto_boot": "0"},  # session 2's reused id was flushed there, never in session 1
        {"session": "3", "seq": "4", "event_id": "3-4"},
        {"valid": "0"},
        {"split": "heldout"},
        {"provenance": "instrument"},
        {"proto_boot": "unknown"},
    ):
        out = _hydrate(events + [{**chat, **changes}], n_units=1, current_session=4)
        expected_refusals = {"stale_proto": 3}
        if changes == {"proto_boot": "unknown"}:
            expected_refusals["session_witness_malformed"] = 1
        assert out["y"] == [] and out["refusals"] == expected_refusals, changes


@pytest.mark.parametrize("kind", ["teach", "turn"])
@pytest.mark.parametrize("later_percepts", [False, True])
@pytest.mark.parametrize("chat_first", [False, True])
def test_conflicting_chat_boot_cannot_revive_crashed_prototype(kind, later_percepts, chat_first):
    """Re-panel A/B: schema-valid chat proof must agree with the session's percept boot."""
    a = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    b = [_ev(2, 4, 1, _vec(4), 0, boot="0", flushed="1")]
    if later_percepts:
        b += [_ev(2, i, i - 3, _vec(i), 0, boot="0", flushed="1") for i in (5, 6, 7)]
    chat = {
        "kind": kind,
        "event_id": "2-8",
        "session": "2",
        "seq": "8",
        "stream": "chat",
        "schema": "r2.v1",
        "valid": "1",
        "split": "train",
        "source": "hello private r2",
        "target": "GREET",
        "provenance": "self",
        "encoder": "text_features n_in=52",
        "proto_boot": "0",
        "proto_flushed": "1",
    }
    if chat_first:
        chat.update(event_id="2-4", seq="4")
        b = [dict(e, event_id=f"2-{int(e['seq']) + 1}", seq=str(int(e["seq"]) + 1)) for e in b]

    def history(witness):
        return a + ([witness] + b if chat_first else b + [witness])

    honest = _hydrate(history(chat), n_units=1, current_session=3)
    assert honest["y"] == ([0, 0, 0] if later_percepts else [])
    assert honest["refusals"] == {"stale_proto": 3}
    forged = _hydrate(history(dict(chat, proto_boot="1")), n_units=1, current_session=3)
    assert forged == {
        "x": [],
        "y": [],
        "categories": set(),
        "refusals": {
            "stale_proto": 3,
            "session_boot_conflict": 1,
            "session_witness_conflict": 4 if later_percepts else 1,
        },
    }, "A11 refuses the conflicted session's own pairs too"


def test_conflicting_chat_only_witness_is_refused_before_its_flush_proves_a_tail():
    a = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    chat = {
        "kind": "teach",
        "event_id": "2-4",
        "session": "2",
        "seq": "4",
        "stream": "chat",
        "schema": "r2.v1",
        "valid": "1",
        "split": "train",
        "source": "hello private r2",
        "target": "GREET",
        "provenance": "self",
        "encoder": "text_features n_in=52",
        "proto_boot": "0",
        "proto_flushed": "0",
    }
    forged = dict(chat, event_id="2-5", seq="5", proto_boot="1", proto_flushed="1")
    out = _hydrate(a + [chat, forged], n_units=1, current_session=3)
    assert out["y"] == []
    assert out["refusals"] == {"stale_proto": 3, "session_boot_conflict": 1}


def _durability_witness(kind, seq, *, boot="0", flushed="1"):
    if kind == "percept":
        return _ev(2, seq, seq - 3, _vec(seq), 0, boot=boot, flushed=flushed)
    return {
        "kind": kind,
        "event_id": f"2-{seq}",
        "session": "2",
        "seq": str(seq),
        "stream": "chat",
        "schema": "r2.v1",
        "valid": "1",
        "split": "train",
        "source": "hello private r2",
        "target": "GREET",
        "provenance": "self",
        "encoder": "text_features n_in=52",
        "proto_boot": boot,
        "proto_flushed": flushed,
    }


def _read_witness_history(events, monkeypatch):
    """Use the shipped strict reader, then the grower's actual restore path."""
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove
    from snn_data_lake.substrate import home, learner_events

    receipt = {
        "complete": True,
        "rows": [{"ordinal": i, "reason": "decoded"} for i in range(len(events))],
    }
    monkeypatch.setattr(home, "load_episodes", lambda **kw: (events, json.dumps(receipt).encode()))
    got, refusals, session, proof = learner_events.scan_learner_events(
        tuple(learner_events.KIND_SPECS), strict=True, return_receipt=True
    )
    assert got == events and not refusals and json.loads(proof)["complete"]
    engine = NeurogenesisSelfImprove(device="cpu", seed=DEV_SEED)
    engine.restore_transitions(got, n_units=1, current_session=session)
    return engine


@pytest.mark.parametrize(
    "first_kind,second_kind",
    [
        ("teach", "teach"),
        ("turn", "turn"),
        ("percept", "teach"),
        ("percept", "turn"),
        ("teach", "percept"),
        ("turn", "percept"),
        ("percept", "percept"),
    ],
)
@pytest.mark.parametrize("false_first", [False, True])
def test_a11_conflicting_boot_refuses_whole_session_in_either_order(
    first_kind, second_kind, false_first, monkeypatch
):
    """No first-witness authority, including a false first percept and its own pairs."""
    crashed = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    boots = ("1", "0") if false_first else ("0", "1")
    witnesses = [
        _durability_witness(first_kind, 4, boot=boots[0]),
        _durability_witness(second_kind, 5, boot=boots[1]),
        _ev(2, 6, 3, _vec(6), 0, boot="0", flushed="1"),
        _ev(2, 7, 4, _vec(7), 0, boot="0", flushed="1"),
    ]
    restored = _read_witness_history(crashed + witnesses, monkeypatch)
    assert restored._buf_x == restored._buf_y == [], "the entire conflicted session is refused"
    assert restored._categories == set()
    assert restored.transition_refusals.get("stale_proto") == 3
    assert restored.transition_refusals.get("session_boot_conflict", 0) >= 1


@pytest.mark.parametrize("kind", ["teach", "turn", "percept"])
def test_a11_agreeing_witnesses_keep_honest_history(kind, monkeypatch):
    """An independent positive control retains both older and same-session pairs."""
    earlier = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    witnesses = [
        _durability_witness(kind, 4, boot="1"),
        _durability_witness(kind, 5, boot="1"),
        _ev(2, 6, 3, _vec(6), 0, boot="1", flushed="1"),
        _ev(2, 7, 4, _vec(7), 0, boot="1", flushed="1"),
    ]
    restored = _read_witness_history(earlier + witnesses, monkeypatch)
    expected = (
        _naive(earlier + witnesses, across_sessions=False)
        if kind == "percept"
        else (_naive(earlier, across_sessions=False) + _naive(witnesses[2:], across_sessions=False))
    )
    assert list(zip(restored._buf_x, restored._buf_y, strict=True)) == expected
    assert restored._categories == {0} and not restored.transition_refusals


@pytest.mark.parametrize(
    "first_kind,second_kind",
    [
        ("teach", "teach"),
        ("turn", "turn"),
        ("percept", "teach"),
        ("percept", "turn"),
        ("teach", "percept"),
        ("turn", "percept"),
        ("percept", "percept"),
    ],
)
@pytest.mark.parametrize("flushed", [("1", "0", "0"), ("0", "2", "1")])
def test_a11_1_decreasing_flushed_counts_refuse_whole_session(
    first_kind, second_kind, flushed, monkeypatch
):
    """A decrease, including after a landed rise, invalidates all session proof."""
    events = [
        _durability_witness(first_kind, 4, flushed=flushed[0]),
        _durability_witness(second_kind, 5, flushed=flushed[1]),
        _ev(2, 6, 3, _vec(6), 0, boot="0", flushed=flushed[2]),
        _ev(2, 7, 4, _vec(7), 0, boot="0", flushed=flushed[2]),
    ]
    restored = _read_witness_history(events, monkeypatch)
    assert restored._buf_x == restored._buf_y == []
    assert restored._categories == set()
    assert restored.transition_refusals == {
        "session_flushed_conflict": 1,
        "session_witness_conflict": 2 + (first_kind == "percept") + (second_kind == "percept"),
    }


@pytest.mark.parametrize("kind", ["teach", "turn", "percept"])
def test_a11_1_landed_flush_rise_keeps_same_session_pairs(kind, monkeypatch):
    """A later landed flush proves earlier percepts even without a later percept."""
    events = [_ev(2, i, i - 3, _vec(i), 0, boot="0", flushed="0") for i in (4, 5, 6)]
    events += [
        _durability_witness(kind, 7, flushed="0"),
        _durability_witness(kind, 8, flushed="1"),
    ]
    # An unrelated teach-only third session prevents current n_units proving session 2.
    next_session = dict(_durability_witness("teach", 9, flushed="0"), session="3", event_id="3-9")
    restored = _read_witness_history(events + [next_session], monkeypatch)
    expected = _naive([e for e in events if e["kind"] == "percept"], across_sessions=False)
    assert list(zip(restored._buf_x, restored._buf_y, strict=True)) == expected
    assert restored._categories == {0} and not restored.transition_refusals


@pytest.mark.parametrize("kind", ["teach", "turn"])
@pytest.mark.parametrize("field", ["proto_boot", "proto_flushed"])
@pytest.mark.parametrize("bad", ["unknown", "", None, "-1", "1.0", 1, True, "\u0661"])
def test_a11_3_malformed_chat_witness_cannot_hide_a_contradiction(kind, field, bad, monkeypatch):
    """Lens A: present malformed optional metadata invalidates the whole session's proof."""
    crashed = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    false = _durability_witness("teach", 4, boot="1", flushed="1")
    truthful = _durability_witness(kind, 5, boot="0", flushed="0")
    truthful[field] = bad
    later = dict(
        _durability_witness("teach", 9, boot="0", flushed="1"),
        session="3",
        event_id="3-9",
    )
    restored = _read_witness_history(crashed + [false, truthful, later], monkeypatch)
    assert restored._buf_x == restored._buf_y == []
    assert restored._categories == set()
    assert restored.transition_refusals.get("session_witness_malformed") == 1
    assert restored.transition_refusals.get("stale_proto") == 3


@pytest.mark.parametrize("kind", ["teach", "turn", "percept"])
@pytest.mark.parametrize("field", ["proto_boot", "proto_flushed"])
@pytest.mark.parametrize("bad", ["unknown", "", None, "-1", "1.0", 1, True, "\u0661"])
@pytest.mark.parametrize("malformed_first", [False, True])
def test_a11_3_malformed_witness_refuses_own_session(kind, field, bad, malformed_first):
    """The pure hydrator also rejects malformed percept witnesses before row filtering."""
    malformed = dict(_durability_witness(kind, 4), **{field: bad})
    good = [_ev(2, i, i - 3, _vec(i), 0, boot="0", flushed="1") for i in (5, 6, 7)]
    events = (
        [malformed, *good] if malformed_first else [*good, dict(malformed, seq="8", event_id="2-8")]
    )
    out = _hydrate(events, n_units=1, current_session=3)
    assert out["x"] == out["y"] == [] and out["categories"] == set()
    assert out["refusals"].get("session_witness_malformed") == 1
    assert out["refusals"].get("session_witness_conflict") == 3


@pytest.mark.parametrize("reason", ["boot", "flushed"])
def test_a11_3_session_conflict_counts_each_discarded_percept(reason, monkeypatch):
    """A11.3 (2): three discarded percepts count three, alongside the session counter."""
    events = [
        _durability_witness("teach", 4, boot="1", flushed="1"),
        _durability_witness(
            "turn",
            5,
            boot="0" if reason == "boot" else "1",
            flushed="0" if reason == "flushed" else "1",
        ),
    ]
    events += [_ev(2, i, i - 5, _vec(i), 0, boot="1", flushed="1") for i in (6, 7, 8)]
    restored = _read_witness_history(events, monkeypatch)
    assert restored._buf_x == restored._buf_y == [] and restored._categories == set()
    assert restored.transition_refusals == {
        f"session_{reason}_conflict": 1,
        "session_witness_conflict": 3,
    }


@pytest.mark.parametrize("kind", ["teach", "turn"])
@pytest.mark.parametrize(
    "missing", [("proto_boot",), ("proto_flushed",), ("proto_boot", "proto_flushed")]
)
def test_a11_3_absent_chat_counts_remain_legacy_without_proof(kind, missing, monkeypatch):
    """Absent fields prove nothing and do not invalidate an otherwise honest session."""
    crashed = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    legacy = _durability_witness(kind, 4, boot="1", flushed="1")
    legacy = {k: v for k, v in legacy.items() if k not in missing}
    later = dict(_durability_witness("teach", 9), session="3", event_id="3-9")
    restored = _read_witness_history(crashed + [legacy, later], monkeypatch)
    assert restored._buf_x == restored._buf_y == []
    assert restored.transition_refusals == {"stale_proto": 3}
    good = [_ev(2, i, i - 4, _vec(i), 0, boot="0", flushed="1") for i in (5, 6, 7)]
    honest = _read_witness_history([legacy, *good, later], monkeypatch)
    assert honest._buf_y == [0, 0] and not honest.transition_refusals


def test_a11_3_well_formed_forged_flush_rise_is_admitted(monkeypatch):
    """A11.3 (3): a well-formed forged rise is admitted without save authentication."""
    events = [_ev(1, i, i, _vec(i), 0, boot="0", flushed="0") for i in (1, 2, 3)]
    forged = dict(_durability_witness("teach", 4), session="1", event_id="1-4")
    next_session = _durability_witness("teach", 8, boot="0", flushed="0")
    reused = dict(_durability_witness("teach", 9), session="3", event_id="3-9")
    restored = _read_witness_history(events + [forged, next_session, reused], monkeypatch)
    assert list(zip(restored._buf_x, restored._buf_y, strict=True)) == _naive(events)
    assert restored._categories == {0} and not restored.transition_refusals


def test_failed_write_leaves_a_pseq_gap_and_no_bridging_pair():
    # attempts 1,2 landed, attempt 3 failed (consumed pseq 3, no event), attempts 4,5 landed
    events = [_ev(1, 1, 1, _vec(1), 0), _ev(1, 2, 2, _vec(2), 1)]
    events += [_ev(1, 3, 4, _vec(4), 2), _ev(1, 4, 5, _vec(5), 0)]
    out = _hydrate(events)
    assert [y for _, y in _pairs(out)] == [1, 0], "pairs 1->2 and 4->5 only; 2->4 is never formed"
    assert out["refusals"] == {}


def test_buffer_cap_keeps_the_last_pairs_and_every_eligible_category():
    events = _chain(1, 1, [0, 1, 2, 3, 4, 5, 6])
    out = _hydrate(events, buffer_cap=3)
    assert out["y"] == [4, 5, 6] and out["categories"] == set(range(7))


# ---- the grower: write-ahead, byte identity, refusal ---------------------------------------------
def _stream(n, seed=DEV_SEED):
    """A learnable stream: prototype per category, the next category follows the last one."""
    rng = random.Random(seed)
    protos = [[rng.random() for _ in range(WIDTH)] for _ in range(4)]
    cat, out = 0, []
    for _ in range(n):
        out.append(([v + 0.05 * rng.random() for v in protos[cat]], cat))
        cat = (cat + 1) % 4 if rng.random() < 0.9 else rng.randrange(4)
    return out


def _engine(**kw):
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    return NeurogenesisSelfImprove(
        refit_every=4, device="cpu", seed=DEV_SEED, online_readout=None, **kw
    )


class _Recorder:
    """A sink that records what the engine would write: `(session, pseq, features, category)`."""

    def __init__(self, session):
        self.session, self.pseq, self.rows = session, 0, []

    def __call__(self, feats, cat):
        self.pseq += 1
        self.rows.append((self.session, self.pseq, list(feats), int(cat)))
        return True

    def events(self, seq0=1):
        return [_ev(s, seq0 + i, p, f, c) for i, (s, p, f, c) in enumerate(self.rows)]


def _drive_pair(sessions=2, n=14):
    """(twin, restored): the twin observed `sessions` sessions through a recording sink (a restart
    between them: `_prev` cleared); `restored` is a fresh engine hydrated from the recorded events
    (restored is returned UNHYDRATED when `sessions` is 2 and callers hydrate themselves)."""
    twin, events = _engine(), []
    for s in range(1, sessions + 1):
        rec = _Recorder(s)
        twin.transition_sink = rec
        twin._prev = None  # a restart: the first percept of a session has no predecessor
        for feats, cat in _stream(n, seed=DEV_SEED + s):
            twin.observe(feats, cat)
        events += rec.events(seq0=len(events) + 1)
    return twin, _engine(), events


def test_hydrated_buffer_equals_the_twins_and_the_refit_after_it_equals_too():
    twin, restored, events = _drive_pair()
    assert restored.restore_transitions(events, n_units=9, current_session=3) == len(twin._buf_x)
    assert restored._buf_x == twin._buf_x and restored._buf_y == twin._buf_y
    assert restored._categories == twin._categories
    assert (restored._prev, restored._prev_cat, restored.observed) == (None, None, 0)
    assert restored.live_n == restored.live_correct == 0
    for _ in range(4):
        twin.tick()
        restored.tick()
    assert twin._model is not None
    for name in ("W", "b", "beta"):
        assert getattr(restored._model, name).equal(getattr(twin._model, name))
    assert restored.trajectory == twin.trajectory


def test_positive_control_a_buffer_shuffled_across_sessions_refits_differently():
    twin, restored, events = _drive_pair()
    restored.restore_transitions(events, n_units=9, current_session=3)
    half = len(restored._buf_y) // 2
    restored._buf_y = restored._buf_y[half:] + restored._buf_y[:half]
    for _ in range(4):
        twin.tick()
        restored.tick()
    assert not restored._model.beta.equal(twin._model.beta) or not restored._model.W.equal(
        twin._model.W
    ), "the buffer content must reach the observable, or equality above proves nothing"


def test_hydration_is_construction_only():
    _, restored, events = _drive_pair()
    restored.restore_transitions(events, n_units=9, current_session=3)
    with pytest.raises(RuntimeError, match="construction-only"):
        restored.restore_transitions(events, n_units=9, current_session=3)
    live = _engine()
    live.observe(*_stream(1)[0])
    with pytest.raises(RuntimeError, match="construction-only"):
        live.restore_transitions(events, n_units=9, current_session=3)


def test_no_sink_is_byte_identical_and_a_landing_sink_changes_nothing():
    def drive(setup):
        eng = _engine()
        setup(eng)
        for feats, cat in _stream(24):
            eng.observe(feats, cat)
        for _ in range(8):
            eng.tick()
        return eng.growth_state_bytes()

    base = drive(lambda e: None)
    assert base is not None
    assert drive(lambda e: setattr(e, "transition_sink", None)) == base
    rec = _Recorder(1)
    assert drive(lambda e: setattr(e, "transition_sink", rec)) == base
    assert len(rec.rows) == 24, "the landing sink saw every observation and still changed nothing"


def test_failed_write_refuses_the_observation_and_resets_prev():
    eng = _engine()
    feats, cat = _stream(3)[0]
    eng.observe(feats, cat)
    eng.observe(*_stream(3)[1])
    snap = (list(eng._buf_x), list(eng._buf_y), set(eng._categories), eng.observed)
    eng.transition_sink = lambda f, c: False
    with pytest.raises(RuntimeError, match="not buffered"):
        eng.observe(*_stream(3)[2])
    assert eng._prev is None
    assert (eng._buf_x, eng._buf_y, eng._categories, eng.observed) == snap
    eng.transition_sink = lambda f, c: 1 / 0  # a raising sink is a failed write, never a crash
    with pytest.raises(RuntimeError, match="not buffered"):
        eng.observe(*_stream(3)[2])
    assert eng._prev is None and (eng._buf_x, eng._buf_y) == (snap[0], snap[1])


# ---- the Brain: the common writer, the all-kind latch, the shared boot pass -------------------
@contextmanager
def _brain():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home() as private:
        home.require_isolated_home()
        brain = make_brain(seed=DEV_SEED, persist=True, device="cpu")
        try:
            yield brain, Path(private)
        finally:
            brain.close()


def _percept_events():
    from snn_data_lake.substrate import home

    return [e for e in home.load_episodes(all_kinds=True) if e.get("kind") == "percept"]


def test_brain_writes_one_write_ahead_event_per_observation():
    from snn_data_lake.growth.neurogenesis_engine import unpack_percept

    with _brain() as (brain, _):
        assert brain._r2_fault is None and brain.neuro.transition_sink is not None
        seen = []
        orig = brain.neuro.observe
        brain.neuro.observe = lambda f, c: (seen.append((list(f), int(c))), orig(f, c))[1]
        for _ in range(12):
            brain.tick()
        events = _percept_events()
        assert seen and len(events) == len(seen) == brain.neuro.observed
        assert [int(e["pseq"]) for e in events] == list(range(1, len(events) + 1))
        assert {e["stream"] for e in events} == {"wake"}
        assert {e["encoder"] for e in events} == {f"arm.percept gained width={WIDTH}"}
        assert all(type(v) is str for e in events for v in e.values())
        assert [(unpack_percept(e["x"]), int(e["category"])) for e in events] == seen


def test_brain_failed_percept_write_latches_every_kind_until_restart(monkeypatch):
    from snn_data_lake.substrate import home

    with _brain() as (brain, _):
        eng, ixn = brain.neuro, brain.brain._ixn
        feats = _stream(4)
        eng.observe(*feats[0])
        eng.observe(*feats[1])
        assert len(_percept_events()) == 2 and len(eng._buf_x) == 1
        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", lambda event: False)
            with pytest.raises(RuntimeError, match="not buffered"):
                eng.observe(*feats[2])
        assert eng._prev is None and len(eng._buf_x) == 1 and eng.observed == 2
        assert brain._r2_fault is not None, "owner #1037: an uncertain failed write latches"
        assert len(_percept_events()) == 2
        # the latch governs the NEXT kind exactly as it governs percept
        refused = ixn._unjournaled_refused
        brain.brain.teach("hello private r2", "GREET")
        assert ixn._unjournaled_refused == refused + 1
        assert home.load_episodes(kinds=("teach",)) == []
        # and the next percept is refused too, consuming its pseq (the gap hydration sees)
        with pytest.raises(RuntimeError, match="not buffered"):
            eng.observe(*feats[3])
        assert brain._r2_pseq == 4


@pytest.mark.parametrize("boot_failure", ["reconcile", "hydrate"])
def test_failed_boot_refuses_observations_instead_of_buffering_unjournaled(
    boot_failure, monkeypatch
):
    """Codex #2844 / owner #1037: a persisting brain whose R2 is closed at boot (reconciliation
    refused, or hydration raised) must REFUSE each observation -- no buffer, `_prev` None, the
    refusal counted, no event -- never fall back to an unjournaled pair."""
    from snn_data_lake.engine.brain import Brain
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    if boot_failure == "reconcile":
        boot = Brain._r2_boot

        def _closed(self):
            boot(self)
            self._r2_fault = {"phase": "boot", "cause": "planted reconciliation refusal"}

        monkeypatch.setattr(Brain, "_r2_boot", _closed)
    else:

        def _raise(self, *a, **k):
            raise RuntimeError("planted hydration failure")

        monkeypatch.setattr(NeurogenesisSelfImprove, "restore_transitions", _raise)
    with _brain() as (brain, _):
        eng = brain.neuro
        feats = _stream(4)
        for f, c in feats[:3]:
            with pytest.raises(RuntimeError, match="not buffered"):
                eng.observe(f, c)
        assert eng._buf_x == [] and eng._buf_y == [] and eng._prev is None
        assert eng.observed == 0 and eng.transition_write_refusals == 3
        assert _percept_events() == []


def test_non_persisting_brain_buffers_with_no_sink():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home():
        home.require_isolated_home()
        brain = make_brain(seed=DEV_SEED, persist=False, device="cpu")
        try:
            eng = brain.neuro
            assert eng.transition_sink is None
            feats = _stream(3)
            for f, c in feats:
                eng.observe(f, c)
            assert len(eng._buf_x) == 2 and eng._prev is not None
            assert eng.transition_write_refusals == 0
        finally:
            brain.close()


def test_a_non_registered_width_is_refused_without_latching_every_kind():
    """rv-p3-a nit (c): KIND_SPECS pins `width=64`; a percept of another learner width must be a
    counted refusal of that observation, not a ValueError that latches EVERY kind."""
    with _brain() as (brain, _):
        eng, ixn = brain.neuro, brain.brain._ixn
        brain.brain.arm.learner.dim = 32
        with pytest.raises(RuntimeError, match="not buffered"):
            eng.observe([0.1] * 32, 0)
        assert brain._r2_fault is None, brain._r2_fault
        assert eng.transition_write_refusals == 1 and _percept_events() == []
        refused = ixn._unjournaled_refused
        brain.brain.teach("hello private r2", "GREET")
        assert ixn._unjournaled_refused == refused, "the other kinds still land"


def test_proto_flushed_is_the_count_at_save_not_after_a_recruit_during_it(monkeypatch):
    """rv-p3-a nit (b): `_flush_prototypes` must record the count the SAVED state holds."""
    from snn_data_lake.substrate import home

    with _brain() as (brain, _):
        learner = brain.brain.arm.learner
        v = _vec(7)
        learner.observe(v)
        held = int(learner.n_units)
        assert held >= 1
        real = home.save_state

        def _save_then_recruit(*a, **k):
            out = real(*a, **k)
            learner.observe([-x for x in _vec(9)])
            learner.observe(_vec(11, 64))
            return out

        monkeypatch.setattr(home, "save_state", _save_then_recruit)
        saved = brain._flush_prototypes()
        assert saved and int(learner.n_units) > held, "the recruit during the save must happen"
        assert brain._r2_proto_flushed == held


@pytest.mark.parametrize("missing", ["schema", "hydrate_branch"])
def test_missing_shipped_percept_integration_refuses_boot(missing, monkeypatch):
    """Panel A3/B2: deliberately remove integration; no fixture may supply it again."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.readout import interaction_learning as il
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate import learner_events as le

    with home.throwaway_home():
        home.require_isolated_home()
        brain = make_brain(seed=DEV_SEED, persist=True, device="cpu")
        try:
            assert brain._r2_fault is None
            for _ in range(12):
                brain.tick()
            assert _percept_events(), "positive control: shipped production writes percepts"
        finally:
            brain.close()
        if missing == "schema":
            monkeypatch.delitem(le.KIND_SPECS, "percept")
            cause = "R2 incomplete or mismatched same-pass receipt"
        else:
            tree = ast.parse(
                textwrap.dedent(inspect.getsource(il.SpikingInteractionLearner.hydrate))
            )
            branches = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.If) and ast.unparse(n.test) == "kind == 'percept'"
            ]
            assert len(branches) == 1, "positive control: mutate the actual shipped percept branch"
            branches[0].test = ast.Constant(False)
            ns = {}
            exec(compile(ast.fix_missing_locations(tree), il.__file__, "exec"), il.__dict__, ns)
            monkeypatch.setattr(il.SpikingInteractionLearner, "hydrate", ns["hydrate"])
            cause = "R2 unbound effect: percept"
        restored = make_brain(seed=DEV_SEED, persist=True, device="cpu")
        try:
            assert restored._r2_fault is not None
            assert cause in restored._r2_fault["cause"], restored._r2_fault
            assert restored.neuro._buf_x == []
            with pytest.raises(RuntimeError, match="not buffered"):
                restored.neuro.observe(_vec(1), 0)
        finally:
            restored.close()


def test_planted_events_in_a_persisting_lake_are_refused_by_hydration():
    """WIDTH, VOCAB and INSTRUMENT planted between real percepts of an earlier session: the next
    boot reconciles (the kind is bound), hydrates only the eligible chain and counts the rest."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home():
        home.require_isolated_home()
        events = _chain(1, 1, [0, 0, 0])  # n_units is 0 on a fresh arm: VOCAB refuses these
        events += [_ev(1, 4, 4, _vec(8, 63), 0, encoder="arm.percept gained width=64")]
        events += [_ev(1, 5, 5, _vec(9), 0, provenance="instrument")]
        for e in events:
            assert home.save_episode(e)
        brain = make_brain(seed=DEV_SEED, persist=True, device="cpu")
        try:
            assert brain._r2_fault is None, brain._r2_fault
            assert brain.neuro._buf_x == [] and brain.neuro.transition_refusals == {
                "vocab": 3,
                "width": 1,
            }, brain.neuro.transition_refusals
            assert brain._session == "2"
        finally:
            brain.close()


# ---- the restart: real children over one private home (the kill itself is replay_hydration's) ----
_CHILD = r"""
import json, sys
from snn_data_lake.engine.brain import make_brain

mode = sys.argv[1]
b = make_brain(seed=15000, persist=True, device="cpu")
neuro = b.neuro
assert b._r2_fault is None, b._r2_fault
log = []
_observe = neuro.observe


def spy(f, c):
    r = _observe(f, c)
    log.append(int(c))
    return r


neuro.observe = spy


def emit(tag, d):
    print("P3 " + tag + " " + json.dumps(d, sort_keys=True, allow_nan=False), flush=True)


def snap():
    return {
        "buf_x": neuro._buf_x, "buf_y": neuro._buf_y, "cats": sorted(neuro._categories),
        "refusals": neuro.transition_refusals, "n_units": b.brain.arm.learner.n_units,
        "observed": neuro.observed, "prev_none": neuro._prev is None,
    }


if mode == "close":
    for i in range(30):  # percepts, a teach between them (a learner effect), then a clean close
        b.tick()
        if i == 12:
            b.brain.teach("hello private r2", "GREET")
    emit("CLOSED", {**snap(), "log": log, "cursor": b.brain._ixn._cursor, "fault": b._r2_fault})
    b.close()
    sys.exit(0)

if mode == "teach_only":
    b.brain.teach("hello private r2", "GREET")
    emit("TEACH_ONLY", {**snap(), "session": b._session, "fault": b._r2_fault})
    b.close()
    sys.exit(0)

if mode == "restore":
    emit("RESTORED", {**snap(), "cursor": b.brain._ixn._cursor, "fault": b._r2_fault})
    b.close()
    sys.exit(0)

sys.exit(2)
"""


def _run_child(mode, home_dir):
    env = {**os.environ, "NORD_HOME": str(home_dir)}
    env.pop("NORD_PERSIST", None)
    for _attempt in range(3):
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "-c", _CHILD, mode, str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        sys.stderr.write(result.stderr)
        # the live-home guard stats the running daemon's directory: its own race, read-only, and
        # it fires at boot before this child has written anything
        if "vanished mid-scan" not in result.stderr or "P3 " in result.stdout:
            break
    assert result.returncode == 0, (
        mode,
        result.returncode,
        result.stdout[-400:],
        result.stderr[-800:],
    )
    out = {}
    for line in result.stdout.splitlines():
        if line.startswith("P3 "):
            _, tag, payload = line.split(" ", 2)
            out[tag] = json.loads(payload)
    return out


def test_restart_hydrated_buffer_and_next_refit_equal_the_uninterrupted_twin(tmp_path):
    """K1g on dev seed 15000, CPU, through the instrument's own roles (R2-c7's one reading): the
    restore after a kill inside the night's `_flush_growth` call holds the twin's buffer and its
    model after the next refit, with no filter and no tolerance. A planted wrong restored model or
    buffer must make the bar fail (a test of the test)."""
    from snn_data_lake.instruments import replay_hydration as rh

    arms = rh.run_seed(DEV_SEED, tmp_path, roles=["res_K1g", "twin_K1g"])
    for role in ("twin_K1g", "res_K1g"):  # a failed arm shows its stderr, never a bare KeyError
        assert "error" not in arms[role], (role, arms[role].get("error"), arms[role].get("stderr"))
    twin, res = arms["twin_K1g"], arms["res_K1g"]
    assert twin["before"]["buf_x"], "the twin must hold transitions past its first refit"
    assert twin["model"]["refit_row"], "the bar must compare a refit"
    assert res["fault"] is None and not res["before"]["refusals"]
    assert res["before"]["prev_none"] and res["before"]["observed"] == 0, "chronology is hydrated"
    assert rh.percept_equal(res, twin)
    for part, wrong in (("W", [[1.0]]), ("beta", [[0.0]]), ("units", -1), ("traj", [])):
        planted = {**res, "model": {**res["model"], part: wrong}}
        assert not rh.percept_equal(planted, twin), part
    assert not rh.percept_equal({**res, "before": {**res["before"], "buf_y": []}}, twin)


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_clean_close_checkpoint_and_restart_keeps_percepts_teach_and_the_chain_reconciled():
    """A checkpoint whose cursor and canonical-prefix digest cover interleaved percept and teach
    events: the next boot reconciles (no fault), hydrates the same pairs and does not touch the
    chronology. The percept's cursor advance is what keeps the final-cursor check true."""
    from snn_data_lake.substrate import home

    with home.throwaway_home() as private:
        home.require_isolated_home()
        first = _run_child("close", private)["CLOSED"]
        assert first["fault"] is None and first["buf_x"] and first["cursor"] is not None
        got = _run_child("restore", private)["RESTORED"]
        assert got["fault"] is None, got["fault"]
        assert got["buf_x"] == first["buf_x"] and got["buf_y"] == first["buf_y"]
        assert got["cats"] == first["cats"] and not got["refusals"]
        assert got["prev_none"] and got["observed"] == 0
        assert got["cursor"] == first["cursor"]


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_three_process_restarts_preserve_pairs_after_a_teach_only_session():
    """Panel B1: a close flush with no subsequent percept must retain its proof in the lake."""
    from snn_data_lake.substrate import home

    with home.throwaway_home() as private:
        home.require_isolated_home()
        first = _run_child("close", private)["CLOSED"]
        second = _run_child("teach_only", private)["TEACH_ONLY"]
        third = _run_child("restore", private)["RESTORED"]
        assert second["session"] == "2" and second["observed"] == 0
        assert first["buf_x"] and first["fault"] is second["fault"] is third["fault"] is None
        for got in (second, third):
            assert not got["refusals"], got["refusals"]
            assert got["buf_x"] == first["buf_x"] and got["buf_y"] == first["buf_y"]
            assert got["cats"] == first["cats"]
