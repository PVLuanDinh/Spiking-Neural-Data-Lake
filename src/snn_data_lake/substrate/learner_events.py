"""Typed R2 reader candidate, UNRUN: strict same-pass receipt, no invented producers.

Ctxread remains schema-only. Percept (R2 P3) has no learner effect; hydrate moves the cursor.
Sleep none sentinels do not establish deciding-mask recovery.
"""

import hashlib
import json
import os
import re
from collections import Counter

from snn_data_lake.substrate import home as _home

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "read_learner_events": (
        "NOT_A_RULE",
        "reads typed learner events from the lake and filters them for eligibility; no weight moves",
    ),
    "scan_learner_events": (
        "NOT_A_RULE",
        "the same read plus the boot counter; returns records, computes no learning update",
    ),
}

SCHEMA = "r2.v1"
EMPTY_PREFIX_DIGEST = hashlib.sha256(b"").hexdigest()
#: G0(c)'s authoritative-teaching events (engine/brain.py `_record_teaching`, schema "g0c.v1") already
#: carry every r2.v1 slot except `split`; the upcaster maps them and nothing else.
LEGACY_SCHEMA = "g0c.v1"

_IDENTITY = ("kind", "event_id", "stream", "session", "seq", "schema", "valid", "split")

#: Per learner kind: the slots an event must carry (never padded) and the encoders it may name. A kind
#: absent here is not yet written by any writer; asking for it is a caller error, not a refusal.
KIND_SPECS = {
    "posed_problem": {
        "slots": _IDENTITY + ("source", "target", "provenance", "encoder"),
        "encoders": frozenset({"g5.pool_features n_in=256"}),
    },
    "teach": {
        "slots": _IDENTITY + ("source", "target", "provenance", "encoder"),
        "encoders": frozenset({"text_features n_in=52"}),
    },
    "turn": {
        "slots": _IDENTITY + ("source", "target", "provenance", "encoder"),
        "encoders": frozenset({"text_features n_in=52"}),
    },
    # G3's context-read journal (g3-growth-loop.md state-branch amendment): one row per stepped
    # turn of a live memory unit, written before the step. episode_id / turn_index / grown_version
    # are ASCII decimals, the digest a lowercase sha256 hex token.
    "ctxread": {
        "slots": _IDENTITY
        + ("episode_id", "turn_index", "encoded_ranks_digest", "grown_version", "encoder"),
        "encoders": frozenset({"text_features n_in=52"}),
    },
    # P2 (2.8 + the 2026-10-06 amendment): one row at the end of every learner-affecting sleep tick,
    # carrying digests of the state it left, so a restart can re-execute the tick and verify it.
    # `pending` and `applied_mask_digest` are sentinels ("none") until G6's producer exists.
    "sleep": {
        "slots": _IDENTITY
        + (
            "sleep_id",
            "night",
            "tick",
            "stage_index",
            "stage_name",
            "presentations_spent",
            "drawn",
            "rng_digest",
            "weights_digest",
            "rewirer_digest",
            "pending",
            "applied_mask_digest",
            "encoder",
        ),
        "encoders": frozenset({"text_features n_in=52"}),
    },
    # R2 part P3 (2.10): the grower's percept-transition record, written BEFORE the observation is
    # buffered. `x` is base64 of little-endian float64, `pseq` the per-session attempt counter,
    # `proto_boot` / `proto_flushed` the arm learner's durable prototype count (2.11 D2).
    "percept": {
        "slots": _IDENTITY
        + (
            "x",
            "category",
            "n_protos",
            "proto_boot",
            "proto_flushed",
            "pseq",
            "provenance",
            "encoder",
        ),
        "encoders": frozenset({"arm.percept gained width=64"}),
    },
    # POST-DATA 2026-10-13, owner2105/2106: one durable aggregate bout, never REM items.
    "sleep_bout": {
        "slots": _IDENTITY
        + (
            "bout",
            "rng_digest",
            "weights_digest",
            "rewirer_digest",
            "drawn",
            "presentations_spent",
            "encoder",
        ),
        "encoders": frozenset({"text_features n_in=52"}),
    },
}

_CTXREAD_DECIMALS = ("episode_id", "turn_index", "grown_version")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


#: the sentinel for a slot that has no value yet (the reader treats an empty slot as `missing_slot`)
NONE_SLOT = "none"


def advance_prefix_digest(digest, event):
    """Round 5 (#2112): bind ordered content using the existing canonical episode codec."""
    from snn_data_lake.substrate.binlog import pack_episode

    return hashlib.sha256(bytes.fromhex(digest) + pack_episode(event)).hexdigest()


def prefix_digest(events, cursor):
    """Digest exactly the validated eligible prefix covered by a checkpoint cursor."""
    digest = EMPTY_PREFIX_DIGEST
    if cursor is None:
        return digest
    for event in events:
        digest = advance_prefix_digest(digest, event)
        if _order(event) == cursor:
            return digest
    raise ValueError("R2 checkpoint cursor is not in the eligible prefix")


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


#: bookkeeping a restart legitimately changes (re-deposits it counted, turns it refused): not state
_COUNTERS = ("_redo_count", "_unjournaled_refused")


def _state_without_counters(learner):
    tree = learner.state_dict()["tree"]
    return {**tree, "attrs": {k: v for k, v in tree["attrs"].items() if k not in _COUNTERS}}


#: The digests a `sleep` event carries and a redone tick must reproduce -- exactly these keys.
SLEEP_DIGESTS = ("rng_digest", "weights_digest", "rewirer_digest")
#: The progress slots a redone tick must also reproduce (panel m3: journaled, so compared).
SLEEP_PROGRESS = ("drawn", "presentations_spent")


def sleep_digests(learner, rewirer):
    """The three digests a `sleep` event carries (and a redone tick must reproduce): the learner's
    RNG state, its whole encoded state tree (weights, stack, replay counts, cursor) and the
    structural rewirer's state (`none` before the first pass). Floats serialize exactly."""
    return {
        "rng_digest": _sha(learner.rng.getstate()),
        "weights_digest": _sha(_state_without_counters(learner)),
        "rewirer_digest": NONE_SLOT if rewirer is None else _sha(rewirer.state_dict()),
    }


#: Refusal reasons, in the order the reader tests them (one reason per refused event).
REASONS = (
    "missing_slot",
    "unknown_schema",
    "unknown_encoder",
    "bad_identity",
    "invalid",
    "heldout",
    "instrument",
    "duplicate_id",
    "seq_regression",
)


def _production_lake():
    return os.path.normcase(
        os.path.abspath(os.path.join(os.path.expanduser("~"), ".snn_data_lake", "brain_lake.bin"))
    )


def _resolved(path):
    return os.path.normcase(os.path.abspath(path or _home.knowledge_path()))


# distinct-from: gates/safety_envelope.py, gates/eprop_transport_gate.py, brainstem/so101_arm.py, __init__.py, engine/live_brain.py -- is_envelope normalizes REPO-RELATIVE spellings lexically; the others are a declaration lookup, a pose test, an alias layer, a wiring test; this compares a lake FILE's physical identity on disk (all five read 2026-10-04)
def _is_production(path):
    """True when `path` names the production lake by its PHYSICAL identity, so a junction or a
    symbolic-link alias cannot bypass the refusal (cp3 #12): the same file when both exist, else the
    same canonical (link-resolved) path."""
    a, b = _resolved(path), _production_lake()
    if os.path.exists(a) and os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _upcast(event):
    """A `g0c.v1` teach event as `r2.v1`: the same slots, plus `split="train"`. Anything else as is."""
    if event.get("schema") == LEGACY_SCHEMA and event.get("kind") == "teach":
        out = dict(event)
        out["schema"] = SCHEMA
        out.setdefault("split", "train")
        return out
    return event


def _order(event):
    return (int(event["session"]), int(event["seq"]))


def _refusal(event, spec, allow_instrument):
    """The reason this learner event is not eligible, or None. Identity checks come first."""
    if any(slot not in event or event[slot] in (None, "") for slot in spec["slots"]):
        return "missing_slot"
    if event["schema"] != SCHEMA:
        return "unknown_schema"
    if event["encoder"] not in spec["encoders"]:
        return "unknown_encoder"
    session, seq = str(event["session"]), str(event["seq"])
    ok = _home.is_decimal_id(session) and _home.is_decimal_id(seq)
    if not ok or event["event_id"] != f"{session}-{seq}":
        return "bad_identity"
    if event["kind"] == "ctxread" and not (
        all(_home.is_decimal_id(event[s]) for s in _CTXREAD_DECIMALS)
        and _SHA256_HEX.fullmatch(str(event["encoded_ranks_digest"]))
    ):
        return "bad_identity"
    if event["kind"] == "posed_problem":
        if (
            set(event) != set(spec["slots"])
            or event["stream"] != "posed"
            or event["provenance"] != "posed"
            or event["split"] != "train"
        ):
            return "bad_posed_contract"
        try:
            from snn_data_lake.readout.posed_problem import posed_source

            value = json.loads(event["source"])
            if posed_source(value) != event["source"] or value["problem_id"] != event["target"]:
                return "bad_posed_source"
        except (ValueError, TypeError, KeyError):
            return "bad_posed_source"
    if event["valid"] != "1":
        return "invalid"
    if event["split"] == "heldout":
        return "heldout"
    if event.get("provenance") == "instrument" and not allow_instrument:
        return "instrument"
    return None


# distinct-from: faculties/theta_gamma_code.py, instruments/carried_state_controls.py, sleep/nightly_skills.py, engine/brain.py, faculties/global_workspace.py, sleep/lake_consolidation.py, sleep/episode_consolidation.py -- word overlap on "key" only (a repr char-sum, a replay diff, an RNG-state tuple, an event writer, a workspace contest, a recall-count content crc, an assembly content crc); none refuses a repeated learner event (all read 2026-10-07)
def _sleep_key(event):
    # A sleep event's redo key (r2 prereg 2.8 amendment): one re-execution per (sleep_id, tick).
    return ("sleep", event["sleep_id"], event["tick"]) if event["kind"] == "sleep" else None


# distinct-from: instruments/pool_selection.py, instruments/prior_drift_harness.py -- phrase overlap only (a stream contrast, a refusal slot); this checks a lake event's identity against the pass (both read 2026-10-04)
def _identity_refusal(event, seen, last):
    """Identity against what the pass has already accepted: a repeated id, or for a `sleep` event a
    repeated redo key `(sleep_id, tick)` (the first wins -- idempotence on identity, not on bytes),
    or a `(session, seq)` at or before the last accepted one (the lake order is authoritative)."""
    if event["event_id"] in seen or _sleep_key(event) in seen:
        return "duplicate_id"
    if last is not None and _order(event) <= last:
        return "seq_regression"
    return None


def scan_learner_events(
    kinds,
    *,
    path=None,
    include_sessions=None,
    exclude_sessions=None,
    allow_instrument=False,
    strict=False,
    return_receipt=False,
):
    """Compatibility reader, or one immutable complete eligible-prefix receipt."""
    kinds = tuple(kinds)
    if any(k not in KIND_SPECS for k in kinds):
        raise ValueError("no learner-event schema for requested kind")
    if return_receipt and not strict:
        raise ValueError("R2 complete receipt requires strict mode")
    if strict and (
        set(kinds) != set(KIND_SPECS)
        or include_sessions is not None
        or exclude_sessions is not None
        or allow_instrument
    ):
        raise ValueError("R2 complete recovery cannot omit kinds/sessions or admit instruments")
    if allow_instrument and _is_production(path):
        raise ValueError("allow_instrument=True refused on production lake")
    if strict:
        episodes, raw_receipt = _home.load_episodes(
            path=path, all_kinds=True, strict=True, return_receipt=True
        )
        proof = json.loads(raw_receipt)
        decoded_rows = [r for r in proof["rows"] if r["reason"] == "decoded"]
        if len(decoded_rows) != len(episodes):
            raise ValueError("R2 episode receipt correspondence mismatch")
    else:
        episodes = _home.load_episodes(path=path, all_kinds=True)
        decoded_rows, proof = [None] * len(episodes), None
    nxt = _home.next_session(episodes)
    refusals = Counter()
    seen, events, last, classified, critical = set(), [], None, [], []
    for raw, row in zip(episodes, decoded_rows, strict=True):
        kind = raw.get("kind")
        required = kind in _home.LEARNER_EVENT_KINDS
        if kind not in kinds:
            if strict and (required or raw.get("schema") in (SCHEMA, LEGACY_SCHEMA)):
                critical.append(row["ordinal"])
                classified.append({"ordinal": row["ordinal"], "reason": "unbound_kind"})
            continue
        event = _upcast(raw)
        reason = _refusal(event, KIND_SPECS[kind], allow_instrument)
        if reason is None and event["split"] != "train":
            reason = "unknown_split"
        if reason is None:
            reason = _identity_refusal(event, seen, last)
        if reason is not None:
            refusals[reason] += 1
            if strict:
                classified.append({"ordinal": row["ordinal"], "reason": reason})
                if reason not in ("invalid", "heldout", "instrument"):
                    critical.append(row["ordinal"])
            continue
        seen.add(event["event_id"])
        if kind == "sleep":
            seen.add(_sleep_key(event))
        last = _order(event)
        events.append(event)
        if strict:
            classified.append(
                {"ordinal": row["ordinal"], "reason": "eligible", "event_id": event["event_id"]}
            )
    if strict:
        proof.update(
            {
                "eligible_ids": [e["event_id"] for e in events],
                "final_cursor": last,
                "next_session": nxt,
                "classified_rows": classified,
                "critical_event_ordinals": critical,
                "complete": proof["complete"] and not critical,
            }
        )
        receipt = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if return_receipt:
            return events, dict(refusals), nxt, receipt
        if not proof["complete"]:
            raise ValueError("R2 critical eligible prefix refused")
    if include_sessions is not None:
        events = [e for e in events if e["session"] in {str(s) for s in include_sessions}]
    if exclude_sessions is not None:
        events = [e for e in events if e["session"] not in {str(s) for s in exclude_sessions}]
    return events, dict(refusals), nxt


def read_learner_events(
    kinds,
    *,
    path=None,
    include_sessions=None,
    exclude_sessions=None,
    allow_instrument=False,
):
    """(events, refusals): the ELIGIBLE learner events of `kinds`, oldest first, and a tally of every
    refused one by reason (`REASONS`). One streaming pass in lake order. `path=None` is the brain's
    own home lake; `allow_instrument=True` admits provenance-`instrument` events and raises
    `ValueError` on the production home. Session filters apply after eligibility, before any caller
    selects (R2 2.1, Codex 26: the complete shared signature)."""
    events, refusals, _ = scan_learner_events(
        kinds,
        path=path,
        include_sessions=include_sessions,
        exclude_sessions=exclude_sessions,
        allow_instrument=allow_instrument,
    )
    return events, refusals


def _selfcheck():
    """Every refusal reason fires on a planted event, a clean event is returned once, and an
    autobiographical episode is never returned -- on a throwaway home."""
    with _home.throwaway_home():

        def ev(seq, **kw):
            e = {
                "kind": "turn",
                "event_id": f"1-{seq}",
                "stream": "chat",
                "session": "1",
                "seq": str(seq),
                "source": f"line {seq}",
                "target": "GREET",
                "provenance": "self",
                "encoder": "text_features n_in=52",
                "schema": SCHEMA,
                "valid": "1",
                "split": "train",
            }
            e.update(kw)
            return e

        _home.save_episode({"kind": "visit", "who": "x", "session": "1"})
        _home.save_episode(ev(1))
        bad = ev(2)
        del bad["target"]
        _home.save_episode(bad)
        _home.save_episode(ev(3, schema="r9"))
        _home.save_episode(ev(4, encoder="other"))
        _home.save_episode(ev(5, event_id="1-99"))
        _home.save_episode(ev(6, valid="0"))
        _home.save_episode(ev(7, split="heldout"))
        _home.save_episode(ev(8, provenance="instrument"))
        _home.save_episode(ev(1, source="byte-distinct repeat"))
        _home.save_episode(ev(9))
        _home.save_episode(ev(8, source="regressed"))  # 1-8 after 1-9: a seq regression
        events, refusals, nxt = scan_learner_events(("turn", "teach"))
    ids = [e["event_id"] for e in events]
    assert ids == ["1-1", "1-9"], ids
    assert set(refusals) == set(REASONS), refusals
    assert all(n == 1 for n in refusals.values()), refusals
    assert nxt == 2, nxt
    print(
        f"learner_events selfcheck: OK -- eligible {ids}, every one of {len(REASONS)} reasons fired once"
    )


if __name__ == "__main__":
    _selfcheck()
