"""R1 / T102 -- the A2 current-schema C8 fixture (r1-bounded-recall.md A2).

Stdlib only; builds plain dicts and touches no lake. The workload generator
(`instruments.recall_audit.make_workload`) takes these lists as its `teach` / `learner`
arguments, so the learner-event schema is stated here once and the generator restates none of it.

Intact arm: ten legacy `g0c.v1` teach records (checked against `learner_events._upcast`) plus one
distractor per kind in `learner_events.KIND_SPECS`, each carrying every current schema / identity /
encoder field with `valid="0"`: classified invalid (no critical refusal), excluded from learning,
and excluded from recall by the ONE predicate (`home.is_autobiographical_episode`).

Refusal arm (a SEPARATE arm on cloned isolated homes): rows the admission check must refuse. It
shows incomplete recovery, a closed writer, no installed session and expected MISS for
excluded-kind recall through the real consumer; those consumer facts are owed to the post-campaign
run.
"""

import json

SESSION = "1"
FIRST_SEQ = 100
N_LEGACY_TEACH = 10

_TEXT = "text_features n_in=52"
_DIGEST = "0" * 64


def legacy_teach_records(n=N_LEGACY_TEACH):
    """`n` G0(c) authoritative-teaching events: schema `g0c.v1`, no `split` (upcast adds it)."""
    return [
        {
            "kind": "teach",
            "event_id": f"{SESSION}-{q}",
            "stream": "chat",
            "session": SESSION,
            "seq": str(q),
            "source": f"key{q}",
            "target": f"intent{q}",
            "provenance": "authoritative",
            "encoder": _TEXT,
            "schema": "g0c.v1",
            "valid": "1",
        }
        for q in range(1, n + 1)
    ]


def _identity(kind, seq, stream="chat", split="train"):
    return {
        "kind": kind,
        "event_id": f"{SESSION}-{seq}",
        "stream": stream,
        "session": SESSION,
        "seq": str(seq),
        "schema": "r2.v1",
        "valid": "0",
        "split": split,
    }


def _posed_source():
    # the canonical bytes `posed_problem.posed_source` returns for the smallest typed op ("reset")
    return json.dumps({"op": "reset", "problem_id": "p1"}, sort_keys=True, separators=(",", ":"))


def distractor_events(first_seq=FIRST_SEQ):
    """One complete, `valid="0"` event per supported learner kind, in `KIND_SPECS` order."""
    builders = [
        lambda s: {
            **_identity("posed_problem", s, stream="posed"),
            "source": _posed_source(),
            "target": "p1",
            "provenance": "posed",
            "encoder": "g5.pool_features n_in=256",
        },
        lambda s: {
            **_identity("teach", s),
            "source": "dkey",
            "target": "dintent",
            "provenance": "authoritative",
            "encoder": _TEXT,
        },
        lambda s: {
            **_identity("turn", s),
            "source": "hello",
            "target": "greet",
            "provenance": "authoritative",
            "encoder": _TEXT,
        },
        lambda s: {
            **_identity("ctxread", s),
            "episode_id": "1",
            "turn_index": "0",
            "encoded_ranks_digest": _DIGEST,
            "grown_version": "0",
            "encoder": _TEXT,
        },
        lambda s: {
            **_identity("sleep", s),
            "sleep_id": "1",
            "night": "1",
            "tick": "1",
            "stage_index": "0",
            "stage_name": "nrem",
            "presentations_spent": "0",
            "drawn": "0",
            "rng_digest": _DIGEST,
            "weights_digest": _DIGEST,
            "rewirer_digest": _DIGEST,
            "pending": "none",
            "applied_mask_digest": "none",
            "encoder": _TEXT,
        },
        lambda s: {
            **_identity("percept", s),
            "x": "AAAAAAAAAAA=",
            "category": "c0",
            "n_protos": "1",
            "proto_boot": "0",
            "proto_flushed": "0",
            "pseq": "1",
            "provenance": "arm",
            "encoder": "arm.percept gained width=64",
        },
        lambda s: {
            **_identity("sleep_bout", s),
            "bout": "1",
            "rng_digest": _DIGEST,
            "weights_digest": _DIGEST,
            "rewirer_digest": _DIGEST,
            "drawn": "0",
            "presentations_spent": "0",
            "encoder": _TEXT,
        },
    ]
    return [b(first_seq + i) for i, b in enumerate(builders)]


def refusal_arm_records(first_seq=FIRST_SEQ + 50):
    """(expected `learner_events._refusal` reason, record) pairs the admission check must refuse.

    The minimal form is the old make_workload's learner row (kind and ids only); the rest are
    otherwise-complete distractors broken in exactly one admission field.
    """
    good = distractor_events(first_seq)
    teach = next(e for e in good if e["kind"] == "teach")
    out = [
        (
            "missing_slot",
            {
                "kind": e["kind"],
                "event_id": e["event_id"],
                "session": e["session"],
                "seq": e["seq"],
            },
        )
        for e in good
    ]
    out.append(
        (
            "unknown_schema",
            {
                **teach,
                "schema": "r2.v0",
                "event_id": f"{SESSION}-{first_seq + 20}",
                "seq": str(first_seq + 20),
            },
        )
    )
    out.append(
        (
            "unknown_encoder",
            {
                **teach,
                "encoder": "text_features n_in=7",
                "event_id": f"{SESSION}-{first_seq + 21}",
                "seq": str(first_seq + 21),
            },
        )
    )
    out.append(("bad_identity", {**teach, "event_id": "1-999", "seq": str(first_seq + 22)}))
    return out
