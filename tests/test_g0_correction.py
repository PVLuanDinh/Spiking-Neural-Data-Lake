"""G0(a): authoritative labels, reachable reds, and the recognition-only safety boundary."""

import copy
from unittest.mock import Mock

import pytest

from snn_data_lake.cortex import interaction_learning as ixn
from snn_data_lake.infra.cockpit import _REPLY_KEEP, Turns, _handler


def learner():
    return ixn.SpikingInteractionLearner(["GREET", "GOAL"], 12, seed=0)


def test_supersession_and_pure_provenance():
    lrn = learner()
    assert lrn.provenance("a goal") == "self"
    lrn.learn_turn("a goal", "GREET")
    lrn.learn_turn("a goal", "GREET")
    lrn.learn_turn(" A  GOAL ", None, feedback="GOAL")
    spy = Mock(wraps=lrn.interact)
    lrn.interact = spy
    lrn.learn_turn("a goal", "GREET")
    assert spy.call_args.args[1] == "GOAL"
    before = (lrn.weights(), dict(lrn._authority), lrn.rng.getstate())
    assert lrn.provenance(" A GOAL ") == "authoritative"
    assert before == (lrn.weights(), lrn._authority, lrn.rng.getstate())


@pytest.mark.parametrize("rewrite", [True, False])
def test_ring_rewrite_and_neutered_supersession(rewrite):
    lrn = learner()
    lrn.provenance_rewrite = rewrite
    for _ in range(2):
        lrn.learn_turn("a goal", "GREET")
    lrn.learn_turn("other", "GREET")
    lrn.learn_turn("a goal", None, feedback="GOAL")
    labels = [label for _, label, key in lrn._replay if key == "a goal"]
    assert labels == (["GOAL"] * 3 if rewrite else ["GREET", "GREET", "GOAL"])
    assert lrn._replay[2][1:] == ("GREET", "other")
    lrn.provenance_supersede = False
    lrn.learn_turn("a goal", "GREET")
    assert lrn._replay[-1][1] == "GREET"


def test_replay_preserves_key():
    lrn = learner()
    lrn.learn_turn(" A  GOAL ", "GREET")
    before = copy.deepcopy(lrn._replay)
    lrn.interact = Mock(wraps=lrn.interact)
    assert lrn.replay(k=3) == 3
    assert all(
        call.kwargs == {"_record": False, "_key": "a goal"} for call in lrn.interact.call_args_list
    )
    assert lrn._replay == before


@pytest.mark.parametrize("feedback", ["UNKNOWN", 1])
def test_unknown_feedback_rejected_without_mutation(feedback):
    lrn = learner()
    with pytest.raises(ValueError):
        lrn.learn_turn("a goal", "GREET", feedback=feedback)
    assert not lrn._authority and not lrn._replay


def test_missing_label_rejected():
    with pytest.raises(ValueError):
        learner().learn_turn("a goal", None)


def test_authority_survives_many_later_corrections():
    lrn = learner()
    lrn.learn_turn("kept", None, feedback="GOAL")
    for index in range(5000):
        lrn.learn_turn(f"other {index}", None, feedback="GOAL")
    assert lrn.provenance("kept") == "authoritative"
    assert lrn._authority["kept"] == "GOAL"


def test_authoritative_learn_turn_keys_the_ring_entry():
    lrn = learner()
    lrn.learn_turn(" A  Goal ", None, feedback="GOAL")
    assert lrn._replay[-1][1:] == ("GOAL", "a goal")


def constructed_results(a=1.0, c=0.0):
    """Distinct from g0_run: constructed scorer inputs, never measured brain results."""
    u, intent = ixn.G0_CORRECTED[0]
    return [
        {
            "seed": seed,
            "arm": arm,
            "head": "constructed",
            "dirty": False,
            "valid": True,
            "invalid_reasons": [],
            "M1": a if arm == "A" else c,
            "M2": 0.5,
            "M3": 0.75,
            "routes": {},
            "ring_audit": {u: [intent]},
            "teaching_events": (
                [
                    {
                        "utterance": u,
                        "label": "GREET",
                        "after_correction": True,
                        "phase": "wake" if arm == "B1" else "replay",
                    }
                ]
                if arm in ("B1", "B2")
                else []
            ),
        }
        for seed in ixn.G0_SEEDS
        for arm in ixn.G0_ARMS
    ]


def test_verdict_incumbent_against_itself_is_red():
    runs = constructed_results(a=0.5, c=0.5)
    before = copy.deepcopy(runs)
    result = ixn.g0_verdict(runs)
    assert result["R-a1"]["verdict"] == "RED"
    assert result["R-a1"]["sign_flip_p"] == 1.0
    assert runs == before


def test_verdict_positive_degenerate_and_reachability():
    result = ixn.g0_verdict(constructed_results())
    assert result["R-a1"]["verdict"] == "PASS"
    assert result["R-a1"]["t"] is None
    assert "undefined" in result["R-a1"]["degenerate_t"]
    assert result["R-a1"]["sign_flip_p"] == 1 / 256
    assert 0 <= result["R-a1"]["conditional_rejection_fraction"] <= 1
    assert result["reachability"]["B1"]["reached"]
    assert result["reachability"]["B2"]["reached"]
    assert result["verdict"] == "PASS"


@pytest.mark.parametrize("fault", ["event", "ring"])
def test_verdict_authority_violation_is_red(fault):
    runs = constructed_results()
    u, _ = ixn.G0_CORRECTED[0]
    if fault == "event":
        runs[0]["teaching_events"].append(
            {
                "utterance": u,
                "label": "GREET",
                "phase": "replay",
                "after_correction": True,
            }
        )
    else:
        runs[0]["ring_audit"][u].append("GREET")
    result = ixn.g0_verdict(runs)
    assert result["R-a2"]["verdict"] == "RED"


def test_verdict_too_many_invalid_seeds_is_red():
    runs = constructed_results()
    for run in runs:
        if run["arm"] == "B0" and run["seed"] < 3:
            run["dirty"] = True
    result = ixn.g0_verdict(runs)
    assert result["R-a1"]["verdict"] == "RED"
    assert len(result["invalid_runs"]) == 3
    assert not result["enough_valid_seeds"]


def test_verdict_seven_valid_seeds_in_one_arm_is_red_with_reason():
    runs = constructed_results()
    run = next(run for run in runs if run["arm"] == "B0" and run["seed"] == ixn.G0_SEEDS[0])
    run["valid"] = False
    run["invalid_reasons"] = ["constructed failure"]
    result = ixn.g0_verdict(runs)
    assert result["verdict"] == "RED"
    assert not result["enough_valid_seeds"]
    assert any("B0 has 7 of 8" in reason for reason in result["validity_reasons"])


def test_verdict_mixed_heads_is_red_with_heads_listed():
    runs = constructed_results()
    runs[0]["head"] = "other"
    result = ixn.g0_verdict(runs)
    assert result["verdict"] == "RED"
    assert not result["common_head"]
    assert result["heads"] == ["constructed", "other"]


def test_g0_spy_records_failure_after_original_raises():
    reasons = []
    events = []
    replay_count = 0

    def raises(*args, **kwargs):
        raise LookupError("constructed")

    def count_replay():
        nonlocal replay_count
        replay_count += 1

    with pytest.raises(LookupError, match="constructed"):
        ixn._g0_observe_interaction(
            raises,
            object(),
            "GOAL",
            _record=False,
            _key="a goal",
            by_key={"a goal": "a goal"},
            corrected={"a goal"},
            events=events,
            reasons=reasons,
            count_replay=count_replay,
        )
    assert reasons == ["replay interaction raised LookupError"]
    assert replay_count == 0
    assert events == []


def test_verdict_missing_reachability_is_red():
    runs = constructed_results()
    for run in runs:
        run["teaching_events"] = []
    result = ixn.g0_verdict(runs)
    assert not result["reachability"]["B1"]["reached"]
    assert not result["reachability"]["B2"]["reached"]
    assert result["verdict"] == "RED"


def test_turns_correction_never_dispatches_command():
    brain = Mock()
    for name in ("respond", "chat", "converse"):
        getattr(brain, name).side_effect = AssertionError("command path reached")
    brain.correct.return_value = {"was": "GREET", "provenance": "authoritative"}
    turns = Turns()
    tid = turns.submit("stop")
    assert turns._q.get_nowait() == ("say", tid, "stop")  # never pump the command
    cid, why = turns.submit_correction(tid, "SAFETY")
    assert why is None and cid != tid
    assert turns.pump(brain) == 1
    brain.correct.assert_called_once_with("stop", "SAFETY")
    assert turns.result(cid)["corrected"] == "SAFETY"
    assert turns.result(cid)["was"] == "GREET"
    for name in ("respond", "chat", "converse"):
        getattr(brain, name).assert_not_called()


def test_canonical_correction_safety_boundary(monkeypatch):
    from snn_data_lake.brainstem.so101_arm import So101Arm
    from snn_data_lake.cortex.brain import Brain, make_brain
    from snn_data_lake.cortex.live_brain import LiveBrain
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        brain = make_brain(seed=0, persist=False)
        try:
            with monkeypatch.context() as patch:
                spies = []
                # The reflex trips through AutonomousArm.apply_observed("EMERGENCY_STOP");
                # So101Arm.apply latches it and _persist_latch records the durable edge.
                for owner, name in (
                    (Brain, "respond"),
                    (LiveBrain, "chat"),
                    (LiveBrain, "converse"),
                    (So101Arm, "apply"),
                    (So101Arm, "_persist_latch"),
                    (brain.brain.arm, "apply_observed"),
                ):
                    spy = Mock(side_effect=AssertionError(f"correction reached {name}"))
                    patch.setattr(owner, name, spy)
                    spies.append(spy)
                for text, intent in (("stop", "SAFETY"), ("move the arm to the left", "ARM")):
                    was = brain.brain.route(text)
                    assert brain.correct(text, intent) == {
                        "intent": intent,
                        "was": was,
                        "provenance": "authoritative",
                    }
                with pytest.raises(ValueError):
                    brain.brain.correct("stop", "UNKNOWN")
                for spy in spies:
                    spy.assert_not_called()
                intents = brain.correction_intents()
                intents.clear()
                assert "SAFETY" in brain.correction_intents()
                patch.setattr(brain.brain, "_ixn", None)
                with pytest.raises(RuntimeError, match="interaction learning is off"):
                    brain.brain.correct("stop", "SAFETY")
        finally:
            brain.close()


@pytest.mark.parametrize("intent", ["", "  ", None, 1, "x" * 33])
def test_turns_rejects_bad_corrections(intent):
    turns = Turns()
    assert turns.submit_correction(99, "GOAL")[0] is None
    tid = turns.submit("hello")
    assert turns.submit_correction(tid, intent)[0] is None
    assert turns.submit_correction(True, "GOAL")[0] is None
    assert turns.submit_correction(str(tid), "GOAL")[0] is None


def test_turn_text_fifo_is_bounded():
    turns = Turns()
    old = turns.submit("old")
    for _ in range(_REPLY_KEEP):
        newest = turns.submit("new")
    assert len(turns._texts) == _REPLY_KEEP
    assert turns.submit_correction(old, "GOAL")[0] is None
    assert turns.submit_correction(newest, "GOAL")[0] is not None


def test_http_correction_queues_and_intents_are_captured():
    import io
    import json

    turns = Turns()
    tid = turns.submit("stop")
    base = _handler(lambda: pytest.fail("request read the brain"), turns, intents=["SAFETY"])
    replies = []

    class Offline(base):
        def __init__(self, path, body=None):
            self.path = path
            raw = json.dumps(body or {}).encode()
            self.rfile = io.BytesIO(raw)
            self.headers = {"Content-Length": str(len(raw))}

        def _send(self, code, body, ctype="application/json"):
            replies.append((code, json.loads(body)))

    Offline("/api/intents").do_GET()
    assert replies.pop() == (200, ["SAFETY"])
    Offline("/api/chat", {"correct": tid, "intent": "SAFETY"}).do_POST()
    code, queued = replies.pop()
    assert code == 200 and queued["queued"]
    assert turns.result(queued["id"]) is None
    Offline("/api/chat", {"correct": 999, "intent": "SAFETY"}).do_POST()
    assert replies.pop()[0] == 400
