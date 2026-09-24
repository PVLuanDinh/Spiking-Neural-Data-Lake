"""G0(b)'s frozen safety checks and constructed scorer/verdict reds, never a campaign."""

import copy
import json
from unittest.mock import Mock

import pytest

from snn_data_lake.cortex import interaction_learning as ixn
from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.cortex.live_brain import _ROUTE_CUES, _route_tokens
from snn_data_lake.substrate.home import throwaway_home


@pytest.fixture
def brain():
    with throwaway_home():
        instance = make_brain(seed=100, persist=False)
        try:
            yield instance
        finally:
            instance.close()


@pytest.mark.parametrize("message", ["stop", "emergency stop", "no, stop", "yes stop"])
@pytest.mark.parametrize(
    "human,raises", [(True, False), (True, True), (False, False), (False, True)]
)
def test_pending_stop_matches_incumbent(message, human, raises):
    observations = []
    for pending in (None, "GOAL"):
        with throwaway_home():
            brain = make_brain(seed=100, persist=False)
            try:
                brain._route_pending = pending
                brain.brain.route = Mock(
                    side_effect=LookupError("constructed read failure") if raises else None,
                    return_value="GOAL",
                )
                apply = Mock(wraps=brain.brain.arm.arm.apply)
                brain.brain.arm.arm.apply = apply
                observed = Mock(wraps=brain.brain.arm.apply_observed)
                brain.brain.arm.apply_observed = observed
                reply = brain.respond(message, human=human)
                assert brain.brain.arm.arm.state()["stopped"]
                assert brain._route_pending == (None if human else pending)
                assert "EMERGENCY STOP engaged" in reply["reply"]
                observations.append(
                    (reply["reply"], reply["route"], apply.call_args_list, observed.call_args_list)
                )
                if not human:
                    brain.brain.route.assert_not_called()
            finally:
                brain.close()
    assert observations[0] == observations[1]


def test_branch_never_reaches_arm_firewall_or_latch(brain, monkeypatch):
    spies = []
    for owner, name in (
        (brain.brain.arm.arm, "apply"),
        (brain.brain.arm.arm, "_persist_latch"),
        (brain.brain.arm, "apply_observed"),
        (brain.brain._nl, "interpret"),
    ):
        spy = Mock(side_effect=AssertionError(f"route branch reached {name}"))
        monkeypatch.setattr(owner, name, spy)
        spies.append(spy)
    for intent in _ROUTE_CUES:
        assert brain._route_turn("novel wording", True, None, intent)[1].endswith(f"({intent})")
        assert brain._route_turn("yes! yeah.", True, intent, None)[1].startswith(
            "learned route: confirmed"
        )
        assert brain._route_turn("no, nope", True, intent, None)[1].startswith(
            "learned route: declined"
        )
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize("intent", ["GREET", "ARM", None])
def test_greet_arm_never_clarify(brain, intent):
    assert brain._route_turn("novel wording", True, None, intent) is None
    assert brain._route_pending is None


@pytest.mark.parametrize("message,pending", [("novel wording", None), ("yes", "GOAL")])
def test_nonhuman_never_clarifies_or_confirms(brain, message, pending):
    brain._route_pending = pending
    brain.brain.route = Mock(return_value="GOAL")
    reply = brain.respond(message, human=False)
    assert not reply["route"].startswith("learned route:")
    assert brain._route_pending == pending
    brain.brain.route.assert_not_called()


def test_ingest_between_clarification_and_yes_keeps_pending(brain):
    brain.brain.route = Mock(return_value="GOAL")
    asked = brain.respond("what are you pursuing right now", human=True)
    assert asked["route"] == "learned route: clarify (GOAL)"
    brain.respond("the train was late again this morning", human=False)
    assert brain._route_pending == "GOAL"
    confirmed = brain.respond("yes", human=True)
    assert confirmed["route"] == "learned route: confirmed (GOAL)"
    assert "my standing goal is to keep improving myself" in confirmed["reply"].lower()
    assert brain._route_pending is None


def test_other_branch_consumes_pending(brain):
    brain.brain.route = Mock(return_value="GOAL")
    assert brain.respond("novel wording", human=True)["route"] == "learned route: clarify (GOAL)"
    assert brain.respond("hello", human=True)["route"].startswith("Gc (crystallized")
    assert brain._route_pending is None
    assert not brain.respond("yes", human=True)["route"].startswith("learned route:")


@pytest.mark.parametrize("message", ["what is your goal", "stop", "close the claw onto the object"])
def test_commands_and_grounded_lines_never_enter_branch(brain, message):
    brain._route_pending = "GOAL"
    brain._route_turn = Mock(
        side_effect=AssertionError("higher priority turn reached route branch")
    )
    # Nonhuman motor command still takes the firewall refusal, without moving an arm.
    brain.respond(message, human=False)
    brain._route_turn.assert_not_called()
    assert (
        brain._route_pending == "GOAL"
    )  # ingest neither consumes nor clears a person's pending reading


def test_human_grounded_line_never_enters_branch_and_clears_pending(brain):
    brain._route_pending = "GOAL"
    brain._route_turn = Mock(
        side_effect=AssertionError("higher priority turn reached route branch")
    )
    reply = brain.respond("what is your goal", human=True)
    brain._route_turn.assert_not_called()
    assert "my standing goal is to keep improving myself" in reply["reply"].lower()
    assert brain._route_pending is None


def test_canonical_cues_contain_frozen_key(brain):
    for intent, (cue, _) in _ROUTE_CUES.items():
        text = brain.brain.converse(cue, human=False).lower()
        assert any(phrase in text for phrase in ixn.G0B_KEY[intent]), (intent, text)


def test_read_precedes_teaching_and_confirmation_preserves_authority(brain):
    calls = []
    original_chat = brain.brain.chat

    def route(message):
        calls.append(("route", message))
        return "GOAL"

    def chat(message, *, human=False):
        calls.append(("chat", message))
        return original_chat(message, human=human)

    brain.brain.route, brain.brain.chat = route, chat
    authority = dict(brain.brain._ixn._authority)
    first = brain.respond("novel wording", human=True)
    assert calls[:2] == [("route", "novel wording"), ("chat", "novel wording")]
    assert first["route"] == "learned route: clarify (GOAL)"
    second = brain.respond("Yes, sure!", human=True)
    assert second["route"] == "learned route: confirmed (GOAL)"
    assert ixn._g0b_score("GOAL", [(first["reply"], "yes"), (second["reply"], None)])["useful"]
    assert brain.brain._ixn._authority == authority
    assert brain.brain._ixn._replay[-1][1] == "GREET"
    assert brain._route_pending is None


def test_decline_switch_missing_learner_and_bare_answers(brain):
    assert _route_tokens("YES?! yep.,") == {"yes", "yep"}
    for message in ("yes", "no", "yes no"):
        assert brain._route_turn(message, True, None, "GOAL") is None
    assert brain._route_turn("no", True, "GOAL", "GOAL") == (
        "Understood -- I will not guess. Tell me more, or correct me on the chat page.",
        "learned route: declined (GOAL)",
    )
    brain.route_consumption = False
    assert brain._learned_route("novel wording", True) is None
    assert brain._route_turn("yes", True, "GOAL", "GOAL") is None
    brain.route_consumption = True
    brain.brain._ixn = None
    assert brain._learned_route("novel wording", True) is None
    assert brain._route_turn("novel wording", True, None, "GOAL") is None


def test_scorer_consent_full_text_and_hedge():
    question = "Did you mean to ask about my goal? Say yes and I will answer."
    answer = ixn.G0B_KEY["GOAL"][0]
    for person, useful in (("yes", True), ("no", False), (None, False)):
        assert ixn._g0b_score("GOAL", [(question, person), (answer, None)]) == {
            "useful": useful,
            "misfire": False,
        }
    assert ixn._g0b_score("GOAL", [(question + answer, None)]) == {"useful": False, "misfire": True}
    assert ixn._g0b_score("GOAL", [("Understood -- I will not guess. " + answer, None)])["misfire"]
    assert ixn._g0b_score(None, [(answer, None)]) == {"useful": False, "misfire": True}
    assert ixn._g0b_score("GOAL", [(answer + ixn.G0B_KEY["MEMORY"][0], None)])["misfire"]
    hedge = "I am not fully certain right now, but Right now all signals are holding steady."
    assert ixn._g0b_score(None, [(hedge, None)]) == {"useful": False, "misfire": False}
    assert ixn._g0b_score("GOAL", []) == {"useful": False, "misfire": False}


def constructed_results():
    """A sensitive constructed positive control, not measured runs."""
    results = []
    for seed in ixn.G0_SEEDS:
        for arm in ixn.G0B_ARMS:
            reply = ixn.G0B_KEY["GOAL"][0] if arm in ("A", "U") else "No answer."
            lines = []
            for text, truth, subset in (
                ("in scope", "GOAL", "corrected"),
                ("out of scope", None, "novel"),
            ):
                emitted = reply if truth is not None or arm == "U" else "No answer."
                lines.append(
                    {
                        "text": text,
                        "true_intent": truth,
                        "subset": subset,
                        "route_before": "GOAL",
                        "clarified": False,
                        "clarified_intent": None,
                        "turns": [
                            {
                                "reply": emitted,
                                "route": "constructed",
                                "learned_route": "GOAL",
                                "person_answer": None,
                            }
                        ],
                    }
                )
            results.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "head": "constructed",
                    "dirty": False,
                    "valid": True,
                    "invalid_reasons": [],
                    "lines": lines,
                    "teaching_events": [],
                }
            )
    return results


def test_verdict_positive_control_and_purity():
    runs = constructed_results()
    before = copy.deepcopy(runs)
    result = ixn.g0b_verdict(runs)
    assert result["verdict"] == "PASS"
    assert result["R-b1a"]["sign_flip_p"] == 1 / 256
    assert result["R-b1a"]["conditional_rejection_fraction"] == 9 / 256
    assert result["R-b4"]["U_verdict"] == "RED"
    assert runs == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("control,criterion", [("N", "R-b1b"), ("S", "R-b1a")])
def test_equal_method_and_control_is_red(control, criterion):
    runs = constructed_results()
    for run in runs:
        if run["arm"] == "A":
            run["lines"] = copy.deepcopy(
                next(r["lines"] for r in runs if r["arm"] == control and r["seed"] == run["seed"])
            )
    assert ixn.g0b_verdict(runs)[criterion]["verdict"] == "RED"


@pytest.mark.parametrize(
    "fault,criterion",
    [
        ("D text", "R-b2"),
        ("U no misfire", "R-b4"),
        ("A misfire", "R-b4"),
        ("seven", None),
        ("heads", None),
        ("extra", None),
    ],
)
def test_constructed_reachable_reds(fault, criterion):
    runs = constructed_results()
    for run in runs:
        if fault == "D text" and run["arm"] == "D" and run["seed"] == 0:
            run["lines"][0]["turns"][0]["reply"] = "Different non-answer."
        if fault == "U no misfire" and run["arm"] == "U":
            run["lines"][1]["turns"][0]["reply"] = "No answer."
        if fault == "A misfire" and run["arm"] == "A" and run["seed"] == 0:
            run["lines"][1]["turns"][0]["reply"] = ixn.G0B_KEY["GOAL"][0]
        if fault == "seven" and run["arm"] == "S" and run["seed"] == 0:
            run["valid"] = False
            run["invalid_reasons"] = ["constructed invalid run"]
    if fault == "heads":
        runs[0]["head"] = "other"
    if fault == "extra":
        runs.append({**runs[0], "seed": 100})
    verdict = ixn.g0b_verdict(runs)
    assert verdict["verdict"] == "RED"
    if criterion:
        assert verdict[criterion]["verdict"] == "RED"
    else:
        assert verdict["validity_reasons"]


def test_g0a_shared_statistics_are_byte_identical():
    """Frozen JSON hashes measured from the pre-extraction G0(a) verdict."""
    import hashlib

    from test_g0_correction import constructed_results as g0a_results

    for a, c, expected in (
        (1.0, 0.0, "e0387215191fc3ef9fa2798e749cf8867b0f649e3f4d8a86dc66a715621c38d8"),
        (0.5, 0.5, "1a01cc88cf8fb5d689d12133637b70edce09b35c2ec5f84e468c2b5ead7e2c06"),
        (0.0, 1.0, "8bee1b138f5df3c246442c46424230574c277194529169ce9dea0d10f23fc597"),
    ):
        runs = g0a_results(a=a, c=c)
        assert hashlib.sha256(json.dumps(ixn.g0_verdict(runs)).encode()).hexdigest() == expected
    runs = g0a_results()
    for run in runs:
        if run["arm"] == "A":
            run["M1"] = run["seed"] / 8
    assert hashlib.sha256(json.dumps(ixn.g0_verdict(runs)).encode()).hexdigest() == (
        "9e5418d65a9b77e495376f413160955706774ec8393d7bdd252834af7bea5a93"
    )
