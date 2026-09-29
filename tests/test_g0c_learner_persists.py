"""G0(c): exact continuation, event-owned authority, atomic refusals and production hooks."""

import copy
import gc
import json
import math
import random
from collections import deque
from contextlib import contextmanager

import pytest

from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.cortex.live_brain import _IXN_INTENTS
from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
from snn_data_lake.readout import interaction_learning as ixn
from snn_data_lake.substrate.home import load_episodes, load_state, save_episode, save_state


@pytest.fixture(autouse=True)
def release_brain_cycles():
    yield
    gc.collect()


@contextmanager
def brain(persist=True):
    instance = make_brain(seed=100, persist=persist)
    try:
        yield instance
    finally:
        instance.close()
        del instance
        gc.collect()


def learner():
    return ixn.SpikingInteractionLearner(
        list(_IXN_INTENTS),
        52,
        seed=100,
        t_post=6,
        stack=PlasticityRuleStack(len(_IXN_INTENTS), seed=100),
    )


def event(key="a goal", intent="GOAL", seq=1, session=1):
    return {
        "kind": "teach",
        "event_id": f"{session}-{seq}",
        "stream": "chat",
        "session": str(session),
        "seq": str(seq),
        "source": key,
        "target": intent,
        "provenance": "authoritative",
        "encoder": "text_features n_in=52",
        "schema": "g0c.v1",
        "valid": "1",
    }


def test_scheduled_round_trip_shared_lazy_and_exact():
    with brain(persist=False) as b:
        ixn._g0c_schedule(b, 100)
        source = b.brain._ixn
        saved = json.loads(json.dumps(source.state_dict(), allow_nan=False))
        fresh = learner()
        stack = fresh.stack
        shared = (stack._short, stack._homeo, stack._intr)
        assert all(v is None for v in stack._meta._bcm)
        fresh.load_state_dict(saved)
        assert fresh.stack is stack
        assert shared == (stack._short, stack._homeo, stack._intr)
        assert any(v is not None for v in stack._meta._bcm)
        assert any(v is not None for v in stack._cons._cas)
        for obj in shared:
            assert any(obj is v for v in stack._pre + stack._post)
        assert fresh.state_dict() == saved
        fresh.rebuild_authority(
            [event(key, intent, seq=i) for i, (key, intent) in enumerate(source._authority.items())]
        )
        assert ixn._g0c_walk(fresh) == ixn._g0c_walk(source)
        for lrn in (source, fresh):
            lrn.learn_turn(ixn.G0_CORRECTED[0][0], "GREET")
            lrn.replay(k=8)
        assert ixn._g0c_walk(fresh) == ixn._g0c_walk(source)


def test_codec_containers_and_float_bits():
    value = {
        "reserved": {"__cls__": "ordinary dict"},
        "tuple": (1, -0.0),
        "set": {"a", "b"},
        "deque": deque([math.nextafter(1.0, 2.0)], maxlen=7),
        "rng": random.Random(19),
    }
    encoded = json.loads(json.dumps(ixn._encode_state(value)))
    restored = ixn._merge_state({}, encoded)
    assert restored["reserved"] == value["reserved"]
    assert restored["tuple"][1].hex() == (-0.0).hex()
    assert restored["deque"][0].hex() == value["deque"][0].hex()
    assert restored["deque"].maxlen == 7
    assert restored["set"] == value["set"]
    assert restored["rng"].getstate() == value["rng"].getstate()


@pytest.mark.parametrize("bad", [object(), lambda: None, {1: "non-string key"}])
def test_r_c4_unknown_value_refused(bad):
    lrn = learner()
    lrn.unknown = bad
    with pytest.raises(ValueError):
        lrn.state_dict()


@pytest.mark.parametrize("fault", ["class", "attrs", "rng", "schema", "lazy_attrs"])
def test_refusal_is_atomic(fault):
    source, target = learner(), learner()
    source.learn_turn("a goal", None, feedback="GOAL")
    saved = source.state_dict()
    before = ixn._g0c_walk(target)
    neuron, stack = target.neurons[0], target.stack
    if fault == "class":
        saved["tree"]["attrs"]["stack"]["__cls__"] = "wrong.Class"
    elif fault == "attrs":
        saved["tree"]["attrs"]["stack"]["attrs"]["unexpected"] = 1
    elif fault == "rng":
        saved["tree"]["attrs"]["rng"]["state"] = []
    elif fault == "schema":
        saved["schema"] = "future"
    else:
        meta = saved["tree"]["attrs"]["stack"]["attrs"]["_meta"]["attrs"]["_bcm"]
        next(v for v in meta if v is not None)["attrs"]["unknown"] = 1
    with pytest.raises(ValueError):
        target.load_state_dict(saved)
    assert target.neurons[0] is neuron and target.stack is stack
    assert ixn._g0c_walk(target) == before


def test_declared_exclusions_keep_runtime_wiring():
    source, target = learner(), learner()
    target.teach_sink = object()
    target._authority = {"existing": "GREET"}
    before = {k: getattr(target, k) for k in ("teach_sink", "_authority")}
    target.load_state_dict(source.state_dict())
    assert all(getattr(target, k) is v for k, v in before.items())


@pytest.mark.parametrize("name", ["delta_sink", "_syn_delta", "readout_zoo"])
def test_frozen_protocol_does_not_silently_exclude_other_state(name):
    lrn = learner()
    assert name in lrn.state_dict()["tree"]["attrs"]
    setattr(lrn, name, object())
    with pytest.raises(ValueError):
        lrn.state_dict()


def test_authority_latest_sequence_and_reconciliation_without_rng():
    lrn = learner()
    lrn.learn_turn("a goal", "GREET")
    rng = lrn.rng.getstate()
    lrn.rebuild_authority([event(seq=9), event(intent="MEMORY", seq=10), event(seq=2)])
    assert lrn._authority == {"a goal": "MEMORY"}
    assert [label for _, label, _ in lrn._replay] == ["MEMORY"]
    assert lrn.rng.getstate() == rng
    lrn.rebuild_authority([event(seq=100), event(intent="ARM", seq=1, session=2)])
    assert lrn._authority == {"a goal": "ARM"}


def test_r_c3_events_without_checkpoint_supersede():
    save_episode(event())
    assert load_state("interaction_learner.json") is None
    with brain() as b:
        b.brain._ixn.learn_turn("a goal", "GREET")
        assert b.brain._ixn._replay[-1][1] == "GOAL"


def test_r_c3_checkpoint_without_events_does_not_supersede():
    source = learner()
    source.learn_turn("a goal", None, feedback="GOAL")
    save_state("interaction_learner.json", source.state_dict())
    assert load_episodes(kinds=("teach",)) == []
    with brain() as b:
        assert b.brain._ixn._authority == {}
        b.brain._ixn.learn_turn("a goal", "GREET")
        assert b.brain._ixn._replay[-1][1] == "GREET"


def test_r_c4_nonpersisting_brain_writes_no_learner_or_teaching():
    with brain(persist=False) as b:
        b.correct("a goal", "GOAL")
        b._enter_wake()
        assert b._flush_interaction_learner() is False
    assert load_state("interaction_learner.json") is None
    assert load_episodes(kinds=("teach",)) == []


def test_r_c4_restore_refusal_is_visible_and_fresh(capsys):
    bad = learner().state_dict()
    bad["tree"]["attrs"]["stack"]["__cls__"] = "wrong.Class"
    save_state("interaction_learner.json", bad)
    with brain() as b:
        assert b.brain._ixn.state_dict() == learner().state_dict()
        err = capsys.readouterr().err
        assert err.count("restore refused; learner stays fresh") == 1
        assert err.isascii()


def test_r_c4_episodes_filtered_before_limit():
    autobiographical = [
        {"source": "first", "target": "memory"},
        {"source": "last", "target": "memory"},
    ]
    for i, episode in enumerate(autobiographical):
        save_episode(episode)
        save_episode(event(seq=i))
    assert load_episodes() == autobiographical
    assert load_episodes(limit=1) == autobiographical[-1:]
    assert load_episodes(kinds=("teach",), limit=1) == [event(seq=1)]
    assert load_episodes(kinds=()) == []
    from snn_data_lake.hippocampus.lake_recall import LakeRecall

    recall = LakeRecall()
    assert recall.refresh() == len(autobiographical)
    assert recall._episodes == autobiographical


def test_production_close_night_and_immediate_teaching():
    with brain() as b:
        b.correct("a goal", "GOAL")
        events = load_episodes(kinds=("teach",))
        assert len(events) == 1
        assert events[0] == event(session=b._teach_session)
        assert load_state("interaction_learner.json") is None
        b._enter_wake()
        assert load_state("interaction_learner.json") == b.brain._ixn.state_dict()
        assert b._flush_interaction_learner() is False
        b.brain._ixn.learn_turn("another goal", "GOAL")
        before_close = copy.deepcopy(b.brain._ixn.state_dict())
    assert load_state("interaction_learner.json") == before_close
    with brain() as restored:
        assert restored.brain._ixn.state_dict() == before_close


def test_save_refusal_is_reported_without_a_record(capsys):
    with brain() as b:
        b.brain._ixn.unknown = object()
        assert b._flush_interaction_learner() is False
        assert "save refused" in capsys.readouterr().err
        assert load_state("interaction_learner.json") is None
        del b.brain._ixn.unknown


def verdict_fixture():
    """Synthetic records test the scorer, without executing registered campaign seeds."""
    results = []
    for seed in ixn.G0_SEEDS:
        arms = {
            role: {
                "head": "one-head",
                "dirty": False,
                "valid": True,
                **{part: ["same"] for part in ixn._G0C_RECORDS},
            }
            for role in ixn.G0C_ROLES
        }
        arms["N"]["tree_before"] = ["cold"]
        arms["P1x"]["checkpoint_ring_labels"] = ["GOAL"]
        for role in ("A_x", "O_rec"):
            arms[role]["recorrection"] = {
                "key": "corrected",
                "intent": "MEMORY",
                "authority": "MEMORY",
                "ring": ["MEMORY" if role == "A_x" else "GOAL"],
            }
            arms[role]["draws"] = [["corrected", "MEMORY"]]
        arms["O_stack"]["tree_before"] = ["cold stack"]
        results.append({"seed": seed, "arms": arms})
    return results


def test_verdict_validity_and_reachable_reds():
    results = verdict_fixture()
    verdict = ixn.g0c_verdict(results)
    assert verdict["verdict"] == "PASS"
    assert verdict["attribution"]["O_stack"]["tree_before"] == 8
    assert verdict["next_step_unexercised"] == ["O_stack"]
    for fault in ("missing_seed", "duplicate_seed", "dirty", "mixed_head", "no_ring", "no_red"):
        broken = copy.deepcopy(results)
        if fault == "missing_seed":
            broken.pop()
        elif fault == "duplicate_seed":
            broken.append(copy.deepcopy(broken[0]))
        elif fault == "dirty":
            broken[0]["arms"]["A"]["dirty"] = True
        elif fault == "mixed_head":
            broken[0]["arms"]["A"]["head"] = "another-head"
        elif fault == "no_ring":
            broken[0]["arms"]["P1x"]["checkpoint_ring_labels"] = []
        else:
            for result in broken:
                result["arms"]["O_rec"]["recorrection"]["ring"] = ["MEMORY"]
        assert ixn.g0c_verdict(broken)["verdict"] == "RED", fault


@pytest.mark.parametrize("part", ixn._G0C_RECORDS)
def test_each_exact_record_can_red_r_c1(part):
    results = verdict_fixture()
    results[0]["arms"]["A"][part] = ["different"]
    assert ixn.g0c_verdict(results)["R-c1"] == "RED"


def test_restore_failure_of_any_kind_leaves_a_working_fresh_brain(monkeypatch, capsys):
    save_state(
        "interaction_learner.json", {"schema": "g0c.v1", "tree": {"__cls__": "x", "attrs": {}}}
    )

    def boom(self, state):
        raise RuntimeError("constructed restore failure")

    monkeypatch.setattr(ixn.SpikingInteractionLearner, "load_state_dict", boom)
    with brain() as b:
        assert b.brain._ixn is not None
        assert b.brain.route("what is your goal") in _IXN_INTENTS
    assert "RuntimeError: constructed restore failure" in capsys.readouterr().err


def test_teaching_sessions_count_up_per_restart_without_the_wall_clock():
    with brain() as b:
        b.correct("what are you pursuing right now", "GOAL")
    with brain() as b:
        b.correct("tell me what happened earlier here", "MEMORY")
    sessions = [e["session"] for e in load_episodes(kinds=("teach",))]
    assert sessions == ["1", "2"]


def test_lazy_classes_restore_without_source(monkeypatch):
    import inspect

    trained = learner()
    for _ in range(3):
        trained.learn_turn("what is your goal", None, feedback="GOAL")
    saved = trained.state_dict()

    def no_source(obj):
        raise OSError("no source in a frozen build")

    monkeypatch.setattr(inspect, "getsource", no_source)
    fresh = learner()
    fresh.load_state_dict(saved)
    assert fresh.state_dict() == saved


def test_lazy_schema_matches_the_constructors():
    import ast
    import importlib
    import inspect
    import textwrap

    for name, attrs in ixn._G0C_LAZY_ATTRS.items():
        module, _, qualname = name.rpartition(".")
        cls = getattr(importlib.import_module(module), qualname)
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
        assigned = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == "self"
        }
        assert assigned == set(attrs), name


def test_checkpoint_read_failure_leaves_a_working_brain(monkeypatch, capsys):
    import snn_data_lake.substrate.home as home_mod

    real = home_mod.load_state

    def failing(name, default=None):
        if name == "interaction_learner.json":
            raise OSError("constructed checkpoint read failure")
        return real(name, default)

    monkeypatch.setattr(home_mod, "load_state", failing)
    with brain() as b:
        assert b.brain.route("what is your goal") in _IXN_INTENTS
    assert "OSError: constructed checkpoint read failure" in capsys.readouterr().err


def test_teaching_event_write_failure_does_not_break_a_correction(monkeypatch, capsys):
    import snn_data_lake.substrate.home as home_mod

    def boom(episode, path=None):
        raise OSError("constructed lake write failure")

    with brain() as b:
        monkeypatch.setattr(home_mod, "save_episode", boom)
        b.correct("what are you pursuing right now", "GOAL")
        assert b.brain._ixn.provenance("what are you pursuing right now") == "authoritative"
    assert "teaching event not written" in capsys.readouterr().err
