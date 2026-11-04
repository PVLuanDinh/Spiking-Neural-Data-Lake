"""Explicit feedback-policy controls, not a scientific acquisition/retention campaign."""

import copy
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from test_g9_growth_adapter import SEED, _factory, _owner
from test_g9_retention_consumption import _events, _masks, _recipe


def _source(text, target, eligible):
    return {
        "op": "engine_teach",
        "problem_id": "ii.xor",
        "turns": [[text, "probe", target]],
        "shifted_label": None,
        "neuron_mask": list(eligible),
        "return_mask": [],
        "return_neutered": False,
    }


def _preview(corrective=True):
    from snn_data_lake.instruments.taught_problems import canonical_incumbent
    from snn_data_lake.readout.posed_problem import PosedActor

    learner = canonical_incumbent(SEED)
    owner = SimpleNamespace(
        brain=SimpleNamespace(_ixn=learner), _g9_actors={}, _g9_apply_receipt=None
    )
    factory = _factory(owner, "ii.xor")
    cycle = _recipe()
    if corrective:
        cycle["feedback_gate"] = "corrective"
    factory["engine_binding"] = {
        "schedule": [[[["cat dog sun sky", "probe", "GREET"]]]],
        "projections": {},
        "cycle": cycle,
    }
    actor = PosedActor(factory, {})
    return owner, learner, actor, {"ii.xor": actor}


@pytest.mark.parametrize("corrective", [False, True])
@pytest.mark.parametrize("right", [False, True])
def test_preview_gates_only_shared_deposits_and_preserves_actual_part_d(corrective, right):
    from snn_data_lake.engine.brain import Brain
    from snn_data_lake.readout.interaction_learning import _utterance_key
    from snn_data_lake.readout.posed_problem import PosedActor

    owner, learner, actor, actors = _preview(corrective)
    text = "cat dog sun sky"
    prediction = learner.route_text(text)
    other = next(label for label in learner.responses if label != prediction)
    target = prediction if right else other
    source = _source(text, target, [True] * len(learner.responses))
    # A misleading authority cache must not decide whether a neural correction is needed.
    learner._authority[_utterance_key(text)] = other if right else target
    control = PosedActor(copy.deepcopy(actor.factory), {})
    control._effect(source, event_id="1-1")
    before = learner.weights()
    clock = learner.stack._global_t
    previous = learner._g9_neuron_mask

    answer = Brain._g9_engine_effect(owner, learner, actors, source, "1-1")

    assert actor.state_dict() == control.state_dict()
    assert actor.teaches == 1
    assert learner._authority[_utterance_key(text)] == target
    assert learner._replay_ids == ["1-1"]
    assert learner._g9_neuron_mask is previous
    assert answer == learner.route_text(text)
    if corrective and right:
        assert learner.weights() == before
        assert learner.stack._global_t == clock
    else:
        assert learner.weights() != before
        assert learner.stack._global_t > clock


def test_preview_corrective_feedback_respects_a_disabled_caller_mask():
    from snn_data_lake.engine.brain import Brain

    owner, learner, actor, actors = _preview()
    text = "cat dog sun sky"
    target = next(label for label in learner.responses if label != learner.route_text(text))
    before = learner.weights()
    Brain._g9_engine_effect(
        owner, learner, actors, _source(text, target, [False] * len(learner.responses)), "1-1"
    )
    assert learner.weights() == before
    assert learner.stack._global_t == 0
    assert actor.teaches == 1
    assert learner._replay_ids == ["1-1"]


def test_preview_checkpoint_prefix_hydrates_without_gating_or_relearning(monkeypatch):
    from snn_data_lake.engine.brain import Brain
    from snn_data_lake.readout.interaction_learning import _utterance_key

    owner, learner, actor, actors = _preview()
    text = "cat dog sun sky"
    target = learner.responses[0]
    before, weights = actor.state_dict(), learner.weights()

    def forbidden(*args, **kwargs):
        raise AssertionError("covered checkpoint event must only hydrate its caches")

    monkeypatch.setattr(learner, "route_text", forbidden)
    monkeypatch.setattr(actor, "_effect", forbidden)
    Brain._g9_engine_effect(
        owner,
        learner,
        actors,
        _source(text, target, [True] * len(learner.responses)),
        "1-1",
        cache_only=True,
    )
    assert actor.state_dict() == before
    assert learner.weights() == weights
    assert learner.stack._global_t == 0
    assert learner._authority[_utterance_key(text)] == target
    assert learner._replay_ids == ["1-1"]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_actual_owner_corrective_feedback_replays_exactly_in_a_fresh_process(checkpoint):
    from snn_data_lake.growth.growth_loop import CanonicalGrowthAdapter
    from snn_data_lake.instruments.taught_problems import PROBLEMS, Turn
    from snn_data_lake.readout.posed_problem import engine_model_digest

    with _owner() as (brain, private):
        problem = PROBLEMS["ii.xor"]
        schedule = tuple(tuple(epoch) for epoch in problem.teach_epochs(SEED, 3))
        CanonicalGrowthAdapter.for_brain(
            brain,
            problem,
            factory=_factory(brain, problem.id),
            teach_schedule=schedule,
            projections={},
            cycle={**_recipe(), "feedback_gate": "corrective"},
        )
        learner = brain.brain._ixn
        text = schedule[0][0][-1].text
        prediction = learner.route_text(text)
        before, clock = learner.weights(), learner.stack._global_t
        brain.learn_engine_episode(problem.id, (Turn(text, "probe", prediction),), **_masks(brain))
        assert learner.weights() == before and learner.stack._global_t == clock
        other = next(label for label in learner.responses if label != learner.route_text(text))
        brain.learn_engine_episode(problem.id, (Turn(text, "probe", other),), **_masks(brain))
        assert learner.stack._global_t > clock
        teaches = [
            e
            for e in _events()
            if e["kind"] == "posed_problem" and json.loads(e["source"])["op"] == "engine_teach"
        ]
        assert len(teaches) == 2
        assert learner._replay_ids == [e["event_id"] for e in teaches]
        assert brain._g9_actors[problem.id].teaches == 2
        expected = {
            "digest": engine_model_digest(learner, brain._g9_actors),
            "authority": dict(learner._authority),
            "ring_ids": list(learner._replay_ids),
            "clock": learner.stack._global_t,
            "prediction": learner.route_text(text),
        }
        brain.close()
        brain._r2_draft_closed = True
        child = """
import gc,json,os
from snn_data_lake.substrate import home
if os.environ['G9_USE_CHECKPOINT']=='0':
 original=home.load_state
 def loaded(name,*a,**k):
  return None if name=='interaction_learner.json' else original(name,*a,**k)
 home.load_state=loaded
from snn_data_lake.engine.brain import make_brain
from snn_data_lake.instruments.taught_problems import Turn
from snn_data_lake.readout.posed_problem import engine_model_digest
b=make_brain(seed=9900,persist=True)
try:
 assert b._r2_fault is None,b._r2_fault
 ixn=b.brain._ixn;a=b._g9_actors['ii.xor'];text=os.environ['G9_TEXT']
 assert a.factory['engine_binding']['cycle']['feedback_gate']=='corrective'
 value={'digest':engine_model_digest(ixn,b._g9_actors),'authority':ixn._authority.copy(),
        'ring_ids':ixn._replay_ids.copy(),'clock':ixn.stack._global_t,
        'prediction':ixn.route_text(text)}
 clock=ixn.stack._global_t;teaches=a.teaches
 b.learn_engine_episode('ii.xor',(Turn(text,'probe',ixn.route_text(text)),),
   neuron_mask=[True]*len(ixn.responses),return_mask=[])
 assert ixn.stack._global_t==clock and a.teaches==teaches+1
 print('CORRECTIVE COLD '+json.dumps(value),flush=True)
finally:
 b.close();del b;gc.collect()
"""
        env = dict(
            os.environ,
            NORD_HOME=str(private),
            NORD_PERSIST="1",
            G9_USE_CHECKPOINT="1" if checkpoint else "0",
            G9_TEXT=text,
        )
        remaining = (int(env["SNDL_R2_CHILD_DEADLINE_NS"]) - time.time_ns()) / 1e9
        assert remaining > 0
        out = subprocess.run(
            [sys.executable, "-B", "-X", "gil=0", "-c", child],
            env=env,
            capture_output=True,
            text=True,
            timeout=remaining,
        )
        assert out.returncode == 0, out.stderr[-1500:]
        line = next(line for line in out.stdout.splitlines() if line.startswith("CORRECTIVE COLD "))
        assert json.loads(line[len("CORRECTIVE COLD ") :]) == expected
