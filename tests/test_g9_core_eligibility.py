"""Protected-core circuit controls, not a claim that the curriculum has been solved."""

import copy
import json
import os
import subprocess
import sys
import time

import pytest
from test_g9_corrective_feedback import _preview
from test_g9_growth_adapter import SEED, _batch, _factory, _owner
from test_g9_retention_consumption import _masks, _recipe


def _core(learner):
    from snn_data_lake.readout.interaction_learning import _encode_state

    return {
        "neurons": _encode_state(learner.neurons),
        "stack": _encode_state(learner.stack),
        "last": _encode_state(learner._last),
        "syn_delta": _encode_state(learner._syn_delta),
    }


def _pure(core_eligible=False):
    from snn_data_lake.growth.grown_code import GrownUnit
    from snn_data_lake.instruments.taught_problems import canonical_incumbent
    from snn_data_lake.readout.interaction_learning import text_features
    from snn_data_lake.readout.posed_problem import unit_rows

    learner = canonical_incumbent(SEED)
    for text, target in (
        ("ram cup emu cod", "GREET"),
        ("cup ram cod emu", "GOAL"),
    ):
        learner.interact(text_features(text, learner.n_in), target)
    unit = GrownUnit(0, (0.1,) * 52, 0.5, 12, 1, (0.0,) * len(learner.responses))
    learner._g9_context = {
        "problem_id": "ii.xor",
        "units": unit_rows((unit,)),
        "afferents": [[0, []]],
        "t_post_g": 15.0,
        "out_eligible": [[True] * len(learner.responses)],
        "core_eligible": [core_eligible] * len(learner.responses),
    }
    code = learner.encode(text_features("cat dog sun sky", learner.n_in))
    return learner, code


def test_frozen_core_deposits_only_on_actual_grown_outputs_with_raw_feedback():
    from snn_data_lake.readout.posed_problem import chat_grown_deposit

    learner, code = _pure()
    reference = copy.deepcopy(learner)
    calls = []
    learner.delta_sink = calls.append
    before = _core(learner)
    units = copy.deepcopy(learner._g9_context["units"])
    chat_grown_deposit(reference, 0, code, 0.5)

    assert learner._deposit(0, code, 0.5) == [0.0] * learner.n_in

    assert _core(learner) == before
    assert calls == []
    assert learner._g9_context["units"] != units
    assert learner._g9_context == reference._g9_context


def test_whole_response_mask_still_blocks_core_and_grown_updates():
    learner, code = _pure()
    learner._g9_neuron_mask = [False] * len(learner.responses)
    before = learner._snapshot()
    assert learner._deposit(0, code, 1.0) == [0.0] * learner.n_in
    assert learner._snapshot() == before


@pytest.mark.parametrize("mask", [None, [False] * 8, [0] * 9, (False,) * 9, "frozen"])
def test_malformed_core_mask_is_refused_before_any_neural_change(mask):
    from snn_data_lake.readout.posed_problem import PosedRefused

    learner, code = _pure()
    learner._g9_context["core_eligible"] = mask
    before = learner._snapshot()
    with pytest.raises(PosedRefused, match="core circuits"):
        learner._deposit(0, code, 1.0)
    assert learner._snapshot() == before


def test_true_core_eligibility_is_exactly_the_legacy_continuation():
    learner, code = _pure(core_eligible=True)
    legacy = copy.deepcopy(learner)
    del legacy._g9_context["core_eligible"]
    before = _core(learner)
    assert learner._deposit(0, code, 1.0) == legacy._deposit(0, code, 1.0)
    assert _core(learner) == _core(legacy)
    assert _core(learner) != before
    assert learner._g9_context["units"] == legacy._g9_context["units"]


def test_actual_actor_derives_core_mask_only_from_its_frozen_recipe():
    from snn_data_lake.readout.interaction_learning import text_features
    from snn_data_lake.readout.posed_problem import _chat_context, actor_context

    _, learner, actor, _ = _preview()
    assert "core_eligible" not in actor_context(actor)
    actor.factory["engine_binding"]["cycle"]["core_plasticity"] = "frozen"
    learner._g9_context = actor_context(actor)
    assert learner._g9_context["core_eligible"] == [False] * len(learner.responses)
    assert _chat_context(learner)[0] is learner._g9_context
    # An explicit pre-growth frozen context has no eligible synapses yet.
    before = learner._snapshot()
    code = learner.encode(text_features("cat dog sun sky", learner.n_in))
    assert learner._deposit(0, code, 1.0) == [0.0] * learner.n_in
    assert learner._snapshot() == before


def _bind_grown(brain):
    from snn_data_lake.growth.growth_loop import CanonicalGrowthAdapter
    from snn_data_lake.instruments.taught_problems import PROBLEMS

    problem = PROBLEMS["ii.xor"]
    schedule = tuple(tuple(epoch) for epoch in problem.teach_epochs(SEED, 3))
    adapter = CanonicalGrowthAdapter.for_brain(
        brain,
        problem,
        factory=_factory(brain, problem.id),
        teach_schedule=schedule,
        projections={
            "0": {
                "feed": [0.0] * 256,
                "return": [
                    {"source": label, "sign": 1, "weight": 0.2, "delay": 0}
                    for label in brain.brain._ixn.responses
                ],
            }
        },
        cycle={**_recipe(), "feedback_gate": "corrective", "core_plasticity": "frozen"},
    )
    _, units, rows, ticket = _batch(brain, adapter)
    adapter.admit(units, rows, ticket)
    return adapter, schedule[0][0][-1].text


@pytest.mark.parametrize("checkpoint", [False, True])
def test_actual_admitted_outputs_learn_and_frozen_core_survives_cold_restore(checkpoint):
    from snn_data_lake.instruments.taught_problems import Turn
    from snn_data_lake.readout.posed_problem import engine_model_digest, unit_rows

    with _owner() as (brain, private):
        adapter, text = _bind_grown(brain)
        learner = brain.brain._ixn
        before = _core(learner)
        units = unit_rows(adapter.segment)
        target = next(label for label in learner.responses if label != learner.route_text(text))
        brain.learn_engine_episode(
            adapter.problem.id,
            (Turn(text, "probe", target),),
            **{
                **_masks(brain),
                "return_mask": [True] * len(brain._g9_actors[adapter.problem.id].port.rows),
            },
        )
        assert _core(learner) == before
        assert unit_rows(brain._g9_actors[adapter.problem.id].units) != units
        expected = {
            "digest": engine_model_digest(learner, brain._g9_actors),
            "core": _core(learner),
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
from snn_data_lake.readout.posed_problem import engine_model_digest,unit_rows
from test_g9_core_eligibility import _core
b=make_brain(seed=9900,persist=True)
try:
 assert b._r2_fault is None,b._r2_fault
 ixn=b.brain._ixn;a=b._g9_actors['ii.xor'];text=os.environ['G9_TEXT']
 assert ixn._g9_context['core_eligible']==[False]*len(ixn.responses)
 value={'digest':engine_model_digest(ixn,b._g9_actors),'core':_core(ixn),
        'clock':ixn.stack._global_t,'prediction':ixn.route_text(text)}
 before=_core(ixn);units=unit_rows(a.units)
 target=next(label for label in ixn.responses if label!=ixn.route_text(text))
 b.learn_engine_episode('ii.xor',(Turn(text,'probe',target),),
   neuron_mask=[True]*len(ixn.responses),return_mask=[True]*len(a.port.rows))
 assert _core(ixn)==before
 assert unit_rows(a.units)!=units
 print('CORE COLD '+json.dumps(value),flush=True)
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
        line = next(line for line in out.stdout.splitlines() if line.startswith("CORE COLD "))
        assert json.loads(line[len("CORE COLD ") :]) == expected
