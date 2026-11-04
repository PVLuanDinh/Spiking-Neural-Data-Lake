"""Real-owner protocol controls, not a deciding growth/pruning campaign."""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_g9_growth_adapter import SEED, _factory, _owner


def _recipe():
    return {
        "schema": "g9.engine.cycle.v1",
        "batch_size": 9,
        "candidate_pool": 128,
        "input_sigma": 1.0,
        "threshold_min": 0.5,
        "threshold_max": 2.0,
        "window": 12,
        "t_post_g": 15.0,
        "trace_epochs": 20,
        "trace_eta": 0.05,
        "feed_norm": 1.0,
        "return_weight": 0.2,
        "return_delay": 0,
        "max_epochs": 400,
        "stable_epochs": 10,
        "selection": "all-active/no-prune",
    }


def _bind(brain, pid="ii.xor", *, factory=None):
    from snn_data_lake.growth.growth_loop import CanonicalGrowthAdapter
    from snn_data_lake.instruments.taught_problems import PROBLEMS

    problem = PROBLEMS[pid]
    schedule = tuple(tuple(tuple(e) for e in epoch) for epoch in problem.teach_epochs(SEED, 400))
    return CanonicalGrowthAdapter.for_brain(
        brain,
        problem,
        factory=_factory(brain, pid) if factory is None else factory,
        teach_schedule=schedule,
        projections={},
        cycle=_recipe(),
    )


def _meter(adapter):
    from snn_data_lake.instruments.joint_mask_search import Meter

    m = Meter(adapter)
    m.freeze((True,) * len(adapter.segment))
    return m


def _events():
    from snn_data_lake.substrate.learner_events import KIND_SPECS, scan_learner_events

    events, *_ = scan_learner_events(tuple(KIND_SPECS), strict=True, return_receipt=True)
    return events


def _masks(brain):
    return {"neuron_mask": [True] * len(brain.brain._ixn.neurons), "return_mask": []}


def _carriers(brain):
    return {
        "shared": brain.brain._ixn._snapshot(),
        "actors": {pid: actor.state_dict() for pid, actor in brain._g9_actors.items()},
    }


def test_joint_episode_uses_actual_shared_neural_output_and_one_event():
    from snn_data_lake.readout.interaction_learning import _utterance_key

    with _owner() as (brain, _):
        adapter = _bind(brain)
        episode = next(adapter.problem.teach_epochs(SEED, 1))[0]
        ixn = brain.brain._ixn
        before = ixn.weights()
        brain.learn_engine_episode(adapter.problem.id, episode, **_masks(brain))
        assert ixn.weights() != before
        assert brain._g9_actors[adapter.problem.id].teaches == 1
        key = _utterance_key(episode[-1].text)
        assert ixn._authority[key] == episode[-1].label
        events = _events()
        teaches = [
            e
            for e in events
            if e["kind"] == "posed_problem" and json.loads(e["source"])["op"] == "engine_teach"
        ]
        assert len(teaches) == 1 and ixn._replay_ids == [teaches[0]["event_id"]]
        assert not any(e["kind"] in {"teach", "turn"} for e in events)
        expected, weights = ixn.route_text(episode[-1].text), ixn.weights()
        ixn._authority[key] = next(label for label in ixn.responses if label != expected)
        answer = brain.read_engine_episode(adapter.problem.id, episode, **_masks(brain))
        assert answer == ixn.route_text(episode[-1].text) == expected
        assert ixn.weights() == weights
        ixn._authority[key] = episode[-1].label


def test_only_actual_bound_meter_can_freeze_and_winner_is_immutable():
    from snn_data_lake.instruments.joint_mask_search import Meter, SelectionLeak
    from snn_data_lake.readout.posed_problem import PosedRefused
    from snn_data_lake.substrate import home

    with _owner() as (brain, _):
        adapter = _bind(brain)
        lake = Path(home.knowledge_path()).read_bytes()
        fake = Meter(object())
        fake.freeze(())
        with pytest.raises(SelectionLeak, match="CONSUMER_REQUIRED"):
            brain.freeze_selection(adapter.problem.id, meter=fake)
        assert Path(home.knowledge_path()).read_bytes() == lake
        meter = _meter(adapter)
        with pytest.raises(SelectionLeak, match="already frozen"):
            meter.freeze(())
        receipt = brain.freeze_selection(adapter.problem.id, meter=meter)
        assert receipt["selection_id"] == _events()[-1]["event_id"]
        frozen = Path(home.knowledge_path()).read_bytes()
        with pytest.raises(PosedRefused, match="ALREADY_FROZEN"):
            brain.freeze_selection(adapter.problem.id, meter=meter)
        assert Path(home.knowledge_path()).read_bytes() == frozen


def test_claim_survives_old_snapshot_and_same_brain_rebind():
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _owner() as (brain, _):
        _bind(brain, "aii")
        adapter = _bind(brain)
        meter = _meter(adapter)
        brain.freeze_selection(adapter.problem.id, meter=meter)
        before = brain._g9_snapshot()
        claim = brain.claim_retention(adapter.problem.id, "aii", meter=meter)
        brain._g9_restore(before)
        assert brain._g9_actors[adapter.problem.id].retention_claims["aii"] == claim["claim_id"]
        rebound = _bind(brain, factory=brain._g9_actors[adapter.problem.id].factory)
        with pytest.raises(PosedRefused, match="ALREADY_CONSUMED"):
            brain.claim_retention(adapter.problem.id, "aii", meter=_meter(rebound))
        brain.claim_retention(adapter.problem.id, adapter.problem.id, meter=_meter(rebound))
        with pytest.raises(PosedRefused, match="ALREADY_CONSUMED"):
            brain.claim_retention(adapter.problem.id, adapter.problem.id, meter=_meter(rebound))


def test_claim_refuses_changed_frozen_model_before_append():
    from snn_data_lake.readout.posed_problem import PosedRefused
    from snn_data_lake.substrate import home

    with _owner() as (brain, _):
        adapter = _bind(brain)
        meter = _meter(adapter)
        brain.freeze_selection(adapter.problem.id, meter=meter)
        lake = Path(home.knowledge_path()).read_bytes()
        neuron = brain.brain._ixn.neurons[0]
        old = neuron.w[0]
        neuron.w[0] += 1.0
        with pytest.raises(PosedRefused, match="STATE_DRIFT"):
            brain.claim_retention(adapter.problem.id, adapter.problem.id, meter=meter)
        assert Path(home.knowledge_path()).read_bytes() == lake
        neuron.w[0] = old
        brain.claim_retention(adapter.problem.id, adapter.problem.id, meter=meter)


def test_failed_final_read_restores_whole_brain_but_stays_consumed():
    from snn_data_lake.instruments.joint_mask_search import RetentionSets
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _owner() as (brain, _):
        old_adapter = _bind(brain, "aii")
        adapter = _bind(brain)
        meter = _meter(adapter)
        brain.freeze_selection(adapter.problem.id, meter=meter)
        brain.claim_retention(adapter.problem.id, "aii", meter=meter)
        before = _carriers(brain)
        sets = RetentionSets.for_problem(old_adapter.problem, SEED)

        def interrupted(episode):
            brain._g9_engine_effect(
                brain.brain._ixn,
                brain._g9_actors,
                {
                    "op": "engine_read",
                    "problem_id": "aii",
                    "turns": [[t.text, t.role, None] for t in episode],
                    "shifted_label": None,
                    **_masks(brain),
                    "return_neutered": False,
                },
                "9900-999999",
            )
            raise RuntimeError("planted final interruption")

        with pytest.raises(RuntimeError, match="final interruption"):
            sets.read_final(
                meter, snapshot=brain._g9_snapshot, restore=brain._g9_restore, read_off=interrupted
            )
        assert _carriers(brain) == before
        with pytest.raises(PosedRefused, match="ALREADY_CONSUMED"):
            brain.claim_retention(adapter.problem.id, "aii", meter=meter)


def test_v2_claim_omission_is_refused_by_carrier_and_covered_prefix():
    from snn_data_lake.readout.posed_problem import PosedActor, PosedRefused

    with _owner() as (brain, _):
        adapter = _bind(brain)
        meter = _meter(adapter)
        brain.freeze_selection(adapter.problem.id, meter=meter)
        brain.claim_retention(adapter.problem.id, adapter.problem.id, meter=meter)
        saved = brain._g9_actors[adapter.problem.id].state_dict()
        stripped = copy.deepcopy(saved)
        stripped.pop("retention_claims")
        with pytest.raises(PosedRefused, match="complete actor"):
            PosedActor.restore(stripped, brain._g9_birth_records())
        erased = copy.deepcopy(saved)
        erased["retention_claims"] = {}
        tampered = dict(brain._g9_actors)
        tampered[adapter.problem.id] = PosedActor.restore(erased, brain._g9_birth_records())
        with pytest.raises(PosedRefused, match="PREFIX_MISMATCH"):
            brain._g9_check_protocol_prefix(brain.brain._ixn, tampered, _events())


@pytest.mark.parametrize("checkpoint", [True, False])
def test_cold_restore_keeps_claim_and_engine_authority_without_heldout_replay(checkpoint):
    with _owner() as (brain, private):
        adapter = _bind(brain)
        episode = next(adapter.problem.teach_epochs(SEED, 1))[0]
        brain.learn_engine_episode(adapter.problem.id, episode, **_masks(brain))
        meter = _meter(adapter)
        brain.freeze_selection(adapter.problem.id, meter=meter)
        brain.claim_retention(adapter.problem.id, adapter.problem.id, meter=meter)
        expected = {
            "authority": dict(brain.brain._ixn._authority),
            "ring_ids": list(brain.brain._ixn._replay_ids),
            "claims": dict(brain._g9_actors[adapter.problem.id].retention_claims),
        }
        brain.close()
        brain._r2_draft_closed = True
        child = """
import gc,json,os
from snn_data_lake.substrate import home
from snn_data_lake.instruments.joint_mask_search import RetentionSets
def forbidden(*a,**k): raise RuntimeError('held-out replay forbidden')
RetentionSets._read=staticmethod(forbidden)
if os.environ['G9_USE_CHECKPOINT']=='0':
 original=home.load_state
 def loaded(name,*a,**k):
  return None if name=='interaction_learner.json' else original(name,*a,**k)
 home.load_state=loaded
from snn_data_lake.engine.brain import make_brain
b=make_brain(seed=9900,persist=True)
try:
 assert b._r2_fault is None,b._r2_fault
 a=b._g9_actors['ii.xor']
 value={'authority':b.brain._ixn._authority,'ring_ids':b.brain._ixn._replay_ids,
        'claims':a.retention_claims}
 print('CLAIM COLD '+json.dumps(value),flush=True)
finally:
 b.close();del b;gc.collect()
"""
        env = dict(
            os.environ,
            NORD_HOME=str(private),
            NORD_PERSIST="1",
            G9_USE_CHECKPOINT="1" if checkpoint else "0",
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
        line = next(line for line in out.stdout.splitlines() if line.startswith("CLAIM COLD "))
        assert json.loads(line[len("CLAIM COLD ") :]) == expected
