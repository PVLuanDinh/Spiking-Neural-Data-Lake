"""Owner1039 retirement amendment2026-10-13; retained and retired sleep controls.

Tests replace old positive assertions for the three retired automatic paths.
Wake primitives and retained structural sleep remain separate requirements.
No historical measurement or frozen cohort is rewritten by this amendment.
"""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("element", ["replay", "benna_fusi", "attractor"])
def test_retired_sleep_neuter_refuses(element):
    from snn_data_lake.engine.brain import Brain

    receiver = SimpleNamespace(_sleep_caps={"structural": True})
    with pytest.raises(ValueError, match="retired sleep capability"):
        Brain.neuter_sleep(receiver, element)
    assert receiver._sleep_caps == {"structural": True}


def test_retained_structural_neuter_and_unknown_refusal():
    from snn_data_lake.engine.brain import Brain

    for element in ("structural", "phase"):
        receiver = SimpleNamespace(_sleep_caps={"structural": True})
        assert Brain.neuter_sleep(receiver, element) is receiver
        assert receiver._sleep_caps == {"structural": False}
    with pytest.raises(ValueError, match="unknown element"):
        Brain.neuter_sleep(receiver, "absent")


def test_structural_step_charges_without_retired_callbacks():
    from snn_data_lake.engine.brain import Brain
    from snn_data_lake.sleep.structural_rewiring import StructuralRewiring

    def retired(*args, **kwargs):
        raise AssertionError("retired automatic replay/rest was reached")

    adopted = []
    stack = SimpleNamespace(
        meta_theta=lambda: [0.0],
        adopt_structural=lambda row, weights, changed: adopted.append(
            (row, tuple(weights), tuple(changed))
        ),
        relax=retired,
        consolidation_calls=retired,
    )
    neuron = SimpleNamespace(w=[0.5, 0.5])
    learner = SimpleNamespace(stack=stack, neurons=[neuron], replay=retired)
    receiver = SimpleNamespace(_seed=0, _sleep_caps={"structural": True})
    rewirer = StructuralRewiring(seed=0)
    rewirer, report = Brain._r2_sleep_step(receiver, learner, rewirer)
    assert set(report) == {"_structural_report"}
    neuron.w[0] = -0.25  # reachable sign-violation red in the retained real rewirer
    rewirer, report = Brain._r2_sleep_step(receiver, learner, rewirer)
    assert report["_structural_report"]["pruned"] == 1
    assert adopted and adopted[-1][0] == 0 and 0 in adopted[-1][2]
    before = tuple(neuron.w)
    receiver._sleep_caps["structural"] = False
    same, report = Brain._r2_sleep_step(receiver, learner, rewirer)
    assert same is rewirer and report == {"_structural_report": None}
    assert tuple(neuron.w) == before


@pytest.mark.parametrize("family", ["F1", "F2", "F3", "F4"])
def test_retired_live_producers_refuse_before_effects(family, monkeypatch, tmp_path):
    from snn_data_lake.instruments import sleep_plasticity_ab as instrument

    def effect(*args, **kwargs):
        raise AssertionError("retired producer reached home/import/child/write")

    monkeypatch.setattr(instrument, "require_scratch_home", effect)
    monkeypatch.setattr(instrument, "guard_out", effect)
    monkeypatch.setattr(instrument, "check_set", effect)
    monkeypatch.setattr(instrument, "_child", effect)
    with pytest.raises(RuntimeError, match="retired live sleep family"):
        instrument.run_bout(family, "absent", 0)
    with pytest.raises(RuntimeError, match="retired live sleep family"):
        instrument.run_arm(str(tmp_path / "uncreated"), family, "absent", 0)
    with pytest.raises(RuntimeError, match="retired live sleep family"):
        instrument.run_family(str(tmp_path / "uncreated"), family, "absent")
    assert not (tmp_path / "uncreated").exists()


def test_retained_wake_and_explicit_replay_primitive():
    from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    learner = SpikingInteractionLearner(["A", "B"], n_in=8, seed=0)
    learner.stack = PlasticityRuleStack(n_neurons=2, seed=0)
    assert learner.replay(k=4) == 0
    learner.interact([0.9, 0.1, 0.0, 0.2, 0.5, 0.0, 0.3, 0.1], "A")
    learner.interact([0.1, 0.8, 0.4, 0.0, 0.0, 0.6, 0.0, 0.2], "B")
    assert len(learner._replay) == 2
    before = learner.stack.consolidation_calls()
    assert before > 0
    assert learner.replay(k=4) == 4
    assert len(learner._replay) == 2
    assert learner.stack.consolidation_calls() > before


def test_retired_observation_schema_refuses_without_migration():
    from snn_data_lake.engine.brain import Brain

    old = {
        "_sleep_replayed": 0,
        "_sleep_consolidate_calls": 0,
        "_plastic_relaxed": 0,
        "_structural_report": None,
    }
    with pytest.raises(ValueError, match="structural-only sleep observables"):
        Brain._r2_validate_observations(old)
    assert Brain._r2_validate_observations({"_structural_report": None}) == {
        "_structural_report": None
    }
