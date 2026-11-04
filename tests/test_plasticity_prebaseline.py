"""Cold baseline ordering and warm continuation, without constructing a Brain or lake."""

import copy
import json

import pytest

from snn_data_lake.plasticity.benna_fusi import BennaFusiSynapse
from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner


def _learner():
    learner = SpikingInteractionLearner(["A", "B"], 4, seed=13, t_post=4)
    learner.stack = PlasticityRuleStack(2, seed=13)
    return learner


def _teach(learner, n):
    for i in range(n):
        stimulus = [float((i + k) % 4) for k in range(4)]
        learner.interact(stimulus, learner.responses[i % 2])


def test_first_deposit_reaches_fast_beaker_before_diffusion(monkeypatch):
    stack = PlasticityRuleStack(1, seed=13)
    assert all(
        rule is not None
        for rule in (stack._short, stack._homeo, stack._intr, stack._meta, stack._ei, stack._cons)
    )

    weights, _ = stack.learning_step(0, [0.3], [1.0], lambda gain, w: [w[0] + 0.06])

    # One fast deposit, then the existing g0=.1 diffusion into capacity-two level 1.
    assert stack._cons._cas[0][0].u == pytest.approx([0.354, 0.303, 0.3, 0.3, 0.3, 0.3])
    assert weights == pytest.approx([0.354])
    assert stack._homeo._target[0] == pytest.approx(0.9)

    # Removing the PRE boundary reproduces the old defect on the same enabled stack.
    with monkeypatch.context() as patch:
        patch.setattr(PlasticityRuleStack, "prepare_update", lambda *args: None)
        red = PlasticityRuleStack(1, seed=13)
        red.learning_step(0, [0.3], [1.0], lambda gain, w: [w[0] + 0.06])
    assert red._cons._cas[0][0].u == [0.36] * 6
    assert red._homeo._target[0] == pytest.approx(1.08)


def test_actual_rstdp_deposit_uses_preupdate_weights_and_activity():
    learner = _learner()
    row = learner.neurons[0]
    row.w = [0.3] * learner.n_in
    code = [0, 1, 2, 3]
    initial = list(row.w)
    activity = [learner._psp(learner.t_post - t) for t in code]

    deltas = learner._deposit(0, code, 1.0)

    assert learner.stack._homeo._target[0] == pytest.approx(
        3.0 * sum(w * a for w, a in zip(initial, activity, strict=True))
    )
    for k, (before, delta) in enumerate(zip(initial, deltas, strict=True)):
        expected = BennaFusiSynapse(n_levels=6, w_max=4.0)
        expected.u = [before] * 6
        expected.deposit(delta)
        expected.diffuse()
        actual = learner.stack._cons._cas[0][k]
        assert actual.u == expected.u
        assert row.w[k] == expected.weight
        assert actual.u[-1] == before
    assert learner.stack._cons._cas[1] is None
    assert learner.stack._homeo._target[1] is None


def test_warm_continuation_is_identical_to_the_old_post_only_path(monkeypatch):
    prepared = _learner()
    _teach(prepared, 12)
    legacy = copy.deepcopy(prepared)
    initial_fields = set(vars(prepared.stack))

    for i in range(32):
        stimulus = [float((i + k) % 4) for k in range(4)]
        label = prepared.responses[i % 2]
        prepared.interact(stimulus, label)
        with monkeypatch.context() as patch:
            patch.setattr(PlasticityRuleStack, "prepare_update", lambda *args: None)
            legacy.interact(stimulus, label)
        assert prepared._snapshot() == legacy._snapshot()

    assert set(vars(prepared.stack)) == initial_fields


@pytest.mark.parametrize("warm", [False, True])
def test_snapshot_restore_continues_cold_or_warm_baselines_exactly(warm):
    learner = _learner()
    if warm:
        _teach(learner, 12)
    saved = json.loads(json.dumps(learner._snapshot()))
    restored = _learner()
    restored._restore_snapshot(saved)
    assert learner._snapshot() == restored._snapshot()

    _teach(learner, 32)
    _teach(restored, 32)
    assert learner._snapshot() == restored._snapshot()


def test_standalone_stabilize_keeps_its_lazy_fallback():
    stack = PlasticityRuleStack(1, seed=13)
    stack.stabilize(0, [0.36], [1.0])
    assert stack._homeo._target[0] == pytest.approx(1.08)
    assert stack._cons._cas[0][0].u == [0.36] * 6


def test_legacy_stack_without_prebaseline_api_still_receives_deposits():
    class LegacyStack:
        def pre_gain(self, neuron_id):
            return 1.0

        def stabilize(self, neuron_id, weights, activity):
            return weights

    learner = _learner()
    learner.stack = LegacyStack()
    before = learner.weights()
    learner._deposit(0, [0, 1, 2, 3], 1.0)
    assert learner.weights()[0] != before[0]
    assert learner.weights()[1] == before[1]
