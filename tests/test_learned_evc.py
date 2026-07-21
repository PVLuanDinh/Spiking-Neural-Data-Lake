"""Tests for the learned expected-value-of-control unit (cortex/learned_evc.py)."""

import numpy as np

from snn_data_lake.cortex.learned_evc import LearnedEVC


def _run_regime(evc, engaged_reward, withheld_reward, n):
    """Alternate a control-withheld default outcome with a control-engaged outcome."""
    for _ in range(n):
        evc.learn(engaged=False, reward=withheld_reward)
        evc.learn(engaged=True, reward=engaged_reward)


def test_allocate_shape_and_bounds():
    evc = LearnedEVC(seed=0)
    g = evc.allocate(0.8, 0.6)
    assert isinstance(g, float)
    assert np.isfinite(g)
    assert 0.0 <= g <= 1.0


def test_allocate_rises_with_conflict_times_stakes():
    evc = LearnedEVC(seed=0)
    ramp = [evc.allocate(c, 1.0) for c in (0.3, 0.5, 0.7, 0.9)]
    assert ramp == sorted(ramp), f"allocation must be monotone in conflict*stakes ({ramp})"
    assert ramp[-1] > ramp[0], "higher conflict*stakes must allocate more control"


def test_control_withheld_when_not_worth_it():
    evc = LearnedEVC(seed=0)
    # benefit = value*conflict*stakes << learned cost -> net <= 0 -> control withheld (~0)
    assert evc.allocate(0.02, 0.02) < 0.05


def test_paying_off_control_raises_value_and_allocation():
    evc = LearnedEVC(seed=0)
    pre = evc.allocate(0.8, 0.6)
    value0, cost0 = evc.value, evc.cost
    _run_regime(evc, engaged_reward=1.0, withheld_reward=0.0, n=40)
    post = evc.allocate(0.8, 0.6)
    assert evc.value > value0, "engaging control that pays off must raise the value-of-control"
    assert evc.cost <= cost0, "paying-off control must not raise the cost-of-control"
    assert post > pre, "paying-off control must raise allocation for the high-conflict probe"


def test_wasteful_control_raises_cost_and_lowers_allocation():
    evc = LearnedEVC(seed=0)
    pre = evc.allocate(0.8, 0.6)
    value0, cost0 = evc.value, evc.cost
    _run_regime(evc, engaged_reward=0.0, withheld_reward=1.0, n=40)
    post = evc.allocate(0.8, 0.6)
    assert evc.cost > cost0, "wasteful engaged control must raise the cost-of-control"
    assert evc.value < value0, "wasteful engaged control must lower the value-of-control"
    assert post < pre, "wasteful control must lower allocation for the high-conflict probe"


def test_opposite_regimes_diverge():
    pay = LearnedEVC(seed=0)
    waste = LearnedEVC(seed=0)
    _run_regime(pay, 1.0, 0.0, n=40)
    _run_regime(waste, 0.0, 1.0, n=40)
    assert pay.allocate(0.8, 0.6) > waste.allocate(0.8, 0.6)


def test_withheld_trials_do_not_move_weights():
    # eligibility gate: with control never engaged, the learned weights are frozen (three-factor).
    evc = LearnedEVC(seed=0)
    value0, cost0 = evc.value, evc.cost
    for _ in range(50):
        evc.learn(engaged=False, reward=1.0)
    assert evc.value == value0, "withheld (non-eligible) trials must not change the value"
    assert evc.cost == cost0, "withheld (non-eligible) trials must not change the cost"


def test_learn_returns_rpe_that_shrinks():
    evc = LearnedEVC(seed=0)
    first = evc.learn(engaged=True, reward=1.0)  # unpredicted -> big surprise
    for _ in range(30):
        evc.learn(engaged=False, reward=1.0)  # baseline catches up to the default
        evc.learn(engaged=True, reward=1.0)
    last = evc.learn(engaged=True, reward=1.0)
    assert first > 0.5, "the first unpredicted reward must be a large positive RPE"
    assert abs(last) < first, "the RPE must shrink as the value is learned"


def test_state_exposes_learned_cost_and_value():
    evc = LearnedEVC(seed=0)
    st = evc.state()
    assert "cost" in st and "value" in st
    assert st["cost"] == round(evc.cost, 4)
    assert st["value"] == round(evc.value, 4)


def test_determinism_same_seed():
    a = LearnedEVC(seed=42)
    b = LearnedEVC(seed=42)
    _run_regime(a, 1.0, 0.0, n=15)
    _run_regime(b, 1.0, 0.0, n=15)
    assert a.allocate(0.8, 0.6) == b.allocate(0.8, 0.6)
    assert a.signature() == b.signature()


def test_different_seed_signature_reproducible():
    a = LearnedEVC(seed=1)
    b = LearnedEVC(seed=1)
    assert a.signature() == b.signature()


def test_selfcheck_main_runs():
    import snn_data_lake.cortex.learned_evc as mod

    mod.main()  # asserts internally; must not raise
