"""Tests for TaskSetLayer (faculties/task_set.py): active-set persistence with a single task set,
R-STDP-gated shifting under sustained conflict/low reward, and the switch-cost transient (a drop
in the bias() mapping readout right after a shift, which recovers as the new set re-settles).

Run: pytest tests/test_task_set.py
"""

from snn_data_lake.faculties.task_set import TaskSetLayer


def test_single_task_set_is_held_despite_relentless_bad_signals():
    layer = TaskSetLayer(n_sets=1, n_in=3, n_actions=2, seed=0)
    for _ in range(20):
        layer.step()
        layer.observe(conflict=0.95, reward=-1.0)  # relentlessly bad, but nothing to shift to
    assert layer.active_set == 0


def test_sustained_conflict_and_low_reward_triggers_a_shift():
    layer = TaskSetLayer(n_sets=3, n_in=3, n_actions=2, seed=1, patience=5)
    start = layer.active_set
    shifted = False
    for _ in range(30):
        layer.step()
        layer.observe(conflict=0.9, reward=-0.6)
        if layer.active_set != start:
            shifted = True
            break
    assert shifted


def test_switch_cost_drops_then_recovers():
    layer = TaskSetLayer(n_sets=3, n_in=3, n_actions=2, seed=2, patience=4)
    start = layer.active_set
    x = [0.6, -0.2, 0.4]
    pre_mag = 0.0
    for _ in range(30):
        layer.step()
        if layer.active_set == start:
            pre_mag = sum(abs(v) for v in layer.bias(x))
        layer.observe(conflict=0.9, reward=-0.6)
        if layer.active_set != start:
            break
    post_mag = sum(abs(v) for v in layer.bias(x))
    for _ in range(25):
        layer.step()
    recovered_mag = sum(abs(v) for v in layer.bias(x))
    assert layer.last_switch_cost > 0.0
    assert post_mag < pre_mag
    assert recovered_mag > post_mag
