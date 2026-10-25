"""P4b: a restart of the lifelong consolidator is not amnesia (arc-2 panel A-B1 / B MAJOR-3).

The restore brings back the learner, labels, utilities and task index, and the replay memory is
rebuilt from the task streams it was filled from. The probe is rv-arc2-a's `ll_restart.py`: learn
task 0, round-trip the state through JSON (the lake), continue in a fresh consolidator. Without the
replay memory the old units decay below the budget and are pruned (task-0 retention 0.000 on seeds
0-4); with it the restarted run is the uninterrupted run.
"""

import json

import pytest

from snn_data_lake.sleep.lifelong_consolidation import D, LifelongConsolidator, _make_tasks

SEEDS = range(5)


def _restarted(seed, tasks, rebuild):
    b = LifelongConsolidator(dim=D, budget=6, seed=seed)  # the Brain's knobs
    b.learn_task(0, tasks[0]["train"])
    c = LifelongConsolidator(dim=D, budget=6, seed=seed)  # the restarted process
    assert c.load_state_dict(json.loads(json.dumps(b.state_dict())))
    if rebuild:
        c.rebuild_memory([t["train"] for t in tasks[:1]])
    for i in range(1, len(tasks)):
        c.learn_task(i, tasks[i]["train"])
    return c


def _uninterrupted(seed, tasks):
    a = LifelongConsolidator(dim=D, budget=6, seed=seed)
    for i, t in enumerate(tasks):
        a.learn_task(i, t["train"])
    return a


@pytest.mark.parametrize("seed", SEEDS)
def test_the_probe_sees_the_amnesia_without_the_replay_memory(seed):
    """Known positive for the instrument: a restart with an empty replay memory loses task 0."""
    tasks = _make_tasks(seed=seed)
    a, c = _uninterrupted(seed, tasks), _restarted(seed, tasks, rebuild=False)
    assert a.accuracy(tasks[0]["test"]) > c.accuracy(tasks[0]["test"])


@pytest.mark.parametrize("seed", SEEDS)
def test_a_restart_with_the_rebuilt_memory_is_the_uninterrupted_run(seed):
    tasks = _make_tasks(seed=seed)
    a, c = _uninterrupted(seed, tasks), _restarted(seed, tasks, rebuild=True)
    assert c.accuracy(tasks[0]["test"]) == a.accuracy(tasks[0]["test"])
    assert c.state_dict() == a.state_dict()
    assert c.memory.buf == a.memory.buf and c.memory.rng.getstate() == a.memory.rng.getstate()


def test_the_rebuilt_memory_is_the_one_learn_task_built():
    tasks = _make_tasks(seed=3)
    a = LifelongConsolidator(dim=D, budget=6, seed=3)
    for i in range(2):
        a.learn_task(i, tasks[i]["train"])
    r = LifelongConsolidator(dim=D, budget=6, seed=3)
    r.rebuild_memory([t["train"] for t in tasks[:2]])
    assert r.memory.buf == a.memory.buf and r.memory.rng.getstate() == a.memory.rng.getstate()
    assert r.n_units == 0, "rebuilding the memory learns nothing"
