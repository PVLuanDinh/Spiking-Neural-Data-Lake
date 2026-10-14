"""G6 A14/C26: registered retention splits, full state resets and judging refusal."""

import pytest

from snn_data_lake.instruments import taught_problems as tp
from snn_data_lake.instruments.joint_mask_search import (
    CoverFixture,
    Meter,
    RetentionSets,
    SelectionLeak,
)
from snn_data_lake.instruments.taught_campaign import Reader


def test_registered_problem_splits_and_real_incumbent_learning_off():
    problem, seed = tp.PROBLEMS["c1d"], 9900
    sets = RetentionSets.for_problem(problem, seed)
    select = tuple(tuple(e) for e in problem.episodes(seed, "validation"))
    final = tuple(tuple(e) for e in problem.episodes(seed, "future"))
    assert sets._select == select and sets._final == final
    assert {tuple(t.text for t in e) for e in select}.isdisjoint(
        tuple(t.text for t in e) for e in final
    )
    learner = tp.canonical_incumbent(seed)
    reader = Reader(learner)
    state = learner._snapshot()
    kwargs = dict(
        snapshot=learner._snapshot, restore=learner._restore_snapshot, read_off=reader.route
    )
    a = sets.read_select(**kwargs)
    assert len(a) == len(select) and learner._snapshot() == state
    meter = Meter(CoverFixture())
    with pytest.raises(SelectionLeak, match="before the winner"):
        sets.read_final(meter, **kwargs)
    meter.freeze((True,) * meter.p.n)
    b = sets.read_final(meter, **kwargs)
    assert len(b) == len(final) and learner._snapshot() == state
    with pytest.raises(SelectionLeak, match="twice"):
        sets.read_final(meter, **kwargs)


def test_legacy_parity_and_stateful_set_reset_even_on_failed_judgment():
    # Constructed stateful consumer makes reset omission observable.
    sets = RetentionSets.for_legacy((0, 1, 2, 3, 4, 5))
    state = [7]
    restores, reads = [], []

    def snapshot():
        return tuple(state)

    def restore(saved):
        restores.append(saved)
        state[:] = saved

    def read_off(item):
        reads.append((item, state[0]))
        state[0] += 1
        return item

    kwargs = dict(snapshot=snapshot, restore=restore, read_off=read_off)
    assert sets.read_select(**kwargs) == (0, 2, 4)
    assert state == [7] and restores == [(7,), (7,)]
    meter = Meter(CoverFixture())
    meter.freeze((True,) * meter.p.n)

    def fail(item):
        read_off(item)
        raise RuntimeError("partial final read")

    with pytest.raises(RuntimeError, match="partial final"):
        sets.read_final(meter, snapshot=snapshot, restore=restore, read_off=fail)
    assert reads[-1] == (1, 7) and state == [7]
    with pytest.raises(SelectionLeak, match="twice"):
        sets.read_final(meter, **kwargs)


def test_empty_retention_partition_refuses():
    with pytest.raises(ValueError, match="nonempty"):
        RetentionSets.for_legacy((1,))
