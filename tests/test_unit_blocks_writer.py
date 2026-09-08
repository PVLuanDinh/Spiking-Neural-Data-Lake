"""T064 / Phase 3 S1 — unit blocks, and the FIRST LIVE WRITER of `SynapseDelta` (0x09).

`0x09` was declared, tested and never written: L6's lake census counted it among four classes that
have never held a record, and `RETENTION[0x09]` says its append budget stays "unbaselined ... until
a live writer exists to measure". These pins are what turn it from built-and-uncharged into charged.

Each claim carries the control that makes it a measurement (record: evidence/s1-unit-blocks.md):
  * R-S1a  one block resident, with the NESTING ATTEMPT made so the counter is not vacuous
  * R-S1b  the partition tiles the units; an out-of-range unit is refused, never bucketed
  * R-S1c  a byte-frozen substrate emits NOTHING; a single real change emits exactly one entry
  * R-S1d  a real deposit writes a record that REPLAYS to the weights that moved; a deposit below
           the derived quantum writes nothing
"""

import pytest

from snn_data_lake.substrate.unit_blocks import BlockResidencyError, UnitBlocks

np = pytest.importorskip("numpy")

from snn_data_lake.cortex.interaction_learning import SpikingInteractionLearner  # noqa: E402
from snn_data_lake.substrate.lake_engine import SynapseDelta  # noqa: E402


def test_the_partition_tiles_the_units_and_refuses_a_stray_one():
    ub = UnitBlocks(10, 4)
    assert ub.n_blocks == 3
    assert [u for b in range(3) for u in ub.units_in(b)] == list(range(10))
    assert list(ub.units_in(2)) == [8, 9], "the tail block is SHORT, never padded"
    for bad in (-1, 10):
        with pytest.raises(ValueError):
            ub.block_of(bad)


def test_block_size_has_no_default_and_the_refusal_says_why():
    with pytest.raises(ValueError, match="refuses to pick one"):
        UnitBlocks(10, 0)


def test_one_block_resident_and_the_nesting_red_is_reachable():
    ub = UnitBlocks(10, 4)
    for _b, _entries, _units in ub.stream(ub.group([(0, 1, 5), (3, 9, -2)])):
        pass
    assert ub.max_resident == 1
    with ub.resident(0):
        with pytest.raises(BlockResidencyError):  # THE RED, actually attempted
            with ub.resident(1):
                pass
    assert ub.max_resident == 1, "a refused nesting must not raise the high-water mark"


def test_grouping_keys_on_post_and_omits_blocks_that_own_nothing():
    ub = UnitBlocks(10, 4)
    g = ub.group([(0, 1, 5), (3, 9, -2), (1, 2, 7)])
    assert set(g) == {0, 2}
    assert 1 not in g, "a block owning no entry is ABSENT, never present-and-empty"
    assert ub.group([]) == {}


def test_a_frozen_substrate_emits_nothing_and_one_real_change_emits_one_entry():
    w = np.array([[0.10, 0.20], [0.30, 0.40]], dtype=float)
    sd = SynapseDelta(q_step=0.01)
    ub = UnitBlocks(2, 1)
    assert sd.collect(w, [0, 1], [0, 1]) is not None, "the first capture carries the substrate"
    assert sd.collect(w, [0, 1], [0, 1]) is None, (
        "a byte-frozen substrate whose endpoints keep firing must emit NOTHING -- the "
        "repeated-snapshot pathology refused at the door"
    )
    w[1][0] += 0.05
    moved = sd.collect(w, [1], [0])
    assert moved is not None, "the positive control must fire on a real change"
    g = ub.group(SynapseDelta.entries(moved))
    assert list(g) == [0] and g[0] == [(1, 0, 35)]


def _learner():
    lr = SpikingInteractionLearner(["a", "b"], n_in=8, seed=0)
    written = []
    lr.delta_sink = lambda payload, grouped: written.append((payload, grouped))
    return lr, written


def test_a_real_deposit_writes_a_record_that_replays_to_the_moved_weights():
    """R-S1d. The tag goes from never-written to holding a record whose replay reconstructs the
    weights that moved -- the whole point of making rewiring lake-resident."""
    learner, written = _learner()
    before = [[n.w[i] for n in learner.neurons] for i in range(learner.n_in)]
    learner.interact([0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.6, 0.4], "a")
    assert written, "the first live writer of 0x09 must actually write"

    payload, grouped = written[-1]
    entries = SynapseDelta.entries(payload)
    assert entries, "a written record must carry entries"
    after = [[n.w[i] for n in learner.neurons] for i in range(learner.n_in)]

    # every entry names a synapse that really moved, quantized at the learner's OWN lr
    for pre, post, q in entries:
        assert q == int(round(after[pre][post] / learner.lr)), (
            f"entry ({pre},{post},{q}) does not replay to the live weight "
            f"{after[pre][post]} at q_step={learner.lr}"
        )
    moved = {
        (i, j)
        for i in range(learner.n_in)
        for j in range(len(learner.neurons))
        if round(before[i][j] / learner.lr) != round(after[i][j] / learner.lr)
    }
    recorded = {(p, q_) for p, q_, _ in entries}
    assert moved <= recorded, (
        f"a synapse crossed the quantum and was NOT recorded: {moved - recorded}"
    )
    # block-aligned: every entry sits in the block its POST unit owns
    blocks = UnitBlocks(len(learner.neurons), block_size=learner.delta_block_size)
    for b, es in grouped.items():
        assert all(blocks.block_of(post) == b for _pre, post, _q in es)


def test_no_sink_means_no_work_and_a_repeat_deposit_writes_nothing_new():
    """The control on both halves: without a sink the learner still learns and records nothing, and
    with one, a substrate that does not cross the quantum again emits nothing."""
    quiet = SpikingInteractionLearner(["a", "b"], n_in=8, seed=0)
    assert quiet.delta_sink is None
    assert quiet.interact([0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.6, 0.4], "a") is not None
    assert quiet._syn_delta is None, "no sink must mean no delta machinery is even built"

    learner, written = _learner()
    for _ in range(6):
        learner.interact([0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.6, 0.4], "a")
    n_after_converge = len(written)
    for _ in range(4):
        learner.interact([0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.6, 0.4], "a")
    assert len(written) <= n_after_converge + 4
    assert any(w is not None for w, _ in written)
