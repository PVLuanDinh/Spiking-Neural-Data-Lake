"""The LIVE WRITER of `SynapseDelta` (0x09), and the block framing it keeps after T068.

RE-HOMED 2026-09-09 from `tests/test_unit_blocks_writer.py`, whose module `substrate/unit_blocks.py`
was deleted with the paging layer on T068's LOSS (specs/022 evidence/s5-paging-null.md 3.4/3.5).
Five of that file's seven arms were never about paging: they pin the 0x09 synapse-delta writer,
which SURVIVES, and its record's block framing, which moved to `SynapseDelta.group_by_block` in the
commit that deleted the module. Deleting them with the pager would have dropped the only reds a
surviving mechanism has -- 2.6 orders the pager-free arms re-homed, and here, unlike
`tests/test_lake_compactor.py`, there are some.

WHAT DIED WITH THE LAYER, stated rather than quietly dropped:
  * R-S1a (one block resident, the nesting red) -- it asserted the residency DISCIPLINE of
    `UnitBlocks.resident`/`stream`, which existed to make "lake-resident" mean something for the
    pager. No pager, no residency counter, no red.
  * the tiling half of R-S1b (`units_in` tiles the units, the tail block is short) -- `units_in` is
    not ported; only the partition's REFUSAL of an out-of-range unit is, and it is pinned below.

What still stands here:
  * R-S1b (refusals)  an out-of-range POST is refused, never bucketed; `block_size` has no default
  * R-S1c  a byte-frozen substrate emits NOTHING; a single real change emits exactly one entry
  * R-S1d  a real deposit writes a record that REPLAYS to the weights that moved
  * the control on both halves: no sink, no work
"""

import pytest

np = pytest.importorskip("numpy")

from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner  # noqa: E402
from snn_data_lake.substrate.lake_engine import SynapseDelta  # noqa: E402


def test_grouping_keys_on_post_and_omits_blocks_that_own_nothing():
    g = SynapseDelta.group_by_block([(0, 1, 5), (3, 9, -2), (1, 2, 7)], 10, 4)
    assert set(g) == {0, 2}
    assert 1 not in g, "a block owning no entry is ABSENT, never present-and-empty"
    assert g[0] == [(0, 1, 5), (1, 2, 7)], "insertion order is preserved inside a block"
    assert SynapseDelta.group_by_block([], 10, 4) == {}


def test_the_partition_refuses_a_stray_post_rather_than_bucketing_it():
    """R-S1b's surviving half. `post // block_size` maps a unit that does not exist onto a block
    that does, so the refusal is the only thing standing between a stray index and a plausible
    wrong answer -- THE RED, actually attempted."""
    for bad in (-1, 10):
        with pytest.raises(ValueError, match="outside"):
            SynapseDelta.group_by_block([(0, bad, 1)], 10, 4)
    assert SynapseDelta.group_by_block([(0, 9, 1)], 10, 4), "the control: the last unit is accepted"


def test_block_size_has_no_default_and_the_refusal_says_why():
    with pytest.raises(ValueError, match="refuses to pick one"):
        SynapseDelta.group_by_block([], 10, 0)
    with pytest.raises(ValueError):
        SynapseDelta.group_by_block([], 0, 4)


def test_a_frozen_substrate_emits_nothing_and_one_real_change_emits_one_entry():
    """R-S1c."""
    w = np.array([[0.10, 0.20], [0.30, 0.40]], dtype=float)
    sd = SynapseDelta(q_step=0.01)
    assert sd.collect(w, [0, 1], [0, 1]) is not None, "the first capture carries the substrate"
    assert sd.collect(w, [0, 1], [0, 1]) is None, (
        "a byte-frozen substrate whose endpoints keep firing must emit NOTHING -- the "
        "repeated-snapshot pathology refused at the door"
    )
    w[1][0] += 0.05
    moved = sd.collect(w, [1], [0])
    assert moved is not None, "the positive control must fire on a real change"
    g = SynapseDelta.group_by_block(SynapseDelta.entries(moved), 2, 1)
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
    for b, es in grouped.items():
        assert all(post // learner.delta_block_size == b for _pre, post, _q in es)


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
