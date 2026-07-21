"""Tests for hippocampus.episode_consolidation.EpisodeConsolidator.

Covers the public API (encode/store/replay/recall/retained), non-catastrophic forgetting
at capacity, deterministic replay, capacity-ring eviction, and the negative control that
shows the balanced replay schedule (not the noise) is what preserves memory.
"""

import numpy as np

from snn_data_lake.hippocampus.episode_consolidation import (
    EpisodeConsolidator,
    _demo_episodes,
)


def _build(n_store, dim=256, capacity=64, seed=0):
    ec = EpisodeConsolidator(dim=dim, capacity=capacity, seed=seed)
    episodes = _demo_episodes(n_store)
    patterns = [ec.encode(e) for e in episodes]
    for p in patterns:
        ec.store(p)
    return ec, episodes, patterns


def _recovers(ec, code, cue_frac=0.5, thresh=0.9, noise=0):
    cue = ec._partial_cue(code, cue_frac, noise)
    out = frozenset(np.flatnonzero(ec.recall(cue)).tolist())
    return len(out & code) / len(code) >= thresh


def test_encode_is_sparse_binary_assembly():
    ec = EpisodeConsolidator(dim=256, capacity=64, seed=0)
    p = ec.encode({"who": "self", "what": "reach", "where": "home"})
    assert p.shape == (256,)
    assert set(np.unique(p)).issubset({0.0, 1.0})
    assert int(p.sum()) == ec.K == 20


def test_encode_content_addressable_and_separating():
    ec = EpisodeConsolidator(dim=256, capacity=64, seed=0)
    e = {"who": "self", "what": "grasp", "where": "left", "when": "3", "affect": "calm"}
    # same episode -> identical assembly; key order in the dict does not matter
    reordered = {k: e[k] for k in reversed(list(e))}
    assert np.array_equal(ec.encode(e), ec.encode(reordered))
    # distinct episodes -> distinct assemblies, and low mutual overlap (sparse coding)
    codes = [frozenset(np.flatnonzero(ec.encode(x)).tolist()) for x in _demo_episodes(64)]
    assert len(set(codes)) == 64
    overlaps = [
        len(codes[i] & codes[j]) for i in range(len(codes)) for j in range(i + 1, len(codes))
    ]
    assert max(overlaps) < ec.K  # never a full collision
    assert np.mean(overlaps) < 0.35 * ec.K  # assemblies are near-orthogonal


def test_encode_depends_on_every_slot():
    ec = EpisodeConsolidator(dim=256, capacity=64, seed=0)
    base = {"who": "self", "what": "reach", "where": "home", "when": "7", "affect": "calm"}
    p0 = frozenset(np.flatnonzero(ec.encode(base)).tolist())
    for slot, new in [
        ("who", "other"),
        ("what", "grasp"),
        ("where", "shelf"),
        ("when", "8"),
        ("affect", "fear"),
    ]:
        changed = dict(base)
        changed[slot] = new
        pc = frozenset(np.flatnonzero(ec.encode(changed)).tolist())
        assert pc != p0, f"changing slot {slot} must change the assembly"


def test_completion_from_partial_cue_at_capacity():
    ec, _, _ = _build(64)
    assert ec.retained(cue_frac=0.5, noise=0) == 1.0
    assert all(_recovers(ec, c) for c in ec.codes)


def test_completion_robust_to_spurious_spikes():
    ec, _, _ = _build(64)
    assert ec.retained(cue_frac=0.5, noise=5) >= 0.95


def test_no_catastrophic_forgetting_after_later_stores_and_replay():
    ec = EpisodeConsolidator(dim=256, capacity=64, seed=0)
    patterns = [ec.encode(e) for e in _demo_episodes(64)]
    early_codes = [frozenset(np.flatnonzero(p).tolist()) for p in patterns[:8]]
    for p in patterns[:8]:
        ec.store(p)
    assert all(_recovers(ec, c) for c in early_codes)
    for p in patterns[8:]:  # 56 later episodes stored on top of the early ones
        ec.store(p)
    ec.replay(steps=100)
    # the FIRST-stored episodes still complete after all later stores + a replay pass
    assert all(_recovers(ec, c) for c in early_codes)
    assert ec.retained() == 1.0


def test_replay_is_non_destructive_and_strengthens_coupling():
    ec, _, _ = _build(64)
    before = ec.retained(cue_frac=0.5, noise=0)
    coupling_before = ec.mean_assembly_coupling()
    n = ec.replay(steps=100)
    after = ec.retained(cue_frac=0.5, noise=0)
    coupling_after = ec.mean_assembly_coupling()
    assert n > 0
    assert after >= before  # consolidation never degrades recall
    assert coupling_after > coupling_before  # intra-assembly coupling is strengthened


def test_replay_is_deterministic():
    a, _, _ = _build(64, seed=0)
    b, _, _ = _build(64, seed=0)
    a.replay(steps=100)
    b.replay(steps=100)
    assert np.array_equal(a.W, b.W)
    assert a.retained() == b.retained()
    assert a.mean_assembly_coupling() == b.mean_assembly_coupling()


def test_different_seed_changes_replay_trajectory():
    a, _, _ = _build(64, seed=0)
    b, _, _ = _build(64, seed=1)
    # storage is seed-independent (same encoded patterns) -> identical weights before replay
    assert np.array_equal(a.W, b.W)
    a.replay(steps=100)
    b.replay(steps=100)
    # replay noise differs by seed -> the consolidated weights diverge
    assert not np.array_equal(a.W, b.W)


def test_capacity_ring_evicts_oldest():
    ec = EpisodeConsolidator(dim=256, capacity=32, seed=0)
    patterns = [ec.encode(e) for e in _demo_episodes(48)]
    for p in patterns:
        ec.store(p)
    assert len(ec.codes) == 32  # capped at capacity
    ec.replay(steps=60)
    # the 32 resident (most recent) episodes all still complete
    assert ec.retained() == 1.0


def test_unbalanced_replay_control_collapses_retention():
    good, _, _ = _build(64, seed=0)
    good.replay(steps=100)
    bad, _, _ = _build(64, seed=0)
    bad._replay_unbalanced_control(steps=100)
    good_ret = good.retained(cue_frac=0.5, noise=0)
    bad_ret = bad.retained(cue_frac=0.5, noise=0)
    assert good_ret == 1.0
    assert bad_ret <= good_ret - 0.15  # the schedule, not the noise, keeps memory


def test_recall_empty_memory_returns_zeros():
    ec = EpisodeConsolidator(dim=64, capacity=8, seed=0)
    out = ec.recall(np.zeros(64))
    assert out.shape == (64,)
    assert out.sum() == 0.0
    assert ec.retained() == 1.0  # vacuously retained when nothing is stored
