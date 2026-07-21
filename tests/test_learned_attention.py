"""Tests for the learned spiking attention gate (cortex/learned_attention.py)."""

import numpy as np

from snn_data_lake.cortex.learned_attention import LearnedAttention


def test_attend_shape_and_bounds():
    att = LearnedAttention(8, n_bands=4, seed=0)
    out = att.attend([0.2, 0.9, 0.1, 0.5, 0.3, 0.4, 0.6, 0.2])
    assert isinstance(out, list) and len(out) == 8
    assert all(np.isfinite(v) for v in out)
    # gate is a [0,1] per-band gain
    assert all(0.0 <= g <= 1.0 for g in att.gate)
    assert len(att.gate) == 4


def test_uniform_probe_reveals_gate():
    # with a ones input, the gated output equals the per-band gate
    att = LearnedAttention(8, n_bands=4, seed=0)
    out = att.attend([1.0] * 8)
    band_dim = 8 // 4
    for b in range(4):
        band_out = out[b * band_dim : (b + 1) * band_dim]
        assert all(abs(v - att.gate[b]) < 1e-9 for v in band_out)


def test_reward_amplifies_rewarded_band():
    rewarded, unrewarded = 0, 2
    att = LearnedAttention(8, n_bands=4, seed=0)
    probe = [1.0] * 8
    att.attend(probe)
    pre = att.gate[rewarded]
    train = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # rewarded band present
    for _ in range(60):
        att.attend(train)
        att.learn(1.0)
    att.attend(probe)
    post = att.gate
    assert post[rewarded] > pre, "reward should raise the rewarded band's gate"
    assert post[rewarded] > post[unrewarded], "rewarded band should out-gate the unrewarded band"


def test_negative_reward_suppresses_band():
    band = 0
    att = LearnedAttention(8, n_bands=4, seed=0)
    probe = [1.0] * 8
    att.attend(probe)
    pre = att.gate[band]
    present = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for _ in range(40):
        att.attend(present)
        att.learn(-1.0)  # punish attending this band
    att.attend(probe)
    assert att.gate[band] < pre, "negative reward should lower the punished band's gate"


def test_goal_biases_toward_similar_features():
    dim = 8
    feats = [1.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.1, 0.1]  # band0, band1 equally salient
    goal = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # matches band1's pattern
    a_no = LearnedAttention(dim, n_bands=4, seed=0)
    a_go = LearnedAttention(dim, n_bands=4, seed=0)
    a_no.attend(feats)
    a_go.attend(feats, goal=goal)
    assert a_go.gate[1] > a_no.gate[1], "goal should boost the goal-similar band"
    assert a_go.gate[1] > a_go.gate[0], "under the goal, band1 should out-gate band0"


def test_focus_reports_winner():
    att = LearnedAttention(8, n_bands=4, seed=0)
    att.attend([0.05, 0.05, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05])  # band1 loud
    band, strength = att.focus
    assert band == 1
    assert 0.0 <= strength <= 1.0


def test_determinism_same_seed():
    x = [0.3, 0.7, 0.2, 0.8, 0.1, 0.5, 0.4, 0.6]
    a = LearnedAttention(8, n_bands=4, seed=42).attend(x)
    b = LearnedAttention(8, n_bands=4, seed=42).attend(x)
    assert a == b


def test_different_seed_signature_reproducible():
    a = LearnedAttention(8, n_bands=4, seed=1)
    b = LearnedAttention(8, n_bands=4, seed=1)
    assert a.signature() == b.signature()


def test_selfcheck_main_runs():
    import snn_data_lake.cortex.learned_attention as mod

    mod.main()  # asserts internally; must not raise
