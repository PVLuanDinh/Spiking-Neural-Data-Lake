"""Tests for faculties/counterfactual_sim.py -- brain-native counterfactual imagination.

Verifies the learned forward model reduces imagination error with local training, the do()-operator
ranks the target-reaching action first, and determinism.

(A fourth test asserted that importing the module pulled in neither numpy nor torch -- "it must
stay in the zero-dependency core". Removed 2026-07-29 with that rule; note at the bottom.)"""

from snn_data_lake.faculties.counterfactual_sim import CounterfactualSim, _toy_experiment


def test_learns_reduces_imagination_error():
    r = _toy_experiment(0)
    assert r["late"] < 0.1 * r["early"], (r["early"], r["late"])


def test_counterfactual_ranks_target_action_first():
    r = _toy_experiment(0)
    ranked_actions = [a for (a, _sc, _s) in r["ranked"]]
    assert ranked_actions[0] == r["best_action"]
    assert set(ranked_actions) == {0, 1, 2, 3}  # every candidate ranked exactly once


def test_determinism_same_seed():
    a = _toy_experiment(0)
    b = _toy_experiment(0)
    assert a["early"] == b["early"]
    assert a["late"] == b["late"]
    assert a["ranked"] == b["ranked"]
    assert a["fingerprint"] == b["fingerprint"]


def test_different_seed_changes_model():
    assert _toy_experiment(0)["fingerprint"] != _toy_experiment(1)["fingerprint"]


def test_imagine_shape_and_no_mutation():
    sim = CounterfactualSim(5, 3, seed=0)
    z = [0.1, -0.2, 0.3, 0.0, 0.4]
    z_copy = list(z)
    traj = sim.imagine(z, 1, 4)
    assert len(traj) == 4
    assert all(len(step) == 5 for step in traj)
    assert z == z_copy  # imagine() must not mutate its input


def test_persistence_prior_at_init():
    # Untrained model predicts "nothing changes" (persistence): z_next ~= z.
    sim = CounterfactualSim(4, 2, seed=3)
    z = [0.5, -0.5, 0.2, -0.1]
    pred = sim._predict(z, 0)
    assert all(abs(pred[i] - z[i]) < 0.5 for i in range(4))


def test_counterfactual_default_return_is_action_score_pairs():
    sim = CounterfactualSim(4, 3, seed=0)
    z = [0.0, 0.1, -0.1, 0.2]
    ranked = sim.counterfactual(z, [0, 1, 2], 2, lambda zi: -sum(v * v for v in zi))
    assert all(len(rec) == 2 for rec in ranked)  # (action, score) by default
    scores = [sc for (_a, sc) in ranked]
    assert scores == sorted(scores, reverse=True)  # best-first
    assert set(sim.last_surprise) == {0, 1, 2}  # surprise surfaced per action


def test_learn_reduces_single_transition_error():
    sim = CounterfactualSim(3, 2, seed=0)
    z, z_next = [0.2, -0.1, 0.3], [0.7, -0.1, 0.3]  # action 0 shifts dim 0 by +0.5
    first = sim.learn(z, 0, z_next)
    for _ in range(200):
        sim.learn(z, 0, z_next)
    last = sim.learn(z, 0, z_next)
    assert last < first
    assert last < 1e-4


# test_import_has_no_numpy_or_torch REMOVED 2026-07-29. It spawned a fresh interpreter and
# asserted importing counterfactual_sim left 'numpy' and 'torch' out of sys.modules -- the last
# EXECUTABLE enforcement of the zero-dependency rule in the tree (the 59 in-module purity asserts
# went on 2026-07-20; this one survived by living in tests/).
#
# What is genuinely lost, because it is not nothing: as a side effect it protected IMPORT-TIME
# HYGIENE for this module, and keeping a heavy dependency out of a light module is legitimate even
# when that dependency is core. But it asserted the retired rule, not that principle -- it named
# numpy and torch as forbidden rather than measuring import cost, and covered 1 module of 245.
# If import time becomes a real constraint, the honest instrument measures import time across the
# package with its own justification, not a purity assert outliving its rule.
