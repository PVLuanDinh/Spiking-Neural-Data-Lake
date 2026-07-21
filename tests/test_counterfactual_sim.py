"""Tests for cortex/counterfactual_sim.py -- brain-native counterfactual imagination.

Verifies the learned forward model reduces imagination error with local training, the do()-operator
ranks the target-reaching action first, determinism, and (via a fresh subprocess) that importing the
module pulls in neither numpy nor torch (it must stay in the zero-dependency core)."""

import os
import subprocess
import sys

from snn_data_lake.cortex.counterfactual_sim import CounterfactualSim, _toy_experiment


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


def test_import_has_no_numpy_or_torch():
    # Fresh interpreter: importing the module must not drag numpy/torch into sys.modules.
    env = dict(os.environ)
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; import snn_data_lake.cortex.counterfactual_sim as m; "
        "assert 'numpy' not in sys.modules, 'numpy leaked'; "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
