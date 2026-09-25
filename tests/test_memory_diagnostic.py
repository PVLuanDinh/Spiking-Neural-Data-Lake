"""Fast synthetic reds for the G1(a) causal contract, and G1(b)'s extensions of the harness: the
reference readout (g1b-pool-selection.md section 2.2), the pool arm, R-b2, R-b8's comparisons and
R-b6's synthetic family (section 2.7)."""

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from snn_data_lake.cortex import memory_diagnostic as md
from snn_data_lake.cortex import recurrent_pool as rp
from snn_data_lake.cortex.growth_stop import GrowthStop, feed_and_decide
from snn_data_lake.cortex.memory_diagnostic import (
    ARMS,
    ASSET_ENV,
    ASSETS,
    GROWER_KEYS,
    ReferenceReadout,
    _feature,
    _lp_sensitive_red,
    _policy,
    _synthetic_stream,
    _View,
    check_r_a1,
    check_r_a2,
    check_r_a2b,
    check_r_a3,
    check_r_a4,
    check_r_b0x_readout,
    check_r_b2,
    onehot_32,
    rb6_stream,
    record_stream,
    replay,
    score,
)
from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove


def test_onehot_and_forbidden_channel():
    assert onehot_32(31)[31] == 1
    with pytest.raises(ValueError):
        onehot_32(32)
    assert check_r_a2b({"events": []})["red"]
    with pytest.raises(ValueError):
        _feature("forbidden", _View(ARMS["forbidden"][1], {"x": [0.0] * 64, "c": 1}))


def test_scorer_uses_current_category():
    stream = {"events": [{"c": c, "segment": "unseen-order"} for c in (0, 0, 1, 1, 0, 1)]}
    oracle = score(stream, [e["c"] for e in stream["events"]], [False] * 6)
    shifted = score(stream, [None, 0, 0, 1, 1, 0], [False] * 6)
    assert oracle["unseen-order"]["accuracy"] == 1
    assert shifted["unseen-order"]["accuracy"] < 1
    assert check_r_a3(stream)["red"]


def test_shared_gate_preserves_poisoned_row_cursor_and_verdict():
    trajectory = [(1, 1, 0.5), (2, 2, float("nan")), (3, 3, 0.6)]
    old = GrowthStop()
    refused = 0
    for _, units, accuracy in trajectory[old.observed("cortex") :]:
        try:
            old.observe_refit("cortex", int(units), float(accuracy))
        except (ValueError, TypeError):
            refused += 1
    expected = old.should_grow("cortex")
    new = GrowthStop()
    verdict, actual_refused = feed_and_decide(new, trajectory)
    assert (verdict, actual_refused, new.observed("cortex")) == (
        expected,
        refused,
        old.observed("cortex"),
    )
    dropped = GrowthStop()
    feed_and_decide(dropped, trajectory[:-1])
    assert dropped.observed("cortex") != new.observed("cortex")


def test_predict_next_is_the_live_decision_and_consumed_once():
    engine = NeurogenesisSelfImprove(seed=3, device="cpu", refit_every=1)
    for i in range(30):
        engine.observe([float(i % 2)] * 64, i % 2)
    engine.tick()
    assert engine._model is not None
    model = engine._model
    original = model.decide
    calls = []

    def counted(rates, beta=None):
        calls.append(1)
        return original(rates, beta)

    model.decide = counted
    expected = int(model.predict(torch.tensor([engine._prev], dtype=torch.float32))[0])
    calls.clear()
    assert engine.predict_next() == expected
    assert engine.predict_next() == expected
    before = engine.live_correct
    engine.observe([0.25] * 64, expected)
    assert len(calls) == 1
    assert engine.live_correct == before + 1
    assert engine._pending_prediction is None


def test_feature_noise_changes_and_history_is_width_pinned():
    view = _View(("x",), {"x": [0.0] * 64})
    assert _feature("noise", view) != _feature("noise", view)
    for arm, (width, channels) in ARMS.items():
        if arm == "forbidden":
            continue
        values = {
            "x": [0.0] * 64,
            "c": 1,
            "own_predictions": None,
            "x_history": [[0.0] * 64] * 3,
            "c_history": [1, None, None, None],
        }
        # G1(b)'s pool arm reads its features from its pool (and refuses without one)
        pool = rp.make_pool("lsm", rp.R_B5_PERSISTENCE_WITNESS, 0) if arm == "pool" else None
        assert len(_feature(arm, _View(channels, values), pool)) == width


def test_replay_equivalence_mutations_and_ledger_reds():
    stream = _synthetic_stream()
    live = replay(stream, "a", "live")
    stream["final"] = {
        key: live[key]
        for key in (
            "live_n",
            "live_correct",
            "live_persist",
            "ticks",
            "observed",
            "trajectory",
            "fingerprint",
        )
    }
    stream["final"]["gate_verdicts"] = live["gate_verdicts"]
    stream["final"].update(live["revision"])
    completed = {(r["tick"], r["event"]): r["fingerprint"] for r in live["refits"]}
    stream["refits"] = [
        dict(row, refit=completed.get((row["tick"], row["event"])))
        for row in stream["sleep"]
        if row["tick"] % stream["config"]["refit_every"] == 0
    ]
    for event, prediction in zip(stream["events"], live["predictions"], strict=True):
        event["prediction"] = prediction
    assert check_r_a1(stream)["verdict"]
    assert check_r_a2(stream)["verdict"]
    assert check_r_a4(stream)["verdict"]
    checked = replay(stream, "a", "live")
    assert checked["config_digest"] and checked["revision"] and checked["execution_id"]
    assert checked["execution_id"] != live["execution_id"]
    incomplete = copy.deepcopy(checked)
    incomplete["predictions"] = []
    with pytest.raises(ValueError, match="incomplete replay events"):
        check_r_a1(stream, incomplete)
    wrong_revision = copy.deepcopy(checked)
    wrong_revision["revision"]["head"] = "other"
    with pytest.raises(ValueError, match="replay provenance"):
        check_r_a1(stream, wrong_revision)
    shifted_refit = copy.deepcopy(stream)
    shifted_refit["refits"][0]["event"] += 1
    assert not check_r_a1(shifted_refit, checked)["numbers"]["equivalent"]
    deleted = copy.deepcopy(stream)
    del deleted["events"][15]["causal_lp"]
    with pytest.raises(ValueError, match="LP ledger"):
        replay(stream, "a", mutate=deleted)
    disconnected = copy.deepcopy(stream)
    disconnected["_no_lp_reader"] = True
    with pytest.raises(ValueError, match="LP ledger"):
        replay(stream, "a", mutate=disconnected)


def test_lp_sensitive_gate_fixture_flips_verdict():
    rows = [(i, i, 0.5) for i in range(15)]
    no_motion = GrowthStop(lp_reader=type("Reader", (), {"lp": lambda self, region: 0.0})())
    values = iter([0.0] * 12 + [10.0] * 3)
    motion = GrowthStop(lp_reader=type("Reader", (), {"lp": lambda self, region: next(values)})())
    assert feed_and_decide(no_motion, rows)[0] is False
    assert feed_and_decide(motion, rows)[0] is True
    assert _lp_sensitive_red()
    assert _lp_sensitive_red("f64")
    assert _lp_sensitive_red("a", "causal")
    assert _lp_sensitive_red("f64", "causal")


@pytest.mark.slow
def test_real_120_tick_recording_r_a1():
    from snn_data_lake.cortex.memory_diagnostic import check_r_a1

    models = Path(__file__).resolve().parents[1] / "models"
    if any(not (models / name).is_file() for name in ASSETS):
        pytest.skip("real perception assets are absent")
    stream = record_stream(900, ticks=120)
    result = check_r_a1(copy.deepcopy(stream))
    assert result["numbers"]["equivalent"] and result["numbers"]["refits"] > 0
    assert not result["numbers"]["eligible"]


def test_view_hides_undeclared_category():
    view = _View(ARMS["a"][1], {"x": [0.0] * 64, "c": 7})
    assert "c" not in view.values
    assert not hasattr(view, "c")
    with pytest.raises(ValueError, match="forbidden"):
        view.get("c")


def test_loaded_live_run_is_refused_by_causal_check(tmp_path):
    import json

    stream = _synthetic_stream()
    paths = {}
    for arm in tuple(ARMS)[:8]:
        result = replay(stream, arm, "live" if arm == "a" else "causal")
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths[arm] = {"base": path}
    with pytest.raises(ValueError, match="replay identity"):
        check_r_a2(stream, paths)


def test_loaded_causal_runs_require_complete_rows_and_distinct_executions(tmp_path, monkeypatch):
    import json

    import snn_data_lake.cortex.memory_diagnostic as diagnostic

    monkeypatch.setattr(diagnostic, "SCORED_ARMS", ("a",))
    stream = _synthetic_stream()
    result = replay(stream, "a")
    stream["final"] = result["revision"]
    path = tmp_path / "run.json"
    missing = dict(result, predictions=[])
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete replay events"):
        check_r_a2(stream, {"a": {"base": path}})
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate replay execution"):
        check_r_a4(stream, {"a": (path, path)})


def test_policy_records_asset_environment(monkeypatch):
    for name in ASSET_ENV:
        monkeypatch.setenv(name, "override")
    policy = _policy()
    assert all(policy[name] == "override" for name in ASSET_ENV)
    with pytest.raises(ValueError, match="asset environment override"):
        record_stream(900, ticks=1)


def test_record_stream_refuses_history_feed_before_assets():
    with pytest.raises(ValueError, match="unsupported in-brain feed"):
        record_stream(900, ticks=1, arm_feed="d")


def test_calibration_red_survives_transition_free_real_segment():
    stream = {
        "events": [
            {"c": 1, "segment": "warm-up"},
            {"c": 1, "segment": "unseen-order"},
            {"c": 1, "segment": "unseen-order"},
        ]
    }
    result = check_r_a3(stream)
    assert result["verdict"] and result["red"]
    assert result["discrimination"]["unseen-order"] == {
        "transitions": 0,
        "discriminating": False,
    }


# ---- G1(b): the reference readout (g1b-pool-selection.md section 2.2), R-b0x's readout part ------

# archive/g1b-2026-09-25/probe_ridge_frozen.out and probe_rb6_red.out, unrounded: the frozen
# estimator's accuracy over the 944 scored events of R-b6's family (and its red twin) on 4242
PROBE_4242 = {
    False: {"e1": 0.871822033898305, "e1-0": 0.7478813559322034},
    True: {"e1": 0.871822033898305, "e1-0": 0.8760593220338984},
}
SHORT = {"candidate": "lsm", "config": rp.R_B5_PERSISTENCE_WITNESS, "form": "carried", "window": 16}


def _stamped():
    """The 48-event fixture with a final block at this revision, so its replays validate."""
    stream = _synthetic_stream()
    stream["final"] = dict(zip(("head", "dirty"), md._revision(), strict=True))
    return stream


def test_readout_fit_worked_by_hand_with_an_unpenalized_bias():
    # pairs ([1,1] -> 0), ([0,1] -> 1): XtX + diag(1, 0) = [[2,1],[1,2]], so
    # W = [[2,-1],[-1,2]] / 3 @ [[1,0],[1,1]] = [[1,-1],[1,2]] / 3; x = [2,1] then scores (1, 0)
    rows = [([1.0], 0), ([0.0], 0), ([2.0], 1), ([0.0], 0)]
    readout = ReferenceReadout(every=2)
    assert md._drive_readout(readout, rows) == [None, None, None, 0]
    np.testing.assert_allclose(readout.w, np.array([[1.0, -1.0], [1.0, 2.0]]) / 3.0, atol=1e-12)
    assert (readout.classes, readout.pairs, readout.solves) == ((0, 1), 3, 1)
    np.testing.assert_array_equal(readout.xtx, [[5.0, 3.0], [3.0, 3.0]])  # the 3rd pair, unsolved
    np.testing.assert_array_equal(readout.xty[0], [3.0, 2.0])
    # the red twin: a bias penalized like every weight gives [[2,-1],[1,2]] / 5
    penalized = md._PenalizedBias(every=2)
    md._drive_readout(penalized, rows)
    np.testing.assert_allclose(penalized.w, np.array([[2.0, -1.0], [1.0, 2.0]]) / 5.0, atol=1e-12)
    assert md._drive_readout(md._GuessBeforeSolve(every=2), rows) == [0, 0, 0, 0]


def test_readout_fixtures_each_beside_its_red():
    result = check_r_b0x_readout()
    assert result["verdict"]
    assert set(result["numbers"]) == {"fit", "tie", "last_solve", "future_only"}
    assert all(result["numbers"].values())
    assert set(result["red"]) == {"bias", "none_before_solve", "tie", "last_solve", "future_only"}
    assert all(result["red"].values())


def test_readout_answers_only_the_categories_present_at_its_last_solve():
    a, b, c = [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    readout = ReferenceReadout(every=2)
    predictions = md._drive_readout(readout, [(a, 0), (b, 0), (c, 1), (c, 2), (c, 0)])
    assert predictions[:3] == [None] * 3 and 2 not in predictions  # 2 paired at event 4, unsolved
    assert readout.classes == (0, 1, 2)  # the 4th pair's solve admits it, for event 6 on
    # an exact tie goes to the lowest id; the last-maximum twin answers the highest
    tie = [([0.0], 0), ([0.0], 0), ([0.0], 1), ([0.0], 1)]
    assert md._drive_readout(ReferenceReadout(every=2), tie)[-1] == 0
    assert md._drive_readout(md._LastMaximum(every=2), tie)[-1] == 1


def test_readout_fingerprint_is_its_own():
    readout = ReferenceReadout(every=2)
    empty = readout.fingerprint()
    assert empty["XtX"] is None and empty["Xty"] is None and empty["W"] is None
    md._drive_readout(readout, [([1.0], 0), ([0.0], 0), ([2.0], 1)])
    printed = readout.fingerprint()
    assert printed["W"] == md._crc(readout.w) and printed["XtX"] == md._crc(readout.xtx)
    assert printed["Xty"] == md._crc(np.stack([readout.xty[0], readout.xty[1]]))
    assert printed["categories"] == [0, 1] and printed["solves"] == 1
    # in a replay it sits under "readout", apart from the grower's "fingerprint": 47 pairs, 2 solves
    run = replay(_synthetic_stream(), "a")
    assert (run["readout"]["fingerprint"]["pairs"], run["readout"]["fingerprint"]["solves"]) == (
        47,
        2,
    )
    assert set(run["fingerprint"]) == {"W", "b", "beta", "prior", "batch_beta"}


def test_a_future_only_category_is_never_predicted_before_it_appears():
    stream = _synthetic_stream()
    t_star = 30
    stream["events"][t_star - 1]["c"] = 7  # a category no earlier event carries
    run = replay(stream, "a")
    assert 7 not in run["readout"]["predictions"][:t_star]
    assert 7 not in run["predictions"][:t_star]


def test_leak_arm_leaks_through_both_learners():
    stream = _synthetic_stream()
    run = replay(stream, "leak")
    categories = [event["c"] for event in stream["events"]]
    assert run["predictions"] == categories and run["readout"]["predictions"] == categories


# ---- G1(b): G1(a)'s arms unchanged (R-b8's with/without half) and the BLAS record --------------


def test_g1a_arms_are_byte_identical_with_the_readout_on_and_off():
    stream = _synthetic_stream()
    for arm in md.SCORED_ARMS:
        on, off = replay(stream, arm), replay(stream, arm, readout=False)
        assert off["readout"] is None and len(on["readout"]["predictions"]) == 48
        assert all(on[key] == off[key] for key in GROWER_KEYS), arm
    # the red: a readout that writes into the grower's input changes them
    with md._writing_readout():
        written = replay(stream, "a")
    base = replay(stream, "a")
    assert written["phi"] != base["phi"] and written["fingerprint"] != base["fingerprint"]
    assert md.ReferenceReadout is ReferenceReadout  # the context restored the class


def test_the_e1_one_element_mutant_turns_the_pre_build_comparison_red():
    stream = _synthetic_stream()
    base = replay(stream, "e1")
    with md._e1_mutant():
        mutant = replay(stream, "e1")
    assert mutant["phi"] != base["phi"]
    assert replay(stream, "e1")["phi"] == base["phi"]  # the context restored _feature


def test_replay_runs_at_one_blas_thread_and_records_it():
    run = replay(_synthetic_stream(), "a")
    assert "blas" in _policy() and run["policy"]["blas"]
    assert all(pool["num_threads"] == 1 for pool in run["policy"]["blas"])
    assert run["seed"] == 17 and run["pool"] is None and run["seconds"] > 0


def test_score_without_floors_is_the_same_computation():
    stream = _synthetic_stream()
    run = replay(stream, "a")
    full = score(stream, run["predictions"], run["fallbacks"])
    lean = score(stream, run["predictions"], run["fallbacks"], floors=False)
    floors = {"truth_entropy", "persistence", "causal_majority", "transitions"}
    for segment, row in lean.items():
        assert row == {key: full[segment][key] for key in row}
        assert set(full[segment]) - set(row) == (floors if row["available"] else set())
    assert list(full["unseen-order"]) == [
        "n",
        "available",
        "accuracy",
        "coverage",
        "prediction_entropy",
        "truth_entropy",
        "prior_fallback",
        "persistence",
        "causal_majority",
        "transitions",
    ]


# ---- G1(b): the pool arm, R-b2 and R-b4's noise ---------------------------------------------


def test_pool_arm_replay_carries_its_binding_and_degeneracy():
    stream = _stamped()
    run = replay(stream, "pool", pool=SHORT)
    md._validate_run(run, stream, stream, "pool", "causal")
    config = rp.canonical("lsm", SHORT["config"])
    fresh = rp.make_pool("lsm", config, stream["seed"], "carried", window=16)
    binding = run["pool"]
    assert binding["candidate"] == "lsm" and binding["form"] == "carried"
    assert binding["config"] == config and binding["config_crc32"] == rp.crc_json(config)
    assert binding["realization_seed"] == rp.realization_seed("lsm", stream["seed"])
    assert binding["w_in"] == md._crc(rp.w_in(stream["seed"]))
    assert binding["topology"] == fresh.params_fingerprint()
    assert binding["initial"] == fresh.fingerprint()
    assert binding["boundary"] == fresh.params_fingerprint()  # g1 learns nothing in its window
    assert (binding["window"], binding["percepts"], binding["frozen"]) == (16, 48, True)
    assert binding["movement"] == {} and set(binding["startup"]) == {"transient", "flagged"}
    # the degeneracy statistics over the 24 "unseen-order" rows, computed apart
    rows = np.asarray([fresh.percept(event["x"]) for event in stream["events"]][24:])
    constant = np.all(rows == rows[0], axis=0)
    assert binding["degeneracy"] == {
        "rows": 24,
        "constant_fraction": float(np.count_nonzero(constant) / 256),
        "distinct_rows": len({tuple(row) for row in rows}),
    }
    assert run["phi"][-1] == md._crc(np.asarray(rows[-1], dtype=np.float32))
    assert binding["state"] == fresh.fingerprint()
    assert len(run["readout"]["predictions"]) == 48


def test_pool_forms_share_the_boundary_and_the_front_end_lesions():
    stream = _synthetic_stream()
    runs = {form: replay(stream, "pool", pool=dict(SHORT, form=form)) for form in rp.FORMS}
    assert len({run["pool"]["boundary"] for run in runs.values()}) == 1
    assert all(run["pool"]["frozen"] for run in runs.values())
    assert runs["front_end"]["pool"]["lesion"]["changed"] == ["w"]
    assert runs["carried"]["pool"]["startup"] is not None
    assert runs["settled"]["pool"]["startup"] is None  # reported once per realization


def test_pool_noise_changes_state_and_phi_and_the_clean_twin_does_not():
    stream = _synthetic_stream()
    clean = replay(stream, "pool", pool=SHORT)
    twin = replay(stream, "pool", pool=SHORT)
    noisy = replay(stream, "pool", pool=dict(SHORT, noise=True))
    assert (noisy["pool"]["noise"], noisy["pool"]["kicks"]) == (True, 32)
    assert noisy["pool"]["state"] != clean["pool"]["state"] and noisy["phi"] != clean["phi"]
    assert twin["pool"]["state"] == clean["pool"]["state"] and twin["phi"] == clean["phi"]
    assert twin["execution_id"] != clean["execution_id"]


def test_pool_arm_refusals_and_r_b2():
    stream = _synthetic_stream()
    with pytest.raises(ValueError, match="pool spec"):
        replay(stream, "pool")
    with pytest.raises(ValueError, match="pool spec"):
        replay(stream, "a", pool=SHORT)
    with pytest.raises(ValueError, match="unknown pool spec"):
        replay(stream, "pool", pool=dict(SHORT, extra=1))
    with pytest.raises(ValueError, match="outside section 2.3's grid"):
        replay(stream, "pool", pool=dict(SHORT, config=dict(SHORT["config"], k=9)))
    with pytest.raises(ValueError, match="from its pool"):
        _feature("pool", _View(ARMS["pool"][1], {"x": [0.0] * 64}))
    result = check_r_b2()
    assert result["verdict"] and result["red"] and result["numbers"]["channels"] == ["x"]
    with pytest.raises(ValueError, match="forbidden"):
        _View(ARMS["pool"][1], {"x": [0.0] * 64, "c": 1}).get("c")


# ---- G1(b): R-b6's synthetic family (section 2.7) ------------------------------------------------


def _readout_accuracy(stream, arm):
    """The readout alone over an arm's features, in replay's order (predict, then the pair)."""
    readout, predictions, xs, cs = ReferenceReadout(), [], [], []
    for event in stream["events"]:
        x, c = event["x"], event["c"]
        values = {
            "x": x,
            "c": c,
            "own_predictions": None,
            "x_history": [xs[-i] if len(xs) >= i else [0.0] * 64 for i in range(1, 4)],
            "c_history": [c] + [cs[-i] if len(cs) >= i else None for i in range(1, 4)],
        }
        predictions.append(readout.predict())
        readout.observe(_feature(arm, _View(ARMS[arm][1], values)), c)
        xs.append(x)
        cs.append(c)
    scored = score(stream, predictions, [False] * len(predictions), floors=False)
    return scored["unseen-order"]["accuracy"]


def test_rb6_stream_reproduces_section_2_7():
    import random

    stream = rb6_stream(4242)
    events, sleep = stream["events"], stream["sleep"]
    assert len(events) == len(sleep) == 1200
    assert stream["seed"] == 4242 and stream["policy"]["threads"] == 1
    assert stream["config"] == {
        "seed": 4242,
        "refit_every": 20,
        "buffer_cap": 1000,
        "l_max": None,
        "eps_tol": 0.28,
        "T": 15,
    }
    assert set(stream["final"]) == {"head", "dirty"}
    rng, symbols = random.Random(4242), []
    while len(symbols) < 1200:
        symbols.extend(((0, 2, 3, 4), (1, 2, 3, 5))[rng.random() < 0.5])
    assert [event["c"] for event in events] == symbols[:1200]
    dims = np.random.default_rng(4242).permutation(64)
    for event in events[:24]:
        x, block = np.asarray(event["x"]), dims[8 * event["c"] : 8 * event["c"] + 8]
        assert sorted(np.flatnonzero(x)) == sorted(block)
        assert np.all(x[block] == 1.73 / np.sqrt(8)) and abs(np.linalg.norm(x) - 1.73) < 1e-12
    for i, (event, tick) in enumerate(zip(events, sleep, strict=True)):
        assert (event["t"], event["tick"]) == (i + 1, 2 * i + 1)
        assert (tick["tick"], tick["brain_tick"], tick["event"]) == (i + 1, 2 * i + 2, i + 1)
        assert event["causal_lp"] is None and tick["live_lp"] is None
        assert event["segment"] == ("warm-up" if i < 256 else "unseen-order")


def test_rb6_history_free_twin():
    positive, twin = rb6_stream(4242), rb6_stream(4242, history_free=True)

    def followers(stream):
        cats = [event["c"] for event in stream["events"]]
        return {b for a, b in itertools.pairwise(cats) if a == 3}

    assert followers(positive) == {4, 5} and followers(twin) == {4}
    assert twin["synthetic"] == {"family": "rb6", "history_free": True}
    assert [4 if e["c"] == 5 else e["c"] for e in positive["events"]] == [
        e["c"] for e in twin["events"]
    ]


def test_readout_reproduces_the_frozen_design_probe_on_4242():
    with rp._blas_threads(1):
        for history_free, expected in PROBE_4242.items():
            stream = rb6_stream(4242, history_free=history_free)
            assert {arm: _readout_accuracy(stream, arm) for arm in expected} == expected
    assert PROBE_4242[False]["e1"] - PROBE_4242[False]["e1-0"] >= md.RB6_BAR  # R-b6 on 4242
    assert PROBE_4242[True]["e1"] - PROBE_4242[True]["e1-0"] < md.RB6_BAR  # and its red twin


def test_rb6_through_replay_reads_both_learners_as_the_design_probes_did():
    stream = rb6_stream(4242)
    runs = {arm: replay(stream, arm) for arm in ("e1", "e1-0")}
    count = len(stream["events"])
    readout = {
        arm: score(stream, run["readout"]["predictions"], [False] * count, floors=False)
        for arm, run in runs.items()
    }
    assert {arm: row["unseen-order"]["accuracy"] for arm, row in readout.items()} == PROBE_4242[
        False
    ]
    grower = {
        arm: score(stream, run["predictions"], run["fallbacks"], floors=False)
        for arm, run in runs.items()
    }
    # probe_rb6_positive_control.out printed the grower's accuracies rounded to 4 places
    assert {arm: round(row["unseen-order"]["accuracy"], 4) for arm, row in grower.items()} == {
        "e1": 0.5551,
        "e1-0": 0.5519,
    }
