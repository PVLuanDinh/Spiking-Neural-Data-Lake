"""Fast synthetic reds for the G1(a) causal contract, and G1(b)'s extensions of the harness: the
reference readout (g1b-pool-selection.md section 2.2), the pool arm, R-b2, R-b8's comparisons and
R-b6's synthetic family (section 2.7)."""

import contextlib
import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from snn_data_lake.growth.growth_stop import GrowthStop, feed_and_decide
from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.instruments.memory_diagnostic import (
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
from snn_data_lake.reservoir import recurrent_pool as rp


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
            # G1(c)'s transport channels as the loop supplies them before b (None: not yet there)
            "w@b": None,
            "grower_activity": None,
        }
        # the pool arms read their features from their pool (and refuse without one)
        pool = rp.make_pool("lsm", rp.R_B5_PERSISTENCE_WITNESS, 0) if arm in md.POOL_ARMS else None
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
    from snn_data_lake.instruments.memory_diagnostic import check_r_a1

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

    import snn_data_lake.instruments.memory_diagnostic as diagnostic

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


# ---- G1(c) 2.6: the pool arm at window 0 -----------------------------------------------------


def test_window_zero_pool_arm_binds_its_window_and_an_empty_movement():
    stream = _stamped()
    run = replay(stream, "pool", pool=dict(SHORT, window=0))
    md._validate_run(run, stream, stream, "pool", "causal")
    config = rp.canonical("lsm", SHORT["config"])
    fresh = rp.make_pool("lsm", config, stream["seed"], "carried", window=0)
    binding = run["pool"]
    assert (binding["window"], binding["percepts"], binding["movement"]) == (0, 48, {})
    assert binding["boundary"] == fresh.params_fingerprint() and binding["frozen"] is True
    assert binding["initial"] == fresh.fingerprint() and binding["pool_ms"] is not None
    assert binding["kicks"] == 0 and set(binding["startup"]) == {"transient", "flagged"}
    # no window to move across: the field is {} in every form, Front-end included (its window-16
    # field is None, because its parameters are the lesion's), so it stays comparable
    for form in rp.FORMS:
        zero = replay(stream, "pool", pool=dict(SHORT, form=form, window=0))
        assert zero["pool"]["movement"] == {} and zero["pool"]["frozen"] is True
        assert zero["pool"]["boundary"] is not None
    assert replay(stream, "pool", pool=dict(SHORT, form="front_end"))["pool"]["movement"] is None
    assert replay(stream, "pool", pool=SHORT)["pool"]["movement"] == {}


def test_carried_at_zero_is_carried_at_256_on_the_fixture_and_the_comparison_can_differ():
    stream = _stamped()
    zero = replay(stream, "pool", pool=dict(SHORT, window=0))
    default = replay(stream, "pool", pool={k: v for k, v in SHORT.items() if k != "window"})
    assert (zero["pool"]["window"], default["pool"]["window"]) == (0, rp.WINDOW)
    assert all(zero[key] == default[key] for key in GROWER_KEYS)
    assert zero["readout"] == default["readout"]
    assert all(zero["pool"][key] == default["pool"][key] for key in ("state", "state_chain"))
    # 48 events never reach window 256's boundary: these fields differ, and are not compared
    assert default["pool"]["boundary"] is None and zero["pool"]["boundary"] is not None
    assert default["pool"]["frozen"] is None and zero["pool"]["frozen"] is True
    # the instrument fires: Settled@0 is not Carried@0 on the same events
    settled = replay(stream, "pool", pool=dict(SHORT, form="settled", window=0))
    assert settled["phi"] != zero["phi"]
    assert settled["pool"]["state_chain"] != zero["pool"]["state_chain"]


def test_window_zero_noise_kicks_before_every_percept():
    stream = _synthetic_stream()
    clean = replay(stream, "pool", pool=dict(SHORT, window=0))
    noisy = replay(stream, "pool", pool=dict(SHORT, window=0, noise=True))
    assert (noisy["pool"]["noise"], noisy["pool"]["kicks"]) == (True, 48)
    assert noisy["pool"]["state"] != clean["pool"]["state"] and noisy["phi"] != clean["phi"]


def test_window_zero_leaves_the_arms_and_the_forms_as_they_were():
    # the order is part of it: SCORED_ARMS is the first eight; G1(c)'s transport arms are appended
    # after the last arm G1(b) had
    assert tuple(ARMS) == (
        ("a", "f64", "d", "d0", "e1", "e1-0", "e2", "e2-0")
        + ("forbidden", "noise", "leak", "pool")
        + ("pool-copy", "pool-act", "pool-swap")
    )
    assert ARMS == {
        "a": (64, ("x",)),
        "f64": (64, ("c",)),
        "d": (96, ("x", "own_predictions")),
        "d0": (96, ("x",)),
        "e1": (256, ("x", "x_history")),
        "e1-0": (256, ("x",)),
        "e2": (192, ("x", "c_history")),
        "e2-0": (192, ("x", "c")),
        "forbidden": (96, ("x",)),
        "noise": (64, ("x",)),
        "leak": (64, ("x",)),
        "pool": (256, ("x",)),
        "pool-copy": (256, ("x", "w@b")),
        "pool-act": (256, ("grower_activity",)),
        "pool-swap": (256, ("x",)),
    }
    assert md.SCORED_ARMS == ("a", "f64", "d", "d0", "e1", "e1-0", "e2", "e2-0")
    assert md._PoolArm.KEYS == ("candidate", "config", "form", "noise", "window")
    assert rp.FORMS == ("carried", "reset", "settled", "front_end")


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


# ---- G1(c) 2.7 and 2.10 "Transport": the read-only tap, the transport arms, the swap control -----

B = 16  # the fixture's b: 256 does not fit 48 events; 16 is SHORT's window, as b is the real one's
CUTS = (24, 32, 40)  # R-a2's three cuts, moved past b on the 48-event fixture
LATE = 8  # the late tap's offset here (40 would run past the fixture's 48 events)


@pytest.fixture(scope="module")
def tapped():
    stream = _stamped()
    return stream, replay(stream, "a", tap=B)


def test_the_transport_arms_are_appended_and_the_scored_arms_do_not_move():
    assert md.SCORED_ARMS == tuple(ARMS)[:8]
    assert md.SCORED_ARMS == ("a", "f64", "d", "d0", "e1", "e1-0", "e2", "e2-0")
    assert tuple(ARMS)[8:12] == ("forbidden", "noise", "leak", "pool")
    assert md.TRANSPORT_B == 256 and md.LATE_TAP_EVENTS == 40
    assert md.TRANSPORT_CHANNELS == {
        "pool-copy": ("x", "w@b"),
        "pool-act": ("grower_activity",),
        "pool-swap": ("x",),
    }
    assert md.TRANSPORT_ARMS == ("pool-copy", "pool-act", "pool-swap")
    assert md.POOL_ARMS == ("pool", *md.TRANSPORT_ARMS)
    assert {arm: ARMS[arm] for arm in md.TRANSPORT_ARMS} == {
        arm: (rp.N, channels) for arm, channels in md.TRANSPORT_CHANNELS.items()
    }
    assert md._PoolArm.KEYS == ("candidate", "config", "form", "noise", "window")


def test_a_default_replay_carries_no_tap_and_the_plain_pool_arm_neither_transport_key(tapped):
    stream, _ = tapped
    plain = replay(stream, "pool", pool=SHORT)
    assert "tap" not in plain and "tap" not in replay(stream, "a")
    assert "encoder" not in plain["pool"] and "transport" not in plain["pool"]
    assert list(plain["pool"])[:11] == [
        "candidate",
        "form",
        "config",
        "config_crc32",
        "realization_seed",
        "window",
        "s_settle",
        "noise",
        "w_in",
        "topology",
        "initial",
    ]


def test_a_tap_and_a_side_are_each_refused_off_their_arms(tapped):
    stream, run = tapped
    for arm, pool in (("e1-0", None), ("pool", SHORT)):
        with pytest.raises(ValueError, match="arm a only"):
            replay(stream, arm, pool=pool, tap=B)
    with pytest.raises(ValueError, match="arm a only"):
        replay(stream, "pool-copy", pool=SHORT, tap=B, side=run["tap"])
    for arm, pool in (("a", None), ("pool", SHORT)):
        with pytest.raises(ValueError, match="transport arms"):
            replay(stream, arm, pool=pool, side=run["tap"])


def test_a_transport_arm_needs_its_pool_spec_and_the_copy_and_activity_arms_a_side(tapped):
    stream, run = tapped
    with pytest.raises(ValueError, match="pool spec"):
        replay(stream, "pool-copy", side=run["tap"])
    for arm in ("pool-copy", "pool-act"):
        with pytest.raises(ValueError, match="needs a side"):
            replay(stream, arm, pool=SHORT)
    # the swap reads nothing from the tap: without one it swaps at TRANSPORT_B, past this fixture
    bare = replay(stream, "pool-swap", pool=SHORT)
    assert bare["pool"]["transport"]["swapped_at_event"] is None and bare["pool"]["encoder"] is None
    with_side = replay(stream, "pool-swap", pool=SHORT, side=run["tap"])
    assert with_side["pool"]["transport"]["swapped_at_event"] == B


def test_a_side_is_refused_unless_it_belongs_to_this_replay(tapped):
    stream, run = tapped
    side = run["tap"]
    ok = replay(stream, "pool-copy", pool=SHORT, side=side)
    md._validate_run(ok, stream, stream, "pool-copy", "causal")
    wrong = {
        "source_digest": 0,
        "config_digest": 1,
        "seed": 18,
        "revision": {"head": "0" * 40, "dirty": False},
    }
    for key, value in wrong.items():
        with pytest.raises(ValueError, match=f"side tap {key}"):
            replay(stream, "pool-copy", pool=SHORT, side=dict(copy.deepcopy(side), **{key: value}))
    # a tap of another stream (a suffix mutation of it) is refused by its digest
    changed = md.mutate_suffix(stream, 30, 5)
    with pytest.raises(ValueError, match="side tap source_digest"):
        replay(changed, "pool-copy", pool=SHORT, side=side)
    short = copy.deepcopy(side)
    short["activity"] = short["activity"][:-1]
    with pytest.raises(ValueError, match="side tap activity"):
        replay(stream, "pool-act", pool=SHORT, side=short)
    with pytest.raises(ValueError, match="side tap"):
        replay(stream, "pool-copy", pool=SHORT, side={})


def test_the_tap_is_refused_where_it_has_nothing_to_read():
    stream = _stamped()
    with pytest.raises(ValueError, match="no model"):
        replay(stream, "a", tap=8)  # the first refit lands at event 12
    with pytest.raises(ValueError, match="no delivery"):
        replay(stream, "a", tap=48)  # the 49th delivery does not exist
    for bad in (0, -1, 2.5, True, "16"):
        with pytest.raises(ValueError, match="tap"):
            replay(stream, "a", tap=bad)


def test_the_tap_records_the_weights_at_b_the_activity_and_every_refit(tapped):
    stream, run = tapped
    tap = run["tap"]
    assert tap["b"] == B and tap["seed"] == stream["seed"] and tap["revision"] == run["revision"]
    assert tap["source_digest"] == run["source_digest"]
    assert tap["config_digest"] == run["config_digest"]
    at = tap["w_at_b"]
    assert at["event"] == B and at["observed"] == B
    W = np.asarray(at["W"])
    assert W.ndim == 2 and W.shape[1] == 64 and W.shape[0] > 0
    # crc_W reads the model's own float32 tensor: it is the W crc of the last refit at or before b,
    # and the listed floats are that tensor exactly
    last = [r for r in run["refits"] if r["event"] <= B][-1]
    assert at["crc_W"] == last["fingerprint"]["W"]
    assert md._crc(torch.tensor(at["W"], dtype=torch.float32)) == at["crc_W"]
    # one weight block per completed refit, at its event
    assert [w["event"] for w in tap["refit_weights"]] == [r["event"] for r in run["refits"]]
    for entry, refit in zip(tap["refit_weights"], run["refits"], strict=True):
        assert md._crc(torch.tensor(entry["W"], dtype=torch.float32)) == refit["fingerprint"]["W"]
    # activity: one slot per delivery, None until the grower holds a model (first refit: event 12)
    first = run["refits"][0]["event"]
    assert len(tap["activity"]) == len(stream["events"]) == 48
    assert all(a is None for a in tap["activity"][:first])
    assert all(a is not None for a in tap["activity"][first:])
    widths = {len(entry["W"]) for entry in tap["refit_weights"]}
    assert {len(a) for a in tap["activity"][first:]} <= widths


def test_the_activity_is_the_rates_of_the_current_percept_read_where_observe_reads_them(
    monkeypatch,
):
    seen = []
    original = NeurogenesisSelfImprove.observe

    def observe(self, features, category):
        model = self._model
        if model is not None and model.W is not None:
            x = torch.tensor([features], dtype=torch.float32)
            seen.append(model._rates(x, model.W, model.b)[0].tolist())
        else:
            seen.append(None)
        return original(self, features, category)

    monkeypatch.setattr(NeurogenesisSelfImprove, "observe", observe)
    run = replay(_stamped(), "a", tap=B)
    assert len(seen) == 48 and run["tap"]["activity"] == seen


def test_each_tap_fault_makes_its_own_tap_and_the_honest_one_comes_back(tapped):
    stream, run = tapped
    with md._late_tap(by=LATE):
        late = replay(stream, "a", tap=B)["tap"]
    assert late["b"] == B and late["w_at_b"]["event"] == B + LATE
    assert late["w_at_b"]["observed"] == B + LATE
    with md._leaky_tap():
        leaky = replay(stream, "a", tap=B)["tap"]
    final = run["refits"][-1]["fingerprint"]["W"]
    assert leaky["w_at_b"]["event"] == B and leaky["w_at_b"]["crc_W"] == final
    assert leaky["w_at_b"]["W"] != run["tap"]["w_at_b"]["W"]
    assert replay(stream, "a", tap=B)["tap"] == run["tap"]  # every fault restored on exit


def test_the_tap_is_read_only_and_a_writing_tap_is_not(tapped):
    stream, on = tapped
    off = replay(stream, "a")
    assert all(on[key] == off[key] for key in GROWER_KEYS) and on["readout"] == off["readout"]
    with md._writing_tap():
        written = replay(stream, "a", tap=B)
    assert any(written[key] != off[key] for key in GROWER_KEYS)
    result = md.check_tap_read_only(stream, b=B)
    assert result["verdict"] and result["red"] is True
    assert all(result["numbers"]["grower"].values()) and result["numbers"]["readout"] is True


def test_the_copy_arm_swaps_the_encoder_once_at_b_and_records_what_it_copied(tapped):
    stream, run = tapped
    tap, seed = run["tap"], stream["seed"]
    plain = replay(stream, "pool", pool=SHORT)
    copied = replay(stream, "pool-copy", pool=SHORT, side=tap)
    md._validate_run(copied, stream, stream, "pool-copy", "causal")
    w_random = rp.w_in(seed)
    expected = rp.transported_encoder(w_random, tap["w_at_b"]["W"])
    W = np.asarray(tap["w_at_b"]["W"])
    units = W.shape[0]
    assert copied["pool"]["transport"] == {
        "form": "copy",
        "copied_at_event": B,
        "L": units,
        "padding_fraction": (256 - units) / 256,
        "raw_row_norms": rp.row_norms(W).tolist(),
        "mean_row_norm": rp.mean_row_norm(w_random),
        "encoder_crc": md._crc(expected),
    }
    # the binding keeps the construction crc and gains the post-swap one
    assert copied["pool"]["w_in"] == md._crc(w_random)
    assert copied["pool"]["encoder"] == md._crc(expected) != copied["pool"]["w_in"]
    # nothing before b moves; from b on the features do
    assert copied["phi"][:B] == plain["phi"][:B] and copied["phi"][B:] != plain["phi"][B:]
    # the same features from a pool whose encoder is swapped by hand, once, at percept b + 1
    fresh = rp.make_pool("lsm", rp.canonical("lsm", SHORT["config"]), seed, "carried", window=16)
    phis = []
    for index, event in enumerate(stream["events"]):
        if index == B:
            fresh.w_in = expected
        phis.append(md._crc(np.asarray(fresh.percept(event["x"]), dtype=np.float32)))
    assert phis == copied["phi"]
    # a swap is no pool state: the pool's own topology is the plain arm's
    assert copied["pool"]["topology"] == plain["pool"]["topology"]


def test_the_activity_arm_drives_the_pool_with_the_zero_padded_activity(tapped):
    stream, run = tapped
    tap, seed = run["tap"], stream["seed"]
    act = replay(stream, "pool-act", pool=SHORT, side=tap)
    md._validate_run(act, stream, stream, "pool-act", "causal")
    fresh = rp.make_pool("lsm", rp.canonical("lsm", SHORT["config"]), seed, "carried", window=16)
    phis = []
    for activity in tap["activity"]:
        x = (
            np.zeros(64)
            if activity is None
            else np.asarray(activity + [0.0] * (64 - len(activity)))
        )
        phis.append(md._crc(np.asarray(fresh.percept(x), dtype=np.float32)))
    assert phis == act["phi"]
    silent = sum(activity is None for activity in tap["activity"])
    assert silent > 0
    assert act["pool"]["transport"] == {
        "form": "activity",
        "silent_events": silent,
        "events": 48,
        "widest": max(len(a) for a in tap["activity"] if a is not None),
    }
    assert act["pool"]["encoder"] is None and act["pool"]["w_in"] == md._crc(rp.w_in(seed))


def test_the_activity_arm_refuses_more_than_64_units(tapped):
    stream, run = tapped
    wide = copy.deepcopy(run["tap"])
    wide["activity"][20] = [0.5] * 65
    with pytest.raises(ValueError, match="more than 64"):
        replay(stream, "pool-act", pool=SHORT, side=wide)


def test_the_swap_arm_swaps_to_a_fresh_draw_at_the_same_b_and_transports_nothing(tapped):
    stream, run = tapped
    seed = stream["seed"]
    w_random = rp.w_in(seed)
    fresh = rp.fresh_encoder(seed, w_random)
    swap = replay(stream, "pool-swap", pool=SHORT, side=run["tap"])
    assert swap["pool"]["transport"] == {
        "form": "swap",
        "swapped_at_event": B,
        "mean_row_norm": rp.mean_row_norm(w_random),
        "encoder_crc": md._crc(fresh),
    }
    assert swap["pool"]["encoder"] == md._crc(fresh) and swap["pool"]["w_in"] == md._crc(w_random)
    plain = replay(stream, "pool", pool=SHORT)
    assert swap["phi"][:B] == plain["phi"][:B] and swap["phi"][B:] != plain["phi"][B:]
    # nothing is transported: another tap's weights (the same b) give the very same replay
    with md._leaky_tap():
        other = replay(stream, "a", tap=B)["tap"]
    assert other["w_at_b"]["W"] != run["tap"]["w_at_b"]["W"]
    again = replay(stream, "pool-swap", pool=SHORT, side=other)
    assert again["phi"] == swap["phi"] and again["pool"]["encoder"] == swap["pool"]["encoder"]
    copied = replay(stream, "pool-copy", pool=SHORT, side=run["tap"])
    assert copied["pool"]["encoder"] != swap["pool"]["encoder"] and copied["phi"] != swap["phi"]


def test_the_loop_always_supplies_the_three_side_values_to_a_transport_arm(tapped, monkeypatch):
    stream, run = tapped
    tap = run["tap"]
    seen = []

    class Spy(md._View):
        def __init__(self, channels, values):
            seen.append(values)
            super().__init__(channels, values)

    monkeypatch.setattr(md, "_View", Spy)
    replay(stream, "pool-copy", pool=SHORT, side=tap)
    assert len(seen) == 48
    for index, values in enumerate(seen):
        assert {"w@b", "grower_activity", "w_live"} <= set(values)
        assert values["w@b"] == (tap["w_at_b"]["W"] if index >= B else None)
        assert values["grower_activity"] == tap["activity"][index]
        earlier = [w for w in tap["refit_weights"] if w["event"] <= index]
        assert values["w_live"] == (earlier[-1]["W"] if earlier else None)
    seen.clear()
    replay(stream, "a")
    assert all("w@b" not in values and "w_live" not in values for values in seen)


def test_a_view_holds_only_its_declared_channels():
    values = {
        "x": [0.0] * 64,
        "c": 1,
        "pairs": 3,
        "w@b": [[1.0] * 64],
        "grower_activity": [0.1],
        "w_live": [[2.0] * 64],
    }
    for arm, declared in md.TRANSPORT_CHANNELS.items():
        view = _View(ARMS[arm][1], values)
        assert set(view.values) == set(declared)
        for forbidden in ("w_live", "c", "pairs"):
            with pytest.raises(ValueError, match="forbidden feature channel"):
                view.get(forbidden)
    with pytest.raises(ValueError, match="forbidden feature channel"):
        _View(ARMS["pool-act"][1], values).get("x")
    with pytest.raises(ValueError, match="forbidden feature channel"):
        _View(ARMS["pool-copy"][1], values).get("grower_activity")
    # a declared channel the loop failed to supply is a KeyError, never a silent None
    with pytest.raises(KeyError):
        _View(ARMS["pool-copy"][1], {"x": [0.0] * 64}).get("w@b")


def test_the_declaration_check_accepts_the_frozen_set_and_refuses_every_other():
    for arm, declared in md.TRANSPORT_CHANNELS.items():
        assert md._declaration_problems(arm, declared) == []
    assert md._declaration_problems("pool-copy", ("x", "w@b", "w_live")) == ["w_live"]
    assert md._declaration_problems("pool-copy", ("x",)) == ["missing w@b"]
    assert md._declaration_problems("pool-act", ("grower_activity", "x")) == ["x"]
    assert md._declaration_problems("pool", ("x",)) == ["not a transport arm"]


def test_the_channel_check_holds_and_each_red_fires(tapped):
    stream, _ = tapped
    result = md.check_transport_channels(stream, SHORT, b=B)
    assert result["verdict"], result
    assert result["red"] == {
        "request_w_live": True,
        "request_w_live_in_replay": True,
        "declares_w_live": True,
        "pool_act_reads_x": True,
    }
    assert result["numbers"]["declared"] == {
        arm: list(channels) for arm, channels in md.TRANSPORT_CHANNELS.items()
    }
    bare = md.check_transport_channels()  # without a stream the replay-level red is not run
    assert bare["verdict"] and "request_w_live_in_replay" not in bare["red"]


def test_the_forbidden_request_raises_through_a_replay_with_w_live_present(tapped):
    stream, run = tapped
    with md._w_live_mutant():
        with pytest.raises(ValueError, match="forbidden feature channel: w_live"):
            replay(stream, "pool-copy", pool=SHORT, side=run["tap"])
        # the mutant keeps the arm's permitted declaration: it is the REQUEST that raises
        assert ARMS["pool-copy"][1] == ("x", "w@b")
    replay(stream, "pool-copy", pool=SHORT, side=run["tap"])  # restored


def test_the_truncated_stream_keeps_b_events_and_the_sleep_before_delivery_b_plus_1():
    stream = _stamped()
    cut = md._truncated(stream, B)
    assert cut["events"] == stream["events"][:B]
    assert cut["sleep"] == [
        s for s in stream["sleep"] if s["brain_tick"] < stream["events"][B]["tick"]
    ]
    assert len(cut["sleep"]) == B and md._source_digest(cut) != md._source_digest(stream)
    ended = replay(cut, "a")
    assert (ended["ticks"], ended["observed"]) == (B, B)
    with pytest.raises(ValueError, match="no delivery"):
        md._truncated(stream, 48)


def test_the_timing_check_holds_with_two_instruments_and_a_late_tap_fails_it(tapped):
    stream, run = tapped
    result = md.check_transport_timing(stream, b=B, late=LATE)
    assert result["verdict"], result
    n = result["numbers"]
    assert n["agree_i"] and n["agree_ii"]
    assert n["tap"]["crc_W"] == n["last_refit_crc"] == n["truncated"]["crc_W"]
    assert n["tap"]["crc_W"] == run["tap"]["w_at_b"]["crc_W"]
    assert (n["tap"]["event"], n["tap"]["observed"]) == (B, B)
    assert (n["truncated"]["events"], n["truncated"]["observed"]) == (B, B)
    # the late tap (copy at b + 8) fails (ii) by position and (i) because a refit moved W between
    assert result["red"] == {"late_fails_ii": True, "late_fails_i": True, "moved_between": True}
    assert n["late"]["tap"]["event"] == B + LATE != n["tap"]["event"]


def test_a_late_tap_over_an_unmoved_weight_fails_the_position_instrument_alone():
    # the refit at event 20 leaves W's crc as it was at 16: (i), which reads the crc, cannot see a
    # tap taken then; (ii), which reads the stream's end position as well, always does
    result = md.check_transport_timing(_stamped(), b=B, late=4)
    assert result["red"] == {"late_fails_ii": True, "late_fails_i": False, "moved_between": False}
    assert result["verdict"]


def test_the_timing_check_run_on_a_late_tap_fails_its_own_position_instrument():
    # the tap under test is itself late (event 20, over a W unchanged since 16): instrument (i)
    # reads the crc and agrees, instrument (ii) reads the position as well and does not
    with md._late_tap(by=4):
        result = md.check_transport_timing(_stamped(), b=B, late=4)
    assert result["numbers"]["tap"]["event"] == B + 4
    assert result["numbers"]["agree_i"] is True and result["numbers"]["agree_ii"] is False
    assert not result["verdict"]


def test_the_causality_check_holds_at_three_cuts_and_the_leaky_tap_changes_every_one(tapped):
    stream, _ = tapped
    result = md.check_transport_causality(stream, SHORT, b=B, cuts=CUTS)
    assert result["verdict"], result
    n = result["numbers"]
    assert n["pool-copy"] == [True] * 3 and n["pool-act"] == [True] * 3
    assert n["tap_crc_unchanged"] == [True] * 3  # the tap re-taken on each mutated stream
    assert result["red"] == {"leaky_tap": [True] * 3}
    assert n["cuts"] == list(CUTS)


def test_each_transport_check_says_no_when_its_red_cannot_fire(tapped, monkeypatch):
    # a check whose constructed fault is replaced by an honest run must refuse to pass: the red is
    # part of the verdict, and a red that does not fire is the check failing
    stream, _ = tapped

    def honest(*args, **kwargs):
        return contextlib.nullcontext()

    monkeypatch.setattr(md, "_writing_tap", honest)
    read_only = md.check_tap_read_only(stream, b=B)
    assert not read_only["verdict"] and read_only["red"] is False
    monkeypatch.setattr(md, "_leaky_tap", honest)
    causality = md.check_transport_causality(stream, SHORT, b=B, cuts=CUTS)
    assert not causality["verdict"] and causality["red"] == {"leaky_tap": [False] * 3}
    monkeypatch.setattr(md, "_late_tap", honest)
    timing = md.check_transport_timing(stream, b=B, late=LATE)
    assert not timing["verdict"] and timing["red"]["late_fails_ii"] is False
    monkeypatch.setattr(md, "_w_live_mutant", honest)
    channels = md.check_transport_channels(stream, SHORT, b=B)
    assert not channels["verdict"]
    assert channels["red"]["request_w_live"] is False
    assert channels["red"]["request_w_live_in_replay"] is False


def test_the_causality_check_refuses_a_cut_not_after_b(tapped):
    stream, _ = tapped
    with pytest.raises(ValueError, match="after b"):
        md.check_transport_causality(stream, SHORT, b=B, cuts=(16, 32, 40))
    with pytest.raises(ValueError, match="after b"):
        md.check_transport_causality(stream, SHORT, b=B)  # R-a2's cuts here are 12, 24, 36
