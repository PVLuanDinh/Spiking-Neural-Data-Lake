"""Fast synthetic reds for the G1(a) causal contract."""

import copy

import pytest
import torch

from snn_data_lake.cortex.growth_stop import GrowthStop, feed_and_decide
from snn_data_lake.cortex.memory_diagnostic import (
    ARMS,
    _feature,
    _lp_sensitive_red,
    _synthetic_stream,
    _View,
    check_r_a1,
    check_r_a2,
    check_r_a2b,
    check_r_a3,
    check_r_a4,
    onehot_32,
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
        assert len(_feature(arm, _View(channels, values))) == width


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
    stream["refits"] = [{"refit": row["fingerprint"]} for row in live["refits"]]
    for event, prediction in zip(stream["events"], live["predictions"], strict=True):
        event["prediction"] = prediction
    assert check_r_a1(stream)["verdict"]
    assert check_r_a2(stream)["verdict"]
    assert check_r_a4(stream)["verdict"]
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


@pytest.mark.slow
def test_real_120_tick_recording_r_a1():
    from snn_data_lake.cortex.memory_diagnostic import check_r_a1

    stream = record_stream(900, ticks=120)
    result = check_r_a1(copy.deepcopy(stream))
    assert result["numbers"]["equivalent"] and result["numbers"]["refits"] > 0
    assert not result["numbers"]["eligible"]
