"""The engine's LIVE instrument (specs/022 L1b, 2026-09-05; the rule it was built beside was deleted
on its null, with its hook): on every wake percept the retained model's prediction of this category
from the PREVIOUS percept is scored BEFORE the transition is buffered, beside a live persistence
floor. The ordering is asserted by hand: the test predicts with the retained model before each
observe() and the engine's live counters must equal that tally exactly."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove  # noqa: E402


def _percept(c, k):
    feats = [0.0] * 64
    feats[c * 8 : c * 8 + 8] = [1.0] * 8
    feats[40 + (k % 5)] = 0.5
    return feats


def test_the_live_instrument_scores_the_previous_percept_before_buffering():
    eng = NeurogenesisSelfImprove(seed=3, device="cpu", refit_every=1)
    g = torch.Generator().manual_seed(3)
    cats = [int(c) for c in torch.randint(0, 3, (160,), generator=g)]
    expect_n = expect_correct = expect_persist = 0
    prev, prev_cat = None, None
    for k, c in enumerate(cats):
        feats = _percept(c, k)
        m = eng._model
        if prev is not None and m is not None and m.W is not None and m.beta is not None:
            pred = int(m.predict(torch.tensor([prev], dtype=torch.float32))[0])
            expect_n += 1
            expect_correct += int(pred == c)
            expect_persist += int(prev_cat == c)
        eng.observe(feats, c)
        prev, prev_cat = feats, c
        if k % 20 == 19:
            eng.tick()
    s = eng.state()
    assert s["live_n"] == expect_n > 0
    assert s["live_accuracy"] == expect_correct / expect_n
    assert s["live_persistence"] == expect_persist / expect_n


def test_the_instrument_is_none_until_a_model_exists_and_nothing_moves_without_a_rule():
    eng = NeurogenesisSelfImprove(seed=4, device="cpu", refit_every=1)
    assert eng.state()["live_accuracy"] is None and eng.state()["live_n"] == 0
    for k in range(40):
        eng.observe(_percept(k % 2, k), k % 2)
    eng.tick()
    W0 = eng._model.W.clone()
    for k in range(40, 60):
        eng.observe(_percept(k % 2, k), k % 2)
    eng.tick()
    # a refit keeps inherited rows byte-identical, then may prune or re-select them: a survivor is
    # an exact copy of a previous row (values never move without a rule)
    survivors = sum(1 for row in eng._model.W if any(torch.equal(row, r0) for r0 in W0))
    assert survivors >= 1
    assert eng.state()["live_n"] > 0
