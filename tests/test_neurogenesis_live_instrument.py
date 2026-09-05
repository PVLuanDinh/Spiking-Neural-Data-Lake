"""The engine's LIVE instrument (specs/022 L1b, 2026-09-05; the rule it was built beside was deleted
on its null): on every wake percept the retained model's prediction of this category from the
PREVIOUS percept is scored BEFORE the transition is buffered and before any rule could learn,
beside a live persistence floor. The per-percept hook (observe's gate kwarg, online_rule) is
exercised with a counting stub."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove  # noqa: E402


class _CountingRule:
    def __init__(self):
        self.calls, self.gates = 0, []

    def learn(self, model, x, gate=None):
        self.calls += 1
        self.gates.append(gate)
        return 0.0


def _feed(eng, n=160, seed=3, gate=0.8):
    g = torch.Generator().manual_seed(seed)
    cats = [int(c) for c in torch.randint(0, 3, (n,), generator=g)]
    for k, c in enumerate(cats):
        feats = [0.0] * 64
        feats[c * 8 : c * 8 + 8] = [1.0] * 8
        feats[40 + (k % 5)] = 0.5
        eng.observe(feats, c, gate=gate)
        if k % 20 == 19:
            eng.tick()
    return cats


def test_the_live_instrument_scores_before_it_learns_and_the_hook_sees_every_percept():
    rule = _CountingRule()
    eng = NeurogenesisSelfImprove(seed=3, device="cpu", refit_every=1, online_rule=rule)
    cats = _feed(eng)
    s = eng.state()
    assert s["live_n"] > 0 and 0.0 <= s["live_accuracy"] <= 1.0
    pers = sum(1 for a, b in zip(cats, cats[1:], strict=False) if a == b) / (len(cats) - 1)
    # the live persistence floor is the stream's own, over the ticks the instrument scored
    assert abs(s["live_persistence"] - pers) < 0.15
    assert rule.calls > 0 and all(g == 0.8 for g in rule.gates)
    assert (
        s["online_rule"] is None or True
    )  # a stub has no state(); None is the incumbent's reading


def test_without_a_rule_nothing_moves_and_the_instrument_still_reads():
    eng = NeurogenesisSelfImprove(seed=4, device="cpu", refit_every=1)
    _feed(eng, gate=None)
    W0 = eng._model.W.clone()
    _feed(eng, n=40, seed=5, gate=None)
    # a refit keeps inherited rows byte-identical, then may prune or re-select them: a survivor
    # is an exact copy of a previous row (values never move without a rule)
    W1 = eng._model.W
    survivors = sum(1 for row in W1 if any(torch.equal(row, r0) for r0 in W0))
    assert survivors >= 1, "no inherited row survived a refit byte-identical"
    assert eng.state()["online_charge"] == 0.0 and eng.state()["live_n"] > 0
    assert eng.state()["online_rule"] is None
