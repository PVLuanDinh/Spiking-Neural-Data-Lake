"""`NGSNN.decide` — the ONE decision rule over a rate matrix, and the class it must never answer.

Opened by SC-012a's review panel, which retracted that arc's verdict
(specs/022-unified-brain-arc/evidence/sc-012a-width.md section 6). `_rates` returns cnt / T and is
non-negative, so a held-out row where no fitted unit fires has an exactly-zero rate vector; a bare
`argmax` resolves that tie to column 0, and `fit` sizes beta as `int(y.max()) + 1` so column 0
exists even when the stream never
emits class 0 as a target. Measured on the live engine at the time: the tie-break decided 45 of 60
held-out rows on one seed (accuracy 0.233) and 60 of 60 on another (0.000).

The three pins, each with the control that makes it a measurement (record:
evidence/scoring-path-abstention.md):

  * THE RED -- a silent pool over a training set whose classes exclude 0: the retired bare argmax
    answers 0 on every row (reconstructed here, so the defect is not merely described), and `decide`
    answers the training majority instead. If these two cannot be made to differ, the rule buys
    nothing and the record's section 2.6 deletes it.
  * THE CONTROL THAT IS NOT THE METHOD -- on rows where at least one unit FIRES, `decide` is
    bit-identical to the bare argmax. An instrument fix that moves a prediction it was not built to
    change is a second defect wearing a repair's clothes.
  * ONE MECHANISM -- the engine's closed-gate branch no longer carries its own argmax; it calls
    `decide`. Asserted by reading the source, because a P2 duplicate is a property of the tree, not
    of a run.
"""

import pathlib

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.hippocampus.ng_snn import NGSNN  # noqa: E402


def _bare_argmax(S, beta):
    """The RETIRED rule, reconstructed. A gate that has only ever been seen green is
    indistinguishable from one that cannot see, so the defect is rebuilt rather than described."""
    return (S @ beta).argmax(dim=1)


def _fitted(seed=0, n=40, d=6):
    """A real fit whose targets are 1..3 -- class 0 exists as a beta column and is NEVER a target,
    which is exactly the engine's shape (it buffers percept_{t-1} -> category_t)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.rand(n, d, generator=g)
    y = torch.tensor([1 + (i % 3) for i in range(n)])
    m = NGSNN(l_max=4, eps_tol=0.0, T=15, device="cpu", seed=seed)
    m.fit(X, y)
    return m, X, y


def test_the_prior_is_recorded_from_the_targets_not_the_width_of_the_one_hot():
    m, _X, y = _fitted()
    assert m.m == 4, "beta spans 0..3, so a column 0 exists"
    assert 0 not in set(y.tolist()), "class 0 is never a target -- the engine's own shape"
    assert m.prior_class_ in {1, 2, 3}, m.prior_class_
    ys = y.tolist()
    assert m.prior_class_ == max(set(ys), key=ys.count)


def test_a_silent_pool_answers_the_prior_where_the_retired_rule_answered_an_impossible_class():
    """THE RED. Every row silent -> the retired argmax answers 0 on all of them; `decide` answers
    the training majority, and counts every row it answered that way."""
    m, _X, _y = _fitted()
    S_silent = torch.zeros(7, m.n_units)

    retired = _bare_argmax(S_silent, m.beta)
    assert retired.tolist() == [0] * 7, (
        "the reconstructed defect must fire: a bare argmax over an all-zero score answers column 0"
    )
    assert 0 != m.prior_class_, "the red is only reachable while 0 is not itself the prior"

    got = m.decide(S_silent)
    assert got.tolist() == [m.prior_class_] * 7, got.tolist()
    assert m.last_abstentions_ == 7, m.last_abstentions_
    assert not torch.equal(got, retired), "the fix must CHANGE this case or it buys nothing"


def test_a_firing_row_is_bit_identical_to_the_retired_rule():
    """THE CONTROL THAT IS NOT THE METHOD. Where any unit fires, `decide` moves NOTHING."""
    m, X, _y = _fitted()
    S = m._rates(X.float(), m.W, m.b)
    firing = S.abs().sum(dim=1) > 0
    assert bool(firing.any()), "the control is vacuous unless some row actually fires"
    S_fire = S[firing]

    assert torch.equal(m.decide(S_fire), _bare_argmax(S_fire, m.beta)), (
        "a correct instrument fix moves ZERO of the predictions it was not built to change"
    )
    assert m.last_abstentions_ == 0

    # and MIXED rows: only the silent ones move
    S_mix = torch.cat([S_fire[:3], torch.zeros(2, m.n_units)], dim=0)
    out = m.decide(S_mix)
    assert torch.equal(out[:3], _bare_argmax(S_fire[:3], m.beta)), "firing rows moved"
    assert out[3:].tolist() == [m.prior_class_] * 2
    assert m.last_abstentions_ == 2


def test_an_unfitted_model_invents_no_prior():
    """Absence is a fact: with no fit there is no majority to answer with, so `decide` stays on the
    bare argmax rather than manufacturing a class."""
    m = NGSNN(l_max=2, T=15, device="cpu", seed=0)
    assert m.prior_class_ is None
    beta = torch.zeros(3, 4)
    S = torch.zeros(2, 3)
    assert m.decide(S, beta).tolist() == _bare_argmax(S, beta).tolist()


def test_predict_routes_through_decide_and_the_engine_keeps_no_second_argmax():
    """ONE MECHANISM (P2). `predict` delegates, and the engine's closed-gate branch calls `decide`
    instead of the inline argmax it used to carry."""
    m, X, _y = _fitted()
    S = m._rates(X.float(), m.W, m.b)
    assert torch.equal(m.predict(X), m.decide(S))

    _eng_path = pathlib.Path("src/snn_data_lake/cortex/neurogenesis_engine.py")
    eng = _eng_path.read_text(encoding="utf-8")
    assert "m.decide(S_te, beta_inst)" in eng, "the closed-gate branch must use the one rule"
    assert "@ beta_inst).argmax(" not in eng, "the retired inline copy is still in the engine"
