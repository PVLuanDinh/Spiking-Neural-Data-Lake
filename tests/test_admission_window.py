"""Phase 2's admission window and instrument (specs/022 evidence/l-parallel-arms.md, 2026-09-05):
the brain-free reds the record panel found missing (PH-4, PH-5, M2). Each rule has a control that
fails it.

  * marginal_holdout covers the category marginal with each category's most recent rows, keeps a
    single-row category in training, holds out exactly n_val rows -- and the deleted recency split
    FAILS the contract on a bursty stream (its held-out slice is one run of one category).
  * L3 rule (i): a constant input admits exactly ONE unit, ALIVE (its readout answers the majority),
    and STAYS at one across the engine's grow-on-top refits (each seeded from the prior model);
    with the rule off the same chain admits one free unit per refit (the in-brain mechanism that
    grew 13-51 units on a constant percept before L3).
  * L3 rule (ii): a seeded all-redundant set on a constant input prunes to ONE survivor.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.hippocampus.ng_snn import NGSNN, marginal_holdout  # noqa: E402


def _bursty(seed, n=400, cats=5, p_switch=0.02):
    rng = random.Random(seed)
    c, out = 0, []
    for _ in range(n):
        if rng.random() < p_switch:
            c = rng.randrange(cats)
        out.append(c)
    return out


def _recency(y, n_val):
    n = len(y)
    return list(range(n - n_val)), list(range(n - n_val, n))


def test_marginal_holdout_covers_the_marginal_and_the_recency_split_does_not():
    for seed in range(4):
        y = _bursty(seed) + [7]  # a single-row category at the end
        tr, va = marginal_holdout(y, 60)
        assert len(va) == 60 and sorted(tr + va) == list(range(len(y)))
        present = {c for c in y if y.count(c) >= 2}
        assert {y[i] for i in va} == present, "every category with >= 2 rows is held out"
        assert 400 in tr, "a single-row category stays in training"
        for c in present:  # the most recent rows of each category
            rows = [i for i, v in enumerate(y) if v == c]
            held = [i for i in va if y[i] == c]
            assert held == rows[-len(held) :]
        assert marginal_holdout(y, 60) == (tr, va)  # deterministic
        # THE CONTROL: the recency split's held-out slice is one run (or two) of a bursty stream
        _, va_r = _recency(y, 60)
        seen_r = {y[i] for i in va_r} - {7}
        assert len(seen_r) < len(present), "the recency window misses categories"
    # many categories: the one-row floors yield and exactly n_val rows are held out
    y = [c for c in range(30) for _ in range(3)]
    tr, va = marginal_holdout(y, 10)
    assert len(va) == 10 and len(tr) == 80


def _constant_task(seed, n=120, d=8, cats=3):
    g = torch.Generator().manual_seed(seed)
    return torch.full((n, d), 0.5), torch.randint(0, cats, (n,), generator=g)


def _chain(x, y, refits=6):
    """The engine's grow-on-top path: each refit is seeded from the prior model's units."""
    W, b, units = None, None, []
    for k in range(refits):
        m = NGSNN(l_max=None, eps_tol=0.28, T=8, device="cpu", seed=k).fit(x, y, init_W=W, init_b=b)
        W, b, units = m.W.clone(), m.b.clone(), units + [m.n_units]
    return m, units


def test_a_constant_input_admits_one_alive_unit_and_the_rule_off_admits_more():
    for seed in range(3):
        x, y = _constant_task(seed)
        m, units = _chain(x, y)
        assert units == [1] * 6, units
        rates = m._rates(x, m.W, m.b)
        assert rates.min().item() > 0.0, (
            "the one unit must be alive (a dead unit's readout answers class 0)"
        )
        maj = torch.bincount(y).argmax().item()
        assert (m.predict(x) == maj).all(), "the one unit's readout answers the training majority"
    # THE CONTROL: with rule (i) off (every candidate reads as varying) the same chain admits one
    # free unit per refit -- the first block's consistency bar is 0.0, so a constant candidate
    # collinear with the seeded units still passes
    x, y = _constant_task(0)
    real_all = torch.Tensor.all

    def fake_all(self, *a, **k):  # the equality test's .all(dim=0) reads False: (i) never fires
        if k.get("dim") == 0 and self.dtype == torch.bool:
            return torch.zeros(self.shape[1], dtype=torch.bool)
        return real_all(self, *a, **k)

    torch.Tensor.all = fake_all
    try:
        _, units_off = _chain(x, y)
    finally:
        torch.Tensor.all = real_all
    assert units_off[-1] > 1, (
        f"with rule (i) off the chain admits one free unit per refit ({units_off})"
    )


def test_an_all_redundant_seeded_set_prunes_to_one_survivor():
    x, y = _constant_task(1)
    g = torch.Generator().manual_seed(1)
    w0 = torch.rand(5, 8, generator=g) * 2 - 1
    b0 = torch.rand(5, generator=g) * 2 - 1
    m = NGSNN(l_max=None, eps_tol=0.28, T=8, device="cpu", seed=1).fit(x, y, init_W=w0, init_b=b0)
    assert m.n_units == 5, (
        "the seeded units are inherited; rule (i) admits no new constant unit past them"
    )
    pruned = m.prune_unearned(x, y)
    assert pruned == 4 and m.n_units == 1, (pruned, m.n_units)
