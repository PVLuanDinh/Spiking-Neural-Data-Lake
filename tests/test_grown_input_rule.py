"""L1b's brain-free reds on the REAL grower (specs/022 evidence/l1b-synaptic-online.md): the
online rule moves NGSNN's input rows toward a planted stream and the closed gate does not; the
row norms (the admission scale) never drift; BCM's LTD is reachable; the moved rows SURVIVE a
grow-on-top refit and a restart of the growth record (P4b); the live instrument scores before
it learns."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove  # noqa: E402
from snn_data_lake.hippocampus.ng_snn import NGSNN  # noqa: E402
from snn_data_lake.plasticity.grown_input_rule import GrownInputRule  # noqa: E402


def _grown(seed, n=200, d=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g)
    y = (x[:, 0] * 3).long().clamp(max=2)
    m = NGSNN(l_max=None, eps_tol=0.28, T=8, device="cpu", seed=seed).fit(x, y)
    assert m.n_units > 1
    return m, x, y


def _cos_rows(W, x):
    return (W @ x) / (W.norm(dim=1) * x.norm()).clamp_min(1e-12)


def test_the_rule_turns_grown_rows_toward_the_stream_and_the_closed_gate_does_not():
    m, x, y = _grown(0)
    m_dead = NGSNN(l_max=None, eps_tol=0.28, T=8, device="cpu", seed=0).fit(x, y)
    target = torch.rand(16, generator=torch.Generator().manual_seed(1))
    norms0 = m.W.norm(dim=1).clone()
    c0 = _cos_rows(m.W, target).clone()
    r, r_dead = GrownInputRule("oja"), GrownInputRule("oja")
    for _ in range(300):
        r.learn(m, target, gate=1.0)
        r_dead.learn(m_dead, target, gate=0.0)
    c1 = _cos_rows(m.W, target)
    fires = m_dead._rates(target.reshape(1, -1), m_dead.W, m_dead.b)[0] > 0
    assert int(fires.sum()) >= 1, "the planted stream must drive at least one grown unit"
    assert bool(((c1 - c0)[fires] > 0.05).all()), (c1 - c0).tolist()
    assert torch.equal(_cos_rows(m_dead.W, target), c0), "the closed gate moved a row"
    assert float((m.W.norm(dim=1) - norms0).abs().max()) < 1e-5, "the admission scale drifted"
    assert r.moved == 300 and r_dead.moved == 0  # moved counts calls with an open gate, not rows


def test_bcm_below_its_threshold_moves_away():
    m, _, _ = _grown(2)
    r = GrownInputRule("bcm")
    strong = torch.ones(16) * 3.0
    for _ in range(40):
        r.learn(m, strong, gate=1.0)  # the firing units' theta climbs toward their rate
    A = m.W @ strong + m.b
    i = int(A.argmax())
    assert float(A[i]) > 1.0
    # the same direction scaled so unit i sits at current ~1.2 (rate 1/3, below its theta)
    weaker = strong * ((1.2 - float(m.b[i])) / float(m.W[i] @ strong))
    before = _cos_rows(m.W, strong).clone()
    r.learn(m, weaker, gate=1.0)
    assert float(_cos_rows(m.W, strong)[i] - before[i]) < 0.0


def _engine_with_rule(kind, seed=3, n=160):
    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_rule=GrownInputRule(kind)
    )
    g = torch.Generator().manual_seed(seed)
    cats = [int(c) for c in torch.randint(0, 3, (n,), generator=g)]
    for k, c in enumerate(cats):
        feats = [0.0] * 64
        feats[c * 8 : c * 8 + 8] = [1.0] * 8  # a category-shaped percept
        feats[40 + (k % 5)] = 0.5
        eng.observe(feats, c, gate=0.8)
        if k % 20 == 19:
            eng.tick()
    return eng, cats


def test_moved_rows_survive_the_refit_and_the_growth_record():
    eng, cats = _engine_with_rule("bcm")
    assert eng._model is not None and eng._model.n_units >= 1
    assert eng.online_rule.moved > 0 and eng._online_charge > 0.0
    W_before = eng._model.W.clone()
    # one more refit: the inherited rows are the LEADING rows and keep their moved values
    for k in range(20):
        eng.observe([0.1] * 64, cats[k], gate=0.0)  # gate closed: nothing moves in between
    eng.tick()
    n = min(int(W_before.shape[0]), int(eng._model.W.shape[0]))
    assert n >= 1
    assert torch.equal(eng._model.W[:n], W_before[:n]), "a refit rewrote the moved inherited rows"
    theta = getattr(eng._model, "bcm_theta", None)
    assert theta is not None and int(theta.shape[0]) == int(eng._model.W.shape[0])
    # the record carries theta and W across a fresh engine (P4b)
    blob = eng.growth_state_bytes()
    fresh = NeurogenesisSelfImprove(seed=99, device="cpu", online_rule=GrownInputRule("bcm"))
    assert fresh.restore_growth_state(blob)
    assert torch.equal(fresh._model.W, eng._model.W)
    assert torch.equal(torch.nan_to_num(fresh._model.bcm_theta), torch.nan_to_num(theta))


def test_the_live_instrument_scores_before_it_learns():
    eng, _ = _engine_with_rule("oja")
    s = eng.state()
    assert (
        s["live_n"] > 0 and 0.0 <= s["live_accuracy"] <= 1.0 and 0.0 <= s["live_persistence"] <= 1.0
    )
    assert s["online_rule"]["kind"] == "oja" and s["online_rule"]["moved"] > 0
