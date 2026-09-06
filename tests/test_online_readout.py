"""L1c's brain-free reds on the real engine (specs/022 evidence/l1c-online-readout.md): the online
readout beats the batch-only engine's live accuracy on a planted chain and the sign-flipped rule
reads below it; the neutered rule is bit-identical to no rule; widening preserves the old classes'
scores; the online beta survives the growth record (P4b)."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove  # noqa: E402
from snn_data_lake.hippocampus.online_readout import OnlineReadout, _chain, _percept  # noqa: E402


def _run(seed, readout, n=600, flip_every=100):
    """The drifting chain (the succession rule reverses every `flip_every`): the batch solve goes
    stale between refits, which is what the online readout's red measures (record section 2.0)."""
    eng = NeurogenesisSelfImprove(seed=seed, device="cpu", refit_every=1, online_readout=readout)
    for i, c in enumerate(_chain(seed, n, flip_every=flip_every)):
        eng.observe(_percept(c, i), c)
        if i % 20 == 19:
            eng.tick()
    return eng


@pytest.mark.parametrize("kind", ["delta", "sign"])
def test_the_online_readout_beats_the_stale_one_and_the_sign_flip_reads_below(kind):
    for seed in (0, 1):
        base, live, flip = (
            _run(seed, None),
            _run(seed, OnlineReadout(kind)),
            _run(seed, OnlineReadout(kind, sign=-1.0)),
        )
        b, acc_live, f = (e.state()["live_accuracy"] for e in (base, live, flip))
        assert acc_live > b + 0.05, (kind, seed, acc_live, b)
        assert f < b, (kind, seed, f, b)
        assert live.online_readout.updates > 0 and live.state()["readout_charge"] > 0.0


def test_the_neutered_readout_is_bit_identical_to_no_rule():
    base, neut = _run(2, None), _run(2, OnlineReadout("delta", mu=0.0))
    assert torch.equal(base._model.beta, neut._model.beta)
    assert torch.equal(base._model.W, neut._model.W)
    assert base.state()["live_accuracy"] == neut.state()["live_accuracy"]
    assert neut.state()["readout_charge"] == 0.0


def test_widening_preserves_the_old_scores_and_the_record_round_trips():
    eng = _run(3, OnlineReadout("delta"))
    m = eng._model
    s = torch.rand(int(m.W.shape[0]))
    before = (s @ m.beta).clone()
    OnlineReadout.widen(m, int(m.beta.shape[1]) + 2)
    after = s @ m.beta
    assert torch.allclose(after[: before.shape[0]], before, atol=1e-6) and bool(
        (after[before.shape[0] :] == 0).all()
    )
    blob = eng.growth_state_bytes()
    fresh = NeurogenesisSelfImprove(seed=99, device="cpu")
    assert fresh.restore_growth_state(blob) and torch.equal(fresh._model.beta, m.beta)
