"""L1c's brain-free reds on the real engine (specs/022 evidence/l1c-online-readout.md section 2.0c):
under a growth gate that closes (beta held stale) the online readouts beat the batch-only engine's
live accuracy on a drifting chain and the sign-flipped rule reads below; ungated the ridge still
does; the neutered rule is bit-identical to no rule; widening preserves the old classes' scores;
the online beta survives the growth record (P4b)."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove  # noqa: E402
from snn_data_lake.hippocampus.online_readout import (  # noqa: E402
    ClosingGate,
    OnlineReadout,
    _chain,
    _percept,
)


def _run(seed, readout, gate=None, n=600):
    torch.set_num_threads(1)
    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_readout=readout, growth_gate=gate
    )
    for i, c in enumerate(_chain(seed, n, flip_every=100)):
        eng.observe(_percept(c, i), c)
        if i % 20 == 19:
            eng.tick()
    return eng


@pytest.mark.parametrize("kind", ["delta", "sign", "ridge"])
def test_under_a_closed_gate_the_online_readout_beats_the_stale_one(kind):
    for seed in (0, 1):
        base = _run(seed, None, ClosingGate(3))
        live = _run(seed, OnlineReadout(kind), ClosingGate(3))
        b, acc_live = base.state()["live_accuracy"], live.state()["live_accuracy"]
        assert acc_live > b + 0.05, (kind, seed, acc_live, b)
        assert live.online_readout.updates > 0 and live.state()["readout_charge"] > 0.0


def test_the_sign_flipped_rule_reads_below_under_a_closed_gate_and_the_ridge_wins_ungated():
    for seed in (0, 1):
        b = _run(seed, None, ClosingGate(3)).state()["live_accuracy"]
        f = _run(seed, OnlineReadout("delta", sign=-1.0), ClosingGate(3)).state()["live_accuracy"]
        assert f < b, (seed, f, b)
    b0 = _run(0, None).state()["live_accuracy"]
    r0 = _run(0, OnlineReadout("ridge")).state()["live_accuracy"]
    assert r0 > b0 + 0.05, (r0, b0)


def test_the_neutered_readout_is_bit_identical_to_no_rule():
    base, neut = _run(2, None), _run(2, OnlineReadout("delta", mu=0.0))
    assert torch.equal(base._model.beta, neut._model.beta)
    assert torch.equal(base._model.W, neut._model.W)
    assert base.state()["live_accuracy"] == neut.state()["live_accuracy"]
    assert neut.state()["readout_charge"] == 0.0


def test_widening_preserves_the_old_scores_and_the_record_round_trips():
    eng = _run(3, OnlineReadout("ridge"))
    m = eng._model
    s = torch.rand(int(m.W.shape[0]))
    before = (s @ m.beta).clone()
    OnlineReadout.widen(m, int(m.beta.shape[1]) + 2)
    after = s @ m.beta
    assert torch.allclose(after[: before.shape[0]], before, atol=1e-6)
    assert bool((after[before.shape[0] :] == 0).all())
    blob = eng.growth_state_bytes()
    fresh = NeurogenesisSelfImprove(seed=99, device="cpu")
    assert fresh.restore_growth_state(blob) and torch.equal(fresh._model.beta, m.beta)
