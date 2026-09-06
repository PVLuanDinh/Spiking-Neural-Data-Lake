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
    _feed,
)


def _run(seed, readout, gate=None, n=600):
    torch.set_num_threads(1)
    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_readout=readout, growth_gate=gate
    )
    _feed(eng, _chain(seed, n, flip_every=100))
    return eng


@pytest.mark.parametrize("kind", ["delta"])
def test_under_a_closed_gate_the_online_readout_beats_the_stale_one(kind):
    for seed in (0, 1):
        base = _run(seed, None, ClosingGate(3))
        live = _run(seed, OnlineReadout(kind), ClosingGate(3))
        b, acc_live = base.state()["live_accuracy"], live.state()["live_accuracy"]
        assert acc_live > b + 0.05, (kind, seed, acc_live, b)
        assert live.online_readout.steps > 0 and live.state()["readout_charge"] > 0.0


def test_the_sign_flipped_rule_reads_below_under_a_closed_gate():
    for seed in (0, 1):
        b = _run(seed, None, ClosingGate(3)).state()["live_accuracy"]
        f = _run(seed, OnlineReadout("delta", sign=-1.0), ClosingGate(3)).state()["live_accuracy"]
        assert f < b, (seed, f, b)


def test_the_deleted_candidates_are_refused_at_construction():
    for kind in ("sign", "ridge"):
        with pytest.raises(ValueError):
            OnlineReadout(kind)


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
    assert torch.allclose(after[: before.shape[0]], before, atol=1e-6)
    assert bool((after[before.shape[0] :] == 0).all())
    blob = eng.growth_state_bytes()
    fresh = NeurogenesisSelfImprove(seed=99, device="cpu")
    assert fresh.restore_growth_state(blob) and torch.equal(fresh._model.beta, m.beta)


def test_the_growth_gates_instrument_is_the_batch_readout_while_the_live_one_learns():
    """Amendment 3: under a gate that closes, the closed-gate refits score the held-out slice
    with the last open-gate solve's beta, so the trajectory the growth gate reads is IDENTICAL
    to the batch-only engine's while the live accuracy differs (the online readout learns on
    the wake stream)."""
    for kind in ("delta",):
        base = _run(5, None, ClosingGate(3))
        live = _run(5, OnlineReadout(kind), ClosingGate(3))
        assert [r[2] for r in live.trajectory] == [r[2] for r in base.trajectory], kind
        assert [r[1] for r in live.trajectory] == [r[1] for r in base.trajectory], kind
        assert live.state()["live_accuracy"] != base.state()["live_accuracy"], kind
        assert torch.equal(live._batch_beta, base._model.beta), kind


def test_the_instrument_survives_a_restart_as_a_tensor_distinct_from_the_online_beta():
    """I-1 / PH-1 / M-1 of the cells' panel: the growth record carries the batch instrument
    beside the online beta; a restored engine's closed-gate trajectory continues identical to a
    restored batch-only twin's (the gate's evidence does not re-couple across a relaunch)."""
    live = _run(6, OnlineReadout("delta"), ClosingGate(3), n=300)
    twin = _run(6, None, ClosingGate(3), n=300)
    assert not torch.equal(live._model.beta, live._batch_beta)
    fresh = NeurogenesisSelfImprove(
        seed=99, device="cpu", online_readout="delta", growth_gate=ClosingGate(0)
    )
    fresh_twin = NeurogenesisSelfImprove(seed=99, device="cpu", growth_gate=ClosingGate(0))
    assert fresh.restore_growth_state(live.growth_state_bytes())
    assert fresh_twin.restore_growth_state(twin.growth_state_bytes())
    assert torch.equal(fresh._batch_beta, twin._model.beta)
    assert torch.equal(fresh._model.beta, live._model.beta)
    tail = _chain(6, 400, flip_every=100)[300:]
    _feed(fresh, tail, 300)
    _feed(fresh_twin, tail, 300)
    assert [r[2] for r in fresh.trajectory] == [r[2] for r in fresh_twin.trajectory]
    # a legacy blob without the member: the only beta there is becomes the instrument (disclosed)
    import io

    import numpy as np

    z = np.load(io.BytesIO(live.growth_state_bytes()), allow_pickle=False)
    legacy = {k: z[k] for k in z.files if k != "batch_beta"}
    buf = io.BytesIO()
    np.savez(buf, **legacy)
    old = NeurogenesisSelfImprove(seed=98, device="cpu", online_readout="delta")
    assert old.restore_growth_state(buf.getvalue()) and torch.equal(
        old._batch_beta, old._model.beta
    )
