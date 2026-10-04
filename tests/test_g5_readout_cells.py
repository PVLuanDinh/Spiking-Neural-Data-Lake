"""G5 / T113's instrument reds (specs/022 evidence/g5-readout-variants.md 2.4, 2.6, 2.7a): the bare
self-check (reach, lifecycle, determinism, the frozen generator, DeltaSched's in-arm matching, the
reading's NOT MET cases), plus the gate's plumbing on a one-seed family -- never the registered
seeds."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.instruments import g5_readout_cells as g  # noqa: E402
from snn_data_lake.readout.online_readout import OnlineReadout  # noqa: E402


def test_bare_self_check():
    g.main([])


def test_lambda_zero_is_the_incumbent_bit_for_bit():
    torch.set_num_threads(1)
    a = g._short(20261003, OnlineReadout("delta"))
    b = g._short(20261003, g.TraceDelta(0.0))
    assert torch.equal(a._model.beta, b._model.beta)
    assert a.state()["live_accuracy"] == b.state()["live_accuracy"]


def test_lambda_out_of_range_refuses():
    with pytest.raises(ValueError):
        g.TraceDelta(1.0)
    with pytest.raises(ValueError):
        g.DeltaSched(-0.1)


def test_generator_refuses_an_unknown_kind_and_is_seeded():
    with pytest.raises(ValueError):
        g.planted_chain(20261003, "walk", 1.0)
    a = g.planted_chain(20261003, "iid", 1.0, n=30)
    assert a != g.planted_chain(20261004, "iid", 1.0, n=30)


def test_reading_rules():
    assert not g.reading("deg", [0.2] * 8)[0]  # ruling 9: degenerate reads NOT MET
    assert not g.reading("ties", [0.3, 0.2, 0.25, 0.4, 0.0, 0.0, 0.0, 0.0])[0]  # 4 non-tied
    met, c, x = g.reading("up", [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2])
    assert met and x["pattern"] == "++++++++" and 0.0 < c.fpr < 0.1  # the FPR is printed, not gated


def test_gate_plumbing_on_one_seed():
    lines = []
    out = g.reachability_gate(
        discovery=(20261003,),
        replication=(20261004,),
        family=(("burst", 2.0, g.LAMBDAS[0]),),
        log=lines.append,
    )
    assert not out["pa1_met"]  # one seed is degenerate: NOT MET by ruling 9
    assert any("PA1 NOT MET" in s for s in lines)
    assert any("independence reference" in s for s in lines)
