"""G5 / T113's instrument reds (specs/022 evidence/g5-readout-variants.md 2.4, 2.6): the bare
self-check (a deterministic runner, the frozen generator, the reading's NOT MET cases). Part A's
trace code and its 2.7a gate were deleted on the gate's null (commit 4ecc90d9 keeps them as run)."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.instruments import g5_readout_cells as g  # noqa: E402


def test_bare_self_check():
    g.main()


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


def test_the_trace_is_gone():
    for name in ("TraceDelta", "DeltaSched", "reachability_gate", "GATE_FAMILY", "LAMBDAS"):
        assert not hasattr(g, name), f"{name} survived its deletion (record 2.7a)"
