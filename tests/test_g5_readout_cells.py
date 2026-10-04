"""G5 / T113's instrument reds (specs/022 evidence/g5-readout-variants.md 2.4, 2.6): the bare
self-check (a deterministic runner, the frozen generator, the reading's NOT MET cases). Part A's
trace code and its 2.7a gate were deleted on the gate's null (commit 4ecc90d9 keeps them as run)."""

import os

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.instruments import g5_readout_cells as g  # noqa: E402
from snn_data_lake.readout.online_readout import OnlineReadout  # noqa: E402


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


def test_planted_readout_refuses_an_unknown_plant():
    with pytest.raises(ValueError):
        g.planted_readout(20261003, "shuffle")


def test_coupled_accuracies_are_prequential_and_bounded():
    acc = g.coupled_accuracies(g.planted_readout(20261003, "iid_errors", n=100))
    assert set(acc) == {"uncoupled", "full", "marg"}
    assert all(0.0 <= v <= 1.0 for v in acc.values())


def _wired(arm, n=300):
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    torch.set_num_threads(1)
    chain = g.planted_chain(20261003, "burst", 2.0, n=n)
    seq = [c for _, c in chain]
    M = 1 + max(seq)
    eng = NeurogenesisSelfImprove(
        seed=20261003,
        device="cpu",
        refit_every=1,
        online_readout=None,
        growth_gate=g.ClosingGate(3),
    )
    ro, cc, cl, te = g.arm_parts(arm, M, seq, 20261003)
    rec = g.instrument_engine(eng, ro, M, cc, cl, te)
    for i, (x, c) in enumerate(chain):
        eng.observe(x, c)
        if i % 20 == 19:
            eng.tick()
    return g.summarize(eng, rec, 20261003, arm), cc


def test_coupled0_is_delta_bit_for_bit_and_learning_is_untouched():
    d, _ = _wired("DELTA")
    c0, cc0 = _wired("COUPLED0")
    c, cc = _wired("COUPLED")
    assert c0["live_acc"] == d["live_acc"] and c0["beta_hash"] == d["beta_hash"]  # B-R2's reach
    assert cc0.moved == 0 and cc.decisions > 0 and cc.state()["counts"] == cc.decisions
    assert c["beta_hash"] == d["beta_hash"] and c["traj"] == d["traj"]  # B-R4: learning untouched
    g.same_rows([d, c0, c])


def test_unknown_arm_refuses():
    with pytest.raises(ValueError):
        g.arm_parts("TRACE", 4, [0, 1], 20261003)


def test_contrast_cells_pairs_one_seed():
    cells = [
        {"seed": s, "arm": a, "live_acc": v}
        for s in range(8)
        for a, v in (("X", 0.5 + 0.01 * s), ("Y", 0.4 + 0.012 * s))
    ]
    met, c, x = g.contrast_cells(cells, "X", "Y")
    assert met and x["nontied"] == 8


def test_brain_cell_refuses_a_live_home_before_any_brain(monkeypatch, tmp_path):
    # the shared guard runs inside the cell: with the guard's default live home set to the cell's
    # own temp home (a decoy), the cell must refuse before make_brain is reached
    import snn_data_lake.substrate.home as home

    built = []
    monkeypatch.setattr(
        home, "_default_live_home", lambda: os.environ.get("NORD_HOME") or str(tmp_path / "decoy")
    )
    import snn_data_lake.engine.brain as brain

    monkeypatch.setattr(brain, "make_brain", lambda **kw: built.append(kw))
    with pytest.raises(home.LiveHomeRefused):
        g.brain_cell((20261003, "DELTA", [0, 1, 0]))
    assert built == [], "a Brain was constructed past a refused home"


def test_pb4_a_count_before_scoring_through_the_harness_raises():
    # Codex cp4 #4: the leak is built through the ACTUAL harness path (instrument_engine), where
    # the truth is counted after the decision but before the live instrument scores it
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove
    from snn_data_lake.readout.categorical_coupling import CategoricalCoupling

    torch.set_num_threads(1)
    chain = g.planted_chain(20261003, "burst", 2.0, n=300)
    M = 1 + max(c for _, c in chain)
    eng = NeurogenesisSelfImprove(
        seed=20261003,
        device="cpu",
        refit_every=1,
        online_readout=None,
        growth_gate=g.ClosingGate(3),
    )
    g.instrument_engine(eng, OnlineReadout("delta"), M, CategoricalCoupling(M), _leak=True)
    with pytest.raises(RuntimeError, match="before it is scored"):
        for i, (x, c) in enumerate(chain):
            eng.observe(x, c)
            if i % 20 == 19:
                eng.tick()
