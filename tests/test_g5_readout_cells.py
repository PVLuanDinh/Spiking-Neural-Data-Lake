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


def test_beta_identity_is_exact_bytes_and_a_sub_rounding_perturbation_breaks_it():
    # Codex cp4 #11: B-R2's bit identity compares exact bytes, dtype and shape; the retired rounded
    # hash (1e-6 grid) would have passed this perturbation
    beta = torch.randn(5, 4, generator=torch.Generator().manual_seed(20261003))
    nudged = beta.clone()
    nudged[0, 0] = torch.nextafter(nudged[0, 0], torch.tensor(float("inf")))
    assert (nudged * 1e6).round().equal((beta * 1e6).round()), "the perturbation is sub-rounding"
    assert g.tensor_digest(nudged) != g.tensor_digest(beta)
    assert g.tensor_digest(beta.clone()) == g.tensor_digest(beta)
    assert g.tensor_digest(beta.double()) != g.tensor_digest(beta)  # dtype is part of identity
    assert g.tensor_digest(beta.reshape(4, 5)) != g.tensor_digest(beta)  # so is shape


def test_same_rows_halts_on_a_moved_refit_floor_and_arm_table_prints_the_floor():
    # Codex cp4 #12: every registered floor is compared; each arm prints its accuracy vs the floor
    row = {
        "seed": 1,
        "arm": "DELTA",
        "widened": 0,
        "cats_hash": "x",
        "cols_hash": "y",
        "live_pers": 0.9,
        "live_acc": 0.8,
        "floors": [(0.5, 0.9), (0.6, 0.92)],
    }
    g.same_rows([row, {**row, "arm": "COUPLED"}])
    with pytest.raises(g.SameRowsHalt, match="refit floors"):
        g.same_rows([row, {**row, "arm": "COUPLED", "floors": [(0.5, 0.9), (0.6, 0.93)]}])
    with pytest.raises(g.SameRowsHalt, match="refit floors"):
        g.same_rows([row, {**row, "arm": "COUPLED", "floors": [(0.5, 0.9)]}])
    lines = []
    g.arm_table([row, {**row, "arm": "COUPLED", "live_acc": 0.85}], log=lines.append)
    assert any("acc - floor -0.0500" in s for s in lines)


def test_a_home_swapped_for_another_populated_dir_is_refused_not_just_a_live_one(
    tmp_path, monkeypatch
):
    # Codex cp4 #7: NORD_HOME must be the yielded home; any other non-live dir is a red
    from snn_data_lake.substrate.home import LiveHomeRefused, throwaway_home

    other = tmp_path / "populated_B"
    other.mkdir()
    with throwaway_home() as fresh:
        assert g.check_fresh_home(fresh)  # the yielded home passes
        monkeypatch.setenv("NORD_HOME", str(other))
        with pytest.raises(LiveHomeRefused, match="not the fresh one"):
            g.check_fresh_home(fresh)
        monkeypatch.setenv("NORD_HOME", fresh)


def _fake_cell(args):
    seed, arm, seq = args
    return {
        "seed": seed,
        "arm": arm,
        "cats": [0, 1, 0, 1],
        "cats_hash": "h",
        "cols_hash": "c",
        "widened": 0,
        "live_pers": 0.5,
        "live_acc": 0.5,
        "floors": [],
        "unified": True,
    }


def test_the_live_lake_check_runs_after_each_stage_and_survives_an_exception(monkeypatch):
    # Codex cp4 #13: a RECORD-stage violation must halt BEFORE the arm stage launches
    from concurrent.futures import ThreadPoolExecutor

    stamps = iter([("a", "b"), ("a2", "b"), ("a2", "b"), ("a2", "b")])
    monkeypatch.setattr(g, "live_lake_stamp", lambda: next(stamps))
    ran = []

    def cell(args):
        ran.append(args[1])
        return _fake_cell(args)

    with pytest.raises(g.LiveLakeMoved, match="RECORD stage"):
        g.run_cells(
            (1,), ("DELTA",), workers=1, log=lambda *_: None, pool=ThreadPoolExecutor, cell=cell
        )
    assert ran == ["RECORD"]  # the arm stage never launched

    # an exception in a stage while the lake moved reports the move, not the hidden exception
    stamps = iter([("a", "b"), ("a", "b"), ("a2", "b")])
    monkeypatch.setattr(g, "live_lake_stamp", lambda: next(stamps))

    def boom(args):
        if args[1] != "RECORD":
            raise ValueError("arm crashed")
        return _fake_cell(args)

    with pytest.raises(g.LiveLakeMoved, match="raised"):
        g.run_cells(
            (1,), ("DELTA",), workers=1, log=lambda *_: None, pool=ThreadPoolExecutor, cell=boom
        )


def test_a_fallback_perception_seed_is_replaced_whole_from_the_reserve(monkeypatch):
    # Codex cp4 #9: `unified` False is an instrument failure; the whole seed is discarded and the
    # next reserve seed replaces it, printed beside it; with no reserve left the failure surfaces
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(g, "live_lake_stamp", lambda: ("a", "b"))

    def cell(args):
        out = _fake_cell(args)
        out["unified"] = args[0] != 2  # seed 2 ran on fallback perception, in both stages
        return out

    lines = []
    rec, cells, rep = g.run_cells(
        (1, 2),
        ("DELTA", "BATCH"),
        workers=1,
        log=lines.append,
        pool=ThreadPoolExecutor,
        cell=cell,
        reserve=(9,),
    )
    assert sorted(rec) == [1, 9] and {c["seed"] for c in cells} == {1, 9} and len(cells) == 4
    assert rep == [(2, 2, 9)] and any("seed 2 FAILED" in s and "replaced by 9" in s for s in lines)
    with pytest.raises(g.InstrumentFailure, match="unified False"):
        g.run_cells(
            (1, 2), ("DELTA",), workers=1, log=lambda *_: None, pool=ThreadPoolExecutor, cell=cell
        )


def _b_cells(coupled_accs, z_acc=0.5, z_beta="b", c_traj=()):
    out = []
    for i, ca in enumerate(coupled_accs):
        base = {"seed": i, "beta_hash": "b", "traj": [(1, 1, 0.5)], "live_acc": 0.5}
        out += [
            {**base, "arm": "DELTA"},
            {**base, "arm": "COUPLED0", "live_acc": z_acc, "beta_hash": z_beta},
            {**base, "arm": "COUPLED", "live_acc": ca, "traj": list(c_traj) or base["traj"]},
        ]
    return out


def test_reach_gate_halts_before_b_r1_and_a_significant_negative_reads_harmful():
    # amendment 2026-10-06 points 1 and 3
    neg = [-0.05, -0.04, -0.06, -0.05, -0.04, -0.07, -0.05, -0.06]
    met, line = g.read_b_r1(_b_cells([0.5 + d for d in neg]))
    assert not met and "HARMFUL" in line and "NOT MET" in line
    pos = [0.05, 0.04, 0.06, 0.05, 0.04, 0.07, 0.05, 0.06]
    met, line = g.read_b_r1(_b_cells([0.5 + d for d in pos]))
    assert met and "HARMFUL" not in line
    null = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
    assert "HARMFUL" not in g.read_b_r1(_b_cells([0.5 + d for d in null]))[1]
    with pytest.raises(g.ReachHalt, match="B-R2"):  # COUPLED0 != DELTA: no verdict at all
        g.read_b_r1(_b_cells([0.6] * 8, z_acc=0.51))
    with pytest.raises(g.ReachHalt, match="B-R4"):  # a coupled arm moved the refit trajectory
        g.read_b_r1(_b_cells([0.6] * 8, c_traj=[(1, 1, 0.9)]))


def _conf_cells(coupled, shuf, marg, perm):
    out = []
    for i in range(8):
        base = {
            "seed": i,
            "beta_hash": "b",
            "traj": [(1, 1, 0.5)],
            "live_acc": 0.5,
            "live_pers": 0.9,
        }
        out += [
            {**base, "arm": "BATCH", "live_acc": 0.3 + 0.01 * i},
            {**base, "arm": "DELTA"},
            {**base, "arm": "COUPLED0"},
            {**base, "arm": "COUPLED", "live_acc": 0.5 + coupled[i]},
            {**base, "arm": "COUPLED_SHUF", "live_acc": 0.5 + shuf[i]},
            {**base, "arm": "COUPLED_MARG", "live_acc": 0.5 + marg[i]},
            {**base, "arm": "COUPLED_PERM", "live_acc": 0.5 + perm[i]},
        ]
    return out


def test_read_part_b_lands_only_when_every_conjunct_holds_and_prints_the_gate():
    # a non-periodic recording, so the circular shift moves many teaching labels
    rec = {i: {"cats": [(k * k // 3) % 4 for k in range(40)]} for i in range(8)}
    up = [0.06, 0.05, 0.07, 0.06, 0.05, 0.08, 0.06, 0.07]
    zero = [0.0, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01]
    lines, landed = g.read_part_b(_conf_cells(up, zero, zero, zero), rec)
    assert landed and "reach gate: PASSED" in lines[0] and any("conjunct 4" in x for x in lines)
    assert any("acc - floor" in x for x in lines)  # the persistence context is printed
    # COUPLED_SHUF - DELTA MET adds a refusal (conjunct 4), so the reading does not land
    assert not g.read_part_b(_conf_cells(up, up, zero, zero), rec)[1]
    # the control that is not the method ties the method: conjunct 1 NOT MET
    assert not g.read_part_b(_conf_cells(up, zero, up, zero), rec)[1]
    # a vacuous (one-category) recording refuses conjunct 3
    vac = {i: {"cats": [2] * 40} for i in range(8)}
    assert not g.read_part_b(_conf_cells(up, zero, zero, zero), vac)[1]
    # a failed reach gate halts the whole read
    bad = _conf_cells(up, zero, zero, zero)
    bad[2]["live_acc"] = 0.9  # COUPLED0 != DELTA on seed 0
    with pytest.raises(g.ReachHalt):
        g.read_part_b(bad, rec)
