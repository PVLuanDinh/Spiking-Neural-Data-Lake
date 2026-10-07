"""G5 / T113's instrument reds (specs/022 evidence/g5-readout-variants.md 2.4, 2.6): the bare
self-check (a deterministic runner, the frozen generator, the reading's NOT MET cases). Part A's
trace code and its 2.7a gate were deleted on the gate's null (commit 4ecc90d9 keeps them as run);
Part B's coupling, arms and planted checks were deleted on B-R1 NOT MET (record 3.6)."""

import os

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


def test_the_coupling_is_gone_and_only_the_incumbent_arms_remain():
    for name in (
        "planted_readout",
        "coupled_accuracies",
        "planted_checks",
        "read_part_b",
        "b_reach_gate",
    ):
        assert not hasattr(g, name), f"{name} survived its deletion (record 3.6)"
    for arm in ("TRACE", "COUPLED", "COUPLED0", "COUPLED_MARG", "COUPLED_SHUF", "COUPLED_PERM"):
        with pytest.raises(ValueError):
            g.arm_parts(arm, 4, [0, 1], 20261003)
    assert g.arm_parts("BATCH", 4, [0, 1], 20261003) == (None, None)
    assert g.arm_parts("DELTA", 4, [0, 1], 20261003)[0] is not None


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
        "live_n": 10,
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


def test_a_seed_that_scored_nothing_is_an_instrument_failure_not_a_later_crash(monkeypatch):
    # Codex g5_b lens-1 M3: unified=True with live_n=0 / None metrics scored no prediction; it must
    # fail INSIDE the seed block (replaceable), not crash the contrast later
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(g, "live_lake_stamp", lambda: ("a", "b"))
    for bad in ({"live_n": 0}, {"live_acc": None}, {"live_pers": None}):

        def cell(args, bad=bad):
            out = _fake_cell(args)
            if args[1] != "RECORD":
                out.update(bad)
            return out

        with pytest.raises(g.InstrumentFailure, match="scored no prediction"):
            g.run_cells(
                (1,), ("DELTA",), workers=1, log=lambda *_: None, pool=ThreadPoolExecutor, cell=cell
            )


def test_a_terminal_exception_still_gets_the_lake_check_and_is_never_replaced(monkeypatch):
    # Codex g5_b lens-1 m3: KeyboardInterrupt (a BaseException) bypassed the lake check; with the
    # lake moved it must report the move, with the lake still it must propagate, not use a reserve
    from concurrent.futures import ThreadPoolExecutor

    def stop(args):
        raise KeyboardInterrupt

    stamps = iter([("a", "b"), ("a2", "b")])
    monkeypatch.setattr(g, "live_lake_stamp", lambda: next(stamps))
    with pytest.raises(g.LiveLakeMoved, match="raised"):
        g.run_cells(
            (1,), ("DELTA",), workers=1, log=lambda *_: None, pool=ThreadPoolExecutor, cell=stop
        )
    monkeypatch.setattr(g, "live_lake_stamp", lambda: ("a", "b"))
    lines = []
    with pytest.raises(KeyboardInterrupt):
        g.run_cells(
            (1,),
            ("DELTA",),
            workers=1,
            log=lines.append,
            pool=ThreadPoolExecutor,
            cell=stop,
            reserve=(9,),
        )
    assert not any("replaced" in s for s in lines)


def test_pa3_exits_nonzero_when_the_positive_control_fails(monkeypatch):
    # Codex g5_b lens-1 m2: `--pa3` printed NOT MET and still exited 0
    monkeypatch.setattr(g, "pa3", lambda: False)
    assert g.main(["--pa3"]) == 1
    monkeypatch.setattr(g, "pa3", lambda: True)
    assert g.main(["--pa3"]) == 0
