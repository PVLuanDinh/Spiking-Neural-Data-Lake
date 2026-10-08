"""G5 / T113's instrument reds (specs/022 evidence/g5-readout-variants.md 2.4, 2.6): the bare
self-check (a deterministic runner, the frozen generator, the reading's NOT MET cases). Part A's
trace code and its 2.7a gate were deleted on the gate's null (commit 4ecc90d9 keeps them as run);
Part B's coupling, arms and planted checks were deleted on B-R1 NOT MET (record 3.6)."""

import os

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.instruments import g5_readout_cells as g  # noqa: E402


@pytest.fixture(autouse=True)
def lake(monkeypatch, tmp_path):
    """A decoy live lake (the module's LIVE_LAKE seam) so no test snapshots the real 3.3 GB lake.
    Returns `move()`: a same-size, same-mtime rewrite of the decoy's brain_lake.bin -- the change
    the retired mtime stamp could not see."""
    root = tmp_path / "decoy_lake"
    root.mkdir()
    f = root / "brain_lake.bin"
    f.write_bytes(b"AAAAAAAA")
    monkeypatch.setattr(g, "LIVE_LAKE", str(root))

    def move():
        st = os.stat(f)
        f.write_bytes(b"BBBBBBBB" if f.read_bytes() == b"AAAAAAAA" else b"AAAAAAAA")
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))

    return move


def test_a_same_size_same_mtime_rewrite_is_a_lake_change_the_old_stamp_missed(lake):
    # the retired stamp read only the mtimes of the home and brain_lake.bin; the snapshot digests
    # content
    def stamp():
        return [os.stat(os.path.join(g.LIVE_LAKE, n)).st_mtime_ns for n in (".", "brain_lake.bin")]

    old = stamp()
    before = g.lake_snapshot(g.LIVE_LAKE)
    lake()
    assert stamp() == old
    with pytest.raises(g.LiveLakeMoved, match="brain_lake.bin"):
        g._lake_unmoved(before, "a test stage")
    g._lake_unmoved(g.lake_snapshot(g.LIVE_LAKE), "a still stage")  # an unchanged lake passes


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
        with monkeypatch.context() as m:  # undone INSIDE the home, never after it is deleted
            m.setenv("NORD_HOME", str(other))
            with pytest.raises(LiveHomeRefused, match="not the fresh one"):
                g.check_fresh_home(fresh)


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


def test_the_live_lake_check_runs_after_each_stage_and_survives_an_exception(lake):
    # Codex cp4 #13: a RECORD-stage violation must halt BEFORE the arm stage launches
    from concurrent.futures import ThreadPoolExecutor

    ran = []

    def cell(args):
        ran.append(args[1])
        if args[1] == "RECORD":
            lake()  # the RECORD stage changes the lake
        return _fake_cell(args)

    with pytest.raises(g.LiveLakeMoved, match="RECORD stage"):
        g.run_cells(
            (1,), ("DELTA",), workers=1, log=lambda *_: None, pool=ThreadPoolExecutor, cell=cell
        )
    assert ran == ["RECORD"]  # the arm stage never launched

    # an exception in a stage while the lake moved reports the move, not the hidden exception
    def boom(args):
        if args[1] != "RECORD":
            lake()
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


@pytest.mark.parametrize("terminal", [KeyboardInterrupt, SystemExit])
def test_a_terminal_exception_still_gets_the_lake_check_and_is_never_replaced(lake, terminal):
    # Codex g5_b lens-1 m3: KeyboardInterrupt (a BaseException) bypassed the lake check; with the
    # lake moved it must report the move, with the lake still it must propagate, not use a reserve
    from concurrent.futures import ThreadPoolExecutor

    def stop(args):
        raise terminal

    def stop_and_move(args):
        lake()
        raise terminal

    with pytest.raises(g.LiveLakeMoved, match="raised"):
        g.run_cells(
            (1,),
            ("DELTA",),
            workers=1,
            log=lambda *_: None,
            pool=ThreadPoolExecutor,
            cell=stop_and_move,
        )
    lines = []
    with pytest.raises(terminal):
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


def _delta_minus_batch(neuter):
    # the B-R0 shape through the surviving wiring: arm_parts -> run_instrumented, brain-free
    diffs = []
    for s in g.PA3_SEEDS[:6]:
        ch = g.oracle_chain(s)
        M = 1 + max(c for _, c in ch)
        batch, _ = g.arm_parts("BATCH", M, None, s)
        delta, _ = g.arm_parts("BATCH" if neuter else "DELTA", M, None, s)
        diffs.append(
            g.run_instrumented(s, ch, delta, M)["live_acc"]
            - g.run_instrumented(s, ch, batch, M)["live_acc"]
        )
    return g.reading("DELTA - BATCH", diffs)


def test_delta_minus_batch_positive_control_is_met_and_a_neutered_delta_is_not():
    # Codex g5_b lens-1 m1: the deletion left no DELTA - BATCH test through the real arm wiring
    met, _, x = _delta_minus_batch(neuter=False)
    assert met and x["nontied"] >= g.MIN_NONTIED
    met0, c0, _ = _delta_minus_batch(neuter=True)  # DELTA wired as BATCH: contrast identically 0
    assert not met0 and c0.degenerate


def test_run_cells_persists_per_seed_records_that_rederive_the_contrast_and_refuse_tampering(
    monkeypatch, tmp_path
):
    # the confirmatory read could not be re-derived (per-seed contrasts were never persisted):
    # run_cells now writes the recorded sequences and every cell as json + sha256, and load_records
    # verifies it
    from concurrent.futures import ThreadPoolExecutor

    def cell(args):
        out = _fake_cell(args)
        out["live_acc"] = 0.5 + 0.01 * args[0] + (0.02 if args[1] == "DELTA" else 0.0)
        return out

    path = tmp_path / "records.json"
    rec, cells, rep = g.run_cells(
        (1, 2, 3, 4, 5, 6),
        ("DELTA", "BATCH"),
        workers=1,
        log=lambda *_: None,
        pool=ThreadPoolExecutor,
        cell=cell,
        record_path=str(path),
    )
    loaded = g.load_records(str(path))
    import json

    assert loaded["cells"] == json.loads(json.dumps(cells))
    assert loaded["replaced"] == json.loads(json.dumps(rep))
    assert sorted(loaded["recorded"]) == sorted(str(s) for s in rec)
    a = g.contrast_cells(loaded["cells"], "DELTA", "BATCH")
    assert a[2]["mean"] == pytest.approx(0.02)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sha256"):
        g.load_records(str(path))


def test_same_rows_refuses_a_nan_floor_or_persistence():
    # Codex g5_b lens-2: abs(NaN) > tol is False, so a NaN floor passed the comparison
    nan = float("nan")
    row = {
        "seed": 1,
        "arm": "DELTA",
        "widened": 0,
        "cats_hash": "x",
        "cols_hash": "y",
        "live_pers": 0.9,
        "floors": [(0.5, 0.9)],
    }
    with pytest.raises(g.SameRowsHalt, match="refit floors"):
        g.same_rows([row, {**row, "arm": "B", "floors": [(nan, 0.9)]}])
    with pytest.raises(g.SameRowsHalt, match="live persistence"):
        g.same_rows([row, {**row, "arm": "B", "live_pers": nan}])


def test_brain_cell_itself_refuses_a_home_that_is_not_the_one_it_was_given(monkeypatch, tmp_path):
    # Codex g5_b lens-2 (bf92393c mutation): the mismatch must be refused through brain_cell, before
    # any Brain is built, not only by the helper
    import contextlib

    import snn_data_lake.engine.brain as brain
    import snn_data_lake.substrate.home as home

    other = tmp_path / "populated_B"
    other.mkdir()
    real = home.throwaway_home

    @contextlib.contextmanager
    def swapped():
        with real() as fresh, monkeypatch.context() as m:
            m.setenv("NORD_HOME", str(other))  # yields A, but the process points at B
            yield fresh

    built = []
    monkeypatch.setattr(home, "throwaway_home", swapped)
    monkeypatch.setattr(brain, "make_brain", lambda **kw: built.append(kw))
    with pytest.raises(home.LiveHomeRefused, match="not the fresh one"):
        g.brain_cell((20261003, "DELTA", [0, 1, 0]))
    assert built == []


def test_a_lake_move_during_a_successful_arm_stage_halts(lake):
    # Codex g5_b lens-2 (d7d20f63 mutation): RECORD leaves the lake still, an arm moves it and
    # returns normally -- the post-arm check, not an exception path, must catch it
    from concurrent.futures import ThreadPoolExecutor

    def cell(args):
        if args[1] != "RECORD":
            lake()  # the arm stage changes the lake and returns normally
        return _fake_cell(args)

    with pytest.raises(g.LiveLakeMoved, match="arm stage"):
        g.run_cells(
            (1,),
            ("DELTA",),
            workers=1,
            log=lambda *_: None,
            pool=ThreadPoolExecutor,
            cell=cell,
        )


def test_an_arm_only_fallback_perception_replaces_the_whole_seed(monkeypatch):
    # Codex g5_b lens-2 (35f92a98 mutation): RECORD is unified, one arm is not; the arm-level check
    # must reject it, and the whole seed (recording and every arm) is replaced from the reserve
    from concurrent.futures import ThreadPoolExecutor

    def cell(args):
        out = _fake_cell(args)
        out["unified"] = not (args[0] == 2 and args[1] == "BATCH")  # seed 2's BATCH arm falls back
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
    assert rep == [(2, 2, 9)] and any("fallback perception" in s for s in lines)


def test_same_rows_halts_on_a_majority_only_floor_difference():
    # Codex g5_b lens-2 (fc3409d2 mutation): comparing only the persistence half of each floor pair
    # passed a moved MAJORITY floor
    row = {
        "seed": 1,
        "arm": "DELTA",
        "widened": 0,
        "cats_hash": "x",
        "cols_hash": "y",
        "live_pers": 0.9,
        "floors": [(0.5, 0.9), (0.6, 0.92)],
    }
    with pytest.raises(g.SameRowsHalt, match="refit floors"):
        g.same_rows([row, {**row, "arm": "B", "floors": [(0.5, 0.9), (0.7, 0.92)]}])


def test_tensor_digest_sees_a_change_at_any_position():
    # Codex g5_b lens-2 (d0acd8f1 mutation): a digest of only the leading bytes passed
    beta = torch.randn(5, 4, generator=torch.Generator().manual_seed(20261003))
    for idx in ((0, 1), (2, 2), (4, 3)):
        nudged = beta.clone()
        nudged[idx] = torch.nextafter(nudged[idx], torch.tensor(float("inf")))
        assert g.tensor_digest(nudged) != g.tensor_digest(beta), idx


# ---- Part D D1: the brain producer and PD-P1 (see the 2026-10-08 notes) ----
def test_producer_factor_is_signed_zero_at_zero_rpe_and_uses_the_one_modulator():
    from snn_data_lake.faculties.competence_executive import DopamineModulator

    p = g.BrainProducer(seed=3)
    assert isinstance(p.modulator, DopamineModulator) and p.modulator._nuc is p.nucleus
    assert p.factor("s0", 0.0) == 0.0  # r == expected (V = 0): zero RPE rests on the tonic
    assert p.factor("s0", 1.0) > 0.0  # a burst: reinforce
    assert (
        g.BrainProducer(seed=3).factor("s0", -1.0) < 0.0
    )  # a wrong answer dips even at V = 0 (+-1 coding)
    p2 = g.BrainProducer(seed=3)
    for _ in range(20):
        p2.factor("s0", 1.0)  # V("s0") learns to expect the outcome
    assert p2.factor("s0", 0.0) < 0.0  # an omitted expected reward: a dip, the sign interact needs
    lr = p.modulator.LR_MIN + (p.modulator.LR_MAX - p.modulator.LR_MIN) * p.nucleus.da_level(0.3)
    assert (
        abs((lr / p.modulator.tonic_lr - 1.0) - g.BrainProducer.to_signed(p.modulator, lr)) < 1e-12
    )


def test_producer_rpe_and_the_modulators_agree_and_a_perturbed_expectation_is_refused():
    p = g.BrainProducer(seed=1)
    p.factor(
        "s0", 1.0
    )  # the build asserts the nucleus's TD rpe equals the modulator's rpe, every call
    g.BrainProducer.agree(0.25, 1.0, 0.75)
    with pytest.raises(AssertionError):
        g.BrainProducer.agree(
            0.25, 1.0, 0.75 + 1e-3
        )  # a perturbed expectation must fail the equality


def test_neutered_producer_gets_the_outcome_but_emits_zero_and_learns_nothing():
    p = g.BrainProducer(seed=2, neutered=True)
    assert [p.factor("s0", 1.0) for _ in range(5)] == [0.0] * 5
    assert p.nucleus.v == {}  # V frozen at 0
    assert p.modulator._gain_n == 5  # the outcome WAS delivered to the modulator


def test_producer_never_sees_the_label_and_the_learner_only_gets_scalars(monkeypatch):
    import inspect

    assert list(inspect.signature(g.BrainProducer.factor).parameters) == ["self", "stim_key", "r"]
    seen = []
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as L

    real = L.interact
    monkeypatch.setattr(
        L, "interact", lambda s, x, fb, **k: (seen.append(fb), real(s, x, fb, **k))[1]
    )
    g.pd_p1_arm(11472, neutered=False, epochs=2)
    assert seen and all(isinstance(fb, float) for fb in seen)  # never the str label


def test_pd_p1_brain_beats_the_neutered_producer_and_the_contrast_can_say_not_met():
    met, c, x = g.pd_p1()
    assert met, (c.t, x)  # the producer ablation can fire: PD-P1 MET
    same = g.reading(
        "self",
        [g.pd_p1_arm(s, neutered=True) - g.pd_p1_arm(s, neutered=True) for s in g.PD_P1_SEEDS],
    )
    assert not same[0], (
        "BRAIN-free control: an arm minus itself is degenerate and must read NOT MET"
    )


def test_pd_p1_seeds_are_the_registered_block_and_cli_exits_nonzero_when_not_met(monkeypatch):
    assert list(g.PD_P1_SEEDS) == list(range(11472, 11480))
    monkeypatch.setattr(
        g,
        "pd_p1",
        lambda: (
            False,
            g.contrast("x", [0.0] * 8, g.T_BAR),
            {"pattern": "0" * 8, "mean": 0.0, "skew": 0.0, "nontied": 0},
        ),
    )
    assert g.main(["--pd-p1"]) == 1
