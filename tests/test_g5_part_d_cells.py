"""G5 / T113 Part D D1's cell runner (evidence/g5-readout-variants.md D.1, D.2, D.4, D.5).

The reds that must be able to fire before any D1 cell is read. Two kinds of pin live here:

  * the measured facts of the feature path -- what the reset form and the carried form actually
  do, what
    the episode-boundary reset restores, and that two independently built pools agree at every step.
    Measured on fixture recording seed 20261003 (SEEDS.md's fixture-RNG row, never a recording
    seed);
    they are what D.1 and note (e) assert.
  * the instrument reds: the same-rows halt, the wrong-width halt, note (f)'s PRODNEUTER == NOTUNE
    invariant, the shuffle's reach, the S1 stateless HALT clause, the seed-block refusal, and the
    reader's
    integration over a fabricated record.

NOTHING here reads a registered seed block. Every number is a fixture diagnostic, disclosed as such;
the registered D1 confirmatory set (11410-11417) is untouched.
"""

import numpy as np
import pytest

from snn_data_lake.instruments import g5_part_d_cells as cells
from snn_data_lake.instruments import g5_part_d_reading as rd
from snn_data_lake.instruments import taught_problems as tp
from snn_data_lake.instruments.g5_readout_cells import pd_pool, pool_features

FIX = 20261003  # the fixture RNG seed (SEEDS.md), never a recording seed
CUE, PROBE = "memorize the plan now", "oak fox"


def _f(pool, text):
    return np.asarray(pool_features(pool, text))


# --------------------------------------------------------------------------------------------------
# The measured feature path (D.1, note (e))
# --------------------------------------------------------------------------------------------------
def test_the_reset_form_is_stateless_per_turn_and_the_carried_form_is_not():
    """The reset form's probe vector is bit-identical to a fresh pool's, so a reset-form reader is
    stateless per turn -- note (e)'s premise for the S1 exact-bound clause. The carried form's is
    not."""
    reset = pd_pool(FIX, "reset")
    _f(reset, CUE)
    after_cue = _f(reset, PROBE)
    fresh = _f(pd_pool(FIX, "reset"), PROBE)
    assert np.array_equal(after_cue, fresh), "the reset form carried state across turns"

    carry = pd_pool(FIX, "carry")
    _f(carry, CUE)
    assert not np.array_equal(_f(carry, PROBE), fresh), "the carried form lost its state"


def test_neuter_carry_is_not_stateless_even_though_its_recurrent_weights_are_zero():
    """G2 #162 point 1 / note (e): under CARRY the zeroed-recurrence pool still carries
    v = v*lam + injected, so it is the S0C1 attribution control and NOT the stateless arm."""
    neuter = pd_pool(FIX, "neuter_carry")
    assert float(np.abs(np.asarray(neuter.w)).max()) == 0.0, "the neuter did not zero it"
    _f(neuter, CUE)
    assert not np.array_equal(_f(neuter, PROBE), _f(pd_pool(FIX, "neuter_carry"), PROBE))
    assert np.array_equal(
        _f(pd_pool(FIX, "neuter_reset"), PROBE), _f(pd_pool(FIX, "reset"), PROBE)
    ), "neuter_reset must equal the reset form at T_in = 1 (note (e))"


def test_the_episode_boundary_reset_restores_a_fresh_pools_probe_vector():
    """D.1: CARRY carries within an episode and resets at the episode boundary (the episode tuple is
    the
    boundary unit). `reset_carriers()` at the boundary restores a fresh pool's probe vector."""
    pool = pd_pool(FIX, "carry")
    _f(pool, CUE)
    inside = _f(pool, PROBE)
    fresh = _f(pd_pool(FIX, "carry"), PROBE)
    pool.reset_carriers()
    assert np.array_equal(_f(pool, PROBE), fresh)
    assert not np.array_equal(inside, fresh)


def test_two_independently_built_pools_agree_at_every_step():
    """The property the same-rows check rests on: the feature path is a deterministic function of
    (recording seed, mode, text sequence), so two pools give identical features at every step."""
    texts = (CUE, PROBE, "set a timer now", PROBE)
    for mode in ("carry", "reset", "neuter_carry", "neuter_reset"):
        a, b = pd_pool(FIX, mode), pd_pool(FIX, mode)
        bad = [t for t in texts if not np.array_equal(_f(a, t), _f(b, t))]
        assert not bad, f"mode {mode} diverged at {bad}"


def test_the_label_neurons_in_is_256_and_the_runner_derives_it():
    """D.1: the pool's unit count is the label neurons' n_in. Derived from the path, never typed."""
    drive = cells._Drive(FIX, "carry", tp.PROBLEMS["i.S0C1"])
    turns = drive.walk(tp.PROBLEMS["i.S0C1"].episodes(FIX, "future")[0])
    assert len(turns[-1].x) == cells.PD_H


# --------------------------------------------------------------------------------------------------
# The instrument reds
# --------------------------------------------------------------------------------------------------
def test_a_wrong_width_path_halts_the_cell():
    with pytest.raises(cells.CellHalt):
        cells._n_in([[cells.Turn("t", [0.0] * 8, True, "GREET")]])


def test_the_same_rows_check_halts_when_a_mode_saw_different_rows():
    """D.2: the same-rows check is REUSED from g5_readout_cells. A mode that saw a different stream
    would make the arm comparison a comparison of streams, so it halts rather than reporting."""
    problem = tp.PROBLEMS["i.S0C1"]
    teach = cells._teach_stream(problem, FIX, 1)
    paths = {"carry": cells._path(FIX, "carry", problem, teach, [])}
    assert cells._same_rows(paths, teach) == len(teach)
    with pytest.raises(cells.CellHalt):
        cells._same_rows(paths, teach[:-1])


def test_prodneuter_equals_notune_through_the_real_cell_path():
    """note (f): with m pinned to 0, `interact`'s scalar branch deposits d = 0, so the
    producer-neutered
    learner IS the untuned learner. The committed reader asserts their equality exactly, so the
    runner
    must produce them as one learner -- measured through the real cell path, not a fixture."""
    for seed in (FIX, 20261004):
        rec = cells.pd_cell(seed, "S0C0", epochs=4, log=lambda *_: None)
        assert rec["RSTDP_BRAIN_PRODNEUTER"] == rec["RSTDP_NOTUNE"], seed


def test_pd_cell_is_deterministic():
    rec = cells.pd_cell(FIX, "S0C1", epochs=3, log=lambda *_: None)
    again = cells.pd_cell(FIX, "S0C1", epochs=3, log=lambda *_: None)
    body = {k: v for k, v in rec.items() if not k.startswith("_")}
    assert body == {k: v for k, v in again.items() if not k.startswith("_")}
    assert rec["_same_rows"] == again["_same_rows"] and rec["_shift_f"] == again["_shift_f"]


def test_som0_and_som_ctrl_are_one_control_and_it_is_not_the_som():
    """D.4 PD-R3: SOM's own control must be NOT MET for SOM to replace the incumbent. SOM0 (the
    width
    at zero) must be a different reader from SOM, or the control is the method itself (P8)."""
    rec = cells.pd_cell(FIX, "S0C1", epochs=4, log=lambda *_: None)
    assert rec["SOM0"] == rec["SOM_CTRL"]
    assert rec["SOM0"] != rec["SOM"], "SOM0 read the same as SOM: the control is the method"


def test_the_incumbents_teaching_reaches_the_untuned_neurons_on_the_sanity_cell():
    """PD-R0's positive control, reachable: on S0C0 (the registered sanity cell) the incumbent's
    label
    teaching beats the untuned neurons. A FIXTURE diagnostic, disclosed -- not a registered read."""
    rec = cells.pd_cell(FIX, "S0C0", epochs=8, log=lambda *_: None)
    assert rec["RSTDP_D"] > rec["RSTDP_NOTUNE"]


def test_the_time_shuffle_of_the_teaching_signal_reaches_the_arm():
    """D.2's shifted arm must be able to lose, or PD-R1's shuffle conjunct cannot fire (P8: a red
    that
    cannot fire is not a red)."""
    rec = cells.pd_cell(FIX, "S0C0", epochs=20, log=lambda *_: None)
    assert rec["RSTDP_BRAIN_SHUF"] != rec["RSTDP_BRAIN"]


def test_cert_for_halts_on_a_stateless_s1_solve():
    """D.4's S1 clause (note (e)'s correction): a PROVEN-STATELESS reader solving a plain S1 cell
    reads
    HALT (hidden state), never 'construction wrong'. The stateless arms are neuter_reset and
    EXPANSION_H
    (measured above); nothing else may trigger it."""
    problem = tp.PROBLEMS["i.S1C1"]
    held = list(problem.episodes(FIX, "future"))
    probe_label = held[-1][-1].label  # each episode's probe is its last turn
    wrong = [("GREET", "GOAL")] * len(held)
    perfect = [(probe_label, probe_label)] * len(held)
    # cert_for reads EVERY stateless arm; an arm with no read is refused (solved's empty-read red)
    cert = cells.cert_for(
        "S1C1",
        FIX,
        reads={"EXPANSION_H": perfect, "RSTDP_D@neuter_reset": perfect},
    )
    assert cert["stateless_solved_s1"] is True
    with pytest.raises(rd.ReadHalt):
        rd.read_d1({FIX: {"S1C1": {a: 0.5 for a in rd.ALL_ARMS}}}, cert, log=lambda *_: None)
    ok = cells.cert_for(
        "S1C1",
        FIX,
        reads={"EXPANSION_H": wrong, "RSTDP_D@neuter_reset": wrong},
    )
    assert ok["stateless_solved_s1"] is False


def test_cert_for_reads_the_expansions_certified_h_solve_only_on_s0c1():
    cert = cells.cert_for("S0C0", FIX, epochs=2)
    assert cert["expansion_solves_s0c1_at_64"] is None  # PD-R2 is not read on the sanity cell


def test_a_non_fixture_seed_block_is_refused_without_the_leads_go():
    """note (d): a registered seed block runs only on the lead's explicit go. 11410-11417 is the D1
    confirmatory set and is HELD; 11500-11507 was spent by the clean PD-P1."""
    assert cells.main(["--seeds", "11410", "11417", "--cells", "S0C0", "--epochs", "1"]) == 2
    assert cells.main(["--seeds", "11500", "11507", "--cells", "S0C0", "--epochs", "1"]) == 2


# --------------------------------------------------------------------------------------------------
# The reader (D.4) over a fabricated record
# --------------------------------------------------------------------------------------------------
def test_the_reader_reads_a_fabricated_record_of_the_shape_the_runner_produces():
    """The runner's output shape is the committed reader's input shape (record 2026-10-08 D.4). A
    fabricated record with the gaps registered in the evidence file reads every bar MET."""
    seeds = tuple(
        range(20261003, 20261011)
    )  # 8 fixture RNG seeds (SEEDS.md), never recording seeds
    up = (0.30, 0.25, 0.35, 0.28, 0.32, 0.27, 0.31, 0.26)
    rec = {}
    for k, s in enumerate(seeds):
        rec[s] = {cell: {a: 0.4 + 0.01 * (k % 3) for a in rd.ALL_ARMS} for cell in rd.CELLS}
    gaps = {
        ("S0C0", "RSTDP_D"): 1.0,
        ("S0C1", "RSTDP_BRAIN"): 1.0,
        ("S0C1", "RSTDP_D"): 1.0,
        ("S0C1", "DELTA"): 2.0,
        ("S1C0", "RSTDP_D"): 1.0,
        ("S1C1", "RSTDP_D"): 1.0,
    }
    for (cell, arm), scale in gaps.items():
        for k, s in enumerate(seeds):
            rec[s][cell][arm] += up[k] * scale
    v = rd.read_d1(
        rec,
        {"expansion_solves_s0c1_at_64": True, "stateless_solved_s1": False},
        log=lambda *_: None,
    )
    assert v["PD-R0"] and v["PD-R1"] and v["PD-R1b"] and v["PD-R3"]["DELTA"] == "REPLACES"
