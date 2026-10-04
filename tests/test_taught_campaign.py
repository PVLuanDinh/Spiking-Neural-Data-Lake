"""G2 / T110's taught campaign (instruments/taught_campaign.py, P3): the G-R2 reading and a red per row, the
driver's preflight refusals, the front-ends (determinism, the default texts byte-identical, teach-only medians
for C1-d), the readers and the attempt on short schedules, B3 on a live reader with its stateful red, and B8's
channel identity under throwaway_home() (the one test that builds a Brain). Seeds: scratch 8900-8901 and the
reds block 8500 only."""

import pytest

from snn_data_lake.instruments import delayed_cue as dc
from snn_data_lake.instruments import taught_campaign as tc
from snn_data_lake.instruments import taught_problems as tp

SCRATCH = 8900


def test_reading_pass_and_every_row_red():
    assert tc.reading(tc.PASS_FLAGS).startswith("PASS: S0C0")
    reds = tc.reading_reds()
    assert reds["all_fired"], reds["fired"]


@pytest.mark.parametrize("seeds,out", [((300,), "x"), ((7003,), "x"), ((12000,), "x"), ((8200,), None)])
def test_preflight_refusals(seeds, out, tmp_path):
    out = str(tc.REPO / "out") if out is None else str(tmp_path / out)
    with pytest.raises(PermissionError):
        tc.preflight(seeds, out, require_clean=False)


@pytest.mark.parametrize("sub", [None, "g2_out", "deep/er"])
def test_preflight_refuses_the_live_home_before_creating_anything(sub):
    # cp1 #1: OUT may never be the live home or a descendant; refused before any mkdir
    out = tc.LIVE_HOME if sub is None else tc.LIVE_HOME / sub
    existed = out.exists()
    with pytest.raises(PermissionError):
        tc.preflight((8200,), str(out), require_clean=False)
    assert out.exists() == existed


def test_isolation_failure_halts(monkeypatch):
    # cp1 #1: a moved live-home mtime after the last write raises, never reports
    stamps = iter([1.0, 2.0])
    monkeypatch.setattr(tc, "_live_mtime", lambda: next(stamps))
    before = tc._live_mtime()
    with pytest.raises(RuntimeError):
        tc._assert_isolated(before)


def test_preflight_refuses_an_arm_naming_future_d(tmp_path):
    arms = (("x", "c1d", "incumbent", "never", "future_d"),)
    with pytest.raises(PermissionError):
        tc.preflight((8200,), str(tmp_path), arms, require_clean=False)


def test_preflight_accepts_the_registered_campaign(tmp_path):
    tc.preflight(tc.CONFIRMATORY_SEEDS, str(tmp_path), require_clean=False)


def test_seed_blocks_inside_g2s_block_and_disjoint():
    blocks = [tc.DEV_SEEDS, tc.CONFIRMATORY_SEEDS, tc.RESERVE_SEEDS, (tc.RED_SEED,), tc.SCRATCH_SEEDS]
    flat = [s for b in blocks for s in b]
    assert len(flat) == len(set(flat)) and all(8000 <= s <= 8999 for s in flat)
    assert not set(flat) & tc.LIVE_SEEDS


def test_presized_deterministic_and_texts_default_byte_identical():
    p = tp.PROBLEMS["i.S0C1"]
    a = tc.presized_expansion(p, 16, SCRATCH)
    b = tc.presized_expansion(p, 16, SCRATCH, texts=tc.stimulus_texts(p, SCRATCH))
    assert a.features("oak fox") == b.features("oak fox")
    assert len(a.features("oak fox")) == 16 and min(a.features("oak fox")) >= 0.0


def test_presized_c1d_centres_on_teach_texts_only():
    p = tp.PROBLEMS["c1d"]
    texts = set(tc.stimulus_texts(p, SCRATCH))
    future = {dc.combo_text("c1d", it) for it in p.items(SCRATCH, "future")}
    assert len(texts) == 32 and not texts & future


def test_ngsnn_front_end_deterministic_cpu():
    p = tp.PROBLEMS["i.S0C1"]
    a = tc.ngsnn_expansion(p, 16, SCRATCH).features("oak fox")
    assert a == tc.ngsnn_expansion(p, 16, SCRATCH).features("oak fox") and len(a) == 16


def test_history_reader_reads_the_window():
    reader = tc.make_reader("history", tp.PROBLEMS["i.S1C0"], SCRATCH)
    turns = tuple(tp.Turn(t, r, None) for t, r in dc.episode_i((1, 0, 1)))
    assert reader.text(turns) == "pen box oak owl"
    one = tc.make_reader("history", tp.PROBLEMS["i.S1C0"], SCRATCH, window=1)
    assert one.text(turns) == "oak owl"


def test_incumbent_solves_s0c0_never_state_cells_short():
    s0c0, s1c0 = tp.PROBLEMS["i.S0C0"], tp.PROBLEMS["i.S1C0"]
    pos = tc.attempt(s0c0, tc.make_reader("incumbent", s0c0, SCRATCH), SCRATCH, 30)
    neg = tc.attempt(s1c0, tc.make_reader("incumbent", s1c0, SCRATCH), SCRATCH, 30)
    assert pos["stable"] and neg["never"] and neg["max_accuracy"] <= 0.5


def test_b3_identity_holds_for_incumbent_and_reds_for_history():
    s1c0 = tp.PROBLEMS["i.S1C0"]
    inc = tc.make_reader("incumbent", s1c0, SCRATCH)
    tc.attempt(s1c0, inc, SCRATCH, 10)
    assert tc.b3_identity(inc, s1c0)["ok"]
    hist = tc.make_reader("history", s1c0, SCRATCH)
    tc.attempt(s1c0, hist, SCRATCH, 20)
    assert not tc.b3_identity(hist, s1c0)["ok"]


def test_shuffled_teaching_never_teaches_the_truth():
    taught = tc._shuffled_labels(SCRATCH)
    true = [dc.GREET, dc.GOAL, dc.GREET, dc.GOAL]
    assert all(taught(true) != true for _ in range(20))


def test_attempt_judges_future_on_combos():
    p = tp.PROBLEMS["ac1d"]
    result = tc.attempt(p, tc.make_reader("incumbent", p, SCRATCH), SCRATCH, 5, judge_future=True)
    assert "future_solved" in result and 0.0 <= result["future_accuracy"] <= 1.0


def test_flags_from_rows_reads_a_planted_capacity_solve():
    rows = {"inc.S0C1": [{"met": False}], "inc.S0C0": [{"met": True}]}
    flags = tc.flags_from(rows, True, True, True)
    assert flags["incumbent_solves_s1_or_c1"] and tc.reading(flags).startswith("RED: construction wrong (a")


def test_b8_twin_equal_and_bare_class_reds():
    # builds a Brain: under throwaway_home(), live home mtime checked inside b8_twin
    ok = tc.b8_twin(SCRATCH, 3)
    assert ok["ok"], ok
    red = tc.b8_twin(SCRATCH, 3, bare=True)
    assert not red["conditions"]["twin_equal"]
