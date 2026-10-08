"""G2 / T110's taught campaign (instruments/taught_campaign.py, P3): the G-R2 reading and a red per
row, the driver's preflight refusals, the front-ends (determinism, the default texts byte-
identical, teach-only medians for C1-d), the readers and the attempt on short schedules, B3 on a
live reader with its stateful red, and B8's channel identity under throwaway_home() (the one
test that builds a Brain). Seeds: scratch 8900-8901 and the reds block 8500 only."""

import os

import pytest

from snn_data_lake.instruments import delayed_cue as dc
from snn_data_lake.instruments import taught_campaign as tc
from snn_data_lake.instruments import taught_problems as tp

SCRATCH = 8900


def test_reading_pass_and_every_row_red():
    assert tc.reading(tc.PASS_FLAGS).startswith("PASS: S0C0")
    reds = tc.reading_reds()
    assert reds["all_fired"], reds["fired"]


@pytest.mark.parametrize(
    "seeds,out", [((300,), "x"), ((7003,), "x"), ((12000,), "x"), (tuple(range(8200, 8208)), None)]
)
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
        tc.preflight(tc.CONFIRMATORY_SEEDS, str(out), require_clean=False)
    assert out.exists() == existed


def test_isolation_failure_halts(tmp_path):
    # cp1 #1 + P2: a same-size, same-mtime rewrite of the (decoy) live lake is caught by the
    # content digest (the old top-level (size, mtime_ns) stamp read it as untouched)
    from snn_data_lake.substrate.home import lake_snapshot

    lake = tmp_path / "brain_lake.bin"
    lake.write_bytes(b"AAAA")
    mtime = lake.stat().st_mtime_ns
    before = lake_snapshot(tmp_path)
    tc._assert_isolated(before, tmp_path)  # untouched: no raise
    lake.write_bytes(b"BBBB")
    os.utime(lake, ns=(mtime, mtime))
    assert lake.stat().st_size == 4 and lake.stat().st_mtime_ns == mtime
    with pytest.raises(RuntimeError):
        tc._assert_isolated(before, tmp_path)


def _ii_rows(seeds, override=None):
    rows = {arm[0]: [{"seed": s, "met": True} for s in seeds] for arm in tc.ARMS_II}
    for arm_id in override or ():
        rows[arm_id] = [{"seed": s, "met": False} for s in seeds]
    return rows


@pytest.mark.parametrize(
    "failing,prefix",
    [
        ((), "PASS (ii)"),
        (("inc.ii.xor",), "DISAGREEMENT"),
        (("inc.ii.twin",), "RED: (ii)'s separable"),
        (("pre.ii.xor",), "RED: (ii)'s presized"),
        (("pre.ii.twin",), "RED: (ii)'s presized"),
    ],
)
def test_reading_ii_rows_each_fire(failing, prefix):
    # cp1 #6: (ii) has its own reading; each row's planted failure gives that row's verdict
    seeds = (8900, 8901)
    assert tc.reading_ii(_ii_rows(seeds, failing), seeds).startswith(prefix)


def test_reading_ii_not_run_and_partial_evidence():
    assert tc.reading_ii(None, (8900,)).startswith("NOT RUN")
    rows = _ii_rows((8900, 8901))
    rows["pre.ii.twin"] = rows["pre.ii.twin"][:1]
    with pytest.raises(ValueError):
        tc.reading_ii(rows, (8900, 8901))


@pytest.mark.parametrize(
    "evidence",
    [
        {"verdict": False, "seeds": list(range(7000, 7008))},
        {"verdict": True, "seeds": [7000]},
        {"verdict": "yes"},
    ],
)
def test_ii_refused_without_g1c_construct_pass(evidence, tmp_path):
    with pytest.raises(PermissionError):
        tc.run_campaign((8900,), str(tmp_path), g1c_construct=evidence, require_clean=False)
    assert tc.g1c_construct_passed({"verdict": True, "seeds": list(range(7000, 7008))})


def _no_arms(*a, **k):
    raise AssertionError("an arm ran after a failed prerequisite")


@pytest.mark.parametrize("failure", ["construction", "b8", "p1_red", "campaign_red"])
def test_a_failed_prerequisite_halts_before_any_arm(failure, tmp_path, monkeypatch):
    # cp1 #2: no deferred halts -- the stage that fails stops the campaign before any arm is built
    monkeypatch.setattr(tc, "run_arm", _no_arms)
    monkeypatch.setattr(tc, "b8_twin", lambda s, e: {"ok": failure != "b8"})
    monkeypatch.setattr(tc.dc, "construction_reds", lambda: {"all_fired": failure != "p1_red"})
    monkeypatch.setattr(tc, "campaign_reds", lambda **k: {"all_fired": failure != "campaign_red"})
    if failure == "construction":
        monkeypatch.setattr(tc.dc, "construction_check", lambda seeds: {"verdict": False})
    record = tc.run_campaign((8900,), str(tmp_path / "out"), require_clean=False)
    assert record["verdict"].startswith(
        "RED: construction" if failure == "construction" else "HALT"
    )
    assert "rows" not in record and (tmp_path / "out" / "g2_campaign.json").exists()


def test_preflight_refuses_an_arm_naming_future_d(tmp_path):
    arms = (("x", "c1d", "incumbent", "never", "future_d"),)
    with pytest.raises(PermissionError):
        tc.preflight(tc.CONFIRMATORY_SEEDS, str(tmp_path), arms, require_clean=False)


def test_preflight_accepts_the_registered_campaign_only_after_the_amendment(tmp_path, monkeypatch):
    # cp1 #7: confirmation is refused until the dated pre-data amendment is entered
    monkeypatch.setattr(tc, "AMENDMENT", None)
    with pytest.raises(PermissionError):
        tc.preflight(tc.CONFIRMATORY_SEEDS, str(tmp_path), require_clean=False)
    with pytest.raises(PermissionError):
        tc.preflight(tc.RESERVE_SEEDS, str(tmp_path), require_clean=False)
    monkeypatch.setattr(tc, "AMENDMENT", "2026-10-04 test")
    tc.preflight(tc.CONFIRMATORY_SEEDS, str(tmp_path), require_clean=False)
    tc.preflight(tc.DEV_SEEDS, str(tmp_path), require_clean=False)


def test_dev_measurement_refuses_non_dev_seeds(tmp_path):
    with pytest.raises(PermissionError):
        tc.dev_measurement(str(tmp_path / "x.jsonl"), seeds=(8200,))


def test_dev_measurement_derives_on_a_planted_grid(tmp_path, monkeypatch):
    # the derivation rules on planted rows: H = 2 x the smallest passing H; E = max(400, 2 x
    # largest, /50)
    def planted(stage, kind, pid, seed, epochs, h, solves=True):
        stable = solves and not (pid == "c1d") and not (stage == "R1" and h == 16)
        return {
            "stage": stage,
            "arm": kind,
            "problem": pid,
            "seed": seed,
            "h": h,
            "epochs": epochs,
            "stable": stable,
            "never": not solves,
            "max_acc": 1.0,
            "to_stable": (100 if stage == "R1" else 230) if stable else None,
            "future_solved": False if pid == "c1d" else None,
            "future_acc": None,
        }

    monkeypatch.setattr(tc, "_dev_row", lambda *a, **k: planted(*a, **k))
    derived = tc.dev_measurement(str(tmp_path / "d.jsonl"), seeds=(8000,))
    assert derived["H"] == 64 and derived["E"] == 500 and derived["c1d_learned_at"] == []


def test_seed_blocks_inside_g2s_block_and_disjoint():
    blocks = [
        tc.DEV_SEEDS,
        tc.CONFIRMATORY_SEEDS,
        tc.RESERVE_SEEDS,
        (tc.RED_SEED,),
        tc.SCRATCH_SEEDS,
    ]
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
    assert tc.b3_on_stream(tc.make_reader("incumbent", s1c0, SCRATCH), s1c0, SCRATCH, 10)
    assert not tc.b3_on_stream(tc.make_reader("history", s1c0, SCRATCH), s1c0, SCRATCH, 20)


class _CueCatcher:
    # a hidden-state reader: it captures the cue from SEPARATE context reads, then answers the
    # probe from it
    def __init__(self):
        self.cue = None

    def route(self, turns):
        if turns[-1].role == "context" and turns[-1].text in dc.CUE:
            self.cue = dc.CUE.index(turns[-1].text)
        return dc.LABELS[self.cue or 0]

    def teach(self, turns, label):
        return None


def test_b3_judged_on_the_actual_stream_catches_a_cue_capturing_reader():
    # cp1 #10: a reader that captures cues during separate context reads solves S1C0 and must fail
    # B3
    s1c0 = tp.PROBLEMS["i.S1C0"]
    assert not tc.b3_on_stream(_CueCatcher(), s1c0, SCRATCH, 2)
    result = tc.attempt(s1c0, _CueCatcher(), SCRATCH, 12)
    assert result[
        "stable"
    ]  # it really solves the state cell, so only B3 stands between it and a PASS


def test_us_attempt_reads_fresh_labels_each_epoch_end():
    # cp1 #9: the attempt path itself gives every epoch-end read its own index, so U_S labels differ
    p = tp.PROBLEMS["U_S"]
    result = tc.attempt(p, tc.make_reader("incumbent", p, SCRATCH), SCRATCH, 4)
    assert result["validation_read_indices"] == [0, 1, 2, 3]
    labels = [
        tuple(e[-1].label for e in p.episodes(SCRATCH, "validation", read_index=i))
        for i in range(4)
    ]
    assert len(set(labels)) == 4 and result["never"]


def test_constant_cue_history_cannot_solve_s1c0():
    # cp1 #15: B6's constant-cue red -- the history text carries no cue, so S1C0 stays at chance
    s1c0 = tp.PROBLEMS["i.S1C0"]
    reader = tc.constant_cue_reader(s1c0, SCRATCH)
    turns = tuple(tp.Turn(t, r, None) for t, r in dc.episode_i((1, 0, 1)))
    assert reader.text(turns) == "jar box oak owl"
    result = tc.attempt(s1c0, reader, SCRATCH, 30)
    assert not result["stable"] and result["max_accuracy"] <= 0.5


def test_campaign_reds_stage_b6_constant_cue_and_b8_bare(monkeypatch):
    # cp1 #15: both registered controls are in the staged red set; a non-firing B8 bare red reads
    # not fired
    monkeypatch.setattr(tc, "attempt", lambda *a, **k: {"stable": False, "never": False})
    monkeypatch.setattr(tc, "b3_on_stream", lambda *a: False)
    monkeypatch.setattr(
        tc, "b8_twin", lambda s, e, bare=False: {"conditions": {"twin_equal": not bare}}
    )
    fired = tc.campaign_reds(epochs=5, h=8)["fired"]
    assert fired["b6_constant_cue"] and fired["b8_bare_class"]
    monkeypatch.setattr(
        tc, "b8_twin", lambda s, e, bare=False: {"conditions": {"twin_equal": True}}
    )
    reds = tc.campaign_reds(epochs=5, h=8)
    assert not reds["fired"]["b8_bare_class"] and not reds["all_fired"]


def test_shuffled_teaching_never_teaches_the_truth():
    taught = tc._shuffled_labels(SCRATCH)
    true = [dc.GREET, dc.GOAL, dc.GREET, dc.GOAL]
    assert all(taught(true) != true for _ in range(20))


def test_attempt_judges_future_on_combos():
    p = tp.PROBLEMS["ac1d"]
    result = tc.attempt(p, tc.make_reader("incumbent", p, SCRATCH), SCRATCH, 5, judge_future=True)
    assert "future_solved" in result and 0.0 <= result["future_accuracy"] <= 1.0


def _full_rows(seeds, override=None):
    rows = {arm[0]: [{"seed": s, "met": True} for s in seeds] for arm in tc.ARMS}
    for arm_id, met in (override or {}).items():
        rows[arm_id] = [{"seed": s, "met": met} for s in seeds]
    return rows


def test_flags_from_rows_reads_a_planted_capacity_solve():
    seeds = (8900, 8901)
    flags = tc.flags_from(_full_rows(seeds, {"inc.S0C1": False}), True, True, True, seeds=seeds)
    assert flags["incumbent_solves_s1_or_c1"] and tc.reading(flags).startswith(
        "RED: construction wrong (a"
    )
    assert tc.reading(tc.flags_from(_full_rows(seeds), True, True, True, seeds=seeds)).startswith(
        "PASS: S0C0"
    )


@pytest.mark.parametrize("rows", [{}, "one_arm_missing", "one_seed_missing"])
def test_flags_from_refuses_missing_evidence(rows):
    # cp1 #5: missing or partial arm rows never read as success
    seeds = (8900, 8901)
    if rows == "one_arm_missing":
        rows = _full_rows(seeds)
        del rows["pre.c1d"]
    elif rows == "one_seed_missing":
        rows = _full_rows(seeds)
        rows["inc.S0C0"] = rows["inc.S0C0"][:1]
    with pytest.raises(ValueError):
        tc.flags_from(rows, True, True, True, seeds=seeds)


@pytest.mark.parametrize("seeds", [(), (8200,) * 8, tuple(range(8200, 8207)), (8300, 8301)])
def test_preflight_refuses_empty_duplicate_and_partial_blocks(seeds, tmp_path):
    # cp1 #5: non-empty, distinct, and a confirmatory or reserve read is that WHOLE block
    with pytest.raises(PermissionError):
        tc.preflight(seeds, str(tmp_path), require_clean=False)


def test_b8_twin_equal_and_bare_class_reds():
    # builds a Brain: under throwaway_home(), live home mtime checked inside b8_twin
    ok = tc.b8_twin(SCRATCH, 3)
    assert ok["ok"], ok
    red = tc.b8_twin(SCRATCH, 3, bare=True)
    assert not red["conditions"]["twin_equal"]


def test_b8_refuses_an_unset_home_before_building_a_brain(monkeypatch):
    # ruling 16: B8 calls the shared guard inside its home block; with NORD_HOME unset it must
    # refuse BEFORE make_brain
    import contextlib

    from snn_data_lake.engine import brain as brain_mod
    from snn_data_lake.substrate import home as home_mod

    @contextlib.contextmanager
    def no_home():
        monkeypatch.delenv("NORD_HOME", raising=False)
        yield

    def build(*a, **k):
        raise AssertionError("make_brain reached without the guard")

    monkeypatch.setattr(home_mod, "throwaway_home", no_home)
    monkeypatch.setattr(brain_mod, "make_brain", build)
    with pytest.raises(home_mod.LiveHomeRefused):
        tc.b8_twin(8900, 1)
