"""G2 / T110's problem contract (instruments/taught_problems.py, P2): the Problem record, its
splits as episodes (orders, presentation counts, the C1-d disjointness, the future_d guard), the
U / U_S fixtures read through a stateless reader, the predicate (B9), the budget (B11) and
learning OFF (B12), with every P2 red. No Brain is built; seeds are the dev block 8000-8003 and
the exposed scratch seed 8900."""

import pytest

from snn_data_lake.instruments import delayed_cue as dc
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments import taught_problems as tp

DEV = (8000, 8001, 8002, 8003)


def test_registry_covers_every_registered_problem():
    assert set(tp.PROBLEMS) == {
        "i.S0C0",
        "i.S1C0",
        "i.S0C1",
        "i.S1C1",
        "c1d",
        "ac1d",
        "ii.xor",
        "ii.twin",
        "aii",
        "U",
        "U_S",
    }
    assert tp.PROBLEMS["i.S1C0"].delay_turns == 1 and tp.PROBLEMS["c1d"].delay_turns == 0


@pytest.mark.parametrize("pid,n", [("i.S0C1", 32), ("c1d", 32), ("ii.xor", 16), ("U", 32)])
def test_validation_presentation_counts(pid, n):
    assert len(tp.PROBLEMS[pid].episodes(8000, "validation")) == n


def test_combo_future_read_is_48_and_disjoint_from_validation():
    p = tp.PROBLEMS["c1d"]
    assert p.presentations(8000, "future") == 48
    val = {e[-1].text for e in p.episodes(8000, "validation")}
    fut = {e[-1].text for e in p.episodes(8000, "future")}
    taught = {e[-1].text for epoch in p.teach_epochs(8000, 1) for e in epoch}
    assert not (val & fut) and not (val & taught) and not (fut & taught)


def test_future_d_refused_to_learners_and_open_to_construction_checks():
    p = tp.PROBLEMS["c1d"]
    with pytest.raises(PermissionError):
        p.episodes(8000, "future_d")
    assert len(p.construction_episodes(8000, "future_d")) == 48
    with pytest.raises(ValueError):
        tp.PROBLEMS["i.S0C1"].construction_episodes(8000, "future_d")


def test_future_d_not_reachable_through_any_learner_path():
    # cp1 #11: neither the generic read helper nor items() grants future_d
    p = tp.PROBLEMS["c1d"]
    with pytest.raises(PermissionError):
        tp.read_split(p, 8000, "future_d", lambda turns: dc.GREET)
    with pytest.raises(PermissionError):
        p.items(8000, "future_d")
    with pytest.raises(TypeError):
        tp.read_split(p, 8000, "future_d", lambda turns: dc.GREET, construction_check=True)


def test_episode_shape_and_labels_on_probes_only():
    episode = tp.PROBLEMS["i.S1C0"].episodes(8000, "validation")[0]
    assert [t.role for t in episode] == ["context", "context", "probe"]
    assert episode[0].label is None and episode[1].label is None and episode[2].label in dc.LABELS


def test_orders_deterministic_and_split_streams_differ():
    p = tp.PROBLEMS["i.S0C1"]
    a = p.episodes(8000, "validation")
    assert a == p.episodes(8000, "validation")
    assert [e[-1].text for e in a] != [e[-1].text for e in p.episodes(8000, "future")]
    assert tp.teaching_sequence(p, 8000, 3) == tp.teaching_sequence(p, 8000, 3)
    assert len(tp.teaching_sequence(p, 8000, 3)) == 24


def test_teach_order_is_one_stream_like_ii_schedule():
    # (ii)'s wrapper teaches the same order as inseparable_mapping._schedule (one stream, P2)
    seq = [e[-1].text for epoch in tp.PROBLEMS["ii.xor"].teach_epochs(7100, 5) for e in epoch]
    ref = [im.TEXTS[uv] for order in im._schedule(7100, 5) for uv in order]
    assert seq == ref


@pytest.mark.parametrize("seed", DEV)
def test_u_never_solved_by_a_stateless_reader(seed):
    p = tp.PROBLEMS["U"]
    for answer in dc.LABELS:  # every constant reader and the probe-keyed reader below
        reads = tp.read_split(p, seed, "validation", lambda turns, a=answer: a)
        assert not tp.solved(p, seed, "validation", reads)
    unit = tp.canonical_incumbent(seed)
    reads = tp.read_split(p, seed, "validation", lambda turns: unit.route_text(turns[-1].text))
    assert not tp.solved(p, seed, "validation", reads)


@pytest.mark.parametrize("pid", ["U", "U_S", "i.S0C1", "c1d"])
def test_a_label_reading_reader_sees_no_labels(pid):
    # cp1 #4: route(turns) receives label-free turns, so reading the target cannot solve anything
    p = tp.PROBLEMS[pid]
    seen = []

    def cheat(turns):
        seen.extend(t.label for t in turns)
        return turns[-1].label

    reads = tp.read_split(p, 8000, "validation", cheat)
    assert set(seen) == {None} and not tp.solved(p, 8000, "validation", reads)


def test_us_reads_draw_fresh_labels():
    p = tp.PROBLEMS["U_S"]
    r0 = [e[-1].label for e in p.episodes(8000, "validation", read_index=0)]
    r1 = [e[-1].label for e in p.episodes(8000, "validation", read_index=1)]
    assert r0 != r1 and len(r0) == 32


def test_us_teaching_runs_on_its_own_stream():
    # cp1 #8: U_S had no teaching labels (labelling None); a full teaching attempt must run
    p = tp.PROBLEMS["U_S"]
    seq = tp.teaching_sequence(p, 8000, 3)
    assert len(seq) == 24 and set(lab for _, _, lab in seq) == set(dc.LABELS)
    assert seq == tp.teaching_sequence(p, 8000, 3)
    teach_labels = [lab for _, _, lab in seq[:8]]
    assert teach_labels != dc.us_labels(8000, "validation", 0, 8)  # its own stream, not a read's
    unit = tp.canonical_incumbent(8000)
    for epoch in p.teach_epochs(8000, 2):
        for episode in epoch:
            unit.learn_turn(episode[-1].text, None, feedback=episode[-1].label)


def test_solved_refuses_a_short_read():
    p = tp.PROBLEMS["i.S0C1"]
    with pytest.raises(ValueError):
        tp.solved(p, 8000, "validation", [(dc.GREET, dc.GREET)] * 31)


def test_stability_helpers():
    assert tp.solves_stably([False] + [True] * 10)
    assert not tp.solves_stably([True] * 9 + [False])
    assert tp.never_solves([False] * 5) and not tp.never_solves([False, True])


def test_b9_b11_b12():
    assert tp.check_b9()["ok"]
    result = tp.check_b11()
    assert result["ok"]
    assert result["floors"] == {
        "i.S0C0": 8,
        "i.S1C0": 8,
        "i.S0C1": 8,
        "i.S1C1": 8,
        "c1d": 32,
        "ac1d": 32,
        "ii.xor": 4,
        "ii.twin": 4,
        "aii": 4,
    }
    assert tp.check_b12(tp.canonical_incumbent(8000), tp.PROBLEMS["i.S1C1"], 8000)["ok"]


def test_open_budget_refuses_an_undersized_positive_baseline():
    # cp1 #12: the registered formula is kept; an opening below the teach-code floor is refused
    c1d = tp.PROBLEMS["c1d"]
    with pytest.raises(ValueError):
        tp.open_budget(c1d, [0, 0, 12], beta=2)  # 24 < 32
    assert tp.open_budget(c1d, [0, 0, 16], beta=2) == 32
    assert tp.open_budget(c1d, []) == 52


def test_budget_rule():
    p = tp.PROBLEMS["c1d"]
    assert tp.budget(p, []) == tp.FIRST_BUDGET == 52
    assert tp.budget(p, [0, 0]) == 52  # zero-spend solves never enter the baseline
    assert tp.budget(p, [10, 30, 20], beta=2) == 40
    with pytest.raises(ValueError):
        tp.budget(p, [10])  # BETA unset until the pre-data amendment


def test_canonical_incumbent_carries_the_brain_stack():
    unit = tp.canonical_incumbent(8000)
    assert (
        type(unit.stack).__name__ == "PlasticityRuleStack"
        and unit.n_in == 52
        and unit.t_post == 6.0
    )


def test_every_p2_red_fires():
    reds = tp.construction_reds(tp.canonical_incumbent)
    assert reds["all_fired"], reds["fired"]
    assert len(reds["fired"]) == 5
