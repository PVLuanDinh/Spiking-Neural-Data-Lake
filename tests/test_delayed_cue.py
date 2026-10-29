"""G2 / T110's problem generators and construction checks (instruments/delayed_cue.py, P1): every
bucket re-derived with zlib.crc32 here (a second instrument beside the module's, which reads the
learner's featurizer), every check B1, B2, B2-A, B3, B10, B10-S, B13, B14 and every constructed
red, the witness units against inseparable_mapping's A5, D2's float separability LP, and the
partition rule. No learner is taught and no Brain is built; seeds are the dev block 8000-8003
and the exposed scratch seed 8900."""

import zlib

import pytest

from snn_data_lake.instruments import delayed_cue as dc
from snn_data_lake.instruments import inseparable_mapping as im

DEV = (8000, 8001, 8002, 8003)


def _crc(word):
    return zlib.crc32(word.encode("ascii")) % 52


@pytest.mark.parametrize(
    "table",
    [dc.BUCKETS_I, dc.BUCKETS_C1D, dc.BUCKETS_AC1D, dc.BUCKETS_AII],
    ids=["i", "c1d", "ac1d", "aii"],
)
def test_buckets_rederived_with_crc32(table):
    assert {w: _crc(w) for w in table} == table


def test_every_family_bucket_is_distinct_and_at_least_8():
    every = [
        b
        for t in (dc.BUCKETS_I, dc.BUCKETS_C1D, dc.BUCKETS_AC1D, dc.BUCKETS_AII, im.BUCKETS_52)
        for b in t.values()
    ]
    assert len(every) == len(set(every)) == 7 + 12 + 12 + 4 + 4
    assert min(every) >= dc.MIN_BUCKET


def test_b1_all_families_pass():
    result = dc.check_b1_all()
    assert result["ok"], {k: v["conditions"] for k, v in result["families"].items()}


def test_b2_structure_exact():
    result = dc.check_b2()
    assert result["ok"], result
    assert result["max_code_deviation"] <= dc.ADDITIVITY_TOL
    assert result["witness_reads"] == {"S0C0": 8, "S1C0_history": 8, "S0C1": 4, "S1C1": 4}


def test_b2_c1d_additive_on_every_combo():
    for family in ("c1d", "ac1d"):
        assert dc.check_b2_combo(family)["ok"]


def test_b2a_and_witnesses():
    assert dc.check_b2a()["right"] == 64
    witnesses = dc.check_witnesses()
    assert witnesses["ok"], witnesses
    assert witnesses["cases"]["c1d"]["of"] == 64


def test_ii_witness_units_agree_with_check_a5():
    units = dc.witness_units("ii")
    answers = {uv: dc.LABELS[dc.xor_bit(units, im.code(im.TEXTS[uv]))] for uv in im.UV}
    assert all(answers[uv] == im.label_xor(*uv) for uv in im.UV)
    assert im.check_a5()["right"] == 4
    assert (
        abs(dc.check_witnesses()["cases"]["ii"]["min_margin"] - im.check_a5()["min_margin"]) < 1e-12
    )


@pytest.mark.parametrize("uv", im.UV, ids=lambda uv: f"uv{uv[0]}{uv[1]}")
def test_ii_witness_units_agree_with_check_a5_on_every_input(uv):
    # check_a5 exposes only `right` and the minimum margin, so each input is fed alone: a texts map
    # whose four keys all hold this input's text reports that input's own margin as the minimum
    units = dc.witness_units("ii")
    phi = im.code(im.TEXTS[uv])
    alone = {key: im.TEXTS[uv] for key in im.UV}
    assert abs(min(u.margin(phi) for u in units) - im.check_a5(alone)["min_margin"]) < 1e-12
    # the same input placed at the one key labelled GOAL, the other three keys kept: check_a5 reads
    # 4 of 4 iff it answers GOAL for this input
    goal_key = (0, 1)
    assert im.label_xor(*goal_key) == im.GOAL
    at_goal_key = {**im.TEXTS, goal_key: im.TEXTS[uv]}
    a5_says_goal = im.check_a5(at_goal_key)["right"] == 4
    assert (dc.LABELS[dc.xor_bit(units, phi)] == im.GOAL) == a5_says_goal


def test_float_lp_separates_separable_not_xor():
    # D2: the float side is a measurement; separable labellings are feasible, the XOR unions
    # infeasible
    def pairs_i(cell, text):
        return [(text(it), dc.label_i(cell, it)) for it in dc.ITEMS_I]

    assert dc.float_separability_lp(pairs_i("S0C0", dc.probe_text)) == "feasible"
    assert dc.float_separability_lp(pairs_i("S1C0", dc.history_text)) == "feasible"
    assert dc.float_separability_lp(pairs_i("S0C1", dc.probe_text)) == "infeasible"
    assert dc.float_separability_lp(pairs_i("S1C1", dc.history_text)) == "infeasible"
    every = dc.combo_items(dc.COMBOS)
    c1d = [(dc.combo_text("c1d", it), dc.combo_label("c1d", it)) for it in every]
    ac1d = [(dc.combo_text("ac1d", it), dc.combo_label("ac1d", it)) for it in every]
    assert dc.float_separability_lp(c1d) == "infeasible"
    assert dc.float_separability_lp(ac1d) == "feasible"


def test_lp_status_distinguishes_unresolved(monkeypatch):
    # cp1 #20: only HiGHS status 2 reads infeasible; another non-zero status is unresolved
    import scipy.optimize

    class R:
        status = 1

    monkeypatch.setattr(scipy.optimize, "linprog", lambda **k: R())
    pairs = [(dc.probe_text(it), dc.label_i("S0C1", it)) for it in dc.ITEMS_I]
    assert dc.float_separability_lp(pairs) == "unresolved:1"


def test_lp_measurements_recorded():
    lp = dc.lp_measurements()
    assert lp == {
        "S0C0": "feasible",
        "S1C0_history": "feasible",
        "S0C1": "infeasible",
        "S1C1_history": "infeasible",
        "c1d_union": "infeasible",
        "ac1d_union": "feasible",
    }


def test_b3_pure_check():
    stateless = {it: dc.label_i("S0C0", it) for it in dc.ITEMS_I}
    assert dc.check_b3(stateless)["ok"]
    stateful = {it: dc.label_i("S1C0", it) for it in dc.ITEMS_I}
    assert not dc.check_b3(stateful)["ok"]


@pytest.mark.parametrize("seed", DEV)
def test_partitions_and_split_checks(seed):
    for family in dc.COMBO_FAMILIES:
        parts, draws = dc.combo_partition(family, seed)
        assert draws >= 1
        assert {k: len(v) for k, v in parts.items()} == dc.SPLIT_SIZES
        assert dc.check_b13(parts)["ok"]
        assert dc.check_b14(family, parts)["ok"]
        assert dc.combo_partition(family, seed) == (parts, draws)  # deterministic


def test_partition_streams_differ_by_family_and_seed():
    a = dc.combo_partition("c1d", 8000)[0]
    assert a != dc.combo_partition("ac1d", 8000)[0]
    assert a != dc.combo_partition("c1d", 8001)[0]


def test_partition_refuses_oversized_splits():
    with pytest.raises(ValueError):
        dc.combo_partition(
            "c1d", 8000, sizes={"teach": 10, "validation": 2, "future": 3, "future_d": 3}
        )


@pytest.mark.parametrize("seed", DEV)
def test_u_fixtures(seed):
    assert dc.check_b10(seed)["ok"]
    assert dc.check_b10s(seed)["ok"]
    # a stateless reader answers each item once per read: wrong on exactly half its passes
    for i in range(len(dc.ITEMS_I)):
        labels = [dc.u_label(seed, i, k) for k in range(dc.PASSES)]
        assert labels.count(dc.GREET) == labels.count(dc.GOAL) == 2


def test_explicit_history_adds_the_cue_only():
    unit = dc.incumbent(8900)
    history = dc.ExplicitHistory(unit)
    item = (1, 0, 1)
    assert history.text(dc.episode_i(item)) == dc.history_text(item) == "pen box oak owl"
    before = unit.weights()
    assert history.route(dc.episode_i(item)) == unit.route_text("pen box oak owl")
    assert unit.weights() == before  # reading teaches nothing


def test_decoy_units_deterministic_by_callers_seed():
    texts = dc.texts_combo("c1d")
    a = dc.decoy_units(texts, 3, 10418)
    b = dc.decoy_units(texts, 3, 10418)
    assert [u.w for u in a] == [u.w for u in b] and [u.theta for u in a] == [u.theta for u in b]
    assert [u.w for u in dc.decoy_units(texts, 3, 10419)] != [u.w for u in a]


def test_witness_refuses_unknown_problem():
    with pytest.raises(ValueError):
        dc.witness_units("S0C0")


def test_every_construction_red_fires():
    reds = dc.construction_reds()
    assert reds["all_fired"], [k for k, v in reds["fired"].items() if not v]
    assert len(reds["fired"]) == 18


def test_r_q_collision_breaks_additivity_and_r_p_cannot():
    # cp1 #14: the registered R/P collision red cannot fire on B2 (P sits at rank 3 whatever q
    # is); the R/Q one does
    assert dc.r_p_collision_keeps_additivity()
    assert dc.construction_reds()["fired"]["b2_c1d_r_q_collision"]
    words = dc.combo_words("c1d")
    assert not dc.check_b1(
        (*words[:4], words[0], *words[5:]), [dc.collided_text((0, 0, 0, 0), "P")]
    )["conditions"]["distinct_buckets"]  # B1's collision refusal still catches the R/P case


def test_construction_check_on_dev_seeds():
    result = dc.construction_check(DEV)
    assert result["verdict"], result["per_seed"]
