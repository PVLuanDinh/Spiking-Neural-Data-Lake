"""Tests for the held-out relational-analogy benchmark and its live cortex solver.

The benchmark's job is to be HONEST, so these tests police honesty rather than a headline score:
the pass rule must require structural evidence, the negative controls must genuinely fail, and the
solver must answer relational analogies without hijacking the queries other experts already own.
Deliberately NOT asserted: that the three known-limit items keep failing -- pinning failures in
place would make a real improvement to the mapper look like a regression.
"""

import pytest

from snn_data_lake.reasoning import analogy_bench as ab
from snn_data_lake.reasoning.reasoning_cortex import build_default_cortex

SOLAR_TO_ATOM = (
    "map attracts(sun,planet) revolves_around(planet,sun) more_massive(sun,planet) "
    ":: attracts(nucleus,electron) revolves_around(electron,nucleus) "
    "more_massive(nucleus,electron)"
)


@pytest.fixture(scope="module")
def rows():
    return ab.evaluate()


@pytest.fixture(scope="module")
def ctl():
    return ab.controls()


@pytest.fixture(scope="module")
def cortex():
    return build_default_cortex(seed=0)


# ---------------------------------------------------------------- the benchmark itself ----------


def test_bench_has_items_of_every_kind(rows):
    kinds = {r["kind"] for r in rows}
    assert kinds == {"classic", "hard", "limit"}
    assert len(rows) >= 10


def test_pass_rule_requires_structural_evidence(rows):
    # a recovered-but-unsupported mapping (systematicity 0) must NOT count as a hit
    for r in rows:
        assert r["passed"] == (r["exact"] and r["sys"] > 0.0)
        if r["exact"] and r["sys"] == 0.0:
            assert not r["passed"]


def test_honest_accuracy_clears_the_floor(rows):
    acc = sum(r["passed"] for r in rows) / len(rows)
    assert 0.5 <= acc <= 1.0


def test_classic_analogies_are_recovered(rows):
    classic = [r for r in rows if r["kind"] == "classic"]
    assert classic
    assert all(r["passed"] for r in classic)


def test_recovered_mappings_are_one_to_one(rows):
    for r in rows:
        assert len(set(r["got"].values())) == len(r["got"]), r["name"]


def test_benchmark_is_deterministic(rows):
    assert [r["got"] for r in ab.evaluate()] == [r["got"] for r in rows]


def test_gold_is_at_least_as_systematic_as_what_was_recovered(rows):
    # if the mapper beat the hand-written gold, the gold was mis-specified, not the solver clever
    for r in rows:
        assert r["sys"] <= r["sys_gold"] + 1e-9, r["name"]


# ---------------------------------------------------------------- the negative controls ---------


def test_vocab_disjoint_control_scores_near_zero(ctl):
    assert ctl["vocab_disjoint"]["sys"] <= 0.01


def test_structure_disjoint_control_stays_low(ctl):
    assert ctl["structure_disjoint"]["sys"] < 0.5


def test_shuffled_correspondence_is_not_recovered(ctl):
    sh = ctl["shuffled"]
    assert not sh["recovered_is_shuffled"]
    assert sh["sys_shuffled"] < sh["sys_gold"]


def test_controls_are_separated_from_real_analogies(rows, ctl):
    best_ctl = max(
        ctl["vocab_disjoint"]["sys"],
        ctl["structure_disjoint"]["sys"],
        ctl["shuffled"]["sys_shuffled"],
    )
    worst_real = min(r["sys"] for r in rows if r["kind"] == "classic")
    assert worst_real > best_ctl


def test_bench_main_self_check_passes():
    assert ab.main() == 0


# ---------------------------------------------------------------- the live cortex solver --------


def test_solver_is_registered(cortex):
    assert "structure-map" in [name for name, _m, _s, _u in cortex.roster()]


def test_solver_answers_a_relational_analogy(cortex):
    out = cortex.route(SOLAR_TO_ATOM)
    assert out["expert"] == "structure-map"
    assert out["known"] is True
    assert out["source"] == "gf"
    assert "sun->nucleus" in out["reply"]
    assert "planet->electron" in out["reply"]
    assert "systematicity" in out["reply"]


def test_solver_reports_its_systematicity(cortex):
    out = cortex.route(
        "pumps(heart,blood) flows_through(blood,artery) attached_to(artery,heart) "
        ":: pumps(pump,water) flows_through(water,pipe) attached_to(pipe,pump)"
    )
    assert out["known"] is True
    assert "heart->pump" in out["reply"]
    assert "systematicity 1.00" in out["reply"]


def test_solver_gaps_when_there_is_no_shared_structure(cortex):
    out = cortex.route(
        "attracts(sun,planet) revolves_around(planet,sun) :: bakes(chef,bread) owns(chef,oven)"
    )
    assert out["expert"] == "structure-map"
    assert out["known"] is False
    assert out["source"] is None
    assert "guess" in out["reply"]


def test_solver_gaps_past_its_measured_search_budget(cortex):
    wide = " ".join(f"cause(a{i},a{i + 1})" for i in range(1, 9))
    wide += " :: " + " ".join(f"cause(b{i},b{i + 1})" for i in range(1, 9))
    out = cortex.route(wide)
    assert out["expert"] == "structure-map"
    assert out["known"] is False
    assert "search budget" in out["reply"]


def test_solver_is_deterministic(cortex):
    assert cortex.route(SOLAR_TO_ATOM) == cortex.route(SOLAR_TO_ATOM)


# ---------------------------------------------------------------- no hijacking ------------------


@pytest.mark.parametrize(
    ("query", "expected_expert"),
    [
        ("abc:abd :: ijk:?", "gf-letter"),
        ("what comes next in 2 4 6 8", "gf-number-series"),
        ("a b c; b c a; c a ?", "gf-raven"),
        ("bind color=red shape=circle size=big ; what is color", "hdc-binding"),
        ("store apple->fruit banana->fruit carrot->veg ; lookup apple", "sdm-recall"),
        ("nearest to c?t among cat car dog bird", "dense-assoc"),
        ("cluster these numbers 1 2 3 20 21 22 40 41 42", "som-cluster"),
        ("triplet stdp weight change at 40 hz", "triplet-stdp"),
        ("arc: 12|30 -> 03|21 ; 56|70 -> 07|65 ; test 89|10", "arc-grid"),
    ],
)
def test_existing_query_types_are_untouched(cortex, query, expected_expert):
    out = cortex.route(query)
    assert out["expert"] == expected_expert
    assert out["known"] is True


def test_letter_analogy_scores_zero_on_the_structure_mapper(cortex):
    from snn_data_lake.reasoning.reasoning_cortex import Query, StructureMappingSolver

    solver = StructureMappingSolver(seed=0)
    assert solver.score(Query("abc:abd :: ijk:?")) == 0.0


def test_bare_is_to_as_phrasing_is_plausible_but_never_fires(cortex):
    from snn_data_lake.reasoning.reasoning_cortex import Query, StructureMappingSolver

    solver = StructureMappingSolver(seed=0)
    score = solver.score(Query("sun is to planet as nucleus is to what"))
    assert 0.0 < score < cortex.threshold  # nonzero, but below the routing threshold
    out = cortex.route("sun is to planet as nucleus is to what")
    assert out["expert"] is None
    assert out["known"] is False


def test_out_of_domain_query_still_gaps_with_no_expert(cortex):
    out = cortex.route("tell me about your weekend plans")
    assert out["expert"] is None
    assert out["known"] is False
    assert out["source"] is None
