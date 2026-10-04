"""G7 / T115 P4 -- the T1 / T4 instrument (instruments/unit_organization.py).

Pre-registration: specs/022-unified-brain-arc/evidence/g7-unit-label-gate.md, 2.6-2.7 and 2.6a-2.6b (frozen
854de0e7, amended da4bcfb7). Pure: no Brain, no lake, no confirmatory seed -- constructed vectors and the
instrument-red seeds of the dev block (13130-13137, 13230-13237).
"""

from __future__ import annotations

import math

import numpy as np

import pytest

from snn_data_lake.instruments import unit_organization as U
from snn_data_lake.instruments.paired_contrast import permutation_fpr

# ---- verdict rules (rulings 9 / 9a) and the exact sign-flip rate (Codex b3 #21) -------------------------


def test_degenerate_superiority_is_not_met_and_degenerate_noninferiority_reads_its_value():
    sup = U.judge("x", [0.3] * 8, "superiority")
    ni = U.judge("x", [0.3] * 8, "noninferiority")
    assert sup.degenerate and not sup.met
    assert ni.degenerate and ni.met
    assert not U.judge("x", [0.0] * 8, "noninferiority").met
    assert not U.judge("x", [-0.1] * 8, "noninferiority").met


def test_exact_flip_rate_fixtures_constant_positive_and_constant_zero():
    # constant positive c: the all-positive flip is degenerate (MET only for NI); the eight 7+/1-
    # flips have t = 3.0 > 1.895 (MET); 6+/2- has t = 1.528 (not MET). Hand-computed.
    assert U.judge("x", [0.5] * 8, "superiority").exact_flip_rate == 8 / 256
    assert U.judge("x", [0.5] * 8, "noninferiority").exact_flip_rate == 9 / 256
    assert U.judge("x", [0.0] * 8, "superiority").exact_flip_rate == 0.0
    assert U.judge("x", [0.0] * 8, "noninferiority").exact_flip_rate == 0.0


def test_exact_flip_rate_matches_the_reference_when_no_flip_is_degenerate():
    diffs = [0.11, 0.23, -0.05, 0.31, 0.07, 0.19, 0.42, 0.02]  # distinct |d|: no flip is constant
    v = U.judge("x", diffs, "superiority")
    assert v.exact_flip_rate == permutation_fpr(diffs)
    assert v.reference_fpr == permutation_fpr(diffs)


# ---- T1: margin, guard, contrasts, decision tables ------------------------------------------------------


def test_delta_ni_is_strictly_positive_and_in_items():
    assert U.delta_ni(0) == 1
    assert U.delta_ni(2.5) == 2.5
    with pytest.raises(ValueError):
        U.delta_ni(-1)


def test_t1_diffs():
    d = U.size_diffs([9, 18], [9, 12])
    assert math.isclose(d[0], math.log(1.2)) and math.isclose(d[1], math.log(1.2) - math.log(1.5))
    e = U.retention_diffs([1, 0], [0, 0], delta=1)
    assert e == [0, 1]
    with pytest.raises(ValueError):
        U.retention_diffs([0], [0], delta=0)


def test_guard_is_per_problem_excludes_censoring_and_refuses_the_whole_contrast():
    ok = {"A": [90, 85, 100, None], "B": [80, 81, 120, 95]}
    passed, detail = U.granularity_guard(ok, k=8)
    assert passed and detail["A"]["median"] == 90 and detail["A"]["censored"] == 1
    one_small = dict(ok, C=[2, 3, 4, 2])
    assert not U.granularity_guard(one_small, k=8)[0]  # ONE judging problem fails -> whole refused
    mostly_censored = dict(ok, D=[200, None, None, None])
    assert not U.granularity_guard(mostly_censored, k=8)[0]
    assert U.granularity_guard({"A": [40, 41, 45]}, k=4)[0]  # U* = 40 for k = 4
    assert not U.granularity_guard({"A": [40, 41, 45]}, k=8)[0]


_NONE = ([False] * 8, [False] * 8)  # (COL arm, reference arm) censor flags per seed: none censored


@pytest.mark.parametrize("n_col, n_ref, refused", [(5, 0, True), (8, 8, True), (0, 5, True), (4, 4, False)])
def test_cp3_8_more_than_half_censored_in_either_arm_refuses_before_scoring(n_col, n_ref, refused):
    # Codex cp3 #8: every arm censored at budget 100 gives COL = 109, SINGLE = 101 -- constant positive
    # size differences that would PASS; the censoring refusal must come first
    flags = ([True] * n_col + [False] * (8 - n_col), [True] * n_ref + [False] * (8 - n_ref))
    size = U.t1_size_decision(_v(True), _v(True), censored=flags)
    struct = U.t1_struct_decision(_v(True), _v(True), owner_o9=True, censored=flags)
    assert (size["verdict"] == "REFUSED (censoring)") is refused
    assert (struct["verdict"] == "REFUSED (censoring)") is refused
    if refused:
        assert size["reverts_to_single"] is False and struct["deletes_structure"] is False


def _v(met):
    return U.ContrastVerdict("x", "noninferiority", (), 0.0, met, False, 0.0, 0.0)


@pytest.mark.parametrize("u, e, verdict, deletes", [
    (None, True, "PASS (retention only; U untested)", False),
    (None, False, "RED", True),
    (True, True, "PASS", False),
    (True, False, "RED", True),
    (False, True, "RED", True),
    (False, False, "RED", True),
])
def test_t1_struct_decision_table(u, e, verdict, deletes):
    out = U.t1_struct_decision(None if u is None else _v(u), _v(e), owner_o9=True, censored=_NONE)
    assert out["verdict"] == verdict and out["deletes_structure"] is deletes


def test_t1_struct_never_deletes_before_o9():
    out = U.t1_struct_decision(_v(True), _v(False), owner_o9=False, censored=_NONE)
    assert out["verdict"] == "RED" and out["deletes_structure"] is False
    assert "O9" in out["disposition"]


def test_t1_size_is_untested_when_its_u_half_is_refused():
    out = U.t1_size_decision(None, _v(True), censored=_NONE)
    assert out["verdict"] == "UNTESTED" and out["reverts_to_single"] is False
    assert U.t1_size_decision(_v(False), _v(True), censored=_NONE)["reverts_to_single"] is True


# ---- T4: the layer term, the exact-M sampler, modularity, the instrument reds ---------------------------


def test_population_index_follows_conn_prob_and_excludes_layer_1():
    from snn_data_lake.reservoir.cortical_microcircuit import POP_NAMES

    assert POP_NAMES[U.population("Layer 2", 1)] == "L2/3e"
    assert POP_NAMES[U.population("Layer 3", -1)] == "L2/3i"
    assert POP_NAMES[U.population("Layer 6", 1)] == "L6e"
    assert U.population("Layer 1", -1) is None


def test_layer_term_is_symmetric_and_a_probability():
    for a in range(8):
        for b in range(8):
            c = U.layer_term(a, b)
            assert c == U.layer_term(b, a) and 0.0 <= c <= 1.0


def _flat_units(n, seed):
    import random

    rng = random.Random(seed)
    return [(rng.random() * 1000.0, rng.random() * 1000.0) for _ in range(n)]


def test_sampler_places_exactly_m_simple_edges_deterministically():
    xy = _flat_units(200, 13230)
    a = U.sample_exact_m(200, 600, xy=xy, kernel=lambda d: np.exp(-d / 100.0), seed=13230)
    b = U.sample_exact_m(200, 600, xy=xy, kernel=lambda d: np.exp(-d / 100.0), seed=13230)
    assert a == b and len(a) == 600 and len(set(a)) == 600
    assert all(i < j for i, j in a)


def test_sampler_refuses_an_infeasible_density():
    xy = _flat_units(10, 13231)
    with pytest.raises(U.Infeasible):
        U.sample_exact_m(10, 46, xy=xy, kernel=lambda d: 1.0, seed=1)
    cols = [i // 3 for i in range(9)]  # columns of 3: at most 2 earlier mates
    with pytest.raises(U.Infeasible):
        U.sample_exact_m(9, 27, xy=_flat_units(9, 13232), kernel=lambda d: 1.0, seed=1, columns=cols)


def _triangles(edges, n):
    nb = [set() for _ in range(n)]
    for i, j in edges:
        nb[i].add(j)
        nb[j].add(i)
    return sum(len(nb[i] & nb[j]) for i, j in edges) // 3


def test_the_shared_partner_term_is_live():
    # the frozen one-shot draw would hold every shared-partner count at 0 (an arriving unit has no
    # neighbours yet): beta would be inert. Draws are sequential, so beta must raise closure.
    xy = _flat_units(300, 13233)
    flat = lambda d: 1.0  # noqa: E731
    t0 = sum(_triangles(U.sample_exact_m(300, 1200, xy=xy, kernel=flat, seed=s), 300)
             for s in range(13233, 13236))
    t5 = sum(_triangles(U.sample_exact_m(300, 1200, xy=xy, kernel=flat, seed=s, beta=5.0), 300)
             for s in range(13233, 13236))
    assert t5 > 2 * t0, (t0, t5)


def test_random_seq_is_the_flat_sequential_sampler():
    xy = _flat_units(100, 13236)
    assert U.random_seq(100, 300, seed=13236) == U.sample_exact_m(
        100, 300, xy=xy, kernel=lambda d: 1.0, seed=13236)


@pytest.mark.parametrize("gnm, seq, credited", [
    (True, True, True), (True, False, False), (False, True, False), (False, False, False)])
def test_rule_is_credited_only_if_it_beats_both_random_arms(gnm, seq, credited):
    v = lambda met: U.ContrastVerdict("x", "superiority", (), 0.0, met, False, 0.0, 0.0)  # noqa: E731
    assert U.t4_rule_credited(v(gnm), v(seq)) is credited


def test_bare_main_fires_its_instrument_reds(capsys):
    assert U.main() == 0
    assert "instrument reds fired: 6 of 6" in capsys.readouterr().out


def test_modularity_separates_a_planted_partition_from_random():
    pytest.importorskip("networkx")
    n, m = 600, 3000
    planted = U.planted_blocks(n, m, blocks=20, between=0.2, seed=13234)
    rand = U.random_graph(n, m, seed=13234)
    assert len(planted) == m and len(rand) == m
    q_p, q_r = U.modularity(planted, n), U.modularity(rand, n)
    assert q_p > 0.6 and q_r < 0.35, (q_p, q_r)
