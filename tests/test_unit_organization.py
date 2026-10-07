"""G7 / T115 P4 -- the T1 / T4 instrument (instruments/unit_organization.py).

Pre-registration: specs/022-unified-brain-arc/evidence/g7-unit-label-gate.md, 2.6-2.7 and 2.6a-2.6b
(frozen 854de0e7, amended da4bcfb7). Pure: no Brain, no lake, no confirmatory seed -- constructed
vectors and the
instrument-red seeds of the dev block (13130-13137, 13230-13237).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from snn_data_lake.instruments import unit_organization as U
from snn_data_lake.instruments.paired_contrast import permutation_fpr

# ---- verdict rules (rulings 9 / 9a) and the exact sign-flip rate (Codex b3 #21) ------------------


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


# ---- T1: margin, guard, contrasts, decision tables -----------------------------------------------


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


@pytest.mark.parametrize(
    "n_col, n_ref, refused", [(5, 0, True), (8, 8, True), (0, 5, True), (4, 4, False)]
)
def test_cp3_8_more_than_half_censored_in_either_arm_refuses_before_scoring(n_col, n_ref, refused):
    # Codex cp3 #8: every arm censored at budget 100 gives COL = 109, SINGLE = 101 -- constant
    # positive size differences that would PASS; the censoring refusal must come first
    flags = ([True] * n_col + [False] * (8 - n_col), [True] * n_ref + [False] * (8 - n_ref))
    size = U.t1_size_decision(_v(True), _v(True), censored=flags)
    struct = U.t1_struct_decision(_v(True), _v(True), owner_o9=True, censored=flags)
    assert (size["verdict"] == "REFUSED (censoring)") is refused
    assert (struct["verdict"] == "REFUSED (censoring)") is refused
    if refused:
        assert size["reverts_to_single"] is False and struct["deletes_structure"] is False


def _v(met):
    return U.ContrastVerdict("x", "noninferiority", (), 0.0, met, False, 0.0, 0.0)


@pytest.mark.parametrize(
    "u, e, verdict, deletes",
    [
        (None, True, "PASS (retention only; U untested)", False),
        (None, False, "RED", True),
        (True, True, "PASS", False),
        (True, False, "RED", True),
        (False, True, "RED", True),
        (False, False, "RED", True),
    ],
)
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


# ---- T4: the layer term, the exact-M sampler, modularity, the instrument reds --------------------


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
        U.sample_exact_m(
            9, 27, xy=_flat_units(9, 13232), kernel=lambda d: 1.0, seed=1, columns=cols
        )


def test_cp3_9_imposed_refuses_a_deficient_arrival_instead_of_repairing_it():
    # Codex cp3 #9: N = 16, M = 16, two columns of 8; an order whose second arrival is in the OTHER
    # column gives it one required edge and no earlier column mate -> REFUSED, never redistributed
    cols = [0] * 8 + [1] * 8
    seed = next(
        s
        for s in range(13230, 13300)
        if len({cols[i] for i in np.random.default_rng(s).permutation(16)[:2]}) == 2
    )
    with pytest.raises(U.Infeasible, match="fewer positive-weight candidates"):
        U.sample_exact_m(
            16, 16, xy=np.zeros((16, 2)), kernel=lambda d: 1.0, seed=seed, columns=cols
        )


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
    t0 = sum(
        _triangles(U.sample_exact_m(300, 1200, xy=xy, kernel=flat, seed=s), 300)
        for s in range(13233, 13236)
    )
    t5 = sum(
        _triangles(U.sample_exact_m(300, 1200, xy=xy, kernel=flat, seed=s, beta=5.0), 300)
        for s in range(13233, 13236)
    )
    assert t5 > 2 * t0, (t0, t5)


def test_random_seq_is_the_flat_sequential_sampler():
    xy = _flat_units(100, 13236)
    assert U.random_seq(100, 300, seed=13236) == U.sample_exact_m(
        100, 300, xy=xy, kernel=lambda d: 1.0, seed=13236
    )


@pytest.mark.parametrize(
    "gnm, seq, credited",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_rule_is_credited_only_if_it_beats_both_random_arms(gnm, seq, credited):
    v = lambda met: U.ContrastVerdict("x", "superiority", (), 0.0, met, False, 0.0, 0.0)  # noqa: E731
    assert U.t4_rule_credited(v(gnm), v(seq)) is credited


def test_bare_main_fires_its_instrument_reds(capsys):
    assert U.main() == 0
    assert "instrument reds fired: 6 of 6" in capsys.readouterr().out


def test_nmi_lives_beside_adjusted_rand_and_matches_a_hand_computed_value():
    from snn_data_lake.gates import growth_coverage_gate as gc

    # a = [0,0,1,1], b = [0,0,0,1]: H(a) = ln 2, H(b) = -(3/4 ln 3/4 + 1/4 ln 1/4),
    # I = 1/2 ln(4/3) + 1/4 ln(2/3) + 1/4 ln 2; NMI = I / ((H(a) + H(b)) / 2)  (arithmetic mean)
    ha = math.log(2)
    hb = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    mi = 0.5 * math.log(4 / 3) + 0.25 * math.log(2 / 3) + 0.25 * math.log(2)
    assert math.isclose(gc.nmi([0, 0, 1, 1], [0, 0, 0, 1]), mi / ((ha + hb) / 2), rel_tol=1e-12)
    assert gc.nmi([0, 0, 1, 1], [5, 5, 7, 7]) == 1.0  # a relabelling is perfect agreement
    assert gc.nmi([0, 0, 0, 0], [1, 1, 1, 1]) == 1.0  # both constant: sklearn's convention
    assert U.nmi is gc.nmi  # the instrument reads the ONE partition-agreement home (P2)


def test_nmi_matches_sklearn_on_the_ir_fixture_when_present():
    sk = pytest.importorskip("sklearn.metrics")
    pytest.importorskip("networkx")
    from snn_data_lake.gates import growth_coverage_gate as gc

    n, truth = 300, list(np.arange(300) * 10 // 300)
    planted = U.planted_blocks(n, 1500, blocks=10, between=0.5, seed=13230)
    _, labels = U.best_partition(planted, n, _FAST)
    ours, ref = gc.nmi(truth, labels), sk.normalized_mutual_info_score(truth, labels)
    assert math.isclose(ours, ref, rel_tol=1e-9)


# three Louvain restarts keep the test cheap; the instrument default is ten
_FAST = (13900, 13901, 13902)


def test_cp3_7_ir1_ir2_pass_on_a_detectable_partition_and_ir3_reports():
    pytest.importorskip("networkx")
    pytest.importorskip("sklearn")
    ir1 = U.ir1(300, 1500, blocks=10, between=0.2, louvain_seeds=_FAST)
    ir2 = U.ir2(300, 1500, louvain_seeds=_FAST)
    ir3 = U.ir3(300, 1500, blocks=10, louvain_seeds=_FAST)
    assert ir1["ok"] and ir1["verdict"].met and min(ir1["nmi"]) >= 0.9, ir1
    assert ir2["ok"] and not ir2["verdict"].met, ir2
    assert "verdict" in ir3 and len(ir3["verdict"].diffs) == 8
    assert U.t4_instruments_refuse(ir1, ir2) is False


def test_cp3_7_ir1_fails_when_the_plant_is_undetectable_and_t4_is_refused():
    pytest.importorskip("networkx")
    pytest.importorskip("sklearn")
    # ~random: no partition to find
    ir1 = U.ir1(300, 1500, blocks=10, between=0.9, louvain_seeds=_FAST)
    assert not ir1["ok"]
    assert U.t4_instruments_refuse(ir1, {"ok": True}) is True
    assert U.t4_instruments_refuse({"ok": True}, {"ok": False}) is True


def test_modularity_separates_a_planted_partition_from_random():
    pytest.importorskip("networkx")
    n, m = 600, 3000
    planted = U.planted_blocks(n, m, blocks=20, between=0.2, seed=13234)
    rand = U.random_graph(n, m, seed=13234)
    assert len(planted) == m and len(rand) == m
    q_p, q_r = U.modularity(planted, n), U.modularity(rand, n)
    assert q_p > 0.6 and q_r < 0.35, (q_p, q_r)


# ---- T4 wiring half on caller-supplied substrate (synthetic: no G4 data) -----------------------


def _blobs(n, seed):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0]])
    return centres[np.arange(n) % 4] + rng.normal(0.0, 40.0, (n, 2))


def test_t4_half_credits_a_planted_distance_rule_and_never_decides_on_imposed():
    # known positive: a local kernel over four blobs must beat both random arms, or it is dead
    out = U.t4_wiring_half(
        80,
        200,
        xy=_blobs(80, 13250),
        pops=None,
        kernel=lambda d: np.exp(-d / 60.0),
        beta=0.0,
        columns=None,
        seeds=tuple(range(13250, 13258)),
        louvain_seeds=(0, 1),
    )
    assert out["refused"] == [] and out["rule_credited"] is True
    assert out["verdicts"]["RULE>RANDOM"].met and out["verdicts"]["RULE>RANDOM-SEQ"].met
    assert not any("IMPOSED" in k for k in out["verdicts"])


def test_t4_half_is_refused_whole_when_the_density_is_infeasible():
    out = U.t4_wiring_half(
        10,
        46,
        xy=_flat_units(10, 13251),
        pops=None,
        kernel=lambda d: 1.0,
        beta=0.0,
        columns=None,
        seeds=(13251, 13252),
        louvain_seeds=(0,),
    )
    assert {"RULE", "RANDOM-SEQ"} <= set(out["refused"])
    assert out["rule_credited"] is None and out["verdicts"] == {}


def test_t4_half_a_refused_imposed_arm_is_reported_not_decisive():
    cols = [0] * 8 + [1] * 8  # cp3 #9: some seeds hand an arrival no earlier column mate
    seed = next(
        s
        for s in range(13250, 13300)
        if len({cols[i] for i in np.random.default_rng(s).permutation(16)[:2]}) == 2
    )
    out = U.t4_wiring_half(
        16,
        16,
        xy=_flat_units(16, 13253),
        pops=None,
        kernel=lambda d: 1.0,
        beta=0.0,
        columns=cols,
        seeds=(seed, *range(13270, 13277)),
        louvain_seeds=(0,),
    )
    assert out["refused"] == ["IMPOSED"] and out["rule_credited"] is not None


def test_random_graph_refuses_a_density_it_cannot_place_exactly():
    # networkx gnm silently returns the complete graph (fewer than M edges) when M exceeds the pairs
    with pytest.raises(U.Infeasible):
        U.random_graph(10, 46, seed=13254)
    assert len(U.random_graph(10, 45, seed=13254)) == 45


# ---- a pooled Part B record: the kernel is per pathway cell (G4 kernel_shape(record, cell)) ----


def _two_classes(n):
    return [i % 2 for i in range(n)]


def _cross_fraction(edges, classes):
    return sum(classes[i] != classes[j] for i, j in edges) / len(edges)


def test_pathway_with_one_kernel_in_every_cell_equals_the_single_kernel():
    n, cls = 40, _two_classes(40)
    k = lambda d: np.exp(-d / 300.0)  # noqa: E731
    xy = _flat_units(n, 13260)
    base = U.sample_exact_m(n, 80, xy=xy, kernel=k, seed=13260)
    same = U.sample_exact_m(
        n, 80, xy=xy, kernel=k, seed=13260, pathway=(cls, 2, {c: k for c in range(4)})
    )
    assert base == same


def test_pathway_kernel_of_a_cell_is_applied_to_that_cell_only():
    # known positive: nearly closed cross-class cells remove the cross edges a single kernel keeps
    n, cls = 40, _two_classes(40)
    k = lambda d: np.ones_like(d)  # noqa: E731
    closed = lambda d: np.full_like(d, 1e-9)  # noqa: E731
    xy = _flat_units(n, 13261)
    pw = (
        cls,
        2,
        {0 * 2 + 1: closed, 1 * 2 + 0: closed},
    )  # cells 0 and 3 absent = the global kernel
    single = U.sample_exact_m(n, 80, xy=xy, kernel=k, seed=13261)
    split = U.sample_exact_m(n, 80, xy=xy, kernel=k, seed=13261, pathway=pw)
    assert _cross_fraction(single, cls) > 0.3
    assert _cross_fraction(split, cls) < 0.05 and len(split) == 80


@pytest.mark.parametrize(
    "bad",
    [
        ([0, 1, 0], 2, {}),  # classes shorter than n
        ([0, 1] * 20, 1, {}),  # a class outside n_class
        ([0, 1] * 20, 2, [lambda d: d]),  # kernels is not a mapping
        ([0, 1] * 20, 2, {4: lambda d: d}),  # a cell outside n_class ** 2
    ],
)
def test_pathway_is_validated_not_repaired(bad):
    with pytest.raises(ValueError):
        U.sample_exact_m(
            40, 80, xy=_flat_units(40, 13262), kernel=lambda d: 1.0, seed=1, pathway=bad
        )


def test_t4_half_hands_the_pathway_to_every_kernel_arm_but_the_flat_one(monkeypatch):
    calls = []
    real = U.sample_exact_m
    user_kernel = lambda d: np.ones_like(d)  # noqa: E731

    def spy(n, m, **kw):
        calls.append((kw["kernel"] is user_kernel, kw.get("pathway") is not None))
        return real(n, m, **kw)

    monkeypatch.setattr(U, "sample_exact_m", spy)
    pw = (_two_classes(30), 2, {c: user_kernel for c in range(4)})
    U.t4_wiring_half(
        30,
        60,
        xy=_flat_units(30, 13263),
        pops=None,
        kernel=user_kernel,
        beta=0.5,
        columns=[i % 3 for i in range(30)],
        seeds=tuple(range(13263, 13271)),
        louvain_seeds=(0,),
        pathway=pw,
    )
    # RULE, DIST-ONLY, NO-LAYER, IMPOSED carry kernel and pathway; FLAT-KERNEL neither
    mine = [c for c in calls if c[0]]
    assert len(mine) >= 24 and all(c == (True, True) for c in mine)  # RULE, DIST-ONLY, NO-LAYER x 8
    assert [c for c in calls if not c[0] and c[1]] == []
    assert len([c for c in calls if not c[0]]) >= 1  # the FLAT-KERNEL arm ran


# ---- Codex D+E lens 1 folds (correctness + P8) ----


@pytest.mark.parametrize("kind", ["superiority", "noninferiority"])
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_judge_refuses_a_nonfinite_difference(kind, bad):
    # Codex D+E correctness: judge("bad", [inf] * 8, "noninferiority").met was True
    with pytest.raises(ValueError):
        U.judge("bad", [bad] * 8, kind)
    with pytest.raises(ValueError):
        U.judge("bad", [0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, bad], kind)


@pytest.mark.parametrize("n_pairs", [0, 1, 7, 9, 16])
def test_judge_decides_on_exactly_the_registered_eight_pairs(n_pairs):
    # T_BAR = 1.895 is the 7-d.f. bar of S = 8; Codex D+E: judge("x", [1], "noninferiority") was MET
    for kind in ("superiority", "noninferiority"):
        with pytest.raises(ValueError):
            U.judge("x", [1.0 + i for i in range(n_pairs)], kind)


def test_ir2_refuses_unmatched_or_short_seed_lists():
    for seeds, twins in (
        ((), ()),
        (U.IR_SEEDS, U.IR_TWIN_SEEDS[:1]),
        (U.IR_SEEDS[:7], U.IR_TWIN_SEEDS),
    ):
        with pytest.raises(ValueError):
            U.ir2(30, 60, seeds=seeds, twin_seeds=twins, louvain_seeds=(0,))


def test_t4_half_reads_its_seed_iterables_once_not_per_arm():
    # Codex D+E: seeds=iter(...) gave RULE eight scores, every later arm none, rule_credited False
    xy = _blobs(40, 13280)
    kw = dict(pops=None, kernel=lambda d: np.exp(-d / 60.0), beta=0.0, columns=None)
    tup = U.t4_wiring_half(40, 100, xy=xy, seeds=U.IR_SEEDS, louvain_seeds=(0,), **kw)
    gen = U.t4_wiring_half(40, 100, xy=xy, seeds=iter(U.IR_SEEDS), louvain_seeds=iter((0,)), **kw)
    assert tup["q"] == gen["q"] and tup["rule_credited"] == gen["rule_credited"]
    assert all(len(v) == 8 for v in gen["q"].values())


def test_t4_half_refuses_an_empty_louvain_seed_list():
    with pytest.raises(ValueError):
        U.t4_wiring_half(
            10, 12, xy=_flat_units(10, 13281), pops=None, kernel=lambda d: 1.0, beta=0.0,
            columns=None, seeds=U.IR_SEEDS, louvain_seeds=(),
        )  # fmt: skip
