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


def test_the_layer_term_is_wiring_priors_and_this_module_keeps_no_copy():
    from snn_data_lake.growth import wiring_prior as wp

    assert not hasattr(U, "population") and not hasattr(U, "layer_term")  # P2: one survivor
    for a in range(8):
        for b in range(8):
            c = wp.layer_term_sym(a, b)
            assert c == wp.layer_term_sym(b, a) and 0.0 <= c <= 1.0


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
        columns=[0] * 30,
        seeds=tuple(range(13263, 13271)),
        louvain_seeds=(0,),
        pathway=pw,
    )
    # RULE, DIST-ONLY, NO-LAYER, IMPOSED carry kernel and pathway; FLAT-KERNEL neither
    # one column holds every unit, so IMPOSED never refuses (its refusal is pinned in
    # test_t4_half_a_refused_imposed_arm_is_reported_not_decisive); exact arm-by-seed counts
    assert [c for c in calls if c[0]] == [
        (True, True)
    ] * 32  # RULE, DIST-ONLY, NO-LAYER, IMPOSED x 8
    assert [c for c in calls if not c[0]] == [(False, False)] * 16  # FLAT-KERNEL and RANDOM-SEQ x 8


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


@pytest.mark.parametrize(
    "kw",
    [
        dict(kernel=lambda d: 1.0, beta=float("nan")),
        dict(kernel=lambda d: 1.0, beta=float("inf")),
        dict(kernel=lambda d: float("nan")),
        dict(kernel=lambda d: float("inf")),
        dict(kernel=lambda d: -1.0),
    ],
)
def test_sampler_refuses_nonfinite_or_negative_weights(kw):
    # Codex D+E: beta=NaN returned eight entries holding only seven distinct edges
    with pytest.raises(ValueError) as e:
        U.sample_exact_m(8, 8, xy=np.zeros((8, 2)), seed=13230, **kw)
    assert not isinstance(e.value, U.Infeasible)  # a malformed input, not a density refusal


def test_planted_blocks_refuses_an_allocation_that_exceeds_a_pair_class():
    # Codex D+E: (10, 30, 2 blocks, between .2) needs 24 within edges, 20 exist: it looped forever
    # (reproduced under `timeout 60`: rc=124 before the fix)
    with pytest.raises(U.Infeasible):
        U.planted_blocks(10, 30, blocks=2, between=0.2, seed=13230)
    with pytest.raises(U.Infeasible):  # the between class: 25 pairs, 30 wanted
        U.planted_blocks(10, 30, blocks=2, between=1.0, seed=13230)
    ok = U.planted_blocks(
        10, 20, blocks=2, between=0.2, seed=13230
    )  # 16 within (of 20) + 4 between
    assert len(ok) == 20 and len(set(ok)) == 20


def test_zero_edge_graph_is_an_instrument_refusal_not_a_zero_division():
    # Codex D+E: t4_wiring_half(10, 0, ...) raised ZeroDivisionError out of networkx's modularity
    with pytest.raises(U.Infeasible):
        U.modularity([], 10, (0,))
    out = U.t4_wiring_half(
        10,
        0,
        xy=_flat_units(10, 13282),
        pops=None,
        kernel=lambda d: 1.0,
        beta=0.0,
        columns=None,
        seeds=U.IR_SEEDS,
        louvain_seeds=(0,),
    )
    assert {"RULE", "RANDOM", "RANDOM-SEQ"} <= set(out["refused"])
    assert out["rule_credited"] is None and out["verdicts"] == {}


def test_t1_struct_red_names_the_deletion_scope_amendment_e_registers():
    # Codex D+E: the RED text claimed E/E wiring deleted; COL has none (amendment (e)): the shared
    # inhibitory unit with E->I and I->E wiring is what goes, the neuron count does not
    out = U.t1_struct_decision(_v(True), _v(False), owner_o9=True, censored=_NONE)
    text = out["disposition"]
    assert out["deletes_structure"] is True
    assert "E->I" in text and "I->E" in text and "E/E" not in text and "E->E" not in text
    assert "neuron count" in text


def test_ir1_refuses_on_the_nmi_conjunct_alone():
    # Codex D+E: the 0.9-between fixture also fails superiority, so dropping NMI left it green.
    # This plant keeps superiority (MET) and misses the bar (min NMI ~0.63): only NMI can refuse it.
    pytest.importorskip("networkx")
    pytest.importorskip("sklearn")
    ir1 = U.ir1(300, 1500, blocks=10, between=0.5, louvain_seeds=_FAST)
    assert ir1["verdict"].met and min(ir1["nmi"]) < U.NMI_BAR
    assert ir1["ok"] is False
    assert U.t4_instruments_refuse(ir1, {"ok": True}) is True


def test_ir2_fails_on_a_constructed_positive_contrast_and_t4_is_refused(monkeypatch):
    # Codex D+E: forcing ir2 to ok=True left both IR tests green. Here RANDOM scores above its twin
    # (a spread, so not degenerate): the control must read MET and ok False.
    q = {s: 0.50 + 0.01 * (i % 3) for i, s in enumerate(U.IR_SEEDS)}
    q.update({t: 0.30 + 0.01 * (i % 2) for i, t in enumerate(U.IR_TWIN_SEEDS)})
    monkeypatch.setattr(U, "random_graph", lambda n, m, *, seed: seed)
    monkeypatch.setattr(U, "modularity", lambda edges, n, seeds=None: q[edges])
    ir2 = U.ir2(30, 60, louvain_seeds=(0,))
    assert ir2["verdict"].met is True and ir2["ok"] is False
    assert U.t4_instruments_refuse({"ok": True}, ir2) is True


def test_layer_term_biases_the_sampler_toward_within_population_edges():
    # the RULE arm's layer term (wiring_prior.layer_term_sym) must reach the sampler: real pops, Layer 2
    # exc vs Layer 6 exc (c = 0.192 / 0.078 within, 0.023 across), flat kernel so the layer term is the
    # ONLY non-flat factor. NO-LAYER (pops=None) is the control: ~1/2 cross edges, RULE far fewer.
    from snn_data_lake.growth.wiring_prior import population

    n, m, half = 40, 80, 20
    pops = np.array([population("Layer 2", 1)] * half + [population("Layer 6", 1)] * half)
    kw = dict(xy=np.zeros((n, 2)), kernel=lambda d: 1.0)

    def cross(edges):
        return sum((a < half) != (b < half) for a, b in edges)

    rule = [U.sample_exact_m(n, m, seed=s, pops=pops, **kw) for s in range(3)]
    ctrl = [U.sample_exact_m(n, m, seed=s, pops=None, **kw) for s in range(3)]
    assert all(r != c for r, c in zip(rule, ctrl))  # a constant layer term draws the control's edges
    rule_frac = sum(map(cross, rule)) / (3 * m)
    ctrl_frac = sum(map(cross, ctrl)) / (3 * m)
    assert ctrl_frac > 0.35 > rule_frac, (rule_frac, ctrl_frac)  # seeded: 46/240 vs 120/240


def test_sampler_pair_class_shares_match_the_layer_term_sym_prediction():
    # review #266: the share test above cannot tell c from c**0.5, a one-sided term, an E/I swap or the one-way
    # layer_term. Here the expected share of every unordered population-pair class is COMPUTED from
    # layer_term_sym and compared per class. Pops: Layer 5 exc (4), Layer 5 inh (5), Layer 6 exc (6) - an
    # inhibitory pop, and pairs whose two directional layer_term values differ strongly (5->4 0.373 vs
    # 4->5 0.060). m = n - 1: every arrival t >= 1 takes exactly ONE partner, so with a flat kernel and no
    # columns P(partner pop b | arriving pop a, earlier composition k) = k_b*s(a,b) / sum_b' k_b'*s(a,b'),
    # and the earlier composition at a uniformly random arrival slot is hypergeometric - enumerated exactly.
    import itertools

    from snn_data_lake.growth.wiring_prior import layer_term_sym, population

    per = 20
    P = [population("Layer 5", 1), population("Layer 5", -1), population("Layer 6", 1)]
    assert P == [4, 5, 6]
    n = per * len(P)
    pops = np.repeat(P, per)
    classes = list(itertools.combinations_with_replacement(P, 2))
    expected = dict.fromkeys(classes, 0.0)
    for a in P:
        rest = {b: per - (a == b) for b in P}
        for t in range(1, n):  # arrival slot t: t earlier units drawn from the other n - 1
            for k0, k1 in itertools.product(range(rest[4] + 1), range(rest[5] + 1)):
                k = {4: k0, 5: k1, 6: t - k0 - k1}
                if not 0 <= k[6] <= rest[6]:
                    continue
                pk = math.prod(math.comb(rest[b], k[b]) for b in P) / math.comb(n - 1, t)
                w = {b: k[b] * layer_term_sym(a, b) for b in P}
                for b in P:
                    expected[tuple(sorted((a, b)))] += pk * w[b] / sum(w.values()) / len(P) / (n - 1)
    assert abs(sum(expected.values()) - 1.0) < 1e-9

    seeds = 40
    counts = dict.fromkeys(classes, 0)
    for s in range(seeds):
        for i, j in U.sample_exact_m(n, n - 1, seed=s, pops=pops, xy=np.zeros((n, 2)), kernel=lambda d: 1.0):
            counts[tuple(sorted((int(pops[i]), int(pops[j]))))] += 1
    N = seeds * (n - 1)
    z = {c: (counts[c] / N - p) / math.sqrt(p * (1 - p) / N) for c, p in expected.items()}
    # binomial z per class; real max |z| over 50 disjoint 40-seed windows 2.85, every reviewed mutant >= 6.5
    assert max(map(abs, z.values())) < 4.0, z


def test_t4_half_draws_rule_and_no_layer_from_the_same_seed_and_keeps_the_max_q_restart():
    # review #271: nothing pinned that RULE and NO-LAYER share the seed (RULE or NO-LAYER drawn at s + 1
    # survived). ONE population: the layer term is a constant, so RULE and NO-LAYER must draw the SAME
    # edges -> Q equal exactly on every seed and the layer verdict degenerate. Also pins Louvain: every
    # Q is the MAX over the restart seeds, checked against each restart run alone (one restart, or keeping
    # the first restart, reads lower on the seeds where restart 2 wins).
    from snn_data_lake.growth.wiring_prior import population

    n, m, seeds, lseeds = 60, 240, tuple(range(13260, 13268)), (13900, 13901)
    out = U.t4_wiring_half(
        n,
        m,
        xy=np.zeros((n, 2)),
        pops=np.array([population("Layer 5", 1)] * n),
        kernel=lambda d: 1.0,
        beta=0.0,
        columns=None,
        seeds=seeds,
        louvain_seeds=lseeds,
    )
    assert out["refused"] == []
    assert out["q"]["RULE"] == out["q"]["NO-LAYER"], (out["q"]["RULE"], out["q"]["NO-LAYER"])
    layer = out["verdicts"]["layer"]
    assert layer.degenerate and layer.met is False

    per_restart = [[U.best_partition(U.random_graph(n, m, seed=s), n, (ls,))[0] for ls in lseeds] for s in seeds]
    assert sum(r[0] < r[1] for r in per_restart) >= 2  # the instrument: restart 2 must win somewhere
    assert out["q"]["RANDOM"] == [max(r) for r in per_restart]


def test_t4_half_credits_the_layer_term_only_when_real_pops_reach_the_rule_arm():
    # review #264: every other t4 test passes pops=None, so RULE and NO-LAYER drew identical edges and the
    # "layer" verdict had no reachable red. Real pops (Layer 2 exc / Layer 6 exc halves), flat kernel at one
    # point so the layer term is the ONLY structure: RULE's within-population bias must raise Q over NO-LAYER
    # on EVERY seed. Mutant RULE pops=pops -> pops=None: all diffs 0.0, verdict degenerate, not met.
    from snn_data_lake.growth.wiring_prior import population

    n, m, half = 60, 240, 30
    pops = np.array([population("Layer 2", 1)] * half + [population("Layer 6", 1)] * half)
    out = U.t4_wiring_half(
        n,
        m,
        xy=np.zeros((n, 2)),
        pops=pops,
        kernel=lambda d: 1.0,
        beta=0.0,
        columns=None,
        seeds=tuple(range(13260, 13268)),
        louvain_seeds=(0, 1),
    )
    assert out["refused"] == []
    layer = out["verdicts"]["layer"]
    assert layer.met is True and not layer.degenerate
    diffs = [r - c for r, c in zip(out["q"]["RULE"], out["q"]["NO-LAYER"], strict=True)]
    assert min(diffs) > 0.02, diffs  # seeded: 0.036..0.074 per seed
