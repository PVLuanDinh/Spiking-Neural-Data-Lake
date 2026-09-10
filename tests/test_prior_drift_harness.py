"""The three-arm drift harness: both SC-006 reds, the positive control, the refusals (T063).

specs/011 asked for a DriftRecord and a seeded / random / deliberately-wrong comparison and left the
task, the distance, the width and the null's design open; specs/022 SC-006 asks the seeded arm to
beat BOTH comparators. The pre-registration `specs/022-unified-brain-arc/evidence/
t063-prior-drift-harness.md` closes those, and this file is the reds and controls it names.

Committed RED, before `cortex/prior_drift_harness.py` exists.

AMENDMENT 2.12 (2026-09-09), committed RED a second time before the amended instrument exists: the
pilot's K rule selected NO K (the task saturated at 1.0000 for every wiring) and the reversed
laminar WRONG arm failed its validity precondition by 70x. The reds below are the three items --
(a) the fixed per-seed orthonormalised random projection to D with the LARGEST-D-under-0.9 guard,
(b) the BAND-PERMUTED-TARGETS wrong arm under the depth-reversing permutation, and (c) the settling
clause read as the absence of a monotone trend over at least three 90 s windows -- plus the campaign
guard that runs the campaign only when a D was selected AND the WRONG arm is valid.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_prior_drift_harness.py -q --no-cov
"""

import math

import pytest

np = pytest.importorskip("numpy")

from snn_data_lake.cortex import prior_drift_harness as H  # noqa: E402
from snn_data_lake.substrate import drift_monitor as DM  # noqa: E402


# ---- P2: the PSI formula lives ONCE ---------------------------------------------------------
def test_psi_is_module_level_and_the_monitor_calls_it():
    assert callable(DM.psi), "the PSI formula must be a module-level function, not a method body"
    ref = [0.25, 0.25, 0.25, 0.25]
    act = [0.4, 0.3, 0.2, 0.1]
    assert DM.psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert DM.psi(act, ref) > 0.0
    calls = []
    real = DM.psi

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    DM.psi = spy
    try:
        mon = DM.DriftMonitor([[i / 20.0] * 4 for i in range(20)], n_bins=4)
        for _i in range(20):
            mon.push([0.9] * 4)
        mon.psi()
    finally:
        DM.psi = real
    assert calls, "DriftMonitor.psi must call the module-level formula, not keep its own copy"


def test_the_harness_imports_the_same_psi():
    assert H.distance("band_pair_weight_share", [0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    a, b = [0.7, 0.3], [0.5, 0.5]
    assert H.distance("band_pair_weight_share", a, b) == pytest.approx(DM.psi(a, b))


# ---- the DriftRecord (specs/011 data-model 4) ------------------------------------------------
def test_drift_record_fields_are_verbatim():
    assert H.DriftRecord._fields == ("statistic", "seeded", "current", "distance", "direction")


def test_zero_distance_is_a_failure_of_the_seeding_approach_not_stability():
    rec = H.DriftRecord("band_pair_weight_share", (0.5, 0.5), (0.5, 0.5), 0.0, 0)
    assert H.is_seeding_failure([rec]) is True
    moved = H.DriftRecord("band_pair_weight_share", (0.5, 0.5), (0.6, 0.4), 0.02, 1)
    assert H.is_seeding_failure([moved]) is False


# ---- RED 1: never moves off the seed ---------------------------------------------------------
def test_red1_names_the_pin_the_harness_applied():
    frozen = H.ArmResult("frozen", True, 0.2, 40, 900, 0, 0.0, _zero_recs())
    fired, msg = H.red1(frozen)
    assert fired is True
    assert "seeding approach failed" in msg and "PINNED (lr = 0 by construction)" in msg
    assert "executed 40" in msg and "attempted 900" in msg and "effective 0" in msg


def test_red1_refuses_to_call_a_learning_arms_zero_pinning():
    """Codex defect 4: a zero can come from cancellation, quantization or absent pairs. Only the
    VERIFIED intervention is called pinning."""
    cancelled = H.ArmResult("seeded", False, 0.2, 40, 900, 0, 0.0, _zero_recs())
    fired, msg = H.red1(cancelled)
    assert fired is False
    assert "PINNED (lr = 0 by construction)" not in msg
    assert "cancelled or below quantization" in msg
    no_pairs = H.ArmResult("seeded", False, 0.2, 40, 0, 0, 0.0, _zero_recs())
    fired, msg = H.red1(no_pairs)
    assert fired is False and "no eligible spike pairs" in msg
    assert "PINNED (lr = 0 by construction)" not in msg
    #: and a zero read from an EMPTY record set is not zero drift either -- there is nothing to
    #: read it from, so red 1 must not fire on it
    assert H.red1(H.ArmResult("frozen", True, 0.2, 40, 900, 0, 0.0, ()))[0] is False


def test_red1_control_a_learning_arm_reads_effective_and_drift_above_zero():
    recs = (
        H.DriftRecord("band_pair_weight_share", (1.0,), (0.9,), 0.01, 1),
        H.DriftRecord("band_fanin_e_fraction", (1.0,), (0.9,), 0.1, 1),
    )
    seeded = H.ArmResult("seeded", False, 0.7, 40, 900, 133, 0.4, recs)
    assert H.drift_control_holds(seeded) is True
    assert H.drift_control_holds(H.ArmResult("seeded", False, 0.7, 40, 900, 0, 0.0, recs)) is False


# ---- the statistic over 8 seeds --------------------------------------------------------------
def test_paired_t_and_the_permutation_fpr():
    diffs = [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.10]
    t = H.paired_t(diffs)
    assert t > H.T_BAR
    fpr = H.permutation_fpr(diffs)
    assert 0.0 <= fpr <= 1.0
    assert H.paired_t([0.0] * 8) == 0.0


def test_ties_count_as_non_positive():
    c = H.contrast("tie", [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
    assert c.t == pytest.approx(H.paired_t([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]))


def test_zero_variance_differences_are_met_only_when_positive():
    pos = H.contrast("degenerate-positive", [0.05] * 8)
    neg = H.contrast("degenerate-zero", [0.0] * 8)
    assert pos.degenerate and pos.met is True
    assert neg.degenerate and neg.met is False


# ---- RED 2 and the SC-006 conjuncts ----------------------------------------------------------
def _acc(seeded, random_, wrong, planted, frozen):
    out = {}
    for s in range(8):
        out[s] = {
            "seeded": H.ArmResult("seeded", False, seeded[s], 40, 900, 100, 0.4, _recs()),
            "random": H.ArmResult("random", False, random_[s], 40, 900, 100, 0.4, _recs()),
            "wrong": H.ArmResult("wrong", False, wrong[s], 40, 900, 100, 0.4, _recs()),
            "planted": H.ArmResult("planted", False, planted[s], 40, 900, 100, 0.4, _recs()),
            "frozen": H.ArmResult("frozen", True, frozen[s], 40, 900, 0, 0.0, _zero_recs()),
        }
    return out


def _recs():
    return (
        H.DriftRecord("band_pair_weight_share", (1.0,), (0.9,), 0.02, 1),
        H.DriftRecord("band_fanin_e_fraction", (1.0,), (0.9,), 0.05, 1),
    )


def _zero_recs():
    return (
        H.DriftRecord("band_pair_weight_share", (1.0,), (1.0,), 0.0, 0),
        H.DriftRecord("band_fanin_e_fraction", (1.0,), (1.0,), 0.0, 0),
    )


def test_all_five_conjuncts_met():
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.30, 0.32],
            [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92],
            [0.20] * 8,
        ),
        wrong_valid=True,
    )
    assert [v["conjuncts"][k]["state"] for k in "abcde"] == ["MET"] * 5
    assert v["sc006"] == "MET"
    text = H.format_verdict(v)
    for k in "abcde":
        assert f"({k})" in text
    assert "per-seed" in text.lower()


def test_a_blind_harness_reads_no_verdict_at_all():
    """(c) NOT MET -> the PLANTED control did not beat RANDOM: nothing else is read."""
    v = H.verdict(
        _acc(
            [0.70] * 8,
            [0.50] * 8,
            [0.30] * 8,
            [0.50] * 8,
            [0.20] * 8,
        ),
        wrong_valid=True,
    )
    assert v["conjuncts"]["c"]["state"] == "NOT MET"
    assert v["sc006"] == "NO VERDICT"
    for k in "abde":
        assert v["conjuncts"][k]["state"] == "NOT READ"


def test_red2_does_not_fire_on_a_failure_to_establish_superiority():
    """Codex defect 1: WRONG must UNDERPERFORM, not merely fail to beat, RANDOM."""
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.50, 0.53, 0.47, 0.52, 0.48, 0.54, 0.49, 0.53],
            [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92],
            [0.20] * 8,
        ),
        wrong_valid=True,
    )
    assert v["conjuncts"]["d"]["state"] == "NOT MET"
    assert "underperformance NOT ESTABLISHED" in H.format_verdict(v)
    assert v["sc006"] == "NOT MET"


def test_seeded_vs_wrong_is_a_direct_contrast():
    """Codex defect 2: differing significance against RANDOM is not a SEEDED-vs-WRONG result. The
    counterexample from that read, verbatim in its numbers."""
    v = H.verdict(
        _acc(
            [0.54, 0.55] * 4,
            [0.50] * 8,
            [0.35, 0.75] * 4,
            [0.90] * 8,
            [0.20] * 8,
        ),
        wrong_valid=True,
    )
    assert v["conjuncts"]["e"]["state"] == "NOT MET", "WRONG's mean exceeds SEEDED's here"
    assert v["sc006"] != "MET"


def test_an_invalid_wrong_arm_makes_red2_not_read():
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.30, 0.32],
            [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92],
            [0.20] * 8,
        ),
        wrong_valid=False,
    )
    assert v["conjuncts"]["d"]["state"] == "NOT READ"
    assert v["sc006"] != "MET"


def test_the_learners_own_criterion_is_separate_from_the_priors():
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.30, 0.32],
            [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92],
            [0.20] * 8,
        ),
        wrong_valid=True,
    )
    assert v["learner"]["disposition"] == "KEPT"
    assert v["learner"]["t"] > H.T_BAR
    v2 = H.verdict(
        _acc(
            [0.70] * 8,
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.30] * 8,
            [0.90] * 8,
            [0.70] * 8,
        ),
        wrong_valid=True,
    )
    assert v2["learner"]["disposition"] == "DELETED"


# ---- the arms, and the nuisance properties that make attribution possible ---------------------
def _art():
    return H.synthetic_artifact(n_units=64, seed=0)


def test_the_synthetic_artifact_is_the_real_format():
    art = _art()
    for k in ("agglo_id", "sign", "layer", "pre_idx", "post_idx", "n_syn"):
        assert k in art
    from snn_data_lake.cortex import h01_ingest as I

    w = I.fan_in_shares(art)
    assert w.shape == art["pre_idx"].shape
    state, syn, kt = I.population(art)
    assert state.n_units == art["agglo_id"].size


def test_random_preserves_degrees_within_sign_class():
    base = _art()
    rnd = H.arm_random(base, seed=1)
    H.check_nuisance(rnd, base, "random")
    for key in ("pre_idx", "post_idx"):
        a = np.bincount(base[key], minlength=base["agglo_id"].size)
        b = np.bincount(rnd[key], minlength=base["agglo_id"].size)
        assert np.array_equal(a, b), f"{key} degree sequence not preserved"
    sgn = base["sign"]
    assert np.array_equal(sgn[base["pre_idx"]].sum(), sgn[rnd["pre_idx"]].sum())
    assert not np.array_equal(
        np.sort(base["pre_idx"] * 10_000 + base["post_idx"]),
        np.sort(rnd["pre_idx"] * 10_000 + rnd["post_idx"]),
    ), "the rewire changed nothing"


def test_random_has_no_self_or_multi_edges():
    rnd = H.arm_random(_art(), seed=2)
    assert not np.any(rnd["pre_idx"] == rnd["post_idx"])
    pairs = rnd["pre_idx"].astype(np.int64) * 1_000_000 + rnd["post_idx"]
    assert np.unique(pairs).size == pairs.size


# ---- 2.12(b): WRONG is the BAND-PERMUTED-TARGETS table ---------------------------------------
# The direction-reversed laminar table is RETIRED as an ARM (its pilot readings stay in the pin):
# the pilot measured PSI(reversed, seeded) = 0.032389 against PSI(random, seeded) = 2.281271, so it
# failed 2.2's validity precondition on this connectome by 70x. What replaces it corrupts the
# DIAGONAL -- the 0.7265 of the band-pair mass a transpose cannot touch.
def test_wrong_is_the_band_permuted_target_table_under_the_depth_reversing_permutation():
    assert H.BAND_PERMUTATION == (5, 4, 3, 2, 1, 0, 6), (
        "pi must reverse cortical depth -- L1<->L6, L2<->L5, L3<->L4, White matter fixed"
    )
    base = _art()
    wrong = H.arm_wrong(base)
    H.check_nuisance(wrong, base, "wrong")
    assert np.array_equal(wrong["pre_idx"], base["pre_idx"]), "every edge KEEPS its pre"
    assert np.array_equal(wrong["n_syn"], base["n_syn"]), "every edge KEEPS its synapse count"
    lay = base["layer"].astype(np.int64)
    want = np.asarray([H.BAND_PERMUTATION[int(b)] for b in lay[base["post_idx"]]], dtype=np.int64)
    got = lay[wrong["post_idx"]]
    assert (got == want).mean() > 0.95, "the post must land in band pi(band(post))"
    assert not np.any(wrong["pre_idx"] == wrong["post_idx"]), "no self-edge may be created"


def test_wrong_keeps_the_within_band_degree_rank_of_the_target():
    """ "the same within-band degree rank": the rank ORDER of the targets inside a band must be
    carried into the mirrored band, so the permutation moves laminar IDENTITY and nothing else."""
    base = H.synthetic_artifact(n_units=140, seed=0)
    wrong = H.arm_wrong(base)
    lay = base["layer"].astype(np.int64)
    rank = H.within_band_degree_rank(base)
    old_r = rank[base["post_idx"].astype(np.int64)]
    new_r = rank[wrong["post_idx"].astype(np.int64)]
    band_ok = lay[wrong["post_idx"]] == np.asarray(
        [H.BAND_PERMUTATION[int(b)] for b in lay[base["post_idx"]]], dtype=np.int64
    )
    #: monotone in rank: a higher-ranked source target maps to a >= ranked mirrored target
    order = np.argsort(old_r[band_ok], kind="stable")
    seq = new_r[band_ok][order]
    assert np.all(np.diff(seq) >= 0), "the within-band rank order was not carried across"


def test_the_band_mass_report_names_what_the_permutation_moved():
    base = _art()
    rep = H.band_mass_report(base, H.arm_wrong(base))
    for key in (
        "diagonal_mass",
        "off_diagonal_mass",
        "inter_band_edge_fraction",
        "asymmetry",
        "off_band_fallback_edges",
        "psi_wrong_vs_seeded",
    ):
        assert key in rep, f"the band-mass report must name {key}"
    assert 0.0 <= rep["diagonal_mass"] <= 1.0
    assert rep["diagonal_mass"] + rep["off_diagonal_mass"] == pytest.approx(1.0, abs=1e-9)


def test_wrong_validity_precondition_is_measured_not_assumed():
    base = _art()
    ok, d_wrong, d_random = H.wrong_validity(H.arm_wrong(base), H.arm_random(base, 3), base)
    assert isinstance(ok, bool)
    assert d_wrong >= 0.0 and d_random >= 0.0
    assert ok == (d_wrong > d_random)


def test_planted_is_a_positive_control_under_the_same_nuisance_constraints():
    base = _art()
    classes = H.drive_classes(base, k=4, seed=0)
    planted = H.arm_planted(base, classes, seed=0)
    H.check_nuisance(planted, base, "planted")


def test_planted_may_repeat_a_pre_post_pair_and_is_not_refused_for_it():
    """2.12's third disclosed reading: "no self-edges or multi-edges" is a property 2.2 states for
    the RANDOM rewire, not a nuisance property every arm holds. The unique-pair form leaves the
    positive control UNBUILDABLE wherever the edge count exceeds the class x block product, so the
    repeat is permitted for PLANTED and PLANTED alone -- and that is asserted, not assumed."""
    base = _art()
    classes = H.drive_classes(base, k=H.K_FIXED, seed=0)
    planted = H.arm_planted(base, classes, seed=0)
    H.check_nuisance(planted, base, "planted")  # must NOT raise
    with pytest.raises(H.ArmRefused):
        H.check_nuisance(planted, base, "planted", unique_edges=True)


# ---- 2.12(a): the task instrument is a FIXED per-seed orthonormal random projection -----------
def test_the_feature_is_a_fixed_per_seed_orthonormalised_gaussian_projection():
    proj = H.random_projection(50, 8, seed=3)
    assert proj.shape == (50, 8)
    assert np.allclose(proj.T @ proj, np.eye(8), atol=1e-9), "the columns must be orthonormal"
    assert np.array_equal(proj, H.random_projection(50, 8, seed=3)), "FIXED per seed"
    assert not np.allclose(proj, H.random_projection(50, 8, seed=4)), "and a draw per seed"
    with pytest.raises(H.ArmRefused):
        H.random_projection(4, 8, seed=0)  # D > n_feat has no orthonormal basis


def test_the_D_ladder_takes_the_LARGEST_D_under_the_ceiling_and_REFUSES_when_none_is():
    assert H.D_CANDIDATES == (4, 8, 16, 32, 64, 128)
    assert H.K_FIXED == 8
    assert H.D_CEILING == 0.9
    readings = [
        {"d": 4, "accuracy": 0.30},
        {"d": 8, "accuracy": 0.55},
        {"d": 16, "accuracy": 0.88},
        {"d": 32, "accuracy": 0.95},
        {"d": 64, "accuracy": 1.0},
        {"d": 128, "accuracy": None},
    ]
    assert H.choose_d(readings) == 16, "the LARGEST D under the ceiling, not the first"
    assert H.choose_d([{"d": d, "accuracy": 1.0} for d in H.D_CANDIDATES]) is None
    assert H.choose_d([{"d": 4, "accuracy": None}]) is None
    assert H.choose_d([]) is None


def test_the_arm_scores_from_a_D_DIMENSIONAL_feature_not_from_the_raw_counts():
    art = _art()
    classes = H.drive_classes(art, k=4, seed=0)
    r = H.run_arm(
        art, "seeded", 0, classes, d=4, blocks=1, t_trial=10, trials_per_class=4, held_out=1
    )
    assert r.feature_dim == 4, "the readout must see D dimensions, never the raw non-driven counts"
    assert 0.0 <= r.accuracy <= 1.0


# ---- 2.12(c): the settling clause, read as the ABSENCE of a monotone trend --------------------
def test_the_settling_clause_is_read_as_the_absence_of_a_monotone_trend_over_three_windows():
    assert H.is_settled([112.000, 122.906, 106.266]) is True
    assert H.is_settled([112.0, 122.9, 106.3, 133.6, 117.8]) is True
    assert H.is_settled([100.0, 110.0, 120.0]) is False, "a rising trend is NOT settled"
    assert H.is_settled([120.0, 110.0, 100.0]) is False, "a falling trend is NOT settled"
    assert H.is_settled([100.0, 110.0]) is False, "fewer than three windows is not a reading"
    assert H.is_settled([]) is False


# ---- 2.12: the campaign runs iff the guard selected a D AND the WRONG arm is valid ------------
def test_the_campaign_guard_refuses_without_a_D_and_refuses_an_invalid_wrong_arm():
    ok, why = H.campaign_guard(
        {
            "D_rule": {"D": 16, "satisfied": True},
            "wrong_validity": {"valid": True},
            "W": {"frac": 1.0},
        }
    )
    assert ok is True, why
    ok, why = H.campaign_guard(
        {
            "D_rule": {"D": None, "satisfied": False},
            "wrong_validity": {"valid": True},
            "W": {"frac": 1.0},
        }
    )
    assert ok is False and "REFUSED ON MEASUREMENT" in why
    ok, why = H.campaign_guard(
        {
            "D_rule": {"D": 16, "satisfied": True},
            "wrong_validity": {"valid": False},
            "W": {"frac": 1.0},
        }
    )
    assert ok is False and "INVALID" in why
    ok, why = H.campaign_guard(
        {"D_rule": {"D": 16, "satisfied": True}, "wrong_validity": {"valid": True}, "W": None}
    )
    assert ok is False


def test_an_arm_that_violates_a_nuisance_property_is_REFUSED():
    base = _art()
    bad = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    bad["sign"] = -bad["sign"]
    with pytest.raises(H.ArmRefused):
        H.check_nuisance(bad, base, "sign-flipped")

    bad2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    bad2["pre_idx"] = bad2["pre_idx"][:-1]
    bad2["post_idx"] = bad2["post_idx"][:-1]
    bad2["n_syn"] = bad2["n_syn"][:-1]
    with pytest.raises(H.ArmRefused):
        H.check_nuisance(bad2, base, "edge-count")

    bad3 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    bad3["n_syn"] = bad3["n_syn"] + 1
    with pytest.raises(H.ArmRefused):
        H.check_nuisance(bad3, base, "weight-multiset")


# ---- width, drive, read-once -----------------------------------------------------------------
def test_band_balanced_subgraph_keeps_every_band_in_proportion():
    art = H.synthetic_artifact(n_units=140, seed=0)
    half = H.band_balanced_subgraph(art, 0.5)
    full = np.bincount(art["layer"].astype(int), minlength=7)
    got = np.bincount(half["layer"].astype(int), minlength=7)
    assert half["agglo_id"].size <= art["agglo_id"].size // 2 + 7
    for b in range(7):
        assert got[b] <= full[b]
        if full[b] >= 2:
            assert got[b] >= 1, "a band must not be emptied by the width rule"


def test_drive_classes_are_disjoint_and_deterministic():
    art = _art()
    a = H.drive_classes(art, k=4, seed=3)
    b = H.drive_classes(art, k=4, seed=3)
    assert a == b
    seen = set()
    for cls in a:
        assert not (seen & set(cls))
        seen |= set(cls)
    assert H.drive_classes(art, k=4, seed=4) != a


def test_the_prior_is_read_once_at_construction():
    """specs/011 data-model: nothing on a tick re-reads the artifact. The arm's table is built from
    the artifact ONCE; mutating the artifact afterwards must not change a running population."""
    art = _art()
    from snn_data_lake.cortex import h01_ingest as I

    _state, syn, _kt = I.population(art)
    before = {k: v[1].copy() for k, v in syn.bundles.items()}
    art["n_syn"] = art["n_syn"] * 7
    for k, v in syn.bundles.items():
        assert np.array_equal(before[k], v[1])


# ---- the offline self-check runs, both reds fire, PLANTED fires ------------------------------
def test_the_offline_selfcheck_runs_and_fires_both_reds():
    out = H.selfcheck()
    assert out["red1"] is True, "red 1 must FIRE in the self-check"
    assert out["red2"] is True, "red 2 must FIRE in the self-check"
    assert out["planted"] is True, "the positive control must fire"
    assert out["seconds"] < 60.0, f"the offline self-check took {out['seconds']:.1f}s"
    assert math.isfinite(out["seconds"])
