"""The three-arm drift harness: both SC-006 reds, the positive control, the refusals (T063).

specs/011 asked for a DriftRecord and a seeded / random / deliberately-wrong comparison and left the
task, the distance, the width and the null's design open; specs/022 SC-006 asks the seeded arm to
beat BOTH comparators. The pre-registration `specs/022-unified-brain-arc/evidence/
t063-prior-drift-harness.md` closes those, and this file is the reds and controls it names.

Committed RED, before `cortex/prior_drift_harness.py` exists.

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
    frozen = H.ArmResult("frozen", True, 0.2, 40, 900, 0, 0.0, ())
    fired, msg = H.red1(frozen)
    assert fired is True
    assert "seeding approach failed" in msg and "PINNED (lr = 0 by construction)" in msg
    assert "executed 40" in msg and "attempted 900" in msg and "effective 0" in msg


def test_red1_refuses_to_call_a_learning_arms_zero_pinning():
    """Codex defect 4: a zero can come from cancellation, quantization or absent pairs. Only the
    VERIFIED intervention is called pinning."""
    cancelled = H.ArmResult("seeded", False, 0.2, 40, 900, 0, 0.0, ())
    fired, msg = H.red1(cancelled)
    assert fired is False
    assert "pinn" not in msg.lower() and "cancelled or below quantization" in msg
    no_pairs = H.ArmResult("seeded", False, 0.2, 40, 0, 0, 0.0, ())
    fired, msg = H.red1(no_pairs)
    assert fired is False and "no eligible spike pairs" in msg


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


def test_wrong_reverses_inter_band_edges_only():
    base = _art()
    wrong = H.arm_wrong(base)
    H.check_nuisance(wrong, base, "wrong")
    lay = base["layer"]
    intra = lay[base["pre_idx"]] == lay[base["post_idx"]]
    assert np.array_equal(base["pre_idx"][intra], wrong["pre_idx"][intra])
    inter = ~intra
    assert np.array_equal(base["pre_idx"][inter], wrong["post_idx"][inter])
    assert np.array_equal(base["post_idx"][inter], wrong["pre_idx"][inter])


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
