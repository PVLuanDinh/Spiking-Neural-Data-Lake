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

DISPOSITION 2026-09-09, committed RED a THIRD time before the no-learner instrument exists. The
campaign (`tools/t063_campaign.json`) fired 2.9's null clauses: the LEARNER was DELETED on its own
criterion -- t(d_f) = -13.0154, the FROZEN twin ahead of the learning arm on every one of the 8
seeds -- so `substrate/kernel_plasticity` is gone and the harness is KEPT as a NO-LEARNER
instrument. Red 1's subject, a constructed lr = 0 pin standing against a LEARNING arm, no longer
exists: there is nothing to pin. So red 1's test becomes the NO-LEARNER REFUSAL ("harness defect:
no learner ran" -> no drift verdict) with the control that its message never says PINNED, conjunct
(a) reads NOT READ, and the learner's disposition is DELETED with its dated reason. Every other red
in this file is unchanged and must stay green: PLANTED, the WRONG validity precondition, the
nuisance refusals, the D ladder, the projection, the settling clause and the campaign guard.

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


# ---- RED 1, SUPERSEDED: the NO-LEARNER REFUSAL -----------------------------------------------
def test_red1_is_the_no_learner_refusal_and_reads_no_drift_verdict():
    """The learner was DELETED on its own criterion, so red 1's subject -- a constructed lr = 0
    pin standing against a LEARNING arm -- does not exist. What red 1 reads now is the HARNESS
    DEFECT 2.9 already names: no learner ran, so drift is not measurable and no verdict is read."""
    arm = H.ArmResult("frozen", 0.2, 0, 0, 0, 0.0, _zero_recs())
    fired, msg = H.red1(arm)
    assert fired is False, "a no-learner harness reads NO drift verdict"
    assert "harness defect: no learner ran" in msg
    assert "kernel_plasticity DELETED" in msg and "2026-09-09" in msg
    assert "FR-008/FR-009" in msg and "remain OPEN" in msg
    #: the four diagnostic lines of 2.5 stay in the message, never collapsed to one number
    assert "executed 0" in msg and "attempted 0" in msg and "effective 0" in msg


def test_the_no_learner_refusal_NEVER_says_PINNED():
    """The control on the refusal. Pinning is attributable ONLY to a verified lr = 0 intervention
    against a learner; with no learner in the tree there is no intervention, so the word must not
    appear for ANY arm -- including the one still named `frozen`."""
    for name in ("seeded", "random", "wrong", "planted", "frozen"):
        _fired, msg = H.red1(H.ArmResult(name, 0.2, 0, 0, 0, 0.0, _zero_recs()))
        assert "PINNED" not in msg, f"{name}: the refusal must not claim a pin that was not applied"
        assert "seeding approach failed" not in msg, f"{name}: a missing learner is not a seeding"


def test_no_learner_holds_reads_all_four_lines_and_both_distances():
    """Replaces red 1's control. A learning arm's control was "effective > 0 and distance > 0";
    with no learner the honest reading is its exact inverse, and it is MEASURED per arm rather
    than assumed from the absence of an import. Its red: any nonzero line breaks it."""
    zero = H.ArmResult("seeded", 0.7, 0, 0, 0, 0.0, _zero_recs())
    assert H.no_learner_holds(zero) is True
    assert H.no_learner_holds(H.ArmResult("seeded", 0.7, 1, 0, 0, 0.0, _zero_recs())) is False
    assert H.no_learner_holds(H.ArmResult("seeded", 0.7, 0, 9, 0, 0.0, _zero_recs())) is False
    assert H.no_learner_holds(H.ArmResult("seeded", 0.7, 0, 0, 3, 0.0, _zero_recs())) is False
    assert H.no_learner_holds(H.ArmResult("seeded", 0.7, 0, 0, 0, 0.5, _zero_recs())) is False
    moved = (
        H.DriftRecord("band_pair_weight_share", (1.0,), (0.9,), 0.01, 1),
        H.DriftRecord("band_fanin_e_fraction", (1.0,), (0.9,), 0.10, 1),
    )
    assert H.no_learner_holds(H.ArmResult("seeded", 0.7, 0, 0, 0, 0.0, moved)) is False
    assert H.no_learner_holds(H.ArmResult("seeded", 0.7, 0, 0, 0, 0.0, ())) is False


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
    """The arms as the no-learner instrument produces them: every diagnostic line zero and every
    DriftRecord at distance 0, because nothing in the tree can move a bundle weight any more. Only
    the ACCURACIES differ between arms, which is exactly what contrasts (b) (c) (d) (e) read."""
    out = {}
    for s in range(8):
        out[s] = {
            "seeded": H.ArmResult("seeded", seeded[s], 0, 0, 0, 0.0, _zero_recs()),
            "random": H.ArmResult("random", random_[s], 0, 0, 0, 0.0, _zero_recs()),
            "wrong": H.ArmResult("wrong", wrong[s], 0, 0, 0, 0.0, _zero_recs()),
            "planted": H.ArmResult("planted", planted[s], 0, 0, 0, 0.0, _zero_recs()),
            "frozen": H.ArmResult("frozen", frozen[s], 0, 0, 0, 0.0, _zero_recs()),
        }
    return out


def _zero_recs():
    return (
        H.DriftRecord("band_pair_weight_share", (1.0,), (1.0,), 0.0, 0),
        H.DriftRecord("band_fanin_e_fraction", (1.0,), (1.0,), 0.0, 0),
    )


def test_the_prior_conjuncts_are_read_and_the_DRIFT_conjunct_is_NOT_READ():
    """Every conjunct the no-learner instrument can still read -- (b) (c) (d) (e) -- is read. (a)
    is NOT READ, so SC-006 can never be called MET again by this harness: an instrument that cannot
    measure drift cannot satisfy a criterion whose first conjunct IS drift."""
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.30, 0.32],
            [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92],
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
        ),
        wrong_valid=True,
    )
    assert [v["conjuncts"][k]["state"] for k in "bcde"] == ["MET"] * 4
    assert v["conjuncts"]["a"]["state"] == "NOT READ"
    assert "kernel_plasticity DELETED" in v["conjuncts"]["a"]["note"]
    assert v["sc006"] == "NOT READ"
    text = H.format_verdict(v)
    for k in "abcde":
        assert f"({k})" in text
    assert "per-seed" in text.lower()
    assert "no learner" in text.lower()


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


def test_the_learners_disposition_is_DELETED_and_carries_its_dated_reason():
    """The learner's criterion was read ONCE, on the campaign, and it FAILED: t(d_f) = -13.0154.
    The disposition is therefore not re-derivable from a no-learner run and must not pretend to be
    -- it is reported as EXECUTED, with the date and the reason, whatever the accuracies say."""
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
    assert v["learner"]["disposition"] == "DELETED"
    assert "kernel_plasticity DELETED" in v["learner"]["note"]
    assert "2026-09-09" in v["learner"]["note"]


def test_the_frozen_arm_is_now_an_IDENTITY_control_not_a_learner_control():
    """With no learner, FROZEN is the SEEDED table run twice under one protocol, so d_f must be
    identically 0 -- and a nonzero d_f is then a DETERMINISM defect in the harness, not a learning
    signal. The contrast is kept for exactly that reading, and it is degenerate by construction."""
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50] * 8,
            [0.30] * 8,
            [0.90] * 8,
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
        ),
        wrong_valid=True,
    )
    c = v["contrasts"]["seeded - frozen"]
    assert c["diffs"] == [0.0] * 8, "FROZEN and SEEDED must be the same run without a learner"
    assert c["degenerate"] is True and c["t"] == 0.0


# ---- the verdict owes each arm's ACCURACY AGAINST CHANCE, and marks a contrast between two
#      floor-bound arms NOT INTERPRETABLE (T063 panel 2026-09-09, finding F22) -------------------
_AT_CHANCE_SEEDED = [0.125, 0.0, 0.25, 0.125, 0.0, 0.25, 0.125, 0.125]
_AT_CHANCE_RANDOM = [0.0, 0.25, 0.125, 0.125, 0.25, 0.0, 0.125, 0.125]
_ABOVE_WRONG = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52]
_ABOVE_PLANTED = [0.875, 0.90, 0.85, 0.88, 0.86, 0.90, 0.87, 0.89]
_ABOVE_FROZEN = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72]
_NOT_INTERPRETABLE = (
    "NOT INTERPRETABLE AS A WIRING CONTRAST (both arms fail the above-chance check)"
)


def test_above_chance_reads_an_arm_against_the_1_over_K_floor_with_the_harnesss_own_statistic():
    """The floor rule the tree already wrote for the FIXTURE -- "a positive control that beats a
    floor is a formality, not a reading" -- lifted out of `synthetic_artifact`'s docstring into ONE
    module-level predicate so the same rule governs the fixture and the campaign (P2). It REPORTS:
    the arm's mean, chance = 1/K, and the harness's own paired t / permutation FPR on
    (accuracy - chance) over the seeds."""
    at_floor = H.above_chance(_AT_CHANCE_SEEDED, 8)
    assert at_floor["chance"] == 0.125
    assert at_floor["mean"] == sum(_AT_CHANCE_SEEDED) / 8.0
    assert at_floor["met"] is False, "an arm whose mean sits ON the floor does not clear it"
    clear = H.above_chance(_ABOVE_PLANTED, 8)
    assert clear["met"] is True and clear["t"] > H.T_BAR


def test_a_contrast_between_two_arms_that_both_fail_the_floor_is_marked_NOT_INTERPRETABLE():
    """(b) in the pinned campaign contrasted two arms 0.50 and 0.62 held-out trials above chance on
    a mean gap of one eighth of one trial. The verdict now says so: every arm's accuracy against
    chance is carried beside the t, and a contrast BOTH of whose arms fail the above-chance check
    is marked NOT INTERPRETABLE AS A WIRING CONTRAST. It is a REPORT -- no conjunct, no verdict and
    no threshold moves."""
    v = H.verdict(
        _acc(
            _AT_CHANCE_SEEDED,
            _AT_CHANCE_RANDOM,
            _ABOVE_WRONG,
            _ABOVE_PLANTED,
            _ABOVE_FROZEN,
        ),
        wrong_valid=True,
        k=8,
    )
    vs = v["arms_vs_chance"]
    assert vs["seeded"]["met"] is False and vs["random"]["met"] is False
    assert vs["wrong"]["met"] is True and vs["planted"]["met"] is True
    assert v["contrasts"]["seeded - random"]["wiring_contrast"] == _NOT_INTERPRETABLE
    assert v["contrasts"]["seeded - wrong"]["wiring_contrast"] == "INTERPRETABLE"
    assert v["contrasts"]["planted - random"]["wiring_contrast"] == "INTERPRETABLE"
    #: the mark REPORTS and never refuses: the verdict reads exactly what it read before
    assert v["conjuncts"]["c"]["state"] == "MET"
    assert v["conjuncts"]["b"]["state"] == "NOT MET"
    assert v["sc006"] == "NOT MET"
    text = H.format_verdict(v)
    assert _NOT_INTERPRETABLE in text
    assert "seeded - chance" in text


def test_every_contrast_is_INTERPRETABLE_when_at_least_one_of_its_arms_clears_the_floor():
    """The mark fires only when BOTH arms fail. One arm above the floor is enough for the contrast
    to be a reading about wiring, which is why the mark is not a refusal."""
    v = H.verdict(
        _acc(
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
            [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52],
            [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.30, 0.32],
            [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92],
            [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72],
        ),
        wrong_valid=True,
        k=8,
    )
    assert all(a["met"] for a in v["arms_vs_chance"].values())
    marks = {c["wiring_contrast"] for c in v["contrasts"].values()}
    assert marks == {"INTERPRETABLE"}
    assert _NOT_INTERPRETABLE not in H.format_verdict(v)


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
    """ "the same within-band degree rank": inside EACH band the rank ORDER must be carried into
    the mirrored band, so the permutation moves laminar IDENTITY and nothing else. Monotone is
    asserted PER SOURCE BAND -- ranks from different bands are not comparable, and a test that
    mixes them measures nothing."""
    base = H.synthetic_artifact(n_units=140, seed=0)
    lay = base["layer"].astype(np.int64)
    target, _slot, _ordered, tgt_band = H.band_rank_map(base)
    rank = H.within_band_degree_rank(base)
    moved = 0
    for band in range(7):
        members = np.nonzero(lay == band)[0]
        if members.size == 0 or int(tgt_band[members[0]]) == band:
            continue
        ordered_members = members[np.argsort(rank[members], kind="stable")]
        mapped = target[ordered_members]
        assert np.all(lay[mapped] == H.BAND_PERMUTATION[band]), "a unit left its mirrored band"
        assert np.all(np.diff(rank[mapped]) >= 0), (
            f"band {band}: the within-band rank order was not carried across"
        )
        moved += int(members.size)
    assert moved > 0, "the fixture must have at least one band the permutation moves"


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
    #: and the arm is MEASURED to have run without a learner, on the real per-arm path
    assert H.no_learner_holds(r) is True, "a no-learner arm must read four zero lines and 0 drift"


# ---- 2.12(c): the settling clause, read as the ABSENCE of a monotone trend --------------------
def test_the_settling_clause_is_read_as_the_absence_of_a_monotone_trend_over_three_windows():
    assert H.is_settled([112.000, 122.906, 106.266]) is True
    assert H.is_settled([112.0, 122.9, 106.3, 133.6, 117.8]) is True
    assert H.is_settled([100.0, 110.0, 120.0]) is False, "a rising trend is NOT settled"
    assert H.is_settled([120.0, 110.0, 100.0]) is False, "a falling trend is NOT settled"
    assert H.is_settled([100.0, 110.0]) is False, "fewer than three windows is not a reading"
    assert H.is_settled([]) is False


def test_the_settling_clause_REFUSES_a_physically_impossible_window():
    """MEASURED DEFECT, 2026-09-09: the first campaign run's settling read
    [-515.156, 0.0, 0.0] and `is_settled` returned TRUE -- the daemon restarted mid-probe, the
    process CPU counter went NEGATIVE against a dead subject, and a run of zeros is not monotone,
    so the clause passed on a reading of nothing. A zero is a claim about the INSTRUMENT first."""
    assert H.is_settled([-515.156, 0.0, 0.0]) is False, "a negative CPU window is not a reading"
    assert H.is_settled([0.0, 0.0, 0.0]) is False, "a daemon with a drive loop never reads 0 CPU"
    assert H.is_settled([112.0, 0.0, 106.3]) is False, "one dead window invalidates the reading"
    assert H.is_settled([112.000, 122.906, 106.266]) is True, "the positive control must survive"


def test_three_windows_FALSE_ALARM_on_a_third_of_settled_hosts_so_the_run_takes_more():
    """MEASURED, 2026-09-09: the re-run's settling read [127.047, 121.172, 109.969] on ONE live
    daemon -- a real reading of a steady host that happened to come out strictly decreasing, so the
    monotone test called it NOT settled. That is not a bad host, it is a weak instrument: over n
    windows drawn from a settled duty cycle, exactly 2 of n! orderings are monotone. At n = 3 that
    is one settled host in three. The floor stays at 3 (the amendment says "at least three"); the
    number of windows a RUN takes is what must be large enough for the test to mean anything."""
    import itertools

    for n, expect in ((3, 2 / 6), (5, 2 / 120)):
        alarms = sum(
            1
            for p in itertools.permutations(range(n))
            if not H.is_settled([float(x) + 1.0 for x in p])
        )
        assert alarms / math.factorial(n) == pytest.approx(expect), (
            f"{n} windows: the monotone rule's false-alarm rate on a settled host"
        )
    assert H.SETTLING_WINDOWS == 3, "the amendment's floor is three windows"
    assert H.SETTLING_MEASURED >= 5, (
        "a run must take at least 5 windows: at 3 the clause rejects a settled host 1 time in 3"
    )


def test_the_settling_report_marks_a_daemon_restart_as_UNSETTLED():
    ok = H.settling_report([("34316", 120.297), ("34316", 121.109), ("34316", 112.016)])
    assert ok["settled"] is True and ok["daemon_restarted"] is False
    assert ok["daemon_pid_per_window"] == ["34316", "34316", "34316"]
    moved = H.settling_report([("34316", 120.297), ("33320", 121.109), ("33320", 112.016)])
    assert moved["daemon_restarted"] is True
    assert moved["settled"] is False, "a host that swapped daemons under the probe is not settled"
    dead = H.settling_report([("34316", -515.156), ("none", 0.0), ("none", 0.0)])
    assert dead["daemon_restarted"] is True and dead["settled"] is False
    assert H.settling_report([])["settled"] is False


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


# ---- the offline self-check runs, every REMAINING red fires ----------------------------------
def test_the_offline_selfcheck_runs_and_fires_every_remaining_red():
    out = H.selfcheck()
    assert out["no_learner"] is True, "every arm must read four zero lines and zero drift"
    assert out["no_learner_message"].startswith("harness defect: no learner ran")
    assert "PINNED" not in out["no_learner_message"]
    assert out["frozen_identity"] is True, "FROZEN must reproduce SEEDED exactly with no learner"
    assert out["red2"] is True, "red 2's predicate must still be reachable"
    assert out["planted"] is True, "the positive control must fire"
    assert out["seconds"] < 60.0, f"the offline self-check took {out['seconds']:.1f}s"
    assert math.isfinite(out["seconds"])
