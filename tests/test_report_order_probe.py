"""The reds for T081 -- specs/022 [US4], the report-order probe
(evidence/t081-report-order.md 2.4).

WRITTEN AND COMMITTED **RED** against the 2.9 step-2 skeleton: every failure is an ASSERTION on
a constructed input, never a collection ImportError (T077's panel named that shape; T079 and T080
followed it). Each red below opens on the step-2 STUB its own reading depends on -- the stubs
return null readings, so each test fails on its FIRST assertion and prints one line naming what
the instrument owes.

WHY THE MODULE IS IMPORTED AS A MODULE and not through `from ... import (names)`, which is the
sibling suites' shape: 2.9 step 2 fixes the skeleton's stub list, and the CONSTRUCTED INPUTS
(`t081_constructed_rows`, `t081_canned_snapshot`, ...) and the MUTATED SCORERS
(`_t081_mutated_scorer`) are not in it -- they arrive at 2.9 step 4. A name-level import of those
would make every red a COLLECTION error at commit 3 instead of an assertion, which is exactly the
shape 2.4 forbids. The module alias costs one prefix and buys six honest reds.

  R1 THE CLONE IS THE WORKSPACE AND NOTHING ELSE -- on a constructed `GlobalWorkspace(seed=0)`
     with one subscriber counting deliveries and six pools posted over 20 ticks: the clone
     re-stepped reproduces the real report on every tick, the BASELINE the ten arms are cloned
     from is unchanged after all of them have been stepped, and the subscriber counts the REAL
     steps only -- a clone delivers to nobody. MUTATION `clone_shares_state`: `_g` is not
     deep-copied, so an arm's step corrupts the baseline the next arm clones.
  R2 THE DISSOCIATION ON A CONSTRUCTED WORKSPACE -- six pools with distinct saliences, the top
     rotating by tick: lesioning the REPORTED content at dose 1.0 moves the report on every
     applicable tick, lesioning an UNREPORTED one moves it on none, and over eight such seeds the
     instrument reads (g0) MET, (c1) MET and (c2) MET -- verdict PASS. MUTATION `blind_scorer`:
     the two arms swapped.
  R3 A CANNED REPORTER READS NOT MET -- the same subject read through a `snapshot` stand-in that
     returns ONE fixed record (whose content is a real pool, so the lesion stays applicable):
     f_named(1.0) = 0 on every seed and (c1) NOT MET. MUTATION `scorer_reads_workspace`: the
     probe reads `ws.current` for the lesion target AND the comparison, and reads MET on a
     reporter that says the same thing every tick -- the probe must read the REPORT, not the
     state.
  R4 A HALLUCINATING REPORTER READS NOT MET -- a `snapshot` stand-in whose content is drawn from
     its own RNG on every call: f_scr(1.0) > 0 and (c2), the C5 row's own dissociation input,
     reads NOT MET with the seeds NAMED. MUTATION `c2_always_met`. Its second catch -- (r) also
     refuses, because the unperturbed twin disagrees too -- is ASSERTED beside the red rather
     than left implicit.
  R5 THE READABILITY REFUSAL -- a clone that SHARES the original's `_rng` re-steps to a different
     report on some tick of a subject whose top two saliences are tied, so (r) REFUSES; the same
     recording with fact 3's clone refuses on nothing. MUTATION `r_ignores_mismatch`.
  R6 THE DOSE LADDER -- constructed per-dose fractions (0.0, 0.1, 0.6, 1.0) read a flip dose of
     0.75, and an all-zero ladder reads "none". MUTATION `ladder_first_dose`: the first dose
     tested, whatever its fraction.

Every red also asserts `t081_reds()[name]`, the module's own self-check line: 2.4 requires each
mutation to be applied IN THE TEST *and* in `t081_reds()`, so the suite and the gate can never
disagree about which reds ran.

No brain, no lake, no network: R1-R6 all run on constructed workspaces. The live reading (a
scratch brain, seeds 0-7, 600 ticks) is NOT here -- the pin is written once by `--t081-campaign`.

  set PYTHONUTF8=1 & set PYTHONPATH=src
  .venv-lake/Scripts/python.exe -m pytest tests/test_report_order_probe.py -q --no-cov
"""

import pytest

from snn_data_lake.cortex import report_order_probe as rop
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace


@pytest.fixture(scope="module")
def reds():
    """`t081_reds()` once per module -- the in-command control every red also asserts."""
    return rop.t081_reds()


# ---------------------------------------------------------------------------------------
# R1 -- THE CLONE IS THE WORKSPACE AND NOTHING ELSE
# ---------------------------------------------------------------------------------------
def test_r1_the_clone_is_the_workspace_and_nothing_else(reds):
    bare = GlobalWorkspace(seed=0)
    clone = rop.t081_clone(bare)

    assert clone is not None, (
        "t081_clone must return a STEP-ABLE COPY of the workspace's competition state (fact 3) "
        "-- the 2.9 step-2 stub returns the null reading, so the whole apparatus is absent"
    )
    assert clone._g is not bare._g, (
        "the clone's attractor charges must be a DEEP COPY: sharing `_g` makes a lesion on the "
        "copy a lesion on the original"
    )
    assert clone._subscribers == [] and clone._out_ring is None, (
        "a clone must deliver to NOBODY and frame NOTHING into the lake "
        f"(subscribers {clone._subscribers}, ring {clone._out_ring})"
    )
    assert "step" not in clone.__dict__, (
        "the instance-level `step` wrapper must be popped out of the clone, or a clone of a "
        "RECORDING workspace re-enters the recorder"
    )

    rec = rop.t081_constructed_rows(seed=0)
    read = rop.t081_read(rec["rows"])
    assert read["n"] == rop.T081_R_TICKS, (
        f"the recording must carry every constructed tick ({read['n']} of {rop.T081_R_TICKS})"
    )
    assert read["mismatch"] == 0, (
        f"the UNPERTURBED clone re-stepped must reproduce the real report on EVERY tick "
        f"(content, availability and strength) -- {read['mismatch']} of {read['n']} differ"
    )
    assert read["pre_intact"] == rop.T081_R_TICKS, (
        f"the BASELINE all ten arms are cloned from must be unchanged after every arm has been "
        f"stepped ({read['pre_intact']} of {rop.T081_R_TICKS} ticks intact) -- otherwise the "
        f"arms are not paired by construction"
    )
    assert rec["deliveries"] == rop.T081_R_TICKS, (
        f"the subscriber must be delivered the REAL steps and nothing else: {rop.T081_R_TICKS} "
        f"ticks carry ten workspace steps each and it counted {rec['deliveries']}"
    )

    # THE MUTATION, RUN: `_g` shared with the original instead of deep-copied.
    shares = rop._t081_mutated_scorer("clone_shares_state")
    assert shares(bare)._g is bare._g, (
        "the mutation must actually SHARE the charges, or it reddens nothing"
    )
    mut = rop.t081_constructed_rows(seed=0, clone_fn=shares)
    mut_read = rop.t081_read(mut["rows"])
    assert mut_read["pre_intact"] < rop.T081_R_TICKS, (
        f"a clone SHARING `_g` must corrupt the baseline the next arm clones "
        f"({mut_read['pre_intact']} of {rop.T081_R_TICKS} ticks still intact)"
    )
    assert mut["deliveries"] == rop.T081_R_TICKS, (
        "even the broken clone must deliver to nobody -- `_subscribers` is detached on both "
        "paths, and a red about state sharing must not be carried by a delivery leak"
    )
    assert reds["R1"], "t081_reds()['R1'] must fire on the same constructed subject the suite used"


# ---------------------------------------------------------------------------------------
# R2 -- THE DISSOCIATION ON A CONSTRUCTED WORKSPACE
# ---------------------------------------------------------------------------------------
def test_r2_the_dissociation_on_a_constructed_workspace(reds):
    assert rop.t081_contrasts([])["c1"] is not None, (
        "t081_contrasts must DECIDE (c1) through substrate/paired_contrast.contrast, even on an "
        "empty sample -- the 2.9 step-2 stub returns the null contrasts, so no conjunct is read"
    )

    per_seed = [rop.t081_constructed_read(seed=s) for s in rop.T081_R_SEEDS]
    got = rop.t081_contrasts(per_seed)
    d = rop.T081_LESION_DOSE

    assert got["n_seeds"] == len(rop.T081_R_SEEDS), (
        f"the contrasts read every one of the {len(rop.T081_R_SEEDS)} seeds (read {got['n_seeds']})"
    )
    assert all(e["f_named"][d] == 1.0 for e in per_seed), (
        f"lesioning the REPORTED content at dose {d} must move the report on EVERY applicable "
        f"tick of a clean subject (read {[e['f_named'][d] for e in per_seed]})"
    )
    assert all(e["f_scr"][d] == 0.0 for e in per_seed), (
        f"lesioning an UNREPORTED pool at dose {d} must move it on NONE "
        f"(read {[e['f_scr'][d] for e in per_seed]})"
    )
    assert got["g0"]["met"], (
        f"the positive control that is not the method: the constructed subject's report must "
        f"VARY, or the dissociation is unreadable on it too ({got['g0']['failures']})"
    )
    assert got["c1"].met and got["c2"]["met"], (
        f"the instrument must read (c1) MET and (c2) MET on eight such seeds "
        f"(c1 {got['c1']}, c2 {got['c2']})"
    )
    assert rop.t081_verdict(got, []) == "PASS", (
        f"with (g0), (c1) and (c2) all MET and no refusal, 2.3's verdict is PASS "
        f"(read {rop.t081_verdict(got, [])!r})"
    )

    # THE MUTATION, RUN: the named and scrambled arms swapped.
    mut = rop.t081_contrasts(per_seed, score=rop._t081_mutated_scorer("blind_scorer"))
    assert not mut["c1"].met, (
        f"a scorer blind to WHICH arm is the named one must lose the dissociation "
        f"(it read {mut['c1']})"
    )
    assert reds["R2"], "t081_reds()['R2'] must fire on the same seeds the suite used"


# ---------------------------------------------------------------------------------------
# R3 -- A CANNED REPORTER READS NOT MET
# ---------------------------------------------------------------------------------------
def test_r3_a_canned_reporter_reads_not_met(reds):
    assert rop.t081_contrasts([])["c1"] is not None, (
        "R3 is a reading of (c1) on a canned recording: t081_contrasts must decide it, and the "
        "2.9 step-2 stub returns the null contrasts"
    )

    d = rop.T081_LESION_DOSE
    canned = [
        rop.t081_constructed_read(seed=s, snapshot_fn=rop.t081_canned_snapshot())
        for s in rop.T081_R_SEEDS
    ]
    got = rop.t081_contrasts(canned)

    assert all(e["applicable_named"] == e["n"] for e in canned), (
        "the canned record's content is a REAL pool, so the named lesion stays applicable on "
        "every tick -- otherwise this red would be testing the applicability floor instead "
        f"({[(e['applicable_named'], e['n']) for e in canned]})"
    )
    assert all(e["f_named"][d] == 0.0 for e in canned), (
        f"a reporter that returns ONE fixed record cannot move under any lesion "
        f"(read {[e['f_named'][d] for e in canned]})"
    )
    assert not got["c1"].met, (
        f"the licensed sentence must read NOT MET on a canned reporter (read {got['c1']})"
    )

    # THE MUTATION, RUN: the probe reads `ws.current` -- the STATE -- instead of the record.
    mut = rop.t081_contrasts(
        [
            rop.t081_constructed_read(
                seed=s,
                snapshot_fn=rop.t081_canned_snapshot(),
                content_of=rop._t081_mutated_scorer("scorer_reads_workspace"),
            )
            for s in rop.T081_R_SEEDS
        ]
    )
    assert mut["c1"].met, (
        f"a probe that reads the WORKSPACE instead of the REPORT reads MET on the very reporter "
        f"that says the same thing every tick (it read {mut['c1']}) -- this row's claim is about "
        f"the report channel's record, and nothing else"
    )
    assert reds["R3"], "t081_reds()['R3'] must fire on the same canned recording the suite used"


# ---------------------------------------------------------------------------------------
# R4 -- A HALLUCINATING REPORTER READS NOT MET
# ---------------------------------------------------------------------------------------
def test_r4_a_hallucinating_reporter_reads_not_met(reds):
    assert rop.t081_contrasts([])["c2"] is not None, (
        "R4 is a reading of (c2), the C5 row's own dissociation input: t081_contrasts must "
        "decide it, and the 2.9 step-2 stub returns the null contrasts"
    )

    d = rop.T081_LESION_DOSE
    hall = [
        rop.t081_constructed_read(seed=s, snapshot_fn=rop.t081_hallucinating_snapshot(s))
        for s in rop.T081_R_SEEDS
    ]
    got = rop.t081_contrasts(hall)

    assert all(e["f_scr"][d] > 0.0 for e in hall), (
        f"a reporter drawing its content from its own RNG must 'move' under the SCRAMBLED lesion "
        f"too (read {[round(e['f_scr'][d], 3) for e in hall]})"
    )
    assert not got["c2"]["met"], (
        f"the dissociation input must read NOT MET on a hallucinating reporter (read {got['c2']})"
    )
    assert got["c2"]["failures"], (
        "the refusal must NAME the seeds whose scrambled lesion moved the report, never only "
        "that some did"
    )
    # THE SECOND CATCH, asserted rather than implied: (r) refuses too, because the UNPERTURBED
    # twin's report disagrees with the real one on a reporter that redraws every call.
    assert all(rop.t081_refusals(e) for e in hall), (
        "a hallucinating reporter must ALSO fail the readability clause -- two instruments "
        "catching the same fault is the reading, not a redundancy"
    )

    # THE MUTATION, RUN: a (c2) that cannot say no.
    mut = rop.t081_contrasts(hall, c2=rop._t081_mutated_scorer("c2_always_met"))
    assert mut["c2"]["met"], (
        "an ALWAYS-ZERO (c2) must pass the hallucinating reporter -- a dissociation input that "
        "cannot refuse is not an input"
    )
    assert reds["R4"], "t081_reds()['R4'] must fire on the same recordings the suite used"


# ---------------------------------------------------------------------------------------
# R5 -- THE READABILITY REFUSAL
# ---------------------------------------------------------------------------------------
def test_r5_the_readability_refusal(reds):
    bare = GlobalWorkspace(seed=0)
    assert rop.t081_clone(bare) is not None, (
        "R5 contrasts fact 3's CLONE with one that shares the original's `_rng`; the 2.9 step-2 "
        "stub returns the null reading, so neither exists yet"
    )

    shared = rop.t081_constructed_read(seed=0, clone_fn=rop.t081_clone_shares_rng, fresh=True)
    clean = rop.t081_constructed_read(seed=0, fresh=True)

    assert shared["mismatch"] > 0, (
        f"a clone SHARING the original's `_rng` must re-step to a DIFFERENT report on some tick "
        f"(mismatch {shared['mismatch']} of {shared['n']})"
    )
    assert clean["mismatch"] == 0, (
        f"the SAME subject read with fact 3's clone must reproduce the report on every tick "
        f"({clean['mismatch']} of {clean['n']} differ) -- otherwise the red is about the subject"
    )
    assert rop.t081_refusals(shared), "(r) must REFUSE the shared-RNG recording, and say why"
    assert any("clone is not the workspace" in why for why in rop.t081_refusals(shared)), (
        f"the refusal must name the READABILITY clause, not some other one "
        f"({rop.t081_refusals(shared)})"
    )
    assert not rop.t081_refusals(clean), (
        f"(r) must refuse NOTHING on the clean twin ({rop.t081_refusals(clean)})"
    )

    # THE MUTATION, RUN: a readability clause that ignores the mismatch.
    mut = rop.t081_refusals(shared, mismatch=rop._t081_mutated_scorer("r_ignores_mismatch"))
    assert not mut, (
        f"a clause that ignores the mismatch count cannot refuse a twin that re-steps to a "
        f"different report ({mut})"
    )
    assert reds["R5"], "t081_reds()['R5'] must fire on the same recording the suite used"


# ---------------------------------------------------------------------------------------
# R6 -- THE DOSE LADDER
# ---------------------------------------------------------------------------------------
def test_r6_the_dose_ladder(reds):
    assert rop.t081_flip_dose(rop.T081_R6_LADDER) == rop.T081_R6_FLIP, (
        f"the flip dose is the SMALLEST dose whose named fraction reaches "
        f"{rop.T081_FLIP_FLOOR}: {rop.T081_R6_LADDER} reads "
        f"{rop.t081_flip_dose(rop.T081_R6_LADDER)!r}, not {rop.T081_R6_FLIP}"
    )
    assert rop.t081_flip_dose(rop.T081_R6_ALL_ZERO) == "none", (
        f"a ladder that never reaches the floor reads 'none', never a dose "
        f"(read {rop.t081_flip_dose(rop.T081_R6_ALL_ZERO)!r})"
    )

    # THE MUTATION, RUN: the first dose tested, whatever its fraction.
    mut = rop._t081_mutated_scorer("ladder_first_dose")
    assert mut(rop.T081_R6_LADDER) != rop.T081_R6_FLIP, (
        f"a reader that returns the FIRST dose reports the ARM ORDER, not the dose response "
        f"(it read {mut(rop.T081_R6_LADDER)!r})"
    )
    assert reds["R6"], "t081_reds()['R6'] must fire on the same ladder the suite used"
