"""The reds for Gate Zero -- specs/022 S6 (evidence/s6-kernel-wire.md) x specs/008 plan.md.

WRITTEN AND COMMITTED **RED**, before `gates/amdahl_gate.py` exists, per the pre-registration's
commit order (s6-kernel-wire.md 2.8 item 2). The import below is the red: collection fails until the
instrument lands. A red nobody has seen fire is a red nobody can trust, and an instrument whose
criteria first appear beside its implementation has no evidence its criteria could ever have failed.

These are the three self-check reds of 2.2(iv) plus the "instrument blind" disposition of 2.6 row 3:

  (a) POSITIVE CONTROL -- a constructed tick harness whose object path is padded with a KNOWN extra
      cost per unit must read a LARGE ceiling. This is the only red that proves the measurement core
      can see a ceiling at all; every DECLINE the gate ever prints is worthless without it.
  (b) REPEATABILITY -- two neutered readings of the same harness must agree, judged by their 95%
      bootstrap bands OVERLAPPING rather than by a hand-picked epsilon.
  (c) SHAM NEUTER -- a stand-in installed on a path with ZERO share must read a ceiling within noise
      of zero. This is the control that is not the method (2.5): it uses the same install machinery
      and the same measurement core, and changes nothing.
  (d) INSTRUMENT BLIND (2.6 row 3) -- when (a) does not fire, the disposition fixed in advance is
      "no reading; fix it first". The gate must REFUSE to take a reading, and must refuse BEFORE it
      constructs a brain, so the refusal is cheap and unmissable.

Every red here runs on the constructed harness, never on the live brain: they are fast, hermetic and
touch no lake. DISCLOSED LIMIT, stated because it is the honest one: the harness shares the
measurement CORE with the live reading but NOT its substrate, so (a) proves the core can see a
ceiling -- it does not prove the core can see one on the live brain. The live reading carries its
own same-substrate control (the sham arm on the brain) for that.
"""

import pytest

from snn_data_lake.gates import amdahl_gate

# wall-clock repeatability: run serially, after the parallel pass (battery: -m timing)
pytestmark = pytest.mark.timing


def test_a_padded_object_path_reads_a_large_ceiling():
    """POSITIVE CONTROL: pad the object path with a known cost per unit; the ceiling must rise."""
    ok, msg, detail = amdahl_gate.red_padded_share()
    assert ok, f"the instrument cannot see a ceiling it was handed on a plate: {msg}"
    assert detail["ceiling_raw"] >= amdahl_gate.PADDED_MIN_CEILING, msg
    # the pad must actually have been installed -- a swap that silently fails to install measures
    # the unmodified code and calls it a pass (specs/008 plan.md, standing invariant 5)
    assert detail["pad_installed"] is True, msg
    assert detail["pad_calls"] > 0, "the padded stand-in was never called -- nothing was padded"


def test_b_two_neutered_readings_on_the_same_harness_agree_within_the_band():
    """REPEATABILITY: the same measurement, twice, must not disagree with itself."""
    ok, msg, detail = amdahl_gate.red_repeatability()
    assert ok, f"the instrument disagrees with itself: {msg}"
    lo1, hi1 = detail["band_a"]
    lo2, hi2 = detail["band_b"]
    assert lo1 <= hi2 and lo2 <= hi1, f"the two bands do not overlap: {detail}"


def test_c_the_sham_neuter_on_a_zero_share_path_reads_within_noise_of_zero():
    """THE CONTROL THAT IS NOT THE METHOD: install a stand-in that removes no work."""
    ok, msg, detail = amdahl_gate.red_sham_neuter()
    assert ok, f"a stand-in on a zero-share path read a ceiling: {msg}"
    # "within noise of zero" as pre-registered (2.2 iv / 2.5): the magnitude bar OR the sham's own
    # paired band containing zero -- AMENDED 2026-09-15 after two batteries redded this line on host
    # load with no tree defect (s6-kernel-wire.md 3.8); the clause that held is named in the message
    assert detail["sham_within_noise"] is True, msg
    assert detail["sham_noise_clause"] in ("magnitude", "noise"), msg
    # the sham must be INSTALLED (else it is not a control, it is an omission) and must be
    # measured to carry zero share (call count 0 is the proof, not the assumption)
    assert detail["sham_installed"] is True, msg
    assert detail["sham_calls"] == 0, "the 'zero-share' path was called -- it is not zero-share"


def test_within_noise_of_zero_clauses_and_their_reds():
    """The amended sham criterion (2026-09-15) on constructed (ceiling, band) pairs; its reds first.

    RED 1: a sham significantly different from zero (band EXCLUDES zero, |c| over the bar) must
    FAIL -- the instrument's own bad draw stays a failure. RED 2: the same with the band on the
    other side of zero. Then the two clauses that HOLD: the magnitude bar alone, and a wide band
    that contains zero (the two battery readings of 2026-09-15, quoted)."""
    f = amdahl_gate.within_noise_of_zero
    assert f(0.08, (0.06, 0.10)) == (False, None), "a band excluding zero must not hold"
    assert f(-0.08, (-0.10, -0.06)) == (False, None), (
        "a band excluding zero must not hold (negative)"
    )
    assert f(0.0046, (-0.00464, 0.01039)) == (True, "magnitude")
    assert f(-0.1025, (-0.28571, 0.00911)) == (True, "noise"), "battery run 2's constructed sham"
    assert f(0.05844, (-0.0802, 0.10295)) == (True, "noise"), "battery run 3's live sham arm"
    # the bar still bites when the band can resolve it: over the bar, the noise clause is the ONLY
    # way through, so a band that does not contain zero is a FAIL even at |c| just over the bar
    assert f(0.0501, (0.0480, 0.0522)) == (False, None)
    # RED 3 (2026-09-16, the T077+T078+T079 stack's battery, s6-kernel-wire.md 3.9): a band that
    # excludes zero but is WIDER than the resolution the decision needs (TOLERANCE 0.15) is not a
    # verdict on the sham -- the instrument could not resolve the draw. It does NOT hold (rc stays
    # 1) and it is named as unresolvable, never as the sham failing. The battery's own numbers:
    assert f(-0.05595, (-0.3138, -0.01658)) == (False, "unresolvable"), (
        "battery-stack6's live sham arm: band 0.297 wide against a 0.15 resolution"
    )
    assert f(0.08, (0.06, 0.10)) == (False, None), "a NARROW band excluding zero is still a FAIL"


def test_the_instrument_blind_disposition_refuses_the_reading():
    """2.6 row 3: the padded-share red does not fire -> the instrument is blind -> NO reading."""
    rc, report = amdahl_gate.run_reading(seed=0, pad_iters=0)
    assert rc == 1, "a blind instrument must exit nonzero"
    assert report.get("refused"), f"a blind instrument reported instead of refusing: {report}"
    assert "blind" in report["refused"].lower()
    # the refusal must carry NO reading: an instrument that prints a ceiling it cannot see is
    # worse than one that prints nothing
    assert "ceiling_raw" not in report
    assert "verdict" not in report


def test_the_repeatability_criterion_fires_on_a_deliberately_broken_arm():
    """THE ONE FIRED ARM, with its control in the same test.

    `red_repeatability` runs the identical unperturbed pair twice, so its criterion had no perturbed
    arm anywhere in the tree: the command that printed "3 reds fired their broken arms" showed no
    criterion failing. Re-read with the second reading on the PADDED rig, the bands must read
    DISJOINT -- a broken arm that PASSES is itself the failure. The control is the pair AS SHIPPED,
    which must still OVERLAP, or the arm is measuring ambient rather than the perturbation.
    """
    fired, msg, detail = amdahl_gate.red_repeatability_broken_arm()
    assert detail["bands_overlap"] is False, f"a broken arm that PASSES is the failure: {msg}"
    assert fired is True, msg
    assert detail["broken_ceiling"] > detail["control_ceiling"], detail

    ok, cmsg, cdetail = amdahl_gate.red_repeatability()
    assert ok, f"the control must still overlap, else the arm reads ambient, not the pad: {cmsg}"
    lo1, hi1 = cdetail["band_a"]
    lo2, hi2 = cdetail["band_b"]
    assert lo1 <= hi2 and lo2 <= hi1, cdetail


def test_the_reds_are_the_same_three_the_pre_registration_names():
    """The gate's own red roster must be the pre-registered one -- not a subset it grew into.

    A pre-registered arm dropped at build time and then reported as measured is the failure this
    guards; the roster is asserted by NAME so a quiet deletion reds here.
    """
    names = [r[0] for r in amdahl_gate.self_check_reds(pad_iters=0, quick=True)]
    assert names == ["padded-share", "repeatability", "sham-neuter"], names


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_the_ceiling_refuses_a_nonpositive_live_arm(bad):
    """A ceiling is 1 - neutered/live; a live arm at or below zero is not a measurement."""
    with pytest.raises(ValueError):
        amdahl_gate.paired_ceiling([bad] * 8, [1.0] * 8, [1.0] * 8, [1.0] * 8)
