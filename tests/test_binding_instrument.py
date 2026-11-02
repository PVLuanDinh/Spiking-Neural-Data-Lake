"""The reds for T080 -- specs/022 [US4], the temporal-binding instrument
(evidence/t080-temporal-binding.md 2.4).

WRITTEN AND COMMITTED **RED** against the 2.9 step-2 skeleton: every failure is an ASSERTION
on a constructed input, never a collection ImportError (T077's panel named that shape; T079
followed it). The stubs return null readings, so each red below fails on its FIRST assertion
and prints one line naming what the instrument owes.

  R1 THE NULL IS MEMBERSHIP-MATCHED -- on a constructed recognition table (200 ticks, 10
     objects, 60 % congruent) every draw's multiset of audio objects equals the table's, and
     the draws' mean bound fraction lies within 3 sd of the product-of-marginals expectation
     (sum over objects of p_v(o) p_a(o)). MUTATION `null_resamples`: the audio objects drawn
     i.i.d. WITH REPLACEMENT, so a draw's multiset differs from the table's.
  R2 THE DECOMPOSITION IS THE BINDER -- a shipped `CrossModalBinder` taught 10 random latent
     pairs (vision 64-d, audio 128-d, values in [0, 1]), 200 probe pairs with every third
     near a prototype: `bind()["handle"] is not None` equals the decomposition
     (`_project` + `_recognize`, both modalities recognised and naming the same object) on
     all 200. MUTATION `ignore_margin`: recognition with the binder's `recog_margin` lowered
     below every reachable coincidence, which differs on the sub-margin pairs.
  R3 THE WINDOW READER -- constructed lag rates above the derived margin at lags 1-3 and at
     or below it from lag 4 read a window of 4; all-above reads "none". MUTATION
     `window_first_lag`: the first lag tested is returned whatever its rate.
  R4 THE POSITIVE CONTROL REFUSES AN INDEPENDENT STREAM -- constructed CLASSIFIER tables
     (the classifiers' congruence, never the binder's report) whose ALIGNED and NULL
     fractions are sign-balanced noise read (g0) NOT MET, which 2.3 maps to REFUSED;
     genuinely aligned tables read MET. MUTATION `g0_always_met`.
  R5 THE DISSOCIATION ON A CONSTRUCTED REPORT -- per-seed tables whose within fraction
     EQUALS the null's q95 on every seed read (b1) NOT MET (a zero-variance sample whose
     common value is not > 0); tables where within exceeds q95 on every seed read MET.
     MUTATION `blind_scorer`: within and null swapped.
  R6 THE AGREEMENT BAND -- a constructed in-vivo null fraction outside [q005, q995] reads
     (b2) REFUSED with the seed NAMED; inside reads holding. MUTATION `band_unbounded`: the
     band is [0, 1], which can refuse nothing.

Every red also asserts `t080_reds()[name]`, the module's own self-check line: 2.4 requires
each mutation to be applied IN THE TEST *and* in `t080_reds()`, so the suite and the gate can
never disagree about which reds ran.

No brain, no lake, no network: R1-R6 all run on constructed inputs. The live reading (a
scratch brain, seeds 0-7, six arms) is NOT here -- the pin is written once by
`--t080-campaign`.

  set PYTHONUTF8=1 & set PYTHONPATH=src
  .venv-lake/Scripts/python.exe -m pytest tests/test_binding_instrument.py -q --no-cov
"""

import pytest

from snn_data_lake.instruments.binding_instrument import (
    T080_BAR,
    T080_R1_TICKS,
    T080_R2_PROBES,
    T080_R3_ALL_ABOVE,
    T080_R3_FIRST_BELOW,
    T080_R3_LAG_RATES,
    T080_R3_Q95,
    T080_R3_WINDOW,
    T080_REDS_N_DRAWS,
    T080_SEEDS,
    _t080_mutated_scorer,
    t080_constructed_binder,
    t080_constructed_pairs,
    t080_constructed_recs,
    t080_constructed_seeds,
    t080_contrasts,
    t080_null,
    t080_recognitions,
    t080_reds,
    t080_window,
)


@pytest.fixture(scope="module")
def reds():
    """`t080_reds()` once per module -- the in-command control every red also asserts."""
    return t080_reds(n_draws=T080_REDS_N_DRAWS)


# ---------------------------------------------------------------------------------------
# R1 -- THE NULL IS MEMBERSHIP-MATCHED
# ---------------------------------------------------------------------------------------
def test_r1_the_null_is_membership_matched(reds):
    recs = t080_constructed_recs(seed=3)
    null = t080_null(recs, 3, n_draws=T080_REDS_N_DRAWS)

    assert null["n"] == T080_R1_TICKS, (
        f"the null must read every scored tick of the table ({null['n']} of {T080_R1_TICKS})"
    )
    assert null["n_draws"] == T080_REDS_N_DRAWS, (
        f"the null must draw what it was asked for ({null['n_draws']} of {T080_REDS_N_DRAWS})"
    )
    assert null["multiset_ok"], (
        "MEMBERSHIP MATCHED: every draw's multiset of audio objects must equal the table's -- "
        "the same features, the same taught K, only the pairing across windows broken"
    )
    assert null["draws_sd"] > 0.0, "a null with no spread carries no band and no margin"
    assert abs(null["mean"] - null["expected"]) <= 3.0 * null["draws_sd"], (
        f"the draws' mean bound fraction ({null['mean']:.4f}) must lie within 3 sd "
        f"({3.0 * null['draws_sd']:.4f}) of the product-of-marginals expectation "
        f"({null['expected']:.4f})"
    )
    assert null["q005"] <= null["q95"] <= null["q995"], (
        f"the band must bracket the margin (q005 {null['q005']:.4f}, q95 {null['q95']:.4f}, "
        f"q995 {null['q995']:.4f})"
    )

    # THE MUTATION, RUN: the audio objects drawn i.i.d. with replacement.
    mut = _t080_mutated_scorer("null_resamples")(recs, 3, T080_REDS_N_DRAWS)
    assert not mut["multiset_ok"], (
        "a RESAMPLING null must break the multiset -- if it does not, the membership match "
        "is not being checked at all"
    )
    assert reds["R1"], "t080_reds()['R1'] must fire on the same table the suite used"


# ---------------------------------------------------------------------------------------
# R2 -- THE DECOMPOSITION IS THE BINDER
# ---------------------------------------------------------------------------------------
def test_r2_the_decomposition_is_the_binder(reds):
    binder, protos = t080_constructed_binder(seed=0)
    rows = t080_constructed_pairs(binder, protos, seed=1)
    got = t080_recognitions(rows, binder)

    assert got["n"] == T080_R2_PROBES, (
        f"the decomposition must read all {T080_R2_PROBES} probe pairs (read {got['n']})"
    )
    bound = sum(1 for r in got["recs"] if r["bound"])
    assert 0 < bound < T080_R2_PROBES, (
        f"the probe set must contain BOTH bound and unbound pairs (bound {bound} of "
        f"{T080_R2_PROBES}) or the agreement is vacuous"
    )
    sub = sum(
        1 for r in got["recs"] if r["s_v"] < binder.recog_margin or r["s_a"] < binder.recog_margin
    )
    assert sub > 0, (
        f"the probe set must contain a SUB-MARGIN pair (margin {binder.recog_margin}) or "
        f"`ignore_margin` has nothing to change"
    )
    assert got["agree"] == 0, (
        f"the decomposition must equal bind()'s own handle on every pair "
        f"({got['agree']} of {got['n']} disagree)"
    )

    # THE MUTATION, RUN: recognition without the margin.
    mut = t080_recognitions(rows, binder, recognize=_t080_mutated_scorer("ignore_margin")(binder))
    assert mut["agree"] > 0, (
        "recognition WITHOUT recog_margin must diverge from bind()'s handle on a sub-margin "
        "pair -- the margin is part of the mechanism, not of the reader"
    )
    assert reds["R2"], "t080_reds()['R2'] must fire on the same binder the suite used"


# ---------------------------------------------------------------------------------------
# R3 -- THE WINDOW READER
# ---------------------------------------------------------------------------------------
def test_r3_the_window_reader(reds):
    q95, rates = T080_R3_Q95, T080_R3_LAG_RATES

    assert t080_window(rates, q95) == T080_R3_WINDOW, (
        f"the window is the SMALLEST lag at or below the derived margin "
        f"(read {t080_window(rates, q95)!r} on {rates} at q95 {q95})"
    )
    assert t080_window(T080_R3_ALL_ABOVE, q95) == "none", (
        "every lag above the margin reads 'none', never a lag"
    )
    assert t080_window(T080_R3_FIRST_BELOW, q95) == 1, (
        "a first lag already at or below the margin reads 1"
    )

    # THE MUTATION, RUN: the first lag tested, whatever its rate.
    mut = _t080_mutated_scorer("window_first_lag")
    assert mut(rates, q95) != T080_R3_WINDOW, (
        f"a reader that returns the FIRST lag must disagree with the window "
        f"(it read {mut(rates, q95)!r})"
    )
    assert reds["R3"], "t080_reds()['R3'] must fire on the same lag table the suite used"


# ---------------------------------------------------------------------------------------
# R4 -- THE POSITIVE CONTROL REFUSES AN INDEPENDENT STREAM
# ---------------------------------------------------------------------------------------
def test_r4_the_positive_control_refuses_an_independent_stream(reds):
    indep = t080_constructed_seeds("independent", seed=4)
    got = t080_contrasts(indep)

    assert got["n_seeds"] == len(T080_SEEDS), (
        f"(g0) is a paired t over all {len(T080_SEEDS)} seeds (read {got['n_seeds']})"
    )
    assert not got["g0"].met, (
        f"an INDEPENDENT stream must read (g0) NOT MET -- 2.3 maps that to REFUSED, "
        f"'the alignment did not reach the brain' (t {got['g0'].t:.4f})"
    )
    aligned = t080_contrasts(t080_constructed_seeds("aligned", seed=4))
    assert aligned["g0"].met and aligned["g0"].t > T080_BAR, (
        f"a genuinely aligned stream must read (g0) MET above the bar "
        f"(t {aligned['g0'].t:.4f} vs {T080_BAR})"
    )

    # THE MUTATION, RUN: a (g0) that cannot say no.
    mut = t080_contrasts(indep, g0=_t080_mutated_scorer("g0_always_met"))
    assert mut["g0"].met, (
        "an ALWAYS-MET (g0) must pass the independent stream -- a positive control that "
        "cannot refuse is not a control"
    )
    assert reds["R4"], "t080_reds()['R4'] must fire on the same tables the suite used"


# ---------------------------------------------------------------------------------------
# R5 -- THE DISSOCIATION ON A CONSTRUCTED REPORT
# ---------------------------------------------------------------------------------------
def test_r5_the_dissociation_on_a_constructed_report(reds):
    at_margin = t080_contrasts(t080_constructed_seeds("at_margin", seed=5))

    assert at_margin["n_seeds"] == len(T080_SEEDS), (
        f"(b1) is a paired t over all {len(T080_SEEDS)} seeds (read {at_margin['n_seeds']})"
    )
    assert at_margin["b1"].degenerate, (
        "within EQUAL to the null's q95 on every seed is a zero-variance sample and must be "
        "recorded as degenerate, never handed a t"
    )
    assert not at_margin["b1"].met, (
        f"within AT the derived margin must read (b1) NOT MET (t {at_margin['b1'].t:.4f})"
    )
    above = t080_contrasts(t080_constructed_seeds("above", seed=5))
    assert above["b1"].met and above["b1"].t > T080_BAR, (
        f"within ABOVE the margin on every seed must read (b1) MET "
        f"(t {above['b1'].t:.4f} vs {T080_BAR})"
    )

    # THE MUTATION, RUN: within and the null swapped.
    mut = t080_contrasts(
        t080_constructed_seeds("above", seed=5), score=_t080_mutated_scorer("blind_scorer")
    )
    assert not mut["b1"].met, (
        f"a scorer blind to WHICH side is the reading must lose the dissociation "
        f"(t {mut['b1'].t:.4f})"
    )
    assert reds["R5"], "t080_reds()['R5'] must fire on the same tables the suite used"


# ---------------------------------------------------------------------------------------
# R6 -- THE AGREEMENT BAND
# ---------------------------------------------------------------------------------------
def test_r6_the_agreement_band(reds):
    outside = t080_constructed_seeds("band_outside", seed=6)
    got = t080_contrasts(outside)

    assert got["b2"]["n_seeds"] == len(T080_SEEDS), (
        f"(b2) reads every one of the {len(T080_SEEDS)} seeds (read {got['b2']['n_seeds']})"
    )
    assert not got["b2"]["met"], (
        "an in-vivo null OUTSIDE the offline null's band must read (b2) REFUSED -- the two "
        "instruments disagree and nothing is reported"
    )
    assert got["b2"]["outside"], (
        "the refusal must NAME the seed that left the band, never only that one did"
    )
    inside = t080_contrasts(t080_constructed_seeds("band_inside", seed=6))
    assert inside["b2"]["met"] and inside["b2"]["inside"] == len(T080_SEEDS), (
        f"an in-vivo null inside the band on every seed must hold "
        f"(inside {inside['b2']['inside']} of {len(T080_SEEDS)})"
    )

    # THE MUTATION, RUN: the band widened to [0, 1].
    mut = t080_contrasts(outside, band=_t080_mutated_scorer("band_unbounded"))
    assert mut["b2"]["met"], (
        "a band of [0, 1] must accept the very reading the real band refused -- an agreement "
        "check that cannot refuse is not a check"
    )
    assert reds["R6"], "t080_reds()['R6'] must fire on the same tables the suite used"


def test_the_live_bound_rate_reaches_state_with_both_c5_sentences_beside_it():
    """2.6's PASS branch (2026-09-16): `Brain.state()["crossmodal_bound_rate"]` is the fraction of
    the last 200 awake ticks with ONE reported object for that tick's two latents, and its
    provenance carries the licensed and the refused sentence and says the daemon's own rate is a
    chance rate."""
    from snn_data_lake.gates.conclusion_protocol import CONCLUSION
    from snn_data_lake.infra import brain_telemetry
    from snn_data_lake.substrate.home import throwaway_home

    licensed, refused, _dis, reading, _date, symbols = CONCLUSION["temporal_binding"]
    assert symbols == ("crossmodal_bound_rate",) and isinstance(reading, float)
    prov = brain_telemetry.PROVENANCE["crossmodal_bound_rate"]
    assert " ".join(licensed.split()[:6]) in prov and "felt present" in prov and "CHANCE" in prov
    with throwaway_home():
        from snn_data_lake.engine.brain import make_brain

        b = make_brain(seed=0, persist=False)
        assert b.state()["crossmodal_bound_rate"] is None, "no awake tick yet -> no rate, never 0.0"
        awake = 0
        while awake < 12:
            awake += b.tick() == "awake"
        rate = b.state()["crossmodal_bound_rate"]
        assert rate is not None and 0.0 <= rate <= 1.0 and len(b._xmodal_bound_hist) == awake


def test_the_charged_reading_re_derives_from_the_pin_and_a_mutated_derivation_does_not():
    """The T080 panel fold (2026-09-16): the C5 row's reading is tied to the committed pin on
    every gate run -- re-derived as the mean over seeds of (within - the null's mean) -- and
    the clause is reachable: the same formula with the null's q95 reads a different number."""
    from snn_data_lake.instruments.binding_instrument import T080_PIN, t080_pin_reading

    assert T080_PIN.exists(), T080_PIN
    derived, recorded, mutated = t080_pin_reading()
    assert derived == recorded == 0.2309
    assert mutated != recorded
    with pytest.raises(AssertionError):
        t080_pin_reading(T080_PIN.with_name("no_such_pin.json"))


def test_a_faulted_binder_is_counted_and_never_a_bound_tick():
    """The T080 panel fold (2026-09-16): a bind() that raises clears the handle and counts a
    fault, so neither `crossmodal_handle` nor the bound rate carries a stale object across the
    faulted tick (the one path the panel measured that could move the live symbol UP)."""
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        from snn_data_lake.engine.brain import make_brain

        b = make_brain(seed=0, persist=False)
        # reach the first awake tick with both latents and a call into the binder
        calls = {"n": 0}
        real_bind = b._xmodal.bind

        def boom(v, a):
            calls["n"] += 1
            raise RuntimeError("constructed binder fault")

        b._xmodal.bind = boom
        # ARC 6 (2026-11-01): CAPPED. The uncapped `while` livelocked a battery for ~2 h when
        # the perception assets (gitignored models/*.pt) were missing: no real percept reached
        # the binder. With the assets the first bind comes in a few ticks; a missing percept
        # now FAILS by name.
        for _ in range(2000):
            if calls["n"]:
                break
            b.tick()
        else:
            tier = b.state().get("perception_tier")
            pytest.fail(
                "the binder was never called in 2000 ticks: no real visual+audio percept "
                f"reached it (perception assets missing? perception_tier={tier!r})"
            )
        assert b.state()["crossmodal_handle"] is None
        assert b.state()["crossmodal_faults"] == calls["n"] >= 1
        assert b._xmodal_bound_hist[-1] is False
        b._xmodal.bind = real_bind
