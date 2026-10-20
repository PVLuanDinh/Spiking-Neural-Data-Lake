"""specs/022 T039f: the insula reaches the ignition threshold it was built to modulate.

The insula was the ONE region CHARGED but UNREACHABLE. The exemption row promised a
coalition-competition consumer; T039b measured that sink OUT OF RANGE (feeling_salience peaks
0.2798 against an admission floor of 0.5505) and refused it. The row then named the sink that
does work -- threshold modulation, which is the producer's own stated claim -- and bound its own
retirement to "the measurement that shows a consumer's own output changing, and by nothing else".

THIS FILE IS THAT MEASUREMENT, and it is deliberately built so the cheap half cannot pass for the
expensive one. A single import in engine/brain.py flips insula to CHARGED with or without a
consumer; what follows tests the consumer.

The P8 burden, one test each:
  * a REACHABLE RED          -- the law must be able to fail (monotonicity, the bound, additivity)
  * a CONTROL that is not the method -- a CONSTANT-salience arm, which the s0-exit's own
                                        adversarial measurement used to kill the previous route
  * PER-ITEM ATTRIBUTION     -- neuter the insula and show the brain's own output moves
  * DELETION ON A NULL       -- a salience that never varies, or a threshold that never moves,
                                fails here rather than being reported as a wire

Run: pytest tests/test_interoceptive_ignition.py
"""

import random
import statistics

import pytest

from snn_data_lake.faculties.interoceptive_integration import BODY_CHANNELS
from snn_data_lake.faculties.region_coherence import RegionCoherence

TICKS = 40
# The OUTCOME A/B runs longer (2026-10-19): on a HERMETIC home the insula moves the
# threshold from tick 4 but first changes WHAT WINS at tick 84 (winner differs on 28 of 200
# ticks; deterministic, seed 0). At 40 ticks that assert had only passed on a NORD_HOME leaked
# from an earlier test in serial order -- with a module home under pytest -n it read thr_diff 35,
# outcome diff 0. TICKS stays 40 for the tests calibrated on it (distinct-value bar TICKS // 2).
OUTCOME_TICKS = 200
GAIN = RegionCoherence.IGNITION_GAIN
CTX = dict(visual=0.55, temperature=24.0, audio=0.3, distance=0.6)


# ==================================================================================================
# The law itself. Pure, fast, and every one of these is a way the merge could be wrong.
# ==================================================================================================
def _old(base, c):
    """The expression T039f replaced, kept verbatim as the comparison target."""
    return base * (1.0 - GAIN * min(1.0, max(0.0, c)))


def test_the_term_is_additive_from_zero():
    """With no interoceptive deviation the law must be BYTE-identical to the one it replaced --
    this is what makes the term's attribution a measurement instead of an argument, and what
    guarantees an absent insula changes no threshold anywhere."""
    for base in (0.2, 0.55, 1.0, 1.19):
        for i in range(0, 21):
            c = i / 20.0
            assert RegionCoherence._adjust(base, c, GAIN, 0.0) == _old(base, c), (base, c)


def test_the_bound_survives_the_second_input():
    """thr must stay in [base*(1-gain), base] and strictly positive for gain < 1 -- the SAME
    precondition the one-input law had. If the merge widened the range, a downstream
    coherence_theta would start range-refusing and the failure would surface as a dead brain."""
    for base in (0.2, 0.55, 1.19):
        for i in range(0, 11):
            for j in range(0, 11):
                thr = RegionCoherence._adjust(base, i / 10.0, GAIN, j / 10.0)
                assert 0.0 < thr <= base, (base, i, j, thr)
                assert thr >= base * (1.0 - GAIN) - 1e-12, (base, i, j, thr)


def test_the_law_is_monotone_in_BOTH_reasons():
    """Either reason to ignite must lower the threshold, holding the other fixed. A merge that was
    monotone in coherence only would look green on every pre-existing test.

    THE SATURATION IS REAL AND IS STATED RATHER THAN ASSERTED AWAY. This test first demanded
    STRICT monotonicity at every held value and FAILED at held == 1.0, correctly: noisy-OR gives
    c_eff = 1 - (1-c)(1-s), so once either reason is maximal the threshold already sits on its
    floor base*(1-gain) and the other reason has nowhere left to move it. That is the BOUND
    working, not a defect -- but it does mean the honest claim is "strictly decreasing below
    saturation, never increasing anywhere", so that is what is checked. The live range does not
    reach saturation: IGNITION_GAIN's own measured distribution records c in [0.138, 0.896] with
    "c=0 and c=1 NEVER occur", and feeling_salience is bounded by z/(1+z) < 1 by construction."""
    base = 0.55
    for held in (0.0, 0.3, 0.7, 1.0):
        rising_s = [RegionCoherence._adjust(base, held, GAIN, k / 10.0) for k in range(11)]
        rising_c = [RegionCoherence._adjust(base, k / 10.0, GAIN, held) for k in range(11)]
        assert all(a >= b for a, b in zip(rising_s, rising_s[1:], strict=False)), (held, rising_s)
        assert all(a >= b for a, b in zip(rising_c, rising_c[1:], strict=False)), (held, rising_c)
        if held < 1.0:  # below saturation the response must be STRICT, not merely non-increasing
            assert all(a > b for a, b in zip(rising_s, rising_s[1:], strict=False)), (
                held,
                rising_s,
            )
            assert all(a > b for a, b in zip(rising_c, rising_c[1:], strict=False)), (
                held,
                rising_c,
            )
    # and the saturation itself, pinned so a later edit cannot quietly turn it into a rise
    assert (
        RegionCoherence._adjust(base, 1.0, GAIN, 0.0)
        == RegionCoherence._adjust(base, 1.0, GAIN, 1.0)
        == pytest.approx(base * (1.0 - GAIN))
    )


def test_neither_reason_is_discarded_when_both_are_present():
    """The control on the choice of combiner. max() would throw the weaker reason away entirely;
    noisy-OR must beat BOTH single-reason thresholds whenever both inputs are non-zero."""
    base, c, s = 0.55, 0.4, 0.3
    both = RegionCoherence._adjust(base, c, GAIN, s)
    assert both < RegionCoherence._adjust(base, c, GAIN, 0.0)
    assert both < RegionCoherence._adjust(base, 0.0, GAIN, s)


def test_ignition_bias_default_is_the_pre_T039f_call():
    """Every caller that predates T039f passes one argument and must be unchanged."""
    rc = RegionCoherence(seed=0)
    assert rc.ignition_bias(0.55) == rc._adjust(0.55, rc.coherence, rc.gain, 0.0)


# ==================================================================================================
# The live brain. One construction per arm, module-scoped, because ticking is the expensive part.
# ==================================================================================================
def _run(neuter, ticks=TICKS, salience=None):
    """Tick a real brain and record what the ignition path did. `neuter` removes the insula --
    the A/B whose difference IS the attribution. `salience` is a per-tick sequence that REPLACES
    the value the insula hands the brain (the organ still observes, so its baseline advances as
    in the live arm): the control arms, which feed a constant or shuffled signal through the
    brain's own consumer."""
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    if neuter:
        b._insula = None
    observed = {"n": 0, "exhausted": False}
    if b._insula is not None:
        replay = iter(salience) if salience is not None else None
        real_observe = b._insula.observe

        def _observe(_real=real_observe, **frame):
            observed["n"] += 1
            value = _real(**frame)
            if replay is None:
                return value
            # the brain swallows exceptions from this call into salience 0.0, so exhaustion is
            # flagged rather than raised
            replacement = next(replay, None)
            observed["exhausted"] |= replacement is None
            return 0.0 if replacement is None else replacement

        b._insula.observe = _observe
    arm = b.brain.arm
    draws = {"n": 0}
    if arm.intero is not None:  # spy on the ORGAN so a second draw per tick would be visible
        real = arm.intero.sense

        def _spy(load, _r=real):
            draws["n"] += 1
            return _r(load)

        arm.intero.sense = _spy
    rows = []
    for _ in range(ticks):
        before = observed["n"]
        b.tick()
        base = b._reentrant.base_ignition_threshold
        c = float(getattr(b, "_region_coherence_val", 0.0) or 0.0)
        bound = bool(getattr(b, "_bound", False))
        s = float(getattr(b, "_feeling_salience", 0.0) or 0.0)
        c_gated = c if bound else 0.0
        rows.append(
            {
                "salience": s,
                "observed": observed["n"] > before,
                "replay_exhausted": observed["exhausted"],
                "bound": bound,
                "coh_gated": c_gated,
                "base": base,
                "live_thr": float(b._reentrant.ignition_threshold),
                "with_s": RegionCoherence._adjust(base, c_gated, GAIN, s),
                "without_s": RegionCoherence._adjust(base, c_gated, GAIN, 0.0),
                "ignited": bool(b._access_state.get("ignited", False)),
                "winner": b._reentrant._last.get("winner"),
            }
        )
        # THE ACCESSOR IS EXERCISED HARD HERE ON PURPOSE: three extra reads per tick. If it took
        # a reading instead of returning the cached one, the draw count below would triple.
        arm.interoception()
        arm.interoception()
        cached = arm.interoception()
        if cached:
            cached["stress"] = -999.0  # a caller mutating its copy must not reach the cache
    b.close()
    return rows, draws["n"]


@pytest.fixture(scope="module")
def live():
    return _run(neuter=False)


@pytest.fixture(scope="module")
def neutered():
    return _run(neuter=True)


@pytest.fixture(scope="module")
def live_long():
    return _run(neuter=False, ticks=OUTCOME_TICKS)


@pytest.fixture(scope="module")
def neutered_long():
    return _run(neuter=True, ticks=OUTCOME_TICKS)


def _observed_trace(rows):
    """Salience on the ticks the insula was actually consulted. On the other ticks the brain
    uses 0.0 without calling the insula, and a replay must not spend its values there."""
    return [r["salience"] for r in rows if r["observed"]]


@pytest.fixture(scope="module")
def constant_long(live_long):
    """The mean of the live salience over the consulted ticks, held constant: same level, no
    timing. (The CONSTANT-control test averages every tick, unconsulted zeros included, so its mean
    differs.)"""
    mean_s = statistics.fmean(_observed_trace(live_long[0]))
    return _run(neuter=False, ticks=OUTCOME_TICKS, salience=[mean_s] * OUTCOME_TICKS)


@pytest.fixture(scope="module")
def shuffled_long(live_long):
    """The live trace, permuted (random.Random(0)): same distribution, no timing."""
    trace = _observed_trace(live_long[0])
    random.Random(0).shuffle(trace)
    return _run(neuter=False, ticks=OUTCOME_TICKS, salience=trace)


def test_the_accessor_reads_the_cache_and_never_takes_a_second_draw(live):
    """The organ's own rule: ONE sense() per tick, because sense() advances its noise stream and a
    second draw per tick forks replay (G5). The loop above calls interoception() three extra times
    per tick, so a drawing accessor would show 4x the draws."""
    rows, draws = live
    assert draws <= TICKS, (
        f"{draws} organ draws over {TICKS} ticks -- interoception() is taking a reading instead "
        "of returning the one _interocept already took; that forks replay"
    )


def test_the_accessor_returns_a_copy_not_the_cache(live):
    """The loop poisoned every returned dict with stress=-999. If the accessor handed out the
    cache itself, the same tick's coalition channels -- which read it too -- would be corrupted,
    and the salience trace below would carry the poison."""
    rows, _ = live
    assert all(r["salience"] == r["salience"] for r in rows)  # no NaN
    assert all(0.0 <= r["salience"] <= 1.0 for r in rows), [
        r["salience"] for r in rows if not 0.0 <= r["salience"] <= 1.0
    ]


def test_the_live_signal_is_not_constant(live):
    """DELETION ON A NULL. A signal that never moves cannot modulate anything, and reporting it as
    a wire would be exactly the dormancy P3 forbids. This is the test that would have killed the
    build; it is here because it nearly does not pass on a shorter run."""
    rows, _ = live
    sal = [r["salience"] for r in rows]
    assert len(set(round(x, 6) for x in sal)) > TICKS // 2, (
        f"feeling_salience took only {len(set(round(x, 6) for x in sal))} distinct values over "
        f"{TICKS} ticks -- a near-constant signal modulates nothing"
    )
    assert max(sal) > 0.05, f"feeling_salience never left the floor (max {max(sal):.4f})"


def test_the_interoceptive_term_moves_the_live_threshold(live):
    """The per-tick counterfactual: the same tick's threshold with and without the term."""
    rows, _ = live
    moved = [r for r in rows if abs(r["with_s"] - r["without_s"]) > 1e-9]
    assert len(moved) > TICKS // 2, f"threshold moved on only {len(moved)} of {TICKS} ticks"
    assert all(r["with_s"] <= r["without_s"] for r in rows), "the term must never RAISE the bar"


def test_it_reaches_the_ticks_the_coherence_term_structurally_cannot(live):
    """The reason this sink is worth taking at all. The synfire binding gate leaves coherence
    contributing exactly 0 on unbound ticks -- measured 34 of 120 bound on the live path -- so on
    the majority of ticks the pre-T039f law could not move the threshold AT ALL. The body can."""
    rows, _ = live
    unbound = [r for r in rows if not r["bound"]]
    assert unbound, "no unbound ticks in this run -- the claim is untested, not proven"
    # THE PRECONDITION, and the first draft of this line compared a value to ITSELF, which is
    # true for any input and proved nothing. What must actually hold is that on an unbound tick
    # the pre-T039f law leaves the threshold AT BASE -- that is what "coherence contributes
    # nothing here" means, and it is the whole reason these ticks are the interesting ones.
    assert all(abs(r["without_s"] - r["base"]) < 1e-12 for r in unbound), [
        (r["without_s"], r["base"]) for r in unbound if abs(r["without_s"] - r["base"]) >= 1e-12
    ]
    reached = [r for r in unbound if abs(r["with_s"] - r["without_s"]) > 1e-9]
    assert len(reached) > len(unbound) // 2, (
        f"the interoceptive term reached only {len(reached)} of {len(unbound)} UNBOUND ticks -- "
        "on those ticks the coherence term contributes nothing, so this is the ground the merge "
        "exists to cover"
    )


def test_a_CONSTANT_salience_control_does_not_reproduce_the_live_threshold_trace(live):
    """THE CONTROL, AND IT IS NOT THE METHOD ITSELF. Pinning the signal to its own live mean is
    the arm that killed the PREVIOUS route into this region (it left the brain byte-identical on
    120/120 ticks). Here it must NOT reproduce the live trace: if a constant offset gave the same
    thresholds, the build would be a calibration change wearing a wire's clothes, and what is
    doing the work would be the mean rather than the body."""
    rows, _ = live
    mean_s = statistics.fmean(r["salience"] for r in rows)
    live_thr = [
        RegionCoherence._adjust(r["base"], r["coh_gated"], GAIN, r["salience"]) for r in rows
    ]
    const_thr = [RegionCoherence._adjust(r["base"], r["coh_gated"], GAIN, mean_s) for r in rows]
    differing = sum(1 for a, b in zip(live_thr, const_thr, strict=True) if abs(a - b) > 1e-9)
    assert differing > TICKS // 2, (
        f"a constant-mean salience reproduced the live threshold on {TICKS - differing} of "
        f"{TICKS} ticks -- the VARIATION is not what is driving this, so the term is an offset"
    )


def test_neutering_the_insula_changes_the_brains_own_output(live_long, neutered_long):
    """PER-ITEM ATTRIBUTION, and the measurement the exemption row bound its retirement to. Two
    real runs at the same seed differing only in whether the insula exists. If the brain's own
    ignition record is identical, the wire changed nothing and P8 requires deleting it rather than
    reporting it."""
    live_rows, _ = live_long
    neut_rows, _ = neutered_long
    assert len(live_rows) == len(neut_rows) == OUTCOME_TICKS

    neut_sal = [r["salience"] for r in neut_rows]
    assert set(round(x, 9) for x in neut_sal) == {0.0}, (
        "the neutered arm still produced salience -- the A/B is not isolating the insula"
    )

    thr_diff = _threshold_distance(live_rows, neut_rows)
    assert thr_diff > 0, (
        "the live ignition threshold is identical with and without the insula -- reachability "
        "without charge, which P3 says is not charge"
    )

    ign_diff = sum(
        1 for a, b in zip(live_rows, neut_rows, strict=True) if a["ignited"] != b["ignited"]
    )
    win_diff = sum(
        1 for a, b in zip(live_rows, neut_rows, strict=True) if a["winner"] != b["winner"]
    )
    assert (ign_diff + win_diff) > 0, (
        f"the threshold moved on {thr_diff} ticks but WHAT IGNITED never changed "
        f"(ignited-diff {ign_diff}, winner-diff {win_diff}) -- a moved number that reaches no "
        "outcome is the worked example this tree keeps re-finding, and it fails here"
    )


def _threshold_distance(a_rows, b_rows):
    """Ticks on which the threshold the workspace actually held differs."""
    return sum(
        1 for a, b in zip(a_rows, b_rows, strict=True) if abs(a["live_thr"] - b["live_thr"]) > 1e-9
    )


def _outcome_distance(a_rows, b_rows):
    """Ticks on which the brain's own ignition record (ignited, winner) differs."""
    return sum(
        1
        for a, b in zip(a_rows, b_rows, strict=True)
        if (a["ignited"], a["winner"]) != (b["ignited"], b["winner"])
    )


def _assert_control_arms_real(live_long, constant_long, shuffled_long):
    """The control arms must be REAL: the injected signal reaches the brain's own threshold write.
    Asserts on the recorded live_thr (the value the workspace actually held), not a
    recomputation of the law."""
    live_sal = _observed_trace(live_long[0])
    const_rows, shuf_rows = constant_long[0], shuffled_long[0]
    assert len(const_rows) == len(shuf_rows) == OUTCOME_TICKS
    live_schedule = [r["observed"] for r in live_long[0]]
    for rows in (const_rows, shuf_rows):
        assert [r["observed"] for r in rows] == live_schedule
        assert not any(r["replay_exhausted"] for r in rows), "the replay ran out of values"
    assert _observed_trace(const_rows) == pytest.approx(
        [statistics.fmean(live_sal)] * len(live_sal)
    )
    assert sorted(_observed_trace(shuf_rows)) == sorted(live_sal)
    assert _observed_trace(shuf_rows) != live_sal, "the shuffle was the identity"
    for rows in (const_rows, shuf_rows):
        assert _threshold_distance(live_long[0], rows) > 0, (
            "the control arm held the live threshold trace -- the injection is inert"
        )
        recomputed = [r["with_s"] for r in rows]
        # coherence is brain state and may differ from the live arm; what must hold is that the
        # live threshold the workspace held IS the law applied to this arm's own signal
        assert max(abs(r["live_thr"] - w) for r, w in zip(rows, recomputed, strict=True)) < 1e-6, (
            "the workspace threshold is not the law applied to the injected salience -- the "
            "control signal did not reach the consumer"
        )


def test_the_control_arms_feed_their_signal_through_the_live_consumer(
    live_long, constant_long, shuffled_long
):
    _assert_control_arms_real(live_long, constant_long, shuffled_long)


def test_an_exhausted_replay_is_flagged_not_swallowed():
    """REACHABLE RED for the exhaustion guard: an empty replay must raise the flag on every
    consulted tick."""
    rows, _ = _run(neuter=False, ticks=10, salience=[])
    consulted = [r for r in rows if r["observed"]]
    assert consulted, "no tick consulted the insula in 10 ticks -- the red is unreachable"
    assert all(r["replay_exhausted"] for r in consulted)
    assert all(r["salience"] == 0.0 for r in consulted)


def test_PREREGISTERED_the_live_signal_moves_the_outcome_more_than_its_own_controls(
    live_long, neutered_long, constant_long, shuffled_long
):
    """PRE-REGISTERED 2026-10-19, written BEFORE the control arms were first run (seed 0,
    OUTCOME_TICKS=200, hermetic home). D(x, y) = ticks on which (ignited, winner) differs.

      N = D(live, neutered)   -- the insula removed
      C = D(live, constant)   -- live mean held constant (level kept, timing destroyed)
      S = D(live, shuffled)   -- live trace permuted     (distribution kept, timing destroyed)

    BAR: N > max(C, S). Removing the body signal must move the outcome more than keeping its
    level or its distribution while destroying only its timing. If the data refute this, the
    null is reported as measured and the bar is NOT retuned.

    The control arms are verified real by `_assert_control_arms_real` before the bar is
    computed, so a refutation cannot be an artefact of an injection that never reached the
    consumer."""
    _assert_control_arms_real(live_long, constant_long, shuffled_long)
    n = _outcome_distance(live_long[0], neutered_long[0])
    c = _outcome_distance(live_long[0], constant_long[0])
    s = _outcome_distance(live_long[0], shuffled_long[0])
    assert n > max(c, s), f"bar refuted: N={n} C={c} S={s}"


def test_the_declared_channel_layout_is_what_the_brain_feeds():
    """The frame engine/brain.py builds must be the concat layout the manifest declares -- if the
    module's BODY_CHANNELS changed and the feed did not, observe() would silently skip channels
    (it ignores unknown keys) and the salience would quietly measure less of the body."""
    assert BODY_CHANNELS == (
        "energy_deficit",
        "curiosity_deficit",
        "social_deficit",
        "stress",
        "caution",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
