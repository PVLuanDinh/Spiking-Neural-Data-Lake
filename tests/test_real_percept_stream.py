"""specs/022 audit s8 rank 2a: the persisted grower is fed the BRAIN'S OWN percept stream.

Until 2026-09-02 `cortex/neurogenesis_engine` synthesized a 5-mode Markov chain for itself on
every sleep tick and predicted it at ~0.7 (its own self-check: 0.67-0.71) -- an experiment ABOUT
growth, never the brain growing on what it perceived (P0's honesty guard). Now `Brain._wake_tick`
hands the engine every wake percept (`res["percept"]`, the attention-gained context vector the arm's
GrowingLearner categorized) with the brain's own category of it (that learner's prototype id), and
the engine grows at night to predict the NEXT category.

Three pins, each with the control that makes it a measurement:
  * NOTHING GROWS FROM NOTHING -- an engine shown no percept refits nothing and reports 0.0,
    against the same engine fed the harness stream, which grows (the retired chain must not
    survive as a silent fallback: P2).
  * THE BRAIN FEEDS THE ENGINE -- every wake tick with a category reaches observe(), none is
    refused, and sleep ticks refit on that stream; against a brain whose feed is cut, whose
    engine then observes nothing (the wire is the difference, not the brain).
  * THE FLOORS RIDE WITH THE ACCURACY -- the engine reports the persistence and majority
    baselines of the SAME held-out slice, and the brain's state() carries them beside the
    accuracy (a 1.0 that is the stream's inertia must say so).

Record: specs/022-unified-brain-arc/evidence/s2-real-percept-stream.md.
"""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.cortex.neurogenesis_engine import (  # noqa: E402
    NeurogenesisSelfImprove,
    SyntheticPerceptStream,
)


def test_nothing_grows_from_nothing_but_the_harness_stream_grows():
    unfed = NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu")
    for _ in range(200):
        unfed.tick()
    assert unfed.trajectory == [] and unfed.grown_units == 0 and unfed.observed == 0
    assert unfed.predict_accuracy == 0.0 and unfed.n_modes == 0 and unfed.baselines == []
    fed = SyntheticPerceptStream(seed=7).drive(
        NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu"), 200
    )
    assert fed.observed == 200 and fed.ticks == 200 and fed.n_modes == 5
    assert fed.trajectory and fed.grown_units > 0, "the same engine fed grows"
    assert len(fed.baselines) == len(fed.trajectory), "one baseline row per refit"
    # the floors are honest numbers on the harness chain: persistence is NOT the structure
    # (a doubly-stochastic chain with peak 0.75 on a PERMUTED preferred mode repeats itself only
    # at the off-diagonal rate), so the accuracy has to clear it to mean anything
    _t, maj, pers = fed.baselines[-1]
    assert 0.0 <= maj <= 1.0 and 0.0 <= pers <= 1.0


def test_observe_refuses_a_bad_category_and_a_ragged_percept():
    eng = NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu")
    with pytest.raises(ValueError):
        eng.observe([0.1] * 8, -1)
    eng.observe([0.1] * 8, 0)
    eng.observe([0.2] * 8, 1)  # first pair buffered here (percept_0 -> category_1)
    with pytest.raises(ValueError):
        eng.observe([0.3] * 9, 1)
    assert eng.observed == 2 and len(eng._buf_x) == 1


def test_heldout_baselines_are_the_two_floors():
    f = NeurogenesisSelfImprove._heldout_baselines
    # the stream [0,0,0,1,1,1,0,0]; rows 4-7 held out (the old contiguous window): the training
    # majority is 0 and two of four held-out rows are 0; persistence by the TRUE predecessor:
    # rows 4,5,6,7 have predecessors 1,1,1,0 against values 1,1,0,0 -> 3 of 4 agree
    y = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0])
    maj, pers = f(y, [0, 1, 2, 3], [4, 5, 6, 7])
    assert maj == 0.5
    assert pers == 0.75
    # a SCATTERED held-out set (L1a's window): rows 2 and 5 -- predecessors 0 and 1 against values
    # 0 and 1 -> persistence 1.0; the training rows' majority is 0 and one of two held-out rows is 0
    maj2, pers2 = f(y, [0, 1, 3, 4, 6, 7], [2, 5])
    assert (maj2, pers2) == (0.5, 1.0)
    # no training rows: no majority to answer with, and a first row with no predecessor is a miss
    assert f(torch.tensor([2, 2]), [], [0, 1]) == (0.0, 0.5)


@pytest.mark.slow
def test_the_brain_feeds_the_engine_and_the_floors_reach_state():
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        from snn_data_lake.cortex.brain import make_brain

        b = make_brain(seed=0, persist=False)
        assert b.neuro is not None
        modes = [b.tick() for _ in range(260)]
        awake = modes.count("awake")
        eng = b.neuro
        assert eng.observed == awake, (
            f"every wake tick must reach observe(): {eng.observed} observed vs {awake} awake"
        )
        assert b._neuro_feed_refusals == 0
        assert eng.ticks == modes.count("asleep"), "sleep ticks drive the refit cadence"
        assert eng.n_modes >= 2, "the arm must have shown the engine more than one category"
        assert eng.trajectory, "at least one night refit on the brain's own stream"
        full = b.state()
        st = full["neurogenesis"]
        for k in (
            "observed",
            "baseline_majority",
            "baseline_persistence",
            "n_modes",
            "feed_refusals",
        ):
            assert k in st, k
        # one representation of a fact: accuracy and units stay TOP-LEVEL (the cockpit's rows),
        # the nested report carries the floors they must be read against
        assert "predict_accuracy" not in st and "grown_units" not in st
        assert abs(full["predict_accuracy"] - eng.predict_accuracy) < 1e-3  # state() rounds
        assert st["baseline_persistence"] is not None, "a refit ran, so the floors are measured"
        assert st["observed"] == awake and st["feed_refusals"] == 0
        # THE CONTROL: cut the wire and the same brain shows the engine nothing
        c = make_brain(seed=0, persist=False)
        c.neuro.observe = lambda _f, _c: (_ for _ in ()).throw(RuntimeError("cut"))
        cm = [c.tick() for _ in range(60)]
        assert c.neuro.observed == 0, "a cut feed shows the engine nothing"
        assert c._neuro_feed_refusals == cm.count("awake") > 0, (
            "a cut wire must be COUNTED on every wake tick, never silent"
        )


# ------------------------------------------------------------------------------------------------
# Phase 2's stream-independent arms (specs/022 evidence/l-parallel-arms.md, 2026-09-05): the
# persistence oracle and the constant percept, at the mechanism's input contract, RED on the tree
# before L1a / L3.
# ------------------------------------------------------------------------------------------------
import concurrent.futures as _cf  # noqa: E402
import itertools as _it  # noqa: E402
import math as _math  # noqa: E402
import statistics as _st  # noqa: E402

L_SEEDS = tuple(range(8))
L_TICKS = 800
T_BAR = 1.895  # paired one-sided t, 7 d.f., alpha 0.05 (the plan's statistic)
TIE_TOL = (
    1e-6  # CHOSEN: a float32 accuracy minus a float64 floor leaves ~1e-8 residues at the floor
)


def _l_cell(args):
    """One in-brain cell: the engine's feed neutered per arm (LIVE untouched; ORACLE_PREVCAT: the
    percept is the current category one-hot, so the buffered pair is one-hot(cat_{t-1}) -> cat_t;
    NEUTER_PERCEPT: a constant; SYNTH: the synthetic chain through the same feed). Returns the
    last-third accuracy, floors and the unit count. Runs in a worker process."""
    seed, arm = args[0], args[1]
    perm = args[2] if len(args) > 2 else None  # SC-012a: the seed's recorded sequence, permuted
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        from snn_data_lake.cortex.brain import make_brain
        from snn_data_lake.cortex.neurogenesis_engine import SyntheticPerceptStream

        b = make_brain(seed=seed, persist=False)
        eng = b.neuro
        real = eng.observe
        synth = SyntheticPerceptStream(seed=seed)
        # SC-012a: admission CAPPED exogenously at N. `eps_tol` must be driven to zero WITH it: at
        # the engine's default 0.28 the greedy pursuit reaches its train-RMSE target at ~2 units and
        # `l_max` never binds, so CAP16 admits 2 units and the grid measures nothing
        # (evidence/sc-012a-width.md section 2.5, where the dissociation is recorded). It is held at
        # ONE value across every cap arm, so it cannot confound acc(N) against acc(M).
        cap = int(arm[3:].split("_")[0]) if arm.startswith("CAP") else None
        if cap is not None:
            eng._l_max = cap
            eng._eps_tol = 0.0
        cat_seq, fed = [], [0]

        def observe(features, category):
            feats = [float(v) for v in features]
            cat = int(category)
            if cap is not None:
                cat_seq.append(cat)  # the PRE-shuffle category, so arms stay comparable
                k = fed[0]
                fed[0] += 1
                if perm is not None and k < len(perm):
                    cat = perm[k]  # the marginal-preserving shuffle
                return real(feats, cat)
            if arm in ("LIVE", "L1C", "L1C_CTRL"):
                return real(feats, cat)
            if arm == "NEUTER_PERCEPT":
                return real([0.5] * len(feats), cat)
            if arm == "ORACLE_PREVCAT":
                oh = [0.0] * 64
                oh[min(cat, 63)] = 1.0
                return real(oh, cat)
            return real(*synth.next())

        eng.observe = observe
        # the brain constructs the L1c survivor for every arm; the incumbent arms (LIVE, the L1b
        # arms) measure the batch-only readout, so it is removed here -- their pins read the refit
        # accuracy, which the decoupled instrument keeps identical either way
        eng.online_readout = None
        if cap is not None:
            # SC-012a's cap arms run the DEPLOYED wiring (make_brain's "delta"), so this gate and
            # the cells in evidence/sc-012a-width.md are the same machine. It cannot move it:
            # the statistic is the refit accuracy, which the decoupled batch instrument keeps
            # identical either way (the reason the incumbent arms above can drop it).
            from snn_data_lake.hippocampus.online_readout import OnlineReadout

            eng.online_readout = OnlineReadout("delta")
        elif arm in (
            "L1C",
            "L1C_CTRL",
        ):  # the online readout (L1c) and its control that is not the method
            from snn_data_lake.hippocampus.online_readout import OnlineReadout

            eng.online_readout = (
                OnlineReadout("delta") if arm == "L1C" else OnlineReadout("delta", sign=-1.0)
            )
        for _ in range(L_TICKS):
            b.tick()
        traj, bl = list(eng.trajectory), list(eng.baselines)
        third = max(1, len(traj) // 3)
        out = {
            "unified": bool(getattr(b, "_perception_unified", False)),
            "seed": seed,
            "arm": arm,
            "units": eng.grown_units,
            "units_max": max([r[1] for r in traj]) if traj else None,  # pre-prune, per refit
            "cat_seq": cat_seq,  # SC-012a's same-rows check; empty for the non-cap arms
            "refits": len(traj),
            "acc": _st.fmean([r[2] for r in traj[-third:]]) if traj else None,
            "pers": _st.fmean([r[2] for r in bl[-third:]]) if bl else None,
            "maj": _st.fmean([r[1] for r in bl[-third:]]) if bl else None,
            "live_acc": eng.state().get("live_accuracy"),
            "live_pers": eng.state().get("live_persistence"),
            # `decide`'s fallback, per REFIT -- L-R4's aliveness discriminator since 2026-09-07
            "abstentions": eng.state().get("decide_abstentions"),
            "unanswerable": eng.state().get("decide_unanswerable"),
            "heldout_n": eng.state().get("decide_heldout_n"),
        }
        b.close()
    return out


def _paired(diffs):
    """(clears, t, exact one-sided sign-flip p, non-tied) -- ties as zeros; DEGENERATE below 5 non-
    tied."""
    n = len(diffs)
    diffs = [0.0 if abs(d) <= TIE_TOL else d for d in diffs]  # a float32 residue is a tie
    non_tied = sum(1 for d in diffs if d != 0.0)
    m = _st.fmean(diffs)
    sd = _st.stdev(diffs) if n > 1 else 0.0
    t = (m / (sd / _math.sqrt(n))) if sd > 0 else (float("inf") if m > 0 else 0.0)
    obs = sum(diffs)
    ge = sum(
        1
        for signs in _it.product((1, -1), repeat=n)
        if sum(s * abs(d) for s, d in zip(signs, diffs, strict=True)) >= obs - 1e-12
    )
    p = ge / 2**n
    return (t > T_BAR and p <= 0.05 and non_tied >= 5), t, p, non_tied


def _run_arms(arms, perms=None):
    jobs = [(s, a, (perms or {}).get(s)) for a in arms for s in L_SEEDS]
    with _cf.ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(_l_cell, jobs))
    return {(r["seed"], r["arm"]): r for r in rows}


def _require_the_live_packs(res):
    """The two reds below are properties of the LIVE percept stream (persistence floor 0.91-1.00).
    On a clone without the gitignored perception packs the extero bands are a synthetic rotation
    (floor 0.0) and the oracle learns its transitions outright -- the strict red would XPASS."""
    if not all(r["unified"] for r in res.values()):
        pytest.skip(
            "the real perception packs are not on this machine; these reds read the live stream"
        )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "L-R1's STANDING RED (evidence/l-parallel-arms.md section 3.2): under L1a's window the "
        "oracle sits AT its persistence floor (mean -0.021, at the floor to 1e-8 on 2 of 8, within "
        "0.022 on 5 of 8; the incumbent window read -0.428) and does not EXCEED it -- the live "
        "stream's transitions are persistence-only; strict so a change that clears the bar must "
        "update the record"
    ),
)
def test_the_persistence_oracle_clears_its_own_floor_under_the_admission_window():
    """L-R1 (plan Part F): the oracle -- handed exactly the information persistence uses -- must
    beat its own persistence floor, paired over 8 seeds at the plan's bar. RED before L1a (s2 v2:
    -0.345, t -3.53): the recency window let no unit earn its place for a rarer category."""
    res = _run_arms(["ORACLE_PREVCAT"])
    _require_the_live_packs(res)
    d = [res[(s, "ORACLE_PREVCAT")]["acc"] - res[(s, "ORACLE_PREVCAT")]["pers"] for s in L_SEEDS]
    clears, t, p, non_tied = _paired(d)
    reach = sum(1 for x in d if x >= -TIE_TOL)
    if non_tied < 5:  # a degenerate run must never masquerade as the standing red
        pytest.fail(f"DEGENERATE-TIES: {non_tied} non-tied seeds ({d})", pytrace=False)
    mean_d = sum(d) / len(d)
    if (
        mean_d < -0.1
    ):  # a REGRESSION of the window (the oracle back near -0.43) is not the standing red
        pytest.fail(f"the oracle fell far below its floor again (mean {mean_d:.3f})", pytrace=False)
    assert clears, (
        f"the persistence oracle must clear its own floor at the bar: t={t:.3f} p={p:.4f} "
        f"reaches the floor on {reach}/8 (diffs {[round(x, 3) for x in d]})"
    )


@pytest.mark.slow
def test_a_constant_percept_admits_one_unit_and_a_structured_stream_admits_more():
    """L-R4 (plan Part F): a constant input carries nothing to discriminate rows, so the grower
    admits exactly ONE unit (the class prior) on every seed, while the structured positive control
    (the synthetic chain) admits more than one on every seed in the same run. RED before L3 (s2
    v2: 24.1 units on the constant percept)."""
    res = _run_arms(["NEUTER_PERCEPT", "SYNTH"])
    _require_the_live_packs(res)
    const = [res[(s, "NEUTER_PERCEPT")]["units"] for s in L_SEEDS]
    synth = [res[(s, "SYNTH")]["units"] for s in L_SEEDS]
    gap = [
        abs(res[(s, "NEUTER_PERCEPT")]["acc"] - res[(s, "NEUTER_PERCEPT")]["maj"]) for s in L_SEEDS
    ]
    assert all(u > 1 for u in synth), (
        f"the positive control must admit > 1 unit on every seed ({synth})"
    )
    assert all(u == 1 for u in const), (
        f"a constant percept must admit exactly 1 unit on every seed ({const})"
    )
    # THE ONE UNIT MUST BE ALIVE, and this pin had to be REBUILT on 2026-09-07.
    #
    # It used to read |acc - majority| <= 1e-3: before `NGSNN.decide`, a DEAD unit was silent, its
    # all-zero readout answered class 0, and it scored ~0.0 -- nowhere near the majority floor --
    # so the gap discriminated. `decide` now answers the training PRIOR on a silent row, so a dead
    # unit lands AT the majority floor too: the old assertion passes for alive and dead alike. The
    # arc that fixed the scoring path BLINDED this red, and its own panel caught it.
    #
    # The repair does not route through accuracy at all. `decide_abstentions` counts the held-out
    # rows this refit answered from the prior, and `decide_unanswerable` counts those it could not
    # answer even from the prior. A unit that FIRES answers some rows itself; a dead one abstains
    # on every row. That is the aliveness question asked directly instead of inferred.
    abst = [res[(s, "NEUTER_PERCEPT")]["abstentions"] for s in L_SEEDS]
    n_te = [res[(s, "NEUTER_PERCEPT")]["heldout_n"] for s in L_SEEDS]
    dead = [s for s, a, n in zip(L_SEEDS, abst, n_te, strict=True) if n and a >= n]
    assert not dead, (
        f"the constant arm's one unit is DEAD at seeds {dead}: it abstained on every held-out row "
        f"(abstentions {abst} of {n_te}), so its accuracy is the prior's, not the unit's"
    )
    unans = [res[(s, "NEUTER_PERCEPT")]["unanswerable"] for s in L_SEEDS]
    assert all(u == 0 for u in unans), (
        f"a silent row was answered by tensor layout, not by the prior ({unans}) -- the retired "
        "scoring defect is firing again"
    )
    # the floor relation is KEPT as a reported companion, no longer as the aliveness test
    assert all(g <= 0.5 for g in gap), (
        f"the constant arm's one unit should sit near its majority floor (|acc - majority| {gap})"
    )


@pytest.mark.slow
def test_the_online_readout_beats_the_stale_one_on_the_wake_stream():
    """L1c-R1 (evidence/l1c-online-readout.md section 2.0c): paired (live_acc_L1C - live_acc_LIVE)
    over 8 seeds clears the plan's bar on the SAME rows, and the survivor's control that is not
    the method does not (R3); the refit-time accuracy does not fall past the bar (R4)."""
    res = _run_arms(["LIVE", "L1C", "L1C_CTRL"])
    _require_the_live_packs(res)
    for s in L_SEEDS:  # same rows
        assert abs(res[(s, "L1C")]["live_pers"] - res[(s, "LIVE")]["live_pers"]) <= TIE_TOL
    d1 = [res[(s, "L1C")]["live_acc"] - res[(s, "LIVE")]["live_acc"] for s in L_SEEDS]
    d3 = [res[(s, "L1C_CTRL")]["live_acc"] - res[(s, "LIVE")]["live_acc"] for s in L_SEEDS]
    d4 = [res[(s, "L1C")]["acc"] - res[(s, "LIVE")]["acc"] for s in L_SEEDS]
    c1, t1, p1, n1 = _paired(d1)
    c3, t3, p3, n3 = _paired(d3)
    if n1 < 5:
        pytest.fail(f"DEGENERATE-TIES: {n1} non-tied seeds", pytrace=False)
    # R4 under the decoupled instrument is an IDENTITY: the gate must not see the online beta
    if max(abs(x) for x in d4) > TIE_TOL:
        pytest.fail(
            f"R4: the refit accuracy differs from the incumbent's (the decoupling leaked): {d4}",
            pytrace=False,
        )
    if c3:
        pytest.fail(
            f"R3: the control that is not the method CLEARED (t {t3:.3f} p {p3:.4f})", pytrace=False
        )
    assert c1, (
        f"L1c-R1: (L1C - LIVE) live accuracy t={t1:.3f} p={p1:.4f} "
        f"diffs {[round(x, 3) for x in d1]}"
    )


@pytest.mark.slow
def test_width_buys_next_category_accuracy_and_the_shuffled_grid_does_not():
    """SC-012a (spec.md:183; plan Phase 2, ratified under OD-5): with admission capped exogenously
    at N, paired (acc(16) - acc(1)) clears the bar AND acc(16) clears its own majority floor, AND
    the same grid with the labels shuffled marginal-preserving does not rise.

    FIRED 2026-09-07, on the SECOND reading (evidence/sc-012a-width.md section 7). The first
    reading was RETRACTED: acc(1) was an artefact of the scoring path -- an all-zero rate row's
    argmax answered column 0, a class the stream never emits as a target -- and conjunct (a) and
    its null (c) both subtracted it. `NGSNN.decide` fixed that (evidence/scoring-path-abstention.md)
    and the grid was re-run: (a) t=2.596 p=0.0234 mean +0.151 on 6 of 8; (b) t=4.011 p=0.0039 mean
    +0.194 on 8 of 8; (c) the shuffled grid FELL, t=-2.346 mean -0.036. The fix CUT the effect from
    +0.465 to +0.151 and it still clears, while the null went from marginally rising to cleanly
    falling -- which is why the surviving effect is read as real rather than as the artefact's
    residue. Qualified, and the record says so: at N=16 the pursuit saturates below the cap on 6 of
    8 seeds, so "16" realizes 9-14 units.

    THIS TEST NO LONGER CARRIES AN XFAIL. It carried one while the criterion was unreadable, with
    the standing note that it would XPASS the day the scoring path was fixed. That day came, the
    grid was re-read, and the pin is now the criterion itself.

    WHAT RAISES WHAT, kept from the xfail era because it is still the right structure: the three
    conjuncts fail through `assert` (AssertionError) -- the criterion's own verdict -- while every
    INSTRUMENT check fails through `pytest.fail` -> `Failed`, which is NOT a subclass of
    AssertionError. The separation exists so an instrument that broke can never be filed as a
    criterion result, in either direction."""
    import random

    real = _run_arms(["CAP1", "CAP16"])
    _require_the_live_packs(real)
    perms = {}
    for s in L_SEEDS:
        seq = list(real[(s, "CAP1")]["cat_seq"])
        if seq != real[(s, "CAP16")]["cat_seq"]:
            pytest.fail(f"same rows: the two caps saw different streams at seed {s}", pytrace=False)
        random.Random(s + 500).shuffle(seq)
        perms[s] = seq
    shuf = _run_arms(["CAP1_SHUF", "CAP16_SHUF"], perms)

    # --- instrument controls: these must not be absorbable as the null -------------------------
    bad = [s for s in L_SEEDS if real[(s, "CAP1")]["units"] != 1]
    if bad:
        pytest.fail(f"the cap's reach control: CAP1 is not 1 unit at seeds {bad}", pytrace=False)
    # THE WIDTH-VARIED CONTROL. The not-fired branch says width buys nothing; that is a causal
    # claim, and it is only earned if the cap actually varied the width. Measured 2026-09-07:
    # CAP1 = 1 unit on 8 of 8, CAP16 = 9-14 (the pursuit saturates below 16 and the prune cuts
    # further, so the claim is bounded by the width REALIZED, never by the cap's label).
    thin = [s for s in L_SEEDS if real[(s, "CAP16")]["units"] <= real[(s, "CAP1")]["units"]]
    if thin:
        pytest.fail(
            f"the cap did not force width at seeds {thin}: CAP16 admitted no more than CAP1, so "
            "neither the criterion nor its null disposition measures width",
            pytrace=False,
        )

    d_a = [real[(s, "CAP16")]["acc"] - real[(s, "CAP1")]["acc"] for s in L_SEEDS]
    d_b = [real[(s, "CAP16")]["acc"] - real[(s, "CAP16")]["maj"] for s in L_SEEDS]
    d_c = [shuf[(s, "CAP16_SHUF")]["acc"] - shuf[(s, "CAP1_SHUF")]["acc"] for s in L_SEEDS]
    ca, ta, pa, na = _paired(d_a)
    cb, tb, pb, nb = _paired(d_b)
    cc, tc, pc, nc = _paired(d_c)
    # (a) and (b) are PASS conjuncts, so few non-tied seeds means no measurement. (c) is a VETO, so
    # ties are evidence FOR the criterion and a motionless null is its STRONGEST reading -- guarding
    # it here would file the cleanest possible fire as an instrument break (the panel's finding).
    if min(na, nb) < 5:
        pytest.fail(f"DEGENERATE-TIES on a PASS conjunct: {na}/{nb} non-tied seeds", pytrace=False)

    # --- THE BLOCKING DEFECT, checked before the conjuncts: is acc(1) a readout at all? --------
    # The criterion subtracts acc(1) in (a) and again in its null (c). If the CAP1 readout never
    # fires on the held-out rows, a bare argmax answers column 0 -- a class the stream never emits
    # as a target -- and acc(1) is a constant wrong answer rather than a baseline. That is how the
    # FIRST reading of this criterion was produced, and why it was retracted. `NGSNN.decide` now
    # answers the training prior on a silent row, so CAP1 lands AT its majority floor instead of at
    # ~0; this check is the regression pin, and it fails through `Failed` because a scoring path
    # that broke again is an INSTRUMENT failure, never a criterion result.
    degenerate = [s for s in L_SEEDS if real[(s, "CAP1")]["acc"] <= 0.02]
    if degenerate:
        pytest.fail(
            f"acc(1) is an ARTEFACT, not a baseline: CAP1 scores <=0.02 at seeds {degenerate} "
            f"(all-zero score vectors resolving to class 0, which is never a target -- "
            f"NGSNN.decide's prior fallback has regressed). SC-012a is unreadable while that "
            f"holds. Computed but NOT to be read as a verdict: (a) t={ta:.3f} p={pa:.4f}; "
            f"(b) t={tb:.3f} p={pb:.4f}; (c) t={tc:.3f} p={pc:.4f}",
            pytrace=False,
        )

    # --- the criterion: (a) AND (b) AND NOT (c) -- reached only once the baseline is real -------
    assert ca and cb and not cc, (
        f"(a) acc(16)-acc(1) t={ta:.3f} p={pa:.4f} diffs {[round(x, 3) for x in d_a]}; "
        f"(b) acc(16)-majority t={tb:.3f} p={pb:.4f}; "
        f"(c) the SHUFFLED grid t={tc:.3f} p={pc:.4f} -- it must NOT clear, and if it does the "
        f"rise is the instrument's whatever (a) and (b) read"
    )
