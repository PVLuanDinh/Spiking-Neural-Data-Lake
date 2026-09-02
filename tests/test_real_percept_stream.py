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
    tr, te = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 1, 0, 0])
    maj, pers = f(tr, te)
    assert maj == 0.5  # training majority is 0; two of four held-out rows are 0
    assert pers == 0.75  # predecessors [1,1,1,0] vs [1,1,0,0] -> 3 of 4 agree
    # no training rows: no majority to answer with (never a peek at the held-out slice), and
    # the first held-out row has no predecessor -> a miss
    assert f(torch.tensor([]), torch.tensor([2, 2])) == (0.0, 0.5)


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
