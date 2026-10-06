"""G6 / T114 part (e): the staged-bout runner, participation census, fabrication guard (C16, C17,
C19, the carriers declared for R2's C20).

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md (2.7, 2.11, P3b).
The brain-backed case builds one short scratch brain under throwaway_home (never the live lake).
"""

import pytest

from snn_data_lake.instruments import staged_sleep_bout as ssb

STAGES = [
    ("nrem", {"presentations": 8, "structural_candidates": None}),
    ("rem", {"presentations": 4, "structural_candidates": None}),
]


def test_self_check_runs():
    ssb.main()


def test_c16_the_guard_refuses_any_flag_on_a_synthetic_item():
    for kw in ({"observed": True}, {"label": "A"}, {"reward": 1.0}):
        with pytest.raises(ssb.FabricationRefused):
            ssb.guard(ssb.Item([0.1], synthetic=True, **kw))
    ssb.guard(ssb.Item([0.1], synthetic=True))


def test_bad_stage_names_and_budgets_refuse():
    with pytest.raises(ValueError):
        ssb.run_bout(ssb._Brain(), [("deep", {"presentations": 1})], rng_seed=0)
    with pytest.raises(ValueError):
        ssb.run_bout(ssb._Brain(), [("nrem", {"presentations": -1})], rng_seed=0)


def test_c19_flat_matches_and_order_is_a_parameter():
    staged = ssb.run_bout(ssb._Brain(), STAGES, rng_seed=3)
    flat = ssb.run_bout(ssb._Brain(), ssb.flat_of(STAGES), rng_seed=3)
    shuffled = ssb.run_bout(ssb._Brain(), list(reversed(STAGES)), rng_seed=3)
    assert ssb.check_compute(staged, flat)
    assert (
        shuffled.stages[0].name == "rem" and shuffled.stages[0].presentations == 0
    )  # nothing replayed yet


def test_carriers_are_one_per_presentation_and_redo_exact():
    a = ssb.run_bout(ssb._Brain(), STAGES, rng_seed=11)
    b = ssb.run_bout(ssb._Brain(), STAGES, rng_seed=11)
    assert len(a.carriers) == a.total_presentations()
    assert [c["bout_rng_draws"] for c in a.carriers] == [c["bout_rng_draws"] for c in b.carriers]
    assert a.carriers[-1]["pause_reason"] == "budget"


def test_a_real_brain_bout_changes_the_learner_and_never_ticks():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.instruments.sleep_pruner_census import _teach
    from snn_data_lake.substrate.home import require_isolated_home, throwaway_home

    with throwaway_home():
        require_isolated_home()
        b = make_brain(seed=12401)
        _teach(b)
        step = b._step
        rep = ssb.run_bout(b, STAGES, rng_seed=12401)
        assert b._step == step
    nrem, rem = rep.stages
    assert nrem.presentations == 8 and nrem.deposits > 0 and "legacy:ixn" in nrem.populations
    assert rem.synthetic == rem.presentations == 4 and rem.deposits == 0
    assert not ssb.participation_red(rep, {"born:none"})
    assert ssb.participation_red(
        rep, {"legacy:ixn"}
    )  # C17: only the 'solved' population moved -> red
