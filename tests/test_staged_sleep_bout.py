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


def test_only_teach_and_unsplit_entries_are_replay_eligible():
    """Codex panel MAJOR: only 'heldout' was excluded; validation and future leaked into replay."""
    keys = [
        None,
        {},
        {"split": "teach"},
        {"split": "validation"},
        {"split": "future"},
        {"split": "future_d"},
        {"split": "heldout"},
        {"split": "novel_split"},
    ]
    ixn = type("I", (), {})()
    ixn._replay = [([0.0], "A", k) for k in keys]
    got = [k for _, _, k in ssb._eligible_ring(ixn)]
    assert got == [None, {}, {"split": "teach"}]


def test_a_nonzero_structural_budget_is_refused_until_a_consumer_exists():
    """Codex panel MAJOR (P3): the field was accepted and never used."""
    for bad in (1, 5):
        with pytest.raises(ValueError):
            ssb.run_bout(
                ssb._Brain(),
                [("nrem", {"presentations": 1, "structural_candidates": bad})],
                rng_seed=0,
            )
    for ok in (None, 0):
        ssb.run_bout(
            ssb._Brain(), [("nrem", {"presentations": 1, "structural_candidates": ok})], rng_seed=0
        )


def test_run_bout_leaves_no_deposit_wrapper_in_the_learner_dict():
    """R3 finding: a bound method left in the learner __dict__ breaks the checkpoint
    save. A learner with no instance-level `_deposit` must have none after the bout."""
    b = ssb._Brain()
    assert "_deposit" not in vars(b.brain._ixn)
    ssb.run_bout(b, STAGES, rng_seed=3)
    assert "_deposit" not in vars(b.brain._ixn)


def test_run_bout_restores_an_instance_level_deposit_it_found():
    b = ssb._Brain()
    own = b.brain._ixn._deposit
    b.brain._ixn._deposit = own
    ssb.run_bout(b, STAGES, rng_seed=3)
    assert vars(b.brain._ixn)["_deposit"] is own


def _real_learner_brain():
    """A REAL `SpikingInteractionLearner` behind the runner's `brain.brain._ixn` handle, taught
    through the real teaching path so the replay ring is filled the way the live path fills it."""
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    class _LB:
        pass

    ixn = SpikingInteractionLearner(("A", "B"), 4, seed=7)
    for code, label in (([1.0, 0, 0, 0], "A"), ([0, 1.0, 0, 0], "B")):
        ixn.interact(list(code), label)
    brain = type("B", (), {})()
    brain.brain = _LB()
    brain.brain._ixn = ixn
    brain._plasticity = None
    brain._step = 0
    return brain, ixn


def test_the_real_g0c_checkpoint_save_is_not_refused_after_a_bout():
    """R3's challenge on G6-2 (#334, option a): 'no bound method left' is not 'the encoder accepts
    the learner'. The K5 defect IS a refused checkpoint save -- G0(c)'s `_encode_state` refuses a
    bound `_deposit` left in the learner's `__dict__` -- so this runs the REAL save over a REAL
    learner after a real bout and restores the record into a fresh one."""
    brain, ixn = _real_learner_brain()
    rep = ssb.run_bout(brain, STAGES, rng_seed=3)
    assert rep.stages[0].deposits > 0, "a bout that taught nothing proves nothing about the save"
    blob = ixn.state_dict()  # raises ValueError while a bound `_deposit` is left in `__dict__`
    fresh = type(ixn)(ixn.responses, ixn.n_in, seed=999)
    fresh.load_state_dict(blob)
    assert [n.w for n in fresh.neurons] == [n.w for n in ixn.neurons]
    assert [c for c, _, _ in fresh._replay] == [c for c, _, _ in ixn._replay]


def test_a_leftover_bound_deposit_still_makes_the_real_save_refuse():
    # the constructed red for the test above: the defect R3 named, reproduced on the real encoder
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    _, ixn = _real_learner_brain()
    ixn._deposit = ixn._deposit  # the defect the had_own restore-or-delete repairs
    with pytest.raises(ValueError, match="unsupported checkpoint value"):
        ixn.state_dict()
    assert isinstance(ixn, SpikingInteractionLearner)
