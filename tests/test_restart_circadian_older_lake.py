"""Finding (iii) fix, second pair (launcher D-79): a restore from a lake WITHOUT the circadian
sub-record (a lake written before it existed), or with one it must refuse, behaves as before --
a fresh day -- and SAYS so at `state()["circadian_restore"]`; never a silent default. Private homes
under tmp_path only; never the live lake. Dev seed 15000 (r2-typed-replay-hydration.md 2.6)."""

SEED = 15000


def test_an_older_lake_restores_a_fresh_day_and_says_so(tmp_path):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        brain = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            note = brain.state()["circadian_restore"]
            assert note is not None and "fresh day" in note, note
            assert brain._step == 0
        finally:
            brain.close()


def test_a_record_of_another_seed_is_refused_visibly(tmp_path):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        home.save_state(
            "circadian_clock.json",
            {"seed": SEED + 1, "day_period": 24, "step": 7, "phase": 0.9, "rng": []},
        )
        brain = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            note = brain.state()["circadian_restore"]
            assert note is not None and note.startswith("refused"), note
            assert brain._step == 0
        finally:
            brain.close()
