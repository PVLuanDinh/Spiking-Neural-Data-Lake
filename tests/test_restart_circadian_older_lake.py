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


def _ticked(seed, ticks):
    """A seed-`seed` brain ticks `ticks` times and closes (the close flushes its clock record)."""
    from snn_data_lake.engine.brain import make_brain

    brain = make_brain(seed=seed, persist=True, device="cpu")
    try:
        for _ in range(ticks):
            brain.tick()
        return brain._step, brain.clock.phase
    finally:
        brain.close()


def test_a_stray_seed_does_not_erase_the_record_it_refused(tmp_path):
    """Panel lens C MAJOR-1 (#6911, P4b 'foreign records preserved'): a brain of another seed that
    refuses the record must KEEP it before its own flush overwrites it, and the next brain of the
    record's seed adopts the kept copy and resumes its phase (the prototype path's R3-M2)."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        step, phase = _ticked(SEED, 5)
        _ticked(SEED + 1, 2)
        brain = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            state = brain.state()
            assert "adopted" in (state["circadian_restore"] or ""), state["circadian_restore"]
            assert (brain._step, brain.clock.phase) == (step, phase)
            assert state["circadian_restored"] == {"step": step, "phase": phase}
        finally:
            brain.close()


def _rewritten(**changes):
    """Rewrite the lake's circadian record with `changes`, then restore a seed-SEED brain;
    returns its (note, _step)."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    name = "circadian_clock.json"
    home.save_state(name, dict(home.load_state(name), **changes))
    brain = make_brain(seed=SEED, persist=True, device="cpu")
    try:
        return brain.state()["circadian_restore"], brain._step
    finally:
        brain.close()


def test_a_record_of_another_clock_or_a_bool_step_is_refused(tmp_path):
    """Panel lens A m2 / m1 (#6910): a record written under other clock constants (tau) is refused,
    and a bool step is malformed, not step 1."""
    from snn_data_lake.substrate import home

    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        _ticked(SEED, 5)
        note, step = _rewritten(tau=25.0)
        assert note is not None and note.startswith("refused") and step == 0, note
        _ticked(SEED, 5)
        note, step = _rewritten(step=True)
        assert note is not None and note.startswith("refused") and step == 0, note
