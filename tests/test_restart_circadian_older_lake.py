"""Finding (iii) fix, second pair (launcher D-79): a restore from a lake WITHOUT the circadian
sub-record (a lake written before it existed), or with one it must refuse, behaves as before --
a fresh day -- and SAYS so at `state()["circadian_restore"]`; never a silent default. Every brain
runs in its own interpreter over a private home under tmp_path (policy lens P-1, #6935); never the
live lake. Dev seed 15000 (r2-typed-replay-hydration.md 2.6)."""

import pytest

from tests.test_restart_circadian import SEED, private_home, run_brain

pytestmark = pytest.mark.slow


def _fresh(out):
    assert out["step"] == 0 and out["restored"] is None, out
    return out["note"]


def test_an_older_lake_restores_a_fresh_day_and_says_so(tmp_path):
    note = _fresh(run_brain(private_home(tmp_path, "h"), seed=SEED))
    assert note is not None and note.startswith("absent"), note


@pytest.mark.parametrize(
    "changes, cause",
    [
        ({"day_period": 24}, "day_period"),  # policy lens P-3 (#6935): one refusal cause per record
        ({"tau": 25.0}, "tau"),  # panel lens A m2 (#6910): another clock's constants
        ({"k": 0.5}, "k"),
        ({"target": 0.3}, "target"),
        ({"noise": 0.1}, "noise"),
    ],
)
def test_a_record_bound_to_another_binding_is_refused_naming_it(tmp_path, changes, cause):
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    note = _fresh(run_brain(home, seed=SEED, rewrite=changes))
    assert note.startswith("refused") and f"[{cause!r}]" in note, note


def test_a_record_of_another_seed_is_refused_naming_the_seed(tmp_path):
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED + 1, ticks=5)  # a valid record in every other respect
    note = _fresh(run_brain(home, seed=SEED))
    assert note.startswith("refused") and "['seed']" in note, note


@pytest.mark.parametrize(
    "changes",
    [
        {"step": True},  # panel lens A m1 (#6910): a bool is not step 1
        {"step": -1},
        {"phase": 0.123},  # disagrees with the replay of `step` ticks: not this clock's history
    ],
)
def test_a_malformed_record_is_refused_as_malformed(tmp_path, changes):
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    note = _fresh(run_brain(home, seed=SEED, rewrite=changes))
    assert note.startswith("refused") and "malformed" in note, note


def test_a_stray_seed_keeps_the_record_and_the_newest_kept_copy_is_adopted(tmp_path):
    """Panel lens C MAJOR-1 (#6911, P4b 'foreign records preserved'): a brain of another seed that
    refuses the record KEEPS it before its own flush overwrites it; the record's seed then adopts
    the NEWEST kept copy of its own (coverage review: two kept copies, at steps 5 and 8)."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    run_brain(home, seed=SEED + 1, ticks=2)
    first = run_brain(home, seed=SEED, ticks=3)
    assert "adopted" in first["note"] and first["restored"]["step"] == 5, first["note"]
    run_brain(home, seed=SEED + 1, ticks=2)
    second = run_brain(home, seed=SEED)
    assert "adopted" in second["note"], second["note"]
    assert second["restored"] == {"step": 8, "phase": first["at"]["phase"]}
    assert (second["step"], second["phase"]) == (8, first["at"]["phase"])


def test_a_failed_keep_never_overwrites_the_foreign_record(tmp_path):
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    stray = run_brain(home, seed=SEED + 1, ticks=2, no_keep=True)
    assert stray["record"]["seed"] == SEED and stray["record"]["step"] == 5, stray["record"]


def test_the_record_is_tens_of_bytes_and_holds_no_rng_state(tmp_path):
    """Policy lens A-2 (#6952): the jitter rng is REPLAYED from the seed, never persisted, so a
    flush appends tens of bytes to _T_STATE, not ~7 KB of MT19937 state."""
    out = run_brain(private_home(tmp_path, "h"), seed=SEED, ticks=5)
    assert "rng" not in out["record"] and out["record_bytes"] < 300, out["record"]


def test_a_kill_right_after_the_growth_flush_restores_that_nights_onset(tmp_path):
    """Policy lens A-1 (#6952): R2's K1g kill is os._exit right after the night's growth flush; the
    clock is flushed FIRST in that persist block, so the restore resumes at that onset."""
    home = private_home(tmp_path, "h")
    killed = run_brain(home, seed=SEED, ticks=800, kill_at_growth_flush=True)
    restored = run_brain(home, seed=SEED)
    assert restored["note"] is None and restored["step"] == killed["killed_at"], restored
