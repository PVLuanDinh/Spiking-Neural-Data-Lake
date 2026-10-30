"""Finding (iii) fix, second pair (launcher D-79): a restore from a lake WITHOUT the circadian
sub-record (a lake written before it existed), or with one it must refuse, behaves as before --
a fresh day -- and SAYS so at `state()["circadian_restore"]`; never a silent default. Every brain
runs in its own interpreter over a private home under tmp_path (policy lens P-1, #6935); never the
live lake. Dev seed 15000 (r2-typed-replay-hydration.md 2.6)."""

import pytest

from tests.test_restart_circadian import SEED, private_home, run_brain

pytestmark = pytest.mark.slow

# a list 1,200 deep: the codec's staging recurses once a level (RecursionError)
_DEEP = 0
for _ in range(1200):
    _DEEP = [_DEEP]


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
        {"phase": 1.5},
        {"rng": 5},  # decodes, but not to a random.Random
        {"rng": [3, [1, 2], None]},  # the hand-rolled list form the shared codec replaced
        {"rng": _DEEP},  # coverage review: a decode that raises RecursionError is malformed too
    ],
)
def test_a_malformed_record_is_refused_as_malformed(tmp_path, changes):
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    note = _fresh(run_brain(home, seed=SEED, rewrite=changes))
    assert note.startswith("refused") and "malformed" in note, note


def test_a_bad_cached_gauss_is_refused_as_malformed(tmp_path):
    """api-usage review P3: CPython's Random.setstate checks the version and the 625 words but
    assigns gauss_next unchecked: a string there would restore, then raise on the next tick."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    note = _fresh(run_brain(home, seed=SEED, rewrite_gauss_next="x"))
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


def test_the_record_carries_the_rng_state_through_the_one_codec(tmp_path):
    """Launcher D-82 on ruling #6978 (brief item 9): a replay from the seed costs O(_step),
    unbounded over a long life, so the jitter rng's STATE rides the record, encoded by
    readout/interaction_learning.py's G0(c) codec (the one Random codec, P2)."""
    out = run_brain(private_home(tmp_path, "h"), seed=SEED, ticks=5)
    assert out["record"]["rng"]["__kind__"] == "random", out["record"]


def test_a_kill_right_after_the_growth_flush_restores_that_nights_onset(tmp_path):
    """Policy lens A-1 (#6952): R2's K1g kill is os._exit right after the night's growth flush; the
    clock is flushed FIRST in that persist block, so the restore resumes at that onset."""
    home = private_home(tmp_path, "h")
    killed = run_brain(home, seed=SEED, ticks=800, kill_at_growth_flush=True)
    restored = run_brain(home, seed=SEED)
    assert restored["note"] is None and restored["step"] == killed["killed_at"], restored


@pytest.mark.parametrize(
    "rewrite, replace",
    [
        ({"rng": [3, [1, 2], None]}, None),  # bound, but this build's codec cannot load it
        ({"step": True}, None),
        (None, [1, 2]),  # not a dict: the restore raises on it
        (None, {}),  # correctness review: a present but empty record is not an absent one
    ],
)
def test_a_record_this_build_cannot_load_is_kept_before_it_is_overwritten(
    tmp_path, rewrite, replace
):
    """Lens C MAJOR-A (#7021), lens A MINOR-5 (#7028), enforcer MINOR-1 (#7053), P4b 'foreign
    records preserved' and home.py's 'a build writing a different layout' is KEPT: a record this
    build refuses as malformed, or whose restore raises (a newer build's layout after a rollback),
    is kept under circadian_clock.refused.clock.<id>.json before the flush overwrites it, and the
    note says so (lens C MINOR-C)."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    spec = {"rewrite": rewrite} if replace is None else {"replace": replace}
    out = run_brain(home, seed=SEED, **spec)
    assert out["step"] == 0 and out["restored"] is None, out
    names = [name for name, rec in out["kept"].items() if rec == out["written"]]
    assert names, (out["kept"], out["written"])
    assert out["note_at_close"].endswith(f"; kept: {names}"), out["note_at_close"]


def test_a_kept_foreign_record_is_named_in_the_note(tmp_path):
    """Lens C MINOR-C (#7021): the prototype path notes `superseded: [...]`; the circadian keep
    appends `; kept: [<name>]` to `state()["circadian_restore"]`."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    stray = run_brain(home, seed=SEED + 1)
    names = [name for name, rec in stray["kept"].items() if rec == stray["written"]]
    assert names and stray["note_at_close"].endswith(f"; kept: {names}"), stray["note_at_close"]


def test_a_failed_circadian_flush_is_counted(tmp_path):
    """Lens A MINOR-1 (#7028): a flush that writes nothing because its keep failed is counted at
    `state()["circadian_flush_failed"]`, never silent (the keep is retried every onset)."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    stray = run_brain(home, seed=SEED + 1, ticks=60, no_keep=True)
    assert stray["flush_failed"] and stray["flush_failed"] >= 1, stray["flush_failed"]
    own = run_brain(home, seed=SEED, ticks=60)  # its record: every onset flush writes
    assert own["nights"] >= 1 and own["flush_failed"] == 0, own


def test_a_malformed_record_adopts_the_newest_kept_copy_it_can_load(tmp_path):
    """Coverage review: a malformed record retries the kept copies, as a foreign one does,
    and skips a kept copy it cannot load. A stray seed keeps seed S's step-5 record; a second
    stray keeps an unloadable record newest; a record bound to S with a bool step then adopts
    the step-5 copy."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    run_brain(home, seed=SEED + 1)
    run_brain(home, seed=SEED + 1, replace=[1, 2])
    out = run_brain(home, seed=SEED, rewrite={"seed": SEED, "step": True})
    assert out["note"].startswith("refused: malformed record; adopted:"), out["note"]
    assert out["restored"]["step"] == 5, out["restored"]


def test_a_malformed_record_never_adopts_a_kept_copy_older_than_its_step(tmp_path):
    """Lens A R3-MINOR-2 (#7181): a malformed record bound to this brain at step 9 (a rollback)
    must not adopt the step-5 kept copy, or the clock jumps back under an 'adopted' note; with no
    kept copy at step >= 9 it is a fresh day."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    run_brain(home, seed=SEED + 1)
    note = _fresh(run_brain(home, seed=SEED, rewrite={"seed": SEED, "step": 9, "phase": 1.5}))
    assert note == "refused: fresh day (malformed record)", note


def test_a_malformed_record_adopts_a_kept_copy_at_its_own_step(tmp_path):
    """Coverage review (round 4): the floor's boundary -- a malformed record of this binding at
    step 5 adopts the step-5 kept copy (step >= floor). A pin: the base adopts it too."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    run_brain(home, seed=SEED + 1)
    out = run_brain(home, seed=SEED, rewrite={"seed": SEED, "step": 5, "phase": 1.5})
    assert out["note"].startswith("refused: malformed record; adopted:"), out["note"]
    assert out["restored"]["step"] == 5, out["restored"]


def test_a_record_missing_a_binding_key_is_malformed_not_loaded(tmp_path):
    """Coverage review (round 4): a record valid but for one missing binding key (written before
    the key joined the binding) is malformed, never loaded under constants it does not name."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    note = _fresh(run_brain(home, seed=SEED, drop="day_period"))
    assert note == "refused: fresh day (malformed record)", note


@pytest.mark.parametrize("replace", [[1, 2], {}])
def test_a_record_not_in_this_builds_layout_is_malformed_and_adopts_a_kept_copy(tmp_path, replace):
    """Lens C MINOR-3/MINOR-4 (#7184): a non-dict record and {} are malformed, not foreign, and
    retry the kept copies: a stray seed keeps seed S's step-5 record, then either adopts it."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    run_brain(home, seed=SEED + 1)
    out = run_brain(home, seed=SEED, replace=replace)
    assert out["note"].startswith("refused: malformed record; adopted:"), out["note"]
    assert out["restored"]["step"] == 5, out["restored"]


@pytest.mark.parametrize("blob", ["null", "{not json"])
def test_a_record_that_is_not_json_is_not_reported_absent(tmp_path, blob):
    """Lens C MINOR-1 (#7184), lens A R3-MINOR-1(a) (#7181): load_state reads a JSON null or a
    non-JSON blob as its default; the note says the record is not JSON and was NOT kept (a
    bytes-level keep is OWED), never 'absent'."""
    note = _fresh(run_brain(private_home(tmp_path, "h"), seed=SEED, replace_bytes=blob))
    assert note == "refused: fresh day (not a JSON record: NOT kept)", note


def test_a_record_the_restore_could_not_read_is_kept_before_it_is_overwritten(tmp_path):
    """Lens A R3-MINOR-1(b) (#7181), enforcer concurring (#7205): when the restore's read
    raises, the first flush re-reads the record and keeps it before the first overwrite."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    out = run_brain(home, seed=SEED, restore_raises=True)
    assert out["step"] == 0 and out["note"].startswith("restore raised"), out["note"]
    names = [name for name, rec in out["kept"].items() if rec == out["written"]]
    assert names and out["note_at_close"].endswith(f"; kept: {names}"), out["note_at_close"]


def test_a_failed_keep_after_a_reread_never_overwrites_the_record(tmp_path):
    """Coverage review (round 4): a re-read record whose keep fails is never overwritten, and
    the failed flush is counted."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    out = run_brain(home, seed=SEED, ticks=60, restore_raises=True, no_keep=True)
    assert out["record"] == out["written"], (out["record"], out["written"])
    assert out["flush_failed"] >= 1, out["flush_failed"]


def test_a_codec_defect_is_reported_as_raised_not_malformed(tmp_path):
    """Lens C MINOR-2 (#7184), lens A R3-MINOR-3 (#7181): load_state_dict refuses only a decode
    failure (KeyError, TypeError, ValueError, RecursionError); a defect in the shared codec reaches
    the restore as 'restore raised: <exc>', still kept, still a fresh day."""
    home = private_home(tmp_path, "h")
    run_brain(home, seed=SEED, ticks=5)
    out = run_brain(home, seed=SEED, codec_raises=True)
    assert out["step"] == 0 and out["note"] == "restore raised: fresh day (codec defect)", out
    assert any(rec == out["written"] for rec in out["kept"].values()), out["kept"]


def test_superseded_circadian_records_are_retired_by_the_nights_consolidation(tmp_path):
    """Lens A MAJOR-A (#7028), lens C MAJOR-B (#7021), enforcer MAJOR-1 (#7053): the night block's
    `klog.consolidate` (nights % 8 == 0, `records` not None) runs `_prune_state_history` before the
    night's flush, so the live circadian history is bounded (at most 8 onsets after a prune, plus
    the survivor and close()'s record), and after a consolidation only the newest record remains.
    A pin of the tree's prune: no circadian code drives it. Enforcer MINOR (#7205): the bound is
    asserted against the unpruned count (one record an onset plus close()'s), which a run whose
    night block never consolidated would reach, and the in-run consolidation's night is read."""
    out = run_brain(private_home(tmp_path, "h"), seed=SEED, ticks=400, consolidate=True)
    assert out["lake_night"] >= 8, out["lake_night"]  # the night block consolidated after night 0
    assert 1 <= out["record_count"] < out["nights"] + 1, (out["record_count"], out["nights"])
    assert (out["consolidated_count"], out["consolidated"]) == (1, out["record"])
