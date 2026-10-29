"""specs/022 T102 (R1): port of the claude/r1 recall fixes F1-F4 onto main (lane r1-port).

Each test is the pre-registered red of one F item, adapted from claude/r1's
tests/test_r1_recall_fixes.py to main's current code (R2's Brain._session and home.next_session
exist here). Persist brains run in a tmp NORD_HOME with home reloaded, the pattern
tests/test_episode_durability.py uses on main.

Run: pytest tests/test_r1_port.py
"""

import importlib

import pytest

from snn_data_lake.substrate import lake_recall as LR

EPS = [
    {"who": "arm", "what": f"w{i % 7}", "where": f"p{i % 5}", "when": f"t{i}", "affect": "calm"}
    for i in range(200)
]
# no `affect` slot
ODD = [{"who": "arm", "what": "lone", "where": "px", "when": f"u{j}"} for j in range(3)]


@pytest.fixture
def H(tmp_path, monkeypatch):
    """The home module over a fresh tmp NORD_HOME."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    from snn_data_lake.substrate import home

    return importlib.reload(home)


@pytest.fixture
def lake_brain(H):
    """A persist brain over 203 episodes: ODD sits mid-lake, far outside the 64-episode window."""
    from snn_data_lake.engine.brain import make_brain

    for e in EPS[:100] + ODD + EPS[100:]:
        assert H.save_episode(e)
    b = make_brain(seed=0, persist=True)
    try:
        yield b
    finally:
        b.close()


def test_f1_a_missing_slot_never_matches_the_string_none(lake_brain):
    # `str(ep.get("affect")) == "None"` matched an episode LACKING affect at confidence 1.0
    got = lake_brain.recall_episode_durable({"when": "u1", "affect": "None"}, min_conf=1.0)
    assert got["source"] == "miss" and not got["episode"], got
    # the positive: the same nonresident episode IS recalled by a cue it can satisfy
    hit = lake_brain.recall_episode_durable({"when": "u1"}, min_conf=1.0)
    assert hit["source"] == "lake" and hit["episode"] == ODD[1], hit


def test_f1_the_subcue_test_is_presence_checked():
    # the same shape in LakeRecall._is_subcue (it gates the calibration null's rejection sampling)
    lr = LR.LakeRecall(episodes=[ODD[1], EPS[0]], seed=0)
    lr.attach([ODD[1], EPS[0]])  # attached mode builds its episode cache here
    assert not lr._is_subcue({"when": "u1", "affect": "None"})
    assert lr._is_subcue({"when": "u1"})
    assert LR.is_subcue({"when": "u1"}, ODD[1]) and not LR.is_subcue({"affect": "None"}, ODD[1])
    assert not LR.is_subcue({}, ODD[1])  # an empty cue addresses nothing


def test_f2_the_window_never_appends_an_episode_the_lake_refused(H, monkeypatch):
    # F2 isolated from F3: every save reports a duplicate, so the window must not grow. The patch
    # works because brain.py imports save_episode INSIDE the tick, at call time -- hoisting that
    # import to module level would leave the patch unused, so `calls` must be > 0 or the red is
    # vacuous.
    # R2's learner events also go through home.save_episode on main: only autobiographical episodes
    # (no `kind`) are reported as duplicates, the rest pass to the real writer.
    from snn_data_lake.engine.brain import make_brain

    real, calls = H.save_episode, []

    def refuse_autobiographical(episode, path=None):
        if "kind" in episode:
            return real(episode, path)
        calls.append(1)
        return False

    b = make_brain(seed=0, persist=True)
    try:
        monkeypatch.setattr(H, "save_episode", refuse_autobiographical)
        before = list(b._episode_window)
        for _ in range(30):
            b.tick()
        assert calls, "the patched save_episode was never reached: a vacuous red"
        assert b._episode_window == before
    finally:
        b.close()
