"""R3 structural-only recovery controls; real Brain, persistent OS-temp homes.

The r2.v1 sleep event stores digests/progress, not an observations object.
Schema REDs rewrite the real redo step's returned observations in the restart
process, without adding an event slot or a migration codec.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_r2_replay_hydration import (
    _close_for_restart,
    _private_brain,
    _restart,
    _semantic,
    _teach_a,
    _teach_b,
)


def _pending_sleep(brain):
    """Checkpoint a genuine sign-crossing input; append its structural effect later."""
    _teach_a(brain)
    _teach_b(brain)
    assert brain._r2_sleep_learning()  # establish real birth signs
    neuron = brain.brain._ixn.neurons[0]
    column = next(c for c, weight in enumerate(neuron.w) if weight > 1e-6)
    neuron.w[column] = -abs(neuron.w[column])
    assert brain._flush_interaction_learner()
    before = _semantic(brain)
    brain._sleep_ticks += 1  # distinct real redo identity (sleep_id, tick)
    assert brain._r2_sleep_learning()
    assert brain._structural_report["pruned"] > 0
    after = _semantic(brain)
    assert after != before
    return after


def test_pending_structural_sleep_matches_uninterrupted_twin_after_os_restart():
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.learner_events import SLEEP_DIGESTS, SLEEP_PROGRESS

    assert SLEEP_DIGESTS == ("rng_digest", "weights_digest", "rewirer_digest")
    assert SLEEP_PROGRESS == ("drawn", "presentations_spent")
    with _private_brain() as (twin, _):
        expected = _pending_sleep(twin)
    with _private_brain() as (brain, private):
        actual = _pending_sleep(brain)
        assert actual == expected
        event = home.load_episodes(kinds=("sleep",))[-1]
        assert event["presentations_spent"] == "0" and event["drawn"] == "none"
        # Crash boundary: the appended effect is absent from the last checkpoint.
        assert home.load_state("interaction_learner.json")["rewirer"]["cycles"] == 1
        brain._r2_close("constructed_crash", RuntimeError("before effect checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
        assert restarted["hydrations"] == 1
        assert restarted["redo"] == 1


def _restart_with_observation_red(private, profile):
    """Fresh process changes only the output schema of the actual pending sleep step."""
    from snn_data_lake.substrate import home

    assert Path(home.require_isolated_home()).resolve() == private.resolve()
    child = r"""
import gc, json, os
from pathlib import Path
from snn_data_lake.engine.brain import Brain, make_brain
from snn_data_lake.substrate import home
bound = Path(home.require_isolated_home()).resolve()
assert bound == Path(os.environ["NORD_HOME"]).resolve()
original = Brain._r2_sleep_step
witnessed = []
def rewritten(brain, learner, rewirer):
    rewirer, observations = original(brain, learner, rewirer)
    assert set(observations) == {"_structural_report"}
    assert observations["_structural_report"]["pruned"] > 0
    witnessed.append(True)
    if os.environ["R3_OBSERVATION_RED"] == "retired":
        observations.update(_sleep_replayed=0, _sleep_consolidate_calls=0, _plastic_relaxed=0)
    else:
        observations["unknown_observation"] = None
    return rewirer, observations
Brain._r2_sleep_step = rewritten
b = make_brain(seed=0, persist=True)
try:
    assert witnessed == [True], "schema RED must reach the genuine pending structural effect"
    assert b._r2_fault is not None and b._r2_fault["phase"] == "reconcile"
    assert "R3 structural-only sleep hydration observations refused" in b._r2_fault["cause"]
    assert b.brain._ixn._cursor is None and b._rewirer is None
    assert not b.brain._ixn._hydrated and b.brain._ixn._hydration_count == 0
    before = b._r2_snapshot(b.brain._ixn)
    sequence = b._r2_seq
    b.brain.teach("blocked r3 teaching", "GREET")
    assert b._r2_seq == sequence
    # Refusal diagnostics may increment; learning/RNG/cache/cursor may not.
    after = b._r2_snapshot(b.brain._ixn)
    for snap in (before, after):
        snap["learner"]["state"]["tree"]["attrs"].pop("_unjournaled_refused")
    assert before == after
    assert b._flush_interaction_learner() is False
    print("R3 SCHEMA REFUSAL " + json.dumps(b._r2_fault, sort_keys=True), flush=True)
finally:
    assert Path(home.home()).resolve() == bound
    b.close(); del b; gc.collect()
print("R3 RED CLOSED", flush=True)
"""
    remaining = (int(os.environ["SNDL_R2_CHILD_DEADLINE_NS"]) - time.time_ns()) / 1e9
    assert remaining > 0
    result = subprocess.run(
        [sys.executable, "-B", "-X", "gil=0", "-X", "utf8", "-c", child],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=remaining,
        env={**os.environ, "R3_OBSERVATION_RED": profile},
    )
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith(b"R3 RED CLOSED")


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
@pytest.mark.parametrize("profile", ["retired", "unknown"])
def test_pending_sleep_schema_red_refuses_without_partial_install(profile):
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, private):
        _pending_sleep(brain)
        brain._r2_close("constructed_crash", RuntimeError("before effect checkpoint"))
        _close_for_restart(brain)
        prefix = Path(home.knowledge_path()).read_bytes()
        _restart_with_observation_red(private, profile)
        assert Path(home.knowledge_path()).read_bytes() == prefix


@pytest.mark.parametrize("slot", ["presentations_spent", "drawn", "weights_digest"])
def test_pending_structural_sleep_retains_digest_and_progress_reds(monkeypatch, slot):
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, private):
        original = home.save_episode
        sleeps = 0

        def corrupt(event, path=None):
            nonlocal sleeps
            if event["kind"] == "sleep":
                sleeps += 1
                # Round 5 (#2112), 2026-10-13: corrupt ONLY the pending second tick.
                # The first tick is checkpoint-covered and now owes a prefix-binding refusal.
                if sleeps == 2:
                    event = dict(event)
                    event[slot] = {
                        "presentations_spent": "8",
                        "drawn": "1",
                        "weights_digest": "0" * 64,
                    }[slot]
            return original(event, path=path)

        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", corrupt)
            _pending_sleep(brain)
        assert sleeps == 2, "the pending-only corruption plant must be reached"
        brain._r2_close("constructed_crash", RuntimeError("before effect checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"]["phase"] == "reconcile"
        assert "R2 sleep redo digest/progress mismatch" in restarted["fault"]["cause"]
        assert restarted["cursor"] is None and restarted["hydrations"] == 0


def test_retained_k7_turnover_detector_reaches_birth_and_planted_signs(monkeypatch):
    monkeypatch.setenv("NORD_PERSIST", "1")
    monkeypatch.delenv("NORD_SELFMOD", raising=False)
    from snn_data_lake.instruments.sleep_plasticity_ab import run_k7

    result = run_k7(0)
    assert result["aa"] == 0.0 and result["planted"] > 0
    assert result["sighted"] is True
