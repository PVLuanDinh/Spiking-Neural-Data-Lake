"""Prospective R2 causal controls. UNRUN; Brain tests require a separate native semantic grant.

The frozen source-only draft is not a runnable acceptance controller. The future controller
must supply a bounded SNDL_R2_CHILD_DEADLINE_NS and preserve complete raw pytest/child streams.
Collection imports no SNDL module. Seed 0 is make_brain's existing nondeciding fixture default.
"""

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _private_brain():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home
    with home.throwaway_home() as private:
        bound = home.require_isolated_home()
        brain = make_brain(seed=0, persist=True)
        assert Path(home.home()).resolve() == Path(bound).resolve()
        try:
            yield brain, Path(private)
        finally:
            assert Path(home.home()).resolve() == Path(bound).resolve()
            home.require_isolated_home()
            if not getattr(brain, "_r2_draft_closed", False):
                brain.close()
            del brain
            gc.collect()
            assert Path(home.home()).resolve() == Path(bound).resolve()


def _close_for_restart(brain):
    from snn_data_lake.substrate import home
    home.require_isolated_home()
    brain.close()
    brain._r2_draft_closed = True
    gc.collect()


def _semantic(brain):
    snap = brain._r2_snapshot(brain.brain._ixn)
    attrs = snap["learner"]["state"]["tree"]["attrs"]
    # Exact refusal/redo diagnostics are asserted separately, never broad "private" exclusions.
    for name in ("_redo_count", "_unjournaled_refused"):
        attrs.pop(name)
    return json.loads(json.dumps(snap, sort_keys=True, allow_nan=False))


def _diagnostics(brain):
    ixn = brain.brain._ixn
    return {
        "redo": ixn._redo_count, "refused": ixn._unjournaled_refused,
        "hydrations": ixn._hydration_count, "fault": copy.deepcopy(brain._r2_fault),
    }


def _teach_a(brain):
    brain.brain.teach("hello private r2", "GREET")


def _teach_b(brain):
    brain.brain.teach("set a private goal", "GOAL")


def _reference(include_b):
    with _private_brain() as (brain, _):
        assert brain._r2_fault is None
        before = _semantic(brain)
        _teach_a(brain)
        a = _semantic(brain)
        assert a != before, "R2 positive A must alter learned state/cache"
        if include_b:
            _teach_b(brain)
            assert _semantic(brain) != a, "R2 positive B must fully finish"
        assert brain.brain._ixn._cursor is not None
        return _semantic(brain)


def _restart(private):
    """Actual OS restart, with no second writer/twin schedule and a required future deadline."""
    from snn_data_lake.substrate import home
    assert Path(home.require_isolated_home()).resolve() == private.resolve()
    assert not sys._is_gil_enabled(), "R2 restart requires the native no-GIL parent"
    deadline = int(os.environ["SNDL_R2_CHILD_DEADLINE_NS"])
    remaining = (deadline - time.time_ns()) / 1e9
    assert remaining > 0, "R2 future controller deadline expired"
    child = r'''
import gc, json, os, sys
from pathlib import Path
from snn_data_lake.substrate import home
from snn_data_lake.engine.brain import make_brain
bound=Path(home.require_isolated_home()).resolve()
assert bound==Path(os.environ["NORD_HOME"]).resolve()
assert not sys._is_gil_enabled()
b=make_brain(seed=0,persist=True)
try:
 s=b._r2_snapshot(b.brain._ixn)
 for k in ("_redo_count","_unjournaled_refused"): s["learner"]["state"]["tree"]["attrs"].pop(k)
 d={"semantic":json.loads(json.dumps(s,sort_keys=True,allow_nan=False)),
    "fault":b._r2_fault,"cursor":b.brain._ixn._cursor,
    "hydrations":b.brain._ixn._hydration_count,"redo":b.brain._ixn._redo_count,
    "receipt_sha":None if b._r2_receipt is None else __import__("hashlib").sha256(b._r2_receipt).hexdigest()}
 print("R2 RESTART RECEIPT "+json.dumps(d,sort_keys=True,allow_nan=False),flush=True)
finally:
 assert Path(home.home()).resolve()==bound
 home.require_isolated_home()
 b.close();del b;gc.collect()
 assert Path(home.home()).resolve()==bound
print("R2 RESTART CLOSED",flush=True)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-X", "gil=0", "-X", "utf8", "-c", child],
        capture_output=True, timeout=remaining,
    )
    # Preserve all native bytes (including bypass/shutdown), not merely the final JSON.
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.flush()
    assert result.returncode == 0, "R2 fresh native process must close successfully"
    assert result.stdout.rstrip().endswith(b"R2 RESTART CLOSED")
    matches = [line for line in result.stdout.splitlines() if line.startswith(b"R2 RESTART RECEIPT ")]
    assert len(matches) == 1, "R2 complete restart receipt must be uniquely attributed"
    return json.loads(matches[0][len(b"R2 RESTART RECEIPT "):])


def test_complete_checkpoint_and_os_restart_match_completed_a_plus_b():
    expected = _reference(True)
    with _private_brain() as (brain, private):
        _teach_a(brain)
        _teach_b(brain)
        actual = _semantic(brain)
        assert actual == expected
        assert brain._flush_interaction_learner()
        assert brain._flush_interaction_learner(), "exact unchanged durable payload is valid dedup"
        # The first process closes all handles while its bound private home is still active.
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected, "R2 complete-prefix restart differs"
        assert restarted["hydrations"] == 1


@pytest.mark.parametrize("arm", ["absent", "landed", "torn"])
def test_failed_b_latches_and_os_restart_uses_actual_durable_prefix(monkeypatch, tmp_path, arm):
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import _LEN, pack_episode
    _T_EPISODE = home._T_EPISODE
    expected_a, expected_ab = _reference(False), _reference(True)
    with _private_brain() as (brain, private):
        _teach_a(brain)
        assert brain._flush_interaction_learner()
        a_state = _semantic(brain)
        a_checkpoint = copy.deepcopy(home.load_state("interaction_learner.json"))
        sequence = brain._r2_seq
        real_save, witnessed = home.save_episode, []
        def fail_b(event, path=None):
            assert event["source"] == "set a private goal"
            witnessed.append(dict(event))
            if arm == "landed":
                assert real_save(event, path=path)
                raise OSError("R2 witnessed landed-B append failure")
            if arm == "torn":
                payload = _T_EPISODE + pack_episode(event)
                with open(home.knowledge_path(), "ab") as stream:
                    stream.write(_LEN.pack(len(payload)) + payload[:-1])
                    stream.flush()
                    os.fsync(stream.fileno())
            return False
        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", fail_b)
            _teach_b(brain)
        assert len(witnessed) == 1, "R2 B failure must be reached, not a pre-admission refusal"
        assert brain._r2_seq == sequence + 1
        assert brain._r2_fault is not None
        assert _semantic(brain) == a_state, "R2 failed append must restore complete A state/RNG"
        assert _diagnostics(brain)["refused"] == 1
        seq_after_b = brain._r2_seq
        brain.brain.teach("blocked private c", "GREET")
        brain.brain._ixn.learn_turn("blocked context c", "GREET")
        assert _semantic(brain) == a_state, "R2 blocked wake/context may not learn"
        assert brain._r2_seq == seq_after_b, "R2 blocked C may not allocate identity"
        assert _diagnostics(brain)["refused"] == 3
        assert brain._r2_sleep_learning() is False
        assert brain._flush_interaction_learner() is False
        assert home.load_state("interaction_learner.json") == a_checkpoint
        raw_prefix = Path(home.knowledge_path()).read_bytes()
        # A real pre-repair artifact, before the fresh process invokes canonical BinLog repair.
        frozen = tmp_path / f"r2-{arm}-raw-prefix.bin"
        frozen.write_bytes(raw_prefix)
        assert frozen.read_bytes() == raw_prefix
        print("R2 PRE-RESTART PREFIX", arm, len(raw_prefix), hashlib.sha256(raw_prefix).hexdigest())
        _close_for_restart(brain)
        receipt = _restart(private)
        expected = expected_ab if arm == "landed" else expected_a
        assert receipt["fault"] is None
        assert receipt["semantic"] == expected, "R2 durable-prefix oracle mismatch"
        assert receipt["semantic"] != (expected_a if arm == "landed" else expected_ab)
        assert frozen.read_bytes() == raw_prefix, "R2 original prefix history must remain immutable"


def test_class_stabilize_partial_b_rolls_back_then_replays_complete_b(monkeypatch):
    from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
    from snn_data_lake.substrate import home
    expected = _reference(True)
    with _private_brain() as (brain, private):
        _teach_a(brain)
        assert brain._flush_interaction_learner()
        a = _semantic(brain)
        a_cursor = brain.brain._ixn._cursor
        real = PlasticityRuleStack.stabilize
        witnessed = []
        def mutate_then_raise(stack, *args, **kwargs):
            result = real(stack, *args, **kwargs)
            witnessed.append(_semantic(brain))
            assert witnessed[-1] != a, "R2 B class control must witness actual partial mutation"
            raise RuntimeError("R2 witnessed class-level partial-B stabilize")
        with monkeypatch.context() as patch:
            patch.setattr(PlasticityRuleStack, "stabilize", mutate_then_raise)
            with pytest.raises(RuntimeError, match="partial-B stabilize"):
                _teach_b(brain)
        assert len(witnessed) == 1
        assert brain.brain._ixn._cursor == a_cursor
        assert _semantic(brain) == a, "R2 complete preeffect state/RNG rollback RED"
        assert brain._plasticity is brain.brain._ixn.stack
        events = home.load_episodes(kinds=("teach",))
        assert events[-1]["source"] == "set a private goal", "R2 admitted B must remain durable"
        assert brain._r2_fault is not None
        assert not brain._flush_interaction_learner()
        _close_for_restart(brain)
        receipt = _restart(private)
        assert receipt["fault"] is None
        assert receipt["semantic"] == expected, "R2 replay must finish independently completed A+B"


def test_latch_and_rollback_neuters_have_named_reachable_reds(monkeypatch):
    from snn_data_lake.engine.brain import Brain
    from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
    from snn_data_lake.substrate import home
    # Cause 1: removing the latch allows a real C frame/effect after a witnessed absent B.
    with _private_brain() as (brain, _):
        _teach_a(brain)
        before = _semantic(brain)
        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", lambda event, path=None: False)
            patch.setattr(Brain, "_r2_close", lambda self, phase, cause: None)
            _teach_b(brain)
        brain.brain.teach("blocked private c", "GREET")
        with pytest.raises(AssertionError, match="latch-neuter RED"):
            assert _semantic(brain) == before, "latch-neuter RED"
    # Cause 2: removing rollback leaves actual delegated stack/neuron/cache mutation behind.
    with _private_brain() as (brain, _):
        _teach_a(brain)
        before = _semantic(brain)
        real = PlasticityRuleStack.stabilize
        def broken(stack, *args, **kwargs):
            real(stack, *args, **kwargs)
            raise RuntimeError("R2 delegated rollback control")
        with monkeypatch.context() as patch:
            patch.setattr(PlasticityRuleStack, "stabilize", broken)
            patch.setattr(Brain, "_r2_restore", lambda *args: None)
            with pytest.raises(RuntimeError, match="delegated rollback"):
                _teach_b(brain)
        with pytest.raises(AssertionError, match="rollback-neuter RED"):
            assert _semantic(brain) == before, "rollback-neuter RED"


def test_strict_same_pass_rejects_complete_noncanonical_episode_and_unbound_percept():
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import pack_episode
    _T_EPISODE = home._T_EPISODE
    from snn_data_lake.substrate.learner_events import KIND_SPECS, scan_learner_events
    with home.throwaway_home():
        home.require_isolated_home()
        assert home._log().append(_T_EPISODE + pack_episode({"kind": "teach"}) + b"trailing")
        events, _, _, receipt = scan_learner_events(
            tuple(KIND_SPECS), strict=True, return_receipt=True,
        )
        proof = json.loads(receipt)
        assert events == []
        assert proof["complete"] is False
        assert proof["critical_ordinals"], "R2 complete malformed episode must be critical"
    with home.throwaway_home():
        home.require_isolated_home()
        assert home.save_episode({"kind": "percept", "schema": "r2.v1", "session": "7"})
        events, _, session, receipt = scan_learner_events(
            tuple(KIND_SPECS), strict=True, return_receipt=True,
        )
        assert session == 8, "all-session boot numbering includes the actual unbound row"
        assert events == []
        assert json.loads(receipt)["complete"] is False


def test_failed_hydration_keeps_sink_bound_and_checkpoint_closed(monkeypatch):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home
    with home.throwaway_home():
        home.require_isolated_home()
        assert home.save_episode({"kind": "percept", "schema": "r2.v1"})
        brain = make_brain(seed=0, persist=True)
        try:
            assert brain._r2_fault is not None
            assert callable(brain.brain._ixn.event_sink)
            before = _semantic(brain)
            brain.brain.teach("blocked private c", "GREET")
            assert _semantic(brain) == before
            assert brain._r2_seq == 0
            assert not brain._flush_interaction_learner()
        finally:
            home.require_isolated_home()
            brain.close()
            del brain
            gc.collect()


def test_one_hydration_and_cache_snapshot_alias_control():
    with _private_brain() as (brain, _):
        _teach_a(brain)
        learner = brain.brain._ixn
        snapshot = learner._snapshot()
        before = copy.deepcopy(snapshot)
        learner._replay[0][0][0] += 1.0
        assert snapshot == before, "R2 snapshot aliases mutable ring stimulus"
        learner._restore_snapshot(snapshot)
        assert learner._snapshot() == snapshot
        assert learner._hydration_count == 1
        with pytest.raises(ValueError, match="once"):
            learner.hydrate([], receipt=b"{}", sleep_step={})
        assert learner._hydration_count == 1
