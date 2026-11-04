"""Prospective R2 structural/sleep/checkpoint controls, UNRUN and not deciding science."""

import copy
import gc
from pathlib import Path

import pytest
from test_r2_replay_hydration import (
    _close_for_restart,
    _private_brain,
    _reference,
    _restart,
    _semantic,
    _teach_a,
    _teach_b,
)


def test_real_structural_carrier_roundtrip_preserves_next_rng_and_topology():
    from snn_data_lake.sleep.structural_rewiring import StructuralRewiring

    # Existing local constructor default; this is a constructed carrier check, not a seed cohort.
    left, right = StructuralRewiring(), StructuralRewiring()
    rows = [[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]
    left.rewire(rows)
    state = left.state_dict()
    right.load_state_dict(state)
    assert left.state_dict() == right.state_dict()
    assert right._regen_signs == left._regen_signs
    assert right._rng.getstate() == left._rng.getstate()
    rows[0][0] = -rows[0][0]  # real identity crossing forces prune/regen/RNG draw
    right_rows = copy.deepcopy(rows)
    left.rewire(rows)
    right.rewire(right_rows)
    assert rows == right_rows
    assert left.state_dict() == right.state_dict()
    assert left.regen_signs(), "R2 carrier positive must actually exercise int-key sign entries"
    # Detached report/topology: changing the returned value cannot change the carrier.
    snapshot = left.state_dict()
    expected = copy.deepcopy(snapshot)
    snapshot["signs"][0][0] = 0
    assert left.state_dict() == expected


@pytest.mark.parametrize("corruption", ["duplicate", "nonfinite", "boolrow", "sign", "rng"])
def test_structural_carrier_refuses_bad_state_without_partial_install(corruption):
    from snn_data_lake.sleep.structural_rewiring import StructuralRewiring

    carrier = StructuralRewiring()
    rows = [[1.0, -1.0]]
    carrier.rewire(rows)
    rows[0][0] = -1.0
    carrier.rewire(rows)
    before = carrier.state_dict()
    bad = copy.deepcopy(before)
    if corruption == "duplicate":
        bad["regen_signs"].append(copy.deepcopy(bad["regen_signs"][0]))
    elif corruption == "nonfinite":
        bad["last"]["turnover"] = float("nan")
    elif corruption == "boolrow":
        bad["regen_signs"][0][0] = True
    elif corruption == "sign":
        bad["signs"][0][0] = 2
    else:
        bad["rng"][1] = []
    with pytest.raises(ValueError):
        carrier.load_state_dict(bad)
    assert carrier.state_dict() == before, "R2 malformed structural carrier mutated real state"


def test_missing_structural_carrier_refuses_before_wake_admission(monkeypatch):
    from snn_data_lake.sleep.structural_rewiring import StructuralRewiring
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, _):
        _teach_a(brain)
        before = _semantic(brain)
        seq = brain._r2_seq
        records = home.load_episodes(kinds=("teach",))
        with monkeypatch.context() as patch:
            patch.setattr(StructuralRewiring, "state_dict", None)
            with pytest.raises(ValueError, match="missing real structural carrier"):
                _teach_b(brain)
        assert _semantic(brain) == before
        assert brain._r2_seq == seq
        assert home.load_episodes(kinds=("teach",)) == records
        assert brain._r2_fault is not None
        assert brain._flush_interaction_learner() is False


@pytest.mark.pool
def test_real_sleep_redo_matches_complete_tick_in_new_process():
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, private):
        _teach_a(brain)
        _teach_b(brain)
        assert brain._flush_interaction_learner()
        before_rng = brain.brain._ixn.rng.getstate()
        # Exactly the common learner block, not the other grower/autotelic sleep capabilities.
        assert brain._r2_sleep_learning()
        expected = _semantic(brain)
        # Owner1039, 2026-10-13: structural sleep has zero replay presentations.
        assert set(brain._r2_observations()) == {"_structural_report"}
        assert brain.brain._ixn.rng.getstate() == before_rng
        assert brain._rewirer.state_dict()["cycles"] > 0
        event = home.load_episodes(kinds=("sleep",))[-1]
        assert event["drawn"] == "none" and event["presentations_spent"] == "0"
        assert event["pending"] == "none" and event["applied_mask_digest"] == "none"
        assert brain.brain._ixn._cursor is not None
        # Preserve wake checkpoint A+B; do not replace it with post-sleep continuation.
        brain._r2_close("constructed_crash", RuntimeError("stop before sleep checkpoint"))
        _close_for_restart(brain)
        receipt = _restart(private)
        assert receipt["fault"] is None
        assert receipt["semantic"] == expected, "R2 actual sleep redo differs from completed tick"


def test_partial_rewire_mutation_rolls_back_real_topology_aliases_and_rng(monkeypatch):
    from snn_data_lake.sleep.structural_rewiring import StructuralRewiring

    with _private_brain() as (brain, _):
        _teach_a(brain)
        assert brain._r2_sleep_learning()
        before = _semantic(brain)
        rewirer, stack = brain._rewirer, brain._plasticity
        rng = rewirer._rng
        original, witnessed = StructuralRewiring.rewire, []

        def real_then_raise(carrier, rows, activity=None):
            original(carrier, rows, activity=activity)
            witnessed.append(carrier.state_dict())
            assert witnessed[-1] != before["rewirer"], (
                "R2 structural control must mutate cycle state"
            )
            raise RuntimeError("R2 actual post-rewire failure")

        with monkeypatch.context() as patch:
            patch.setattr(StructuralRewiring, "rewire", real_then_raise)
            with pytest.raises(RuntimeError, match="post-rewire failure"):
                brain._r2_sleep_learning()
        assert len(witnessed) == 1
        assert _semantic(brain) == before, "R2 structural rollback RED"
        assert brain._rewirer is rewirer
        assert brain._rewirer._rng is rng
        assert brain._plasticity is stack is brain.brain._ixn.stack
        assert brain._r2_fault is not None


def test_sleep_append_failure_restores_full_state_and_keeps_latch(monkeypatch):
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, _):
        _teach_a(brain)
        before = _semantic(brain)
        real_save, witnessed = home.save_episode, []

        def sleep_refusal(event, path=None):
            if event["kind"] == "sleep":
                witnessed.append(dict(event))
                return False
            return real_save(event, path=path)

        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", sleep_refusal)
            with pytest.raises(OSError, match="append returned False"):
                brain._r2_sleep_learning()
        assert len(witnessed) == 1
        assert witnessed[0]["presentations_spent"] == "0"
        assert _semantic(brain) == before
        seq = brain._r2_seq
        assert not brain._r2_sleep_learning()
        brain.brain.teach("blocked private c", "GREET")
        assert brain._r2_seq == seq
        assert _semantic(brain) == before


@pytest.mark.parametrize("arm", ["absent", "landed", "torn"])
@pytest.mark.pool
def test_checkpoint_failure_preserves_actual_event_prefix_and_reconciles_complete_b(
    monkeypatch, tmp_path, arm
):
    import os

    from snn_data_lake.substrate import home

    expected = _reference(True)
    with _private_brain() as (brain, private):
        _teach_a(brain)
        assert brain._flush_interaction_learner()
        a_checkpoint = copy.deepcopy(home.load_state("interaction_learner.json"))
        _teach_b(brain)
        real_save, witnessed = home.save_state, []

        def refuse_checkpoint(name, value, *args, **kwargs):
            assert name == "interaction_learner.json"
            witnessed.append(copy.deepcopy(value))
            if arm == "landed":
                assert real_save(name, value, *args, **kwargs)
                raise OSError("R2 checkpoint landed then failed")
            if arm == "torn":
                with open(home.knowledge_path(), "ab") as stream:
                    stream.write(b"\x7f\x00")
                    stream.flush()
                    os.fsync(stream.fileno())
            return False

        with monkeypatch.context() as patch:
            patch.setattr(home, "save_state", refuse_checkpoint)
            assert brain._flush_interaction_learner() is False
        assert len(witnessed) == 1
        assert brain._r2_fault is not None
        if arm == "absent":
            assert home.load_state("interaction_learner.json") == a_checkpoint
        raw = Path(home.knowledge_path()).read_bytes()
        retained = tmp_path / f"checkpoint-{arm}-before-repair.bin"
        retained.write_bytes(raw)
        _close_for_restart(brain)
        receipt = _restart(private)
        assert receipt["fault"] is None
        assert receipt["semantic"] == expected, "R2 checkpoint recovery may not erase completed B"
        assert retained.read_bytes() == raw


def test_mutable_zoo_or_external_sink_is_refused_without_learning(monkeypatch):
    with _private_brain() as (brain, _):
        _teach_a(brain)
        learner = brain.brain._ixn
        sequence = brain._r2_seq
        # Actual optional interface remains unbound, not erased from a permissive snapshot.
        with monkeypatch.context() as patch:
            patch.setattr(learner, "readout_zoo", object())
            with pytest.raises(ValueError, match="optional zoo"):
                _teach_b(brain)
        assert brain._r2_fault is not None
        assert brain._r2_seq == sequence
    with _private_brain() as (brain, _):
        learner = brain.brain._ixn
        with monkeypatch.context() as patch:
            patch.setattr(learner, "delta_sink", lambda value: None)
            with pytest.raises(ValueError, match="external effect"):
                _teach_a(brain)
        assert brain._r2_seq == 0
        assert brain._r2_fault is not None


def test_rearm_neuter_and_always_refuse_have_reachable_positive_failures(monkeypatch):
    from snn_data_lake.engine.brain import Brain, make_brain
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner
    from snn_data_lake.substrate import home

    real_hydrate = SpikingInteractionLearner.hydrate

    def declared_incomplete(learner, *args, **kwargs):
        report = real_hydrate(learner, *args, **kwargs)
        report["complete"] = False  # actual executed prefix, made explicitly non-admissible
        return report

    with home.throwaway_home():
        home.require_isolated_home()
        with monkeypatch.context() as patch:
            patch.setattr(SpikingInteractionLearner, "hydrate", declared_incomplete)
            brain = make_brain(seed=0, persist=True)
        try:
            assert brain._r2_fault is not None
            assert not brain._flush_interaction_learner()
        finally:
            home.require_isolated_home()
            brain.close()
            del brain
            gc.collect()
    with home.throwaway_home():
        home.require_isolated_home()
        with monkeypatch.context() as patch:
            patch.setattr(SpikingInteractionLearner, "hydrate", declared_incomplete)
            patch.setattr(Brain, "_r2_may_rearm", staticmethod(lambda report: True))
            brain = make_brain(seed=0, persist=True)
        try:
            brain.brain.teach("blocked private c", "GREET")
            with pytest.raises(AssertionError, match="rearm-neuter RED"):
                assert brain._r2_fault is not None, "rearm-neuter RED"
            assert brain.brain._ixn._cursor is not None
        finally:
            home.require_isolated_home()
            brain.close()
            del brain
            gc.collect()
    with _private_brain() as (brain, _):
        before = _semantic(brain)
        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", lambda event, path=None: False)
            _teach_a(brain)
        with pytest.raises(AssertionError, match="always-refuse RED"):
            assert _semantic(brain) != before, "always-refuse RED"
