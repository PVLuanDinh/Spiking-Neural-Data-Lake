"""Owner2106 (2026-10-13): real aggregate bouts, complete-prefix OS restart and named REDs."""

import json

import pytest
from test_r2_replay_hydration import (
    _close_for_restart,
    _private_brain,
    _restart,
    _semantic,
    _teach_a,
    _teach_b,
)

STAGES = [
    ("nrem", {"presentations": 8, "structural_candidates": None}),
    ("rem", {"presentations": 4, "structural_candidates": None}),
]


def _pending_bout(brain):
    from snn_data_lake.instruments.staged_sleep_bout import run_bout

    _teach_a(brain)
    _teach_b(brain)
    assert brain._flush_interaction_learner()
    before = _semantic(brain)
    report = run_bout(brain, STAGES, rng_seed=0)
    assert report.stages[0].deposits > 0 and report.stages[1].synthetic == 4
    assert _semantic(brain) != before
    return _semantic(brain)


@pytest.mark.pool
def test_bout_crash_before_checkpoint_redoes_exactly_in_a_fresh_process():
    with _private_brain() as (twin, _):
        expected = _pending_bout(twin)
    with _private_brain() as (brain, private):
        assert _pending_bout(brain) == expected
        brain._r2_close("constructed_crash", RuntimeError("before bout checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
        assert restarted["redo"] == 1 and restarted["hydrations"] == 1


@pytest.mark.pool
def test_bout_checkpoints_are_not_redone_on_restart():
    with _private_brain() as (brain, private):
        expected = _pending_bout(brain)
        assert brain._flush_interaction_learner()
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected and restarted["redo"] == 0


def test_append_precedes_live_effect_and_rem_never_enters_lake_ring_or_labels(monkeypatch):
    from snn_data_lake.instruments.staged_sleep_bout import run_bout
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, _):
        _teach_a(brain)
        _teach_b(brain)
        before = _semantic(brain)
        events = home.load_episodes(all_kinds=True)
        append = home.save_episode
        seen = []

        def observed(event, path=None):
            assert _semantic(brain) == before, "live learning happened before admission"
            seen.append(event)
            return append(event, path)

        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", observed)
            report = run_bout(brain, STAGES, rng_seed=0)
        assert len(seen) == 1 and seen[0]["kind"] == "sleep_bout"
        assert home.load_episodes(all_kinds=True) == events + seen
        assert report.stages[1].synthetic == 4 and report.stages[1].deposits == 0
        after = _semantic(brain)
        assert after["learner"]["cache"] == before["learner"]["cache"]
        assert set(json.loads(seen[0]["bout"])) == {"profile", "seed", "stages"}


@pytest.mark.parametrize("field", ["profile", "extra", "seed", "weights_digest", "drawn"])
@pytest.mark.pool
def test_tampered_bout_refuses_by_name_without_installing_a_partial_prefix(field):
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import pack_episode, unpack_episode

    with _private_brain() as (brain, private):
        _pending_bout(brain)
        checkpoint = home.load_state("interaction_learner.json")
        brain._r2_close("constructed_crash", RuntimeError("pending bout"))
        _close_for_restart(brain)
        log = home._log()
        records = list(log.replay())
        for i, payload in enumerate(records):
            if payload[:1] != home._T_EPISODE:
                continue
            event = unpack_episode(payload[1:])
            if event["kind"] != "sleep_bout":
                continue
            if field in {"profile", "extra", "seed"}:
                plan = json.loads(event["bout"])
                plan[field] = "unknown" if field != "seed" else 1
                event["bout"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
            else:
                event[field] = "0" * 64 if field == "weights_digest" else "[]"
            records[i] = home._T_EPISODE + pack_episode(event)
        log.rewrite(records)
        restarted = _restart(private)
        assert restarted["fault"] is not None
        assert "sleep_bout" in restarted["fault"]["cause"]
        assert restarted["hydrations"] == 0 and restarted["cursor"] is None
        assert home.load_state("interaction_learner.json") == checkpoint


@pytest.mark.parametrize("landed", [False, True])
@pytest.mark.pool
def test_uncertain_bout_append_rolls_back_latches_and_restarts_from_real_prefix(
    monkeypatch, landed
):
    from snn_data_lake.instruments.staged_sleep_bout import run_bout
    from snn_data_lake.substrate import home

    with _private_brain() as (twin, _):
        expected = _pending_bout(twin)
    with _private_brain() as (brain, private):
        _teach_a(brain)
        _teach_b(brain)
        assert brain._flush_interaction_learner()
        before = _semantic(brain)
        append = home.save_episode

        def uncertain(event, path=None):
            if landed:
                assert append(event, path)
            raise OSError("constructed uncertain sleep_bout append")

        with monkeypatch.context() as patch:
            patch.setattr(home, "save_episode", uncertain)
            with pytest.raises(OSError, match="uncertain sleep_bout"):
                run_bout(brain, STAGES, rng_seed=0)
        assert _semantic(brain) == before
        assert not brain._flush_interaction_learner()
        with pytest.raises(RuntimeError, match="sleep_bout refused by latch"):
            run_bout(brain, STAGES, rng_seed=0)
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == (expected if landed else before)


def test_persistent_callbacks_refuse_instead_of_inventing_redo_decisions():
    from snn_data_lake.instruments.staged_sleep_bout import run_bout
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, _):
        before = _semantic(brain)
        events = home.load_episodes(all_kinds=True)
        with pytest.raises(ValueError, match="sleep_bout validation/wake callbacks are unbound"):
            run_bout(brain, STAGES, rng_seed=0, validate=lambda: 0)
        assert _semantic(brain) == before and home.load_episodes(all_kinds=True) == events
