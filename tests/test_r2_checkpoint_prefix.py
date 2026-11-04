"""Round 5, 2026-10-13: checkpoint content binding after review CHALLENGE #2112."""

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
from test_r2_sleep_bout import STAGES


def _completed(brain):
    from snn_data_lake.instruments.staged_sleep_bout import run_bout

    _teach_a(brain)
    _teach_b(brain)
    brain.brain._ixn.learn_turn("a private turn", "GREET")
    run_bout(brain, STAGES, rng_seed=0)
    assert brain._flush_interaction_learner()
    return _semantic(brain)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("sleep_bout", "seed"),
        ("sleep_bout", "weights_digest"),
        ("teach", "source"),
        ("teach", "target"),
        ("turn", "source"),
        ("turn", "target"),
        ("sleep", "weights_digest"),
    ],
)
@pytest.mark.pool
def test_checkpointed_payload_tamper_refuses_fresh_and_closed(kind, field):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import pack_episode, unpack_episode

    with _private_brain() as (brain, private):
        fresh = _semantic(brain)
        assert _completed(brain) != fresh
        if kind == "sleep":
            assert brain._r2_sleep_learning()
            assert brain._flush_interaction_learner()
        _close_for_restart(brain)
        checkpoint = home.load_state("interaction_learner.json")
        log = home._log()
        records = list(log.replay())
        changed = 0
        for i, payload in enumerate(records):
            if payload[:1] != home._T_EPISODE:
                continue
            event = unpack_episode(payload[1:])
            if event["kind"] != kind or changed:
                continue
            if field == "seed":
                plan = json.loads(event["bout"])
                assert plan["seed"] == 0
                plan["seed"] = 1
                event["bout"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
            elif field == "weights_digest":
                event[field] = "0" * 64
            elif field == "source":
                event[field] = "changed private source"
            else:
                event[field] = "GOAL" if event[field] == "GREET" else "GREET"
            records[i] = home._T_EPISODE + pack_episode(event)
            changed += 1
        assert changed == 1, "the RED must mutate exactly one eligible covered event"
        log.rewrite(records)
        restarted = _restart(private)
        assert restarted["fault"] is not None
        assert "checkpoint canonical prefix digest mismatch" in restarted["fault"]["cause"]
        assert restarted["semantic"] == fresh
        assert restarted["hydrations"] == 0 and restarted["cursor"] is None
        assert restarted["redo"] == 0
        assert home.load_state("interaction_learner.json") == checkpoint
        frozen = list(home._log().replay())
        closed = make_brain(seed=0, persist=True)
        try:
            assert closed._r2_fault is not None and closed._session is None
            before = _semantic(closed)
            _teach_a(closed)
            assert _semantic(closed) == before
            assert not closed._r2_sleep_learning()
            assert not closed._flush_interaction_learner()
        finally:
            closed.close()
        assert home.load_state("interaction_learner.json") == checkpoint
        assert list(home._log().replay()) == frozen


@pytest.mark.parametrize("binding", ["missing", None, "", "not-a-digest", "0" * 64])
@pytest.mark.pool
def test_unbound_or_malformed_checkpoint_refuses_without_install(binding):
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, private):
        fresh = _semantic(brain)
        _completed(brain)
        _close_for_restart(brain)
        checkpoint = home.load_state("interaction_learner.json")
        if binding == "missing":
            checkpoint.pop("prefix_digest", None)
        else:
            checkpoint["prefix_digest"] = binding
        assert home.save_state("interaction_learner.json", checkpoint)
        restarted = _restart(private)
        assert restarted["fault"] is not None
        assert "checkpoint" in restarted["fault"]["cause"]
        assert "prefix" in restarted["fault"]["cause"]
        assert restarted["semantic"] == fresh
        assert restarted["hydrations"] == 0 and restarted["cursor"] is None
        assert home.load_state("interaction_learner.json") == checkpoint


@pytest.mark.parametrize("pending_tail", [False, True])
@pytest.mark.pool
def test_untampered_bound_prefix_and_pending_tail_restart_identically(pending_tail):
    with _private_brain() as (brain, private):
        expected = _completed(brain)
        if pending_tail:
            _teach_b(brain)
            expected = _semantic(brain)
            brain._r2_close("constructed_crash", RuntimeError("pending tail after checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
        assert restarted["hydrations"] == 1
        assert restarted["redo"] == int(pending_tail)
        # The first restart's close writes the recomputed full-prefix binding, including redo.
        again = _restart(private)
        assert again["fault"] is None and again["semantic"] == expected
        assert again["hydrations"] == 1 and again["redo"] == int(pending_tail)


@pytest.mark.pool
def test_empty_prefix_checkpoint_is_bound_and_restarts():
    with _private_brain() as (brain, private):
        expected = _semantic(brain)
        assert brain._flush_interaction_learner()
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
        assert restarted["cursor"] is None and restarted["redo"] == 0
