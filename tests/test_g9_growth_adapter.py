"""Bounded ordinary adapter controls, no G9 deciding campaign or calibration."""

import copy
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

SEED = 9900


@contextmanager
def _owner():
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home() as private:
        brain = make_brain(seed=SEED, persist=True)
        try:
            assert brain._r2_fault is None
            yield brain, private
        finally:
            if not getattr(brain, "_r2_draft_closed", False):
                brain.close()
            del brain
            gc.collect()


def _factory(brain, pid):
    return {
        "problem_id": pid,
        "labels": list(brain.brain._ixn.responses),
        "seed": SEED,
        "arm": "RSTDP_BRAIN",
        "population_origin": "born",
        "units": [],
        "birth_rows": [],
        "w_feed": [[] for _ in range(256)],
        "t_post_g": 15.0,
        "source_signs": {label: 1 for label in brain.brain._ixn.responses},
        "attachments": [],
        "out_eligible": [],
    }


def _bind(brain, pid="ii.xor", factory=None, projections=None):
    from snn_data_lake.growth.growth_loop import CanonicalGrowthAdapter
    from snn_data_lake.instruments.taught_problems import PROBLEMS

    problem = PROBLEMS[pid]
    schedule = tuple(
        tuple(tuple(episode) for episode in epoch) for epoch in problem.teach_epochs(SEED, 3)
    )
    if projections is None:
        projections = {
            "0": {
                "feed": [0.0] * 256,
                "return": [
                    {"source": label, "sign": 1, "weight": 0.2, "delay": 0}
                    for label in brain.brain._ixn.responses
                ],
            }
        }
    return CanonicalGrowthAdapter.for_brain(
        brain,
        problem,
        factory=_factory(brain, pid) if factory is None else factory,
        teach_schedule=schedule,
        projections=projections,
    )


def _batch(brain, adapter, *, birth=True):
    from snn_data_lake.growth.grown_code import GrownUnit
    from snn_data_lake.growth.growth_loop import BudgetValues, Ledger, make_admission
    from snn_data_lake.substrate import home

    ledger = Ledger(home.knowledge_path(), BudgetValues(52, None))
    pid = adapter.problem.id
    ledger.open_problem(pid, adapter.problem.n_teach_codes(SEED))
    unit = GrownUnit(0, (0.1,) * 52, 0.5, 12, 1, (0.0,) * len(brain.brain._ixn.responses))
    admission = make_admission(
        site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0), density_per_mm2=2.7e4, seed=SEED
    )
    rows, ticket = admission(
        (unit,),
        pid,
        ledger.next_batch_id(),
        resolve_problem=lambda p: p == pid,
        known_unit=ledger.known_unit,
    )
    if birth:
        ledger.record_units((unit,), rows, pid, ledger.next_batch_id())
        ledger.record_spend(pid, 1, 3)
    return ledger, (unit,), rows, ticket


def _restart(private, *, resume=False):
    remaining = (int(os.environ["SNDL_R2_CHILD_DEADLINE_NS"]) - time.time_ns()) / 1e9
    assert remaining > 0
    child = """
import gc,json,os
from snn_data_lake.engine.brain import make_brain
from snn_data_lake.substrate import home
from snn_data_lake.growth.growth_loop import CanonicalGrowthAdapter
from snn_data_lake.instruments.taught_problems import PROBLEMS,Turn
b=make_brain(seed=9900,persist=True)
try:
 assert b._r2_fault is None,b._r2_fault
 a=b._g9_actors['ii.xor'];f=a.factory;c=f['engine_binding']
 s=tuple(tuple(tuple(Turn(*t) for t in episode) for episode in epoch) for epoch in c['schedule'])
 adapter=CanonicalGrowthAdapter.for_brain(b,PROBLEMS['ii.xor'],factory=f,teach_schedule=s,projections=c['projections'])
 assert adapter.readiness()==(
  'retention_select','actual grown-source/admission binding','G6 deciding mask/progress carrier')
 if os.environ['G9_RESUME']=='1': adapter.attempt(1)
 receipt={'actor':a.state_dict(),'births':b._g9_birth_records()}
 print('G9 RESTART '+json.dumps(receipt,sort_keys=True),flush=True)
finally:
 b.close();del b;gc.collect()
print('G9 RESTART CLOSED',flush=True)
"""
    env = os.environ.copy()
    env.update(NORD_HOME=str(private), NORD_PERSIST="1", G9_RESUME="1" if resume else "0")
    result = subprocess.run(
        [sys.executable, "-B", "-X", "gil=0", "-X", "utf8", "-c", child],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=remaining,
    )
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith(b"G9 RESTART CLOSED")
    rows = [line for line in result.stdout.splitlines() if line.startswith(b"G9 RESTART {")]
    assert len(rows) == 1
    return json.loads(rows[0][len(b"G9 RESTART ") :])


def test_actual_adapter_admits_only_after_durable_birth_and_survives_restart():
    from snn_data_lake.growth.growth_loop import LedgerError
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _owner() as (brain, private):
        adapter = _bind(brain)
        assert adapter.segment == ()
        assert adapter.readiness() == (
            "retention_select",
            "actual grown-source/admission binding",
            "G6 deciding mask/progress carrier",
        )
        with pytest.raises(LedgerError, match="G6 deciding mask/progress carrier"):
            adapter.require_ready()
        _, units, rows, ticket = _batch(brain, adapter, birth=False)
        with pytest.raises(PosedRefused, match="G9_DURABLE_BIRTH_REQUIRED"):
            adapter.admit(units, rows, ticket)
        assert not ticket.consumed and adapter.segment == ()
        ledger, units, rows, ticket = _batch(brain, adapter)
        adapter.admit(units, rows, ticket)
        assert ticket.consumed and adapter.segment == units
        factory = brain._g9_actors[adapter.problem.id].factory
        assert factory["engine_binding"]["projections"]["0"]["feed"] == [0.0] * 256
        expected = brain._g9_actors[adapter.problem.id].state_dict()
        assert len(ledger.birth_records()) == 1
        brain.close()
        brain._r2_draft_closed = True
        restored = _restart(private)
        assert restored["actor"] == expected
        assert set(restored["births"]) == {"0"}


def test_attempt_uses_actual_writer_and_resumes_schedule_in_fresh_process():
    from snn_data_lake.growth.growth_loop import LedgerError

    with _owner() as (brain, private):
        adapter = _bind(brain)
        _, units, rows, ticket = _batch(brain, adapter)
        adapter.admit(units, rows, ticket)
        actor = brain._g9_actors[adapter.problem.id]
        before = actor.reader.state_dict()
        assert type(adapter.attempt(1)) is bool
        count = len(adapter._engine_binding["schedule"][0])
        assert actor.teaches == count
        assert actor.reader.state_dict() != before
        with pytest.raises(LedgerError, match="pre-frozen teaching schedule"):
            adapter.attempt(3)
        brain.close()
        brain._r2_draft_closed = True
        restored = _restart(private, resume=True)
        assert restored["actor"]["teaches"] == count * 2


def test_retention_consumer_restores_complete_state_and_forbids_early_or_repeat_final(monkeypatch):
    from snn_data_lake.growth.growth_loop import LedgerError
    from snn_data_lake.instruments.joint_mask_search import Meter, SelectionLeak
    from snn_data_lake.substrate import home

    with _owner() as (brain, _):
        old_adapter = _bind(brain, "ii.twin")
        _, units, rows, ticket = _batch(brain, old_adapter)
        old_adapter.admit(units, rows, ticket)
        old_adapter.attempt(1)
        adapter = _bind(brain)
        old = brain._g9_actors["ii.twin"]
        before, chat = old.state_dict(), brain.brain._ixn._snapshot()
        lake = Path(home.knowledge_path()).read_bytes()
        selected = adapter.read_retention("ii.twin")
        assert selected and old.state_dict() == before
        assert brain.brain._ixn._snapshot() == chat
        assert Path(home.knowledge_path()).read_bytes() == lake
        real_read = adapter._retention_read

        def failed_read(pid, episode):
            real_read(pid, episode)
            raise RuntimeError("planted retention interruption")

        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_retention_read", failed_read)
            with pytest.raises(RuntimeError, match="retention interruption"):
                adapter.read_retention("ii.twin")
        assert old.state_dict() == before
        assert brain.brain._ixn._snapshot() == chat
        assert Path(home.knowledge_path()).read_bytes() == lake
        meter = Meter(object())
        with pytest.raises(SelectionLeak, match="before the winner"):
            adapter.read_retention("ii.twin", meter=meter)
        meter.freeze((True,))
        with pytest.raises(LedgerError, match="G9_RETENTION_FINAL"):
            adapter.read_retention("ii.twin", meter=meter)
        assert old.state_dict() == before
        assert Path(home.knowledge_path()).read_bytes() == lake


def test_binding_drift_width_and_common_latch_refuse_before_effect():
    from snn_data_lake.growth.growth_loop import LedgerError
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _owner() as (brain, _):
        bad = {"0": {"feed": [0.0] * 52, "return": []}}
        seq = brain._r2_seq
        with pytest.raises(PosedRefused, match="G9_POOL256_WIDTH"):
            _bind(brain, projections=bad)
        assert brain._r2_seq == seq and brain._g9_actors == {}
        adapter = _bind(brain)
        actor = brain._g9_actors[adapter.problem.id]
        factory = copy.deepcopy(actor.factory)
        changed = copy.deepcopy(factory["engine_binding"]["projections"])
        changed["0"]["feed"][0] = 0.1
        with pytest.raises(PosedRefused, match="G9_ENGINE_CONFIG_DRIFT"):
            _bind(brain, factory=factory, projections=changed)
        seq = brain._r2_seq
        brain._r2_close("planted", RuntimeError("test"))
        with pytest.raises(LedgerError, match="common owner/latch"):
            adapter.attempt(1)
        assert brain._r2_seq == seq


def _planted_restore_failure(old, fail_on):
    """Fail only the `fail_on`-th rollback of `old`; the real restore still runs otherwise."""
    real, calls = old.rollback, []

    def rollback(state):
        calls.append(1)
        if len(calls) == fail_on:
            raise RuntimeError("planted final restore failure")
        return real(state)

    old.rollback = rollback
    return calls


def test_retention_restore_failure_closes_the_common_latch_and_keeps_both_errors():
    from snn_data_lake.growth.growth_loop import LedgerError

    with _owner() as (brain, _):
        _bind(brain, "ii.twin")
        adapter = _bind(brain)
        old = brain._g9_actors["ii.twin"]
        calls = _planted_restore_failure(old, 2)
        with pytest.raises(RuntimeError, match="planted final restore failure"):
            adapter.read_retention("ii.twin")
        assert len(calls) == 2
        assert brain._r2_fault is not None and "retention" in brain._r2_fault["phase"]
        assert not brain._flush_interaction_learner(), "checkpoint must refuse behind the latch"
        with pytest.raises(LedgerError, match="latch"):
            adapter.read_retention("ii.twin")


def test_retention_read_and_restore_failures_are_both_propagated(monkeypatch):
    with _owner() as (brain, _):
        _bind(brain, "ii.twin")
        adapter = _bind(brain)
        old = brain._g9_actors["ii.twin"]
        _planted_restore_failure(old, 2)

        def failed_read(pid, episode):
            raise ValueError("planted read failure")

        monkeypatch.setattr(adapter, "_retention_read", failed_read)
        with pytest.raises(BaseException) as caught:
            adapter.read_retention("ii.twin")
        text = repr(caught.value) + repr(getattr(caught.value, "exceptions", ""))
        text += repr(caught.value.__cause__) + repr(caught.value.__context__)
        assert "planted read failure" in text and "planted final restore failure" in text
        assert brain._r2_fault is not None


def _unspent_second_batch(brain, adapter):
    """Crash window: batch 1 was born in the lake, then the process died before its spend row."""
    from snn_data_lake.growth.grown_code import GrownUnit
    from snn_data_lake.growth.growth_loop import make_admission

    ledger, units, rows, ticket = _batch(brain, adapter, birth=False)
    ledger.record_units(units, rows, adapter.problem.id, ledger.next_batch_id())
    return ledger, units, rows, ticket, GrownUnit, make_admission


def test_admission_refuses_a_born_batch_the_reconciler_refuses():
    from snn_data_lake.growth.batch_reconciler import reconcile_batches
    from snn_data_lake.growth.growth_loop import LedgerError

    with _owner() as (brain, _):
        adapter = _bind(brain)
        ledger, units, rows, ticket, *_ = _unspent_second_batch(brain, adapter)
        with pytest.raises(LedgerError, match="no spend row"):
            reconcile_batches(ledger, adapter.problem.id)
        sequence = brain._r2_seq
        with pytest.raises(LedgerError, match="no spend row"):
            adapter.admit(units, rows, ticket)
        assert not ticket.consumed and adapter.segment == ()
        assert brain._r2_seq == sequence and brain._r2_fault is None
        # Positive control: once the spend row lands the SAME batch is admitted.
        ledger.record_spend(adapter.problem.id, 1, 3)
        adapter.admit(units, rows, ticket)
        assert ticket.consumed and adapter.segment == units


def test_resource_receipt_refuses_an_unreconciled_population(monkeypatch):
    from snn_data_lake.growth.growth_loop import LedgerError
    from snn_data_lake.readout.posed_problem import resource_receipt

    with _owner() as (brain, _):
        adapter = _bind(brain)
        ledger, units, rows, ticket, *_ = _unspent_second_batch(brain, adapter)
        # Plant the pre-fix condition: an admission that skipped reconciliation.
        monkeypatch.setattr(type(brain), "_g9_reconcile", lambda self, pids: None, raising=False)
        adapter.admit(units, rows, ticket)
        monkeypatch.undo()
        assert ledger.unsettled(adapter.problem.id) == 1
        with pytest.raises(LedgerError, match="no spend row"):
            resource_receipt({"engine": brain._g9_actors[adapter.problem.id]}, "engine", ledger)
        ledger.record_spend(adapter.problem.id, 1, 3)
        receipt = resource_receipt(
            {"engine": brain._g9_actors[adapter.problem.id]}, "engine", ledger
        )
        arm = receipt["arms"]["engine"]
        assert arm["U"] == 1 and arm["candidate_evaluations"] == 3


def test_restart_refuses_to_rearm_an_actor_whose_population_is_unreconciled():
    with _owner() as (brain, private):
        adapter = _bind(brain)
        ledger, units, rows, ticket = _batch(brain, adapter)
        adapter.admit(units, rows, ticket)
        # A later batch is born, then the process dies before its spend row lands.
        from snn_data_lake.growth.grown_code import GrownUnit
        from snn_data_lake.growth.growth_loop import make_admission

        second = GrownUnit(1, (0.1,) * 52, 0.5, 12, -1, (0.0,) * len(brain.brain._ixn.responses))
        admission = make_admission(
            site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0), density_per_mm2=2.7e4, seed=SEED
        )
        pid, batch = adapter.problem.id, ledger.next_batch_id()
        labels, _ = admission(
            (second,), pid, batch, resolve_problem=lambda p: p == pid, known_unit=ledger.known_unit
        )
        ledger.record_units((second,), labels, pid, batch)
        brain._r2_close("constructed_crash", RuntimeError("birth recorded, spend lost"))
        brain.close()
        brain._r2_draft_closed = True
        gc.collect()
        remaining = (int(os.environ["SNDL_R2_CHILD_DEADLINE_NS"]) - time.time_ns()) / 1e9
        assert remaining > 0
        child = (
            "import gc,json\n"
            "from snn_data_lake.engine.brain import make_brain\n"
            f"b=make_brain(seed={SEED},persist=True)\n"
            "try:\n"
            " r={'actors':sorted(b._g9_actors),'fault':b._r2_fault}\n"
            " print('A3 RESTART '+json.dumps(r))\n"
            "finally:\n"
            " b.close();del b;gc.collect()\n"
        )
        env = os.environ.copy()
        env.update(NORD_HOME=str(private), NORD_PERSIST="1")
        result = subprocess.run(
            [sys.executable, "-B", "-X", "gil=0", "-X", "utf8", "-c", child],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=remaining,
        )
        assert result.returncode == 0, result.stderr[-400:]
        line = [x for x in result.stdout.splitlines() if x.startswith(b"A3 RESTART ")][0]
        report = json.loads(line[len(b"A3 RESTART ") :])
        assert report["actors"] == [] and report["fault"]["phase"] == "reconcile"
        assert "no spend row" in report["fault"]["cause"]


def _frozen_meter():
    from snn_data_lake.instruments.joint_mask_search import Meter

    meter = Meter(object())
    meter.freeze((True,))
    return meter


def test_final_retention_is_refused_across_same_brain_rebind():
    """B1: the one-shot final judgment must not reopen when the adapter is rebound."""
    from snn_data_lake.growth.growth_loop import LedgerError
    from snn_data_lake.instruments.joint_mask_search import SelectionLeak

    with _owner() as (brain, _):
        _bind(brain, "ii.twin")
        adapter = _bind(brain)
        meter = _frozen_meter()
        old = brain._g9_actors["ii.twin"]
        before = old.state_dict()
        with pytest.raises(LedgerError, match="G9_RETENTION_FINAL"):
            adapter.read_retention("ii.twin", meter=meter)
        rebound = _bind(brain, factory=brain._g9_actors["ii.xor"].factory)
        with pytest.raises(LedgerError, match="G9_RETENTION_FINAL"):
            rebound.read_retention("ii.twin", meter=meter)
        # Positive controls: selecting reads stay available; an unfrozen winner keeps its name.
        assert rebound.read_retention("ii.twin")
        with pytest.raises(SelectionLeak, match="before the winner"):
            rebound.read_retention("ii.twin", meter=type(meter)(object()))
        assert old.state_dict() == before


def test_final_retention_is_refused_in_a_fresh_process():
    with _owner() as (brain, private):
        _bind(brain, "ii.twin")
        _bind(brain)
        brain.close()
        brain._r2_draft_closed = True
        remaining = (int(os.environ["SNDL_R2_CHILD_DEADLINE_NS"]) - time.time_ns()) / 1e9
        assert remaining > 0
        child = (
            "import gc,json\n"
            "from snn_data_lake.engine.brain import make_brain\n"
            "from snn_data_lake.growth.growth_loop import CanonicalGrowthAdapter\n"
            "from snn_data_lake.instruments.joint_mask_search import Meter\n"
            "from snn_data_lake.instruments.taught_problems import PROBLEMS,Turn\n"
            f"b=make_brain(seed={SEED},persist=True)\n"
            "try:\n"
            " a=b._g9_actors['ii.xor'];f=a.factory;c=f['engine_binding']\n"
            " s=tuple(tuple(tuple(Turn(*t) for t in e) for e in ep) for ep in c['schedule'])\n"
            " ad=CanonicalGrowthAdapter.for_brain(b,PROBLEMS['ii.xor'],factory=f,"
            "teach_schedule=s,projections=c['projections'])\n"
            " m=Meter(object());m.freeze((True,))\n"
            " out=[]\n"
            " for _ in range(2):\n"
            "  try: ad.read_retention('ii.twin',meter=m);out.append('accepted')\n"
            "  except Exception as e: out.append(str(e)[:40])\n"
            " print('B1 CHILD '+json.dumps(out))\n"
            "finally:\n"
            " b.close();del b;gc.collect()\n"
        )
        env = os.environ.copy()
        env.update(NORD_HOME=str(private), NORD_PERSIST="1")
        result = subprocess.run(
            [sys.executable, "-B", "-X", "gil=0", "-X", "utf8", "-c", child],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=remaining,
        )
        assert result.returncode == 0, result.stderr[-400:]
        line = [x for x in result.stdout.splitlines() if x.startswith(b"B1 CHILD ")][0]
        out = json.loads(line[len(b"B1 CHILD ") :])
        assert len(out) == 2 and all(x.startswith("G9_RETENTION_FINAL") for x in out), out
