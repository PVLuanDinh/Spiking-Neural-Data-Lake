"""A self-consistent checkpoint factory must still match its original common bind event."""

import copy
import json
from pathlib import Path

import pytest
from test_g9_growth_adapter import _batch, _factory, _owner
from test_g9_growth_adapter import _bind as _legacy_bind
from test_g9_retention_consumption import _bind, _events, _recipe


def test_valid_repacked_allowed_recipe_cannot_replace_the_covered_factory():
    from snn_data_lake.readout.posed_problem import (
        PosedActor,
        PosedRefused,
        pack_engine_binding,
        unpack_engine_binding,
    )

    with _owner() as (brain, _):
        adapter = _bind(brain)
        pid = adapter.problem.id
        brain._g9_check_protocol_prefix(brain.brain._ixn, brain._g9_actors, _events())
        saved = brain._g9_actors[pid].state_dict()
        config = unpack_engine_binding(saved["factory"]["engine_binding"])
        config["cycle"]["feedback_gate"] = "corrective"
        saved["factory"]["engine_binding"] = pack_engine_binding(config)
        # The replacement is structurally valid and its codec checksum is genuine.
        altered = PosedActor.restore(saved, brain._g9_birth_records())
        with pytest.raises(PosedRefused, match="G9_FACTORY_PREFIX_MISMATCH"):
            brain._g9_check_protocol_prefix(brain.brain._ixn, {pid: altered}, _events())


@pytest.mark.parametrize("key,value", [("feedback_gate", "other"), ("core_plasticity", False)])
def test_unknown_cycle_modes_refuse_the_bind_before_append(key, value):
    from snn_data_lake.readout.posed_problem import PosedRefused, pack_engine_binding
    from snn_data_lake.substrate import home

    with _owner() as (brain, _):
        factory = _factory(brain, "ii.xor")
        config = {
            "schedule": [[[["cat dog sun sky", "probe", "GREET"]]]],
            "projections": {},
            "cycle": {**_recipe(), key: value},
        }
        factory["engine_binding"] = pack_engine_binding(config)
        path = Path(home.knowledge_path())
        lake = path.read_bytes() if path.exists() else None
        with pytest.raises(PosedRefused, match="unsupported explicit"):
            brain.bind_part_d_reader(factory)
        assert (path.read_bytes() if path.exists() else None) == lake
        assert brain._g9_actors == {}
        assert brain._r2_fault is None


def test_complete_carrier_without_its_original_bind_or_with_a_duplicate_bind_is_refused():
    from snn_data_lake.readout.posed_problem import PosedActor, PosedRefused

    with _owner() as (brain, _):
        adapter = _bind(brain)
        pid = adapter.problem.id
        actor = PosedActor.restore(brain._g9_actors[pid].state_dict(), brain._g9_birth_records())
        events = _events()
        bind = next(
            e
            for e in events
            if e["kind"] == "posed_problem" and json.loads(e["source"])["op"] == "bind"
        )
        without_bind = [e for e in events if e is not bind]
        for prefix in (without_bind, [*events, copy.deepcopy(bind)]):
            with pytest.raises(PosedRefused, match="G9_FACTORY_PREFIX_MISMATCH"):
                brain._g9_check_protocol_prefix(brain.brain._ixn, {pid: actor}, prefix)


def test_legacy_binding_and_actual_admission_dynamic_fields_keep_the_original_factory():
    with _owner() as (brain, _):
        adapter = _legacy_bind(brain)
        pid = adapter.problem.id
        original = copy.deepcopy(brain._g9_actors[pid].factory)
        assert set(original["engine_binding"]) == {"schedule", "projections"}
        brain._g9_check_protocol_prefix(brain.brain._ixn, brain._g9_actors, _events())
        _, units, rows, ticket = _batch(brain, adapter)
        adapter.admit(units, rows, ticket)
        assert ticket.consumed
        assert brain._g9_actors[pid].factory["units"] != original["units"]
        assert brain._g9_actors[pid].factory["engine_binding"] == original["engine_binding"]
        brain._g9_check_protocol_prefix(brain.brain._ixn, brain._g9_actors, _events())
