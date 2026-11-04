"""Lossless frozen input source and corruption controls; no behavioral result."""

import base64
import copy
import hashlib
import json
import zlib

import pytest

from snn_data_lake.readout.posed_problem import (
    PosedRefused,
    pack_engine_binding,
    unpack_engine_binding,
)


def _config():
    return {
        "schedule": [[[["ram cup emu cod", "probe", "GREET"]]]] * 400,
        "projections": {"0": {"feed": [0.125] * 256, "return": []}},
        "cycle": {"schema": "g9.engine.cycle.v1", "feedback_gate": "corrective"},
    }


def test_compact_binding_roundtrips_exact_source_without_redrawing():
    config = _config()
    carrier = pack_engine_binding(config)
    assert unpack_engine_binding(json.loads(json.dumps(carrier))) == config
    assert len(json.dumps(carrier)) < len(json.dumps(config)) / 10
    assert carrier["cycle"] == config["cycle"]
    legacy = {"schedule": config["schedule"], "projections": config["projections"]}
    clone = unpack_engine_binding(legacy)
    clone["schedule"].clear()
    assert legacy["schedule"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_size", True),
        ("raw_size", 0),
        ("raw_size", 16 * 1024 * 1024 + 1),
        ("sha256", "0" * 64),
        ("payload", "not base64!"),
        ("schema", "future.codec"),
        ("cycle", {"schema": "g9.engine.cycle.v1", "feedback_gate": "changed"}),
    ],
)
def test_corrupt_binding_refuses(field, value):
    carrier = pack_engine_binding(_config())
    carrier[field] = value
    with pytest.raises(PosedRefused, match="G9_ENGINE_BINDING_CODEC"):
        unpack_engine_binding(carrier)


@pytest.mark.parametrize("change", ["truncated", "trailing", "noncanonical", "nan"])
def test_compressed_stream_and_canonical_source_are_strict(change):
    carrier = pack_engine_binding(_config())
    compressed = base64.b64decode(carrier["payload"])
    if change == "truncated":
        compressed = compressed[:-1]
    elif change == "trailing":
        compressed += zlib.compress(b"another stream")
    else:
        raw = (
            json.dumps(_config(), indent=2)
            if change == "noncanonical"
            else '{"cycle":{},"projections":{},"schedule":NaN}'
        ).encode()
        compressed = zlib.compress(raw)
        carrier["raw_size"] = len(raw)
        carrier["sha256"] = hashlib.sha256(raw).hexdigest()
    carrier["payload"] = base64.b64encode(compressed).decode()
    before = copy.deepcopy(carrier)
    with pytest.raises(PosedRefused, match="G9_ENGINE_BINDING_CODEC"):
        unpack_engine_binding(carrier)
    assert carrier == before
