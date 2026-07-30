"""A frozen LM artifact must say whether its training numbers were EVER recorded.

`provenance()` answers "unknown" for two different situations: a trainer run that left a field
empty, and a file written before the provenance schema existed. The first is fillable, the second
is permanent -- every PROVENANCE_KEY is a training-run value and none is derivable from the
weights, so without the original checkpoint a pre-schema artifact can never be re-stamped
truthfully. `provenance_status()` draws that line, and these tests hold it.
"""

import os

import pytest

from snn_data_lake.reasoning.ar_spiking_lm_serve import (
    DEFAULT_ARTIFACT,
    PROVENANCE_KEYS,
    SpikingLMServe,
)


def _serve_with_meta(meta):
    """A serve whose load succeeded, with the metadata under test. Exercises the real method -- only
    the load is bypassed, because the point is the metadata branch, not the numpy reader."""
    srv = SpikingLMServe.__new__(SpikingLMServe)
    srv._ok = True
    srv.meta = meta
    return srv


def test_status_is_none_when_no_artifact_loaded():
    srv = SpikingLMServe(path=os.path.join("nonexistent", "no_such_artifact.npz"))
    assert srv.available() is False
    assert srv.provenance_status() is None, "no artifact must report None, never a status"
    assert srv.provenance() is None, "honest-unavailable, same contract as logprob"


def test_recorded_when_the_block_is_present():
    srv = _serve_with_meta({"provenance": {"checkpoint_step": 1399, "seed": 0}})
    assert srv.provenance_status() == "recorded"


def test_unrecorded_when_the_block_is_absent():
    srv = _serve_with_meta({"format": "spiking_lm.v1", "trained": True})
    assert srv.provenance_status() == "unrecorded"


def test_empty_block_is_unrecorded_not_recorded():
    """An empty dict is not a recording. Calling it "recorded" would claim the run testified."""
    assert _serve_with_meta({"provenance": {}}).provenance_status() == "unrecorded"


def test_a_filled_field_and_a_missing_field_are_both_unknown_but_the_status_differs():
    """The conflation this method exists to break: same provenance() output, different truth.

    A recorded run that left `seed` empty and a pre-schema file both read "unknown" for `seed`. Only
    the second can never be fixed for that file."""
    recorded = _serve_with_meta({"provenance": {"checkpoint_step": 1399, "seed": None}})
    pre_schema = _serve_with_meta({"trained": True})
    assert recorded.provenance()["seed"] == "unknown"
    assert pre_schema.provenance()["seed"] == "unknown"
    assert recorded.provenance_status() != pre_schema.provenance_status()


def test_provenance_still_returns_every_key_either_way():
    """The pre-existing contract must survive: every schema key present, no crash, no raise."""
    for meta in ({"trained": True}, {"provenance": {"checkpoint_step": 7}}):
        prov = _serve_with_meta(meta).provenance()
        assert set(prov) == set(PROVENANCE_KEYS)


def test_shipped_artifact_is_reported_as_unrecorded_not_silently_passed():
    """The claim this task exists for. models/ is gitignored, so skip when the artifact is absent --
    but when it IS present, it must not quietly look provenanced."""
    if not os.path.exists(DEFAULT_ARTIFACT):
        pytest.skip("models/ar_spiking_lm.npz absent (gitignored); nothing to check")
    srv = SpikingLMServe()
    if not srv.available():
        pytest.skip("artifact present but not loadable in this environment")
    assert srv.provenance_status() == "unrecorded", (
        "the shipped real-scale LM predates the provenance schema; if this ever flips to "
        "'recorded', the artifact was replaced and DEPTH_CEILINGS.md's note must be revisited"
    )
    prov = srv.provenance()
    assert all(prov[k] == "unknown" for k in PROVENANCE_KEYS), (
        "an unrecorded artifact must not report any training number as fact"
    )
