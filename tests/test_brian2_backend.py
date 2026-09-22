import json
import sys
from types import MappingProxyType

import numpy as np
import pytest
import torch

from snn_data_lake.substrate.brian2_backend import (
    CONTRACT,
    BrianLIF,
    _parity_arm,
    brian_reference_trace,
    conjunct_c,
    conjunct_g,
    conjunct_i,
    conjunct_m,
    conjunct_p,
    conjunct_r,
    cost_report,
    fast_sigmoid,
    reds,
)


@pytest.mark.parametrize("reset", ["subtract", "zero", "none"])
def test_torch_bridge_matches_brian2_reference(reset):
    layer = BrianLIF(beta=0.8, threshold=1.0, reset_mechanism=reset)
    torch_membrane = torch.zeros(3)
    currents = ([0.8, 1.2, -0.1], [0.5, 0.4, 1.5], [0.2, 0.3, 0.1])
    brian_spikes, brian_membranes, version = brian_reference_trace(
        currents,
        np.zeros(3),
        beta=0.8,
        reset_mechanism=reset,
    )
    assert version == "2.10.1"
    assert not sys._is_gil_enabled()
    for index, current in enumerate(currents):
        torch_spikes, torch_membrane = layer(
            torch.tensor(current, dtype=torch.float32), torch_membrane
        )
        np.testing.assert_array_equal(torch_spikes.detach().numpy(), brian_spikes[index])
        np.testing.assert_allclose(
            torch_membrane.detach().numpy(), brian_membranes[index], atol=1e-6
        )


def test_fast_sigmoid_keeps_hard_spikes_and_supplies_gradient():
    current = torch.tensor([0.9, 1.1], requires_grad=True)
    spikes, _ = BrianLIF(beta=0.9, spike_grad=fast_sigmoid())(current)
    assert spikes.tolist() == [0.0, 1.0]
    spikes.sum().backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert torch.count_nonzero(current.grad) == 2


def test_unknown_reset_mechanism_is_rejected():
    with pytest.raises(ValueError, match="reset_mechanism"):
        BrianLIF(beta=0.9, reset_mechanism="mystery")


def test_contract_is_frozen_and_complete():
    assert isinstance(CONTRACT, MappingProxyType)
    assert set(CONTRACT) == {
        "equation",
        "threshold",
        "reset",
        "schedule",
        "dt",
        "units",
        "delays",
        "refractory",
        "adaptive_threshold",
        "precision",
        "learning",
        "rng",
    }
    with pytest.raises(TypeError):
        CONTRACT["delays"] = "supported"


def test_precision_contract_names_the_derived_bound():
    assert "2^-23" in CONTRACT["precision"]


def test_reference_runs_at_the_bridges_float32_parameters():
    arm = _parity_arm((0, "subtract", 0.95, 0.7))
    assert arm["beta_ref"] == float(np.float32(arm["beta"]))


def test_checkpoint_json_preserves_float32_state_and_parameters():
    layer = BrianLIF(0.95, threshold=0.7, reset_mechanism="none")
    layer(torch.tensor([0.123456789, -1.23456789, 0.70000001]))
    payload = json.loads(json.dumps(layer.checkpoint()))
    restored = BrianLIF(0.5)
    assert restored.restore(payload) is restored
    assert restored.checkpoint() == layer.checkpoint()
    assert torch.equal(restored.mem, layer.mem)
    current = torch.tensor([0.6, -0.4, 1.2])
    for actual, expected in zip(restored(current), layer(current), strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.slow
def test_conjunct_p():
    result = conjunct_p(seed=0)
    assert len(result["arms"]) == 27
    for arm in result["arms"]:
        assert arm["spike_mismatches_f32"] == arm["spike_mismatches_f64"] == 0
        assert arm["max_membrane_diff_f64"] <= 1e-9
    assert result["ok"], result


def test_conjunct_m():
    result = conjunct_m()
    assert result["ok"], result


@pytest.mark.slow
def test_conjunct_c():
    result = conjunct_c()
    assert result["ok"], result


def test_conjunct_g():
    result = conjunct_g()
    assert result["ok"], result


def test_conjunct_r():
    result = conjunct_r()
    assert result["ok"], result


def test_conjunct_i():
    result = conjunct_i()
    assert result["ok"], result


@pytest.fixture(scope="module")
def red_results():
    # Each mutant runs once; seven independent assertions share the measured table.
    return reds()


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R4", "R5", "R6", "R7"])
def test_red(name, red_results):
    assert red_results[name]["fires"], red_results[name]


def test_cost_report():
    result = cost_report()
    assert set(result) == {"32x16", "1x1"}
    for arm in result.values():
        assert set(arm) == {"calls", "wall_s", "payload_bytes", "peak_working_set_bytes"}
        assert arm["calls"] == 5
