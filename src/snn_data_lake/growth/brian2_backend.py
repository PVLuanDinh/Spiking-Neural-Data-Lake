"""Brian2 is the canonical LIF simulator; Torch is the differentiable training bridge.

Brian2 deliberately models neurons with equations and event thresholds instead of exposing a
``torch.nn.Module``.  The project still needs Torch autograd for its existing surrogate-gradient
trainers, so :class:`BrianLIF` implements the same discrete equation with Torch tensors while
``brian_reference_trace`` executes it through a real Brian2 ``NeuronGroup`` in an isolated
subprocess. Isolation matters: Brian2 2.10.1's compiled dynamic-array extension re-enables the GIL
on CPython 3.14t, while this project's main process has a hard free-threaded contract. The
self-check compares both paths and asserts that the parent process keeps the GIL disabled.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from threading import Lock
from types import MappingProxyType

import numpy as np
import torch
import torch.nn as nn

SpikeGradient = Callable[[torch.Tensor], torch.Tensor]
RESET_MECHANISMS = {"subtract", "zero", "none"}
MODULE_NAME = "snn_data_lake.growth.brian2_backend"
MAX_REFERENCE_CELLS = 2_000_000
_REFERENCE_LAUNCHES = 0
_REFERENCE_LOCK = Lock()
CONTRACT = MappingProxyType(
    {
        "equation": "v[t] = beta*v[t-1] + I[t]",
        "threshold": "spike iff v >= theta; equality spikes",
        "reset": "same step, after the threshold test; subtract | zero | none",
        "schedule": "integrate, threshold, reset -- within one step; Brian2 "
        "run_regularly(..., when='groups') then threshold/reset",
        "dt": "one step = 1 ms in the reference; the bridge is unitless per step",
        "units": "v, I, theta dimensionless",
        "delays": "none",
        "refractory": "none",
        "adaptive_threshold": "none",
        "precision": "bridge computes in the caller's dtype (float32 at every migrated call site); "
        "beta and theta are stored float32 and the float64 reference is given those stored values; "
        "on the registered seeded parity traces only (T <= 64): spikes exact, membranes within "
        "T * 2^-23 * max(1, max|v_reference|) for float32 and within 1e-9 for float64 -- acceptance "
        "tolerances for the measured traces, not universal guarantees: threshold-adjacent rounding "
        "can flip a spike and its reset (tests/test_brian2_backend.py::test_threshold_rounding_limit) "
        "and cancellation can exceed the float32 tolerance",
        "learning": "the bridge is a LAYER: it computes a surrogate derivative only when "
        "a caller runs autograd and never steps an optimizer; governed learning rules are untouched",
        "rng": "none owned; deterministic given inputs",
    }
)
MIGRATED_CALL_SITES = (
    "snn_data_lake.training.trainer",
    "snn_data_lake.training.offload",
    "snn_data_lake.training.mnist",
    "snn_data_lake.training.spiking_lm",
    "snn_data_lake.instruments.deep_stdp",
    "snn_data_lake.instruments.snn_mnist_stdp",
    "snn_data_lake.reasoning.developmental_loop",
    "snn_data_lake.reasoning.snn_dreamer_6dof",
    "snn_data_lake.reasoning.snn_motor_control",
    "snn_data_lake.reasoning.snn_ssm_shd",
    "snn_data_lake.reasoning.snn_world_model_rl",
)
# Section 1.3 actually enumerates twenty classes, plus KernelTick; retain every name.
CUSTOM_KERNELS = (
    "snn_data_lake.growth.lif_neuron::LIFNeuron",
    "snn_data_lake.growth.alif_neuron::ALIFNeuron",
    "snn_data_lake.growth.adex_neuron::AdEx",
    "snn_data_lake.growth.autapse_neuron::AutapseNeuron",
    "snn_data_lake.growth.dendritic_neuron::DendriticNeuron",
    "snn_data_lake.growth.srm_neuron::SRMNeuron",
    "snn_data_lake.growth.srm_neuron::IFNeuron",
    "snn_data_lake.plasticity.eprop_kernel::EpropLIFBase",
    "snn_data_lake.growth.heterogeneous_microcircuit::_LIFCell",
    "snn_data_lake.growth.heterogeneous_microcircuit::_AdExCell",
    "snn_data_lake.growth.neuron_zoo::_AdExCell",
    "snn_data_lake.growth.neuron_zoo::_IzhikevichCell",
    "snn_data_lake.growth.neuron_zoo::NeuronZoo",
    "snn_data_lake.faculties.spiking_pid::_LIFPopulation",
    "snn_data_lake.plasticity.r_stdp::RSTDPNeuron",
    "snn_data_lake.plasticity.bcm_plasticity::BCMNeuron",
    "snn_data_lake.plasticity.homeostatic_scaling::HomeostaticNeuron",
    "snn_data_lake.plasticity.oja_pca::OjaNeuron",
    "snn_data_lake.growth.basic_components::_DemoLIFLayer",
    "snn_data_lake.faculties.gwt_deep::LIF",
    "snn_data_lake.reservoir.kernel_tick::KernelTick",
)


class _FastSigmoid(torch.autograd.Function):
    """Heaviside forward pass with the SuperSpike fast-sigmoid derivative."""

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.slope = slope
        return (membrane >= 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        grad = grad_output / (ctx.slope * membrane.abs() + 1.0).square()
        return grad, None


def fast_sigmoid(slope: float = 25.0) -> SpikeGradient:
    """Return the surrogate derivative used by the existing Torch training paths."""

    def spike(membrane: torch.Tensor) -> torch.Tensor:
        return _FastSigmoid.apply(membrane, float(slope))

    return spike


class BrianLIF(nn.Module):
    """Differentiable Torch bridge for the Brian2 discrete LIF equation.

    Brian2 applies threshold/reset in the current timestep. This matches the project's exported
    serving formats and closes the former one-step reset mismatch. ``reset_mechanism`` accepts
    ``subtract``, ``zero``, or ``none``.
    """

    def __init__(
        self,
        beta: float | torch.Tensor,
        threshold: float | torch.Tensor = 1.0,
        spike_grad: SpikeGradient | None = None,
        reset_mechanism: str = "subtract",
        **unsupported,
    ) -> None:
        super().__init__()
        for key in unsupported:
            if key in {"refractory", "delay", "dt", "adaptive_threshold"}:
                entry = "delays" if key == "delay" else key
                raise NotImplementedError(f"{key} unsupported; {entry}: {CONTRACT[entry]}")
        if unsupported:
            raise TypeError(f"unexpected keyword argument: {next(iter(unsupported))!r}")
        if reset_mechanism not in RESET_MECHANISMS:
            expected = ", ".join(sorted(RESET_MECHANISMS))
            raise ValueError(f"reset_mechanism must be one of {expected}; got {reset_mechanism!r}")
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float32))
        self.register_buffer("threshold", torch.as_tensor(threshold, dtype=torch.float32))
        self.register_buffer("mem", torch.zeros(0), persistent=False)
        self.spike_grad = spike_grad or fast_sigmoid()
        self.reset_mechanism = reset_mechanism

    def reset_mem(self) -> torch.Tensor:
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def init_leaky(self) -> torch.Tensor:
        """Compatibility spelling used by the project's existing recurrent loops."""

        return self.reset_mem()

    def checkpoint(self) -> dict:
        """JSON-safe state; tolist preserves every float32 value, including shaped membranes."""
        return {
            "beta": float(self.beta),
            "threshold": float(self.threshold),
            "reset_mechanism": self.reset_mechanism,
            "mem": self.mem.detach().to(dtype=torch.float32).cpu().tolist(),
        }

    def restore(self, payload):
        for key in ("beta", "threshold", "reset_mechanism", "mem"):
            if key not in payload:
                raise KeyError(key)
        if payload["reset_mechanism"] not in RESET_MECHANISMS:
            raise ValueError(f"unknown reset_mechanism: {payload['reset_mechanism']!r}")
        self.beta = torch.tensor(payload["beta"], dtype=torch.float32)
        self.threshold = torch.tensor(payload["threshold"], dtype=torch.float32)
        self.reset_mechanism = payload["reset_mechanism"]
        self.mem = torch.tensor(payload["mem"], dtype=torch.float32)
        return self

    def forward(
        self, input_current: torch.Tensor, membrane: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = self.mem
        if membrane.shape != input_current.shape:
            membrane = torch.zeros_like(input_current)
        membrane = membrane.to(dtype=input_current.dtype)

        beta = self.beta.clamp(0, 1).to(dtype=input_current.dtype)
        threshold = self.threshold.to(dtype=input_current.dtype)
        integrated = beta * membrane + input_current
        spikes = self.spike_grad(integrated - threshold)
        reset = spikes.detach()
        if self.reset_mechanism == "subtract":
            next_membrane = integrated - reset * threshold
        elif self.reset_mechanism == "zero":
            next_membrane = integrated * (1.0 - reset)
        else:
            next_membrane = integrated

        self.mem = next_membrane
        return spikes, next_membrane


def brian_reference_step(
    input_current,
    membrane,
    *,
    beta: float,
    threshold: float = 1.0,
    reset_mechanism: str = "subtract",
) -> tuple[np.ndarray, np.ndarray]:
    """Execute one isolated Brian2 LIF step and return ``(spikes, membrane)`` arrays."""

    spikes, membranes, _ = brian_reference_trace(
        [input_current],
        membrane,
        beta=beta,
        threshold=threshold,
        reset_mechanism=reset_mechanism,
    )
    return spikes[0], membranes[0]


def brian_reference_trace(
    input_currents,
    membrane,
    *,
    beta: float,
    threshold: float = 1.0,
    reset_mechanism: str = "subtract",
    timeout_s: float = 120.0,
    sleep_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run a current trace in an isolated Brian2 process.

    Brian2's cp314t wheel is installable, but importing its Cython dynamic-array extension enables
    the GIL. Keeping that import in the child preserves the parent brain's free-threaded runtime.

    ``sleep_s`` is a TEST-ONLY child delay for the timeout negative control.
    """
    try:
        result = _reference_result(
            input_currents,
            membrane,
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            timeout_s=timeout_s,
            sleep_s=sleep_s,
        )
        return (
            np.asarray(result["spikes"], dtype=np.float64),
            np.asarray(result["membranes"], dtype=np.float64),
            str(result["version"]),
        )
    finally:
        _assert_free_threaded()


def _assert_free_threaded() -> None:
    if sys._is_gil_enabled():
        raise RuntimeError("Brian2 reference enabled the GIL in the parent process")


def _reference_result(
    input_currents,
    membrane,
    *,
    beta,
    threshold=1.0,
    reset_mechanism="subtract",
    timeout_s=120.0,
    sleep_s=None,
) -> dict:
    global _REFERENCE_LAUNCHES
    if reset_mechanism not in RESET_MECHANISMS:
        raise ValueError(f"unknown reset mechanism: {reset_mechanism!r}")
    currents = np.asarray(input_currents, dtype=np.float64)
    previous = np.asarray(membrane, dtype=np.float64)
    if currents.ndim < 1 or currents.shape[1:] != previous.shape:
        raise ValueError("every input current must have the same shape as membrane")
    if currents.size > MAX_REFERENCE_CELLS:
        raise ValueError(f"steps * cells exceeds MAX_REFERENCE_CELLS={MAX_REFERENCE_CELLS}")
    payload = {
        "currents": currents.tolist(),
        "membrane": previous.tolist(),
        "beta": float(beta),
        "threshold": float(threshold),
        "reset_mechanism": reset_mechanism,
    }
    if sleep_s is not None:
        payload["sleep_s"] = sleep_s
    encoded = json.dumps(payload)
    with _REFERENCE_LOCK:
        _REFERENCE_LAUNCHES += 1
    try:
        completed = subprocess.run(
            [sys.executable, "-m", MODULE_NAME, "--reference-trace"],
            input=encoded,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Brian2 reference exceeded timeout_s={timeout_s}") from exc
    finally:
        _assert_free_threaded()
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()
        raise RuntimeError(f"Brian2 reference process failed: {' '.join(detail)}")
    marker = "BRIAN2_RESULT="
    line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        raise RuntimeError("Brian2 reference process returned no result")
    result = json.loads(line[len(marker) :])
    result["payload_bytes"] = len(encoded.encode("utf-8"))
    return result


def _run_reference_child(payload) -> dict:
    """Child-only Brian2 runner. Never call this in the long-lived brain process."""

    # TEST-ONLY payload field; never alters the Brian2 model or schedule.
    if "sleep_s" in payload:
        time.sleep(payload["sleep_s"])
    import brian2 as b2

    currents = np.asarray(payload["currents"], dtype=np.float64)
    previous = np.asarray(payload["membrane"], dtype=np.float64)
    shape = previous.shape
    flat_currents = currents.reshape(currents.shape[0], -1)
    previous = previous.reshape(-1)
    reset_mechanism = payload["reset_mechanism"]
    clock = b2.Clock(dt=1 * b2.ms)
    group = b2.NeuronGroup(
        previous.size,
        """
        v : 1
        drive : 1
        decay : 1 (constant)
        threshold_value : 1 (constant)
        """,
        threshold="v >= threshold_value",
        reset=(
            "v -= threshold_value"
            if reset_mechanism == "subtract"
            else "v = 0"
            if reset_mechanism == "zero"
            else ""
        ),
        clock=clock,
    )
    group.v = previous
    group.decay = float(np.clip(payload["beta"], 0.0, 1.0))
    group.threshold_value = float(payload["threshold"])
    step = group.run_regularly("v = decay * v + drive", dt=clock.dt, when="groups")
    monitor = b2.SpikeMonitor(group)
    network = b2.Network(group, step, monitor)

    old_target = b2.prefs.codegen.target
    spike_trace = []
    membrane_trace = []
    previous_counts = np.zeros(previous.size, dtype=np.int64)
    try:
        b2.prefs.codegen.target = "numpy"
        for current in flat_currents:
            group.drive = current
            network.run(clock.dt)
            counts = np.asarray(monitor.count).copy()
            spike_trace.append((counts > previous_counts).astype(np.float64).reshape(shape))
            membrane_trace.append(np.asarray(group.v[:]).reshape(shape).copy())
            previous_counts = counts
    finally:
        b2.prefs.codegen.target = old_target
    try:
        import psutil
    except ImportError:
        peak_working_set = "ABSENT"
    else:
        peak_working_set = getattr(psutil.Process().memory_info(), "peak_wset", "ABSENT")
    if sys.platform == "linux":
        import resource

        # psutil exposes peak_wset on Windows only; Linux reports this child's peak RSS (KiB -> bytes).
        peak_working_set = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "spikes": np.asarray(spike_trace).tolist(),
        "membranes": np.asarray(membrane_trace).tolist(),
        "version": b2.__version__,
        "peak_working_set_bytes": peak_working_set,
    }


def _seeded_trace(seed=0, threshold=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 2, (32, 16)), rng.uniform(0, threshold, 16)


def _torch_trace(layer, currents, membrane=None, dtype=torch.float32):
    spikes, membranes = [], []
    if membrane is not None:
        layer.mem = torch.tensor(membrane, dtype=dtype)
    with torch.no_grad():
        for current in currents:
            spike, mem = layer(torch.tensor(current, dtype=dtype))
            spikes.append(spike.cpu().numpy().copy())
            membranes.append(mem.cpu().numpy().copy())
    return np.asarray(spikes), np.asarray(membranes)


def _parity_arm(arm):
    seed, reset, beta, threshold = arm
    currents, initial = _seeded_trace(seed, threshold)
    actual_s_f32, actual_v_f32 = _torch_trace(
        BrianLIF(beta, threshold, reset_mechanism=reset), currents, initial, dtype=torch.float32
    )
    actual_s_f64, actual_v_f64 = _torch_trace(
        BrianLIF(beta, threshold, reset_mechanism=reset), currents, initial, dtype=torch.float64
    )
    expected_s, expected_v, _ = brian_reference_trace(
        currents,
        initial,
        beta=float(np.float32(beta)),
        threshold=float(np.float32(threshold)),
        reset_mechanism=reset,
    )
    max_v_reference = float(np.max(np.abs(expected_v)))
    bound_f32 = len(currents) * 2**-23 * max(1.0, max_v_reference)
    diff_f32 = float(np.max(np.abs(actual_v_f32 - expected_v)))
    diff_f64 = float(np.max(np.abs(actual_v_f64 - expected_v)))
    mismatches_f32 = int(np.count_nonzero(actual_s_f32 != expected_s))
    mismatches_f64 = int(np.count_nonzero(actual_s_f64 != expected_s))
    return {
        "reset": reset,
        "beta": beta,
        "threshold": threshold,
        "beta_ref": float(np.float32(beta)),
        "threshold_ref": float(np.float32(threshold)),
        "max_v_reference": max_v_reference,
        "bound_f32": bound_f32,
        "max_membrane_diff_f32": diff_f32,
        "spike_mismatches_f32": mismatches_f32,
        "max_membrane_diff_f64": diff_f64,
        "spike_mismatches_f64": mismatches_f64,
        "ok": mismatches_f32 == mismatches_f64 == 0 and diff_f32 <= bound_f32 and diff_f64 <= 1e-9,
    }


def conjunct_p(seed=0) -> dict:
    """Twenty-seven independent arms; bounded concurrency amortizes child startup."""
    arms = [
        (seed, r, b, t)
        for r in ("subtract", "zero", "none")
        for b in (0.5, 0.8, 0.95)
        for t in (0.7, 1.0, 1.5)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_parity_arm, arms))
    return {
        "ok": all(row["ok"] for row in results),
        "seed": seed,
        "max_membrane_diff_f32": max(row["max_membrane_diff_f32"] for row in results),
        "max_bound_f32": max(row["bound_f32"] for row in results),
        "max_membrane_diff_f64": max(row["max_membrane_diff_f64"] for row in results),
        "spike_mismatches": sum(
            row["spike_mismatches_f32"] + row["spike_mismatches_f64"] for row in results
        ),
        "arms": results,
    }


def conjunct_m() -> dict:
    from snn_data_lake.training.trainer import SpikingReadout, _generator, rate_encode

    currents, spikes = [], []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(0)
        model = SpikingReadout(beta=0.9)
        batch = torch.randint(0, 2, (8, model.fc.in_features)).float()
        encoded = rate_encode(batch, 25, _generator("cpu", 0), "cpu")
        current_hook = model.fc.register_forward_hook(
            lambda _m, _args, out: currents.append(out.detach().numpy().copy())
        )
        spike_hook = model.lif.register_forward_hook(
            lambda _m, _args, out: spikes.append(out[0].detach().numpy().copy())
        )
        try:
            summed = model(encoded).numpy()
        finally:
            current_hook.remove()
            spike_hook.remove()
    expected, _, _ = brian_reference_trace(currents, np.zeros_like(currents[0]), beta=0.9)
    mismatches = int(np.count_nonzero(np.asarray(spikes) != expected))
    sums_equal = np.array_equal(summed, expected.sum(axis=0))
    return {
        "ok": mismatches == 0 and sums_equal,
        "steps": len(spikes),
        "batch": len(batch),
        "spike_mismatches": mismatches,
        "sums_equal": sums_equal,
    }


def _continuation_child(role, path):
    """Each role is invoked in its own interpreter; only JSON crosses processes."""
    path = Path(path)
    currents, initial = _seeded_trace()
    layer = BrianLIF(0.8)
    if role == "A":
        _torch_trace(layer, currents[:12], initial)
        path.write_text(json.dumps(layer.checkpoint()), encoding="utf-8")
        return
    if role in {"B", "B-omit"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if role == "B-omit":
            del payload["mem"]
            layer = _MutantMissingMem(0.8)
        layer.restore(payload)
        spikes, membranes = _torch_trace(layer, currents[12:])
    elif role == "C":
        spikes, membranes = _torch_trace(layer, currents, initial)
    else:
        raise ValueError(f"unknown continuation role: {role}")
    path.with_suffix(f".{role}.json").write_text(
        json.dumps({"spikes": spikes.tolist(), "membranes": membranes.tolist()}),
        encoding="utf-8",
    )


def _continuation_trial(role="B"):
    with tempfile.TemporaryDirectory(prefix="brian2-continuation-") as directory:
        path = Path(directory) / "checkpoint.json"
        for child_role in ("A", role, "C"):
            subprocess.run(
                [sys.executable, "-m", MODULE_NAME, "--continuation", child_role, str(path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        resumed = json.loads(path.with_suffix(f".{role}.json").read_text(encoding="utf-8"))
        full = json.loads(path.with_suffix(".C.json").read_text(encoding="utf-8"))
    differences = {}
    for key in ("spikes", "membranes"):
        actual = np.asarray(resumed[key], dtype=np.float32)
        expected = np.asarray(full[key], dtype=np.float32)[12:]
        differences[key + "_equal"] = bool(np.array_equal(actual, expected))
        differences[key + "_mismatches"] = int(np.count_nonzero(actual != expected))
    return {
        "ok": differences["spikes_equal"] and differences["membranes_equal"],
        "processes": 3,
        "steps": len(resumed["spikes"]),
        **differences,
    }


def conjunct_c() -> dict:
    return _continuation_trial()


def conjunct_g() -> dict:
    parent_gil = sys._is_gil_enabled()
    child = subprocess.run(
        [sys.executable, "-c", "import brian2, sys; print(sys._is_gil_enabled())"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    child_gil = child.stdout.strip() == "True"
    return {
        "ok": not parent_gil and child_gil,
        "parent_gil": parent_gil,
        "child_gil": child_gil,
        "child_returncode": child.returncode,
    }


def _refused(call, exception, name):
    try:
        call()
    except exception as exc:
        return {"refused": name in str(exc), "error": str(exc)}
    return {"refused": False, "error": None}


def _resource_refusals():
    before = _REFERENCE_LAUNCHES
    size = _refused(
        lambda: brian_reference_trace(
            np.zeros((1, MAX_REFERENCE_CELLS + 1)), np.zeros(MAX_REFERENCE_CELLS + 1), beta=0.8
        ),
        ValueError,
        "MAX_REFERENCE_CELLS",
    )
    launches = _REFERENCE_LAUNCHES - before
    started = time.perf_counter()
    slow = _refused(
        lambda: brian_reference_trace([[0.0]], [0.0], beta=0.8, sleep_s=10, timeout_s=0.5),
        TimeoutError,
        "0.5",
    )
    elapsed = time.perf_counter() - started
    return {
        "ok": size["refused"] and launches == 0 and slow["refused"],
        "oversize": size,
        "oversize_launches": launches,
        "slow": slow,
        "timeout_elapsed_s": elapsed,
        "requested_cells": MAX_REFERENCE_CELLS + 1,
    }


def conjunct_r() -> dict:
    keywords = {
        key: _refused(lambda key=key: BrianLIF(0.8, **{key: 2}), NotImplementedError, key)
        for key in ("refractory", "delay", "dt", "adaptive_threshold")
    }
    unknown = _refused(lambda: BrianLIF(0.8, typo=2), TypeError, "typo")
    missing = {}
    for key in ("beta", "threshold", "reset_mechanism", "mem"):
        payload = BrianLIF(0.8).checkpoint()
        del payload[key]
        missing[key] = _refused(
            lambda payload=payload: BrianLIF(0.8).restore(payload), KeyError, key
        )
    invalid = BrianLIF(0.8).checkpoint()
    invalid["reset_mechanism"] = "invalid"
    reset = _refused(lambda: BrianLIF(0.8).restore(invalid), ValueError, "reset_mechanism")
    resource = _resource_refusals()
    return {
        "ok": all(x["refused"] for x in (*keywords.values(), *missing.values(), unknown, reset))
        and resource["ok"],
        "keywords": keywords,
        "unknown_keyword": unknown,
        "missing_keys": missing,
        "unknown_reset": reset,
        "resources": resource,
    }


def conjunct_i() -> dict:
    """Parse import statements, not comments or historic mentions of snntorch."""
    root = Path(__file__).resolve().parents[2]
    bridge, retired = set(), set()
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                names = [base, *(base + "." + alias.name for alias in node.names)]
            if MODULE_NAME in names:
                bridge.add(module)
            if any(name == "snntorch" or name.startswith("snntorch.") for name in names):
                retired.add(module)
    missing = []
    for name in CUSTOM_KERNELS:
        module, class_name = name.split("::")
        try:
            resolved = getattr(importlib.import_module(module), class_name)
            if not isinstance(resolved, type):
                missing.append(name)
        except (ImportError, AttributeError) as exc:
            missing.append(f"{name}: {exc}")
    expected = set(MIGRATED_CALL_SITES)
    return {
        "ok": bridge == expected and not retired and not missing,
        "call_sites": sorted(bridge),
        "unexpected": sorted(bridge - expected),
        "missing_call_sites": sorted(expected - bridge),
        "snntorch": sorted(retired),
        "custom_kernel_count": len(CUSTOM_KERNELS),
        "missing_kernels": missing,
    }


def cost_report() -> dict:
    """Five-call medians using the same reference launch path; no acceptance bound."""
    report = {}
    for steps, cells in ((32, 16), (1, 1)):
        times, sizes, peaks = [], [], []
        for _ in range(5):
            started = time.perf_counter()
            result = _reference_result(np.zeros((steps, cells)), np.zeros(cells), beta=0.8)
            times.append(time.perf_counter() - started)
            sizes.append(result["payload_bytes"])
            peaks.append(result["peak_working_set_bytes"])
        report[f"{steps}x{cells}"] = {
            "calls": len(times),
            "wall_s": median(times),
            "payload_bytes": median(sizes),
            "peak_working_set_bytes": "ABSENT" if "ABSENT" in peaks else median(peaks),
        }
    return report


class _MutantThreshold(BrianLIF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spike_grad = lambda x: (x > 0).to(x.dtype)


class _MutantLateReset(BrianLIF):
    def forward(self, input_current, membrane=None):
        membrane = self.mem if membrane is None else membrane
        previous_spike = (membrane >= self.threshold).to(input_current.dtype)
        self.mem = self.beta * membrane + input_current - previous_spike * self.threshold
        return (self.mem >= self.threshold).to(input_current.dtype), self.mem


class _MutantZeroReset(BrianLIF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_mechanism = "zero"


class _MutantSchedule(BrianLIF):
    def forward(self, input_current, membrane=None):
        membrane = self.mem if membrane is None else membrane
        spikes = (membrane >= self.threshold).to(input_current.dtype)
        self.mem = self.beta * membrane + input_current - spikes * self.threshold
        return spikes, self.mem


class _MutantMissingMem(BrianLIF):
    def restore(self, payload):
        return super().restore({"mem": [], **payload})


class _MutantSilentFallback(BrianLIF):
    def __init__(self, *args, **kwargs):
        kwargs.pop("refractory", None)
        super().__init__(*args, **kwargs)


def reds() -> dict:
    """Constructed implementations, never semantic switches on the shipped layer."""
    currents, initial = np.array([[1.0], [0.5], [1.4], [0.1]]), np.zeros(1)
    expected_s, expected_v, _ = brian_reference_trace(
        currents, initial, beta=float(np.float32(0.8)), threshold=float(np.float32(1.0))
    )

    def compare(layer):
        spikes, membranes = _torch_trace(layer, currents, initial)
        mismatch = int(np.count_nonzero(spikes != expected_s))
        diff = float(np.max(np.abs(membranes - expected_v)))
        bound = len(currents) * 2**-23 * max(1.0, float(np.max(np.abs(expected_v))))
        return {
            "ok": mismatch == 0 and diff <= bound,
            "spike_mismatches": mismatch,
            "max_membrane_diff": diff,
            "second_step_membrane_diff": float(np.max(np.abs(membranes[1] - expected_v[1]))),
        }

    shipped = compare(BrianLIF(0.8))
    threshold = compare(_MutantThreshold(0.8))
    late, zero = compare(_MutantLateReset(0.8)), compare(_MutantZeroReset(0.8))
    schedule = compare(_MutantSchedule(0.8))
    _assert_free_threaded()
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from {MODULE_NAME} import _assert_free_threaded; "
            "import brian2; _assert_free_threaded()",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    payload = BrianLIF(0.8).checkpoint()
    del payload["mem"]
    omitted = _refused(lambda: BrianLIF(0.8).restore(payload), KeyError, "mem")
    lost_state = _continuation_trial("B-omit")
    refusal = _refused(lambda: BrianLIF(0.8, refractory=2), NotImplementedError, "refractory")
    fallback = _refused(
        lambda: _MutantSilentFallback(0.8, refractory=2), NotImplementedError, "refractory"
    )
    resource = _resource_refusals()
    return {
        "R1": {
            "fires": shipped["ok"] and not threshold["ok"],
            "shipped": shipped,
            "mutated": threshold,
        },
        "R2": {
            "fires": shipped["ok"]
            and not late["ok"]
            and not zero["ok"]
            and late["second_step_membrane_diff"] > 1e-6,
            "shipped": shipped,
            "mutated": {"late": late, "zero": zero},
        },
        "R3": {
            "fires": shipped["ok"] and not schedule["ok"],
            "shipped": shipped,
            "mutated": schedule,
        },
        "R4": {
            "fires": child.returncode == 1
            and "RuntimeError" in child.stderr
            and "enabled the GIL" in child.stderr,
            "shipped": {"parent_gil": sys._is_gil_enabled()},
            "mutated": {"returncode": child.returncode, "stderr": child.stderr.strip()},
        },
        "R5": {
            "fires": omitted["refused"] and not lost_state["ok"],
            "shipped": omitted,
            "mutated": lost_state,
        },
        "R6": {
            "fires": refusal["refused"] and not fallback["refused"],
            "shipped": refusal,
            "mutated": fallback,
        },
        "R7": {
            "fires": resource["ok"] and resource["timeout_elapsed_s"] < 2,
            "shipped": resource,
            "mutated": {"sleep_s": 10, "requested_cells": resource["requested_cells"]},
        },
    }


def _self_check() -> None:
    currents = (
        np.array([0.4, 1.2, -0.2]),
        np.array([0.8, 0.1, 1.6]),
        np.array([0.3, 0.5, 0.2]),
    )
    for reset in sorted(RESET_MECHANISMS):
        layer = BrianLIF(beta=0.9, reset_mechanism=reset)
        torch_mem = torch.zeros(3)
        brian_spikes, brian_membranes, brian_version = brian_reference_trace(
            currents,
            np.zeros(3),
            beta=0.9,
            reset_mechanism=reset,
        )
        for index, current in enumerate(currents):
            torch_spikes, torch_mem = layer(torch.tensor(current, dtype=torch.float32), torch_mem)
            assert np.array_equal(torch_spikes.detach().numpy(), brian_spikes[index])
            assert np.allclose(torch_mem.detach().numpy(), brian_membranes[index], atol=1e-6)

    current = torch.tensor([0.95, 1.05], requires_grad=True)
    spikes, _ = BrianLIF(beta=0.9, spike_grad=fast_sigmoid())(current)
    spikes.sum().backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert torch.count_nonzero(current.grad) == current.numel()
    assert not sys._is_gil_enabled(), "Brian2 reference enabled the GIL in the parent process"
    print(f"brian2 {brian_version}: isolated reference and Torch surrogate bridge agree")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--reference-trace":
        print("BRIAN2_RESULT=" + json.dumps(_run_reference_child(json.loads(sys.stdin.read()))))
    elif len(sys.argv) == 4 and sys.argv[1] == "--continuation":
        _continuation_child(sys.argv[2], sys.argv[3])
    else:
        _self_check()
        accepted = True
        for name, conjunct in (
            ("p", conjunct_p),
            ("m", conjunct_m),
            ("c", conjunct_c),
            ("g", conjunct_g),
            ("r", conjunct_r),
            ("i", conjunct_i),
        ):
            result = conjunct()
            print(name + "=" + json.dumps(result), flush=True)
            accepted = accepted and result["ok"]
        print("k=" + json.dumps(cost_report()), flush=True)
        red_results = reds()
        print("reds=" + json.dumps(red_results), flush=True)
        raise SystemExit(0 if accepted and all(r["fires"] for r in red_results.values()) else 1)
