"""
NeuroChipGPUStream -- AirLLM-style WEIGHT-STREAMING for the GPU neuromorphic chip emulator.

AirLLM (github.com/lyogavin/airllm) runs a 70B model on a 4GB GPU by LAYER-STREAMING:
load one transformer layer's weights from disk, compute, free it, load the next -- so
peak resident memory is the LARGEST SINGLE LAYER, not the whole model.

This applies that architectural idea to the GPU neuromorphic chip emulator: the full
dense [N x N] synapse weight matrix does NOT fit VRAM for large N (1M neurons -> ~4 TB).
Instead we SHARD the weight matrix BY CORE (each core's weight rows), persist each shard
to disk, and during run() STREAM each core's weight block from disk -> GPU, compute that
core's contribution to the tick, then FREE it. Peak VRAM = ONE core's weight block, not
the whole matrix -> the emulated chip can far exceed VRAM.

It reproduces neuro_chip_gpu.NeuroChipGPU's EXACT dynamics and semantics (LIF via
v = v_rest + (v - v_rest)*decay + current, spike v>=v_threshold, reset; AER routing
current = W.T @ last_spikes; SynOps = spikes x fan-out; energy consts E_AC_FP32_45NM
and E_SYNOP_LOIHI), so results are bit-for-bit identical to the in-VRAM version --
streaming changes only WHERE the weights live, never the computation.

Concept credit: AirLLM (weight/layer streaming); Loihi/Akida AER routing architecture;
energy constants IMPORTED from neuro_chip_gpu, which imports them from
snn_data_lake.instruments.energy (Horowitz 2014, arXiv:2210.13107) -- one definition, not three;
LIF dynamics match neuro_chip_gpu.NeuroChipGPU exactly, vectorized as tensor ops.

CEILING (honest): same functional/architectural ceiling as neuro_chip_gpu -- NOT
cycle-accurate silicon, NOT AirLLM-the-library, NOT an LLM. Streaming trades VRAM
for disk I/O (shard load+free per core per tick), buying the ability to emulate chips
far larger than VRAM; it adds no new physical fidelity.

PERFORMANCE: streaming is only paid when it is NEEDED. If the whole sharded matrix fits
an on-device budget (~80% of free VRAM on CUDA; 0 on CPU so CPU always streams as before),
run() loads every shard ONCE and keeps it resident -- no per-tick disk reload + host->device
copy, and no per-tick torch.cuda.empty_cache(). Only a matrix that EXCEEDS the budget falls
back to the original per-tick out-of-core streaming (the module's reason to exist). The
device comes from NORD_DEVICE / NORD_GPU, default CPU (opt-in GPU) -- see _pick_device; this
sentence used to credit "the shared NORD_GPU/NORD_DEVICE selector (research/torch_device.py)",
a file that does not exist and an import that never once resolved.
On CUDA the per-core W_c.T @ spikes matmul runs under bf16 autocast
(Blackwell tensor cores), accumulated in fp32; pass run(autocast=False) for bit-for-bit fp32
agreement with neuro_chip_gpu (what the agreement self-check uses). CPU is byte-identical to
before: autocast is ignored and the budget is 0, so CPU still streams in fp32.

Usage:
  import torch
  from research.neuromorphic_torch.neuro_chip_gpu_stream import NeuroChipGPUStream

  chip = NeuroChipGPUStream(n_neurons=1000000, n_cores=1024)
  chip.set_weights(W_sharded_dir)  # or chip.add_core() + chip.connect()
  metrics = chip.run(external_input, ticks=50)
  print(metrics)  # {spikes, synops, energy_ac_j, energy_loihi_j, device,
                  #  max_resident_weight_bytes, full_matrix_bytes, ticks}
"""

import sys
import time
import math
import os
import tempfile
import shutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Energy constants: ONE definition, taken from the module this file exists to reproduce
# bit-for-bit. neuro_chip_gpu imports them from snn_data_lake.instruments.energy; both files used to
# carry their own literal copy under the comment "hardcoded to keep zero-dep on the core", which
# P7's abolition of the zero-dependency rule (2026-07-20) voided and P2 forbids -- three live
# definitions of two numbers, in a file whose entire claim is byte-for-byte agreement with one of
# the other two. Measured before the swap: all three copies were already equal, so no number moves.
# Both documented invocations are covered: the docstring's
# `from research.neuromorphic_torch.neuro_chip_gpu_stream import ...` (package form, relative
# import) and `python research/neuromorphic_torch/neuro_chip_gpu_stream.py` (script form, where
# __package__ is empty and only the flat name resolves). The except is narrow and the fallback
# imports the SAME names, so a genuine failure still raises -- from the second attempt, loudly.
try:
    from .neuro_chip_gpu import E_AC_FP32_45NM, E_SYNOP_LOIHI  # noqa: E402
except ImportError:
    from neuro_chip_gpu import E_AC_FP32_45NM, E_SYNOP_LOIHI  # noqa: E402


def _pick_device():
    """Device from NORD_DEVICE / NORD_GPU, defaulting to CPU (opt-in GPU). ASCII-only output.

    REWRITTEN 2026-08-04. This used to try `from torch_device import resolve_device` first, after
    putting research/ on sys.path, and described that module in the file's own docstring as "the
    shared NORD_GPU/NORD_DEVICE selector (research/torch_device.py)". THERE IS NO SUCH FILE --
    research/ holds no top-level .py at all (it has three subdirectories). So the import raised on
    every single call, was swallowed by a bare `except Exception:`, and the fallback below is what
    has actually resolved the device for the life of the feature, while the docstring credited a
    module that does not exist. That is this repo's silent-degradation shape exactly: a stage fails,
    swallows it, and the placeholder path returns a plausible answer. The swallow also left a
    permanent sys.path entry behind on behalf of a module it never found.
    The fallback was correct, so the fix is to make it the ONLY path and say so.
    """
    import torch  # guaranteed present: the constructor guards on TORCH_AVAILABLE first
    want = os.environ.get("NORD_DEVICE")
    if not want and os.environ.get("NORD_GPU") == "1":
        want = "cuda"
    if want and want.startswith("cuda") and torch.cuda.is_available():
        return want
    return "cpu"


def _resident_budget_bytes(device, override=None):
    """Bytes of weights allowed to stay RESIDENT on `device`. `override` (bytes) wins;
    else on CUDA use ~80% of currently-free VRAM, on CPU return 0 so shards always stream
    (out-of-core path unchanged from the original disk-streaming behavior)."""
    import torch
    if override is not None:
        return int(override)
    if str(device).startswith("cuda"):
        free, _total = torch.cuda.mem_get_info()  # GPU-VERIFY: free-VRAM query (CUDA-only)
        return int(free * 0.8)                    # GPU-VERIFY: keep 20% headroom for activations
    return 0


class NeuroChipGPUStream:
    """
    Streaming neuromorphic chip emulator: weight matrix sharded by core on disk,
    one core's weights resident on GPU/CPU at a time.
    """

    def __init__(self, n_neurons, n_cores=1, tau_m=10.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0):
        """
        Initialize a streaming GPU-backed neuromorphic chip.

        Args:
            n_neurons: total number of neurons (flattened across all cores)
            n_cores: number of cores (for per-core spike tracking; also determines shard count)
            tau_m: membrane time constant
            v_threshold: spike threshold
            v_reset: reset potential after spike
            v_rest: resting potential
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for NeuroChipGPUStream")

        self.n_neurons = n_neurons
        self.n_cores = n_cores
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest

        # Decay constant (vectorized LIF): exp(-1/tau_m)
        self.decay = math.exp(-1.0 / tau_m)

        # Pick device via the shared NORD_GPU/NORD_DEVICE selector (default CPU, opt-in GPU)
        self.device = _pick_device()

        # State: membrane potentials (all neurons, on device; stays resident)
        self.v = torch.full((n_neurons,), v_rest, dtype=torch.float32, device=self.device)

        # Core assignment: neuron index -> core id (for per-core spike counting)
        # Divide neurons evenly across cores
        self.core_assignment = torch.zeros(n_neurons, dtype=torch.int64, device=self.device)
        neurons_per_core = n_neurons // n_cores
        for c in range(n_cores):
            start = c * neurons_per_core
            end = start + neurons_per_core if c < n_cores - 1 else n_neurons
            self.core_assignment[start:end] = c

        # Weight shards: store on disk in a temp directory
        self.shard_dir = tempfile.mkdtemp(prefix="neuro_chip_gpu_stream_")
        self._shard_paths = [os.path.join(self.shard_dir, f"shard_core_{c}.pt") for c in range(n_cores)]

        # Initialize empty shards (will be filled by add_core or set_weights)
        self.W_shape = (n_neurons, n_neurons)  # full matrix shape for reference
        self._shard_bytes = [0] * n_cores      # bytes per shard for tracking
        self._max_resident_weight_bytes = 0

        # Create placeholder shards
        for c in range(n_cores):
            empty_shard = torch.zeros((neurons_per_core if c < n_cores - 1 else n_neurons - c * neurons_per_core,
                                       n_neurons),
                                      dtype=torch.float32)
            torch.save(empty_shard, self._shard_paths[c])

    def add_core(self, core_id, n_neurons_in_core):
        """
        Explicitly add a core with a given number of neurons (for API consistency).
        Initializes its shard to zeros; call connect() to build weights.
        """
        if core_id < 0 or core_id >= self.n_cores:
            raise IndexError(f"core_id {core_id} out of range [0, {self.n_cores})")
        # Initialize shard for this core
        shard = torch.zeros((n_neurons_in_core, self.n_neurons), dtype=torch.float32)
        torch.save(shard, self._shard_paths[core_id])

    def connect(self, src_core, src_neuron, dst_neuron, weight):
        """
        Set a single synapse: src_core's src_neuron -> global dst_neuron with weight.
        Loads shard, updates, saves back (1 core resident at a time).
        """
        # Load the core shard
        shard = torch.load(self._shard_paths[src_core], weights_only=True)
        # Update the weight
        shard[src_neuron, dst_neuron] = weight
        # Save back
        torch.save(shard, self._shard_paths[src_core])

    def set_weights(self, W):
        """
        Load a full [n_neurons, n_neurons] weight matrix and shard it by core.

        Args:
            W: torch.Tensor or numpy array of shape [n_neurons, n_neurons]
        """
        if not torch.is_tensor(W):
            W = torch.tensor(W, dtype=torch.float32)
        W = W.to(dtype=torch.float32)
        assert W.shape == (self.n_neurons, self.n_neurons), \
            f"Weight matrix shape {W.shape} != ({self.n_neurons}, {self.n_neurons})"

        # Shard W by core (each core's rows)
        neurons_per_core = self.n_neurons // self.n_cores
        for c in range(self.n_cores):
            start = c * neurons_per_core
            end = start + neurons_per_core if c < self.n_cores - 1 else self.n_neurons
            shard = W[start:end, :]
            torch.save(shard, self._shard_paths[c])
            self._shard_bytes[c] = shard.element_size() * shard.nelement()

    def run(self, external_input=None, ticks=1, vram_budget_bytes=None, autocast=True):
        """
        Time-stepped event loop. If the whole sharded matrix fits the on-device budget the
        shards are loaded ONCE and kept resident; otherwise each core's weights are streamed
        from disk one at a time, contribute to the global current, then freed (out-of-core).

        Each timestep:
          1. For each core k in order:
             a. LOAD W_k (core k's weight shard) from disk to device
             b. Accumulate W_k.T @ last_spikes into the global current vector
             c. FREE W_k from device
          2. Add scheduled external input
          3. Update v via vectorized LIF: v = v_rest + (v - v_rest)*decay + current
          4. Detect spikes (v >= threshold), reset spiked neurons
          5. Accumulate SynOps = (number of spikes) x (fan-out per spike)

        Args:
            external_input: dict {tick: torch.Tensor} or None
                           if provided, external_input[t] is a [n_neurons] current vector
            ticks: number of timesteps
            vram_budget_bytes: max bytes of weights kept resident on-device (None = auto:
                               ~80% of free VRAM on CUDA, 0 on CPU so shards always stream).
            autocast: on CUDA, run the per-core matmul under bf16 autocast (Blackwell tensor
                      cores), accumulated in fp32; ignored on CPU. Pass False for bit-for-bit
                      fp32 agreement with neuro_chip_gpu.

        Returns:
            dict with keys:
              - spikes: total spike count
              - synops: total synaptic operations (spike x fan-out)
              - energy_ac_j: energy via 45nm AC model
              - energy_loihi_j: energy via Loihi model
              - ticks: number of timesteps executed
              - device: str, 'cpu' or 'cuda'
              - max_resident_weight_bytes: peak bytes of weight blocks on device
              - full_matrix_bytes: total size of full [n_neurons, n_neurons] weight matrix
              - per_core_spikes: list of spike counts per core
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required")

        if external_input is None:
            external_input = {}

        # Compute full matrix size for reference
        full_matrix_bytes = self.n_neurons * self.n_neurons * 4  # float32

        # Reset state
        self.v = torch.full((self.n_neurons,), self.v_rest, dtype=torch.float32, device=self.device)

        total_spikes = 0
        total_synops = 0
        per_core_spikes = [0] * self.n_cores
        last_spikes = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self._max_resident_weight_bytes = 0

        # ---- Residency decision: if the whole sharded matrix fits the on-device budget,
        # load every shard ONCE and keep it resident (no per-tick disk reload + H2D copy).
        # Only when it EXCEEDS the budget do we fall back to per-tick out-of-core streaming.
        total_shard_bytes = self.n_neurons * self.n_neurons * 4  # row-shards partition the full matrix
        budget = _resident_budget_bytes(self.device, vram_budget_bytes)
        resident_weights = None
        if total_shard_bytes <= budget:
            # GPU-VERIFY: resident path is only taken on CUDA (budget is 0 on CPU -> always streams)
            resident_weights = []
            for c in range(self.n_cores):
                W_c = torch.load(self._shard_paths[c], weights_only=True).to(device=self.device, dtype=torch.float32)
                resident_weights.append(W_c)
            # all shards resident at once -> peak resident weight bytes is their sum
            self._max_resident_weight_bytes = total_shard_bytes

        use_autocast = bool(autocast) and str(self.device).startswith("cuda")
        neurons_per_core = self.n_neurons // self.n_cores

        for t in range(ticks):
            # (1) Accumulate each core's contribution into the tick current
            current = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)

            for c in range(self.n_cores):
                if resident_weights is not None:
                    W_c = resident_weights[c]                    # already on device, no reload
                else:
                    # out-of-core: stream this core's shard disk -> device for this tick
                    W_c = torch.load(self._shard_paths[c], weights_only=True).to(device=self.device, dtype=torch.float32)
                    resident_bytes = W_c.element_size() * W_c.nelement()
                    self._max_resident_weight_bytes = max(self._max_resident_weight_bytes, resident_bytes)

                # W_c is [neurons_in_core_c, n_neurons], so W_c.T @ seg gives core c's
                # contribution from its neurons to all neurons.
                seg = last_spikes[c * neurons_per_core:
                                  (c + 1) * neurons_per_core if c < self.n_cores - 1 else self.n_neurons]
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # GPU-VERIFY: Blackwell tensor cores
                        contrib = torch.matmul(W_c.T, seg)
                    current = current + contrib.float()          # accumulate in fp32
                else:
                    current = current + torch.matmul(W_c.T, seg)

                if resident_weights is None:
                    # FREE the streamed shard; NO per-tick empty_cache (the caching allocator
                    # reuses this block for the next load -- empty_cache here only stalled the GPU).
                    del W_c

            # (2) Add scheduled external input
            if t in external_input:
                ext = external_input[t]
                if not torch.is_tensor(ext):
                    ext = torch.tensor(ext, dtype=torch.float32, device=self.device)
                else:
                    ext = ext.to(device=self.device, dtype=torch.float32)
                assert ext.shape == (self.n_neurons,), f"External input shape {ext.shape} != ({self.n_neurons},)"
                current = current + ext

            # (3) Vectorized LIF step
            #     v = v_rest + (v - v_rest)*decay + current
            self.v = self.v_rest + (self.v - self.v_rest) * self.decay + current

            # (4) Detect spikes and reset
            spikes = (self.v >= self.v_threshold).float()
            self.v[spikes.bool()] = self.v_reset

            # Count spikes and per-core spikes
            n_spikes = int(spikes.sum().item())
            total_spikes += n_spikes

            # Per-core spike tracking
            if n_spikes > 0:
                spike_indices = torch.where(spikes.bool())[0]
                spike_cores = self.core_assignment[spike_indices]
                for core_id in range(self.n_cores):
                    count = int((spike_cores == core_id).sum().item())
                    per_core_spikes[core_id] += count

            # (5) SynOp accounting: spikes x fan-out per spike
            #     For each spike at src, count nonzero entries in W[src, :] (fan-out)
            if n_spikes > 0:
                spike_indices = torch.where(spikes.bool())[0]
                for src_idx in spike_indices:
                    # Determine which core owns this source neuron
                    src_core = int(self.core_assignment[src_idx].item())
                    if resident_weights is not None:
                        W_c = resident_weights[src_core]         # reuse resident shard, no disk reload
                    else:
                        # Load that core's shard to count fan-out
                        W_c = torch.load(self._shard_paths[src_core], weights_only=True)
                    # Local index within the core (neurons_per_core defined before the tick loop)
                    src_local = int(src_idx) - src_core * neurons_per_core
                    if src_local < W_c.shape[0]:
                        fan_out = int((W_c[src_local] != 0).sum().item())
                        total_synops += fan_out
                    if resident_weights is None:
                        del W_c

            # Queue spikes for next tick
            last_spikes = spikes

        # Energy calculation
        energy_ac_j = total_synops * E_AC_FP32_45NM
        energy_loihi_j = total_synops * E_SYNOP_LOIHI

        return {
            "spikes": total_spikes,
            "synops": total_synops,
            "energy_ac_j": energy_ac_j,
            "energy_loihi_j": energy_loihi_j,
            "ticks": ticks,
            "device": self.device,
            "max_resident_weight_bytes": self._max_resident_weight_bytes,
            "full_matrix_bytes": full_matrix_bytes,
            "per_core_spikes": per_core_spikes,
        }

    def cleanup(self):
        """Remove shard directory and all files."""
        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir, ignore_errors=True)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("torch not installed - skipping neuro_chip_gpu_stream smoke (expected without the [gpu] extra)")
        sys.exit(0)

    print("=" * 70)
    print("NEURO_CHIP_GPU_STREAM — AirLLM-style weight-streaming emulator")
    print("=" * 70)

    try:
        # =========================================================================
        # TEST 1: AGREEMENT — Reproduce the in-VRAM neuro_chip_gpu on a small case
        # =========================================================================
        print("\n[AGREEMENT TEST]")

        # Import the in-VRAM reference (relative import from same directory)
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from neuro_chip_gpu import NeuroChipGPU

        # Build a tiny 2-neuron chip: neuron 0 -> neuron 1 with weight 1.5
        # Input: neuron 0 driven by 1.5 current at tick 0

        # In-VRAM reference
        chip_ref = NeuroChipGPU(n_neurons=2, n_cores=2, tau_m=10.0)
        W_ref = torch.zeros((2, 2), dtype=torch.float32)
        W_ref[0, 1] = 1.5  # neuron 0 -> neuron 1, weight 1.5
        chip_ref.add_synapses(W_ref)

        ext_input = {0: torch.tensor([1.5, 0.0], dtype=torch.float32)}
        metrics_ref = chip_ref.run(external_input=ext_input, ticks=4)

        # Streaming version -- autocast=False so the comparison is exact fp32 on any device
        # (bf16 tensor-core autocast is the default on CUDA; the agreement path opts out).
        chip_stream = NeuroChipGPUStream(n_neurons=2, n_cores=2, tau_m=10.0)
        chip_stream.set_weights(W_ref)
        metrics_stream = chip_stream.run(external_input=ext_input, ticks=4, autocast=False)

        # Verify agreement
        assert metrics_ref["spikes"] == metrics_stream["spikes"], \
            f"Spike count mismatch: {metrics_ref['spikes']} != {metrics_stream['spikes']}"
        assert metrics_ref["synops"] == metrics_stream["synops"], \
            f"SynOp count mismatch: {metrics_ref['synops']} != {metrics_stream['synops']}"
        assert metrics_ref["per_core_spikes"] == metrics_stream["per_core_spikes"], \
            f"Per-core spikes mismatch: {metrics_ref['per_core_spikes']} != {metrics_stream['per_core_spikes']}"

        print(f"  Small chip: 2 neurons, neuron 0 driven at tick 0")
        print(f"    In-VRAM:  spikes={metrics_ref['spikes']}, synops={metrics_ref['synops']}")
        print(f"    Streaming: spikes={metrics_stream['spikes']}, synops={metrics_stream['synops']}")
        print("  PASS: Bit-for-bit agreement with in-VRAM reference")

        # =========================================================================
        # TEST 2: VRAM BOUND — Verify max_resident_weight_bytes < full_matrix_bytes
        # =========================================================================
        print("\n[VRAM BOUND TEST]")

        # Create a multi-core chip where full matrix would be large
        n_neurons_large = 256
        n_cores_large = 4

        torch.manual_seed(0)
        W_large = torch.bernoulli(torch.full((n_neurons_large, n_neurons_large), 0.01)).float()
        W_large[W_large > 0] = 0.1 + 0.9 * torch.rand_like(W_large[W_large > 0])

        chip_bound = NeuroChipGPUStream(n_neurons=n_neurons_large, n_cores=n_cores_large, tau_m=10.0)
        chip_bound.set_weights(W_large)

        # vram_budget_bytes=1 forces the out-of-core streaming path (shards exceed the budget)
        # so the one-core-resident bound is genuinely exercised on CPU AND GPU.
        ext_input_bound = {0: torch.ones(n_neurons_large) * 0.5}
        metrics_bound = chip_bound.run(external_input=ext_input_bound, ticks=10, vram_budget_bytes=1)

        full_bytes = metrics_bound["full_matrix_bytes"]
        max_resident_bytes = metrics_bound["max_resident_weight_bytes"]

        # max_resident should be exactly one core's block (n_neurons / n_cores * n_neurons)
        expected_shard_bytes = (n_neurons_large // n_cores_large) * n_neurons_large * 4
        assert max_resident_bytes <= expected_shard_bytes * 1.01, \
            f"Max resident bytes {max_resident_bytes} exceeds one core {expected_shard_bytes}"
        assert max_resident_bytes < full_bytes, \
            f"Max resident {max_resident_bytes} should be < full matrix {full_bytes}"

        ratio = full_bytes / max_resident_bytes
        print(f"  Multi-core chip: {n_neurons_large} neurons, {n_cores_large} cores")
        print(f"    Full matrix:         {full_bytes:,} bytes")
        print(f"    Max resident shard:  {max_resident_bytes:,} bytes")
        print(f"    Ratio (emulated chip is {ratio:.1f}x larger than peak VRAM): {ratio:.1f}x")
        print("  PASS: VRAM bound verified (one core resident at a time)")

        # =========================================================================
        # TEST 3: SCALE — Large multi-core chip with streaming
        # =========================================================================
        print("\n[SCALE TEST]")

        torch.manual_seed(0)
        n_neurons_scale = 1024
        n_cores_scale = 16

        chip_scale = NeuroChipGPUStream(n_neurons=n_neurons_scale, n_cores=n_cores_scale, tau_m=10.0)

        # Random sparse connectivity
        sparsity = 0.005
        W_scale = torch.bernoulli(torch.full((n_neurons_scale, n_neurons_scale), sparsity)).float()
        W_scale[W_scale > 0] = 0.1 + 0.9 * torch.rand_like(W_scale[W_scale > 0])
        chip_scale.set_weights(W_scale)

        # Drive with sinusoidal input
        n_driven_scale = 8
        ext_input_scale = {}
        for t in range(20):
            ext = torch.zeros(n_neurons_scale)
            ext[:n_driven_scale] = 0.5 + 0.3 * math.sin(float(t) * 0.1)
            ext_input_scale[t] = ext

        start = time.time()
        metrics_scale = chip_scale.run(external_input=ext_input_scale, ticks=20)
        elapsed_ms = (time.time() - start) * 1000

        print(f"  Large chip: {n_neurons_scale} neurons, {n_cores_scale} cores")
        print(f"    Sparsity: {sparsity:.2%}")
        print(f"    Total spikes: {metrics_scale['spikes']}")
        print(f"    Total SynOps: {metrics_scale['synops']}")
        print(f"    Energy (AC):    {metrics_scale['energy_ac_j']:.3e} J")
        print(f"    Energy (Loihi): {metrics_scale['energy_loihi_j']:.3e} J")
        print(f"    Full matrix:    {metrics_scale['full_matrix_bytes']:,} bytes")
        print(f"    Max resident:   {metrics_scale['max_resident_weight_bytes']:,} bytes")
        print(f"    Wall-clock:     {elapsed_ms:.1f} ms")
        print(f"    Device:         {metrics_scale['device']}")
        print("  PASS: Large chip executed without error, streaming per-core")

        # Clean up
        chip_ref.v = None
        chip_stream.cleanup()
        chip_bound.cleanup()
        chip_scale.cleanup()

        # =========================================================================
        # FINAL CHECK
        # =========================================================================
        print("\n" + "=" * 70)
        print(f"smoke OK: agreement test passed, VRAM bound verified, scale test passed, device={metrics_scale['device']}")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
