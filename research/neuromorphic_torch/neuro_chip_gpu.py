"""
NeuroChip GPU — torch/GPU-accelerated vectorized twin of neuro_chip.py

This is the torch-vectorized, GPU-scalable sibling of src/snn_data_lake/neuro_chip.py.
Same functional/architectural emulator (event-driven cores + AER spike routing + SynOp/energy/
latency accounting), but VECTORIZED in torch so neurons and synapses run as tensor operations.
Reproduces the SAME execution model and semantics: per-timestep LIF dynamics on flattened
neuron populations, AER routing as a weight matrix, 1-tick router hops, deterministic SynOp
and energy accounting.

This is NOT cycle-accurate silicon, NOT AirLLM, NOT an LLM accelerator. It is a mechanical
vectorization of the stdlib emulator to fixed GPU/CPU backends for scaling to thousands of
neurons and running inference fast. The ceiling is the same: functional/architectural, not
physics-level fidelity.

Credit: Loihi/Akida AER routing architecture; energy constants from energy.py (Horowitz 2014,
arXiv:2210.13107); LIF dynamics match lif_neuron.LIFNeuron exactly, vectorized as tensor ops.

Usage:
  import torch
  from research.neuromorphic_torch.neuro_chip_gpu import NeuroChipGPU

  chip = NeuroChipGPU(n_neurons=4096, n_cores=16)
  chip.add_synapses(W_sparse)  # connectivity matrix [n_neurons, n_neurons]
  metrics = chip.run(external_input, ticks=50)
  print(metrics)  # {spikes, synops, energy_ac_j, energy_loihi_j, device, per_core_spikes, ...}

Torch is imported lazily (guarded). If torch is absent, the module itself remains importable
but run() will fail gracefully. The smoke check skips if torch is not available.
"""

import sys
import time
import math

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Energy constants (cited, from energy.py) — hardcoded to keep zero-dep on the core
E_AC_FP32_45NM = 0.9e-12      # accumulate-only op, 45 nm CMOS (Horowitz 2014)
E_SYNOP_LOIHI = 23.6e-12      # per-SynOp on Loihi incl. routing (arXiv:2210.13107)


class NeuroChipGPU:
    """Vectorized neuromorphic chip emulator: flattened neuron populations + AER routing as tensors."""

    def __init__(self, n_neurons, n_cores=1, tau_m=10.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0):
        """
        Initialize a GPU-backed neuromorphic chip.

        Args:
            n_neurons: total number of neurons (flattened across all cores)
            n_cores: number of cores (for per-core spike tracking; placement-invariant)
            tau_m: membrane time constant
            v_threshold: spike threshold
            v_reset: reset potential after spike
            v_rest: resting potential
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for NeuroChipGPU")

        self.n_neurons = n_neurons
        self.n_cores = n_cores
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest

        # Decay constant (vectorized LIF): exp(-1/tau_m)
        self.decay = math.exp(-1.0 / tau_m)

        # Pick device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # State: membrane potentials (all neurons, flattened)
        self.v = torch.full((n_neurons,), v_rest, dtype=torch.float32, device=self.device)

        # Connectivity: weight matrix W[src, dst] = weight (sparse or dense; we use dense for simplicity)
        # Initially no connections
        self.W = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=self.device)

        # Core assignment: neuron index -> core id (for per-core spike counting)
        # Divide neurons evenly across cores
        self.core_assignment = torch.zeros(n_neurons, dtype=torch.int64, device=self.device)
        neurons_per_core = n_neurons // n_cores
        for c in range(n_cores):
            start = c * neurons_per_core
            end = start + neurons_per_core if c < n_cores - 1 else n_neurons
            self.core_assignment[start:end] = c

    def add_synapses(self, W):
        """
        Load a connectivity matrix.

        Args:
            W: torch.Tensor or numpy array of shape [n_neurons, n_neurons],
               where W[src, dst] = weight (0 means no connection).
        """
        if not torch.is_tensor(W):
            W = torch.tensor(W, dtype=torch.float32)
        W = W.to(device=self.device, dtype=torch.float32)
        assert W.shape == (self.n_neurons, self.n_neurons), \
            f"Weight matrix shape {W.shape} != ({self.n_neurons}, {self.n_neurons})"
        self.W = W

    def run(self, external_input=None, ticks=1):
        """
        Time-stepped event loop: advance neurons and route spikes via AER.

        Each timestep:
          1. Deliver spikes from LAST tick through W as input current
          2. Add scheduled external input (if provided)
          3. Update v via vectorized LIF: v = v_rest + (v - v_rest)*decay + current
          4. Detect spikes (v >= threshold), reset spiked neurons
          5. Accumulate SynOps = (number of spikes) x (fan-out per spike)

        Args:
            external_input: dict {tick: torch.Tensor} or None
                           if provided, external_input[t] is a [n_neurons] current vector
            ticks: number of timesteps

        Returns:
            dict with keys:
              - spikes: total spike count
              - synops: total synaptic operations (spike x fan-out)
              - energy_ac_j: energy via 45nm AC model
              - energy_loihi_j: energy via Loihi model
              - ticks: number of timesteps executed
              - device: str, 'cpu' or 'cuda'
              - per_core_spikes: list of spike counts per core
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required")

        if external_input is None:
            external_input = {}

        # Reset state
        self.v = torch.full((self.n_neurons,), self.v_rest, dtype=torch.float32, device=self.device)

        total_spikes = 0
        total_synops = 0
        per_core_spikes = [0] * self.n_cores
        last_spikes = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)

        for t in range(ticks):
            # (1) AER routing: deliver spikes from last tick as currents
            #     current = W.T @ last_spikes (each neuron receives weighted spikes from its inputs)
            current = torch.matmul(self.W.T, last_spikes)

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
                # For each spiking neuron, count its nonzero fan-out
                for src_idx in spike_indices:
                    fan_out = int((self.W[src_idx] != 0).sum().item())
                    total_synops += fan_out
            else:
                fan_out = 0

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
            "per_core_spikes": per_core_spikes,
        }


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("torch not installed - skipping neuro_chip_gpu smoke (expected without the [gpu] extra)")
        sys.exit(0)

    print("=" * 70)
    print("NEURO_CHIP_GPU — torch/GPU-accelerated vectorized emulator")
    print("=" * 70)

    # =========================================================================
    # TEST 1: AGREEMENT — Reproduce stdlib reference on a small case
    # =========================================================================
    print("\n[AGREEMENT TEST]")
    # Build a tiny 2-neuron chip: neuron 0 -> neuron 1 with weight 1.5
    # Input: neuron 0 driven by 1.5 current at tick 0
    # Expected: neuron 0 fires at tick 0 (v = 0 + (0-0)*decay + 1.5 = 1.5 >= 1.0)
    #           neuron 1 receives 1.5 at tick 1 via AER hop, fires at tick 1
    chip = NeuroChipGPU(n_neurons=2, n_cores=2, tau_m=10.0)
    W = torch.zeros((2, 2), dtype=torch.float32)
    W[0, 1] = 1.5  # neuron 0 -> neuron 1, weight 1.5
    chip.add_synapses(W)

    # External input: drive neuron 0 with 1.5 current at tick 0
    ext_input = {0: torch.tensor([1.5, 0.0], dtype=torch.float32)}
    metrics = chip.run(external_input=ext_input, ticks=4)

    # Verify: neuron 0 spikes once (tick 0), neuron 1 spikes once (tick 1)
    # per_core_spikes[0] should be 1 (neuron 0), per_core_spikes[1] should be 1 (neuron 1)
    # synops should be 1 (one spike from neuron 0 with fan-out 1)
    assert metrics["spikes"] == 2, f"Expected 2 total spikes, got {metrics['spikes']}"
    assert metrics["synops"] == 1, f"Expected 1 SynOp, got {metrics['synops']}"
    assert metrics["per_core_spikes"][0] == 1, f"Expected 1 spike on core 0, got {metrics['per_core_spikes'][0]}"
    assert metrics["per_core_spikes"][1] == 1, f"Expected 1 spike on core 1, got {metrics['per_core_spikes'][1]}"
    print(f"  Small chip: 2 neurons, input -> neuron 0 fires @ tick 0, AER hop -> neuron 1 fires @ tick 1")
    print(f"    Total spikes: {metrics['spikes']}, SynOps: {metrics['synops']}")
    print(f"    Per-core: {metrics['per_core_spikes']}")
    print("  PASS: Agreement with reference arithmetic")

    # =========================================================================
    # TEST 2: SCALE — Large random chip demonstrating GPU speed
    # =========================================================================
    print("\n[SCALE TEST]")
    torch.manual_seed(0)
    n_neurons_large = 4096
    n_cores_large = 16
    chip_large = NeuroChipGPU(n_neurons=n_neurons_large, n_cores=n_cores_large, tau_m=10.0)

    # Build a random sparse connectivity (seeded for reproducibility)
    # ~1% sparsity: ~40k nonzero entries in a 4096 x 4096 matrix
    sparsity = 0.01
    W_large = torch.bernoulli(torch.full((n_neurons_large, n_neurons_large), sparsity)).float()
    # Scale weights to [0.1, 1.0] for non-zero entries
    W_large[W_large > 0] = 0.1 + 0.9 * torch.rand_like(W_large[W_large > 0])
    chip_large.add_synapses(W_large)

    # Drive a few random neurons with constant input
    n_driven = 10
    ext_input_large = {}
    for t in range(50):
        ext = torch.zeros(n_neurons_large)
        # Drive first n_driven neurons with varying currents
        ext[:n_driven] = 0.5 + 0.3 * torch.sin(torch.tensor(float(t) * 0.1))
        ext_input_large[t] = ext

    start = time.time()
    metrics_large = chip_large.run(external_input=ext_input_large, ticks=50)
    elapsed_ms = (time.time() - start) * 1000

    print(f"  Large chip: {n_neurons_large} neurons, {n_cores_large} cores, {int(W_large.sum().item())} connections")
    print(f"    Sparsity: {sparsity:.2%}")
    print(f"    Total spikes: {metrics_large['spikes']}")
    print(f"    Total SynOps: {metrics_large['synops']}")
    print(f"    Energy (AC):    {metrics_large['energy_ac_j']:.3e} J")
    print(f"    Energy (Loihi): {metrics_large['energy_loihi_j']:.3e} J")
    print(f"    Wall-clock:     {elapsed_ms:.1f} ms")
    print(f"    Device:         {metrics_large['device']}")
    print("  PASS: Large chip executed without error")

    # =========================================================================
    # FINAL CHECK
    # =========================================================================
    print("\n" + "=" * 70)
    print("smoke OK: agreement test passed, scale test passed, device=" + metrics_large["device"])
    print("=" * 70)
    sys.exit(0)
