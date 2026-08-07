# Data Model: Finish the GeNN Port

Entities are runtime/measurement structures — this feature persists nothing to the lake.

## E1. SpikeEncoding (shared with the CPU oracle)

The deterministic burst-latency encoding of one MNIST image.

- **times**: int tensor, spike timestamps in [0, T), T=30
- **indices**: int tensor, same length, input pixel ids in [0, 784)
- Producer: `reasoning/snn_mnist_stdp_fast.encode_latency` (REUSED verbatim — the port imports
  it; reimplementing it would break the determinism contract and P2)
- Invariants: pixel >= 0.10 emits >= 1 spike; count proportional to intensity (cap FAST_BURST);
  first spike at round((1-intensity)*(T-1)); NO RNG anywhere → identical input, identical output
- GPU upload form: concatenated per-pixel-sorted timestamp array + per-neuron [start, end)
  index pairs (the SpikeSourceArray layout: startSpike/endSpike vars + spikeTimes array)

## E2. NetworkState (device-resident)

- **g** [M x 784]: synaptic weights, init uniform*0.3 then row-normalized to W_NORM; clamp
  [0, WMAX] enforced by the plasticity rule
- **V** [M]: membrane; per-step `V = V*BETA + Isyn`; zeroed on any population spike (global
  reset) and between images
- **theta** [M]: adaptive threshold; winner += THETA_PLUS; *= THETA_DECAY per learn step;
  PERSISTS across images (the homeostasis that separates prototypes)
- **preTrace** [784]: presynaptic trace; *= PRE_DECAY per step; += 1.0 on input spike; zeroed
  between images
- **winnerFlag** [M] (+ scalar fired flag): written by the WTA reduction each step; threshold
  condition reads it; cleared after the step
- State transitions per timestep (order is the contract — it mirrors
  FastStdpNetwork.run's statement order): decay traces → deliver input spikes (trace bump +
  current) → WTA reduction (argmax(V - theta) vs FAST_THRESH) → winner emits true spike →
  post_spike LTP + theta bump → global V reset if fired → theta decay (learn only)
- Per-image transitions: zero V/preTrace/counts; row-normalize g (learn only)

## E3. AssignmentAndScore (host-resident)

- **counts** [M per image]: winner spike counts read back per image
- **assignment** [M]: argmax over class-conditional summed responses on the training set
- **accuracy**: fraction of test images whose argmax class-summed response matches the label
- Same math as `fast_assign`/`fast_eval`; numpy on host (counts are tiny)

## E4. EvidenceRecord (specs/012-finish-genn-port/evidence.md)

One entry per measured run, append-only, verbatim outputs:

- **preflight**: pip install output tail, pygenn version, minimal-model compile/run result,
  compiler + CUDA discovery notes; on failure the verbatim error (the STOP record)
- **cpu-baseline**: config (NORD_M=300 NORD_TRAIN=6000 NORD_TEST=1000, seed 0, SKIP_RATE=1),
  accuracy, wall-clock — the same-box oracle figure
- **parity**: same config on GPU; accuracy, wall-clock, |delta| vs cpu-baseline (bar: <= 3 pp)
- **control**: lr=0 run, accuracy (bar: < 20%)
- **determinism**: two encodes + two GPU runs of one image, spike-train equality (bar: exact)
- **payoff-300 / payoff-1600**: full 60k, accuracy + wall-clock vs the CPU loop's measured
  figures; the honest statement of what the port bought
- **red-demo**: the constructed reds — mutated constant fails the codegen check; the criterion
  can fail

## E5. LedgerRow (docs/P4C_BACKLOG.md)

The mandated-build row transitions: MANDATED BUILD → BUILT+MEASURED (with the evidence
pointers) or → BLOCKED (with the verbatim pre-flight blocker awaiting owner decision). No third
state.
