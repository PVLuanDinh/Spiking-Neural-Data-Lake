# Research scan — cycle 3 (v0.64)

_Extensive multi-source sweep (arXiv, neuroscience via biorxiv/pubmed/consensus, neuromorphic-hardware,
robotics/event, GitHub ecosystem): 49 candidates → deduped against everything built in cycles 1–2 →
ranked by on-thesis × leverage × zero-dep. Verified "already built" claims against the source files
(e.g. `spiking_vsa.py` has bind/bundle/permute/cleanup only — no unbind/resonator)._

**Top pick (next zero-dep build):** #1 Log-sum-ReLU energy for the dense associative memory — a one-kernel
swap in `research/dense_associative_memory.py` that removes the softmax/overflow fragility while keeping
exponential capacity. Effort S, hardens the store core the whole lake recalls over.

## Build now — pure-stdlib, on-thesis (verdict BUILD-zerodep)

| # | Direction | Fit | Effort | Source | Note |
|---|-----------|-----|:-----:|--------|------|
| 1 | **Log-sum-ReLU / Epanechnikov energy** for dense AM | store | S | arXiv:2506.10801 | ReLU replaces `exp()` → no clamp/logsumexp; keeps exponential capacity + exact recall. Upgrades `dense_associative_memory.py`. |
| 2 | **Resonator network** — VSA factorization (unbind-and-cleanup) | reason | M | Nat.Mach.Intell. 6:641 (2024) / arXiv:2208.12880 | Built VSA can't invert a bound product; a resonator adds decomposition/analogy over the existing HD codebook. Extends `spiking_vsa.py`. |
| 3 | **Three-factor learning of synaptic delays** (timing as a learned param) | sequence/reason | M | arXiv:2601.00668 (2026) | Thesis is "reason in spike timing" yet everything learns weights; delay-learning is the on-thesis training upgrade (+20% over weights-only, SHD BPTT-parity). |
| 4 | **Modular Composite Representation (MCR)** HDC substrate | reason/store | S | arXiv:2511.09708 (2025) | Modular-integer algebra, binary accuracy at ~4× less memory (123 datasets). Denser codebook swap for `spiking_vsa.py`. |
| 5 | **Distributed (non-WTA) dense AM**, exp-in-hidden capacity | store | M | arXiv:2601.00984 (2026) | Removes exactly the k-WTA bottleneck of `spiking_storage_prototype.py` (WTA = linear capacity → distributed = exponential). |
| 6 | **Quadtree lossless event compression** | store | M | arXiv:2411.07155 (2024) | 2-D spatial axis orthogonal to the built 1-D delta+varint; doubles as the low-latency streaming codec (→ C3). |
| 7 | **Bayesian binary-weight SNN** — calibrated uncertainty | reason/search | S | arXiv:2401.15453 (2024) | Bernoulli mask + N spike-count passes → confidence on KG queries + novelty gate. Nothing emits confidence today. |
| 8 | **Lossless compression of index IDs / edge links** | store | M | arXiv:2501.10479 (Meta, 2025) | Compresses the `.spk` lookup metadata (ids/edges), not the payload — ~7× on ids, complementary to #6. |
| 9 | **NeuroBench Eff-AC / activation-sparsity metrics** | energy | S | arXiv:2304.04640 (Nat.Comms 2025) | Adopt the metric *definitions* in `energy.py` (skip the torch harness) → citable, leaderboard-legible efficiency numbers. |
| 14 | **Free-energy-constraining spiking control** (spikes fire only if they cut free energy) | robot-loop | M | arXiv:2603.09729 (2026) | Predictive-coding controller with principled sparsity + graceful degradation under noise/collision/delay — the PID/MPC path lacks this. |

## Research-tier / positioning (torch / GPU / hardware — quarantined, map to open gaps)

| # | Direction | Gap | Source | Note |
|---|-----------|-----|--------|------|
| 10 | **NIR export** for Paradigm B/C | C1 | Nat.Comms 2024 (s41467-024-52259-9); github/neuromorphs/NIR | Emit a small NIR graph JSON → Loihi 2 + SpiNNaker 2 + Speck at once, vs a bespoke Lava translator. Highest-leverage C1 move. |
| 11 | **Local-rule baselines (DLL / S-TLLR)** | B1 | arXiv:2501.09976; arXiv:2306.15220 | Open-source local-learning alternatives to benchmark against the built STDP/surrogate models. |
| 12 | **DVS128-Gesture flagship** via Tonic + QAT/NIR recipe | B2 | arXiv:2504.06748; github/neuromorphs/tonic | Reuse Tonic (don't hand-roll AEDAT); 94.13% on-chip recipe is copyable. Replaces off-theme N-MNIST. |
| 13 | **Loihi 2 delay-SNN pipeline** | C1 | arXiv:2510.13757 (2025) | Published recipe to map delay-based temporal-order computation (Paradigm B) to Loihi 2. |
| 15 | **Firmware heartbeat/dead-man watchdog** + hard relay de-energize | D1 | arXiv:2604.24242; Standard Bots | Concrete firmware pattern; built actuator is only software-gated (`SAFETY_CALIBRATED`). |
| 16 | **Neuromorphic incipient-slip early-warning** | D1 | arXiv:2509.09546 (2025) | Fires ~360 ms before gross slip — a tactile pre-emption signal to drive the D1 watchdog. |

## Rejected (19, grouped)
- **Overlapping VSA/memory variants** (GC-VSA grid cells, LARS-VSA/HDSymbolicAttention, Hopfield-Fenchel-Young, continuous-time compressed Hopfield, Spiking TEM) — the resonator (#2) + MCR (#4) + distributed-AM (#5) already cover the un-built VSA/memory deltas.
- **torch-tier plasticity pile-ons** (burst imitation, feedback-as-plasticity, ccRNN cerebellar, AGMP astrocyte, synaptic-delay KG) — the learning path is already crowded (RPE-STDP, valence-STDP, surrogate, Dreamer, +#3).
- **Silicon substrates** (FeFET TCAM, SOT-MRAM CIM, memristor CIM-SSM, ASTER PIM) — no accessible hardware; position-only.
- **Off-thesis / solved-elsewhere** (product-key memory layers, ANN→SNN CRPI fix, LQR controller bench, SGNNBench) and **CI supply-chain hardening (C4)** — routine ops, not research.

## Sources
Primary arXiv ids inline above; neuroscience via biorxiv/pubmed/consensus; ecosystem via GitHub (NIR,
Tonic). Full 49-candidate raw set + per-source notes in the workflow transcript.
