# Spiking Neural Data Lake

[![CI](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml/badge.svg)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/Aighluvsekks/Spiking-Neural-Data-Lake?label=release)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Core: zero deps](https://img.shields.io/badge/core-zero%20dependencies-success.svg)](#clone--run)
[![Tests: 84% coverage](https://img.shields.io/badge/tests-84%25%20coverage-success.svg)](#tests--ci)
[![CI matrix: Linux + Windows](https://img.shields.io/badge/CI-Linux%20%2B%20Windows%20%C2%B7%20py3.11--3.13-success.svg)](#tests--ci)

**Store, search, and reason over data _in spike timing_ — and drive a robot arm with the same
event-driven spiking networks, instead of dense floating-point tensors.**

Two systems on one thesis: (1) a **neuromorphic data lake** that stores, queries, and reasons
over sparse spike events, and (2) a **real-time robot-arm control loop** — sense → think → act —
whose reflex, reward learning, stress state, motor controller, and model-based-RL policy are
_all_ spiking networks. The product core is **pure Python standard library, zero runtime
dependencies**, `pip install`-able, and every efficiency claim is _measured_ against a dense
baseline and reproducible from the printed metrics.

---

## Why this matters

- **The question, made measurable.** Can we store, recall, and act on data with **less compute
  and less storage** using spikes? Every prototype measures the two target metrics — compute
  (synaptic operations) and storage (bytes) — against a dense baseline. Numbers below are
  reproduced by running the file.
- **Efficiency, quantified.** Spiking inference does a synaptic op only when a neuron fires:
  **3.6–83.5× fewer operations** across these models, with 1-bit spike events vs 32-bit
  activations, and **61–162× less data moved** for storage/query.
- **A complete neuromorphic sense→think→act loop, not a slide.** A real ESP32 arm is driven end
  to end: the reflex, the reward learning (RPE dopamine), the stress state (cortisol), the
  population-coded PID, and a model-based-RL policy+critic are all spiking — through a **fail-safe
  serial actuator path gated on an explicit `SAFETY_CALIBRATED` flag** — recognizing the builder's
  real captured gestures at **95%**.
- **It scales unchanged.** The local Medallion lakehouse PoC was lifted to GCP and **run
  end-to-end (Phase 1 verified)**: Terraform → GCS Bronze → Dataproc Serverless → BigLake/BigQuery
  reproduced the local Gold features exactly.
- **Engineered like a product.** Zero-dependency core, `pip install -e .`, console entry points,
  **84% test coverage**, and a CI matrix of `{Linux, Windows} × {3.11, 3.12, 3.13}` — the
  Windows-first hardware product is actually tested on Windows.

> **Honesty policy (read the [Limitations](#limitations--open-problems)).** Numbers are
> fixed-seed and the CHANGELOG documents negative results. Where a figure is a _target_ rather
> than achieved (the MNIST ~95% leg), or an abstraction rather than biophysics (the
> neuromodulators), or a plumbing demo rather than a benchmark (N-MNIST), it says so.

---

## Status at a glance

| Area | State |
|------|-------|
| **Data-lake paradigms A / B / C** (store · in-storage search · relational reason) | ✅ complete, zero-dep |
| **Robot-arm neuromorphic control loop** (reflex · dopamine · cortisol · PID · interpreter) | ✅ complete in sim; real-arm actuator path shipped (safety-gated) |
| **Spiking model-based RL** (surrogate-gradient policy → world-model → 6-DOF Dreamer) | ✅ research lineage complete (needs torch, own venv) |
| **Local Medallion lakehouse** (Bronze→Silver→Gold, Parquet + Delta PoC) | ✅ complete |
| **GCP lakehouse Phase 1** (Terraform → Dataproc Serverless → BigLake/BigQuery) | ✅ **verified end-to-end on live GCP** |
| **GPU training (conductance D&C ~95% leg)** | ⏳ 90.0% reproduced (1600n); 6400n/~95% run **blocked by a GCP billing hold**, not a code gap |
| **Engineering** | ✅ zero-dep core · `pip install` · 80/80 self-checks · pytest 85% · ruff · CI matrix |

Current version: **v1.7** · 107 tagged releases (v0.1 → v1.7). See [CHANGELOG.md](CHANGELOG.md) for the full arc.

---

## Results (measured)

All figures are reproduced by running the referenced file (or, for the GPU/torch rows, recorded
in the CHANGELOG for the stated config). Compute = synaptic operations; storage = bytes.

### Data lake — store, search, reason (pure stdlib, in CI)

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Associative memory | recall from 40%-corrupted cue | capacity / robustness | **80 patterns @ 99.6%**, recall holds through **60% corruption** |
| Associative memory | memory footprint | factored vs dense | **874× smaller** (600 B vs 512 KB; O(P·k), not N²) |
| Telemetry hub (Paradigm A) | sparse `.spk` store | size / query I/O | **61× smaller** than a dense raster; windowed query reads **2%** of the file |
| In-storage query (Paradigm B) | coincidence **+ temporal order** | host transfer | **162× less** data to host; order-discriminating (FAR→NEAR ≠ NEAR→FAR) |
| Relational KG (Paradigm C) | link prediction | Hits@1 | **50% TransE · 100% RotatE** (cyclic relations; relation algebra verified) |
| Associative memory (dense) | capacity vs the classical bound | patterns stored | classical **0.138·N** → **dense n=4 clean past 2·N**; **log-sum-ReLU energy** matches softmax capacity, exp-free (no overflow) |
| Reasoning (Paradigm C, VSA) | hyperdimensional bind/bundle/permute | compositional query | **context-steered arm command** (`bind(APPROACH,NEAR)`→GRIP vs …FAR→WAIT) + role–filler recall |
| Sequence model (LSM) | temporal-pattern classification | accuracy | **96%** (chance 33%), event-driven; trained deep SSM on real **SHD → 32% @ 20-class** (torch) |

### Compute-efficiency prototypes (pure stdlib, in CI)

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Supervised spiking classifier | 4-shape recognition | accuracy / compute | **100%**, **3.6× fewer ops** (113.7k SynOps vs 409.6k dense MACs) |
| Temporal (TTFS) coding | latency inference + early exit | SynOps vs rate | **83.5× fewer** (5.3k vs 442.8k), 100% accuracy |
| Spike-driven MoE | firing-rate routing | router params / compute | **0 learned router-net params** (8 biases vs 512), **4× compute cut** (top-2 of 8 experts) |

### Robot-arm neuromorphic loop (real hardware data)

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Gesture recognition (order-aware Paradigm B) | builder's real ESP32 captures | accuracy | **95%** (APPROACH 98% · RETREAT 85% · IDLE 100%) — up from 63% with a direction-blind template matcher |
| Closed control loop | gesture → command → reward → learn | behavior | rewarded gesture's value rises to **APPROACH**; collision → **EMERGENCY_STOP** + cortisol stress → sharpened reflex |

### Spiking model-based RL (research lineage; needs torch, own venv, not in CI)

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Surrogate-gradient spiking policy | 2-DOF reach | reach error | **0.319 → 0.053 m** (BPTT through differentiable FK) |
| Spiking forward-dynamics + MPC | 2-DOF reach through a learned model | reach error | **0.073 m** (world-model loss 1.34 → 0.004) |
| Joint spiking world-model RL (Dreamer) | 2-DOF, both nets spiking | reach distance | **2.695 → 0.116 m (23×)** via imagined rollouts |
| Dreamer-lite, 6-DOF redundant arm | high-dim, imagination-trained actor-critic | reach / critic | **1.03 → 0.48 m**; critic loss 31.6 → 0.45 *(world model is an MLP; actor+critic spiking — see limitations)* |

### Trainable MNIST models (needs torch/GPU, own venv, not in CI)

| Model | Mechanism | Result |
|-------|-----------|--------|
| Unsupervised STDP (CPU) | rate-coded, adaptive threshold, hard-WTA | **82.3%** (300 neurons / 6k) |
| Conductance Diehl & Cook (GPU, RTX 5070) | BindsNET, exc/inh populations | **90.0% @ 1600n** (scale-tuned) · 86.4% @400n · 6400/default 47.8% |
| N-MNIST event-camera ingest | 1.25M real DVS events → Bronze → nearest-prototype | 71.0% (200/100, _no learning_) — a pipeline/plumbing demo, off-theme; see limitations |

> **The two "95%"s are different.** The robot-arm **gesture** 95% is _achieved_ on real data.
> The MNIST conductance-D&C **95%** is the _literature target_ (Diehl & Cook, 6400n + full 60k);
> best measured here is **90.0%**, and the tuned 6400n GPU run is currently blocked by a billing
> hold (below), not a method gap.

---

## Architecture

### The three data-system paradigms (all complete)

| Paradigm | Capability | Implementation | Since |
|----------|------------|----------------|:----:|
| **A** — telemetry hub | store + query multi-channel spike trains as sparse events (batch **and** streaming) | `snn_data_lake/spike_telemetry_hub.py`, `streaming_hub.py` | v0.12 |
| **B** — in-storage search | compile a query into an SNN, stream stored spikes, emit only matches — **coincidence + temporal order** | `snn_data_lake/paradigm_b_engine.py` (+ `research/paradigm_b_genn.py` GPU) | v0.20 |
| **C** — relational embeddings | a knowledge graph _in spike timing_: link prediction, anomaly scoring, full relation algebra | `research/spike_knowledge_graph*.py` (TransE / RotatE / algebra) | v0.21–25 |

### The robot-arm brain-stack (sense → think → act)

A biologically-motivated, layered controller — every box is a spiking network:

```
ESP32 sensor ─▶ encode ─▶ spike data lake ─▶ order-aware match ─▶ Interpreter ─▶ command ─▶ arm
                              ▲                     │                   │            (sim or real servos)
   reflex (ms, hardwired) ────┘  preempts cognition │                   ▼
   dopamine RPE (fast) ◀───── reward ◀── outcome ◀──┴──────── cortisol (slow stress state)
```

| Layer | Module | Role |
|-------|--------|------|
| Reflex (instinct) | `reflex.py` | sign-aware fast LIF on nociceptor channels — fires STOP/WITHDRAW _before_ the slow path |
| Reward learning | `valence_stdp.py` | reward-prediction-error (dopamine) STDP → learned APPROACH/AVOID |
| Stress state | `cortisol.py` | slow tonic scalar → hypervigilance (lower reflex threshold), harder aversive learning, caution bias |
| Recognition | `signal_loop.py`, `gesture_recognition.py`, `learned_matcher.py` | encode → in-lake match (order-aware) → label; continual "grow-your-own-vocabulary" learning |
| Command | `interpreter.py`, `closed_loop.py` | matched label → concrete command (`JOINT_A_ROTATE`, `GRIPPER_CLOSE`, `EMERGENCY_STOP`) + reward feedback |
| Low-level control | `population_encoding.py` (RBF), `spiking_pid.py` (NEF) | population-coded spiking PID + trajectory-deviation comparator |
| Body | `arm_sim.py` | 2-link planar arm — exact stdlib FK+collision (CI-green) or optional headless PyBullet physics |
| Actuator (real HW) | `serial_arm.py`, `arm_bridge.py`, `deploy/command_arm.ino` | command-out to ESP32 servos, **fail-safe** (write error raises), refuses an uncalibrated port; UDP back-channel closes the split-process reward loop |

### Lakehouse + cloud path

Local **Medallion** topology (`lakehouse/medallion.py`): Bronze raw events → Silver
binned/aligned → Gold features (firing rate, population synchrony, inverse-compression ratio,
co-firing view), over columnar Parquet, with a `data_quality.gate()` blocking promotion to Gold;
a **Delta Lake PoC** (`lakehouse/delta_demo.py`) adds a transaction log + `OPTIMIZE` compaction +
time-travel. The **same path runs on GCP** (`gcp/`, `infra/`): **Phase 1 verified end-to-end** on
a live project — Terraform → GCS Bronze → Dataproc Serverless Medallion → BigLake/BigQuery
reproduced the local Gold exactly (synchrony_cv 0.399, burst channels 7 & 42). Pay-per-use, no
standing cluster.

---

## Clone & run

Python 3.11+. The core pulls **no runtime dependencies**:

```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
cd Spiking-Neural-Data-Lake
pip install -e .                 # editable install; core = zero runtime deps

snn-demo                                       # whole loop in one file: sensor → recognize → command → arm + reflex
snn-arm                                        # live loop on the builder's REAL captured gestures (95%)
snn-loop                                       # full closed loop: gesture → command → reward → learn
python -m snn_data_lake.gesture_recognition    # order-aware recognition + confusion table
```

**Optional extras** — opt in per path, or all at once:

```bash
pip install -e ".[arm]"     # pyserial — live ESP32 / real actuator (python -m snn_data_lake.live_arm --serial COM8)
pip install -e ".[lake]"    # polars + pyarrow + deltalake — Medallion + Delta PoC
pip install -e ".[gpu]"     # torch + snntorch + bindsnet + numpy — MNIST / GPU + spiking-MBRL research
pip install -e ".[nmnist]"  # tonic — real N-MNIST event data
pip install -e ".[genn]"    # pygenn — GeNN GPU ports (needs a C++ compiler + CUDA)
pip install -e ".[dev]"     # pytest + pytest-cov + ruff + pre-commit
pip install -e ".[all]"     # arm + lake + gpu + nmnist (genn excluded — needs a compiler)
```

### Tests & CI

Two complementary gates — a zero-dep smoke gate and a coverable suite:

```bash
python run_selfchecks.py                 # the 80 pure-stdlib module self-checks (exactly what CI runs)
pip install -e ".[dev]" && pytest        # unit + edge-case suite, 84% line coverage (--cov-fail-under=70)
ruff check .                             # lint
```

CI runs the self-checks on a **matrix of `{ubuntu-latest, windows-latest} × {3.11, 3.12, 3.13}`**
plus separate lint and coverage jobs. The 80-module list is canonical in
[`run_selfchecks.py`](run_selfchecks.py) (CI just runs it); the torch/GPU models are excluded by
design and live in `research/`.

### GPU

Verified on an RTX 5070 (Blackwell sm_120, `torch 2.11.0+cu128`):

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install bindsnet
python research/eth_mnist_bindsnet.py --gpu     # 400n/20k → 86.4% (verified)
# the verified 90% run (scale-tuned):
NORD_M=1600 NORD_TRAIN=60000 NORD_INH=60 NORD_THETA_PLUS=0.20 python research/eth_mnist_bindsnet.py --gpu
```

The scaling recipe: as neurons grow, _lower_ per-synapse inhibition and _raise_ theta_plus. A
naive 6400/default run collapses to 47.8%; the tuned 6400n + multi-epoch (~day-scale, best as
parallel Vertex jobs) is the ~95% leg — currently gated on billing (see limitations).

> **Full local + cloud runbook:** [docs/RUNNING.md](docs/RUNNING.md).
> **Hardware bring-up (BOM, wiring, firmware, safety):** [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md).
> **Prioritized engineering/research audit:** [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

---

## Repository layout

```
spiking-neural-data-lake/
  src/snn_data_lake/            SHIPPED PRODUCT — installable, zero-dep core (48 modules + lakehouse)
    spike_telemetry_hub.py        Paradigm A: sparse .spk store + windowed queries
    streaming_hub.py              streaming Paradigm A: live append + record-while-query (.spkl) + opt-in hot_motifs analytics
    spike_stream_sketch.py        Count-Min heavy-hitter sketch — fixed-memory hot-motif analytics for the stream
    short_term_plasticity.py      additive onset/novelty filter over a streamed channel (Tsodyks-Markram, opt-in)
    paradigm_b_matcher.py         Paradigm B: coincidence matcher (CPU reference)
    paradigm_b_engine.py          Paradigm B: coincidence + temporal-sequence query engine
    spike_preprocessing.py        deterministic latency encode + Van Rossum + Victor-Purpura (q) + Silver denoise
    proximity_index.py            hierarchical Victor-Purpura proximity index for spike queries (no Euclidean/HNSW)
    lif_neuron.py                 canonical leaky integrate-and-fire neuron (leak/threshold/reset/refractory)
    spike_slicer.py               spike-density event-stream segmenter (LIF boundary neuron, min-segment aware)
    is_snn_forward.py             depth-stable multi-layer LIF forward pass (rate fixed-point init, no BatchNorm)
    lt_gate.py                    dual-time-constant neuron: fast reset + slow non-reset memory trace (anti-forget)
    spike_pooling.py              max-pool for spike maps — keeps sparse high-freq spikes avg-pool would smear
    energy.py                     SynOps -> estimated Joules/inference vs a dense-ANN baseline (cited)
    drift_monitor.py              online drift/novelty watch (PSI + novelty rate) — warn before collapse
    htm_sequence_memory.py        HTM temporal memory + opt-in lake SequenceAnomalyMonitor (out-of-order = temporal anomaly)
    population_encoding.py        RBF/Gaussian population rate encoder
    signal_loop.py                real-time loop: encode → lake → match → Interpreter (+ continual learn)
    interpreter.py                matched label → robot command + outcome feedback
    closed_loop.py                full in-process loop: signal → reflex → match → reward → learn  (snn-loop)
    arm_config.py                 arm calibration in one place (distance bins, IR gate, SAFETY_CALIBRATED)
    reflex.py                     fast-path instinct (STOP/WITHDRAW before cognition)
    valence_stdp.py               reward-prediction-error (dopamine) STDP → APPROACH/AVOID
    bcm_plasticity.py             BCM competitive selectivity on the classifier shapes (opt-in)
    homeostatic_scaling.py        additive rate-stabilizer (multiplicative synaptic scaling, opt-in)
    intrinsic_plasticity.py       additive excitability self-tuner (Triesch, info-max, scale-invariant, opt-in)
    izhikevich_neuron.py          additive richer neuron option — RS/IB/CH/FS regimes (reflex LIF stays default)
    cortisol.py                   slow tonic stress-state neuromodulation
    gesture_recognition.py        order-aware recognition of the builder's real gestures (95%)
    arm_sim.py                    2-link planar arm (stdlib FK+collision, optional PyBullet)
    spiking_pid.py                NEF-style population-coded spiking PID controller
    cpg_oscillator.py             Matsuoka CPG rhythmic motor primitive — drives the arm in a safe bounded sweep
    ring_attractor.py             CAN angle integrator + opt-in spiking proprioception (encoder-free joint estimate, slip detect)
    live_arm.py                   live hardware loop: ESP32 → recognize → Interpreter → arm  (snn-arm)
    serial_arm.py                 real-servo actuator sink: fail-safe + SAFETY_CALIBRATED gate
    arm_bridge.py                 UDP OUTCOME back-channel (closes the split-process loop)
    sensor_csv_ingest.py          adapter for the builder's timestamped serial CSV
    make_sensor_dataset.py        synthetic sensor dataset (ultrasonic + IR)
    sensor_demo.py                end-to-end sensor demo
    data_quality.py               Gold-promotion gates (schema/encodability/immutability/sanity)
    snn_classifier.py             trainable spiking classifier (weights = long-term storage)
    nir_export.py                 export the trained classifier to a NIR graph (Loihi 2 / SpiNNaker 2 / Speck deploy)
    learned_matcher.py            learned classifier + Van Rossum novelty gate (hybrid matcher)
    rank_order_coding.py          gain-invariant recognition (Thorpe rank code) — robust to sensor gain / rig change
    tempotron.py                  supervised spike-timing gesture matcher (one tempotron per command)
    lca_sparse_coding.py          LCA sparse coding + opt-in LCAMatcher (sparse-coding recognition over the library)
    sparse_distributed_memory.py  distributed associative-recall matcher option (Kanerva SDM)
    spiking_som.py                additive SOM clustering backend for gesture discovery (opt-in)
    fly_hash.py                   FlyHash LSH index — fast candidate filter over stored windows (opt-in)
    demo.py                       the whole loop in one runnable file  (snn-demo)
    lakehouse/medallion.py        Bronze→Silver→Gold PoC (Parquet + polars)
    lakehouse/delta_demo.py       Delta Lake PoC (transaction log + OPTIMIZE + time-travel)

  research/                     QUARANTINED — benchmarks & experiments, kept for reproducibility (NOT the product)
    spiking_storage_prototype.py, test_prototype.py   factored associative memory + stress sweeps (zero-dep)
    hopfield_baseline.py                              classical Hopfield 0.138·N capacity baseline (zero-dep)
    dense_associative_memory.py                       modern/dense associative memory: super-linear capacity (zero-dep)
    spiking_vsa.py                                    hyperdimensional/VSA reasoning: bind/bundle/permute (zero-dep)
    resonator_network.py                              VSA factorization: decompose a bound product (zero-dep)
    mcr_hdc.py                                        modular-composite HDC: denser codebook, ~1.5x fewer bits (zero-dep)
    bayesian_snn.py                                   calibrated uncertainty via Bernoulli-masked passes (zero-dep)
    spiking_ssm.py                                    Liquid State Machine reservoir: temporal sequence memory (zero-dep)
    spike_compression.py                              spike-native lossless codec (delta+varint) vs gzip (zero-dep)
    event_quadtree_compression.py                     2-D quadtree event codec: exploits burst clustering (zero-dep)
    index_compression.py                              order-invariant id/edge-list compression (zero-dep)
    delay_learning.py                                 three-factor synaptic-delay learning (timing, not weight) (zero-dep)
    free_energy_control.py                            predictive-coding spiking control: sparse, noise-robust (zero-dep)
    loop_latency.py                                   end-to-end sensor→command latency p50/p95/p99 (zero-dep)
    temporal_coding_storage.py                        TTFS latency coding, 83.5× fewer ops (zero-dep)
    snn_moe_classifier.py                             spike-driven MoE, 0-param router (zero-dep)
    spike_knowledge_graph{,_rotate}.py, spike_kg_relations.py   Paradigm C: SpikE / RotatE / algebra (zero-dep)
    srm_neuron.py, stats_bounds.py, nmnist_ingest.py, make_results_plot.py   (zero-dep)
    snn_mnist_stdp*.py, snn_mnist_dc.py, eth_mnist_bindsnet.py               MNIST STDP / D&C (torch/GPU)
    snn_moe_stdp_mnist.py, snn_storage_core_snntorch.py, paradigm_b_genn.py  (torch/GPU)
    snn_motor_control.py, snn_world_model_rl.py, snn_dreamer_6dof.py         spiking-MBRL lineage (torch)
    neuromorphic_torch/           SpikeSlicer / IS-SNN / LT-Gate / Max-Former — torch+SpikingJelly subproject (own venv, not core/CI)

  gcp/          Dataproc / Vertex / BigQuery / BigLake / Pub-Sub-Dataflow scaffold + deploy guide (Phase 1 verified)
  infra/        Terraform — GCP-native lakehouse infrastructure (GCS, BQ, BigLake, Artifact Registry, Pub/Sub)
  deploy/       command_arm.ino — ESP32 actuator firmware (ECHO_ONLY default, fail-safe E-STOP)
  docs/         RUNNING.md · HARDWARE_SETUP.md · IMPROVEMENTS.md · STORAGE_FORMATS.md · arduino_contract.md
  run_selfchecks.py             canonical zero-dep self-check runner (80 modules)
  pyproject.toml                installable package, zero-dep core, optional extras, console scripts
```

---

## Competitive context

- **vs dense ANNs** — a synaptic op only when a neuron fires: **3.6–83.5× fewer operations**
  here, and 1-bit spike events vs 32-bit activations.
- **vs rate coding** — deterministic temporal (TTFS) coding cuts inference ~83×; latency STDP
  trains 2.1× faster at 7.9× fewer SynOps (measured −6.2 pt accuracy tradeoff).
- **vs the literature** — unsupervised STDP MNIST tops out at ~95% (Diehl & Cook 2015, 6400n +
  60k). This repo reaches **82.3% CPU** (300n) and **90.0% GPU** (1600n, scale-tuned); the
  6400n/~95% leg is a compute/ops task, not a method gap.
- **vs [Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model)**
  (a 1B-param pure-SNN LLM) — the same primitives (LIF, STDP, sparse WTA / firing-rate MoE,
  attractor memory) scaled down to small, _verifiable_ demos.
- **Path to neuromorphic silicon** — the in-storage query (Paradigm B) and the spiking control
  loop are architected for event-driven hardware (Loihi 2 / Lava, Akida — Intel's 1,152-Loihi-2 Hala
  Point reaches 15 TOPS/W INT8); on-chip, SynOps convert
  directly to a Joules-per-inference energy claim — the stdlib `energy.py` already estimates it from
  measured SynOps (≈18× for the supervised classifier under cited 45 nm constants: the 3.6× op-count
  win × the MAC/AC energy asymmetry).

---

## Roadmap

_Landed (v1.7, research):_ **Resonate-and-fire neuron** (`research/resonate_and_fire.py`) — a frequency-selective
spiking neuron: where LIF/Izhikevich/SRM integrate on amplitude, this resonates only near its natural frequency
(band-pass); a bank decomposes a 10+20 Hz signal to exactly {10, 20} — neuromorphic spectral analysis, no FFT.
Zero-dep, deterministic.

_Landed (v1.6, research):_ **Oja's rule PCA** (`research/oja_pca.py`) — a single linear neuron whose weight
converges to the top principal component via a local Hebbian + self-normalizing update (|cos| 0.999, ‖w‖ ≈ 1);
Sanger's rule extracts the top-2 in order. The neuromorphic route to unsupervised feature learning, distinct
from BCM selectivity. Zero-dep, deterministic.

_Landed (v1.5, research):_ **Spike Bloom filter** (`research/spike_bloom.py`) — fixed-memory pattern membership
("have I seen this exact pattern?"): 500 patterns in 4096 bits, 0 false negatives, FP 0.021 ≈ the 0.022 analytic
bound. The membership counterpart to v1.3's Count-Min frequency sketch. The research→core integration pass
reached its clean limit at 4 slices (v1.1–v1.4); per the no-force policy this cycle self-refuels with new research.

_Landed (v1.4, integration):_ **LCA sparse coding → core** — the v0.96 Locally Competitive Algorithm promoted
into the product as `LCAMatcher`: an opt-in sparse-coding recognizer that uses the enrolled command references as
its dictionary and reads the command off the dominant atom (clean 5/5, jittered 100%) — a matcher option beside
rank-order/tempotron, recognition-only. Fourth slice of the research→core pass.

_Landed (v1.3, integration):_ **Count-Min sketch → core** — the v0.98 heavy-hitter sketch promoted into the
product as `StreamingHub.hot_motifs(window, top_k)`: a read-only, fixed-memory analytics method surfacing the
most frequent channel-coactivation motifs over the live stream (a recurring `{1,4,7}` co-activation is recovered
amid background noise). Additive; append/query/replay untouched. Third slice of the research→core pass.

_Landed (v1.2, integration):_ **HTM temporal memory → core** — the v0.97 HTM promoted into the product with
`SequenceAnomalyMonitor`: a read-only lake observer that runs over the product encoder (`encode_latency`), learns
the normal window order, and flags a valid signal arriving out of sequence as a temporal anomaly (trained
transitions 0.00, out-of-order burst 1.00) — complements `drift_monitor`. Second slice of the research→core pass.

_Landed (v1.1, integration):_ **Ring attractor → core** — the v0.95 CAN promoted from research into the zero-dep
product core with `ProprioceptiveEstimator`: a read-only spiking observer that integrates the arm's commanded
joint deltas into an encoder-free angle estimate and flags uncommanded joint slip (tracks a live `ArmSim` to
0.2°, catches a 30° disturbance) — additive, safety-envelope untouched. First slice of the research→core pass.

_Landed (v1.0, research):_ **NEF integrator** (`research/nef_integrator.py`) — the third NEF principle,
dynamics: a recurrent LIF population becomes a neural integrator / line attractor that HOLDS its state (working
memory, drift −0.004 over 30 steps) and integrates input over time. Completes the NEF trio (represent → transform
→ dynamics). Zero-dep, deterministic.

_Landed (v0.99, research):_ **NEF compute** (`research/nef_compute.py`) — the Neural Engineering Framework:
a heterogeneous LIF population represents a scalar (decode RMSE 0.012) and computes a nonlinear function of it
(x², RMSE 0.030) from the same spikes by choosing least-squares decoders — the spiking-computation substrate
(Nengo/Loihi). Zero-dep, deterministic.

_Landed (v0.98, research):_ **Spike-stream sketch** (`research/spike_stream_sketch.py`) — a zero-dep Count-Min
Sketch + top-K tracker: heavy-hitter **motif detection** over an unbounded spike-pattern stream in fixed memory
(3 frequent motifs recovered from 1503 distinct patterns in 1024 counters; one-sided error within ε·N). The
data-lake / streaming-analytics counterpart to the neural modules. Zero-dep, deterministic.

_Landed (v0.97, research):_ **HTM temporal memory** (`research/htm_sequence_memory.py`) — Numenta high-order
sequence prediction over SDRs: the store learns to predict the next spike-pattern from temporal context, telling
apart sequences that share a middle (`ABC`→D, `XBC`→E, both 1.00/0.00) where a first-order model can't; novel
transitions burst as an anomaly signal. Zero-dep, deterministic.

_Landed (v0.96, research):_ **LCA sparse coding** (`research/lca_sparse_coding.py`) — the Rozell Locally
Competitive Algorithm as a lateral-inhibition network: leaky integration + thresholding + competition find a
sparse code over an overcomplete dictionary (exact 3-atom support recovery, 3-sparse vs a dense 11-atom
correlation code). Zero-dep, deterministic — the encoder that produces the sparsity the store recalls over.

_Landed (v0.95, research):_ **Ring attractor** (`research/ring_attractor.py`) — a continuous attractor network
that holds a persistent activity bump (analog angle memory) and integrates angular velocity into absolute angle
(linear R^2 1.000, out-and-back returns to 0.0 deg): encoder-free **joint-state dead-reckoning** for the arm.
Zero-dep, deterministic. A future opt-in proprioceptive estimator for a revolute joint (arXiv:2604.14021).

_Landed (v0.94, research):_ **Polychronization** (`research/polychronization.py`) — memory stored in spike
**timing**: a `DelayNetwork` of `LIFNeuron` coincidence detectors with per-synapse conduction delays holds
multiple Izhikevich-style Polychronous Neuronal Groups over one shared anchor pool; timing alone (not rate, not
weights) selects which group ignites — the network-scale form of the "reason in spike timing" thesis. Zero-dep.

_Landed (v0.93, stdlib core):_ **Neuromorphic phases in pure Python** — the zero-dep twin of the v0.92 torch
subproject, as CI-covered core modules: `lif_neuron` (canonical LIF), `spike_slicer` (density-based stream
segmenter), `is_snn_forward` (depth-stable multi-layer forward pass, rate fixed-point init, no BatchNorm),
`lt_gate` (dual-timescale anti-forgetting), `spike_pooling` (high-freq-preserving max-pool). Reuses the
existing Van Rossum / Victor-Purpura / TTFS / population encoders. 70/70 self-checks, core stays zero-dep.

_Landed (v0.92, research/torch — quarantined):_ **Trained-SNN subproject** (`research/neuromorphic_torch/`)
— the framework phases of the neuromorphic-DB brief as an isolated torch+SpikingJelly subproject (own venv,
NOT core deps / CI): SpikeSlicer+SPA-Loss, IS-SNN (native surrogate-gradient STBP, no ANN->SNN, cupy-default
w/ CPU fallback), LT-Gate dual-timescale anti-forgetting, Max-Former max-pool anti-low-pass. All
CPU-smoke-verified (`smoke_check.py`); reuses the stdlib core encoders/metrics; core stays zero-dep.

_Landed (v0.91, stdlib):_ **Victor-Purpura distance + hierarchical proximity index**
(`spike_preprocessing.victor_purpura_distance`, `proximity_index.py`) — a tunable-q spike-train edit
distance (q->0 = rate/count, large q = coincidence) + a medoid-hierarchical VP index (recall@1 100%,
scans ~25% of patterns), no Euclidean/HNSW/IVF. Also scaffolds `research/neuromorphic_torch/` — a
quarantined torch+SpikingJelly subproject (own venv, NOT core deps, NOT CI) for the trained-SNN phases
(SpikeSlicer / IS-SNN / LT-Gate / Max-Former). Reuses the existing stdlib Van Rossum / TTFS / population.

_Landed (v0.90, product integration):_ **Izhikevich neuron promoted into the product**
(`src/snn_data_lake/izhikevich_neuron.py`) — an additive richer neuron option reproducing the RS / IB /
CH / FS cortical firing regimes from a 2-equation model; available to the arm/reflex path, with the
safety-critical reflex LIF unchanged as the default. Completes the "primitives into the main program"
arc (8 research primitives integrated, v0.83–v0.90; all additive, defaults/safety/determinism intact).

_Landed (v0.89, product integration):_ **Intrinsic plasticity promoted into the product**
(`src/snn_data_lake/intrinsic_plasticity.py`) — an additive Triesch excitability self-tuner: driven by
scalar projections of noisy classifier shapes it pulls the output mean to ~mu (0.50->0.16) and holds it
under a 3x input scale (gain 1.34->0.43). Optional; no default changes.

_Landed (v0.88, product integration):_ **Homeostatic scaling promoted into the product**
(`src/snn_data_lake/homeostatic_scaling.py`) — an additive multiplicative rate-stabilizer: driven by a
real classifier shape it holds the rate at 1.08 while a Hebbian-only control runs away to ~4.5e11,
preserving relative weights (drift 6e-27). Optional tool; the product's learners already self-stabilize,
so no default changes.

_Landed (v0.87, product integration):_ **BCM plasticity promoted into the product**
(`src/snn_data_lake/bcm_plasticity.py`) — `develop_selectivity` runs BCM's sliding-threshold rule on
the classifier's distinct shape inputs; one unit self-organizes to a single shape (selectivity
0.11->0.67, winner 1.99 vs 0.25). Additive/opt-in; the recognition path is unchanged.

_Landed (v0.86, product integration):_ **Short-term plasticity promoted into the product**
(`src/snn_data_lake/short_term_plasticity.py`) — `novelty_filter` runs a streamed channel through a
Tsodyks-Markram depressing synapse so onsets ring out (0.48) and sustained bursts fade (0.04); an
additive, opt-in onset detector over `streaming_hub` channels — the default store/query path is
unchanged.

_Landed (v0.85, product integration):_ **FlyHash index promoted into the product**
(`src/snn_data_lake/fly_hash.py`) — `FlyHashIndex` returns the most tag-similar stored windows for a
query (bio-inspired LSH, recall@1 100%, same/different-command tag overlap 59.6 vs 2.5); an additive,
opt-in candidate filter — the default matcher/cluster scan is unchanged.

_Landed (v0.84, product integration):_ **Spiking SOM promoted into the product**
(`src/snn_data_lake/spiking_som.py`) — `som_cluster` is an additive Kohonen clustering backend for
gesture discovery (groups unknown windows by BMU at 100% purity); opt-in, the default single-link
`cluster_unknowns` path is unchanged.

_Landed (v0.83, product integration):_ **SDM matcher promoted into the product**
(`src/snn_data_lake/sparse_distributed_memory.py`) — `SDMMatcher` recognizes a window by Kanerva
distributed associative recall of the enrolled command id: 100% on jittered windows. An additive,
noise-robust matcher option (not the live default).

_Landed (v0.82, product integration):_ **Tempotron matcher promoted into the product**
(`src/snn_data_lake/tempotron.py`) — `TempotronMatcher` enrolls one supervised spike-timing tempotron
per command on the live library and recognizes jittered windows at 100%; a third matcher mechanism
(spike timing) beside the distance and rate readouts.

_Landed (v0.81, product integration):_ **NIR export promoted into the product**
(`src/snn_data_lake/nir_export.py`) — `export_classifier` maps the trained snn_classifier weights to a
NIR graph Input(64)->Linear->LIF(4)->Output(4), validated + round-tripped, portable to Loihi 2 /
SpiNNaker 2 / Speck / Xylo. The neuromorphic-deploy path (roadmap C1) now exports the real recognizer.

_Landed (v0.80, product integration):_ **Spiking CPG promoted into the product**
(`src/snn_data_lake/cpg_oscillator.py`) — a Matsuoka half-center CPG now drives ArmSim as a rhythmic
motor primitive: `cpg_sweep` sweeps joint A smoothly through a bounded 29..61 deg band via the
safety-gated command API (collision + EMERGENCY_STOP still apply), 14 reversals, collision-free. The
robot-arm counterpart to v0.79's data-lake integration.

_Landed (v0.79, product integration):_ **Rank-order coding promoted into the product**
(`src/snn_data_lake/rank_order_coding.py`) — a gain-invariant `RankOrderMatcher` enrolled on the live
command library: 100% recognition, exactly invariant under a per-capture rig-gain change where the
Van Rossum template matcher drops to 92%. First step of building the research primitives into the
data-lake / arm program.

_Landed (v0.78, zero-dep):_ **Intrinsic plasticity** (`intrinsic_plasticity.py`) — Triesch's rule: a
neuron tunes its own gain/bias to drive its output distribution toward an exponential (info-max)
target, pulling the output mean to 0.15 and holding it under a 3x input scale (gain 1.27->0.35). The
intrinsic-excitability counterpart to synaptic homeostatic scaling.

_Landed (v0.77, zero-dep):_ **Short-term plasticity** (`short_term_plasticity.py`) — Tsodyks-Markram
dynamic synapses: each spike transiently depletes resources (depression) and raises release
probability (facilitation), making a synapse a temporal filter. Depressing synapse fades to 0.05x
over a train; facilitating builds to 1.98x; recovers to baseline after a gap.

_Landed (v0.76, zero-dep):_ **BCM plasticity** (`bcm_plasticity.py`) — the Bienenstock-Cooper-Munro
sliding-threshold rule: a neuron self-organizes to respond to ONE input pattern (selectivity
0.15->0.67, winner 2.66 vs 0.27) via a modification threshold that chases its own activity. The
classic rate-based selectivity / feature-learning rule.

_Landed (v0.75, zero-dep):_ **Izhikevich neuron** (`izhikevich_neuron.py`) — the 2-equation model
reproducing cortical firing regimes (RS/IB/CH/FS) from four params at ~13 flops/step: fast-spiking 38
vs regular-spiking 8 spikes, chattering bursts at ISI-CV 1.63 vs 0.18. A richer, still-cheap neuron
primitive beside the repo's LIF.

_Landed (v0.74, zero-dep):_ **Homeostatic scaling** (`homeostatic_scaling.py`) — Turrigiano synaptic
scaling: a neuron multiplicatively rescales its synapses toward a rate set-point, holding the rate at
1.07 (vs Hebbian-only runaway to ~2.4e9) while preserving every relative weight ratio (drift 2e-16).
The missing stabilizer for the STDP learning stores.

_Landed (v0.73, zero-dep):_ **Spiking CPG** (`cpg_oscillator.py`) — a Matsuoka half-center
oscillator: two mutually-inhibiting adapting neurons give a stable anti-phase rhythm (corr -0.85)
for arm motion, with two motor knobs — adaptation time-constant sets tempo (23->40 cycles), tonic
drive sets vigor (amplitude 1.00->1.57). Pairs with spiking_pid + arm_sim.

_Landed (v0.72, zero-dep):_ **Rank-order coding** (`rank_order_coding.py`) — Thorpe's gain-invariant
code: classify by the ORDER afferents fire, not magnitude. 100% clean and 100% after a monotonic
contrast change (exact invariance), where a magnitude baseline collapses 100%->67%. Distinct from
the TTFS latency store.

_Landed (v0.71, zero-dep):_ **Tempotron** (`tempotron.py`) — the Gutig-Sompolinsky supervised
spike-timing rule: one leaky neuron learns to fire vs stay silent on spatiotemporal spike patterns
via its peak-voltage gradient; 50%->100% accuracy, learned voltage margin 1.06>1.00>0.85. The
supervised-temporal counterpart to the unsupervised STDP models.

_Landed (v0.70, zero-dep):_ **Spiking SOM** (`spiking_som.py`) — a Kohonen self-organizing map
(WTA BMU + lateral neighbourhood) as an unsupervised topographic index over the store: quantization
error 0.49->0.07, topographic ratio 0.36, 100% BMU class purity on separable clusters. Completes the
store arc index->memory->organize (FlyHash / SDM / SOM).

_Landed (v0.69, zero-dep):_ **Sparse Distributed Memory** (`sparse_distributed_memory.py`) —
Kanerva's distributed content-addressable store (random hard-location addresses + Hamming-radius
activation + bit counters): clean recall 100%, cleans a 15%-corrupted cue to ~90%, degrades
gracefully — the distributed memory behind the FlyHash index.

_Landed (v0.68, zero-dep):_ **FlyHash** (`fly_hash.py`) — the fruit-fly olfactory LSH (sparse
expansion + WTA -> sparse binary tag) as an approximate-nearest-neighbour index over the store;
near-duplicate vs random tag-overlap 45.6 vs 2.4, recall@1 100% on 5%-jittered queries. Turns the
brute-force query scan into a candidate lookup with the repo's own primitives.

_Landed (v0.67, zero-dep):_ **NIR export** (`nir_export.py`) — emit the spiking models as a
Neuromorphic Intermediate Representation graph (JSON core + optional bridge to canonical NIR), so
Paradigm B/C export to Loihi 2 / SpiNNaker 2 / Speck / Xylo at once instead of a per-chip translator
(research scan #10, roadmap C1).

_Landed (v0.65–v0.66, zero-dep):_ modern-Hopfield energies incl. **log-sum-ReLU** (dense AM),
**resonator** VSA factorization, **MCR** HDC substrate, three-factor **delay learning**, **quadtree** +
order-invariant **index** compression, **Bayesian uncertainty**, **free-energy control**, NeuroBench
metrics — the cycle-3 research scan ([docs/RESEARCH_SCAN.md](docs/RESEARCH_SCAN.md)).

_Landed (v0.64, zero-dep):_ **live drift/novelty monitor** (D3, `drift_monitor.py`),
**end-to-end loop latency** p50/p95/p99 (D2, `loop_latency.py`), and a **spike-native lossless codec**
(`spike_compression.py`, ~5× vs gzip on the sparse store).

Still near-term, from the cited audit in [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md):

- **Unblock the ~95% GPU leg** — clear the GCP billing hold, run the tuned 6400n/L4 Vertex job
  with checkpoint/resume on Spot, write the result back as an artifact.
- **Surrogate-gradient (BPTT) MNIST baseline** via snnTorch — the ~98–99% supervised upper bound
  as a scientific control against unsupervised STDP.
- **DVS128 Gesture as the flagship event benchmark** (loadable via Tonic) — on-theme, replaces
  off-theme N-MNIST, and drives the same Interpreter→arm loop from a real event camera.
- **SynOps → Joules** *(analytical helper landed — `energy.py`)* — fold an energy column into
  the results table and validate against measured neuromorphic silicon.
- **Loihi 2 / Lava port of Paradigm B** — validate on the Lava CPU backend, then measure real
  silicon energy under INRC access.
- **Actuator safety hardening** — host↔arm heartbeat/dead-man timeout, command rate limit,
  firmware-side joint/velocity clamps.

---

## Limitations & open problems

Stated plainly — a researcher should be able to trust the numbers:

- **~95% MNIST is a target, not achieved.** Best measured is **90.0%** (1600n, scale-tuned). The
  6400n/~95% run needs the recipe + multiple epochs (day-scale); the tuned Vertex job is currently
  **blocked by a GCP billing "dunning" hold**, not a code gap.
- **The 6-DOF Dreamer's world model is a plain MLP.** A fully-spiking world model measurably
  failed to regress 6-joint FK (recon ~0.9 m); the **actor and critic are spiking**, the world
  model is not. The arm is a **planar** redundant arm, not a 3D manipulator.
- **Real-arm "learning" is partly simulated pending the builder.** Reward/success is an
  `--assume-success` placeholder and IR sensor fusion is **off** (`arm_config.IR_PRESENT = None`)
  until a calibration capture and a real success signal land; distance alone decides presence.
- **N-MNIST 71% is a plumbing demo, not a benchmark.** Real DVS data, nearest-prototype, no
  learning — off-theme vs the storage/robotics thesis (SOTA ~99.3%); the audit recommends
  replacing it with DVS128 Gesture.
- **The streaming cloud Bronze leg is unverified.** Batch Phase 1 is verified end-to-end; the
  Pub/Sub→Dataflow streaming sink still writes vanilla Parquet (small-file problem) — the Delta
  fix is proven _locally_, to be applied on first cloud streaming run.
- **Latency↔rate gap (−6.2 pts)** and **determinism is mandatory for the query path** (Poisson
  re-encoding gives different spike trains, so stochastic encoding can't be used for matching).
- **Neuromodulators are abstractions**, not biophysics — single global scalars for
  dopamine/cortisol, no D1/D2 or HPA-axis detail. **Paradigm B's real win needs neuromorphic
  silicon**; the GeNN GPU ports need CUDA + a C++ compiler and are drop-in accelerators, not
  independently re-validated.

Numbers are fixed-seed; rerun to reproduce.

---

## Provenance

Designs come from two research briefs (in [`research/`](research/)) surveying SNN data-storage
methods, plus an external architectural assessment that mapped the concept to the three paradigms.
The spike-driven MoE is a port of Project Nord's `SpikeDrivenMoE`. Every version is a git tag +
GitHub release ([CHANGELOG.md](CHANGELOG.md)). License: [MIT](LICENSE).
