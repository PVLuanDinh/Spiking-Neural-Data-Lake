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
| **Data-lake paradigms A / B / C** (store · in-storage search · relational reason) | ✅ complete, zero-dep; Paradigm-C **reason** ships in-core two ways — spike-timing (`paradigm_b_engine`) + VSA/HD algebra (`vsa_reasoning`, neuromorphic-adjacent, not spike-timing) |
| **Robot-arm neuromorphic control loop** (reflex · dopamine · cortisol · PID · interpreter) | ✅ complete in sim; real-arm actuator path shipped (safety-gated) |
| **Spiking model-based RL** (surrogate-gradient policy → world-model → 6-DOF Dreamer) | ✅ research lineage complete (needs torch, own venv) |
| **Local Medallion lakehouse** (Bronze→Silver→Gold, Parquet + Delta PoC) | ✅ complete |
| **GCP lakehouse Phase 1** (Terraform → Dataproc Serverless → BigLake/BigQuery) | ✅ **verified end-to-end on live GCP** |
| **GPU training (conductance D&C ~95% leg)** | ⏳ 90.0% reproduced (1600n); 6400n/~95% run **blocked by a GCP billing hold**, not a code gap |
| **Engineering** | ✅ zero-dep core · `pip install` · 107/107 self-checks · pytest 85% · ruff · CI matrix |

Current version: **v2.36** · 146 tagged releases (v0.1 → v2.36). See [CHANGELOG.md](CHANGELOG.md) for the full arc.

**⬇ Download (Windows, no Python needed):** [spiking-neural-data-lake-2.36.0-windows-x64.zip](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/releases/latest/download/spiking-neural-data-lake-2.36.0-windows-x64.zip) — unzip and double-click `snn-ui.exe`. **Use your own data + hardware:** replay bundled sample captures or **Open** your own; **Open** your own `.spk` or run the **Import wizard** (spike-event *or* dense sensor CSV → a queryable store); **Connect** a real serial arm (calibration-gated) to drive hardware from Arm Control; type NL commands; run Diagnostics. (Real-arm serial needs a source install: `pip install ".[arm]"`. Or clone: `python run_selfchecks.py`.)

The arm primitives ship as **opt-in, off-by-default** modes on `live_arm` — all routed through the safety-gated
`apply()`: `--proprioception` (read-only encoder-free joint estimate + slip watchdog), `--reach DEG` (spiking-PID
joint positioning), `--sweep` (CPG rhythmic bounded sweep). Defaults and the arm safety envelope are unchanged.

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
snn-ui                                         # zero-dep tkinter desktop control panel (arm + recognition + lake)
python -m snn_data_lake.gesture_recognition    # order-aware recognition + confusion table
```

### Desktop control panel

The product's full UX is a **zero-dependency tkinter desktop app** (tkinter ships with Python, so
the core stays dependency-free). One window with three tabs, each driving the already-verified code
paths — **Arm Control** (a live 2-link arm on a canvas; every button, including a red
EMERGENCY_STOP, routes motion through the safety-gated `ArmSim.apply()`), **Live Recognition** (step
/ auto-play a captured gesture through the same `classify → Interpreter → arm` path `live_arm` uses),
and **Lake Browser** (open a `.spk` store and run windowed `(channel, [t0,t1))` queries with a
"reads X% of the file" partial-read stat).

```bash
snn-ui                                          # launch the control panel
python -m snn_data_lake.control_panel --gui     # same, as a module
python -m snn_data_lake.control_panel --smoke   # headless-safe self-check (what CI runs; no display needed)
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
python run_selfchecks.py                 # the 107 pure-stdlib module self-checks (exactly what CI runs)
pip install -e ".[dev]" && pytest        # unit + edge-case suite, 84% line coverage (--cov-fail-under=70)
ruff check .                             # lint
```

CI runs the self-checks on a **matrix of `{ubuntu-latest, windows-latest} × {3.11, 3.12, 3.13}`**
plus separate lint and coverage jobs. The 107-module list is canonical in
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
  src/snn_data_lake/            SHIPPED PRODUCT — installable, zero-dep core (72 modules + lakehouse)
  local_llm/                    OPTIONAL local-first NL interface — AirLLM on-device backend ([llm] extra, no cloud)
  packaging/                    build the tkinter GUI into a standalone Windows .exe (PyInstaller spec + Inno Setup installer)
    spike_telemetry_hub.py        Paradigm A: sparse .spk store + windowed queries
    streaming_hub.py              streaming Paradigm A: live append + record-while-query (.spkl) + opt-in hot_motifs analytics
    spike_stream_sketch.py        Count-Min heavy-hitter sketch — fixed-memory hot-motif analytics for the stream
    short_term_plasticity.py      additive onset/novelty filter over a streamed channel (Tsodyks-Markram, opt-in)
    paradigm_b_matcher.py         Paradigm B: coincidence matcher (CPU reference)
    paradigm_b_engine.py          Paradigm B: coincidence + temporal-sequence query engine
    vsa_reasoning.py              Paradigm C reason: VSA/HD bind·bundle·permute compositional queries (dense HD vectors, NOT spike-timing; complements paradigm_b_engine)
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
    multimodal_encoder.py         fuse visual/temperature/audio/distance into the N=64 banded spike substrate (reuses population_encoding + encode_latency)
    growing_learner.py            online growing prototype learner — novelty-triggered neurogenesis over the fused multimodal spikes (ART-like, reuses encode_multimodal)
    autonomous_loop.py            autonomous adaptive arm — grow from multimodal data + learn actions by reward; every motion safety-gated (reflex/E-STOP)
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
    demo.py                       the whole loop in one runnable file  (snn-demo)
    control_panel.py              zero-dep tkinter desktop control panel: arm + recognition + lake tabs  (snn-ui)
    lakehouse/medallion.py        Bronze→Silver→Gold PoC (Parquet + polars)
    lakehouse/delta_demo.py       Delta Lake PoC (transaction log + OPTIMIZE + time-travel)

  research/                     QUARANTINED — benchmarks & experiments, kept for reproducibility (NOT the product)
    spiking_storage_prototype.py, test_prototype.py   factored associative memory + stress sweeps (zero-dep)
    hopfield_baseline.py                              classical Hopfield 0.138·N capacity baseline (zero-dep)
    dense_associative_memory.py                       modern/dense associative memory: super-linear capacity (zero-dep)
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
  run_selfchecks.py             canonical zero-dep self-check runner (81 modules)
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

_Landed (v2.36, product):_ **Efference copy / forward model — cancel reafference, flag exafference (von Holst / Wolpert).**
`forward_model.py` adds action-conditioned sensory prediction: an **efference copy** of the motor command feeds an
internal **forward model** that predicts the next sensory state `ŝ_{t+1} = s_t + dt·(â·s_t + b̂·u_t)`, and the
residual `e = s_{t+1} − ŝ_{t+1}` **cancels self-generated sensation** (reafference) while **flagging external
perturbations** (exafference) — the reafference principle (von Holst & Mittelstaedt 1950) + internal forward models
(Wolpert, Ghahramani & Jordan, *Science* 269:1880, 1995). Self-check on a self-contained abstract linear plant: a
matched model **cancels reafference** (mean self-generated error −2.9e-04, down to the real sensor-noise floor 4.9e-03);
an injected external perturbation **spikes the error to 0.198 vs a 0.005 baseline** and is flagged at exactly the
perturbed step; the **same sensory change (Δs=0.044)** is read as **self** (`e≈0`, not flagged) when commanded but as
**external** (`e=0.056`, flagged) when it comes from an equal unmodeled force — same Δs, opposite interpretation,
because only the command carries an efference copy; an online delta rule **learns a mismatched plant** (error 0.013 →
0.000007, 1876× lower; `â,b̂` converge to the true −0.3, 0.8); deterministic. It is a **read-only** predictor +
comparator — it emits no motor command and drives no hardware (the 5 safety-envelope files are neither touched nor
imported). Distinct from `predictive_coding` (static latent-cause inference, no action/efference copy), `spiking_pid`
(feedback control that emits a command), `ring_attractor`'s proprioceptive dead-reckoning (state integration, not
sensory-feedback prediction), and `drift_monitor` (statistical, not action-conditioned). Pure-stdlib core
(`dependencies = []`), **107/107** self-checks. Ceiling: a reduced linear forward model + comparator on a synthetic
plant — the efference-copy level, not a spiking motor-control network, not cognition.

_Landed (v2.35, product):_ **Astrocyte / tripartite synapse — slow, local, activity-driven glial gain (Araque).**
`astrocyte_modulation.py` adds the **third element** of the synapse — an astrocyte that wraps a group of synapses,
slowly integrates their local activity into intracellular calcium, and once Ca crosses a threshold releases
gliotransmitters that scale synaptic efficacy on a slow timescale (Araque, Parpura, Sanzgiri & Haydon,
*Trends Neurosci* 22:208, 1999; Volterra & Meldolesi, *Nat Rev Neurosci* 6:626, 2005). Per-domain Ca
`ca += ca_rise·activity − ca_decay·ca` (τ_ca = 50 ticks ≫ the LIF membrane τ_m = 10); a threshold gates
gliotransmission; a potentiating gain `1 + gain·(ca − ca_thresh)` follows Ca slowly and relaxes slowly back to 1.
Self-check: **sustained** high activity ramps Ca past threshold (crossing at tick 34 — a slow integrator, not a
per-spike response) and engages the gain to **1.52**, while a **brief** burst stays subthreshold (gain **1.00**); the
gain **persists 1.49 fifteen ticks after the drive stops** (slow); it is **local** — an active domain (gain 1.50) and
a quiet domain (gain 1.00) coexist in the same astrocyte, which a global scalar cannot represent; and it changes
downstream transmission (a LIF fires **5 → 13** spikes under the modulated gain); deterministic. Distinct from the
**global scalar** neuromodulators `cortisol`/`acetylcholine` (one value for the whole network), the **fast per-spike**
`short_term_plasticity` (Tsodyks-Markram), and the neuron's own `homeostatic_scaling`/`intrinsic_plasticity`
set-points. Pure-stdlib core (`dependencies = []`), **106/106** self-checks; the 5 safety-envelope files untouched.
Ceiling: a reduced slow-Ca + threshold-gliotransmission model on synthetic activity — phenomenological glial gain,
not IP₃/Ca-induced-Ca-release biophysics, no astrocyte network, not cognition.

_Landed (v2.34, product):_ **Synfire chain — stable feedforward propagation of a synchronous spike volley (Abeles / Diesmann).**
`synfire_chain.py` adds a precise-timing network phenomenon: a feedforward chain of coincidence-neuron pools through
which a synchronous **pulse packet** (`a` spikes with temporal jitter `σ`) either **propagates** to a stable
attractor or **extinguishes** — the landmark result that cortex can transmit precisely-timed synchronous activity
(Abeles, *Corticonics*, 1991; Diesmann, Gewaltig & Aertsen, *Nature* 402:529, 1999). Each pool neuron leaky-integrates
the previous pool's spike times (the leak *is* the coincidence window) and fires at its threshold crossing; the output
timing jitter follows the membrane slope, so coincident volleys sharpen and dispersed ones smear. Self-check (12 pools
× 100 neurons) shows the **survival/extinction separatrix**: a strong synchronous packet `(a0=90, σ0=3)` propagates to
**full activation (100/100)** while its jitter **self-sharpens 2.91 → 0.01** (locking to an attractor, tail activity
constant); a weak packet `(a0=35, σ0=3)` **extinguishes to 0** at layer 1; and — the key point — a strong-but-**smeared**
packet `(a0=90, σ0=9)` **also dies (0)**, so it is **synchrony, not spike count, that carries the packet**;
deterministic. Distinct from `kuramoto_sync` (phase-oscillator coherence, an order parameter — not a propagating spike
packet), `neuronal_avalanches` (a stochastic power-law cascade — the opposite of a stable locked volley), and the
research delay-based `delay_learning`/`polychronization` (matched conduction delays, not zero-lag feedforward
synchrony). Pure-stdlib core (`dependencies = []`), **105/105** self-checks; the 5 safety-envelope files untouched.
Ceiling: a reduced LIF-pool coincidence chain on a homogeneous feedforward chain (no recurrence/plasticity) — the
packet-propagation level of synchrony, not biophysics, not cognition.

_Landed (v2.33, product):_ **Neuronal avalanches — critical branching + scale-free cascade statistics (Beggs-Plenz).**
`neuronal_avalanches.py` adds a network-level *criticality* axis: a Galton-Watson branching process where each
active unit spawns `Binomial(fanout, σ/fanout)` descendants (mean = the **branching ratio σ**), and an avalanche is
a cascade from one seed unit until it dies out — its **size** and **duration** are the observables (Beggs & Plenz,
*J Neurosci* 23:11167, 2003). σ is the control parameter driving a **subcritical → critical → supercritical**
transition. Self-check (N=1000, 20k avalanches): the **measured** branching ratio matches the set σ (0.807 / 1.001 /
1.200, err < 0.05 — so σ genuinely controls the dynamics); mean avalanche size grows with σ
(**5.2 ≪ 117.7 ≪ 1758.1**); at criticality the size distribution is an approximate **power law** — a log-log density
slope of **−1.509** over `s∈[4,256]`, right on the mean-field exponent **−1.5** — and the critical tail is **27.6×**
the subcritical max (150 → 4134); deterministic. Distinct from `wilson_cowan` (deterministic E/I rate ODE with
attractors — no stochastic branching, no avalanche statistics, no criticality knob) and `kuramoto_sync` (phase-
oscillator synchronization — a coherence transition, not cascade statistics). Pure-stdlib core (`dependencies = []`),
**104/104** self-checks; the 5 safety-envelope files untouched. Ceiling: a reduced mean-field Galton-Watson process
on a homogeneous network (no spatial structure, no synaptic weights, no refractory dynamics) — the avalanche-
statistics level of criticality, not spiking biophysics, not cognition.

_Landed (v2.32, product):_ **Evidence accumulation — perceptual decision-making by drift-diffusion (Gold-Shadlen).**
`evidence_accumulation.py` adds a *decision-making* capability the core lacked: noisy sensory evidence is
**integrated over time to a decision bound**, and the crossing yields both a **choice** and a **reaction time** —
the algorithmic drift-diffusion / bounded-accumulator model of two-choice perceptual decisions (Ratcliff, *Psychol
Rev* 85:59, 1978; Gold & Shadlen, *Annu Rev Neurosci* 30:535, 2007; the spiking realization is Wang, *Neuron* 36:955,
2002). `DriftDiffusion(drift, bound, noise, dt, seed)` integrates `x += drift·dt + noise·N(0,1)·√dt` (Euler-Maruyama)
until `|x| ≥ bound`; a `RaceAccumulator` gives the N-alternative leaky-competing-accumulator race. Self-check shows
the three defining signatures: a **psychometric** curve (accuracy rises with evidence strength — **0.53 at zero drift
→ 1.00 at strong drift**, monotone); a **speed-accuracy tradeoff** (fixed drift, raising the bound buys accuracy at
the cost of latency — low bound `(acc 0.75, RT 26.8)` vs high bound `(acc 0.97, RT 139.1)`); a **chronometric** curve
(stronger evidence is decided faster — RT `98.9 → 30.0` steps); and the strongest of three racers wins the majority
(174/300); deterministic. Distinct from `valence_stdp`/`closed_loop` (value-based action selection — *learns* a value
via reward RPE, no time-integration/bound/RT/speed-accuracy) and `spiking_pid` (continuous control to a setpoint, not
a discrete commitment). Pure-stdlib core (`dependencies = []`), **103/103** self-checks; the 5 safety-envelope files
untouched. Ceiling: a reduced Euler-Maruyama drift-diffusion model on synthetic evidence — the algorithmic level of
the decision, not a Wang-2002 spiking attractor network, not cognition.

_Landed (v2.31, product):_ **Wilson-Cowan neural mass — E/I population dynamics + bifurcation repertoire.**
`wilson_cowan.py` adds a mesoscale dynamical-systems axis: two coupled mean firing rates, E (excitatory) and I
(inhibitory), each relaxing toward a sigmoid of its net input (Wilson & Cowan, *Biophys J* 12:1, 1972):
`τ_E·dE/dt = −E + f(w_EE·E − w_EI·I + P)`, `τ_I·dI/dt = −I + f(w_IE·E − w_II·I + Q)`. The **same** equations, under
three parameter presets, exhibit three qualitatively different attractors — the model's hallmark. Self-check (forward
Euler, `[0,1]²` invariant so rates stay bounded): a **stable fixed point** (6 random initial conditions all converge
to one steady `(E,I)=(0.065,0.032)`, spread `3e-17`, final `|dE/dt|=6e-17`); a **limit-cycle oscillation** (E→I→E
loop with slower inhibition — tail peak-to-peak amp **0.934**, 21 mean-crossings, bounded in [0.02,0.95], mid-run amp
= end amp so it's sustained not decaying, and dt-robust so not an Euler artifact); and **bistability + hysteresis**
(strong E self-excitation — ramping the drive P up vs down traces two branches, max gap **0.976** at P=2.0, 8 of 25 P
values with two coexisting stable states = a history-dependent switch/memory). Distinct from `cpg_oscillator` (a
Matsuoka half-center — one *designed* locomotor rhythm with an adaptation variable), `kuramoto_sync` (*phase*
oscillators + synchronization order parameter), and `divisive_normalization` (an *instantaneous* algebraic gain, no
dynamics). Pure-stdlib core (`dependencies = []`), **102/102** self-checks; the 5 safety-envelope files untouched.
Ceiling: the classic 2-population mean-field ODEs with a fixed Euler step and hand-picked presets — not spiking
biophysics, no spatial extent/delays, not cognition.

_Landed (v2.30, product):_ **Hierarchical predictive coding — perceptual inference by prediction-error (Rao-Ballard).**
`predictive_coding.py` adds the second dominant theory of cortical inference (alongside v2.28's neural sampling): a
2-level generative hierarchy (Rao & Ballard, *Nat Neurosci* 2:79, 1999; Friston free-energy view) where each level
**predicts the level below and only the prediction error propagates up**. With causes r1, r2 and generative weights
W1, W2: `x_hat = W1·r1`, `e0 = x − x_hat`; `r1_hat = W2·r2`, `e1 = r1 − r1_hat`. **Inference** descends the total
error energy `E = |e0|² + |e1|²` (`r1 += lr(W1ᵀe0 − e1)`, `r2 += lr·W2ᵀe1`) until it settles; **learning** then takes
a Hebbian step on the weights (`W1 += outer(e0,r1)`, `W2 += outer(e1,r2)`) — it learns a generative dictionary.
Self-check: for a **representable** input inference drives `E → 5.4e-09` monotonically (recon MSE 5.3e-10); the
hierarchy is genuinely 2-level — with a matched top-down prediction (`e1=0`) `r2` is **frozen (moves 0.0)**, but a
residual `e1` **moves it (0.039)**; **learning lowers** the per-epoch reconstruction MSE `0.145 → 0.030` (4.8×,
monotone); deterministic. Distinct from `lca_sparse_coding` (single-layer lateral inhibition — no top-down/bottom-up
hierarchy), `valence_stdp`/`r_stdp` (*reward* prediction error, not *sensory*), and `htm_sequence_memory` (temporal
next-SDR prediction, not a static generative hierarchy). Pure-stdlib core (`dependencies = []`), **101/101**
self-checks; the 5 safety-envelope files untouched. Ceiling: a small linear 2-level Rao-Ballard net on synthetic
data (gradient inference + Hebbian learning), tiny dims so the representable-input test is exact — not spiking
biophysics, no precision/attention weighting, not cognition.

_Landed (v2.29, product):_ **Kuramoto synchronization — emergent phase-locking of coupled oscillators.**
`kuramoto_sync.py` adds the canonical model of collective synchronization (Kuramoto 1975; Strogatz, *Physica D*
143:1, 2000): N phase oscillators with seeded natural frequencies, all-to-all sine-coupled, whose collective
coherence is read off the **order parameter** `r·e^{iψ} = (1/N)Σⱼe^{iθⱼ}` (`r∈[0,1]`: 0 = incoherent, 1 =
phase-locked). Above a **critical coupling** `K_c` the population spontaneously synchronizes; below it, `r` stays at
the finite-N floor `~1/√N`. Integrated in the exact mean-field form `dθᵢ/dt = ωᵢ + K·r·sin(ψ−θᵢ)` (an exact
identity for all-to-all coupling, O(N)/step). Self-check (N=256, Gaussian spread, `K_c≈1.60`): the **phase
transition** — `mean_r=0.080` at low `K` (near the 0.062 floor) vs **0.976** at high `K`; steady-state `r` is
**monotone in K** across a sweep spanning `K_c` (`[0.070, 0.073, 0.104, 0.930, 0.986]`); **identical oscillators
fully lock** (`r=1.000`); at high `K` the instantaneous-frequency spread **collapses to 0** (entrainment to a common
frequency); deterministic. Distinct from `cpg_oscillator` (a Matsuoka half-center — a designed 2-cell rhythm
*generator*), `ring_attractor` (a single bump), and `theta_gamma_code` (discrete phase slots): this is *emergent*
population synchronization (a phase transition), the substrate for neural-oscillation coherence /
communication-through-coherence. Pure-stdlib core (`dependencies = []`), **100/100** self-checks; the 5
safety-envelope files untouched. Ceiling: reduced mean-field phase oscillators (fixed Euler, all-to-all, synthetic
frequencies) — not biophysical spiking, no delays/topology, not cognition.

_Landed (v2.28, product):_ **Neural sampling — probabilistic inference in spikes (Buesing/Maass).**
`neural_sampling.py` adds a genuinely new capability class — **posterior sampling from a defined generative joint**.
A recurrent network of stochastic spiking neurons z∈{0,1}ᴺ with symmetric weights implements the neural-sampling
hypothesis (Buesing, Bill, Nessler & Maass, *PLoS Comput Biol* 7:e1002211, 2011): each single-site update spikes
neuron i with probability `σ(u_i)`, `u_i = b_i + Σ_j W_ij z_j` — exactly the Boltzmann conditional
`p(z_i=1 | z_rest)` — so the chain's stationary distribution **is** the Boltzmann joint
`p(z) ∝ exp(Σ b_i z_i + ½ Σ W_ij z_i z_j)`; the spike train is an MCMC sample of it. Clamping units to evidence
turns the same chain into **posterior sampling** `p(z_free | z_clamped)` — Bayesian inference computed by spiking.
Self-check on a 3-neuron net (2³ states enumerated for the exact distribution): the empirical distribution matches
Boltzmann to **TV 0.0066** (< 0.03; 40 000 samples), marginals + pairwise `E[z₀z₁]` match (err ≤ 0.0066), a clamp
`z₂=1` recovers the exact posterior (**conditional TV 0.0046**, clamped unit provably fixed), deterministic.
Distinct from `research/bayesian_snn` (MC-dropout ensemble-variance *uncertainty bars* — no defined joint): this is
the opposite object, a defined Boltzmann `p(z)` sampled by the dynamics. Pure-stdlib core (`dependencies = []`),
**99/99** self-checks; the 5 safety-envelope files untouched. Ceiling: the abstract discrete-time
stochastic-binary-neuron chain (single-site Gibbs, sigmoid firing) — reduced Buesing 2011, no refractory
continuous-time dynamics / biophysics / learning; small nets so the exact distribution is enumerable.

_Landed (v2.27, product):_ **Grid cells — 2D path integration on a hexagonal lattice (entorhinal cortex).**
`grid_cells.py` implements the oscillatory-interference grid-cell model (Hafting, Fyhn, Molden, Moser & Moser,
*Nature* 436:801, 2005; Burgess, Barry & O'Keefe, *Hippocampus* 17:801, 2007). A grid cell fires on a **hexagonal
lattice** of locations as an agent crosses a plane: the rate field is the interference of three plane waves whose
wave-vectors point **60° apart** (`|b_k| = 4π/(√3·λ)`), so the nearest-peak spacing is exactly the grid period λ,
rotated by the orientation and translated by the spatial phase. `step(velocity, dt)` accumulates 2D position, so
integrating a velocity trajectory **dead-reckons** the true displacement; a `GridModulePopulation` at ≥2 spacings
mirrors the multi-scale entorhinal modules. Self-check (measured from the rate map, not assumed): the 6 nearest
peaks sit at **mean distance 0.997 vs set λ=1.00** and **60.0° apart** (hexagonal); a straight leg lands at `v·t`
(err **2e-16**) and a closed square loop returns to the start (err **8e-17**); spacing/orientation/phase set the
measured geometry (0.6→0.598, +15°→rotates 15.0°, phase 0.2→translates 0.200, period unchanged); deterministic.
Distinct from `ring_attractor` (a single **1D** heading bump integrating angular velocity) — this is **2D** spatial
position on a periodic lattice. Pure-stdlib core (`dependencies = []`), **98/98** self-checks; the 5 safety-envelope
files untouched. Completes the spatial-navigation set: head-direction ring + grid path-integration + hippocampus.

_Landed (v2.26, product):_ **Theta-gamma code — ordered working memory by phase multiplexing (Lisman).**
`theta_gamma_code.py` implements the canonical theta-gamma neural code (Lisman & Idiart, *Science* 267:1512, 1995;
Lisman & Jensen, *Neuron* 77:1002, 2013): several working-memory items are multiplexed in time — each item is the
ensemble that fires in a distinct **gamma** sub-cycle, and one slow **theta** cycle nests K gamma cycles, so the
items are ordered by theta **phase** (item i at phase `2πi/K`). The gamma-per-theta ratio sets the capacity (~**7±2**).
Self-check: a sequence reads back **in order** by phase; phases are monotonic + distinct (position → phase); overflow
past K=7 **evicts the oldest** (`push 0..9 → keep [3..9]`); each gamma slot recovers its own item with distinct
ensembles (clean separation); deterministic. Distinct from `ring_attractor` (a single spatial bump) and
`htm_sequence_memory` (next-SDR prediction) — this holds *several* items at once and reads back their *order* by
phase. Pure-stdlib core (`dependencies = []`), **97/97** self-checks; the 5 safety-envelope files untouched.

_Landed (v2.25, product):_ **Acetylcholine — the encode-vs-retrieve neuromodulator (Hasselmo).** `acetylcholine.py`
completes the neuromodulator set (dopamine RPE in `valence_stdp`, tonic stress in `cortisol`, now ACh): a
novelty-driven **mode switch** between feedforward **encoding** (high ACh → afferent input enhanced, recurrent
suppressed, plasticity ON — write new memories) and recurrent **retrieval/consolidation** (low ACh → feedforward
attenuated, recurrent reactivation, plasticity off — recall stored). Self-check on a tiny Hebbian recurrent store:
the gains flip with ACh (`ff 0.99/rec 0.01/plast 0.99` → `ff 0.20/rec 1.00/plast 0.00`); a pattern **written under
high ACh is recalled from a 25%-degraded cue under low ACh (recall 1.00)**; ACh **habituates** (higher on novel than
familiar); writing a second pattern doesn't corrupt the first. Grounded in Hasselmo (*TiCS* 3:351, 1999) — directly
complements the v2.9 hippocampus DG/CA3. Distinct from `cortisol` (stress) and `valence_stdp` (reward). Pure-stdlib
core (`dependencies = []`), **96/96** self-checks; the 5 safety-envelope files untouched.

_Landed (v2.24, product):_ **Divisive normalization — the canonical gain-control computation (Carandini-Heeger).**
`divisive_normalization.py` adds fast, instantaneous pool gain control: each unit's response is its drive divided by
the summed activity of a normalization pool — `r_i = d_i^n / (σ^n + Σ_j d_j^n)` — the single operation Carandini &
Heeger (*Nat Rev Neurosci* 13:51, 2012) show recurs across V1/MT/LIP/attention/olfaction. Self-check shows the
canonical signatures: **saturation** (×8 input → winner grows only ×1.56 toward an asymptote), **contrast-invariant
tuning** (winner + rank order preserved, peak-normalized shape drift 7e-18 — the shared denominator cancels),
**cross-suppression** (a strong pool competitor drops a target 0.72 → 0.10), and a **bounded** pool output (<1). A
thin spiking view drives LIF units at the normalized gains. Distinct from `homeostatic_scaling` (slow multiplicative
set-point), `intrinsic_plasticity` (single-neuron self-gain), and `lca` lateral inhibition (subtractive, not
divisive). Pure-stdlib core (`dependencies = []`), **95/95** self-checks; the 5 safety-envelope files untouched.

_Landed (v2.23, product):_ **Dendritic neuron — two-compartment BAC firing (Larkum).** `dendritic_neuron.py` adds
the neuron family's first **nonlinear dendritic** unit (every other neuron is a point cell): a reduced
two-compartment pyramidal neuron (soma/basal + apical tuft) that does **coincidence detection** via BAC firing
("backpropagation-activated Ca²⁺ spike", Larkum, *Trends Neurosci* 2013; reduced models arXiv:2107.05336 /
arXiv:2311.06074). A somatic spike backpropagates a bump into the apical compartment; only when it lands on
**coincident** apical (top-down) input does `V_a` cross the Ca²⁺ threshold → a dendritic **plateau** boosts the soma →
a **burst**. Self-check shows the Larkum signature: basal-only fires at a max rate (**0.167**, no plateau), apical-only
is silent (**0.000**), and apical+basal coincident **bursts at 0.950 — 5.7× supralinear**, above what basal alone can
reach. Distinct from the point neurons (`lif`/`izhikevich`) and `htm` (binary segments, no plateau). Pure-stdlib core
(`dependencies = []`), **94/94** self-checks; the 5 safety-envelope files untouched. Ceiling: a reduced rate/spike
caricature (not conductance-based HH), synthetic input, not cognition.

_Landed (v2.22, product):_ **Inhibitory STDP — excitation/inhibition balance (Vogels 2011).** `inhibitory_stdp.py`
adds the plasticity family's first **inhibitory** rule: the canonical Vogels, Sprekeler, Zenke, Clopath & Gerstner
mechanism (*Science* 334:1569, 2011). Inhibitory synapses onto a LIF neuron adapt by spike timing (`Δw = η(x_post − α)`
on a pre-spike, `η·x_pre` on a post-spike, `α = 2·ρ0·τ`) — a negative-feedback controller that drives the neuron's
firing rate to a target ρ0 and meets excitation with co-tuned inhibition (detailed E/I balance / network stability).
Self-check: a runaway rate (0.143, ~2.9× ρ0) is pulled to **0.050 ≈ ρ0** as inhibition grows from 0; under weak drive
the inhibitory weights **shrink** (not over-suppressed); deterministic. Distinct from `homeostatic_scaling`
(multiplicative excitatory scaling) and `intrinsic_plasticity` (neuron gain). Pure-stdlib core (`dependencies = []`),
**93/93** self-checks; the 5 safety-envelope files untouched.

_Landed (v2.21, product):_ **Real hardware + an ingest wizard in the app.** Two follow-ups to the usability pass:
(1) **Connect a real serial arm** — Arm Control gets a `serial:` COM field + **Connect**; on connect the buttons route
through `serial_arm.SerialArm` (the real actuator sink) *instead of* the simulator, and the canvas still renders (the
SerialArm mirrors an `ArmSim`). **No safety bypass:** `SerialArm` enforces the calibration gate + wire allow-list +
E-STOP freeze, so an uncalibrated rig is **refused** with an honest message (verified: uncalibrated → `RuntimeError`,
stays on the sim; connected → E-STOP still freezes the wire). Real-arm serial needs a source install with `.[arm]`
(pyserial); the exe reports that. (2) **Ingest wizard** — the Lake tab's "Import data..." opens a guided dialog that
turns a **spike-event CSV** (`channel,time`) *or* a **dense sensor CSV** (rows = time, columns = channels, thresholded)
into a queryable `.spk` — the ingest → store → query loop, end to end, in the GUI. **92/92** self-checks (`--smoke` now
exercises the dense import + the serial gate); the 5 safety-envelope files are untouched.

_Landed (v2.20, product):_ **Usable desktop app — load your own data, not just demos.** After an honest usability
review found the app was largely a demo shell (no way to bring your own data anywhere; Live Recognition **empty** on
a fresh unzip; a read-only-install crash), the tkinter app now lets you **Open your own `.spk`**, **Import a CSV of
spike events (`channel,time`) into a queryable store**, and **Open your own gesture-capture CSV** — via file dialogs
wired to the existing loaders. The release now **ships the sample gesture captures** (bundled in the exe, resolved via
`sys._MEIPASS`) so Live Recognition is populated on first launch; `ensure_demo_store` is crash-guarded (temp-dir
fallback on a read-only install); buttons are honestly labelled; and the dev self-test tab is reframed as
**Diagnostics**. Verified: the exe launched from an empty folder finds **3 captures** and builds a demo store. **92/92**
self-checks; the 5 safety-envelope files untouched.

_Landed (v2.19, product):_ **Frequency-gated cell-assembly memory (FIP/FID).** `frequency_memory.py` implements
the frequency-induced mechanism of Lei et al. 2025 (*Neurocomputing* 623:129361,
[DOI](https://doi.org/10.1016/j.neucom.2025.129361)): firing **frequency gates plasticity** — high-frequency neurons
enter **FIP** (plastic) and bind into a recurrent **cell assembly** via Hebbian + structural potentiation, while
low-frequency neurons enter **FID** (frozen), protecting earlier memories. Self-check demonstrates the paper's claimed
properties on synthetic patterns: **frequency-gating** (a one-shot pattern forms nothing; a *repeated* one forms an
assembly), formation, **recollection** (a 40% partial cue completes the full pattern, 0 spurious), **separation**
(distinct patterns → low-overlap assemblies, Jaccard 0.20), and **interference-immunity** (recall of A survives
learning B, its neurons FID-frozen). Distinct from `bcm`/`growing_learner`/`hippocampus` (frequency gates plasticity
on/off → recurrent assemblies). Pure-stdlib core (`dependencies = []`), **92/92** self-checks; the 5 safety-envelope
files untouched. Honest ceiling: rate-based caricature (exact paper equations paywalled), synthetic patterns, not cognition.

_Landed (v2.18, product):_ **Out-of-core streaming — emulate chips bigger than memory (AirLLM-inspired).**
`neuro_chip_stream.py` applies AirLLM's layer-streaming idea to the chip emulator: the chip's **cores are streamed
from disk one at a time** (load a core's weights + membrane state → step → save → free → next), so peak resident
memory is **one core** and the neuron count is bounded by **disk, not RAM**. Self-check proves **streaming-invariance**
(bit-for-bit identical spikes to the in-memory `neuro_chip`) and `max_resident_cores == 1`. A torch/GPU weight-streaming
twin (`research/neuromorphic_torch/neuro_chip_gpu_stream.py`, optional `[gpu]`) streams per-core weight blocks disk→VRAM
so a chip can exceed VRAM (demo: emulated chip 16× larger than peak resident weights), agreeing bit-for-bit with the
in-VRAM version. This is AirLLM's *architecture* (streaming to bypass the memory wall) applied honestly — **not** the
AirLLM library and not an LLM. Trades memory for per-tick disk I/O. Pure-stdlib core (`dependencies = []`), **91/91**
self-checks; the 5 safety-envelope files untouched.

_Landed (v2.17, product):_ **Neuromorphic-chip emulator — run the whole SNN locally without silicon.** `neuro_chip.py`
is a functional/architectural emulator of an event-driven neuromorphic processor: multiple **cores** (populations of
`LIFNeuron`s) wired by an on-chip **AER spike router**, time-stepped, with SynOps/**energy** (reuses `energy.py`)/latency
accounting — the software twin of the Loihi/Akida silicon the repo targets via `nir_export`, so the neuromorphic stack
runs entirely in software (no physical chip, no cloud). Self-check proves **mapping-invariance** (the same net on 1 core
vs N cores gives identical spikes — the router doesn't change the computation), AER packets crossing cores, and energy
accounting. A torch/GPU-vectorized twin (`research/neuromorphic_torch/neuro_chip_gpu.py`, optional `[gpu]` extra) scales
it to thousands of neurons (4096-neuron demo). **Not AirLLM** — AirLLM runs LLMs and can't emulate a spiking chip (see
[docs/AIRLLM_ANALYSIS.md](docs/AIRLLM_ANALYSIS.md)). Honest ceiling: functional/architectural, not cycle-accurate.
Pure-stdlib core (`dependencies = []`), **90/90** self-checks; the 5 safety-envelope files untouched.

_Landed (v2.16, product):_ **R-STDP — reward-modulated STDP with an eligibility trace (distal reward).** A new
member of the plasticity family (`r_stdp.py`): spike-timing STDP writes to a slowly-decaying per-synapse
**eligibility trace**, and a **delayed** reward gates that trace into the weights — the Izhikevich-2007 solution to
the distal-reward problem, integrated from **BrainCog** (Zeng et al., *Patterns* 2023,
[DOI](https://doi.org/10.1016/j.patter.2023.100789)) learning-rule #5. Genuinely distinct from `valence_stdp` (which
is RPE over feature-presence with *immediate* reward, no spike timing, no delay). Self-check: causal pairing
potentiates, anti-causal depresses, and the causally-paired synapse still wins after a 100-step delay over a
distractor active at reward time (paired **+0.070** vs distractor **-0.001**) — the distal-reward hallmark. Pure
stdlib, `dependencies = []`, deterministic, **89/89** self-checks; the 5 safety-envelope files untouched.

_Landed (v2.15, product):_ **Core Runner — the ENTIRE core runs inside the app (and the .exe).** The tkinter
control panel gains a fifth tab that runs **every CORE module's own self-check in-process** (via `runpy`, isolated
`sys.argv`), PASS/FAIL live — no `python` subprocess, so it works inside the frozen `.exe`. The canonical list is
`run_selfchecks.CORE` (bundled into the exe), so this is the single-source-of-truth core, not a duplicate. A
`--run-core` CLI runs the whole core headlessly (`snn-ui.exe --run-core` → **54/54 core self-checks passed**,
verified in the packaged exe). The `--smoke` gate now also asserts the runner. Confirms the download genuinely
contains and runs the complete core. Zero-dep core (`dependencies = []`), **88/88** self-checks, the 5
safety-envelope files untouched.

_Landed (v2.14, product):_ **NL Command tab — the natural-language addon, in the desktop app.** The
tkinter control panel (`control_panel`) gains a fourth tab: type an English command and `nl_interface` maps it
to a **safe allow-listed** arm action — driven through the **same gated `ArmSim.apply()`** as the Arm Control
buttons (reflex / E-STOP / collision) — or a lake query, or a reject. The backend can only pick a label, never a
raw command. Default is the zero-dep `StubBackend` (deterministic, in the smoke gate); an optional on-device
AirLLM backend stays CLI-opt-in (`SNN_LLM=1 python local_llm/run_local.py`). Rebuilt into the downloadable
`.exe`/zip. Zero-dep core (`dependencies = []`), **88/88** self-checks, the 5 safety-envelope files untouched.

_Landed (v2.13, product):_ **One-click download — the Windows app bundled as a release asset.**
`packaging/make_release_bundle.py` (stdlib-only) zips the built `dist/snn-ui.exe` with `QUICKSTART.txt`, `LICENSE`,
and the docs into `spiking-neural-data-lake-<ver>-windows-x64.zip`, attached to the GitHub release — so users
**download and run with no build step and no Python** (unzip → double-click `snn-ui.exe`; verify with
`snn-ui.exe --smoke`). Verified end-to-end here: the bundled exe extracted from the zip runs `--smoke` and exits 0.
Build-time only; core stays `dependencies = []`, the safety envelope is untouched, **88/88** self-checks.

_Landed (v2.12, product):_ **Installable Windows `.exe` — the desktop app packaged with PyInstaller.** The
zero-dep tkinter control panel (`control_panel`) builds into a standalone `dist/snn-ui.exe` (onefile, windowed,
~10 MB) via `pip install ".[exe]"` + `python packaging/build_exe.py` — Python + tkinter bundled, **no third-party
runtime dependency** (the torch/polars/pyserial/airllm addons are excluded). Double-click to run the full GUI on a
machine with no Python; verify headlessly with `snn-ui.exe --smoke`. An optional
[Inno Setup](https://jrsoftware.org/isinfo.php) script (`packaging/snn-ui.iss`) wraps it into a `setup.exe` with a
Start-Menu entry + uninstaller. PyInstaller is a build-time extra — the core stays `dependencies = []`; the arm
safety envelope is untouched (every GUI button still routes through `ArmSim.apply()`). Guide:
[docs/BUILD_EXE.md](docs/BUILD_EXE.md).

_Landed (v2.11, product):_ **Run the entire stack locally, no cloud — optional AirLLM natural-language interface.**
`src/snn_data_lake/nl_interface.py` (zero-dep, CORE) turns English into a SAFE intent: it maps a phrase to an
EXISTING allow-listed arm command (`serial_arm.is_valid_command`) or a structured `spike_telemetry_hub` query, or
REJECTS it. The LLM only chooses *which* allow-listed action — never the command string — so it can never inject a
raw motor command (the self-check proves an adversarial backend returning raw commands is rejected). The backend is
injected: a stdlib `StubBackend` (the CI-tested default + offline fallback) or an optional on-device
[AirLLM](https://github.com/lyogavin/airllm) model (`pip install ".[llm]"`, quarantined under `local_llm/`,
CPU-capable, excluded from the zero-dep CI). `local_llm/run_local.py` runs the whole stack — core + local lakehouse
+ NL interface — with zero cloud; see [docs/RUNNING_LOCAL.md](docs/RUNNING_LOCAL.md). Every motor command still flows
through the safety-gated `ArmSim.apply()`; the 5 envelope files are untouched. Pure stdlib core, `dependencies = []`,
**88/88 self-checks**. Ceiling: keyword/label routing + a fixed allow-list, not language understanding.

_Landed (v2.10, product):_ **Hippocampal SWR replay — sample-efficient value learning.** `experience_replay.py`
adds a sharp-wave-ripple **experience-replay** buffer (forward+reverse consolidation of `(unit, action, reward)`
transitions), grounded in Nasrin et al. 2021 ([DOI](https://doi.org/10.1016/j.cogsys.2021.07.008)). Wired into
`autonomous_loop.py` as a **default-OFF opt-in** (`AutonomousArm(replay=True)`): on quiescent ticks the arm
consolidates past experience and learns the rewarded action in **~2× fewer live interactions** (self-check: a
decisive value in 3 live ticks vs 7). SAFETY: replay **only** mutates the value table — it never selects or
emits a motor command, never runs on a danger tick; the self-check asserts danger → E-STOP + a frozen arm *with
replay on*. Pure stdlib, core `dependencies = []`, **87/87 self-checks**, the 5 safety-envelope files untouched.
Ceiling: uniform-sampled tabular replay, no prioritization/model/rollouts — not planning, not a world model.

_Landed (v2.9, product):_ **Hippocampal DG→CA3 memory — pattern separation + completion.** `hippocampus.py`
adds a dentate-gyrus **pattern-SEPARATION** stage (sparse random expansion + k-WTA → near-orthogonal codes for
similar inputs) feeding a CA3 **pattern-COMPLETION** attractor (a partial/noisy cue falls into the basin of the
stored trace it overlaps most), grounded in Nasrin et al. 2021 *"Spiking Neural Networks and Hippocampal
Function"* ([DOI](https://doi.org/10.1016/j.cogsys.2021.07.008)). Self-check: separation ratio **2.01**,
completion **0.997** on 30%-masked cues, and it is **additive & non-degrading** (`recall()` recognizes noisy
cues at least as well as the raw `GrowingLearner` cosine, which stays the default). Pure stdlib, core
`dependencies = []`, **86/86 self-checks**, safety envelope untouched. Ceiling: single-shot autoassociative
attractor, no temporal sequence, not cognition; synthetic streams.

_Landed (v2.8, product):_ **Autonomous adaptive arm — grows from multimodal data + learns actions, safely.**
`autonomous_loop.py` ties the arc together: multimodal sensing → the growing learner (a unit per novel context)
→ reward-modulated action selection (RPE + cortisol) → the arm. Autonomous = no hardcoded gesture→command map;
it grows its own repertoire AND learns which action each context should trigger from reward (self-check: policy
moves GRIPPER_CLOSE → GRIPPER_OPEN by reward). SAFETY holds: danger → gated `EMERGENCY_STOP` before any action,
every motion through `ArmSim.apply()`, and a learned action after the stop does not move the frozen arm — the 5
safety-envelope files untouched. 85/85 self-checks. Ceiling: adaptive + growing, not cognition; synthetic streams.

_Landed (v2.7, product):_ **Growing spiking learner — novelty-triggered neurogenesis.** `growing_learner.py`:
`GrowingLearner.observe(pattern) → (unit_id, grew)` is an online, adaptive-resonance-style learner over the fused
`[0,1]^N=64` multimodal spike-feature vectors (REUSING `multimodal_encoder.encode_multimodal`). It STARTS SMALL
(zero prototypes) and GROWS: below `vigilance` (0.85) cosine similarity a pattern is NOVEL, so it RECRUITS a new
prototype initialized to it (neurogenesis, `grew=True`); otherwise it refines the nearest unit toward the pattern
(running-average / Hebbian, `grew=False`). `classify` is recognition-only; `n_units` is the grown capacity. This
is the first-class ONLINE version of `signal_loop.cluster_unknowns` (which already grows the gesture vocabulary in
batch) — the growth engine for a robot arm that expands from multimodal data. Honest scope: online prototype
recruitment (ART-like), NOT deep / gradient learning; synthetic streams. The self-check streams 5 distinct
multimodal patterns and grows to EXACTLY 5 units, recognizes fresh jittered copies (1.00 accuracy), grows to 6 on a
novel pattern, and is deterministic. Pure stdlib, core `dependencies = []`, **84/84 self-checks**, safety envelope
untouched.

_Landed (v2.6, product):_ **Multimodal spike encoder — the fusion front-end.** `multimodal_encoder.py`:
`encode_multimodal(visual, temperature, audio, distance) → [0,1]^N` fuses four sensor modalities into the
existing N=64 spike substrate, BANDED by modality (visual `[0:16]`, temperature `[16:32]`, audio `[32:48]`,
distance `[48:64]`). Each modality is clamped to `[0,1]` then population-encoded into its band by REUSING
`population_encoding.PopulationEncoder.activations` (per-element Gaussian activations averaged, so the band
stays in `[0,1]`); a `None` modality is a graceful dropout (zeroed band, others still encode). The fused vector
feeds `spike_preprocessing.encode_latency` UNCHANGED, so the code flows through the existing matcher/learner
pipeline like today's single-modality window. Honest scope: a spike ENCODER (not "thinking"), synthetic streams,
the foundation for a robot arm that grows from multimodal data. Deterministic, pure stdlib, core `dependencies =
[]`, **83/83 self-checks**, safety envelope untouched.

_Landed (v2.5, performance):_ **Two measured wins, behaviour identical.** No algorithm changed.
**Closed-form Van Rossum fast path** (`spike_preprocessing.py`): `encode_latency` emits ≤1 spike/channel, so
the live matcher compares ≤1-spike trains — a single causal spike's per-channel discrete L2² is a geometric
series whose closed form is an exact algebraic identity of the same sum. Guarded on the actual ≤1-spike
condition (falls back to the unchanged general multi-spike loop otherwise): **~200 → ~18 µs/call (~11×)**,
value-identical to the discrete sum (**max |Δ| ≤ 1e-9**, measured 2.7e-15 over 3k+ cases) and matcher
decisions **bit-identical over 700 windows** (0 flips). **Opt-in stream group-commit** (`streaming_hub.py`):
new `flush_every=K` knob — **default 1 = flush-every-event, byte-for-byte unchanged**; `K>1` buffers and
flushes once per K appends (and always on query/close), trading up to K-1 events of hard-crash durability for
~K-fold fewer flush syscalls. Core `dependencies = []`, **82/82 self-checks**, safety envelope untouched.

_Landed (v2.4, performance):_ **Four measured compute wins, outputs identical.** No algorithm changed —
only redundant work removed, each proven output-identical / accuracy-unchanged before shipping: the live
**hybrid matcher** stops recomputing the Van Rossum nearest distance + re-encoding the window (`hybrid_match`
returns them; `signal_loop` reuses) — **~2.5 → ~1.3 ms/window**, decision bit-identical over 600 windows;
**Van Rossum distance** hoists `math.exp` out of the per-feature loop and skips channels silent on both sides —
**~300 → ~200 µs/call**, byte-exact over 918 cases; **LCA** default iterations 300 → 80 — **~3.6× faster**,
matcher accuracy flat (clean 5/5, jittered 100% at 300/150/80); **telemetry ingest** tracks the last timestamp
instead of rescanning the whole channel each append — **O(n²) → O(n), ~165× on 3000 incremental ingests**,
stored `.spk` bytes byte-identical. Core `dependencies = []`, **82/82 self-checks**, safety envelope untouched.

_Landed (v2.3, product):_ **Desktop control panel — the zero-dep tkinter UX.** A single-window
`ttk.Notebook` (`control_panel.py`, `snn-ui`) with three tabs: **Arm Control** (a live 2-link arm on
a `tk.Canvas` from `ArmSim` forward kinematics — links, end-effector, floor, obstacle; red on
`collision()`; buttons incl. a big red EMERGENCY_STOP that move the arm **only** through the
safety-gated `apply()`), **Live Recognition** (step / auto-play a builder capture through the same
`gesture_recognition.classify → Interpreter → arm` path `live_arm` uses), and **Lake Browser** (open
a `.spk` store, run windowed `spike_telemetry_hub` queries with the "reads X% of file" stat). tkinter
is stdlib, so core `dependencies = []` is unchanged; the smoke is headless-safe (asserts the
display-free logic, then constructs+destroys the widgets, skipping on `tk.TclError`) and runs in CI —
**82/82 self-checks**. The safety envelope (`reflex` / `serial_arm` / `arm_sim` / `arm_config` /
`interpreter`) is untouched.

_Landed (v2.2, product):_ **Arm primitives wired into the loop (opt-in).** The built-but-unwired arm
primitives are now genuine, off-by-default features on `live_arm`, all routed through the safety-gated
`apply()`: `ring_attractor.ProprioceptiveEstimator` → `--proprioception` (read-only encoder-free joint
estimate + uncommanded-slip watchdog; `run()` gained an optional `observer=None`), `spiking_pid.drive_to`
→ `--reach DEG` (spiking-PID joint positioning), and `cpg_oscillator.cpg_sweep` → `--sweep` (CPG rhythmic
bounded sweep). Defaults are byte-for-byte unchanged and the safety envelope (`reflex` / `serial_arm` /
`arm_sim` / `arm_config` / `interpreter`) is untouched. No new modules — 81/81 self-checks, core
`dependencies = []` unchanged.

_Landed (v2.1, product):_ **Paradigm-C reasoning promoted to core** — `vsa_reasoning.py` (renamed from
`research/spiking_vsa.py`) fills the "reason" verb with VSA / hyperdimensional bind·bundle·permute
compositional queries (context-steered arm command, role–filler recall). Honestly labelled: it is dense
bipolar (+/-1) HD-vector algebra, neuromorphic-**adjacent**, NOT spike-timing — it complements
`paradigm_b_engine`, which reasons in genuine spike timing. Stdlib-only (dropped the `stats_bounds` /
seed side-effect); core `dependencies = []` unchanged, 81/81 self-checks (a tier move, not an addition).

_Landed (v2.0, scope realignment):_ **Core refocused on genuinely-spiking modules.** Moved the three
"spiking-in-name-only" modules — `spiking_som` (a plain Kohonen SOM), `sparse_distributed_memory` (binary
Hamming recall), `fly_hash` (continuous-vector LSH) — out of the installable `snn_data_lake` core into
`research/`, since they compute over dense/binary vectors, not spike events. The shipped core now contains only
genuinely-spiking + supporting-infra modules. Followed the v1.9 Fable 5 scope-alignment review; core `dependencies
= []` unchanged, 81/81 self-checks (a tier move, not a removal).

_Landed (v1.9, core hardening):_ **Actuator safety + determinism + robustness** — EMERGENCY_STOP now freezes the
serial wire, calibration gate is non-bypassable, the default matcher is deterministic (see CHANGELOG). From the
Fable 5 core review.

_Landed (v1.8, research):_ **AdEx neuron** (`research/adex_neuron.py`) — the adaptive exponential integrate-and-fire
neuron (Brette & Gerstner 2005): LIF plus an exponential spike-initiation term and a slow adaptation current `w`
(subthreshold `a`, spike-triggered jump `b`). The adapting cell fires fast then slows — ISI grows 11.6 → 76 ms
(17 vs 103 spikes for the non-adapting twin under the same drive); f-I curve monotone; silent below rheobase.
Spike-frequency adaptation = a neuron reporting change, not level. Zero-dep, deterministic.

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
