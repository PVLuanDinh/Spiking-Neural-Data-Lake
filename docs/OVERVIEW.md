# Spiking Neural Data Lake — Architectural Overview

## What it is

The **Spiking Neural Data Lake** is a pure-Python neuromorphic system built on one thesis: sparse, event-driven spikes can store, query, and reason over
data — and drive a real robot arm — with measurably less compute and storage than dense
floating-point tensors. It is not a single demo but three integrated systems built on shared
primitives (leaky-integrate-and-fire neurons, spike-timing codes, Hebbian/STDP learning):
a spiking-network toolkit, a spike-native data lakehouse, and a closed-loop robot-arm
controller.

The **core product** (`src/snn_data_lake/`, 373 modules; 189 in the self-check gate) declares two runtime dependencies
(`dependencies = ["torch", "numpy"]` in `pyproject.toml`) — it is `pip install`-able and runs
anywhere Python 3.11+ runs, with no GPU required. Every efficiency
claim in the README (synaptic-operation counts, storage bytes, query I/O) is measured against
a dense baseline and reproducible by running the file that makes the claim — the project has
an explicit honesty policy (see CHANGELOG) about which numbers are achieved vs. targets.

A second, clearly separated tier — **`research/`** (45 files) plus a `neuromorphic_torch/`
subproject — holds GPU/torch experiments (MNIST STDP/Diehl-and-Cook, spiking model-based RL,
knowledge-graph embeddings). This tier is ruff/CI-excluded and needs its own virtual
environment; it is explicitly **not** part of the shipped product.

## The three pillars

1. **Spiking-neural-network toolkit** — neuron models (LIF, Izhikevich, Hodgkin-Huxley,
   Adaptive-Exponential, resonate-and-fire, dendritic, SRM), plasticity/learning rules (STDP,
   BCM, homeostatic scaling, intrinsic plasticity, three-factor delay learning, Oja PCA),
   neural coding (rate, population/RBF, temporal/TTFS, rank-order, multimodal fusion),
   cognitive-map / hippocampal mechanisms (ring attractor, HTM sequence memory, DG-separation
   + CA3 completion), and a family of streaming sketches (Count-Min, HyperLogLog, MinHash,
   AMS F2, DGIM windowed count, P² quantiles, Morris counter, reservoir sampling).

2. **Spiking data-lakehouse** — three data-system "paradigms" implemented directly over
   spike trains: **A** (telemetry hub: store + query sparse multi-channel spike events,
   batch and streaming), **B** (in-storage search: compile a query into an SNN that streams
   stored spikes and emits only coincidence/temporal-order matches), and **C** (relational
   reasoning: a knowledge graph in spike timing — link prediction via TransE/RotatE — plus a
   complementary VSA/hyperdimensional-algebra reasoner). A local Medallion lakehouse
   (Bronze -> Silver -> Gold, Parquet, optional Delta Lake PoC) sits on top.

3. **Robot-arm control loop** — a biologically-layered controller where sensor data becomes
   spikes, spikes are matched against a learned/stored vocabulary in the data lake, matches
   become commands, and every command is safety-gated before it reaches a simulated or real
   servo arm. Reward (dopamine-like RPE) and stress (cortisol-like tonic state) modules close
   the loop with online learning.

## The end-to-end closed loop

```
ESP32 sensor --> spike-encode --> spike data lake --> order-aware match --> Interpreter --> command --> arm
 (distance/IR)   (multimodal/        (.spk store,      (Van Rossum /                        (JOINT_*, GRIP,   (sim So101Arm
                  latency encode)     Paradigm A/B)      learned classifier)                  EMERGENCY_STOP)   or real servos
                                          ^                                      |                                  |
     reflex (fast LIF, ms) --------------+ preempts cognition                   v                                  |
     dopamine RPE (reward) <---------------------------------- outcome <-------------------------------------------+
     cortisol (slow stress state) <---- modulates reflex threshold + learning rate
```

Concretely, across the modules that implement each stage:

- **Sense + encode** — `esp32_encoder_path.py` / `sensor_csv_ingest.py` /
  `multimodal_encoder.py` / `population_encoding.py` turn raw ultrasonic/IR (and
  visual/audio bands, held at zero until those sensors land) into a fused N=64 spike
  vector; `spike_preprocessing.py` provides deterministic latency encoding and Van
  Rossum / Victor-Purpura distance metrics.
- **Store** — `spike_telemetry_hub.py` / `streaming_hub.py` persist the spike stream as
  a sparse `.spk`/`.spkl` event log with windowed queries that read only the touched slice.
- **Match/recognize** — `signal_loop.py`, `gesture_recognition.py`, and
  `learned_matcher.py` run the encoded signal through `paradigm_b_engine.py` (order-aware
  coincidence matching) or a trained `snn_classifier.py`, with `growing_learner.py`
  available for novelty-triggered online vocabulary growth.
- **Interpret + command** — `interpreter.py` turns a matched label into a concrete,
  validated command; `closed_loop.py` / `autonomous_loop.py` wire the full loop together
  (console entry points `snn-loop`, `snn-arm`, `snn-demo`).
- **Reflex (fast path)** — `reflex.py` runs a sign-aware fast LIF over nociceptor/safety
  channels and can fire STOP/WITHDRAW *before* the slower recognition path completes.
- **Act** — `so101_arm.py` (the MuJoCo SO-101: six joints, real units, mass, gravity and
  actuator forces matched to the STS3215) or `serial_arm.py` (real ESP32 servos over serial;
  it mirrors the same body) — **every** motion,
  sim or real, is routed through the `So101Arm.apply()` safety gate.
- **Reward + stress + learn** — `valence_stdp.py` implements dopamine-style
  reward-prediction-error STDP toward APPROACH/AVOID; `cortisol.py` is a slow tonic
  stress scalar that raises reflex sensitivity and hardens aversive learning under
  repeated danger; `arm_bridge.py` closes the loop over UDP (now token-authenticated) for
  split-process deployments.
- **Low-level control** — `population_encoding.py` (RBF population code) and
  `spiking_pid.py` (NEF-style population-coded spiking PID) drive joint trajectories;
  `cpg_oscillator.py` (Matsuoka CPG) and `ring_attractor.py` (CAN integrator / encoder-free
  proprioception) provide rhythmic motion and self-state estimation primitives.

Measured headline results (see README for the full, reproducible table): gesture
recognition on real captured ESP32 gestures at **95%** (order-aware, up from
63% direction-blind); associative-memory recall through 60% cue corruption; 61-162x less
data moved for lake storage/query versus dense baselines; 3.6-83.5x fewer synaptic
operations than dense-ANN equivalents across the classifier/temporal-coding prototypes.

## Module families (rough counts)

| Family | Examples | Approx. count |
|---|---|---|
| Neuron models | `lif_neuron`, `izhikevich_neuron`, `hodgkin_huxley`, `srm_neuron` (research), resonate-and-fire, AdEx, dendritic | ~8-10 |
| Plasticity / learning | `valence_stdp`, `bcm_plasticity`, `homeostatic_scaling`, `intrinsic_plasticity`, `delay_learning` (research) | ~6-8 |
| Coding / encoding | `population_encoding`, `population_vector`, `multimodal_encoder`, `spike_preprocessing`, `rank_order_coding` | ~8 |
| Cognitive-map / memory | `ring_attractor`, `htm_sequence_memory`, hippocampal DG/CA3 (research), `lca_sparse_coding` | ~5-6 |
| Streaming sketches | `spike_stream_sketch` (Count-Min), `spike_cardinality` (HyperLogLog), `minhash`, `frequency_moment` (AMS), `sliding_window_count` (DGIM), `streaming_quantile` (P²), `morris_counter`, `reservoir_sampling` | ~9-10 |
| Data-lake core (Paradigms A/B/C) | `spike_telemetry_hub`, `streaming_hub`, `paradigm_b_engine`, `paradigm_b_matcher`, `vsa_reasoning`, `proximity_index` | ~7 |
| Lakehouse | `lakehouse/medallion.py`, `lakehouse/delta_demo.py` | 2 |
| Arm loop + safety envelope | `reflex`, `serial_arm`, `so101_arm`, `arm_config`, `interpreter`, `closed_loop`, `autonomous_loop`, `live_arm`, `arm_bridge`, `spiking_pid`, `cpg_oscillator` | ~12-15 |
| Sensor/data ingest | `esp32_encoder_path`, `sensor_csv_ingest`, `make_sensor_dataset`, `sensor_demo` | ~4 |
| Classifier / recognition | `snn_classifier`, `learned_matcher`, `gesture_recognition`, `tempotron`, `nir_export` | ~5 |
| UI / demo / entrypoints | `demo.py` (`snn-demo`), `substrate/cockpit.py` (the ONE frontend, served by the daemon) | 2 |
| **Core total** | `src/snn_data_lake/**/*.py` | **~91** |
| **Research (quarantined)** | `research/**/*.py` (MNIST STDP/D&C, spiking-MBRL/Dreamer, KG embeddings, compression, associative memory) | **~45** |

The exact, canonical module list that CI runs is `run_selfchecks.py` — it is the single
source of truth (previously this list drifted between the CI workflow and docs; now both
point at one file).

## Dependency design & the core-vs-research tiers

The core's `pyproject.toml` declares `dependencies = ["torch", "numpy"]` (user policy,
2026-07-18: the torch tier is on by default; the former zero-dep-core contract is removed).
`pip install -e .` (or the packaged `.exe`) pulls those two on Python 3.11+, so the project
is clone-and-run and fully covered by a CI matrix of `{ubuntu-latest, windows-latest} x
{3.11, 3.12, 3.13}` — the Windows-first hardware product is actually exercised on Windows,
not just Linux CI runners. That matrix now installs the full dependency set: the torch-free
matrix job and the `NORD_TORCH=0` off-switch it exercised were both removed on 2026-07-29 with
the zero-dependency rule they implemented.

Everything beyond those two is pushed behind **optional extras** and kept
out of the core import path:

- `[arm]` — `pyserial`, for the real ESP32/serial actuator path (lazily imported; CI never
  opens a port).
- `[lake]` — `polars` + `pyarrow` + `deltalake`, for the Medallion/Delta lakehouse PoCs.
- `[gpu]` — `torch` + `snntorch` + `bindsnet` + `numpy`, for MNIST/GPU and spiking-MBRL
  research (its own venv).
- `[nmnist]` — `tonic`, for real N-MNIST event-camera data.
- `[genn]` — `pygenn`, for GeNN GPU ports (needs a C++ compiler + CUDA).
- `[llm]` — an optional local-first AirLLM backend for a natural-language command
  interface (`local_llm/`), quarantined from core, not in CI.
- `[exe]` — `pyinstaller` (build-time only), to package the brain daemon + its cockpit
  into a standalone Windows executable (`integrations/packaging/`).
- `[dev]` — `pytest`, `pytest-cov`, `ruff`, `pre-commit` for the test/lint gates.

This produces a strict two-tier structure:

- **Core** (`src/snn_data_lake/`) — CI-gated on every commit via three gates: the
  core self-check runner (`python run_selfchecks.py`, ~91/91 modules — the count
  moves as modules land), the pytest suite (>=70% coverage gate, currently ~85%), and
  `ruff check .`.
- **`research/`** — GPU/torch prototypes and literature reproductions, explicitly
  ruff-excluded and CI-excluded. Kept for reproducibility and lineage, not shipped as
  product. A `neuromorphic_torch/` research subproject (SpikingJelly-based) is similarly
  quarantined with its own environment.

Supporting invariants worth knowing when reading or touching the code: seeding always uses
a per-instance `random.Random(seed)` (never module-level import-time seeding, to keep
demos deterministic and reproducible); persisted hashes use `zlib.crc32()` rather than
Python's salted builtin `hash()`; and `print()` output is kept ASCII-only because Windows'
default `cp1252` console encoding will crash on stray Unicode.

## Safety model (brief)

Five modules form the **safety envelope** around the physical arm and are treated as
change-controlled: `reflex.py` (emergency-stop reflex), `serial_arm.py` (real hardware
serial interface, fail-safe — a write error raises rather than silently continuing),
`so101_arm.py` (the MuJoCo SO-101 body, whose gate every command — sim or real — is gated
through via `So101Arm.apply()`), `arm_config.py` (calibration, limits, the
`SAFETY_CALIBRATED` flag), and `interpreter.py` (command validation). The real serial path
refuses to operate on an uncalibrated port, with no bypass; recent hardening closed a
fail-*open* gap so non-finite (`NaN`/`inf`) sensor readings are now treated as maximal
danger rather than silently passing safety comparisons. Full hardware bring-up (BOM,
wiring, firmware contract, safety envelope in depth) is documented in
`docs/HARDWARE_SETUP.md`; the firmware itself defaults to `ECHO_ONLY` with a fail-safe
E-STOP (`deploy/command_arm.ino`).

## Honest scope & ceilings

This project is explicit, in its own README and CHANGELOG, about what it is *not*: it is
**adaptive, growing, biologically-inspired control** — not artificial general intelligence,
not "cognition." Specific ceilings called out in the project's own voice:

- The robot-arm gesture-recognition 95% is *achieved* on real captured
  ESP32 data; the unsupervised-STDP MNIST ~95% figure is a *literature target*
  (Diehl & Cook 2015, 6400 neurons + full 60k) — the best measured result here is
  **90.0%** at 1600 neurons, and the tuned 6400-neuron run has not yet been executed at
  that scale, not a method gap.
- Several demos use **synthetic modality streams** (e.g., `make_sensor_dataset.py`) rather
  than live sensors; the live ESP32 path currently covers distance + IR temperature only,
  with visual/audio bands held at zero pending hardware.
- The lakehouse is a **PoC at Medallion scale**, not a production query engine.
- The N-MNIST event-camera ingest is explicitly flagged as a plumbing/pipeline demo
  (nearest-prototype, no learning), off-theme from the project's core claims.
- The arm control loop is **sim-first**; the real-hardware path exists and is
  safety-gated, but most iteration and the fullest test coverage happens in `so101_arm.py`.
- Modules explicitly avoid speculative abstraction, and each CHANGELOG entry names a
  concrete "ceiling" for what was *not* built, on the theory that an honest limitation
  beats an inflated claim.

## Where to start

- **`README.md`** — the full pitch, the measured-results tables, competitive context, and
  the complete repository layout.
- **`python run_selfchecks.py`** — the fastest way to confirm the whole core
  works on your machine; it is exactly what CI runs, module by module.
- **`python learn_daemon.py`** — the always-on brain, which serves the **cockpit** (the ONE
  frontend, `substrate/cockpit.py`) at `http://127.0.0.1:8137/`: one page carrying the live
  anatomical brain view over the real connectome, the simulated 2-link arm and its workspace,
  a 6-DOF hardware torque-budget target (a plan from the build notes, labelled NOT BUILT),
  the live telemetry strip, and chat. Loopback only. The cockpit is not a separate launcher:
  `serve()` never constructs a Brain (Policy 4), so the daemon that owns the tick loop is the
  only thing that serves it.
- **`python -m snn_data_lake.substrate.spike_telemetry_hub --query store.spk --window 0,10`**
  — open a `.spk` store and run windowed queries (`--ingest CSV --out store.spk` imports one).
- From there: `snn-demo` for the one-file whole-loop demo, `snn-arm` for the live/replay
  gesture-to-arm loop, `snn-loop` for the full closed loop with reward learning, and
  `docs/RUNNING.md` / `docs/HARDWARE_SETUP.md` for the full local + hardware runbook.
