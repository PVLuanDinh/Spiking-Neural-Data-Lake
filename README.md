# Spiking Neural Data Lake

[![CI](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml/badge.svg)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/Aighluvsekks/Spiking-Neural-Data-Lake?label=release)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python toolkit for event-driven (spiking) computation, in three parts:

- **A neuromorphic primitive library:** 373 modules under `src/snn_data_lake/`, 187 of them mapped
  into the brain atlas,
  organized by brain anatomy (cortex, hippocampus, basal_ganglia, amygdala, cerebellum, brainstem,
  thalamus, neuromodulation, plasticity, and a non-anatomical substrate tier), wired by an explicit
  `connectome`. Integrate-and-fire and
  adaptive-threshold neurons, STDP/BCM/e-prop/ReSuMe plasticity, WTA and reservoir networks, spike
  encoders/decoders, and streaming sketches. Each module is importable as
  `snn_data_lake.<region>.<module>` (the legacy `snn_data_lake.<module>` path still works) and
  carries its own deterministic self-check. See `snn_data_lake.brain_atlas` for the region map.
- **A spike-native data lakehouse:** sparse `.spk` event storage with windowed queries, an
  in-storage coincidence/temporal-order query engine (Paradigm B), a VSA/hyperdimensional
  relational-reasoning layer (Paradigm C), and a Bronze→Silver→Gold Medallion pipeline.
- **A simulated 2-DOF robot-arm control loop:** gesture recognition → command → reward learning,
  with a reflex, a population-coded PID, and neuromodulation, where every motor command passes a
  safety gate (`ArmSim.apply()`) and the real serial path is disabled until explicitly calibrated.

`numpy` and `torch` are core dependencies (`dependencies = ["torch", "numpy"]`) — the former
zero-dependency-core contract was removed project-wide on 2026-07-20, and its last mechanism, the
`NORD_TORCH=0` off-switch, was removed on 2026-07-29. There is no torch-free mode: if torch is
missing the install is broken, and the brain's stdlib fallback exists to keep that survivable rather
than to make it a supported configuration. Optional integrations (real
hardware, Parquet/Delta, GPU training, on-device LLM) live behind extras. Every efficiency figure
below is measured against a dense baseline and reproduced by running the referenced module.

## Install

```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake
cd Spiking-Neural-Data-Lake
pip install -e .            # core (pulls torch + numpy)
pip install -e ".[all]"     # + optional extras (arm, lake, gpu, nmnist, llm, ...)
```

Python 3.11+. Full guide, including per-extra notes, in [docs/INSTALL.md](docs/INSTALL.md).
GPU / accelerator notes (opt-in CUDA for training) in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
Train on a GPU and export weights back to the core: [docs/TRAINING.md](docs/TRAINING.md).

## Quickstart

```bash
snn-demo     # the whole neuromorphic loop in one runnable file
snn-arm      # live gesture -> arm loop (add --serial/--actuate for real hardware)
snn-loop     # full closed loop: gesture -> command -> reward -> learn

python learn_daemon.py      # the always-on brain; serves the cockpit -- the ONE frontend --
                            # at http://127.0.0.1:8137/ (live brain, arm, hardware target, chat)

python run_selfchecks.py    # run every module's self-check (277; 278 with nengo, 271 torch-free)
```

## Results (measured)

All figures are reproduced by running the referenced module. Compute = synaptic operations
(SynOps); storage = bytes. These three tables run in CI on the torch-free matrix job (numpy only).

**Data lake: store, search, reason**

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Associative memory | recall from 40%-corrupted cue | capacity / robustness | 80 patterns @ 99.6%, holds through 60% corruption |
| Associative memory | memory footprint | factored vs dense | 874x smaller (600 B vs 512 KB; O(P·k), not N²) |
| Telemetry hub (Paradigm A) | sparse `.spk` store | size / query I/O | 61x smaller than a dense raster; windowed query reads 2% of the file |
| In-storage query (Paradigm B) | coincidence + temporal order | host transfer | 162x less data to host; order-discriminating (FAR→NEAR ≠ NEAR→FAR) |
| Relational KG (Paradigm C) | link prediction | Hits@1 | 50% TransE / 100% RotatE (cyclic relations; relation algebra verified) |
| Reasoning (Paradigm C, VSA) | bind/bundle/permute | compositional query | context-steered arm command + role-filler recall |
| Sequence model (LSM) | temporal-pattern classification | accuracy | 96% (chance 33%), event-driven |

**Compute efficiency**

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Supervised spiking classifier | 4-shape recognition | accuracy / compute | 100%, 3.6x fewer ops (113.8k SynOps vs 409.6k dense MACs) |
| Temporal (TTFS) coding | latency inference + early exit | SynOps vs rate | 83.5x fewer (5.3k vs 442.8k), 100% accuracy |
| Spike-driven MoE | firing-rate routing | router params / compute | 0 learned router params (8 biases vs 512), 4x compute cut (top-2 of 8) |

**Robot-arm neuromorphic loop (real captured sensor data)**

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Gesture recognition (order-aware Paradigm B) | reference ESP32 captures | accuracy | 95% (APPROACH 98% · RETREAT 85% · IDLE 100%), up from 63% with a direction-blind matcher |
| Closed control loop | gesture → command → reward → learn | behavior | rewarded gesture's value rises; collision → EMERGENCY_STOP + cortisol-sharpened reflex |

**Requires torch/GPU, own venv, not in CI** (recorded in [CHANGELOG.md](CHANGELOG.md) for the stated config):
spiking model-based RL (surrogate-gradient policy 0.319→0.053 m; Dreamer 2.695→0.116 m; 6-DOF
Dreamer-lite 1.03→0.48 m), and MNIST (unsupervised STDP 82.3% CPU; conductance Diehl & Cook 90.0%
@1600n on an RTX 5070).

> The two "95%"s differ. The robot-arm gesture 95% is *achieved* on real captured data. The
> MNIST conductance Diehl & Cook 95% is the *literature target*; best measured here is 90.0%.

## Repository layout

```
src/snn_data_lake/   373 modules; 187 of them mapped into the brain atlas across 17 regions (cortex,
                     hippocampus, basal_ganglia, amygdala, cerebellum, brainstem, thalamus,
                     neuromodulation, plasticity, substrate, ...) + brain_atlas.py (region->module
                     map) + connectome.py (region->region wiring) + lakehouse/ (Medallion
                     Bronze→Gold) + reasoning/ (folded research builds; ruff- and coverage-excluded)
research/            benchmarks and experiments, kept for reproducibility (ruff-excluded, not the product)
tests/               pytest suite (>=70% coverage)
integrations/        packaging/ (daemon + cockpit -> standalone Windows .exe, PyInstaller + Inno Setup),
                     local_llm/ (optional on-device NL interface, AirLLM backend, [llm] extra, no cloud),
                     deploy/ (ESP32 actuator firmware, fail-safe, echo-only by default), robot arm/
docs/                install, hardware setup, storage formats, architecture notes
run_selfchecks.py    canonical self-check runner (single source of truth for the module list)
```

Safety-critical arm code (`brainstem/reflex.py`, `brainstem/serial_arm.py`, `brainstem/arm_sim.py`,
`brainstem/arm_config.py`, `brainstem/interpreter.py`) is documented in [SECURITY.md](SECURITY.md).

## Testing & CI

Four gates, all green on every push:

```bash
python run_selfchecks.py               # 277 module self-checks (deterministic, assert-based)
pytest                                 # unit suite, >=70% line coverage gate (currently 90.19%)
python -m ruff check .                 # lint (E, F, I, UP, B)
python -m ruff format --check src tests   # formatting (CI checks this scope exactly)
```

CI runs the matrix `{ubuntu, windows} × {3.11, 3.12, 3.13}` plus lint and coverage. See
[CONTRIBUTING.md](CONTRIBUTING.md) to add a module.

## Limitations & open problems

- **~95% MNIST is a target, not achieved.** Best measured is 90.0% (1600 neurons, scale-tuned);
  the 6400-neuron run needs the full recipe, multiple epochs, and a GPU — not a code gap.
- **The 6-DOF Dreamer's world model is a plain MLP.** A fully-spiking world model failed to
  regress 6-joint FK; the actor and critic are spiking, the world model is not. The arm is planar.
- **Real-arm "learning" is partly simulated.** Reward/success uses an `--assume-success`
  placeholder and IR sensor fusion is off (`arm_config.IR_PRESENT = None`) until a calibration
  capture and a real success signal land; distance alone decides presence.
- **N-MNIST 71% is a plumbing demo, not a benchmark:** real DVS data, nearest-prototype, no
  learning (SOTA ~99.3%); off-theme vs the storage/robotics work.
- **The query path requires determinism:** Poisson re-encoding yields different spike trains, so
  stochastic encoding cannot be used for matching.
- **Neuromodulators are abstractions**, not biophysics (single global scalars for
  dopamine/cortisol). Paradigm B's real efficiency win needs neuromorphic silicon; the GeNN GPU
  ports are drop-in accelerators, not independently re-validated.

Numbers are fixed-seed; rerun to reproduce.

## Download (Windows, no Python needed)

A standalone executable is published on the
[v3.27 release](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/releases/tag/v3.27): unzip
and run `snn-ui.exe`, which starts the brain daemon and opens the cockpit in your browser (verify
headlessly with `snn-ui.exe --smoke`). A full-source zip is also attached there for a Python 3.11+
install via `install.bat` / `install.sh`.

## Provenance & license

Designs come from two SNN data-storage research briefs (in [`research/`](research/)) plus an
external architectural assessment that mapped the concept to the three paradigms; the spike-driven
MoE is a port of Project Nord's `SpikeDrivenMoE`. Every version is a git tag and GitHub release
([CHANGELOG.md](CHANGELOG.md)). License: [MIT](LICENSE).
