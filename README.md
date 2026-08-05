# Spiking Neural Data Lake

[![CI](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml/badge.svg)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/Aighluvsekks/Spiking-Neural-Data-Lake?label=release)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/releases)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python toolkit for event-driven (spiking) computation, in three parts:

- **A neuromorphic primitive library:** 376 modules under `src/snn_data_lake/`, 202 of them mapped
  into the brain atlas across 17 regions
  (cortex, hippocampus, basal_ganglia, amygdala, cerebellum, brainstem, thalamus, hypothalamus,
  midbrain, pons, medulla, epithalamus, olfactory, claustrum, neuromodulation, plasticity, and a
  non-anatomical substrate tier), wired by an explicit
  `connectome`. Integrate-and-fire and
  adaptive-threshold neurons, STDP/BCM/e-prop/ReSuMe plasticity, WTA and reservoir networks, spike
  encoders/decoders, and streaming sketches. Each module is importable as
  `snn_data_lake.<region>.<module>` (the legacy `snn_data_lake.<module>` path still works) and
  carries its own deterministic self-check. See `snn_data_lake.brain_atlas` for the region map.
- **A spike-native data lakehouse:** sparse `.spk` event storage with windowed queries, an
  in-storage coincidence/temporal-order query engine (Paradigm B), a VSA/hyperdimensional
  relational-reasoning layer (Paradigm C), and a Bronze→Silver→Gold Medallion pipeline.
- **A 6-DOF SO-101 robot-arm control loop (MuJoCo body, real serial path):** gesture recognition →
  command → reward learning, with a reflex, a population-coded PID, and neuromodulation, where every
  motor command passes the `So101Arm.apply()` safety gate
  (`src/snn_data_lake/brainstem/so101_arm.py`, `class So101Arm` / `def apply`) and the real serial
  path is disabled until explicitly calibrated (`serial_arm.assert_calibrated`).

  *(Corrected 2026-08-04: this line said "simulated 2-DOF" and named the gate `ArmSim.apply()` — a
  class absent from `src/` entirely, and a DOF count belonging to the retired `arm_sim.py`. The
  envelope code is unchanged; only its description was wrong, which is its own hazard: this is the
  README's one sentence telling a reader which code gates every motor command, and it named a file
  that does not exist. `SECURITY.md`'s table and `docs/POLICIES.md` already had the live five.)*

`torch`, `numpy` and `mujoco==3.11.0` are the core dependencies — **three, not two** (`pyproject.toml`
`dependencies`; mujoco is exact-pinned because the solver version IS the physics). The former
zero-dependency-core contract was removed project-wide on 2026-07-20, and its last mechanism, the
`NORD_TORCH=0` off-switch, was removed on 2026-07-29. There is no torch-free mode: if torch is
missing the install is broken, and the brain's stdlib fallback exists to keep that survivable rather
than to make it a supported configuration. Per **P7** (`docs/POLICIES.md`) any dependency that helps
this become a phenomenal brain is accepted, and "it would be another dependency" is not a valid
reason to reject one. Optional integrations (real
hardware, Parquet/Delta, GPU training, on-device LLM) live behind extras. Every efficiency figure
below is measured against a dense baseline and reproduced by running the referenced module.

## Install

```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake
cd Spiking-Neural-Data-Lake
pip install -e .            # core (pulls torch + numpy)
pip install -e ".[all]"     # + optional extras (arm, lake, gpu, nmnist, llm, ...)
```

Python 3.12+ (`pyproject.toml` `requires-python`; raised from 3.11 on 2026-08-05 — the reason and the
cost are in the comment beside that field). Full guide, including per-extra notes, in
[docs/INSTALL.md](docs/INSTALL.md).
GPU / accelerator notes (opt-in CUDA for training) in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
Train on a GPU and export weights back to the core: [docs/TRAINING.md](docs/TRAINING.md).

## Quickstart

```bash
snn-demo     # the whole neuromorphic loop in one runnable file
snn-arm      # live gesture -> arm loop (add --serial/--actuate for real hardware)
snn-loop     # full closed loop: gesture -> command -> reward -> learn

python learn_daemon.py      # the always-on brain; serves the cockpit -- the ONE frontend --
                            # at http://127.0.0.1:8137/ (live brain, arm, hardware target, chat)

python run_selfchecks.py    # run every registered module's self-check; the runner is the one
                            # source of truth for the list (no count is quoted here -- it went
                            # stale at "277" and five documents had copied it)
```

## Results (measured)

All figures are reproduced by running the referenced module. Compute = synaptic operations
(SynOps); storage = bytes. These three tables run in CI on the `self-checks` matrix job
(`{ubuntu, windows} x {3.12, 3.13}`, full dependency set — the 3.11 leg was removed on 2026-08-05
with the `requires-python` floor raise).

*(Corrected 2026-08-04: this said "the torch-free matrix job (numpy only)". That job was DELETED on
2026-07-29 — see the comment at the top of `.github/workflows/ci.yml` — because it asserted the
brain still works without one of its hard dependencies, a contract the project no longer has. The
matrix it carried moved into `self-checks`.)*

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
src/snn_data_lake/   376 modules; 202 of them mapped into the brain atlas across 17 regions (cortex,
                     hippocampus, basal_ganglia, amygdala, cerebellum, brainstem, thalamus,
                     neuromodulation, plasticity, substrate, ...) + brain_atlas.py (region->module
                     map) + connectome.py (region->region wiring) + lakehouse/ (Medallion
                     Bronze→Gold) + reasoning/ (161 modules; still outside the three P3 gates,
                     ruff and the coverage denominator -- P4c debt, target zero, being promoted)
research/            benchmarks and experiments (ruff-excluded) -- P4c debt on the same terms
tests/               pytest suite (>=70% coverage)
integrations/        packaging/ (daemon + cockpit -> standalone Windows .exe, PyInstaller + Inno Setup),
                     local_llm/ (optional on-device NL interface, AirLLM backend, [llm] extra, no cloud),
                     deploy/ (ESP32 actuator firmware, fail-safe, echo-only by default), robot arm/
docs/                install, hardware setup, storage formats, architecture notes
tools/               gates run by CI's lint job (citation_guard, red_reachability, dependency_ledger)
                     + built-but-unregistered gates (doc_command_guard -- P4c debt) + measurement
                     harnesses (charge_probe, p6_gate)
run_selfchecks.py    canonical self-check runner (single source of truth for the module list)
```

**P4c: there is no "outside the brain" in this tree.** `research/`, `reasoning/`, `integrations/`
and `tools/` are ruff- or gate-excluded *tiers*, which is a fact about the build config, never a
disposition. Every file here is part of the brain or retired — "it is only a benchmark / only the
launcher" is not a third category. The exclusions above are recorded **debt**, not design.
*(Corrected 2026-08-04: this block described `research/` as "not the product", the exact framing
P4c abolishes.)*

Safety-critical arm code (`brainstem/reflex.py`, `brainstem/serial_arm.py`, `brainstem/so101_arm.py`,
`brainstem/arm_config.py`, `brainstem/interpreter.py`) is documented in [SECURITY.md](SECURITY.md).

## Testing & CI

```bash
python run_selfchecks.py                     # every registered module's self-check (deterministic, assert-based)
pytest                                       # unit suite, >=70% line coverage gate (the gate is the number; CI prints the rest)
python -m ruff check .                       # lint (E, F, I, UP, B)
python -m ruff format --check src tests tools   # formatting (CI checks THIS scope exactly)
```

CI (`.github/workflows/ci.yml`) runs four jobs, not four commands:

- **`self-checks`** — `{ubuntu, windows} × {3.12, 3.13}`, full dependency set, `MUJOCO_GL=disable`.
- **`wheel`** — asserts the vendored SO-101 model (`scene.xml` + 13 STL meshes) is inside the built wheel.
- **`lint`** — `ruff check` + `ruff format --check src tests tools`, then four repo/policy gates:
  `tools/citation_guard.py` (a tracked file may not cite an untracked `.py` as runnable),
  `tools/red_reachability.py --assert` (**P8** — a success criterion with no reachable red is deleted,
  not weakened), `tools/dependency_ledger.py --assert` (**P7** — every import declared, every
  declaration used) and `tools/dependency_ledger.py --self-check` (**P8** applied to the ledger's own
  criteria — each is mutated against a string copy of `pyproject.toml` and must fire).
- **`test`** — pytest + coverage.

`tools/doc_command_guard.py` (a documented `python -m` command must be runnable) is **built, clean
and NOT in CI** — run it by hand. Registering it is open P4c work.

*(Corrected 2026-08-04: this section said "Four gates, all green on every push" and listed the format
scope as `src tests`. CI's scope is `src tests tools`; the `--check` would pass locally and fail in CI.
It also quoted `277` self-checks and `90.19%` coverage — both stale, and both the kind of number that
goes stale on any promotion, so they now point at the gate instead of restating its output.
**Corrected again 2026-08-05:** the same pass then listed `doc_command_guard` as a fourth lint gate.
Measured against `.github/workflows/ci.yml`, the lint job had five checking steps — `ruff check`,
`ruff format --check`, `citation_guard`, `red_reachability --assert`, `dependency_ledger` — and
`doc_command_guard` appears in no workflow, in `run_selfchecks.py`, in `.pre-commit-config.yaml` or
in `pyproject.toml`. Counting a gate CI does not run is the same error as the "277" it replaced.
**Corrected a third time, later on 2026-08-05, and this one was self-inflicted:** the
`requires-python` floor-raise pass edited the `lint` job in the same commit — adding `--assert` to
the ledger step and registering `dependency_ledger.py --self-check` as a new one — and left both the
bullet above ("three repo/policy gates", `dependency_ledger` "report-only") and this enumeration
standing. RE-DERIVED by parsing the YAML rather than reading it: the `lint` job now has **9 steps**,
of which **6 are checking steps** — `ruff check .`, `ruff format --check src tests tools`,
`citation_guard`, `red_reachability --assert`, `dependency_ledger --assert`,
`dependency_ledger --self-check` — the other three being `checkout`, `setup-python` and
`pip install -e ".[dev]"`. A count of CI steps written in a pass that edits CI is stale before the
pass ends, which is the same shape recorded against `specs/001/plan.md` under P7's residue list.)*

See [CONTRIBUTING.md](CONTRIBUTING.md) to add a module.

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
headlessly with `snn-ui.exe --smoke`). A full-source zip is also attached there for a source install
via `install.bat` / `install.sh`. **That "3.11+" was left standing here and is now scoped rather than
edited into a falsehood:** it describes the source zip attached to the **v3.27** release, whose
`requires-python` genuinely was `">=3.11"`. The CURRENT tree requires **3.12+** (floor raised
2026-08-05), so a source install from `main` needs 3.12; the v3.27 artifact is unchanged and still
installs on 3.11.

## Provenance & license

Designs come from two SNN data-storage research briefs (in [`research/`](research/)) plus an
external architectural assessment that mapped the concept to the three paradigms; the spike-driven
MoE is a port of Project Nord's `SpikeDrivenMoE`. Every version is a git tag and GitHub release
([CHANGELOG.md](CHANGELOG.md)). License: [MIT](LICENSE).
