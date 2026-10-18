# Spiking Neural Data Lake: a reproducible artifact for spike-native data systems and safety-gated neuromorphic robotics

**Repository:** https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake
**Version:** v2.70 (v0.1 → v2.70; see CHANGELOG.md)
**License:** MIT
**Core dependencies:** torch, numpy

---

## Abstract

Spiking Neural Data Lake asks a single measurable question: can sparse, event-driven
spikes — instead of dense floating-point tensors — store data, answer queries over it, and
drive a physical robot arm, at a measured compute and storage advantage? The project answers
"yes, on these prototypes, by these margins" and shows its work. It packages three things on
one set of shared primitives (leaky-integrate-and-fire neurons, spike-timing codes,
Hebbian/STDP learning): a **spiking data-lakehouse** with three storage/query/reasoning
paradigms implemented directly over spike trains, a **closed-loop, safety-gated robot-arm
controller** whose reflex, reward learning, stress state, and motor control are themselves
spiking networks, and a **library of ~95 neuromorphic primitives** (neuron models,
plasticity rules, coding schemes, streaming sketches, cognitive-map mechanisms) that both
systems are built from. The product core runs on Python 3.11+ with no GPU required, is
covered by a CI matrix of `{ubuntu, windows} × {3.11, 3.12, 3.13}`, and treats "every claim
reruns by
running the file that makes it" as a design invariant rather than an afterthought. Measured
results include 61–162× less data moved for spike-native storage and query versus dense
baselines, 3.6–83.5× fewer synaptic operations across compute-efficiency prototypes, and
**95% gesture recognition on real captured ESP32 hardware data**. The project is
equally explicit about what it has *not* achieved: the unsupervised-STDP MNIST ~95% figure
quoted in the literature (Diehl & Cook 2015, 6400 neurons) is a target, not a result — the
best measured accuracy here is **90.0% at 1600 neurons**, and the tuned 6400-neuron run has
not yet been executed at that scale (a compute-cost gap, not a method gap). This is offered to the community as a
reproducible engineering-and-research artifact, not a state-of-the-art accuracy paper: read
it for the data-systems-over-spikes paradigm, the safety-gated bio-layered arm-control
architecture, and a clone-and-run baseline you can fork, extend, or hold your
own work against.

---

## 1. Motivation / problem

Two threads in neuromorphic computing are usually pursued separately. One is **spiking
networks as a compute substrate** — event-driven, sparse, potentially far more
energy-efficient than dense tensor math (the case made repeatedly for Loihi, Akida, and
similar silicon). The other is **data systems**: how you store, index, query, and reason over
data at all. Neuromorphic work rarely asks the data-systems question directly — most SNN
research targets a benchmark classification task (MNIST, DVS128, SHD) rather than asking
"what does a *storage engine* or a *query engine* look like if the medium is spike timing
instead of bytes and SQL?"

Separately, robotics work that uses spiking controllers is usually a research prototype that
stops at the simulator boundary, without a safety envelope suitable for touching real
hardware, and without the reflex/neuromodulator layering that gives biological motor control
its robustness.

This project's thesis is that both threads can share one substrate. If sparse spike events
are cheaper to move and store than dense floats (a claim this repo measures, not assumes),
then the same event-driven primitives that make a data lake efficient can drive a
reflex-gated, reward-learning robot-arm loop — and a single codebase can
demonstrate both, reproducibly, on a laptop.

## 2. Contributions

Each contribution is tagged by its evidentiary status:
**[validated]** — measured, in CI, reproducible by running the file;
**[PoC]** — working end-to-end but at prototype scale or with a stated simplification;
**[prototype]** — a first working version, correctness/engineering demonstrated, not yet hardened or benchmarked against alternatives.

### 2.1 Data systems over spike trains — three paradigms **[validated (A/B), PoC (C)]**

Rather than treating spikes only as a neural-network activation, the repo implements three
explicit data-system paradigms directly over spike trains:

- **Paradigm A — telemetry hub.** Store and query multi-channel spike events as sparse
  records (batch and streaming), with windowed reads that touch only the requested slice of
  the file. **[validated]** — `spike_telemetry_hub.py` / `streaming_hub.py`, in CI.
- **Paradigm B — in-storage search / query compilation.** A query is compiled into a small
  SNN that streams stored spikes and emits only coincidence-and-temporal-order matches — the
  query executes *in the storage medium* rather than reading everything out first, and is
  order-discriminating (it distinguishes FAR→NEAR from NEAR→FAR, not just co-occurrence).
  **[validated]** — `paradigm_b_engine.py`, in CI; a GeNN GPU port exists in `research/` as an
  accelerator, not independently re-validated.
- **Paradigm C — relational reasoning.** A knowledge graph *in spike timing*: link
  prediction via TransE and RotatE embeddings, plus a complementary VSA/hyperdimensional
  reasoner (bind/bundle/permute) for compositional queries. This paradigm's core reasoning
  ships in-core; the KG-embedding research lineage (TransE/RotatE, full relation algebra)
  lives in `research/`. **[PoC]** — link prediction and VSA composition work and are measured
  (see Results), but this is a research-scale KG, not a production reasoning engine.
- **Retired: Medallion lakehouse PoC.** An earlier Bronze→Silver→Gold Parquet PoC with a
  data-quality gate and a Delta Lake demo existed. It was retired on 2026-10-17
  (commit 9e378fb2, with its `[lake]` extra) and ships nothing today.

### 2.2 ~90 neuromorphic primitives **[validated]**

The `src/snn_data_lake/` core (~91 modules) implements a working library of neuron models
(LIF, Izhikevich, Hodgkin-Huxley, Adaptive-Exponential, resonate-and-fire, dendritic, SRM),
plasticity/learning rules (STDP, BCM, homeostatic scaling, intrinsic plasticity, three-factor
delay learning, Oja PCA), neural coding schemes (rate, population/RBF, temporal/TTFS,
rank-order, multimodal fusion), cognitive-map/hippocampal mechanisms (ring attractor, HTM
sequence memory, dentate-gyrus separation + CA3 completion), and a family of streaming
sketches reimplemented for spike/event streams (Count-Min, HyperLogLog, MinHash, AMS F2,
DGIM windowed count, P² quantiles, Morris counter, reservoir sampling) — every one with its
own deterministic, assert-based self-check. This is a contribution in its own right: a
citable, reproducible reference implementation of textbook/literature primitives, each
runnable on its own — useful as a teaching artifact, a correctness baseline, or a component
library for other neuromorphic work.

### 2.3 Biologically-layered, safety-gated robot-arm control loop **[validated in sim, PoC on real hardware]**

The arm-control loop is not a single spiking classifier bolted onto a servo — it is a layered
stack where sensing, reflex, cognition, reward, and stress are each separate spiking
subsystems, all routed through one safety gate:

- **Reflex** (`reflex.py`) — a fast, sign-aware LIF over nociceptor/safety channels that can
  fire STOP/WITHDRAW *before* the slower recognition path completes.
- **Reward learning** (`valence_stdp.py`) — dopamine-style reward-prediction-error STDP that
  shapes APPROACH/AVOID valence online.
- **Stress state** (`cortisol.py`) — a slow tonic scalar that raises reflex sensitivity and
  hardens aversive learning under repeated danger (hypervigilance as an emergent property of
  a scalar, not a hand-coded rule).
- **Recognition** (`signal_loop.py`, `gesture_recognition.py`, `learned_matcher.py`, with
  `growing_learner.py` for novelty-triggered online vocabulary growth) — order-aware matching
  through Paradigm B or a trained classifier.
- **Safety envelope** — every motor command, sim or real, is routed through `So101Arm.apply()`;
  the real serial actuator path (`serial_arm.py`) is fail-safe (a write error raises rather
  than silently continuing) and refuses to operate on an uncalibrated port, with no bypass.
  Non-finite (NaN/inf) sensor readings are treated as maximal danger rather than silently
  passing safety comparisons.

**[validated]** in simulation (`arm_sim.py`, CI-green, stdlib forward-kinematics + collision);
**[PoC]** on real hardware — the real ESP32 servo path is shipped and safety-gated, but reward
signal on real hardware is currently a placeholder (`--assume-success`) pending full sensor
fusion (see Limitations).

### 2.4 Reproducibility as a first-class contribution **[validated]**

A CI matrix that actually exercises the claimed platform (Windows, not just Linux), a
>=70%-enforced coverage gate (currently 84%), and a documented policy that persisted hashing
uses `zlib.crc32()` (never Python's salted `hash()`) and seeding always uses per-instance
`random.Random(seed)` (never import-time global seeding) — is treated by the project as
scientific infrastructure, not packaging hygiene. The claim is: every number in the README
traces to a file you can run yourself, on your own machine, and get the same answer.

## 3. System overview

Three pillars share one primitive layer:

1. **Spiking-neural-network toolkit** — neuron models, plasticity, coding, cognitive-map
   mechanisms, streaming sketches (§2.2).
2. **Spiking data-lakehouse** — Paradigms A/B/C (§2.1); the earlier Medallion PoC is retired.
3. **Robot-arm control loop** — sense → think → act, safety-gated throughout (§2.3).

End-to-end closed loop (text diagram, see `docs/OVERVIEW.md` for the fully annotated version):

```
ESP32 sensor --> spike-encode --> spike data lake --> order-aware match --> Interpreter --> command --> arm
 (distance/IR)   (multimodal/        (.spk store,      (Van Rossum /                        (JOINT_*, GRIP,   (sim So101Arm
                  latency encode)     Paradigm A/B)      learned classifier)                  EMERGENCY_STOP)   or real servos
                                          ^                                      |                                  |
     reflex (fast LIF, ms) --------------+ preempts cognition                   v                                  |
     dopamine RPE (reward) <---------------------------------- outcome <-------------------------------------------+
     cortisol (slow stress state) <---- modulates reflex threshold + learning rate
```

For the full module-by-module walk (each pipeline stage mapped to its implementing file),
see `docs/OVERVIEW.md`, which is kept current as the single architectural source of truth.

## 4. Methods

**Neuron models.** LIF as the workhorse; Izhikevich, Hodgkin-Huxley, Adaptive-Exponential
(AdEx), resonate-and-fire, dendritic-compartment, and SRM models available for
biophysical-fidelity tradeoffs.

**Plasticity / learning.** Classical pair-based STDP; BCM; homeostatic (theta) scaling and
intrinsic plasticity (both needed to prevent single-neuron WTA collapse, discovered in the
project's own v0.3 MNIST work); three-factor delay learning (timing itself as a learned
parameter, not just weights); Oja's rule for online PCA; reward-prediction-error (dopamine)
STDP for valence learning in the arm loop.

**Neural coding.** Rate coding, population/RBF coding (used for the NEF-style spiking PID),
temporal/time-to-first-spike (TTFS) coding with early exit, rank-order coding, and a
multimodal fusion encoder that turns raw sensor bands (currently distance + IR temperature
from a real ESP32; visual/audio bands reserved but held at zero pending hardware) into a
fused N=64 spike vector, plus deterministic latency encoding and Van Rossum / Victor-Purpura
spike-train distance metrics for matching. Encoding is deliberately deterministic where used
for query matching — Poisson/stochastic re-encoding is explicitly avoided there because it
would make the same query non-reproducible.

**Data-lake paradigms.** See §2.1 for A/B/C. Query execution in Paradigm B is genuinely
in-storage: the query is compiled into a small SNN and streamed against the stored spikes,
rather than materializing the full record set before filtering.

**Arm loop + safety envelope.** See §2.3. The controller is explicitly layered on a
biological analogy (reflex → cognition → reward/stress modulation) rather than being a single
end-to-end network, and the safety gate is architecturally separated from the
learning/recognition path so that a bug or an unexpected input in the "smart" path cannot
bypass the "dumb" gate.

## 5. Results (measured)

All figures below are reproduced by running the referenced file (README.md, "Results
(measured)" section); GPU/torch rows are recorded in CHANGELOG.md for the stated
configuration. Compute = synaptic operations (SynOps); storage = bytes.

| Metric | Achieved here | Baseline / target | Source |
|---|---|---|---|
| Associative-memory recall from 40%-corrupted cue | 80 patterns @ 99.6%; holds through 60% corruption | — | README.md, "Data lake" results table |
| Associative-memory footprint | 874× smaller (600 B vs 512 KB) | dense O(N²) baseline | README.md, "Data lake" results table |
| Telemetry hub (Paradigm A) storage | 61× smaller than a dense raster; windowed query reads 2% of the file | dense raster | README.md, "Data lake" results table |
| In-storage query (Paradigm B) host transfer | 162× less data to host; order-discriminating (FAR→NEAR ≠ NEAR→FAR) | full-transfer baseline | README.md, "Data lake" results table |
| Relational KG (Paradigm C) link prediction, Hits@1 | 50% TransE · 100% RotatE (cyclic relations) | — | README.md, "Data lake" results table |
| Dense associative memory capacity | dense n=4 clean recall past 2·N; log-sum-ReLU energy matches softmax capacity, exp-free | classical Hopfield bound 0.138·N | README.md, "Data lake" results table |
| Sequence model (LSM) temporal-pattern classification | 96% (chance 33%), event-driven, stdlib | — | README.md, "Data lake" results table |
| Deep SSM on real Spiking Heidelberg Digits (SHD), 20-class | 32% (torch, research tier) | — | README.md, "Data lake" results table |
| Supervised spiking classifier, 4-shape recognition | 100% accuracy, 3.6× fewer ops (113.8k SynOps vs 409.6k dense MACs) | dense same-architecture MAC baseline | README.md, "Compute-efficiency prototypes" table |
| Temporal (TTFS) coding, latency inference + early exit | 83.5× fewer SynOps (5.3k vs 442.8k), 100% accuracy | rate coding | README.md, "Compute-efficiency prototypes" table |
| Spike-driven MoE routing | 0 learned router-net params (8 biases vs 512); 4× compute cut (top-2 of 8 experts) | learned dense router | README.md, "Compute-efficiency prototypes" table |
| Gesture recognition (order-aware Paradigm B), real ESP32 captures | **95%** (APPROACH 98% · RETREAT 85% · IDLE 100%) | 63% with a direction-blind template matcher | README.md, "Robot-arm neuromorphic loop" table — **achieved on real captured hardware data** |
| Unsupervised STDP MNIST, CPU | 82.3% (300 neurons / 6k images) | literature ~95% (Diehl & Cook 2015, 6400n + 60k) | README.md, "Trainable MNIST models" table |
| Conductance Diehl & Cook MNIST, GPU (RTX 5070, BindsNET) | **90.0% @ 1600 neurons** (scale-tuned); 86.4% @ 400n; 47.8% @ 6400n with untuned defaults | literature ~95% target — the tuned 6400n/~95% run **has not yet been executed at that scale, not a method gap** | README.md, "Trainable MNIST models" table + "Status at a glance" |
| N-MNIST event-camera ingest (real DVS data, 1.25M events, nearest-prototype, no learning) | 71.0% | flagged off-theme "pipeline/plumbing demo," not a benchmark (SOTA ~99.3% elsewhere in the literature) | README.md, "Trainable MNIST models" table + Limitations |
| Test coverage | 84% (`--cov-fail-under=70` enforced gate) | — | README.md badges + `pyproject.toml` `[tool.pytest.ini_options]` |
| Core module self-checks | 123/123 modules pass | — | README.md "Status at a glance"; `run_selfchecks.py` |

**The honest distinction that matters most:** the robot-arm **gesture** 95% is *achieved* on
real ESP32 hardware captures. The MNIST conductance-Diehl-and-Cook "~95%" is
a *literature target* the project has not yet reached — the best number actually measured
here is 90.0% at 1600 neurons, and the recipe for the tuned 6400-neuron/~95% run is known but
has not yet been run at that scale, which the project states plainly is a
compute/access gap, not evidence the method doesn't scale.

## 6. Reproducibility

- **Two core runtime dependencies.** `pip install -e .` (or the packaged Windows `.exe`)
  pulls torch and numpy on Python 3.11+; no GPU is required. Every other third-party need
  (serial I/O, Parquet/Delta, GPU/training wheels, event-camera data, an LLM backend,
  PyInstaller packaging) is an opt-in extra.
- **CI matrix:** `{ubuntu-latest, windows-latest} × {3.11, 3.12, 3.13}`, plus separate lint
  (`ruff check .`) and coverage jobs. The Windows-first hardware product is actually tested on
  Windows, not inferred from Linux CI.
- **Deterministic seeding.** Every stochastic component uses a per-instance
  `random.Random(seed)`; there is no import-time global `random.seed()` anywhere in core, so
  demos and self-checks are bit-for-bit reproducible run to run.
- **"Every claim reruns by running the file that makes it."** Each number in the README's
  results tables is generated by the module it's attributed to; the CHANGELOG additionally
  records the exact GPU/torch configuration (env vars, hardware) for the research-tier rows
  that require a separate venv.
- **Two independent verification gates.** `python run_selfchecks.py` runs every core module's
  `__main__` self-check (123/123, exactly what CI runs) as a fast smoke gate;
  `pytest` (via the `[dev]` extra) runs the debuggable/coverable unit suite with a
  `--cov-fail-under=70` enforced gate (currently 84%).
- **Two install paths for non-developers.** A standalone Windows `.exe` (no Python needed,
  `snn-ui.exe --smoke` for headless verification) and a full-source installer
  (`install.bat` / `install.sh`) that builds a venv and installs every optional extra it can,
  reporting exactly which extras succeeded.
- **Explicit core/research separation.** `src/snn_data_lake/` (the ~91-module product core) is
  CI-gated on every commit; `research/` (~45 files) holds GPU/torch experiments, is
  ruff-excluded and CI-excluded, and needs its own virtual environment — kept for lineage and
  reproducibility, not shipped as product.

## 7. Positioning vs. prior work

This project is not competing with snnTorch, BindsNET, SpikingJelly, or Nengo as an SNN
training framework, and it is not claiming SOTA accuracy against Diehl & Cook 2015 or modern
DVS128-Gesture leaderboards. It is honest about that (§9, Limitations). What is genuinely
different:

- **vs. snnTorch / BindsNET / SpikingJelly (SNN training frameworks).** These are the tools
  the project's own `research/` tier *uses* (e.g., BindsNET for the conductance Diehl & Cook
  runs) — this repo is not a competing framework. The contribution is one level up: a
  data-systems architecture (store/query/reason) and a safety-gated robotics loop built *on
  top of* spiking primitives, most of which are hand-rolled specifically so the product core
  doesn't need any of these frameworks.
- **vs. Nengo / NEF-style frameworks.** The arm loop borrows the NEF idea of population-coded
  representations for the spiking PID controller, but implements it directly rather than
  depending on Nengo (the `[nengo]` bridge is an opt-in extra, not a core dependency).
- **vs. Intel Loihi / neuromorphic silicon.** The project targets these as a *future substrate*
  rather than a dependency — Paradigm B's in-storage query and the spiking control loop are
  architected for event-driven hardware, and the stdlib `energy.py` module already estimates a
  Joules-per-inference figure from measured SynOps under cited per-op energy constants, as a
  first step toward an on-chip energy claim, without requiring Loihi hardware to reproduce the
  repo's own numbers.
- **vs. Diehl & Cook 2015 / unsupervised-STDP MNIST literature.** Treated explicitly as a
  target, not a competitor: this repo reaches 82.3% (CPU, 300 neurons) and 90.0% (GPU, 1600
  neurons, scale-tuned) against the literature's ~95% at 6400 neurons + full 60k images — a
  compute-and-access gap that the project documents rather than obscures.
- **vs. neuromorphic databases / event-camera research.** Most "spikes as data" work in the
  literature is about *encoding* sensor data as spikes for a downstream network (e.g., DVS
  event cameras feeding an SNN classifier). This project instead asks what a **storage and
  query engine** looks like natively over spike timing — sparse windowed reads, in-storage
  query compilation, and a knowledge graph embedded in spike timing — which is a less common
  framing in the neuromorphic-hardware or event-camera literature.
- **What is different, stated fairly.** The differentiator is not raw accuracy — it's (1) a
  clone-and-run reproducibility bar rare in SNN research code, (2) the
  data-lake-over-spikes framing applied to storage/query/reasoning rather than just
  classification, and (3) an integrated, safety-gated path from spikes to a real actuator with
  a fail-safe hardware interface, not just a simulator. This is offered as a reproducible
  engineering-and-research artifact — a place to fork, extend, or benchmark against — not a
  SOTA-accuracy claim.

## 8. Limitations

Stated in the project's own voice (README "Limitations & open problems", CHANGELOG, and
`docs/OVERVIEW.md`):

- **The ~95% MNIST figure is a target, not an achieved result.** Best measured is 90.0%
  (1600 neurons, scale-tuned). The 6400-neuron/~95% run needs the known recipe plus multiple
  epochs (day-scale compute) that has not yet been run at that scale — a compute-access gap,
  not a code or method gap.
- **Some demos use synthetic modality streams.** `make_sensor_dataset.py` and similar
  generate synthetic sensor data for several demos rather than live sensor capture; the live
  ESP32 path currently covers distance + IR temperature only, with visual/audio bands held at
  zero pending hardware.
- **The Medallion lakehouse PoC is retired (2026-10-17, commit 9e378fb2);** the paradigms are research-scale, not a production query engine.
- **The N-MNIST event-camera ingest is an explicitly flagged plumbing/pipeline demo** —
  nearest-prototype matching, no learning — off-theme relative to the project's core
  storage/robotics thesis (literature SOTA on comparable event benchmarks is ~99.3%); the
  project's own research audit recommends replacing it with DVS128 Gesture.
- **The arm control loop is sim-first.** The real-hardware serial path exists and is
  safety-gated, but the fullest test coverage and most iteration happens in `arm_sim.py`; on
  real hardware, reward/success signal is currently an `--assume-success` placeholder and IR
  sensor fusion is off pending a calibration capture.
- **The 6-DOF Dreamer world model is a plain MLP**, not a spiking network — only the actor and
  critic are spiking in that research-tier result; a fully-spiking world model measurably
  failed to regress 6-joint forward kinematics in the project's own testing. The simulated arm
  is planar, not a full 3D manipulator.
- **`[genn]` needs a C++ compiler + CUDA** and is a best-effort, skippable extra; the GeNN GPU
  ports are drop-in accelerators for existing algorithms, not independently re-validated
  results.
- **Neuromodulators (dopamine, cortisol) are single global scalars**, not biophysical
  simulations — no D1/D2 receptor detail or HPA-axis modeling.
- **Latency-coding vs. rate-coding tradeoff is measured, not free**, and has two distinct
  measured operating points versus rate coding — a tuned run (~2.1× faster training, −6.2 point
  accuracy cost) and an untuned run (7.9× fewer SynOps, −13.5 point cost). The larger SynOps
  saving and the smaller accuracy cost come from *different* configurations, not one free lunch.

## 9. Future work / collaboration

The project maintains a literature-grounded research backlog (`docs/RESEARCH_SCAN.md`,
cycle-3 scan of 49 candidates ranked by on-thesis fit, effort, and implementation feasibility) and a
broader engineering/research audit (`docs/IMPROVEMENTS.md`). Concrete extension points for a
collaborator:

- **Log-sum-ReLU / Epanechnikov energy for the dense associative memory** (top-ranked
  next build in the research scan) — a one-kernel swap that removes softmax/overflow
  fragility while keeping exponential capacity; effort S.
- **Resonator-network VSA factorization** (unbind-and-cleanup) to extend `spiking_vsa.py`
  beyond bind/bundle/permute toward decomposition and analogy over the existing
  hyperdimensional codebook.
- **Three-factor learning of synaptic delays** — an on-thesis upgrade given the project's
  "reason in spike timing" framing, since most current learning is still weights-only.
- **NeuroBench Eff-AC / activation-sparsity metric definitions** adopted into `energy.py` for
  a citable, leaderboard-legible efficiency number (without requiring the full NeuroBench
  torch harness).
- **NIR export for Paradigm B/C** — emitting a small NIR graph JSON would let the in-storage
  query engine target Loihi 2, SpiNNaker 2, and Speck via one shared IR instead of a bespoke
  per-chip translator; flagged as the highest-leverage hardware-portability move.
- **DVS128-Gesture as the flagship event benchmark**, replacing the off-theme N-MNIST result,
  reusing the Tonic dependency already present for N-MNIST ingest.
- **Firmware-level heartbeat/dead-man watchdog** with hard relay de-energize — the actuator
  safety envelope today is software-gated (`SAFETY_CALIBRATED`) only; a hardware watchdog
  would close a real gap for anyone extending this to unattended real-hardware operation.
- **A collaborator could plug in** at any of three natural seams: (1) a new neuron/plasticity
  primitive dropped into `src/snn_data_lake/` with its own self-check, following the existing
  pattern; (2) a new research direction under `research/` (own venv, ruff/CI-excluded, no
  obligation to match the core's dependency set); or (3) a hardware target — the sense/act boundary
  (`esp32_encoder_path.py` in, `serial_arm.py`/`arm_bridge.py` out) is the intended
  integration point for a different sensor or actuator.

## 10. How to engage

- **Repository:** https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake
- **Clone and run (developers):**
  ```bash
  git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
  cd Spiking-Neural-Data-Lake
  pip install -e .                 # core install: torch + numpy
  python run_selfchecks.py         # 123/123 module self-checks — exactly what CI runs
  snn-demo                         # one-file whole-loop demo
  python learn_daemon.py           # the always-on brain, serving the cockpit (the ONE
                                   # frontend) at http://127.0.0.1:8137/
  ```
- **No-Python install (Windows):** download the standalone `.exe` release from the repo's
  Releases page; run `snn-ui.exe --smoke` for a headless verification.
- **Full-source install (all optional extras, best-effort):** `install.bat` (Windows) /
  `install.sh` (Linux/macOS) from the full-source release bundle.
- **Architecture reference:** `docs/OVERVIEW.md` (kept current, single source of truth for the
  module map); `docs/RUNNING.md` (local runbook); `docs/HARDWARE_SETUP.md` (BOM,
  wiring, firmware, safety envelope in depth).
- **Suggested citation stub:**
  ```
  Aighluvsekks. Spiking Neural Data Lake (v2.70) [Software]. 2026.
  https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake
  ```
  (No formal paper or DOI exists at time of writing — cite the tagged software release and
  version number; the CHANGELOG.md entry for the cited tag documents the exact state and
  measured numbers as of that release.)
