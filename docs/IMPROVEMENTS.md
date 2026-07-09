# Improvement Report — Spiking Neural Data Lake

_Prepared 2026-07-01. A prioritized, cited audit of the repo across four areas: engineering
& code quality, research/scientific direction, neuromorphic hardware & MLOps, and the
product/robotics loop. Findings are grounded in the code at the commit on `main` (v0.60) and
in current SNN literature (sources at the end)._

## How to read this

Each recommendation carries a priority and an effort estimate:

- **P0** — high leverage, low/medium effort. Do these first.
- **P1** — high value, more effort or depends on a P0.
- **P2** — worthwhile once P0/P1 land, or research bets.

Every item states _why_ (grounded in a file or a source) and gives concrete
acceptance criteria so it can be turned into an issue verbatim.

---

## Executive summary

This is an unusually disciplined research repo: a genuinely zero-dependency core, an
`assert`-based self-check on every stdlib module, clean git hygiene (no venvs, state, or
data committed — 92 tracked files, all source/docs), honest "limitations" sections, and a
per-version CHANGELOG tied to tags. The engineering gaps below are the normal cost of moving
fast, not signs of neglect.

The five highest-leverage moves:

1. **Add a real test framework (pytest) beside the self-checks** — today failures surface as
   subprocess stdout truncated to the last 2 000 chars (`run_selfchecks.py`), which is hard to
   debug and impossible to selectively run or measure for coverage. (P0)
2. **Make CI a matrix — and include Windows.** The product targets Windows (`record_and_run.ps1`,
   `COM8` serial, PowerShell docs) but CI only runs Ubuntu + Python 3.11. A whole supported
   platform is untested. (P0)
3. **Add a surrogate-gradient (BPTT) baseline** via snnTorch, which is _already_ a declared
   dependency. Unsupervised STDP tops out at 82.3% CPU / 90% GPU; surrogate-gradient SNNs reach
   ~98–99% on MNIST-class tasks. This is a scientific control _and_ an honest ceiling. (P0)
4. **Adopt DVS128 Gesture as the flagship benchmark.** It is a _gesture_ dataset — exactly the
   product's domain — loadable through Tonic, which the repo already uses for N-MNIST. It replaces
   the off-theme N-MNIST 71% result with a recognized benchmark (SOTA 99.3%). (P0)
5. **Turn the compute thesis into an energy claim.** The project measures SynOps; multiply by a
   citable per-op energy (~23 pJ/SynOp on Loihi) to report Joules/inference vs. a dense MAC
   baseline. That converts "3.6–83× fewer ops" into a defensible efficiency number. (P1)

---

## A. Engineering & code quality

### A1 — Adopt pytest alongside the self-checks (P0)

**Why.** There is no test framework in the repo — no `pytest`/`unittest` imports, no
`conftest.py`, no `pytest.ini`/`tox.ini`. "Tests" are `if __name__ == "__main__"` asserts run as
isolated subprocesses by `run_selfchecks.py`, which on failure prints only
`r.stdout[-2000:] + r.stderr[-2000:]`. That is good smoke coverage but it (a) truncates
tracebacks, (b) can't run a single case or parametrize, (c) yields no coverage signal, and
(d) mixes "does it import and run" with "is the numeric result correct."

**Do.** Keep `run_selfchecks.py` as the zero-dep smoke gate (it's a real asset), and add a
`tests/` suite that imports modules and asserts on behavior — especially edge cases the demos
don't hit: empty/one-sample windows in `normalize`/`resize_to_n`, out-of-range channels in
`data_quality.check_schema`, the Lowe ratio-test boundary in `signal_loop`, `reflex` firing at
exactly the threshold, `serial_arm` refusing an uncalibrated real port. Move `pytest` into a
`test` optional-dependency group.

**Acceptance.** `pytest` runs green locally and in CI; line coverage reported (target ≥70% on
`src/snn_data_lake/`); at least the safety-critical modules (`serial_arm`, `reflex`, `arm_config`,
`data_quality`) have explicit edge-case tests.

### A2 — CI matrix across Python versions and OS, incl. Windows (P0)

**Why.** `.github/workflows/ci.yml` runs one job: `ubuntu-latest`, Python 3.11. The package
declares `requires-python = ">=3.11"` (so 3.12/3.13 are claimed but untested) and the _product_
is Windows-first (`record_and_run.ps1`, `--serial COM8`, `CLOUDSDK_PYTHON` notes). Windows path
handling, UTF-8 console behavior (the runner already fights cp1252), and PowerShell flows are
exactly where a stdlib project breaks, and none of it is exercised.

**Do.** Expand to a matrix: `{ubuntu-latest, windows-latest} × {3.11, 3.12, 3.13}`. Add pip
caching (`actions/setup-python` `cache: pip`), a `concurrency` block to cancel superseded runs,
and separate jobs for lint/type/coverage (below). Keep the self-check job as-is.

**Acceptance.** Green on all six cells; a Windows failure would now block merge.

### A3 — Introduce static typing incrementally (P1)

**Why.** There are effectively zero type annotations across the 52 Python files (a scan for
return arrows finds only occurrences inside strings/comments). For a codebase with real
interfaces — `ArmSim`/`SerialArm` are explicitly "drop-in" duck-typed twins, matchers return
`(label, dist, …)` tuples — types would catch contract drift that tests miss and document the
"same surface" claims the docstrings make in prose.

**Do.** Add annotations to public surfaces first: `spike_preprocessing` (`encode_latency`,
`van_rossum_distance`), `signal_loop` (`normalize`, `parse_line`, `build_matcher`),
`data_quality` gates, and the `ArmSim`/`SerialArm` protocol. Add `mypy` (or `pyright`) in a
non-blocking CI job first, then ratchet to blocking on `src/`.

**Acceptance.** `mypy src/snn_data_lake` passes on the annotated modules; a `py.typed` marker
ships in the wheel once coverage is meaningful.

### A4 — Add ruff (lint + format) and pre-commit (P0, tiny)

**Why.** No `ruff`/`black`/`flake8`/`isort`/`pre-commit` config exists. Style is currently
consistent by hand; a linter makes that free and catches real bugs (unused imports, shadowing,
mutable defaults) across `research/` too.

**Do.** Add `[tool.ruff]` to `pyproject.toml` (select `E,F,I,UP,B`), a `.pre-commit-config.yaml`,
and a CI lint job. Autofix once, commit the diff.

**Acceptance.** `ruff check .` and `ruff format --check .` are green in CI.

### A5 — Single source of dependency truth + reproducible locks (P1)

**Why.** `requirements.txt` and `pyproject.toml`'s optional groups both list the heavy stack, and
they can drift. `requirements.txt` pins only `snntorch==1.0.0` (with a good comment on _why_);
`torch`, `torchvision`, `bindsnet`, `polars`, `tonic`, `pygenn` float. For a repo whose whole
pitch is "clone, run, reproduce the printed metrics," an unpinned `torch` is a reproducibility
hole — the CHANGELOG already records one such break (a newer snntorch shifted a recon metric).

**Do.** Make `pyproject.toml` the source of truth and either delete `requirements.txt` or generate
it from the extras. Add a lockfile for the research stack (`uv lock` or `pip-tools`) and pin
upper bounds on the GPU deps. Document the CUDA-build install separately (it already is, well).

**Acceptance.** One declared dependency set; a committed lock reproduces the training env; CI
installs from the lock for the (optional) GPU/lake jobs.

### A6 — Optional NumPy fast-path for hot numeric loops (P2)

**Why.** The core is pure-Python by design (a real strength, and the reason CI is zero-dep). But
the hot paths — Van Rossum distance, latency encoding, the associative-memory recall — are
Python loops over lists. For large sweeps that's slow. A NumPy path (behind the existing extras,
default off) keeps the zero-dep guarantee while making big runs practical — _provided_ it
reproduces the deterministic results bit-for-bit (determinism is load-bearing for the query path,
per the README).

**Do.** Add a `fast` extra with a NumPy implementation of the two or three hottest kernels,
selected at runtime, with a self-check asserting parity against the stdlib reference.

**Acceptance.** `van_rossum_distance` NumPy vs stdlib agree to float tolerance in a test; a
documented speedup on a 10k-event benchmark.

---

## B. Research & scientific direction

### B1 — Add a supervised surrogate-gradient (BPTT) baseline (P0)

**Why.** Every trainable model here learns by STDP (unsupervised) or hard-WTA, and the honest
ceiling is 82.3% CPU / 90.0% GPU on MNIST, with the last leg to ~95% framed as a _compute_
problem. In the wider field, **surrogate-gradient training reaches near-ANN accuracy — ~98–99%
on MNIST-class tasks, within 1–2% of a dense ANN** [1][2]. The repo already depends on snnTorch,
whose entire reason for existing is surrogate-gradient descent [2]. Adding one BPTT-trained SNN
gives (a) a strong, fast, single-GPU accuracy ceiling, and (b) a clean scientific contrast —
"biologically-plausible local STDP vs. global surrogate-gradient" — which is a more interesting
result than either alone.

**Do.** A `research/snn_mnist_surrogate.py` (snnTorch LIF + `surrogate.fast_sigmoid`, BPTT),
reporting accuracy _and_ SynOps next to the STDP models. Frame it explicitly as the supervised
upper bound.

**Acceptance.** ≥97% MNIST from the surrogate model, logged alongside SynOps, in the results
table with the "local vs global learning" framing.

### B2 — Make DVS128 Gesture the flagship event benchmark (P0)

**Why.** The event-camera result today is N-MNIST at 71.0% (nearest-prototype, no learning). It's
off-theme (handwritten digits) and well below what the same pipeline could show.
**DVS128 Gesture is an 11-class hand/arm-gesture event dataset** — the product's actual domain —
with 1176 train / 288 test samples, loadable through **Tonic, which the repo already uses**
[5][6]. Current SOTA is **99.3%** (SpikeVision) with directly-trained spiking transformers at
~98.6% [3][4]. Even a modest surrogate-gradient conv-SNN lands in the 90s.

**Do.** Add `research/dvs_gesture.py` (Tonic `DVSGesture` loader + the B1 surrogate model). Then
close the loop: feed its predictions into the existing `Interpreter → arm` path for a real
event-camera → robot-arm demo, replacing synthetic/ultrasonic input. SHD/SSC (Spiking Heidelberg
Digits/Commands) [7] is the analogous _temporal_ benchmark and a natural fit for the streaming /
telemetry side.

**Acceptance.** A DVS Gesture number in the results table with train/test split and seed; bonus:
`snn-arm` can be driven from DVS Gesture events end-to-end.

### B3 — Convert the compute thesis into an energy claim (P1)

**Why.** The project's premise is "less compute" measured as SynOps, and it reports ratios
(3.6–83× fewer ops). SynOps is the right primitive, but the field quantifies it:
**SynOps = Σ(spikes_k × fanout_k)**, and a single SynOp costs on the order of **~23 pJ on Loihi**;
on Loihi ~86% of energy is synaptic-operation energy [8]. Multiplying the already-measured SynOps
by a cited per-op figure — and comparing to a dense ANN's MAC energy — turns the headline into
**Joules per inference vs. dense**, which is the number reviewers and the INRC proposal will ask
for.

**Do.** Add an `energy` helper (a few lines) that takes any model's measured SynOps and prints an
estimated energy alongside a dense-MAC baseline, with the per-op constants and their citation in
the docstring. Fold an energy column into the results table.

**Acceptance.** Each efficiency claim in the README also states estimated J/inference vs dense,
with the SynOps formula and the pJ/op source cited.

### B4 — Baseline the associative memory against (modern) Hopfield (P2)

**Why.** The "80 patterns @ 99.6%, 874× smaller" associative-memory result is strong but stands
alone. Naming the baseline makes it land: classical Hopfield capacity is ~0.14·N, and modern
Hopfield networks (2020) store exponentially many patterns — a natural yardstick for both the
capacity and the factored-storage claims.

**Do.** Add a short comparison in `research/` (or just the docstring/README) stating the classical
bound and, if feasible, a modern-Hopfield capacity reference point next to the measured numbers.

**Acceptance.** The capacity claim is reported relative to a named baseline, not in isolation.

### B5 — Report mean ± std over seeds for headline metrics (P1)

**Why.** The README notes numbers are fixed-seed ("rerun to reproduce"), which is honest but
single-sample. A headline like "90.0% @ 1600n" or "82.3% CPU" is more credible as mean ± std over
N seeds. The repo already has `research/stats_bounds.py`, so the machinery exists.

**Do.** Run the load-bearing benchmarks over ≥5 seeds; report mean ± std (or a 95% CI) in the
results table; keep the fixed seed for the CI self-check.

**Acceptance.** Top-line accuracy/compression/Hits@k figures show dispersion, not point values.

---

## C. Neuromorphic hardware & MLOps

### C1 — Stand up the Loihi 2 / Lava path for Paradigm B (P1)

**Why.** The README states Paradigm B's real win "needs neuromorphic silicon," and the repo
already contains an **INRC proposal** (`INRC Proposal - Spiking Neural Data Lake.docx`) — so
Loihi access is already the intended direction. **Lava is Intel's open-source framework that maps
the same high-level Python network to CPU _or_ Loihi 2** [9][10], so the port is _testable without
hardware_ and becomes a real-silicon energy measurement once INRC access lands. Loihi 2 supports
up to ~1M neurons / 128 neurocores [10]. For the _edge_ robot-arm path, **BrainChip Akida** is the
pragmatic counterpart — commercially available (unlike Loihi, which is research-only via INRC),
event-driven, with on-chip learning [11][12]; **SpiNNaker2** (153 ARM cores + ML/neuromorphic
accelerators) is the other research platform [11].

**Do.** Port `paradigm_b_engine` to a Lava `Process` + `ProcessModel`, validate on the Lava CPU
backend against the current CPU matcher (bit-identical matches), and gate a real-silicon run on
INRC access — reporting measured energy per query (the number B3 estimates analytically).

**Acceptance.** Lava CPU port reproduces the coincidence/sequence matches; a documented plan
(and, once INRC lands, measured energy) for the Loihi 2 run.

---

## D. Product & robotics loop

### D1 — Harden the actuator safety envelope (P0 for any real-hardware use)

**Why.** The existing gates are genuinely good: `SerialArm` refuses to open a real port unless
`arm_config.SAFETY_CALIBRATED` is `True`, a serial write error _raises_ rather than driving blind,
and `Reflex` preempts cognition on danger channels. But three defense-in-depth pieces are missing
for real servos: (a) a **host→arm heartbeat / dead-man timeout** — if the host stalls or the loop
dies, nothing tells the arm to hold, because safety currently lives on the host side; (b) a
**command rate limit / bounded queue** so a runaway loop can't flood the UART; (c) a **firmware-side
limit** in `deploy/command_arm.ino` so the ESP32 enforces joint/velocity bounds even if the host
sends garbage (never trust the host).

**Do.** Add a watchdog: the firmware holds a safe pose if no valid command arrives within N ms;
the host sends a periodic heartbeat. Rate-limit commands host-side. Enforce joint/velocity clamps
in firmware. Write a one-page hazard analysis (what can fail, what catches it) into
`docs/HARDWARE_SETUP.md`.

**Acceptance.** Yank the cable / kill the host mid-motion → arm holds safe within the timeout; a
malformed command is rejected by firmware, not executed.

### D2 — Measure and publish end-to-end loop latency (P1)

**Why.** The system claims real-time control (`--window`, `STRIDE < W` for "lower live latency")
but there are no latency numbers. For a control loop, sensor→command latency (and its tail) is as
important as recognition accuracy.

**Do.** Instrument the `signal_loop`/`closed_loop` path to record per-stage timing; report p50/p95/p99
sensor→command latency at a stated sample rate, sim and (if available) on the real UART.

**Acceptance.** A latency table in `docs/RUNNING.md`; a regression check that flags if median
latency crosses a budget.

### D3 — Live drift detection and re-calibration (P1)

**Why.** `arm_config.py` states plainly that recognition is tuned to a specific reference sensor
mount — "move the rig + the 95% recognition collapses" — and IR fusion is disabled because it's
uncalibrated. The repo already has `bins_from_capture()` (derive bins from real data) and
`data_quality` gates; the missing piece is applying them to _live_ input to notice drift before it
becomes silent misclassification.

**Do.** Add a drift monitor on the live stream (distribution shift vs the enrolled reference /
rising novelty rate) that warns and can trigger `bins_from_capture()` re-derivation; provide a
guided re-enrollment routine (extend `record_and_run.ps1`).

**Acceptance.** Moving the sensor triggers a drift warning rather than a quiet accuracy drop; a
one-command recalibration restores recognition.

### D4 — Unify the demo on a real event stream (P2)

**Why.** The closed loop runs on synthetic/ultrasonic signals; the event-camera work (N-MNIST) is
separate and off-theme. B2's DVS Gesture model would let the _same_ `Interpreter → arm` loop run
on a real event-camera gesture stream — one coherent, on-theme story instead of two.

**Do.** Route DVS Gesture predictions into the Interpreter and reuse the arm path.

**Acceptance.** `snn-arm` (or a new entry point) drives the arm from real DVS Gesture events.

---

## Prioritized roadmap

| # | Item | Area | Priority | Effort | Payoff |
|---|------|------|:--------:|:------:|--------|
| A1 | pytest suite beside self-checks | Eng | P0 | M | Debuggable, coverable tests |
| A2 | CI matrix incl. Windows + 3.12/3.13 | Eng | P0 | S | Tests the platform you ship |
| A4 | ruff + pre-commit | Eng | P0 | S | Free bug-catching, consistency |
| B1 | Surrogate-gradient baseline (snnTorch) | Research | P0 | M | ~98–99% ceiling + control |
| B2 | DVS128 Gesture as flagship benchmark | Research | P0 | M | On-theme, recognized SOTA task |
| D1 | Actuator watchdog + firmware limits | Product | P0* | M | Safe real-hardware operation |
| A3 | Incremental type hints + mypy | Eng | P1 | M | Contract safety, docs |
| A5 | One dependency source + lockfile | Eng | P1 | S | Reproducibility |
| B3 | SynOps → Joules energy claim | Research | P1 | S | Defensible efficiency number |
| B5 | Mean ± std over seeds | Research | P1 | S | Credible headline metrics |
| C1 | Lava/Loihi 2 port of Paradigm B | HW | P1 | L | Real-silicon energy win |
| D2 | End-to-end latency measurement | Product | P1 | S | Real-time claim, quantified |
| D3 | Live drift detection / recalibration | Product | P1 | M | Robustness to rig changes |
| A6 | Optional NumPy fast-path | Eng | P2 | M | Speed, zero-dep preserved |
| B4 | Hopfield baseline for assoc. memory | Research | P2 | S | Contextualized capacity claim |
| D4 | Unify demo on DVS Gesture stream | Product | P2 | M | One coherent on-theme demo |

_*D1 is P0 only if/when real servos are driven; it's not on the critical path for the sim._

---

## Sources

1. A Practical Tutorial on Spiking Neural Networks (surrogate-gradient MNIST ~98–99%), MDPI, 2025 — https://www.mdpi.com/2673-4117/6/11/304
2. snnTorch documentation — surrogate gradients & Tonic integration — https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
3. SpikeVision: a fully spiking transformer-inspired model — 99.3% on DVS128 Gesture (TU/e) — https://research.tue.nl/en/publications/spikevision-a-fully-spiking-neural-network-transformer-inspired-m/
4. Spikformer: When Spiking Neural Network Meets Transformer (DVS Gesture ~98%) — https://arxiv.org/pdf/2209.15425
5. Tutorial 7 — Neuromorphic Datasets with Tonic + snnTorch — https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
6. DVS Gesture via Tonic (loader, 1176/288 split) — https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_stmnist.html
7. The Heidelberg Spiking Data Sets (SHD/SSC) — https://www.researchgate.net/publication/348092152_The_Heidelberg_Spiking_Data_Sets_for_the_Systematic_Evaluation_of_Spiking_Neural_Networks
8. An Analytical Estimation of Spiking Neural Networks Energy Efficiency (~23 pJ/SynOp; SynOps formula) — https://arxiv.org/pdf/2210.13107
9. Intel Advances Neuromorphic with Loihi 2 and the Lava software framework — https://www.intc.com/news-events/press-releases/detail/1502/intel-advances-neuromorphic-with-loihi-2-new-lava-software
10. A Look at Loihi 2 — Open Neuromorphic (1M neurons / 128 neurocores; Lava maps to CPU or Loihi) — https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/
11. Neuromorphic Hardware Guide (SpiNNaker2, Akida) — Open Neuromorphic — https://open-neuromorphic.org/neuromorphic-computing/hardware/
12. Top Neuromorphic Chips in 2025 (Akida commercial availability) — https://www.elprocus.com/top-neuromorphic-chips-in-2025/
