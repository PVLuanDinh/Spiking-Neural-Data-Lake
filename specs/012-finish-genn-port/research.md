# Phase 0 Research: Finish the GeNN Port

**Date**: 2026-08-07. All facts below measured or fetched this session unless marked otherwise.

## R1. How pygenn installs (the [genn] extra is currently uninstallable)

**Decision**: Install GeNN 5.4.0 from the GitHub source archive via pip; repoint the `[genn]`
extra to a PEP 508 direct reference.

**Facts (measured 2026-08-07)**:
- `pygenn` is NOT on PyPI (`https://pypi.org/pypi/pygenn/json` → 404). The PyPI package `genn`
  (0.7.7) is an unrelated project. The declared extra `genn = ["pygenn", "ml_genn"]` therefore
  cannot install from the default index — `pip install -e .[genn]` fails today.
- GeNN GitHub releases 5.0.0–5.4.0 carry ZERO binary assets. Official install docs
  (genn-team.github.io, fetched): "GeNN needs to be installed from source", recommended route
  `pip install https://github.com/genn-team/genn/archive/refs/tags/<tag>.zip`. Build needs
  VS 2019+ with C++ workload (Windows) + CUDA toolkit; `CUDA_PATH` is set automatically by the
  CUDA installer on Windows. Legacy build deps: pybind11, psutil, pkgconfig, setuptools>=61.
- Extra fix: `pygenn @ https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip`. Direct
  references are fine here (package is not published to PyPI). `ml_genn` (the sibling trainer's
  dependency) stays as-is; whether it resolves is measured in the pre-flight, not assumed.

**Alternatives considered**: conda (GeNN docs: planned, not available); GitHub release wheels
(none exist); leaving the extra broken and installing ad hoc (violates "the extra stood up").

## R2. Box readiness (measured 2026-08-07)

- nvcc: CUDA 13.3 (V13.3.73) on PATH. Exceeds the module's stated CUDA 12.8+ (sm_120) need.
  GeNN 5.4-vs-CUDA-13.3 compatibility is UNKNOWN until a model compiles — that is pre-flight
  scenario 2, not an assumption.
- Host compiler: VS 2022 Community installed (`vswhere` finds it); `cl.exe` NOT on plain PATH.
  GeNN on Windows locates MSVC via its build system; if compile fails on discovery, the
  documented fallback is running from a VS developer environment. Measured in pre-flight.
- GPU: RTX 5070 (sm_120), 12.8 GB VRAM.
- Gate env `.venv-lake`: Python 3.14 free-threading (cp314t). pygenn not installed.

## R3. Free-threading (cp314t) risk — bounded, does not block the gates

**Decision**: Proceed; the two failure shapes are contained by existing project structure.

- Build risk: GeNN's C++ extension must build against the cp314t ABI. pybind11 recent versions
  support free-threading; GeNN 5.4's pin is unknown → measured by the pre-flight install.
- Import risk: an extension that does not declare free-threading support makes CPython
  RE-ENABLE the GIL at import (RuntimeWarning). Bounded twice over: the package guard
  `_require_free_threaded()` (src/snn_data_lake/__init__.py:59) checks
  `sysconfig.get_config_var("Py_GIL_DISABLED")` — a BUILD property that a runtime GIL flip does
  not change, so the guard cannot trip; and `run_selfchecks.py` executes each module's
  self-check in an ISOLATED SUBPROCESS, so a GIL flip inside this module's check cannot
  serialize the rest of the gate run. Consequence recorded, not hidden: if pygenn is not
  ft-safe, the module's own process runs with the GIL on — irrelevant to correctness, and GPU
  kernels don't care.

## R4. What the port must actually mirror (read from the oracle, not the docstring)

The skeleton's `build()` (built-in LIF, TauRefrac=T, fixed Vthresh, no inhibition) does NOT
match the CPU reference's semantics. Read from `FastStdpNetwork.run()`
(reasoning/snn_mnist_stdp_fast.py:100-136):

1. **Membrane**: `mem = mem*BETA + sum(W[:,fired_inputs])` per step (BETA=0.92; equivalent to
   exponential decay with tau = -1/ln(BETA) ≈ 12.0 steps under dt=1 — a mapping, not identity).
2. **Hard GLOBAL winner-take-all, not synaptic inhibition**: per timestep, `argmax(mem - theta)`
   fires iff its excess exceeds FAST_THRESH; at most ONE spike per step population-wide; on a
   spike the ENTIRE population's membrane resets to zero. This is the "lateral inhibition" the
   ledger row names — it is a global reduction + broadcast reset, and **no GeNN built-in neuron
   or synapse model expresses it**.
3. **Adaptive threshold**: winner `theta += THETA_PLUS`; every step (learn only)
   `theta *= THETA_DECAY`. theta persists ACROSS images (prototype homeostasis).
4. **Plasticity (the codegen that already exists)**: on the winner's spike,
   `W[w] += FAST_LR*(x_pre - FAST_XTAR)`, clamp [0, WMAX]; `x_pre` decays by
   `exp(-1/TAU_PRE)` per step, +1.0 on input spike. Optional pair-LTD (FAST_AMINUS) is
   DEFAULT-OFF and measured to degrade accuracy — NOT ported (P1 exception c: inert, default-off).
5. **Per-image reset**: mem, x_pre (and x_post) start at zero each image; per-image (learn only)
   row normalization `W[i] *= W_NORM / sum(W[i])`.
6. **Encoding**: deterministic burst latency (encode_latency): pixel >= 0.10 emits
   ceil(intensity*FAST_BURST) spikes starting at `round((1-intensity)*(T-1))`, T=30. Reused
   VERBATIM from the CPU module (it produces plain (times, indices) tensors) — the port uploads
   exact timestamps; encoding is shared, not reimplemented (P2).
7. **Assign/eval**: class-conditional response counting on the training set, prediction =
   argmax over per-class summed responses of assigned neurons. Plain numpy on host — spike
   counts come off the GPU per image.

**Constants** (reasoning/snn_mnist_stdp.py:73-93): T=30, BETA=0.92, W_NORM=78.0, WMAX=1.0,
TAU_PRE=20.0, N_IN=784; M, TRAIN_N, TEST_N, THETA_PLUS, THETA_DECAY env-tunable
(NORD_M/NORD_TRAIN/NORD_TEST/NORD_TPLUS/NORD_TDECAY); FAST_* in snn_mnist_stdp_fast.py:47-64.
Reference config for parity: NORD_M=300, NORD_TRAIN=6000 (the "76.0% at M=300/6k" run).

## R5. GPU strategy for the hard WTA (the design core)

**Decision**: EXACT-MIRROR the reference semantics using GeNN custom updates; do NOT substitute
biologically-styled synaptic inhibition.

- The parity criterion (±3 pp of the CPU figure) is the bar; an inhibitory-population WTA
  (Diehl&Cook two-population style) changes the mechanism (multiple winners per step, soft
  competition) and cannot promise parity — it would be a different learner wearing the port's
  name.
- GeNN 5 mechanism inventory that covers each piece:
  - custom NEURON model (sim/threshold/reset code) carries mem (mult-decay via BETA as a
    param-mapped exponential), theta, and a winner flag; threshold condition = "this neuron is
    the flagged winner" so the WINNER EMITS A TRUE SPIKE and the existing weight-update
    codegen's `post_spike_code` (the LTP) fires unchanged.
  - custom UPDATE with a REDUCTION computes argmax(mem - theta) per step and writes the winner
    flag (GeNN custom updates support reductions and run on demand from the host loop).
  - a second custom update performs the global membrane reset after a winning step, and a
    per-image reset update zeroes mem/traces between images.
  - row normalization: custom update over the weight-update model's `g` (per-row sum + scale) —
    the established GeNN pattern for Diehl&Cook-style normalization.
- Host loop consequence, stated honestly: the per-step sequence is
  `custom_update("wta"); step_time(); [reset if fired]` driven from Python — T=30 steps/image,
  so ~90 kernel launches/image ≈ 5.4M launches for the 60k payoff run. Kernel-launch overhead
  (~10-20 us) bounds this at minutes-scale; the win over CPU is the parallel state update at
  width M, which is the claim the payoff run MEASURES (P5: the crossover is a number). pygenn's
  normal usage is exactly this host-driven `step_time()` loop.
- **Numerics**: with dt=1.0, exact BETA-multiply decay is expressible directly in the custom
  neuron's sim_code (`V *= beta;` then add Isyn) — no exp() mapping error at all. The 3 pp
  parity band then absorbs float-order and atomics differences only.

**Alternatives considered**: inhibitory population WTA (rejected: parity unpromisable — recorded
above); per-step host readback of mem + host argmax (rejected: defeats the port — device-side
reduction exists); Vthresh built-in LIF as-is (rejected: no theta, no WTA — that is the
skeleton's known gap, the thing being finished).

## R6. Self-check design (CI has no card; green must stay falsifiable)

**Decision**: The gate-context self-check imports NO pygenn and asserts the codegen INTEGRITY;
the GPU run is an explicit opt-in.

- The C++ body strings (pre_dynamics/pre_spike/post_spike) move to module-level constants,
  importable with no toolchain. The self-check asserts the rule text carries the exact
  potentiation form (`lr * (preTrace - xtar)`), clamp bounds, and that the params fed to the
  rule are byte-equal to the CPU constants (FAST_LR, FAST_XTAR, WMAX, PRE_DECAY). Reachable
  red: mutate any constant or the C++ form → the check fails (demonstrated once, recorded).
- On a toolchain box, the self-check ALSO builds the model object (no training) to catch API
  drift; guarded so CI without pygenn still exits 0 via the stub path (existing behavior kept).
- The full train/assign/evaluate runs only via `python -m ...` with `GENN_RUN=1` (or CLI) —
  never inside the three-gate loop (a multi-minute GPU compile+train does not belong in
  run_selfchecks; the subprocess isolation contains the GIL flip either way, see R3).

## R7. Where the module lands + dispositions

**Decision**: Promote to `src/snn_data_lake/plasticity/snn_mnist_stdp_genn.py` (the rule is a
plasticity mechanism; `plasticity/` is the anatomical home for learning rules). REGIONS +
run_selfchecks registration in the same change. Dormancy disposition: `dormancy_gate.EXEMPT`
with reason "opt-in GPU training harness; runs via __main__/GENN_RUN on a GeNN box; gate
self-check covers codegen integrity" — same class as other benchmark harnesses. Check
`method_dormancy_gate` on the new module's public functions and add `EXEMPT_METHOD` entries
with dispositions if the learning-verb heuristic matches. Cross-tier imports (constants from
`reasoning/snn_mnist_stdp*`) are registered in `substrate/import_manifest.py` — the existing
pattern for governed→reasoning reaches.

**Parity oracle**: re-measure the CPU baseline ON THIS BOX at the pinned config
(NORD_M=300 NORD_TRAIN=6000 NORD_TEST=1000 SKIP_RATE=1, seed 0) during implementation, and
compare the GPU run against THAT freshly measured figure (recording both it and the historical
76.0%). Same box, same data, same config — dodges config archaeology and build-difference
noise; the historical figure remains the sanity anchor.

## R8. P2 non-duplication record (the ledger asked for this to be written down)

- `training/mnist_genn.py`: supervised gradient training (ml_genn EventProp) exporting serving
  weights for the core's `trained_brain` format; an UNVERIFIED scaffold. Different question:
  supervised readout training for serving vs unsupervised local-plasticity prototype learning
  at scale. Sole overlaps are the dataset and the backend — the ledger's documented
  different-question-same-vocabulary shape (`readout_boost` precedent). No merge owed by this
  lane. IF a future benchmark ever puts both on the same question with a metric, that is a new
  P2 event with its own measured pass.
- `reasoning/paradigm_b_genn.py`: GeNN as a query engine (coincidence detection), no learning
  at all. Not a duplicate of anything in this lane. Shared pygenn version constraints are the
  only coupling; the pre-flight's chosen tag (5.4.0) is recorded where all three consumers can
  cite it.
- e-prop invariant: untouched — the ported rule is local plasticity (pre trace x post spike),
  no gradient, no weight transport. `training/mnist_genn.py`'s EventProp debt belongs to that
  module's own lane.
