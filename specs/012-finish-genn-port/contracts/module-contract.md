# Module Contract: plasticity/snn_mnist_stdp_genn

The module's external surface after finishing. No public API beyond this — it is a harness, not
a library other faculties import (dormancy disposition: EXEMPT-with-reason).

## Import contract (any box, toolchain or not)

- `import snn_data_lake.plasticity.snn_mnist_stdp_genn` succeeds with NO toolchain: pygenn
  import stays guarded; `_HAVE_GENN` False → stub behavior. NO module-scope side effects
  (no chdir, no env mutation, no RNG seeding — the promotion-exposed-bugs precedent).
- C++ rule bodies + the param mapping are module-level constants importable without pygenn
  (the falsifiable half of the CI self-check).

## Self-check contract (run_selfchecks registration; isolated subprocess)

- No toolchain: prints the stub statement (requirements + where the rule is CPU-verified),
  ASSERTS codegen integrity (rule text carries `lr * (preTrace - xtar)`, clamp [0, wmax];
  params byte-equal to FAST_LR/FAST_XTAR/WMAX/PRE_DECAY), exits 0. Red is reachable by
  mutating either side.
- With toolchain: additionally constructs the GeNNModel (build objects, no nvcc compile, no
  training) to catch pygenn API drift; still bounded seconds, exits 0.
- NEVER trains in the gate. ASCII-only output.

## Run contract (explicit opt-in, GeNN box)

```
python -m snn_data_lake.plasticity.snn_mnist_stdp_genn            # stub OR full run
```

- Full run requires `_HAVE_GENN` and `GENN_RUN=1` (explicit opt-in; without it, __main__ on a
  toolchain box prints status + exits — a fresh clone's run_selfchecks must never hit a
  multi-minute nvcc compile by surprise).
- Config env vars (all shared with the CPU oracle, same names, same defaults):
  `NORD_M`, `NORD_TRAIN`, `NORD_TEST`, `NORD_TPLUS`, `NORD_TDECAY`, `FAST_THRESH`, `FAST_LR`,
  `FAST_XTAR`, `FAST_BURST`. New, this module only: `GENN_RUN=1` (run), `GENN_LR0=1`
  (untrained-control mode), `GENN_SEED` (default 0).
- Output: config header, per-phase progress, final `acc=<float>` + `train=<seconds>` lines
  (machine-readable enough for the evidence file), determinism-check line. Exit 0 on
  completion; nonzero + verbatim error on toolchain failure mid-run.

## pyproject contract

- `[genn]` extra installs on this box:
  `genn = ["pygenn @ https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip", "ml_genn"]`
  (final tag per pre-flight; ml_genn untouched unless the pre-flight measures it blocking the
  extra as a whole — then its own one-line fix with the measurement recorded).

## Gate contract

- REGIONS: registered under `plasticity`; `run_selfchecks.py`: registered; `dormancy_gate.py`:
  EXEMPT entry with the R7 reason; `method_dormancy_gate`: EXEMPT_METHOD entries only if the
  heuristic matches, each with a disposition; `import_manifest.py`: entries for the
  cross-tier constant imports (reasoning/snn_mnist_stdp, reasoning/snn_mnist_stdp_fast).
- ruff: module inside lint coverage, clean. pytest: coverage unaffected or improved.
