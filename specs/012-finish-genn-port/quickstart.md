# Quickstart: Validate the Finished GeNN Port

Every command from repo root, in the ONE gate env (`.venv-lake`). Full contract details:
[contracts/module-contract.md](contracts/module-contract.md). Every run's output lands verbatim
in [evidence.md](evidence.md).

## 0. Pre-flight (BLOCKING — T000 semantics)

```powershell
.venv-lake\Scripts\pip install -e .[genn]
.venv-lake\Scripts\python -c "import pygenn; print(pygenn.__version__)"
.venv-lake\Scripts\python tools_or_scratch\genn_smoke.py   # minimal model: build+compile+step (script created by the pre-flight task; deleted or kept per its task)
```

**Expected**: install succeeds; version prints (5.4.0); smoke model compiles via nvcc and steps
on the CUDA backend with sane output. **On ANY failure**: record verbatim in evidence.md,
STOP the feature, ledger row -> BLOCKED, owner decision. No port work commits.

## 1. Gates stay green at every commit (all three, both toolchain states)

```powershell
.venv-lake\Scripts\python run_selfchecks.py
.venv-lake\Scripts\python -m pytest
.venv-lake\Scripts\python -m ruff check .
```

The module's own self-check additionally passes with pygenn UNINSTALLED (CI shape) — verify at
least once by running it with the import forced off (the stub path) and confirm it still
ASSERTS the codegen cross-check (not vacuous green).

## 2. Same-box CPU oracle (once)

```powershell
$env:NORD_M="300"; $env:NORD_TRAIN="6000"; $env:NORD_TEST="1000"; $env:SKIP_RATE="1"
.venv-lake\Scripts\python -m snn_data_lake.reasoning.snn_mnist_stdp_fast
```

**Expected**: acc near the historical 76.0%. Record the measured figure — it is the parity
target.

## 3. Parity run (US2)

```powershell
$env:NORD_M="300"; $env:NORD_TRAIN="6000"; $env:NORD_TEST="1000"; $env:GENN_RUN="1"
.venv-lake\Scripts\python -m snn_data_lake.plasticity.snn_mnist_stdp_genn
```

**Expected**: |acc_gpu - acc_cpu_baseline| <= 3 pp. Determinism line printed (identical
encode -> identical trains). Wall-clock recorded.

## 4. Untrained control (US2 — the reachable red for "it learns")

```powershell
$env:GENN_LR0="1"   # same run, learning disabled
.venv-lake\Scripts\python -m snn_data_lake.plasticity.snn_mnist_stdp_genn
```

**Expected**: acc < 20%. If this scores like the trained run, the harness is measuring itself —
red, stop, fix.

## 5. Payoff runs (US3)

```powershell
$env:GENN_LR0=""; $env:NORD_TRAIN="60000"; $env:NORD_TEST="10000"
.venv-lake\Scripts\python -m snn_data_lake.plasticity.snn_mnist_stdp_genn          # M=300
$env:NORD_M="1600"
.venv-lake\Scripts\python -m snn_data_lake.plasticity.snn_mnist_stdp_genn          # M=1600
```

**Expected**: both complete; accuracy + wall-clock recorded beside the CPU loop's figures.
Whatever the numbers say is what the ledger gets.

## 6. Constructed red (once, recorded)

Temporarily mutate one rule constant (e.g. FAST_XTAR passed to the codegen) -> the self-check's
codegen cross-check FAILS -> revert. Record the demonstration in evidence.md (P8 debt: the
criterion can fail).

## 7. Close-out

- Module docstring: stale "cannot run on the dev box" replaced by measured environment facts.
- docs/P4C_BACKLOG.md ledger row: MANDATED BUILD -> BUILT+MEASURED (evidence pointers) or
  BLOCKED (verbatim blocker).
- `graphify update .` after code changes.
