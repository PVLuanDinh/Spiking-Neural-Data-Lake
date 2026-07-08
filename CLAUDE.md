# Spiking Neural Data Lake — Claude Code Onboarding

## Overview

**Spiking Neural Data Lake** is a pure-stdlib, zero-dependency neuromorphic + robot-arm control loop + spiking data-lakehouse. Core codebase: ZERO runtime dependencies (`dependencies = []` in pyproject). That design is intentional and non-negotiable—never add a runtime dependency to core.

**Core entrypoints:**
- `src/snn_data_lake/` — 95 modules, importable as `snn_data_lake.<module>`
- Console scripts: `snn-demo`, `snn-arm`, `snn-loop`, `snn-ui`

## Layout

```
src/snn_data_lake/          95 core modules (zero-dep, vendored/stdlib only)
tests/                      pytest suite, >=70% coverage
research/                   optional research code, ruff-excluded
gcp/, infra/, robot arm/    optional integrations, ruff-excluded
local_llm/                  optional LLM backend, quarantined
packaging/                  PyInstaller GUI build, ruff-excluded
```

**Optional extras** (pyproject): `arm` (pyserial), `lake` (polars/pyarrow/deltalake), `gpu`, `nmnist`, `llm`, `genn`, `exe` (pyinstaller, build-time), `dev` (pytest/ruff/pre-commit), `all`.

## Three Gate Commands (all must stay green)

```bash
# 1. Smoke test — runs every core module's __main__ self-check
python run_selfchecks.py

# 2. Unit tests — >=70% coverage, uses .venv-lake
pytest

# 3. Lint — rules E, F, I, UP, B (E501/E702/E731/E741 relaxed on src/**)
python -m ruff check .
```

**CI matrix:** {ubuntu, windows} × {py3.11, 3.12, 3.13} + lint + coverage.

## Safety Envelope (Robot-Arm Control)

These five modules are safety-critical. Modify **only with explicit approval** and re-run module self-checks after:

- `reflex.py` — emergency stop reflex
- `serial_arm.py` — real hardware serial interface
- `arm_sim.py` — physics simulator (safety mirrors live arm)
- `arm_config.py` — calibration + limits
- `interpreter.py` — command validation

Every motor command flows through `ArmSim.apply()` safety gate. Real arm has calibration gate with no bypass.

## Determinism & Style Invariants

- **Seeding:** use `random.Random(seed)` (per-instance). Never module-level import-time seeding.
- **Hashing:** for persisted values, use `zlib.crc32()`, never Python's builtin `hash()`.
- **ASCII-only output:** Windows cp1252 crashes on non-ASCII. Keep `print()` statements ASCII.
- **Line length:** E501 relaxed on `src/**` intentionally—core is hand-aligned. Do NOT reflow.

## Build & Entrypoints

**Demo / data-lake / arm control:**
```bash
snn-demo              # Quick neuromorphic demo
snn-arm               # Real or simulated arm control
snn-loop              # Full closed-loop with brain + arm
```

**GUI (Windows .exe):**
```bash
pip install ".[exe]"
python packaging/build_exe.py
# Produces: dist/snn-ui.exe
```

**Full install with all optional deps:**
```bash
pip install ".[all]"
```

## Knowledge Graph

A graphify knowledge graph exists at `graphify-out/graph.json` for codebase orientation. Use:
```bash
graphify query "<your-question>"
graphify explain "<concept>"
graphify path "<module-A>" "<module-B>"
```

See `graphify-out/GRAPH_REPORT.md` and `graph.html` for full architecture review.
