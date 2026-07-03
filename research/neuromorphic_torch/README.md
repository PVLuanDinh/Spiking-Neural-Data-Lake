# neuromorphic_torch — trained-SNN subproject (torch + SpikingJelly, quarantined)

A **self-contained torch/SpikingJelly subproject** implementing the framework-heavy neuromorphic phases
(SpikeSlicer, IS-SNN embeddings, LT-Gate memory, Max-Former pooling). It is deliberately **isolated from the
product core**:

- The product core (`src/snn_data_lake/`) is **pure standard library, zero runtime dependencies** — that is the
  point of the repo (see `../../README.md`, `pyproject.toml: dependencies = []`). This subproject does **not**
  change that.
- Nothing here is imported by the core package, registered in `run_selfchecks.py`, or run by the GitHub CI matrix
  (`{ubuntu,windows} × {3.11,3.12,3.13}`, no GPU). So it can never break clone-and-run or red the core CI.
- It needs **torch + SpikingJelly** (and optionally **cupy/CUDA 12.x** for the accelerated backend), installed in
  its **own venv**:

```bash
cd research/neuromorphic_torch
python -m venv .venv-neuro
.venv-neuro\Scripts\activate          # Windows  (source .venv-neuro/bin/activate on Linux/mac)
pip install -r requirements.txt
python smoke_check.py                 # env + per-phase smoke (CPU fallback if no cupy)
```

## Reuse over reinvention

The **stdlib encoders and metrics already in the core are reused**, not reimplemented in torch:

- Van Rossum distance (tunable `tau`), TTFS `encode_latency` → `snn_data_lake.spike_preprocessing`
- Population coding (RBF receptive fields) → `snn_data_lake.population_encoding`
- Spike-native in-storage query engine → `snn_data_lake.paradigm_b_engine`
- Victor-Purpura distance (tunable `q`) + hierarchical proximity index → core `snn_data_lake` (Phase 1, stdlib)

Torch is used **only** for the pieces that genuinely need training / CUDA kernels.

## Phase map

| Phase | Module | What |
|-------|--------|------|
| 1 (core, stdlib — not here) | `snn_data_lake.spike_preprocessing` / `proximity_index` | Van Rossum (reuse) + Victor-Purpura + hierarchical index |
| 2 | `encoding/spikeslicer.py` | SpikeSlicer middleware: lightweight SNN trained with SPA-Loss to slice event streams; fed by the stdlib TTFS/population encoders |
| 3 | `embedding/is_snn.py` | IS-SNN embedding trained natively via surrogate-gradient STBP (no ANN→SNN); intrinsically-stable init (no BatchNorm); `cupy` backend default, CPU fallback |
| 4 | `memory/lt_gate.py`, `memory/maxformer_pool.py` | LT-Gate dual-time-constant neurons (anti-forgetting) + Max-Former max-pool (anti-low-pass) |

Phase modules are added one at a time, each stopping for review. `query/` stays a thin shim over the core VP index.
