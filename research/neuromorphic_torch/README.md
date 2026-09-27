# neuromorphic_torch — trained-SNN subproject (torch + SpikingJelly)

A torch/SpikingJelly subproject implementing the framework-heavy neuromorphic phases (SpikeSlicer,
IS-SNN embeddings, LT-Gate memory, Max-Former pooling), plus the GPU neuromorphic-chip emulators
`neuro_chip_gpu.py` / `neuro_chip_gpu_stream.py`.

> **Corrected 2026-08-04 (P7 + P4c).** This section previously read: *"The product core
> (`src/snn_data_lake/`) is **pure standard library, zero runtime dependencies** — that is the point
> of the repo (see `../../README.md`, `pyproject.toml: dependencies = []`)."* Every clause of that
> is now false. `pyproject.toml` declares `torch`, `numpy` and `mujoco==3.11.0` as UNCONDITIONAL
> core dependencies; the zero-dependency rule was abolished project-wide on 2026-07-20 and P7
> replaced it with *"any dependency that helps this become a phenomenal brain is accepted"*. It also
> said *"Nothing here is imported by the core package … so it can never break clone-and-run"*, which
> P4c rejects as a disposition: **there is no "outside the brain"** — a file in this tree is part of
> the brain or it is retired. The isolation described below is a real fact about which venv the
> phase modules need, and it is kept on those terms; it is not a statement that this directory is
> exempt from anything.

- It needs **torch** (already a core dependency) and, for the four phase modules, **SpikingJelly**
  (optionally **cupy/CUDA 12.x** for the accelerated backend). `neuro_chip_gpu*.py` need torch only.
  SpikingJelly is a real adoption, not a quarantine: `docs/POLICIES.md` P7 records it as *adopted*
  into this directory on 2026-07-02. A separate venv keeps the deterministic gate env (`.venv-lake`)
  free of it:

```bash
cd research/neuromorphic_torch
python -m venv .venv-neuro
.venv-neuro\Scripts\activate          # Windows  (source .venv-neuro/bin/activate on Linux/mac)
pip install -r requirements.txt
python smoke_check.py                 # env + per-phase smoke (CPU fallback if no cupy)
```

## Reuse over reinvention (P2: one definition, not two)

The encoders and metrics already in the core are **reused**, not reimplemented in torch:

- Van Rossum distance (tunable `tau`), TTFS `encode_latency` → `snn_data_lake.spike_preprocessing`
- Population coding (RBF receptive fields) → `snn_data_lake.population_encoding`
- Spike-native in-storage query engine → `snn_data_lake.paradigm_b_engine`
- Victor-Purpura distance (tunable `q`) + hierarchical proximity index → core `snn_data_lake` (Phase 1)
- **Energy constants** (`E_AC_FP32_45NM`, `E_SYNOP_LOIHI`) → `snn_data_lake.instruments.energy`.
  Added to this list 2026-08-04: `neuro_chip_gpu.py` and `neuro_chip_gpu_stream.py` each carried
  their own literal copy, under the comment *"hardcoded to keep zero-dep on the core"* — three live
  definitions of two numbers, in files whose whole claim is that their energy accounting is the
  core's. The reason was void (P7) and the copies were free to drift; they are now imported. All
  three were measured equal before the swap, so no published number moves.

Torch is used **only** for the pieces that genuinely need training / CUDA kernels — which is a
statement about where torch *helps*, not about a dependency budget (P7 removed the budget).

## Phase map

| Phase | Module | What |
|-------|--------|------|
| 1 (core — not here) | `snn_data_lake.spike_preprocessing` / `proximity_index` | Van Rossum (reuse) + Victor-Purpura + hierarchical index |
| 2 | `encoding/spikeslicer.py` | SpikeSlicer middleware: lightweight SNN trained with SPA-Loss to slice event streams; fed by the stdlib TTFS/population encoders |
| 3 | `embedding/is_snn.py` | IS-SNN embedding trained natively via surrogate-gradient STBP (no ANN→SNN); intrinsically-stable init (no BatchNorm); `cupy` backend default, CPU fallback |
| 4 | `memory/lt_gate.py`, `memory/maxformer_pool.py` | LT-Gate dual-time-constant neurons (anti-forgetting) + Max-Former max-pool (anti-low-pass) |

Phase modules are added one at a time, each stopping for review. `query/` stays a thin shim over the core VP index.
