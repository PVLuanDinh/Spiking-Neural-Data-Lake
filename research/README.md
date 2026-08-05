# research/

**Under P4c (`docs/POLICIES.md`) there is no "outside the brain" in this project.** Everything in
this directory is either part of the brain or retired; *"it is only research"* is not a disposition.
This file used to open with *"Kept for reproducibility and history — **not** part of the shipped
product"*, which is exactly the framing P4c abolishes, and the rest of it had gone stale besides.

## What is actually here (verified against the tree, 2026-08-04)

`research/` holds **no top-level `.py` at all** — three subprojects and a literature corpus:

| path | what | status |
|---|---|---|
| `arc_agi/` | ARC-AGI-3-style interactive-reasoning harness + the live-arcade bridge. Drives real brain faculties (`CuriosityDrive`, `RaceAccumulator`, `MetacognitiveMonitor`, `vsa_reasoning`). | self-checking; `rule_inducer.py` and `brain_player.py` run with no optional deps |
| `ng_snn/` | NG-SNN reproduction (SoftHebb + LIF + ridge readout) and the N-MNIST disk-streaming loader `data.py`. | **DUPLICATED** — every `.py` here also exists under `src/snn_data_lake/reasoning/ng_snn/`. See below. |
| `neuromorphic_torch/` | torch + SpikingJelly subproject (SpikeSlicer, IS-SNN, LT-Gate, Max-Former) plus the GPU neuromorphic-chip emulators `neuro_chip_gpu*.py`. | needs `torch` (+ `spikingjelly` for the four phase modules) |
| `Predictive-coding/` | 26 papers (PDF + extracted markdown). Reference material, not code. | data |
| `conceptual-data-storage-method.md`, `training-snns-for-data-storage.md`, `brain_web.html`, `nengo_gui_*.cfg` | design notes and GUI layouts | data |

**The module list this file used to carry is gone, not moved-and-forgotten.** It named
`spiking_storage_prototype`, `hopfield_baseline`, `snn_moe_classifier`, `spiking_ssm`,
`spike_knowledge_graph`, `srm_neuron` and a dozen more as living here in two tiers, and gave
`PYTHONPATH=src python research/<name>.py` as the way to run them. The v3.78 reorg (`627b2e2b`)
moved every one of them into `src/snn_data_lake/reasoning/` (and several have since been promoted
further into anatomical regions). None of those files has been in this directory for months, and the
run line could not have worked for any of them.

## Running things here

Run from the **repo root**, with `src` importable:

```bash
PYTHONPATH=src .venv-lake/Scripts/python research/arc_agi/rule_inducer.py      # relational induction proof
PYTHONPATH=src .venv-lake/Scripts/python research/arc_agi/run_arc.py           # brain vs random on 4 toy games
.venv-lake/Scripts/python research/neuromorphic_torch/neuro_chip_gpu.py        # chip emulator smoke (needs torch)
.venv-cu128/Scripts/python research/ng_snn/nmnist_pipeline.py                  # needs torch + tonic + the dataset
```

## Open P4c debt in this directory, stated rather than filed away

- **`ng_snn/` is duplicated into `src/snn_data_lake/reasoning/ng_snn/`, and both copies are live.**
  Measured 2026-08-04 by parsing both and comparing ASTs with docstrings stripped: **7 of 12 files
  are AST-identical**; 4 of the remaining 5 differ ONLY by the governed copy having dropped an unused
  `import sys` and replaced `sys.path.insert(0, HERE)` with `pass  # migrated: ...`; the fifth
  (`softhebb_mnist.py`) has an identical body and only unused imports removed. So they are the same
  code — P2's "two mechanisms for one job" in its purest form.
  It is **not resolved here**, because P2's honesty guard forbids deleting before the survivor is
  proven equivalent-or-better and the *governed* copy is currently the WORSE one: dropping the
  `sys.path` bootstrap broke its own drivers, which still do `from model import NGSNN` by bare name
  (`docs/AUDIT_BACKLOG.md` H10 records this — four drivers raise `ModuleNotFoundError`, and
  `DATA_ROOT`/`CACHE_DIR` resolve into a `src/snn_data_lake/data` that does not exist). Retiring the
  working copy in favour of a broken one would be a regression wearing P2's clothes. Fixing the
  governed copy's imports is the prerequisite, and it lives in `reasoning/`.
- **`research/ng_snn/data.py` has two consumers outside this directory** —
  `training/surrogate_snn.py::ng_snn_data_file` and `training/eprop_nmnist.py::_ng_snn_dir`, both of
  which locate it by walking up from `__file__`. Those are the two encoders `cortex/brain.py` loads
  on the live construction path, so this directory is load-bearing for the brain's perception tier,
  not an archive.
