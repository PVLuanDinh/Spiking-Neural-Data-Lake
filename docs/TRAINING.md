# Training & acceleration

`pip install .` pulls the core dependencies (`torch`, `numpy`); the former zero-dependency-core
contract was removed project-wide on 2026-07-20. **Training is a first-class, opt-in GPU tier** that
trains on CUDA and **exports its weights back into the core**, so an ordinary install serves
GPU-trained models.

## Regenerating the perception weights — read this first on a fresh clone

`models/` is **gitignored** (`.gitignore:76`) and has zero tracked files, because the checkpoints are
large binaries. A fresh clone therefore has **no real perception**: `cortex/brain.py:545` and `:577`
look for `models/eprop_nmnist.pt` and `models/eprop_shd.pt`, and without them the brain falls back to
`perception_tier=stdlib` with `perceiving_real=False` — a synthetic band instead of event-camera input,
and `LearnedAttention.attend` never called at all.

That is by design, but until 2026-07-28 it was written down nowhere, which made the whole perception
tier an unreproducible claim. The two commands that restore it:

```bash
.venv-cu128/Scripts/python training/eprop_nmnist.py    # -> models/eprop_nmnist.pt  (vision, N-MNIST)
.venv-cu128/Scripts/python training/eprop_shd.py       # -> models/eprop_shd.pt     (audio, SHD)
```

Both need their datasets on disk (`data/nmnist_cache`, `data/SHD`) and the GPU env `.venv-cu128`. To
point the brain at checkpoints elsewhere, set `SNN_EPROP_CKPT` / `SNN_EPROP_AUDIO_CKPT`. Anything that
reports a perception number should say which of these two states it measured in — see
[`COVERAGE_BASELINE.md`](COVERAGE_BASELINE.md).

## Install the training tier

```bash
pip install ".[train]"      # torch, snntorch, numpy
```

## Train on GPU, deploy zero-dep

`snn-train` trains a surrogate-gradient spiking classifier (snnTorch, BPTT) on the same 8×8
4-shape task as the core `snn_classifier`, then exports the learned weights in the core's `W[c][i]`
layout:

```bash
NORD_GPU=1 snn-train --gpu --epochs 20 --checkpoint model.pt --export-core weights.json
```

GPU is opt-in (`--gpu` or `NORD_GPU=1` / `NORD_DEVICE=cuda`); it falls back to CPU with a clear
message if CUDA is unavailable, and the CPU path is deterministic. The exported `weights.json` loads
straight into the zero-dep core:

```python
from snn_data_lake import snn_classifier as c
W = c.load_weights("weights.json")     # pure stdlib, no torch
currents, counts, synops = c.forward(W, sample, n_classes=4)
```

`load_weights` validates the format and shape at the trust boundary, so a mismatched or hand-edited
file fails loudly rather than silently misclassifying. This is the **train-fast-on-GPU,
deploy-zero-dep bridge**: the GPU optimizer improves the same `W[c][i]` readout the core runs.

**Honest caveat on the transfer.** The export is *format*-faithful, but the two inference paths are
not identical: the trainer optimizes LIF output-spike dynamics (leaky membrane + threshold), while
the core serves a *linear spike-count* readout (`currents[c] = Σ_i W[c][i]·count[i]`, argmax). Treat
the export as a GPU-optimized weight **init** and verify it in the core after export
(`load_weights` → `forward`/`evaluate`) rather than assuming the trainer's test accuracy carries over
unchanged.

## Training a model larger than VRAM (disk/CPU offload)

For models too big to fit in GPU memory, `--offload` trains with gradient checkpointing and a
CPU-offloaded optimizer (`snn_data_lake/training/offload.py`):

```bash
snn-train --offload --epochs 20
```

At production scale, use HuggingFace `accelerate` (`device_map`) or DeepSpeed ZeRO-Offload, which
stream optimizer/parameter state between GPU, CPU, and disk during the **backward** pass. This is
the correct mechanism for *training* something larger than VRAM.

## A real trained brain (MNIST) the whole stack uses

Train a real MNIST classifier fast on the GPU and load it into the **zero-dep core** so the
lakehouse, matcher, and arm loop can all call it:

```bash
snn-train --mnist --gpu --epochs 20 --hidden 256 --export-brain brain.json
```

The MNIST trainer (`snn_data_lake.training.mnist`) is a surrogate-gradient snnTorch net
(784 → hidden LIF → 10, mixed-precision, checkpointed) that **matches the core's serving semantics
exactly** — so unlike the 8×8 classifier bridge above, the exported weights transfer *faithfully*
(the trainer re-runs its export through the zero-dep core and reports the core-served accuracy next
to the torch one). Serve it with no torch:

```python
from snn_data_lake import trained_brain as tb
brain = tb.load_brain("brain.json")     # pure stdlib, validated
label = tb.predict(brain, pixels)       # pixels in [0,1], length 784
```

`trained_brain` is a general multi-layer spiking readout — rate-code the input, leaky-integrate-fire
hidden layers, accumulate output currents, argmax; a single-layer brain reduces exactly to the core
`snn_classifier`. This is the integration point: anything in the stack that needs a trained model
calls `trained_brain.predict`/`accuracy`, no GPU or torch at serve time.

### Faster training on GeNN (optional, best-effort)

`snn-train --mnist --genn` trains the same brain on GeNN's code-generated CUDA backend (often faster
than snnTorch for SNNs), exporting to the same `trained_brain` format. Needs the `[genn]` extra
(`pip install ".[genn]"`, plus a C++ compiler + CUDA). **Caveat:** `training/mnist_genn.py` is a
best-effort scaffold written against the ml_genn 2.x API and **not verified against a live GeNN
install** — version-sensitive spots are flagged `# VERIFY(ml_genn):`, and GeNN's LIF dynamics only
*approximate* the core's, so weights transfer more loosely than the snnTorch path (`verify_transfer`
reports the gap). Treat it as a starting point to iterate on your GPU, not a finished trainer.

## AirLLM — large-LLM local inference (not training)

AirLLM streams a large language model's transformer layers from disk one at a time, so a 70B+ model
runs *inference* in low VRAM. It powers the optional local NL-interface backend:

```bash
pip install ".[llm]"
python local_llm/run_local.py --model <hf-model-id> --prompt "..."
```

**Be clear on the distinction:** AirLLM is **inference** disk-streaming (fit a big model in low VRAM
to run it); the `--offload` path above is **training** disk/CPU-streaming (train a big model's
gradients larger than VRAM). AirLLM has no backward pass, so it is not a trainer.

## Reasoning: the SNN brain + a local LLM

> **Corrected 2026-07-28.** This section used to document `brain_reasoning.BrainReasoner` with a
> runnable import and two calls. **That module does not exist** — it was retired in the Policy-2/3
> wave and is recorded as such at `substrate/dormancy_gate.py:74`. The example would have raised
> `ModuleNotFoundError` for anyone who tried it. What follows is the path that actually exists.

The brain's language layer is `reasoning/spike_chat_lake.py::LakeChat`, reached live from
`cortex/live_brain.py:450`, with `reasoning/rose_language.py` (`cortex/brain.py:3349`) behind it. It
answers from the brain's own durable lake — learned facts, not a language model — and it is honest
about gaps rather than fabricating: a query with no supporting fact returns `known=False`.

AirLLM's role, if you want it, is unchanged and remains **optional and small**: it backs the
natural-language *command* front-end (`cortex/nl_interface.py` via `local_llm/airllm_backend.py`,
`[llm]` extra), classifying an English phrase into one of ~10 fixed labels. It is not in the
self-check gate and CI never installs it. See [AIRLLM_ANALYSIS.md](AIRLLM_ANALYSIS.md) for why that
is a poor fit for AirLLM specifically — the default backend is a deterministic stdlib keyword router.

## What needs what

| Capability | Command / entry | Requires |
|------------|-----------------|----------|
| Clone-and-run inference | `snn-demo`, `import snn_data_lake` | nothing (stdlib) |
| GPU training → export to core | `snn-train --gpu --export-core` | `.[train]` (torch, snntorch), a CUDA GPU |
| GPU MNIST brain → serve zero-dep | `snn-train --mnist --export-brain`; `trained_brain.predict` | `.[train]` to train; nothing to serve |
| Train larger than VRAM | `snn-train --offload` | `.[train]`, + `accelerate`/DeepSpeed for scale |
| GeNN-backend GPU training | `snn-train --mnist --genn` | `.[genn]` (pygenn+ml_genn, compiler+CUDA); best-effort scaffold |
| Reason over the brain in NL | `brain_reasoning.BrainReasoner(...).explain` | nothing (stdlib summary); `.[llm]` for a real LLM |
| Large-LLM local inference | `local_llm/run_local.py` | `.[llm]` (airllm, torch) |
| GPU SNN research training | `research/*` (`NORD_GPU=1`) | `.[gpu]` |

## Status

The zero-dep core, the `load_weights` bridge, and the clean "install the extra" messages are covered
by the standard CI/self-check gates. The **torch training tier is not in CI** (no GPU runner) — it is
code-reviewed and byte-compiled with a CPU fallback, but the CUDA path should be smoke-tested on your
own GPU on first use.
