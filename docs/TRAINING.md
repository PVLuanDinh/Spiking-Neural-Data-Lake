# Training & acceleration

The core (`src/snn_data_lake/`) is zero-dependency and stays that way: `pip install .` gives a
clone-and-run inference library with no torch, no GPU, no compiled extensions. **Training is a
first-class, opt-in GPU tier** — the one place the zero-dependency rule is loosened — that trains
on CUDA and **exports its weights back into the zero-dep core**, so a clone-and-run install serves
GPU-trained models with nothing installed but the standard library.

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

## What needs what

| Capability | Command / entry | Requires |
|------------|-----------------|----------|
| Clone-and-run inference | `snn-demo`, `import snn_data_lake` | nothing (stdlib) |
| GPU training → export to core | `snn-train --gpu --export-core` | `.[train]` (torch, snntorch), a CUDA GPU |
| GPU MNIST brain → serve zero-dep | `snn-train --mnist --export-brain`; `trained_brain.predict` | `.[train]` to train; nothing to serve |
| Train larger than VRAM | `snn-train --offload` | `.[train]`, + `accelerate`/DeepSpeed for scale |
| Large-LLM local inference | `local_llm/run_local.py` | `.[llm]` (airllm, torch) |
| GPU SNN research training | `research/*` (`NORD_GPU=1`) | `.[gpu]` |

## Status

The zero-dep core, the `load_weights` bridge, and the clean "install the extra" messages are covered
by the standard CI/self-check gates. The **torch training tier is not in CI** (no GPU runner) — it is
code-reviewed and byte-compiled with a CPU fallback, but the CUDA path should be smoke-tested on your
own GPU on first use.
