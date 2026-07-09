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
| Train larger than VRAM | `snn-train --offload` | `.[train]`, + `accelerate`/DeepSpeed for scale |
| Large-LLM local inference | `local_llm/run_local.py` | `.[llm]` (airllm, torch) |
| GPU SNN research training | `research/*` (`NORD_GPU=1`) | `.[gpu]` |

## Status

The zero-dep core, the `load_weights` bridge, and the clean "install the extra" messages are covered
by the standard CI/self-check gates. The **torch training tier is not in CI** (no GPU runner) — it is
code-reviewed and byte-compiled with a CPU fallback, but the CUDA path should be smoke-tested on your
own GPU on first use.
