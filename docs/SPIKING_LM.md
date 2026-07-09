# Growing the spiking language model

The repo ships a **from-scratch all-spiking language model** (SpikeGPT / SpikingBERT lineage): a
transformer whose neurons are LIF spiking units. Two halves:

- **Trainer** — `src/snn_data_lake/training/spiking_lm.py` (torch/snnTorch, GPU, the `[gpu]` extra).
  Char-level, embedding → spiking blocks (LIF + an RWKV-style spiking token-mixer that approximates
  linear attention) → spike-driven readout, trained with surrogate-gradient BPTT.
- **Serve** — `src/snn_data_lake/spiking_lm_serve.py` (**zero-dependency**). Loads the trainer's
  exported `snn_lm/1` weight file and runs the same spiking forward in pure Python, so a clone-and-run
  install generates text from a trained spiking LM with **no torch**.

```bash
pip install ".[gpu]"                                    # torch + snntorch
snn-train --lm --corpus mytext.txt --gpu --export-lm lm.json
```
```python
from snn_data_lake.spiking_lm_serve import load_lm, generate
print(generate(load_lm("lm.json"), "the ", n=200))      # zero-dep generation
```

**Honest scale check.** As shipped this is a *tiny research toy* (d_model 96, 2 blocks, seq 64,
char-level). A single RTX 5070 (12 GB) can grow it to a *small* model (~tens of millions of params),
**not** a frontier LLM. SpikeGPT reached 216 M params on ~45 GB of text with real compute. The steps
below are the honest growth path — each is a real lever, none is a shortcut to a large model.

## The growth levers (in the order that pays off)

1. **Real corpus, not the toy.** `--corpus big.txt` (e.g. TinyStories, WikiText, your own text). More
   and more-varied text is the single biggest quality lever. Perplexity is the metric — track it.
   Pull one straight off the web with the firewalled fetcher (stdlib, no cloud):
   `snn-fetch-corpus https://www.gutenberg.org/files/11/11-0.txt alice.txt` then
   `snn-train --lm --corpus alice.txt`. The fetcher (`snn_data_lake.substrate.web_ingest`) is
   http/https + public-IP only and is deliberately outside the arm control loop — it just writes a
   local text file.
2. **Subword tokenizer.** Char-level caps quality and lengthens sequences. Moving to BPE/subword (e.g.
   a `tokenizers` BPE) shortens sequences and raises the ceiling. *(The char tokenizer is the toy default;
   a subword tokenizer is the first real upgrade — it changes `vocab`/`stoi`/`itos` only.)*
3. **Scale the config.** `--d-model` 96→256→512, `--blocks` 2→6→12, `--seq-len` 64→256→1024. Params ≈
   `3·d²·blocks + 2·vocab·d`. Bigger context + depth is where capacity comes from.
4. **Fit it in VRAM.** BPTT memory grows with `seq_len × blocks`. Levers, cheapest first: mixed
   precision (already on for `--gpu`), **gradient checkpointing** (`torch.utils.checkpoint` over the
   time unroll — recompute activations in the backward pass), smaller batch, and the **`--offload`**
   path (CPU/disk optimizer state) for a model larger than VRAM.
5. **Truncated BPTT for long context.** The RWKV-style mixer is O(T) *forward*, but full BPTT memory is
   O(T). Truncating the backward window (carry the mixer `state` forward, backprop only the last K
   steps) keeps memory bounded as `seq_len` grows — the key trick for long-context spiking LMs.
6. **Train longer + better.** More steps, an LR warmup+cosine schedule, gradient clipping, and
   `--checkpoint`/`--resume` so a multi-hour/day run survives interruption.
7. **Spiking-specific quality.** Tune `beta` (membrane decay) and the surrogate gradient; upgrade the
   token-mixer from the current simplified linear-attention approximation toward SpikeGPT's fuller
   **spiking-RWKV time-mixing** (the biggest architecture lever); exploit spike sparsity for efficiency.
8. **Scale out.** For beyond one GPU: DDP/FSDP, or DeepSpeed ZeRO / `accelerate` offload for very large
   parameter/optimizer state.

## How to tell it's working

- **Perplexity** on held-out text going down (`perplexity(...)` in the trainer).
- **Samples** getting more coherent (`generate(...)`, torch or the zero-dep serve).
- Compare against a **dense** transformer of the same parameter count trained the same way — the honest
  question for any spiking model is "how close to the dense baseline, at what sparsity/energy."

## The train → serve bridge

`export_lm` writes the trainer's weights as a `snn_lm/1` JSON that `spiking_lm_serve` reads and runs
zero-dep — so you train on the GPU and *deploy* the spiking LM with nothing but the standard library.
**Caveat:** the pure-Python serve matches the trainer's LIF dynamics by construction, but the torch
trainer is GPU-only and not run in CI, so smoke-test the exported model through
`spiking_lm_serve.generate` on first use (the trainer's `verify_transfer` does exactly this).

**Bottom line:** this is a real, growable spiking LM with an honest path to a *small* trained model —
not a promise of a large LLM. The dense LLM you run with AirLLM is a different thing; this one you grow
from scratch, spike by spike.
