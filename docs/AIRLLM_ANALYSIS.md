# AirLLM and the Spiking Neural Data Lake: an infrastructure analysis

**Question:** can AirLLM (https://github.com/lyogavin/airllm) infrastructure run the
"full core program" of this repo locally, and what does that actually require?

**Short answer:** the question is backwards. The full core already runs locally on its
**two declared dependencies** (torch, numpy) and does not use AirLLM at all. AirLLM powers exactly one
**optional** add-on: the natural-language front-end (`nl_interface` via
`local_llm/airllm_backend.py`). For that job — classifying an English phrase into one
of ~10 fixed labels — AirLLM works, but it is heavy machinery for a small task. The
built-in `StubBackend` or a small instruct model covers it; AirLLM only earns its keep
if you deliberately want a genuinely large (70B+) local model and accept the disk and
latency cost.

All claims below were checked against the AirLLM README at `main` and the PyPI page
(airllm 3.0.1, released 2026-06-30) and against this repo at v2.13.0. Anything that
could not be verified is flagged in the "Unverified claims" section.

---

## 1. What AirLLM actually is

Credit: AirLLM by Gavin Li (lyogavin), https://github.com/lyogavin/airllm, MIT-style
open source. Verified against its README and PyPI:

- **Layer-streaming inference.** AirLLM runs very large LLMs on tiny VRAM by keeping
  **one transformer layer on the GPU at a time**: it loads a layer's weights from
  disk, runs it, frees it, loads the next. Peak VRAM tracks the *layer* size, not the
  *model* size. Headline numbers from the project: Llama 3.x 70B on a 4GB GPU, Llama
  3.1 405B on 8GB, DeepSeek-V3 (671B) on ~12GB, Qwen3-235B (MoE) on ~3GB (v3.0 adds
  FP8 support). No quantization is required; optional 4-bit/8-bit *block-wise
  compression* (via `bitsandbytes`) exists purely to speed up the disk-load bottleneck
  (README claims up to 3x).
- **The tradeoff is disk I/O, and AirLLM says so itself.** The README states the
  bottleneck is "mainly at the disk loading". Every forward pass re-streams every
  layer from disk, so **per-token latency is high by construction** — this is the real
  ceiling, not VRAM. Prefetching overlaps load and compute for roughly a 10% gain.
  Exact tokens/sec figures are not published by the project and are not invented here;
  qualitatively, interactive multi-second-or-worse per-token latency should be
  expected for large models, worse on CPU and on spinning disks.
- **Disk footprint is full-size.** Weights live on disk at full size (a 70B model in
  fp16 is ~140GB by arithmetic: 2 bytes/param). On first inference AirLLM additionally
  **decomposes the model into per-layer shards** in the Hugging Face cache — the
  README warns to ensure sufficient disk space and offers `delete_original=True` to
  reclaim the un-split copy (i.e. transiently up to ~2x model size on disk).
- **Real dependency surface.** `pip install airllm` pulls (verified from its
  `setup.py`): `torch>=2.4`, `transformers>=4.49,<5.13`, `accelerate>=1.0`,
  `safetensors`, `huggingface-hub`, `scipy`, `sentencepiece`, `tqdm`. That is a
  multi-GB install (torch alone). `bitsandbytes` is optional.
- **CPU and platforms.** CPU inference is supported since v2.10.1 (2024-08). MacOS is
  supported via `mlx`. **Windows is not mentioned anywhere in the README** — since the
  stack is plain PyTorch it plausibly works, but it is not documented or tested by the
  project (flagged below).
- **First run needs the network** to download weights from Hugging Face; after
  caching, inference is fully offline.
- **API.** HF-style: `from airllm import AutoModel; m = AutoModel.from_pretrained(id);
  m.generate(...)`. The README example is GPU-centric (`input_ids.cuda()`).

## 2. Does the FULL CORE need AirLLM? No.

Verified in this repo:

- `pyproject.toml`: `dependencies = ["torch", "numpy"]` — the core needs those two and
  nothing else. AirLLM appears only in the optional extra `llm = ["airllm", "torch"]`.
- `python run_selfchecks.py` runs **277 modules** (195 CORE product modules +
  82 RESEARCH PoCs + torch/nengo-gated) with no GPU, no network, no LLM. The whole core program — the
  robot-arm control loop (`reflex`, E-STOP, collision-gated `arm_sim`/`serial_arm`,
  `interpreter`, `autonomous_loop`, `closed_loop`), the spiking lakehouse
  (`spike_telemetry_hub`, `streaming_hub`), and the cockpit (`substrate/cockpit`, the ONE
  frontend, shipped as a Windows .exe) — is in the CORE list and runs without AirLLM.
- `local_llm/airllm_backend.py` is **not** in the self-check gate, and CI never
  installs the `[llm]` extra (verified: no `airllm`/`[llm]` reference in
  `.github/workflows/ci.yml`). Its own smoke test skips gracefully when
  airllm/torch are absent.
- `local_llm/run_local.py` runs the full local stack and **defaults to
  `StubBackend`**, a deterministic stdlib keyword router. AirLLM is opt-in via
  `SNN_LLM=1` or `--llm`, and falls back to the stub if the import fails.

So "run the full core program locally with AirLLM" reduces to: the core runs locally
regardless; AirLLM can additionally power the optional English-command front-end.

## 3. What AirLLM is asked to do here — and the mismatch

The backend contract (`nl_interface`) is: `.complete(prompt) -> str` returning **one
label** out of ~10 (`close, open, hold, home, retract, stop, rotate_a, rotate_b,
query, reject`). The label is then mapped to a *fixed, repo-owned* command constant,
re-validated against the serial allow-list, or rejected. That is a tiny
classification/routing task.

AirLLM's entire value proposition is the opposite end of the spectrum: running
70B-671B models that don't fit in VRAM. The mismatch is stark:

- The repo's own default model for the AirLLM backend is
  **Qwen/Qwen2.5-0.5B-Instruct** — a model small enough to load *fully* in plain
  `transformers` on nearly any machine (~1GB of weights). At that size AirLLM's
  layer-streaming solves a problem that does not exist, while still costing the
  torch/transformers install, the HF download, and the layer-decomposition step.
- For the models where AirLLM genuinely shines (70B+), per-token disk-streamed latency
  makes an *interactive* arm/lake control loop sluggish — a poor fit for a front-end
  whose output is a single word.
- The `StubBackend` already covers the shipped demo phrases deterministically, at zero
  install cost, and is what CI proves.

**When AirLLM helps here:** the user specifically wants a *large* local model (better
paraphrase/multilingual robustness, or they simply want their local 70B doing the
routing), has hundreds of GB of disk, tolerates latency, and needs everything offline
after the first download. **When it is overkill:** everything else — label routing is
fully served by the stub or by a 0.5B-3B instruct model loaded normally.

## 4. If you do want AirLLM locally: concrete config

```bash
pip install ".[llm]"                  # pulls airllm + torch (airllm itself brings transformers, accelerate, ...)
SNN_LLM=1 python local_llm/run_local.py   # or: python local_llm/run_local.py --llm
```

- **Model:** keep the default `Qwen/Qwen2.5-0.5B-Instruct` (~1GB-class download,
  CPU-viable) unless you have a reason to go big. To go big, pass a larger `model_id`
  to `AirLLMBackend` — a 70B fp16 model means roughly **140GB+ of weights plus the
  layer-split copy** in the HF cache (use `delete_original` to halve it), a 4GB-VRAM
  floor on GPU, and high per-token latency.
- **CPU vs GPU:** CPU works (airllm >= 2.10.1) but is slow; any CUDA GPU with >= 4GB
  VRAM improves matters. A fast NVMe SSD matters more than usual because the
  bottleneck is layer loading.
- **First run:** requires network for the HF download + a one-time layer-decomposition
  pass; subsequent runs are offline from cache.
- **Determinism and safety are preserved by the repo, not by the LLM:** the backend
  generates greedily (`do_sample=False`) and reduces free-text output to a label by
  scanning for label words, returning `reject` on anything unrecognized or on any
  inference error. `nl_interface` then re-validates every arm command against
  `serial_arm.is_valid_command` (its self-check includes adversarial-backend injection
  tests). The LLM can never emit a raw command; the worst a misbehaving model can do
  is pick the *wrong allow-listed* action or get rejected — and every motor command
  still passes through the `ArmSim`/`SerialArm` collision/E-STOP/reflex envelope.
  Note the honest limit: the firewall guarantees *safety*, not *correctness* — a
  mislabel like "open" instead of "hold" is still an allow-listed, safety-gated move.

## 5. Risks, gaps, failure modes

- **Disk churn + latency (structural):** every token re-streams all layers from disk.
  This is AirLLM's real ceiling and it cannot be configured away, only mitigated
  (compression, prefetch, NVMe).
- **Disk exhaustion on first run:** model download + layer-split copy; the AirLLM
  README's own FAQ names out-of-disk as a common failure. Budget ~2x model size
  transiently.
- **First-run network dependency** and multi-GB `pip install` (torch); on Windows,
  torch installs are usually fine, but `bitsandbytes` (only needed for the optional
  compression flag) has a history of Windows friction.
- **Windows support for AirLLM itself is undocumented** by the upstream project (no
  README mention). The repo's own core and StubBackend path are Windows-proven; the
  AirLLM path on Windows is plausible-but-unverified.
- **Model rambling:** mitigated by design — the backend never trusts free text; it
  scans for a label and otherwise returns `reject`.
- **No determinism guarantee from the LLM:** greedy decoding helps, but only the
  allow-list firewall makes the system safe; treat the LLM as untrusted input.
- **API drift:** `airllm`'s `generate()` surface has shifted across versions (the
  current README example passes `use_cache`/`return_dict_in_generate` and calls
  `.cuda()`; the backend calls a plainer signature and indexes `output_ids[0]`). The
  `[llm]` extra pins nothing, and CI never runs this path, so an airllm or
  transformers release can silently break the backend. The failure mode is at least
  benign: any exception in `complete()` returns `reject`.

## 6. Decision guide

| Your need | Use | Cost |
|---|---|---|
| Deterministic label routing, zero extra install, CI-provable, offline | `StubBackend` (the default) | none beyond the core install |
| More robust free-text phrasing on modest hardware | Small instruct model (0.5B-3B), via the AirLLM backend's default Qwen2.5-0.5B or plain `transformers` | ~multi-GB pip install, ~1GB weights, CPU-viable; AirLLM's layer-streaming adds no value at this size |
| A genuinely large local model (70B+) doing the routing; offline after download; privacy | AirLLM with a large `model_id` | 100s of GB disk, first-run download + split, high per-token latency, >= 4GB VRAM (or slow CPU) |

## 7. Claims not verified here

- Any exact tokens/sec or ms/token latency number for AirLLM (upstream publishes
  none; none are asserted).
- AirLLM on **Windows** (undocumented upstream; untested in this analysis).
- That airllm 3.0.1 cleanly runs `Qwen/Qwen2.5-0.5B-Instruct` specifically — v3.0's
  README lists Qwen3, Llama 3.x/4, DeepSeek V2/V3, Phi-4, Gemma; Qwen2.5 is the same
  architecture family as models AirLLM has long supported, but this exact pairing was
  not executed here (nothing was downloaded or run, per this task's constraints).
- `docs/RUNNING_LOCAL.md`'s "~500MB" download figure for Qwen2.5-0.5B (bf16 weights on
  HF are closer to ~1GB) and its "~5-10s per inference on CPU" estimate — plausible
  repo estimates, not re-measured.

## 8. Verdict

The full core program of the Spiking Neural Data Lake **already runs locally on `torch` +
`numpy` alone** — 277 self-check modules, the arm safety loop, the lakehouse, and
the GUI need no GPU, no network, and no LLM, so "can AirLLM run the core" is moot:
nothing in the core requires it. AirLLM is a legitimate, verified-real piece of
infrastructure whose specific talent — streaming 70B-671B models layer-by-layer
through tiny VRAM at the price of disk-bound per-token latency — is aimed at a much
bigger problem than this repo's NL front-end has: picking one of ten labels. For that
job the pragmatic defaults are the built-in `StubBackend` (deterministic, CI-proven)
or a small instruct model; AirLLM is the right choice only for users who specifically
want a large local model doing the routing and will pay for it in disk and latency.
The integration as shipped is honest about this: opt-in, lazily imported, stub
fallback, and firewalled so that no model — big, small, or broken — can ever emit a
raw arm command.
