# Performance & acceleration

The zero-dependency core (`src/snn_data_lake/`) is CPU / standard-library by design and
stays that way. Its modules are KB-scale spiking primitives and a real-time arm control loop —
they are memory- and latency-bound, not throughput-bound, so a GPU adds overhead without a
speedup. Acceleration lives in optional extras, applied only where it measurably helps:
large-batch neural-network **training** in `research/`.

## What each extra accelerates

| Extra | Installs | Accelerates | Mechanism |
|-------|----------|-------------|-----------|
| `[gpu]` | torch, torchvision, snntorch, bindsnet, numpy | the `research/` torch/snnTorch training scripts | CUDA via `torch.set_default_device` |
| `[genn]` | pygenn | the `research/*_genn` GPU SNN ports | GeNN compiles a CUDA backend directly |
| `[llm]` | airllm, torch | the optional NL interface (`local_llm/`) for **large** local models | AirLLM layer-by-layer disk streaming |

`pip install -e .` installs none of these EXTRAS, but it does pull the core dependencies:
`dependencies = ["torch", "numpy"]` (corrected 2026-07-28 -- this line quoted `dependencies = []`
as a literal citation long after the zero-dependency contract was removed).

## Using the GPU for training

The torch training scripts are **opt-in** GPU, defaulting to CPU so a plain run reproduces the
numbers of record exactly. Request the GPU with an environment variable (or the `--gpu` /
`--device` flags on `eth_mnist_bindsnet.py`):

```bash
# recent NVIDIA cards need the matching CUDA wheel FIRST (the extras pin bare `torch`).
# RTX 5070 = Blackwell sm_120 -> the cu128 wheel (the repo's verified GPU env = .venv-cu128, torch 2.11.0+cu128):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gpu]"          # or ".[train]" -- torch already satisfied by the cu128 build above
NORD_GPU=1 python research/snn_mnist_dc.py          # or NORD_DEVICE=cuda:0
```

`research/torch_device.py::resolve_device()` returns `cuda` only when the GPU is both requested
and `torch.cuda.is_available()`, else falls back to CPU with a clear message; `enable()` then
calls `torch.set_default_device(...)` so every tensor and module built afterwards lands on the
GPU. Scripts that construct an explicit `torch.Generator()` bind it to the resolved device so the
RNG matches the tensors.

Covered scripts: `snn_mnist_dc`, `snn_mnist_stdp`, `snn_mnist_stdp_fast`, `snn_moe_stdp_mnist`,
`snn_motor_control`, `snn_world_model_rl`, `snn_dreamer_6dof`, `snn_storage_core_snntorch`,
`snn_ssm_shd` (via the shared helper), and `eth_mnist_bindsnet` (its own `--gpu` flag).

## The GeNN ports

`research/snn_mnist_stdp_genn.py` and `research/paradigm_b_genn.py` use GeNN, which generates and
compiles a CUDA backend at build time (needs a C++ compiler + CUDA). They are drop-in GPU
accelerators for those specific models, not independently re-validated benchmarks.

## AirLLM (NL interface only)

AirLLM's value is disk-streaming a transformer whose weights do not fit in VRAM — it streams
layers from disk so a 70B+ model runs on a small GPU. That helps exactly one optional component:
the natural-language front-end backend (`local_llm/airllm_backend.py`). It is *not* a general
speedup — the core's spiking modules are KB-scale, so streaming them from disk is pure I/O
overhead. The zero-dependency `StubBackend` covers the default NL path; use AirLLM only if you
deliberately want a large local model. See [AIRLLM_ANALYSIS.md](AIRLLM_ANALYSIS.md).

## Neural capacity scales with real data (measured 2026-07-28)

How wide the brain grows is set by how much real structure it is given, not by a constant. NG-SNN
grows a hidden unit only while it still reduces error on rows the solve did not fit to; there is no
preset ceiling (`l_max=None` derives the hard bound as L <= N).

Streamed from Hugging Face (`ylecun/mnist`, 8x8 features, held-out on unseen rows, `patience=16`):

| train rows | earned capacity | held-out accuracy |
|---:|---:|---:|
| 1,000 | 53 | 0.794 |
| 2,000 | 1,997 | 0.929 |
| 2,500 | 126 | 0.879 |
| 5,000 | 160 | 0.901 |
| 10,000 | 2,334 | 0.956 |

(The 2,500/5,000 rows were measured at `patience=3`, the old default, and are kept to show what the
impatient setting cost -- 126 units and 0.879 where 2,000 rows at `patience=16` reached 1,997 and
0.929.)

**Two findings worth more than the table.**

A **fixed cap of 200** -- the previous default -- loses to earned capacity at every scale tried
(0.878 vs 0.929 at 2k rows; 0.908 vs 0.956 at 10k). It is an artificial ceiling that truncates real
learning, and it bound at exactly 200/200 on every synthetic task tried, including trivially easy
ones.

**A synthetic benchmark gave the wrong tuning.** On a 6-class prototype task, growing past a few
hundred units looked actively harmful, which argued for `patience=3`. Real data reversed it: capacity
kept paying an order of magnitude further. The synthetic task's tight clusters saturate quickly, and
generalising from them to real perception was the error. Tune capacity policy on the data the system
will actually see.

**Why this matters for the GPU, and where it stops.** The measured CUDA-graph crossover is N ~ 4,096
(see [POLICIES.md](POLICIES.md) P5): below it numpy wins, above it the RTX 5070 wins by 3-6x. Live
populations are 24-240 wide, which is why the live tick does not belong on the GPU. Grown capacity
gets far closer -- but it **saturates before the crossover on this task**, measured on the card:

| rows | grown L | held-out | fit time (RTX 5070) |
|---:|---:|---:|---:|
| 10,000 | 2,254 | 0.955 | 64 s |
| 16,000 | 2,576 | 0.957 | 269 s |
| 24,000 | **2,071** | **0.960** | 554 s |

Capacity peaks around 16k rows and then FALLS while accuracy still rises: 24k rows produced 20% FEWER
units and the best held-out score of the three. That is the earn-your-place rule behaving correctly --
a larger held-out slice locates the optimum sooner, so the pool stops earlier and generalises better.
Fewer neurons, better brain.

An earlier extrapolation from the steep 1k-10k segment suggested a few hundred thousand rows would
reach 4,096. These three points refute it: the curve is not merely flattening, it is non-monotonic and
saturated. MNIST at 8x8 runs out of exploitable structure around 2,000-2,600 units, and past that the
extra rows are repeats of what the pool already represents.

So the honest statement is: **capacity is data-limited, but only up to the richness of the task.**
Reaching GPU-relevant width needs a HARDER problem -- more classes, higher-dimensional input, real
temporal structure -- not simply more rows of an easy one. Note also that fit cost grows superlinearly
(4x for 1.6x data, the O(L^2 N) ridge solve), so the card is already earning its place on the growth
tier even below the elementwise crossover.

## Disk-streaming & growth (the lake scales out-of-core)

The spike lake is built to grow beyond RAM without adding a heavy dependency. Two layers:

- **Streaming sketches (pure stdlib, genuinely sublinear-memory, never store the stream):** HyperLogLog
  cardinality, DGIM sliding-window count (O(log^2 W)), reservoir sampling, AMS F2, P^2 quantiles, Count-Min +
  heavy hitters, MinHash. These summarize an unbounded stream in fixed tiny memory -- the crown jewels of the
  growth story.
- **On-disk spike store (pure stdlib, mmap/seek):** the `.spkl` streaming hub + the content-addressable recall
  index read/hold only a BOUNDED working set -- an old-range query SEEKS the file for the needed block instead
  of slurping the whole thing, so resident memory does not grow O(total-events) as the lake fills. (The seek-based
  indexed reader pattern is shared with `spike_telemetry_hub.load()`.)
- **`[lake]` medallion (optional, polars/pyarrow/deltalake):** a partitioned bronze/silver/gold Parquet/Delta
  lakehouse with the polars streaming engine (`scan_parquet(...).sink_parquet(...)`) + `optimize.compact()` +
  time-travel, for large-scale out-of-core recompute. Never touches the core.

Rule of thumb: keep resident memory bounded and let the disk hold the growth; add the `[lake]` extra only when
you outgrow the stdlib `.spkl` path.

## Status / caveat

The CPU path is unchanged and covered by CI. The GPU path is code-reviewed and byte-compiled, but
these torch scripts run in their own venv and are **not** part of the CI matrix (CI has no GPU), so
the CUDA path is not executed here — smoke-test it on your CUDA machine on first use.
