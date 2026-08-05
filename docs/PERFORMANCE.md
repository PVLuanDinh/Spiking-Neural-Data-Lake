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
NORD_GPU=1 python -m snn_data_lake.reasoning.snn_mnist_dc     # or NORD_DEVICE=cuda:0
```

> Path corrected 2026-07-30: `research/` became `src/snn_data_lake/reasoning/` in v3.78 (`627b2e2b`),
> so the runnable form is `python -m snn_data_lake.reasoning.<module>`. Every bare module name listed
> in "Covered scripts" below lives there now.

`reasoning/torch_device.py::resolve_device()` returns `cuda` only when the GPU is both requested
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

Streamed by `substrate/hf_stream.py` (`ylecun/mnist`, 8x8 features, `delta=8`), scored on the
dataset's own `test` split -- 2,000 rows the fit never sees, majority-class floor 0.117:

| train rows | grow to the bound | `patience=16` | `patience=48` (default) | fixed cap 200 |
|---:|---:|---:|---:|---:|
| 2,000 | L=2,000 · 0.901 | L=392 · 0.868 | **L=640 · 0.883** | L=200 · 0.826 |
| 10,000 | L=9,984 · 0.951 | L=1,520 · 0.937 | **L=2,096 · 0.939** | L=200 · 0.871 |

**An earlier version of this table was wrong, and how it was wrong is the useful part.** It reported
capacity "held-out on unseen rows" while the score came from rows the growth solve had already fit --
the validation slice was inside the ridge solve. In-sample residual falls monotonically under greedy
pursuit, so the stop could never fire: pre-fix, growth ran to the hard bound at *every* scale (2,000
of 2,000 rows; 3,000 of 3,000; 9,984 of 10,000) and those cap hits were being reported as earned
capacity. The per-row curve that used to sit here is withdrawn rather than patched, because every row
of it was scored the same broken way.

**Three findings worth more than the table.**

A **fixed cap of 200** loses to earned capacity at every scale tried, and this is the one claim that
survived every correction (0.826 vs 0.883 at 2k rows; 0.871 vs 0.939 at 10k). It is an artificial
ceiling that truncates real learning, and it bound at exactly 200/200 on every synthetic task tried,
including trivially easy ones.

**Honest early stopping costs accuracy here, and that is worth stating plainly.** Growing to the
bound really is the most accurate setting on this data -- the extra units were not merely memorising,
they kept paying right up to one-per-row. What the corrected stop buys is a 3-5x smaller model for
1-2 points, plus the capability the old code lacked entirely: on labels with no signal in them it
stops at 16 units instead of spending all 600, while still taking 256 on learnable data of identical
shape. That 16x signal-to-noise gap was 1.0x before the fix.

Every number in this section is reproducible with `tools/capacity_curve.py`
(`control` | `grid` | `saturation`) -- tracked precisely because the previous version of this table
could not be re-derived by anyone who doubted it.

**Two constants were tuned against that broken measurement**, so both were re-derived. `patience=48`
because 48 and 128 give identical results at both scales -- past 48 the contribution criterion
saturates on its own and patience stops being the binding constraint, which makes 48 the point the
data picks. And the stop scores held-out **RMSE**, not the argmax accuracy the classifier is graded
on: scoring argmax directly was tried and measured worse (L=272 · 0.859 and L=688 · 0.918), because
accuracy over a few hundred rows is a coarse step function that sits on plateaus long enough to
exhaust patience while the residual still registers the gains underneath. The smooth proxy is the
better stopping signal precisely because it is smooth.

**Why this matters for the GPU, and where it stops.** The measured CUDA-graph crossover is N ~ 4,096
(see [POLICIES.md](POLICIES.md) P5): below it numpy wins, above it the RTX 5070 wins by 3-6x. Live
populations are 24-240 wide, which is why the live tick does not belong on the GPU. Grown capacity
**reaches the crossover**, measured on the card (`tools/capacity_curve.py saturation`):

| rows | grown L | test acc | fit time (RTX 5070) |
|---:|---:|---:|---:|
| 10,000 | 2,096 | 0.939 | 6 s |
| 16,000 | 2,808 | 0.948 | 94 s |
| 24,000 | **4,512** | **0.955** | 494 s |

Capacity and accuracy both rise monotonically, and at 24k rows the pool is 4,512 wide -- past the
N ~ 4,096 elementwise crossover. None of the three is a bound hit (`l_max` was 8,000 / 12,800 /
19,200 respectively), so each is a width the data actually earned.

**This reverses what this section used to say**, and the reversal is instructive. The previous table
reported capacity peaking near 16k and then FALLING to 2,071 at 24k, and concluded that MNIST at 8x8
"runs out of exploitable structure around 2,000-2,600 units" and that reaching GPU-relevant width
would need a harder problem -- more classes, higher-dimensional input, real temporal structure. That
non-monotonic peak was an artifact of the leaked stopping criterion described above, not a property
of the data. With the split made real the curve is monotone and clears the crossover on the same
easy task. The lesson is narrower and more useful than the one it replaces: an apparent saturation
is a claim about the STOPPING RULE at least as much as about the data.

Fit cost grows superlinearly (5x for 1.5x data, the O(L^2 N) ridge solve), so the card earns its
place on the growth tier at every scale here -- and now for the direct reason too, not only the
indirect one.

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

## Disk-tier measurements (2026-07-29)

Measured before touching anything, per the neuter-and-measure rule. Live daemon (pid 804, wake+sleep,
45 s py-spy @ 200 Hz, 7,319 samples) plus offline benchmarks on `.venv-lake` (Python 3.11, NVMe).

*(The "Python 3.11" here is a record of the interpreter these numbers were measured on and is
deliberately NOT edited — re-labelling a past measurement with a version it did not run under would
be a fabrication. It is flagged instead: `pyproject.toml` raised `requires-python` to `">=3.12"` on
2026-08-05 and `.venv-lake` is Python 3.11.15, so the env every figure on this page was taken in no
longer satisfies the project's own floor. Nothing here is invalidated — the measurements happened —
but a re-run after `.venv-lake` is recreated on 3.12 is a re-measurement on a different interpreter,
not a reproduction, and should be reported as such. See `docs/POLICIES.md` P5 for the open item.)*

**Disk is NOT a live-tick bottleneck — that is the headline null result.**

- Wake ticks: **0.00%** of samples in any I/O-classified leaf frame. The wake tick is compute
  (`episode_consolidation._settle` 14.1%, `vsa_reasoning` coincidence ~10.6%, prefrontal spiking).
- Sleep: 17.5% of all samples sit in I/O-adjacent leaves, but **15.6% is `zipfile` zlib *compression
  CPU*** inside `numpy.savez_compressed` (nightly skills npz), not disk — the actual
  `fileobj.write` leaf is **0.20%**. `binlog` (brain_lake.bin, fsync-per-append) leaves: 1.8%.
  `streaming_hub` leaves: 0.15%.
- The durable stores are small: `brain_lake.bin` 7.3 MB / 19,775 FACT records, full replay **0.05 s**;
  `~/.snn_data_lake/lake.spkl` **0 bytes** (the spike half of the durable home has never received an
  event); the only live `.spkl` with data is `brain_data/chat_wm.spkl` (6 MB). `data/telemetry.jsonl`
  is 160 MB and append-only-growing (it was 69 MB when last audited) — a disposition question, not a
  tick-latency one.

**The out-of-core contract holds at scale** (first actual-RSS demonstration): a 20M-event / 152.6 MB
store written at 2.06M events/s (`flush_every=65536`); a **fresh process** replays it in 6.3 s going
14.6 -> 21.0 MB RSS (coarse index 19,584 entries = N/1024; resident tail 11,520 events), then a
whole-lake single-channel query streams all 152.5 MB from disk returning 312,500 hits with peak RSS
**53.6 MB** (mostly the O(hits) result list itself) — resident memory stays bounded while the file is
~7x larger. An old-range query seeks 512 KB (0.3% of the file) in 6.3 ms.

**Append flush cost (neutered A/B):** `flush_every=1` 255k events/s vs `flush_every=4096` 2.03M/s vs
flush pinned to a no-op 2.01M/s — the per-append `flush()` IS the entire 8x gap (group-commit
recovers all of it), and nothing on the live path appends anywhere near those rates.

**One real disk defect found and fixed:** `MemoryLake`'s durable resume recovered its `last_t` cursor
with `query_window(0, 2^32)` over every channel; each channel's blocks span the whole interleaved
log, so the resume **re-read 62.8x the file** (383.5 MB for a 6.1 MB 64-channel store, 8.72 s).
`StreamingHub.max_time()` (an O(1) resident scalar rebuilt by replay) now answers it: same store,
same `last_t`/`n_events`, **0.25 s and 0 query bytes** (~35x faster resume). The packed-width silent
misread (100 events reopened as 1 without `packed=True`) is fixed by the self-describing packed-store
header — see STORAGE_FORMATS.md.
