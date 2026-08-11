# Evidence: Finish the GeNN Port (specs/012)

Append-only. Every measured run verbatim, per data-model.md E4. Box: Windows 11, RTX 5070
(sm_120), CUDA 13.3 (nvcc V13.3.73), VS 2022 Community, `.venv-lake` = CPython 3.14
free-threading (cp314t).

## preflight

**2026-08-07, in progress. Five measured blockers so far, each with its fix:**

1. `pip install -e .[genn]` failed at the PROJECT's own editable metadata: hatchling refuses
   PEP 508 direct references — `ValueError: Dependency #1 of option 'genn' ... cannot be a
   direct reference unless field 'tool.hatch.metadata.allow-direct-references' is set to
   'true'`. FIX: that key added to pyproject.
2. `ml-genn` is not on PyPI either (`from versions: none`) — the extra's second entry was
   equally fiction. FIX: direct reference to the ml_genn_2_4_0 tag archive.
3. ml_genn's archive root has no setup.py/pyproject (monorepo). FIX: `#subdirectory=ml_genn`
   fragment.
4. pygenn's setup.py (line 283) runs `msbuild genn.sln /t:cuda_backend ...` — msbuild is not
   on PATH outside a VS developer prompt: `FileNotFoundError: [WinError 2]`. FIX: run pip
   under `vcvars64.bat`. With that, GeNN's core + single-threaded-CPU backends COMPILE clean
   (warnings only).
5. **The box's CUDA 13.3 install is INCOMPLETE** — nvcc + 4 libs only:
   `include\crt\` missing entirely (the flat `host_config.h` is a deprecation wrapper that
   includes `crt/host_config.h`, so nothing CUDA-side could ever compile), no `nvvm\`
   (no `cicc.exe` device compiler — nvcc dies "The system cannot find the path specified"),
   no `bin\crt\link.stub`, no `cudart64_13.dll`. Even the trivial
   `nvcc -arch=sm_120 -c cuda_test.cu` fails on the raw install. FIX (no admin, no writes to
   Program Files): assembled a COMPLETE local toolkit at `D:\AI-project\cuda-local` from the
   incomplete install + official NVIDIA PyPI wheels version-matched to the installed nvcc
   V13.3.73 — `nvidia-cuda-crt==13.3.73` (include\crt, 25 headers),
   `nvidia-nvvm==13.3.73` (nvvm\bin\cicc.exe + libdevice), `nvidia-cuda-nvcc==13.3.73`
   (bin\crt link stubs), `nvidia-cuda-runtime==13.3.29` (cudart64_13.dll). With
   `CUDA_PATH=D:\AI-project\cuda-local` + its bin on PATH + vcvars64:
   `nvcc -arch=sm_120 -c cuda_test.cu` -> **exit 0** (verified, .obj produced).

6. With crt fixed, the cuda_backend compile then failed on `curand_kernel.h` (cuRAND is its own
   component). FIX: merged `nvidia-curand==10.4.3.29` (the cu13-line wheel: curand_kernel.h +
   curand64_10.dll) and the `nvidia-cuda-cccl==13.3.3.4.1` include tree into
   `D:\AI-project\cuda-local`.

**RESULT: the extra STANDS UP (attempt 7, 2026-08-07).**

```
Successfully built spiking-neural-data-lake ml-genn pygenn
Successfully installed ... ml-genn-2.3.1 pygenn-5.4.0 ...
  (pygenn wheel: pygenn-5.4.0-cp314-cp314t-win_amd64.whl -- built locally against the
   FREE-THREADED ABI)
```

**Smoke (genn_smoke.py, then at specs/012-finish-genn-port/, since 2026-08-11 at tests/instruments/): PASS, exit 0.**

```
pygenn version: 5.4.0
GIL warnings at import: ["The global interpreter lock (GIL) has been enabled to load module
 'pygenn._genn', which has not declared that it can run safely without the GIL. ..."]
building (nvcc compile)...
backend: _Backend
out spikes: t=[2.0, 2.0, 4.0, 4.0] id=[1, 0, 1, 0]
SMOKE OK: compiled, stepped on device, 4 spikes recorded
```

- The R3 prediction is now a measurement: pygenn's extension re-enables the GIL at import
  (RuntimeWarning above). Contained exactly as designed: the package guard checks the BUILD
  property, and run_selfchecks isolates per-module subprocesses. Do NOT force `PYTHON_GIL=0`
  around an extension that declares itself unsafe.
- cudafe intermediates (`tmpxft_*...cudafe1.cpp`) + nvcc-compiled neuronUpdate/synapseUpdate/
  customUpdate objects confirm the CUDA backend, not the CPU fallback.
- **Runtime env contract measured**: every GeNN model build needs `CUDA_PATH=D:\AI-project\cuda-local`,
  `D:\AI-project\cuda-local\bin` on PATH, and the MSVC toolchain active (`vcvars64.bat`) —
  set PATH BEFORE calling vcvars in cmd (%PATH% expands at parse time; setting it after wipes
  the vcvars entries).

## cpu-baseline (2026-08-10)

The oracle on THIS box (`snn_mnist_stdp_fast`, NORD_M=300 NORD_TRAIN=6000 NORD_TEST=1000
SKIP_RATE=1, seed 0):

```
M=300  T=30  train=6000  test=1000  FAST_THRESH=2.0
FAST (deterministic latency, precomputed)...
  precompute=4.6s  acc=79.0%  train=26.5s  train SynOps=749,974,200
self-check OK: fast latency STDP learns (>=55% vs 10% chance)
```

Same-box 79.0% vs the historical 76.0% in the old docstring — torch-build drift; the
SAME-BOX figure is the parity anchor (research.md R7).

## equivalence (toy-scale exact-mirror proof, 2026-08-10)

`genn_equivalence.py` (then at specs/012-finish-genn-port/, since 2026-08-11 at tests/instruments/) — GPU port vs a float32 numpy replica of
`FastStdpNetwork.run()` (statement-for-statement, dims-parameterised), 4 random toy images,
M=8 N=16 T=10, sequential learning with per-image normalisation:

```
EQUIVALENCE OK: 4 images, counts exact, dW<=1e-4, theta<=1e-5, 2x bitwise
```

Winner-count vectors EXACT per image, weight matrices to 1e-4 (float32 GPU vs float64-free
numpy), theta to 1e-5, and two GPU runs bitwise identical (cross-run determinism).

**Three defects the harness caught on the way — each was invisible from the code and loud in
the measurement:**

1. **A silent network.** The custom weight-update model had the trace bump in
   `pre_spike_code` but no `addToPost(g)` anywhere: GeNN delivery is explicit, and a WU
   model without it delivers NOTHING (V flat 0, zero spikes, all 12 steps). Fix:
   `pre_spike_syn_code = "addToPost(g);"`, LTP moved to `post_spike_syn_code` (the per-
   synapse string with `g` access; the per-neuron `post_spike_code` cannot touch weights).
2. **The trace-depth lag, settled by measurement.** Per-synapse dW ratios named it: with
   `prevTrace1` the GPU trace was `d * x_pre(t)` uniformly (ratio 0.948 = one extra decay)
   plus a t+1 bump leak; with `prevTrace2` every ratio is 1.0000. The winner's post-learn
   executes at the start of kernel t+3 and reads `prevTrace2` = state after kernel t =
   x_pre(t) EXACTLY.
3. **The last-step winner's LTP never landed.** A t=T-1 winner spikes at kernel T+1 and its
   post-learn runs at the start of kernel T+2 — one kernel past the old T+2-step loop. Its
   SpikeCount and theta bump DID land (reset_code, kernel T+1), so the state was HALF-applied
   and normalisation reshaped the whole row. Fix: T+3 kernels per image, WTA gated to
   boundaries 2..T+1, kernel T+2 a bare flush. The tell in the data: only t=T-1 winners
   diverged.

Also measured and REPRESENTATIONALLY REFUSED: a duplicate (pixel, time) spike pair bumps the
oracle's trace once but delivers weight twice (fancy-index `+=`), and the SpikeSourceArray
cannot express that — it would smear the duplicate to the next kernel (measured 1+d trace).
`encode_latency` never emits duplicates (consecutive distinct burst times), so `pack_schedule`
now raises ValueError on one rather than silently mis-running a future encoder.

## parity (2026-08-10)

Reference config (NORD_M=300 NORD_TRAIN=6000 NORD_TEST=1000, seed 0), GENN_RUN=1:

```
M=300  T=30  train=6000  test=1000  thresh=2.0  lr0=False
encode=4.6s  determinism_encode=OK
schedule: 13000 images, 5387067 spikes
build+load (nvcc)=14.8s
train=7.5s
assign+eval=7.2s
acc=0.7970
```

**GPU 0.7970 vs same-box CPU oracle 0.7900 (cpu-baseline above): +0.7pp, inside the
pre-registered +/-3pp band.** Train wall-clock 7.5s vs CPU 26.5s = **3.5x** at M=300
(before the M-scaling payoff runs). The residual 0.7pp is float32-order effects on WTA
margins at full scale — the toy harness (#equivalence) shows the mechanism itself is exact.

## control (2026-08-10)

Same config, GENN_LR0=1 (lr pinned 0.0, learnFlag 0 — weights stay at their seeded random
values; the SAME compiled model, only dynamic params moved):

```
M=300  T=30  train=6000  test=1000  thresh=2.0  lr0=True
train=6.2s
acc=0.3270
```

**Untrained 0.3270 vs trained 0.7970: learning adds +47.0pp.** The control sits above 10%
chance BY DESIGN of the harness being controlled: the assign/eval half maps neurons to
classes by response profile, which classifies weakly over random projections. The honest
reading: 0.327 is what the ARCHITECTURE gives; 0.797 - 0.327 is what the LEARNING RULE
buys. (The oracle behaves the same way — this mirrors, not excuses.)

## determinism (2026-08-10)

Same seed, full re-run: `acc=0.7970` — identical to the digit. Plus in-run
`determinism_encode=OK` (encoder re-run equality) and the harness's two-bitwise-runs check.

## red-demo (2026-08-10)

Constructed reds, verbatim (in-process mutation, tree untouched; exit 1 both):

```
>>> m.GENN_POST_SPIKE = m.GENN_POST_SPIKE.replace('prevTrace2', 'preTrace')
AssertionError: LTP form lost from codegen

>>> m.GENN_PRE_SPIKE_SYN = ''
AssertionError: synaptic delivery lost — a silent network
```

Clean run immediately after: `self-check OK: codegen integrity (7 asserts, every red
reachable)`. The 8th assert (rule params vs oracle constants) was DELETED, not demonstrated:
`rule_params()` is BUILT FROM those constants, so the comparison round-trips and its red is
unreachable — P8 says delete, and the hazard is closed structurally (one source feeds
init_weight_update).

## payoff-300 (2026-08-11)

Full MNIST (NORD_TRAIN=60000 NORD_TEST=10000), M=300, seed 0:

```
GPU (GENN_RUN=1):             CPU oracle (same config, SKIP_RATE=1):
encode=44.4s                    precompute=44.2s
schedule: 76000 images,         acc=53.1%  train=286.7s
  31536720 spikes                 train SynOps=7,451,824,500
build+load (nvcc)=0.2s  <- binary cache hit
train=75.1s
assign+eval=16.8s
acc=0.5478
```

**Parity holds at scale: GPU 0.5478 vs CPU 0.531 = +1.7pp, inside the +/-3pp band.
Speed: train 75.1s vs 286.7s = 3.8x at M=300/60k.**

**The honest headline: BOTH sides get WORSE with 10x more data** (GPU 0.797 -> 0.548, CPU
0.790 -> 0.531). That is the ALGORITHM, mirrored faithfully: single-epoch STDP with
THETA_DECAY=1.0 accumulates theta bumps forever, so over 60k images high-response neurons
drive their own thresholds sky-high and response collapses. The CPU oracle's own >=55%
self-check assert FAILS at this config (exit 1) — its bar was tuned at 6k. The v0.19
docstring's "~95% GPU regime (with enough neurons + full 60k)" aspiration is REFUTED for
this rule as configured; reaching it is theta-hyperparameter work (NORD_TDECAY < 1,
multi-epoch), which belongs to the RULE's lane, not the port's. The port's claim — the GPU
IS the CPU rule, faster — is what these numbers show.

## payoff-1600 (2026-08-11)

Full MNIST, M=1600, seed 0 — both sides measured on this box:

```
GPU (GENN_RUN=1):             CPU oracle (same config, SKIP_RATE=1):
encode=44.6s                    precompute=73.8s
build+load (nvcc)=12.5s         acc=83.9%  train=487.3s
train=115.0s                      train SynOps=39,743,064,000
assign+eval=16.6s
acc=0.8458
```

**Parity: GPU 0.8458 vs CPU 0.839 = +0.7pp. Speed: train 115.0s vs 487.3s = 4.2x.**

And the result that redeems the scale story: **M=1600 FIXES the 60k theta collapse on both
sides** (M=300: 0.548/0.531; M=1600: 0.846/0.839) — with 5.3x more neurons the per-neuron
theta accumulation spreads and single-epoch STDP survives full MNIST. 0.846 is this rule
family's best full-MNIST figure in the tree. The v0.19 "enough neurons + full 60k" clause
was half right: neurons yes, and the ~95% aspiration still needs theta-hyperparameter work
(the RULE's lane).

## honest comparison (T025)

| config | CPU acc | GPU acc | delta | CPU train | GPU train | speedup |
|---|---|---|---|---|---|---|
| M=300, 6k/1k | 0.790 | 0.797 | +0.7pp | 26.5s | 7.5s | 3.5x |
| M=300, 60k/10k | 0.531 | 0.548 | +1.7pp | 286.7s | 75.1s | 3.8x |
| M=1600, 60k/10k | 0.839 | 0.846 | +0.7pp | 487.3s | 115.0s | 4.2x |

Every delta inside the pre-registered +/-3pp band; toy scale is EXACT (#equivalence), so
the full-scale residual is float32 summation order on WTA margins, not mechanism. Speedup
grows with M (3.5x -> 4.2x) — the GPU's margin widens exactly where the port exists to go.
What the GPU does NOT buy: the per-image host round-trip (startSpike/endSpike push, g
normalisation pull/push) bounds the speedup; a device-side normalisation custom update is
the known next lever if this lane ever needs more. Untrained control 0.327 (architecture)
vs 0.797 trained (rule) at the reference config; encode determinism OK on every run; same
seed reproduces acc to the digit.
