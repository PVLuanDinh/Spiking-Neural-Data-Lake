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

**Smoke (specs/012-finish-genn-port/genn_smoke.py): PASS, exit 0.**

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

## cpu-baseline

PENDING

## parity

PENDING

## control

PENDING

## determinism

PENDING

## payoff-300

PENDING

## payoff-1600

PENDING

## red-demo

PENDING
