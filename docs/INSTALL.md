# Install & Run — two ways to get the Spiking Neural Data Lake

There are two distributions, for two different needs.

| | **Standalone app** | **Full-source install** (this doc) |
|---|---|---|
| File | `spiking-neural-data-lake-<ver>-windows-x64.zip` | `spiking-neural-data-lake-<ver>-full-src.zip` |
| Needs Python? | **No** | **Yes — CPython 3.14 free-threaded (`3.14t`)** |
| What runs | The always-on brain daemon + its cockpit (the ONE frontend) in your browser | **Everything in the repo** — the core, all optional-dependency features, the demos, the tests, and the `research/` tree |
| Optional deps (pyserial, polars, torch, …) | Not included | **Installed** (best-effort per machine) |
| Size | ~11 MB | small download; the installer then pulls the deps |

If you just want to click and use the app, grab the **Standalone app** zip — unzip, double-click `snn-ui.exe`, and the cockpit opens in your browser. This document is for the **full-source install**, which sets up everything the repo can do.

---

## Full-source install

### Requirements
- **CPython 3.14, free-threaded build (`3.14t`)** on your PATH: `pyproject.toml` requires `>=3.14`, and the package
  refuses a GIL build at import. The python.org installer offers the free-threaded binaries as an option; `uv python
  install 3.14t` also works. *(Corrected 2026-10-28, cloud audit: this line said "Python 3.11, 3.12, or 3.13", and
  the table above said "3.11+".)*
- **A Rust toolchain** (`rustup` from <https://rustup.rs>; `rust-toolchain.toml` at the repo root pins the
  compiler version, which rustup installs on first use). `pip install .` (and `-e .`, and a wheel build)
  compiles the `snn_data_lake._native` extension from `native/sndl-python` through maturin **at install
  time** -- the prefrontal kernel is a hard runtime dependency, so an install without the toolchain fails
  at the build step rather than later at import. To rebuild it in place after a Rust change:
  `python -m snn_data_lake.infra.native_build`.
- Network access (the installer downloads the optional dependencies).
- Disk: a few hundred MB for the base extras; **~2 GB more** if the `gpu` (torch) stack installs.

### Steps

1. **Unzip** `spiking-neural-data-lake-<ver>-full-src.zip`.
2. **Run the installer** from the unzipped folder:
   - **Windows:** double-click **`install.bat`** (or run it in a terminal).
   - **Linux / macOS:** `bash install.sh`
3. That's it. The installer:
   - creates a virtual environment at `./.venv`,
   - installs the package **editable** (`pip install -e .`) plus every optional extra it can, **in stages**, and
   - prints a summary of what installed and exactly how to run each part.

### What gets installed

Installed in order; **only the core is required** — every extra is best-effort, so a heavy or
compiler-gated dependency can never break the base install:

| Stage | What | Notes |
|---|---|---|
| `core` | the base package | **always installs** (pulls torch + numpy) |
| `dev` | pytest, pytest-cov, ruff, pre-commit | run the tests + lint |
| `arm` | pyserial | drive a **real** serial robot arm |
| `nmnist` | tonic | real N-MNIST event-camera data |
| `llm` | airllm | on-device natural-language command backend |
| `exe` | pyinstaller | rebuild the standalone Windows `.exe` yourself |
| `gpu` | torchvision, scikit-learn (on core Torch + Brian2) | the `research/` GPU experiments (~2 GB; CPU-only torch works anywhere, CUDA is used if present) |
| `genn` | pygenn | the GeNN GPU port — **needs a C++ compiler + CUDA**; expected to be absent on most machines |

### Honest ceilings ("no matter what" has two real limits)

- **`genn`** requires a C++ compiler and the CUDA toolkit. On a machine without them, this stage
  fails — the installer reports it and moves on; everything else still works.
- **`gpu`** pulls ~2 GB of torch. CPU-only torch installs on any machine; GPU acceleration only
  kicks in if you have a CUDA-capable NVIDIA GPU + drivers. A slow network or low disk can fail
  this stage; the core and the other extras are unaffected.

To skip the heavy stages: `install.bat --skip gpu,genn` (Windows) or `bash install.sh --skip gpu,genn`.

### Run it

Activate the venv, then use any entry point:

```
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python learn_daemon.py    # the always-on brain; serves the cockpit (the ONE frontend)
                          # at http://127.0.0.1:8137/ -- live brain, arm, chat
snn-demo                  # one-file whole-loop demo
snn-loop                  # full closed loop: sensor -> arm -> reward -> learn
snn-arm --help            # live/replay gesture -> arm (add --serial COMx for hardware)
python run_selfchecks.py  # run every core module's self-check
pytest                    # the test suite (needs the 'dev' extra)
python research/<module>.py   # a research experiment (needs the 'gpu' extra)
```

### Re-running / troubleshooting
- The installer is **idempotent** — re-run it any time; it reuses the existing `.venv` and re-installs.
- If `gpu` failed on a flaky download, just re-run with `--skip genn` to retry torch without the compiler-gated stage.
- Driving a **real** robot arm additionally requires **calibration** — see [HARDWARE_SCOPE.md](HARDWARE_SCOPE.md) and [HARDWARE_SETUP.md](HARDWARE_SETUP.md). The serial arm refuses to move with placeholder limits (no bypass).
