#!/usr/bin/env python3
"""Full-repo installer for the Spiking Neural Data Lake.

Requires Python 3.11+. Builds a virtual environment and installs the package plus every
optional extra it can on THIS machine, in stages, reporting each one. The core package is
the ONLY hard requirement; every extra is best-effort, so a heavy or compiler-gated
dependency (genn) can never brick the base install -- you always end up with a working
core, plus a clear report of which optional features are available here.

    python integrations/packaging/install.py                 # venv at ./.venv, install all
    python integrations/packaging/install.py --venv .venv    # explicit venv location
    python integrations/packaging/install.py --skip gpu,genn # skip the extra GPU wheels + the
                                                             # compiler-gated port

This script has ZERO third-party dependencies (standard library only) -- it must run BEFORE
anything is installed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv

MIN_PY = (3, 11)

# (extra, human description, hard_required). extra=None -> the core package itself.
STAGES = [
    (None,     "core (pulls the default torch + numpy dependencies)",     True),
    ("dev",    "tests + lint (pytest, pytest-cov, ruff, pre-commit)",     False),
    ("arm",    "real serial robot arm (pyserial)",                        False),
    ("lake",   "lakehouse query engine (polars, pyarrow, deltalake)",     False),
    ("nmnist", "N-MNIST event-camera data (tonic)",                       False),
    ("llm",    "on-device NL command backend (airllm)",                   False),
    ("exe",    "build the standalone Windows exe (pyinstaller)",          False),
    ("gpu",    "GPU / torch research (torch, torchvision, numpy, scikit-learn)", False),
    ("genn",   "GeNN GPU port (pygenn -- needs a C++ compiler + CUDA)",   False),
]


def die(msg):
    print("\nERROR: " + msg, file=sys.stderr)
    raise SystemExit(1)


def repo_root():
    # integrations/packaging/install.py -> the repo/bundle root is three levels up.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def venv_python(vdir):
    if os.name == "nt":
        return os.path.join(vdir, "Scripts", "python.exe")
    return os.path.join(vdir, "bin", "python")


def run(cmd, cwd):
    """Run a command, STREAMING its output (so long installs like torch show progress).
    Returns True on exit code 0, else False."""
    try:
        return subprocess.run(cmd, cwd=cwd).returncode == 0
    except OSError as e:
        print(f"  (could not launch: {e})", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Install the Spiking Neural Data Lake and every optional extra it can.")
    ap.add_argument("--venv", default=None, help="virtualenv directory (default: <repo>/.venv)")
    ap.add_argument("--skip", default="", help="comma-separated extras to skip, e.g. gpu,genn")
    args = ap.parse_args()

    if sys.version_info < MIN_PY:
        die(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ is required; this is {sys.version.split()[0]}. "
            "Install a newer Python from https://www.python.org/downloads/ and re-run.")

    root = repo_root()
    if not os.path.exists(os.path.join(root, "pyproject.toml")):
        die(f"pyproject.toml not found under {root} -- run this from inside the repo/bundle.")

    vdir = os.path.abspath(args.venv or os.path.join(root, ".venv"))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print("=" * 72)
    print("Spiking Neural Data Lake -- full install (core + every available extra)")
    print("=" * 72)
    print(f"repo   : {root}")
    print(f"python : {sys.version.split()[0]}  ({sys.executable})")
    print(f"venv   : {vdir}")
    if skip:
        print(f"skip   : {', '.join(sorted(skip))}")
    print("-" * 72)

    vpy = venv_python(vdir)
    if not os.path.exists(vpy):
        print("creating virtual environment ...")
        try:
            venv.EnvBuilder(with_pip=True, clear=False).create(vdir)
        except Exception as e:  # noqa: BLE001 - report any venv failure clearly and stop
            die(f"could not create a virtual environment at {vdir}: {e}")
    if not os.path.exists(vpy):
        die(f"virtualenv python missing after creation: {vpy}")

    # a modern pip/setuptools/wheel resolves current wheels; best-effort.
    print("upgrading pip / setuptools / wheel ...")
    run([vpy, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], root)
    print("-" * 72)

    results = []
    for extra, desc, required in STAGES:
        name = extra or "core"
        if extra in skip:
            results.append((name, "SKIP"))
            print(f"[skip] {name:6}  {desc}")
            continue
        target = "." if extra is None else f".[{extra}]"
        print(f"\n>>> installing {name}: {desc}")
        ok = run([vpy, "-m", "pip", "install", "-e", target], root)
        results.append((name, "OK" if ok else "FAIL"))
        if ok:
            print(f"[ ok ] {name}")
        else:
            print(f"[FAIL] {name}: install failed (see the pip output just above).")
            if required:
                die("the CORE install failed -- cannot continue. Fix the error above and re-run.")
            print(f"       {name} is OPTIONAL -- continuing without it.")

    # ---- report ----
    print("\n" + "=" * 72)
    print("INSTALL SUMMARY")
    print("=" * 72)
    for name, status in results:
        print(f"  {status:4}  {name}")
    okset = [n for n, s in results if s == "OK"]
    failset = [n for n, s in results if s == "FAIL"]
    print("-" * 72)
    print(f"installed : {', '.join(okset) or '(none)'}")
    if failset:
        print(f"unavailable on this machine (optional): {', '.join(failset)}")
        if "genn" in failset:
            print("  - genn (pygenn) needs a C++ compiler + CUDA toolkit; absent on most")
            print("    machines. Expected.")
        if "gpu" in failset:
            print("  - gpu pulls the GPU/training wheels; a network/disk/CUDA hiccup can fail")
            print("    it. The core + other extras still work. Retry later with:")
            print("      python integrations/packaging/install.py --skip genn")

    # ---- how to run ----
    if os.name == "nt":
        act = os.path.join(vdir, "Scripts", "activate")
    else:
        act = "source " + os.path.join(vdir, "bin", "activate")
    print("\n" + "=" * 72)
    print("RUN IT  (activate the venv, then use any of these)")
    print("=" * 72)
    print(f"  {act}")
    print("  python learn_daemon.py    # the always-on brain; serves the cockpit (the ONE")
    print("                            # frontend) at http://127.0.0.1:8137/")
    print("  snn-demo                  # one-file whole-loop demo")
    print("  snn-loop                  # full closed loop: sensor -> arm -> reward -> learn")
    print("  snn-arm --help            # live/replay gesture -> arm (--serial COMx = hardware)")
    print("  python run_selfchecks.py  # run every core module's self-check")
    print("  pytest                    # the test suite (needs the 'dev' extra)")
    # Corrected 2026-08-04: this line read `python research/<module>.py`, which has not been a
    # runnable shape since the v3.78 reorg moved every top-level research module into
    # src/snn_data_lake/reasoning/. research/ now holds only SUBDIRECTORIES (arc_agi, ng_snn,
    # neuromorphic_torch), so there is no `research/<module>.py` to run, and the trainers the
    # 'gpu'/'train' extras are actually for live in training/.
    print("  # trainers ('train'/'gpu' extras):    python training/<module>.py")
    print("  # research subprojects:               python research/<subdir>/<module>.py")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
