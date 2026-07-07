"""Assemble a FULL-SOURCE release bundle (.zip): the entire runnable repo + the installer.

Unlike make_release_bundle.py (which zips only the standalone zero-dep GUI exe), this ships
everything needed to install and run the WHOLE repo -- source, research/, docs, tests, sample
captures, and the install.bat / install.sh / packaging/install.py bootstrap -- for the
"require Python 3.11+, install all extras" distribution.

    python packaging/make_full_bundle.py
    # -> dist/spiking-neural-data-lake-<ver>-full-src.zip

The user unzips it and runs install.bat (Windows) or ./install.sh (POSIX); the installer builds
a venv and pip-installs the package plus every optional extra it can on that machine.
"""
import fnmatch
import os
import sys
import tomllib
import zipfile
from pathlib import Path

# Directories never worth shipping (rebuilt on the user's machine, or VCS/tooling noise).
EXCLUDE_DIRS = {
    ".git", "dist", "build", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", "graphify-out", ".idea", ".vscode",
    # generated / downloaded data, ML artifacts, and IaC provider caches -- all regenerated
    # or re-downloaded on the user's machine; never part of a *source* bundle.
    "data", "datasets", ".terraform", "checkpoints", "wandb", "mlruns", "outputs", ".cache",
}
EXCLUDE_DIR_GLOBS = [".venv*", "venv", "*.egg-info"]      # every local virtualenv, build metadata
EXCLUDE_FILE_GLOBS = [
    "*.pyc", "*.pyo", "*.spk", "*.spc", "*.log", ".DS_Store", "*.zip",
    # model / dataset binaries -- large and regenerable; a source bundle ships code, not weights/data.
    "*.pt", "*.pth", "*.onnx", "*.npy", "*.npz", "*.h5", "*.hdf5", "*.tar", "*.gz", "*.tgz",
    "*.bin", "*.pkl", "*.parquet", "*.arrow", "*.pdf", "*.mp4", "*.avi", "*.pgm", "*.aedat",
]
MAX_FILE_BYTES = 5 * 1024 * 1024   # skip any file > 5 MB (source is small; big = data/model artifact)


def excluded_dir(name):
    return name in EXCLUDE_DIRS or any(fnmatch.fnmatch(name, g) for g in EXCLUDE_DIR_GLOBS)


def excluded_file(name):
    return any(fnmatch.fnmatch(name, g) for g in EXCLUDE_FILE_GLOBS)


def main():
    root = Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]

    top = f"spiking-neural-data-lake-{version}"
    out = root / "dist" / f"{top}-full-src.zip"
    out.parent.mkdir(exist_ok=True)

    count = 0
    skipped_big = 0
    skipped_bytes = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not excluded_dir(d))
            for fn in sorted(filenames):
                if excluded_file(fn):
                    continue
                fp = Path(dirpath) / fn
                try:
                    sz = fp.stat().st_size
                except OSError:
                    continue
                if sz > MAX_FILE_BYTES:   # a large file in source = a data/model artifact -> drop it
                    skipped_big += 1
                    skipped_bytes += sz
                    continue
                rel = fp.relative_to(root)
                zf.write(fp, arcname=os.path.join(top, str(rel)))
                count += 1

    # sanity: the installer entry points MUST be in the bundle
    names = zipfile.ZipFile(out).namelist()
    required = [f"{top}/install.bat", f"{top}/install.sh", f"{top}/packaging/install.py",
                f"{top}/pyproject.toml", f"{top}/run_selfchecks.py"]
    missing = [r for r in required if r not in names]
    if missing:
        print(f"ERROR: bundle is missing required entries: {missing}", file=sys.stderr)
        sys.exit(1)

    size = out.stat().st_size / (1024 * 1024)
    print(f"Created: {out}")
    print(f"Files: {count}   Size: {size:.1f} MB   Top-level: {top}/")
    if skipped_big:
        print(f"Skipped {skipped_big} file(s) over {MAX_FILE_BYTES // (1024 * 1024)} MB "
              f"({skipped_bytes / (1024 * 1024):.0f} MB total) -- datasets/models/artifacts, "
              f"regenerated or re-downloaded on install/run, not shipped as source.")
    print("Verified present: install.bat, install.sh, packaging/install.py, pyproject.toml, run_selfchecks.py")
    print("Distribution: unzip -> run install.bat (Windows) or ./install.sh (POSIX).")
    sys.exit(0)


if __name__ == "__main__":
    main()
