"""Assemble a distributable release bundle (.zip) for the Spiking Neural Data Lake.

Two modes:

  python integrations/packaging/make_bundle.py --exe
      -> dist/spiking-neural-data-lake-<ver>-windows-x64.zip
      Zips the prebuilt standalone executable (dist/snn-ui.exe -- the daemon + cockpit) with
      QUICKSTART, LICENSE, and the build/run docs. Build the exe first:
      python integrations/packaging/build_exe.py

  python integrations/packaging/make_bundle.py --full
      -> dist/spiking-neural-data-lake-<ver>-full-src.zip
      Zips the entire runnable repo (source, research/, docs, tests, sample captures,
      install.bat / install.sh / integrations/packaging/install.py) minus VCS/tooling noise
      and data/model artifacts, for the "install Python 3.11+ and all extras" distribution.

Pure standard library. The <ver> string is read from pyproject.toml so a bundle can
never ship a version mismatched from the package.
"""
import argparse
import fnmatch
import os
import re
import sys
import tomllib
import zipfile
from pathlib import Path

PACKAGING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING_DIR.parent.parent          # integrations/packaging -> repo root

# --- full-source bundle: what NOT to ship (rebuilt/re-downloaded on the user's machine) ---
EXCLUDE_DIRS = {
    ".git", "dist", "build", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", "graphify-out", ".idea", ".vscode",
    "data", "datasets", ".terraform", "checkpoints", "wandb", "mlruns", "outputs", ".cache",
}
EXCLUDE_DIR_GLOBS = [".venv*", "venv", "*.egg-info"]
EXCLUDE_FILE_GLOBS = [
    "*.pyc", "*.pyo", "*.spk", "*.spc", "*.log", ".DS_Store", "*.zip",
    "*.pt", "*.pth", "*.onnx", "*.npy", "*.npz", "*.h5", "*.hdf5", "*.tar", "*.gz", "*.tgz",
    "*.bin", "*.pkl", "*.parquet", "*.arrow", "*.pdf", "*.mp4", "*.avi", "*.pgm", "*.aedat",
]
MAX_FILE_BYTES = 5 * 1024 * 1024   # a large file in source = a data/model artifact -> drop it


def read_version():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def sync_iss_version(version):
    """Keep the Inno Setup installer AppVersion in lockstep with pyproject before bundling."""
    iss_path = PACKAGING_DIR / "snn-ui.iss"
    if not iss_path.exists():
        return
    text = iss_path.read_text(encoding="utf-8")
    new_text = re.sub(r"(?m)^AppVersion=.*$", f"AppVersion={version}", text)
    if new_text != text:
        iss_path.write_text(new_text, encoding="utf-8")
        print(f"synced {iss_path.name} AppVersion -> {version}")


def _excluded_dir(name):
    return name in EXCLUDE_DIRS or any(fnmatch.fnmatch(name, g) for g in EXCLUDE_DIR_GLOBS)


def _excluded_file(name):
    return any(fnmatch.fnmatch(name, g) for g in EXCLUDE_FILE_GLOBS)


def build_exe_bundle(version):
    sync_iss_version(version)
    exe_path = REPO_ROOT / "dist" / "snn-ui.exe"
    if not exe_path.exists():
        print("dist/snn-ui.exe not found - build it first: "
              "python integrations/packaging/build_exe.py", file=sys.stderr)
        sys.exit(1)

    files = [
        (exe_path, "snn-ui.exe"),
        (PACKAGING_DIR / "QUICKSTART.txt", "QUICKSTART.txt"),
        (REPO_ROOT / "LICENSE", "LICENSE"),
        (REPO_ROOT / "docs" / "BUILD_EXE.md", "BUILD_EXE.md"),
        (REPO_ROOT / "docs" / "RUNNING_LOCAL.md", "RUNNING_LOCAL.md"),
    ]
    for src, _ in files:
        if not src.exists():
            print(f"Error: {src} not found", file=sys.stderr)
            sys.exit(1)

    out = REPO_ROOT / "dist" / f"spiking-neural-data-lake-{version}-windows-x64.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files:
            zf.write(src, arcname=arcname)

    size = out.stat().st_size / (1024 * 1024)
    print(f"Created: {out}")
    print(f"Size: {size:.2f} MB")
    print("Contents:")
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            print(f"  {name}")


def build_full_bundle(version):
    top = f"spiking-neural-data-lake-{version}"
    out = REPO_ROOT / "dist" / f"{top}-full-src.zip"
    out.parent.mkdir(exist_ok=True)

    count = skipped_big = skipped_bytes = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = sorted(d for d in dirnames if not _excluded_dir(d))
            for fn in sorted(filenames):
                if _excluded_file(fn):
                    continue
                fp = Path(dirpath) / fn
                try:
                    sz = fp.stat().st_size
                except OSError:
                    continue
                if sz > MAX_FILE_BYTES:
                    skipped_big += 1
                    skipped_bytes += sz
                    continue
                rel = fp.relative_to(REPO_ROOT)
                zf.write(fp, arcname=os.path.join(top, str(rel)))
                count += 1

    names = zipfile.ZipFile(out).namelist()
    required = [f"{top}/install.bat", f"{top}/install.sh",
                f"{top}/integrations/packaging/install.py",
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
    print("Verified present: install.bat, install.sh, integrations/packaging/install.py, "
          "pyproject.toml, run_selfchecks.py")
    print("Distribution: unzip -> run install.bat (Windows) or ./install.sh (POSIX).")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble a release bundle for the Spiking Neural Data Lake.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--exe", action="store_true",
                      help="bundle the prebuilt dist/snn-ui.exe (windows-x64.zip)")
    mode.add_argument("--full", action="store_true",
                      help="bundle the full runnable source tree (full-src.zip)")
    args = parser.parse_args()

    version = read_version()
    if args.exe:
        build_exe_bundle(version)
    else:
        build_full_bundle(version)
    sys.exit(0)


if __name__ == "__main__":
    main()
