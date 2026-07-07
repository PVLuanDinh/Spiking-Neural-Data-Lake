"""
Assemble a Windows release bundle (.zip) for the Spiking Neural Data Lake desktop app.

Usage:
  python packaging/build_exe.py            # first: produces dist/snn-ui.exe
  python packaging/make_release_bundle.py   # then: produces dist/spiking-neural-data-lake-<ver>-windows-x64.zip

The resulting .zip is ready to attach to a GitHub release as the end-user download.
"""

import re
import sys
import tomllib
import zipfile
from pathlib import Path


def _sync_iss_version(iss_path: Path, version: str) -> None:
    """Keep the Inno Setup installer version in lockstep with pyproject (audit P2: AppVersion had
    gone stale at 2.12 vs 2.52.0). Rewrite AppVersion so a release never ships a mismatched installer."""
    if not iss_path.exists():
        return
    text = iss_path.read_text(encoding="utf-8")
    new_text = re.sub(r"(?m)^AppVersion=.*$", f"AppVersion={version}", text)
    if new_text != text:
        iss_path.write_text(new_text, encoding="utf-8")
        print(f"synced {iss_path.name} AppVersion -> {version}")


def main():
    # Derive repo root from this script's location
    script_path = Path(__file__).resolve()
    packaging_dir = script_path.parent
    repo_root = packaging_dir.parent

    # Read version from pyproject.toml
    pyproject_path = repo_root / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        version = config["project"]["version"]
    except Exception as e:
        print(f"Error reading {pyproject_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Keep the installer version in lockstep with pyproject before bundling (audit P2).
    _sync_iss_version(packaging_dir / "snn-ui.iss", version)

    # Check that snn-ui.exe exists
    exe_path = repo_root / "dist" / "snn-ui.exe"
    if not exe_path.exists():
        print(
            "dist/snn-ui.exe not found - build it first: python packaging/build_exe.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Define files to include in the zip
    files_to_include = [
        (exe_path, "snn-ui.exe"),
        (packaging_dir / "QUICKSTART.txt", "QUICKSTART.txt"),
        (repo_root / "LICENSE", "LICENSE"),
        (repo_root / "docs" / "BUILD_EXE.md", "BUILD_EXE.md"),
        (repo_root / "docs" / "RUNNING_LOCAL.md", "RUNNING_LOCAL.md"),
    ]

    # Verify all source files exist
    for src, _ in files_to_include:
        if not src.exists():
            print(f"Error: {src} not found", file=sys.stderr)
            sys.exit(1)

    # Create the zip archive
    zip_filename = f"spiking-neural-data-lake-{version}-windows-x64.zip"
    zip_path = repo_root / "dist" / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_include:
            zf.write(src, arcname=arcname)

    # Report success
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created: {zip_path}")
    print(f"Size: {zip_size_mb:.2f} MB")
    print("Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            print(f"  {name}")

    sys.exit(0)


if __name__ == "__main__":
    main()
