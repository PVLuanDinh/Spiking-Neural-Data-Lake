"""Build `snn_data_lake._native` (the binding crate native/sndl-python) and place it in the package: the ONE
mechanism for a developer, a lane and CI (never `maturin develop`, which adds an editable `.pth` into the
shared venv pointing at whichever worktree ran it).

`maturin build --release` makes a real wheel; its filename TAG is asserted to be the free-threaded,
version-specific `cp3XY-cp3XYt-*` (Phase 0 item 8a: asking for abi3 on 3.14t does NOT fail, maturin silently
builds cp314t, and an abi3 wheel built elsewhere cannot load on a free-threaded interpreter, so the tag is the
gate, not a build error). The extension file is then extracted from that wheel into `src/snn_data_lake/`
(polars places `polars/_polars.abi3.so` inside its package the same way), because `.venv-lake` imports
`snn_data_lake` from `src/` (editable `.pth` + every lane's `PYTHONPATH=src`): a `pip install` of the wheel
would land `snn_data_lake/_native.pyd` in a site-packages directory that `src/snn_data_lake` shadows.
(maturin 1.15 lays the wheel out as `_native/_native<EXT_SUFFIX>`; only the file name is relied on.)

usage: python -m snn_data_lake.infra.native_build [--maturin PATH] [--python PATH] [--out DIR]
(lives in the governed package, not tools/: P4c, every build is governed; the p4c_ratchet tools zone is
closed to new files.)
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]  # src/snn_data_lake/infra/<this> -> repo root
PKG = ROOT / "src" / "snn_data_lake"
MANIFEST = ROOT / "native" / "sndl-python" / "Cargo.toml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # PATH first, then the interpreter's own scripts dir: the [dev] extra installs maturin into the venv,
    # whose Scripts/ (bin/) is not on PATH when the venv python is called by absolute path.
    venv_maturin = shutil.which("maturin", path=str(pathlib.Path(sys.executable).parent))
    ap.add_argument(
        "--maturin", default=shutil.which("maturin") or venv_maturin, help="maturin executable"
    )
    ap.add_argument("--python", default=sys.executable, help="interpreter to build for")
    ap.add_argument("--out", default=ROOT / "native" / "target" / "wheels", help="wheel directory")
    a = ap.parse_args(argv)
    if not a.maturin:
        sys.exit("maturin not found on PATH: pass --maturin")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sndl_native-*.whl"):
        old.unlink()  # exactly one wheel is judged below
    cmd = [a.maturin, "build", "--release", "-m", str(MANIFEST), "-i", a.python, "-o", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True)  # cwd=ROOT: rust-toolchain.toml pins the compiler
    (wheel,) = out.glob("sndl_native-*.whl")
    tag = wheel.stem.split("-", 2)[2]  # <python>-<abi>-<platform>
    v = f"{sys.version_info.major}{sys.version_info.minor}"
    if not tag.startswith(f"cp{v}-cp{v}t-"):
        sys.exit(f"refusing {wheel.name}: tag {tag!r} is not the free-threaded cp{v}-cp{v}t build")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")  # .cp314t-win_amd64.pyd / .cpython-314t-...so
    with zipfile.ZipFile(wheel) as z:
        (name,) = [n for n in z.namelist() if pathlib.PurePosixPath(n).name == "_native" + suffix]
        for stale in PKG.glob("_native.*"):
            stale.unlink()  # one extension in the package, never two interpreters' worth
        target = PKG / pathlib.PurePosixPath(name).name
        target.write_bytes(z.read(name))
    print(f"native: {wheel.name} ({tag}) -> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
