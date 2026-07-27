"""Build dist/snn-ui.exe from integrations/packaging/snn-ui.spec.

The .exe runs the always-on learning daemon and opens its cockpit in a browser. Needs the
build-time `exe` extra (PyInstaller is NOT a runtime dependency):

    pip install ".[exe]"
    python integrations/packaging/build_exe.py    # clean build -> dist/snn-ui.exe

Verify the result headlessly (no display needed):

    dist/snn-ui.exe --smoke
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # integrations/packaging -> repo root
SPEC = os.path.join(HERE, "snn-ui.spec")


def main():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        sys.exit('PyInstaller not installed. Run:  pip install ".[exe]"')
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", SPEC]
    print("building:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)         # run from repo root -> dist/ + build/ land there
    if rc == 0:
        exe = os.path.join(ROOT, "dist", "snn-ui.exe")
        print(f"\nbuilt: {exe}\nverify headlessly:  {exe} --smoke")
    sys.exit(rc)


if __name__ == "__main__":
    main()
