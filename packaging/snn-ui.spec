# PyInstaller spec — build a standalone Windows .exe of the zero-dep tkinter desktop app.
#
#   pip install ".[exe]"
#   python packaging/build_exe.py           # or: pyinstaller packaging/snn-ui.spec
#
# Produces dist/snn-ui.exe (onefile, windowed). tkinter is in the standard library, so the
# bundle carries Python + tkinter only — NO third-party runtime dependency (the optional
# torch/polars/pyserial/... addons are explicitly excluded below and never bundled).
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.dirname(SPECPATH)                 # packaging/ -> repo root  (SPECPATH: PyInstaller global)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)                          # so collect_submodules can import the package

hiddenimports = collect_submodules("snn_data_lake")   # bundle every snn_data_lake module

a = Analysis(
    [os.path.join(SPECPATH, "snn_ui_main.py")],
    pathex=[SRC, SPECPATH],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["numpy", "torch", "torchvision", "snntorch", "bindsnet",   # optional-dep addons:
              "polars", "pyarrow", "deltalake", "serial", "airllm",       # never part of the GUI
              "transformers", "tonic", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="snn-ui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                              # windowed GUI (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
