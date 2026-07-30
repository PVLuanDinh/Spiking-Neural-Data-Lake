# PyInstaller spec -- build a standalone Windows .exe of the brain daemon + its cockpit.
#
#   pip install ".[exe]"
#   python integrations/packaging/build_exe.py     # or: pyinstaller integrations/packaging/snn-ui.spec
#
# Produces dist/snn-ui.exe (onefile, console). The .exe runs the always-on learning daemon and
# opens the cockpit (substrate/cockpit.py -- a stdlib http.server page) in the default browser.
# The bundle carries Python + numpy; torch and the other optional-dep addons are excluded below
# and never shipped; the brain reaches its stdlib tier by catching the failed torch import.
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.dirname(os.path.dirname(SPECPATH))  # integrations/packaging -> repo root  (SPECPATH: PyInstaller global)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)                          # so collect_submodules can import the package

hiddenimports = collect_submodules("snn_data_lake") + ["learn_daemon"]   # bundle every
# snn_data_lake module + learn_daemon (the root-level script the entry point runs; it is not part
# of the package, so PyInstaller only finds it via ROOT on pathex below)

a = Analysis(
    [os.path.join(SPECPATH, "snn_ui_main.py")],
    pathex=[SRC, SPECPATH, ROOT],                # ROOT so PyInstaller finds learn_daemon.py
    binaries=[],
    datas=[],                                    # the cockpit page is a module constant and its
                                                 # geometry is computed -- no data files to ship
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "snntorch", "bindsnet",             # optional-dep addons:
              "polars", "pyarrow", "deltalake", "serial", "airllm",       # never part of the app
              "transformers", "tonic", "pytest",
              "snn_data_lake.hippocampus.ng_snn"],                        # torch-gated core module
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
    console=True,               # the daemon logs its boot banner, the cockpit URL, and per-cycle
                                # lines to stdout, and --smoke prints its result there. Windowed
                                # would throw all of that away -- the UI is the browser page.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
