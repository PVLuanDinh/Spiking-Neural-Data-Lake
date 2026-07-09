"""PyInstaller entry point: launch the Spiking Neural Data Lake desktop GUI.

`control_panel` run as a bare script defaults to the headless `--smoke` self-check (so CI
never blocks on a window). The packaged .exe instead calls `control_panel.main()`, which
opens the tkinter window. `snn-ui.exe --smoke` still forwards to the headless self-check,
which is exactly how the build is verified without a display.
"""
import os
import sys

# Frozen (onefile) builds collect the src/ package into the bundle, so `snn_data_lake`
# imports normally. For a plain `python packaging/snn_ui_main.py` run, add repo-root/src.
if not getattr(sys, "frozen", False):
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_root, "src"))

from snn_data_lake.substrate.control_panel import main

if __name__ == "__main__":
    main()
