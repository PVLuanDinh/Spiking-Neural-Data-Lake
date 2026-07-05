# Build the desktop app as a Windows `.exe` (installable)

The product ships a zero-dependency **tkinter** desktop control panel
(`snn_data_lake.control_panel` — Arm Control, Live Recognition, Lake Browser). Because
tkinter is in the Python standard library, it packages into a **standalone Windows `.exe`
with no third-party runtime dependency** — the bundle carries Python + tkinter only.

PyInstaller is a **build-time** tool, so it lives in the optional `exe` extra; the core
stays `dependencies = []`.

## 1. Build the standalone `.exe`

```bash
pip install ".[exe]"              # installs PyInstaller (build-time only)
python packaging/build_exe.py     # -> dist/snn-ui.exe   (onefile, windowed)
```

`dist/snn-ui.exe` is portable: copy it anywhere and double-click to launch the full GUI.
No Python install required on the target machine.

Verify the build headlessly (no display needed — forwards to the same self-check CI runs):

```bash
dist/snn-ui.exe --smoke           # exits 0 on success
```

## 2. (Optional) Make an installer with Start-Menu shortcut

The `.exe` above is already double-click-runnable. To produce a proper `setup.exe`
(installs to Program Files, Start-Menu entry, uninstaller), compile the included
[Inno Setup](https://jrsoftware.org/isinfo.php) script:

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\snn-ui.iss
# -> packaging/installer/snn-ui-setup.exe
```

## Notes

- **Windowed, no console.** `snn-ui.exe` opens the GUI directly (the `control_panel`
  script default of the headless self-check is overridden by `packaging/snn_ui_main.py`).
- **Optional-dep addons are excluded** from the bundle (torch / polars / pyserial / airllm
  / ...): the GUI needs none of them, keeping the `.exe` small.
- **Safety envelope unchanged.** Every arm button in the GUI still routes through the
  safety-gated `ArmSim.apply()` — packaging changes nothing about the control path.
- Build artifacts (`build/`, `dist/`) are git-ignored; only the packaging sources are
  committed.
