# Build the desktop app as a Windows `.exe` (installable)

The product ships the **cockpit** (`snn_data_lake.infra.cockpit`) — the ONE frontend, a
single page served over the live brain by a `http.server` from the standard library. The `.exe`
is therefore not a GUI shell: it *is* the always-on learning daemon. It runs
`learn_daemon.main()`, which builds the one brain and serves the cockpit, then opens the page in
the default browser.

There is deliberately no standalone cockpit launcher. `cockpit.serve()` never constructs a Brain
(Policy 4), so anything that started the page on its own would either stand up a second brain
against the same durable lake or duplicate the daemon's drive loop.

PyInstaller is a **build-time** tool, so it lives in the optional `exe` extra.

## 1. Build the standalone `.exe`

```bash
pip install ".[exe]"                            # installs PyInstaller (build-time only)
python integrations/packaging/build_exe.py      # -> dist/snn-ui.exe   (onefile, console)
```

`dist/snn-ui.exe` is portable: copy it anywhere and double-click it. No Python install required
on the target machine. A console window shows the boot banner, the cockpit URL, and the
per-cycle learning line; the browser opens at `http://127.0.0.1:8137/` once the page answers
(override the port with `NORD_TELEMETRY_PORT`).

Verify the build headlessly (no display needed — forwards to the cockpit's own self-check):

```bash
dist/snn-ui.exe --smoke           # exits 0 on success
```

`--smoke` serves the page, checks it against the brain's telemetry contract, round-trips one
chat turn through the queue, and runs with `NORD_PERSIST=0` so a verification can never write
the durable lake.

## 2. (Optional) Make an installer with Start-Menu shortcut

The `.exe` above is already double-click-runnable. To produce a proper `setup.exe`
(installs to Program Files, Start-Menu entry, uninstaller), compile the included
[Inno Setup](https://jrsoftware.org/isinfo.php) script:

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" integrations\packaging\snn-ui.iss
# -> integrations/packaging/installer/snn-ui-setup.exe
```

## Notes

- **Console, not windowed.** The UI is the browser page; the console carries the daemon's log
  and the `--smoke` result, which a windowed build would throw away.
- **One brain at a time.** The daemon takes an OS-level single-instance lock on the durable
  home. A second launch exits immediately, and its browser tab shows the cockpit of the
  instance already running.
- **torch is excluded** from the bundle (a ~2 GB wheel), along with the other optional-dep
  addons (polars / pyserial / airllm / ...). The frozen brain therefore lands on its deterministic
  stdlib tier the same way a broken install does — the torch import fails and the brain catches it.
  It used to pin `NORD_TORCH=0` to force that; the flag was removed on 2026-07-29 and pinning it
  became a no-op, so the pin went too. `numpy` is a declared core dependency and **is** bundled.
- **No data files.** The cockpit's page is a module constant and its anatomical layout is
  computed at request time (`infra/brain_geometry.py`), so the spec ships no `datas`.
- **Safety envelope unchanged.** The arm is the internal simulator and every command still
  routes through the safety-gated `So101Arm.apply()` — packaging changes nothing about the
  control path, and the cockpit binds loopback only.
- Build artifacts (`build/`, `dist/`) are git-ignored; only the packaging sources are
  committed.
