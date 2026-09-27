"""SNDL desktop app: the cockpit in a native window, rendered on the GPU.

WHAT THIS IS. `infra/cockpit.py` is the ONE frontend -- a stdlib HTTP page the always-on daemon
serves over the live brain. Until now the only ways to see it were a browser tab or the frozen
`snn-ui.exe`, which also just opens a browser (`integrations/packaging/snn_ui_main.py`). This gives it
a real desktop window.

WHY A SEPARATE PROCESS FOR THE DAEMON (the architectural constraint that shapes everything here).
pywebview's `webview.start()` must own the MAIN thread on Windows. `learn_daemon.main()` also wants
its process's main thread -- it owns the tick loop and is the only thing allowed to touch the brain.
Both cannot have it. So the daemon runs as its OWN process and this one owns the window:

    [desktop process]  main thread -> webview window  -> HTTP -> [daemon process] main thread -> brain

That is also why this does not duplicate `snn_ui_main.py` rather than replacing it (Policy 2): the
frozen exe runs the daemon IN-process and opens a browser, which is correct for a bundle that must be
one executable. This is the source-checkout desktop shell. They share the one thing worth sharing --
the cockpit URL and the "is it really our page" check.

POLICY 4. This process NEVER constructs a Brain. It spawns or reuses the daemon and renders its page.
`learn_daemon` holds an OS single-instance lock, so launching this while a daemon is already running
attaches to that daemon instead of standing up a second brain against the same durable lake.

THE GPU CLAIM, AND HOW TO CHECK IT RATHER THAN BELIEVE IT. On Windows pywebview renders through
WebView2 (Edge/Chromium), which composites on the GPU and exposes WebGL/WebGPU. `--gpu-info` asks the
running page which adapter it actually got, via WEBGL_debug_renderer_info, and prints the string the
driver returns. If it does not say your card, the window is on a software rasteriser and the claim is
false -- which is the whole reason this flag exists. Policy 5's honesty guard applies to rendering as
much as to compute: a GPU path ships with its measurement.

    python integrations/desktop/cockpit_desktop.py             # open the desktop window
    python integrations/desktop/cockpit_desktop.py --gpu-info  # print the renderer the window got
    python integrations/desktop/cockpit_desktop.py --smoke     # headless: no window, exits 0
"""

import os
import subprocess
import sys
import time
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `if _ROOT not in sys.path: sys.path.insert(0, _ROOT)` stood here. DELETED 2026-08-05, for the
# SAME reason `_prefer_discrete_gpu` below exists: a module-scope write to process-global state.
# It was also DEAD. MEASURED before deletion: importing this module added
# '<repo>' to sys.path for the rest of the process, and this module imports NOTHING from the repo
# ('snn_data_lake' not in sys.modules after import -- verified). `_ROOT` is still used, for the
# daemon's script path and its cwd, and neither needs the path entry.
# This is the third of the three side effects P4c's daemon promotion named (os.chdir /
# sys.path.insert / os.environ.setdefault); the environ one was moved into main() in the same pass
# that left this one at column 0.

# Snapshots taken BEFORE anything below can mutate them. The smoke asserts both are unchanged,
# which is what "importing this module has no side effects" means as a MEASUREMENT rather than as a
# comment -- and unlike the `_GPU_ARGS_APPLIED` sentinel these fire for ANY module-scope write, not
# only for one that remembers to set the sentinel. Reds for both were constructed and measured; see
# the assert block in main().
_SYS_PATH_AT_IMPORT = tuple(sys.path)
_WEBVIEW2_ENV_AT_IMPORT = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS")

PORT = int(os.environ.get("NORD_TELEMETRY_PORT", "8137"))
URL = f"http://127.0.0.1:{PORT}/"
BOOT_TIMEOUT = 180.0  # a cold first boot builds the whole brain before the port binds

# Chromium reads these from the environment at launch; pywebview passes them through to the WebView2
# host. Ask it for the discrete GPU and the modern graphics stack.
_WEBVIEW2_GPU_ARGS = "--enable-features=Vulkan,WebGPU --use-angle=d3d11 --force_high_performance_gpu"


def _prefer_discrete_gpu():
    """Set the WebView2 launch arguments. Called from main(), NOT at module scope.

    This was a bare module-scope `os.environ.setdefault(...)`, which is the same latent bug the P4c
    promotion of the daemon found and removed three times over (`os.chdir`, `sys.path.insert`,
    `os.environ.setdefault("NORD_GPU","1")` -- POLICIES.md P4c). Importing a module must not change
    the process environment for everything imported after it: `run_selfchecks.py` drives part of the
    suite through runpy in ONE process, so an import-time environment write here would apply to every
    module that follows. Under P4c this file is part of the brain and is a candidate for exactly that
    kind of registration, so the side effect had to go before it becomes reachable rather than after.
    Only the desktop process wants this, and only when it is about to open a window.
    """
    global _GPU_ARGS_APPLIED
    os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", _WEBVIEW2_GPU_ARGS)
    _GPU_ARGS_APPLIED = True


# False until main() opens a window. The smoke asserts it is still False, which is what "importing
# this module has no side effects" means as a MEASUREMENT rather than as a comment.
_GPU_ARGS_APPLIED = False


def cockpit_is_up(port=PORT):
    """True only if the page answering on `port` is OUR cockpit.

    A bare socket connect is not enough. If the port was already taken, the daemon prints "cockpit
    not started" and keeps learning headless -- so something answers that is not ours. Fetching the
    page and looking for its title is the only honest check. (Same reasoning, and the same marker,
    as snn_ui_main._cockpit_is_up.)
    """
    try:
        with urllib.request.urlopen(URL, timeout=2) as r:
            return "SNDL cockpit" in r.read(4096).decode("utf-8", "replace")
    except OSError:
        return False


def ensure_daemon(timeout=BOOT_TIMEOUT):
    """Return when the cockpit answers. Spawns the daemon only if one is not already serving.

    Never raises on a spawn failure -- an already-running daemon is the common case and the most
    desirable one, so a failed spawn is only fatal if nothing answers by the deadline.
    """
    if cockpit_is_up():
        return "attached to a daemon that was already running"
    proc = subprocess.Popen(
        [sys.executable, os.path.join(_ROOT, "learn_daemon.py")],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cockpit_is_up():
            return f"started the daemon (pid {proc.pid})"
        if proc.poll() is not None and not cockpit_is_up():
            raise RuntimeError(
                f"learn_daemon exited with code {proc.returncode} before the cockpit answered"
            )
        time.sleep(0.5)
    raise TimeoutError(f"cockpit did not answer at {URL} within {timeout:.0f}s")


# The renderer string the GPU driver reports for the live window. UNMASKED_RENDERER_WEBGL is the
# unspoofed name; without the debug extension Chromium returns a generic "WebGL" string, and saying
# so is more useful than printing the generic one as if it were the adapter.
_GPU_JS = """
(function () {
  try {
    var c = document.createElement('canvas');
    var gl = c.getContext('webgl2') || c.getContext('webgl');
    if (!gl) return 'NO WEBGL CONTEXT (software rasteriser or blocked)';
    var dbg = gl.getExtension('WEBGL_debug_renderer_info');
    if (!dbg) return 'WebGL present but WEBGL_debug_renderer_info unavailable: ' +
                     gl.getParameter(gl.RENDERER);
    return gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) +
           '  |  vendor: ' + gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) +
           '  |  webgpu: ' + (('gpu' in navigator) ? 'available' : 'absent');
  } catch (e) { return 'probe failed: ' + e; }
})();
"""


def main():
    argv = sys.argv[1:]

    if "--smoke" in argv:
        # Headless verification for CI and for `run_selfchecks`-style gating: prove the module
        # imports, the URL is well formed and the daemon-detection path runs -- WITHOUT opening a
        # window, spawning a brain, or writing the durable lake.
        assert URL.startswith("http://127.0.0.1:"), URL
        assert isinstance(cockpit_is_up(), bool)
        # The GPU arguments must NOT be applied by the smoke: it opens no window, and applying them
        # would be the module-scope environment write this file just removed, relocated into a
        # verification path. RED is reachable: call _prefer_discrete_gpu() before main() and this
        # fires. MEASURED 2026-08-05 by doing exactly that -- AssertionError, exit 1.
        assert not _GPU_ARGS_APPLIED, (
            "the WebView2 GPU arguments were applied without a window being opened -- importing or "
            "smoke-checking this module must not write the process environment"
        )
        # The sentinel above is NOT sufficient on its own, and saying so is the point of these two.
        # It only fires when someone calls `_prefer_discrete_gpu()`; a bare
        # `os.environ.setdefault(...)` or `sys.path.insert(...)` re-added at column 0 -- the exact
        # defect shape this file was edited to remove, and the one P4c's daemon promotion found
        # three times -- leaves the sentinel False and would pass. These compare against snapshots
        # taken at the top of the module, so they catch the write itself rather than one caller of
        # it. Neither can false-red on a pre-existing external value: the snapshot IS whatever the
        # environment already held.
        # REDS CONSTRUCTED AND MEASURED 2026-08-05, by re-adding each removed line below its
        # snapshot: `sys.path.insert(0, _ROOT)` -> the sys.path assert fires;
        # `os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", _WEBVIEW2_GPU_ARGS)` ->
        # the environ assert fires. Both were run.
        assert tuple(sys.path) == _SYS_PATH_AT_IMPORT, (
            "sys.path changed between importing this module and its smoke -- a module-scope "
            "sys.path write is a process-global side effect (P4c); run_selfchecks drives part of "
            "the suite through runpy in ONE process"
        )
        assert (
            os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS") == _WEBVIEW2_ENV_AT_IMPORT
        ), (
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS changed between importing this module and its "
            "smoke -- importing must not write the process environment"
        )
        import webview  # noqa: F401  -- import must work or the app cannot open

        print(f"cockpit_desktop smoke OK: url={URL}, pywebview importable, no window, no brain")
        print("  import-time process writes: none (sys.path and WEBVIEW2 env both unchanged since import)")
        return 0

    _prefer_discrete_gpu()
    import webview

    where = ensure_daemon()
    print(f"[desktop] {where}; opening the window at {URL}")

    window = webview.create_window(
        "SNDL cockpit", URL, width=1500, height=980, min_size=(900, 600)
    )

    if "--gpu-info" in argv:

        def _report(w):
            # Give WebView2 a moment to finish first paint before asking for the adapter.
            time.sleep(1.5)
            try:
                print("[desktop] renderer:", w.evaluate_js(_GPU_JS))
            except Exception as e:  # noqa: BLE001 -- reporting failure must not kill the window
                print(f"[desktop] gpu probe failed: {type(e).__name__}: {e}")

        webview.start(_report, window)
    else:
        webview.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
