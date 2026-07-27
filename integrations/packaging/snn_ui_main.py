"""PyInstaller entry point: run the always-on daemon and open the cockpit in a browser.

The frozen app has no window of its own. The ONE frontend is `substrate/cockpit.py` -- a stdlib
HTTP server that the LEARNING DAEMON stands up over the live brain (learn_daemon.py calls
`cockpit.serve(brain, port=NORD_TELEMETRY_PORT, ...)`). So the .exe IS the daemon: it runs
`learn_daemon.main()` on the MAIN thread -- the thread that owns the tick loop and is the only
thing allowed to touch the brain -- while a helper thread waits for the cockpit to answer and
then opens the default browser at it.

The cockpit is deliberately not launched on its own. `cockpit.serve()` never constructs a Brain
(Policy 4), so a standalone launcher would either stand up a SECOND brain against the same
durable lake or duplicate the daemon's drive loop. One brain, one loop, one frontend.

    snn-ui.exe            # run the brain, serve the cockpit, open the browser
    snn-ui.exe --smoke    # headless self-check (no display, no durable write); exits 0

The bundle carries no torch (a ~2 GB wheel that a desktop build has no business shipping), so a
frozen run pins NORD_TORCH=0 and the brain takes its deterministic stdlib tier instead of the
torch growth tier. A source checkout with torch installed is unaffected.
"""

import os
import sys
import threading
import time
import urllib.request
import webbrowser

# Frozen (onefile) builds collect the src/ package AND the root-level learn_daemon into the
# bundle, so both import normally. For a plain `python integrations/packaging/snn_ui_main.py`
# run, add the repo root (learn_daemon.py lives there) and repo-root/src (the package).
if not getattr(sys, "frozen", False):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, "src"))

# The daemon builds the whole brain BEFORE serve() binds the port, and a cold first boot with no
# durable home yet is slow. This is the wait for the page, not for a socket.
BROWSER_TIMEOUT = 180.0


def _cockpit_is_up(port):
    """True only if the page answering on `port` is OUR cockpit.

    A bare connect() is not enough: if the port was already taken the daemon prints "cockpit not
    started" and keeps learning headless, and opening a browser at a port we never bound would
    show whatever else is listening. Fetching the page and looking for its title is the only
    honest check that there is something of ours to show.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            return "SNDL cockpit" in r.read(4096).decode("utf-8", "replace")
    except OSError:
        return False


def _wait_and_open(port, timeout=BROWSER_TIMEOUT):
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _cockpit_is_up(port):
            webbrowser.open(url)
            return
        time.sleep(0.5)
    print(f"[snn-ui] cockpit did not answer within {timeout:.0f}s -- open {url} once it does.")


def main():
    if "--smoke" in sys.argv:
        # cockpit.main() serves the page, checks the telemetry contract against the live brain,
        # and round-trips a chat turn through the queue. No display, and NORD_PERSIST=0 so a
        # verification run can never write the durable lake.
        from snn_data_lake.substrate import cockpit

        cockpit.main()
        return

    if getattr(sys, "frozen", False):
        # no torch in the bundle -> take the stdlib tier deliberately, rather than crash on it
        os.environ.setdefault("NORD_TORCH", "0")

    port = int(os.environ.get("NORD_TELEMETRY_PORT", "8137"))
    opener = threading.Thread(target=_wait_and_open, args=(port,), daemon=True)
    opener.start()

    # Imported here, not at module scope: learn_daemon chdir's and edits sys.path at IMPORT time,
    # so it must not run before the paths above are in place.
    import learn_daemon

    learn_daemon.main()
    # main() returns immediately when another daemon already holds the single-instance lock. That
    # daemon's cockpit is already up, so give the opener a moment to find it before we exit.
    opener.join(timeout=10)


if __name__ == "__main__":
    main()
