"""
Web server for the Spiking Neural Data Lake chat -- pure Python standard library.

Wires a Claude-style chat UI onto the REAL spiking pipeline in chat_app/spiking_chat.py
(text -> spikes -> data lake -> spike-pattern intent -> grounded decode). This is NOT a
large language model: replies are decoded from the brain's actual state and shown honestly.

Run (from the repo root):
    PYTHONUTF8=1 PYTHONPATH=src python chat_app/server.py
    -> Serving on http://127.0.0.1:8765

Headless self-check (CI-safe, ephemeral port, no litter):
    PYTHONUTF8=1 PYTHONPATH=src python chat_app/server.py --smoke

Routes:
    GET  /              -> the chat page (static/index.html)
    GET  /static/<f>    -> static assets
    POST /api/chat      -> {"message": "..."} -> the pipeline dict as JSON
    GET  /api/state     -> current goal / lake_events / wm_events (status line)

Stdlib only (http.server, json, threading, urllib for the self-check). No third-party deps.
"""
import argparse
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---- make both snn_data_lake (src/) and spiking_chat (chat_app/) importable ----
# Works when launched as `PYTHONPATH=src python chat_app/server.py` from the repo root:
# we still add src/, the repo root, and this dir so the imports resolve either way.
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../chat_app
_ROOT = os.path.dirname(_HERE)                               # repo root
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:                                                        # launched from the repo root
    from chat_app.spiking_chat import SpikingChat
except ModuleNotFoundError:                                 # launched from inside chat_app/
    from spiking_chat import SpikingChat
from snn_data_lake.brainstem.autonomous_loop import ACTIONS  # noqa: E402  (after sys.path setup)

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = os.path.join(_HERE, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")
LAKE_PATH = os.path.join(_HERE, "data", "conversation.spkl")   # persistent conversation lake

MAX_BODY_BYTES = 65536       # reject oversized request bodies before reading them
MAX_MESSAGE_CHARS = 4000     # reject oversized chat messages

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def _content_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in _CONTENT_TYPES:
        return _CONTENT_TYPES[ext]
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def make_handler(chat, lock, neuro=False, full=False):
    """Build a request handler bound to one engine guarded by `lock`. `neuro`=True -> a
    NeurogenesisSelfImprove (state()/tick() API); `full`=True -> a FullBrainDaemon
    (state()/respond()/tick() API, the integrated brain); else a SpikingChat (brain API)."""

    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "SNDLChat/1.0"

        # -- quiet per-request logging (keep stdout clean; prints stay ASCII) --
        def log_message(self, *args):
            pass

        # -- small response helpers --
        def _send_bytes(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _send_json(self, obj, status=200):
            self._send_bytes(json.dumps(obj).encode("utf-8"),
                             "application/json; charset=utf-8", status)

        def _send_error_json(self, status, message):
            self._send_json({"error": message}, status)

        # -- routing --
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                return self._serve_index()
            if path == "/favicon.ico":
                return self._send_bytes(b"", "image/x-icon", 204)
            if path == "/api/state":
                return self._serve_state()
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            return self._send_error_json(404, "not found")

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            path = urlsplit(self.path).path
            if path == "/api/chat":
                return self._handle_chat()
            return self._send_error_json(404, "not found")

        # -- handlers --
        def _serve_index(self):
            try:
                with open(INDEX_HTML, "rb") as f:
                    data = f.read()
            except OSError:
                return self._send_bytes(b"index.html not found",
                                        "text/plain; charset=utf-8", 500)
            self._send_bytes(data, "text/html; charset=utf-8")

        def _serve_static(self, rel):
            # normalize + confine to STATIC_DIR (no path traversal)
            rel = rel.split("?", 1)[0]
            abspath = os.path.normpath(os.path.join(STATIC_DIR, rel))
            static_root = os.path.normpath(STATIC_DIR)
            if not (abspath == static_root or abspath.startswith(static_root + os.sep)):
                return self._send_error_json(404, "not found")
            if not os.path.isfile(abspath):
                return self._send_error_json(404, "not found")
            try:
                with open(abspath, "rb") as f:
                    data = f.read()
            except OSError:
                return self._send_error_json(404, "not found")
            self._send_bytes(data, _content_type(abspath))

        def _serve_state(self):
            try:
                with lock:
                    if neuro or full:                  # self-improve engines expose state()
                        state = dict(chat.state())     # goal/ticks/grown_units/predict/circadian
                    else:
                        goal = chat.brain.goal
                        action_goal = ACTIONS[goal] if goal is not None else None
                        wm = chat.brain.arm.lake_wm
                        state = {
                            "goal": chat.meta_goal or action_goal,   # meta-goal when self-improving
                            "lake_events": chat.lake.n_events(),
                            "wm_events": wm.n_events() if wm is not None else 0,
                        }
                        if chat.self_improve:                    # live self-improvement telemetry
                            state["action_goal"] = action_goal
                            state["ticks"] = chat._ticks
                            state["learning_progress"] = round(chat._lp, 6)
                self._send_json(state)
            except Exception:                       # never leak a traceback to the client
                self._send_error_json(500, "internal error")

        def _handle_chat(self):
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                return self._send_error_json(400, "invalid Content-Length")
            if length <= 0:
                return self._send_error_json(400, "empty request body")
            if length > MAX_BODY_BYTES:
                return self._send_error_json(413, "request body too large")

            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return self._send_error_json(400, "invalid JSON body")
            if not isinstance(data, dict):
                return self._send_error_json(400, "expected a JSON object")

            message = data.get("message")
            if not isinstance(message, str):
                return self._send_error_json(400, "'message' must be a string")
            if message.strip() == "":
                return self._send_error_json(400, "'message' must not be empty")
            if len(message) > MAX_MESSAGE_CHARS:
                return self._send_error_json(413, "'message' is too long")

            try:
                with lock:                          # serialize the (stateful) engine
                    if neuro:                       # neurogenesis daemon: report self-improve state
                        st = chat.state()
                        acc = st.get("predict_accuracy", 0.0)
                        result = {
                            "reply": (f"I am self-improving via NEUROGENESIS: I have grown "
                                      f"{st.get('grown_units', 0)} NG-SNN neurons to predict my "
                                      f"own next perception, currently {acc * 100:.0f}% accurate "
                                      f"over {st.get('ticks', 0)} ticks (chance is "
                                      f"{100 // st.get('n_modes', 5)}%). Goal: {st.get('goal')}. "
                                      f"This is a perception-only self-model -- the reflex "
                                      f"E-STOP and gated arm are untouched."),
                            **st,
                        }
                    else:
                        result = chat.respond(message)
                self._send_json(result)
            except Exception:                       # never leak a traceback to the client
                self._send_error_json(500, "internal error running the spiking pipeline")

    return ChatHandler


def build_server(port, lake_path, self_improve=False, neurogenesis=False, full_brain=False):
    """Create (httpd, engine, lock). The engine is a SpikingChat, or -- in NEUROGENESIS mode -- a
    NeurogenesisSelfImprove that grows NG-SNN neurons to predict its own next perception, or -- in
    FULL-BRAIN mode -- a FullBrainDaemon: the integrated LiveBrain driven on a hypothalamic wake/
    sleep cycle that sleep-gates neurogenesis + replay consolidation (torch tier,
    goal=self-improvement). The torch engines are imported LAZILY so plain/smoke/self-improve modes
    stay pure-stdlib (no torch)."""
    os.makedirs(os.path.dirname(lake_path), exist_ok=True)
    if full_brain:
        # the unified brain: owns the self-model + introspection + the SpikeGPT language center
        from snn_data_lake.cortex.brain import Brain
        chat = Brain(seed=0, lake_path=lake_path)
    elif neurogenesis:
        from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove
        chat = NeurogenesisSelfImprove(seed=0)
    else:
        chat = SpikingChat(lake_path=lake_path, self_improve=self_improve)
    lock = threading.Lock()
    httpd = ThreadingHTTPServer((HOST, port), make_handler(chat, lock, neurogenesis, full_brain))
    return httpd, chat, lock


def _daemon_loop(chat, lock, stop_event, engine_tick=False, interval=0.02):
    """The SELF-IMPROVING DAEMON: a background thread that ticks the engine continuously, so it
    improves from its own experience even when no one is chatting. `engine_tick`=True -> the engine
    exposes its own tick() (NEUROGENESIS grows NG-SNN neurons; FULL-BRAIN runs the integrated
    wake/sleep brain); else it ticks the curiosity-driven SpikingChat via tick_self_improve().
    Guarded by `lock` (the SAME lock the handlers use) so daemon + requests never race. Safe:
    perception/value only, no motor route; a tick must never crash the server."""
    tick = chat.tick if engine_tick else chat.tick_self_improve
    while not stop_event.is_set():
        try:
            with lock:
                tick()
        except Exception:
            pass
        stop_event.wait(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spiking Neural Data Lake chat server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to bind (default {DEFAULT_PORT})")
    parser.add_argument("--smoke", action="store_true",
                        help="run the headless self-check and exit")
    parser.add_argument("--self-improve", action="store_true",
                        help="run as a SELF-IMPROVING DAEMON: a background thread ticks the "
                             "curiosity-driven brain continuously so it learns (gated/safe)")
    parser.add_argument("--neurogenesis", action="store_true",
                        help="NEUROGENESIS self-improving daemon: grow NG-SNN neurons to "
                             "predict the brain's own next perception (torch tier; self-improve)")
    parser.add_argument("--full-brain", action="store_true",
                        help="FULL-BRAIN daemon (DEFAULT): drive the integrated LiveBrain on a "
                             "hypothalamic wake/sleep cycle that sleep-gates neurogenesis (torch)")
    parser.add_argument("--static", action="store_true",
                        help="opt OUT of the daemon: serve the plain no-daemon chat. Without this "
                             "and no other mode flag, --full-brain runs.")
    args = parser.parse_args(argv)

    if args.smoke:
        return smoke()

    # DEFAULT MODE = full-brain: with no explicit mode flag (and not --static), run the full brain.
    if not (args.self_improve or args.neurogenesis or args.full_brain or args.static):
        args.full_brain = True

    engine_tick = args.neurogenesis or args.full_brain   # engines with their own tick()
    daemon_on = args.self_improve or engine_tick
    httpd, chat, lock = build_server(args.port, LAKE_PATH, self_improve=args.self_improve,
                                     neurogenesis=args.neurogenesis, full_brain=args.full_brain)
    port = httpd.server_address[1]
    stop_event = threading.Event()
    if daemon_on:                                 # start the background self-improvement daemon
        threading.Thread(target=_daemon_loop,
                         args=(chat, lock, stop_event, engine_tick), daemon=True).start()
    print(f"Serving on http://{HOST}:{port}")
    print(f"  chat page  -> http://{HOST}:{port}/")
    print(f"  lake file  -> {LAKE_PATH}")
    if args.full_brain:
        print("  mode       -> FULL-BRAIN daemon (integrated LiveBrain + wake/sleep cycle; "
              "sleep-gated neurogenesis; goal=self-improvement, safe)")
    elif args.neurogenesis:
        print("  mode       -> NEUROGENESIS daemon (NG-SNN grows to predict next perception; "
              "goal=self-improvement, safe)")
    elif args.self_improve:
        print("  mode       -> SELF-IMPROVING DAEMON (curiosity-driven learning; gated/safe)")
    else:
        print("  mode       -> static chat (no daemon; --full-brain is the default otherwise)")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        stop_event.set()
        httpd.shutdown()
        httpd.server_close()
        if hasattr(chat, "close"):                # SpikingChat has close(); neuro engine has none
            chat.close()
    return 0


def smoke():
    """Headless CI-safe self-check: bind an ephemeral port, drive one real turn, assert the
    pipeline JSON + the honest consciousness answer + that the page is served, then clean up."""
    import urllib.error
    import urllib.request

    tmpdir = tempfile.mkdtemp(prefix="sndl_chat_smoke_")
    lake_path = os.path.join(tmpdir, "conversation.spkl")
    httpd, chat, _lock = build_server(0, lake_path)   # port 0 -> ephemeral
    port = httpd.server_address[1]
    assert port != DEFAULT_PORT, "smoke must bind an ephemeral port, not the fixed one"
    base = f"http://{HOST}:{port}"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def post_chat(message):
        req = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps({"message": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        # (1) one real turn -> a grounded reply + visible pipeline stats
        r1 = post_chat("hello")
        assert isinstance(r1.get("reply"), str) and r1["reply"].strip(), "reply must be non-empty"
        assert isinstance(r1.get("input_spikes"), int), "input_spikes must be an integer"
        assert isinstance(r1.get("lake_events"), int), "lake_events must be an integer"
        assert isinstance(r1.get("intent"), str) and r1["intent"], "intent must be present"

        # (2) the consciousness question is answered honestly (no sentience claim)
        r2 = post_chat("are you conscious?")
        assert "not a conscious agent" in r2["reply"], "must answer consciousness honestly"

        # (3) the page is actually served, with the header + composer
        with urllib.request.urlopen(base + "/", timeout=30) as resp:
            html = resp.read().decode("utf-8")
        assert "Spiking Neural Data Lake" in html, "page must contain the header title"
        assert 'id="composer"' in html, "page must contain the composer"

        # (4) input validation returns a JSON error, never a traceback
        try:
            post_chat("")
            raise AssertionError("empty message must be rejected")
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            assert e.code == 400 and "error" in body, "empty message -> JSON 400 error"
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat.close()
        shutil.rmtree(tmpdir, ignore_errors=True)     # leave no litter

    print(f"self-check OK: served / (header + composer), POST /api/chat 'hello' -> "
          f"{r1['input_spikes']} spikes, lake {r1['lake_events']} events, intent {r1['intent']}; "
          f"'are you conscious?' -> honest ('not a conscious agent'); empty message -> 400 JSON; "
          f"ephemeral port {port}, no litter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
