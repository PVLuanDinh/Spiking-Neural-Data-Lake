r"""
BRAIN WEB -- the Spike Chat interface (the Claude-style UI) wired LIVE to the real unified Brain.

research/brain_web.html is the chat UI. Its earlier standalone version faked the brain in JavaScript;
here the JS is gone and every turn is a real call into research/brain_loop.Brain -- the SAME brain that
learns in the background and remembers on your SSD. So the page you type into is the actual spiking
brain: retrieval + contradiction reasoning + the re-entrant train of thought, all gated by ignition in
the spiking global workspace, over the raw-binary knowledge on the SSD.

  GET  /            -> the UI
  GET  /api/state   -> {n_known, facts, concepts, learning, learned, gpu}  (polled: "knows N" grows live
                       as the ALWAYS-ON background learner learns while you talk)
  POST /api/chat    -> {text} -> {reply, chips, intent, src, known}
                       "learn: q | a" teaches (durable, raw binary); "think: X" runs a train of thought;
                       anything else is a normal turn (reason -> recall -> honest gap).

Uses stdlib http.server only (no flask/fastapi). Binds 127.0.0.1 (LOCAL ONLY -- this exposes a
brain that can be taught, so it is deliberately not served on 0.0.0.0).

  PYTHONPATH=src .venv-lake/Scripts/python research/brain_web.py --serve            # CPU brain + UI
  NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/brain_web.py --serve --develop --gpu
  (no flags -> the deterministic offline self-check)
Deps: numpy + stdlib (torch only for the opt-in --develop perception).
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from snn_data_lake.reasoning.brain_loop import Brain
from snn_data_lake.reasoning.spike_chat_lake import _tok

_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_web.html")


def _turn(brain, text):
    """One UI turn -> the real brain. Returns the JSON the page renders."""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return {"intent": "gap", "chips": [], "reply": ""}
    if low.startswith("learn:"):
        body = t[len("learn:"):]
        if "|" not in body or not body.split("|", 1)[0].strip() or not body.split("|", 1)[1].strip():
            return {"intent": "teach", "chips": ["teach"],
                    "reply": "To teach me, use:  learn: <question or topic> | <answer>"}
        q, a = body.split("|", 1)
        brain.teach(_tok(q), a.strip())
        return {"intent": "learn", "chips": ["learn", "+1 fact"],
                "reply": f"Learned -- and written to my SSD as raw binary, so I will still know it after a "
                         f"restart. I now know {brain.chat.n_known} things.",
                "src": "durable: knowledge.bin"}
    if low.startswith("think:"):
        seed = t[len("think:"):].strip()
        chain, ff = brain.think(seed)
        if len(chain) >= 2:
            return {"intent": "think", "chips": ["think", "internal"], "reply": " -> ".join(chain),
                    "src": f"a train of thought held internally, each hop gated by ignition; a feedforward "
                           f"J-space gets {ff} and would have to write it down to go further."}
        return {"intent": "think", "chips": ["think", "gap"],
                "reply": f"'{seed}' did not spark a chain -- I have not linked enough about it yet."}

    out = brain.converse(t)
    source = out.get("source")
    if source == "reason":
        chips, src = ["reason", "IGNITED"], "reasoning won ignition in the global workspace, over plain recall"
    elif source == "recall":
        chips = ["recall", "KNOWN cov=%d%%" % round(float(out.get("coverage") or 0) * 100)]
        src = "a learned fact ignited into conscious access"
    elif source == "introspection":
        chips, src = ["ask_conscious", "agnostic"], None
    else:
        chips, src = ["gap", "GAP"], "nothing reached conscious access -- an honest gap, not a guess"
    return {"intent": source or "gap", "chips": chips, "reply": out["reply"], "src": src,
            "known": bool(out.get("known"))}


class _Handler(BaseHTTPRequestHandler):
    brain = None
    gpu = False

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(_HTML, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            b = self.brain
            self._send(200, json.dumps({"n_known": b.chat.n_known, "facts": b.klog.count(),
                                        "concepts": b.n_learned, "learning": bool(b.learning),
                                        "learned": len(b.learned_log), "gpu": bool(self.gpu)}))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/teach_bulk"):
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/teach_bulk":
            # MASS INGEST: {"facts":[{cue,answer,triple?,src?},...]} -> ONE lock + ONE fsync.
            # Much faster than looping /api/chat "learn:": profiling showed a single teach costs ~1.5ms
            # of real work but ~97ms over HTTP under lock contention with the background learner, so
            # amortizing BOTH the fsync and the lock over a batch is where the win is.
            try:
                wrote = self.brain.teach_many(body.get("facts") or [])
            except Exception as e:
                self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))
                return
            self._send(200, json.dumps({"written": wrote, "n_known": self.brain.chat.n_known,
                                        "facts": self.brain.klog.count()}))
            return
        self._send(200, json.dumps(_turn(self.brain, str(body.get("text", "")))))

    def log_message(self, *a):
        return                                             # quiet: the brain's own prints are the log


def serve(brain, port=8756, gpu=False, forever=True):
    _Handler.brain, _Handler.gpu = brain, gpu
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)   # LOCAL ONLY (teachable brain)
    if not forever:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    print(f"\n  SPIKE CHAT  ->  http://127.0.0.1:{port}")
    print("  (the brain keeps learning in the background while you talk; Ctrl-C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] the brain is going to sleep; everything it learned is already on the SSD.")
    finally:
        srv.server_close()
    return srv


def _selfcheck():
    """Deterministic, offline: the UI is served and every turn hits the REAL brain."""
    import shutil
    import tempfile
    import urllib.request

    prev = os.environ.get("NORD_HOME")
    work = tempfile.mkdtemp(prefix="snn_web_")
    os.environ["NORD_HOME"] = work
    try:
        print("=" * 92)
        print("BRAIN WEB -- the Spike Chat UI wired LIVE to the real unified Brain")
        print("=" * 92)
        b = Brain(seed=0)
        print(b.boot())
        b._develop_symbolic(n_concepts=6)                  # give it concepts to converse about
        srv = serve(b, port=8757, gpu=False, forever=False)
        base = "http://127.0.0.1:8757"

        def post(text):
            req = urllib.request.Request(base + "/api/chat", method="POST",
                                         data=json.dumps({"text": text}).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=10).read())

        page = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=10).read())
        r_recall = post("what is digit 3")
        r_reason = post("the sun rises from the west")
        r_think = post("think: 3")
        r_teach = post("learn: what is a glorb | a glorb is a purple fruit.")
        r_after = post("what is a glorb")
        r_gap = post("what is a zzzz")
        srv.shutdown()
        srv.server_close()

        print(f"[ui]     GET / -> {len(page):,} bytes of the Spike Chat UI")
        print(f"[state]  {state}")
        for tag, r in (("recall", r_recall), ("reason", r_reason), ("think", r_think),
                       ("teach", r_teach), ("after-teach", r_after), ("gap", r_gap)):
            print(f"[{tag:11}] {str(r['reply'])[:88]}")
        print("=" * 92)

        assert "Spike Chat" in page and "/api/chat" in page, "the UI must be served and call the real API"
        assert state["n_known"] > 0, "the state endpoint must report what the brain knows"
        assert r_recall["known"] and "numeral" in r_recall["reply"], "a turn must hit the REAL brain's memory"
        assert "east" in r_reason["reply"].lower() and r_reason["intent"] == "reason", (
            "the UI must get the brain's real contradiction correction")
        assert " -> " in r_think["reply"], f"think: must return a real train of thought, got {r_think}"
        assert r_teach["intent"] == "learn" and r_after["known"], "teaching through the UI must be remembered"
        assert not r_gap["known"] and r_gap["intent"] == "gap", "an unknown must be an honest gap"
        b.close()
        print(f"self-check OK: the Spike Chat UI ({len(page):,} B) is served and wired LIVE to the real brain "
              f"-- recall, contradiction reasoning ('{r_reason['reply'][:40]}...'), a train of thought "
              f"({r_think['reply'][:28]}...), teaching (durable, then answered), and honest gaps all come "
              f"from the actual Brain, not from JavaScript. Local-only (127.0.0.1).")
        return 0
    finally:
        os.environ.pop("NORD_HOME", None)
        if prev is not None:
            os.environ["NORD_HOME"] = prev
        shutil.rmtree(work, ignore_errors=True)


def main():
    args = sys.argv[1:]
    if "--serve" not in args:
        return _selfcheck()
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8756
    b = Brain(seed=0)
    print(b.boot())
    b.start_learning()                                     # ALWAYS-ON while you chat
    # Serve FIRST (daemon thread) so the UI is usable immediately, then run the GPU curriculum in the
    # MAIN thread (CUDA belongs in the main thread) -- the brain gets smarter while the page is open.
    serve(b, port=port, gpu=False, forever=False)
    print(f"\n  SPIKE CHAT  ->  http://127.0.0.1:{port}")
    print("  (the brain keeps learning in the background while you talk; Ctrl-C to stop)\n")
    if "--develop" in args:
        print("[develop] conv-STDP perception curriculum starting on the GPU -- the UI is ALREADY live; "
              "the brain will know more as this finishes...")
        res = b.develop(mode="perceptual" if "--gpu" in args else "auto")
        _Handler.gpu = res.get("mode") != "symbolic"
        print(f"[develop] done ({'GPU perception' if _Handler.gpu else 'symbolic'}); "
              f"the brain now knows {b.chat.n_known} things.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[stop] the brain is going to sleep; everything it learned is already on the SSD.")
    finally:
        b.close()
    return 0


if __name__ == "__main__":
    main()
