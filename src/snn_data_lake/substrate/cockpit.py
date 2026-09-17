"""The SNDL cockpit: the ONE frontend. Live brain, live arm, hardware target, and chat, in one page.

WHAT THIS REPLACES
    Four surfaces used to show fragments of this brain: a matplotlib 3D window (chat_app/spiking_chat
    plus brain_activity_3d / brain_netplot / brain_region_activity), and the read-only dashboard that
    used to live inside brain_telemetry. They disagreed about what was live, none of them could show
    the arm and the chat together, and three of them needed matplotlib to draw a scatter plot. This is
    the single surface now; the others are retired.

THE DESIGN IS BORROWED, THE HONESTY MODEL IS NOT
    The visual language -- glass plates on a dark ground, a 1px cyan border, the L-shaped layout with a
    weighted right column over a seven-cell bottom bar, bar rows aligned on a common track, threshold
    recolouring -- is taken from mhflocke's Bittle dashboard (github.com/MarcHesse/mhflocke,
    src/viz/bittle_dashboard.py). It is a good-looking, information-dense layout and there is no reason
    to reinvent it.

    Their *data* model is not borrowed, on purpose. In their renderer the neuron cloud advances
    `state.angle` and `pulse_phase` every frame regardless of spikes (src/viz/brain_3d.py:261-262),
    while the spike raster refreshes only every --log-every training steps against a pose that
    refreshes every 10 -- so the cloud spins smoothly through ~100 frames of frozen spike data, and the
    motion reads as neural activity when it is a screensaver. Their synapse lines are seeded RNG, not
    topology, and their README says the opposite of their code.

    So here: rotation is a CAMERA, labelled as one, and it is the only thing that moves without data.
    Activity is carried by colour and radius only. The tracts between regions are the real typed
    connectome (snn_data_lake.connectome), not decoration. Anything the snapshot cannot supply renders
    as an em-dash, and anything the snapshot CAN supply but that does not mean what a live bar implies
    -- frozen at construction, sleep-only, a literal, a duplicate -- is drawn with its
    brain_telemetry.PROVENANCE tag beside it. A dashboard is the easiest place in a project to lie.

    ONE SECTION IS NOT BRAIN STATE, and it is named here rather than left for a reader to discover:
    `device`, the DEVICE ROUTING plate, comes from `substrate/device.decisions()`. A routing
    decision is a fact about the PROCESS, so `snapshot(brain)` -- which is handed a Brain -- has
    nothing to read it from. See `_snapshot` for why that does not weaken brain_telemetry's rule 1
    and for the measurement that fixes what its tag is allowed to say.

NOT A SECOND BRAIN (Policy 4)
    `serve()` never constructs a Brain. It is handed the live one. `main()` is the only place that may
    build a brain, and it takes the daemon's single-instance lock first, so a cockpit and a daemon can
    never be two writers on one durable lake.

WHY CHAT GOES THROUGH A QUEUE AND NOT THE REQUEST THREAD
    There is no lock anywhere in the brain stack -- a repo-wide search for threading primitives in
    src/snn_data_lake finds only this server's own thread. `Brain.respond()` is not a read: it writes
    `_help_drive` into the motor-drive path the tick consumes, exposes the language organ to
    plasticity that the sleep tick durably saves, runs an R-STDP update on the interaction learner, and
    can route a command through the gated arm. Calling that from an HTTP handler while the daemon ticks
    the same object is a data race on a brain that is also writing to disk. `Turns` therefore hands the
    text to whoever owns the tick loop; the reply is picked up by polling.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from snn_data_lake.brainstem.serial_arm import is_valid_command
from snn_data_lake.substrate.brain_telemetry import snapshot, verify_contract
from snn_data_lake.substrate.device import decisions

# The longest an allow-listed command token can be. Not a policy choice -- the allow-list itself is
# the policy, and this only stops an unbounded string from being copied before it is rejected.
_MAX_CMD_CHARS = 64
_MAX_CHAT_CHARS = 2000  # matches the daemon's own truncation of an ingested turn
_REPLY_KEEP = 64  # completed replies retained for polling before the oldest is dropped


class Acts:
    """Arm commands queued for the thread that owns the tick loop. Owner-authorised 2026-08-03.

    This is a NEW ACTUATION SURFACE -- a local HTTP client can now move the arm -- so every property
    that makes it defensible is stated here and enforced below rather than assumed.

    1. ALLOW-LIST ONLY, TWICE. `is_valid_command` is the wire allow-list, the same single source of
       truth `serial_arm` and `nl_interface` use. It is checked at SUBMIT (so a bad request gets an
       immediate 400 instead of a silent drop) and again at PUMP (so nothing that skipped the front
       door reaches the arm). The client never supplies a command the allow-list has not already
       accepted; it CHOOSES among tokens, it does not compose one.
    2. THE TICK THREAD EXECUTES, NOT THE REQUEST THREAD. There is no lock anywhere in the brain
       stack, and `mj_forward` running concurrently with the tick's `mj_step` on one `mjData` is
       undefined behaviour in MuJoCo -- not a race that produces a stale read, a race that produces
       garbage or a crash. Same reason `Turns` exists for chat, with a harder failure mode.
    3. THE GATE STILL HOLDS. Execution is `arm.apply()`, so the E-STOP latch, the collision check
       and the clamp all apply exactly as they do to the brain's own commands. A latched arm ignores
       these; MEASURED, a gated command moves 0.000000 deg while stopped.
    4. BOUNDED. One command per tick by default, so a client cannot starve the loop by flooding it.
    """

    def __init__(self):
        self._q = queue.Queue()
        self._done = {}
        # BOUNDED, like Turns. `_done` was written on every pump and never trimmed, so an
        # always-on daemon accumulated one record per arm command for the life of the process --
        # an unbounded dict inside the thing whose whole job is to keep running. Turns already
        # carries exactly this eviction (`_order` + _REPLY_KEEP) and Acts did not, which is the
        # kind of gap two near-identical classes produce.
        self._order = []
        self._lock = threading.Lock()
        self._n = 0
        self.applied = 0
        self.refused = 0

    def submit(self, command):
        """Queue an allow-listed command. Returns (ticket, None) or (None, reason)."""
        cmd = str(command or "").strip()[:_MAX_CMD_CHARS]
        if not cmd:
            return None, "empty command"
        if not is_valid_command(cmd):
            # The FRONT DOOR check. Rejecting here means a typo is a 400 the operator can see,
            # rather than a ticket that quietly never happens.
            return None, f"{cmd!r} is not an allow-listed command"
        with self._lock:
            self._n += 1
            tid = self._n
        self._q.put((tid, cmd))
        return tid, None

    def result(self, tid):
        with self._lock:
            return self._done.get(int(tid))

    def pending(self):
        return self._q.qsize()

    def pump(self, brain, budget=1):
        """Drain up to `budget` commands ON THE CALLING THREAD -- the owner of the tick loop."""
        n = 0
        while n < budget:
            try:
                tid, cmd = self._q.get_nowait()
            except queue.Empty:
                break
            n += 1
            try:
                if not is_valid_command(cmd):  # defence in depth: never trust the queue
                    self.refused += 1
                    rec = {"command": cmd, "applied": False, "reason": "not allow-listed"}
                else:
                    # T037: the cockpit is the brain's EFFERENT surface (P4c), so its
                    # motion passes the SAME tapped passage the loop uses -- the review
                    # panel proved the direct apply left the cockpit E-STOP invisible to
                    # the efference-copy boundary. Still the one gated path to motion:
                    # apply_observed calls So101Arm.apply() and only adds observation.
                    # T037a: the cockpit IS the deliberate reset -- a human's HOME / RETRACT_ALL
                    # is the one thing that re-arms a latch restored from a previous process.
                    st = brain.brain.arm.apply_observed(cmd, deliberate=True)
                    self.applied += 1
                    restored = st.get("latch_source") == "restored"
                    rec = {
                        "command": cmd,
                        "applied": True,
                        "stopped": bool(st.get("stopped")),
                        "theta_deg": [round(float(x), 4) for x in (st.get("theta_deg") or [])],
                        # A latched arm accepts the ticket and does not move. Saying so is the
                        # difference between "your command ran" and "your command was ignored".
                        "note": (
                            (
                                "E-STOP latched (restored at boot; HOME resets); no motion"
                                if restored
                                else "E-STOP latched; no motion"
                            )
                            if st.get("stopped")
                            else ""
                        ),
                    }
            except Exception as e:  # a malformed command must not stop the daemon's loop
                self.refused += 1
                rec = {"command": cmd, "applied": False, "reason": str(e)}
            with self._lock:
                self._done[tid] = rec
                self._order.append(tid)
                while len(self._order) > _REPLY_KEEP:
                    self._done.pop(self._order.pop(0), None)
        return n


class Turns:
    """Chat turns queued for the thread that owns the tick loop. The frontend never touches the brain.

    Submit returns a ticket immediately rather than blocking the request thread: `respond()` has no
    bounded runtime (the torch-tier demo prints per-reply seconds because they vary that much), and a
    blocked handler thread holding a slow reply is how a dashboard takes down the daemon it watches.
    """

    def __init__(self):
        self._q = queue.Queue()
        self._done = {}
        self._order = []
        self._lock = threading.Lock()
        self._n = 0

    def submit(self, text):
        """Queue a turn. Returns its ticket id."""
        with self._lock:
            self._n += 1
            tid = self._n
        self._q.put((tid, str(text)[:_MAX_CHAT_CHARS]))
        return tid

    def result(self, tid):
        """The finished reply dict for a ticket, or None while it is still queued."""
        with self._lock:
            return self._done.get(int(tid))

    def pending(self):
        return self._q.qsize()

    def pump(self, brain, budget=1):
        """Drain up to `budget` turns ON THE CALLING THREAD. The owner of the tick loop calls this.

        Returns the number of turns handled. A failing turn is recorded as an error reply rather than
        raised: a malformed message must not be able to stop the daemon's loop.
        """
        n = 0
        while n < budget:
            try:
                tid, text = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                out = brain.respond(text, human=True)  # T037a: a person typed this
                rec = {
                    "reply": out.get("reply", ""),
                    "route": out.get("route", ""),
                    "somatic_recall": out.get("somatic_recall"),
                    "t": time.time(),
                }
            except Exception as e:
                rec = {
                    "reply": "",
                    "route": "",
                    "error": f"{type(e).__name__}: {e}",
                    "t": time.time(),
                }
            with self._lock:
                self._done[tid] = rec
                self._order.append(tid)
                while len(self._order) > _REPLY_KEEP:
                    self._done.pop(self._order.pop(0), None)
            n += 1
        return n


def _geometry():
    """The anatomical layout, computed once. Hand-authored coordinates; REAL connectome edges."""
    from snn_data_lake.substrate import brain_geometry

    return brain_geometry.layout()


# The ONE section of the served payload that is not brain state, and why it is assembled here
# rather than inside `brain_telemetry.snapshot()`: `snapshot(brain)` is handed a Brain, and a
# routing decision is a fact about the PROCESS -- there is no attribute on the brain to read it
# from. brain_telemetry's rule 1 ("snapshot() is the ONLY source a panel may read") is kept where
# it bites: every BRAIN number on this page still comes from snapshot() and nowhere else, this one
# section comes from a single named public accessor, and it is tagged through the SAME provenance
# table as everything else rather than being given a private honesty mechanism of its own.
#
# WHAT THE TAG HAS TO SAY, and it is the part that took measuring rather than wiring. The stated
# purpose of `device.decisions()` was "so the cockpit can answer 'why is the GPU idle'". Measured
# over 200 ticks of a live `make_brain()` process: ONE workload reaches the router, while
# `Brain._perception_device` is set to 'cuda' by an availability check that never calls `select()`,
# and peak CUDA allocation is 3,147,264 bytes. The GPU is NOT idle, so a plate captioned with that
# question would be false on its face. It is captioned as what it is -- a log of the choices that
# asked the router -- and the tag carries the rest.
#
# THE TAG IS SHORT ON PURPOSE, and the first version of it was not. `tag()` renders inline inside
# `big()`, in a bottom-bar cell one eighth of the page wide, so the 360-character paragraph this
# started as measured a 1,921px scrollHeight in a 108px cell -- against 146-365px for the seven
# cells beside it. The number it was supposed to qualify was pushed out of the plate entirely, so
# the honesty tag defeated the panel it was attached to. `spikes.midbrain` is the tree's only other
# long PROVENANCE string and it is never rendered through `tag()` at all -- `drawRegions` uses it
# as a BOOLEAN (`flagged=!!PROV[...]`) -- so it was not the precedent it looked like. DATED
# 2026-09-16 (the T082 panel): that sentence stood at its own date and is left as written; four
# LONGER strings are now rendered through `tag()` under `.row .v`'s nowrap -- the integration row's
# (516 characters) and the T080, T081 and T082 consciousness rows' (1,806 / 3,352 / 4,316; no symbol
# is named here on purpose -- the C5 surface gate reads this file's text) -- each clipped
# on BOTH axes in the ACCESS plate (T082 record 3.7). The tag
# carries the claim; the plate's `note` carries the consequence; the measurement behind both is in
# the block above and in `device.decisions()`.
_DEVICE_PROV = "router log, not a GPU census"


def _snapshot(brain):
    """`brain_telemetry.snapshot()` plus the `device` section the DEVICE ROUTING plate draws.

    Every served endpoint goes through this, including the offline `_http_get` used by the
    self-check, so the payload the criterion inspects is the payload a browser receives.
    """
    snap = snapshot(brain)
    snap["device"] = decisions()
    snap["provenance"] = {**(snap.get("provenance") or {}), "device_routing": _DEVICE_PROV}
    return snap


def _handler(get_snap, turns, get_view=None, acts=None):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, qs = self.path.partition("?")
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, COCKPIT_HTML, "text/html; charset=utf-8")
                if path == "/api/geom":
                    return self._send(200, json.dumps(_geometry()))
                if path == "/api/arm.png":
                    # Requesting is all this thread does. Drawing happens on the tick thread --
                    # see substrate/arm_view.py -- so a browser poll only registers interest and
                    # collects whatever the last tick produced.
                    view = get_view() if get_view is not None else None
                    if view is None:
                        return self._send(503, json.dumps({"error": "no arm view"}))
                    view.request()
                    png, rev = view.frame()
                    if png is None:
                        # 503 with the REASON, not a placeholder image. A grey rectangle would
                        # be indistinguishable from a genuinely grey arm.
                        return self._send(503, json.dumps({"error": rev}))
                    return self._send(200, png, "image/png")
                if path == "/api/look":
                    # VIEWPOINT ONLY. Steers the camera, never the arm -- there is no path from
                    # here to a motor command, which is why it needs no allow-list. Bad numbers
                    # are clamped by ArmView.look() rather than rejected, so a dragged cockpit
                    # cannot 400 mid-gesture.
                    look_view = get_view() if get_view is not None else None
                    if look_view is None:
                        return self._send(404, json.dumps({"error": "no arm view"}))
                    q = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)

                    def _f(name):
                        try:
                            return float(q[name]) if name in q else None
                        except ValueError:
                            return None

                    cam = look_view.look(_f("az"), _f("el"), _f("d"))
                    look_view.request()  # a drag is also a statement that somebody is watching
                    return self._send(200, json.dumps(cam))
                # An id-less (or non-digit-id) GET on either queue endpoint is LOUD, never
                # the pending default. The old shape returned {"pending": true}
                # UNCONDITIONALLY for a missing id -- a constant wearing a queue-state's
                # clothes (P4c) -- and it cost a morning of false "30-minute reply stall"
                # observations against a daemon that was answering in three seconds: an
                # external poller reading the bare GET could not distinguish "you forgot
                # the ticket id" from "your ticket is queued". The frontend and the smoke
                # test always pass ?id=, so only the misleading path changes shape.
                if path == "/api/act":
                    tid = dict(p.split("=", 1) for p in qs.split("&") if "=" in p).get("id", "")
                    if acts is None or not tid.isdigit():
                        return self._send(
                            200,
                            json.dumps(
                                {
                                    "error": "id required (use /api/act?id=<ticket>)",
                                    "pending_count": acts.pending() if acts is not None else 0,
                                }
                            ),
                        )
                    return self._send(200, json.dumps(acts.result(tid) or {"pending": True}))
                if path == "/api/chat":
                    tid = dict(p.split("=", 1) for p in qs.split("&") if "=" in p).get("id", "")
                    if not tid.isdigit():
                        return self._send(
                            200,
                            json.dumps(
                                {
                                    "error": "id required (use /api/chat?id=<ticket>)",
                                    "pending_count": turns.pending(),
                                }
                            ),
                        )
                    return self._send(200, json.dumps(turns.result(tid) or {"pending": True}))
                snap = get_snap()
                if path == "/api/state":
                    return self._send(200, json.dumps(snap["cognition"]))
                if path == "/api/spikes":
                    return self._send(200, json.dumps(snap["spikes"]))
                if path == "/api/all":
                    return self._send(200, json.dumps(snap))
                return self._send(404, json.dumps({"error": "not found"}))
            except Exception as e:
                # The 500 itself can fail: if the client already gave up, the socket is gone and
                # end_headers() raises ConnectionAbortedError *inside* the handler for the first
                # error. That second exception is unhandled, so socketserver prints a full traceback
                # -- which is how this log reached 5.4 MB of WinError 10053 dumps. An error reply is
                # best-effort by nature; nobody is listening.
                try:
                    return self._send(500, json.dumps({"error": str(e)}))
                except OSError:
                    return None

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/api/act":
                # OWNER-AUTHORISED ACTUATION, 2026-08-03. The request thread does NOT touch the arm:
                # it validates against the wire allow-list and hands a ticket to the tick thread.
                # 404 when no queue was supplied, so a cockpit served without one cannot actuate.
                if acts is None:
                    return self._send(404, json.dumps({"error": "actuation not enabled"}))
                try:
                    n = min(int(self.headers.get("Content-Length") or 0), _MAX_CMD_CHARS * 8)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    tid, why = acts.submit(body.get("command"))
                    if tid is None:
                        return self._send(400, json.dumps({"error": why}))
                    return self._send(200, json.dumps({"id": tid, "queued": True}))
                except Exception as e:
                    return self._send(500, json.dumps({"error": str(e)}))
            if path != "/api/chat":
                return self._send(404, json.dumps({"error": "not found"}))
            try:
                # Cap the read at the length a turn can actually be. Content-Length is a claim made
                # by the client, and `read(n)` on an unbounded claim allocates whatever it asks for.
                # Loopback-only binding makes this low-risk, not no-risk -- anything local can POST.
                n = min(int(self.headers.get("Content-Length") or 0), _MAX_CHAT_CHARS * 4)
                body = json.loads(self.rfile.read(n) or b"{}")
                text = (body.get("text") or "").strip()
                if not text:
                    return self._send(400, json.dumps({"error": "empty message"}))
                return self._send(200, json.dumps({"id": turns.submit(text), "queued": True}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        def log_message(self, *a):  # keep the brain's stdout clean
            return

    return H


def serve(brain, port=8137, turns=None, block=True, acts=None):
    """Serve the cockpit over the live brain. Does NOT construct a brain -- Policy 4: exactly one.

    Binds loopback only. This is a window onto a local process, not a network service, and `/api/chat`
    accepts writes -- so it must not become reachable off the machine by default.
    """
    turns = Turns() if turns is None else turns

    class _Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            """A client that hangs up is normal, not an incident.

            The default implementation prints the whole traceback to stderr. A browser tab closing
            mid-poll, or a curl that times out, produces ConnectionAbortedError/ConnectionResetError
            every time -- and the daemon appends stderr to a log file, so this alone wrote 5.4 MB of
            WinError 10053 dumps and buried the messages that actually mattered. Anything that is NOT
            a disconnect still surfaces.
            """
            import sys
            import traceback

            if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
                traceback.print_exc()

    def _view():
        try:
            return brain.brain.arm.arm.view
        except Exception:
            return None

    # `acts` MUST be forwarded: `_handler` 404s /api/act when it is None, so omitting it here made
    # every cockpit arm button -- including EMERGENCY_STOP -- a no-op while the daemon was correctly
    # passing a queue in (`substrate/learn_daemon.py`, the `serve(...)` call in `main()`). The
    # actuation tests all construct Acts() directly and never traverse serve(), so they proved the
    # mechanism and said nothing about production.
    #
    # That cite was a line number into `learn_daemon.py` until 2026-08-04, and it resolved against
    # the 32-line ROOT SHIM -- a file with no such line at all, since the daemon moved into the
    # package under P4c. Naming the symbol instead, per the lesson `self_mod_harness.py` records.
    # (The stale number is deliberately not repeated here: a corrected cite that still contains a
    # resolvable-looking `file.py:NNN` token re-arms every scanner that looks for one.)
    srv = _Server(("127.0.0.1", int(port)), _handler(lambda: _snapshot(brain), turns, _view, acts))
    srv.turns = turns
    # the BOUND port, not the requested one: port=0 means "OS, pick one", and printing the request
    # would tell the reader to open http://127.0.0.1:0/
    bound = srv.server_address[1]
    print(f"[cockpit] http://127.0.0.1:{bound}/  (live brain, arm, hardware target, chat)")
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return srv


COCKPIT_HTML = r"""<!doctype html><meta charset="utf-8"><title>SNDL cockpit</title>
<style>
/* palette lifted from mhflocke bittle_dashboard.py L24-55 -- their exact video values */
:root{
 --cy:#00c8dc; --ice:#a0c3eb; --blu:#3c82c8; --gold:#ffc83c; --red:#ff3c28; --vio:#b96ef0;
 --grn:#32c864; --org:#ffa028; --grey:#96a0b4; --wht:#dce1eb; --pink:#ff64a0; --lime:#a0ff50;
 --glass:rgba(6,10,18,.90); --edge:rgba(0,200,220,.47); --barbg:#191e2a; --deep:#0a0c12;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--deep);color:var(--wht);overflow:hidden;
 font:13px/1.4 ui-monospace,Consolas,"DejaVu Sans Mono",monospace}
.plate{background:var(--glass);border:1px solid var(--edge);border-radius:6px}
h2{font:600 13px/1 ui-monospace,Consolas,monospace;letter-spacing:.06em;color:rgba(0,200,220,.94);
 margin:0 0 8px;padding:5px 0 0 12px}
#app{display:grid;height:100vh;padding:10px;gap:6px;
 grid-template-columns:1fr 460px 360px;grid-template-rows:28px 1fr 110px;
 grid-template-areas:"brand brand brand" "scene widgets chat" "bottom bottom bottom"}
#brand{grid-area:brand;display:flex;align-items:center;gap:14px;padding:0 12px;
 background:rgba(8,10,16,.63);border:1px solid rgba(0,200,220,.20);border-radius:4px;font-size:12px}
#brand b{color:var(--cy);letter-spacing:.14em}
#prog{flex:1;height:2px;background:#16203a;border-radius:2px;overflow:hidden}
#prog i{display:block;height:100%;background:var(--cy);width:0}
#scene{grid-area:scene;position:relative;overflow:hidden}
/* ARM IS THE MAIN STAGE. The arm picture fills the scene; the anatomy render is demoted to a
   mini alongside WORKSPACE and REGIONS. object-fit:contain so the MuJoCo frame keeps its aspect
   instead of being stretched into a lie about the arm's proportions. */
#armStage{position:absolute;inset:0;display:flex;align-items:center;justify-content:center}
#armImg{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;
 border-radius:6px;background:#0b0f16}
#armNone{position:absolute;font-size:12px;color:var(--grey);letter-spacing:.06em}
#armStage{cursor:grab}#armStage.drag{cursor:grabbing}
#armCtl{position:absolute;right:12px;top:10px;z-index:4;display:flex;flex-direction:column;
 gap:4px;align-items:stretch;width:118px}
#armCtl button{background:rgba(0,200,220,.12);border:1px solid var(--edge);color:var(--cy);
 border-radius:4px;padding:4px 6px;cursor:pointer;font:11px ui-monospace,Consolas,monospace}
#armCtl button:hover{background:rgba(0,200,220,.24)}
#armCtl button.estop{background:rgba(255,70,70,.14);border-color:rgba(255,90,90,.55);color:#ff8a8a}
#armCtl #actMsg{font-size:10px;color:var(--grey);min-height:12px;word-break:break-word}
#brain3d{position:absolute;inset:0;width:100%;height:100%;cursor:grab}
#brain3d:active{cursor:grabbing}
#cam{position:absolute;left:12px;top:10px;font-size:11px;color:var(--grey);z-index:3}
#cam b{color:var(--cy)}
#trace{position:absolute;left:12px;right:12px;bottom:12px;height:86px;z-index:2}
.mini{position:absolute;bottom:110px;width:196px;height:196px;z-index:2}
#mmWorld{left:12px}#mmBrain{left:216px}#mmAnat{left:420px}
#mmAnat{overflow:hidden;border-radius:6px}
.mini canvas{width:100%;height:100%;display:block;border-radius:6px}
.mini .lbl{position:absolute;left:8px;top:6px;font-size:11px;letter-spacing:.10em;z-index:1}
/* THE COLUMN SCROLLS AND ITS FIRST PLATE OWNS THE SCROLL ORIGIN.
   Every plate here used to be a SHARE of a fixed-height column (flex:N -- grow N, shrink 1,
   basis 0) inside overflow:hidden, so a plate's height followed the OTHER plates' numbers
   rather than its own text, every plate was cut, and what was cut could not be reached at
   all. Now each plate is exactly its own content's height (flex:0 0 auto -- no grow factor,
   because grow can only fire on a viewport far taller than this page is ever opened at, and
   a factor that never fires is a number someone will one day tune to no effect) and the
   column itself scrolls, so nothing rendered is out of reach. The FIRST plate's top offset
   is 0 by construction: its in-view budget is the whole column, and no other plate's
   content, wrap or number can move it -- which is why the plate that owes an in-view
   criterion sits at the top of this column rather than fourth down it. scrollbar-gutter:
   stable holds the text measure constant whether or not the column currently overflows (it
   does not, for the first second after a reload, before the first /api/all answers), so a
   wrapped sentence's height does not depend on how tall some other plate happens to be. */
#widgets{grid-area:widgets;display:flex;flex-direction:column;gap:6px;
 overflow-x:hidden;overflow-y:auto;scrollbar-gutter:stable}
#widgets>div{padding-bottom:8px;overflow:hidden}
#chat{grid-area:chat;display:flex;flex-direction:column;overflow:hidden}
#log{flex:1;overflow-y:auto;padding:4px 12px;display:flex;flex-direction:column;gap:8px}
#log .m{max-width:100%;white-space:pre-wrap;word-break:break-word}
#log .u{color:var(--ice)}#log .u b{color:var(--cy)}
#log .b b{color:var(--gold)}
#log .rt{color:var(--grey);font-size:11px}
#log .er{color:var(--red)}
#cf{display:flex;gap:6px;padding:8px 10px;border-top:1px solid rgba(0,200,220,.20)}
#ci{flex:1;background:#0c1018;border:1px solid #1e2a3e;border-radius:4px;color:var(--wht);
 padding:6px 8px;font:13px ui-monospace,Consolas,monospace}
#ci:focus{outline:none;border-color:var(--edge)}
#cs{background:rgba(0,200,220,.14);border:1px solid var(--edge);color:var(--cy);border-radius:4px;
 padding:6px 12px;cursor:pointer;font:13px ui-monospace,monospace}
#cs:disabled{opacity:.45;cursor:default}
#bottom{grid-area:bottom;display:grid;grid-template-columns:repeat(8,1fr);gap:6px}
#bottom>div{overflow:hidden;position:relative}
.row{display:flex;justify-content:space-between;gap:8px;padding:1px 12px;font-size:12px}
.row .k{color:var(--grey);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .v{color:var(--wht);white-space:nowrap}
.big{font:700 26px/1 ui-monospace,Consolas,monospace;padding:2px 12px 4px;color:var(--cy)}
.sub{font-size:11px;color:var(--grey);padding:0 12px 1px}
.bar{margin:2px 12px 5px;height:10px;background:rgba(25,30,42,.71);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:3px}
.na{color:#4d5a70!important}
.tag{color:var(--org);font-size:10px;letter-spacing:.02em}
.note{color:var(--grey);font-size:11px;padding:3px 12px 0;font-style:italic}
/* Every other value on this page is a NUMBER, so `.row .v` is nowrap and a long tag simply runs
   off the plate. These are SENTENCES, and a sentence cut at the plate edge says something the
   author did not write -- so this block wraps, and the provenance string (thousands of characters
   on some rows) is bounded and scrollable inside a collapsed details instead of a nowrap tag. */
.c5{white-space:normal;font-size:11px;line-height:1.35;padding:2px 12px}
.c5 i{color:var(--grey);font-style:normal}
.c5 .rf{color:var(--org)}
.c5 details{margin-top:1px}
.c5 summary{color:var(--grey);font-size:10px;letter-spacing:.04em;cursor:pointer}
.c5 details>div{white-space:normal;color:var(--grey);font-size:10px;line-height:1.3;
 max-height:96px;overflow:auto;padding:2px 0 0}
canvas{display:block}
@media(max-width:1500px){#app{grid-template-columns:1fr 420px;
 grid-template-areas:"brand brand" "scene widgets" "chat chat" "bottom bottom";
 grid-template-rows:28px 1fr 220px 110px}}
</style>
<div id="app">
 <div id="brand"><b>SNDL</b><span id="bver">--</span><span id="bcirc">--</span>
   <span id="btick">--</span><span id="bnight">--</span><div id="prog"><i></i></div>
   <span id="berr" style="color:var(--red)"></span></div>

 <div id="scene" class="plate">
   <div id="armStage"><img id="armImg" alt="live arm" style="display:none" draggable="false">
     <div id="armNone">arm view unavailable</div></div>
   <div id="armCtl">
     <button data-cmd="HOME">HOME</button>
     <button data-cmd="JOINT_SHOULDER_PAN_ROTATE(+6deg)">pan +</button>
     <button data-cmd="JOINT_SHOULDER_PAN_ROTATE(-6deg)">pan -</button>
     <button data-cmd="JOINT_SHOULDER_LIFT_ROTATE(+6deg)">lift +</button>
     <button data-cmd="JOINT_SHOULDER_LIFT_ROTATE(-6deg)">lift -</button>
     <button data-cmd="JOINT_ELBOW_FLEX_ROTATE(+6deg)">elbow +</button>
     <button data-cmd="JOINT_ELBOW_FLEX_ROTATE(-6deg)">elbow -</button>
     <button data-cmd="EMERGENCY_STOP" class="estop">E-STOP</button>
     <span id="actMsg"></span>
   </div>
   <div id="cam"><b>ARM</b> &middot; <span id="camtxt">camera rotating</span><br>
     <span style="color:#4d5a70">MuJoCo, drawn on the tick thread on demand. the frame counter
     in the ARM panel is the only thing telling a live picture from a frozen one.</span></div>
   <div class="mini plate" id="mmWorld"><div class="lbl" style="color:var(--cy)">WORKSPACE</div>
     <canvas id="cvWorld"></canvas></div>
   <div class="mini plate" id="mmBrain"><div class="lbl" style="color:var(--vio)">REGIONS</div>
     <canvas id="cvBrain"></canvas></div>
   <div class="mini plate" id="mmAnat"><div class="lbl" style="color:var(--cy)">ANATOMY</div>
     <canvas id="brain3d"></canvas></div>
   <canvas id="trace" class="plate"></canvas>
 </div>

 <div id="widgets">
   <div class="plate" style="flex:0 0 auto"><h2>ARM</h2><div id="wArm"></div></div>
<div class="plate" style="flex:0 0 auto"><h2>MOTOR MIX</h2><div id="wMotor"></div></div>
   <div class="plate" style="flex:0 0 auto"><h2>BEHAVIOR &amp; EMOTION</h2><div id="wBeh"></div></div>
   <div class="plate" style="flex:0 0 auto"><h2>CONSCIOUSNESS -- what is measured, what it licenses</h2><div id="wC5"></div></div>
   <div class="plate" style="flex:0 0 auto"><h2>HARDWARE TARGET &mdash; NOT BUILT</h2><div id="wHw"></div></div>
   <div class="plate" style="flex:0 0 auto"><h2>NEUROMODULATORS</h2><div id="wNm"></div></div>
   <div class="plate" style="flex:0 0 auto"><h2>WORLD MODEL</h2><div id="wWm"></div></div>
 </div>

 <div id="chat" class="plate"><h2>CHAT &mdash; canonical Brain.respond</h2>
   <div id="log"></div>
   <div id="cf"><input id="ci" placeholder="say something to the brain" autocomplete="off">
     <button id="cs">send</button></div>
 </div>

 <div id="bottom">
   <div class="plate"><h2>COVERAGE</h2><div id="bCov"></div></div>
   <div class="plate"><h2>REACH</h2><div id="bReach"></div></div>
   <div class="plate"><h2>STABILITY</h2><div id="bStab"></div></div>
   <div class="plate"><h2>AFFECT</h2><canvas id="cvVA"></canvas><div id="bVA"></div></div>
   <div class="plate"><h2>ACCESS</h2><div id="bAcc"></div></div>
   <div class="plate"><h2>LIFE</h2><div id="bLife"></div></div>
   <div class="plate"><h2>LAKE &amp; CENSUS</h2><div id="bLake"></div></div>
   <div class="plate"><h2>DEVICE ROUTING</h2><div id="bDev"></div></div>
 </div>
</div>
<script>
const $=i=>document.getElementById(i);
const EM='—';
let PROV={}, GEOM=null, HIST=[], AZ=-0.6, EL=0.30, DRAG=null, SPIN=true;

/* ---------- formatting: absent renders as an em-dash, never a zero bar ---------- */
function fmt(v,d){ if(v===null||v===undefined||v==='')return EM;
 if(Array.isArray(v))return v.map(x=>fmt(x,d)).join(' / ');   /* a vector is not a scalar */
 if(typeof v==='number'){ if(!isFinite(v))return EM;
  return (Math.abs(v)>=1000)?v.toFixed(0):v.toFixed(d===undefined?3:d);} return String(v);}
function tag(k){ const t=PROV[k]; return t?` <span class="tag">${t}</span>`:''; }
function row(k,v,d,pk){ const miss=(v===null||v===undefined);
 return `<div class="row"><span class="k">${k}</span><span class="v${miss?' na':''}">${fmt(v,d)}${tag(pk||'')}</span></div>`;}
/* A bar is a CLAIM: "this is one number, and it lies in [0,1]". Anything else -- a vector, a
   non-finite, a string -- gets the plain row instead. `neuromod_astrocyte` is what taught this:
   it is a list of per-domain glial gains that run above 1.0, and drawing it as a bar rendered
   "1,1.0024" as though it were a saturated level. */
function bar(k,v,col,pk){
 if(v===null||v===undefined||typeof v!=='number'||!isFinite(v))return row(k,v,3,pk);
 const p=Math.max(0,Math.min(1,v))*100;
 return row(k,v,3,pk)+`<div class="bar"><i style="width:${p}%;background:${col||'var(--cy)'}"></i></div>`;}
/* `pk` added 2026-08-04 (specs/010, C30). `big` had NO provenance slot, so the LARGEST number on
   the page was the one value that structurally could not carry a tag -- and `access_index` was
   drawn by it directly above two `row()`s that do carry one, making the untagged headline read as
   the more trustworthy of the three. A renderer that cannot express "this cannot vary" guarantees
   the most prominent number never says it. */
function big(v,d,col,pk){return `<div class="big"${col?` style="color:${col}"`:''}>${fmt(v,d)}${tag(pk||'')}</div>`;}
function sub(t){return `<div class="sub">${t}</div>`;}

/* threshold recolouring, mhflocke's convention */
const cGood=v=>v>.66?'var(--grn)':v>.33?'var(--gold)':'var(--red)';
const cLoad=v=>v>.66?'var(--red)':v>.33?'var(--gold)':'var(--grn)';

/* ---------- hi-dpi canvas ---------- */
function fit(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;
 if(c.width!==Math.round(r.width*d)||c.height!==Math.round(r.height*d)){
  c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);}
 const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return[x,r.width,r.height];}

/* ---------- the anatomical brain. rotation is a CAMERA and is labelled as one. ---------- */
const RC={cortex:'#00c8dc',hippocampus:'#b96ef0',amygdala:'#ff3c28',basal_ganglia:'#ffc83c',
 cerebellum:'#32c864',brainstem:'#ffa028',thalamus:'#3c82c8',hypothalamus:'#ff64a0',
 midbrain:'#a0ff50',medulla:'#96a0b4',pons:'#96a0b4',epithalamus:'#a0c3eb',olfactory:'#4d5a70',
 claustrum:'#a0c3eb',neuromodulation:'#ff64a0',plasticity:'#a0ff50',substrate:'#96a0b4',
 reasoning:'#dce1eb'};
function proj(p,w,h){
 const ca=Math.cos(AZ),sa=Math.sin(AZ),ce=Math.cos(EL),se=Math.sin(EL);
 const x=p.x*ca-p.y*sa, y=p.x*sa+p.y*ca;           /* yaw about the vertical axis */
 const y2=y*ce-p.z*se,  z2=y*se+p.z*ce;             /* pitch */
 const s=Math.min(w,h)*0.42, d=1/(1+z2*0.28);       /* mild perspective */
 return [w/2+x*s*d, h/2-y2*s*d, z2, d];
}
function drawBrain(spikes){
 const [x,w,h]=fit($('brain3d')); x.clearRect(0,0,w,h);
 if(!GEOM){x.fillStyle='#4d5a70';x.font='13px monospace';x.fillText('loading anatomy...',16,h/2);return;}
 const cen={}; GEOM.regions.forEach(r=>cen[r.name]={x:r.x,y:r.y,z:r.z});
 /* real connectome tracts, drawn first so nodes sit on top */
 x.lineWidth=1;
 GEOM.edges.forEach(([a,b])=>{ if(!cen[a]||!cen[b])return;
  const A=proj(cen[a],w,h), B=proj(cen[b],w,h);
  const act=Math.max(spikes[a]||0,spikes[b]||0);
  x.strokeStyle=`rgba(60,130,200,${0.10+0.42*act})`;
  x.beginPath();x.moveTo(A[0],A[1]);x.lineTo(B[0],B[1]);x.stroke();});
 /* modules, depth sorted back to front */
 const pts=GEOM.modules.map(m=>{const P=proj(m,w,h);return{m,X:P[0],Y:P[1],Z:P[2],D:P[3]};})
   .sort((p,q)=>q.Z-p.Z);
 pts.forEach(p=>{
  const v=spikes[p.m.region]; const has=(v!==null&&v!==undefined);
  const a=has?v:0;
  const r=(1.6+3.4*a)*p.D;
  x.globalAlpha=has?(0.34+0.62*a):0.13;          /* no signal -> ghosted, never bright */
  x.fillStyle=RC[p.m.region]||'#96a0b4';
  x.beginPath();x.arc(p.X,p.Y,Math.max(0.7,r),0,7);x.fill();});
 x.globalAlpha=1;
 /* region labels at centroids, brightest first */
 x.font='10px ui-monospace,monospace';
 GEOM.regions.forEach(r=>{const P=proj(r,w,h);const v=spikes[r.name]||0;
  x.fillStyle=`rgba(220,225,235,${0.30+0.60*v})`;
  x.fillText(r.name,P[0]+6,P[1]-6);});
}

/* ---------- rolling rate trace. one sample per poll; the buffer IS the history. ---------- */
function drawTrace(spikes){
 const [x,w,h]=fit($('trace')); x.clearRect(0,0,w,h);
 const ks=Object.keys(spikes);
 const mean=ks.length?ks.reduce((s,k)=>s+(spikes[k]||0),0)/ks.length:null;
 if(mean!==null){HIST.push(mean); if(HIST.length>180)HIST.shift();}
 x.strokeStyle='#16203a';x.lineWidth=1;
 [0.25,0.5,0.75].forEach(f=>{x.beginPath();x.moveTo(0,h*f);x.lineTo(w,h*f);x.stroke();});
 x.fillStyle='rgba(0,200,220,.90)';x.font='10px ui-monospace,monospace';
 x.fillText('MEAN REGION RATE  '+(mean===null?EM:mean.toFixed(3))+'   n='+HIST.length,8,13);
 if(HIST.length<2)return;
 x.beginPath();
 HIST.forEach((v,i)=>{const px=i/(HIST.length-1)*w, py=h-6-v*(h-22);
  i?x.lineTo(px,py):x.moveTo(px,py);});
 x.strokeStyle='#00c8dc';x.lineWidth=1.5;x.stroke();
 x.lineTo(w,h);x.lineTo(0,h);x.closePath();x.fillStyle='rgba(0,200,220,.10)';x.fill();
}

/* ---------- workspace minimap: the arm's real reachable set ---------- */
function drawWorld(b,d){
 const [x,w,h]=fit($('cvWorld')); x.clearRect(0,0,w,h);
 x.fillStyle='rgba(10,15,10,.78)';x.fillRect(0,0,w,h);
 const cx=w/2,cy=h/2,R=Math.min(w,h)*0.40;
 x.strokeStyle='rgba(0,180,220,.45)';x.lineWidth=1;
 x.beginPath();x.arc(cx,cy,R,0,7);x.stroke();          /* the arm's measured reach */
 /* ee_mm is 3-D millimetres and the circle is the arm's own max horizontal radius, so the
    dot lands where the gripper actually is rather than where a hardcoded reach implied. */
 const ee=(b&&b.ee_mm)||null, reach=(b&&b.reach_mm)||null;
 if(ee&&ee.length>=2&&reach){
  const px=cx+ee[0]/reach*R, py=cy-ee[1]/reach*R;
  x.fillStyle=b.collision?'#ff3c28':'#00ff64';
  x.beginPath();x.arc(px,py,3.5,0,7);x.fill();
  x.strokeStyle='rgba(0,255,100,.55)';x.beginPath();x.moveTo(cx,cy);x.lineTo(px,py);x.stroke();
 }else{x.fillStyle='#4d5a70';x.font='10px monospace';
  x.fillText(ee?'no reach reported':'no arm pose',10,h-10);}
 x.fillStyle='rgba(150,160,180,.9)';x.font='10px ui-monospace,monospace';
 const c=d?d.cells_covered:null;
 x.fillText('cells '+(c===null||c===undefined?EM:c),8,h-8);
}

/* ---------- region map: 17 bars, honest about which are not independent ---------- */
function drawRegions(s){
 const [x,w,h]=fit($('cvBrain')); x.clearRect(0,0,w,h);
 x.fillStyle='rgba(10,10,15,.78)';x.fillRect(0,0,w,h);
 const ks=Object.keys(s).sort(); if(!ks.length){x.fillStyle='#4d5a70';x.font='10px monospace';
  x.fillText('no spike snapshot',10,20);return;}
 const bh=(h-22)/ks.length;
 x.font='9px ui-monospace,monospace';
 ks.forEach((k,i)=>{const v=s[k]||0,y=18+i*bh, flagged=!!PROV['spikes.'+k];
  x.fillStyle='#141c2e';x.fillRect(64,y+1,w-72,bh-2);
  x.fillStyle=flagged?'#4d5a70':(v>.66?'#00c8dc':v>.33?'#32c864':'#ffa028');
  x.fillRect(64,y+1,(w-72)*v,bh-2);
  x.fillStyle=flagged?'#4d5a70':'#a8b4c8';x.fillText(k.slice(0,11),4,y+bh-1);});
}

/* ---------- affect circumplex ---------- */
function drawVA(v,a){
 const [x,w,h]=fit($('cvVA')); x.clearRect(0,0,w,h);
 const cx=w/2,cy=h/2,R=Math.min(w,h)/2-4;
 x.strokeStyle='#1c2740';x.lineWidth=1;
 x.beginPath();x.arc(cx,cy,R,0,7);x.stroke();
 x.beginPath();x.moveTo(cx-R,cy);x.lineTo(cx+R,cy);x.moveTo(cx,cy-R);x.lineTo(cx,cy+R);x.stroke();
 if(v===null||v===undefined||a===null||a===undefined){
  x.fillStyle='#4d5a70';x.font='10px monospace';x.fillText('no affect',cx-22,cy+3);return;}
 x.fillStyle='#00c8dc';
 x.beginPath();x.arc(cx+Math.max(-1,Math.min(1,v))*R,cy-Math.max(-1,Math.min(1,a))*R,4,0,7);x.fill();
}

/* ---------- panels ---------- */
let LAST_AREV=null, LAST_AREV_T=0;
function paint(d){
 const s=d.cognition||{}, sp=d.spikes||{}, b=d.body||{}, dv=d.derived||{},
       g=d.gated||{}, hw=d.hardware||{};
 PROV=d.provenance||{};

 $('bver').textContent='cockpit';
 $('bcirc').textContent=(s.circadian||'--').toUpperCase();
 $('btick').textContent='tick '+((s.wake_ticks||0)+(s.sleep_ticks||0));
 $('bnight').textContent='night '+(s.nights_elapsed??'--');
 $('prog').firstElementChild.style.width=(((s.phase||0)%1)*100).toFixed(1)+'%';

 drawBrain(sp); drawTrace(sp); drawWorld(b,dv); drawRegions(sp);

  /* ARM -- the picture, plus the counter that says whether it is still moving.
     A frozen renderer and a stationary arm are pixel-identical; without the revision
     nothing on this page could tell them apart, which is the failure PROVENANCE exists for. */
  /* armFrame() owns the <img> -- it fetches blobs at 5 Hz and also holds the demand gate
     open. paint() only tracks the REVISION, which is the honesty counter: the picture and
     the counter are deliberately fed by different paths, so a frozen renderer showing a
     plausible still cannot also fake the number that would expose it. */
  const av=d.arm_view||{}, arev=av.arm_view_revision;
  if(!av.arm_view_available){
    $('armImg').style.display='none'; $('armNone').style.display='';
  }
  if(arev!==LAST_AREV){LAST_AREV=arev; LAST_AREV_T=Date.now();}
  const stale=(LAST_AREV_T&&(Date.now()-LAST_AREV_T)>4000);
  $('wArm').innerHTML =
    row('frames rendered',arev===undefined?null:arev)
    +row('picture',av.arm_view_available?(stale?'STALE (not advancing)':'live'):'unavailable')
    +(av.arm_view_reason?`<div class="note">${av.arm_view_reason}</div>`:'')
    +`<div class="note">drawn on the tick thread, on demand. the frame counter is the
       only thing distinguishing a live picture from a frozen one.</div>`;
 $('camtxt').textContent = SPIN?'camera rotating (drag to steer)':'camera held';

 /* MOTOR MIX -- the CPG share, and whether the goal actually overrode habit */
 const cw=dv.cpg_weight;
 $('wMotor').innerHTML =
   (cw===null||cw===undefined?row('CPG share',null):bar('CPG share',cw,'var(--vio)'))
  +row('learned share',(cw===null||cw===undefined)?null:1-cw)
  +row('rhythm spread',dv.cpg_spread_max)
  +row('goal overrode habit',dv.td_flipped===null||dv.td_flipped===undefined?null:(dv.td_flipped?'YES':'no'))
  +`<div class="note">top-down vs habit on the last gated selection.</div>`;

 /* BEHAVIOR & EMOTION */
 $('wBeh').innerHTML =
   row('drive',s.dominant_drive)+row('emotion',s.emotion)
  +bar('curiosity',s.curiosity_mag,'var(--gold)')
  +bar('dopamine',s.da_level,'var(--gold)')
  +row('RPE',s.rpe)
  +`<div class="note">wake drive is two-valued by construction (explore / fidget).</div>`;

 /* CONSCIOUSNESS -- the whole C5 registry, served at d.conclusion by
    brain_telemetry.conclusion_rows() over substrate/conclusion_protocol. THREE THINGS THIS PLATE
    DOES THAT A ROW IN THE FIXED-HEIGHT BOTTOM STRIP STRUCTURALLY COULD NOT:
      - the licensed AND the refused sentence render in the SAME VIEW, wrapped -- refusal 3 as a
        rendering rather than as a sentence in a policy file;
      - the CHARGED reading prints beside the LIVE state value, which is a different quantity;
      - a row with no reading is ABSENT, so an instrument that did not pass prints nothing at all.
    NO tag() on the value row: the provenance strings run to thousands of characters and `.row .v`
    is nowrap, which is exactly how the old rows rendered ZERO visible provenance characters. Here
    the string sits in a collapsed details, wrapped and bounded.
    The three rows' own notes, moved with them out of the ACCESS plate (P2: one place per reading):
    C5 row temporal_binding (T080, 2026-09-16), CHARGED by tools/t080_temporal_binding.json on an aligned scratch stream.
      licensed -- features inside one window are bound into one reported object more often than features across windows;
      refused -- this licenses no claim that there is a felt present, a stream of experience, or a duration lived.
      Its live value is Brain.state()["crossmodal_bound_rate"], the daemon's OWN rate, which its independent percept
      streams hold at the null's level -- a chance rate, never the within-window rate the pin measured.
    C5 row report_order (T081, 2026-09-16), CHARGED by tools/t081_report_order.json on a scratch brain.
      licensed -- perturbing one named intermediate changes the report, so the report is generated from the internal state;
      refused -- this licenses no claim that the report is a first-person account of anything.
      Its live value is Brain.state()["report_order_margin"], the charge margin the campaign's flip dose was read
      against; that dose read 1.0 on every seed BY CONSTRUCTION, so it predicts nothing about the flip, and the
      margin itself can read NEGATIVE (24 of 100 ticks). Not clamped -- the provenance says why.
    C5 row social_cognition (T082, 2026-09-16), CHARGED at 0.4437 by tools/t082_social_cognition.json (head 6d4a65f).
      licensed -- predicting another agent improves when the system models that agent's belief rather than the world alone;
      refused -- this licenses no claim that the system understands another mind, or that it has one.
      Its live value is the LENGTH of Brain.state()["recursive_tom_false_beliefs"], a LIST of the agents whose belief
      diverges from the world -- NOT the charged reading, which is a campaign accuracy difference measured elsewhere. */
 const cc=d.conclusion||[];
 const esc5=t=>String(t).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
 $('wC5').innerHTML = cc.length ? cc.map(e=>{
   const sy=e.symbols||[], lv=e.live||{};
   /* a list-valued state key renders as its LENGTH, which is what the ACCESS row rendered too --
      moving a reading must not quietly change the quantity shown */
   const val=sy.map(k=>Array.isArray(lv[k])?fmt(lv[k].length,0):fmt(lv[k])).join(' / ');
   const pv=sy.map(k=>PROV[k]||'').filter(Boolean).join('   ||   ');
   return `<div class="row"><span class="k">${esc5(e.instrument)}`
    +`${e.task?' '+esc5(e.task):''}</span>`
    +`<span class="v">live ${val} &middot; reading ${fmt(e.reading,4)}</span></div>`
    +`<div class="c5"><i>licensed:</i> ${esc5(e.licensed)}<br>`
    +`<span class="rf"><i>refused:</i> ${esc5(e.refused)}</span>`
    +(pv?`<details><summary>provenance</summary><div>${esc5(pv)}</div></details>`:'')
    +`</div>`;
  }).join('')
  : `<div class="note">no instrument carries a reading yet, and a row without one prints nothing.</div>`;

 /* HARDWARE TARGET -- a plan from the build notes, never telemetry */
 if(hw.joints){
  $('wHw').innerHTML =
    row('target',hw.target_dof+'-DOF · '+hw.reach_mm+'mm · '+hw.payload_kg+'kg')
   +row('we simulate',hw.sim_dof+'-DOF, dynamics '+hw.sim_has_dynamics)
   +row('12V supply',hw.peak_amps_3_stall+'A peak / '+hw.supply_amps+'A '+(hw.supply_ok?'OK':'UNDERSIZED'))
   +hw.joints.map(j=>{const bad=!j.ok,zero=j.required_kgcm===0;
     const col=bad?'var(--red)':(zero?'#4d5a70':'var(--grn)');
     return `<div class="row"><span class="k${zero?' na':''}">${j.joint}`+
      `${j.estimated?' <span class="tag">est</span>':''}</span>`+
      `<span class="v" style="color:${col}">${j.required_kgcm.toFixed(2)} / ${j.rated_kgcm.toFixed(2)}`+
      `${j.margin_x!==null?'  '+j.margin_x.toFixed(2)+'×':''}${bad?' UNDER':''}</span></div>`;}).join('')
   +`<div class="note">${(hw.underpowered||[]).length?'UNDERPOWERED: '+hw.underpowered.join(', ')+'. ':''}`
   +`Rated = stall/4 per the build notes. Static gravity, horizontal pose, payload only. Nothing measured.</div>`
   /* THE MODEL'S OWN ANSWER, next to the formula's. The formula weighs the arm at 0 kg and uses a
      hand-written lever; the MuJoCo body has real link mass and real geometry, and it is HARSHER.
      Shown together on purpose -- correcting the formula silently would hide that it was optimistic. */
   +(hw.measured_torque?(
      row('from the model',`${hw.poses_within_rating} of ${hw.poses_swept} swept poses hold `
        +`${hw.measured_torque.payload_kg}kg`)
     +hw.measured_torque.joints.filter(j=>j.required_kgcm>0.001).map(j=>{const bad=!j.ok;
       return `<div class="row"><span class="k">${j.joint} <span class="tag">model</span></span>`+
        `<span class="v" style="color:${bad?'var(--red)':'var(--grn)'}">`+
        `${j.required_kgcm.toFixed(2)} / ${j.rated_kgcm.toFixed(2)}`+
        `${j.margin_x!==null?'  '+j.margin_x.toFixed(2)+'×':''}${bad?' UNDER':''}</span></div>`;}).join('')
     +`<div class="note">MuJoCo gravity load over a ${Math.round(Math.sqrt(hw.poses_swept))}×`
      +`${Math.round(Math.sqrt(hw.poses_swept))} shoulder×elbow sweep, arm mass included. `
      +`${(hw.underpowered_measured||[]).length?'UNDERPOWERED: '+hw.underpowered_measured.join(', ')+'. ':''}`
      +`Still a target, not a sensor: nothing here is built.</div>`
   ):row('from the model','unavailable - no mujoco, formula only'))
   /* HOW TO MAKE IT BUILDABLE. The plan lists five ways out; which COMBINATIONS suffice is
      arithmetic, and the headline is that the original spec is not among them. */
   +(hw.buildable_recipes&&hw.buildable_recipes.length?(
      row('buildable',`${hw.buildable_count} of ${hw.buildable_tried} configurations, `
        +`none keeping 300mm AND 0.5kg`)
     +hw.buildable_recipes.slice(0,3).map(c=>
       `<div class="row"><span class="k">${c.payload_kg}kg @ ${c.reach_mm}mm</span>`+
       `<span class="v">${c.extra_reduction}:1${c.counterbalance?' +cbal':''}`+
       `${c.parallel_shoulder>1?' +2nd servo':''}  ${c.tightest_margin_x}× ${c.tightest_joint}</span></div>`
      ).join('')
     +`<div class="note">Extra reduction appears in every feasible configuration; without it nothing `
      +`closes. A second shoulder servo does not help the elbow, which is what binds.</div>`
   ):'');
 }

 /* NEUROMODULATORS */
 $('wNm').innerHTML =
   bar('dopamine',s.da_level,'var(--gold)')+bar('serotonin',s.neuromod_serotonin,'var(--grn)')
  +bar('noradrenaline',s.neuromod_noradrenaline,cLoad(s.neuromod_noradrenaline||0))
  +bar('acetylcholine',s.neuromod_acetylcholine,'var(--ice)')
  +bar('cortisol',s.cortisol,cLoad(s.cortisol||0))
  +row('astrocyte gain',s.neuromod_astrocyte)   /* per-domain, >1.0 -- a gain, not a level */
  +`<div class="note">astrocyte is a per-domain gain, not a [0,1] level.</div>`;

 /* WORLD MODEL + the not-computed readouts, stated as such */
 $('wWm').innerHTML =
   row('pc surprise',s.pc_surprise)+row('predict acc',s.predict_accuracy)
   /* rank 2: the floors predict acc must clear on the SAME held-out slice (persistence = "the
      same category as the last one"); on the brain's own bursty percept stream an accuracy
      under its persistence floor learned nothing the stream's inertia did not supply */
  +row('predict floor: persistence',(s.neurogenesis||{}).baseline_persistence)
  +row('predict floor: majority',(s.neurogenesis||{}).baseline_majority)
  +row('place error',dv.place_error)+row('attended modality',s.attended_modality)
  +Object.entries(g).map(([k,o])=>
    `<div class="row"><span class="k">${k}</span><span class="v na">${fmt(o.value)}`+
    `<span class="tag"> ${o.computed?'not applied':'NOT COMPUTED'}</span></span></div>`).join('');

 /* ---- bottom bar ---- */
 $('bCov').innerHTML=big(dv.cells_covered,0,undefined,'cells_covered')+sub('workspace cells reached')+row('new cells / cov tick',dv.coverage_rate,3,'coverage_rate')+row('moved / cov tick',dv.coverage_motion,3,'coverage_motion')+row('window',dv.coverage_window,0,'coverage_window')+row('scale',dv.coverage_scale_cells,0,'coverage_scale_cells')+row('restored',dv.coverage_restored,0,'coverage_restored')
   +sub('the one signal that says improving');
 $('bReach').innerHTML=big(dv.distance_from_origin,2)+sub('map distance from origin')
   +row('displacement',dv.displacement)
   +`<div class="note">no dt in the sim: not a velocity.</div>`;
 const gt=dv.ground_ticks;
 $('bStab').innerHTML=big(dv.recoveries,0,gt?'var(--org)':'var(--grn)')+sub('postural recoveries')
   +row('ground ticks',gt)+row('collision',b.collision||'clear')
   +row('E-STOP',b.stopped?('STOPPED'+(b.latch_source==='restored'?' (restored at boot from '+String(b.latch_restored_from||'?').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))+'; HOME resets)':'')):'clear');
 $('bVA').innerHTML=row('emotion',s.emotion)+row('V / A',
   (dv.valence===undefined?EM:fmt(dv.valence,2))+' / '+(dv.arousal===undefined?EM:fmt(dv.arousal,2)));
 drawVA(dv.valence,dv.arousal);
 $('bAcc').innerHTML=big(s.access_index,3)+sub('global access index')
   +row('ignitions',dv.ignitions,0)+row('percept holds bus',dv.ws_percept_ignited,0)
   +row('PCI',s.pci,3,'pci')+row('integration',s.integration_competence,3,'integration_competence')
   /* The three C5 readings LEFT this plate at T090 (P2: one place per reading). They print on the
      CONSCIOUSNESS plate in the widgets column, where the sentences each one licenses and refuses
      render beside it and the provenance wraps instead of running off a nowrap value span. */
   +`<div class="note">ignitions = broadcast changed hands; percept holds bus = this tick's percept seized the slot (1/0); the others can hold high on one content.</div>`;
 $('bLife').innerHTML=big(s.nights_elapsed,0,undefined,'nights_elapsed')+sub('nights slept')
   +row('units admitted',s.grown_units,0,'grown_units')+row('reasoning acc',s.reasoning_accuracy)
   +row('reliability',s.reliability,undefined,'reliability');
 $('bLake').innerHTML=big(dv.lake_events,0)+sub('lake working-memory events')
   +row('reachable',(dv.coverage_reachable===undefined?null:dv.coverage_reachable+' / '+dv.coverage_total),undefined,'coverage_reachable')
   +row('dormant',dv.coverage_dormant,undefined,'coverage_dormant')
   +`<div class="note">reachable, not proven to charge behaviour.</div>`;
 /* DEVICE ROUTING -- substrate/device.decisions(), the choices that ASKED the measured router.
    The headline is the COUNT of workloads that asked, not a device: a device drawn large would
    read as "this is where the brain runs", and the tag exists because it is not. `why` is the
    router's own recorded reason, rendered verbatim rather than re-derived here -- a page that
    recomputed it could disagree with the decision it is describing. Empty is drawn as a sentence,
    never as a zero device: "nothing asked" and "everything chose CPU" are different facts. */
 const rt=d.device||{}, rk=Object.keys(rt).sort();
 $('bDev').innerHTML=big(rk.length,0,undefined,'device_routing')
   +sub('workloads routed by measurement')
   +(rk.length
      ? rk.map(k=>row(k,rt[k].device+'  n='+fmt(rt[k].n,0))
          +`<div class="note">${rt[k].why}</div>`).join('')
      : `<div class="note">nothing has asked the router in this process yet.</div>`)
   +`<div class="note">only work that calls device.select() appears here; _perception_device is
      chosen outside it, so a cpu row is not a claim that the GPU is idle.</div>`;
}

/* ---------- chat: post a ticket, poll for the reply ---------- */
function say(cls,who,text,route){
 const d=document.createElement('div'); d.className='m '+cls;
 d.innerHTML=`<b>${who}</b> ${text.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}`
  +(route?`<div class="rt">${route}</div>`:'');
 $('log').appendChild(d); $('log').scrollTop=$('log').scrollHeight; return d;
}
async function send(){
 const t=$('ci').value.trim(); if(!t)return;
 $('ci').value=''; $('cs').disabled=true;
 say('u','you',t);
 const wait=say('b','brain','thinking...');
 try{
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:t})});
  const {id,error}=await r.json();
  if(error||!id){wait.innerHTML='<b>brain</b> <span class="er">'+(error||'not queued')+'</span>';return;}
  for(let i=0;i<600;i++){
   await new Promise(r=>setTimeout(r,500));
   const q=await (await fetch('/api/chat?id='+id)).json();
   if(q.pending)continue;
   if(q.error){wait.innerHTML='<b>brain</b> <span class="er">'+q.error+'</span>';return;}
   wait.innerHTML=`<b>brain</b> ${(q.reply||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}`
     +(q.route?`<div class="rt">${q.route}</div>`:'');
   $('log').scrollTop=$('log').scrollHeight; return;
  }
  wait.innerHTML='<b>brain</b> <span class="er">no reply within 5 min &mdash; is the daemon pumping turns?</span>';
 }catch(e){wait.innerHTML='<b>brain</b> <span class="er">'+e+'</span>';}
 finally{$('cs').disabled=false;$('ci').focus();}
}
$('cs').onclick=send;
$('ci').addEventListener('keydown',e=>{if(e.key==='Enter')send();});

/* camera drag */
const bc=$('brain3d');
bc.addEventListener('mousedown',e=>{DRAG=[e.clientX,e.clientY,AZ,EL];SPIN=false;});
window.addEventListener('mouseup',()=>{DRAG=null;});
window.addEventListener('mousemove',e=>{if(!DRAG)return;
 AZ=DRAG[2]+(e.clientX-DRAG[0])*0.01; EL=Math.max(-1.4,Math.min(1.4,DRAG[3]-(e.clientY-DRAG[1])*0.01));});
bc.addEventListener('dblclick',()=>{SPIN=!SPIN;});

/* ---- THE MUJOCO STAGE IS INTERACTIVE -------------------------------------------------
   Drag orbits the camera, wheel dollies it. Both go to /api/look, which steers the
   VIEWPOINT only -- there is no path from this handler to a motor command. Arm motion is
   the separate, allow-listed /api/act below. */
let CAM={az:90,el:-20,d:1.6}, CDRAG=null, CAM_PEND=false;
const stage=$('armStage');
function pushCam(){
  if(CAM_PEND) return;                      /* one flight at a time: a drag emits far more
                                               events than the tick thread can render */
  CAM_PEND=true;
  fetch('/api/look?az='+CAM.az.toFixed(1)+'&el='+CAM.el.toFixed(1)+'&d='+CAM.d.toFixed(3))
    .then(r=>r.json()).then(c=>{if(c&&c.azimuth!==undefined){CAM.az=c.azimuth;CAM.el=c.elevation;CAM.d=c.distance;}})
    .catch(()=>{}).then(()=>{CAM_PEND=false;});
}
stage.addEventListener('mousedown',e=>{CDRAG=[e.clientX,e.clientY,CAM.az,CAM.el];
  stage.classList.add('drag'); e.preventDefault();});
window.addEventListener('mouseup',()=>{if(CDRAG){CDRAG=null;stage.classList.remove('drag');}});
window.addEventListener('mousemove',e=>{if(!CDRAG)return;
  CAM.az=CDRAG[2]-(e.clientX-CDRAG[0])*0.4;
  CAM.el=Math.max(-89,Math.min(89,CDRAG[3]+(e.clientY-CDRAG[1])*0.3));
  pushCam();});
stage.addEventListener('wheel',e=>{e.preventDefault();
  CAM.d=Math.max(0.15,Math.min(12,CAM.d*(e.deltaY>0?1.1:0.9))); pushCam();},{passive:false});

/* ---- ACTUATION ----------------------------------------------------------------------
   Owner-authorised 2026-08-03. The button only POSTs a string; every command is checked
   against the wire allow-list server-side and executed on the tick thread, never here.
   A latched arm accepts the ticket and does not move, and says so -- "ran" and "was
   swallowed" must not look the same. */
document.querySelectorAll('#armCtl button').forEach(function(btn){
  btn.addEventListener('click',function(){
    const cmd=btn.getAttribute('data-cmd'), msg=$('actMsg');
    msg.textContent='...';
    fetch('/api/act',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({command:cmd})})
      .then(r=>r.json().then(j=>({ok:r.ok,j:j})))
      .then(function(res){
        if(!res.ok){msg.textContent='refused: '+(res.j.error||'?'); return;}
        const id=res.j.id;
        setTimeout(function(){
          fetch('/api/act?id='+id).then(r=>r.json()).then(function(rec){
            if(rec.pending){msg.textContent='queued';}
            else if(rec.stopped){msg.textContent='SWALLOWED (arm latched)';}
            else{msg.textContent=cmd.split('(')[0].toLowerCase()+' ok';}
          }).catch(()=>{msg.textContent='?';});
        },400);
      }).catch(function(){msg.textContent='send failed';});
  });
});

async function loop(){
 try{
  if(!GEOM){try{GEOM=await (await fetch('/api/geom')).json();}catch(e){}}
  const d=await (await fetch('/api/all')).json();
  $('berr').textContent=''; paint(d);
 }catch(e){$('berr').textContent='disconnected';}
}
function spin(){ if(SPIN&&!DRAG)AZ+=0.0016; requestAnimationFrame(spin); }

/* The arm gets its own faster loop. /api/all runs at 1 Hz, which is fine for numbers and
   far too slow for a moving arm; and because rendering is DEMAND-GATED (arm_view.request()
   with a TTL), a slow poll does not merely look choppy -- it lets the gate lapse and the
   tick thread stops drawing at all. Fetching as a blob and swapping object URLs avoids
   both the browser cache and a flash of empty <img> between frames. */
let ARM_URL=null, ARM_BUSY=false;
function armFrame(){
  if(ARM_BUSY) return;                       /* never queue behind a slow render */
  ARM_BUSY=true;
  fetch('/api/arm.png?t='+Date.now())
    .then(function(r){ return r.ok?r.blob():null; })
    .then(function(b){
      if(b){
        const u=URL.createObjectURL(b), img=$('armImg');
        img.src=u; img.style.display=''; $('armNone').style.display='none';
        if(ARM_URL) URL.revokeObjectURL(ARM_URL);   /* or the tab leaks one blob per frame */
        ARM_URL=u;
      }
    })
    .catch(function(){})
    .then(function(){ ARM_BUSY=false; });
}
loop(); setInterval(loop,1000); spin();
armFrame(); setInterval(armFrame,200);
</script>
"""


def _http_get(brain, path):
    """Drive one GET through the REAL handler without a socket. Returns (code, body bytes).

    Subclasses the handler rather than imitating it, so the code under test is the code that
    serves browsers -- a hand-rolled stand-in would be free to disagree with it. Only the
    socket-facing methods are replaced; `_send` and `do_GET` are the originals.
    """
    captured = {}

    def _view():
        try:
            return brain.brain.arm.arm.view
        except Exception:
            return None

    base = _handler(lambda: _snapshot(brain), Turns(), _view)

    class _Offline(base):
        def __init__(self):  # deliberately NOT calling BaseHTTPRequestHandler.__init__
            self.path = path

        def send_response(self, code, *_a):
            captured["code"] = code

        def send_header(self, *_a):
            pass

        def end_headers(self):
            pass

        @property
        def wfile(self):
            class _W:
                @staticmethod
                def write(data):
                    captured["body"] = data

            return _W()

    _Offline().do_GET()
    return captured.get("code"), captured.get("body", b"")


def main():
    """Self-check: the page is served, the contract holds, and a chat turn round-trips via the queue."""
    import urllib.request

    from snn_data_lake.cortex.brain import make_brain

    # NORD_TORCH=0 dropped 2026-07-29 with the zero-dependency rule: this self-check now builds the
    # brain in the configuration users actually get. NORD_PERSIST stays -- it guards the durable lake,
    # which is a real invariant and has nothing to do with dependencies.
    #
    # SCOPED 2026-08-04 (specs/010, C34). This was a bare `os.environ.setdefault("NORD_PERSIST","0")`
    # that was never restored, so running this self-check disabled durability for EVERY consumer in
    # the process afterwards -- the same unscoped-environment defect `home.main()`'s `finally` block
    # had already been fixed for, one module over ("restored in the finally, NOT popped").
    # `tests/test_selfchecks.py` runs every module's `__main__` in ONE interpreter, which is exactly
    # the shape that turns a local convenience into a global one.
    #
    # `persist=False` is now passed EXPLICITLY rather than leaned on via the environment: it states
    # the invariant this self-check actually wants ("never write the durable lake") at the call
    # site, and it no longer depends on a global that anything else can observe.
    # ---- Acts._done is BOUNDED (added 2026-08-05) --------------------------------------------
    # The eviction landed the day before with no criterion of any kind, in the always-on daemon,
    # where "grows forever" is the whole failure mode. P8: every criterion owes a reachable red.
    # RED, CONSTRUCTED AND RUN: delete the three eviction lines in `Acts.pump` and this fires with
    # len(_done)=200; shrink the retention and it fires the other way. It runs BEFORE the brain is
    # built so a stub arm is enough -- pump() only needs `brain.brain.arm.apply_observed` (T037:
    # the tapped passage the cockpit route now uses; the stub mirrors the real interface).
    # distinct-from: brainstem/so101_arm.py -- the REAL apply; this is the self-check's
    #   inert stand-in for the eviction-bound test, motion-free by construction. Read.
    class _StubArm:
        @staticmethod
        def apply(cmd):
            return {"stopped": False, "theta_deg": []}

    class _StubBrain:
        class brain:  # noqa: N801 -- mirrors the real attribute chain pump() walks
            class arm:  # noqa: N801
                arm = _StubArm

                # distinct-from: brainstem/autonomous_loop.py -- the REAL apply_observed (the
                #   tapped passage); this mirrors its signature for the eviction test only. Read.
                @staticmethod
                def apply_observed(
                    cmd, deliberate=False
                ):  # T037a: the route passes deliberate=True
                    return _StubArm.apply(cmd)

    _bounded = Acts()
    for _ in range(_REPLY_KEEP * 3 + 8):
        _bounded.submit("HOLD")
    _bounded.pump(_StubBrain, budget=_REPLY_KEEP * 3 + 8)
    assert _bounded.applied == _REPLY_KEEP * 3 + 8, (
        f"the pump must have run every queued command ({_bounded.applied})"
    )
    assert len(_bounded._done) == _REPLY_KEEP and len(_bounded._order) == _REPLY_KEEP, (
        "Acts._done must be BOUNDED at _REPLY_KEEP like Turns -- an always-on daemon that keeps one"
        f" record per arm command forever leaks: got {len(_bounded._done)}/{len(_bounded._order)}"
    )
    assert _bounded.result(_REPLY_KEEP * 3 + 8) is not None, (
        "the NEWEST reply must survive eviction"
    )
    assert _bounded.result(1) is None, "the OLDEST reply must be the one evicted, not the newest"

    _prior_persist = os.environ.get("NORD_PERSIST")
    os.environ["NORD_PERSIST"] = "0"
    brain = make_brain(seed=0, persist=False)
    for _ in range(8):
        brain.tick()

    # The arm picture, end to end. Rendering happens on the TICK thread, and this self-check
    # IS the tick thread, so the sequence a browser produces is reproduced deliberately: the
    # first request only registers demand and correctly gets a 503 with a reason, the next tick
    # draws, and the request after that carries actual PNG bytes.
    _arm = brain.brain.arm.arm
    _first_code, _first_body = _http_get(brain, "/api/arm.png")
    assert _first_code == 503, f"an undrawn view must not claim success (got {_first_code})"
    assert json.loads(_first_body)["error"], "a 503 must carry a reason, not an empty body"

    _drew = _arm.view.render_if_wanted(_arm.model, _arm.data)
    # `is not None`, NOT truthiness -- the fourth and last copy of the shape fixed in
    # arm_view.main() at e9e1832b. render_if_wanted returns None when it did not render and the
    # revision number when it did, and revision 0 is FALSY: a renderer that draws but stops
    # advancing its counter skips this block, so the three asserts below -- including the
    # end-to-end one, that the panel's counter agrees with the renderer -- never run, and this
    # self-check exits 0. Production cannot reach 0 today (the counter post-increments), so this
    # changes no behaviour; it makes the asserts reachable under the defect they exist to catch.
    if _drew is not None:  # environments without a GL context keep the honest 503 path above
        _code, _png = _http_get(brain, "/api/arm.png")
        assert _code == 200, f"a drawn frame was not served: {_code}"
        assert _png[:8] == b"\x89PNG\r\n\x1a\n", "the route did not return PNG bytes"
        assert snapshot(brain)["arm_view"]["arm_view_revision"] == _drew, (
            "the panel's frame counter disagrees with the renderer"
        )
    _arm.view.close()

    # the no-orphan-panel gate, now covering every section the page reads
    missing = verify_contract(snapshot(brain))
    assert not missing, f"cockpit reads keys the brain cannot supply: {missing}"

    # the page must reference only keys the contract knows about -- a panel naming a key that no
    # longer exists is exactly the silent blank this gate is for
    for key in (
        "cells_covered",
        "cpg_weight",
        "td_flipped",
        "coverage_reachable",
        "provenance",
        "device_routing",
    ):
        assert key in COCKPIT_HTML, f"cockpit page lost its reference to {key}"
    assert "make_brain" not in COCKPIT_HTML, "the page must never imply it constructs a brain"

    # EVERY PANEL THE SCRIPT WRITES TO MUST EXIST, derived rather than listed. The loop above is a
    # SUBSTRING test, which cannot tell "the plate is on the page" from "the plate's name appears
    # somewhere on the page" -- deleting the <div id="bDev"> plate while `$('bDev')` stays in the
    # script passed it. That is not cosmetic: `paint()` runs inside `loop()`'s try/catch, so one
    # null target throws, the catch writes 'disconnected', and EVERY panel silently stops
    # refreshing while the error strip blames the connection. `bDev` was listed above and is not
    # any more -- this subsumes it and covers the other 32 the list never named.
    _writes = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", COCKPIT_HTML))
    _ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', COCKPIT_HTML))
    assert not (_writes - _ids), (
        f"the script writes to element(s) the page does not define: {sorted(_writes - _ids)}. "
        "paint() would throw on a null target and the whole dashboard would freeze reading "
        "'disconnected'"
    )

    # ---- DEVICE ROUTING: the page shows what the ROUTER decided, and FOLLOWS it ----------------
    # P3 disposition for `substrate/device.decisions()` (2026-08-06). RETIRE was refused on
    # evidence rather than taste: `cortex/neurogenesis_engine.main()`'s claim 6 already reads it
    # and asserts the refit is WIDTH-routed, so deleting the function deletes a criterion that HAS
    # a reachable red -- and P8 deletes criteria with no red, never ones with one. What was
    # genuinely missing was the REPORT half, and this plate is it. A report is not a P6
    # faculty-to-faculty read: no faculty's own output changes because a page renders this, and it
    # raises no P6 count.
    #
    # THE CRITERION IS NOT "THE SECTION EXISTS". A hardcoded 'cpu', a value re-derived from
    # `cuda_available()`, and a section captured once at import all render identically to a live
    # one right up until the decision moves. So the decision is MOVED and the payload is compared
    # against `decisions()` -- never against itself, which is the shape that cannot fail.
    #
    # CARD-INDEPENDENT BY CONSTRUCTION (P5: CI has no GPU), and this is also what nearly made it
    # vacuous. With no CUDA present `select(w, n=1)` and `select(w, n=None)` take the SAME branch
    # and record the SAME reason, so a probe built on width alone would compare two identical
    # values on CI and pass while proving nothing. The override path differs on every machine, so
    # the second probe pins `explicit=` instead.
    from snn_data_lake.substrate.device import cuda_available, select

    def _served():
        _c, _b = _http_get(brain, "/api/all")
        assert _c == 200, f"/api/all did not serve (got {_c})"
        return json.loads(_b)

    _pay = _served()
    assert "device" in _pay, (
        "the served payload carries no `device` section -- the plate would draw an empty state "
        "indistinguishable from 'nothing has asked the router'"
    )
    assert _pay.get("provenance", {}).get("device_routing"), (
        "the routing section reached the page WITHOUT its provenance tag. An untagged router log "
        "reads as a census of where the brain runs, and measurement says it is not one: one "
        "workload asks the router while _perception_device is set to 'cuda' outside it"
    )
    # PRESENT IS NOT TRUE. Checking only that the tag is a non-empty string accepts its own
    # inversion: setting it to "a complete census of every device this brain uses" -- the exact
    # claim the measurement above was taken to REFUSE -- passed. P4c puts the whole weight of this
    # panel's honesty on the provenance column, so a truthiness test is the wrong instrument. Two
    # things are pinned instead, and neither judges English:
    #   (1) the tag the page SERVES is the tag reviewed here, so nothing in the serving path can
    #       swap it;
    #   (2) `_perception_device` -- the thing the tag and the plate note both CITE as the choice
    #       made outside this router -- actually exists on the brain. If it is renamed or removed,
    #       the note names nothing and the tag becomes unfalsifiable rather than merely wrong.
    # What stays review-enforced, stated so it is not mistaken for covered: no assertion can tell
    # whether the SENTENCE in `_DEVICE_PROV` is honest. It pins the constant, not its meaning.
    assert _pay["provenance"]["device_routing"] == _DEVICE_PROV, (
        f"the served provenance tag is not the one this module declares: "
        f"{_pay['provenance']['device_routing']!r} != {_DEVICE_PROV!r}"
    )
    assert getattr(brain, "_perception_device", None) is not None, (
        "the plate note and the tag both cite `_perception_device` as the device choice made "
        "outside this router. That attribute is gone, so the page now cites nothing and the "
        "'not a GPU census' claim can no longer be checked against anything"
    )

    # EVERY FIELD IS READ WITH `.get()`, and that is not defensive style -- it is what keeps the
    # diagnostic. Read with `[...]`, a payload MISSING a field dies of KeyError before its
    # assertion can evaluate, so the mutant that drops width and reason (see `availability` below)
    # fails the self-check with a bare `KeyError: 'n'` instead of the sentence naming the defect.
    # Measured: that mutant scored BROKEN rather than RED in the no-card configuration until this
    # changed. A criterion whose negative is a crash is one this project has recorded being fooled
    # by before.
    _dev_a = select("cockpit_probe", n=1)
    _seen_a = _served()["device"].get("cockpit_probe")
    assert _seen_a is not None, "a decision select() recorded never reached the served payload"
    assert _seen_a.get("device") == _dev_a, (
        f"the page shows device {_seen_a.get('device')!r} where the router decided {_dev_a!r}"
    )
    assert _seen_a.get("n") == 1 and _seen_a.get("why") == decisions()["cockpit_probe"]["why"], (
        f"the served row is not the router's own record: {_seen_a}. A panel that re-derives the "
        "device from cuda_available() carries no width and no reason, and looks right until asked "
        "why -- which is the only question this section exists to answer"
    )

    # MOVE THE DECISION. Pinning via `explicit` changes which RULE fires, so the reason differs on
    # a card and without one; the width differs either way.
    _dev_b = select("cockpit_probe", n=7, explicit="cpu")
    _seen_b = _served()["device"]["cockpit_probe"]
    assert _seen_b.get("why") != _seen_a.get("why") and _seen_b.get("n") == 7, (
        f"the served routing did not follow the router when the decision changed: {_seen_a} -> "
        f"{_seen_b}. A section captured once renders exactly like a live one until this moves"
    )
    assert _seen_b.get("device") == _dev_b, (
        f"the page shows {_seen_b.get('device')!r} where the router decided {_dev_b!r}"
    )
    if cuda_available():
        # Only a machine with a card can make the DEVICE COLUMN ITSELF differ -- without one every
        # answer is 'cpu' and there is no flip to observe. Named rather than silently skipped, and
        # the two asserts above are what carry this criterion on a cardless CI runner.
        _dev_c = select("cockpit_probe", n=None)
        _seen_c = _served()["device"]["cockpit_probe"]
        assert _dev_c == "cuda" and _seen_c.get("device") == "cuda", (
            f"unbounded work must reach the card and the page must say so: router {_dev_c!r}, "
            f"page {_seen_c.get('device')!r}"
        )

    turns = Turns()
    # port=0 -> OS-assigned. A hardcoded port makes a self-check fail for reasons unrelated to what it
    # tests (anything else holding the port, including a second copy of itself), and a gate that cries
    # wolf gets ignored.
    srv = serve(brain, port=0, turns=turns, block=False)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/", timeout=10) as r:
            page = r.read().decode()
        assert "SNDL cockpit" in page, "index did not serve the cockpit page"

        with urllib.request.urlopen(f"{base}/api/geom", timeout=20) as r:
            geom = json.loads(r.read().decode())
        assert geom["modules"] and geom["edges"], "geometry endpoint returned an empty layout"

        # a turn must NOT be answered by the request thread: nothing has pumped yet
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps({"text": "what is your goal"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            tid = json.loads(r.read().decode())["id"]
        with urllib.request.urlopen(f"{base}/api/chat?id={tid}", timeout=10) as r:
            assert json.loads(r.read().decode()).get("pending"), (
                "a turn was answered without the owner loop pumping it -- that is the data race "
                "this queue exists to prevent"
            )

        # ...and once the owner thread pumps, the reply lands
        assert turns.pump(brain, budget=4) == 1, "pump did not drain exactly the one queued turn"
        with urllib.request.urlopen(f"{base}/api/chat?id={tid}", timeout=10) as r:
            got = json.loads(r.read().decode())
        assert not got.get("pending"), "reply never surfaced after pump"
        assert got.get("reply"), f"empty reply: {got}"
    finally:
        srv.shutdown()
        srv.server_close()
        brain.close()
        # RESTORE, not pop -- popping would drop a value the CALLER set, which is the precise
        # mistake `home.main()`'s own `finally` records having made ("restored in the finally, NOT
        # popped"). Cited by SYMBOL: the old `home.py:311` resolved onto an unrelated docstring
        # line inside `save_selfbuild`, three hundred lines from the block it meant.
        if _prior_persist is None:
            os.environ.pop("NORD_PERSIST", None)
        else:
            os.environ["NORD_PERSIST"] = _prior_persist

    print(
        f"cockpit OK: page served, {len(geom['modules'])} modules / {len(geom['edges'])} tracts, "
        f"contract clean, chat queued-then-pumped (reply {len(got['reply'])} chars, "
        f"route={got.get('route', '')[:28]!r})"
    )


if __name__ == "__main__":
    main()
