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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from snn_data_lake.substrate.brain_telemetry import snapshot, verify_contract

_MAX_CHAT_CHARS = 2000  # matches the daemon's own truncation of an ingested turn
_REPLY_KEEP = 64  # completed replies retained for polling before the oldest is dropped


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
                out = brain.respond(text)
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


def _handler(get_snap, turns):
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
                if path == "/api/chat":
                    tid = dict(p.split("=", 1) for p in qs.split("&") if "=" in p).get("id", "")
                    rec = turns.result(tid) if tid.isdigit() else None
                    return self._send(200, json.dumps(rec or {"pending": True}))
                snap = get_snap()
                if path == "/api/state":
                    return self._send(200, json.dumps(snap["cognition"]))
                if path == "/api/spikes":
                    return self._send(200, json.dumps(snap["spikes"]))
                if path == "/api/all":
                    return self._send(200, json.dumps(snap))
                return self._send(404, json.dumps({"error": "not found"}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        def do_POST(self):
            if self.path.split("?")[0] != "/api/chat":
                return self._send(404, json.dumps({"error": "not found"}))
            try:
                n = int(self.headers.get("Content-Length") or 0)
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


def serve(brain, port=8137, turns=None, block=True):
    """Serve the cockpit over the live brain. Does NOT construct a brain -- Policy 4: exactly one.

    Binds loopback only. This is a window onto a local process, not a network service, and `/api/chat`
    accepts writes -- so it must not become reachable off the machine by default.
    """
    turns = Turns() if turns is None else turns
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), _handler(lambda: snapshot(brain), turns))
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
#brand .sep{color:#2a3purple}
#prog{flex:1;height:2px;background:#16203a;border-radius:2px;overflow:hidden}
#prog i{display:block;height:100%;background:var(--cy);width:0}
#scene{grid-area:scene;position:relative;overflow:hidden}
#brain3d{position:absolute;inset:0;width:100%;height:100%;cursor:grab}
#brain3d:active{cursor:grabbing}
#cam{position:absolute;left:12px;top:10px;font-size:11px;color:var(--grey);z-index:3}
#cam b{color:var(--cy)}
#trace{position:absolute;left:12px;right:12px;bottom:12px;height:86px;z-index:2}
.mini{position:absolute;bottom:110px;width:196px;height:196px;z-index:2}
#mmWorld{left:12px}#mmBrain{left:216px}
.mini canvas{width:100%;height:100%;display:block;border-radius:6px}
.mini .lbl{position:absolute;left:8px;top:6px;font-size:11px;letter-spacing:.10em;z-index:1}
#widgets{grid-area:widgets;display:flex;flex-direction:column;gap:6px;overflow:hidden}
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
#bottom{grid-area:bottom;display:grid;grid-template-columns:repeat(7,1fr);gap:6px}
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
   <canvas id="brain3d"></canvas>
   <div id="cam"><b>ANATOMY</b> &middot; <span id="camtxt">camera rotating</span><br>
     <span style="color:#4d5a70">positions are hand-authored anatomy, not measured.
     tracts are the real connectome. colour+size = live rate.</span></div>
   <div class="mini plate" id="mmWorld"><div class="lbl" style="color:var(--cy)">WORKSPACE</div>
     <canvas id="cvWorld"></canvas></div>
   <div class="mini plate" id="mmBrain"><div class="lbl" style="color:var(--vio)">REGIONS</div>
     <canvas id="cvBrain"></canvas></div>
   <canvas id="trace" class="plate"></canvas>
 </div>

 <div id="widgets">
   <div class="plate" style="flex:3"><h2>MOTOR MIX</h2><div id="wMotor"></div></div>
   <div class="plate" style="flex:4"><h2>BEHAVIOR &amp; EMOTION</h2><div id="wBeh"></div></div>
   <div class="plate" style="flex:5"><h2>HARDWARE TARGET &mdash; NOT BUILT</h2><div id="wHw"></div></div>
   <div class="plate" style="flex:3"><h2>NEUROMODULATORS</h2><div id="wNm"></div></div>
   <div class="plate" style="flex:3"><h2>WORLD MODEL</h2><div id="wWm"></div></div>
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
function big(v,d,col){return `<div class="big"${col?` style="color:${col}"`:''}>${fmt(v,d)}</div>`;}
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
 x.beginPath();x.arc(cx,cy,R,0,7);x.stroke();          /* reach = 2.0 sim units */
 const ee=(b&&b.ee)||null;
 if(ee&&ee.length>=2){
  const px=cx+ee[0]/2*R, py=cy-ee[1]/2*R;
  x.fillStyle=b.collision?'#ff3c28':'#00ff64';
  x.beginPath();x.arc(px,py,3.5,0,7);x.fill();
  x.strokeStyle='rgba(0,255,100,.55)';x.beginPath();x.moveTo(cx,cy);x.lineTo(px,py);x.stroke();
 }else{x.fillStyle='#4d5a70';x.font='10px monospace';x.fillText('no arm pose',10,h-10);}
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
   +`Rated = stall/4 per the build notes. Static gravity, horizontal pose, payload only. Nothing measured.</div>`;
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
  +row('place error',dv.place_error)+row('attended modality',s.attended_modality)
  +Object.entries(g).map(([k,o])=>
    `<div class="row"><span class="k">${k}</span><span class="v na">${fmt(o.value)}`+
    `<span class="tag"> ${o.computed?'not applied':'NOT COMPUTED'}</span></span></div>`).join('');

 /* ---- bottom bar ---- */
 $('bCov').innerHTML=big(dv.cells_covered,0)+sub('workspace cells reached')
   +sub('the one signal that says improving');
 $('bReach').innerHTML=big(dv.distance_from_origin,2)+sub('map distance from origin')
   +row('displacement',dv.displacement)
   +`<div class="note">no dt in the sim: not a velocity.</div>`;
 const gt=dv.ground_ticks;
 $('bStab').innerHTML=big(dv.recoveries,0,gt?'var(--org)':'var(--grn)')+sub('postural recoveries')
   +row('ground ticks',gt)+row('collision',b.collision||'clear')
   +row('E-STOP',b.stopped?'STOPPED':'clear');
 $('bVA').innerHTML=row('emotion',s.emotion)+row('V / A',
   (dv.valence===undefined?EM:fmt(dv.valence,2))+' / '+(dv.arousal===undefined?EM:fmt(dv.arousal,2)));
 drawVA(dv.valence,dv.arousal);
 $('bAcc').innerHTML=big(s.access_index,3)+sub('global access index')
   +row('PCI',s.pci,3,'pci')+row('integration',s.integration_competence,3,'integration_competence');
 $('bLife').innerHTML=big(s.nights_elapsed,0)+sub('nights slept')
   +row('grown units',s.grown_units)+row('reasoning acc',s.reasoning_accuracy)
   +row('reliability',s.reliability);
 $('bLake').innerHTML=big(dv.lake_events,0)+sub('lake working-memory events')
   +row('reachable',(dv.coverage_reachable===undefined?null:dv.coverage_reachable+' / '+dv.coverage_total))
   +row('dormant',dv.coverage_dormant)
   +`<div class="note">reachable, not proven to charge behaviour.</div>`;
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

async function loop(){
 try{
  if(!GEOM){try{GEOM=await (await fetch('/api/geom')).json();}catch(e){}}
  const d=await (await fetch('/api/all')).json();
  $('berr').textContent=''; paint(d);
 }catch(e){$('berr').textContent='disconnected';}
}
function spin(){ if(SPIN&&!DRAG)AZ+=0.0016; requestAnimationFrame(spin); }
loop(); setInterval(loop,1000); spin();
</script>
"""


def main():
    """Self-check: the page is served, the contract holds, and a chat turn round-trips via the queue."""
    import urllib.request

    from snn_data_lake.cortex.brain import make_brain

    os.environ.setdefault("NORD_TORCH", "0")
    os.environ.setdefault("NORD_PERSIST", "0")  # a self-check must never write the durable lake
    brain = make_brain(seed=0)
    for _ in range(8):
        brain.tick()

    # the no-orphan-panel gate, now covering every section the page reads
    missing = verify_contract(snapshot(brain))
    assert not missing, f"cockpit reads keys the brain cannot supply: {missing}"

    # the page must reference only keys the contract knows about -- a panel naming a key that no
    # longer exists is exactly the silent blank this gate is for
    for key in ("cells_covered", "cpg_weight", "td_flipped", "coverage_reachable", "provenance"):
        assert key in COCKPIT_HTML, f"cockpit page lost its reference to {key}"
    assert "make_brain" not in COCKPIT_HTML, "the page must never imply it constructs a brain"

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

    print(
        f"cockpit OK: page served, {len(geom['modules'])} modules / {len(geom['edges'])} tracts, "
        f"contract clean, chat queued-then-pumped (reply {len(got['reply'])} chars, "
        f"route={got.get('route', '')[:28]!r})"
    )


if __name__ == "__main__":
    main()
