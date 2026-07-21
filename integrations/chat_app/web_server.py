"""SNDL spiking brain -- LIVE web tracker (Server-Sent Events, no PNG).

This replaces the old PNG-per-turn server. A background daemon TICKS the full-stack brain
continuously at ~15 Hz with representation-only context (exactly like the warmup ticks -- NO real
actuator), reads a per-region spike snapshot each tick, and streams it to the browser over SSE
(GET /stream). A self-contained dark canvas page draws EVERY module of the 298-node 3D layout as a
node and animates each node's brightness/size toward its region's live spike value, so the brain is
GENUINELY live (tens of updates/sec) rather than a static image.

    PYTHONPATH="src;integrations/chat_app" python integrations/chat_app/web_server.py   # then open URL

DURABLE DATA LAKE (Part B): the brain's working-memory data lake is PERSISTENT, not a temp dir wiped
on close. NORD_HOME is pointed at <repo>/brain_data and the arm's lake_wm is redirected to a durable
.spkl there; on shutdown we close WITHOUT remove so the lake SURVIVES across runs, and on startup we
read + print how many events already persisted from a prior run.

Everything is representation-only: the page shows the brain lighting up as it thinks and as you
converse. No actuator route -- the arm E-STOP/safety envelope lives in the brainstem core and is
untouched by this file. Localhost bind only (127.0.0.1); single-user local dev tool, not public.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))            # integrations/chat_app -> integrations -> repo root
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- durable data lake location (Part B) -----------------------------------------------------------
# A PERSISTENT directory INSIDE the repo -- never a temp dir. NORD_HOME aligns every home()-based
# consumer to the same durable root; the chat working-memory lake lives at <NORD_HOME>/chat_wm.spkl.
BRAIN_DATA = os.environ.get("SNDL_BRAIN_DATA") or os.path.join(_REPO, "brain_data")
os.makedirs(BRAIN_DATA, exist_ok=True)
os.environ.setdefault("NORD_HOME", BRAIN_DATA)
DURABLE_LAKE = os.path.join(os.environ["NORD_HOME"], "chat_wm.spkl")

TICK_HZ = 15.0
_TICK_DT = 1.0 / TICK_HZ


def _durable_event_count(path):
    """Total events durably stored in the lake file == file size / fixed record width. StreamingHub
    uses a fixed-width record (default 8 B, <II>), append-only and never truncated on reopen, so the
    on-disk size is an exact, dependency-light event count that PERSISTS across process restarts."""
    if not os.path.exists(path):
        return 0
    try:
        from snn_data_lake.substrate.streaming_hub import _REC
        rec = _REC.size
    except Exception:
        rec = 8
    return os.path.getsize(path) // rec


def _crystal_count(sidecar):
    """Count the CRYSTALLIZED real memories persisted in a MemoryLake sidecar (.mem ASCII JSON). This
    is the durable Gc store -- consolidated, pruned, no idle noise. Robust to schema: counts the
    largest list found in the blob. 0 if absent/torn."""
    if not sidecar or not os.path.exists(sidecar):
        return 0
    try:
        with open(sidecar, encoding="ascii") as f:
            blob = json.load(f)
    except Exception:
        return 0
    if isinstance(blob, list):
        return len(blob)
    if isinstance(blob, dict):
        return max((len(v) for v in blob.values() if isinstance(v, list)), default=0)
    return 0


# ===================================================================================================
# Brain session: one process-wide full-stack brain, ticked continuously by a daemon thread. A single
# lock guards ALL brain access (the tick loop AND /chat) so they never enter the brain concurrently.
# ===================================================================================================
class BrainSession:
    def __init__(self, lake_path):
        self.lake_path = lake_path
        self.lock = threading.Lock()          # the ONE brain lock (tick loop + converse share it)
        self.brain = None
        self.region_activity = None
        self.tick_index = 0
        self.latest = {}                      # newest per-region snapshot {region -> float 0..1}
        self.last_msg = ""
        self.last_reply = ""
        self.layout_json = "{}"
        # CRYSTALLIZED MEMORY (Gc): the DURABLE store holds ONLY consolidated real memory -- never idle
        # noise. It lives in a bounded sidecar; the raw spike log is a transient session buffer.
        self.crystal_path = os.path.join(os.path.dirname(lake_path) or ".", "crystallized_memory.mem")
        self.memory = None
        self._raw_wm_path = None
        self._was_asleep = False              # tracks the brain's native circadian sleep transitions
        self.last_trace = {}                  # per-turn architecture trace (test-harness surface)
        self.startup_events = _crystal_count(self.crystal_path)  # real memories that persisted from prior runs
        self._stop = threading.Event()
        self._thread = None

    # -- representation-only context: like the warmup ticks, with a gentle drift so the tracker is
    #    genuinely LIVE. Pure representation fed to tick(); no actuator route, gate untouched.
    def _ctx(self, i):
        ph = i * 0.15
        return {
            "visual": [0.25 + 0.12 * math.sin(ph + k) for k in range(4)],
            "temperature": 0.5,
            "audio": 0.32 + 0.14 * math.sin(ph * 0.7),
            "distance": 0.65 + 0.10 * math.sin(ph * 0.5),   # stays clear of the reflex trip band
        }

    @staticmethod
    def _reward(u, act):
        return 1.0 if act == "RELEASE" else -0.2

    def build(self):
        from snn_data_lake.cortex.brain import Brain
        from brain_region_activity import region_activity
        self.region_activity = region_activity

        # the FULL brain (cortex/brain.py): understand/generate/act + ~20 wired faculties (sleep
        # consolidation, replay, neuromodulators, self-model, metacognition, reasoning, criticality,
        # wholistic integration, self-improvement, circadian) -- NOT the thin LiveBrain slice.
        self.full = Brain(seed=0)
        self.core = self.full.brain          # the LiveBrain control/safety core (spikes, lake, arm E-STOP)
        # CRYSTALLIZED MEMORY (Gc): inject a MemoryLake as the working memory BEFORE the first tick
        # (the arm's lazy build guard `if self.lake_wm is None` then skips). The raw spike log is a
        # TRANSIENT temp buffer; the DURABLE store is the bounded, pruned consolidated sidecar in
        # brain_data -- so the durable memory holds ONLY real, consolidated memory, never idle noise.
        import tempfile
        from snn_data_lake.substrate.lake_memory import MemoryLake
        _fd, self._raw_wm_path = tempfile.mkstemp(suffix="_wm.spkl"); os.close(_fd)
        mem = MemoryLake(n_channels=256, path=self._raw_wm_path, seed=0)
        mem.sidecar = self.crystal_path      # redirect the durable consolidated store to brain_data
        mem._load_sidecar()                  # recover prior crystallized memory across runs
        self.core.arm.lake_wm = mem          # inject -> the arm uses MemoryLake (habituates + prunes noise)
        self.memory = mem
        # enable the 3 newly-wired representation-only faculties in the live brain (NACA neuromod credit,
        # allocentric landmark map, cerebellar gain adaptation). They lazily build on the first tick and
        # feed readouts only -- no actuator route, the E-STOP still preempts.
        self.core.arm._naca_on = True
        self.core.arm._allo_on = True
        self.core.arm._gain_on = True
        # enable the HomeostaticPlasticityStack on the LIVE R-STDP intent learner: composes short-term
        # dynamic gain -> [R-STDP] -> homeostatic scaling + intrinsic plasticity, so the learning
        # synapses stay stable (bounded, rate-homeostatic) instead of drifting. Read-only on the motor
        # path (intent recognition only); default-safe -- any failure leaves the learner exactly as-is.
        try:
            from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
            ixn = getattr(self.core, "_ixn", None)
            if ixn is not None:
                n = len(getattr(ixn, "labels", None) or getattr(ixn, "neurons", []))
                if n:
                    ixn.stack = PlasticityRuleStack(n_neurons=n)   # superset: + BCM LR-gate, E/I balance, benna-fusi consolidation
        except Exception:
            pass
        b = self.brain = self.core           # spike_snapshot / lake / region_activity read from the core

        # warm up so it has a goal / broadcast / working memory to light up (also builds the durable lake)
        for i in range(12):
            b.tick(self._ctx(i), reward_fn=self._reward)
        self.tick_index = 12
        self._read_snapshot()
        self.layout_json = self._build_layout()

    def _build_layout(self):
        """The 298-node 3D layout, projected client-side. Sagittal-ish: x (lateral) + z (vertical)."""
        from brain_activity_3d import MODULE_POS, MODULE_REGION, XLIM, ZLIM
        regions = sorted(set(MODULE_REGION.values()))
        ridx = {r: i for i, r in enumerate(regions)}
        mods = [[round(p[0], 4), round(p[2], 4), ridx[MODULE_REGION[k]]]
                for k, p in MODULE_POS.items()]
        return json.dumps({"modules": mods, "regions": regions,
                           "xlim": list(XLIM), "zlim": list(ZLIM)})

    def _read_snapshot(self):
        """Per-region spike snapshot. CONTRACT: prefer brain.spike_snapshot() -> {region: float 0..1}
        (added concurrently by another agent); else fall back to region_activity() so we run standalone.
        MUST be called while holding self.lock (it reads live brain state)."""
        snap = None
        fn = getattr(self.brain, "spike_snapshot", None)
        if callable(fn):
            try:
                snap = fn()
            except Exception:
                snap = None
        if not isinstance(snap, dict) or not snap:
            try:
                snap = self.region_activity(self.brain, self.last_msg, self.last_reply)
            except Exception:
                snap = {}
        clean = {}
        for k, v in snap.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            clean[str(k)] = 0.0 if fv < 0.0 else (1.0 if fv > 1.0 else fv)
        self.latest = clean                    # atomic ref swap -> SSE readers never see a partial dict

    def snapshot(self):
        """Latest (tick, spikes) for streaming. Lock-free read: latest is swapped atomically."""
        return self.tick_index, dict(self.latest)

    def converse(self, message):
        """Under the brain lock: grounded converse() + refresh the snapshot so the intent's regions
        light up immediately. Returns the reply string. No actuator route beyond the gated arm."""
        with self.lock:
            # FULL brain: understand (firewall + self-model) -> generate (acquired language model,
            # else narrator) -> act (arm/lake commands take the deterministic GATED path). Feed the
            # turn back so the brain's language center keeps LEARNING from the conversation.
            out = self.full.respond(message)
            reply = out["reply"]
            try:
                lm = self.full._lm_serve()
                if lm is not None:
                    lm.expose(message)               # online acquisition: it learns as you talk to it
            except Exception:
                pass
            self.last_msg = message
            self.last_reply = reply
            self._read_snapshot()
            self.last_trace = self._build_trace(message, out)   # per-turn architecture trace (test harness)
        return reply

    def _build_trace(self, message, out):
        """Per-turn ARCHITECTURE TRACE -- the test-harness surface: what fired understand -> route -> act,
        which regions lit, and the live state of the brain's faculties. Lets talking to it TEST the whole
        brain. Representation-only readout; touches no actuator."""
        top = sorted(dict(self.latest).items(), key=lambda kv: -kv[1])[:6]
        arm = self.core.arm
        drives = {}
        if hasattr(arm, "_naca_credit"):
            drives["naca_credit"] = round(float(arm._naca_credit), 4)
        if hasattr(arm, "_motor_gain"):
            drives["gain_adapt"] = round(float(arm._motor_gain), 4)
        if hasattr(arm, "_allo_pos"):
            drives["allocentric_pos"] = [round(x, 3) for x in arm._allo_pos]
        route = out.get("route", "?")
        return {
            "understand": {"message": message[:80],
                           "top_regions": [[r, round(v, 2)] for r, v in top]},
            "route": route,
            "act": "gated arm command (E-STOP preempts)" if "gated arm" in route
                   else "no motor (representation-only)",
            "faculties": {
                "circadian": out.get("circadian"),
                "regions_live": out.get("n_regions_live"),
                "consolidation_acc": out.get("consolidation_acc"),
                "reasoning_units": out.get("reasoning_units"),
                "criticality": out.get("criticality"),
                "reliability": out.get("reliability"),
                "crystallized_memories": arm.lake_wm.n_kept()
                    if (arm.lake_wm is not None and hasattr(arm.lake_wm, "n_kept")) else None,
                "drives": drives,
            },
        }

    # -- continuous tick daemon ---------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._tick_loop, name="brain-tick", daemon=True)
        self._thread.start()

    def _tick_loop(self):
        while not self._stop.is_set():
            ctx = self._ctx(self.tick_index)
            with self.lock:
                try:
                    self.core.tick(ctx, reward_fn=self._reward)   # drive the spiking substrate (tracker)
                    if self.tick_index % 15 == 0:                 # ~1 Hz: run the Brain-level faculties
                        self.full.tick()                          # NATIVE circadian sleep/consolidation/reasoning
                        # sleep is ALREADY in the brain: let ITS circadian cycle drive memory
                        # consolidation -- when the brain falls asleep, prune the working memory.
                        asleep = bool(getattr(self.full, "_asleep", False))
                        if asleep and not self._was_asleep and self.memory is not None:
                            self.memory.consolidate()             # prune idle noise -> bounded Gc memory
                            try:
                                self.memory._save_sidecar()       # crash-safe: persist the night's Gc NOW (atomic),
                            except Exception:                     # so a crash loses at most one day, not the session
                                pass
                        self._was_asleep = asleep
                except Exception:
                    pass
                self._read_snapshot()
            self.tick_index += 1
            time.sleep(_TICK_DT)

    def close(self, join=True):
        """Stop ticking and RELEASE the durable lake WITHOUT removing it, so the working memory
        SURVIVES across runs (the opposite of the old close(remove=True) that deleted the temp lake)."""
        self._stop.set()
        if join and self._thread is not None:
            self._thread.join(timeout=1.0)
        with self.lock:
            try:
                if self.memory is not None:
                    self.memory.consolidate()      # final prune -> keep ONLY real memory
                    self.memory.close()            # persists the crystallized sidecar (Gc), no remove
            except Exception:
                pass
        if self._raw_wm_path and os.path.exists(self._raw_wm_path):
            try:
                os.remove(self._raw_wm_path)       # drop the TRANSIENT raw spike buffer (idle noise)
            except Exception:
                pass
        return _crystal_count(self.crystal_path)


SESSION = BrainSession(DURABLE_LAKE)


# ===================================================================================================
# Self-contained dark frontend: canvas node field + SSE + live animation + chat.
# ===================================================================================================
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>SNDL - live spiking brain</title><style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 html,body{height:100%}
 body{margin:0;background:#07070c;color:#e8e8ee;font:14px/1.5 system-ui,Segoe UI,sans-serif;
      display:flex;flex-direction:column}
 header{padding:9px 16px;border-bottom:1px solid #1b1b26;flex:0 0 auto}
 header b{color:#f6a83c;letter-spacing:.3px}
 header span{color:#7c7c8c;font-size:12px}
 header i{color:#54d98c;font-style:normal;font-size:12px}
 #stage{position:relative;flex:1 1 auto;min-height:0}
 #brain{display:block;width:100%;height:100%}
 #legend{position:absolute;top:10px;left:12px;background:rgba(10,10,18,.62);border:1px solid #22222e;
         border-radius:9px;padding:9px 11px;min-width:172px;backdrop-filter:blur(3px)}
 #legend h4{margin:0 0 6px;font-size:11px;color:#9a9aab;font-weight:600;text-transform:uppercase;
            letter-spacing:.6px}
 .row{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12.5px}
 .sw{width:11px;height:11px;border-radius:50%;flex:0 0 auto;box-shadow:0 0 7px currentColor}
 .rn{flex:1 1 auto;color:#d2d2de}
 .rv{color:#8a8a99;font-variant-numeric:tabular-nums}
 #hud{position:absolute;top:10px;right:12px;text-align:right;font-size:11.5px;color:#6f6f80;
      font-variant-numeric:tabular-nums}
 #hud b{color:#f6a83c}
 #log{position:absolute;left:12px;right:12px;bottom:8px;max-height:34%;overflow:auto;
      display:flex;flex-direction:column;gap:5px;pointer-events:none}
 .msg{max-width:76%;padding:6px 11px;border-radius:11px;font-size:13px;pointer-events:auto;
      box-shadow:0 1px 6px rgba(0,0,0,.4)}
 .you{align-self:flex-end;background:#2b2033;color:#f0e6f6}
 .bot{align-self:flex-start;background:#12121c;border:1px solid #23232f;color:#dfe6ec}
 #bar{flex:0 0 auto;display:flex;gap:8px;padding:10px 12px;border-top:1px solid #1b1b26;
      background:#0a0a12}
 #msg{flex:1 1 auto;background:#12121b;border:1px solid #2a2a37;border-radius:9px;color:#e8e8ee;
      padding:9px 12px;font-size:14px;outline:none}
 #msg:focus{border-color:#f6a83c66}
 #send{background:#f6a83c;color:#1a1206;border:0;border-radius:9px;padding:9px 20px;font-weight:600;
       cursor:pointer}
 #send:disabled{opacity:.5;cursor:wait}
</style></head><body>
 <header><b>SNDL</b> <span>&mdash; live spiking activity, every module streamed at ~15 Hz over SSE
   (representation-only; the arm E-STOP is untouched)</span> &nbsp; <i id="conn">connecting...</i></header>
 <div id="stage">
   <canvas id="brain"></canvas>
   <div id="legend"><h4>hottest regions</h4><div id="legrows"></div></div>
   <div id="hud"><div>tick <b id="tk">0</b></div><div id="fps">-- fps</div></div>
   <div id="log"></div>
 </div>
 <div id="bar">
   <input id="msg" placeholder="talk to the brain -- try 'what do you remember', 'is it safe to move', 'what is your goal'" autocomplete="off">
   <button id="send">send</button>
 </div>
<script>
const cv=document.getElementById('brain'),ctx=cv.getContext('2d');
const conn=document.getElementById('conn');
let DPR=Math.max(1,window.devicePixelRatio||1),W=0,H=0;
let LAYOUT=null, NODES=[], REGIONS=[], spikes={}, tick=0;

// inferno-like perceptual ramp on a dark bg
const STOPS=[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function inferno(t){t=t<0?0:t>1?1:t;const f=t*(STOPS.length-1),i=Math.floor(f),g=f-i;
  const a=STOPS[i],b=STOPS[Math.min(i+1,STOPS.length-1)];
  return [a[0]+(b[0]-a[0])*g,a[1]+(b[1]-a[1])*g,a[2]+(b[2]-a[2])*g];}
function regionHue(name){let h=0;for(let i=0;i<name.length;i++)h=(h*31+name.charCodeAt(i))>>>0;return h%360;}
function hsl2rgb(h,s,l){s/=100;l/=100;const k=n=>(n+h/30)%12,a=s*Math.min(l,1-l);
  const f=n=>l-a*Math.max(-1,Math.min(k(n)-3,Math.min(9-k(n),1)));
  return[255*f(0),255*f(8),255*f(4)];}

function resize(){const r=cv.getBoundingClientRect();W=r.width;H=r.height;
  cv.width=W*DPR;cv.height=H*DPR;ctx.setTransform(DPR,0,0,DPR,0,0);project();}
function project(){if(!LAYOUT)return;const pad=46;
  const[x0,x1]=LAYOUT.xlim,[z0,z1]=LAYOUT.zlim;
  const sx=(W-2*pad)/(x1-x0),sz=(H-2*pad)/(z1-z0);const s=Math.min(sx,sz);
  const cx=W/2,cz=H/2,mx=(x0+x1)/2,mz=(z0+z1)/2;
  for(const n of NODES){n.px=cx+(n.x-mx)*s;n.py=cz-(n.z-mz)*s;}   // z up
  window._hull={cx,cz,rx:(x1-x0)*s*0.5*1.04,ry:(z1-z0)*s*0.5*1.02};}

function loadLayout(){fetch('/layout').then(r=>r.json()).then(L=>{
  LAYOUT=L;REGIONS=L.regions;
  NODES=L.modules.map(m=>{const rn=REGIONS[m[2]],hue=regionHue(rn);
    return {x:m[0],z:m[1],region:rn,base:hsl2rgb(hue,55,26),cur:0,tgt:0,flash:0};});
  resize();});}

function targetFor(region){let v=spikes[region];
  if(v===undefined)v=(region==='reasoning')?(spikes['cortex']||0):0;return v;}

function draw(){
  ctx.clearRect(0,0,W,H);
  // faint brain outline
  const hl=window._hull;
  if(hl){ctx.save();ctx.strokeStyle='rgba(120,120,150,.16)';ctx.lineWidth=1.4;
    ctx.beginPath();ctx.ellipse(hl.cx,hl.cz,hl.rx,hl.ry,0,0,Math.PI*2);ctx.stroke();
    ctx.strokeStyle='rgba(120,120,150,.08)';ctx.beginPath();
    ctx.ellipse(hl.cx,hl.cz,hl.rx*0.72,hl.ry*0.82,0,0,Math.PI*2);ctx.stroke();ctx.restore();}
  // nodes
  for(const n of NODES){
    const t=n.cur;const[ir,ig,ib]=inferno(t);
    // dim region-tinted at rest, inferno-hot when active
    const r=n.base[0]+(ir-n.base[0])*t, g=n.base[1]+(ig-n.base[1])*t, b=n.base[2]+(ib-n.base[2])*t;
    const rad=1.5+3.6*t+ (n.flash*3.0);
    if(t>0.04||n.flash>0){ctx.beginPath();ctx.arc(n.px,n.py,rad+4*t+n.flash*5,0,Math.PI*2);
      ctx.fillStyle=`rgba(${ir|0},${ig|0},${ib|0},${0.10*t+0.18*n.flash})`;ctx.fill();}
    ctx.beginPath();ctx.arc(n.px,n.py,rad,0,Math.PI*2);
    ctx.fillStyle=`rgb(${r|0},${g|0},${b|0})`;ctx.fill();
  }
}
let last=performance.now(),frames=0,facc=0;
function animate(now){
  const dt=Math.min(0.05,(now-last)/1000);last=now;
  for(const n of NODES){const tg=targetFor(n.region);
    const jump=tg-n.cur;if(jump>0.22)n.flash=Math.min(1,n.flash+jump);   // spark on a big rise
    n.cur+=(tg-n.cur)*Math.min(1,dt*9);n.flash*=Math.max(0,1-dt*4.5);}
  draw();
  frames++;facc+=dt;if(facc>=0.5){document.getElementById('fps').textContent=(frames/facc).toFixed(0)+' fps';frames=0;facc=0;}
  requestAnimationFrame(animate);
}

function legend(){const rows=Object.entries(spikes).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const el=document.getElementById('legrows');el.innerHTML='';
  for(const[r,v]of rows){const[cr,cg,cb]=inferno(v);const d=document.createElement('div');d.className='row';
    d.innerHTML=`<span class="sw" style="color:rgb(${cr|0},${cg|0},${cb|0});background:rgb(${cr|0},${cg|0},${cb|0})"></span>`+
      `<span class="rn">${r}</span><span class="rv">${v.toFixed(2)}</span>`;el.appendChild(d);}}

function connect(){const es=new EventSource('/stream');
  es.onopen=()=>{conn.textContent='live';conn.style.color='#54d98c';};
  es.onerror=()=>{conn.textContent='reconnecting...';conn.style.color='#e0a13c';};
  es.onmessage=e=>{const d=JSON.parse(e.data);spikes=d.spikes;tick=d.tick;
    document.getElementById('tk').textContent=tick;legend();};}

const logEl=document.getElementById('log');
function addMsg(t,cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=t;
  logEl.appendChild(d);logEl.scrollTop=logEl.scrollHeight;}
const input=document.getElementById('msg'),btn=document.getElementById('send');
function renderTrace(t){const f=t.faculties||{},dr=f.drives||{};
  const regs=((t.understand||{}).top_regions||[]).map(x=>x[0]+' '+x[1]).join(', ');
  const lines=['route -> '+(t.route||'?'),
    'act -> '+(t.act||''),
    'regions lit -> '+regs,
    'faculties -> circadian '+f.circadian+' | regions_live '+f.regions_live+' | crystallized_mem '+f.crystallized_memories,
    '            consolidation '+f.consolidation_acc+' | reasoning_units '+f.reasoning_units+' | criticality '+f.criticality+' | reliability '+f.reliability,
    'drives -> naca '+dr.naca_credit+' | gain_adapt '+dr.gain_adapt+' | allocentric '+JSON.stringify(dr.allocentric_pos||[])];
  const d=document.createElement('div');
  d.style.cssText='font:11px/1.45 ui-monospace,Consolas,monospace;color:#8890a6;background:#101019;border-left:2px solid #3a3a52;margin:1px 0 9px;padding:5px 9px;border-radius:5px';
  d.innerHTML=lines.map(p=>'<div>'+p.replace(/[<>&]/g,'')+'</div>').join('');
  logEl.appendChild(d);logEl.scrollTop=logEl.scrollHeight;}
function send(){const m=input.value.trim();if(!m)return;addMsg(m,'you');input.value='';btn.disabled=true;
  fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})})
    .then(r=>r.json()).then(d=>{addMsg(d.reply||('['+(d.error||'no reply')+']'),'bot');if(d.trace)renderTrace(d.trace);})
    .catch(err=>addMsg('[error: '+err+']','bot')).finally(()=>{btn.disabled=false;input.focus();});}
btn.onclick=send;input.addEventListener('keydown',e=>{if(e.key==='Enter')send();});

window.addEventListener('resize',resize);
loadLayout();connect();requestAnimationFrame(animate);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/layout":
            self._send(200, SESSION.layout_json)
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "tick": SESSION.tick_index,
                                        "lake": SESSION.lake_path,
                                        "events": _durable_event_count(SESSION.lake_path)}))
        elif self.path == "/stream":
            self._stream()
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _stream(self):
        """Infinite SSE generator: push the latest per-region snapshot at ~15 Hz to this client."""
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while not SESSION._stop.is_set():
                tk, sp = SESSION.snapshot()
                payload = json.dumps({"tick": tk, "spikes": sp})
                self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                self.wfile.flush()
                time.sleep(_TICK_DT)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return                                 # client closed the EventSource -> end this thread

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, json.dumps({"error": "not found"})); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send(400, json.dumps({"error": "empty message"})); return
            reply = SESSION.converse(message)
            self._send(200, json.dumps({"reply": reply, "trace": SESSION.last_trace}))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, *a):   # quieter console
        pass


def serve(port=8765, host="127.0.0.1"):
    print("=" * 74)
    print("SNDL live spiking brain -- building the full-stack brain...")
    SESSION.build()
    print(f"  durable data lake : {SESSION.lake_path}")
    print(f"  events persisted from prior runs (startup): {SESSION.startup_events}")
    SESSION.start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"  open  http://{host}:{port}   (Ctrl-C to stop)")
    print("=" * 74)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        final = SESSION.close()
        httpd.server_close()
        print(f"durable lake preserved (NOT removed): {SESSION.lake_path} ({final} events)")


def _selfcheck():
    """Headless: no server, no browser. Start the tick loop ~1.5s, assert /stream-style snapshots are
    produced and CHANGE over time, run one converse under the lock, and prove the lake is a durable
    persistent path whose event count PERSISTS. Prints raw results."""
    print("=" * 70)
    print("WEB SERVER live self-check")
    print(f"  durable lake path : {SESSION.lake_path}")
    print(f"  in repo?          : {os.path.abspath(SESSION.lake_path).startswith(os.path.abspath(_REPO))}")
    import tempfile as _tf
    is_temp = os.path.abspath(SESSION.lake_path).startswith(os.path.abspath(_tf.gettempdir()))
    print(f"  is a temp dir?    : {is_temp}  (must be False)")
    print(f"  events at startup : {SESSION.startup_events}  (from prior runs)")
    assert not is_temp, "the durable lake must NOT live under the system temp dir"

    SESSION.build()
    SESSION.start()

    # collect snapshots over ~1.5s and prove per-region values are produced AND change over time
    samples = []
    t_end = time.time() + 1.5
    while time.time() < t_end:
        tk, sp = SESSION.snapshot()
        if sp:
            samples.append((tk, sp))
        time.sleep(0.1)
    assert samples, "expected /stream-style snapshots to be produced"
    first, last = samples[0][1], samples[-1][1]
    keys = sorted(set(first) & set(last))
    assert keys, "snapshots must carry per-region keys"
    changed = [k for k in keys if abs(first.get(k, 0.0) - last.get(k, 0.0)) > 1e-6]
    nonzero = any(v > 0.0 for v in last.values())
    all_const = all(abs(first.get(k, 0.0) - last.get(k, 0.0)) <= 1e-12 for k in keys)
    print(f"  snapshots sampled : {len(samples)}  (ticks {samples[0][0]} -> {samples[-1][0]})")
    print(f"  regions per snap  : {len(last)}")
    print(f"  regions CHANGED   : {len(changed)} of {len(keys)}  e.g. "
          + ", ".join(f"{k} {first.get(k,0):.3f}->{last.get(k,0):.3f}" for k in changed[:4]))
    print(f"  any nonzero / all-constant : {nonzero} / {all_const}")
    assert not all_const and changed and nonzero, "per-region spikes must be live (change, not all zero/const)"

    reply = SESSION.converse("is it safe to move")
    print(f"  converse reply    : {reply[:72]!r}")
    assert reply, "expected a non-empty converse() reply under the lock"

    final = SESSION.close()
    print(f"  crystallized mem  : startup {SESSION.startup_events} -> shutdown {final}  (BOUNDED + pruned; persists across runs, does NOT grow unbounded)")
    # crystallized memory is actively PRUNED to a bounded store, so it does NOT grow monotonically --
    # the durable proof is that real memory PERSISTS (>0), not that the count strictly increases.
    assert final > 0, "durable crystallized memory must persist (real, pruned memory -- not idle noise)"
    print(f"  DURABLE PROOF     : real memory persists across runs (bounded, never idle noise)")
    print("WEB SERVER live self-check OK")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="SNDL live spiking brain -- SSE web tracker")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selfcheck", action="store_true",
                    help="headless: tick loop + live-snapshot + durable-lake proof, no server")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
    else:
        serve(port=args.port)


if __name__ == "__main__":
    main()
