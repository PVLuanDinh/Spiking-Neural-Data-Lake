"""Brain telemetry: ONE snapshot contract, an append-only recorder, and a stdlib HTTP server.

WHY THIS EXISTS
    `Brain.state()` has carried the docstring "Telemetry for /api/state" since it was written, and
    `research/brain_web.html` has been fetching `/api/state` -- but nothing in this repo ever served
    it. `LiveBrain.spike_snapshot()` documents itself as "the interface contract the live web tracker
    reads" and had zero consumers. Both halves of a dashboard existed; the wire between them did not.

    This module is that wire, and nothing more. It computes no new brain state. It reads what the
    canonical Brain already produces every tick.

THE HONESTY CONTRACT (the whole point -- read before adding a panel)
    A telemetry surface is the easiest place in a project to lie, because a plausible number is
    indistinguishable from a real one once it is rendered as a bar. Three rules hold the line:

    1. `snapshot()` is the ONLY source a panel may read. If a value is not in here, the panel does
       not get built. No reaching into private attributes from the frontend.
    2. A value that is absent renders as `None` -> the UI draws an em-dash, NEVER a zero bar. A zero
       bar is a claim ("this signal is live and reads zero"); an em-dash is the truth ("no signal").
    3. `verify_contract()` fails loudly if a key the dashboard references stops existing. That is the
       no-orphan-panel gate: it makes a silently-dead panel a build error instead of a lie on screen.

    Corollary, stated because it is tempting: several SNDL signals are computed-but-not-routed
    (`_motor_gain`, `_dev_fm_gain`, `_comp_ema` -- their own source comments say "NEVER applied").
    Those are exposed here under `gated:` with `applied: False`. A panel showing them MUST say so.
    A gain that never reaches a motor is not a control signal; it is a number.

NOT A SECOND BRAIN (Policy 4)
    This module never constructs a Brain. It takes one that already exists. `serve()` is handed the
    live brain by its caller -- the daemon or the chat app -- so there is exactly one brain and this
    is a window onto it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Keys the bundled dashboard reads. verify_contract() asserts every one is producible, so deleting a
# brain signal breaks the build instead of silently blanking a panel.
DASHBOARD_KEYS = (
    "circadian",
    "phase",
    "wake_ticks",
    "sleep_ticks",
    "nights_elapsed",
    "neuromod_serotonin",
    "neuromod_noradrenaline",
    "neuromod_acetylcholine",
    "cortisol",
    "curiosity_mag",
    "da_level",
    "rpe",
    "emotion",
    "dominant_drive",
    "access_index",
    "criticality",
    "region_coherence",
    "integration_competence",
    "grown_units",
    "predict_accuracy",
    "reasoning_accuracy",
    "reliability",
    "pc_surprise",
    "attended_modality",
    "map_position",
    # derived-section keys (verify_contract checks cognition | derived, so these belong here too --
    # they were omitted once and the dashboard silently drew em-dashes for affect until a human
    # noticed. That is exactly the regression this tuple exists to turn into a build failure.)
    "valence",
    "arousal",
)


def _num(v, nd=4):
    """Round a value if it is a finite number; otherwise hand it back untouched (incl. None)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return None if not math.isfinite(v) else round(float(v), nd)


def snapshot(brain):
    """The single telemetry contract: everything a panel may read, and nothing invented.

    Sections are kept separate so a reader can tell WHERE a number came from, which is most of what
    makes telemetry trustworthy:
      cognition  -- Brain.state(), the canonical ~105-key view
      spikes     -- LiveBrain.spike_snapshot(), 17 REAL per-region rates in [0,1] (region-resolved,
                    NOT cell-type-resolved; do not label these as neuron populations)
      body       -- ArmSim.state(). Carries `moves: False` because only `effector` currently changes.
      derived    -- arithmetic over the above, computed here so state() does not carry two
                    representations of the same fact
      gated      -- real variables their own source marks as never applied. `applied` is False.
    """
    out = {"t": time.time(), "cognition": {}, "spikes": {}, "body": {}, "derived": {}, "gated": {}}

    try:
        st = brain.state()
        out["cognition"] = {k: _num(v) for k, v in st.items()}
    except Exception as e:  # telemetry must never take the brain down with it
        out["cognition"] = {"_error": str(e)}
        st = {}

    try:
        out["spikes"] = {k: _num(v) for k, v in brain.brain.spike_snapshot().items()}
    except Exception:
        out["spikes"] = {}

    arm = None
    try:
        arm = brain.brain.arm.arm.state()
        out["body"] = {k: _num(v) for k, v in arm.items()}
        # Stated, not hidden: the joints do not move on the canonical tick. A dashboard that let a
        # viewer assume otherwise would be the exact failure this module exists to prevent.
        out["body"]["moves"] = bool(any(abs(float(a)) > 1e-9 for a in arm.get("theta_deg", []) or []))
    except Exception:
        out["body"] = {}

    # ---- derived (arithmetic only; no new state) ----
    pos = st.get("map_position") or []
    can = st.get("grid_can_xy") or []
    try:
        if len(pos) >= 2:
            out["derived"]["distance_from_origin"] = _num(math.hypot(float(pos[0]), float(pos[1])))
        if len(pos) >= 2 and len(can) >= 2:
            out["derived"]["place_error"] = _num(
                math.hypot(float(can[0]) - float(pos[0]), float(can[1]) - float(pos[1]))
            )
    except Exception:
        pass

    # affect arrives as the pair `affective_appraisal: [valence, arousal]`; the circumplex wants the
    # two axes separately. Unpack rather than teach the frontend the tuple's index order.
    app = st.get("affective_appraisal") or []
    if len(app) >= 2:
        out["derived"]["valence"] = _num(app[0])
        out["derived"]["arousal"] = _num(app[1])

    # ---- gated: real, live, and NOT routed to any action. Their source comments say so. ----
    a = getattr(getattr(brain, "brain", None), "arm", None)
    if a is not None:
        for key, attr in (
            ("motor_gain", "_motor_gain"),
            ("fm_gain", "_dev_fm_gain"),
            ("competence_ema", "_comp_ema"),
        ):
            if hasattr(a, attr):
                out["gated"][key] = {"value": _num(getattr(a, attr, 0.0)), "applied": False}
    return out


def verify_contract(snap):
    """Return the DASHBOARD_KEYS that the snapshot cannot supply. Empty list == no orphan panels."""
    have = set(snap.get("cognition", {})) | set(snap.get("derived", {}))
    return [k for k in DASHBOARD_KEYS if k not in have]


class Recorder:
    """Append-only JSONL of one snapshot per call. The FLOG equivalent: record first, render later.

    Deliberately dependency-free and deliberately not on the hot path by default -- the caller
    decides when to `write()`, so a daemon can record every tick and a chat app can record none.
    """

    def __init__(self, path, every=1):
        self.path = str(path)
        self.every = max(1, int(every))
        self.n = 0
        self._fh = None

    def write(self, brain):
        self.n += 1
        if self.n % self.every:
            return None
        snap = snapshot(brain)
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(snap, ensure_ascii=True) + "\n")
        self._fh.flush()
        return snap

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _handler(get_snap):
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
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, DASHBOARD_HTML, "text/html; charset=utf-8")
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

        def log_message(self, *a):  # keep the brain's stdout clean
            return

    return H


def serve(brain, port=8137, block=True):
    """Serve the live brain's telemetry. Does NOT construct a brain -- Policy 4: exactly one brain.

    Binds loopback only. This is an observability surface over a running process, not a public API,
    and it should not become reachable from the network by default.
    """
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), _handler(lambda: snapshot(brain)))
    print(f"[telemetry] http://127.0.0.1:{port}/  (/api/state /api/spikes /api/all)")
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


DASHBOARD_HTML = """<!doctype html><meta charset="utf-8"><title>SNDL brain telemetry</title>
<style>
:root{--bg:#080b12;--pan:#0e1420;--ln:#1c2740;--fg:#c7d4ea;--dim:#5f708c;--cy:#39c6e8;--am:#e8a33d;--gr:#4fd48a;--rd:#e85f6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
header{padding:10px 16px;border-bottom:1px solid var(--ln);display:flex;gap:18px;align-items:baseline}
h1{font-size:15px;margin:0;color:var(--cy);letter-spacing:.14em}
.badge{padding:1px 8px;border:1px solid var(--ln);border-radius:9px;font-size:11px}
main{display:grid;grid-template-columns:1fr 340px;gap:12px;padding:12px}
.pan{background:var(--pan);border:1px solid var(--ln);border-radius:6px;padding:10px 12px}
.pan h2{font-size:10px;letter-spacing:.16em;color:var(--cy);margin:0 0 8px;font-weight:600}
.row{display:flex;justify-content:space-between;gap:10px;padding:2px 0}
.row span:last-child{color:#fff}
.bar{height:5px;background:#16203352;border-radius:3px;overflow:hidden;margin:3px 0 6px}
.bar i{display:block;height:100%;background:var(--cy)}
.na{color:var(--dim)!important}
.note{color:var(--dim);font-size:11px;margin-top:6px;font-style:italic}
canvas{width:100%;display:block}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
</style>
<header>
  <h1>SNDL</h1><span class="badge" id="circ">--</span><span class="badge" id="tk">--</span>
  <span class="badge" id="ni">--</span><span class="badge" id="err" style="color:var(--rd)"></span>
</header>
<main>
 <div>
  <div class="pan"><h2>REGION SPIKE RATES &mdash; live, 17 regions</h2>
    <canvas id="sp" height="230"></canvas>
    <div class="note">Region-resolved, not cell-type-resolved. Absent snapshot renders empty, never fabricated.</div>
  </div>
  <div class="grid2" style="margin-top:12px">
    <div class="pan"><h2>WORLD MODEL</h2><div id="wm"></div></div>
    <div class="pan"><h2>LEARNING</h2><div id="lr"></div></div>
  </div>
  <div class="pan" style="margin-top:12px"><h2>BODY &mdash; 2-DOF planar arm</h2><div id="bd"></div>
    <div class="note" id="bdnote"></div></div>
 </div>
 <div>
  <div class="pan"><h2>NEUROMODULATORS</h2><div id="nm"></div></div>
  <div class="pan" style="margin-top:12px"><h2>AFFECT</h2>
    <canvas id="va" height="150"></canvas><div id="af"></div></div>
  <div class="pan" style="margin-top:12px"><h2>ACCESS &amp; INTEGRATION</h2><div id="ai"></div></div>
  <div class="pan" style="margin-top:12px"><h2>COMPUTED, NOT APPLIED</h2><div id="gt"></div>
    <div class="note">Live variables their own source marks as never routed to an action.</div></div>
 </div>
</main>
<script>
const $=i=>document.getElementById(i);
const fmt=v=>(v===null||v===undefined||v==='')?'&mdash;':(typeof v==='number'?(Math.abs(v)<1e-4&&v!==0?v.toExponential(1):v.toFixed(Math.abs(v)>=100?0:3)):v);
const na=v=>(v===null||v===undefined)?' class="na"':'';
function rows(el,pairs){el.innerHTML=pairs.map(([k,v])=>
 `<div class="row"><span>${k}</span><span${na(v)}>${fmt(v)}</span></div>`).join('')}
function bars(el,pairs){el.innerHTML=pairs.map(([k,v])=>{
 if(v===null||v===undefined)return `<div class="row"><span>${k}</span><span class="na">&mdash;</span></div>`;
 const p=Math.max(0,Math.min(1,v))*100;
 return `<div class="row"><span>${k}</span><span>${fmt(v)}</span></div><div class="bar"><i style="width:${p}%"></i></div>`}).join('')}
function spikes(c,s){const ks=Object.keys(s).sort();const ctx=c.getContext('2d');
 const w=c.width=c.clientWidth*2,h=c.height=460;ctx.scale(1,1);ctx.clearRect(0,0,w,h);
 if(!ks.length){ctx.fillStyle='#5f708c';ctx.font='22px monospace';ctx.fillText('no spike snapshot',20,40);return}
 const bh=h/ks.length;ctx.font='20px ui-monospace,monospace';
 ks.forEach((k,i)=>{const v=s[k]||0,y=i*bh;
  ctx.fillStyle='#162033';ctx.fillRect(300,y+4,w-340,bh-8);
  ctx.fillStyle=v>.66?'#39c6e8':v>.33?'#4fd48a':'#e8a33d';ctx.fillRect(300,y+4,(w-340)*v,bh-8);
  ctx.fillStyle=v?'#c7d4ea':'#5f708c';ctx.fillText(k,8,y+bh/2+7);
  ctx.fillStyle='#fff';ctx.fillText(v.toFixed(2),w-70,y+bh/2+7)})}
function circumplex(c,v,a){const ctx=c.getContext('2d');const w=c.width=c.clientWidth*2,h=c.height=300;
 ctx.clearRect(0,0,w,h);const cx=w/2,cy=h/2,r=Math.min(cx,cy)-14;
 ctx.strokeStyle='#1c2740';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,cy,r,0,7);ctx.stroke();
 ctx.beginPath();ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.stroke();
 if(v===null||v===undefined||a===null||a===undefined){ctx.fillStyle='#5f708c';ctx.font='18px monospace';
  ctx.fillText('no affect signal',cx-70,cy-8);return}
 const x=cx+Math.max(-1,Math.min(1,v))*r,y=cy-Math.max(-1,Math.min(1,a))*r;
 ctx.fillStyle='#39c6e8';ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fill()}
async function tick(){
 try{const r=await fetch('/api/all');const d=await r.json();$('err').textContent='';
  const s=d.cognition||{},g=d.gated||{},b=d.body||{},dv=d.derived||{};
  $('circ').textContent=(s.circadian||'--').toUpperCase();
  $('tk').textContent='tick '+((s.wake_ticks||0)+(s.sleep_ticks||0));
  $('ni').textContent='night '+(s.nights_elapsed??'--');
  spikes($('sp'),d.spikes||{});
  bars($('nm'),[['dopamine',s.da_level],['serotonin',s.neuromod_serotonin],
   ['noradrenaline',s.neuromod_noradrenaline],['acetylcholine',s.neuromod_acetylcholine],
   ['cortisol',s.cortisol]]);
  circumplex($('va'),dv.valence,dv.arousal);
  rows($('af'),[['emotion',s.emotion],['drive',s.dominant_drive],['valence',dv.valence],
   ['arousal',dv.arousal],['RPE',s.rpe]]);
  bars($('ai'),[['access index',s.access_index],['criticality',s.criticality],
   ['region coherence',s.region_coherence],['integration',s.integration_competence]]);
  rows($('wm'),[['pc surprise',s.pc_surprise],['curiosity mag',s.curiosity_mag],
   ['attended modality',s.attended_modality],['distance',dv.distance_from_origin],
   ['place error',dv.place_error]]);
  rows($('lr'),[['grown units',s.grown_units],['predict acc',s.predict_accuracy],
   ['reasoning acc',s.reasoning_accuracy],['reliability',s.reliability]]);
  rows($('bd'),[['theta (deg)',(b.theta_deg||[]).join(', ')],['end effector',(b.ee||[]).join(', ')],
   ['effector',b.effector],['collision',b.collision],['deviation',b.deviation]]);
  $('bdnote').textContent = b.moves===false
   ? 'Joints have not moved: the canonical tick emits no motor tokens, so only `effector` changes.' : '';
  $('gt').innerHTML=Object.keys(g).length?Object.entries(g).map(([k,o])=>
   `<div class="row"><span>${k}</span><span>${fmt(o.value)} <em class="na">not applied</em></span></div>`).join('')
   :'<div class="row"><span class="na">none</span></div>';
 }catch(e){$('err').textContent='disconnected'}
}
tick();setInterval(tick,1000);
</script>
"""


def main():
    """Self-check: the contract holds against a real brain, the recorder round-trips, HTTP answers."""
    import os
    import tempfile
    import urllib.request

    from snn_data_lake.cortex.brain import make_brain

    os.environ.setdefault("NORD_TORCH", "0")
    brain = make_brain(seed=0)
    for _ in range(12):
        brain.tick()

    snap = snapshot(brain)
    for sec in ("cognition", "spikes", "body", "derived", "gated"):
        assert sec in snap, f"snapshot missing section {sec}"
    assert len(snap["spikes"]) == 17, f"expected 17 regions, got {len(snap['spikes'])}"

    missing = verify_contract(snap)
    assert not missing, f"no-orphan-panel gate: dashboard reads keys the brain cannot supply: {missing}"

    # every gated readout must declare itself unapplied -- this is what stops it becoming theatre
    for k, v in snap["gated"].items():
        assert v["applied"] is False, f"gated readout {k} claims applied=True; route it or relabel it"

    # recorder round-trips as JSONL
    tmp = os.path.join(tempfile.gettempdir(), "sndl_telemetry_selfcheck.jsonl")
    if os.path.exists(tmp):
        os.remove(tmp)
    rec = Recorder(tmp, every=1)
    for _ in range(3):
        brain.tick()
        rec.write(brain)
    rec.close()
    with open(tmp, encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    assert len(lines) == 3, f"recorder wrote {len(lines)} lines, expected 3"
    assert lines[0]["spikes"], "recorded snapshot has no spike section"
    os.remove(tmp)

    # server answers on loopback
    srv = serve(brain, port=8139, block=False)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8139/api/state", timeout=5) as r:
            got = json.loads(r.read().decode())
        assert "circadian" in got, "/api/state did not return brain state"
        with urllib.request.urlopen("http://127.0.0.1:8139/api/spikes", timeout=5) as r:
            sp = json.loads(r.read().decode())
        assert len(sp) == 17, "/api/spikes did not return 17 regions"
    finally:
        srv.shutdown()
        srv.server_close()

    print(
        f"brain_telemetry OK: {len(snap['cognition'])} cognition keys, {len(snap['spikes'])} regions, "
        f"{len(snap['gated'])} gated readouts, contract clean, recorder+HTTP verified"
    )


if __name__ == "__main__":
    main()
