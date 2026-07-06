"""
v2.14 — Desktop control panel: the product's zero-dep tkinter UX (arm + recognition + lake + NL command).

tkinter is Python's standard-library GUI, so this ships the "full UX" with NO new dependency:
one window, a ttk.Notebook with three tabs that drive the exact same verified code paths the
headless demos use —

  1. Arm Control     : a live 2-link arm on a tk.Canvas (forward kinematics from ArmSim), driven
                       ONLY through the safety-gated ArmSim.apply(); collision turns it red; a big
                       red EMERGENCY_STOP freezes it.
  2. Live Recognition: step / auto-play a builder capture's windows through the SAME path live_arm
                       uses — gesture_recognition.classify -> Interpreter -> arm reaction.
  3. Lake Browser    : open a .spk telemetry store, show channels / events / bytes, run a windowed
                       (channel, [t0,t1)) query via spike_telemetry_hub and report "reads X% of file".
  4. NL Command      : type an English command; nl_interface maps it to an allow-listed arm command
                       (driven through the SAME gated ArmSim.apply() as tab 1) or a lake query, or
                       REJECTS it. Zero-dep StubBackend; the backend can only pick a label, never a
                       raw command (an optional on-device AirLLM backend stays CLI-opt-in via run_local).
  5. Core Runner     : run the ENTIRE core project from inside the app — every CORE module's own
                       self-check, in-process (no python subprocess), PASS/FAIL live. Proves the whole
                       core is bundled and runnable in the packaged .exe. Also headless: snn-ui --run-core.

No arm motion bypasses the envelope: every button calls ArmSim.apply(cmd) (reflex / E-STOP /
collision gate untouched). Nothing here reimplements recognition, the interpreter, or the store.

  snn-ui                                        # console entry point (pip install -e .)
  python -m snn_data_lake.control_panel --gui   # launch the window as a module
  python -m snn_data_lake.control_panel --smoke # headless-safe CI self-check (no display needed)
  python -m snn_data_lake.control_panel --run-core  # run the ENTIRE core in-process (headless), exit 0/1
"""
import math
import os
import tkinter as tk
from tkinter import ttk

from snn_data_lake import arm_sim, gesture_recognition, spike_telemetry_hub
from snn_data_lake.interpreter import Interpreter
from snn_data_lake.live_arm import SENSOR_COMMANDS
from snn_data_lake.nl_interface import NLInterface, StubBackend

STEP_DEG = 15                                  # per-click joint increment
DEMO_SPK = os.path.join("data", "telemetry.spk")


# ---- pure (display-free) helpers: unit-tested headlessly by the smoke -----------
def world_to_canvas(x, y, w, h, scale):
    """Map arm-world metres (base at origin, +y up) to integer canvas pixels (+y down).
    Base sits at (w/2, 0.6*h); `scale` = pixels per metre. Deterministic, no Tk needed."""
    ox, oy = w // 2, int(h * 0.6)
    return ox + int(round(x * scale)), oy - int(round(y * scale))


def find_captures():
    """The builder's labeled captures gesture_recognition replays, as existing CSV paths."""
    paths = [os.path.join(gesture_recognition.DIR, f"{g}.csv") for g in gesture_recognition.GESTURES]
    return [p for p in paths if os.path.exists(p)]


def pct_read(bytes_read, total_bytes):
    """Windowed-query efficiency: % of the .spk file a partial read touched."""
    return 100.0 * bytes_read / total_bytes if total_bytes else 0.0


def ensure_demo_store(path=DEMO_SPK):
    """Return a .spk store path, building a tiny deterministic one if none exists (zero random)."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hub = spike_telemetry_hub.SpikeTelemetryHub(8, 1000)
    for c in range(8):
        hub.ingest(c, range(c, 1000, 10 + c))     # regular, sorted, reproducible
    hub.save(path)
    return path


# ---- tab 1: Arm Control ---------------------------------------------------------
class ArmTab(ttk.Frame):
    """2-link arm on a Canvas; every button routes through the safety-gated ArmSim.apply()."""

    W, H, SCALE = 480, 360, 90

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.arm = arm_sim.ArmSim()                # stdlib backend; the gated safety API
        self.canvas = tk.Canvas(self, width=self.W, height=self.H, bg="white",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=0, column=0, rowspan=8, padx=(0, 10))

        btns = [("JOINT_A +", f"JOINT_A_ROTATE(+{STEP_DEG}deg)"),
                ("JOINT_A -", f"JOINT_A_ROTATE(-{STEP_DEG}deg)"),
                ("JOINT_B +", f"JOINT_B_ROTATE(+{STEP_DEG}deg)"),
                ("JOINT_B -", f"JOINT_B_ROTATE(-{STEP_DEG}deg)"),
                ("GRIPPER OPEN", "GRIPPER_OPEN"),
                ("GRIPPER CLOSE", "GRIPPER_CLOSE"),
                ("HOME", "HOME"),
                ("RETRACT_ALL", "RETRACT_ALL")]
        for r, (label, cmd) in enumerate(btns):
            ttk.Button(self, text=label, width=16,
                       command=lambda c=cmd: self.send(c)).grid(row=r, column=1, sticky="ew", pady=1)

        stop = tk.Button(self, text="EMERGENCY_STOP", bg="#c00", fg="white",
                         activebackground="#900", activeforeground="white",
                         font=("TkDefaultFont", 10, "bold"),
                         command=lambda: self.send("EMERGENCY_STOP"))
        stop.grid(row=len(btns), column=1, sticky="ew", pady=(8, 2))

        self.status = ttk.Label(self, text="", font=("TkFixedFont", 9), justify="left")
        self.status.grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.render()

    def send(self, cmd):
        self.arm.apply(cmd)                        # ONLY path to motion -> reflex / E-STOP gate holds
        self.render()

    def render(self):
        c, arm = self.canvas, self.arm
        c.delete("all")
        colliding = arm.collision()
        # floor
        fx0, fy = world_to_canvas(-2.2, arm_sim.FLOOR_Y, self.W, self.H, self.SCALE)
        fx1, _ = world_to_canvas(2.2, arm_sim.FLOOR_Y, self.W, self.H, self.SCALE)
        c.create_line(fx0, fy, fx1, fy, fill="#654321", width=2)
        c.create_text(fx1, fy + 8, text="floor", fill="#654321", anchor="e", font=("TkDefaultFont", 7))
        # obstacle
        ox, oy = arm_sim.OBSTACLE
        px, py = world_to_canvas(ox, oy, self.W, self.H, self.SCALE)
        rr = int(arm_sim.OBSTACLE_R * self.SCALE)
        c.create_oval(px - rr, py - rr, px + rr, py + rr, outline="#c80", width=2, dash=(3, 2))
        c.create_text(px, py, text="obstacle", fill="#c80", font=("TkDefaultFont", 7))
        # links (base -> joint1 -> ee) via forward kinematics
        link = "#c00" if colliding else "#06c"
        j1x = arm.l1 * math.cos(arm.theta[0])
        j1y = arm.l1 * math.sin(arm.theta[0])
        eex, eey = arm.ee()
        p0 = world_to_canvas(0, 0, self.W, self.H, self.SCALE)
        p1 = world_to_canvas(j1x, j1y, self.W, self.H, self.SCALE)
        p2 = world_to_canvas(eex, eey, self.W, self.H, self.SCALE)
        c.create_line(*p0, *p1, fill=link, width=6, capstyle="round")
        c.create_line(*p1, *p2, fill=link, width=6, capstyle="round")
        for (px_, py_) in (p0, p1):
            c.create_oval(px_ - 4, py_ - 4, px_ + 4, py_ + 4, fill="#333")
        # end-effector: gripper open (hollow) vs closed (filled)
        eefill = "#080" if arm.gripper else "white"
        c.create_oval(p2[0] - 6, p2[1] - 6, p2[0] + 6, p2[1] + 6, fill=eefill, outline="#080", width=2)

        s = arm.state()
        grip = "CLOSED" if s["gripper"] else "open"
        self.status.configure(
            text=(f"theta(deg) A={s['theta_deg'][0]:5.1f} B={s['theta_deg'][1]:5.1f}   "
                  f"ee=({s['ee'][0]:+.2f},{s['ee'][1]:+.2f})   gripper={grip}\n"
                  f"stopped={s['stopped']}   collision={s['collision']}   "
                  f"deviation={s['deviation']:.3f} m   backend={s['backend']}"),
            foreground=("#c00" if colliding or s["stopped"] else "#000"))


# ---- tab 2: Live Recognition ----------------------------------------------------
class RecognitionTab(ttk.Frame):
    """Replay a labeled capture through classify -> Interpreter -> arm, exactly as live_arm does."""

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.interp = Interpreter(commands=SENSOR_COMMANDS)   # same map live_arm uses
        self.arm = arm_sim.ArmSim()
        self.windows = []
        self.idx = 0
        self.playing = False

        top = ttk.Frame(self); top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="capture:").pack(side="left")
        self.pick = ttk.Combobox(top, width=34, state="readonly", values=find_captures())
        self.pick.pack(side="left", padx=4)
        if self.pick["values"]:
            self.pick.current(0)
        self.pick.bind("<<ComboboxSelected>>", lambda _e: self.load())
        ttk.Button(top, text="Load", command=self.load).pack(side="left", padx=2)
        ttk.Button(top, text="Step", command=self.step).pack(side="left", padx=2)
        self.play_btn = ttk.Button(top, text="Auto-play", command=self.toggle)
        self.play_btn.pack(side="left", padx=2)

        self.head = ttk.Label(self, text="load a capture to begin", font=("TkFixedFont", 10))
        self.head.grid(row=1, column=0, sticky="w", pady=6)

        self.log = tk.Text(self, width=78, height=16, font=("TkFixedFont", 9), state="disabled")
        self.log.grid(row=2, column=0, sticky="nsew")
        sb = ttk.Scrollbar(self, command=self.log.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)
        if self.pick["values"]:
            self.load()

    def load(self):
        self.playing = False
        self.play_btn.configure(text="Auto-play")
        path = self.pick.get()
        self.arm = arm_sim.ArmSim()
        self.windows = gesture_recognition.windows(gesture_recognition.load_csv(path)) if path else []
        self.idx = 0
        self._logline("=" * 60)
        self._logline(f"loaded {os.path.basename(path)} -> {len(self.windows)} windows")
        self.head.configure(text=f"windows: {len(self.windows)}   (Step or Auto-play)")

    def step(self):
        if self.idx >= len(self.windows):
            self._logline("-- end of capture --")
            self.playing = False
            self.play_btn.configure(text="Auto-play")
            return
        blk = self.windows[self.idx]
        gesture = gesture_recognition.classify(blk)                # recognition (reused)
        cmd, _ = self.interp.interpret({"match": gesture})         # gesture -> command (reused)
        self.arm.apply(cmd)                                        # gated arm reaction
        ee = self.arm.state()["ee"]
        grip = "CLOSED" if self.arm.gripper else "open"
        self.idx += 1
        self.head.configure(text=f"window {self.idx}/{len(self.windows)}   gesture={gesture}   "
                                 f"command={cmd}   ee=({ee[0]:.2f},{ee[1]:.2f})")
        self._logline(f"win {self.idx:3d}  gesture={gesture:<13} cmd={cmd:<13} "
                      f"gripper={grip} ee=({ee[0]:+.2f},{ee[1]:+.2f})")

    def toggle(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Auto-play")
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing:
            return
        self.step()
        if self.idx >= len(self.windows):
            self.playing = False
            self.play_btn.configure(text="Auto-play")
            return
        self.after(400, self._tick)

    def _logline(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


# ---- tab 3: Lake Browser --------------------------------------------------------
class LakeTab(ttk.Frame):
    """Open a .spk store; run a windowed (channel, [t0,t1)) partial read via the telemetry hub."""

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.path = None
        self.nbytes = 0

        top = ttk.Frame(self); top.grid(row=0, column=0, columnspan=6, sticky="ew")
        ttk.Button(top, text="Open / build demo .spk", command=self.open_store).pack(side="left")
        self.summary = ttk.Label(top, text="no store loaded", font=("TkFixedFont", 9))
        self.summary.pack(side="left", padx=10)

        form = ttk.Frame(self); form.grid(row=1, column=0, columnspan=6, sticky="w", pady=6)
        self.e_ch = self._field(form, "channel", "7")
        self.e_t0 = self._field(form, "t0", "0")
        self.e_t1 = self._field(form, "t1", "1000")
        ttk.Button(form, text="Run query", command=self.query).pack(side="left", padx=6)

        self.result = tk.Text(self, width=78, height=18, font=("TkFixedFont", 9), state="disabled")
        self.result.grid(row=2, column=0, columnspan=6, sticky="nsew")
        self.open_store()

    def _field(self, parent, label, default):
        ttk.Label(parent, text=label).pack(side="left")
        e = ttk.Entry(parent, width=8)
        e.insert(0, default)
        e.pack(side="left", padx=(2, 8))
        return e

    def open_store(self):
        self.path = ensure_demo_store()
        self.nbytes = os.path.getsize(self.path)
        hub = spike_telemetry_hub.SpikeTelemetryHub.load(self.path)
        self.summary.configure(text=f"{os.path.basename(self.path)}   channels={hub.n}   "
                                    f"events={hub.n_events()}   bytes={self.nbytes}")
        self._show(f"opened {self.path}: {hub.n} channels, {hub.n_events()} events, "
                   f"{self.nbytes} bytes\nenter channel + [t0,t1) and Run query.")

    def query(self):
        try:
            ch, t0, t1 = int(self.e_ch.get()), int(self.e_t0.get()), int(self.e_t1.get())
        except ValueError:
            self._show("channel / t0 / t1 must be integers")
            return
        try:
            events, read = spike_telemetry_hub.disk_query(self.path, t0, t1, [ch])
        except (IndexError, ValueError) as e:
            self._show(f"query error: {e}")
            return
        head = events[:40]
        lines = [f"query channel={ch} window=[{t0},{t1})",
                 f"events returned : {len(events)}",
                 f"bytes read      : {read}  ({pct_read(read, self.nbytes):.1f}% of the "
                 f"{self.nbytes}-byte file)",
                 "-" * 50,
                 "  ".join(f"t={t}" for t, _c in head) or "(no events in window)"]
        if len(events) > len(head):
            lines.append(f"... ({len(events) - len(head)} more)")
        self._show("\n".join(lines))

    def _show(self, text):
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("end", text)
        self.result.configure(state="disabled")


# ---- tab 4: NL Command ----------------------------------------------------------
class NLTab(ttk.Frame):
    """Type an English command; nl_interface maps it to a SAFE intent that drives the SAME gated
    paths as the other tabs. Arm intents route through the Arm Control tab's ArmSim.apply() (reflex /
    E-STOP / collision gate); lake intents run a windowed disk_query. The backend can only pick an
    allow-listed LABEL, never a raw command. Default backend is the zero-dep StubBackend (a local
    on-device AirLLM backend stays CLI-opt-in: SNN_LLM=1 python local_llm/run_local.py)."""

    def __init__(self, master, armtab):
        super().__init__(master, padding=8)
        self.nl = NLInterface(backend=StubBackend())   # zero-dep, deterministic keyword router
        self.armtab = armtab                           # arm intents drive the SAME gated arm as tab 1

        top = ttk.Frame(self); top.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Label(top, text="command:").pack(side="left")
        self.entry = ttk.Entry(top, width=48)
        self.entry.pack(side="left", padx=4)
        self.entry.bind("<Return>", lambda _e: self.run())
        ttk.Button(top, text="Run", command=self.run).pack(side="left", padx=2)

        ttk.Label(self, foreground="#555", font=("TkDefaultFont", 8),
                  text="examples:  close the gripper   open   rotate joint A   go home   STOP   "
                       "read telemetry channel 7 window 1000").grid(row=1, column=0, columnspan=3,
                                                                    sticky="w", pady=(4, 0))
        ttk.Label(self, foreground="#555", font=("TkDefaultFont", 8),
                  text="backend: StubBackend (zero-dep). Optional on-device AirLLM: "
                       "SNN_LLM=1 python local_llm/run_local.py").grid(row=2, column=0, columnspan=3, sticky="w")

        self.log = tk.Text(self, width=78, height=15, font=("TkFixedFont", 9), state="disabled")
        self.log.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        sb = ttk.Scrollbar(self, command=self.log.yview); sb.grid(row=3, column=2, sticky="ns", pady=(6, 0))
        self.log.configure(yscrollcommand=sb.set)
        self._logline("Type an English command and press Run (or Enter). Arm actions appear on the Arm Control tab.")

    def run(self):
        text = self.entry.get().strip()
        if not text:
            return
        intent = self.nl.interpret(text)               # -> allow-listed arm cmd / lake query / reject
        kind = intent["kind"]
        if kind == "arm":
            self.armtab.send(intent["command"])        # SAME gated apply as tab 1 (reflex / E-STOP / collision)
            s = self.armtab.arm.state()
            self._logline(f"> {text}\n    ARM  {intent['command']}  -> ee=({s['ee'][0]:+.2f},{s['ee'][1]:+.2f}) "
                          f"stopped={s['stopped']} collision={s['collision']}")
        elif kind == "lake":
            self._lake(text, intent["query"])
        else:
            self._logline(f"> {text}\n    REJECTED ({intent.get('reason', '')}) -- no action taken")
        self.entry.delete(0, "end")

    def _lake(self, text, q):
        path = ensure_demo_store()
        chans = q["channels"] or [0]
        t1 = q["window"]
        try:
            events, read = spike_telemetry_hub.disk_query(path, 0, t1, chans)
        except (IndexError, ValueError) as e:
            self._logline(f"> {text}\n    LAKE query error: {e}")
            return
        self._logline(f"> {text}\n    LAKE channels={chans} window=[0,{t1}) -> {len(events)} events, "
                      f"{read} bytes read")

    def _logline(self, text):
        self.log.configure(state="normal"); self.log.insert("end", text + "\n"); self.log.see("end")
        self.log.configure(state="disabled")


# ---- core runner: run the ENTIRE core project in-process (no python subprocess) --
def core_modules():
    """The canonical CORE self-check list (single source of truth: run_selfchecks.CORE),
    minus control_panel itself (this IS the running app). run_selfchecks is bundled into the
    .exe so the whole core is runnable from the packaged app, not just from a source tree."""
    try:
        from run_selfchecks import CORE
    except Exception:
        return []
    return [m for m in CORE if m != "control_panel"]


def run_core_selfcheck(mod):
    """Run one CORE module's own self-check IN-PROCESS (its __main__ block, via runpy) and
    capture output. Returns (ok, text). No subprocess -> works inside the frozen .exe. sys.argv
    is isolated to [mod] so a module sees no stray app flags (mimics `python -m ...`)."""
    import io
    import runpy
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = [mod]
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
        return True, buf.getvalue()
    except SystemExit as e:                            # a module may sys.exit(0) on success
        return (e.code in (0, None)), buf.getvalue()
    except BaseException as e:                          # AssertionError / any self-check failure
        return False, f"{buf.getvalue()}\n{type(e).__name__}: {e}"
    finally:
        sys.argv = saved_argv


def run_all_core(emit=print):
    """Run EVERY core module's self-check in-process; emit 'PASS/FAIL <mod>' + a summary.
    Returns True iff all passed. Backs both the --run-core CLI and the Core Runner tab."""
    mods = core_modules()
    failed = [m for m in mods if not _emit_result(m, emit)]
    emit("=" * 56)
    emit(f"{len(mods) - len(failed)}/{len(mods)} core self-checks passed"
         + (f"   FAILED: {', '.join(failed)}" if failed else ""))
    return not failed


def _emit_result(mod, emit):
    ok, _out = run_core_selfcheck(mod)
    emit(f"{'PASS' if ok else 'FAIL'}  {mod}")
    return ok


# ---- tab 5: Core Runner ---------------------------------------------------------
class CoreTab(ttk.Frame):
    """Run the ENTIRE core project from inside the app: every CORE module's own self-check,
    in-process, PASS/FAIL live. Proves the whole core is bundled and runnable in the packaged
    .exe (no python interpreter / subprocess needed)."""

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.mods = core_modules()
        top = ttk.Frame(self); top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.run_btn = ttk.Button(top, text=f"Run all {len(self.mods)} core self-checks", command=self.run_all)
        self.run_btn.pack(side="left")
        self.summary = ttk.Label(top, text=f"{len(self.mods)} core modules", font=("TkFixedFont", 9))
        self.summary.pack(side="left", padx=10)

        self.log = tk.Text(self, width=78, height=20, font=("TkFixedFont", 9), state="disabled")
        self.log.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        sb = ttk.Scrollbar(self, command=self.log.yview); sb.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        self.log.configure(yscrollcommand=sb.set)
        self.rowconfigure(1, weight=1); self.columnconfigure(0, weight=1)
        self._logline(f"{len(self.mods)} core modules discovered. "
                      "Press Run to execute the entire core in-process.")

    def run_all(self):
        self._queue = list(self.mods); self._passed = 0; self._failed = []
        self.run_btn.configure(state="disabled")
        self._clear(); self._logline("=" * 60)
        self._logline(f"running {len(self._queue)} core self-checks in-process...")
        self.after(1, self._next)

    def _next(self):
        if not self._queue:
            self.run_btn.configure(state="normal")
            self._logline("=" * 60)
            self._logline(f"{self._passed}/{len(self.mods)} core self-checks passed"
                          + (f"   FAILED: {', '.join(self._failed)}" if self._failed else ""))
            return
        mod = self._queue.pop(0)
        ok, _out = run_core_selfcheck(mod)
        if ok:
            self._passed += 1
        else:
            self._failed.append(mod)
        self._logline(f"{'PASS' if ok else 'FAIL'}  {mod}")
        self.summary.configure(text=f"{self._passed} passed, {len(self._failed)} failed, {len(self._queue)} left",
                               foreground=("#c00" if self._failed else "#080"))
        self.after(1, self._next)

    def _clear(self):
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")

    def _logline(self, text):
        self.log.configure(state="normal"); self.log.insert("end", text + "\n"); self.log.see("end")
        self.log.configure(state="disabled")


# ---- app ------------------------------------------------------------------------
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spiking Neural Data Lake — Control Panel")
        nb = ttk.Notebook(self.root)
        arm_tab = ArmTab(nb)                           # kept: the NL tab drives this same gated arm
        nb.add(arm_tab, text="Arm Control")
        nb.add(RecognitionTab(nb), text="Live Recognition")
        nb.add(LakeTab(nb), text="Lake Browser")
        nb.add(NLTab(nb, arm_tab), text="NL Command")
        nb.add(CoreTab(nb), text="Core Runner")
        nb.pack(fill="both", expand=True)

    def mainloop(self):
        self.root.mainloop()


def _smoke():
    """Headless-safe self-check: assert the display-free logic, then TRY the GUI and skip if headless."""
    # (a) non-GUI logic that needs no display -----------------------------------
    w, h, scale = 480, 360, 90
    assert world_to_canvas(0, 0, w, h, scale) == (w // 2, int(h * 0.6)), "base must map to canvas origin"
    assert world_to_canvas(1, 0, w, h, scale) == (w // 2 + scale, int(h * 0.6)), "+x moves right"
    assert world_to_canvas(0, 1, w, h, scale) == (w // 2, int(h * 0.6) - scale), "+y moves up (flipped)"
    caps = find_captures()
    assert isinstance(caps, list) and all(os.path.exists(p) for p in caps), "capture discovery broken"
    assert pct_read(100, 1000) == 10.0 and pct_read(5, 0) == 0.0, "lake percent-read math wrong"

    path = ensure_demo_store()
    hub = spike_telemetry_hub.SpikeTelemetryHub.load(path)
    events, read = spike_telemetry_hub.disk_query(path, 0, hub.dur, [0])
    assert read <= os.path.getsize(path), "partial read cannot exceed the file"
    assert all(0 <= t < hub.dur for t, _c in events), "windowed query returned out-of-window events"

    # command strings the buttons emit must be ones the gated apply() actually acts on
    arm = arm_sim.ArmSim()
    arm.apply(f"JOINT_A_ROTATE(+{STEP_DEG}deg)")
    assert arm.state()["theta_deg"][0] == float(STEP_DEG), "arm button command is not gated-apply valid"

    # NL Command tab (display-free): an English phrase maps to a gated-apply-valid command; unsafe -> reject
    nlx = NLInterface(backend=StubBackend())
    assert nlx.interpret("close the gripper")["command"] == "GRIPPER_CLOSE", "NL 'close' phrase mis-mapped"
    assert nlx.interpret("delete everything")["kind"] == "reject", "unsafe NL phrase must be rejected"
    arm2 = arm_sim.ArmSim(); arm2.apply(nlx.interpret("close the gripper")["command"])
    assert arm2.state()["gripper"], "an NL-mapped command must drive the gated arm"

    # Core Runner (display-free): the whole core is discoverable + runnable in-process
    core = core_modules()
    assert core and "control_panel" not in core, "core module list empty or includes the app itself"
    ok, _out = run_core_selfcheck("lif_neuron")
    assert ok, "a core module must run to completion in-process (lif_neuron)"

    # (b) TRY to construct the whole GUI, then destroy WITHOUT mainloop ----------
    try:
        app = App()
        app.root.update_idletasks()
        app.root.destroy()
        mode = "GUI constructed + destroyed (display present)"
    except tk.TclError:
        mode = "GUI construction skipped (no display)"
        print("control_panel smoke: GUI construction skipped (no display)")

    print("self-check OK: canvas FK transform, capture discovery, lake partial-read stats, "
          f"gated-apply commands, NL command routing, core runner ({len(core)} modules); {mode}")


def main():
    """snn-ui entry point: launch the desktop app (or --smoke for the headless self-check)."""
    import sys
    if "--smoke" in sys.argv:
        _smoke()
        return
    if "--run-core" in sys.argv:                    # run the ENTIRE core in-process (headless) -> exit 0/1
        sys.exit(0 if run_all_core() else 1)
    try:
        App().mainloop()
    except tk.TclError as e:
        print(f"control_panel: no display available ({e}). "
              "Run with --smoke for the headless self-check.")


if __name__ == "__main__":
    import sys
    # `python -m snn_data_lake.control_panel` (what run_selfchecks.py / CI runs, no args) defaults
    # to the headless-safe self-check so the runner never blocks on a window; a recognised flag
    # (--gui / --smoke / --run-core) routes to main() instead.
    if any(f in sys.argv for f in ("--gui", "--smoke", "--run-core")):
        main()
    else:
        _smoke()
