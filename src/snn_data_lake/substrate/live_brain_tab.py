"""
Live Brain tab -- drive and OBSERVE the integrated brain live, from the desktop UI.

A 6th control-panel tab (a ttk.Frame that the orchestrator adds to the existing notebook) that
runs ONE LiveBrain (the whole stack: Global Workspace + amygdala threat + basal-ganglia gate +
interoceptive arousal + prefrontal executive, with the data lake AS working memory) and lets the
user interact with it: feed the multimodal context, reward an action so the prefrontal executive
forms a GOAL, raise the body's homeostatic load, or fire a DANGER context that trips the reflex
E-STOP. Every panel refreshes each tick from the brain's live functional state.

SAFETY (non-negotiable): the arm is driven ONLY through LiveBrain.tick() (the gated arm loop). The
DANGER button just supplies a close-distance context so the reflex EMERGENCY_STOP fires -- there is
NO direct actuator path here, and this UI reuses (never re-derives) the forward kinematics.

This shows FUNCTIONAL correlates of the running integration. It does NOT assert or imply the brain
is conscious or sentient -- the Global Workspace panel is a functional global-broadcast (GWT)
readout, not phenomenal consciousness.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.live_brain_tab
"""

import math
import tkinter as tk
from tkinter import scrolledtext, ttk

from snn_data_lake.brainstem import arm_sim
from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.autonomous_loop import ACTIONS
from snn_data_lake.cortex.live_brain import LiveBrain
from snn_data_lake.substrate.control_panel import world_to_canvas

WS_HEADER = "Global Workspace (functional global-broadcast, GWT) - not phenomenal consciousness"
SAFE_D = CONTACT * 3.0  # a comfortably-in-range distance (no reflex)
DANGER_D = CONTACT * 0.5  # a close-contact distance -> trips the reflex E-STOP


def _ctx(v, tmp, au, d):
    """The context dict LiveBrain.tick() expects (a tiny local copy of live_brain._ctx)."""
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


class LiveBrainTab(ttk.Frame):
    """Run + observe a LiveBrain live: feed context, reward a goal, raise load, or fire danger;
    watch the Global Workspace broadcast, the lake-backed working memory, attention/affect, and the
    gated arm. Drives the arm ONLY through LiveBrain.tick() -- no new actuator path."""

    W, H, SCALE = 380, 300, 70  # canvas geometry (reach l1+l2=2m fits at scale 70)

    def __init__(self, master, seed=0):
        super().__init__(master, padding=8)
        self._seed = seed
        self.brain = LiveBrain(seed=seed)
        self.playing = False
        self.ignitions = 0  # count of Global-Workspace broadcast changes
        self._prev_thought = None
        self._last_action = "(none)"

        self._build_controls()
        self._build_arm_panel()
        self._build_state_panels()
        self._build_inputs()
        self._build_chat()
        self.bind("<Destroy>", self._on_destroy)
        self._refresh()

    # ---- widget construction ----------------------------------------------------
    def _build_controls(self):
        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(top, text="Step", command=self.step_once).pack(side="left")
        self.play_btn = ttk.Button(top, text="Auto-play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=2)
        ttk.Button(top, text="Pause", command=self.pause).pack(side="left", padx=2)
        ttk.Button(top, text="Reset", command=self.reset).pack(side="left", padx=2)
        tk.Button(
            top,
            text="DANGER",
            bg="#c00",
            fg="white",
            activebackground="#900",
            activeforeground="white",
            font=("TkDefaultFont", 9, "bold"),
            command=self.danger,
        ).pack(side="left", padx=(10, 2))

        ttk.Label(top, text="reward:").pack(side="left", padx=(10, 0))
        self.reward_var = tk.StringVar(value="(no reward)")
        rc = ttk.Combobox(
            top,
            width=14,
            state="readonly",
            textvariable=self.reward_var,
            values=["(no reward)", *ACTIONS],
        )
        rc.pack(side="left", padx=2)
        ttk.Label(
            top,
            text="(+1 for it, -1 else -> PFC forms a goal)",
            foreground="#555",
            font=("TkDefaultFont", 8),
        ).pack(side="left")

    def _build_arm_panel(self):
        armf = ttk.LabelFrame(self, text="Arm (2-link, gated) - driven only via LiveBrain.tick()")
        armf.grid(row=1, column=0, sticky="nw", padx=(0, 8))
        self.canvas = tk.Canvas(
            armf,
            width=self.W,
            height=self.H,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888",
        )
        self.canvas.pack()
        self.arm_status = ttk.Label(armf, text="", font=("TkFixedFont", 9), justify="left")
        self.arm_status.pack(anchor="w", pady=(4, 0))

    def _build_state_panels(self):
        right = ttk.Frame(self)
        right.grid(row=1, column=1, sticky="nw")

        wsf = ttk.LabelFrame(right, text=WS_HEADER)
        wsf.pack(fill="x", pady=(0, 6))
        self.ws_label = ttk.Label(wsf, text="", font=("TkFixedFont", 9), justify="left")
        self.ws_label.pack(anchor="w")
        self.ws_count = ttk.Label(wsf, text="", font=("TkFixedFont", 9))
        self.ws_count.pack(anchor="w")

        wmf = ttk.LabelFrame(right, text="Working memory (READ BACK FROM THE DATA LAKE)")
        wmf.pack(fill="x", pady=(0, 6))
        self.wm_goal = ttk.Label(wmf, text="", font=("TkFixedFont", 9))
        self.wm_goal.pack(anchor="w")
        self.wm_events = ttk.Label(wmf, text="", font=("TkFixedFont", 9))
        self.wm_events.pack(anchor="w")
        self.wm_recall = ttk.Label(
            wmf, text="", font=("TkFixedFont", 9), justify="left", wraplength=360
        )
        self.wm_recall.pack(anchor="w")

        aff = ttk.LabelFrame(right, text="Attention / affect")
        aff.pack(fill="x")
        self.pb_threat, self.lb_threat = self._labeled_bar(aff, 0, "amygdala threat")
        self.pb_intero, self.lb_intero = self._labeled_bar(aff, 1, "interoceptive caution")
        self.pb_cort, self.lb_cort = self._labeled_bar(aff, 2, "cortisol caution")

    def _labeled_bar(self, parent, row, name):
        ttk.Label(parent, text=name, font=("TkFixedFont", 9), width=22, anchor="w").grid(
            row=row, column=0, sticky="w", pady=1
        )
        pb = ttk.Progressbar(parent, length=140, maximum=1.0, mode="determinate")
        pb.grid(row=row, column=1, padx=4)
        lb = ttk.Label(parent, text="0.00", font=("TkFixedFont", 9), width=6)
        lb.grid(row=row, column=2, sticky="w")
        return pb, lb

    def _build_inputs(self):
        inf = ttk.LabelFrame(
            self, text="Context inputs (everything that drives the functional signals)"
        )
        inf.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.v_visual = self._slider(inf, 0, "visual", 0.0, 1.0, 0.9)
        self.v_temp = self._slider(inf, 1, "temperature", 0.0, 1.0, 0.1)
        self.v_audio = self._slider(inf, 2, "audio", 0.0, 1.0, 0.1)
        self.v_dist = self._slider(
            inf, 3, f"distance (< {CONTACT:.2f} m -> danger)", 0.0, 1.0, 0.30
        )
        self.v_load = self._slider(inf, 4, "homeostatic load", 0.0, 2.0, 0.0)

    def _slider(self, parent, row, label, frm, to, default):
        ttk.Label(parent, text=label, font=("TkFixedFont", 9), width=26, anchor="w").grid(
            row=row, column=0, sticky="w", pady=1
        )
        var = tk.DoubleVar(value=default)
        val = ttk.Label(parent, text=f"{default:.2f}", font=("TkFixedFont", 9), width=6)
        ttk.Scale(
            parent,
            from_=frm,
            to=to,
            variable=var,
            length=260,
            command=lambda v, lb=val: lb.configure(text=f"{float(v):.2f}"),
        ).grid(row=row, column=1, padx=4, sticky="ew")
        val.grid(row=row, column=2, sticky="w")
        return var

    def _build_chat(self):
        """A chat panel: type a line, the brain answers -- GROUNDED in its real state (goal, memory,
        workspace, safety) and honest about consciousness; arm commands go through the gated arm. Not
        a general chatbot / language model."""
        chatf = ttk.LabelFrame(
            self,
            text="Chat with the brain (grounded in its real state -- a functional model, not a chatbot)",
        )
        chatf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        chatf.columnconfigure(0, weight=1)
        self.chat_log = scrolledtext.ScrolledText(
            chatf, height=7, wrap="word", font=("TkFixedFont", 9), state="disabled"
        )
        self.chat_log.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self.chat_entry = ttk.Entry(chatf)
        self.chat_entry.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        self.chat_entry.bind("<Return>", lambda _e: self._send_chat())
        ttk.Button(chatf, text="Send", command=self._send_chat).grid(
            row=1, column=1, padx=4, pady=(0, 4)
        )
        self._chat_say(
            "brain",
            "Hi. Try 'hello', 'what is your goal', 'what do you remember', "
            "'are you conscious', or a command like 'close the effector'.",
        )

    def _chat_say(self, who, text):
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"{who}: {text}\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def _send_chat(self):
        text = self.chat_entry.get().strip()
        if not text:
            return
        self.chat_entry.delete(0, "end")
        self._chat_say("you", text)
        self._chat_say(
            "brain", self.brain.converse(text)
        )  # grounded; a command goes through the gated arm
        self._refresh()  # a chat command may have moved the gated arm -> repaint

    # ---- driving the brain ------------------------------------------------------
    def _reward_fn(self):
        """A reward_fn that returns +1 for the selected action, -1 otherwise (None when off)."""
        target = self.reward_var.get()
        if target not in ACTIONS:
            return None

        def rf(_unit, action, _t=target):
            return 1.0 if action == _t else -1.0

        return rf

    def step_once(self):
        """One live tick over the current slider context (through the gated arm)."""
        ctx = _ctx(self.v_visual.get(), self.v_temp.get(), self.v_audio.get(), self.v_dist.get())
        return self._tick(ctx)

    def danger(self):
        """One tick with a close-contact context -> the reflex E-STOP fires (no direct actuator path)."""
        self.pause()
        return self._tick(_ctx(0.5, 0.5, 0.5, DANGER_D))

    def _tick(self, ctx):
        res = self.brain.tick(ctx, reward_fn=self._reward_fn(), homeostatic_load=self.v_load.get())
        self._last_action = res.get("action", "(none)")
        thought = self.brain.thought
        if thought != self._prev_thought:  # a workspace ignition / broadcast change
            self.ignitions += 1
        self._prev_thought = thought
        self._refresh(res)
        return res

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Auto-play")
        if self.playing:
            self._auto()

    def _auto(self):
        if not self.playing:
            return
        self.step_once()
        self.after(500, self._auto)  # same self.after() timer the other tabs use

    def pause(self):
        self.playing = False
        self.play_btn.configure(text="Auto-play")

    def reset(self):
        """Start a fresh LiveBrain (cleaning up the old lake temp file first)."""
        self.pause()
        self._close_lake()
        self.brain = LiveBrain(seed=self._seed)
        self.ignitions = 0
        self._prev_thought = None
        self._last_action = "(none)"
        self._refresh()

    # ---- rendering --------------------------------------------------------------
    def _refresh(self, res=None):
        res = res or {}
        state = res.get("state") or self.brain.arm.arm.state()

        # Global Workspace -- the current broadcast content + a running ignition count.
        self.ws_label.configure(text=f"thought (broadcast content): {self.brain.thought!r}")
        self.ws_count.configure(text=f"ignition / broadcast changes: {self.ignitions}")

        # Working memory -- read BACK from the data lake (lake-backed WM).
        lw = self.brain.arm.lake_wm
        n_events = lw.n_events() if lw is not None else 0
        recent = self.brain.recall(6)
        compact = " ".join(f"t{t}:{len(ch)}" for t, ch in recent) or "(empty)"
        goal = self.brain.goal
        goal_txt = ACTIONS[goal] if goal is not None else "(none)"
        self.wm_goal.configure(text=f"PFC goal: {goal_txt}")
        self.wm_events.configure(text=f"lake WM spikes stored: {n_events}")
        self.wm_recall.configure(text=f"recall(6) from lake (tick:active-channels): {compact}")

        # Attention / affect -- per-tick threat + interoceptive caution + cortisol.
        threat = float(res.get("threat") or 0.0)
        intero = float(res.get("intero") or 0.0)
        cortisol = float(self.brain.arm.cort.caution_bias())
        self._set_bar(self.pb_threat, self.lb_threat, threat)
        self._set_bar(self.pb_intero, self.lb_intero, intero)
        self._set_bar(self.pb_cort, self.lb_cort, cortisol)

        self._render_arm(state)

    @staticmethod
    def _set_bar(pb, lb, value):
        pb.configure(value=max(0.0, min(1.0, value)))
        lb.configure(text=f"{value:.2f}")

    def _render_arm(self, state):
        """Draw the 2-link arm from the gated ArmSim -- REUSING world_to_canvas + the ArmTab render
        approach (no re-derived FK). Red when the arm is E-STOPped or colliding."""
        c, arm = self.canvas, self.brain.arm.arm  # geometry from the gated sim
        c.delete("all")
        colliding = arm.collision()
        stopped = bool(state.get("stopped"))
        # floor
        fx0, fy = world_to_canvas(-2.2, arm_sim.FLOOR_Y, self.W, self.H, self.SCALE)
        fx1, _ = world_to_canvas(2.2, arm_sim.FLOOR_Y, self.W, self.H, self.SCALE)
        c.create_line(fx0, fy, fx1, fy, fill="#654321", width=2)
        # obstacle
        ox, oy = arm_sim.OBSTACLE
        px, py = world_to_canvas(ox, oy, self.W, self.H, self.SCALE)
        rr = int(arm_sim.OBSTACLE_R * self.SCALE)
        c.create_oval(px - rr, py - rr, px + rr, py + rr, outline="#c80", width=2, dash=(3, 2))
        c.create_text(px, py, text="obstacle", fill="#c80", font=("TkDefaultFont", 7))
        # links (base -> joint1 -> ee) via forward kinematics (reused from the sim)
        link = "#c00" if (colliding or stopped) else "#06c"
        j1x = arm.l1 * math.cos(arm.theta[0])
        j1y = arm.l1 * math.sin(arm.theta[0])
        eex, eey = arm.ee()
        p0 = world_to_canvas(0, 0, self.W, self.H, self.SCALE)
        p1 = world_to_canvas(j1x, j1y, self.W, self.H, self.SCALE)
        p2 = world_to_canvas(eex, eey, self.W, self.H, self.SCALE)
        c.create_line(*p0, *p1, fill=link, width=6, capstyle="round")
        c.create_line(*p1, *p2, fill=link, width=6, capstyle="round")
        for px_, py_ in (p0, p1):
            c.create_oval(px_ - 4, py_ - 4, px_ + 4, py_ + 4, fill="#333")
        eefill = "#080" if arm.effector else "white"
        c.create_oval(
            p2[0] - 6, p2[1] - 6, p2[0] + 6, p2[1] + 6, fill=eefill, outline="#080", width=2
        )
        if stopped:
            c.create_text(
                self.W // 2, 14, text="E-STOP", fill="#c00", font=("TkDefaultFont", 12, "bold")
            )

        grip = "CLOSED" if state.get("effector") else "open"
        self.arm_status.configure(
            text=(
                f"theta(deg) A={state['theta_deg'][0]:5.1f} B={state['theta_deg'][1]:5.1f}  "
                f"ee=({state['ee'][0]:+.2f},{state['ee'][1]:+.2f})  effector={grip}\n"
                f"action={self._last_action}  stopped={state['stopped']}  "
                f"collision={state['collision']}"
            ),
            foreground=("#c00" if (colliding or stopped) else "#000"),
        )

    # ---- cleanup ----------------------------------------------------------------
    def _close_lake(self, remove=True):
        lw = self.brain.arm.lake_wm
        if lw is not None:
            try:
                lw.close(remove=remove)
            except Exception:
                pass

    def close(self, remove=True):
        """Stop auto-play and release/remove this brain's lake temp file (no litter)."""
        self.playing = False
        self._close_lake(remove=remove)

    def _on_destroy(self, event=None):
        if event is not None and event.widget is not self:  # ignore child-widget destroys
            return
        self.close(remove=True)


def smoke():
    """Headless-safe self-check: assert the display-free live-brain logic, then TRY the GUI + tab
    and skip on no display -- leaving no lake temp-file litter."""
    grasp = _ctx(0.9, 0.1, 0.1, SAFE_D)

    def reward(_unit, action):
        return 1.0 if action == "RELEASE" else -1.0

    # (a) the brain LEARNS a goal + stores working memory in the data lake ---------
    brain = LiveBrain(seed=1)
    for _ in range(90):
        brain.tick(grasp, reward_fn=reward)
    assert brain.alive(), "the live brain must wire in every region"
    assert brain.goal == ACTIONS.index("RELEASE"), (
        f"the PFC must form the rewarded goal ({ACTIONS[brain.goal] if brain.goal is not None else None})"
    )
    assert brain.arm.lake_wm is not None and brain.arm.lake_wm.n_events() > 0, (
        "the working memory must be recorded in the data lake"
    )
    wm = brain.recall(6)
    assert 0 < len(wm) <= 6, "recall must return a bounded window read back from the lake"

    # (b) DANGER: a close context trips the reflex E-STOP and freezes the arm ------
    guard = LiveBrain(seed=2)
    ev = guard.tick(_ctx(0.5, 0.5, 0.5, DANGER_D))
    assert ev["action"] == "EMERGENCY_STOP" and ev["reflex"], (
        "a danger context must trip the reflex E-STOP with the whole brain live"
    )
    assert guard.arm.arm.state()["stopped"], "the reflex E-STOP must freeze the arm"

    # (c) determinism: same seed -> identical behaviour ---------------------------
    def run(seed):
        b = LiveBrain(seed=seed)
        acts = [b.tick(grasp, reward_fn=reward)["action"] for _ in range(12)]
        b.arm.lake_wm.close(remove=True)
        return acts

    assert run(5) == run(5), "the live brain must be deterministic"

    # (c2) chat: the brain answers GROUNDED in its real state, and never claims consciousness ------
    _g = brain.converse("what is your goal?")
    assert "improving myself" in _g.lower() and "RELEASE" in _g, (
        "chat must report the standing telos + sub-goal"
    )
    _c = brain.converse("are you conscious?")
    assert "neither claim" in _c and "not a conscious agent" not in _c, (
        "chat must answer consciousness AGNOSTICALLY (report correlates, neither claim nor deny), not DENY it"
    )

    brain.arm.lake_wm.close(remove=True)  # clean up the temp lake files
    if guard.arm.lake_wm is not None:
        guard.arm.lake_wm.close(remove=True)

    # (d) TRY to construct the GUI + tab, then destroy WITHOUT mainloop -----------
    root = None
    try:
        root = tk.Tk()
        tab = LiveBrainTab(root)
        root.update_idletasks()
        tab.close(remove=True)
        mode = "GUI constructed + destroyed (display present)"
    except tk.TclError:
        mode = "GUI construction skipped (no display)"
        print("live_brain_tab smoke: GUI construction skipped (no display)")
    finally:
        if root is not None:
            root.destroy()

    print(
        f"self-check OK: live brain forms the prefrontal goal {ACTIONS[brain.goal]} (rewarded); "
        f"working memory read back from the data lake ({len(wm)} recent ticks, bounded); "
        f"chat answers grounded in real state (honest about consciousness); "
        f"danger -> gated E-STOP + freeze holds; deterministic; {mode}"
    )


if __name__ == "__main__":
    smoke()
