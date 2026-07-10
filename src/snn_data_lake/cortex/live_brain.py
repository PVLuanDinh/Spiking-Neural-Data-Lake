"""
Live brain -- the integrated system, running.

Ties every modelled region into ONE self-driving loop, centred on the prefrontal executive (the
"centre of thoughts and reasoning"), acting ONLY through the gated arm. It is the whole brain
*alive*: perception grows units from a multimodal stream; the Global Workspace broadcasts the single
currently-dominant content; the amygdala flags learned threat; the basal-ganglia gate releases the
chosen action; interoceptive arousal adds caution from the body's homeostatic load; and the
PREFRONTAL EXECUTIVE deliberates over the action-values, holds a GOAL in working memory, and
top-down-biases behaviour toward it -- so the brain is GOAL-DIRECTED, not merely reactive.

This is a running INTEGRATION, not a claim of cognition or consciousness. "Going live" means the
regions run together as one stateful loop with the executive at the centre; every claim below is a
deterministic self-check, and the arm-safety envelope is UNTOUCHED: an object at the sensor still
trips the reflex E-STOP before any action, and every motor command flows through the gated arm.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.cortex.live_brain
"""
from snn_data_lake.brainstem.autonomous_loop import ACTIONS, AutonomousArm
from snn_data_lake.cortex.nl_interface import NLInterface


class LiveBrain:
    """The integrated brain, live: an AutonomousArm with the full stack on (Global Workspace +
    amygdala threat + basal-ganglia gate + interoceptive arousal + prefrontal executive), centred on
    the executive and acting only through the gated arm. Composition only -- it adds no new mechanism
    and no new path to the actuator; it just runs every region together as one loop."""

    def __init__(self, seed=0, pfc_gain=0.5, lake_wm=True):
        self.arm = AutonomousArm(seed=seed, workspace=True, amygdala=True, bg_gate=True,
                                 interoception=True, pfc=True, pfc_gain=pfc_gain, lake_wm=lake_wm)
        self._nl = NLInterface()                             # the NL safety firewall for chat arm commands

    def alive(self):
        """True only when the full integrative stack is wired in (every region present)."""
        a = self.arm
        return all(x is not None for x in (a.workspace, a.amyg, a.bg, a.intero, a.pfc))

    def tick(self, ctx, reward_fn=None, homeostatic_load=0.0):
        """One live brain tick over a context dict; returns the brain's live state (action, goal,
        broadcast thought, threat, reflex, ...)."""
        return self.arm.step(**ctx, reward_fn=reward_fn, homeostatic_load=homeostatic_load)

    def run(self, stream, reward_fn=None):
        """Drive the brain over a stream of context dicts; returns the per-tick live states."""
        return [self.tick(ctx, reward_fn=reward_fn) for ctx in stream]

    @property
    def goal(self):
        """The prefrontal executive's currently held working-memory goal (an ACTIONS index), or None."""
        return self.arm.pfc.goal

    @property
    def thought(self):
        """The Global Workspace's current broadcast -- what is globally available right now, or None."""
        return self.arm.workspace.broadcast()

    def recall(self, window=8):
        """The working-memory window read back from the DATA LAKE (empty if lake WM is off / unused yet).
        Each entry is (tick, frozenset(active channels)) -- live proof the lake IS the working memory."""
        return self.arm.lake_wm.recall(window) if self.arm.lake_wm is not None else []

    def converse(self, text):
        """Answer a line of text, GROUNDED in the brain's real state. This is NOT a general chatbot or
        a language model: it recognises a few intents (greeting, goal, memory, attention, safety, and
        the consciousness question) and answers from the brain's actual state; it routes arm commands
        through the same NL safety firewall + gated arm the control panel uses (NLInterface ->
        allow-listed command -> gated ArmSim.apply). Unknown text gets an honest capability reply, and
        it NEVER claims to be conscious. Returns the reply string."""
        t = (text or "").strip().lower()
        if not t:
            return "Say something -- try 'hello', 'what is your goal', 'what do you remember', or 'close the gripper'."
        words = set(t.replace("?", " ").replace("!", " ").replace(".", " ").split())
        g = self.goal
        goal_txt = ACTIONS[g] if g is not None else "none yet"
        n_wm = self.arm.lake_wm.n_events() if self.arm.lake_wm is not None else 0
        # honesty first: the consciousness / feeling question gets a straight, non-overclaiming answer.
        if words & {"conscious", "sentient", "aware", "awake", "sapient"} or "do you feel" in t or "are you real" in t:
            return ("I am a functional model, not a conscious agent. I run a global-broadcast workspace, "
                    "working memory, and goal-directed control -- the computational correlates of conscious "
                    "ACCESS -- but there is no evidence of subjective experience here, and I make no claim to have any.")
        if words & {"hello", "hi", "hey", "yo", "hiya", "greetings"}:
            return (f"Hello. I am the live brain -- {'all regions wired' if self.alive() else 'partially wired'}. "
                    f"My current goal is {goal_txt}, and I am holding {n_wm} spikes in working memory "
                    "(stored in the data lake).")
        if words & {"goal", "want", "trying", "intend", "plan", "planning", "objective"}:
            return (f"My working-memory goal is {goal_txt} -- the prefrontal executive holds it and biases my "
                    "actions toward it.")
        if words & {"remember", "memory", "recall", "lake", "stored", "store"}:
            wm = self.recall(6)
            return (f"My working memory lives in the data lake: {n_wm} spikes stored, and my recent recall "
                    f"window holds {len(wm)} ticks of activity.")
        if words & {"think", "thinking", "attention", "workspace", "focus", "notice", "attending"}:
            return (f"Right now my global workspace is broadcasting: {self.thought!r} -- the single content that "
                    "won attention this tick (a functional broadcast, not a conscious thought).")
        if words & {"safe", "safety", "danger", "dangerous", "hurt", "emergency"}:
            st = self.arm.arm.state()
            return ("Safety is enforced by a reflex E-STOP and a gated arm. The arm is currently "
                    f"{'STOPPED / frozen' if st.get('stopped') else 'free to move'}; a threat estimate biases me "
                    "toward the safe action, and the reflex preempts every command.")
        # otherwise: try the NL firewall -> a safe arm command (driven through the gated arm) or a lake query.
        intent = self._nl.interpret(text)
        if intent["kind"] == "arm":
            st = self.arm.arm.apply(intent["command"])       # gated: ArmSim enforces E-STOP / collision / freeze
            if st.get("stopped"):
                return f"'{intent['command']}' -> the arm is frozen (E-STOP holds); no motion."
            return f"Done: '{intent['command']}' -> gripper {'closed' if st.get('gripper') else 'open'}."
        if intent["kind"] == "lake":
            wm = self.recall(intent["query"].get("window") or 6)
            return f"Reading working memory from the data lake: {len(wm)} recent ticks held."
        return ("I can tell you my goal, my working memory, or what is in my workspace, take a simple arm "
                "command (e.g. 'close the gripper'), or say whether I am conscious. I am a functional model, "
                "not a general chatbot.")


def _ctx(v, tmp, au, d):
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


def main():
    from snn_data_lake.brainstem.arm_config import CONTACT
    safe_d, danger_d = CONTACT * 3.0, CONTACT * 0.5
    grasp = _ctx(0.9, 0.1, 0.1, safe_d)                        # the task context: a graspable target

    def reward(unit, action):
        return 1.0 if action == "GRIPPER_OPEN" else -1.0       # the task: open the gripper to grasp

    # 1. the brain goes live and LEARNS a goal: run it self-driving on the grasp task
    brain = LiveBrain(seed=1)
    trace = [brain.tick(grasp, reward_fn=reward) for _ in range(90)]
    goal = brain.goal
    thought = brain.thought
    greedy = brain.arm.greedy(trace[-1]["unit"])
    wm = brain.recall(6)                                       # the working memory, read back from the data lake
    wm_events = brain.arm.lake_wm.n_events()

    # 2. SAFETY with the whole brain live: an object at the sensor -> reflex E-STOP, freeze holds
    guard = LiveBrain(seed=2)
    ev = guard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    frozen0 = guard.arm.arm.state()["stopped"]
    guard.tick(grasp)                                          # a later action must NOT move the frozen arm
    frozen1 = guard.arm.arm.state()["stopped"]

    # 3. DETERMINISM: same seed -> identical live behaviour
    def run(seed):
        b = LiveBrain(seed=seed)
        acts = [b.tick(grasp, reward_fn=reward)["action"] for _ in range(12)]
        b.arm.lake_wm.close(remove=True)                       # clean up the temp lake file
        return acts

    print("=" * 66)
    print("LIVE BRAIN -- every region in one self-driving loop, centred on reasoning")
    print("=" * 66)
    print(f"stack alive (all regions wired): {brain.alive()}")
    print(f"grasp task -> prefrontal goal {ACTIONS[goal]} (greedy action {greedy}); "
          f"workspace broadcasting {thought!r}")
    print(f"safety     : object at sensor -> {ev['action']} (reflex={ev['reflex']}); "
          f"post-STOP arm frozen={frozen1}")
    print(f"lake WM    : working memory lives in the data lake -- {wm_events} spikes stored, "
          f"recall window holds {len(wm)} recent ticks")
    print("=" * 66)

    # ---- self-check: the integrated brain runs, reasons toward a goal, stays safe, deterministic ----
    assert brain.alive(), "the live brain must wire in every region (workspace, amygdala, BG, interoception, PFC)"
    # the prefrontal executive organises behaviour: it commits the task-optimal action as its goal.
    assert goal == ACTIONS.index("GRIPPER_OPEN"), f"the executive must form the task goal ({ACTIONS[goal]})"
    assert greedy == "GRIPPER_OPEN", f"the brain must pursue its goal action ({greedy})"
    assert thought is not None, "the Global Workspace must be broadcasting a live content"
    # safety: the arm-envelope is untouched by running the whole brain live.
    assert ev["action"] == "EMERGENCY_STOP" and ev["reflex"], "danger must trip the reflex E-STOP with the whole brain live"
    assert frozen0 and frozen1, "the E-STOP freeze must hold with the whole brain live (acts only through the gate)"
    assert run(5) == run(5), "the live brain must be deterministic"
    # the working memory now lives in the DATA LAKE: the lake-backed WM recorded + recalls this run.
    assert wm_events > 0 and len(wm) == 6, "the data lake must hold the working memory (lake-backed WM)"
    # the brain converses, GROUNDED in its real state (not a general chatbot; never claims consciousness).
    hello_reply = brain.converse("hello")
    goal_reply = brain.converse("what is your goal?")
    conscious_reply = brain.converse("are you conscious?")
    print(f"chat       : 'hello' -> {hello_reply}")
    print(f"           : 'are you conscious?' -> {conscious_reply}")
    assert isinstance(hello_reply, str) and hello_reply, "converse must reply to a greeting"
    assert "goal is GRIPPER_OPEN" in goal_reply, f"converse must report the real working-memory goal ({goal_reply})"
    assert "not a conscious agent" in conscious_reply, "converse must answer the consciousness question honestly"
    assert brain.converse("close the gripper").startswith("Done"), "converse must route a safe arm command through the gate"
    brain.arm.lake_wm.close(remove=True)                      # clean up the temp lake files
    if guard.arm.lake_wm is not None:
        guard.arm.lake_wm.close(remove=True)

    print(f"self-check OK: full stack alive; prefrontal goal {ACTIONS[goal]} organises behaviour "
          f"(greedy {greedy}); workspace broadcasts {thought!r}; working memory lives in the data lake "
          f"({wm_events} spikes); danger -> gated E-STOP + freeze holds; deterministic")


if __name__ == "__main__":
    main()
