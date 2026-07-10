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


class LiveBrain:
    """The integrated brain, live: an AutonomousArm with the full stack on (Global Workspace +
    amygdala threat + basal-ganglia gate + interoceptive arousal + prefrontal executive), centred on
    the executive and acting only through the gated arm. Composition only -- it adds no new mechanism
    and no new path to the actuator; it just runs every region together as one loop."""

    def __init__(self, seed=0, pfc_gain=0.5):
        self.arm = AutonomousArm(seed=seed, workspace=True, amygdala=True, bg_gate=True,
                                 interoception=True, pfc=True, pfc_gain=pfc_gain)

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

    # 2. SAFETY with the whole brain live: an object at the sensor -> reflex E-STOP, freeze holds
    guard = LiveBrain(seed=2)
    ev = guard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    frozen0 = guard.arm.arm.state()["stopped"]
    guard.tick(grasp)                                          # a later action must NOT move the frozen arm
    frozen1 = guard.arm.arm.state()["stopped"]

    # 3. DETERMINISM: same seed -> identical live behaviour
    def run(seed):
        b = LiveBrain(seed=seed)
        return [b.tick(grasp, reward_fn=reward)["action"] for _ in range(12)]

    print("=" * 66)
    print("LIVE BRAIN -- every region in one self-driving loop, centred on reasoning")
    print("=" * 66)
    print(f"stack alive (all regions wired): {brain.alive()}")
    print(f"grasp task -> prefrontal goal {ACTIONS[goal]} (greedy action {greedy}); "
          f"workspace broadcasting {thought!r}")
    print(f"safety     : object at sensor -> {ev['action']} (reflex={ev['reflex']}); "
          f"post-STOP arm frozen={frozen1}")
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

    print(f"self-check OK: full stack alive; prefrontal goal {ACTIONS[goal]} organises behaviour "
          f"(greedy {greedy}); workspace broadcasts {thought!r}; danger -> gated E-STOP + freeze holds; deterministic")


if __name__ == "__main__":
    main()
