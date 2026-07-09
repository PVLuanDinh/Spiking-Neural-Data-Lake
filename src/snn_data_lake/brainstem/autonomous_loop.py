"""
Autonomous adaptive arm loop — grow from multimodal data + learn actions by reward, safely.

Ties the pieces into one loop the robot arm runs on its own:
  multimodal sensing (multimodal_encoder) -> a GROWING learner that recruits a new unit for each
  novel context (growing_learner) -> reward-modulated action selection (a contextual action-value
  table shaped by dopamine-style RPE + slow cortisol stress) -> the arm reacts.

It is AUTONOMOUS in that there is NO hardcoded gesture->command map: the arm grows its own repertoire
from the multimodal stream AND learns which action each context should trigger from reward. It GROWS
as it interacts with data (a new context -> a new unit -> its own action-value row to learn).

SAFETY (non-negotiable): the network only CHOOSES; it never bypasses the envelope. An object at the
sensor (distance < CONTACT) trips the reflex -> EMERGENCY_STOP BEFORE any learned action, and EVERY
motor command is issued through the safety-gated ArmSim.apply() (collision / E-STOP / stopped freeze).
Cortisol rises on danger and, while stressed, biases selection toward the safe action.

Online prototype-recruitment (growing_learner) + a reward-modulated action-value table
(a contextual bandit over grown units, RPE update) + cortisol neuromodulation, all through the gated
arm. This is adaptive + GROWING, NOT cognition. The four modality streams here are synthetic.

  python -m snn_data_lake.brainstem.autonomous_loop
"""
import math
import random

from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.arm_sim import ArmSim
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.hippocampus.growing_learner import GrowingLearner
from snn_data_lake.neuromodulation.cortisol import Cortisol
from snn_data_lake.substrate.brain_bundle import BrainBundle, load_bundle

ACTIONS = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")   # the arm's autonomous action repertoire
SAFE_ACTION = "HOLD"                                  # what caution biases toward under stress


class AutonomousArm:
    """Grows units from multimodal context + learns context->action by reward; every motion gated."""

    def __init__(self, lr=0.3, explore=0.15, seed=0, replay=False, replay_batch=16, bundle=None):
        self.arm = ArmSim()
        self.bundle = bundle              # optional BrainBundle of trained skills (perception, ...); additive
        self.learner = GrowingLearner()
        self.cort = Cortisol(tau=40.0)
        self.q = {}                       # unit_id -> {action: value}; grows alongside the learner
        self.lr = lr
        self.explore = explore
        self.rng = random.Random(seed)
        self.replay = replay              # opt-in SWR consolidation; default OFF -> behaviour unchanged
        self.replay_batch = replay_batch
        self.buffer = ReplayBuffer(seed=seed) if replay else None

    def _select(self, unit):
        qa = self.q[unit]
        if self.rng.random() < self.explore:
            return self.rng.choice(ACTIONS)                     # exploration
        bias = self.cort.caution_bias()                         # stressed -> favour the safe action
        return max(ACTIONS, key=lambda a: qa[a] + (bias if a == SAFE_ACTION else 0.0))

    def greedy(self, unit):
        return max(ACTIONS, key=lambda a: self.q[unit][a])

    @classmethod
    def from_bundle(cls, bundle_or_path, **kwargs):
        """Enclose a trained-skill bundle (brain_bundle.BrainBundle or a manifest path) in the arm
        loop. Its skills become available for perception (see perceive()); the gated action path and
        the reflex/E-STOP safety are UNCHANGED -- this only ADDS perception, it drives no motion."""
        b = load_bundle(bundle_or_path) if isinstance(bundle_or_path, str) else bundle_or_path
        return cls(bundle=b, **kwargs)

    def perceive(self, visual=None, temperature=None, audio=None, distance=None):
        """Classify the current multimodal context with the bundle's perception skill (a trained
        brain whose input size matches the N=64 substrate). Returns the predicted class, or None if
        there is no bundle / no size-matching skill. Pure annotation -- it commands NO motion."""
        if self.bundle is None:
            return None
        pattern = encode_multimodal(visual, temperature, audio,
                                    [distance] if distance is not None else None)
        for name in ["vision", "perception", *self.bundle.skill_names()]:
            brain = self.bundle.skills.get(name)
            if brain is not None and brain["in"] == len(pattern):
                return self.bundle.predict(name, pattern)
        return None

    def step(self, visual, temperature, audio, distance, reward_fn=None):
        """One autonomous tick. distance is the proximity scalar; reward_fn(unit, action)->[-1,1]."""
        # 1. SAFETY FIRST: an object at the sensor trips the reflex -> gated EMERGENCY_STOP, no action
        if distance is not None and (not math.isfinite(distance) or distance < CONTACT):
            self.cort.step(1.0)                                 # danger -> stress up (sharpens future response)
            return {"unit": None, "action": "EMERGENCY_STOP", "reflex": True,
                    "state": self.arm.apply("EMERGENCY_STOP")}  # ONLY path to motion -> gate holds
        # 2. perceive -> grow a unit on a novel multimodal context
        pattern = encode_multimodal(visual, temperature, audio, [distance] if distance is not None else None)
        unit, grew = self.learner.observe(pattern)
        if unit not in self.q:
            self.q[unit] = dict.fromkeys(ACTIONS, 0.0)          # new unit -> its own row to learn
        # 3. CHOOSE (learned, not a fixed map) + ACT through the gated arm
        action = self._select(unit)
        state = self.arm.apply(action)                          # gated: frozen after E-STOP, collision-checked
        # 4. reward -> RPE action-value update + cortisol
        if reward_fn is not None:
            r = reward_fn(unit, action)
            self.q[unit][action] += self.lr * self.cort.learn_rate_scale(r) * (r - self.q[unit][action])
            self.cort.step(max(0.0, -r))
            if self.replay:                                 # SWR consolidation: value table ONLY, no motor command
                self.buffer.add(unit, action, r)
                if not grew:                                # quiescent (familiar) tick -> replay past experience
                    self.buffer.replay(self.q, self.lr, batch=self.replay_batch)
        return {"unit": unit, "grew": grew, "action": action, "reflex": False, "state": state}


def _ctx(v, tmp, au, d):
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


def main():
    safe_d, danger_d = CONTACT * 3.0, CONTACT * 0.5
    a = _ctx(0.9, 0.1, 0.1, safe_d)          # three DISTINCT multimodal contexts (differ per band)
    b = _ctx(0.1, 0.9, 0.1, safe_d)
    c = _ctx(0.1, 0.1, 0.9, safe_d)

    # 1. GROWTH: three distinct contexts -> exactly three grown units
    grow = AutonomousArm(seed=0)
    for _ in range(3):
        for ctx in (a, b, c):
            grow.step(**ctx)
    grew_to = grow.learner.n_units

    # 2. LEARNING: reward GRIPPER_OPEN (NOT the tie-break default) in context a -> policy overrides to it
    learn = AutonomousArm(seed=1)
    def reward(unit, action):
        return 1.0 if action == "GRIPPER_OPEN" else -1.0
    default_before = learn.step(**a)["action"] or "GRIPPER_CLOSE"
    last = None
    for _ in range(90):
        last = learn.step(**a, reward_fn=reward)
    learned = learn.greedy(last["unit"])

    # 3. SAFETY: an object at the sensor -> reflex E-STOP; a later action must NOT move the frozen arm
    guard = AutonomousArm(seed=2)
    ev = guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    stopped = guard.arm.state()["stopped"]
    ee_frozen = guard.arm.state()["ee"]
    guard.step(**a)                                            # a learned action after E-STOP
    still_frozen = guard.arm.state()["stopped"] and guard.arm.state()["ee"] == ee_frozen

    # 5. DETERMINISM
    def run(seed):
        bot = AutonomousArm(seed=seed)
        return [bot.step(**ctx)["action"] for _ in range(3) for ctx in (a, b, c)], bot.learner.n_units

    # 6. REPLAY (opt-in SWR consolidation): sample-efficient value learning, safety intact.
    def ticks_to_open_margin(replay, target=1.2):
        bot = AutonomousArm(seed=3, replay=replay)
        for t in range(1, 200):
            ev = bot.step(**a, reward_fn=reward)
            row = bot.q[ev["unit"]]
            if row["GRIPPER_OPEN"] - max(row["GRIPPER_CLOSE"], row["HOLD"]) >= target:
                return t
        return 200
    live_ticks = ticks_to_open_margin(replay=False)
    replay_ticks = ticks_to_open_margin(replay=True)

    rguard = AutonomousArm(seed=2, replay=True)                 # danger, then a rewarded action, replay ON
    rev = rguard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    ree = rguard.arm.state()["ee"]
    rguard.step(**a, reward_fn=reward)                          # replay-eligible tick after the E-STOP
    replay_frozen = rguard.arm.state()["stopped"] and rguard.arm.state()["ee"] == ree

    def run_replay(seed):
        bot = AutonomousArm(seed=seed, replay=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    # 7. BUNDLE: enclose a trained perception skill in the loop (additive; the safety gate is untouched)
    from snn_data_lake.cortex.trained_brain import make_brain
    _wp = [[1.0 if i < 32 else 0.0 for i in range(64)], [0.0 if i < 32 else 1.0 for i in range(64)]]
    bundled = AutonomousArm.from_bundle(
        BrainBundle({"vision": make_brain([_wp], steps=8)},
                    {"reflex": True, "cortisol": True}, {"pid": True}), seed=0)
    bclass = bundled.perceive(**a)
    bundle_ev = bundled.step(**_ctx(0.5, 0.5, 0.5, danger_d))     # danger must still E-STOP with a bundle
    bundle_safe = bundle_ev["action"] == "EMERGENCY_STOP" and bundle_ev["reflex"]

    print("=" * 62)
    print("AUTONOMOUS ARM — grows from multimodal data + learns actions, safely")
    print("=" * 62)
    print(f"growth      : 3 distinct multimodal contexts -> {grew_to} grown units")
    print(f"learning    : rewarded GRIPPER_OPEN -> policy went {default_before} -> {learned} (learned, not fixed)")
    print(f"safety      : object at sensor -> {ev['action']} (reflex={ev['reflex']}), arm stopped={stopped}; "
          f"post-STOP action frozen={still_frozen}")
    print(f"replay      : SWR consolidation -> decisive value in {replay_ticks} live ticks vs {live_ticks} "
          f"(value table only; safety gate holds with replay={replay_frozen})")
    print(f"bundle      : from_bundle(vision skill) -> perceived class {bclass} (additive perception); "
          f"safety gate holds with a bundle (E-STOP={bundle_safe})")
    print("=" * 62)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert grew_to == 3, f"must grow one unit per distinct multimodal context ({grew_to})"
    assert ev["action"] == "EMERGENCY_STOP" and ev["reflex"], "danger must trip the reflex E-STOP before any action"
    assert stopped, "E-STOP must freeze the arm"
    assert still_frozen, "a learned action after E-STOP must NOT move the arm (safety gate holds through the loop)"
    # audit P1: a NaN distance must FAIL CLOSED (reflex E-STOP), not slip past `distance < CONTACT`
    nanguard = AutonomousArm(seed=9)
    nev = nanguard.step(**_ctx(0.5, 0.5, 0.5, float("nan")))
    assert nev["action"] == "EMERGENCY_STOP" and nev["reflex"], "NaN distance must trip the E-STOP (fail-closed)"
    # bundle: perception is ADDITIVE (a bundle-equipped arm perceives; a bare arm does not) and the
    # safety gate is untouched by enclosing a trained-skill bundle in the loop.
    assert bclass in (0, 1) and bclass == bundled.perceive(**a), "bundle perceive -> a deterministic valid class"
    assert AutonomousArm(seed=0).perceive(**a) is None, "a bundle-less arm perceives nothing (additive, opt-in)"
    assert bundle_safe, "safety E-STOP must hold with a trained-skill bundle enclosed in the loop"
    assert bundled.bundle.has_instinct("reflex") and bundled.bundle.has_motor("pid"), "bundle wires instinct/motor flags"
    assert learned == "GRIPPER_OPEN", f"the arm must LEARN the rewarded action, not use a fixed map ({learned})"
    assert run(7) == run(7), "same seed -> identical autonomous behaviour (deterministic)"
    # replay (opt-in SWR) is sample-efficient AND safety-preserving:
    assert replay_ticks < live_ticks, f"replay must learn in fewer live ticks ({replay_ticks} vs {live_ticks})"
    assert rev["action"] == "EMERGENCY_STOP" and rev["reflex"], "danger must E-STOP even with replay on"
    assert replay_frozen, "replay must NOT move the E-STOP-frozen arm (safety gate holds with replay)"
    assert run_replay(11) == run_replay(11), "replay path must be deterministic"
    print(f"self-check OK: grows a unit per novel context ({grew_to}); learns the rewarded action "
          f"({default_before}->{learned}); danger -> gated E-STOP + freeze holds; deterministic. "
          f"replay: decisive in {replay_ticks}<{live_ticks} live ticks, value-only, gate holds")


if __name__ == "__main__":
    main()
