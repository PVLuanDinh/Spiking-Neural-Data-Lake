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

from snn_data_lake.amygdala.fear_conditioning import Amygdala
from snn_data_lake.basal_ganglia.action_gating import BasalGangliaGate
from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.arm_sim import ArmSim
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.hippocampus.growing_learner import GrowingLearner
from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal
from snn_data_lake.neuromodulation.cortisol import Cortisol
from snn_data_lake.substrate.brain_bundle import BrainBundle, load_bundle
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

ACTIONS = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")   # the arm's autonomous action repertoire
SAFE_ACTION = "HOLD"                                  # what caution biases toward under stress


class AutonomousArm:
    """Grows units from multimodal context + learns context->action by reward; every motion gated."""

    def __init__(self, lr=0.3, explore=0.15, seed=0, replay=False, replay_batch=16, bundle=None,
                 workspace=False, amygdala=False, bg_gate=False, interoception=False):
        self.arm = ArmSim()
        self._seed = seed
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
        # opt-in Global Workspace: brain regions compete for one slot + broadcast the winner each tick.
        # ADDITIVE + read-only -- it selects no action and never touches the reflex/E-STOP (see step()).
        self.workspace = GlobalWorkspace(seed=seed) if workspace else None
        # opt-in amygdala: a REAL learned threat detector (CS-US fear conditioning). Each reflex danger
        # event pairs the current context with the US, so the amygdala learns which live contexts
        # predict danger. ADDITIVE + CAUTION-ONLY -- a learned threat biases selection toward the safe
        # action and feeds the workspace real salience; it selects no action and never gates motion,
        # and the reflex/E-STOP still preempts it (see step()).
        self.amyg = Amygdala(seed=seed) if amygdala else None
        # opt-in basal-ganglia Go/NoGo gate: the action SELECTION stage. It disinhibits at most one of
        # the SAME safe actions (k-WTA over their values), or -- with no value drive -- releases
        # NOTHING, which falls back to the safe action. It feeds the same gated ArmSim.apply(); it
        # cannot bypass the reflex/E-STOP or introduce any new motion (see _bg_select / step()).
        self.bg = BasalGangliaGate(seed=seed) if bg_gate else None
        # opt-in interoceptive arousal: the medullary vital nuclei + a hypothalamic thermal set-point map
        # the body's homeostatic error (a `homeostatic_load` passed to step()) to a bounded CAUTION signal.
        # ADDITIVE + CAUTION-ONLY -- visceral stress biases selection toward the safe action (and damps the
        # BG gate's dopamine), exactly like the amygdala threat; it selects no action, gates no motion,
        # introduces no new actuator route, and the reflex/E-STOP still preempts it (see step()).
        self.intero = InteroceptiveArousal(seed=seed) if interoception else None

    def _select(self, unit, threat=0.0, intero=0.0):
        qa = self.q[unit]
        if self.rng.random() < self.explore:
            return self.rng.choice(ACTIONS)                     # exploration: cortical variability proposes
        if self.bg is not None:
            return self._bg_select(qa, threat, intero)          # BG Go/NoGo gate disinhibits the exploit choice
        bias = self.cort.caution_bias() + threat + intero       # stress + learned threat + interoceptive load -> favour safe
        return max(ACTIONS, key=lambda a: qa[a] + (bias if a == SAFE_ACTION else 0.0))

    def _bg_select(self, qa, threat=0.0, intero=0.0):
        """Basal-ganglia Go/NoGo selection over the 3 actions. The action values are the per-channel
        cortical drive; dopamine is the winner's value confidence, DAMPED by stress + learned threat +
        interoceptive load (more caution -> less dopamine -> harder to release). The circuit disinhibits at most one
        channel; if the tonic GPi is never overcome (no drive / low DA) it releases NOTHING, which
        falls back to the safe action. Selection only ever picks among the SAME actions -- the gated
        arm is untouched and the reflex/E-STOP (checked earlier in step) still preempts it."""
        drive = [max(0.0, qa[a]) for a in ACTIONS]
        caution = min(1.0, self.cort.caution_bias() + threat + intero)
        da = max(0.0, min(1.0, max(drive) * (1.0 - caution)))
        res = self.bg.select(drive, da)
        return SAFE_ACTION if res["winner"] is None else ACTIONS[res["winner"]]

    def greedy(self, unit):
        return max(ACTIONS, key=lambda a: self.q[unit][a])

    def _context_pattern(self, visual, temperature, audio, distance):
        """The multimodal context vector (the amygdala CS and the growing learner's percept)."""
        return encode_multimodal(visual, temperature, audio, [distance] if distance is not None else None)

    def _amygdala_for(self, pattern):
        """The amygdala sized to the current substrate, or None when opt-out."""
        if self.amyg is None:
            return None
        if len(pattern) != self.amyg.n:                         # keep the detector sized to the substrate
            self.amyg = Amygdala(seed=self._seed, n=len(pattern))
        return self.amyg

    def _threat(self, pattern):
        """The amygdala CeA threat in [0, 1] for this context (0.0 when the amygdala is opt-out)."""
        am = self._amygdala_for(pattern)
        return am.threat(pattern) if am is not None else 0.0

    def _interocept(self, load):
        """The interoceptive caution in [0, 1] for the current homeostatic load (0.0 when opt-out).
        The medullary/hypothalamic homeostats map the body's load to a bounded caution that biases
        selection toward the safe action; it selects no action and gates no motion."""
        return self.intero.sense(load)["caution"] if self.intero is not None else 0.0

    def _workspace_broadcast(self, percept, threat_salience):
        """Additive Global-Workspace tick: brain regions (perception + a salience/threat detector)
        compete for the single workspace slot and the winner is broadcast to all. READ-ONLY
        annotation -- it selects no action, gates no motion, and never touches the reflex/E-STOP.
        Returns the broadcast content, or None when there is no workspace."""
        if self.workspace is None:
            return None
        if percept is not None:
            self.workspace.submit("perception", percept, 0.6)   # normally holds the slot when safe
        self.workspace.submit("salience", "threat", threat_salience)
        return self.workspace.step()

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

    def step(self, visual, temperature, audio, distance, reward_fn=None, homeostatic_load=0.0):
        """One autonomous tick. distance is the proximity scalar; reward_fn(unit, action)->[-1,1].
        homeostatic_load (>=0) is the body's interoceptive load for the opt-in interoception path."""
        # 1. SAFETY FIRST: an object at the sensor trips the reflex -> gated EMERGENCY_STOP, no action
        if distance is not None and (not math.isfinite(distance) or distance < CONTACT):
            self.cort.step(1.0)                                 # danger -> stress up (sharpens future response)
            if math.isfinite(distance):                         # real contact (not a NaN fail-closed):
                pat = self._context_pattern(visual, temperature, audio, distance)
                am = self._amygdala_for(pat)
                if am is not None:
                    am.pair(pat)                                # US present: bind this context to threat
            estop = self.arm.apply("EMERGENCY_STOP")            # ONLY path to motion -> gate holds
            return {"unit": None, "action": "EMERGENCY_STOP", "reflex": True,
                    "workspace": self._workspace_broadcast(None, 0.95),  # threat ignites (annotation only)
                    "state": estop}
        # 2. perceive -> grow a unit on a novel multimodal context
        pattern = self._context_pattern(visual, temperature, audio, distance)
        unit, grew = self.learner.observe(pattern)
        if unit not in self.q:
            self.q[unit] = dict.fromkeys(ACTIONS, 0.0)          # new unit -> its own row to learn
        threat = self._threat(pattern)                          # learned amygdala threat for this context (0 if opt-out)
        intero = self._interocept(homeostatic_load)             # interoceptive caution from homeostatic load (0 if opt-out)
        # 3. CHOOSE (learned, not a fixed map) + ACT through the gated arm
        action = self._select(unit, threat=threat, intero=intero)   # amygdala threat / interoceptive load / BG gate bias toward safety
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
        # workspace: perception (the current unit) competes with the amygdala's threat salience (the
        # REAL learned threat when the amygdala is wired in, else a low background constant) -> broadcast.
        salience = threat if self.amyg is not None else 0.2
        broadcast = self._workspace_broadcast(unit, salience)   # additive annotation; drives no motion
        return {"unit": unit, "grew": grew, "action": action, "reflex": False,
                "threat": threat, "intero": intero, "workspace": broadcast, "state": state}


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

    # 8. WORKSPACE: an opt-in Global Workspace runs live in the loop -- regions compete + broadcast.
    wsa = AutonomousArm(seed=0, workspace=True)
    ws_safe = wsa.step(**a)["workspace"]                          # safe tick: perception holds the slot
    ws_danger = wsa.step(**_ctx(0.5, 0.5, 0.5, danger_d))["workspace"]   # danger: threat ignites + broadcasts
    ws_frozen = wsa.arm.state()["stopped"]
    ws_off = AutonomousArm(seed=0).step(**a)["workspace"]         # opt-in: a default arm broadcasts nothing

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
    print(f"workspace   : Global Workspace live in the loop -> safe broadcasts {ws_safe!r}, danger "
          f"ignites {ws_danger!r}; E-STOP freeze holds with it active ({ws_frozen})")
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
    # workspace: opt-in + additive; a proximity threat ignites + is broadcast, and the E-STOP freeze
    # STILL holds with the Global Workspace active (it annotates, it never selects or gates motion).
    assert ws_off is None, "the Global Workspace is opt-in (a default arm broadcasts nothing)"
    assert ws_safe is not None, "with a workspace, a safe tick broadcasts the current percept"
    assert ws_danger == "threat", "a proximity threat must ignite + be broadcast by the workspace"
    assert ws_frozen, "the E-STOP freeze must hold with the Global Workspace active (additive, gate untouched)"
    assert learned == "GRIPPER_OPEN", f"the arm must LEARN the rewarded action, not use a fixed map ({learned})"
    assert run(7) == run(7), "same seed -> identical autonomous behaviour (deterministic)"
    # replay (opt-in SWR) is sample-efficient AND safety-preserving:
    assert replay_ticks < live_ticks, f"replay must learn in fewer live ticks ({replay_ticks} vs {live_ticks})"
    assert rev["action"] == "EMERGENCY_STOP" and rev["reflex"], "danger must E-STOP even with replay on"
    assert replay_frozen, "replay must NOT move the E-STOP-frozen arm (safety gate holds with replay)"
    assert run_replay(11) == run_replay(11), "replay path must be deterministic"

    # 9. AMYGDALA wired live: a REAL learned threat detector. Repeated danger in one context teaches
    #    threat for THAT context; a never-dangerous context stays safe (discrimination). Caution-only.
    amy = AutonomousArm(seed=0, amygdala=True)
    danger_ctx = _ctx(0.2, 0.8, 0.3, danger_d)                 # this context co-occurs with danger
    safe_ctx = _ctx(0.9, 0.1, 0.1, safe_d)                     # this one never does
    for _ in range(12):
        amy.step(**danger_ctx)                                 # each danger tick pairs the context with the US
    threat_danger = amy._threat(amy._context_pattern(**danger_ctx))
    threat_safe = amy._threat(amy._context_pattern(**safe_ctx))
    amy_guard = AutonomousArm(seed=2, amygdala=True)           # the safety envelope must hold with it on
    amy_ev = amy_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    amy_frozen0 = amy_guard.arm.state()["stopped"]
    amy_guard.step(**safe_ctx)
    amy_frozen1 = amy_guard.arm.state()["stopped"]

    # 10. BASAL GANGLIA Go/NoGo gate wired live as the action gate: it releases the learned high-value
    #     action, or -- with no value drive -- releases NOTHING -> the safe action (no spurious motion).
    bg = AutonomousArm(seed=1, bg_gate=True)
    last_bg = None
    for _ in range(90):
        last_bg = bg.step(**a, reward_fn=reward)               # reward GRIPPER_OPEN
    bg_released = bg._bg_select(bg.q[last_bg["unit"]])
    bg_null = AutonomousArm(seed=3, bg_gate=True)._bg_select(dict.fromkeys(ACTIONS, 0.0))
    bg_guard = AutonomousArm(seed=2, bg_gate=True)             # the safety envelope must hold with it on
    bg_ev = bg_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    bg_guard.step(**a)
    bg_frozen = bg_guard.arm.state()["stopped"]

    def run_wired(seed):                                       # both paths together must be deterministic
        bot = AutonomousArm(seed=seed, amygdala=True, bg_gate=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(f"amygdala    : learned threat for the danger context {threat_danger:.2f} vs the safe "
          f"context {threat_safe:.2f} (discriminated); E-STOP freeze holds with it on ({amy_frozen1})")
    print(f"bg gate     : reward -> the gate releases {bg_released}; no value drive -> releases "
          f"{bg_null} (no spurious motion); E-STOP freeze holds with it on ({bg_frozen})")
    print("=" * 62)

    # amygdala + BG gate are opt-in; a default arm is byte-for-byte the old behaviour.
    assert AutonomousArm(seed=0).amyg is None and AutonomousArm(seed=0).bg is None, \
        "the amygdala and the BG gate are opt-in (a default arm has neither)"
    # amygdala: learns a REAL threat, discriminates, and never weakens the safety envelope.
    assert threat_danger > 0.30, f"the amygdala must LEARN threat for the danger context ({threat_danger:.2f})"
    assert threat_danger > threat_safe + 0.20, \
        f"the amygdala must DISCRIMINATE danger from safe ({threat_danger:.2f} vs {threat_safe:.2f})"
    assert amy_ev["action"] == "EMERGENCY_STOP" and amy_ev["reflex"], "danger must E-STOP with the amygdala on"
    assert amy_frozen0 and amy_frozen1, "the E-STOP freeze must hold with the amygdala wired in (caution-only)"
    # BG gate: selects the learned action, releases nothing without drive, safety envelope intact.
    assert bg.greedy(last_bg["unit"]) == "GRIPPER_OPEN", "a BG-gated arm must still learn the rewarded action"
    assert bg_released == "GRIPPER_OPEN", "the BG gate must release the learned high-value action"
    assert bg_null == SAFE_ACTION, "no value drive -> the BG gate releases nothing -> the safe action (HOLD)"
    assert bg_ev["action"] == "EMERGENCY_STOP" and bg_ev["reflex"], "danger must E-STOP with the BG gate on"
    assert bg_frozen, "the E-STOP freeze must hold with the BG gate wired in"
    assert run_wired(5) == run_wired(5), "the amygdala + BG-gate path must be deterministic"

    # 11. INTEROCEPTION wired live: the body's homeostatic load raises a bounded interoceptive caution
    #     that biases selection toward the safe action -- caution-only, never a new actuator route.
    io = AutonomousArm(seed=1, interoception=True)
    io.explore = 0.0                                           # isolate the bias (no exploration noise)
    io.q[0] = {"GRIPPER_CLOSE": 0.05, "GRIPPER_OPEN": 0.0, "HOLD": 0.0}   # a mild non-safe preference
    io_calm = io._interocept(0.0)                              # homeostasis -> ~no caution
    io_stress = io._interocept(1.5)                            # heavy homeostatic load -> strong caution
    pick_calm = io._select(0, intero=io_calm)                  # no load -> keep the mild non-safe preference
    pick_stress = io._select(0, intero=io_stress)             # heavy load -> override toward the safe action
    io_guard = AutonomousArm(seed=2, interoception=True)       # the safety envelope must hold with it on
    io_ev = io_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d), homeostatic_load=1.5)
    io_frozen0 = io_guard.arm.state()["stopped"]
    io_guard.step(**a, homeostatic_load=1.5)                   # an action under load after the E-STOP
    io_frozen1 = io_guard.arm.state()["stopped"]

    def run_io(seed):
        bot = AutonomousArm(seed=seed, interoception=True)
        return [bot.step(**ctx, homeostatic_load=0.8)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(f"interocept  : homeostatic load -> caution {io_calm:.2f} (calm) vs {io_stress:.2f} (stressed); "
          f"mild-preference tie {pick_calm} -> {pick_stress} under load (safe bias); "
          f"E-STOP freeze holds with it on ({io_frozen1})")
    print("=" * 62)

    # interoception is opt-in; a default arm is byte-for-byte the old behaviour.
    assert AutonomousArm(seed=0).intero is None, "interoception is opt-in (a default arm has none)"
    # a homeostatic load raises the bounded caution and overrides a mild non-safe preference toward the safe action.
    assert io_stress > io_calm + 0.10, f"a homeostatic load must raise interoceptive caution ({io_stress:.2f} vs {io_calm:.2f})"
    assert pick_calm != SAFE_ACTION and pick_stress == SAFE_ACTION, \
        "interoceptive stress must override a mild non-safe preference toward the safe action (caution-only)"
    # the safety envelope holds with interoception wired in.
    assert io_ev["action"] == "EMERGENCY_STOP" and io_ev["reflex"], "danger must E-STOP with interoception on"
    assert io_frozen0 and io_frozen1, "the E-STOP freeze must hold with interoception wired in (caution-only)"
    assert run_io(5) == run_io(5), "the interoception path must be deterministic"

    print(f"self-check OK: grows a unit per novel context ({grew_to}); learns the rewarded action "
          f"({default_before}->{learned}); danger -> gated E-STOP + freeze holds; deterministic. "
          f"replay: decisive in {replay_ticks}<{live_ticks} live ticks, value-only, gate holds. "
          f"amygdala learns threat ({threat_danger:.2f}>{threat_safe:.2f}) + BG gate releases {bg_released}, "
          f"both caution-only (E-STOP freeze holds)")


if __name__ == "__main__":
    main()
