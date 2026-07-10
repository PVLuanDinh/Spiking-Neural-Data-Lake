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
from snn_data_lake.basal_ganglia.latent_q import LatentQ
from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.arm_sim import ArmSim
from snn_data_lake.cerebellum.hjepa_serve import HJepaServe
from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.hippocampus.growing_learner import GrowingLearner
from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal
from snn_data_lake.neuromodulation.cortisol import Cortisol
from snn_data_lake.neuromodulation.curriculum import Curriculum
from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive
from snn_data_lake.substrate.brain_bundle import BrainBundle, load_bundle
from snn_data_lake.substrate.lake_working_memory import LakeWorkingMemory
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

ACTIONS = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")   # the arm's autonomous action repertoire
SAFE_ACTION = "HOLD"                                  # what caution biases toward under stress


class AutonomousArm:
    """Grows units from multimodal context + learns context->action by reward; every motion gated."""

    def __init__(self, lr=0.3, explore=0.15, seed=0, replay=False, replay_batch=16, bundle=None,
                 workspace=False, amygdala=False, bg_gate=False, interoception=False,
                 pfc=False, pfc_gain=0.5, lake_wm=False, lake_wm_path=None,
                 jepa=False, jepa_horizon=4, hjepa_serve=None, curiosity=False, curiosity_gain=0.5,
                 latent_q=False):
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
        # opt-in prefrontal executive: the "centre of thoughts" -- it deliberates over the current
        # action-values (evidence), holds a GOAL in working memory, and emits a TOP-DOWN bias toward it.
        # ADDITIVE + top-down PREFERENCE ONLY -- it raises the goal action's score/drive so behaviour is
        # goal-directed (and persists across ticks), but it selects no action, gates no motion, adds no
        # actuator route, and caution (threat/interoceptive/stress) + the reflex/E-STOP still preempt it.
        self.pfc = PrefrontalExecutive(len(ACTIONS), seed=seed) if pfc else None
        self.pfc_gain = pfc_gain
        # opt-in lake working memory: each tick's activity (context pattern + one-hot goal + one-hot
        # action) is stored in the durable spike lake (StreamingHub .spkl) and read back with recall(),
        # so the working memory literally LIVES IN THE DATA LAKE. ADDITIVE + storage only -- it records
        # after the action is chosen, never touches the reflex/E-STOP path, and adds no actuator route.
        self._lake_wm_on = lake_wm
        self._lake_wm_path = lake_wm_path
        self.lake_wm = None                                   # lazily sized on the first tick (needs len(pattern))
        self._pattern_len = None
        self._tick_t = 0
        # opt-in JEPA latent world model: a learned encoder enc(context)->z + an action-conditioned latent
        # predictor pred(z,a)->z' (cerebellum/jepa_world_model), trained self-supervised ONLINE from the live
        # tick stream. When on (with the executive), the prefrontal executive PLANS over it -- model-based
        # lookahead in LATENT space -- to set its goal, instead of the reactive value deliberation; the result
        # is the SAME top-down PREFERENCE (bounded by pfc_gain) fed to the SAME gated selection. ADDITIVE +
        # read-only to safety: it drives no motion and adds no actuator route; the reflex/E-STOP runs FIRST and
        # still preempts, and every motor command stays gated (see step()). Lazily sized on the first tick.
        self._jepa_on = jepa
        self.jepa = None
        self.jepa_horizon = max(1, min(4, int(jepa_horizon)))   # bounded rollout (the learned map can diverge)
        self._z_goal = None                                     # reward-EMA of latents = the latent goal to plan toward
        self._jepa_prev = None                                  # (prev pattern, prev action) for the online (z,a,z') update
        # opt-in SERVED H-JEPA: a GPU-trained 2-level hierarchical JEPA (cerebellum/hjepa_serve.HJepaServe, FROZEN).
        # The executive plans over the ABSTRACT (Level-1) latent: z0 = encode0(pattern) each tick fills a K-window,
        # z1 = encode1(window) is the abstract situation, and plan() rolls the L1 predictor over it. Like the online
        # jepa it ONLY biases the executive's goal (bounded by pfc_gain, gated) -- it drives no motion, adds no
        # actuator route, and the reflex/E-STOP runs first. Served/frozen: no online training in the loop.
        self.hjepa_serve = hjepa_serve
        self._hj_window = []                                    # rolling K-window of Level-0 latents (feeds encode1)
        self._z1_goal = None                                    # reward-EMA of the abstract Level-1 latent (plan goal)
        # opt-in CURIOSITY (intrinsic motivation): the JEPA world-model prediction error becomes a
        # LEARNING-PROGRESS reward that drives SELF-DIRECTED exploration -- the arm learns even with NO external
        # reward, and it seeks learnable-novel contexts (not noise; learning-progress rejects the noisy-TV trap).
        # Needs the JEPA (for the prediction error). ADDITIVE + read-only to safety: the intrinsic reward feeds
        # only the value update -- it drives no motion, adds no actuator route, creates no stress, and the
        # reflex/E-STOP + amygdala/interoceptive caution still preempt every command (like a baby drawn to
        # novelty while external guards keep it safe).
        self.curiosity = CuriosityDrive(seed=seed) if curiosity else None
        self.curiosity_gain = curiosity_gain
        self._dev_rng = random.Random(seed + 7)                 # task-selection rng for develop() (kept off step()'s rng)
        # opt-in SHARED LATENT Q (the transfer substrate): replace the TABULAR per-context q[unit][action] with a
        # shared linear action-value Q(x,a)=W[a].x over the LAKE-STORED pattern representation x (the [0,1]^64
        # encode_multimodal vector the lake records; basal_ganglia/latent_q.LatentQ). ONE weight set across ALL
        # contexts, so a reward update generalizes to nearby contexts -> mastery TRANSFERS (unlike tabular, which
        # learns each context in isolation). HONEST FINDING (v3.17): I first tried this over the SELF-SUPERVISED
        # JEPA latent, but a linear Q over it is at CHANCE -- the SSL latent is trained to PREDICT, not aligned
        # with the reward/task, so it is not linearly separable for the task (measured 0.33 vs 1.00 over the raw
        # pattern). So the substrate rides the lake-stored PATTERN, which IS task-linear. ADDITIVE to safety: it
        # only changes HOW the action VALUE is computed -- the reflex/E-STOP runs first and every command still
        # flows through the gated ArmSim. Flag OFF => the tabular q path is byte-identical.
        self._latent_q_on = latent_q
        self.latent_q = None                                    # lazily built on the first tick (sized to len(pattern))

    def _select(self, unit, threat=0.0, intero=0.0, pfc_bias=None, qa=None):
        if qa is None:                                          # default = the tabular row; the latent-Q row is passed in
            qa = self.q[unit]
        if self.rng.random() < self.explore:
            return self.rng.choice(ACTIONS)                     # exploration: cortical variability proposes
        if self.bg is not None:
            return self._bg_select(qa, threat, intero, pfc_bias)    # BG Go/NoGo gate disinhibits the exploit choice
        bias = self.cort.caution_bias() + threat + intero       # stress + learned threat + interoceptive load -> favour safe
        td = pfc_bias or [0.0] * len(ACTIONS)                   # prefrontal top-down preference per action
        return max(ACTIONS, key=lambda a: qa[a] + (bias if a == SAFE_ACTION else 0.0) + td[ACTIONS.index(a)])

    def _bg_select(self, qa, threat=0.0, intero=0.0, pfc_bias=None):
        """Basal-ganglia Go/NoGo selection over the 3 actions. The action values are the per-channel
        cortical drive; dopamine is the winner's value confidence, DAMPED by stress + learned threat +
        interoceptive load (more caution -> less dopamine -> harder to release). The circuit disinhibits at most one
        channel; if the tonic GPi is never overcome (no drive / low DA) it releases NOTHING, which
        falls back to the safe action. Selection only ever picks among the SAME actions -- the gated
        arm is untouched and the reflex/E-STOP (checked earlier in step) still preempts it."""
        td = pfc_bias or [0.0] * len(ACTIONS)                   # prefrontal top-down preference raises the goal action's drive
        drive = [max(0.0, qa[a]) + td[i] for i, a in enumerate(ACTIONS)]
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

    def _pfc_bias(self, unit, pattern=None, qa=None):
        """The prefrontal top-down bias over ACTIONS for this unit (all-zeros when opt-out). The
        executive deliberates over the current action-values, holds a goal in working memory, and
        biases selection toward it -- a top-down PREFERENCE; it selects no action and gates no motion.

        When the JEPA latent world model is on (and a reward-derived latent goal has formed), the executive
        instead PLANS over it -- model-based lookahead in LATENT space: z = enc(context), then commit the
        action whose bounded latent rollout best reaches the latent goal (PrefrontalExecutive.plan over the
        JEPA predictor). The result is the SAME top-down preference into the SAME gated selection; it drives
        no motion and adds no actuator route."""
        if self.pfc is None:
            return [0.0] * len(ACTIONS)
        # (highest priority) SERVED H-JEPA: accumulate the Level-0 latent window and, once it is full and an
        # abstract goal has formed, the executive PLANS over the ABSTRACT (Level-1) latent (rolls the L1
        # predictor). The abstract latent summarises the recent K-tick situation, so the goal tracks the slow
        # regime rather than the instantaneous frame. Still just a gated top-down bias; drives no motion.
        if self.hjepa_serve is not None and pattern is not None:
            self._hj_window.append(self.hjepa_serve.encode0(pattern))
            if len(self._hj_window) > self.hjepa_serve.K:
                self._hj_window.pop(0)
            if len(self._hj_window) == self.hjepa_serve.K and self._z1_goal is not None:
                z1 = self.hjepa_serve.encode1(self._hj_window)
                zg = self._z1_goal

                def goal_value(z):
                    return -sum(abs(z[i] - zg[i]) for i in range(len(zg)))
                self.pfc.plan(z1, ACTIONS, self.hjepa_serve, goal_value, horizon=self.jepa_horizon)
                return self.pfc.top_down_bias(strength=self.pfc_gain)
            # window still filling / no abstract goal yet -> fall through to the reactive bootstrap below
        if self.jepa is not None and self._z_goal is not None and pattern is not None:
            z = self.jepa.encode(pattern)
            self.pfc.plan(z, ACTIONS, self.jepa, self.jepa.make_goal_value(self._z_goal),
                          horizon=self.jepa_horizon)            # latent lookahead -> a working-memory goal
        else:
            row = qa if qa is not None else self.q[unit]        # the shared latent-Q row when on, else the tabular row
            evidence = [max(0.0, row[a]) for a in ACTIONS]
            self.pfc.update(evidence)                           # reactive gated deliberation -> a WM goal
        return self.pfc.top_down_bias(strength=self.pfc_gain)

    def _lake_write(self, pattern, goal, action):
        """Store this tick in the lake working memory (opt-in). The active bits of the context pattern,
        a one-hot GOAL, and a one-hot ACTION are appended to the durable spike lake, so the working
        memory lives in the data lake (read back via self.lake_wm.recall()). Layout: channels [0, P) =
        pattern; [P, P+A) = goal one-hot; [P+A, P+2A) = action one-hot (P=len(pattern), A=len(ACTIONS)).
        Storage only -- it records AFTER the action and never affects selection or motion."""
        if not self._lake_wm_on:
            return
        p, n_act = len(pattern), len(ACTIONS)
        if self.lake_wm is None:
            self.lake_wm = LakeWorkingMemory(p + 2 * n_act, path=self._lake_wm_path, seed=self._seed)
            self._pattern_len = p
        active = [i for i, v in enumerate(pattern) if v]
        if goal is not None:
            active.append(p + goal)
        active.append(p + n_act + ACTIONS.index(action))
        self.lake_wm.write(self._tick_t, active)
        self._tick_t += 1

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

    def _competence(self, tasks):
        """Fraction of a stage's contexts whose greedy learned action already matches the optimal action.
        Contexts seen during training are familiar, so quantizing them here grows no new unit."""
        ok = 0
        for ctx, opt in tasks:
            pat = self._context_pattern(ctx["visual"], ctx["temperature"], ctx["audio"], ctx["distance"])
            if self.latent_q is not None:                       # shared latent-Q: greedy over the lake pattern
                if self.latent_q.greedy(pat) == opt:
                    ok += 1
            else:
                unit, _grew = self.learner.observe(pat)
                if unit in self.q and self.greedy(unit) == opt:
                    ok += 1
        return ok / len(tasks) if tasks else 0.0

    def develop(self, stages, curriculum, batch=6, max_batches=50):
        """SELF-PACED developmental learning: work on the CURRENT curriculum stage's tasks and ADVANCE to the
        next stage AUTONOMOUSLY when this arm's own learning progress (curiosity) on it dries up -- the
        Curriculum reads mastery off the learning-progress signal (not a clock). Needs curiosity + jepa.
        `stages` = a list of stages, each a list of (context_dict, optimal_action); reward = +1 for the optimal
        action else -1. Learning flows through the SAME gated step() -- the reflex/E-STOP still preempts and no
        new motor path is added. Returns a per-stage trace (stage, mastered, competence, advanced)."""
        trace = []
        while not curriculum.done:
            s = curriculum.stage
            tasks = stages[s]
            advanced = False
            for _ in range(max_batches):
                lp_sum, n = 0.0, 0
                for _ in range(batch):
                    ctx, opt = tasks[self._dev_rng.randrange(len(tasks))]
                    ev = self.step(reward_fn=(lambda u, a, o=opt: 1.0 if a == o else -1.0), **ctx)
                    if ev["unit"] is not None and self.curiosity is not None:
                        lp_sum += self.curiosity.learning_progress(ev["unit"])
                        n += 1
                if curriculum.observe(lp_sum / n if n else 0.0):
                    advanced = True
                    break
            trace.append({"stage": s, "mastered": curriculum.mastered,
                          "competence": self._competence(tasks), "advanced": advanced})
            if not advanced:
                break                                           # stalled on this stage (never mastered) -> stop
        return trace

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
        if self._jepa_on and self.jepa is None:
            self.jepa = JepaWorldModel(ACTIONS, in_dim=len(pattern), seed=self._seed)  # lazily sized on first tick
        # value row: the SHARED latent-Q over the JEPA latent (transfer substrate), else the tabular per-context row
        z_lq = None
        qa = self.q[unit]
        if self._latent_q_on:
            if self.latent_q is None:
                self.latent_q = LatentQ(ACTIONS, len(pattern), seed=self._seed)  # shared value over the lake-stored pattern
            z_lq = pattern                                       # the [0,1]^64 lake representation (task-linear, stable)
            qa = self.latent_q.values(z_lq)                     # shared value row -> generalizes across contexts
        threat = self._threat(pattern)                          # learned amygdala threat for this context (0 if opt-out)
        intero = self._interocept(homeostatic_load)             # interoceptive caution from homeostatic load (0 if opt-out)
        pfc_bias = self._pfc_bias(unit, pattern, qa)           # prefrontal top-down goal preference over qa (latent-Q or tabular)
        # 3. CHOOSE (learned, not a fixed map) + ACT through the gated arm
        action = self._select(unit, threat=threat, intero=intero, pfc_bias=pfc_bias, qa=qa)   # threat/intero caution + PFC bias
        state = self.arm.apply(action)                          # gated: frozen after E-STOP, collision-checked
        goal_val = self.pfc.goal if self.pfc is not None else None   # the prefrontal working-memory goal
        self._lake_write(pattern, goal_val, action)             # opt-in: store this tick in the lake working memory
        # 4. reward -> RPE action-value update + cortisol. CURIOSITY (opt-in): the JEPA prediction error for
        #    the just-lived transition becomes a learning-progress INTRINSIC reward, so the arm learns EVEN
        #    WITHOUT an external reward (self-directed exploration). The intrinsic reward feeds the value update
        #    only -- no motor route, no stress; the reflex/E-STOP + caution still preempt.
        r_int = 0.0
        if self.curiosity is not None and self.jepa is not None and self._jepa_prev is not None:
            perr = self.jepa.prediction_error(self._jepa_prev[0], self._jepa_prev[1], pattern)
            r_int = self.curiosity_gain * self.curiosity.intrinsic_reward(unit, perr)  # observes + rewards learning progress
        if reward_fn is not None or r_int != 0.0:
            r_ext = reward_fn(unit, action) if reward_fn is not None else 0.0
            r = r_ext + r_int                               # external task reward + intrinsic curiosity reward
            if self.latent_q is not None and z_lq is not None:
                self.latent_q.update(z_lq, action, r)       # SHARED latent-Q update -> generalizes to nearby-z contexts (transfer)
            else:
                self.q[unit][action] += self.lr * self.cort.learn_rate_scale(r) * (r - self.q[unit][action])
            self.cort.step(max(0.0, -r_ext))                # stress from EXTERNAL negative reward only (curiosity is not stress)
            if reward_fn is not None:                       # the task-reward-only paths (replay + latent goals), unchanged
                if self.replay and self.latent_q is None:   # SWR consolidation: TABULAR value table ONLY, no motor command
                    self.buffer.add(unit, action, r_ext)
                    if not grew:                            # quiescent (familiar) tick -> replay past experience
                        self.buffer.replay(self.q, self.lr, batch=self.replay_batch)
                if self.jepa is not None and r_ext > 0.0:   # positive TASK reward -> move the latent goal here (EMA)
                    zt = self.jepa.encode(pattern)
                    self._z_goal = zt if self._z_goal is None else \
                        [0.9 * g + 0.1 * z for g, z in zip(self._z_goal, zt)]
                if self.hjepa_serve is not None and r_ext > 0.0 and len(self._hj_window) == self.hjepa_serve.K:
                    z1t = self.hjepa_serve.encode1(self._hj_window)   # abstract latent at a rewarded tick -> the goal EMA
                    self._z1_goal = z1t if self._z1_goal is None else \
                        [0.9 * g + 0.1 * z for g, z in zip(self._z1_goal, z1t)]
        # JEPA (opt-in): learn the latent world model ONLINE from the live (z_prev, a_prev, z_now) transition.
        # OFF the safety path -- after the gated action + lake write, drives no motion, adds no actuator route.
        if self.jepa is not None:
            if self._jepa_prev is not None:
                pp, pa = self._jepa_prev
                self.jepa.train_step(pp, pa, pattern)           # one self-supervised latent-prediction update
            self._jepa_prev = (pattern, action)
        # workspace: perception (the current unit) competes with the amygdala's threat salience (the
        # REAL learned threat when the amygdala is wired in, else a low background constant) -> broadcast.
        salience = threat if self.amyg is not None else 0.2
        broadcast = self._workspace_broadcast(unit, salience)   # additive annotation; drives no motion
        return {"unit": unit, "grew": grew, "action": action, "reflex": False,
                "threat": threat, "intero": intero, "goal": goal_val,
                "wm_events": (self.lake_wm.n_events() if self.lake_wm is not None else 0),
                "workspace": broadcast, "state": state}


def _ctx(v, tmp, au, d):
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


def _tiny_hjepa(path, in_dim, k, l0, l1, actions, seed):
    """Write a tiny valid snn_hjepa/1 file (hand-set single-layer MLPs) for the self-check: it exercises the
    served-H-JEPA WIRING + safety deterministically. NOT a trained model (the real GPU-trained arm H-JEPA is
    research/hjepa_arm_torch.py); the wiring/safety is what CI covers here."""
    import json as _json
    rng = random.Random(seed)
    n = len(actions)

    def _lin(o, i):
        return [[rng.uniform(-0.3, 0.3) for _ in range(i)] for _ in range(o)]

    def _enc(i, o):
        return [[_lin(o, i), [0.0] * o]]                    # one Linear layer (in -> out); a valid minimal MLP
    obj = {
        "format": "snn_hjepa/1", "in_dim": in_dim, "K": k, "latent0": l0, "latent1": l1,
        "fine_actions": list(actions), "coarse_actions": list(actions), "tau": 0.99,
        "enc0": _enc(in_dim, l0), "enc1": _enc(l0, l1),
        "enc0_target": _enc(in_dim, l0), "enc1_target": _enc(l0, l1),
        "pred0": _enc(l0 + n, l0), "pred1": _enc(l1 + n, l1),
        "predictors": {"pred0": "residual_mlp_unit", "pred1": "mlp_unit"}, "meta": {},
    }
    with open(path, "w", encoding="ascii") as f:
        _json.dump(obj, f)


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

    # 12. PREFRONTAL EXECUTIVE wired live: it deliberates over the action-values, commits a GOAL to
    #     working memory, and top-down-biases selection toward it -- goal-directed, caution-safe.
    pf = AutonomousArm(seed=1, pfc=True)
    for _ in range(90):
        pf.step(**a, reward_fn=reward)                          # reward GRIPPER_OPEN
    pfc_goal = pf.pfc.goal                                      # the executive's held working-memory goal
    pfc_bias = pf.pfc.top_down_bias()
    goal_is_rewarded = pfc_goal == ACTIONS.index("GRIPPER_OPEN")
    held = all(pf.pfc.maintain() == pfc_goal for _ in range(5))   # WM holds the goal across an input-free delay
    pf_guard = AutonomousArm(seed=2, pfc=True)                  # the safety envelope must hold with it on
    pf_ev = pf_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    pf_frozen0 = pf_guard.arm.state()["stopped"]
    pf_guard.step(**a)                                          # an action after the E-STOP
    pf_frozen1 = pf_guard.arm.state()["stopped"]

    def run_pf(seed):
        bot = AutonomousArm(seed=seed, pfc=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(f"prefrontal  : deliberates over values -> working-memory goal {ACTIONS[pfc_goal]} "
          f"(rewarded={goal_is_rewarded}, held across delay={held}); E-STOP freeze holds with it on ({pf_frozen1})")
    print("=" * 62)

    # the prefrontal executive is opt-in; a default arm has none (byte-for-byte the old behaviour).
    assert AutonomousArm(seed=0).pfc is None, "the prefrontal executive is opt-in (a default arm has none)"
    # it commits the high-value (rewarded) action as its goal and holds it in working memory.
    assert goal_is_rewarded, f"the executive must commit the rewarded action as its goal ({ACTIONS[pfc_goal]})"
    assert pfc_bias[pfc_goal] == max(pfc_bias) and max(pfc_bias) > 0.0, "the top-down bias must favour the held goal"
    assert held, "working memory must hold the goal across an input-free delay (persistent activity)"
    # the safety envelope holds with the executive wired in (top-down preference only, never a motor route).
    assert pf_ev["action"] == "EMERGENCY_STOP" and pf_ev["reflex"], "danger must E-STOP with the prefrontal executive on"
    assert pf_frozen0 and pf_frozen1, "the E-STOP freeze must hold with the executive wired in (top-down preference only)"
    assert run_pf(5) == run_pf(5), "the prefrontal path must be deterministic"

    # 13. LAKE WORKING MEMORY wired live: each tick's activity is stored in the durable spike lake and
    #     recalled from it -- the data lake IS the working memory (opt-in, storage only, gate untouched).
    lw = AutonomousArm(seed=1, pfc=True, lake_wm=True)
    for _ in range(8):
        lw.step(**a, reward_fn=reward)
    wm = lw.lake_wm.recall(4)                                   # the recent working-memory window, read from the lake
    wm_events = lw.lake_wm.n_events()
    lw.lake_wm.reopen()                                         # discard the in-RAM object -> replay purely from the .spkl
    wm_after = lw.lake_wm.recall(4)
    lake_backed = wm_events > 0 and wm == wm_after and len(wm) == 4
    lw_guard = AutonomousArm(seed=2, lake_wm=True)              # the safety envelope must hold with it on
    lw_ev = lw_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    lw_frozen0 = lw_guard.arm.state()["stopped"]
    lw_guard.step(**a)                                          # a normal tick after the E-STOP
    lw_frozen1 = lw_guard.arm.state()["stopped"]

    def run_lw(seed):
        bot = AutonomousArm(seed=seed, pfc=True, lake_wm=True)
        acts = [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]
        bot.lake_wm.close(remove=True)
        return acts

    det_lw = run_lw(5) == run_lw(5)
    lw.lake_wm.close(remove=True)                              # clean up the temp lake files
    lw_guard.lake_wm.close(remove=True)

    print(f"lake WM     : {wm_events} spikes in the durable lake; recall holds {len(wm)} recent ticks; "
          f"survives reopen={wm == wm_after}; E-STOP freeze holds with it on ({lw_frozen1})")
    print("=" * 62)

    assert AutonomousArm(seed=0).lake_wm is None, "lake working memory is opt-in (a default arm has none)"
    assert lake_backed, "the working memory must live in the durable lake (survive reopen) with a bounded window"
    assert lw_ev["action"] == "EMERGENCY_STOP" and lw_ev["reflex"], "danger must E-STOP with lake WM on"
    assert lw_frozen0 and lw_frozen1, "the E-STOP freeze must hold with lake WM wired in (storage only)"
    assert det_lw, "the lake-WM path must be deterministic"

    # 14. JEPA latent world model wired live into the REACTIVE path: it learns a latent world model ONLINE
    #     from the tick stream, and the executive PLANS over it (in latent space) to set its goal -- a
    #     bounded top-down bias into the SAME gated selection; opt-in, drives no motion, reflex/E-STOP first.
    jp = AutonomousArm(seed=1, pfc=True, jepa=True)
    for _ in range(90):
        jp.step(**a, reward_fn=reward)                          # reward GRIPPER_OPEN
    jp_std = jp.jepa.latent_std()                              # a non-degenerate learned latent (did not collapse)
    jp_goal_formed = jp._z_goal is not None                    # a reward-derived latent goal formed from the stream
    jp_goal = jp.pfc.goal                                      # the executive's goal, now set by JEPA latent planning
    jp_guard = AutonomousArm(seed=2, pfc=True, jepa=True)       # the safety envelope must hold with it on
    jp_ev = jp_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    jp_frozen0 = jp_guard.arm.state()["stopped"]
    jp_guard.step(**a)                                          # a JEPA-planned action after the E-STOP
    jp_frozen1 = jp_guard.arm.state()["stopped"]

    def run_jp(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(f"jepa        : latent world model learns online (latent std {jp_std:.3f}); reward-latent goal "
          f"formed={jp_goal_formed}; executive goal {ACTIONS[jp_goal] if jp_goal is not None else None} set by "
          f"latent planning; E-STOP freeze holds with it on ({jp_frozen1})")
    print("=" * 62)

    # JEPA is opt-in; a default arm has none (byte-for-byte the old behaviour).
    assert AutonomousArm(seed=0).jepa is None, "the JEPA world model is opt-in (a default arm has none)"
    assert jp.jepa is not None and jp_std > 0.0, "the JEPA world model must be wired in and learn a non-degenerate latent"
    assert jp_goal_formed, "a reward-derived latent goal must form from the live stream"
    assert jp_goal is not None, "the executive must hold a goal set by JEPA latent planning"
    # the safety envelope holds with JEPA wired LIVE into the reactive path (top-down preference only, no motor route).
    assert jp_ev["action"] == "EMERGENCY_STOP" and jp_ev["reflex"], "danger must E-STOP with JEPA on"
    assert jp_frozen0 and jp_frozen1, "the E-STOP freeze must hold with JEPA wired live (top-down preference only)"
    assert run_jp(5) == run_jp(5), "the JEPA path must be deterministic"

    # 15. SERVED H-JEPA wired live: the executive plans over the ABSTRACT (Level-1) latent of a GPU-trained
    #     2-level hierarchical JEPA (cerebellum/hjepa_serve). A hand-built tiny served model exercises the
    #     WIRING + safety here (the real arm-trained model is research/hjepa_arm_torch.py, GPU); opt-in.
    import os as _os
    import tempfile as _tf
    hj_path = _os.path.join(_tf.gettempdir(), "snn_hjepa_arm_test.json")
    _tiny_hjepa(hj_path, 64, 4, 4, 3, ACTIONS, seed=0)
    served = HJepaServe.load(hj_path)
    _os.remove(hj_path)
    hja = AutonomousArm(seed=1, pfc=True, hjepa_serve=served)
    for _ in range(90):
        hja.step(**a, reward_fn=reward)
    hj_windowed = len(hja._hj_window) == served.K
    hj_goal_formed = hja._z1_goal is not None
    hj_goal = hja.pfc.goal
    hj_guard = AutonomousArm(seed=2, pfc=True, hjepa_serve=served)     # the safety envelope must hold with it on
    hj_ev = hj_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    hj_frozen0 = hj_guard.arm.state()["stopped"]
    hj_guard.step(**a)
    hj_frozen1 = hj_guard.arm.state()["stopped"]

    def run_hj(seed):
        bot = AutonomousArm(seed=seed, pfc=True, hjepa_serve=served)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(f"hjepa-serve : abstract (Level-1) H-JEPA planning -- K-window filled={hj_windowed}, abstract goal "
          f"formed={hj_goal_formed}; executive goal {ACTIONS[hj_goal] if hj_goal is not None else None} set by "
          f"abstract planning; E-STOP freeze holds with it on ({hj_frozen1})")
    print("=" * 62)

    assert AutonomousArm(seed=0).hjepa_serve is None, "the served H-JEPA is opt-in (a default arm has none)"
    assert hj_windowed, "the Level-0 latent K-window must fill"
    assert hj_goal_formed, "a reward-EMA abstract (Level-1) goal must form from the live stream"
    assert hj_goal is not None, "the executive must hold a goal set by ABSTRACT H-JEPA planning"
    # the safety envelope holds with the served H-JEPA driving abstract planning (top-down preference only, no motor route).
    assert hj_ev["action"] == "EMERGENCY_STOP" and hj_ev["reflex"], "danger must E-STOP with the served H-JEPA on"
    assert hj_frozen0 and hj_frozen1, "the E-STOP freeze must hold with abstract H-JEPA planning (top-down only)"
    assert run_hj(5) == run_hj(5), "the served-H-JEPA path must be deterministic"

    # 16. CURIOSITY (intrinsic motivation): the JEPA world-model prediction error becomes a learning-progress
    #     intrinsic reward that drives SELF-DIRECTED learning -- the arm learns EVEN WITH NO external reward.
    #     Opt-in, needs the JEPA; drives no motion, reflex/E-STOP still preempts.
    cur = AutonomousArm(seed=1, pfc=True, jepa=True, curiosity=True)
    for _ in range(60):
        cur.step(**a)                                          # NO reward_fn -> ONLY curiosity drives learning
    cur_learned = any(abs(v) > 1e-6 for row in cur.q.values() for v in row.values())
    dull = AutonomousArm(seed=1, pfc=True, jepa=True, curiosity=False)   # fair control: no curiosity AND no reward
    for _ in range(60):
        dull.step(**a)
    dull_learned = any(abs(v) > 1e-6 for row in dull.q.values() for v in row.values())
    cur_guard = AutonomousArm(seed=2, pfc=True, jepa=True, curiosity=True)   # the safety envelope must hold with it on
    cur_ev = cur_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    cur_frozen0 = cur_guard.arm.state()["stopped"]
    cur_guard.step(**a)
    cur_frozen1 = cur_guard.arm.state()["stopped"]

    def run_cur(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True, curiosity=True)
        return [bot.step(**ctx)["action"] for _ in range(6) for ctx in (a, b, c)]
    cur_det = run_cur(5) == run_cur(5)

    print(f"curiosity   : self-directed learning with NO external reward -- curious arm's values changed="
          f"{cur_learned} vs no-curiosity control changed={dull_learned}; E-STOP freeze holds with it on ({cur_frozen1})")
    print("=" * 62)

    assert AutonomousArm(seed=0).curiosity is None, "curiosity is opt-in (a default arm has none)"
    assert cur_learned, "curiosity must drive value learning WITHOUT any external reward (self-directed)"
    assert not dull_learned, "the fair control: no curiosity AND no external reward -> nothing learns"
    assert cur_ev["action"] == "EMERGENCY_STOP" and cur_ev["reflex"], "danger must E-STOP with curiosity on"
    assert cur_frozen0 and cur_frozen1, "the E-STOP freeze must hold with curiosity wired in (intrinsic reward, no motor route)"
    assert cur_det, "the curiosity path must be deterministic"

    # 17. CURRICULUM (self-paced developmental learning): the arm works stages of increasing difficulty and
    #     ADVANCES autonomously when its own learning progress (curiosity) on a stage dries up -- cumulative,
    #     self-graduating. Opt-in (needs curiosity + jepa); learning flows through the SAME gated step().
    devC = [(_ctx(0.9, 0.1, 0.1, safe_d), "GRIPPER_OPEN"),     # 4 DISTINCT contexts, each with an optimal action
            (_ctx(0.1, 0.9, 0.1, safe_d), "GRIPPER_CLOSE"),
            (_ctx(0.1, 0.1, 0.9, safe_d), "HOLD"),
            (_ctx(0.9, 0.9, 0.1, safe_d), "GRIPPER_OPEN")]
    dev_stages = [devC[:1], devC[:2], devC[:3], devC[:4]]      # 1,2,3,4 contexts -> 4 stages of increasing difficulty
    dev = AutonomousArm(seed=1, pfc=True, jepa=True, curiosity=True)
    dtrace = dev.develop(dev_stages, Curriculum(n_stages=len(dev_stages), seed=1))
    dev_mastered = dtrace[-1]["mastered"] if dtrace else 0     # stages the arm graduated itself through
    dev_final_comp = dtrace[-1]["competence"] if dtrace else 0.0
    dev_guard = AutonomousArm(seed=2, pfc=True, jepa=True, curiosity=True)   # safety with the developmental loop
    dev_guard.develop([[(_ctx(0.5, 0.5, 0.5, danger_d), "HOLD")]], Curriculum(n_stages=1, seed=2), batch=2, max_batches=1)
    dev_frozen = dev_guard.arm.state()["stopped"]              # a danger context in the curriculum still E-STOPs + freezes

    def run_dev(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True, curiosity=True)
        return [t["mastered"] for t in bot.develop(dev_stages, Curriculum(n_stages=len(dev_stages), seed=seed))]
    dev_det = run_dev(5) == run_dev(5)

    print(f"curriculum  : self-paced developmental learning -- arm graduated itself through {dev_mastered}/"
          f"{len(dev_stages)} stages (final-stage competence {dev_final_comp:.2f}); safety freeze holds ({dev_frozen})")
    print("=" * 62)

    assert AutonomousArm(seed=0)._dev_rng is not None, "the arm carries a developmental task-selection rng"
    assert dev_mastered >= 2, f"the arm must autonomously master + advance through multiple curriculum stages ({dev_mastered})"
    assert all(t["advanced"] for t in dtrace[:dev_mastered]), \
        "each mastered stage must have ADVANCED on its own learning-progress dry-up (self-paced, not a clock)"
    assert dev_final_comp >= 0.5, f"the arm must reach real competence on the stages it masters ({dev_final_comp:.2f})"
    assert dev_frozen, "the E-STOP freeze must hold with the developmental curriculum loop (learning through the gated step)"
    assert dev_det, "the developmental curriculum path must be deterministic"

    # 18. SHARED LATENT-Q (the transfer substrate): a shared linear action-value over the LAKE-STORED pattern
    #     replaces the tabular per-context q, so mastery GENERALIZES to genuinely-novel contexts (NEW units)
    #     where tabular has no learned value. Opt-in; only changes HOW the value is computed (gated selection +
    #     reflex/E-STOP unchanged). HONEST FINDING: a linear Q over the SELF-SUPERVISED JEPA latent is at chance
    #     (not task-aligned; measured 0.33 vs 1.00 over the pattern) -- the substrate rides the task-linear lake
    #     pattern, and the deep clean transfer proof is in basal_ganglia/latent_q's own self-check.
    def _lq_ctx(v, t, au):
        return _ctx(v, t, au, safe_d)
    LQ_RULE = ["GRIPPER_OPEN", "GRIPPER_CLOSE", "HOLD"]         # optimal = argmax over (visual, temperature, audio) band
    lq_train = [(_lq_ctx(*[0.9 if i == d else 0.1 for i in range(3)]), LQ_RULE[d]) for d in range(3) for _ in range(3)]
    lq_hold = []                                               # dominant band + a partial secondary -> NEW units, SAME rule
    for d in range(3):
        for sec in range(3):
            if sec != d:
                lv = [0.2, 0.2, 0.2]
                lv[d], lv[sec] = 0.8, 0.45
                lq_hold.append((_lq_ctx(*lv), LQ_RULE[d]))

    def _lq_run(use_latent_q):
        bot = AutonomousArm(seed=1, pfc=True, jepa=True, curiosity=True, latent_q=use_latent_q)
        for _ in range(500):
            ctx, opt = lq_train[bot._dev_rng.randrange(len(lq_train))]
            bot.step(reward_fn=(lambda u, a, o=opt: 1.0 if a == o else -1.0), **ctx)
        return bot._competence(lq_train), bot._competence(lq_hold)
    lqq_train, lqq_hold = _lq_run(True)
    tab_train, tab_hold = _lq_run(False)
    lq_guard = AutonomousArm(seed=2, pfc=True, jepa=True, latent_q=True)   # the safety envelope must hold with it on
    lq_ev = lq_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    lq_guard.step(**_lq_ctx(0.9, 0.1, 0.1))
    lq_frozen = lq_guard.arm.state()["stopped"]

    def run_lq(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True, curiosity=True, latent_q=True)
        return [bot.step(reward_fn=(lambda u, a: 1.0 if a == "GRIPPER_OPEN" else -1.0), **lq_train[0][0])["action"]
                for _ in range(8)]
    lq_det = run_lq(5) == run_lq(5)

    print(f"latent-Q    : SHARED value over the lake pattern -- held-out (NEW units, same rule) competence "
          f"{lqq_hold:.2f} vs tabular {tab_hold:.2f} (TRANSFER); train {lqq_train:.2f}; freeze holds ({lq_frozen})")
    print("=" * 62)

    assert AutonomousArm(seed=0).latent_q is None, "the shared latent-Q is opt-in (a default arm has none)"
    assert lqq_train >= 0.8, f"the shared latent-Q must learn the rule family with ONE shared function ({lqq_train:.2f})"
    assert lqq_hold >= 0.8 and lqq_hold >= tab_hold + 0.3, \
        f"the shared latent-Q must GENERALIZE to new-unit contexts where tabular cannot ({lqq_hold:.2f} vs {tab_hold:.2f})"
    assert lq_ev["action"] == "EMERGENCY_STOP" and lq_ev["reflex"] and lq_frozen, "danger must E-STOP with the latent-Q on"
    assert lq_det, "the latent-Q path must be deterministic"

    print(f"self-check OK: grows a unit per novel context ({grew_to}); learns the rewarded action "
          f"({default_before}->{learned}); danger -> gated E-STOP + freeze holds; deterministic. "
          f"replay: decisive in {replay_ticks}<{live_ticks} live ticks, value-only, gate holds. "
          f"amygdala learns threat ({threat_danger:.2f}>{threat_safe:.2f}) + BG gate releases {bg_released}, "
          f"both caution-only (E-STOP freeze holds)")


if __name__ == "__main__":
    main()
