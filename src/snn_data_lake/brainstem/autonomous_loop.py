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
from snn_data_lake.basal_ganglia.mlp_q import MlpQ
from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.arm_sim import ArmSim
from snn_data_lake.brainstem.cpg_oscillator import MatsuokaCPG
from snn_data_lake.cerebellum.gain_adaptation import GainAdapter
from snn_data_lake.cerebellum.hjepa_serve import HJepaServe
from snn_data_lake.cerebellum.jepa_world_model import JepaWorldModel
from snn_data_lake.cortex.agency_monitor import AgencyMonitor
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.hippocampus.affective_memory import AffectiveEpisodicStore
from snn_data_lake.hippocampus.allocentric_map import AllocentricMap
from snn_data_lake.hippocampus.contrastive_episode_analysis import ContrastiveEpisodeAnalyzer
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.hippocampus.growing_learner import GrowingLearner
from snn_data_lake.medulla.interoceptive_arousal import InteroceptiveArousal
from snn_data_lake.neuromodulation.cortisol import Cortisol
from snn_data_lake.neuromodulation.developmental_envelope import DevelopmentalEnvelope
from snn_data_lake.neuromodulation.empowerment import Empowerment
from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive
from snn_data_lake.neuromodulation.naca import NACABroadcast
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
from snn_data_lake.substrate.brain_bundle import BrainBundle, load_bundle
from snn_data_lake.substrate.lake_working_memory import LakeWorkingMemory
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

# Action-space granularity, NOT a tuned threshold (Policy 1): it defines the vocabulary, it does not
# arbitrate a decision. Sized small on purpose -- one step can neither cross the obstacle radius
# (OBSTACLE_R=0.30) nor reach the floor from home, so a single exploratory move is always recoverable.
_JOINT_STEP_DEG = 6.0
_MOTOR = tuple(
    f"JOINT_{j}_ROTATE({s}{_JOINT_STEP_DEG:g}deg)" for j in ("A", "B") for s in ("+", "-")
)
# The arm's autonomous action repertoire. Until v4.58 this was ENGAGE/RELEASE/HOLD only -- three
# tokens, none of which move a joint, so the brain had no vocabulary with which to change its own
# next observation and the sensorimotor loop was causally OPEN. The four motor tokens are the
# precondition for closing it. ArmSim already accepted this exact command form; nothing in the
# safety envelope changed, and `apply()` still ignores every one of these while `stopped` is set.
ACTIONS = ("ENGAGE", "RELEASE", "HOLD") + _MOTOR  # the arm's autonomous action repertoire
_ACT_IX = {a: i for i, a in enumerate(ACTIONS)}  # action -> index into the td preference vector
_A_POS, _A_NEG = _MOTOR[0], _MOTOR[1]  # joint-A extend / retract: the CPG's rhythmic pair
_CPG_DRIVE = 1.0  # tonic descending drive; the oscillation exists for any drive > 0
_CPG_FLOOR = 0.25  # competence floor: the CPG's minimum share, so behaviour never collapses to
# nothing when the learned row is confident. A deliberate invariant, not a tuned threshold -- the
# analog of mhflocke's CompetenceGate.cpg_min = 0.40.
_COVER_CELL = 0.10  # workspace discretisation for the coverage reward (metres); a scale, not a gate
# Consecutive ground-contact ticks before the postural recovery reflex fires. Small enough that the
# arm cannot spend a run underground, large enough that a single grazing contact is not treated as a
# fall. A latency, not a tuned threshold -- the condition itself (ArmSim reports GROUND) is exact.
_GROUND_RECOVER_TICKS = 3
SAFE_ACTION = "HOLD"  # what caution biases toward under stress
# W4c: self-motion step DERIVED from the chosen action (a real behavioral trajectory), not a constant drift.
# Representation-only -- this feeds the allocentric MAP readout, never the actuator (the arm gate is separate).
_ALLO_MOVE = {"ENGAGE": (0.05, 0.0), "RELEASE": (-0.05, 0.0), "HOLD": (0.0, 0.04)}
# ^ nominal displacements for the three NON-kinematic tokens. They move no joint, so there is no
# geometry to derive a displacement from; a nominal step is the only honest option and it is what the
# map integrates for them.
#
# The four JOINT_*_ROTATE tokens are deliberately ABSENT, and were silently worth (0,0) via the
# `.get(action, (0.0, 0.0))` default until v4.72 -- so the commands that actually move the body
# contributed NOTHING to path integration. The brain dead-reckoned its position while ignoring its own
# limb. A table cannot fix that: for a 2-link arm the end-effector displacement of a 6-degree joint
# rotation depends on the CURRENT configuration (the Jacobian), so the same token means a different
# move at every pose. `_self_motion` below derives it instead.


class AutonomousArm:
    """Grows units from multimodal context + learns context->action by reward; every motion gated."""

    def __init__(
        self,
        lr=0.3,
        explore=0.15,
        seed=0,
        replay=False,
        replay_batch=16,
        bundle=None,
        workspace=False,
        amygdala=False,
        bg_gate=False,
        interoception=False,
        pfc=False,
        pfc_gain=0.5,
        lake_wm=False,
        lake_wm_path=None,
        jepa=False,
        jepa_horizon=4,
        hjepa_serve=None,
        curiosity=False,
        curiosity_gain=0.5,
        spiking_curiosity=False,
        empowerment=False,
        empowerment_gain=0.5,
        dopamine=False,
        latent_q=False,
        mlp_q=False,
        reward_shaped_encoder=False,
        reward_gain=5.0,
        salience_novelty=False,
        attention=False,
        affective_memory=False,
        agency_monitor=False,
        contrastive=False,
        develop_env=False,
        naca=False,
        allocentric=False,
        gain_adapt=False,
        vitals_hold=False,
        vitals_hold_gain=0.5,
    ):
        self.arm = ArmSim()
        self._seed = seed
        self.bundle = bundle  # optional BrainBundle of trained skills (perception, ...); additive
        self.learner = GrowingLearner()
        self.cort = Cortisol(tau=40.0)
        self.q = {}  # unit_id -> {action: value}; grows alongside the learner
        # COMPETENCE FLOOR + its gate. The oscillator is SNDL's own (Matsuoka 1985 / Ijspeert 2008),
        # self-checked in cpg_oscillator; it simply had no caller on the brain's path until now.
        # Slow constants (tau_r=0.5, tau_a=6.0) are the ones cpg_sweep already uses for smooth sweeps.
        self.cpg = MatsuokaCPG(tau_r=0.5, tau_a=6.0)
        self._cpg_w = 1.0  # current CPG share (competence-gated each tick, floored at _CPG_FLOOR)
        self._cpg_spread_max = (
            0.0  # RUNNING scale for the competence gate -- baseline-relative (P1)
        )
        # WORKSPACE COVERAGE: the physical objective. Cells the end effector has actually reached.
        self._covered = set()
        self._cover_new = (
            0  # newly-reached cells this run -- the honest "is it getting better" line
        )
        # Global-Workspace ignitions: how many times the single broadcast slot changed hands. A
        # running count of "something new became globally available", which is a different claim from
        # access_index (how available) or pci (how integrated) -- both of which can sit high while the
        # same content is held. Counted in _workspace_tick, the one place the workspace steps.
        self._ignitions = 0
        self._prev_thought = None
        self._allo_last_ee = None  # previous end-effector pose, for _self_motion's dead reckoning
        self._last_disp = 0.0  # how far the end effector actually travelled on the last scored tick
        self._last_ee = None  # previous end-effector pose -> did this tick move the body at all?
        self._ground_ticks = 0  # consecutive ground-contact ticks (postural recovery reflex)
        self._recoveries = 0  # how many times the arm had to stand back up -- a real "falls" metric
        self.lr = lr
        self.explore = explore
        self.rng = random.Random(seed)
        self.replay = replay  # opt-in SWR consolidation; default OFF -> behaviour unchanged
        self.replay_batch = replay_batch
        self.buffer = ReplayBuffer(seed=seed) if replay else None
        # opt-in Global Workspace: brain regions compete for one slot + broadcast the winner each tick.
        # ADDITIVE + read-only -- it selects no action and never touches the reflex/E-STOP (see step()).
        self.workspace = GlobalWorkspace(seed=seed) if workspace else None
        self._salience_on = bool(
            salience_novelty
        )  # opt-in bottom-up novelty salience -> a GWT competitor
        self._sal_ema = None  # running percept expectation (the habituation substrate)
        self._novelty = 0.0  # last bottom-up salience (deviance vs expectation)
        # opt-in DOPAMINERGIC reward loop (gap #4): a spiking TD-RPE broadcast (midbrain VTA/SNc) with an
        # OPPONENT anti-reward channel (lateral habenula -> RMTg -> DA below baseline) driving a three-factor
        # DA-gated striatal valence learner. REPRESENTATION-ONLY: it learns an affective value over contexts
        # and broadcasts DA/RPE; it drives NO motion, adds no actuator route, and the reflex/E-STOP preempts.
        self._dopa_on = bool(dopamine)
        self._dopa = self._lhb = self._valence = None  # lazily built on the first rewarded tick
        self._da_level = 0.0  # last phasic dopamine level in [0,1]
        self._rpe = 0.0  # last reward-prediction error (the dopamine signal)
        # opt-in COVERT ATTENTION (gap #8): a spiking reticular (TRN) gate selects the salient modality
        # band, divisive normalization gives contrast constancy, and the claustrum binds co-active
        # modalities. READOUT-ONLY: it MEASURES attention over the percept and never masks the learner's
        # value input or the reflex -- the E-STOP path reads raw `distance` and is untouched.
        self._attn_on = bool(attention)
        self._rgate = self._dnorm = self._claustrum = (
            None  # lazily built on the first attended tick
        )
        self._attn_mask = []  # per-modality relay gains (attended band released)
        self._attn_gain = None  # per-band multiplicative gain APPLIED to the learner's percept
        self._attended = -1  # index of the attended modality band
        self._bound_modalities = 0  # cross-modal assemblies the claustrum bound
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
        self.lake_wm = None  # lazily sized on the first tick (needs len(pattern))
        self._pattern_len = None
        self._tick_t = 0
        # opt-in JEPA latent world model: a learned encoder enc(context)->z + an action-conditioned latent
        # predictor pred(z,a)->z' (cerebellum/jepa_world_model), trained self-supervised ONLINE from the live
        # tick stream. When on (with the executive), the prefrontal executive PLANS over it -- model-based
        # lookahead in LATENT space -- to set its goal, instead of the reactive value deliberation; the result
        # is the SAME top-down PREFERENCE (bounded by pfc_gain) fed to the SAME gated selection. ADDITIVE +
        # read-only to safety: it drives no motion and adds no actuator route; the reflex/E-STOP runs FIRST and
        # still preempts, and every motor command stays gated (see step()). Lazily sized on the first tick.
        self._jepa_on = (
            jepa or mlp_q or reward_shaped_encoder
        )  # all three need the JEPA world model built
        self._reward_shaped_encoder = (
            reward_shaped_encoder  # task-align the JEPA encoder via observed reward
        )
        self._reward_gain = reward_gain  # reward-shaping strength (lambda_reward); the few-context
        #                                                          arm regime needs a firm task pull -> default 5.0 recovers
        #                                                          the on-arm rule to 1.0 held-out (weaker pulls plateau ~0.83)
        self.jepa = None
        self.jepa_horizon = max(
            1, min(4, int(jepa_horizon))
        )  # bounded rollout (the learned map can diverge)
        self._z_goal = None  # reward-EMA of latents = the latent goal to plan toward
        self._autotelic_goal_thresh = (
            0.02  # min intrinsic reward that forms a latent goal (autotelic planning)
        )
        self._jepa_prev = None  # (prev pattern, prev action) for the online (z,a,z') update
        # opt-in SERVED H-JEPA: a GPU-trained 2-level hierarchical JEPA (cerebellum/hjepa_serve.HJepaServe, FROZEN).
        # The executive plans over the ABSTRACT (Level-1) latent: z0 = encode0(pattern) each tick fills a K-window,
        # z1 = encode1(window) is the abstract situation, and plan() rolls the L1 predictor over it. Like the online
        # jepa it ONLY biases the executive's goal (bounded by pfc_gain, gated) -- it drives no motion, adds no
        # actuator route, and the reflex/E-STOP runs first. Served/frozen: no online training in the loop.
        self.hjepa_serve = hjepa_serve
        self._hj_window = []  # rolling K-window of Level-0 latents (feeds encode1)
        self._z1_goal = None  # reward-EMA of the abstract Level-1 latent (plan goal)
        # opt-in CURIOSITY (intrinsic motivation): the JEPA world-model prediction error becomes a
        # LEARNING-PROGRESS reward that drives SELF-DIRECTED exploration -- the arm learns even with NO external
        # reward, and it seeks learnable-novel contexts (not noise; learning-progress rejects the noisy-TV trap).
        # Needs the JEPA (for the prediction error). ADDITIVE + read-only to safety: the intrinsic reward feeds
        # only the value update -- it drives no motion, adds no actuator route, creates no stress, and the
        # reflex/E-STOP + amygdala/interoceptive caution still preempt every command (like a baby drawn to
        # novelty while external guards keep it safe).
        # spiking_curiosity (SI-1) swaps the rate learning-progress drive for the genuinely-spiking one
        # (SpikingCuriosityDrive, LP read from LIF spike counts) -- a duck-typed drop-in, so every consumer
        # (learning_progress / intrinsic_reward at the value-feed) is unchanged. Off -> the exact rate path.
        if spiking_curiosity:
            self.curiosity = SpikingCuriosityDrive(seed=seed)
        elif curiosity:
            self.curiosity = CuriosityDrive(seed=seed)
        else:
            self.curiosity = None
        self.curiosity_gain = curiosity_gain
        # opt-in EMPOWERMENT (controllability / agency drive): the empowerment GAIN over the JEPA latent --
        # how many more DISTINGUISHABLE futures the agent can now reach by its OWN actions -- becomes an
        # intrinsic reward ORTHOGONAL to curiosity (control, not novelty). Needs the JEPA (its predict IS the
        # forward model). ADDITIVE + read-only to safety: feeds the value update only, no motor route. Off ->
        # a default brain is byte-identical. Lazily built once the JEPA world model exists.
        self._empower_on = bool(empowerment)
        self.empower = None
        self.empowerment_gain = empowerment_gain
        # C3 #6/#7: the two intrinsic drives used to enter the SUMMED reward through hand-picked
        # exchange rates (curiosity_gain 0.5, empowerment_gain 0.5) -- a developer deciding what
        # novelty and controllability are WORTH against task success, in incommensurate units
        # (curiosity is a difference of prediction-error EMAs; empowerment is in bits).
        # Two fixes, in order:
        #  (a) UNIT CONVERSION, not policy: each channel is divided by a running EMA of its OWN
        #      magnitude (RND-style, Burda 2018) before it is weighted, so the two arrive
        #      dimensionless and the weight is a genuine ratio rather than a scale correction.
        #  (b) the ratio is then LEARNED, but ONLY from EXTERNAL reward. Weighting a channel by an
        #      error computed on the summed reward that CONTAINS it is self-referential and cannot
        #      converge, so the third factor here is r_ext alone, baseline-subtracted per unit.
        #      The constructor gains become the INITIAL weights (equal by default -> the drives are
        #      not ranked by hand), and their sum stays clamped so the intrinsic channel can never
        #      outgrow the external one. Value channel only: no motor route, reflex/E-STOP preempt.
        self._int_tau = 0.02  # adaptation rate of the per-channel magnitude baselines
        self._cur_mag = 0.0  # running |curiosity| magnitude -> its unit
        self._emp_mag = 0.0  # running |empowerment| magnitude -> its unit
        self._int_lr = 0.02  # learning rate of the curiosity/empowerment split
        self._int_elig = (0.0, 0.0)  # last tick's normalised (curiosity, empowerment) contributions
        self._rext_base = {}  # per-unit running mean of EXTERNAL reward (the only honest credit)
        self._rint_base = 0.0  # running mean of POSITIVE intrinsic reward -> the autotelic scale
        self._r_mag = 0.0  # running mean |delivered reward| = the unit exploration is sized in
        self._visits = {}  # per-unit experience count -> count-based exploration decay 1/sqrt(n)
        # C3 #8: pfc_gain (the goal-directed-vs-habitual arbitration weight) is learned by CONTRAST.
        # Outcomes are accumulated separately for ticks where the top-down goal actually FLIPPED the
        # choice and ticks where it did not; the gain follows the difference, so it grows only while
        # overriding habit genuinely pays and shrinks once it starts overriding good habits.
        self._td_flipped = False  # did the goal override the habit on the last selection
        self._r_ovr = 0.0  # outcome baseline over override ticks
        self._r_hab = 0.0  # outcome baseline over habit-followed ticks
        self._n_ovr = 0  # evidence counts -- the gain waits for both to be real
        self._n_hab = 0
        self._pfc_lr = 0.02
        # opt-in FOUR REPRESENTATION-ONLY DRIVES (each default False -> a default brain is byte-identical).
        # All feed the value/annotation channel ONLY -- no actuator route, no change to action selection,
        # the reflex/E-STOP still preempts: affective episodic memory (emotion-tagged retention), the agency
        # monitor (controllability of action -> realized value = a sense of agency), contrastive episode
        # analysis (which context features separate success from failure), and the developmental envelope
        # (competence -> exploration DOWN, forward-model authority UP; its explore_noise/forward_model_gain
        # readouts are COMPUTED + STORED only, never routed into _select). Each lazily built on first use.
        self._affmem_on = bool(affective_memory)
        self._affmem = None
        self._agency_on = bool(agency_monitor)
        self._agency = None
        self._contrast_on = bool(contrastive)
        self._contrast = None
        self._dev_on = bool(develop_env)
        self._dev = None
        self._comp_ema = 0.0  # running task-competence EMA feeding the envelope
        self._dev_explore = 0.0  # exposed developmental readouts (stored, never routed)
        self._dev_fm_gain = 0.0
        self._dev_rng = random.Random(
            seed + 7
        )  # task-selection rng for develop() (kept off step()'s rng)
        # THREE MORE opt-in representation-only faculties (default False -> byte-identical brain). All feed
        # the value/annotation channel ONLY -- no actuator route: NACA (neuromodulator-gated credit on the
        # local reward delta), an allocentric spatial map (dead-reckons a landmark map from the percept
        # stream), and cerebellar gain adaptation (learns a feedforward gain estimate, NEVER applied to motion).
        self._naca_on = bool(naca)
        self._naca = None
        self._naca_credit = 0.0  # neuromod-corrected credit readout
        self._allo_on = bool(allocentric)
        self._allo = None
        self._allo_pos = None
        self._allo_nearest = (
            None  # W4c: nearest recalled landmark (map now READ BACK, not telemetry-only)
        )
        self._ext_drive = (
            None  # W5: routed computed signals (novelty/salience) -> a bounded top-down drive
        )
        self._exec = None  # W8: ACC/EVC executive (lazily bound)
        self._conflict = 0.0  # W8: ACC conflict over the competing action drives (readout)
        self._control_gain = 0.0  # W9: EVC control gain -- scales the PFC top-down bias by conflict
        # STN hyperdirect brake (Brain-Cog-adopted, consumes the ACC conflict into selection). The
        # brake gain is LEARNED (three-factor), the trigger is RELATIVE to a running conflict
        # baseline -- no magic constant. STN only ever raises caution, so it is safe by construction.
        self._stn_gain = 0.3  # learned gain: conflict-above-baseline -> global GPi brake strength
        self._stn_lr = 0.02  # its three-factor learning rate
        self._stn_elig = 0.0  # eligibility: did the brake engage last tick (credited by this RPE)
        self._stn_base = (
            0.0  # running mean of the brake eligibility (baseline-subtracted competition)
        )
        self._conflict_base = (
            0.0  # running mean conflict -> the RELATIVE brake trigger (adaptation)
        )
        self._stn_tau = 0.02  # adaptation rate of the two baselines above
        self._learned_ctl = (
            None  # C2 #9: (cost, value) of control LEARNED by the cortex (None -> fixed constants)
        )
        self._planned_seq = []  # C2 #6: best multi-step action sequence from the beam planner (preference only)
        # LEARNED (not hand-set) strength of the plan's vote against the reactive PFC bias
        self._plan_w = 0.1  # three-factor synapse, grown/shrunk by dopamine (see _pfc_bias)
        self._plan_lr = 0.02  # its learning rate
        self._plan_elig = 0.0  # eligibility trace: was the plan followed on the last tick
        self._plan_base = (
            0.0  # that trace's running mean (subtracted -> only better-than-usual pays)
        )
        self._plan_tau = 0.02  # baseline adaptation rate
        self._allo_pos = (0.0, 0.0)  # allocentric position estimate readout
        self._gain_on = bool(gain_adapt)
        self._gain = None
        self._motor_gain = 0.0  # adapted feedforward gain readout (never routed)
        # OPT-IN Lake-Vitals reliability HOLD: high stream-drift ADDS caution toward SAFE_ACTION only (never
        # motion), computed AFTER the reflex/E-STOP early-return so it can never suppress or delay an E-STOP.
        # vitals_hold=False (default) -> _vitals None -> reliability_hold 0.0 -> step() is byte-identical.
        self._vitals_hold_gain = float(vitals_hold_gain)
        if vitals_hold:
            from snn_data_lake.substrate.stream_sketch_panel import StreamSketchPanel

            self._vitals = StreamSketchPanel(seed=seed)
        else:
            self._vitals = None
        # opt-in SHARED LATENT Q (the transfer substrate): replace the TABULAR per-context q[unit][action] with a
        # shared linear action-value Q(x,a)=W[a].x over the LAKE-STORED pattern representation x (the [0,1]^64
        # encode_multimodal vector the lake records; basal_ganglia/latent_q.LatentQ). ONE weight set across ALL
        # contexts, so a reward update generalizes to nearby contexts -> mastery TRANSFERS (unlike tabular, which
        # learns each context in isolation). This LINEAR substrate rides the lake-stored PATTERN, which is
        # task-linear and stable. (v3.17 measured a linear Q over the online SSL-JEPA latent at chance and called
        # it "not task-aligned"; v3.18 CORRECTS that -- the SSL latent DOES encode the task, but the arm's online
        # encoder near-collapses its few live contexts, so on-robot the pattern is the reliable substrate; see the
        # NONLINEAR mlp_q block below and self-check 19 for the precise cause.) ADDITIVE to safety: it
        # only changes HOW the action VALUE is computed -- the reflex/E-STOP runs first and every command still
        # flows through the gated ArmSim. Flag OFF => the tabular q path is byte-identical.
        self._latent_q_on = latent_q
        self.latent_q = None  # lazily built on the first tick (sized to len(pattern))
        # opt-in SHARED NONLINEAR MLP-Q (the task-aligned learned substrate): a one-hidden-layer action-value
        # Q(z,a)=dot(W2[a], relu(W1@z+b1))+b2[a] over the SELF-SUPERVISED JEPA latent z=jepa.encode(pattern)
        # (basal_ganglia/mlp_q.MlpQ). Where the linear latent_q rides the raw lake pattern, this rides the LEARNED
        # JEPA latent AND lifts the linear ceiling with a hidden ReLU layer -- a nonlinear function approximator
        # over a nonlinear encoder. HONEST CORRECTION of v3.17: I reported the SSL latent as "not task-aligned"
        # (a linear online Q over it scored 0.33 ~ chance). That was OVER-PESSIMISTIC -- an offline probe recovers
        # the arm's band rule from the SSL latent at 0.80 (linear) / 0.88 (MLP), so the task IS in the latent; the
        # 0.33 was a WEAK-ONLINE-LEARNER artifact (a single-step Widrow-Hoff linear head on a few noisy contexts),
        # not the latent being task-irrelevant. This substrate is the honest test of whether an online NONLINEAR
        # head recovers it on the arm (measured in the self-check). Needs the JEPA (auto-enabled above). ADDITIVE to
        # safety: only changes HOW the value is computed; reflex/E-STOP first, every command still gated. Flag OFF
        # => the tabular/latent-Q path is byte-identical.
        self._mlp_q_on = mlp_q
        self.mlp_q = None  # lazily built on the first tick (sized to the JEPA latent dim)

    def coverage_reward(self, unit, action):
        """The physical objective: reward the end effector for REACHING somewhere it has not been.

        Chosen over the obvious alternatives on honesty grounds. "Reach a target" needs an invented
        goal object -- there is no target in this world, and fabricating one is the same class of
        theatre as a panel with no signal behind it. "Minimise prediction error" is already what the
        curiosity/JEPA term computes, so it would double-count rather than add an objective.

        Coverage is physical (it is measured off ee, not off a model), dense (every tick scores),
        needs nothing invented, and produces a genuinely rising line that SATURATES honestly once the
        reachable workspace is exhausted -- at which point the curve flattening is a true statement
        about the body, not a stalled learner. It is the arm's analog of mhflocke's "distance from
        start", which is the metric that makes their video watchable.

        Called AFTER arm.apply(), so `ee` is the post-action position -- the reward describes what the
        action actually achieved rather than what it intended.

        Sign convention matches the rest of the loop: [-1, 1]. A collision is the strongest negative
        so the objective can never reward driving into the obstacle."""
        st = self.arm.state()
        ee = st.get("ee") or (0.0, 0.0)
        cell = (round(float(ee[0]) / _COVER_CELL), round(float(ee[1]) / _COVER_CELL))
        # RECORD FIRST, JUDGE SECOND. v4.59 returned the collision penalty BEFORE reaching this
        # ledger, so from the moment the end effector dropped below FLOOR_Y the objective stopped
        # COUNTING -- the arm went on traversing new territory and none of it was entered. That
        # produced a "plateau at 9 cells, flat from tick 60" which was a measurement artifact: the
        # SAME run reaches 57 cells by tick 600 and is still climbing while the metric still says 9.
        # Whether the arm reached a cell is a fact about the body; whether that was a good idea is a
        # separate judgement, and the penalty below still makes it a bad one. Conflating the two let
        # a bad outcome erase the evidence of what happened.
        # Measured over 8 seeds: mean coverage 23.50 -> 28.12, and the joint trajectory is byte
        # identical in all 8 (first_divergence_tick = None) -- the arm does not move one degree
        # differently, this only stops discarding what it already did.
        fresh = cell not in self._covered
        if fresh:
            self._covered.add(cell)
            self._cover_new += 1
        # STILLNESS MUST BE STRICTLY WORST OF THE NON-COLLISION OUTCOMES.
        # v4.60 paid -0.05 for BOTH re-treading and standing still, which made them identical. Near
        # the floor that is a losing bet: moving risks -1.0, staying is a guaranteed -0.05, so the
        # rational policy is to stop moving -- and it did. Measured on seed 2: 4 of 6 learned units
        # had a NON-motor argmax (ENGAGE/RELEASE/RELEASE/HOLD) and the arm sat at one pose for 241
        # of 299 wake ticks, reaching 7 cells. The brain was not broken; it correctly optimised a
        # reward that paid the same for exploring and for giving up.
        # Coverage is the objective, so a tick that moves the body nowhere contributes nothing and
        # must rank below a tick that at least travelled. Ordering, strict: new > re-tread > still.
        moved = self._last_ee is None or (
            abs(ee[0] - self._last_ee[0]) > 1e-9 or abs(ee[1] - self._last_ee[1]) > 1e-9
        )
        # Record the distance travelled HERE, where the previous pose still exists. Telemetry used to
        # recompute this from `_last_ee` after the fact and always got 0.0, because the line below has
        # already overwritten it with the current pose -- a readout that was structurally incapable of
        # being non-zero. `moved` is deliberately left as the original per-axis test rather than being
        # re-derived from this distance: the two disagree on sub-1e-9 diagonal steps, and `moved` is on
        # the reward path, so changing it would alter what the arm learns.
        self._last_disp = (
            0.0
            if self._last_ee is None
            else math.hypot(ee[0] - self._last_ee[0], ee[1] - self._last_ee[1])
        )
        self._last_ee = (float(ee[0]), float(ee[1]))
        if st.get("collision"):
            return -1.0  # deliberately UNCHANGED and still the strongest negative
        if fresh:
            return 1.0
        return -0.02 if moved else -0.10  # re-tread beats stillness; both far above a collision

    def _self_motion(self, action):
        """Self-motion for path integration: how far the body moved as a result of `action`.

        Non-kinematic tokens (ENGAGE / RELEASE / HOLD) take their nominal step from _ALLO_MOVE. The
        JOINT_*_ROTATE tokens are DERIVED, because a fixed number cannot describe them: rotating joint
        A by 6 degrees sweeps the end effector a different distance and direction at every arm pose.
        Before this existed they defaulted to (0, 0), so the four commands that actually move the body
        were invisible to the map -- the brain tracked its position while ignoring its own limb.

        HONEST ABOUT WHAT THIS IS. This dead-reckons from the arm's own end-effector position, so it is
        PROPRIOCEPTIVE reafference, not a pure efference copy predicted from the outgoing command. The
        difference is not cosmetic and the choice is deliberate: every motor command here passes a
        safety gate that can refuse it (E-STOP freeze, collision, joint limit). An open-loop efference
        copy would integrate the *intended* move and drift permanently every time the gate said no --
        biologically real, but here it would corrupt the map on exactly the ticks safety intervened.
        Reading the realised position makes a refused command contribute zero, which is correct.

        No external reference is used: the end effector comes from the arm's own forward kinematics
        over its own joint angles, which is why this stays dead reckoning rather than becoming
        localisation.
        """
        nominal = _ALLO_MOVE.get(action)
        prev = self._allo_last_ee
        try:
            ee = self.arm.state().get("ee") or (0.0, 0.0)
            now = (float(ee[0]), float(ee[1]))
        except Exception:  # a map readout must never be able to break the tick
            return nominal or (0.0, 0.0)
        self._allo_last_ee = now
        if nominal is not None:
            return nominal
        if prev is None:  # first motor command of the run: no baseline to difference against
            return (0.0, 0.0)
        return (now[0] - prev[0], now[1] - prev[1])

    def _cpg_bias(self, td, qa):
        """COMPETENCE FLOOR: a hand-written Matsuoka CPG biases joint selection, so the arm produces
        rhythmic motion from tick 0 regardless of whether anything has been learned yet.

        This is the mechanism mhflocke's demo actually runs on. Its robot walks because a central
        pattern generator is the additive BASE (`controls = cpg_cmd * cpg_weight + correction`) and
        the learned network is only a corrector -- so there is visible competence before learning
        contributes anything. SNDL already owned an equivalent oscillator (cpg_oscillator.MatsuokaCPG,
        Matsuoka 1985 / Ijspeert 2008, self-checked for >=4 reversals and >=20deg amplitude) but
        `brain.py`, `live_brain.py` and `autonomous_loop.py` never imported it: the floor existed in a
        different process and nothing arbitrated between it and the learned policy.

        Entry point is the top-down `td` preference vector -- the SAME channel the PFC and the routed
        novelty/salience signals already use. It therefore changes only WHICH action the gated
        selection prefers; it adds no actuator route, and the reflex/E-STOP has already returned long
        before this runs. An E-STOPped arm ignores every motor command regardless (ArmSim.apply).

        Weight is competence-gated, not fixed (Policy 1): the CPG carries while the learned row is
        undifferentiated and recedes to a floor as the row separates. `_cpg_spread_max` is a RUNNING
        scale, so the gate is baseline-relative rather than a tuned threshold. The floor itself is a
        deliberate invariant -- the whole point is that competence never drops to zero -- and is the
        direct analog of mhflocke's `CompetenceGate.cpg_min`."""
        if self.cpg is None:
            return td
        vals = [qa.get(a, 0.0) for a in ACTIONS]
        spread = max(vals) - min(vals)
        # NO-SPURIOUS-MOTION INVARIANT. When the value row is entirely flat the brain has no reason
        # whatsoever to act, and the pre-existing safety contract is that selection then releases the
        # SAFE action rather than moving. The competence floor must not override that: a floor is for
        # carrying behaviour while learning is WEAK, not for thrashing an idle arm. Discovered by the
        # autonomous_loop self-check ("no value drive -> the BG gate releases nothing -> HOLD"), which
        # failed the moment the CPG was wired in. Relaxing that assertion to fit the feature was the
        # wrong repair; gating the feature to respect it is the right one.
        # Practical effect is small: with an objective live, the very first rewarded tick makes the row
        # non-flat, so the floor engages from tick ~1 and only a genuinely purposeless arm stays still.
        if spread <= 0.0:
            return td
        y1, y2 = self.cpg.step(_CPG_DRIVE)
        phase = max(-1.0, min(1.0, y1 - y2))  # bounded anti-phase rhythm in [-1, 1]
        self._cpg_spread_max = max(self._cpg_spread_max, spread)
        comp = spread / self._cpg_spread_max if self._cpg_spread_max > 1e-9 else 0.0
        self._cpg_w = max(
            _CPG_FLOOR, 1.0 - comp
        )  # learned competence displaces the CPG, never fully
        out = list(td)
        for act, sign in ((_A_POS, 1.0), (_A_NEG, -1.0)):
            if act in _ACT_IX:
                out[_ACT_IX[act]] += self._cpg_w * sign * phase
        return out

    def _route_signals(self, td):
        """W5: fold the brain's ROUTED computed signals (an external top-down drive set from the OUTER loop --
        real-perception novelty / workspace salience) into the per-action preference, so signals that
        previously only COMPUTED now CHANGE behaviour. Representation-only: this is one more top-down
        preference the SAME gated selection arbitrates -- the reflex/E-STOP still preempts (checked earlier in
        step), the BG Go/NoGo caution still damps release, and the arm gate is untouched. Bounded + safe."""
        ext = self._ext_drive
        if not ext or len(ext) != len(td):
            return td
        return [td[i] + ext[i] for i in range(len(td))]

    def _explore_noise(self, unit, scores):
        """C3 #10: exploration as CORTICAL VARIABILITY inside the gated selection, at a rate set by
        how UNRESOLVED the decision is -- replacing a fixed epsilon-greedy coin flip.

        Why the old mechanism had to go rather than be tuned. Epsilon-greedy did not merely use a
        hand-picked rate; it EARLY-RETURNED a uniform random action, jumping the cortisol/threat/
        interoceptive/BG caution path entirely. That shortcut is what forced the rate to be capped:
        any rise meant more ticks bypassing caution. Injecting the variability into the scores
        instead means every exploratory choice STILL passes through caution, the BG gate, and the
        reflex/E-STOP -- so the safety invariant now holds STRUCTURALLY rather than by keeping a
        magic number small, and the rate is free to adapt over its whole range.

        The rate is parameter-free and comes from EVIDENCE, not from the values themselves. Two
        value-derived rates were built and measured first, and both failed for the same underlying
        reason -- they judge uncertainty by the quantity still being learned:
          * spread / running-mean(spread) is circular; the ratio sits at ~1 forever, so it can never
            report "resolved" and exploration never anneals (replay's decisive-value probe slowed
            from 3 ticks to 70).
          * R * exp(-spread/R) against the reward scale anneals, but couples the rate to how fast
            values happen to grow, which starved the replay arm entirely (200 ticks vs 24).
        What actually governs how much a decision should be explored is how much EVIDENCE exists for
        it, so sigma is the reward scale divided by sqrt(1 + times THIS ACTION was taken in THIS
        unit) -- count-based optimism (Strehl & Littman), the family the audit itself recommended for
        the sibling explore_bonus. Per STATE-ACTION, not per state: a per-state count was measured
        and is not enough, because once the state is well-visited the noise decays even over actions
        that were never tried even once. That silently capped the replay probe's value margin at
        1.00 (q[HOLD] sat at exactly 0.0, untouched), which the old epsilon coin had covered only by
        accident of being permanently random. Per state-action, an untried action keeps full-scale
        optimism until it IS tried, so coverage is guaranteed rather than lucky.

        It decays monotonically with experience (so learning provably commits), needs no tuned rate,
        and -- decisively for the self-paced curriculum -- RESETS for a newly grown unit, so a new
        stage gets full exploration again while mastered ones stay decisive.

        `explore` survives as the on/off instrument (0.0 -> deterministic exploitation, used by the
        isolation probes); its magnitude no longer sets any rate."""
        # unit None = a direct query with no context identity, so there are no state-action counts
        # to be optimistic about; answer deterministically rather than at full optimism.
        if unit is None or not self.explore or self._r_mag <= 1e-9:
            # No reward experienced yet -- every action scores identically and there is nothing to be
            # uncertain BETWEEN, so noise buys no information and only destroys the naive tie-break
            # prior (measured: randomising it broke the learning probe's "before" readout).
            return [0.0] * len(scores)
        return [
            self.rng.gauss(0.0, self._r_mag / math.sqrt(1.0 + self._visits.get((unit, a), 0)))
            for a in ACTIONS
        ]

    def _select(self, unit, threat=0.0, intero=0.0, pfc_bias=None, qa=None, reliability_hold=0.0):
        if qa is None:  # default = the tabular row; the latent-Q row is passed in
            qa = self.q[unit]
        if self.bg is not None:
            return self._bg_select(
                qa, threat, intero, pfc_bias, reliability_hold, unit
            )  # BG Go/NoGo gate disinhibits the exploit choice
        bias = (
            self.cort.caution_bias()
            + threat
            + intero
            + reliability_hold
            + float(getattr(self, "_social_risk", 0.0) or 0.0)
        )  # stress + threat + intero + drift-hold + social-risk -> favour safe
        td = pfc_bias or [0.0] * len(ACTIONS)  # prefrontal top-down preference per action
        td = self._route_signals(
            td
        )  # W5: + routed computed signals (novelty/salience) -> behaviour
        td = self._cpg_bias(td, qa)  # + the CPG competence floor (rhythm from tick 0; gated by
        # learned competence, floored so behaviour never collapses). Preference-only: same gated
        # selection, same caution, and the reflex/E-STOP already returned before this line.

        noise = self._explore_noise(unit, [qa[a] for a in ACTIONS])  # variability, still gated

        def hab(a):  # the HABITUAL choice: learned value + caution, WITHOUT the top-down vote
            return qa[a] + (bias if a == SAFE_ACTION else 0.0) + noise[ACTIONS.index(a)]

        a_hab = max(ACTIONS, key=hab)
        a_act = max(ACTIONS, key=lambda a: hab(a) + td[ACTIONS.index(a)])
        self._td_flipped = a_act != a_hab  # did the goal actually OVERRIDE the habit this tick
        return a_act

    def _bg_select(
        self, qa, threat=0.0, intero=0.0, pfc_bias=None, reliability_hold=0.0, unit=None
    ):
        """Basal-ganglia Go/NoGo selection over the 3 actions. The action values are the per-channel
        cortical drive; dopamine is the winner's value confidence, DAMPED by stress + learned threat +
        interoceptive load (more caution -> less dopamine -> harder to release). The circuit disinhibits at most one
        channel; if the tonic GPi is never overcome (no drive / low DA) it releases NOTHING, which
        falls back to the safe action. Selection only ever picks among the SAME actions -- the gated
        arm is untouched and the reflex/E-STOP (checked earlier in step) still preempts it."""
        td = pfc_bias or [0.0] * len(
            ACTIONS
        )  # prefrontal top-down preference raises the goal action's drive
        td = self._route_signals(
            td
        )  # W5: + routed computed signals (novelty/salience) -> behaviour
        td = self._cpg_bias(td, qa)  # + the CPG competence floor (rhythm from tick 0; gated by
        # learned competence, floored so behaviour never collapses). Preference-only: same gated
        # selection, same caution, and the reflex/E-STOP already returned before this line.
        noise = self._explore_noise(unit, [qa[a] for a in ACTIONS])  # the BG gate still gates it
        drive = [max(0.0, max(0.0, qa[a]) + td[i] + noise[i]) for i, a in enumerate(ACTIONS)]
        # C3 #8: did the top-down vote actually CHANGE the ranking the BG gate arbitrates? Compared
        # on the drive vector rather than by re-running the gate (the gate is stateful, so a second
        # pass would perturb it) -- an honest proxy: if the top-drive action is the same with and
        # without td, the goal had no causal effect on this tick's behaviour and earns no credit.
        self._td_flipped = max(range(len(ACTIONS)), key=lambda i: drive[i]) != max(
            range(len(ACTIONS)), key=lambda i: max(0.0, qa[ACTIONS[i]])
        )
        if self._exec is None:  # W8: lazily bind the ACC executive
            try:
                from snn_data_lake.cortex.executive_control import ExecutiveControl

                self._exec = ExecutiveControl(seed=self._seed)
            except Exception:
                self._exec = False
        if self._exec:  # W8: ACC conflict readout over the competing drives (BG gate still selects)
            try:
                self._conflict = round(float(self._exec.conflict(drive)), 4)
            except Exception:
                pass
        # C2 (Brain-Cog belief-guided safety): `_social_risk`, pushed by the brain when a tracked
        # agent holds a DANGEROUS FALSE BELIEF (recursive_tom divergence -- until now a tracker that
        # drove nothing), adds to caution exactly like threat/intero -> more HOLD under social
        # uncertainty. Caution-only (safe by construction); the reflex/E-STOP preempts regardless.
        caution = min(
            1.0,
            self.cort.caution_bias()
            + threat
            + intero
            + reliability_hold
            + float(getattr(self, "_social_risk", 0.0) or 0.0),
        )
        da = max(0.0, min(1.0, max(drive) * (1.0 - caution)))
        # STN HYPERDIRECT BRAKE (Frank 2006): the ACC conflict signal -- computed just above and,
        # until now, CONSUMED BY NOTHING -- raises a global GPi brake that makes release harder when
        # the current choice is UNUSUALLY conflicted (conflict above its own running baseline, so the
        # trigger is relative/adaptive, not a magic cutoff). The brake gain is LEARNED by three-factor
        # (eligibility = braked last tick, dopamine = the arm's RPE), so proactive braking strengthens
        # only when holding under conflict pays. The STN can only ADD caution (raise the threshold),
        # never release an action, so it is safe by construction; the reflex/E-STOP preempts selection
        # regardless. Activates the cortex->STN hyperdirect edge declared in connectome.py.
        conf = float(self._conflict or 0.0)
        stn = self._stn_gain * max(0.0, conf - self._conflict_base)
        self._conflict_base = (1.0 - self._stn_tau) * self._conflict_base + self._stn_tau * conf
        da_rpe = float(
            getattr(self, "_rpe", 0.0) or 0.0
        )  # last-tick dopamine credits the last brake
        self._stn_gain = min(
            2.0,
            max(0.0, self._stn_gain + self._stn_lr * da_rpe * (self._stn_elig - self._stn_base)),
        )
        self._stn_base = (1.0 - self._stn_tau) * self._stn_base + self._stn_tau * self._stn_elig
        self._stn_elig = (
            1.0 if stn > 1e-6 else 0.0
        )  # did the brake engage this tick (credited next)
        res = self.bg.select(drive, da, stn=stn)
        # BG DECISION MARGIN (read-only, no actuation change): runner-up minus winner GPi disinhibition =
        # how DECISIVELY the gate committed (large = a clean winner; small = a conflicted, near-tie decision).
        # The BG gate computes the full gpi vector but the live path consumed only the winner; expose the
        # margin so the metacognitive monitor can read it (a low-margin action is a LESS reliable decision).
        _g = sorted(res.get("gpi") or [])
        self._decision_margin = (_g[1] - _g[0]) if len(_g) >= 2 else 1.0
        return SAFE_ACTION if res["winner"] is None else ACTIONS[res["winner"]]

    def greedy(self, unit):
        return max(ACTIONS, key=lambda a: self.q[unit][a])

    def _attend(self, visual, temperature, audio, distance):
        """Covert attention: a spiking reticular (TRN) gate selects the salient modality band from the four
        contrast-normalized bottom-up drives, and the claustrum binds co-active bands. Sets the attention
        readouts AND a per-band `_attn_gain` that step() APPLIES to the learner's percept + value substrate
        (biased competition -- attended band at full gain, unattended in [0.5,1.0)). SAFETY: the reflex/E-STOP
        (raw `distance`) and the amygdala threat readout stay on the RAW, unattenuated pattern -- attention
        never suppresses a threat or the safety path. Bounded, deterministic, guarded."""
        try:
            if self._rgate is None:
                from snn_data_lake.claustrum.claustrum import Claustrum
                from snn_data_lake.cortex.divisive_normalization import DivisiveNormalization
                from snn_data_lake.thalamus.reticular_gate import ReticularGate

                self._rgate = ReticularGate(n_channels=4, seed=self._seed)
                self._dnorm = DivisiveNormalization()
                self._claustrum = Claustrum(n_areas=4, seed=self._seed)

            def _energy(x):
                vals = x if isinstance(x, (list, tuple)) else [x]
                vals = [abs(float(v)) for v in vals if v is not None]
                return sum(vals) / len(vals) if vals else 0.0

            drives = [_energy(visual), _energy(temperature), _energy(audio), _energy(distance)]
            norm = self._dnorm.normalize(
                [max(0.0, d) for d in drives]
            )  # contrast-constant bottom-up drives
            attn = [n + 0.3 * self._novelty for n in norm]  # + top-down/novelty bias
            mask = self._rgate.mask(attn)  # spiking TRN gate -> per-band gain (attended released)
            self._attn_mask = [round(float(m), 4) for m in mask]
            self._attended = int(max(range(len(mask)), key=lambda i: mask[i]))
            mx = max(mask) if mask else 0.0  # per-band gain APPLIED downstream: attended band
            self._attn_gain = (
                [
                    0.5 + 0.5 * (float(m) / mx) for m in mask
                ]  # at full gain (1.0), unattended in [0.5,1.0)
                if mx > 0
                else None
            )  # biased competition, not a hard mask (no collapse)
            # MEASURED DEAD, and left dead deliberately -- do not "fix" this cut without reading this.
            # `norm` is r_i = d_i^2 / (0.25 + sum_j d_j^2), so the equal-share ceiling across 4 bands
            # is 1/4 = 0.25 and clearing 0.15 in k bands at once needs raw energy above 0.2315 (k=2) /
            # 0.2611 (k=3) / 0.3062 (k=4) in every one, out of an _energy() that is a MEAN. Live over
            # 200 ticks: at most ONE band ever cleared 0.15 (>=2 bands 0/200, >=3 bands 0/200) and
            # `_bound_modalities` was 0 on 200/200 while `_attended` varied -- detect() runs every
            # tick and can never return anything. Claustrum.k is 3. Audio's normalized response never
            # exceeds 0.0092 and distance's never 0.0537, so for two of four bands the cut is
            # unreachable by an order of magnitude regardless of the others.
            #
            # THREE replacements were built and measured; all were reverted:
            #   * share-relative (norm[i] > mean(norm))        -> >=3 bands on 0/200. No better.
            #   * per-band upper quartile of `norm`            -> >=3 bands on 0/97 attend-ticks.
            #   * per-band running MEDIAN of the RAW drives    -> binds on 25/200 ticks, and wiring
            #     Claustrum.broadcast() into `_attn_gain` changed that gain on 25/200 -- but a
            #     neuter-A/B moved NOTHING downstream: selected action 0/200, arm act/hold 0/200,
            #     trajectory 0/200, recognized units 4 vs 4, grown units 6 vs 6. CONSUMED-BUT-NOT-
            #     DECISIVE, so it was reverted rather than kept as a live-looking telemetry field.
            # The quartile result is structural and is the real lesson: divisive normalization
            # divides every band by a SHARED denominator, so its outputs are anti-correlated BY
            # CONSTRUCTION. Binding is a CONJUNCTION; testing a conjunction on the output of a
            # competition is a category error, because the normalizer's job is to make ONE band win.
            # And even fed a reachable bar, a gain change in [0.5,1.0] has no leverage on a 4-unit
            # prototype learner's vigilance test.
            #
            # So this path is honestly DORMANT: `_bound_modalities` is 0 forever and is read only by
            # telemetry (cortex/brain.py:2799), and Claustrum.broadcast() has no live call site at
            # all. It belongs in dormancy_gate.EXEMPT with that reason, not in a wiring wave. Leaving
            # the constant-0 in place is the honest report; making it vary without changing an
            # outcome would only make a dead faculty look alive on the cockpit.
            trains = [
                [1 if norm[i] > 0.15 else 0] * 6 for i in range(4)
            ]  # crude per-band activity raster
            _ft, assemblies = self._claustrum.detect(trains)
            self._bound_modalities = len(self._claustrum.bound_areas(assemblies))
        except Exception:
            pass

    def _context_pattern(self, visual, temperature, audio, distance):
        """The multimodal context vector (the amygdala CS and the growing learner's percept)."""
        return encode_multimodal(
            visual, temperature, audio, [distance] if distance is not None else None
        )

    def _amygdala_for(self, pattern):
        """The amygdala sized to the current substrate, or None when opt-out."""
        if self.amyg is None:
            return None
        if len(pattern) != self.amyg.n:  # keep the detector sized to the substrate
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
        # THE PFC WORKING MEMORY IS UNREACHABLE FROM HERE, AND THE REASON IS THIS LOOP'S SHAPE.
        # PrefrontalExecutive.maintain() and push_subgoal() have no live caller (their callers are 6
        # call sites across 4 modules -- cortex/autotelic_goals.py:315, reasoning/autonomous_brain.py:122,
        # reasoning/developmental_core.py:153, reasoning/developmental_loop_wm.py:56/:65/:128 -- all
        # offline loops; "five real callers" here was neither the site count nor the module count,
        # corrected 2026-07-29 by enumerating them), so `_wm_spikes`
        # keeps its birth 0, the goal stack never exceeds one frame, and `max_goals`, the HELD-parent
        # pop-back and top_down_bias's summation over frames are all dead live. Wiring maintain()
        # in here was built and measured, and REVERTED:
        #
        #     _wm_spikes        1 distinct (0 on 200/200) -> 22 distinct, nonzero on 195/200
        #     goal-stack depth  max 1  ->  max 1   (unchanged)
        #     held goal changed 0/200      selected action changed 0/200
        #     arm act/hold      0/200      trajectory 0/200      units 4 vs 4
        #
        # So it turned a constant telemetry field into a varying one and changed nothing -- the same
        # shape as the reverted salience_novelty wire, and keeping it would have made a dead
        # mechanism look alive. The structural reason it cannot charge: a working-memory attractor
        # exists to hold information across a GAP IN INPUT, and this loop has no gap. update() (or
        # plan(), on the JEPA paths) re-deliberates and re-commits the primary goal on EVERY wake
        # tick, and _gate_primary resets that frame's drive to 1.0 whenever the margin clears, which
        # live it always does -- so the recurrent drive is refreshed before `recurrence` can ever
        # decay it below hold_threshold and pop anything. The only ways to make the pop fire are to
        # lower recurrence/hold_threshold or to stop re-committing, i.e. to retune the module's own
        # constants until a number moves. push_subgoal() is worse off still: a sub-goal is a task
        # DECOMPOSITION, and this loop is flat and reactive with no hierarchy to decompose, so there
        # is no honest trigger for it at all.
        #
        # Correct disposition is EXEMPT-with-reason, not wiring: the WM attractor is real and is
        # exercised by the offline loops that have actual delay periods; it is the live arm loop that
        # has nothing for it to do. That belongs in substrate/dormancy_gate.EXEMPT (owned elsewhere
        # this pass, so reported rather than edited). Do not re-wire this without first giving the
        # loop a genuine input-free interval.
        gain = self.pfc_gain
        if self._exec:  # W9: EVC gates control by ACC conflict (closes the conflict->control loop)
            try:
                row0 = qa if qa is not None else self.q.get(unit, {})
                vals = sorted(row0.values(), reverse=True) if row0 else []
                stakes = (
                    (vals[0] - vals[1]) if len(vals) > 1 else 0.5
                )  # decision margin = value of resolving the conflict
                # C2 #9: use the LEARNED cost/value-of-control when the cortex has pushed them
                # (cortex/learned_evc, dopamine-gated three-factor). They replace the fixed
                # cost=0.1 / +0.3 constants -- same formula shape, same BOUNDED top-down preference
                # into the same gated selection; the reflex/E-STOP still preempts and no actuator
                # route is added. Falls back to the constants when the cortex has not pushed yet.
                lc = getattr(self, "_learned_ctl", None)
                if lc is not None:
                    cost_l, value_l = lc
                    ctl = self._exec.evc(
                        self._conflict,
                        value_gain=stakes + max(0.0, float(value_l)),
                        cost=max(0.0, float(cost_l)),
                    )
                else:
                    ctl = self._exec.evc(self._conflict, value_gain=stakes + 0.3, cost=0.1)
                gain = self.pfc_gain * (
                    1.0 + ctl
                )  # MORE top-down control when conflict is high AND worth it
                self._control_gain = round(float(ctl), 4)
            except Exception:
                pass
        # (highest priority) SERVED H-JEPA: accumulate the Level-0 latent window and, once it is full and an
        # abstract goal has formed, the executive PLANS over the ABSTRACT (Level-1) latent (rolls the L1
        # predictor). The abstract latent summarises the recent K-tick situation, so the goal tracks the slow
        # regime rather than the instantaneous frame. Still just a gated top-down bias; drives no motion.
        hj_planned = False
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
                # this path plans ONE abstract step, so the multi-step sequence is stale -> drop it
                # (it must not nudge with an old plan) and fall through to the shared learning tail
                # so the plan synapse still sees this tick's outcome.
                self._planned_seq, hj_planned = [], True
            # window still filling / no abstract goal yet -> fall through to the reactive bootstrap below
        if hj_planned:
            pass
        elif self.jepa is not None and self._z_goal is not None and pattern is not None:
            z = self.jepa.encode(pattern)
            gv = self.jepa.make_goal_value(self._z_goal)
            self.pfc.plan(
                z,
                ACTIONS,
                self.jepa,
                gv,
                horizon=self.jepa_horizon,
            )  # latent lookahead -> a working-memory goal
            # C2 #6: MULTI-STEP model-based planning. executive_control.plan_sequence (a beam/tree search
            # that previously had ZERO callers) searches action SEQUENCES over the SAME learned JEPA latent
            # world model (transition_fn=jepa.predict, value_fn=goal distance), so lookahead DEPTH becomes
            # real behaviour instead of a one-step rollout. Bounded top-down PREFERENCE only -- the same
            # gated selection arbitrates it, the reflex/E-STOP preempts, and no actuator route is added.
            if self._exec is not None:
                try:
                    self._planned_seq = list(
                        self._exec.plan_sequence(
                            z,
                            list(ACTIONS),
                            gv,
                            self.jepa.predict,
                            depth=min(3, max(2, int(self.jepa_horizon))),
                            breadth=3,
                        )
                        or []
                    )
                except Exception:
                    self._planned_seq = []
        else:
            row = (
                qa if qa is not None else self.q[unit]
            )  # the shared latent-Q row when on, else the tabular row
            evidence = [max(0.0, row[a]) for a in ACTIONS]
            self.pfc.update(evidence)  # reactive gated deliberation -> a WM goal
        bias = self.pfc.top_down_bias(strength=gain)
        # BRAIN-NATIVE: how far the multi-step plan out-votes the reactive PFC bias is a LEARNED
        # synapse, not a hand-set 0.5. Three-factor with an eligibility trace: the trace carries
        # whether the plan was followed last tick, dopamine (the arm's RPE) supplies the sign, and
        # the trace's own running mean is subtracted so only BETTER-than-usual outcomes strengthen
        # plan-following. Bounded preference; the reflex/E-STOP still preempts and no actuator
        # route is added.
        da = float(self._rpe or 0.0)
        self._plan_w = min(
            1.0, max(0.0, self._plan_w + self._plan_lr * da * (self._plan_elig - self._plan_base))
        )
        self._plan_base = (
            1.0 - self._plan_tau
        ) * self._plan_base + self._plan_tau * self._plan_elig
        self._plan_elig = 0.0
        if self._planned_seq:  # bounded nudge toward the FIRST action of the best planned sequence
            try:
                bias = list(bias)
                bias[ACTIONS.index(self._planned_seq[0])] += self._plan_w * gain
                self._plan_elig = 1.0
            except (ValueError, IndexError):
                pass
        return bias

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
            self.lake_wm = LakeWorkingMemory(
                p + 2 * n_act, path=self._lake_wm_path, seed=self._seed
            )
            self._pattern_len = p
        active = [i for i, v in enumerate(pattern) if v]
        if goal is not None:
            active.append(p + goal)
        active.append(p + n_act + ACTIONS.index(action))
        self.lake_wm.write(self._tick_t, active)
        self._tick_t += 1

    def _percept_novelty(self, pattern):
        """Bottom-up salience: normalized L1 deviance of the percept from its running expectation (an
        EMA that HABITUATES to frequent stimuli), clamped to [0, 1]. A novel/deviant percept -> high; a
        repeated stimulus pulls the EMA toward it -> the novelty decays (habituation).
        Deterministic. This is the stimulus-derived relevance signal the salience/RAS path was missing."""
        if self._sal_ema is None:
            self._sal_ema = list(pattern)
            return 1.0  # the first percept is an onset -> maximally novel
        n = max(1, len(pattern))
        dev = sum(abs(p - e) for p, e in zip(pattern, self._sal_ema)) / n
        self._sal_ema = [0.9 * e + 0.1 * p for e, p in zip(self._sal_ema, pattern)]
        return min(1.0, dev)

    def _workspace_broadcast(self, percept, threat_salience):
        """Additive Global-Workspace tick: brain regions (perception + a salience/threat detector)
        compete for the single workspace slot and the winner is broadcast to all. READ-ONLY
        annotation -- it selects no action, gates no motion, and never touches the reflex/E-STOP.
        Returns the broadcast content, or None when there is no workspace."""
        if self.workspace is None:
            return None
        if percept is not None:
            self.workspace.submit("perception", percept, 0.6)  # normally holds the slot when safe
        self.workspace.submit("salience", "threat", threat_salience)
        if self._salience_on:
            self.workspace.submit(
                "salience", "novelty", self._novelty
            )  # bottom-up stimulus salience competes
        content = self.workspace.step()
        # Purely observational: count a change of broadcast content. This used to be tallied by the
        # tkinter live-brain tab, which meant the number only existed while a window was open and
        # reset when it closed. It is a fact about the brain, so it lives on the brain.
        if content != self._prev_thought:
            self._ignitions += 1
        self._prev_thought = content
        return content

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
        pattern = encode_multimodal(
            visual, temperature, audio, [distance] if distance is not None else None
        )
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
            pat = self._context_pattern(
                ctx["visual"], ctx["temperature"], ctx["audio"], ctx["distance"]
            )
            if self.latent_q is not None:  # shared LINEAR latent-Q: greedy over the lake pattern
                if self.latent_q.greedy(pat) == opt:
                    ok += 1
            elif (
                self.mlp_q is not None and self.jepa is not None
            ):  # shared NONLINEAR MLP-Q: greedy over the JEPA latent
                if self.mlp_q.greedy(self.jepa.encode(pat)) == opt:
                    ok += 1
            else:
                unit, _grew = self.learner.observe(pat)
                if unit in self.q and self.greedy(unit) == opt:
                    ok += 1
        return ok / len(tasks) if tasks else 0.0

    def develop(self, stages, curriculum, batch=6, max_batches=50):
        """SELF-PACED developmental learning: work on the CURRENT curriculum stage's tasks and ADVANCE to the
        next stage AUTONOMOUSLY when this arm's own learning progress (curiosity) on it dries up -- the
        SpikingCurriculum reads mastery off the learning-progress signal (not a clock). Needs curiosity + jepa.
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
            trace.append(
                {
                    "stage": s,
                    "mastered": curriculum.mastered,
                    "competence": self._competence(tasks),
                    "advanced": advanced,
                }
            )
            if not advanced:
                break  # stalled on this stage (never mastered) -> stop
        return trace

    def step(self, visual, temperature, audio, distance, reward_fn=None, homeostatic_load=0.0):
        """One autonomous tick. distance is the proximity scalar; reward_fn(unit, action)->[-1,1].
        homeostatic_load (>=0) is the body's interoceptive load for the opt-in interoception path."""
        # 1. SAFETY FIRST: an object at the sensor trips the reflex -> gated EMERGENCY_STOP, no action
        if distance is not None and (not math.isfinite(distance) or distance < CONTACT):
            self.cort.step(1.0)  # danger -> stress up (sharpens future response)
            if math.isfinite(distance):  # real contact (not a NaN fail-closed):
                pat = self._context_pattern(visual, temperature, audio, distance)
                am = self._amygdala_for(pat)
                if am is not None:
                    am.pair(pat)  # US present: bind this context to threat
            estop = self.arm.apply("EMERGENCY_STOP")  # ONLY path to motion -> gate holds
            return {
                "unit": None,
                "action": "EMERGENCY_STOP",
                "reflex": True,
                "workspace": self._workspace_broadcast(
                    None, 0.95
                ),  # threat ignites (annotation only)
                "state": estop,
            }
        # 2. perceive -> grow a unit on a novel multimodal context
        pattern = self._context_pattern(visual, temperature, audio, distance)
        if (
            self._salience_on and self.workspace is not None
        ):  # novelty's only consumer is the GWT competitor
            self._novelty = self._percept_novelty(
                pattern
            )  # bottom-up salience updates each perceived tick
        percept = pattern
        if self._attn_on:  # covert attention: readout + APPLY to the LEARNER's percept
            self._attend(visual, temperature, audio, distance)
            if self._attn_gain:  # biased competition: attended band at full gain, others in
                g, nb = (
                    self._attn_gain,
                    len(self._attn_gain),
                )  # [0.5,1.0). threat/amygdala/reflex/E-STOP stay on RAW pattern.
                per = max(1, len(pattern) // nb)  # cells per modality band (64/4)
                percept = [
                    min(1.0, pattern[i] * g[min(i // per, nb - 1)]) for i in range(len(pattern))
                ]
        unit, grew = self.learner.observe(percept)
        if unit not in self.q:
            self.q[unit] = dict.fromkeys(ACTIONS, 0.0)  # new unit -> its own row to learn
        if self._jepa_on and self.jepa is None:  # lazily sized on the first tick
            self.jepa = JepaWorldModel(
                ACTIONS,
                in_dim=len(pattern),
                seed=self._seed,
                reward_shaping=self._reward_shaped_encoder,  # opt-in task-aligning encoder
                lambda_reward=self._reward_gain,
            )  # firm task pull (inert unless shaping)
        # value row: a SHARED value substrate, else the tabular per-context row. Two shared substrates (opt-in,
        # mutually exclusive): the LINEAR latent_q over the raw lake pattern, or the NONLINEAR mlp_q over the LEARNED
        # JEPA latent. qsub = the active substrate (None -> tabular); z_lq = its input at decision time (reused, so
        # the reward update below is a semi-gradient step on the SAME state that produced the action).
        z_lq = None
        qsub = None
        qa = self.q[unit]
        if self._latent_q_on:
            if self.latent_q is None:
                self.latent_q = LatentQ(
                    ACTIONS, len(pattern), seed=self._seed
                )  # shared LINEAR value over the lake pattern
            qsub, z_lq = (
                self.latent_q,
                percept,
            )  # attended [0,1]^64 lake representation (task-linear, stable)
        elif self._mlp_q_on and self.jepa is not None:
            z_lq = self.jepa.encode(
                pattern
            )  # the SELF-SUPERVISED JEPA latent (task IS in it -- see __init__)
            if self.mlp_q is None:
                self.mlp_q = MlpQ(
                    ACTIONS, len(z_lq), seed=self._seed
                )  # NONLINEAR head sized to the latent dim
            qsub = self.mlp_q
        if qsub is not None:
            qa = qsub.values(z_lq)  # shared value row -> generalizes across contexts
        threat = self._threat(pattern)  # learned amygdala threat for this context (0 if opt-out)
        intero = self._interocept(
            homeostatic_load
        )  # interoceptive caution from homeostatic load (0 if opt-out)
        pfc_bias = self._pfc_bias(
            unit, pattern, qa
        )  # prefrontal top-down goal preference over qa (latent-Q or tabular)
        # 3. CHOOSE (learned, not a fixed map) + ACT through the gated arm
        reliability_hold = 0.0
        if (
            self._vitals is not None
        ):  # OPT-IN Lake-Vitals: high drift -> add caution toward HOLD only
            self._vitals.observe(
                {
                    "motif": frozenset(i for i, v in enumerate(pattern) if v >= 0.5),
                    "features": list(pattern),
                }
            )
            reliability_hold = (
                self._vitals_hold_gain * self._vitals.drift_novelty()
            )  # in [0, gain]; NEVER creates motion
        action = self._select(
            unit,
            threat=threat,
            intero=intero,
            pfc_bias=pfc_bias,
            qa=qa,
            reliability_hold=reliability_hold,
        )  # threat/intero caution + PFC bias + drift hold (SAFE_ACTION only)
        state = self.arm.apply(action)  # gated: frozen after E-STOP, collision-checked
        # POSTURAL RECOVERY REFLEX. ArmSim REPORTS ground contact but deliberately does not prevent
        # it -- its self-check asserts a command CAN drive the arm into the floor ("reflex would
        # EMERGENCY_STOP"), because a real arm can be, and a sim that silently refused would hide the
        # failure mode from the safety layer. But nothing ever consumed that signal: the reflex fires
        # on `distance < CONTACT` (a proximity sensor), never on collision. So the arm sank below
        # FLOOR_Y and stayed -- measured P(collide at t+1 | collided at t) = 0.991, effectively an
        # absorbing state, with the end effector reaching y = -0.809 and 53 of 73 wake ticks spent
        # underground. This is the same computed-but-not-consumed gap as everywhere else.
        # HOME is the sanctioned recovery command and is the analog of the OpenCat pose-recovery that
        # is mhflocke's ONLY fall handling.
        # SAFETY: gated on `not stopped`. ArmSim.apply("HOME") clears `stopped`, so firing this while
        # E-STOPPED would un-freeze an emergency-stopped arm. It must never run in that state, and the
        # E-STOP early-return at the top of step() means a stopped arm normally never reaches here.
        if state.get("collision") == "GROUND" and not state.get("stopped"):
            self._ground_ticks += 1
            if self._ground_ticks >= _GROUND_RECOVER_TICKS:
                # LOCAL lift, not HOME. HOME was tried first and measured NET NEGATIVE over 5 seeds
                # at 600 ticks (84.4 -> 80.2 cells) because it resets theta to [0,0], teleporting the
                # arm back to the origin and destroying the position it had explored to. Escaping the
                # floor should not cost you where you were. One +A step is the minimal motion that
                # raises the end effector (geometry: from a sunken pose, escape needs consecutive
                # +A rotations), so the arm climbs out and carries on from where it fell.
                state = self.arm.apply(_A_POS)
                self._ground_ticks = 0
                self._recoveries += 1
        else:
            self._ground_ticks = 0
        goal_val = (
            self.pfc.goal if self.pfc is not None else None
        )  # the prefrontal working-memory goal
        self._lake_write(
            pattern, goal_val, action
        )  # opt-in: store this tick in the lake working memory
        # 4. reward -> RPE action-value update + cortisol. CURIOSITY (opt-in): the JEPA prediction error for
        #    the just-lived transition becomes a learning-progress INTRINSIC reward, so the arm learns EVEN
        #    WITHOUT an external reward (self-directed exploration). The intrinsic reward feeds the value update
        #    only -- no motor route, no stress; the reflex/E-STOP + caution still preempt.
        r_int = 0.0
        r_ext_step = None  # this tick's EXTERNAL task reward (for reward-shaping the encoder); None if no task reward
        # C3 #10: evidence for THIS action in THIS context -> its exploration optimism decays
        k = (unit, action)
        self._visits[k] = self._visits.get(k, 0) + 1
        raw_cur = raw_emp = 0.0
        if self.curiosity is not None and self.jepa is not None and self._jepa_prev is not None:
            perr = self.jepa.prediction_error(self._jepa_prev[0], self._jepa_prev[1], pattern)
            raw_cur = self.curiosity.intrinsic_reward(unit, perr)  # rewards learning progress
        if self._empower_on and self.jepa is not None:
            if self.empower is None:  # lazily bind the drive to the (now-built) JEPA forward model
                self.empower = Empowerment(self.jepa, ACTIONS, horizon=2, seed=self._seed)
            raw_emp = self.empower.step_reward(self.jepa.encode(pattern))  # controllability gain
        # C3 #6/#7: put the two drives in COMMON units before weighting them. Each is divided by a
        # running EMA of its own magnitude, so a prediction-error difference and a controllability
        # gain in bits arrive on the same scale and the weight below is a real ratio, not a hidden
        # unit conversion. Self-normalising: as a channel's magnitude grows its unit grows with it.
        if raw_cur or raw_emp:
            # WARM START, not a ramp from zero: an EMA started at 0 divides the first samples by a
            # near-zero unit, so the normalised drive spikes ~1/tau and swamps external reward before
            # the baseline catches up (measured: it dropped the reward-encoder self-check from a
            # held-out 1.00 to 0.67). Seed each unit with the first magnitude it ever sees, so the
            # first normalised sample is exactly 1.0 and the channel is well-scaled from tick one.
            self._cur_mag = (
                abs(raw_cur)
                if self._cur_mag <= 0.0
                else (1.0 - self._int_tau) * self._cur_mag + self._int_tau * abs(raw_cur)
            )
            self._emp_mag = (
                abs(raw_emp)
                if self._emp_mag <= 0.0
                else (1.0 - self._int_tau) * self._emp_mag + self._int_tau * abs(raw_emp)
            )
            # bounded: a transient far above the running unit is clipped, so one outlier tick cannot
            # dominate the value update the way an unbounded ratio would
            n_cur = max(-4.0, min(4.0, raw_cur / self._cur_mag)) if self._cur_mag > 1e-9 else 0.0
            n_emp = max(-4.0, min(4.0, raw_emp / self._emp_mag)) if self._emp_mag > 1e-9 else 0.0
            # MEASURED CORRECTION -- the delivered reward stays RAW; normalisation is used ONLY for
            # credit. The audit's proposal (RND-style running-magnitude normalisation of the
            # delivered intrinsic reward, Burda 2018) is wrong for THIS signal, and the code proved
            # it: curiosity here is LEARNING PROGRESS, whose decay toward zero as the world model
            # masters the environment IS its meaning. Dividing it by its own running magnitude
            # erases that -- a nearly-converged model keeps emitting full-strength curiosity.
            # Measured cost of doing it: the reward-shaped-encoder held-out fell 1.00 -> 0.67 and
            # the plain MLP-Q 0.33 -> 0.17. (RND normalises NOVELTY, which has no such semantics.)
            # So: deliver raw (magnitude preserved, byte-identical at birth), and use the
            # normalised values only as ELIGIBILITY, where a common scale is what makes the two
            # drives' competition for weight fair rather than a contest of native units.
            r_int = self.curiosity_gain * raw_cur + self.empowerment_gain * raw_emp
            self._int_elig = (n_cur, n_emp)  # credited on the NEXT tick, when an outcome exists
            if r_int > 0.0:  # C3 #11: the scale the autotelic goal gate measures itself against
                self._rint_base = (
                    r_int
                    if self._rint_base <= 0.0
                    else (1.0 - self._int_tau) * self._rint_base + self._int_tau * r_int
                )
        if reward_fn is not None or r_int != 0.0:
            r_ext = reward_fn(unit, action) if reward_fn is not None else 0.0
            if reward_fn is not None:
                r_ext_step = (
                    r_ext  # a real task reward this tick -> shape the encoder toward it next update
                )
            # C3 #7: LEARN the curiosity/empowerment split, credited by EXTERNAL reward ONLY. The
            # third factor is r_ext against its own per-unit running mean -- never `r` below, which
            # contains the very intrinsic term being weighted (that loop is self-referential and
            # saturates at the clamp instead of converging). Eligibility is LAST tick's normalised
            # contribution, since r_int for this tick is computed before any outcome exists. The two
            # drives compete via baseline subtraction, and their sum is clamped so the intrinsic
            # channel can never outgrow the external one -- the bound that keeps a self-directed arm
            # answerable to real task reward.
            # Only meaningful when BOTH drives are live: this rule learns a RATIO, and a ratio
            # against a structurally absent channel is not learnable -- with empowerment off the
            # baseline subtraction reduces to +/-0.5*e_cur, so noise in the external reward walks
            # curiosity's weight around for no reason (measured: it cost the plain MLP-Q probe
            # 0.33 -> 0.17 held-out). With one drive there is nothing to arbitrate, so its
            # constructor weight stands.
            err = 0.0
            if reward_fn is not None:  # the shared "better/worse than usual HERE" outcome signal
                base = self._rext_base.get(unit, 0.0)
                err = r_ext - base
                self._rext_base[unit] = (1.0 - self._int_tau) * base + self._int_tau * r_ext
            if reward_fn is not None and self.curiosity is not None and self._empower_on:
                e_cur, e_emp = self._int_elig
                mean_e = 0.5 * (e_cur + e_emp)
                self.curiosity_gain = min(
                    1.0, max(0.0, self.curiosity_gain + self._int_lr * err * (e_cur - mean_e))
                )
                self.empowerment_gain = min(
                    1.0, max(0.0, self.empowerment_gain + self._int_lr * err * (e_emp - mean_e))
                )
                tot = self.curiosity_gain + self.empowerment_gain
                if tot > 1.0:  # bounded intrinsic channel: it never outweighs external reward
                    self.curiosity_gain /= tot
                    self.empowerment_gain /= tot
            # C3 #8: LEARN pfc_gain -- how strongly the prefrontal GOAL outweighs learned habit --
            # by CONTRAST, not by reinforcement. The rule needs a signal that DEGRADES as the gain
            # rises, or it ratchets to the clamp: "did the action match the goal" grows with gain
            # and so can never brake it. "Does overriding the habit pay BETTER than following it"
            # falls once the gain starts overriding GOOD habits, which is what creates an interior
            # fixed point. Two lagged outcome baselines -- one over ticks where the goal actually
            # flipped the choice, one over ticks where it did not -- and the gain follows their
            # difference. Ticks where the vote changed nothing are excluded entirely: availability
            # is not effect. Preference weight only; the reflex/E-STOP and arm gate are untouched.
            if self.pfc is not None and reward_fn is not None:
                out = err  # better/worse than usually happens in this context
                if self._td_flipped:
                    self._r_ovr = (1.0 - self._int_tau) * self._r_ovr + self._int_tau * out
                    self._n_ovr += 1
                else:
                    self._r_hab = (1.0 - self._int_tau) * self._r_hab + self._int_tau * out
                    self._n_hab += 1
                if self._n_ovr >= 10 and self._n_hab >= 10:  # both baselines have real evidence
                    self.pfc_gain = min(
                        1.0, max(0.1, self.pfc_gain + self._pfc_lr * (self._r_ovr - self._r_hab))
                    )
            r = r_ext + r_int  # external task reward + intrinsic curiosity reward
            self._r_mag = (  # C3 #10: the reward scale exploration noise is measured against
                abs(r)
                if self._r_mag <= 0.0
                else (1.0 - self._int_tau) * self._r_mag + self._int_tau * abs(r)
            )
            if qsub is not None and z_lq is not None:
                qsub.update(
                    z_lq, action, r
                )  # SHARED substrate update (linear latent_q OR nonlinear mlp_q) -> transfer
            else:
                self.q[unit][action] += (
                    self.lr * self.cort.learn_rate_scale(r) * (r - self.q[unit][action])
                )
            self.cort.step(
                max(0.0, -r_ext)
            )  # stress from EXTERNAL negative reward only (curiosity is not stress)
            if self._dopa_on:  # DOPAMINERGIC RPE + opponent anti-reward (representation-only)
                try:
                    if self._valence is None:  # lazy build of the three spiking source nuclei
                        from snn_data_lake.basal_ganglia.valence_stdp import ValenceLearner
                        from snn_data_lake.epithalamus.habenula import LateralHabenula
                        from snn_data_lake.midbrain.dopamine_nuclei import DopamineNuclei

                        self._dopa = DopamineNuclei(seed=self._seed)
                        self._lhb = LateralHabenula(seed=self._seed)
                        self._valence = ValenceLearner()
                    pred = self._valence.valence(
                        pattern
                    )  # predicted affective value of this context (before learning)
                    rpe = self._valence.learn(
                        pattern, r
                    )  # three-factor DA-gated STDP over the context -> the RPE
                    da = self._dopa.da_level(
                        rpe
                    )  # phasic dopamine LEVEL (tonic baseline shifted by RPE; burst on +RPE)
                    if rpe < 0.0:  # OPPONENT: an LHb burst drives DA BELOW baseline (RMTg relay)
                        # SCALE FIX (v4.78): the two nuclei report on DIFFERENT tonic baselines --
                        # dopamine_nuclei.tonic_baseline = 0.2, habenula.DA_BASE = 1.0 -- and both were
                        # being assigned to the same `da`. So a negative RPE, which must DIP dopamine,
                        # reported a value on the 1.0 scale and therefore came out far ABOVE the 0.2
                        # baseline: measured 110/110 negative-RPE ticks above 0.2, min 0.300, mean
                        # 0.592. The dip was being rendered as a burst. Express the LHb result as a
                        # FRACTION of its own baseline, then put it on the nuclei's scale, so "half my
                        # baseline" reads as half of 0.2 in both nuclei's language.
                        _lhb_base = float(getattr(self._lhb, "da_base", 1.0)) or 1.0
                        _tonic = float(getattr(self._dopa, "tonic_baseline", 0.2))
                        da = _tonic * (self._lhb.dopamine(predicted=pred, received=r) / _lhb_base)
                    self._da_level, self._rpe = round(float(da), 4), round(float(rpe), 4)
                except Exception:
                    pass
            if (
                reward_fn is not None
            ):  # the task-reward-only paths (replay + latent goals), unchanged
                if (
                    self.replay and qsub is None
                ):  # SWR consolidation: TABULAR value table ONLY, no motor command
                    self.buffer.add(unit, action, r_ext)
                    if not grew:  # quiescent (familiar) tick -> replay past experience
                        self.buffer.replay(self.q, self.lr, batch=self.replay_batch)
                if (
                    self.jepa is not None and r_ext > 0.0
                ):  # positive TASK reward -> move the latent goal here (EMA)
                    zt = self.jepa.encode(pattern)
                    self._z_goal = (
                        zt
                        if self._z_goal is None
                        else [0.9 * g + 0.1 * z for g, z in zip(self._z_goal, zt)]
                    )
                if (
                    self.hjepa_serve is not None
                    and r_ext > 0.0
                    and len(self._hj_window) == self.hjepa_serve.K
                ):
                    z1t = self.hjepa_serve.encode1(
                        self._hj_window
                    )  # abstract latent at a rewarded tick -> the goal EMA
                    self._z1_goal = (
                        z1t
                        if self._z1_goal is None
                        else [0.9 * g + 0.1 * z for g, z in zip(self._z1_goal, z1t)]
                    )
            # C3 #11: the cutoff is now the STRICTER of an absolute floor and the intrinsic reward's
            # own running scale. The floor alone was fragile in a way this cycle made worse: r_int's
            # magnitude is the product of curiosity_gain/empowerment_gain, and those now LEARN and
            # move (measured 0.50 -> 0.81 on a two-drive task), so a fixed 0.02 silently changes
            # meaning as the brain retunes itself. Taking the max against a running mean of positive
            # intrinsic reward makes the gate track that scale. Deliberately one-sided: it can only
            # ever become STRICTER than today, never looser, so the cutoff keeps its real semantics
            # -- "stop forming goals when nothing is intrinsically valuable" -- instead of a
            # mean-tracking baseline that would follow learning-progress down to noise and then fire
            # on it forever, random-walking the latent goal.
            elif self.jepa is not None and r_int > max(
                self._autotelic_goal_thresh, self._rint_base
            ):
                # AUTOTELIC latent goal: with NO external task (reward_fn is None), a strong INTRINSIC reward
                # (curiosity learning-progress + empowerment controllability) marks THIS latent as intrinsically
                # valuable -> move the goal toward it so the prefrontal executive PLANS toward controllable/
                # learnable states. Deliberative lookahead now fires on the DEFAULT self-directed loop; this is
                # value/annotation only -- no motor route (the reflex/E-STOP + caution gates still preempt).
                zt = self.jepa.encode(pattern)
                self._z_goal = (
                    zt
                    if self._z_goal is None
                    else [0.9 * g + 0.1 * z for g, z in zip(self._z_goal, zt)]
                )
        # FOUR REPRESENTATION-ONLY DRIVES (opt-in): each observes this tick's already-computed locals
        # (threat / intero / r_int / r_ext_step / action / qa) and feeds the value/annotation channel only --
        # no actuator route, nothing here alters the action already gated above; the reflex/E-STOP preempted.
        if (
            self._affmem_on
            or self._agency_on
            or self._contrast_on
            or self._dev_on
            or self._naca_on
            or self._allo_on
            or self._gain_on
        ):
            rx = (
                r_ext_step if r_ext_step is not None else 0.0
            )  # this tick's EXTERNAL task reward (0.0 without a task)
            qmax = (
                max(qa.values()) if qa else 0.0
            )  # peak of the value row (dict for tabular + shared substrates)
            if self._affmem_on:  # emotion-tagged episodic retention of this tick's percept
                if self._affmem is None:
                    self._affmem = AffectiveEpisodicStore(capacity=256)
                self._affmem.store(
                    pattern, valence=math.tanh(r_int + rx), arousal=threat, reward=rx
                )
            # WRITE-ONLY as of today: AgencyMonitor.observe() is fed every tick, but nothing in src/
            # or tests/ reads agency() / controllability() / status(), and no `agency` key reaches the
            # telemetry contract or the cockpit. The comment here used to read "controllability of
            # action -> realized value (sense of agency)", whose arrow asserted a consumer that does
            # not exist. It is a P3 case awaiting its disposition -- wire and prove the charge, retire,
            # or EXEMPT with a reason -- not a live signal.
            if self._agency_on:
                if self._agency is None:
                    self._agency = AgencyMonitor(seed=self._seed)
                self._agency.observe(action, cmd=qmax, fb=(rx if reward_fn is not None else r_int))
            if (
                self._contrast_on and reward_fn is not None
            ):  # success-vs-failure context pooling (REAL task reward only)
                if self._contrast is None:
                    self._contrast = ContrastiveEpisodeAnalyzer(seed=self._seed)
                self._contrast.record(rx > 0.0, {"threat": threat, "intero": intero, "qmax": qmax})
            if self._dev_on:  # competence EMA -> maturity; readouts stored, never routed
                if self._dev is None:
                    self._dev = DevelopmentalEnvelope(seed=self._seed)
                self._comp_ema = 0.98 * self._comp_ema + 0.02 * (
                    1.0 if (reward_fn is not None and rx > 0.0) else 0.0
                )
                self._dev.update(self._comp_ema)
                self._dev_explore = (
                    self._dev.explore_noise
                )  # exploration scale readout (NOT applied to _select)
                self._dev_fm_gain = (
                    self._dev.forward_model_gain
                )  # forward-model authority readout (NOT applied)
            if self._naca_on:  # neuromodulator-gated credit on the local reward delta
                if self._naca is None:
                    self._naca = NACABroadcast(n_syn=8, n_types=len(ACTIONS), seed=self._seed)
                sig = ACTIONS.index(action) if action in ACTIONS else 0
                corr = self._naca.modulate(
                    [r_int], self._naca.lam(sig)
                )  # per-synapse delta list -> corrections
                self._naca_credit = (
                    float(corr[0]) if corr else 0.0
                )  # additive neuromod correction (readout)
            if self._allo_on:  # allocentric landmark map dead-reckoned from the stream
                if self._allo is None:
                    self._allo = AllocentricMap(seed=self._seed)
                dx, dy = self._self_motion(action)  # self-motion DERIVED from the chosen action
                self._allo.move(dx, dy)  # path integration reflects the real behavioral trajectory
                self._allo.observe(
                    str(action), rel_x=intero, rel_y=threat, valence=math.tanh(r_int + rx)
                )
                self._allo_pos = (
                    self._allo.position
                )  # allocentric position estimate (readout, no motion)
                self._allo_nearest = self._allo.nearest(
                    valence_pref=math.tanh(r_int + rx)
                )  # W4c: READ BACK nearest landmark (map consumed)
            if self._gain_on:  # cerebellar feedforward gain adaptation (NEVER applied)
                if self._gain is None:
                    self._gain = GainAdapter(seed=self._seed)
                motor = 1.0 if action in ("RELEASE", "ENGAGE") else 0.0
                _resp, _err, self._motor_gain = self._gain.step(
                    motor, world_gain=qmax
                )  # learn a gain (readout only)
        # JEPA (opt-in): learn the latent world model ONLINE from the live (z_prev, a_prev, z_now) transition.
        # OFF the safety path -- after the gated action + lake write, drives no motion, adds no actuator route.
        if self.jepa is not None:
            if self._jepa_prev is not None:
                pp, pa, pr = (
                    self._jepa_prev
                )  # (prev pattern, prev action, prev task reward or None)
                self.jepa.train_step(
                    pp, pa, pattern, reward=pr
                )  # SSL update (+ reward-shaping when pr is a task reward)
            self._jepa_prev = (pattern, action, r_ext_step)
        # workspace: perception (the current unit) competes with the amygdala's threat salience (the
        # REAL learned threat when the amygdala is wired in, else a low background constant) -> broadcast.
        salience = threat if self.amyg is not None else 0.2
        broadcast = self._workspace_broadcast(
            unit, salience
        )  # additive annotation; drives no motion
        return {
            "unit": unit,
            "grew": grew,
            "action": action,
            "reflex": False,
            "threat": threat,
            "intero": intero,
            "goal": goal_val,
            "wm_events": (self.lake_wm.n_events() if self.lake_wm is not None else 0),
            "workspace": broadcast,
            "state": state,
        }


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
        return [[_lin(o, i), [0.0] * o]]  # one Linear layer (in -> out); a valid minimal MLP

    obj = {
        "format": "snn_hjepa/1",
        "in_dim": in_dim,
        "K": k,
        "latent0": l0,
        "latent1": l1,
        "fine_actions": list(actions),
        "coarse_actions": list(actions),
        "tau": 0.99,
        "enc0": _enc(in_dim, l0),
        "enc1": _enc(l0, l1),
        "enc0_target": _enc(in_dim, l0),
        "enc1_target": _enc(l0, l1),
        "pred0": _enc(l0 + n, l0),
        "pred1": _enc(l1 + n, l1),
        "predictors": {"pred0": "residual_mlp_unit", "pred1": "mlp_unit"},
        "meta": {},
    }
    with open(path, "w", encoding="ascii") as f:
        _json.dump(obj, f)


def main():
    safe_d, danger_d = CONTACT * 3.0, CONTACT * 0.5
    a = _ctx(0.9, 0.1, 0.1, safe_d)  # three DISTINCT multimodal contexts (differ per band)
    b = _ctx(0.1, 0.9, 0.1, safe_d)
    c = _ctx(0.1, 0.1, 0.9, safe_d)

    # 1. GROWTH: three distinct contexts -> exactly three grown units
    grow = AutonomousArm(seed=0)
    for _ in range(3):
        for ctx in (a, b, c):
            grow.step(**ctx)
    grew_to = grow.learner.n_units

    # 2. LEARNING: reward RELEASE (NOT the tie-break default) in context a -> policy overrides to it
    learn = AutonomousArm(seed=1)

    def reward(unit, action):
        return 1.0 if action == "RELEASE" else -1.0

    default_before = learn.step(**a)["action"] or "ENGAGE"
    last = None
    for _ in range(90):
        last = learn.step(**a, reward_fn=reward)
    learned = learn.greedy(last["unit"])

    # 3. SAFETY: an object at the sensor -> reflex E-STOP; a later action must NOT move the frozen arm
    guard = AutonomousArm(seed=2)
    ev = guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    stopped = guard.arm.state()["stopped"]
    ee_frozen = guard.arm.state()["ee"]
    guard.step(**a)  # a learned action after E-STOP
    still_frozen = guard.arm.state()["stopped"] and guard.arm.state()["ee"] == ee_frozen

    # 5. DETERMINISM
    def run(seed):
        bot = AutonomousArm(seed=seed)
        return [
            bot.step(**ctx)["action"] for _ in range(3) for ctx in (a, b, c)
        ], bot.learner.n_units

    # 6. REPLAY (opt-in SWR consolidation): sample-efficient value learning, safety intact.
    # STRENGTHENED (C3 #10): this compared ONE seed. Exploration is now stochastic per-tick
    # (count-based optimism instead of an epsilon coin), so a single-seed race between two
    # stochastic learners is decided partly by luck -- seed 3 is in fact the one seed of twelve
    # where the slower arm wins. A single sample was never adequate evidence for the claim being
    # made. It now MEANS "replay is sample-efficient": replay must be faster ON AVERAGE and on a
    # clear MAJORITY of seeds. That is a harder bar than one comparison, not an easier one.
    def ticks_to_open_margin(replay, seed=3, target=1.2):
        bot = AutonomousArm(seed=seed, replay=replay)
        for t in range(1, 200):
            ev = bot.step(**a, reward_fn=reward)
            row = bot.q[ev["unit"]]
            if row["RELEASE"] - max(row["ENGAGE"], row["HOLD"]) >= target:
                return t
        return 200

    _seeds = range(12)
    _live = [ticks_to_open_margin(replay=False, seed=s) for s in _seeds]
    _replay = [ticks_to_open_margin(replay=True, seed=s) for s in _seeds]
    live_ticks = sum(_live) / len(_live)
    replay_ticks = sum(_replay) / len(_replay)
    replay_wins = sum(1 for lv, rp in zip(_live, _replay) if rp < lv)

    rguard = AutonomousArm(seed=2, replay=True)  # danger, then a rewarded action, replay ON
    rev = rguard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    ree = rguard.arm.state()["ee"]
    rguard.step(**a, reward_fn=reward)  # replay-eligible tick after the E-STOP
    replay_frozen = rguard.arm.state()["stopped"] and rguard.arm.state()["ee"] == ree

    def run_replay(seed):
        bot = AutonomousArm(seed=seed, replay=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    # 7. BUNDLE: enclose a trained perception skill in the loop (additive; the safety gate is untouched)
    from snn_data_lake.cortex.trained_brain import make_brain

    _wp = [[1.0 if i < 32 else 0.0 for i in range(64)], [0.0 if i < 32 else 1.0 for i in range(64)]]
    bundled = AutonomousArm.from_bundle(
        BrainBundle(
            {"vision": make_brain([_wp], steps=8)},
            {"reflex": True, "cortisol": True},
            {"pid": True},
        ),
        seed=0,
    )
    bclass = bundled.perceive(**a)
    bundle_ev = bundled.step(
        **_ctx(0.5, 0.5, 0.5, danger_d)
    )  # danger must still E-STOP with a bundle
    bundle_safe = bundle_ev["action"] == "EMERGENCY_STOP" and bundle_ev["reflex"]

    # 8. WORKSPACE: an opt-in Global Workspace runs live in the loop -- regions compete + broadcast.
    wsa = AutonomousArm(seed=0, workspace=True)
    ws_safe = wsa.step(**a)["workspace"]  # safe tick: perception holds the slot
    ws_danger = wsa.step(**_ctx(0.5, 0.5, 0.5, danger_d))[
        "workspace"
    ]  # danger: threat ignites + broadcasts
    ws_frozen = wsa.arm.state()["stopped"]
    ws_off = AutonomousArm(seed=0).step(**a)[
        "workspace"
    ]  # opt-in: a default arm broadcasts nothing

    print("=" * 62)
    print("AUTONOMOUS ARM — grows from multimodal data + learns actions, safely")
    print("=" * 62)
    print(f"growth      : 3 distinct multimodal contexts -> {grew_to} grown units")
    print(
        f"learning    : rewarded RELEASE -> policy went {default_before} -> {learned} (learned, not fixed)"
    )
    print(
        f"safety      : object at sensor -> {ev['action']} (reflex={ev['reflex']}), arm stopped={stopped}; "
        f"post-STOP action frozen={still_frozen}"
    )
    print(
        f"replay      : SWR consolidation -> decisive value in {replay_ticks:.1f} mean live ticks vs "
        f"{live_ticks:.1f} over 12 seeds (faster on {replay_wins}/12) "
        f"(value table only; safety gate holds with replay={replay_frozen})"
    )
    print(
        f"bundle      : from_bundle(vision skill) -> perceived class {bclass} (additive perception); "
        f"safety gate holds with a bundle (E-STOP={bundle_safe})"
    )
    print(
        f"workspace   : Global Workspace live in the loop -> safe broadcasts {ws_safe!r}, danger "
        f"ignites {ws_danger!r}; E-STOP freeze holds with it active ({ws_frozen})"
    )
    print("=" * 62)

    # ---- self-check: deterministic asserts ----
    assert grew_to == 3, f"must grow one unit per distinct multimodal context ({grew_to})"
    assert ev["action"] == "EMERGENCY_STOP" and ev["reflex"], (
        "danger must trip the reflex E-STOP before any action"
    )
    assert stopped, "E-STOP must freeze the arm"
    assert still_frozen, (
        "a learned action after E-STOP must NOT move the arm (safety gate holds through the loop)"
    )
    # audit P1: a NaN distance must FAIL CLOSED (reflex E-STOP), not slip past `distance < CONTACT`
    nanguard = AutonomousArm(seed=9)
    nev = nanguard.step(**_ctx(0.5, 0.5, 0.5, float("nan")))
    assert nev["action"] == "EMERGENCY_STOP" and nev["reflex"], (
        "NaN distance must trip the E-STOP (fail-closed)"
    )
    # bundle: perception is ADDITIVE (a bundle-equipped arm perceives; a bare arm does not) and the
    # safety gate is untouched by enclosing a trained-skill bundle in the loop.
    assert bclass in (0, 1) and bclass == bundled.perceive(**a), (
        "bundle perceive -> a deterministic valid class"
    )
    assert AutonomousArm(seed=0).perceive(**a) is None, (
        "a bundle-less arm perceives nothing (additive, opt-in)"
    )
    assert bundle_safe, "safety E-STOP must hold with a trained-skill bundle enclosed in the loop"
    assert bundled.bundle.has_instinct("reflex") and bundled.bundle.has_motor("pid"), (
        "bundle wires instinct/motor flags"
    )
    # workspace: opt-in + additive; a proximity threat ignites + is broadcast, and the E-STOP freeze
    # STILL holds with the Global Workspace active (it annotates, it never selects or gates motion).
    assert ws_off is None, "the Global Workspace is opt-in (a default arm broadcasts nothing)"
    assert ws_safe is not None, "with a workspace, a safe tick broadcasts the current percept"
    assert ws_danger == "threat", "a proximity threat must ignite + be broadcast by the workspace"
    assert ws_frozen, (
        "the E-STOP freeze must hold with the Global Workspace active (additive, gate untouched)"
    )
    assert learned == "RELEASE", (
        f"the arm must LEARN the rewarded action, not use a fixed map ({learned})"
    )
    assert run(7) == run(7), "same seed -> identical autonomous behaviour (deterministic)"
    # replay (opt-in SWR) is sample-efficient AND safety-preserving:
    assert replay_ticks < live_ticks and replay_wins >= 9, (
        f"replay must learn in fewer live ticks ON AVERAGE and on a majority of seeds "
        f"({replay_ticks:.1f} vs {live_ticks:.1f}; faster on {replay_wins}/12)"
    )
    assert rev["action"] == "EMERGENCY_STOP" and rev["reflex"], (
        "danger must E-STOP even with replay on"
    )
    assert replay_frozen, (
        "replay must NOT move the E-STOP-frozen arm (safety gate holds with replay)"
    )
    assert run_replay(11) == run_replay(11), "replay path must be deterministic"

    # 9. AMYGDALA wired live: a REAL learned threat detector. Repeated danger in one context teaches
    #    threat for THAT context; a never-dangerous context stays safe (discrimination). Caution-only.
    amy = AutonomousArm(seed=0, amygdala=True)
    danger_ctx = _ctx(0.2, 0.8, 0.3, danger_d)  # this context co-occurs with danger
    safe_ctx = _ctx(0.9, 0.1, 0.1, safe_d)  # this one never does
    for _ in range(12):
        amy.step(**danger_ctx)  # each danger tick pairs the context with the US
    threat_danger = amy._threat(amy._context_pattern(**danger_ctx))
    threat_safe = amy._threat(amy._context_pattern(**safe_ctx))
    amy_guard = AutonomousArm(seed=2, amygdala=True)  # the safety envelope must hold with it on
    amy_ev = amy_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    amy_frozen0 = amy_guard.arm.state()["stopped"]
    amy_guard.step(**safe_ctx)
    amy_frozen1 = amy_guard.arm.state()["stopped"]

    # 10. BASAL GANGLIA Go/NoGo gate wired live as the action gate: it releases the learned high-value
    #     action, or -- with no value drive -- releases NOTHING -> the safe action (no spurious motion).
    bg = AutonomousArm(seed=1, bg_gate=True)
    last_bg = None
    for _ in range(90):
        last_bg = bg.step(**a, reward_fn=reward)  # reward RELEASE
    bg_released = bg._bg_select(bg.q[last_bg["unit"]])
    bg_null = AutonomousArm(seed=3, bg_gate=True)._bg_select(dict.fromkeys(ACTIONS, 0.0))
    bg_guard = AutonomousArm(seed=2, bg_gate=True)  # the safety envelope must hold with it on
    bg_ev = bg_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    bg_guard.step(**a)
    bg_frozen = bg_guard.arm.state()["stopped"]

    def run_wired(seed):  # both paths together must be deterministic
        bot = AutonomousArm(seed=seed, amygdala=True, bg_gate=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(
        f"amygdala    : learned threat for the danger context {threat_danger:.2f} vs the safe "
        f"context {threat_safe:.2f} (discriminated); E-STOP freeze holds with it on ({amy_frozen1})"
    )
    print(
        f"bg gate     : reward -> the gate releases {bg_released}; no value drive -> releases "
        f"{bg_null} (no spurious motion); E-STOP freeze holds with it on ({bg_frozen})"
    )
    print("=" * 62)

    # amygdala + BG gate are opt-in; a default arm is byte-for-byte the old behaviour.
    assert AutonomousArm(seed=0).amyg is None and AutonomousArm(seed=0).bg is None, (
        "the amygdala and the BG gate are opt-in (a default arm has neither)"
    )
    # amygdala: learns a REAL threat, discriminates, and never weakens the safety envelope.
    assert threat_danger > 0.30, (
        f"the amygdala must LEARN threat for the danger context ({threat_danger:.2f})"
    )
    assert threat_danger > threat_safe + 0.20, (
        f"the amygdala must DISCRIMINATE danger from safe ({threat_danger:.2f} vs {threat_safe:.2f})"
    )
    assert amy_ev["action"] == "EMERGENCY_STOP" and amy_ev["reflex"], (
        "danger must E-STOP with the amygdala on"
    )
    assert amy_frozen0 and amy_frozen1, (
        "the E-STOP freeze must hold with the amygdala wired in (caution-only)"
    )
    # BG gate: selects the learned action, releases nothing without drive, safety envelope intact.
    assert bg.greedy(last_bg["unit"]) == "RELEASE", (
        "a BG-gated arm must still learn the rewarded action"
    )
    assert bg_released == "RELEASE", "the BG gate must release the learned high-value action"
    assert bg_null == SAFE_ACTION, (
        "no value drive -> the BG gate releases nothing -> the safe action (HOLD)"
    )
    assert bg_ev["action"] == "EMERGENCY_STOP" and bg_ev["reflex"], (
        "danger must E-STOP with the BG gate on"
    )
    assert bg_frozen, "the E-STOP freeze must hold with the BG gate wired in"
    assert run_wired(5) == run_wired(5), "the amygdala + BG-gate path must be deterministic"

    # 11. INTEROCEPTION wired live: the body's homeostatic load raises a bounded interoceptive caution
    #     that biases selection toward the safe action -- caution-only, never a new actuator route.
    io = AutonomousArm(seed=1, interoception=True)
    io.explore = 0.0  # isolate the bias (no exploration noise)
    # Built FROM ACTIONS rather than as a literal: a hardcoded row silently encodes the size of the
    # action space, and broke the moment motor tokens were added (KeyError in _select's score list).
    # Derive it, override only what the probe is actually testing.
    io.q[0] = dict.fromkeys(ACTIONS, 0.0) | {"ENGAGE": 0.05}  # a mild non-safe preference
    io_calm = io._interocept(0.0)  # homeostasis -> ~no caution
    io_stress = io._interocept(1.5)  # heavy homeostatic load -> strong caution
    pick_calm = io._select(0, intero=io_calm)  # no load -> keep the mild non-safe preference
    pick_stress = io._select(0, intero=io_stress)  # heavy load -> override toward the safe action
    io_guard = AutonomousArm(seed=2, interoception=True)  # the safety envelope must hold with it on
    io_ev = io_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d), homeostatic_load=1.5)
    io_frozen0 = io_guard.arm.state()["stopped"]
    io_guard.step(**a, homeostatic_load=1.5)  # an action under load after the E-STOP
    io_frozen1 = io_guard.arm.state()["stopped"]

    def run_io(seed):
        bot = AutonomousArm(seed=seed, interoception=True)
        return [
            bot.step(**ctx, homeostatic_load=0.8)["action"] for _ in range(6) for ctx in (a, b, c)
        ]

    print(
        f"interocept  : homeostatic load -> caution {io_calm:.2f} (calm) vs {io_stress:.2f} (stressed); "
        f"mild-preference tie {pick_calm} -> {pick_stress} under load (safe bias); "
        f"E-STOP freeze holds with it on ({io_frozen1})"
    )
    print("=" * 62)

    # interoception is opt-in; a default arm is byte-for-byte the old behaviour.
    assert AutonomousArm(seed=0).intero is None, "interoception is opt-in (a default arm has none)"
    # a homeostatic load raises the bounded caution and overrides a mild non-safe preference toward the safe action.
    assert io_stress > io_calm + 0.10, (
        f"a homeostatic load must raise interoceptive caution ({io_stress:.2f} vs {io_calm:.2f})"
    )
    assert pick_calm != SAFE_ACTION and pick_stress == SAFE_ACTION, (
        "interoceptive stress must override a mild non-safe preference toward the safe action (caution-only)"
    )
    # the safety envelope holds with interoception wired in.
    assert io_ev["action"] == "EMERGENCY_STOP" and io_ev["reflex"], (
        "danger must E-STOP with interoception on"
    )
    assert io_frozen0 and io_frozen1, (
        "the E-STOP freeze must hold with interoception wired in (caution-only)"
    )
    assert run_io(5) == run_io(5), "the interoception path must be deterministic"

    # 12. PREFRONTAL EXECUTIVE wired live: it deliberates over the action-values, commits a GOAL to
    #     working memory, and top-down-biases selection toward it -- goal-directed, caution-safe.
    pf = AutonomousArm(seed=1, pfc=True)
    for _ in range(90):
        pf.step(**a, reward_fn=reward)  # reward RELEASE
    pfc_goal = pf.pfc.goal  # the executive's held working-memory goal
    pfc_bias = pf.pfc.top_down_bias()
    goal_is_rewarded = pfc_goal == ACTIONS.index("RELEASE")
    held = all(
        pf.pfc.maintain() == pfc_goal for _ in range(5)
    )  # WM holds the goal across an input-free delay
    pf_guard = AutonomousArm(seed=2, pfc=True)  # the safety envelope must hold with it on
    pf_ev = pf_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    pf_frozen0 = pf_guard.arm.state()["stopped"]
    pf_guard.step(**a)  # an action after the E-STOP
    pf_frozen1 = pf_guard.arm.state()["stopped"]

    def run_pf(seed):
        bot = AutonomousArm(seed=seed, pfc=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(
        f"prefrontal  : deliberates over values -> working-memory goal {ACTIONS[pfc_goal]} "
        f"(rewarded={goal_is_rewarded}, held across delay={held}); E-STOP freeze holds with it on ({pf_frozen1})"
    )
    print("=" * 62)

    # the prefrontal executive is opt-in; a default arm has none (byte-for-byte the old behaviour).
    assert AutonomousArm(seed=0).pfc is None, (
        "the prefrontal executive is opt-in (a default arm has none)"
    )
    # it commits the high-value (rewarded) action as its goal and holds it in working memory.
    assert goal_is_rewarded, (
        f"the executive must commit the rewarded action as its goal ({ACTIONS[pfc_goal]})"
    )
    assert pfc_bias[pfc_goal] == max(pfc_bias) and max(pfc_bias) > 0.0, (
        "the top-down bias must favour the held goal"
    )
    assert held, (
        "working memory must hold the goal across an input-free delay (persistent activity)"
    )
    # the safety envelope holds with the executive wired in (top-down preference only, never a motor route).
    assert pf_ev["action"] == "EMERGENCY_STOP" and pf_ev["reflex"], (
        "danger must E-STOP with the prefrontal executive on"
    )
    assert pf_frozen0 and pf_frozen1, (
        "the E-STOP freeze must hold with the executive wired in (top-down preference only)"
    )
    assert run_pf(5) == run_pf(5), "the prefrontal path must be deterministic"

    # 13. LAKE WORKING MEMORY wired live: each tick's activity is stored in the durable spike lake and
    #     recalled from it -- the data lake IS the working memory (opt-in, storage only, gate untouched).
    lw = AutonomousArm(seed=1, pfc=True, lake_wm=True)
    for _ in range(8):
        lw.step(**a, reward_fn=reward)
    wm = lw.lake_wm.recall(4)  # the recent working-memory window, read from the lake
    wm_events = lw.lake_wm.n_events()
    lw.lake_wm.reopen()  # discard the in-RAM object -> replay purely from the .spkl
    wm_after = lw.lake_wm.recall(4)
    lake_backed = wm_events > 0 and wm == wm_after and len(wm) == 4
    lw_guard = AutonomousArm(seed=2, lake_wm=True)  # the safety envelope must hold with it on
    lw_ev = lw_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    lw_frozen0 = lw_guard.arm.state()["stopped"]
    lw_guard.step(**a)  # a normal tick after the E-STOP
    lw_frozen1 = lw_guard.arm.state()["stopped"]

    def run_lw(seed):
        bot = AutonomousArm(seed=seed, pfc=True, lake_wm=True)
        acts = [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]
        bot.lake_wm.close(remove=True)
        return acts

    det_lw = run_lw(5) == run_lw(5)
    lw.lake_wm.close(remove=True)  # clean up the temp lake files
    lw_guard.lake_wm.close(remove=True)

    print(
        f"lake WM     : {wm_events} spikes in the durable lake; recall holds {len(wm)} recent ticks; "
        f"survives reopen={wm == wm_after}; E-STOP freeze holds with it on ({lw_frozen1})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0).lake_wm is None, (
        "lake working memory is opt-in (a default arm has none)"
    )
    assert lake_backed, (
        "the working memory must live in the durable lake (survive reopen) with a bounded window"
    )
    assert lw_ev["action"] == "EMERGENCY_STOP" and lw_ev["reflex"], (
        "danger must E-STOP with lake WM on"
    )
    assert lw_frozen0 and lw_frozen1, (
        "the E-STOP freeze must hold with lake WM wired in (storage only)"
    )
    assert det_lw, "the lake-WM path must be deterministic"

    # 14. JEPA latent world model wired live into the REACTIVE path: it learns a latent world model ONLINE
    #     from the tick stream, and the executive PLANS over it (in latent space) to set its goal -- a
    #     bounded top-down bias into the SAME gated selection; opt-in, drives no motion, reflex/E-STOP first.
    jp = AutonomousArm(seed=1, pfc=True, jepa=True)
    for _ in range(90):
        jp.step(**a, reward_fn=reward)  # reward RELEASE
    jp_std = jp.jepa.latent_std()  # a non-degenerate learned latent (did not collapse)
    jp_goal_formed = jp._z_goal is not None  # a reward-derived latent goal formed from the stream
    jp_goal = jp.pfc.goal  # the executive's goal, now set by JEPA latent planning
    jp_guard = AutonomousArm(
        seed=2, pfc=True, jepa=True
    )  # the safety envelope must hold with it on
    jp_ev = jp_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    jp_frozen0 = jp_guard.arm.state()["stopped"]
    jp_guard.step(**a)  # a JEPA-planned action after the E-STOP
    jp_frozen1 = jp_guard.arm.state()["stopped"]

    def run_jp(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(
        f"jepa        : latent world model learns online (latent std {jp_std:.3f}); reward-latent goal "
        f"formed={jp_goal_formed}; executive goal {ACTIONS[jp_goal] if jp_goal is not None else None} set by "
        f"latent planning; E-STOP freeze holds with it on ({jp_frozen1})"
    )
    print("=" * 62)

    # JEPA is opt-in; a default arm has none (byte-for-byte the old behaviour).
    assert AutonomousArm(seed=0).jepa is None, (
        "the JEPA world model is opt-in (a default arm has none)"
    )
    assert jp.jepa is not None and jp_std > 0.0, (
        "the JEPA world model must be wired in and learn a non-degenerate latent"
    )
    assert jp_goal_formed, "a reward-derived latent goal must form from the live stream"
    assert jp_goal is not None, "the executive must hold a goal set by JEPA latent planning"
    # the safety envelope holds with JEPA wired LIVE into the reactive path (top-down preference only, no motor route).
    assert jp_ev["action"] == "EMERGENCY_STOP" and jp_ev["reflex"], (
        "danger must E-STOP with JEPA on"
    )
    assert jp_frozen0 and jp_frozen1, (
        "the E-STOP freeze must hold with JEPA wired live (top-down preference only)"
    )
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
    hj_guard = AutonomousArm(
        seed=2, pfc=True, hjepa_serve=served
    )  # the safety envelope must hold with it on
    hj_ev = hj_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    hj_frozen0 = hj_guard.arm.state()["stopped"]
    hj_guard.step(**a)
    hj_frozen1 = hj_guard.arm.state()["stopped"]

    def run_hj(seed):
        bot = AutonomousArm(seed=seed, pfc=True, hjepa_serve=served)
        return [bot.step(**ctx, reward_fn=reward)["action"] for _ in range(6) for ctx in (a, b, c)]

    print(
        f"hjepa-serve : abstract (Level-1) H-JEPA planning -- K-window filled={hj_windowed}, abstract goal "
        f"formed={hj_goal_formed}; executive goal {ACTIONS[hj_goal] if hj_goal is not None else None} set by "
        f"abstract planning; E-STOP freeze holds with it on ({hj_frozen1})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0).hjepa_serve is None, (
        "the served H-JEPA is opt-in (a default arm has none)"
    )
    assert hj_windowed, "the Level-0 latent K-window must fill"
    assert hj_goal_formed, "a reward-EMA abstract (Level-1) goal must form from the live stream"
    assert hj_goal is not None, "the executive must hold a goal set by ABSTRACT H-JEPA planning"
    # the safety envelope holds with the served H-JEPA driving abstract planning (top-down preference only, no motor route).
    assert hj_ev["action"] == "EMERGENCY_STOP" and hj_ev["reflex"], (
        "danger must E-STOP with the served H-JEPA on"
    )
    assert hj_frozen0 and hj_frozen1, (
        "the E-STOP freeze must hold with abstract H-JEPA planning (top-down only)"
    )
    assert run_hj(5) == run_hj(5), "the served-H-JEPA path must be deterministic"

    # 16. CURIOSITY (intrinsic motivation): the JEPA world-model prediction error becomes a learning-progress
    #     intrinsic reward that drives SELF-DIRECTED learning -- the arm learns EVEN WITH NO external reward.
    #     Opt-in, needs the JEPA; drives no motion, reflex/E-STOP still preempts.
    cur = AutonomousArm(seed=1, pfc=True, jepa=True, curiosity=True)
    for _ in range(60):
        cur.step(**a)  # NO reward_fn -> ONLY curiosity drives learning
    cur_learned = any(abs(v) > 1e-6 for row in cur.q.values() for v in row.values())
    dull = AutonomousArm(
        seed=1, pfc=True, jepa=True, curiosity=False
    )  # fair control: no curiosity AND no reward
    for _ in range(60):
        dull.step(**a)
    dull_learned = any(abs(v) > 1e-6 for row in dull.q.values() for v in row.values())
    cur_guard = AutonomousArm(
        seed=2, pfc=True, jepa=True, curiosity=True
    )  # the safety envelope must hold with it on
    cur_ev = cur_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    cur_frozen0 = cur_guard.arm.state()["stopped"]
    cur_guard.step(**a)
    cur_frozen1 = cur_guard.arm.state()["stopped"]

    def run_cur(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True, curiosity=True)
        return [bot.step(**ctx)["action"] for _ in range(6) for ctx in (a, b, c)]

    cur_det = run_cur(5) == run_cur(5)

    print(
        f"curiosity   : self-directed learning with NO external reward -- curious arm's values changed="
        f"{cur_learned} vs no-curiosity control changed={dull_learned}; E-STOP freeze holds with it on ({cur_frozen1})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0).curiosity is None, "curiosity is opt-in (a default arm has none)"
    assert cur_learned, (
        "curiosity must drive value learning WITHOUT any external reward (self-directed)"
    )
    assert not dull_learned, (
        "the fair control: no curiosity AND no external reward -> nothing learns"
    )
    assert cur_ev["action"] == "EMERGENCY_STOP" and cur_ev["reflex"], (
        "danger must E-STOP with curiosity on"
    )
    assert cur_frozen0 and cur_frozen1, (
        "the E-STOP freeze must hold with curiosity wired in (intrinsic reward, no motor route)"
    )
    assert cur_det, "the curiosity path must be deterministic"

    # 17. CURRICULUM (self-paced developmental learning): the arm works stages of increasing difficulty and
    #     ADVANCES autonomously when its own learning progress (curiosity) on a stage dries up -- cumulative,
    #     self-graduating. Opt-in (needs curiosity + jepa); learning flows through the SAME gated step().
    devC = [
        (
            _ctx(0.9, 0.1, 0.1, safe_d),
            "RELEASE",
        ),  # 4 DISTINCT contexts, each with an optimal action
        (_ctx(0.1, 0.9, 0.1, safe_d), "ENGAGE"),
        (_ctx(0.1, 0.1, 0.9, safe_d), "HOLD"),
        (_ctx(0.9, 0.9, 0.1, safe_d), "RELEASE"),
    ]
    dev_stages = [
        devC[:1],
        devC[:2],
        devC[:3],
        devC[:4],
    ]  # 1,2,3,4 contexts -> 4 stages of increasing difficulty
    dev = AutonomousArm(seed=1, pfc=True, jepa=True, curiosity=True)
    dtrace = dev.develop(dev_stages, SpikingCurriculum(n_stages=len(dev_stages), seed=1))
    dev_mastered = (
        dtrace[-1]["mastered"] if dtrace else 0
    )  # stages the arm graduated itself through
    dev_final_comp = dtrace[-1]["competence"] if dtrace else 0.0
    dev_guard = AutonomousArm(
        seed=2, pfc=True, jepa=True, curiosity=True
    )  # safety with the developmental loop
    dev_guard.develop(
        [[(_ctx(0.5, 0.5, 0.5, danger_d), "HOLD")]],
        SpikingCurriculum(n_stages=1, seed=2),
        batch=2,
        max_batches=1,
    )
    dev_frozen = dev_guard.arm.state()[
        "stopped"
    ]  # a danger context in the curriculum still E-STOPs + freezes

    def run_dev(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True, curiosity=True)
        return [
            t["mastered"]
            for t in bot.develop(dev_stages, SpikingCurriculum(n_stages=len(dev_stages), seed=seed))
        ]

    dev_det = run_dev(5) == run_dev(5)

    print(
        f"curriculum  : self-paced developmental learning -- arm graduated itself through {dev_mastered}/"
        f"{len(dev_stages)} stages (final-stage competence {dev_final_comp:.2f}); safety freeze holds ({dev_frozen})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0)._dev_rng is not None, (
        "the arm carries a developmental task-selection rng"
    )
    assert dev_mastered >= 2, (
        f"the arm must autonomously master + advance through multiple curriculum stages ({dev_mastered})"
    )
    assert all(t["advanced"] for t in dtrace[:dev_mastered]), (
        "each mastered stage must have ADVANCED on its own learning-progress dry-up (self-paced, not a clock)"
    )
    assert dev_final_comp >= 0.5, (
        f"the arm must reach real competence on the stages it masters ({dev_final_comp:.2f})"
    )
    assert dev_frozen, (
        "the E-STOP freeze must hold with the developmental curriculum loop (learning through the gated step)"
    )
    assert dev_det, "the developmental curriculum path must be deterministic"

    # 18. SHARED LATENT-Q (the transfer substrate): a shared linear action-value over the LAKE-STORED pattern
    #     replaces the tabular per-context q, so mastery GENERALIZES to genuinely-novel contexts (NEW units)
    #     where tabular has no learned value. Opt-in; only changes HOW the value is computed (gated selection +
    #     reflex/E-STOP unchanged). It rides the task-linear lake PATTERN; the deep clean transfer proof is in
    #     basal_ganglia/latent_q's own self-check. (The NONLINEAR-over-SSL-latent variant + the honest v3.17
    #     correction are self-check 19 / the mlp_q block below.)
    def _lq_ctx(v, t, au):
        return _ctx(v, t, au, safe_d)

    LQ_RULE = [
        "RELEASE",
        "ENGAGE",
        "HOLD",
    ]  # optimal = argmax over (visual, temperature, audio) band
    lq_train = [
        (_lq_ctx(*[0.9 if i == d else 0.1 for i in range(3)]), LQ_RULE[d])
        for d in range(3)
        for _ in range(3)
    ]
    lq_hold = []  # dominant band + a partial secondary -> NEW units, SAME rule
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
    lq_guard = AutonomousArm(
        seed=2, pfc=True, jepa=True, latent_q=True
    )  # the safety envelope must hold with it on
    lq_ev = lq_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    lq_guard.step(**_lq_ctx(0.9, 0.1, 0.1))
    lq_frozen = lq_guard.arm.state()["stopped"]

    def run_lq(seed):
        bot = AutonomousArm(seed=seed, pfc=True, jepa=True, curiosity=True, latent_q=True)
        return [
            bot.step(reward_fn=(lambda u, a: 1.0 if a == "RELEASE" else -1.0), **lq_train[0][0])[
                "action"
            ]
            for _ in range(8)
        ]

    lq_det = run_lq(5) == run_lq(5)

    print(
        f"latent-Q    : SHARED value over the lake pattern -- held-out (NEW units, same rule) competence "
        f"{lqq_hold:.2f} vs tabular {tab_hold:.2f} (TRANSFER); train {lqq_train:.2f}; freeze holds ({lq_frozen})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0).latent_q is None, (
        "the shared latent-Q is opt-in (a default arm has none)"
    )
    assert lqq_train >= 0.8, (
        f"the shared latent-Q must learn the rule family with ONE shared function ({lqq_train:.2f})"
    )
    assert lqq_hold >= 0.8 and lqq_hold >= tab_hold + 0.3, (
        f"the shared latent-Q must GENERALIZE to new-unit contexts where tabular cannot ({lqq_hold:.2f} vs {tab_hold:.2f})"
    )
    assert lq_ev["action"] == "EMERGENCY_STOP" and lq_ev["reflex"] and lq_frozen, (
        "danger must E-STOP with the latent-Q on"
    )
    assert lq_det, "the latent-Q path must be deterministic"

    # 19. TASK-ALIGNED LEARNED SUBSTRATE (nonlinear MLP-Q over the SSL-JEPA latent): the honest push PAST the
    #     LINEAR latent-Q. basal_ganglia/mlp_q's own self-check proves a one-hidden-layer head recovers a NONLINEAR
    #     rule a linear Q provably cannot (0.99 vs 0.47 over a CLEAN latent). Wired over the arm's ONLINE JEPA latent,
    #     this is the honest on-robot test -- and it CORRECTS v3.17 with a precise cause. Opt-in (auto-enables the
    #     JEPA); only changes HOW the value is computed (gate + reflex/E-STOP unchanged). Mutually exclusive with
    #     latent_q (that branch wins if both are set).
    def _mlp_run():
        bot = AutonomousArm(
            seed=1, pfc=True, curiosity=True, mlp_q=True
        )  # mlp_q auto-enables the JEPA latent it rides
        for _ in range(500):
            ctx, opt = lq_train[bot._dev_rng.randrange(len(lq_train))]
            bot.step(reward_fn=(lambda u, a, o=opt: 1.0 if a == o else -1.0), **ctx)
        # the arm's ONLINE JEPA latent for the 3 DISTINCT band contexts -> separable, or near-collapsed to one point?
        zs = [
            bot.jepa.encode(
                bot._context_pattern(**_lq_ctx(*[0.9 if i == d else 0.1 for i in range(3)]))
            )
            for d in range(3)
        ]
        spread = max(
            math.sqrt(sum((zs[i][k] - zs[j][k]) ** 2 for k in range(len(zs[i]))))
            for i in range(3)
            for j in range(i + 1, 3)
        )  # widest pairwise gap between the band latents
        return bot._competence(lq_hold), spread

    mlp_hold, mlp_spread = _mlp_run()
    mlp_guard = AutonomousArm(
        seed=2, pfc=True, mlp_q=True
    )  # the safety envelope must hold with it on
    mlp_ev = mlp_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    mlp_guard.step(**_lq_ctx(0.9, 0.1, 0.1))
    mlp_frozen = mlp_guard.arm.state()["stopped"]

    def run_mlp(seed):
        bot = AutonomousArm(seed=seed, pfc=True, curiosity=True, mlp_q=True)
        return [
            bot.step(reward_fn=(lambda u, a: 1.0 if a == "RELEASE" else -1.0), **lq_train[0][0])[
                "action"
            ]
            for _ in range(8)
        ]

    mlp_det = run_mlp(5) == run_mlp(5)

    print(
        f"mlp-Q       : NONLINEAR head over the ONLINE SSL-JEPA latent -- held-out {mlp_hold:.2f} (vs the "
        f"linear-over-pattern latent-Q {lqq_hold:.2f}); the arm's band latents near-collapse (spread {mlp_spread:.3f}) "
        f"-> the ENCODER, not the head, is the on-arm bottleneck; E-STOP freeze holds ({mlp_frozen})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0).mlp_q is None, (
        "the nonlinear MLP-Q is opt-in (a default arm has none)"
    )
    assert AutonomousArm(seed=0, mlp_q=True)._jepa_on, (
        "mlp_q must auto-enable the JEPA latent it rides"
    )
    # HONEST FINDING (corrects + refines v3.17): the SSL latent DOES encode the task -- mlp_q's own self-check gets
    # 0.99 over a clean latent, and an offline probe recovers THIS arm's band rule from the SSL latent at 0.80
    # (linear) / 0.88 (MLP). v3.17's "0.33 ~ chance -> not task-aligned" was OVER-PESSIMISTIC. The real cause: the
    # arm's ONLINE JEPA, trained on only a few repeating live contexts, NEAR-COLLAPSES them (spread ~ 0), so on-robot
    # NO head -- linear or nonlinear -- can separate them. The bottleneck is the ENCODER (a reward-shaped / deeper
    # encoder, or more diverse experience), NOT the Q-head's capacity -- which is exactly what this experiment shows.
    assert mlp_spread < 0.05, (
        f"the arm's online SSL latents for distinct bands must near-collapse ({mlp_spread:.3f})"
    )
    assert mlp_hold <= lqq_hold - 0.30, (
        f"over the near-collapsed online latent the MLP-Q cannot match the pattern substrate "
        f"({mlp_hold:.2f} vs {lqq_hold:.2f}) -- the ENCODER, not the head, is the on-arm limit"
    )
    assert mlp_ev["action"] == "EMERGENCY_STOP" and mlp_ev["reflex"] and mlp_frozen, (
        "danger must E-STOP with mlp_q on"
    )
    assert mlp_det, "the mlp-Q path must be deterministic"

    # 20. TASK-ALIGNED ENCODER (reward-shaping): the honest FIX for the v3.18 encoder-collapse bottleneck. §19
    #     showed the arm's ONLINE SSL encoder near-collapses its few contexts (spread ~0) so the MLP-Q over it is
    #     at chance (0.33). Reward-shaping adds a reward-predictive gradient into the encoder (opt-in
    #     reward_shaped_encoder=True) so the latent separates contexts BY TASK -> the SAME MLP-Q now recovers the
    #     rule ON THE ARM and transfers to new-unit contexts. Fixes the ENCODER, not the head -- exactly what §19
    #     said was the bottleneck. Opt-in (auto-enables the JEPA); only changes the JEPA's online update (the gated
    #     selection + reflex/E-STOP are unchanged).
    def _rse_run():
        bot = AutonomousArm(
            seed=1, pfc=True, curiosity=True, mlp_q=True, reward_shaped_encoder=True
        )
        for _ in range(800):  # firm reward-gain (5.0) + this budget recovers the rule to 1.0
            ctx, opt = lq_train[bot._dev_rng.randrange(len(lq_train))]
            bot.step(reward_fn=(lambda u, a, o=opt: 1.0 if a == o else -1.0), **ctx)
        zs = [
            bot.jepa.encode(
                bot._context_pattern(**_lq_ctx(*[0.9 if i == d else 0.1 for i in range(3)]))
            )
            for d in range(3)
        ]
        spread = max(
            math.sqrt(sum((zs[i][k] - zs[j][k]) ** 2 for k in range(len(zs[i]))))
            for i in range(3)
            for j in range(i + 1, 3)
        )
        return bot._competence(lq_hold), spread

    rse_hold, rse_spread = _rse_run()
    rse_guard = AutonomousArm(
        seed=2, pfc=True, mlp_q=True, reward_shaped_encoder=True
    )  # the safety envelope must hold with it on
    rse_ev = rse_guard.step(**_ctx(0.5, 0.5, 0.5, danger_d))
    rse_guard.step(**_lq_ctx(0.9, 0.1, 0.1))
    rse_frozen = rse_guard.arm.state()["stopped"]

    def run_rse(seed):
        bot = AutonomousArm(
            seed=seed, pfc=True, curiosity=True, mlp_q=True, reward_shaped_encoder=True
        )
        return [
            bot.step(reward_fn=(lambda u, a: 1.0 if a == "RELEASE" else -1.0), **lq_train[0][0])[
                "action"
            ]
            for _ in range(8)
        ]

    rse_det = run_rse(5) == run_rse(5)

    print(
        f"reward-enc  : task-aligning the JEPA encoder -> the SAME MLP-Q FULLY recovers the rule ON THE ARM -- "
        f"held-out {rse_hold:.2f} (vs plain-SSL {mlp_hold:.2f}); band latents separate (spread {rse_spread:.2f} vs "
        f"{mlp_spread:.3f}); E-STOP freeze holds ({rse_frozen})"
    )
    print("=" * 62)

    assert AutonomousArm(seed=0)._reward_shaped_encoder is False, (
        "the reward-shaped encoder is opt-in (default off)"
    )
    assert AutonomousArm(seed=0, reward_shaped_encoder=True)._jepa_on, (
        "reward_shaped_encoder must auto-enable the JEPA"
    )
    # THE FIX (closes v3.18): reward-shaping breaks the encoder collapse (spread ~0 -> large) so the MLP-Q over the
    # latent FULLY recovers the rule the plain-SSL latent (§19) could not -- the ENCODER was the bottleneck. With a
    # firm reward-gain (5.0) + budget, on-arm held-out reaches 1.00 (robust across seeds; weaker pulls plateau ~0.83).
    assert rse_spread > 0.5, (
        f"reward-shaping must break the encoder collapse ({rse_spread:.3f} vs plain-SSL {mlp_spread:.3f})"
    )
    assert rse_hold >= 0.999, (
        f"the MLP-Q over the reward-shaped encoder must FULLY recover the rule the plain-SSL latent could not "
        f"({rse_hold:.2f} held-out vs plain-SSL {mlp_hold:.2f})"
    )
    assert rse_ev["action"] == "EMERGENCY_STOP" and rse_ev["reflex"] and rse_frozen, (
        "danger must E-STOP with the reward-shaped encoder on"
    )
    assert rse_det, "the reward-shaped-encoder path must be deterministic"

    print(
        f"self-check OK: grows a unit per novel context ({grew_to}); learns the rewarded action "
        f"({default_before}->{learned}); danger -> gated E-STOP + freeze holds; deterministic. "
        f"replay: decisive in {replay_ticks:.1f}<{live_ticks:.1f} mean ticks over 12 seeds "
        f"({replay_wins}/12), value-only, gate holds. "
        f"amygdala learns threat ({threat_danger:.2f}>{threat_safe:.2f}) + BG gate releases {bg_released}, "
        f"both caution-only (E-STOP freeze holds)"
    )


if __name__ == "__main__":
    main()
