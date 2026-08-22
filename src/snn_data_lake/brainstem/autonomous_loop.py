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

import functools as _functools
import math
import random

from snn_data_lake.amygdala.fear_conditioning import Amygdala
from snn_data_lake.basal_ganglia.action_gating import BasalGangliaGate
from snn_data_lake.basal_ganglia.latent_q import LatentQ
from snn_data_lake.basal_ganglia.mlp_q import MlpQ
from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.cpg_oscillator import MatsuokaCPG
from snn_data_lake.brainstem.serial_arm import is_valid_command
from snn_data_lake.brainstem.so101_arm import _DEG as _SO101_DEG
from snn_data_lake.brainstem.so101_arm import So101Arm
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
from snn_data_lake.substrate.home import _T_COALITION, _log
from snn_data_lake.substrate.lake_working_memory import LakeWorkingMemory
from snn_data_lake.substrate.signal_ring import LakeJournaller, SignalRing
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "AutonomousArm._adapt_actuation": (
        "NOT_A_RULE",
        "adjusts how long a motor command may act, from whether it arrived. A timing PARAMETER of the "
        "body, not a synapse",
    ),
    "AutonomousArm._pfc_bias": (
        "NOT_A_RULE",
        "returns the prefrontal top-down bias vector over actions (all-zeros when opt-out). A read of "
        "executive state",
    ),
}

# Action-space granularity, NOT a tuned threshold (Policy 1): it defines the vocabulary, it does
# not arbitrate a decision. Sized small on purpose so a single exploratory move is always
# recoverable. The incumbent justified the size by what one step could NOT do on the old body
# ("cannot cross OBSTACLE_R=0.30, cannot reach the floor from home"); both facts were properties
# of a 2-link arm with reach 2.0 and neither survives the body swap, so the size now comes from
# arm_config, where it sits next to the measured joint limits it has to stay small against.
_JOINT_STEP_DEG = arm_config.DEFAULT_STEP_DEG


# distinct-from: brainstem/live_arm.py -- _joint_obs_delta parses a command STRING for one
#   observed joint's delta; this converts the WHOLE servo setpoint vector's units. Read.
def _commanded_deg(arm):
    """The servo setpoint in DEGREES, for the efference copy. So101Arm keeps `_commanded`
    in qpos RADIANS while state()['theta_deg'] is degrees; the copy folds both through the
    ONE proprioceptive fold, so the unit conversion lives HERE at the tap -- the envelope
    cannot grow an accessor without an envelope change, and a radians/degrees mix inside
    one fold is exactly the specs/011 class of defect. None when the body has no setpoint."""
    cmd = getattr(arm, "_commanded", None)
    if cmd is None:
        return None
    return [float(v) / _SO101_DEG for v in cmd]


_MOTOR = tuple(
    arm_config.command_for(n, s * _JOINT_STEP_DEG) for n in arm_config.ROTATABLE for s in (+1, -1)
)
# The arm's autonomous action repertoire. Until v4.58 this was ENGAGE/RELEASE/HOLD only -- three
# tokens, none of which move a joint, so the brain had no vocabulary with which to change its own
# next observation and the sensorimotor loop was causally OPEN. The four motor tokens are the
# precondition for closing it. ArmSim already accepted this exact command form; nothing in the
# safety envelope changed, and `apply()` still ignores every one of these while `stopped` is set.
ACTIONS = ("ENGAGE", "RELEASE", "HOLD") + _MOTOR  # the arm's autonomous action repertoire
# Every action must pass the WIRE allow-list, checked once at import. HOLD is the deliberate
# exception the safety envelope already documents: it is a decision not to move, consumed by
# this loop, and is intentionally not transmissible to a real arm.
_UNCHECKED = frozenset({"HOLD"})
assert all(a in _UNCHECKED or is_valid_command(a) for a in ACTIONS), (
    "an autonomous action is not accepted by the wire allow-list: "
    f"{[a for a in ACTIONS if a not in _UNCHECKED and not is_valid_command(a)]}"
)

_ACT_IX = {a: i for i, a in enumerate(ACTIONS)}  # action -> index into the td preference vector


def _ee_m(st):
    """End-effector position in METRES from a body state dict.

    The body reports `ee_mm` because millimetres are the unit the real SO-101 is specified and
    machined in. Everything in this loop -- the coverage cell, the allocentric step -- was
    written in metres, so the conversion happens once, here, rather than at each reader where
    one site would eventually be missed.
    """
    ee = st["ee_mm"]
    return (float(ee[0]) / 1000.0, float(ee[1]) / 1000.0, float(ee[2]) / 1000.0)


# The CPG's rhythmic pair, bound by joint NAME rather than by position in _MOTOR. The incumbent
# indexed _MOTOR[0] and _MOTOR[1], which happened to be joint A's two directions only because
# the tuple was built A-first; reordering the joint table would have silently re-pointed the
# oscillator at a different joint with nothing failing. shoulder_lift is the successor to the
# old joint A -- the same pitch joint, and the one the English cue "shoulder" already meant.
_CPG_JOINT = "shoulder_lift"
_A_POS = arm_config.command_for(_CPG_JOINT, +_JOINT_STEP_DEG)
_A_NEG = arm_config.command_for(_CPG_JOINT, -_JOINT_STEP_DEG)
_CPG_DRIVE = 1.0  # tonic descending drive; the oscillation exists for any drive > 0
_CPG_FLOOR = 0.25  # competence floor: the CPG's minimum share, so behaviour never collapses to
# nothing when the learned row is confident. A deliberate invariant, not a tuned threshold -- the
# analog of mhflocke's CompetenceGate.cpg_min = 0.40.
# Workspace discretisation for the coverage reward, in metres; a scale, not a gate.
#
# DERIVED, because the incumbent's 0.10 m was 1/20th of a reach of 2.0 and the new arm reaches
# roughly a quarter of a metre -- transcribing 0.10 across would have made a single cell wider
# than the entire workspace, so the arm would have scored one cell no matter where it went and
# the coverage reward would have read as "explored everything" from the first tick. What is
# worth carrying over is the RESOLUTION the incumbent chose (twenty cells per reach), not the
# length it happened to work out to on a different body.
_COVER_CELLS_PER_REACH = 20  # the incumbent's resolution: reach 2.0 / cell 0.10


@_functools.lru_cache(maxsize=1)
def _cover_cell():
    return (arm_config.MAX_REACH_MM / 1000.0) / _COVER_CELLS_PER_REACH


@_functools.lru_cache(maxsize=4)
def _lattice_ball_cells(r):
    """Integer triples (a, b, c) with a^2 + b^2 + c^2 <= r^2 -- the cells a ball of radius r
    cells can touch, counted exactly. O(r^3) once per resolution, then cached."""
    rr = int(r) * int(r)
    return sum(
        1
        for a in range(-int(r), int(r) + 1)
        for b in range(-int(r), int(r) + 1)
        for c in range(-int(r), int(r) + 1)
        if a * a + b * b + c * c <= rr
    )


#: specs/022 T038 (FR-027): the coverage ledger is LAKE-PERSISTED -- the name of its record in the
#: ONE lake (home.save_state / load_state, the P4b carrier time_cells already rides) and the format
#: tag a restore refuses to mis-read.
COVERAGE_STATE = "coverage_cells"
COVERAGE_FMT = 1
#: RESOLUTION of the live coverage RATE (new cells per tick over the last N coverage ticks) -- a
#: measurement window like _COVER_CELLS_PER_REACH is a grid, not an arbitrating threshold: the
#: rate is reported, never compared against a number. 200 ticks is the incumbent's 600-tick
#: measurement split in thirds.
_COVER_RATE_WINDOW = 200


# Consecutive ground-contact ticks before the postural recovery reflex fires. Small enough that the
# arm cannot spend a run underground, large enough that a single grazing contact is not treated as a
# fall. A latency, not a tuned threshold -- the condition itself (ArmSim reports GROUND) is exact.
_GROUND_RECOVER_TICKS = 3
SAFE_ACTION = "HOLD"  # what caution biases toward under stress
#: The reward CONTRACT's bound -- `reward_fn(unit, action) -> [-1, 1]`, enforced where the reward
#: enters `step` (out-of-contract values are clamped and the clamps counted). Not a tuned rate: it
#: is the maximum ACHIEVABLE reward, which is what count-based optimism is defined in terms of, and
#: changing it would change what the objective is allowed to say rather than how boldly the arm
#: explores. Read only by `_explore_noise`.
#:
#: TWO HONEST CONSEQUENCES, both measured while reviewing this choice. (a) Sigma is in CONTRACT
#: units, not in the units of what an objective actually pays, so a caller paying well under the
#: contract explores more than its own value margins need: a +-0.25 objective abandons its settled
#: optimum on 23.4% of ticks against 0.0% under a scale that adapts to it. No in-tree caller pays
#: that -- coverage_reward pays the full +-1.0 -- so it is disclosed rather than fixed, and it is
#: the first thing to re-measure if a quieter objective is ever added. (b) The clamp bounds
#: `r_ext`, not `r = r_ext + r_int`: on a curiosity arm the delivered value reaches 1.65, so ~35%
#: of rewards exceed this ceiling and their actions are, strictly, under-explored. Unchanged from
#: the retired scale, whose ceiling was also 1.0 after the same clamp (39.5% exceeded it there).
_REWARD_MAX = 1.0
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
        self.arm = So101Arm()
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
        # T038: per-tick freshness AND per-tick motion over the last _COVER_RATE_WINDOW coverage
        # ticks -> the live RATE and its companion beside the count. The rate alone cannot say WHY
        # it is zero (the review constructed both: a pre-covered workspace with the body moving and
        # an E-STOPped body both read rate 0.0); the PAIR can -- motion > 0 with rate 0 is ground
        # being retrodden, motion 0 is a body that stopped.
        self._cover_fresh = []
        self._cover_moved = []
        self._last_cover_cell = None
        #: T038: cells restored at construction from the lake (0 = cold start or refused) + why
        self._cover_restored = 0
        self.coverage_restore_note = "disabled"
        # Global-Workspace ignitions: how many times the broadcast HEAD changed hands (the workspace
        # holds up to `capacity` contents; this counts the top one). A running count of "something
        # new became globally available", which is a different claim from access_index (how
        # available) or pci (how integrated) -- both of which can sit high while the same content is
        # held. Counted in _workspace_tick, the one place the workspace steps.
        #
        # THIS COUNTER WAS STRUCTURALLY PINNED AT 1 AND IS NOW LIVE, which is worth knowing before
        # comparing it against any figure recorded before 2026-08-06. The bus used to subsidise a
        # content perception had already superseded, so the head never changed after the first
        # ignition: READ OFF `_ignitions` over 120 live ticks of `make_brain(seed=0, refit_every=10,
        # persist=False)` at `clock.phase = 0.25`, it was 1 -- the None -> first-percept transition
        # and nothing after it. With `thalamus/global_workspace.py::_defended` scoping that subsidy
        # to a tenancy the source still asserts, the same run counts 56. A rise here is the counter
        # starting to work, not the workspace becoming unstable -- inter-source competition is
        # unchanged (the workspace's own churn criterion still reads 9 against the argmax control's
        # 54); what moved is that perception's own percept changes now reach the head.
        #
        # Both numbers are READ FROM THIS ATTRIBUTE, not inferred from `thought`'s change count.
        # They differ from it (55 changes over the same run) because this is incremented only on the
        # wake ticks where `_workspace_broadcast` runs, and the first increment is the transition out
        # of None. A count taken on a different tick population is a different count.
        self._ignitions = 0
        self._ws_percept_ignited = False  # per-tick: did the percept seize the slot (the 4f port)
        self._prev_thought = None
        self._allo_last_ee = None  # previous end-effector pose, for _self_motion's dead reckoning
        self._last_disp = 0.0  # how far the end effector actually travelled on the last scored tick
        self._last_ee = None  # previous end-effector pose -> did this tick move the body at all?
        self._ground_ticks = 0  # consecutive ground-contact ticks (postural recovery reflex)
        self._recoveries = 0  # how many times the arm had to stand back up -- a real "falls" metric
        # Per-command running dz (metres) coincident with issuing it, measured by the tap: the
        # recovery reflex's compass. Learned, never assumed -- the incumbent's fixed +shoulder_lift
        # was written for the old 2-link joint A and points DOWN on the SO-101. Measured at the
        # recovery site itself (seed 2, 600 ticks): its 189 firings moved the end effector by at
        # most ONE MICRON in either direction (max +1e-6 m, mean -4e-6 m) and not one of them
        # cleared the GROUND collision on the tick it fired; five of eight seeds spent 456-577 of
        # 600 ticks in the floor.
        self._climb = {}
        self._climb_errors = 0  # passages whose height change could not be read
        # E-STOP SELF-CLEARS. Counted, never inferred: the authorisation to let the brain re-arm
        # itself was granted on condition that a self-clearing LOOP stays visible. A rising count
        # against a flat tick count is a brain tripping and clearing repeatedly, which is a
        # different failure from never clearing and must not look the same from outside.
        self._estop_clears = 0
        # NORD_ESTOP_SELF_CLEAR=1 enables the owner-authorised self-re-arm. Off by default so the
        # ten "the E-STOP freeze must hold" assertions across this module, cortex/brain.py and
        # cortex/live_brain.py keep testing exactly what they always tested.
        import os as _o  # noqa: PLC0415 -- module-level `os` is imported lazily further down

        self._estop_self_clear = _o.environ.get("NORD_ESTOP_SELF_CLEAR") == "1"
        self.lr = lr
        self.explore = explore
        self.rng = random.Random(seed)
        self.replay = replay  # opt-in SWR consolidation; default OFF -> behaviour unchanged
        self.replay_batch = replay_batch
        self.buffer = ReplayBuffer(seed=seed) if replay else None
        # opt-in Global Workspace: brain regions compete for one slot + broadcast the winner each tick.
        # ADDITIVE + read-only -- it selects no action and never touches the reflex/E-STOP (see step()).
        self.workspace = GlobalWorkspace(seed=seed) if workspace else None
        # T031: the bus's settled coalitions persist to the ONE lake -- SignalRing transport +
        # LakeJournaller drained every broadcast tick (_workspace_broadcast). `_log()` is the
        # CACHED accessor to the live BinLog, never a second construction over the live lake
        # (P4b). Capturing the instance for the arm's life matches KnowledgeLog's precedent;
        # the exposure (an out-of-band writer tearing the file mid-life) requires violating
        # the one-daemon-one-lake discipline every existing writer already relies on --
        # disclosed, not defended. Slots: publishes/tick <= the workspace's own capacity and
        # the drain runs the same tick, so 256 is deep headroom; overrun would mean the drain
        # stopped -- and overrun is FATAL at the journaller, never journaled around.
        #
        # base_ordinal: coalition identity CONTINUES FROM THE LAKE (count of records already
        # under the tag -- the KnowledgeLog counting precedent), so a restarted arm mints ids
        # the dedup lake has never seen. Instance-derived identity measurably lost restarted
        # lives to byte-identical dedup and collided ids across sessions (T031 review).
        if self.workspace is not None:
            self._coalition_ring = SignalRing(slots=256)
            _coalition_log = _log()
            _n0 = sum(1 for p in _coalition_log.replay() if p[:1] == _T_COALITION)
            self.workspace.attach_ring(self._coalition_ring, base_ordinal=_n0)
            self._coalition_journal = LakeJournaller(self._coalition_ring, _coalition_log)
            self._coalition_drained = 0  # records the lake actually wrote (accounting)
            self._coalition_deduped = 0  # replica refusals (see the drain site for semantics)
        else:
            self._coalition_ring = None
            self._coalition_journal = None
            self._coalition_drained = 0
            self._coalition_deduped = 0
        self._salience_on = bool(
            salience_novelty
        )  # opt-in bottom-up novelty salience -> a GWT competitor
        self._sal_ema = None  # running percept expectation (the habituation substrate)
        self._novelty = 0.0  # last bottom-up salience (deviance vs expectation)
        # The DERIVED salience perception submits to the one shared bus, and the running mean that
        # scales it. Exact running mean (sum-free Welford form), so there is no tau to pick -- see
        # _percept_salience for why a mean and not an EMA. `_sal_n == 0` means "no percept has been
        # scored yet"; the first sample IS its own mean, which is the honest "no baseline exists".
        self._percept_sal = 0.0
        self._sal_base = 0.0
        self._sal_n = 0
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
        self._last_intero = None  # this tick's full sense() reading (T031 coalition channels)
        # T037: the efference-copy boundary -- unconditional (pure observation + one
        # LOCAL model update per envelope passage; it issues nothing and gates nothing).
        from snn_data_lake.substrate.efferent_boundary import EfferentBoundary

        self.efferent = EfferentBoundary(seed=seed)
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
        self._reward_clamps = 0  # reward_fn returned outside its documented [-1, 1] contract
        self._paid = False  # has any external reward been delivered? (gates exploration at all)
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

    def _adapt_actuation(self, state):
        """Update how long a motor command is allowed to act, from whether it arrived.

        The body starts at one physics step per command -- the integrator's own step, and far
        too short for a position servo to travel anywhere: measured, 0.020 deg of a commanded
        45. Rather than transcribe a settling time, the loop raises the rate while commands
        fail to arrive and lets it fall back while they succeed, so the arm converges on
        whatever its own dynamics need and RE-converges if the body, the load or the payload
        changes. That is the property a fixed step count cannot have.

        A latched arm is skipped: it reports not-reached because it is frozen, which says
        nothing about rate, and feeding that in would drive the rate to its ceiling during
        every E-STOP.
        """
        if not isinstance(state, dict) or state.get("stopped"):
            return
        n = getattr(self.arm, "substeps", None)
        if n is None:
            return
        if state.get("reached"):
            self.arm.set_substeps(n - 1)  # give it back slowly: a longer tick is a real cost
        else:
            self.arm.set_substeps(max(n + 1, n * 2))  # find it fast

    # ---- T038 (FR-027): the coverage ledger's two measured properties -----------------------
    # distinct-from: cortex/time_cells.py -- save()/restored() persist the CLOCK's thresholds
    #   with a seed/drive lineage; this persists the coverage CELLS with a cell-size/reach
    #   lineage. Same carrier (home.save_state), different state, each refusing its own
    #   mismatch. Read.
    # distinct-from: cortex/neurogenesis_engine.py -- growth_state_bytes/restore_growth_state is
    #   the grown SUBSTRATE's own chain-record path. Read.
    def coverage_rate(self, window=None):
        """NEW cells per coverage tick over the last `window` coverage ticks (default: the whole
        _COVER_RATE_WINDOW ledger). 0.0 before any coverage tick -- read `coverage_window()`
        beside it, an empty window is 'nothing measured yet', not 'saturated'. After a restore the
        adopted ground is not fresh, so this is new-since-restore: the same exploration reads lower
        warm than cold (0.69 vs 0.12 at 100 ticks, measured), which is what it should say.
        Reported, never judged."""
        hist = self._cover_fresh if window is None else self._cover_fresh[-int(window) :]
        return float(sum(hist)) / len(hist) if hist else 0.0

    def coverage_motion(self, window=None):
        """Coverage ticks in the window on which the end effector's CELL changed, as a fraction --
        the rate's companion. rate 0 with motion > 0 is ground being retrodden (the workspace at
        this resolution is used up, or the policy circles); rate 0 with motion 0 is a body that
        stopped, which is what the landing run was (HOLD 100/100 from tick 501). 0.0 before two
        coverage ticks."""
        hist = self._cover_moved if window is None else self._cover_moved[-int(window) :]
        return float(sum(hist)) / len(hist) if hist else 0.0

    def coverage_window(self):
        """Coverage ticks currently in the ledger (0.._COVER_RATE_WINDOW): the denominator the two
        fractions above are read over. Served so a 0.0 can be told from an empty window."""
        return len(self._cover_fresh)

    @staticmethod
    def coverage_scale_cells():
        """The workspace SCALE in cells that `cells_covered` is read against, DERIVED (P1) from the
        body: the lattice points within a ball of radius `_COVER_CELLS_PER_REACH` cells (the
        sampled horizontal reach at the cell resolution), counted exactly -- the continuous volume
        undercounts a small-r lattice (r=2 holds 33 points; 4/3*pi*8 = 33.5).

        A SCALE, NOT A BOUND, and named so. The reach is sampled horizontally from the shoulder
        while cells are indexed from the base frame, and the gripper rises above the horizontal
        reach (z to 0.516 m against 0.477 m): the landing run reaches 22.3 cells from the origin
        and 4 of its 107 cells lie outside this ball. `cells / scale` (0.24% at landing) is an
        order of magnitude, never a plateau someone pinned."""
        return _lattice_ball_cells(int(_COVER_CELLS_PER_REACH))

    def coverage_state(self):
        """The ledger as a lake record: cells + lineage (cell size, reach) + the format tag.
        Nothing else -- a field the restore does not adopt is a claim the lake keeps forever."""
        return {
            "fmt": COVERAGE_FMT,
            "cell_m": float(_cover_cell()),
            "reach_mm": float(arm_config.MAX_REACH_MM),
            "cells": sorted([int(a), int(b), int(c)] for a, b, c in self._covered),
        }

    def save_coverage(self):
        """Persist the ledger into the ONE lake (P4b). True if a record was written; False when
        there is nothing to write or nothing changed (home.save_state refuses both -- the lake
        is append-only). Never raises: persistence must not take down the caller.

        Size: ~14 bytes per cell as JSON (1.5 KB at landing). The record rides _T_STATE, whose
        per-tick append budget (tools/lake_append_baseline.json, tag 7) trips near ~7,000 cells
        in one flush; the flush is change-gated so nothing re-appends an unchanged ledger.
        ponytail: a packed encoding (~6 B/cell) is the upgrade if the body ever covers that much."""
        if not self._covered:
            return False
        try:
            from snn_data_lake.substrate import home

            return bool(home.save_state(COVERAGE_STATE, self.coverage_state()))
        except Exception:
            return False

    def restore_coverage(self, blob):
        """Adopt a persisted ledger WHOLESALE, or refuse it and say why. Never mutates on refusal.
        Refuses: nothing persisted, a foreign format, a cell-size or reach lineage mismatch (a
        cell index under another grid is another body's cell), malformed cells. Returns
        (adopted_count, reason) and records both on the instance (`coverage_restore_note`)."""
        if not isinstance(blob, dict):
            self.coverage_restore_note = "nothing persisted"
            return 0, self.coverage_restore_note
        why = None
        try:  # ONE guard for every field: a record that raises is a record that was not refused
            if blob.get("fmt") != COVERAGE_FMT:
                why = f"fmt {blob.get('fmt')!r} != {COVERAGE_FMT}"
            elif abs(float(blob.get("cell_m", -1.0)) - _cover_cell()) > 1e-12:
                why = f"cell size {blob.get('cell_m')} != {_cover_cell()} (the grid changed)"
            elif abs(float(blob.get("reach_mm", -1.0)) - float(arm_config.MAX_REACH_MM)) > 1e-9:
                why = (
                    f"reach {blob.get('reach_mm')} != {arm_config.MAX_REACH_MM} (the body changed)"
                )
            else:
                cells = set()
                for a, b, c in blob.get("cells", []):
                    if not all(isinstance(v, int) for v in (a, b, c)):
                        raise ValueError("non-integer cell index")  # int(1.9) is another cell
                    cells.add((a, b, c))
        except (TypeError, ValueError) as e:
            why = f"malformed record ({type(e).__name__}: {e})"
        if why:
            self.coverage_restore_note = "refused: " + why
            return 0, self.coverage_restore_note
        self._covered = set(cells)
        self._cover_new = len(
            cells
        )  # keeps `_cover_new == len(_covered)`, the invariant telemetry relies on
        self._cover_restored = len(cells)
        self.coverage_restore_note = f"restored {len(cells)} cells from the lake"
        return len(cells), self.coverage_restore_note

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
        # THREE dimensions now, because the body has them. The incumbent's arm was planar, so
        # (x, y) named a position exactly; on a 6-DOF arm two poses that differ only in height
        # would collapse to the same cell and lifting the gripper straight up would count as
        # exploring nothing.
        ee = _ee_m(st)
        c = _cover_cell()
        cell = (round(ee[0] / c), round(ee[1] / c), round(ee[2] / c))
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
        self._cover_fresh.append(1 if fresh else 0)  # T038: the rate's ledger ...
        self._cover_moved.append(  # ... and its motion companion (cell changed since the last coverage tick)
            1 if (self._last_cover_cell is not None and cell != self._last_cover_cell) else 0
        )
        self._last_cover_cell = cell
        if len(self._cover_fresh) > _COVER_RATE_WINDOW:
            del self._cover_fresh[0]
            del self._cover_moved[0]
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
        # NO `blocked` BRANCH, AND THE NULL IS THE POINT (measured 2026-08-05, P8 guard (d)).
        #
        # A -0.30 branch lived here, directly below the collision return, and NEVER FIRED.
        # `So101Arm.state()` computes blocked = (not reached) and (collision is not None or
        # any(saturated)) and not moving. So the branch was reachable only on
        # `blocked and collision is None`, i.e. only via the `saturated`-ALONE disjunct -- and
        # on this body that disjunct never co-occurs with the other two conjuncts -- and the
        # reason is in the MODEL, not in the sampling. Read off the scene: actuator_forcerange
        # 3.35 Nm, dof_frictionloss 0.052, dof_damping 0.6, so a saturated joint in free space
        # settles at (3.35 - 0.052) / 0.6 = 5.50 rad/s, while `moving` trips at
        # ||droop_tol|| = 7.15e-4 rad/s -- about 7,700x lower. (An arithmetic bound from the
        # model, not an observed peak.) A saturated joint in free space therefore CANNOT be
        # stationary: something has to be holding it, the only thing on this body that can is
        # contact, and the line above already pays -1.0 for that. `blocked`
        # is therefore a strict SUBSET of `collision` here, and the collision branch shadows all
        # of it. Its own counter `_blocked_ticks` was never even created (0 readers tree-wide
        # either way -- `brain_telemetry` reads `_covered`/`_last_disp`/`_ignitions`, not it).
        #
        # Three independent measurements, all re-run for this deletion:
        #   * LIVE, 3 seeds x 600 ticks with this function as the reward_fn (what
        #     `cortex/brain.py::Brain._wake_tick` passes on every wake tick), driven by a
        #     random.Random(seed) visual/temperature/audio/distance stream: 538 blocked states
        #     (0, 263, 275 by seed), ZERO of them collision-free, -0.30 returned 0 times. The
        #     COUNT is protocol-dependent and must not be quoted as if it were not: an
        #     independent audit of the same nominal "3 seeds x 600 live ticks" read 784 blocked
        #     (37, 196, 551) off a different driving stream. It is the ZERO that is load-bearing
        #     and the zero is what both readings agree on.
        #   * CONSTRUCTED, a sweep of every rotatable joint across its full commandable range at
        #     5 actuation rates x 2 settle depths plus a gripper stall, 656 samples: 20 blocked,
        #     332 saturated-and-collision-free, and 0 blocked-and-collision-free.
        #   * ADVERSARIAL, five protocols aimed at DIFFERENT stall mechanisms (full-range MOVETO
        #     sweep; 100,000 single-step ticks hunting the overshoot turnaround, where a
        #     saturating servo's ||qvel|| crosses zero with the error still large; gripper
        #     stall; E-STOP mid-flight, which returns state() without stepping; random command
        #     streams): 114,835 samples, 7,840 blocked, 12,314 saturated-and-collision-free,
        #     0 blocked-and-collision-free. That last pair is the empirical half of the argument
        #     above -- saturation without contact is COMMON, and it was moving every time.
        #
        # The comment that justified the branch claimed the flag "fires on 28% of in-limit
        # targets". The FLAG does fire (263/600 and 275/600 on two of three live seeds); the
        # BRANCH consumed 0% of it. The 28% itself did not reproduce under any protocol
        # constructed here -- the sweep above reads 3.0% (20/656).
        #
        # DELETED rather than weakened, per P8 falsifier F11. The deletion is behaviour-neutral
        # by MEASUREMENT, not by inference: the pre-edit function (loaded from the git blob) and
        # this one were driven with the same seeded stream for 3 x 600 ticks and produced an
        # identical reward at every single tick, an identical final joint pose, and an identical
        # covered-cell count. Two tempting "fixes" are both
        # wrong and are named so they are not re-attempted: reordering this above the collision
        # return would demote the strongest negative, and relaxing `moving` or a joint's droop
        # tolerance to let saturation alone count as obstruction is what once made the objective
        # pay -0.30 for an ordinary accelerating move and +0.15 for standing still (see
        # `test_blocked_is_false_while_the_arm_accelerates`). Saturation is not obstruction.
        # If `blocked` is ever to be paid for, the reachable condition is
        # `blocked and collision is None`, and it must be MEASURED non-zero BEFORE the branch is
        # written back -- it was 0 of 1,800 live ticks and 0 of 656 constructed samples here.
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
            # metres, to stay in the same units as _ALLO_MOVE's nominal displacements. The
            # body reports millimetres; differencing mm against a nominal step in metres
            # would have scaled every real motion up by a thousand.
            ee = _ee_m(self.arm.state())
            now = (ee[0], ee[1])
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
        it, so sigma is the reward scale -- `_REWARD_MAX`, the contract's ceiling, i.e. what the
        objective COULD pay rather than what it has paid -- divided by sqrt(1 + times THIS ACTION
        was taken in THIS unit) -- count-based optimism (Strehl &
        Littman), the family the audit itself recommended for
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
        if unit is None or not self.explore or not self._paid:
            # Three ways there is nothing to be uncertain BETWEEN. A direct query with no context
            # identity has no state-action counts to be optimistic about. `explore` off is the
            # isolation probes' deterministic instrument. And before any reward has been DELIVERED
            # every action scores identically, so noise buys no information and only destroys the
            # naive tie-break prior -- measured when this clause was briefly dropped: the module's
            # own learning probe read RELEASE as its BEFORE action, which is the action it is
            # supposed to LEARN, collapsing 'learned, not fixed' into a coin that landed right.
            return [0.0] * len(scores)
        # THE SCALE IS WHAT THE OBJECTIVE COULD PAY, NOT WHAT IT HAS PAID. Two OBSERVED scales were
        # shipped and both starved, each in a regime the other's benchmark did not contain:
        #   * a running MEAN of |r| converges to whatever the CURRENT action pays, so under any
        #     settled action it annealed exploration to nothing -- the coverage objective's HOLD lock
        #     (sigma 0.005 against a 0.29 value margin after 400 ticks of -0.10; flip probability 0);
        #   * a running PEAK of |r_ext| looks immune to that, and is -- on a COLD body, which is the
        #     only body the 8-seed benchmark contained. On the LIVE daemon the arm restores its
        #     coverage ledger, so it never earns the +1.0 for a fresh cell, and near HOME it never
        #     collides: the only rewards it observes are the -0.10 stillness term and the -0.02
        #     re-tread, so the peak is starved to 0.10 and the same annealing returns (measured over
        #     194 arm steps: max |r_ext| = 0.10, never once the 1.0 a fresh cell or a collision
        #     would have paid). Measured in the daemon's own configuration
        #     (make_brain against a copy of the real lake, 400 ticks): theta moved on 9 of 400 ticks,
        #     motion 0.015, ZERO new cells -- against 139 of 400, motion 0.469 and 56 new cells here.
        # Optimism is defined in terms of the maximum ACHIEVABLE reward (Strehl & Littman, the family
        # cited above), and the contract bounds that at _REWARD_MAX -- a bound this loop ENFORCES
        # rather than assumes. An observed scale cannot bootstrap: the arm has to explore to sample
        # the reward that would license the exploration.
        return [
            self.rng.gauss(0.0, _REWARD_MAX / math.sqrt(1.0 + self._visits.get((unit, a), 0)))
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
        channel; if the tonic GPi is never overcome it releases NOTHING, which falls back to the safe
        action. Selection only ever picks among the SAME actions -- the gated arm is untouched and
        the reflex/E-STOP (checked earlier in step) still preempts it.

        WHAT OVERCOMES THE TONIC GPi, MEASURED (T039d, and this paragraph replaces a claim that was
        FALSE on the live call shape). This method used to say 'no drive -> releases NOTHING', and
        the self-check that certified it passed a default `unit=None`, for which `_explore_noise`
        returns zeros -- so it held at any exploration scale, including 1e6, and could never go red.
        On the LIVE shape (a real grown unit, zero cortical drive, untried actions) the split is:
          explore OFF -> the safe action on 200 of 200 seeds  -- the GATE adds no drive of its own;
          explore ON  -> a non-safe action on 186 of 200 seeds -- COUNT-BASED OPTIMISM releases it.
        The second half is not spurious motion, it is what optimism is FOR: an untried action carries
        full optimism until it is tried, and a gate that refused to release without prior value is
        the cold-start deadlock T038a/T038b measured (a body that will not move cannot learn that
        moving pays). Safety does not rest on this line -- the reflex/E-STOP preempts earlier in
        step() and every released command still passes the gated arm."""
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

    # distinct-from: medulla/interoceptive_arousal.py -- the ORGAN this method reads (it owns
    #   sense()); this is the arm's per-tick read of it. distinct-from: hypothalamus/
    #   homeostatic_drives.py -- the drives producer on the declared insula tract, not read
    #   here. Both sites read.
    def _interocept(self, load):
        """The interoceptive caution in [0, 1] for the current homeostatic load (0.0 when opt-out).
        The medullary/hypothalamic homeostats map the body's load to a bounded caution that biases
        selection toward the safe action; it selects no action and gates no motion.

        T031: the FULL reading is cached on `_last_intero` for the same tick's coalition
        channels (_workspace_broadcast) -- ONE sense() per tick, never a second call, because
        sense() advances the organ's own noise stream and dynamics (G5: a second draw per
        tick would fork replay)."""
        if self.intero is None:
            return 0.0
        self._last_intero = self.intero.sense(load)
        return self._last_intero["caution"]

    # distinct-from: medulla/interoceptive_arousal.sense() -- the ORGAN'S draw, which advances
    #   its noise stream; this returns the reading ALREADY taken. Both sites read.
    def interoception(self):
        """This tick's cached interoceptive reading as a plain dict, or {} when opt-out.

        A READ OF THE CACHE, NEVER A SECOND DRAW. `_interocept` above states the organ's rule:
        ONE sense() per tick, because sense() advances InteroceptiveArousal's own noise stream
        and dynamics, so a second call per tick forks replay (G5). This accessor exists so a
        consumer outside the arm can have the body reading WITHOUT being able to take it --
        the only safe shape for that, and the reason it returns a COPY: a caller that mutated
        the cache would corrupt the same tick's coalition channels, which read it too.

        Empty (never None) when interoception is opt-out, so a consumer reads one type.
        specs/022 T039f: the insula's live feed."""
        return dict(self._last_intero) if self._last_intero else {}

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

    def _percept_salience(self, deviance):
        """Perception's claim on the workspace, DERIVED from the percept itself.

        THE DEFECT THIS REPLACES, measured before the edit on 150 LIVE wake ticks of the canonical
        brain (`make_brain()` + `Brain.tick()`, subjective day forced): every submission reaching
        the one shared bus was `("perception", 0..3, 0.6)` -- salience distinct=1, n_changes=0 --
        and the broadcast content was the same value on all 150 ticks. A bus carrying a literal is
        a bus carrying no information, so every downstream read of it was inert BY CONSTRUCTION,
        and adding readers to it would have raised a read count without moving anything (the
        force-wiring P6 explicitly forbids). The signal had to be fixed before any route could mean
        anything.

        Two halves, both DERIVED, neither hand-picked:

        * WHERE IN THE BAND -- `deviance / (deviance + mu)`, with `mu` the EXACT running mean of the
          deviance INCLUDING this sample. Baseline-relative per P1, and a running mean is chosen
          over an EMA precisely because it has no rate constant to pick: there is no tau in here to
          be a magic number in disguise. It reads 0.5 on the first sample by construction (one
          sample is its own mean), which is the honest value for "no baseline exists yet"; it rises
          toward 1 as the percept deviates further ABOVE its own history and falls toward 0 as it
          settles below it.
        * HOW WIDE THE BAND IS -- read off the CONSUMER's own declared parameters, not chosen here.
          `ignition_threshold` is the bus's rheobase: a pool driven below it emits no spike at all
          (`global_workspace` sets `_v_th = ignition_threshold / _LEAK` for exactly this), so
          nothing below it is a claim at all. `1 - (self_boost + hysteresis)` is the strongest claim
          a percept may make while an incumbent holding the slot can still be DISPLACED by a
          maximal competitor -- the invariant that keeps a proximity threat able to seize the
          workspace. Sizing the band from the bus's own constants is what makes this a derivation
          rather than a second guess at 0.6.

        THE BAND IS VERIFIED, NOT ONLY DERIVED, because the linear drive arithmetic underestimates
        an incumbent's advantage. Swept against the real `GlobalWorkspace` at defaults, first-tick
        perception salience 0.05..1.00 in steps of 0.05 followed by a threat submission at 0.95 AND
        at 1.00: salience <= 0.50 never ignites at all, salience >= 0.85 makes the incumbent
        percept indisplaceable by EITHER threat value, and the admissible window is exactly
        [0.55, 0.80]. The band this returns is [0.55, 0.75] on the defaults -- inside it with
        margin, by construction rather than by luck.

        A RUNNING MEAN AND NOT SOMETHING CLEVERER -- RE-DERIVED 2026-08-06, BECAUSE THE NULL THAT
        USED TO JUSTIFY IT IS DEAD AND ITS REPLACEMENT POINTS THE OTHER WAY. The live deviance
        series is right-skewed and opens with a one-off onset of 1.0, so a mean-relative ratio sits
        below 0.5 and uses only about half the band -- which looks like a reason to reach for a
        median or a rank. This paragraph used to answer that with a null: all three forms replayed
        through the REAL `GlobalWorkspace` "broadcast exactly ONE content value with ZERO flips",
        so the extra machinery bought nothing and the parameter-free form shipped by default.
        THAT MEASUREMENT WAS TAKEN ON AN ARBITER THAT COULD NOT HAVE DISTINGUISHED THEM. It held a
        content its own source had already superseded (`thalamus/global_workspace.py::_defended`
        records the defect and the fix), so EVERY producer-side form was going to read one content
        and zero flips -- the null was a property of the consumer, and reading it as a verdict on
        these three forms was measuring the bus and scoring the signal.

        RE-RUN on the fixed arbiter, same three forms, same captured live series (74 wake ticks,
        4 contents), and the comparison now discriminates -- so it is a real P2 result rather than
        a tie broken on parsimony:

            form             salience range        distinct   heads  flips  TRACKS the percept
            running mean     [0.5684, 0.6716]            74       4     73     74/74
            running median   [0.5834, 0.6806]            71       4     73     74/74
            rank(64 deep)    [0.5500, 0.7395]            63       4     59     60/74

        The same three on the PRE-FIX arbiter read 1/0, 1/0 and 3/8 heads/flips and tracked 25, 25
        and 26 of 74 -- which is the old null, reproduced, and the reason it must not be quoted.
        The running mean is TIED-BEST at 74/74 and the rank form LOSES 14 ticks: its window puts a
        rank-0 sample at exactly the rheobase, where the percept makes no audible claim at all and
        the workspace keeps broadcasting the previous one. So the form with no parameters is now
        the form that is at least as good AND has no parameters, which is a stronger reason to ship
        it than the one this paragraph used to give.

        WHAT THIS CHANNEL CAN NOW MOVE -- the paragraph here used to open "WHY THE BROADCAST CONTENT
        STILL CANNOT MOVE, and it is NOT this signal's fault", and that is CLOSED. It was true, and
        its cause was one level up: the bus paid its occupant `self_boost + hysteresis = 0.25` of
        defence while the widest claim a compliant producer may make above the rheobase is
        `hi - lo = 0.20`, and the occupant kept that subsidy even after perception had moved to a
        different percept. With the subsidy scoped to a tenancy the source is still asserting, the
        broadcast follows this signal: measured over 120 live ticks of `make_brain(seed=0,
        refit_every=10, persist=False)` at `clock.phase = 0.25`, `cortex/live_brain.py::thought`
        takes 4 distinct values with 55 changes and NAMES THE PERCEPT SUBMITTED THIS TICK on 56 of
        56 posting ticks (and holds the last one on all 64 ticks where perception is silent, which
        is the hysteresis working rather than failing). The same run on the pre-fix bus: 1 distinct
        value, 0 changes, 19 of 56. What is still capped is the RANGE, not the content -- this
        channel offers a downstream reader at most `hi - lo = 0.20` of swing, against the > 0.30
        shift `cortex/access_consciousness` was measured to need before its own output moves, and
        that ceiling is a property of `thalamus/global_workspace.py`'s constants.

        Checked at the four degenerate inputs, because a P1 rewrite in this tree already shipped a
        `mu/1e-12 = 1e12` singularity onto a live path. ZERO deviance -> ratio 0 -> the bare
        rheobase: a percept identical to its own expectation makes the minimum claim. SILENCE
        (zero deviance AND a zero baseline, i.e. an unvarying stream from the first tick) -> the
        sum is exactly 0 and the guard returns the rheobase instead of dividing. SINGLE sample ->
        exactly 0.5 as above. An EMPTY pattern -> `_percept_novelty` already floors its divisor at
        1, so the deviance is 0 and this is the ZERO case. There is no epsilon anywhere in here:
        the only guard is an exact `> 0.0` test, so there is no denominator that can be small
        rather than zero."""
        ws = self.workspace
        if ws is None:
            return 0.0
        dev = float(deviance)
        self._sal_n += 1
        # The baseline RULE moved to `GlobalWorkspace.next_baseline` 2026-08-06 and is now shared
        # with the access bid. This line's arithmetic is unchanged -- it was already the running
        # mean, and it is the form the shared rule adopted, because a running mean carries no time
        # constant while an EMA does (P1). Sharing the MAP was not enough: two producers computing
        # `baseline` by different rules feed ratios that are not on one scale, and the shared map
        # inherits that difference intact. Measured: symmetrising only the map left the repoint
        # byte-identical.
        self._sal_base = ws.next_baseline(self._sal_base, dev, self._sal_n)
        # The band map MOVED to `thalamus/global_workspace.py::band_claim` 2026-08-06 and is now
        # SHARED with cortex/brain.py::_update_access -- not copied, called. This method's arithmetic
        # is byte-for-byte what it was (same floor, same ceiling, same self-normalising ratio, same
        # hi<=lo guard, same total<=0 guard); what changed is that ONE definition of "a compliant
        # claim" now serves every producer, so two producers are finally competing in the same
        # coordinate system. P2: the derivation exists once. Keeping a second copy here -- even an
        # identical one -- is how the two drifted apart in the first place.
        return ws.band_claim(dev, self._sal_base)

    def _workspace_broadcast(self, percept, threat_salience):
        """Additive Global-Workspace tick: brain regions (perception + a salience/threat detector)
        compete for the workspace's slots and the ranked head is returned. READ-ONLY annotation --
        it selects no action, gates no motion, and never touches the reflex/E-STOP.
        Returns the head broadcast content, or None when there is no workspace.

        Perception's salience is DERIVED per tick (see _percept_salience); it was the literal 0.6
        until this pass. The threat channel carries the amygdala's REAL learned output and always
        did -- see the caller in step() for the measured reason it reads 0.0 on this brain.

        OCCUPANCY ON THIS BUS IS 1 OF 4, MEASURED, and that is a producer fact rather than a
        capacity one: `perception` is the only admissible submitter, because `salience`/threat posts
        a structural 0.0 on every wake tick (the amygdala is never paired on the wake path -- the
        self-check proves that null in both directions). So the head IS the whole workspace here,
        `broadcast_all()` returns a 1-tuple, and no tick of this bus is decided by a rank tie-break.
        Adding a second admissible producer is what would change that, and it is not this method's
        work."""
        if self.workspace is None:
            return None
        if percept is not None:
            # T031: the perception coalition carries the tick's REAL interoceptive state as
            # typed channels -- medulla's declared arousal axes (stress -> arousal, caution),
            # read from the reading _interocept already took THIS tick (never a second
            # sense(); see _interocept). Salience-only when interoception is opt-out: absent
            # channels stay absent, never defaulted (the record contract's honesty rule).
            ch = {"salience": self._percept_sal}
            if self._last_intero is not None:
                ch["arousal"] = float(self._last_intero["stress"])
                ch["caution"] = float(self._last_intero["caution"])
            self.workspace.submit_coalition("perception", percept, ch)
        self.workspace.submit("salience", "threat", threat_salience)
        if self._salience_on:
            self.workspace.submit(
                "salience", "novelty", self._novelty
            )  # bottom-up stimulus salience competes
        content = self.workspace.step()
        if self._coalition_journal is not None:
            # T031: settled coalition records -> the ONE lake, same tick they settle. A raise
            # here (RingOverrun) is the P4b contract working: never journal around a hole.
            # drain() returns how many the lake ACTUALLY wrote; the shortfall against the
            # ring head is COUNTED and exposed, not raised -- because under lake-derived ids
            # a byte-identical refusal is structurally a REPLICA refusal, and refusing a
            # replica is the lake's P4 one-brain guard working, not loss. Measured in the
            # brain's own determinism claim: two same-seed lockstep Brains (d1/d2) mint
            # identical records from the same base and the lake correctly stores ONE -- a
            # strict published==persisted raise here reddened that legitimate refusal. The
            # restart-amnesia hole this counter once guarded is closed UPSTREAM by identity
            # (attach_ring's base_ordinal): a restarted life mints ids the lake has never
            # held, so its records can never byte-collide -- proven by the workspace
            # self-check's restart arm and its retired-scheme red.
            wrote = self._coalition_journal.drain()
            self._coalition_drained += wrote
            self._coalition_deduped = self._coalition_ring.head - self._coalition_drained
        # The ignition VERDICT, ported 2026-08-12 from cortex/competence_executive.WorkspaceBroadcast
        # ahead of that wrapper's retirement -- the one expression the wrapper added over the bus:
        # did THIS loop's own percept submission seize the slot this tick. Distinct from
        # `_ignitions` below, which counts content CHANGES across ticks. Consumed by
        # substrate/brain_telemetry ("ws_percept_ignited"), the same surface that reads `_ignitions`.
        self._ws_percept_ignited = percept is not None and content == percept
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

    # distinct-from: thalamus/reentrant_workspace.py -- that is the theta AUTHORITY over
    #   workspace coherence; this is the motor tap pairing one apply() with one
    #   observation. distinct-from: cortex/live_brain.py -- the composed tick; this is
    #   one envelope passage inside it. Sites read; term overlap is the spec citation.
    # distinct-from: substrate/efferent_boundary.py -- declare_ports registers the lane's ports;
    #   this is the loop's passage that feeds it. Read.
    # distinct-from: substrate/envelope_hash_gate.py, substrate/lake_rebuild_gate.py -- gates
    #   over bytes and the index; phrase overlap only. Read.
    # distinct-from: thalamus/reentrant_workspace.py -- the theta authority door; phrase
    #   overlap only ("the ONE statement"). Read.
    def apply_observed(self, command, *, deliberate=False):
        """THE tapped envelope passage (specs/022 T037): every apply this loop issues --
        and the cockpit's direct efferent route -- goes through here so the efference-copy
        boundary observes the passage. So101Arm.apply() remains the ONLY path to motion;
        this adds observation, never gating. The T037 review panel found the first draft
        tapped 2 of step()'s 4 apply sites (the HOME self-clear -- ON in the production
        daemon -- and the postural recovery passed unobserved), so the tap became the one
        mechanism every site calls instead of four hand-copies. The tap is guarded: an
        exception in observation must never break actuation (the E-STOP passage above
        all), so it is swallowed AND COUNTED (efferent.tap_errors -- a counted disclosure,
        not a silent one).

        `deliberate` (T037a, OD-4, the durable latch) is FORWARDED to the gate, not decided
        here: a latch RESTORED from a previous process is cleared only by a deliberate reset --
        the cockpit's ticket or chat passes `deliberate=True` -- and this loop's own automated
        HOME (the self-clear in step(), or a clearing token the policy picked) leaves the
        default and is refused by So101Arm.apply() itself, which counts it
        (`latch_refusals`, on the arm's state). A refused clear is still a passage the
        efference-copy boundary observes (the T037 1:1 step/tick contract holds)."""
        _pre = self.arm.state() or {}
        state = self.arm.apply(command, deliberate=deliberate)  # the gate decides
        try:
            self.efferent.observe(
                command,
                _pre.get("theta_deg"),
                _commanded_deg(self.arm),
                (state or {}).get("theta_deg"),
            )
        except Exception:
            self.efferent.tap_errors += 1  # the passage completed; the tap's failure is counted
        # The reflex's compass: the end effector's height change COINCIDENT with issuing this
        # command, EMA'd over every issuance. The joints are servos that follow a target over
        # several ticks, so one sample is lagged (it carries earlier commands' motion too). ONLY
        # THE ARGMAX IS CONSUMED and only the argmax is trustworthy: against an independent method
        # (from a sunken pose, repeat one token 12x and total the dz -- only shoulder_lift(-6),
        # elbow_flex(-6) and wrist_flex(-6) climb) the argmax lands on a genuinely climbing token
        # on 8 of 8 seeds, while the full RANK ORDER does not survive the lag (seed 3 ranks
        # shoulder_lift(-6) BELOW shoulder_lift(+6)). Nothing may be built on this map's rank
        # order without re-measuring it. The alternative -- scoring the reflex's own firings over
        # its 3-tick window -- was measured worse (2026 -> 1692 cells over 8 seeds): too few
        # samples, in a window the learner's own moves pollute.
        # ONLY WHAT THE REFLEX CAN FIRE, AND ONLY WHEN THE BODY COULD MOVE. Restricted to `_MOTOR`
        # because that is all `_recovery_command` reads, which also bounds the dict at 10 entries
        # and keeps a human's cockpit jog -- issued at an unrelated pose, through this same tapped
        # passage -- out of a reflex's evidence. Skipped while the arm is latched: the gate refuses
        # motion there, so every issuance would fold a dz of 0.0 into the map (the review built a
        # 300-tick E-STOP episode that shrank every entry toward zero and FLIPPED the pick).
        if command in _MOTOR and not (_pre.get("stopped") or (state or {}).get("stopped")):
            try:
                dz = _ee_m(state or {})[2] - _ee_m(_pre)[2]
                prev = self._climb.get(command)
                self._climb[command] = (
                    dz if prev is None else (1.0 - self._int_tau) * prev + self._int_tau * dz
                )
            except Exception:
                # A body that stops reporting ee_mm freezes the compass while the reflex goes on
                # firing whatever it last said. Counted, never inferred (the file's own norm): an
                # empty result must not look like a dead detector.
                self._climb_errors += 1
        return state

    def _recovery_command(self):
        """The postural-recovery token, chosen by the BODY'S OWN measured response (P1): the motor
        token with the highest running dz in `_climb` (the tap's per-issuance statistic, see
        apply_observed); an untried token first (optimism: unknown beats known-bad, and the map
        fills within the first ground episodes). The incumbent fired a
        fixed +shoulder_lift with the comment 'one +A step is the minimal motion that raises the
        end effector' -- true of the retired 2-link body, false on the SO-101, where +shoulder_lift
        points DOWN: its 189 firings on seed 2 moved the effector by at most one micron and not one
        of them cleared the collision, and the floor became an absorbing state on 5 of 8 seeds."""
        if not _MOTOR:  # a body with no rotatable joint has nothing to climb with
            return SAFE_ACTION
        untried = [a for a in _MOTOR if a not in self._climb]
        if untried:
            return untried[0]
        return max(_MOTOR, key=lambda a: self._climb[a])

    def step(self, visual, temperature, audio, distance, reward_fn=None, homeostatic_load=0.0):
        """One autonomous tick. distance is the proximity scalar; reward_fn(unit, action)->[-1,1].
        homeostatic_load (>=0) is the body's interoceptive load for the opt-in interoception path."""
        # 1. SAFETY FIRST: an object at the sensor trips the reflex -> gated EMERGENCY_STOP, no action
        if distance is not None and (not math.isfinite(distance) or distance < CONTACT):
            self.cort.step(1.0)  # danger -> stress up (sharpens future response)
            # A NEGATIVE distance is a MISSING reading, not contact. The ESP32 firmware sends a
            # -1 idle sentinel when nothing is within range (substrate/esp32_encoder_path.py
            # documents it), and -1.0 is finite and below CONTACT -- so before this guard the
            # split below was two-way (NaN = fail-closed, finite = real contact) with no case for
            # a sentinel, and it landed on the wrong side: every idle sample bound a fear
            # association to a context in which nothing was there. The E-STOP above is NOT
            # conditioned on this -- a reading we cannot trust still stops the arm, because
            # "no object" and "sensor said nothing" are not the same claim -- but a fabricated
            # US must never reach the amygdala.
            if math.isfinite(distance) and distance >= 0.0:  # real contact (not a NaN, not idle):
                pat = self._context_pattern(visual, temperature, audio, distance)
                am = self._amygdala_for(pat)
                if am is not None:
                    am.pair(pat)  # US present: bind this context to threat
            # T037: the tapped passage -- a stop is a passage, observed like any other
            estop = self.apply_observed("EMERGENCY_STOP")  # ONLY path to motion -> gate holds
            return {
                "unit": None,
                "action": "EMERGENCY_STOP",
                "reflex": True,
                "workspace": self._workspace_broadcast(
                    None, 0.95
                ),  # threat ignites (annotation only)
                "state": estop,
            }
        # 1b. SELF-CLEAR THE LATCH -- owner-authorised 2026-08-03, and narrow on purpose.
        #
        # Reaching this line MEANS the breach predicate above evaluated False for THIS sample: the
        # distance is finite and at or beyond CONTACT. That is the whole safety argument, and it is
        # made by CONTROL FLOW rather than by re-reading a value that could have gone stale -- a
        # re-check would be a second opinion about the same sample, which is not a second opinion.
        #
        # The authorisation was specific: the brain MAY clear, but ONLY when the breach condition
        # reads clear, and every clear is COUNTED so a self-clearing loop is visible rather than
        # silent. An unconditional clear would let the brain clear WHILE STILL IN BREACH, which turns
        # reflex.py's persistence latch -- a counter that exists precisely so a sustained moderate
        # overload can never be ignored -- into a no-op. That is what a runaway looks like from the
        # inside, and it is why this is gated rather than free.
        #
        # HOME is the sanctioned clear (`so101_arm._CLEARS_LATCH`); it and RETRACT_ALL are the same
        # operation there. HOME's known cost is recorded at :1377 -- it resets theta to [0,0] and
        # measured NET NEGATIVE for GROUND recovery (84.4 -> 80.2 cells over 5 seeds) because it
        # destroys the explored position. That objection is about exploration score, not safety, and
        # it does not transfer: after an emergency stop a KNOWN pose is the point.
        #
        # Without this, a cockpit-tripped E-STOP deadlocks silently: AutonomousArm.step() issues
        # neither HOME nor RETRACT_ALL, so nothing on the live path could ever re-arm the arm.
        # OPT-IN, DEFAULT OFF -- and the default was decided by the codebase, not by preference.
        # Wiring this on by default failed EIGHT E-STOP assertions in this module plus one each in
        # cortex/brain.py and cortex/live_brain.py, all of the form "the E-STOP freeze must hold".
        # Ten independent checks encode "a latched arm stays latched across later ticks" as a design
        # invariant, and every one of them ticks a SAFE context after the stop -- so they were really
        # asserting that the freeze survives regardless of whether the danger is gone.
        #
        # Rewriting ten safety assertions to accommodate a new behaviour is silencing checks at
        # scale, however defensible each rewrite looks on its own. So the capability is opt-in: the
        # gates keep their invariant untouched, and the daemon -- which is where a deadlocked E-STOP
        # actually matters, because a cockpit actuation route can trip one -- turns it on explicitly
        # via learn_daemon.bat, exactly as NORD_TIME_CELLS_RESTORE does for growth.
        if self._estop_self_clear and self.arm.state().get("stopped"):
            # ONLY path to motion -> the gate still holds; tapped (T037). T037a: a latch this
            # process did not trip (restored from the sidecar at boot) is REFUSED here -- only
            # the cockpit's deliberate HOME clears it -- and a refused clear is not a clear.
            st = self.apply_observed("HOME")
            if not (st or {}).get("stopped"):
                self._estop_clears += 1

        # 2. perceive -> grow a unit on a novel multimodal context
        pattern = self._context_pattern(visual, temperature, audio, distance)
        if self.workspace is not None:
            # ONE deviance computation per tick, and it now has a consumer that is always live:
            # perception's own claim on the single workspace slot (_percept_salience). It used to be
            # computed ONLY under `salience_novelty`, whose sole consumer was a separate GWT
            # competitor -- so on a default brain the bus's own drive was a literal while the
            # stimulus-derived signal that should have set it was never computed at all.
            #
            # `self._novelty` stays gated exactly as before, deliberately. It is a TELEMETRY row
            # (live_brain.spike_snapshot -> spikes.midbrain), that row is tagged constant in
            # brain_telemetry.PROVENANCE with a measurement behind the tag, and un-gating it here
            # would also charge `_attend`'s `0.3 * self._novelty` term -- a different wire, in a
            # different lane, that would change what the LEARNER sees. One deviance, one new
            # consumer, no collateral rewiring.
            nov = self._percept_novelty(pattern)
            self._percept_sal = self._percept_salience(nov)
            if self._salience_on:
                self._novelty = nov  # bottom-up salience updates each perceived tick
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
        # Defence in depth: the same allow-list the chat path runs, applied to a token this
        # loop composed itself. A token that reached here by another route -- a bundle, a
        # learned policy, a direct caller -- is refused rather than actuated.
        if action not in _UNCHECKED and not is_valid_command(action):
            raise ValueError(f"autonomous action rejected by the wire allow-list: {action!r}")
        # T037: the tapped passage -- around the envelope, never in it (apply_observed)
        state = self.apply_observed(action)  # gated: frozen after E-STOP, collision-checked
        self._adapt_actuation(state)
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
                # Stand back up WHERE IT FELL (HOME measured net negative for recovery: it destroys
                # the explored position). Which token climbs is the BODY'S call -- see
                # _recovery_command: the fixed +A this site fired until 2026-08-22 pointed into the
                # floor on this body, which is how five of eight seeds ended a run underground.
                state = self.apply_observed(self._recovery_command())  # tapped (T037): a passage
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
            if r_ext < -1.0 or r_ext > 1.0:
                # `reward_fn(unit, action) -> [-1, 1]` is this method's documented contract, and
                # enforcing it here is what lets `_explore_noise` size optimism by `_REWARD_MAX`:
                # the bound is real because nothing can exceed it, rather than assumed because
                # every in-tree caller happens to honour it. Clamped rather than raised -- a
                # caller's arithmetic slip must not take the brain down mid-tick -- and COUNTED,
                # because a silent clamp is a lie about what the objective asked for.
                self._reward_clamps += 1
                r_ext = max(-1.0, min(1.0, r_ext))
            self._paid = self._paid or reward_fn is not None
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
            if qsub is not None and z_lq is not None:
                qsub.update(
                    z_lq, action, r
                )  # SHARED substrate update (linear latent_q OR nonlinear mlp_q) -> transfer
            else:
                self.q[unit][action] += (
                    self.lr * self.cort.learn_rate_scale(r) * (r - self.q[unit][action])
                )
            # STRESS IS AN EVENT THE BODY REPORTS, NOT THE OBJECTIVE'S RUNNING OPINION. A collision
            # is such an event and reads 1.0 -- the same convention as the reflex's own untouched
            # cort.step(1.0). Four candidate feeds were measured against four regime reds (R1 a
            # steady shaping cost must not pin it; R2 sustained real danger must not read calm;
            # R3 an all-positive regime must not read stressed; R4 a constant-bad regime):
            #   raw -r_ext (the incumbent): R1 FAILS -- the coverage objective's steady -0.10
            #     stillness term pins it at the cap (0.3 x 0.10 / (1 - exp(-1/40)) = 1.2 > 1.0) for
            #     400 collision-free ticks, and the 0.6 caution bias on HOLD then IS the HOLD lock;
            #   -err (baseline-relative): R2 and R4 FAIL. `err` is zero-mean by construction, so
            #     max(0, -err) measures reward VARIANCE, not adversity: 500 ticks in the floor at
            #     -1.0 habituated to 0.019 (the brain least cautious where danger is constant)
            #     while an ALL-POSITIVE varying regime read 0.482;
            #   this rule: R1 0.082, R2 1.000, R3 0.000, and 8-seed coverage within 1% of the best.
            # R4 IS A DELIBERATE REFRAMING, NOT A PASS: a task reward that is uniformly bad with no
            # adverse body event now produces no stress (0.19, and that residue is real collisions).
            # Cortisol here biases toward HOLD, amplifies aversive learning and lowers the reflex
            # threshold -- all danger machinery -- and a uniform penalty carries no information
            # about danger or about what to do differently, so manufacturing caution from it is the
            # stand-still pathology in miniature. Physical danger keeps two other channels: the
            # reflex's own cort.step(1.0) and the amygdala's learned threat term.
            self.cort.step(1.0 if (state or {}).get("collision") else 0.0)
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
        # workspace: perception (the current unit, at its DERIVED claim) competes with the
        # amygdala's threat salience -> broadcast.
        #
        # THE `else 0.2` IS GONE. An arm with no amygdala has no threat detector, so the honest
        # threat on that arm is the absence of one, and `_threat()` already returns exactly 0.0
        # there; the 0.2 was a background literal standing in for a faculty that is not present.
        # Removing it removes a constant AND removes the only case where this channel reported a
        # threat nothing had detected.
        #
        # THE CHANNEL ALREADY CARRIES THE REAL AMYGDALA OUTPUT -- and on this brain that output is
        # 0.0, which is a MEASURED NULL and not a literal. Measured over 150 live wake ticks: the
        # amygdala's `w_exc` and `w_inh` are still ALL ZERO at the end of the run, i.e. `pair()` was
        # never called once. The reason is structural and upstream: `pair()` fires only inside the
        # reflex branch of this method (`distance < CONTACT`), and `Brain._safe_ctx()` says in its
        # own docstring that it pins distance SAFE so the E-STOP never trips. So nothing on the
        # live wake path ever presents a US, the amygdala never conditions, and the channel
        # correctly reports no threat. It is NOT structurally incapable of carrying one: driven
        # with real danger contexts the same channel reaches 0.90 and wins the slot (see the
        # threat-channel check in main()). Making this channel "vary" by feeding it interoceptive
        # load or cortisol would be a signal wearing another faculty's name, which is the shape
        # this file is being cleaned of, so it is left honest and the null is reported.
        salience = threat
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
    # T031: this self-check constructs workspace-enabled arms, and every such arm opens the
    # lake `home()` resolves and journals IMMORTAL coalition records into it. A bare
    # `python -m` run inherits the AMBIENT environment -- which is the LIVE lake (measured:
    # one bare run wrote 254 harness records before this guard). setdefault keeps the gate
    # runner's isolated NORD_HOME (run_selfchecks.py sets one for the whole run) and
    # sandboxes only the bare invocation. Inside main() on purpose -- a module-scope env
    # write is the exact learn_daemon defect P4c records.
    import os as _os
    import tempfile as _tempfile

    _os.environ.setdefault("NORD_HOME", _tempfile.mkdtemp(prefix="arm_selfcheck_"))
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
    ee_frozen = guard.arm.state()["ee_mm"]
    guard.step(**a)  # a learned action after E-STOP
    still_frozen = guard.arm.state()["stopped"] and guard.arm.state()["ee_mm"] == ee_frozen

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
    ree = rguard.arm.state()["ee_mm"]
    rguard.step(**a, reward_fn=reward)  # replay-eligible tick after the E-STOP
    replay_frozen = rguard.arm.state()["stopped"] and rguard.arm.state()["ee_mm"] == ree

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
    ws_safe_ign = wsa._ws_percept_ignited  # the ported verdict: percept seized the slot
    ws_danger = wsa.step(**_ctx(0.5, 0.5, 0.5, danger_d))[
        "workspace"
    ]  # danger: threat ignites + broadcasts
    ws_danger_ign = wsa._ws_percept_ignited  # reflex tick submits NO percept -> verdict False
    # The percept-seizure verdict (ported 2026-08-12 from the retired WorkspaceBroadcast) must
    # DISCRIMINATE: True on the safe tick (perception holds the slot -- ws_safe above IS the
    # percept), False on the reflex tick (percept is None there by construction). Reds: invert the
    # verdict expression, or pin it to either constant -- one of the two halves fires.
    assert ws_safe_ign and not ws_danger_ign, (
        f"the percept-seizure verdict must be True on the safe tick and False on the reflex tick "
        f"(got safe={ws_safe_ign}, danger={ws_danger_ign}) -- the ported ignition verdict is dead "
        "or inverted"
    )
    ws_frozen = wsa.arm.state()["stopped"]
    ws_off = AutonomousArm(seed=0).step(**a)[
        "workspace"
    ]  # opt-in: a default arm broadcasts nothing

    print("=" * 62)
    print("AUTONOMOUS ARM -- grows from multimodal data + learns actions, safely")
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

    # 8b. THE SALIENCE REACHING THE SHARED BUS VARIES WITH LIVE STATE (P1 + P6).
    #
    # Until this pass perception submitted the literal 0.6 on every tick: measured over 150 LIVE
    # wake ticks of the canonical brain, the perception channel had distinct=1 / n_changes=0 and
    # the broadcast content never changed once. A bus carrying a constant is a bus carrying no
    # information, so every declared read of it was inert BY CONSTRUCTION -- which is why adding
    # readers to it could never have been progress.
    #
    # ITS REACHABLE RED IS CONSTRUCTED AND RUN (P8), not asserted: the identical stream, the
    # identical seed and the identical scoring, on an arm whose _percept_salience is PINNED to the
    # incumbent's 0.6, must collapse back to one distinct value. A green above a control that
    # cannot go red proves nothing, and this control is the constant this pass removed.
    def _bus_salience_trace(bot, n=120):
        seen = []
        # T031 moved perception onto the typed-channel door (submit_coalition); the spy
        # watches THAT door, reading the salience channel -- spying the 3-arg submit here
        # would count an empty trace and fail the red-control assert vacuously.
        real_submit = bot.workspace.submit_coalition

        def spy(source, content, channels):
            if source == "perception":
                seen.append(round(float(channels["salience"]), 6))
            return real_submit(source, content, channels)

        bot.workspace.submit_coalition = spy
        srng = random.Random(4)  # per-call instance rng -> both arms see the SAME stream
        for _ in range(n):
            bot.step(
                visual=[srng.random() for _ in range(4)],
                temperature=[srng.random()],
                audio=[srng.random() for _ in range(4)],
                distance=safe_d,
                reward_fn=bot.coverage_reward,
            )
        bot.workspace.submit_coalition = real_submit
        return seen

    sal_live = _bus_salience_trace(AutonomousArm(seed=0, workspace=True))
    sal_pin_arm = AutonomousArm(seed=0, workspace=True)
    sal_pin_arm._percept_salience = lambda _dev: 0.6  # the incumbent literal == the RED control
    sal_pinned = _bus_salience_trace(sal_pin_arm)
    sal_n = len(set(sal_live))
    sal_ch = sum(1 for x, y in zip(sal_live, sal_live[1:]) if x != y)
    red_n = len(set(sal_pinned))
    red_ch = sum(1 for x, y in zip(sal_pinned, sal_pinned[1:]) if x != y)
    ws_lo = wsa.workspace.ignition_threshold
    ws_hi = 1.0 - (wsa.workspace.self_boost + wsa.workspace.hysteresis)
    assert red_n == 1 and red_ch == 0, (
        f"the RED control must reproduce the constant this pass removed "
        f"({red_n} distinct, {red_ch} changes) -- if it does not, this harness cannot fail"
    )
    assert sal_n > 1 and sal_ch > 0, (
        f"the perception salience reaching the shared bus must VARY with live state "
        f"({sal_n} distinct, {sal_ch} changes over {len(sal_live)} ticks)"
    )
    assert min(sal_live) >= ws_lo, (
        f"a claim below the bus's own rheobase is not a claim ({min(sal_live)} < {ws_lo})"
    )
    assert max(sal_live) <= ws_hi, (
        f"perception must stay DISPLACEABLE by a maximal competitor ({max(sal_live)} > {ws_hi}) -- "
        "above this bound an incumbent percept cannot be seized from by a proximity threat"
    )

    # P1 demands the new baseline be checked at its degenerate inputs, and this tree has already
    # shipped a mu/1e-12 = 1e12 singularity onto a live path, so they are RUN rather than reasoned.
    deg = AutonomousArm(seed=0, workspace=True)
    deg_single = deg._percept_salience(1.0)  # SINGLE sample: it is its own mean -> exactly mid-band
    deg_zero = deg._percept_salience(0.0)  # ZERO deviance against a live baseline -> rheobase
    sil = AutonomousArm(seed=0, workspace=True)
    deg_silence = [sil._percept_salience(0.0) for _ in range(4)]  # SILENCE: 0 signal AND 0 baseline
    emp = AutonomousArm(seed=0, workspace=True)
    emp._percept_novelty([])  # onset seeds the (empty) expectation
    deg_empty = emp._percept_salience(emp._percept_novelty([]))  # EMPTY pattern -> 0 deviance
    assert deg_single == ws_lo + (ws_hi - ws_lo) * 0.5, (
        f"one sample is its own mean, so the first claim must be mid-band ({deg_single})"
    )
    assert deg_zero == ws_lo and deg_empty == ws_lo, (
        f"ZERO deviance and an EMPTY pattern must return the bare rheobase "
        f"({deg_zero}, {deg_empty})"
    )
    assert all(v == ws_lo for v in deg_silence) and all(math.isfinite(v) for v in deg_silence), (
        f"SILENCE must return the rheobase rather than divide by a vanishing baseline "
        f"({deg_silence})"
    )

    # 8c. THE THREAT CHANNEL IS A MEASURED NULL, NOT A DEAD WIRE.
    #
    # It carries the amygdala's REAL learned output and always did. On the canonical brain that
    # output is 0.0 for a structural, upstream reason: `pair()` fires only inside this method's
    # reflex branch (distance < CONTACT), and `Brain._safe_ctx()` pins distance SAFE so the reflex
    # never trips -- verified by reading the amygdala after 150 live wake ticks, where `w_exc` and
    # `w_inh` were still entirely zero. Nothing on the live wake path presents a US, so there is no
    # threat to report and the channel correctly reports none.
    #
    # Both directions are RUN here so the null is proven to BE a null: conditioned with real danger
    # the SAME channel carries a learned, non-zero threat; a never-endangered arm keeps reporting
    # exactly 0.0 on the same bus.
    thr_ctx_danger = _ctx(0.2, 0.8, 0.3, danger_d)
    thr_ctx_recall = _ctx(0.2, 0.8, 0.3, safe_d)  # same context, no longer dangerous
    thr_seen = []

    def _thr_trace(bot, ticks):
        real_submit = bot.workspace.submit

        def spy(source, content, salience):
            if content == "threat":
                thr_seen.append(round(float(salience), 6))
            return real_submit(source, content, salience)

        bot.workspace.submit = spy
        for ctx in ticks:
            bot.step(**ctx)
        bot.workspace.submit = real_submit

    thr_arm = AutonomousArm(seed=0, workspace=True, amygdala=True)
    _thr_trace(thr_arm, [thr_ctx_danger] * 12)  # each danger tick pairs the context with the US
    thr_arm.arm.apply("HOME")  # the sanctioned latch clear, so a safe tick scores normally
    thr_seen.clear()
    _thr_trace(thr_arm, [thr_ctx_recall])  # SAFE tick, conditioned context -> real learned threat
    thr_learned = thr_seen[-1]
    thr_seen.clear()
    thr_naive = AutonomousArm(seed=0, workspace=True, amygdala=True)
    _thr_trace(thr_naive, [thr_ctx_recall])  # never endangered -> the honest zero
    thr_null = thr_seen[-1]
    assert thr_null == 0.0, (
        f"an amygdala that has never been paired must report NO threat, not a background "
        f"literal ({thr_null})"
    )
    assert thr_learned > 0.0, (
        f"the threat channel must be able to carry a REAL learned threat -- a null that cannot "
        f"become non-zero is a dead wire, not a null ({thr_learned})"
    )
    assert thr_learned == round(thr_arm._threat(thr_arm._context_pattern(**thr_ctx_recall)), 6), (
        "the value on the threat channel must BE the amygdala's own output"
    )
    print(
        f"bus salience: perception submits {sal_n} distinct values / {sal_ch} changes over "
        f"{len(sal_live)} ticks in [{min(sal_live):.4f}, {max(sal_live):.4f}]; the PINNED-0.6 "
        f"control gives {red_n} distinct / {red_ch} changes (the RED is reachable). Threat channel: "
        f"{thr_null:.4f} unconditioned, {thr_learned:.4f} after real danger (a measured null)"
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

    # 10. BASAL GANGLIA Go/NoGo gate wired live as the action gate: it releases the learned
    #     high-value action; with no value drive the GATE adds nothing of its own, and what does
    #     release an untried action is count-based OPTIMISM (T039d -- the old claim here said 'no
    #     spurious motion' and was measured false on the live call shape).
    bg = AutonomousArm(seed=1, bg_gate=True)
    last_bg = None
    for _ in range(90):
        last_bg = bg.step(**a, reward_fn=reward)  # reward RELEASE
    bg_released = bg._bg_select(bg.q[last_bg["unit"]])

    # THE LIVE CALL SHAPE, both arms. The retired probe passed the DEFAULT unit=None, for which
    # `_explore_noise` returns zeros -- so it certified 'HOLD' at any exploration scale (measured:
    # 0.0, 1.0 and 1e6 all passed) and could never go red. Passing a real grown unit is what makes
    # each half below capable of failing, and each half is the other's red.
    def _bg_driveless(explore):
        rel = 0
        for _s in range(40):
            _arm = AutonomousArm(seed=_s, bg_gate=True, explore=explore)
            _arm.step(**a, reward_fn=reward)
            _u = next(iter(_arm.q))
            _arm.q[_u] = dict.fromkeys(ACTIONS, 0.0)  # zero cortical drive
            _arm._visits = {}  # every action untried -> full optimism where optimism is on
            if _arm._bg_select(_arm.q[_u], unit=_u) != SAFE_ACTION:
                rel += 1
        return rel

    bg_null_releases = _bg_driveless(explore=0.0)  # the gate alone
    bg_optimism_releases = _bg_driveless(explore=1.0)  # optimism live
    bg_null = SAFE_ACTION if bg_null_releases == 0 else "(released)"
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
        f"bg gate     : reward -> the gate releases {bg_released}; with NO value drive the gate "
        f"alone releases {bg_null} ({bg_null_releases}/40 non-safe), and optimism releases an "
        f"untried action on {bg_optimism_releases}/40 -- exploration, not spurious motion; "
        f"E-STOP freeze holds with it on ({bg_frozen})"
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
    # The PAIR, on the live call shape. Each half is the other's reachable red: if optimism ever
    # stopped releasing, the second assert fires; if the gate itself started manufacturing drive,
    # the first does. The retired single assert could do neither.
    assert bg_null_releases == 0, (
        f"with optimism OFF the BG gate must add no drive of its own -- released "
        f"{bg_null_releases}/40 non-safe actions at zero cortical drive"
    )
    assert bg_optimism_releases > 0, (
        f"count-based optimism must be able to release an UNTRIED action ({bg_optimism_releases}/40)"
        " -- a gate that refuses without prior value is the cold-start deadlock T038a/T038b fixed"
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


def declare_ports(manifest):
    """Manifest registrant (specs/022 T030, the producer half of the composite's bus):
    the arm's per-tick coalition bids -- perception, threat, and (when enabled) novelty
    submits into the shared GlobalWorkspace this loop OWNS and exclusively clocks
    (_workspace_broadcast is the bus's ONE step() per tick; two clocks would
    double-settle the competition). The edge is declared by the consumer
    (global_workspace). NOTE the safety adjacency, unchanged by declaration: workspace
    output never gates motion directly -- the reflex/E-STOP path preempts before the
    workspace is consulted, and on a reflex tick only the threat bid is submitted."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "autonomous_loop",
        Port(
            name="coalition_out",
            region="brainstem",
            shape=(1,),
            dtype="f4",
            unit="coalition_bid",
            encoding="EVENT",
            kind="OUT",
            declaring_module="autonomous_loop",
            delivery_policy="all-or-none",
        ),
    )
    # T037: the efferent lane's producer -- the chosen action token, observed at every
    # envelope passage by the efference-copy boundary (its subscription edge is declared
    # by the consumer, substrate/efferent_boundary). Symbolic token, EVENT-encoded per
    # the TYPE HONESTY discipline; So101Arm.apply() remains the only path to motion.
    manifest.declare_port(
        "autonomous_loop",
        Port(
            name="action_out",
            region="brainstem",
            shape=(1,),
            dtype="f4",
            unit="action_token",
            encoding="EVENT",
            kind="OUT",
            declaring_module="autonomous_loop",
            delivery_policy="all-or-none",
        ),
    )


if __name__ == "__main__":
    main()
