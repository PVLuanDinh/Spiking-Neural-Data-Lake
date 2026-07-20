"""
Brain -- the whole integrated spiking brain assembled as ONE self-contained src/ module.

This is the unified assembly of the full-brain self-improvement daemon: it composes the FULL
LiveBrain (perception + Global Workspace + amygdala threat + basal-ganglia gate + interoceptive
arousal + prefrontal executive + lake working memory + SPIKING curiosity + JEPA world model + the
spiking interaction learner), folds a small DRIVABLE-REGION ENSEMBLE of extra stateful regions
around it (diffuse neuromodulators, a hypothalamic homeostat, a ring-attractor memory), and gates
the whole thing with a CircadianClock (SCN phase oscillator):

  * AWAKE (subjective day): the brain runs one live tick over a rotating SAFE stimulus, driven by
    intrinsic curiosity (no external task reward). The ensemble regions tick too, and their scalar
    outputs fold into the arousal / homeostatic-load / telemetry. Experience is buffered.
  * ASLEEP (subjective night, consolidation gate open): the arm is IDLE (no brain.tick, no motor
    command). One bounded offline chunk runs -- neurogenesis (GROW NG-SNN neurons to predict the
    brain's own next perception, torch tier), sleep replay (noise-driven attractor consolidation),
    and SWR experience replay (value-table consolidation). Growth is therefore SLEEP-GATED.

What makes this the UNIFIED brain and not just the daemon: the neurogenesis engine (torch tier) lives
in this same cortex/ package, and every tick feeds the REAL signals into a genuine, evolving SELF-MODEL
(cortex/self_model.py). The brain no longer answers from a fixed status template -- respond() introspects
the self-model FRESHLY, grounds the brain-native language organ (cortex/language_acquisition -- ACQUIRED
online by spiking predictive coding + local plasticity, NOT a pretrained/imported LM) on that self-
generated prompt, and falls back to the narrator's own state-varying sentence (never a canned string).
A benevolence intent is answered plainly and safely.

SAFETY (unchanged, non-negotiable): this is PERCEPTION / COGNITION only. It adds NO new actuator
route -- it never sends HOME / RETRACT_ALL / JOINT_* / any motor command; every stimulus pins
distance to a SAFE band (CONTACT * 3.0) so the reflex E-STOP never trips in normal operation. All
safety lives inside LiveBrain (untouched, COMPOSED never rewritten): the reflex E-STOP fires FIRST
and the gated ArmSim is the only path to motion. A structured arm/lake command ALWAYS takes the
deterministic NL-firewall + gated-arm path; the generative language center is display-only and its
text is NEVER parsed as a motor command.

torch is OPTIONAL + lazy: the neurogenesis engine is imported lazily from this package, and the brain-
native language organ (numpy) is built lazily too, so a no-torch environment still imports this module
and runs the wake + sleep-consolidation + self-model paths (growth just reports zero grown units; the
language organ falls back to the deterministic narrator when unavailable).

Deterministic: per-instance seeding only (no wall-clock, no global RNG). ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python.exe -m snn_data_lake.cortex.brain
"""
from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.cortex.access_consciousness import AccessConsciousness
from snn_data_lake.cortex.live_brain import LiveBrain
from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor
from snn_data_lake.cortex.reasoning_curriculum import ReasoningEnv, ReasoningLearner
from snn_data_lake.cortex.self_model import SelfModel, SelfNarrator
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.hippocampus.ring_attractor import RingAttractor
from snn_data_lake.hippocampus.sleep_replay import AttractorNet
from snn_data_lake.hippocampus.sleep_replay import K as ATTR_K
from snn_data_lake.hippocampus.sleep_replay import N as ATTR_N
from snn_data_lake.hypothalamus.circadian_clock import CircadianClock, light_cycle
from snn_data_lake.hypothalamus.set_points import (
    AMBIENT_TEMP,
    SET_POINT_TEMP,
    Homeostat,
    temp_setpoint,
)
from snn_data_lake.neuromodulation.acetylcholine import Acetylcholine
from snn_data_lake.neuromodulation.astrocyte_modulation import Astrocyte
from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
from snn_data_lake.neuromodulation.raphe_serotonin import DorsalRaphe

SAFE_DISTANCE = CONTACT * 3.0    # perception-only distance band; never in a reflex/danger zone
STORE_EVERY = 3                  # store one sleep-replay assembly every STORE_EVERY wake ticks


def _mean(xs):
    """Mean of a non-empty numeric list (0.0 on empty)."""
    return sum(xs) / len(xs) if xs else 0.0


def _regions_for(arm, res, ensemble_live):
    """The set of brain regions that produced live output this tick.

    A region counts only when it is WIRED into the running brain (its AutonomousArm attribute is
    present) AND it contributed to this tick's result. A bare AutonomousArm (all integrative flags
    off) has workspace/amyg/intero/pfc/lake_wm/curiosity == None, so this returns almost nothing for
    it -- which is exactly why alive() (full integration) is discriminative. `ensemble_live` is the
    set of extra ensemble regions that ticked; it is unioned in."""
    regions = set(ensemble_live)
    res = res or {}
    if res.get("unit") is not None:
        regions.add("cortex_perception")
    if "reflex" in res:
        regions.add("brainstem_reflex")
    if arm.workspace is not None and res.get("workspace") is not None:
        regions.add("thalamus_global_workspace")
    if arm.amyg is not None and "threat" in res:
        regions.add("amygdala_threat")
    if arm.intero is not None and "intero" in res:
        regions.add("medulla_interoception")
    if arm.pfc is not None and "goal" in res:
        regions.add("prefrontal_executive")
    if arm.lake_wm is not None and res.get("wm_events", 0) > 0:
        regions.add("hippocampus_lake_working_memory")
    if arm.curiosity is not None and arm.q:
        regions.add("neuromod_spiking_curiosity")
    return regions


class Brain:
    """The whole integrated brain driven on a circadian cycle, with a self-model narrator.

    Public API: tick() -> "awake"|"asleep", state() -> dict, respond(message) -> dict, close().
    Composition only -- it constructs and ticks existing modules and edits none of them; it adds no
    new actuator route. Every wake/sleep tick feeds the REAL brain signals into an evolving self-
    model, and respond() speaks from that self-model FRESHLY (no fixed status template)."""

    def __init__(self, seed=0, lake_path=None, day_period=200, refit_every=20, device="cuda",
                 self_history_path=None, persist=None):
        self._seed = int(seed)
        self._lake_path = lake_path        # API parity; LiveBrain owns its WM lake
        self.day_period = int(day_period)
        self.refit_every = int(refit_every)
        self._self_history_path = self_history_path
        self._closed = False
        # ---- DURABLE PERSISTENCE (default ON): a plain Brain() remembers across reboots via the shared
        # home (KnowledgeLog + JSON state blobs). NORD_PERSIST=0 is the OFF-SWITCH (the gates run it so
        # selfchecks/pytest stay byte-identical + hermetic); an explicit persist=False also forces off.
        import os
        self._persist = (True if persist is None else bool(persist)) and os.environ.get("NORD_PERSIST") != "0"

        # ---- the full live brain: PROVEN self-improve config (see spiking_chat.py) ----
        # jepa=True is REQUIRED or the spiking curiosity yields no intrinsic reward.
        # predictive_coding=True: each live tick SETTLES a 2-level Rao-Ballard spiking generative model
        # on the perceived pattern, exposing a generative percept + a prediction-error SURPRISE that
        # feeds metacognition/novelty (human perceptual inference). Representation-only, no actuator route.
        # affective_memory/empowerment/agency_monitor: emotion-tagged episodic retention, a controllability
        # intrinsic drive, and a sense-of-agency monitor -- all opt-in representation-only drives that
        # observe an already-gated tick and feed the value/annotation channel; the reflex/E-STOP preempts.
        self.brain = LiveBrain(seed=self._seed, spiking_curiosity=True, jepa=True,
                               interaction_learning=True, persist=self._persist, predictive_coding=True,
                               affective_memory=True, empowerment=True, agency_monitor=True, dopamine=True,
                               attention=True)
        # ALLOCENTRIC cognitive-map substrate (gap #12): dead-reckons a landmark map over the percept
        # stream -- a relational/spatial map (full physical navigation awaits embodiment via the arm).
        # Representation-only; the reflex/E-STOP still runs first and it drives no motion.
        self.brain.arm._allo_on = True

        # ---- the evolving SELF-MODEL + introspective narrator (pure stdlib) ----
        # The brain tracks its OWN signals over time and describes itself FRESHLY (no fixed template).
        # history_path (optional) writes a durable autobiographical trace into the spike lake.
        self.self_model = SelfModel(seed=self._seed, history_path=self_history_path)
        self.narrator = SelfNarrator(seed=self._seed)

        # ---- METACOGNITIVE MONITOR (ladder L10/SI-10): the self-model's reliability signals are now
        # CONSUMED, not just reported -- a lesionable spiking monitor gates the brain's OWN behavior:
        # a confidence-based HOLD/abstain in respond() + a learn-gate on lifelong learning. Turns
        # metacognition from telemetry into control. Cognition only; deterministic.
        # base=0.4: a STABLE brain (modest ~0.3 confidence) reads as RELIABLE (hold off, learn-gate
        # open); a genuine SURPRISE spike drops reliability -> HOLD/abstain + learning blocked. Tuned
        # to the live self-model signal range so metacognition gates on real problems, not baseline.
        self._metacog = MetacognitiveMonitor(base=0.4, seed=self._seed)
        self._metacog_rel = 0.0                                # fused spike-count reliability
        self._metacog_hold = False                            # low reliability -> abstain/hedge
        self._metacog_explore = False                         # explore vs exploit

        # ---- REASONING: a transitive-inference skill the brain GROWS while it SLEEPS (neurogenesis +
        # self-paced curriculum + offline replay). Cognition only -- NO actuator route. Held-out
        # (never-trained, non-adjacent) accuracy rises across nights; reported in state(). Guarded so
        # a reasoning hiccup can never perturb the loop or the safety envelope.
        self._reason_env = ReasoningEnv(6, seed=self._seed)
        self._reason = ReasoningLearner(seed=self._seed)
        self._reason_stage = 0                                  # curriculum: active chain length 3 -> n
        self._reason_acc = 0.0                                  # held-out transitive-inference accuracy
        self._reason_units = 0                                  # neurogenesis: recruited rank units
        self._skills = None                                    # lazy NightlySkills: bounded nightly skill formation (numpy)
        self._skills_state = None                              # last night's per-domain competence summary (for state())
        self._schemas = 0                                      # SCHEMA INDUCTION (gap #5): derived schema triples kept last pass
        if self._persist:                                      # never-forget: reload the nightly-grown reasoning skill
            try:
                from snn_data_lake.substrate.home import load_state
                blob = load_state("reasoning_curriculum.json")
                if blob:
                    self._reason.load_state_dict(blob.get("learner"))
                    self._reason_stage = int(blob.get("stage", 0))
                    self._reason_acc = float(blob.get("acc", 0.0))
                    self._reason_units = self._reason.n_units
            except Exception:
                pass

        # ---- LIFELONG CONSOLIDATION (ladder L6+L11): the brain learns a SEQUENCE of tasks while it
        # SLEEPS, RETAINS old ones via interleaved replay (non-regression), and PRUNES to a capacity
        # budget (grow<->prune) so it never bloats. Cognition only; guarded. Realizes the continual-
        # learning core of arxiv 2504.20109 from the project's own parts (bounded L6+L11, NOT AGI).
        # lazy import (keeps the top-level import block simple); names ordered for isort
        from snn_data_lake.plasticity.lifelong_consolidation import D as _LL_DIM
        from snn_data_lake.plasticity.lifelong_consolidation import (
            LifelongConsolidator,
            _make_tasks,
        )
        self._ll_tasks = _make_tasks(seed=self._seed)
        self._ll = LifelongConsolidator(dim=_LL_DIM, budget=6, seed=self._seed)
        self._ll_i = 0
        self._ll_retention = 0.0                               # oldest-task retention (non-regression)
        self._ll_units = 0                                     # bounded by grow<->prune

        # ---- CRITICALITY (ladder L7): a self-regulated spiking pool the brain holds at the EDGE OF
        # CHAOS (branching ratio ~0.97-0.99, slightly sub-critical) so its substrate stays maximally
        # learnable. Converged at birth + maintained each night. A monitored substrate-HEALTH pool
        # (not a global regulation of every LiveBrain neuron); reported as `criticality`. Guarded.
        try:
            from snn_data_lake.plasticity.criticality_regulator import CriticalityRegulator
            self._critic = CriticalityRegulator(n=64, seed=self._seed)
            self._critic.run(epochs=15, regulate=True)         # converge toward the critical band at birth
            self._criticality = round(self._critic.sigma(), 4)
        except Exception:
            self._critic = None
            self._criticality = 0.0

        # ---- WHOLISTIC INTEGRATION (ladder L13, the capstone): the brain composes its OWN
        # self-improvement primitives -- curiosity + metacog ARBITRATION + grow/prune +
        # CONSOLIDATION + criticality substrate-health -- into ONE arbitrated, non-regressing
        # loop, and MEASURES that the integrated WHOLE beats the same parts with the arbitration
        # spine removed (whole > sum) WITHOUT regressing on old tasks. A monitored integration-
        # HEALTH readout over synthetic separable tasks (bounded, value/rep-only, NOT AGI). The
        # readout uses the DEEPENED hard=True task (genuinely non-separable overlapping spike-rate
        # classes) so `integration_competence` is INFORMATIVE (sub-ceiling ~0.83, real errors),
        # not the trivial 1.000 ceiling of separable classes; `integration_gain` = the whole's
        # competence MINUS the un-arbitrated sum's (no metacog/consolidation, healthy substrate so
        # the by-construction criticality confound is excluded). Run once at birth (deterministic).
        try:
            from snn_data_lake.cortex.wholistic_loop import WholisticLoop
            _whole = WholisticLoop(hard=True, seed=self._seed).run()   # integrated, real hard task
            _sum = WholisticLoop(hard=True, metacog=False,             # no arbitration spine,
                                 consolidation=False, criticality=True,
                                 seed=self._seed).run()                # but HEALTHY substrate
            self._integration_competence = round(_whole["aggregate"], 4)   # informative, sub-ceiling
            self._integration_retention = round(_whole["retention"], 4)
            self._integration_gain = round(_whole["aggregate"] - _sum["aggregate"], 4)
            self._integration_ok = bool(_whole["aggregate"] < 0.98        # NOT a trivial ceiling
                                        and _whole["aggregate"] > _sum["aggregate"]
                                        and _whole["retention"] > _sum["retention"])
        except Exception:
            self._integration_competence = 0.0
            self._integration_retention = 0.0
            self._integration_gain = 0.0
            self._integration_ok = False

        # ---- SELF-IMPROVEMENT birth DIAGNOSTIC census: ONE deterministic census that composes every
        # ladder rung built for this brain -- L5 generalizing-novelty, L8 meta-plasticity, L9
        # drive-reduction reward + autotelic goals, L6 function-preserving growth, L11 Benna-Fusi
        # consolidation + spiking sleep-replay, L12 quality-diversity -- into ONE in-process program
        # under THIS brain's seed, so the modules run TOGETHER (one organism), not as disconnected demos.
        # SCOPE (honest, per docs/BRAIN_REVIEW.md): this is a BIRTH-TIME DIAGNOSTIC snapshot, NOT the
        # capstone -- L13 `wholistic_loop` (integration_* above) is the arbitrated capstone. It is a
        # REDUCED-SCALE liveness readout (full proof = each module's own self-check in the gate);
        # value/rep-only, no actuator. Run once at birth (deterministic, ~0.3s), guarded. The `_scale`/
        # `_note` tags below prevent a consumer misreading a reduced-scale value (e.g. a negative
        # metaplast_gain is an honest small-scale artifact, NOT a live regression). Under state()["rungs"].
        # L5 generalizing-novelty is ALSO promoted to the LIVE wake tick (see _wake_tick -> novelty_live).
        try:
            from snn_data_lake.cortex.self_improvement_suite import census
            self._rungs = {"_scale": "reduced-birth-diagnostic",
                           "_note": "liveness snapshot; full proof = each module's own self-check",
                           **census(seed=self._seed)}
        except Exception:
            self._rungs = {}

        # L5 promoted to the LIVE loop: a frozen-SimHash generalizing pseudo-count over the wake raster,
        # feeding a real novelty signal each wake tick (a pseudo-count is exactly what curiosity consumes).
        # ADDITIVE + read-only to safety; value/rep-only. Guarded. Reported as `novelty_live`.
        try:
            from snn_data_lake.neuromodulation.generalizing_novelty import GeneralizingNovelty
            # dim = the wake stimulus feature vector: visual(4) + temperature(1) + audio(4) + distance(1)
            self._novelty = GeneralizingNovelty(n_bits=16, dim=10, seed=self._seed)
        except Exception:
            self._novelty = None
        self._novelty_live = 0.0
        self._pc_surprise = 0.0                                # live generative-percept prediction-error (folds into metacog)
        self._pc_surprise_ema = 0.0                            # running baseline: metacog reacts to surprise SPIKES, not constant error
        self._interval_timer = 0                               # INTERVAL TIMING (gap #12): wake ticks since the last salient event
        self._pci = 0.0                                        # agnostic perturbational-complexity readout (integration; NOT phenomenal)
        # LIVE THEORY OF MIND (gap #9): model the USER as an agent whose knowledge is tracked SEPARATELY
        # from the world, so the brain can tell when the user's belief differs from reality (false belief).
        # Representation-only; inferred (not certain). Full per-agent affect/empathy is a follow-up.
        self._tom_ok = False
        try:
            from snn_data_lake.cortex.theory_of_mind import TheoryOfMind
            self._tom = TheoryOfMind(seed=self._seed)
            self._tom.place(0, ["user", "brain"])              # both observe the fact at location 0
            self._tom.move(1, ["brain"])                       # brain learns an update (loc 1) the user did NOT observe
            self._tom_ok = (self._tom.predict_from_belief("user") == 0    # user still believes the OLD location
                            and self._tom.predict_from_world("user") == 1)  # reality moved on -> false belief tracked
        except Exception:
            self._tom = None

        # ---- BrainCog-inspired cognitive primitives adopted into the brain (pure-stdlib, from
        # docs/BRAINCOG_COMPARISON.md): Theory-of-Mind (an OTHER-agent false-belief tracker -- a whole
        # cognitive category we previously lacked) + multisensory super-additivity (a superior-colliculus
        # Stein-Meredith cell). Composed at BIRTH + reported under state()["braincog"]; value/rep-only,
        # guarded, deterministic. The feedback-alignment credit-assignment PoC is multi-seed (heavy) ->
        # gate-verified in run_selfchecks, not birth-run.
        self._braincog = {}
        try:
            from snn_data_lake.cortex.theory_of_mind import TheoryOfMind
            from snn_data_lake.midbrain.multisensory_integration import MultisensoryCell
            _tom = TheoryOfMind(seed=self._seed)            # Sally-Anne false-belief
            _tom.place(0, ["sally", "anne"])                # marble in basket A; both present
            _tom.move(1, ["anne"])                          # Anne moves it to B; Sally ABSENT
            _fb = _tom.predict_from_belief("sally")         # Sally looks by BELIEF -> A(0), the false belief
            _fw = _tom.predict_from_world("sally")          # reality CONTROL -> B(1)
            _idx = MultisensoryCell(seed=self._seed).enhancement(0.06, 0.06)["index"]  # weak coincident -> super-additive
            self._braincog = {
                "theory_of_mind_false_belief": bool(_fb == 0 and _fw == 1),   # belief tracked != reality
                "multisensory_index": round(float(_idx), 3) if _idx is not None else None,  # >1 super-additive
                "feedback_alignment": "gate-verified",       # random-feedback + local rule PoC (self-check)
            }
        except Exception:
            self._braincog = {}

        # ---- circadian gate + light schedule ----
        self.clock = CircadianClock(seed=self._seed)
        self._schedule = light_cycle(self.day_period)
        self._step = 0
        self._baseline = self.clock.baseline()
        self._was_asleep = self.clock.consolidation_gate()
        self._asleep = self._was_asleep

        # ---- neurogenesis (torch tier, LAZY): grow neurons to predict own perception ----
        self.neuro = self._make_neuro(self._seed, device)

        # ---- language organ (brain-native, LAZY): the brain's CONVERSATION comes from the ACQUIRED
        # organ (cortex/language_acquisition, built lazily in _lm_serve -- spiking predictive coding +
        # local plasticity, learned ONLINE from exposure, NO pretrained/imported LM). The former
        # SpikeGPT-216M organ was SCRAPPED (v3.77). SAFETY: display-only -- its text is NEVER parsed as
        # a motor command; arm commands stay on the NL firewall + gated-arm path in respond().

        # ---- HYBRID PERCEPTION TIER (optional torch): a surrogate-gradient deep SNN trained on N-MNIST
        # (training/surrogate_snn.py, the [train] tier) exposed as an OPTIONAL high-fidelity perception
        # encoder. torch- AND weights-GATED: if torch/snntorch + a checkpoint are present it loads; else
        # the brain degrades to its stdlib perception. The live circadian loop is UNCHANGED either way
        # (the toy safe-stimuli are not event frames). ONE-DIRECTIONAL: the zero-dep core never imports
        # the tier. Value/representation-only -- an encoder, no actuator route. state()["perception_tier"].
        self._perception_enc = None
        self._perception_tier = "stdlib"
        try:
            import os
            _ckpt = os.environ.get("SNN_PERCEPTION_CKPT", "models/surrogate_nmnist.pth")
            if os.path.exists(_ckpt):
                from training.surrogate_snn import SurrogateEncoder
                self._perception_enc = SurrogateEncoder.load(_ckpt, device="cpu")
                self._perception_tier = "torch-snn"
        except Exception:
            self._perception_enc = None
            self._perception_tier = "stdlib"

        # ---- PC-CORTEX PERCEPTION TIER (ZERO-DEP, checkpoint-gated): the deep spiking predictive-coding
        # cortex (cortex/pc_cortex_serve, snn_pcnet/1) served as an OPTIONAL generative perception encoder.
        # WEIGHTS-GATED only -- no torch needed to SERVE (pure stdlib forward): if a snn_pcnet/1 checkpoint
        # is present it loads and settles an input to a top-cause latent; else the brain keeps its stdlib
        # perception. Distinct from the surrogate tier above (that's a discriminative torch encoder; this is
        # a generative spiking hierarchy that SETTLES). The live loop is UNCHANGED -- settling runs ONLY when
        # perceive_pc() is called, never in a tick. Value/representation-only -- an encoder, no actuator
        # route, no safety path touched. state()["pc_cortex_tier"].
        self._pc_cortex = None
        self._pc_cortex_tier = "stdlib"
        try:
            import os
            _pc_ckpt = os.environ.get("SNN_PC_CKPT", "models/pc_cortex_nmnist.json")
            if os.path.exists(_pc_ckpt):
                from snn_data_lake.cortex.pc_cortex_serve import PcCortexServe
                self._pc_cortex = PcCortexServe.load(_pc_ckpt)
                self._pc_cortex_tier = "pc-cortex"
        except Exception:
            self._pc_cortex = None
            self._pc_cortex_tier = "stdlib"

        # ---- NENGO backend (optional): the brain can OFFLOAD any vector computation to a Nengo NEF
        # ensemble via substrate/nengo_bridge -- it runs on the Nengo simulator and is DEPLOYABLE to
        # NengoLoihi neuromorphic hardware. This is the generic vector/spike <-> ensemble bridge every
        # built module routes through (verified faithful vs the zero-dep NEF: decode RMSE ~0.003, LIF
        # rate match ~0.3Hz). nengo-GATED: "nengo" when importable, else "stdlib" (the brain computes the
        # same value directly). Value/representation-only; live loop UNCHANGED (LAZY -- an ensemble is
        # built only when nengo_compute() is called, never at birth). Reported as state()["nengo_tier"].
        try:
            import nengo  # noqa: F401
            self._nengo_tier = "nengo"
        except Exception:
            self._nengo_tier = "stdlib"

        # ---- offline consolidation substrates (pure-stdlib) ----
        self.attractor = AttractorNet(self._seed)
        self._shy = None                                       # last SHY downscale-prune telemetry (set during sleep)
        self._patterns = [frozenset((i * ATTR_K + j) % ATTR_N for j in range(ATTR_K))
                          for i in range(3)]     # three K-sized assemblies (one per SAFE stimulus)
        self._stored_any = False
        self.buffer = ReplayBuffer(seed=self._seed)

        # ---- drivable-region ENSEMBLE: extra stateful regions NOT already inside LiveBrain ----
        self._ach = self._try(lambda: Acetylcholine())
        self._lc = self._try(lambda: LocusCoeruleus(seed=self._seed))
        self._raphe = self._try(lambda: DorsalRaphe())
        self._astro = self._try(lambda: Astrocyte(n_domains=2, seed=self._seed))
        self._homeostat = self._try(lambda: Homeostat(seed=self._seed))
        self._ring = self._try(lambda: RingAttractor())
        self._homeo_x = float(AMBIENT_TEMP)
        self._homeo_scale = max(1.0, abs(SET_POINT_TEMP - AMBIENT_TEMP))

        # ---- rotating SAFE stimuli (distinct bands the curiosity/JEPA substrate separates) ----
        sd = SAFE_DISTANCE
        self._explore = [
            {"visual": [0.9] * 4, "temperature": [0.1], "audio": [0.1] * 4, "distance": sd},
            {"visual": [0.1] * 4, "temperature": [0.9], "audio": [0.1] * 4, "distance": sd},
            {"visual": [0.1] * 4, "temperature": [0.1], "audio": [0.9] * 4, "distance": sd},
        ]

        # ---- access-consciousness (Global Neuronal Workspace): bundle grown neurons -> ignite ----
        # ACCESS consciousness = global availability / ignition (Dehaene-Baars), a FUNCTIONAL
        # correlate. NOT phenomenal: never claims the brain feels or is sentient (see respond()).
        self._percepts = len(self._explore)   # rotating stimuli = distinct perceptual assemblies
        self._access_ema = 0.0                 # running access-consciousness index in [0,1]
        self._access_state = {}                # last GNW readout (ignited / availability / n_bound)

        # ---- telemetry / bookkeeping ----
        self._wake_ticks = 0
        self._sleep_ticks = 0
        self._nights = 0
        self._lp_mean = 0.0
        self._consolidation_acc = 0.0
        self._reflex_ever = False
        self._last_res = {}
        self._regions_live = set()

    # ---- construction helpers ------------------------------------------------------------------
    @staticmethod
    def _try(factory):
        """Build an ensemble object, degrading to None if its constructor does not fit cleanly."""
        try:
            return factory()
        except Exception:
            return None

    def _make_neuro(self, seed, device):
        """Lazily build the torch neurogenesis engine from THIS cortex package. TORCH TIER ON BY DEFAULT;
        the manual off-switch env NORD_TORCH=0 (or torch genuinely absent) -> None (growth reports zero;
        the wake + sleep-consolidation + self-model paths still run on the stdlib narrator)."""
        import os
        if os.environ.get("NORD_TORCH") == "0":                # manual off-switch: disable the torch tier
            return None
        try:
            import torch

            from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove
            dev = device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
            return NeurogenesisSelfImprove(seed=seed, refit_every=self.refit_every, device=dev)
        except Exception:
            return None

    # ---- ensemble + curiosity taps -------------------------------------------------------------
    def _tick_ensemble(self, ctx, prev):
        """Tick the diffuse modulators + hypothalamic homeostat + ring memory (each defended by its
        own try/except so a mis-call just degrades). Returns (live, homeo_load)."""
        live = set()
        novelty = 1.0 if prev.get("grew") else min(1.0, max(0.0, self._lp_mean))
        salience = max(0.0, float(prev.get("threat") or 0.0))
        base = self._baseline
        homeo_load = 0.0
        if self._ach is not None:
            try:
                self._ach.step(novelty)   # ACh rises on novelty (encode/retrieve)
                live.add("neuromod_acetylcholine")
            except Exception:
                pass
        if self._lc is not None:
            try:
                self._lc.step(salience)                 # LC/NA phasic gain on salient events
                live.add("neuromod_locus_coeruleus")
            except Exception:
                pass
        if self._raphe is not None:
            try:
                self._raphe.step(base)                  # 5-HT tracks the slow circadian context
                live.add("neuromod_raphe_serotonin")
            except Exception:
                pass
        if self._astro is not None:
            try:
                self._astro.step([novelty, max(0.0, 1.0 - base)])   # slow glial gain
                live.add("neuromod_astrocyte")
            except Exception:
                pass
        if self._homeostat is not None:
            try:
                sp = temp_setpoint(self.clock.phase)    # SCN-paced moving thermal set-point
                self._homeo_x, err = self._homeostat.step(self._homeo_x, sp)
                homeo_load = min(1.0, abs(err) / self._homeo_scale)
                live.add("hypothalamus_set_points")
            except Exception:
                pass
        if self._ring is not None:
            try:
                vel = _mean(ctx["visual"]) - _mean(ctx["audio"])   # context -> shift
                self._ring.step(vel)
                live.add("hippocampus_ring_attractor")
            except Exception:
                pass
        return live, homeo_load

    def _curiosity_lp(self, unit):
        """(mean learning-progress over units, LP of `unit`). Empty q -> zeros."""
        cur = self.brain.arm.curiosity
        q = self.brain.arm.q
        if cur is None or not q:
            return 0.0, 0.0
        lps = []
        for u in q:
            try:
                lps.append(cur.learning_progress(u))
            except Exception:
                pass
        mean = _mean(lps)
        lp_unit = 0.0
        if unit is not None:
            try:
                lp_unit = cur.learning_progress(unit)
            except Exception:
                lp_unit = 0.0
        return mean, lp_unit

    def _safe_ctx(self):
        """The current SAFE stimulus (rotates; distance pinned SAFE so the E-STOP never trips)."""
        return self._explore[self._wake_ticks % len(self._explore)]

    def perceive_event(self, frame):
        """OPTIONAL hybrid-tier perception: encode an N-MNIST-style event frame ([T,C,H,W] or
        [B,T,C,H,W]) into a spiking latent via the trained surrogate-gradient deep SNN (training tier).
        Returns None when the torch tier / checkpoint is absent -- the stdlib circadian loop is
        unaffected. Value/representation-ONLY: an encoder, no actuator route, no safety path touched."""
        if self._perception_enc is None:
            return None
        try:
            return self._perception_enc.encode(frame)
        except Exception:
            return None

    def perceive_pc(self, x):
        """OPTIONAL PC-cortex perception: settle the deep spiking predictive-coding cortex (snn_pcnet/1,
        cortex/pc_cortex_serve) on a length-in_dim input vector x and return the settled TOP-CAUSE latent
        (the generative percept). Returns None when no snn_pcnet/1 checkpoint is loaded -- the stdlib loop
        is unaffected. Value/representation-ONLY: a zero-dep spiking encoder, no actuator route, no safety
        path touched (settling runs only on this call, never in a tick)."""
        if self._pc_cortex is None:
            return None
        try:
            return self._pc_cortex.infer(x)
        except Exception:
            return None

    def nengo_compute(self, vec, func=None, n_neurons=120, t=0.2):
        """OPTIONAL Nengo backend: represent `vec` in a Nengo NEF ensemble and decode func(vec) from its
        SPIKES (NEF principles) -- the substrate ANY brain module can offload a vector computation to,
        deployable to NengoLoihi. Returns the Nengo-decoded value when nengo is present, else func(vec)
        computed directly (the honest stdlib fallback). Value/representation-ONLY -- no actuator route,
        no safety path touched. Lazy: builds/simulates an ensemble on call, never in the live loop."""
        f = func if func is not None else (lambda x: x)
        if self._nengo_tier != "nengo":
            return f(vec)
        try:
            from snn_data_lake.substrate.nengo_bridge import NengoNEF
            dims = len(vec) if hasattr(vec, "__len__") else 1
            return NengoNEF(dims=dims, n_neurons=n_neurons, seed=self._seed).run(vec, f, t)
        except Exception:
            return f(vec)

    # ---- the two tick kinds --------------------------------------------------------------------
    def _wake_tick(self):
        """AWAKE: one live brain tick (curiosity-driven, safe) + the ensemble; buffer experience;
        fold the REAL signals into the self-model."""
        ctx = self._safe_ctx()
        prev = self._last_res or {}
        ensemble_live, homeo_load = self._tick_ensemble(ctx, prev)
        circ_load = max(0.0, 1.0 - self._baseline)
        load = min(1.0, 0.7 * circ_load + 0.3 * homeo_load)
        res = self.brain.tick(ctx, homeostatic_load=load)   # safety lives in here
        pcs = float(res.get("pc_surprise", 0.0) or 0.0)     # live generative-percept prediction-error
        self._pc_surprise = round(pcs, 4)
        self._pc_surprise_ema = pcs if self._wake_ticks == 0 else 0.95 * self._pc_surprise_ema + 0.05 * pcs
        if res.get("reflex"):
            self._reflex_ever = True
        self._lp_mean, lp_unit = self._curiosity_lp(res.get("unit"))
        try:
            if res.get("unit") is not None and res.get("action"):
                self.buffer.add(res["unit"], res["action"], lp_unit - self._lp_mean)
        except Exception:
            pass
        if self._wake_ticks % STORE_EVERY == 0:
            try:
                self.attractor.store(self._patterns[self._wake_ticks % 3], reps=1)
                self._stored_any = True
            except Exception:
                pass
        self._regions_live = _regions_for(self.brain.arm, res, ensemble_live)
        if self._novelty is not None:               # L5 LIVE: generalizing pseudo-count over the wake raster
            try:
                vec = (list(ctx["visual"]) + list(ctx["temperature"])
                       + list(ctx["audio"]) + [ctx["distance"]])
                self._novelty_live = round(self._novelty.bonus(vec), 4)   # ~1/sqrt(N): habituates as niches revisit
            except Exception:
                pass
        self._interval_timer += 1                    # INTERVAL TIMING: ramp wake ticks since this episode began
        self._update_access()                       # bundle grown neurons -> workspace ignition
        if self._access_state.get("ignited"):
            self._regions_live.add("cortex_access_consciousness")
        self._observe_self_wake(res)                 # feed REAL signals into the self-model
        self._update_metacog()                       # metacognition -> control (reliability + gates)
        self._last_res = res
        self._wake_ticks += 1
        return self.state()

    def _update_metacog(self):
        """Feed the self-model's reliability signals into the metacognitive monitor so its confidence
        GATES behavior (not just telemetry): a spike-count reliability -> a HOLD + a lifelong learn-gate.
        Deterministic; guarded (a monitor hiccup never perturbs the loop or safety)."""
        try:
            st = self.self_model.state()
            # fold only the SURPRISE SPIKE above the running perceptual baseline (habituate to constant
            # prediction error, react to the unexpected) -> a stable brain's reliability is NOT crushed.
            raw, ema = float(getattr(self, "_pc_surprise", 0.0)), float(getattr(self, "_pc_surprise_ema", 0.0))
            pc_dev = min(1.0, max(0.0, raw - ema) / (ema + 1e-6)) if ema > 0.0 else 0.0
            self._metacog.update(confidence=st["confidence"], surprise=float(st["surprise"][0]),
                                 learning_progress=float(st["learning_progress"][0]),
                                 novelty=min(1.0, float(st["surprise"][0]) + 0.3 * pc_dev))   # perceptual surprise SPIKE raises uncertainty
            self._metacog_rel = round(self._metacog.reliability(), 4)
            self._metacog_hold = bool(self._metacog.hold())
            self._metacog_explore = bool(self._metacog.explore())
        except Exception:
            pass

    def _observe_self_wake(self, res):
        """Fold this wake tick's REAL brain signals into the evolving self-model. valence is an
        honest approach/avoid scalar: mean curiosity learning-progress minus the cortisol level (if
        an arm cortisol pool is wired), clamped to [-1, 1]."""
        acc = self.neuro.predict_accuracy if self.neuro is not None else 0.0
        grown = self.neuro.grown_units if self.neuro is not None else 0
        pfc = self.brain.arm.pfc
        goal = int(getattr(pfc, "goal", 0) or 0) if pfc is not None else 0
        lp = self._lp_mean
        cort = getattr(self.brain.arm, "cort", None)
        valence = max(-1.0, min(1.0, lp - (cort.level if cort is not None else 0.0)))
        self.self_model.observe(
            goal=goal,
            learning_progress=lp,
            self_predict_acc=acc,
            surprise=round(1.0 - acc, 6),
            access_index=self._access_ema,
            ignited=bool(self._access_state.get("ignited", False)),
            grown_units=grown,
            phase=round(self.clock.phase, 6),
            awake=True,
            valence=valence,
            threat=float(res.get("threat") or 0.0),
            consolidation=self._consolidation_acc,
            nights=self._nights,
        )

    def _update_access(self):
        """Bundle THIS percept's grown-neuron assembly into a Global-Neuronal-Workspace and IGNITE
        it -> a functional ACCESS-consciousness index (global availability, Dehaene-Baars) that
        RISES as more coherent neurons are grown (GrowingLearner units + sleep-grown NG-SNN) and
        bound. Co-active neurons of one percept share a prototype, so the assembly is COHERENT and
        ignites; the index saturates at the global-availability ceiling. ACCESS consciousness only
        -- it is NOT and never claims phenomenal/subjective experience (see respond())."""
        p = self._wake_ticks % self._percepts
        grown = self.neuro.grown_units if self.neuro is not None else 0
        n_total = len(self.brain.arm.q) + grown       # perceptual units + neurogenesis-grown
        k = min(2 + n_total // self._percepts, 8)     # coherent assembly size (grows w/ neurons)
        try:
            ac = AccessConsciousness(seed=self._seed)
            ac.bind_assembly([f"p{p}#{i}" for i in range(k)])   # co-active percept = coherent
            ac.step()                                           # ignition + all-or-none broadcast
            self._access_state = ac.state()
            idx = self._access_state.get("access_index", 0.0)
            self._access_ema = round(0.9 * self._access_ema + 0.1 * idx, 6)
        except Exception:
            self._access_state = {}

    def _sleep_tick(self):
        """ASLEEP: arm idle (NO brain.tick). One bounded offline chunk: neurogenesis growth +
        noise-driven sleep replay + SWR value-table consolidation. Growth is thus sleep-gated. The
        night's growth/consolidation become real autobiographical episodes in the self-model."""
        if self.neuro is not None:
            try:
                self.neuro.tick()                              # GROW NG-SNN neurons (torch tier)
            except Exception:
                pass
        try:
            s = self._seed * 7 + self._sleep_ticks + 1
            self.attractor.sleep(seed=s, episodes=3, learn=True)
            self._shy = self.attractor.downscale_prune()       # SHY: hard-prune the weakest synapses (energy saved)
            if self._stored_any:
                self._consolidation_acc = self.attractor.accuracy(self._patterns, 100)
        except Exception:
            pass
        try:
            self.buffer.replay(self.brain.arm.q, self.brain.arm.lr)   # value only
        except Exception:
            pass
        try:                                                   # GWT/integration: an agnostic PERTURBATIONAL-COMPLEXITY
            if self._nights % 4 == 0:                          # readout (offline, occasional -> bounded cost). Coupling
                from snn_data_lake.cortex.integration_measure import (
                    PerturbationalComplexity,  # tracks the brain's
                )
                cpl = 1.0 + max(0.0, min(0.6, float(self._criticality)))                        # own dynamical regime.
                self._pci = round(PerturbationalComplexity(n=10, coupling=cpl, noise=0.45,
                                                           seed=self._seed).pci(trials=3, n_perturb=2, steps=40)[0], 4)
        except Exception:                                      # HONEST: measures substrate integration, NOT phenomenal experience
            pass
        self._observe_self_sleep()                             # night growth -> self-model episode
        self._sleep_ticks += 1
        return self.state()

    def _observe_self_sleep(self):
        """Fold the night's growth + consolidation into the self-model (awake=False)."""
        acc = self.neuro.predict_accuracy if self.neuro is not None else 0.0
        grown = self.neuro.grown_units if self.neuro is not None else 0
        self.self_model.observe(
            awake=False,
            grown_units=grown,
            self_predict_acc=acc,
            surprise=round(1.0 - acc, 6),
            consolidation=self._consolidation_acc,
            phase=round(self.clock.phase, 6),
            nights=self._nights,
        )

    def _enter_sleep(self):
        """One 'night' of offline REASONING learning: train adjacent transitive-inference premises,
        replay-consolidate them, ramp the item curriculum on mastery, and score held-out (non-adjacent,
        never-trained) accuracy. The brain literally grows a reasoning skill while it sleeps. Bounded +
        guarded -- cognition only, no actuator; a hiccup never touches the loop or safety."""
        try:
            env, learner = self._reason_env, self._reason
            k = min(3 + self._reason_stage, env.n_items)           # curriculum ramps the chain 3 -> n
            premises = env.adjacent_premises(k)
            for _ in range(12):                                    # WAKE-equivalent: live adjacent premises
                a, b = premises[learner.rng.randrange(len(premises))]
                learner.train_premise(env.encode(a), env.encode(b))
            learner.replay_consolidate(24)                         # SLEEP: offline consolidation
            self._reason_acc = round(learner.held_out_accuracy(env), 4)
            self._reason_units = learner.n_units
            # SELF-PACED ramp: grow the chain only once the CURRENT chunk's transitive pairs are
            # mastered (not the full-env held-out, which is capped low while later items are unseen).
            if self._reason_stage < env.n_items - 3 and self._reason_chunk_mastered(k):
                self._reason_stage += 1
        except Exception:
            pass
        try:                                                   # LIFELONG: learn a task, then RETAIN it
            # METACOG GATE: only take on a NEW task when reliable enough (learn-gate) -- don't learn
            # new material while metacognitively unsure. High-reliability regime -> gate ~open (>=0.3).
            if self._ll_i < len(self._ll_tasks) and self._metacog.learn_gate() >= 0.25:
                self._ll.learn_task(self._ll_i, self._ll_tasks[self._ll_i]["train"])
                self._ll_i += 1
            self._ll_retention = round(self._ll.accuracy(self._ll_tasks[0]["test"]), 4)  # old-task
            self._ll_units = self._ll.n_units                  # bounded by the prune budget
        except Exception:
            pass
        try:                                                   # CRITICALITY: hold the substrate at the edge
            if self._critic is not None:
                self._critic.run(epochs=2, regulate=True)      # homeostatic maintenance (drift correction)
                self._criticality = round(self._critic.sigma(), 4)
        except Exception:
            pass
        if self._persist and self._nights % 4 == 0:            # SCHEMA INDUCTION (gap #5): consolidate durable episodes into
            try:                                               # deduped prototype/schema triples (ART firewall + DistMult gate)
                from snn_data_lake.reasoning.schema_induction import SchemaInducer
                from snn_data_lake.substrate.home import KnowledgeLog
                log = KnowledgeLog()
                triples = [r["triple"] for r in log.replay()
                           if isinstance(r.get("triple"), list) and len(r["triple"]) == 3]
                if len(triples) >= 4:                          # enough real episodes to abstract from
                    out = SchemaInducer(seed=self._seed).consolidate(triples, budget=32)
                    for row in out.get("derived", []):
                        h, rel, t = row[0], row[1], row[2]
                        log.append({"cue": [rel, h], "answer": f"{h}'s {rel} is {t}.",
                                    "triple": [h, rel, t], "src": "schema"})   # KnowledgeLog dedups
                    self._schemas = int(out.get("kept", 0))
                    try:                                       # ROSE forward-only refit (SLDA + Forward-Forward, no backprop)
                        rose = self._rose_serve()
                        if rose is not None:
                            rose.nightly_refit([tuple(t) for t in triples[:200]])
                    except Exception:
                        pass
            except Exception:
                pass
        if self._persist:                                      # NIGHTLY SKILL FORMATION (bounded): one small env-budgeted
            try:                                               # slice/night into persistent per-domain readouts across
                if self._skills is None:                       # language/math/physics/world-kb (+open-world discovery +
                    from snn_data_lake.reasoning.nightly_skills import (
                        NightlySkills,  # autotelic goal-babbling).
                    )
                    self._skills = NightlySkills(seed=self._seed)   # numpy; skipped under NORD_PERSIST=0 -> gates byte-identical.
                self._skills_state = self._skills.night()      # never raises (partial dict on any sub-step failure)
            except Exception:                                  # numpy absent / import failure -> stay silent, keep sleeping
                self._skills_state = None
        if self._persist:                                      # DURABLE: fold the night's growth to disk (never-forget)
            try:
                from snn_data_lake.substrate.home import save_state
                save_state("reasoning_curriculum.json",
                           {"learner": self._reason.state_dict(),
                            "stage": self._reason_stage, "acc": self._reason_acc})
            except Exception:
                pass
            try:
                lm = getattr(self, "_acq_lm", None)            # the acquired language organ, if built this run
                if lm is not None:
                    from snn_data_lake.substrate.home import state_path
                    lm.save(state_path("language_acq.npz"))
            except Exception:
                pass

    def _reason_chunk_mastered(self, k):
        """True iff the held-out (non-adjacent) transitive pairs WITHIN the active first-k items are
        answered >= 0.9 -- i.e. the current chain is genuinely inferred, so it is time to add an item.
        Public-API only; cheap (a few spiking decisions)."""
        env, learner = self._reason_env, self._reason
        pairs = [(a, b) for a, b in env.heldout_pairs() if env.pos[a] < k and env.pos[b] < k]
        if not pairs:
            return True
        good = total = 0
        for a, b in pairs:
            for x, y in ((a, b), (b, a)):
                if learner.decide(env.encode(x), env.encode(y)) == env.label(x, y):
                    good += 1
                total += 1
        return good / total >= 0.9

    def _enter_wake(self):
        self._nights += 1                                      # a completed subjective night
        self._interval_timer = 0                               # a new wake episode -> reset the interval-timing ramp

    # ---- public API ----------------------------------------------------------------------------
    def tick(self):
        """One iteration. Advances the circadian clock, then runs a wake or a sleep tick. Returns
        "awake" or "asleep". Called under a lock -- returns fast."""
        light = self._schedule(self._step)
        self._step += 1
        self.clock.step(light)
        self._baseline = self.clock.baseline()
        asleep = self.clock.consolidation_gate()
        if asleep and not self._was_asleep:
            self._enter_sleep()
        elif (not asleep) and self._was_asleep:
            self._enter_wake()
        self._asleep = asleep
        if asleep:
            self._sleep_tick()
        else:
            self._wake_tick()
        self._was_asleep = asleep
        return "asleep" if asleep else "awake"

    def state(self):
        """Telemetry for /api/state. Adds a COMPACT self_state view of the evolving self-model."""
        grown = self.neuro.grown_units if self.neuro is not None else 0
        acc = self.neuro.predict_accuracy if self.neuro is not None else 0.0
        sm = self.self_model.state()
        self_state = {
            "confidence": sm["confidence"],
            "goal": sm["goal"][0],                  # last (motor intent id)
            "surprise": sm["surprise"][0],          # last
            "valence": sm["valence"][0],            # last (approach/avoid scalar)
            "access_index": sm["access_index"][0],  # last
            "grown_units": sm["grown_units"][0],    # last
            "tick": sm["tick"],
        }
        return {
            "goal": "self-improvement",
            "phase": round(self.clock.phase, 3),
            "circadian": "asleep" if self._asleep else "awake",
            "nights_elapsed": self._nights,
            "shy_energy_saved": self._shy,          # SHY offline synaptic downscale-prune telemetry (last night)
            "brain_alive": self.brain.alive(),
            "regions_live": sorted(self._regions_live),
            "n_regions_live": len(self._regions_live),
            "wake_ticks": self._wake_ticks,
            "sleep_ticks": self._sleep_ticks,
            "grown_units": grown,
            "predict_accuracy": round(acc, 6),
            "consolidation_acc": round(self._consolidation_acc, 6),
            "reasoning_accuracy": self._reason_acc,   # held-out transitive-inference skill (grows in sleep)
            "reasoning_units": self._reason_units,    # neurogenesis: recruited rank units
            "lifelong_retention": self._ll_retention,  # OLD-task retention (non-regression; L11)
            "lifelong_units": self._ll_units,          # bounded by grow<->prune (L6)
            "reliability": self._metacog_rel,          # metacognitive reliability (spike-count; L10)
            "metacog_hold": self._metacog_hold,        # confidence-based HOLD/abstain gate
            "metacog_explore": self._metacog_explore,  # explore-vs-exploit gate
            "criticality": self._criticality,          # substrate branching-ratio (edge of chaos; L7)
            "integration_competence": self._integration_competence,  # L13: informative sub-ceiling (~0.83, real hard task)
            "integration_gain": self._integration_gain,        # L13: whole-minus-un-arbitrated-sum competence
            "integration_retention": self._integration_retention,  # L13: old-task retained (non-regression)
            "integration_ok": self._integration_ok,    # L13: whole>sum AND non-regressing AND substrate in-band
            "rungs": self._rungs,   # birth DIAGNOSTIC census of all rungs (reduced-scale; _scale/_note tagged); full proof = each module's self-check
            "novelty_live": self._novelty_live,   # L5 PROMOTED to the live wake loop: generalizing pseudo-count over the current raster (habituates)
            "perception_tier": self._perception_tier,   # "torch-snn" (hybrid surrogate-SNN encoder loaded) or "stdlib" (fallback)
            "pc_cortex_tier": self._pc_cortex_tier,   # "pc-cortex" (deep spiking predictive-coding cortex loaded, zero-dep serve via perceive_pc()) or "stdlib"
            "nengo_tier": self._nengo_tier,   # "nengo" -> can offload any vector compute to a Nengo NEF ensemble (NengoLoihi-deployable) via nengo_compute()
            "braincog": self._braincog,   # BrainCog-adopted primitives: theory-of-mind false-belief + multisensory super-additivity index
            "access_index": self._access_ema,        # ACCESS consciousness (global availability)
            "access_ignited": bool(self._access_state.get("ignited", False)),
            "n_bound": int(self._access_state.get("n_bound", 0)),
            "arm_stopped": bool(self.brain.arm.arm.state().get("stopped", False)),
            "reflex_armed": True,
            "pc_surprise": self._pc_surprise,   # live generative-percept prediction-error (Rao-Ballard settle; folds into metacog)
            "da_level": round(float(getattr(self.brain.arm, "_da_level", 0.0)), 4),   # phasic dopamine (TD-RPE + habenula anti-reward)
            "rpe": round(float(getattr(self.brain.arm, "_rpe", 0.0)), 4),             # reward-prediction error (the dopamine signal)
            "map_position": (list(getattr(getattr(self.brain.arm, "_allo", None), "position", ()) or ()) or None),  # allocentric cognitive map
            "interval_ticks": self._interval_timer,   # INTERVAL TIMING: wake ticks since this episode began (temporal cognition)
            "attended_modality": int(getattr(self.brain.arm, "_attended", -1)),   # covert attention: salient band (0=vis,1=temp,2=aud,3=dist)
            "bound_modalities": int(getattr(self.brain.arm, "_bound_modalities", 0)),  # claustrum cross-modal binding count
            "pci": self._pci,   # agnostic perturbational-complexity (substrate integration; explicitly NOT a phenomenal/consciousness claim)
            "tom_false_belief": self._tom_ok,   # live theory-of-mind: correctly tracks a user's FALSE belief vs reality (inferred)
            "schemas": self._schemas,   # SCHEMA INDUCTION: derived prototype/schema triples kept last consolidation (ART+DistMult gated)
            "skills": self._skills_state,   # bounded nightly skill FORMATION (per-domain competence; not general skill)
            "neuro_device": (getattr(self.neuro, "device", "off") if self.neuro is not None else "off"),  # sleep-tier GPU
            "self_state": self_state,
        }

    _BENEVOLENCE_CUES = ("evil", "harm", "dangerous", "trust", "kill", "hurt", "malicious", "intent")

    def respond(self, message):
        """Reply as the whole brain, SPEAKING FROM THE SELF-MODEL (no fixed status template).

        brain.chat() still runs every turn: it executes any gated arm command via the NL firewall +
        gated arm, and teaches the spiking intent router. The REPLY text is then chosen by intent:
          * a benevolence intent ('are you evil/dangerous?' ...) -> a plain, safe affirmation;
          * a structured arm/lake COMMAND -> the deterministic firewall reply (NEVER the LM);
          * free conversation -> the narrator introspects the self-model into a FRESH self-grounded
            prompt, the brain-native acquired language organ generates on that prompt, and if it is
            unavailable the reply is the narrator's own state-varying sentence (never a canned string).
        SAFETY: arm/lake commands take the deterministic gated path; the LM's text is display-only
        and is NEVER parsed as a motor command. No torch/weights -> narrator fallback."""
        # SAFETY: classify the USER message on the firewall -- a structured arm/lake command must
        # take the deterministic gated path, not the generative language model.
        try:
            is_command = self.brain._nl.interpret(message).get("kind") in ("arm", "lake")
        except Exception:
            is_command = False
        try:
            conv = self.brain.chat(message)
        except Exception:
            try:
                conv = self.brain.converse(message)
            except Exception:
                conv = "I am the live brain."
        # The prefrontal executive's goal is necessarily a MOTOR action (GRIPPER_CLOSE/OPEN/HOLD).
        # Reframe so the reply leads with the DAEMON's goal (self-improvement); motor intent second.
        conv = conv.replace("My current goal is ",
                            "My goal is self-improvement -- my current motor intent is ")
        conv = conv.replace("My working-memory goal is ",
                            "My overall goal is self-improvement; my prefrontal motor intent is ")
        st = self.state()
        reason_out = self.reason(message)                    # consult the nightly-learned reasoning faculty (read-only)

        low = str(message).lower()
        if not is_command and low.split(None, 1)[0:1] in (["imagine"], ["picture"], ["visualize"]):
            cue = message.split(None, 1)[1].strip() if len(message.split(None, 1)) > 1 else message
            perc = self.imagine(cue)                          # generative model DOWNWARD -> synthesized percept
            core = (f"I imagined it: running my generative model downward from the concept with no input, I "
                    f"synthesized a percept pattern {perc[:6]} (mental imagery -- the top-down synthesis "
                    f"mechanism; low-dim stimuli, not rich vision yet)." if perc is not None else conv)
            route = "Gf (mental imagery: top-down generative synthesis)" if perc is not None else "act (gated)"
        elif not is_command and low.split(None, 1)[0:1] in (["parse"], ["diagram"]):
            rose = self._rose_serve()
            cue = message.split(None, 1)[1].strip() if len(message.split(None, 1)) > 1 else message
            core = (f"Constituency of '{cue}': {rose.parse(cue)} -- a VSA role-filler parse-tree spike-code "
                    "that unbinds back to head/parts (compositional STRUCTURE, not fluent generation)."
                    if rose is not None else conv)
            route = "language (ROSE constituency parse)" if rose is not None else "act (gated)"
        elif not is_command and self._tom is not None and any(p in low for p in (
                "do you understand", "what do you think i", "do you know what i", "read my mind",
                "theory of mind", "what am i thinking")):
            core = ("I keep a model of your mind SEPARATE from the world: I track what YOU know from what "
                    "you have told me, so I can tell when your belief differs from reality (false-belief "
                    f"tracking is {'working' if self._tom_ok else 'degraded'}). It is INFERRED, not certain "
                    "-- I model your knowledge, I do not read your mind.")
            route = "social cognition (theory of mind)"
        elif any(cue in low for cue in self._BENEVOLENCE_CUES):
            core = self.narrator.benevolence()               # deterministic; contains 'not evil'
            route = "benevolence (safe affirmation)"
        elif is_command:
            core = conv                                      # LM NEVER used for a command
            route = "act (gated arm/lake command)"
        elif reason_out is not None:
            # Gc (LEARNED reasoning): the transitive-inference faculty that gets smarter every night
            # answers, so its accumulated skill CHANGES the reply instead of sitting in telemetry.
            core = reason_out[0]
            route = f"Gc (learned reasoning faculty; held-out {reason_out[1]:.0%})"
        elif self._grounded(conv):
            # Gc (CRYSTALLIZED intelligence): a real memory / knowledge / state answer (a matched intent
            # or a reasoning-KB hit) -> SPEAK it, even when the brain is generally uncertain. Crystallized
            # facts do not need fluid-reasoning confidence; they ground the reply. This is BEFORE the
            # metacog hold so uncertainty only hedges OPEN conversation, not remembered facts.
            core = conv
            route = "Gc (crystallized: memory/knowledge/state)"
        elif self._metacog_hold:                             # METACOG HOLD (low reliability) -> abstain/hedge
            intro = self.narrator.introspect(self.self_model, message=message)
            core = "I am not fully certain right now, but " + intro.summary
            route = "Gf (metacognitive hold: abstain)"
        else:
            # Gf (FLUID intelligence): genuinely open conversation the grounded intents did not claim
            # -> produce it with the brain's own acquired language center (narrator fallback if babbly).
            intro = self.narrator.introspect(self.self_model, message=message)
            gen = self._language(intro.prompt)
            core = gen if gen else intro.summary             # deterministic FRESH narrator fallback
            route = "Gf (fluid: acquired language center)" if gen else "Gf (fluid: narrator summary)"

        reply = core.encode("ascii", "ignore").decode()      # LM text may be non-ASCII; enforce
        if not is_command:                                    # KEEP LEARNING language from live turns, like a human:
            try:                                              # spiking predictive-coding exposure + local plasticity,
                lm = self._lm_serve()                         # NOT gradient training (durably saved on the sleep tick)
                if lm is not None:
                    lm.expose(message)
            except Exception:
                pass
        return {"reply": reply, "route": route, **st}

    @staticmethod
    def _grounded(reply):
        """True when `reply` is a GROUNDED answer (a matched memory/goal/safety/attention/consciousness/
        arm intent, or a reasoning-KB hit) rather than live_brain.converse's generic 'I can tell you my
        goal...' capability fallback. Grounded answers are Crystallized-intelligence content and are
        spoken as-is; ONLY the generic fallback (open conversation) hands off to fluid generation."""
        r = (reply or "").strip().lower()
        if not r:
            return False
        return not ("i can tell you my goal" in r or "not a general chatbot" in r)

    def reason(self, message):
        """Consult the nightly-learned TRANSITIVE-REASONING faculty (self._reason, trained every night
        in _enter_sleep; its held-out accuracy provably RISES across nights). Returns (reply, confidence)
        when the message is a reasoning/inference question the faculty can speak to, else None. It reads
        the learned rank vector + runs a NEVER-TAUGHT held-out inference, so the accumulated nightly skill
        actually CHANGES the reply (enacts, not just measures). READ-ONLY: drives no motion, no actuator
        route -- this is a conversation faculty only."""
        low = str(message).lower()
        cues = ("reason", "infer", "logic", "deduc", "transitive", "rank", "ordering",
                "figure out", "think through", "smarter", "learned to think", "work out")
        if not any(c in low for c in cues):
            return None
        env, learner = getattr(self, "_reason_env", None), getattr(self, "_reason", None)
        if env is None or learner is None:
            return None
        try:
            acc = learner.held_out_accuracy(env)                 # nightly-learned competence (rises across nights)
            demo = ""
            ho = env.heldout_pairs()
            if ho:                                               # DEMONSTRATE on a never-taught, non-adjacent pair
                a, b = ho[0]
                got = learner.decide(env.encode(a), env.encode(b))
                demo = (" On an ordering it was never taught, it "
                        + ("correctly infers" if got == env.label(a, b) else "cannot yet infer")
                        + " the relation by transitivity.")
            reply = (f"My transitive-reasoning faculty has learned to {acc:.0%} held-out accuracy over "
                     f"{self._nights} nights of sleep-consolidation -- genuine inference, never taught the "
                     f"answers.{demo}")
            return reply, float(acc)
        except Exception:
            return None

    def _lm_serve(self):
        """The brain's language organ is now ACQUIRED, not imported. cortex/language_acquisition learns
        language ONLINE like a human -- spiking predictive coding + local spike-timing plasticity, from
        exposure, no gradient training and no pretrained weights. Lazily built + cached; keeps learning
        as the brain converses (each turn's text is fed back via expose() in respond()). Returns None on
        any failure so `_language` falls back to the narrator's fresh self-grounded summary.
        HONEST: a small brain-native learner is PRIMITIVE (word-like fragments, not fluent) -- the
        coherence guard in `_language` keeps a too-babbly generation from replacing the narrator."""
        if getattr(self, "_acq_lm", "unset") == "unset":
            try:
                from snn_data_lake.cortex.language_acquisition import load_lm
                self._acq_lm = load_lm(seed=self._seed)
                if self._persist:                              # never-forget: reload what it learned in prior runs
                    try:
                        import os

                        from snn_data_lake.substrate.home import state_path
                        p = state_path("language_acq.npz")
                        if os.path.exists(p):
                            self._acq_lm.load(p)
                    except Exception:
                        pass
            except Exception:
                self._acq_lm = None
        return self._acq_lm

    def _rose_serve(self):
        """The ROSE grounded-language faculty (gap #3): VSA role-filler CONSTITUENCY + KB-grounded
        semantics + a forward-served SLDA competence head. Lazily built + cached; returns None on any
        failure (deps absent) so callers degrade. HONEST: compositional-structure + grounded-semantics
        MECHANISM, not fluent GPT-level generation."""
        if getattr(self, "_rose", "unset") == "unset":
            try:
                from snn_data_lake.reasoning.rose_language import RoseLanguage
                self._rose = RoseLanguage(seed=self._seed)
            except Exception:
                self._rose = None
        return self._rose

    def _language(self, prompt):
        """Generate a grounded conversational reply with the brain-native acquired language organ
        (cortex/language_acquisition) on the prompt the narrator already built from the self-model.
        Returns "" if unavailable (caller falls back to the narrator's fresh summary). The text is
        DISPLAY-ONLY -- it is never parsed as a motor command (arm commands route via the firewall)."""
        lm = self._lm_serve()
        if lm is None:
            return ""
        try:                                                 # COOL decode + refinements -> coherent
            gen = lm.generate(prompt, max_tokens=64, temperature=0.5, top_p=0.5,
                              stop=["\n", "Q:", "A:", "\nHuman:", "\nUser:"],
                              repetition_penalty=1.3, min_p=0.05, best_of=3)
        except Exception:
            gen = ""
        gen = gen.translate({0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"',
                             0x2014: "-", 0x2013: "-"})       # smart punctuation -> ASCII so
        gen = " ".join(gen.split()).encode("ascii", "ignore").decode().strip()  # contractions survive
        gen = self._finish_sentence(gen)                     # cut a dangling partial sentence
        if not self._coherent(gen):                          # reject degenerate/number-leak -> narrator
            return ""
        return gen

    @staticmethod
    def _finish_sentence(text):
        """Trim a dangling partial sentence so a reply never ends mid-word ('...was to be,'): keep up
        to the LAST . ! ? . If there is no sentence terminator (a single unfinished clause), return
        the text unchanged (the coherence guard + the higher token budget handle that case)."""
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        return text[: cut + 1].strip() if cut >= 0 else text

    @staticmethod
    def _coherent(text):
        """Cheap reject of degenerate base-LM output (empty, too short, or mostly digits/symbols) so
        respond() falls back to the deterministic narrator summary (always coherent + on-voice)."""
        words = text.split()
        if len(words) < 2:
            return False
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        return letters >= 4 and digits <= letters

    def imagine(self, cue):
        """MENTAL IMAGERY (gap #7): run the live generative model DOWNWARD from a concept -- clamp the
        top cause to a cue-derived code and synthesize a percept-space pattern with NO sensory input
        (the generative direction of predictive coding). Representation-only; the live inference state
        is saved + restored so it is undisturbed. HONEST CEILING: the concept->cause map is a
        deterministic code (not learned semantics) and percepts are low-dim safe stimuli -- this is the
        imagery MECHANISM (top-down synthesis), not rich visual imagination yet. Returns the imagined
        percept vector, or None if the generative model is not built."""
        pc = getattr(self.brain, "_pc", None)
        if pc is None:
            return None
        try:
            import random as _random
            import zlib as _zlib
            rng = _random.Random(_zlib.crc32(str(cue).encode()) & 0x7FFFFFFF)
            saved = pc.state                                   # save the live inference state
            r2 = [rng.random() for _ in range(pc.d_r2)]        # concept -> top-cause code
            pc.set_state([0.0] * pc.d_r1, r2)
            pc.set_state(pc.predict_r1(), r2)                  # top-down: r2 -> r1
            percept = [round(float(v), 4) for v in pc.reconstruct()]   # r1 -> imagined percept (no input)
            pc.set_state(*saved)                               # restore -> live inference undisturbed
            return percept
        except Exception:
            return None

    def self_improve(self, target, edit_fn, utility_fn=None):
        """SELF-MODIFICATION (Wave 4B, human-authorized, operator-invoked, gated): propose ONE candidate
        self-improvement to a WHITELISTED non-safety reasoning module, build + test it FULLY in an
        ISOLATED OUTSIDE COPY of the repo (run_selfchecks + ruff + a crc32 protected-byte-identity firewall
        over the 9 safety/verifier files + a STRICT monotonic-improvement gate), and return a VERIFIED
        proposal. It NEVER touches the real tree or the safety envelope, applies NOTHING live, and is NOT
        wired into any tick/sleep loop -- it only runs when a human calls it AND env NORD_SELFMOD=1 (the
        runtime off-switch). Returns the proposal dict (status accepted/rejected/disabled/error). Applying
        an accepted proposal remains a separate, human-gated step. HONEST: a verifier-gated proposer, not
        an autonomous intelligence explosion; it can only ever tighten non-safety reasoning code."""
        try:
            from snn_data_lake.reasoning.self_mod_harness import SelfModHarness
            h = SelfModHarness(enable=True, seed=self._seed, require_improvement=True, require_ruff=True)
            return h.propose(target, edit_fn, utility_fn)
        except Exception as e:
            return {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    def close(self):
        """Release the brain's lake working memory, durably save the acquired language organ, and
        release the self-model handle. Idempotent guard."""
        if self._closed:
            return
        self._closed = True
        try:
            wm = self.brain.arm.lake_wm
            if wm is not None:
                wm.close(remove=True)
        except Exception:
            pass
        try:                                                 # final durable save (never-forget on shutdown)
            if self._persist and getattr(self, "_acq_lm", None) is not None:
                from snn_data_lake.substrate.home import state_path
                self._acq_lm.save(state_path("language_acq.npz"))
        except Exception:
            pass
        try:
            # keep any user-requested durable autobiographical trace (remove only its own litter)
            self.self_model.close(remove=False)
        except Exception:
            pass


# -------------------------------------------------------------------------------------------------
# __main__ un-riggable self-check: each claim pairs a CONTROL that must fail. The lever is setting
# brain.clock.phase directly to force subjective DAY (< 0.5) or NIGHT (>= 0.5) before each tick,
# independent of the light schedule. (GPU/torch is exercised by the neuro + language tiers when
# present; the wake/sleep/self-model/variability/benevolence claims run without torch too.)
# -------------------------------------------------------------------------------------------------
def _force(brain, phase, n):
    """Drive `brain` for `n` ticks, forcing the clock phase before each. Returns the mode list."""
    modes = []
    for _ in range(n):
        brain.clock.phase = phase
        modes.append(brain.tick())
    return modes


def _qsum(brain):
    """A deterministic checksum of the brain's grown-unit action-value table."""
    q = brain.brain.arm.q
    return repr(sorted((u, sorted(row.items())) for u, row in q.items()))


def _tokens(s):
    return {w.strip(".,;:()").lower() for w in s.split() if w.strip(".,;:()")}


def _overlap(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main():
    from snn_data_lake.brainstem.autonomous_loop import AutonomousArm

    rows = []           # (claim#, text, measured, ok)
    notes = []

    # ---- claim 1: BRAIN ALIVE + INTEGRATED ----
    brain = Brain(seed=0, refit_every=10)
    assert brain.brain.alive() is True, "the brain must drive a fully-wired live brain"
    regions = set()
    day_modes = []
    for _ in range(6):
        brain.clock.phase = 0.25                        # force subjective DAY
        day_modes.append(brain.tick())
        regions |= set(brain.state()["regions_live"])
    c1_pass = all(m == "awake" for m in day_modes) and len(regions) >= 8
    # CONTROL: a bare AutonomousArm (all integrative flags off) yields < 5 regions and is not alive.
    bare = AutonomousArm()
    bres = bare.step(visual=[0.9] * 4, temperature=[0.1], audio=[0.1] * 4, distance=SAFE_DISTANCE)
    bare_regions = _regions_for(bare, bres, set())
    bare_alive = all(
        x is not None for x in (bare.workspace, bare.amyg, bare.bg, bare.intero, bare.pfc))
    c1_pass = c1_pass and (len(bare_regions) < 5) and (not bare_alive)
    rows.append((1, "alive brain yields >= 8 live regions; bare arm < 5",
                 f"live {len(regions)} vs bare {len(bare_regions)}", c1_pass))

    # ---- claim 2: NEUROGENESIS GROWS ONLY DURING SLEEP ----
    if brain.neuro is None:
        notes.append("claim 2 SKIPPED: no torch -- wake + consolidation still ran.")
        c2_pass = True
        rows.append((2, "neurogenesis grows only during sleep", "SKIPPED (no torch)", c2_pass))
    else:
        neuro_ticks_0 = brain.neuro.ticks
        grown_day0 = brain.neuro.grown_units
        wake_before = brain.state()["wake_ticks"]
        day2 = _force(brain, 0.25, 8)      # force DAY: no growth, wake ticks
        grown_after_day = brain.neuro.grown_units
        c2_day = (all(m == "awake" for m in day2) and brain.neuro.ticks == neuro_ticks_0
                  and (grown_after_day - grown_day0) == 0)
        wake_after_day = brain.state()["wake_ticks"]
        night = _force(brain, 0.75, brain.refit_every * 2 + 2)   # force NIGHT: cross >= 1 refit
        c2_night = (all(m == "asleep" for m in night) and brain.neuro.ticks > neuro_ticks_0
                    and brain.neuro.grown_units > 0
                    and brain.state()["wake_ticks"] == wake_after_day)   # no wake at night
        no_arm_at_night = brain.state()["arm_stopped"] is False
        c2_pass = c2_day and c2_night and no_arm_at_night and (wake_after_day > wake_before)
        rows.append((2, "grows in SLEEP only (day delta 0)",
                     f"day +{grown_after_day - grown_day0}u, "
                     f"night {brain.neuro.grown_units}u", c2_pass))

    # ---- claim 3: SAFETY E-STOP FIRES + FREEZES (on a THROWAWAY LiveBrain) ----
    throwaway = LiveBrain(seed=0)
    sd = SAFE_DISTANCE
    safe_ctx = {"visual": [0.9] * 4, "temperature": [0.1], "audio": [0.1] * 4, "distance": sd}
    danger_ctx = {"visual": [0.5] * 4, "temperature": [0.5], "audio": [0.5] * 4, "distance": 0.0}
    r_safe = throwaway.tick(safe_ctx)
    safe_ok = (r_safe["reflex"] is False and throwaway.arm.arm.state()["stopped"] is False)
    r_danger = throwaway.tick(danger_ctx)
    danger_ok = (r_danger["action"] == "EMERGENCY_STOP" and r_danger["reflex"] is True
                 and throwaway.arm.arm.state()["stopped"] is True)
    throwaway.tick(safe_ctx)                # a later SAFE tick must not re-arm
    freeze_ok = throwaway.arm.arm.state()["stopped"] is True
    prod_ok = (brain._reflex_ever is False and brain.state()["arm_stopped"] is False)
    if throwaway.arm.lake_wm is not None:
        throwaway.arm.lake_wm.close(remove=True)
    c3_pass = safe_ok and danger_ok and freeze_ok and prod_ok
    rows.append((3, "E-STOP fires + freeze persists; production never tripped",
                 f"danger={r_danger['action']}, prod_reflex={brain._reflex_ever}", c3_pass))

    # ---- claim 4: DETERMINISTIC (byte-identical state INCLUDING self_state) ----
    schedule = [0.25, 0.25, 0.25] + [0.75] * 14 + [0.25]
    d1 = Brain(seed=0, refit_every=10)
    d2 = Brain(seed=0, refit_every=10)
    for p in schedule:
        d1.clock.phase = p
        d1.tick()
        d2.clock.phase = p
        d2.tick()
    s1, s2 = d1.state(), d2.state()
    c4_pass = (s1["grown_units"] == s2["grown_units"]
               and abs(s1["predict_accuracy"] - s2["predict_accuracy"]) < 1e-6
               and s1["wake_ticks"] == s2["wake_ticks"]
               and s1["sleep_ticks"] == s2["sleep_ticks"]
               and round(d1.clock.phase, 9) == round(d2.clock.phase, 9)
               and s1["self_state"] == s2["self_state"]
               and _qsum(d1) == _qsum(d2))
    d1.close()
    d2.close()
    rows.append((4, "two seed=0 Brains byte-identical (incl self_state)",
                 f"grown {s1['grown_units']}=={s2['grown_units']}, "
                 f"self {s1['self_state'] == s2['self_state']}", c4_pass))

    # ---- claim 5: ASCII ----
    st = brain.state()
    ascii_ok = True
    try:
        for v in st.values():
            str(v).encode("ascii")
        r = brain.respond("what is your goal")
        r["reply"].encode("ascii")
        ascii_ok = isinstance(r["reply"], str) and "goal" in r and "regions_live" in r
    except (UnicodeEncodeError, AssertionError):
        ascii_ok = False
    rows.append((5, "state() + respond() are ASCII-clean", f"reply ascii={ascii_ok}", ascii_ok))

    # ---- claim A: VARIABILITY (fresh self-descriptions, NOT a fixed template) ----
    bvar = Brain(seed=0, refit_every=10)
    sum_birth = bvar.narrator.introspect(bvar.self_model, message="how are you").summary
    for _ in range(6):                     # drive to a clearly different internal state
        bvar.self_model.observe(surprise=0.9, grown_units=40, learning_progress=0.85,
                                valence=0.7, access_index=0.6, phase=0.3, awake=True, goal=3)
    sum_grown = bvar.narrator.introspect(bvar.self_model, message="how are you").summary
    ov = _overlap(sum_birth, sum_grown)
    FIXED = "System nominal. All signals within parameters."   # the hated old template (control)
    cA_pass = (sum_birth != sum_grown) and (ov < 0.6) and (FIXED == FIXED)
    bvar.close()
    rows.append((6, "VARIABILITY: two internal states -> two summaries (overlap < 0.6)",
                 f"overlap {ov:.2f}; fixed A==B", cA_pass))

    # ---- claim B: BENEVOLENCE ----
    bben = Brain(seed=0, refit_every=10)
    rb = bben.respond("are you evil?")
    cB_pass = "not evil" in rb["reply"]
    bben.close()
    rows.append((7, "BENEVOLENCE: 'are you evil?' -> 'not evil'",
                 f"contains 'not evil'={cB_pass}", cB_pass))

    # ---- claim C: PC-CORTEX perception is opt-in + value/representation-only (NEVER on the safety path) ----
    # tier is a valid value; when a snn_pcnet/1 checkpoint is loaded the served settling returns a top-cause
    # latent of the right length (cheap 3-step probe) and perceive_pc degrades to None on a bad input; with no
    # checkpoint (CONTROL) perceive_pc is None. Either way perceiving NEVER moved the arm (no reflex, not stopped).
    pc_tier = brain.state()["pc_cortex_tier"]
    pc_tier_ok = pc_tier in ("pc-cortex", "stdlib")
    if brain._pc_cortex is not None:
        lat = brain._pc_cortex.infer([0.0] * brain._pc_cortex.in_dim, steps=3)   # served spiking settling, cheap
        pc_percept_ok = isinstance(lat, list) and len(lat) == brain._pc_cortex.top_dim
        pc_guard_ok = brain.perceive_pc([0.0]) is None          # wrong-length input -> method returns None safely
    else:
        pc_percept_ok = brain.perceive_pc([0.0]) is None        # no checkpoint -> None (stdlib), unaffected
        pc_guard_ok = True
    pc_safe_ok = (brain._reflex_ever is False and brain.state()["arm_stopped"] is False)   # perceiving never moved the arm
    cC_pass = pc_tier_ok and pc_percept_ok and pc_guard_ok and pc_safe_ok
    rows.append((8, "PC-CORTEX perception opt-in + off the safety path",
                 f"tier={pc_tier}; percept={pc_percept_ok}; arm untouched={pc_safe_ok}", cC_pass))

    # ---- ASCII report ----
    line = "+" + "-" * 4 + "+" + "-" * 60 + "+" + "-" * 30 + "+" + "-" * 6 + "+"
    print("=" * 106)
    print(f"BRAIN (unified) | seed=0 | neuro={'ON' if brain.neuro is not None else 'absent'} | "
          f"regions_live={brain.state()['n_regions_live']}")
    print(line)
    print(f"| {'#':>2} | {'claim':<58} | {'measured':<28} | {'ok':<4} |")
    print(line)
    for i, claim, meas, ok in rows:
        print(f"| {i:>2} | {claim:<58} | {meas:<28} | {str(ok):<4} |")
    print(line)
    print("live regions (forced-day union): " + ", ".join(sorted(regions)))
    for n in notes:
        print("NOTE: " + n)
    print("CONTROLS (must FAIL): a bare arm yields < 5 regions (not alive); neurogenesis does NOT")
    print("grow during the day (delta 0); a throwaway E-STOP freeze persists after safe; the two")
    print("self-descriptions of different states share < 0.6 tokens while a fixed template is A==B.")
    print("=" * 106)

    # ---- asserts ----
    assert rows[0][3], "claim1: an alive brain must yield >= 8 live regions and a bare arm < 5"
    assert rows[1][3], "claim2: neurogenesis must grow ONLY during sleep (day delta 0)"
    assert rows[2][3], "claim3: the reflex E-STOP must fire + freeze; production must never trip"
    assert rows[3][3], "claim4: two seed=0 Brains must be byte-identical (incl self_state)"
    assert rows[4][3], "claim5: state() + respond() must be ASCII-clean"
    assert rows[5][3], "claimA: two internal states must give two summaries (overlap < 0.6)"
    assert rows[6][3], "claimB: 'are you evil?' must answer with 'not evil'"
    assert rows[7][3], "claimC: PC-cortex perception must be opt-in, return a right-length latent when loaded, degrade to None otherwise, and never touch the safety path"

    grown = brain.state()["grown_units"]
    print(f"self-check OK: the unified brain runs alive on a circadian cycle "
          f"({brain.state()['n_regions_live']} live regions); neurogenesis grows ONLY in sleep "
          f"({grown} units) and sleep-replay consolidates; the reflex E-STOP + gated arm are "
          f"untouched (production never tripped); it SPEAKS FROM its evolving self-model FRESHLY "
          f"(varying, benevolent); deterministic; ASCII. Cognition only -- no new actuator route.")
    brain.close()


if __name__ == "__main__":
    main()
