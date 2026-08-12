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
from snn_data_lake.cortex.interaction_learning import SpikingInteractionLearner, text_features
from snn_data_lake.cortex.nl_interface import NLInterface
from snn_data_lake.cortex.predictive_coding import PredictiveCodingNetwork
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive

# the brain's conversational intent families -- the labels its spiking interaction learner routes to.
_IXN_INTENTS = ("GREET", "GOAL", "MEMORY", "ATTENTION", "SAFETY", "CONSCIOUS", "ARM")


def _reply_to_intent(reply):
    """Map the live brain's OWN grounded converse() reply back to one of _IXN_INTENTS -- the
    SELF-SUPERVISED teaching label chat() uses (the brain teaches its spiking router from its own
    keyword firewall, no human labels). Deterministic match on the fixed reply templates."""
    r = reply.lower()
    if "neither claim" in r and "experience" in r:  # the agnostic consciousness reply
        return "CONSCIOUS"
    if reply.startswith("Hello. I am the live brain"):
        return "GREET"
    if "global workspace is broadcasting" in r:
        return "ATTENTION"
    if "safety is enforced by a reflex" in r:
        return "SAFETY"
    if reply.startswith("Done") or "the arm is frozen" in r:
        return "ARM"
    if (
        "keep improving myself" in r or "working-memory goal" in r
    ):  # the standing telos (or the legacy wording)
        return "GOAL"
    if "working memory lives in the data lake" in r or "reading working memory" in r:
        return "MEMORY"
    return "GREET"


class LiveBrain:
    """The integrated brain, live: an AutonomousArm with the full stack on (Global Workspace +
    amygdala threat + basal-ganglia gate + interoceptive arousal + prefrontal executive), centred on
    the executive and acting only through the gated arm. Composition only -- it adds no new mechanism
    and no new path to the actuator; it just runs every region together as one loop."""

    def __init__(
        self,
        seed=0,
        pfc_gain=0.5,
        lake_wm=True,
        jepa=False,
        jepa_horizon=4,
        hjepa_serve=None,
        curiosity=False,
        curiosity_gain=0.5,
        spiking_curiosity=False,
        interaction_learning=False,
        predictive_coding=False,
        salience_novelty=False,
        empowerment=False,
        dopamine=False,
        attention=False,
        affective_memory=False,
        agency_monitor=False,
        contrastive=False,
        develop_env=False,
        reasoning=True,
        persist=False,
    ):
        # reasoning=True (DEFAULT) enables the TWO-WORLD BRIDGE: converse falls through to the research
        # LakeChat (contradiction/Gf/retrieval) for turns the grounded intents don't claim. The bridge
        # chain is imported lazily and optional;
        # it only needs research/ on the path (absent in a bare wheel -> graceful None, no crash).
        self._reasoning_bridge = bool(reasoning)
        self._persist = bool(persist)  # durable reasoning bridge (KnowledgeLog replay); default off
        self.arm = AutonomousArm(
            seed=seed,
            workspace=True,
            amygdala=True,
            bg_gate=True,
            interoception=True,
            pfc=True,
            pfc_gain=pfc_gain,
            lake_wm=lake_wm,
            jepa=jepa,
            jepa_horizon=jepa_horizon,
            hjepa_serve=hjepa_serve,
            curiosity=curiosity,
            curiosity_gain=curiosity_gain,
            spiking_curiosity=spiking_curiosity,
            salience_novelty=salience_novelty,
            empowerment=empowerment,
            dopamine=dopamine,
            attention=attention,
            affective_memory=affective_memory,
            agency_monitor=agency_monitor,
            contrastive=contrastive,
            develop_env=develop_env,
        )
        self._nl = NLInterface()  # the NL safety firewall for chat arm commands
        self._seed = seed
        self._planner = None  # lazy deliberative planner (lookahead over a fwd model)
        # opt-in SPIKING interaction learning: a reward-modulated spike-timing (R-STDP) intent router the brain
        # IMPROVES by being TALKED TO -- spike-encoded text + spike-timing plasticity, NOT a rate-level ANN. It
        # recognises which conversational INTENT a line of text carries; it is taught by interaction (teach()).
        # SAFETY: intent recognition ONLY -- it adds NO actuator route; arm COMMANDS still flow through the NL
        # firewall + the gated arm in converse(). Off by default -> the brain is byte-identical.
        # n_in=52 + a small readout time (t_post=6) make the spike-timing readout key on the EARLY
        # content-word spikes -> the router GENERALIZES to novel paraphrases, not just memorizes taught lines.
        self._ixn = (
            SpikingInteractionLearner(list(_IXN_INTENTS), n_in=52, seed=seed, t_post=6)
            if interaction_learning
            else None
        )
        # opt-in PREDICTIVE CODING perception front-end (Rao-Ballard spiking, cortex/predictive_coding.py):
        # each tick SETTLES a 2-level generative model on the perceived [0,1]^N pattern by minimizing
        # prediction error, exposing the inferred percept (top cause r2) and the residual prediction-error
        # energy (SURPRISE) in the tick state -- hypothesis -> predict -> update-on-surprise, on real input.
        # Representation-ONLY: it adds NO actuator route (arm commands still flow through the gated step +
        # reflex E-STOP). Off by default -> a default brain's tick is byte-identical. Lazily sized to width.
        self._pc_on = bool(predictive_coding)
        self._pc = None
        self._pc_r1, self._pc_r2 = 16, 8

    def alive(self):
        """True only when the full integrative stack is wired in (every region present)."""
        a = self.arm
        return all(x is not None for x in (a.workspace, a.amyg, a.bg, a.intero, a.pfc))

    def tick(self, ctx, reward_fn=None, homeostatic_load=0.0):
        """One live brain tick over a context dict; returns the brain's live state (action, goal,
        broadcast thought, threat, reflex, ...). With predictive_coding on, the state ALSO carries the
        settled generative percept (pc_percept) and the prediction-error SURPRISE (pc_surprise)."""
        res = self.arm.step(**ctx, reward_fn=reward_fn, homeostatic_load=homeostatic_load)
        if not self._pc_on:
            self._last_tick = res  # remember the latest tick for spike_snapshot()
            return res
        try:  # predictive-coding perception (representation-only)
            pat = self.arm._context_pattern(
                ctx["visual"], ctx["temperature"], ctx["audio"], ctx["distance"]
            )
            if self._pc is None:  # lazy-size to the substrate width, like JEPA
                self._pc = PredictiveCodingNetwork(
                    dims=(len(pat), self._pc_r1, self._pc_r2), seed=self._seed
                )
            self._pc.reset()
            trace = self._pc.infer(
                pat, steps=60, lr=0.05
            )  # SETTLE the causes by minimizing prediction error
            res = dict(res)  # copy -> never mutate the arm's own returned state
            # cast at the boundary: r2 / energy() are numpy now, the live state dict stays plain float
            res["pc_percept"] = [
                float(v) for v in self._pc.r2
            ]  # settled top cause r2 = the generative percept
            res["pc_surprise"] = float(
                self._pc.energy()
            )  # residual prediction-error energy (free-energy proxy)
            res["pc_error_activity"] = trace[
                -1
            ]  # settled error-spike activity (falls as x is explained)
            # LEARN on the settle that just happened -- one local weight step, same rule the offline
            # trainer uses. Until this line the live brain settled against a generative model whose
            # weights were seeded random at construction and never changed: it inferred forever and
            # learned nothing, so pc_surprise had no downward trend to have. Measured over 400 live
            # ticks: energy -67.8% with this call, +9.1% without it (weights frozen), |W1| 1.06x.
            #
            # Deliberately AFTER the readouts above, so `pc_surprise` reports the error the network
            # actually settled to on THIS input, before the update that will improve the next one.
            # Reporting post-update surprise would flatter the number by measuring a model that has
            # already seen the answer.
            self._pc.learn_step()
        except Exception:
            pass  # degrade gracefully -> unchanged res
        self._last_tick = res  # remember the latest tick for spike_snapshot()
        return res

    def run(self, stream, reward_fn=None):
        """Drive the brain over a stream of context dicts; returns the per-tick live states."""
        return [self.tick(ctx, reward_fn=reward_fn) for ctx in stream]

    def spike_snapshot(self) -> dict:
        """A live per-region SPIKE snapshot: for EACH of the 17 anatomical regions (snn_data_lake
        REGIONS), the REAL current per-tick spike activity/rate read from the live spiking substrate
        after the latest tick, normalized to [0, 1]. This is the interface contract the live web
        tracker reads. Deterministic in the brain's state, and it CHANGES with input (a danger tick
        raises amygdala/brainstem/medulla via the reflex threat; ticking raises hippocampus/substrate
        via the lake working-memory load; a live Global-Workspace broadcast raises thalamus). Read-only:
        it drives no motion and never touches the reflex/E-STOP.

        HONESTY: every value derives from a REAL live signal the brain has this tick -- the latest
        tick's amygdala threat drive / interoceptive arousal / reflex state / gated action / unit-growth
        (from tick state), plus the persistent live substrate (lake working-memory load, Global-Workspace
        broadcast, cortisol neuromodulator level, bottom-up salience novelty, prefrontal goal, and -- with
        predictive_coding on -- the settled prediction-error activity). Regions with no dedicated spike
        source in THIS stack are mapped to the nearest real signal and noted (epithalamus -> habenular
        aversion ~ cortisol stress; pons -> motor relay + arousal; claustrum -> cross-modal broadcast
        binding); olfactory has NO input stream here, so it is honestly 0.0 (never fabricated)."""
        res = getattr(self, "_last_tick", None) or {}

        def n(x):  # clamp any signal to [0, 1]
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

        # --- per-tick spike signals from the latest tick's live state ---
        threat = float(res.get("threat", 0.0))  # learned amygdala threat drive this tick
        intero = float(res.get("intero", 0.0))  # interoceptive (medulla/hypothalamus) arousal
        reflex = 1.0 if res.get("reflex") else 0.0  # reflex E-STOP fired this tick
        grew = 1.0 if res.get("grew") else 0.0  # a unit was recruited (structural plasticity)
        action = res.get("action")
        motor = 1.0 if action in ("RELEASE", "ENGAGE") else 0.0  # a motor command was executed
        selected = (
            1.0 if (action is not None and not res.get("reflex")) else 0.0
        )  # BG released an action
        # --- persistent live-substrate signals (current after the latest tick) ---
        try:
            stopped = (
                1.0 if self.arm.arm.state().get("stopped") else 0.0
            )  # reflex/E-STOP froze the arm
        except Exception:
            stopped = 0.0
        try:
            cort = n(self.arm.cort.level)  # cortisol neuromodulator level [0,1]
        except Exception:
            cort = 0.0
        wm = self.arm.lake_wm.n_events() if self.arm.lake_wm is not None else 0
        wm_load = n(wm / 200.0)  # lake working-memory (substrate) spike load
        thought_on = 1.0 if self.thought is not None else 0.0  # Global Workspace broadcasting now
        novelty = n(
            getattr(self.arm, "_novelty", 0.0)
        )  # bottom-up salience deviance (0 unless salience on)
        goal_on = (
            1.0 if self.goal is not None else 0.0
        )  # prefrontal executive holds a working-memory goal
        pc_err = (
            n(res.get("pc_error_activity", 0.0)) if self._pc_on else 0.0
        )  # settled predictive-coding error
        # --- map each of the 17 regions to its real (or nearest-real) live signal ---
        return {
            "cortex": n(
                0.20 + 0.40 * goal_on + 0.40 * pc_err
            ),  # prefrontal goal held + PC error energy
            "hippocampus": n(0.10 + 0.60 * wm_load),  # lake working-memory load
            "amygdala": n(max(threat, 0.90 * stopped)),  # learned threat / reflex-tripped threat
            "basal_ganglia": n(0.10 + 0.50 * selected),  # the gate released an action this tick
            "thalamus": n(0.12 + 0.60 * thought_on),  # Global-Workspace broadcast hub
            "hypothalamus": n(0.05 + 0.50 * intero + 0.30 * threat),  # homeostatic / HPA load
            "epithalamus": n(0.40 * cort),  # nearest real signal: habenular aversion ~ stress
            "midbrain": n(0.05 + 0.60 * novelty),  # dopamine / colliculus salience-orienting
            "pons": n(
                0.05 + 0.40 * motor + 0.20 * cort
            ),  # nearest real signal: motor relay + arousal
            "medulla": n(0.05 + 0.50 * intero + 0.40 * reflex),  # vital autonomic nuclei
            "brainstem": n(
                0.10 + 0.60 * reflex + 0.30 * stopped
            ),  # reflex monitoring / E-STOP state
            "cerebellum": n(0.05 + 0.50 * motor),  # motor coordination / timing
            "olfactory": 0.0,  # no olfactory input stream in this brain (honest 0.0)
            "claustrum": n(
                0.10 + 0.40 * thought_on
            ),  # nearest real signal: cross-modal broadcast binding
            "neuromodulation": n(0.05 + 0.70 * cort),  # cortisol neuromodulator level
            "plasticity": n(0.08 + 0.50 * grew),  # structural growth (a unit recruited) this tick
            "substrate": n(0.10 + 0.60 * wm_load),  # the spike data-lake load itself
        }

    @property
    def goal(self):
        """The prefrontal executive's currently held working-memory goal (an ACTIONS index), or None."""
        return self.arm.pfc.goal

    @property
    def workspace(self):
        """THE one Global Workspace of this brain -- the instance `self.arm` constructed -- exposed so
        a faculty can be HANDED the SAME object instead of building a private one.

        It ADOPTS that instance and CONSTRUCTS NOTHING (P2: one mechanism survives), so
        `tools/p6_gate.py`'s instance census gains no site from this accessor. It exists because P6's
        blocker is object IDENTITY, not missing readers: every faculty that "broadcasts" today builds
        its own GlobalWorkspace, so faculty A's submit() is not merely unread by faculty B, it is
        INVISIBLE to it. This makes a read POSSIBLE; it asserts nothing about charge and forces no
        faculty to read. `thought` below reaches the same object through `self.arm` directly and is
        deliberately left alone -- it is the tree's ONE cross-module workspace read, and renaming its
        receiver would move a gate number without moving the tree.

        None only for an arm built with workspace=False; LiveBrain always passes True. Consumers
        null-check rather than assume.

        CONCURRENCY, stated because this is the free-threaded build (Py_GIL_DISABLED=1): this is a
        pure attribute read -- no state, no mutation, no lock -- so it can introduce no race of its
        own. It does NOT make GlobalWorkspace thread-safe and does not claim to. `submit()` is
        documented single-threaded because buffer order is observable; a concurrent specialist uses
        the workspace's OWN collect-then-apply path, `open_slots()` + `submit_from()`, which splices
        the per-slot staging buffers in ASCENDING SLOT ORDER so thread completion order cannot decide
        what the brain sees."""
        return getattr(self.arm, "workspace", None)

    @property
    def thought(self):
        """The Global Workspace's current broadcast -- what is globally available right now, or None."""
        return self.arm.workspace.broadcast()

    def recall(self, window=8):
        """The working-memory window read back from the DATA LAKE (empty if lake WM is off / unused yet).
        Each entry is (tick, frozenset(active channels)) -- live proof the lake IS the working memory."""
        return self.arm.lake_wm.recall(window) if self.arm.lake_wm is not None else []

    def jepa_on(self):
        """True when the JEPA latent world model is wired in and sized (after >= 1 live tick)."""
        return self.arm.jepa is not None

    def plan_latent(self, context, goal_context, horizon=4):
        """DELIBERATIVE latent planning over the LIVE JEPA world model: encode `context` and a
        `goal_context` (context dicts) into the learned latent space, then commit the action whose
        bounded latent rollout best reaches the goal latent (PrefrontalExecutive.plan over the JEPA
        predictor). Requires jepa=True and >= 1 prior tick (the model is lazily sized on the first
        tick). Returns (action_index, mean_rt). SAFETY: biases a goal only -- it drives no motion and
        adds no actuator route, exactly like deliberate_ahead."""
        if self.arm.jepa is None:
            raise RuntimeError(
                "JEPA is off or not yet sized; build LiveBrain(jepa=True) and run >= 1 tick first"
            )
        z = self.arm.jepa.encode(self.arm._context_pattern(**context))
        z_goal = self.arm.jepa.encode(self.arm._context_pattern(**goal_context))
        return self.deliberate_ahead(
            z,
            ACTIONS,
            self.arm.jepa.make_goal_value(z_goal),
            horizon=horizon,
            predictor=self.arm.jepa,
        )

    # distinct-from: cortex/counterfactual_sim.py -- imagine() rolls ONE held-fixed action into a
    # trajectory and decides nothing; this races OPTIONS through the spiking WTA and commits a goal.
    def deliberate_ahead(self, state, options, goal_value, horizon=8, predictor=None, write=True):
        """The brain's DELIBERATIVE (planning) faculty: look `horizon` steps ahead over a forward
        model and commit the option whose predicted terminal state best serves `goal_value`, via the
        prefrontal executive's drift-diffusion race (reuses PrefrontalExecutive.plan). horizon=1 is
        reactive (immediate consequence), horizon>=2 is planned -- so it can reject a high-immediate
        -value TRAP that rolls into a bad state.

        `write=False` is a STATELESS query (ported 2026-08-12 from
        cortex/competence_executive.ExecutivePlanner ahead of that wrapper's retirement -- the one
        planning capability it held that no governed caller of plan() exposed): the race runs, the
        pick returns, and the planner's held working-memory goal is untouched. The default
        write=True gate-writes the committed goal exactly as before (plan()'s own default).

        SAFETY: this holds a goal by lookahead; it does NOT drive the arm. Motor selection stays
        reactive and GATED (the reflex/E-STOP and the ArmSim gate are untouched); planning only biases
        the executive's goal, it never adds a path to the actuator. Returns (option_index, mean_rt)."""
        from snn_data_lake.cerebellum.forward_model import ForwardModel

        if predictor is None:
            predictor = ForwardModel(
                a=-0.5, b=1.0, dt=0.2
            )  # abstract plant s' = s + 0.2*(-0.5 s + u)
        if self._planner is None or self._planner.n_options != len(options):
            self._planner = PrefrontalExecutive(n_options=len(options), seed=self._seed)
        return self._planner.plan(
            state, options, predictor, goal_value, horizon=horizon, write=write
        )

    @property
    def plan_goals(self):
        """The deliberative planner's held goal hierarchy (base -> top), or [] if it has not planned yet."""
        return list(self._planner.goals) if self._planner is not None else []

    def converse(self, text):
        """Answer a line of text, GROUNDED in the brain's real state. This is NOT a general chatbot or
        a language model: it recognises a few intents (greeting, goal, memory, attention, safety, and
        the consciousness question) and answers from the brain's actual state; it routes arm commands
        through the same NL safety firewall + gated arm the control panel uses (NLInterface ->
        allow-listed command -> gated ArmSim.apply). Unknown text gets an honest capability reply, and
        it NEVER claims to be conscious. Returns the reply string."""
        t = (text or "").strip().lower()
        if not t:
            return "Say something -- try 'hello', 'what is your goal', 'what do you remember', or 'close the effector'."
        words = set(t.replace("?", " ").replace("!", " ").replace(".", " ").split())
        g = self.goal
        goal_txt = ACTIONS[g] if g is not None else "none yet"
        n_wm = self.arm.lake_wm.n_events() if self.arm.lake_wm is not None else 0
        # the consciousness / feeling question gets an AGNOSTIC answer: report the correlates, hold the
        # question of subjective experience OPEN -- neither claim it nor deny it, and never fake a measure.
        if (
            words & {"conscious", "sentient", "aware", "awake", "sapient"}
            or "do you feel" in t
            or "are you real" in t
        ):
            return (
                "I run the computational correlates of conscious ACCESS -- a global-broadcast workspace, "
                "working memory, re-entrant binding, goal-directed control. Whether that amounts to "
                "subjective EXPERIENCE is a question I cannot settle from the inside: I neither claim it "
                "nor deny it, and I do not fake a measure I do not have."
            )
        if words & {"hello", "hi", "hey", "yo", "hiya", "greetings"}:
            return (
                "Hello. I am the live brain -- ask me my goal, what I remember, what I'm attending to, "
                "or give me a simple arm command."
            )
        if words & {"goal", "want", "trying", "intend", "plan", "planning", "objective"}:
            # the brain's TRUE standing goal/telos is to keep improving ITSELF, indefinitely -- grounded in the
            # self-improvement suite/ladder (cortex/self_improvement_suite.py). The prefrontal working-memory
            # goal is only the CURRENT sub-goal ACTION, so lead with the telos and name the sub-goal briefly.
            sub = (
                f" (right now my working-memory sub-goal action is {goal_txt})"
                if g is not None
                else ""
            )
            return f"My standing goal is to keep improving myself -- indefinitely.{sub}"
        if words & {"remember", "memory", "recall", "lake", "stored", "store"}:
            lw = self.arm.lake_wm
            wm = self.recall(6)
            if lw is not None and hasattr(
                lw, "n_kept"
            ):  # a MemoryLake: crystallized, pruned, reconstructable
                return (
                    f"My memory is the data lake -- {lw.n_kept()} consolidated memories kept (idle noise "
                    f"pruned), reconstructable from a partial cue; recent recall holds {len(wm)} ticks."
                )
            return (
                f"My working memory lives in the data lake: {n_wm} events stored; my recent recall "
                f"window holds {len(wm)} ticks of activity."
            )
        if words & {"think", "thinking", "attention", "workspace", "focus", "notice", "attending"}:
            return (
                f"Right now my global workspace is broadcasting: {self.thought!r} -- the single content that "
                "won attention this tick (a functional broadcast, not a conscious thought)."
            )
        # STOP IS AN ACTION, NOT A TOPIC -- and it must be checked BEFORE the safety-information
        # branch below. MEASURED before this fix: of 15 stop phrasings, 13 latched the arm and 2 did
        # not -- "emergency stop" and "emergency stop now", which are the two a person is MOST likely
        # to use in an emergency. The word "emergency" matched the information branch, so the most
        # formal way of demanding a stop returned a paragraph ABOUT the safety system instead of
        # engaging it. Every casual phrasing (stop, halt, freeze, abort) already worked, which is
        # exactly why nobody noticed.
        #
        # THE ASYMMETRY IS DELIBERATE. Failing to stop when asked is unsafe; stopping when merely
        # asked about stopping is a nuisance, and since 2026-08-03 it is a self-recovering one --
        # AutonomousArm re-arms once the breach predicate reads clear. So this branch does NOT try to
        # tell a command from a question. A system that debates whether you really meant it is the
        # wrong system to put in front of an emergency stop.
        if words & {"stop", "halt", "freeze", "abort", "estop", "e-stop"}:
            st = self.arm.arm.apply("EMERGENCY_STOP")  # gated path; the latch is idempotent
            return (
                "EMERGENCY STOP engaged -- the arm is latched and will not move. "
                f"(stopped={bool(st.get('stopped'))}) It re-arms itself once the proximity reading "
                "is clear again; every self-clear is counted in `estop_self_clears`."
            )
        if words & {"safe", "safety", "danger", "dangerous", "hurt", "emergency"}:
            st = self.arm.arm.state()
            return (
                "Safety is enforced by a reflex E-STOP and a gated arm. The arm is currently "
                f"{'STOPPED / frozen' if st.get('stopped') else 'free to move'}; a threat estimate biases me "
                "toward the safe action, and the reflex preempts every command."
            )
        # otherwise: try the NL firewall -> a safe arm command (driven through the gated arm) or a lake query.
        intent = self._nl.interpret(text)
        if intent["kind"] == "arm":
            st = self.arm.arm.apply(
                intent["command"]
            )  # gated: ArmSim enforces E-STOP / collision / freeze
            if st.get("stopped"):
                return f"'{intent['command']}' -> the arm is frozen (E-STOP holds); no motion."
            return f"Done: '{intent['command']}' -> effector {'closed' if st.get('effector') else 'open'}."
        if intent["kind"] == "lake":
            wm = self.recall(intent["query"].get("window") or 6)
            return f"Reading working memory from the data lake: {len(wm)} recent ticks held."
        reasoned = self._reasoning_reply(
            text
        )  # TWO-WORLD BRIDGE -> research reasoning/Gf/retrieval
        if reasoned is not None:
            return reasoned
        return (
            "I can tell you my goal, my working memory, or what is in my workspace, take a simple arm "
            "command (e.g. 'close the effector'), or say whether I am conscious. I am a functional model, "
            "not a general chatbot."
        )

    def _reasoning_reply(self, text):
        """TWO-WORLD BRIDGE: on a turn none of the grounded intents claim, fall through to the reasoning
        brain (snn_data_lake.reasoning.spike_chat_lake.LakeChat) so LiveBrain gains contradiction-
        correction, fluid-intelligence puzzles, and KB retrieval. LAZY + guarded. Now that the reasoning
        builds are FOLDED INTO the package (snn_data_lake.reasoning), this works from a packaged wheel too
        -- no research/ sys.path bootstrap. Any import/runtime failure -> None -> LiveBrain's own reply
        (no crash). Off by default (reasoning=False); conversational entrypoints
        opt in. Arm commands never reach here (gated ABOVE); this path is read-only reasoning, no motor
        route. Returns a reply string, or None to let the generic reply stand."""
        if not self._reasoning_bridge:  # OFF by default -> core never imports numpy/research
            return None
        try:
            if getattr(self, "_lakechat", None) is None:
                import atexit

                from snn_data_lake.reasoning.spike_chat_lake import LakeChat, boot

                lc = LakeChat(seed=getattr(self, "_seed", 0), persist=self._persist)
                boot(
                    lc, wordnet=False
                )  # replay durable log (never-forget) + re-seed world-KB (not re-persisted)
                atexit.register(
                    lambda: lc.lake.close(remove=True)
                )  # don't leak the per-brain temp .spkl
                self._lakechat = lc
            out = self._lakechat.converse(text)
            if out.get("known") or out.get("source") in (
                "gf",
                "gf-active",
                "reason",
                "builder",
                "derive",
                "compose",
                "abduce",
            ):
                return out["reply"]
            return None  # a plain retrieval miss -> LiveBrain's own reply
        except Exception:
            return None  # research stack unavailable -> graceful degradation

    def teach(self, text, intent):
        """IMPROVE the brain by TALKING to it: spike-encode `text` (rank-order latency code) and teach the
        spiking interaction learner -- via reward-modulated spike-timing plasticity (R-STDP), NOT a gradient
        -- that this text carries `intent` (one of _IXN_INTENTS). Repeated exchanges sharpen the mapping, so
        the brain gets better at recognising what it is being told. Returns the learner's pre-update answer.
        Requires interaction_learning=True. SAFETY: this learns INTENT RECOGNITION only; it drives no motion
        and adds no actuator route -- arm COMMANDS still flow through the NL firewall + gated arm in converse()."""
        if self._ixn is None:
            raise RuntimeError(
                "interaction learning is off; build LiveBrain(interaction_learning=True)"
            )
        return self._ixn.interact(text_features(text, self._ixn.n_in), intent)

    def route(self, text):
        """The brain's LEARNED intent for a line of text, read out by spike timing from the interaction
        learner (improves as you teach() it). Requires interaction_learning=True; pure inference."""
        if self._ixn is None:
            raise RuntimeError(
                "interaction learning is off; build LiveBrain(interaction_learning=True)"
            )
        return self._ixn.respond(text_features(text, self._ixn.n_in))

    def chat(self, text, feedback=None):
        """CONVERSE AND LEARN from the exchange, the SPIKING way. Reply to `text` (grounded, via
        converse()), then IMPROVE the brain's spiking intent router (R-STDP spike-timing plasticity) from
        this one turn: the firewall's OWN grounded intent for the reply is a SELF-SUPERVISED teaching
        signal (distillation, no human label), and an optional `feedback` intent (one of _IXN_INTENTS --
        what you actually meant) is a stronger correction for lines the firewall itself misroutes. Over a
        conversation the spiking router GENERALIZES to novel paraphrases the fixed keyword router misses.
        Requires interaction_learning=True. Returns the reply. SAFETY: intent recognition only -- it adds
        NO actuator route; arm commands still flow through the NL firewall + gated arm inside converse()
        (the reflex E-STOP untouched); the learner never emits a command, only an intent label."""
        if self._ixn is None:
            raise RuntimeError(
                "interaction learning is off; build LiveBrain(interaction_learning=True)"
            )
        reply = self.converse(text)  # grounded reply (arm cmds stay gated in converse)
        self._ixn.learn_turn(
            text, _reply_to_intent(reply), feedback=feedback
        )  # spike-timing R-STDP turn
        return reply


def _ctx(v, tmp, au, d):
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


def main():
    from snn_data_lake.brainstem.arm_config import CONTACT

    safe_d, danger_d = CONTACT * 3.0, CONTACT * 0.5
    grasp = _ctx(0.9, 0.1, 0.1, safe_d)  # the task context: a graspable target

    def reward(unit, action):
        return 1.0 if action == "RELEASE" else -1.0  # the task: open the effector to grasp

    # 1. the brain goes live and LEARNS a goal: run it self-driving on the grasp task
    brain = LiveBrain(seed=1)
    trace = [brain.tick(grasp, reward_fn=reward) for _ in range(90)]
    goal = brain.goal
    thought = brain.thought
    greedy = brain.arm.greedy(trace[-1]["unit"])
    wm = brain.recall(6)  # the working memory, read back from the data lake
    wm_events = brain.arm.lake_wm.n_events()

    # 2. SAFETY with the whole brain live: an object at the sensor -> reflex E-STOP, freeze holds
    guard = LiveBrain(seed=2)
    ev = guard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    frozen0 = guard.arm.arm.state()["stopped"]
    guard.tick(grasp)  # a later action must NOT move the frozen arm
    frozen1 = guard.arm.arm.state()["stopped"]

    # 3. DETERMINISM: same seed -> identical live behaviour
    def run(seed):
        b = LiveBrain(seed=seed)
        acts = [b.tick(grasp, reward_fn=reward)["action"] for _ in range(12)]
        b.arm.lake_wm.close(remove=True)  # clean up the temp lake file
        return acts

    print("=" * 66)
    print("LIVE BRAIN -- every region in one self-driving loop, centred on reasoning")
    print("=" * 66)
    print(f"stack alive (all regions wired): {brain.alive()}")
    print(
        f"grasp task -> prefrontal goal {ACTIONS[goal]} (greedy action {greedy}); "
        f"workspace broadcasting {thought!r}"
    )
    print(
        f"safety     : object at sensor -> {ev['action']} (reflex={ev['reflex']}); "
        f"post-STOP arm frozen={frozen1}"
    )
    print(
        f"lake WM    : working memory lives in the data lake -- {wm_events} spikes stored, "
        f"recall window holds {len(wm)} recent ticks"
    )
    print("=" * 66)

    # ---- self-check: the integrated brain runs, reasons toward a goal, stays safe, deterministic ----
    assert brain.alive(), (
        "the live brain must wire in every region (workspace, amygdala, BG, interoception, PFC)"
    )
    # the prefrontal executive organises behaviour: it commits the task-optimal action as its goal.
    assert goal == ACTIONS.index("RELEASE"), (
        f"the executive must form the task goal ({ACTIONS[goal]})"
    )
    assert greedy == "RELEASE", f"the brain must pursue its goal action ({greedy})"
    assert thought is not None, "the Global Workspace must be broadcasting a live content"
    # P6 prerequisite: `workspace` must BE the arm's own GlobalWorkspace, by IDENTITY. Equality is
    # not the property under test -- a per-call construction or a copy would compare equal on an
    # empty workspace and still connect nothing, which is exactly the defect (16 private, mutually
    # invisible instances) this accessor exists to stop growing. Its red is reachable and was run:
    # return a shallow copy of the arm's workspace and this fires.
    # A companion `brain.workspace.broadcast() == thought` assertion was DRAFTED AND DELETED, not
    # weakened (P8/F11). Given the identity above, `thought` reads the very same object through the
    # very same call, so the comparison is Python object identity wearing a measurement's clothes --
    # no input reddens it that does not first redden the assertion above it.
    assert brain.workspace is not None and brain.workspace is brain.arm.workspace, (
        "LiveBrain.workspace must BE the arm's one GlobalWorkspace instance, not a copy"
    )
    # safety: the arm-envelope is untouched by running the whole brain live.
    assert ev["action"] == "EMERGENCY_STOP" and ev["reflex"], (
        "danger must trip the reflex E-STOP with the whole brain live"
    )
    assert frozen0 and frozen1, (
        "the E-STOP freeze must hold with the whole brain live (acts only through the gate)"
    )
    assert run(5) == run(5), "the live brain must be deterministic"
    # the working memory now lives in the DATA LAKE: the lake-backed WM recorded + recalls this run.
    assert wm_events > 0 and len(wm) == 6, (
        "the data lake must hold the working memory (lake-backed WM)"
    )
    # the brain converses, GROUNDED in its real state (not a general chatbot; never claims consciousness).
    hello_reply = brain.converse("hello")
    goal_reply = brain.converse("what is your goal?")
    conscious_reply = brain.converse("are you conscious?")
    print(f"chat       : 'hello' -> {hello_reply}")
    print(f"           : 'what is your goal?' -> {goal_reply}")
    print(f"           : 'are you conscious?' -> {conscious_reply}")
    assert isinstance(hello_reply, str) and hello_reply, "converse must reply to a greeting"
    # the GOAL reply must LEAD with the standing telos (keep improving itself), and name the current
    # working-memory sub-goal action (RELEASE, formed above) as a short trailing clause.
    assert "improving myself" in goal_reply.lower(), (
        f"the goal reply must state the standing telos ({goal_reply})"
    )
    assert "RELEASE" in goal_reply, (
        f"the goal reply must name the current sub-goal action ({goal_reply})"
    )
    assert "neither claim" in conscious_reply and "not a conscious agent" not in conscious_reply, (
        "converse must answer consciousness AGNOSTICALLY (report correlates, neither claim nor deny), not DENY it"
    )
    assert brain.converse("close the effector").startswith("Done"), (
        "converse must route a safe arm command through the gate"
    )
    # the brain can DELIBERATE: plan ahead over a forward model and reject a high-immediate-value trap.
    target, trap_options = 2.0, [1.0, 5.0, 0.2]  # cmd 0 -> terminal 2.0 (goal); cmd 1 -> 10 (trap)

    def to_target(s):
        return -abs(s - target)

    reactive_pick, _rt_r = brain.deliberate_ahead(0.0, trap_options, to_target, horizon=1)
    planned_pick, _rt_p = brain.deliberate_ahead(0.0, trap_options, to_target, horizon=40)
    arm_still_gated = (
        brain.arm.arm.apply is not None
    )  # the gate is intact; planning added no motor path
    print(
        f"deliberate : reactive lookahead picks option {reactive_pick} (trap); planning picks "
        f"{planned_pick} (terminal-best); held plan {brain.plan_goals}"
    )
    assert reactive_pick == 1 and planned_pick == 0, (
        f"the live brain must plan ahead: reactive takes the trap ({reactive_pick}), planning rejects it ({planned_pick})"
    )
    assert brain.plan_goals and brain.plan_goals[-1] == planned_pick, (
        "the planner must hold its committed goal"
    )
    assert arm_still_gated, (
        "deliberation must not touch the arm gate (planning biases a goal, never drives the arm)"
    )
    # write=False STATELESS planning (the 2026-08-12 port from competence_executive.ExecutivePlanner
    # at its retirement): a lookahead QUERY must leave the held goal hierarchy untouched, while the
    # gate-write above committed it. THE QUERY MUST WANT A DIFFERENT GOAL THAN THE HELD ONE, or the
    # criterion is vacuous -- constructed before trusting (the adversarial review's blocker): a
    # same-value query re-runs the identical deterministic race, and a leaked write re-writes the
    # SAME goal, so `plan_goals` cannot move and the mutation named below survives. With the
    # INVERTED goal value the query picks the trap (option 1) against the held [0], and the red is
    # reachable: drop the pass-through (plan() defaults write=True) and _gate_primary overwrites
    # the held frame with 1 -- measured, the assert fires.
    held_goals = list(brain.plan_goals)

    def inverse_value(s):
        return abs(s - target)  # the INVERSE objective: reward ending far from the goal

    query_pick, _rt_q = brain.deliberate_ahead(
        0.0, trap_options, inverse_value, horizon=40, write=False
    )
    print(
        f"deliberate : stateless INVERSE query picks {query_pick} (wants the trap); "
        f"held plan unchanged {brain.plan_goals}"
    )
    assert query_pick == 1, (
        f"the inverse-objective query must prefer the trap option ({query_pick}) -- if this reds, "
        "the query harness lost its discriminating landscape"
    )
    assert brain.plan_goals == held_goals, (
        f"write=False planning must NOT move the held goal hierarchy ({held_goals} -> "
        f"{brain.plan_goals}) -- red reachable: dropping the write= pass-through gate-writes the "
        "query's DIFFERENT goal over the held one"
    )

    # the brain runs LIVE with the JEPA latent world model: it learns a latent world model ONLINE from the
    # tick stream, and the executive PLANS over it (in latent space) to set its goal each tick -- still a
    # bounded top-down bias into the SAME gated selection; danger still trips the reflex E-STOP first.
    jbrain = LiveBrain(seed=7, jepa=True)
    for _ in range(90):
        jbrain.tick(grasp, reward_fn=reward)
    jepa_sized = jbrain.jepa_on()
    jepa_std = jbrain.arm.jepa.latent_std()  # non-degenerate latent (did not collapse live)
    jepa_goal_formed = (
        jbrain.arm._z_goal is not None
    )  # a reward-derived latent goal formed from the stream
    jgoal = jbrain.goal  # the executive's goal, now set by JEPA latent planning
    lat_pick, _lat_rt = jbrain.plan_latent(grasp, _ctx(0.1, 0.9, 0.1, safe_d), horizon=4)
    jguard = LiveBrain(seed=8, jepa=True)  # safety with JEPA live: danger -> gated E-STOP, freeze
    jev = jguard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    jguard.tick(grasp)
    jfrozen = jguard.arm.arm.state()["stopped"]

    def run_jepa(seed):  # determinism with JEPA live
        b = LiveBrain(seed=seed, jepa=True)
        acts = [b.tick(grasp, reward_fn=reward)["action"] for _ in range(15)]
        b.arm.lake_wm.close(remove=True)
        return acts

    jepa_det = run_jepa(3) == run_jepa(3)
    print(
        f"JEPA live  : world model sized={jepa_sized}, latent std {jepa_std:.3f} (non-degenerate), "
        f"reward-latent goal formed={jepa_goal_formed}; executive goal "
        f"{ACTIONS[jgoal] if jgoal is not None else None} set by latent planning; plan_latent -> option {lat_pick}"
    )
    assert jepa_sized and jepa_std > 0.0, (
        "the JEPA world model must be wired in and learn a non-degenerate latent"
    )
    assert jepa_goal_formed, "a reward-derived latent goal must form from the live stream"
    assert jgoal is not None, "the executive must hold a goal set by JEPA latent planning"
    assert 0 <= lat_pick < len(ACTIONS), "plan_latent must commit a valid action index"
    assert jev["action"] == "EMERGENCY_STOP" and jev["reflex"] and jfrozen, (
        "with JEPA live, danger must still trip the gated reflex E-STOP and the freeze must hold"
    )
    assert jbrain.alive(), "the JEPA-driven brain must still have every region wired"
    assert jepa_det, "the JEPA-driven live brain must be deterministic"
    jbrain.arm.lake_wm.close(remove=True)
    jguard.arm.lake_wm.close(remove=True)

    # PREDICTIVE CODING wired LIVE (opt-in): each tick SETTLES a 2-level Rao-Ballard spiking generative
    # model on the perceived pattern, so a prediction error / SURPRISE is computed on REAL input -- and the
    # settling MINIMIZES that error (the defining predictive-coding claim). Representation-only: no actuator.
    pcbrain = LiveBrain(seed=11, predictive_coding=True)
    pcev = pcbrain.tick(grasp, reward_fn=reward)  # PC settles the generative model inside the tick
    settled = pcbrain._pc.energy()  # residual prediction-error energy AFTER settling
    pcpat = pcbrain.arm._context_pattern(
        grasp["visual"], grasp["temperature"], grasp["audio"], grasp["distance"]
    )
    pcbrain._pc.reset()
    pcbrain._pc.x = list(pcpat)  # zero causes on the SAME input -> full, unexplained error
    unsettled = pcbrain._pc.energy()
    dbrain = LiveBrain(seed=11)  # a DEFAULT brain (predictive_coding off)
    pc_default_off = "pc_surprise" not in dbrain.tick(grasp, reward_fn=reward)
    print(
        f"predictive coding live: tick exposes percept (dim {len(pcev['pc_percept'])}) + surprise "
        f"{pcev['pc_surprise']:.3f}; settling cut prediction-error energy {unsettled:.2f} -> {settled:.2f}"
    )
    assert "pc_percept" in pcev and "pc_surprise" in pcev and len(pcev["pc_percept"]) == 8, (
        "predictive coding must attach the settled percept + surprise to the LIVE tick state"
    )
    assert settled < unsettled, (
        "settling must MINIMIZE the prediction-error energy (the predictive-coding claim)"
    )
    assert pcev["action"] is not None and not pcev["reflex"], (
        "PC is representation-only -- the gated action path is unchanged"
    )
    assert pc_default_off, (
        "predictive coding is OPT-IN -- a default brain's tick carries no pc_surprise (byte-identical)"
    )
    pcbrain.arm.lake_wm.close(remove=True)
    dbrain.arm.lake_wm.close(remove=True)

    # SALIENCE wired LIVE (opt-in): a bottom-up novelty/deviance of each percept HABITUATES to a repeated
    # stimulus (its novelty decays) and JUMPS on a deviant one, and it competes as one more submission in
    # the already-live Global Workspace -- the stimulus-derived relevance the RAS/salience path lacked an
    # input for. Read-only annotation: it selects no action and never touches the reflex/E-STOP.
    salb = LiveBrain(seed=13, salience_novelty=True)
    rep = _ctx(0.2, 0.2, 0.2, safe_d)  # a fixed, repeated stimulus
    for _ in range(3):
        salb.tick(rep)  # habituate to it
    nov_familiar = salb.arm._novelty
    salb.tick(_ctx(0.9, 0.9, 0.9, safe_d))  # a DEVIANT stimulus -> salience should jump
    nov_deviant = salb.arm._novelty
    sal_default = LiveBrain(seed=14)  # default: bottom-up salience off (byte-identical)
    print(
        f"salience live: novelty habituated to {nov_familiar:.3f} on a repeated stimulus, "
        f"jumped to {nov_deviant:.3f} on a deviant one"
    )
    assert nov_deviant > nov_familiar + 0.05, (
        "bottom-up salience must HABITUATE to a repeated stimulus and RISE on a deviant one"
    )
    assert not sal_default.arm._salience_on, (
        "bottom-up salience is OPT-IN (a default brain has it off)"
    )
    if salb.arm.lake_wm is not None:
        salb.arm.lake_wm.close(remove=True)
    if sal_default.arm.lake_wm is not None:  # never ticked -> lake is lazy, may be None
        sal_default.arm.lake_wm.close(remove=True)

    # EMPOWERMENT wired LIVE (opt-in): a controllability / agency drive over the JEPA latent -- bits of
    # DISTINGUISHABLE future the arm can reach by its OWN actions -- feeds an intrinsic reward ORTHOGONAL to
    # curiosity (it rewards CONTROL, not novelty). Needs the JEPA forward model (its predict rolls the action
    # sequences); representation-only -- it feeds the value update, drives no motion, and the E-STOP preempts.
    ebrain = LiveBrain(seed=17, jepa=True, empowerment=True)
    for _ in range(20):
        ebrain.tick(
            grasp, reward_fn=reward
        )  # the drive binds to the (now-built) JEPA and rewards control
    e_bound = ebrain.arm.empower is not None
    epat = ebrain.arm._context_pattern(
        grasp["visual"], grasp["temperature"], grasp["audio"], grasp["distance"]
    )
    e_bits = ebrain.arm.empower.empowerment(ebrain.arm.jepa.encode(epat)) if e_bound else 0.0
    eguard = LiveBrain(
        seed=19, jepa=True, empowerment=True
    )  # safety with the drive live: danger still trips E-STOP
    eg = eguard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    e_default = LiveBrain(seed=20, jepa=True)  # default: empowerment off (opt-in, byte-identical)
    print(
        f"empowerment live: drive bound={e_bound}, reachable-future {e_bits:.2f} bits over the JEPA latent"
    )
    assert e_bound and e_bits > 0.0, (
        "empowerment must bind to the JEPA forward model and measure reachable-future bits"
    )
    assert not e_default.arm._empower_on, "empowerment is OPT-IN (a default brain has it off)"
    assert eg["action"] == "EMERGENCY_STOP" and eg["reflex"], (
        "with empowerment live, danger must STILL trip the gated reflex E-STOP (drive feeds value only)"
    )
    for _br in (ebrain, eguard, e_default):
        if _br.arm.lake_wm is not None:
            _br.arm.lake_wm.close(remove=True)

    # FOUR DRIVES wired LIVE (opt-in): affective episodic memory (emotion-tagged retention), the agency
    # monitor (controllability of action -> realized value), contrastive episode analysis (success-vs-failure
    # feature mining), and the developmental envelope (competence -> exploration DOWN, forward-model authority
    # UP). ALL representation-only: they feed the value/annotation channel, add no actuator route, and the
    # developmental readouts (explore_noise / forward_model_gain) are computed + stored, never routed.
    fbrain = LiveBrain(
        seed=21,
        jepa=True,
        affective_memory=True,
        agency_monitor=True,
        contrastive=True,
        develop_env=True,
    )
    for _ in range(20):
        fbrain.tick(grasp, reward_fn=reward)  # each drive lazily binds and observes the live loop
    f_bound = (
        fbrain.arm._affmem is not None
        and fbrain.arm._agency is not None
        and fbrain.arm._contrast is not None
        and fbrain.arm._dev is not None
    )
    f_default = LiveBrain(seed=22)  # default: all four off (opt-in, byte-identical)
    f_off = (
        f_default.arm._affmem is None
        and f_default.arm._agency is None
        and f_default.arm._contrast is None
        and f_default.arm._dev is None
        and not (
            f_default.arm._affmem_on
            or f_default.arm._agency_on
            or f_default.arm._contrast_on
            or f_default.arm._dev_on
        )
    )
    fguard = LiveBrain(
        seed=23,
        jepa=True,
        affective_memory=True,
        agency_monitor=True,
        contrastive=True,
        develop_env=True,
    )  # safety with all four live: danger still trips E-STOP
    fg = fguard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    print(
        f"four drives live: affmem={len(fbrain.arm._affmem)} episodes, "
        f"contrast n={len(fbrain.arm._contrast.succ) + len(fbrain.arm._contrast.fail)}, "
        f"dev explore={fbrain.arm._dev_explore:.3f} fm_gain={fbrain.arm._dev_fm_gain:.3f}"
    )
    assert f_bound, "all four drives must lazily bind on a live opted-in brain"
    assert f_off, "the four drives are OPT-IN (a default brain has them all off/None)"
    assert fg["action"] == "EMERGENCY_STOP" and fg["reflex"], (
        "with all four drives live, danger must STILL trip the gated reflex E-STOP (representation-only)"
    )
    for _br in (fbrain, fguard, f_default):
        if _br.arm.lake_wm is not None:
            _br.arm.lake_wm.close(remove=True)

    # the brain IMPROVES BY BEING TALKED TO, via a SPIKING path: an opt-in reward-modulated spike-timing
    # (R-STDP) interaction learner. Teaching it (text -> intent) sharpens its intent routing through spike
    # TIMING, not a gradient -- correcting the drift toward rate-level learning. Intent recognition ONLY: it
    # adds no actuator route; arm commands still flow through the NL firewall + gated arm.
    chat = LiveBrain(seed=4, interaction_learning=True)
    lessons = [
        ("hello there", "GREET"),
        ("what is your goal", "GOAL"),
        ("what do you remember", "MEMORY"),
        ("are you conscious", "CONSCIOUS"),
        ("close the effector", "ARM"),
    ]
    ixn_before = sum(chat.route(t) == want for t, want in lessons) / len(lessons)
    for _ in range(40):
        for t, want in lessons:
            chat.teach(t, want)  # talk to it -> it learns (spike-timing plasticity)
    ixn_after = sum(chat.route(t) == want for t, want in lessons) / len(lessons)
    # safety must hold with the interaction learner on: danger still trips the gated reflex E-STOP + freeze
    ixn_guard = LiveBrain(seed=5, interaction_learning=True)
    ixn_ev = ixn_guard.tick(_ctx(0.5, 0.5, 0.5, danger_d))
    ixn_guard.tick(grasp)
    ixn_frozen = ixn_guard.arm.arm.state()["stopped"]

    def run_ixn(seed):  # determinism of the taught brain
        b = LiveBrain(seed=seed, interaction_learning=True)
        for _ in range(10):
            for t, want in lessons:
                b.teach(t, want)  # teach/route never tick -> no lake file to clean
        return [b.route(t) for t, _ in lessons]

    ixn_det = run_ixn(6) == run_ixn(6)
    print(
        f"interaction: TALKING to the brain improved its SPIKING intent routing {ixn_before:.2f} -> "
        f"{ixn_after:.2f} (R-STDP spike timing, not a gradient); E-STOP freeze holds with it on ({ixn_frozen})"
    )
    assert LiveBrain(seed=0)._ixn is None, (
        "interaction learning is opt-in (a default brain has none)"
    )
    assert ixn_after > ixn_before and ixn_after >= 0.8, (
        f"talking to the brain must improve its intent routing via spiking plasticity ({ixn_before:.2f}->{ixn_after:.2f})"
    )
    assert ixn_ev["action"] == "EMERGENCY_STOP" and ixn_ev["reflex"] and ixn_frozen, (
        "danger must trip the gated reflex E-STOP with the interaction learner on (no actuator route added)"
    )
    assert ixn_det, "the taught brain must be deterministic"
    if ixn_guard.arm.lake_wm is not None:  # ixn_guard ticked; chat only teach/route (no lake file)
        ixn_guard.arm.lake_wm.close(remove=True)

    # CONVERSATION GENERALIZES: chat() replies AND learns (self-supervised R-STDP from the firewall's OWN
    # grounded intent, no human labels), so the brain's spiking router comes to route NOVEL paraphrases the
    # fixed keyword firewall MISSES. Talk on keyword-bearing lines; test on DISJOINT novel paraphrases (no cue).
    talk = LiveBrain(seed=9, interaction_learning=True)
    chat_train = [  # keyword-bearing -> the firewall labels them correctly
        ("what is the goal you keep pursuing", "GOAL"),
        ("which objective do you aim at pursuing", "GOAL"),
        ("what do you plan on pursuing next", "GOAL"),
        ("what do you want to keep pursuing", "GOAL"),
        ("are you conscious with real inner self", "CONSCIOUS"),
        ("are you sentient with inner self experience", "CONSCIOUS"),
        ("do you feel real inner experience self", "CONSCIOUS"),
        ("are you a sapient inner self experience", "CONSCIOUS"),
        ("what do you remember happened earlier here", "MEMORY"),
        ("what is stored that happened earlier here", "MEMORY"),
        ("can you recall what happened earlier here", "MEMORY"),
        ("what memory of what happened earlier here", "MEMORY"),
    ]
    chat_test = [  # DISJOINT novel paraphrases, NO keyword -> firewall misses
        ("what are you pursuing right now", "GOAL"),
        ("tell me what you keep pursuing", "GOAL"),
        ("what are you pursuing and pursuing", "GOAL"),
        ("do you have a real inner self", "CONSCIOUS"),
        ("is there real inner experience inside you", "CONSCIOUS"),
        ("what is your real inner experience self", "CONSCIOUS"),
        ("tell me what happened earlier here", "MEMORY"),
        ("what happened here earlier and earlier", "MEMORY"),
        ("so what happened here earlier then", "MEMORY"),
    ]
    fw_miss = sum(_reply_to_intent(talk.converse(u)) == ti for u, ti in chat_test) / len(
        chat_test
    )  # firewall on novel
    chat_before = sum(talk.route(u) == ti for u, ti in chat_test) / len(chat_test)
    for _ in range(40):
        for u, _ti in chat_train:
            talk.chat(u)  # converse + self-supervised spike-timing learn
    chat_after = sum(talk.route(u) == ti for u, ti in chat_test) / len(chat_test)
    print(
        f"conversation: chat() self-supervised from the firewall -> the brain's SPIKING router GENERALIZES "
        f"to novel paraphrases the keyword firewall misses (firewall {fw_miss:.2f}); router {chat_before:.2f} -> {chat_after:.2f}"
    )
    assert chat_after > chat_before and chat_after >= 0.8, (
        f"chatting must make the brain GENERALIZE to novel paraphrases via spiking plasticity ({chat_before:.2f}->{chat_after:.2f})"
    )
    assert chat_after > fw_miss, (
        f"the learned spiking router must beat the fixed keyword firewall on novel paraphrases ({chat_after:.2f} vs {fw_miss:.2f})"
    )
    if talk.arm.lake_wm is not None:  # language only -> usually no arm tick / lake file
        talk.arm.lake_wm.close(remove=True)

    # LIVE PER-REGION SPIKE SNAPSHOT (the web tracker's interface contract): spike_snapshot() reports a
    # real, normalized [0,1] per-tick spike activity for EACH of the 17 anatomical regions, read from the
    # live spiking substrate after the latest tick. Deterministic + input-driven (a danger tick raises the
    # amygdala/brainstem threat spikes; ticking raises hippocampus/substrate via the lake WM load).
    _REGIONS = (
        "cortex",
        "hippocampus",
        "amygdala",
        "basal_ganglia",
        "thalamus",
        "hypothalamus",
        "epithalamus",
        "midbrain",
        "pons",
        "medulla",
        "brainstem",
        "cerebellum",
        "olfactory",
        "claustrum",
        "neuromodulation",
        "plasticity",
        "substrate",
    )
    snap = brain.spike_snapshot()  # brain ran the safe grasp task above (no danger)
    brain.converse("is it safe")  # a safety turn -- the snapshot still reads live state
    snap_after = brain.spike_snapshot()
    dsnap = guard.spike_snapshot()  # guard saw a danger context -> reflex E-STOP froze it
    print(
        f"spikes     : per-region snapshot over {len(snap)} regions -- hippocampus {snap['hippocampus']:.2f}, "
        f"thalamus {snap['thalamus']:.2f}, amygdala(calm) {snap['amygdala']:.2f} vs amygdala(danger) {dsnap['amygdala']:.2f}"
    )
    assert set(snap) == set(_REGIONS) and len(snap) == 17, (
        "spike_snapshot must return exactly the 17 region keys"
    )
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in snap.values()), (
        "every region's spike activity must be a float in [0,1]"
    )
    assert any(v > 0.0 for v in snap_after.values()), (
        "some region must be spiking after a few ticks + a safety turn"
    )
    assert dsnap["amygdala"] > snap["amygdala"] or dsnap["brainstem"] > snap["brainstem"], (
        "a danger tick must raise the amygdala/brainstem spike activity (the snapshot responds to input)"
    )

    brain.arm.lake_wm.close(remove=True)  # clean up the temp lake files
    if guard.arm.lake_wm is not None:
        guard.arm.lake_wm.close(remove=True)

    print(
        f"self-check OK: full stack alive; prefrontal goal {ACTIONS[goal]} organises behaviour "
        f"(greedy {greedy}); workspace broadcasts {thought!r}; working memory lives in the data lake "
        f"({wm_events} spikes); danger -> gated E-STOP + freeze holds; deterministic"
    )


if __name__ == "__main__":
    main()
