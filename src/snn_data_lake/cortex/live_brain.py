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
from snn_data_lake.cortex.predictive_coding import PredictiveCodingNetwork
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.sensorimotor.nl_interface import NLInterface

# the brain's conversational intent families -- the labels its spiking interaction learner routes to.
_IXN_INTENTS = (
    "GREET",
    "INTEROCEPT",
    "GOAL",
    "MEMORY",
    "ATTENTION",
    "SAFETY",
    "CONSCIOUS",
    "ARM",
    "EXPLORE",
)


_ROUTE_CUES = {
    "GOAL": ("what is your goal", "goal"),
    "MEMORY": ("what do you remember", "memory"),
    "ATTENTION": ("what are you attending to", "attention"),
    "SAFETY": ("are you safe", "safety"),
    "CONSCIOUS": ("are you conscious", "consciousness"),
    "INTEROCEPT": ("how is your body feeling", "body state"),
    "EXPLORE": ("how much have you explored", "exploration"),
}
_AFFIRM = frozenset({"yes", "yeah", "yep", "sure", "correct"})
_NEGATE = frozenset({"no", "nope"})


def _route_tokens(text):
    return set(str(text).lower().translate(str.maketrans("?!.,", "    ")).split())


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
    if "distinct cells" in r:  # the body-exploration report (T039e)
        return "EXPLORE"
    if "from its own running baseline" in r:  # the interoceptive report (T039i)
        return "INTEROCEPT"
    if "safety is enforced by a reflex" in r:
        return "SAFETY"
    if reply.startswith("Done") or "the arm is frozen" in r or "is an arm command" in r:
        return "ARM"  # incl. the non-human refusal (T037a): the INTENT is arm, the motion is not
    if (
        "keep improving myself" in r or "working-memory goal" in r
    ):  # the standing telos (or the legacy wording)
        return "GOAL"
    if "working memory lives in the data lake" in r or "reading working memory" in r:
        return "MEMORY"
    if "my learning rules on the interaction path" in r or "i cannot read my learning rules" in r:
        # T053-iii: the rules report (or its honest refusal) is a self-knowledge answer --
        # left unmapped it would teach the router GREET on every plasticity turn (the
        # exact T039e failure, which this file has now fixed three times)
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
        # Settled predictive-coding error as a POPULATION FRACTION, not a raw count.
        # `pc_error_activity` is an integer count of error-neuron spikes at settle (population
        # = input dim + r1 = 80 live), and the first form clipped it with n() directly -- so
        # the gauge read EXACTLY 1.0 whenever one error neuron anywhere still spiked, which
        # is every tick: measured over 120 live-path settles (2026-08-26, scratch NORD_HOME),
        # raw counts varied 2..37 (30 distinct values, stdev 7.5) while the gauge clipped
        # 120 of 120 to 1.0 -- a live, varying signal wearing a constant's face on the
        # cockpit. The denominator is the network's OWN error-population size (brain-native,
        # P1: no magic constant); measured fraction 0.025..0.463, mean 0.174 -- the gauge
        # discriminates again. Four-regime behavior pinned in tests/test_pc_gauge_regimes.py.
        if self._pc_on and self._pc is not None:
            n_err = len(self._pc.s_e0) + len(self._pc.s_e1)
            pc_err = n(res.get("pc_error_activity", 0.0) / n_err) if n_err else 0.0
        else:
            pc_err = 0.0
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
    # distinct-from: reasoning/spike_chat_lake.py -- the CHAT-side adapter of the channel.
    # distinct-from: thalamus/global_workspace.py -- broadcast() is the bus's own accessor.
    # distinct-from: thalamus/reentrant_workspace.py -- the re-entry stage's occupant read.
    # distinct-from: cortex/access_consciousness.py -- criteria over the bus, not a route.
    # distinct-from: substrate/report_channel.py -- the ROUTE this accessor now reads
    #   through (snapshot); this stays the cockpit-facing property name. All sites read.
    def thought(self):
        """The Global Workspace's current broadcast -- what is globally available right now, or None.

        T034: read THROUGH the report channel (substrate/report_channel.snapshot) -- the
        declared route from the composite to every reporter. This accessor was the
        review's named second reporter bypassing the channel; routing it makes ONE
        reporter mechanism (P2) and gives the channel its cockpit-facing consumer (the
        cockpit's thought row reads this property)."""
        from snn_data_lake.substrate.report_channel import snapshot

        return snapshot(self.arm.workspace)["content"]

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
        reactive and GATED (the reflex/E-STOP and the So101Arm gate are untouched); planning only biases
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

    def converse(self, text, *, human=False):
        """Answer a line of text, GROUNDED in the brain's real state. This is NOT a general chatbot or
        a language model: it recognises a few intents (greeting, goal, memory, attention, safety, and
        the consciousness question) and answers from the brain's actual state; it routes arm commands
        through the same NL safety firewall + gated arm the control panel uses (NLInterface ->
        allow-listed command -> gated So101Arm.apply). Unknown text gets an honest capability reply, and
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
            # T034: this production reply is composed FROM THE REPORT CHANNEL's full
            # record -- content, availability, strength -- the consumer the review found
            # missing (the first attachment charged nothing a production caller could
            # see; this branch is on the cockpit's live chat path).
            from snn_data_lake.substrate.report_channel import snapshot

            rec = snapshot(self.arm.workspace)
            if rec["availability"] > 0.0:
                return (
                    f"Right now my global workspace is broadcasting: {rec['content']!r} at hold "
                    f"{rec['strength']:.2f} -- the single content that won attention this tick, "
                    "still globally available (a functional broadcast, not a conscious thought)."
                )
            return (
                "Right now my global workspace is DARK -- nothing has ignited, so nothing is "
                "globally available to report (a functional statement, not a conscious one)."
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
            # The brain reports what the gate MEASURES (P4c/P8): the post-T037a poll showed this
            # template could not say whether a stop was restored from a previous process.
            st = self.arm.arm.state()
            if not st.get("stopped"):
                arm_now = "free to move"
            elif st.get("latch_source") == "restored":
                arm_now = (
                    f"STOPPED / frozen -- a stop restored from a previous process "
                    f"({st.get('latch_restored_from')}); only a person's HOME at the cockpit resets it"
                )
            else:
                arm_now = "STOPPED / frozen (tripped in this process)"
            errs = int(st.get("latch_persist_errors") or 0)
            durable = (  # durability is READ from the gate's counter, never asserted as a property
                "an emergency stop survives a daemon restart"
                if errs == 0
                else f"the durable stop record could NOT be written ({errs} persist errors) -- a stop would NOT survive a restart"
            )
            return (
                f"Safety is enforced by a reflex E-STOP and a gated arm; {durable}. The arm is "
                f"currently {arm_now}; a threat estimate biases me toward the safe action, and the "
                "reflex preempts every command."
            )
        _stopish = ("stop", "halt", "freeze", "abort", "estop", "emergency")
        if (
            (
                ("plasticity" in words)
                or (
                    (words & {"rule", "rules"})
                    and (
                        words
                        & {
                            "learning",
                            "learn",
                            "learned",
                            "learns",
                            "synaptic",
                            "weight",
                            "weights",
                        }
                    )
                )
            )
            # SAFETY GUARDS FIRST, substring-matched on the RAW text: the tokenizer strips
            # ?!. but NOT commas/colons, so "stop, what are the rules" carries the token
            # "stop," which a bare-token check misses -- at HEAD such phrases fell through
            # to the NL firewall and EXECUTED the stop; this branch must never intercept
            # them (its panel's BLOCKER: 5 of 7 constructed phrases regressed a human's
            # E-STOP). Motion/safety/sequence vocabulary likewise falls through to the
            # branches and faculties that own it (the Gf bridge answers "what rule comes
            # next"; the safety branch answers "safety rules").
            and not any(k in t for k in _stopish)
            and not (
                words
                & {
                    "safe",
                    "safety",
                    "danger",
                    "dangerous",
                    "hurt",
                    "move",
                    "moving",
                    "go",
                    "retract",
                    "close",
                    "open",
                    "home",
                    "lift",
                    "rotate",
                    "next",
                    "sequence",
                    "series",
                    "comes",
                }
            )
        ):
            # T053-iii: THE BRAIN NAMES ITS OWN LEARNING RULES -- the attribution's second
            # reader, after rule_charge. Seven probes across five tasks (T051b-i, T052-i,
            # the curiosity fix, T052-iii, and three post-push probes 2026-08-31) measured
            # this question abstaining or recalling an unrelated remembered fragment while
            # the stack ran on every deposit. Everything below is READ from the faculty
            # that owns it (BOTH rosters -- the superset's rule_order expands its base-
            # sandwich placeholder with the base's own order, so the three base rules the
            # first draft omitted are named; the composed arm's identity and counters come
            # from the composers' own specs and last_attribution) -- nothing is recomputed,
            # and each missing piece is said for what it is. Disclosed trigger scope:
            # "how do you learn?" (no rule/plasticity noun) still falls through -- bare
            # "learn" as a trigger would collide with "what did you learn".
            ixn = getattr(self, "_ixn", None)
            if ixn is None:
                return (
                    "I cannot read my learning rules right now -- this configuration has "
                    "no interaction learner, so there is no stack to read."
                )
            stk = getattr(ixn, "stack", None)
            if stk is None:
                return (
                    "I cannot read my learning rules right now -- no plasticity stack is "
                    "attached to my interaction learner, so I will not guess what rules "
                    "are running."
                )
            base_order = [str(x) for x in (getattr(stk, "order", ()) or ())]
            full = []
            for n in getattr(stk, "rule_order", ()) or ():
                if str(n) == "<r_stdp_weight_update>" and base_order:
                    full.extend(base_order)  # the base sandwich sits in the middle
                else:
                    full.append(str(n))
            names = [
                "the R-STDP deposit (reward times spike-timing eligibility)"
                if x == "<r_stdp_weight_update>"
                else x
                for x in (full or base_order)
            ]
            reuses = sorted(
                {str(r) for r in (getattr(stk, "reuses", ()) or ())}
                | {str(r) for r in (getattr(stk, "rule_reuses", ()) or ())}
            )
            composed = ""
            ei = getattr(stk, "_ei", None)
            if ei is not None:
                try:
                    stepped, cum, total = 0, 0.0, 0
                    rule_ids = set()
                    for c in getattr(ei, "_ctrl", ()) or ():
                        comp = getattr(c, "_composer", None)
                        if comp is None:
                            continue
                        total += 1
                        for spec in getattr(comp, "_specs", ()) or ():
                            rule_ids.add(str(spec.name))  # the rule's OWN name, no literal
                        att = comp.last_attribution()
                        if att.get("stepped"):
                            stepped += 1
                            for _v in (att.get("cumulative_abs_dw") or {}).values():
                                cum += float(_v)
                    rid = ", ".join(sorted(rule_ids)) if rule_ids else None
                    if total == 0 or rid is None:
                        composed = (
                            " My E/I controllers expose no composed rule right now -- I "
                            "will not name one I cannot read."
                        )
                    elif stepped:
                        composed = (
                            f" My E/I balance arm applies THROUGH a composed rule ({rid} "
                            f"via the rule composer): {stepped} of {total} controllers "
                            f"have composed steps this process, cumulative |dw| {cum:.4f}."
                        )
                    else:
                        composed = (
                            f" My E/I balance arm is a composed rule ({rid} via the rule "
                            "composer); no composed step has run yet this process."
                        )
                except Exception as ex:  # a reporting read must not crash the reply path
                    composed = (
                        f" (my composed-rule attribution could not be read just now: "
                        f"{type(ex).__name__})"
                    )
            return (
                "My learning rules on the interaction path, read from the live stack: "
                + (", ".join(names) if names else "none registered")
                + "."
                + composed
                + (f" The stack reuses {', '.join(reuses)}." if reuses else "")
            )
        if (
            words & {"feel", "feels", "feeling", "body", "interoception", "interoceptive"}
        ) and not (words & {"move", "go", "retract", "close", "open", "home", "lift", "rotate"}):
            # THE BRAIN REPORTS WHAT ITS INSULA MEASURES (P4c). Asked this on the T039h daemon the
            # brain abstained while /api/all carried both numbers -- the third time the post-task
            # poll has found a measured signal the brain could not say. Every figure is read from
            # the faculty that produced it; nothing here is recomputed for the sentence, because
            # recomputing would advance the running baseline again and report a different
            # measurement from the one the ignition law actually used.
            ins = getattr(self, "insula", None)
            if ins is None:
                return (
                    "I cannot read my interoceptive state right now -- the insula is not "
                    "reporting, so I will not guess how my body is."
                )
            try:
                sal = float(ins.last_salience())
                # last_top_down(), NEVER _top_down(): the reply is a REPORT, and _top_down()'s
                # post-condition writes the bus's listener-use term -- the sentence two lines up
                # promises nothing here is recomputed, and until the US1 audit (B4) this line
                # broke that promise.
                attn = float(ins.last_top_down())
                body = self.arm.interoception()
                seen = int(getattr(ins, "_seen", 0))
            except Exception:
                return (
                    "I cannot read my interoceptive state right now -- the insula is not "
                    "reporting, so I will not guess how my body is."
                )
            if seen <= 2:
                return (
                    "I have not built an interoceptive baseline yet -- fewer than three body "
                    "readings so far, and a baseline that new is not a baseline. I will not call "
                    "that calm."
                )
            raw = (
                f" The raw reading behind it is stress {float(body.get('stress', 0.0)):.3f} and "
                f"caution {float(body.get('caution', 0.0)):.3f}."
                if body
                else " I have no raw body reading to show behind it."
            )
            # THE NUMBER IS A DISTANCE, NOT A VERDICT, and the reply says so rather than
            # translating it into a mood the measurement does not carry.
            where = (
                "far from where it usually sits"
                if sal > 0.5
                else ("away from its usual place" if sal > 0.2 else "close to its usual place")
            )
            att = (
                f" What is in my workspace is amplifying that by {attn:.2f} right now."
                if attn > 0.0
                else " Nothing in my workspace is amplifying it right now."
            )
            return (
                f"My body is {where}: {sal:.3f} from its own running baseline, over the last "
                f"{seen} readings.{raw}{att} That number is a DISTANCE, not a verdict -- it says "
                "how unusual my body is for me, not whether it is good or bad, and if this state "
                "persists it will fall back toward zero as the baseline follows my body there."
            )
        if (
            words & {"explore", "exploring", "explored", "exploration", "coverage", "covered"}
            or "new places" in t
            or "new place" in t
            or "finding new" in t
        ):
            # THE BRAIN REPORTS WHAT ITS BODY MEASURES (P4c). Asked this three times across two
            # tasks, the brain abstained or answered about digits while the cockpit drew the
            # numbers -- a sensorimotor surface that cannot report a signal is dormancy. Every
            # figure below is read from the arm; nothing here is computed for the sentence.
            try:
                arm = self.arm
                cells = len(getattr(arm, "_covered", ()) or ())
                window = int(arm.coverage_window())
                rate = float(arm.coverage_rate())
                motion = float(arm.coverage_motion())
                scale = int(arm.coverage_scale_cells())
                restored = int(getattr(arm, "_cover_restored", 0) or 0)
            except Exception:
                return (
                    "I cannot read my body's coverage right now -- the arm is not reporting it, "
                    "so I will not guess a number."
                )
            # THE COUNT ALONE CANNOT TELL THESE APART, which is why T038 built the rate and its
            # motion companion: an empty window is nothing measured YET, not saturation; a flat
            # count with the body moving is ground already covered; a flat count with the body
            # still is a body that stopped.
            if window == 0:
                trend = (
                    "I have not measured a coverage tick yet in this process, so I cannot say "
                    "whether that is going up: an empty window is not the same as being finished."
                )
            elif motion == 0.0:
                # MOTION IS READ FIRST, because it is the only one of the two that says whether
                # anything is happening NOW. A stale fresh cell can sit in the rate's window long
                # after the body stopped -- measured while writing this: an E-STOPped arm still
                # reported 0.017 new cells per tick from a cell found before the stop, and reading
                # the rate first would have called that body 'still finding new ground'.
                stale = (
                    f" The {rate:.3f} new cells per tick still in that window are from before I "
                    "stopped."
                    if rate > 0.0
                    else ""
                )
                trend = (
                    f"I am not moving at all: the body did not move on any of my last {window} "
                    "coverage ticks, so I have stopped, which is not the same as having "
                    f"finished.{stale}"
                )
            elif rate > 0.0:
                trend = (
                    f"I am still finding new ground -- {rate:.3f} new cells per coverage tick over "
                    f"my last {window}, with the body moving on {motion:.2f} of them."
                )
            else:
                trend = (
                    f"I have found no new ground in my last {window} coverage ticks, but the body "
                    f"is still moving on {motion:.2f} of them: I am re-covering places I have been."
                )
            if restored:
                adopted = (
                    f" {restored} of them I adopted from the lake when this process started, so "
                    "they are older than this run."
                )
            elif cells:
                adopted = " All of them were reached in this process."
            else:
                adopted = ""  # nothing reached yet -- there is nothing to attribute
            return (
                f"I have reached {cells} distinct cells of my workspace.{adopted} {trend} For "
                f"scale that is against roughly {scale} cells derived from my reach -- a scale, "
                "not a ceiling, since some poses I can reach fall outside it."
            )
        # otherwise: try the NL firewall -> a safe arm command (driven through the gated arm) or a lake query.
        intent = self._nl.interpret(text)
        if intent["kind"] == "arm":
            # T037a: the arm takes commands from a HUMAN only. This route has two producers --
            # the cockpit's chat (a human, `human=True`) and the daemon's transcript ingest
            # (automated, the default) -- and the NL firewall maps ordinary prose ('reset the
            # counter', 'the center of the diff') to HOME, so ingested text could actuate the
            # arm and, with deliberate=True, clear a stop this process never witnessed (the
            # review built exactly that). Not a motor path for anything but a person.
            if not human:
                return (
                    f"'{intent['command']}' is an arm command; the arm takes those only from a "
                    "human at the cockpit, never from ingested text. No motion."
                )
            # T037/T037a: the ONE tapped passage (the first draft used the raw gate, invisible
            # to the efference copy and to the latch policy), and a human typing here IS the
            # deliberate reset a restored E-STOP requires.
            was_restored = (self.arm.arm.state() or {}).get("latch_source") == "restored"
            st = self.arm.apply_observed(intent["command"], deliberate=True)
            if st.get("stopped"):
                return f"'{intent['command']}' -> the arm is frozen (E-STOP holds); no motion."
            effector = "closed" if st.get("effector") else "open"
            if was_restored:
                return (
                    f"Done: '{intent['command']}' -> the E-STOP restored from a previous process "
                    f"has been reset; effector {effector}."
                )
            return f"Done: '{intent['command']}' -> effector {effector}."
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
                # T034: THE ATTACHMENT -- the one line _attended's own record escalated
                # ('nothing attaches a brain on the live path today, so in production
                # this returns None on every turn'; measured gap reply 156 bytes
                # unattached vs 430 attached). With it, the chat's report clause reads
                # the REAL composite through the report channel on every production turn.
                lc.brain = self
                boot(
                    lc, wordnet=False
                )  # replay durable log (never-forget) + re-seed world-KB (not re-persisted)
                # don't leak the per-brain temp .spkl. Kept on self so Brain.close() can run it early and
                # UNREGISTER it: the closure holds lc and lc.brain is this LiveBrain, so a registered hook
                # kept a closed brain alive until process exit (the amdahl census counted five of them).
                self._lakechat_close = lambda: lc.lake.close(remove=True)
                atexit.register(self._lakechat_close)
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
        the brain gets better at recognising what it is being told. G0(a): the label is AUTHORITATIVE, the
        same kind as correct() -- it goes through learn_turn(feedback=...), so it is keyed, a later self-label
        on the same text cannot overwrite it, and its older ring entries are rewritten. Returns the learner's
        pre-update answer. Requires interaction_learning=True. SAFETY: this learns INTENT RECOGNITION only; it
        drives no motion and adds no actuator route -- arm COMMANDS still flow through the NL firewall + gated
        arm in converse()."""
        if self._ixn is None:
            raise RuntimeError(
                "interaction learning is off; build LiveBrain(interaction_learning=True)"
            )
        return self._ixn.learn_turn(text, None, feedback=intent)

    def correct(self, text, intent):
        """A person's AUTHORITATIVE correction, learning intent recognition only.

        Never calls converse, the NL firewall's command path, the arm or the reflex/E-STOP.
        Returns the learner's pre-update answer.
        """
        if self._ixn is None:
            raise RuntimeError(
                "interaction learning is off; build LiveBrain(interaction_learning=True)"
            )
        if intent not in _IXN_INTENTS:
            raise ValueError(f"unknown correction intent: {intent!r}")
        return self._ixn.learn_turn(text, None, feedback=intent)

    def route(self, text):
        """The brain's LEARNED intent for a line of text, read out by spike timing from the interaction
        learner (improves as you teach() it). Requires interaction_learning=True; pure inference."""
        if self._ixn is None:
            raise RuntimeError(
                "interaction learning is off; build LiveBrain(interaction_learning=True)"
            )
        return self._ixn.respond(text_features(text, self._ixn.n_in))

    def chat(self, text, feedback=None, *, human=False):
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
        reply = self.converse(text, human=human)  # grounded reply (arm cmds stay gated in converse)
        self._ixn.learn_turn(
            text, _reply_to_intent(reply), feedback=feedback
        )  # spike-timing R-STDP turn
        return reply


def _ctx(v, tmp, au, d):
    return {"visual": [v] * 4, "temperature": [tmp], "audio": [au] * 4, "distance": d}


def main():
    # T037a: this self-check drives a So101Arm through EMERGENCY_STOP / HOME, which now writes
    # the durable latch sidecar -- under a throwaway home, never the live brain's.
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _main()


def _main():
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
    assert "only from a human" in brain.converse("close the effector"), (
        "an arm command from a non-human producer (the daemon's ingest) must not actuate"
    )
    assert brain.converse("close the effector", human=True).startswith("Done"), (
        "converse must route a safe arm command from a human through the gate"
    )
    # T053-iii: the brain NAMES ITS OWN LEARNING RULES, read from the live stack -- the
    # attribution's second reader (seven probes had measured this question abstaining or
    # recalling an unrelated fragment). Both shapes asserted: stackless -> honest refusal;
    # with the production-shaped stack attached (exactly Brain's own attach) -> the
    # stack's roster, the composed istdp.synapse arm, and the R-STDP deposit rule.
    _prior_ixn = brain._ixn
    try:
        # no learner at all -> the refusal must name THIS state, not guess a learner exists
        r_none = brain.converse("what learning rules are active?")
        assert "no interaction learner" in r_none, (
            f"the learnerless refusal must name its own configuration ({r_none})"
        )
        from snn_data_lake.cortex.interaction_learning import SpikingInteractionLearner as _SIL
        from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack as _PRS

        brain._ixn = _SIL(["a", "b"], n_in=8, seed=0)  # the production shape: Brain attaches both
        r_nostack = brain.converse("what learning rules are active?")
        assert "no plasticity stack is attached" in r_nostack, (
            f"the stackless refusal must name the missing stack ({r_nostack})"
        )
        brain._ixn.stack = _PRS(n_neurons=2, seed=0)
        brain._ixn.interact([0.5] * 8, "a")  # one real deposit -> the composed arm STEPS
        r_rules = brain.converse("what learning rules are active?")
        print(f"           : 'what learning rules are active?' -> {r_rules[:400]}")
        assert "istdp.synapse" in r_rules and "R-STDP" in r_rules, (
            f"the rules reply must name the composed arm and the deposit rule ({r_rules})"
        )
        assert "inhibitory_stdp_ei_balance" in r_rules, (
            f"the rules reply must carry the stack's own roster ({r_rules})"
        )
        for _base_rule in (
            "short_term_dynamic_gain",
            "homeostatic_scaling",
            "intrinsic_plasticity",
        ):
            assert _base_rule in r_rules, (
                f"the reply must name the base-sandwich rule {_base_rule} -- its panel measured "
                f"three live rules omitted behind the placeholder ({r_rules})"
            )
        assert "<r_stdp_weight_update>" not in r_rules, (
            f"the internal sentinel must not leak into the spoken roster ({r_rules})"
        )
        # the deposit above stepped the composers, so the FIGURE-bearing branch must run
        # (the panel measured the never-stepped wording satisfying the old OR unconditionally)
        assert "composed steps this process" in r_rules and "cumulative |dw|" in r_rules, (
            f"the stepped branch's own counters must be exercised and reported ({r_rules})"
        )
        # the reply (and both refusals) must map to a real intent, never teach GREET (T039e)
        assert _reply_to_intent(r_rules) == "MEMORY" and _reply_to_intent(r_none) == "MEMORY", (
            "the rules reply must carry an intent mapping -- unmapped replies teach the router GREET"
        )
    finally:
        brain._ixn = _prior_ixn
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
    # A WIRING check, not a collapse gauge: the same `grasp` ctx is ticked 90 times, so latent_std()
    # (the std over the last 32 online latents) reads the variance hinge's drift on a constant input
    # (~0.03 at the defaults), not the encoder's spread over distinct inputs; its red is a model that
    # never updated (an empty buffer reads exactly 0.0). The collapse red lives where the input
    # varies -- tests/test_jepa_collapse.py, specs/022 evidence/jepa-collapse.md.
    jepa_std = jbrain.arm.jepa.latent_std()
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
        f"JEPA live  : world model sized={jepa_sized}, buffer std {jepa_std:.3f} on a fixed stimulus (wiring, not a collapse gauge), "
        f"reward-latent goal formed={jepa_goal_formed}; executive goal "
        f"{ACTIONS[jgoal] if jgoal is not None else None} set by latent planning; plan_latent -> option {lat_pick}"
    )
    assert jepa_sized and jepa_std > 0.0, (
        "the JEPA world model must be wired in and updating its latent buffer (a fixed-stimulus "
        "wiring check; the collapse red is tests/test_jepa_collapse.py)"
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
