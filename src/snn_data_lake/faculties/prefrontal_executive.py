"""
Prefrontal central executive -- a GENUINELY SPIKING working-memory + decision core producing a
top-down bias over goals (the "center of thoughts and reasoning").

The prefrontal cortex holds a GOAL in working memory as PERSISTENT SPIKING activity, DECIDES by a
spiking winner-take-all competition between option pools, and broadcasts a TOP-DOWN BIAS that
downstream action selection consumes. Neither the decision nor the memory is a rate-level shortcut:
both are built from leaky-integrate-and-fire (LIF) neurons that leak, integrate current, cross a
threshold, emit a discrete spike, and reset.

Mechanism (spiking, deterministic, per-instance rng):
  1. DECISION = a Wang-2002-style spiking decision ATTRACTOR (replaces the old LCA drift-diffusion
     race). Each option is a small pool of LIF neurons with recurrent self-excitation; one shared
     inhibitory LIF pool receives excitation from all option pools and projects inhibition back onto
     all of them -- winner-take-all competition. Each option's evidence is injected as an external
     input current. The pool that first drives its population firing rate to a sustained high level
     (crosses `bound`) WINS; the step index at that crossing is the reaction time. A shared feed-
     forward normalization current (graded in the total evidence) makes a WIDER evidence gap commit
     SOONER (the chronometric signature) and a narrower gap commit later. deliberate() runs one such
     spiking decision and returns (winner, rt).
  2. WORKING MEMORY = a persistent-activity bistable spiking ATTRACTOR. A held goal is a pool of LIF
     neurons locked into a self-sustaining high-firing state: a recurrent self-excitation drive keeps
     the pool SPIKING across an input-free delay (maintain() advances one delay tick with NO external
     input and the pool still emits spikes). The `recurrence` gain sets how strongly the drive self-
     sustains: near 1 the firing persists (slow metastable decay); at 0 the recurrent current vanishes
     and the pool falls silent within a tick. A frame whose drive falls below `hold_threshold` is
     released (its pool stops sustaining) and the goal fades / pops back to its parent.
  3. Gated updating (stability vs flexibility). update() decides a stable winner (majority over
     commit_races spiking decisions), then GATES the write into working memory: it overwrites the held
     goal ONLY if the winner's raw evidence margin over the best alternative exceeds `gate_threshold`;
     otherwise the held goal PERSISTS. Weak new evidence cannot dislodge a maintained goal (stability);
     strong new evidence can (flexibility).
  4. Top-down bias. top_down_bias(strength) emits a length-n_options preference vector: `strength`
     summed on every held goal (primary + sub-goals) and zero elsewhere -- the executive's top-down
     vote over options that action selection can read (all zeros when nothing is held).
  5. Model-based deliberation (lookahead). plan(state, options, predictor, goal_value, horizon) scores
     each candidate by rolling a forward model `predictor.predict(s, command)` forward `horizon` steps
     and evaluating `goal_value` on the terminal predicted state, then commits via the SAME spiking
     decision attractor (over the rollout values). A short horizon is reactive (the immediate one-step
     consequence); a long horizon is planned (the terminal consequence), so lookahead can reject a
     high-immediate-value TRAP that rolls into a bad state. The predictor is injected (any object with
     predict(state, command)) -- the class does not couple to a specific forward model.
  6. Richer working memory (goal hierarchy). Working memory is a bounded goal STACK of frames, not a
     single scalar. update() sets/gates the primary (base) goal; push_subgoal(evidence) decides and
     pushes a sub-goal above it. maintain() decays and re-runs the TOP frame's spiking pool only (lower
     goals are HELD frozen while a sub-goal is pursued) and pops a faded frame, popping a sub-goal back
     to its parent. The single-goal API is unchanged.

Fair control (un-riggable):
  - persistence vs decay: after a commit, several maintain() delay ticks with NO input still leave the
    WM pool SPIKING and the goal held; an otherwise-identical executive with recurrence=0 loses the
    firing (and the goal) within a tick. The recurrent self-excitation sustains the memory, not input.
  - stability vs flexibility: holding goal A, WEAK evidence for B (margin below the gate) leaves the
    goal at A; STRONG evidence for B (margin above the gate) switches it to B.
  - decision is spiking: over many trials the committed option is the one whose LIF pool wins the WTA
    competition (the highest-evidence option), and a wider evidence gap commits FASTER (shorter rt).
  - determinism: two executives with the same seed reproduce identical goals, biases and reaction
    times. Across seeds only the REACTION TIME moves -- measured over seeds 0..199 the goals, WTA
    winner and bias are identical on every seed, so the choices are evidence-driven, not RNG-driven.

Honest framing: a reduced, genuinely-spiking model of the prefrontal central executive -- persistent-
activity working memory (a bistable LIF attractor that keeps firing through a delay), spiking WTA
decision-making (Wang-2002-style attractor), gated updating, and a top-down bias over goals. It HOLDS
and UPDATES a goal and biases behaviour; it is NOT cognition, planning-in-depth, or consciousness.
"Reasoning" here means a spiking competition resolved to a commitment, nothing more. Honest gap: the
persistent state is metastable -- its decay time-constant is set by the `recurrence` self-excitation
gain rather than emerging from spike-frequency adaptation -- so a long enough delay lets it drift
below threshold (which is what pops a stale sub-goal); the firing itself is real LIF spiking throughout.

THE SYNAPTIC RETENTION/RELEASE PAIR (specs/022 T056, ported from the retired
`plasticity/wm_attractor.py` -- P2's port-then-retire; exactly one WM mechanism survives).
Every goal frame now carries a Tsodyks-Markram synapse (`short_term_plasticity.DynamicSynapse`,
imported never re-implemented; the elapsed-time bookkeeping is this module's, the arithmetic
stays the incumbent's -- the same split T055 disclosed). Three decoupled clocks, three jobs:
  - `recurrence` (per maintained tick, TOP frame only) is the COMMITMENT policy -- unchanged
    incumbent semantics: a pursued goal's drive fades and pops on `hold_threshold`.
  - facilitation u on `tau_f` is RETENTION: the frame's memory, written ONLY by spikes (a
    maintain tick that emits no spikes refreshes nothing), decaying between refreshes. A
    background frame whose decayed u can no longer ignite its pool is DEAD and is CULLED --
    the fix for T055's measured confound, where frozen lower frames survived any delay for
    free and the incumbent's hit count ROSE with distractor count.
  - resources R on `tau_d` are RELEASE, and they BOUND THE REFRESH CYCLE: one rehearsal
    channel serves background frames round-robin, a frame holds the channel only while its
    resources still ignite the pool (the dwell), and depletion is what yields the channel
    (the handoff). `neuter("resources")` cuts R's route to the serve current: the dwell
    never ends, the channel hogs one frame, and every other background frame starves -- the
    reachable red that makes R's charge a measurement here, where T055 measured it
    saturated away at wm_attractor's own operating point.
CAPACITY FALLS OUT of those constants instead of being declared: `max_goals` now DEFAULTS
to the derived K = T_max/t_s + 1 (`faculties/wm_capacity.derive_capacity`, measured on
this module's own frames; +1 because the active frame rides its own per-tick refresh
outside the rotation). An explicit `max_goals` argument is still honored (callers own their
caps); the DEFAULT is no longer a picked 4. Derivation runs AT CONSTRUCTION (memoized on
every determinant of the probe -- U, taus, seed, n_options, membrane), so a refusal lands
in __init__, never inside the first push_subgoal. The rehearsal gain 1.1 is a CHOSEN
operating point, disclosed with its cliff in wm_capacity's docstring.

FOLD DISCLOSURES (panel, 2026-09-01), each measured:
  - VIABILITY IS REFUSED AT CONSTRUCTION, not discovered as a zombie store: a U at or
    below this instance's ignition floor (the floor moves with leak/gain and, at the
    cliff, with seed: 0.155-0.178 over 400 seeds; default U=0.2 clears seed 0 by 27%)
    would cull every committed goal on its first maintain tick. The old code held at any
    leak because nothing gated retention; the port gates it, so the non-viable corner of
    the (U, leak) plane is now an explicit ValueError instead of a silent hold -- callers
    in-tree all use the defaults, which are viable.
  - THE DERIVED-DEFAULT PATH REFUSES A NON-DEFAULT MEMBRANE (leak != 0.2): the derivation
    is validated at md=0.8 only (drift 0 there, ~2 at md=0.9, ~6 at md=0.95).
  - P3, said plainly: the rehearsal channel rides maintain(), whose live-caller census is
    unchanged by this port -- offline loops and self-checks (engine/autonomous_loop's
    comment carries the measured 2026-07 revert of a maintain() wire: 0/200 outcome
    changes). The channel's charge is proven by its own neuter reds, not by a live tick
    consumer; wiring the stack into the live loop remains T060/T061 scope and P3 forbids
    force-wiring it here.
  - P1, the new constants classified: U/tau_f/tau_d defaults are the ported synapse's
    parameters (T055's disposition, regime- and viability-refused rather than trusted);
    the rehearsal gain is CHOSEN (cliff in wm_capacity); pointer/horizon values in
    self-check arms are harness bounds, not behaviour arbiters.
  - Top-frame pops do not re-anchor the rotation pointers; measured across a 360-config
    sweep this reorders service within a rotation (49/360) and changes held outcomes in
    0/360 -- benign, and said here rather than silently relied on.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.prefrontal_executive
"""

import math
import random

from snn_data_lake.faculties.wm_capacity import CapacityRefusal, derive_capacity, seed_viability
from snn_data_lake.plasticity.short_term_plasticity import DynamicSynapse

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "PrefrontalExecutive.top_down_bias": (
        "NOT_A_RULE",
        "returns a length-n_options preference vector summed over held goals. A read of current goal "
        "state, computing no update at all",
    ),
    "PrefrontalExecutive.update": (
        "NOT_A_RULE",
        "decides over evidence and GATES the write into working memory. Control flow over a memory "
        "slot, caught on the verb `update`",
    ),
}


_REFRESH_GAIN = 1.1  # rehearsal-channel gain: the ONE chosen operating point (cliff in wm_capacity)
# (U, tau_f, tau_d, seed, n_options, md) -> derived max_goals. The key carries EVERY
# determinant the probe frame reads: seed AND n_options fix _wm_bias (the het stream
# burns 5*n_options decision-bias draws first), md carries `leak`. The first key dropped
# leak and n_options, and the panel measured the consequence: byte-identical constructors
# answered 97 or raised, depending on what else the process had constructed first.
_DERIVED_CAP = {}


# distinct-from: growth/lif_neuron.py / growth/alif_neuron.py -- per-neuron OBJECTS with their
#   own state; this module has always inlined its pools as raw membrane arrays (the decision
#   attractor at _decide_spiking and _run_wm_pool both do), and this helper is that existing
#   inline loop EXTRACTED so maintain() and the capacity probe run one arithmetic, not two.
#   Sites read.
# distinct-from: faculties/evidence_accumulation.py -- a decision race over currents; this
#   is the WM pool micro-step only. Site read.
def _lif_pool_run(Vs, cur, bias, md, steps):
    """Advance one LIF pool `steps` micro-steps at input current `cur` (threshold 1.0, reset
    0.0), mutating Vs. Extracted from _run_wm_pool so the rehearsal channel and the capacity
    derivation measure exactly what maintain() executes."""
    spikes = 0
    n = len(Vs)
    for _ in range(steps):
        for k in range(n):
            v = md * Vs[k] + cur + bias[k]
            if v >= 1.0:
                spikes += 1
                v = 0.0
            Vs[k] = v
    return spikes


# distinct-from: plasticity/wm_attractor.py (RETIRED by this change, T056) -- its _decayed
#   pure-read + last-touch bookkeeping is PORTED here as the unique capability P2 requires
#   ported before deletion; the synapse arithmetic stays short_term_plasticity's. Site read.
class _RehearsalFrame:
    """One goal frame's synaptic retention/release state plus the rehearsal arithmetic.

    Wraps ONE Tsodyks-Markram synapse (imported, never re-implemented) with the elapsed-time
    bookkeeping DynamicSynapse deliberately lacks (it decays only inside spike(dt) -- the same
    honest split wm_attractor disclosed, ported with the mechanism). This class is both the
    live rehearsal unit maintain() serves AND the probe wm_capacity measures -- by
    construction the same code."""

    __slots__ = ("syn", "last", "Vs", "_bias", "_md", "_n", "_steps", "_gain")

    def __init__(self, syn, bias, md, n, steps, gain=_REFRESH_GAIN):
        self.syn = syn
        self.last = 0.0
        self.Vs = [0.0] * n
        self._bias, self._md, self._n, self._steps, self._gain = bias, md, n, steps, gain

    def decayed(self, now):
        """(u, R) decayed to `now` -- a pure read, no state written (a read that refreshed the
        trace would be the instrument writing what it reports)."""
        s = self.syn
        gap = now - self.last
        if gap <= 0:
            return s.u, s.R
        u = s.u * math.exp(-gap / s.tau_f)
        r = 1.0 - (1.0 - s.R) * math.exp(-gap / s.tau_d)
        return u, r

    def ignites(self, u):
        """Does a FRESH pool ignite at facilitation u and FULL resources? The boundary this
        test draws is the derived retention floor -- no picked constant."""
        return _lif_pool_run([0.0] * self._n, self._gain * u, self._bias, self._md, self._steps) > 0

    def retention_dead(self, now):
        """The memory has decayed past its own ignition floor: nothing can read it back."""
        u, _r = self.decayed(now)
        return not self.ignites(u)

    def event(self, now):
        """One spike-driven refresh event THROUGH the imported synapse (facilitate + deplete)."""
        self.syn.spike(max(now - self.last, 1.0))
        self.last = now

    def serve(self, now, resources_live=True):
        """One rehearsal tick: run the pool at the synapse-scaled current; a firing pool writes
        one refresh event. 'fired' holds the channel; 'depleted' (resources can no longer
        ignite, facilitation still could) yields it; 'dead' (facilitation itself below the
        floor) says the memory is gone. resources_live=False is neuter('resources'): R's only
        route into the serve current is cut."""
        u, r = self.decayed(now)
        cur = self._gain * u * (r if resources_live else 1.0)
        if _lif_pool_run(self.Vs, cur, self._bias, self._md, self._steps) > 0:
            self.event(now)
            return "fired"
        return "depleted" if self.ignites(u) else "dead"


class PrefrontalExecutive:
    """Central executive over n_options goals: a persistent-activity SPIKING working memory, gated
    updating, and a spiking winner-take-all decision attractor emitting a top-down bias.

    Hold a goal across a delay with maintain() (the WM pool keeps spiking); update it by gated
    deliberation with update(); read the raw spiking commitment with deliberate(); read the held goal
    via .goal; and read the top-down preference over options with top_down_bias()."""

    def __init__(
        self,
        n_options,
        gate_threshold=0.5,
        recurrence=0.98,
        hold_threshold=0.2,
        bound=1.0,
        noise=1.0,
        dt=0.01,
        leak=0.2,
        inhibition=0.1,
        commit_races=25,
        max_goals=None,
        seed=0,
        U=0.2,
        tau_f=1500.0,
        tau_d=200.0,
    ):
        if tau_f <= tau_d:
            raise ValueError(
                f"synaptic WM requires tau_f > tau_d (facilitation must outlive depression, "
                f"or nothing holds the memory); got tau_f={tau_f} tau_d={tau_d}"
            )
        self.n_options = n_options
        self.gate_threshold = gate_threshold  # evidence margin needed to overwrite the held goal
        self.recurrence = recurrence  # WM recurrent self-excitation gain (~1 -> persistent)
        self.hold_threshold = hold_threshold  # persistent drive fades (pool released) below this
        self.bound = bound  # spiking decision bound (sustained-rate threshold)
        self.noise = noise  # background input noise on the LIF pools
        self.dt = dt  # nominal integration time-step (LIF steps are discrete)
        self.leak = leak  # LIF membrane leak (sets the membrane decay factor)
        self.inhibition = inhibition  # shared-inhibition strength (WTA competition)
        self.commit_races = commit_races  # spiking decisions majority-voted into a stable commit
        self._max_goals_arg = max_goals  # None -> DERIVED from the synaptic constants (T056)
        self.seed = seed
        self.U = U  # T-M baseline utilization (the ported pair's parameters)
        self.tau_f = tau_f  # facilitation / RETENTION time constant
        self.tau_d = tau_d  # depression / RELEASE (resource-recovery) time constant
        self.rng = random.Random(seed)
        self._stack = []  # goal stack: [[index, drive, Vs, rehearsal-frame], ...] base .. top
        self._wm_spikes = 0  # spikes emitted by the active WM pool on the last maintain()
        self._t = 0.0  # the maintenance clock (one tick per maintain())
        self._focus = 0  # rehearsal-channel rotation pointer over background frames
        self._sweep = 0  # death-bookkeeping pointer -- SEPARATE from _focus, see _serve_background
        self._resources_live = True  # neuter("resources") cuts R's route to the serve current
        self._rehearsal_live = True  # neuter("rehearsal") cuts the refresh, keeps the bookkeeping

        # ---- spiking (LIF) constants; dimensionless, discrete-step. threshold=1.0, reset=0.0 -------
        md = 1.0 - leak
        self._md = 0.5 if md < 0.5 else (0.95 if md > 0.95 else md)  # membrane decay (from leak)
        # decision attractor (Wang-2002 WTA): one LIF pool per option + a shared inhibitory pool
        self._n_dec = 5  # LIF neurons per option pool
        self._n_inh = 3  # shared inhibitory pool size
        self._t_max = 200  # max integration steps per spiking decision
        self._g_ext = 0.55  # external evidence -> input current gain
        self._b0 = 0.02  # baseline input current
        self._w_plus = 0.50  # recurrent self-excitation within an option pool
        self._w_inh = 4.0 * inhibition  # shared inhibitory pool -> option pools (recurrent WTA)
        self._w_ie = 1.2  # option pools -> inhibitory pool excitation
        self._w_cross = 0.50  # feed-forward cross-inhibition (immediate, chronometric)
        self._init_v = 0.10  # random initial membrane charge (seed-dependent start)
        self._theta = 0.26 * bound  # sustained-firing-rate threshold to declare a winner
        self._trace_decay = 0.90  # rate-proxy (low-pass firing) smoothing
        self._noise_amp = 0.05 * noise  # background noise amplitude
        # working-memory attractor (persistent-activity bistable LIF pool)
        self._n_wm = 6  # LIF neurons in a working-memory pool
        self._wm_steps = 8  # LIF micro-steps advanced per maintain() delay tick
        self._wm_gain = 0.55  # recurrent drive -> input current gain
        # fixed per-neuron heterogeneity (deterministic; async firing without per-step randomness)
        het = random.Random((seed * 2654435761 + 1) & 0xFFFFFFFF)
        self._dec_bias = [0.05 * (het.random() - 0.5) for _ in range(self._n_dec * n_options)]
        self._wm_bias = [0.03 + 0.04 * het.random() for _ in range(self._n_wm)]
        # ---- T056 construction-time refusals (panel fold; loud, never silent) ---------------------
        # Seed viability: frames are seeded at u=U, and the retention cull removes a frame
        # whose decayed u cannot ignite. A U at or below THIS instance's floor (the floor
        # moves with leak/gain/seed) means every committed goal dies on its first maintain
        # tick with the pool still firing -- the panel measured exactly that silently at
        # U=0.15, and at the DEFAULT U with leak>=0.3. Refused here, like tau_f <= tau_d.
        floor, viable = seed_viability(self._probe_frame, self.U)
        if not viable:
            raise ValueError(
                f"synaptic WM cannot hold at these constants: the frame seed u=U={U} sits "
                f"at or below this instance's ignition floor {floor:.4f} (set by "
                f"leak={leak}, gain={_REFRESH_GAIN}, seed={seed}) -- every committed goal "
                f"would be culled on its first maintain tick. Raise U or lower leak."
            )
        if max_goals is None:
            _ = self.max_goals  # derive NOW: every derived-path refusal (regime, seed
            # viability, membrane band -- the band check lives in the GETTER so the
            # setter's `p.max_goals = None` route cannot bypass it, fold panel) lands at
            # construction, not inside the first push_subgoal (total at HEAD, stays total)

    @property
    def max_goals(self):
        """Bounded goal-stack depth (primary + sub-goals). Explicitly passed values are honored
        verbatim; the DEFAULT is DERIVED -- K = T_max/t_s from this instance's own synaptic
        constants, +1 for the active frame (specs/022 T056; the old declared 4 was
        unfalsifiable). Lazy and memoized: live paths that never stack goals never derive."""
        if self._max_goals_arg is not None:
            return self._max_goals_arg
        # The derived-default path is validated ONLY at the default membrane: measured
        # derived-vs-held drift is 0 at md=0.8, ~2 at md=0.9, ~6 at md=0.95. The check
        # sits HERE, on every derived read, so no route into the derived path (the
        # constructor's eager read or the setter's `p.max_goals = None`) can bypass it.
        if self._md != 0.8:
            raise CapacityRefusal(
                f"the DERIVED default max_goals is validated only at the default "
                f"membrane (leak=0.2 -> md=0.8; got md={self._md}); measured "
                f"derived-vs-held drift reaches 6 frames at md=0.95. Pass max_goals "
                f"explicitly for a non-default membrane."
            )
        key = (self.U, self.tau_f, self.tau_d, self.seed, self.n_options, self._md)
        if key not in _DERIVED_CAP:
            _DERIVED_CAP[key] = (
                derive_capacity(self._probe_frame, self.tau_f, seed_u=self.U)["K"] + 1
            )
        return _DERIVED_CAP[key]

    @max_goals.setter
    def max_goals(self, value):
        self._max_goals_arg = value

    def _probe_frame(self):
        """A fresh rehearsal frame over THIS instance's constants -- the unit wm_capacity
        measures, and exactly the unit maintain() serves (same class, same arithmetic)."""
        return _RehearsalFrame(
            DynamicSynapse(self.U, self.tau_f, self.tau_d),
            self._wm_bias,
            self._md,
            self._n_wm,
            self._wm_steps,
        )

    # distinct-from: plasticity/wm_attractor.py (retired, T056) -- this is its neuter idiom
    #   ported with the mechanism; verdict vocabulary stays rule_composer's. Site read.
    # distinct-from: faculties/microcircuit.py::neuter and plasticity/rule_composer.py::
    #   neutered -- same IDIOM, different substrate: a neuter has to know its own internals,
    #   which is why the tree keeps one per module. Sites read.
    def neuter(self, element):
        """Neuter ONE named element IN PLACE -- the per-element attribution arm (P8 #3).

          "resources" -- cuts R's ONLY route into the rehearsal current: the serve step
                         drives the pool as if resources were always full, so the dwell
                         never ends, the channel hogs one background frame, and every
                         other background frame starves. R's charge, as a reachable red.
          "rehearsal" -- cuts the refresh half of the channel while KEEPING the death
                         bookkeeping (frames are still culled when their memory decays
                         past the ignition floor): background memories then die on tau_f.
                         The channel's charge, as a reachable red.

        Returns self so an arm can chain. Refuses an unknown element rather than silently
        neutering nothing -- a no-op neuter reads as a passing control."""
        if element == "resources":
            self._resources_live = False
        elif element == "rehearsal":
            self._rehearsal_live = False
        else:
            raise ValueError(
                f"unknown element {element!r}: this store's neuterable elements are "
                f"'resources' (R's route into the rehearsal current) and 'rehearsal' "
                f"(the refresh half of the background channel)"
            )
        return self

    @property
    def goal(self):
        """The currently held goal index (top of the goal stack), or None if nothing is maintained."""
        return self._stack[-1][0] if self._stack else None

    @property
    def goals(self):
        """The full held goal hierarchy, base -> top, as a list of indices (empty if none held)."""
        return [frame[0] for frame in self._stack]

    # ---- spiking decision attractor ---------------------------------------------------------------
    def _decide_spiking(self, evidence):
        """One Wang-2002-style spiking winner-take-all decision over `evidence` (>= 0 drive per option).

        Each option is a recurrently self-exciting LIF pool; a shared inhibitory LIF pool drives the
        competition. The first option pool whose population firing rate reaches a sustained high level
        (crosses `bound`) WINS; a wider evidence gap commits sooner. Returns (winner_index, rt_steps)."""
        n = len(evidence)
        e_ref = 0.0
        for e in evidence:
            if e > e_ref:
                e_ref = e
        if e_ref <= 0.0:
            return 0, 1  # no evidence -> no race (tie -> index 0)
        scaled = [(e if e > 0.0 else 0.0) / e_ref for e in evidence]
        sum_scaled = 0.0
        for s in scaled:
            sum_scaled += s
        nd, ni, md = self._n_dec, self._n_inh, self._md
        g_ext, b0, w_plus = self._g_ext, self._b0, self._w_plus
        w_inh, w_ie, w_cross = self._w_inh, self._w_ie, self._w_cross
        td, theta, namp = self._trace_decay, self._theta, self._noise_amp
        dec_bias = self._dec_bias
        rnd = self.rng.random
        # feed-forward cross-inhibition: each pool is suppressed by the MEAN drive of the others, so a
        # wider evidence gap gives the leader a higher net current from step 1 -> a faster commit.
        denom = (n - 1) if n > 1 else 1
        cross = [w_cross * (sum_scaled - scaled[i]) / denom for i in range(n)]
        # LIF membranes (seed-dependent random initial charge -> the seed changes the whole trajectory)
        init_v = self._init_v
        Vd = [[init_v * rnd() for _ in range(nd)] for _ in range(n)]
        Vi = [init_v * rnd() for _ in range(ni)]
        trace = [0.0] * n  # per option-pool firing-rate proxy
        trace_i = 0.0  # inhibitory-pool firing-rate proxy
        winner = None
        rt = self._t_max
        for t in range(1, self._t_max + 1):
            inh = w_inh * trace_i  # shared spiking inhibition this step
            spk = [0] * n
            for i in range(n):
                cur = (
                    g_ext * scaled[i]
                    - cross[i]
                    + w_plus * trace[i]
                    - inh
                    + b0
                    + namp * (rnd() - 0.5)
                )
                base = i * nd
                col = Vd[i]
                s = 0
                for k in range(nd):
                    v = md * col[k] + cur + dec_bias[base + k]  # LIF: leak + integrate current
                    if v >= 1.0:  # threshold -> discrete spike -> reset
                        s += 1
                        v = 0.0
                    col[k] = v
                spk[i] = s
            exc = w_ie * (
                trace[0] if n == 1 else sum(trace)
            )  # option pools excite the inhibitory pool
            si = 0
            for k in range(ni):
                v = md * Vi[k] + exc + namp * (rnd() - 0.5)
                if v >= 1.0:
                    si += 1
                    v = 0.0
                Vi[k] = v
            for i in range(n):
                trace[i] = td * trace[i] + (1.0 - td) * (spk[i] / nd)
            trace_i = td * trace_i + (1.0 - td) * (si / ni)
            best = _argmax(trace)
            if trace[best] >= theta:  # first pool to sustained high firing wins
                winner = best
                rt = t
                break
        if winner is None:
            winner = _argmax(trace)  # timed out -> the strongest pool so far
        return winner, rt

    def deliberate(self, evidence):
        """Raw spiking commitment over `evidence` (one input drive per option), NO WM write.

        Runs the spiking winner-take-all decision attractor: evidence maps onto each option pool's
        external input current. Returns (goal, rt) -- the winning option and the reaction-time steps."""
        return self._decide_spiking([float(e) for e in evidence])

    def _stable_commit(self, evidence):
        """Decide a STABLE commitment for a working-memory write: run commit_races spiking decisions and
        take the majority winner (a WM write integrates more evidence than a single reflexive decision).
        Returns the majority-winning option index."""
        votes = [0] * len(evidence)
        for _ in range(self.commit_races):
            winner, _rt = self.deliberate(evidence)
            votes[winner] += 1
        return _argmax(votes)

    def update(self, evidence):
        """Decide over `evidence`, then GATE the write into working memory (the PRIMARY goal).

        Overwrites the primary (base) goal only if the winner's evidence margin over the best
        alternative exceeds gate_threshold (or nothing is held yet); otherwise the held goal
        persists. Returns the current goal index."""
        winner = self._stable_commit(evidence)
        alternatives = [evidence[j] for j in range(len(evidence)) if j != winner]
        margin = evidence[winner] - (max(alternatives) if alternatives else float("-inf"))
        self._gate_primary(winner, margin)
        return self.goal

    def _gate_primary(self, idx, margin):
        """Gate the write of the PRIMARY (base) goal: overwrite it only if `margin` clears the gate
        threshold (or nothing is held yet), keeping any sub-goals. Shared by update() and plan()."""
        if not self._stack or margin >= self.gate_threshold:
            if self._stack:
                self._stack[0] = self._new_frame(idx)  # overwrite the primary goal (keep sub-goals)
            else:
                self._stack.append(self._new_frame(idx))  # first goal

    def _rollout_value(self, state, command, predictor, goal_value, horizon):
        """Roll `predictor.predict(s, command)` forward `horizon` steps from `state` (at least one --
        the immediate one-step consequence) and score the TERMINAL predicted state with goal_value."""
        s = state
        for _ in range(max(1, horizon)):
            s = predictor.predict(s, command)
        return goal_value(s)

    def plan(self, state, options, predictor, goal_value, horizon=1, write=True):
        """Model-based deliberation: score each candidate in `options` by a `horizon`-step rollout of
        `predictor` from `state`, evaluate goal_value on the terminal predicted state, then commit via
        the SAME spiking decision attractor (over the rollout values). horizon=1 is reactive (the
        immediate one-step consequence); horizon>=2 is planned (the terminal consequence), so lookahead
        can reject a high-immediate-value trap. If `write`, the commitment gate-updates the primary goal
        exactly like update(). Returns (goal_index, mean_rt)."""
        values = [self._rollout_value(state, o, predictor, goal_value, horizon) for o in options]
        lo = min(values)
        drifts = [v - lo + 0.1 for v in values]  # shift rollout scores to positive input drives
        votes = [0] * len(options)
        rts = []
        for _ in range(self.commit_races):
            w, rt = self.deliberate(drifts)
            votes[w] += 1
            rts.append(rt)
        goal = _argmax(votes)
        if write:
            alternatives = [drifts[j] for j in range(len(drifts)) if j != goal]
            margin = drifts[goal] - (max(alternatives) if alternatives else float("-inf"))
            self._gate_primary(goal, margin)
        return goal, sum(rts) / len(rts)

    def push_subgoal(self, evidence):
        """Decide a stable commitment over `evidence` and push it as a SUB-GOAL above the primary.

        The sub-goal becomes the active (top) goal; the parent is HELD (frozen) beneath it until the
        sub-goal fades and pops back to it. Bounded by max_goals (drops the oldest frame when full).
        Returns the current (sub-)goal index."""
        winner = self._stable_commit(evidence)
        if len(self._stack) >= self.max_goals:
            self._stack.pop(0)  # bounded WM: drop the oldest (base) frame
            # a removal at background index 0 slides every frame down one slot; the
            # rotation pointers must follow or they skip a frame's turn (the same
            # adjustment _serve_background makes for its own culls -- panel fold)
            if self._focus > 0:
                self._focus -= 1
            if self._sweep > 0:
                self._sweep -= 1
        self._stack.append(self._new_frame(winner))
        return self.goal

    def _new_frame(self, index):
        """A fresh working-memory frame: [goal index, self-excitation drive=1.0, LIF membranes,
        rehearsal frame]. The synapse is seeded at u=U -- the committing decision itself was a
        spiking event (a whole WTA race fired to choose this goal), so the first facilitation
        is spike-justified, not free."""
        rf = self._probe_frame()
        rf.syn.u = self.U
        rf.last = self._t
        return [index, 1.0, [0.0] * self._n_wm, rf]

    # distinct-from: faculties/task_set.py -- that module deliberately runs THE SAME LIF equation
    #   (its own docstring cites PrefrontalExecutive._run_wm_pool) on its own active-set pool;
    #   this is the incumbent method it cites, now delegating to the one extracted loop above.
    #   Site read.
    def _run_wm_pool(self, frame):
        """Advance a frame's persistent-activity LIF pool `_wm_steps` micro-steps with NO external
        input -- only the recurrent self-excitation drive. Returns the number of spikes emitted (the
        persistent firing that sustains the goal across the delay). Mutates the frame's membranes."""
        cur = self._wm_gain * frame[1]  # recurrent drive -> input current (no ext input)
        return _lif_pool_run(frame[2], cur, self._wm_bias, self._md, self._wm_steps)

    def maintain(self):
        """Advance one delay tick with NO input. The recurrent self-excitation drive on the TOP goal
        decays by `recurrence` and its LIF pool is re-run -- so a held goal keeps SPIKING
        (persistent activity); a top frame whose drive falls below hold_threshold is released and
        popped (a sub-goal pops back to its parent, the last goal fades to None). The top frame's
        own spiking REFRESHES its synapse (spike-driven: a silent tick writes nothing).

        Lower goals are no longer held frozen for free (T055 measured that freeze as a confound):
        one resource-gated rehearsal channel serves them round-robin -- see _serve_background --
        and a frame whose facilitation has decayed past its ignition floor is CULLED, including a
        parent that comes back up dead after a long reign. Returns the currently held goal, or
        None once the stack empties."""
        self._t += 1.0
        while self._stack and self._stack[-1][3].retention_dead(self._t):
            self._stack.pop()  # a frame whose memory died cannot resume as the active goal
        if not self._stack:
            self._wm_spikes = 0
            return None
        top = self._stack[-1]
        top[1] *= self.recurrence  # recurrent drive decays across the input-free delay
        self._wm_spikes = self._run_wm_pool(top)  # persistent LIF spiking on the surviving drive
        if self._wm_spikes:
            top[3].event(self._t)  # the active goal's own spikes refresh its synapse
        if top[1] < self.hold_threshold:  # drive too weak to self-sustain -> release goal
            self._stack.pop()
        self._serve_background()
        return self.goal

    def _serve_background(self):
        """One tick of the rehearsal channel over the background (non-top) frames.

        The rotation pointer visits one frame per tick. A frame whose memory has decayed past
        its ignition floor is culled (pure read -- this bookkeeping survives neuter('rehearsal'),
        so a cut channel makes deaths VISIBLE rather than hiding them). Otherwise, when the
        refresh is live, the frame is SERVED: it holds the channel while its resources still
        ignite the pool (the dwell) and yields it on depletion (the handoff) -- resources are
        what bound the refresh cycle, which is why a capacity falls out of the time constants
        (K = T_max/t_s, derived and guarded in faculties/wm_capacity)."""
        bg = self._stack[:-1]
        if not bg:
            return
        # Death bookkeeping on its OWN pointer. The first build tied it to the serve pointer,
        # and the R-neuter red exposed that as a defect: a hogged channel froze the pointer,
        # so the starvation the hog caused stayed INVISIBLE (live 6 == neutered 6) -- the
        # detector was parked exactly where the failure could not appear. A dead frame is
        # culled by the sweep within one rotation regardless of what the serve channel does.
        self._sweep %= len(bg)
        sw = bg[self._sweep]
        if sw[3].retention_dead(self._t):
            # A removal below a pointer slides every later frame down one slot; an unadjusted
            # pointer then skips a frame's turn. ROTATION FAIRNESS, said precisely (the fold
            # panel caught this comment claiming the 97-vs-90 capacity gap, which belonged
            # to the tau_d regime finding -- ablating this adjustment alone does not move
            # the guard-1 result): the re-anchor keeps service order stable under removals;
            # its outcome effect is pointwise-rare, and tests exercise the branch directly.
            self._stack.remove(sw)  # the memory is gone; reporting the goal would be confabulation
            if self._sweep < self._focus:
                self._focus -= 1
            bg = self._stack[:-1]
            if not bg:
                return
        else:
            self._sweep += 1
        if not self._rehearsal_live:  # neuter("rehearsal"): bookkeeping sweeps, refresh is cut
            return
        self._focus %= len(bg)
        fr = bg[self._focus]
        verdict = fr[3].serve(self._t, resources_live=self._resources_live)
        if verdict == "depleted":
            self._focus += 1  # depleted resources yield the channel: the handoff
        elif verdict == "dead":
            if self._focus < self._sweep:
                self._sweep -= 1
            self._stack.remove(fr)  # died since the sweep's last visit; cull now, not next pass

    def top_down_bias(self, strength=1.0):
        """A length-n_options top-down preference vector: `strength` SUMMED over every held goal
        (primary + sub-goals) and 0.0 elsewhere -- the whole active goal hierarchy votes. All zeros
        when nothing is held."""
        bias = [0.0] * self.n_options
        for frame in self._stack:
            bias[frame[0]] += strength
        return bias


def _argmax(values):
    """Index of the largest value (tie-break by lowest index)."""
    return max(range(len(values)), key=lambda i: (values[i], -i))


def _mean_rt(pfc, evidence, n):
    """Mean spiking-decision rt (steps) over n raw deliberations of `evidence`."""
    return sum(pfc.deliberate(evidence)[1] for _ in range(n)) / n


def main():

    from snn_data_lake.faculties.forward_model import ForwardModel  # the injected lookahead model

    n_opt = 4

    # ---- 1. working-memory persistence: PERSISTENT SPIKING vs crippled maintenance ----------------
    pfc = PrefrontalExecutive(n_options=n_opt, seed=0)
    pfc.update([2.0, 0.1, 0.1, 0.1])  # strong evidence for option 0 -> commit goal
    committed = pfc.goal
    held_trace = []
    wm_spike_trace = []
    for _ in range(6):  # delay ticks with NO input
        held_trace.append(pfc.maintain())  # the WM pool keeps firing (persistent activity)
        wm_spike_trace.append(pfc._wm_spikes)  # discrete spikes emitted this input-free tick
    persisted = pfc.goal

    crippled = PrefrontalExecutive(n_options=n_opt, recurrence=0.0, seed=0)  # no recurrence
    crippled.update([2.0, 0.1, 0.1, 0.1])
    crippled_trace = [crippled.maintain() for _ in range(3)]
    crippled_wm_spikes = crippled._wm_spikes  # the pool falls silent once the drive collapses
    decayed = crippled.goal

    # ---- 2. gated updating: stability (weak B) vs flexibility (strong B) ---------------------------
    gate = PrefrontalExecutive(n_options=3, seed=1)
    gate.update([2.0, 0.2, 0.2])  # commit goal A = 0
    goal_a = gate.goal
    gate.update([0.5, 0.7, 0.4])  # WEAK evidence for B (margin < gate) -> keep A
    after_weak = gate.goal
    gate.update([0.2, 2.0, 0.2])  # STRONG evidence for B (margin > gate) -> B
    after_strong = gate.goal

    # ---- 3. spiking decision: WTA winner + chronometric rt ordering --------------------------------
    delib = PrefrontalExecutive(n_options=3, seed=2)
    competing = [0.3, 0.8, 1.8]  # option 2 has the strongest evidence
    n_win = 60
    wins = [0, 0, 0]
    sample_rt = None
    for _ in range(n_win):
        g, rt = delib.deliberate(competing)
        wins[g] += 1
        if sample_rt is None:
            sample_rt = rt
    strongest = _argmax(competing)
    weak_rt = _mean_rt(delib, [0.6, 0.1, 0.1], 30)  # weak dominant evidence -> slow commitment
    strong_rt = _mean_rt(delib, [2.2, 0.1, 0.1], 30)  # strong dominant evidence -> fast commitment

    # ---- 4. top-down bias reflects the goal --------------------------------------------------------
    biaser = PrefrontalExecutive(n_options=n_opt, seed=3)
    bias_none = biaser.top_down_bias()
    biaser.update([2.0, 0.1, 0.1, 0.1])  # goal 0
    bias_0 = biaser.top_down_bias()
    biaser.update([0.1, 0.1, 0.1, 2.0])  # goal 3
    bias_3 = biaser.top_down_bias()

    # ---- 5. model-based lookahead: reactive picks the immediate TRAP; planning rejects it ----------
    # abstract plant s' = s + 0.2*(-0.5*s + u): a constant command u converges to steady state 2u.
    fm_trap = ForwardModel(a=-0.5, b=1.0, dt=0.2)
    target = 2.0

    def goal_value(s):  # closer to the target state -> higher value
        return -abs(s - target)

    options = [1.0, 5.0, 0.2]  # cmd 0 -> s*=2.0 (goal); cmd 1 -> s*=10 (trap)
    reactive = PrefrontalExecutive(n_options=len(options), seed=10)
    reactive_goal, reactive_rt = reactive.plan(0.0, options, fm_trap, goal_value, horizon=1)
    planned = PrefrontalExecutive(n_options=len(options), seed=10)
    planned_goal, planned_rt = planned.plan(0.0, options, fm_trap, goal_value, horizon=40)
    # one-step vs terminal values, to SHOW the trap is real (option 1 best now, worst later)
    immediate_vals = [reactive._rollout_value(0.0, o, fm_trap, goal_value, 1) for o in options]
    terminal_vals = [planned._rollout_value(0.0, o, fm_trap, goal_value, 40) for o in options]

    # ---- 5b. the lookahead composes with the REAL forward model + is deterministic -----------------
    fm_chk = ForwardModel(a=-0.5, b=1.0, dt=0.2)
    manual = 0.0
    for _ in range(3):  # the model's own forward chain, by hand
        manual = fm_chk.predict(manual, 1.0)
    via_rollout = PrefrontalExecutive(n_options=2, seed=0)._rollout_value(
        0.0, 1.0, fm_chk, lambda s: s, 3
    )

    def plan_scene(seed):
        p = PrefrontalExecutive(n_options=len(options), seed=seed)
        return p.plan(0.0, options, ForwardModel(a=-0.5, b=1.0, dt=0.2), goal_value, horizon=40)

    plan_det = plan_scene(9) == plan_scene(9)

    # ---- 5c. planning keeps the chronometric signature: a wider terminal gap commits faster --------
    wide = PrefrontalExecutive(n_options=2, seed=11)
    _wg, wide_rt = wide.plan(0.0, [1.0, 5.0], fm_trap, goal_value, horizon=40)  # terminals 2 vs 10
    narrow = PrefrontalExecutive(n_options=2, seed=11)
    _ng, narrow_rt = narrow.plan(
        0.0, [1.0, 1.05], fm_trap, goal_value, horizon=40
    )  # terminals 2 vs 2.1

    # ---- 6. richer working memory: a sub-goal over a primary, pops back on decay --------------------
    hier = PrefrontalExecutive(n_options=4, seed=8)
    hier.update([2.0, 0.1, 0.1, 0.1])  # primary goal 0
    primary_goal = hier.goal
    hier.push_subgoal([0.1, 0.1, 2.0, 0.1])  # sub-goal 2, pushed above the primary
    sub_goal = hier.goal
    hier_goals = list(hier.goals)  # [0, 2] -- the held hierarchy
    hier_bias = hier.top_down_bias()  # the whole hierarchy votes (nonzero on 0 and 2)
    popped_back = None  # the sub-goal decays and pops back to the parent
    for _ in range(400):
        g = hier.maintain()
        if len(hier.goals) == 1:
            popped_back = g
            break
    after_pop = list(hier.goals)

    print("=" * 72)
    print("PREFRONTAL EXECUTIVE -- spiking working memory + spiking WTA decision -> top-down bias")
    print("=" * 72)
    print(
        f"options: {n_opt} | gate: {pfc.gate_threshold} | recurrence: {pfc.recurrence} | "
        f"hold threshold: {pfc.hold_threshold} | LIF pool: {pfc._n_dec}/opt + {pfc._n_inh} inh"
    )
    print(
        f"commit goal {committed}; after 6 delay ticks (no input) -> still held: {persisted} "
        f"(trace {held_trace})"
    )
    print(
        f"persistent spiking: WM pool spikes per input-free tick {wm_spike_trace} "
        f"(crippled recurrence=0 -> {crippled_wm_spikes} spikes, trace {crippled_trace}, goal {decayed})"
    )
    print("-" * 72)
    print(
        f"gated updating: hold A={goal_a}; weak evidence for B -> {after_weak} (stays A); "
        f"strong evidence for B -> {after_strong} (switches to B)"
    )
    print("-" * 72)
    print(
        f"spiking WTA over evidence {competing} ({n_win} trials): win counts {wins} -> "
        f"strongest option {strongest} wins {wins[strongest]}/{n_win} (sample rt {sample_rt} steps)"
    )
    print(
        f"chronometric: weak-evidence rt {weak_rt:.1f} steps vs strong-evidence rt "
        f"{strong_rt:.1f} steps -> stronger commits faster"
    )
    print("-" * 72)
    print(f"top-down bias: no goal -> {bias_none}; goal 0 -> {bias_0}; goal 3 -> {bias_3}")
    print("-" * 72)
    print(
        f"lookahead   : options {options} immediate values {[round(v, 2) for v in immediate_vals]} "
        f"-> terminal values {[round(v, 2) for v in terminal_vals]}"
    )
    print(
        f"deliberation: reactive (h=1) commits option {reactive_goal} (the immediate-best TRAP); "
        f"planned (h=40) commits option {planned_goal} (the terminal-best)"
    )
    print(f"chronometric: wide terminal gap rt {wide_rt:.1f} < narrow gap rt {narrow_rt:.1f} steps")
    print(
        f"hierarchy   : primary goal {primary_goal}; push sub-goal -> held {hier_goals}, bias "
        f"{hier_bias}; sub-goal decays -> pops back to primary {after_pop}"
    )
    print("=" * 72)

    # ---- un-riggable self-checks (deterministic asserts) ------------------------------------------
    # (0) GENUINELY SPIKING: LIF pools, discrete spikes, persistent firing.
    assert all(isinstance(s, int) for s in wm_spike_trace), (
        "WM activity must be discrete spike counts"
    )
    assert all(s > 0 for s in wm_spike_trace), (
        f"the WM pool must keep EMITTING SPIKES across zero-input delay ticks ({wm_spike_trace})"
    )
    assert crippled_wm_spikes == 0, (
        f"with no recurrence the WM pool must fall silent (no persistent spiking), got {crippled_wm_spikes}"
    )
    assert isinstance(sample_rt, int) and 0 < sample_rt <= delib._t_max, (
        f"a spiking decision must resolve at a discrete reaction-time step ({sample_rt})"
    )

    # (1) persistence: recurrence sustains the goal across a delay; crippled maintenance loses it.
    assert committed == 0, "a strong evidence commitment must set the goal"
    assert all(g == 0 for g in held_trace), "the goal must persist across delay ticks (no input)"
    assert persisted == committed, "the recurrence must sustain the held goal across the delay"
    assert decayed is None, "with no recurrence the same goal must decay to None within a tick"

    # (2) stability vs flexibility: weak new evidence cannot overwrite; strong new evidence can.
    assert goal_a == 0, "the executive must first commit goal A"
    assert after_weak == goal_a, "weak new evidence (margin below gate) must NOT overwrite goal A"
    assert after_strong == 1 and after_strong != goal_a, (
        "strong new evidence (margin above gate) must switch the goal to B"
    )

    # (3) spiking WTA decision: strongest evidence wins the majority; stronger evidence is faster.
    assert wins[strongest] == max(wins) and wins[strongest] > n_win / 2, (
        f"the strongest-evidence option's LIF pool must win the WTA majority ({wins})"
    )
    assert strong_rt < weak_rt, (
        f"stronger evidence must commit faster ({strong_rt:.1f} vs {weak_rt:.1f} steps)"
    )

    # (4) top-down bias reflects the held goal, switches with it, and is all-zeros when none held.
    assert bias_none == [0.0] * n_opt, "no held goal must yield an all-zero top-down bias"
    assert _argmax(bias_0) == 0, "the top-down bias must peak on the held goal (0)"
    assert _argmax(bias_3) == 3, "the top-down bias must peak on the held goal (3)"
    assert bias_0 != bias_3, "the top-down bias must switch when the goal switches"
    assert biaser.top_down_bias(strength=2.0)[3] == 2.0, "bias strength must scale the held option"

    # (5) determinism: same seed -> identical goals, biases, rts. Across seeds only the rt moves.
    def scenario(seed):
        p = PrefrontalExecutive(n_options=3, seed=seed)
        out = [p.update([2.0, 0.2, 0.2]), p.maintain()]
        out.append(p.deliberate([0.5, 1.5, 0.3]))
        out.append(tuple(p.top_down_bias()))
        out.append(p.update([0.1, 0.1, 2.0]))
        return out

    same_a = scenario(5)
    same_b = scenario(5)
    assert same_a == same_b, "same seed must reproduce identical goals, biases and rts"

    # `assert same_a != diff_c` ("a different seed must differ", scenario(5) vs scenario(6)) used to
    # stand here. Swept over seeds it holds 2/7. Measured over seeds 0..199 this scenario produces
    # exactly ONE decision tuple -- goal 0, maintained 0, winner 1, bias (1.0, 0.0, 0.0), goal 2 -- on
    # every single seed; the only seed-varying quantity is the deliberation RT, which takes just
    # {11, 12, 13, 15} and is CONSTANT across runs of 16+ consecutive seeds. Seeds 5 and 6 happened to
    # land on RT 12 vs 13, so the clause was reporting one lucky adjacent pair as evidence that the
    # seed is wired in. There is no window size that makes "a different seed differs" true here.
    # The real invariant is the opposite one, it is stronger, and it is what is asserted now: the
    # executive's CHOICES are evidence-driven, not RNG-driven -- only its reaction time is stochastic.
    def _choice_part(o):
        return (o[0], o[1], o[2][0], o[3], o[4])  # everything except the deliberation RT

    choices = {_choice_part(scenario(s)) for s in range(8)}
    rts = {scenario(s)[2][1] for s in range(200)}
    assert len(choices) == 1, (
        f"goals/winner/bias must be evidence-driven, identical on every seed (got {len(choices)})"
    )
    assert len(rts) > 1, f"the deliberation RT must be the stochastic part (got {rts})"

    # (6) model-based lookahead changes behaviour: reactive takes the immediate trap, planning avoids it.
    assert _argmax(immediate_vals) == 1, (
        f"the trap must be the immediate-best option (values {immediate_vals})"
    )
    assert _argmax(terminal_vals) == 0, (
        f"a different option must be the terminal-best (values {terminal_vals})"
    )
    assert reactive_goal == 1, (
        f"reactive deliberation (h=1) must take the immediate trap ({reactive_goal})"
    )
    assert planned_goal == 0 and planned_goal != reactive_goal, (
        f"planning (h=40) must reject the trap and commit the terminal-best ({planned_goal})"
    )

    # (7) the lookahead composes with the REAL forward model and is deterministic.
    assert via_rollout == manual, (
        f"the rollout must equal the forward model's own prediction chain ({via_rollout} vs {manual})"
    )
    assert plan_det, "same seed must reproduce an identical plan (goal + rt)"

    # (8) planning keeps the chronometric signature: a wider terminal value gap commits faster.
    assert wide_rt < narrow_rt, (
        f"a wider terminal-value gap must commit faster ({wide_rt:.1f} vs {narrow_rt:.1f} steps)"
    )

    # (9) richer working memory: a sub-goal sits above the primary, the hierarchy votes, it pops back.
    assert primary_goal == 0 and sub_goal == 2 and sub_goal != primary_goal, (
        f"push_subgoal must place a sub-goal above the primary ({hier_goals})"
    )
    assert hier_goals == [0, 2], f"the held hierarchy must be primary then sub-goal ({hier_goals})"
    assert hier_bias[0] > 0 and hier_bias[2] > 0 and sum(1 for b in hier_bias if b > 0) == 2, (
        f"the top-down bias must vote for the whole held hierarchy ({hier_bias})"
    )
    assert popped_back == primary_goal and after_pop == [primary_goal], (
        f"a decayed sub-goal must pop back to the held primary goal (got {popped_back}, stack {after_pop})"
    )

    # (10) THE T056 PORT'S OWN REDS -- retention, release, and the derived capacity, at fast
    # constants (tau_f=300, tau_d=200; regime-legal) with recurrence=1.0 so the commitment
    # policy does not drain the stack while the MEMORY is being measured, and commit_races=1
    # so the pushes are cheap. The full derived==held +/-1 guard runs in wm_capacity's own
    # self-check; these arms pin the store-side charges.
    def _stacked(n_frames, neutered=None):
        p = PrefrontalExecutive(
            n_options=4,
            commit_races=1,
            seed=0,
            recurrence=1.0,
            max_goals=10**9,
            tau_f=300.0,
            tau_d=200.0,
        )
        if neutered:
            p.neuter(neutered)
        ev = [[1.0 if j == (i % 4) else 0.0 for j in range(4)] for i in range(4)]
        for i in range(n_frames):
            p.push_subgoal(ev[i % 4])
            for _ in range(4):  # attend each new goal to facilitation equilibrium
                p.maintain()
        for _ in range(1000):  # > 3 * tau_f: past the unrefreshed retention horizon
            p.maintain()
        return len(p.goals)

    held_live = _stacked(6)
    held_no_R = _stacked(6, neutered="resources")
    held_no_reh = _stacked(6, neutered="rehearsal")
    assert held_live == 6, (
        f"within capacity, the rehearsal channel must keep every background frame alive "
        f"({held_live} of 6 held) -- the freeze is gone and its work must actually be done"
    )
    assert held_no_R <= 2 and held_no_R < held_live, (
        f"R'S CHARGE: cutting resources' route into the rehearsal current must let the dwell "
        f"hog the channel and STARVE the other background frames (live {held_live} vs "
        f"neutered {held_no_R}) -- T055 measured R saturated away at wm_attractor's own "
        f"operating point; in this port the refresh cycle is resource-bounded BY CONSTRUCTION"
    )
    assert held_no_reh == 1, (
        f"THE CHANNEL'S CHARGE: with the refresh cut (bookkeeping kept), every background "
        f"memory must decay past its ignition floor and be CULLED, leaving only the active "
        f"goal ({held_no_reh} held) -- deaths stay visible, they are not hidden by the neuter"
    )
    cap_default = PrefrontalExecutive(
        n_options=4, commit_races=1, seed=0, tau_f=300.0, tau_d=40.0
    ).max_goals
    cap_terms = derive_capacity(
        PrefrontalExecutive(
            n_options=4, commit_races=1, seed=0, tau_f=300.0, tau_d=40.0
        )._probe_frame,
        300.0,
    )
    assert cap_default == cap_terms["K"] + 1, (
        f"the DEFAULT max_goals must be the derived K+1, not a declared number "
        f"({cap_default} vs {cap_terms['K']}+1)"
    )
    explicit = PrefrontalExecutive(n_options=4, commit_races=1, seed=0, max_goals=2)
    for i in range(3):
        ev3 = [0.0] * 4
        ev3[i % 4] = 1.0
        explicit.push_subgoal(ev3)
    assert len(explicit.goals) == 2, (
        f"an explicitly passed max_goals must be honored verbatim ({explicit.goals})"
    )
    try:
        PrefrontalExecutive(n_options=2, tau_f=100.0, tau_d=200.0)
        raise AssertionError("tau_f <= tau_d must refuse (depression-dominant cannot hold)")
    except ValueError as e:
        assert "tau_f > tau_d" in str(e)
    try:
        PrefrontalExecutive(n_options=2).neuter("nonsense")
        raise AssertionError("an unknown neuter element must refuse, not no-op")
    except ValueError as e:
        assert "unknown element" in str(e)

    print(
        f"self-check OK: goal {committed} persists as PERSISTENT SPIKING across delay "
        f"(WM spikes {wm_spike_trace}; crippled -> {crippled_wm_spikes});"
    )
    print(
        f"  stability keeps A={goal_a} under weak B, flexibility switches to {after_strong}; spiking "
        f"WTA wins {wins[strongest]}/{n_win} and commits faster ({strong_rt:.1f} < {weak_rt:.1f} steps);"
    )
    print(
        f"  top-down bias tracks the goal; deterministic; lookahead: reactive takes trap option "
        f"{reactive_goal}, planning commits option {planned_goal}; sub-goal {sub_goal} pops back to "
        f"{primary_goal}"
    )
    print(
        f"  T056 port: within-capacity hold {held_live}/6; neuter('resources') starves to "
        f"{held_no_R} (R bounds the refresh cycle); neuter('rehearsal') culls to {held_no_reh} "
        f"(deaths visible); default max_goals DERIVED = {cap_default} (K={cap_terms['K']}+1 "
        f"active, tau_f=300); explicit cap honored; tau_f<=tau_d refused"
    )


if __name__ == "__main__":
    main()
