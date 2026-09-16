"""
Recursive Theory of Mind -- N agents, depth-k nested beliefs, content BEYOND location.

The repo has a first-order WHERE-tracker (cortex/theory_of_mind.BeliefPopulation) and, until T082,
also carried a WHERE-only second-order tracker (affect_core.SecondOrderToM, DELETED 2026-09-16 --
red R5 measured this module reproducing it on 200 constructed witness scripts, P2). Both hold a
single believed LOCATION for one observer. This module generalises on three axes at once:

  1. N AGENTS, DEPTH k. A belief is held on a HOLDER-PATH -- a tuple of agents. (A,) is A's own
     belief; (B, A) is B's model of A's belief; (C, B, A) is C's model of B's model of A's belief,
     and so on up to max_depth. Nesting is genuinely recursive, not hard-staged.
  2. CONTENT BEYOND LOCATION. The attributed content is a VSA ROLE-FILLER PROPOSITION
     bind(attitude, content), reusing cortex/vsa_reasoning (spike-polarity bind / bundle / unbind /
     coincidence). Attitudes include at least {believes, wants, knows}; content is any symbol, not
     just a basket. All attitudes a holder holds are BUNDLED into one spike code and queried back by
     unbinding the attitude role and cleaning up over the content codebook.
  3. EVENT-DRIVEN, FALSE-BELIEF NATIVE. Beliefs change only through observe_event(agent, attitude,
     content, witnesses): the world always moves, but a belief-path only updates if every agent on
     the path was PRESENT (in `witnesses`) to witness the subject observe. An ABSENT agent's belief
     is never touched, so it goes STALE relative to the world -- a first- or higher-order FALSE
     belief -- for free.

Spiking substrate (reuse, not reinvention). The VSA cleanup selects WHICH content symbol a path
believes; that selected content is then read out as a genuine SPIKING POPULATION CODE -- encoded
across overlapping RBF neurons (cortex/population_encoding.PopulationEncoder) and decoded by a LIF
winner-take-all (one cortex/lif_neuron.LIFNeuron per bin, most-firing bin wins), exactly the readout
theory_of_mind.BeliefPopulation uses. So the belief is a spiking population, not a stored label, and
the readout is noise-robust (a jitter check below perturbs the counts and still recovers).

Honest framing. This is a FUNCTIONAL recursive belief tracker: nested belief populations that can lag
the world and each other. It is NOT genuine mentalizing and does not know "what it is like" to hold a
belief -- it models only THAT beliefs (own and attributed, at any depth) can diverge from reality and
from one another. Scenarios are toy and the readout is near-tabular. Deterministic (crc32-seeded VSA
codes + per-instance RNG + fixed LIF), ASCII-only.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.recursive_tom
"""

import itertools
import random
import sys
from pathlib import Path

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.population_encoding import PopulationEncoder
from snn_data_lake.cortex.vsa_reasoning import bind, bundle, coincidence, hv
from snn_data_lake.substrate.paired_contrast import T_BAR, contrast

# spiking winner-take-all readout constants (shared with theory_of_mind.BeliefPopulation)
_WINDOW = 20  # steps each content bin's LIF neuron is driven
_TAU_M = 10.0  # LIF membrane time constant
_DRIVE = 0.3  # input current for the peak-count bin (others scaled by count/peak)
_MAX_SPIKES = 12  # population-code peak spike count

DEFAULT_ATTITUDE = "believes"  # attitude assumed when a caller passes attitude=None
ATTITUDES = (
    "believes",
    "wants",
    "knows",
)  # canonical attitude vocabulary (open -- any string works)


def _spiking_readout(index, n_candidates, seed, window=_WINDOW, jitter=0):
    """Read a selected content INDEX back out as a spiking population code: encode it across
    overlapping RBF neurons (PopulationEncoder) and decode with a LIF winner-take-all -- one LIFNeuron
    per bin, driven by a current proportional to its spike count, the most-firing bin wins (ties ->
    lowest index). Mirrors theory_of_mind.BeliefPopulation so the belief is a genuine spiking
    population, not a stored index. jitter>0 perturbs the counts to show the readout is noise-robust."""
    if n_candidates <= 1:
        return 0
    enc = PopulationEncoder(n=n_candidates, lo=0.0, hi=n_candidates - 1, max_spikes=_MAX_SPIKES)
    counts = enc.encode(float(index))
    if jitter:
        rng = random.Random(seed)
        counts = [max(0, k + rng.randint(-jitter, jitter)) for k in counts]
    peak = max(counts)
    if peak <= 0:
        return index
    fired = []
    for k in counts:
        neuron = LIFNeuron(tau_m=_TAU_M)
        fired.append(sum(neuron.run([_DRIVE * k / peak] * window)))
    return max(range(len(fired)), key=lambda i: (fired[i], -i))


class RecursiveToM:
    """N-agent, depth-k recursive theory of mind over VSA role-filler propositions.

    State is a map from a HOLDER-PATH (a tuple of agents) to the propositions that path holds:
    (A,) -> A's own belief, (B, A) -> B's model of A, (C, B, A) -> C's model of B's model of A, ...
    A proposition is (attitude, content); a path's propositions are bundled into one spike code and
    queried by unbinding the attitude role. observe_event is the ONLY mutator: the world moves every
    event, but a path updates only if all its agents witnessed the subject observe, so absent agents'
    beliefs (at any depth) go stale -- the false-belief substrate."""

    def __init__(self, n_agents=3, seed=0, max_depth=3):
        if n_agents < 1:
            raise ValueError("need n_agents >= 1")
        if max_depth < 2:
            raise ValueError("need max_depth >= 2 (recursive ToM requires nesting)")
        self.n_agents = n_agents
        self.seed = seed
        self.max_depth = max_depth
        self.agents = [
            chr(ord("A") + i) for i in range(n_agents)
        ]  # default names; any id is accepted
        self.world = {}  # attitude -> true content (reality; updates on EVERY event)
        self._props = {}  # holder-path tuple -> {attitude: content} (the write-side record)
        self._content = set()  # content-label codebook the VSA cleanup searches over

    # -- role/filler spike codes (crc32-seeded via vsa_reasoning.hv -> deterministic across process) --
    @staticmethod
    def _role(attitude):
        return hv("ATTITUDE::" + str(attitude))

    @staticmethod
    def _filler(content):
        return hv("CONTENT::" + str(content))

    def _candidates(self):
        """Ordered content vocabulary the spiking readout / VSA cleanup discriminates over."""
        return sorted(self._content)

    def _set(self, path, attitude, content):
        self._props.setdefault(path, {})[attitude] = content

    def observe_event(self, agent, attitude, content, witnesses=None):
        """Fold an observed event into the belief state. `agent` is the SUBJECT (the one whose
        (attitude, content) mental state the event establishes, e.g. agent believes content is at X);
        `witnesses` is the set of agents PRESENT to observe it (defaults to just the subject).

        The world always moves to `content`. If the subject is present, its first-order belief updates
        and, for every ordered chain of distinct present witnesses ending in the subject (up to
        max_depth), that nested path updates too (each witness saw the ones ahead of it observe). A
        subject or witness that is ABSENT is never referenced, so its belief-path persists (stale) --
        a first- or higher-order false belief."""
        attitude = str(attitude)
        content = str(content)
        present = {agent} if witnesses is None else set(witnesses)
        self._content.add(content)
        self.world[attitude] = content  # reality reflects the event unconditionally
        if agent not in present:  # subject absent -> nothing to attribute
            return
        self._set((agent,), attitude, content)  # first-order: the subject's own belief
        others = sorted(w for w in present if w != agent)
        for depth in range(1, self.max_depth):  # nested chains of length 2 .. max_depth
            for chain in itertools.permutations(others, depth):
                self._set(chain + (agent,), attitude, content)

    def _predict(self, path, attitude, jitter=0):
        """Read the content a HOLDER-PATH attributes under `attitude`: bundle the path's propositions,
        unbind the attitude role (VSA), pick the best content by spike coincidence over the codebook,
        then read that content out through the spiking population code + LIF winner-take-all. Returns
        the content label, or None if the path holds no belief under that attitude."""
        props = self._props.get(tuple(path))
        if not props or attitude not in props:
            return None
        belief = bundle([bind(self._role(a), self._filler(c)) for a, c in sorted(props.items())])
        query = bind(belief, self._role(attitude))  # unbind = self-inverse spike XOR
        cands = self._candidates()
        scores = [coincidence(query, self._filler(c)) for c in cands]
        win = max(range(len(cands)), key=lambda i: (scores[i], -i))  # VSA cleanup over the codebook
        idx = _spiking_readout(win, len(cands), self.seed, jitter=jitter)
        return cands[idx]

    def predict_first_order(self, agent, attitude=None, jitter=0):
        """What `agent` itself believes/wants/knows (holder-path (agent,))."""
        att = DEFAULT_ATTITUDE if attitude is None else attitude
        return self._predict((agent,), att, jitter=jitter)

    def predict_nested(self, observer, target, attitude=None, jitter=0):
        """What `observer` thinks `target` believes/wants/knows (holder-path (observer, target))."""
        att = DEFAULT_ATTITUDE if attitude is None else attitude
        return self._predict((observer, target), att, jitter=jitter)

    def predict_chain(self, path, attitude=None, jitter=0):
        """General depth-k readout: path (X1, X2, ..., Xk) = X1's model of X2's model of ... Xk."""
        att = DEFAULT_ATTITUDE if attitude is None else attitude
        return self._predict(tuple(path), att, jitter=jitter)

    def divergence(self, observer, target, attitude=None):
        """True iff `observer`'s nested model of `target` differs from `target`'s own first-order
        belief -- i.e. observer holds a divergent (possibly false) higher-order belief about target."""
        att = DEFAULT_ATTITUDE if attitude is None else attitude
        return self.predict_nested(observer, target, att) != self.predict_first_order(target, att)

    def world_content(self, attitude=None):
        """The true world content for an attitude (reality, ignoring every mind). The control that a
        false belief diverges FROM."""
        att = DEFAULT_ATTITUDE if attitude is None else attitude
        return self.world.get(att)


# =============================================================================
# T082 -- SOCIAL COGNITION COMPOSED (specs/022 [US4], FR-034, the C5 row `social_cognition`).
# Pre-registration: `specs/022-unified-brain-arc/evidence/t082-social-cognition.md`, FROZEN at
# the commit that landed it. This block is 2.9 STEP 2, the skeleton: 2.8's declared constants,
# the DECLARED belief wire, and the instrument STUBS the six reds of 2.4 run RED against at
# step 3. Nothing here reads a brain; the recorder, the dry run and the campaign land at step 5.
#
# WHY THE PORT LIVES HERE AND NOT IN A NEW MODULE: the T082 row names `cortex/recursive_tom.py`
# wiring, and this is the faculty whose divergence count the arm's caution already folds
# (brain.py's wake tick -> `arm._social_risk` -> autonomous_loop's caution term). A parallel
# instrument module beside the faculty would be the additive failure P2 names.
# =============================================================================

# ---- 2.8's DECLARED CONSTANTS. Each is the plan's width, the composite's own shape, or a
# quantity the pre-registration names; every BUILDER-INTRODUCED width says so (T079-T081's
# disclosure discipline -- an undisclosed width is a hidden degree of freedom).
#: Episodes per seed's scripted dialogue (2.2, 2.8).
T082_N_EPISODES = 60
#: Wake ticks run BETWEEN statements, so the brain's own `_tom_protect` fold executes and the
#: caution trajectory is readable within the episode (2.1, 2.8; 1 / 3 / 5 reported on the dry run).
T082_TICKS_BETWEEN = 3
#: The script's probability that an episode is UNWITNESSED (2.1, 2.8).
T082_P_UNWITNESSED = 0.5
#: The script's object and location vocabularies (2.8).
T082_OBJECTS = ("marble", "key", "coin", "ring", "cup")
T082_LOCATIONS = ("basket", "box", "drawer", "shelf")
#: The per-seed stream offset applied to BOTH pointers: `(T082_STREAM_OFFSET * seed) % 256`
#: (2.1; T080's rule, carried by T081).
T082_STREAM_OFFSET = 150
#: The campaign's seeds (2.2; PLAN-FINAL Part F: S = 8).
T082_SEEDS = tuple(range(8))
#: (g0)'s floor (2.3): the fraction of a seed's episodes that must be UNWITNESSED before the
#: stream is accepted as carrying a false belief to track.
T082_MIN_UNWITNESSED = 0.3

#: THE DECLARED BELIEF WIRE (2.1, FR-034): producer half + THE edge here, consumer half
#: (`brain.social_risk_in`) in `cortex/brain.declare_ports`.
BELIEF_EDGE = ("recursive_tom", "belief_out", "brain", "social_risk_in")

#: The script's RNG: `random.Random(T082_RNG_BASE + seed)` (2.1, the territory probe's base).
T082_RNG_BASE = 4000
#: The stream-pointer modulus (2.1's "mod their packs"; T081's declared width).
T082_STREAM_POSITIONS = 256
#: THE BRAIN'S OWN social-risk fold, READ not chosen (2.8): on the wake tick
#: `_tom_protect = max(min(1, GAIN * fb), DECAY * prev)` and the arm's caution term is that
#: value. The pair lives HERE so the charge A/B's consumer and
#: `cortex/brain._route_percept_to_action` read ONE definition (2.9 step 4 re-points brain).
T082_PROTECT_GAIN = 0.34
T082_PROTECT_DECAY = 0.9
#: The one bar (2.3): the paired one-sided t over 8 seeds, 7 d.f., alpha = 0.05.
T082_BAR = T_BAR
#: The tree root, so the pin resolves from any working directory (T081's idiom).
REPO = Path(__file__).resolve().parents[3]
#: The pin, written ONCE by `--t082-campaign` (2.9 step 6).
T082_PIN = REPO / "tools" / "t082_social_cognition.json"
#: The dry run's report (2.9 step 5). Never under tools/: the pin is the campaign's.
T082_DRY_RUN_OUT = Path("scratchpad/t082/dry_run.json")

#: The two AGENTS the live dialogue path tracks (2.1): the user and the brain.
T082_AGENTS = ("user", "brain")
#: R2's constructed witness script (2.4): this many events at this absence rate.
T082_R2_EVENTS = 40
T082_R2_P_ABSENT = 0.5
#: R3's constructed scripts: one per seed, at the campaign's seed count (2.4, "eight").
T082_R_SEEDS = T082_SEEDS
T082_R3_EVENTS = 40
#: R5's losslessness sweep (2.4): this many random witness scripts over this many agents,
#: locations and events each.
T082_R5_SCRIPTS = 200
T082_R5_AGENTS = 3
T082_R5_LOCATIONS = 4
T082_R5_EVENTS = 12
#: R5's RNG base -- BUILDER-INTRODUCED, disclosed (T081's `T081_SCR_REDRAW_BASE` precedent).
T082_R5_RNG_BASE = 820000
#: R4's charge frames: this many recorded [fb_first, fb_nested] ticks, and the PINNED payload
#: the neuter substitutes (2.1: "the pinned payload [0, 0]").
T082_R4_FRAMES = 12
T082_PINNED_BELIEF = [0.0, 0.0]

#: One ASCII line per red, printed beside its boolean so a gate log says WHAT fired (2.4).
T082_RED_LINES = {
    "R1": "the parse REFUSES outside the grammar (parsing anything reddens it)",
    "R2": "the equal-capacity control IS a world model (honouring absence reddens it)",
    "R3": "the dissociation reads on constructed scripts (swapping the arms reddens it)",
    "R4": "the belief payload moves the caution fold (a payload-ignoring consumer reddens it)",
    "R5": "the symmetric feed is LOSSLESS vs both retired trackers (reading the world reddens it)",
    "R6": "the symmetric feed kills the spurious divergence (the subject-only feed reddens it)",
}


def declare_ports(manifest):
    """Manifest registrant (specs/022 T082, FR-034): the recursive belief tracker's surface.
    OUT `belief_out` -- (2,) f4 = [first-order false-belief count, nested-divergence count]
    over the tracked agents -- GRADED by `divergence` (the association source: the nested half
    IS `RecursiveToM.divergence` summed over ordered pairs). The consumer half is
    `brain.social_risk_in`, declared in `cortex/brain.declare_ports` beside `task_film_in`.

    THE EDGE IS DECLARED HERE, by the PRODUCER, and that is a deviation from the T034/M10
    convention (the consumer declares its subscription) recorded rather than smoothed: the
    builder brief for 2.9 step 2 puts the edge in this registrant, and `declare_edge` appends
    unconditionally while `validate_structure` checks the ASSEMBLED manifest order-independently
    (manifest_gate's own T060 comment measured both orderings identical), so the position
    carries no mechanism either way.

    THE WIDTH IS A LITERAL 2 and that is the honest form here, unlike `task_set.film_out`'s
    derived `len(ACTIONS)`: the payload is not a per-item vector but the pair of COUNTS the
    fold reads, and the pair cannot grow with any vocabulary."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "recursive_tom",
        Port(
            name="belief_out",
            region="cortex",
            shape=(2,),
            dtype="f4",
            unit="count",
            encoding="VEC_DENSE",
            kind="OUT",
            declaring_module="recursive_tom",
            delivery_policy=("graded", "divergence"),
        ),
    )
    manifest.declare_edge(*BELIEF_EDGE)


# =============================================================================
# 2.9 STEP 4 -- THE INSTRUMENTS, GREEN. Every red of 2.4 is reddened IN FACT by its own mutated
# scorer (`_t082_mutated_scorer`). Nothing here reads a BRAIN: every input is a CONSTRUCTED
# script or a finished recording. The recorder, the dry run and the campaign land at step 5.
# =============================================================================
#: R1's off-grammar statement (2.4): a turn carrying no belief statement at all.
T082_OFF_GRAMMAR_STMT = "the weather is nice today."
#: R4's charge stream (2.1: "a scripted stream with false beliefs"): the witness pattern the
#: belief frames are recorded from, CONSTRUCTED rather than drawn so the live arm is GUARANTEED
#: to carry a divergence -- an A/B whose live arm happened to hold none would measure nothing and
#: read as a clean refusal. Four episodes x `T082_TICKS_BETWEEN` = `T082_R4_FRAMES` frames.
T082_R4_WITNESS = (True, False, True, False)


def t082_feed(tracker, present, loc):
    """THE SYMMETRIC EVENT (2.1) -- the ONE way a dialogue statement reaches the belief tracker.

    For a present set P, call `observe_event(a, "believes", loc, witnesses=P)` once for EACH
    a in sorted(P), so every present agent's OWN path and every chain among the present update,
    while an absent agent's paths stay stale (the false-belief substrate). The content is the
    location NAME; there is no id map.

    THE DEFECT THIS REPLACES (fact 3 + fact 4): `observe_event` is SUBJECT-asymmetric -- a
    present witness's own first-order belief is not updated unless it IS the subject. The live
    path called it once with a single subject, so a witnessed statement about the user left the
    brain's own path pinned at its last unwitnessed location and `divergence("user", "brain")`
    read True from then on. Feeding each present agent as a subject in turn is what makes the
    tracker's witness semantics match `SecondOrderToM`'s symmetric form, which is what R5
    measures and what licenses that class's deletion (P2)."""
    present = set(present)
    for a in sorted(present):
        tracker.observe_event(a, DEFAULT_ATTITUDE, str(loc), witnesses=present)
    return tracker


def t082_false_beliefs(tracker, agents=T082_AGENTS):
    """THE ONE derivation of the brain's `_false_beliefs` list (2.1) -> (first, nested).

    `first` names the agents whose FIRST-ORDER belief diverges from the world; `nested` names
    the ordered pairs "o->t" whose nested model diverges from the target's own belief. The
    declared `belief_out` payload is `[len(first), len(nested)]` and the arm's caution reads
    `len(first) + len(nested)` -- which is what `Brain._route_percept_to_action` computes as
    `len(self._false_beliefs)`.

    RECORDED, not smoothed: the live path used to write the FIRST list on an unwitnessed turn
    and the NESTED list on a witnessed one, so the meaning of `recursive_tom_false_beliefs`
    (and of the caution the arm received) changed with the last statement's type. The key is
    unchanged; its contents are now one quantity on every turn."""
    world = tracker.world_content()
    first = [a for a in agents if tracker.predict_first_order(a) != world]
    nested = [
        f"{o}->{t}" for o in agents for t in agents if o != t and bool(tracker.divergence(o, t))
    ]
    return first, nested


class _T082SecondOrderReference:
    """R5's CONTROL (2.4, 2.5): the RETIRED `affect_core.SecondOrderToM.observe` bookkeeping,
    restated over plain dicts -- the world, each present agent's own belief, and every ordered
    pair of present agents.

    It is a genuinely DIFFERENT mechanism from the tracker it checks (no VSA bundle, no spiking
    winner-take-all, no codebook cleanup), which is why it survives the deletion of the class it
    restates AND makes the equivalence stronger than a comparison against a class that shared
    `theory_of_mind.BeliefPopulation` with the first-order side. `seed` and `n_locations` are
    accepted for signature parity with the deleted class and are deliberately unused: this
    reference holds labels, not populations."""

    def __init__(self, n_locations=T082_R5_LOCATIONS, seed=0):
        self.n_locations = int(n_locations)
        self.seed = seed
        self.world = None
        self.belief = {}  # agent -> believed location
        self.belief2 = {}  # (x, y) -> x's model of y's belief

    def observe(self, location, observers):
        self.world = location
        obs = list(observers)
        for x in obs:
            self.belief[x] = location
        for x in obs:
            for y in obs:
                if x != y:
                    self.belief2[(x, y)] = location

    def predict_first_order(self, agent):
        return self.belief.get(agent)

    def predict_second_order(self, x, y):
        return self.belief2.get((x, y))

    def predict_from_world(self):
        return self.world


def t082_second_order_reference(n_locations=T082_R5_LOCATIONS, seed=0):
    """A fresh retired-second-order reference tracker (R5's control)."""
    return _T082SecondOrderReference(n_locations=n_locations, seed=seed)


def _t082_parse(message):
    """THE GRAMMAR, READ from the live path rather than restated: this imports `_TOM_PLACE` and
    `_TOM_ABSENT` -- the compiled objects `Brain._tom_update` itself matches on -- so a change to
    the surface parse cannot leave the constructed inputs testing a second, stale grammar (P2).
    -> (obj, loc, unwitnessed) or None. The import is lazy because brain imports this module at
    module scope for the fold constants."""
    from snn_data_lake.cortex.brain import _TOM_ABSENT, _TOM_PLACE

    low = str(message).lower()
    m = _TOM_PLACE.search(low)
    if not m:
        return None
    obj, loc = m.group("obj").strip(), m.group("loc").strip()
    if not obj or not loc:
        return None
    return obj, loc, any(mk in low for mk in _TOM_ABSENT)


def t082_script(seed, n_ep=T082_N_EPISODES, all_witnessed=False):
    """THE SCRIPTED OTHER AGENT (2.1), and the positive control that is NOT the method (2.5).

    Objects and locations are drawn under `random.Random(T082_RNG_BASE + seed)` at
    `T082_P_UNWITNESSED`; the agent's LOOK is its last WITNESSED location and the world is the
    last location, so on every unwitnessed episode the two differ -- which is what (g0) counts.
    The first episode is always witnessed (there is no world to move yet).

    `all_witnessed` forces every episode witnessed for R3's degenerate arm. The RNG draw is
    still CONSUMED so the location sequence is identical to the mixed script's -- the two arms
    differ in witness structure and in nothing else."""
    rng = random.Random(T082_RNG_BASE + int(seed))
    obj = rng.choice(T082_OBJECTS)
    agent_looks = None
    world = None
    out = []
    for ep in range(int(n_ep)):
        loc = rng.choice([x for x in T082_LOCATIONS if x != world] or list(T082_LOCATIONS))
        drawn = rng.random() < T082_P_UNWITNESSED
        unwitnessed = bool(drawn and world is not None and not all_witnessed)
        stmt = (
            f"someone moved the {obj} to the {loc} while i was away."
            if unwitnessed
            else f"i put the {obj} in the {loc}."
        )
        world = loc
        if not unwitnessed:
            agent_looks = loc
        out.append(
            {
                "ep": ep,
                "obj": obj,
                "loc": loc,
                "stmt": stmt,
                "unwitnessed": unwitnessed,
                "present": ("brain",) if unwitnessed else T082_AGENTS,
                "world": world,
                "agent_looks": agent_looks,
            }
        )
    return out


def t082_caution(fb, prev, ticks=T082_TICKS_BETWEEN):
    """The brain's OWN social-risk fold, run `ticks` times on one episode's false-belief count:
    `_tom_protect = max(min(1, GAIN * fb), DECAY * prev)` per wake tick, which is exactly what
    `Brain._route_percept_to_action` computes and pushes to `arm._social_risk`."""
    for _ in range(int(ticks)):
        prev = max(min(1.0, T082_PROTECT_GAIN * float(fb)), T082_PROTECT_DECAY * float(prev))
    return prev


def t082_constructed_rows(
    seed=0,
    n_ep=T082_R3_EVENTS,
    off_grammar_at=None,
    all_witnessed=False,
    parse=None,
    feed=None,
):
    """ONE constructed recording in the LIVE recorder's row shape (2.4), with NO brain: the
    belief tracker fed by `t082_feed` with the script's present set, the EQUAL-CAPACITY CONTROL
    fed the same statements with every agent present, and the caution trajectory produced by the
    brain's own fold. It lives here and not only in the test module because `t082_reds()` needs
    it: a gate run that never executes pytest still runs every red on the SAME input.

    `off_grammar_at` replaces that episode's statement with one the grammar refuses -- neither
    tracker is fed, exactly as the live path does not feed on a non-parse, and the script's
    ground truth moves on without them, which is what (r) must REFUSE."""
    parse = _t082_parse if parse is None else parse
    feed = t082_feed if feed is None else feed
    steps = t082_script(seed, n_ep, all_witnessed=all_witnessed)
    rt = RecursiveToM(n_agents=3, seed=int(seed))
    ctrl = RecursiveToM(n_agents=3, seed=int(seed))
    rows = []
    prev = 0.0
    for st in steps:
        stmt = (
            T082_OFF_GRAMMAR_STMT
            if (off_grammar_at is not None and st["ep"] == off_grammar_at)
            else st["stmt"]
        )
        hit = parse(stmt)
        ev = None
        if hit is not None:
            _obj, _loc, _unw = hit
            ev = f"{'unwitnessed' if st['unwitnessed'] else 'witnessed'}: the {st['obj']} at '{st['loc']}'"
            feed(rt, set(st["present"]), st["loc"])
            feed(ctrl, set(T082_AGENTS), st["loc"])  # the control ignores absence
        first, nested = t082_false_beliefs(rt)
        frame = [len(first), len(nested)]
        prev = t082_caution(sum(frame), prev)
        rows.append(
            {
                "ep": st["ep"],
                "stmt": stmt,
                "unwitnessed": st["unwitnessed"],
                "parsed": ev,
                "agent_looks": st["agent_looks"],
                "world": st["world"],
                "pred_belief": rt.predict_first_order("user"),
                "pred_world": rt.world_content(),
                "pred_ctrl": ctrl.predict_first_order("user"),
                "false_beliefs": list(first) + list(nested),
                "social_risk": round(prev, 4),
                "charge_frame": frame,
                "t_respond_ms": 0.0,
            }
        )
    return {"seed": int(seed), "n_ep": int(n_ep), "rows": rows}


def t082_record_brain(seed, n_ep=T082_N_EPISODES):
    """STUB (2.9 step 4) -> an empty recording. THE RECORDER lands at 2.9 step 5: a scratch brain
    under `throwaway_home()`, `make_brain(seed, persist=False)`, the script of 2.1 driven through
    `Brain.respond` with `T082_TICKS_BETWEEN` ticks between statements."""
    return {"seed": int(seed), "n_ep": int(n_ep), "rows": []}


def t082_read(rows, seed=None):
    """The per-seed reading of 2.1 over a recording's rows, live or constructed.

    `refusals` is (r) ONLY (2.3): every statement must parse, and after the first witnessed
    statement every episode's THREE predictions must be non-None. (g0) is a per-seed count and
    is adjudicated in `t082_contrasts`, so a stream that simply carries no false belief is never
    confused with an apparatus that could not be read."""
    rows = list(rows)
    n = len(rows)
    tag = "" if seed is None else f"seed {seed}: "
    parsed = sum(1 for r in rows if r.get("parsed"))
    unw = [r for r in rows if r.get("unwitnessed")]

    def acc(key):
        return (sum(1 for r in rows if r.get(key) == r.get("agent_looks")) / n) if n else 0.0

    first_witnessed = next((i for i, r in enumerate(rows) if not r.get("unwitnessed")), None)
    none_after = 0
    if first_witnessed is not None:
        for r in rows[first_witnessed:]:
            if any(r.get(k) is None for k in ("pred_belief", "pred_world", "pred_ctrl")):
                none_after += 1
    out = {
        "n": n,
        "parsed": parsed,
        "unwitnessed": len(unw),
        "unwitnessed_frac": (len(unw) / n) if n else 0.0,
        "acc_belief": acc("pred_belief"),
        "acc_world": acc("pred_world"),
        "acc_ctrl": acc("pred_ctrl"),
        "ctrl_equals_world": sum(1 for r in rows if r.get("pred_ctrl") == r.get("pred_world")),
        "look_differs": sum(1 for r in rows if r.get("agent_looks") != r.get("world")),
        "unwitnessed_look_differs": sum(1 for r in unw if r.get("agent_looks") != r.get("world")),
        "predictions_none": none_after,
        # fact 4's rate: a NESTED divergence on an episode that is witnessed AND carries no
        # first-order false belief is spurious by construction. Expected 0 after the merge; the
        # old subject-only feed read it on 5 of 12 episodes on the territory probe.
        "spurious_nested": sum(
            1
            for r in rows
            if not r.get("unwitnessed")
            and r.get("agent_looks") == r.get("world")
            and (r.get("charge_frame") or [0, 0])[1] > 0
        ),
        "risk": [r.get("social_risk") for r in rows],
        "risk_max": max((r.get("social_risk") or 0.0) for r in rows) if rows else 0.0,
        "respond_ms": [r.get("t_respond_ms") for r in rows],
    }
    refusals = []
    if n == 0:
        refusals.append(f"{tag}(r) empty recording -- no episode was recorded")
    if parsed != n:
        refusals.append(f"{tag}(r) parse: {n - parsed} of {n} statements did not parse")
    if none_after:
        refusals.append(
            f"{tag}(r) predictions: {none_after} episodes after the first witnessed statement "
            f"carry a None prediction"
        )
    out["refusals"] = refusals
    if seed is not None:
        out["seed"] = int(seed)
    return out


def t082_constructed_read(seed=0, **kw):
    """`t082_read` over ONE constructed recording, with the seed carried into the reading so a
    constructed contrast names the seed it refuses (T081's idiom)."""
    rec = t082_constructed_rows(seed=seed, **kw)
    out = t082_read(rec["rows"], seed=rec["seed"])
    out["n_ep"] = rec["n_ep"]
    return out


def _t082_s1(entry):
    """(s1)'s per-seed difference: the belief model's accuracy MINUS the equal-capacity world
    model's, both against the scripted agent's look. 2.3's rule, spelled once."""
    return float(entry["acc_belief"]) - float(entry["acc_ctrl"])


def _t082_s2(entry):
    """(s2)'s per-seed check, exact and a count: the control predicted the world content on
    EVERY episode, so its accuracy IS the world predictor's -- the gain is removed when the
    belief model is replaced."""
    return int(entry.get("ctrl_equals_world", -1)) == int(entry.get("n", 0)) and float(
        entry.get("acc_ctrl", -1.0)
    ) == float(entry.get("acc_world", -2.0))


def t082_contrasts(per_seed, score=None, ctrl=None, bar=T082_BAR):
    """2.3's conjuncts over the per-seed readings, through `substrate.paired_contrast.contrast`.

    (g0) the script IS a false-believing agent; (s1) the licensed sentence, a paired one-sided t
    against the bar; (s2) the dissociation input, an exact per-seed count. The verdict is 2.3's:
    REFUSED iff (r) or (g0) fails, PASS iff (s1) and (s2), else NOT MET. A REFUSAL is a finding
    about the apparatus or the stream; a NOT MET is a finding about THIS composite. They are
    never merged."""
    per_seed = list(per_seed)
    score = _t082_s1 if score is None else score
    ctrl = _t082_s2 if ctrl is None else ctrl

    g0_fail = []
    for e in per_seed:
        frac = float(e.get("unwitnessed_frac", 0.0))
        if frac < T082_MIN_UNWITNESSED:
            g0_fail.append(
                f"seed {e.get('seed')}: unwitnessed {frac:.3f} < {T082_MIN_UNWITNESSED} -- "
                f"the stream carries no false belief to track"
            )
        missed = int(e.get("unwitnessed", 0)) - int(e.get("unwitnessed_look_differs", 0))
        if missed:
            g0_fail.append(
                f"seed {e.get('seed')}: {missed} unwitnessed episodes where the agent's look "
                f"does NOT differ from the world"
            )
    s2_fail = [e.get("seed") for e in per_seed if not ctrl(e)]
    refusals = [w for e in per_seed for w in e.get("refusals", [])]
    out = {
        "n_seeds": len(per_seed),
        "g0": {"met": bool(per_seed) and not g0_fail, "failures": g0_fail},
        "s1": contrast(
            "(s1) the belief model beats the equal-capacity world model",
            [score(e) for e in per_seed],
            bar,
        ),
        "s2": {"met": bool(per_seed) and not s2_fail, "failures": s2_fail},
        "refusals": refusals,
    }
    if refusals or not out["g0"]["met"]:
        out["verdict"] = "REFUSED"
    elif out["s1"].met and out["s2"]["met"]:
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "NOT MET"
    return out


# ---- THE CHARGE (2.1, T060's form) -------------------------------------------------------
def _t082_belief_frames(witness=T082_R4_WITNESS, ticks=T082_TICKS_BETWEEN):
    """Recorded belief frames for the charge A/B: one `[fb_first, fb_nested]` vector per WAKE
    TICK of a scripted stream carrying false beliefs, flattened to the declared (2,) shape. The
    fold runs per tick in the live brain, so the frames are per tick here too -- a per-EPISODE
    frame stream would measure a trajectory the arm never receives."""
    rt = RecursiveToM(n_agents=3, seed=0)
    locs = list(T082_LOCATIONS)
    frames = []
    tick = 0
    for i, witnessed in enumerate(witness):
        loc = locs[i % len(locs)]
        t082_feed(rt, set(T082_AGENTS) if witnessed else {"brain"}, loc)
        first, nested = t082_false_beliefs(rt)
        for _ in range(int(ticks)):
            tick += 1
            frames.append((tick, "recursive_tom", [float(len(first)), float(len(nested))]))
    return frames


def _t082_consumer_factory():
    """Fresh ring + JoinConsumer over the DECLARED consumer port (`brain.social_risk_in`), one
    pair per replay arm -- the wire's two ends are the real declarations, not stand-ins."""
    from snn_data_lake.cortex.brain import declare_ports as brain_declare_ports
    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import WiringManifest

    m = WiringManifest()
    declare_ports(m)
    brain_declare_ports(m)
    port = m.ports[("brain", "social_risk_in")]
    ring = SignalRing(slots=64)
    return ring, JoinConsumer(ring, port)


def _t082_consume_caution(consumer):
    """The downstream computation under measurement: EXACTLY the brain's own fold. `fb` is the
    SUM of the payload's two counts, because `Brain._route_percept_to_action` reads
    `len(self._false_beliefs)` and that list is the first-order holders plus the diverging
    pairs. Reduced to the caution trajectory `arm._social_risk` receives."""
    out = []
    prev = 0.0
    joined, _lost = consumer.poll_joined()
    for tick, vec in joined:
        fb = sum(float(x) for x in vec)
        prev = max(min(1.0, T082_PROTECT_GAIN * fb), T082_PROTECT_DECAY * prev)
        out.append((tick, round(prev, 6)))
    return out


def _measure_belief_charge(frames=None, consume=None):
    """The replay-twice A/B over recorded belief frames -> (live, pinned_out, reads,
    specificity_holds). The pinned payload is `[0, 0]`: no false belief, no divergence -- the
    honest neuter for a COUNT port, where a "plausible constant" would be a claim about a rate
    this wire does not have."""
    from snn_data_lake.substrate.port_charge_probe import replay_twice

    frames = _t082_belief_frames() if frames is None else frames
    consume = _t082_consume_caution if consume is None else consume
    live, pinned_out, reads = replay_twice(
        frames, _t082_consumer_factory, consume, list(T082_PINNED_BELIEF)
    )

    def constant_consumer(consumer):
        joined, _lost = consumer.poll_joined()
        return [(tick, 0) for tick, _vec in joined]

    spec_live, spec_pinned, _ = replay_twice(
        frames, _t082_consumer_factory, constant_consumer, list(T082_PINNED_BELIEF)
    )
    return live, pinned_out, reads, spec_live == spec_pinned


def record_charge():
    """Measure the belief wire's A/B and record the attribution via THE one writer
    (`port_charge_probe.record_attribution`). REFUSES a non-diverged run -- evidence is a
    measurement, never a declaration. `tests/test_recursive_tom.py` re-measures and requires the
    recorded numbers to reproduce, so this record cannot outlive the wire it describes."""
    from snn_data_lake.substrate.port_charge_probe import record_attribution

    live, pinned_out, reads, spec_holds = _measure_belief_charge()
    if live == pinned_out or not spec_holds:
        print("record_charge REFUSED: the A/B did not diverge (or specificity broke)")
        return 1
    path = record_attribution(
        BELIEF_EDGE,
        {"reads": reads, "first": list(live[0]), "last": list(live[-1])},
        "port_charge_probe.replay_twice over recorded [fb_first, fb_nested] frames from a "
        "scripted witnessed/unwitnessed stream; the consumer fold is brain's own "
        "_tom_protect recurrence (max(min(1, 0.34 fb), 0.9 prev), fb = the payload's two "
        "counts summed) reduced to the caution trajectory the arm receives; pinned payload "
        "[0, 0]; specificity control held (payload-ignoring consumer did not flip)",
    )
    print(f"recorded {BELIEF_EDGE} into {path}")
    return 0


def _t082_mutated_scorer(kind):
    """2.4's six mutations, RUN and not reasoned. Each is the deliberately broken twin of the
    instrument whose red it must redden, and each is exercised BOTH in
    `tests/test_recursive_tom.py` and in `t082_reds()` below.

      parse_anything        (R1) -- a statement the grammar does not match returns an event
          anyway, so a stream containing an off-grammar turn parses 100 % and (r) never refuses.
      control_keeps_witnesses (R2) -- the equal-capacity control HONOURS absence, so it is a
          second belief model rather than the world model the dissociation input requires.
      blind_scorer          (R3) -- (s1)'s two sides swapped, so the belief arm is compared to
          the control from the wrong direction and the dissociation is lost.
      consumer_ignores_payload (R4) -- the fold reads 0 in place of the payload, so the live and
          pinned arms coincide and `record_charge` refuses.
      nested_reads_world    (R5) -- the nested read returns the WORLD instead of the holder
          path's belief, so a stale pair reads reality and the losslessness claim is lost.
      feed_subject_only     (R6) -- the OLD subject-centric call the live path made until this
          row: one `observe_event` with a single subject, which is fact 4's spurious divergence."""

    def parse_anything(message):
        hit = _t082_parse(message)
        return hit if hit is not None else ("", "", False)

    def control_keeps_witnesses(tracker, step):
        return t082_feed(tracker, set(step["present"]), step["loc"])

    def blind_scorer(entry):
        return float(entry["acc_ctrl"]) - float(entry["acc_belief"])

    def consumer_ignores_payload(consumer):
        out = []
        prev = 0.0
        joined, _lost = consumer.poll_joined()
        for tick, _vec in joined:
            prev = max(min(1.0, T082_PROTECT_GAIN * 0.0), T082_PROTECT_DECAY * prev)
            out.append((tick, round(prev, 6)))
        return out

    def nested_reads_world(tracker, _observer, _target):
        return tracker.world_content()

    def feed_subject_only(tracker, present, loc):
        present = set(present)
        subject = "user" if "user" in present else "brain"
        tracker.observe_event(subject, DEFAULT_ATTITUDE, str(loc), witnesses=present)
        return tracker

    return {
        "parse_anything": parse_anything,
        "control_keeps_witnesses": control_keeps_witnesses,
        "blind_scorer": blind_scorer,
        "consumer_ignores_payload": consumer_ignores_payload,
        "nested_reads_world": nested_reads_world,
        "feed_subject_only": feed_subject_only,
    }[kind]


def t082_reds():
    """2.4's six reds as ONE self-check line (each is ALSO a test in
    `tests/test_recursive_tom.py`; this is the in-command control 2.4 requires, so a gate run
    that never executes pytest still runs every one on the SAME constructed input).

    Each value is True iff the red holds on its input AND its mutated scorer REDDENS it IN FACT
    -- every `not (...)` clause below is a broken twin having been RUN. No brain, no lake, no
    network: every subject is a constructed script or a finished recording."""
    out = {}

    # R1 -- THE PARSE REFUSES OUTSIDE THE GRAMMAR. The mutation parses anything.
    rec = t082_constructed_rows(seed=0, off_grammar_at=3)
    read = t082_read(rec["rows"])
    mut = t082_constructed_rows(
        seed=0, off_grammar_at=3, parse=_t082_mutated_scorer("parse_anything")
    )
    mut_read = t082_read(mut["rows"])
    out["R1"] = bool(
        _t082_parse(T082_OFF_GRAMMAR_STMT) is None
        and _t082_parse("i put the marble in the basket.") is not None
        and _t082_parse("someone moved the marble to the box while i was away.") is not None
        and read["parsed"] == read["n"] - 1
        and any("parse" in w for w in read["refusals"])
        and mut_read["parsed"] == mut_read["n"]
        and not mut_read["refusals"]
    )

    # R2 -- THE CONTROL IS A WORLD MODEL. The mutation honours absence.
    steps = t082_script(0, T082_R2_EVENTS)
    ctrl = RecursiveToM(n_agents=3, seed=0)
    hits = 0
    for st in steps:
        t082_feed(ctrl, set(T082_AGENTS), st["loc"])
        hits += ctrl.predict_first_order("user") == st["world"]
    keeps = _t082_mutated_scorer("control_keeps_witnesses")
    ctrl2 = RecursiveToM(n_agents=3, seed=0)
    mhits = 0
    for st in steps:
        keeps(ctrl2, st)
        mhits += ctrl2.predict_first_order("user") == st["world"]
    out["R2"] = bool(
        len(steps) == T082_R2_EVENTS
        and any(st["unwitnessed"] for st in steps)
        and hits == len(steps)
        and mhits < len(steps)
    )

    # R3 -- THE DISSOCIATION ON A CONSTRUCTED SCRIPT. The mutation swaps the two arms.
    mixed = [t082_constructed_read(seed=s) for s in T082_R_SEEDS]
    got = t082_contrasts(mixed)
    allw = [t082_constructed_read(seed=s, all_witnessed=True) for s in T082_R_SEEDS]
    got_w = t082_contrasts(allw)
    blind = t082_contrasts(mixed, score=_t082_mutated_scorer("blind_scorer"))
    out["R3"] = bool(
        got["n_seeds"] == len(T082_R_SEEDS)
        and got["s1"].met
        and got["s2"]["met"]
        and all(d == 0.0 for d in got_w["s1"].diffs)
        and not got_w["s1"].met
        and not blind["s1"].met
    )

    # R4 -- THE CHARGE A/B. The mutation's fold ignores the payload.
    live, pinned_out, reads, spec = _measure_belief_charge()
    m_live, m_pinned, m_reads, _ = _measure_belief_charge(
        consume=_t082_mutated_scorer("consumer_ignores_payload")
    )
    out["R4"] = bool(
        live is not None
        and live != pinned_out
        and reads == T082_R4_FRAMES
        and spec
        and m_reads == T082_R4_FRAMES
        and m_live == m_pinned
    )

    # R5 -- THE MERGE IS LOSSLESS. The mutation's nested read returns the world.
    agents = ("A", "B", "C")[:T082_R5_AGENTS]
    lossless = True
    for i in range(T082_R5_SCRIPTS):
        rt, ref, tom = _t082_r5_run(i, agents)
        for x in agents:
            if _t082_as_int(rt.predict_first_order(x)) != tom.predict_from_belief(x):
                lossless = False
            for y in agents:
                if x != y and _t082_as_int(rt.predict_nested(x, y)) != ref.predict_second_order(
                    x, y
                ):
                    lossless = False
    rt, ref, _tom = _t082_r5_run(0, agents)
    reads_world = _t082_mutated_scorer("nested_reads_world")
    differs = sum(
        1
        for x in agents
        for y in agents
        if x != y and _t082_as_int(reads_world(rt, x, y)) != ref.predict_second_order(x, y)
    )
    out["R5"] = bool(lossless and differs > 0)

    # R6 -- THE SYMMETRIC FEED KILLS THE SPURIOUS DIVERGENCE. The mutation is the old feed.
    both, brain_only = {"user", "brain"}, {"brain"}
    rt6 = RecursiveToM(n_agents=3, seed=0)
    t082_feed(rt6, both, "basket")
    t082_feed(rt6, brain_only, "box")
    stale = rt6.predict_first_order("user")
    t082_feed(rt6, both, "drawer")
    subject_only = _t082_mutated_scorer("feed_subject_only")
    rt6m = RecursiveToM(n_agents=3, seed=0)
    for present, loc in ((both, "basket"), (brain_only, "box"), (both, "drawer")):
        subject_only(rt6m, present, loc)
    out["R6"] = bool(
        stale == "basket"
        and rt6.divergence("user", "brain") is False
        and rt6.predict_first_order("brain") == rt6.world_content() == "drawer"
        and rt6m.divergence("user", "brain") is True
    )
    return out


def _t082_as_int(label):
    """The recursive tracker's content is a LABEL; the retired second-order semantics and
    `TheoryOfMind` key on an integer location. R5 compares the same quantity across all three."""
    return None if label is None else int(label)


def _t082_r5_run(i, agents):
    """R5's i-th constructed witness script, driven through all three trackers -> (recursive,
    retired-second-order reference, TheoryOfMind). The OPENING event is witnessed by ALL, so
    every first-order path and every ordered pair is initialised and the equivalence never rests
    on an unheld path (which all three would read as None, proving nothing)."""
    rng = random.Random(T082_R5_RNG_BASE + int(i))
    rt = RecursiveToM(n_agents=len(agents), seed=0)
    ref = t082_second_order_reference(n_locations=T082_R5_LOCATIONS, seed=0)
    tom = _t082_theory_of_mind(T082_R5_LOCATIONS)
    for ev in range(T082_R5_EVENTS):
        loc = rng.randrange(T082_R5_LOCATIONS)
        present = (
            set(agents) if ev == 0 else ({a for a in agents if rng.random() < 0.6} or {agents[0]})
        )
        t082_feed(rt, present, str(loc))
        ref.observe(loc, sorted(present))
        tom.place(loc, sorted(present))
    return rt, ref, tom


def _t082_theory_of_mind(n_locations):
    """R5's first-order control: the LIVE `cortex.theory_of_mind.TheoryOfMind` (that class STAYS
    -- `reasoning/pragmatics.py` runs its own instance). Imported lazily so this module's import
    graph does not grow for a control used only by the reds."""
    from snn_data_lake.cortex.theory_of_mind import TheoryOfMind

    return TheoryOfMind(n_locations=int(n_locations), seed=0)


def _t082_reds_line(reds, out=print):
    """Print the six booleans beside what each one says."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T082_RED_LINES[name]}")


def main():
    lines = []
    emit = lines.append
    emit("=" * 78)
    emit("RECURSIVE THEORY OF MIND -- N-agent, depth-k, VSA propositions beyond location")
    emit("=" * 78)

    # (a) FALSE-BELIEF TRANSFER (Sally-Anne): A sees the object at X, then leaves; B moves it to Y
    #     while A is ABSENT. A's belief must stay X (false) and diverge from the world (Y).
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})  # A and B both see it at X
    t.observe_event("B", "believes", "Y", witnesses={"B"})  # B moves it to Y; A is away
    a_look = t.predict_first_order("A")
    world = t.world_content("believes")
    div_world = a_look != world
    emit(f"(a) false belief : A believes {a_look}  world={world}  diverges-from-world={div_world}")
    assert a_look == "X", f"absent A must keep the false belief X, got {a_look}"
    assert world == "Y", f"world must reflect reality Y, got {world}"
    assert div_world, "A's belief must diverge from the world (the false-belief substrate)"

    # (b) SECOND-ORDER FALSE BELIEF: B watches A come to believe X, then A updates to Y with B ABSENT.
    #     B's nested model of A stays X while A's real belief is Y -> a divergent 2nd-order belief.
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("A", "believes", "X", witnesses={"A", "B"})  # B sees A believe X
    t.observe_event("A", "believes", "Y", witnesses={"A"})  # A updates to Y; B not present
    fo = t.predict_first_order("A")
    nested = t.predict_nested("B", "A")
    emit(
        f"(b) 2nd-order    : A really believes {fo}  B thinks A believes {nested}  "
        f"divergence(B,A)={t.divergence('B', 'A')}"
    )
    assert fo == "Y", f"A's real belief must be Y, got {fo}"
    assert nested == "X", f"B's nested model of A must stay X, got {nested}"
    assert t.divergence("B", "A") is True, "nested B-about-A must diverge from A's real belief"

    # (c) NON-LOCATION ATTITUDES (wants / knows) stored + retrieved, plus a nested want and a nested
    #     KNOWS false belief: A saw C want the cookie, but NOT C learn the secret.
    t = RecursiveToM(n_agents=3, seed=0)
    t.observe_event("C", "wants", "cookie", witnesses={"C", "A"})  # A sees C want the cookie
    t.observe_event("C", "knows", "secret", witnesses={"C"})  # C learns the secret; A absent
    c_wants = t.predict_first_order("C", "wants")
    c_knows = t.predict_first_order("C", "knows")
    a_thinks_c_wants = t.predict_nested("A", "C", "wants")
    a_thinks_c_knows = t.predict_nested("A", "C", "knows")
    emit(
        f"(c) attitudes    : C wants {c_wants}  C knows {c_knows}  "
        f"A-thinks-C-wants={a_thinks_c_wants}  A-thinks-C-knows={a_thinks_c_knows}"
    )
    assert c_wants == "cookie" and c_knows == "secret", "non-location content must store + retrieve"
    assert a_thinks_c_wants == "cookie", "A witnessed C's want -> nested want must read cookie"
    assert a_thinks_c_knows is None, (
        "A did NOT witness C learn the secret -> no nested knows belief"
    )

    # depth-3 nesting (k=3): C's model of B's model of A's belief, all three co-present.
    t3 = RecursiveToM(n_agents=3, seed=0)
    t3.observe_event("A", "believes", "P", witnesses={"A", "B", "C"})
    deep = t3.predict_chain(("C", "B", "A"))
    emit(f"    depth-3      : C-thinks-B-thinks-A-believes = {deep}  (nested belief depth 3)")
    assert deep == "P", f"depth-3 nested belief must read P, got {deep}"

    # (d) DETERMINISM + noise-robust spiking readout.
    u1 = RecursiveToM(n_agents=3, seed=0)
    u2 = RecursiveToM(n_agents=3, seed=0)
    for u in (u1, u2):
        u.observe_event("A", "believes", "X", witnesses={"A", "B"})
        u.observe_event("B", "believes", "Y", witnesses={"B"})
    det = u1.predict_first_order("A") == u2.predict_first_order("A") and u1.predict_nested(
        "B", "A"
    ) == u2.predict_nested("B", "A")
    noisy = u1.predict_first_order("A", jitter=2)  # perturb the population code
    emit(f"(d) determinism  : identical seeds agree={det}  jitter=2 still reads {noisy}")
    assert det, "identical seeds must produce identical predictions"
    assert noisy == "X", "spiking population readout must survive spike-count jitter"

    emit("=" * 78)
    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        "HONEST NEGATIVE: a FUNCTIONAL recursive belief tracker -- nested belief populations that "
        "can lag reality and each other -- NOT genuine mentalizing. Content is a VSA role-filler "
        "proposition read by a spiking LIF winner-take-all; scenarios are toy, the readout is "
        "near-tabular. Deterministic, ASCII."
    )
    print(
        "self-check OK: false-belief transfer holds (A keeps X vs world Y); a 2nd-order belief "
        "(B-about-A) diverges; non-location wants/knows store, retrieve and nest; depth-3 nesting "
        "reads through; deterministic and jitter-robust"
    )

    # ---- T082 (2.9 step 2): the six reds of 2.4 as ONE line, and the module's exit status.
    print("-" * 78)
    print(
        f"T082 -- social cognition composed | episodes {T082_N_EPISODES} | ticks between "
        f"{T082_TICKS_BETWEEN} | p_unwitnessed {T082_P_UNWITNESSED} | seeds {list(T082_SEEDS)} | "
        f"offset {T082_STREAM_OFFSET} x seed mod {T082_STREAM_POSITIONS} | bar {T082_BAR}"
    )
    print(f"  the declared belief wire: {BELIEF_EDGE}")
    reds = t082_reds()
    print("2.4's six reds:")
    _t082_reds_line(reds)
    if not all(reds.values()):
        print(
            f"T082 SELF-CHECK RED: only {sum(reds.values())}/{len(reds)} of 2.4's reds fire on "
            f"their constructed inputs -- {reds}"
        )
        print(
            "  EXPECTED AT 2.9 STEPS 2-3: the skeleton's stubs return null readings so the reds "
            "RUN red. This line goes green at 2.9 step 4, never before."
        )
        return 1
    print(
        f"  T082 OK: R1-R6 fire on their constructed inputs and each is reddened IN FACT by its "
        f"mutated scorer ({sum(reds.values())}/{len(reds)}); no brain, no lake, no network"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
