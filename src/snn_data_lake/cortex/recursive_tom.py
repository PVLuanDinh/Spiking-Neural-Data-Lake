"""
Recursive Theory of Mind -- N agents, depth-k nested beliefs, content BEYOND location.

The repo already has a first-order WHERE-tracker (cortex/theory_of_mind.BeliefPopulation) and a
WHERE-only second-order tracker (cortex/affect_core.SecondOrderToM). Both hold a single believed
LOCATION for one observer. This module generalises on three axes at once:

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
from snn_data_lake.substrate.paired_contrast import T_BAR

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
# 2.9 STEP 2's STUBS. Each returns a NULL reading, so the six reds of 2.4 RUN RED at step 3
# against the real thing rather than against an import error. Every one is implemented at
# step 4 (the instruments) or step 5 (the recorder).
# =============================================================================
def t082_script(seed, n_ep=T082_N_EPISODES):
    """STUB (2.9 step 2) -> []. THE SCRIPTED OTHER AGENT (2.1): objects and locations drawn
    under `random.Random(T082_RNG_BASE + seed)`, p_unwitnessed = `T082_P_UNWITNESSED`, the
    agent's look = its last WITNESSED location, the world = the last location."""
    del seed, n_ep
    return []


def t082_record_brain(seed, n_ep=T082_N_EPISODES):
    """STUB (2.9 step 2) -> an empty recording. THE RECORDER (2.1, landed at step 5): a scratch
    brain under `throwaway_home()`, `make_brain(seed, persist=False)`, the script driven through
    `Brain.respond` with `T082_TICKS_BETWEEN` ticks between statements."""
    return {"seed": int(seed), "n_ep": int(n_ep), "rows": []}


def t082_read(rows):
    """STUB (2.9 step 2) -> {}. The per-seed reading of 2.1: acc_belief / acc_world / acc_ctrl
    against the agent's look, the unwitnessed fraction, the parse count, the spurious
    nested-divergence count on witnessed-only stretches, the social-risk trajectory."""
    del rows
    return {}


def t082_contrasts(per_seed):
    """STUB (2.9 step 2) -> {}. 2.3's conjuncts through `substrate.paired_contrast.contrast`."""
    del per_seed
    return {}


def _measure_belief_charge():
    """STUB (2.9 step 2) -> (None, None, [], False). T060's form over recorded belief frames:
    `port_charge_probe.replay_twice`, the consumer = the brain's own `_tom_protect` fold
    reduced to the caution trajectory, the pinned payload `T082_PINNED_BELIEF`, the
    specificity control a constant consumer."""
    return None, None, [], False


def record_charge():
    """STUB (2.9 step 2) -> 1 (a refusal). Measures the belief wire's A/B and records the
    attribution through THE one writer (`port_charge_probe.record_attribution`); REFUSES a
    non-diverged A/B -- evidence is a measurement, never a declaration."""
    print("record_charge REFUSED: the T082 charge instrument is still 2.9 step 2's stub")
    return 1


def t082_reds():
    """STUB (2.9 step 2) -> six False. 2.4's six reds as ONE self-check line: each True iff the
    red holds on its constructed input AND its mutated scorer REDDENS it IN FACT. Null here so
    `main()` carries the red until 2.9 step 4, exactly as 2.9 orders."""
    return dict.fromkeys(("R1", "R2", "R3", "R4", "R5", "R6"), False)


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
