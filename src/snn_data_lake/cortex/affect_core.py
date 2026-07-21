"""
affect_core -- dimensional (valence, arousal) -> CATEGORICAL emotion, affective CONTAGION,
and SECOND-ORDER theory of mind.  Deterministic.

Where the repo stood (honest baseline this module deepens):
  * affect was ONE 1-D approach/avoid valence scalar (basal_ganglia/valence_stdp.ValenceLearner,
    cortex/self_model valence axis) plus an |rpe| / threat arousal PROXY -- no discrete emotions.
  * theory of mind was FIRST-ORDER, WHERE-only (cortex/theory_of_mind.TheoryOfMind: one belief
    population per agent that can lag the world -> a first-order false belief).
  * empathy, emotion-recognition, pragmatics, and SECOND-ORDER ToM were ABSENT.

This module adds four faculties on top of that substrate, reusing the repo's spiking primitives
(cortex/lif_neuron.LIFNeuron and cortex/theory_of_mind.BeliefPopulation) rather than reinventing:

  (a) CATEGORICAL EMOTION from the 2-D core.  Russell's affect CIRCUMPLEX (Russell 1980,
      "A circumplex model of affect", JPSP 39:1161-1178) lays discrete emotions out as anchors on
      a circle whose axes are valence (x) and arousal (y).  AffectCore places prototype anchors
      (joy/excited/fear/anger/sadness/boredom/calm/contentment) at their canonical circumplex
      angles and classifies an incoming (valence, arousal) point by a SPIKING winner-take-all:
      one LIFNeuron per emotion, each driven by a current proportional to the point's radial-basis
      similarity to that anchor (soft lateral-inhibition sharpening), the most-firing emotion wins.
      This is the same population-code + spiking-WTA readout theory_of_mind uses for WHERE-beliefs,
      repurposed for WHICH-emotion.  Crucially it can tell FEAR from ANGER -- both low-valence,
      high-arousal, but at distinct circumplex angles (the tightest, most honest pair: 30 deg apart).

  (b) RECOGNITION.  recognize(valence, arousal) -> label, and recognize_episode(ep) reads the
      (valence, arousal) tags off an affect-tagged percept/episode (duck-typed .valence/.arousal,
      e.g. a hippocampus.affective_memory.Episode) and names the emotion.

  (c) AFFECTIVE EMPATHY / CONTAGION (Hatfield, Cacioppo & Rapson 1993, "Emotional contagion").
      empathize(other_v, other_a, gain) shifts the agent's OWN affect toward another agent's
      affect by a contagion gain -- a local leaky-integrator pull (convex combination), NOT a
      gradient.  Repeated exposure converges own affect onto the other's and can FLIP the agent's
      own categorical emotion (a calm agent next to a terrified one drifts toward fear).

  (d) SECOND-ORDER ToM (WHERE-only).  SecondOrderToM tracks "what A believes B believes" about an
      object's location as a NESTED belief population -- the first-order idea extended one level.
      belief2[(A,B)] updates only when A witnesses B observe; when A does not, A's model of B's
      belief goes STALE -> a SECOND-ORDER false belief.  It resolves the nested (ice-cream-van,
      Perner & Wimmer 1985) task: John thinks Mary thinks the van is at the park, though Mary has
      since learned it moved -- a false belief John holds ABOUT Mary's belief, distinct from both
      Mary's actual (first-order) belief and the world.

HONEST FRAMING.  This is a dimensional->categorical MAPPING plus nested-belief BOOKKEEPING. The
anchors are a structural prior (Russell's circumplex), not learned; classification is a spiking
nearest-prototype readout; contagion is a local dynamical pull; the nested ToM is stale-vs-fresh
population persistence.  It is NOT felt emotion: nothing here experiences fear, anger, or empathy,
and it makes no claim of phenomenal affect.  It models THAT affect sorts into categories, that an
agent's affect can be pulled by another's, and that a belief can be held about a belief -- not
WHAT any of that is like.  Toy scenarios, near-tabular readouts.

Determinism / style: per-instance random.Random(seed); ASCII-only output; NO import-time side
effects (all work is under __main__ / explicit calls).  Additive and safety-inert: this module
imports NOTHING from brainstem, touches no actuator and no safety-envelope file, never selects an
action and never emits a motor command.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.affect_core
"""

import collections
import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.theory_of_mind import BeliefPopulation

# Russell-circumplex prototype anchors as (name, angle_deg). The affect circle is centred on
# neutral (valence 0, arousal 0.5); an anchor at angle t has valence = cos(t), arousal = 0.5 +
# 0.5*sin(t). Angles follow Russell's layout: pleasant-high-arousal (joy) upper-right, unpleasant-
# high-arousal (fear then anger) upper-left, unpleasant-low-arousal (sadness/boredom) lower-left,
# pleasant-low-arousal (calm/contentment) lower-right. Fear (120) and anger (150) sit 30 deg apart
# -- the tightest pair, the honest hard case for telling two negative high-arousal states apart.
_PROTOTYPES = (
    ("joy", 45.0),
    ("excited", 80.0),
    ("fear", 120.0),
    ("anger", 150.0),
    ("sadness", 205.0),
    ("boredom", 245.0),
    ("calm", 310.0),
    ("contentment", 350.0),
)

_SIGMA = 0.5  # RBF width on the circumplex (unit-radius anchors; min anchor gap 30 deg)
_SHARP = 4  # WTA contrast exponent: soft lateral inhibition sharpening winner over runners
_WINDOW = 50  # LIF drive window for the spiking winner-take-all readout
_TAU_M = 10.0  # LIF membrane time constant
_DRIVE = 0.30  # input current for the peak-similarity emotion (others scaled by (sim/peak)**_SHARP)


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _geom(valence, arousal):
    """Map a (valence, arousal) affect point to circumplex geometry (x, y). valence is the x-axis;
    arousal in [0,1] maps to y = 2*arousal - 1 so neutral arousal 0.5 sits on the x-axis."""
    return (valence, 2.0 * arousal - 1.0)


class AffectCore:
    """Dimensional (valence, arousal) core with a categorical emotion readout + affective contagion.

    Holds the agent's OWN (valence, arousal) affect. classify/recognize name the discrete emotion of
    ANY (valence, arousal) point via a spiking winner-take-all over Russell-circumplex prototype
    anchors; empathize() pulls the agent's own affect toward another agent's affect (contagion).
    Anchors are a fixed structural prior (not learned); nothing here experiences the emotions it names."""

    def __init__(self, own_valence=0.0, own_arousal=0.5, seed=0, sigma=_SIGMA):
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance RNG (readout jitter / noise-robustness path)
        self.sigma = sigma
        self.own_v = _clamp(float(own_valence), -1.0, 1.0)
        self.own_a = _clamp(float(own_arousal), 0.0, 1.0)
        self.emotions = [name for name, _ in _PROTOTYPES]
        # prototype anchors in circumplex geometry: (cos t, sin t) == (valence, 2*arousal-1) at r=1
        self._proto = [
            (math.cos(math.radians(t)), math.sin(math.radians(t))) for _, t in _PROTOTYPES
        ]

    # ---- (a)/(b) categorical emotion readout --------------------------------------------------
    def _similarities(self, valence, arousal):
        """Radial-basis similarity of an affect point to every prototype anchor (Gaussian on the
        circumplex). Each is in (0, 1]; the nearest anchor scores highest."""
        gx, gy = _geom(valence, arousal)
        s2 = 2.0 * self.sigma * self.sigma
        return [math.exp(-((gx - px) ** 2 + (gy - py) ** 2) / s2) for (px, py) in self._proto]

    def classify_index(self, valence, arousal):
        """Analytic nearest-prototype index (exact argmax similarity; ties -> lowest index). The
        SPIKING readout below reproduces this; this is the reference the readout is checked against."""
        sims = self._similarities(valence, arousal)
        return max(range(len(sims)), key=lambda i: (sims[i], -i))

    def _spiking_wta(self, sims, jitter=0.0):
        """Spiking winner-take-all: one LIFNeuron per emotion, each driven for _WINDOW steps by a
        current _DRIVE*(sim/peak)**_SHARP (soft lateral inhibition sharpens the winner). The most-
        firing emotion wins (ties -> lowest index). `jitter` (>0) perturbs the normalised similarities
        via the per-instance RNG to show the readout is noise-robust. Returns an emotion index."""
        peak = max(sims)
        if peak <= 0.0:
            return None
        counts = []
        for s in sims:
            ratio = s / peak
            if jitter:
                ratio = _clamp(ratio + self.rng.uniform(-jitter, jitter), 0.0, 1.0)
            current = _DRIVE * (ratio**_SHARP)
            neuron = LIFNeuron(tau_m=_TAU_M)
            counts.append(sum(neuron.run([current] * _WINDOW)))
        return max(range(len(counts)), key=lambda i: (counts[i], -i))

    def recognize_index(self, valence, arousal, jitter=0.0):
        """Spiking-WTA emotion index for an affect point."""
        return self._spiking_wta(self._similarities(valence, arousal), jitter=jitter)

    def recognize(self, valence, arousal, jitter=0.0):
        """Recognise the categorical emotion of an affect point -> label (spiking-WTA readout)."""
        return self.emotions[self.recognize_index(valence, arousal, jitter=jitter)]

    def recognize_episode(self, episode, jitter=0.0):
        """Name the emotion of an affect-tagged percept/episode. Duck-typed: reads .valence/.arousal
        (e.g. a hippocampus.affective_memory.Episode) or ['valence']/['arousal'] off a mapping."""
        if hasattr(episode, "valence") and hasattr(episode, "arousal"):
            v, a = episode.valence, episode.arousal
        else:
            v, a = episode["valence"], episode["arousal"]
        return self.recognize(v, a, jitter=jitter)

    # ---- (c) affective empathy / contagion ----------------------------------------------------
    def affect(self):
        """The agent's current own (valence, arousal)."""
        return (self.own_v, self.own_a)

    def own_emotion(self, jitter=0.0):
        """The categorical label of the agent's own current affect."""
        return self.recognize(self.own_v, self.own_a, jitter=jitter)

    def empathize(self, other_v, other_a, gain=0.4):
        """Affective contagion: pull the agent's OWN affect toward another agent's (other_v, other_a)
        by a contagion `gain` in [0,1] -- a local convex/leaky-integrator update, NOT a gradient:
            own <- own + gain*(other - own).
        gain=0 leaves own affect unchanged; gain=1 snaps onto the other; repeated calls converge.
        Returns the new own (valence, arousal)."""
        g = _clamp(float(gain), 0.0, 1.0)
        self.own_v = _clamp(self.own_v + g * (float(other_v) - self.own_v), -1.0, 1.0)
        self.own_a = _clamp(self.own_a + g * (float(other_a) - self.own_a), 0.0, 1.0)
        return (self.own_v, self.own_a)


class SecondOrderToM:
    """SECOND-ORDER, WHERE-only theory of mind: track what agent A believes agent B believes about
    an object's location, as a NESTED belief population.

    Reuses cortex.theory_of_mind.BeliefPopulation (spiking population code + LIF winner-take-all
    readout) for every belief, at both levels. observe(location, observers) updates the world, each
    PRESENT agent's first-order belief, AND -- for every ordered pair of present agents (x, y) -- x's
    model of y's belief (x witnessed y observe). Agents/pairs not present are untouched, so their
    belief goes STALE relative to the world: a first- or SECOND-order FALSE belief."""

    def __init__(self, n_locations=4, seed=0):
        self.n_locations = n_locations
        self.seed = seed
        self.world = None
        self.belief = {}  # agent -> BeliefPopulation (first-order: what agent believes)
        self.belief2 = {}  # (x, y) -> BeliefPopulation (second-order: what x believes y believes)

    def _b(self, agent):
        if agent not in self.belief:
            self.belief[agent] = BeliefPopulation(self.n_locations, seed=self.seed)
        return self.belief[agent]

    def _b2(self, x, y):
        if (x, y) not in self.belief2:
            self.belief2[(x, y)] = BeliefPopulation(self.n_locations, seed=self.seed)
        return self.belief2[(x, y)]

    def observe(self, location, observers):
        """The object is placed/seen at `location`. Update the world; every present agent's first-
        order belief; and, for every ordered pair of present agents, the first agent's model of the
        second agent's belief (each present agent witnesses the others observing)."""
        self.world = location
        obs = list(observers)
        for x in obs:
            self._b(x).observe(location)
        for x in obs:
            for y in obs:
                if x != y:
                    self._b2(x, y).observe(location)

    def predict_first_order(self, agent, jitter=0):
        """Where `agent` will look, read from ITS OWN tracked belief (first-order)."""
        return self._b(agent).believed_location(jitter=jitter)

    def predict_second_order(self, x, y, jitter=0):
        """Where `x` thinks `y` will look, read from x's tracked model of y's belief (second-order)."""
        return self._b2(x, y).believed_location(jitter=jitter)

    def predict_from_world(self):
        """CONTROL predictor: ignore all minds, return reality. Fails every false-belief case."""
        return self.world


# ============================ un-riggable self-check ==========================
def _loc(i):
    return "?" if i is None else chr(ord("A") + int(i))


# a minimal affect-tagged episode, duck-typed like hippocampus.affective_memory.Episode
_Episode = collections.namedtuple("_Episode", ["valence", "arousal"])


def _proto_va(angle_deg, radius):
    """A (valence, arousal) point at circumplex `angle_deg` and `radius` (radius 1 == an anchor)."""
    t = math.radians(angle_deg)
    return (radius * math.cos(t), 0.5 + 0.5 * radius * math.sin(t))


def _confusion_section(emit):
    """(1) CATEGORICAL SEPARABILITY: sample points inside each emotion's circumplex wedge, classify
    with the SPIKING readout, and prove each region maps to its own label -- vs a random-label control
    at chance. Reports the per-emotion accuracy, the confusion matrix, and the fear/anger hard pair."""
    core = AffectCore(seed=0)
    names = core.emotions
    k = len(names)
    rng = random.Random(20240718)
    per_class = 40
    confusion = [[0] * k for _ in range(k)]  # confusion[true][pred]
    spike_vs_analytic_disagree = 0
    total = 0
    rand_hits = 0

    for true_idx, (_, angle) in enumerate(_PROTOTYPES):
        for _ in range(per_class):
            # sample strictly inside the emotion's Voronoi wedge: angle +/- 8 deg, radius in [0.8, 1.0]
            a = angle + rng.uniform(-8.0, 8.0)
            r = rng.uniform(0.8, 1.0)
            v, ar = _proto_va(a, r)
            pred = core.recognize_index(v, ar)  # spiking-WTA label
            analytic = core.classify_index(v, ar)  # exact nearest prototype
            if pred != analytic:
                spike_vs_analytic_disagree += 1
            confusion[true_idx][pred] += 1
            if rng.randrange(k) == true_idx:  # random-label control
                rand_hits += 1
            total += 1

    correct = sum(confusion[i][i] for i in range(k))
    acc = correct / total
    rand_acc = rand_hits / total
    fear_i, anger_i = names.index("fear"), names.index("anger")
    fear_anger_confuse = confusion[fear_i][anger_i] + confusion[anger_i][fear_i]

    emit("(1) CATEGORICAL EMOTION SEPARABILITY (spiking-WTA over Russell circumplex)")
    emit(f"    {k} emotions, {per_class} sampled points each ({total} total)")
    emit("    per-emotion accuracy:")
    for i, name in enumerate(names):
        emit(
            f"      {name:<12} {confusion[i][i]:>2}/{per_class} = {confusion[i][i] / per_class:.2f}"
        )
    emit(f"    overall accuracy      = {acc:.3f}   (spiking-WTA)")
    emit(f"    random-label control  = {rand_acc:.3f}   (chance ~ 1/{k} = {1 / k:.3f})")
    emit(f"    fear<->anger confusions = {fear_anger_confuse}  (the tightest pair, 30 deg apart)")
    emit(
        f"    spiking-WTA vs analytic nearest-prototype disagreements = {spike_vs_analytic_disagree}"
    )

    assert acc == 1.0, f"clean-wedge categorical accuracy must be confusion-free, got {acc:.3f}"
    assert fear_anger_confuse == 0, (
        f"fear and anger must be separated ({fear_anger_confuse} confusions)"
    )
    assert spike_vs_analytic_disagree == 0, (
        "spiking-WTA readout must reproduce the analytic nearest-prototype"
    )
    assert rand_acc < 0.5 * acc, f"random-label control must be near chance, got {rand_acc:.3f}"
    return acc, rand_acc, fear_anger_confuse


def _recognition_section(emit):
    """(2) RECOGNITION from an affect-tagged episode: name the emotion off an Episode's (valence,
    arousal) tags. Distinct affect tags -> distinct correct labels; a wrong-tag control mislabels."""
    core = AffectCore(seed=0)
    cases = [
        ("fear", _proto_va(120.0, 0.95)),
        ("anger", _proto_va(150.0, 0.95)),
        ("joy", _proto_va(45.0, 0.95)),
        ("sadness", _proto_va(205.0, 0.95)),
        ("calm", _proto_va(310.0, 0.95)),
    ]
    emit("(2) RECOGNITION from affect-tagged episodes (recognize_episode -> label)")
    hits = 0
    for expected, (v, a) in cases:
        ep = _Episode(valence=v, arousal=a)
        got = core.recognize_episode(ep)
        ok = got == expected
        hits += ok
        emit(
            f"    episode (v={v:+.2f}, a={a:.2f}) -> {got:<11} (expected {expected}) {'OK' if ok else 'MISS'}"
        )
    # control: a joy-tagged episode must NOT read as fear (distinct regions -> distinct labels)
    joy_ep = _Episode(valence=_proto_va(45.0, 0.95)[0], arousal=_proto_va(45.0, 0.95)[1])
    assert core.recognize_episode(joy_ep) != "fear", (
        "a joyful episode must not be recognised as fear"
    )
    assert hits == len(cases), "recognition must name every affect-tagged episode correctly"
    emit(f"    recognised {hits}/{len(cases)} episodes correctly")
    return hits, len(cases)


def _contagion_section(emit):
    """(3) AFFECTIVE CONTAGION: a calm agent exposed to a terrified agent drifts measurably toward
    the other's affect and eventually FLIPS its own categorical emotion. gain=0 and random-shift
    controls do NOT move systematically toward the other."""

    def dist(core, tv, ta):
        gx, gy = _geom(core.own_v, core.own_a)
        ox, oy = _geom(tv, ta)
        return math.hypot(gx - ox, gy - oy)

    other_v, other_a = _proto_va(120.0, 0.98)  # a terrified other (fear region)
    start_v, start_a = _proto_va(310.0, 0.90)  # a calm agent

    core = AffectCore(own_valence=start_v, own_arousal=start_a, seed=1)
    start_emotion = core.own_emotion()
    d0 = dist(core, other_v, other_a)
    dists = [d0]
    for _ in range(8):
        core.empathize(other_v, other_a, gain=0.4)
        dists.append(dist(core, other_v, other_a))
    end_emotion = core.own_emotion()
    d_end = dists[-1]

    # gain=0 control: no movement
    ctl0 = AffectCore(own_valence=start_v, own_arousal=start_a, seed=1)
    ctl0.empathize(other_v, other_a, gain=0.0)
    d_ctl0 = dist(ctl0, other_v, other_a)

    # random-shift control: move own affect by random deltas of comparable magnitude (not toward other)
    ctlr = AffectCore(own_valence=start_v, own_arousal=start_a, seed=1)
    rng = random.Random(7)
    for _ in range(8):
        ctlr.own_v = _clamp(ctlr.own_v + rng.uniform(-0.4, 0.4), -1.0, 1.0)
        ctlr.own_a = _clamp(ctlr.own_a + rng.uniform(-0.2, 0.2), 0.0, 1.0)
    d_ctlr = dist(ctlr, other_v, other_a)

    emit("(3) AFFECTIVE CONTAGION / EMPATHY (own affect pulled toward another's)")
    emit(
        f"    start: own={start_emotion} (v={start_v:+.2f}, a={start_a:.2f}); "
        f"other=fear (v={other_v:+.2f}, a={other_a:.2f})"
    )
    emit(
        f"    distance to other over 8 contagion steps: {d0:.3f} -> {d_end:.3f} "
        f"(reduced {100 * (d0 - d_end) / d0:.0f}%)"
    )
    emit(f"    own emotion flipped: {start_emotion} -> {end_emotion}")
    emit(
        f"    control gain=0.0 distance = {d_ctl0:.3f} (unchanged); random-shift distance = {d_ctlr:.3f}"
    )

    assert all(dists[i + 1] < dists[i] for i in range(len(dists) - 1)), (
        "contagion must move own affect monotonically toward the other"
    )
    assert d_end < 0.25 * d0, (
        f"repeated contagion must converge onto the other ({d_end:.3f} vs {d0:.3f})"
    )
    assert end_emotion == "fear" and start_emotion != "fear", (
        f"contagion must flip the own categorical emotion toward the other ({start_emotion}->{end_emotion})"
    )
    assert abs(d_ctl0 - d0) < 1e-9, "gain=0 must not move own affect"
    assert d_ctlr > d_end, "a random shift must NOT converge onto the other the way contagion does"
    return d0, d_end, start_emotion, end_emotion


def _run_van(park, church, told_by, seed=0):
    """Stage the second-order ice-cream-van task on n_locations=4.
    1. Van at `park`; John AND Mary present (each sees the other see it).
    2. Mary leaves; van moves to `church`; only John witnesses (Mary absent).
    3. The vendor tells Mary the van moved to `church`. `told_by` lists who WITNESSES the telling
       -- ('Mary',) for the classic second-order FALSE belief (John does not witness it),
       ('John','Mary') for the true-belief control (John witnesses Mary being told)."""
    tom = SecondOrderToM(n_locations=4, seed=seed)
    tom.observe(park, observers=("John", "Mary"))
    tom.observe(church, observers=("John",))
    if told_by:
        tom.observe(church, observers=told_by)
    return tom


def _second_order_section(emit):
    """(4) SECOND-ORDER ToM: the nested (ice-cream-van) false belief resolves correctly -- John thinks
    Mary thinks the van is at the PARK, though Mary has learned it moved. Distinct from Mary's actual
    (first-order) belief AND the world; a random guesser is at chance; a true-belief control agrees."""
    emit("(4) SECOND-ORDER THEORY OF MIND (nested WHERE-belief, ice-cream-van)")
    scenarios = [(0, 2), (1, 3), (2, 0), (3, 1)]  # (park, church), varied so nothing is hardcoded
    second_hits = 0
    world_hits = 0
    first_hits = 0
    rng = random.Random(99)
    rand_hits = 0
    n_loc = 4
    for park, church in scenarios:
        tom = _run_van(park, church, told_by=("Mary",))
        so = tom.predict_second_order(
            "John", "Mary"
        )  # John's model of Mary's belief -> PARK (false)
        fo = tom.predict_first_order("Mary")  # Mary's actual belief          -> CHURCH
        wo = tom.predict_from_world()  # reality                       -> CHURCH
        correct = park  # the second-order false-belief answer
        second_hits += so == correct
        first_hits += fo == correct
        world_hits += wo == correct
        rand_hits += rng.randrange(n_loc) == correct
        emit(
            f"    van {_loc(park)}->{_loc(church)} (Mary told, John unaware): "
            f"2nd-order(John of Mary)={_loc(so)}  Mary's-belief={_loc(fo)}  world={_loc(wo)}  "
            f"answer={_loc(correct)}"
        )
        assert so == park, (
            f"second-order belief must be the stale PARK basket {_loc(park)}, got {_loc(so)}"
        )
        assert fo == church and wo == church, "Mary's actual belief and the world are the CHURCH"

    n = len(scenarios)
    emit(
        f"    second-order predictor accuracy = {second_hits / n:.2f}  (resolves the nested false belief)"
    )
    emit(
        f"    first-order (Mary's belief)     = {first_hits / n:.2f}  (CONTROL: predicts CHURCH, wrong)"
    )
    emit(
        f"    world (reality) predictor       = {world_hits / n:.2f}  (CONTROL: predicts CHURCH, wrong)"
    )
    emit(f"    random guesser                  = {rand_hits / n:.2f}  (chance ~ 1/{n_loc})")

    assert second_hits == n, "second-order predictor must pass every nested false-belief case"
    assert first_hits == 0 and world_hits == 0, (
        "first-order and world predictors must FAIL the second-order question (they predict the church)"
    )

    # true-belief control: John WITNESSES Mary being told -> John's model of Mary updates to CHURCH
    true_tom = _run_van(0, 2, told_by=("John", "Mary"))
    so_true = true_tom.predict_second_order("John", "Mary")
    fo_true = true_tom.predict_first_order("Mary")
    emit(
        f"    true-belief control (John sees Mary told): 2nd-order={_loc(so_true)} == Mary's-belief="
        f"{_loc(fo_true)}  (agree -> divergence is specific to the unwitnessed telling)"
    )
    assert so_true == fo_true == 2, (
        "when John witnesses the telling, his second-order belief tracks Mary's"
    )
    return second_hits, n, first_hits, world_hits


def main():
    lines = []

    def emit(s):
        lines.append(s)

    emit("=" * 78)
    emit("AFFECT CORE -- dimensional->categorical emotion + contagion + second-order ToM")
    emit("=" * 78)

    acc, rand_acc, fa = _confusion_section(emit)
    emit("-" * 78)
    rec_hits, rec_n = _recognition_section(emit)
    emit("-" * 78)
    d0, d_end, start_emotion, end_emotion = _contagion_section(emit)
    emit("-" * 78)
    so_hits, so_n, fo_hits, wo_hits = _second_order_section(emit)
    emit("-" * 78)

    # (5) DETERMINISM + ASCII
    a1 = AffectCore(own_valence=0.6, own_arousal=0.2, seed=3)
    a2 = AffectCore(own_valence=0.6, own_arousal=0.2, seed=3)
    a1.empathize(-0.5, 0.9, gain=0.4)
    a2.empathize(-0.5, 0.9, gain=0.4)
    det_affect = a1.affect() == a2.affect() and a1.own_emotion() == a2.own_emotion()
    t1 = _run_van(1, 3, told_by=("Mary",))
    t2 = _run_van(1, 3, told_by=("Mary",))
    det_tom = t1.predict_second_order("John", "Mary") == t2.predict_second_order("John", "Mary")
    # noise-robust categorical readout: jittered similarities still recover the label
    noisy = AffectCore(seed=5).recognize(*_proto_va(120.0, 0.95), jitter=0.05)
    emit("(5) DETERMINISM + NOISE-ROBUSTNESS + ASCII")
    emit(f"    identical seeds/inputs -> identical affect+emotion : {det_affect}")
    emit(f"    identical scenario -> identical second-order answer : {det_tom}")
    emit(f"    jitter=0.05 on similarities still recognises fear   : {noisy == 'fear'}")
    assert det_affect and det_tom, (
        "identical inputs produced different outputs -- not deterministic"
    )
    assert noisy == "fear", "spiking categorical readout must survive similarity jitter"

    emit("=" * 78)

    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        "HONEST NEGATIVE: this is dimensional->categorical MAPPING (fixed Russell-circumplex anchors, "
        "a spiking nearest-prototype readout) + nested-belief BOOKKEEPING (stale-vs-fresh population "
        "persistence). It is NOT felt emotion -- nothing here experiences fear, anger, or empathy; the "
        "scenarios are toy and the readouts near-tabular. The fear/anger pair is the tightest (30 deg) "
        "and is the honest hard case."
    )
    print(
        f"self-check OK: {len(AffectCore().emotions)} emotions separable (accuracy {acc:.2f}, "
        f"fear/anger confusions {fa}) vs random-label control {rand_acc:.2f}; recognised {rec_hits}/{rec_n} "
        f"affect-tagged episodes; contagion converged {d0:.2f}->{d_end:.2f} and flipped "
        f"{start_emotion}->{end_emotion}; second-order false belief resolved {so_hits}/{so_n} while the "
        f"first-order and world controls failed ({fo_hits}/{so_n}, {wo_hits}/{so_n}); deterministic, ASCII"
    )


if __name__ == "__main__":
    main()
