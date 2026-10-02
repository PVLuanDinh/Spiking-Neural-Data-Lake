"""
affect_core -- dimensional (valence, arousal) -> CATEGORICAL emotion and affective CONTAGION.
Deterministic.

T082 RETIREMENT (2026-09-16, specs/022 evidence/t082-social-cognition.md 2.1): this module also
carried `SecondOrderToM`, a WHERE-only second-order belief tracker, and it is DELETED here with
its ice-cream-van self-check section. It was one of THREE trackers the same dialogue statement
fed in `Brain._tom_update`, and the chat, the state key and the telemetry each read a different
one, so one turn had three answers (the row's fact 1). `faculties/recursive_tom.RecursiveToM` under
the symmetric feed generalises it (N agents, depth k, content beyond location), and red R5
measures the equivalence on 200 constructed witness scripts before the deletion -- 1200/1200
ordered pairs identical to this class, and `recursive_tom.t082_second_order_reference` (which
survives, as R5's control) identical to it on the same 1200. P2: exactly one mechanism survives.

Where the repo stood (honest baseline this module deepens):
  * affect was ONE 1-D approach/avoid valence scalar (plasticity/valence_stdp.ValenceLearner,
    faculties/self_model valence axis) plus an |rpe| / threat arousal PROXY -- no discrete emotions.
  * empathy and emotion-recognition were ABSENT.

This module adds three faculties on top of that substrate, reusing the repo's spiking primitives
(growth/lif_neuron.LIFNeuron) rather than reinventing:

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
      e.g. a faculties.affective_memory.Episode) and names the emotion.

  (c) AFFECTIVE EMPATHY / CONTAGION (Hatfield, Cacioppo & Rapson 1993, "Emotional contagion").
      empathize(other_v, other_a, gain) shifts the agent's OWN affect toward another agent's
      affect by a contagion gain -- a local leaky-integrator pull (convex combination), NOT a
      gradient.  Repeated exposure converges own affect onto the other's and can FLIP the agent's
      own categorical emotion (a calm agent next to a terrified one drifts toward fear).

HONEST FRAMING.  This is a dimensional->categorical MAPPING.  The anchors are a structural prior
(Russell's circumplex), not learned; classification is a spiking nearest-prototype readout;
contagion is a local dynamical pull.  It is NOT felt emotion: nothing here experiences fear,
anger, or empathy, and it makes no claim of phenomenal affect.  It models THAT affect sorts into
categories and that an agent's affect can be pulled by another's -- not WHAT any of that is like.
Toy scenarios, near-tabular readouts.

P6 -- MEASURED NULL, 2026-08-05, recorded because it is the result rather than the absence of one.
This module was examined for a faculty-to-faculty read in the same pass that wired
faculties/metacognitive_monitor.py to faculties/executive_control.py, and NO wire was shipped here. The
reason is a measurement, not a judgement: this faculty's OWN affect is a live CONSTANT.  Measured on
a hermetic brain (`make_brain(seed=0, persist=False)`, 300 ticks, one process): `affect()` returns
(0.0, 0.5) on 300/300 ticks -- ONE distinct value.  The cause is structural, not a tuning accident:
`own_v` / `own_a` are written ONLY by `empathize()`, whose single live call site is inside
`Brain.respond()` and fires only when a chat turn mentions another agent's affect, so a tick run
never touches them.  And the reads are worse than the writes -- tree-wide, `own_emotion()` and
`affect()` have ZERO callers outside this module's own self-check.  What brain.py actually consumes
is `recognize(valence, arousal)`, a STATELESS call whose arguments come from the self-model and the
arm's RPE, bypassing own affect entirely.

So feeding a peer faculty's output into own affect would move a value that nothing reads -- P6's
"a route that carries nothing", and P3 shape 5 (a constant wearing a signal's clothes) sitting
underneath it.  The honest disposition is: this is P3 shape-5 dormancy in `own_v`/`own_a`, the fix
is a CONSUMER for own affect rather than another producer into it, and until that consumer exists
no wire in is shippable.  That is open work, not a settled state.

Determinism / style: per-instance random.Random(seed); ASCII-only output; NO import-time side
effects (all work is under __main__ / explicit calls).  Additive and safety-inert: this module
imports NOTHING from brainstem, touches no actuator and no safety-envelope file, never selects an
action and never emits a motor command.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.affect_core
"""

import collections
import math
import random

from snn_data_lake.growth.lif_neuron import LIFNeuron

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
        (e.g. a faculties.affective_memory.Episode) or ['valence']/['arousal'] off a mapping."""
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


# ============================ un-riggable self-check ==========================
# (`_loc`, the integer-location renderer, went with `SecondOrderToM` at T082: nothing else in
# this module speaks in location ids.)
# a minimal affect-tagged episode, duck-typed like faculties.affective_memory.Episode
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


def main():
    lines = []

    def emit(s):
        lines.append(s)

    emit("=" * 78)
    emit("AFFECT CORE -- dimensional->categorical emotion + contagion")
    emit("=" * 78)

    acc, rand_acc, fa = _confusion_section(emit)
    emit("-" * 78)
    rec_hits, rec_n = _recognition_section(emit)
    emit("-" * 78)
    d0, d_end, start_emotion, end_emotion = _contagion_section(emit)
    emit("-" * 78)
    # section (4), the ice-cream-van nested false belief, went with `SecondOrderToM` at T082
    # (2.1). Its criterion is not weakened, it MOVED: `faculties/recursive_tom`'s red R5 measures
    # the nested equivalence on 200 constructed witness scripts and the module's own self-check
    # (a) / (b) keep a nested false belief reachable-red on every gate run.

    # (4) DETERMINISM + ASCII
    a1 = AffectCore(own_valence=0.6, own_arousal=0.2, seed=3)
    a2 = AffectCore(own_valence=0.6, own_arousal=0.2, seed=3)
    a1.empathize(-0.5, 0.9, gain=0.4)
    a2.empathize(-0.5, 0.9, gain=0.4)
    det_affect = a1.affect() == a2.affect() and a1.own_emotion() == a2.own_emotion()
    # noise-robust categorical readout: jittered similarities still recover the label
    noisy = AffectCore(seed=5).recognize(*_proto_va(120.0, 0.95), jitter=0.05)
    emit("(4) DETERMINISM + NOISE-ROBUSTNESS + ASCII")
    emit(f"    identical seeds/inputs -> identical affect+emotion : {det_affect}")
    emit(f"    jitter=0.05 on similarities still recognises fear   : {noisy == 'fear'}")
    assert det_affect, "identical inputs produced different outputs -- not deterministic"
    assert noisy == "fear", "spiking categorical readout must survive similarity jitter"

    emit("=" * 78)

    report = "\n".join(lines)
    assert report.isascii(), "self-check produced non-ASCII output"
    print(report)
    print(
        "HONEST NEGATIVE: this is dimensional->categorical MAPPING (fixed Russell-circumplex anchors, "
        "a spiking nearest-prototype readout). It is NOT felt emotion -- nothing here experiences fear, "
        "anger, or empathy; the scenarios are toy and the readouts near-tabular. The fear/anger pair is "
        "the tightest (30 deg) and is the honest hard case."
    )
    print(
        f"self-check OK: {len(AffectCore().emotions)} emotions separable (accuracy {acc:.2f}, "
        f"fear/anger confusions {fa}) vs random-label control {rand_acc:.2f}; recognised {rec_hits}/{rec_n} "
        f"affect-tagged episodes; contagion converged {d0:.2f}->{d_end:.2f} and flipped "
        f"{start_emotion}->{end_emotion}; deterministic, ASCII. SECOND-ORDER ToM RETIRED at T082 -- "
        "faculties/recursive_tom is the one belief tracker"
    )


if __name__ == "__main__":
    main()
