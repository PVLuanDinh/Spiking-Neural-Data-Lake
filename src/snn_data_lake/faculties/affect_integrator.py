"""
Affect integrator -- fuses the brain's LEARNED neuromodulators into a coupled
(valence, arousal) core affect and SELF-ORGANIZES emergent emotion categories.

GAP this closes:
  faculties/affect_core.py maps emotion onto a FIXED Russell-circumplex lookup, and the
  learned neuromodulators -- amygdala threat, dopaminergic reward, serotonergic mood,
  cortisol load -- were never wired INTO it. So the felt emotion could not move with the
  chemistry the rest of the brain is busy learning. This module is that missing coupling:
  the four live neuromodulator readouts drive a spiking (valence, arousal) core, and the
  experienced affect points GROW their own emotion prototypes over time instead of being
  read off a hand-coded circumplex.

Brain-native mechanism (spiking, gradient-free):
  * AROUSAL  -- a locus-coeruleus / noradrenaline (LC-NE) style pool of LIF neurons
    (growth.lif_neuron.LIFNeuron) with heterogeneous thresholds, driven by SALIENCE
    (threat + cortisol dominate, reward adds). Arousal is the pool's spike-rate readout:
    more salience -> more NE firing -> higher arousal. (Aston-Jones & Cohen LC-NE gain.)
  * VALENCE  -- an OPPONENT LIF integrator: an appetitive pool (dopaminergic reward +
    a SLOW serotonergic mood tone) versus an aversive pool (amygdala threat + cortisol
    load). Signed valence = appetitive_rate - aversive_rate in [-1, 1]. Serotonin enters
    as a slow leaky (EMA) mood tone, so a sustained good mood lifts valence gradually
    while a reward spike lifts it at once.
  * EMOTION CATEGORIES -- unsupervised competitive / SOM clustering of the experienced
    (valence, arousal) points into EMERGENT prototypes via the repo's growing
    neurogenesis learner (growth.growing_learner.GrowingLearner): recruit a new
    prototype when an affect point matches none well enough, else Hebbian-refine the
    nearest (winner-take-all). The 2-D affect point is spread onto a non-negative
    population code (sensorimotor.population_encoding.PopulationEncoder) so the learner's cosine
    match behaves as a locality metric. label(v, a) returns the nearest EMERGENT prototype
    id -- never a hand-coded circumplex sector.

Reuses (does NOT reimplement):
  * growth.lif_neuron.LIFNeuron            -- the spiking arousal + opponent-valence pools
  * sensorimotor.population_encoding.PopulationEncoder -- (v, a) -> non-negative locality code
  * growth.growing_learner.GrowingLearner   -- neurogenesis + Hebbian competitive refine

Self-contained: integrate() takes the four scalar neuromodulator signals directly, so this
runs without the full brain. REPRESENTATION-ONLY: it computes a felt-affect readout; it holds
no actuator route and imports nothing under brainstem/. ASCII output, deterministic
(random.Random(seed) per instance; zlib.crc32 for the persisted-state fingerprint).
(The line above read "Pure standard library" until 2026-08-04; under P7 a stdlib-purity boast is
residue of the abolished zero-dependency rule, not a design property. The pools here are 6-8 units
wide, where plain arithmetic is simply the right tool -- a statement about the width, not a rule.)

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.affect_integrator
"""

import math
import random
import zlib

from snn_data_lake.growth.growing_learner import GrowingLearner
from snn_data_lake.growth.lif_neuron import LIFNeuron
from snn_data_lake.infra._mathutils import clamp01
from snn_data_lake.sensorimotor.population_encoding import PopulationEncoder


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


#: MERGED 2026-08-15 into `infra/_mathutils.clamp01` (8 copies -> 1). Alias keeps the local
#: spelling so no call site changes; the MECHANISM is singular.
_clamp01 = clamp01


class AffectIntegrator:
    """Fuse (threat, reward, mood, cortisol) -> spiking (valence, arousal) core affect, and
    self-organize emergent emotion prototypes over the experienced affect points.

    Parameters
      seed       : per-instance RNG seed (heterogeneous pool thresholds; deterministic).
      n_lc       : size of the LC-NE arousal pool.
      n_val      : size of each opponent (appetitive / aversive) valence pool.
      steps      : timesteps the pools are run per integrate() to read out a rate.
      vigilance  : GrowingLearner novelty threshold -- below it, a new emotion is recruited.
      lr         : GrowingLearner Hebbian refinement rate for a familiar emotion.
      n_field    : Gaussian receptive fields per affect axis (population-code width / axis).
      mood_tau   : serotonergic slow-mood time constant (larger = slower tone).
      drive_gain : current scale mapping a normalized [0,1] drive onto the LIF F-I curve.
    """

    def __init__(
        self,
        seed=0,
        n_lc=8,
        n_val=6,
        steps=40,
        vigilance=0.82,
        lr=0.2,
        n_field=7,
        mood_tau=6.0,
        drive_gain=1.3,
    ):
        self._rng = random.Random(seed)
        self.seed = seed
        self.steps = steps
        self.drive_gain = drive_gain

        # LC-NE arousal pool: heterogeneous thresholds give a smooth, graded rate readout.
        self._lc = [self._mk_neuron() for _ in range(n_lc)]
        # Opponent valence pools: appetitive (+) vs aversive (-).
        self._pos = [self._mk_neuron() for _ in range(n_val)]
        self._neg = [self._mk_neuron() for _ in range(n_val)]

        # Serotonergic slow mood tone (leaky EMA); the ONLY state carried across integrate() calls.
        self._mood_decay = math.exp(-1.0 / mood_tau)
        self._mood_slow = 0.0

        # Emergent emotion categories: population-coded (v, a) -> growing neurogenesis learner.
        self._enc_v = PopulationEncoder(n=n_field, lo=-1.0, hi=1.0, max_spikes=1, overlap=1.0)
        self._enc_a = PopulationEncoder(n=n_field, lo=0.0, hi=1.0, max_spikes=1, overlap=1.0)
        self._som = GrowingLearner(vigilance=vigilance, lr=lr, dim=2 * n_field)

        self._last_v = 0.0
        self._last_a = 0.0

    def _mk_neuron(self):
        """A LIF unit with a small per-unit threshold jitter (pool heterogeneity, seeded)."""
        return LIFNeuron(
            tau_m=10.0, v_threshold=1.0 + self._rng.uniform(-0.15, 0.15), v_reset=0.0, refractory=0
        )

    def _pool_rate(self, pool, drive01):
        """Run a LIF pool at a constant normalized drive for self.steps; return spike rate in [0,1]."""
        current = self.drive_gain * _clamp01(drive01)
        spikes = 0
        for n in pool:
            n.reset_state()
            for _ in range(self.steps):
                spikes += n.step(current)
        return spikes / (len(pool) * self.steps)

    def _code(self, v, a):
        """Population-code an affect point (v in [-1,1], a in [0,1]) to a non-negative locality vector."""
        return self._enc_v.activations(_clamp(v, -1.0, 1.0)) + self._enc_a.activations(_clamp01(a))

    def integrate(self, threat, reward, mood, cortisol):
        """Fuse the four neuromodulator readouts into core affect.

        threat, reward, cortisol : aversive / appetitive / load magnitudes (clamped to [0,1]).
        mood                     : serotonergic tone (clamped to [-1,1]); enters as a SLOW EMA.
        Returns (valence in [-1,1], arousal in [0,1]).
        """
        t = _clamp01(threat)
        r = _clamp01(reward)
        c = _clamp01(cortisol)
        m = _clamp(mood, -1.0, 1.0)

        # Serotonergic slow mood tone.
        self._mood_slow = self._mood_decay * self._mood_slow + (1.0 - self._mood_decay) * m
        ms = self._mood_slow

        # Arousal: LC-NE salience drive (threat + cortisol dominate; reward is also salient).
        salience = (t + c + 0.5 * r) / 2.0
        arousal = self._pool_rate(self._lc, salience)

        # Valence: opponent appetitive vs aversive rate difference.
        appetitive = 0.6 * r + 0.6 * max(0.0, ms)
        aversive = 0.6 * t + 0.6 * c + 0.4 * max(0.0, -ms)
        pos = self._pool_rate(self._pos, appetitive)
        neg = self._pool_rate(self._neg, aversive)
        valence = _clamp(pos - neg, -1.0, 1.0)

        self._last_v, self._last_a = valence, arousal
        return valence, arousal

    def observe(self, v, a):
        """Grow or refine an emotion prototype from an experienced affect point.
        Returns (prototype_id, grew) -- grew=True if a NEW emotion was recruited (neurogenesis)."""
        uid, grew = self._som.observe(self._code(v, a))
        return uid, grew

    def label(self, v=None, a=None):
        """Nearest EMERGENT emotion prototype id for (v, a); with no args, labels the most
        recent integrate() point. Returns -1 if no prototypes have been grown yet."""
        if v is None:
            v = self._last_v
        if a is None:
            a = self._last_a
        return self._som.classify(self._code(v, a))

    def describe(self, emotion_id):
        """Decode an emergent prototype back to its (valence, arousal) centroid in affect space
        (interpretability only -- the identity is the emergent id, not this readout)."""
        if emotion_id < 0 or emotion_id >= self.n_emotions:
            raise IndexError(f"no emotion prototype {emotion_id} (have {self.n_emotions})")
        proto = self._som.prototypes[emotion_id]
        nf = self._enc_v.n
        v = self._decode_axis(self._enc_v, proto[:nf])
        a = self._decode_axis(self._enc_a, proto[nf:])
        return v, a

    @staticmethod
    def _decode_axis(enc, acts):
        """Activation-weighted centroid of an encoder's field centers (population decode)."""
        total = sum(acts)
        if total <= 0.0:
            return 0.0
        return sum(c * w for c, w in zip(enc.centers, acts)) / total

    @property
    def n_emotions(self):
        """Number of emergent emotion prototypes grown so far."""
        return self._som.n_units

    def signature(self):
        """Stable zlib.crc32 fingerprint of the persisted self-organized emotion map (NOT builtin
        hash) -- two identically-seeded, identically-driven integrators must share it."""
        payload = ";".join(",".join(f"{x:.6f}" for x in p) for p in self._som.prototypes)
        return zlib.crc32(payload.encode("ascii"))


def main():
    # ---- (a) NEUROMODULATOR FUSION: threat/cortisol -> negative+aroused; reward -> positive ----
    def core(threat, reward, mood, cortisol):
        ai = AffectIntegrator(seed=0)  # fresh: no carried mood tone
        return ai.integrate(threat, reward, mood, cortisol)

    v_neg, a_neg = core(threat=0.9, reward=0.0, mood=0.0, cortisol=0.9)  # high threat + load
    v_pos, a_pos = core(threat=0.0, reward=0.9, mood=0.0, cortisol=0.0)  # high reward, safe

    # ---- (b) SELF-ORGANIZED EMOTIONS: distinct affect regions grow distinct prototypes ----
    REGIONS = [
        (-0.75, 0.80),  # high-arousal negative  (distress / fear)
        (0.75, 0.80),  # high-arousal positive  (excitement / elation)
        (0.70, 0.15),  # low-arousal positive   (calm / contentment)
        (-0.70, 0.20),  # low-arousal negative   (dejection / sadness)
    ]
    JIT = 0.05

    ai = AffectIntegrator(seed=0)
    rng = random.Random(1)
    for _ in range(8):  # interleave repeats of each region
        for cv, ca in REGIONS:
            ai.observe(
                _clamp(cv + rng.uniform(-JIT, JIT), -1.0, 1.0),
                _clamp01(ca + rng.uniform(-JIT, JIT)),
            )
    grew = ai.n_emotions

    # same-region points share an id; different regions get different ids
    probe_rng = random.Random(99)
    region_ids = []
    same_region_consistent = True
    for cv, ca in REGIONS:
        ids = {
            ai.label(
                _clamp(cv + probe_rng.uniform(-JIT, JIT), -1.0, 1.0),
                _clamp01(ca + probe_rng.uniform(-JIT, JIT)),
            )
            for _ in range(6)
        }
        same_region_consistent &= len(ids) == 1  # one region -> one prototype
        region_ids.append(next(iter(ids)))
    distinct_regions = len(set(region_ids)) == len(REGIONS)

    # ---- (c) DETERMINISM: same seed + same protocol -> identical outputs + emotion map ----
    def build_map(seed):
        m = AffectIntegrator(seed=seed)
        rr = random.Random(1)
        for _ in range(8):
            for cv, ca in REGIONS:
                m.observe(
                    _clamp(cv + rr.uniform(-JIT, JIT), -1.0, 1.0),
                    _clamp01(ca + rr.uniform(-JIT, JIT)),
                )
        return m

    sig_a = build_map(0).signature()
    sig_b = build_map(0).signature()
    core_repeat = core(threat=0.9, reward=0.0, mood=0.0, cortisol=0.9)

    print("=" * 72)
    print("AFFECT INTEGRATOR -- neuromodulators -> spiking (valence, arousal) + emergent emotions")
    print("=" * 72)
    print(f"(a) threat+cortisol : valence {v_neg:+.2f}  arousal {a_neg:.2f}   (negative, aroused)")
    print(f"    reward, safe    : valence {v_pos:+.2f}  arousal {a_pos:.2f}   (positive)")
    print(f"(b) emergent emotions grown from {len(REGIONS)} affect regions : n_emotions={grew}")
    print(
        f"    region -> prototype id : {region_ids}  (same-region consistent={same_region_consistent})"
    )
    for eid in sorted(set(region_ids)):
        dv, da = ai.describe(eid)
        print(f"      emotion {eid}: centroid (v={dv:+.2f}, a={da:.2f})")
    print(f"(c) determinism : signature {sig_a} == {sig_b}")
    print("=" * 72)

    # ---- self-check: deterministic asserts, ASCII only ----
    assert v_neg < -0.2, f"threat+cortisol must give NEGATIVE valence ({v_neg:+.2f})"
    assert v_pos > 0.2, f"reward must give POSITIVE valence ({v_pos:+.2f})"
    assert v_pos - v_neg > 0.5, (
        f"reward vs threat valence must be sign-separated ({v_pos:+.2f} vs {v_neg:+.2f})"
    )
    assert a_neg > 0.4, f"threat+cortisol must give HIGH arousal ({a_neg:.2f})"
    assert a_neg > a_pos, (
        f"threat salience must arouse more than a bare reward ({a_neg:.2f} vs {a_pos:.2f})"
    )
    assert grew >= 2, f"distinct affect regions must GROW multiple emotion prototypes (grew {grew})"
    assert grew == len(REGIONS), (
        f"{len(REGIONS)} well-separated regions should grow {len(REGIONS)} emotions (grew {grew})"
    )
    assert same_region_consistent, (
        "same-region affect points must map to the SAME emergent emotion id"
    )
    assert distinct_regions, (
        f"different affect regions must map to DIFFERENT emotion ids ({region_ids})"
    )
    assert sig_a == sig_b, (
        f"same seed + protocol must yield an identical emotion map ({sig_a} vs {sig_b})"
    )
    assert core_repeat == (v_neg, a_neg), (
        "same seed + inputs must give identical core affect (deterministic)"
    )
    print(
        f"self-check OK: threat->({v_neg:+.2f},{a_neg:.2f}) reward->({v_pos:+.2f},{a_pos:.2f}) sign-separated; "
        f"grew {grew} emergent emotions, same-region consistent + regions distinct; deterministic (crc32 {sig_a})"
    )


if __name__ == "__main__":
    main()
