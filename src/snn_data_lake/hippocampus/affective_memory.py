"""
v3.77 - affective_memory: AFFECT-TAGGED EPISODIC MEMORY with somatic-marker retrieval
and amygdala-modulated (importance-weighted) replay.

Grounded in Damasio's SOMATIC-MARKER HYPOTHESIS (Descartes' Error, 1994) and the
amygdala's modulation of hippocampal consolidation (McGaugh 2004, "The amygdala
modulates the consolidation of memories of emotionally arousing experiences",
Annu. Rev. Neurosci. 27:1-28): emotionally salient events are (a) preferentially
RETAINED, (b) tagged with a bodily/affective marker (valence, arousal) that can be
used to retrieve "when was I last this afraid?" independently of episode content, and
(c) preferentially REPLAYED during offline consolidation. Emotion does not just colour
memory -- it decides which memories survive and which get rehearsed.

This is the missing prioritization layer over `hippocampus.experience_replay.ReplayBuffer`,
whose ceiling is explicitly "uniform-sampled tabular replay ... no prioritization". That
buffer samples EVERY stored transition with equal probability (`rng.sample`). Real
hippocampal replay is biased by amygdala arousal toward the emotionally significant.
`AffectiveEpisodicStore` supplies that bias: it is a capacity-bounded episodic store that
(1) EVICTS by emotional importance (not recency), (2) retrieves by content OR by affect,
and (3) hands `ReplayBuffer.replay` an importance-weighted sample instead of a uniform one.

  importance(ep) = |valence| + arousal + 0.5*|reward|

WIRING (documented only -- this module edits nothing else):
  - arousal  <- amygdala.fear_conditioning.Amygdala.threat(cs)           (threat readout, 0..1;
                the class name was `FearConditioning` here, which is defined nowhere in the tree --
                corrected 2026-08-04, the class has always been `Amygdala`)
  - valence  <- cortex.self_model.SelfModel valence-axis, or
                plasticity.valence_stdp.ValenceLearner.valence()      (learned good/bad axis)
  - reward   <- the same live reward the arm loop already emits
  At store time the caller passes (fragment, valence, arousal, reward). At consolidation the
  caller does:  eps = store.consolidate(budget, seed);  then converts each Episode to the
  (unit, action, reward) transitions ReplayBuffer expects and calls ReplayBuffer.replay --
  amygdala-modulated replay instead of uniform. No safety surface is touched: like
  experience_replay, this is value/memory consolidation only -- it NEVER selects an action,
  NEVER calls the arm, NEVER emits a motor command.

Determinism: per-instance / per-call random.Random(seed); zlib.crc32 for episode keys
(never builtin hash); ASCII-only output. Additive / opt-in: nothing changes unless you
build a store and call it.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.affective_memory
"""

import collections
import math
import random
import zlib

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "AffectiveEpisodicStore.consolidate": (
        "NOT_A_RULE",
        "amygdala-weighted replay SAMPLING: draws episodes with replacement by importance. It selects "
        "what to replay; the plasticity happens in whatever consumes the sample",
    ),
}

# Immutable episode record. `key` is a zlib.crc32 fingerprint of the raw fragment bytes;
# `fragment_mean` is the pooled content vector used for cosine recall; valence/arousal/reward
# are the affective tags; `tick` is the store-order counter (recency, used only to PROVE that
# eviction is emotion-weighted rather than recency-based).
Episode = collections.namedtuple(
    "Episode", ["key", "fragment_mean", "valence", "arousal", "reward", "tick"]
)


def _importance(ep):
    """Emotional salience of an episode: |valence| + arousal + 0.5*|reward|.

    This is the single quantity that governs BOTH retention (eviction keeps the argmax over
    the store) AND replay probability (consolidate weights samples by it). High-affect,
    high-arousal, strongly-rewarded/punished episodes score highest."""
    return abs(ep.valence) + ep.arousal + 0.5 * abs(ep.reward)


def _cosine(a, b):
    """Pure-python cosine similarity dot/(||a||*||b||); returns 0.0 on any zero/empty vector."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0.0:
        return 0.0
    return dot / denom


class AffectiveEpisodicStore:
    """Capacity-bounded, affect-tagged episodic memory.

    Retention is emotion-weighted (evict argmin importance, keep the salient), retrieval is by
    content (cosine + salience bonus) OR by affect (somatic-marker: nearest valence/arousal),
    and consolidation samples episodes weighted by importance -> amygdala-modulated replay to
    feed the existing uniform ReplayBuffer. Deterministic, safety-inert."""

    def __init__(self, capacity=256):
        self.capacity = max(1, int(capacity))
        self.episodes = []  # list[Episode], length <= capacity
        self._tick = 0  # monotonic store-order counter (recency marker)

    def __len__(self):
        return len(self.episodes)

    # ---- ingestion -------------------------------------------------------------------

    @staticmethod
    def _pool(fragment):
        """Reduce a fragment to a content vector. A flat list[float] passes through; a
        sequence of frames list[list[float]] is mean-pooled column-wise (the 'mean' in
        fragment_mean, matching hippocampal time-averaging of a trajectory fragment)."""
        if fragment and isinstance(fragment[0], (list, tuple)):
            cols = len(fragment[0])
            acc = [0.0] * cols
            for frame in fragment:
                for j in range(cols):
                    acc[j] += float(frame[j])
            n = len(fragment)
            return [v / n for v in acc]
        return [float(x) for x in fragment]

    @staticmethod
    def _fragment_bytes(fragment):
        """Deterministic ASCII byte image of the (flattened) fragment for zlib.crc32 keying.
        Rounded to 6 dp so trivially-close fragments do not fork keys; never builtin hash."""
        if fragment and isinstance(fragment[0], (list, tuple)):
            flat = [float(x) for frame in fragment for x in frame]
        else:
            flat = [float(x) for x in fragment]
        return ",".join(f"{x:.6f}" for x in flat).encode("ascii")

    def store(self, fragment, valence, arousal, reward):
        """Encode one episode. When at capacity, EVICT the least emotionally important episode
        first (argmin importance, keep >= 1) -- emotion-weighted retention, NOT recency -- then
        append the new one. Returns the stored Episode."""
        ep = Episode(
            key=zlib.crc32(self._fragment_bytes(fragment)),
            fragment_mean=self._pool(fragment),
            valence=float(valence),
            arousal=float(arousal),
            reward=float(reward),
            tick=self._tick,
        )
        self._tick += 1
        if len(self.episodes) >= self.capacity:
            self._evict_least_important()
        self.episodes.append(ep)
        return ep

    def _evict_least_important(self):
        """Drop the single least-important episode (keep >= 1). Recency is irrelevant here."""
        if len(self.episodes) <= 0:
            return
        idx = min(range(len(self.episodes)), key=lambda i: _importance(self.episodes[i]))
        self.episodes.pop(idx)

    # ---- retrieval -------------------------------------------------------------------

    def recall_similar(self, cue, k=3):
        """Top-k by content: cosine(cue, fragment_mean) + 0.1*(|valence|+arousal) salience bonus
        (emotionally charged episodes are slightly easier to bring to mind). Returns list[Episode]."""
        scored = []
        for ep in self.episodes:
            score = _cosine(cue, ep.fragment_mean) + 0.1 * (abs(ep.valence) + ep.arousal)
            scored.append((score, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def recall_by_emotion(self, v_target, a_target, k=3):
        """Somatic-marker retrieval: top-k episodes whose (valence, arousal) is CLOSEST to the
        target affect -- 'when was I last this afraid?' -- ranked by -(|dv| + |da|), ignoring
        content entirely. Returns list[Episode]."""
        scored = [
            (-(abs(ep.valence - v_target) + abs(ep.arousal - a_target)), ep) for ep in self.episodes
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    # ---- consolidation ---------------------------------------------------------------

    def consolidate(self, budget, seed):
        """Amygdala-modulated replay sampling: draw `budget` episodes WITH replacement, weighted
        by importance (high-affect episodes over-represented) via a fresh random.Random(seed) --
        deterministic. Falls back to uniform if every episode is affectively neutral. Returns the
        sampled list[Episode] for the caller to feed into ReplayBuffer.replay (as transitions)."""
        if not self.episodes:
            return []
        rng = random.Random(seed)
        weights = [_importance(ep) for ep in self.episodes]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(self.episodes)
        return rng.choices(self.episodes, weights=weights, k=budget)


def main():
    print("=" * 68)
    print("AFFECTIVE EPISODIC MEMORY (Damasio somatic marker + amygdala replay)")
    print("=" * 68)

    # (a) EMOTION-WEIGHTED RETENTION, NOT RECENCY. The HIGH-affect episode is stored FIRST
    #     (it is the OLDEST -> a recency/ring policy would evict it first). Under capacity
    #     pressure it survives while a LOW-affect episode is evicted.
    s = AffectiveEpisodicStore(capacity=2)
    high = s.store(
        [1.0, 0.0, 0.0], valence=0.95, arousal=0.90, reward=1.0
    )  # tick 0, oldest, imp=2.35
    low1 = s.store(
        [0.0, 1.0, 0.0], valence=0.00, arousal=0.05, reward=0.0
    )  # tick 1,        imp=0.05
    low2 = s.store(
        [0.0, 0.0, 1.0], valence=0.00, arousal=0.10, reward=0.0
    )  # tick 2 -> forces an eviction
    keys = [ep.key for ep in s.episodes]
    assert high.tick < low1.tick < low2.tick, (
        "high-affect episode must be the OLDEST (recency bait)"
    )
    assert high.key in keys, "emotion-weighted retention must KEEP the high-affect episode"
    assert low1.key not in keys, (
        "the low-affect episode must be evicted, not the older high-affect one"
    )
    assert low2.key in keys and len(s) == 2, "newest low episode stored; capacity respected"

    # (b) SOMATIC MARKER BEATS CONTENT. e_content's fragment == the cue but it is affectively
    #     neutral; e_fear's content is orthogonal to the cue but its affect matches the fear cue.
    #     recall_by_emotion must return e_fear; recall_similar must return e_content.
    s2 = AffectiveEpisodicStore(capacity=10)
    cue = [1.0, 0.0, 0.0]
    e_content = s2.store(
        [1.0, 0.0, 0.0], valence=0.00, arousal=0.05, reward=0.0
    )  # content twin, neutral
    e_fear = s2.store(
        [0.0, 0.0, 1.0], valence=-0.90, arousal=0.90, reward=-1.0
    )  # content stranger, terrified
    by_emotion = s2.recall_by_emotion(-0.90, 0.90, k=1)
    by_content = s2.recall_similar(cue, k=1)
    assert by_emotion[0].key == e_fear.key, (
        "somatic-marker recall must return the affect match (fear)"
    )
    assert by_content[0].key == e_content.key, (
        "content recall must return the content match, not fear"
    )

    # (c) IMPORTANCE-WEIGHTED SAMPLING OVER-REPRESENTS HIGH IMPORTANCE vs UNIFORM (deterministic).
    #     Same seed for both draws so the comparison is apples-to-apples and un-riggable.
    s3 = AffectiveEpisodicStore(capacity=16)
    hi = s3.store([1.0, 0.0], valence=0.90, arousal=0.90, reward=1.0)  # imp = 2.30
    for i in range(5):
        s3.store([0.0, float(i + 1)], valence=0.0, arousal=0.02, reward=0.0)  # imp = 0.02 each
    budget, seed_c = 1000, 123
    weighted = s3.consolidate(budget, seed=seed_c)
    uniform = random.Random(seed_c).choices(s3.episodes, weights=[1.0] * len(s3), k=budget)
    w_hi = sum(1 for ep in weighted if ep.key == hi.key)
    u_hi = sum(1 for ep in uniform if ep.key == hi.key)
    assert w_hi > u_hi, (
        f"importance sampling must over-represent the salient episode ({w_hi} <= {u_hi})"
    )
    assert w_hi > budget // len(s3), "weighted count must exceed the uniform 1/N expectation"

    # (d) DETERMINISM: same seed -> identical consolidate sample.
    d1 = [ep.key for ep in s3.consolidate(500, seed=7)]
    d2 = [ep.key for ep in s3.consolidate(500, seed=7)]
    assert d1 == d2, "same seed must yield an identical consolidate sample"

    print(
        f"(a) RETENTION : kept high-affect (imp={_importance(high):.2f}, oldest tick={high.tick}); "
        f"evicted low-affect (imp={_importance(low1):.2f}) -> emotion beats recency"
    )
    print(
        "(b) SOMATIC   : recall_by_emotion -> fear episode; recall_similar -> content episode "
        "(marker beats content)"
    )
    print(
        f"(c) REPLAY    : salient episode sampled {w_hi}/{budget} weighted vs {u_hi}/{budget} uniform "
        f"(over-represented)"
    )
    print(f"(d) DETERMINISM: same seed -> identical {len(d1)}-episode sample")
    print("=" * 68)
    print(
        f"self-check OK: affect-weighted retention (keep imp={_importance(high):.2f}, drop "
        f"imp={_importance(low1):.2f}); somatic-marker recall beats content; amygdala-modulated "
        f"replay over-represents salient {w_hi}>{u_hi}; deterministic"
    )


if __name__ == "__main__":
    main()
