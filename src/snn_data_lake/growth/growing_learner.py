"""
v2.7 — growing spiking learner: novelty-triggered NEUROGENESIS over the fused
multimodal spike-feature space.

The repo already GROWS a gesture vocabulary offline in signal_loop.cluster_unknowns
(batch single-link clustering of recorded unknowns). This is the ONLINE, first-class
version of that idea: an adaptive-resonance-style learner that STARTS SMALL (zero
prototypes) and RECRUITS a new prototype unit the moment it meets a pattern no
existing unit recognizes, while REFINING existing units on familiar ones. So its
capacity literally grows as multimodal data streams in.

Reuses (does NOT reimplement):
  * multimodal_encoder.encode_multimodal -> the [0,1]^N=64 fused spike-feature vector
    every prototype lives in (the self-check builds genuinely multimodal inputs with it).
  * spike_preprocessing.N -> the feature-space dimensionality.

Mechanism (ART-like, cosine match):
  observe(pattern):
    - no units yet, or best cosine similarity < vigilance  -> RECRUIT a new prototype
      initialized to `pattern` (neurogenesis), return (new_id, grew=True).
    - else assign to the nearest unit and REFINE it toward `pattern` by a small learning
      rate (running average / Hebbian), return (that_id, grew=False).
  classify(pattern): nearest existing prototype (recognition only, never grows).

Online prototype recruitment + running-average refinement (ART-like), NOT
deep / gradient learning; cosine is the whole match rule. Deterministic (no RNG in the
learner). The multimodal streams the self-check drives it with are SYNTHETIC. This
GROWS capacity as data arrives; it does not "think".

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.growing_learner
"""

import math

from snn_data_lake.growth.growable import grow_vector
from snn_data_lake.sensorimotor.multimodal_encoder import encode_multimodal
from snn_data_lake.sensorimotor.spike_preprocessing import N


def _cosine(a, b):
    """Cosine similarity of two equal-length vectors; 0 if either has zero norm.
    Inputs are [0,1]^N (non-negative) so this stays in [0,1]."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class GrowingLearner:
    """Online growing prototype learner over N-dim [0,1] feature vectors.

    vigilance: novelty threshold in [0,1]. best similarity below it -> recruit.
    lr:        refinement step; proto <- (1-lr)*proto + lr*pattern on a familiar hit."""

    def __init__(self, vigilance=0.85, lr=0.2, dim=N):
        self.vigilance = vigilance
        self.lr = lr
        self.dim = dim
        self.prototypes = []  # list of [0,1]^dim vectors

    @property
    def n_units(self):
        return len(self.prototypes)

    def _best(self, pattern):
        """(unit_id, similarity) of the nearest prototype, or (-1, -1.0) if none. Fails loud on a
        width mismatch (cosine's zip would otherwise silently truncate) -- so a consumer that forgot
        to grow_dim alongside a widened producer is caught, not masked."""
        if len(pattern) != self.dim:
            raise ValueError(
                f"pattern width {len(pattern)} != learner dim {self.dim}; grow_dim to widen first"
            )
        best_id, best_sim = -1, -1.0
        for i, p in enumerate(self.prototypes):
            s = _cosine(pattern, p)
            if s > best_sim:
                best_id, best_sim = i, s
        return best_id, best_sim

    def observe(self, pattern):
        """Learn from one pattern. Recruit a new unit if novel (grew=True), else refine
        the nearest unit toward it (grew=False). Returns (unit_id, grew)."""
        best_id, best_sim = self._best(pattern)
        if best_id < 0 or best_sim < self.vigilance:  # novel -> neurogenesis
            self.prototypes.append(list(pattern))
            return len(self.prototypes) - 1, True
        proto = self.prototypes[best_id]  # familiar -> refine toward it
        self.prototypes[best_id] = [
            (1.0 - self.lr) * w + self.lr * x for w, x in zip(proto, pattern)
        ]
        return best_id, False

    def classify(self, pattern):
        """Nearest existing prototype id (recognition only; never grows). -1 if empty."""
        return self._best(pattern)[0]

    # ---- the lake record (specs/022 audit s8 rank 2b, 2026-09-02) ---------------------------
    # P4b: what the brain learns goes IN the lake. Until now this learner had NO persistence path
    # anywhere in the tree -- the brain's prototypes of what it has perceived re-grew from zero on
    # every daemon relaunch (the daemon relaunches on every commit), and since rank 2a the
    # neurogenesis grower's categories ARE the arm learner's prototype ids. The record is a plain
    # JSON-able dict so it rides the ONE named-state record (substrate/home.save_state), never a
    # sidecar and never a new tag. WHICH instances persist is the consumer's business: Brain
    # persists the arm's context learner and the vision/audio learners; the affect integrator's
    # SOM and the lifelong consolidator's learner (two more live instances) and the arm's Q-table
    # over these ids do NOT persist yet -- said in Brain._flush_prototypes' docstring too.
    # distinct-from: growth/neurogenesis_engine.py -- restore_growth_state() rebuilds NG-SNN tensors
    #   from an npz blob. distinct-from: faculties/spiking_jepa.py, sensorimotor/rsnn_encoder.py -- load
    #   weight matrices from tagged JSON. distinct-from: engine/wiring_manifest.py -- rebuilds
    #   ports from a dict. distinct-from: faculties/reasoning_curriculum.py -- that module's OWN
    #   state_dict (codebook + ranks). Each tenant serialises its own object under the repo's
    #   state_dict/load_state_dict convention; this one serialises prototypes. All sites read.
    STATE_FORMAT = "growing_learner.v1"

    def state_dict(self):
        """The learner as a JSON-able dict: format tag, vigilance, lr, dim, every prototype."""
        return {
            "fmt": self.STATE_FORMAT,
            "vigilance": float(self.vigilance),
            "lr": float(self.lr),
            "dim": int(self.dim),
            "prototypes": [[float(v) for v in p] for p in self.prototypes],
        }

    def load_state_dict(self, d):
        """Rebuild the prototypes from `state_dict()`. True if restored.

        Refuses rather than half-restores (the persistence contract every lake tenant here keeps):
        a wrong format tag, a prototype of the wrong width, a non-finite value, a vigilance or lr
        outside [0, 1], or a non-empty live learner (a learner that already recruited is never
        overwritten -- ids in use must stay stable) leave this object exactly as it was. Every
        field the record carries is ADOPTED (vigilance, lr, dim, prototypes): a field the restore
        does not read is a claim the lake keeps forever. The record's dim wins for an EMPTY
        learner; whether that width still matches the live producer is the CONSUMER's question --
        `Brain` refuses an arm record of another width at restore, and grows a narrower
        vision/audio learner with grow_dim (or drops a wider one, counted) at the first percept."""
        if not isinstance(d, dict) or d.get("fmt") != self.STATE_FORMAT:
            return False
        if self.prototypes:
            return False
        try:
            dim = int(d["dim"])
            vig, lr = float(d["vigilance"]), float(d["lr"])
            protos = [[float(v) for v in p] for p in d["prototypes"]]
        except (KeyError, TypeError, ValueError):
            return False
        if dim <= 0 or any(len(p) != dim for p in protos):
            return False
        if not (0.0 <= vig <= 1.0 and 0.0 <= lr <= 1.0):
            return False
        if any(not math.isfinite(v) for p in protos for v in p):
            return False
        self.vigilance, self.lr, self.dim = vig, lr, dim
        self.prototypes = protos
        return True

    def grow_dim(self, new_dim):
        """Widen the feature space this learner lives in, function-preservingly (shared growth
        primitive growth.growable.grow_vector). Every existing prototype is zero-extended, so a
        pattern the learner already knew -- whose NEW dimensions are 0, because that modality was
        absent when it was learned -- still recognises BIT-IDENTICALLY (cosine is unchanged when the
        appended dims are 0 on both sides). Only patterns that actually populate the new dimensions
        can differ, which is exactly the added capability. This lets the learner grow together with a
        widening substrate (e.g. a new sensor modality) instead of being pinned to its birth width.
        Monotone: shrinking is refused so a grown memory never forgets by truncation."""
        if new_dim < self.dim:
            raise ValueError(f"growth is monotone: cannot shrink dim {self.dim} -> {new_dim}")
        self.prototypes = [grow_vector(p, new_dim) for p in self.prototypes]
        self.dim = new_dim


def _jitter(vals, amt, rng):
    """Small per-element input noise, clamped to [0,1] (a 'jittered' sensor reading)."""
    return [min(1.0, max(0.0, v + rng.uniform(-amt, amt))) for v in vals]


def _encode(levels, rng=None, amt=0.0):
    """Build a genuinely multimodal [0,1]^N vector from four per-modality levels
    (visual, temperature, audio, distance), optionally jittered. Uses the repo's
    real encode_multimodal so the learner sees the fused spike-feature substrate."""
    v, t, a, d = _jitter(levels, amt, rng) if amt else levels
    return encode_multimodal(visual=[v], temperature=t, audio=[a], distance=d)


def main():
    import random

    # 5 DISTINCT multimodal signatures: each is (visual, temp, audio, distance) levels.
    # Combinations are chosen so encoded bands land on well-separated RBF centers.
    PATTERNS = [
        (0.1, 0.1, 0.1, 0.1),  # all-low
        (0.9, 0.9, 0.9, 0.9),  # all-high
        (0.1, 0.9, 0.1, 0.9),  # alternating A
        (0.9, 0.1, 0.9, 0.1),  # alternating B
        (0.5, 0.5, 0.5, 0.5),  # all-mid
    ]
    NOVEL = (0.7, 0.7, 0.7, 0.7)  # a 6th signature distinct from all five
    JIT = 0.03  # input jitter; distinct patterns must exceed within-jitter

    learner = GrowingLearner(vigilance=0.85, lr=0.2)

    print("=" * 60)
    print("GROWING SPIKING LEARNER (neurogenesis over multimodal spikes)")
    print("=" * 60)

    def run_stream(seed):
        """Interleave K=5 patterns x repeats, each jittered. Returns the assignment
        sequence [(unit_id, grew)] so two runs can be compared for determinism."""
        lrn = GrowingLearner(vigilance=0.85, lr=0.2)
        rng = random.Random(seed)
        seq = []
        for _ in range(6):  # 6 repeats of each pattern
            for lv in PATTERNS:
                seq.append(lrn.observe(_encode(lv, rng, JIT)))
        return lrn, seq

    # 1. GROWTH: streaming 5 distinct patterns (repeated + jittered) grows to EXACTLY 5.
    learner, seq = run_stream(seed=0)
    assert learner.n_units == 5, f"expected 5 units, grew to {learner.n_units}"
    assert sum(1 for _, grew in seq if grew) == 5, "exactly 5 recruitment events expected"

    # 2. RECOGNITION: a fresh jittered copy of each pattern classifies to its own unit.
    rng = random.Random(100)
    hits = 0
    expected = list(range(5))  # first-seen order == unit id
    for want, lv in zip(expected, PATTERNS):
        if learner.classify(_encode(lv, rng, JIT)) == want:
            hits += 1
    acc = hits / len(PATTERNS)
    assert acc >= 0.9, f"recognition accuracy too low: {acc:.2f}"

    # 3. NOVELTY -> GROWTH: a genuinely new 6th pattern recruits a 6th unit.
    uid, grew = learner.observe(_encode(NOVEL))
    assert grew and uid == 5 and learner.n_units == 6, "novel pattern must grow to 6 units"

    # 4. STABILITY: re-observing familiar patterns does NOT grow (stays 6).
    rng = random.Random(200)
    for lv in PATTERNS + [NOVEL]:
        _, g = learner.observe(_encode(lv, rng, JIT))
        assert not g, "a familiar pattern must not recruit a new unit"
    assert learner.n_units == 6, f"units grew during stability check: {learner.n_units}"

    # 5. DETERMINISTIC: same stream (same seed) -> identical prototypes + assignments.
    a_lrn, a_seq = run_stream(seed=7)
    b_lrn, b_seq = run_stream(seed=7)
    assert a_seq == b_seq, "assignment sequence must be identical for the same seed"
    assert a_lrn.prototypes == b_lrn.prototypes, "final prototype set must be identical"

    # 6. WIDTH GROWTH (function-preserving, via the shared growable primitive): widen the feature
    #    space and a KNOWN pattern (new dimensions absent -> zeros) still recognises to the same unit.
    wl, _ = run_stream(seed=0)
    probe = _encode(PATTERNS[2])
    id_before, units_before = wl.classify(probe), wl.n_units
    wl.grow_dim(N + 16)  # 64 -> 80: a new sensor modality appears
    assert wl.dim == N + 16 and all(len(p) == N + 16 for p in wl.prototypes), (
        "every prototype must widen"
    )
    assert wl.n_units == units_before, "widening changes the width, not the unit count"
    assert wl.classify(probe + [0.0] * 16) == id_before, (
        "a known pattern (new modality absent) must still recognise after widening (function-preserving)"
    )

    # 7) THE LAKE RECORD round-trips byte-equal and REFUSES rather than half-restores (rank 2b).
    import json as _json

    rec = _json.loads(_json.dumps(learner.state_dict()))  # through JSON, as the lake carries it
    back = GrowingLearner(vigilance=0.5, lr=0.9)  # deliberately different: the record must win
    assert back.load_state_dict(rec) and back.prototypes == learner.prototypes, "round trip"
    assert (back.vigilance, back.lr) == (learner.vigilance, learner.lr), "every field adopted"
    assert all(back.classify(_encode(lv)) == learner.classify(_encode(lv)) for lv in PATTERNS), (
        "a restored learner must classify every known pattern to the same id"
    )
    assert not back.load_state_dict(rec), "a learner that already recruited is never overwritten"
    for bad in (
        dict(rec, fmt="nope"),
        dict(rec, prototypes=rec["prototypes"] + [[0.5] * (N - 1)]),
        dict(rec, prototypes=[[float("inf")] * N]),
        dict(rec, vigilance=1.5),
        "not a dict",
    ):
        fresh = GrowingLearner(vigilance=0.85, lr=0.2)
        assert not fresh.load_state_dict(bad) and fresh.n_units == 0, "refusal must leave it empty"

    print(f"vigilance={learner.vigilance}  lr={learner.lr}  dim={N}  jitter={JIT}")
    print("1 GROWTH      : 5 distinct patterns -> n_units=5")
    print(f"2 RECOGNITION : accuracy {acc:.2f} ({hits}/{len(PATTERNS)})")
    print("3 NOVELTY     : 6th pattern grew=True -> n_units=6")
    print("4 STABILITY   : familiar patterns did not grow -> n_units=6")
    print("5 DETERMINISM : identical prototypes + assignments across two runs")
    print(
        f"6 WIDTH GROW  : dim {N} -> {N + 16} function-preserving (known pattern still recognised)"
    )
    print("7 LAKE RECORD : state_dict round-trips through JSON; 5 bad records refused whole")
    print("=" * 60)
    print(
        f"self-check OK: grew to {learner.n_units} units by neurogenesis, "
        f"recognition acc {acc:.2f}, deterministic, feature space grows without forgetting"
    )


if __name__ == "__main__":
    main()
