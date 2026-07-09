"""
v2.9 - hippocampus: DG pattern-SEPARATION -> CA3 pattern-COMPLETION memory
(stdlib-only, zero deps).

Grounded in Nasrin et al. 2021, "Spiking Neural Networks and Hippocampal Function"
(Cognitive Systems Research; DOI 10.1016/j.cogsys.2021.07.008): the dentate gyrus (DG)
uses SPARSE firing to build near-orthogonal codes for similar inputs (pattern
SEPARATION -> reduced interference), and CA3 uses recurrent attractor dynamics to
COMPLETE a stored trace from a partial / noisy cue.

Reuses (does NOT reimplement): spike_preprocessing.N (feature-space size); the
self-check drives it with multimodal_encoder.encode_multimodal and compares against
growing_learner.GrowingLearner -- the same [0,1]^N substrate both of those use.

CA3 variant: a single-shot MAX-OVERLAP attractor (the cue's DG code falls into the
basin of the stored pattern whose sparse code it overlaps most). Chosen over a Hopfield
settle because it is exactly the nearest-attractor rule, fully deterministic, and needs
no convergence proof -- the simplest thing that completes honestly.

This improves representational SEPARATION and recall ROBUSTNESS -- it is NOT
cognition / thinking / AGI. Ceiling: a single-shot autoassociative attractor (max-overlap
vote over sparse codes) -- no temporal sequence, no episodic ordering, no online
plasticity; the streams the self-check drives it with are SYNTHETIC. Additive / opt-in:
the default GrowingLearner cosine path is unchanged; recall() is a robustness add-on you
route noisy cues through.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.hippocampus
"""
import random

from snn_data_lake.cortex.spike_preprocessing import N


class DentateGyrus:
    """Pattern separation by sparse random EXPANSION + k-WTA (fly-hash / DG).

    m = dim*expansion random projections, each summing a fixed `samples`-sized random
    subset of input features; separate() keeps the top-k = round(m*wta) winners. Two
    similar inputs land on largely DIFFERENT winner sets -> near-orthogonal sparse codes.
    This is the fruit-fly / cerebellar expansion mechanism (Dasgupta, Stevens & Navlakha,
    "A neural algorithm for a fundamental computing problem", Science 2017) and the DG
    sparse separation of Nasrin et al. 2021."""

    def __init__(self, dim=N, expansion=20, wta=0.05, samples=6, seed=0):
        self.dim = dim
        self.m = dim * expansion
        rng = random.Random(seed)                       # all RNG seeded -> deterministic
        self.proj = [tuple(rng.sample(range(dim), samples)) for _ in range(self.m)]
        self.k = max(1, round(self.m * wta))

    def separate(self, x):
        """[0,1]^dim vector -> frozenset of the top-k projection ids (sparse DG code).
        Projection activation = sum of x over its tap set; ties break to the lower id."""
        act = [sum(x[i] for i in taps) for taps in self.proj]
        top = sorted(range(self.m), key=lambda j: (act[j], -j), reverse=True)[:self.k]
        return frozenset(top)


class CA3:
    """Autoassociative pattern COMPLETION as a single-shot max-overlap attractor.

    Each stored item keeps its DG sparse code + its original [0,1]^N vector. complete()
    returns the stored item whose sparse code OVERLAPS the cue code most -- the attractor
    basin the (noisy / partial) cue falls into. A corrupted cue still lands in the right
    basin because DG separation + overlap voting is robust. Deterministic; ties -> lower id."""

    def __init__(self):
        self.codes = []                                 # per item: DG sparse code (frozenset)
        self.origs = []                                 # per item: original [0,1]^N vector

    def imprint(self, code, x):
        self.codes.append(code)
        self.origs.append(list(x))
        return len(self.codes) - 1

    def complete(self, cue):
        """Stored id whose code overlaps `cue` most (-1 if empty). Lower id wins ties."""
        if not self.codes:
            return -1
        return max(range(len(self.codes)), key=lambda i: (len(cue & self.codes[i]), -i))


class Hippocampus:
    """DG separation + CA3 completion as one opt-in memory over [0,1]^N vectors."""

    def __init__(self, dim=N, seed=0):
        self.dg = DentateGyrus(dim=dim, seed=seed)
        self.ca3 = CA3()

    def store(self, x):
        """DG-separate x, imprint the attractor in CA3, return the new item id."""
        return self.ca3.imprint(self.dg.separate(x), x)

    def recall(self, cue):
        """DG-separate the cue, CA3-complete, return the stored id it falls into (-1 empty)."""
        return self.ca3.complete(self.dg.separate(cue))

    def reconstruct(self, cue):
        """Original stored vector of the recalled id ([] if empty)."""
        rid = self.recall(cue)
        return list(self.ca3.origs[rid]) if rid >= 0 else []


def main():
    from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
    from snn_data_lake.hippocampus.growing_learner import GrowingLearner, _cosine

    def enc(lv):
        v, t, a, d = lv
        return encode_multimodal(visual=[v], temperature=t, audio=[a], distance=d)

    def jitter(v, rng, amt):
        return [min(1.0, max(0.0, w + rng.uniform(-amt, amt))) for w in v]

    def mask(v, rng, frac):
        return [0.0 if rng.random() < frac else w for w in v]        # zero ~frac of features

    # 6 well-separated multimodal (visual, temp, audio, distance) signatures.
    PATTERNS = [
        (0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.9, 0.9), (0.1, 0.9, 0.1, 0.9),
        (0.9, 0.1, 0.9, 0.1), (0.5, 0.5, 0.5, 0.5), (0.7, 0.3, 0.5, 0.2),
    ]
    vecs = [enc(lv) for lv in PATTERNS]
    K = len(vecs)

    print("=" * 60)
    print("HIPPOCAMPUS (DG pattern-separation -> CA3 pattern-completion)")
    print("=" * 60)

    hip = Hippocampus(seed=0)
    ids = [hip.store(v) for v in vecs]

    # 1 SEPARATION: two similar inputs -> DG codes far less similar than their cosine.
    srng = random.Random(1)
    x1 = [srng.random() for _ in range(N)]
    x2 = [srng.random() if srng.random() < 0.30 else w for w in x1]  # ~30% features changed
    c1, c2 = hip.dg.separate(x1), hip.dg.separate(x2)
    raw_cos = _cosine(x1, x2)
    overlap_frac = len(c1 & c2) / hip.dg.k
    sep_ratio = raw_cos / max(overlap_frac, 1e-9)
    assert sep_ratio > 1.0, f"DG failed to separate: ratio {sep_ratio:.2f} <= 1"

    # 2 COMPLETION: a 30%-masked (partial) cue recalls the correct stored attractor.
    crng = random.Random(42)
    T = 300
    hits = 0
    for _ in range(T):
        pid = crng.randrange(K)
        if hip.recall(mask(vecs[pid], crng, 0.30)) == ids[pid]:
            hits += 1
    comp_acc = hits / T
    assert comp_acc >= 0.9, f"completion accuracy too low: {comp_acc:.3f}"

    # 3 ADDITIVE ROBUSTNESS: on the SAME noisy cue, hippocampus completion recognizes at
    # least as well as the learner's raw nearest-prototype cosine (opt-in, non-degrading).
    learner = GrowingLearner()
    for v in vecs:
        learner.prototypes.append(list(v))              # 1 prototype per pattern, id-aligned
    nrng = random.Random(7)
    raw_hits = hip_hits = 0
    for _ in range(T):
        pid = nrng.randrange(K)
        cue = jitter(vecs[pid], nrng, 0.12)             # identical cue fed to both paths
        raw_hits += (learner.classify(cue) == pid)
        hip_hits += (hip.recall(cue) == ids[pid])
    raw_acc, hip_acc = raw_hits / T, hip_hits / T
    assert hip_acc >= raw_acc, f"hippocampus worse than raw cosine: {hip_acc:.3f} < {raw_acc:.3f}"

    # 4 DETERMINISM: a fresh same-seed instance gives identical DG codes and recalls.
    h2 = Hippocampus(seed=0)
    for v in vecs:
        h2.store(v)
    assert hip.ca3.codes == h2.ca3.codes, "same seed must give identical DG codes"
    drng = random.Random(9)
    det = True
    for _ in range(50):
        pid = drng.randrange(K)
        cue = jitter(vecs[pid], drng, 0.12)
        if hip.recall(cue) != h2.recall(cue):
            det = False
            break
    assert det, "same seed must give identical recall results"

    print(f"units m={hip.dg.m}  active k={hip.dg.k}  stored={K}  dim={N}")
    print(f"1 SEPARATION  : raw_cos={raw_cos:.3f} code_overlap={overlap_frac:.3f} "
          f"ratio={sep_ratio:.2f} (>1 -> DG decorrelates similar inputs)")
    print(f"2 COMPLETION  : 30%-masked cue recall acc={comp_acc:.3f} over {T} trials")
    print(f"3 ADDITIVE    : noisy-cue acc raw={raw_acc:.3f} hip={hip_acc:.3f} (hip >= raw)")
    print("4 DETERMINISM : same seed -> identical DG codes and recalls")
    print("=" * 60)
    print(f"self-check OK: separation ratio {sep_ratio:.2f}, completion {comp_acc:.3f}, "
          f"additive hip {hip_acc:.3f} >= raw {raw_acc:.3f}, deterministic")


if __name__ == "__main__":
    main()
