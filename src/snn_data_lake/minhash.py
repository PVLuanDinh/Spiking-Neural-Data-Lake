"""
MinHash -- estimate the Jaccard resemblance between two motif-sets from fixed-size signatures (pure stdlib).

The lake stores motifs as SETS of active features. A recurring question is not "how big is one set?" but
"how SIMILAR are two named sets A and B?" -- the Jaccard resemblance J(A,B) = |A intersect B| / |A union B|,
a number in [0,1]. MinHash (Broder, "On the resemblance and containment of documents", Proc. Compression and
Complexity of Sequences, 1997) estimates J(A,B) WITHOUT ever intersecting the sets. Pick K deterministic hash
functions h_1..h_K (fold a per-permutation seed into a hashlib digest -- NEVER the builtin hash(), which is
per-process salted). The MinHash SIGNATURE of a set S is sig_k(S) = min over s in S of h_k(s), one minimum per
permutation -- K numbers no matter how large S is. The key fact: P(sig_k(A) == sig_k(B)) = J(A,B), because the
per-permutation minimum collides exactly when the overall-minimum element lies in the intersection. So the
FRACTION of the K signature slots where A and B agree is an unbiased estimator of J(A,B), with standard error
~ 1/sqrt(K). Two signatures are compared slot-by-slot; the sets themselves never touch.

Contrast with the sibling sketches, made explicit (having read both):
  * spike_cardinality.py (HyperLogLog) estimates the DISTINCT COUNT of ONE stream -- a size F0, not a
    comparison. It answers "how many distinct motifs are in A?" with leading-zero-rank registers. MinHash
    answers a different question -- "how similar are A and B?" -- a resemblance in [0,1] from per-permutation
    minima, not a cardinality. Different question, different sketch.
  * research/fly_hash.py is the fruit-fly olfactory LSH: a sparse random expansion + winner-take-all producing a
    sparse binary TAG, used as an approximate-NEAREST-NEIGHBOUR retrieval INDEX over stored windows ("find me the
    most similar STORED item"). MinHash is not an index and does no retrieval; it estimates a similarity VALUE
    between two GIVEN sets by comparing their min-wise signatures slot-by-slot -- the min-wise-independent-
    permutation estimator of Jaccard, a different algorithm and a different output.

Classic K-permutation MinHash (one hashlib digest per permutation) estimating Jaccard on synthetic
sets -- the set-similarity-sketch level; no bottom-k / one-permutation-MinHash optimization, no LSH banding for
retrieval; not a spiking implementation, not cognition.

  python -m snn_data_lake.minhash        # (also run by run_selfchecks.py)
"""
import hashlib
import math

_MAX64 = (1 << 64) - 1                        # empty-slot sentinel: every real hash is <= this


class MinHash:
    """K-permutation MinHash: signature(S) = [min_s h_k(s) for k], Jaccard estimate = fraction of equal slots."""

    def __init__(self, num_perm=128, seed=0):
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = num_perm
        self.seed = seed
        # one 8-byte blake2b digest per (element, permutation); the seed is folded in once and each permutation
        # index k is appended, so h_k(s) = blake2b(seed || s || k) -- K deterministic, independent hashes.
        self._seed_digest = hashlib.blake2b(self._to_bytes(seed), digest_size=8)
        self._perm = [k.to_bytes(4, "big") for k in range(num_perm)]

    @staticmethod
    def _to_bytes(elem):
        """Deterministic byte encoding of a set element (NOT builtin hash(), which is per-process salted)."""
        if isinstance(elem, bytes):
            return elem
        if isinstance(elem, str):
            return elem.encode("utf-8")
        return repr(elem).encode("utf-8")

    def signature(self, iterable):
        """The K-slot MinHash signature of a set: for each permutation k, the minimum hash over the elements."""
        sig = [_MAX64] * self.num_perm
        for elem in set(iterable):                     # duplicates cannot change a minimum; iterate distinct
            base = self._seed_digest.copy()            # blake2b(seed) precomputed; add element once...
            base.update(self._to_bytes(elem))
            for k in range(self.num_perm):
                h = base.copy()                        # ...then fold in permutation k -> h_k(elem)
                h.update(self._perm[k])
                v = int.from_bytes(h.digest(), "big")
                if v < sig[k]:
                    sig[k] = v
        return tuple(sig)

    @staticmethod
    def similarity(sig_a, sig_b):
        """Jaccard estimate = fraction of the K signature slots where the two signatures agree."""
        if len(sig_a) != len(sig_b):
            raise ValueError("signatures must have the same length (same num_perm)")
        if not sig_a:
            return 1.0
        equal = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return equal / len(sig_a)

    def jaccard(self, set_a, set_b):
        """Convenience: estimate J(A,B) directly from two iterables via their signatures."""
        return self.similarity(self.signature(set_a), self.signature(set_b))


def exact_jaccard(a, b):
    """True Jaccard |A intersect B| / |A union B| (self-check oracle only). Empty vs empty defined as 1.0."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def main():
    num_perm = 128
    mh = MinHash(num_perm=num_perm, seed=0)
    se = 1.0 / math.sqrt(num_perm)             # standard error ~ 1/sqrt(K)
    bound = 3.0 * se                           # accept within 3 standard errors

    # ---- 1. accuracy across the range: slide B over a fixed A to hit known true J(A,B) ----
    base = set(range(600))
    sig_a = mh.signature(base)
    shifts = (32, 106, 200, 323, 491)          # chosen so true J(A,B) ~ 0.9, 0.7, 0.5, 0.3, 0.1
    rows = []
    for shift in shifts:
        other = set(range(shift, shift + 600))
        true_j = exact_jaccard(base, other)
        est = mh.similarity(sig_a, mh.signature(other))
        rows.append((true_j, est, abs(est - true_j)))

    # ---- 2. extremes: identical sets -> ~1.0 ; disjoint sets -> ~0.0 ----
    ident = mh.similarity(sig_a, mh.signature(base))            # same set: every slot must agree
    left = set(range(500))
    right = set(range(1_000_000, 1_000_500))                    # no shared element -> true J = 0
    disj = mh.jaccard(left, right)

    # ---- 3. fixed signature size: K numbers regardless of |set| (a real set would grow with |set|) ----
    big = set(range(50000))
    sig_big = mh.signature(big)
    sig_small = mh.signature(set(range(10)))

    # ---- 4. determinism: same num_perm/seed/set -> identical signature AND identical estimate ----
    m1 = MinHash(num_perm=num_perm, seed=0)
    m2 = MinHash(num_perm=num_perm, seed=0)
    det_set = set(range(1500))
    det_sig = m1.signature(det_set) == m2.signature(det_set)
    det_est = m1.jaccard(det_set, base) == m2.jaccard(det_set, base)

    print("=" * 68)
    print("MINHASH -- Jaccard resemblance J(A,B) from fixed-size min-wise signatures")
    print("=" * 68)
    print(f"num_perm K = {num_perm}; std err ~ 1/sqrt(K) = {se:.4f}; 3/sqrt(K) bound = {bound:.4f}")
    print(f"{'true J(A,B)':>12}  {'estimate':>10}  {'abs err':>9}")
    for true_j, est, err in rows:
        print(f"{true_j:>12.4f}  {est:>10.4f}  {err:>9.4f}")
    print(f"extremes: identical sets est {ident:.4f} (> 0.99); disjoint sets est {disj:.4f} (< 0.02)")
    print(f"fixed signature: |set|=50000 -> len(sig)={len(sig_big)}; |set|=10 -> "
          f"len(sig)={len(sig_small)}; num_perm={num_perm}")
    print(f"determinism: two fresh instances -> identical signature {det_sig} and estimate {det_est}")
    print("=" * 68)

    # ---- asserts: deterministic, zero-dep; genuine estimates, not rigged ----
    for true_j, _est, err in rows:
        assert err < bound, f"J(A,B)={true_j:.4f}: abs err {err:.4f} exceeds 3/sqrt(K) bound {bound:.4f}"
    assert ident > 0.99, f"identical sets must estimate ~1.0 ({ident:.4f})"
    assert disj < 0.02, f"disjoint sets must estimate ~0.0 ({disj:.4f})"
    assert len(sig_big) == num_perm == len(sig_small), \
        f"signature must be exactly {num_perm} slots regardless of |set|"
    assert num_perm < 50000, "signature size must be independent of (and far below) |set|"
    assert det_sig and det_est, "same num_perm/seed/set must give identical signature and estimate"
    print(f"self-check OK: Jaccard within {bound:.4f} across true J in [0.1, 0.9]; identical {ident:.2f} / "
          f"disjoint {disj:.2f}; {num_perm}-slot signature for |set|=50000; deterministic")


if __name__ == "__main__":
    main()
