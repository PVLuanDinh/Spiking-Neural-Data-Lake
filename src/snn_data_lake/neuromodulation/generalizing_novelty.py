r"""
Generalizing pseudo-count -- a SimHash over the LIF spike raster (SI-4 / ladder L5).

Honest framing (read this first). This is a FROZEN random-hyperplane SimHash of the
BINARY spike vector (the LIF raster): K signed random hyperplanes drawn once from a
seeded random.Random and FROZEN -> a K-bit sign code -> a bucket id (zlib.crc32 of the
code). Each bucket keeps a VISIT COUNT, and the exploration BONUS for a raster is the
classic pseudo-count bonus ~ 1/sqrt(N_bucket) (Bellemare et al. 2016, "Unifying
Count-Based Exploration and Intrinsic Motivation"). Rasters that hash to the SAME bucket
SHARE visitation credit, so SIMILAR spike states generalize -- unlike a tabular per-exact-
key novelty that treats every bit-different raster as brand new.

What this IS and IS NOT (do not overclaim):
  * It is a frozen random PROJECTION over the raster -- a legit neuromorphic primitive
    (the fruit-fly LSH sibling lives in research/fly_hash.py; that one is sparse-expansion
    winner-take-all, this one is a signed sign-code SimHash). It is spiking-COMPATIBLE
    BOOKKEEPING, NOT a learned rate ANN and NOT a spiking learner. No weights are trained;
    the projection never changes after __init__.
  * It is a VISITATION proxy, NOT learning progress. It measures "have I been near this
    spike state?", never "am I getting better at predicting it?". Pair it with the
    learning-progress curiosity kernel (neuromodulation/intrinsic_motivation.py) for the
    latter; this module only supplies the generalizing count.
  * OVERCLAIM TO AVOID: it does NOT "fix curiosity's collapse WHEN THE ENCODER COLLAPSES".
    SimHash runs OVER the encoder's raster, so it INHERITS an encoder collapse -- if the
    upstream encoder maps every state to a near-identical raster, they all hash into ONE
    busy bucket and a genuinely novel state's bonus is WRONGLY suppressed. The __main__
    self-check demonstrates exactly this failure and PRINTS it (honest negative, not
    asserted away). The provable win is NARROWER: it fixes EXACT-KEY brittleness for
    rasters that DO separate states but merely differ bit-for-bit.

Determinism & style invariants (per CLAUDE.md):
  * per-instance random.Random(seed); the projection is frozen at __init__ (no import-time
    or global RNG, no wall-clock).
  * bucket id via zlib.crc32 of the sign-code bytes -- NEVER the builtin hash().
  * pure stdlib (math, random, zlib, substrate._mathutils.dot); ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.generalizing_novelty
"""
import math
import random
import zlib

from snn_data_lake.substrate._mathutils import dot


class GeneralizingNovelty:
    """Frozen random-hyperplane SimHash over a binary spike raster -> generalizing count.

    bucket(raster)      -- read-only crc32 bucket id of the K-bit sign code.
    pseudo_count(raster)-- read-only current visit count of the raster's bucket (0 if unseen).
    bonus(raster)       -- observe (increment the bucket count) THEN return ~1/sqrt(count):
                           1.0 for a bucket's first visit, habituating as the bucket fills.

    n_bits=K sets the granularity: larger K -> finer buckets (less generalization, fewer
    false aliases), smaller K -> coarser buckets (more generalization, more aliasing);
    n_bits=0 collapses everything into a single bucket (used as a control). dim is the raster
    length. Deterministic; the K signed hyperplanes are drawn once from random.Random(seed)
    and never change."""

    def __init__(self, n_bits=16, dim=64, seed=0):
        self.n_bits = n_bits
        self.dim = dim
        rng = random.Random(seed)
        # K signed random hyperplanes through the origin, FROZEN at construction.
        self._planes = tuple(
            tuple(rng.uniform(-1.0, 1.0) for _ in range(dim))
            for _ in range(n_bits)
        )
        self._counts = {}  # bucket id -> visit count

    def _code(self, raster):
        """K-bit sign code: bit j = 1 iff the raster is on the +side of hyperplane j."""
        return bytes(1 if dot(h, raster) >= 0.0 else 0 for h in self._planes)

    def bucket(self, raster):
        """crc32 bucket id of the sign code (deterministic; never builtin hash())."""
        return zlib.crc32(self._code(raster))

    def pseudo_count(self, raster):
        """Read-only visit count of this raster's bucket (does NOT observe)."""
        return self._counts.get(self.bucket(raster), 0)

    def bonus(self, raster):
        """Observe the raster (increment its bucket) and return the ~1/sqrt(N) bonus."""
        b = self.bucket(raster)
        c = self._counts.get(b, 0) + 1
        self._counts[b] = c
        return 1.0 / math.sqrt(c)


# --------------------------------------------------------------------------- #
# self-check helpers (pure stdlib, deterministic)
# --------------------------------------------------------------------------- #
def _ranks(xs):
    """1-based ranks with ties broken by AVERAGE rank (for Spearman)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    """Spearman rank correlation; 0.0 when either side is constant (guarded denominator)."""
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((r - mx) ** 2 for r in rx)
    vy = sum((r - my) ** 2 for r in ry)
    d = math.sqrt(vx * vy)
    return cov / d if d > 0.0 else 0.0


def _flip(rng, raster, nbits):
    """Return a copy of the binary raster with nbits distinct positions flipped."""
    x = list(raster)
    for i in rng.sample(range(len(x)), nbits):
        x[i] ^= 1
    return x


def _distinct_states(rng, nov, count, dim):
    """count random binary rasters whose SimHash buckets are all distinct (held-out set)."""
    states, seen = [], set()
    while len(states) < count:
        s = [rng.randint(0, 1) for _ in range(dim)]
        b = nov.bucket(s)
        if b in seen:
            continue
        seen.add(b)
        states.append(s)
    return states


def main():
    dim = 64          # matches the LIF raster width (cortex.spike_preprocessing.N)
    n_states = 8
    scale = 20        # true revisit counts are multiples of `scale` -> cleanly rank-ordered

    print("=" * 68)
    print("GENERALIZING NOVELTY -- SimHash pseudo-count over the spike raster (SI-4)")
    print("=" * 68)

    # ---- Claim 1: MONOTONE in TRUE revisitation on HELD-OUT states ---------- #
    # Observe NOISY (bit-flipped) versions of each base state; then QUERY the clean
    # base raster (never observed exactly -> held out). A generalizing count credits
    # the bit-different visits to the clean state's bucket; a tabular exact-key count
    # would see every noisy visit as brand new and score every base state at 0.
    nov = GeneralizingNovelty(n_bits=8, dim=dim, seed=1)
    rng = random.Random(1)
    states = _distinct_states(rng, nov, n_states, dim)
    true_counts = [(i + 1) * scale for i in range(n_states)]
    for s, r in zip(states, true_counts):
        for _ in range(r):
            nov.bonus(_flip(rng, s, 2))          # each visit differs by 2 bits
    predicted = [nov.pseudo_count(s) for s in states]  # held-out clean query
    rho = _spearman(predicted, true_counts)
    thr = 0.85
    print(f"(1) MONOTONE: Spearman(bucket count, true revisits) = {rho:.3f} "
          f"(threshold >= {thr})")
    print(f"    held-out clean-raster counts: {predicted}")
    print(f"    true revisit counts:          {true_counts}")

    # ---- Claim 2: CONTROLS that FAIL (collapse the correlation to ~chance) --- #
    # (2a) single-bucket-for-all (n_bits=0): every raster -> one bucket -> every
    #      pseudo_count is identical -> zero variance -> Spearman collapses to 0.
    solo = GeneralizingNovelty(n_bits=0, dim=dim, seed=1)
    for s, r in zip(states, true_counts):
        for _ in range(r):
            solo.bonus(_flip(rng, s, 2))
    solo_pred = [solo.pseudo_count(s) for s in states]
    rho_solo = _spearman(solo_pred, true_counts)
    # (2b) shuffled-bucket assignment: query each state with a RANDOM other state's
    #      bucket count -> the bucket<->state pairing is destroyed -> ~chance.
    shuf_rng = random.Random(7)
    shuf_vals = []
    for _ in range(20):
        perm = predicted[:]
        shuf_rng.shuffle(perm)
        shuf_vals.append(_spearman(perm, true_counts))
    rho_shuf = sum(shuf_vals) / len(shuf_vals)
    print(f"(2) CONTROL single-bucket (n_bits=0): Spearman = {rho_solo:.3f} -> FAILS")
    print(f"    CONTROL shuffled-bucket (mean of 20): Spearman = {rho_shuf:.3f} -> FAILS")

    # ---- Claim 3: FALSE-ALIAS rate across a K-sweep (granularity tradeoff) --- #
    # Fraction of DISTINCT states that land in a bucket shared by another distinct
    # state. Small K -> coarse -> aliases distinct states; large K -> fine -> few
    # collisions. This is the generalization<->resolution knob (printed, not asserted).
    pool_rng = random.Random(3)
    pool = [[pool_rng.randint(0, 1) for _ in range(dim)] for _ in range(40)]
    print("(3) FALSE-ALIAS rate vs K (distinct states colliding into a shared bucket):")
    alias = {}
    for k in (2, 4, 8, 16, 24, 32):
        hn = GeneralizingNovelty(n_bits=k, dim=dim, seed=5)
        buckets = [hn.bucket(s) for s in pool]
        seen = {}
        for b in buckets:
            seen[b] = seen.get(b, 0) + 1
        collided = sum(1 for b in buckets if seen[b] > 1)
        alias[k] = collided / len(pool)
        print(f"      K={k:2d} -> false-alias {alias[k]:.2f}")

    # ---- Claim 4: HONEST NEGATIVE -- SimHash INHERITS an encoder collapse ---- #
    # A deliberately COLLAPSED encoder maps EVERY state (familiar OR novel) to a
    # near-identical raster. All familiar visits pile into one busy bucket; a
    # genuinely-novel state is squashed into that SAME bucket, so its bonus is
    # WRONGLY suppressed. Contrast a healthy SEPARATING encoder that gives the novel
    # state its own fresh bucket (bonus ~1.0). This is the overclaim we refuse to make.
    cn_rng = random.Random(11)
    base = [cn_rng.randint(0, 1) for _ in range(dim)]

    def collapsed_encode(_state):
        return _flip(cn_rng, base, 1)            # ignores the state -> near-identical raster

    collapsed = GeneralizingNovelty(n_bits=16, dim=dim, seed=9)
    for _ in range(30):
        collapsed.bonus(collapsed_encode("familiar"))
    novel_state = [1 - b for b in base]          # the bitwise OPPOSITE -> genuinely novel
    collapsed_bonus = collapsed.bonus(collapsed_encode(novel_state))

    separating = GeneralizingNovelty(n_bits=16, dim=dim, seed=9)
    for _ in range(30):
        separating.bonus(_flip(cn_rng, base, 1))   # familiar rasters (near `base`)
    separating_bonus = separating.bonus(novel_state)  # its OWN raster -> fresh bucket

    print("(4) HONEST NEGATIVE (printed, NOT asserted away) -- encoder-collapse inheritance:")
    print(f"      collapsed encoder: genuinely-novel state bonus = {collapsed_bonus:.3f} "
          f"(WRONGLY suppressed -- squashed into the busy bucket)")
    print(f"      separating encoder: same novel state bonus     = {separating_bonus:.3f} "
          f"(correct -- fresh bucket)")
    print("      => SimHash runs OVER the raster, so it INHERITS the collapse; it fixes")
    print("         exact-key brittleness, NOT a collapsed upstream encoder.")

    print("=" * 68)

    # ---- deterministic asserts (each claim + its failing control) ----------- #
    assert rho >= thr, f"pseudo-count must be monotone in true revisits (Spearman {rho:.3f})"
    assert rho_solo < 0.30, f"single-bucket control must collapse to chance ({rho_solo:.3f})"
    assert abs(rho_shuf) < 0.30, f"shuffled-bucket control must collapse to chance ({rho_shuf:.3f})"
    assert rho > rho_solo + 0.4 and rho > abs(rho_shuf) + 0.4, \
        "the real hash must beat BOTH controls by a wide margin"
    assert alias[2] > alias[32], \
        f"false-alias must shrink as K grows (K=2 {alias[2]:.2f} vs K=32 {alias[32]:.2f})"
    # the honest negative is a REAL failure, so we assert the failure EXISTS (not that it is fixed)
    assert collapsed_bonus < 0.30 < separating_bonus, \
        "collapsed encoder must WRONGLY suppress the novel bonus that a separating encoder keeps"
    # determinism: same seed -> identical frozen projection -> identical bucket
    r0 = [random.Random(99).randint(0, 1) for _ in range(dim)]
    assert GeneralizingNovelty(seed=0).bucket(r0) == GeneralizingNovelty(seed=0).bucket(r0), \
        "same seed -> identical frozen projection -> identical bucket (deterministic)"

    print(f"self-check OK: monotone Spearman {rho:.3f} >= {thr}; controls collapse "
          f"({rho_solo:.3f}, {rho_shuf:.3f}); false-alias K=2 {alias[2]:.2f} > K=32 "
          f"{alias[32]:.2f}; honest-negative collapse printed; deterministic.")


if __name__ == "__main__":
    main()
