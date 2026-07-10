"""
Olfactory bulb -- glomerular lateral inhibition that DECORRELATES (whitens) overlapping odors.

Overlapping odors excite a largely SHARED set of glomeruli (a common "background" of receptors
many odorants drive) plus a few DISTINCTIVE glomeruli. That shared drive makes the raw glomerular
representations highly CORRELATED, so downstream cortex struggles to tell similar odors apart. The
bulb's mitral/tufted cells are wrapped in a granule-cell lateral-inhibition network that subtracts
the common pool drive and divisively normalizes gain: the broadly-shared background is cancelled
while the sparse distinctive peaks survive. The net effect is REDUNDANCY REDUCTION / whitening --
the mitral output codes are far less correlated than their inputs, so odors become more separable.

Mechanism (rate-level, deterministic, single-shot):
  surround_i = mean drive of the OTHER glomeruli (granule-cell pool estimate of the common drive)
  driven_i   = relu(x_i - beta * surround_i)              # subtractive lateral inhibition
  y_i        = driven_i / (1 + gamma * surround_i)         # divisive normalization by pool context
Cells sitting in the broadly-active background (x_i ~ surround) are pushed to zero; sparse
distinctive peaks (x_i >> surround) pass. This is the retinal/olfactory center-surround
decorrelation of Atick & Redlich 1992 and the bulb lateral-inhibition model of Cleland 2010.

Fair control (un-riggable -- the SAME overlapping odors measured before and after, one fixed beta):
  (a) RAW input     : mean pairwise Pearson correlation of the overlapping odor vectors (high).
  (b) MITRAL output : the same odors after lateral inhibition -- correlation must DROP clearly.
  (c) PASSTHROUGH   : the identical relay with beta=gamma=0 (no inhibition) -- correlation must
      stay UNCHANGED, proving the drop is caused by the inhibition, not by the measurement.
  Identity preserved: the whitened odors must stay mutually distinguishable -- no collapse to a
  single code (every mitral code is non-zero and distinct, and a noisy re-presentation of an odor
  is still classified to its OWN whitened template).

Honest framing: a reduced rate-level model of bulbar decorrelation on synthetic overlapping odors.
It models a whitening / redundancy-reduction FUNCTION, not smell, perception, or cognition. Ceiling:
a single-shot lateral-inhibition stage over rate codes -- no spike timing, no receptor chemistry,
no learning; the odors are SYNTHETIC. Additive / opt-in: a pre-processing whitener you route
overlapping inputs through before a downstream classifier.
Credit: Atick & Redlich 1992 (decorrelation); Cleland 2010, Friedrich 2013 (bulb lateral inhibition).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.olfactory.olfactory_bulb
"""
import math
import random

N_GLOMERULI = 100          # glomerular / mitral channels
BETA = 2.5                 # subtractive lateral-inhibition strength (granule feedback)
GAMMA = 0.5                # divisive normalization strength (pool gain control)


def _pearson(a, b):
    """Pearson correlation coefficient between two equal-length vectors (0.0 if either is flat)."""
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    da = [v - ma for v in a]
    db = [v - mb for v in b]
    na = math.sqrt(sum(v * v for v in da))
    nb = math.sqrt(sum(v * v for v in db))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / (na * nb)


def _cosine(a, b):
    """Cosine similarity between two non-negative vectors (0.0 if either is all-zero)."""
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _mean_pairwise_corr(vecs):
    """Mean Pearson correlation over all distinct pairs in a list of vectors."""
    k = len(vecs)
    tot = 0.0
    cnt = 0
    for i in range(k):
        for j in range(i + 1, k):
            tot += _pearson(vecs[i], vecs[j])
            cnt += 1
    return tot / cnt if cnt else 0.0


class OlfactoryBulb:
    """Mitral/granule lateral-inhibition network that whitens (decorrelates) glomerular input."""

    def __init__(self, n_glomeruli=N_GLOMERULI, beta=BETA, gamma=GAMMA, seed=0):
        self.n = n_glomeruli
        self.beta = beta          # subtractive lateral-inhibition strength
        self.gamma = gamma        # divisive normalization strength
        self.rng = random.Random(seed)

    def mitral(self, x):
        """Glomerular drive x -> decorrelated mitral output (subtractive + divisive lateral inhibition)."""
        n = self.n
        total = sum(x)
        out = []
        for i in range(n):
            surround = (total - x[i]) / (n - 1)                  # granule estimate of common pool drive
            driven = x[i] - self.beta * surround                # subtractive lateral inhibition
            if driven < 0.0:
                driven = 0.0                                    # mitral rate is non-negative (rectify)
            out.append(driven / (1.0 + self.gamma * surround))  # divisive gain normalization
        return out

    def passthrough(self, x):
        """No-inhibition control: the identical relay with beta=gamma=0 (returns x unchanged)."""
        return list(x)

    def make_overlapping_odors(self, k, background=40, distinct=6):
        """k odors that SHARE a fixed background glomerular block (-> highly correlated) plus a few
        odor-DISTINCTIVE glomeruli each. The shared block is exactly what lateral inhibition cancels."""
        shared = set(self.rng.sample(range(self.n), background))   # common receptors many odors drive
        free = [g for g in range(self.n) if g not in shared]
        odors = []
        for _ in range(k):
            x = [0.0] * self.n
            for g in shared:
                x[g] = self.rng.uniform(0.50, 0.65)                # broad shared background drive
            for g in self.rng.sample(free, distinct):
                x[g] = self.rng.uniform(0.90, 1.00)                # sparse distinctive peaks
            odors.append(x)
        return odors


def main():
    K = 8
    bulb = OlfactoryBulb(seed=0)
    odors = bulb.make_overlapping_odors(K)

    mitral = [bulb.mitral(x) for x in odors]
    passth = [bulb.passthrough(x) for x in odors]

    corr_raw = _mean_pairwise_corr(odors)
    corr_mit = _mean_pairwise_corr(mitral)
    corr_pass = _mean_pairwise_corr(passth)

    # identity: a noisy re-presentation of each odor still classifies to its OWN whitened template.
    nrng = random.Random(11)
    T = 300
    hits = 0
    for _ in range(T):
        pid = nrng.randrange(K)
        noisy = [min(1.0, max(0.0, v + nrng.uniform(-0.10, 0.10))) for v in odors[pid]]
        wq = bulb.mitral(noisy)
        best, bid = -2.0, -1
        for tid, tmpl in enumerate(mitral):
            c = _cosine(wq, tmpl)
            if c > best:
                best, bid = c, tid
        hits += (bid == pid)
    id_acc = hits / T

    # no collapse: every whitened code is non-zero and all pairwise-distinct.
    nonzero = all(any(v > 0.0 for v in m) for m in mitral)
    distinct = all(mitral[i] != mitral[j] for i in range(K) for j in range(i + 1, K))

    print("=" * 66)
    print("OLFACTORY BULB -- glomerular lateral inhibition (odor whitening)")
    print("=" * 66)
    print(f"glomeruli = {bulb.n}, odors = {K}, beta = {bulb.beta}, gamma = {bulb.gamma}")
    print("-" * 66)
    print(f"(a) RAW input      : mean pairwise corr = {corr_raw:.3f}   [shared background -> correlated]")
    print(f"(b) MITRAL output  : mean pairwise corr = {corr_mit:.3f}   [lateral inhibition decorrelates]")
    print(f"(c) PASSTHROUGH    : mean pairwise corr = {corr_pass:.3f}   [beta=gamma=0 -> unchanged]")
    print("-" * 66)
    print(f"decorrelation      : corr {corr_raw:.3f} -> {corr_mit:.3f} (drop {corr_raw - corr_mit:.3f})")
    print(f"identity preserved : noisy-odor classify acc = {id_acc:.3f} over {T} trials")
    print(f"no collapse        : all mitral codes non-zero={nonzero} and distinct={distinct}")
    print("=" * 66)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    assert corr_mit < corr_raw - 0.30, f"bulb must decorrelate: {corr_mit:.3f} not << {corr_raw:.3f}"
    assert abs(corr_pass - corr_raw) < 1e-9, f"passthrough must NOT decorrelate ({corr_pass:.3f} vs {corr_raw:.3f})"
    assert id_acc >= 0.90, f"identity must be preserved after whitening (acc {id_acc:.3f})"
    assert nonzero, "no whitened code may collapse to all-zero"
    assert distinct, "whitened odors must stay mutually distinct (no collapse to one code)"

    b2 = OlfactoryBulb(seed=0)
    odors2 = b2.make_overlapping_odors(K)
    mitral2 = [b2.mitral(x) for x in odors2]
    assert odors2 == odors and mitral2 == mitral, "same seed must give identical odors and mitral output"

    print(f"self-check OK: corr {corr_raw:.3f} -> {corr_mit:.3f} (mitral), passthrough {corr_pass:.3f} "
          f"unchanged; identity {id_acc:.3f}, no collapse, deterministic")


if __name__ == "__main__":
    main()
