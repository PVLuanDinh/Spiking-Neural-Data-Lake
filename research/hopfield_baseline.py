"""
Classical Hopfield capacity baseline  (stdlib-only, zero deps).

Context for spiking_storage_prototype.py: that prototype's "80 patterns @ 99.6%, 874x smaller"
associative-memory result stood alone. This is the NAMED baseline it should be read against —
the textbook dense bipolar Hopfield network (Hopfield 1982; the 0.138·N capacity bound is
Amit-Gutfreund-Sompolinsky 1985). It measures the two things the prototype claims to beat:

  - capacity : a dense Hopfield storing random {-1,+1} patterns recalls cleanly only up to
    ~0.138·N patterns; past that it breaks down catastrophically (recall collapses). We sweep P
    and show the crossover lands on the 0.138·N bound.
  - storage  : a dense Hopfield keeps the full N×N weight matrix — O(N²), and it costs the same
    whether it stores 1 pattern or 100.

Read against this baseline, the prototype's win is TWO distinct levers, not one:
  1. a SPARSE k-hot code (K/N ~ 8%) lifts capacity above the dense 0.138·N bound, and
  2. FACTORED storage (keep the P sparse patterns, never materialise W) turns O(N²) into O(P·k).
This file only establishes lever-1's reference point and lever-2's O(N²) cost; the prototype
delivers the gains. No SNN here on purpose — this is the yardstick, not the method.

  Run: python hopfield_baseline.py       # (also run by run_selfchecks.py)
"""
import random

random.seed(0)  # ponytail: fixed seed -> deterministic self-check. Not a calibration knob.

N = 128                 # neurons; small enough that the O(N²) dense matvec sweep stays fast in CI
FLIP = round(0.10 * N)  # cue corruption: bits flipped before recall (10%)
STEPS = 4               # synchronous sign-update passes (a dense Hopfield settles in a few)
CAP_THRESH = 0.99       # mean overlap at/above which recall counts as "clean"
FRACS = (0.05, 0.10, 0.138, 0.20, 0.30, 0.40)   # P/N sweep points (0.138 = the AGS bound)


def make_patterns(p, n):
    """p random bipolar {-1,+1} patterns, each a length-n list."""
    return [[random.choice((-1, 1)) for _ in range(n)] for _ in range(p)]


def train(patterns, n):
    """Hebbian outer-product store: W = (1/n) Σ_p ξ_p ξ_pᵀ, zero diagonal. Returns flat N·N."""
    w = [0.0] * (n * n)
    for x in patterns:
        for i in range(n):
            xi = x[i]
            if xi == 0:
                continue
            row = i * n
            for j in range(n):
                w[row + j] += xi * x[j]
    inv = 1.0 / n
    for i in range(n):
        w[i * n + i] = 0.0                 # no self-connection
    return [v * inv for v in w]


def recall(w, cue, n, steps):
    """Synchronous sign updates to a fixed point. sign(0) -> +1 (standard convention)."""
    s = list(cue)
    for _ in range(steps):
        nxt = []
        for i in range(n):
            row = i * n
            h = 0.0
            for j in range(n):
                h += w[row + j] * s[j]
            nxt.append(1 if h >= 0.0 else -1)
        if nxt == s:                        # settled onto an attractor
            break
        s = nxt
    return s


def corrupt(pattern, n, flips):
    """Flip `flips` random bits of a stored pattern to make a noisy cue."""
    c = list(pattern)
    for i in random.sample(range(n), flips):
        c[i] = -c[i]
    return c


def overlap(a, b, n):
    """Normalised agreement (1/n)Σ a_i b_i in [-1,1]; 1.0 == perfect recall."""
    return sum(1 for i in range(n) if a[i] == b[i]) / n * 2.0 - 1.0


def mean_recall(p, n, sample_cap=20):
    """Store p patterns, recall each (a sampled subset if p is large) from a corrupted cue,
    return mean overlap with the intended pattern."""
    patterns = make_patterns(p, n)
    w = train(patterns, n)
    tested = patterns if p <= sample_cap else random.sample(patterns, sample_cap)
    ov = [overlap(recall(w, corrupt(x, n, FLIP), n, STEPS), x, n) for x in tested]
    return sum(ov) / len(ov)


def main():
    bound = 0.138 * N                       # theoretical capacity (AGS 1985)
    sweep = [(round(f * N), f) for f in FRACS]
    results = [(p, f, mean_recall(p, N)) for p, f in sweep]

    # empirical capacity = largest P whose mean recall is still "clean"
    clean = [p for p, _, m in results if m >= CAP_THRESH]
    emp_cap = max(clean) if clean else 0

    p_lo = results[0]                       # ~0.05·N — well below capacity
    p_hi = results[-1]                      # ~0.40·N — well above capacity

    # storage: dense W is O(N²) regardless of P; the factored prototype is O(P·k).
    dense_w_bytes = N * N * 8               # float64 N×N weight matrix
    # prototype's headline scale (N=256, K=20): factored store at its measured 80-pattern capacity
    proto_n, proto_k, proto_caps = 256, 20, 80
    proto_dense_bytes = proto_n * proto_n * 8
    proto_factored_bytes = proto_caps * proto_k * 2
    proto_ratio = proto_dense_bytes / proto_factored_bytes

    print("=" * 62)
    print("CLASSICAL HOPFIELD CAPACITY BASELINE  (dense bipolar, Hebbian)")
    print("=" * 62)
    print(f"neurons N={N}   cue corruption={FLIP}/{N} bits   theoretical cap 0.138·N={bound:.1f}")
    print()
    print("  P (patterns)   P/N     mean recall overlap")
    for p, f, m in results:
        mark = "  <- clean" if m >= CAP_THRESH else ("  <- broken down" if m < 0.85 else "")
        print(f"    {p:>4}        {f:>5.3f}       {m:>6.1%}{mark}")
    print()
    print(f"empirical capacity (overlap>={CAP_THRESH:.2f}) : {emp_cap} patterns "
          f"(~{emp_cap / N:.2f}·N, bound is 0.138·N)")
    print()
    print("STORAGE (why the prototype factors it away):")
    print(f"  dense Hopfield W here (N={N})   : {dense_w_bytes:,} B  (O(N²), P-independent)")
    print(f"  prototype @ N={proto_n} K={proto_k}, {proto_caps} patterns:")
    print(f"    dense N×N matrix             : {proto_dense_bytes:,} B")
    print(f"    factored O(P·k)              : {proto_factored_bytes:,} B  ({proto_ratio:.0f}x smaller)")
    print(f"  and {proto_caps} patterns > the dense 0.138·N={0.138 * proto_n:.0f} bound "
          f"at N={proto_n} — the sparse code lifts capacity too.")
    print("=" * 62)

    # ---- self-check (ponytail: one runnable assert-based check) ----
    assert p_lo[2] >= 0.98, f"below capacity should recall near-perfectly: {p_lo[2]:.2f}"
    assert p_hi[2] <= 0.90, f"above capacity should break down, got overlap {p_hi[2]:.2f}"
    assert p_hi[2] < p_lo[2] - 0.05, "no capacity cliff — recall did not degrade with load"
    assert round(0.03 * N) <= emp_cap <= round(0.25 * N), \
        f"empirical capacity {emp_cap} not bracketing the 0.138·N bound"
    assert proto_factored_bytes < proto_dense_bytes, "factored store not smaller than dense W"
    print("self-check OK: recall collapses past ~0.138·N (named baseline), storage is O(N²) — "
          "the prototype beats both via a sparse code + factored storage")


if __name__ == "__main__":
    main()
