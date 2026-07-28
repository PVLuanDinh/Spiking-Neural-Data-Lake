"""
Dense (modern) associative memory — the capacity spectrum above classical Hopfield.

hopfield_baseline.py established the classical yardstick: recall collapses past ~0.138·N.
Krotov & Hopfield (2016; 2021, arXiv:2601.00984 for the distributed variant) showed a DENSE
associative memory with a higher-order interaction stores far more: capacity grows ~N^(n-1)
for an energy F(x)=x^n, so a small n>2 already blows past the classical linear bound. Survey:
"Modern Methods in Associative Memory" (arXiv:2507.06211); the modern continuous form is the
softmax that connects to Transformer attention.

This module needs no new machinery — the SAME update, parameterised by the interaction degree n,
sweeps the whole spectrum:
  s_i <- sign( Σ_p [ F(a_p + ξ_p,i) - F(a_p - ξ_p,i) ] ),  a_p = ξ_p·s - ξ_p,i·s_i,  F(x)=x^n.
  n=2 reduces EXACTLY to classical Hopfield ( F(a+ξ)-F(a-ξ) = 4·a·ξ ), n=4 is dense.
So one demo places the repo's associative-memory story on a named spectrum:
  classical (n=2, ~0.138·N)  ->  the k-WTA factored prototype (spiking_storage_prototype.py:
  80 patterns @ N=256 in O(P·k) = 874× smaller than dense W)  ->  distributed-dense (n=4 here,
  clean recall well past P=2·N). The prototype's storage win (FACTORING) and this capacity win
  (the DENSE nonlinearity) are orthogonal levers — both beat the classical baseline, differently.

v0.65 — also contrasts the two MODERN (continuous) energies at the dense load: the softmax / log-sum-exp
form (Ramsauer 2020, ↔ Transformer attention) and the log-sum-ReLU / Epanechnikov energy
(arXiv:2506.10801). LSR matches softmax's high capacity but is exp-FREE, so it needs none of the
max-subtraction / clamping the softmax requires to avoid overflow — a real robustness win for the
recall path (naive exp(β·dot) overflows once β·dot > ~709).

  Run: python dense_associative_memory.py       # (also run by run_selfchecks.py)
"""

import math
import random

import snn_data_lake.reasoning.stats_bounds as sb  # research/ is on sys.path when run as a script (B5: mean±std bands)

random.seed(0)  # module-level determinism; per-run seeds below are fixed too

N = 48  # neurons; small enough that the n=4 sweep stays fast in CI
FLIP = round(0.10 * N)  # cue corruption before recall
STEPS = 3  # synchronous update passes (dense AM settles fast)
DEGREES = (2, 4)  # 2 = classical Hopfield; 4 = dense (super-linear capacity)
SWEEP = (0.1, 0.5, 1.0, 2.0)  # P/N sweep points
SEEDS = range(5)  # ≥5 seeds -> mean ± std (roadmap B5)
SAMPLE = 12  # patterns recall-tested per point (speed)
THRESH = 0.95  # mean overlap at/above which recall is "clean"


def make_patterns(p, n, rng):
    return [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(p)]


def dense_recall(patterns, cue, degree, steps):
    """Synchronous higher-order Hopfield update; degree 2 == classical, >2 == dense."""
    s = list(cue)
    n = len(cue)
    for _ in range(steps):
        dots = [sum(pat[j] * s[j] for j in range(n)) for pat in patterns]
        nxt = []
        for i in range(n):
            field = 0
            for p, pat in enumerate(patterns):
                xi = pat[i]
                a = dots[p] - xi * s[i]  # field on i with neuron i removed
                field += (a + xi) ** degree - (a - xi) ** degree
            nxt.append(1 if field >= 0 else -1)
        if nxt == s:
            break
        s = nxt
    return s


def overlap(a, b, n):
    return sum(1 for i in range(n) if a[i] == b[i]) / n * 2.0 - 1.0


def corrupt(pattern, n, flips, rng):
    c = list(pattern)
    for i in rng.sample(range(n), flips):
        c[i] = -c[i]
    return c


def mean_recall(p, degree, n, seed):
    """Store p patterns, recall a sampled subset from corrupted cues, mean overlap."""
    rng = random.Random(seed)
    patterns = make_patterns(p, n, rng)
    tested = patterns if p <= SAMPLE else rng.sample(patterns, SAMPLE)
    ov = [
        overlap(dense_recall(patterns, corrupt(x, n, FLIP, rng), degree, STEPS), x, n)
        for x in tested
    ]
    return sum(ov) / len(ov)


# ---- modern (continuous) energies: retrieval = Σ_p ξ_p · w_p, w_p a kernel over dot_p = ξ_p·s ----
def softmax_weights(dots, beta, guard=True):
    """Modern-Hopfield softmax (log-sum-exp) weights. Needs the max-subtraction guard, else
    exp(β·dot) overflows once β·dot > ~709 (raises OverflowError in pure Python)."""
    z = [beta * d for d in dots]
    m = max(z) if guard else 0.0
    e = [math.exp(zi - m) for zi in z]  # guard=False -> may raise OverflowError
    tot = sum(e) or 1.0
    return [ei / tot for ei in e]


def lsr_weights(dots, beta, window=1.0):
    """Log-sum-ReLU / Epanechnikov weights (arXiv:2506.10801): a bounded ReLU top-band, NO exp.
    Concentrates weight on patterns within `window` of the best match — softmax-like retrieval,
    overflow-free (no max-subtraction needed)."""
    z = [beta * d for d in dots]
    mz = max(z)
    r = [max(0.0, zi - (mz - window)) for zi in z]  # bounded; exp-free
    tot = sum(r) or 1.0
    return [ri / tot for ri in r]


def modern_recall(patterns, cue, weights_fn, steps):
    """Synchronous modern-Hopfield retrieval with a pluggable weight kernel; sign for bipolar."""
    s = list(cue)
    n = len(cue)
    for _ in range(steps):
        dots = [sum(pat[j] * s[j] for j in range(n)) for pat in patterns]
        w = weights_fn(dots)
        nxt = [
            1 if sum(patterns[p][j] * w[p] for p in range(len(patterns))) >= 0 else -1
            for j in range(n)
        ]
        if nxt == s:
            break
        s = nxt
    return s


def mean_recall_modern(patterns, weights_fn, n, rng):
    tested = patterns if len(patterns) <= SAMPLE else rng.sample(patterns, SAMPLE)
    ov = [
        overlap(modern_recall(patterns, corrupt(x, n, FLIP, rng), weights_fn, STEPS), x, n)
        for x in tested
    ]
    return sum(ov) / len(ov)


def main():
    # sweep both interaction degrees over the same loads, averaged over seeds
    table = {}
    for deg in DEGREES:
        for f in SWEEP:
            p = round(f * N)
            table[(deg, f)] = [mean_recall(p, deg, N, s) for s in SEEDS]

    # classical (n=2) empirical capacity: largest swept P still recalling cleanly
    classical_cap = max(
        (round(f * N) for f in SWEEP if sum(table[(2, f)]) / len(SEEDS) >= THRESH), default=0
    )
    # the separating comparison at P = 2·N
    dense_hi = table[(4, 2.0)]
    classical_hi = table[(2, 2.0)]
    dm, ds = sb.mean_std(dense_hi)
    cm, cs = sb.mean_std(classical_hi)

    print("=" * 64)
    print("DENSE ASSOCIATIVE MEMORY — capacity above the classical Hopfield bound")
    print("=" * 64)
    print(f"N={N}  cue corruption={FLIP}/{N}  seeds={len(list(SEEDS))}  (mean overlap ± std)")
    print("\n  P/N     P    n=2 (classical)      n=4 (dense)")
    for f in SWEEP:
        m2, s2 = sb.mean_std(table[(2, f)])
        m4, s4 = sb.mean_std(table[(4, f)])
        print(
            f"  {f:>4.1f}  {round(f * N):>4}    {m2:>6.1%} ± {s2:>4.1%}       {m4:>6.1%} ± {s4:>4.1%}"
        )
    print()
    print(
        f"classical (n=2) empirical capacity : ~{classical_cap} patterns (~{classical_cap / N:.2f}·N,"
        f" bound 0.138·N — see hopfield_baseline.py)"
    )
    print(
        f"at P=2·N={round(2 * N)}:  classical {cm:.1%} ± {cs:.1%}   vs   dense {dm:.1%} ± {ds:.1%}"
    )
    print(
        "spectrum: classical 0.138·N  ->  k-WTA factored prototype (80 patterns @ N=256, 874×"
        " smaller storage)  ->  distributed-dense (n=4, clean past 2·N). Factoring vs dense"
        " nonlinearity are orthogonal wins."
    )
    print("=" * 64)

    # ---- modern energies at the dense load (P=2·N): softmax vs log-sum-ReLU (arXiv:2506.10801) ----
    rng = random.Random(0)
    pats = make_patterns(round(2 * N), N, rng)
    sm = mean_recall_modern(pats, lambda d: softmax_weights(d, beta=1.0), N, rng)
    lsr = mean_recall_modern(pats, lambda d: lsr_weights(d, beta=1.0), N, rng)
    big = 20.0  # β·dot up to 20·48=960 -> exp overflows
    try:
        softmax_weights([float(N), 0.0], beta=big, guard=False)
        overflowed = False
    except OverflowError:
        overflowed = True
    sm_big = mean_recall_modern(pats, lambda d: softmax_weights(d, beta=big), N, rng)
    lsr_big = mean_recall_modern(pats, lambda d: lsr_weights(d, beta=big), N, rng)

    print(
        f"modern energies @ P=2·N: softmax {sm:.1%} | log-sum-ReLU {lsr:.1%}  (both dense-capacity)"
    )
    print(
        f"robustness @ beta={big:.0f}: naive exp overflows={overflowed} | guarded-softmax {sm_big:.1%}"
        f" | LSR {lsr_big:.1%} (exp-free, no clamp)"
    )
    print("=" * 64)

    # ---- self-check: one runnable assert-based check; deterministic seeds ----
    assert round(0.03 * N) <= classical_cap <= round(0.30 * N), (
        f"n=2 should behave ~linearly (capacity {classical_cap} not near 0.138·N)"
    )
    assert dm >= THRESH, f"dense (n=4) should still recall cleanly at 2·N: {dm:.2f}"
    assert cm <= 0.75, f"classical (n=2) should be broken down at 2·N: {cm:.2f}"
    # clean statistical separation (mean±3σ bands do not overlap) — the B5 reuse
    d_lo, _ = sb.bound(dense_hi, k=3.0)
    _, c_hi = sb.bound(classical_hi, k=3.0)
    assert d_lo > c_hi, f"dense/classical capacity bands overlap at 2·N ({d_lo:.2f} !> {c_hi:.2f})"
    assert sm >= THRESH and lsr >= THRESH, (
        f"modern energies not dense-capacity: sm={sm:.2f} lsr={lsr:.2f}"
    )
    assert overflowed, (
        "naive exp(beta*dot) should overflow at large beta (this is why LSR/guard matter)"
    )
    assert lsr_big >= THRESH, (
        f"log-sum-ReLU must stay robust (exp-free) at large beta: {lsr_big:.2f}"
    )
    print(
        "self-check OK: n=2 == classical; n=4 + modern softmax/LSR store past 2·N; "
        "LSR matches softmax capacity exp-free (robust where naive exp overflows)"
    )


if __name__ == "__main__":
    main()
