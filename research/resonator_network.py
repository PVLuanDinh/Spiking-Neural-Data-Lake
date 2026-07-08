"""
Resonator network — VSA FACTORIZATION (the decode vsa_reasoning lacks).

vsa_reasoning can bind/bundle/permute/cleanup, but it cannot INVERT a bound product back into its
factors: given c = bind(role, filler) it recovers a filler only when you already hold the role. A
resonator network (Frady, Kent, Olshausen, Sommer 2020; Kleyko; Nat.Mach.Intell. 2024 arXiv:2208.12880)
solves the factorization c = x1 ⊛ x2 ⊛ … by superposition search: each factor estimate is unbound from
c using the OTHER estimates, then "cleaned up" by a similarity-weighted projection onto its codebook,
iterated to a fixed point. This turns the reason tier from "recall a prototype" into "decompose a
composite / answer analogy queries" over the exact bipolar HD vectors the repo already uses.

  Run: python resonator_network.py       # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake import vsa_reasoning as vsa  # core module (PYTHONPATH=src); reuse hv/bind/sim/D

random.seed(0)

M = 12            # codebook size per factor
STEPS = 25        # resonator iterations per restart
RESTARTS = 3      # random restarts (escape spurious fixed points; keep best reconstruction)
TRIALS = 20       # random factorizations to test


def cleanup_project(v, codebook):
    """Resonator cleanup: superposition of codebook vectors weighted by similarity, then sign.
    (Soft projection — NOT argmax — so the estimate can move smoothly toward the right factor.)"""
    sims = [vsa.sim(v, c) for c in codebook]
    acc = [0.0] * vsa.D
    for s, c in zip(sims, codebook):
        for i in range(vsa.D):
            acc[i] += s * c[i]
    return [1 if a >= 0 else -1 for a in acc]


def _reconstruct(idx, books):
    v = books[0][idx[0]]
    for f in range(1, len(books)):
        v = vsa.bind(v, books[f][idx[f]])
    return v


def _resonate(c, books, steps):
    """One resonator run from the current `est` seeds to a fixed point; returns index estimate."""
    est = c[1]
    for _ in range(steps):
        new = []
        for f, bk in enumerate(books):
            unbound = c[0]
            for g, o in enumerate(est):
                if g != f:                                 # unbind the OTHER factors (bind = self-inverse)
                    unbound = vsa.bind(unbound, o)
            new.append(cleanup_project(unbound, bk))
        if new == est:
            break
        est = new
    return [max(range(len(bk)), key=lambda k: vsa.sim(e, bk[k])) for e, bk in zip(est, books)]


def factorize(comp, books, steps=STEPS, restarts=RESTARTS):
    """Recover one index per codebook s.t. bind(chosen...) ≈ comp. Random restarts escape spurious
    fixed points; keep the estimate whose reconstruction best matches the composite."""
    best_idx, best_s = None, -2.0
    for r in range(restarts):
        if r == 0:
            seeds = [vsa.bundle(bk) for bk in books]        # superposition of each book
        else:
            rr = random.Random(r * 7 + 1)
            seeds = [[rr.choice((-1, 1)) for _ in range(vsa.D)] for _ in books]
        idx = _resonate((comp, seeds), books, steps)
        s = vsa.sim(_reconstruct(idx, books), comp)
        if s > best_s:
            best_s, best_idx = s, idx
    return best_idx


def main():
    rng = random.Random(0)

    def run(n_factors):
        books = [[vsa.hv(f"F{f}_{i}") for i in range(M)] for f in range(n_factors)]
        ok = 0
        for _ in range(TRIALS):
            true = [rng.randrange(M) for _ in range(n_factors)]
            c = books[0][true[0]]
            for f in range(1, n_factors):
                c = vsa.bind(c, books[f][true[f]])
            if factorize(c, books) == true:
                ok += 1
        return ok / TRIALS

    acc2 = run(2)
    acc3 = run(3)

    print("=" * 58)
    print("RESONATOR NETWORK — VSA factorization (decompose a bound product)")
    print("=" * 58)
    print(f"D={vsa.D}  codebook/factor={M}  steps={STEPS}  trials={TRIALS}")
    print(f"2-factor recovery : {acc2:.0%}   (search space {M ** 2})")
    print(f"3-factor recovery : {acc3:.0%}   (search space {M ** 3})")
    print("gives Paradigm C the decode/analogy step bind/bundle/cleanup alone cannot: recover the")
    print("factors of a composite by superposition search over the existing bipolar codebook.")
    print("=" * 58)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert acc2 >= 0.9, f"2-factor resonator should recover reliably: {acc2:.2f}"
    assert acc3 >= 0.6, f"3-factor recovery should beat chance ({1 / M ** 3:.1e}) clearly: {acc3:.2f}"
    print(f"self-check OK: resonator factorizes 2-factor at {acc2:.0%}, 3-factor at {acc3:.0%} "
          f"(vs chance {1 / M ** 3:.1e}) — the unbind vsa_reasoning lacked")


if __name__ == "__main__":
    main()
