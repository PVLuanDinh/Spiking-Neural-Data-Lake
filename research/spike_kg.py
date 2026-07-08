"""
Paradigm C -- spiking knowledge-graph embeddings (SpikE), unified runner.

Merged from three modules that shared the KG-embedding scaffold; select one with
--model {transe,rotate,relations} (default: all three, so the self-check still
reproduces every original assert). Each mode is the verbatim original logic:

  transe    <- spike_knowledge_graph.py         (TransE in spike-time: translational
                                                  lattice KG, link prediction + anomaly).
  rotate    <- spike_knowledge_graph_rotate.py  (RotatE spike-phase: cyclic relations,
                                                  RotatE beats a TransE baseline).
  relations <- spike_kg_relations.py            (relation algebra: per-relation link
                                                  prediction + inverse/symmetric/composition).

Each mode re-seeds random(0) and prints/asserts exactly as its origin module did.

Run:  python research/spike_kg.py [--model {transe,rotate,relations,all}]
"""
import argparse
import math
import random


def run_transe():
    import math
    import random

    random.seed(0)

    GRID = [4, 4, 4]  # 3-axis lattice -> entities = product, relations = axes
    D = 20            # neurons per entity population (embedding dim = spike-latency vector)
    N_ENT = GRID[0] * GRID[1] * GRID[2]   # 64 entities
    N_REL = len(GRID)                      # 3 relations (one temporal offset per axis)
    MARGIN = 1.0
    LR = 0.1
    EPOCHS = 250


    # ---- tiny vector ops (stdlib) -----------------------------------------------
    def rand_vec(scale=1.0):
        return [random.uniform(-scale, scale) for _ in range(D)]


    def sub(a, b): return [a[i] - b[i] for i in range(D)]
    def add(a, b): return [a[i] + b[i] for i in range(D)]
    def norm(a):   return math.sqrt(sum(x * x for x in a)) or 1e-9
    def unit(a):
        n = norm(a)
        return [x / n for x in a]


    # ---- build a knowledge graph WITH translational (spike-shift) structure ------
    def build_graph():
        """A lattice knowledge graph: entities are 3-D grid cells, each relation r is the
        step +1 along axis r. Tail = head's neighbour one step along that axis. Translation
        holds exactly (neighbour = head + axis-offset), so TransE-in-spike-time embeds it
        cleanly and link prediction generalises the offset to held-out edges."""
        def to_idx(c):
            i = 0
            for k in range(len(GRID)):
                i = i * GRID[k] + c[k]
            return i
        coords = [(a, b, c) for a in range(GRID[0]) for b in range(GRID[1]) for c in range(GRID[2])]
        triples = []
        for cc in coords:
            for r in range(len(GRID)):
                if cc[r] + 1 < GRID[r]:
                    nb = list(cc); nb[r] += 1
                    triples.append((to_idx(cc), r, to_idx(tuple(nb))))
        random.shuffle(triples)
        cut = int(len(triples) * 0.85)
        return triples[:cut], triples[cut:]


    # ---- learn embeddings from triples only (TransE margin-ranking SGD) ----------
    def clip_ball(a):
        """TransE entity constraint: keep inside the unit ball (project only if outside).
        Unlike forcing onto the unit sphere, this lets the lattice spread out."""
        n = norm(a)
        return a if n <= 1.0 else [x / n for x in a]


    def train(train_triples):
        E = [rand_vec(0.5) for _ in range(N_ENT)]
        R = [rand_vec(0.1) for _ in range(N_REL)]
        for _ in range(EPOCHS):
            random.shuffle(train_triples)
            for h, r, t in train_triples:
                tn = random.randrange(N_ENT)                 # corrupt the tail
                if tn == t:
                    continue
                p = sub(add(E[h], R[r]), E[t])               # h + r − t   (true)
                n = sub(add(E[h], R[r]), E[tn])              # h + r − t'  (corrupt)
                if MARGIN + norm(p) - norm(n) <= 0:
                    continue                                 # already separated
                dp, dn = unit(p), unit(n)
                g = sub(dp, dn)                              # shared grad for h and r
                for i in range(D):
                    E[h][i] -= LR * g[i]
                    R[r][i] -= LR * g[i]
                    E[t][i] -= LR * (-dp[i])
                    E[tn][i] -= LR * (dn[i])
            E = [clip_ball(e) for e in E]                    # TransE: keep entities in unit ball
        return E, R


    # ---- relational inference + anomaly scoring ---------------------------------
    def score(E, R, h, r, t):
        """spike-time mismatch ||s_h + δ_r − s_t|| — low = plausible triple."""
        return norm(sub(add(E[h], R[r]), E[t]))


    def evaluate(E, R, test):
        hits1 = hits3 = 0
        mrr = 0.0
        for h, r, t in test:
            ranked = sorted(range(N_ENT), key=lambda e: score(E, R, h, r, e))
            rank = ranked.index(t) + 1
            hits1 += rank == 1
            hits3 += rank <= 3
            mrr += 1.0 / rank
        n = len(test)
        return hits1 / n, hits3 / n, mrr / n


    def main():
        train_t, test_t = build_graph()
        E, R = train(train_t)
        h1, h3, mrr = evaluate(E, R, test_t)

        # anomaly: true test triples vs random (false) triples
        true_d = sum(score(E, R, h, r, t) for h, r, t in test_t) / len(test_t)
        rnd = [(random.randrange(N_ENT), random.randrange(N_REL), random.randrange(N_ENT))
               for _ in range(len(test_t))]
        false_d = sum(score(E, R, h, r, t) for h, r, t in rnd) / len(rnd)

        print("=" * 60)
        print("PARADIGM C — relational spiking knowledge graph (SpikE)")
        print("=" * 60)
        print(f"entities={N_ENT}  relations={N_REL}  spike-dims D={D}")
        print(f"triples: {len(train_t)} train / {len(test_t)} test\n")
        print("LINK PREDICTION (head, rel, ?) -> rank tails by spike-time translation:")
        print(f"  Hits@1 : {h1:.1%}   (random ~ {1/N_ENT:.1%})")
        print(f"  Hits@3 : {h3:.1%}")
        print(f"  MRR    : {mrr:.3f}")
        print()
        print("ANOMALY SCORING (||s_h + δ_r − s_t||, lower = plausible):")
        print(f"  mean score, true triples  : {true_d:.3f}")
        print(f"  mean score, random triples: {false_d:.3f}  ({false_d/true_d:.1f}x higher)")
        print("=" * 60)

        assert h1 >= 0.5, f"link prediction too weak: Hits@1={h1:.2f}"
        assert false_d > 2 * true_d, "anomaly scoring does not separate true vs false"
        print("self-check OK: relational inference works (Hits@1>=50%, random~1.7%), "
              "anomalies score far higher than true triples")
    main()


def run_rotate():
    import math
    import random

    random.seed(0)
    TWO_PI = 2.0 * math.pi

    D = 4             # phase dims (a ring is low-dim; fewer dims = less fragmentation)
    N_ENT = 40
    N_REL = 5
    MARGIN = 0.5
    LR = 0.05
    EPOCHS = 200


    def wrap(x):
        """fold an angle into (-pi, pi] — the circular difference."""
        return (x + math.pi) % TWO_PI - math.pi


    def sign(x):
        return (x > 0) - (x < 0)


    def build_cyclic():
        """Modular ring KG: tail = (head + shift_r) mod N_ENT. Pure cyclic structure."""
        shifts = random.sample(range(1, N_ENT), N_REL)
        triples = [(h, r, (h + shifts[r]) % N_ENT)
                   for h in range(N_ENT) for r in range(N_REL)]
        random.shuffle(triples)
        cut = int(len(triples) * 0.85)
        return triples[:cut], triples[cut:]


    # ---- RotatE: entities = phases, relations = phase shifts ----------------------------
    # Proper RotatE distance: chord length between unit phasors = 2|sin(δ/2)|, with a smooth
    # gradient sign(δ)·cos(δ/2). LR decay helps the phases settle onto the ring.
    ROT_EPOCHS, ROT_LR, ROT_MARGIN, K_NEG = 600, 0.1, 1.0, 5


    def train_rotate(train_t):
        TH = [[random.uniform(0, TWO_PI) for _ in range(D)] for _ in range(N_ENT)]
        PH = [[random.uniform(-0.3, 0.3) for _ in range(D)] for _ in range(N_REL)]

        def cdist(a):
            return 2.0 * abs(math.sin(wrap(a) / 2.0))

        def sc(h, r, t):
            return sum(cdist(TH[h][d] + PH[r][d] - TH[t][d]) for d in range(D))

        for ep in range(ROT_EPOCHS):
            lr = ROT_LR * (1.0 - 0.7 * ep / ROT_EPOCHS)        # decay
            random.shuffle(train_t)
            for h, r, t in train_t:
                sp = sc(h, r, t)
                for _ in range(K_NEG):                          # several negatives -> sharper
                    tn = random.randrange(N_ENT)
                    if tn == t or ROT_MARGIN + sp - sc(h, r, tn) <= 0:
                        continue
                    for d in range(D):
                        dp = wrap(TH[h][d] + PH[r][d] - TH[t][d])
                        dn = wrap(TH[h][d] + PH[r][d] - TH[tn][d])
                        gp = sign(dp) * math.cos(dp / 2.0)     # smooth chord gradient
                        gn = sign(dn) * math.cos(dn / 2.0)
                        TH[h][d] = (TH[h][d] + lr * (-gp + gn)) % TWO_PI
                        PH[r][d] = (PH[r][d] + lr * (-gp + gn)) % TWO_PI
                        TH[t][d] = (TH[t][d] + lr * gp) % TWO_PI
                        TH[tn][d] = (TH[tn][d] - lr * gn) % TWO_PI
        return TH, PH, sc


    # ---- TransE baseline (real vectors, addition) — for contrast on the SAME cyclic KG --
    def train_transe(train_t):
        def rv(s=1.0): return [random.uniform(-s, s) for _ in range(D)]
        E = [rv() for _ in range(N_ENT)]
        R = [rv(0.1) for _ in range(N_REL)]

        def nrm(a): return math.sqrt(sum(x * x for x in a)) or 1e-9

        def sc(h, r, t): return nrm([E[h][d] + R[r][d] - E[t][d] for d in range(D)])

        for _ in range(EPOCHS):
            random.shuffle(train_t)
            for h, r, t in train_t:
                tn = random.randrange(N_ENT)
                if tn == t or MARGIN + sc(h, r, t) - sc(h, r, tn) <= 0:
                    continue
                p = [E[h][d] + R[r][d] - E[t][d] for d in range(D)]
                n = [E[h][d] + R[r][d] - E[tn][d] for d in range(D)]
                np_, nn_ = nrm(p), nrm(n)
                for d in range(D):
                    gp, gn = p[d] / np_, n[d] / nn_
                    E[h][d] -= LR * (gp - gn)
                    R[r][d] -= LR * (gp - gn)
                    E[t][d] -= LR * (-gp)
                    E[tn][d] -= LR * (gn)
            m = max(nrm(e) for e in E)
            if m > 1:
                E = [[x / m for x in e] for e in E]
        return E, R, sc


    def evaluate(score_fn, test_t):
        h1 = mrr = 0
        for h, r, t in test_t:
            ranked = sorted(range(N_ENT), key=lambda e: score_fn(h, r, e))
            rank = ranked.index(t) + 1
            h1 += rank == 1
            mrr += 1.0 / rank
        return h1 / len(test_t), mrr / len(test_t)


    def main():
        train_t, test_t = build_cyclic()
        _, _, t_sc = train_transe(list(train_t))
        th1, tmrr = evaluate(t_sc, test_t)
        _, _, r_sc = train_rotate(list(train_t))
        rh1, rmrr = evaluate(r_sc, test_t)

        print("=" * 60)
        print("PARADIGM C — CYCLIC relations: RotatE vs TransE (spike phase)")
        print("=" * 60)
        print(f"cyclic KG: {N_ENT} entities, {N_REL} relations (modular ring)")
        print(f"triples: {len(train_t)} train / {len(test_t)} test  (random Hits@1 ~ {1/N_ENT:.1%})\n")
        print(f"TransE (addition)  : Hits@1 {th1:5.1%}   MRR {tmrr:.3f}   <- can't model cyclic")
        print(f"RotatE (rotation)  : Hits@1 {rh1:5.1%}   MRR {rmrr:.3f}   <- phase shift = cyclic")
        print("=" * 60)

        assert rh1 >= 0.8, f"RotatE failed on cyclic KG: Hits@1={rh1:.2f}"
        assert rh1 > th1 + 0.3, "RotatE should clearly beat TransE on cyclic relations"
        print("self-check OK: RotatE solves cyclic relations (Hits@1>=80%) where TransE fails")
    main()


def run_relations():
    import math
    import random

    random.seed(0)
    TWO_PI = 2.0 * math.pi

    N = 24            # entities on a ring (Z_N)
    D = 4
    ROT_EPOCHS, ROT_LR, ROT_MARGIN, K_NEG = 700, 0.12, 1.0, 6

    # relation shifts on Z_N, chosen to exhibit each algebraic pattern
    SHIFT = {
        "F": 5,           # a generic relation
        "F_inv": -5,      # its inverse  -> φ ≈ −φ_F
        "S": N // 2,      # symmetric (shift by N/2 is its own inverse) -> φ ≈ π
        "A": 3,
        "B": 7,
        "C": 10,          # composition A∘B  (3 + 7) -> φ_C ≈ φ_A + φ_B
    }
    RELS = list(SHIFT)
    RIDX = {r: i for i, r in enumerate(RELS)}


    def wrap(x):
        return (x + math.pi) % TWO_PI - math.pi


    def sign(x):
        return (x > 0) - (x < 0)


    def build():
        triples = [(h, RIDX[r], (h + SHIFT[r]) % N) for h in range(N) for r in RELS]
        random.shuffle(triples)
        cut = int(len(triples) * 0.85)
        return triples[:cut], triples[cut:]


    def train(train_t):
        TH = [[random.uniform(0, TWO_PI) for _ in range(D)] for _ in range(N)]
        PH = [[random.uniform(-0.3, 0.3) for _ in range(D)] for _ in range(len(RELS))]

        def cd(a):
            return 2.0 * abs(math.sin(wrap(a) / 2.0))

        def sc(h, r, t):
            return sum(cd(TH[h][d] + PH[r][d] - TH[t][d]) for d in range(D))

        for ep in range(ROT_EPOCHS):
            lr = ROT_LR * (1.0 - 0.7 * ep / ROT_EPOCHS)
            random.shuffle(train_t)
            for h, r, t in train_t:
                sp = sc(h, r, t)
                for _ in range(K_NEG):
                    tn = random.randrange(N)
                    if tn == t or ROT_MARGIN + sp - sc(h, r, tn) <= 0:
                        continue
                    for d in range(D):
                        dp = wrap(TH[h][d] + PH[r][d] - TH[t][d])
                        dn = wrap(TH[h][d] + PH[r][d] - TH[tn][d])
                        gp = sign(dp) * math.cos(dp / 2.0)
                        gn = sign(dn) * math.cos(dn / 2.0)
                        TH[h][d] = (TH[h][d] + lr * (-gp + gn)) % TWO_PI
                        PH[r][d] = (PH[r][d] + lr * (-gp + gn)) % TWO_PI
                        TH[t][d] = (TH[t][d] + lr * gp) % TWO_PI
                        TH[tn][d] = (TH[tn][d] - lr * gn) % TWO_PI
        return TH, PH, sc


    def per_relation_hits1(sc, test_t):
        by_rel = {r: [0, 0] for r in range(len(RELS))}
        for h, r, t in test_t:
            rank1 = min(range(N), key=lambda e: sc(h, r, e)) == t
            by_rel[r][0] += rank1
            by_rel[r][1] += 1
        return {RELS[r]: (c / max(1, n)) for r, (c, n) in by_rel.items()}


    def circ(a, b):
        """mean circular distance between two phase vectors, in [0, π]."""
        return sum(abs(wrap(a[d] - b[d])) for d in range(D)) / D


    def main():
        train_t, test_t = build()
        TH, PH, sc = train(train_t)
        hits = per_relation_hits1(sc, test_t)

        # learned-phase algebra checks
        neg_F = [-PH[RIDX["F"]][d] for d in range(D)]
        inv_err = circ(PH[RIDX["F_inv"]], neg_F)                      # φ_Finv ≈ −φ_F
        # symmetric = self-inverse: 2·φ_S ≡ 0 (so φ_S ∈ {0, π}, per-dim)
        sym_err = sum(abs(wrap(2.0 * PH[RIDX["S"]][d])) for d in range(D)) / D
        comp = [PH[RIDX["A"]][d] + PH[RIDX["B"]][d] for d in range(D)]
        comp_err = circ(PH[RIDX["C"]], comp)                          # φ_C ≈ φ_A + φ_B

        print("=" * 60)
        print("PARADIGM C DEPTH — richer relation types (relation algebra)")
        print("=" * 60)
        print(f"cyclic KG Z_{N}, {len(RELS)} relations, {len(train_t)} train / {len(test_t)} test\n")
        print("Per-relation link prediction (Hits@1):")
        for r in RELS:
            print(f"  {r:6} (shift {SHIFT[r]:+d}) : {hits[r]:5.0%}")
        print()
        print("Learned-phase ALGEBRA (mean circular error, 0 = exact, lower better):")
        print(f"  inverse      φ(F_inv) ≈ −φ(F) : {inv_err:.3f} rad")
        print(f"  symmetric    2·φ(S) ≈ 0       : {sym_err:.3f} rad")
        print(f"  composition  φ(C) ≈ φ(A)+φ(B) : {comp_err:.3f} rad")
        print(f"  (random baseline ≈ {math.pi/2:.2f} rad; symmetric is looser — two basins 0/π)")
        print("=" * 60)

        assert all(h >= 0.8 for h in hits.values()), f"a relation type failed: {hits}"
        # inverse/composition recover tightly; symmetric (self-inverse, 2φ≡0) is looser but
        # still well below the random ~1.57 rad baseline.
        assert inv_err < 0.4 and comp_err < 0.4 and sym_err < 0.7, \
            f"phase algebra not recovered: inv={inv_err:.2f} sym={sym_err:.2f} comp={comp_err:.2f}"
        print("self-check OK: all relation types predicted (Hits@1>=80%) AND the algebra "
              "(inverse/symmetric/composition) recovered in the learned phases")
    main()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Spiking knowledge-graph embeddings (SpikE): TransE / RotatE / relation algebra.")
    ap.add_argument("--model", choices=["transe", "rotate", "relations", "all"], default="all",
                    help="which embedding model to run (default: all three).")
    args = ap.parse_args(argv)
    order = ["transe", "rotate", "relations"] if args.model == "all" else [args.model]
    runners = {"transe": run_transe, "rotate": run_rotate, "relations": run_relations}
    for name in order:
        runners[name]()


if __name__ == "__main__":
    main()
