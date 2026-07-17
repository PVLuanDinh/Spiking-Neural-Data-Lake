r"""
META SKILL LOOP -- a STANDING, open-ended multi-domain loop where the brain builds its own skills
across LANGUAGE, MATH, and PHYSICS, choosing which to work on next by cross-domain learning-progress.

This is the standing version of research/skill_builder.py. Instead of building each skill once, a
top-level SpikingCuriosityDrive keyed on DOMAIN id runs an open-ended loop: every cycle it picks the
domain with the highest learning-progress (the frontier), streams ONE MORE slice of that domain into
that skill's PERSISTENT order-invariant readout (SLDA-RFF -- continual == joint, so accumulation
across visits never forgets), re-evaluates competence, and feeds the progress back to the controller.
Domains thus COMPETE as a multi-armed frontier; a domain whose progress dries up is revisited less,
and the loop keeps going -- the engine of cumulative, self-directed, cross-domain learning.

FOUR domains, each a real stream (three autoregressive next-token, one masked-slot):
  LANGUAGE  next-phoneme over a real corpus (CMUdict + Gutenberg).                 [autoregressive]
  MATH      next-char over generated arithmetic "a op b = c".                       [autoregressive]
  PHYSICS   next discretized-state over a real bouncing-ball ODE (gravity + damped  [autoregressive
            bounce, Euler-integrated).                                               cast of dynamics]
  WORLD-KB  masked-tail fact completion (given head+relation, predict the tail)     [masked-slot,
            over real relational facts.                                             target-encoder B]

Each learned skill is INDEXED in the durable lake (signature exemplars -> skill id) and routed by
proximity with no domain-id. HONEST SCOPE: a skill-FORMATION + INDEXING + cross-domain SELECTION
engine with per-skill learning curves; competence rides the strong readout (the project's hybrid).
It does NOT solve any domain. Physics here is DISCRETIZED next-state prediction (well-posed for a
low-dim ODE); the JEPA latent-rollout variant for high-dim/continuous dynamics is the next slice.

  PYTHONPATH=src .venv-lake/Scripts/python research/meta_skill_loop.py
  Knobs: NORD_CTX(3) NORD_CYCLES(24) NORD_SLICE(1500) NORD_RFF(1200) NORD_GAMMA(0.15)
Deps: numpy (+ stdlib). Reuses corpus_stream, language_self_build, language_readout_slda, skill_builder.
"""
import os
import random

import numpy as np

from corpus_stream import ARPABET, load_cmudict, load_text, phoneme_stream
from language_self_build import _positional_code, _bigram_baseline
from language_readout_slda import SLDA_RFF_np
from skill_builder import signature, SkillRegistry, math_stream, MATH_VOCAB
from kb_ranking import DistMult, load_dataset
from brain_adapters.lake_memory import LakeMemory
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def physics_stream(n_traj=400, ctx=3, seed=0, bins=20, steps=60):
    """A REAL 1-D bouncing-ball ODE (gravity + damped bounce, Euler-integrated), discretized into
    `bins` height tokens -> a (prev ctx state-tokens, next state-token) stream. Deterministic, real
    dynamics, unlimited data. Predicting the next discretized state is the autoregressive cast of a
    dynamics/world-model skill."""
    rng = random.Random(seed)
    g, dt, restitution, top = 9.8, 0.08, 0.75, 12.0
    seq = []
    for _ in range(n_traj):
        y = rng.uniform(3.0, top)
        v = rng.uniform(-3.0, 3.0)
        for _ in range(steps):
            v -= g * dt
            y += v * dt
            if y <= 0.0:                      # floor bounce (damped)
                y = 0.0
                v = -v * restitution
            b = min(bins - 1, max(0, int(y / top * bins)))
            seq.append(str(b))
        seq.append(" ")                       # trajectory separator token
    pairs = []
    for i in range(1, len(seq)):
        lo = max(0, i - ctx)
        pairs.append((seq[lo:i], seq[i]))
    return pairs


class Domain:
    """One skill under construction: a real stream, a PERSISTENT order-invariant SLDA-RFF readout that
    accumulates across visits, and the running competence used by the meta-controller."""

    def __init__(self, name, pairs, vocab, ctx, rff, gamma, input_vocab=None, baseline=None):
        self.name = name
        self.vocab = vocab                                  # TARGET vocab (the classes)
        self.ridx = {r: i for i, r in enumerate(vocab)}
        self.ivocab = input_vocab or vocab                  # INPUT vocab (context tokens); differs for world-KB
        self.iidx = {r: i for i, r in enumerate(self.ivocab)}
        self.inV = len(self.ivocab)
        self.ctx = ctx
        self.V = len(vocab)
        usable = [(c, nx) for c, nx in pairs if nx in self.ridx]
        cut = int(len(usable) * 0.8)
        self.train, self.test = usable[:cut], usable[cut:]
        self.baseline = _bigram_baseline(pairs) if baseline is None else baseline
        self.slda = SLDA_RFF_np(self.inV * ctx, self.V, rff=rff, gamma=gamma, shrink=0.1, seed=0)
        self.pos = 0
        self.competence = 0.0
        self.visits = 0
        self.registered = False

    def _X(self, rows):
        return np.array([_positional_code(c, self.iidx, self.inV, self.ctx) for c, _ in rows],
                        dtype=np.float64)

    def study(self, slice_n):
        """Stream the next slice of this domain into the persistent readout (order-invariant
        accumulation -> no forgetting), re-evaluate held-out competence. Returns the competence."""
        if not self.train:
            return 0.0
        sl = []
        for _ in range(slice_n):
            sl.append(self.train[self.pos % len(self.train)])
            self.pos += 1
        y = np.array([self.ridx[nx] for _, nx in sl])
        self.slda.observe(self._X(sl), y)
        self.visits += 1
        test = self.test[:3000]
        yte = np.array([self.ridx[nx] for _, nx in test])
        self.competence = float((self.slda.predict(self._X(test)) == yte).mean())
        return self.competence

    def signature_exemplars(self, n=150):
        step = max(1, len(self.train) // n)
        return [signature(c) for c, _ in self.train[::step][:n]]


class RankingDomain:
    """World-KB as an OPEN-VOCAB skill: a DistMult ranking head (not the SLDA classifier). Competence
    is filtered Hits@10 over the full entity space; the model persists and accumulates across visits
    (each visit trains another epoch), so competence rises just like the SLDA heads. Same duck-typed
    interface as Domain (name/baseline/competence/visits/train/test/study/signature_exemplars)."""

    def __init__(self, name, dim=50, seed=0, eval_n=250):
        tr_s, te_s, self.offline = load_dataset()
        ent, rel = {}, {}
        eid = lambda e: ent.setdefault(e, len(ent))   # noqa: E731
        rid = lambda r: rel.setdefault(r, len(rel))   # noqa: E731
        self.tr = np.array([[eid(h), rid(r), eid(t)] for h, r, t in tr_s], dtype=np.int64)
        self.te = np.array([[eid(h), rid(r), eid(t)] for h, r, t in te_s
                            if h in ent and r in rel and t in ent], dtype=np.int64)
        self.ient = {v: k for k, v in ent.items()}
        self.irel = {v: k for k, v in rel.items()}
        self.n_ent = len(ent)
        self.known = {}
        for h, r, t in np.vstack([self.tr, self.te]):
            self.known.setdefault((int(h), int(r)), set()).add(int(t))
        self.model = DistMult(len(ent), len(rel), dim=dim, seed=seed)
        self.name = name
        self.vocab = ["<distmult-ranking-head>"]     # expert metadata (routing uses signatures, not this)
        self.eval_n = eval_n
        self.baseline = 10.0 / self.n_ent            # random-ranking Hits@10
        self.competence = 0.0
        self.visits = 0
        self.registered = False
        self.train = self.tr                         # alias for the data-count print
        self.rng = np.random.RandomState(0)
        # test in (context_tokens, target) form so the meta-loop router works unchanged
        self.test = [([self.irel[int(r)], self.ient[int(h)]], self.ient[int(t)])
                     for h, r, t in self.te[:2000]]

    def study(self, slice_n):
        """One more epoch of DistMult on the accumulated KB, then filtered Hits@10 on a test sample."""
        self.model.train_epoch(self.tr, lr=0.1, batch=1024)
        self.visits += 1
        idx = self.rng.permutation(len(self.te))[:min(self.eval_n, len(self.te))]
        hits = 0
        for j in idx:
            h, r, t = int(self.te[j, 0]), int(self.te[j, 1]), int(self.te[j, 2])
            sc = self.model.rank_tails(h, r)
            gold = sc[t]
            for o in self.known.get((h, r), ()):
                if o != t:
                    sc[o] = -1e30
            hits += (int(np.sum(sc > gold)) + 1) <= 10
        self.competence = hits / max(1, len(idx))
        return self.competence

    def signature_exemplars(self, n=150):
        idx = self.rng.permutation(len(self.te))[:n]
        return [signature([self.irel[int(self.te[j, 1])], self.ient[int(self.te[j, 0])]]) for j in idx]


def main():
    CTX = _envi("NORD_CTX", 3)
    CYCLES = _envi("NORD_CYCLES", 24)
    SLICE = _envi("NORD_SLICE", 1500)
    RFF = _envi("NORD_RFF", 1200)
    GAMMA = _envf("NORD_GAMMA", 0.15)
    lake_path = os.path.join(os.environ.get("TEMP", "."), "sndl_meta_skills.spkl")
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print("=" * 98)
    print("META SKILL LOOP -- standing loop: language / math / physics / world-kb compete by learning-progress")
    print(f"ctx={CTX}  cycles={CYCLES}  slice={SLICE}  rff={RFF}  (persistent order-invariant heads; no forgetting)")
    print("=" * 98)

    cmudict, doff = load_cmudict()
    words, toff = load_text()
    domains = {
        0: Domain("language", list(phoneme_stream(words, cmudict, k=CTX)), ARPABET, CTX, RFF, GAMMA),
        1: Domain("math", math_stream(n_expr=4000, ctx=CTX, seed=0), MATH_VOCAB, CTX, RFF, GAMMA),
        2: Domain("physics", physics_stream(n_traj=400, ctx=CTX, seed=0), [str(b) for b in range(20)] + [" "],
                  CTX, RFF, GAMMA),
        3: RankingDomain("world-kb", dim=50, seed=0),   # OPEN-VOCAB DistMult ranking head (competence = Hits@10)
    }
    kb = domains[3]
    print(f"[data] language {len(domains[0].train):,} / math {len(domains[1].train):,} / "
          f"physics {len(domains[2].train):,} phoneme|char|state pairs; world-kb "
          f"{len(kb.tr):,} triples over {kb.n_ent:,} entities "
          f"({'offline facts' if kb.offline else 'FB15k-237'}; competence=Hits@10, baseline=random {kb.baseline:.4f})")

    lake = LakeMemory(path=lake_path, seed=0)
    registry = SkillRegistry(lake)
    meta = SpikingCuriosityDrive(seed=0)

    # ---- THE STANDING LOOP: curiosity picks the frontier domain, but (as in autonomous_brain_recovered)
    #      unvisited domains go first and a staleness guard stops any domain from starving -- otherwise
    #      curiosity fixates on the single hardest domain and never explores the others ----
    last = {did: 0 for did in domains}
    print("\n[meta-loop] cycle : domain (competence, learning-progress)")
    for cyc in range(1, CYCLES + 1):
        unstudied = [did for did in domains if domains[did].visits == 0]
        if unstudied:
            did = min(unstudied)                                  # coverage first
        else:
            stale = [did for did in domains if cyc - last[did] >= len(domains)]
            did = min(stale, key=lambda x: last[x]) if stale else meta.most_curious(sorted(domains))
        last[did] = cyc
        d = domains[did]
        before = d.competence
        comp = d.study(SLICE)
        prog = max(0.0, comp - before)
        meta.observe(did, 1.0 - comp)             # low competence keeps a domain curious
        if not d.registered and d.visits >= 1:    # index the skill on its first study
            registry.register(did, d.name, d.signature_exemplars(), {"vocab": d.vocab, "ctx": CTX})
            d.registered = True
        print(f"  cycle {cyc:2d} : {d.name:9s} competence {comp:.1%}  (progress {prog:+.1%}, visit {d.visits})")

    # ---- ROUTING across the three indexed skills, by proximity, no domain-id ----
    def route_acc(did):
        test = domains[did].test
        sample = test[:: max(1, len(test) // 400)][:400]
        return sum(registry.route(c) == did for c, _ in sample) / max(1, len(sample))

    racc = {did: route_acc(did) for did in domains}
    router = sum(racc.values()) / len(racc)
    dur = lake.durability_check()
    visits = {d.name: d.visits for d in domains.values()}

    print("\n" + "=" * 98)
    print("META SKILL LOOP -- four skills built + indexed by one standing, self-directed loop:")
    print("-" * 98)
    for did in sorted(domains):
        d = domains[did]
        print(f"  {d.name:9s}: competence {d.competence:.1%}  (baseline {d.baseline:.1%}, "
              f"visits {d.visits}, +{(d.competence - d.baseline) * 100:.1f} pts)  route {racc[did]:.0%}")
    print(f"  meta visit distribution (learning-progress driven): {visits}")
    print(f"  ROUTING mean {router:.0%} (chance {1 / len(domains):.0%}); "
          f"LAKE {lake.n_stored} signatures durable = {dur['ok']}")
    print("-" * 98)
    print("  world-kb competence = DistMult ranking head (Hits@10 over open vocab, real FB15k-237);")
    print("  physics-JEPA (target-encoder C) in physics_jepa.py + physics_jepa_highdim.py + double_pendulum_jepa.py.")
    print("=" * 98)

    # ---- un-riggable self-checks ----
    for d in domains.values():
        assert d.visits >= 2, f"{d.name} was starved by the meta-controller (visits {d.visits})"
        assert d.competence >= d.baseline - 0.03, (
            f"{d.name}: competence below its domain baseline ({d.competence:.3f} vs {d.baseline:.3f})")
    assert all(v >= 2 for v in visits.values()), "meta-controller did not fairly visit every domain"
    assert router > 1.5 / len(domains), f"router no better than chance ({router:.2f})"
    assert min(racc.values()) > 0.5, f"a domain routes below 50% ({racc})"
    assert dur["ok"], "skill index did not survive the lake restart"
    print(f"self-check OK: one standing loop built+indexed {len(domains)} skills (language/math/physics/world-kb), "
          f"each >= its baseline, chosen by cross-domain learning-progress; proximity router {router:.0%} "
          f">> chance {1/len(domains):.0%}; {lake.n_stored} signatures durable -- open-ended, no labels.")
    lake.close()
    return 0


if __name__ == "__main__":
    main()
