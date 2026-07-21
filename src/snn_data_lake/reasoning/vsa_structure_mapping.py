"""
Structure-mapping analogy over VSA spike codes -- relations-over-relations (Gentner).

`vsa_reasoning` gives a single-shift Kanerva analogy (bind(a,b) then unbind) and `resonator_network`
factorizes ONE flat composite into its factors. Neither performs analogical STRUCTURE-MAPPING:
Gentner's (1983) claim is that an analogy maps a whole relational SYSTEM as a mutually-constraining
set of object correspondences -- objects match not by surface identity (sun vs nucleus look nothing
alike) but by the configuration of relations they participate in (systematicity). This module adds
that: it binds relational roles into per-object SYSTEM signatures and recovers the one-to-one object
correspondence by resonator-style iterative unbinding + cleanup, choosing the correspondence that
maximises systematicity (relations that share arguments in the base should map to relations that
share arguments in the target).

Substrate reuse (no new algebra invented here):
  - bind / bundle come straight from `snn_data_lake.vsa_reasoning` -- the two-line spike-polarity
    codes and their spike-XOR bind (self-inverse, so bind == unbind) and superposition-WTA bundle.
    These ops are dimension-agnostic, so they run unchanged at this module's `dim`.
  - the object-correspondence recovery is the resonator idea from `snn_data_lake.reasoning.
    resonator_network` (Frady, Kent, Olshausen, Sommer 2020): a soft codebook cleanup (superposition
    weighted by similarity) inside an iterate-to-fixed-point loop with random restarts. `cleanup_project`
    is reused directly when `dim` equals the resonator's native `vsa.D`; at other `dim` a
    dimension-correct copy is used (resonator_network.cleanup_project hardcodes vsa.D in its sim).

Encoding. A "situation" is a set of propositions relation(arg0, arg1, ...). A proposition is a
bundle of role-filler bindings bind(role_i, filler); a situation is the bundle of its propositions.
An object's SYSTEM signature is the bundle, over every argument slot it fills, of bind(relation,
role) -- deliberately dropping the object's own identity and its partners' identities, so the
signature carries pure relational role. Two objects correspond when their signatures are most
coincident AND the mapping preserves shared arguments (the mutual constraint, enforced by iterating
the signature with the partners' currently-mapped images).

Determinism: atomic spike codes are crc32(label)-seeded (no module-level RNG); restarts use a
per-instance random.Random(seed). ASCII output only.

  Run: python -m snn_data_lake.reasoning.vsa_structure_mapping     # asserts, prints one OK line
"""

import random
import sys
import zlib

from snn_data_lake import (
    vsa_reasoning as vsa,  # reuse spike codes + bind/bundle (dimension-agnostic)
)
from snn_data_lake.reasoning import resonator_network as rn  # reuse resonator cleanup at native dim


class StructureMapper:
    """Map a source relational system onto a target system by systematicity-maximising object
    correspondence, recovered with resonator-style iterative unbinding over VSA spike codes."""

    def __init__(self, dim=1024, seed=0, restarts=8, refine_iters=4):
        self.dim = dim
        self.seed = seed
        self.restarts = restarts
        self.refine_iters = refine_iters
        self.rng = random.Random(seed)
        self._cache = {}

    # ---- atomic spike codes (crc32-seeded; deterministic across processes, no global RNG) ----
    def _sym(self, label):
        v = self._cache.get(label)
        if v is None:
            r = random.Random(zlib.crc32(label.encode()))
            v = [r.choice((-1, 1)) for _ in range(self.dim)]
            self._cache[label] = v
        return v

    def _role(self, i):
        return self._sym(f"role#{i}")

    def _sim(self, a, b):
        # centred spike coincidence in [-1,1] at THIS dim (vsa.sim hardcodes vsa.D, so compute here).
        return (2 * vsa.coincidence(a, b) - self.dim) / self.dim

    def _cleanup_project(self, v, codebook):
        """Resonator soft cleanup: superposition of codebook codes weighted by similarity, then sign.
        Reuses resonator_network directly at its native dim; dimension-correct copy otherwise."""
        if self.dim == vsa.D:
            return rn.cleanup_project(v, codebook)
        sims = [self._sim(v, c) for c in codebook]
        acc = [0.0] * self.dim
        for s, c in zip(sims, codebook):
            for i in range(self.dim):
                acc[i] += s * c[i]
        return [1 if a >= 0 else -1 for a in acc]

    # ---- proposition / situation / object encoding ----
    def _prop_hv(self, prop):
        """prop = (relation, arg0, arg1, ...). Bundle of role-filler bindings (reuse bind/bundle)."""
        rel, args = prop[0], prop[1:]
        parts = [vsa.bind(self._sym("rel:" + rel), self._sym("relfill:" + rel))]
        for i, a in enumerate(args):
            parts.append(vsa.bind(self._role(i), self._sym("obj:" + str(a))))
        return vsa.bundle(parts)

    def encode(self, props):
        """Situation -> one spike-code hypervector (bundle of its proposition codes)."""
        return vsa.bundle([self._prop_hv(p) for p in props])

    def _objects(self, props):
        seen = []
        for prop in props:
            for a in prop[1:]:
                if a not in seen:
                    seen.append(a)
        return seen

    def _signature(self, obj, props, partner_map=None):
        """Object SYSTEM signature: bundle over the slots `obj` fills of bind(relation, role).
        Identity-free -> pure relational role. When `partner_map` is given, also bind in the
        partners' currently-mapped images (the mutual constraint that makes the mapping systematic)."""
        parts = []
        for prop in props:
            rel, args = prop[0], list(prop[1:])
            for i, a in enumerate(args):
                if a != obj:
                    continue
                parts.append(vsa.bind(self._sym("rel:" + rel), self._role(i)))
                if partner_map is not None:
                    for j, other in enumerate(args):
                        if j == i:
                            continue
                        img = partner_map.get(other)
                        if img is not None:
                            parts.append(
                                vsa.bind(
                                    vsa.bind(self._sym("rel:" + rel), self._role(j)),
                                    self._sym("obj:" + str(img)),
                                )
                            )
        if not parts:
            return self._sym("empty-sig")
        return vsa.bundle(parts)

    # ---- systematicity score ----
    def score(self, mapping, source, target):
        """Systematicity: fraction of source relations R(a,b,..) whose image R(map a, map b,..) is
        present in the target, measured in the spike domain (exact coincidence over prop codes)."""
        if not source:
            return 0.0
        tgt_codes = [self._prop_hv(p) for p in target]
        preserved = 0
        for prop in source:
            rel, args = prop[0], prop[1:]
            if any(a not in mapping for a in args):
                continue
            image = (rel,) + tuple(mapping[a] for a in args)
            img_code = self._prop_hv(image)
            if any(vsa.coincidence(img_code, tc) == self.dim for tc in tgt_codes):
                preserved += 1
        return preserved / len(source)

    # ---- correspondence recovery ----
    def _greedy(self, pref, base_objs, tgt_objs):
        """One-to-one assignment from a preference matrix, highest-preference pair first."""
        pairs = sorted(((pref[b][t], b, t) for b in base_objs for t in tgt_objs), reverse=True)
        mapping, used = {}, set()
        for _, b, t in pairs:
            if b in mapping or t in used:
                continue
            mapping[b] = t
            used.add(t)
        return mapping

    def _resonator_mapping(self, source, target):
        """Seed from identity-free role-signature similarity, then iterate the resonator constraint
        (re-score each object's signature given the partners' currently-mapped images)."""
        base_objs, tgt_objs = self._objects(source), self._objects(target)
        tgt_sig = {t: self._signature(t, target) for t in tgt_objs}
        tgt_books = [tgt_sig[t] for t in tgt_objs]
        pref = {}
        for b in base_objs:
            est = self._cleanup_project(self._signature(b, source), tgt_books)
            pref[b] = {t: vsa.coincidence(est, tgt_sig[t]) for t in tgt_objs}
        mapping = self._greedy(pref, base_objs, tgt_objs)
        for _ in range(self.refine_iters):
            tgt_sig2 = {
                t: self._signature(t, target, partner_map={o: o for o in tgt_objs})
                for t in tgt_objs
            }
            pref = {}
            for b in base_objs:
                sig_b = self._signature(b, source, partner_map=mapping)
                pref[b] = {t: vsa.coincidence(sig_b, tgt_sig2[t]) for t in tgt_objs}
            new_map = self._greedy(pref, base_objs, tgt_objs)
            if new_map == mapping:
                break
            mapping = new_map
        return mapping

    def _hillclimb(self, mapping, source, target, base_objs, tgt_objs):
        """Local ascent on systematicity by swapping / reassigning targets (escapes weak seeds)."""
        best, best_s = dict(mapping), self.score(mapping, source, target)
        improved = True
        while improved:
            improved = False
            for b1 in base_objs:
                for t in tgt_objs:
                    cand = dict(best)
                    holder = next((k for k, v in cand.items() if v == t), None)
                    if holder is not None:
                        cand[holder] = cand.get(b1)
                    cand[b1] = t
                    cand = {k: v for k, v in cand.items() if v is not None}
                    s = self.score(cand, source, target)
                    if s > best_s:
                        best, best_s, improved = cand, s, True
        return best, best_s

    def map(self, source_props, target_props):
        """Recover the best base->target OBJECT correspondence (dict). Chooses the mapping that
        maximises systematicity across the resonator seed and `restarts` hill-climbed restarts."""
        base_objs, tgt_objs = self._objects(source_props), self._objects(target_props)
        if not base_objs or not tgt_objs:
            return {}
        candidates = [self._resonator_mapping(source_props, target_props)]
        for _ in range(self.restarts):
            shuffled = list(tgt_objs)
            self.rng.shuffle(shuffled)
            seed_map = {b: shuffled[i] for i, b in enumerate(base_objs) if i < len(shuffled)}
            candidates.append(seed_map)
        best, best_s = {}, -1.0
        for cand in candidates:
            climbed, s = self._hillclimb(cand, source_props, target_props, base_objs, tgt_objs)
            if s > best_s:
                best, best_s = climbed, s
        return best


def main():
    # ---- analog 1: solar system -> atom (2 objects, 3 shared relations) ----
    solar = [
        ("attracts", "sun", "planet"),
        ("revolves_around", "planet", "sun"),
        ("more_massive", "sun", "planet"),
    ]
    atom = [
        ("attracts", "nucleus", "electron"),
        ("revolves_around", "electron", "nucleus"),
        ("more_massive", "nucleus", "electron"),
    ]
    sm = StructureMapper(dim=1024, seed=0)
    m1 = sm.map(solar, atom)
    s1 = sm.score(m1, solar, atom)

    # ---- analog 2: a 3-object causal chain A->B->C mapped onto X->Y->Z ----
    base_chain = [
        ("cause", "A", "B"),
        ("cause", "B", "C"),
        ("bigger", "A", "C"),
    ]
    tgt_chain = [
        ("cause", "X", "Y"),
        ("cause", "Y", "Z"),
        ("bigger", "X", "Z"),
    ]
    sm2 = StructureMapper(dim=1024, seed=0)
    m2 = sm2.map(base_chain, tgt_chain)
    s2 = sm2.score(m2, base_chain, tgt_chain)

    # ---- non-analog: target shares no relational structure with the solar system ----
    junk = [
        ("likes", "cat", "dog"),
        ("owns", "dog", "ball"),
        ("near", "ball", "cat"),
    ]
    m3 = sm.map(solar, junk)
    s3 = sm.score(m3, solar, junk)

    # ---- determinism ----
    det = StructureMapper(dim=1024, seed=0).map(solar, atom) == m1 and StructureMapper(
        dim=512, seed=0
    ).map(base_chain, tgt_chain) == StructureMapper(dim=512, seed=0).map(base_chain, tgt_chain)

    print("=" * 64)
    print("VSA STRUCTURE-MAPPING -- relational-system analogy (Gentner) over spike codes")
    print("=" * 64)
    print(f"dim={sm.dim}  restarts={sm.restarts}  ops: bind/bundle from vsa, resonator cleanup")
    print(f"solar->atom  : {m1}  systematicity={s1:.2f}")
    print(f"chain A->B->C: {m2}  systematicity={s2:.2f}")
    print(f"non-analog   : {m3}  systematicity={s3:.2f}  (low: no shared structure)")
    print("recovers object correspondence as one mutually-constrained set, not one isolated shift.")
    print("=" * 64)

    # ---- self-check: recovery of the known mapping, low systematicity off-analog, determinism ----
    assert m1 == {"sun": "nucleus", "planet": "electron"}, f"solar->atom mapping wrong: {m1}"
    assert s1 == 1.0, f"correct analog should be fully systematic: {s1}"
    assert m2 == {"A": "X", "B": "Y", "C": "Z"}, f"causal-chain mapping wrong: {m2}"
    assert s2 == 1.0, f"causal-chain analog should be fully systematic: {s2}"
    assert s3 < 0.34, f"non-analogous target should be weakly systematic: {s3}"
    assert s1 > s3, "an analogous target must be more systematic than a non-analogous one"
    assert det, "mapping must be deterministic given the seed"
    assert "torch" not in sys.modules, "structure mapping runs on the spike-code VSA substrate"
    print(
        "self-check OK: recovered solar->atom and A->B->C one-to-one correspondences at full "
        "systematicity, off-analog stays low, deterministic"
    )


if __name__ == "__main__":
    main()
