r"""
LAKE SKILLS -- neuroplastic skill formation over the data lake's content signatures.

Repeated PRACTICE of two concepts that "fire together" strengthens a Hebbian weight between
them until the pair "wires together"; a connected cluster of wired concepts is a SKILL, and once
a practiced pair is strong enough it becomes AUTOMATIC (effortful -> permanent, reflexive ability).
A new input is ROUTED to the nearest skill by content-signature proximity ("reuse skill: route
input -> expert"). Weights genuinely accumulate with practice, clusters genuinely emerge from the
accumulated weights, and routing genuinely uses signature proximity -- nothing is faked.

HONEST FRAMING -- what this really is:
  This is PROXIMITY CLUSTERING + a HEBBIAN CO-ACTIVATION COUNTER over 64-d content-signatures.
  "Fire together, wire together" here means: every practice() call increments a real per-pair
  weight counter (Hebbian potentiation), a "skill" is a connected component of the graph whose
  edge weights cross a WIRED threshold, and "automatic" is that same weight crossing a higher
  threshold. It is NOT biological synaptogenesis: there are no synapses, neurotransmitters,
  dendritic spines, or protein synthesis -- just an accumulating scalar per concept pair and a
  content-vector distance. The neuroscience words name a mechanism analogy, not a biological claim.

CLUSTERING APPROACH -- direct signature distance (cosine), NOT ProximityIndex:
  ProximityIndex (src/snn_data_lake/substrate/proximity_index.py) indexes spike-EVENT lists
  ([(t, feature), ...]) under the Victor-Purpura spike-train edit metric. Our concepts are raw
  64-d crc32 bag-of-token signatures, not spike trains; pushing them through encode_latency to fit
  that metric would distort the geometry the signatures already define. So routing uses plain cosine
  proximity over signatures (the simplest correct choice for this data), while the skill clusters
  come from connected components of the Hebbian weight graph -- the part that actually "wires".

Reuses:
  research/skill_builder.py :: signature(tokens) -> list[float] length 64 (crc32 bag-of-tokens,
  normalized to [0,1]); crc32 is the project's persistence-stable hash, never builtin hash().
"""
import os
import sys

import numpy as np

# research/ is sys.path[0] when run as a script, but be explicit so imports are robust.
pass  # migrated: sys.path bootstrap removed (now an importable package)
from snn_data_lake.reasoning.skill_builder import signature   # noqa: E402  (path set up above)


class SkillMap:
    """Concepts wired into skills by practice-driven Hebbian co-activation counters.

    observe()   -> register a concept and store its content signature.
    practice()  -> co-activation: strengthen the Hebbian weight between two concepts (accumulates).
    skills()    -> connected components of the >= WIRED weight graph (each component is a skill).
    is_automatic() -> True once a pair's weight crosses the higher AUTOMATIC threshold.
    route()     -> the skill cluster nearest an input, by content-signature cosine proximity.
    """

    def __init__(self, learning_rate=0.1, wired=0.45, automatic=0.85):
        self.lr = learning_rate
        self.wired = wired          # weight at/above which a pair is "wired" (part of one skill)
        self.automatic = automatic  # weight at/above which a pair is "automatic" (permanent skill)
        self.sig = {}               # concept name -> np.ndarray content signature (the content key)
        self.w = {}                 # concept name -> {other name -> Hebbian co-activation weight}

    @staticmethod
    def _name(words):
        return " ".join(words)

    def observe(self, words):
        """Register a concept (idempotent); store its signature the first time it is seen."""
        name = self._name(words)
        if name not in self.sig:
            self.sig[name] = np.asarray(signature(words), dtype=float)
            self.w.setdefault(name, {})
        return name

    def practice(self, a_words, b_words):
        """Co-activate a and b once: fire-together -> strengthen (Hebbian potentiation, symmetric).
        Repeated calls ACCUMULATE, so effortful practice grows the weight toward automatic."""
        a = self.observe(a_words)
        b = self.observe(b_words)
        if a == b:
            return 0.0
        nb = self.w[a]
        nb[b] = nb.get(b, 0.0) + self.lr
        self.w[b][a] = self.w[b].get(a, 0.0) + self.lr
        return nb[b]

    def weight(self, a_words, b_words):
        return self.w.get(self._name(a_words), {}).get(self._name(b_words), 0.0)

    def is_automatic(self, a_words, b_words):
        """True once the practiced weight crosses the AUTOMATIC threshold (effortful -> automatic)."""
        return self.weight(a_words, b_words) >= self.automatic

    def skills(self):
        """Return the current skills: sorted tuples of concept names, one per connected component of
        the graph whose edge weights are >= WIRED. Concepts with no wired edge are not yet a skill."""
        adj = {n: set() for n in self.sig}
        for a, nbrs in self.w.items():
            for b, wt in nbrs.items():
                if wt >= self.wired:
                    adj[a].add(b)
                    adj[b].add(a)
        seen, comps = set(), []
        for start in self.sig:
            if start in seen or not adj[start]:
                continue
            stack, comp = [start], set()
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.add(x)
                stack.extend(adj[x] - seen)
            comps.append(tuple(sorted(comp)))
        return sorted(comps)

    def route(self, words):
        """Reuse skill: route an input to the nearest skill by content-signature cosine proximity.
        Returns {concept, similarity, skill} where skill is the wired cluster owning the nearest
        concept (or the lone concept if it is not yet wired into a skill)."""
        q = np.asarray(signature(words), dtype=float)
        qn = np.linalg.norm(q) or 1.0
        best, best_sim = None, -1.0
        for name, s in self.sig.items():
            sim = float(q @ s) / (qn * (np.linalg.norm(s) or 1.0))
            if sim > best_sim:
                best_sim, best = sim, name
        for comp in self.skills():
            if best in comp:
                return {"concept": best, "similarity": best_sim, "skill": comp}
        return {"concept": best, "similarity": best_sim, "skill": (best,)}


def _selfcheck():
    """Deterministic, offline assert-based self-check: practice wires concepts into skills."""
    sm = SkillMap(learning_rate=0.1, wired=0.45, automatic=0.85)

    # three groups of related concepts; concept names are their content words (collision-free bins).
    groups = {
        "MATH": [["add"], ["multiply"], ["divide"]],
        "BIO": [["cell"], ["gene"], ["protein"]],
        "MUSIC": [["note"], ["chord"], ["tempo"]],
    }
    for g in groups.values():
        for c in g:
            sm.observe(c)

    # PRACTICE WITHIN each group repeatedly; the pair we watch flip to automatic is add<->multiply.
    REPS = 12
    probe_a, probe_b = ["add"], ["multiply"]
    auto_early = None
    for rep in range(REPS):
        for g in groups.values():
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    sm.practice(g[i], g[j])          # co-activation within the group
        if rep == 2:                                  # after 3 reps (weight ~0.3): still effortful
            auto_early = sm.is_automatic(probe_a, probe_b)
    auto_late = sm.is_automatic(probe_a, probe_b)     # after 12 reps (weight ~1.2): now automatic

    # PRACTICE ACROSS groups only ONCE -> stays weak and below the WIRED threshold.
    sm.practice(["add"], ["cell"])

    within = sm.weight(["add"], ["multiply"])         # ~1.2 (12 reps)
    across = sm.weight(["add"], ["cell"])             # ~0.1 (1 rep)

    sk = sm.skills()
    want = sorted(tuple(sorted(" ".join(c) for c in g)) for g in groups.values())

    # ROUTE new in-domain inputs (unseen token added) to the correct skill by signature proximity.
    r_math = sm.route(["add", "numbers"])
    r_bio = sm.route(["gene", "strand"])
    r_music = sm.route(["tempo", "song"])

    # ----- ASCII report -----
    id_of = {name: i for i, comp in enumerate(sk) for name in comp}
    gname = {" ".join(sorted(" ".join(c) for c in g)): k for k, g in groups.items()}
    print("=" * 68)
    print("LAKE SKILLS -- fire-together-wire-together over content signatures")
    print("=" * 68)
    print(f"learning_rate={sm.lr}  WIRED>={sm.wired}  AUTOMATIC>={sm.automatic}")
    print("-" * 68)
    print(f"{'concept':<12}{'skill_id':<10}{'skill (cluster)'}")
    for comp in sk:
        label = gname.get(" ".join(comp), "?")
        for name in comp:
            print(f"{name:<12}{id_of[name]:<10}{label}")
    print("-" * 68)
    print("Hebbian co-activation weights (accumulated by practice):")
    print(f"  within  add <-> multiply : {within:5.2f}   "
          f"(wired={within >= sm.wired}, automatic={within >= sm.automatic})")
    print(f"  across  add <-> cell     : {across:5.2f}   "
          f"(wired={across >= sm.wired}, automatic={across >= sm.automatic})")
    print(f"  ratio within/across      : {within / across:5.1f}x")
    print("-" * 68)
    print("effort -> automatic (same pair, more practice):")
    print(f"  add <-> multiply is_automatic  after  3 reps: {auto_early}")
    print(f"  add <-> multiply is_automatic  after {REPS} reps: {auto_late}")
    print("-" * 68)
    print("route new in-domain input -> nearest skill (signature proximity):")
    for label, r in (("MATH", r_math), ("BIO", r_bio), ("MUSIC", r_music)):
        print(f"  {label:<6} probe -> concept '{r['concept']}' "
              f"(cos={r['similarity']:.2f}) -> skill {sorted(r['skill'])}")
    print("=" * 68)

    # ----- deterministic asserts -----
    # 1) practice made within-group wiring vastly stronger than across-group.
    assert within >= sm.wired, f"within-group pair must be wired ({within})"
    assert across < sm.wired, f"across-group pair must stay unwired ({across})"
    assert within > 5 * across, f"within must dominate across ({within} vs {across})"

    # 2) skills() recovers exactly the three practiced groups.
    assert sk == want, f"clusters must recover the practiced groups\n got {sk}\n want {want}"
    assert len(sk) == 3, f"expected 3 skills, got {len(sk)}"

    # 3) repeated practice flips a pair from effortful to automatic (False early, True after N reps).
    assert auto_early is False, "pair must NOT be automatic after only 3 reps (still effortful)"
    assert auto_late is True, "pair MUST be automatic after 12 reps (practice made it permanent)"

    # 4) a new in-domain input routes to the correct skill cluster by signature proximity.
    assert set(r_math["skill"]) == {"add", "multiply", "divide"}, r_math
    assert set(r_bio["skill"]) == {"cell", "gene", "protein"}, r_bio
    assert set(r_music["skill"]) == {"note", "chord", "tempo"}, r_music

    print("self-check OK: repeated practice wired co-activated concepts into 3 skills that route "
          "new inputs automatically (fire-together-wire-together); honestly a proximity cluster + "
          "Hebbian co-activation counter over content signatures, not biological synaptogenesis")


if __name__ == "__main__":
    _selfcheck()
