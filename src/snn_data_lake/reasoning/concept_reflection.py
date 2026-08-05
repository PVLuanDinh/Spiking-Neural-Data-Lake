r"""
CONCEPT REFLECTION -- the developmental loop THINKS about what it just learned.

The autonomous developmental loop (reasoning/autonomous_brain_recovered.py) PERCEIVES -> holds an
objective in WM -> LEARNS a new concept (order-invariant readout) -> grows neurons -> consolidates to
the lake. It monitors HOW WELL it learns (MetacognitiveMonitor: confidence/surprise/learning-progress)
-- but it never reflects on the CONTENT: what the new concept is, how it relates to what it already
knows. A general learning/thinking brain should THINK about what it is learning.

This module adds that reflection, THROUGH THE SPIKING GLOBAL WORKSPACE, reusing the serial re-entrant
train of thought (reasoning/train_of_thought.py) that a feedforward J-space cannot form internally:

  learn a new concept c  ->  REFLECT(c):
    ignite c in the workspace -> its nearest ALREADY-LEARNED concept (by feature-prototype proximity)
    is the next candidate -> if it is related enough it IGNITES and is chained -> that concept seeds
    the next hop -> ... -> a train of thought over the brain's OWN concepts, held internally.
  Each consecutive pair the brain THOUGHT ABOUT is WIRED TOGETHER (fire together, wire together): a
  Hebbian concept-association weight is strengthened. So reflection is LOAD-BEARING -- over the loop it
  BUILDS a concept-association graph (what the brain has figured out about how its concepts relate),
  not a read-only annotation.

HONEST SCOPE (paramount): the "relatedness" that drives each hop is FEATURE-PROTOTYPE COSINE proximity
(concepts whose learned representations are close), gated by real spiking ignition -- NOT semantic
understanding and NOT a language model. The contribution is: the loop now runs a genuine serial
train of thought over its own concepts when it learns, and wires the ones it thinks about together.
The association is proximity; the workspace supplies the ignition-gated serial chaining + hold.

  PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.concept_reflection
Deps: numpy + stdlib. Reuses train_of_thought (spiking GlobalWorkspace); zero core edits.
"""

import numpy as np

from snn_data_lake.reasoning.train_of_thought import TrainOfThought


def _cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ConceptReflector:
    """Reflect on a newly-learned concept by chaining a train of thought over the brain's OWN concepts
    (nearest by feature-prototype proximity), each hop gated by spiking ignition, and WIRE the concepts
    it thinks about together (Hebbian). Accumulates a concept-association graph across the loop."""

    def __init__(self, threshold=0.55, top_k=3, seed=0):
        self.threshold = (
            threshold  # min prototype cosine for an association to "come to mind" (ignite)
        )
        self.top_k = top_k  # at most this many candidates compete per hop
        self.seed = seed
        self.protos = {}  # concept (str) -> prototype vector (np.ndarray)
        self.hebb = {}  # frozenset({a, b}) -> association weight (strengthened by reflection)

    def register(self, concept, prototype):
        """Record a concept's learned prototype (mean feature vector). Concept must be a hashable str."""
        self.protos[str(concept)] = np.asarray(prototype, dtype=float)
        return self

    def _associate(self, rheobase):
        """associate(broadcast, chain) -> [(concept, salience), ...]: the nearest OTHER learned concepts
        by prototype cosine, above threshold, mapped to a supra-rheobase workspace drive. Only concepts
        related enough IGNITE (come to mind); unrelated ones stay sub-threshold.

        `rheobase` is the ignition threshold READ OFF the workspace this train of thought will drive
        (see `reflect`), never a literal. It used to be `min(1.0, 0.55 + 0.44 * sim)` with a comment
        naming 0.55 as "rheobase" -- a second, silent copy of
        `thalamus/global_workspace.GlobalWorkspace.ignition_threshold`. Two definitions of one value
        is the P2 defect, and the tree already carries the correct idiom
        (`spike_chat_lake.LakeChat.__init__`: `self._rheobase = self._new_workspace().ignition_threshold`).
        The affine map is now derived from it rather than hand-tuned to it: sim in [threshold, 1]
        lands in (rheobase, 1.0], so every candidate that clears the similarity gate is
        supra-threshold BY CONSTRUCTION at any rheobase, instead of only at 0.55.

        Boundary-checked (a P1 rewrite that divides by a running quantity is how the last one
        introduced a singularity): there is no division here. rheobase=0.0 -> sal == sim > 0;
        rheobase=1.0 -> sal == 1.0, which ignites nothing -- correct, since a workspace whose
        rheobase is 1.0 cannot be ignited by any salience in [0,1] at all. Monotone in `sim`, so the
        candidate ORDER (and hence the workspace competition) is unchanged from the old form."""

        def associate(broadcast, chain):
            if broadcast not in self.protos:
                return []
            v = self.protos[broadcast]
            seen = set(chain)
            cands = []
            for c, p in self.protos.items():
                if c == broadcast or c in seen:
                    continue
                sim = _cos(v, p)
                if sim >= self.threshold:
                    sal = min(1.0, rheobase + (1.0 - rheobase) * sim)
                    cands.append((c, sal))
            cands.sort(key=lambda x: (-x[1], x[0]))
            return cands[: self.top_k]

        return associate

    def reflect(self, new_concept, max_steps=8):
        """Run a serial re-entrant TRAIN OF THOUGHT over the brain's concepts from `new_concept`, and
        WIRE each consecutive pair it thinks about together (Hebbian). Returns the chain of concepts
        the brain thought about (>=1; just the seed if nothing related ignites)."""
        c = str(new_concept)
        tot = TrainOfThought(seed=self.seed)
        # The rheobase of the workspace this thought will actually run in. `think()` rebuilds
        # `tot.ws` from the same constructor, so reading it here is the value that will gate.
        # Read DIRECTLY, not `getattr(..., 0.55)` (corrected 2026-08-05). That default was the same
        # literal 0.55 the P1 fix above exists to delete -- a second, silent copy of
        # `GlobalWorkspace.ignition_threshold` surviving as a fallback -- and it was DEAD: measured,
        # `TrainOfThought.__init__` and `.think()` both assign `self.ws = GlobalWorkspace(...)`,
        # which always defines the attribute. A workspace that ever lacked it should raise here, not
        # silently revert every reflection in the brain to a hardcoded threshold that no longer
        # matches the one doing the gating. Tightening, not a loosening: nothing that reached the
        # old default reaches anything different now.
        rheobase = float(tot.ws.ignition_threshold)
        chain = tot.think(c, self._associate(rheobase), max_steps=max_steps)
        for a, b in zip(chain, chain[1:]):  # fire together, wire together
            k = frozenset((a, b))
            self.hebb[k] = self.hebb.get(k, 0.0) + 1.0
        return chain

    def association_graph(self):
        """The concept-association graph the brain has built by reflecting: {(a, b): weight}, sorted."""
        return {
            tuple(sorted(k)): w
            for k, w in sorted(self.hebb.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
        }

    def strongest_link(self):
        """The concept pair the brain has most strongly wired together (or None)."""
        if not self.hebb:
            return None
        k = max(self.hebb, key=lambda kk: self.hebb[kk])
        return tuple(sorted(k)), self.hebb[k]


# =====================================================================================================
# DETERMINISTIC OFFLINE SELF-CHECK (numpy + stdlib; ASCII-only).
# =====================================================================================================
def _proto(cluster, idx, dim=16):
    """A prototype in a per-cluster band of dims (within-cluster cosine ~1, across-cluster ~0), plus a
    tiny per-item perturbation so cluster-mates are near but not identical. Deterministic, no RNG."""
    v = [0.0] * dim
    base = (cluster * 4) % dim
    for j in range(4):
        v[(base + j) % dim] = 1.0
    v[(base + idx) % dim] += 0.15
    return v


def main():

    # three concept clusters (A: a0,a1,a2 ; B: b0,b1 ; and one isolated concept z) ------------------
    protos = {
        "a0": _proto(0, 0),
        "a1": _proto(0, 1),
        "a2": _proto(0, 2),
        "b0": _proto(1, 0),
        "b1": _proto(1, 1),
        "z": _proto(3, 0),  # its own band -> related to nothing else
    }
    ref = ConceptReflector(threshold=0.55, seed=0)
    for c, p in protos.items():
        ref.register(c, p)

    # ---- reflect on a concept in cluster A: the train of thought must stay within the related cluster
    chain_a = ref.reflect("a0", max_steps=8)
    # ---- reflect on the isolated concept: nothing related ignites -> the brain thinks of only itself
    chain_z = ref.reflect("z", max_steps=8)
    # ---- reflect on A again + a B concept: repeated reflection STRENGTHENS the wired links -----------
    ref.reflect("a1")
    ref.reflect("a2")
    chain_b = ref.reflect("b0")
    graph = ref.association_graph()
    strongest = ref.strongest_link()

    print("=" * 90)
    print(
        "CONCEPT REFLECTION -- the loop thinks (train of thought) about what it learns + wires it"
    )
    print("=" * 90)
    print(f"reflect('a0') train of thought : {' -> '.join(chain_a)}")
    print(f"reflect('z')  (isolated)       : {' -> '.join(chain_z)}  (nothing related ignited)")
    print(f"reflect('b0') train of thought : {' -> '.join(chain_b)}")
    print("concept-association graph the brain BUILT by reflecting (fire together, wire together):")
    for (a, b), w in graph.items():
        print(f"    {a} <-> {b}   weight {w:.0f}")
    print(f"strongest wired concept pair   : {strongest}")
    print("=" * 90)

    # ---- self-checks (deterministic) ----------------------------------------------------------------
    # (1) reflecting on a cluster-A concept chains ONLY related (cluster-A) concepts, never B or z.
    assert len(chain_a) >= 2, (
        "reflection must chain the new concept to >=1 related concept it thought about"
    )
    assert chain_a[0] == "a0", "the train of thought must start at the reflected concept"
    assert all(c.startswith("a") for c in chain_a), (
        f"reflection must stay within the related cluster, got {chain_a}"
    )
    assert "z" not in chain_a and not any(c.startswith("b") for c in chain_a), (
        "unrelated concepts must NOT ignite"
    )
    # (2) an isolated concept ignites no association -> honest: the brain thinks of only itself (len 1).
    assert chain_z == ["z"], f"an isolated concept must reflect to itself only, got {chain_z}"
    # (3) reflection is LOAD-BEARING: it wired related concepts together, and repetition strengthened them.
    assert graph, "reflection must build a concept-association graph (wire concepts together)"
    assert all(
        a.startswith("a") == b.startswith("a") for (a, b) in graph if not (a == "z" or b == "z")
    ), "wired pairs must be within-cluster (the brain wires together what it thought about)"
    assert strongest is not None and strongest[1] >= 2, (
        "repeated reflection must STRENGTHEN a wired link (weight >= 2)"
    )
    # (4) honest: only proximity + spiking workspace; no language model in the path.

    print(
        f"self-check OK: the loop REFLECTS on a new concept via a spiking train of thought over its OWN "
        f"concepts (reflect('a0') -> {' -> '.join(chain_a)}), stays within the related cluster, thinks "
        f"only of itself for an isolated concept (honest), and WIRES what it thought about together "
        f"(strongest link {strongest[0]} weight {strongest[1]:.0f}) -- reflection is load-bearing, not "
        f"annotation. Relatedness is feature-prototype proximity gated by ignition, not semantics."
    )
    return 0


if __name__ == "__main__":
    main()
