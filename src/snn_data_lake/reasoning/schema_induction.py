r"""
Offline SCHEMA INDUCTION -- the sleep-quiescent consolidation pass that grows a bounded, deduped
spiking semantic graph of PROTOTYPE/SCHEMA nodes out of the durable data lake's FLAT triple list.

THE GAP THIS FILLS
------------------
Durable knowledge in `substrate/home.py::KnowledgeLog` is a flat cue->answer + triple list. There is
no SCHEMA FORMATION: repeated episodes (dog isa mammal, cat isa mammal, ...) never generalise into a
hierarchical prototype ("things that are mammals") nor into a higher-level abstraction (dog isa animal,
via mammal isa animal). This module runs ONE bounded nightly pass that consolidates such episodes into
prototype nodes and emits DERIVED `isa`/`part-of`-style schema triples (tagged src='schema') ready to
be appended to the KnowledgeLog. It NEVER mutates the log itself -- the caller wires the append.

LITERATURE GROUNDING (mechanisms actually implemented here, not decoration)
---------------------------------------------------------------------------
* Complementary Learning Systems / systems consolidation (McClelland 1995; Kumaran-Hassabis-McClelland
  2016): a fast one-shot store (the KnowledgeLog's exact triples = hippocampus) is consolidated OFFLINE
  into slow, abstracted neocortical schemas via REPLAY. Here consolidate() replays the stored triples
  (a few shuffled passes = the reactivation stream) into a slow abstraction layer.
* Generative teacher-student consolidation (Spens & Burgess 2024, Nat Hum Behav): the stored items are
  the one-shot TEACHER; reactivation samples them as the replay stream that trains the schema layer.
  Our replay passes over the durable triples are exactly that sampled teacher stream (real memory only,
  no synthesised noise -- we replay what is actually stored).
* ART vigilance as the FALSE-SCHEMA FIREWALL (Grossberg, Adaptive Resonance Theory): a prototype is
  updated ONLY when the top-down expectation (the category's committed spike template) matches the
  bottom-up input above a VIGILANCE threshold rho; a mismatch triggers RESET + allocate-a-new-
  uncommitted-unit (neurogenesis). Vigilance is the abstraction knob and the safety mechanism: a
  triple that does not resonate can never corrupt an existing prototype -- it gets its own unit.
* Self-organising spiking map with local plasticity (spiking-SOM / STDP): the prototype templates are
  formed by LOCAL fast-learning (w <- I AND w, an intersection / coincidence rule), NOT backprop.

WHY ART-1, NOT reasoning/spiking_som.SOM
----------------------------------------
spiking_som.SOM is a Kohonen map over FLOAT vectors with a Gaussian neighbourhood and no reject path.
Schema induction needs the vigilance/reset/allocate firewall -- the one mechanism that guarantees a
non-matching episode cannot be forced into (and corrupt) a committed category. So, as the design brief
permits, this module implements a compact ART-1-style vigilance clustering over spiking triple
signatures instead. The signatures ARE spike codes (sparse binary spike patterns, crc32-seeded like
cortex/vsa_reasoning.hv), and the plasticity is local intersection learning -- spiking-native, no
rate-ANN backprop.

DESIGN (spiking-native, bounded, deduped)
-----------------------------------------
1. SIGNATURE: each triple (h, rel, t) -> a sparse binary SPIKE signature. The (rel, tail) "parent"
   identity gets the bit budget (k_rel + k_tail); the head gets a minority (k_head). So two triples
   with the same parent overlap strongly (share the parent bits) but not perfectly (heads differ) --
   vigilance rho then decides how much head-variation one prototype tolerates (the abstraction knob).
2. ART CLUSTER: present replayed signatures to committed prototypes; resonate above rho (fast-learn
   w <- I AND w, so the template ERODES to the shared parent bits = the extracted invariant) or, on
   reset, allocate a new prototype (bounded by max_prototypes). Members are deduped per prototype.
3. DERIVE: for an upper link (b rel c) whose (rel, b) prototype is SUPPORTED (>= min_support members),
   emit for each member a the transitive schema triple (a rel c) -- e.g. dog isa mammal + cat isa
   mammal (supported prototype) + mammal isa animal  ==>  dog isa animal, cat isa animal.
4. CONFIDENCE GATE: score each candidate derived triple with reasoning/kb_ranking.DistMult (trained on
   the ORIGINAL triples). DistMult is deliberately non-transitive, so the gate is FILTERED Hits@K: rank
   every entity as a tail for (a, rel), drop the other known true tails, and keep the candidate only if
   its tail lands in the top-k. A popular category hub (e.g. `animal`) ranks high as an `isa` tail for
   any member -- exactly the abstraction to keep -- while an arbitrary tail ranks low and is rejected.
   Two firewalls in series: ART support (structure) + DistMult filtered-rank (confidence).
5. BOUNDED / DEDUPED: <= max_prototypes prototypes, <= budget derived triples per night, canonical
   (h, rel, t) dedup, output sorted for determinism.

HONEST SCALE NOTE (see also `state()['_scale']`)
------------------------------------------------
This is BOUNDED schema abstraction with an ART false-schema firewall + a DistMult confidence gate. It
is NOT general concept learning: it only GENERALISES relations already present in the lake (transitive
closure through well-supported category hubs). It does not invent new relations, form cross-domain
analogies, or synthesise a solution program at test time (no fluid-reasoning claim). Its ceiling is the
relational structure the durable triples already contain; the value is that repeated real episodes
become durable, deduped, confidence-checked prototypes/schemas instead of staying a flat list.

Determinism: random.Random(seed) + np.random.RandomState(seed) per call (both reset at the top of
consolidate); crc32 (not builtin hash) for signature bits; no wall-clock, no global RNG. Deps: numpy
(pulled transitively by reasoning/kb_ranking.DistMult). NO torch. Representation/cognition only -- this
module never imports brainstem and issues no actuator command. consolidate() is guarded: on any
internal failure it returns a partial dict, it never raises.

  Run: python -m snn_data_lake.reasoning.schema_induction     # (also run by run_selfchecks.py)
"""

import random
import zlib
from collections import Counter


class SchemaInducer:
    """ART-vigilance-gated spiking prototype map + DistMult confidence gate over a durable triple list.

    consolidate(triples, budget) runs ONE bounded nightly consolidation pass and returns the DERIVED
    schema triples (tagged 'schema') plus counts. It does NOT touch any KnowledgeLog. Idempotent-ish:
    canonical dedup + a per-call RNG reset make repeated calls on the same triples stable.
    """

    def __init__(self, seed=0, vigilance=0.7, max_prototypes=256):
        self.seed = int(seed)
        self.vigilance = float(vigilance)  # ART rho: the abstraction knob / false-schema firewall
        self.max_prototypes = int(
            max_prototypes
        )  # hard cap on committed prototypes (bounded memory)

        # spike-signature bit budgets (parent = rel+tail dominates; head is a minority -> vigilance bites)
        self.D = 512
        self.k_rel = 16
        self.k_tail = 16
        self.k_head = 8

        # ART + derivation knobs
        self.alpha = (
            0.5  # ART choice-function tie-breaker (prefers smaller, more specific templates)
        )
        self.min_support = (
            2  # a hub must generalise >= this many members before we derive through it
        )
        self.replay_passes = 2  # teacher-student reactivation passes over the stored triples

        # DistMult confidence-gate knobs (compact; a nightly pass over a small store)
        self.dm_dim = 24
        self.dm_epochs = 250
        self.dm_lr = 0.15
        self.dm_batch = 64
        self.dm_topk_frac = (
            0.34  # keep a candidate whose tail is in the top ceil(frac*n_ent) FILTERED ranks
        )

        # rolling state (last consolidate)
        self._n_proto = 0
        self._n_schema = 0

    # ---- signatures: sparse binary SPIKE codes (crc32-seeded, deterministic across processes) --------
    def _fbits(self, token, k):
        """k spike positions for a feature token, seeded by crc32(token) (NOT builtin hash)."""
        rng = random.Random(zlib.crc32(token.encode("ascii", "ignore")))
        return {rng.randrange(self.D) for _ in range(k)}

    def _sig_bits(self, h, r, t):
        """Spike signature: parent (rel,tail) bits dominate; head bits are the tunable minority."""
        bits = set()
        bits |= self._fbits("R|" + r, self.k_rel)
        bits |= self._fbits("T|" + t, self.k_tail)
        bits |= self._fbits("H|" + h, self.k_head)
        return bits

    # ---- input hygiene: coerce to (h, rel, t) str tuples, drop malformed, dedup preserving order ------
    @staticmethod
    def _normalize(triples):
        out, seen = [], set()
        for tr in triples or []:
            if not isinstance(tr, (list, tuple)) or len(tr) < 3:
                continue
            key = (str(tr[0]), str(tr[1]), str(tr[2]))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    # ---- ART-1-style vigilance clustering with local intersection fast-learning ----------------------
    def _art_cluster(self, clean):
        """Replay the triples and cluster their spike signatures into prototypes. A prototype updates
        ONLY on resonance above vigilance (w <- I AND w, local coincidence learning); otherwise reset +
        allocate a new uncommitted unit (bounded by max_prototypes). Returns [{parent, members}]."""
        protos = []  # each: {"w": set[int], "members": set[triple], "parents": Counter[(rel,tail)]}
        order = list(clean)
        for _ in range(self.replay_passes):
            self.rng.shuffle(order)  # reactivation stream (sampled teacher)
            for tr in order:
                inp = self._sig_bits(*tr)
                if not inp:
                    continue
                # rank committed prototypes by ART choice function T = |I & w| / (alpha + |w|)
                scored = []
                for idx, p in enumerate(protos):
                    inter = len(inp & p["w"])
                    scored.append((inter / (self.alpha + len(p["w"])), inter, idx))
                scored.sort(key=lambda x: (-x[0], x[2]))  # desc T, idx tie-break -> deterministic
                placed = False
                for _t, inter, idx in scored:
                    if inter / len(inp) >= self.vigilance:  # VIGILANCE match test (the firewall)
                        p = protos[idx]
                        p["w"] = inp & p["w"]  # fast-learn: template erodes to invariant
                        p["members"].add(tr)
                        p["parents"][(tr[1], tr[2])] += 1
                        placed = True
                        break  # first resonant category wins
                if not placed and len(protos) < self.max_prototypes:
                    protos.append(
                        {"w": set(inp), "members": {tr}, "parents": Counter({(tr[1], tr[2]): 1})}
                    )
                # if the cap is hit and nothing resonated, the episode stays UNCOMMITTED (stay bounded)
        out = []
        for p in protos:
            parent, _ = p["parents"].most_common(1)[0]  # dominant (rel, tail) of the prototype
            heads = {tr[0] for tr in p["members"] if (tr[1], tr[2]) == parent}
            out.append({"parent": parent, "members": heads})
        return out

    # ---- DistMult confidence gate (reuse reasoning/kb_ranking.DistMult) ------------------------------
    def _train_distmult(self, clean):
        """Train a compact DistMult on the ORIGINAL triples for the confidence gate. Returns
        (model, ent_map, rel_map, known_tails, topk) or Nones. `known_tails` supports FILTERED ranking
        (do not penalise a candidate for the other true tails of its query); `topk` is the Hits@K
        acceptance depth. DistMult is deliberately NON-transitive, so we do NOT gate on the candidate's
        raw score (a valid transitive tail scores low); we gate on its FILTERED RANK -- a popular
        category hub (e.g. `animal`) ranks high as an `isa` tail for any member, which is exactly the
        abstraction we want to keep, while an arbitrary tail ranks low and is rejected."""
        try:
            import numpy as np

            from snn_data_lake.reasoning.kb_ranking import DistMult
        except Exception:
            return None, None, None, None, None
        ent, rel = {}, {}

        def eid(e):
            return ent.setdefault(e, len(ent))

        def rid(r):
            return rel.setdefault(r, len(rel))

        rows = np.array([[eid(h), rid(r), eid(t)] for (h, r, t) in clean], dtype=np.int64)
        model = DistMult(len(ent), len(rel), dim=self.dm_dim, seed=self.seed)
        for _ in range(self.dm_epochs):
            model.train_epoch(
                rows, self.dm_lr, self.dm_batch
            )  # rows shuffled in place; rows intact
        known = {}
        for h, r, t in rows:
            known.setdefault((int(h), int(r)), set()).add(int(t))
        topk = max(2, round(self.dm_topk_frac * len(ent)))
        return model, ent, rel, known, topk

    def _confident(self, model, ent, rel, known, topk, a, r, c):
        """Filtered Hits@K confidence: rank every entity as a tail for (a, r), remove the OTHER known
        true tails, and accept iff the candidate c lands in the top-k. Unseen symbols cannot be judged
        -> defer to the structural (ART support) gate."""
        if a not in ent or c not in ent or r not in rel:
            return True
        scores = (model.E[ent[a]] * model.R[rel[r]]) @ model.E.T
        gold = scores[ent[c]]
        for other in known.get((ent[a], rel[r]), ()):  # filtered ranking
            if other != ent[c]:
                scores[other] = -1e30
        rank = int((scores > gold).sum()) + 1
        return rank <= topk

    # ---- the one bounded nightly pass ----------------------------------------------------------------
    def consolidate(self, triples, budget=64):
        """Run ONE bounded consolidation pass over the durable triples. Returns
        {"prototypes": int, "derived": [[h,rel,t,'schema'], ...], "kept": int, "rejected": int}.
        Guarded: on any failure returns a partial dict, never raises. Does NOT mutate any KnowledgeLog."""
        # confidence_gate_ran: whether the SECOND firewall actually ran. _train_distmult() returns
        # (None,)*5 on any import failure, and the filter below is `if model is not None and not
        # confident(...)` -- so a missing numpy/kb_ranking silently downgrades "two firewalls in
        # series" to one, passes EVERY candidate, and leaves `rejected` reading a constant 0 with
        # nothing in the result to say the gate was skipped. Callers that write these triples to the
        # durable lake need to be able to tell those two zeros apart (v4.78).
        result = {
            "prototypes": 0,
            "derived": [],
            "kept": 0,
            "rejected": 0,
            "confidence_gate_ran": False,
        }
        try:
            self.rng = random.Random(self.seed)  # per-call reset -> deterministic
            clean = self._normalize(triples)
            if len(clean) < 2:
                self._n_proto, self._n_schema = 0, 0
                return result

            protos = self._art_cluster(clean)
            result["prototypes"] = len(protos)
            self._n_proto = len(protos)

            # supported hubs only: (rel, hub) -> set of member heads that are `rel hub`
            proto_by_parent = {}
            for p in protos:
                if len(p["members"]) >= self.min_support:
                    proto_by_parent.setdefault(p["parent"], set()).update(p["members"])

            model, ent, rel, known, topk = self._train_distmult(clean)
            result["confidence_gate_ran"] = model is not None

            original = set(clean)
            seen, derived, rejected = set(), [], 0
            for b, relation, c in clean:  # (b rel c) is a candidate UPPER link
                members = proto_by_parent.get((relation, b))  # heads a with (a rel b), supported
                if not members:
                    continue
                for a in sorted(members):  # sorted -> deterministic (no set order)
                    cand = (a, relation, c)
                    if a == c or cand in original or cand in seen:
                        continue
                    seen.add(cand)
                    if model is not None and not self._confident(
                        model, ent, rel, known, topk, a, relation, c
                    ):
                        rejected += 1
                        continue
                    if len(derived) >= budget:
                        break
                    derived.append([a, relation, c, "schema"])
                if len(derived) >= budget:
                    break

            derived.sort(key=lambda d: (d[0], d[1], d[2]))  # canonical order -> idempotent output
            result["derived"] = derived
            result["kept"] = len(derived)
            result["rejected"] = rejected
            self._n_schema = len(derived)
        except Exception as exc:  # guard: partial dict, never raise
            result["_note"] = "partial: " + str(exc)[:80]
        return result

    def state(self):
        """A compact, HONEST snapshot of the last consolidation."""
        return {
            "prototypes": self._n_proto,
            "schemas": self._n_schema,
            "_scale": (
                f"bounded ART-vigilance schema abstraction (rho={self.vigilance:.2f}): transitive/"
                f"prototype schemas gated by a support>={self.min_support} hub + a DistMult "
                f"confidence gate; NOT general concept learning -- it only generalises relations "
                f"already in the lake (no new relation invention, no cross-domain analogy), "
                f"<={self.max_prototypes} prototypes / <=budget derived per night."
            ),
        }


def _selfcheck():
    # (1) a clear taxonomy: two mammals, two animals-of-mammal/bird, a chain to abstract through.
    taxonomy = [
        ["dog", "isa", "mammal"],
        ["cat", "isa", "mammal"],
        ["mammal", "isa", "animal"],
        ["sparrow", "isa", "bird"],
        ["bird", "isa", "animal"],
    ]
    si = SchemaInducer(seed=0, vigilance=0.7)
    res = si.consolidate(taxonomy, budget=64)
    derived = {(d[0], d[1], d[2]) for d in res["derived"]}

    # (2) CONTROL: random unrelated triples -> the ART support + DistMult firewall should keep ~nothing.
    crng = random.Random(3)
    pool = [f"e{i}" for i in range(40)]
    rels = ["r0", "r1", "r2", "r3"]
    control = [[crng.choice(pool), crng.choice(rels), crng.choice(pool)] for _ in range(30)]
    sic = SchemaInducer(seed=0, vigilance=0.7)
    resc = sic.consolidate(control, budget=64)

    # (3) determinism: two fresh seed=0 instances agree exactly.
    r_a = SchemaInducer(seed=0).consolidate(taxonomy)
    r_b = SchemaInducer(seed=0).consolidate(taxonomy)

    print("=" * 78)
    print("SCHEMA INDUCTION -- offline sleep-tick consolidation (ART vigilance + DistMult gate)")
    print("=" * 78)
    print(
        f"taxonomy  : {len(taxonomy)} triples -> {res['prototypes']} prototypes; "
        f"derived schemas (src='schema'):"
    )
    for d in res["derived"]:
        print(f"            {d[0]:<8} {d[1]:<5} {d[2]:<8}  [{d[3]}]")
    print(
        f"            kept={res['kept']}  rejected={res['rejected']}  "
        f"confidence_gate_ran={res['confidence_gate_ran']}"
    )
    print(
        f"control   : {len(control)} random triples -> {resc['prototypes']} prototypes; "
        f"kept={resc['kept']} rejected={resc['rejected']} (firewall)"
    )
    print(f"state     : {si.state()['_scale']}")
    print("=" * 78)

    # ---- asserts ----
    # (1) at least one correct higher-level schema induced (transitive closure through a supported hub)
    assert ("dog", "isa", "animal") in derived, (
        "must derive dog isa animal via the mammal->animal chain"
    )
    assert ("cat", "isa", "animal") in derived, (
        "must derive cat isa animal via the mammal->animal chain"
    )
    assert res["kept"] >= 1, "at least one confident schema expected on a clean taxonomy"
    # (2) the firewall rejects noise: candidates were generated from the random data but the ART-support
    #     + DistMult filtered-rank gate rejected them, keeping far fewer schemas than the real taxonomy.
    assert resc["kept"] <= 1, "ART support + DistMult firewall must reject random-triple noise"
    assert resc["kept"] < res["kept"], "control must yield fewer schemas than the real taxonomy"
    assert resc["rejected"] >= 1, (
        "the DistMult gate must visibly reject noise candidates on the control"
    )
    # (3) determinism
    assert r_a == r_b, "two seed=0 runs must be identical"
    # (4) bounded
    assert res["prototypes"] <= si.max_prototypes, "prototypes must respect the cap"
    assert len(res["derived"]) <= 64, "derived must respect the budget"
    # ASCII-only output guarantee for anything we would hand back to the KnowledgeLog
    for d in res["derived"]:
        assert all(ord(ch) < 128 for part in d[:3] for ch in str(part)), (
            "derived triples must be ASCII"
        )

    print(
        f"self-check OK: induced {res['kept']} confident schema(s) incl. dog/cat isa animal from "
        f"{res['prototypes']} prototypes; control noise -> {resc['kept']} kept (firewall); "
        f"deterministic; bounded (<={si.max_prototypes} protos, <=budget derived)."
    )


if __name__ == "__main__":
    _selfcheck()
