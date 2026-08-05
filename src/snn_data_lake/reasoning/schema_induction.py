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
3. TRANSITIVITY GATE (the FIRST firewall, and the one that decides truth): only relations that
   legitimately CHAIN may be derived through. `_TRANSITIVE` is ported verbatim from
   `reasoning/derivation.py:37` (P2 -- port the unique capability rather than re-invent it; the
   self-check below asserts the two copies have not drifted). Without it this module chained ANY
   relation: `mary has hat` + `john has hat` (parent ("has","hat"), support 2) + `hat has brim`
   derived `mary has brim` / `john has brim` -- MEASURED, 2 false schemas, and the caller
   (`cortex/brain.py`) appends them to the ONE durable lake with src='schema', which
   `hippocampus/lake_consolidation.py:17` classifies as genuinely-learned and RETAINS. The DistMult
   gate structurally cannot be this check: it is deliberately NON-transitive (see 5).
4. DERIVE: for an upper link (b rel c) whose (rel, b) prototype is SUPPORTED (>= min_support members),
   emit for each member a the transitive schema triple (a rel c) -- e.g. dog isa mammal + cat isa
   mammal (supported prototype) + mammal isa animal  ==>  dog isa animal, cat isa animal.
5. CONFIDENCE GATE: score each candidate derived triple with reasoning/kb_ranking.DistMult (trained on
   the ORIGINAL triples). DistMult is deliberately non-transitive, so the gate is FILTERED Hits@K: rank
   every entity as a tail for (a, rel), drop the other known true tails, and keep the candidate only if
   its tail lands in the top-k. A popular category hub (e.g. `animal`) ranks high as an `isa` tail for
   any member -- exactly the abstraction to keep -- while an arbitrary tail ranks low and is rejected.
   Three firewalls in series: transitivity (truth) + ART support (structure) + DistMult (confidence).
   The DistMult scorer can DIVERGE -- its objective is trilinear and therefore unbounded in embedding
   norm, and `kb_ranking.train_epoch` applies no norm constraint. MEASURED, and stated with its store
   so it is re-derivable: counting NON-FINITE entries of the ENTITY table `model.E` after the 250
   epochs below, a 115-triple store (12 category->top links + 50x{isa,has} + 3 has-links, 69 entities)
   gives 1,656; the same shape at 1,615 triples (819 entities) gives 19,656; the 89-triple live lake
   gives 0. The relation table `model.R` adds a further 48 non-finite entries in both diverging cases
   (2 relations x dim 24) -- the figures above are E only, which is why a re-count that includes R
   reads 1,704 / 19,704. Re-verified 2026-08-04 against both stores; all six numbers reproduce. NaN
   is the WORST failure available here, because `_confident` then compares NaN to NaN, every
   `scores > gold` is False, every candidate ranks 1st and is ACCEPTED, `rejected` stays 0, and the
   gate reports that it ran -- unchecked schemas into the durable lake, wearing a green flag. So
   `_train_distmult` verifies the trained tables are FINITE and otherwise reports that the gate did
   not run, which is a state the caller already refuses to write on.
   The obvious fix -- the standard DistMult max-norm constraint ||e|| <= 1 -- was implemented and
   REJECTED ON MEASUREMENT, recorded here so it is not re-tried: it removes the NaN in every case
   above, and it also moves the true `dog isa animal` from filtered rank 2 to rank 5 against topk=2,
   i.e. it makes the gate reject exactly the abstraction this module exists to keep. Constraining the
   norm at dim=24 costs more discrimination than the divergence costs. The residual defect is in
   `reasoning/kb_ranking.py`'s unconstrained trainer, not here; this module refuses to trust it
   rather than silently laundering it.
6. BOUNDED / DEDUPED: <= max_prototypes prototypes, <= budget derived triples per night, canonical
   (h, rel, t) dedup, output sorted for determinism.

HONEST SCALE NOTE (see also `state()['_scale']`)
------------------------------------------------
This is BOUNDED schema abstraction with an ART false-schema firewall + a DistMult confidence gate. It
is NOT general concept learning: it only GENERALISES relations already present in the lake (transitive
closure through well-supported category hubs). It does not invent new relations, form cross-domain
analogies, or synthesise a solution program at test time (no fluid-reasoning claim). Its ceiling is the
relational structure the durable triples already contain; the value is that repeated real episodes
become durable, deduped, confidence-checked prototypes/schemas instead of staying a flat list.

DEPENDENCIES (P7, and what the retired zero-dependency rule cost here)
----------------------------------------------------------------------
numpy is an UNCONDITIONAL core dependency of this project, so it is imported at module scope. It used
to be imported inside `_train_distmult`'s try/except and described as "pulled transitively by
kb_ranking" -- residue of the retired rule that made a core dependency read as accidental, and worse,
made a MISSING NUMPY and a BROKEN CROSS-TIER IMPORT indistinguishable: both returned (None,)*5 and
silently downgraded the firewall chain. Now numpy is hard, and the guarded lazy import covers exactly
one thing -- the governed -> ungoverned edge into `reasoning/kb_ranking` -- so `confidence_gate_ran`
False has a single, nameable cause.

What was NOT converted, with the measurement P7 requires for a rejection: `_art_cluster` keeps stdlib
`set` intersection rather than packed numpy bitsets. Measured as `t_art / (t_art + t_distmult)`,
MEDIAN OF 5 runs (a single run of this is noisy enough to mislead -- an earlier single-shot pass
recorded 13.7 / 15.3 / 11.9 and the third of those does not reproduce): 89-triple live lake **12.0%**
(9.8-14.9 over the 5), 415 triples **15.2%** (12.7-19.5), 1,615 triples **15.3%** (14.4-16.8). numpy
DistMult training is the other ~85%. A packed bitset rewrite could remove part of a ~1/7 slice while
changing the tie-break ordering inside a firewall. Rejected on that measurement, not on dependency
count -- and note the rejection does not turn on the exact percentage: it survives anywhere in the
10-20% band the repeats span.

Determinism: `random.Random(seed)` is reset at the top of consolidate() and drives the ART replay
shuffle; the numpy side is `np.random.RandomState(seed)` created inside `kb_ranking.DistMult(seed=)`
per call, i.e. in `_train_distmult`, NOT at the top of consolidate -- this paragraph said "both reset
at the top of consolidate" and only the stdlib one is. MEASURED while constructing the red for the
determinism assert, and worth knowing before anyone tunes it: varying ONLY the replay-shuffle seed
over 24 draws changed the output on NONE of 4 store shapes (taxonomy, a 23-triple ambiguous
hierarchy, a 67-triple hub set, 120 noisy triples), so every bit of this module's seed-sensitivity
today comes from the DistMult init. The replay shuffle is faithful to the CLS grounding above and is
currently INERT on the result -- an open question about `_art_cluster`, not a claim that it is wrong.
crc32 (not builtin hash) for signature bits, over UTF-8 bytes so that two entities
differing only outside ASCII cannot collide onto one signature; no wall-clock, no global RNG. NO
torch -- CPU/stdlib+numpy only, so this runs on the no-GPU gate (P5). E-PROP: nothing here is on a
learning path. The only gradient in the module is DistMult's hand-written SGD (kb_ranking, offline,
in-method), which trains a throwaway per-call SCORER used as a read-only filter; it updates no brain
weight, there is no .backward(), no BPTT, and no autograd graph. Representation/cognition only -- this
module never imports brainstem and issues no actuator command. consolidate() is guarded: on any
internal failure it returns a partial dict, it never raises.

  Run: python -m <this module's dotted path>   # printed by the self-check; also run by run_selfchecks
"""

import random
import zlib
from collections import Counter

import numpy as np

# Relations that legitimately CHAIN (a isa b, b isa c => a isa c). PORTED VERBATIM from
# `reasoning/derivation.py:37`, whose own comment attributes it to `brain_loop.Brain._TRANSITIVE` --
# a module that no longer exists anywhere in this tree, so that provenance is repeated here as
# derivation.py's claim, not as a live reference. P2 says port the
# unique capability rather than fork a second mechanism, so this is a copy of a shared VOCABULARY, not
# a second deduction engine, and `_selfcheck` asserts the two copies are still identical. It is
# deliberately duplicated rather than imported: after promotion this module sits in a governed region
# and a module-scope import of the ruff-/coverage-excluded `reasoning/` tier would convert a guarded
# lazy edge into a hard governed dependency.
# "has" / "resembles" / "causes" do NOT chain, and bare "is" marks PREDICATION ("sky is blue"), not
# class membership -- chaining any of them manufactures false conclusions, which this module then
# writes to the ONE durable lake.
_TRANSITIVE = frozenset({"isa", "partof", "part_of", "subclassof", "kindof", "typeof"})


def _chains(relation):
    """True if `relation` legitimately supports transitive derivation. Same normalisation as
    `reasoning/derivation.py:82` (lowercase, strip spaces) so the two copies decide alike."""
    return str(relation).lower().replace(" ", "") in _TRANSITIVE


class SchemaInducer:
    """ART-vigilance-gated spiking prototype map + DistMult confidence gate over a durable triple list.

    consolidate(triples, budget) runs ONE bounded nightly consolidation pass and returns the DERIVED
    schema triples (tagged 'schema') plus counts. It does NOT touch any KnowledgeLog. Idempotent-ish:
    canonical dedup + a per-call RNG reset make repeated calls on the same triples stable.
    """

    # ---- P1 (no-hardcode) declaration for every value below that ARBITRATES BEHAVIOUR -------------
    # Carried explicitly across the tier boundary rather than silently, per P1's honesty guard. The
    # caller (`cortex/brain.py`) constructs a FRESH SchemaInducer every 4th night and persists no
    # inducer state, so for ALL of these a running-baseline form is additionally not reproducible from
    # the persisted policy -- P1 exception (b) applies to the whole block, on top of the per-value
    # reason given:
    #   vigilance    -- P1 (a): a fixed point on a NORMALIZED output. It thresholds
    #                   |I & w| / |I|, a ratio in [0,1]; dimensionless, like `p >= 0.3`.
    #   alpha        -- P1 (a): tie-break regulariser in the ART choice ratio |I & w| / (alpha + |w|);
    #                   it orders normalized quantities and sets no scale of its own.
    #   dm_topk_frac -- P1 (a): already the baseline-relative form. The acceptance depth is
    #                   `frac * n_entities`, i.e. relative to the store, NOT a fixed k.
    #   min_support  -- definitional floor, not a tuned knob: 2 is the smallest integer for which the
    #                   word "repeated" is true, and a single episode is not a repetition. Below 2 the
    #                   quantity is incoherent, so there is nothing for a baseline to learn.
    #   D/k_*        -- representation CAPACITY (code width and per-feature spike counts), not
    #                   arbiters. They set the resolution at which `vigilance` -- the actual arbiter --
    #                   is evaluated; the head/parent bit ratio is what gives rho something to bite on.
    #   dm_dim/epochs/lr/batch, replay_passes -- optimiser and replay budgets of a throwaway per-call
    #                   scorer, not thresholds on any decision. The divergence they can cause is
    #                   DETECTED (the finiteness check in `_train_distmult`), not tuned away and not
    #                   norm-constrained -- see 5, where the max-norm fix is rejected on measurement.
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
        """k spike positions for a feature token, seeded by crc32(token) (NOT builtin hash).

        UTF-8, not `encode("ascii", "ignore")`. The old encoder DELETED every non-ASCII character
        before hashing, so any two entities differing only outside ASCII received the SAME signature
        and were clustered into one prototype -- which then cross-derives between them into the
        durable lake. The lake is fed by chat and web ingest, so non-ASCII entities are reachable.
        Bit-exact for ASCII input (verified: identical bit sets under both encoders), so the live
        consumer's numbers are unchanged by this."""
        rng = random.Random(zlib.crc32(token.encode("utf-8")))
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
        abstraction we want to keep, while an arbitrary tail ranks low and is rejected.

        Returning Nones is the ONE way this method says "the confidence gate did not run", and the
        caller turns that into `confidence_gate_ran=False`. It has exactly two causes, both real and
        both reachable: (1) the governed -> ungoverned import of `reasoning/kb_ranking` failed, and
        (2) training did not produce a usable scorer -- a NON-FINITE embedding table. (2) is the case
        this method used to report as a healthy gate: NaN embeddings make every `scores > gold`
        comparison False, so every candidate ranks 1st and is accepted, `rejected` stays 0, and the
        caller wrote unchecked schemas to the durable lake believing the evidence was good."""
        try:
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
        if not (np.isfinite(model.E).all() and np.isfinite(model.R).all()):
            return None, None, None, None, None  # degenerate scorer -> the gate did NOT run
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
        # confidence_gate_ran: whether the CONFIDENCE firewall actually ran. _train_distmult() returns
        # (None,)*5 when it cannot, and the filter below is `if model is not None and not
        # confident(...)` -- so without the flag a broken gate silently downgrades the firewall chain,
        # passes EVERY candidate, and leaves `rejected` reading a constant 0 with nothing in the
        # result to say so. Callers that write these triples to the durable lake need to tell those
        # zeros apart (v4.78); `cortex/brain.py` refuses the durable write when this is False.
        # The flag now covers a THIRD state it used to report as healthy: a NaN-diverged scorer.
        # Reachable red for the False branch is constructed in _selfcheck below.
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
                if not _chains(relation):
                    continue  # TRANSITIVITY FIREWALL -- chaining `has`/`causes` fabricates schemas
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
                f"bounded ART-vigilance schema abstraction (rho={self.vigilance:.2f}): schemas gated "
                f"by a transitive-relation whitelist ({len(_TRANSITIVE)} relations) + a "
                f"support>={self.min_support} hub + a DistMult confidence gate; NOT general concept "
                f"learning -- it only generalises the {len(_TRANSITIVE)} relations that legitimately "
                f"chain, and only where they are already in the lake (no new relation invention, no "
                f"cross-domain analogy), <={self.max_prototypes} prototypes / <=budget per night."
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
    #     `rejected` counts DistMult-gate rejections only, and one 30-triple draw often yields no
    #     candidate at all (no supported hub with an upper link), so on a single draw the count is a
    #     coin flip: re-measured across 7 seed shifts, the old single-draw `rejected >= 1` held on
    #     3/7 seeds (rejected was 0 on 4 of them, 0..4 overall). across 32 fresh draws per seed the
    #     gate rejects on 5..14 of the 32 draws, 8..29 rejections in total -- so the visibility
    #     claim is asserted below on the 32-draw TOTAL (same >= 1 bar, 32x the sample, ~1 s extra);
    #     the kept firewalls stay on the pinned first draw, which is byte-identical to the old one.
    #     The control relations MUST be drawn from _TRANSITIVE. They used to be r0..r3, which the
    #     transitivity firewall now rejects before a candidate is ever built, leaving `rejected_total`
    #     structurally 0. MEASURED 2026-08-04: with rels=r0..r3 the total is 0 on all 7 seed shifts x
    #     32 draws; with chaining rels it is 8..23 (>=1 on every seed, an 8x margin on the bar below).
    #     Note what that failure would have been, because the earlier draft of this comment had the
    #     P8 term backwards: a structurally-0 total makes `rejected_total >= 1` permanently RED -- no
    #     reachable GREEN -- so it fails loudly rather than passing vacuously. The vacuous-pass shape
    #     P8 forbids is the opposite one, and is what the DELETED block at the bottom records.
    crng = random.Random(3)
    pool = [f"e{i}" for i in range(40)]
    rels = sorted(_TRANSITIVE)[:4]
    control = [[crng.choice(pool), crng.choice(rels), crng.choice(pool)] for _ in range(30)]
    sic = SchemaInducer(seed=0, vigilance=0.7)
    resc = sic.consolidate(control, budget=64)
    rejected_total = resc["rejected"]
    for _ in range(31):  # 31 more control draws from the same stream -> the aggregate count
        extra = [[crng.choice(pool), crng.choice(rels), crng.choice(pool)] for _ in range(30)]
        rejected_total += SchemaInducer(seed=0, vigilance=0.7).consolidate(extra, budget=64)[
            "rejected"
        ]

    # (3) determinism: two fresh seed=0 instances agree exactly.
    r_a = SchemaInducer(seed=0).consolidate(taxonomy)
    r_b = SchemaInducer(seed=0).consolidate(taxonomy)

    # (4) TRANSITIVITY FIREWALL, with its red constructed rather than assumed. This exact input used
    #     to derive `mary has brim` + `john has brim` and hand them to cortex/brain.py, which appends
    #     them to the ONE durable lake as genuinely-learned knowledge. `has` does not chain.
    non_transitive = [
        ["mary", "has", "hat"],
        ["john", "has", "hat"],
        ["hat", "has", "brim"],
        ["sue", "has", "hat"],
    ]
    res_nt = SchemaInducer(seed=0, vigilance=0.7).consolidate(non_transitive, budget=64)
    #     ...and the SAME shape with a chaining relation must still derive, so the firewall is shown
    #     to reject the relation and not merely to reject everything (a control for the control).
    transitive_twin = [[h, "partof", t] for (h, _r, t) in non_transitive]
    res_tt = SchemaInducer(seed=0, vigilance=0.7).consolidate(transitive_twin, budget=64)

    # (5) CONFIDENCE-GATE RED: the flag exists to separate "0 rejected, evidence was good" from
    #     "0 rejected, nothing was checked". Nothing observed it False anywhere, so the distinction
    #     was untested in both directions. Force the governed -> ungoverned import to fail and read it.
    import sys

    _KB = "snn_data_lake.reasoning.kb_ranking"
    _saved, _had = sys.modules.get(_KB), _KB in sys.modules
    sys.modules[_KB] = None  # None makes `from ... import DistMult` raise ImportError
    try:
        res_off = SchemaInducer(seed=0, vigilance=0.7).consolidate(taxonomy, budget=64)
    finally:
        if _had:
            sys.modules[_KB] = _saved
        else:
            sys.modules.pop(_KB, None)

    # (5b) DIVERGED-SCORER RED, added 2026-08-04 because the finiteness check in `_train_distmult`
    #      -- the guard section 5 above calls the WORST failure available here -- had NO criterion
    #      anywhere. Measured: deleting that check leaves this entire self-check GREEN, because every
    #      other store here is small enough that DistMult converges. So the module's headline safety
    #      addition was untested. This is the 115-triple store section 5 quotes (12 category->top
    #      links, 50 entities x {isa, has}, 3 has-links; 69 entities), and its ENTITY table reaches
    #      1,656 non-finite entries after the 250 training epochs -- so the gate must report that it
    #      did NOT run even though `kb_ranking` imported fine, which (5) and (A12) pin as the other
    #      branch. Deterministic: `random.Random(11)` here and legacy `RandomState(0)` inside
    #      DistMult, both stream-stable across numpy versions. Costs ~75 ms.
    _drng = random.Random(11)
    _cats = [f"cat{i}" for i in range(12)]
    diverging = [[c, "isa", _drng.choice(["animal", "artifact", "plant"])] for c in _cats]
    for _i in range(50):
        diverging.append([f"e{_i}", "isa", _drng.choice(_cats)])
        diverging.append([f"e{_i}", "has", _drng.choice(["part_a", "part_b", "part_c"])])
    diverging += [[p, "has", "brim"] for p in ("part_a", "part_b", "part_c")]
    res_div = SchemaInducer(seed=0, vigilance=0.7).consolidate(diverging, budget=32)

    # (6) SIGNATURE COLLISION RED: the seed encoder used to be encode("ascii", "ignore"), which
    #     DELETED non-ASCII characters before hashing -- so "cafe<accent>" and "cafe" hashed to one
    #     signature and were clustered into a single prototype. The lake is fed by chat/web ingest,
    #     so this input is reachable. Under the old encoder this assert is RED.
    _sig = SchemaInducer(seed=0)
    #     Written as an escape, not a literal accented character, so the SOURCE FILE stays pure ASCII
    #     (this module is executed in-process by tests/test_selfchecks.py on the Windows legs).
    bits_hi = _sig._fbits("H|caf\u00e9", _sig.k_head)
    bits_lo = _sig._fbits("H|caf", _sig.k_head)

    # (7) DRIFT CANARY for the ported whitelist: this module carries a COPY of
    #     reasoning/derivation.py's _TRANSITIVE (P2 -- port, do not fork). Imported lazily and only
    #     here, so the governed tier gains no module-scope dependency on the ungoverned one.
    try:
        from snn_data_lake.reasoning.derivation import _TRANSITIVE as _DERIV_TRANSITIVE
    except Exception as exc:  # pragma: no cover - reported, never silently passed
        _DERIV_TRANSITIVE, _drift = None, f"canary SKIPPED ({type(exc).__name__})"
    else:
        _drift = "identical" if _DERIV_TRANSITIVE == _TRANSITIVE else "DRIFTED"

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
        f"control   : {len(control)} random triples over {len(rels)} chaining relations -> "
        f"{resc['prototypes']} prototypes; kept={resc['kept']} rejected={resc['rejected']} "
        f"(firewall); 32 draws -> {rejected_total} rejections total"
    )
    print(
        f"transitive: 'has' chain (mary/john has hat + hat has brim) -> kept={res_nt['kept']} "
        f"(refused); same shape as 'partof' -> kept={res_tt['kept']} (derived)"
    )
    print(
        f"gate-off  : kb_ranking import blocked -> confidence_gate_ran="
        f"{res_off['confidence_gate_ran']} rejected={res_off['rejected']} "
        f"(cortex/brain.py refuses the durable write on this)"
    )
    print(
        f"gate-nan  : {len(diverging)}-triple store, import intact, DistMult diverged -> "
        f"confidence_gate_ran={res_div['confidence_gate_ran']} "
        f"(the {res_div['kept']} candidates are NOT written)"
    )
    print(
        f"whitelist : {len(_TRANSITIVE)} chaining relations; vs reasoning/derivation.py: {_drift}"
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
    assert rejected_total >= 1, (
        "the DistMult gate must visibly reject noise candidates across the 32 control draws"
    )
    # (3) determinism
    assert r_a == r_b, "two seed=0 runs must be identical"
    # (4) TRANSITIVITY FIREWALL -- red constructed above, both directions.
    assert res_nt["kept"] == 0, (
        "must NOT chain 'has': mary/john has hat + hat has brim does not license 'has brim'"
    )
    assert res_tt["kept"] >= 1, (
        "the firewall must reject the RELATION, not the shape: the same triples under 'partof' derive"
    )
    # (5) CONFIDENCE-GATE RED -- the flag must be observable False, and must say so honestly.
    assert res_off["confidence_gate_ran"] is False, (
        "confidence_gate_ran must read False when the kb_ranking import fails"
    )
    assert res["confidence_gate_ran"] is True, "the gate must actually run on the healthy path"
    # (5b) the OTHER dead-gate cause: a scorer that diverged to non-finite, with the import intact.
    assert res_div["confidence_gate_ran"] is False, (
        "a NaN-diverged DistMult must report that the gate did NOT run, not a healthy gate"
    )
    # (6) SIGNATURE COLLISION -- red under the retired encode("ascii", "ignore").
    assert bits_hi != bits_lo, (
        "entities differing only outside ASCII must not share one spike signature"
    )
    # (7) PORTED-WHITELIST DRIFT -- red if either copy is edited alone.
    if _DERIV_TRANSITIVE is not None:
        assert _DERIV_TRANSITIVE == _TRANSITIVE, (
            "ported _TRANSITIVE has drifted from reasoning/derivation.py -- one vocabulary, two copies"
        )
    # DELETED per P8. Every deletion below was RE-CHECKED 2026-08-04 by constructing inputs rather
    # than by reading the code, and the check changed one of the three verdicts:
    #   `res["prototypes"] <= si.max_prototypes` -- _art_cluster only appends under
    #      `len(protos) < self.max_prototypes`, so the cap is a structural invariant of the loop.
    #      Probed at max_prototypes in {0,1,2,3,256} x stores of 50..400 triples: never breached.
    #   `len(res["derived"]) <= 64` -- the `if len(derived) >= budget: break` guards precede every
    #      append, so the budget is likewise structural. Probed at budget in {0,1,2,8,64}: never
    #      breached (it reaches the budget exactly, which is the assert's boundary, not a red).
    #   `res_off["rejected"] == 0` -- DELETED 2026-08-04, and it is the one the rewrite left in.
    #      `rejected` increments only inside `if model is not None and not self._confident(...)`, so
    #      "the gate did not run" (model is None) forces rejected to 0 by construction -- exactly the
    #      shape P8 forbids, asserting a fact about the code path back to itself. Hunted for a red
    #      across 8 stores (taxonomy, non-transitive, control noise, a 301-triple wide taxonomy, a
    #      dedup-heavy store, empty, malformed input) under a blocked kb_ranking import AND under the
    #      NaN-diverged scorer -- all 8 returned rejected=0. Nothing here is lost: (5)'s two
    #      surviving asserts already pin the flag False when the gate is dead and True when it is not,
    #      and both of those DO redden (measured: leave the import intact -> True; block it -> False).
    #   `derived triples must be ASCII` -- deleted, but NOT for the reason first recorded. That reason
    #      said no reachable input could redden it; there is one, and it was constructed: feed a
    #      non-ASCII entity and the derived triple carries it through (`['caf<e-acute>','isa','chose',
    #      'schema']`). The criterion is deleted because it is WRONG, not because it is vacuous -- the
    #      ONE lake is UTF-8 on its binary path (`substrate/binlog.py:196`), and a non-ASCII derived
    #      triple written through `KnowledgeLog.append` and re-read from disk in a fresh log round-
    #      trips byte-identical (measured against a scratch NORD_HOME). CLAUDE.md's ASCII rule governs
    #      `print()`, which this self-check obeys with ASCII literals. Restoring the assert would pin
    #      an invariant the durable path does not actually have.

    print(
        f"self-check OK: induced {res['kept']} confident schema(s) incl. dog/cat isa animal from "
        f"{res['prototypes']} prototypes; control noise -> {resc['kept']} kept (firewall); "
        f"non-transitive 'has' chain refused ({res_nt['kept']}) while 'partof' derives "
        f"({res_tt['kept']}); gate-off observable; deterministic."
    )
    print(f"           run: python -m {__spec__.name if __spec__ else __name__}")


if __name__ == "__main__":
    _selfcheck()
