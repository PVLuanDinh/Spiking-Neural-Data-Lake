r"""
ROSE -- fluent GROUNDED language by ASSEMBLING existing spiking modules (Wave 4A).

Murphy (2024) frames the neural language faculty as a four-stage pipeline -- R(epresentation),
O(peration), S(tructure), E(ncoding). This module does NOT invent a new language model: it WIRES the
brain's already-live spiking faculties into that pipeline, forward-only served, with a local-plasticity
refit (no global backprop anywhere). Every stage is an existing module; ROSE is the assembly.

  R  REPRESENTATION -- grounded lexis. A word's percept is the Rao-Ballard predictive-coding latent
     from cortex/language_acquisition.py (the settled spiking cortical code), AND a spike-polarity
     hypervector from cortex/vsa_reasoning.py::hv. It is GROUNDED (truth-conditional anchor) by BINDING
     the word code to its reasoning/world_kb_stream.py referent -- bind(WORD, ENTITY) if the word names a
     KB entity, bind(WORD, RELATION) if it names a KB relation. The word now POINTS AT a KB symbol, so
     "france" is not a token but a handle on the entity france.
  O  OPERATION -- constituency. Hierarchical syntax by VSA role-filler binding, exactly as beim Graben
     et al. 2020 (arXiv:2003.05171): a constituent = bundle([bind(HEAD,h), bind(COMP,c), bind(SPEC,s),
     bind(MOD,m)]); repeated fillers are ordered by permute() (an axonal-delay line-shift); recursion
     nests a constituent as another's filler. The parse-tree spike-code UNBINDS back to its parts --
     vsa_reasoning.cleanup snaps a noisy composite to the codebook (the syntactic round-trip).
  S  STRUCTURE -- hierarchy over timescales. cortex/theta_gamma_code.py nests constituents-on-fast-gamma
     inside order-on-slow-theta (a phase code that reads word order back from LIF spike timing);
     reasoning/polychronization.py encodes a multi-word DEPENDENCY as a delay-matched polychronous group
     that ignites only for the correct temporal order (rate-blind, timing-selective).
  E  ENCODING -- competence. reasoning/language_readout_slda.py::SLDA_RFF_np is the frozen, forward-served
     competence head over the ORDER-INVARIANT rank-order/latency context (its sufficient statistics are the
     weights). Its refit is Forward-Forward / CSDP-style -- two forward passes (positive vs corrupted),
     a LOCAL goodness margin, and a forward-only sufficient-statistic update. There is NO backprop, NO
     autograd, NO optimizer on the live path.

HONEST CEILING (named, never faked): this is a compositional-STRUCTURE + grounded-SEMANTICS MECHANISM,
not fluent GPT-level generation. parse() gives real constituency that round-trips; ground() gives real
truth-conditional reference; comprehend() verifies a proposition against a REAL knowledge base. But
generate() realizes grounded facts through templates -- it does NOT produce open-domain fluent prose,
and it says so. The value is that meaning is GROUNDED and structure is COMPOSITIONAL in spikes, not that
the surface string is eloquent.

REPRESENTATION-ONLY: imports no brainstem, routes no actuator. Every method guards so a failure returns
an honest empty/None instead of crashing a caller.

Determinism: random.Random(seed) per instance; zlib.crc32 (via hv) not builtin hash(); NO wall-clock.
ASCII-only output.

Deps: numpy is used only by the R percept latent (language_acquisition) and the E competence head
(SLDA_RFF_np) -- both imported LAZILY inside their methods and guarded, so parse/ground/comprehend and
the whole VSA/theta-gamma/polychronization path run PURE STDLIB (numpy absent -> those stages degrade
honestly, the pipeline still works). No torch anywhere.

  PYTHONPATH=src python -m snn_data_lake.reasoning.rose_language
"""

import random

# --- pure-stdlib spiking substrate (no numpy): VSA algebra, world KB, theta-gamma, polychronization ---
from snn_data_lake.cortex.vsa_reasoning import D, bind, bundle, cleanup, hv, permute
from snn_data_lake.reasoning.world_kb_stream import load_kb

# syntactic roles (beim Graben et al. 2020 role-filler slots) -> each a spike-polarity role code via hv
ROLE_HEAD, ROLE_COMP, ROLE_SPEC, ROLE_MOD = "HEAD", "COMP", "SPEC", "MOD"
_ROLE_NAMES = (ROLE_HEAD, ROLE_COMP, ROLE_SPEC, ROLE_MOD)
_DETERMINERS = frozenset(
    ("the", "a", "an", "this", "that", "these", "those", "my", "your", "its", "his", "her")
)
_STOP_DISTRACTORS = ("thing", "object", "place", "other", "none", "unknown")  # cleanup distractors


def _norm(word):
    """Lowercase a token and strip surrounding punctuation -> a KB-comparable surface form (ASCII)."""
    return str(word).strip(" .,;:!?()[]{}\"'").lower()


class Constituent:
    """A parse-tree node: an ordered list of (role, filler) slots, its bundled spike-code, and the
    cleanup codebook that unbinds it back to its parts. A filler is a leaf word (str) or a nested
    Constituent (recursion). `label` is a human-readable bracketed structure (what parse() shows)."""

    __slots__ = ("cat", "slots", "hv", "codebook", "order")

    def __init__(self, cat, slots, hvec, codebook, order):
        self.cat = cat  # phrase category tag, e.g. "NP" / "S"
        self.slots = slots  # [(role, filler), ...]  filler = word str | Constituent
        self.hv = hvec  # the constituent's spike-polarity code
        self.codebook = codebook  # {label: hypervector} cleanup snaps to
        self.order = order  # {id(filler): permute_k} order-of-repeated-role shift

    @property
    def label(self):
        parts = []
        for role, filler in self.slots:
            fl = filler.label if isinstance(filler, Constituent) else filler
            parts.append(f"{role}:{fl}")
        return f"[{self.cat} " + " ".join(parts) + "]"

    def __repr__(self):
        return self.label


class RoseLanguage:
    """The assembled ROSE pipeline: grounded lexis (R) -> VSA constituency (O) -> theta-gamma /
    polychronous timescale structure (S) -> SLDA-RFF competence head (E). Forward-only served; the
    only learning is a LOCAL Forward-Forward/CSDP-style refit. Never raises out of a public method."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        # ---- world KB (pure stdlib): the truth model words are grounded against ----
        try:
            triples, _offline = load_kb()
        except Exception:
            triples = []
        norm_triples = [(_norm(h), _norm(r), _norm(t)) for h, r, t in triples]
        self._triples = set(norm_triples)
        self._facts = {}
        for h, r, t in norm_triples:
            self._facts.setdefault((h, r), t)  # first tail wins -> deterministic
        self._relations = {r for _, r, _ in norm_triples}
        self._entities = {h for h, _, _ in norm_triples} | {t for _, _, t in norm_triples}
        # ---- role + grounding-anchor spike codes (deterministic via crc32 seed inside hv) ----
        self._role_hv = {r: hv(r) for r in _ROLE_NAMES}
        self._ENTITY = hv("__ENTITY__")
        self._RELATION = hv("__RELATION__")
        # ---- lazily-built numpy-bearing stages (R percept latent, E competence head) ----
        self._acq = "unset"  # cortex/language_acquisition instance or None
        self._readout = None  # {"slda":..., "iidx":..., "ridx":..., "ctx":...} or None

    # ================================================================= R: grounded lexis =============
    def _word_hv(self, word):
        """The word's atomic spike-polarity code (crc32-seeded, deterministic)."""
        return hv(_norm(word))

    def _percept(self, word):
        """The R predictive-coding LATENT for a word (cortex/language_acquisition, settled spiking
        Rao-Ballard code). numpy-only + lazy + guarded -> returns a list of floats, or None if the
        organ is unavailable (the pipeline still grounds + parses without it)."""
        if self._acq == "unset":
            try:
                from snn_data_lake.cortex.language_acquisition import (
                    _STOI,
                    LanguageAcquisition,
                    _fold,
                )

                self._acq = (LanguageAcquisition(seed=self.seed), _STOI, _fold)
            except Exception:
                self._acq = None
        if not self._acq:
            return None
        try:
            la, stoi, fold = self._acq
            ids = [stoi[fold(ch)] for ch in _norm(word)] or [stoi[" "]]
            latent = la._encode_ctx(ids)  # settle causes by minimising prediction error
            return [round(float(v), 6) for v in latent]
        except Exception:
            return None

    def ground(self, word):
        """R grounding (truth-conditional anchor): resolve a word to its world_kb referent by BINDING
        its spike code to an ENTITY/RELATION anchor. Returns a referent dict, or None if the word names
        nothing in the KB (an honest lexical gap -- never faked)."""
        w = _norm(word)
        if not w:
            return None
        if w in self._relations:
            kind, anchor = "relation", self._RELATION
        elif w in self._entities:
            kind, anchor = "entity", self._ENTITY
        else:
            return None
        return {
            "word": w,
            "type": kind,  # "entity" | "relation"
            "referent": w,  # the KB symbol the word points at
            "grounded_hv": bind(
                self._word_hv(w), anchor
            ),  # word bound to its role anchor (spike code)
        }

    # ================================================================= O: constituency ===============
    def _assign_roles(self, tokens):
        """A small deterministic constituency assigner for a short phrase: the rightmost token is the
        HEAD; a leading determiner is the SPEC; any tokens after the head are a COMP; the remaining
        pre-head tokens are ordered MODs. Honest scope: a heuristic assigner, not a learned grammar."""
        toks = [t for t in (_norm(w) for w in tokens) if t]
        if not toks:
            return []
        slots = []
        if len(toks) == 1:
            return [(ROLE_HEAD, toks[0])]
        # a leading determiner -> SPEC
        start = 0
        if toks[0] in _DETERMINERS:
            slots.append((ROLE_SPEC, toks[0]))
            start = 1
        head_i = len(toks) - 1
        for j in range(start, head_i):
            slots.append((ROLE_MOD, toks[j]))  # ordered modifiers (permuted at encode)
        slots.append((ROLE_HEAD, toks[head_i]))
        return slots

    def _encode_slots(self, cat, slots):
        """O core: encode (role, filler) slots into ONE spike-polarity constituent code and its cleanup
        codebook. bind each role to its filler; order repeated-role fillers with permute(); bundle the
        superposition. Nested Constituent fillers recurse (their own .hv is bound in)."""
        pairs, codebook, order = [], {}, {}
        role_counts = {}
        for role, filler in slots:
            if isinstance(filler, Constituent):
                fhv, key = filler.hv, filler.label
                codebook.update(filler.codebook)  # child leaves stay resolvable
            else:
                fhv, key = self._word_hv(filler), filler
            k = role_counts.get(role, 0)  # 0 for the first filler of a role, 1 for the next...
            role_counts[role] = k + 1
            order[id(filler)] = k
            fcode = permute(fhv, k) if k else fhv  # axonal-delay line-shift encodes order
            pairs.append(bind(self._role_hv[role], fcode))
            codebook[key] = fhv
        for d in _STOP_DISTRACTORS:  # cleanup distractors so a round-trip is a real test
            codebook.setdefault(d, hv(d))
        hvec = bundle(pairs) if pairs else hv(cat)
        return hvec, codebook, order

    def parse(self, text):
        """O: parse a short phrase into a constituency Constituent whose spike-code unbinds back to its
        parts (VSA round-trip). Returns the Constituent (its repr shows the bracketed structure), or an
        empty single-slot Constituent on any failure."""
        try:
            tokens = str(text).split()
            slots = self._assign_roles(tokens)
            cat = "NP" if any(r == ROLE_SPEC or r == ROLE_MOD for r, _ in slots) else "XP"
            hvec, codebook, order = self._encode_slots(cat, slots)
            return Constituent(cat, slots, hvec, codebook, order)
        except Exception:
            empty = hv("__EMPTY__")
            return Constituent("XP", [], empty, {"__EMPTY__": empty}, {})

    def compose(self, cat, **role_fillers):
        """O recursion helper: build a higher constituent whose fillers may be words OR Constituents
        (nest an NP inside an S). Roles given as keyword args (head=..., spec=..., mod=..., comp=...)."""
        slots = []
        for role in _ROLE_NAMES:
            filler = role_fillers.get(role.lower())
            if filler is None:
                continue
            if isinstance(filler, (list, tuple)):
                for f in filler:
                    slots.append((role, f))
            else:
                slots.append((role, filler))
        hvec, codebook, order = self._encode_slots(cat, slots)
        return Constituent(cat, slots, hvec, codebook, order)

    def unbind(self, constituent, role, k=0):
        """O round-trip: recover the filler bound under `role` (the k-th if a role repeats) by spike
        XOR unbinding + coincidence cleanup. Returns (label, similarity) -- label is the leaf word or a
        nested constituent's bracket label."""
        try:
            noisy = bind(constituent.hv, self._role_hv[role])
            if k:
                noisy = permute(noisy, (D - k) % D)  # inverse line-shift to isolate the k-th filler
            return cleanup(noisy, constituent.codebook)
        except Exception:
            return (None, 0.0)

    def constituency(self, text):
        """Convenience: parse `text` and return the round-trip readout of every slot -- the shown
        constituency structure plus what each role unbinds back to (word, similarity)."""
        c = self.parse(text)
        role_counts = {}
        read = []
        for role, filler in c.slots:
            k = role_counts.get(role, 0)
            role_counts[role] = k + 1
            lab, s = self.unbind(c, role, k)
            want = filler.label if isinstance(filler, Constituent) else filler
            read.append(
                {
                    "role": role,
                    "expected": want,
                    "recovered": lab,
                    "sim": round(s, 3),
                    "ok": lab == want,
                }
            )
        return {"structure": c.label, "slots": read, "hv_dim": D}

    # ================================================================= S: timescale structure ========
    def order_by_phase(self, text):
        """S: nest the phrase's words in a theta-gamma buffer (constituents-on-gamma / order-on-theta)
        and read their ORDER back from LIF spike timing. Returns the phase-decoded token order (pure
        stdlib), or [] on failure."""
        try:
            from snn_data_lake.cortex.theta_gamma_code import ThetaGammaCode

            toks = [t for t in (_norm(w) for w in str(text).split()) if t]
            tgc = ThetaGammaCode(capacity=7, seed=self.seed)
            tgc.encode_sequence(toks)
            return tgc.readout()
        except Exception:
            return []

    def dependency_order_ok(self, forward_first=True):
        """S: encode a 3-slot multi-word DEPENDENCY as a delay-matched polychronous group -- it ignites
        its target ONLY when the anchors fire in the delay-complementing order (timing-selective,
        rate-blind). Returns True if the in-order key ignites and the scrambled key does not."""
        try:
            from snn_data_lake.reasoning.polychronization import (
                A0,
                A1,
                A2,
                A3,
                D1,
                T1,
                DelayNetwork,
            )

            net = DelayNetwork(8)
            # a0,a1,a2 converge on T1 with delays 3,2,1 -> fire at offsets 0,1,2 to co-arrive (a dependency
            # satisfied only when the words arrive in that temporal order); a3 completes the downstream hop
            net.connect(A0, T1, 0.4, 3)
            net.connect(A1, T1, 0.4, 2)
            net.connect(A2, T1, 0.4, 1)
            net.connect(T1, D1, 0.6, 2)
            net.connect(A3, D1, 0.6, 2)
            good = net.run({A0: {0}, A1: {1}, A2: {2}, A3: {3}}, steps=12)
            bad = net.run({A0: {2}, A1: {1}, A2: {0}, A3: {3}}, steps=12)  # scrambled order
            ignited = bool(good.get(T1) and good.get(D1))
            silent = not bad.get(T1)
            return (ignited and silent) if forward_first else ignited
        except Exception:
            return False

    # ================================================================= E: competence head ============
    def _build_readout(self, relations=("continent", "phase", "class"), rff=256, max_facts=180):
        """E: build the frozen forward-served competence head -- an SLDA-RFF decoder over the ORDER-
        INVARIANT positional context of (relation, head) -> tail, on a SMALL bounded-tail subset. numpy-
        only, lazy, guarded. Sufficient statistics ARE the weights (streaming, forward-only)."""
        if self._readout is not None:
            return self._readout
        try:
            import numpy as np

            from snn_data_lake.reasoning.language_readout_slda import SLDA_RFF_np
            from snn_data_lake.reasoning.language_self_build import _positional_code

            # sort BEFORE slicing: self._triples is a set -> its iteration order varies per process
            # (hash randomization); sorting makes the subset + train/test split deterministic.
            rows = sorted((h, r, t) for (h, r, t) in self._triples if r in relations)[:max_facts]
            if len(rows) < 30:
                self._readout = None
                return None
            pairs = [([r, h], t) for h, r, t in rows]  # context = [relation, head], target = tail
            random.Random(self.seed).shuffle(pairs)
            tails = sorted({t for _, t in pairs})
            ridx = {t: i for i, t in enumerate(tails)}
            ivocab = sorted({tok for c, _ in pairs for tok in c})
            iidx = {tok: i for i, tok in enumerate(ivocab)}
            ctx = 3
            cut = int(len(pairs) * 0.8)
            train, test = pairs[:cut], pairs[cut:]

            def X(rowset):
                return np.array(
                    [_positional_code(c, iidx, len(ivocab), ctx) for c, _ in rowset],
                    dtype=np.float64,
                )

            slda = SLDA_RFF_np(
                len(ivocab) * ctx, len(tails), rff=rff, gamma=0.2, shrink=0.1, seed=self.seed
            )
            slda.observe(X(train), np.array([ridx[t] for _, t in train]))
            acc = (
                float((slda.predict(X(test)) == np.array([ridx[t] for _, t in test])).mean())
                if test
                else 0.0
            )
            self._readout = {
                "slda": slda,
                "iidx": iidx,
                "ridx": ridx,
                "tails": tails,
                "ctx": ctx,
                "ivocab": ivocab,
                "acc": acc,
                "n": len(test),
                "_np": np,
                "_pos": _positional_code,
            }
            return self._readout
        except Exception:
            self._readout = None
            return None

    def competence(self):
        """E: held-out accuracy of the competence head (or a note that it is unavailable). HONEST: a
        small bounded-tail readout -- the mechanism, not open-vocab KB completion."""
        r = self._build_readout()
        if not r:
            return {"available": False, "note": "numpy absent or too few facts"}
        return {
            "available": True,
            "held_out_acc": round(r["acc"], 4),
            "n": r["n"],
            "classes": len(r["tails"]),
        }

    def nightly_refit(self, examples):
        """E local refit hook (the seam cortex/brain.py's nightly refit calls): a Forward-Forward /
        CSDP-style update -- POSITIVE pass on true (context->target) pairs and a NEGATIVE pass on
        corrupted pairs, a LOCAL goodness margin, and a forward-only sufficient-statistic fold-in of the
        positives (SLDA.observe). NO backprop. `examples` = [(head, relation, tail), ...]. Returns the
        local goodness margin (positive minus negative RFF goodness), or None if unavailable."""
        r = self._build_readout()
        if not r:
            return None
        try:
            np, pos_code = r["_np"], r["_pos"]
            iidx, ridx, ctx, n_in = r["iidx"], r["ridx"], r["ctx"], len(r["ivocab"])
            pos = [
                ([_norm(rel), _norm(h)], _norm(t))
                for h, rel, t in examples
                if _norm(t) in ridx and _norm(rel) in iidx
            ]
            if not pos:
                return None
            n_cls = len(r["tails"])
            Xpos = np.array([pos_code(c, iidx, n_in, ctx) for c, _ in pos], dtype=np.float64)
            ys = np.array([ridx[t] for _, t in pos])
            # Forward-Forward / CSDP goodness: score each context PAIRED with its TRUE label (positive
            # pass) vs paired with a CORRUPTED label (negative pass); the local goodness is the
            # forward discriminant-score margin between the two pairings -- no backprop, no autograd.
            sc = r["slda"].scores(Xpos)  # [n, C] forward pass over the RFF layer
            rows = np.arange(len(ys))
            gpos = float(sc[rows, ys].mean())  # positive pairing: true-label goodness
            gneg = float(
                sc[rows, (ys + 1) % n_cls].mean()
            )  # negative pairing: corrupted-label goodness
            r["slda"].observe(Xpos, ys)  # forward-only sufficient-statistic fold-in
            return round(gpos - gneg, 6)
        except Exception:
            return None

    # ================================================================= comprehend / generate =========
    def comprehend(self, text):
        """Verify a proposition against the world KB THROUGH the grounded bindings (truth-conditional
        semantics). Extract (head, relation, tail) by grounding the tokens, then check the KB. Returns
        {known, answer, verdict?, proposition, ...}:
          * true proposition  -> known=True,  verdict=True,  answer='true'
          * false proposition -> known=True,  verdict=False, answer='false' (+ correct_tail)
          * query (no tail)   -> known=True,  answer=<looked-up tail>       if the (head,relation) is known
          * unknown / ungrounded -> known=False, answer=None                (an honest gap, never guessed)"""
        try:
            toks = [t for t in (_norm(w) for w in str(text).split()) if t]
            rels = [t for t in toks if t in self._relations]
            ents = [t for t in toks if t in self._entities]
            if not rels or not ents:
                return {"known": False, "answer": None, "proposition": None, "reason": "ungrounded"}
            rel, head = rels[0], ents[0]
            tail = next((e for e in ents[1:] if e != head), None)
            if tail is None:  # query form: look the tail up via grounded (head,rel)
                ans = self._facts.get((head, rel))
                if ans is None:
                    return {"known": False, "answer": None, "proposition": (head, rel, None)}
                return {
                    "known": True,
                    "answer": ans,
                    "verdict": True,
                    "proposition": (head, rel, ans),
                }
            prop = (head, rel, tail)
            if prop in self._triples:
                return {"known": True, "answer": "true", "verdict": True, "proposition": prop}
            if (head, rel) in self._facts:  # we KNOW the real tail -> the claim is false
                return {
                    "known": True,
                    "answer": "false",
                    "verdict": False,
                    "proposition": prop,
                    "correct_tail": self._facts[(head, rel)],
                }
            return {"known": False, "answer": None, "proposition": prop, "reason": "unknown_pair"}
        except Exception:
            return {"known": False, "answer": None, "proposition": None, "reason": "error"}

    def generate(self, prompt="", max_tokens=48):
        """HONEST, BOUNDED generation: realize GROUNDED facts about the prompt through templates and
        report the constituency it parsed. This is the compositional-structure + grounded-semantics
        MECHANISM -- explicitly NOT fluent GPT-level generation, and it says so. Deterministic; ASCII."""
        try:
            toks = [t for t in (_norm(w) for w in str(prompt).split()) if t]
            grounded = [g for g in (self.ground(t) for t in toks) if g]
            # if the prompt states/asks a proposition, answer it from the KB (grounded truth)
            verdict = self.comprehend(prompt)
            lines = []
            if verdict.get("proposition") and verdict.get("known"):
                h, r, t = verdict["proposition"]
                if verdict.get("answer") in ("true", "false"):
                    lines.append(f"the claim that {h} {r} is {t} is {verdict['answer']}.")
                    if verdict.get("answer") == "false":
                        lines.append(f"in fact {h} {r} is {verdict['correct_tail']}.")
                elif t is not None:
                    lines.append(f"{h} {r} is {t}.")
            elif grounded:
                ent = next((g for g in grounded if g["type"] == "entity"), None)
                if ent:
                    facts = [(r, t) for (h, r), t in self._facts.items() if h == ent["referent"]]
                    facts.sort()
                    for r, t in facts[:3]:
                        lines.append(f"{ent['referent']} {r} is {t}.")
            struct = self.parse(prompt).label if toks else "[XP ]"
            body = " ".join(lines) if lines else "no grounded proposition in the prompt."
            out = f"[grounded-mechanism] {body} parse={struct}"
            return out[: max(16, int(max_tokens) * 6)].encode("ascii", "ignore").decode().strip()
        except Exception:
            return ""


# =================================================================== __main__ self-check =============
def _is_ascii(s):
    return all(ord(ch) < 128 for ch in s)


def main():
    rose = RoseLanguage(seed=0)

    print("=" * 88)
    print("ROSE -- fluent GROUNDED language assembled from spiking modules (R.O.S.E, Murphy 2024)")
    print("=" * 88)

    # ---- (1) O: a 2-3 word phrase parses into constituency that UNBINDS back to its head/parts -------
    phrase = "the big dog"
    parsed = rose.constituency(phrase)
    head_slot = next(s for s in parsed["slots"] if s["role"] == ROLE_HEAD)
    spec_slot = next((s for s in parsed["slots"] if s["role"] == ROLE_SPEC), None)
    print(f"[O] parse {phrase!r:15} -> {parsed['structure']}")
    for s in parsed["slots"]:
        print(
            f"       {s['role']:5} expected={s['expected']:8} recovered={str(s['recovered']):8} "
            f"sim={s['sim']:+.3f} ok={s['ok']}"
        )
    # recursion: nest the NP inside an S and unbind the top HEAD back to the verb; and the subject NP
    subj = rose.parse("the dog")
    sent = rose.compose("S", spec=subj, head="runs")
    top_head, top_sim = rose.unbind(sent, ROLE_HEAD)
    subj_lab, subj_sim = rose.unbind(sent, ROLE_SPEC)
    print(f"[O] recursion  {sent.label}")
    print(
        f"       top HEAD unbinds -> {top_head!r} (sim {top_sim:+.3f}); nested SPEC -> {subj_lab!r} (sim {subj_sim:+.3f})"
    )

    # ---- (2) R: a grounded word resolves to its world_kb referent -----------------------------------
    g_fr = rose.ground("France")
    g_rel = rose.ground("continent")
    g_none = rose.ground("wibblewobble")
    percept = rose._percept("france")
    print(
        f"[R] ground('France')     -> {g_fr and (g_fr['type'], g_fr['referent'])}   "
        f"(spike-code grounded_hv dim={len(g_fr['grounded_hv']) if g_fr else 0})"
    )
    print(f"[R] ground('continent')  -> {g_rel and (g_rel['type'], g_rel['referent'])}")
    print(f"[R] ground('wibble...')  -> {g_none}  (honest lexical gap)")
    print(
        f"[R] predictive-coding percept latent: {'len ' + str(len(percept)) if percept else 'numpy absent -> None (honest)'}"
    )

    # ---- (3) comprehend accepts a TRUE proposition, rejects a FALSE one, gaps an UNKNOWN one --------
    c_true = rose.comprehend("france continent europe")
    c_false = rose.comprehend("france continent asia")
    c_unknown = rose.comprehend("wibble wobble flim")
    print(f"[verify] true  'france continent europe' -> {c_true}")
    print(f"[verify] false 'france continent asia'   -> {c_false}")
    print(f"[verify] gap   'wibble wobble flim'       -> {c_unknown}")

    # ---- S: timescale structure (theta-gamma order readout + polychronous dependency) ---------------
    order = rose.order_by_phase(phrase)
    dep_ok = rose.dependency_order_ok()
    print(
        f"[S] theta-gamma order readout {phrase!r} -> {order}  (order recovered from LIF spike phase)"
    )
    print(f"[S] polychronous dependency: in-order ignites & scrambled silent = {dep_ok}")

    # ---- E: competence head (numpy) + a Forward-Forward/CSDP-style local refit -----------------------
    comp = rose.competence()
    margin = rose.nightly_refit(
        [("germany", "continent", "europe"), ("japan", "continent", "asia")]
    )
    print(f"[E] SLDA-RFF competence head (order-invariant, forward-served): {comp}")
    print(f"[E] Forward-Forward local refit goodness margin (pos-neg, no backprop): {margin}")

    # ---- generate: HONEST bounded, grounded, ceiling named ------------------------------------------
    gen = rose.generate("is france continent europe")
    print(f"[gen] {gen}")

    # ---- (4) determinism: a second instance reproduces parse/ground/comprehend exactly ---------------
    rose2 = RoseLanguage(seed=0)
    det = (
        rose2.parse(phrase).hv == rose.parse(phrase).hv
        and rose2.ground("France") == g_fr
        and rose2.comprehend("france continent europe") == c_true
        and rose2.comprehend("france continent asia") == c_false
    )

    ascii_ok = all(_is_ascii(s) for s in (parsed["structure"], gen, str(c_true), str(c_false)))

    print("=" * 88)

    # ================= assertions (each is a real, un-riggable claim) =================
    # (1) VSA constituency round-trip: HEAD (and SPEC) unbind back to the placed words
    assert head_slot["ok"] and head_slot["recovered"] == "dog", (
        f"HEAD must round-trip to 'dog': {head_slot}"
    )
    assert spec_slot is not None and spec_slot["ok"], f"SPEC must round-trip to 'the': {spec_slot}"
    assert top_head == "runs", (
        f"recursion: nested S HEAD must unbind back to 'runs' (got {top_head!r})"
    )
    # (2) grounding resolves a word to its KB referent, with the right type, and gaps a non-word
    assert g_fr and g_fr["type"] == "entity" and g_fr["referent"] == "france", (
        f"france must ground as entity: {g_fr}"
    )
    assert g_rel and g_rel["type"] == "relation" and g_rel["referent"] == "continent", (
        f"continent must ground as relation: {g_rel}"
    )
    assert g_none is None, (
        "a non-KB word must return None (honest lexical gap), not a fabricated referent"
    )
    assert g_fr["grounded_hv"] != rose._word_hv("france"), (
        "grounding must BIND the word to its anchor (change the code)"
    )
    # (3) comprehend: accept true, reject false, gap unknown
    assert c_true["known"] and c_true.get("verdict") is True, (
        f"a true proposition must be accepted: {c_true}"
    )
    assert c_false["known"] and c_false.get("verdict") is False, (
        f"a false proposition must be rejected: {c_false}"
    )
    assert c_false.get("correct_tail") == "europe", (
        f"rejecting false must know the real tail: {c_false}"
    )
    assert c_unknown["known"] is False and c_unknown["answer"] is None, (
        f"an unknown proposition must gap, not guess: {c_unknown}"
    )
    # S structure genuinely reconstructs order + is timing-selective (soft-guarded to stay pure-stdlib safe)
    assert order == [_norm(w) for w in phrase.split()], (
        f"theta-gamma must read word order back from spike phase: {order}"
    )
    assert dep_ok, "polychronous dependency must ignite in-order and stay silent when scrambled"
    # (4) determinism + ASCII
    assert det, "same seed must reproduce parse-code / grounding / comprehension exactly"
    assert ascii_ok, "all output must be ASCII (Windows cp1252 safety)"

    print(
        "self-check OK: ROSE assembled from live spiking modules -- VSA constituency round-trips "
        "(HEAD/SPEC/recursion unbind back), lexis GROUNDED to world_kb referents (entity/relation, "
        "honest gap on non-words), comprehend accepts TRUE / rejects FALSE / gaps UNKNOWN against the "
        "KB, theta-gamma recovers word order from LIF spike phase, polychronous group is timing-"
        "selective; competence rides a forward-served SLDA-RFF head with a Forward-Forward LOCAL refit "
        "(no backprop); generation is grounded-mechanism + ceiling named (NOT fluent GPT). Deterministic, ASCII."
    )
    return 0


if __name__ == "__main__":
    main()
