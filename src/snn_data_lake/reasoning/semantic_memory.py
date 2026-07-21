"""
SEMANTIC memory recall -- meaning-based retrieval over VSA spike-polarity codes.

THE GAP (honest): every recall path the brain has is LEXICAL. spike_chat_lake._lake_recall gates a
hit on ">=1 shared content word with the query" (a crc32 bag-of-tokens signature ranks candidates,
but the GATE is shared surface words), and _retrieve ranks by cue/query word coverage. So a
paraphrase with different words retrieves NOTHING: store "paris is france's capital", ask with a
cue that shares zero content words, and the lexical gate provably returns [] -- zero semantic
generalization.

THIS MODULE: a SemanticStore that encodes each remembered fact as a HIGH-DIMENSIONAL vector built
on the same substrate conventions as cortex/vsa_reasoning.py -- deterministic crc32-seeded bipolar
spike-polarity ATOMS (one spike per dimension on an up(+1)/down(-1) line), superposed (bundled)
into a fact code -- and recalls by VSA cosine (centred spike coincidence), not by shared words.

Meaning comes from LOCAL HEBBIAN LEARNING, two forms, both gradient-free:
  * learn_cooccurrence(sentences) -- RANDOM INDEXING (Kanerva/Sahlgren): each token accumulates
    the atomic index codes of the tokens it co-occurs with (a per-synapse Hebbian counter). Two
    words used in the same contexts (doctor/physician, healed/cured) accumulate the same context
    mass and their semantic codes converge -- distributional synonymy, learned, not hand-coded.
  * equate(a, b) -- explicit Hebbian pairing (one strong co-occurrence event), for taught synonyms.

A token's SEMANTIC code = its atomic spike code + alpha * (normalised Hebbian context mass).
A fact/cue code = superposition (sum of unit semantic codes) of its content tokens, plus an
optional role-filler component: when a copula subject parses ("paris IS ..."), the subject's
semantic code is BOUND to a fixed ROLE_SUBJECT spike code by the per-dimension polarity mask
(the same self-inverse spike-XOR bind as vsa_reasoning, applied as a sign mask).

recall(cue, k) ranks stored facts by cosine of these codes -> "the capital of france" retrieves a
fact stored as "paris is france's capital", and after Hebbian learning a cue sharing ZERO content
words ("the doctor cured the peasant") still retrieves "the physician healed the farmer" -- the
case the lexical gate provably misses. lexical_recall() reimplements that shared-word gate as the
in-file baseline so the self-check can show it returning nothing on the same cue.

HONEST LIMITS: meaning is bounded to the encoded vocabulary -- a synonym is only near its partner
if the pair co-occurred in interchangeable contexts (or was equated); there is no morphology
("freeze" and "freezes" are unrelated atoms), no world knowledge, and this is NOT language
understanding. The learned context mass is a graded (rate-like) Hebbian accumulator over discrete
spike-code atoms, not itself a spike train. No gradients anywhere: no backward, no BPTT, so no
e-prop-vs-BPTT cosine applies.

  Run: PYTHONPATH=src python -m snn_data_lake.reasoning.semantic_memory
Determinism: per-instance random.Random / zlib.crc32 only, no import-time work.
"""

import random
import re
import zlib

# mirror of spike_chat_lake._STOP (verbatim) so the lexical-gate baseline here is a faithful
# stand-in for _lake_recall's honesty gate, and semantic encoding keys on the same content words.
_STOP = {
    "what",
    "is",
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "are",
    "you",
    "do",
    "does",
    "i",
    "me",
    "my",
    "tell",
    "about",
    "was",
    "were",
    "who",
    "which",
    "how",
    "and",
    "it",
    "its",
    "for",
    "with",
    "that",
    "this",
    "there",
    "s",
}

_COPULAS = ("is", "are", "was", "were", "means")


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _content(tokens):
    return [t for t in tokens if t not in _STOP]


class SemanticStore:
    """Meaning-based fact memory: remember(text) / recall(cue, k) ranked by VSA cosine.

    dim         spike-code dimensionality (each dim = one spike on the up/down line)
    seed        determinism seed folded into every atomic code (crc32-seeded per token)
    alpha       weight of the learned Hebbian context mass vs the atomic code (0 = lexical atoms only)
    min_sim     recall threshold: below this cosine a fact is NOT returned (honest gap, ~chance guard)
    role_weight weight of the ROLE_SUBJECT-bound component when a copula subject parses (0 disables)
    """

    def __init__(self, dim=2048, seed=0, alpha=1.0, min_sim=0.10, role_weight=0.35):
        self.dim = dim
        self.seed = seed
        self.alpha = alpha
        self.min_sim = min_sim
        self.role_weight = role_weight
        self._idx_cache = {}  # token -> atomic bipolar spike code
        self._ctx = {}  # token -> Hebbian co-occurrence accumulator (graded synaptic counters)
        self._epoch = 0  # bumped on every learning event -> invalidates semantic/fact caches
        self._sem_cache = {}  # token -> (epoch, semantic code)
        self._facts = []  # fact_id -> (text, content tokens, subject token or None)
        self._fact_cache = {}  # fact_id -> (epoch, fact code)
        self._role_subj = self._index("__ROLE_SUBJECT__")

    # ---------------- substrate: atomic spike codes + Hebbian context ----------------

    def _index(self, token):
        """Deterministic atomic spike-polarity code for a token (crc32-seeded, one spike/dim)."""
        v = self._idx_cache.get(token)
        if v is None:
            rng = random.Random(zlib.crc32(f"{self.seed}|{token}".encode()))
            v = [rng.choice((-1.0, 1.0)) for _ in range(self.dim)]
            self._idx_cache[token] = v
        return v

    def _ctx_add(self, token, vec, w=1.0):
        acc = self._ctx.get(token)
        if acc is None:
            acc = [0.0] * self.dim
            self._ctx[token] = acc
        for i, x in enumerate(vec):
            acc[i] += w * x

    def _sem(self, token):
        """Semantic code = atomic code + alpha * sqrt(D) * unit(Hebbian context mass)."""
        hit = self._sem_cache.get(token)
        if hit is not None and hit[0] == self._epoch:
            return hit[1]
        v = list(self._index(token))
        ctx = self._ctx.get(token)
        if ctx is not None:
            n = sum(x * x for x in ctx) ** 0.5
            if n > 0.0:
                s = self.alpha * (self.dim**0.5) / n
                v = [a + s * b for a, b in zip(v, ctx)]
        self._sem_cache[token] = (self._epoch, v)
        return v

    @staticmethod
    def _unit(v):
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n else v

    @staticmethod
    def _cos(a, b):
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    # ---------------- learning (local, Hebbian, gradient-free) ----------------

    def learn_cooccurrence(self, sentences):
        """Random-indexing distributional learning: within each sentence, every content token
        accumulates the ATOMIC codes of its co-occurring content tokens (whole-sentence window).
        Purely local Hebbian counting -- tokens sharing contexts converge in code space."""
        for s in sentences:
            toks = _content(_tok(s))
            for i, t in enumerate(toks):
                for j, u in enumerate(toks):
                    if i != j:
                        self._ctx_add(t, self._index(u))
        self._epoch += 1

    def equate(self, a, b, strength=5.0):
        """Explicit synonym teaching: one strong Hebbian pairing event each way. `strength` is the
        weight relative to a single corpus co-occurrence (default outweighs incidental counts)."""
        a, b = a.lower(), b.lower()
        self._ctx_add(a, self._index(b), strength)
        self._ctx_add(b, self._index(a), strength)
        self._epoch += 1

    # ---------------- memory: remember / encode / recall ----------------

    @staticmethod
    def _subject(tokens):
        """Content token just before the first copula ('paris is ...' -> paris), else None."""
        for cop in _COPULAS:
            if cop in tokens:
                pre = _content(tokens[: tokens.index(cop)])
                if pre:
                    return pre[-1]
                return None
        return None

    def remember(self, text):
        """Store one fact/sentence; returns its fact_id (insertion order)."""
        toks = _tok(text)
        self._facts.append((text, _content(toks), self._subject(toks)))
        return len(self._facts) - 1

    def _encode(self, ctoks, subj):
        v = [0.0] * self.dim
        for t in ctoks:
            u = self._unit(self._sem(t))
            for i, x in enumerate(u):
                v[i] += x
        if subj is not None and self.role_weight:
            u = self._unit(self._sem(subj))
            r = self._role_subj
            w = self.role_weight
            for i, x in enumerate(u):
                v[i] += w * r[i] * x  # role-filler bind: spike-polarity mask (spike XOR)
        return v

    def encode(self, text):
        """Meaning code for arbitrary text (superposition of semantic codes + subject role)."""
        toks = _tok(text)
        return self._encode(_content(toks), self._subject(toks))

    def _fact_vec(self, fid):
        hit = self._fact_cache.get(fid)
        if hit is not None and hit[0] == self._epoch:
            return hit[1]
        _, ctoks, subj = self._facts[fid]
        v = self._encode(ctoks, subj)
        self._fact_cache[fid] = (self._epoch, v)
        return v

    def recall(self, cue, k=3, min_sim=None):
        """Top-k stored facts by MEANING (VSA cosine), best first: [(fact_text, cosine), ...].
        Facts below min_sim are withheld -- an unrelated cue returns [] (honest gap)."""
        th = self.min_sim if min_sim is None else min_sim
        q = self.encode(cue)
        scored = sorted(
            ((self._cos(q, self._fact_vec(i)), i) for i in range(len(self._facts))),
            key=lambda p: (-p[0], p[1]),
        )
        return [(self._facts[i][0], round(c, 4)) for c, i in scored[:k] if c >= th]

    def lexical_recall(self, cue, k=3):
        """BASELINE: the shared-content-word gate (the property _lake_recall/_retrieve gate on).
        Returns [(fact_text, shared_word_count), ...] -- provably [] on a zero-overlap paraphrase."""
        qc = set(_content(_tok(cue)))
        hits = []
        for i, (text, ctoks, _) in enumerate(self._facts):
            n = len(qc & set(ctoks))
            if n:
                hits.append((n, -i, text))
        hits.sort(reverse=True)
        return [(t, n) for n, _, t in hits[:k]]

    def sim_tokens(self, a, b):
        """Cosine between two tokens' semantic codes (introspection)."""
        return self._cos(self._sem(a.lower()), self._sem(b.lower()))


# ---------------------------- self-check ----------------------------

_FACTS = [
    "the physician healed the farmer",  # 0  <- hard-paraphrase target (distributional)
    "paris is france's capital",  # 1  <- hard-paraphrase target (equate)
    "water freezes at zero degrees",  # 2
    "the mitochondria powers the cell",  # 3
    "honey is made by bees",  # 4
    "the violin belongs to the string family",  # 5
    "steel is an alloy of iron and carbon",  # 6
    "the sahara is the largest hot desert",  # 7
]

# distributional corpus: each synonym pair appears in interchangeable frames (that INTERCHANGE is
# the only evidence the Hebbian rule gets -- nothing is hand-labelled as a synonym).
_CORPUS = [
    "the physician examined the sick child",
    "the doctor examined the sick child",
    "a physician prescribes strong medicine",
    "a doctor prescribes strong medicine",
    "the physician works at the village hospital",
    "the doctor works at the village hospital",
    "the nurse healed the wounded soldier",
    "the nurse cured the wounded soldier",
    "rest healed the broken ankle",
    "rest cured the broken ankle",
    "time healed the deep scar",
    "time cured the deep scar",
    "the farmer plowed the muddy field",
    "the peasant plowed the muddy field",
    "the farmer harvested golden wheat",
    "the peasant harvested golden wheat",
    "the farmer fed the hungry oxen",
    "the peasant fed the hungry oxen",
]


def _build():
    store = SemanticStore(seed=0)
    for f in _FACTS:
        store.remember(f)
    store.learn_cooccurrence(_CORPUS)
    store.equate("capital", "seat")
    store.equate("france", "french")
    return store


def main():
    print("=" * 66)
    print("SEMANTIC recall (VSA meaning codes) vs the LEXICAL shared-word gate")
    print("=" * 66)
    store = _build()
    print(f"substrate: crc32-seeded bipolar spike-polarity atoms, D={store.dim}")
    print(
        "learning : local Hebbian co-occurrence (random indexing) + Hebbian pairing; NO gradients"
    )
    print(
        f"store    : {len(_FACTS)} facts | corpus {len(_CORPUS)} sentences | equated capital~seat, france~french"
    )

    # ---- [1] learned token geometry: synonyms converge, unrelated words stay orthogonal ----
    pairs = (("doctor", "physician"), ("cured", "healed"), ("peasant", "farmer"))
    syn = {f"{a}~{b}": store.sim_tokens(a, b) for a, b in pairs}
    unrel = store.sim_tokens("doctor", "volcano")
    print(
        "\n[1] token geometry (cosine): "
        + "  ".join(f"{k}={v:+.2f}" for k, v in syn.items())
        + f"  doctor~volcano={unrel:+.2f}"
    )
    for k, v in syn.items():
        assert v > 0.35, f"distributional synonyms must converge: {k}={v:.2f}"
    assert abs(unrel) < 0.10, f"unrelated tokens must stay ~orthogonal: {unrel:.2f}"

    # ---- [2] HARD paraphrase: ZERO shared content words -- the lexical gate provably misses ----
    hard = [
        ("the doctor cured the peasant", 0),  # doctor/cured/peasant vs physician/healed/farmer
        ("the french seat of government", 1),
    ]  # french/seat/government vs paris/france/capital
    print("\n[2] HARD paraphrase (zero shared content words):")
    hard_ok, hard_cos = 0, []
    for cue, tgt in hard:
        lex = store.lexical_recall(cue)
        assert lex == [], f"hard cue must share no content word with ANY fact: {lex}"
        top2 = store.recall(cue, k=2, min_sim=-1.0)
        top_text, top_cos = top2[0]
        margin = top_cos - (top2[1][1] if len(top2) > 1 else 0.0)
        ok = top_text == _FACTS[tgt] and top_cos >= store.min_sim
        hard_ok += ok
        hard_cos.append(top_cos)
        print(f"    cue '{cue}'")
        print("      lexical gate    -> []  (no shared word: crc32 bag baseline misses)")
        print(
            f"      semantic recall -> '{top_text}'  cos={top_cos:.2f} margin=+{margin:.2f}  "
            f"[{'OK' if ok else 'MISS'}]"
        )
        assert margin > 0.15, f"winner must clearly beat the runner-up: margin={margin:.2f}"
    assert hard_ok == len(hard), f"hard-paraphrase top-1 {hard_ok}/{len(hard)}"

    # ---- [3] paraphrase ranking sanity (shared words allowed) ----
    easy = [
        ("the capital of france", 1),
        ("who healed the farmer", 0),
        ("what powers the cell", 3),
        ("which desert is the largest", 7),
    ]
    easy_ok = 0
    for cue, tgt in easy:
        got = store.recall(cue, k=1)
        easy_ok += bool(got) and got[0][0] == _FACTS[tgt]
    print(f"\n[3] easy-paraphrase ranking: top-1 {easy_ok}/{len(easy)}")
    assert easy_ok == len(easy)

    # ---- [4] controls that MUST fail / sit at chance ----
    # (a) IDENTITY control: same store, same facts, NO Hebbian learning -> paraphrase at chance.
    bare = SemanticStore(seed=0)
    for f in _FACTS:
        bare.remember(f)
    c_un = bare._cos(bare.encode(hard[0][0]), bare._fact_vec(0))
    assert abs(c_un) < bare.min_sim, f"untrained store must sit at chance: {c_un:.3f}"
    assert bare.recall(hard[0][0]) == [], "untrained store must return nothing (below gate)"
    # (b) RANDOM-CUE control: gibberish cues score ~0 and do not pass the recall gate.
    rng = random.Random(1)
    bests = []
    for _ in range(50):
        cue = " ".join(
            "".join(rng.choice("bcdfghjklmnpqrstvwz") for _ in range(7)) for _ in range(3)
        )
        bests.append(store.recall(cue, k=1, min_sim=-1.0)[0][1])
    mean_b = sum(bests) / len(bests)
    max_b = max(bests)
    alarms = sum(1 for b in bests if b >= store.min_sim)
    assert mean_b < store.min_sim and alarms <= 2, (
        f"random cues must sit at chance: mean={mean_b:.3f}"
    )
    # (c) HONEST GAP: an in-format but truly-unrelated cue returns nothing.
    gap = store.recall("the volcano erupted last night")
    assert gap == [], f"unrelated cue must be an honest gap: {gap}"
    print(
        f"[4] controls: untrained cos={c_un:+.3f} -> recall []  |  "
        f"random cues (n=50) best-cos mean={mean_b:.3f} max={max_b:.3f} gate-passes={alarms}/50  |  "
        f"unrelated cue -> [] (honest gap)"
    )

    # ---- [5] determinism: an identically-built store reproduces identical cosines ----
    store2 = _build()
    same = all(
        store2.recall(c, k=2, min_sim=-1.0) == store.recall(c, k=2, min_sim=-1.0) for c, _ in hard
    )
    print(f"[5] determinism: rebuilt store reproduces identical recalls = {same}")
    assert same

    print(
        f"\nmetric: hard-paraphrase (zero-overlap) top-1 = {hard_ok}/{len(hard)} "
        f"(cos {hard_cos[0]:.2f}, {hard_cos[1]:.2f}) vs lexical baseline 0/{len(hard)} "
        f"and untrained control cos {c_un:+.3f} (~chance)"
    )
    print("self-check OK: meaning-based recall retrieves paraphrases the lexical shared-word gate")
    print("provably misses; unrelated and untrained controls sit at chance and return nothing.")
    print("HONEST: bounded to the encoded vocabulary + taught/co-occurred synonyms; no morphology;")
    print(
        "not language understanding. Learned mass is a graded Hebbian accumulator over spike atoms."
    )


if __name__ == "__main__":
    main()
