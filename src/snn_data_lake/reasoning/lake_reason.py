r"""
LAKE REASON -- CONTRADICTION-CORRECTION over the chat lake's stored triples.

HONEST SCOPE (this is the point, read it before trusting the output): this is NOT a general
reasoner and does NOT "think". It is a narrow CONTRADICTION-CORRECTION check over facts the brain
has ALREADY STORED as (subject, relation, value) triples. It does exactly one thing:

  parse an input as a CLAIM  "<subject> ... <relation-verb> ... <value>"
  -> look up a stored triple with the SAME (subject, relation)
  -> if the claimed value is a RECOGNIZED OPPOSITE of the stored value (a small, explicit
     antonym/direction table -- east<->west, up<->down, rises<->sets, hot<->cold, ...),
     reply with the correction, appending a stored CAUSAL fact as the reason IF one is linked.
  -> otherwise stay SILENT (return None).

It returns None -- no correction, no false confidence, no invented reason -- whenever:
  * the subject or relation is not in the lake (unknown -> we do not know it is wrong),
  * the claim AGREES with the stored value,
  * the claimed value merely DIFFERS but is not a recognized opposite (could be a synonym / new info),
  * the input is not a parseable claim.

It never fabricates a fact and never fabricates a reason: the reason is only attached when an
actual stored causal triple links the two opposing values; with no such fact it gives the plain
correction. The logic is fully GENERAL over whatever triples are stored -- the sun/earth example
only works because those two facts are present in the lake, nothing about them is hardcoded.

Records consumed (the chat-lake schema): {"cue":[...], "answer":"...",
    "triple":["subject","relation","object"] (optional), "src":"..." (optional)}.
The structured "triple" is what makes contradiction reasoning possible.

Deps: stdlib (re). No imported language model.
Self-check: PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python research/lake_reason.py
"""
import re
import sys

# Bidirectional opposites. A claim only counts as a CONTRADICTION when its value is the recognized
# opposite of the stored value -- so a mere difference (a synonym, a refinement) is NOT "corrected".
_ANTONYM_PAIRS = [
    ("east", "west"), ("north", "south"), ("up", "down"), ("left", "right"),
    ("rises", "sets"), ("rise", "set"), ("hot", "cold"), ("bigger", "smaller"),
    ("larger", "smaller"), ("faster", "slower"), ("open", "closed"), ("true", "false"),
]
_ANTONYMS = {}
for _a, _b in _ANTONYM_PAIRS:
    _ANTONYMS.setdefault(_a, set()).add(_b)
    _ANTONYMS.setdefault(_b, set()).add(_a)

# Tokens carrying no subject/relation/value content -- stripped when locating the claimed value.
# NOTE: direction words (up/down/left/right) are deliberately NOT here; they can BE the value.
_FILLERS = {"the", "a", "an", "from", "in", "to", "of", "on", "at", "into", "onto",
            "toward", "towards", "by", "with", "it", "its", "is", "was", "are", "were"}
_ARTICLES = {"the", "a", "an"}

# a stored triple may be attached as the "because" reason ONLY if its relation is genuinely causal /
# directional -- otherwise a triple that merely co-mentions both values ("compass shows east and west")
# would fabricate a causal LINK the facts never assert. Co-mention is necessary, not sufficient.
_CAUSAL_RELS = {"rotates", "rotate", "causes", "cause", "moves", "move", "spins", "spin",
                "orbits", "orbit", "turns", "turn", "drives", "drive", "flows", "flow", "makes", "make"}


def _tok(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _opposite(x, y):
    """True iff x and y are a recognized opposite pair (order-independent)."""
    return y in _ANTONYMS.get(x, ()) or x in _ANTONYMS.get(y, ())


class Reasoner:
    """Detects when a stated CLAIM contradicts a stored triple and returns the correction plus a
    stored causal reason. Indexes stored triples by (subject, relation-root)."""

    def __init__(self, facts):
        self._by_subj_rel = {}     # (subject, rel_root) -> dict(rel_display, value, subject, object)
        self._triples = []         # every stored triple, for causal-reason linking
        self._subjects = set()     # known subjects (fast "is this even about something I know")
        for rec in facts:
            triple = rec.get("triple") if isinstance(rec, dict) else None
            if not triple or len(triple) < 3:
                continue            # a record with no triple carries no contradiction structure
            subj_raw, rel_raw, obj_raw = triple[0], triple[1], triple[2]
            subj_toks = _tok(subj_raw)
            rel_toks = _tok(rel_raw)
            if not subj_toks or not rel_toks:
                continue
            subj = subj_toks[0]
            rel_root = rel_toks[0]
            entry = {
                "rel_display": " ".join(rel_toks),   # "rises_in" -> "rises in"
                "value": str(obj_raw).strip(),
                "subject": subj_raw,
                "object": str(obj_raw),
            }
            self._by_subj_rel.setdefault((subj, rel_root), []).append(entry)   # LIST: never overwrite
            self._subjects.add(subj)
            self._triples.append((subj_raw, rel_toks, str(obj_raw)))

    def _find_causal_reason(self, matched_subj, stored_value, claimed_value):
        """A stored triple is the REASON iff its object mentions BOTH opposing values -- i.e. a fact
        that explains the direction/relationship between them (e.g. ('earth','rotates','west to east')
        explains east-vs-west). Returns a reason phrase or None. Never invents; only links real facts."""
        sv = stored_value.lower()
        cv = claimed_value.lower()
        for subj_raw, rel_toks, obj in self._triples:
            obj_toks = set(_tok(obj))
            if (rel_toks and rel_toks[0] in _CAUSAL_RELS       # the relation must genuinely be causal...
                    and sv in obj_toks and cv in obj_toks      # ...and its object must relate both values
                    and _tok(subj_raw)[:1] != [matched_subj]):
                subj_title = subj_raw.strip().title()   # "earth" -> "Earth"
                return "the %s %s %s" % (subj_title, " ".join(rel_toks), obj.strip())
        return None

    def respond(self, text):
        """Return a correction string if `text` is a claim that CONTRADICTS a stored fact, else None."""
        if not text or not str(text).strip():
            return None
        # NEGATION: a negated claim ("the sun rises NOT in the west") inverts the contradiction test in
        # ways this shallow parser cannot track, so it would "correct" a TRUE negated claim. Abstain --
        # honest silence beats a confident wrong correction.
        if re.search(r"\b(not|never)\b|n't", str(text).lower()):
            return None
        toks = _tok(text)
        if not toks:
            return None

        # Subject: the first token that is actually a known subject (skips leading articles/filler,
        # and cleanly yields None for unknown subjects -> no hallucinated correction).
        subject = None
        for t in toks:
            if t in _ARTICLES:
                continue
            if t in self._subjects:
                subject = t
                break
        if subject is None:
            return None

        # Relation: a claim token that matches a stored relation-root for THIS subject.
        rels = {rel for (s, rel) in self._by_subj_rel if s == subject}
        rel_root = None
        for t in toks:
            if t in rels:
                rel_root = t
                break
        if rel_root is None:
            return None
        entries = self._by_subj_rel[(subject, rel_root)]   # ALL stored triples for this subject+relation

        # Claimed value: the last content token that is not the subject, the relation root, or filler.
        claimed = None
        for t in toks:
            if t == subject or t == rel_root or t in _FILLERS:
                continue
            claimed = t
        if claimed is None:
            return None

        # Agreement: the claimed value matches ANY stored value for this subject+relation -> consistent.
        if any(claimed in _tok(e["value"]) for e in entries):
            return None
        # Contradiction: the claimed value is a RECOGNIZED OPPOSITE of some stored single-token value.
        # (Iterating every entry fixes a last-wins bug where a second same-root triple hid the first.)
        for entry in entries:
            stored_key = _tok(entry["value"])
            stored_single = stored_key[0] if len(stored_key) == 1 else None
            if stored_single is not None and _opposite(claimed, stored_single):
                correction = "No, it %s the %s" % (entry["rel_display"], entry["value"])
                reason = self._find_causal_reason(subject, stored_single, claimed)
                return "%s because %s." % (correction, reason) if reason else "%s." % correction
        return None


def _self_check():
    facts = [
        # the two facts the correction reasons over (stored triples enable the contradiction check)
        {"cue": ["sun", "rises"], "answer": "the sun rises in the east.",
         "triple": ["sun", "rises_in", "east"]},
        {"cue": ["earth", "rotates"], "answer": "the earth rotates west to east.",
         "triple": ["earth", "rotates", "west to east"]},
        # unrelated stored facts (must never be mistaken for a reason or a subject)
        {"cue": ["france", "continent"], "answer": "france's continent is europe.",
         "triple": ["france", "continent", "europe"]},
        {"cue": ["water", "boils"], "answer": "water boils at 100c.",
         "triple": ["water", "boils_at", "100c"]},
        {"cue": ["hello"], "answer": "hi."},   # no triple -> indexed as nothing, causes no crash
    ]
    r = Reasoner(facts)

    c1 = r.respond("The sun rises from the west")   # contradiction -> correction + linked reason
    c2 = r.respond("The sun rises in the east")      # true claim -> silence
    c3 = r.respond("glorbs fly to the north")        # unknown subject -> silence
    c4 = r.respond("hello there")                    # not a claim -> silence

    print("=" * 88)
    print("LAKE REASON -- contradiction-correction over stored triples (self-check)")
    print("=" * 88)
    print("case 1  claim : The sun rises from the west")
    print("        reply : %r" % (c1,))
    print("case 2  claim : The sun rises in the east")
    print("        reply : %r  (true claim -> not corrected)" % (c2,))
    print("case 3  claim : glorbs fly to the north")
    print("        reply : %r  (unknown subject -> no hallucinated correction)" % (c3,))
    print("case 4  claim : hello there")
    print("        reply : %r  (not a claim)" % (c4,))
    print("-" * 88)

    # 1) contradiction is corrected, names the stored value AND the linked causal reason (the Earth rotating)
    assert c1 is not None, "a stored contradiction must be corrected, not ignored"
    assert "east" in c1.lower(), "the correction must state the stored value (east)"
    assert "earth" in c1.lower() and "rotat" in c1.lower(), "must attach the linked reason (Earth rotating)"
    # 2) a TRUE claim is never "corrected"
    assert c2 is None, "a claim that AGREES with the stored fact must not be corrected"
    # 3) unknown subject -> no invented correction (honesty)
    assert c3 is None, "an unknown subject must yield None, never a fabricated correction"
    # 4) non-claim -> silence
    assert c4 is None, "a non-claim must yield None"
    # honesty guard: a mere difference (not a recognized opposite) is left alone
    assert r.respond("water boils at 50c") is None, "a non-opposite difference must not be corrected"

    # honesty: a NON-CAUSAL co-mention triple must NOT be fabricated into a "because" reason
    r2 = Reasoner([
        {"cue": ["sun"], "answer": "east.", "triple": ["sun", "rises_in", "east"]},
        {"cue": ["compass"], "answer": "e/w.", "triple": ["compass", "shows", "east and west"]},
    ])
    c_noncausal = r2.respond("the sun rises from the west")
    assert c_noncausal is not None and "because" not in c_noncausal, (
        "a non-causal triple that merely co-mentions both values must NOT become a fabricated reason")

    # correctness: a second same-root triple must NOT hide the first (last-wins bug)
    r3 = Reasoner([
        {"cue": ["sun"], "answer": "east.", "triple": ["sun", "rises_in", "east"]},
        {"cue": ["sun"], "answer": "somewhere.", "triple": ["sun", "rises_up", "somewhere"]},
    ])
    assert r3.respond("the sun rises from the west") is not None, (
        "a later same-(subject,relation-root) triple must not silently destroy an earlier correction")

    # honesty: a TRUE negated claim must not be "corrected"
    assert r.respond("the sun does not rise in the west") is None, (
        "a negated claim must be abstained on, not falsely corrected")

    # and no imported language model anywhere in the path

    print("self-check OK: corrects contradictions from STORED facts (sun 'west' -> "
          "'No, it rises in the east because the Earth rotates west to east.'), stays SILENT on "
          "unknown subjects, true claims, non-opposite differences, and non-claims (honest, no "
          "hallucinated corrections) -- this is contradiction-correction over stored triples, NOT a "
          "general reasoner.")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
