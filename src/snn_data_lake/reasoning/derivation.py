"""Multi-hop transitive DEDUCTION over a records list -- free-function form.

This module lifts the proven BFS transitive-closure deduction out of
``reasoning/brain_loop.py`` (the ``_relation_graph`` / ``_derive`` /
``_reason_question`` methods) into stateless FREE FUNCTIONS that operate on a
plain ``records`` list -- each record shaped
``{"cue": [...], "answer": "...", "triple": ["subject", "relation", "object"]}``
(the ``records`` attribute of ``reasoning/spike_chat_lake.py::LakeChat``).

The algorithm is unchanged: breadth-first search from a subject, following ONLY
relations that legitimately chain, up to a bounded number of hops, returning the
PROOF PATH that settles the question -- showing its work instead of asserting.

Honest ceiling (carried verbatim from brain_loop's design):
  * This is DEDUCTION over TAUGHT facts only -- it derives nothing that was not
    stored, and invents nothing.
  * Only genuinely TRANSITIVE relations chain (isa / partof / subclassof /
    kindof / typeof). "has", "resembles", "causes" and bare "is" (predication,
    e.g. "sky is blue") are deliberately EXCLUDED -- chaining them would
    manufacture false conclusions.
  * Bounded to <= 6 hops.
  * ABSENCE OF A PROOF IS NOT FALSITY. A genuine miss returns ``None`` (the
    caller's honest gap then stands); it is never dressed up as a "no".

Uses only ``re`` + ``sys``; no brainstem import.
"""

import re
import sys

# relations that CHAIN (a IsA b, b IsA c => a IsA c). "has"/"resembles"/"causes"
# do NOT chain, and pretending they do would manufacture false conclusions -- so
# they are deliberately excluded. bare "is" is EXCLUDED too: it marks PREDICATION
# ("sky is blue"), not class membership, and chaining it fabricates ("sky is
# blue" + "blue isa color" => "sky is a color"). Genuine class facts use
# isa/kindof/typeof/subclassof; predication does not license deduction.
# (Verbatim from brain_loop.Brain._TRANSITIVE.)
_TRANSITIVE = frozenset({"isa", "partof", "part_of", "subclassof", "kindof", "typeof"})


def _tok(text):
    """Lowercase alphanumeric tokens -- the same tokeniser LakeChat uses for triple nodes."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _art(word):
    """'a' / 'an' agreement for readable proofs (no side effects)."""
    return "an" if word[:1] in "aeiou" else "a"


def relation_graph(records):
    """subject -> [(relation, object)] over every stored triple: the brain's own knowledge graph.

    Nodes are the joined content-token form of a triple's subject/object (so
    "living thing" keys consistently whether taught with one space or many).
    Faithful port of ``brain_loop.Brain._relation_graph`` operating on a records list.
    """
    adj = {}
    for rec in (records or []):
        t = rec.get("triple") if isinstance(rec, dict) else None
        if not t or len(t) < 3:
            continue
        adj.setdefault(" ".join(_tok(t[0])), []).append((str(t[1]), " ".join(_tok(t[2]))))
    return adj


def _derive_to_goal(adj, subject, target, max_hops=6):
    """DEDUCE by chaining transitive relations toward a SPECIFIC target: breadth-first from
    `subject`, following only relations that legitimately chain, looking for `target`.
    Returns the PROOF as ``[(a, rel, b), ...]`` or ``None``.

    Verbatim port of ``brain_loop.Brain._derive`` (max_depth renamed max_hops).
    """
    start, goal = " ".join(_tok(subject)), " ".join(_tok(target))
    if not start or not goal or start not in adj:
        return None
    seen, queue = {start}, [(start, [])]
    while queue:
        node, path = queue.pop(0)
        if len(path) >= max_hops:
            continue
        for rel, obj in adj.get(node, []):
            if rel.lower().replace(" ", "") not in _TRANSITIVE:
                continue                                  # only chain what actually chains
            step = path + [(node, rel, obj)]
            if obj == goal:
                return step                               # proof found
            if obj not in seen:
                seen.add(obj)
                queue.append((obj, step))
    return None


def derive(records, subject, relation, max_hops=6):
    """Chase a SPECIFIC transitive `relation` from `subject` to its terminus over the stored
    triples, returning ``(answer, proof_path)`` or ``None``.

    ``answer`` is the farthest object reachable along `relation` (e.g. derive(dog, isa) walks
    dog -> mammal -> animal -> living thing and answers "living thing"); ``proof_path`` is the
    ``[(a, rel, b), ...]`` chain. Same transitive-relation gate and hop bound as the yes/no
    deduction -- a `relation` that does not legitimately chain (or a subject with no such edge)
    yields ``None`` (honest: it will not fabricate a link).
    """
    adj = relation_graph(records)
    start = " ".join(_tok(subject))
    rel_norm = str(relation).lower().replace(" ", "")
    if not start or rel_norm not in _TRANSITIVE or start not in adj:
        return None
    proof, node, seen = [], start, {start}
    while len(proof) < max_hops:
        nxt = None
        for rel, obj in adj.get(node, []):
            if rel.lower().replace(" ", "") == rel_norm and obj not in seen:
                nxt = (rel, obj)
                break                                     # first matching edge (deterministic)
        if nxt is None:
            break
        proof.append((node, nxt[0], nxt[1]))
        node = nxt[1]
        seen.add(node)
    if not proof:
        return None
    return (node, proof)


def _chain_str(proof):
    """'a -> b -> c' rendering of a proof path (the shown work)."""
    return " -> ".join([proof[0][0]] + [p[2] for p in proof])


def reason_question(text, records):
    """Answer a question by DERIVING it (with the proof shown), or return ``None`` honestly.

    Two question shapes are parsed:
      * yes/no       -- "is a dog an animal" / "are dogs mammals"
      * what/which   -- "what is the isa of dog" / "which is the partof of finger"

    On a proof, returns
    ``{"known": True, "coverage": 1.0, "ignited": True, "source": "derive",
       "reply": "<answer> (I worked it out: a -> b -> c)"}``.
    A genuine miss returns ``None`` -- absence of a proof is NOT falsity, so it is never
    rendered as a "no"; the caller's honest gap stands. Nothing is fabricated.
    """
    if not text:
        return None
    adj = relation_graph(records)
    t = str(text).strip().lower()

    # form 1: yes/no. The article group MUST require a trailing space, else "(?:a|an|the)?"
    # eats the "a" of "animal" and leaves "n animal" as the target (the 2-hop bug brain_loop fixed).
    m = re.match(r"^(?:is|are)\s+(?:(?:an|a|the)\s+)?(.+?)\s+(?:(?:an|a|the)\s+)?([\w' -]+?)\s*\??$", t)
    if m:
        subj, targ = m.group(1).strip(), m.group(2).strip()
        proof = _derive_to_goal(adj, subj, targ)
        if proof:
            answer = proof[-1][2]
            return {"known": True, "coverage": 1.0, "ignited": True, "source": "derive",
                    "reply": f"Yes -- {proof[0][0]} is {_art(answer)} {answer}. "
                             f"(I worked it out: {_chain_str(proof)})"}
        return None                                       # no proof -> honest None (not a "no")

    # form 2: "what/which is the <relation> of <subject>"
    m2 = re.match(r"^(?:what|which)\s+is\s+the\s+([\w'-]+?)\s+of\s+(.+?)\s*\??$", t)
    if m2:
        relation, subj = m2.group(1).strip(), m2.group(2).strip()
        got = derive(records, subj, relation)
        if got:
            answer, proof = got
            return {"known": True, "coverage": 1.0, "ignited": True, "source": "derive",
                    "reply": f"{answer} (I worked it out: {_chain_str(proof)})"}
        return None
    return None


# ------------------------------------------------------------- headless self-check -------------------


def _selfcheck():
    # a taught chain (dog -> mammal -> animal -> living thing) + a couple of unrelated triples.
    records = [
        {"cue": ["dog"], "answer": "a dog is a mammal.", "triple": ["dog", "isa", "mammal"]},
        {"cue": ["mammal"], "answer": "a mammal is an animal.", "triple": ["mammal", "isa", "animal"]},
        {"cue": ["animal"], "answer": "an animal is a living thing.",
         "triple": ["animal", "isa", "living thing"]},
        {"cue": ["car"], "answer": "a car has wheels.", "triple": ["car", "has", "wheels"]},
        {"cue": ["rain"], "answer": "rain causes wet ground.", "triple": ["rain", "causes", "wet ground"]},
    ]

    print("=" * 88)
    print("DERIVATION -- multi-hop transitive deduction (BFS <=6 hops, proof-carrying, over taught facts)")
    print("=" * 88)

    # (1) a multi-hop proof: dog is an animal, via dog -> mammal -> animal.
    got = reason_question("is a dog an animal", records)
    assert got is not None, "expected a proof dict, got None"
    assert got["known"] and got["source"] == "derive", got
    assert "dog -> mammal -> animal" in got["reply"], f"reply must show the multi-hop chain: {got['reply']!r}"
    print(f"(1) proof   : {got['reply']}")

    # (2) a NO-PATH control refuses: nothing chains dog to "mineral", so the proof is refused (None).
    miss = reason_question("is a dog a mineral", records)
    assert miss is None, f"a no-path query must refuse (None), got {miss!r}"
    print("(2) no-path : refused (None) -- absence of a proof is not falsity")

    # (2b) graph shape + terminus derivation are faithful.
    adj = relation_graph(records)
    assert adj["dog"] == [("isa", "mammal")], adj.get("dog")
    term = derive(records, "dog", "isa")
    assert term is not None and term[0] == "living thing", term
    print(f"(2b) derive : dog --isa--> {term[0]}  [{_chain_str(term[1])}]")

    # (3) determinism: identical inputs -> identical outputs.
    assert reason_question("is a dog an animal", records) == reason_question("is a dog an animal", records)
    print("(3) determinism: stable across repeated calls")

    # (4) a non-chaining relation ("causes") never fabricates a deduction.
    assert reason_question("is rain wet ground", records) is None
    assert derive(records, "rain", "causes") is None
    print("(4) non-chaining relations (causes/has/is) refuse to deduce -- no fabrication")

    print("self-check OK: transitive deduction reproduces brain_loop (proof-carrying BFS, honest refusal)")


if __name__ == "__main__":
    _selfcheck()
