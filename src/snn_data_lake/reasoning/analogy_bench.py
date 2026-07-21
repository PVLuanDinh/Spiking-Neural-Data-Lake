"""
HELD-OUT relational-analogy benchmark for `vsa_structure_mapping.StructureMapper`.

WHY THIS EXISTS. The brain's other Gf number -- the RPM/Raven score -- is a GRAMMAR SELF-TEST: the
solver is scored on puzzles emitted by the same generator it inverts, so the number measures
self-consistency, not fluid intelligence. This benchmark deliberately avoids that failure mode:

  - every item is HAND-WRITTEN from the analogy literature (Gentner's structure-mapping corpus:
    solar-system/atom, water-flow/heat-flow, plus circulatory/plumbing, army/ant-colony, food
    chains, command hierarchies). No generator produced them, so no generator can be inverted.
  - the relation vocabulary is the ITEM's, never the mapper's. StructureMapper has no trained
    vocabulary at all -- its atomic codes are crc32(label)-seeded on demand -- so an unseen
    predicate is not a special case for it, and these labels appear nowhere in its self-check.
  - items 1-3 of the CLASSIC block are the only overlap with the module's own self-check
    (solar->atom); it is kept because it is THE canonical Gentner example, and it is flagged in
    the printout so the reader can discount it.

SCORING, and why mapping-equality alone would be dishonest. An item PASSES only when the mapper
returns the gold object correspondence AND backs it with structural evidence (systematicity > 0).
That second clause is not decoration: on the relation-renamed item the mapper returns a
gold-LOOKING correspondence at systematicity 0.00 -- with no shared predicate there is no gradient
to climb, so the answer is an ungrounded coin-flip. Counting that as a hit would inflate the score
with a guess. It is scored as a FAILURE.

The benchmark is NOT tuned to the solver. Items were written once, measured once, and the failures
are kept and reported. Three of the twelve items fail, and the printout names them.

CONTROLS (a benchmark nothing fails cannot detect failure). Three negative controls must FAIL:
  1. vocab-disjoint  -- an unrelated target sharing no predicate: systematicity must be ~0.
  2. structure-disjoint -- the SAME predicates but no object shared across relations, so the
     systematic core cannot be preserved: systematicity must stay far below the true analogy's.
  3. shuffled-correspondence -- a derangement of a real analogy's gold map must NOT be recovered,
     and must score strictly below gold.

Determinism: a FRESH StructureMapper per item (its rng advances inside map(), so a reused instance
would drift across calls). ASCII output only.

  Run: python -m snn_data_lake.reasoning.analogy_bench     # prints the honest score, then asserts
"""

from snn_data_lake.reasoning.vsa_structure_mapping import StructureMapper

DIM, SEED = 1024, 0


def _chain(n, src_tag="a", tgt_tag="b"):
    """An n-object causal chain mapped onto another. The identity correspondence is the UNIQUE
    systematicity maximum (only the head has no incoming link, only the tail no outgoing one),
    so the gold is well posed at every n -- this probes SEARCH depth, not ambiguity."""
    src = [("cause", "%s%d" % (src_tag, i), "%s%d" % (src_tag, i + 1)) for i in range(1, n)]
    tgt = [("cause", "%s%d" % (tgt_tag, i), "%s%d" % (tgt_tag, i + 1)) for i in range(1, n)]
    gold = {"%s%d" % (src_tag, i): "%s%d" % (tgt_tag, i) for i in range(1, n + 1)}
    return src, tgt, gold


# (name, kind, source_props, target_props, gold_correspondence)
BENCH = [
    # ---- CLASSIC: isomorphic hand-written analogs from the literature -----------------------
    (
        "solar_system->atom",
        "classic",
        [
            ("attracts", "sun", "planet"),
            ("revolves_around", "planet", "sun"),
            ("more_massive", "sun", "planet"),
            ("hotter", "sun", "planet"),  # no target image: analogies carry over PARTIALLY
        ],
        [
            ("attracts", "nucleus", "electron"),
            ("revolves_around", "electron", "nucleus"),
            ("more_massive", "nucleus", "electron"),
        ],
        {"sun": "nucleus", "planet": "electron"},
    ),
    (
        "circulatory->plumbing",
        "classic",
        [
            ("pumps", "heart", "blood"),
            ("flows_through", "blood", "artery"),
            ("attached_to", "artery", "heart"),
        ],
        [
            ("pumps", "pump", "water"),
            ("flows_through", "water", "pipe"),
            ("attached_to", "pipe", "pump"),
        ],
        {"heart": "pump", "blood": "water", "artery": "pipe"},
    ),
    (
        "army->ant_colony",
        "classic",
        [
            ("commands", "general", "soldier"),
            ("defends", "soldier", "fort"),
            ("larger_than", "fort", "soldier"),
        ],
        [
            ("commands", "queen", "worker"),
            ("defends", "worker", "nest"),
            ("larger_than", "nest", "worker"),
        ],
        {"general": "queen", "soldier": "worker", "fort": "nest"},
    ),
    (
        "water_flow->heat_flow",
        "classic",
        [  # Gentner's canonical pressure/temperature analogy; `flow` is TERNARY
            ("greater", "beaker", "vial"),
            ("flow", "beaker", "vial", "water"),
            ("conduit", "pipe", "water"),
        ],
        [
            ("greater", "coffee", "icecube"),
            ("flow", "coffee", "icecube", "heat"),
            ("conduit", "bar", "heat"),
        ],
        {"beaker": "coffee", "vial": "icecube", "water": "heat", "pipe": "bar"},
    ),
    (
        "food_chain->marine_chain",
        "classic",
        [("eats", "fox", "rabbit"), ("eats", "rabbit", "grass"), ("faster", "fox", "rabbit")],
        [("eats", "shark", "fish"), ("eats", "fish", "algae"), ("faster", "shark", "fish")],
        {"fox": "shark", "rabbit": "fish", "grass": "algae"},
    ),
    (
        "command_chain->orchestra",
        "classic",
        [
            ("directs", "general", "colonel"),
            ("directs", "colonel", "private"),
            ("outranks", "general", "private"),
            ("obeys", "private", "colonel"),
        ],
        [
            ("directs", "conductor", "lead"),
            ("directs", "lead", "player"),
            ("outranks", "conductor", "player"),
            ("obeys", "player", "lead"),
        ],
        {"general": "conductor", "colonel": "lead", "private": "player"},
    ),
    # ---- HARD: distractors and tie-breaks that REQUIRE systematicity ------------------------
    (
        "atom+distractor",
        "hard",
        [
            ("attracts", "sun", "planet"),
            ("revolves_around", "planet", "sun"),
            ("more_massive", "sun", "planet"),
        ],
        [  # magnet/filing share 2 of the 3 predicates -- only revolves_around breaks the tie
            ("attracts", "nucleus", "electron"),
            ("revolves_around", "electron", "nucleus"),
            ("more_massive", "nucleus", "electron"),
            ("attracts", "magnet", "filing"),
            ("more_massive", "magnet", "filing"),
        ],
        {"sun": "nucleus", "planet": "electron"},
    ),
    (
        "two_planets->two_electrons",
        "hard",
        [  # earth and mars are separated ONLY by `farther` -- no other cue exists
            ("orbits", "earth", "sun"),
            ("orbits", "mars", "sun"),
            ("more_massive", "sun", "earth"),
            ("more_massive", "sun", "mars"),
            ("farther", "mars", "earth"),
        ],
        [
            ("orbits", "e1", "nuc"),
            ("orbits", "e2", "nuc"),
            ("more_massive", "nuc", "e1"),
            ("more_massive", "nuc", "e2"),
            ("farther", "e2", "e1"),
        ],
        {"earth": "e1", "mars": "e2", "sun": "nuc"},
    ),
    (
        "family->org_chart",
        "hard",
        [  # branching hierarchy; the two leaves are separated only by `older`
            ("parent", "grandpa", "dad"),
            ("parent", "dad", "son"),
            ("parent", "dad", "daughter"),
            ("older", "son", "daughter"),
        ],
        [
            ("parent", "founder", "director"),
            ("parent", "director", "alice"),
            ("parent", "director", "bob"),
            ("older", "alice", "bob"),
        ],
        {"grandpa": "founder", "dad": "director", "son": "alice", "daughter": "bob"},
    ),
    # ---- LIMIT: probes of known capability edges. Expected to be hard; kept whatever happens.
    (
        "renamed_relations",
        "limit",
        [  # identical STRUCTURE, different predicate LABELS -- the realistic cross-domain case
            ("attracts", "sun", "planet"),
            ("revolves_around", "planet", "sun"),
            ("more_massive", "sun", "planet"),
        ],
        [
            ("pulls", "nucleus", "electron"),
            ("circles", "electron", "nucleus"),
            ("heavier", "nucleus", "electron"),
        ],
        {"sun": "nucleus", "planet": "electron"},
    ),
]
BENCH += [
    ("causal_chain_n%d" % n, "limit", _chain(n)[0], _chain(n)[1], _chain(n)[2]) for n in (6, 8)
]


def evaluate(items=None, dim=DIM, seed=SEED):
    """Score each item with a FRESH mapper. PASS = gold correspondence recovered AND supported by
    structural evidence (systematicity > 0) -- a match at systematicity 0 is a guess, not a hit."""
    out = []
    for name, kind, src, tgt, gold in items if items is not None else BENCH:
        mapper = StructureMapper(dim=dim, seed=seed)
        got = mapper.map(src, tgt)
        sys_got = mapper.score(got, src, tgt)
        sys_gold = mapper.score(gold, src, tgt)
        out.append(
            {
                "name": name,
                "kind": kind,
                "got": got,
                "gold": gold,
                "sys": sys_got,
                "sys_gold": sys_gold,
                "exact": got == gold,
                "passed": got == gold and sys_got > 0.0,
            }
        )
    return out


def controls(dim=DIM, seed=SEED):
    """The three negative controls. Each must FAIL to be recovered / must score near zero."""
    solar = BENCH[0][2]
    kitchen = [("bakes", "chef", "bread"), ("owns", "chef", "oven"), ("inside", "bread", "oven")]
    m1 = StructureMapper(dim=dim, seed=seed)
    map1 = m1.map(solar, kitchen)

    scatter = [  # same predicates as solar, but NO object appears in two relations
        ("attracts", "p", "q"),
        ("revolves_around", "r", "s"),
        ("more_massive", "t", "u"),
        ("hotter", "v", "w"),
    ]
    m2 = StructureMapper(dim=dim, seed=seed)
    map2 = m2.map(solar, scatter)

    _n, _k, csrc, ctgt, cgold = BENCH[1]  # circulatory->plumbing
    shuffled = {"heart": "water", "blood": "pipe", "artery": "pump"}  # a derangement of gold
    m3 = StructureMapper(dim=dim, seed=seed)
    recovered = m3.map(csrc, ctgt)
    return {
        "vocab_disjoint": {"map": map1, "sys": m1.score(map1, solar, kitchen)},
        "structure_disjoint": {"map": map2, "sys": m2.score(map2, solar, scatter)},
        "shuffled": {
            "recovered": recovered,
            "shuffled": shuffled,
            "sys_gold": m3.score(cgold, csrc, ctgt),
            "sys_shuffled": m3.score(shuffled, csrc, ctgt),
            "recovered_is_shuffled": recovered == shuffled,
        },
    }


def main():
    rows = evaluate()
    ctl = controls()
    hits = sum(r["passed"] for r in rows)
    acc = hits / len(rows)

    print("=" * 96)
    print("HELD-OUT RELATIONAL-ANALOGY BENCHMARK -- StructureMapper (Gentner SMT over VSA codes)")
    print("=" * 96)
    print("items are hand-written from the analogy literature, NOT emitted by a grammar the solver")
    print("inverts. PASS = gold correspondence recovered AND systematicity > 0 (evidence, not luck).")
    print("-" * 96)
    print("%-28s %-8s %-6s %-9s %s" % ("item", "kind", "result", "sys/gold", "recovered"))
    for r in rows:
        print(
            "%-28s %-8s %-6s %.2f/%.2f  %s"
            % (
                r["name"],
                r["kind"],
                "PASS" if r["passed"] else "FAIL",
                r["sys"],
                r["sys_gold"],
                r["got"] if not r["passed"] else "gold",
            )
        )
    print("-" * 96)
    print("HONEST ACCURACY: %d/%d = %.3f" % (hits, len(rows), acc))
    by_kind = {}
    for r in rows:
        agg = by_kind.setdefault(r["kind"], [0, 0])
        agg[0] += r["passed"]
        agg[1] += 1
    print(
        "  by kind: %s"
        % "  ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in sorted(by_kind.items()))
    )
    fails = [r["name"] for r in rows if not r["passed"]]
    print("  FAILURES (kept, not tuned away): %s" % (", ".join(fails) if fails else "none"))
    print(
        "  note: 'renamed_relations' returns a gold-LOOKING map at systematicity %.2f -- with no\n"
        "        shared predicate there is no gradient, so it is a guess and is scored a FAILURE."
        % next(r["sys"] for r in rows if r["name"] == "renamed_relations")
    )
    print(
        "  note: the long causal chains settle into a SHIFTED near-miss local optimum rather than\n"
        "        the unique maximum -- a real search-depth limit, not a scoring artifact."
    )
    print("-" * 96)
    print("NEGATIVE CONTROLS (each MUST fail):")
    vd, sd, sh = ctl["vocab_disjoint"], ctl["structure_disjoint"], ctl["shuffled"]
    print("  vocab-disjoint      : systematicity=%.3f  map=%s" % (vd["sys"], vd["map"]))
    print("  structure-disjoint  : systematicity=%.3f  map=%s" % (sd["sys"], sd["map"]))
    print(
        "  shuffled-corr       : recovered=%s  sys_gold=%.2f sys_shuffled=%.2f"
        % ("SHUFFLED (bad)" if sh["recovered_is_shuffled"] else "gold (correct)",
           sh["sys_gold"], sh["sys_shuffled"])
    )
    print("=" * 96)

    # ---- assertions: the real analogies MOSTLY pass, and every control FAILS ------------------
    assert acc >= 0.5, "the held-out analogies must mostly pass (got %.3f)" % acc
    classic = [r for r in rows if r["kind"] == "classic"]
    assert all(r["passed"] for r in classic), "every canonical classic analog must be recovered"

    assert vd["sys"] <= 0.01, "a vocab-disjoint target must be ~0 systematic (got %.3f)" % vd["sys"]
    assert sd["sys"] < 0.5, (
        "a structure-disjoint target must stay far below a true analogy (got %.3f)" % sd["sys"]
    )
    assert not sh["recovered_is_shuffled"], "a shuffled correspondence must NOT be recovered"
    assert sh["sys_shuffled"] < sh["sys_gold"], "the shuffled map must score below gold"

    # the controls must be separated from the real analogies, not merely low in isolation
    best_ctl = max(vd["sys"], sd["sys"], sh["sys_shuffled"])
    worst_real = min(r["sys"] for r in rows if r["kind"] == "classic")
    assert worst_real > best_ctl, (
        "the weakest real analogy (%.2f) must outscore the strongest control (%.2f)"
        % (worst_real, best_ctl)
    )

    # determinism: a second pass reproduces every mapping exactly
    assert [r["got"] for r in evaluate()] == [r["got"] for r in rows], "bench must be deterministic"

    print(
        "self-check OK: %d/%d held-out analogies recovered with structural evidence (%.1f%%); "
        "%d kept failures; all 3 negative controls failed as required; deterministic."
        % (hits, len(rows), 100.0 * acc, len(fails))
    )
    return 0


if __name__ == "__main__":
    main()
