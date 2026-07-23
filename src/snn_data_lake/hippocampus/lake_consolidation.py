"""
Sleep consolidation of the SYMBOLIC durable lake -- the PRUNE half of a never-off brain's memory.

The spike lake already lives: hippocampus/plasticity consolidation prunes spike/prototype units by a
use-decayed utility trace ("use it or lose it") every night. The SYMBOLIC lake (home.KnowledgeLog over
brain_lake.bin: taught facts, chat turns, autonomously-ingested knowledge) did NOT -- sleep only ever
APPENDED to it (schema induction), so it grew monotonically and re-derivable dictionary bulk (WordNet /
ConceptNet, ingested by the tens of thousands) piled up. This module is the missing prune + consolidation
pass the brain runs during sleep, so the durable lake stays LIVE and BOUNDED instead of ballooning.

Policy (deterministic, recency-ordered; pure function of the record list):
  * CAP RE-DERIVABLE BULK: records whose src is a re-derivable INGEST (wordnet / conceptnet / ... -- bulk
    dictionaries the boot re-derives, not genuinely-learned experience) are kept only up to a small budget
    (the most-recent). The rest are pruned -- they RE-DERIVE at boot, they are not lost knowledge. This is
    what dissolves the "store the 80k WordNet or rebuild it?" dilemma: sleep keeps a live working set and
    lets the rest re-derive, exactly as a brain does not durably hoard every dictionary entry it ever read.
  * RETAIN GENUINELY-LEARNED: taught / chat / reflection / perception / wikipedia / skill / schema records
    are kept, subject only to a large SAFETY cap so a runaway can never unbound the lake.
  * order preserved (oldest -> newest), so replay order and "most recent survives the cap" both hold.

HONEST SCOPE: v1 utility = recency + src-class (re-derivable vs learned). The full biological form --
strengthen a record each time it is RECALLED and prune the never-recalled learned records first ("use it
or lose it" over experience, not just over dictionary bulk) -- is the next increment; it needs a per-record
recall trace the retrieval path does not yet write. v1 already stops the monotonic growth + the WordNet
balloon, which is the gap the always-on brain actually hit.

Determinism (CLAUDE.md): pure function; no random, no wall-clock, no hash(); ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.lake_consolidation
"""

# re-derivable ingest sources: bulk knowledge the boot rebuilds (world-KB seed + dictionary gates), so it
# need not be hoarded durably. A src is matched by its prefix before ':' (e.g. "wordnet:dog" -> "wordnet").
RE_DERIVABLE_SRC = frozenset({"wordnet", "conceptnet", "worldkb", "world-kb"})


def _src_class(record):
    """'re-derivable' if the record came from a bulk re-derivable ingest, else 'learned'."""
    src = str(record.get("src", ""))
    head = src.split(":", 1)[0].strip().lower()
    return "re-derivable" if head in RE_DERIVABLE_SRC else "learned"


class LakeConsolidation:
    """The sleep-time PRUNE policy for the symbolic durable lake. consolidate(records) -> the survivors a
    KnowledgeLog.consolidate() rewrite should keep, so the lake stays live + bounded on a never-off brain."""

    def __init__(self, rederivable_budget=3000, learned_cap=250000):
        # keep at most this many of the most-recent re-derivable (dictionary) records; the rest re-derive.
        self.rederivable_budget = int(rederivable_budget)
        # a large safety ceiling on genuinely-learned records (recency-kept); v2 makes this recall-weighted.
        self.learned_cap = int(learned_cap)

    def consolidate(self, records):
        """records: list of fact dicts in append order (oldest..newest). Returns the kept subset, order
        preserved: ALL learned (up to learned_cap, newest kept) + the newest `rederivable_budget`
        re-derivable. Pure -- does not touch disk (KnowledgeLog.consolidate rewrites with the result)."""
        learned, rederivable = [], []
        for r in records:
            (rederivable if _src_class(r) == "re-derivable" else learned).append(r)
        # cap each class to its most-recent budget (records are in append order, so tail = newest)
        if len(rederivable) > self.rederivable_budget:
            rederivable = rederivable[-self.rederivable_budget :]
        if len(learned) > self.learned_cap:
            learned = learned[-self.learned_cap :]
        kept = set(id(r) for r in learned) | set(id(r) for r in rederivable)
        # re-emit in the ORIGINAL append order so replay order + dedup identity are preserved
        return [r for r in records if id(r) in kept]

    def report(self, records):
        """A dry-run summary (counts) of what consolidate() would keep/prune -- for logging in sleep."""
        kept = self.consolidate(records)
        return {
            "total": len(records),
            "kept": len(kept),
            "pruned": len(records) - len(kept),
            "re_derivable_in": sum(1 for r in records if _src_class(r) == "re-derivable"),
            "learned_in": sum(1 for r in records if _src_class(r) == "learned"),
        }


# ================================================================================================
def main():
    width = 92
    print("=" * width)
    print("LAKE CONSOLIDATION -- sleep-time prune of the symbolic durable lake (live + bounded)")
    print("=" * width)

    # a lake that ballooned with re-derivable dictionary bulk + a handful of genuinely-learned records.
    records = []
    for i in range(50):
        records.append({"cue": [f"word{i}"], "answer": f"def {i}", "src": f"wordnet:word{i}"})
    for i in range(20):
        records.append(
            {"cue": [f"concept{i}"], "answer": f"rel {i}", "src": f"conceptnet:concept{i}"}
        )
    learned = [
        {
            "cue": ["pybullet", "arm"],
            "answer": "keep the pybullet backend installed.",
            "src": "chat:user",
        },
        {
            "cue": ["one", "lake"],
            "answer": "one durable lake for the whole brain.",
            "src": "chat:assistant",
        },
        {
            "cue": ["calculus"],
            "answer": "Calculus: the study of change.",
            "src": "wikipedia:Calculus",
        },
        {"cue": ["skill", "3"], "answer": "skill 3 discovered.", "src": "skill-discovery"},
    ]
    records += learned

    lc = LakeConsolidation(rederivable_budget=10, learned_cap=1000)
    rep = lc.report(records)
    kept = lc.consolidate(records)

    print(
        f"  in: {rep['total']} records ({rep['re_derivable_in']} re-derivable, {rep['learned_in']} learned)"
    )
    print(f"  kept {rep['kept']} / pruned {rep['pruned']}  (re-derivable capped at 10)")

    # every genuinely-learned record survives (use-it experience is never pruned by the dictionary cap)
    for r in learned:
        assert r in kept, f"a genuinely-learned record was wrongly pruned: {r}"
    # re-derivable bulk is capped to the budget (the newest kept), the rest pruned (they re-derive)
    kept_rederiv = [r for r in kept if _src_class(r) == "re-derivable"]
    assert len(kept_rederiv) == 10, f"re-derivable not capped to 10: got {len(kept_rederiv)}"
    # the KEPT re-derivable are the MOST RECENT (word49..word40 region + none older) -> order preserved
    assert kept_rederiv == [r for r in records if _src_class(r) == "re-derivable"][-10:], (
        "cap must keep newest"
    )
    # order preserved overall (a subsequence of the input)
    idx = [records.index(r) for r in kept]
    assert idx == sorted(idx), "consolidate must preserve append order"
    # a lake with ONLY learned records is unchanged (nothing to prune)
    assert lc.consolidate(learned) == learned, "a lake of only learned records must be untouched"
    print(
        f"  [OK] all {len(learned)} learned retained; re-derivable capped to newest 10; order preserved"
    )

    print("-" * width)
    print(
        "self-check OK: sleep consolidation caps re-derivable dictionary bulk (WordNet/ConceptNet -- they"
    )
    print(
        "re-derive at boot) + retains genuinely-learned experience + preserves append order. The PRUNE"
    )
    print(
        "half that keeps a never-off brain's symbolic lake LIVE + BOUNDED. Recall-weighted 'use it or"
    )
    print("lose it' over learned records is the next increment (needs a per-record recall trace).")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
