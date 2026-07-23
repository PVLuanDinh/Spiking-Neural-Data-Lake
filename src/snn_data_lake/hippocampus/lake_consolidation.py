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

UTILITY (v2 -- use it or lose it): a RecallTrace counts how often each record is RETRIEVED as an answer
(bumped on the LakeChat recall path, persisted under the home). When the learned set exceeds its cap,
consolidate(records, trace) keeps the HIGHEST recall-utility (tie -> most recent), so the never-recalled
old records fall out FIRST -- strengthen-on-use + prune-the-unused, over experience, not just dictionary
bulk. Without a trace it falls back to recency (v1). Together v1 (cap re-derivable) + v2 (utility-rank the
learned cap) stop the monotonic growth + the WordNet balloon a never-off brain otherwise hits.

Determinism (CLAUDE.md): pure function; no random, no wall-clock, no hash(); ASCII-only prints.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.lake_consolidation
"""

import zlib

# re-derivable ingest sources: bulk knowledge the boot rebuilds (world-KB seed + dictionary gates), so it
# need not be hoarded durably. A src is matched by its prefix before ':' (e.g. "wordnet:dog" -> "wordnet").
RE_DERIVABLE_SRC = frozenset({"wordnet", "conceptnet", "worldkb", "world-kb"})


def _src_class(record):
    """'re-derivable' if the record came from a bulk re-derivable ingest, else 'learned'."""
    src = str(record.get("src", ""))
    head = src.split(":", 1)[0].strip().lower()
    return "re-derivable" if head in RE_DERIVABLE_SRC else "learned"


def record_key(record):
    """A stable content id for a lake record: crc32 of its cue + answer (never builtin hash, so it is
    reproducible across processes). The key the recall trace counts against."""
    payload = "|".join(str(c) for c in record.get("cue", [])) + "||" + str(record.get("answer", ""))
    return format(zlib.crc32(payload.encode("utf-8")), "08x")


class RecallTrace:
    """The 'USE IT' half of use-it-or-lose-it: a durable per-record recall counter. Every time a stored
    record is RETRIEVED as an answer, bump() increments its count; sleep consolidation then prunes the
    NEVER-recalled learned records first (not merely the oldest). Persisted as a JSON state blob under the
    home (substrate.home.save_state / load_state), so the utility signal survives restarts like the lake it
    ranks. In-RAM bump is cheap; save() is called on the daemon's persist / sleep, not per retrieval."""

    def __init__(self, counts=None):
        if counts is not None:
            self._counts = dict(counts)
        else:
            from snn_data_lake.substrate.home import load_state

            self._counts = dict(load_state("recall_trace.json", default={}) or {})

    def bump(self, record, by=1):
        """Record that `record` was recalled/used. Returns its new count."""
        k = record_key(record)
        self._counts[k] = self._counts.get(k, 0) + int(by)
        return self._counts[k]

    def get(self, record):
        """The recall count of `record` (0 if never recalled)."""
        return self._counts.get(record_key(record), 0)

    def prune_to(self, live_records):
        """Drop trace entries for records no longer in the lake (so the blob does not grow unbounded)."""
        live = {record_key(r) for r in live_records}
        self._counts = {k: v for k, v in self._counts.items() if k in live}

    def save(self):
        """Persist the trace under the home (atomic JSON state blob)."""
        from snn_data_lake.substrate.home import save_state

        save_state("recall_trace.json", self._counts)


class LakeConsolidation:
    """The sleep-time PRUNE policy for the symbolic durable lake. consolidate(records) -> the survivors a
    KnowledgeLog.consolidate() rewrite should keep, so the lake stays live + bounded on a never-off brain."""

    def __init__(self, rederivable_budget=3000, learned_cap=250000):
        # keep at most this many of the most-recent re-derivable (dictionary) records; the rest re-derive.
        self.rederivable_budget = int(rederivable_budget)
        # a large safety ceiling on genuinely-learned records (recency-kept); v2 makes this recall-weighted.
        self.learned_cap = int(learned_cap)

    def consolidate(self, records, trace=None):
        """records: list of fact dicts in append order (oldest..newest). Returns the kept subset, order
        preserved. Re-derivable capped to its newest budget; learned capped to learned_cap. USE IT OR LOSE
        IT (v2): when learned exceeds the cap AND a RecallTrace is given, keep the highest recall-utility
        (tie -> most recent) so the NEVER-recalled old records fall out first, not merely the oldest;
        without a trace it falls back to recency (v1). Pure -- does not touch disk (KnowledgeLog.consolidate
        rewrites with the result)."""
        learned, rederivable = [], []
        for i, r in enumerate(records):
            (rederivable if _src_class(r) == "re-derivable" else learned).append((i, r))
        # cap each class to its budget (records are in append order, so the tail = newest)
        if len(rederivable) > self.rederivable_budget:
            rederivable = rederivable[-self.rederivable_budget :]
        if len(learned) > self.learned_cap:
            if (
                trace is not None
            ):  # use-it-or-lose-it: rank by (recall count, recency), keep the top cap
                learned = sorted(learned, key=lambda ir: (trace.get(ir[1]), ir[0]))[
                    -self.learned_cap :
                ]
            else:  # v1: recency only
                learned = learned[-self.learned_cap :]
        kept = {i for i, _ in learned} | {i for i, _ in rederivable}
        # re-emit in the ORIGINAL append order so replay order + dedup identity are preserved
        return [r for i, r in enumerate(records) if i in kept]

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

    # ---- v2 USE IT OR LOSE IT: over a tight learned cap, a RECALLED old record survives a newer unused one ----
    lg = [
        {"cue": ["a"], "answer": "old recalled", "src": "chat:user"},  # oldest, but recalled a lot
        {"cue": ["b"], "answer": "mid unused", "src": "chat:user"},
        {"cue": ["c"], "answer": "new unused", "src": "chat:user"},  # newest, never recalled
    ]
    trace = RecallTrace(counts={})
    for _ in range(5):
        trace.bump(lg[0])  # the OLD record is the one that gets used
    v1 = LakeConsolidation(learned_cap=1).consolidate(
        lg
    )  # recency -> keeps the NEWEST ("new unused")
    v2 = LakeConsolidation(learned_cap=1).consolidate(
        lg, trace=trace
    )  # utility -> keeps the RECALLED old one
    assert v1 == [lg[2]], f"v1 (recency) should keep the newest: {v1}"
    assert v2 == [lg[0]], (
        f"v2 (use-it-or-lose-it) should keep the RECALLED record, not the newest: {v2}"
    )
    assert trace.get(lg[0]) == 5 and trace.get(lg[1]) == 0, "recall trace counts wrong"
    # trace persists across a fresh instance + prune_to drops entries for gone records
    trace.prune_to([lg[0]])
    assert trace.get(lg[1]) == 0 and trace.get(lg[0]) == 5, "prune_to must keep live, drop gone"
    print(
        "  [OK] v2 use-it-or-lose-it: a RECALLED old record beats a newer never-recalled one under the cap"
    )

    print("-" * width)
    print(
        "self-check OK: sleep consolidation caps re-derivable dictionary bulk (WordNet/ConceptNet -- they"
    )
    print(
        "re-derive at boot) + retains genuinely-learned experience + preserves append order. The PRUNE"
    )
    print(
        "half that keeps a never-off brain's symbolic lake LIVE + BOUNDED. v2 RecallTrace adds use-it-or-"
    )
    print("lose-it: a recalled learned record survives the cap over a newer never-recalled one.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
