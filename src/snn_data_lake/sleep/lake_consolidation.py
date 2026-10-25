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

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sleep.lake_consolidation
"""

import zlib

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "LakeConsolidation.consolidate": (
        "NOT_A_RULE",
        "returns the kept subset of fact records in order. RECORD pruning against the durable store "
        "-- P4b storage, not plasticity",
    ),
}

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
    NEVER-recalled learned records first (not merely the oldest). In-RAM bump is cheap; save() is called
    on the daemon's persist / sleep, not per retrieval.

    P4b VIOLATION CLOSED 2026-08-05, and the shape of the fix is worth keeping. These counts are
    LEARNED state -- they are what the brain has found worth keeping, and `LakeConsolidation.consolidate`
    uses them to decide which lake records SURVIVE the night -- and they were persisted to
    `<home>/recall_trace.json`, a JSON sidecar beside `brain_lake.bin`, which is what P4b forbids in its
    own words ("not a sidecar file, not a JSON blob"). It was LIVE, not hypothetical:
    `reasoning/spike_chat_lake` bumps it on the recall path and saves it, and `engine/brain.py`'s sleep
    tick reads it back to rank the survivors.

    THE LINES BELOW DID NOT CHANGE. `substrate.home.save_state`/`load_state` did: they now write a
    `_T_STATE` record INTO the one lake, keyed by the name the caller already passes, and import an
    existing sidecar once on first read (2,505 B of real recall history on the operator's home). The
    audit that found this proposed a dedicated `_T_RECALL` tag plus a `save_recall_trace`/
    `load_recall_trace` pair here; that would have been the fifth near-identical channel for one job,
    so the mechanism moved instead of the caller (P2: exactly one survives).

    What that buys, concretely, is the cost `_T_SELFBUILD` was created to fix for the self-build
    learner's weights: `consolidate()` can now SEE the trace, so sleep-time compaction no longer ranks
    the lake by a signal the lake does not contain, and a foreign-record rewrite keeps the two in step
    because there are no longer two stores. Proven across a process boundary in main() below -- "it
    persists" is a claim about a NEW process, and inspecting this object's own `_counts` proves
    nothing."""

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
        """Persist the trace INTO the one durable lake (`_T_STATE`, name "recall_trace.json").

        Returns True if a record was written, False if the trace was unchanged since the last save --
        the lake is append-only and sleep consolidation preserves every non-FACT record, so re-storing
        an identical trace on every sleep tick would grow the one store and add nothing."""
        from snn_data_lake.substrate.home import save_state

        return save_state("recall_trace.json", self._counts)


class LakeConsolidation:
    """The sleep-time PRUNE policy for the symbolic durable lake. consolidate(records) -> the survivors a
    KnowledgeLog.consolidate() rewrite should keep, so the lake stays live + bounded on a never-off brain."""

    def __init__(self, rederivable_budget=3000, learned_cap=250000):
        # keep at most this many of the most-recent re-derivable (dictionary) records; the rest re-derive.
        self.rederivable_budget = int(rederivable_budget)
        # A ceiling on genuinely-learned records that IS NOT APPLIED while _T_FACT is reclaimable=False --
        # see consolidate(). Kept as a field because the registry, not this class, decides: flip 0x01 to
        # reclaimable and the cap takes effect again without touching this file (specs/022 L6, 2026-09-06).
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
        # THE LABEL DECIDES, NOT THIS POLICY. _T_FACT is IMMORTAL / reclaimable=False, and this cap used to
        # drop learned records under it every eighth night -- the contradiction L5's gate found (specs/022
        # L6). While the class is non-reclaimable the cap is REFUSED; the re-derivable budget above still
        # applies, under the per-record exemption RETENTION[0x01]'s own reason now declares.
        from snn_data_lake.substrate.home import _T_FACT
        from snn_data_lake.substrate.lake_engine import retention_of

        learned_reclaimable = retention_of(_T_FACT[0])[1]
        if learned_reclaimable and len(learned) > self.learned_cap:
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
    # ---- L-R7 (specs/022 L6, 2026-09-06): THE LABEL REFUSES THE CAP -------------------------------------
    # This block asserted the capping behaviour above until L6, which is what makes it a reachable red
    # rather than a decoration: with _T_FACT IMMORTAL / reclaimable=False, a cap of ONE over three learned
    # records must now keep all three. Revert either half of the L6 change and this line fires.
    from snn_data_lake.substrate import lake_engine as _le

    assert _le.retention_of(0x01)[1] is False, (
        "0x01 must be non-reclaimable for this red to mean anything"
    )
    kept_all = LakeConsolidation(learned_cap=1).consolidate(lg, trace=trace)
    assert kept_all == lg, f"a cap dropped a LEARNED record of a non-reclaimable class: {kept_all}"

    # CONTROL 1 (not the method): the REGISTRY decides, not this class. Flip 0x01 to reclaimable and the
    # same call caps again, with the use-it-or-lose-it ranking intact -- so the red above is the label
    # doing work, not the policy having been disabled.
    _was = _le.RETENTION[0x01]
    _le.RETENTION[0x01] = (_was[0], True, _was[2])
    try:
        v1 = LakeConsolidation(learned_cap=1).consolidate(lg)  # recency -> the NEWEST
        v2 = LakeConsolidation(learned_cap=1).consolidate(
            lg, trace=trace
        )  # utility -> the RECALLED one
    finally:
        _le.RETENTION[0x01] = _was
    assert v1 == [lg[2]], f"v1 (recency) should keep the newest: {v1}"
    assert v2 == [lg[0]], (
        f"v2 (use-it-or-lose-it) should keep the RECALLED record, not the newest: {v2}"
    )
    assert _le.RETENTION[0x01] == _was, "the control must restore the registry it borrowed"

    # CONTROL 2 (not the method): compaction still happens in the same pass. The re-derivable budget binds
    # under the per-record exemption RETENTION[0x01]'s reason declares -- a change that simply stopped
    # consolidating would pass the red above and fail here.
    _bulk = [
        {"cue": ["w", str(i)], "answer": f"w{i}.", "src": "wordnet:noun"} for i in range(5)
    ] + [{"cue": ["kept"], "answer": "learned.", "src": "chat:user"}]
    _capped = LakeConsolidation(rederivable_budget=2, learned_cap=1).consolidate(_bulk)
    assert sum(1 for r in _capped if _src_class(r) == "re-derivable") == 2, _capped
    assert _bulk[-1] in _capped, (
        "the learned record must survive a pass that reclaims re-derivable bulk"
    )
    print(
        "  [OK] L-R7: the IMMORTAL label refuses the learned cap (3 of 3 kept at cap=1); flipping 0x01 to "
        "reclaimable caps again with the recall ranking intact; re-derivable bulk is still reclaimed 5 -> 2"
    )
    assert trace.get(lg[0]) == 5 and trace.get(lg[1]) == 0, "recall trace counts wrong"
    # prune_to drops trace entries for records no longer in the lake, and keeps the live ones.
    # REACHABLE RED, constructed and run (P8/F11) -- 2026-08-04b. The previous form pruned a record
    # whose count was ALREADY 0, so `get(lg[1]) == 0` held whether or not prune_to did anything:
    # MEASURED, the assert passed with prune_to replaced by a no-op. lg[1] is therefore bumped FIRST,
    # so the drop is observable -- a no-op prune_to leaves 3 here and the assert fires.
    # A comment here used to say the durable round-trip was "not testable from this module today",
    # because the trace's durable path was the sidecar. It is testable now and it is tested below,
    # in a SEPARATE PROCESS -- which is the only form of the claim P4b accepts.
    trace.bump(lg[1], by=3)
    assert trace.get(lg[1]) == 3, "setup: the record about to be pruned must carry a nonzero count"
    trace.prune_to([lg[0]])
    assert trace.get(lg[1]) == 0 and trace.get(lg[0]) == 5, "prune_to must keep live, drop gone"
    print(
        "  [OK] v2 use-it-or-lose-it: a RECALLED old record beats a newer never-recalled one under the cap"
    )

    # ---- P4b: the trace is IN the one lake, proven ACROSS A PROCESS BOUNDARY -------------------
    # "It persists" is a claim about a NEW process. This one writes the trace here, EXITS nothing --
    # it hands the lake to a child interpreter that has never seen this object -- and the child has
    # to rebuild the same counts from `brain_lake.bin` alone. An in-process re-read would pass
    # against a pure RAM store and is what `retained() == 1.000` looked like over an empty lake.
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prior_home = os.environ.get("NORD_HOME")  # restored, never popped: an in-process gate run
    work = tempfile.mkdtemp(prefix="lake_consol_home_")  # shares one interpreter across modules
    legacy_home = tempfile.mkdtemp(prefix="lake_consol_legacy_")  # bound BEFORE the try: the
    try:  # finally cleans both, and a failure above must not raise NameError over the real one
        os.environ["NORD_HOME"] = work
        from snn_data_lake.substrate.home import KnowledgeLog, state_path

        rec = {"cue": ["one", "lake"], "answer": "one durable lake for the whole brain."}
        KnowledgeLog().append(rec)  # a FOREIGN (FACT) record written BEFORE the trace
        durable = RecallTrace()
        assert durable.get(rec) == 0, (
            "a fresh home must start with an empty trace, not a fabricated one"
        )
        for _ in range(3):
            durable.bump(rec)
        assert durable.save() is True, "the first save of a nonempty trace must write a record"
        assert durable.save() is False, "an unchanged trace must not be re-appended to the one lake"

        child = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            f"import os; os.environ['NORD_HOME'] = {work!r}\n"
            "from snn_data_lake.sleep.lake_consolidation import RecallTrace\n"
            "from snn_data_lake.substrate.home import KnowledgeLog\n"
            f"rec = {rec!r}\n"
            "t = RecallTrace()\n"
            "assert t.get(rec) == 3, ('recall count lost across the process boundary', t.get(rec))\n"
            "assert KnowledgeLog().count() == 1, 'a _T_STATE write dropped a FACT record'\n"
            "print('  [OK] child process rebuilt count=3 from brain_lake.bin alone; 1 FACT intact')\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        assert rc == 0, f"a SEPARATE process could not read the trace back from the lake (rc={rc})"
        assert not os.path.exists(state_path("recall_trace.json")), (
            "the trace must go IN brain_lake.bin -- no recall_trace.json sidecar may be created"
        )
        # An `assert os.path.exists(knowledge_path())` stood here and was DELETED, not weakened
        # (P8 F11): the `KnowledgeLog().append(rec)` eight lines up creates that file
        # unconditionally, so no constructible input makes it fire, and no mutation reaches it --
        # every neuter that could leave the lake uncreated trips an earlier assert first. Measured.
        # `closed_loop.py` keeps the same check because there it IS reachable (nothing else in that
        # self-check writes to the lake) and the red was constructed and run.

        # sleep consolidation rewrites the FACT side; the trace is a foreign record and must survive.
        klog = KnowledgeLog()
        klog.consolidate(list(klog.replay()))
        assert RecallTrace().get(rec) == 3, "sleep consolidation of the FACT side dropped the trace"

        # MIGRATION, not deletion: a pre-P4b sidecar in a fresh home is imported and kept on disk
        # as `<name>.imported`, so a lake reset cannot import it a second time.
        os.environ["NORD_HOME"] = legacy_home
        with open(os.path.join(legacy_home, "recall_trace.json"), "w", encoding="ascii") as f:
            f.write(json.dumps({record_key(rec): 9}))
        assert RecallTrace().get(rec) == 9, "a pre-P4b recall_trace.json sidecar was not imported"
        assert os.path.exists(os.path.join(legacy_home, "recall_trace.json.imported")), (
            "the legacy sidecar must be kept on disk, renamed inert -- migrating is not deleting"
        )
        print("  [OK] pre-P4b recall_trace.json imported into the lake; the file is kept, renamed")
    finally:
        if prior_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prior_home
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(legacy_home, ignore_errors=True)

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
    print(
        "The trace itself lives IN brain_lake.bin under _T_STATE (P4b), proven by a SEPARATE process"
    )
    print(
        "rebuilding it from disk, and it survives the FACT-side sleep rewrite as a foreign record."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
