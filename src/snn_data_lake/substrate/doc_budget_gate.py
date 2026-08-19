"""
Doc budget gate -- a word ceiling on the STANDING documents, so prose growth is visible.

WHY. This project's documents are its instruments: `docs/POLICIES.md` decides what may be built and
is read before every task. It has no size control, and it grows every time anything is corrected --
1,701 lines on 2026-08-07, 1,966 on 2026-08-16, with roughly 100 of those added in a single session.
Nobody decided that; it accumulated. The failure this guards is recorded in this repo's own words --
*"prose density is a defect surface"*, from two audit rounds whose blockers were nearly all ONE
defect: a claim outliving what it described. A document nobody can hold in their head is one where a
stale sentence hides.

ADOPTED 2026-08-16 from DeepSeek Harness's `verify-doc-budgets` (MIT). Three of its design choices
are kept because each solves a real problem: count WORDS (`wc -w`) so reformatting is not a
violation; budget only an OPT-IN manifest of standing docs rather than every file; and FAIL when a
budgeted file is MISSING, so a rename cannot silently drop a ceiling. Its "ceilings ratchet down
with headroom" idea survives as the TIGHTENABLE report. What was deliberately NOT adopted from that
project is its "never run the full suite locally, CI owns coverage" convention, which contradicts
this repo's one-change-per-commit-with-the-full-gate-green invariant -- importing a convention
wholesale is how an invariant gets relaxed by accident.

WHAT IS AND IS NOT BUDGETED, decided by COUNTING FIRST (231 tracked .md, 875,704 words):

    NOT budgeted -- HISTORICAL RECORDS, where growing is CORRECT:
      CHANGELOG.md (76,624 w)            an entry describes a tree that existed
      specs/** (280,945 w, 138 files)    what was true at a HEAD
      docs/coverage/** (58,691 w)        evidence at a measurement date
      research/**/md/**                  vendored papers, not this project's prose
    NOT budgeted -- invisible to a git-based gate:
      CLAUDE.md    UNTRACKED (gitignored). A ceiling on it would report MISSING in every clone.

Budgeting the records would be the allowlist-shaped failure that killed the broad citation rule at
475 findings. The budgeted set is FIVE standing documents -- a work list.

  python -m snn_data_lake.substrate.doc_budget_gate            # report
  python -m snn_data_lake.substrate.doc_budget_gate --assert   # exit 1 over budget or missing
  python -m snn_data_lake.substrate.doc_budget_gate --self-check
"""

import os
import sys

# distinct-from: reasoning/arc_diagnostic.py -- its "node budget" bounds how many GRAPH NODES a
#   synthesis search may reach at a depth; this bounds WORDS in a document. distinct-from:
#   basal_ganglia/r_stdp.py and substrate/region_wiring.py -- those count CPU WORKERS/THREADS to
#   size parallelism. distinct-from: hippocampus/sleep_replay.py -- returns an experiment's measured
#   numbers. distinct-from: cortex/laminar_local_readout.py -- constructs REDs for its OWN claims.
#   All five were read; none measures documentation, and there is nothing to reuse either way.

_HERE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

#: path -> word ceiling. Set 2026-08-16 at roughly +5% over the measured count, so the gate lands
#: green and can only be TIGHTENED. A ceiling is not a target: when a document drops well under
#: its ceiling, lower it in the same change -- the TIGHTENABLE report exists to make that visible.
#: Raising one is allowed and must say WHY in the commit, never as a reflex to clear a red.
BUDGETS = {
    # RAISED 29100 -> 29250 on 2026-08-18, in the SAME commit as the growth, with the reason here
    # because a ceiling raised quietly is a ratchet that has stopped ratcheting. specs/020's Phase 8
    # amendment had to CORRECT ITSELF in place: it stated INSTRUMENT_FALSIFIED is triggered by a rule
    # missing the bar, when `gradient_claim_gate::verdict_for` keys it solely on high coherence
    # invalidating the null -- an inversion that licensed "left UNJUDGED" for any rule that merely
    # FAILS. Carrying both the corrected rule AND the measured refutation costs words, and ~90 were
    # already condensed away first. Cutting the remaining 18 would have deleted a measurement, which
    # is the trade this ceiling exists to prevent, not to force.
    # RAISED AGAIN 29250 -> 29500 on 2026-08-18, same commit as the growth. A third adversarial
    # review of the specs/020 amendment returned 14 findings, FOUR of them BLOCKERs, and every
    # one replaced a FALSE statement in the authoritative file: that INSTRUMENT_FALSIFIED could
    # be read as a token about the RULE; that the transport gate is a ratchet (it asserts ZERO);
    # that a train-tier exemption existed (`training.cli` is an `_ENTRYPOINT`, so the tier is
    # REACHED); and that `dormancy_gate.reachable()` could decide scope (it masks to
    # BRAIN_MODULES). Carrying the corrected rules plus the OPEN HOLE they expose costs words.
    # ~200 were condensed away and the trainer enumeration RELOCATED to the spec record first;
    # the remainder is operative rule text, and deleting it to fit would trade correctness for
    # word count -- the trade this ceiling exists to prevent, not to force.
    # RAISED A THIRD TIME 29500 -> 29750 on 2026-08-18 (fourth adversarial round, 20 findings,
    # 4 BLOCKERs -- among them: a test DOES detect governed-tier reachability flips, the 22 was
    # 21, limit (c) had rotted, and the constitution restated its way into two contradictions).
    # THREE raises in one day means this ceiling is not ratcheting for POLICIES.md; the standing
    # remedy, recorded here so it is not lost: the e-prop invariant block now carries FOUR
    # generations of amendment text and owes a CONSOLIDATION pass -- one statement per rule,
    # correction narratives moved to the spec record -- after which this ceiling comes back DOWN.
    "docs/POLICIES.md": 28083,  # LOWERED from 29750 on 2026-08-18 (specs/021 US4):
    # the two "Real past incidents" forensics sections (P7 + P8, 1,853 words by this gate's
    # own counter) relocated BYTE-EXACT to docs/POLICIES_EVIDENCE.md with pointers left in
    # place; ceiling re-anchored to measured + 100. HONEST RESIDUE: the e-prop invariant
    # block's own consolidation (the debt named above) remains OPEN -- the T024 count showed
    # the SC-006 target reachable without touching freshly-stabilized text, so it was not.
    "docs/POLICIES_EVIDENCE.md": 2002,  # relocated forensics; ceiling = measured + 60
    "docs/INVENTION_REGISTER.md": 950,  # specs/022 T002: the eight P8 invention debt packages
    # (index only -- measurements live in specs/022 evidence/). Measured 844 at landing; ceiling
    # = measured + ~106 so a ninth invention forces a deliberate ceiling raise with its reason here.
    "README.md": 2264,  # 2,156
    "docs/ANATOMY.md": 1968,  # 1,874
    "AGENTS.md": 1065,  # 1,014
    "SECURITY.md": 488,  # 465
}

_TIGHTEN_AT = 0.15  # flag a ceiling sitting more than this fraction above its document


def count_words(text):
    """`wc -w` equivalent: whitespace-delimited tokens.

    WORDS, not bytes or lines, and the choice is load-bearing: a `ruff format` reflow or a
    line-ending change moves a byte or line count without a word of prose being added, and a gate
    that fires on reformatting is a gate that gets switched off.
    """
    return len(text.split())


def measure(budgets=None, root=None):
    """[(path, words, ceiling, status)] with status 'ok' | 'OVER' | 'MISSING' | 'BAD'."""
    budgets = BUDGETS if budgets is None else budgets
    root = _ROOT if root is None else root
    rows = []
    for path, ceiling in sorted(budgets.items()):
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
            rows.append((path, -1, ceiling, "BAD"))
            continue
        try:
            with open(os.path.join(root, path.replace("/", os.sep)), encoding="utf-8") as fh:
                words = count_words(fh.read())
        except OSError:
            # MISSING is a FAILURE, never a skip. A renamed or deleted document must update this
            # manifest in the SAME change, or the ceiling silently applies to nothing -- the
            # staleness rule `dormancy_gate.EXEMPT` and the e-prop gate's EXEMPT both carry.
            rows.append((path, -1, ceiling, "MISSING"))
            continue
        rows.append((path, words, ceiling, "ok" if words <= ceiling else "OVER"))
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_assert = "--assert" in argv
    rows = measure()

    print("=" * 78)
    print("DOC BUDGET GATE -- word ceilings on the STANDING documents")
    print("=" * 78)
    print(f"{'status':<9}{'words':>8} /{'ceiling':>8}  {'headroom':>9}  path")
    tighten = []
    for path, words, ceiling, status in rows:
        if words < 0:
            print(f"{status:<9}{'--':>8} /{ceiling!s:>8}  {'--':>9}  {path}")
            continue
        head = (ceiling - words) / ceiling
        print(f"{status:<9}{words:>8} /{ceiling:>8}  {head:>8.1%}  {path}")
        if status == "ok" and head > _TIGHTEN_AT:
            tighten.append((path, words, ceiling))
    print()

    if tighten:
        # A ceiling far above its document is not a pass; it is an unenforced budget.
        print("  TIGHTENABLE -- these ceilings sit far above their documents. Lower them:")
        for path, words, ceiling in tighten:
            print(f"    {path}: {words} words under a {ceiling} ceiling")
        print()

    failed = [r for r in rows if r[3] != "ok"]
    for path, words, ceiling, status in failed:
        if status == "OVER":
            print(f"  OVER    {path}: {words} words exceeds {ceiling}. CONDENSE or RELOCATE.")
            print("          Raising the ceiling is allowed and must say WHY in the commit.")
        elif status == "MISSING":
            print(f"  MISSING {path}: budgeted but not on disk -- renamed or deleted?")
            print("          Update BUDGETS in the SAME change; a ceiling applying to nothing")
            print("          reads as coverage of a document nobody is watching.")
        else:
            print(f"  BAD     {path}: ceiling must be a positive integer, got {ceiling!r}")
    print("gate OK" if not failed else f"gate FAIL: {len(failed)} budget finding(s)")
    return 1 if (do_assert and failed) else 0


def _selfcheck():
    """P8 (1): construct the RED for every status this gate can report.

    It lands GREEN on the real tree, which is the shape that makes a gate untrustworthy unless its
    reds are shown REACHABLE -- so each is built against a constructed manifest here rather than
    waiting for a document to drift into one.
    """
    import contextlib
    import io

    print("DOC BUDGET GATE self-check")

    real = measure()
    assert real, "the manifest is empty -- a gate over nothing"
    bad_rows = [r for r in real if r[3] != "ok"]
    assert not bad_rows, f"the real tree is not green: {bad_rows}"
    print(f"  green  the real tree: {len(real)} budgeted document(s), every one under ceiling")

    over = measure(budgets={"docs/POLICIES.md": 10})
    assert over[0][3] == "OVER", f"the OVER red did not fire: {over}"
    print("  RED    a document above its ceiling")

    missing = measure(budgets={"docs/NO_SUCH_DOCUMENT.md": 100})
    assert missing[0][3] == "MISSING", f"the MISSING red did not fire: {missing}"
    print("  RED    a budgeted document not on disk (renamed/deleted without a manifest edit)")

    for bad_ceiling in (0, -5, "600", True):
        got = measure(budgets={"README.md": bad_ceiling})
        assert got[0][3] == "BAD", f"the BAD-ceiling red did not fire for {bad_ceiling!r}: {got}"
    print("  RED    a non-positive / non-integer ceiling (0, -5, '600', True)")

    # The failure must reach the EXIT CODE, not merely the printed report: a check whose red does
    # not change `rc` is a log line, which is the exact defect `~/.claude/gates.py` was built for.
    saved = BUDGETS.copy()
    try:
        BUDGETS.clear()
        BUDGETS["docs/POLICIES.md"] = 10
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main(["--assert"])
        assert rc == 1, f"--assert returned {rc} on an over-budget document; it must be 1"
    finally:
        BUDGETS.clear()
        BUDGETS.update(saved)
    print("  RED    --assert exits 1, not merely prints")

    print(f"self-check OK: 4 constructed reds, {len(real)} documents budgeted")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
