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

WHAT IS AND IS NOT BUDGETED, decided by COUNTING FIRST (counted 2026-08-16: 231 tracked .md,
875,704 words -- dated because it is the census that justified this scoping, not a live figure;
it reads 280 tracked .md today):

    NOT budgeted -- HISTORICAL RECORDS, where growing is CORRECT:
      CHANGELOG.md (76,624 w)            an entry describes a tree that existed
      specs/** (280,945 w, 138 files)    what was true at a HEAD
      docs/coverage/** (58,691 w)        evidence at a measurement date
      research/**/md/**                  vendored papers, not this project's prose
    NOT budgeted -- invisible to a git-based gate:
      CLAUDE.md    UNTRACKED (gitignored). A ceiling on it would report MISSING in every clone.

Budgeting the records would be the allowlist-shaped failure that killed the broad citation rule at
475 findings. The budgeted set is the standing documents listed in BUDGETS below -- a work
list. The count is deliberately NOT written here: this sentence said FIVE while BUDGETS held
seven, because a ceiling is added without the prose being reread. A literal that must be kept in
step with a table it sits above will drift again; the table is the count.

  python -m snn_data_lake.gates.doc_budget_gate            # BARE = ASSERT: exit 1 on findings
  python -m snn_data_lake.gates.doc_budget_gate --assert   # same (kept for CI-spelling parity)
  python -m snn_data_lake.gates.doc_budget_gate --self-check

BARE = ASSERT since the US1 policy audit (B7, 2026-08-25). The bare form -- the ONLY form
anything automated ran (run_selfchecks registers modules with no argv) -- printed `gate FAIL`
and exited 0, so SECURITY.md sat 63.9% over its ceiling through three green batteries and
POLICIES.md's enforcement table asserted `bare = assert` about a gate that had never done it.
The identical defect was found and fixed for dependency_ledger (ci.yml records it: `a gate
that cannot fail is not a gate; it is a log line`) and for gradient_claim_gate; this module
never received the fix. Siblings dormancy_gate and method_dormancy_gate carry no flag at all.
"""

import os
import sys

# distinct-from: reasoning/arc_diagnostic.py -- its "node budget" bounds how many GRAPH NODES a
#   synthesis search may reach at a depth; this bounds WORDS in a document. distinct-from:
#   plasticity/r_stdp.py and infra/region_wiring.py -- those count CPU WORKERS/THREADS to
#   size parallelism. distinct-from: hippocampus/sleep_replay.py -- returns an experiment's measured
#   numbers. distinct-from: readout/laminar_local_readout.py -- constructs REDs for its OWN claims.
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
    # LOWERED 28150 -> 12100 on 2026-08-30 (owner directive: POLICIES.md is a CONSTITUTION),
    # in the SAME commit as the split: every dated measurement, census, correction narrative
    # and incident-forensics block relocated BYTE-EXACT to docs/POLICIES_EVIDENCE.md under
    # "POLICIES.md constitution split"; rules, honesty guards, standing verdicts and
    # derivation pins stay. Measured 11,957 by this gate's counter; ceiling = measured + ~140.
    # THIS LOWERED CEILING IS THE ENFORCEMENT of the constitution scope rule stated at the
    # top of POLICIES.md -- status re-accumulating there reds the build instead of rotting.
    # The 2026-08-18 consolidation debt (e-prop block, "after which this ceiling comes back
    # DOWN") is hereby DISCHARGED. New gate rows still grow the enforcement table BY DESIGN;
    # raises stay minimum-viable and stated per raise. RE-ANCHORED 12100 -> 12350 in the same
    # split commit: the split's adversarial review panel (5 lenses, 22 agents) confirmed six
    # dropped RULE clauses (an owner-adjudication obligation, two do-not-re-propose
    # prohibitions, an exemption holding condition, and guard clauses) and they were restored
    # -- rule text, not status; measured 12,294 after restoration plus a final timelessness
    # pass (flat status assertions converted to conditional/derivable forms), ceiling =
    # measured + ~56, minimum-viable.
    # RAISED 12800 -> 13906 on 2026-09-25 by the MEASURED delta (12,723 -> 13,829 words, +1,106, this
    # gate's own count): the owner's engine directive P4d (recorded PROVISIONAL as a disclosed session
    # call; specs/022 evidence/engine-structure-2026-09-25.md) and its pointers -- the P2 corollary's
    # owner exception, the e-prop transport distinction, P3's ANATOMY IS DATA, two P0 avenue rows and
    # two enforcement-table rows. The 77 words of pre-raise headroom are carried, not spent. Stated
    # because it is owner-visible: the owner LOWERED this ceiling on 2026-08-30, and this raise is about
    # 2.5x the largest precedent (+450 below).
    # RAISED 13906 -> 14028 on 2026-09-25 by the MEASURED delta (13,829 -> 13,951, +122), the docs panel fold's
    # null rule: one sentence, verbatim in P4d and in the e-prop invariant's engine pointer, replacing four
    # disagreeing lapse wordings, with P4d's red arms matched to the plan's G-R9. Owner-visible.
    # RAISED 14028 -> 14083 on 2026-09-25 by the MEASURED delta (13,951 -> 14,006, +55), the fold's honest labels:
    # the label vocabulary marked a session call and widened so the scaffold refusal can fire, P4d's disclosure
    # covering its wording, the P2 corollary's exception and the e-prop distinction marked session calls. The
    # pre-fold headroom (77) stays carried, so every word this fold adds is a stated raise.
    # RAISED 14083 -> 14116 on 2026-09-25 by the MEASURED delta (14,006 -> 14,039, +33), the owner's decisions the
    # arc had dropped: the reorganization clause restored to P4d's quoted source and stated in its body (plan G8),
    # and "Spiking only can win" recorded in P4d.
    # RAISED 14116 -> 14178 on 2026-09-25 by the MEASURED delta (14,039 -> 14,101, +62), the fold's policy
    # coherence: the stop rule's incumbent named (P2), the solved predicate's P1 form, "certified" dropped, one
    # tuning rule per readout JOB, the engine marked a directive the G rows build, P4d in the enforcement table's
    # review cell, the sleep field moved to the saved record, and the engine-works red's enforcement after G9.
    # RAISED 14178 -> 14276 on 2026-09-29 by the MEASURED delta (14,168 -> 14,266, +98), owner-approved: the owner
    # chose the fix-forward option whose stated scope included a small doc-budget raise (the words are the
    # session's option text, not the owner's). The raise pays for the 2026-09-29 audit's P4d attribution fix -- the owner's two 2026-09-27 sentences quoted
    # verbatim, the operational reading labelled a session call with its evidence, the (P2)/lake-record sleep
    # bullet 73b1f992 had dropped restored, and "sleep-plasticity" added to Not claimed. The 10 words of
    # pre-raise headroom are carried, not spent.
    "docs/POLICIES.md": 14276,  # EARLIER: raised +450 on 2026-09-02 for the owner's batched-battery amendment and the
    # execution-record note that lands in the same arc; NOT minimum-viable on purpose (the headroom is
    # for that note), a disclosed departure from the +~56 convention below
    # Superseded history of the 28150 ceiling, kept for the ratchet record:
    # RAISED 28083 -> 28150 on 2026-08-26 (specs/022 T046), in the
    # SAME commit as the growth: the policy_manifest's unclaimed-gate ratchet (baseline EMPTY,
    # may never grow) requires every runnable gate NAMED in this file, and T046 lands
    # gates/growth_coverage_gate.py -- one enforcement-table hygiene row (measured 68
    # words by this gate's counter -- the draft comment guessed ~45 -- at 8 words of
    # pre-edit headroom). The enforcement table grows one row per gate BY DESIGN; this
    # ceiling polices PROSE, and prose relocation stays the remedy for prose growth. NOTE
    # the working convention has tightened: the 2026-08-18 re-anchor granted measured+100,
    # this raise is minimum-viable (single-digit headroom), stated rather than implied.
    # Previous adjustment: LOWERED from 29750 on 2026-08-18 (specs/021 US4):
    # the two "Real past incidents" forensics sections (P7 + P8, 1,853 words by this gate's
    # own counter) relocated BYTE-EXACT to docs/POLICIES_EVIDENCE.md with pointers left in
    # place; ceiling re-anchored to measured + 100. HONEST RESIDUE: the e-prop invariant
    # block's own consolidation (the debt named above) remains OPEN -- the T024 count showed
    # the SC-006 target reachable without touching freshly-stabilized text, so it was not.
    # RAISED 2002 -> 2650 on 2026-08-19 (specs/022 T003), in the SAME commit as the growth: the
    # e-prop consolidation debt (named OPEN at the 021 lowering above) is paid by relocating the
    # mlp_q route-rejection + gf.py adjudication narratives here byte-exact. The raise is the
    # relocation's other half -- POLICIES.md's ceiling stays LOWERED and gains real headroom.
    # RAISED 2650 -> 3550 on 2026-08-25 (US1 policy audit), in the SAME commit as the growth:
    # POLICIES.md had sat 2.4% OVER its ceiling since T039f with nothing able to say so (the
    # audit's B7); the discharge relocates the three P5 GPU measurement blocks (crossover
    # 07-28, target-scale 08-03, CUDA graphs 07-28 -- 911 words by this gate's counter) here
    # byte-exact with a pointer. Words conserved, moved not grown; POLICIES.md returns under
    # its LOWERED ceiling with real headroom.
    # RAISED 3550 -> 19800 on 2026-08-30, the other half of the constitution split above:
    # 29 blocks (1,003 lines) relocated BYTE-EXACT from POLICIES.md. Words conserved across
    # the pair (31,597 -> 31,552 total), moved not grown. Measured 19,595; ceiling =
    # measured + ~200. This file is the DATED-EVIDENCE ledger; unlike POLICIES.md it grows
    # whenever evidence relocates here, and each growth lands with its relocation commit.
    # RAISED 19800 -> 19828 on 2026-09-25 by the MEASURED delta (19,772 -> 19,800, +28), the docs panel fold: the
    # P4d incident block states plainly that none of its three is an incident P4d would have caught, as the
    # adding-a-policy bar requires, and what P4d earns its place on instead.
    "docs/POLICIES_EVIDENCE.md": 19828,
    # CONVENTION NOTE (T051b-iv panel): the module docstring's "+5% over measured" rule is the
    # RETIRED 2026-08-16 anchor; the governing practice since T046 is MINIMUM-VIABLE raises
    # (measured + 0, stated per raise), so 0.0%-headroom rows are deliberate and the
    # TIGHTENABLE report is structurally quiet for them.
    # RAISED 2056 -> 2212 on 2026-09-24: G1(a)'s row 13, the recorded-stream replay with a causally lagged gate
    # signal (FR-031; T109) -- measured 2056 -> 2212, +156 words (str.split); measured + 0.
    # RAISED 1919 -> 2056 on 2026-09-24: G0(c)'s row 12, the restart checkpoint and lake-sourced authority its
    # pre-registration promised (FR-031; T108) -- measured 1919 -> 2056, +137 words (str.split); measured + 0.
    # RAISED 1783 -> 1919 on 2026-09-24: G0(b)'s row 11, the confirmed-clarification rule its pre-registration
    # promised (FR-031; T108) -- measured 1783 -> 1919, +136 words (str.split); measured + 0, the minimum-viable raise.
    # RAISED 1680 -> 1783 on 2026-09-24: G0(a)'s row 10, the provenance rule its pre-registration promised (FR-031;
    # T108) -- measured 1680 -> 1783, +103 words (str.split, this gate's own count); measured + 0, the minimum-viable
    # raise.
    # RAISED 1644 -> 1680 on 2026-09-16: T080's MECHANIZED line under 6 (measured 1644 -> 1680, +36 words;
    # measured + 0, the minimum-viable raise, the same rule as the 2026-09-15 raise below).
    "docs/INVENTION_REGISTER.md": 2212,  # RAISED 1618 -> 1644 on 2026-09-15: T079's MECHANIZED line under 6
    # (measured 1618 -> 1644, +26 words; measured + 0, the minimum-viable raise). EARLIER:
    # RAISED 1595 -> 1618 on 2026-09-15: T077's MECHANIZED line under
    # S6 (FR-031 requires a register row per landed P8 invention; the row is two lines, condensed twice
    # to 23 words). Measured + 0, the same minimum-viable raise as the one below.
    # EARLIER: RAISED 1543 -> 1595 on 2026-09-09: row 1's DELETED ON
    # ITS NULL line (T068 LOSS -- the paging invention was measured at a 35.2x tick cost against a
    # 3.0x bound fixed before the run and removed). Measured + 0, the same minimum-viable raise as
    # every one before it. The note is 4 lines: the disposition, the date, the evidence pointer and
    # the tag carrying the deleted code -- the numbers stay in evidence/s5-paging-null.md, because
    # this file is the index and never the measurement. A deletion COSTS words here rather than
    # saving them, which is the register working as designed: the row stays, the mechanism goes.
    # (superseded reason, kept for the trail) RAISED 1517 -> 1543 on 2026-09-08: T066's MECHANIZED
    # line on row 1 (measured + 0, the same minimum-viable raise as every one before it)
    # (superseded reason, kept for the trail) RAISED 1380 -> 1517 on 2026-09-03, SAME DAY as the raise
    # below and in the same arc: the W3 review panel found row 9's RED field overstated (7 criteria /
    # 8 mutants, one of them vacuous) and its NULL field stating the EXPERIMENT's revert condition
    # instead of the instrument's own deletion condition, which is what P8 clause 5 asks for. The
    # corrected row carries the real counts (11 criteria, 22 mutants), the disclosure of what the
    # panel found, and both nulls. A second raise inside one arc is the ceiling working: each one is
    # an owner-visible admission in the diff, which is exactly why 0.0% headroom is the design.
    # (superseded reason, kept for the trail) RAISED 1152 -> 1380 on 2026-09-03 (specs/022 Phase 1 W3),
    # in the SAME commit as the growth: invention #9, the two-chain world-red instrument. This
    # register is a per-invention list that grows by construction -- FR-031 names eight and W3 is
    # the first of them to land, with L1b, L2, T066, T073, T077 and T078 still owed -- so the
    # ceiling tracks landings rather than bounding them; what it bounds is the WORDS PER ROW. The
    # row was first drafted at 329 words (2.5x its neighbours) and CONDENSED to 228 before this
    # raise, so 101 of the 329 were paid by editing, not by the ceiling. 228 words, minimum viable
    # raise (measured + 0), the T046/T047/T050/T051a practice -- the 0.0%-headroom ceiling is the
    # design: it forces exactly this deliberate decision instead of absorbing drift silently.
    # (superseded reason, kept for the trail) RAISED 1124 -> 1152 on 2026-08-26 (specs/022 T051b-iv),
    # in the SAME commit as the growth: row 3's MECHANIZED line gains the iv sandwich-merge
    # entry AND the iv panel's P2 clause (merge-must-reduce NOT discharged: two incumbents
    # before and after), and its no-tissue parens shrink to T053 only. 28 words, minimum
    # viable raise (measured + 0), the T046/T047/T050/T051a practice -- the iii panel
    # flagged the 0.0%-headroom ceiling as forcing exactly this deliberate decision, which
    # is the ceiling doing its job, not a defect.
    # Prior raise record: 1047 -> 1124 (T051a),
    # in the SAME commit as the growth: row 3 gains its MECHANIZED line (the composer kernel
    # landed) AND pays two debts the T051a panel found unpaid -- the DISCLOSURE correction
    # (arbitration order is bare insertion order, NOT learned/recorded as the row claimed;
    # the learned form now owed, undated) and the ANTI-RIGGING statement (live readers zero;
    # recoverable from parts). 77 words, minimum viable raise (measured + 0), the
    # T046/T047/T050 practice. (Prior raise: 1047 at T047 for row 4's A/B numbers.)
    # Previous: specs/022 T002: the eight P8 invention debt packages
    # (index only -- measurements live in specs/022 evidence/). Measured 844 at landing; ceiling
    # = measured + ~106 so a ninth invention forces a deliberate ceiling raise with its reason here.
    "README.md": 2264,  # 2,156
    "docs/ANATOMY.md": 1968,  # 1,874
    # RAISED 1065 -> 1067 on 2026-09-25 by the MEASURED delta (1,046 -> 1,048, +2), the docs panel fold: the
    # entry count states its rule ("amendments uncounted"), which the policy manifest's sub-clause census needs.
    "AGENTS.md": 1067,  # 1,014
    # Added 2026-09-22 with the file (its panel's policy-honesty and cold-read lenses): the
    # GitHub-only rendering of the operator protocol claims to carry no project state, and this
    # ceiling (measured + ~5%) is what makes status re-accumulating there red instead of silent.
    # RAISED 9650 -> 9814 on 2026-09-25 by the MEASURED delta (9,636 -> 9,800 words, +164, this gate's own
    # count): the SYNC DUTY for the owner's engine directive (POLICIES P4d) -- the design-under-test
    # paragraph, P4d among the binding sections, REGIONS named as code packages, and the re-proposal
    # guard's engine rows (one refused row deleted, the owner having overturned it). The 14 words of
    # pre-raise headroom are carried, not spent.
    # RAISED 9814 -> 9820 on 2026-09-25 by the MEASURED delta (9,800 -> 9,806, +6), the docs panel fold: the
    # reservoir is spiking (owner decision "Spiking only can win") and G8 reorganizes the code packages (owner).
    # RAISED 9820 -> 9821 on 2026-09-25 by the MEASURED delta (9,806 -> 9,807, +1): "e.g." restores the owner's
    # tuning rules as examples, not a closed list.
    "docs/AGENT_COLD_START.md": 9821,  # 9,189 at introduction
    # 488 -> 713 on 2026-09-09 (measured + 0): f441949 added 225 words of OPERATIVE prohibitions --
    # the standing conditions for a third-party agent (Codex) on this tree (read-only only, never
    # the lake / the envelope / ~/.claude, Codex output is data, the untracked-file check, a
    # .codex/config.toml is an incident, the Stop gate off). A list of prohibitions is not
    # condensable narrative -- cutting it to fit would delete conditions, the trade this ceiling
    # exists to prevent; .claude/ is untracked here, so they must live in a tracked standing
    # document; and 518b94c's remedy for this same file (800 -> 488) condensed a narrative, which
    # does not apply. The raise lands in its own commit because f441949 had already landed
    # (rewriting it would rewrite the arc's record commit for nothing) -- the deviation stated.
    "SECURITY.md": 713,  # 713
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
    # BARE = ASSERT (US1 audit B7): `do_assert` is accepted for CI-spelling parity but the exit
    # code no longer depends on it -- the registered bare form was structurally incapable of
    # failing, which is how three OVER findings survived green batteries.
    del do_assert
    return 1 if failed else 0


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
