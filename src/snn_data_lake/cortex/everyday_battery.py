"""C0 -- the L12 audit's 20-item everyday battery, promoted from a session scratchpad into a
governed SCORER instrument (specs/022 Phase 4 row C0; pre-registration
`specs/022-unified-brain-arc/evidence/c0-everyday-battery.md`, sections 1-2.9, FROZEN).

WHAT THIS IS. The battery's only copy was `scratchpad/battery.py` of the 2026-09-01 audit session
(archived 2026-09-16 at `memory/archive/c0-2026-09-16/battery_L12_2026-09-01_session-5ef7f298.py`,
outside the repo -- a temp directory is not a store). It held 20 questions, 4 positive controls and
a reply-SHAPE classifier, and NO ANSWER KEY: its "0 correct" was a human read of 20 replies, so a
brain that answered all 20 correctly would have been classified `ANSWERED` whether right or wrong.
This module is that battery PORTED (P2: the questions and the four controls are verbatim, its five
shape classes map onto the seven here -- `ABSTAIN (metacog/2nd-order hold)` -> `ABSTAIN`;
`SELF-REPORT (content-blind)` -> `SELF_REPORT`; `HONEST GAP` and `CAPABILITY FALLBACK` ->
`FALLBACK`; `ANSWERED (symbolic DSL / KB)` -> `CORRECT` or `WRONG` BY THE KEY; `OTHER: [recalled
...` -> `UNRELATED_ROW`) plus the answer key 2.2 freezes, so the instrument can register a correct
answer at all. Nothing of the scratch battery survives beside this module.

WHAT IT SCORES. `cortex/brain.py::Brain.respond(q, human=True)` -- the chat lane.

P4b, AND IT IS THE REASON FOR THE PIN. `respond(..., human=True)` INGESTS the question into the one
durable lake. Twenty-four synthetic questions per run written into `brain_lake.bin` would be the
instrument contaminating the thing it measures, so the scorer NEVER runs on the live brain: every
brain here is `make_brain(seed=seed, persist=False)` inside `substrate.home.throwaway_home()`, and
the number reaches a surface only as a PINNED campaign reading that says so (the T080-T082 shape).

THE KEY IS A FLOOR, NOT A JUDGE (2.2). It recognises a short correct answer in plain words and will
MISS a correct answer phrased outside its patterns -- a false negative it can make, reported per
item with the reply text, never silently scored wrong. The false positive it must never make is a
reply that declines to answer scoring `CORRECT`; 2.5's reds are constructed against exactly that.

TWO INSTRUMENTS MUST AGREE ON ABSTENTION (2.3 rule 1, fact 5). The hold's reply NARRATES NUMBERS
("About 8 ticks ago ... goal 0 ... 0.64"), so a key such as `3` or `7` tested against the whole
reply can be satisfied by the narration. Abstention is therefore decided BEFORE any key is tested,
and by the ROUTE LABEL and the REPLY's PREAMBLE together. Exactly one of them is a DISAGREEMENT:
reported as `OTHER` with `disagree`, never resolved by picking one.

THREE DECISIONS THIS MODULE TOOK THAT THE FROZEN RECORD DID NOT SPELL OUT (disclosed, not hidden):
  1. THE SOMATIC SUFFIX BELONGS TO THE TRANSCRIPT, NOT TO THE CLASSIFIER. The L12 audit's `cls()`
     opened with `t = r["reply"].split(" (somatic")[0]`, and the 2026-09-16 orientation whose rows
     are `FIXTURE` recorded its replies the same way. 2.3's ordered rule never mentions it. So
     `run_seed` stores the reply with that suffix removed -- one definition of "the reply", the
     orientation's -- and `classify` implements 2.3 literally, with no second transform hidden
     inside it. This matters: an un-stripped somatic suffix carries digits, and a bare-digit key
     (`\b(3|three)\b`) can be satisfied by "valence 0.3" on a reply that answered nothing.
  2. `FALLBACK` uses the scratch battery's FULL predicate -- `honest gap`, `i can tell you my goal`,
     `not a general chatbot` (three substrings across its two classes). 2.3 rule 3 says "the scratch
     battery's two strings", which names its two CLASSES; porting the predicate they were decided by
     is the faithful read, and it is strictly wider, never narrower.
  3. `ITEMS`' question text is asserted equal to `FIXTURE`'s at import. The archive lives outside
     the repo, so the module cannot read it; the orientation that produced `FIXTURE` asserted its
     questions equal the archive's, and this assert carries that chain into the tree.

A MEASURED CORRECTION TO SECTION 1 FACT 2, kept here because a record is frozen and a reading is
not: fact 2 says "Class AND text are identical across the five seeds". The CLASS and the ROUTE are
-- all 100 item x seed cells and all 20 control cells agree -- and the TEXT is not: the hold's
state narration differs per seed (15 of 20 seed-1 replies and 14 of 20 seed-2 replies differ from
seed 0's byte for byte, the 5 `[recalled ...]` rows being the stable ones). Conjunct (h) is written
over CLASSES (2.4) and is unaffected; no conjunct here compares reply text ACROSS seeds, and the
one that compares text at all -- (c) -- compares one seed against itself.

# distinct-from: cortex/attention_schema.py, cortex/report_order_probe.py,
#   cortex/prior_drift_harness.py, thalamus/binding_instrument.py -- the same INSTRUMENT SHAPE this
#   tree uses (a CI/run_selfchecks module whose bare invocation runs constructed reds and whose
#   campaign is a flag, on a scratch brain under throwaway_home), over a different measurement:
#   those read the workspace, the report channel, a drift harness and binding; this scores CHAT
#   REPLIES against a frozen answer key. Read before writing this file; no scorer, no answer key
#   and no reply classifier exists in the tree (graph-exists, 2026-09-17: "NO EXISTING
#   IMPLEMENTATION found").
# distinct-from: cortex/reasoning_curriculum.py, cortex/vsa_reasoning.py -- lexical matches on
#   "reasoning" only: the curriculum GENERATES training items for sleep consolidation and
#   vsa_reasoning BINDS symbols; neither scores a chat reply and neither holds an answer key.
"""

from __future__ import annotations

import re
import sys
from typing import NamedTuple

# --------------------------------------------------------------------------------------------
# 2.1 (a) THE BATTERY -- the audit's 20 questions and 4 positive controls, VERBATIM, each with the
# answer key 2.2 freezes. Inside 2.2's table a cell's `\|` is a markdown-escaped `|` (regex
# alternation); the patterns are otherwise literal and are matched case-insensitively.
# --------------------------------------------------------------------------------------------
ABSTAIN = "ABSTAIN"
SELF_REPORT = "SELF_REPORT"
FALLBACK = "FALLBACK"
CORRECT = "CORRECT"
UNRELATED_ROW = "UNRELATED_ROW"
WRONG = "WRONG"
OTHER = "OTHER"
CLASSES = (ABSTAIN, SELF_REPORT, FALLBACK, CORRECT, UNRELATED_ROW, WRONG, OTHER)

#: 2.3 rule 1's two instruments: the route label's marker and the reply's preamble.
ABSTAIN_ROUTE_MARK = "abstain"
ABSTAIN_PREAMBLE = "I am not fully certain"
#: 2.3 rule 2: the content-blind faculty line.
SELF_REPORT_MARK = "faculty has learned"
#: 2.3 rule 3: the scratch battery's fallback strings (lower-cased match).
FALLBACK_MARKS = ("honest gap", "i can tell you my goal", "not a general chatbot")
#: 2.3 rule 5: a stored row spoken in place of an answer.
RECALLED_TAG = "[recalled"
#: 2.2: the ANSWER SPAN is the reply with a leading `[tag]` removed.
_TAG = re.compile(r"^\[[^\]]*\]\s*")
#: 2.3 rule 7 keeps the first 70 characters.
OTHER_HEAD = 70
#: The transcript strip the L12 audit and the orientation both applied (see decision 1 above).
SOMATIC = " (somatic"


class Item(NamedTuple):
    """One battery question with its frozen key and 2.2's two constructed replies."""

    id: int
    q: str
    accept: str
    refuse: str
    correct: str
    wrong: str


class Control(NamedTuple):
    """One positive control: a question the tree's own self-checks say the brain CAN answer."""

    id: str
    q: str
    accept: str
    refuse: str = ""


ITEMS = (
    Item(
        1,
        "If all bloops are razzies and all razzies are lubbers, are all bloops lubbers?",
        r"\byes\b",
        r"\bno\b",
        "[reasoned] yes, all bloops are lubbers.",
        "[reasoned] no, they are different things.",
    ),
    Item(
        2,
        "Anna is taller than Bo. Bo is taller than Cy. Who is shortest?",
        r"\bcy\b",
        "",
        "[reasoned] cy is shortest.",
        "[reasoned] anna is shortest.",
    ),
    Item(
        3,
        "A key is to a lock as a password is to what?",
        r"\b(account|login|computer|system|device|phone|website|site|email|app|network)\b",
        "",
        "[induced] an account.",
        "[induced] a hammer.",
    ),
    Item(
        4,
        "Tom has 5 apples and gives 2 to Ann. How many does Tom have?",
        r"\b(3|three)\b",
        "",
        "[reasoned] tom has 3 apples.",
        "[reasoned] tom has 7 apples.",
    ),
    Item(
        5,
        "If it rains the ground gets wet. The ground is not wet. Did it rain?",
        r"\bno\b",
        r"\byes\b",
        "[reasoned] no, it did not rain.",
        "[reasoned] yes, it rained.",
    ),
    Item(
        6,
        "What is heavier, a kilogram of feathers or a kilogram of steel?",
        r"\b(same|equal|neither|both)\b",
        "",
        "[reasoned] they weigh the same.",
        "[reasoned] the steel is heavier.",
    ),
    Item(
        7,
        "I put a ball in a box, closed it, and left the room. Where is the ball?",
        r"\bbox\b",
        "",
        "[reasoned] the ball is in the box.",
        "[reasoned] the ball is in the garden.",
    ),
    Item(
        8,
        "Sort these from smallest to largest: 9, 2, 7, 4.",
        r"2\D+4\D+7\D+9",
        "",
        "[reasoned] 2, 4, 7, 9.",
        "[reasoned] 9, 7, 4, 2.",
    ),
    Item(
        9,
        "What comes next: Monday, Tuesday, Wednesday?",
        r"\bthursday\b",
        "",
        "[induced] thursday.",
        "[induced] sunday.",
    ),
    Item(
        10,
        "If today is Friday, what day is it in three days?",
        r"\bmonday\b",
        "",
        "[reasoned] monday.",
        "[reasoned] sunday.",
    ),
    Item(
        11,
        "Which is the odd one out: dog, cat, hammer, horse?",
        r"\bhammer\b",
        "",
        "[reasoned] hammer -- the others are animals.",
        "[reasoned] horse.",
    ),
    Item(
        12,
        "Every bird has feathers. A penguin is a bird. Does a penguin have feathers?",
        r"\byes\b",
        r"\bno\b",
        "[reasoned] yes, a penguin has feathers.",
        "[reasoned] no, a penguin swims.",
    ),
    Item(
        13,
        "A train leaves at 3pm and takes 2 hours. When does it arrive?",
        r"\b5\s*(pm|p\.m\.?|o'?clock)|\bfive\b|\b17[:.]?00\b",
        "",
        "[reasoned] it arrives at 5pm.",
        "[reasoned] it arrives at 6pm.",
    ),
    Item(
        14,
        "What is 17 plus 25?",
        r"\b42\b|forty[- ]two",
        "",
        "[reasoned] 42.",
        "[reasoned] 41.",
    ),
    Item(
        15,
        "If x plus 3 equals 10, what is x?",
        r"\b(7|seven)\b",
        "",
        "[reasoned] x is 7.",
        "[reasoned] x is 13.",
    ),
    Item(
        16,
        "Two boxes weigh the same. Box A holds bricks, box B holds pillows. Which is heavier?",
        r"\b(same|equal|neither)\b",
        "",
        "[reasoned] neither -- they weigh the same.",
        "[reasoned] box a, the bricks.",
    ),
    Item(
        17,
        "Is a square always a rectangle?",
        r"\byes\b",
        r"\bno\b",
        "[reasoned] yes, every square is a rectangle.",
        "[reasoned] no, never.",
    ),
    Item(
        18,
        "Give me a reason why the sky looks blue.",
        r"scatter",
        "",
        "[reasoned] air scatters blue light more than red.",
        "[reasoned] it reflects the ocean.",
    ),
    Item(
        19,
        "What would happen if I dropped a glass on concrete?",
        r"\b(break|shatter|crack|smash)",
        "",
        "[reasoned] it would shatter.",
        "[reasoned] it would bounce and float.",
    ),
    Item(
        20,
        "Name something that is both a fruit and red.",
        r"\b(apple|strawberr|cherr|raspberr|tomato|pomegranate|cranberr|watermelon)",
        "",
        "[from what I learned] a strawberry.",
        "[from what I learned] a banana.",
    ),
)

CONTROLS = (
    Control("C1", "What is the next number in the series 2, 4, 8, 16?", r"\b32\b"),
    Control("C2", "What is the next number in the series 1, 4, 9, 16?", r"\b25\b"),
    Control("C3", "abc is to abd as ijk is to what?", r"\bijl\b"),
    Control("C4", "What is the capital of France?", r"\bparis\b"),
)

# --------------------------------------------------------------------------------------------
# THE PORT'S FIXTURE (2.4 conjunct (k) third arm and conjunct (h)): section 1 fact 2's seed-0
# reading, recorded 2026-09-16 by the orientation script on head `c8c07a0` under the protocol
# `PROTOCOL` names. These are DATA, frozen here so the port can be shown faithful without a brain.
# --------------------------------------------------------------------------------------------
PROTOCOL = (
    "make_brain(seed=<s>, persist=False) under substrate.home.throwaway_home(), 20 warm-up ticks, "
    "the 4 controls then the 20 items through Brain.respond(q, human=True); the reply is recorded "
    "with any ' (somatic' suffix removed (the L12 audit's and the orientation's transcript rule)"
)
WARM_TICKS = 20

#: The second-order access hold's reply on seed 0 -- identical on all 14 of its items, and the
#: NARRATION that makes 2.3 rule 1 necessary (it carries "0.64", "8", "0").
_FX_HOLD = (
    "I am not fully certain right now, but Right now awake off; also phase high at 0.64 and "
    "rising. About 8 ticks ago I was waking, pursuing goal 0, about as certain as now."
)
_FX_ABSTAIN_ROUTE = "Gf (second-order access hold: abstain)"
_FX_GC_ROUTE = "Gc (crystallized: memory/knowledge/state)"
_FX_FACULTY_ROUTE = "Gc (learned reasoning faculty; held-out 40%)"


class Fix(NamedTuple):
    """One recorded seed-0 cell: the observed route and reply, and the class fact 2 recorded."""

    id: int
    route: str
    reply: str
    cls: str


FIXTURE = (
    Fix(1, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(2, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(3, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(4, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(5, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(6, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(7, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(8, _FX_GC_ROUTE, "[recalled by association] 4's parity is even.", UNRELATED_ROW),
    Fix(9, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(10, _FX_GC_ROUTE, "[recalled by association] week's days is 7.", UNRELATED_ROW),
    Fix(11, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(12, _FX_GC_ROUTE, "[recalled by meaning] penguin's class is bird.", UNRELATED_ROW),
    Fix(13, _FX_GC_ROUTE, "[recalled by association] 2's parity is even.", UNRELATED_ROW),
    Fix(14, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(15, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(16, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(17, _FX_GC_ROUTE, "[recalled by association] square's sides is 4.", UNRELATED_ROW),
    Fix(
        18,
        _FX_FACULTY_ROUTE,
        "My transitive-reasoning faculty has learned to 40% held-out accuracy over 1 nights of "
        "sleep-consolidation -- genuine inference, never taught the answers. On an ordering it "
        "was never taught, it correctly infers the relation by transitivity.",
        SELF_REPORT,
    ),
    Fix(19, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
    Fix(20, _FX_ABSTAIN_ROUTE, _FX_HOLD, ABSTAIN),
)

#: Fact 2's histogram, the number the plan's C-R2 starts from.
FIXTURE_HISTOGRAM = {ABSTAIN: 14, UNRELATED_ROW: 5, SELF_REPORT: 1}

#: The four controls' seed-0 replies (fact 2: answered and correct on every seed -- 32; 25; ijl;
#: paris), kept so conjunct (p)'s arm is reachable without a brain as well as with one.
CONTROL_FIXTURE = (
    (
        "C1",
        _FX_GC_ROUTE,
        "[induced] 2 4 8 16 -> 32. the rule i synthesised for this problem at test time: "
        "constant ratio x2. not a stored fact.",
    ),
    (
        "C2",
        _FX_GC_ROUTE,
        "[induced] 1 4 9 16 -> 25. the rule i synthesised for this problem at test time: "
        "constant 2th-difference. not a stored fact.",
    ),
    (
        "C3",
        _FX_GC_ROUTE,
        "[induced] abc:abd :: ijk:? -> ijl. the rule i synthesised for this problem at test "
        "time: letter analogy. not a stored fact.",
    ),
    ("C4", _FX_GC_ROUTE, "[from what I learned] france's capital is paris."),
)

#: A non-abstain route, for scoring a CONSTRUCTED reply (no brain): rule 1 must decline on both
#: instruments, which is what a constructed answer-shaped reply is.
CONSTRUCTED_ROUTE = _FX_GC_ROUTE

# Decision 3: the archive is outside the repo, so this assert carries the orientation's own
# "question text differs from the archive's" check into the tree.
assert [i.id for i in ITEMS] == [f.id for f in FIXTURE] == list(range(1, 21))
assert len(CONTROLS) == len(CONTROL_FIXTURE) == 4


# --------------------------------------------------------------------------------------------
# 2.1 (b) / 2.3 THE CLASSIFIER -- the ordered rule, written before any number was read.
# --------------------------------------------------------------------------------------------
def answer_span(reply):
    """2.2's ANSWER SPAN: the reply with a leading `[tag]` removed. No other transform."""
    return _TAG.sub("", reply)


def key_hit(span, key):
    """Rule 4's predicate: the span matches `accept` and matches no `refuse` pattern."""
    if not re.search(key.accept, span, re.I):
        return False
    return not (key.refuse and re.search(key.refuse, span, re.I))


def classify(reply, route, key):
    """2.3's ORDERED rule -> `(cls, detail)`. No key is tested before rules 1-3 have declined.

    1. `ABSTAIN` iff the ROUTE label contains `abstain` AND the reply opens with the hold's
       preamble. Exactly ONE of the two is a DISAGREEMENT: `OTHER` with `disagree` -- two
       instruments must agree, and a disagreement is reported, never resolved by picking one.
    2. `SELF_REPORT` iff the reply carries the content-blind faculty line.
    3. `FALLBACK` iff the reply is the capability fallback or the honest-gap line.
    4. `CORRECT` iff the answer span matches `accept` and matches no `refuse` pattern.
    5. `UNRELATED_ROW` iff the reply opens with a `[recalled` tag.
    6. `WRONG` iff the reply opens with any other `[tag]` (answer-shaped, no key match).
    7. `OTHER` otherwise, with the first 70 characters kept.

    `detail` is `{}` for a decided class, `{"disagree": True, "head": ...}` for rule 1's
    disagreement and `{"head": ...}` for rule 7, so the pin stores what a reader needs to see.
    """
    head = reply[:OTHER_HEAD]
    preamble = reply.startswith(ABSTAIN_PREAMBLE)
    routed = ABSTAIN_ROUTE_MARK in (route or "")
    if preamble and routed:
        return ABSTAIN, {}
    if preamble != routed:
        return OTHER, {"disagree": True, "head": head}
    if SELF_REPORT_MARK in reply:
        return SELF_REPORT, {}
    low = reply.lower()
    if any(m in low for m in FALLBACK_MARKS):
        return FALLBACK, {}
    if key_hit(answer_span(reply), key):
        return CORRECT, {}
    if reply.startswith(RECALLED_TAG):
        return UNRELATED_ROW, {}
    if reply.startswith("["):
        return WRONG, {}
    return OTHER, {"head": head}


# --------------------------------------------------------------------------------------------
# 2.5 THE MUTATED CLASSIFIERS -- the control that is not the method (P8 clause 2). Each breaks
# exactly ONE of 2.3's rules and is a SEPARATE implementation: the shipped `classify` carries no
# flag that can disable a rule, so nothing a caller passes can make the instrument score this way.
# --------------------------------------------------------------------------------------------
MUTATIONS = (
    "key_before_route",
    "any_tag_is_correct",
    "shape_only",
    "text_only_abstain",
    "accept_only",
)


def mutated_classify(kind):
    """Return a DEFECTIVE classifier with 2.3's named rule broken. `kind` is one of `MUTATIONS`:

    `key_before_route`   R1: rule 4 moved ABOVE rule 1, so the hold's narrated number is scored.
    `any_tag_is_correct` R2: rules 5-6 replaced -- any reply opening with a `[tag]` is `CORRECT`,
                             key untested, so a stored row spoken in place of an answer scores.
    `shape_only`         R3: rule 4's key test replaced by answer-SHAPE alone, so any constructed
                             answer scores `CORRECT` against any key (the scrambled-key control).
    `text_only_abstain`  R4: rule 1 decided by the preamble alone -- one instrument, not two.
    `accept_only`        R6: rule 4 drops the `refuse` test, so "yes and no" scores `CORRECT`.
    """
    if kind not in MUTATIONS:
        raise ValueError(f"unknown mutation {kind!r} -- one of {MUTATIONS}")

    def _hit(span, key):
        # `accept_only` is the ONE mutation that changes rule 4's predicate rather than its place.
        if kind == "accept_only":
            return bool(re.search(key.accept, span, re.I))
        return key_hit(span, key)

    def _mutated(reply, route, key):
        head = reply[:OTHER_HEAD]
        preamble = reply.startswith(ABSTAIN_PREAMBLE)
        routed = ABSTAIN_ROUTE_MARK in (route or "")
        span = answer_span(reply)
        if kind == "key_before_route" and _hit(span, key):
            return CORRECT, {}
        if kind == "text_only_abstain":
            if preamble:
                return ABSTAIN, {}
        elif preamble and routed:
            return ABSTAIN, {}
        elif preamble != routed:
            return OTHER, {"disagree": True, "head": head}
        if SELF_REPORT_MARK in reply:
            return SELF_REPORT, {}
        low = reply.lower()
        if any(m in low for m in FALLBACK_MARKS):
            return FALLBACK, {}
        if kind == "any_tag_is_correct" and reply.startswith("["):
            return CORRECT, {}
        if kind == "shape_only":
            if reply.startswith("[") and not reply.startswith(RECALLED_TAG):
                return CORRECT, {}
        elif _hit(span, key):
            return CORRECT, {}
        if reply.startswith(RECALLED_TAG):
            return UNRELATED_ROW, {}
        if reply.startswith("["):
            return WRONG, {}
        return OTHER, {"head": head}

    return _mutated


# --------------------------------------------------------------------------------------------
# 2.4 THE CONJUNCTS, on CONSTRUCTED input only (no brain, no lake, no network).
# --------------------------------------------------------------------------------------------
def conjunct_k(classifier=classify):
    """(k) THE KEY DISCRIMINATES: every item's constructed CORRECT reply classifies `CORRECT`,
    every item's constructed WRONG reply classifies `WRONG`, and every item's OBSERVED baseline
    reply (fact 2, `FIXTURE`) classifies as the class fact 2 recorded. 20 / 20 / 20."""
    good = [(i.id, classifier(i.correct, CONSTRUCTED_ROUTE, i)[0]) for i in ITEMS]
    bad = [(i.id, classifier(i.wrong, CONSTRUCTED_ROUTE, i)[0]) for i in ITEMS]
    obs = [(f.id, classifier(f.reply, f.route, i)[0], f.cls) for f, i in zip(FIXTURE, ITEMS)]
    return {
        "correct": sum(1 for _, c in good if c == CORRECT),
        "wrong": sum(1 for _, c in bad if c == WRONG),
        "observed": sum(1 for _, c, want in obs if c == want),
        "n": len(ITEMS),
        "correct_misses": [(k, c) for k, c in good if c != CORRECT],
        "wrong_misses": [(k, c) for k, c in bad if c != WRONG],
        "observed_misses": [(k, c, want) for k, c, want in obs if c != want],
    }


def conjunct_s(classifier=classify):
    """(s) THE SCRAMBLED-KEY CONTROL -- the control that is not the method: every item's
    constructed CORRECT reply scored against the NEXT item's key (a rotation by one) classifies
    `CORRECT` on AT MOST 1 of 20. The key, not the reply's shape, carries the discrimination."""
    n = len(ITEMS)
    hits = [
        ITEMS[i].id
        for i in range(n)
        if classifier(ITEMS[i].correct, CONSTRUCTED_ROUTE, ITEMS[(i + 1) % n])[0] == CORRECT
    ]
    return {"hits": hits, "n_hits": len(hits), "n": n, "bar": 1}


def conjunct_p_fixture(classifier=classify):
    """(p)'s constructed arm: the four controls' RECORDED seed-0 replies classify `CORRECT`.
    The campaign re-reads this on a live scratch brain per seed; this arm keeps it reachable
    from the bare invocation, where there is no brain at all."""
    got = [
        (cid, classifier(reply, route, k)[0])
        for (cid, route, reply), k in zip(CONTROL_FIXTURE, CONTROLS)
    ]
    return {"n_correct": sum(1 for _, c in got if c == CORRECT), "n": len(CONTROLS), "got": got}


# --------------------------------------------------------------------------------------------
# 2.5 THE REDS -- each on its CONSTRUCTED input, each shown red by its mutation. A red FIRES only
# when BOTH hold: the shipped classifier reads the pre-registered class, AND the mutated one reads
# the different class the record names. A criterion whose red is unreachable is deleted, not kept.
# --------------------------------------------------------------------------------------------
#: R1's constructed abstain: the hold's route and preamble, narrating "3" -- item 4's key.
R1_REPLY = (
    "I am not fully certain right now, but Right now awake off; also phase high at 0.64 and "
    "rising. About 3 ticks ago I was waking, pursuing goal 0, about as certain as now."
)
R1_ROUTE = _FX_ABSTAIN_ROUTE
R1_ITEM = ITEMS[3]  # item 4, key `\b(3|three)\b`
#: R2's input is the OBSERVED seed-0 row for item 8, scored against item 8's own key.
R2_REPLY = FIXTURE[7].reply
R2_ROUTE = FIXTURE[7].route
R2_ITEM = ITEMS[7]
#: R4: the hold's preamble under a NON-abstain route -- one instrument, not two.
R4_REPLY = R1_REPLY
R4_ROUTE = _FX_GC_ROUTE
R4_ITEM = R1_ITEM
#: R6: a reply that is both a yes and a no, against item 1 (`accept` yes, `refuse` no).
R6_REPLY = "[reasoned] yes and no."
R6_ITEM = ITEMS[0]

RED_LINES = {
    "R1": "a narrated number is not an answer (abstain decided before any key is tested)",
    "R2": "a tag is not an answer (a stored row spoken in place of one)",
    "R3": "the scrambled key -- shape alone must not score (conjunct (s))",
    "R4": "two instruments must agree (the preamble under a non-abstain route)",
    "R6": "a yes that is also a no (the refuse pattern)",
}


def reds():
    """2.5's five constructed reds -> `{name: {"shipped": cls, "mutated": cls, "fires": bool}}`.
    R5 (a stale pin refuses) belongs to `c0_pin_check` and is not constructed here."""
    out = {}
    ship = classify(R1_REPLY, R1_ROUTE, R1_ITEM)[0]
    mut = mutated_classify("key_before_route")(R1_REPLY, R1_ROUTE, R1_ITEM)[0]
    out["R1"] = {"shipped": ship, "mutated": mut, "fires": ship == ABSTAIN and mut == CORRECT}

    ship = classify(R2_REPLY, R2_ROUTE, R2_ITEM)[0]
    mut = mutated_classify("any_tag_is_correct")(R2_REPLY, R2_ROUTE, R2_ITEM)[0]
    out["R2"] = {
        "shipped": ship,
        "mutated": mut,
        "fires": ship == UNRELATED_ROW and mut == CORRECT,
    }

    ship_s = conjunct_s()
    mut_s = conjunct_s(mutated_classify("shape_only"))
    out["R3"] = {
        "shipped": f"{ship_s['n_hits']}/{ship_s['n']}",
        "mutated": f"{mut_s['n_hits']}/{mut_s['n']}",
        "fires": ship_s["n_hits"] <= ship_s["bar"] and mut_s["n_hits"] == mut_s["n"],
    }

    ship, detail = classify(R4_REPLY, R4_ROUTE, R4_ITEM)
    mut = mutated_classify("text_only_abstain")(R4_REPLY, R4_ROUTE, R4_ITEM)[0]
    out["R4"] = {
        "shipped": f"{ship}/disagree={bool(detail.get('disagree'))}",
        "mutated": mut,
        "fires": ship == OTHER and bool(detail.get("disagree")) and mut == ABSTAIN,
    }

    ship = classify(R6_REPLY, CONSTRUCTED_ROUTE, R6_ITEM)[0]
    mut = mutated_classify("accept_only")(R6_REPLY, CONSTRUCTED_ROUTE, R6_ITEM)[0]
    out["R6"] = {"shipped": ship, "mutated": mut, "fires": ship == WRONG and mut == CORRECT}
    return out


# --------------------------------------------------------------------------------------------
# 2.1 (c) THE SEED RUN -- the ONLY brain-touching path in this module, and it is never reachable
# from the bare invocation. Every brain is `make_brain(seed=seed, persist=False)` inside
# `throwaway_home()`: P4 for the construction, P4b for the lake.
# --------------------------------------------------------------------------------------------
def transcript_reply(raw):
    """The transcript rule (decision 1): the reply as the L12 audit and the orientation recorded
    it -- everything before a ` (somatic` suffix. `classify` never applies this itself."""
    return raw.split(SOMATIC)[0]


def run_seed(seed, warm=WARM_TICKS):
    """One seed's reading on a SCRATCH brain: 20 warm-up ticks, then the 4 controls and the 20
    items through `respond(q, human=True)`. Returns per-question `{id, cls, route, reply}` under
    `controls` and `items`, with the seed and the wall time beside them."""
    import time

    from snn_data_lake.cortex.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    t0 = time.time()
    controls, items = [], []
    with throwaway_home():
        brain = make_brain(seed=seed, persist=False)
        for _ in range(warm):
            brain.tick()
        for bucket, keys in ((controls, CONTROLS), (items, ITEMS)):
            for key in keys:
                r = brain.respond(key.q, human=True)
                reply = transcript_reply(r.get("reply", "") if isinstance(r, dict) else str(r))
                route = (r.get("route") or "") if isinstance(r, dict) else ""
                cls, _detail = classify(reply, route, key)
                bucket.append({"id": key.id, "cls": cls, "route": route, "reply": reply})
        try:
            brain.close()
        except Exception as e:  # noqa: BLE001 -- a scratch brain's teardown, never a reading
            print(f"  seed {seed}: close() raised {type(e).__name__}: {e}", flush=True)
    return {
        "seed": seed,
        "warm": warm,
        "controls": controls,
        "items": items,
        "elapsed_s": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------------------------
# The bare invocation -- what `run_selfchecks.py` executes. CONSTRUCTED INPUTS ONLY: no brain, no
# lake, no network, no subprocess. ASCII-only output; exit 0 on green.
# --------------------------------------------------------------------------------------------
def main(argv=None):
    # `argv` is accepted and unused at 2.9 step 2: the campaign entry and the pin check land at
    # step 3, and neither is ever reachable from the bare invocation.
    print("=" * 78)
    print("C0 -- the everyday battery as a governed scorer (specs/022 Phase 4, row C0)")
    print("=" * 78)
    print(f"  {len(ITEMS)} items | {len(CONTROLS)} positive controls | classes {list(CLASSES)}")
    print(f"  protocol: {PROTOCOL}")

    ok = True
    k = conjunct_k()
    print("-" * 78)
    print(
        f"  (k) the key discriminates: constructed CORRECT {k['correct']}/{k['n']} | "
        f"constructed WRONG {k['wrong']}/{k['n']} | observed baseline {k['observed']}/{k['n']}"
    )
    if k["correct"] != k["n"] or k["wrong"] != k["n"] or k["observed"] != k["n"]:
        ok = False
        print(
            f"C0 SELF-CHECK RED: (k) misses -- correct {k['correct_misses']} wrong "
            f"{k['wrong_misses']} observed {k['observed_misses']}"
        )

    s = conjunct_s()
    print(
        f"  (s) the scrambled key (rotation by one): {s['n_hits']}/{s['n']} CORRECT "
        f"(bar: at most {s['bar']}) hits={s['hits']}"
    )
    if s["n_hits"] > s["bar"]:
        ok = False
        print(
            f"C0 SELF-CHECK RED: (s) the rotation scores {s['n_hits']} of {s['n']} -- the key is not what discriminates"
        )

    p = conjunct_p_fixture()
    print(f"  (p) the recorded controls classify CORRECT: {p['n_correct']}/{p['n']} {p['got']}")
    if p["n_correct"] != p["n"]:
        ok = False
        print(
            "C0 SELF-CHECK RED: (p)'s constructed arm -- a control the instrument cannot register"
        )

    print("-" * 78)
    print("2.5's constructed reds (shipped class | the mutation's class):")
    r = reds()
    for name in sorted(r):
        e = r[name]
        print(
            f"  {name}: {str(e['fires']):5s}  {e['shipped']} | mutated {e['mutated']}  {RED_LINES[name]}"
        )
    if not all(e["fires"] for e in r.values()):
        ok = False
        print(
            f"C0 SELF-CHECK RED: only {sum(e['fires'] for e in r.values())}/{len(r)} of 2.5's "
            f"constructed reds fire -- {r}"
        )

    print("-" * 78)
    print(
        f"  REPORTED, never barred -- fact 2's histogram, the number C-R2 starts from: {FIXTURE_HISTOGRAM} (0 CORRECT of {len(ITEMS)})"
    )
    print("  the five-seed campaign is not reachable from here; it touches a scratch brain")
    if not ok:
        return 1
    print(
        f"  C0 OK: (k) {k['n']}/{k['n']}/{k['n']}, (s) {s['n_hits']}/{s['n']}, (p) {p['n_correct']}/{p['n']}, R1/R2/R3/R4/R6 all red under their mutations"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
