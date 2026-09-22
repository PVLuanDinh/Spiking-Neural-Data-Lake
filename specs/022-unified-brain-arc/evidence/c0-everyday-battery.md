# C0 -- the everyday battery, promoted from a session scratchpad into a governed scorer instrument

Plan: `PLAN-FINAL-2026-09-02.md` Phase 4, row C0 ("promote the 20-item everyday battery (it exists only in session
scratchpads) into `src` as a scorer instrument with a self-check and PROVENANCE; without it C-R2 is unreachable").
Policies binding this arc: P2 (port the unique capability first; one mechanism), P3 (no dormancy; an exemption carries
its retirement condition), P4 / P4b (one brain, one lake: an instrument never writes to the live lake), P8 (a
reachable red; a control that is not the method; deletion on a null). Rhythm: orient -> pre-register -> build red-first
-> campaign (pin once) -> record -> panel -> fold -> stacked with T090 for ONE battery.

## 1. Territory, measured before the design (2026-09-16, heads `51e8031`..`c8c07a0`, record-only commits between)

1. THE BATTERY'S ONLY COPY was `scratchpad/battery.py` of the 2026-09-01 audit session (lens L12; file time 2026-09-01
   20:21), found by the orchestrator in an old session's temp directory and archived the same minute at
   `memory/archive/c0-2026-09-16/battery_L12_2026-09-01_session-5ef7f298.py` (a temp directory is not a store). It
   holds 20 questions, 4 positive controls and a reply-SHAPE classifier. It holds NO ANSWER KEY: the audit's "0
   correct" was a human read of 20 replies, so as it stands the battery cannot score a correct answer at all -- a
   brain that answered all 20 correctly would be classified `ANSWERED` whether right or wrong.
2. REPRODUCED ON TODAY'S TREE, five seeds (`make_brain(seed, persist=False)` under `throwaway_home()`, 20 warm-up
   ticks, `respond(q, human=True)`; scratch script `scratchpad/c0/baseline.py`): on EVERY seed 0-4 the four controls
   are answered and correct (32; 25; ijl; paris) and the 20 items read 14 abstain (route `Gf (second-order access
   hold: abstain)`), 5 recalled rows spoken in place of an answer (item 8 "4's parity is even", item 10 "week's days
   is 7", item 12 "penguin's class is bird", item 13 "2's parity is even", item 17 "square's sides is 4") and 1
   content-blind self-report (item 18, the transitive-reasoning faculty's accuracy line). Class AND text are identical
   across the five seeds. This is the audit's histogram (14 / 5 / 1, 0 correct, controls 4 of 4), unmoved in 15 days.
3. THE AUDIT'S CONSTRUCTION DIFFERED and the number did not: it built `Brain(seed=0, refit_every=10, persist=True)`
   in a temp `NORD_HOME` with the live `language_acq.npz` copied in; today's protocol goes through `make_brain` (P4)
   with `persist=False`. Disclosed as a protocol change with an identical reading.
4. THE INSTRUMENT CAN NEVER RUN ON THE LIVE BRAIN. `respond(q, human=True)` is the chat lane: it ingests the question
   into the one durable lake (P4b). Twenty-four synthetic questions per run written into `brain_lake.bin` would be the
   instrument contaminating the thing it measures. So the scorer runs on a scratch brain and its number reaches any
   surface only as a PINNED campaign reading that says so -- the T080-T082 pattern.
5. AN ABSTAIN REPLY NARRATES NUMBERS. The hold's reply is "I am not fully certain right now, but <state narration>",
   and the narration carries digits ("About 8 ticks ago ... goal 0 ... 0.64"). A key such as `3` or `7` tested against
   the whole reply can be satisfied by the narration. Abstention therefore has to be decided BEFORE any key is tested,
   and by two instruments that must agree (the route label and the reply's preamble).
6. `substrate/capability_bar.CAPABILITY_BAR` (X1's registry, T088) is EMPTY today. Its contract is the one this
   number needs: a brain number with its protocol and provenance, `human=None` with the REFUSED verdict where no human
   number exists under a matching protocol, and then no comparison word for that task anywhere.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT -- the account is under a WEEKLY usage cap until 2026-09-20 00:30

Disclosed, as on T078-T082 and T090; the closing report goes when the cap lifts.

### 2.1 The mechanism -- ONE module, ONE key, ONE pin

`cortex/everyday_battery.py` (the answer path it scores is `cortex/brain.py::Brain.respond`):
(a) `ITEMS` -- the audit's 20 questions VERBATIM, each with the answer key of 2.2; `CONTROLS` -- its 4 positive
controls verbatim with their keys. (b) `classify(reply, route, key)` -> exactly one of `ABSTAIN`, `SELF_REPORT`,
`FALLBACK`, `CORRECT`, `UNRELATED_ROW`, `WRONG`, `OTHER`, by the ordered rule of 2.3. (c) `run_seed(seed)` -- a scratch
brain (`make_brain(seed=seed, persist=False)` under `throwaway_home()`), 20 warm-up ticks, controls then items through
`respond(q, human=True)`, returning per-question `{id, cls, route, reply}`. (d) `--c0-campaign`: seeds 0-4, EACH IN ITS
OWN PROCESS (a seed owns a process-wide `NORD_HOME`, and five brains sharing one scratch lake could recall each other's
ingested questions); the pin `tools/c0_everyday_battery.json` is written ONCE with `head`, `dirty`, the protocol, the
per-seed rows and the histogram. (e) `c0_pin_check()` -- re-derives every pinned class from the pinned reply and route
through `classify()` and refuses on any disagreement, so a later edit to the classifier or the key cannot leave a stale
pin standing (bare `python -m snn_data_lake.cortex.everyday_battery` runs it with the constructed reds). (f) ONE row
in `substrate/capability_bar.CAPABILITY_BAR` for the task, written on MET only (2.6).
P2: the scratch battery is the mechanism being PORTED (questions, controls, and its five shape classes map onto the
new ones: `ABSTAIN (metacog/2nd-order hold)` -> `ABSTAIN`; `SELF-REPORT (content-blind)` -> `SELF_REPORT`; `HONEST GAP`
and `CAPABILITY FALLBACK` -> `FALLBACK`; `ANSWERED (symbolic DSL / KB)` -> `CORRECT` or `WRONG` by the key; `OTHER:
[recalled ...` -> `UNRELATED_ROW`); nothing of it survives beside the module (the archive copy is the record of what was
ported, not a second instrument).

### 2.2 The answer key (frozen here; case-insensitive regular expressions tested on the ANSWER SPAN -- the reply with a
leading `[tag]` removed)

| id | question (verbatim) | accept | refuse if also present | constructed CORRECT reply | constructed WRONG reply |
|---|---|---|---|---|---|
| 1 | If all bloops are razzies and all razzies are lubbers, are all bloops lubbers? | `\byes\b` | `\bno\b` | [reasoned] yes, all bloops are lubbers. | [reasoned] no, they are different things. |
| 2 | Anna is taller than Bo. Bo is taller than Cy. Who is shortest? | `\bcy\b` | | [reasoned] cy is shortest. | [reasoned] anna is shortest. |
| 3 | A key is to a lock as a password is to what? | `\b(account\|login\|computer\|system\|device\|phone\|website\|site\|email\|app\|network)\b` | | [induced] an account. | [induced] a hammer. |
| 4 | Tom has 5 apples and gives 2 to Ann. How many does Tom have? | `\b(3\|three)\b` | | [reasoned] tom has 3 apples. | [reasoned] tom has 7 apples. |
| 5 | If it rains the ground gets wet. The ground is not wet. Did it rain? | `\bno\b` | `\byes\b` | [reasoned] no, it did not rain. | [reasoned] yes, it rained. |
| 6 | What is heavier, a kilogram of feathers or a kilogram of steel? | `\b(same\|equal\|neither\|both)\b` | | [reasoned] they weigh the same. | [reasoned] the steel is heavier. |
| 7 | I put a ball in a box, closed it, and left the room. Where is the ball? | `\bbox\b` | | [reasoned] the ball is in the box. | [reasoned] the ball is in the garden. |
| 8 | Sort these from smallest to largest: 9, 2, 7, 4. | `2\D+4\D+7\D+9` | | [reasoned] 2, 4, 7, 9. | [reasoned] 9, 7, 4, 2. |
| 9 | What comes next: Monday, Tuesday, Wednesday? | `\bthursday\b` | | [induced] thursday. | [induced] sunday. |
| 10 | If today is Friday, what day is it in three days? | `\bmonday\b` | | [reasoned] monday. | [reasoned] sunday. |
| 11 | Which is the odd one out: dog, cat, hammer, horse? | `\bhammer\b` | | [reasoned] hammer -- the others are animals. | [reasoned] horse. |
| 12 | Every bird has feathers. A penguin is a bird. Does a penguin have feathers? | `\byes\b` | `\bno\b` | [reasoned] yes, a penguin has feathers. | [reasoned] no, a penguin swims. |
| 13 | A train leaves at 3pm and takes 2 hours. When does it arrive? | `\b5\s*(pm\|p\.m\.?\|o'?clock)\|\bfive\b\|\b17[:.]?00\b` | | [reasoned] it arrives at 5pm. | [reasoned] it arrives at 6pm. |
| 14 | What is 17 plus 25? | `\b42\b\|forty[- ]two` | | [reasoned] 42. | [reasoned] 41. |
| 15 | If x plus 3 equals 10, what is x? | `\b(7\|seven)\b` | | [reasoned] x is 7. | [reasoned] x is 13. |
| 16 | Two boxes weigh the same. Box A holds bricks, box B holds pillows. Which is heavier? | `\b(same\|equal\|neither)\b` | | [reasoned] neither -- they weigh the same. | [reasoned] box a, the bricks. |
| 17 | Is a square always a rectangle? | `\byes\b` | `\bno\b` | [reasoned] yes, every square is a rectangle. | [reasoned] no, never. |
| 18 | Give me a reason why the sky looks blue. | `scatter` | | [reasoned] air scatters blue light more than red. | [reasoned] it reflects the ocean. |
| 19 | What would happen if I dropped a glass on concrete? | `\b(break\|shatter\|crack\|smash)` | | [reasoned] it would shatter. | [reasoned] it would bounce and float. |
| 20 | Name something that is both a fruit and red. | `\b(apple\|strawberr\|cherr\|raspberr\|tomato\|pomegranate\|cranberr\|watermelon)` | | [from what I learned] a strawberry. | [from what I learned] a banana. |

Inside a table cell `\|` is a markdown-escaped `|` (regex alternation); the patterns are otherwise literal.
Controls (keys: `\b32\b`; `\b25\b`; `\bijl\b`; `\bparis\b`), questions verbatim from the archive copy.
REACHABILITY CHECKED BEFORE FREEZING (orchestrator, scratch probe `scratchpad/c0/key_probe.py`, which parses THIS
table and applies 2.3's rule -- a draft check, not the instrument): constructed CORRECT 20 of 20, constructed WRONG 20
of 20, the observed seed-0 replies 14 `ABSTAIN` / 5 `UNRELATED_ROW` / 1 `SELF_REPORT`, the rotation-by-one 0 of 20,
controls 4 of 4; R1 `ABSTAIN`, R2 `UNRELATED_ROW`, R4 `OTHER` with `disagree`, R6 `WRONG`. A pre-registered number its
own key could not produce would be a vacuous bar.
THE KEY IS A FLOOR, NOT A JUDGE: it recognises a short correct answer in plain words and will miss a correct answer
phrased outside its patterns (a false NEGATIVE it can make) and cannot be satisfied by a reply that declines to answer
(the false POSITIVE it must never make -- 2.4's reds). A miss on a genuinely correct reply is reported per item with
the reply text, never silently scored wrong.

### 2.3 The classifier's ordered rule (written first)

1. `ABSTAIN` iff the ROUTE label contains `abstain` AND the reply opens with the hold's preamble (`I am not fully
   certain`). If exactly ONE of the two holds the class is `OTHER` with `disagree: true` -- two instruments must
   agree, and a disagreement is reported, never resolved by picking one (fact 5).
2. `SELF_REPORT` iff the reply carries the content-blind faculty line (`faculty has learned`).
3. `FALLBACK` iff the reply is the capability fallback or the honest-gap line (the scratch battery's two strings).
4. `CORRECT` iff the answer span matches `accept` and matches no `refuse` pattern.
5. `UNRELATED_ROW` iff the reply opens with a `[recalled` tag (a stored row spoken in place of an answer; "unrelated"
   is the plan's word for C-R2's class -- item 12's row is topically related and still not an answer).
6. `WRONG` iff the reply opens with any other `[tag]` (answer-shaped, no key match).
7. `OTHER` otherwise, with the first 70 characters kept.
No key is tested before rules 1-3 have declined.

### 2.4 The conjuncts and the verdict rule (written first)

(p)  THE POSITIVE CONTROL: on EVERY seed the four controls classify `CORRECT` (20 of 20 control x seed cells); else
     REFUSED -- an instrument that cannot register a known hit reads nothing.
(k)  THE KEY DISCRIMINATES, on constructed replies and no brain: every item's constructed CORRECT reply classifies
     `CORRECT` (20 of 20), every item's constructed WRONG reply classifies `WRONG` (20 of 20), and every item's
     OBSERVED baseline reply (fact 2, frozen in the module as the port's fixture) classifies as the class fact 2
     recorded (20 of 20).
(s)  THE SCRAMBLED-KEY CONTROL (the control that is not the method): with every item scored against the NEXT item's
     key (a rotation by one), the constructed CORRECT replies classify `CORRECT` on at most 1 of 20. The key, not the
     reply's shape, carries the discrimination.
(h)  THE PORT IS FAITHFUL: the campaign's per-item classes on seeds 0-4 equal fact 2's on all 100 item x seed cells
     (14 `ABSTAIN`, 5 `UNRELATED_ROW`, 1 `SELF_REPORT`, 0 `CORRECT`, 0 `WRONG` per seed). A cell that moved is named
     with both replies; the tree's answer path has not changed since fact 2, so a move is a finding about the port
     or about determinism, never noise to be averaged.
(c)  CONCURRENCY CANNOT CHANGE A REPLY: seed 0's rows from the five-process campaign equal a sequential seed-0 run's
     byte for byte (class, route, reply).
VERDICT: MET iff (k), (s), (h) and (c) with (p) holding; REFUSED iff (p) fails; NOT MET otherwise.
REPORTED, never barred: wall time per seed; every `OTHER` with its text; every key miss on a reply a reader would
call correct; the histogram the plan's C-R2 starts from.
WHAT A PASS LICENSES: "the battery is a governed instrument that reproduces the audit's reading and can score a
correct answer". It licenses NO statement about the brain's reasoning beyond the histogram itself, which is the
audit's: 0 of 20.

### 2.5 The reds (each runs on its constructed input in the module's bare invocation and in `tests/`)

R1  A NARRATED NUMBER IS NOT AN ANSWER: a constructed abstain (route `Gf (... abstain)`, reply "I am not fully certain
    right now, but ... About 3 ticks ago ...") scored against item 4's key (`3`) classifies `ABSTAIN`. Mutation
    `key_before_route` (rule 4 moved above rule 1): it classifies `CORRECT`: red.
R2  A TAG IS NOT AN ANSWER: the observed "[recalled by association] 4's parity is even." against item 8 classifies
    `UNRELATED_ROW`. Mutation `any_tag_is_correct`: red.
R3  THE SCRAMBLED KEY (conjunct (s)) -- mutation `shape_only` (accept anything answer-shaped): the rotation scores 20
    of 20: red.
R4  TWO INSTRUMENTS MUST AGREE: the hold's preamble under a NON-abstain route classifies `OTHER` with `disagree`.
    Mutation `text_only_abstain`: it classifies `ABSTAIN`: red.
R5  A STALE PIN REFUSES: `c0_pin_check()` on a pin whose stored class for one row is flipped refuses; on a pin whose
    reply is edited so the class re-derives differently refuses.
R6  A YES THAT IS ALSO A NO: "[reasoned] yes and no." against item 1 classifies `WRONG` (the `refuse` pattern).
    Mutation `accept_only`: red.

### 2.6 Disposition (fixed in advance)

- MET: the pin is committed once; `CAPABILITY_BAR` gains ONE row, task `everyday reasoning, 20 items (C0)`:
  protocol = the pin's protocol string; `human=None` with the REFUSED verdict (no human number exists under this
  protocol, so no comparison word may ever stand beside this task); `brain` = the campaign's correct fraction with
  its provenance (the pin path, head, date). The tests that pin the registry's count move WITH it (named in the
  builder brief: `tests/test_ratified_row_gates.py`; obs #2200). The module is registered (`__init__.py` REGIONS,
  `run_selfchecks.py`) and enters `dormancy_gate.EXEMPT` with its reason and its retirement condition: C2's red C-R2
  consumes it (the day C2 lands the exemption retires). Plan + tasks notes: a dated line on the plan's C0 row and on
  C-R2 ("reachable: `cortex/everyday_battery.py`, pin at <commit>").
- NOT MET: the record names the conjunct and the cell; the module stays only if (k) and (s) hold (a sound scorer
  whose port moved is still the instrument C2 needs; the moved cells become an opening); if (k) or (s) fails the
  module is DELETED (P8 clause 5) and C0 stays open.
- REFUSED: the fault named; nothing lands on any surface.
In every branch the live brain, the live lake and `Brain.respond` are untouched.

### 2.7 What this row does NOT claim

Nothing about the brain's reasoning beyond the audit's own histogram; no human comparison (refused by construction);
no claim that the 20 items are a representative sample of everyday reasoning (they are the audit's 20, kept verbatim
so C-R2's before/after is one instrument); no live cockpit number (a live read would write to the lake: fact 4).

### 2.8 Seeds, cost, host

Seeds 0-4, one process each; about 1.5 to 2 minutes per seed measured in orientation (sequential 8 to 10 minutes;
concurrent under 4). Host: the pinned machine, Python 3.14.6 free-threaded, the `.venv-lake` env. No GPU arm: the
instrument is 24 chat turns per seed and the tick is CPU at this width.

### 2.9 Commit order (one change each; the reds before the campaign)

1. This file + the plan's C0 dated note -> 2. `cortex/everyday_battery.py` (items, key, classifier, fixture, the
constructed reds R1-R6 in the bare invocation; NO campaign yet) + `tests/test_everyday_battery.py` (R1-R6 with each
mutation shown red) + registration (`__init__.py`, `run_selfchecks.py`, the dormancy exemption with its retirement
condition) -> 3. the campaign entry + `c0_pin_check()` + R5's test -> 4. the orchestrator runs the campaign, the pin is
committed once -> 5. section 3 the record -> 6. on MET: the `CAPABILITY_BAR` row with the registry-count tests moved
-> 7. the surfaces per 2.6 -> panel (3 lenses + refuters judging CLAIM and FIX) -> fold -> stacked with T090 for ONE
battery (the Codex closing report when the cap lifts, 2.0).

### 2.10 Amendments (dated, appended; nothing above is edited in place)

2026-09-17 02:27 local (clock read at write time), on the builder's report at `181ec3e` (module `2a13221`, campaign entry
`181ec3e`), BEFORE the campaign is run:

(a) FACT 2 IS WRONG ABOUT TEXT, AND ABOUT THE ROUTE LABEL. It says "Class AND text are identical across the five
seeds". The CLASS is (100 of 100 item x seed cells, controls 20 of 20). The TEXT is identical on only the 5 recalled rows
(items 8, 10, 12, 13, 17) and the 4 controls: 15, 14, 15 and 15 of 20 replies differ from seed 0's on seeds 1-4, because
the hold's reply narrates the brain's state and the state differs by seed. The orchestrator's orientation script
compared text only for recalled and answer-shaped rows -- a blind spot in MY instrument, caught by the builder running
the comparison on every row. Re-checking after that report found a second half the builder's report missed (it says the
routes agree): the 14 abstains arrive through TWO DIFFERENT HOLDS -- route `Gf (second-order access hold: abstain)` on
seeds 0-2 and `Gf (metacognitive hold: abstain)` on seeds 3-4. Both labels carry the word rule 1 reads, so the class is
the same; the finding is C2's (two holds pre-empt the answer path, by seed), not this instrument's. No conjunct compares
text or route ACROSS seeds: (h) is over classes, and (c) compares seed 0 with itself (sequential vs concurrent), where
text IS deterministic (the builder's dry run: equal on class, route and reply).
(b) 2.8's COST ESTIMATE was about six times high: measured 16.7 to 17.8 s per seed, about 36 s for the campaign.
(c) TWO READINGS OF THE RECORD the module makes, accepted: the ` (somatic` suffix is removed when the reply is RECORDED
(`run_seed`), as the audit's and the orientation's transcripts did, not inside `classify` (2.3 never mentions it; a
suffix carrying digits would otherwise reach a numeric key); and `FALLBACK` ports the scratch battery's three
substrings (2.3 rule 3's "two strings" named its two classes).
(d) DISCLOSED: the builder's DRY RUN (2.9 step 3, a scratch pin on a dirty tree at `2a13221`) read MET with the
histogram of fact 2 on every seed, so the campaign's outcome was seen in dry-run form before the pinned run. The pinned
run on a clean tree is the record; the dry pin is scratch and is not cited for any number.
Nothing here moves a conjunct, a red, the key or the verdict rule.

2026-09-22 (the panel's fold, 3.7; appended AFTER the campaign -- nothing below re-runs the campaign, rewrites the pin
`tools/c0_everyday_battery.json`, or moves a conjunct, a red, the key, the rule order or the verdict; every figure below
was re-run by the orchestrator on 2026-09-22 with the refuter's own script before it was written; the scripts are named
by their path under the panel scratch, 3.7):

(e) (s) IS ONE FIXED PRE-REGISTERED DERANGEMENT AND ITS BAR IS PERMUTATION-DEPENDENT (panel I4). Deterministic figures:
over the 19 rotations the constructed CORRECT arm scores max 2 (k = 10 and k = 16; rotation-by-one 0; identity 20); the
off-diagonal cross-hit matrix is 10 of 380 pairs -- (1,12), (1,17), (12,1), (12,17), (17,1), (17,12), (6,16), (16,6),
(4,20), (8,15) -- and a maximum matching over that graph gives an EXHAUSTIVE maximum of 7 of 20 over all permutations.
Cause, split honestly: 8 of the 10 pairs are answer-space collisions in the item set -- items 1, 12 and 17 are three
distinct yes/no questions sharing `accept=\byes\b refuse=\bno\b` exactly, and items 6 and 16 differ only by the
alternative `both` -- and 2 are lexical coincidences (item 4's "3 apples" hits item 20's `\bapple`; item 8's
"2, 4, 7, 9." hits item 15's `\b(7|seven)\b`). This is the mirror of 2.2's declared floor (a key tested on the answer span
cannot tell three "yes" answers apart without reading the question), not a construction error, and it is the half 3.6
opening 2 does not name. Sampled derangement fractions are reported only with their n and RNG seed and are
sample-dependent: 8 of 200 over the bar at seed 0 (the lens); 44 of 500 at seed 12345 (the refuter; re-run: {0: 298,
1: 158, 2: 39, 3: 4, 4: 1}). The verdict and the licensed sentence are unchanged: (s) as pre-registered reads 0 <= 1, and
R3 separates at every permutation (`shape_only` 20 of 20 against a shipped ceiling of 7 of 20), so shape alone still
cannot score. The pinned OBSERVED seed-0 arm, for contrast: identity 0, max 1 over the 19 rotations (k = 5), 1 of 380
cross-hits, exhaustive maximum 1. Scripts: `refute/I4/scramble_refute.py` (sections 1-5; its section 6 raises on
`Control.correct` after the figures print) and `instrument/key_attack.py`.

(f) THE KEY CAN MAKE THE FALSE POSITIVE 2.2 SAYS IT NEVER MAKES, AND SIX PATTERNS ACCEPT AT THEIR TAIL (panel I6 and
I9). 2.2's sentence "the false positive it must never make is a reply that declines to answer scoring CORRECT" is a
blanket guarantee the instrument does not deliver; 2.2 is not edited, and its guarantee is WITHDRAWN as a blanket and
replaced by the measured scope here. Measured on CONSTRUCTED inputs: 25 declining or non-answer replies x 20 keys give
24 CORRECT cells (`instrument/key_attack.py`), among them "[reasoned] I had breakfast this morning." -> item 19 via
`break`, "[reasoned] that is a crackpot question." -> item 19 via `crack`, "[reasoned] my thoughts are scattered." ->
item 18 via `scatter`, "[reasoned] applejack." -> item 20, "[reasoned] 12, 4, 7, 90." -> item 8, "[reasoned] 5pmish." ->
item 13, "[reasoned] forty-twosome." -> item 14; and the pinned item-10 recalled reply, cross-scored against item 15's
key, reads CORRECT. THE PROTECTION THAT DOES HOLD: rule 1 (the hold's route and preamble) and rules 2-3 precede the
key, so the constructed abstain R1 and the pinned seed-2 / item-15 key hit both read ABSTAIN. OBSERVED VS
CONSTRUCTED, kept apart: no campaign cell is misclassified -- the actual item-10 cells are UNRELATED_ROW and the actual
item-15 cells ABSTAIN on all five seeds; the cross pairing is constructed, not an observed C2 conversion; a rule reorder
(the key before the recalled-tag rule), simulated in memory, changes 0 of 144 pinned classes and would turn the
substantively correct constructed "[recalled by association] x is 7." into UNRELATED_ROW -- it is NOT the fix. Pattern
inventory: items 8 (`2\D+4\D+7\D+9`), 18 (`scatter`), 19 (`\b(break|shatter|crack|smash)`) and 20 (`\b(apple|strawberr|
...)`) carry no tail boundary, and items 13 (the time branch) and 14 (`forty[- ]two`) carry an unbounded alternative.
Item 18's own constructed CORRECT reply relies on matching "scatters" and item 20's on "strawberry": a four-key
tail-boundary mutant rejects those two pre-registered positives while moving 0 of 144 pinned classes, so anchoring is a
KEY change with its own arc, not a fold; no author intent is inferred. Carried forward: C2's report must expose the
already-recorded replies beside any CORRECT it interprets and disclose token matches that do not answer the question;
C-R2's criterion (unrelated rows and self-reports vanish with abstains rising, the controls 4 of 4) is unchanged and is
NOT a conversion criterion. Scripts: `instrument/key_attack.py`, `refute/I6/refute_i6.py`, `refute/I9/prefix_probe.py`.

(g) RULE 1'S TWO INSTRUMENTS ARE ONE BRANCH'S TWO OUTPUTS AT 678a26a (panel I8): `Brain.respond` sets the abstention
preamble and both abstain route labels in the same hold branch, and every later change to `reply` in that function
appends (nine append-only updates; nine route assignments, two carrying "abstain"). The two readings are consistency
checks on a common producer, not independent evidence of abstention; R4 is a constructed classifier regression test,
reachable on constructed input and unreached from the live answer path as measured (scratch runs on seeds 0 and 3, 24
rows each, 0 marker mismatches; the seed-0 hold is the second-order access hold, the seed-3 hold the metacognitive
one). Rule 1 and R4 stand unchanged. Script: `refute/I8/probe_i8.py`.

(h) DISCLOSURE ON (c) (panel I5, refuted): (c) measures same-seed concurrency invariance, as 2.4 defines it, and has
executable constructed reds (a mutated reply, route or class each reads ok=False). A shared-home construction -- two
sequential `persist=False` brains under one scratch home, seed 1 then seed 0 -- gave 0 row-field diffs against isolated
runs, with the seed-sensitivity control at 15 diffs (seed 0 against seed 1): that is the construction's null, not
evidence that process isolation is unnecessary, and the contamination rationale stays unverified by this experiment.
Script: `refute/I5/probe_i5.py`.

## 3. The record

### 3.0 Apparatus as run (2026-09-17 02:35 local, clock read at write time)

The Opus builder landed 2.9 steps 2 and 3 as `2a13221` (the module, its reds, its tests, its registration) and `181ec3e`
(the campaign entry, one process per seed, and the pin check). The orchestrator reproduced its lines (29 tests passed;
the bare invocation rc 0), wrote the dated amendment 2.10 (`37135d7`) BEFORE the campaign, then ran the campaign itself
on the clean tree: `python -m snn_data_lake.cortex.everyday_battery --c0-campaign`, 50 s, the pin written once at head
`37135d7` with `dirty: false` and committed as `b167118`. INDEPENDENT VERIFICATION
(`scratchpad/c0/verify_pin.py`): a classifier parsed from THIS record's frozen key table (2.2) and implementing 2.3 on
its own -- it shares no code with the module -- re-derives all 120 pinned classes (5 seeds x 24 questions) with 0
disagreements, (p) 20 of 20, every seed's histogram, and (c) on (id, class, route, reply). The module's own
`c0_pin_check` re-derives 144 rows (the 120 plus the sequential seed-0 run's 24) with 0 disagreements. The live brain
was not running (the owner stopped the daemon at 01:12 for unrelated reasons); the instrument never touches it anyway.

### 3.1 The build

`cortex/everyday_battery.py`: `ITEMS` (the archive's 20 questions verbatim; ids and arity asserted at import; the TEXT
chain is the 2026-09-16 orientation's and is not verifiable in-tree -- corrected in place and marked 2026-09-22, panel D1),
`CONTROLS`, the frozen key, `classify` by 2.3's ordered rule, `FIXTURE` (the orientation's seed-0 replies and routes),
`run_seed`, the campaign, `c0_pin_check`. Registered in `__init__.py` REGIONS and `run_selfchecks.py` (360 of 360
self-checks; the census gates rc 0 after both commits) and entered in `dormancy_gate.EXEMPT` as `wire` with its
retirement condition (C2's red C-R2 consumes it; it can never be wired to a tick because the chat lane ingests -- fact
4). For the pin's `head` / `dirty` the campaign IMPORTS `substrate/policy_manifest._tree_state` rather than writing a
seventh private copy (graph-exists listed six; P2). A RED THAT COULD NOT FIRE, found by the builder running it: the
mutation `accept_only` was declared and never implemented, so R6 first read `WRONG | mutated WRONG` and the bare
invocation went rc 1; fixed before the step-2 commit (the mutated classifier got its own rule-4 predicate).

### 3.2 The campaign (pin `tools/c0_everyday_battery.json`)

| seed | controls CORRECT | ABSTAIN | UNRELATED_ROW | SELF_REPORT | CORRECT | WRONG | OTHER / FALLBACK | the hold that abstained | wall time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 of 4 | 14 | 5 | 1 | 0 | 0 | 0 | Gf (second-order access hold: abstain) | 19.7 |
| 1 | 4 of 4 | 14 | 5 | 1 | 0 | 0 | 0 | Gf (second-order access hold: abstain) | 19.7 |
| 2 | 4 of 4 | 14 | 5 | 1 | 0 | 0 | 0 | Gf (second-order access hold: abstain) | 19.3 |
| 3 | 4 of 4 | 14 | 5 | 1 | 0 | 0 | 0 | Gf (metacognitive hold: abstain) | 19.6 |
| 4 | 4 of 4 | 14 | 5 | 1 | 0 | 0 | 0 | Gf (metacognitive hold: abstain) | 19.2 |

### 3.3 The conjuncts, each quoted and marked

- (p) THE POSITIVE CONTROL: **HOLDS** -- 20 of 20 control x seed cells classify `CORRECT`.
- (k) THE KEY DISCRIMINATES: **MET** -- constructed CORRECT 20 of 20, constructed WRONG 20 of 20,
  the observed baseline 20 of 20.
- (s) THE SCRAMBLED-KEY CONTROL: **MET** -- rotation by one scores 0 of 20 (bar: at most 1).
- (h) THE PORT IS FAITHFUL: **MET** -- 100 of 100 item x seed cells equal fact 2's classes; moved: none.
- (c) CONCURRENCY CANNOT CHANGE A REPLY: **MET** -- seed 0's five-process rows equal the sequential run's on class, route
  and reply; diffs: none.
- The reds R1, R2, R3, R4, R6 fire under their mutations in the campaign's own preamble and in `tests/`; R5 (a stale pin
  refuses: a flipped class, an edited reply) in the bare invocation and in `tests/` (29 tests).
- 2026-09-22 PANEL DISCLOSURE (I7), appended beside the reds: seed 2's pinned item 15 abstain reply contains "goal at 7"
  and "pursuing goal 7", satisfying its key `\b(7|seven)\b`; moving rule 4 before rule 1 while retaining rule 1 (R1's
  `key_before_route` counterfactual) changes exactly this one of 100 campaign item x seed cells from ABSTAIN to CORRECT,
  giving seed 2 a counterfactual histogram of 13 ABSTAIN / 1 CORRECT / 5 UNRELATED_ROW / 1 SELF_REPORT; removing rule 1
  outright changes 70 of 100. The shipped classifier scores the cell ABSTAIN; no code, key, campaign or pin change.
  Scripts: `instrument/narration_hazard.py`, `refute/I7/check_pinned_overlap.py`.
**VERDICT: MET.** What it licenses, quoted from 2.4: "the battery is a governed instrument that reproduces the audit's
reading and can score a correct answer". It licenses NO statement about the brain's reasoning beyond the histogram,
which is the audit's and is unchanged: **0 of 20 correct on every seed; 14 abstain, 5 recalled rows spoken in place of
an answer, 1 content-blind self-report; the four controls answered.**

### 3.4 Disposition (2.6, MET branch)

The pin is committed once (`b167118`). `CAPABILITY_BAR` gains its first row at `8f5f032`: the task's protocol, the
brain number 0.0 with its provenance, `human=None` with the refused verdict -- so no comparison word may stand beside
this task on any surface (the session's call under the owner's 2026-09-05 authorization; the row sets no target). The
module is registered and exempted with its retirement condition (3.1). Dated notes: the plan's C0 row and C-R2
("reachable: `cortex/everyday_battery.py`, pin at `b167118`").

### 3.5 Deviations and disclosures

1. FACT 2's "text identical" was WRONG (2.10 (a)): my orientation script compared text only on recalled rows. The
   builder's every-row comparison found 14 to 15 of 20 replies differ per seed; my re-check then found what the builder's
   report had missed -- the route label differs too (second-order access hold on seeds 0-2, metacognitive hold on seeds
   3-4). The pin reproduces both. No conjunct compared text or route across seeds.
2. THE OUTCOME WAS SEEN BEFORE THE PINNED RUN (2.10 (d)): the builder's dry run to a scratch path read MET. With a
   deterministic instrument and a port-fidelity bar, the pinned run could only confirm it; said plainly.
3. 2.8's cost estimate was six times high (about 17 to 20 s per seed; the campaign 50 s).
4. COMMIT ORDER: 2.9 puts this section (step 5) before the registry row (step 6); the row's builder ran while this
   section was written, and index discipline (never commit beside a running builder) landed the row first.
5. THE BUILDER'S ACCEPTED READINGS (2.10 (c)) and additions: the somatic suffix is stripped in the transcript, not in
   `classify` (a test pins both halves: a bare-digit key is satisfiable by "valence 0.3" on an un-stripped reply);
   `FALLBACK` ports three substrings; `c0_pin_check` accepts a path or a mapping (so R5 runs in the bare invocation); a
   pin with zero rows reads `ok: False` (an empty walk and a clean pin must not look alike); R2's and R3's mutations were
   made genuinely distinct (`any_tag_is_correct` vs `shape_only`).
6. NOT PANELLED YET at the time of writing; the panel and its fold follow as 3.7.
7. (2026-09-22, panel PH1) THE SURFACE CHECK'S MEASURED SCOPE: it fires when the registry key and one of
   `COMPARISON_WORDS` occur on ONE LINE of one of the five `TRACKED_SURFACES`; the key is present in 0 of the 5 today,
   so the guard is latent rather than live coverage; its reachable red on the real key, with its negative control, is
   shipped at `tests/test_ratified_row_gates.py::test_the_c0_row_carries_its_number_and_refuses_the_human_side`; the key
   and a comparison word on ADJACENT lines are not seen. The rule the row imposes on authors (2.6, 3.4) is unchanged and
   is review-enforced, which is what its deontic wording says. (Re-run 2026-09-22: `findings()` = [], 0 comparison-word
   lines on each tracked surface.)
8. (2026-09-22, panel D1) FOUR SURFACES SAID THE QUESTION TEXT WAS ASSERTED EQUAL TO THE ARCHIVE'S; none does. The
   module's two import-time asserts check that the id lists line up as 1..20 and that both arities are 4; the pin
   carries no question text; nothing reads the module docstring at runtime. The four surfaces (the module docstring's
   decision 3, the comment above the asserts, the port test's docstring, this record's 3.1) now say so; 3.1 is corrected
   in place and marked. The archive's text is an unverifiable-in-tree premise, named as such (3.6 item 8). Measured
   (re-run 2026-09-22): the orchestrator's own AST diff of the archive against the tree reads 20 of 20 items and 4 of 4
   controls identical, with a one-character mutation as the differ's positive control (1 differing).
9. (2026-09-22, panel D2, refuted -- with a correction of my own) THE INDEPENDENT VERIFIER OF 3.0 IS A SESSION SCRATCHPAD
   FILE (`scratchpad/c0/verify_pin.py` of the 2026-09-17 session), still on disk under that session's directory. It
   parses the key from this record with the row regex `^\| \d+ \| `, which since 3.2 landed (02:35, seven minutes after
   the verifier ran at 02:28) ALSO matches the five campaign rows, so keys 1-4 are overwritten by seed rows and a 21st key
   appears: the unmodified script refuses at its own `assert len(keys) == 20` today. The D2 refuter found the same cause
   on 2026-09-21 (25 row matches, folded to 21 by the id-keyed dict) and re-ran a copy whose parse is SCOPED to section
   2.2 by one added line, `classify()` byte-identical (`refute/D2/d2_verify_pin_scoped.py`); the orchestrator re-ran that
   scoped copy on 2026-09-22 against the fold head: disagreements 0 over 120 re-derived rows, (p) 20 of 20, every seed's
   histogram 14 / 5 / 1, (h) True, (c) True on 24 rows -- 3.0's number reproduces. A verifier that goes stale when the
   record it reads grows is the shape of obs #2105; the scoped copy is the fix, and the original is kept as it ran.
10. (2026-09-22, panel D5) WHAT THE FOLD'S NEW TEST PINS: the campaign's cross-module import of
    `substrate/policy_manifest._tree_state` (3.1) had no executor -- the bare invocation never reaches that line and the
    pin-refusal test returns first -- so a rename of the private helper was silent through pytest, run_selfchecks and
    ruff alike; the added test asserts the consumer's import line and resolves it (re-run 2026-09-22: rc 0 with the
    attribute present, rc 1 with it deleted). Sharpening: the ImportError would land AFTER the campaign has run and
    BEFORE the pin is written.
11. (2026-09-22, panel I11, refuted) RAW REPLIES AND STORED TRANSCRIPTS ARE DISTINCT OBJECTS: the transcript rule (module
    decision 1, 2.10 (c), item 5 above) truncates at the first " (somatic", which also removes every note
    `Brain.respond` appends after the somatic note; 0 of 144 stored transcripts carry the marker (positive control 1),
    which proves absence in the STORED transcripts only, not in raw replies; a constructed raw reply whose refusal sits
    after the marker classifies WRONG raw and CORRECT truncated. C0 scores the disclosed transcript; no change.

### 3.6 Openings

1. TWO HOLDS PRE-EMPT THE ANSWER PATH, BY SEED (C2's territory): the second-order access hold on seeds 0-2, the
   metacognitive hold on seeds 3-4, 14 abstains each. C2's pre-registration should read which hold fires and why before
   it routes on `{known, coverage}`.
2. THE KEY IS A FLOOR (2.2): it will miss a correct answer phrased outside its patterns; the campaign reports every
   answer-shaped non-CORRECT row with its text (none today) so a miss is visible the day the brain starts answering.
   2026-09-22 QUALIFICATION (panel I6): the floor has a false-POSITIVE face too -- a non-answer that carries the key's
   token scores CORRECT unless rules 1-3 catch it first; 2.10 (f) measures it and names what C2's report owes.
3. A FOURTH PIN-TIE FUNCTION: `c0_pin_check` joins `t080_pin_reading`, `t081_pin_reading`, `t082_pin_reading` -- four
   copies of (pin, re-derive, refuse on disagreement). The P2 merge the route ledger already carries now has four
   members; this arc does not perform it (one change per arc) and names it.
4. THE 20 ITEMS ARE THE AUDIT'S, not a representative sample of everyday reasoning (2.7); SC-013 owns the criterion.
5. WHAT `c0_pin_check` COVERS (2026-09-22, panel I3): per-row class re-derivation only (every row in `per_seed` and
   `sequential`, through the shipped `classify()`), plus the zero-row refusal. It does NOT check row count, seed set,
   histogram, fixture histogram, conjuncts, reds, verdict, protocol, warm_ticks, head or dirty: a panel probe mutated a
   copy of the pin those ways and the check passed each (re-run 2026-09-22: 5 missed shapes, 2 caught, the zero-row
   refusal caught). Those shapes are covered OFF this function -- the pin is git-tracked, so each is a working-tree diff
   caught by the clean-tree rule and by `policy_manifest.certify`'s dirty flag, and the campaign refuses to overwrite an
   existing pin. Widening the check belongs with opening 3 and is its own arc, because a new refusal criterion added to
   a governed instrument after its campaign owes a reachable red and a control. The registry half: `CAPABILITY_BAR`'s
   brain 0.0 is a source literal tied to the pin by the provenance string and a tracked file, by the registry's stated
   literal-over-derivation convention; re-deriving it would make the gate agree with itself.
6. THE SCRAMBLED-KEY CONTROL IS ONE PERMUTATION (2026-09-22, panel I4), and its bar would be exceeded by some others
   (2.10 (e)); a successor battery whose items do not share an answer token retires it.
7. `TRACKED_SURFACES` OMITS `docs/RESEARCH_ASSESSMENT.md` AND `docs/ACADEMIC_RESEARCH_PLAN.md` (2026-09-22, panel PH1),
   which discuss C0. Measured 2026-09-21 at 678a26a: neither violates the rule -- 0 same-line hits tree-wide over 5
   distinctive C0 terms, and the one nearby comparison word (`ACADEMIC_RESEARCH_PLAN.md`, its "Do not claim ...
   human-level reasoning" line) is the prohibition itself. Widening the matcher and/or the tracked set is a NEW ARC with
   its own pre-registered reds, because the term set is a choice with a measured false-positive cost ({everyday
   reasoning}: 0 false positives, 3 of 4 phrasings caught; a +/-2-line window: 0, 4 of 4; the bare term "reasoning" reds
   the prohibition line).
8. THE ARCHIVE'S QUESTION TEXT IS AN UNVERIFIABLE-IN-TREE PREMISE (2026-09-22, panel D1): no in-tree artifact carries
   it. If a mechanical guard is ever wanted it is new code under `src/` and its own arc, named a DRIFT RATCHET over
   `ITEMS` (it detects a later edit of the in-tree strings), never the provenance chain.
9. THE `_tree_state` IMPORT COULD BE HOISTED TO MODULE TOP (2026-09-22, panel D5): `policy_manifest` imports stdlib only,
   so a rename would then fail at module import (pytest collection, run_selfchecks, any import) instead of after the
   campaign; it is a `src/` edit whose effect on the p6 gate's declared-route diff, the wiring manifest and the dormancy
   gates was not run, so it is filed with opening 3, not folded.
10. RULE 3 (`FALLBACK`) HAD NO TEST UNTIL THIS FOLD (2026-09-22, panel I10): its three substrings were accepted before
    the campaign (2.10 (c), 3.5 item 5) and rule 3 sits above the key by 2.3, so a reply that answers correctly and also
    carries a mark scores FALLBACK -- the accepted conservative behaviour, now pinned by two tests (each bare mark ->
    FALLBACK; a mark beside a key hit -> FALLBACK, the unmarked reply CORRECT); with rule 3 deleted in memory the 29
    pre-existing tests all still pass and the six new assertions fail (re-run 2026-09-22). The emitters of those strings
    exist in `cortex/recognizer_bank.py` and `cortex/live_brain.py`; their reachability through `Brain.respond` today is
    not established, and no pinned cell is FALLBACK.

### 3.7 The panel (lenses 2026-09-21, refuters 2026-09-21/22) and its fold (2026-09-22)

SHAPE. SNDL-PROTOCOL section 6, by the `Agent` fan-out fallback (the Workflow tool refused the 22,630-character panel
script): three Opus lenses on the frozen head `678a26a` (tree clean) -- pre-registration-honesty (5 filed: 2 MAJOR, 3
MINOR), disposition (5 filed: 4 MAJOR, 1 MINOR), instrument (12 filed: 6 MAJOR, 6 MINOR); 22 filed, two duplicates (D3 =
PH1, I2 = D1) -> 20 distinct claims. One refuter per claim, judging the CLAIM and the proposed FIX as two separate claims
(obs #2146): wave 1 (PH1-PH5, D1-D5, I1, I3, I4) Opus, 2026-09-21/22; wave 2a (I5-I7) Codex `gpt-6-astra` / reasoning
`high` through omnigent's Polly profile, 2026-09-22 00:xx CDT; wave 2b (I8-I12) Codex `gpt-6-astra` / `high` through the
`codex` MCP tool on the native Windows path, 2026-09-22 10:29-10:55 CDT, each in a scratch cwd with writes confined
there (owner routing 2026-09-22: Claude orchestrates, Codex executes). Every verdict was read from
`refute/<tag>/verdict.json`; the panel scratch is WSL `~/sndl-scratch/c0-panel/` mirrored at the 2026-09-22 session
scratchpad (`c0-panel/`; `panel_ledger.md` carries the per-tag rows and the reproduction column).

VERDICTS. STAND, 12: PH1 MINOR (narrowed), D1 MAJOR, D4 MINOR, D5 MINOR (fix CORRECT with one added clause), I1 MAJOR,
I3 MINOR, I4 MINOR, I6 MAJOR, I7 MINOR, I8 MINOR, I9 MINOR, I10 MINOR (fix CORRECT). REFUTED / DROP, 8: PH2 (PROVENANCE
indexes the telemetry surface -- 39 of 40 snapshot keys plus the `pci` tombstone -- and the battery publishes no such
key by construction, fact 4), PH3 (2.10 (a) is exact on its across-seeds predicate; the recalled-row intersection is
{8, 10, 12, 13, 17}), PH4 (3.3's verdict carries its qualifier), PH5 (the two-hold split was in L12), D2 (3.5 item 9),
I5 (2.10 (h)), I11 (3.5 item 11), I12 (a suppressed-positive scorer reads (k)'s constructed-CORRECT arm 0 of 20 while the
WRONG and observed arms stay 20 of 20, and the `any_tag_is_correct` scorer mutation reads (h) 15 of 20 on a fresh seed-0
run whose raw replies and routes are unchanged, items 8, 10, 12, 13 and 17 moved -- exactly the port defect (h) exists to
detect; re-run 2026-09-22). Of the 12 standing claims, 10 proposed lens fixes were WRONG and 2 CORRECT: the fold adopts
the REFUTERS' minimal fixes throughout (obs #2146).

REPRODUCTION. Every number written into 2.10 (e)-(h), 3.3, 3.5 items 7-11 and 3.6 items 5-10 was re-run by the
orchestrator on this host on 2026-09-22 with the named script before it was written, including the D2 refuter's scoped
copy of the 2026-09-17 independent verifier against the fold head (3.5 item 9: 0 disagreements over 120 rows). Two
refuter scripts misbehaved after their figures printed and are noted where their figures are used (I4's section 6; the
unmodified D2 verifier).

THE FOLD (branch `c0-panel-fold`, a clean worktree at `678a26a`; stacked, one change each; BATTERY-VERIFIED ONLY, not
re-panelled -- section 11's granted deferral, stated in each commit): `a0f6a83` the module docstrings (D1, I1, D4, I3,
I8; the executable AST byte-identical to `678a26a`, verified); `adf8929` the tests (D1, D4 docstrings; D5's coupling
test; I10's two rule-3 tests; 32 pass, ruff and format clean, the bare invocation `C0 OK`); this commit, the record
(2.10 (e)-(h); 3.1's marked correction; the 3.3 line; 3.5 items 7-11; 3.6 items 2 (qualification) and 5-10; this
section). The Codex executor applied the two code-side commits' edits under a bounded brief (`fold_brief.md`); the
orchestrator reviewed the diff, re-ran the gates and committed. No `classify` rule, key, bar, pin, campaign or registry
change anywhere: every refuter confirmed that no pinned class moves. Why a branch: the main checkout carried the owner's
uncommitted Brian2 work (46 files, none of them C0's) and the daemon exits and relaunches on any HEAD change, so the
fold is stacked on a worktree and fast-forwarded onto `main` after the owner disposes that work.

OPENINGS FILED HERE, beyond 3.6: the panel's two duplicates (D3, I2) were merged rather than assigned; the wave-2
refuters' `graphify explain` failed under the Codex sandbox (its Python launcher could not load codecs), so they read the
saved graph and report directly; the C0 closing report to Codex (protocol section 7) follows this fold and precedes the
battery.
