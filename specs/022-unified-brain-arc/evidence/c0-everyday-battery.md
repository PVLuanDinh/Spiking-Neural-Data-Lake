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

`cortex/everyday_battery.py`: `ITEMS` (the archive's 20 questions verbatim, asserted equal to the fixture's at import),
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

### 3.6 Openings

1. TWO HOLDS PRE-EMPT THE ANSWER PATH, BY SEED (C2's territory): the second-order access hold on seeds 0-2, the
   metacognitive hold on seeds 3-4, 14 abstains each. C2's pre-registration should read which hold fires and why before
   it routes on `{known, coverage}`.
2. THE KEY IS A FLOOR (2.2): it will miss a correct answer phrased outside its patterns; the campaign reports every
   answer-shaped non-CORRECT row with its text (none today) so a miss is visible the day the brain starts answering.
3. A FOURTH PIN-TIE FUNCTION: `c0_pin_check` joins `t080_pin_reading`, `t081_pin_reading`, `t082_pin_reading` -- four
   copies of (pin, re-derive, refuse on disagreement). The P2 merge the route ledger already carries now has four
   members; this arc does not perform it (one change per arc) and names it.
4. THE 20 ITEMS ARE THE AUDIT'S, not a representative sample of everyday reasoning (2.7); SC-013 owns the criterion.
