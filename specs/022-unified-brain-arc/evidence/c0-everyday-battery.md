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
