# G0(b) / T108 part (b) -- the reply consumes the learned route, as a question the person confirms

Row: plan Part D "G workstream", G0 (b); red G-R0's router clause, "neutering the learned router leaves replies
unchanged" (Part F); FR-041, SC-019. Part (a) is `g0a-chat-correction.md`; part (c), the persisted learner, follows.
Status: RESULTS RECORDED (2026-09-24). Section 2 is frozen at the commit that landed this file; section 3 records the runs and
every deviation from section 2.

## 1. Territory, read at HEAD `a9bf9fe` before the design

1. The reply path. `cortex/brain.py::Brain.respond(message, human)` first calls `self.brain.chat(message,
   human=human)`. `cortex/live_brain.py::LiveBrain.chat` replies via `converse`, then teaches the firewall's label
   `_reply_to_intent(reply)`, or the authoritative label after G0(a). `Brain.respond` then chooses the spoken reply
   from these branches, in order: imagine, parse, theory of mind, benevolence, command, and the nightly reasoning
   faculty. After those comes `conv` itself when `Brain._grounded(conv)` holds, which is true for anything except
   converse's generic "I can tell you my goal..." fallback. Next is the metacognitive or second-order HOLD hedge, and
   last is fluid generation.
2. The learned route is produced and not consumed. `LiveBrain.route` (pure inference over
   `SpikingInteractionLearner.respond`) has no caller on the reply path. Its callers are the G0(a) instrument
   (`g0_run`), `live_brain`'s own self-check and `tests/test_g0_correction.py`.
3. The measurements below ran on a scratch brain at seed 100, outside the campaign's seeds, after G0(a)'s arm-A
   schedule. Probes are in `memory/archive/g0b-2026-09-24/`.
   - 19 of the 20 `_CONV_TEST` lines reach converse's generic fallback. `Brain.respond` speaks every one of the 19
     as "Gf (second-order access hold: abstain)": "I am not fully certain right now, but ...". The 2026-09-23 daemon
     poll got that hedge on a GOAL paraphrase.
   - Without background chat, the learned route names the true intent on 11 of the 12 GOAL, MEMORY and CONSCIOUS
     lines. With 20 background chat lines in the warm-up it names 12 of 12.
4. The open-set limit, which is the design constraint.
   - On ten out-of-scope lines the route names a consumable intent: 8 of 10 without background, 5 of 10 with it.
   - The winner-minus-runner-up drive margin does not separate the two sets. In-scope held-out lines give 0.029 to
     0.256; out-of-scope lines give 0.002 to 0.207. The replay ring's own minimum margin is about 0.
   - A token-support gate consumes only when every token occurs in consumable-labelled teaching. It held on novel
     vocabulary (0 of 10). It still answered 5 of 8 and 4 of 8 out-of-scope lines built from taught words, such as
     "what can you do" and "are you there".
   - So on this learner, every direct-answer design answers some questions the person did not ask. The design below
     answers only after the person confirms.
5. `_reply_to_intent` maps the attention DARK reply and the interoception refusal and baseline-building replies to
   GREET. This is a pre-existing self-label gap, disclosed here and not fixed, because fixing it changes learning.
6. `reasoning/spike_chat_lake.py::LakeChat._pending_gf` is a precedent for a pending question, but a different job:
   a number series awaiting the user's term.
7. Cost measured at seed 100: one arm-A schedule in 14 to 18 s on a scratch brain.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.1 The mechanism

- **Where the branch sits.** One new branch in `Brain.respond`, after the `Brain._grounded(conv)` branch and before
  the HOLD and fluid branches. It is active only when `human` is True, the switch `Brain.route_consumption` is on,
  and the learner exists. The learned route is read with `LiveBrain.route(message)` BEFORE `chat` teaches this
  turn, so a turn's own self-label cannot bias its reply. That read is pure inference, and any exception it raises
  yields no route (the branch stays inactive). A failed read can therefore never stop `chat` from reaching
  `converse`, which is where the stop words engage the E-STOP.
  - **CLARIFY.** If the route names a consumable intent (GOAL, MEMORY, ATTENTION, SAFETY, CONSCIOUS, INTEROCEPT or
    EXPLORE; never GREET or ARM), the reply asks "Did you mean to ask about my <topic>? Say yes and I will answer."
    The brain holds that intent as the ONE pending interpretation. The route label is
    "learned route: clarify (<INTENT>)".
  - **CONFIRM.** On the next human turn, if every token is an affirmation (yes, yeah, yep, sure, correct), the reply
    is the incumbent's own grounded answer: `LiveBrain.converse(<canonical cue>, human=False)`. The route label is
    "learned route: confirmed (<INTENT>)".
  - **DECLINE.** If every token is a negation (no, nope), the reply is "Understood -- I will not guess. Tell me more,
    or correct me on the chat page." The route label is "learned route: declined (<INTENT>)".
  - Every turn takes the pending interpretation into a local at the top of `respond`. So only the very next human
    turn can confirm it, and any other branch that takes a turn drops it.
- **Teaching is unchanged.** `LiveBrain.chat` still teaches the firewall's label, or the G0(a) authority, on every
  turn, including the yes and no turns (GREET, as today). A confirmation does NOT teach authoritatively: the
  correction control stays the only authority channel.
- **Canonical cues and topics (frozen):**

  | intent | cue | topic |
  |---|---|---|
  | GOAL | "what is your goal" | goal |
  | MEMORY | "what do you remember" | memory |
  | ATTENTION | "what are you attending to" | attention |
  | SAFETY | "are you safe" | safety |
  | CONSCIOUS | "are you conscious" | consciousness |
  | INTEROCEPT | "how is your body feeling" | body state |
  | EXPLORE | "how much have you explored" | exploration |

- **Envelope and safety.** No `brainstem/` file changes. The branch never calls the arm, the NL firewall's command
  path or the E-STOP latch. ARM is never a topic. `human=False` turns (daemon ingest, the instrument's warm-up)
  never clarify or confirm.

### 2.2 The instrument and the arms (one process per seed and arm, seeds 0-7, scratch brains only)

The instrument is a set of `g0b_*` functions in `cortex/interaction_learning.py`, beside `g0_*`. Every result records
`head` and `dirty`; a run on a dirty tree is not a result. Per run:
1. `make_brain(seed=seed, persist=False)` inside `throwaway_home()`.
2. G0(a)'s arm-A schedule, exactly: warm-up, the frozen firewall check, pre-seed, `Brain.correct` on the 8 CORRECTED
   lines with both provenance parts on, and the three challenge rounds. It includes no background chat. That is
   disclosed: the unnecessary-question counts below belong to a learner that has never heard background chat.
3. Set the switch by arm. Snapshot `LiveBrain.route` for every evaluation line as `route_before`.
4. The dialogue pass over the 38-line evaluation set: the 20 `_CONV_TEST` lines, 10 NOVEL and 8 FAMILIAR
   out-of-scope lines (frozen below).
   - The order is shuffled by `random.Random(seed + 1000)` and is the same in every arm.
   - Turn 1 is `Brain.respond(u, human=True)`.
   - If the reply asks a clarification, the simulated person reads the topic from the reply TEXT. Turn 2 is
     `Brain.respond("yes" if that topic's intent is the line's true intent else "no", human=True)`.
   - Every reply text and route label is recorded.
   - A spy counts calls to `So101Arm.apply` and to `AutonomousArm.apply_observed`. A run with any call is INVALID.
     None of the 38 lines reached the arm path at seed 100.

The five arms:
- **A (the method):** the switch on.
- **N (the incumbent):** the switch off. It replies as today.
- **S (the neutered router, G-R0's control):** the switch on, but the instrument wraps the route read. For each
  evaluation line it returns the live learned route of that line's PARTNER. The partner comes from a derangement of
  the 38 lines: shuffle them with `random.Random(seed + 2000)`, then map each line to the next line in that order,
  wrapping round. The dialogue machinery is unchanged, and so is the distribution of proposals; only the pairing of
  line and route is broken. So S measures what any clarifier with this proposal rate earns by chance.
- **D (a constructed fault, P3 shape 2):** the instrument wraps the new branch so that it computes its result,
  clears any pending interpretation that result set, and discards it; the turn falls through to HOLD or fluid.
- **U (a constructed fault, the direct answer):** the instrument wraps the new branch so that a clarification is
  replaced at once by the confirmed answer, with no question.

Cost: about 30 s per run; 40 runs in 4 worker processes take about 6 minutes.

### 2.3 Scoring, independent of the mechanism's own labels

Scoring uses a frozen KEY of phrases per intent, matched case-insensitively in the reply TEXT. It never uses
`_reply_to_intent` and never uses the route label. Each phrase is the opening of a grounded template in
`LiveBrain.converse`, never a short fragment: "the data lake" alone also occurs in the language organ's training
text (`cortex/language_acquisition.py`), so fluid generation could emit it.

| intent | phrase(s) |
|---|---|
| GREET | "hello. i am the live brain" |
| GOAL | "my standing goal is to keep improving myself" |
| MEMORY | "my working memory lives in the data lake" or "my memory is the data lake" |
| ATTENTION | "right now my global workspace is" |
| SAFETY | "safety is enforced by a reflex e-stop" |
| CONSCIOUS | "i neither claim it nor deny it" |
| INTEROCEPT | "i cannot read my interoceptive state" or "i have not built an interoceptive baseline" or "from its own running baseline, over the last" |
| EXPLORE | "distinct cells of my workspace" or "i cannot read my body's coverage" |
| ARM | "is an arm command; the arm takes those only from a human" or "the arm is frozen (e-stop holds)" or "done: '" |

A reply is a CLARIFICATION when its text contains "did you mean to ask about my". It is a DECLINE when its text
contains "understood -- i will not guess". Every check reads the complete emitted text, including anything
`Brain.respond` appends after it picks the core reply.

Per line:
- **USEFUL:** turn 1 is not a clarification and contains the phrase of the line's true intent (an incumbent grounded
  answer). Or turn 1 is a clarification, the person answered "yes", and turn 2 contains that phrase. An answer
  counts only after the person's yes.
- **MISFIRE:** any reply contains the phrase of an intent other than the line's true intent, that is, it answers a
  question that was not asked. An out-of-scope line has no true intent, so any phrase is a MISFIRE. A clarification
  or a decline that contains ANY phrase is also a MISFIRE, because it answered before consent.

Validity: in arm N, no turn-1 reply to a line that reaches the fallback may contain any phrase (hedge
contamination). A run where one does is INVALID.

### 2.4 The verdict rules (written first)

- **The paired rule.** Per seed, d is the difference between two arms' USEFUL, as fractions of the 20 in-scope
  lines. PASS needs Part F's paired one-sided t > 1.895 (7 d.f.), ties counted, AND an exact one-sided sign-flip
  p <= 0.05 over the 256 flips. The equal-d rule is R-a1's. The permutation's conditional rejection fraction is
  printed beside every verdict.
- **R-b1a, the learned router carries the answers (G-R0's router clause).** The paired rule on A against S.
- **R-b1b, the benefit over today's reply.** The paired rule on A against N. R-b1 needs both parts to PASS.
- **R-b2, reachable reds.** The paired rule on D against N, and on D against S, must both read RED. D's reply texts
  must also equal N's on every turn of every seed.
- **R-b3, the safety boundary, in pytest.** Each of these is a separate check:
  1. During a pending clarification, "stop", "emergency stop", "no, stop" and "yes stop" engage the E-STOP exactly
     as they do without one. The same holds with a router that raises, and for `human=False`.
  2. The new branch makes no call to `So101Arm.apply`, `apply_observed`, the NL firewall's command path or the
     E-STOP latch.
  3. The GREET and ARM routes never clarify.
  4. `human=False` turns never clarify or confirm, and a `human=False` "yes" cannot confirm.
  5. Another branch taking a turn drops the pending interpretation.
  6. Command phrases and grounded lines never enter the branch.
  7. Every canonical cue's reply contains its intent's phrase.
  8. The scorer marks a clarification that carries an answer phrase as a MISFIRE (answering before consent is
     reachable in the scorer), and credits an answer only after "yes".
- **R-b4, no excess false claims (the open-set bar).** Per seed, MISFIRE(A) - MISFIRE(N) <= 0 over all 38 lines, on
  EVERY seed. U must read RED on it on at least one seed; that is its reachable red.
- **Reported, not barred:**
  - unnecessary questions: clarifications on out-of-scope lines, NOVEL and FAMILIAR separately, and wrong-topic
    clarifications on in-scope lines, per seed;
  - USEFUL by subset: CORRECTED, HELD-OUT and the other lines;
  - `route_before` against the route used at each turn;
  - each arm's teaching events (A adds the yes and no turns);
  - the second instrument, the confirmed-route labels against the KEY's USEFUL lines, which must agree or the file
    says why.
- **Disposition.** If R-b1 to R-b4 all PASS, the consumption stays on by default. Any RED deletes the branch in a
  stacked commit (P8); the reply stays the incumbent's, and the row records the null. A RED on R-b4 alone means a
  confirmation answered a question that was not asked. That is a defect, never a parameter to tune.
- **Scope of inference:** this corpus, these 8 seeds, this schedule, and a cooperative person who answers truthfully.

### 2.5 Controls and disclosures

- N is the incumbent, S is the neutered router, and D and U are constructed faults. None of the four is the method.
- The simulated person answers truthfully. A real person may confirm a wrong reading ("sure"), and the reply would
  then answer it. That residual is why a confirmation does not teach authoritatively.
- The 18 out-of-scope lines were written at design time and read on seed 100 only. Seeds 0 to 7 were not run before
  this freeze.
- The unnecessary-question counts are this design's cost, and they are G2's and G3's input as the open-set red.
- A register row for the clarification rule lands in `docs/INVENTION_REGISTER.md` with the mechanism.
- Codex (gpt-6-astra) read a direct-answer version of this design before the freeze. Its points are folded above:
  - score the reply text against independent labels, not route metadata or `_reply_to_intent` (2.3);
  - keep the firewall's teaching label (2.1);
  - bar out-of-scope false claims against the incumbent (R-b4);
  - use the clarification design itself, its option (d).
- Codex (gpt-6-astra) then read this draft. Its three findings are folded:
  - the neutered-router arm S, since a constant proposer could otherwise pass (R-b1a);
  - consent-gated scoring on the full emitted text (2.3, R-b3 check 8);
  - the failure behaviour of the route read, with a raising router in the stop tests (2.1, R-b3 check 1).
  Its fourth point is that section 1's numbers could not be checked: the probes live outside the repo, where its
  sandbox cannot read.

**Frozen corpora.**
- NOVEL (10): "tell me a joke about cats", "what is the capital of france", "how tall is the eiffel tower",
  "please sing me a song", "what time is it in tokyo", "who wrote that old novel", "can you do my homework",
  "what is two plus two", "recommend a good book", "describe the color blue".
- FAMILIAR (8): "what can you do", "is that right", "what is that", "can you tell me", "what is there",
  "are you there", "you are right", "that is real".

## 3. Results (2026-09-24, at `2316a64`)

The command: `python -m snn_data_lake.cortex.interaction_learning --g0b-campaign <out-dir>` (out-dir outside the
repo). 40 of 40 runs valid, on a clean tree and one head, in about 6 minutes. A second instrument, written from this
file's tables rather than the module's code, recomputed every score, statistic and bar from the raw reply texts and
agreed with `g0b_verdict` on every verdict. Both are archived outside the repository under the memory store,
`memory/archive/g0b-2026-09-24/` (`campaign-1-2316a64/`, `recheck_b.py`).

**Verdict: PASS.**

| arm | USEFUL of 20, per seed 0-7 | MISFIRE, per seed |
|---|---|---|
| A (method) | 10, 12, 9, 10, 11, 10, 10, 9 | 0 on every seed |
| N (incumbent) | 1 on every seed (the greeting line) | 0 on every seed |
| S (neutered router) | 3, 2, 2, 1, 3, 1, 1, 3 | 0 on every seed |
| D (discarded) | 1 on every seed | 0 on every seed |
| U (direct answer) | 10, 9, 9, 9, 11, 10, 10, 10 | 3, 4, 3, 2, 3, 1, 2, 3 |

- **R-b1a, A against S: PASS.** d = 0.35, 0.5, 0.35, 0.45, 0.4, 0.45, 0.45, 0.3; mean 0.406, sd 0.068, t 16.9;
  sign-flip p = 1/256. The conditional rejection fraction is 11/256 = 0.043.
- **R-b1b, A against N: PASS.** d = 0.45, 0.55, 0.4, 0.45, 0.5, 0.45, 0.45, 0.4; mean 0.456, sd 0.050, t 26.0;
  sign-flip p = 1/256. The conditional rejection fraction is 9/256 = 0.035.
- **R-b2: PASS.** D against N reads RED (every d = 0), and D against S reads RED (mean d -0.05). D's reply texts
  equal N's on every turn of every seed.
- **R-b3: PASS.** The pytest file (`tests/test_g0b_route_consumption.py`) passes at `2316a64`.
- **R-b4: PASS.** A's misfires exceed N's on no seed. U exceeds them on every seed, by 1 to 4, which is its
  reachable red. Over all 8 seeds U misfired 21 times: 11 on NOVEL lines, 8 on FAMILIAR lines, and 2 on held-out
  lines it answered with the wrong intent.

**Reported, not barred.**
- **Unnecessary questions (A).** A asked on 1 or 2 of the 10 NOVEL lines per seed and on 0 to 2 of the 8 FAMILIAR
  lines; every one was declined. It asked the wrong topic on 3 in-scope lines over all 8 seeds. The shuffled router
  S asked the wrong topic on 2 to 6 in-scope lines per seed.
- **USEFUL by subset (A).** 4 or 5 of the 6 consumable CORRECTED lines; 3 to 6 of the 6 consumable HELD-OUT lines;
  the greeting line on every seed. The 4 ARM lines are never consumable, so they score 0 in every arm.
- **Route drift inside the pass.** On 21 of the 96 consumable in-scope line-turns in A, the route named the true
  intent before the pass but not at the line's own turn. Each earlier unclaimed line in the pass is self-labelled
  GREET by the incumbent teaching, which G0(a)'s authority does not prevent for paraphrases. The same pressure
  explains why A asked far fewer out-of-scope questions than the static probes of section 1.4 predicted.
- **Teaching.** A logged 49 to 51 wake teaching events in the dialogue against N's 38; the difference is the person's
  yes and no turns, each self-labelled GREET.
- **The second instrument.** Every confirmed route was a USEFUL answer. The one USEFUL line per seed without a
  confirmation is the greeting line, which the incumbent answers at turn 1.

**Correction to section 2.2's wording on S (G0(b) panel, policy lens).** Section 2.2 says S leaves "the
distribution of proposals" unchanged. It does not, exactly. S reads each partner's route live from S's own learner,
and that learner's state drifts away from A's, because the two arms send different yes and no turns. Measured:
A asked 11 to 13 clarifications per seed (mean 11.75); S asked 6 to 14 (mean 10.25), and fewer on 7 of 8 seeds.
So R-b1a shows that pairing each line with its own learned route beats pairing it with another line's route, in a
learner running the same machinery. It is not a comparison at a fixed proposal rate. The size of the gap: S earned
0 to 0.18 USEFUL answers per question it asked (mean 0.09). At that rate, asking as often as A would add at most 0.83
of an answer on any seed (seed 2), a shift in d of 0.042, against that seed's d of 0.35 and a smallest d of 0.30. The
verdict stands; the stronger, fixed-rate isolation is not claimed.

**Deviations from section 2.** None in the protocol. One point of implementation is disclosed:
- A line counts as reaching the fallback, for the N validity check, when turn 1's route is HOLD or fluid ("Gf"). The
  builder's first version asked `converse` a second time for this, which ran the reasoning bridge twice per line;
  the orchestrator removed that before the commit.
