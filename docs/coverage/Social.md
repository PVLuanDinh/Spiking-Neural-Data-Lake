# Coverage evidence -- Social

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| tom_first_order_false_belief | 40 | src/snn_data_lake/faculties/theory_of_mind.py:101#TheoryOfMind; src/snn_data_lake/engine/brain.py:3950#_tom_report | - | auditor:Social live-brain reply probe (make_brain seed=0, 5 ticks + 7 dialogue turns) | 2026-07-30 |
| tom_belief_guided_caution | 45 | src/snn_data_lake/engine/brain.py:1737#_route_percept_to_action; src/snn_data_lake/engine/autonomous_loop.py:903#_bg_select | social_tom_caution_60.txt | refuter:Social -- band 61 REFUTED to 45 / social_tom_refute_probe.py arm 60 at seeds 0,1,2: 23, 0 and 4 action flips, seed 1 exit 1 CONSUMED-BUT-NOT-DECISIVE (social_tom_refute_seed1_60.txt) | 2026-07-30 |
| empathy_affective_contagion | 40 | src/snn_data_lake/faculties/affect_core.py:198#empathize; src/snn_data_lake/engine/brain.py:3726#respond | - | auditor:Social live-brain reply probe (empathy clause observed in the reply) | 2026-07-30 |
| pragmatics_implicature | 40 | src/snn_data_lake/reasoning/pragmatics.py:356#implicature; src/snn_data_lake/reasoning/spike_chat_lake.py:1326#converse | - | auditor:Social method-instrumented live chat -- 5 calls in 6 turns, candidate won the reply twice | 2026-07-30 |
| tom_second_order | 35 | src/snn_data_lake/faculties/affect_core.py:210#SecondOrderToM; src/snn_data_lake/engine/brain.py:3945#_tom_report | - | auditor:Social live-brain reply probe (2nd-order clause observed in the reply) | 2026-07-30 |
| tom_recursive_depth_k | 30 | src/snn_data_lake/faculties/recursive_tom.py:82#RecursiveToM; src/snn_data_lake/engine/brain.py:3878#_tom_update | - | auditor:Social live-brain probe -- charged at depth 2 over 2 agents; depth-k/N-agent/non-location capability never exercised | 2026-07-30 |
| empathy_help_drive_to_action | 25 | src/snn_data_lake/engine/brain.py:1737#_route_percept_to_action; src/snn_data_lake/engine/brain.py:1562#_route_percept_to_action | help_60.txt | auditor:Social / tools/charge_probe.py help 60 (exit 2) + a chat-live pinned-weight neuter at 0/60 | 2026-07-30 |
| pragmatics_speech_act | 15 | src/snn_data_lake/reasoning/pragmatics.py:269#act; src/snn_data_lake/reasoning/pragmatics.py:590#classify | - | auditor:Social method-instrumented live chat -- act() called 0 times in 6 turns while implicature() got 5 | 2026-07-30 |
| joint_attention | 0 | - | - | auditor:Social tree-wide search for shared/triadic attention, referential pointing, gaze-following | 2026-07-30 |

**Mean over the nine rows = 270/9 = 30.0.** Before the adversarial pass recorded below it was
286/9 = 31.8; one row fell, 61 -> 45, and nothing else moved.

*DATED NOTE 2026-09-16 (specs/022 T082): the `tom_second_order` row's citation above is STALE and is left standing rather than rewritten -- the class it names was RETIRED at T082 (2026-09-16) under P2, after R5 measured the surviving recursive tracker reproducing its `predict_second_order` readings on 200 constructed witness scripts (1,200 of 1,200 ordered pairs). The second-order capability is not lost: it is read from `faculties/recursive_tom.py`'s `predict_nested` on the same dialogue path, and the merged feed is symmetric, so the spurious nested divergence this row's faculty could not see is gone (0 of 60 episodes per seed against 5 of 12 on the retired feed; `evidence/t082-social-cognition.md`).*

---

## Adversarial pass: the one band-41+ claim fell, 61 -> 45, and the category went 31.8 -> 30.0

`COVERAGE_RUBRIC.md:92-95` hands every score above 40 to a refuter who defaults to refuted. This
category had exactly one such row. Five new recorded artifacts, all reproducing under
`coverage_score.py --rerun`, are attached; the probe is
`docs/coverage/artifacts/social_tom_refute_probe.py` (ruff check + ruff format clean).

**What survived the attack**, first, because it is most of what was claimed:

- **The artifact is genuine, not a hand-written transcript.** `--rerun social_tom_caution_60`
  reproduces exactly: recorded exit 0 / now 0, observables
  `(('0','60'),('24','60'),('0','60'),('0','60'),('0','60'))` unchanged. The three in-process
  falsification controls really do run and really do return 0/60.
- **The neuter is airtight and the data-descriptor reasoning is right.** Re-implemented
  independently, it reproduces 0/60 act and 24/60 trajectory at seed 0 to the tick.
- **Eligibility holds.** The signal reaches 0.34 on 30 of 60 ticks in the wired arm at every seed
  tried and 0.0 in the neutered arm, so a negative here is a real negative and not a dead wire.
- **The roster is not padded**, and the one omission that would have flattered the mean checks out:
  `readout/interaction_learning.py`'s only non-self-check constructors are
  `faculties/language_self_build.py:83` and `cortex/readout_fair_compare.py:65`, so it is on Language's
  roster rather than this one. The nine-row split really does score lower than a four-row
  per-faculty-maximum roster, as the section below shows.

**What fell: five measurements the audited probe did not take.**

1. **The charge does not survive a second seed.** `social_tom_refute_seed1_60.txt` runs the same
   wire, the same dialogue and the same 60 ticks at seed 1 and returns **CONSUMED-BUT-NOT-DECISIVE,
   exit 1 -- 0/60 act, 0/60 trajectory, 0/60 selected action** -- with the signal live at 0.34 on 30
   of 60 ticks and the neuter verifiably held. Seed 2 (`social_tom_refute_seed2_60.txt`, exit 0)
   charges but barely: 8/60 trajectory, **4/60 action**, first divergence at tick 52 of 60. Across
   three seeds the wire's behavioural effect is 23 action ticks, then 0, then 4. Band 61-80 requires
   that neutering the signal make the action "measurably differ"; at seed 1 it measurably does not.
   `Executive.md:240-243` named single-seed as "the single biggest caveat on the three 61s" in this
   sweep. Spent here, the second seed did not explain the result -- it removed it.
2. **The 24/60 is one divergence restated 24 times.** `social_tom_refute_seed0_60.txt` adds the two
   columns the audited probe lacks: onset is tick **36** and the trajectory diff is **contiguous**
   from 36 to 59, so `24 == 60 - onset`, the identity `Executive.md:23-52` proved across seven A/Bs.
   The selected action diverges on ticks 36-58 and re-converges at 59. One decision changed, at one
   tick; everything after it is two brains in different states.
3. **The signal carries one bit, so "belief-guided" is not what was measured.**
   `social_tom_trigger_forms.txt` puts ten surface forms of the SAME false-belief event through a
   live `respond()`. Four fire and six do not -- passive voice, a verb outside the hand-written list,
   a third-party subject, a synonym for "moved", and the case where the absent party is the brain
   rather than the user. In all four that do fire the caution is **0.3400 exactly** with `fb == 1`,
   regardless of object, location, wording or which agent diverges. With caveat B below (the term
   latches at 0.34 permanently), what reaches the arm is a boolean "a regex matched once", not a
   graded belief state. *Held to my own standard:* this artifact prints no `n/N ticks` line, so its
   fingerprint is empty and `--rerun` compares only its exit code -- the ten verdicts are in the
   recorded stdout verbatim and a reader should re-run the command rather than trust the count.
4. **It is not caution.** `arm HOLD rate A(wired) vs B(neutered)` is **0.000 vs 0.000 at all three
   seeds**, and act/hold is 0/60 at all three. `brain.py:1619-1622` says the brain "HOLDS more under
   social uncertainty"; measured, holding never increases at all. What the term does is lower `da` at
   `autonomous_loop.py:730` and shuffle the choice between two non-HOLD actions. A signal that moves
   the action, but not in the direction its own mechanism claims, is a weaker charge than one that
   does.
5. **The wire is not separable from the readout rows.** `social_tom_refute_source_60.txt` neuters
   `Brain._false_beliefs` -- the ToM readout one layer upstream, the same object
   `tom_recursive_depth_k` is scored on -- and returns a **byte-identical** behavioural result to
   neutering the arm attribute: 0/60 act, 24/60 trajectory, 23/60 action, onset 36, the same
   divergent tick list. Rows 1, 2 and 6 of this table are therefore three scores over one measured
   event: a single `_TOM_PLACE` match. That is the shape of the defect that cost Executive 9 points.
   I did not restructure another auditor's roster in an adversarial pass, but the arithmetic belongs
   on the record: this 61 was **the only row in the file whose presence raised the mean** (without it,
   225/8 = 28.1), and it is the one row that is a wire rather than a mechanism.

**Why 45, and not lower, and not 0.** Band 41-60 is compelled and I am not refuting it:
`_social_risk` is read on every tick at `autonomous_loop.py:728` and `:659`, it enters `caution` and
therefore `da`, and at two of three seeds it changes the selected action. That is squarely "changes
an internal read". Band 61 is not compelled, on this sweep's own precedent --
`Executive.md:102-124` twice declined the band an exit-0 A/B licensed, landing at exactly 45, for
charges that were thin (`stn`, 2/60 actions) or fragile (`pfc_gain`, one action in 200). This charge
is 23 actions at one seed, 4 at another and none at a third. Rule 4 pushes the same way rather than
the other: the signal is 0.0 on 60 of 60 bare autonomous ticks, so nothing here runs at all unless a
human types one of the four working sentence shapes.

**The instrument finding this pass leaves behind, and it is not about Social.** A single seed is
enough to manufacture a band-61 out of a band-45 wire, the scorer cannot see it, and no artifact
format records the seed. Every 61 in this ledger was measured at seed 0 only.

---

## The roster, and why it has nine rows for four named faculties

`COVERAGE_RUBRIC.md:34` claims Social as "theory of mind, empathy, pragmatics, joint attention". The
nine rows are those four at the granularity the CODE separates them: theory of mind is four distinct
mechanisms in four different files (a first-order location-belief population, a nested second-order
tracker, an N-agent VSA depth-k tracker, and the one wire from any of them to behaviour), empathy is
two (affective contagion into the reply, and the drive into the arm), pragmatics is two (implicature
and the speech-act classifier), joint attention is absent.

This split is not padding in the sense `docs/coverage/README.md:66-70` warns about, and the arithmetic
says so rather than my assurance. Padding lifts a score by adding trivially-live rows. Scoring the four
named faculties at their best sub-mechanism -- theory of mind 45, empathy 40, pragmatics 40, joint
attention 0 -- gives 125/4 = **31.3**. The nine-row split gives **30.0**. The finer roster scores this
category LOWER, because splitting exposes the 0, the 15, the 25 and the 30 that a per-faculty maximum
would hide behind one charged wire.

**Corrected by the adversarial pass:** the claim that the four ToM rows "do not share an observable
fingerprint" was true of the artifacts on file and false of the brain. They share one --
`social_tom_refute_source_60.txt` shows that neutering the readout and neutering the wire are the same
experiment down to the divergent tick list. The scorer's W3 check could not see it because only one of
the four rows carried an artifact at all, which is a hole in the check as much as an error in the file:
**W3 can only compare rows that have artifacts, so a category with one artifact can never trip it.**

## Note 1 -- four rows are capped by the instrument, not measured to be telemetry

**Four of these nine faculties have runtime evidence that places them at band 41-60, and all four
are scored at 40 or below because no conforming A/B artifact can be produced for them.** This is the
single most important thing in this file, so it is stated before any per-faculty detail.

`tools/charge_probe.py` neuters one key of `Brain._drive_w` and watches `arm_stopped` and
`map_position`. That is the whole of its addressable surface. Of the nine faculties here, exactly one
(`empathy_help_drive_to_action`) is a `_drive_w` key, and the probe never chats, so the signals that
only a dialogue turn can raise read as permanently zero to it. Of the remaining eight, four are a
readout whose consumer is the REPLY -- which `COVERAGE_RUBRIC.md:53` explicitly names as a band-41-60
consumer, "a reply, a memory write, a self-model term" -- one is an attribute on the arm rather than a
key in `_drive_w`, one is a live module's uncalled method, one is charged on an axis other than the one
it claims, and one is absent. The probe can address none of them.

For the one faculty whose evidence reached the arm rather than the reply (`tom_belief_guided_caution`,
an arm attribute -- scored 61 on this evidence and **cut to 45 by the adversarial pass above**)
I closed the gap properly rather than quoting a number no artifact backs:
`docs/coverage/artifacts/social_tom_ab_probe.py` is tracked, ruff-clean, keeps charge_probe's
observables and exit-code contract, and **runs its own three falsification controls in the same
process and exits 2 if any of them fires** -- because a bespoke probe that can only report the result
it was written to find is not evidence. `social_tom_caution_60.txt` is its recorded run and it
reproduces under `--rerun`. What I did NOT do is record against my session-local scratchpad scripts:
that produces an artifact which dies on `--rerun` for everyone else, with the same signature as a
forged transcript.

For the remaining four, no probe I could write would be honest evidence at reasonable cost: the
consumer is a natural-language reply, so "the output changed" has no observable as crisp as
`map_position`, and inventing a string-diff criterion would be choosing my own pass mark. **A reviewer
should read those four 40s as "the ceiling this evidence standard allows", not as "measured to be
telemetry-only".** The runtime findings behind them are in the per-faculty section with their exact
commands, and they earned no band.

The remaining instrument gap is the top-ranked gap at the bottom of this file.

## Note 2 -- the one measured behavioural charge, as originally argued at 61 (now 45)

*Read this section against the adversarial pass above, which refuted the band it argues for. The
measurements in it are correct and reproduce; what fell is the inference from them.*

`faculties/recursive_tom` divergence -> `_tom_protect` -> `arm._social_risk` (`brain.py:1627`) is read as
caution by both selection paths: `autonomous_loop.py:728` inside the basal-ganglia gate and `:659` in
the non-BG fallback. Recorded in `social_tom_caution_60.txt` at exit 0. Both arms execute the IDENTICAL
dialogue, so everything else a chat turn does is held constant and only `_social_risk` differs:

    signal A(wired) max / live ticks        : 0.34 / 30 of 60
    signal B(neutered) max                  : 0.0  (held=True)
    act/hold decision changed by the signal : 0/60 ticks
    arm trajectory changed by the signal    : 24/60 ticks

**Originally scored 61, the bottom of the band, for three reasons.** The act/hold decision never
flipped -- only the resulting trajectory moved, so this is the weakest thing band 61-80 covers. The
trigger is developer-constructed typed English (caveat A). And the signal is 0.0 on every autonomous
tick, so what is charged is a conditional wire, not a standing faculty. **The refuter's verdict: all
three of those reasons argue against the band rather than for its bottom edge, and a second seed
turns the charge off entirely. 45.**

The first neuter I wrote was **leaky and I nearly reported its negative**. Assigning a
`dict`-subclass to `arm.__dict__` does not neuter anything: CPython stores instance attributes with
`PyDict_SetItem` on the internal dict pointer, which bypasses a subclass `__setitem__`, so the
"neutered" arm still read 0.34 and the run returned NOT-DECISIVE. The working neuter is a data
descriptor -- `type(arm)._social_risk = property(getter -> 0.0, setter -> no-op)` -- which wins over
the instance dict in both directions. This is the same class of defect
`tools/charge_probe.py:54-67` documents for `_drive_w`, one layer down, and it fails in the same
direction: a partial neuter returns a confident negative, and the negative is the verdict that gets a
live faculty retired.

Three falsification controls, all reporting 0/60 act and 0/60 trajectory, and all three run inside
the recorded artifact rather than beside it -- the probe exits 2 if any of them fires, so the
attached exit 0 is itself the statement that none did:

- sham descriptor on `_social_risk_sham`, with chat -- does the class patch itself perturb the run?
- neuter `_social_risk` with NO chat, so the signal is 0.0 in both arms.
- determinism: two identical unpatched runs.

The probe also refuses to give a verdict unless the neuter verifiably held (B's max signal is 0.0)
and the signal verifiably rose in A -- so a leak like the one above, or a dialogue that failed to
establish a false belief, exits 2 instead of returning a confident negative.

And the harness was validated against the control the brief names: `coverage_score.py --rerun
drive_60` reproduced exactly -- recorded exit 0, observables `(('0','60'), ('22','60'))`, 22/60
trajectory changes. The harness is not degenerate.

**Caveat A, and it is why this is not scored above 61.** The signal is identically 0.0 on **60 of 60
bare autonomous ticks**. `_false_beliefs` is written in exactly one
place, `_tom_update` (`brain.py:3210`), which is reached only from `respond()` at `brain.py:2877`, and
only when the turn matches the hand-written `_TOM_PLACE` regex (`brain.py:83`) plus one of the 13
absence idioms in `_TOM_ABSENT` (`brain.py:89`). So on the daemon's autonomous loop this faculty is
absent in effect under rubric rule 4, and it becomes decisive only once a human types a sentence of a
specific shape. Both facts are true at once and neither cancels the other: it is a real charge with a
trigger that a developer wrote by hand, which is also why band 81+ is out of reach.

*Refuter's addition, measured rather than argued:* "a sentence of a specific shape" is now a number.
`social_tom_trigger_forms.txt` puts ten wordings of one false-belief event through a live `respond()`
and **four fire**. A user who is deliberately trying to establish a false belief -- passive voice, a
synonym for "moved", a third-party subject -- raises nothing 6 times out of 10. That is how close this
precondition sits to the rule-4 examples in `COVERAGE_RUBRIC.md:72-74`, and it is why the refuter reads
caveat A as an argument against band 61 rather than an argument for its lower edge.

**Caveat B, a defect found while measuring.** `brain.py:1621` calls this a "decaying" caution and
`:1624` computes `max(min(1.0, 0.34*fb), 0.9*prev)`. `_false_beliefs` is never cleared, so `fb` stays
1 forever and the expression pins at exactly 0.34 permanently -- `max(0.34, 0.306) = 0.34`. The
measured trace is 0.34 on 30 consecutive ticks with no decay at all. It is a latch, not a decay.

*Refuter's addition:* caveat B is stronger than "a defect", it is a bound on what the faculty can be
credited with. Every one of the four firing wordings in `social_tom_trigger_forms.txt` produces
`fb == 1` and `_social_risk == 0.3400`, so `0.34 * fb` never once takes a second value and the latch
never once decays. The entire information content of this social readout, as it reaches behaviour, is
one bit that can only ever be set. That is what caps the row at 45 as much as the seed sensitivity
does.

## What each row is, and what I actually verified

**tom_first_order_false_belief (40).** `theory_of_mind.TheoryOfMind` carries the believed location as
a spiking population code with an LIF winner-take-all readout. It is live from the chat path and its
output reaches the reply verbatim -- on a real run the brain answered *"You would look for the marble
at 'basket' -- that is where you last saw it -- but it is actually at 'table'."* while
`state()["tom_live"]` read `{'obj': 'marble', 'belief': 'basket', 'world': 'table', 'false_belief':
True}`. That is a band-41-60 consumer under `COVERAGE_RUBRIC.md:53`, capped here per note 1.

Separately: `brain.py:448-461` runs a Sally-Anne script ONCE in `__init__` and freezes the boolean
into `state()["braincog"]["theory_of_mind_false_belief"]`, which read `True` on every one of 60 ticks
without being recomputed. A birth-time self-check whose result is republished as live telemetry is a
rule-4 zero, and it is the most likely origin of the "80 on a single A/B" that
`COVERAGE_BASELINE.md:32` recorded and this ledger already refuses. It is called out here rather than
given a row, because it is a telemetry artefact of this faculty and not a faculty of its own.

**tom_second_order (35).** `affect_core.SecondOrderToM` is a genuinely separate nested tracker,
observed at `brain.py:3237`/`:3259` and read at `:3290`. Verified in a reply: *"(2nd-order: and you'd
expect ME to think it is at 'basket')."* Scored below first-order because its readout fires only
inside the `bel != wor` branch of `_tom_report`, so it is strictly narrower than the faculty it hangs
off, and because `predict_second_order` is only ever asked one fixed pair, `("user", "brain")`.

*DATED NOTE 2026-09-16 (specs/022 T082): RETIRED at T082 (2026-09-16) -- the class this paragraph describes no longer exists, and the paragraph is kept verbatim as the record of what was measured when it did. P2's condition was met first: R5 fed 200 constructed witness scripts (3 agents, 4 locations, 12 events each) through both mechanisms and read `predict_nested(x, y)` equal to the retired `predict_second_order(x, y)` on 1,200 of 1,200 ordered pairs, with the comparison run against the LIVE class before its deletion. The narrow readout this row scored below first-order -- one fixed pair, fired only inside one branch of the report -- is what the merge removes: the surviving tracker is N-agent and depth-k, its output is a declared port with measured charge into the arm's caution, and its live list is published on `Brain.state()` (`evidence/t082-social-cognition.md` 2.1).*

**tom_recursive_depth_k (30).** `recursive_tom.RecursiveToM` is constructed at `brain.py:767` and
charged: it produces `_false_beliefs`, which reaches the reply at `brain.py:3127` (verified:
*"(theory of mind: I'm tracking a divergent belief for user ...)"*) and the arm via note 2. But the
capability the module exists for is not the part that runs. The live path registers exactly two
agents, calls only `predict_first_order` and `divergence`, and only ever uses the `believes`
attitude. `predict_chain`, `max_depth=3`, N>2 agents and the `wants`/`knows` attitudes -- the three
axes `recursive_tom.py:1-20` names as its reason to exist over the two older trackers -- are
exercised nowhere but the module's own `main()`. So the module is live and consumed while the faculty
in the row title is not, which is the honest reading of 30 rather than 40.

**empathy_affective_contagion (40).** `affect_core.empathize` (`affect_core.py:179`) is called at
`brain.py:3071` when a turn hits one of nine feeling cues, and the result reaches the reply --
verified: *"(empathy: sensing your feeling, my affect shifted toward it -- I now feel anger.)"*.
Note the content defect in that quote: the user said *"i am sad today"*, and `brain.py:3066-3069`
maps every negative cue to the single pair `(-0.6, 0.6)`, which lands in the anger cell of the
categoriser. High arousal is assigned to sadness by construction, so the contagion label is wrong for
the most common distress word it handles. This is a reason band 81+ (holds outside the constructed
case) is not reachable as built, independently of the artifact problem.

**empathy_help_drive_to_action (25).** The one Social faculty the tracked probe CAN address, and the
answer is negative. `help_60.txt` records `tools/charge_probe.py help 60` at exit 2 with 0/60 act
flips and 0/60 trajectory flips. The probe reports UNKNOWN rather than inert, correctly, because
`help` has no learned value table so it cannot establish eligibility from outside -- but I can
establish it from inside, two ways, and both strengthen the negative rather than rescuing it:

  1. On 60 bare ticks `_help_drive` is 0.0 on every one, so during the recorded probe the source
     genuinely never activated. The artifact's `moved=True` (weight 0.1 -> 0.102) is exactly the
     competitive `-mean_e` movement `charge_probe.py:124-135` warns is not evidence of activation.
  2. With a dialogue turn raising `_help_drive` to 0.456, an airtight pinned-weight neuter of
     `_drive_w["help"]` -- identical dialogue in both arms -- still gave **0/60 act flips and 0/60
     trajectory flips** while the neutered arm's weight held at 0.0 and the wired arm's at 0.0889.
     Consumed but not decisive, with the signal demonstrably live.

Band 25 rather than 15 because the source is on the tick every tick: it sits in the eligibility dict
at `brain.py:1361` and competes in the three-factor weight update whether or not it ever fires. Band
25 rather than 40 because its output changes nothing, measured, with the signal live.

One observation I could not attribute: `_help_drive` decayed for ten ticks (0.3876 -> 0.0898) and then
held at 0.0898 for every remaining tick I sampled, rather than continuing toward zero. A wake/sleep
phase change gating `_route_percept_to_action` is the obvious candidate and I did not confirm it. It
does not move the band -- the neuter A/B is 0/60 either way -- but a constant sitting where a computed
value belongs is this project's known silent-degradation tell and someone should look.

**pragmatics_implicature (40), and this is an upgrade with a reason.** The row currently in this
ledger scored 25 off a static correction. Runtime instrumentation of the live chat organ -- wrapping
`Pragmatics.implicature` on the class and driving six turns through `respond()` -- counted **5 calls**,
and the candidate it emits at `spike_chat_lake.py:1000-1003` won the reply outright on two utterances:
*"can you pass the salt?"* -> *"[pragmatics] literally a question, but you mean: request to pass the
salt -- an indirect request to ACT, not a yes/no question about ability"*, and *"some of the students
passed"* -> *"implicates 'not all' -- scalar implicature on 'some' (Quantity maxim)"*. It does not
merely append: it beat the somatic/metacognitive candidates for the reply body. That is a verified
band-41-60 consumer, capped at 40 per note 1.

**pragmatics_speech_act (15).** `Pragmatics.act` (`pragmatics.py:269`) is the six-class Searle
classifier and `classify` (`:590`) is the VSA rank-order Hebbian learner for the same map. Both are
inside a module that IS live -- the `Pragmatics` organ is constructed on the brain's `LakeChat` at
`spike_chat_lake.py:224` and confirmed present at runtime. Neither is ever called. The same
instrumented run that counted 5 `implicature` calls counted **0 `act` calls**, and `implicature` does
not call `act` internally. This is method-level dormancy inside a reachable module, which
`gates/dormancy_gate.py` cannot see by construction -- and I state it from the runtime counter,
not from the grep, because `COVERAGE_RUBRIC.md:65-68` records that this repo's static dormancy claims
have been wrong in both directions.

**joint_attention (0).** Genuinely absent. I searched the tree for shared or triadic attention,
referential pointing, gaze-following and attention directed at another agent's focus. Every `gaze`
hit is oculomotor with no second agent in it -- `faculties/sc_deep_layers.py` turns a selected
orienting winner into a saccade command and `faculties/gain_adaptation.py` adapts the VOR. There is
attention (`faculties/learned_attention.py`) and there are other agents (the ToM trackers), and nothing
joins them: no mechanism represents "the other is attending to X" or aligns the brain's own
attentional selection to another agent's. Citation is `-`, which the evidence standard permits at
band 0 rather than inviting an invented one.

## Capability inventory (Policy 2: what would be lost if a file were retired)

Format: **file** -- unique capability -- live?

- **`faculties/theory_of_mind.py`** -- first-order WHERE-belief as a spiking population with an LIF WTA
  readout; the Sally-Anne substrate -- live on the chat path, plus a birth-frozen flag.
- **`faculties/affect_core.py` (`SecondOrderToM`)** -- nested "A believes B believes" over the same
  belief population -- live on the chat path, one fixed agent pair.

  *DATED NOTE 2026-09-16 (specs/022 T082): RETIRED at T082 (2026-09-16) -- this bullet's class was deleted under P2 once R5 showed `faculties/recursive_tom.py` reproducing its readings on 1,200 of 1,200 ordered pairs over 200 constructed witness scripts; the chat path's nested belief now comes from that ONE tracker. The bullet above it is also narrowed rather than wrong: `faculties/theory_of_mind.py` STAYS (`reasoning/pragmatics.py`'s implicature engine runs its own instance) but its LIVE CHAT FEED and the birth-frozen flag that bullet names are both gone with the same merge (`evidence/t082-social-cognition.md` 2.1).*
- **`faculties/affect_core.py` (`empathize`)** -- affective contagion as a convex pull of own
  valence/arousal toward another's -- live on the chat path, 9 cue phrases.
- **`faculties/recursive_tom.py`** -- N agents, depth-k holder paths, VSA role-filler propositions
  beyond location, `wants`/`knows` attitudes -- module live, but the depth-k, N-agent and
  non-location axes are exercised only in its own `main()`.
- **`reasoning/pragmatics.py` (`implicature`)** -- Gricean maxim step to speaker meaning, ToM-gated
  -- live, and it wins the reply.
- **`reasoning/pragmatics.py` (`act`, `classify`)** -- 6-class Searle speech-act taxonomy plus a
  brain-native VSA/Hebbian learner for it -- no live caller, 0 calls measured.
- **`engine/brain.py:1624-1627`** -- the only wire from any social readout to behaviour -- live once
  chat has raised it; latched, not decaying.
- **(nothing)** -- joint / triadic attention -- absent.

Adjacent and deliberately NOT scored here, because they are not one of the four faculties this
category claims and another category may hold them: `readout/interaction_learning.py`
(`SpikingInteractionLearner`, R-STDP learning from conversational feedback) is constructed live --
`brain.py:201` passes `interaction_learning=True` -- and its neurons are consolidated by the sleep
plasticity cascade at `brain.py:2303`, but `teach`, `learn_turn` and `respond` have no caller outside
`live_brain.py`'s own self-check, so nothing ever teaches it and nothing reads its routing. A learner
whose synapses are consolidated nightly and never written by an interaction is worth someone's row;
it should not be mine. `sensorimotor/gesture_recognition.py` classifies approach/retreat from arm distance
captures, which is perception of an object's motion rather than a social signal.

## Gaps, ranked by distance_to_100 x feasibility

1. **A reply-level A/B criterion (highest, and it is an instrument gap, not a brain gap).** Four of
   nine rows are capped at 40 below their measured band because the rubric names "a reply" as a
   band-41-60 consumer and no instrument in this tree can A/B one. The missing piece is not code, it
   is an agreed observable: what counts as "the reply changed" when the consumer is a natural-language
   string. Until that is fixed at rubric level, any probe I write is choosing its own pass mark.
   `docs/coverage/artifacts/social_tom_ab_probe.py` should be generalised and moved to `tools/` by
   whoever owns that directory -- its arm-attribute neuter and self-failing controls apply to every
   category, not just this one.
2. **Make one social readout fire without a human typing a regex-shaped sentence.** Every faculty
   here is 0.0 on the autonomous tick. Until a social signal has a source other than `_TOM_PLACE`
   matching typed English, rule 4 keeps the whole category near its floor no matter how good the
   trackers are.
3. **Call `Pragmatics.act`.** A 6-class classifier and a brain-native learner for it are built,
   tested and reachable, and take one call site at `spike_chat_lake.py:997` to move from 15 to the
   band `implicature` already occupies. Smallest edit in this file's gap list.
4. **Exercise the depth-k axes.** `recursive_tom` supports N agents, depth 3 and `wants`/`knows`; the
   live path uses two agents, depth 2 and `believes`. Registering the agents a dialogue actually
   mentions would make the module's reason for existing live rather than self-checked.
5. **Joint attention from zero.** The two halves exist (`learned_attention`, the ToM trackers) and
   nothing binds them. This is a capability build, not a wiring wave, and it is the one row here that
   cannot be moved by any amount of plumbing.
6. **Fix the two content defects found while measuring**: the `_tom_protect` latch (note 2, caveat B)
   and the sad-maps-to-anger valence/arousal pair at `brain.py:3066-3069`.
7. **Record every A/B at three seeds, and put the seed in the artifact header** (added by the
   adversarial pass, and it is an instrument gap that outranks everything above except gap 1). One
   seed was the whole difference between band 61 and band 45 on the only row in this category that
   had a recorded A/B at all, and neither `coverage_score.py` nor the artifact format can see the
   problem: the seed appears nowhere in a recorded header, so two artifacts of the same faculty at
   different seeds are indistinguishable from a rerun. `social_tom_refute_probe.py` takes the seed as
   argv[3] specifically so the three runs are separate, comparable artifacts; the cheap version of
   this fix is a `seeds:` line written by `record` and a W-level warning when only one seed exists.

## Comparison to prior numbers, stated carefully

**30.0 is not comparable to the 38 that `COVERAGE_BASELINE.md:17` recorded for Social.** That
baseline is withdrawn as unreproducible and `docs/coverage/README.md:42-62` forbids the comparison;
its per-faculty table was never committed, so there is nothing to diff against. For what it is worth,
the mechanism most likely behind its high number is identified above: the birth-frozen
`theory_of_mind_false_belief` boolean, which is a rule-4 zero.

**Against the two-row version of this file it replaces (12.5), the increase is roster, not
capability.** That file had one scored row at 25 and one UNSCORED placeholder, and
`docs/coverage/README.md:39-41` says plainly that a category with one row is not a measured category.
Nothing in the brain moved between the two files. Of the changes to the one row they share,
`pragmatics_implicature` 25 -> 40 is an upgrade earned by a runtime call counter and a reply the
candidate won, replacing a static call-site correction.

**And the honest shape of the result is a downgrade twice over.** Exactly one faculty in this category
has a recorded, control-guarded, reproducible A/B, and after the adversarial pass **nothing in this
category reaches band 61**: the one row that did is 45, because the same wire at seed 1 changes no
action, no trajectory and no decision, and because the belief content it is named for turns out to
reach the arm as a single latched bit. Four more faculties have runtime evidence at band 41-60 that
the evidence standard scores at 40, since no instrument here can A/B a reply. Everything else is
telemetry, a method nothing calls, or absent.

**The single-sentence version: this brain models other minds in some detail, almost never acts on it,
and the one time it does act the effect is one flipped decision at one seed out of three.**

Rule 5 says downgrades are results, so, plainly: 31.8 -> 30.0, one band-41+ claim attacked and one
refuted, and no number in this file went up. The audited measurements were sound and the artifact was
real; what was missing was the second seed, the action column, and the question of whether a signal
called "belief-guided caution" carries any belief or produces any caution. It carries one bit and it
produces neither more holding nor a graded response.
