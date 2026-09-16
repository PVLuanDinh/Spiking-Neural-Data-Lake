# T082 -- social cognition composed: ONE recursive belief tracker on the dialogue path, a declared port with measured charge, and the other-agent prediction read against an equal-capacity world model

Row: tasks.md T082 [US4] ("Social cognition composed: `recursive_tom` as an L3 faculty with a declared port,
representation-level disclosure; in `cortex/recursive_tom.py` wiring"); FR-034 (composed onto the typed bus as
declared ports with charge verification); the C5 row `social_cognition` (`substrate/conclusion_protocol.py:161-168`:
licensed "predicting another agent improves when the system models that agent's belief rather than the world
alone"; refused "this licenses no claim that the system understands another mind, or that it has one";
dissociation input "a world-state model of equal capacity replacing the belief model must remove the gain");
PLAN-FINAL Part F row C-R11 (added with this file).

## 1. Territory, measured before the design (each fact with its source line, read at HEAD `de101bf`)

1. THREE TRACKERS OF ONE EVENT (the P2 debt this row settles). `cortex/brain.py` builds `TheoryOfMind` (:527-538,
   first-order WHERE, `_tom`, plus a birth-time Sally-Anne run whose result `_tom_ok` is published as
   `tom_false_belief` :4196 and again inside `braincog["theory_of_mind_false_belief"]` :549-565 -- a constant
   computed once at construction, True on every brain), `affect_core.SecondOrderToM` (:802-811, `_tom2`, WHERE-only
   second order) and `recursive_tom.RecursiveToM(n_agents=3)` (:1022-1042, `_recursive_tom`; the module's own
   docstring: it "generalises" both of the others "on three axes at once"). All three are fed by the SAME dialogue
   statement in `_tom_update(message)` (:4625-4690; called from `respond` :4274 on every non-command turn): the
   grammar `_TOM_PLACE` (:120-125, "<subj> put/moved the <obj> in/to the <loc>") plus the absence markers
   `_TOM_ABSENT` (:128-137, "while i was away", "behind my back", ...) decide WITNESSED (`_tom.place(lid, ["user",
   "brain"])`, `_tom2.observe`, `_recursive_tom.observe_event("user", "believes", loc, witnesses={"user", "brain"})`)
   or UNWITNESSED (`_tom.move(lid, ["brain"])`, `_tom2.observe(lid, ["brain"])`, `observe_event("brain", ...,
   witnesses={"brain"})`). The chat's belief report `_tom_report` (:4692-4720, routed at :4326 for "where do i
   think / believe" questions) reads `_tom` and `_tom2`; the reply annotation at :4537-4545 and the state key
   `recursive_tom_false_beliefs` (:4048) read `_recursive_tom`; `tom_live` (:4197, `_tom_live()`) reads `_tom`.
2. THE ONE CONSUMER THAT CHARGES: the arm's caution. On the wake tick (`_route_percept_to_action` :2222-2230) `fb =
   len(self._false_beliefs)` folds into `_tom_protect = max(min(1, 0.34 fb), 0.9 _tom_protect)` and is pushed to
   `self.brain.arm._social_risk`, which `AutonomousArm` adds to its caution term at autonomous_loop.py:1115 and
   :1197 (`caution = min(1, caution_bias + threat + intero + reliability_hold + _social_risk)`; the drive is scaled by
   `1 - caution`) -- "recursive_tom divergence -- until now a tracker that drove nothing". No manifest port declares
   this wire: `recursive_tom` is not in `manifest_gate.REGISTRANTS` (:46-106) and `tools/region_charge_evidence.json`
   carries no belief edge (grep 2026-09-16). FR-034's sibling, the executive, shows the house form: `task_set.
   declare_ports` (:260-292, the width DERIVED from `ACTIONS`), `_measure_film_charge` / `record_charge` (:347-387,
   `port_charge_probe.replay_twice` with a pinned payload and a constant-consumer specificity control,
   `record_attribution` into `tools/region_charge_evidence.json`) and `tests/test_task_film.py` (:30-70: the declared
   shape matches the live one, the committed manifest carries the wire, the committed charge evidence reproduces).
3. THE RECURSIVE TRACKER'S EVENT FORM IS SUBJECT-ASYMMETRIC. `RecursiveToM.observe_event(agent, attitude, content,
   witnesses)` (recursive_tom.py:123-144) moves the world unconditionally, updates the SUBJECT's own path and every
   chain of present witnesses ENDING IN the subject; a present witness's OWN first-order belief is not updated
   unless it is the subject. `divergence(o, t)` (:177-181) compares `predict_nested(o, t)` with `predict_first_order(t)`.
   `SecondOrderToM.observe(location, observers)` (affect_core.py:236-248) updates EVERY present agent's own belief
   and every ordered pair -- the symmetric form. `TheoryOfMind` is also consumed by `reasoning/pragmatics.py:469-471`
   (its own instance, the implicature engine); `SecondOrderToM` has no consumer but brain.py:807; `RecursiveToM` has
   `tests/test_recursive_tom.py` (8 tests, the module alone) and run_selfchecks.py:132.
4. THE APPARATUS EXISTS AND THE LIVE FEED HAS A DEFECT. Measured 2026-09-16 04:38:55 local (`scratchpad/t082/
   probe_tom.py`, seed 0, scratch home, `persist=False`, 12 scripted statements through `Brain.respond` -- "i put the
   ring in the <loc>." witnessed, "someone moved the ring to the <loc> while i was away." unwitnessed, p = 0.5; the
   SCRIPT keeps the agent's own belief = its last witnessed location and the world = the last location): 12/12
   statements parsed (`_tom_update` 1.0-2.2 ms; `respond` 112-121 ms per turn after a 3.08 s first turn); 4
   unwitnessed episodes; the recursive tracker's first-order prediction of the user matched the agent's look on
   12/12, the world content on 9/12, and an EQUAL-CAPACITY WORLD CONTROL (a second `RecursiveToM(n_agents=3, seed=0)`
   fed the same statements with every agent witnessing) on 9/12 -- exactly the world's accuracy. `_false_beliefs`
   read `['user']` on the unwitnessed episodes and then `['user->brain']` on EVERY later witnessed episode (5 of 12):
   the witnessed event's subject is the user, so the brain's own path stays at its last unwitnessed location while
   the user's model of the brain was never written, and `divergence("user", "brain")` reads True from then on -- a
   spurious, persistent nested false belief born of fact 3's asymmetry. Meanwhile `_tom_report()` (reading `_tom`)
   said "no divergence right now" and `_tom2.predict_second_order("user", "brain")` read location id 1 while the
   recursive `predict_nested("user", "brain")` read None: three trackers, three answers on one dialogue. The arm's
   `_social_risk` stayed 0.0 because the fold runs on the wake tick and the probe never ticked between statements.
5. WHAT NO GATE SEES TODAY. `CONCLUSION["social_cognition"]` has `reading` None and `surface_symbols` (); no tracked
   surface prints `recursive_tom_false_beliefs`, `tom_live` or `tom_false_belief` (grep of cockpit.py, OVERVIEW,
   PITCH, README 2026-09-16); `tests/test_ratified_row_gates.py`'s arming fixture reaches `social_cognition` at T081's
   surfaces commit (this is the last C4 row: the fixture then arms nothing and the test's arming arm is rewritten to
   a constructed row). `node graph-exists.js "recursive theory of mind declared port with charge; other-agent
   prediction against an equal-capacity world model"`: no existing implementation (2026-09-16).

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT -- the account is under a WEEKLY usage cap until 2026-09-20 00:30

Pre-registered without the read, as T079-T081 were; the closing report is owed when the cap lifts.

### 2.1 The mechanism -- ONE tracker, ONE symmetric event, ONE declared wire with measured charge

THE MERGE (P2; built before any number): `Brain._tom_update` feeds `RecursiveToM` ONLY, through a SYMMETRIC event:
a statement with present set P (`{"user", "brain"}` witnessed; `{"brain"}` unwitnessed) calls
`observe_event(a, "believes", loc, witnesses=P)` once for EACH a in P, so every present agent's own path and every
chain among the present update, and an absent agent's paths stay stale (the false-belief substrate); the content is
the location NAME (no id map). `_tom.place/move` and `_tom2.observe` calls are DELETED from `_tom_update`;
`_tom_report` reads the recursive tracker (first order `predict_first_order("user")` vs `world_content()`; the
second-order line from `predict_nested("user", "brain")`); `_tom_live()` reads it; the birth-time Sally-Anne
run, `_tom_ok`, `tom_false_belief` and `braincog["theory_of_mind_false_belief"]` are DELETED (a constant computed at
construction is a marker, F11); `affect_core.SecondOrderToM` is DELETED after R5 shows `predict_nested` reproduces
`predict_second_order` on constructed witness scripts; `TheoryOfMind` STAYS (pragmatics.py's engine) and its live
instance `_tom` / `_tom_locs` / `_tom_obj` go with the feed (`_tom_obj` becomes the recursive tracker's last object).
THE WIRE (FR-034): `recursive_tom.declare_ports(manifest)` -- OUT `belief_out`, region `cortex`, shape (2,) f4 =
[first-order false-belief count, nested-divergence count] over the tracked agents, unit `count`, encoding
`VEC_DENSE`, delivery_policy ("graded", "divergence"); the consumer IN port `brain.social_risk_in` (join_policy
`sum`) and the edge `recursive_tom.belief_out -> brain.social_risk_in` (`BELIEF_EDGE`), registered in
`manifest_gate.REGISTRANTS` before `brain`, `tools/wiring_manifest.nir` regenerated in the same commit. THE CHARGE
(T060's form): `_measure_belief_charge()` = `port_charge_probe.replay_twice` over RECORDED belief frames (the dry
run's per-tick [fb_first, fb_nested] vectors from a scripted stream with false beliefs), the consumer = the brain's
own fold (`_tom_protect` recurrence -> the arm's caution term at autonomous_loop.py:1197, reduced to the caution
trajectory), the pinned payload [0, 0], the specificity control a constant consumer; `record_charge()` REFUSES a
non-diverged A/B and otherwise writes the attribution into `tools/region_charge_evidence.json`; the self-check and
`tests/test_recursive_tom.py` re-measure and require the committed numbers to reproduce. THE INSTRUMENT (C4; the
`t082_*` functions in `cortex/recursive_tom.py`): `t082_script(seed, n_ep)` -- the scripted other agent (objects
and locations drawn under `random.Random(4000 + seed)`, p_unwitnessed = 0.5, the agent's look = its last witnessed
location, the world = the last location); `t082_record_brain(seed, n_ep)` -- a scratch brain under `throwaway_home()`
(`make_brain(seed, persist=False)`), both stream pointers offset `150 * seed`, per episode: the statement through
`Brain.respond` (the live dialogue path), then `T082_TICKS_BETWEEN` = 3 ticks (so the fold runs), then the brain's
predictions of the agent -- BELIEF `predict_first_order("user")`, WORLD `world_content()` -- and the CONTROL: an
equal-capacity `RecursiveToM(n_agents=3, seed)` fed the same statements with every present set replaced by the full
set (the absence markers ignored -- "a world-state model of equal capacity replacing the belief model"); rows
carry the parse event, the two lists of divergences, the arm's `_social_risk` after the ticks, the timings.
`t082_read(rows)`: per seed acc_belief, acc_world, acc_ctrl against the agent's look, the unwitnessed fraction, the
count of episodes whose agent look differs from the world, the parse count, the nested-divergence count on
witnessed-only stretches (fact 4's spurious rate, expected 0 after the merge), the social-risk trajectory.
`t082_contrasts(per_seed)`: 2.3's conjuncts through `substrate/paired_contrast.contrast`.

### 2.2 The arms (one command, seeds 0-7, a scratch brain; the daemon's brain is never touched)

Per seed one scripted stream of N = `T082_N_EPISODES` = 60 statements; every episode carries the three predictors
(BELIEF, WORLD, CONTROL) on the same statement, paired by construction. The barred X is the accuracy of a
predictor against the agent's look over the episodes; the barred quantity is the composed tracker's first-order
prediction of the other agent -- the disposition noun (2.6) and this X are the SAME object. Every recording also
carries the charge frames (per tick [fb_first, fb_nested]) for 2.1's A/B.

### 2.3 The statistic and the verdict rule (written first)

The plan's seed-paired statistic from `substrate/paired_contrast`: over the 8 seeds a one-sided paired t against
`T_BAR` = 1.895 with the exact sign-flip FPR beside it; a zero-variance sample is MET iff its common value is > 0.
No majority-of-seeds rule. Conjuncts, each MET or NOT MET and printed:

(p)  DETERMINISM: two recordings of seed 0 at 20 episodes are byte-identical on every recorded field, else REFUSED.
(r)  READABLE: on every seed every statement parses (`_tom_update` returns an event string) and every episode's
     three predictions are non-None after the first witnessed statement; else REFUSED (the grammar or the
     apparatus, named per seed).
(g0) THE SCRIPT IS A FALSE-BELIEVING AGENT (the positive control that is not the method): on every seed at least
     30 % of the episodes are unwitnessed and on EVERY unwitnessed episode the agent's look differs from the world;
     else REFUSED ("the stream carries no false belief to track", never a STOP).
(s1) THE BELIEF MODEL BEATS THE EQUAL-CAPACITY WORLD MODEL (the licensed sentence): t over seeds of
     (acc_belief - acc_ctrl) > 1.895.
(s2) THE CONTROL IS A WORLD MODEL (the dissociation input): on every seed the control's prediction equals the world
     content on EVERY episode, so acc_ctrl == acc_world exactly -- the gain is removed when the belief model is
     replaced; a count, exact.
VERDICT: PASS iff (s1) and (s2) with (p), (r), (g0) holding; NOT MET iff (s1) or (s2) fails; REFUSED iff (p), (r) or
(g0) fail. REPORTED, never barred: the charge A/B (the live vs pinned caution trajectories, the reads, the
specificity control), the spurious nested-divergence count under the merged feed (fact 4's 5/12 on the old feed is
quoted beside it), the social-risk trajectory per seed, `respond` ms per turn, the parse grammar's two accepted
forms (the surface parse is hand-specified -- disclosed), the second-order line's readings.

### 2.4 The reds (RUN RED before the mechanism; in-command with their controls thereafter; each reddened by a mutated scorer)

R1  THE PARSE REFUSES OUTSIDE THE GRAMMAR: "the weather is nice today." returns None and (r) reads REFUSED on a
    constructed stream containing it; the two script forms parse. Mutation `parse_anything` (a non-match returns an
    event): red.
R2  THE CONTROL IS A WORLD MODEL: on a constructed 40-event witness script (50 % absences) the control's first-order
    prediction equals the world on 40/40. Mutation `control_keeps_witnesses` (the control honours absence): differs
    on the absent events: red.
R3  THE DISSOCIATION ON A CONSTRUCTED SCRIPT: eight constructed mixed scripts read (s1) MET and (s2) MET; eight
    all-witnessed scripts read (s1) NOT MET (acc_belief == acc_ctrl, a degenerate zero). Mutation `blind_scorer`
    (belief and control swapped): red.
R4  THE CHARGE A/B: recorded belief frames carrying false beliefs drive the live fold to a caution trajectory that
    diverges from the pinned-[0, 0] one; the constant consumer does not flip. Mutation `consumer_ignores_payload`
    (the fold reads 0 in place of the payload): no divergence -> `record_charge` REFUSES: red.
R5  THE MERGE IS LOSSLESS: on 200 random witness scripts (3 agents, 4 locations, 12 events each) the symmetric feed
    into `RecursiveToM` reads `predict_nested(x, y)` equal to `SecondOrderToM.predict_second_order(x, y)` after
    `observe(loc, observers)` for every ordered pair, and `predict_first_order` equal to `TheoryOfMind.
    predict_from_belief` for every agent. Mutation `nested_reads_world`: red.
R6  THE SYMMETRIC FEED KILLS THE SPURIOUS DIVERGENCE (fact 4 reconstructed): after an unwitnessed then a witnessed
    statement, `divergence("user", "brain")` is False and `predict_first_order("brain")` equals the world.
    Mutation `feed_subject_only` (the old subject-only call): the divergence reads True: red.
Each red runs on its constructed input and FAILS its assertion at commit 3 of 2.9; at commit 4 each is green and its
mutation is applied IN THE TEST and in `t082_reds()` in the module's self-check line (six booleans asserted).

### 2.5 The controls that are not the method

The equal-capacity world model (the same class, the absence gating removed -- the C5 row's own dissociation
input); the WORLD predictor (reality, no mind); the all-witnessed script (R3); the constant consumer in the charge
A/B (specificity); the old subject-only feed (R6, the reconstructed defect); the determinism twin (p).

### 2.6 Disposition (fixed in advance; the honesty lens adjudicates any reading of these words)

- PASS: on this composite the composed tracker's prediction of a scripted other agent beats an equal-capacity model
  of the world alone at the bar over 8 seeds, and the replacement removes the gain exactly. CHARGED:
  `conclusion_protocol.CONCLUSION["social_cognition"]` gets `reading` = the mean over seeds of (acc_belief -
  acc_ctrl) (an accuracy difference in [0, 1], the licensed sentence's "improves") and `surface_symbols` =
  ("recursive_tom_false_beliefs",) -- the LIVE state key the tracker already publishes; the cockpit prints it as a
  consciousness row WITH the licensed and refused sentences in the same view, the phrase "representation-level" and
  a PROVENANCE line naming the pin, the scripted agent and the charge evidence. tasks.md T082 -> [X] PASS.
- NOT MET: the belief model does not beat the equal-capacity world model at the bar, or the replacement does not
  remove the gain exactly. NO reading, nothing printed; tasks.md T082 -> [X] NOT MET with the symbol
  `social_belief_gain` named and printed nowhere (T077's shape).
- REFUSED: the fault named ((p), (r) or (g0)); the row stays [ ]; no surface changes.
- THE MERGE, THE WIRE AND THE CHARGE EVIDENCE land in the build (2.9 steps 4-5) on every branch: they are P2's and
  FR-034's deliverables, not the verdict's -- as T079's feed switch was.
In every branch `tests/test_ratified_row_gates.py`'s arming arm is rewritten to a constructed row (no symbol-less C4
row remains). The noun in every branch is "the composed tracker's first-order prediction of the other agent" -- the
barred X of 2.2.

### 2.7 What this row does NOT claim

No understanding of another mind, no mind (the refused sentence); the other agent is a SCRIPT whose look is its last
witnessed location -- the row measures that the composed faculty tracks witness-gated beliefs through the live
dialogue path and that its output charges the arm's caution, nothing about people; the parse is hand-specified (two
sentence forms; a statement outside them is refused, never guessed); nothing about `TheoryOfMind` as pragmatics uses
it; nothing about the LM narrator's reply text (display-only; the reply annotation is read, not scored); no
aggregate consciousness score; "representation-level" throughout.

### 2.8 Widths, cost, host, declared constants and their sensitivity

Seeds 0-7; per seed 60 episodes x (`respond` ~120 ms + 3 ticks x ~55 ms) ~17 s, 8 seeds ~2.5 min sequential in one
process; the charge A/B under a second; seed 0's twin. Refusal: above 30 min the campaign is re-sized and this
section amended. DECLARED CONSTANTS and their consequence, COMPUTED in 3.0 (the dry run): p_unwitnessed 0.5 (the
unwitnessed fraction per seed reported), `T082_TICKS_BETWEEN` = 3 (the fold's 0.9 decay per tick makes the caution
trajectory readable within the episode; the trajectory at 1 and 5 ticks reported on the dry run), the objects (5)
and locations (4), N = 60, the offset 150 x seed, the charge fold constants 0.34 / 0.9 (the brain's own, read not
chosen). Host: Python 3.14.6 free-threaded; the brain's own torch encoders on the RTX 5070 where present; no GeNN.

### 2.9 Commit order (one change each; the stubs before the reds so the reds RUN red; surfaces before the record)

1. This file + PLAN-FINAL Part F row C-R11 (T082) -> 2. the skeleton: `t082_*` stubs and constants in
`cortex/recursive_tom.py`, `declare_ports` + `BELIEF_EDGE` + `_measure_belief_charge` / `record_charge` stubs,
`brain.declare_ports` gains the IN port `social_risk_in`, `manifest_gate.REGISTRANTS` gains the registrant,
`tools/wiring_manifest.nir` regenerated; `main()` printing the six booleans; census gates -> 3.
`tests/test_recursive_tom.py` gains R1-R6 committed RED (each runs on its constructed input and fails; output
pasted) -> 4. the instruments green, each red reddened by its mutated scorer; THE MERGE in `brain.py` (the symmetric
feed; the `_tom` / `_tom2` feeds, the birth-time run, `_tom_ok`, `tom_false_belief`, the braincog flag deleted;
`_tom_report` and `_tom_live` re-pointed; `SecondOrderToM` deleted from `affect_core.py`); census gates -> 5. the
recorder + the dry run on ONE seed at the campaign width (the charge A/B measured and `record_charge` run -- the
evidence committed HERE, refused if the A/B does not diverge; the spurious-divergence count; the caution
trajectory at 1 / 3 / 5 ticks; the cost; the offset check) + `--t082-campaign` + the pin writer
(`tools/t082_social_cognition.json`, once); NO barred conjunct read there -> 6. the campaign, the pin (once) -> 7.
surfaces and annotations per 2.6 (tasks.md's dated note; PLAN-FINAL C4 note; spec.md SC-009 and FR-034 notes; the
row-gate fixture's rewrite; on PASS the conclusion row and the cockpit row) -> 8. section 3 the record -> panel
(3 lenses + refuters judging CLAIM and FIX) -> fold -> stacked with T080 / T081 for ONE battery (the Codex closing
report when the cap lifts, 2.0).
