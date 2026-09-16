# T079 -- the agency instrument: the efference-copy mismatch red on the live body (specs/022 Phase 4 / C4; plan row :164; register #6's agency item; SC-009; the C5 row `agency`)

Sections 1-2 are the PRE-REGISTRATION, written 2026-09-15 (21:05 local onward; git's clock is the record of when they
froze) BEFORE a line of this row's code exists in the tree and frozen at the commit that lands them; 2.0 records why no
second model read the design; amendments are dated appended sections; section 3 is written after the runs. Every number
in sections 1-2 is a source line, a plan constant, or an apparatus probe named with its script and the numbers it
printed (no gate statistic has been read on any brain -- the two probes, `scratchpad/t079/probe_apparatus.py` and
`scratchpad/t079/probe_streams.py`, printed marginals, a per-joint Pearson over ONE seed's 146 passages and counts,
quoted in 1.3, and no controllability reading of the instrument, no floor, no contrast).

## 1. Territory, measured before the design (each fact with its source line, read at HEAD `82d60b6`)

1. THE INCUMBENT IS WRITE-ONLY AND FED THE WRONG QUANTITY. `cortex/agency_monitor.AgencyMonitor` (per-channel
   streaming |Pearson| of command vs feedback over EMAs, alpha 0.02; a two-counter hysteresis machine: 'lost' after
   `delay` = 48 consecutive readings below `dead_thresh` = 0.35, revived only after `recover_n` = 32 consecutive
   readings at or above `high_thresh` = 0.55; `warmup` = 120 observations DERIVED as 2.4 EMA memories; `var_floor`
   1e-4 leaves an unexercised channel UNSCORED) is constructed by `brainstem/autonomous_loop.AutonomousArm.step`
   when `agency_monitor=True` (`cortex/brain.py:261` sets it) and fed every awake tick at
   `autonomous_loop.py:2383-2386` with `observe(action, cmd=qmax, fb=reward)` -- the channel is the ACTION LABEL,
   the "command" is the peak of the value row and the "feedback" is this tick's reward. That is not an efference
   copy. The module's own LIVE STATUS (measured 2026-07-28, its docstring) records the consequence: over 600 live
   ticks one channel of seven crossed warmup, `agency_signal` took two values, wiring `route_around` into action
   selection was REJECTED on that measurement, and the disposition was "exempt-awaiting-sufficient-per-channel-
   experience". The site's own comment (:2378-2382) says "WRITE-ONLY as of today ... nothing in src/ or tests/
   reads agency() / controllability() / status()". Re-measured 1.3: on the make_brain composition the feed scores 0
   channels after 146 awake ticks (`n_scored` 0, `agency_signal` 0.0). Registered: `run_selfchecks.py:121`,
   `__init__.py:180` REGIONS; reachable on the live tick so not in `dormancy_gate.EXEMPT`; NO test in `tests/`
   references it (grep, 2026-09-15).
2. THE EFFERENCE COPY EXISTS, AS ONE SCALAR. `substrate/efferent_boundary.EfferentBoundary.observe(action,
   theta_pre, commanded_after, theta_post)` (:85-117) is called on EVERY envelope passage from
   `AutonomousArm.apply_observed` (:1818-1845; all four of step()'s apply sites and the cockpit's route; a tap
   exception is swallowed and COUNTED in `tap_errors`). It forms `u = proprioception_delta(theta_pre,
   commanded_after)` and `ds = proprioception_delta(theta_pre, theta_post)` -- `substrate/afferent_boundary.
   proprioception_delta` (:156-181) wraps EACH joint's delta to [-180, 180) then AVERAGES over the shared joints to
   ONE signed scalar, "acceptable for a one-scalar copy because the live loop commands one joint per tick" -- and
   trains the incumbent `cerebellum/forward_model.ForwardModel` (1-D LMS). `last_copy` carries `{action, s, next,
   u, err, envelope_tick}`: no per-joint vector survives the fold. Its consumer disposition names THIS row: "The
   spec's consumers are T069 ... and T079 (the agency instrument's efference-copy mismatch red, US4); this lane is
   their declared substrate" (:11-17). The T037 embodiment contract `tests/test_efferent_boundary.py` pins that the
   tap is observation-only (neuter `observe` -> identical actions and trajectory) and that step and envelope tick
   move 1:1.
3. APPARATUS FACTS, measured 2026-09-15 20:58-21:02 (`scratchpad/t079/probe_streams.py`, seed 0, 300 ticks,
   scratch home, `persist=False`; the tap's triple captured by wrapping the INSTANCE's `efferent.observe`; the
   command per joint parsed from the ACTION TOKEN -- `JOINT_<NAME>_ROTATE(<+/-N>deg)` gives that joint +/-N and
   every other joint 0; the achieved delta per joint from theta_post - theta_pre wrapped): 146 awake ticks, ONE
   passage per awake tick (146/146), 61.5 ms per brain-tick, `tap_errors` 0, never latched. Actions: ENGAGE 48, six
   joint rotations 9-14 each, HOLD 6, RELEASE 3 (13 distinct). Per ROTATABLE joint (`arm_config.ROTATABLE`, five:
   shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll; the gripper is ENGAGE/RELEASE and excluded):
   commanded on 14 / 27 / 15 / 17 / 16 of 146 passages; the achieved delta on a commanded passage averages 5.58 /
   6.00 / 6.00 / 5.81 / 6.00 deg (the full `DEFAULT_STEP_DEG` = 6.0 step lands WITHIN THE SAME TICK) and 0.045 /
   0.001 / 0.001 / 0.026 / 0.000 deg on an uncommanded one; same-tick Pearson(u_j, ds_j) +0.945 / +1.000 / +1.000 /
   +0.983 / +1.000, lag-1 -0.08 / +0.15 / +0.05 / +0.14 / -0.00 -- the coupling is same-tick and near-perfect on
   this servo body, so a 'lost' channel on an intact body is not a live risk and the instrument's warmup (120
   observations) is reached by passage 120 when every joint is observed on every passage. THE SANCTIONED
   EXAFFERENCE PORT: `brainstem/so101_arm.So101Arm.inject_disturbance(joint, deg, *, allow_latched=False)`
   (:319-352; "TEST-ONLY: shove a joint without commanding it, to stand in for an external force ... writes qpos
   directly, leaves `_commanded` untouched ... runs mj_forward rather than mj_step ... Nothing on the live tick calls
   this"; latch-aware since 2026-08-03). Shoving ONE joint by U(-A, A) deg after every passage (the servo then
   corrects on the next apply, so the joint's achieved delta carries a component the command did not cause):
   elbow_flex at A = 3 / 6 / 12 deg reads same-tick Pearson +0.818 / +0.344 / +0.173 (intact +1.000);
   shoulder_lift at A = 12 reads +0.184; every UNSHOVED joint stays at +0.945..+1.000 in every run; the end-effector
   deviation of a 12-deg elbow shove (lever 180 mm) is ~0.04 m against `TRAJ_DEV_MAX` = 0.15 m and no run latched.
   DISCLOSED, not adjudicated: the loop's POLICY reacts to the disturbance -- the shoved joint's own rotate token is
   chosen 74-95 of 146 passages (elbow) or another joint's 99 of 146 (wrist_roll under a shoulder_lift shove) --
   so "motor control preserved elsewhere" is read on the COUPLING of the other joints, never on the action
   distribution, which is reported. `scratchpad/t079/probe_apparatus.py` also measured the bare `AutonomousArm`
   under random sensors: ENGAGE on 300/300 ticks, no joint ever commanded, 2.3 ms/tick -- the recording MUST run on
   the make_brain composition (the T078 apparatus), whose percept stream drives the rotations.
4. THE C5 ROW IS WRITTEN AND ITS NOUNS BIND THIS DESIGN. `substrate/conclusion_protocol.CONCLUSION["agency"]`
   (:134-141): licensed "the system separates outcomes it caused from outcomes it observed, above a cause-blind
   twin's rate"; refused "this licenses no claim of felt authorship, ownership of action, or free will";
   dissociation input "identical outcomes with the causal link removed must collapse the separation"; reading
   None; `C4_TASK["agency"] = "T079"`; surface symbols () with the comment "T079 unbuilt". The arming arm
   (`row_findings`): a row whose task is marked [X] in tasks.md while declaring no surface symbol is a red
   (`tests/test_ratified_row_gates.py::test_refusal_three_arms_itself_when_a_c4_instrument_task_lands`). So on a
   PASS this row DECLARES its symbol and writes its reading; on NOT MET it stays [ ] or is marked [X] with the
   symbol named and printed nowhere (T077's shape) -- 2.6 fixes which.
5. THE PLAN ROW. tasks.md :164: "T079 [P] [US4] Agency instrument: efference-copy mismatch red (perturb one
   effector channel -> agency collapses for that channel, motor control preserved elsewhere);
   `cortex/agency_monitor.py` becomes consumer-backed (write-only today, measured)". PLAN-FINAL :914 "T079
   (agency; needs O1)" -- O1 (the E-STOP cleared at the cockpit) was executed 2026-09-02 (:1131) and is not
   needed by this row anyway: every recording is a scratch composition. spec.md FR-028 names "agency
   (efference-copy mismatch)"; `docs/INVENTION_REGISTER.md` 6 lists it under the consciousness dissociation
   battery with RED / CONTROLS / DISCLOSURE / NULL clauses ("an instrument whose red never fires is not an
   instrument and is deleted").
6. THE APPARATUS THIS ROW REUSES (P2; no second implementation): the paired one-sided t and its bar from
   `substrate/paired_contrast` (`contrast`, `T_BAR = 1.895`, the exact sign-flip FPR; module-level import); the
   recording pattern of `attention_schema.t078_record_brain` / `metacognitive_monitor.t077_record_brain` (a scratch
   brain under `throwaway_home()`, `make_brain(seed, persist=False)`, per-tick rows, an INSTANCE's method wrapped to
   capture what it is fed); the mutated-scorer reachability pattern of `_t078_mutated_scorer` / `t078_reds`; the
   order statistic `_q95` (its third spelling; T078 3.5 item 23 -- this row IMPORTS `attention_schema._q95` rather
   than spelling a fourth, and the lift into `_mathutils` stays the successor's); the incumbent `AgencyMonitor`
   whole (its EMAs, thresholds and state machine are the instrument -- nothing is re-implemented); the token
   vocabulary `arm_config.token_for` / `command_for` / `ROTATABLE` / `JOINTS` (envelope file: READ, never edited).
   P2 TARGET: `brainstem/live_arm._joint_obs_delta` (:60) parses ONE joint's rotate token by a private regex; the
   per-joint intended delta this row needs is the same parse over every joint, so ONE parser lands in the boundary
   and `live_arm` calls it (one mechanism). NOT in the tree (graph-exists, 2026-09-15): a per-joint efference copy
   (commanded vs achieved delta per joint); a controllability reading against a shuffle floor; the in-vivo
   one-channel dissociation. Those are this row's new symbols, in `cortex/agency_monitor.py` and
   `substrate/efferent_boundary.py`.
7. WHAT T077's AND T078's FOLDS BIND ON THIS DESIGN (evidence/t077-metacognition.md 3.5 items 9-27, 3.7-3.8;
   evidence/t078-attention-schema.md 3.5 items 8-27, 3.7): reds reddened in fact by mutated scorers; no positive
   control that is an identity of the statistic; every per-tick input of every barred conjunct in the pin; every
   declared constant's sensitivity COMPUTED in 3.0, in the dry run, never promised; the DETECTION FLOOR of the
   contrast pre-registered as a plant ladder run BEFORE the campaign and reported; a NOT MET's null half read at
   two budgets where a budget exists; |X| and every readability precondition's realized margin printed; the
   disposition noun and the barred X pre-registered as the SAME object; the stimulus stream VARIED per seed;
   surfaces and annotations land BEFORE the record commit; no scratch citation without its command and numbers;
   cost re-measured on the built apparatus; a waiter matches every terminal state.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT -- the account is under a WEEKLY usage cap until 2026-09-20 00:30

T078's closing report was refused at 20:49:26 with "try again at Sep 20th, 2026 12:30 AM" (T078 3.5 item 27). No
second model can read this design before it freezes. Per the 2.0 precedent of T078 (its 2.10), the row proceeds; if
a read lands after this point it is folded as a dated 3.5 item (post-number: it may name defects, it may not move a
bar), never as an amendment. The T079 closing report is owed when the cap lifts, disclosed in the record if the push
precedes it.

### 2.1 The mechanism -- ONE instrument, ONE feed, the dissociation read in vivo

THE FEED (the P2 replacement; unconditional, built before any number): the per-joint efference copy. On every
envelope passage `apply_observed` already holds the tap's triple (theta_pre, commanded_after, theta_post) and the
action token. `substrate/efferent_boundary` gains `proprioception_deltas(theta_before, theta_after) -> list` (the
per-joint wrapped deltas that `proprioception_delta` then averages -- the scalar is redefined as the mean of this
list, one fold) and `intended_deltas(action) -> dict joint -> deg` (the token parser over `arm_config.ROTATABLE`:
`JOINT_<NAME>_ROTATE(<+/-N>deg)` -> +/-N on that joint, 0 elsewhere; MOVETO, ENGAGE/RELEASE/HOLD/HOME/EMERGENCY_STOP
and any non-rotatable name -> all 0; `live_arm._joint_obs_delta` becomes a one-line call into it). `last_copy`
gains `u_j` and `ds_j` (the two per-joint lists) beside the scalars. `AutonomousArm.apply_observed` feeds, INSIDE the
same guarded block as the tap and after it, `self._agency.observe(joint, u_j[joint], ds_j[joint])` for every
rotatable joint on every passage (so every joint is observed on every passage and the warmup counts passages), and
the tick-level feed at :2383-2386 (`observe(action, cmd=qmax, fb=reward)`) is DELETED -- exactly one feed survives
(P2), the monitor's lazy construction moves with it, `_agency_on` keeps gating it. T037's neuter contract stays
green by construction: the feed reads the triple and the token, drives nothing.

THE READER (`cortex/agency_monitor.py`, the row's named file): `t079_record_brain(seed, n_ticks, shove=None,
amp=0.0)` -- a scratch brain under `throwaway_home()` (`make_brain(seed, persist=False)`, the stream offset of 2.2),
`efferent.observe` wrapped on the INSTANCE to capture the triple and the action per passage; per passage a row
`{t, awake, action, u (5), ds (5), ctrl (5, `controllability(j)` read after the feed), status (5), n (5)}` plus the
tick's `kind`; when `shove` is a joint name, `So101Arm.inject_disturbance(shove, U(-amp, amp))` is called ONCE after
every passage's feed with a per-seed `random.Random(7000 + seed)` (the loop's own RNG untouched), never while
latched. `t079_read(rows)` -> per joint: `c_j` = the MEAN of `ctrl_j` over the passages with `n_j >= warmup` (the
instrument's own streaming reading, averaged over its scored window; NOT the last value), `status_j` at the end,
`scored_j` = (`n_j >= warmup` and the command variance over the scored window >= `var_floor`), the passage count and
the latch count. `t079_floor(rows, seed, n_shuffle)` -> per joint the CAUSE-BLIND TWIN: the same feed with `u_j`
PERMUTED across the scored passages (the outcomes identical, the causal link removed -- the C5 row's own words),
`n_shuffle` = 2,000 draws through the SAME EMA machine batched as a (2,000 x passages) torch recurrence on CPU
(GIL off, one process; CUDA where present), `floor_j` = `_q95` of the 2,000 mean-controllability draws and
`floor_median_j` beside it. `t079_contrasts(intact, shoved)` -> 2.3's conjuncts.

### 2.2 The arms and the pairing (one command, seeds 0-7, a scratch brain; the daemon's brain is never touched)

Per seed s in 0..7, TWO recordings of N = 1,200 ticks on the same seed and the same stream offset, run sequentially in
one process (throwaway_home is process-wide): INTACT (no shove) and SHOVED (J* = `elbow_flex`, the mid-chain joint,
A = 2 x `DEFAULT_STEP_DEG` = 12 deg -- the disturbance equals twice the command scale, so on this body the coupling
of J* is expected to fall to ~0.17 (1.3) while a 6-deg shove sits AT the loss threshold (0.344 vs 0.35) and a 3-deg
shove leaves it above (0.818); the amplitude is derived from the step, not chosen). THE STREAM IS VARIED PER SEED
(T078 opening 9): after construction `brain._real_ptr = 150 * s` (the pack is walked modulo its length by
`_perceive_real`), and the dry run checks the first 50 frame labels differ between seeds 0 and 1; if they do not the
campaign is REFUSED at the apparatus and this clause is amended, dated. The barred X is the per-joint pair
(u_j, ds_j) as 2.1 defines it, for the five rotatable joints; the barred quantity is the instrument's own
`controllability` -- the disposition noun (2.6) and this X are the SAME object. Every recording also carries: the
action per passage, the tick kind, `tap_errors`, `stopped`, `latch_source`, the passages-per-tick count.

### 2.3 The statistic and the verdict rule (written first)

The plan's seed-paired statistic from `substrate/paired_contrast`: over the 8 seeds a one-sided paired t against
`T_BAR` = 1.895 with the exact sign-flip FPR beside it; a zero-variance sample is MET iff its common value is > 0.
No majority-of-seeds rule. Conjuncts, each MET or NOT MET and printed:

(p)  DETERMINISM: two INTACT recordings of seed 0 at 100 ticks are byte-identical on every recorded field, else REFUSED.
(r)  READABLE: on every seed and BOTH arms every rotatable joint is `scored` (n >= warmup = 120 passages; command
     variance >= var_floor over the scored window), the passage count equals the awake-tick count, `tap_errors` = 0
     and no recording latched (`stopped` never True), else REFUSED (the instrument's or the apparatus's own fault,
     named per seed and arm).
(g0) THE ESTIMATOR SEES (the positive control that is not the method): for EVERY joint j, t over seeds of
     (c_j^intact - floor_j^intact) > 1.895 -- the intact coupling clears its cause-blind twin's rate; any joint NOT
     MET -> REFUSED ("blind on this body", never a STOP).
(a1) COLLAPSE ON THE PERTURBED CHANNEL: t over seeds of (c_J*^intact - c_J*^shoved) > 1.895.
(a2) PRESERVED ELSEWHERE: for EACH k != J*: t over seeds of ((c_k^shoved - c_k^intact) + DEADBAND) > 1.895, where
     DEADBAND = `high_thresh - dead_thresh` = 0.20 is the instrument's OWN dead-band (a drop smaller than the band
     the machine itself treats as noise); all four MET, else NOT MET.
(a3) THE STATUS MACHINE AGREES: `status_J*^shoved` == 'lost' on 8 of 8 seeds AND no other (joint, arm, seed) is
     'lost' -- counts, exact.
VERDICT: PASS iff (g0) and (a1) and (a2) and (a3); NOT MET iff (g0) MET and any of (a1)-(a3) NOT MET; REFUSED iff
(p), (r) or (g0) fail. REPORTED, never barred: the per-joint floors and their medians; the shove ladder of 2.9 step 5
(A = 3 / 6 / 12 on one seed, the dry run: the detection floor of (a1) and of the status machine, printed BEFORE the
campaign); the action distribution per arm (the policy's reaction to the shove); the same-tick vs lag-1 Pearson per
joint; the incumbent feed's `n_scored` on the same ticks (0 expected); the Monte-Carlo sd of `floor_j` over 12
independent 2,000-draw floors on the dry-run recording (2.8's n_shuffle sensitivity, computed in 3.0).

### 2.4 The reds (RUN RED before the mechanism; in-command with their controls thereafter; each reddened by a mutated scorer)

R1  THE FLOOR IS A FLOOR: a constructed coupled channel (fb = 0.9 u + noise, 600 passages) reads c above its
    shuffle floor; a decoupled channel (fb independent of u) reads c at or below it. Mutation `floor_unpermuted`
    (the floor drawn from the UNshuffled sequence) -> the decoupled channel reads "above": red.
R2  THE MACHINE'S HYSTERESIS: on a constructed healthy -> decoupled -> recovered stream (the module's own
    `_experiment`, reused), a single high reading while 'lost' does NOT flip to 'ok' and a single low reading while
    'ok' does NOT flip to 'lost'. Mutation `no_hysteresis` (`delay` = `recover_n` = 0) flips on one reading: red.
R3  AN UNEXERCISED CHANNEL IS UNSCORED, NEVER LOST: u == 0 on every passage -> `scored` False, status never 'lost',
    (r) refuses. Mutation `no_floor` (`var_floor` = 0) scores it and (r) reads: red.
R4  THE PARSER: `intended_deltas("JOINT_ELBOW_FLEX_ROTATE(+6deg)")` is +6 on elbow_flex and 0 on the other four;
    "(-6deg)" is -6; ENGAGE / RELEASE / HOLD / HOME / EMERGENCY_STOP -> all 0; a rotate token naming a joint outside
    `ROTATABLE` -> all 0 (refused, not parsed); `live_arm._joint_obs_delta` agrees on shoulder_lift. Mutation
    `parse_any_name` accepts the foreign joint: red.
R5  THE PER-JOINT FOLD: `proprioception_deltas` wraps each joint (a 359.9 -> 0.1 move folds to +0.2 deg, the T037
    seam red, per joint) and `proprioception_delta` equals its mean bit-for-bit on 200 random triples. Mutation
    `no_wrap`: red.
R6  THE DISSOCIATION ON A CONSTRUCTED BODY: five synthetic channels through the shipped `AgencyMonitor`, one of them
    decoupled after passage 300 by an additive U(-2, 2) x command-scale disturbance -- (a1) MET, (a2) MET on the four,
    (a3) MET; the twin with NO decoupling reads (a1) NOT MET and (a3) NOT MET. Mutation `blind_scorer` (c_j
    replaced by 1 - c_j on the shoved arm) inverts (a1): red. Mutation `deadband_zero` (DEADBAND = 0) makes (a2)
    depend on float noise: reported as the constant's sensitivity, not a red.
Each red runs on its constructed input and FAILS its assertion at commit 3 of 2.9; at commit 4 each is green and its
mutation is applied IN THE TEST and in `t079_reds()` in the module's self-check line (six booleans asserted).

### 2.5 The controls that are not the method

The cause-blind twin (the shuffle floor, 2.1) -- the same outcomes, the causal link removed, exactly the C5 row's
dissociation input; the INTACT arm on the same seed and stream as the paired partner of the shoved arm; the four
UNSHOVED joints inside the shoved recording (the within-body control that "preserved elsewhere" is read on); the
module's own constructed `_experiment` (R2) with its known trajectory; the shove ladder (reported); the determinism
twin (p). The incumbent feed (action label / qmax / reward) is NOT a control -- it is retired at build time on the
2026-07-28 and 1.3 measurements and its `n_scored` on the same ticks is reported for the record.

### 2.6 Disposition (fixed in advance; the honesty lens adjudicates any reading of these words)

- PASS: the instrument separates a channel whose outcomes it caused from one whose outcomes it observed, on this
  body, above the cause-blind twin's rate, with the other channels preserved. CHARGED: `conclusion_protocol.
  CONCLUSION["agency"]` gets `reading` = the mean over seeds and rotatable joints of (c_j^intact - floor_j^intact)
  (a number in [0, 1], the licensed sentence's "above a cause-blind twin's rate") and `surface_symbols` =
  ("agency_controllability",); `Brain.state()` publishes `agency_controllability` = the live composition's
  `AgencyMonitor.agency()["agency_signal"]` and the cockpit prints it as a consciousness row WITH the licensed and
  refused sentences in the same view and a PROVENANCE line naming the per-joint feed, the sim body and the pin --
  the consumer P3 asked for. tasks.md T079 -> [X] PASS. `route_around` stays UNWIRED (the 2026-07-28 rejection
  stands; disclosed).
- NOT MET: the instrument does not resolve a 12-deg external force on one joint of this body from an intact one at
  the bar, or the machine disagrees. NO reading, NO symbol, nothing printed; the per-joint feed STAYS (it is the P2
  replacement of a measured-meaningless feed, not a claim) and `AgencyMonitor`'s P3 disposition becomes EXEMPT with
  this record as its reason (it is consumed by no one and measured unable to separate on this body); tasks.md T079
  -> [X] NOT MET with the symbol named and printed nowhere (T077's shape, so the arming arm sees it). Nothing is
  deleted: the constructed reds fired (R1-R6), so the instrument is an instrument; the null is a fact about this
  body's coupling under this disturbance.
- REFUSED: the apparatus's or instrument's fault named ((p), (r) or (g0)); the row stays [ ]; no surface changes.
The noun in every branch is "the per-joint command-to-outcome coupling the instrument reads" -- the barred X of 2.2.

### 2.7 What this row does NOT claim

No felt authorship, no ownership of action, no free will (the refused sentence); nothing about the real arm
(`SAFETY_CALIBRATED` False; the body is the MuJoCo SO-101, the shove is `inject_disturbance`, a test-only sim
exafference); nothing about the loop's POLICY reaction to a disturbance (reported); nothing about a dead effector
(no joint is ever disabled -- the port that exists shoves, so the perturbation is an external force on a working
servo); nothing about the daemon's brain; no aggregate consciousness score.

### 2.8 Widths, cost, host, declared constants and their sensitivity

Seeds 0-7, N = 1,200 ticks per recording, two arms -> 19,200 brain-ticks at the measured 61.5 ms (1.3, seed 0 --
RE-MEASURED on the built apparatus in 3.0 before the campaign) -> ~20 min sequential, plus the 100-tick twin and the
floors (2,000 draws x 5 joints x 16 recordings as batched torch recurrences over ~585 scored passages each --
seconds). Refusal: above 3 h the campaign is re-sized and this section amended. DECLARED CONSTANTS and their
consequence, COMPUTED in 3.0 (the dry run): the incumbent's alpha 0.02 / dead_thresh 0.35 / high_thresh 0.55 / delay
48 / recover_n 32 / warmup 120 / var_floor 1e-4 (read, not chosen; DEADBAND 0.20 derived); A = 12 deg = 2 x
DEFAULT_STEP_DEG (the ladder 3 / 6 / 12 reported in 3.0); n_shuffle 2,000 (the floor's Monte-Carlo sd over 12
independent floors, in 3.0); the stream offset 150 x seed; J* = elbow_flex. Host: Python 3.14.6 free-threaded, torch
2.11.0+cu128 on the RTX 5070 where present; GeNN not used (no GPU tick is the instrument here).

### 2.9 Commit order (one change each; the stubs before the reds so the reds RUN red; surfaces before the record)

1. This file + PLAN-FINAL Part F row C-R8 (T079) -> 2. the skeleton: `efferent_boundary.proprioception_deltas` /
`intended_deltas` / `last_copy` per-joint fields as STUBS (all-zero lists), `agency_monitor.t079_read` / `t079_floor`
/ `t079_contrasts` / `t079_reds` returning null readings, the registrations -> 3. `tests/test_agency_monitor.py`,
R1-R6 committed RED (each runs on its constructed input and fails; output pasted) -> 4. the instruments green, each
red reddened by its mutated scorer in-file and in `t079_reds()`; `live_arm._joint_obs_delta` routed through the
parser; census gates -> 5. the feed switch in `apply_observed` (the tick-level feed deleted) + the recorder +
`--t079-campaign` + the determinism precondition + the pin writer (`tools/t079_agency.json`, once); the apparatus
dry run on ONE seed at 300 ticks: the ladder A = 3 / 6 / 12 REPORTED (its output named with its script in 3.0),
the stream-offset check, the floor's Monte-Carlo sd, the cost re-measured; NO barred conjunct is read there (one
seed cannot carry a paired t) -> 6. the campaign, the pin (once) -> 7. surfaces and annotations per 2.6 (tasks.md's
dated note; PLAN-FINAL C4 note; spec.md SC-009 note; on PASS the conclusion row, `Brain.state()` and the cockpit
row; on NOT MET / REFUSED nothing in `conclusion_protocol`, `brain_telemetry` or the cockpit changes) -> 8. section
3 the record -> panel (3 lenses + refuters judging CLAIM and FIX) -> fold -> the battery on the stacked head (the
Codex closing report when the cap lifts, 2.0).

### 2.10 Amendment, dated 2026-09-15 22:10 local, BEFORE the campaign (one readability clause corrected on an apparatus
### fact; the build's deviations disclosed) -- no gate's form, bar, disposition, arm or N changes

(r) AS WRITTEN WOULD REFUSE THE CAMPAIGN ON THE BODY'S NORMAL BEHAVIOUR. 2.3 (r) requires "the passage count equals the
awake-tick count". The 300-tick dry run read 146 passages over 146 awake ticks (2.9 step 5, `scratchpad/t079/dry_run.json`),
and the pre-registration inferred one passage per awake tick from it. The apparatus run at the CAMPAIGN width
(`scratchpad/t079/probe_width.py`, seed 0, 1,200 ticks, both arms, 172 s -- an APPARATUS reading on one seed, disclosed:
it read per-joint `c`, the status machine and both floor constructions on seed 0 and computed NO paired t and no verdict)
read 623 passages over 593 awake ticks on the INTACT arm and 593 over 593 on the shoved arm. The 30 extra passages are the
POSTURAL RECOVERY REFLEX of `AutonomousArm.step` (autonomous_loop.py, the `_GROUND_RECOVER_TICKS` block after the main
apply): on a ground contact the loop issues a SECOND `apply_observed` in the same tick, which T037 counts as a passage, the
tap observes, and 2.1's feed feeds. A recovery passage is exactly what the copy should see -- a command with its outcome --
and refusing a recording for carrying it would refuse the body for standing up. AMENDED: (r)'s clause reads "every awake
tick carries at least ONE passage (the T037 1:1 step-and-tick contract, read per tick), the extra passages (recoveries and
self-clears, `passages - awake`) are COUNTED, fed and reported per seed and arm as `extra_passages`, never a refusal"; the
recorder already records `passages_this_tick`. Nothing else moves: (p), (g0), (a1), (a2), (a3), the bar, the arms, N, the
shove, the floor. Landed as the commit after this one, before the campaign, with its constructed red (an awake tick with no
passage REFUSES; an awake tick with two passages READS, `extra_passages` 1).

THE BUILD'S DEVIATIONS FROM 2.1-2.9, each disclosed in-code and here (the builder's report, `scratchpad/t079/builder_brief.md`
was the brief): (i) `_q95` was LIFTED into `substrate/_mathutils.q95` (`attention_schema._q95` now its alias) rather than
imported from `attention_schema`, because `agency_monitor` is LIVE and `attention_schema` is dormancy-EXEMPT, so importing it
-- even lazily -- makes `dormancy_gate` name the exemption stale and exit 1 (probed on a throwaway edit before the commit);
the formula is verbatim and no consumer's number moves -- T078 3.5 item 23's successor lift, taken by this row because the
gate forced it. (ii) `proprioception_deltas` is DEFINED in `afferent_boundary` (where `proprioception_delta` lives; the
efferent module imports the afferent one at module top, so the reverse would be an import cycle) and re-exported by
`efferent_boundary`; its units are the scalar's (degrees / 180), so `u_j` is in degrees and `ds_j` in normalised units --
Pearson is scale-invariant and `var_floor` reads the command variance, so no reading moves. (iii) R1's `floor_unpermuted`
reddens the COUPLED channel's margin (every draw equals the reading exactly, so the decoupled channel still reads "at its
floor"), and R6's `blind_scorer` reddens the reading through (a2), not (a1) -- each red is reddened in fact, on the clause
that fires. (iv) The recorder wraps `apply_observed` as well as `efferent.observe`, because the feed runs after the tap
returns and a `ctrl` read inside the inner wrap would be one passage stale (T077 3.2's forbidden pairing). (v)
`live_arm._joint_obs_delta` now reads 0.0 on a MOVETO token (the retired regex returned the setpoint as a delta; unreachable
on its own path either way). (vi) Builder constants, in-file: the reds' `n_shuffle` 200 (the campaign's 2,000 is untouched),
R6's 900 passages at a 1-in-10 command duty (1.3's measured shape), the dry run's 300 ticks and 12 floor repeats.

TWO APPARATUS FACTS FROM THE CAMPAIGN-WIDTH RUN, reported now so 3.0 cannot present them as new: (a) the cause-blind
twin's floor as BUILT (2.1's words: `u_j` permuted over the SCORED window, the EMA carrying real coupling across warmup)
reads q95 0.160-0.226 on seed 0's intact arm against 0.106-0.112 when the permutation covers the WHOLE recording -- the
as-built floor is HIGHER, so (g0) is harder to meet, never easier; the as-built construction stays the pre-registered one
and the whole-recording floor is reported beside it in 3.3 as the constant's sensitivity. (b) The intact coupling at this
width is not ~1.0 on every joint (seed 0: 0.99 / 0.95 / 0.86 / 1.00 / 1.00): the policy drives shoulder_lift and elbow_flex
into their limits and the ground (30 recoveries), where a commanded step achieves less than its delta -- the body's fact,
inside the reading as pre-registered.
