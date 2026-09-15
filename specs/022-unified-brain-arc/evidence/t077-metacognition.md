# T077 -- the metacognition instrument as an extension of the incumbents (specs/022 Phase 4 / C4; plan row :162; SC-009's first half)

Sections 1-2 are the PRE-REGISTRATION, written 2026-09-15 (05:10-10:20 local) BEFORE a line of this row's code exists in
the tree and frozen at the commit that lands them; 2.0 records the Codex design read that shaped them; amendments are
dated appended sections; section 3 is written after the runs. Every number in sections 1-2 is a source line, a plan
constant, or an apparatus probe named with its script (no campaign number has been seen).

## 1. Territory, measured before the design (each fact with its source line, read at HEAD `97db848`)

1. Two LIF reliability pools exist and both are the row's incumbents. `cortex/metacognitive_monitor.py`
   (`MetacognitiveMonitor`, :157; `update` :261-310 drives 24 LIFNeurons x 16 steps and reads `reliability()` as
   spikes/(24*16); gates `learn_gate` :330, `explore` :337, `grow_trigger` :346, `hold`/`abstain` :361/:376 with the
   hold bar a running low quantile `hold_p = 0.10` of the pool's own rate, :209-236; `lesion=True` makes the GATES
   no-ops, `update` still runs the pool). `cortex/second_order_access.py` (`SecondOrderAccess`, :78; `observe` :127-164,
   the same pool shape over availability / access strength / a hard IGNITION-FAILURE inhibition / the first-order
   confidence; `hold` :170 at `c_hold = 0.30`; `calibration` :175 = the running agreement of the thresholded call with
   "truly available", a binary decision's accuracy, 0.5 chance on a balanced stream).
2. The monitor is LIVE on every wake tick: constructed `MetacognitiveMonitor(base=0.4, seed=self._seed)` at
   brain.py:325 (no lesion kwarg reaches it); fed at `_wake_tick` :2468 -> `_update_metacog` :2473-2517 with
   confidence = `self_model.state()["confidence"] * decision_conf` (decision_conf = the BG margin over its own running
   EMA, :2495-2497), surprise = `st["surprise"][0]`, learning_progress, novelty = surprise + 0.3 * pc_dev, conflict =
   the ACC conflict (:2506-2512). Consumers: `_metacog_rel` -> state() :3947 (the cockpit's `reliability` row);
   `_metacog_hold` -> the chat abstention branch :4346-4353; `learn_gate()` -> the lifelong learn gate :3590;
   `_metacog_explore`. Its P6 charge was measured 2026-08-05 (the module's docstring :60-113): reliability differs
   295/300 ticks, the HOLD route changes 3/30 sampled replies, and -- its own words -- the `learn_gate()` consumer
   "never fires in this configuration, in ANY arm ... inert for reasons that predate this wire", with "what is NOT
   claimed: that abstaining more often is BETTER". A measured route charge and an inert learn consumer; NOT an
   error-tracking claim and NOT a learning-outcome licence (2.0 point e).
3. `SecondOrderAccess` is constructed at brain.py:340-344 and observed ONLY on the chat path (`respond` :4236-4246,
   first_order_conf = `_metacog_rel`); consumer `_second_order_hold` :4346/:4352. brain.py's own `_access_state`
   reader table (:2971-2980) reads `respond :3435 SecondOrderAccess 0/56 0/56 NONE`: pinning the ignition flag True
   or False through this reader moved no channel on 56 awake ticks. That is the P3 shape the tree recorded and never
   adjudicated; its unique inputs vs the monitor are availability, access strength and the ignition-failure term.
   COUNTER-EVIDENCE, one live turn (2026-09-15 06:21, the post-push daemon poll on a brain 41 wake ticks old): the
   chat reply routed as "Gf (second-order access hold: abstain)" -- `_second_order_hold` was True and DECIDED the
   route. So the hold is not inert on a young brain's chat path; the 0/56 table measured the ignition FLAG's reach
   through this reader, not the hold's. Arm D measures both counts before any P2 act; the retirement branch needs
   0 of M on BOTH counts and this single turn already makes that unlikely.
4. The first-order stream with held-out items, live: `cortex/neurogenesis_engine.py` :306-322 -- on each wake tick
   the PREVIOUS percept's prediction `pred = m.decide(S)[0]` is scored against the current category BEFORE any learning
   on that tick (`live_n += 1; live_correct += int(pred == cat)`): PREQUENTIAL correctness, each item held out by
   construction. The engine is fed inside `_wake_tick` (brain.py ~:2400, `if self.neuro is not None and res.get("unit")
   is not None`), BEFORE `_update_metacog` (:2468) -- so the second-order read at tick t is the one in force when the
   prediction scored at tick t+1 was made: the pairing lag is ONE tick. Daemon poll 2026-09-15 00:10: live_accuracy
   0.932 over 486 items (`neurogenesis.live_accuracy`), i.e. ~6.8 % errors.
5. The C5 row this instrument fills: `substrate/conclusion_protocol.py` :114-121 -- licensed "the system's confidence
   tracks its own error rate better than a fixed prior does, on held-out items"; refused "this licenses no claim that
   the system knows that it knows in any experiential sense"; dissociation "shuffled confidence-outcome pairing must
   destroy the tracking, or the instrument is a marker"; reading None; surfaces (). Once tasks.md marks T077 `[X]`
   the row must declare a surface symbol (:222-226); a reading needs a symbol (:218); a surface that prints the symbol
   must carry the licensed sentence's first six words within a 7-line window (:299-312). `SURFACE_BASELINE = 1`
   (cockpit.py:920, integration) may fall, never rise.
6. The paired t the plan's rows reuse: `cortex/prior_drift_harness.contrast(a, b)` -- paired one-sided t over 8 seeds
   against 1.895 with the 2^8 sign-flip FPR (T063 :417 at HEAD; T070 reused it). P2: this row adds no second one.
7. Governing words, quoted in `scratchpad/t077/design.md` section 0: tasks.md :162; spec.md :84 and FR-028 :145;
   SYNTHESIS.md :64 and S6 :84; INVENTION_REGISTER.md :80-96; PLAN-FINAL :913-916 (C4) and Part F :1011-1013 (S = 8,
   t > 1.895, FPR printed, no statistic normalized by a quantity a control destroys, no majority-of-seeds rule);
   FR-031 (T077 is a P8 invention owing a register row).

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex, sent 09:32, answered 09:37, 2026-09-15; `scratchpad/t077/codex_design_read.md`, archived)

Seven points, each reproduced at its cited site before this section changed; what each did to the text:
1. THE FLOOR was a heuristic wearing a derivation (it dropped (1 - m/N), reasoned from Brier for an AUROC statistic, and
   repurposed the 8-seed bar as a per-stream z). REPLACED: readability is decided per seed on the permutation null of
   auroc2 for THAT stream (ties included, 2,000 draws; the exact 2^S enumeration of `permutation_fpr`
   (prior_drift_harness.py:424) is the tree precedent for exact nulls): READABLE iff both outcome classes are present
   AND the null's 95th percentile is at or below 0.5 + rho, with rho = 0.25 DECLARED -- the midpoint of the statistic's
   range between chance and the oracle -- and disclosed as chosen, not derived. Its consequence is computed, not
   declared: at N = 600 the null sd sqrt((N+1)/(12 m (N-m))) puts q95 under 0.75 from about m = 4 minority outcomes,
   which is the count the retired heuristic named. The incumbent's own synthetic world was probed as a tree-derived
   resolution and REFUSED for that use: it reads auroc2 = 1.000 on all 8 seeds (a saturated positive,
   `probe_ref_strength.py`).
2. DELETING THE CHAT ABSTENTION ON A PERCEPTION NULL is domain over-reach (POLICIES :763 targets the invented method
   that measures nothing; :188 requires a matched-input A/B before any incumbent retires). ACCEPTED: 2.6 deletes nothing
   of the incumbents on any null; the reading is recorded and the row's C5 sentence is not written.
3. THE LAG'S JUSTIFICATION was wrong (the prediction is MADE and SCORED on the same tick, from `_prev`, at
   neurogenesis_engine.py:306-322, before `_update_metacog` at brain.py:2468). CORRECTED: the prospective read is the
   LAST FED read strictly before the scoring tick; scored events are aligned by tick index, skipped ticks are skipped.
4. ARM D's counts establish neither uselessness nor correctness (P3 wants outcome evidence :280; P2 wants a matched-input
   comparison :188); porting the ignition term alone drops two capabilities; keeping both pools does not satisfy "one
   mechanism survives". ACCEPTED, with the row's words read this way: the ONE mechanism this row lands is the
   calibration scorer (`track_error` replaces `SecondOrderAccess.calibration()`); the two POOLS' merge is a P2 arc of its
   own, not this row's deliverable, and arm D's counts are that arc's input. 2.6 retires no pool.
5. (b4) WAS A NULL-AS-CONJUNCT. REPLACED by two positive equivalence contrasts (TOST) with the margin sized from the run's
   own seed dispersion, plus a first-order COMPETENCE floor against the engine's own persistence predictor.
(d) THE BIGGEST DEFECT: the spec's dissociation (spec.md:84) is "lesion the second-order layer -> first-order preserved
   while calibration collapses" IN VIVO; the draft lesioned only the gates and used an offline passthrough. REPLACED:
   the LESIONED arm removes the second-order READ (the pool's output replaced by the first-order input, gates off) and
   (b3) is the in-vivo contrast intact vs lesioned; the offline passthrough stays as a reported control.
(e) THE "MEASURED LEARNING OUTCOME" LICENCE for `learn_gate` was FALSE -- the module's own record (:96-113) says the
   lifelong-learning consumer "never fires in this configuration, in ANY arm" and "what is NOT claimed: that abstaining
   more often is BETTER". CORRECTED in 2.6 and 1.2: the incumbent's gates carry a measured route charge (3/30 routes)
   and an inert learn consumer; neither is a licence this row relies on or revokes.

### 2.1 The mechanism (P2: one calibration scorer, one paired t, the incumbent pools -- T077 adds ONE function)

`cortex/metacognitive_monitor.py` gains `track_error(conf, correct, t_bar=1.895, n_perm=200, seed)` -- the ONE
scorer of "confidence tracks error" in the tree -- and a campaign entry `--t077-campaign` that writes
`tools/t077_metacognition.json` once (the pin; never re-run against it). No new module (T070 3.5 item 17). The
statistic is rank-based: `auroc2` = P(conf on a correct item > conf on an incorrect item), ties 1/2 (type-2
discrimination); the fixed prior's auroc2 is exactly 0.5. Reported beside, never barred: `brier_gain` = Brier(prequential
prior) - Brier(conf), the prior at item i being the running accuracy of items < i (no lookahead; the pool's rate is not a
probability and a fitted rescaling would be lookahead). `SecondOrderAccess.calibration()` is a second calibration
mechanism (a thresholded call's accuracy): it is RE-POINTED to `track_error` over its `_hist` (2.9 step 3), so exactly
one calibration scorer survives. The two LIF pools are NOT merged or retired by this row (2.0 point 4): their P2
adjudication is a successor arc with arm D's counts as its input, named in 2.7.

### 2.2 The arms (one command, seeds 0-7, a scratch brain; the daemon's brain is never touched)

Population: `make_brain(seed, persist=False)` under a throwaway `NORD_HOME` (`throwaway_home()`), N = 600 ticks (2.8).
Per tick the campaign records: the monitor's five inputs as fed at brain.py:2506-2512 (by wrapping the INSTANCE's
`update`, a keyword call), the unrounded read it returned, `_metacog_hold`, `learn_gate()`, the engine's prequential
outcome and persistence hit (the deltas of `live_correct` and `live_persist` when `live_n` increments,
neurogenesis_engine.py:321-323) and the vision classifier's per-frame hit (`_perc_hits` delta). Nothing in `src/` is
instrumented for the campaign. PAIRING (2.0 point 3): the outcome scored at tick t is paired with the LAST FED read
strictly before t (the prospective read; the prediction is made and scored on tick t from the previous percept, before
the monitor's update on t); ticks without a scored outcome, and outcomes with no fed read before them, are skipped and
counted.
- INTACT -- the brain as built.
- LESIONED (2.0 point d) -- the same seed with the second-order LAYER removed in vivo: the campaign sets
  `brain._metacog.lesion = True` (every gate a no-op, :330-380) AND replaces the instance's `reliability()` by the
  first-order confidence input it was last fed (so `_metacog_rel`, the read every consumer sees, becomes the first-order
  signal). The pool still runs (`update` has no lesion branch); nothing it emits reaches a consumer. Recorded exactly as
  INTACT: the same pairs, so (b3) and (b4) are read on THIS run.
- PASSTHROUGH (offline, from INTACT's recording; REPORTED beside (b3), never barred) -- the first-order input scored in
  place of the second-order read on the intact stream.
- SHUFFLED (offline) -- the pairing permuted, `n_perm` = 200 permutations per seed (the C5 dissociation input).
- ORACLE (offline) -- conf := outcome (the positive control).
- LAG CONTROL (offline, reported) -- the pairing shifted by +5 ticks; reported beside, not barred.
- PERCEPTION STREAM (recorded in the same run, REPORTED beside, never barred): the vision classifier's per-frame
  correctness (`pred == label`, brain.py:1653-1656, `_perc_hits` delta per tick; accuracy ~0.93 on the scratch brain
  at 600 ticks) paired with the same second-order read at lag 0 (the frame is scored on the tick it is perceived,
  before `_update_metacog`) -- a second first-order stream with ground truth, so the claim is READ on the readout's
  prequential stream (barred) and SEEN on the perception stream (reported); disagreement between them is recorded,
  not adjudicated (obs #2106's shape).
- ARM D, the P3 probe (its own command, before the campaign; ALREADY RUN ONCE on seed 0, 2026-09-15 06:35,
  `scratchpad/t077/probe_armD_cost.log`): M = 40 chat turns on the INTACT brain (`respond`, 5 ticks between turns),
  recording `_second_order_hold`, `_metacog_hold`, conf2 and the route; then the same turns with `observe()` neutered
  (conf2 pinned to its intact mean); then a FLIP neuter (conf2 pinned to 1.0, hold False) counting the routes the gate
  decides. Seed 0 read: hold True 39/40 (36 with the monitor's hold False), conf2 mean 0.0084 vs `c_hold` 0.30,
  pin-to-mean changed 0/40 routes and 1/40 holds. The campaign repeats it on seeds 0-7 and writes the counts in 3.0
  BEFORE any act of 2.6's arm-D branch.

### 2.3 The statistic and the verdict rule (written first)

Every barred conjunct is `contrast()`'s paired one-sided t over 8 seeds against 1.895 with its sign-flip FPR printed;
a "null" is never a conjunct (2.0 point 5).
(a) READABLE, per seed (2.0 point 1): both outcome classes present AND the permutation null of auroc2 for that stream
    (2,000 shuffles of the pairing, ties at 1/2) has its 95th percentile <= 0.5 + rho, rho = 0.25 DECLARED (the
    statistic's midpoint between chance and the oracle; chosen, disclosed). Any seed unreadable -> the campaign is
    REFUSED (no reading), not partially read. The minority count and q95 are printed per seed.
(b1) TRACKING: t(auroc2_intact - 0.5) > 1.895.
(b2) NOT A MARKER: t(auroc2_intact - auroc2_shuffled_median) > 1.895; per seed, auroc2_intact above its q95 is reported.
(b3) CALIBRATION COLLAPSES UNDER THE LESION (the dissociation's calibration half, in vivo): t(auroc2_intact -
     auroc2_lesioned) > 1.895, both read on their own runs' pairs. The offline passthrough contrast is reported beside.
(b4) FIRST-ORDER PRESERVED UNDER THE LESION (the behavioural half), as TWO positive equivalence contrasts (TOST) with the
     margin Delta = the standard deviation over the 8 seeds of acc_intact (the run's own dispersion, a rule fixed here,
     a number computed there): t((acc_intact - acc_lesioned) + Delta) > 1.895 AND t(Delta - (acc_intact - acc_lesioned))
     > 1.895, where acc is the prequential accuracy over the run's scored outcomes.
(b5) NOT BLIND: oracle auroc2 = 1.0 on every seed and above that seed's q95.
(b6) FIRST-ORDER COMPETENCE (the floor (b4) would otherwise lack): t(acc_intact - persistence_intact) > 1.895, where
     persistence is the engine's own "previous category repeats" predictor (`live_persist`/`live_n`).
(c)  R0: the offline replay of `MetacognitiveMonitor(seed)` on INTACT's recorded inputs reproduces the recorded read
     exactly on every fed tick of every seed.
PASS = (a), (b1)-(b6), (c) all MET. REPORTED, never barred: per-input attribution (delta-auroc2 with each of the five
inputs pinned to its stream mean in the replay, P8 clause 3), the +5-tick lag control, the perception stream's auroc2,
brier_gain, and arm D's counts.

### 2.4 The reds (RUN RED before the mechanism; in-command with their controls thereafter)

R0 REPLAY -- the offline replay reproduces the recorded read byte-identically; a recorded read that differs by 1e-9 on
   ONE tick is CAUGHT, and an input perturbation large enough to move one spike is caught. DISCLOSED (measured
   2026-09-15 07:30 before the freeze, `scratchpad/t077/check_impl.py`): the read is a spike COUNT over 24 x 16 steps,
   quantized at 1/384, so an INPUT perturbation of 1e-9 is invisible by construction -- the first draft's red was on
   the wrong side and would have passed vacuously.
R1 DEGENERACY -- a constructed stream with 2 errors in 400 -> `readable=False`, NO auroc2 emitted (its null q95 sits
   near 0.84, above 0.75); the control: 6 errors in 400 is readable (q95 near 0.69).
R2 RELAY -- a constructed stream whose second-order read IS the first-order input -> the relay contrast is exactly 0.
R5 A LESION THAT DOES NOT COLLAPSE -- a constructed pair of runs whose "lesioned" read equals the intact read -> (b3)'s
   contrast is exactly 0 and reads NOT MET (the in-vivo half of the dissociation can fail).
R3 MARKER -- the shuffled pairing on a constructed tracker -> auroc2 inside the permutation band while the oracle sits
   outside it (R3 is the dissociation input; its control is the oracle).
R4 NOT RELAY-PROOF -- the monitor's own synthetic world (its self-check :559-662) with hold gating commits: accuracy
   DEPENDS on the gate -> (b4) reads NOT MET on that construction.
Each red is a test in `tests/test_metacognition_instrument.py` and a line of the module's `--self-check`, committed
RED first (2.9 step 1).

### 2.5 The controls that are not the method

The shuffle (R3 / b2), the oracle (b5), the passthrough (b3), the lesioned twin (b4), the lag control (reported), and
the per-input neuters. None reads the second-order pool's output as its own input.

### 2.6 Disposition on a null (fixed in advance; the honesty lens adjudicates any reading of these words)

- PASS: `CONCLUSION["metacognition"].reading` := the 8-seed mean auroc2; surfaces := ("metacog_tracking",); the
  cockpit prints that row with the licensed sentence inside its 7-line window and a PROVENANCE line naming the stream,
  N, the seeds and the pin; `SURFACE_BASELINE` unchanged; tasks.md `[X]`.
- NULL on any of (b1)-(b4), (b6), with (a), (b5) and (c) MET (the instrument works; the claim about the layer does not
  hold on this stream): the reading is recorded NOT MET with every conjunct's number; `reading` stays None and no surface
  symbol is declared; the C5 licensed sentence is NOT written; the cockpit's `reliability` row and its PROVENANCE gain a
  dated line naming what was measured (map/territory); NOTHING of the incumbents is deleted or re-labelled beyond that
  line (2.0 points 2 and e: P8 clause 5 targets an invented method that measures nothing, and this instrument measured;
  the incumbents' own status -- a measured route charge of 3/30 and an inert learn consumer -- is theirs, disclosed,
  not this row's to revoke). The instrument itself STAYS: its reds fire and its reading is a number about a mechanism.
- REFUSED on (a), (b5) or (c): no reading; the instrument's own fault named; no disposition fires; the row stays open.
- ARM D (2.2): its counts (hold-True fraction, conf2 distribution, pin-to-mean route changes, flip-neuter route changes)
  are written in section 3 and are the INPUT of a successor P2 arc that adjudicates the two pools on a matched-input A/B
  (POLICIES :188) with outcome evidence (:280). This row retires no pool and ports nothing (2.0 point 4).

### 2.7 What this row does NOT claim

The two LIF pools' P2 fate (merge, port, or retire) is NOT decided here: arm D's counts are the successor arc's input,
and "one mechanism survives" is read as the one calibration scorer (2.0 point 4; the panel's honesty lens is asked
whether that reading is a relabelling of the row's words). The incumbent `learn_gate` consumer is inert by the
monitor's own record (:96-113) -- carried as a P3 opening, not touched here.
No experiential claim (the C5 refused sentence, verbatim on every surface). The reading is about ONE stream: the
perception readout's prequential correctness on a scratch brain at the deployed width, N ticks, 8 seeds -- not chat
answers, not reasoning, not the arm. "Live" = the live tick's own signals recorded off a scratch brain; the daemon's
brain is untouched. Prequential correctness is the held-out reading; a separate held-out SET is not scored. The
per-input attribution is reported, never barred. No number in this file is a reading; every number above is a source
line or a constant the plan already fixes.

### 2.8 Widths, cost, host

MEASURED 2026-09-15 06:35 (the probe above): 50 ticks of `make_brain(seed=0, persist=False)` under a scratch NORD_HOME
in 3.1 s = 62.7 ms/tick (24 awake), so N = 600 x 8 seeds x 2 arms projects to 10 min sequential. N is therefore set to
600 and the campaign runs the seeds SEQUENTIALLY in one process (no thread or process pool; 8-way parallelism would
buy ~9 min and cost determinism questions). The refusal rule stays: re-size if a re-measured projection exceeds 3 h.
THE FLOOR AT THIS N, MEASURED 2026-09-15 07:55 (`scratchpad/t077/probe_perr.py`, seeds 0 and 1, 1,200 ticks each): the
readout's prequential errors at t = 300 / 600 / 900 / 1,200 read 0 / 46 / 93 / 165 of 122 / 270 / 417 / 569 scored on
seed 0 and 30 / 57 / 63 / 81 of 126 / 277 / 425 / 571 on seed 1 -- the first 300 ticks of seed 0 carry NO error (three
categories only), so a shorter campaign would REFUSE on (a); at N = 600 both seeds sit an order above the floor of 4.
The scratch brain's readout is weaker than the daemon's (0.83 / 0.79 vs 0.93 over 486), disclosed: the reading is
about the scratch brain's own stream. A 1,200-tick apparatus dry run (80 ticks, one seed, pin redirected) ran end to
end in 33 s; its numbers are not readings. Every arm prints unbuffered; the log is line-buffered (obs #2170). Host: Python 3.14.6 free-threaded, torch 2.11.0+cu128 on the RTX 5070 where the brain
already uses it; GeNN not used (no GPU tick in this row).

### 2.9 Commit order (one change each; reds first; census gates after every symbol-adding commit)

1. `tests/test_metacognition_instrument.py` + the `--self-check` lines, RED (R0-R4 fail to import or fail).
2. `track_error` + the replay + the reds green; `make_brain(metacog_lesion=)` threaded (byte-identical default).
3. `SecondOrderAccess.calibration()` re-pointed to `track_error` over its `_hist` (one scorer survives; its self-check's
   four controls re-expressed on the same scorer); SYNTHESIS S6 row.
4. The cost probe (2.8) recorded; the campaign; the pin `tools/t077_metacognition.json`.
5. The conclusion row / cockpit / PROVENANCE per 2.6; INVENTION_REGISTER row; PLAN-FINAL Part F row C-R6; spec.md SC-009
   dated note; tasks.md T077.
6. Section 3 the record -> panel (3 lenses + refuters, CLAIM and FIX separately) -> fold -> Codex closing report.

### 2.10 Amendment, dated 2026-09-15 ~11:10 local, BEFORE any campaign number exists: the paired t is LIFTED out of the
### EXEMPT harness (an apparatus fact the first campaign commit measured; no conjunct, control or disposition changes)

WHAT WAS SEEN. Commit order step 4 placed the campaign entry in `cortex/metacognitive_monitor.py` with
`prior_drift_harness.contrast` imported LAZILY inside `run_t077_campaign` (2.1 named the trap and thought a lazy import
avoided it). `substrate/dormancy_gate --assert` read RED on that tree: "5 stale EXEMPT entry/entries -- now REACHABLE:
['determinism_gate', 'h01_ingest', 'kernel_state', 'kernel_tick', 'prior_drift_harness']" -- the gate counts an import
inside a function or conditional as reachability ("This gate is PERMISSIVE by construction"), so a live CORE module
importing the EXEMPT harness at any depth stales those rows. Retiring them to green the gate is the lie T070's record
refused (t070-integration-kernel.md 3.5, `dormancy_gate.py`'s own row text); restating `contrast` in the monitor is the
P2 defect.

WHAT CHANGES. `T_BAR`, `Contrast`, `paired_t`, `permutation_fpr` and `contrast` move VERBATIM from
`cortex/prior_drift_harness.py` (T063's 2.6, reused by T070) into `substrate/paired_contrast.py`, a live-safe module
registered in `__init__.py` REGIONS and `run_selfchecks.py` with its own self-check (its reds: a sign-balanced sample
NOT MET, a reversed sample NOT MET, the exact 1/256 FPR of eight same-sign differences, the degenerate branches); the
harness imports and re-exports the five names (every `prior_drift_harness.contrast` call and T063's tests resolve
unchanged; the harness keeps every other symbol and its exemption); the campaign imports `contrast` from the new
module. ONE implementation survives, where any module may reach it. No number in 2.3 changes; the reds of 2.4 are
unchanged; the monitor's `T_BAR` restatement is replaced by the import (the reds' equality assertion still holds by
identity).

WHAT DOES NOT CHANGE. 2.9's steps and their order (this is step 4's first commit); the scorer (48750eb) and the
re-pointing (5261eb0) stand as committed.

## 3. Results (written 2026-09-15 after the campaign; nothing above this line was edited after 2.10)

### 3.1 The build, red-first, and the commits

`47c41aa` the pre-registration (sections 1-2.9 + PLAN-FINAL Part F row C-R6) -> `654106b` the reds R0-R5 committed RED
(`ImportError: cannot import name 'RHO'`) -> `48750eb` `track_error` + `replay_reliability` in the incumbent, the reds
green (7 passed; the label entropy reuses `integration_measure.source_entropy` at the graphify guard's naming) ->
`5261eb0` `SecondOrderAccess.calibration()` re-pointed to the scorer (its self-check: calibration 1.000 against shuffled
0.500 / first-order 0.641 / fixed-prior 0.500 / ignition-echo 0.864; SYNTHESIS S6 row) -> `14807ef` the paired t LIFTED
into `substrate/paired_contrast.py` (2.10; the dormancy gate had read five EXEMPT rows stale on the lazy import) ->
`07fd69a` the campaign entry (`--t077-campaign`; the apparatus dry-run on one seed at 80 ticks refused on (a) as it
must) -> `d4e86f6` the pin. Every red of 2.4 fired on its constructed input before the campaign; R0 also held on every
live recording (the campaign asserts the exact replay before scoring a seed).

### 3.2 The campaign (`tools/t077_metacognition.json`, run at `07fd69a`, committed `d4e86f6`)

2026-09-15 10:06-10:25 local, 1,140 s, under a throwaway NORD_HOME: seeds 0-7, N = 600 ticks, an INTACT and a LESIONED
scratch brain per seed, arm D three times per seed (intact, pin-to-mean, flip). Prospective pairs per seed 270 / 277 /
274 / 279 / 279 / 279 / 269 / 279 (0 skipped); scored outcomes equal the pairs (every scored tick had a fed read before
it); perception frames 294 / 299 / 299 / 298 / 298 / 292 / 293 / 296. Readability (a) MET on every seed: q95 0.5777 /
0.5715 / 0.5954 / 0.5784 / 0.5721 / 0.5770 / 0.5865 / 0.6148 against 0.75; the minority class ran 18-57 of ~275.

*AMENDMENT, appended 2026-09-15 after the panel (the pairing lag; REPORTED beside, never barred; no re-pin, no
conjunct changes). The record reported "0 skipped" and no lag distribution. Re-recorded on the campaign's own
recording path (`t077_record_brain`, INTACT arm, seeds 0-7, scratch home, 305 s; the instrument validated by
reproducing every pinned intact auroc2 exactly and seed 0's lag5 0.3473): (a) the prospective pairing's TICK lag on
seed 0 is 247 pairs at lag 1, 11 at 13, 10 at 14, 2 at 15 -- exactly 23 of 270 (8.5 %) above 1, and exactly 23 on
every one of the 8 seeds (pooled 2,206 pairs: median 1, mean 2.05, max 15); (b) why: the monitor is fed on wake
ticks only (seed 0: 294 awake / 306 asleep of 600), so a sleep bout between two fed ticks makes the tick gap 13-15;
(c) the CONTROL that makes it nominal, not stale: on 270/270 pairs of every seed the paired value is bit-identical
to `mon.reliability()` as it stands ENTERING the scoring tick (0 of the non-fed ticks move the read;
`_update_metacog` has one call site, inside `Brain._wake_tick`), so the pairing uses the freshest read that predates
the outcome and anything fresher would be lookahead; (d) the lag-1-subset reading, reported beside: auroc2 0.3927 /
0.6846 / 0.6132 / 0.4636 / 0.5336 / 0.6967 / 0.4185 / 0.4919, (b1) t +0.8930 and (b2) t +0.8851 against 1.895 --
VERDICT UNCHANGED, still NOT MET, still below chance on seeds 0, 3, 6, 7 (a post-hoc cut is not the reading and
moves toward the claim the row refused, which is why it is reported and not read); (e) 1.4's trailing clause "the
pairing lag is ONE tick" holds only WITHIN a wake bout; it was already superseded by 2.0 point 3 and 2.2 PAIRING,
which is what the campaign implements. 1.4 stays as frozen; this paragraph corrects it.*

### 3.3 The readings, conjunct by conjunct (every number from the pin; the bar 1.895 over 8 seeds; FPR = the sign-flip rate)

| conjunct | statistic | per seed (0..7) | t | FPR | reads |
|---|---|---|---|---|---|
| (a) readable | q95 <= 0.75, both classes | q95 0.5777 0.5715 0.5954 0.5784 0.5721 0.5770 0.5865 0.6148 | -- | -- | **MET** on 8/8 |
| (b1) tracking | auroc2 - 0.5 | auroc2 0.3828 0.6694 0.6092 0.4571 0.5364 0.6775 0.3907 0.4477 | +0.5049 | 0.0430 | **NOT MET** (above its own q95 on seeds 1, 2, 5; BELOW CHANCE on 0, 3, 6, 7) |
| (b2) not a marker | auroc2 - shuffle median | shuffle median 0.4986 0.5015 0.4999 0.4986 0.5016 0.4994 0.4976 0.4977 | +0.5246 | 0.0430 | **NOT MET** |
| (b3) calibration collapses under the lesion | auroc2_intact - auroc2_lesioned | lesioned 0.5932 0.5000 0.4529 0.4907 0.3714 0.5226 0.7030 0.3352 | +0.3736 | 0.0508 | **NOT MET** (the lesioned read tracks BETTER than the intact one on seeds 0, 3, 6) |
| (b3o) passthrough, reported | auroc2_intact - auroc2_passthrough | identical to (b3) on every seed (3.5 item 16) | +0.3736 | 0.0508 | reported |
| (b4) first-order preserved (TOST) | acc_intact - acc_lesioned, Delta = 0.0465 | 0.0 on every seed (acc 0.8296 0.7942 0.9124 0.8387 0.8100 0.8351 0.8773 0.9355 on BOTH arms) | 0.0 / 0.0 (degenerate) | 0.0 | **MET, DEGENERATE** (3.5 item 2) |
| (b5) not blind | oracle = 1.0 and > q95 | 1.0 on every seed | -- | -- | **MET** |
| (b6) competence over persistence | acc_intact - persistence | persistence 0.9481 0.9495 0.9380 0.8817 0.8781 0.9355 0.9480 0.9892 | **-5.2502** | 0.0469 | **NOT MET** on 8/8: the readout is below "repeat the previous category" |
| (c) R0 | exact replay of the live recording | held on every seed | -- | -- | **MET** |

> (a) SENSITIVITY OF THE DECLARED rho, computed 2026-09-15 after the panel from the pin's own q95s (no re-run):
> 2.0 point 1 computed one consequence of rho = 0.25 (q95 under 0.75 from about 4 minority outcomes) and not the
> load-bearing one. Over the two BARRED arms the largest q95 is 0.6148 (seed 7, intact; seed 7 lesioned 0.6112),
> so (a) is MET for every rho >= 0.11484 and 2.3's "any seed unreadable -> the campaign is REFUSED" would fire for
> any rho below it: the declared rho clears its own refusal threshold by 2.18x, seed 7 binding. At rho = 0.10 seed 7
> refuses on both barred arms; at rho = 0.05 all eight intact streams refuse. The perception stream is REPORTED,
> never barred (2.3), and cannot refuse the campaign; at the declared rho it passes on every seed (q95
> 0.6050-0.6161) and would refuse on all eight below rho = 0.1161. rho = 0.25 was frozen in the pre-registration at
> `47c41aa` (09:47:05) and in code at `48750eb` (09:51:16), both before the campaign entry `07fd69a` (10:06:11) and
> the run at 10:06:23 -- the constant was declared, not moved; what was missing was this number, not the disclosure.
> Five of eight intact readings sit inside their own null band (seeds 0, 3, 4, 6, 7 below their q95; 1, 2, 5
> above), which 3.3's (b1) row already prints; (a) certifies that a stream COULD resolve an effect of rho, not
> that the effect read is resolved -- the t over 8 seeds is the pre-registered reading and is not re-cut here.

**VERDICT: NOT MET** -- (b1), (b2), (b3) and (b6) NOT MET; (a), (b4), (b5), (c) MET. The instrument is not blind (b5),
the streams are readable (a), the replay is exact (c); the CLAIM -- that the monitor's read tracks the brain's own
error rate better than a fixed prior, on held-out items -- is not established on this stream: the paired t sits at a
quarter of the bar, and on four of eight seeds the read is anti-correlated with correctness.

REPORTED beside (never barred). Per-input attribution by exact replay (delta auroc2 with the input pinned to its
stream mean): `conflict` +0.2476 / -0.2354 / -0.0165 / +0.0375 / -0.1192 / -0.2472 / +0.2646 / +0.0055 -- the ACC
conflict term is what moves the read's relation to correctness, in BOTH directions across seeds; `surprise` -0.0525 /
+0.0443 / -0.0351 / +0.0309 / +0.0135 / +0.0140 / -0.1171 / -0.0579; `learning_progress` -0.0316 / +0.0048 / +0.0055 /
+0.0056 / +0.0195 / +0.0029 / -0.0153 / +0.0276; `confidence` +0.0004 / 0.0 / +0.0003 / +0.0003 / +0.0017 /
+0.0007 / +0.0003 / 0.0 -- a live but near-null channel on this stream (the nonzero deltas are themselves proof the
pinned replay changed reads; nothing here says "no spike moved"); `novelty` 0.0 on all 8 seeds STRUCTURALLY, not
empirically: `MetacognitiveMonitor.update` stores it as `self._nov` and its `drive` expression contains no novelty
term, so the replayed read is independent of it by construction and this neuter cannot return a nonzero number on
any input -- the read-side attribution has FOUR live channels, not five (the two gates that read `_nov`, `explore`
and `grow_trigger`, are not readouts here; 3.4 i and opening 3; corrected 2026-09-15 after the panel). The +5-tick
lag control (five FED reads back, 5-19 ticks on this run -- 3.5 item 19) reads 0.3473 / 0.6440 / 0.5707 / 0.5134 / 0.5971 / 0.5832 / 0.3645 /
0.4656 -- the same range as the prospective pairing, so whatever relation exists is not specific to the tick before
the score. The perception stream (the vision classifier's per-frame hits, same-tick) reads auroc2 0.4270 / 0.3885 /
0.5412 / 0.4036 / 0.4305 / 0.4749 / 0.4831 / 0.4638 -- no tracking there either. `brier_gain` is negative on every seed
(-0.3261 / -0.3197 / -0.4729 / -0.3966 / -0.3392 / -0.2804 / -0.3843 / -0.5981): the read is not on a probability scale.
ARM D (2.2): `_second_order_hold` True on 39/40 turns on seed 0 and 40/40 on seeds 1-7; conf2 mean 0.0084 (max 0.336)
and 0.0076 (max 0.276) on seeds 0 and 5, 0.0000 with max 0.0 on the other six; the pin-to-mean neuter changed 0/40
routes on every seed; the FLIP neuter (hold False) changed 21 / 21 / 22 / 22 / 19 / 23 / 21 / 21 of 40 routes -- the
constant second-order hold decides about half of the chat path's routes on every seed.

### 3.4 The disposition applied (2.6, the NULL branch: nothing of the incumbents is deleted)

(a), (b5) and (c) MET: the instrument works -- its reds fired, its positive control reads 1.0 above every band, its
replay is exact. (b1), (b2), (b3), (b6) NOT MET: the claim about the layer does not hold on this stream. By 2.6, fixed
before any number: the reading is recorded NOT MET with every conjunct's number (3.3); `CONCLUSION["metacognition"]`'s
`reading` stays None (the surface symbol declared at `c12a91c` is NOT 2.6's -- 2.6's NULL branch says no symbol is
declared; see 3.5 item 9); the C5 licensed sentence is NOT written anywhere; the cockpit's `reliability` row and its
PROVENANCE gain a dated line naming what was measured (the fused spike-count of self-model signals; NOT an
error-tracking confidence on the readout's stream: t +0.50 against 1.895, four seeds below chance); NOTHING of the
incumbents is deleted or re-labelled beyond that line (2.0 points 2 and e). The instrument STAYS: `track_error`,
`replay_reliability`, the campaign entry and the pin are the record's apparatus, reachable and red-tested.

Two facts the disposition did not need but the record must carry: (i) the LESION CHANGED NOTHING in the first-order
stream -- accuracy identical to four decimals on both arms of every seed, so the lesioned auroc2 equals the offline
passthrough exactly; (b4) is MET by degeneracy, not by tolerance, and the monitor's gates (learn / explore / grow /
hold) are shown NOT to touch the readout's tick over 600 ticks (the monitor's own record already read the learn
consumer inert; this measures the whole gate set on the perception/readout path); (ii) (b6) is a finding about the
READOUT, not the monitor: on every seed the prequential readout scores BELOW the trivial persistence predictor
(0.79-0.94 against 0.88-0.99), and the daemon's own cockpit numbers at the time of writing read the same way
(live_accuracy 0.932 against live_persistence 0.994 over 486 items). That is L1c's online readout on the scratch
brain's stream failing a competence floor this row is the first to compute; it is carried as a named opening (3.6),
not adjudicated here.

### 3.5 Deviations from 2.1-2.10, each with its number (dated 2026-09-15)

1. The paired t was lifted out of the EXEMPT harness (2.10) after the dormancy gate read five stale rows on the lazy
   import 2.1 had believed safe -- an apparatus fact measured on the first campaign commit; no conjunct changed.
2. The LESIONED arm's outcome stream was BYTE-IDENTICAL to the intact arm's on all 8 seeds (acc differences 0.0), so
   (b4)'s TOST contrasts were degenerate (`contrast` records a zero-variance sample as degenerate and MET iff its
   common value is positive; Delta 0.0465 > 0). 2.3 wrote (b4) as two positive contrasts expecting variance; the
   record reads the degenerate branch as the pre-registered rule states it and names it here rather than smoothing it
   into "preserved at the bar".
3. `SecondOrderAccess`'s self-check assertion "calibration must NOT be perfect" could not survive the statistic change
   (5261eb0): the rank-based read on its constructed stream is exactly 1.000; the honest-limit assertion is now on
   the CALLS (`errors > 0`). Disclosed in that commit and in the SYNTHESIS S6 row.
4. The first draft of `substrate/paired_contrast`'s self-check asserted an FPR of 1/256 from expectation; the exact
   enumeration reads 13/256 for eight same-sign differences; the assertion is now the enumeration (14807ef).
5. The `learn_gate` "measured learning outcome" licence the first draft of 2.6 relied on was FALSE by the monitor's own
   record (2.0 point e); nothing in the applied disposition rests on it.
6. R0's red moved from "an input perturbed by 1e-9" to "a recorded read perturbed by 1e-9" before the freeze (2.4): the
   read is spike-quantized at 1/384 and the first form would have passed vacuously.
7. No `metacog_lesion` kwarg was threaded through `make_brain` (the draft's 2.2): the campaign sets the instance's
   `lesion` attribute and replaces its `reliability()`; brain.py is untouched by this row.
8. (appended 2026-09-15, before the panel) TWO STATED TIMES IN THIS FILE ARE WRONG AND GIT IS THE INSTRUMENT: the header
   says sections 1-2 were written "05:10-10:20 local" and 2.10's heading says "~11:10 local"; `git log --format=%ci`
   records the freeze `47c41aa` at 09:47:05 -0500, 2.10's commit `14807ef` at 10:05:47, the campaign entry `07fd69a` at
   10:06:11, the campaign's own start at 10:06:23 (its log) and the pin `d4e86f6` at 10:27:08. The stated times were the
   session's estimates written without reading a clock (obs #2171's shape: an amendment's stated time owes an instrument
   git cannot supply, and here git supplies the better one). What the times were meant to establish -- that 2.10 was
   written BEFORE any campaign number -- git establishes on its own: `14807ef` precedes `07fd69a` precedes the run that
   produced `d4e86f6`. The headings are left as written; this item corrects them.
9. (appended 2026-09-15, after the panel; BLOCKER C2) THE SURFACE SYMBOL IS A POST-NUMBER DEVIATION FROM 2.6, AND 3.4
   CALLED IT COMPLIANCE. Frozen 2.6's NULL branch reads, verbatim: "`reading` stays None and no surface symbol is
   declared". What was done: `c12a91c` (10:33:56, after the pin `d4e86f6` at 10:27:08) declared
   `CONCLUSION["metacognition"].symbols = ("metacog_tracking",)` and `f27f436` (10:34:04) marked tasks.md `[X]`. The
   cause, named without excuse: 2.6's NULL branch and 1.5's gate rule (a `[X]` C4 task with no symbol is a red,
   conclusion_protocol's arming arm) cannot both hold once the row is `[X]`, the pre-registration did not reconcile
   them, and the session chose to keep the arm ARMED rather than green it by emptying the symbol -- after the number,
   and written into 3.4 (now corrected in place) as "By 2.6". The invariant that makes the choice harmless but not
   pre-registered: nothing prints the symbol, no reading exists, and refusal 3 polices the name if it ever appears.
   The registry value STAYS (the panel's refuter measured that emptying it reds the gate's own positive control);
   a coupled red the arc missed is fixed in the fold: `tests/test_ratified_row_gates.py` asserted
   `_task_done("T077") is False` and was RED from `f27f436` on (re-pointed at T078, `e7dbb4d`). No pre-registered
   text is edited and no 2.11 is written: a post-number amendment to a disposition is the thing being refused.
10. (appended 2026-09-15, after the panel; BLOCKER C1) R2 AND R5 WERE f(x)==f(x) CHECKS, AND 3.1's SENTENCE IS
   WITHDRAWN. As committed at `654106b` and greened at `48750eb`, R2 scored a COPY of the first-order stream and
   asserted equality with the original, and R5 scored one stream twice; the refuter ran both bodies against a
   constant stub, a wrong-signed scorer and a pure-relay stub that discards the confidence stream -- all passed --
   and against a 24-stream sweep of the constructed family: no input reddens them. Neither called `contrast`, and
   the `654106b` red was a collection `ImportError`, so neither ran on its constructed input in the red state. 3.1's
   "Every red of 2.4 fired on its constructed input before the campaign" is FALSE for R2 and R5 and is withdrawn,
   not repaired; R0, R1, R3 and R4 stand. Rebuilt at `475fbe4` (after the campaign) in the refuter's minimal form on
   the campaign's own `contrast` over eight constructed seeds: the relay arm and the no-collapse twin read degenerate
   at exactly 0.0 and NOT MET, the non-relay control and the collapsing twin read MET, and two deliberately mutated
   scorers (a drifting one, a confidence-blind one) redden each in fact, in the test and in the module. 2.4's "and
   a line of the module's --self-check" was ALSO not carried until `475fbe4` (`t077_reds`, six booleans asserted by
   main()). The verdict, the pin and the disposition are unaffected: (b3) read NOT MET in fact (+0.3736), so no
   positive was manufactured. 2.4, PLAN-FINAL C-R6 and the `654106b` message stay as written.
11. (appended 2026-09-15, after the panel; C4 / C8) THE ONE SURFACE SENTENCE 2.6 AUTHORIZED ASSERTED A CHARGE NOBODY
   MEASURED, AND THE INERTNESS RECORD IT WOULD HAVE BEEN CORRECTED TO IS ITSELF FALSE. The PROVENANCE line landed at
   `c12a91c` read "It gates learning and the chat abstention"; the learn half asserts charge on the one surface whose
   job is the reach/charge distinction (P3). Two lenses filed it; their refuters disagreed on the replacement, and the
   tree settles it: the monitor's own record (:96-113, carried into 2.0 point e, 2.7 and item 5 above) -- "the
   lifelong-learner branch `learn_gate()` gates at cortex/brain.py never fires in this configuration, in ANY arm" --
   is FALSE at this head. Re-probed on exactly that configuration (`scratchpad/t077/probe_learn_gate.py`:
   make_brain(seed=0 and 1, persist=False), scratch home, 300 ticks, natural clock): the branch gated at
   `Brain._enter_sleep` fires on 3 of 3 lifelong tasks on both seeds (_ll_i 3, _ll_units 6, _ll_retention 1.0,
   13 / 12 sleep entries). The 2026-08-05 A/B measured only that the lifelong counters did not DIFFER between arms --
   equally consistent with the branch firing identically in both, which it does: the no-difference-means-inert
   inference this row's own openings warn about. What the read does NOT do there, measured the same day
   (`scratchpad/t077-panel/fold_gates_and_learn.py`): pinned to a constant 0.0 for 300 ticks the learner still
   learns all 3 tasks with the same units and retention (the branch compares the gate against an EMA written from
   that same gate one line earlier, so any constant passes); a contrived monotonically decaying gate blocks it
   (_ll_i 1, _ll_units 4). Consumed, and not decisive at any constant value. The surface clause is corrected at
   `57ec145` to "It decides the chat abstention route (the monitor's own per-item attribution, 2026-08-05: 3 of 30
   sampled replies); it is READ by the lifelong learn gate in cortex/brain.py::Brain._enter_sleep -- consumed there,
   NOT shown decisive"; the docstring record is amended by a dated paragraph at `233a728`; 2.0 point e, 2.7 and item
   5 stay as written and this item corrects them. No number moves; 2.6's null branch is untouched.
12. (appended 2026-09-15, after the panel; C5 / C12) ONE CLAUSE OF 2.10 DID NOT LAND UNTIL THE FOLD. 2.10's WHAT
   CHANGES states "the monitor's `T_BAR` restatement is replaced by the import"; `git log -L` shows the literal at
   metacognitive_monitor.py entered at `48750eb` and `14807ef` never touched it, so at the pin and at `9293a7f` the
   live CORE module still declared its own `T_BAR = 1.895` beside `substrate/paired_contrast.py`'s -- the P2
   duplicate 2.10 was written to end, in a live module, and items 1-8 did not name it. Every OTHER sub-change of 2.10
   had landed (the five names in `paired_contrast`, its REGIONS and `run_selfchecks` registrations, the harness's
   re-export, the campaign's import of `contrast`). Landed at `9e4ab1b`: the literal deleted, the name imported at
   module level (no cycle, no E402, no new dormancy edge); the block comments that still described the pre-2.10 tree
   (a lazy import of the EXEMPT harness, and the false claim that a lazy import escapes the T070 trap) corrected at
   `d03b934`. No number moves -- the two literals were equal and the equality red asserted it on every run since
   `654106b`; `test_the_bar_is_the_plan_s_bar` is untouched and its message's "restates" now reads stale by one
   word, disclosed here rather than edited into a committed red.
13. (appended 2026-09-15, after the panel; C6) THE RE-POINTED CALIBRATION IS CIRCULAR ON ITS CONSTRUCTED STREAM AND
   ITS BINDING CONTROL WAS MISSING. `SecondOrderAccess`'s ground truth, `truly = ignited and availability >=
   avail_majority`, is a deterministic function of two of the four values `observe()` is handed; recomputed and
   scored as the confidence it reads auroc2 1.000 on the self-check's stream, and availability alone reads 1.000 --
   the monitor TIES that rival, it does not beat it. None of the four controls of 2.5 was that rival (shuffled 0.500,
   first-order echo 0.641, fixed prior 0.500, ignition echo 0.864), so "calibration 1.000 beats ignition-echo 0.864"
   (3.1, item 3, SYNTHESIS S6) survived only because the ceiling was absent. Not P8 clause 2 (2.5 holds: no control
   reads the pool's output as its own input); the mirror defect, a required CEILING control missing. Landed at
   `b84d6bc` in the refuter's form: CONTROL 5 beside the four, printed in the (c) block, asserted as the TIE the data
   supports (a monitor below the recomputed target is the red), never as the monitor beating its own inputs; the
   circularity named in the module docstring and in `calibration()`'s docstring with the number 1.000, never 0.864;
   the price measured and written where a future wirer sees it -- 2,000 permutations over the full history, each an
   O(n log n) rank pass, 34 / 64 / 119 / 246 ms at n = 50 / 100 / 200 / 400 on this host, from a former O(1) read;
   `_hist` and the 2,000 draws left alone (truncating either changes the statistic silently). The retired
   statistic's write-only accumulator `_agree` deleted at `0a3d443` (P2 residue of `5261eb0`). The score's lack of a
   live reader is opening (5) in 3.6.
14. (appended 2026-09-15, after the panel; C7) ITEM 8's REPAIR ARGUMENT IS WITHDRAWN; THE FILE EVIDENCE REPLACES IT.
   Item 8 argued that git establishes 2.10-before-any-campaign-number because `14807ef` precedes `07fd69a` precedes
   the run that produced `d4e86f6`. The inference is invalid and this tree refutes it: `git show
   14807ef:src/snn_data_lake/cortex/metacognitive_monitor.py` contains no `run_t077_campaign` (0 hits), yet
   `scratchpad/t077/dry_run_module_pin.json` (mtime 10:05:35.771) was written by `mm.run_t077_campaign(seeds=(0,),
   n_ticks=80, pin=...)` off `src/` twelve seconds BEFORE `14807ef` (10:05:47). The campaign entry was run from the
   UNCOMMITTED working tree, so commit order cannot bound number order -- git was the wrong instrument for the
   quantity it was asked to resolve (obs #2171's shape, applied to item 8 itself). What was actually seen, and when:
   three 80-tick / one-seed apparatus dry runs preceded the campaign -- one earlier than either surviving pin (2.8's
   "33 s" was already in the 09:42:23 draft of this file and matches neither surviving pin, so that run's pin was
   overwritten), `dry_run_pin.json` at 09:45:00.825 (two minutes BEFORE the freeze `47c41aa` at 09:47:05; elapsed
   44.9 s), and `dry_run_module_pin.json` at 10:05:35.771 (elapsed 36.6 s). Each carries a per-seed block (14 pairs,
   the five stream dicts with their refusals, acc 1.0000 / 1.0000, persistence, the arm-D counts, the raw streams)
   and the verdict `{"a_readable": false}`. Therefore 2.10's heading "BEFORE any campaign number exists" and this
   file's section-2 header "(no campaign number has been seen)" are false as literally worded -- true only of the
   REGISTERED campaign (600 ticks x 8 seeds x 2 arms, started 10:06:23). 2.8 and 3.1 did disclose a dry run, so
   nothing was concealed, but 2.8 names it without its script and so fails the header's own rule. Per this file's
   convention the frozen headings are not edited; this item corrects them. What does not move: no conjunct, bar,
   control or disposition -- the dry runs REFUSED on (a) ("one outcome class only", acc 1.0000) and previewed no
   reading; section 3's NOT MET stands as recorded.
15. (appended 2026-09-15, after the panel; C10) ARM D's `pin_to_mean_routes_changed: 0` IS ENTAILED ON ALL EIGHT SEEDS,
   NOT MEASURED; 3.3's "changed 0/40 routes on every seed" is ONE datum, not eight. `SecondOrderAccess.observe` has
   one call site, inside `Brain.respond`, and `t077_arm_d` samples `_conf2` immediately after each `respond`, so the
   40 recorded values are EVERY observe call of the run. On seeds 1, 2, 3, 4, 6, 7 every call reads exactly 0.0
   (3.3's own "max 0.0"), so pinning the mean 0.0 is the identity -- re-run on seed 3 under a scratch home
   (`scratchpad/t077-panel/fold_armd.py`): the intact and pinned output tuples are byte-identical. On seed 5 the
   pinned values (0.0 / 0.026 / 0.276, from the pin) and the pinned mean 0.0076 all lie below `c_hold` 0.30, so
   `hold()` -- the only quantity the chat route gate reads -- cannot differ; its 0/40 is entailed by the threshold.
   On seed 0 the pin can move the hold on exactly ONE turn of 320 (turn 5, conf2 0.3359 against the pinned 0.0084);
   re-run, both arms read `_metacog_hold` True on that turn and the same route string, "Gf (metacognitive hold:
   abstain)" -- decided by the metacognitive hold, not the second-order one. The pin-to-mean count therefore carries
   no information about the second-order pool's charge on this path, and the successor P2 arc must not read it as
   evidence of inertness; the arm-D charge finding rests entirely on the FLIP neuter (21 / 21 / 22 / 22 / 19 / 23 /
   21 / 21 of 40), non-degenerate on every seed and unaffected by this item. 3.3 printed the precondition beside the
   count and 3.6 (2) already names the pool "silent by construction"; this item supplies the inference those numbers
   entail. SYNTHESIS S6 carries the same qualifier (`a5eff15`). Nothing above section 3 is edited.
16. (appended 2026-09-15, after the panel; C11) THE ENTAILING CAUSE OF (b3) == (b3o), FROM THE PIN: on all 8 seeds
   BOTH streams are byte-identical -- `streams.lesioned_read[i] == streams.first_order[i]` and
   `streams.lesioned_outcome[i] == streams.outcome[i]` for every i -- so `per_seed[s]["lesioned"] ==
   per_seed[s]["passthrough"]` as whole dicts, and (b3)'s diffs, t (+0.3736) and FPR (0.0508) ARE (b3o)'s. 3.4 (i)
   and item 2 gave only "accuracy identical to four decimals" / "(acc differences 0.0)", which fixes neither
   sequence and does not entail equal auroc2 (equal accuracy admits auroc2 0.0 and 1.0 on the same confidence
   stream); the read stream's identity is the missing conjunct, and the barred conjunct's cause was asserted, not
   shown. The consequence, with no conjunct changed: (b3) STAYS NOT MET exactly as pre-registered (2.6's REFUSED
   branch covers (a), (b5), (c) only and is not invoked); what changes is its interpretation -- the lesion did not
   perturb the recorded trajectory on any tick of any seed, so the in-vivo arm produced no information the intact
   recording did not already contain, and spec.md's in-vivo dissociation ("first-order preserved while calibration
   collapses") was NOT EXERCISED on this run. 3.3's (b3) gloss "the lesioned read tracks BETTER than the intact one on
   seeds 0, 3, 6" is a statement about the offline passthrough, not about a lesion. This is 3.6 opening (3) stated
   where (b3) is read. For the successor, a CHARGE assert and not a stream-divergence assert: before any in-vivo (b3)
   is read, prove the second-order read is CONSUMED on the scored tick (perturbing it moves a downstream quantity
   before the outcome is scored); do NOT adopt `lesioned_read != first_order` -- under this lesion design that
   quantity is the first-order input by construction, so the assert would demand the first-order stream DIVERGE,
   refusing the very result the spec asks for and pulling against (b4). 3.4 (i) and item 2 stay as written.
17. (appended 2026-09-15, after the panel; C13) SEED 1's FED `confidence` IS IDENTICALLY 0.0 ON ALL 277 SCORED PAIRS
   (every other seed: 100-167 distinct values in [0, 0.019]). That seed's LESIONED and PASSTHROUGH arms therefore
   score a CONSTANT stream: their auroc2 = 0.5 is the fixed prior's exact value (2.1; `second_order_access` CONTROL 3
   asserts the same number), not a measured discrimination, and their permutation null has ZERO width (q95 0.5,
   every draw 0.5). (a) is MET as pre-registered and the null is not confoundable in the direction (a) guards --
   chance cannot manufacture a large auroc2 on a constant stream -- so the disposition stays NULL; this item does not
   re-read 2.6 (the refuter refuted the filed REFUSED consequence, and the filed fix -- a refusal on a constant
   stream -- crashes CONTROL 3). Two consequences the record must carry: (i) seed 1's (b3) diff equals its (b1) diff
   (0.169418) BECAUSE that arm reads exactly 0.5 -- a one-seed coincidence, not a shared conjunct (checked on all 8:
   seed 1 only; (b1) never reads the lesioned arm); (ii) dropping seed 1 from (b3) moves t from +0.3736 to +0.0621
   against 1.895 -- NOT MET either way, and seed 1 was a POSITIVE contribution, so this disclosure makes (b3) more
   null, not less. 3.3's (a) row prints the eight INTACT q95s; the gate is intact AND lesioned, and the lesioned q95s
   are 0.5750 / 0.5000 / 0.5996 / 0.5752 / 0.5711 / 0.5773 / 0.5943 / 0.6112. `track_error` now emits
   `conf_distinct` on every stream and a red beside R1 covers the confidence axis (`e544e59`); the constant-stream
   reading itself is unchanged. Opening (6) in 3.6 carries the seed.
18. (appended 2026-09-15, after the panel; C14) (b2)'s NULL IS ANALYTIC, NOT MEASURED. With average ranks the
   label-permutation null of auroc2 has mean exactly 0.5 in closed form (the ranks sum to n(n+1)/2, so E[auroc2] =
   0.500000000000; computed per seed from the pin, all 8). `shuffled_auroc2` is therefore a 2,000-draw Monte-Carlo
   estimate of the constant 0.5: the pinned medians run 0.4976-0.5016, and 60 independent 2,000-draw re-estimates on
   seed 6's pinned streams give mean 0.4997, sd 0.0015 (the pinned -0.0024 is 1.6 sd). (b2)'s per-seed differences
   depart from (b1)'s by at most 0.002375, and (b2) recomputed against the exact 0.5 reads t +0.5049, which IS (b1).
   As a BARRED conjunct (b2) adds no evidence independent of (b1) and cannot disagree with it; its non-redundant half
   is its second clause, auroc2 above its own per-seed q95 (3/8: seeds 1, 2, 5, in 3.3). 2.3 stands as frozen, (b2)
   stays a conjunct, the verdict is unchanged -- NOT MET with or without it; the row's shuffle control with teeth is
   R3 (2.4), a constructed-tracker red that ran red first. Said at the site too (`d03b934`).
19. (appended 2026-09-15, after the panel; C17) THE "+5-TICK LAG CONTROL" SHIFTS BY 5 FED READS, NOT 5 TICKS.
   `t077_pairs_prospective`'s `gap` indexes fed reads, and the monitor is fed on awake ticks only, so the control's
   realised shift on the run -- re-recorded on the campaign's own path, the instrument validated by reproducing the
   pin's seed-0 lag5 0.3473 and lag1 0.3828 exactly -- is, pooled over 2,206 gap-5 pairs on seeds 0-7: min 5, median
   5, mean 10.20, max 19 ticks, with 41.4 % of pairs above 5 ticks; the intact pairing's own lag is median 1, mean
   2.05, max 15 (3.2's amendment), so 1.4's "ONE tick" is the median, not the bound. The quantity named in 2.2, 2.3
   and 3.3 is not the quantity measured (obs #2105: a metric name is not what it measures). The lag control is
   REPORTED and never barred (2.3, 2.5); no conjunct or verdict moves, and 3.3's conclusion survives a fortiori
   because the realised control reaches FURTHER back than its name claims. 2.2, 2.3 and PLAN-FINAL C-R6 stay as
   written; the docstring and the call site say "fed reads, not ticks" (`d03b934`); the pin is not re-run.
20. (appended 2026-09-15, after the panel; C18) THE SIX `scratchpad/t077/` CITATIONS CANNOT BE RE-OPENED FROM THE
   TREE. `scratchpad/` is gitignored (.gitignore:170); `git ls-files scratchpad` returns nothing, and none of
   design.md, codex_design_read.md, probe_ref_strength.py, probe_armD_cost.log, check_impl.py or probe_perr.py exists
   at this head (they exist only in the session's temp scratchpad). Where each cited fact survives in the tracked
   record: 1.7 `design.md` -- transcription only; every governing word carries its own tracked site in the same
   sentence. 2.0 `codex_design_read.md` -- the seven points are reproduced in 2.0 itself; no archive exists in this
   tree, so the Codex provenance (09:32 / 09:37) is this file's own statement, not independently checkable. 2.0
   `probe_ref_strength.py` -- the saturation stands in tracked source: `SecondOrderAccess`'s self-check reads
   calibration 1.000 (3.1, item 3, now item 13's ceiling); run it to reproduce. 2.2 `probe_armD_cost.log` -- the
   seed-0 read reproduces EXACTLY from the pin's `per_seed.0.arm_d` (turns 40, so_hold_true 39, conf2_mean
   0.0083984375, pin_to_mean_routes_changed 0, holds_changed 1). 2.4 `check_impl.py` -- the quantization stands in
   tracked source (`pop_size=24`; the `replay_reliability` docstring's "24 x 16 steps", 1/384). 2.8 `probe_perr.py`
   -- the N = 600 column reproduces from the pin (seed 0 n_pairs 270 / minority 46; seed 1 277 / 57); the 300 / 900 /
   1,200 columns and the "first 300 ticks of seed 0 carry no error" argument built on them are UNRECOVERABLE. 2.8's
   62.7 ms/tick is unrecoverable and the projection it carried was NOT met: 2.8 projected "10 min sequential" for
   600 ticks x 8 seeds x 2 arms; the pin records `elapsed_s = 1139.7` (19.0 min), 1.9x the projection; 2.8's refusal
   rule (re-size above 3 h) was not tripped, so no conjunct or bar is affected. No barred conjunct of 2.3 and no
   number in 3.3 / 3.4 depends on any of these probes -- every reading is read from the pin. Probes checkable on
   FUTURE rows is a forward-only convention for a later commit, never framed as recovering T077's.
21. (appended 2026-09-15, after the panel; C19) THE FOUR-GATE NULL OF 3.4 (i) AND 3.6 (3) IS CHECKABLE FROM THE PIN;
   ITS POSITIVE CONTROL IS NOT. The null itself the pin carries: `acc_intact == acc_lesioned` to six places and
   `streams.outcome == streams.lesioned_outcome` exactly, on all 8 seeds at 600 ticks. What the pin does NOT carry is
   the evidence that the lesion MOVED the gates: `t077_record_brain` records `_metacog_hold` and `learn_gate()` per
   tick (2.2's registered list) and the campaign discards both -- neither reaches `per_seed`, the verdict or the
   printed log (the pinned `arm_d.so_hold_true` is `_second_order_hold` on the CHAT path, a different quantity) --
   and `explore()` / `grow_trigger()` are recorded nowhere. The control, re-derived OUT of the campaign at 150 ticks
   on seed 0 under a scratch home (`scratchpad/t077-panel/fold_gates_and_learn.py`; the pin untouched): hold True
   16/150 intact vs 0/150 lesioned; `_metacog_explore` 147/150 vs 0/150; `grow_trigger()` 7/150 vs 9/150;
   `learn_gate()` 57 distinct values vs the single 1.0 -- all four gates fire and all four differ under the lesion,
   while the 49 scored outcomes are identical (all correct on both arms at that width). The four-gate scope is the
   pre-registration's own (2.2: `lesion = True` makes every gate a no-op), so 3.4 (i) and opening (3) stand as
   written; this item supplies the control they leaned on without pinning. Putting the four counts INTO the per-seed
   pin would mean hand-deleting a write-once artifact and re-running the campaign after its verdict -- that belongs
   to the successor row 3.6 (3) names, whose pre-registration should pin the four gate counts for both arms.
22. (appended 2026-09-15, after the panel; C20) THREE NUMBERS IN THIS FILE OWE AN INSTRUMENT; NONE TOUCHES A
   CONJUNCT. (a) 2.8's "A 1,200-tick apparatus dry run (80 ticks, one seed, pin redirected) ran end to end in 33 s"
   names no script (this file's own header rule) and matches no preserved log: "1,200-tick" is a paste from the
   preceding probe_perr sentence; the dry run was 80 ticks, one seed, as its parenthetical says; the preserved
   PRE-freeze run logged 45 s (`dry_run.log`, 09:45:00; pin elapsed_s 44.9) and the POST-freeze module run 37 s
   (`dry_run_module.log`, 10:05:35; elapsed_s 36.6); the 33 s run's log was overwritten, and the sentence was written
   at 09:42:23 before either preserved log existed. Its own words hold -- "its numbers are not readings". (b) 3.1's
   "R0 also held on every live recording" is scoped wrong: R0 is pre-registered in 2.3 (c) on INTACT's recorded
   inputs, and `run_t077_campaign` makes the module's only `expect=` call on the intact rows; the lesioned recording
   is scored with no `expect=`. Read it as "R0 held on every INTACT live recording, 8 of 8". The lesioned arm's
   consumer-visible read is `reliability()` replaced by the first-order passthrough, so R0 as written has no
   lesioned counterpart to run; what the pin does show, unasserted, is item 16's byte-identity. (c) 3.4 (ii)'s
   daemon numbers are NOT "at the time of writing": live_accuracy 0.932 over 486 items is verbatim 1.4's poll of
   00:10, 10.4 h before the record commit `09892f2` (10:33:51), and live_persistence 0.994 occurs exactly once in
   this file with no poll, command or time anywhere in the row's artifacts -- WITHDRAWN as unsourced. A FRESH dated
   poll, 2026-09-15 11:22:16 -0500, `GET http://127.0.0.1:8137/api/all` -> `cognition.neurogenesis`: live_accuracy
   0.99776, live_persistence 0.99916, live_n 3570, baseline_persistence 1.0. A different daemon lifetime (the
   counters are process-lifetime by the cockpit's own PROVENANCE), so the withdrawn pair is unrecoverable, not merely
   uncited. The DIRECTION 3.4 (ii) and 3.6 (1) carry -- the readout below persistence on the daemon -- holds at this
   poll, at a far smaller gap. That is what tasks.md's "and on the daemon" now rests on (its citation changed).
23. (appended 2026-09-15, after the panel; C22) CONJUNCT (c) AND THE PER-INPUT ATTRIBUTION ARE ASSERTED IN-RUN AND
   ARE NOT RE-DERIVABLE FROM THE PIN. `run_t077_campaign` builds the per-tick input stream and the unrounded pool
   read and asserts (c) on them, but writes neither to `tools/t077_metacognition.json`: `per_seed[s]["streams"]`
   carries only read / first_order / outcome / lesioned_read / lesioned_outcome, and the pin contains no occurrence
   of "inputs", "rel" or "replay". Every OTHER barred conjunct re-derives from the pin exactly -- (b1) / (b2) auroc2,
   (b3)'s lesioned auroc2 and (b5)'s oracle recompute to the last digit from the pinned streams (verified in the
   fold) -- so (c) is the one barred conjunct held to a lower record standard than the rest, named here rather than
   left implicit. What DOES establish (c): the assert is unguarded (no try/except in `run_t077_campaign`) and
   strictly upstream of the single pin write, so a complete 8-seed pin entails it did not fire; and its detector has
   a committed positive control (R0, which catches both a 1e-9-perturbed read and a perturbed input). What is not
   independently checkable: the five per-input neuter deltas (P8 clause 3, reported in 3.3), whose values are pinned
   but whose derivation is not -- REPORTED, never barred (2.3). The verdict, the disposition branch and (c)'s MET are
   unchanged. Pinning `inputs` and `rel` is deferred to the successor row that runs a campaign of this shape: the pin
   is written once, `run_t077_campaign` refuses while it exists, and editing the writer now would leave uncharged
   code (P3) and a source whose `streams` shape no longer matches the pin it produced.
24. (appended 2026-09-15, after the panel; C23) 3.4 (ii)'s "a competence floor this row is the first to compute" IS
   FALSE, and the sentence's own numbers say so. The floor is an INCUMBENT on both readings: `live_persistence`
   (cortex/neurogenesis_engine.py, the counter fed on the wake stream) landed 2026-09-05 in `4b77571` under the
   engine's own comment "beside its own live persistence floor" (L1b); the refit-slice floors `baseline_persistence`
   / `baseline_majority` landed 2026-09-02 in `2867ed6`, are printed as cockpit rows ("predict floor: persistence /
   majority") and `PROVENANCE["predict_accuracy"]` (dated 2026-09-02) already states the read-against rule; both are
   firing on the daemon now (the 11:22 poll of item 22: baseline_persistence 1.0, live_persistence 0.99916, live_n
   3570). T077 computes no floor: the campaign's per-tick `persist` is a first difference of the incumbent counter.
   What is new here is the READING ONLY -- the incumbent floor read as a barred conjunct (b6) with the plan's 8-seed
   paired statistic. 3.4 (ii) and 3.6 (1) stay as written; this item corrects them. The two floors are not two
   instruments of one quantity (obs #2106 does not apply): `_heldout_baselines` scores persistence on the refit's
   held-out rows, `live_persist` prequentially over the whole wake stream -- different slices by construction.
25. (appended 2026-09-15, after the panel; C26) THE RE-POINTED SCORER HAS NO LIVE-TICK CONSUMER, and 2.1 does not say
   so. `SecondOrderAccess.calibration()` has three call sites: `state()` and the module's own self-check (twice).
   `state()` is called nowhere in `src/` outside that self-check; `brain.py` uses only `observe()` and `hold()`, as
   section 1 item 3's consumer list already implies without stating. The method is NOT dormant -- `cortex.second_
   order_access` is CORE (`run_selfchecks.py`, run in-process), so `calibration()` executes and is asserted against
   its controls on every gate run (the shape `docs/AUDIT_BACKLOG.md`'s largest class calls a policy
   misapplication: a documented, measured state read as a violation). What the P2 act of 2.9 step 3 therefore removed is a duplicate
   IMPLEMENTATION of the calibration statistic -- which is what P2's own counting rule measures (survivors that are
   still REACHABLE) -- and not a second reading on any live path. "One mechanism survives" is true of the scorer and
   of nothing else; 2.7 already concedes it says nothing about the two pools. SYNTHESIS S6 carries the clause
   (`a5eff15`); opening (5) in 3.6 carries `state()`. No source is wired to green a gate.
26. (appended 2026-09-15, after the panel; C27) 2.9's COMMIT ORDER WAS INVERTED BETWEEN STEPS 5 AND 6, AND GIT IS THE
   INSTRUMENT: section 3 was committed at `09892f2` (10:33:51 -0500) before the step-5 surfaces `c12a91c` (10:33:56)
   and annotations `f27f436` (10:34:04) that it describes, and both later commits name "(2.9 step 5)" in their own
   subjects. Two of 3.4's statements were therefore not true of the tree at their own commit: `CONCLUSION[
   "metacognition"]`'s surface symbol `metacog_tracking` (the field was still `()` with the comment "T077 is unbuilt:
   nothing prints it") and the cockpit `reliability` row's dated PROVENANCE line -- both landed in `c12a91c` five
   seconds later. 2.10 had re-affirmed the order ("2.9's steps and their order ... stand as committed"), so this is a
   deviation, not an amendment. Nothing measured is affected and the tree at `9293a7f` carries record and surfaces
   consistently; the per-arc rhythm is not breached (the last commit before the push is a record commit).
   Re-ordering is not available -- a rebase would break 3.1's own committed hash chain -- so the disclosure is the
   remedy; the fold lands its surfaces BEFORE its record commit.

### 3.6 What this row does NOT claim, and the openings it leaves

No experiential claim; the C5 refused sentence stands and the licensed one is not written. The reading is about ONE
stream (the scratch brain's prequential readout at 600 ticks, 8 seeds) and, reported beside, its perception stream;
nothing is claimed about chat answers, reasoning or the arm, nor about the daemon's brain. NOT claimed: that the
monitor's read is useless -- it decides chat routes (its own record: 3/30) and arm D shows the second-order hold
deciding ~half of them; what is measured is that neither read tracks the readout's error. OPENINGS, each named for a
successor row: (1) the readout below persistence on every seed and on the daemon (3.4 ii) -- L1c's online readout
owes a competence floor; (2) the two LIF pools' P2 fate on a matched-input A/B with arm D's counts as input (2.0
point 4): a hold that is True on 319 of 320 chat turns from a pool silent by construction; (3) the monitor's gates
touching nothing on the readout's tick (3.4 i) -- P3's "consumed but not decisive" shape for `learn_gate`,
`explore` and `grow_trigger` on this path; (4) the `conflict` term as the mover of the read's relation to
correctness, in both directions across seeds -- a successor that wants a tracking read starts there.

*DATED 2026-09-15 (the panel, C21): the row's own words (tasks.md T077, SYNTHESIS L7) name a DERIVED label-entropy
floor that REFUSES below. What shipped is (i) a refusal at label entropy exactly 0, expressed as `minority == 0`
(`source_entropy` of an all-one-class outcome string is 0.0), and (ii) a refusal on each stream's own permutation null
against a DECLARED rho = 0.25, 2.0 point 1's pre-freeze replacement. No DERIVED positive entropy bar exists:
`entropy_bits` is written into the pin and read by no code -- on this campaign all 8 seeds read readable at entropy
0.3451-0.7333 and the readability decision is byte-identical with the field deleted. The row's word DERIVED is NOT
satisfied by the shipped floor; the substitution was disclosed pre-number in 2.0 / 2.3 (a) and on both surfaces, so it
is a disclosed non-delivery, not a post-hoc relabelling; and it is filed here, not in 3.5, because it was decided in
2.0 BEFORE the freeze and before any number. The field stays (it is the row's only trace of "label-entropy" and the pin
is committed) and is disclosed as a reported-only descriptor in a live CORE module, not a consumed one.*

OPENINGS ADDED 2026-09-15 after the panel: (5) `SecondOrderAccess.state()` and the calibration SCORE it carries -- a
method on a live-path module with no live-path call site (the P3 amendment's shape, not shape 4: it is not called,
rather than called and inert); its docstring and the distinct-from note at `cortex/arealization.py` both call it a
"per-tick" snapshot, which no caller makes true, and the P2 survivor's score (3.5 item 13's circular 1.000) reaches no
live reader. A successor row either gives it a consumer that changes an outcome or retires it; wiring it to green a
gate is the force-wire P3's honesty guard forbids, and this NULL row does not touch incumbent source. (6) the fed
`confidence` (self_model confidence x decision confidence at the monitor's call site in `Brain`) identically 0.0 on one
seed of eight (3.5 item 17) -- a fact about the self-model / BG margin, not about the scorer; it makes that seed's
passthrough control vacuous and is the successor's input, not this row's to adjudicate.

### 3.7 Amendment, appended 2026-09-15 after the panel: (b5) WITHDRAWN under P8 clause 1 (C9)

`out["oracle"] = _auroc2([float(y) for y in correct], correct)` never reads `conf`. By the Mann-Whitney form with
ties at 1/2, conf := outcome puts every incorrect item at one rank and every correct item at a higher one, so the
statistic is exactly 1.0 for EVERY stream with both outcome classes present -- invariant to the recorded read.
`minority == 0` already refuses before it, and `verdict["b5"]` is computed only inside `if readable:`, where (a)
guarantees both classes and q95 <= 0.5 + rho = 0.75 < 1.0 -- so BOTH of (b5)'s conjuncts (`oracle == 1.0` on every
seed; `oracle > q95` on every seed) are ENTAILED by (a). Measured on seed 0's pinned streams (`scratchpad/t077-panel/
fold_verify_pin.py`): the read as pinned auroc2 0.3828 / oracle 1.0; the read scrambled 0.5330 / 1.0; pure noise
0.5721 / 1.0; a constant read 0.5000 / 1.0; an ANTI-CORRELATED read (auroc2 exactly 0.0000, the worst possible) 1.0
above its q95 0.5503; the identity holds at n_ok in {1, 300, 599} of 600. The pin itself shows oracle 1.0 on all 16
intact / lesioned streams. So (b5) had no reachable red on any stream that clears (a); as a 2.6 REFUSAL trigger it
could never fire independently, and 3.4's "its positive control reads 1.0 above every band" is a statement about
`_auroc2`'s sign, not about the campaign.

DECLARED, and its consequence stated so the withdrawal is not a bar moved after the number: (b5) is WITHDRAWN as a
barred conjunct of PASS and as a 2.6 REFUSAL trigger (P8 clause 1 / F11: a criterion with no reachable red is
deleted, not weakened). Nothing in the reading changes -- `verdict["pass"]` was already False and removing a MET
conjunct cannot change it; 2.6's NULL branch still fires on (a) and (c) MET, so nothing of the incumbents is deleted or
re-labelled; 3.4's "the instrument works" rests on the surviving legs -- (a)'s per-stream permutation refusal (R1),
(c)'s exact replay with its 1e-9 red (R0), and R2 / R3 / R5 firing on constructed inputs (3.5 item 10). 3.4's sentence
"its positive control reads 1.0 above every band" is withdrawn. P8 clause 4, disclosed late: a precondition made the
check vacuous and that was not disclosed at pre-registration. 2.3, 2.5, 2.6, 3.3's (b5) row and 3.4 are left
byte-unchanged; the code keeps the key `oracle` and the (b5) computation so the written pin stays readable, with the
docstring, the computing line and the two test messages re-worded to "the orientation identity of `_auroc2`"
(`d03b934`). If a successor row wants a real external positive control of the recording + pairing path, it is built
OFFLINE over the recorded rows -- overwrite each row's read with a planted function of the outcome that
`t077_pairs_prospective` will actually assign to that read, and assert auroc2 near 1.0 through the campaign's own
pairing and scorer -- never as a new in-vivo arm on this frozen tree.

### 3.8 The panel fold (2026-09-15; 39 agents, three lenses, refuters judging CLAIM and FIX separately)

The panel (instrument / pre-registration-honesty / disposition lenses at head `9293a7f`) filed 36 findings; 27 STOOD
after refutation (2 BLOCKER, 10 MAJOR, 15 MINOR at the refuters' corrected severities -- three filed BLOCKERs were
downgraded and four MAJORs became MINOR) and 9 were REFUTED; and every one of the 27 filed fixes was judged WRONG by
its refuter (obs #2146's shape, 27 of 27). The fold landed the REFUTERS' minimal fixes, never the panel's, as eleven
stacked commits plus this record: `475fbe4` R2 / R5 rebuilt on `contrast` with mutated-scorer reds and the six reds
as one self-check line; `e7dbb4d` the conclusion gate's arming red re-pointed (RED since `f27f436`); `e544e59`
`conf_distinct` + the confidence-axis red; `9e4ab1b` the `T_BAR` import; `d03b934` the monitor's stale wording;
`b84d6bc` CONTROL 5, the ceiling; `0a3d443` `_agree` deleted; `57ec145` the PROVENANCE clause; `233a728` the
docstring's learn-gate amendment; `a5eff15` the SYNTHESIS row; then this commit (3.2's amendment, 3.3 x3, 3.4 x1,
3.5 items 9-26, 3.6, 3.7) and the dated pointer lines in tasks.md, spec.md SC-009 and PLAN-FINAL. Every number this
fold wrote into the record was REPRODUCED by the fold's own probes before it was written (the pin re-read; the
8-seed recording re-run for the lag facts; arm D re-run on seeds 0 and 3; the four gates and the learn gate re-probed;
the ceiling control and the calibration price measured) -- the refuters' output was data, not authority.

Where two refuters disagreed the tree decided: C4's and C8's replacements for the PROVENANCE clause conflicted, and
a probe (3.5 item 11) showed C8's wording would have put a refuted inertness claim on the surface; C2's refuter allowed
an in-place cut of 3.4 while others held section 3 append-only -- the fold did both (the false clause cut, the item
appended), because a false sentence left standing in post-campaign prose is not the same as frozen text. NOT done,
and why: C13's third item (new verdict keys in the campaign writer) and C22's pin fields -- the writer refuses while
the pin exists, so the code would be uncharged and the source shape would stop matching the artifact; the seed-1 and
(c) facts went into 3.5 instead. C9's external positive control is a successor's, not this tree's. The 9 refuted
findings, by title: the `conflict` neuter as a channel ablation (mechanical facts reproduce, the causal charge never
measured); tasks.md's "and on the daemon" contradicting 3.6 (it does not); item 8 repairing two of six stated times
(the other four have a better instrument); the pin not storing the per-tick inputs (the observation stands, the claim
built on it does not -- it re-appears as C22 in its correct form); (b3o) tabled as a separate statistic (already in
the file); the pre-freeze dry run showing the degeneracies (its evidence does not exist on this tree); tasks.md `[X]`
relabelling the P2 clause after Codex (the reading predates the number and the pre-registration was never touched);
(b4) MET at t = 0.0 as a surviving null-as-conjunct (the arithmetic reproduces, the defect does not); no PROVENANCE
line for `live_accuracy` (the facts reproduce, the charge does not).

THE FOLD IS ITSELF UNREVIEWED CODE (obs #35): it was not re-panelled; it is battery-verified only -- the arc's
battery, named in the closing record commit of this push, is its review. The verdict, the pin, the disposition and
the four openings of 3.6 stand as recorded; the fold added openings (5) and (6), withdrew (b5), and moved no bar.
