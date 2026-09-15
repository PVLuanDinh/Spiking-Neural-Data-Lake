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
