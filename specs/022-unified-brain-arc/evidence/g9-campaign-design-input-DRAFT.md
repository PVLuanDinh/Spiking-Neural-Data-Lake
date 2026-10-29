# G9 campaign: design input (owner 2026-10-25) -- DRAFT

**Status: DRAFT. Nothing in this file is a pre-registration.** No threshold, seed, arm or bar here is frozen. G9's
campaign pre-registration does not exist at `c4c10b28`: `git ls-tree -r c4c10b28 specs/022-unified-brain-arc/evidence`
lists no `*g9*` file. This file collects design input so the pre-registration, when written, starts from it. It changes no
frozen text. Pre-data, pre-build; written statically at base `c4c10b28` by lane hw4-g9 (Sonnet, docs only). It had no
panel and no policy-enforcer read when written.

Owner words, verbatim (2026-10-25, to claude-engine-1024): "Fold the representational-drift report's items 3-4 into G5's
relational-readout pre-registration and items 1-2 into G9's campaign, as design input." Items 3-4 went to G5 Part E
(g5e-prep, `claude/g5e-prep-1029@a2184bef`, addendum X.1-X.2). This file takes items 1-2. Numbering is section 3 of
`lanes/research-1025-drift/REPORT.md` (the HW4 .docx files carry the same content as RQ4, RQ5 and the RQ6 box, with no
numbered items). The tree already holds the owner's note on the same two items as a dated note on T117
(`tasks.md:257`); this file adds what that note leaves open.

Anchors, all at `c4c10b28`: PLAN = `specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md`; TASKS =
`specs/022-unified-brain-arc/tasks.md`.

## Item 1 -- measure drift across the engine's own events

- **Claim, quoted:** "Present a fixed test-input set before/after (a) a growth step, (b) a sleep bout, (c) a
  fresh-process restart. Record reservoir spikes; compute population-vector correlation, ensemble rate correlation,
  tuning-curve correlation vs elapsed events, and RSA-matrix stability."
- **Source, as the report cites it:** Deitch, Rubin & Ziv 2021, Curr Biol 31(19):4327-4339.e6, doi
  10.1016/j.cub.2021.07.062 (responses to the same natural movie drift over minutes to days; drift is driven more by
  ensemble rate change than tuning change; relations between population patterns stay stable); Marks & Goard 2021, Nat
  Commun 12(1):5169, doi 10.1038/s41467-021-25436-3 (drift is stimulus dependent). Both checked on PubMed by the report
  (2026-10-25). The measurement recipe itself is the report's construction, not a paper result.
- **What it implies for G9:** a new ENDPOINT family, called "criterion 1" by the F3 note (`d1-readout-staleness.md:550-553`),
  read at three event types.
  - Event (a), growth step: binds to the loop step "G3's loop grows column-sized batches on the map and feeds them into
    the reservoir" (PLAN:2038-2040; TASKS:257 "grow and feed (T111)").
  - Event (b), sleep bout: binds to "Sleep prunes and saves (G6)" (PLAN:2041).
  - Event (c), fresh-process restart: binds to "A FRESH PROCESS then rebuilds the reservoir from lake records"
    (PLAN:2041-2042), and is the same event as G-R9 (iv) and (vi) (PLAN:2331).
  - It is a READING of the arms G9 already has; it adds no arm. It is a secondary endpoint, not a replacement for the
    solved-predicate endpoints of G-R9 (i)-(vii): drift is not a pass or fail on the engine, and no bar for it is set here.
  - Disclosed limit to carry: the report reads drift against "elapsed events" (a discrete axis). The mouse curves are
    over minutes to days. The engine has no time axis until G9 defines one, so the biology is a reference SHAPE, not a
    numeric target (same limit as Part E X.1, last bullet).
  - Disclosed limit to carry: pruned units read 0 and born units are absent from the pre side, so a population-vector
    correlation after growth or sleep compares different unit sets. The reading function must say which set it used
    (X.4 of the g5e addendum fixes this by input identity and a common unit list).
- **File:line it would bind to (at c4c10b28):** TASKS:257 (T117's dated 2026-10-25 note already names it as design
  input); PLAN:2036-2053 (the G9 arm list); PLAN:2331 (G-R9, where a drift endpoint would be added as a secondary
  column, not as a new red).
- **Needs a run before G9 can freeze?** YES, one: a PILOT on the G9 engine for the SD of the drift readings under the
  frozen-engine arm (item 2 below), because P1 forbids picking a seed count by habit and the report's own item 6 asks for a
  pilot SD. `TASKS:257` fixes "S = 8 under Part F's rule" for the solve endpoints. S = 8 is not derived from any drift
  variance, so it is not a sample size for this reading. The pilot cannot run until the engine arms exist (G9 runs last,
  after G1(c), G3, G5 and G6; PLAN:2057), so G9 cannot freeze this endpoint before then. Everything else in item 1 is a
  static edit.

## Item 2 -- per-part attribution through G9's own arms

- **Claim, quoted:** "Per-part attribution reuses G9's arms: growth OFF at final size, sleep OFF, transport neutered
  (norm-matched random), return path neutered."
- **Source:** the same two papers motivate the question (what drives drift); the attribution design is the report's, built
  on G9's arm list. No paper result bears on which engine part causes drift.
- **What it implies for G9, arm by arm** (the report's four against PLAN:2036-2053):

  | Report's arm | In G9's list at c4c10b28? | Implication |
  |---|---|---|
  | growth OFF at final size | YES: "growth OFF at the engine's final size (presized, with SC-012b's equal-count silencing control)" (PLAN:2050) | reuse; the drift reading is taken on this arm too. No change. |
  | sleep OFF | **NO.** G9's arms contain no sleep-off arm; sleep is a step of the loop (PLAN:2041) | a NEW ARM is owed if sleep's share of drift is to be attributed. It is a design choice, not a fold: the panel decides whether to add it (cost: one more arm at S = 8). If it is not added, the attribution for sleep is a DISCLOSED LIMIT (drift after a sleep bout is read, but not compared with a no-sleep twin). |
  | transport neutered (norm-matched random) | YES, per form: "a norm-matched random matrix for the copy form; the brain's drive time-shuffled for the activity form" (PLAN:2047-2048) | reuse. The report's words "norm-matched random" fit the copy form only; the activity form is time-shuffled, so the report's wording is NOT carried literally. |
  | return path neutered | YES (PLAN:2049) | reuse. |

  The report's list also omits two of G9's arms that drift attribution would need to be complete: the feed-neutered arm
  and the reservoir-removed arm (PLAN:2044-2046). A removed reservoir has no reservoir spikes to read, so the drift
  reading is undefined there; the draft records that as "not readable", not as zero drift.
- **Report item 5 (a control that is not the method) is the missing floor for items 1-2.** A frozen engine (no growth, no
  sleep) is the noise floor of every drift reading. g5e-prep's addendum calls it G9's criterion 5 (X.4); no such
  criterion is in the tree at c4c10b28, since G9 has no pre-registration. It is also the arm that gives the PILOT SD for
  the P1 design. The draft puts it first in the order of owed items.
- **Disclosed limit (selection bias, from the HW4 RQ5 text):** drift of units that were pruned or born is not drift of the
  same unit. Attribution must be reported on the units common to both sides, and the count of dropped units reported next
  to it, or a part that prunes more would look like it drifts more.
- **File:line it would bind to (at c4c10b28):** PLAN:2036-2057 (the arm list; the sleep-off arm would be inserted here);
  TASKS:257 (the arm list repeated, same insertion); PLAN:2331 G-R9 (ii) (the attribution wording "each attributed
  separately" already exists, so attribution is a natural place for the drift reading to sit).
- **Needs a run before G9 can freeze?** The sleep-off arm needs no run to be CHOSEN. The attribution contrasts need the
  same pilot as item 1. A sample size for the contrasts (each arm against the full engine) is a P1 item: derive it from
  the pilot SD, then a sequential design with named looks, an alpha-spending boundary per look, a non-binding futility
  bound and a hard cap whose power stays at least .8 (checked by simulation before data). A futility stop or an uncrossed
  cap is not a null (P8 null-scale item 4).

## F3 overlap: the shared drift-reading function

- `d1-readout-staleness.md:550-553` (A0c F3) names G9's criterion-1 instrument as one of three claimants for ONE
  drift-reading function ("Whichever builds first owns ONE drift-reading function, and the others import it"); `:705`
  (C15) says a fourth claimant, MAP.md D0's `instruments/representational_drift.py`, "is added".
- Item 1 above IS that criterion-1 instrument: its three correlation readings (population vector, ensemble rate, tuning
  curve) are what `read_drift(pre, post, *, probe_ids, units=None, metric)` in the g5e addendum X.4 takes as
  `metric in {"pv", "rate", "tuning"}`. G9 imports it and keeps no local copy. The RSA-matrix stability reading is NOT in
  X.4's three metrics: it is the one piece of item 1 the shared function does not yet cover. The freeze either adds
  `"rsa"` to X.4's metric set or builds it inside G9; adding it to the shared function is the P2-safe default because the
  RSA matrix is the same correlation primitive applied to probe-by-probe geometry.
- Static finding carried from X.4: `infra/_mathutils.pearson` has no zero-variance guard (`infra/_mathutils.py:44-51`).
  A silent 0.0 on a flat reservoir response would read as "no drift" in item 1. The guard belongs in `read_drift`.
- Status of the claimants at c4c10b28: none of the four exists in the tree (X.3 row 5). Which builds first is open;
  G9 runs last (PLAN:2057), so G9 is the least likely to build it. G9 should not wait for it, and should not write a
  fifth copy. P2 (two mechanisms for one job) is the rule.

## Raising-ambition record (self-apply, owner 2026-10-25)

- **Axis:** Depth (settle what a careful static read can settle before G9 has a pre-registration).
- **Ladder:** (1) quote the two items and map them to the arm list; (2) bind each to file:line at c4c10b28, find what the
  arm list lacks (the sleep-off arm, the missing floor, the RSA reading), and say what needs a run; (3) draft G9's
  drift endpoint with the sequential design filled in and the arms edited into the PLAN.
- **PICKED:** rung 2.
- **The rung above, ruled out:** rung 3. POLICIES P1 rules out filling in the design: P1 requires a sample size derived
  from a measured variance, and there is no measured variance for engine drift (no pilot can run: campaign rules forbid
  any run, and G9's arms are unbuilt). POLICIES P8 independently bars claiming a pre-registered red for the reading
  before it has been counted. Editing the PLAN's arm list would also change frozen text, which this lane may not do.
- **Outside judge:** the policy-enforcer's static review against POLICIES.md, then the integrator's panel. Every
  statement above carries a file:line at c4c10b28 that either reader can re-open. No statement is a measurement.

## Owed before G9 can freeze this input (summary)

1. OPEN, panel choice: add a sleep-off arm to PLAN:2036-2053 and TASKS:257, or disclose that sleep is not attributed.
2. OPEN, edit: the frozen-engine floor arm stated as a G9 arm (it is not one at c4c10b28).
3. OPEN, edit: RSA-matrix stability added to the shared `read_drift` metric set, or built in G9 (P2).
4. NEEDS A RUN (after the campaign, after G9's arms exist): the pilot SD of the drift readings; then the P1 sequential
   design (looks, boundary per look, futility bound, cap, simulation) is derived from it.
5. OPEN, edit: say which unit set each reading is taken over, and report the dropped-unit count beside it.

## Panel fold 1, 2026-10-29, pre-data

Append-only. Nothing above this heading is edited; where this addendum changes a sentence above, it says which line and
the addendum governs. Still a DRAFT, still no pre-registration, still no run. Base `c4c10b28`. Findings: lens A #6290
(A-C1..C3), lens B #6287 (B-M1, B-M2, B-m1, B-m2), lens C #6298 (C-3a..C-3e), policy-enforcer #6292 (NIT) and its
correction #6303 (concurs with A-C1, A-C2, B-M1, B-M2, C-3a; withdraws its AGREE). Each finding by id:

1. **A-C1 (pilot SD from the wrong arm; no direction, no smallest effect): FIXED.** Lines 47-48 and 72-75 are superseded
   by: "the SD that sizes each contrast is the SD of the per-seed PAIRED DIFFERENCE (arm minus full engine) on shared
   seeds; the pilot runs the full engine and every contrast arm on the same seeds. The frozen-engine arm is the noise
   floor only, never the variance source." Why it closes the finding: a frozen arm with fixed inputs can read correlation
   near 1 on every seed (SD near 0), which would size a tiny, unpowered n; the paired-difference SD is the quantity the
   contrast's test statistic divides by. Also registered as a rule: before the pilot, each contrast registers a DIRECTION
   (the arm's drift reading versus the full engine's, at the event that arm's part acts on) and a SMALLEST EFFECT OF
   INTEREST, as Part E registers its Delta*. The numeric smallest effects are NOT chosen here: that is OPEN item 7 below.
   Line 37 ("no bar ... set here") stands for this DRAFT; the pre-registration may not carry it.
2. **A-C2 and B-M1 (event (c) reads the P4b gap): FIXED.** Added to item 1 as a rule, carried from G5 Part E E.2
   (`g5-readout-variants.md:1881-1892` at `c4c10b28`): "the drift the frozen floor does not absorb is a persistence
   defect, not something for a readout to hide ... a restart P-E0 MET, files a P4b red against the rebuild." So event (c)
   is not a neutral reading. Restart drift above the frozen floor files a P4b red first. It is read as drift only for
   state the pre-registration NAMES as deliberately not persisted (E.2's example: RNG streams, transient membrane
   potentials), shown by a control not to be learned state (the drift vanishes when only that state is restored). Order:
   event (c) is read after #6231's persistence build is GREEN (four live learners are unrestored at `c4c10b28`, #6231
   DRAFT :36-44), or under that named exception. A bit-identical rebuild gives zero restart drift by construction and
   the restart reading is then NOT READ. Why it closes the finding: without it G9 would attribute amnesia to engine parts.
3. **A-C3 and C-3a (X.4 cross-reference false; two unit-set rules): FIXED here, cross-lane part OPEN.** Lines 41-43 are
   corrected: X.4 at `a2184bef` keeps pruned units read 0, so it does NOT fix the bias; the sentence "X.4 fixes this by
   input identity and a common unit list" is withdrawn. Rule for this draft, stated here: the drift function takes
   `units` explicitly, defaults to the INTERSECTION (present and not pruned on both sides) and returns `n_common`,
   `n_pruned`, `n_born` beside the reading; a zero-fill of pruned units is a Part E READOUT rule (E.4), never a
   drift-reading rule. This is the rule lens C proposed for #6244 (C-2a) and lens A for #6244 B1. OPEN (not mine to edit):
   #6244's own fold must adopt the same rule, or the integrator chooses at merge. One function, one unit-set rule.
4. **B-M2 (the reader has no sensitivity control): FIXED.** New owed item 6 (below): planted-drift reds for the drift
   reader, counted before ANY engine read: identical pre and post reads 1.0; a planted permutation or a planted rate
   change reads lower; a flat response raises the flat flag. Lens B's point is accepted: the frozen floor is a noise floor,
   not a proof the reader can see drift. (Policy-enforcer #6303 concurs and withdraws its own acceptance of the floor as
   "the control".) The same three reds are also owed in #6244 X.4 (lens B, #6284); they are one set, owed once.
5. **B-m1 (P1, "one more arm at S = 8"): FIXED.** The words "cost: one more arm at S = 8" (line 65) and the S = 8 remark
   (lines 49-50) are superseded by: every G9 endpoint, solve endpoints included, owes its own "Sample size (P1)" section
   in the pre-registration; no seed count here is a decision, `TASKS:257`'s "S = 8 under Part F's rule" is not carried
   into this draft as a sample size.
6. **B-m2 (P8 item 3, no sleep-off arm): FIXED.** One line, binding if the panel does not add the arm: "no claim about
   sleep's share of drift is made; drift after a sleep bout is read, and not compared with a no-sleep twin."
7. **C-3b (the flat-vector sentence is wrong): FIXED.** Lines 99-100 are corrected after reading
   `infra/_mathutils.py:44-51` at `c4c10b28` (`return cov / math.sqrt(va * vb)`): on a flat vector `va * vb` is 0 and
   this RAISES `ZeroDivisionError`; it does not return 0.0. The silent 0.0 belongs to `grid_place_map._pearson`'s guard
   (lens C, `grid_place_map.py:56-57`, not re-read by this lane). And a correlation of 0 is maximal decorrelation, i.e.
   maximal drift, not "no drift". The call-site guard in the drift reader stands: it must flag a flat response, not
   crash and not return a number.
8. **C-3c (RQ5 attribution): FIXED.** Line 76's "from the HW4 RQ5 text" is withdrawn. The selection-bias limit is this
   draft's own (and consistent with E.4's FIXED rule); HW4 RQ5 (`HW4-v3.txt:42-47`) has no such statement.
9. **C-3d and policy-enforcer NIT (#6292) ("criterion 5" label): FIXED.** Line 73's wording is replaced by "report item 5
   (`REPORT.md:44`) and the HW4 noise-floor bullet (`HW4-v3.txt:52`), proposed as a G9 floor arm; there is no numbered
   G9 criterion 5 in the tree". Read "proposed G9 criterion 5" in any place the label appears.
10. **C-3e (TASKS:257 contradicts itself on sleep OFF): FIXED.** Owed item 1 (line 121) is extended: `tasks.md:257`'s dated
    2026-10-25 note lists "sleep OFF" among "G9's own arms", while its own arm list and `PLAN:2043-2053` have none; the
    panel's choice on item 1 also corrects that note.
11. **Disputed: none.** Every finding above reproduced against `c4c10b28` or a draft head, no finding is disputed.

### Owed list, replacing the summary at lines 119-126 (that list stays above, unedited; this one governs)

1. OPEN, panel choice: add a sleep-off arm to `PLAN:2036-2053` and `TASKS:257`, or take fold item 6's disclosure; and
   correct `TASKS:257`'s self-contradicting note either way (fold item 10).
2. OPEN, edit: the frozen-engine floor arm stated as a G9 arm (not one at `c4c10b28`); a floor, not a variance source.
3. OPEN, edit: RSA-matrix stability added to the shared drift reader's metric set (P2-safe default), or built in G9.
4. NEEDS A RUN, after the campaign and after G9's arms exist: the PILOT of the full engine and every contrast arm on
   shared seeds; SD of each per-seed paired difference; then the P1 sequential design (named looks, alpha-spending
   boundary per look, non-binding futility bound, hard cap = fixed-sample target times the inflation factor, power at the
   cap at least .8, checked by simulation before data). A futility stop or an uncrossed cap is not a null (P8 item 4).
5. OPEN, edit: one unit-set rule (intersection default; `n_common`, `n_pruned`, `n_born` returned), agreed with #6244
   (fold item 3).
6. OWED before any engine read (P8 item 2): the drift reader's planted reds (identity 1.0; planted permutation or rate
   change lower; flat response flagged), counted and logged. Static, no engine needed.
7. OPEN, choice with no run needed: per contrast, the registered direction and the smallest effect of interest, before the
   pilot (fold item 1). Not chosen in this draft; the numeric values belong to the pre-registration author.
8. ORDER: event (c) restart reading only after #6231's build is GREEN, or under E.2's named-not-persisted exception
   (fold item 2); restart drift above the floor files a P4b red.


## ADDENDUM 2026-10-29 (lane g9-gaps, D-65 item 3): the two owed arms, designed; RSA built

Append-only. Nothing above is edited. Still a DRAFT, still not a pre-registration: no threshold, seed, bar or sample
size below is frozen. Written statically at base `7c1eebe2` (the DRAFT's own commit; code identical to `c4c10b28`).
No test or brain was run (campaign rules). It closes "Owed" items 1-3 as DESIGN (the panel still chooses) and item 3 as
code (see the last section). Items 4-5 stay open.

### A. The sleep-off arm (owed item 1)

- **What it holds fixed:** the seed, the problems and their taught inputs, the number of Brain ticks, the wake ticks'
  inputs, growth ON (G3's loop), transport, feed, return path, and the probe set of the drift reading. It changes ONE
  thing: no sleep bout ever starts.
- **What it isolates:** sleep's own share of (i) the solve endpoints and (ii) the drift reading, against the full
  engine. Without it the DRAFT's item 2 can only DISCLOSE that sleep is not attributed.
- **How sleep is turned off at the current tree (file:line, 7c1eebe2):**
  - `Brain.tick` reads `self.clock.consolidation_gate()` once per tick and enters `_enter_sleep` / `_sleep_tick` on it
    (`engine/brain.py:3910-3917`). That is the gate's only tick caller.
  - `instruments/sleep_pressure.py:305-320`, `gate_from(brain, decide)`, answers that call from `decide()` for a block
    and restores it exactly. `gate_from(brain, lambda: False)` is sleep OFF with no new mechanism. REUSE it; do not write
    a second override (P2).
  - **Not the same thing, and easy to mistake:** `Brain.neuter_sleep` (`engine/brain.py:3623-3638`) turns off only
    the STRUCTURAL rewiring (`_sleep_caps["structural"]`, set `:370`, read `:6013`). The other three names raise
    (retired, owner1039). `_sleep_tick` still runs growth (`:3358`), the episode replay (`:3369-3372`) and
    `_r2_sleep_learning` (`:3376-3377`). So `neuter_sleep` is one NAMED part of sleep, not "sleep OFF"; the arm uses the
    gate.
- **Confound to carry (found at the hook, not in the DRAFT):** `_sleep_tick`'s own docstring says "Growth is thus
  sleep-gated" (`engine/brain.py:3352-3354`): the NG-SNN `grower.tick()` runs only inside it. A gate-off arm therefore
  also stops that growth, so "sleep OFF" and "growth OFF" are NOT separable on the incumbent unless the contrast is read
  against the growth-OFF arm (PLAN:2050) too. Whether G3's column-sized growth is also sleep-gated is unknown here (G3
  does not exist at this base). The pre-registration must read that before it claims a sleep-only effect.
- **Open, unverified:** does the lake record anything outside a sleep bout? The fresh-process leg ("sleep prunes and
  saves", PLAN:2041; G-R9 iv and vi) may have nothing to rebuild from in a sleep-off twin. Not established statically.
  If so, that leg is read as "not readable" for this arm, never as drift 0 (same rule as the reservoir-removed arm).
- **Its control (one that is not the method):** the TRANSPARENCY control. The same `gate_from` override is installed, but
  `decide()` replays the clock's own scheduled gate sequence read off a copy of the clock (the construction
  `sleep_pressure.py:303` already uses). Sleep stays ON and the override is the only change. If this arm differs from
  the unmodified engine, the harness, not sleep, moved the reading. A sleep-off result is read only against this twin.
- **Its red:** an arm that claims "sleep off" while a sleep bout runs must be refused. The observable is the brain's own
  `_asleep` / `_was_asleep` flags (`engine/brain.py:3915,3920`): count `_enter_sleep` calls over the run. Red = the
  override is not installed (or `decide()` returns True once) and the count is above 0 on the OFF arm: the run exits 3.
  The same count must be above 0 on the ON and transparency arms (positive control: a counter that always reads 0 would
  pass the OFF arm for free).
- **Sample-size input (P1):** the SD of the PAIRED per-seed difference (engine minus sleep-off) of each drift metric and
  of the solve endpoint, from the frozen-engine pilot (section B). No SD exists, so no S is derived here and the DRAFT's
  "S = 8" is not carried to this contrast. Then the P1 sequential design: named looks, a named alpha-spending boundary
  per look, a non-binding futility bound, a hard cap = fixed-sample target x the design's inflation factor, power at the
  cap >= .8 checked by simulation before data. A futility stop or an uncrossed cap is not a null (P8 null-scale item 4).
- **Cost:** one arm (plus its transparency twin) at the same S as the other arms.

### B. The frozen-engine floor arm (owed item 2; the report's item 5; g5e-prep calls it G9 criterion 5)

- **What it holds fixed:** everything the engine arm holds (seed, problems, inputs, ticks, probe set, the SAME events:
  a growth-step slot, a sleep-bout slot, a fresh-process restart). It changes the engine to one that WRITES nothing: no
  growth, no sleep, no weight update, no pruning.
- **What it isolates:** the noise floor of every drift reading: how far the probe responses move across the same events
  when no part of the engine changed anything (rng and tick-state carry-over, restore nondeterminism, input-order
  effects). Every nonzero drift in the other arms is read against it. It is also the only arm that yields the PILOT SD
  the sleep-off arm and the contrasts need.
- **How the engine is frozen at the current tree:** THERE IS NO WHOLE-ENGINE FREEZE. A search of `engine/` for a
  `def freeze` finds none. The only `freeze` hit near the topic, `growth/similarity_sized_growth.py:124-128`, makes ONE
  perceptron unit's weight array read-only; it is not the engine. The arm must be COMPOSED from switches that exist:
  - sleep OFF: `gate_from(brain, lambda: False)` (`sleep_pressure.py:305-320`);
  - structural rewiring off: `brain.neuter_sleep("structural")` (`engine/brain.py:3623`);
  - the plasticity rule stack off: `Brain._plasticity` (`:365`, built `:375-381`, installed as `brain._ixn.stack`
    `:381`); whether setting it to None stops EVERY weight write is not established here;
  - growth off: no `grower.tick()` (`:3358`, sleep-gated) and no `grower.observe` (`:2577`, wake);
  - the learning-off precedent: the `evaluate` child of `instruments/sleep_plasticity_ab.py` runs with "learning off
    (weights + rng digested before and after, K4)" (`:16-17`, `:563`), and `route_text` is pure inference (`:242`).
  The composition is the build's job. It is NOT assumed complete: the proof is the digest below.
- **Its control (one that is not the method):** a PLANTED-CHANGE twin. Take the frozen engine and plant one known,
  small, exact change (for example scale one stored weight row by a fixed factor between the two readings). Each drift
  metric must move off its floor by the planted amount, in the right direction. It shows the reading can see a change at
  all, so a flat floor is not an instrument that reads zero for everything. (This is not the engine and not the
  frozen engine, so it controls the reader, not the method.)
- **Its red:** the freeze is verified, not declared. `state_digest(b)` (`sleep_plasticity_ab.py:221`, used `:392`) of
  weights plus rng before and after the whole event sequence must be EQUAL. Red = a freeze built with one write path
  left open (for example the plasticity stack still on) moves the digest, and the arm exits 3 instead of reporting a
  floor. A floor measured on an engine that was not frozen would be read as "drift is small".
- **Sample-size input (P1):** the floor arm IS the pilot. It supplies the across-seed SD of each drift metric (pv, rate,
  tuning, rsa), so no S for the drift endpoint can be derived until it has been run on the built G9 engine (G9 runs
  last, PLAN:2057). Report the minimum detectable difference at the cap beside every contrast. Until then the design is
  "derive S from the pilot SD", not a number.
- **Unit-set rule (DRAFT item 5):** a frozen engine has no born or pruned units, so its readings use the full unit set;
  the engine arms use the common unit list and report the dropped-unit count beside it.

### C. RSA, code (owed item 3, as code)

- Disposition of `graph-exists.js` (run 2026-10-29, `--root` this worktree): "representational similarity RSA drift
  between two response matrices": NO LEXICAL MATCH, UNKNOWN not absence; "RSA": NO LEXICAL MATCH; "pairwise
  dissimilarity matrix correlation": NO LEXICAL MATCH; "drift reading population vector correlation": 4 hits,
  `grid_place_map._pearson` (distinct-from: a guarded grid-code correlation), `drift_monitor.psi` (distinct-from: input
  distribution drift), `prior_drift_harness.choose_d` and `h01_ingest.load_ei_counts` (distinct-from: unrelated). A
  source read found no RSA function.
- Built: `rsa_drift(pre, post)` in `infra/_mathutils.py`, beside `pearson`, commits `fd987f3f` (test, first) then
  `f0f50385` (code). It lives in `_mathutils` and NOT in F3's `instruments/representational_drift.py` because that file
  does not exist, and a new instrument module needs a census / dormancy row that only the integrator may add. When
  F3's module is built, `rsa_drift` moves there as a one-line import swap. Correspondence is by probe index.
- Honest limits: it returns 1.0 for "same geometry" and does not decide a bar. A flat probe row or a flat geometry
  REFUSES (ValueError), not 0.0, per X.4's flag rule; the caller turns that into exit 3. It is wired into no campaign.
- P2, named and not fixed here: `instruments/rpe_charge.py:343` defines its OWN `pearson`; `faculties/agency_monitor.py:1118`
  `_t079_pearson` and `faculties/grid_place_map.py:49` `_pearson` are guarded copies of the same primitive.
