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
