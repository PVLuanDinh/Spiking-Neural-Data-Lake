# The critical path to G9 (2026-10-29)

**Analysis only. It decides nothing; the owner decides.**

- **Author:** cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`, on the SNDL launcher's instruction of 2026-10-29.
- **Base:** `origin/main` = `ee224e50`.
- **Sources:** `specs/022-unified-brain-arc/tasks.md` (the rows and its Dependencies section), `PLAN-FINAL-2026-09-02.md`
  (the G workstream's "Order and limits"), and the evidence files cited per row. Everything not cited is marked *my
  inference*.

**What I cannot see.**
- `claude/int3` or any integrator state beyond `origin/main`.
- The owner's chat and the board: OWNER_QUESTIONS, OWNER_DECISIONS, SEEDS and every `#NNNN` item. Where an evidence
  file quotes a board ruling, I rely on its quotation.
- The `claude/*` row branches the G rows built on (for example `claude/r2`).
- The owner host's campaign wall-clock times.

So every effort figure below is an estimate from the repository's own history, not a measurement.

**The target.** G9 is T117. The owner's definition of done is all rows plus G9 on `origin/main`.

## 1. Measured state

**At `ee224e50`:** 123 rows done and 38 open (`grep -c` on `- [x]` / `- [ ]` in tasks.md). This matches the
launcher's figures.

**History.** These are counts of tasks.md on `origin/main` at the last commit on or before each date (`git rev-list
-1 --before`):

| date | done | open | total |
|---|---|---|---|
| 2026-09-01 | 92 | 27 | 119 |
| 2026-09-15 | 110 | 27 | 137 |
| 2026-09-29 | 116 | 38 | 154 |
| 2026-10-13 | 122 | 32 | 154 |
| 2026-10-20 | 122 | 32 | 154 |
| 2026-10-29 | 123 | 38 | 161 |

**From 2026-09-15 to 2026-10-29** (6.3 weeks):
- 13 rows closed, about 2.1 per week.
- 24 rows added, about 3.8 per week.
- Open rose from 27 to 38.

My 2026-09-15 snapshot reads 110 done, where the launcher's figure is 111. That is a one-row difference from the
snapshot's time of day; the trend is the same.

**Where the 24 added rows came from** (row text and dates):

| source | rows | count |
|---|---|---|
| the owner's plan expansions: B/R/A 2026-09-22; G 2026-09-23; the engine 2026-09-25 | T101-T107, T108-T114, T115-T117 | 17 |
| owner-started research | T118 OG1, T121 L1b-STALE, T122 E1c | 3 |
| debt that review panels found | T111a, T119 (done), T120, T123 | 4 |

**Throughput.**
- From 2026-10-06 to 2026-10-28, `origin/main` carries 20 "closing execution record ... battery GREEN" commits: about
  0.87 battery-green pushes ("arcs") a day.
- Of those 20, one touched the G9 chain: engine arc 1, "G3 T0/T1".
- The rest went to the trainer retirement (B2-B4), ENC, INSULA, the P8 amendment, P4b, P5b, E1b, E1c and the L1b-STALE
  pre-registration.
- **The arc rate is not the bottleneck. The share of arcs spent on the G9 chain is.**

## 2. The G9 dependency chain (what the plan says)

**tasks.md, Dependencies, "Added 2026-09-25 (the engine; supersedes the 2026-09-23 line)":**

> T108 -> T109(a) -> T109(b) (T115's record alongside) -> T116 ... -> T109(c) -> T110 -> T111 -> T113 -> T112 -> T114
> -> T117. T114 still waits for T103/T104; T115's label gate lands with T111 at the latest.

**PLAN-FINAL, G9:** "G9 runs last, after G1(c), G3, G5 and G6."
**tasks.md, 2026-10-24:** "T111a sits inside T111: it is step T6 of the one-grower merge ... so it follows T1."

**Already done on that chain:**
- T108 (G0);
- T109 (G1, including G1(c), `evidence/g1c-carried-state.md` section 3);
- T112 (G4);
- T115 (G7, with its label gate);
- T116 (G8);
- T104 (R3, a recorded null).

**Open on it:** T110, T111 (+ T111a), T113, T103, T114 and T117.

## 3. All 38 open rows

**Columns.**
- **G9 needs it?** DIRECT = named in T117's row or G9's plan text. TRANS = needed by a DIRECT row. NO = neither.
- **Source:** the citation that says so, or *inference*.

| row | what it needs | what needs it | G9 needs it? | source |
|---|---|---|---|---|
| T110 G2 | each problem's solved predicate and budget pre-registered; item (ii)'s taught campaign; the Q-AII stack repair (below) | T111, T114, T117 ("T111, T114 and T117 read them") | DIRECT | T110 row; T117 row "on T110's problems" |
| T111 G3 | T110; T115's label gate (done); T111a inside it | T113, T114, T117 | DIRECT | T117 row "grow and feed (T111)" |
| T111a G3 T6 | G3 T0/T1 (landed in engine arc 1) | T111; T118 Stage 3 | TRANS | tasks.md 2026-10-24 line |
| T113 G5 | T111's grown units | T114, T117 | DIRECT | T117 row "readout and return (T113)" |
| T103 R2 | the rebuilt campaign instrument (`evidence/r2-typed-replay-hydration.md` A5) | T114 | TRANS | tasks.md "T114 still waits for T103/T104" |
| T114 G6 | T103, T104 (done), T111, T113 | T117 | DIRECT | T117 row "sleep, prune and save (T114)" |
| T117 G9 | all of the above, plus its own pre-registration and campaign harness (neither exists) | T087 under the owner's definition | -- | T117 row |
| T102 R1 | its own pre-registration (ready to freeze, `r1-bounded-recall.md`) | T105 (A1) | NO | main's ruling: "R1 -> R2/T103 -> R3/T104" is "an order, not a dependency, so R2 does not wait on R1" (`r2-typed-replay-hydration.md` debate row 16, section 2 item 3) |
| T121 L1b-STALE | its parked build (`claude/l1b-stale-r3-1028`), then Stage 0, pilot, A1 | an STDP arm of T113, if one is ever added | NO | `g5-readout-variants.md:1181`: "No unsupervised STDP arm is registered; if one is added later it carries L1b's co-adaptation controls" |
| T122 E1c | B1-B3, pilot, campaign | T123 | NO | T122 row; T123 row |
| T123 P2 `_plan_w`/`_stn_gain` | T122's verdict | -- | NO | T123 row "Owed after E1c's verdict" |
| T118 OG1 | Stage 0-1; Stage 3 waits for T111a | -- | NO | tasks.md 2026-10-23: "T118 is off the G9 critical path" |
| T120 P2 sidecar | code LANDED (`8c81961d`); the owner moves `data/selfbuild_state.json` into the home once | -- | NO | T120 row "REMAINING, the owner's" |
| T093 L4 | the append half built; the persistence half "owes a named carrier" | T103 coordinates its carrier inventory with it | NO (coordination only) | T093 row; T103 row "coordinate the carrier inventory with T093" |
| T089 L5 IMMORTAL | its gate, counted on today's tree first | T074's archive ("an IMMORTAL lake record class") | NO | T089 and T074 rows |
| T088 X1 capability registry | `gates/capability_bar.py` | -- | NO | T088 row |
| T072 proposer | T071 (certified) | T073, T074, T083, T100, T107 | NO, *inference* (see 3.1) | T072 row; Dependencies "T072+ on T071" |
| T073 evaluator cascade | T072 | T074, T083, T100, T107 | NO, *inference* (see 3.1) | T073 row |
| T074 archive + MAP-Elites | T073; T089's IMMORTAL class | T083, T100, T107 | NO, *inference* (see 3.1) | T074 row |
| T075 cockpit self-build rows | -- | T083 | NO | T075 row |
| T076 curriculum | -- | T083 (exit row only) | NO | T076 row: "MECHANISM WITHDRAWN ... only T076's US4 exit row remains and it is WITHDRAWN with the mechanism" |
| T078 attention schema | the brain-side stimulus port first | T083 | NO | T078 row: "STOPPED AT G3 ... the row stays `[ ]`; ... the successor row owes the brain-side stimulus port" |
| T083 US4 exit | T072-T078 | T087 | NO | T083 row |
| T064 unit blocks | void | -- | NO | T064 row: "the row's deliverable is void until a successor paging row"; its surviving 0x09 path was ported |
| T065 binding width | the owed persisted-bytes regression | -- | NO | T065 row: "PARTIAL"; "this row's 'the paging exit runs AT this width' clause is VOID" |
| T051c-wave rule census | 56 undispositioned census rows | -- | NO | T051c-wave row: "no single-build deadline, ratchet-policed" |
| T096 N1 | -- | T097 | NO | Dependencies "Added 2026-09-14": "This arm does not ... block unrelated work" |
| T097 N2 | T096, T101 (done) | T098 | NO | the same |
| T098 N3 | T097 | T099 | NO | the same |
| T099 N4 | T098 | T100 | NO | the same |
| T100 N5 | T099, T073/T074 | -- | NO | the same |
| T105 A1 | T102 | T106 | NO | Dependencies "Combined 2026-09-22": "R1 -> A1/T105 -> A2/T106" |
| T106 A2 | T105 | T107 | NO | the same |
| T107 A3 | T106; T073/T074 for autonomous promotion | -- | NO | T107 row |
| T084 cockpit sweep | the rows whose instruments it displays | T087 | NO | T084 row |
| T085 scale derivation | -- | -- | NO | T085 row |
| T086 consistency sweep | -- | T087 | NO | T086 row |
| T087 quickstart + close-out | everything | -- (the last row) | NO for G9; the last row under the owner's definition | T087 row |

**Count:** 7 rows on the G9 path (T110, T111, T111a, T113, T103, T114, T117) and 31 not.

### 3.1 The one reading that would change the path

**The plan's "no bypass" sentences:**
- tasks.md 2026-09-23: "No G row bypasses T073/T074, Phase 4's exit or FR-025/OD-2."
- PLAN-FINAL G: "nothing here bypasses the phase exits, T073/T074 or FR-025/OD-2."

**My reading: they bind AUTONOMOUS PROMOTION, not G9's scratch-brain campaign.** The same sentence for A and N says
so explicitly (PLAN-FINAL:1147, "Neither A nor N bypasses T073/T074, Phase 4's exit or FR-025/OD-2 for autonomous
promotion"). T117 runs "in scratch brains" and promotes nothing into the live brain.

**This is my inference.** The G sentences omit "for autonomous promotion". If the owner reads them literally, T072,
T073, T074 (and T089 through T074), and probably T083 enter the path, roughly doubling it.

**Owner question 1:** does "no bypass" bind G9's scratch-brain campaign, or only a promotion into the live brain?

## 4. The minimal set that reaches G9, in dependency order

**0. The Q-AII stack repair. It has NO tasks row.**
- `evidence/g2-taught-problems.md`, the 2026-10-13 dated note: the owner answered Q-AII "Fix the stack first" (board
  #2134/#2135, quoted there).
- So "A_ii and (ii) stay parked until a separately pre-registered repair of the E/I balance x Benna-Fusi consolidation
  x homeostatic scaling interaction makes the CANONICAL brain learn A_ii".
- `evidence/g3-growth-loop.md` A6 also carries "readable A_ii and #2135 release" as UNVERIFIED. G2 item (ii) and G3's
  retention both wait on it.
- It is the first link of the chain and is invisible in tasks.md. I list it; the owner decides whether it becomes a
  row.

**The chain:**
1. **T110 G2.** Item (i)'s confirmatory campaign is read (2026-10-07, section 5.1 of the G2 file). Item (ii) waits
   on step 0. C1-d is "REPRESENTABLE, NOT SHOWN LEARNABLE", and its disposition (O11) is the owner's.
2. **T111 G3, with T111a inside it.** The draft is not frozen. A6 lists the UNVERIFIED runners: six-arm/T1/Astra/feed,
   C9-C19's constructed reds, the N=256 feed control and the night costs.
3. **T113 G5.** Parts A-D: the D1 dev freeze is committed and the 2026-10-15 amendment is "QUALIFIED, pending D1/D2";
   A4: "components PRESENT; registered campaign ABSENT".
   - Part E (the drift-surviving relational readout) waits on "The G9 engine (owner 2026-10-23: gap experiments only
     on the G9 engine)" (E.8). That is a cycle with T117 unless Part E closes after G9's harness exists. *My
     inference:* T113 can close on Parts A-D, with Part E carried to G9's harness.
   - **Owner question 2:** does T113 close without Part E?
4. **T103 R2, in parallel with steps 1-3.** It is independent of them. A5: the campaign instrument "was never ported
   ... REBUILT on main in its own lane" against the 2026-10-15 amendment; only dev seeds 15000-15007 are exposed.
5. **T114 G6.** After T103 and T113. A4: "Overall disposition remains BLOCKED. The ABSENT deciding integration will be
   rebuilt on main in its own build arc", covering parts (a)-(d).
6. **T117 G9.** No pre-registration exists (no evidence file for G9). Some plumbing does exist: `Brain._g9_apply_source`,
   `_g9_context` and `_g9_reconcile`, cited in `g3-growth-loop.md` 2.1-2.3. The campaign itself needs:
   - its pre-registration;
   - the harness;
   - ten arms at S = 8;
   - the fresh-process rebuild;
   - the folded drift criteria 1-2.

## 5. Rows G9 does not need (for the owner; none is recommended for deletion)

**A. Close as carried, or an owner act.** The row's own text already records the disposition, and nothing remains to
build for the row as written:

| row | what remains |
|---|---|
| T120 | one owner act (move the file). The code landed. |
| T076 | the mechanism withdrawn and retired by owner decision; the exit row is "WITHDRAWN with the mechanism". |
| T064 | the deliverable void; the surviving capability ported. |
| T065 | the binding-width clause VOID; PARTIAL certified (s2-exit). |
| T078 | STOPPED AT G3 by its own registered branch; "the successor row owes the brain-side stimulus port". |
| T093 | the append half built; the persistence half "owes a named carrier". *My inference:* the P4b learner-persistence draft (`cloud/persist-learners-prereg-1028`) is such a carrier for four learners. |

**B. Self-improvement (US4):** T072, T073, T074, T075 and T083 (T076 and T078 are in A). These are candidates to
defer past G9, unless section 3.1 is read literally.

**C. Workstreams the plan itself says block nothing:**
- N: T096-T100.
- A: T105-T107, with T102 as A's root.

**D. Instruments and sweeps:** T088, T089, T084, T085, T086, T051c-wave.
- T051c-wave has no deadline by design: "advances opportunistically ... ratchet-policed". It cannot close by itself;
  it needs either a closing condition or an owner disposition.

**E. Owner-started research, running on the same host:** T118, T121, T122, T123.
- None blocks G9.
- **They compete with it for the host.** T121's row says "Runs queue behind T122, one at a time" (delegate D-13).
- E1c's campaign has a cap of 704 runs at 11 looks.

**F. T087,** the close-out. It is the last row under the owner's definition, after G9.

## 6. Effort on the minimal path, in arcs

**What an arc is:** build + panel + one battery + push, about one day here (measured: 0.87 battery-green pushes a
day, 2026-10-06 to 10-28).

**What the history says.**
- G1(c), one campaign row, took 2026-09-25 to 10-06: about 11 days, with several arcs.
- G4 and G7 each landed as one stacked batch (batch 3, 2026-10-10).

So a campaign row costs about 3-12 arcs, depending on how much instrument is missing. The ranges below are *my
inference* from that and from each row's own owed list.

| row | low | high | the reason | the cost is unknown until |
|---|---|---|---|---|
| step 0, the Q-AII repair | 3 | 10+ | pre-register, build, measure; it can return a null (P8) | **UNKNOWN.** One dev-seed pilot measuring A_ii learnability with each of the three interacting mechanisms ablated in turn would bound it. |
| T110 | 2 | 4 | item (ii)'s taught campaign, the results record, O11 | -- |
| T111 + T111a | 6 | 12 | the largest UNVERIFIED list on the path: runners, C9-C19 reds, feed control at N=256, night costs; T111a is 1-2 arcs inside | **UNKNOWN at scale.** A6 "UNVERIFIED scale/cost": the real N=256 feed control. Measure one timed N=256 seed. |
| T113 (A-D) | 4 | 8 | D1/D2, the campaign instrument (A4 says absent), the confirmatory read | -- |
| T103 | 3 | 5 | the instrument rebuild, the dev read, the confirmatory read | **PARTLY UNKNOWN.** The rebuild lane's state is on a branch I cannot see. |
| T114 | 5 | 10 | the deciding integration (a)-(d) is absent; BLOCKED | -- |
| T117 | 6 | 12 | no pre-registration and no harness; ten arms at S = 8; the fresh-process rebuild | **UNKNOWN wall-clock.** One timed engine arm on one seed on the owner host would fix the campaign's compute cost. |
| **total** | **29** | **61+** | | |

**Host time is a separate cost.** Campaigns run one at a time on the owner host (delegate D-13, cited in T121). Every
G campaign queues behind whatever holds the host. Today that is E1c (T122), then L1b-STALE (T121).

## 7. Timeline

**Low case: about 35-45 days, so mid-December 2026.** It assumes:
- the owner points every arc at the chain after E1d;
- step 0 works on its first pre-registration;
- each row comes in at its low estimate;
- T103 runs alongside steps 1-3;
- T113 closes without Part E;
- section 3.1 is read as "promotion only";
- no new row is added on the path;
- E1c and L1b-STALE either finish first or yield the host.

**High case: about 75-90 days of dedicated arcs, so late January to February 2027.** It assumes:
- step 0 takes two pre-registrations;
- the rows come in at their high estimates;
- the E1c and L1b-STALE campaigns hold the host first;
- section 3.1 is read literally, which adds US4's T072-T074 at perhaps 10-20 arcs more.

**At the observed share** (1 of 20 arcs on the chain since 2026-10-06), neither case is reached. The date depends on
the owner's priority order, not on throughput.

**The owner's definition (all rows) adds the 31 off-path rows.**
- Group A (6 rows) is mostly owner dispositions, not arcs.
- B, C and D are each multi-row workstreams: N alone is five campaign-scale rows.
- *My inference:* "all rows" is not reachable this year at today's rates unless groups B to D are deferred or closed
  as carried by the owner.

### 7.1 The row-growth problem

Since 2026-09-15, rows were added at about 3.8 a week and closed at about 2.1. **17 of the 24 additions were the
owner's own plan expansions**, so a process rule alone cannot fix it. A rule that would stop the open count rising:

1. **One in, one out.** A row may be opened only if, in the same commit, one of three things holds:
   - it is on the G9 minimal path;
   - an open row is closed, retired or carried by the owner;
   - the owner explicitly accepts the net rise.

   tasks.md's own discipline line ("one commit ...") would carry it.
2. **Panel-found debt goes INTO the row it was found in,** as an owed item, unless it blocks G9. T111a, T120 and T123
   were each a row of their own.
3. **Owner-started research rows** (T118, T121, T122) get either an explicit G9-priority slot or a deferral date when
   they are opened. They then cannot silently take the host from the path.
4. **A row with no closing condition** (T051c-wave) gets one, or an owner disposition.

## 8. For the owner (each is a decision this file does not make)

1. Does "no bypass" (section 3.1) bind G9's scratch-brain campaign?
2. Does T113 close on Parts A-D, with Part E carried to G9's harness?
3. Should the Q-AII stack repair (section 4, step 0) become a tasks row on the path?
4. The priority order on the one host: the G9 chain against E1c, L1b-STALE and OG1.
5. For groups A-D in section 5: defer, retire or close as carried, row by row.
6. The row-growth rule in section 7.1, or another.
