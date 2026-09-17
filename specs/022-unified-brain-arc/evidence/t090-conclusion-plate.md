# T090 (C5) -- the conclusion plate: every charged reading WITH its licensed and refused sentences IN VIEW

Pre-registered 2026-09-16 (sections 1-2.9 frozen at the commit that lands this file; any later amendment is dated and
appended). The C5 row (PLAN-FINAL Phase 4; `PROPOSED-ROWS-2026-09-06.md` C5; tasks.md T090; Part F C-R5) is BUILT as
a registry and a gate (`substrate/conclusion_protocol.py`, 2026-09-06) and every C4 row now carries its disposition --
but the row's own conjunct on the cockpit, "prints it as a consciousness row WITH the licensed and refused sentences in
the same view", is NOT MET AS RENDERED on all three charged rows (T080 record 3.5 item 11; T081 3.5 item 21; T082 3.7).
This arc lands that conjunct and closes T090's checkbox. It is a SURFACE arc: no instrument, no seeds, no t.

## 1. Territory, measured before the design (2026-09-16, head `633dad1`, the live daemon's page in the built-in browser pane)

1. THE PLATE IS A CELL OF A FIXED-HEIGHT GRID ROW. `substrate/cockpit.py`: `#app` is `grid-template-rows: 28px 1fr 110px`;
   `#bottom` is eight equal columns with `#bottom>div{overflow:hidden}`; the ACCESS plate (`#bAcc`) is one of them.
   Measured with the panel's measurement script (`scratchpad/t082-panel/refute/measure_access_plate.js`, re-run by the
   orchestrator): at the pane's 1420 x 1321 the plate is 170 x 110 px (scrollHeight 317); at an emulated 1920 x 1080 it
   is 232 x 110 (scrollHeight 286). Of its seven rows only `ignitions` (top 75, bottom 94 px) is visible; `percept holds
   bus` (94-113), `PCI` (113-132), `integration` (132-151), `cross-modal bound rate` (151-169), `report-order margin`
   (169-188) and `social false beliefs` (188-207) are clipped at BOTH widths.
2. THE VALUE SPAN IS NOWRAP AND CARRIES THE WHOLE PROVENANCE. `row()` renders `${fmt(v)}${tag(pk)}` inside `.v`, whose
   rule is `.row .v{white-space:nowrap}`; `tag()` injects the FULL `PROVENANCE[k]` string as a `.tag` span. Tag lengths:
   `pci` 516, `integration_competence` 22, `crossmodal_bound_rate` 1,806, `report_order_margin` 3,352,
   `recursive_tom_false_beliefs` 4,316 characters; the value spans measure 2,953 / 165 / 10,330 / 19,139 / 24,606 px at
   both widths; visible tag characters 0 on every one. The conjunct fails on both axes (T082 record 3.7).
3. THE SENTENCES REACH THE PAGE ONLY AS SOURCE. The licensed and refused sentences of the three charged rows sit in JS
   comments above each `row()` call (cockpit.py:925-944) and inside the provenance strings (T082's at characters 1,118
   and 1,230); no rendered element carries them.
4. THE GATE READS SOURCE TEXT. `conclusion_protocol.surface_findings()` scans TRACKED_SURFACES (cockpit.py, OVERVIEW,
   PITCH, README) line by line: a symbol on a line with the licensed sentence's first six words within three lines is
   a credit, without them a debt; `SURFACE_BASELINE` = 1 (integration's reading on the PCI row, a T070 debt; may fall,
   never rise). It is satisfied by the comments of fact 3 and cannot see rendering (obs #2195); it is also TRIPPED by a
   symbol named in a comment (obs #2199).
5. THE REGISTRY TODAY. `CONCLUSION` rows with a reading: `temporal_binding` 0.2309 (`crossmodal_bound_rate`),
   `report_order` 1.0 (`report_order_margin`), `social_cognition` 0.4437 (`recursive_tom_false_beliefs`); `integration`,
   `metacognition`, `agency` None (T070 / T077 / T079 NOT MET), `attention_schema` None (T078 `[ ]` by its G3 STOP). A
   NOT MET row prints nothing (the standing refusal).
6. THE TELEMETRY HAS NO CONCLUSION SECTION. `brain_telemetry.snapshot()` ends with `out["provenance"] = dict(PROVENANCE)`;
   `verify_contract()` checks `DASHBOARD_KEYS` over cognition | derived and `DASHBOARD_SECTION_KEYS` per section; the
   page reads `/api/all` at 1 Hz (`PROV = d.provenance`). The chat's reply (T081's step 7) already prints every charged
   row's sentences from `CONCLUSION` -- one registry, two consumers.
7. ROOM EXISTS. `#widgets` is a 460 px flex column, 910 px tall at 1920 x 1080 (`#widgets>div{overflow:hidden}`), holding
   six plates with flex 2 / 3 / 4 / 5 / 3 / 3.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT -- the account is under a WEEKLY usage cap until 2026-09-20 00:30

Pre-registered anyway, as the rhythm requires; the arc's closing report joins T078-T082's when the cap lifts.

### 2.1 The mechanism -- ONE registry, ONE served section, ONE plate that wraps

(a) `brain_telemetry.conclusion_rows(cognition)`: a pure function over `conclusion_protocol.CONCLUSION` returning, for
every row whose reading is not None, `{instrument, task, reading, licensed, refused, dissociation, symbols, live}` with
`live` = the cognition value of each surface symbol; a row with a reading but a missing sentence REFUSES (raises); a
NOT MET row is absent. `snapshot()` serves it as `out["conclusion"]` (on `/api/state` and `/api/all`);
`DASHBOARD_SECTION_KEYS` gains the section so `verify_contract` guards it. (b) The cockpit gains a CONSCIOUSNESS plate
in `#widgets` (after BEHAVIOR & EMOTION; flex 5) rendering each served entry as: the instrument and task, the live
value(s) with the charged reading, then the licensed sentence and the refused sentence in a `.c5` block whose rule is
`white-space:normal` (11 px), and a collapsed `<details>` holding the row's PROVENANCE string wrapped. (c) The three C5
rows and their comments LEAVE the ACCESS plate (P2: one place per reading; the ACCESS plate keeps `ignitions`, `percept
holds bus`, `PCI`, `integration`). (d) `surface_findings` gains a MECHANISM clause beside its text clause: over the live
registry, `conclusion_rows()` must carry both sentences for every charged row (a blank sentence is a finding); the text
clause keeps covering every tracked surface, the ACCESS plate's remaining rows and the other docs. Counted on today's
tree before the rule lands (2.9 step 3).

### 2.2 What is measured (one page, two viewports, no seeds)

The RENDERED geometry, read in the built-in browser pane on the live daemon's page by the measurement script committed
as `tests/instruments/t090_measure_plate.js` (the panel's script, extended to the new plate): per charged row, the
bounding boxes of the licensed sentence, the refused sentence and the live value against the plate's visible box, the
container's computed `white-space`, the `<details>` collapsed and expanded, at the pane's own width and at an emulated
1920 x 1080. Run by the orchestrator (a browser is not CI); the script and its printed output are the record's 3.x. The
barred X is "sentence characters fully inside the plate's visible box"; the disposition noun is the same.

### 2.3 The conjuncts and the verdict rule (written first)

(p)  SERVED: `/api/state` and `/api/all` carry `conclusion` equal to `conclusion_rows(CONCLUSION, cognition)` on the
     same snapshot (a test); `verify_contract` lists no orphan; else REFUSED.
(r)  READABLE AS RENDERED: the plate exists in the served HTML outside `#bottom`, its `.c5` container's COMPUTED
     `white-space` is `normal`, and the measurement script finds every charged row on it; else REFUSED (the apparatus).
(v1) IN VIEW: for EVERY charged row, at BOTH widths, every character of the licensed sentence and of the refused
     sentence lies inside the plate's visible box, and the row's live value lies in that same box -- a count over
     rows x widths, all of them; one clipped character is NOT MET.
(v2) THE REFUSAL HOLDS: no row with reading None appears on the plate (a count, 0), and no verdict word appears on it
     (the gate's VERDICT_WORDS); else NOT MET.
(g)  THE GATE: `conclusion_protocol --assert` rc 0, surface debt at or under its baseline (1), the mechanism clause
     green on the live registry and red on its constructed blank (R4); else NOT MET.
VERDICT: MET iff (v1), (v2) and (g) with (p) and (r) holding; NOT MET iff any fails; REFUSED iff (p) or (r) fails.
REPORTED, never barred: the plate's height used at each width, the `<details>` expanded geometry (the provenance
wrapped and scrollable within the plate), the ACCESS plate's remaining rows' visibility (unchanged debt: fact 1).

### 2.4 The reds (each runs on its constructed input; R1 is the orchestrator's rendered red, R2-R5 are tests)

R1  THE BEFORE READS ZERO: the measurement script on the ACCESS plate at head `633dad1` reads visible sentence
    characters 0 on every charged row at both widths (fact 2) -- the red the arc exists to turn.
R2  A CHARGED ROW WITHOUT ITS SENTENCE REFUSES: `conclusion_rows` on a constructed registry with a reading and a blank
    refused sentence raises; mutation `serve_anyway`: red.
R3  A NOT-MET ROW NEVER SERVES: a constructed registry with one charged and one None row yields exactly one entry;
    mutation `serve_all_rows`: red.
R4  THE GATE'S MECHANISM CLAUSE FIRES: a constructed registry whose licensed sentence is blanked is a finding; the live
    registry is not; mutation `text_only` (the clause removed): the blank passes: red.
R5  THE SOURCE RATCHET (declared a ratchet of text, never the rendered red): the served cockpit HTML places the
    CONSCIOUSNESS plate outside `#bottom` and carries the `.c5{white-space:normal}` rule; mutation `nowrap`: red.

### 2.5 The controls that are not the method

The BEFORE measurement (the same script, the same page, the ACCESS plate) at both widths; the second width itself (a
plate that fits one width and not the other is NOT MET); the `<details>` collapsed state (the sentences must be in
view WITHOUT expanding anything); the gate's text clause left in place (the mechanism clause is added beside it, never
in place of it).

### 2.6 Disposition (fixed in advance)

- MET: tasks.md T090 -> [X] with the two widths' numbers; a dated line on T080's, T081's and T082's surface notes
  (tasks.md, spec.md SC-009, PLAN-FINAL, WholeBrain.md): "the same-view conjunct MET AS RENDERED at <commit> (T090)";
  the C5 row's conjunct is read MET for every charged row; `SURFACE_BASELINE` unchanged (integration's PCI-row debt is
  T070's); the ACCESS plate's other clipped rows stay a REPORTED debt (opening).
- NOT MET: the plate stays as built (it prints only what is measured); the record names the width and the row; T090
  stays [ ]; nothing on the three rows' notes changes.
- REFUSED: the fault named ((p) or (r)); nothing lands on any surface.
In every branch the chat's reply is untouched (it already reads the registry) and no instrument, reading, symbol or
pin moves.

### 2.7 What this row does NOT claim

Nothing about consciousness (the plate prints readings and sentences; the refused sentence stands beside each); no
aggregate across rows; the PROVENANCE column stays the honesty column and is not shortened; the rendered measurement is
the orchestrator's browser run at two widths, not CI (CI carries R2-R5: the mechanism and the text ratchet); the other
clipped ACCESS rows (`percept holds bus`, `PCI`, `integration`) are not this arc's; the daemon's live values are what
they are (a chance binding rate, a negative margin, an empty belief list) and the plate says so through each row's
provenance.

### 2.8 Widths, cost, host

Two viewports: the pane's own (about 1420 x 1321) and 1920 x 1080 (emulated); the page at 1 Hz; the daemon restarts on
every commit (read after the restart, not before); the measurement takes under a minute per width. Host: the pinned
machine; the built-in browser pane.

### 2.9 Commit order (one change each; the section and tests before the plate; the count before the gate's rule)

1. This file + tasks.md T090's dated note + PLAN-FINAL C-R5's dated note -> 2. `brain_telemetry.conclusion_rows()` +
`out["conclusion"]` + the contract entry + tests R2/R3 -> 3. the gate's mechanism clause, COUNTED on today's tree first
(expected 0 findings on the live registry; R4's constructed blank 1) -> 4. the plate (HTML + CSS + JS; the three C5 rows
and their comments moved out of ACCESS; R5's test; `tests/instruments/t090_measure_plate.js`); census gates -> 5. the
orchestrator's rendered measurement at both widths (R1's BEFORE is fact 2; the AFTER is 3.x) -> 6. the surfaces per 2.6
-> 7. section 3 the record -> panel (3 lenses + refuters judging CLAIM and FIX) -> fold -> stacked with C0 for ONE
battery (the Codex closing report when the cap lifts, 2.0).

### 2.10 Amendments (dated, appended; nothing above is edited in place)

2026-09-16 22:33 local (clock read at write time), on the builder's report at `59141bf`, BEFORE the rendered measurement (2.9
step 5) is run:

(a) (p) NARROWED to `/api/all`. 2.1(a) and 2.3(p) wrote "`/api/state` and `/api/all`". The territory: `/api/state`
serves `snap["cognition"]` only -- the flat `Brain.state()` map policed by `DASHBOARD_KEYS` / `verify_contract` -- so a
list-valued `conclusion` there would put a non-cognition value inside the cognition contract; `/api/all` is what the
page reads at 1 Hz and what the plate reads. (p) now reads: `/api/all` carries `conclusion` equal to
`conclusion_rows(cognition)` on the same snapshot (identity by construction in `snapshot()`, pinned by
`test_a_scratch_brains_snapshot_serves_todays_three_charged_rows` and the contract test); `/api/state` carries no
`conclusion`. Measured on the live daemon at `59141bf` by the orchestrator: `/api/all` sections
`arm_view, body, cognition, conclusion, derived, device, gated, hardware, provenance, spikes, t`, 3 rows
(temporal_binding 0.2309, report_order 1.0, social_cognition 0.4437, registry order); `/api/state` no `conclusion`.
Not a relaxation of the verdict: (p) still REFUSES on a missing or unequal section.

(b) The contract entry lives in `verify_contract` as a required top-level key, not in `DASHBOARD_SECTION_KEYS` (which
lists dict-valued sections; `conclusion` is a list). The no-orphan test covers it (`..._is_a_contract_key_whose_absence_
is_a_finding`).

(c) DISCLOSED BEFORE THE NUMBER: the builder's arithmetic on the committed text PREDICTS (v1) NOT MET at both widths
(plate content about 325 px against about 223 px available at 1420 x 1321 and about 175 px at 1920 x 1080; the plate is
flex 5 of 25 in a column of viewport height minus 170 px; the load-bearing assumption is a 6.05 px monospace advance at
11 px). The layout is NOT adjusted before the measurement: 2.6's NOT MET branch applies as written if the prediction
holds; any layout change is a stacked commit AFTER section 3 records the first reading, and its re-measurement is a
disclosed SECOND reading, never the first.

## 3. The record

### 3.1 The FIRST rendered reading (orchestrator, 2026-09-16 22:40 local, clock read at write time; head `51e8031`, the daemon read AFTER its post-commit restart)

Apparatus as run, with its one deviation disclosed: the built-in browser pane was HIDDEN at read time (`innerWidth` 0),
so the "pane's own" width of 2.2 / 2.8 did not exist; BOTH viewports were emulated (`resize_window`): 1420 x 1321 (fact
2's pane size) and 1920 x 1080. The script is the committed `tests/instruments/t090_measure_plate.js`, pasted verbatim
into the page; its two outputs are kept beside the arc's scratch as `measure_first_1420x1321.json` and
`measure_first_1920x1080.json` (extracted byte-exact from the session transcript, never retyped).

| viewport | plate box (px) | plate content height (px) | charged row | live value in box | licensed chars in box | refused chars in box | `.c5` computed white-space | `<details>` |
|---|---|---|---|---|---|---|---|---|
| 1420x1321 | 420 x 174 | 324 | temporal_binding T080 | yes | 102 / 102 | 97 / 97 | normal | collapsed |
| 1420x1321 | 420 x 174 | 324 | report_order T081 | yes | 105 / 105 | 0 / 77 | normal | collapsed |
| 1420x1321 | 420 x 174 | 324 | social_cognition T082 | NO | 0 / 105 | 0 / 84 | normal | collapsed |
| 1920x1080 | 460 x 171 | 324 | temporal_binding T080 | yes | 102 / 102 | 97 / 97 | normal | collapsed |
| 1920x1080 | 460 x 171 | 324 | report_order T081 | yes | 62 / 105 | 0 / 77 | normal | collapsed |
| 1920x1080 | 460 x 171 | 324 | social_cognition T082 | NO | 0 / 105 | 0 / 84 | normal | collapsed |

Sentence characters inside the plate's visible box, all charged rows: 304 of 570 at 1420 x 1321;
261 of 570 at 1920 x 1080. Row x width cells with EVERY character and the live value in the
box: 2 of 6 (temporal_binding at both widths). Live values at read time (the daemon's, not campaign readings):
crossmodal bound rate 0.258 / 0.170, report-order margin 0.000 / 0.074, false beliefs 0 / 0.

Conjuncts, each quoted and marked:
- (p) SERVED (as narrowed by 2.10 (a)): HOLDS -- `/api/all` carries `conclusion` with the three charged rows in registry
  order on the live daemon; identity with `conclusion_rows(cognition)` by construction, pinned by the scratch-brain test
  (`tests/test_conclusion_plate.py`, 7 passed, reproduced by the orchestrator with the coverage gate off).
- (r) READABLE AS RENDERED: HOLDS -- the plate is on the page outside `#bottom` (`insideBottom` false at both widths),
  every `.c5` block's COMPUTED `white-space` is `normal`, the script found 3 of 3 charged rows.
- (v1) IN VIEW: **NOT MET** -- at 1420 x 1321 report_order's refused sentence (0 of 77) and all of social_cognition
  (value, 0 of 105, 0 of 84) lie outside the box; at 1920 x 1080 report_order's licensed sentence is cut at character 62
  of 105, its refused sentence is out (0 of 77), and all of social_cognition is out. One clipped character is NOT MET;
  here 266 and 309 are.
- (v2) THE REFUSAL HOLDS: HOLDS -- served rows 3 = charged rows 3 (no None row on the plate), verdict words found: 0.
- (g) THE GATE: HOLDS -- `conclusion_protocol --assert` rc 0, surface debt 1 at its baseline (integration's PCI row,
  T070's), the mechanism clause 0 findings on the live registry and 1 on its constructed blank (R4).

**VERDICT OF THE FIRST READING: NOT MET** ((v1) fails at both widths; (p) and (r) hold, so it is a NOT MET and not a
REFUSED). The builder's arithmetic (2.10 (c)) predicted this before the number was read: plate content 324 px measured
(predicted about 325) against 174 px and 171 px available (predicted about 223 and 175 -- the 1420-wide page has a fourth
grid row, the 220 px chat strip of the `max-width:1500px` rule, which the prediction did not count).

R1's turn, read by the same instrument: BEFORE (fact 2, `633dad1`) 0 visible sentence characters on every charged row at
both widths; AFTER, first reading: 304 and 261 of 570. The red turned for ONE row of three.

REPORTED, never barred:
- The cause is the column, not the plate. `#widgets` is a fixed-height flex column with `overflow:hidden`, and EVERY
  plate in it is clipped at both widths (content height against box height, px, at 1920 x 1080 / 1420 x 1321):
  ARM 105 vs 74 / 76; MOTOR MIX 145 vs 106 / 108; BEHAVIOR & EMOTION 180 vs 139 / 141; CONSCIOUSNESS 324 vs 171 / 174;
  HARDWARE TARGET 520 vs 171 / 174; NEUROMODULATORS 250 vs 106 / 108; WORLD MODEL 203 vs 106 / 108. Clipped content is
  unreachable: the column does not scroll.
- THIS ARC'S OWN COST on the six other plates: the new plate raised the column's flex total from 20 to 25, so every
  other plate lost about a fifth of its height at `59141bf` (no rendered BEFORE of those six exists; the figure is
  the flex arithmetic, 910 px column less gaps, disclosed as arithmetic).
- The ACCESS plate after the three C5 rows left it: 4 rows, 1 in view (`ignitions`); `percept holds bus`, `PCI`,
  `integration` out of view at both widths (fact 1's debt, unchanged in kind).
- `<details>` collapsed on every row at both widths (2.5's control: nothing was expanded to read the sentences).

2.6's NOT MET branch applies to this reading as written: T090 stays [ ]; nothing on T080's, T081's or T082's notes
changes. What follows (3.2) is a change to the ARTIFACT under an UNCHANGED (v1), pre-registered before it is built.
