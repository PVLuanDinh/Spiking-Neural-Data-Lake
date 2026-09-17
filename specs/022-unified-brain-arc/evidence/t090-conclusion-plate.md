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

### 3.2 PRE-REGISTRATION OF THE SECOND READING (2026-09-16 23:18 local, clock read at write time), frozen BEFORE anything is built

PROVENANCE AND NUMBERING OF THIS SECTION, disclosed. The text below is the synthesis of a design panel run by the
orchestrator after 3.1 (three independent Opus designers -- column-scroll, plate-priority, relocate -- each attacked
by two refuters, one on the claim and one on the instrument, then one synthesis; 10 agents, read-only on the tree).
All six refuter verdicts came back `survives: false` as filed; the synthesis adopts the column-scroll mechanism WITH
its refuters' minimal fixes grafted (scores 8 and 7.5 against 4 / 4 and 6 / 5) and rejects the other two on measured
facts. Its numbers are DATA: the orchestrator re-checked the source cites it rests on against the tree at `d6008d2`
(`body{overflow:hidden}`; the `#widgets` and `#widgets>div` rules; the seven inline `flex:N` plates in DOM order ARM,
MOTOR MIX, BEHAVIOR & EMOTION, CONSCIOUSNESS, HARDWARE TARGET, NEUROMODULATORS, WORLD MODEL; the `#wC5` fill assigned
inside `paint(d)` under `setInterval(loop,1000)`), and every PREDICTED figure below is a prediction the two rendered
runs either reproduce or contradict. The subsections 3.2.0-3.2.6 are this pre-registration's own; the RESULTS land
after it as 3.3 (the extended instrument's continuity run and its reds R7, R7b, R7c, R8 at the UNCHANGED page) and 3.4
(the second rendered reading with R6a and R6b). APPARATUS NOTE: where the first reading PASTED the committed script
into the page, the extended script is loaded as its committed BYTES (a script element pointing at a local read-only
static file server over `tests/instruments/`, the entry point then called from the pane); the page gains one script
node in `head`, layout-neutral, and the pasted form stays the fallback -- disclosed whichever is used.

The FIRST reading (3.1) read (v1) NOT MET at both widths and 2.6's NOT MET branch applied as written:
T090 stays `[ ]`, nothing moved on T080's, T081's or T082's notes. 3.1 also named the cause and it is
not the plate: `#widgets` is a fixed-height flex column with `overflow:hidden`, EVERY plate in it is
clipped, and the column does not scroll, so 868 px (1920 x 1080) / 852 px (1420 x 1321) of rendered
content is unreachable. What follows is a change to the ARTIFACT under an UNCHANGED criterion,
pre-registered before it is built, and read as a disclosed SECOND reading per 2.10 (c) - never as a
re-take of the first.

#### 3.2.0 THE CRITERION IS (v1), UNCHANGED, QUOTED FROM 2.3

> (v1) IN VIEW: for EVERY charged row, at BOTH widths, every character of the licensed sentence and
> of the refused sentence lies inside the plate's visible box, and the row's live value lies in that
> same box -- a count over rows x widths, all of them; one clipped character is NOT MET.

2.5's control stands with it: the `<details>` stays COLLAPSED - the sentences must be in view WITHOUT
expanding anything. Not one word of (v1) is amended, narrowed or supplemented. Nothing below weakens
it; everything below either tightens the INSTRUMENT that reads it or REPORTS a quantity beside it.

#### 3.2.1 THE CHANGE (one mechanism, P2; no flag, no second plate, no second layout)

`#widgets` becomes a scroll container (`overflow-x:hidden;overflow-y:auto;scrollbar-gutter:stable`);
every plate's inline `flex:N` becomes `flex:0 0 auto`, so a plate's height is its own content's
height and no plate can be squeezed below it; and the CONSCIOUSNESS plate moves to be the FIRST child
of the column. The third part is what makes the geometry follow from the content rather than from the
page: at the scroll origin the plate's top offset is 0 by construction, so its visible box is
intersect([0, H_c5], [0, H_column]) and (v1) reduces to ONE inequality over ONE plate's own content,
H_c5 <= H_column, with no term contributed by any other plate's content, wrap or number. No number is
tuned: the grow factors are REMOVED rather than repurposed (grow can only fire on a viewport of
1947 px / 2173 px and therefore never fires here, and a factor that never fires is a number a later
reader would tune to no effect), and the one environment-dependent quantity, the scrollbar gutter, is
shown below to be one the criterion is insensitive to across 0-46 px and is MEASURED and printed
rather than assumed. Zero JS changes: the fill at cockpit.py:884 and the comment block at :855-881
are untouched, so the surface gate's text credit and `SURFACE_BASELINE` = 1 are untouched by
construction. The `@media(max-width:1500px)` rule is untouched; one mechanism serves both layouts.

#### 3.2.2 THE INSTRUMENT EXTENSION - STRICTLY HARDER, NEVER DIFFERENT

The committed `tests/instruments/t090_measure_plate.js` tests each character against
`plate.getBoundingClientRect()` alone. Five extensions, each making it harder to read a character as
IN VIEW, each owing its own red:

(E1) THE STRICT VISIBLE BOX. The census runs against intersect(the plate's border rect, the CLIENT
(padding) box of EVERY ancestor to the root whose computed `overflow-x` or `overflow-y` is not
`visible`, the viewport). Client boxes, not border boxes: a box with overflow != visible clips at its
padding edge, and its scrollbar gutter lies outside that edge. The chain actually walked is printed
by selector with each member's rect and computed overflow - data, not an enumeration assumed in
advance. Because it starts from the border rect and only intersects, the strict box is a SUBSET of
the committed box by construction.
   TWO HONEST NOTES, stated rather than claimed. `#app` sets no overflow and is not a clipper. `body`
   (cockpit.py:512) has COMPUTED overflow `hidden` but its USED value is `visible` - the root
   propagates body's overflow to the viewport - so including it costs nothing here, its client box
   coinciding with the viewport; it is kept because a future non-100vh layout is the case it guards.
   And the VIEWPORT term can never fire alone on this page (44+910 = 954 < 1080; 44+925 = 969 < 1321,
   and a short viewport shrinks the 1fr column before it can clip it). It is kept because it costs
   nothing and it is explicitly NOT presented as a conjunct with a red of its own.

(E2) SCROLL IS A PRECONDITION, NOT AN ASSUMPTION. Before measuring, the scrollTop/scrollLeft of every
chain member and of `document.scrollingElement` are read; any non-zero value REFUSES the reading with
no numbers. Browsers restore a scroll container's offset across reloads, and a silent wrong number
here would have been indistinguishable from a pass.

(E3) THE COLLAPSED `<details>` IS A PRECONDITION TOO. 2.5 gives the collapsed state the same standing
as the second width; it now gets the same refusal. An open `<details>` on the plate REFUSES with no
numbers, where the committed script reported it as a field.

(E4) INVISIBLE TEXT IS NO LONGER IN VIEW. The committed census counts a zero-area character as
visible ("a collapsed space") WITHOUT checking that it is whitespace, so every character of a
`display:none` sentence counts as in view; and `visibility:hidden` preserves layout, so its rects are
real, correctly positioned, and pass even the intersected test. The barred quantity was satisfiable
by text nobody can see. The zero-rect branch is restricted to actual whitespace, and the census
REFUSES a sentence whose parent's computed `visibility` is not `visible`, whose `opacity` parses to
0, or whose `font-size` parses to 0.

(E5) ONE BOX FOR ONE CRITERION. `valueInside` is routed through the SAME strict box as the sentences.
(v1) bars both in the same box; after the change the plate's box IS its content, so the committed
plate-only test for the value becomes unconditionally true, and today it could not produce a false
MET only because `row()` happens to render the value ABOVE its own sentences - an accident of row
order, not a property of the design.

THE EXTENSION'S OWN RENDERED REDS:
  R6a THE ANCESTOR TERM, TOTAL (after the change; a disclosed page mutation, never folded into a
      measurement run). Set `#widgets.scrollTop = 400`. The plate spans column offsets 0..326, so it
      is entirely above the clip: the STRICT census must read 0 of 570 sentence characters and 0 of 3
      live values, while `insideCharsPlate` still reads 570 - the plate's own rect scrolls with its
      content. Restore scrollTop 0.
  R6b THE ANCESTOR TERM, ISOLATING, AT scrollTop 0, ON THE MEASUREMENT RUN ITSELF, WITH NO MUTATION.
      R6a is not an attribution: at scrollTop 400 the plate is outside the clip AND outside the
      viewport, so two terms fire together. HARDWARE TARGET spans column offsets 786..1308 against a
      column clip bottom of 910 (viewport y 954) and a viewport bottom of 1080, so its rows between
      y 954 and 1080 are inside their own plate's box AND inside the viewport AND outside the
      ancestor clip. `rowsInPlateOnly - rowsInView > 0` on that plate is the ancestor term firing
      ALONE. If it reads 0, the extension is not measuring the ancestor clip and the reading is void.
      DISCLOSED: this red is reachable only AFTER the change - today every plate box lies inside the
      column, so the ancestor clip cannot cut what the plate box does not already cut.
  R7  THE SCROLL PRECONDITION'S RED (reachable at the UNCHANGED head): run the entry point at
      `#widgets.scrollTop = 400` -> `{refused:'scrolled: ...'}` and no numbers. Mutation
      `trust_the_scroll` (the check deleted): numbers printed from a scrolled page: RED.
  R7b THE `<details>` PRECONDITION'S RED (unchanged head): in one synchronous expression set a
      `#wC5 details.open = true` then call the entry point -> refused. Disclosed as a constructed
      mutation; the 1 Hz rewrite closes it on the next tick. Mutation
      `details_reported_not_refused`: RED.
  R7c THE INVISIBLE-TEXT RED (unchanged head): set one `.c5` block's inline `visibility:hidden` and
      re-run. The COMMITTED plate-only census reports all its characters INSIDE; the repaired census
      REFUSES that sentence with 0 inside. Restore. This is the red the instrument that took the
      FIRST reading could not turn, and it is why this extension is stricter rather than merely
      different.
  R7, R7b and R7c all run at the UNCHANGED page, BEFORE cockpit.py is touched. That is this arc's
  red-first: the instrument is proved able to fail before the artifact is changed.
  R9 `no_scroll`, R10 `no_gutter`, R11 `shrinkable`, R12 `reorder` are SOURCE ratchets in
  tests/test_conclusion_plate.py, declared ratchets of text exactly as R5 is, never rendered reds.
  R1-R5 are unchanged and still required. R5 is untouched and still passes: the plate stays outside
  `#bottom`, `.c5{white-space:normal` stays, the three symbols stay out of `row()` calls,
  `d.conclusion` stays.

#### 3.2.3 THE CONTROLS THAT ARE NOT THE METHOD

1. R8 THE CONTINUITY CONTROL, DIRECTIONAL AND ATTRIBUTED. The extended script is run FIRST against
   the unchanged page (the head that lands the instrument touches only .js and .md, so the page is
   byte-identical to `51e8031`'s). `insideCharsPlate` - the committed predicate, code path unchanged -
   must reproduce 3.1 EXACTLY: 304 of 570 at 1420 x 1321 and 261 of 570 at 1920 x 1080. `insideChars`
   must satisfy insideChars <= insideCharsPlate on EVERY sentence, and every non-zero difference must
   be itemised and attributed to a NAMED box term (the plate's border-vs-padding edge, an ancestor
   clip, the viewport). A stricter instrument is entitled to read lower; it is not entitled to read
   lower for an unnamed reason, and it may never read HIGHER. This control is written in this form
   ON PURPOSE: the extension tightens the plate's own edge by 1 px per side, `report_order`'s
   licensed line 2 at 1420 x 1321 sits within about 1 px of that edge, and 105 of the recorded 304
   hang on it - an exact-reproduction bar would have failed on the extension working correctly.
2. THE SECOND WIDTH, as 2.5 wrote it: a plate that fits one width and not the other is NOT MET.
3. THE `<details>` COLLAPSED STATE - now a CHECKED precondition (E3) with its own red, rather than a
   reported field.
4. THE ACCESS CENSUS, byte-identical in shape, at both widths, before and after. Its debt (4 rows,
   1 in view) is fact 1's and is unchanged in kind; if it MOVES, something touched `#bottom` that
   should not have and the reading is void.
5. THE SIX OTHER PLATES, measured by the SAME script at both heads and both widths - the P3/P4c
   ledger below is a count, not arithmetic.
6. THE GATE re-run after the change: `conclusion_protocol --assert` rc 0 with surface debt at or
   under its baseline of 1. cockpit.py is a TRACKED_SURFACE and the gate reads SOURCE TEXT, so the
   new comment is a gate input: it names no registry symbol in prose, including the three-character
   substring in `integration`'s symbol tuple, and no VERDICT_WORD. The gate is RUN, not reasoned
   about; a rise above 1 is a RED that blocks the commit, not a warning.
7. THE PREDICTED GEOMETRY BELOW IS WRITTEN BEFORE THE READING, as 2.10 (c) did, so a wrong model
   shows up as a residual rather than as a story.

#### 3.2.4 PREDICTED GEOMETRY (arithmetic on a validated model, disclosed as arithmetic)

The flex model reproduces ALL FOURTEEN measured box heights to the pixel (base = padding-bottom 8 +
2 x 1 px border = 10 under `*{box-sizing:border-box}`; free = column - 36 gaps - 70; share = free/25;
at 910: 74.32/106.48/138.64/170.80/170.80/106.48/106.48 = 874.0; at 925:
75.52/108.28/141.04/173.80/173.80/108.28/108.28 = 889.0). The line-box model reproduces the measured
scrollHeight 324 to 0.1 px (`.row` 12 x 1.4 + 2 = 18.8; `.c5` = 2 + n x 14.85 + 1 + 13.5 + 2; h2 26;
plate padding 8 -> 3 x 96.7 + 34 = 324.1). Everything below rides those two.

Content border boxes: C5 326, ARM 107, MOTOR 147, BEHAVIOR 182, HARDWARE 522, NEUROMOD 252,
WORLD MODEL 205 = 1741 + 36 gaps = 1777 px against a 910 / 925 px column: the column scrolls and no
plate grows or shrinks. Offsets after the move: C5 0..326, ARM 332..439, MOTOR 445..592,
BEHAVIOR 598..780, HARDWARE 786..1308, NEUROMOD 1314..1566, WORLD MODEL 1572..1777.
THE BARRED INEQUALITY: 326 <= 910 (margin 584) and 326 <= 925 (margin 599); the plate occupies
viewport y 44..370 at both widths, inside 1080 and 1321. PREDICTED (v1): 570 of 570 sentence
characters and 3 of 3 live values in view at both widths, 6 of 6 row x width cells complete.

GROWTH, as an inequality rather than a hope. H_c5 = 26 + n x entry + 8 (+2 borders):
n=4 -> 422.8; n=5 -> 519.5; n=7, the full registry and the design's real ceiling -> 712.9, margin
197 / 212. REACHABLE PESSIMISTIC (word-boundary wrap adding one line to every sentence,
111.55 px/row): n=7 -> 816.85, margin 93 / 108 - PASSES at both widths. The all-three-line case
(126.4 px/row -> 920.8, which would fail at 910) is NOT REACHABLE with the measured registry: three
lines need a rendered line over 125.2 chars at 1420 and over 138.6 at 1920 (at a 15 px gutter), and
the MEASURED maximum over all seven rows, charged and not, is 120 characters. The residual is a
future arc writing a longer sentence, and the instrument prints H_c5, the column height and the
per-sentence LINE COUNTS every reading, so the margin shrinks visibly before it fails.

THE GUTTER IS MEASURED, NOT ASSUMED. Text width = plate - 2 border - 24 padding. At a 15 px gutter:
419 px (1920) and 379 px (1420), i.e. 69.3 and 62.6 characters at 6.05 px. The longest line rendered
today is 115 characters; two lines need >= 57.5 chars/line = 348 px of text width, i.e. a gutter
<= 46 px at the 420 column. Every realistic gutter (10-17 px) clears that by 29-36 px, so the gutter
cannot flip the verdict - but the record PRINTS the column's measured clientWidth and each plate's
measured width at both viewports instead of inheriting this assumption, and prints the measured
character advance instead of 2.10 (c)'s never-measured 6.05.

#### 3.2.5 REPORTED, NEVER BARRED

1. EVERY PLATE'S CONTENT-VS-BOX, BEFORE AND AFTER, at both widths, read by the SAME extended script.
   BEFORE (3.1's figures, re-read at the instrument's head): box / scrollHeight / clipped-away px,
   all seven; total clipped 868 px at 1920 x 1080 and 852 px at 1420 x 1321, unreachable because the
   column does not scroll. AFTER: box, scrollHeight, clientHeight, `clipped` (which must read false
   on all seven), top offset in the column, visible px at scrollTop 0, and rowsTotal / rowsInView /
   rowsInPlateOnly.
2. THE AT-A-GLANCE LEDGER, BOTH HALVES, IN NUMBERS (P3/P4c: the brief forbids making any other
   plate's content less reachable without saying so and measuring it). REACHABILITY RISES FOR ALL
   SEVEN: 859 of 1727 content px reachable at 1920 and 875 of 1727 at 1420 today, 1727 of 1727 at
   both widths after, and four plates complete with no interaction where none was complete before.
   VISIBILITY AT scrollTop 0 IS A MIXED LEDGER AND THE LOSSES ARE REAL (box px, 1920 / 1420):
   CONSCIOUSNESS 171->326 / 174->326 (complete), ARM 74->107 / 76->107 (complete), MOTOR MIX
   106->147 / 108->147 (complete), BEHAVIOR & EMOTION 139->182 / 141->182 (complete), HARDWARE
   TARGET 171->124 / 174->139 (124 and 139 of 522), NEUROMODULATORS 106->0 / 108->0 (below the
   fold), WORLD MODEL 106->0 / 108->0 (below the fold). Column total visible 873->886 (1920) and
   889->901 (1420): essentially unchanged, redistributed. THREE PLATES LOSE AT-A-GLANCE VISIBILITY
   and two of them lose it entirely, in exchange for being fully reachable for the first time. Named
   in particular because it is the page's own honesty ledger: WORLD MODEL is where the
   not-computed / not-applied readouts render, so the surface's report of what it does NOT compute
   is the first thing to leave view at scrollTop 0.
3. THE SCROLL POSITION. `scrollTop` and `scrollLeft` of every clipping ancestor and of
   `document.scrollingElement`, printed at every reading even when green - the precondition's
   evidence, not its assumption. Both must be 0 at both widths or the reading is REFUSED.
4. THE COLUMN: clientHeight, clientWidth (the measured scrollbar gutter), scrollHeight, computed
   overflow-x/overflow-y, read twice one second apart. Under `flex:0 0 auto` the column's
   scrollHeight now VARIES at 1 Hz where the flex arithmetic used to pin it: `#wWm` ends with a map
   over the gated readouts (a variable row count), `#wHw` renders filtered torque rows and sliced
   recipes and is EMPTY until the first snapshot carrying joints (then jumps about 520 px). This
   cannot move the barred plate - nothing is above it - but it is a new behaviour of a daily page
   and it is measured, not narrated.
5. WRAP, MEASURED: the line count of every sentence at both widths, and the measured character
   advance. The predicted "no re-wrap" is a prediction; the record carries the reading.
6. THE INLINE AXIS, MEASURED AND NOT FIXED: every `.c5` block's scrollWidth vs clientWidth. The
   block-axis claim "the box is its content" does not extend to the inline axis - the plate's width
   is fixed by the column and it keeps `overflow:hidden`, so an unbreakable token longer than the
   measure would be clipped horizontally. Unreachable with today's prose (longest word about 12
   characters against a 62-character measure) and caught loudly by (v1) if it ever happens, so it is
   MEASURED now rather than fixed by a declaration that can never fire. OPENING, with its fix:
   `overflow-wrap:anywhere` on the `.c5` rule, to be landed by the arc that first writes a sentence
   carrying such a token - that arc has a real red.
7. THE PROVENANCE COLUMN IS STILL UNREACHABLE, AND "100% REACHABLE" MEANS PLATE-SCROLL CONTENT ONLY.
   `$('wC5').innerHTML` is assigned unconditionally inside `paint(d)`, which runs under
   `setInterval(loop,1000)`, so every `<details>` is destroyed and recreated CLOSED once a second and
   its 1,806 / 3,352 / 4,316 characters cannot be held open or scrolled by a human. This is
   pre-existing, is not made worse by this arc, and is NOT counted in the reachability figures above.
   It is entered here as the one C5 reading that is rendered and unreachable, and named as an
   OPENING with its fix (re-apply `open` by row key after the innerHTML assignment) - a JS change to
   a different function, deliberately not smuggled into a layout arc. It also makes 2.3's REPORTED
   item, the `<details>` expanded geometry, unmeasurable without a declared mutation, and that is
   said here rather than quietly dropped.
8. The `<details>` collapsed on every row at both widths (now the precondition's output, not a
   reported field), and the ACCESS census unchanged.

#### 3.2.6 DISPOSITION (fixed in advance)

MET requires ALL of: the orchestrator's RENDERED measurement, at BOTH widths, on a FRESH load, with
both preconditions green (every scrollTop 0, no `<details>` open), reading `insideChars` = chars on
EVERY licensed and EVERY refused sentence of EVERY charged row (570 of 570 at each width),
`valueInside` true on all three rows at both widths, 6 of 6 row x width cells complete, AND (v2) and
(g) re-read and holding ((v2): served rows = charged rows, verdict words 0; (g): `--assert` rc 0 with
surface debt at or under 1 and the mechanism clause green on the live registry and red on its
constructed blank), with (p) and (r) holding. A source ratchet, a flex prediction, an arithmetic
margin and a green pytest are NOT the measurement and cannot substitute for it.
- MET: `tasks.md` T090 -> `[X]` with BOTH widths' numbers AND the first reading's numbers beside
  them; the dated line on T080's, T081's and T082's surface notes (tasks.md, spec.md SC-009,
  PLAN-FINAL, WholeBrain.md): "the same-view conjunct MET AS RENDERED at <commit> (T090, second
  reading)"; `SURFACE_BASELINE` unchanged; the ACCESS plate's other clipped rows stay a REPORTED
  debt; and the three plates that lost at-a-glance visibility go on the record as a NAMED OPENING in
  the numbers of 3.2.5 item 2, never softened into "everything is now reachable".
- NOT MET: the record names the WIDTH, the ROW and the CHARACTER INDEX, and the layout change is
  REVERTED IN FULL (one revert commit backing out the CSS, the two HTML commits and the ratchet).
  T090 stays `[ ]`; nothing on the three rows' notes changes. The reachability gain measured in
  3.2.5 item 2 does NOT buy the change a reprieve: it is a different claim, and if it is worth
  keeping it returns as its own arc with its own pre-registered criterion. A null deletes the
  change; that is the licence working.
- REFUSED: if either precondition fires, or (p) or (r) fails, the reading is REFUSED, NO numbers are
  reported, and the run is re-taken on a fresh load. A refusal is never a NOT MET and never a MET,
  and a reading is never re-run until it comes out green - the refusal names its fault first.
THE VERDICT LINE WILL CARRY BOTH READINGS AND WHAT CHANGED BETWEEN THEM, in this shape:
  "FIRST READING (3.1, head `51e8031`): (v1) NOT MET - 304 of 570 sentence characters at
   1420 x 1321 and 261 of 570 at 1920 x 1080, 2 of 6 row x width cells complete, the cause being a
   fixed-height `#widgets` column with `overflow:hidden` in which every plate was clipped and
   nothing scrolled. SECOND READING (3.4, head `<commit>`): (v1) <MET|NOT MET> - <numbers>. WHAT
   CHANGED: the ARTIFACT (the column scrolls, each plate is its own content's height, the plate owns
   the scroll origin) and the INSTRUMENT (strictly harder: the strict visible box, two preconditions
   that refuse, and invisible text no longer counted as in view). THE CRITERION DID NOT CHANGE -
   (v1) is quoted verbatim at 3.2.0 and the committed sub-measurement reproduced 3.1's 304 and 261
   bit-exact at the unchanged page before anything was built."
In every branch the chat's reply is untouched, and no instrument, reading, symbol or pin moves.

#### 3.2.7 Amendment (2026-09-16 23:38 local, clock read at write time; BEFORE any red of 3.2.2 is run; on the instrument builder's report at `fff1a57`)

R7'S SCROLLER AT THE UNCHANGED HEAD IS THE PLATE, NOT THE COLUMN. 3.2.2 wrote R7 as `#widgets.scrollTop = 400`. At the
unchanged head `#widgets` does not overflow -- its seven flex children fill it exactly and each clips its own overflow
-- so its scrollHeight equals its clientHeight and an assigned scrollTop CLAMPS TO 0: the precondition would not fire
and R7 would read green for the wrong reason (the builder's flag; a silently clamped red is the claim "the instrument
can fail" failing quietly). The CONSCIOUSNESS plate itself overflows (content 324 px in a 171 / 174 px box), is
`overflow:hidden` (scrollable by script) and is a member of the clip chain. So at this head R7 sets THE PLATE's
scrollTop to 150 and reads the refusal; the orchestrator first reads back the assigned scrollTop on BOTH nodes and
records which one took the offset (the clamp on `#widgets` is recorded as data, not assumed). After the layout change
the column is the scroller and R6a / R7 are re-run on `#widgets` as 3.2.2 wrote them. R7b and R7c are each run as ONE
synchronous expression (the 1 Hz repaint rewrites `#wC5` and would undo a constructed mutation between two calls).
R7c's "committed census" arm is the first reading's script (`59141bf`'s bytes), byte-preserved and wrapped as a
callable so both censuses read the SAME mutated page in the same expression. The mutants `trust_the_scroll` and
`details_reported_not_refused` are scratch copies of the committed script with ONE anchored replacement each (the
builder's `make_mutants.py`, each anchor matched exactly once); they are never committed. Nothing else in 3.2 moves.
