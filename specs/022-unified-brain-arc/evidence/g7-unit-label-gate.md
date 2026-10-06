# G7 / T115 -- the unit-label gate (P4d's provenance check, red G-R7) and the designs of T1 and T4 (pre-registration DRAFT)

Row: PLAN-FINAL G workstream, G7 / T115 (owner decision 2026-09-25: C, "grown columns on an anatomical map"); red G-R7
(Part F); POLICIES P4d's gate specification ("a provenance check (every grown-unit record carries its position, layer,
sign, label and the problem that licensed it, and a saved record the sleep that kept or pruned it; an unlabeled unit is
refused)"). Inherited and frozen: the decision record `evidence/g7-organizing-unit.md` (landed docs-only, `9b377ac9`),
its sections 3 (the label vocabulary), 6 (the four tests) and 7 (the red); Part F's preamble (S = 8, paired one-sided
t > 1.895, ties counted, exact sign-flip FPR printed, no majority-of-seeds rule). Every item below that is not a
restatement of those is marked **(G7 proposal)**. **Status: P0 FREEZE TEXT (wave 1, folded 2026-10-04 after the lead
read, the advisor read and the Codex critique; see the Review record at the end); the lead commits it. Items pending the
owner (O1-O4, O6, O8, O9) carry registered fallbacks. No code, no run beyond the one cost probe of section 2.12.** Every number not marked
*measured* is a **projection**.

## 1. Territory at the base `e1b3ae3c` (re-taken 2026-10-03 for this draft)

1. **No label gate exists.** A search of `src/` for a provenance or label gate (`provenance_gate|label_gate|unlabel|
   refuse.*label|region_label|problem_id`) fires on known positives (`gates/immortal_retention.py:91` KNOWN_UNLABELED,
   `growth/placement.py:891`) and on nothing that refuses a grown unit. `graphify query` on the same question returns no
   gate node.
2. **The incumbent label is `growth/placement.py` (G8 moved it from `cortex/`; the decision record's paths predate the
   move).** `place(n, seed, region)` returns `(depth, layer, sign)` per unit: depth is FRACTIONAL laminar depth in [0, 1],
   layer is drawn from H01's measured occupancy, sign from H01's measured E-fraction (`assign_sign`, Layer 1 refused
   excitatory by construction, `placement.py:147-162`). Non-cortex regions take the GRN arm `_place_grn`
   (`placement.py:366-379`): uniform depth, the region's settled label in the layer slot, **sign `None`** (no measured
   E-fraction outside H01 cortex). `place_grown` (`placement.py:407-439`) is the one admission rule today: a safety-seam
   name and a SCAFFOLD region are refused BY NAME. No (x, y), no region field, no problem id, no unit id exists in the
   record.
3. **The live grower writes the label and nothing reads it.** `growth/neurogenesis_engine.py::_refit` calls `place()`
   directly (`:484-495`, not `place_grown`), keeps placement aligned with the surviving rows after `prune_unearned`
   (`:503-505`, "follow the survivors": a pruned unit's placement is DROPPED), and persists it as OPTIONAL arrays
   `place_depth/place_layer/place_sign` beside the weights with no `GROWTH_FORMAT` bump (`:619-648`; the code cites
   73,490 `_T_GROWTH` records, 508 MB, at its last census). `restore_growth_state` DERIVES a fresh placement with
   `place(n_units, seed=self.seed)` whenever the blob lacks those arrays or they are malformed (`:767-781`): a derived
   label is byte-shaped exactly like a born one. The 2026-09-02 neuter-A/B (`evidence/s1-placement-null.md`, pinned by
   `tests/test_placement_charge_null.py`) measured placement WRITE-ONLY on the live path, and G5 (wave 1, its own
   measurement, relayed in D5) finds placement drawn independently of a unit's weights, so a layer/sign selection is equal
   in distribution to a size-matched random mask (P(L5 or L6 excitatory) per unit 0.1706).
4. **Label-creation call sites** (an AST scan of `src/` for calls to `place`, `place_grown`, `_place_grn`,
   `compose_growers`, 2026-10-03): outside `placement.py`'s own `_selfcheck`/`main`, exactly two live-tier sites, both in
   the one grower -- `neurogenesis_engine._refit:492` (birth) and `neurogenesis_engine.restore_growth_state:781`
   (derivation) -- plus `placement.compose_growers._resize:485` (no live caller). `faculties/theory_of_mind.py:130,148`
   call an unrelated `place` (a name collision any name-keyed scan must resolve by import, a known false-positive shape).
5. **The map at today's grain** (*measured*, the cost probe of 2.12): inverting `growth/arealization.py::REGION_COORDS`
   on the DECLARED (neuromere, zone) coordinate, 7 of the 16 labels are resolved uniquely (cerebellum, epithalamus,
   hypothalamus, midbrain, neuromodulation, pons, thalamus); the cell ('sp', 'alar') holds 7 labels (amygdala,
   basal_ganglia, claustrum, cortex, hippocampus -- growable -- with insula and olfactory -- SCAFFOLD); ('r5', 'basal')
   holds brainstem and medulla (both SCAFFOLD). `neuromodulation` is in `REGION_COORDS` but in neither `GROWABLE` nor
   `SCAFFOLD` (`growth/growth_program.py:120-134`).
6. **A discrepancy with the decision record, disclosed, not taken as a design win.** The record's section 3 says only
   the neuromere is load-bearing (`placement.region_prior` pins the zone inert) and that 'r1' (cerebellum with
   neuromodulation) cannot be told apart from refused territory. Reading the DECLARED zone, as item 5 does, separates
   them. Whether the gate's map may read the declared zone is open question O1 (section 3); this draft registers both
   readings and runs neither as a rescue of the other.
7. **What the reused statistic is.** `instruments/paired_contrast.py` (`paired_t`, `permutation_fpr`, `contrast`,
   `T_BAR = 1.895`) is Part F's statistic in one place; T1 and T4 import it (P2: no second statistic).
8. **Live state** (memory `project-snn-engine-inventory-1003.md`, verified 2026-10-03): the organizing unit is a
   decision record only; depth/layer/sign are written live with a null charge; position, region label and licensing
   problem are absent; T1-T4 are spec-only; no engine gate is built. The grower holds 2-4 units at a time in the
   record's own words, so a map "means little until hundreds exist".

## 2. Pre-registration

### 2.1 The questions, and what a result licenses

- **Q-gate (G-R7).** Can a grown unit enter the brain without its full label (position, layer, Dale sign, region label,
  the problem that grew it), or with a position that reads refused territory, and does a gate refuse it -- with per-field
  attribution and specificity? And does a SAVED unit record name the sleep that kept or pruned it? A green licenses only
  "the gate refuses the constructed defects and admits the constructed label, and no label-creation site escapes it";
  it never licenses "labels charge an outcome" (P3: the gate is not the labels' consumer; their charge is owed by G5's
  return-by-label and G4's T2).
- **Q-T1 (the growth-unit race, design only; runs in G3's campaign and in G9).** At the same neuron budget, on the same
  problems, do column-sized batches solve with no more than 20% more neurons than single-neuron growth and forget no
  more? A pass licenses keeping the column batch as the growth unit; it licenses nothing about growth versus no growth
  (that is G-R3's presized arm, which T1 reads and never re-runs).
- **Q-T4 (emergent vs imposed organization, design only; wiring half runs in G4, task half in G9).** Does rule-grown
  wiring (layer target x distance kernel x shared-partner bonus, with a wiring cost) produce higher modularity than
  random attachment at the same unit and edge count? A pass licenses "the rule produces modular structure"; never
  "the modules are regions" (the map is T2's) and never a task benefit (G9's half).

### 2.2 The label contract (the record the gate checks) -- (G7 proposal; the interface G3, G4 and G6 are asked to refute)

One record per grown unit, EXTENDING `placement.py`'s persisted `(depth, layer, sign)` (decision record section 5: no
second label store). New fields ride as OPTIONAL arrays in the growth blob under the existing additive contract
(`neurogenesis_engine.py:619-625`: no `GROWTH_FORMAT` bump; an old blob restores, an old reader ignores new arrays).

| field | value rule (each is a refusal reason with its own field name) | source |
|---|---|---|
| `unit_id` | stable, unique within the brain; survives prune re-indexing | G3 (the loop) |
| `batch_id` | the column batch the unit was grown in (a single unit is a batch of one); the population key G5's return path, G6's census and per-batch feed attribution read (decision record section 4) | G3 (the loop) |
| `position` | `(ap, dv, x_um, y_um, depth)`: `ap, dv` in [0, 1) are the MAP coordinate the region is read from, binned on the declared partitions `arealization.NEUROMERES` (14) and `ZONES` (4), carrying no metric; `x_um, y_um` finite, NATIVE micrometres in the unit's local tangential sheet (an unbounded plane per map cell, so a kernel is never truncated at an edge), the frame G4's TANGENTIAL kernel measures distance in (D6: G4 refits the rule's kernel in a tangential frame derived from H01's own layer circles; T3's verdict stays on 3-D soma distance); distances are taken only between units of the same cell; `depth` in [0, 1] (the existing field; it enters the rule only through the layer target, so no cortical thickness is needed) | `place_labeled` (G7) draws each batch's unit offsets at H01's tangential soma density around the batch centre; the centre, in um, from G3's site rule |
| `layer` | in `placement.LAYER_NAMES`, AND equal to the layer READ from `depth` through `placement.DEPTH_BANDS` (position-layer consistency) | `place()` |
| `sign` | +1 or -1 (never `None`, never 0); Layer 1 never +1 (`assign_sign`'s structural rule); immutable per `unit_id` (Dale) | `place()` |
| `region` | a `REGION_COORDS` name; in `GROWABLE` (which already excludes the five `SCAFFOLD` names and `neuromodulation`); CONSISTENT with `position`'s map cell (the cell's label set contains it). "The cell is not exclusively refused territory" is IMPLIED by these two (a cell holding a growable label is not exclusively refused), so it is not a separate check and R10 has nothing to neuter there | read off the map (below) |
| `region_read` | `"map"` when the cell resolves to that one label, `"name"` when the cell holds several labels and the grower's region chose among them (disclosed, never silent) | the gate computes it |
| `problem_id` | a non-empty id that RESOLVES through the injected problem registry: the key of G3's persisted problem ledger record, which EQUALS G2's `Problem.id` (one id, stable across restarts, P4b); lineage, not ownership (plan line 28: a batch may serve later tasks) | G2 constructs, G3's ledger persists |
| `origin` | one of `"born"`, `"derived"`, `"legacy"`, decided by journal evidence (2.6b(1)); only `"born"` is admissible; the persisted array is a recomputed cache, never evidence | `classify_restored` (restore) / the gate (admission) |
| `sleep_id`, `verdict` (SAVED records only; G6's names) | `verdict` in {`"kept"`, `"pruned"`}; `sleep_id` resolves through the injected sleep registry (G6's sleep record); both absent on an admission record | G6(d), appended to G3's unit journal |

**Who writes what (agreed with G3, debate D1).** G7 owns the extension of `growth/placement.py` and exposes
`place_labeled(n, *, site=(ap, dv), problem_id, batch_id, seed)` returning full admission records (the cortex arm's
depth/layer/sign drawn exactly as `place()` draws them, so its pinned golden does not move), plus the population reader
`label_of(records, unit_id)` (the record or `None`) and `labels_by(records, key)`, key in {region, batch_id,
problem_id, layer, origin} -> {value: unit ids} (G5 and G6 read
labels through it and define none of their own; `origin` added at G6's request, D4 -- it separates born from
legacy/derived units, it does not identify scaffold, which is never admitted). G3 never builds a label; its column batch holds `batch_id` and calls
`place_labeled`.

**Roles and the write-once sign (D8, agreed with G3; lead read 2026-10-03; Codex b3 #28 folded).** The signature
becomes `place_labeled(n, *, site, problem_id, batch_id, seed, roles=None)`. With `roles=None` the draw is exactly
`place()`'s (layer from measured occupancy, depth inside that layer's band, sign from `assign_sign`), byte-identical, so
the pinned cortex crc does not move. With `roles` given (a length-n tuple of `"E"`/`"I"`), the sign IS the role
(E = +1, I = -1) and the layer is drawn from H01's JOINT distribution conditioned on that sign, P(layer | sign)
proportional to occupancy(layer) x P(sign | layer), from the same measured E-fractions `assign_sign` reads; depth is drawn
inside that layer's `DEPTH_BANDS` interval. **What is and is not H01's (Codex b3 #28):** a role-conditioned draw keeps
H01's CONDITIONAL P(layer | sign) only; the E/I MIXTURE is imposed by the role tuple, so the population's unconditional
layer and sign marginals are NOT H01's (a SINGLE arm is 100 % E; a COL or BATCH9 batch is 8 E : 1 I, 1/9 inhibitory).
Unconditional H01 marginals are claimed for the `roles=None` draw only, and every instrument that reports a grown
population's layer or sign shares prints the imposed role composition beside them, with H01's measured cortex E-fraction
read from `placement`, never restated as a number here. An E unit is never L1 (P(E | L1) = 0 already). G3
passes an explicit `roles` tuple on EVERY grown batch (a SINGLE is `roles=("E",)`), and `column_batch.py` keeps no E/I
role of its own: it reads E/I from the label's sign through `label_of` (one store, P2; a role/sign disagreement cannot
exist by construction). **The sign is write-once per `unit_id`:** readers (G3's deposit-time sign clipping, its D9) may
cache it at admission; `check_saved`'s Dale check (first_sign from G3's journal birth entry) is what keeps that cache
sound. G3's D9 sign clipping on output synapses checks the WEIGHTS and names the synapse; this gate checks the LABEL and
names the unit -- two objects, not two checkers of one.

**Reading the region off the map.** The gate inverts `REGION_COORDS` on the cell `(NEUROMERES[floor(14 ap)],
ZONES[floor(4 dv)])`. Where the cell holds exactly one label, that label must equal `region` (`region_read = "map"`).
Where it holds several (only ('sp', 'alar') among growable cells at this grain), `region` must be one of them and the
gate records `region_read = "name"`: the label is NAME-assigned there, and the decision record's own fallback holds
("until the map resolves positions finer, place_grown's name-based SCAFFOLD refusal stays the working guard").
Under the record's strict sentence ("any position the map cannot tell apart from their territory [is] refused") every
('sp', 'alar') unit -- every cortex unit -- would be refused today; this draft registers the fallback, not the strict
sentence, and puts the choice to the owner (O2).

**The frozen position-to-region contract (one rule; batch 1 Codex G7 #1, read although that review was void, see the
Review record).** In order, each a refusal with field `position` or `region`: (1) `ap` or `dv` non-finite or outside
`[0, 1)` -> refused, `position` (no clamping; `floor` is taken only on in-range values, so a boundary value 1.0 is out of
range, never cell 13 or zone 3); (2) the cell `(NEUROMERES[floor(14 ap)], ZONES[floor(4 dv)])` holds no `REGION_COORDS`
label -> refused, `region`, reason `unknown territory` (UNKNOWN is never admitted); (3) the cell holds exactly one label
-> `region` must equal it (`region_read = "map"`); (4) the cell holds several labels -> admitted ONLY under the O2
fallback, `region` must be one of them AND in `GROWABLE` (`region_read = "name"`, disclosed); if the owner rules the
strict sentence, rule (4) becomes a refusal with reason `ambiguous territory` and nothing else changes. **Consequence for
the positive control:** R6 needs demonstrably admissible territory; under the fallback that is ('sp', 'alar') with
`region = "cortex"` (the only region with a Dale sign, O3); under the strict sentence NO admissible cortex position exists
at today's grain, so R6 has no positive, the gate cannot show specificity, and its LANDING is HELD (not landed as a
refuse-all gate) until a finer pallial map exists -- disclosed here so a strict ruling's cost is known before the owner
rules.

**Why only cortex units are admissible today (a consequence, surfaced, not hidden).** A Dale sign needs a measured
E-fraction; only H01 cortex has one, so `_place_grn` returns `sign = None` for every other region and the gate refuses
it. `GROWER_REGIONS` routes `ng_snn` and `function_preserving_growth` to hippocampus (`placement.py:393-397`). Owner
question O3.

### 2.3 The gate (`src/snn_data_lake/gates/unit_label_gate.py`, G7 proposal)

Pure, stdlib plus the incumbents' constants (no second vocabulary: `REGION_COORDS`, `NEUROMERES`, `ZONES`, `GROWABLE`,
`SCAFFOLD`, `LAYER_NAMES`, `DEPTH_BANDS` are imported, never copied).

- `check_admission(unit, *, resolve_problem) -> Verdict(admitted, field, reason)` -- the FIRST failing field in the
  fixed order `unit_id, batch_id, position, layer, sign, region, problem_id, origin`;
  `require_admissible(units: Sequence[Mapping], *, resolve_problem: Callable[[str], bool], known_unit:
  Callable[[str], bool]) -> AdmissionTicket` raises `UnitLabelRefused(unit_id, field, reason)`. G3's loop calls it for
  EVERY grown unit (SHAM units included) before the row joins the learner. A presized arm's units are a constructed
  network, not an admission, and are not gated.
- **`unit_id` uniqueness (Codex b3 #9 BLOCKER, folded).** `known_unit(unit_id)` is True iff G3's authoritative unit
  journal (its IMMORTAL `'unit'` birth rows, P4b) already holds a birth entry with that id; `require_admissible` refuses,
  field `unit_id`, (a) any id that `known_unit` already knows (reason `duplicate across batches`) and (b) any id appearing
  twice in the incoming batch (reason `duplicate within batch`), before any other field is read. **Restoration is not
  admission:** restore never calls `require_admissible`; it calls `classify_restored(rows, journal) -> {unit_id: origin}`,
  which gives `origin = "born"` ONLY to a restored row whose `unit_id` has a journal birth entry AND whose restored label
  equals that entry field for field; a row with no journal entry is `derived` (re-derived labels) or `legacy` (no label
  arrays), and a row whose label differs from its journal entry is `derived` and printed as a mismatch. So origin is
  read from journal EVIDENCE, never from the blob's own `origin` array (batch 1 Codex G7 #3), and `derived`/`legacy`
  cannot become an admission escape: neither is admissible and neither can be re-admitted under a known id.
  AMENDED 2026-10-04 (D21): a journaled id whose restored label differs is REFUSED, not `derived` -- see the dated pre-data amendment at the end.
- **The gate precedes admission structurally (Codex b3 #8, folded).** `require_admissible` returns an `AdmissionTicket`
  naming exactly the admitted `unit_id`s; G3's single row-append point (its `admit_rows(rows, ticket)`, an interface ask
  to G3 registered in 2.9) refuses rows whose ids the ticket does not cover, and a ticket is consumed once. "A grower
  calls the gate somewhere" is therefore never the evidence; the append itself requires the gate's output.
- `check_saved(record, *, resolve_problem, resolve_committed, first_sign) -> SavedVerdict` with THREE outcomes (D20,
  agreed with G6): `KEPT`/`PRUNED`, `PENDING`, `REFUSED`. Order: every admission rule first (with `origin` allowed
  `"born"` only) and Dale immutability against `first_sign[unit_id]` (read from G3's journal birth entry, never from the
  record under check) -- any failure is `REFUSED`, field named; then the sleep clause: the record carries `sleep_id`,
  `mask_digest` and `verdict` in {`"kept"`, `"pruned"`}, and `resolve_committed(sleep_id, mask_digest)` is True iff G6's
  `sleep_mask_commit` row with that EXACT pair exists (G6's A12: prepare -> verdict rows carrying `mask_digest` -> apply
  -> checkpoint `applied_mask_digest` -> commit; a restore that finds the mask applied WRITES the missing commit row,
  idempotent). A committed verdict gives `KEPT`/`PRUNED`; a unit whose only verdicts are uncommitted (a rolled-back
  mask's, which stay uncommitted permanently, P4b) is `PENDING` -- "not yet kept or pruned by any sleep", never refused
  as unlabeled and never satisfying the saved-record clause; a missing `sleep_id`, `mask_digest` or `verdict`, or a
  verdict outside the two values, is `REFUSED`. G6(d)'s save calls it on the joined view (G3's journal birth entry +
  G6's verdict record); G6's planted record without `sleep_id` (its C15) is refused HERE, and G-R6's clause "a kept or
  pruned unit's saved record does not name the sleep" reads THIS check (P2: one checker; G6 confirmed, D3 and D20).
  `resolve_committed` reads only `(sleep_id, mask_digest)`; no other commit-row field enters it.
- `require_labeled_targets(target_ids, records) -> None` (D5, at G5's request) -- the return path's port calls it when it
  builds or rebuilds its projection (never per tick): an unknown id, a legacy or derived unit, or a record failing the
  admission check is refused. One checker for "is this an admitted labeled unit" (P2). Unit records are resolved through
  `placement.label_of(records, unit_id)`.
- `census(records) -> {born, derived, legacy, refused_by_field}` over a set of unit records (a throwaway home in the
  self-check; the live lake only by an explicit command, never in `run_selfchecks`).
- **The creation-site scan (the asserting arm, the `immortal_retention` pattern).** An AST pass over the governed tier:
  every function that calls a label-creation function of `growth.placement` (resolved by import, so `theory_of_mind`'s
  `place` is not a hit) must also call `require_admissible`/`check_admission`, or be a row of
  `KNOWN_UNGATED_CREATION_SITES` -- a shrink-only debt table printed on every run with its owning row and what retires
  it, and a `stale_known()` red when a listed site no longer creates labels. Day-one rows (projection, from 1.4):
  `growth/neurogenesis_engine.py::_refit` (owner G3/T111: the clock-driven refit admits units with no problem, no
  region, no map position; retired when the one grower's admission goes through G3's problem-keyed loop) and
  `growth/neurogenesis_engine.py::restore_growth_state` (owner G3/T111: must classify its rows by `classify_restored`, never "born" without a journal entry;
  retired when its rows are classified by `classify_restored` from journal evidence). While either debt table is
  non-empty the report prints, in those words, that the sentence "no unlabeled grown unit is admitted" is FALSE, and
  **T115 stays OPEN** (batch 1 Codex G7 #3): the gate's landing stacked in G3's arc lands the checker and its reds, but
  T115 closes only when every live birth path in `GROWN_SEGMENTS` admits through a ticket, or the owner records a scoped
  exception (O4).
- **Coverage is anchored on row ADMISSION, not label creation (Codex b3 #8 BLOCKER, folded; replaces the earlier
  "the census's grower list is the scan's list").** Three instruments, each independent of label creation:
  (a) **`GROWN_SEGMENTS`**, a declared inventory in the gate of every live module and attribute that owns grown weight
  rows (day one, projection from 1.3-1.4: `growth/neurogenesis_engine.py` `W`; G3's grown segment when it lands), each
  row naming its owning plan row; (b) **the row-growth scan**, an AST pass over the governed tier for functions that
  GROW an array's leading dimension on an attribute (`np.vstack`, `np.concatenate`/`np.append` on axis 0, `np.resize`,
  `np.pad` of axis 0, slice-assignment past the end, `torch.cat` on dim 0), with NO reference to placement: every hit must
  sit in a `GROWN_SEGMENTS` module and pass through that segment's ticket-checked append point, or be a row of
  `KNOWN_UNGATED_ROW_GROWTH` (the shrink-only debt table, printed every run with owner and retirement; `stale_known()`
  reds a row that no longer grows rows); a hit outside both reds; (c) **the runtime census**, per `GROWN_SEGMENTS` row,
  classifies every row by `classify_restored`'s journal rule (2.6b(1)) and prints `born / derived / legacy`. Because the
  three origins partition the rows, their sum always equals `rows` and is NOT the red; **the red is: in a segment with no
  open debt row, ANY row classified `derived` or `legacy`** (a row present without a ticketed, journaled birth = an
  unticketed admission). A segment WITH an open debt row prints its derived/legacy count and is not red (the debt
  table's FALSE sentence and T115-OPEN carry it). The label-creation scan above stays as a SECOND, independent arm (a site that creates labels
  without the gate). Neither scan nor census alone is reported as "every grown unit is labeled"; the claim needs all
  three green and both debt tables empty.

### 2.4 Checks, each with a constructed, reachable red (the build arcs owe every one; a red that does not fire halts its arc)

G-R7's controls, verbatim from the record section 7, then this draft's additions:

| # | construction | must | clause served |
|---|---|---|---|
| R1 | a grown unit record with no label fields | refused | G-R7 "admitted without its unit label" |
| R2 | five fully labeled units, each with ONE of position, layer, Dale sign, region label, problem removed | each refused, and the verdict names exactly the removed field (a wrong field name reds: per-field attribution) | G-R7 per-field |
| R2b | a fully labeled unit with `unit_id` removed; one with `batch_id` removed | each refused, that field named (R2 keeps G-R7's five fields verbatim; R2b covers this draft's two additions, so R10 has a red to stop for each) | (G7 proposal) the added identity fields |
| R3 | a fully labeled cortex unit, `region = "insula"` | refused by NAME (SCAFFOLD) | G-R7 "a neuron placed in refused territory (insula)" -- fires by name at today's grain |
| R4 | `region = "cortex"`, position in ('r3', 'basal') (pons) and in ('r5', 'basal') (medulla/brainstem) | refused by POSITION through the consistency check (the cell's labels, all refused, do not include cortex) | G-R7 "a neuron whose position reads a refused label" -- fires by position |
| R5 | `region = "neuromodulation"` (the 16th name, in neither partition) | refused through the `GROWABLE` check (a named case, not a separate check) | record section 3 names it |
| R6 | one fully labeled cortex unit with a resolving problem | ADMITTED (a refuse-all gate reds) | G-R7 specificity |
| R7 | invalid values, one each: layer not a cortical layer; depth outside [0, 1]; depth in the L5 band with layer L2; sign 0, `None`, Layer-1 +1; `x_um` NaN; unknown region name; a problem id that does not resolve; `origin = "derived"` | each refused, field named | (G7 proposal) value rules, not presence only |
| R8 | a restored pre-label blob (no `place_*` arrays) on a throwaway engine | its units counted LEGACY, never BORN; re-admitting them refused (`origin`) | (G7 proposal) the restore-derivation hole of 1.3 |
| R9 | a saved record without `sleep_id` (G6 supplies the plant, its C15); without `mask_digest`; with `verdict = "evicted"`; a re-save with the opposite sign | each `REFUSED`, field named | P4d's saved-record clause; Dale |
| R9b | (D20, with G6) (a) a record whose only verdict is uncommitted; (b) a verdict whose `sleep_id` has a commit row with a DIFFERENT `mask_digest`; (c) a rolled-back prepared mask's verdict, read again after a later restore; (d) a restore-completed mask's verdict once restore has written its commit row | (a)-(c) `PENDING`, never `KEPT`/`PRUNED`, never `REFUSED` as unlabeled; (d) `KEPT`/`PRUNED` | P4d's saved-record clause, crash-consistent (G6's A12) |
| R10 | the gate with ONE check neutered at a time (monkeypatched to pass) | the matching constructed red above STOPS firing, every other still fires | (G7 proposal) the control that is not the method: each check is load-bearing, none is decorative |
| R11 | a constructed module whose function calls `placement.place` with no gate call and no debt row; and a debt row naming a function that no longer creates labels | scan reds (`findings`, `stale_known`) | (G7 proposal) the asserting arm can fire |
| R12 | a throwaway incumbent engine refit that grows N > 0 units with today's code | census counts exactly N legacy units, 0 born; with the segment's debt row present it prints (not red), with the debt row removed it REDS | (G7 proposal) a zero from the census is a claim about the census: the counter is proven on a known positive |
| R13 | a return projection (G5's connection 6) naming a legacy unit, an unknown unit id, and a record with no region; and one naming an admitted born unit | the first three refused by `require_labeled_targets`, the last accepted | (G7 proposal, D5 with G5) the return path projects only into admitted labeled units; a scaffold unit cannot be admitted, so a scaffold target is "not admitted" |
| R14 | (Codex b3 #8) a constructed module that grows a weight attribute's rows with `np.vstack` and calls NO placement function and NO gate | the row-growth scan reds (a hit outside `GROWN_SEGMENTS` and outside the debt table) while the label-creation scan stays silent -- proving the coverage arm sees what the creation arm cannot | coverage independent of label creation |
| R15 | (Codex b3 #8) a fixture segment that appends rows BEFORE calling the gate, or with a ticket that does not cover them, or with a consumed ticket | the append is refused; and on a fixture segment (no debt row) whose append bypasses the ticket check, the runtime census reds (the bypassed rows classify `legacy`/`derived`, never `born`) | the gate precedes admission, structurally |
| R16 | (Codex b3 #9) one batch with two fully valid units sharing a `unit_id` | refused, field `unit_id`, reason `duplicate within batch` | uniqueness within a batch |
| R17 | (Codex b3 #9) a fully valid unit whose `unit_id` already has a journal birth entry, in a later batch | refused, field `unit_id`, reason `duplicate across batches` | uniqueness across batches |
| R18 | (Codex b3 #9) restore of (a) a journaled unit whose label equals its entry; (b) a row with no journal entry; (c) a journaled unit whose restored label differs | (a) `born` and NOT refused as a duplicate (restoration is not admission); (b) `derived`/`legacy`; (c) `derived` with the mismatch printed | restoration distinguished from admission; origin from journal evidence |
| R19 | (batch 1 #1) `ap = 1.0`; `dv` NaN; a position whose cell holds no `REGION_COORDS` label | each refused (`position`, `position`, `region` "unknown territory") | the frozen position-to-region contract |

AMENDED 2026-10-04 (D21): R18(c) now reads REFUSED (tampered born unit), not `derived` -- see the dated pre-data amendment at the end.

The gate's own determinism: verdicts are pure functions of the record and the registries (no RNG); the fixture RNG uses
seeds 13010-13019. **Home safety (board PROTOCOL 4b):** R8 and R12 build an engine and therefore run under
`substrate.home.throwaway_home()` (or a fresh scratch `NORD_HOME`), never the live `C:\Users\vinhl\.snn_data_lake`, whose
mtime each arc checks after; nothing here starts the daemon.

### 2.5 What a red deletes, and what it does not

- A G-R7 red is a defect of the gate or of a creation site, not a finding about the brain: the landing halts and the
  defect is fixed (no disposition deletes anything). A red that cannot be made to fire deletes THAT check (P8 F11), and
  the record says which.
- The **null disposition is a causal claim** (`feedback-a-pre-registered-disposition-is-a-causal-claim`): "the census
  found 0 unlabeled admissions" claims the census sees unlabeled units; R12's positive control runs FIRST, in the same
  command, and a census that misses R12's N is REFUSED, never read as a clean lake.
- T1 / T4 reds (2.6, 2.7) delete the part of C each tests and the owner is told (record section 6's disposition, from
  the brief the owner chose C on). Nothing here deletes an owner-drawn part of the engine (P4d: the two networks, the
  reservoir, the return path); the growth loop itself is owner-drawn, so a T1 reading that "growth earns nothing" (G-R3's
  presized arm) is reported to the owner, never a deletion.

### 2.6 T1 -- the growth-unit race (design; the run waits, 2.9)

- **Where it runs (debate D2, agreed with G3, ratified by main 2026-10-03, O5):** inside G3's campaign, as G3's "T1 + Astra" family on
  G3's seeds 9300-9307 (reserve 9308-9315), so every arm pairs on one seed set; this row's T1 confirmatory seeds are
  released (2.10). G3's runner executes; this row's instrument (P4) computes the decision.
- **Arms, same problems, same paired seeds, same total neuron budget (E and I units both count -- a column's shared
  inhibitory unit is not free):** COL(k), column-sized batches of k excitatory LIF units with one shared inhibitory unit
  (G3's `growth/column_batch.py`); SINGLE, single units; both use ONE site rule (G3's: the map site where error builds
  up) and ONE targeting rule (attach to the worst-residual label neurons -- "next to the worst readouts"), so the arms
  differ ONLY in unit size and T1 isolates the unit; RAND-SITE, the same column batches at uniformly random growable
  map sites (a control that is not the method: location carries nothing); PRESIZED, a fresh network at the COL arm's
  final size, run by G3 ON THESE SEEDS (G-R3's presized arm runs on 9100-9107 and is not paired with T1, so T1 cannot
  read it there). Within-site `x_um, y_um` offsets come from `place_labeled`. **The whole T1 family (2.6, T1-size and
  T1-struct of 2.6a) waits on G4's T3 at FAMILY level** (G3's Codex #3, agreed; DEPENDENCIES.md's G7 row): no T1 arm runs
  kernel-free.
- **Primary k = 8** (8 E + 1 I = 9 units per batch; the owner-saved Astra contract: "roughly 8-10 excitatory neurons
  with shared inhibition"); k in {4, 12} (G3's grid) are secondary, reported, never rescuing a primary red.
- **Endpoint per seed:** G3's U (its 2.8): the number of admitted units -- E and I alike -- at which the solved predicate
  (G2's, fixed at posing) first holds on validation AND then on the future presentations with learning OFF. `F` = the
  number of task A's distinct FUTURE ITEMS read correctly right before B minus right after B, learning OFF (G3's C11; in
  ITEMS -- stateless reads make repeated presentations of one item identical, so the resolution is distinct items). A
  problem paused UNSOLVED at its budget is censored.
- **The retention margin (G3's Codex #5, agreed; ruling 15).** `delta_NI = max(F_bound, 1 item)`, strictly positive, in
  ITEMS of task A's future set, fixed before any seed; `F_bound` is G3's factual C11 bound (it may be 0). The item count
  is filled by G2's C1-d-shaped separable task A (ruling 15: about 12 or more future items, one item about 8 %), landing as
  a dated pre-data amendment after the G1(c) campaign; **until it lands, the only registered A is A_ii with 4 future
  items, and with a 1-item margin on a 4-item A the bar resolves retention only in 25 % steps** (G3's wording) --
  disclosed as coarse. A retention red is worded "retention not shown non-inferior within delta_NI", never "demonstrated
  interference".
- **Decision (burden on the column unit, the G-R3 house style: a part must show it pays):** d_s = log 1.2 -
  log(U(COL)/U(SINGLE)); e_s = delta_NI - (F(COL) - F(SINGLE)).
  PASS iff MET on both (paired one-sided t over S = 8 > 1.895 through `paired_contrast`, the degenerate rule of 2.6b(6),
  the exact sign-flip rate of 2.6b(7) printed), and only where 2.6a's guard admits the U contrast. Otherwise
  T1 is RED: the column batch is reverted to single-unit growth and that part of C deleted, the owner told. The
  presized clause ("a presized network matching both means growth itself earns nothing") is read from PRESIZED on the
  same seeds and reported to the owner, never a deletion (2.5). **Censoring:** G3 scores an unsolved arm `budget +
  batch` (budget + 9 for COL, budget + 1 for SINGLE), which compresses U(COL)/U(SINGLE) toward 1 and favours columns at
  the margin; so if more than half the seeds are censored in either arm the contrast is REFUSED (not a pass, not a red)
  and reported. **The reading the draft did not take:** "red only when columns are SHOWN to cost 20% more" would keep
  the column unit on an underpowered null; put to main as O6.
- **Power before any confirmatory seed.** A dev pilot (G3's exposed 9040-9047, D2) estimates the sd of d_s and e_s; if
  the projected power at S = 8 is below 0.8 (CHOSEN, disclosed), the row requests a Part F AMENDMENT from main/the owner (Part F reserves
  S > 8 to G1: "every other seed-paired red stays at S = 8") BEFORE any confirmatory seed runs; it never pre-registers
  S > 8 itself.

### 2.6a T1 split: T1-size guarded, T1-struct added (joint G3+G7; lead read 2026-10-03; Codex b3 #22, #23 folded; PENDING OWNER O9)

The granularity guard binds 2.6 now (it only ever refuses, never deletes). T1-struct's DELETION authority is pending the
owner's O9: until O9 is answered a T1-struct red is reported to the owner and nothing is deleted; T1-size, behind the
guard, keeps the record's disposition (2.6).

- **The defect (G3's D2, checked with G2).** G3's capacity unit is a threshold/LIF unit over the frozen 52-rank code,
  read linearly by the label neurons; G2's problems need U_SINGLE of about 2-4 (XOR = OR - AND, a linear read of two
  units; disjoint XOR blocks share units; parity is capped at n <= 6 by the 6-rank horizon). Then U(COL) >= 9 >=
  2 x U(SINGLE) and d_s = log 1.2 - log(U(COL)/U(SINGLE)) < 0 on every seed: T1 reds BY CONSTRUCTION (granularity),
  an entailed red, not a reachable one (P8), which would delete the column part of C on an instrument artifact.
  Growing SINGLE in batches of 9 instead (G3's first option) entails the mirror PASS (both arms quantize to 9,
  d_s = log 1.2 on every seed) and swaps the record's question (unit size) for another (unit structure).
- **T1-size (the record's question, kept).** COL(8 E + 1 I) vs SINGLE (batch of one), as 2.6. **Granularity guard
  (Codex b3 #22, folded):** the threshold is U* = 80 (k = 8) or 40 (k = 4). Derivation: the worst quantization overshoot
  is k units (one batch less one, plus the I unit), and k / U <= 0.10 keeps it at half the 20 % bound; CHOSEN, disclosed.
  The guard is established **for EVERY judging problem**, not on a pooled median: the judging problem set is fixed by G2
  before the pilot (ruling 15's task A and the capacity-cell problems T1 names), and for each problem the pilot's (G3's
  9040-9047) median U(SINGLE) is taken over the pilot seeds where SINGLE SOLVED; censoring sentinels (`budget + batch`)
  are EXCLUDED from that evidence, and a problem where more than half the pilot seeds are censored FAILS the guard. If
  ANY judging problem fails, the WHOLE U contrast is REFUSED (reported untested, neither pass nor red, nothing deleted);
  no problem and no seed is dropped to make the guard pass, and the confirmatory set is never re-chosen after the
  pilot. On G2's present problems T1-size therefore has no reading in this wave; reported to the owner as untested.
- **T1-struct (new registered contrast).** COL(8 E + 1 I, shared I with E->I and I->E wiring) vs BATCH9: nine independent
  units placed by the SAME call `place_labeled(9, roles=("E",)*8 + ("I",))` at the same site, with the same site and
  targeting rules; BATCH9's I unit takes frozen-code input like any single and projects only onto label neurons (no
  E->I or I->E wiring). The arms match total count AND E/I composition, so they differ only in the column's internal
  structure. Its retention half e'_s = delta_NI - (F(COL) - F(BATCH9)) (2.6's margin, in items) reads at ANY size and is
  the cleaner retention contrast (2.6's e_s is count-confounded in the column's favour: more neurons, more capacity,
  less interference); its U half u'_s = log 1.2 - log(U(COL)/U(BATCH9)) carries the same per-problem guard (both arms
  quantize to 9, entailed below it). Seeds: G3's 9300-9307 (reserve 9308-9315), pilot 9040-9047, as 2.6.
- **T1-struct decision, one exact table (Codex b3 #23, folded).** Both halves are BURDEN-OF-PROOF non-inferiority bars:
  the column structure must SHOW it pays, so failure to establish NI is the red (the G-R3 house style), and the red is
  worded "column structure not shown non-inferior within delta_NI (retention) / 20 % (units)", never "the column forgets
  more" -- no demonstrated-harm test is registered, so no harm claim is ever made.

  | U half (guard) | retention half e'_s | T1-struct verdict | disposition (after O9 = yes; before O9: reported, nothing deleted) |
  |---|---|---|---|
  | REFUSED | MET | PASS on retention only; U reported "untested" | structure kept |
  | REFUSED | NOT MET | RED | structure deleted, owner told |
  | MET | MET | PASS | structure kept |
  | MET | NOT MET | RED | structure deleted, owner told |
  | NOT MET | any | RED | structure deleted, owner told |

  T1-size's table is the same with SINGLE for BATCH9 and its disposition the record's (2.6: the column batch reverts to
  single-unit growth); its U half REFUSED leaves T1-size UNTESTED as a whole (its retention half alone is count-confounded
  and never decides T1-size).
- **Dale on the weights is G3's (its D9):** sign clipping on grown and reservoir output synapses at every deposit (E >= 0,
  I <= 0) with a planted-flip red; without it E/I would be a cosmetic label and e'_s void. T1-struct waits on it.
- **Owner question O9** (section 3): may T1-struct's retention contrast stand in for the record's T1 while no judging
  problem passes the guard (median solved U(SINGLE) >= 80 on every problem)?

### 2.6b Self-defects 1-5 folded, rulings 9/9a, exact sign-flip reporting (lead read 2026-10-03; Codex b3 #21 folded)

1. **One origin partition, decided by JOURNAL EVIDENCE (the rule `classify_restored` applies; Codex b3 #9).** Every
   unit row is exactly one of: `born` -- its `unit_id` has a journal birth entry (written through a ticket) and its
   label equals that entry field for field; `derived` -- it carries the FULL label field set (2.2's table) but has no
   journal birth entry or differs from it (re-derived or forged labels); `legacy` -- it LACKS the full label field set
   (e.g. today's rows with only `place_depth/place_layer/place_sign` and no `unit_id`, or no label arrays at all). Only
   `born` is admissible. The persisted `origin` array is a recomputed CACHE: restore overwrites it with
   `classify_restored`'s result and no checker ever reads it as evidence. The census prints the partition per segment
   (2.3(c); its red is any derived/legacy row in a segment with no debt row, since the partition always sums to `rows`).
   R8 reads the same way: the restored pre-label blob's units are `legacy` (no full label set), never `born`, before
   and after G7's stacked `neurogenesis_engine.py` commit; re-admission is refused either way. R12's today-refit rows are
   `legacy` unambiguously.
   AMENDED 2026-10-04 (D21): `derived` = full label set with NO journal entry; a differing journaled label is a refusal -- see the dated pre-data amendment at the end.
2. **Stated couplings.** (a) R6 and every cortex admission depend on O2: under the fallback a cortex unit is admitted
   with `region_read = "name"`; under the strict sentence the gate refuses every ('sp', 'alar') unit and G3 can admit
   nothing -- a ruling for the strict sentence blocks G3's loop until a finer pallial map exists. (b) R8 cannot be green
   until G7's stacked `neurogenesis_engine.py` commit lands (after G3's lifecycle refactor, R2 and R3; D7).
3. **The census's grower list** -- SUPERSEDED by Codex b3 #8: the self-fold here ("the scan's own list") made coverage
   depend on label creation, so a row-growing path with no placement call was invisible. Coverage is now anchored on
   `GROWN_SEGMENTS` + the row-growth scan + the runtime census (2.3), with reds R14 and R15.
4. **Dale's `first_sign`** is read from G3's journal birth entry for that `unit_id`, never from the record under check.
5. **Position-layer consistency (R7)** reads layer from depth on `DEPTH_BANDS`' half-open `[lo, hi)` intervals, the same
   ones `place()` draws from (`placement.py:241-242`: `lo + rng.random() * (hi - lo)`), so every born label passes and a
   boundary depth is never ambiguous.
6. **Rulings 9 and 9a (degenerate contrasts).** A zero-variance paired sample (every d_s equal, sd = 0):
   - **Superiority contrasts** (T4's main RULE > RANDOM, its three attribution contrasts, and IR1): reads **NOT MET**,
     disclosed as stricter than `paired_contrast.contrast`'s zero-variance rule (ruling 9; G1(c) 2.3 precedent). IR2 (the
     false-positive control, which must NOT pass) is read by the same rule, so a degenerate IR2 sample counts as "did not
     pass".
   - **Non-inferiority bars** (T1-size's d_s and e_s; T1-struct's U half and e'_s): read by the common value, > 0 MET,
     <= 0 NOT MET, with "degenerate" printed beside the verdict (ruling 9a). The granularity guard of 2.6a still comes
     first: a T1-size or T1-struct U contrast that fails the per-problem guard is REFUSED before any degenerate reading,
     because an entailed common value (e.g. log 1.2 on every seed) is exactly what the guard exists to refuse.
7. **Exact sign-flip rate under the ACTUAL verdict rule (Codex b3 #21, folded).** `paired_contrast.contrast` reports
   FPR = 0 for a degenerate sample without enumerating flips, so changing its MET flag does not repair the reporting.
   This row's instrument (`unit_organization.py`, P4; `paired_contrast.py` is not edited) computes, for every contrast,
   the exact rate over all 2^S = 256 sign flips of the observed d_s, each flipped sample judged by THIS draft's verdict
   rule (t > 1.895 for a non-degenerate flip; ruling 9 for superiority / 9a for non-inferiority on a degenerate one; the
   T1 guard is a property of the data, not of the signs, and is not re-applied per flip). That is the "exact sign-flip
   FPR printed" of Part F. `paired_contrast`'s own reference rate is printed beside it, LABELLED as the reference t-test
   rate. Example (Codex): eight identical positive d_s -- a 7+/1- flip has t = 3 and is MET, so the operative rate is not
   0. Registered reporting fixtures (seeds none; constructed vectors): constant-positive (eight equal d_s > 0) and
   constant-zero (eight d_s = 0), each with its hand-computed exact rate asserted, plus one non-degenerate vector checked
   against a direct enumeration.

### 2.7 T4 -- emergent vs imposed organization (design; the wiring run waits on G4's T3)

- **Arms at the same unit set N and EXACTLY the same undirected edge count M:** RULE, units added one at a time with
  edge probability proportional to layer target (`CONN_PROB`, read through placement -- G4) x distance kernel (the
  TANGENTIAL kernel G4's Part B refits after T3 decides the form on 3-D soma distance, D6; record section 3: a
  hand-picked width is refused) x (1 + beta x shared partners), the wiring cost
  entering through the distance term; RANDOM, random attachment at exactly M edges (`gnm`, the record's red);
  IMPOSED, units assigned to fixed spatial columns of k = 8 with the rule applied inside a column only (reported beside
  the wiring half; it is the comparator of the task half, G9).
- **Who does what (D6, agreed with G4):** this row owns the scorer (`instruments/unit_organization.py`) and this
  decision's pre-registration; G4's `growth/wiring_prior.py` supplies the RULE generator (it holds no per-unit field: it
  reads labeled records through `place_labeled`/`labels_by` and stores only its kernel record in
  `tools/g4_t3_campaign.json`) and G4 runs the wiring half on this row's T4 seeds (2.10). Every kernel arm waits on T3
  and G4's Part B; the beta arm waits on G4's Part B2.
- **beta is never tuned on modularity** (that would make the red circular): G4 fits it on H01 in a data-only Part B2 --
  P(connect) against shared-partner count at matched distance, DEGREE-CORRECTED (p ~ out_i x in_j x g(d) x
  (1 + beta m_ij)), scored by edge-level hold-out (partners counted from training edges only), after two planted
  controls (a kernel-only world with H01's degree sequences must read beta not above its null; a planted beta > 0 world
  must recover beta). Either control failing REFUSES the fit and T4 runs at beta = 0 with the shared-partner clause
  reported untested. Power is G4's projection (mean in+out degree about 9.8; most pairs share no partner).
- **Substrate (D6, G4's challenge 2):** `place_labeled` units in ONE cortical map cell, tangential positions at H01's
  TANGENTIAL (surface) soma density, READ from `tools/g4_t3_campaign.json` where G4's Part B records it per layer and
  pooled in the layer-circle frame (this draft's first figure, about 2,300 per mm^2, divided by x * y, a plane that
  contains H01's radial axis -- a cross-section density, WITHDRAWN on G4's refutation; G4's projection of the surface
  density is about 2.7e4 per mm^2, which stacks every layer at one sheet point, as the contract's (x_um, y_um) + depth
  implies). N is chosen MECHANICALLY after G4's Part B and before any T4 seed as the smallest of {1000, 2000, 4000}
  whose sheet side is at least 6 x the TANGENTIAL sigma from Part B (not T3's 3-D sigma); **if none of the three
  satisfies it, T4's wiring half is REFUSED** (reported, nothing deleted; no larger N is improvised). G3's real grown
  population is a secondary substrate, reported when it holds enough units, never the decision.
- **Graph construction, frozen (Codex b3 #24, folded).** Every arm yields a SIMPLE UNDIRECTED graph (no self-loops, no
  multi-edges) on the same N units with EXACTLY M edges.
  - **M:** `M = round(N x dbar / 2)`, `dbar` = the mean degree of H01's SYMMETRIZED simple graph over the cortical layers
    the substrate uses, READ from `tools/g4_t3_campaign.json` (an ask to G4's Part B, registered in 2.9). If that number is
    absent, T4 waits; no chosen density stands in for it. M is fixed per N before any T4 seed and is the same in every arm.
    AMENDED 2026-10-04 (D22): `dbar`'s population and the T4 layers are pinned -- see the dated pre-data amendment at the end.
  - **Directed to undirected:** `CONN_PROB` is directed layer-to-layer; the pair's layer term is `c(a, b) = 1 - (1 -
    p_ab)(1 - p_ba)` (the probability of at least one directed contact), so the undirected weight is symmetric by
    construction. Pair weight `w_ij = c(L_i, L_j) x g(d_ij) x (1 + beta x m_ij)`, `g` = G4's tangential kernel, `d_ij` the
    tangential distance, `m_ij` = shared neighbours of i and j in the graph built SO FAR.
  - **Exact-M sequential sampler (RULE and its attribution variants DIST-ONLY, flat-kernel, no-layer, and IMPOSED):** units
    arrive in a seeded random order (the arm's seed). Arriving unit t (t = 1..N) draws exactly `m_t` partners among the
    t - 1 earlier units, WITHOUT replacement, with probability proportional to `w`, by Efraimidis-Spirakis keys
    `u^(1/w)` from the seeded RNG (zero-weight candidates are never drawn). Allotment: `q = M // (N - 1)`, `m_t =
    min(t - 1, q)`, and the remainder `M - sum m_t` adds one edge each to the LAST arriving units that still have a
    spare earlier candidate, so `sum m_t = M` exactly. A variant sets its removed term to 1 (flat kernel: `g = 1`; no
    layer: `c = 1`; DIST-ONLY: `beta = 0`); IMPOSED restricts candidates to the unit's own fixed spatial column of 8.
  - **RANDOM:** `networkx.gnm_random_graph(N, M, seed)` (simple, undirected, exactly M).
  - **Infeasible case:** if `M > N(N - 1)/2`, or any arriving unit has fewer positive-weight candidates than its `m_t`
    (kernel underflow; for IMPOSED, fewer earlier column mates than `m_t`, certain once `q > 7`), that ARM is REFUSED for
    all seeds, reported, never repaired by changing the density or the order. A refused RULE or RANDOM refuses T4's
    wiring half; a refused attribution variant leaves its term NOT CLAIMED (untested); a refused IMPOSED is reported
    (it decides nothing in the wiring half).
- **Statistic:** Newman modularity Q of the Louvain partition (`networkx 3.6.1 louvain_communities`, resolution 1.0),
  R = 10 restarts with seeds 13900-13909 on every graph, Q = the maximum; unweighted, symmetrized; the same estimator on
  every arm. Measured once (2.12): at N = 1000, mean degree 20, an Erdos-Renyi graph scores Q = 0.2011 and its twin
  0.1993, a planted block model (20 blocks, 20% between-block edges) 0.7411 -- which is why the red is "vs random
  attachment", never "vs zero".
- **Instrument reds, run before any arm is scored (seeds 13230-13237):** IR1 a planted block model at the same N and M
  must beat RANDOM by t > 1.895 and its Louvain partition must recover the planted one (NMI >= 0.9, CHOSEN, disclosed);
  IR2 RANDOM against a RANDOM twin must NOT pass (the false-positive control); IR3 the planted model at 50%
  between-block edges, reported to locate the detection floor. A failure of IR1 or IR2 REFUSES T4.
- **Decision (the record's literal red):** d_s = Q(RULE) - Q(RANDOM); PASS iff `contrast` is MET at S = 8; otherwise
  RED: the rule-grown organization claim (regions emerge from the wiring rule) is deleted and the owner told; the map
  (T2) is not judged here. **What a PASS licenses is narrow (G4's challenge 1, conceded):** on spatially placed units
  RULE > RANDOM is near-guaranteed by the distance kernel alone (random geometric graphs are strongly modular), so a
  PASS licenses only "rule wiring is more modular than random attachment".
- **Registered attribution contrasts (P8 item 3; each S = 8, t > 1.895 through `contrast`, on the same positions at
  matched N and M; each credits ONE term only if MET):** the shared-partner term iff Q(RULE) > Q(DIST-ONLY, beta = 0);
  the distance term iff Q(RULE) > Q(RULE with a flat kernel); the layer term iff Q(RULE) > Q(RULE without the layer
  target). A term not credited is NOT CLAIMED; it is deleted (owner told) only if G9's task half also fails to credit
  it, so modularity alone never deletes a term the task score might pay for. The draft's earlier degree-preserving
  double-edge-swap secondary is DROPPED: it destroys space as well as structure, so it could only pass.
- **Task half:** RULE vs IMPOSED on the engine's task score, red "below imposed columns" -- G9 pre-registers its bar.

### 2.8 T2 and T3

G4's, by the record's section 6 and the plan's G4 row; this file maps them and registers nothing for them.

### 2.9 Dependencies (what waits; DEPENDENCIES.md lists G7 waiting on G4's T3 for T2/T3)

| item | waits for | proceeds now |
|---|---|---|
| gate P1 (pure core, reds R1-R7 with R2b, R9-R11) | nothing | yes, on fixtures |
| R13 wired (the return path's target check) | G5's port (connection 6) | `require_labeled_targets` on fixtures |
| R8, R12, the census and the debt table | G3's admitting code and the one grower's conversion; the label fields' persistence (2.2) | design; fixture tests |
| the gate's LANDING on main | G3's first code that admits labeled grown units (record section 7; T115 closes then) | stacked in G3's arc with its admitting commit (main's ruling) |
| the save half wired | G6(d)'s save | `check_saved` on fixtures |
| T1's run (the whole family) | G2's capacity-cell problems and ruling 15's task A amendment (the item count of delta_NI); G3's loop, column batch, solved predicate, budget, F_bound and D9 sign clipping; G4's T3 kernel (family level); the gate live; the owner's O9 for any T1-struct deletion | instrument + bookkeeping |
| the gate's admission interface | G3: `admit_rows(rows, ticket)` as the single row-append point of its grown segment (ticket-checked, consume-once) and `known_unit` / `classify_restored` reading its IMMORTAL `'unit'` journal (Codex b3 #8, #9; interface asks registered here, G3 to confirm at its build) | the gate's side on fixtures |
| T4's wiring run | G4's T3 verdict, Part B (tangential kernel, surface density, and -- an ask registered here -- `dbar`, H01's symmetrized simple-graph mean degree over the substrate's layers, for M) and Part B2 (beta); CONN_PROB read through placement (G4) | modularity instrument + IR1-IR3 + the exact-M sampler on fixtures |
| T4's task half | G9 | -- |

AMENDED 2026-10-04 (D22): the T4 wiring-run row's `dbar` wording is superseded -- see the dated pre-data amendment at the end.

**Dependencies this draft found beyond DEPENDENCIES.md** (reported to main, recorded there 2026-10-03): T1 waits on G2
and G3 as well as G4; T4's wiring half waits on G4's T3 and its Part B (tangential sigma, surface density) and the beta
arm on Part B2; the save half waits on G6(d)'s write-ahead verdict rows; the return-path target check (R13) waits on
G5's port.

### 2.10 Seeds (block 13000-13999; grep of `specs/ src/ tests/ tools/` for integers 13000-13999 in a seed context,
with the same filter firing on 4242 (13 hits) and 7000 (10 hits): one hit, a commit hash `2a13221` in PLAN line 2050,
not a seed)

| use | seeds | status |
|---|---|---|
| wave-1 cost probe | 13000-13007 | EXPOSED (used 2026-10-03) |
| gate fixture RNG | 13010-13019 | EXPOSED (dev) |
| T1 power pilot | G3's 9040-9047 (exposed dev, D2: it pairs with G3's PRESIZED there) | EXPOSED (G3's block) |
| spare dev | 13100-13107 | unused |
| (released, D2) T1 confirmatory / reserve | 13110-13117 / 13120-13127 | RELEASED: T1 runs on G3's 9300-9307 (reserve 9308-9315), ratified by main 2026-10-03 |
| T1 instrument reds (the bookkeeping's constructed fixtures) | 13130-13137 | dev |
| T4 dev / confirmatory / reserve | 13200-13207 / 13210-13217 / 13220-13227 | dev / RESERVED / RESERVED |
| T4 instrument reds IR1-IR3 | 13230-13237 | dev |
| Louvain restarts (every graph) | 13900-13909 | estimator seeds, not recordings |

### 2.11 Arc split and the files each part touches (claims filed for the new files; hot files claimed at build)

- **P0 (docs):** this file; the plan's dated notes (the orchestrator edits PLAN-FINAL and tasks.md).
- **P1 (the pure gate):** `src/snn_data_lake/gates/unit_label_gate.py`, `tests/test_unit_label_gate.py`; registrations
  in the commit that adds the module: `run_selfchecks.py`, `src/snn_data_lake/__init__.py` (REGIONS). It is built on
  fixtures now and LANDS STACKED IN G3's ARC with G3's first admitting commit (main's ruling, 2026-10-03), never as a
  dormant EXEMPT module (a gate with no live caller is the dormancy shape this project refuses to fake).
- **P2 (the label fields; D1 agreed with G3):** `growth/placement.py` (`place_labeled`, `labels_by`, `label_of`; claimed
  2026-10-03 on the board),
  then -- as a separate stacked commit AFTER G3's P2 byte-identical lifecycle refactor, R2's `transition_sink` /
  `hydrate_transitions` and R3's `replay_rows` land on that file (D7; main sequences the claim), G3 reviewing --
  `growth/neurogenesis_engine.py` (additive optional arrays for the new fields, no `GROWTH_FORMAT` bump; restore
  recomputes the `origin` cache by `classify_restored` from G3's journal; claimed only then). No placement golden may move (the cortex arm's crc is pinned).
- **P3 (the wire):** G3's admitting code calls `require_admissible`; the debt table's rows retire as G3 converts the
  grower; the assert turns on.
- **P4 (the instrument):** `src/snn_data_lake/instruments/unit_organization.py` (T1 bookkeeping with censoring, the
  two contrasts through `paired_contrast`; T4's modularity scorer with IR1-IR3), `tests/test_unit_organization.py`.
- **P5 (the save half):** G6(d)'s save calls `check_saved`.
Each arc: registrations in its module's commit, the census after every symbol-adding commit, one Codex gpt-6.1-sol
panel before the battery (the orchestrator sends it), a fold stated as re-panelled or battery-verified only.

### 2.12 Cost at real scale

*Measured* 2026-10-03, this host, `.venv-lake` (3.14.6t), one run, seeds 13000-13007, possibly under G1(c) load
(scratch script out of tree, total wall 3.3 s; it built no Brain and imported only `placement`, `arealization`,
`growth_program` and networkx; the live home's newest write is still 2026-09-27, checked after): a gate-shaped check (field presence, map read, depth band, sign rule,
problem lookup) over 100,000 placed records: 0.044 s, 0.44 us per record; the prototype's per-field, insula and
pons-territory refusals fired and the full label was admitted. Louvain: N = 1000, mean degree 20 -- 0.13 s per
restart (ER), 0.06 s (planted); N = 4000 -- 0.94 s and 0.29 s.
*Projections:* T4's wiring half at N = 1000 is about 7 arms x 8 seeds x 10 restarts x 0.13 s = 73 s plus graph
generation (O(N^2) pair probabilities); at N = 4000 about 10 minutes. The gate adds nothing measurable to an admission.
A census of the live lake would read every `_T_GROWTH` blob's `place_*` arrays (73,490 records, 508 MB at the code's
own last count): not measured, never in `run_selfchecks`. T1, from G3's numbers (D2): 400 teach epochs x 4 texts on
the bare taught learner 0.175 s (*measured by G3*, seed 9000), one Kesler LP 2-6 ms, `place(10)` 0.03 ms; a
column-growth attempt 2-10 s per seed-arm (G3's projection), so T1's six arms (COL 4/8/12, SINGLE, RAND-SITE,
PRESIZED) x 8 seeds is about 2-8 minutes plus the dev pilot.

### 2.13 Scope

Built by this row: the gate (P1), its census and scan, the T1/T4 instrument (P4), and -- if D1 resolves so -- the label
fields (P2). Not built: G3's loop, column batch or site rule; G4's kernel, beta fit or the T2 prior; G6's save; G9's task
half; the re-keying of the region-charge and atlas-node rules to labeled populations (POLICIES P3 "ANATOMY IS DATA"
names G7 and G4 for it; the plan's G7 row does not; O8).

## 3. Open questions (for the owner, through main)

*Main's rulings, 2026-10-03:* O5 RATIFIED (T1 inside G3's campaign on 9300-9307, reserve 9308-9315, pilot 9040-9047;
13110-13127 released); the gate lands STACKED IN G3's ARC, never as a dormant EXEMPT module (P3); O1-O4, O6 and O8 are
queued verbatim in the board's OWNER_QUESTIONS.md for one owner brief at the wave's end, and each keeps its registered
fallback below, **pending the owner**; O7 is resolved by G4's Part B2 (D6).

- **O1** May the gate's map read the DECLARED zone of `REGION_COORDS` (resolving neuromodulation from cerebellum in 'r1'),
  or only the neuromere, as the decision record section 3 states? The zone is a declaration, not a GRN computation.
- **O2** The record's strict sentence refuses any position the map cannot tell apart from refused territory -- at
  today's grain every ('sp', 'alar') unit, i.e. every cortex unit. This draft registers the record's fallback (name-based
  refusal there, `region_read = "name"`, disclosed). Confirm, or rule that growth waits for a finer pallial map.
- **O3** Only H01 cortex has a measured E-fraction, so only cortex units can carry a Dale sign and pass the gate.
  Accept "cortex-only admission until another region has a measured sign prior", or rule a sign source.
- **O4** The live clock-driven grower admits unlabeled, problem-less units every refit. Land the gate with a printed,
  shrink-only debt row owned by G3 (this draft), or hold the landing until G3 converts the one grower?
- **O5** (to main, ratify) G3 and G7 agreed (D2) that T1 runs inside G3's campaign on G3's 9300-9307 (reserve
  9308-9315) with PRESIZED re-run there, and this row released 13110-13127. SEEDS.md's "use only yours" is kept in
  letter (each row runs its own block) but T1's decision is computed on another row's seeds: ratify or overrule.
- **O6** (to the owner; G3 co-signs) T1's burden is on the column unit (non-inferiority at 20%, the G-R3 style), and a
  red DELETES the column part of C (record section 6). So an underpowered T1 deletes it. The pre-registered guard is
  power-before-confirmatory (2.6: a dev pilot; projected power < 0.8 at S = 8 -> an S > 8 amendment is requested from
  main before any confirmatory seed). The other reading (red only when columns are SHOWN 20% worse) keeps the unit on
  an underpowered null. The owner should choose before the freeze.
- **O9** (G3 co-signs; queued in OWNER_QUESTIONS.md) T1 as written reds by
  construction on G2's problems (2.6a). May T1-struct's retention contrast stand in for the record's T1 while no problem
  class reaches median U_SINGLE >= 80? Until answered, T1-size is reported UNTESTED and nothing is deleted.
- **O7** (RESOLVED, D6) beta is fitted on H01 by G4's Part B2 with two planted controls; fallback beta = 0 with the
  shared-partner clause reported untested.
- **O8** Scope: P3's "ANATOMY IS DATA" re-keying names G7 and G4; the plan's G7 row does not list it. Which row, when?

## 4. Debate log (wave 1)

| # | challenge | from | answer | resolution |
|---|---|---|---|---|
| D1 | who edits `growth/placement.py` and `neurogenesis_engine.py` for the new fields; is G3's `growth/column_batch.py` a second label store? | G7 -> G3; G3 asked the same | G7 owns the placement extension (`place_labeled`, `labels_by`, `label_of`); G7 writes the two `neurogenesis_engine.py` changes as its own stacked commit after G3's P2 lifecycle refactor, G3 reviewing; `column_batch.py` keeps `batch_id` and the E/I role and NO label | RESOLVED (2.2, 2.11) |
| D2 | G3's four asks (gate signature and when called; problem id; SINGLE's position rule; what T1's 20% bar is on) + G7's refutations: R-a U must count the shared I unit; R-b T1's decision direction; R-c `budget + batch` censoring biases the ratio; R-d T1's PRESIZED unpaired (9100 vs 9300) | G3 <-> G7 | `require_admissible` at admission for every grown unit incl. SHAM, presized not gated; `problem_id` = G3's ledger key = G2's `Problem.id`; COL and SINGLE share one site and one targeting rule; G3 conceded R-a, R-d, accepted R-b, R-c; T1 moves onto G3's 9300-9307 with PRESIZED re-run there, pilot on G3's 9040-9047; G7 released 13110-13127; batch grid {4, 8, 12} primary 8; Part F's S > 8 is an amendment, not a pre-registration line | RESOLVED; O5 and O6 to main/owner |
| D3 | the saved-record clause: field names and one checker | G6 -> G7 | G7 adopts G6's `unit_id, problem_id, sleep_id, verdict`; G-R6's clause reads `check_saved` (no second validator); G6's C15 supplies the planted record; `resolve_sleep` reads G6's `sleep_bout` row on G3's `_T_GOAL` problem record (renamed by G6 from 'sleep' to avoid R2's `_T_EPISODE` kind 'sleep') | RESOLVED; `resolve_sleep` SUPERSEDED by D20's `resolve_committed` |
| D4 | G7 finding: today's prune drops a pruned unit's placement (`neurogenesis_engine.py:503-505`), so a 'pruned' saved record cannot exist to check; G6 asked for an `origin` key in `labels_by` | G7 -> G6; G6 -> G7 | G6 made the sleep save WRITE-AHEAD (every unit in the mask's domain gets its verdict row before the mask applies; its new C18 crash test) and records the incumbent drop as a carried finding in G6(a); G7 added `origin`, with the precision that origin separates born from legacy and does not identify scaffold (G6 now keys legacy populations by explicit module enumeration) | RESOLVED |
| D5 | G5 needs an id-addressable label record and asks whether the gate covers the return path's TARGETS; then asked whether the gate should refuse a target of a different `problem_id` | G5 -> G7 | `label_of`/`labels_by` on the placement record; new `require_labeled_targets` and red R13, called at projection build only, never per tick; the problem-membership filter stays G5's selection rule because `problem_id` is lineage, not ownership (plan line 28; G3's reusable-batch arm) | RESOLVED |
| D6 | G7 -> G4: the position frame (is T3's kernel 3D or tangential? distance only within a map cell? does `growth/wiring_prior.py` store positions of its own?), the T4 split, beta never tuned on modularity. G4 -> G7: CH1 RULE > RANDOM is near-guaranteed by the kernel alone and the double-edge swap destroys space too (attribution); CH2 which positions do the N = 1000 arms use; Q4 who owns um per sheet unit | G4 <-> G7 | G4: T3's verdict on 3-D soma distance, the RULE's kernel refitted in a tangential frame from H01's layer circles (no cortical thickness needed); kernel within a map cell only; `wiring_prior.py` holds no per-unit field; beta fitted on H01 in G4's Part B2 with two planted controls, fallback beta = 0; split accepted (G7 scorer + decision, G4 generator + run). G7 conceded CH1 (attribution contrasts registered, double-edge swap dropped) and CH2 (substrate named, N chosen mechanically against 6 sigma_fitted); Q4 is G7's and vanishes in the contract (x_um, y_um native um on an unbounded sheet per cell). G4's follow-up refutation: G7's 2,300/mm^2 was a cross-section density (x-y contains the radial axis); CONCEDED -- the surface density and the TANGENTIAL sigma are read from G4's Part B record in `tools/g4_t3_campaign.json` | RESOLVED (2.2, 2.7) |
| D7 | R2 holds `neurogenesis_engine.py`'s claim (transition sink + hydration called right after `restore_growth_state`); do G7's arrays restore anything buffer-like? | R2 -> G7 | no overlap: G7's arrays are per-unit label rows aligned with W (unit_id, batch_id, ap, dv, x_um, y_um, region, problem_id, origin); nothing of `_buf_x`, `_buf_y`, `_categories`; restore only stamps `origin = "derived"` in the existing re-derivation branch (superseded at the fold: restore recomputes the origin cache from journal evidence, 2.6b(1); still no buffer overlap); order G3 refactor / R2 / R3 / G7, stacked, main sequencing | RESOLVED |
| D8 | can COL and BATCH9 both be exactly 8 E + 1 I when `place_labeled` has no role and the sign follows the drawn layer? | G3 -> G7 | `place_labeled(..., roles=None)`: default byte-identical; with roles, sign = role and layer from H01's P(layer \| sign); `column_batch.py` DROPS its E/I role (supersedes D1's "keeps the E/I role") and reads it from the label sign; the sign is write-once per `unit_id`, so G3's deposit-time cache is sound | RESOLVED (2.2) |
| D9 | T1's 20 % bound is red by construction on G2's problems (U_SINGLE about 2-4); G3's option (i) batches of 9 singles | G3 -> G7 | (i) entails the mirror PASS and swaps the question; split into T1-size behind a granularity guard and T1-struct (COL vs BATCH9); G3 accepted; G3's own D9 (Dale sign clipping on output weights) registered as complementary | RESOLVED; O9 to the owner (2.6a) |
| D10 | a zero retention margin turns lossless equality into a deletion (G3's Codex #5) | G3 <-> G7 | `delta_NI = max(F_bound, 1 item)` in items of task A's future set; ruling 15 grows A to about 12 items; until then 25 % steps, disclosed; red worded "not shown non-inferior within delta_NI" | RESOLVED (2.6) |
| D20 | (G6's number) `check_saved` must resolve the keeping/pruning sleep from COMMITTED verdicts only (G6's A12, after its Codex #17) | G6 -> G7 | agreed with three amendments G6 accepted: outcomes KEPT/PRUNED, PENDING, REFUSED (label checks first); `resolve_committed(sleep_id, mask_digest)` on the exact pair; reds R9b (a)-(d); restore writes the missing commit row (G6's A12); the lead dropped G6's proposed `completed_by` field (no reader) | RESOLVED (2.3, 2.4) |

## 5. Results

None. Nothing has run but the cost probe of 2.12.

## Review record

- **Lead read:** 2026-10-03, board `p0_reports\G7.md` section "=== LEAD READ ===" (L1-L5) and DEPENDENCIES.md rulings 9,
  9a, 10, 15; folded (2.2, 2.6, 2.6a, 2.6b).
- **Advisor read:** CLI lead session, Fable, 2026-10-03 (run by the lead).
- **Codex critique (the review of THIS draft):** batch 3, `scratchpad\codex_p0\b3.critique.md` (G7 findings 8, 9, 21,
  22, 23, 24, 28), reply file `scratchpad\codex_p0\b3.reply.txt`, sha256
  `1c8abc4a99bfa18c7e1d97d042bb1240c8b4cf8f893f88277bec5c38fae52e57` (re-hashed by this row 2026-10-04), model
  gpt-6.1-sol; a static review (no runs). Dispositions, all FIXED in text, none refuted:
  - #8 BLOCKER, coverage census -> 2.3 coverage anchored on `GROWN_SEGMENTS` + a row-growth scan + the runtime census
    (its red = any derived/legacy row in a segment with no debt row; the advisor's fold read caught that the first fold's
    `rows > born + derived + legacy` could never fire, the partition sums to `rows`);
    the `AdmissionTicket` makes the gate precede the append structurally; reds R14, R15; 2.6b(3) marked superseded.
  - #9 BLOCKER, `unit_id` uniqueness -> `known_unit` against G3's journal, within- and across-batch refusals, restore
    classified from journal evidence by `classify_restored`; reds R16-R18.
  - #21 MAJOR, exact sign-flip reporting -> 2.6b(7): all 256 flips under this draft's verdict rule, reference rate
    labelled separately, constant-positive and constant-zero fixtures.
  - #22 MAJOR, granularity guard -> per judging problem, censored values excluded, whole-contrast REFUSED, no selection.
  - #23 MAJOR, T1-struct decision -> one exact table; burden-of-proof NI failure is the red, worded without a harm claim;
    the U-REFUSED cases stated; deletion pending O9.
  - #24 MAJOR, T4 graph construction -> M rule from H01's `dbar`, symmetrization, the exact-M Efraimidis-Spirakis
    sampler, the infeasible-case disposition, REFUSED when no N meets 6 sigma.
  - #28 MINOR, H01 marginals under forced roles -> imposed composition disclosed; unconditional marginals claimed for
    `roles=None` only.
- **Batch 1's G7 review was VOID for this row:** the lead's prompt builder sent the decision record
  `g7-organizing-unit.md` instead of this draft (that review's own finding 4 caught it). Its findings 1 (one
  position-to-region contract with UNKNOWN handling) and 3 (enumerate birth paths; keep T115 OPEN; classify restored units
  from journal evidence) were read anyway and folded (2.2's frozen contract and R19; 2.3's T115-OPEN clause and
  `classify_restored`); its finding 2 (a zero NI margin) is D10. They are recorded as read, not as a review of this text.
- **Folds since the critique:** the fold is itself unreviewed text -- the Codex critique reviewed the PROPOSED version,
  so every change listed above is battery-free design text that no reviewer has re-read; stated plainly, per the
  project's rule.

## DATED PRE-DATA AMENDMENT 2026-10-04 (with G3's and G4's replies; no data read)

Amends the frozen text of `854de0e7`; the passages it supersedes carry one-line `AMENDED 2026-10-04` pointers and are
otherwise unchanged. Raised by G3's and G4's replies to this row's cross-row asks (2.9); nothing has been run, measured
or read since the freeze. Unreviewed amendment text, stated plainly.

**D21 -- a restored label that contradicts its birth record is REFUSED (G3's concern, agreed).** In `classify_restored`
(2.3) and the origin partition (2.6b(1)): a restored row whose `unit_id` HAS a journal birth entry but whose label
differs from that entry in any field is a TAMPERED or corrupted born unit and is **REFUSED**, the red naming the unit
and the differing fields. It is NOT classified `derived`: that would launder the change into a debt row. `derived` now
means exactly: the FULL label field set present and NO journal birth entry for its `unit_id`. `born` and `legacy` are
unchanged. This is consistent with the write-once sign (2.2): a journaled unit's label is write-once as a whole. Red
R18(c) changes accordingly: a journaled unit whose restored label differs -> REFUSED, naming the unit and the differing
fields (was: `derived` with the mismatch printed). G3 confirms `classify_restored` at its build as a dated pre-data
amendment on claude/g3.

**D22 -- `dbar` and the T4 layers pinned (G4's two definitions, agreed).** In 2.7's M bullet: `dbar` is the mean degree
of the INDUCED subgraph of H01's symmetrized simple graph on the units of the T4 layers (an edge counts only if both
ends are in those layers), read from `tools/g4_t3_campaign.json` Part B and taken only after T3 reads (G4's A13).
**The T4 layers** are the cortical layers `CONN_PROB` has a population for: **L1 excluded** (no `CONN_PROB` population,
so an L1 unit would have no defined layer term and its every arrival would be infeasible) and **white matter
excluded**. The T4 substrate's `place_labeled` draws (`roles=None`) are restricted to the same layers by rejecting draws
outside them, disclosed; so the substrate's layer mix is H01's occupancy conditioned on those layers, not H01's full
marginal. Beside `dbar` G4's record carries N_units, the undirected simple edge count, the reciprocal-pair fraction and
the reconstruction-density disclosure (H01's observed density 3.46e-4, far below biological, which `dbar` inherits);
this row prints that disclosure beside every T4 result. At a sparse `dbar` the kernel arms are near-trivially local;
that only sharpens the already-registered narrowness of a PASS ("more modular than random", CH1). G4 confirms `dbar`
in its Part B amendment on claude/g4.

**2.9, the T4 wiring-run row, superseded:** waits for G4's T3 verdict, Part B (tangential kernel, surface density, and
`dbar` = the mean degree of the induced symmetrized simple-graph subgraph on the T4 layers -- L1 and white matter
excluded -- taken after T3, D22) and Part B2 (beta); CONN_PROB read through placement (G4). Proceeds now: modularity
instrument + IR1-IR3 + the exact-M sampler on fixtures.

**Debate log rows added:**

| # | challenge | from | answer | resolution |
|---|---|---|---|---|
| D21 | a restored row with a journal birth entry but a different label, classified `derived`, launders tampering into a debt row | G3 -> G7 | REFUSED as a tampered born unit, naming the unit and fields; `derived` = full label set, no journal entry; R18(c) amended | RESOLVED (this amendment) |
| D22 | `dbar`: induced subgraph or all-partner degree? which layers, and does L1 count? | G4 -> G7 | induced subgraph on the T4 layers; L1 excluded (no `CONN_PROB` population) and white matter excluded; substrate draws restricted to the same layers; density 3.46e-4 disclosed | RESOLVED (this amendment) |

## DATED PRE-DATA AMENDMENT 2026-10-04 (b) -- lead ruling 23 and the owner's answers; found by the build; no data read

Nothing confirmatory has run; every number below is from the build's own fixture tests (dev seeds 13010-13019,
13230-13237) or the owner's answers as relayed by the lead. Unreviewed amendment text, stated plainly.

**The owner's answers (relayed by the lead, 2026-10-04).** O9: the T1 split is ACCEPTED -- T1-size is REFUSED
below median U(SINGLE) 80 (per judging problem, 2.6a) and T1-struct's retention contrast stands in for the record's
T1; T1-struct's deletion authority therefore now exists (2.6a's table applies with `owner_o9 = True`). O1-O3:
admission is the NAME fallback in the ambiguous ('sp', 'alar') cell (`STRICT_TERRITORY = False`, rule (4) of 2.2's
contract) and cortex only.

**`place_labeled`'s signature as built (2.2; the frozen signature was underspecified).** `place_labeled(n, *,
unit_ids, region, site, problem_id, batch_id, seed, centre_um, density_per_mm2, roles=None)`: `unit_ids` (G3
supplies them, one per unit); `region` (selects the arm; `roles` outside cortex REFUSED, O3); `centre_um` (the batch
centre from G3's site rule); `density_per_mm2` with NO default (G4's Part B records it; the build's fixtures use a
labelled 2.7e4). Tangential offsets come from an RNG stream separate from the laminar one, so `roles=None` stays
byte-identical to `place()` (asserted against `place(50)`; placement's golden self-check passes).

**T4 fix 1 -- sequential partner draw (2.7's sampler).** The frozen one-shot Efraimidis-Spirakis keyed draw held every
shared-partner count at 0 (an arriving unit has no neighbours when its keys are drawn), so beta was inert and the
shared-partner contrast could never be credited. Partners are drawn ONE AT A TIME, re-weighting after each pick; with
fixed weights this has the keyed draw's distribution, so the other arms are unchanged. Instrument red (fixture: flat
kernel, N = 300, M = 1200, seeds 13233-13235): triangles 607 at beta = 0 vs 1510 at beta = 5.

**T4 fix 2 -- RANDOM-SEQ.** The sequential generator alone is not uniform (older arrivals collect edges: 607
triangles vs about 86 expected for G(n, m) at that density), so RULE vs gnm mixes the rule with the arrival process.
Added arm RANDOM-SEQ: the same sampler with every term flat (kernel 1, no layer term, beta 0). **RULE is credited --
T4's wiring half PASSES -- only if it beats BOTH gnm (the record's literal red) AND RANDOM-SEQ**, each by 2.7's
contrast at S = 8 on the T4 seeds.

**The concrete T4 layer term (D22, made concrete).** `CONN_PROB` is Potjans-Diesmann's 8-population table
(`reservoir/cortical_microcircuit.py`, C[target][source], L2/3 merged), not an H01 table: a unit's population is
(layer, Dale sign) -> {Layer 2, Layer 3} -> L2/3, L4, L5, L6, each e/i; Layer 1 has no population.

**The scope rule (ruling 23).** P4d's "grown unit" is a unit grown on the ENGINE'S GROWTH PATH. Row growth found
elsewhere by the row-growth scan -- `readout/growable_readout.py::GrowableReadout._grow_one`,
`readout/laminar_local_readout.py::LaminarLocalReadout.grow`, `reasoning/developmental_loop.py::GrowableSTDP.
maybe_recruit` -- is OUT of the gate's scope by a rule the gate PRINTS on every run (`OUT_OF_SCOPE_ROW_GROWTH`),
listed for the owner, never counted as debt; a scope row that stops growing rows reds as stale.

**networkx** (3.6.1 in `.venv-lake`) is not a declared dependency: the owner's decision; the instrument imports it
lazily (Louvain modularity and gnm only).

| # | challenge | from | answer | resolution |
|---|---|---|---|---|
| D23 | the frozen one-shot keyed draw makes beta inert; the arrival process confounds RULE vs gnm | G7 build -> lead | sequential draw; RANDOM-SEQ added, RULE credited only if it beats both | RULED (lead, 23) |
| D24 | unlabeled row growth outside the engine's growth path (readout, reasoning) | G7 build -> lead | out of scope by a printed rule, listed for the owner | RULED (lead, 23) |

## DATED PRE-DATA AMENDMENT 2026-10-04 (c) -- IR2's twin seeds and the one NMI home (the cp3 fold; no data read)

Nothing confirmatory has run. Unreviewed amendment text, stated plainly.

**IR2's twin seeds (2.7's instrument reds, 2.10's seed table).** IR2 compares RANDOM against a RANDOM TWIN, and a
twin needs its own seed stream: **13240-13247** (dev, instrument red; one twin per IR seed 13230-13237, paired in
order). Inside G7's block 13000-13999; a grep of `specs/ src/ tests/ tools/` for 13240-13247 (with the same filter
firing on 4242: 37 hits) finds only `instruments/unit_organization.py`'s own constant and two decimals
(0.13241, 0.13244) in `tools/t078_attention_schema.json`, which are values, not seeds. 2.10's table gains the row:

| use | seeds | status |
|---|---|---|
| T4 instrument red IR2: the RANDOM twin | 13240-13247 | dev (exposed) |

**The NMI bar stays frozen; its computation has one home.** IR1's recovery bar is unchanged (NMI >= 0.9 on every IR
seed). NMI is computed by `gates/growth_coverage_gate.nmi`, beside that module's `adjusted_rand` -- the tree's one
partition-agreement home (P2, lead ruling on cp3) -- numpy-only, arithmetic normalization (scikit-learn's default),
checked against a hand-computed value and against scikit-learn on an IR fixture; no ARI substitution.

## DATED PRE-DATA AMENDMENT 2026-10-06 (d) -- unit_id and batch_id are non-negative integers (G3 refutation; lead-approved; no data read)

G3 read the gate against its journal and found a representation mismatch: 2.x's identity checks required non-empty
STRINGS (`gates/unit_label_gate.py` `_check_unit_id` / `_check_batch_id`), while G3's write-ahead birth row carries
`int(unit.unit_id)` / `int(batch_id)` (`growth_loop.py:211-228`, `grown_code.py:40`). Two representations of one id is a
bug farm, so the gate now accepts **a non-bool int >= 0 only** and refuses strings, bools, negatives, floats and None
for both fields. Nothing else moves: `AdmissionTicket` / `classify_restored` compare by `==` (ints work unchanged);
the field order, the duplicate rules and every other check are untouched. New red: each of `"7"`, `True`, `-1`, `1.0`,
`None` is refused naming the field (tests/test_unit_label_gate.py, `test_ids_are_non_negative_ints_only`). Fixtures
re-keyed to ints. Agreed interface with G3 (birth row carries every LABEL_FIELDS key; `known_unit`, `birth_records`,
`first_sign` over the immortal 'unit' rows; G3's `admit_rows` sets GROWN_SEGMENTS' admit point, its two debt rows retire
later, `restore_growth_state` in G7's neurogenesis_engine commit). Unreviewed amendment text.

## DATED PRE-DATA AMENDMENT 2026-10-06 (e) -- owner decision A2 answers O9: T1-struct is registered, its retention half stands in (no T1 data read)

**What already stood.** 2.6a (written 2026-10-03, joint G3+G7) already registers T1-struct (COL 8E+1I vs BATCH9, nine
independent units placed by the same `place_labeled(9, roles=("E",)*8 + ("I",))` call at the same site, same targeting
rules), its exact decision table, G3's seeds 9300-9307 (reserve 9308-9315, pilot 9040-9047; ruling 5) and the per-problem
granularity guard (U* = 80 for k = 8). It was marked PENDING OWNER O9. This amendment adds no clause; it records the
owner's answer and makes the table's "after O9 = yes" column the live one.

**The owner's answer.** OWNER_DECISIONS_2026-10-04.md item A2 (the owner's delegation, ratified in the owner's words
2026-10-06): "ACCEPT the T1 split. T1-size is REFUSED (untested, nothing deleted) unless the median single-neuron U >= 80;
T1-struct (column vs 9 independent units) is pre-registered and its retention half stands in meanwhile." So: O9 = YES.
T1-struct's retention contrast e'_s stands in for the record's T1 while no judging problem passes the guard; T1-size is
UNTESTED and deletes nothing in that case.

**What is live, stated once, nothing re-chosen.**
- U half (T1-struct and T1-size): non-inferiority bound 20 % (d = log 1.2 on the log unit ratio), behind the
  per-problem guard of 2.6a; guard REFUSED = untested, never a red and never a pass.
- Retention half e'_s = delta_NI - (F(COL) - F(BATCH9)), reads at any size; paired one-sided t over S = 8 > 1.895;
  degenerate (zero-variance) samples read by ruling 9a (common value > 0 MET, <= 0 NOT MET, "degenerate" printed);
  exact sign-flip rate printed (2.6b).
- Seeds: G3's 9300-9307 (reserve 9308-9315), computed inside G3's campaign and labelled as such in both
  pre-registrations; nothing runs on G7's own block for T1.
- What a RED deletes: the column STRUCTURE (the shared inhibitory unit with E->I and I->E wiring that the frozen COL
  arm adds over BATCH9) -- NOT the column's neuron count, and nothing owner-drawn -- and the owner is told. The frozen
  COL definition carries no E->E wiring; none is added here. T1-size's red keeps the record's disposition (2.6).
- The condition A2 set (a dated pre-data amendment BEFORE any T1 data is read) holds: no T1 arm has run, no pilot
  reading of U(SINGLE) exists in this worktree, and the dev seeds are G3's.

Unreviewed amendment text; G3 (co-signer of O9) refutes it before the lead reads it.

## DATED PRE-DATA AMENDMENT 2026-10-06 (f) -- (e)'s three omissions and T1's judging set (G3's refutation; no T1 data read)

Nothing confirmatory has run. G3 refuted (e) against its frozen g3-growth-loop.md and found three omissions; (e) is not
rewritten, it is completed here (additive). Unreviewed amendment text, stated plainly.

**(e) completed.**
1. **T1 waits on G4's T3 result.** G3's frozen text registers: no T1 arm (T1-size or T1-struct) runs before G4's T3 result;
   the kernel-free exemption does NOT apply to T1; any exception needs a board amendment. 2.6's family-level wait (T1 waits on
   G4's T3) is unchanged. Owner decision A2's "its retention half stands in meanwhile" means the retention READING may stand
   in for the record's T1 once it exists; it does not lift this wait.
2. **e'_s's precondition is pending.** Dale clipping on the weights (G3's D9) is half landed: `clip_to_sign` and the restore
   refusal exist (`growth/grown_code.py`, G3 cp1 #17); the DEPOSIT-site wiring (every R-STDP deposit onto the grown segment)
   waits on the learner port in `readout/interaction_learning.py` (R2's P1 first), and the reservoir-segment clip waits on
   G3's P5. Until both land, e'_s is NOT runnable and T1-struct has no reading.
3. **Naming.** The owner's A2 answers what this record calls O9 and G3's record calls O10 ("O10 = G7's O9").

**T1's judging set (2.6a: "fixed by G2 before the pilot"; neither frozen text names ids; G3 and G7 agree below, G2 reports
counts only).** The problems' own tables (g2-taught-problems.md, cells S0C0 / S1C0 / S0C1 / S1C1: state x capacity needs;
B5: the incumbent solves S0C0 on every seed) decide it; no dev reading of A_ii enters:
- **U / granularity-guard set J = { i.S0C1 }** now. It is the growth-licensing capacity cell with no state need. i.S1C1
  (capacity AND state) is added to J by a dated amendment when G3's state branch lands (G3's amendment 98d92203: no memory
  unit is grown in any confirmatory run until R2's ctxread kind and G3's reconstruct_state test land). i.S0C0 (no state, no
  capacity; the incumbent solves it, U = 0 by construction) and i.S1C0 (state only) are not capacity-growth problems and are
  not judged for U. c1d is a retention task B, not a U-judged problem.
- **Retention task A for each B (ruling 15 / G2's table):** B = c1d -> A = ac1d; B = i.S0C1 -> A = ii.twin (the separable twin).
- **(ii) problems** (ii.xor, ii.twin, aii): disposition follows G2's A_ii attribution (pending); nothing is excluded or
  admitted on an unattributed dev reading.
- Future items counted in distinct items, reproduced at seeds 8000 and 9300 (tp.PROBLEMS[id].items(seed, "future"), G2's
  head 99d6193d): ac1d 12, c1d 12, ii.* 4, i.* 8 (identical across the two seeds).
