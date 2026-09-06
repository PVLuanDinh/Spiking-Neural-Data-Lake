# The gates for X1, L5 and C5 (2026-09-06) — pre-registration

The owner ratified three rows on 2026-09-06 (`PROPOSED-ROWS-2026-09-06.md`, plan Phase 2 / 3 / 4, tasks T088–T090,
reds X-R1 / L-R6 / C-R5) and then asked for their gates. Each row binds ONLY once its gate exists and its output has
been COUNTED on the tree — the cross-cutting invariant, and the reason this arc exists. Lessons carried from this
session: a control must run in the mechanism's own REGIME; a mechanism that moves what a gate reads must keep the
gate's instrument fixed; the closing record commit is the last commit before a push; and — the one that reshapes a
row below — grep the tree before building, because the mechanism may already be there.

## 1. Territory (code at `cd4c02a`, read 2026-09-06)

### 1.1 Where a new gate may live, measured

- **The tools zone is RATCHETED and FULL.** `tools/p4c_ratchet.py --assert` reads 30 ungoverned builds under
  `tools/` against a baseline of 30, and the backlog "may only fall". A new `tools/*.py` would take it to 31 and red
  the ratchet. **The rows' named paths `tools/capability_bar.py` and `tools/conclusion_protocol.py` are therefore
  not buildable as written**; both gates move to the governed tier under `src/snn_data_lake/substrate/`. Amending a
  ratified row's named path is the session's call under the 2026-09-05 authorization, disclosed here and in the
  plan rows.
- **The policy manifest takes an unclaimed-gate census.** `policy_manifest._selfcheck_modules()` collects every
  `substrate.<module>` registered in `run_selfchecks.py`, keeps those matching
  `GATE_NAME_RE = (gate|guard|envelope|manifest)`, and fails when one of them is not named anywhere in
  `docs/POLICIES.md` (`UNCLAIMED_GATE`, baseline empty). These three are PLAN-row instruments, not policy gates:
  naming one `*_gate` would demand a POLICIES edit that no policy asked for. **They are named without the census's
  tokens** (`capability_bar`, `immortal_retention`, `conclusion_protocol`), which keeps the census honest rather
  than silencing it — and the row L5's own name in the plan (`substrate/immortal_class_gate.py`) is amended to
  match. **This bullet's last sentence was WRONG and the panel measured it.** It read "reachability for P3 comes the
  way every gate's does: a `run_selfchecks.py` registration, which is a live root of `dormancy_gate`." It is
  not. `dormancy_gate._imports()` is an AST walk over `Import`/`ImportFrom` nodes only, and `run_selfchecks.py`
  registers modules as STRING literals in a list — so the registration confers no reachability whatsoever, and
  the gate reported all three as DORMANT the first time it ran. The three take the disposition their siblings
  already carry: `EXEMPT` / `permanent`, the same row `policy_manifest`, `lake_rebuild_gate`, `replay_gate` and
  `determinism_gate` hold, each with the condition that would retire it. See §3.4.3.

### 1.2 L5's mechanism ALREADY EXISTS — the row is re-scoped (P2, the find that reshapes this arc)

`substrate/lake_engine.py` opens: *"every record class in the ONE lake is labeled IMMORTAL or COMPACTABLE at
introduction, here, in RETENTION. 'Remember everything' means an appended record is either retained forever or
provably reconstructible before any compaction reclaims it — COMPACTABLE is a claim about reconstruction, never
permission to forget."* `RETENTION` holds eleven tags today, each with a label, a `reclaimable` flag and a reason;
`AppendBudget` already carries the bytes/tick flood assert against `tools/lake_append_baseline.json`;
`lake_migrate.TAG_DISPOSITIONS` is the second view of the same census and `lake_engine` already asserts the two
agree. **So L5 must NOT introduce a class — that would be two mechanisms for one job.** What the row promised and
the tree does NOT yet enforce, measured:

1. **No eviction path is held to the label.** The registry LABELS; the docstring says of the reclaimable flag "it
   licenses reclaim, it never performs it". Whether any reclaimer exists today is counted by the gate, not assumed.
2. **No cross-process proof over the CLASS.** Restart proofs exist per record kind (the growth blob, the state
   frames); nothing asserts that every IMMORTAL tag written before a process boundary returns after it.
3. **No measured storage growth curve**, so "never forgets" ships without its price.
4. **A finding the row must carry:** the brain's grown substrate is **not** immortal today. `_T_GROWTH` (0x03) and
   `_T_GROWTH_DELTA` (0x08) are COMPACTABLE-with-a-reconstruction-proof; the L1c batch instrument rides inside those
   records. Under the registry's own contract that is legitimate ("provably reconstructible"), and it is exactly the
   sentence the owner's "never forgets" needs to be measured against rather than assumed.

### 1.3 X1 and C5 have no incumbent

`graph-exists` finds no capability/protocol/provenance registry and no declaration registry for the consciousness
instruments. The nearest kin, and the pattern both gates copy, is `GRADIENT_CLAIM` — a module-scope literal dict read
STATICALLY by a gate that never imports the module. C5's first counted run is expected to red the live surface SC-009
already names: `reasoning/consciousness_probe.py` (whose `probe_access` / `probe_integration` markers the audit found
without a reachable red) and the cockpit's PCI and integration rows.

## 2. Pre-registration (written before any gate code)

### 2.1 `substrate/capability_bar.py` (X1's gate, T088)

**Mechanism.** A module-scope literal `CAPABILITY_BAR` registry: one row per claim, each
`(task, protocol, human_number, human_provenance, brain_number, brain_provenance, date, verdict)`, read statically by
the gate's own AST reader (the GRADIENT_CLAIM pattern — the gate never imports the registry's module). `--assert`
fails when: a brain number is present without a protocol or a provenance; a human number is present without a
provenance that either names a measurement on this host with its protocol recorded, or quotes the cited protocol's
text; a row's verdict is a comparison word (`parity`, `above`, `below`) while either number is absent; or any tracked
owner-facing surface (the cockpit, `docs/*.md`, the specs' PROVENANCE rows) prints one of the registry's task names
beside a comparison word without a complete row. **Refusal arm, first-class:** a row whose human number cannot be
obtained is written with `human_number = None` and `verdict = "REFUSED: no human number under a matching protocol"`,
and the gate then FAILS any surface that uses a comparison word for that task at all.

**Reachable red (constructed, in the self-check):** four synthetic registries — a brain number without a protocol, a
cited human number without quoted protocol text, a comparison verdict with a missing number, and a surface line using
a comparison word for a REFUSED task — each must be caught; and a complete row plus a clean surface must pass (the
positive control, so the gate is not refusing everything).

**Counted output on today's tree, expected:** zero rows and the refusal message — the registry starts empty, because
no capability claim has been earned. That is the count the row binds on.

### 2.2 `substrate/immortal_retention.py` (L5's gate, T089 — re-scoped by §1.2)

**Mechanism.** The gate polices `lake_engine.RETENTION` and the paths that could reclaim against it. `--assert`:
(a) every tag in `RETENTION` has a label in {IMMORTAL, COMPACTABLE}, a reason of at least N characters, and a
`reclaimable` flag consistent with its label (an IMMORTAL tag with `reclaimable=True` is a contradiction and fails);
(b) the census agrees with `lake_migrate.TAG_DISPOSITIONS` in both directions (already asserted by `lake_engine`'s
own check — the gate re-runs it here so one command reads the whole contract); (c) every RECLAIMER in the tree is
enumerated by an AST scan for calls that delete, truncate or rewrite lake records, and each must consult
`retention_of` in the same function — **and the count of reclaimers is PRINTED, so an honest zero is visible rather
than a silent pass**; (d) a cross-process round trip over the class: a scratch home, one record appended per IMMORTAL
tag, the process boundary crossed, every record returned byte-identical; (e) the growth curve is measured, not
projected: bytes per thousand ticks per IMMORTAL tag over a recorded scratch run, written to
`tools/immortal_growth_baseline.json` and ratcheted thereafter.

**Reachable red (constructed):** an IMMORTAL row flipped to `reclaimable=True`; a tag present in one census and not
the other; a synthetic reclaimer function that drops records without consulting `retention_of`; a round trip in which
one immortal record is withheld. Positive control: the real registry and a real round trip pass in the same command.

**Counted output on today's tree, expected:** eleven tags (seven or more IMMORTAL, two COMPACTABLE with their
reconstruction proofs), reclaimers found = 0, and the growth curve's first measurement. The row's honest sentence,
which the record will carry: **the grown substrate is COMPACTABLE-with-a-proof, not immortal.**

### 2.3 `substrate/conclusion_protocol.py` (C5's gate, T090)

**Mechanism.** A module-scope literal `CONCLUSION` registry: one row per C4 instrument, each
`(instrument, licensed_sentence, refused_sentence, dissociation_input, reading, date)`, read statically. `--assert`
fails when: an instrument named in the plan's C4 row has no registry row; a row lacks its dissociation input; a
tracked surface prints an instrument's reading without its licensed sentence within the same view; any code computes
an aggregate across two or more instrument readings (an AST scan for arithmetic over registry-named symbols); or a
chat/answer path answers a consciousness question from a stored string rather than from a measured quantity — the
last is the live defect SC-009 already names.

**Reachable red (constructed):** a registry row missing its dissociation; a synthetic surface line printing a reading
alone; a synthetic aggregate expression; a synthetic answer path returning a constant. Positive control: a complete
row and a compliant surface pass.

**Counted output on today's tree, expected:** seven rows MISSING (C4 is unbuilt — the registry ships with the seven
instrument names and empty readings, which is the honest state), and the live findings against
`reasoning/consciousness_probe.py` and the cockpit's PCI / integration rows, printed and counted. The row binds on
that count; the findings themselves are C4's work, not this arc's.

### 2.4 What this arc does NOT do

It does not build C4's instruments, does not set a capability target, does not name the immortal class differently
from `RETENTION`, and does not fix the consciousness answer path — each of those is its own row's work. It builds
three instruments and counts their output, which is what the invariant asks before the rows bind. Every gate ships
with its constructed reds in its own `main()` (the tree's self-check idiom), a `run_selfchecks.py` registration, and
a brain-free test file; none of them touches the safety envelope, the daemon or the live lake.

---

## 3. Counted output on the tree (2026-09-06, all three built and asserted)

The invariant this arc exists to satisfy: **implement the gate and COUNT its output before the rule lands.** Below is
what each gate printed on today's tree, and where the count differs from §2's expectation.

### 3.1 X1 — `substrate/capability_bar.py --assert`

```
CAPABILITY BAR -- 0 row(s); tracked surfaces: 5
  (empty: no capability claim has been earned.)
findings: 0
```

As pre-registered. The register ships EMPTY and that is the honest state: no capability claim on this tree carries a
protocol, a human number with a measured or quoted provenance, and a brain number beside it. The gate's value today is
the surface arm — five tracked surfaces are scanned for comparison words attached to an incomplete or REFUSED row, and
none fires because there are no rows to compare against. The four constructed reds fire in `main()` and in
`tests/test_ratified_row_gates.py`; the empty register is therefore a measured zero, not an unplugged detector.

### 3.2 L5 — `substrate/immortal_retention.py --assert`

| Measured | Value |
|---|---|
| lake tags labelled | 11 |
| IMMORTAL | 9 (`0x01 0x02 0x04 0x05 0x06 0x07 0x09 0x0a 0x0b`) |
| COMPACTABLE | 2 (`0x03 _T_GROWTH`, `0x08 _T_GROWTH_DELTA`) |
| reclaimers of durable records that do not consult the label | **1** (`substrate/home.py::consolidate`) |
| scratch-home deletions EXCLUDED and listed | 8 |
| immortal tags with an append baseline | 3 of 9 |
| cross-process byte-identity findings | 0, and its red fires |

Three things in that table are disclosures, not passes.

**The grown substrate is COMPACTABLE, not immortal.** `_T_GROWTH` and `_T_GROWTH_DELTA` carry a
reconstruction proof instead of a never-evict guarantee. The registry's own contract permits it, and the gate prints
the fact on every run so it can never be quietly assumed. "Never forgets" in this project means *never evicts the
IMMORTAL class*, not *stores everything*.

**The zero was wrong, and it was this project's archetypal failure.** The first shipped scan printed *0 durable
reclaimers*. The adversarial panel of 2026-09-06 measured that `_RECLAIM_CALLS` held eight destructive call names
and not `rewrite` — and **this repo reclaims by rewriting**: `KnowledgeLog.consolidate` (`substrate/home.py:558`)
destroys records by rebuilding the payload list and calling `BinLog.rewrite`, using none of the eight. So the zero
was a claim about the instrument. Worse, §2.2 of this very file had pre-registered the arm as "an AST scan for calls
that delete, truncate **or rewrite** lake records" — the rewrite arm was dropped during the build and the result was
then recorded as measured.

With `rewrite` added, the census reports **one** durable reclaimer, and it is live rather than latent-in-theory: the
sleep tick (`cortex/brain.py`, every eighth night when `_persist`) runs `LakeConsolidation().consolidate(records,
trace=RecallTrace())` — *use it or lose it*, learned records capped at 250,000 and re-derivable at 3,000 — then
`KnowledgeLog.consolidate(survivors)`. `_T_FACT` is tag `0x01`, labelled IMMORTAL with the reason "learned knowledge
is never reclaimed". **The declared class and the running brain disagree**, so the sentence "the lake never evicts
the IMMORTAL class" is FALSE on this tree, and the gate now prints exactly that sentence whenever the debt table is
non-empty.

The drop is inside `if len(learned) > learned_cap`, so it is LATENT while the learned count stays under 250,000. A
latent reclaimer is precisely what a gate is for — P3's *reachability is not charge* read in the other direction.

It is carried in `KNOWN_UNLABELED`, which is the opposite of the scratch exclusion: an entry is a red held in the
open with its owning plan row and its retirement condition, it is PRINTED as debt on every run, `findings()` asserts
on any reclaimer *not* listed, and `stale_known()` asserts on any listed entry that stops reclaiming. The table may
only shrink. **Row L6 (T091) owns the fix** — either the learned cap stops dropping a non-reclaimable class, or
`0x01`'s label is corrected to say what is true. It is explicitly not fixed by widening the scratch markers.

The earlier instrument errors stand as recorded: a substring scan matched `str.removeprefix` (13 false findings), and
an exact-call-name scan then matched 8 deletions that only ever touch a scratch home. The final form walks AST `Call`
nodes for exact names and **lists all 8 scratch exclusions in the report** so the exclusion is auditable.

**The process-boundary arm was a fixture, and now it is not.** As first shipped, `round_trip()` wrote its own
framing with a plain `open()` and read it back in the same interpreter — no lake code under test, no second process,
and no input that could make it fail. A criterion with no reachable red is what P8 forbids, and all three panel
lenses caught it independently. It now uses the tree's own idiom (`binlog.py`'s PROPERTY 3): a **child interpreter**
appends one record per IMMORTAL tag through `BinLog` into a scratch home, and this process re-opens the same file
with a fresh `BinLog` and checks byte identity per tag. `withhold` names a tag the child skips, and the self-check
runs that red FIRST — measured: withholding `0x01` produces `tag 0x01: an immortal record did not survive the process
boundary byte-identical`. The empty return is believable only because of it.

**Six of nine immortal tags have no append baseline.** That is real, open debt, printed rather than asserted: the
budget arm can only bound growth for tags it has a baseline for. It is L5's own follow-on work, and naming it here is
what stops the 3-of-9 from reading as coverage.

### 3.3 C5 — `substrate/conclusion_protocol.py --assert`

```
CONCLUSION PROTOCOL -- 7 of 7 instrument rows; missing []
asserted findings (rows + the aggregate refusal): 0
reported, not asserted -- surfaces owed to C4/T081: 1 (baseline 1, may fall never rise)
  - src/snn_data_lake/substrate/cockpit.py:920: integration's reading (integration_competence)
    printed without its licensed sentence in view
```

All seven rows are filled, each with its licensed sentence, its refused sentence and its dissociation input, and every
`reading` is `None` — C4 is unbuilt, and a reading without its instrument would be a fabrication.

**Where this differs from §2's expectation, and why.** §2 expected findings against `reasoning/consciousness_probe.py`
*and* the cockpit's PCI and integration rows. The measured count is **one**. The first run flagged six, of which five
were the word "integration" appearing in prose or in a directory name — a bare instrument NAME is not a reading. The
scan was re-keyed onto each row's declared `surface_symbols` (how that instrument's number actually reaches a reader),
and only the cockpit's `integration_competence` print survives. The consciousness probe does not appear because it
prints nothing under a declared symbol: **that is a limitation of this instrument, not a clean bill for that file.**
C4/T081 retires the probe; this gate does not audit it today.

**Refusal 3 now arms itself.** As first shipped, the surface scan keyed on each row's declared
`surface_symbols`, and six of seven rows declare none — so the refusal covered exactly one instrument, and the
baseline of 1 was measured through that blindness (panel finding, corrected severity MINOR). Each row now names the
C4 task that builds it, and a row whose task is marked `[X]` in `tasks.md` while declaring no surface symbol is a
finding. Landing T077–T082 without saying how the number reaches a reader reddens this gate instead of silently
leaving the arm off. The red is constructed by pointing a symbol-less row at T034, a task that IS complete on this
tree; the unarmed control stays clean.

Surface findings are REPORTED against `SURFACE_BASELINE = 1` and may fall, never rise. They are C4/T081's defects, not
this arc's, so they are visible and non-worsening rather than blocking every battery. The rows and the
aggregate refusal ASSERT.

### 3.4 Deviations from the pre-registration, disclosed

1. **All three gates live in `substrate/`, not the paths the ratified rows named** (`tools/capability_bar.py`,
   `substrate/immortal_class_gate.py`, `tools/conclusion_protocol.py`). Every sibling gate lives in `substrate/`, and
   the p4c ratchet counts `tools/` files as ungoverned against a baseline of 30 that may only fall. The plan, tasks
   and the ratified-rows file were amended to name what was built.
2. **L5's gate is `immortal_retention`, not `immortal_class_gate`** — §1.2's find. `lake_engine.RETENTION` and
   `lake_migrate.TAG_DISPOSITIONS` already declare the classes, so the gate READS both and cross-checks them rather
   than declaring a second mechanism (P2).
3. **A pre-registered expectation was FALSIFIED, and three EXEMPT rows were added to `dormancy_gate`.** §1.1 had
   stated that a `run_selfchecks.py` registration is a live root of `dormancy_gate` and would carry these three
   through P3. It is not: `_imports()` walks `Import`/`ImportFrom` nodes and the registration is a string in a list,
   so the gate reported all three DORMANT on its first run. All three are CI/run_selfchecks instruments, never tick
   faculties, so they take the `policy_manifest` disposition the tree already uses for `lake_rebuild_gate`,
   `replay_gate` and `determinism_gate`. Each row states what would retire it, so the stale-exemption arm can bite.
4. **The gates' first counted output was itself wrong in two places, and the panel is what found it.** §4 records
   the fold. Both defects were in this arc's own instruments, not in the tree it audits.

### 3.5 Gate state at the close of the build

`capability_bar`, `immortal_retention`, `conclusion_protocol`, `dormancy_gate`, `method_dormancy_gate`,
`policy_manifest` and `tools/p4c_ratchet.py` all `--assert` rc=0. `tests/test_ratified_row_gates.py` is 7 passed.
The three rows now bind: each has a gate, and each gate's output has been counted on this tree.

---

## 4. The adversarial panel, and what it changed (2026-09-06)

Three lenses (mechanism, policy, instrument discipline) read commit `3ccbf6e` independently, and every finding
they raised was handed to a separate agent whose instruction was to REFUTE it. Twelve findings were verified;
**seven survived refutation and five were refuted**, which is the number worth recording — a panel that
confirms everything is not measuring anything.

### 4.1 What survived, and what it changed

| # | Severity | Where | What it changed |
|---|---|---|---|
| 1 | BLOCKER (found twice, mechanism + policy) | `immortal_retention._RECLAIM_CALLS` | The census could not see rewrite-shaped reclamation, which is the shape this repo uses. `rewrite` added; the real reclaimer surfaced; `KNOWN_UNLABELED` created to carry it in the open; a rewrite-shaped red added. |
| 2 | MAJOR (found three times, all lenses) | `immortal_retention.round_trip` | Crossed no process boundary, touched no lake code, had no reachable red. Rebuilt on `binlog.py`'s cross-process idiom with a `withhold` red run before the green. |
| 3 | MINOR | `conclusion_protocol.surface_findings` | Refusal 3 was opt-in through `surface_symbols`; it now arms itself from the plan's own task-completion marks. |
| 4 | MINOR | §1.1 of this record | The `run_selfchecks`-is-a-live-root claim was false. Corrected, and marked as a falsified pre-registration rather than only recording the exemptions. |

### 4.2 What was refuted, and why that matters

Five findings did not survive. Three claimed `capability_bar`'s surface arm is dead because the registry is
empty; the verifiers showed the vacuity is the arm's pre-registered scope (a comparison word is only a defect
*for a registered task*), that the arm's red fires in both the self-check and the test file, and that all five
tracked surfaces contain zero comparison words today. One claimed the aggregate refusal should key on
`surface_symbols` like the surface scan; the verifier measured that the proposed change would make the check
*provably* zero (only one row declares symbols) and would lose four realistic aggregation shapes the current
scan catches — of 72,813 AST nodes scanned, 64 already carry exactly one instrument name, one token away from
firing. One claimed the `_RECORD_REFS` filter suppressed 43 real reclaimers; the verifier inspected all 43 and
found none destroys a durable lake record. Recorded because a refuted finding is evidence about the
instrument too.

### 4.3 The honest caveat about this section

**The fold is itself unreviewed code.** These four changes were written in response to the panel and have not
been through a panel of their own. They were verified by: each new arm's constructed red run before its green,
the module self-checks, the 10 brain-free tests, and the arc's battery. That is battery-verified, not
panel-reviewed, and the distinction is stated here rather than left for a reader to assume.
