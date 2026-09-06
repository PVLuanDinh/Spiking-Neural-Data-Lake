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
  match. Reachability for P3 comes the way every gate's does: a `run_selfchecks.py` registration, which is a live
  root of `dormancy_gate`.

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
