# Phase 0 Research: Policy Manifest + Release Audit

Every decision below rests on a measurement taken on this tree at HEAD `f61e4a94` (2026-08-18),
not on a reading of documentation. Where a measurement refuted the obvious design, that is stated.

## R1 — Derivation source: the enforcement table + policy headings, two independent parses

**Decision**: the manifest derives from `docs/POLICIES.md` alone, using independent parses that
must reconcile (FR-003): (a) policy HEADINGS — `^## P\d` parents plus the GENERAL sub-clause
pattern `^### P\d[a-z]*( amendment)?` (anchored alternation; an enumerated `P4b|P4c` list is
itself the hand-maintained list FR-002 forbids, and its unanchored form had a precedence bug
matching 16 lines) — give the policy set; (b) the enforcement TABLE, located by STRUCTURAL
anchor (the first head-matter markdown table, header containing `mechanized in CI`), never by
line numbers, gives per-policy enforcement kind and gate commands; (c) the GATE-OUTPUT table
(the second head-matter table, header `| gate | output | in CI? |`) holds executed argv rows
and is the measured source for commands the enforcement row does not spell out.

**Rationale**: FR-002 forbids a hand-maintained parallel copy; the audit section of the spec (P2)
requires deriving from or superseding the existing table. Deriving keeps POLICIES.md the single
hand-edited authority and makes the manifest a parse — drift between the two parses is a loud
failure, not a silent skew. Superseding (deleting the table for a machine block) was rejected: it
edits policy-document structure before the instrument has counted anything, the wrong order under
FR-012, and risks FR-013's no-content-change rule.

**Measured (corrected 2026-08-18 by /speckit-analyze)**: 14 policy headings under the general
pattern — 9 parents + 5 sub-clauses (`P3 amendment`, `P4b`, `P4c`, `P4c amendment`, `P5a`; the
first draft's enumerated list counted 12 and silently dropped two). 10 table rows (P0–P8 +
e-prop; the two hygiene rows self-declare *"not a policy"* and are excluded by their own label).
Sub-clauses have no table rows. SECOND correction: the first draft claimed P4c's section body
names `tools/p4c_ratchet.py` — FALSE; measured, the single mention in the whole document lives
in the gate-output table (source (c) above), which is why (c) is a derivation source at all.

**Reconciliation rule**: sub-clause headings roll up under their parent `## P4` row. `e-prop` is
TABLE-ONLY — it has a table row and NO heading anywhere in POLICIES.md (measured: 0 e-prop
headings; the invariant lives under `## Cross-cutting invariants`). The identity is therefore
`parent_headings + len(TABLE_ONLY_IDS) == table_policy_rows` with `TABLE_ONLY_IDS = {"e-prop"}`
pinned in the module beside a comment naming why; measured at HEAD: 9 + 1 == 10. Any mismatch is
a derivation failure (FR-003/FR-007). A future table-only invariant EXTENDS the set — it is a row
kind, not a special case. (The first draft of this rule wrote the naive headings==rows form,
which is false on the real tree; /speckit-analyze caught it pre-build.)

**Alternatives considered**: parse only section bodies (rejected — the table already encodes
enforcement kind, and re-deriving kind from prose is guesswork); a YAML sidecar (rejected —
that IS the hand-maintained parallel copy FR-002 forbids).

## R2 — Liveness cross-check: POLICIES.md is not enough, and that is a finding, not a flaw

**Measured (re-counted after /speckit-analyze)**: the unclaimed census at HEAD is THREE, not one —
`safety_envelope`, `doc_budget_gate` and `ingest_guard` each have 0 POLICIES.md mentions while
registered in `run_selfchecks.py`. A POLICIES-only manifest is structurally blind to all three.
Two obligations follow: (a) `run_selfchecks.py` registers ~50 substrate modules, most of them not
gates, so the census REQUIRES a defined gate predicate — stated in the module, tested, never
implicit; (b) the instrument's own `--assert` step (added in US2) makes `policy_manifest` a
member of its own discovery surface, so the module carries a NAMED self-disposition, printed in
every audit report — a silently self-exempting auditor is itself the finding class.

**Decision**: the audit cross-checks the derived command set against two runnable surfaces —
`.github/workflows/ci.yml` (`--assert` steps) and `run_selfchecks.py` registrations — and reports
two distinct drift kinds: `NOT_IN_CI` (a policy names a command no CI step runs) and
`UNCLAIMED_GATE` (a runnable gate no policy claims — `safety_envelope` is the constructed-red that
already exists on this tree, so the first audit run has a REACHABLE RED for this verdict without
building a fixture).

**Rationale**: SI 3 (confident negative needs a demonstrated positive) — a manifest that only reads
the governing document certifies the document's self-image. The cross-check is what makes the audit
about the TREE.

## R3 — Module home: `substrate/`, artifact home: `tools/*.json`

**Decision**: `src/snn_data_lake/substrate/policy_manifest.py`; certification record at
`tools/policy_certification.json`.

**Rationale**: measured in specs/020 (plan.md, Item 2): a new `.py` under `tools/` reddens the p4c
ratchet on its own commit (corpus is `*.py`, zone pinned 30/30), while a `.json` there costs the
ratchet nothing — the working precedent is `tools/p4c_ungoverned_baseline.json`. Same forces, same
answer as `gradient_claim_gate`. Registration: `__init__.py` REGIONS + `run_selfchecks.py` + a CI
lint step `python -m snn_data_lake.substrate.policy_manifest --assert` (the P3 live path).

## R4 — What the default audit RUNS: per-policy gate commands, never the full suite

**Decision**: `--audit` executes each manifest row's command in a subprocess (each is a
seconds-scale `--assert`); the three full gate commands (`run_selfchecks.py`, `pytest`, `ruff`) are
NOT run by default — `--certify --full` runs them for the release lane.

**Rationale**: the spec's own edge case — a ~30-minute default audit will be skipped, and a skipped
audit's certification means nothing. Measured: the 5 current `--assert` gates complete in seconds
each. Subprocess execution (not import) keeps one gate's state from leaking into another —
specs/020's FR-013b lesson.

## R5 — Enforcement kinds: `absolute` / `ratchet` / `review-only`, derived not guessed

**Decision**: kind derives from the table row's own words: a mechanized row whose text says
RATCHET (or whose section body does — P4c, P6) is `ratchet`; a mechanized non-ratchet row is
`absolute`; a `—` in the mechanized column with "yes" under review is `review-only`; e-prop's
"PARTIALLY" splits honestly into its two gate rows plus a `review-only` remainder row rather than
being averaged into one kind.

**Rationale**: FR-008 — `review-only` is an honest kind, not a synonym for green. A ratchet asserts
"no worse", never "satisfied" (POLICIES.md's own words); collapsing it into `absolute` would let a
green ratchet read as a satisfied policy, the exact overclaim this repo keeps correcting.

**Sub-clause kinds (added 2026-08-18)**: sub-clauses are first-class OUTPUT rows with their OWN
kind, defaulting to `review-only` when nothing names their gate — NEVER inheriting the parent's
command. P4b (the durable-lake clause, which nothing mechanizes) must render NOT MACHINE-DECIDED;
inheriting P4's `replication_guard` would be the exact overclaim FR-008 exists to prevent. P4c's
`ratchet` kind and `tools/p4c_ratchet.py` command come from the gate-output table, not from
inheritance. Sub-clause rows are EXCLUDED from the C1 reconciliation count (which stays
`parents + table_only == table_rows`).

## R6 — US4 companion file: `docs/POLICIES_EVIDENCE.md`, budgeted from its first commit

**Decision**: relocated forensics land in `docs/POLICIES_EVIDENCE.md`; the SAME commit adds it to
`doc_budget_gate.BUDGETS` with a ceiling counted from its initial content, and LOWERS
`docs/POLICIES.md`'s ceiling to its new measured size (headroom denial — spec SC-006 note).

**Rationale**: an unbudgeted companion file is where the prose debt silently reflates;
`doc_budget_gate.py` already prints the instruction for exactly this case ("Update BUDGETS in the
SAME change"). Ordering: US4 runs LAST, after US1's counts are recorded (FR-012).

## R7 — Templates + agent dispatch (US3): pointer + one runnable line, no restatement

**Decision**: `plan-template.md`'s Constitution Check section gains the runnable check
(`python -m snn_data_lake.substrate.policy_manifest --audit`) and a pointer to `docs/POLICIES.md`;
`spec-template.md` and `tasks-template.md` each gain a one-line pointer. Agent dispatch (FR-010)
delivers `--manifest` OUTPUT (regenerated at dispatch time), never a stored copy.

**Rationale**: measured at HEAD — 0 POLICIES.md references in plan/spec/tasks templates. The Phase 8
drift-class rule binds: subordinate documents POINT, they never restate; a template that embedded
policy text would be the fourth copy of the claims this project just spent nine review rounds
un-duplicating. A stored manifest copy is likewise the copy that drifts — regeneration is the fix
(FR-002's logic applied to dispatch).
