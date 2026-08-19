# Phase 1 Data Model: Policy Manifest + Release Audit

All entities are in-memory dataclasses inside `substrate/policy_manifest.py`; only Certification is
persisted (JSON). The manifest is generated AND committed (`tools/policy_manifest.json`, the
spec's recorded assumption: an uncommitted artifact cannot be reviewed) — but derivation stays
the source of truth: `--assert` fails when the committed copy differs from a fresh derivation
(the p4c-baseline sync pattern), and the ONLY edit path is regeneration (FR-002).

## Policy

One governing rule, derived from a `docs/POLICIES.md` heading + its enforcement-table row.

| field | type | source | validation |
|---|---|---|---|
| `id` | str (`"P0"`..`"P8"`, `"e-prop"`) | heading parse; `e-prop` is TABLE-ONLY (enforcement-table row, no heading — R1) | identity: parents + table-only == table rows (9 + 1 == 10 at HEAD) |
| `title` | str | heading text | non-empty |
| `subclauses` | list[str] — discovered by `^### P\d[a-z]*( amendment)?`, never enumerated (5 at HEAD incl. `P3 amendment`, `P5a`) | `###` headings | first-class OUTPUT rows with their OWN kind: default `review-only` unless their own text or the gate-output table names a gate, NEVER inheriting the parent's command (P4b renders NOT MACHINE-DECIDED); excluded from the reconciliation identity, which stays parents + table-only == table rows |
| `enforcement_kind` | `absolute` \| `ratchet` \| `review-only` | table row wording (R5) | e-prop may carry per-command kinds |
| `commands` | list[GateCommand] | backticked tokens in table row + sub-clause bodies | may be empty (that is `NO_GATE`, never an error) |

## GateCommand

| field | type | notes |
|---|---|---|
| `argv` | list[str] | e.g. `["python", "tools/p6_gate.py", "--assert"]`; always subprocess, never import (R4) |
| `in_ci` | bool | cross-check vs `.github/workflows/ci.yml` (R2) |
| `in_selfchecks` | bool | cross-check vs `run_selfchecks.py` registrations (R2) |

## ManifestRow

One Policy flattened for output — what `--manifest` prints (text table + `--json`). Derived, never
edited; regenerating IS the update path.

## AuditResult

Per policy, after `--audit` executes each row's commands.

| verdict | meaning | exit-code contribution |
|---|---|---|
| `PASS` | every command ran and exited 0 | green |
| `FAIL` | a command ran and exited non-zero | red |
| `CANNOT_RUN` | command missing/unlaunchable — DISTINCT from FAIL (FR-005) | red |
| `NO_GATE` | an `absolute`/`ratchet` policy with no command (a `review-only` policy with no command is REVIEW_ONLY, never NO_GATE — FR-004's predicate) | red for certification, honest in report |
| `REVIEW_ONLY` | no machine decides this policy — neither green nor violation (FR-008) | neutral, listed |

Plus two tree-level drift findings (R2), reported alongside per-policy rows:
`NOT_IN_CI` (a claimed command no CI step runs) and `UNCLAIMED_GATE` (a runnable gate no policy
claims — live instances at HEAD `f61e4a94`, measured: `safety_envelope`, `doc_budget_gate`,
`ingest_guard`; the census is derived from a stated gate predicate, and the instrument's own
gate carries a named, printed self-disposition).

Derivation failures are their own loud category, not verdicts: empty/truncated policy walk,
heading-vs-table reconciliation mismatch (FR-003, FR-007) → nonzero exit before any row prints.

## Certification

The persisted output of `--certify` (FR-006): `tools/policy_certification.json`.

| field | type | notes |
|---|---|---|
| `tree` | str | `git rev-parse HEAD` + dirty flag; a certification of a dirty tree SAYS so |
| `results` | list[AuditResult] | complete, including REVIEW_ONLY rows |
| `commands_run` | list[argv + exit code + duration] | what was run, so certification is re-derivable, not trusted |
| `certified` | bool | true ONLY when no FAIL / CANNOT_RUN / NO_GATE, zero UNCLAIMED_GATE drift, and derivation reconciled — REVIEW_ONLY rows never block (Clarifications 2026-08-18) |
| `review_only` | list[str] | policies no machine decides, disclosed first-class in the certificate |
| `full` | bool | whether `--full` also ran the three suite-level gates (R4) |

State transitions: none persisted. A new `--certify` OVERWRITES the JSON — history lives in git,
not in the artifact (one durable store per concern; the lake rule's shape applied to tooling).
