# Phase 1 Data Model: Policy Compliance (P0-P8)

This feature is a remediation, so its "data model" is the small set of records the two new
instruments emit and the audit consumes. No brain-state schema changes. **No new durable store** —
nothing here is written to `brain_lake.bin` (P4b).

## Entity: Finding

A located policy violation. Produced by the audit, consumed by `tasks.md`.

| Field | Type | Rule |
|---|---|---|
| `policy` | enum `P0..P8` | must name a policy that exists in POLICIES.md |
| `severity` | enum `critical` / `major` / `minor` | critical = reaches a human or breaks arrival |
| `file` | path | repo-relative, must exist |
| `line` | int? | optional; **content is authoritative, not the number** — line numbers drift |
| `evidence` | text | the quoted line(s) plus how it was confirmed |
| `violation` | text | the clause it breaks, quoted from POLICIES.md |
| `fix` | text | minimal remedy; may legitimately be "correct the claim, not the code" |
| `verdict` | Verdict | **required before the finding may drive a change** (FR-011) |

**State transitions**: `proposed → verified → {applied, refuted, deferred}`.
A `refuted` finding is retained with its correction, never deleted — the refutation is the result.
A `deferred` finding must carry a named blocker (FR-012).

## Entity: Verdict

The adversarial re-check of a Finding. Its default is refutation.

| Field | Type | Rule |
|---|---|---|
| `still_true` | bool | did the cited content actually verify? |
| `real_violation` | bool | or does an EXEMPT entry / documented exception already cover it? |
| `fix_is_safe` | bool | would the fix break a gate, the envelope, determinism, or force-wire? |
| `correction` | text | empty only if the finding held exactly |

**Rule**: a Finding drives a change only when `still_true ∧ real_violation ∧ fix_is_safe`. Any
false blocks the change and the Finding is re-scoped or deferred.

## Entity: DependencyLedgerEntry

Emitted by `tools/dependency_ledger.py`. One row per package, in either direction.

| Field | Type | Rule |
|---|---|---|
| `package` | str | distribution name, normalized (`scikit-learn`, not `sklearn`) |
| `declared_in` | enum `core` / extra name / `NONE` | from `pyproject.toml` |
| `imported_at` | list[file:line] | empty means declared-and-unused |
| `guarded` | bool | is the import inside a try/except or lazy? unguarded + undeclared is worse |
| `status` | enum | `OK` / `UNDECLARED` / `UNUSED` / `EXEMPT` |
| `reason` | text | **required** when `EXEMPT` |

**Validation rules**:
- `UNDECLARED` ⟺ `imported_at` non-empty ∧ `declared_in == NONE`. Severity rises when `guarded` is
  false, because an unguarded top-level import breaks a clean install outright.
- `UNUSED` ⟺ `declared_in != NONE` ∧ `imported_at` empty ∧ no reason.
- Import name → distribution name mapping is required and is the instrument's main correctness risk
  (`sklearn`→`scikit-learn`, `PIL`→`pillow`, `cv2`→`opencv-python`). A wrong mapping produces a
  **false UNDECLARED**, so the mapping is explicit and tested, never inferred.
- Standard library and first-party (`snn_data_lake`) imports are excluded. `sys.stdlib_module_names`
  is the authority for the former — hand-maintained stdlib lists rot.

## Entity: Criterion

Emitted by `tools/red_reachability.py`. One row per registered success criterion, marker, probe
verdict or scored gate.

| Field | Type | Rule |
|---|---|---|
| `name` | str | `module.function` or `module.Class.method` — **class-granular**, per the lesson that same-named defs in one module overwrite each other and report against the wrong class |
| `emits` | enum `bool` / `score` / `verdict` | what it returns |
| `reads_system` | bool | **does it take the system under test as input?** A criterion with no parameters cannot be falsified about anything |
| `red_input` | text? | the recorded input under which it reports FAIL |
| `red_demonstrated` | bool | has that input actually been run and observed to fail? |
| `status` | enum | `RED_REACHABLE` / `NO_RED` / `UNKNOWN` / `EXEMPT` |
| `consumers` | list[file:line] | who reads the verdict — **drives severity, per P8's load-bearing rule** |

**Validation rules**:
- `NO_RED` requires deletion, not re-thresholding (F11, Standing Invariant 2).
- **`UNKNOWN` is a first-class outcome, not a failure.** Standing Invariant 3: where the instrument
  cannot distinguish "has no red" from "we could not construct one", it reports UNKNOWN. Auto-marking
  UNKNOWN as `NO_RED` would make this gate commit the sin it exists to catch.
- `reads_system == false` is sufficient on its own for `NO_RED`: a criterion invariant to the entire
  system under test is unfalsifiable at every level of the computational hierarchy (research R1).
- Severity is a function of `consumers`, not of the criterion's ambition — a criterion whose verdict
  reaches a human ranks above one read by nothing.

## Entity: Claim

A statement in a document, docstring or comment asserting a capability, policy status, or measured
number. Consumed by Phase C.

| Field | Type | Rule |
|---|---|---|
| `text` | str | the asserted statement |
| `site` | file:line | where it is asserted |
| `source` | file:line? | where the backing measurement or policy lives |
| `status` | enum | `AGREES` / `STALE` / `UNBACKED` / `SUPERSEDED` |

**Rule**: a `STALE` claim is corrected in place. A `SUPERSEDED` number keeps the old value **and**
records the new one — the project's convention is that a document describing a tree that existed is
correct history, and erasing it loses the record of what changed.

## Non-entities (recorded so scope stays bounded)

- **No brain-state schema change.** No new `_T_*` record tag, no lake migration.
- **No new telemetry key.** Phase C adds PROVENANCE *entries* for existing keys; it does not
  introduce signals.
- **No safety-envelope model.** Those five files are report-only.
