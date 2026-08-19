# Quickstart Validation: Policy Manifest + Release Audit

Prerequisites: `.venv-lake` (the ONE gate env), repo root. Contracts: [contracts/cli.md](contracts/cli.md);
entities: [data-model.md](data-model.md).

## 1. The manifest derives and reconciles

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.policy_manifest --manifest
```

Expect: one row per policy (P0–P8 + e-prop, sub-clauses rolled up under P4; e-prop is TABLE-ONLY,
no heading of its own), each with enforcement kind and command-or-NONE; exit 0. The identity
parents + table_only == table rows (9 + 1 == 10 at HEAD) — check by eye once; the tool asserts
it every run.

## 2. The first audit is honestly non-green (SC-002)

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.policy_manifest --audit
```

Expect: every review-only policy listed as `REVIEW_ONLY`/`NO_GATE` (the derived set — P0, P1, P2,
P5 at plan time; the tool's output is authoritative, not this sentence), mechanized policies
PASS/FAIL from their real gates, and the UNCLAIMED_GATE census non-empty (three members measured
at analysis time: `safety_envelope`, `doc_budget_gate`, `ingest_guard`) unless their rows have
since landed. An all-green first run = the audit is blind → stop and fix the instrument.

## 3. Certification refuses on today's tree, and the refusal names why (SC-004, FR-004)

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.policy_manifest --certify
```

Expect: nonzero exit; refusal names every unclaimed census member (three at analysis time, until
their table rows land post-count) and any true NO_GATE policy; REVIEW_ONLY rows are disclosed, never
blocking. `tools/policy_certification.json` is written with `certified: false` and the full
command log (FR-006). In CI this runs as a non-blocking job on every push to main.

## 4. Constructed reds all fire (SI 5)

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.policy_manifest --self-check
```

Expect: FAIL, CANNOT_RUN, NO_GATE, empty-walk, and reconciliation-mismatch fixtures each produce
their verdict; exit 0 only when every red is reachable.

## 5. The three project gates stay green (FR-013: nothing relaxed)

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py
```

```bash
.venv-lake/Scripts/python.exe -m pytest -q
```

(The gate is the WHOLE suite; `-m pytest tests/test_policy_manifest.py -q` is the fast iteration
loop, not the gate.)

```bash
.venv-lake/Scripts/python.exe -m ruff check .
```

Also: `git diff --stat` shows zero `brainstem/` lines at every step of this feature.

## 6. US3: a generated plan carries the runnable check

Generate any new feature from the templates; the plan's Constitution Check section must contain the
`--audit` invocation as a command, not a description of one.

## 7. US4 (LAST, gated on step 2's recorded counts — FR-012)

After the consolidation commit:

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.doc_budget_gate
```

Expect: `docs/POLICIES.md` measurably below 28,559 words (SC-006 — at least 1,191 recovered,
counted by the budget gate's own counter) with its CEILING lowered in the same commit, and
`docs/POLICIES_EVIDENCE.md` present in the budget table with its own ceiling. Every relocated
worked case reachable from a pointer at its old site.
